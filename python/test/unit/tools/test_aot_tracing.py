import glob
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import pytest
import torch
from dataclasses import dataclass

import triton
from triton.tools.aot.tracing import MatMulConfig, MatMulKernelTracer, TraceConfig

FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"

# ------------------------------------------------------------------------------------------------------------ #
"""Configs for matmul kernels 

"""
MATMUL_ARGS = [
    "C",
    "A",
    "B",
    "M",
    "N",
    "K",
    "stride_cm",
    "stride_cn",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
]

MATMUL_CONSTANTS = ["BLOCK_M", "BLOCK_N", "BLOCK_K"]

DEFAULT_MATMUL_DTYPES = OrderedDict(
    {
        "C": "*fp32",
        "A": "*fp16",
        "B": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
    }
)
DEFAULT_MATMUL_HINTS = OrderedDict(
    {
        "C": 16,
        "A": 16,
        "B": 16,
        "M": None,
        "N": None,
        "K": None,
        "stride_cm": None,
        "stride_cn": 1,
        "stride_am": None,
        "stride_ak": 1,
        "stride_bk": 16,
        "stride_bn": 1,
    }
)

DEFAULT_MATMUL_CONSTANTS = OrderedDict({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16})

# Specialization Configs

NO_HINTS = {k: None for k in MATMUL_ARGS}
STRIDE_CM_HINTS = {
    k: (v if k != "stride_cm" else 16) for k, v in DEFAULT_MATMUL_HINTS.items()
}
STRIDE_AM_HINTS = {
    k: (v if k != "stride_am" else 16) for k, v in DEFAULT_MATMUL_HINTS.items()
}
STRIDE_CM_AM_HINTS = {
    k: (v if k != "stride_cm" and k != "stride_am" else 16)
    for k, v in DEFAULT_MATMUL_HINTS.items()
}

# Signatures
DEFAULT_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32:1, i32, i32:1, i32:16, i32:1, 16, 16, 16"
STRIDE_CM_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32:16, i32:1, i32, i32:1, i32:16, i32:1, 16, 16, 16"
STRIDE_AM_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32:1, i32:16, i32:1, i32:16, i32:1, 16, 16, 16"
STRIDE_CM_AM_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32:16, i32:1, i32:16, i32:1, i32:16, i32:1, 16, 16, 16"
NO_HINT_SIGNATURE = (
    "*fp32, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 16, 16, 16"
)

# ------------------------------------------------------------------------------------------------------------ #


"""
Utilities for generating reference AOT kernels 
"""


# Copied from test/unittest/tools/test_aot.py
class AOTScriptRunner:
    """Wrapper around `triton.tools` for AOT compilation

    Runs `triton.tools.compile` and `triton.tools.link` in subprocesses
    """

    @staticmethod
    def compile_kernel(
        *, dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path
    ):
        compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

        subprocess.run(
            [
                sys.executable,
                compiler_path,
                "-n",
                kernel_name,
                "--signature",
                signature,
                "--out-name",
                out_name,
                "-o",
                out_path,
                "-w",
                str(num_warps),
                "-g",
                grid,
                kernel_path,
            ],
            check=True,
            cwd=dir,
        )

    # @staticmethod
    # def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
    #     # compile all desired configs
    #     for ha in ha_hb_hints:
    #         for hb in ha_hb_hints:
    #             sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
    #             name = f"matmul_{dtype}"
    #             grid = f"M/{BM}, N/{BN}, 1"
    #             AOTScriptRunner._compile_kernel(
    #                 dir=dir,
    #                 signature=sig,
    #                 kernel_name="kernel",
    #                 out_name=name,
    #                 out_path=name,
    #                 num_warps=1,
    #                 grid=grid,
    #                 kernel_path=kernel_path,
    #             )

    @staticmethod
    def link_aot_kernels(dir, kernel_name):
        linker_path = os.path.join(triton.tools.__path__[0], "link.py")

        # link all desired configs
        h_files = glob.glob(os.path.join(dir, "*.h"))
        subprocess.run(
            [sys.executable, linker_path] + h_files + ["-o", kernel_name],
            check=True,
            cwd=dir,
        )


# Generates AOT reference kernels for matmul


def generate_signature(
    dtypes: OrderedDict,
    hints: OrderedDict,
    constant_vals: OrderedDict,
):
    assert set(dtypes.keys()) == set(MATMUL_ARGS)
    assert set(hints.keys()) == set(MATMUL_ARGS)

    args = []
    for arg in MATMUL_ARGS:
        dtype = dtypes[arg]
        hint = hints[arg]
        if hint:
            args.append(f"{dtype}:{str(hint)}")
        else:
            args.append(f"{dtype}")

    args_str = ", ".join(args)
    consts = []
    for const in MATMUL_CONSTANTS:
        consts.append(f"{constant_vals[const]}")
    consts_str = ", ".join(consts)
    signature = ", ".join([args_str, consts_str])
    return signature


def generate_matmul_reference(
    kernel_path, out_dir, BM=16, BN=16, BK=16, hints=["", ":16"], dtype="fp16"
):
    AOTScriptRunner.compile_aot_kernels(out_dir, kernel_path, dtype, BM, BN, BK, hints)
    AOTScriptRunner.link_aot_kernels(out_dir, "matmul")


def compile_matmul_kernels(
    signatures,
    num_warps,
    grids,
    out_dir=None,
    kernel_path=FIXTURES_DIR / "kernels" / "matmul_kernel.py",
):
    if isinstance(signatures, str):
        signatures = [signatures]
    if isinstance(num_warps, int):
        num_warps = [num_warps] * len(signatures)
    if isinstance(grids, str):
        grids = [grids] * len(signatures)
    assert len(signatures) == len(num_warps) == len(grids)

    kernel_name = "matmul"

    out_dir = out_dir or FIXTURES_DIR / "aot_reference_kernels"

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    for s, w, g in zip(signatures, num_warps, grids):
        AOTScriptRunner.compile_kernel(
            dir=out_dir,
            signature=s,
            kernel_name=kernel_name,
            out_name=kernel_name,
            out_path=kernel_name,
            num_warps=w,
            grid=g,
            kernel_path=kernel_path,
        )
        AOTScriptRunner.link_aot_kernels(out_dir, kernel_name)


@pytest.mark.parametrize(
    "dtypes, hints, constants, expected_signature",
    [
        (
            DEFAULT_MATMUL_DTYPES,
            DEFAULT_MATMUL_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            DEFAULT_SIGNATURE,
        ),
        (DEFAULT_MATMUL_DTYPES, NO_HINTS, DEFAULT_MATMUL_CONSTANTS, NO_HINT_SIGNATURE),
        (
            DEFAULT_MATMUL_DTYPES,
            STRIDE_CM_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            STRIDE_CM_SIGNATURE,
        ),
        (
            DEFAULT_MATMUL_DTYPES,
            STRIDE_AM_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            STRIDE_AM_SIGNATURE,
        ),
        (
            DEFAULT_MATMUL_DTYPES,
            STRIDE_CM_AM_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            STRIDE_CM_AM_SIGNATURE,
        ),
    ],
    ids=["default", "no_hints", "stride_cm", "stride_am", "stride_cm_am"],
)
def test_default_signature(dtypes, hints, constants, expected_signature):
    signature = generate_signature(
        dtypes=dtypes,
        hints=hints,
        constant_vals=constants,
    )

    assert (
        signature == expected_signature
    ), f"Expected signature: {expected_signature}, Actual signature: {signature}"


"""
Replicate the compiled kernel headers and sources for matmul kernels in `test_aot.py`
    
4 cases
    - no specialization for stride_cm, stride_am
    - no specialization for stride_cm only
    - no specialization for stride_am only
    - implicit specializations for both by passing appropriately shaped tensors (divisible by 16)

"""


def trace_matmul_kernels():
    trace_dir = Path("traced_kernels").absolute()
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    specializations = [("stride_cm", "stride_am"), ("stride_cm",), ("stride_am",)]
    trace_configs = []
    for spec in specializations:
        trace_configs.append(TraceConfig(do_not_specialize=spec, trace_dir=trace_dir))

    trace_configs.append(TraceConfig())

    # Use default MatmulConfig (16 x 16 x 16), dtype_in = fp16, dtype_out = fp32
    kernel_dir = (
        Path(triton.__path__[0]).parent.absolute()
        / "test"
        / "unit"
        / "tools"
        / "fixtures"
        / "kernels"
    )
    matmul_config = MatMulConfig()
    matmul_tracer = MatMulKernelTracer(kernel_dir)
    kernel_configs = [matmul_config] * len(trace_configs)

    traces, outputs, checks = matmul_tracer.trace(
        kernel_configs=kernel_configs, trace_configs=trace_configs
    )

    for actual, expected in zip(outputs, checks):
        is_close = torch.allclose(actual, expected, atol=1e-1, rtol=0)
        print(f"Is close {is_close}")


DEFAULT_MATMUL_CONFIG = MatMulConfig(
    dtype_in=torch.float16,
    dtype_out=torch.float32,
    M=16,
    N=16,
    K=16,
    BLOCK_M=16,
    BLOCK_N=16,
    BLOCK_K=16,
    seed=0,
)
NO_HINT_TRACE_CONFIG = TraceConfig(do_not_specialize=None, num_warps=1)


def test_kernel_compilation():
    out_dir = FIXTURES_DIR / "aot_reference_kernels"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compile_matmul_kernels(NO_HINT_SIGNATURE, 1, "M/16, N/16, 1", out_dir=out_dir)
    kernel_headers = list(out_dir.glob("*.h"))
    kernel_sources = list(out_dir.glob("*.c"))
    print(kernel_headers)
    print(kernel_sources)
    assert len(kernel_headers) == 1
    assert len(kernel_sources) == 1


@dataclass
class MatmulTestConfig:
    dtypes: OrderedDict
    hints: OrderedDict
    constants: OrderedDict
    num_warps: int
    grid: str


TEST_CONFIGS = {
    "no_hints": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=NO_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    )
}


def _tt_to_torch(tt):
    if "16" in tt:
        return torch.float16
    elif "32" in tt:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype {tt}")


@pytest.mark.parametrize("config_name", [("no_hints")], ids=lambda x: x.upper())
def test_single_trace(
    config_name,
):
    from triton.tools.aot.compiler import AOT_C_CUDA_Compiler

    # Set up directories for reference and traced kernel artifacts
    test_dir = (Path(__file__).parent / "test").absolute()
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    reference_aot_dir = test_dir / "reference_aot_kernels"
    reference_aot_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = test_dir / "traced_kernels"
    trace_dir.mkdir(parents=True, exist_ok=True)

    test_config = TEST_CONFIGS[config_name]
    signature = generate_signature(
        test_config.dtypes, test_config.hints, test_config.constants
    )

    kernel_path = FIXTURES_DIR / "kernels" / "matmul_kernel.py"

    compile_matmul_kernels(
        signature,
        num_warps=test_config.num_warps,
        grids=test_config.grid,
        out_dir=reference_aot_dir,
        kernel_path=kernel_path,
    )
    reference_headers = list(reference_aot_dir.glob("*.h"))
    reference_sources = list(reference_aot_dir.glob("*.c"))

    # Construct MatMulConfig and TraceConfig
    dtype_in = _tt_to_torch(test_config.dtypes["A"])
    dtype_out = _tt_to_torch(test_config.dtypes["C"])

    # Assume that M, N, K are divisible by 16; defaults to 16
    matmul_config = MatMulConfig(
        dtype_in=dtype_in,
        dtype_out=dtype_out,
        BLOCK_M=test_config.constants["BLOCK_M"],
        BLOCK_N=test_config.constants["BLOCK_N"],
        BLOCK_K=test_config.constants["BLOCK_K"],
    )
    assert matmul_config.M % matmul_config.BLOCK_M == 0
    assert matmul_config.N % matmul_config.BLOCK_N == 0
    assert matmul_config.K % matmul_config.BLOCK_K == 0

    do_not_specialize = [k for k in test_config.hints if test_config.hints[k] is None]
    trace_config = TraceConfig(
        do_not_specialize=test_config.hints,
        num_warps=test_config.num_warps,
        trace_dir=trace_dir,
    )

    # Use default MatmulConfig (16 x 16 x 16), dtype_in = fp16, dtype_out = fp32
    matmul_tracer = MatMulKernelTracer(kernel_path.parent)
    traces, outputs, checks = matmul_tracer.trace(
        kernel_configs=[matmul_config], trace_configs=[trace_config]
    )
    trace = traces[0]
    import json

    with open(trace_dir / f"{trace.kernel_name}-jit_args.json", "w") as fp:
        json.dump({k: str(v) for k, v in trace.jit_args.items()}, fp, indent=2)
    # trace: TraceArtifact = traces[0]
    # for actual, expected in zip(outputs, checks):
    #     is_close = torch.allclose(actual, expected, atol=1e-1, rtol=0)
    #     print(f"Is close {is_close}")

    compiler = AOT_C_CUDA_Compiler(
        kernel_name=trace.kernel_name,
        compiled_binary=trace.compiled_binary,
        jit_args=trace.jit_args,
        jit_fn=trace.jit_fn,
    )

    with open(trace_dir / f"{trace.kernel_name}-compiled.h", "w") as fp:
        fp.write(compiler.generate_header())
    with open(trace_dir / f"{trace.kernel_name}-compiled.c", "w") as fp:
        fp.write(compiler.generate_source())

    # Check that the generated code is the same as the reference code
    # reference_header = (
    #     trace_dir
    #     / trace.kernel_name
    #     / "matmul.9abb00f7_0d1d2d3de4de5de6de7c8de9c10de11c.h"
    # ).read_text()
    # reference_source = (
    #     trace_dir
    #     / trace.kernel_name
    #     / "matmul.9abb00f7_0d1d2d3de4de5de6de7c8de9c10de11c.c"
    # ).read_text()
    # check_codegen(compiler.generate_header(), reference_header)
    # check_codegen(compiler.generate_source(), reference_source)

    from triton.tools.aot.linker import AOT_C_CUDA_Linker

    # headers = list(trace.kernel_path.parent.glob("*.h"))
    linker = AOT_C_CUDA_Linker(headers)
    # result = linker.generate()
    # with open(trace_dir / f"{trace.kernel_name}-linked.h", "w") as fp:
    #     fp.write(result.header)
    # with open(trace_dir / f"{trace.kernel_name}-linked.cu", "w") as fp:
    #     fp.write(result.source)
    # reference_header = (trace_dir / trace.kernel_name / "matmul.h").read_text()
    # reference_source = (trace_dir / trace.kernel_name / "matmul.c").read_text()
    # check_codegen(result.header, reference_header)
    # check_codegen(result.source, reference_source)


def _preprocess_src(src):
    return list(filter(lambda x: x.strip(), src.split("\n")))


def check_codegen(actual: str, expected: str):
    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)

    for actual, expected in zip(actual_lines, expected_lines):
        assert actual == expected, f"Expected: \n{expected}\nActual: \n{actual}"
