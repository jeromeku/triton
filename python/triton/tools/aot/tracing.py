import glob
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import torch
from dataclasses import dataclass

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
from triton.tools.aot import CompiledArtifact, JITCompileArgs

"""
Utilities for generating reference AOT kernels 
"""

KERNELS_DIR = Path(__file__).parent / "fixtures" / "kernels"


# Copied from test/unittest/tools/test_aot.py
class AOTScriptRunner:
    """Wrapper around `triton.tools` for AOT compilation

    Runs `triton.tools.compile` and `triton.tools.link` in subprocesses
    """

    @staticmethod
    def _compile_kernel(
        dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path
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

    @staticmethod
    def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
        # compile all desired configs
        for ha in ha_hb_hints:
            for hb in ha_hb_hints:
                sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
                name = f"matmul_{dtype}"
                grid = f"M/{BM}, N/{BN}, 1"
                AOTScriptRunner._compile_kernel(
                    dir=dir,
                    signature=sig,
                    kernel_name="kernel",
                    out_name=name,
                    out_path=name,
                    num_warps=1,
                    grid=grid,
                    kernel_path=kernel_path,
                )

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


def generate_matmul_reference(
    kernel_path, out_dir, BM=16, BN=16, BK=16, hints=["", ":16"], dtype="fp16"
):
    AOTScriptRunner.compile_aot_kernels(out_dir, kernel_path, dtype, BM, BN, BK, hints)
    AOTScriptRunner.link_aot_kernels(out_dir, "matmul")


# Tracing Tools #


@dataclass
class TraceConfig(dict):
    """Kwargs passed to `JITFunction.run`"""

    # Launch params
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    num_ctas: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = True

    # JIT options
    do_not_specialize: List[str] = None
    debug: bool = False
    noinline: bool = False

    # Additional options
    extern_libs: List[str] = None
    stream: Optional[int] = None
    warmup: bool = False
    device: Optional[int] = None
    device_type: Optional[str] = None
    # Trace options
    trace: bool = True
    trace_dir: Optional[Path] = None

    def __post_init__(self):
        self.update(self.__dict__)


# Kernel-specific configs


@dataclass
class MatMulConfig(dict):
    dtype_in: torch.dtype = torch.float16
    dtype_out: torch.dtype = torch.float32
    M: int = 16
    N: int = 16
    K: int = 16
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 16
    BLOCK_K: tl.constexpr = 16
    seed: torch.seed = 0

    def __post_init__(self):
        self.update(self.__dict__)


class KernelTracer(ABC):
    KERNEL: str

    @abstractmethod
    def build_args(self, config):
        """Set args for the kernel as an OrderedDict"""
        ...

    @abstractmethod
    def build_constants(self, config):
        """Set constants for the kernel as a dict"""
        ...

    @abstractmethod
    def build_grid(self, config):
        """Set grid for the kernel as a callable or a 3-tuple of ints"""
        ...

    def check_specializations(self, params, expected_specializations):
        no_specs = [p.name for p in params if p.do_not_specialize]

        assert set(no_specs) == set(
            expected_specializations
        ), f"Incorrect specializations, expected {expected_specializations}"

    def check_args(self, args, expected_args):
        assert len(args) == len(
            expected_args
        ), f"Incorrect number of args, expected {expected_args}"
        for i, (expected_key, actual_key) in enumerate(zip(expected_args, args.keys())):
            assert (
                expected_key == actual_key
            ), f"Incorrect arg name at position {i}, expected {expected_key}, got {actual_key}"

    def check_constants(self, kernel_constants, expected_constants):
        assert set(kernel_constants.keys()) == set(
            expected_constants
        ), f"Incorrect constants, expected {expected_constants}"

    def trace(self, kernel_config, trace_config: TraceConfig):
        """Trace a kernel with the given args and constants

        Args:
            additional_jit_kwargs: number of warps, specializations, etc. -- see `triton.runtime.jit.JITFunction.run` special args.
        """

        do_not_specialize = trace_config.pop("do_not_specialize")
        debug = trace_config.pop("debug")
        noinline = trace_config.pop("noinline")

        jitted_fn = JITFunction(
            self.KERNEL,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
        )
        if do_not_specialize:
            self.check_specializations(jitted_fn.params, do_not_specialize)

        expected_constants = [p.name for p in jitted_fn.params if p.is_constexpr]
        expected_args = [p.name for p in jitted_fn.params if not p.is_constexpr]
        # Check do not specialize

        args = self.build_args(kernel_config)
        self.check_args(args, expected_args)

        constants = self.build_constants(kernel_config)
        self.check_constants(constants, expected_constants)

        grid = self.build_grid(kernel_config)

        compilation_artifact: CompiledArtifact = jitted_fn[grid](
            *args.values(),
            **constants,
            **trace_config,
        )
        return compilation_artifact


class MatMulKernelTracer(KernelTracer):
    KERNEL = KERNELS_DIR / "matmul_kernel.py"

    def trace(
        self, kernel_configs: List[MatMulConfig], trace_configs: List[TraceConfig]
    ):
        outputs = []
        checks = []
        traces = []
        for kconfig, tconfig in zip(kernel_configs, trace_configs):
            trace = super().trace(kconfig, tconfig)
            traces.append(trace)
            checks.append(self.CHECK)
            outputs.append(self.C.detach().cpu())
        return traces, outputs, checks

    def build_args(self, config: MatMulConfig):
        # Set up matrices
        torch.manual_seed(config.seed)

        A_shape = config.M, config.K
        B_shape = config.K, config.N
        C_shape = config.M, config.N
        A = torch.randn(A_shape, dtype=config.dtype_in, device="cuda").contiguous()
        B = torch.randn(B_shape, dtype=config.dtype_in, device="cuda").contiguous()
        C = torch.empty(C_shape, dtype=config.dtype_out, device="cuda")

        # Save for verification
        self.CHECK = torch.matmul(A, B).to(config.dtype_out).detach().cpu()
        self.C = C
        M, K = A.shape
        _, N = B.shape

        stride_cm = C.stride(0)
        stride_cn = C.stride(1)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)

        args = OrderedDict(
            C=C,
            A=A,
            B=B,
            M=M,
            N=N,
            K=K,
            stride_cm=stride_cm,
            stride_cn=stride_cn,
            stride_am=stride_am,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            stride_bn=stride_bn,
        )

        return args

    def build_constants(self, config: MatMulConfig):
        return {
            "BLOCK_M": config.BLOCK_M,
            "BLOCK_N": config.BLOCK_N,
            "BLOCK_K": config.BLOCK_K,
        }

    def build_grid(self, config: MatMulConfig):
        grid = lambda META: (
            (
                triton.cdiv(config.M, META["BLOCK_M"]),
                triton.cdiv(config.N, META["BLOCK_N"]),
                1,
            )
        )
        return grid


def trace_matmul_kernels():
    """Replicate the compiled kernel headers and sources for matmul kernels in `test_aot.py`"""

    trace_dir = Path("traced_kernels").absolute()
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # 4 cases
    # no specialization for stride_cm, stride_am
    # no specialization for stride_cm only
    # no specialization for stride_am only
    # implicit specializations for both by passing appropriately shaped tensors (divisible by 16)

    specializations = [("stride_cm", "stride_am"), ("stride_cm",), ("stride_am",)]
    trace_configs = []
    for spec in specializations:
        trace_configs.append(TraceConfig(do_not_specialize=spec, trace_dir=trace_dir))

    trace_configs.append(TraceConfig())

    # Use default MatmulConfig (16 x 16 x 16), dtype_in = fp16, dtype_out = fp32
    matmul_config = MatMulConfig()
    matmul_tracer = MatMulKernelTracer()

    kernel_configs = [matmul_config] * len(trace_configs)

    traces, outputs, checks = matmul_tracer.trace(
        kernel_configs=kernel_configs, trace_configs=trace_configs
    )

    for actual, expected in zip(outputs, checks):
        is_close = torch.allclose(actual, expected, atol=1e-1, rtol=0)
        print(f"Is close {is_close}")
