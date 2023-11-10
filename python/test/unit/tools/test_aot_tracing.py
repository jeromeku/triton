import json
import shutil
from pathlib import Path
from typing import List

import pytest

from triton.compiler.compiler import instance_descriptor
from triton.runtime.jit import JITFunction
from triton.tools.aot.compiler import AOT_C_CUDA_Compiler as AOTCompiler
from triton.tools.aot.compiler import AOTCompilationResult
from triton.tools.aot.linker import AOT_C_CUDA_Linker as AOTLinker
from triton.tools.aot.linker import AOTLinkerResult
from triton.tools.aot.tracing import MatMulConfig, MatMulKernelTracer, TraceConfig

FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"

from .matmul_configs import *
from .matmul_utils import *

# ------------------------------------------------------------------------------------------------------------ #
# Tests for `AOTScriptRunner`, which is a wrapper around `triton.tools.compile` and `triton.tools.link`


## Test `generate_signature` for creating signatures to pass to `triton.tools.compile` CLI given dtypes, hints, and constants.
# See `matmul_configs.py` for details
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
def test_signature(dtypes, hints, constants, expected_signature):
    signature = AOTScriptRunner.generate_signature(
        dtypes=dtypes,
        hints=hints,
        constant_vals=constants,
    )

    assert (
        signature == expected_signature
    ), f"Expected signature: {expected_signature}, Actual signature: {signature}"


## Test wrapper for `triton.tools.compile` for generating reference kernels
# See
def test_kernel_compilation():
    out_dir = FIXTURES_DIR / "aot_reference_kernels"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    AOTScriptRunner.compile_matmul_kernels(
        NO_HINT_SIGNATURE, 1, "M/16, N/16, 1", out_dir=out_dir
    )
    kernel_headers = list(out_dir.glob("*.h"))
    kernel_sources = list(out_dir.glob("*.c"))
    print(kernel_headers)
    print(kernel_sources)
    assert len(kernel_headers) == 1
    assert len(kernel_sources) == 1


"""
Tests to replicate the matmul kernels in `test_aot.py` using the refactored object-oriented AOTCompiler and AOTLinker classes.
Currently AOTCompiler and AOTLinker are abstract classes for generating kernel headers / sources and linking these into dispatchable kernels.
The AOT_C_CUDA_Compiler and AOT_C_CUDA_Linker classes are concrete implementations of these abstract classes for generating C/CUDA kernels.

See `triton/tools/aot/compiler.py` and `triton/tools/aot/linker.py` for details.

    - "default" - no specialization for `stride_cm`, `stride_am`
    ```
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
    ```
    - "stride_cm" - `stride_cm` specialized to 16
    - "stride_am" - `stride_am` specialized to 16
    - "stride_cm_am" - `stride_cm` and `stride_am` specialized to 16

Additional test cases:
    - "no_hints" - no specialization for any arg
    - "all_hints" - specialization for all args

See `matmul_configs.py` for config details
"""

TEST_CONFIGS = {
    "default": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=DEFAULT_MATMUL_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    # Same as default except `stride_cm` also specialized to 16
    "stride_cm": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=STRIDE_CM_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    # Same as default except `stride_am` also specialized to 16
    "stride_am": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=STRIDE_AM_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    # Same as default except `stride_cm` and `stride_am` both specialized to 16
    "stride_cm_am": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=STRIDE_CM_AM_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    "no_hints": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=NO_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    "all_hints": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=ALL_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
}

# ------------------------------------------------------------------------------------------------------------ #


# Small utilities for checking that generated code matches reference code
def _preprocess_src(src):
    return list(filter(lambda x: x.strip(), src.split("\n")))


def check_codegen(actual: str, expected: str, skip: List[str] = None, verbose=False):
    """Check that the generated code is the same as the reference code

    Checks for exact text line by line. Ignores lines containing text in `ignore`.

    Args:
        actual (str): generated code
        expected (str): reference
        ignore (List[str], optional): skip line if contains text in `ignore`. Defaults to None.
    """
    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)
    mismatches = []

    for lineno, (actual, expected) in enumerate(zip(actual_lines, expected_lines), 1):
        if skip and any(i in actual for i in skip):
            continue
        if actual != expected:
            mismatches.append(lineno)
            if verbose:
                print(
                    f"Line {lineno} mismatch:\n  Actual: {actual[:100]}\n  Expected: {expected[:100]}"
                )
    assert (
        not mismatches
    ), f'Mismatch in generated code at lines {", ".join(str(l) for l in mismatches)}'
    # if mismatches:
    #     mismatch_str = "\n".join(mismatches)
    #     raise ValueError(f"Mismatches:\n {mismatch_str}")


# ------------------------------------------------------------------------------------------------------------ #
class TestMatMulCodegen:
    @pytest.fixture(
        scope="class",
        params=[
            # Single config tests
            ("all_hints",),
            ("no_hints",),
            ("default",),
            ("stride_cm",),
            ("stride_am",),
            ("stride_cm_am",),
        ],  # ("no_hints",),
        ids=lambda params: "|".join([p.upper() for p in params]),
    )
    def configs(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def kernel_configs(self, configs):
        return [TEST_CONFIGS[cfg] for cfg in configs]

    @pytest.fixture(scope="class")
    def test_dir(self, configs):
        test_dir = (
            Path(__file__).parent
            / "matmul_codegen_test"
            / "_".join(cfg for cfg in configs)
        ).absolute()
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    @pytest.fixture(scope="class")
    def reference_dir(self, test_dir):
        reference_dir = test_dir / "reference_aot_kernels"
        reference_dir.mkdir(parents=True, exist_ok=True)
        return reference_dir

    @pytest.fixture(scope="class")
    def codegen_dir(self, test_dir):
        codegen_dir = test_dir / "codegen_kernels"
        codegen_dir.mkdir(parents=True, exist_ok=True)
        return codegen_dir

    @pytest.fixture(scope="class")
    def kernel_path(self):
        kernel_path = FIXTURES_DIR / "kernels" / "matmul_kernel.py"
        return kernel_path

    @pytest.fixture(scope="class")
    def kernel_name(self):
        """Must match the name of the kernel in `matmul_kernel.py`"""
        return "matmul"

    @pytest.fixture(scope="class")
    def expected_kernels(
        self, kernel_name, reference_dir: Path, kernel_configs, kernel_path: Path
    ):
        signatures = [
            AOTScriptRunner.generate_signature(
                kernel_config.dtypes, kernel_config.hints, kernel_config.constants
            )
            for kernel_config in kernel_configs
        ]

        num_warps = [kernel_config.num_warps for kernel_config in kernel_configs]
        grids = [kernel_config.grid for kernel_config in kernel_configs]

        AOTScriptRunner.compile_matmul_kernels(
            kernel_name,
            signatures,
            num_warps=num_warps,
            grids=grids,
            out_dir=reference_dir,
            kernel_path=kernel_path,
        )
        headers = list(reference_dir.glob("*.h"))
        sources = list(reference_dir.glob("*.c"))

        linked_header = list(reference_dir.glob(f"{kernel_name}.h"))
        kernel_headers = list(set(headers) - set(linked_header))
        linked_source = list(reference_dir.glob(f"{kernel_name}.c"))
        kernel_sources = list(set(sources) - set(linked_source))

        jit_args = list(reference_dir.glob("*jit_args.json"))
        compiler_params = list(reference_dir.glob("*params.json"))

        return AOTScriptResult(
            kernel_headers=kernel_headers,
            kernel_sources=kernel_sources,
            linked_header=linked_header,
            linked_source=linked_source,
            jit_args=jit_args,
            compiler_params=compiler_params,
        )

    def test_script_gen(self, expected_kernels):
        assert len(expected_kernels.linked_header) == 1
        assert len(expected_kernels.kernel_headers) >= 1

        assert len(expected_kernels.linked_source) == 1
        assert len(expected_kernels.kernel_sources) >= 1

    def _parse_jit_args(self, args_path):
        class JITArgTypes:
            """Expected types for JIT args"""

            INT = ["num_warps", "num_stages", "num_ctas", "device"]
            STRING = ["device_type"]
            DICT = [
                "signature",
                "constants",
                "_original_signature",
                "_original_constants",
            ]
            LIST = ["grid", "configs"]
            BOOL = ["enable_warp_specialization", "enable_fp_fusion", "debug"]

        class JITArgDeserializer:  # Need special handling for
            @staticmethod
            def deserialize(args):
                parsed_args = {}
                for k, v in args.items():
                    if k in JITArgTypes.INT:
                        parsed_args[k] = int(v)
                    elif k in JITArgTypes.BOOL:
                        if v.lower() == "true":
                            parsed_args[k] = True
                        elif v.lower() == "false":
                            parsed_args[k] = False
                        else:
                            raise ValueError(f"Invalid bool value {v}")
                    elif k in JITArgTypes.DICT:
                        # Cast arg positions to ints for signature and constants
                        parsed_args[k] = {int(k): v for k, v in v.items()}
                    elif k == "configs":
                        # Create instance descriptors from dict representation
                        parsed_args[k] = [instance_descriptor(**cfg) for cfg in v]
                    else:
                        parsed_args[k] = v
                return parsed_args

        raw_args = json.load(args_path.open())
        return JITArgDeserializer.deserialize(raw_args)

    @pytest.fixture(scope="class")
    def codegen_kernels(self, kernel_name, kernel_path, expected_kernels, codegen_dir):
        jit_args = []
        for p in expected_kernels.jit_args:
            jit_args.append(self._parse_jit_args(p))

        compiled_results = []
        for args in jit_args:
            print(args)
            jit_fn = JITFunction(kernel_path)
            compiler = AOTCompiler(
                kernel_name=kernel_name,
                jit_args=args,
                jit_fn=jit_fn,
                save_dir=codegen_dir,
            )
            compiled_result: AOTCompilationResult = compiler.generate()
            compiled_results.append(compiled_result)
        return compiled_results

    @pytest.fixture(scope="class")
    def codegen_linked_kernels(self, kernel_name, codegen_kernels, codegen_dir: Path):
        headers = [t.header_path for t in codegen_kernels]
        linker = AOTLinker(
            kernel_name=kernel_name,
            headers=headers,
            save_dir=codegen_dir,
        )

        linker_result: AOTLinkerResult = linker.generate()
        return linker_result

    def test_aot_compiler_params(
        self,
        expected_kernels,
        codegen_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels
        actual_params = [k._compiler_params for k in codegen_kernels]
        for actual, expected in zip(actual_params, expected_kernels.compiler_params):
            expected = json.load(expected.open())
            for k in actual.keys():
                assert (
                    actual[k] == expected[k]
                ), f"{k.upper()} not equal\n\tExpected: {expected[k]}, Actual: {actual[k]}"

    def test_aot_codegen_kernel_headers(
        self,
        expected_kernels,
        codegen_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels
        actual_headers = [k.header for k in codegen_kernels]
        for actual, expected in zip(
            sorted(actual_headers), sorted(expected_kernels.kernel_headers)
        ):
            expected = expected.read_text()
            check_codegen(actual, expected)

    def test_aot_codegen_kernel_sources(
        self,
        expected_kernels,
        codegen_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels
        actual_sources = [k.source for k in codegen_kernels]
        for actual, expected in zip(
            sorted(actual_sources), sorted(expected_kernels.kernel_sources)
        ):
            expected = expected.read_text()
            check_codegen(actual, expected)

    def test_aot_codegen_linked_header(
        self,
        expected_kernels,
        codegen_linked_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels

        check_codegen(
            codegen_linked_kernels.header_path.read_text(),
            expected_kernels.linked_header[0].read_text(),
        )

    def test_aot_codegen_linked_source(
        self,
        expected_kernels,
        codegen_linked_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels

        check_codegen(
            codegen_linked_kernels.source_path.read_text(),
            expected_kernels.linked_source[0].read_text(),
        )


"""
Tests for tracing JIT functions and generating AOT kernels from the traced functions.
Motivation is that rather than having the user input a signature, we utilize the existing JIT / compilation machinery to 
generate the necessary inputs to `triton.compiler.compile`.

AOTTracer is an abstract class for tracing JIT functions and generating AOT kernels from the traced functions that utilizes AOTCompiler 
and AOTLinker to generate the AOT kernels. 

AOTTracer is implemented by MatMulKernelTracer, which is a concrete implementation of AOTTracer for the matmul kernel.

Since traced code will differ from scripted (`triton.tools.compile`) due to handling of specializations and signatures,
we don't test for exact codegen matches. In particular, since the AOT script does not handle all specializations in
the JITTed path such as `ids_of_folded_args` and `divisible_by_8`, any codegen'ed that depend on the `config` passed to 
`triton.compiler.compile` will necessarily differ.

This includes:
- the suffix attached generated kernel header and source files
- fields in the `triton.tools.compile.{h,c}` templates such as `kernel_name` and `triton_kernel_name`
- the generated cubin image, which will differ due to the differing specializations.

Hence, when comparing the generated code, we skip the aforementioned fields (`SKIP_PARAMS` attribute).
See `triton/tools/aot/tracing.py` for details.
"""


class TestMatMulTrace:
    SKIP_PARAMS = ["kernel_name", "triton_kernel_name", "bin_size", "bin_data"]

    @pytest.fixture(
        scope="class",
        params=[("all_hints",)],  # ("no_hints",),
        ids=lambda params: "|".join([p.upper() for p in params]),
    )
    def configs(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def kernel_configs(self, configs):
        return [TEST_CONFIGS[cfg] for cfg in configs]

    @pytest.fixture(scope="class")
    def test_dir(self, configs):
        test_dir = (
            Path(__file__).parent
            / "matmul_trace_test"
            / "_".join(cfg for cfg in configs)
        ).absolute()
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    @pytest.fixture(scope="class")
    def reference_dir(self, test_dir):
        reference_aot_dir = test_dir / "reference_aot_kernels"
        reference_aot_dir.mkdir(parents=True, exist_ok=True)
        return reference_aot_dir

    @pytest.fixture(scope="class")
    def trace_dir(self, test_dir):
        trace_dir = test_dir / "traced_kernels"
        trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_dir

    @pytest.fixture(scope="class")
    def kernel_path(self):
        kernel_path = FIXTURES_DIR / "kernels" / "matmul_kernel.py"
        return kernel_path

    @pytest.fixture(scope="class")
    def kernel_name(self):
        """Must match the name of the kernel in `matmul_kernel.py`"""
        return "matmul"

    @pytest.fixture(scope="class")
    def expected_kernels(
        self, kernel_name, reference_dir: Path, kernel_configs, kernel_path: Path
    ):
        signatures = [
            AOTScriptRunner.generate_signature(
                kernel_config.dtypes, kernel_config.hints, kernel_config.constants
            )
            for kernel_config in kernel_configs
        ]

        num_warps = [kernel_config.num_warps for kernel_config in kernel_configs]
        grids = [kernel_config.grid for kernel_config in kernel_configs]

        AOTScriptRunner.compile_matmul_kernels(
            kernel_name,
            signatures,
            num_warps=num_warps,
            grids=grids,
            out_dir=reference_dir,
            kernel_path=kernel_path,
        )
        headers = list(reference_dir.glob("*.h"))
        sources = list(reference_dir.glob("*.c"))

        linked_header = list(reference_dir.glob(f"{kernel_name}.h"))
        kernel_headers = list(set(headers) - set(linked_header))
        linked_source = list(reference_dir.glob(f"{kernel_name}.c"))
        kernel_sources = list(set(sources) - set(linked_source))

        jit_args = list(reference_dir.glob("*jit_args.json"))
        compiler_params = list(reference_dir.glob("*params.json"))

        return AOTScriptResult(
            kernel_headers=kernel_headers,
            kernel_sources=kernel_sources,
            linked_header=linked_header,
            linked_source=linked_source,
            jit_args=jit_args,
            compiler_params=compiler_params,
        )

    # @pytest.fixture(scope="class")
    @pytest.fixture(scope="class")
    def traced_kernels(
        self,
        trace_dir,
        kernel_path,
        kernel_configs,
    ) -> List[AOTCompilationResult]:
        trace_configs = []
        matmul_configs = []
        for kernel_config in kernel_configs:
            dtype_in = tt_to_torch(kernel_config.dtypes["A"])
            dtype_out = tt_to_torch(kernel_config.dtypes["C"])

            # Assume that M, N, K are divisible by 16; defaults to 16
            matmul_config = MatMulConfig(
                dtype_in=dtype_in,
                dtype_out=dtype_out,
                BLOCK_M=kernel_config.constants["BLOCK_M"],
                BLOCK_N=kernel_config.constants["BLOCK_N"],
                BLOCK_K=kernel_config.constants["BLOCK_K"],
            )
            assert matmul_config.M % matmul_config.BLOCK_M == 0
            assert matmul_config.N % matmul_config.BLOCK_N == 0
            assert matmul_config.K % matmul_config.BLOCK_K == 0

            do_not_specialize = [
                k for k in kernel_config.hints if kernel_config.hints[k] is None
            ]
            trace_config = TraceConfig(
                do_not_specialize=do_not_specialize,
                num_warps=kernel_config.num_warps,
                trace_dir=trace_dir,
                trace_grid=[g.strip() for g in kernel_config.grid.split(",")],
            )
            trace_configs.append(trace_config)
            matmul_configs.append(matmul_config)

        # Use default MatmulConfig (16 x 16 x 16), dtype_in = fp16, dtype_out = fp32
        matmul_tracer = MatMulKernelTracer(kernel_path.parent)
        traces, *_ = matmul_tracer.trace(
            kernel_configs=matmul_configs, trace_configs=trace_configs
        )
        # Save jit args
        for trace in traces:
            with open(trace_dir / f"{trace.kernel_name}-jit_args.json", "w") as fp:
                json.dump({k: str(v) for k, v in trace._jit_args.items()}, fp, indent=2)
            with open(
                trace_dir / f"{trace.kernel_name}-compiler_params.json", "w"
            ) as fp:
                json.dump(
                    {k: str(v) for k, v in trace._compiler_params.items()}, fp, indent=2
                )
        return traces

    @pytest.fixture(scope="class")
    def linked_traces(self, kernel_name, traced_kernels, trace_dir: Path):
        headers = [t.header_path for t in traced_kernels]
        linker = AOTLinker(
            kernel_name=kernel_name,
            headers=headers,
            save_dir=trace_dir,
        )

        linker_result: AOTLinkerResult = linker.generate()
        return linker_result

    @pytest.fixture
    def skip_params(self, traced_kernels):
        skip = {}
        for trace in traced_kernels:
            for k, v in trace._compiler_params.items():
                if k in self.SKIP_PARAMS:
                    skip[k] = str(v)
        return skip

    def extract_kernel_sig(self, trace: AOTCompilationResult):
        return "_".join(trace.params["kernel_name"].split("_")[1:])

    def extract_sig_hash(self, trace: AOTCompilationResult):
        return self.extract_kernel_sig(trace).split("_")[0]

    # Only test for presence of `sig hash` in header file name
    # Per above documentation, the suffix will differ due to the differing specializations.
    def test_kernel_header_files(self, traced_kernels, expected_kernels, skip_params):
        for trace in traced_kernels:
            kernel_sig = self.extract_kernel_sig(trace)
            sig_hash_suffix = kernel_sig.split("_")
            assert len(sig_hash_suffix) == 2

            sig_hash = sig_hash_suffix[0]

            assert any(sig_hash in str(h) for h in expected_kernels.kernel_headers)

    def test_kernel_header_match(self, traced_kernels, expected_kernels, skip_params):
        for trace in traced_kernels:
            sig_hash = self.extract_sig_hash(trace)
            expected_header = [
                h for h in expected_kernels.kernel_headers if sig_hash in str(h)
            ][0].read_text()
            actual_header = trace.header
            check_codegen(
                actual_header,
                expected_header,
                skip=list(skip_params.values()),
                verbose=True,
            )

    def test_kernel_source_match(self, traced_kernels, expected_kernels, skip_params):
        for trace in traced_kernels:
            sig_hash = self.extract_sig_hash(trace)
            expected_source = [
                s for s in expected_kernels.kernel_sources if sig_hash in str(s)
            ][0].read_text()
            actual_source = trace.source
            check_codegen(
                actual_source,
                expected_source,
                skip=list(skip_params.values()),
                verbose=True,
            )

    def test_linked_header_match(self, linked_traces, expected_kernels, skip_params):
        expected_header = expected_kernels.linked_header[0].read_text()
        actual_header = linked_traces.header

        check_codegen(
            actual_header,
            expected_header,
            skip=list(skip_params.values()),
            verbose=True,
        )

    def test_linked_source_match(
        self, kernel_name, linked_traces, expected_kernels, skip_params
    ):
        expected_source = expected_kernels.linked_source[0].read_text()
        actual_source = linked_traces.source

        check_codegen(
            actual_source,
            expected_source,
            skip=list(skip_params.values()),
            verbose=True,
        )


# @pytest.mark.parametrize("config_name", [("no_hints")], ids=lambda x: x.upper())
# def test_single_trace(
#     config_name,
# ):
#     from triton.tools.aot.compiler import AOT_C_CUDA_Compiler
#     from triton.tools.aot.tracing import AOTTraceResult

#     # Set up directories for reference and traced kernel artifacts
#     test_dir = (Path(__file__).parent / "test").absolute()
#     if test_dir.exists():
#         shutil.rmtree(test_dir)
#     test_dir.mkdir(parents=True, exist_ok=True)

#     reference_aot_dir = test_dir / "reference_aot_kernels"
#     reference_aot_dir.mkdir(parents=True, exist_ok=True)

#     trace_dir = test_dir / "traced_kernels"
#     trace_dir.mkdir(parents=True, exist_ok=True)

#     test_config = TEST_CONFIGS[config_name]
#     signature = generate_signature(
#         test_config.dtypes, test_config.hints, test_config.constants
#     )

#     kernel_path = FIXTURES_DIR / "kernels" / "matmul_kernel.py"

#     compile_matmul_kernels(
#         signature,
#         num_warps=test_config.num_warps,
#         grids=test_config.grid,
#         out_dir=reference_aot_dir,
#         kernel_path=kernel_path,
#     )
#     reference_headers = list(reference_aot_dir.glob("*.h"))
#     reference_sources = list(reference_aot_dir.glob("*.c"))

#     # Construct MatMulConfig and TraceConfig
#     dtype_in = _tt_to_torch(test_config.dtypes["A"])
#     dtype_out = _tt_to_torch(test_config.dtypes["C"])

#     # Assume that M, N, K are divisible by 16; defaults to 16
#     matmul_config = MatMulConfig(
#         dtype_in=dtype_in,
#         dtype_out=dtype_out,
#         BLOCK_M=test_config.constants["BLOCK_M"],
#         BLOCK_N=test_config.constants["BLOCK_N"],
#         BLOCK_K=test_config.constants["BLOCK_K"],
#     )
#     assert matmul_config.M % matmul_config.BLOCK_M == 0
#     assert matmul_config.N % matmul_config.BLOCK_N == 0
#     assert matmul_config.K % matmul_config.BLOCK_K == 0

#     do_not_specialize = [k for k in test_config.hints if test_config.hints[k] is None]
#     trace_config = TraceConfig(
#         do_not_specialize=test_config.hints,
#         num_warps=test_config.num_warps,
#         trace_dir=trace_dir,
#     )

#     # Use default MatmulConfig (16 x 16 x 16), dtype_in = fp16, dtype_out = fp32
#     matmul_tracer = MatMulKernelTracer(kernel_path.parent)
#     traces, *_ = matmul_tracer.trace(
#         kernel_configs=[matmul_config], trace_configs=[trace_config]
#     )
#     trace: AOTTraceResult = traces[0]

#     compiled_header = trace.compilation_result.header_path
#     compiled_source = trace.compilation_result.source_path
#     linked_header = trace.linker_result.header_path
#     linked_source = trace.linker_result.source_path

#     assert all(
#         p.exists()
#         for p in [compiled_header, compiled_source, linked_header, linked_source]
#     )

#     reference_header_files = [p.name for p in reference_headers]
#     reference_source_files = [p.name for p in reference_sources]

#     actual_headers = [compiled_header.name, linked_header.name]
#     actual_sources = [compiled_source.name, linked_source.name]

#     assert set(reference_header_files) == set(
#         actual_headers
#     ), f"Expected: {reference_header_files}, Actual: {actual_headers}"
#     assert set(reference_source_files) == set(
#         actual_sources
#     ), f"Expected: {reference_source_files}, Actual: {actual_sources}"

#     kernel_sig = "_".join(trace.compilation_result.params["kernel_name"].split("_")[1:])
#     expected_kernel_header = [f for f in reference_headers if kernel_sig in str(f)][
#         0
#     ].read_text()
#     actual_kernel_header = compiled_header.read_text()
#     check_codegen(actual_kernel_header, expected_kernel_header)
# Check matching filenames

# trace: AOTTraceResult = traces[0]
# import json

# with open(trace_dir / f"{trace.kernel_name}-jit_args.json", "w") as fp:
#     json.dump({k: str(v) for k, v in trace.jit_args.items()}, fp, indent=2)
# # trace: TraceArtifact = traces[0]
# # for actual, expected in zip(outputs, checks):
# #     is_close = torch.allclose(actual, expected, atol=1e-1, rtol=0)
# #     print(f"Is close {is_close}")

# compiler = AOT_C_CUDA_Compiler(
#     kernel_name=trace.kernel_name,
#     compiled_binary=trace.compiled_binary,
#     jit_args=trace.jit_args,
#     jit_fn=trace.jit_fn,
# )

# with open(trace_dir / f"{trace.kernel_name}-compiled.h", "w") as fp:
#     fp.write(compiler.generate_header())
# with open(trace_dir / f"{trace.kernel_name}-compiled.c", "w") as fp:
#     fp.write(compiler.generate_source())

# # Check that the generated code is the same as the reference code
# # reference_header = (
# #     trace_dir
# #     / trace.kernel_name
# #     / "matmul.9abb00f7_0d1d2d3de4de5de6de7c8de9c10de11c.h"
# # ).read_text()
# # reference_source = (
# #     trace_dir
# #     / trace.kernel_name
# #     / "matmul.9abb00f7_0d1d2d3de4de5de6de7c8de9c10de11c.c"
# # ).read_text()
# # check_codegen(compiler.generate_header(), reference_header)
# # check_codegen(compiler.generate_source(), reference_source)

# from triton.tools.aot.linker import AOT_C_CUDA_Linker

# # headers = list(trace.kernel_path.parent.glob("*.h"))
# linker = AOT_C_CUDA_Linker(headers)
# # result = linker.generate()
# # with open(trace_dir / f"{trace.kernel_name}-linked.h", "w") as fp:
# #     fp.write(result.header)
# # with open(trace_dir / f"{trace.kernel_name}-linked.cu", "w") as fp:
# #     fp.write(result.source)
# # reference_header = (trace_dir / trace.kernel_name / "matmul.h").read_text()
# # reference_source = (trace_dir / trace.kernel_name / "matmul.c").read_text()
# # check_codegen(result.header, reference_header)
# # check_codegen(result.source, reference_source)
