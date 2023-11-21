import json
import re
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

from .matmul_configs import (
    ALL_HINTS,
    DEFAULT_MATMUL_CONSTANTS,
    DEFAULT_MATMUL_DTYPES,
    DEFAULT_MATMUL_HINTS,
    DEFAULT_SIGNATURE,
    NO_HINT_SIGNATURE,
    NO_HINTS,
    STRIDE_AM_HINTS,
    STRIDE_AM_SIGNATURE,
    STRIDE_CM_AM_HINTS,
    STRIDE_CM_AM_SIGNATURE,
    STRIDE_CM_HINTS,
    STRIDE_CM_SIGNATURE,
)
from .matmul_utils import (
    AOTScriptResult,
    AOTScriptRunner,
    MatmulTestConfig,
    tt_to_torch,
)

FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"

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
def test_kernel_compilation(tmp_path):
    AOTScriptRunner.compile_matmul_kernels(
        "matmul", NO_HINT_SIGNATURE, 1, "M/16, N/16, 1", out_dir=tmp_path
    )
    kernel_headers = list(tmp_path.glob("*.h"))
    kernel_sources = list(tmp_path.glob("*.c"))
    assert len(kernel_headers) == 2
    assert len(kernel_sources) == 2


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
            mismatches.append((lineno, actual, expected))
            if verbose:
                print(
                    f"Line {lineno} mismatch:\n  Actual: {actual[:100]}\n  Expected: {expected[:100]}"
                )

    assert (
        not mismatches
    ), f'Mismatch in generated code at lines {", ".join(str(l) for l in mismatches)}'


def check_linked_source(
    actual: str, expected: str, skip: List[str] = None, verbose=False
):
    """Linked source files generated by traced kernels will differ from reference source files due to differing specializations.

    We check specifically that the dispatcher conditions match and that the dispatched function call args match.
    The actual function names will differ due to the suffix attached to the generated function name which depends on
    specialization config.
    """

    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)
    mismatches = []
    for i in range(len(actual_lines)):
        actual_line = actual_lines[i]

        if actual_line.lstrip().startswith("if") and "return" in actual_lines[i + 1]:
            if verbose:
                print(
                    f"Checking dispatch condition:\nActual: {actual_line}\n{actual_lines[i+1]}"
                )
            # Check that the actual dispatch condition exists in the expected source
            assert actual_line in expected_lines
            # Parse return statement for args -- function name won't match because of suffix
            actual_dispatch_fn = actual_lines[i + 1]
            match = re.search(r"\((.*?)\)", actual_dispatch_fn)
            assert match is not None
            actual_dispatch_args = match.group(1).split(",")
            expected_line = expected_lines[expected_lines.index(actual_line)]
            expected_dispatch_fn = expected_lines[expected_lines.index(actual_line) + 1]
            if verbose:
                print(f"Expected: {expected_line}\n{expected_dispatch_fn}")
            assert "return" in expected_dispatch_fn
            match = re.search(r"\((.*?)\)", expected_dispatch_fn)
            assert match is not None
            expected_dispatch_args = match.group(1).split(",")
            assert actual_dispatch_args == expected_dispatch_args
        else:
            if skip and any(i in actual_line for i in skip):
                continue

            if actual_line != expected_lines[i]:
                mismatches.append((i, actual, expected))
                if verbose:
                    print(
                        f"Line {i} mismatch:\n  Actual: {actual[:100]}\n  Expected: {expected[:100]}"
                    )
        assert not mismatches


# ------------------------------------------------------------------------------------------------------------ #
class TestMatMulCodegen:
    @pytest.fixture(
        scope="class",
        params=[
            # Single config tests
            ("default",),
            ("stride_cm",),
            ("stride_am",),
            ("stride_cm_am",),
            ("all_hints",),
            ("no_hints",),
            # Multi config tests
            ("all_hints", "no_hints"),
            ("default", "stride_cm", "stride_am", "stride_cm_am"),
        ],
        ids=lambda params: "-".join([p.upper() for p in params]),
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

        yield test_dir

        shutil.rmtree(test_dir)

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
        actual_params = sorted(actual_params, key=lambda x: x["kernel_name"])
        for actual in actual_params:
            kernel_sig = "_".join(actual["kernel_name"].split("_")[1:])
            expected = [
                p for p in expected_kernels.compiler_params if kernel_sig in str(p)
            ][0]
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
        params=[
            # Single config tests
            ("default",),
            ("stride_cm",),
            ("stride_am",),
            ("stride_cm_am",),
            ("all_hints",),
            ("no_hints",),
            # Multi config tests
            ("all_hints", "no_hints"),
            ("default", "stride_cm", "stride_am", "stride_cm_am"),
        ],  # ("no_hints",),
        ids=lambda params: "-".join([p.upper() for p in params]),
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

        yield test_dir

        shutil.rmtree(test_dir)

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
    def link_skips(self, traced_kernels):
        skip = []
        for trace in traced_kernels:
            for k, v in trace._compiler_params.items():
                if k in self.SKIP_PARAMS:
                    skip.append(str(v))
        return skip

    def get_skips(self, trace):
        skip = {}
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
    def test_kernel_header_files(self, traced_kernels, expected_kernels):
        for trace in traced_kernels:
            kernel_sig = self.extract_kernel_sig(trace)
            sig_hash_suffix = kernel_sig.split("_")
            assert len(sig_hash_suffix) == 2

            sig_hash = sig_hash_suffix[0]

            assert any(sig_hash in str(h) for h in expected_kernels.kernel_headers)

    def test_kernel_header_match(self, traced_kernels, expected_kernels):
        for trace in traced_kernels:
            sig_hash = self.extract_sig_hash(trace)
            expected_header = [
                h for h in expected_kernels.kernel_headers if sig_hash in str(h)
            ][0].read_text()
            actual_header = trace.header
            skip = self.get_skips(trace)
            check_codegen(
                actual_header,
                expected_header,
                skip=list(skip.values()),
                verbose=True,
            )

    def test_kernel_source_match(self, traced_kernels, expected_kernels):
        for trace in traced_kernels:
            sig_hash = self.extract_sig_hash(trace)
            expected_source = [
                s for s in expected_kernels.kernel_sources if sig_hash in str(s)
            ][0].read_text()
            actual_source = trace.source
            skip = self.get_skips(trace)
            check_codegen(
                actual_source,
                expected_source,
                skip=list(skip.values()),
                verbose=True,
            )

    def test_linked_header_match(self, linked_traces, expected_kernels, link_skips):
        expected_header = expected_kernels.linked_header[0].read_text()
        actual_header = linked_traces.header

        check_codegen(
            actual_header,
            expected_header,
            skip=link_skips,
            verbose=True,
        )

    def test_linked_source_match(self, linked_traces, expected_kernels, link_skips):
        expected_source = expected_kernels.linked_source[0].read_text()
        actual_source = linked_traces.source

        check_linked_source(
            actual_source,
            expected_source,
            skip=link_skips,
            verbose=True,
        )


# -----#

from typing import Optional

import torch

import triton

key_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
key_head_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
value_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")

cached_bin = None


@torch.no_grad()
def paged_flash_attention_fwd(
    output: torch.Tensor,  # [num_seqs, num_candidates, num_heads, head_dim]
    query: torch.Tensor,  # [num_seqs, num_candidates, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, head_size // x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    scale: float,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [num_seqs]
    block_size: int,
    max_context_len: int,
    alibi_slops: Optional[torch.Tensor],
    medusa_attn_mask: Optional[torch.Tensor],  # [num_candidates, num_candidates]
):
    global key_block_idx, key_head_idx, value_block_idx
    # does not support alibi for now
    assert alibi_slops is None

    global cached_bin

    # query = query.expand(1, -1, -1, -1)
    # output = output.expand(1, -1, -1, -1)
    # medusa_attn_mask = medusa_attn_mask[0][0]

    BLOCK_M = 16
    BLOCK_N = 128
    assert BLOCK_N % block_size == 0

    # FIXME(sunpeng17): avoid hardcode
    X = 16 // torch.tensor([], dtype=query.dtype).element_size()

    # batch, num_candidates, num_heads, head_dim = query.shape
    num_candidates, num_heads, head_dim = query.shape
    batch = 1
    head_dim = query.shape[-1]

    ################### DEBUG ONLY #########################
    # q = torch.zeros([BLOCK_M, head_dim], dtype=query.dtype, device=query.device)
    # k = torch.zeros([head_dim, BLOCK_N], dtype=query.dtype, device=query.device)
    # qk = torch.zeros([BLOCK_M, BLOCK_N], dtype=query.dtype, device=query.device)
    # p = torch.zeros([BLOCK_M, BLOCK_N], dtype=query.dtype, device=query.device)
    # v = torch.zeros([BLOCK_N, head_dim], dtype=query.dtype, device=query.device)
    # am = torch.zeros([BLOCK_M, BLOCK_N], dtype=query.dtype, device=query.device)
    ################### DEBUG ONLY #########################

    assert num_candidates <= BLOCK_M  # some code in the kernel assumes only 1 BLOCK_M
    assert head_dim % X == 0

    # batch, head, q block
    # normally we don't need to partition q because num_candidates are typically small
    # but we keep the code here for future extension
    #
    # a block in the grid processes a BLOCK_M sequences
    #
    # TODO(sunpeng17): possible future parallel alone K and V blocks?
    grid = (batch, num_heads, triton.cdiv(num_candidates, BLOCK_M))

    if cached_bin is not None:
        bin = cached_bin
        stream = torch.cuda.current_stream().cuda_stream
        args = [
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            block_tables,
            context_lens,
            medusa_attn_mask,
            scale,
            num_candidates,
            output.stride(0),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            query.stride(0),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            key_cache.stride(4),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            medusa_attn_mask.stride(2),
            medusa_attn_mask.stride(3),
            key_block_idx,
            key_head_idx,
            value_block_idx,
        ]
        bin.c_wrapper(
            grid[0],
            grid[1],
            grid[2],
            bin.num_warps,
            bin.num_ctas,
            bin.clusterDims[0],
            bin.clusterDims[1],
            bin.clusterDims[2],
            bin.shared,
            stream,
            bin.cu_function,
            CompiledKernel.launch_enter_hook,
            CompiledKernel.launch_exit_hook,
            bin,
            *args,
        )
        return

    # print(f'*{output.dtype}, *{query.dtype}, *{key_cache.dtype}, *{value_cache.dtype}, '
    #       f'*{head_mapping.dtype}, *{block_tables.dtype}, *{context_lens.dtype}, *{medusa_attn_mask.dtype}, '
    #       f'{str(type(scale))}, {str(type(num_candidates))}, {",".join([str(type(output.stride(0)))] * 4)}, '
    #       f'{",".join([str(type(query.stride(0)))] * 4)}, '
    #       f'{",".join([str(type(key_cache.stride(0)))] * 5)}, '
    #       f'{",".join([str(type(value_cache.stride(0)))] * 4)}, '
    #       f'{",".join([str(type(block_tables.stride(0)))] * 2)}, '
    #       f'{",".join([str(type(medusa_attn_mask.stride(0)))] * 2)}, '
    #       f'*{key_block_idx.dtype}, *{key_head_idx.dtype}, *{value_block_idx.dtype}, '
    #       f'{BLOCK_M}, {head_dim}, {BLOCK_N}, {block_size}, {BLOCK_N // block_size}, {head_dim // X}, {X}'
    #       )
    num_warps = 4 if head_dim <= 64 else 8
    cached_bin = _fwd_kernel[grid](
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        medusa_attn_mask,
        ################### DEBUG ONLY #########################
        # q, k,
        # qk,
        # p,
        # v,
        # am,
        # q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        # qk.stride(0), qk.stride(1),
        # p.stride(0), p.stride(1),
        # v.stride(0), v.stride(1),
        # am.stride(0), am.stride(1),
        ################### DEBUG ONLY #########################
        scale,
        num_candidates,
        output.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        query.stride(0),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_cache.stride(4),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        medusa_attn_mask.stride(2),
        medusa_attn_mask.stride(3),
        key_block_idx,
        key_head_idx,
        value_block_idx,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=block_size,
        PAGES_PER_BLOCK_N=BLOCK_N // block_size,  # asserted divisible
        BLOCK_DKEYCACHE=head_dim // X,  # asserted divisible
        X=X,
        num_warps=num_warps,
        num_stages=1,
    )

    ################### DEBUG ONLY #########################
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(precision=10)
    # print(f'after attn : q: {q}, shape: {q.shape}')
    # print(f'after attn : k: {k}, shape: {k.shape}')
    # print(f'after attn : qk: {qk}, shape: {qk.shape}')
    # print(f'after attn : p: {p}, shape: {p.shape}')
    # print(f'after attn : v: {v}, shape: {v.shape}')
    # print(f'after attn : medusa mask : {medusa_attn_mask}, shape: {medusa_attn_mask.shape}')
    # print(f'after attn : attn mask : {am}, shape: {am.shape}')
    # torch.set_printoptions(profile="default") # reset
    ################### DEBUG ONLY #########################

    # with nvtx.annotate("final output change", color="yellow"):
    #     output = output.squeeze(0)

    return


def generate_medusa_attn_mask(medusa_choices, device="cuda"):
    """
    Generate buffers related to the Medusa structure.

    This function generates various buffers used in the Medusa structure, which is a complex data structure consisting of a tree-like hierarchy. The buffers include indices, attention masks, position identifiers, and more.

    Args:
        medusa_choices (torch.Tensor): A tensor containing choices for the Medusa structure.
        context_len: int for context lengths
        dtype: data type of the mask tensor
        device (str, optional): Target device for the generated buffers. Defaults to "cuda".

    Returns:
        The attention mask designed specifically for the Medusa structure, ensuring proper attention computation.
    """
    medusa_choices = torch.tensor(medusa_choices)
    cumulative_product = torch.cumprod(medusa_choices, dim=0)
    medusa_len = cumulative_product.sum().item()
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)

    # 2. Update the Medusa attention mask
    prev_cumprod_sum = -1
    for i in range(medusa_choices.size(0)):
        cumprod_sum = cumulative_product[:i].sum().item()
        if prev_cumprod_sum != -1:
            parent_idx = (
                torch.arange(prev_cumprod_sum, cumprod_sum)
                .repeat(medusa_choices[i], 1)
                .transpose(0, 1)
                .flatten()
            )
            medusa_attn_mask[
                cumprod_sum : cumprod_sum + parent_idx.size(0)
            ] += medusa_attn_mask[parent_idx]
        prev_cumprod_sum = cumulative_product[:i].sum().item()
    return medusa_attn_mask.to(device)


import itertools
import operator
import random
from collections import OrderedDict
from dataclasses import dataclass

import triton.language as tl


@dataclass
class DataClassDict:
    def __post_init__(self):
        self.update(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__.values())


@dataclass(kw_only=True)
class PagedFlashAttentionStrideArgs(DataClassDict):
    stride_ob: int
    stride_os: int
    stride_oh: int
    stride_od: int
    stride_qb: int
    stride_qs: int
    stride_qh: int
    stride_qd: int
    stride_kb: int
    stride_kh: int
    stride_kxd: int
    stride_kbs: int
    stride_kx: int
    stride_vb: int
    stride_vh: int
    stride_vd: int
    stride_vbs: int
    stride_bts: int
    stride_btb: int
    stride_mm_row: int
    stride_mm_col: int


@dataclass(kw_only=True)
class PagedFlashAttentionConstants(DataClassDict):
    BLOCK_M: tl.constexpr
    BLOCK_DMODEL: tl.constexpr
    BLOCK_N: tl.constexpr
    BLOCK_SIZE: tl.constexpr
    PAGES_PER_BLOCK_N: tl.constexpr
    BLOCK_DKEYCACHE: tl.constexpr
    X: tl.constexpr


@dataclass(kw_only=True)
class PagedFlashAttentionCacheArgs(DataClassDict):
    key_cache: torch.Tensor  # [num_blocks, num_heads, head_size // x, block_size, x]
    value_cache: torch.Tensor  # [num_blocks, num_heads, head_size, block_size]
    head_mapping: torch.Tensor  # [num_heads]
    block_tables: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor  # [num_seqs]


@dataclass(kw_only=True)
class PagedFlashAttentionIndexArgs(DataClassDict):
    key_block_idx: torch.Tensor
    key_head_idx: torch.Tensor
    value_block_idx: torch.Tensor


@dataclass(kw_only=True)
class PagedFlashAttentionInputOutputArgs(DataClassDict):
    output: torch.Tensor
    query: torch.Tensor  # [num_seqs, num_candidates, num_heads, head_dim]


@dataclass(kw_only=True)
class PagedFlashAttentionMedusaArgs(DataClassDict):
    medusa_attn_mask: torch.Tensor
    scale: float
    num_candidates: int


@dataclass(kw_only=True)
class PagedFlashAttentionArgs(DataClassDict):
    input_output_args: PagedFlashAttentionInputOutputArgs
    cache_args: PagedFlashAttentionCacheArgs
    medusa_args: PagedFlashAttentionMedusaArgs
    stride_args: PagedFlashAttentionStrideArgs
    index_args: PagedFlashAttentionIndexArgs
    constants: PagedFlashAttentionConstants

    def __post_init__(self):
        super().__post_init__(self)

        full_args = OrderedDict(
            **self.input_output_args,
            **self.cache_args,
            **self.medusa_args,
            **self.stride_args,
            **self.index_args,
            **self.constants,
        )
        self.__dict__.update(full_args)

        # self.__dict__.pop("cache_args")
        # self.__dict__.pop("stride_args")
        # self.__dict__.pop("constants")


@dataclass
class SingleQueryArgs(DataClassDict):
    num_sequences: int
    num_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    dtype: torch.dtype
    num_kv_heads: int = None
    medusa_choices: List[int] = None


SINGLE_QUERY_CONFIG = [SingleQueryArgs(7, 40, 16, 128, 10240, torch.float16, [1, 3, 4])]


def construct_paged_fa_args(
    num_sequences,
    num_heads,
    head_size,
    block_size,
    num_blocks,
    dtype,
    num_kv_heads,
    medusa_choices,
    MAX_SEQ_LEN,
    BLOCK_M=16,
    BLOCK_N=128,
):
    medusa_candidates = sum(itertools.accumulate(medusa_choices, operator.mul))
    medusa_attn_mask = generate_medusa_attn_mask(medusa_choices, device="cuda")

    # Input / Output Args
    qkv = torch.empty(
        num_sequences,
        medusa_candidates,
        3,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    qkv.uniform_(-1e-3, 1e-3)

    # query shape: [num_sequences, medusa_candidates, num_heads, head_size]
    query, _, _ = qkv.unbind(dim=2)

    output = torch.empty(
        num_sequences,
        medusa_candidates,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )

    input_output_args = PagedFlashAttentionInputOutputArgs(
        output=output,
        query=query,
    )

    # Cache Args
    X = 16 // torch.tensor([], dtype=query.dtype).element_size()

    key_block_shape = (
        num_heads,
        head_size // X,
        block_size,
        X,
    )

    key_cache = torch.empty(
        size=(num_blocks, *key_block_shape),
        dtype=dtype,
        device="cuda",
    )
    key_cache.uniform_(-1e-3, 1e-3)

    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(
        size=(num_blocks, *value_block_shape), dtype=dtype, device="cuda"
    )
    value_cache.uniform_(-1e-3, 1e-3)

    context_lens = [
        max(random.randint(1, MAX_SEQ_LEN), medusa_candidates + 10)
        for _ in range(num_sequences)
    ]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    cache_args = PagedFlashAttentionCacheArgs(
        key_cache=key_cache,
        value_cache=value_cache,
        head_mapping=head_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
    )

    # Medusa Args
    scale = float(1.0 / (head_size**0.5))
    num_candidates, num_heads, head_dim = query.shape

    medusa_args = PagedFlashAttentionMedusaArgs(
        medusa_attn_mask=medusa_attn_mask,
        scale=scale,
        num_candidates=num_candidates,
    )

    # Stride Args
    # taking medusa candidates into account
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_sequences):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    assert num_heads % num_kv_heads == 0

    num_queries_per_kv = num_heads // num_kv_heads

    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )

    stride_args = PagedFlashAttentionStrideArgs(
        stride_ob=output.stride(0),
        stride_os=output.stride(0),
        stride_oh=output.stride(1),
        stride_od=output.stride(2),
        stride_qb=query.stride(0),
        stride_qs=query.stride(0),
        stride_qh=query.stride(1),
        stride_qd=query.stride(2),
        stride_kb=key_cache.stride(0),
        stride_kh=key_cache.stride(1),
        stride_kxd=key_cache.stride(2),
        stride_kbs=key_cache.stride(3),
        stride_kx=key_cache.stride(4),
        stride_vb=value_cache.stride(0),
        stride_vh=value_cache.stride(1),
        stride_vd=value_cache.stride(2),
        stride_vbs=value_cache.stride(3),
        stride_bts=block_tables.stride(0),
        stride_btb=block_tables.stride(1),
        stride_mm_row=medusa_attn_mask.stride(2),
        stride_mm_col=medusa_attn_mask.stride(3),
    )

    # Index Args
    key_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
    key_head_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
    value_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
    index_args = PagedFlashAttentionIndexArgs(
        key_block_idx=key_block_idx,
        key_head_idx=key_head_idx,
        value_block_idx=value_block_idx,
    )

    # Constants
    assert BLOCK_N % block_size == 0
    constant_args = PagedFlashAttentionConstants(
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=block_size,
        PAGES_PER_BLOCK_N=BLOCK_N // block_size,  # asserted divisible
        BLOCK_DKEYCACHE=head_dim // X,  # asserted divisible
        X=X,
    )

    return PagedFlashAttentionArgs(
        input_output_args=input_output_args,
        cache_args=cache_args,
        medusa_args=medusa_args,
        stride_args=stride_args,
        index_args=index_args,
        constants=constant_args,
    )


@pytest.mark.parametrize("query_config", SINGLE_QUERY_CONFIG)
@pytest.mark.parametrize("MAX_SEQ_LEN", [2048])
def test_paged_attention(query_config, MAX_SEQ_LEN) -> List[AOTCompilationResult]:
    kernel_path = "/notebooks/torch-extensions/extension_builder/triton-aot/python/test/unit/tools/fixtures/kernels/triton-paged-attention/paged_attention.py"
    trace_dir = Path(__file__).parent / kernel_path.name.split(".")[0]

    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    (
        num_sequences,
        num_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        num_kv_heads,
        medusa_choices,
    ) = query_config

    TEST_SEED = 0
    torch.manual_seed(TEST_SEED)
    random.seed(TEST_SEED)

    args = construct_paged_fa_args(
        num_sequences,
        num_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        num_kv_heads,
        medusa_choices,
        MAX_SEQ_LEN,
    )

    num_candidates, num_heads, head_dim = args.query.shape
    batch = 1
    head_dim = args.query.shape[-1]

    grid = (batch, num_heads, triton.cdiv(num_candidates, args.BLOCK_M))
    num_warps = 4 if head_dim <= 64 else 8

    # trace_config = TraceConfig(
    #     trace_dir=trace_dir,
    # )
    # jit_fn = JITFunction(kernel_path)
    # kernel_name = jit_fn.__name__
    # results = jit_fn
    # compiler = AOTCompiler(
    #     kernel_name=kernel_name,
    #     jit_args=args,
    #     jit_fn=jit_fn,
    #     save_dir=codegen_dir,
    # )
    # compiled_result: AOTCompilationResult = compiler.generate()
    # compiled_result: AOTCompilationResult = compiler.generate()
