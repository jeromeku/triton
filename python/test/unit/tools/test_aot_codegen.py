import os
from pathlib import Path

import pytest
import torch

import triton
from triton.runtime.jit import JITFunction
from triton.tools.aot import AOT_C_CUDA_ParamsBuilder, HeaderGenerator, SourceGenerator
from triton.tools.jitted_aot import CompiledArtifact

# TODO: Create test cases for matmul; replicate existing aot tests
# Create matmul kernel fixture
# Create kernel headers / sources and linker headers and sources using AOT scripts
# Compare to generated headers and sources from refactored codegen classes
# Rename AOT_JIT to AOT_Trace
# Migrate test_jit_aot tests to test_aot_trace


# TODO: refactor jit.py to use refactored codegen classes
def _preprocess_src(src):
    return list(filter(lambda x: x.strip(), src.split("\n")))


def check_codegen(actual: str, expected: str):
    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)

    for actual, expected in zip(actual_lines, expected_lines):
        assert actual == expected, f"Expected: \n{expected}\nActual: \n{actual}"


def check_dir(dir):
    if os.path.exists(dir):
        import shutil

        shutil.rmtree(dir)

    os.makedirs(dir)
    return dir


@pytest.mark.parametrize("kernel_file", ["add_kernel.py"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16],
    ids=lambda x: str(x),
)
def test_aot_compiler_params(kernel_path, dtype, kernel_file):
    N = 1024
    BLOCK_SIZE = 1024
    NUM_WARPS = 4
    seed = 0

    torch.manual_seed(seed)
    x = torch.ones(N, dtype=dtype, device="cuda")  # torch.rand(size, device="cuda")
    y = torch.ones(N, dtype=dtype, device="cuda")

    test_kernel = kernel_path / kernel_file

    # Set up aot kernel directory
    test_dir = Path("aot_compilation_spec_test").absolute()
    check_dir(test_dir)
    from triton.runtime.jit import JITFunction

    test_fn = JITFunction(test_kernel)
    kernel_name = test_fn.__name__
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    # Run aot jit
    compilation_artifact: CompiledArtifact = test_fn[grid](
        x,
        y,
        output,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        trace=True,
        trace_dir=test_dir,
    )

    expected_spec = compilation_artifact.compiler_spec
    compiler = AOT_C_CUDA_ParamsBuilder(
        kernel_name,
        compiled_binary=compilation_artifact.compiled_binary,
        jit_args=compilation_artifact.jit_args,
        jit_fn=test_fn,
    )
    actual_spec = compiler.build()

    expected_keys = set(expected_spec.keys())
    actual_keys = set(actual_spec.keys())
    assert (
        expected_keys == actual_keys
    ), f"Expected: {expected_keys}\nActual: {actual_keys}"

    for k in expected_keys:
        assert (
            expected_spec[k] == actual_spec[k]
        ), f"Key: {k}\nExpected: {expected_spec[k]}\nActual: {actual_spec[k]}"


@pytest.mark.parametrize("kernel_file", ["add_kernel.py"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16],
    ids=lambda x: str(x),
)
def test_aot_compiler_codegen(
    dtype,
    kernel_path,
    kernel_file,
    # reference_compiler_header,
    # reference_compiler_source,
):
    from triton.tools.aot import AOT_C_CUDA_Compiler

    # Test params
    N = 1024
    BLOCK_SIZE = 1024
    NUM_WARPS = 4
    seed = 0

    torch.manual_seed(seed)
    x = torch.ones(N, dtype=dtype, device="cuda")  # torch.rand(size, device="cuda")
    y = torch.ones(N, dtype=dtype, device="cuda")

    test_kernel = kernel_path / kernel_file

    # Set up aot kernel directory
    test_dir = Path("aot_compilation_codegen_test_f16").absolute()
    check_dir(test_dir)

    test_fn = JITFunction(test_kernel)
    kernel_name = test_fn.__name__
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    # Run aot jit
    compilation_artifact: CompiledArtifact = test_fn[grid](
        x,
        y,
        output,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        trace=True,
        trace_dir=test_dir,
    )
    # compiler = AOT_C_CUDA_Compiler(
    #     kernel_name=kernel_name,
    #     compiled_binary=compilation_artifact.compiled_binary,
    #     jit_args=compilation_artifact.jit_args,
    #     jit_fn=test_fn,
    # )

    # header = compiler.generate_header()

    # check_codegen(header, reference_compiler_header)

    # source = compiler.generate_source()
    # with open(test_dir / "generated_source.cu", "w") as fp:
    #     fp.write(source)
    # check_codegen(source, reference_compiler_source)


def test_aot_header_parser(headers):
    from triton.tools.aot.parsers import HeaderParser

    parser = HeaderParser()
    kernels = parser.parse(headers)
    assert len(kernels) == 1
    # TODO: Add more tests


@pytest.fixture(params=["add_kernel.8d4b99fa_0d1d2d3de.h"])
def parsed_C_CUDA_kernel_metas(headers_path, request):
    from triton.tools.aot.parsers import C_CUDA_HeaderParser

    header_file = headers_path / request.param

    parser = C_CUDA_HeaderParser()
    kernels = parser.parse([header_file])
    return kernels


@pytest.fixture
def C_CUDA_header_generator(parsed_C_CUDA_kernel_metas):
    from triton.tools.aot import C_CUDA_HeaderGenerator

    generator = C_CUDA_HeaderGenerator(parsed_C_CUDA_kernel_metas)
    return generator


@pytest.fixture
def C_CUDA_header_generator(parsed_C_CUDA_kernel_metas):
    from triton.tools.aot import C_CUDA_SourceGenerator

    generator = C_CUDA_SourceGenerator(parsed_C_CUDA_kernel_metas)
    return generator


def test_aot_linker_algo_decl(
    C_CUDA_header_generator: HeaderGenerator, reference_algo_decl
):
    actual_decl = C_CUDA_header_generator._make_algo_decls()
    check_codegen(actual_decl, reference_algo_decl)


def test_aot_linker_algo_get_num_algo_decl(
    C_CUDA_header_generator: HeaderGenerator, reference_get_num_algo_decl
):
    actual_decl = C_CUDA_header_generator._make_get_num_algos_decl()
    check_codegen(actual_decl, reference_get_num_algo_decl)


def test_aot_linker_global_decl(
    C_CUDA_header_generator: HeaderGenerator, reference_global_decl
):
    actual_decl = C_CUDA_header_generator._make_global_decl()
    check_codegen(actual_decl, reference_global_decl)


def test_aot_linker_header_codegen(
    C_CUDA_header_generator: HeaderGenerator, reference_header
):
    actual_header = C_CUDA_header_generator.generate()
    check_codegen(actual_header, reference_header)


def test_aot_linker_func_pointer_defs(
    C_CUDA_source_generator: SourceGenerator, reference_func_pointer_defs
):
    defs = C_CUDA_source_generator._make_func_pointers()
    check_codegen(actual=defs, expected=reference_func_pointer_defs)


def test_aot_linker_const_dispatcher_defs(
    C_CUDA_source_generator: SourceGenerator,
    reference_const_dispatcher_defs,
):
    defs = C_CUDA_source_generator._make_kernel_meta_const_dispatcher()
    check_codegen(actual=defs, expected=reference_const_dispatcher_defs)


def test_aot_linker_source_gen_dispatcher_defs(
    C_CUDA_source_generator: SourceGenerator, reference_dispatcher_defs
):
    defs = C_CUDA_source_generator._make_defs()

    check_codegen(actual=defs, expected=reference_dispatcher_defs)


def test_aot_linker_load_defs(
    C_CUDA_source_generator: SourceGenerator, reference_load_defs
):
    actual_defs = C_CUDA_source_generator._make_kernel_load_defs()
    check_codegen(actual_defs, reference_load_defs)


def test_aot_linker_get_num_algos_def(
    C_CUDA_source_generator: SourceGenerator, reference_get_num_algos_def
):
    actual_defs = C_CUDA_source_generator._make_get_num_algos_def()
    check_codegen(actual_defs, reference_get_num_algos_def)


def test_aot_linker_default_algo_def(
    C_CUDA_source_generator: SourceGenerator, reference_default_algo_def
):
    actual_def = C_CUDA_source_generator._make_default_algo_kernel_def()
    check_codegen(actual_def, reference_default_algo_def)


# @pytest.parametrize("headers", ["add_kernel.8d4b99fa_0d1d2d3de.h"])
# def test_aot_linker_codegen(headers, reference_header, reference_source):
#     from triton.tools.aot import AOT_C_CUDA_Linker

#     linker = AOT_C_CUDA_Linker(headers)
#     result = linker.generate()
#     check_codegen(result.header, reference_header)
#     check_codegen(result.source, reference_source)
