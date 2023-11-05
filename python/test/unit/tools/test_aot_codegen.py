import json
import os
from pathlib import Path

import pytest
import torch

import triton
from triton.tools.aot import (
    AOTCompilerParamsBuilder,
    HeaderGenerator,
    Linker,
    SourceGenerator,
)
from triton.tools.jitted_aot import CompiledArtifact


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


@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("test_data_fn", ["ones"])
def test_aot_add_kernel_compiler_params(dtype, test_data_fn):
    if test_data_fn == "randn" and "int" in str(dtype):
        pytest.skip("randn not supported for int types")
    # Test params
    N = 1024
    BLOCK_SIZE = 1024
    NUM_WARPS = 4
    seed = 0
    data_generator = getattr(torch, test_data_fn)

    torch.manual_seed(seed)
    x = data_generator(N, dtype=dtype, device="cuda")  # torch.rand(size, device="cuda")
    y = data_generator(N, dtype=dtype, device="cuda")

    test_kernel = Path(__file__).parent / "fixtures" / "vector_add_kernel.py"

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
    compiler = AOTCompilerParamsBuilder(
        kernel_name,
        compiled_binary=compilation_artifact.compiled_binary,
        jit_args=compilation_artifact.jit_args,
        jit_fn=test_fn,
    )
    actual_spec = compiler.build()

    with open(test_dir / "expected_spec.json", "w") as f:
        json.dump(expected_spec, f)

    with open(test_dir / "actual_spec.json", "w") as f:
        json.dump(actual_spec, f)

    expected_keys = set(expected_spec.keys())
    actual_keys = set(actual_spec.keys())
    assert (
        expected_keys == actual_keys
    ), f"Expected: {expected_keys}\nActual: {actual_keys}"

    for k in expected_keys:
        assert (
            expected_spec[k] == actual_spec[k]
        ), f"Key: {k}\nExpected: {expected_spec[k]}\nActual: {actual_spec[k]}"


def test_aot_compiler_params():
    pass


def test_aot_header_parser(headers):
    from triton.tools.aot.parsers import HeaderParser

    parser = HeaderParser()
    kernels = parser.parse(headers)
    assert len(kernels) == 1
    # TODO: Add more tests


def test_aot_linker_algo_decl(header_generator: HeaderGenerator, reference_algo_decl):
    actual_decl = header_generator.make_algo_decls()
    check_codegen(actual_decl, reference_algo_decl)


def test_aot_linker_algo_get_num_algo_decl(
    header_generator: HeaderGenerator, reference_get_num_algo_decl
):
    actual_decl = header_generator.make_get_num_algos_decl()
    check_codegen(actual_decl, reference_get_num_algo_decl)


def test_aot_linker_global_decl(
    header_generator: HeaderGenerator, reference_global_decl
):
    actual_decl = header_generator.make_global_decl()
    check_codegen(actual_decl, reference_global_decl)


def test_aot_linker_header_codegen(header_generator: HeaderGenerator, reference_header):
    actual_header = header_generator.generate()
    print(f"actual:\n{actual_header}")
    print(f"reference:\n{reference_header}")
    check_codegen(actual_header, reference_header)


def test_aot_linker_header_gen(headers, linker_test_dir, reference_header):
    from triton.tools.aot import link

    out_path = linker_test_dir / "kernel"
    linker = link.Linker(headers, out_path=out_path.absolute())
    kernels = linker.parse_headers()
    header_file, meta = linker.generate_headers(kernels)

    assert os.path.exists(header_file)

    with open(header_file, "r") as f:
        actual_lines = [line.strip() for line in f]

    expected_lines = [line.strip() for line in reference_header.split("\n")]

    for expected, actual in zip(expected_lines, actual_lines):
        assert expected == actual


def test_aot_linker_func_pointer_defs(
    source_generator: SourceGenerator, reference_func_pointer_defs
):
    defs = source_generator.make_func_pointers()
    print(f"defs:\n{defs}")
    print(f"reference:\n{reference_func_pointer_defs}")
    check_codegen(actual=defs, expected=reference_func_pointer_defs)


def test_aot_linker_const_dispatcher_defs(
    source_generator: SourceGenerator,
    reference_const_dispatcher_defs,
):
    defs = source_generator.make_kernel_meta_const_dispatcher()
    print(f"defs:\n{defs}")
    print(f"reference:\n{reference_const_dispatcher_defs}")
    check_codegen(actual=defs, expected=reference_const_dispatcher_defs)


def test_aot_linker_source_gen_dispatcher_defs(
    source_generator: SourceGenerator, reference_dispatcher_defs
):
    defs = source_generator.make_defs()

    check_codegen(actual=defs, expected=reference_dispatcher_defs)


def test_aot_linker_load_defs(source_generator: SourceGenerator, reference_load_defs):
    actual_defs = source_generator.make_kernel_load_defs()
    check_codegen(actual_defs, reference_load_defs)


def test_aot_linker_get_num_algos_def(
    source_generator: SourceGenerator, reference_get_num_algos_def
):
    actual_defs = source_generator.make_get_num_algos_def()
    check_codegen(actual_defs, reference_get_num_algos_def)


def test_aot_linker_default_algo_def(
    source_generator: SourceGenerator, reference_default_algo_def
):
    actual_def = source_generator.make_default_algo_kernel_def()
    check_codegen(actual_def, reference_default_algo_def)


def test_aot_linker_source_codegen(source_generator: SourceGenerator, reference_source):
    actual_source = source_generator.generate()
    print(f"actual:\n{actual_source}")
    print(f"reference:\n{reference_source}")
    check_codegen(actual_source, reference_source)


def test_aot_linker_codegen(headers, reference_header, reference_source):
    linker = Linker(headers)
    result = linker.generate()
    check_codegen(result.header, reference_header)
    check_codegen(result.source, reference_source)
