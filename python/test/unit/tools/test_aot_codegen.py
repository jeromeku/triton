import os
from typing import Dict, List

from triton.tools.aot import KernelLinkerMeta
from triton.tools.aot.codegen import HeaderGenerator, SourceGenerator


def _preprocess_src(src):
    return list(filter(lambda x: x.strip(), src.split("\n")))


def check_codegen(actual: str, expected: str):
    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)

    for actual, expected in zip(actual_lines, expected_lines):
        assert actual == expected, f"Expected: \n{expected}\nActual: \n{actual}"


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
