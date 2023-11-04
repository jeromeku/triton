import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from pytest import MonkeyPatch

import triton
from triton.common import cuda_include_dir, libcuda_dirs
from triton.runtime.jit import JITFunction
from triton.tools.aot import KernelLinkerMeta


@pytest.fixture(scope="session")
def clear_triton_cache():
    from triton.runtime.cache import default_cache_dir, default_dump_dir

    cache_dir = default_cache_dir()
    dump_dir = default_dump_dir()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)

    yield


@pytest.fixture
def aot_kernel_dir():
    test_dir = Path("aot_test_kernels").absolute()

    if os.path.exists(test_dir):
        import shutil

        shutil.rmtree(test_dir)

    os.makedirs(test_dir)

    yield test_dir


#    shutil.rmtree(test_dir)


@pytest.fixture
def headers():
    headers_path = (Path(__file__).parent / "fixtures" / "compiler").absolute()
    return headers_path.glob("*.h")


@pytest.fixture
def linker_test_dir():
    test_dir = (Path(__file__).parent / "linker_test").absolute()

    if os.path.exists(test_dir):
        import shutil

        shutil.rmtree(test_dir)

    os.makedirs(test_dir)

    yield test_dir


@pytest.fixture
def reference_header():
    header = """
#include <cuda.h>

CUresult add_kernel_1024_warps4xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
void load_add_kernel_1024_warps4xstages3();
void unload_add_kernel_1024_warps4xstages3();
    
int add_kernel_get_num_algos(void);

CUresult add_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id);
void load_add_kernel();
void unload_add_kernel();
"""
    return header.strip()


@pytest.fixture
def reference_algo_decl():
    header = """
CUresult add_kernel_1024_warps4xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
void load_add_kernel_1024_warps4xstages3();
void unload_add_kernel_1024_warps4xstages3();
"""
    return header.strip()


@pytest.fixture
def reference_header_get_num_algo_decl():
    src = """
int add_kernel_get_num_algos(void);
"""
    return src.strip()


@pytest.fixture
def reference_header_kernel_default_load():
    src = """
CUresult add_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id);
void load_add_kernel();
void unload_add_kernel();    
"""
    return src.strip()


@pytest.fixture
def reference_dispatcher_defs():
    defs = """
// launcher for: add_kernel_1024_warps4xstages3
CUresult add_kernel_8d4b99fa_0d1d2d3de(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);

CUresult add_kernel_1024_warps4xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements){
  if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (output_ptr % 16 == 0) && (n_elements % 16 == 0))
    return add_kernel_8d4b99fa_0d1d2d3de(stream, x_ptr, y_ptr, output_ptr, n_elements);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: add_kernel_1024_warps4xstages3
void load_add_kernel_8d4b99fa_0d1d2d3de();
void load_add_kernel_1024_warps4xstages3() {
  load_add_kernel_8d4b99fa_0d1d2d3de();
}

// unload for: add_kernel_1024_warps4xstages3
void unload_add_kernel_8d4b99fa_0d1d2d3de();
void unload_add_kernel_1024_warps4xstages3() {
  unload_add_kernel_8d4b99fa_0d1d2d3de();
}
"""
    return defs.strip()


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


@pytest.fixture
def parsed_kernel_metas(headers):
    from triton.tools.aot.parsers import HeaderParser

    parser = HeaderParser()
    kernels = parser.parse(headers)
    return kernels


def test_aot_linker_algo_decl_codegen(
    parsed_kernel_metas: Dict[str, List[KernelLinkerMeta]], reference_algo_decl
):
    from triton.tools.aot import HeaderGenerator

    header_gen = HeaderGenerator(kernels=parsed_kernel_metas)
    actual_decl = header_gen.make_algo_decls()
    print("actual_decl:\n", actual_decl)
    print("\n")
    print("reference_header_kernel_decl:\n", reference_algo_decl)
    check_codegen(actual_decl, reference_algo_decl)


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


def test_aot_linker_source_gen_dispatcher_defs(
    headers, linker_test_dir, reference_dispatcher_defs
):
    from triton.tools.aot import link
    from triton.tools.aot.codegen import SourceGenerator

    out_path = linker_test_dir / "kernel"
    linker = link.Linker(headers, out_path=out_path.absolute())
    kernels = linker.parse_headers()
    header_file, meta = linker.generate_headers(kernels)
    src_gen = SourceGenerator(kernels=kernels, meta=meta)
    defs = src_gen.make_defs()

    check_codegen(actual=defs, expected=reference_dispatcher_defs)
