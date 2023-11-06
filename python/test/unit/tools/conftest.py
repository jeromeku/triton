import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import triton

# @pytest.fixture(scope="session")
# def clear_triton_cache():
#     from triton.runtime.cache import default_cache_dir, default_dump_dir

#     cache_dir = default_cache_dir()
#     dump_dir = default_dump_dir()

#     if os.path.exists(cache_dir):
#         shutil.rmtree(cache_dir)

#     if os.path.exists(dump_dir):
#         shutil.rmtree(dump_dir)

#     yield


# @pytest.fixture
# def aot_kernel_dir():
#     test_dir = Path("aot_test_kernels").absolute()

#     if os.path.exists(test_dir):
#         import shutil

#         shutil.rmtree(test_dir)

#     os.makedirs(test_dir)

#     yield test_dir


#    shutil.rmtree(test_dir)


# @pytest.fixture
# def linker_test_dir():
#     test_dir = (Path(__file__).parent / "linker_test").absolute()

#     if os.path.exists(test_dir):
#         import shutil

#         shutil.rmtree(test_dir)

#     os.makedirs(test_dir)

#     yield test_dir


@pytest.fixture
def fixture_path(request):
    return (Path(__file__).parent / "fixtures").absolute()


@pytest.fixture
def kernel_path(fixture_path):
    return fixture_path / "kernels"


@pytest.fixture
def headers_path(fixture_path):
    return fixture_path / "headers"


@pytest.fixture
def sources_path(fixture_path):
    return fixture_path / "sources"


@pytest.fixture
def reference_compiler_params(fixture_path):
    with open(fixture_path / "compiler_params.json", "r") as fp:
        return json.load(fp)


@pytest.fixture
def headers(headers_path):
    return list(headers_path.glob("*.h"))


# --- Reference codegen fixtures ---#

REFERENCE_HEADERS = {}

REFERENCE_SOURCES = {}

REFERENCE_DECL = {
    "add_kernel": {
        "8d4b99fa_0d1d2d3de": {
            "algo_decl": """
CUresult add_kernel_1024_warps4xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
void load_add_kernel_1024_warps4xstages3();
void unload_add_kernel_1024_warps4xstages3();
""",
            "get_num_algo_decl": """
int add_kernel_get_num_algos(void);
""",
            "global_decl": """
CUresult add_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id);
void load_add_kernel();
void unload_add_kernel();
""",
        }
    }
}


# --- Utilities for generating reference AOT kernels ---#
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


# def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
#     # compile all desired configs
#     for ha in ha_hb_hints:
#         for hb in ha_hb_hints:
#             sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
#             name = f"matmul_{dtype}"
#             grid = f"M/{BM}, N/{BN}, 1"
#             _compile_kernel(
#                 dir=dir,
#                 signature=sig,
#                 kernel_name="kernel",
#                 out_name=name,
#                 out_path=name,
#                 num_warps=1,
#                 grid=grid,
#                 kernel_path=kernel_path,
#             )


def link_aot_kernels(kernel_header_path, kernel_name):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(kernel_header_path, "*.h"))
    subprocess.run(
        [sys.executable, linker_path] + h_files + ["-o", kernel_name],
        check=True,
        cwd=kernel_header_path,
    )


def generate_reference_specs():
    """Generate reference specs for the aot kernels using existing AOT tools"""
    pass


# REFERENCE_DEFS = \
#   dict(\
#     add_kernel = {
#         8d4b99fa_0d1d2d3de = dict(
#             dispatcher_defs =  """
# // launcher for: add_kernel_1024_warps4xstages3
# CUresult add_kernel_8d4b99fa_0d1d2d3de(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);

# CUresult add_kernel_1024_warps4xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements){
#   if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (output_ptr % 16 == 0) && (n_elements % 16 == 0))
#     return add_kernel_8d4b99fa_0d1d2d3de(stream, x_ptr, y_ptr, output_ptr, n_elements);

#   return CUDA_ERROR_INVALID_VALUE;
# }

# // load for: add_kernel_1024_warps4xstages3
# void load_add_kernel_8d4b99fa_0d1d2d3de();
# void load_add_kernel_1024_warps4xstages3() {
#   load_add_kernel_8d4b99fa_0d1d2d3de();
# }

# // unload for: add_kernel_1024_warps4xstages3
# void unload_add_kernel_8d4b99fa_0d1d2d3de();
# void unload_add_kernel_1024_warps4xstages3() {
#   unload_add_kernel_8d4b99fa_0d1d2d3de();
# }
# """,
#             func_pointer_defs = """
# typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
# kernel_func_t add_kernel_kernels[] = {
#   add_kernel_1024_warps4xstages3,
# };
# """,
#             const_dispatcher_defs = """
# CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id){
#   assert (algo_id < (int)sizeof(add_kernel_kernels));
#   return add_kernel_kernels[algo_id](stream, x_ptr, y_ptr, output_ptr, n_elements);
# }
# """,
#             load_defs = """
# void load_add_kernel(void){
#   load_add_kernel_1024_warps4xstages3();
# }

# void unload_add_kernel(void){
#   unload_add_kernel_1024_warps4xstages3();
# }
# """,
#         )
#     )
# )


@pytest.fixture
def reference_compiler_header(headers, fixture_path: Path):
    if any("add_kernel" in str(h) for h in headers):
        return (fixture_path / "add_kernel.8d4b99fa_0d1d2d3de.h").read_text().strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


@pytest.fixture
def reference_compiler_source(headers, fixture_path: Path):
    if any("add_kernel" in str(h) for h in headers):
        return (fixture_path / "add_kernel.8d4b99fa_0d1d2d3de.c").read_text().strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_ALGO_DECL = """
CUresult add_kernel_1024_warps4xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
void load_add_kernel_1024_warps4xstages3();
void unload_add_kernel_1024_warps4xstages3();
"""


@pytest.fixture
def reference_algo_decl(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_ALGO_DECL.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_GET_NUM_ALGO_DECL = """
int add_kernel_get_num_algos(void);
"""


@pytest.fixture
def reference_get_num_algo_decl(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_GET_NUM_ALGO_DECL.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_GLOBAL_DECL = """
CUresult add_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id);
void load_add_kernel();
void unload_add_kernel();
"""


@pytest.fixture
def reference_global_decl(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_GLOBAL_DECL.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_HEADER = """
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


@pytest.fixture
def reference_header(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_HEADER.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_LOAD_DEFS = """
void load_add_kernel(void){
  load_add_kernel_1024_warps4xstages3();
}

void unload_add_kernel(void){
  unload_add_kernel_1024_warps4xstages3();
}
"""


@pytest.fixture
def reference_load_defs(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_LOAD_DEFS.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_GET_NUM_ALGOS_DEF = """
int add_kernel_get_num_algos(void){
  return (int)sizeof(add_kernel_kernels);
}
"""


@pytest.fixture
def reference_get_num_algos_def(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_GET_NUM_ALGOS_DEF.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_DEFAULT_ALGO_DEF = """
CUresult add_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements){
  return add_kernel(stream, x_ptr, y_ptr, output_ptr, n_elements, 0);
}
"""


@pytest.fixture
def reference_default_algo_def(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_DEFAULT_ALGO_DEF.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")


REFERENCE_ADD_KERNEL_SOURCE = """
#include <cuda.h>
#include <stdint.h>
#include <assert.h>

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

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
kernel_func_t add_kernel_kernels[] = {
  add_kernel_1024_warps4xstages3,
};

int add_kernel_get_num_algos(void){
  return (int)sizeof(add_kernel_kernels);
}

CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id){
  assert (algo_id < (int)sizeof(add_kernel_kernels));
  return add_kernel_kernels[algo_id](stream, x_ptr, y_ptr, output_ptr, n_elements);
}

void load_add_kernel(void){
  load_add_kernel_1024_warps4xstages3();
}

void unload_add_kernel(void){
  unload_add_kernel_1024_warps4xstages3();
}


CUresult add_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements){
  return add_kernel(stream, x_ptr, y_ptr, output_ptr, n_elements, 0);
}
"""


@pytest.fixture
def reference_source(headers):
    if any("add_kernel" in str(h) for h in headers):
        return REFERENCE_ADD_KERNEL_SOURCE.strip()
    else:
        raise ValueError(f"Unknown header: {list(headers)}")
