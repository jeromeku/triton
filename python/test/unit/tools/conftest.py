import os
import shutil
from pathlib import Path

import pytest


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
def parsed_kernel_metas(headers):
    from triton.tools.aot.parsers import HeaderParser

    parser = HeaderParser()
    kernels = parser.parse(headers)
    return kernels


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
def reference_get_num_algo_decl():
    src = """
int add_kernel_get_num_algos(void);
"""
    return src.strip()


@pytest.fixture
def reference_global_decl():
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


@pytest.fixture
def reference_func_pointer_defs():
    defs = """
typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
kernel_func_t add_kernel_kernels[] = {
  add_kernel_1024_warps4xstages3,
};
"""
    return defs.strip()


@pytest.fixture
def reference_const_dispatcher_defs():
    defs = """
CUresult add_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id){
  assert (algo_id < (int)sizeof(add_kernel_kernels));
  return add_kernel_kernels[algo_id](stream, x_ptr, y_ptr, output_ptr, n_elements);
}
"""
    return defs.strip()


@pytest.fixture
def header_generator(parsed_kernel_metas):
    from triton.tools.aot.codegen import HeaderGenerator

    return HeaderGenerator(kernels=parsed_kernel_metas)


@pytest.fixture
def source_generator(parsed_kernel_metas):
    from triton.tools.aot.codegen import SourceGenerator

    return SourceGenerator(kernels=parsed_kernel_metas)
