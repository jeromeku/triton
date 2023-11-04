import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pytest import MonkeyPatch

import triton
from triton.common import cuda_include_dir, libcuda_dirs
from triton.runtime.jit import JITFunction

add_kernel_src = """
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    N,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
"""


def gen_kernel_library(dir, libname):
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(
        ["gcc"] + c_files + ["-I", cuda_include_dir(), "-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))
    subprocess.run(
        ["gcc"] + o_files + ["-shared", "-o", libname, "-L", libcuda_dirs()[0]],
        check=True,
        cwd=dir,
    )


def gen_add_test_bin(
    dir,
    N,
    kernel_name,
    dtype_in,
    dtype_out,
    kernel_lib_name=None,
    exe="test",
    algo_id=0,
):
    if "16" in dtype_in:
        num_bytes_in = 2
        in_fmt_str = "%hd"
    elif "32" in dtype_in:
        num_bytes_in = 4
        in_fmt_str = "%d"

    if "16" in dtype_out:
        num_bytes_out = 2
        out_fmt_str = "%hd"

    elif "32" in dtype_out:
        num_bytes_out = 4
        out_fmt_str = "%d"
    print("num_bytes_in: ", num_bytes_in)
    print("in_fmt_str: ", in_fmt_str)
    headers = f"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "{kernel_name}.h"

"""

    utils_src = f"""
static void write_buffer_to_csv(char *filename, {dtype_out} *buffer, int size) {{
   FILE *file = fopen(filename, "w");
   if (file == NULL) {{
       printf("Could not open file %s\\n", filename);
       return;
   }}
   printf("Writing to %s\\n", filename);
   for (int i = 0; i < size; i++) {{
       fprintf(file, "{out_fmt_str}", buffer[i]);
       if (i < size - 1) {{
           fprintf(file, ",");
       }}
   }}
   fclose(file);
}}

static void read_csv_to_buffer(char *filename, {dtype_in} *buffer, int size) {{
   FILE *file = fopen(filename, "r");
   if (file == NULL) {{
       printf("Could not open file %s\\n", filename);
       return;
   }}
   int index = 0;
   printf("Reading from %s\\n", filename);
   while (fscanf(file, "{in_fmt_str},", &buffer[index]) != EOF && index < size) {{
       index++;
   }}
   fclose(file);
}}
"""
    test_utils_src = headers + utils_src
    test_src = f"""
int main(int argc, char **argv) {{
  int N = {N};

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr x, y, out;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&x, N * {num_bytes_in});
  cuMemAlloc(&y, N * {num_bytes_in});
  cuMemAlloc(&out, N * {num_bytes_out});
  cuStreamCreate(&stream, 0);
  load_add_kernel();

  // initialize input data
  {dtype_in} hx[N];
  {dtype_in} hy[N];
  memset(hx, 0, N * {num_bytes_in});
  memset(hy, 0, N * {num_bytes_in});
  read_csv_to_buffer(argv[1], hx, N);
  read_csv_to_buffer(argv[2], hy, N);
  cuMemcpyHtoD(x, hx, N * {num_bytes_in});
  cuMemcpyHtoD(y, hy, N * {num_bytes_in});

  // launch kernel
  cuStreamSynchronize(stream);
  CUresult ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = add_kernel_default(stream, x, y, out, N);
  }} else {{
    ret = add_kernel(stream, x, y, out, N, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  cuStreamSynchronize(stream);

  // read data
  {dtype_out} hout[N];
  memset(hout, 0, N * {num_bytes_out});
  cuMemcpyDtoH(hout, out, N * {num_bytes_out});    
  write_buffer_to_csv(argv[3], hout, N);

  // free cuda handles
  unload_add_kernel();
  cuMemFree(x);
  cuMemFree(y);
  cuMemFree(out);
  cuCtxDestroy(ctx);
}}
"""
    kernel_lib_name = kernel_lib_name or kernel_name
    src = test_utils_src + test_src
    with open(os.path.join(dir, f"{exe}.c"), "w") as file:
        file.write(src)
    subprocess.run(
        ["gcc"]
        + [
            f"{exe}.c",
            "-I",
            cuda_include_dir(),
            "-L",
            libcuda_dirs()[0],
            "-l",
            "cuda",
            "-L",
            dir,
            "-l",
            kernel_lib_name,
            "-o",
            exe,
        ],
        check=True,
        cwd=dir,
    )


def write_triton_kernels(dir, src, util_src):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(util_src)

    return kernel_path


def write_tt_kernel(dir, src, name):
    kernel_path = os.path.join(dir, name)
    with open(kernel_path, "w") as file:
        file.write(src)

    return Path(kernel_path).absolute()


def _find_kernel_name(kernel_path):
    import ast

    with open(kernel_path) as fp:
        tree = ast.parse(fp.read())
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert len(fns) == 1
    return fns[0].name


def _dtype_map(ty):
    return {np.int16: "i16", np.int32: "i32", np.float16: "fp16", np.float32: "fp32"}[
        ty
    ]


def compile_kernel_add(
    dir,
    kernel_path,
    N,
    dtype=np.float16,
    BLOCK_SIZE=1024,
    name=None,
    num_warps=1,
    specializations=None,
):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")
    # x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr

    sig = f"*{_dtype_map(dtype)}, *{_dtype_map(dtype)}, *{_dtype_map(dtype)}, i32, {BLOCK_SIZE}"
    print("SIGNATURE: ", sig)
    name = name or _find_kernel_name(kernel_path)
    grid = f"N/{BLOCK_SIZE}, 1, 1"
    num_warps = str(num_warps)
    subprocess.run(
        [
            sys.executable,
            compiler_path,
            "-n",
            name,
            "--signature",
            sig,
            "--out-name",
            name,
            "-o",
            name,
            "-w",
            num_warps,
            "-g",
            grid,
            kernel_path,
        ],
        check=True,
        cwd=dir,
    )


def link_aot_kernels(dir, out_name="kernel"):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run(
        [sys.executable, linker_path] + h_files + ["-o", out_name], check=True, cwd=dir
    )


def generate_test_data(dir, shape, file_name, dtype=np.float32, seed=0, ext="csv"):
    x = np.random.randn(np.prod(shape)).astype(dtype).reshape(shape)
    x_path = os.path.join(dir, f"{file_name}.{ext}")
    x.ravel().tofile(x_path, sep=",")
    return x, x_path


from enum import StrEnum


class DataGeneratorType(StrEnum):
    NUMPY = "numpy"
    TORCH = "torch"


def _generate_dummy_data(
    dir,
    shape,
    file_name,
    dtype,
    generator_type=DataGeneratorType.TORCH,
    gen_fn="ones",
    ext="csv",
):
    generator_ty = torch if generator_type == DataGeneratorType.TORCH else np
    assert isinstance(dtype, generator_ty.dtype)
    assert hasattr(generator_ty, gen_fn)

    generator = getattr(generator_ty, gen_fn)

    x = generator(shape, dtype=dtype)
    x_path = os.path.join(dir, f"{file_name}.{ext}")
    if dtype == generator_ty.float16:
        write_dtype = generator_ty.int16
    elif dtype == generator_ty.float32:
        write_dtype = generator_ty.int32
    else:
        write_dtype = dtype

    out_arr = x.view(write_dtype).reshape(-1)
    if DataGeneratorType.TORCH:
        out_arr = out_arr.cpu().numpy()
    out_arr.tofile(x_path, sep=",")

    return x, x_path


def normalize_dtype_to_np(ty) -> np.dtype:
    return getattr(np, str(ty).split(".")[-1])


def normalize_serialized_dtype(dtype):
    # read data and compare against reference
    if "float16" in str(dtype):
        conversion_type = np.int16
    elif "float32" in str(dtype):
        conversion_type = np.int32
    else:
        conversion_type = normalize_dtype_to_np(dtype)
    return conversion_type


def serialize_data_to_disk(x, save_dir, file_name, ext="csv"):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    conversion_type = normalize_serialized_dtype(x.dtype)

    save_path = os.path.join(save_dir, file_name + f".{ext}")
    x.view(conversion_type).reshape(-1).tofile(save_path, sep=",")

    return x, save_path


def generate_dummy_data(
    dir,
    shape,
    file_name,
    dtype,
    generator_type=DataGeneratorType.TORCH,
    gen_fn="ones",
    ext="csv",
):
    generator_ty = torch if generator_type == DataGeneratorType.TORCH else np
    assert isinstance(dtype, generator_ty.dtype)
    assert hasattr(generator_ty, gen_fn)

    generator = getattr(generator_ty, gen_fn)

    x = generator(shape, dtype=dtype)
    x_path = os.path.join(dir, f"{file_name}.{ext}")
    if dtype == generator_ty.float16:
        write_dtype = generator_ty.int16
    elif dtype == generator_ty.float32:
        write_dtype = generator_ty.int32
    else:
        write_dtype = dtype

    out_arr = x.view(write_dtype).reshape(-1)
    if DataGeneratorType.TORCH:
        out_arr = out_arr.cpu().numpy()
    out_arr.tofile(x_path, sep=",")

    return x, x_path


def check_dir(dir):
    if os.path.exists(dir):
        import shutil

        shutil.rmtree(dir)

    os.makedirs(dir)
    return dir


# Converts TT type to C-compatible type
# strictly for passing numpy / torch types to C
_TT_TO_C = {
    "i1": "int32_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp16": "int16_t",
    "bf16": "int16_t",
    "fp32": "int32_t",
    "f32": "int16_t",
    "fp64": "int64_t",
}


def ty_to_TT(arg):
    return JITFunction._type_of(JITFunction._key_of(arg))


def TT_to_C(ty_str):
    return _TT_TO_C[ty_str.replace("*", "")]


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


# TODO:
# Add test for multiple traces
# Refactor AOT tracing
# Automate python -> Rust compilation
# Test PTX NVRTC compilation
# Refactor link.py to library


def test_multiple_traces_add():
    pass


@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("test_data_fn", ["ones"])
def test_aot_cubin_add(dtype, test_data_fn, clear_triton_cache):
    if test_data_fn == "randn" and "int" in str(dtype):
        pytest.skip("randn not supported for int types")
    # Test params
    N = 1024
    BLOCK_SIZE = 1024
    NUM_WARPS = 4
    seed = 0
    data_generator = getattr(torch, test_data_fn)
    executable_name = "test"

    torch.manual_seed(seed)
    x = data_generator(N, dtype=dtype, device="cuda")  # torch.rand(size, device="cuda")
    y = data_generator(N, dtype=dtype, device="cuda")

    test_kernel = Path(__file__).parent / "fixtures" / "vector_add_kernel.py"

    # Set up aot kernel directory
    test_dir = Path("aot_compilation").absolute()
    check_dir(test_dir)

    with MonkeyPatch.context() as mp:
        mp.setenv("TRITON_AOT_KERNEL_DIR", str(test_dir))
        print("TRITON_AOT_KERNEL_DIR: ", os.environ["TRITON_AOT_KERNEL_DIR"])
        from triton.runtime.jit import JITFunction

        test_fn = JITFunction(test_kernel)
        kernel_name = test_fn.__name__
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        # Run aot jit
        bin = test_fn[grid](
            x,
            y,
            output,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
            compile_so=False,
        )

    with open(test_dir / f"{kernel_name}.cubin", "wb") as f:
        f.write(bin.asm["cubin"])

    with open(test_dir / f"{kernel_name}.ptx", "w") as f:
        f.write(bin.asm["ptx"])


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
    headers_path = (Path(__file__).parent / "fixtures" / "linker").absolute()
    return headers_path.glob("*.h")


@pytest.fixture
def linker_test_dir():
    test_dir = (Path(__file__).parent / "linker_test").absolute()

    if os.path.exists(test_dir):
        import shutil

        shutil.rmtree(test_dir)

    os.makedirs(test_dir)

    yield test_dir


def test_aot_linker(headers, linker_test_dir):
    from triton.tools.aot import link

    out_path = linker_test_dir / "kernel"
    linker = link.Linker(headers, out_path=out_path.absolute())
    kernels = linker.parse_headers()
    header_file = linker.generate_headers(kernels)

    assert os.path.exists(header_file)


@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("BLOCK_SIZE", [1024])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.int16, torch.int32],
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("test_data_fn", ["ones", "randn"])
def test_aot_jit_add(aot_kernel_dir, dtype, test_data_fn, N, BLOCK_SIZE):
    # Set up test
    if test_data_fn == "randn" and "int" in str(dtype):
        pytest.skip("randn not supported for int types")
    # Test params
    NUM_WARPS = 4
    seed = 0
    data_generator = getattr(torch, test_data_fn)
    executable_name = "test"

    torch.manual_seed(seed)
    x = data_generator(N, dtype=dtype, device="cuda")  # torch.rand(size, device="cuda")
    y = data_generator(N, dtype=dtype, device="cuda")

    test_kernel = Path(__file__).parent / "fixtures" / "vector_add_kernel.py"

    with MonkeyPatch.context() as mp:
        mp.setenv("TRITON_AOT_KERNEL_DIR", str(aot_kernel_dir))
        print("TRITON_AOT_KERNEL_DIR: ", os.environ["TRITON_AOT_KERNEL_DIR"])
        from triton.runtime.jit import JITFunction

        test_fn = JITFunction(test_kernel)
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        # Run aot jit
        compilation_artifact = test_fn[grid](
            x,
            y,
            output,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
            compile_so=True,
        )
    # DUMP MLIR_IR
    # Compilation pipeline: PTX and CUBIN generation
    kernel_binary = compilation_artifact.bin
    kernel_path = compilation_artifact.kernel_path

    assert os.path.exists(kernel_path)
    compilation_srcs = glob.glob(os.path.join(kernel_path, "*.c"))
    compilation_headers = glob.glob(os.path.join(kernel_path, "*.h"))

    assert len(compilation_srcs) == 2
    assert len(compilation_headers) == 2

    tt_type = ty_to_TT(x)
    dtype_in = dtype_out = TT_to_C(tt_type)

    # TODO: Add way to manually set kernel path
    kernel_name = test_fn.__name__

    gen_kernel_library(kernel_path, f"lib{kernel_name}.so")
    gen_add_test_bin(
        kernel_path, N, kernel_name, dtype_in, dtype_out, exe=executable_name
    )
    assert os.path.exists(os.path.join(kernel_path, f"{executable_name}"))

    data_dir = Path("test_data").absolute()
    check_dir(data_dir)

    x, x_path = serialize_data_to_disk(x, data_dir, "x")

    y, y_path = serialize_data_to_disk(y, data_dir, "y")
    out_path = os.path.join(data_dir, "out.csv")

    # run test case
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = kernel_path

    subprocess.run(
        [f"./{executable_name}", x_path, y_path, out_path],
        env=env,
        check=True,
        cwd=kernel_path,
    )
    assert os.path.exists(out_path)

    expected = x + y
    conversion_type = normalize_serialized_dtype(dtype)
    actual = np.genfromtxt(out_path, delimiter=",", dtype=conversion_type)
    actual = actual.reshape((N,)).view(normalize_dtype_to_np(dtype))

    assert np.allclose(actual, expected)


# def _test_compile_link_add():
#     from pathlib import Path

#     N = 1024
#     BLOCK_SIZE = 1024
#     NUM_WARPS = 4
#     kernel_dir = Path("aot_test_kernels").absolute()
#     check_dir(kernel_dir)
#     dtype = np.float32

#     kernel_path = write_tt_kernel(kernel_dir, add_kernel_src, "add_kernel.py")
#     kernel_name = _find_kernel_name(kernel_path)
#     compile_kernel_add(
#         kernel_dir,
#         kernel_path,
#         N,
#         name=kernel_name,
#         dtype=dtype,
#         BLOCK_SIZE=BLOCK_SIZE,
#         num_warps=NUM_WARPS,
#     )
#     link_aot_kernels(kernel_dir, out_name=kernel_name)

#     type_converter = {
#         "i1": "int32_t",
#         "i8": "int8_t",
#         "i16": "int16_t",
#         "i32": "int32_t",
#         "i64": "int64_t",
#         "u32": "uint32_t",
#         "u64": "uint64_t",
#         "fp16": "int16_t",
#         "bf16": "int16_t",
#         "fp32": "int32_t",
#         "f32": "int16_t",
#         "fp64": "int64_t",
#     }

#     dtype_in = dtype_out = type_converter[(_dtype_map(dtype))]
#     print("dtype_in: ", dtype_in)

#     executable_name = "test"
#     gen_kernel_library(kernel_dir, f"lib{kernel_name}.so")
#     gen_add_test_bin(
#         kernel_dir,
#         N,
#         dtype_in=dtype_in,
#         dtype_out=dtype_out,
#         kernel_name=kernel_name,
#         exe=executable_name,
#     )

#     # Generate test data
#     seed = 0
#     data_dir = Path("test_data").absolute()
#     check_dir(data_dir)

#     data_generator = generate_dummy_data
#     x, x_path = data_generator(data_dir, (N,), file_name="x", seed=seed, dtype=dtype)
#     y, y_path = data_generator(data_dir, (N,), file_name="y", seed=seed, dtype=dtype)
#     out_path = os.path.join(data_dir, "out.csv")
#     expected = x + y
#     # print(f"EXPECTED: {expected}")

#     # run test case
#     env = os.environ.copy()
#     env["LD_LIBRARY_PATH"] = kernel_dir

#     subprocess.run(
#         [f"./{executable_name}", x_path, y_path, out_path],
#         env=env,
#         check=True,
#         cwd=kernel_dir,
#     )

#     # read data and compare against reference
#     if dtype == np.float16:
#         conversion_type = np.int16
#     elif dtype == np.float32:
#         conversion_type = np.int32
#     else:
#         conversion_type = dtype

#     actual = np.genfromtxt(out_path, delimiter=",", dtype=conversion_type)
#     actual = actual.reshape((N,)).view(dtype)
#     EXPECTED_VAL = 2.0

#     def compute_stats(x):
#         actual_counts = np.isclose(x, EXPECTED_VAL).sum()
#         return actual_counts

#     print(f"ACTUAL counts: {compute_stats(actual)}")
#     print(f"Actual: {actual[:10]} {actual[-10:]}")
#     # c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
#     # np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)
