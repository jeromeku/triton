import glob
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

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


def generate_reference_specs():
    """Generate reference specs for the aot kernels using existing AOT tools"""
    pass


class AOTCompilerRunner:
    def __init__(self):
        self.compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    def compile_kernel(
        self,
        kernel_dir,
        signature,
        kernel_name,
        out_name,
        out_path,
        num_warps,
        num_stages,
        grid,
        kernel_path,
    ):
        subprocess.run(
            [
                sys.executable,
                self.compiler_path,
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
                "-ns",
                str(num_stages),
                "-g",
                grid,
                kernel_path,
            ],
            check=True,
            cwd=kernel_dir,
        )

    def _preprocess_jit_args(self, jit_args):
        pass


class KernelTracer(ABC):
    KERNEL: str
    COMPILER: AOTCompilerRunner

    def __init__(self, do_not_specialize=None, debug=None, noinline=None):
        self.jitted_fn = JITFunction(
            self.KERNEL,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
        )

        self.constant_names = [p.name for p in self.jitted_fn.params if p.is_constexpr]
        self.arg_names = [p.name for p in self.jitted_fn.params if not p.is_constexpr]

        self._initialize_kernel_params()

    def _initialize_kernel_params(self):
        self.args = self.set_args()
        self.constants = self.set_constants()
        self.grid = self.set_grid()

    def check_args(self, kernel_args):
        assert len(kernel_args) == len(
            self.arg_names
        ), f"Incorrect number of args, expected {self.arg_names}"

    def check_constants(self, kernel_constants):
        assert set(kernel_constants.keys()) == set(
            self.constant_names
        ), f"Incorrect constants, expected {self.constant_names}"

    @abstractmethod
    def set_args(self):
        """Set args for the kernel as a tuple

        **Order matters!**
        """
        ...

    @abstractmethod
    def set_constants(self):
        """Set constants for the kernel as a dict"""
        ...

    @abstractmethod
    def set_grid(self):
        """Set grid for the kernel as a callable or a 3-tuple of ints"""
        ...

    def trace(self, **additional_jit_kwargs):
        """Trace a kernel with the given args and constants

        Args:
            additional_jit_kwargs: number of warps, specializations, etc. -- see `triton.runtime.jit.JITFunction.run` special args.
        """
        self.check_args(self.args)
        self.check_constants(self.constants)

        compilation_artifact: CompiledArtifact = self.jitted_fn[self.grid](
            *self.args,
            **self.constants,
            **additional_jit_kwargs,
            trace=True,
        )
        return compilation_artifact


ADD_KERNEL = """
#From Triton vector_add_kernel tutorial
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  
    y_ptr,  
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
"""


class AddKernelTracer(KernelTracer):
    KERNEL = KERNELS_DIR / "add_kernel.py"

    def __init__(self, dtype, N, BLOCK_SIZE):
        self.dtype = dtype
        self.N = N
        self.BLOCK_SIZE = BLOCK_SIZE
        super().__init__()

    def set_args(
        self,
    ):
        x = torch.ones(
            self.N, dtype=self.dtype, device="cuda"
        )  # torch.rand(size, device="cuda")
        y = torch.ones(self.N, dtype=self.dtype, device="cuda")
        output = torch.empty_like(x)
        return (x, y, output, self.N)

    def set_constants(self):
        return {"BLOCK_SIZE": self.BLOCK_SIZE}

    def set_grid(self):
        return lambda meta: (triton.cdiv(self.N, meta["BLOCK_SIZE"]),)


# add_kernel_tracer = AddKernelTracer(dtype=torch.float16, N=1024, BLOCK_SIZE=1024)

# artifact = add_kernel_tracer.trace()
# for k, v in artifact.jit_args.items():
#     print(f"{k}: {v}")


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


# Edge case kernel with no specialization
# def compile_aot_kernel_no_specialization(dir, kernel_path, dtype, BM, BN, BK):
#     # compile all desired configs
#     sig = f"*fp32, *{dtype}, *{dtype}, i32, i32, i32, i32, i32, i32, i32, i32, i32, {BM}, {BN}, {BK}"
#     name = f"matmul_{dtype}"
#     grid = f"M/{BM}, N/{BN}, 1"
#     _compile_kernel(
#         dir=dir,
#         signature=sig,
#         kernel_name="kernel",
#         out_name=name,
#         out_path=name,
#         num_warps=1,
#         grid=grid,
#         kernel_path=kernel_path,
#     )


def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
    # compile all desired configs
    for ha in ha_hb_hints:
        for hb in ha_hb_hints:
            sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
            name = f"matmul_{dtype}"
            grid = f"M/{BM}, N/{BN}, 1"
            _compile_kernel(
                dir=dir,
                signature=sig,
                kernel_name="kernel",
                out_name=name,
                out_path=name,
                num_warps=1,
                grid=grid,
                kernel_path=kernel_path,
            )


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
    compile_aot_kernels(out_dir, kernel_path, dtype, BM, BN, BK, hints)
    link_aot_kernels(out_dir, "matmul")


# MATMUL_KERNEL = KERNELS_DIR / "matmul_kernel.py"
# OUT_DIR = Path("aot_matmul_ref").absolute()
# if not OUT_DIR.exists():
#     OUT_DIR.mkdir(parents=True, exist_ok=True)

# generate_matmul_reference(MATMUL_KERNEL, OUT_DIR)
# TODO: replicate refernece matmul kernel
# Cases: stride_cm: specialize (16) / no specialize
#        stride_am: specialize (16) / no specialize
# Pass do_not_specialize to jit function by param name


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


class MatMulKernelTracer(KernelTracer):
    KERNEL = KERNELS_DIR / "matmul_kernel.py"

    def __init__(self, **config):
        self.config = MatMulConfig(**config)
        super().__init__()

    def set_args(
        self,
    ):
        # Set up matrices
        torch.manual_seed(self.config.seed)

        A_shape = self.config.M, self.config.K
        B_shape = self.config.K, self.config.N
        C_shape = self.config.M, self.config.N
        A = torch.randn(A_shape, dtype=self.config.dtype_in, device="cuda").contiguous()
        B = torch.randn(B_shape, dtype=self.config.dtype_in, device="cuda").contiguous()
        C = torch.empty(C_shape, dtype=self.config.dtype_out, device="cuda")

        # Save for verification
        self.CHECK = torch.matmul(A, B).to(self.config.dtype_out).detach().cpu()
        self.C = C
        M, K = A.shape
        _, N = B.shape

        stride_cm = C.stride(0)
        stride_cn = C.stride(1)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)

        args = (
            C,
            A,
            B,
            M,
            N,
            K,
            stride_cm,
            stride_cn,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
        )

        return args

    def set_constants(self):
        return {
            "BLOCK_M": self.config.BLOCK_M,
            "BLOCK_N": self.config.BLOCK_N,
            "BLOCK_K": self.config.BLOCK_K,
        }

    def set_grid(self):
        grid = lambda META: (
            triton.cdiv(self.config.M, META["BLOCK_M"])
            * triton.cdiv(self.config.N, META["BLOCK_N"]),
        )
        return grid


# sys.path.insert(0, KERNELS_DIR)

matmul_config = MatMulConfig()
matmul_tracer = MatMulKernelTracer(**matmul_config)
artifact = matmul_tracer.trace(num_warps=1)
# Check

expected = matmul_tracer.CHECK
actual = matmul_tracer.C.detach().cpu()
print(f"Expected dtype: {expected.dtype}")
print(f"Actual dtype: {actual.dtype}")
print(f"Expected shape: {expected.shape}")
print(f"Actual shape: {actual.shape}")
print(f"Expected avg: {expected.mean()}")
print(f"Actual avg: {actual.mean()}")
print(f"Expected {expected}")
print(f"Actual {actual}")

is_close = torch.allclose(expected, actual, atol=1e-1, rtol=0)
print(f"Is close {is_close}")
