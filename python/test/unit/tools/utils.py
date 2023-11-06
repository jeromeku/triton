import glob
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch

import triton
from triton.runtime.jit import JITFunction
from triton.tools.aot import CompiledArtifact

"""
Utilities for generating reference AOT kernels 
"""


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


class KernelTracer:
    KERNEL: str

    def __init__(self, do_not_specialize=None, debug=None, noinline=None):
        self.jitted_fn = JITFunction(
            self.KERNEL,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
        )
        self.arg_names = self.jitted_fn.arg_names
        self.constant_names = [p.name for p in self.jitted_fn.params if p.is_constexpr]

        self._initialize_kernel_params()

    def _initialize_kernel_params(self):
        self.args = self.set_args()
        self.constants = self.set_constants()
        self.grid = self.set_grid()

    def check_args(self, kernel_args):
        assert len(kernel_args) == len(self.arg_names)

    def check_constants(self, kernel_constants):
        assert set(kernel_constants.keys()) == set(self.jitted_fn.constant_names)

    @abstractmethod
    def set_args(self):
        """Set args for the kernel as a tuple"""
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
            trace_dir=self.save_path,
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
    KERNEL = ADD_KERNEL

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
        return x, y, output, self.N

    def set_constants(self):
        return {"BLOCK_SIZE": self.BLOCK_SIZE}

    def set_grid(self):
        return lambda meta: (triton.cdiv(self.N, meta["BLOCK_SIZE"]),)


def trace_add_kernel(kernel_path):
    """Automatically generate parameters for AOT compilation by kernel tracing

    Args:
        kernel_path (str | Path): path to the kernel file
    """
    N = 1024
    BLOCK_SIZE = 1024
    NUM_WARPS = 4
    seed = 0
    dtype = torch.float16

    torch.manual_seed(seed)
    x = torch.ones(N, dtype=dtype, device="cuda")  # torch.rand(size, device="cuda")
    y = torch.ones(N, dtype=dtype, device="cuda")

    # Set up aot kernel directory
    test_dir = Path("aot_compilation_spec_test").absolute()
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
