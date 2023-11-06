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
    KERNEL_PATH: (str | Path)

    def __init__(
        self, kernel_path, save_path, do_not_specialize=None, debug=None, noinline=None
    ):
        self.jitted_fn = JITFunction(
            self.KERNEL_PATH,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
        )
        self.arg_names = self.jitted_fn.arg_names
        self.constant_names = [p.name for p in self.jitted_fn.params if p.is_constexpr]
        self.save_path = save_path

    def check_args(self, kernel_args):
        assert len(kernel_args) == len(self.arg_names)

    def check_constants(self, kernel_constants):
        assert set(kernel_constants.keys()) == set(self.jitted_fn.constant_names)

    def trace(
        self, kernel_args_tup, kernel_constants_map, *, grid, **additional_jit_kwargs
    ):
        """Trace a kernel with the given args and constants

        Args:
            kernel_args_tup (tuple): tuple of kernel arguments
            kernel_constants_map (dict): dictionary of kernel constants
            grid (callable): callable that returns the grid size
            additional_jit_kwargs: number of warps, specializations, etc. -- see `triton.runtime.jit.JITFunction.run` special args.
        """
        self.check_args(kernel_args_tup)
        self.check_constants(kernel_constants_map)

        compilation_artifact: CompiledArtifact = self.jitted_fn[grid](
            *kernel_args_tup,
            **kernel_constants_map,
            **additional_jit_kwargs,
            trace=True,
            trace_dir=self.save_path,
        )
        return compilation_artifact


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
