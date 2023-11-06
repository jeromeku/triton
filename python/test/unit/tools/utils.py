import glob
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

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
class TraceConfig(dict):
    """Kwargs passed to `JITFunction.run`"""

    # Launch params
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    num_ctas: Optional[int] = None
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = True

    # JIT options
    do_not_specialize: List[str] = None
    debug: bool = False
    noinline: bool = False

    # Additional options
    extern_libs: List[str] = None
    stream: Optional[int] = None
    warmup: bool = False

    # Trace options
    trace: bool = True
    trace_dir: Optional[Path] = None

    def __post_init__(self):
        self.update(self.__dict__)


@dataclass
class MatMulConfig:
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


class KernelTracer(ABC):
    KERNEL: str
    COMPILER: AOTCompilerRunner

    # def __init__(self, do_not_specialize=None, debug=None, noinline=None):
    #     self.jitted_fn = JITFunction(
    #         self.KERNEL,
    #         do_not_specialize=do_not_specialize,
    #         debug=debug,
    #         noinline=noinline,
    #     )

    #     self.constant_names = [p.name for p in self.jitted_fn.params if p.is_constexpr]
    #     self.arg_names = [p.name for p in self.jitted_fn.params if not p.is_constexpr]

    # self._initialize_kernel_params()

    def _initialize_kernel_params(self):
        self.args = self.set_args()
        self.constants = self.set_constants()
        self.grid = self.set_grid()

    def check_args(self, kernel_args):
        assert len(kernel_args) == len(
            self.arg_names
        ), f"Incorrect number of args, expected {self.arg_names}"
        for i, (expected_key, actual_key) in enumerate(
            zip(self.arg_names, kernel_args.keys())
        ):
            assert (
                expected_key == actual_key
            ), f"Incorrect arg name at position {i}, expected {expected_key}, got {actual_key}"

    def check_constants(self, kernel_constants):
        assert set(kernel_constants.keys()) == set(
            self.constant_names
        ), f"Incorrect constants, expected {self.constant_names}"

    @abstractmethod
    def set_args(self):
        """Set args for the kernel as an OrderedDict"""
        ...

    @abstractmethod
    def set_constants(self):
        """Set constants for the kernel as a dict"""
        ...

    @abstractmethod
    def set_grid(self):
        """Set grid for the kernel as a callable or a 3-tuple of ints"""
        ...

    def trace(self, kernel_config, trace_config: TraceConfig):
        """Trace a kernel with the given args and constants

        Args:
            additional_jit_kwargs: number of warps, specializations, etc. -- see `triton.runtime.jit.JITFunction.run` special args.
        """

        self.jitted_fn = JITFunction(
            self.KERNEL,
            do_not_specialize=trace_config.pop("do_not_specialize"),
            debug=trace_config.pop("debug"),
            noinline=trace_config.pop("noinline"),
        )

        self.constant_names = [p.name for p in self.jitted_fn.params if p.is_constexpr]
        self.arg_names = [p.name for p in self.jitted_fn.params if not p.is_constexpr]

        args = self.set_args(kernel_config)
        self.check_args(args)

        constants = self.set_constants(kernel_config)
        self.check_constants(constants)

        compilation_artifact: CompiledArtifact = self.jitted_fn[self.grid](
            *args.values(),
            **constants,
            **trace_config,
        )
        return compilation_artifact


class MatMulKernelTracer(KernelTracer):
    KERNEL = KERNELS_DIR / "matmul_kernel.py"

    def __init__(
        self,
        config: MatMulConfig,
        do_not_specialize=None,
        debug=None,
        noinline=None,
    ):
        self.config = config
        self.checks = []
        super().__init__(
            do_not_specialize=do_not_specialize, debug=debug, noinline=noinline
        )

    def trace(self, configs: List[MatMulConfig]):
        pass

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

        args = OrderedDict(
            C=C,
            A=A,
            B=B,
            M=M,
            N=N,
            K=K,
            stride_cm=stride_cm,
            stride_cn=stride_cn,
            stride_am=stride_am,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            stride_bn=stride_bn,
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


matmul_config_16x16x16 = MatMulConfig()
matmul_config_256x256x256 = MatMulConfig()
matmul_config = matmul_config_256x256x256
do_not_specialize = ["stride_cm, stride_am"]
matmul_tracer = MatMulKernelTracer(matmul_config, do_not_specialize=do_not_specialize)


trace_dir = Path("traced_kernels").absolute()
if not trace_dir.exists():
    trace_dir.mkdir(parents=True, exist_ok=True)

artifact = matmul_tracer.trace(num_warps=1, trace_dir=trace_dir)
# Check signatures

expected = matmul_tracer.CHECK
actual = matmul_tracer.C.detach().cpu()
is_close = torch.allclose(expected, actual, atol=1e-1, rtol=0)
print(f"Is close {is_close}")


# TODO use ordereddict for args
# class AddKernelTracer(KernelTracer):
#     KERNEL = KERNELS_DIR / "add_kernel.py"

#     def __init__(self, dtype, N, BLOCK_SIZE):
#         self.dtype = dtype
#         self.N = N
#         self.BLOCK_SIZE = BLOCK_SIZE
#         super().__init__()

#     def set_args(
#         self,
#     ):
#         x = torch.ones(
#             self.N, dtype=self.dtype, device="cuda"
#         )  # torch.rand(size, device="cuda")
#         y = torch.ones(self.N, dtype=self.dtype, device="cuda")
#         output = torch.empty_like(x)
#         return (x, y, output, self.N)

#     def set_constants(self):
#         return {"BLOCK_SIZE": self.BLOCK_SIZE}

#     def set_grid(self):
#         return lambda meta: (triton.cdiv(self.N, meta["BLOCK_SIZE"]),)
