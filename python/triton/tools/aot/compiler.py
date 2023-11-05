import binascii
import glob
import hashlib
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import dataclass

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.compiler import CompiledKernel
from triton.compiler.make_launcher import ty_to_cpp
from triton.debugging import TRITON_AOT_KERNEL_DIR
from triton.runtime.jit import JITFunction

InstanceDescriptor = namedtuple(
    "instance_descriptor",
    ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
)


@dataclass
class CompiledArtifact:
    bin: CompiledKernel
    kernel_path: str


@dataclass
class Grid:
    x: int
    y: int
    z: int

    def __str__(self):
        return f"[{self.x}, {self.y}, {self.z}]"


@dataclass
class JITCompileArgs(dict):
    signature: Dict[int, str]
    device: int
    constants: Dict[int, int]
    num_warps: int
    num_ctas: int
    num_stages: int
    enable_warp_specialization: bool
    enable_fp_fusion: bool
    extern_libs: Dict[str, str]
    configs: tuple[InstanceDescriptor]
    debug: bool
    device_type: str
    grid: Grid

    def __post_init__(self):
        self.update(self.__dict__)


def hash_signature(signature: List[str]):
    m = hashlib.sha256()
    m.update(" ".join(signature).encode())
    return m.hexdigest()[:8]


DEFAULT_HEADER_TEMPLATE = """
#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
CUresult{_placeholder} {kernel_name}(CUstream stream, {signature});
"""

DEFAULT_SOURCE_TEMPLATE = """
/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <cuda.h>


// helpers to check for cuda errors
#define CUDA_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(CUresult code, const char *file, int line) {{
  if (code != CUDA_SUCCESS) {{
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

// globals
#define CUBIN_NAME {kernel_name}_cubin
CUmodule {kernel_name}_mod = NULL;
CUfunction {kernel_name}_func = NULL;
unsigned char CUBIN_NAME[{bin_size}] = {{ {bin_data} }};


void unload_{kernel_name}(void) {{
    CUDA_CHECK(cuModuleUnload({kernel_name}_mod));
}}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_{kernel_name}() {{
    int dev = 0;
    void *bin = (void *)&CUBIN_NAME;
    int shared = {shared};
    CUDA_CHECK(cuModuleLoadData(&{kernel_name}_mod, bin));
    CUDA_CHECK(cuModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {{
      CUDA_CHECK(cuFuncSetCacheConfig({kernel_name}_func, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_CHECK(cuFuncSetAttribute({kernel_name}_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }}
}}

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return cuLaunchKernel({kernel_name}_func, gX, gY, gZ, {num_warps} * 32, 1, 1, {shared}, stream, args, NULL);
}}
"""


@dataclass
class _DataClassDict(dict):
    def __post_init__(self):
        self.update(self.__dict__)


@dataclass
class AOTSignatureArgs(_DataClassDict):
    meta_sig: str
    signature_str: List[str]
    sig_hash: str
    const_sig: str


@dataclass
class AOTSignatureParams(_DataClassDict):
    signature: str
    full_signature: str
    algo_info: str


@dataclass
class AOTArgs(_DataClassDict):
    arg_names: List[str]
    arg_types: List[str]


@dataclass
class AOTArgParams(_DataClassDict):
    arg_pointers: str
    num_args: int


@dataclass
class AOTFunctionNameParams(_DataClassDict):
    kernel_name: str
    triton_kernel_name: str


@dataclass
class AOTCubinParams(_DataClassDict):
    bin_size: int
    bin_data: str


@dataclass
class AOTGridParams(_DataClassDict):
    gridX: int
    gridY: int
    gridZ: int


@dataclass
class AOTParams(_DataClassDict):
    function_name_params: AOTFunctionNameParams
    cubin_params: AOTCubinParams
    signature_params: AOTSignatureParams
    arg_params: AOTArgParams
    grid_params: AOTGridParams
    kernel_docstring: str
    shared: int
    num_warps: int
    _placeholder: str

    def build(self):
        params = {
            **self.function_name_params,
            **self.cubin_params,
            **self.signature_params,
            **self.arg_params,
            **self.grid_params,
        }
        for k, v in self.__dict__.items():
            if not k.endswith("_params"):
                params.update({k: v})
        return params


class AOTCompiler:
    def __init__(
        self,
        kernel_name,
        compiled_binary: CompiledKernel,
        jit_args: JITCompileArgs,
        jit_fn: JITFunction,
        header_template: str | Path = DEFAULT_HEADER_TEMPLATE,
        source_template: str | Path = DEFAULT_SOURCE_TEMPLATE,
    ):
        self.kernel_name = kernel_name
        self.compiled_binary = compiled_binary
        self.jit_args = jit_args
        self.jit_fn = jit_fn
        self.header_template = header_template
        self.source_template = source_template

    def _hash_signature(self, sig: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(sig).encode())
        return m.hexdigest()[:8]

    def generate_signatures(self):
        meta_sig = f"warps{self.jit_args.num_warps}xstages{self.jit_args.num_stages}"
        signature_str = [str(s) for s in self.jit_args.signature.values()]
        sig_hash = self._hash_signature(signature_str + [meta_sig])
        const_sig = "x".join([str(v) for v in self.jit_args.constants.values()])
        return AOTSignatureArgs(meta_sig, signature_str, sig_hash, const_sig)

    def generate_docstring(self):
        doc_string = [
            f"{self.jit_fn.arg_names[i]}={self.jit_args.constants[i]}"
            for i in self.jit_args.constants.keys()
        ]
        doc_string += [
            f"num_warps={self.jit_args.num_warps}",
            f"num_stages={self.jit_args.num_stages}",
        ]
        return doc_string

    def generate_args(self):
        arg_names = []
        arg_types = []

        config = self.jit_args.configs[0]
        for i in self.jit_args.signature.keys():
            if i not in config.equal_to_1:
                arg_names += [self.jit_fn.arg_names[i]]
                arg_types += [self.jit_args.signature[i]]
        return AOTArgs(arg_names, arg_types)

    def generate_arg_params(self, args: AOTArgs):
        arg_pointers = ", ".join([f"&{arg}" for arg in args.arg_names])
        num_args = len(args.arg_names)
        return AOTArgParams(arg_pointers, num_args)

    def generate_function_name_params(self, sig_hash: str) -> AOTFunctionNameParams:
        config = self.jit_args.configs[0]
        suffix = kernel_suffix(self.jit_args.signature.values(), config)
        func_name = "_".join([self.kernel_name, sig_hash, suffix])
        triton_kernel_name = "_".join([self.kernel_name, suffix])
        return AOTFunctionNameParams(
            kernel_name=func_name, triton_kernel_name=triton_kernel_name
        )

    def generate_cubin_params(self):
        hex_ = str(binascii.hexlify(self.compiled_binary.asm["cubin"]))[2:-1]
        bin_data = ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])])
        return AOTCubinParams(bin_size=len(hex_), bin_data=bin_data)

    def generate_signature_params(
        self, args: AOTArgs, signatures: AOTSignatureArgs
    ) -> AOTSignatureParams:
        signature = ", ".join(
            [
                f"{ty_to_cpp(ty)} {name}"
                for name, ty in zip(args.arg_names, args.arg_types)
            ]
        )

        full_signature = ", ".join(
            [
                f"{ty_to_cpp(self.jit_args.signature[i])} {self.jit_fn.arg_names[i]}"
                for i in self.jit_args.signature.keys()
            ]
        )
        algo_info = "_".join([signatures.const_sig, signatures.meta_sig])

        return AOTSignatureParams(signature, full_signature, algo_info)

    def generate_grid_params(self):
        grid_params = AOTGridParams(
            self.jit_args.grid.x, self.jit_args.grid.y, self.jit_args.grid.z
        )
        return grid_params

    def build_full_params(
        self,
        function_name_params,  # kernel_name, triton_kernel_name
        cubin_params,  # bin_size, bin_data
        signature_params,  # signature, full_signature, algo_info
        arg_params,  # arg_pointers, num_args
        grid_params,  # gridX, gridY, gridZ
        kernel_docstring,
        shared,
        num_warps,
        _placeholder="",
    ):
        params = AOTParams(
            function_name_params=function_name_params,
            cubin_params=cubin_params,
            signature_params=signature_params,
            arg_params=arg_params,
            grid_params=grid_params,
            kernel_docstring=kernel_docstring,
            shared=shared,
            num_warps=num_warps,
            _placeholder=_placeholder,
        )
        return params.build()

    def generate_full_params(self, **kwargs):
        signatures = self.generate_signatures()
        args = self.generate_args()

        arg_params = self.generate_arg_params(args)
        function_name_params = self.generate_function_name_params(
            sig_hash=signatures.sig_hash
        )
        cubin_params = self.generate_cubin_params()
        signature_params = self.generate_signature_params(args, signatures)
        grid_params = self.generate_grid_params()

        # remaining params
        kernel_docstring = self.generate_docstring()
        shared = self.compiled_binary.shared
        num_warps = self.jit_args.num_warps
        _placeholder = ""

        # build full spec
        params = self.build_full_params(
            function_name_params=function_name_params,
            cubin_params=cubin_params,
            signature_params=signature_params,
            arg_params=arg_params,
            grid_params=grid_params,
            kernel_docstring=kernel_docstring,
            shared=shared,
            num_warps=num_warps,
            _placeholder=_placeholder,
        )
        if kwargs:
            params.update(kwargs)
        return params


def create_aot_kernel(
    bin: CompiledKernel,
    jit_fn: JITFunction,
    jit_args: JITCompileArgs,
):
    kernel_name = jit_fn.__name__
    ## Create AOT artifacts
    meta_sig = f"warps{jit_args.num_warps}xstages{jit_args.num_stages}"
    signature_str = [str(s) for s in jit_args.signature.values()]
    sig_hash = hash_signature(signature_str + [meta_sig])
    const_sig = "x".join([str(v) for v in jit_args.constants.values()])
    doc_string = [
        f"{jit_fn.arg_names[i]}={jit_args.constants[i]}"
        for i in jit_args.constants.keys()
    ]
    doc_string += [
        f"num_warps={jit_args.num_warps}",
        f"num_stages={jit_args.num_stages}",
    ]

    arg_names = []
    arg_types = []

    config = jit_args.configs[0]
    for i in jit_args.signature.keys():
        if i not in config.equal_to_1:
            arg_names += [jit_fn.arg_names[i]]
            arg_types += [jit_args.signature[i]]

    # dump C stub code
    suffix = kernel_suffix(jit_args.signature.values(), config)
    func_name = "_".join([kernel_name, sig_hash, suffix])
    triton_kernel_name = "_".join([kernel_name, suffix])
    hex_ = str(binascii.hexlify(bin.asm["cubin"]))[2:-1]
    params = {
        "kernel_name": func_name,  # func_name_params
        "triton_kernel_name": triton_kernel_name,  # func_name_params
        "bin_size": len(hex_),  # cubin_params
        "bin_data": ", ".join(
            [f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]
        ),  # cubin_params
        "signature": ", ".join(
            [f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]
        ),  # signature_params
        "full_signature": ", ".join(
            [
                f"{ty_to_cpp(jit_args.signature[i])} {jit_fn.arg_names[i]}"
                for i in jit_args.signature.keys()
            ]
        ),  # signature_params
        "algo_info": "_".join([const_sig, meta_sig]),  # signature_params
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),  # arg_params
        "num_args": len(arg_names),  # arg_params
        "gridX": jit_args.grid.x,  # grid_params
        "gridY": jit_args.grid.y,  # grid_params
        "gridZ": jit_args.grid.z,  # grid_params
        "kernel_docstring": doc_string,  # docstring
        "shared": bin.shared,  #
        "num_warps": jit_args.num_warps,
        "_placeholder": "",
    }

    kernel_dir = os.environ.get("TRITON_AOT_KERNEL_DIR", TRITON_AOT_KERNEL_DIR)
    out_dir = Path(kernel_dir) / kernel_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["h", "c"]:
        template_path = Path(__file__).parent.parent / "tools" / f"compile.{ext}"
        out_name = Path(kernel_name).with_suffix(f".{sig_hash}_{suffix}.{ext}")

        with (out_dir / out_name).open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))

    link_aot_kernel(out_dir, kernel_name)
    return out_dir


def link_aot_kernel(kernel_path, dispatcher_name):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(kernel_path, "*.h"))
    subprocess.run(
        [sys.executable, linker_path] + h_files + ["-o", dispatcher_name],
        check=True,
        cwd=kernel_path,
    )
