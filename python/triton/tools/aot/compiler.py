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


class AOTCompiler:
    def __init__(
        self,
        kernel_name,
        compiled_binary: CompiledKernel,
        jit_args: JITCompileArgs,
        header_template: str | Path = DEFAULT_HEADER_TEMPLATE,
        source_template: str | Path = DEFAULT_SOURCE_TEMPLATE,
    ):
        self.kernel_name = kernel_name
        self.compiled_binary = compiled_binary
        self.jit_args = jit_args
        self.header_template = header_template
        self.source_template = source_template


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
        "kernel_name": func_name,
        "triton_kernel_name": triton_kernel_name,
        "bin_size": len(hex_),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join(
            [f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]
        ),
        "full_signature": ", ".join(
            [
                f"{ty_to_cpp(jit_args.signature[i])} {jit_fn.arg_names[i]}"
                for i in jit_args.signature.keys()
            ]
        ),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args": len(arg_names),
        "kernel_docstring": doc_string,
        "shared": bin.shared,
        "num_warps": jit_args.num_warps,
        "algo_info": "_".join([const_sig, meta_sig]),
        "gridX": jit_args.grid.x,
        "gridY": jit_args.grid.y,
        "gridZ": jit_args.grid.z,
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
