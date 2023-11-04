import binascii
import hashlib
import os
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import dataclass

from triton.compiler.code_generator import kernel_suffix
from triton.compiler.compiler import CompiledFn
from triton.compiler.make_launcher import ty_to_cpp
from triton.runtime.jit import JITFunction

InstanceDescriptor = namedtuple(
    "instance_descriptor",
    ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
)


@dataclass
class CompiledArtifact:
    bin: CompiledFn
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
    constants: Dict[int, Any]
    num_warps: int
    num_ctas: int
    num_stages: int
    enable_warp_specialization: bool
    enable_fp_fusion: bool
    extern_libs: Dict[str, str]
    config: tuple[InstanceDescriptor]
    debug: bool
    device_type: str
    grid: Grid

    def __post_init__(self):
        self.update(self.__dict__)


def hash_signature(signature: List[str]):
    m = hashlib.sha256()
    m.update(" ".join(signature).encode())
    return m.hexdigest()[:8]


def create_AOT_artifacts(jit_fn: JITFunction, jit_args: JITCompileArgs):
    kernel_name = jit_fn.__name__
    ## Create AOT artifacts
    meta_sig = f"warps{jit_args.num_warps}xstages{jit_args.num_stages}"
    signature_str = [str(s) for s in jit_args.signature.values()]
    sig_hash = hash_signature(signature_str + [meta_sig])
    const_sig = "x".join([str(v) for v in jit_args.constants.values()])
    doc_string = [
        f"{jit_fn.arg_names[i]}={jit_args.constants[i]}" for i in jit_args.keys()
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

    return out_dir


def link_aot_artifacts(kernel_path, dispatcher_name):
    import glob
    import subprocess
    import sys

    import triton

    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(kernel_path, "*.h"))
    subprocess.run(
        [sys.executable, linker_path] + h_files + ["-o", dispatcher_name],
        check=True,
        cwd=kernel_path,
    )
