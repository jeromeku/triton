from abc import ABC, abstractmethod
from pathlib import Path

from dataclasses import dataclass

import triton
from .. import DEFAULT_TRACE_DIR
from .codegen import AOT_C_CUDA_ParamsBuilder, AOTCompilerParamsBuilder, JITCompileArgs
from triton.compiler.compiler import CompiledKernel, instance_descriptor
from triton.runtime.jit import JITFunction


class AOT_Compiler(ABC):
    PARAM_BUILDER_CLS: AOTCompilerParamsBuilder

    def __init__(
        self,
        kernel_name,
        jit_args: JITCompileArgs,
        jit_fn: JITFunction,
        save_dir: Path = None,
        compiled_binary: CompiledKernel = None,
    ):
        self.kernel_name = kernel_name
        self.jit_args = jit_args

        self.save_dir = save_dir or DEFAULT_TRACE_DIR
        self.compiled_binary = compiled_binary or triton.compile(jit_fn, **jit_args)

        self.params_builder = self.PARAM_BUILDER_CLS(
            kernel_name=kernel_name,
            compiled_binary=self.compiled_binary,
            jit_args=jit_args,
            jit_fn=jit_fn,
        )
        self.params = self.build_params()

    def build_params(self):
        return self.params_builder.build()

    @abstractmethod
    def generate_header(self):
        ...

    @abstractmethod
    def generate_source(self):
        ...


@dataclass
class AOTCompilationResult:
    kernel_name: str
    header: str
    source: str
    params: dict
    header_path: str | Path
    source_path: str | Path
    compiled_binary: CompiledKernel
    # For debugging
    _jit_args: JITCompileArgs
    _compiler_params: dict


class AOT_C_CUDA_Compiler(AOT_Compiler):
    """Creates C CUDA library for accessing Triton jitted kernels"""

    PARAM_BUILDER_CLS = AOT_C_CUDA_ParamsBuilder

    def generate_header(self):
        # Filter params for header keys
        header_params = {
            k: v
            for k, v in self.params.items()
            if k in self.params_builder.HEADER_TEMPLATE.PARAMS
        }
        # Generate header
        header = self.params_builder.HEADER_TEMPLATE.TEMPLATE.format(**header_params)
        return header

    def generate_source(self):
        # Filter params for source keys
        source_params = {
            k: v
            for k, v in self.params.items()
            if k in self.params_builder.SOURCE_TEMPLATE.PARAMS
        }
        # Generate source
        source = self.params_builder.SOURCE_TEMPLATE.TEMPLATE.format(**source_params)
        return source

    def generate(self):
        header = self.generate_header()
        source = self.generate_source()
        suffix = "_".join(f'{self.params["kernel_name"]}'.split("_")[1:])
        file_name = f"{self.kernel_name}.{suffix}"
        header_name = f"{file_name}.h"
        source_name = f"{file_name}.c"

        header_path = self.save_dir / header_name
        source_path = self.save_dir / source_name

        with open(header_path, "w") as fp:
            fp.write(header)
        with open(source_path, "w") as fp:
            fp.write(source)

        return AOTCompilationResult(
            kernel_name=self.kernel_name,
            header=header,
            source=source,
            params=self.params,
            header_path=self.save_dir / header_name,
            source_path=self.save_dir / source_name,
            compiled_binary=self.compiled_binary,
            _jit_args=self.jit_args,
            _compiler_params=self.params,
        )


# def create_aot_kernel(
#     bin: CompiledKernel,
#     jit_fn: JITFunction,
#     jit_args: JITCompileArgs,
# ):
#     kernel_name = jit_fn.__name__
#     ## Create AOT artifacts
#     meta_sig = f"warps{jit_args.num_warps}xstages{jit_args.num_stages}"
#     signature_str = [str(s) for s in jit_args.signature.values()]
#     sig_hash = hash_signature(signature_str + [meta_sig])
#     const_sig = "x".join([str(v) for v in jit_args.constants.values()])
#     doc_string = [
#         f"{jit_fn.arg_names[i]}={jit_args.constants[i]}"
#         for i in jit_args.constants.keys()
#     ]
#     doc_string += [
#         f"num_warps={jit_args.num_warps}",
#         f"num_stages={jit_args.num_stages}",
#     ]

#     arg_names = []
#     arg_types = []

#     config = jit_args.configs[0]
#     for i in jit_args.signature.keys():
#         if i not in config.equal_to_1:
#             arg_names += [jit_fn.arg_names[i]]
#             arg_types += [jit_args.signature[i]]

#     # dump C stub code
#     suffix = kernel_suffix(jit_args.signature.values(), config)
#     func_name = "_".join([kernel_name, sig_hash, suffix])
#     triton_kernel_name = "_".join([kernel_name, suffix])
#     hex_ = str(binascii.hexlify(bin.asm["cubin"]))[2:-1]
#     params = {
#         "kernel_name": func_name,  # func_name_params
#         "triton_kernel_name": triton_kernel_name,  # func_name_params
#         "bin_size": len(hex_),  # cubin_params
#         "bin_data": ", ".join(
#             [f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]
#         ),  # cubin_params
#         "signature": ", ".join(
#             [f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]
#         ),  # signature_params
#         "full_signature": ", ".join(
#             [
#                 f"{ty_to_cpp(jit_args.signature[i])} {jit_fn.arg_names[i]}"
#                 for i in jit_args.signature.keys()
#             ]
#         ),  # signature_params
#         "algo_info": "_".join([const_sig, meta_sig]),  # signature_params
#         "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),  # arg_params
#         "num_args": len(arg_names),  # arg_params
#         "gridX": jit_args.grid.x,  # grid_params
#         "gridY": jit_args.grid.y,  # grid_params
#         "gridZ": jit_args.grid.z,  # grid_params
#         "kernel_docstring": doc_string,  # docstring
#         "shared": bin.shared,  #
#         "num_warps": jit_args.num_warps,
#         "_placeholder": "",
#     }

#     kernel_dir = os.environ.get("TRITON_AOT_KERNEL_DIR", TRITON_AOT_KERNEL_DIR)
#     out_dir = Path(kernel_dir) / kernel_name
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for ext in ["h", "c"]:
#         template_path = Path(__file__).parent.parent / "tools" / f"compile.{ext}"
#         out_name = Path(kernel_name).with_suffix(f".{sig_hash}_{suffix}.{ext}")

#         with (out_dir / out_name).open("w") as fp:
#             fp.write(Path(template_path).read_text().format(**params))

#     link_aot_kernel(out_dir, kernel_name)
#     return out_dir


# def link_aot_kernel(kernel_path, dispatcher_name):
#     linker_path = os.path.join(triton.tools.__path__[0], "link.py")

#     # link all desired configs
#     h_files = glob.glob(os.path.join(kernel_path, "*.h"))
#     subprocess.run(
#         [sys.executable, linker_path] + h_files + ["-o", dispatcher_name],
#         check=True,
#         cwd=kernel_path,
#     )
