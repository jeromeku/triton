from dataclasses import dataclass

from .codegen import AOT_C_CUDA_ParamsBuilder, AOTCompilerParamsBuilder, JITCompileArgs
from .compiler import AOT_C_CUDA_Compiler
from triton.compiler.compiler import CompiledKernel


@dataclass
class CompiledArtifact:
    compiled_binary: CompiledKernel
    kernel_path: str
    compiler_spec: dict
    jit_args: JITCompileArgs
