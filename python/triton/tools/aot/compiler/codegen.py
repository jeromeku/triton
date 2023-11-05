import binascii
import hashlib
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, List

from dataclasses import dataclass

from .templates import (
    DEFAULT_AOT_C_CUDA_HEADER_TEMPLATE,
    DEFAULT_AOT_C_CUDA_SOURCE_TEMPLATE,
    AOTTemplate,
)
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.compiler import CompiledKernel
from triton.compiler.make_launcher import ty_to_cpp
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


class AOTCompilerParamsBuilder(ABC):
    HEADER_TEMPLATE: AOTTemplate
    SOURCE_TEMPLATE: AOTTemplate

    @abstractmethod
    def build(self, **kwargs):
        ...

    def _validate(self, template, params: Dict[str, Any]):
        """Check that all template params are present"""

        missing_keys = template.PARAMS - set(params.keys())
        if missing_keys:
            raise ValueError(
                f"Missing following expected keys in template {template.TEMPLATE_NAME}: {missing_keys}"
            )

    def _validate_header(self, params):
        self._validate(self.HEADER_TEMPLATE, params)

    def _validate_source(self, params):
        self._validate(self.SOURCE_TEMPLATE, params)


class AOT_C_CUDA_ParamsBuilder(AOTCompilerParamsBuilder):
    HEADER_TEMPLATE = DEFAULT_AOT_C_CUDA_HEADER_TEMPLATE
    SOURCE_TEMPLATE = DEFAULT_AOT_C_CUDA_SOURCE_TEMPLATE

    def __init__(
        self,
        kernel_name,
        compiled_binary: CompiledKernel,
        jit_args: JITCompileArgs,
        jit_fn: JITFunction,
    ):
        self.kernel_name = kernel_name
        self.compiled_binary = compiled_binary
        self.jit_args = jit_args
        self.jit_fn = jit_fn

    def _hash_signature(self, sig: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(sig).encode())
        return m.hexdigest()[:8]

    def _generate_signatures(self):
        meta_sig = f"warps{self.jit_args.num_warps}xstages{self.jit_args.num_stages}"
        signature_str = [str(s) for s in self.jit_args.signature.values()]
        sig_hash = self._hash_signature(signature_str + [meta_sig])
        const_sig = "x".join([str(v) for v in self.jit_args.constants.values()])
        return AOTSignatureArgs(meta_sig, signature_str, sig_hash, const_sig)

    def _generate_docstring(self):
        doc_string = [
            f"{self.jit_fn.arg_names[i]}={self.jit_args.constants[i]}"
            for i in self.jit_args.constants.keys()
        ]
        doc_string += [
            f"num_warps={self.jit_args.num_warps}",
            f"num_stages={self.jit_args.num_stages}",
        ]
        return doc_string

    def _generate_args(self):
        arg_names = []
        arg_types = []

        config = self.jit_args.configs[0]
        for i in self.jit_args.signature.keys():
            if i not in config.equal_to_1:
                arg_names += [self.jit_fn.arg_names[i]]
                arg_types += [self.jit_args.signature[i]]
        return AOTArgs(arg_names, arg_types)

    def _generate_arg_params(self, args: AOTArgs):
        arg_pointers = ", ".join([f"&{arg}" for arg in args.arg_names])
        num_args = len(args.arg_names)
        return AOTArgParams(arg_pointers, num_args)

    def _generate_function_name_params(self, sig_hash: str) -> AOTFunctionNameParams:
        config = self.jit_args.configs[0]
        suffix = kernel_suffix(self.jit_args.signature.values(), config)
        func_name = "_".join([self.kernel_name, sig_hash, suffix])
        triton_kernel_name = "_".join([self.kernel_name, suffix])
        return AOTFunctionNameParams(
            kernel_name=func_name, triton_kernel_name=triton_kernel_name
        )

    def _generate_cubin_params(self):
        hex_ = str(binascii.hexlify(self.compiled_binary.asm["cubin"]))[2:-1]
        bin_data = ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])])
        return AOTCubinParams(bin_size=len(hex_), bin_data=bin_data)

    def _generate_signature_params(
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

    def _generate_grid_params(self):
        grid_params = AOTGridParams(
            self.jit_args.grid.x, self.jit_args.grid.y, self.jit_args.grid.z
        )
        return grid_params

    def _build_full_params(
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

    def build(self, **kwargs):
        signatures = self._generate_signatures()
        args = self._generate_args()

        arg_params = self._generate_arg_params(args)
        function_name_params = self._generate_function_name_params(
            sig_hash=signatures.sig_hash
        )
        cubin_params = self._generate_cubin_params()
        signature_params = self._generate_signature_params(args, signatures)
        grid_params = self._generate_grid_params()

        # remaining params
        kernel_docstring = self._generate_docstring()
        shared = self.compiled_binary.shared
        num_warps = self.jit_args.num_warps
        _placeholder = ""

        # build full spec
        params = self._build_full_params(
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

        self._validate_header(params)
        self._validate_source(params)
        return params
