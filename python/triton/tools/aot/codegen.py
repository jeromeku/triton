from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .parsers import KernelLinkerMeta


class SignatureGenerator:
    @staticmethod
    def gen_signature_with_full_args(m):
        return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])

    @staticmethod
    def gen_signature(m):
        arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
        arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
        sig = ", ".join([f"{ty} {arg}" for ty, arg in zip(arg_types, arg_names)])
        return sig


DEFAULT_ALGO_DECL_TEMPLATE = """
CUresult {name}(CUstream stream, {args});
void load_{name}();
void unload_{name}();
"""

DEFAULT_GLOBAL_DECL_TEMPLATE = """
CUresult {orig_kernel_name}_default(CUstream stream, {default_args});
CUresult {orig_kernel_name}(CUstream stream, {full_args}, int algo_id);
void load_{orig_kernel_name}();
void unload_{orig_kernel_name}();
"""


class HeaderGenerator:
    signature_generator = SignatureGenerator
    ALGO_DECL_TEMPLATE = DEFAULT_ALGO_DECL_TEMPLATE
    GLOBAL_DECL_TEMPLATE = DEFAULT_GLOBAL_DECL_TEMPLATE

    @classmethod
    def make_algo_decls(cls, name: str, metas: Sequence[KernelLinkerMeta]) -> str:
        """Generate declarations of kernels with meta-parameter and constant values"""
        args = cls.signature_generator.gen_signature_with_full_args(metas[-1])
        return cls.ALGO_DECL_TEMPLATE.format(
            name=name,
            args=args,
        )

    @classmethod
    def make_get_num_algos_decl(cls, meta: KernelLinkerMeta) -> str:
        src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
        return src

    @classmethod
    def make_global_decl(cls, meta: KernelLinkerMeta) -> str:
        """Generate declarations of kernels with meta-parameter and constant values"""
        return cls.GLOBAL_DECL_TEMPLATE.format(
            orig_kernel_name=meta.orig_kernel_name,
            default_args=cls.signature_generator.gen_signature_with_full_args(meta),
            full_args=cls.signature_generator.gen_signature_with_full_args(meta),
        )


# DEFAULT_ALGO_KERNEL_TEMPLATE = """
# CUresult {orig_kernel_name}_default(CUstream stream, {default_kernel_args}){{\n
#     return {orig_kernel_name}(stream, {kernel_arg_names}, 0);\n
# }}\n
# """


# def make_default_algo_kernel(
#     meta: KernelLinkerMeta, signature_generator=SignatureGenerator
# ) -> str:
#     """Generate dispatcher function for kernels with different meta-parameter and constant values"""
#     original_kernel_name = meta.orig_kernel_name
#     default_kernel_args = signature_generator.gen_signature_with_full_args(meta)
#     kernel_arg_names = ", ".join(meta.arg_names)

#     src = DEFAULT_ALGO_KERNEL_TEMPLATE.format(
#         orig_kernel_name=original_kernel_name,
#         default_kernel_args=default_kernel_args,
#         kernel_arg_names=kernel_arg_names,
#     )
#     return src

# src = f"CUresult {meta.orig_kernel_name}_default(CUstream stream, {signature_generator.gen_signature_with_full_args(meta)}){{\n"
# src += (
#     f"  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n"
# )
# src += "}\n"
# return src


# generate dispatcher function for kernels with different integer value hints
# def make_kernel_hints_dispatcher(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
#     src = f"// launcher for: {name}\n"
#     for meta in sorted(metas, key=lambda m: -m.num_specs):
#         src += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {gen_signature(meta)});\n"
#     src += "\n"

#     src += (
#         f"CUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])}){{"
#     )
#     src += "\n"
#     for meta in sorted(metas, key=lambda m: -m.num_specs):
#         cond_fn = (
#             lambda val, hint: f"({val} % {hint} == 0)"
#             if hint == 16
#             else f"({val} == {hint})"
#             if hint == 1
#             else None
#         )
#         conds = " && ".join(
#             [
#                 cond_fn(val, hint)
#                 for val, hint in zip(meta.arg_names, meta.sizes)
#                 if hint is not None
#             ]
#         )
#         src += (
#             f"  if ({conds})\n" if any(meta.sizes) else "if (1)\n"
#         )  # Edge case where no specializations hence no dispatching required
#         arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
#         src += f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n"
#     src += "\n"
#     src += "  return CUDA_ERROR_INVALID_VALUE;\n"
#     src += "}\n"

#     for mode in ["load", "unload"]:
#         src += f"\n// {mode} for: {name}\n"
#         for meta in sorted(metas, key=lambda m: -m.num_specs):
#             src += f"void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
#         src += f"void {mode}_{name}() {{"
#         src += "\n"
#         for meta in sorted(metas, key=lambda m: -m.num_specs):
#             src += (
#                 f"  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
#             )
#         src += "}\n"
#     return src


# generate dispatcher function for kernels with different meta-parameter and constant values
# def make_kernel_meta_const_dispatcher(meta: KernelLinkerMeta) -> str:
#     src = f"CUresult {meta.orig_kernel_name}(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id){{\n"
#     src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels));\n"
#     src += f"  return {meta.orig_kernel_name}_kernels[algo_id](stream, {', '.join(meta.arg_names)});\n"
#     src += "}\n"
#     return src


# generate definition of function pointers of kernel dispatchers based on meta-parameter and constant values
# def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
#     # the table of hint dispatchers
#     src = f"typedef CUresult (*kernel_func_t)(CUstream stream, {gen_signature_with_full_args(meta)});\n"
#     src += f"kernel_func_t {meta.orig_kernel_name}_kernels[] = {{\n"
#     for name in names:
#         src += f"  {name},\n"
#     src += "};\n"
#     return src


# generate definition for load/unload functions for kernels with different meta-parameter and constant values
# def make_kernel_load_def(names: str, meta: KernelLinkerMeta) -> str:
#     src = ""
#     for mode in ["load", "unload"]:
#         src += f"void {mode}_{meta.orig_kernel_name}(void){{\n"
#         for name in names:
#             src += f"  {mode}_{name}();\n"
#         src += "}\n\n"
#     return src


# def make_get_num_algos_decl(meta: KernelLinkerMeta) -> str:
#     src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
#     return src


# def make_get_num_algos_def(meta: KernelLinkerMeta) -> str:
#     src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
#     src += f"  return (int)sizeof({meta.orig_kernel_name}_kernels);\n"
#     src += "}\n"
#     return src
