from typing import Dict, List, Optional, Sequence

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

    def __init__(self, kernels: Dict[str, KernelLinkerMeta]) -> None:
        self.kernels = kernels
        meta_lists = [meta for _, meta in self.kernels.items()]
        self.meta = meta_lists[0][0]

    def _make_algo_decl(self, name: str, metas: List[KernelLinkerMeta]):
        args = self.signature_generator.gen_signature_with_full_args(metas[-1])
        return self.ALGO_DECL_TEMPLATE.format(
            name=name,
            args=args,
        )

    def make_algo_decls(self) -> str:
        """Generate declarations of kernels with meta-parameter and constant values"""
        algo_decls = []

        for name, meta in self.kernels.items():
            algo_decls.append(self._make_algo_decl(name, meta))

        return "\n".join(algo_decls).strip()

    def make_get_num_algos_decl(self, meta: Optional[KernelLinkerMeta] = None) -> str:
        meta = meta or self.meta
        src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
        return src

    def make_global_decl(self, meta: Optional[KernelLinkerMeta] = None) -> str:
        """Generate declarations of kernels with meta-parameter and constant values"""
        meta = meta or self.meta
        return self.GLOBAL_DECL_TEMPLATE.format(
            orig_kernel_name=meta.orig_kernel_name,
            default_args=self.signature_generator.gen_signature_with_full_args(meta),
            full_args=self.signature_generator.gen_signature_with_full_args(meta),
        )


DEFAULT_SOURCE_INCLUDES = [
    "#include <cuda.h>",
    "#include <stdint.h>",
    "#include <assert.h>",
]


class SourceGenerator:
    signature_generator = SignatureGenerator
    SOURCE_INCLUDES = DEFAULT_SOURCE_INCLUDES

    def __init__(
        self,
        kernels: Dict[str, KernelLinkerMeta],
        meta: Optional[KernelLinkerMeta] = None,
    ) -> None:
        self.kernels = kernels
        if meta is None:
            meta_lists = [meta for name, meta in self.kernels.items()]
            meta = meta_lists[0][0]
        self.meta = meta

    def _condition_fn(self, val, hint):
        if hint == 16:
            return f"({val} % {hint} == 0)"
        elif hint == 1:
            return f"({val} == {hint})"
        else:
            return None

    def _make_dispatcher_conditions(self, metas: List[KernelLinkerMeta]):
        src = ""
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            conds = " && ".join(
                [
                    self._condition_fn(val, hint)
                    for val, hint in zip(meta.arg_names, meta.sizes)
                    if hint is not None
                ]
            )
            src += (
                f"  if ({conds})\n" if any(meta.sizes) else "if (1)\n"
            )  # Edge case where no specializations hence no dispatching required
        return src

    def _make_dispatcher_load_defs(self, name, metas):
        src = ""
        for mode in ["load", "unload"]:
            src += f"\n// {mode} for: {name}\n"
            for meta in sorted(metas, key=lambda m: -m.num_specs):
                src += f"void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
            src += f"void {mode}_{name}() {{"
            src += "\n"
            for meta in sorted(metas, key=lambda m: -m.num_specs):
                src += f"  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
            src += "}\n"
        return src

    def _make_kernel_hints_dispatcher(
        self, name: str, metas: Sequence[KernelLinkerMeta]
    ) -> str:
        # generate dispatcher function for kernels with different integer value hints
        docs_str = f"// launcher for: {name}\n"
        fn_sig = ""
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            fn_sig += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {self.signature_generator.gen_signature(meta)});\n"
        # src += "\n"

        kernel_sig = f"CUresult {name}(CUstream stream, {self.signature_generator.gen_signature_with_full_args(metas[-1])}){{"
        # src += "\n"

        dispatcher_conds = self._make_dispatcher_conditions(metas)

        arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
        return_statements = f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n"
        return_statements += "\n"
        return_statements += "  return CUDA_ERROR_INVALID_VALUE;\n"
        return_statements += "}\n"

        load_defs = self._make_dispatcher_load_defs(name, metas)

        src = "\n".join(
            [
                docs_str + fn_sig,
                kernel_sig,
                dispatcher_conds + return_statements + load_defs,
            ]
        )
        return src

    def make_defs(self):
        defs = []
        for name, metas in self.kernels.items():
            defs.append(self._make_kernel_hints_dispatcher(name, metas))
        return "\n".join(defs)

    def make_func_pointers(self) -> str:
        # the table of hint dispatchers
        src = f"typedef CUresult (*kernel_func_t)(CUstream stream, {self.signature_generator.gen_signature_with_full_args(self.meta)});\n"
        src += f"kernel_func_t {self.meta.orig_kernel_name}_kernels[] = {{\n"
        for name in self.kernels.keys():
            src += f"  {name},\n"
        src += "};\n"

        return src

    def make_kernel_meta_const_dispatcher(
        self,
        meta: KernelLinkerMeta = None,
    ) -> str:
        meta = meta or self.meta
        src = f"CUresult {meta.orig_kernel_name}(CUstream stream, {self.signature_generator.gen_signature_with_full_args(meta)}, int algo_id){{\n"
        src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels));\n"
        src += f"  return {meta.orig_kernel_name}_kernels[algo_id](stream, {', '.join(meta.arg_names)});\n"
        src += "}\n"
        return src

    def make_kernel_load_defs(self, meta: KernelLinkerMeta = None) -> str:
        meta = meta or self.meta
        src = ""
        for mode in ["load", "unload"]:
            src += f"void {mode}_{meta.orig_kernel_name}(void){{\n"
            for name in self.kernels.keys():
                src += f"  {mode}_{name}();\n"
            src += "}\n\n"
        return src

    def make_get_num_algos_def(self, meta: KernelLinkerMeta = None) -> str:
        meta = meta or self.meta
        src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
        src += f"  return (int)sizeof({meta.orig_kernel_name}_kernels);\n"
        src += "}\n"
        return src

    def make_default_algo_kernel_def(self, meta: KernelLinkerMeta = None) -> str:
        meta = meta or self.meta
        src = f"CUresult {meta.orig_kernel_name}_default(CUstream stream, {self.signature_generator.gen_signature_with_full_args(meta)}){{\n"
        src += f"  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n"
        src += "}\n"
        return src

    def generate(self):
        includes = "\n".join(self.SOURCE_INCLUDES)

        defs = self.make_defs()
        func_pointers_def = self.make_func_pointers()
        meta_const_def = self.make_kernel_meta_const_dispatcher()
        get_num_algos_def = self.make_get_num_algos_def()
        load_unload_def = self.make_kernel_load_defs()
        default_algo_kernel = self.make_default_algo_kernel_def()
        src = "\n".join(
            [
                defs,
                func_pointers_def,
                get_num_algos_def,
                meta_const_def,
                load_unload_def,
                default_algo_kernel,
            ]
        )
        return "\n\n".join([includes, src])


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
