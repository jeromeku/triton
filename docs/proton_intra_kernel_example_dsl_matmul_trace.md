# Trace: Gluon + Proton intra-kernel profiling (`example_dsl_matmul.py`)

This is a **frame-by-frame, end-to-end** trace of the Gluon Hopper GEMM tutorial:
[third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L1).

Your requirement (explicitly honored here):
- Trace the **entire compilation pipeline** (Python → C++ MLIR → back), including **ALL compilation passes**, not only Proton-specific ones.
- Trace into **C++ sources** where the Python API crosses into MLIR/LLVM/Proton.

## What this tutorial does (big picture)

- Builds a **blocked matmul** kernel in **Gluon** using Hopper primitives:
  - **TMA** async copies (`ttng.async_tma_copy_*`)
  - **mbarrier** synchronization
  - **WGMMA** (warp-group MMA, `ttng.warp_group_dot`)
- Wraps key regions in **Proton DSL scopes** (`pl.enter_scope`, `with pl.scope(...)`, `pl.exit_scope`).
- Starts Proton with `backend="instrumentation"` so Triton:
  - **injects** Proton lowering passes into the MLIR pipeline, and
  - enables the DSL builtins so `pl.*` becomes IR ops.
- Emits output files:
  - `gemm.hatchet` (op-measure mode) or
  - `gemm.chrome_trace` (timeline trace mode).

---

## Key files (map)

- [third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L1) — tutorial entrypoint + kernel.
- [python/triton/profiler/profile.py](../python/triton/profiler/profile.py#L50) — `proton.start()` / `proton.finalize()`.
- [python/triton/profiler/hooks/hook.py](../python/triton/profiler/hooks/hook.py#L33) — `HookManager` wires compile/load/launch hooks.
- [python/triton/profiler/hooks/instrumentation.py](../python/triton/profiler/hooks/instrumentation.py#L130) — registers **Proton lowering passes** into the compiler.
- [python/triton/profiler/language.py](../python/triton/profiler/language.py#L36) — `pl.enter_scope/exit_scope` → `triton_proton.create_proton_record(...)`.
- [third_party/proton/Dialect/triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L20) — C++: creates `proton::RecordOp` + exposes Proton pass constructors.
- [python/triton/experimental/gluon/_runtime.py](../python/triton/experimental/gluon/_runtime.py#L13) — `GluonJITFunction`, `GluonASTSource(ext="ttgir")`.
- [python/triton/runtime/jit.py](../python/triton/runtime/jit.py#L355) — `fn[grid](*args)` → `JITCallable.run(...)`.
- [python/triton/compiler/compiler.py](../python/triton/compiler/compiler.py#L226) — `compile(...)` runs the staged compilation pipeline.
- [python/triton/compiler/code_generator.py](../python/triton/compiler/code_generator.py#L274) — `CodeGenerator` uses **GluonOpBuilder** when `is_gluon=True`.
- [python/src/gluon_ir.cc](../python/src/gluon_ir.cc#L700) — C++: GluonOpBuilder emits TTGIR ops (WGMMA, TMA, mbarrier).
- [python/src/passes.cc](../python/src/passes.cc#L31) — C++: Python wrappers for core MLIR pass constructors.
- [third_party/nvidia/backend/compiler.py](../third_party/nvidia/backend/compiler.py#L325) — NVIDIA backend: `gluon_to_ttgir`, `make_llir`, `make_ptx`, `make_cubin`.
- [third_party/nvidia/triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L21) — C++: Python wrappers for NVIDIA-specific passes.
- [lib/Dialect/Gluon/Transforms/Inline.cpp](../lib/Dialect/Gluon/Transforms/Inline.cpp#L1) — C++: Gluon inliner pass.
- [lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp](../lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp#L99) — C++: layout inference for coalescing.
- [lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp](../lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp#L44) — C++: resolves `AutoEncoding` using `gluon.set_auto_layout`.
- [lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp](../lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp#L150) — C++: loop-aware CSE.
- [lib/Dialect/TritonGPU/Transforms/CombineTensorSelectAndIf.cpp](../lib/Dialect/TritonGPU/Transforms/CombineTensorSelectAndIf.cpp#L80) — C++: combines `arith.select` + `scf.if`.
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp#L23) — C++: allocate shared memory (NV) + offsets.
- [lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp](../lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp#L64) — C++: allocate warp-group IDs / total warps.
- [lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp](../lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp#L75) — C++: global scratch allocation.
- [lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp#L171) — C++: proxy fence insertion.
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#L239) — C++: TTGIR → LLVM conversion pass factory.
- [third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp](../third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp#L643) — C++: lowers WGMMA/TMA-ish NVGPU ops.
- [python/src/llvm.cc](../python/src/llvm.cc#L515) — C++: LLVM `optimize_module` + `translate_to_asm` bindings.
- [third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#L360) — C++: Proton `RecordOp` → ProtonGPU circular buffer.
- [third_party/proton/Dialect/lib/ProtonGPUToLLVM/AllocateProtonSharedMemory.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/AllocateProtonSharedMemory.cpp#L13) — C++: reserves shared memory for Proton buffer.
- [third_party/proton/Dialect/lib/ProtonGPUToLLVM/AllocateProtonGlobalScratchBuffer.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/AllocateProtonGlobalScratchBuffer.cpp#L13) — C++: allocates global scratch for profiling.
- [third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/ConvertProtonGPUToLLVM.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/ConvertProtonGPUToLLVM.cpp#L46) — C++: ProtonGPU → LLVM conversion.
- [third_party/proton/csrc/Proton.cpp](../third_party/proton/csrc/Proton.cpp#L61) — C++: `libproton` runtime API (`start/finalize/enter_instrumented_op`).

---

## Key functions index

| Function | File | Purpose |
|----------|------|---------|
| `blocked_matmul_pipelined_kernel` | [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L140) | Gluon kernel: TMA double-buffer + WGMMA; emits Proton scopes. |
| `blocked_matmul_pipelined` | [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L209) | Host wrapper: builds `TensorDescriptor`s and launches kernel. |
| `config_helper` | [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L35) | Builds `proton.mode.Default(...)` based on CLI flags. |
| `proton.start` / `proton.finalize` | [profile.py](../python/triton/profiler/profile.py#L50) | Starts/stops a libproton profiling session; registers hooks. |
| `HookManager.register` | [hook.py](../python/triton/profiler/hooks/hook.py#L92) | Installs compile/load/launch hooks into `triton.knobs.runtime.*`. |
| `InstrumentationHook.activate` | [instrumentation.py](../python/triton/profiler/hooks/instrumentation.py#L146) | Turns on instrumentation + injects Proton MLIR passes into backend. |
| `KernelInterface.__getitem__` | [jit.py](../python/triton/runtime/jit.py#L364) | Implements `fn[grid](...)` launch syntax. |
| `JITCallable.run` | [jit.py](../python/triton/runtime/jit.py#L706) | Compiles (if needed) then launches the kernel. |
| `compile` | [compiler.py](../python/triton/compiler/compiler.py#L226) | Runs the staged compilation pipeline (`ttgir → llir → ptx → cubin`). |
| `GluonASTSource.make_ir` | [gluon/_runtime.py](../python/triton/experimental/gluon/_runtime.py#L13) | Creates initial MLIR module and runs AST → TTGIR codegen. |
| `CodeGenerator.__init__` | [code_generator.py](../python/triton/compiler/code_generator.py#L274) | Picks Gluon vs Triton builder/semantic for codegen. |
| `CUDABackend.gluon_to_ttgir` | [nvidia/compiler.py](../third_party/nvidia/backend/compiler.py#L325) | **TTGIR stage** pipeline for Gluon kernels (explicit pass list). |
| `CUDABackend.make_llir` | [nvidia/compiler.py](../third_party/nvidia/backend/compiler.py#L344) | **LLIR stage** pipeline (explicit pass list + instrumentation hooks). |
| `llvm.optimize_module` / `llvm.translate_to_asm` | [llvm.cc](../python/src/llvm.cc#L515) | Runs LLVM’s O3 pipeline and emits PTX/obj. |

---

## Diagrams

### Sequence diagram (end-to-end)

```
Python main
  |
  |  proton.start(... backend="instrumentation")
  v
HookManager.register(InstrumentationHook)
  |  -> InstrumentationHook.activate()
  |     - flags.instrumentation_on = True
  |     - backend.compiler.instrumentation = Instrumentation(pass injectors)
  |     - knobs.compilation.instrumentation_mode = "cuda:default:..."
  v
blocked_matmul_pipelined(A,B,C,...)
  |  -> TensorDescriptor.from_tensor(...) x3
  v
blocked_matmul_pipelined_kernel[grid](a_desc,b_desc,c_desc,...)
  |
  |  KernelInterface.__getitem__ returns launcher -> JITCallable.run(...)
  v
compile(src=GluonASTSource, language=GLUON, ext="ttgir")
  |
  |  src.make_ir(): AST -> initial TTGIR using GluonOpBuilder (C++)
  |  stage "ttgir": gluon_to_ttgir() pass pipeline
  |  stage "llir" : make_llir() pass pipeline (+ injected Proton passes)
  |  MLIR -> LLVM IR -> llvm.optimize_module(O3) -> PTX -> ptxas -> CUBIN
  v
kernel.run(..., launch_enter_hook, launch_exit_hook)
  |
  |  HookManager.enter -> InstrumentationHook.enter -> libproton.enter_instrumented_op
  |  device executes kernel; Proton RecordOps write timing entries to buffer
  |  HookManager.exit  -> InstrumentationHook.exit  -> libproton.exit_instrumented_op
  v
proton.finalize()
  |
  v
libproton.finalize(...) writes gemm.hatchet or gemm.chrome_trace
```

### Data flow (A/B/C → descriptors → IR → binary → trace files)

```
PyTorch tensors (A,B,C) [device fp16]
  |
  | TensorDescriptor.from_tensor(base, shape, strides, block_shape, layout)
  v
Host descriptor objects (Python dataclasses)
  |
  | passed as kernel args
  v
TTGIR (MLIR ModuleOp)
  - ttng.async_tma_copy_* ops
  - ttng.warp_group_dot / wait ops
  - proton.record ops (from pl.enter_scope / pl.scope / pl.exit_scope)
  |
  | (explicit MLIR pass pipelines; see below)
  v
LLIR (MLIR LLVM dialect) -> LLVM IR (string) -> PTX -> CUBIN
  |
  | runtime launch hooks write/read profiling buffer
  v
Output artifacts:
  - gemm.chrome_trace  (timeline)
  - gemm.hatchet       (tree/metrics)
```

### Module relationships (who calls whom)

```
example_dsl_matmul.py
  |-- proton.start() -------------------------------> python/triton/profiler/profile.py
  |                                                   |-- HookManager.register()
  |                                                   |-- libproton.start()  (C++)
  |
  |-- blocked_matmul_pipelined() -------------------> TensorDescriptor.from_tensor()
  |
  |-- blocked_matmul_pipelined_kernel[grid](...) ---> KernelInterface.__getitem__()
                                                      |-- JITCallable.run()
                                                           |-- compiler.compile()
                                                                |-- GluonASTSource.make_ir()
                                                                |-- CUDABackend.gluon_to_ttgir()  (passes)
                                                                |-- CUDABackend.make_llir()       (passes + Proton inject)
                                                                |-- llvm.optimize_module()        (LLVM passes)
                                                                |-- llvm.translate_to_asm()       (PTX)
                                                                |-- ptxas                         (cubin)
```

---

## Trace (frame-by-frame)

### Frame 0 — `__main__`: configure + start profiling + run GEMM

Source: [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L223)

```python
if __name__ == "__main__":
    if not is_hopper():
        raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

    M, N, K = 512, 512, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

    op_measure, mode = config_helper(description)

    if op_measure:
        proton.start("gemm", backend="instrumentation", mode=mode)
    else:
        proton.start("gemm", data="trace", backend="instrumentation", mode=mode)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)

    proton.finalize()
```

**State before**
- No profiling session.
- `flags.instrumentation_on == False` (so `pl.*` builtins would be no-ops in IR).

**State after `proton.start(... backend="instrumentation")`**
- A libproton session exists and is active.
- Triton compilation is now in **instrumentation mode**:
  - DSL builtins emit Proton IR ops.
  - Backend compilation pipelines will have Proton passes injected.

---

### Frame 1 — `config_helper`: build an instrumentation mode string

Source: [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L35)

Key point: this returns a `proton.mode.Default(...)` object, which is later stringified and stored in `triton.knobs.compilation.instrumentation_mode`.

```python
if args.warp_sampling:
    mode = proton.mode.Default(
        optimizations=opts,
        sampling_strategy="selective",
        sampling_options=args.warp_ids,
        buffer_type=buf,
    )
else:
    mode = proton.mode.Default(optimizations=opts, buffer_type=buf)
```

**Invariant**
- `mode` is purely a configuration object until `InstrumentationHook.activate()` serializes it into a string and installs the corresponding compiler passes.

---

### Frame 2 — `proton.start`: register hooks + enable instrumentation

#### 2.1 Python: `proton.start` calls into libproton and installs hooks

Source: [profile.py](../python/triton/profiler/profile.py#L50)

```python
session = libproton.start(name, context, data, backend, mode_str)

if backend == "instrumentation":
    HookManager.register(InstrumentationHook(mode), session)
```

**State change**
- `libproton.start(...)` (C++) creates a new profiling session.
- `HookManager.register(...)` immediately calls `InstrumentationHook.activate()`.

#### 2.2 Python: `HookManager.register` wires hooks into Triton runtime knobs

Source: [hook.py](../python/triton/profiler/hooks/hook.py#L92)

```python
HookManager.session_hooks[session][hook] = True
if hook not in HookManager.active_hooks:
    hook.activate()
    HookManager.active_hooks.append(hook)

knobs.runtime.kernel_load_end_hook.add(HookManager.init_handle)
knobs.runtime.launch_enter_hook.add(HookManager.enter)
knobs.runtime.launch_exit_hook.add(HookManager.exit)
```

This is where the later `kernel_load_end_hook` and `launch_enter/exit_hook` callbacks come from.

#### 2.3 Python: `InstrumentationHook.activate` injects Proton passes into compilation

Source: [instrumentation.py](../python/triton/profiler/hooks/instrumentation.py#L146)

```python
flags.instrumentation_on = True

def to_llvmir_passes(pm):
    triton_proton.add_convert_proton_to_protongpu(...)
    triton_passes.common.add_cse(pm)
    triton_proton.add_allocate_proton_shared_memory(pm)

def to_llvm_passes(pm):
    triton_proton.add_allocate_proton_global_scratch_buffer(pm)
    triton_proton.add_convert_proton_nvidia_gpu_to_llvm(pm)

backends[backend_name].compiler.instrumentation = Instrumentation({
    "ttgpuir_to_llvmir": lambda pm: to_llvmir_passes(pm),
    "llvmir_to_llvm": lambda pm: to_llvm_passes(pm),
})

triton.knobs.compilation.instrumentation_mode = str(self.mode)
```

**Important implications**
- `flags.instrumentation_on = True` means **inside the kernel AST**, calls like `pl.enter_scope("...")` now emit IR ops (they don’t early-return).
- The backend compiler class (e.g. NVIDIA `CUDABackend`) gains an `instrumentation` object that can **patch pass managers** at specific points.

#### 2.4 C++: `pl.enter_scope` becomes a `proton::RecordOp`

Python lowering site: [python/triton/profiler/language.py](../python/triton/profiler/language.py#L36)

```python
return tl.tensor(
    triton_proton.create_proton_record(semantic.builder, is_start, scope_name),
    tl.void
)
```

C++ binding: [third_party/proton/Dialect/triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L75)

```c++
m.def("create_proton_record",
      [](TritonOpBuilder &opBuilder, bool isStart, const std::string &name) {
        auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(), name);
        opBuilder.create<proton::RecordOp>(isStart, nameAttr);
      });
```

So every `pl.enter_scope/exit_scope` in the Gluon kernel’s AST produces **Proton dialect** ops at codegen time.

---

### Frame 3 — host wrapper: build `TensorDescriptor`s and launch kernel

Source: [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L209)

```python
a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
...
a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
...
blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)
```

#### 3.1 How layouts are chosen (compile-time shape → swizzle)

Source: [python/triton/experimental/gluon/language/_layouts.py](../python/triton/experimental/gluon/language/_layouts.py#L394)

```python
if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
    swizzle_byte_width = 128
elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
    swizzle_byte_width = 64
...
```

This chooses the largest compatible swizzle to reduce the number of TMA/MMA messages.

#### 3.2 Host descriptor validation (alignment + block shape)

Source: [python/triton/experimental/gluon/nvidia/hopper.py](../python/triton/experimental/gluon/nvidia/hopper.py#L10)

```python
assert self.base.data_ptr() % 16 == 0
assert self.strides[-1] == 1
assert isinstance(self.layout, NVMMASharedLayout)
```

**Invariant**
- The descriptor’s layout+alignment constraints must match what TMA expects (16B alignment, contiguous inner dim).

---

### Frame 4 — kernel launch syntax: `fn[grid](...)` → `JITCallable.run(...)`

#### 4.1 `__getitem__` records the grid

Source: [python/triton/runtime/jit.py](../python/triton/runtime/jit.py#L364)

```python
def __getitem__(self, grid) -> T:
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

#### 4.2 `run(...)` compiles (if needed) then launches

Source: [python/triton/runtime/jit.py](../python/triton/runtime/jit.py#L706)

```python
kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
...
kernel = kernel_cache.get(key, None)
if kernel is None:
    kernel = self._do_compile(...)
...
kernel.run(..., knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
```

**State note**
- `instrumentation_mode` becomes part of backend options, which can affect:
  - pass pipelines (Proton instrumentation),
  - debug/sanitizers,
  - allocations (profiling buffers).

---

## Frame 5 — compilation pipeline (ALL stages + ALL explicit MLIR passes)

This section is the “no hand-waving” part: it enumerates every **explicit** compilation stage and every **explicit** MLIR pass installed by Triton + the instrumentation hook for this Gluon kernel.

### 5.1 Stage driver: `compiler.compile(...)` runs per-stage transforms

Source: [python/triton/compiler/compiler.py](../python/triton/compiler/compiler.py#L226)

```python
stages = dict()
backend.add_stages(stages, options, src.language)
...
module = src.make_ir(target, options, codegen_fns, module_map, context)
...
for ext, compile_ir in list(stages.items())[first_stage:]:
    next_module = compile_ir(module, metadata)
    module = next_module
```

### 5.2 NVIDIA backend stage list for Gluon (`ttgir → llir → ptx → cubin`)

Source: [third_party/nvidia/backend/compiler.py](../third_party/nvidia/backend/compiler.py#L545)

```python
elif language == Language.GLUON:
    stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(...)
stages["llir"] = lambda src, metadata: self.make_llir(...)
stages["ptx"] = lambda src, metadata: self.make_ptx(...)
stages["cubin"] = lambda src, metadata: self.make_cubin(...)
```

### 5.3 Stage `ttgir`: `gluon_to_ttgir` — pass list (complete)

Source: [third_party/nvidia/backend/compiler.py](../third_party/nvidia/backend/compiler.py#L325)

```python
passes.gluon.add_inliner(pm)
passes.gluon.add_infer_coalesced_encodings(pm)
passes.gluon.add_resolve_auto_encodings(pm)
nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
passes.gluon.add_canonicalizer(pm)
passes.common.add_sccp(pm)
passes.ttir.add_loop_aware_cse(pm)
passes.gluon.add_canonicalizer(pm)
passes.ttgpuir.add_combine_tensor_select_and_if(pm)
```

### 5.4 Stage `llir`: `make_llir` — pass list (complete)

Source: [third_party/nvidia/backend/compiler.py](../third_party/nvidia/backend/compiler.py#L344)

```python
passes.ttgpuir.add_combine_tensor_select_and_if(pm)
passes.ttgpuir.add_allocate_warp_groups(pm)
passes.convert.add_scf_to_cf(pm)
passes.gluon.add_inliner(pm)
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)

if "consan" in options.instrumentation_mode:
    passes.ttgpuir.add_concurrency_sanitizer(pm)

passes.ttgpuir.add_allocate_global_scratch_memory(pm)
nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)

# --- Proton injection point #1 (TTGIR→LLVMIR) ---
if CUDABackend.instrumentation:
    CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)

nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
passes.common.add_canonicalizer(pm)
passes.common.add_cse(pm)
nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
passes.common.add_canonicalizer(pm)
passes.common.add_cse(pm)
passes.common.add_symbol_dce(pm)
passes.convert.add_nvvm_to_llvm(pm)
passes.llvmir.add_di_scope(pm)  # (guarded in source)

# --- Proton injection point #2 (LLVMIR→LLVM) ---
if CUDABackend.instrumentation:
    CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
```

### 5.5 Proton-injected pass lists (complete)

Source: [python/triton/profiler/hooks/instrumentation.py](../python/triton/profiler/hooks/instrumentation.py#L158)

```python
def to_llvmir_passes(pm):
    triton_proton.add_convert_proton_to_protongpu(...)
    triton_passes.common.add_cse(pm)
    triton_proton.add_allocate_proton_shared_memory(pm)

def to_llvm_passes(pm):
    triton_proton.add_allocate_proton_global_scratch_buffer(pm)
    triton_proton.add_convert_proton_nvidia_gpu_to_llvm(pm)
```

At runtime, these are inserted exactly at the two patch points shown in `make_llir`.

---

## Frame 6 — Gluon DSL codegen: Python AST → TTGIR ops (C++ builder)

### 6.1 Gluon AST source: `ext="ttgir"` and language=GLUON

Source: [python/triton/experimental/gluon/_runtime.py](../python/triton/experimental/gluon/_runtime.py#L13)

```python
class GluonASTSource(ASTSource):
    def __init__(...):
        self.language = Language.GLUON
        self.ext = "ttgir"
```

### 6.2 AST → TTGIR: CodeGenerator uses GluonOpBuilder (C++)

Source: [python/triton/compiler/code_generator.py](../python/triton/compiler/code_generator.py#L276)

```python
if is_gluon:
    self.builder = gluon_ir.GluonOpBuilder(context)
    self.semantic = GluonSemantic(self.builder)
```

This is a hard boundary:
- **Python** walks the kernel’s AST, but
- **C++** (`GluonOpBuilder`) creates MLIR ops as the semantic layer requests them.

### 6.3 Compile-time constexpr functions inside the kernel

The kernel calls `pick_wgmma_layout(...)` as a `gl.constexpr`. That runs at compile-time (during AST visiting), not at runtime on GPU.

Source: [python/triton/runtime/jit.py](../python/triton/runtime/jit.py#L1089)

```python
def __call__(self, *args, _semantic=None, **kwargs):
    args = [_unwrap_if_constexpr(x) for x in args]
    res = self.fn(*args, **kwargs)
    if _semantic is None:
        return res
    return constexpr(res)
```

So `pick_wgmma_layout(...)` returns a *Python object* representing a layout, then gets wrapped as a Triton `constexpr` for IR consumption.

### 6.4 Kernel body: Gluon ops + Proton scopes

Source: [example_dsl_matmul.py](../third_party/proton/tutorials/intra_kernel/example_dsl_matmul.py#L140)

```python
pl.enter_scope("blocked_matmul_pipelined_kernel")

a_smem = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
b_smem = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)

bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
mbarrier.init(bar, count=1)

for k in range(0, K, BLOCK_K):
    a = a_smem.index(index)
    b = b_smem.index(index)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)

    with pl.scope("tma_loads_issue"):
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b)
    ...
    with pl.scope("wgmma_issue"):
        acc = warpgroup_mma(a, b, acc, is_async=True)
    index ^= 1

with pl.scope("wgmma_last_wait"):
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
c_smem.store(acc.to(dtype))
fence_async_shared()
tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
tma.store_wait(pendings=0)

pl.exit_scope("blocked_matmul_pipelined_kernel")
```

Below are the critical “Gluon DSL → C++ builder → MLIR op” mappings.

#### 6.4.1 `gl.allocate_shared_memory` → `_semantic.allocate_shared(...)`

Source: [python/triton/experimental/gluon/language/_core.py](../python/triton/experimental/gluon/language/_core.py#L486)

```python
return _semantic.allocate_shared(element_ty, shape, layout, value)
```

This eventually emits TTGIR allocations (`ttg.local_alloc` / memdesc constructs) via the semantic builder.

#### 6.4.2 `a_smem.index(i)` → `memdesc_index` → C++ `create_memdesc_index`

Python: [python/triton/experimental/gluon/language/_core.py](../python/triton/experimental/gluon/language/_core.py#L338)

```python
@builtin
def index(self, index, _semantic: GluonSemantic = None):
    return _semantic.memdesc_index(self, index)
```

C++ builder binding: [python/src/gluon_ir.cc](../python/src/gluon_ir.cc#L702)

```c++
.def("create_memdesc_index",
     [](GluonOpBuilder &self, Type resultType, Value src, Value index) -> Value {
       return self.create<ttg::MemDescIndexOp>(resultType, src, index);
     })
```

#### 6.4.3 `warpgroup_mma(...)` → C++ `create_warpgroup_mma` → `ttng::WarpGroupDotOp`

Python wrapper: [python/triton/experimental/gluon/language/nvidia/hopper/__init__.py](../python/triton/experimental/gluon/language/nvidia/hopper/__init__.py#L75)

```python
handle = _semantic.builder.create_warpgroup_mma(..., is_async)
```

C++ builder binding: [python/src/gluon_ir.cc](../python/src/gluon_ir.cc#L728)

```c++
.def("create_warpgroup_mma",
     [](GluonOpBuilder &self, Value a, Value b, Value acc, Value useAcc,
        ... bool isAsync) -> Value {
       return self.create<ttng::WarpGroupDotOp>(a, b, acc, useAcc, ..., isAsync);
     })
```

Later, the NVIDIA LLVM lowering step rewrites these into PTX asm (see `ConvertNVGPUToLLVM` patterns below).

#### 6.4.4 `tma.async_copy_global_to_shared(...)` → C++ `create_async_tma_copy_global_to_local`

Python wrapper: [python/triton/experimental/gluon/language/nvidia/hopper/tma.py](../python/triton/experimental/gluon/language/nvidia/hopper/tma.py#L107)

```python
_semantic.builder.create_async_tma_copy_global_to_local(...)
```

C++ builder binding: [python/src/gluon_ir.cc](../python/src/gluon_ir.cc#L827)

```c++
.def("create_async_tma_copy_global_to_local",
     [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &coord,
        Value barrier, Value result, Value pred, bool multicast) {
       self.create<ttng::AsyncTMACopyGlobalToLocalOp>(descPtr, coord, barrier, result, pred, multicast);
     })
```

---

## Frame 7 — “ALL compilation passes” cross-reference (Python call → C++ pass)

This is the mapping from each Python `passes.*.add_*` call to its C++ implementation entrypoint.

### 7.1 Pass wrapper entrypoints (C++)

Core pass wrappers: [python/src/passes.cc](../python/src/passes.cc#L31)
- This is where `passes.common.add_*`, `passes.ttir.add_*`, `passes.ttgpuir.add_*`, `passes.gluon.add_*`, etc. are bound to C++ `create*Pass` functions.

NVIDIA pass wrappers: [third_party/nvidia/triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L21)
- This is where `nvidia.passes.*.add_*` are bound to NV-specific pass factories.

Proton pass wrappers: [third_party/proton/Dialect/triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L84)
- This is where `triton_proton.add_*` are bound to Proton pass factories.

### 7.2 Stage `ttgir` passes (Gluon): what each one is

1) `passes.gluon.add_inliner`
- Wrapper: [passes.cc](../python/src/passes.cc#L141)
- C++ pass: `gluon-inline` → [Inline.cpp](../lib/Dialect/Gluon/Transforms/Inline.cpp#L1)

2) `passes.gluon.add_infer_coalesced_encodings`
- Wrapper: [passes.cc](../python/src/passes.cc#L141)
- C++ pass: `gluon-infer-coalesced-encodings` → [InferCoalescedEncodings.cpp](../lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp#L99)

3) `passes.gluon.add_resolve_auto_encodings`
- Wrapper: [passes.cc](../python/src/passes.cc#L141)
- C++ pass: `gluon-resolve-auto-encodings` → [ResolveAutoEncodings.cpp](../lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp#L44)

4) `nvidia.passes.ttnvgpuir.add_tma_lowering`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L51)
- C++ pass: `triton-nvidia-tma-lowering` → [TMALowering.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp#L20)

5) `passes.gluon.add_canonicalizer` (runs twice)
- Wrapper: [passes.cc](../python/src/passes.cc#L141)
- C++ pass: Gluon canonicalize → [Canonicalize.cpp](../lib/Dialect/Gluon/Transforms/Canonicalize.cpp#L26)

6) `passes.common.add_sccp`
- Wrapper: [passes.cc](../python/src/passes.cc#L31)
- C++ pass: MLIR `createSCCPPass` (upstream MLIR)

7) `passes.ttir.add_loop_aware_cse`
- Wrapper: [passes.cc](../python/src/passes.cc#L42)
- C++ pass: `triton-loop-aware-cse` → [LoopAwareCSE.cpp](../lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp#L150)

8) `passes.ttgpuir.add_combine_tensor_select_and_if`
- Wrapper: [passes.cc](../python/src/passes.cc#L58)
- C++ pass: `tritongpu-combine-tensor-select-and-if` → [CombineTensorSelectAndIf.cpp](../lib/Dialect/TritonGPU/Transforms/CombineTensorSelectAndIf.cpp#L80)

### 7.3 Stage `llir` passes: what each one is

1) `passes.ttgpuir.add_combine_tensor_select_and_if`
- Same as above → [CombineTensorSelectAndIf.cpp](../lib/Dialect/TritonGPU/Transforms/CombineTensorSelectAndIf.cpp#L80)

2) `passes.ttgpuir.add_allocate_warp_groups`
- Wrapper: [passes.cc](../python/src/passes.cc#L84)
- C++ pass: `tritongpu-allocate-warp-groups` → [AllocateWarpGroups.cpp](../lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp#L64)

3) `passes.convert.add_scf_to_cf`
- Wrapper: [passes.cc](../python/src/passes.cc#L126)
- C++ pass: MLIR `createSCFToControlFlowPass` (upstream MLIR)

4) `passes.gluon.add_inliner`
- Same as above → [Inline.cpp](../lib/Dialect/Gluon/Transforms/Inline.cpp#L1)

5) `nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(capability, ptxVersion)`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L21)
- C++ pass: `AllocateSharedMemoryNv` → [Allocation.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp#L23)

6) `nvidia.passes.ttnvgpuir.add_allocate_tensor_memory`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L51)
- C++ pass: tensor memory allocation → [TensorMemoryAllocation.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp#L391)

7) `nvidia.passes.ttnvgpuir.add_check_matmul_two_cta`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L51)
- C++ pass: [CheckMatmulTwoCTAs.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp#L18)

8) `passes.ttgpuir.add_concurrency_sanitizer` (only if `"consan"` is enabled)
- Wrapper: [passes.cc](../python/src/passes.cc#L58)
- C++ pass: [ConcurrencySanitizer.cpp](../lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp#L154)

9) `passes.ttgpuir.add_allocate_global_scratch_memory`
- Wrapper: [passes.cc](../python/src/passes.cc#L58)
- C++ pass: [GlobalScratchMemoryAllocation.cpp](../lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp#L75)

10) `nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(capability)`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L44)
- C++ pass: [ProxyFenceInsertion.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp#L171)

11) **(Injected)** `triton_proton.add_convert_proton_to_protongpu(...)`
- Wrapper: [triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L84)
- C++ pass: [ProtonToProtonGPUPass.cpp](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#L398)

12) **(Injected)** `triton_passes.common.add_cse`
- Wrapper: [passes.cc](../python/src/passes.cc#L31)
- C++ pass: MLIR `createCSEPass` (upstream MLIR)

13) **(Injected)** `triton_proton.add_allocate_proton_shared_memory`
- Wrapper: [triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L99)
- C++ pass: [AllocateProtonSharedMemory.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/AllocateProtonSharedMemory.cpp#L13)

14) `nvidia.passes.ttgpuir.add_to_llvmir(capability, ptxVersion)`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L21)
- C++ pass: Convert TTGIR → LLVM (MLIR) → [TritonGPUToLLVM.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#L239)

15) `passes.common.add_canonicalizer`
- Wrapper: [passes.cc](../python/src/passes.cc#L31)
- C++ pass: MLIR `createCanonicalizerPass` (upstream MLIR)

16) `passes.common.add_cse`
- Wrapper: [passes.cc](../python/src/passes.cc#L31)
- C++ pass: MLIR `createCSEPass` (upstream MLIR)

17) `nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L65)
- C++ pass: `convert-nv-gpu-to-llvm` → [NVGPUToLLVMPass.cpp](../third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp#L643)

18) `nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm`
- Wrapper: [triton_nvidia.cc](../third_party/nvidia/triton_nvidia.cc#L67)
- C++ pass: `convert-warp-specialize-to-llvm` → [ConvertWarpSpecializeToLLVM.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp#L248)

19) `passes.common.add_symbol_dce`
- Wrapper: [passes.cc](../python/src/passes.cc#L31)
- C++ pass: MLIR `createSymbolDCEPass` (upstream MLIR)

20) `passes.convert.add_nvvm_to_llvm`
- Wrapper: [passes.cc](../python/src/passes.cc#L126)
- C++ pass: MLIR `createConvertNVVMToLLVMPass` (upstream MLIR)

21) `passes.llvmir.add_di_scope` (if enabled)
- Wrapper: [passes.cc](../python/src/passes.cc#L135)
- C++ pass: `mlir::createLLVMDIScope` (upstream MLIR)

22) **(Injected)** `triton_proton.add_allocate_proton_global_scratch_buffer`
- Wrapper: [triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L106)
- C++ pass: [AllocateProtonGlobalScratchBuffer.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/AllocateProtonGlobalScratchBuffer.cpp#L13)

23) **(Injected)** `triton_proton.add_convert_proton_nvidia_gpu_to_llvm`
- Wrapper: [triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L99)
- C++ pass: [ConvertProtonGPUToLLVM.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/ConvertProtonGPUToLLVM.cpp#L46)

---

## Frame 8 — LLVM pipeline + PTX emission (C++ bindings)

After `make_llir`, the NVIDIA backend does:
1) MLIR LLVM dialect → LLVM IR module (`llvm.to_module`)
2) attach data layout / link extern libs
3) **LLVM optimization pipeline** (O3) via `llvm.optimize_module`
4) emit PTX via `llvm.translate_to_asm`

### 8.1 LLVM optimize pipeline entrypoint (C++)

Source: [python/src/llvm.cc](../python/src/llvm.cc#L515)

```c++
m.def("optimize_module", [](llvm::Module *mod, const llvm::OptimizationLevel &opt, ...) {
  ...
  PassBuilder pb(...);
  ModulePassManager mpm;
  pb.registerVectorizerStartEPCallback([&](llvm::FunctionPassManager &fpm, ...) {
    fpm.addPass(BreakStructPhiNodesPass());
    fpm.addPass(InstCombinePass());
  });
  mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
  mpm.run(*mod, mam);
});
```

**Note on “ALL passes”**
- Triton explicitly enumerates the **MLIR** passes (Frames 5–7).
- LLVM’s `buildPerModuleDefaultPipeline(O3)` expands to a large internal pass pipeline owned by LLVM. Triton triggers it here, but the individual pass list lives in LLVM, not in this repository.

### 8.2 PTX emission entrypoint (C++)

Source: [python/src/llvm.cc](../python/src/llvm.cc#L644)

```c++
m.def("translate_to_asm", [](std::string llvmIR, std::string triple, std::string proc, ...) {
  ...
  obj = translateLLVMIRToASM(*module, triple, proc, features, flags, ...);
});
```

---

## Frame 9 — runtime capture: launch hooks → `libproton` (C++)

### 9.1 Launch enter/exit call into libproton

Source: [python/triton/profiler/hooks/instrumentation.py](../python/triton/profiler/hooks/instrumentation.py#L253)

```python
def enter(self, metadata: LazyDict) -> None:
    libproton.enter_instrumented_op(stream, func, self._data_ptr(), alloc_size)

def exit(self, metadata: LazyDict) -> None:
    libproton.exit_instrumented_op(stream, func, self._data_ptr(), alloc_size)
```

### 9.2 C++: libproton bindings dispatch to `SessionManager`

Source: [third_party/proton/csrc/Proton.cpp](../third_party/proton/csrc/Proton.cpp#L61)

```c++
m.def("start", [](const std::string &path, ..., const std::string &mode) {
  auto sessionId = SessionManager::instance().addSession(...);
  SessionManager::instance().activateSession(sessionId);
  return sessionId;
});

m.def("enter_instrumented_op", [](uint64_t streamId, uint64_t functionId, uint64_t buffer, size_t size) {
  SessionManager::instance().enterInstrumentedOp(streamId, functionId, reinterpret_cast<uint8_t *>(buffer), size);
});
```

**State**
- The profiling buffer pointer comes from `CudaAllocator` (a `torch.empty(uint8)` allocation) and is passed down as an integer pointer.
- The kernel’s in-device Proton writes target this buffer; `libproton` uses metadata to interpret it at finalize time.

---

## Appendix — one concrete “Gluon op → pass → lowered op” example (TMA)

Even though the kernel uses `ttng.async_tma_copy_*` ops directly, the compilation pipeline still includes a TMA lowering pass which is responsible for rewriting various descriptor/TMA constructs into the async+barrier form.

Source: [TMALowering.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp#L26)

```c++
InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);
BarrierExpectOp::create(rewriter, loc, barrierAlloc, sizeInBytes, pred);
AsyncTMACopyGlobalToLocalOp::create(..., barrierAlloc, alloc, pred);
WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase);
InvalBarrierOp::create(rewriter, loc, barrierAlloc);
```

This is the canonical “TMA load”: allocate destination, set up barrier, issue async copy, wait, invalidate barrier.
