# Trace: `example_dsl.py` (Proton DSL intra-kernel profiling)
This document traces the end-to-end execution path of:

- [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L1)

It covers both the **Triton DSL vector add** and the **Gluon DSL Hopper GEMM** examples, and follows the profiling signal from:

`pl.scope(...)` / `pl.enter_scope(...)` in Python → `proton.record` MLIR ops → ProtonGPU lowering → runtime buffer capture → host-side parsing → `.hatchet` / `.chrome_trace` output.

---

## Big Picture (What this tutorial does)

When you run `python3 example_dsl.py`, the file executes **two** `__main__` blocks:

1. A Triton DSL `add_kernel` with embedded Proton scopes (`pl.enter_scope`, `pl.scope`, `pl.exit_scope`)
2. A Hopper-only Gluon DSL `blocked_matmul_pipelined_kernel` with embedded Proton scopes around TMA + WGMMA steps

Both runs:

- Parse CLI args via `config_helper(...)`
- Start Proton in **instrumentation** backend mode
- Launch a kernel
- Finalize Proton (which dumps trace output files)

---

## Key Files (clickable)

- [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L1) — the tutorial entrypoint
- [proton/language.py](../third_party/proton/proton/language.py#L1) — `pl.enter_scope`, `pl.scope`, `pl.enable_semantic`
- [proton/profile.py](../third_party/proton/proton/profile.py#L50) — `proton.start()`, `proton.finalize()`
- [proton/hooks/hook.py](../third_party/proton/proton/hooks/hook.py#L8) — `HookManager` and hook chains
- [proton/hooks/instrumentation.py](../third_party/proton/proton/hooks/instrumentation.py#L118) — compilation + runtime hook for instrumentation
- [runtime/jit.py](../python/triton/runtime/jit.py#L706) — `JITFunction.run(...)` passes launch hooks into the launcher
- [compiler/compiler.py](../python/triton/compiler/compiler.py#L226) — compilation pipeline + `metadata_group` artifacts + kernel load hooks
- [language/core.py](../python/triton/language/core.py#L34) — `@builtin` requires `_semantic` (compile-time injection)
- [compiler/code_generator.py](../python/triton/compiler/code_generator.py#L274) — where `_semantic` is injected + how `with ...:` is lowered

**C++ / MLIR / runtime**

- [Dialect/triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L20) — pybind `create_proton_record(...)`
- [ProtonOps.td](../third_party/proton/Dialect/include/Dialect/Proton/IR/ProtonOps.td#L16) — `proton.record` op definition
- [ScopeIdAllocation.cpp](../third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp#L48) — pairs start/end + validates scopes + computes parentage
- [ProtonToProtonGPUPass.cpp](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#L74) — lowers `proton.record` → `proton.gpu.circular_store`
- [Utility.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/Utility.cpp#L68) — packs scope-id + clock into 2×u32 record and computes writer predicate
- [TargetInfo.cpp (NVIDIA)](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.cpp#L12) — reads `%clock`/`%clock_hi`
- [NvidiaPatternProtonGPUOpToLLVM.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.cpp#L27) — lowers `circular_store` to PTX `st.global` / shared stores

**Proton C++ runtime + parsing**

- [Proton.cpp](../third_party/proton/csrc/Proton.cpp#L61) — `libproton.start`, `enter_instrumented_op`, `exit_instrumented_op`
- [Session.cpp](../third_party/proton/csrc/lib/Session/Session.cpp#L177) — `SessionManager` dispatch
- [InstrumentationProfiler.cpp](../third_party/proton/csrc/lib/Profiler/Instrumentation/InstrumentationProfiler.cpp#L47) — parses mode options + copies/decodes buffer
- [CircularLayoutParser.cpp](../third_party/proton/common/lib/TraceDataIO/CircularLayoutParser.cpp#L38) — turns records into `(start,end)` events
- [EntryDecoder.cpp](../third_party/proton/common/lib/TraceDataIO/EntryDecoder.cpp#L27) — decodes packed 64b cycle + scope id + start/end

---

## Key Functions Index

| Function / Symbol | File | Purpose |
|---|---|---|
| `config_helper()` | [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L35) | CLI → `(op_measure, mode)` for instrumentation backend |
| `add_kernel()` | [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L100) | Triton vector add kernel with nested Proton scopes |
| `blocked_matmul_pipelined_kernel()` | [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L204) | Gluon Hopper matmul kernel with scoped TMA/WGMMA phases |
| `pl.enable_semantic()` | [proton/language.py](../third_party/proton/proton/language.py#L28) | Enables Triton semantic gating for Proton DSL in Triton kernels |
| `pl.enter_scope()/exit_scope()` | [proton/language.py](../third_party/proton/proton/language.py#L45) | Builtins that emit `proton.record` ops during compilation |
| `proton.start()/finalize()` | [proton/profile.py](../third_party/proton/proton/profile.py#L50) | Starts session, registers hooks, finalizes and dumps output |
| `HookManager.register()` | [proton/hooks/hook.py](../third_party/proton/proton/hooks/hook.py#L92) | Installs kernel-load + launch hooks into `triton.knobs.runtime` |
| `InstrumentationHook.activate()` | [proton/hooks/instrumentation.py](../third_party/proton/proton/hooks/instrumentation.py#L146) | Enables instrumentation flags and injects compiler pass hooks |
| `create_proton_record()` | [triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L76) | C++: emits `proton::RecordOp` into IR builder |
| `replaceProtonRecordOp()` | [ProtonToProtonGPUPass.cpp](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#L74) | C++: `proton.record` → `ReadCounterOp` + `CircularStoreOp` |
| `lowerCircularStoreOpHelper()` | [Utility.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/Utility.cpp#L68) | C++: encodes scope+clock record and computes store predicate |
| `InstrumentationProfiler::exitInstrumentedOp()` | [InstrumentationProfiler.cpp](../third_party/proton/csrc/lib/Profiler/Instrumentation/InstrumentationProfiler.cpp#L181) | Copies device buffer → host and parses to events |

---

## Execution Flow Diagrams

### 1) Sequence diagram (Python → compile → GPU → runtime parse)

```text
Python (example_dsl.py)
  |
  |  proton.start(... backend="instrumentation" ...)
  v
proton/profile.py:start()
  |
  |  HookManager.register(InstrumentationHook)
  v
HookManager (registers knobs.runtime hooks)
  |
  |  InstrumentationHook.activate()
  |    - flags.instrumentation_on = True
  |    - backends[nvidia].compiler.instrumentation = Instrumentation(...)
  v
Kernel launch (JITFunction.run)
  |
  |  (first time) compile() → artifacts (*.ttgir, *.json, *.cubin/hsaco)
  |    - pl.enter_scope emits proton.record ops during AST→IR
  |    - instrumentation.patch adds Proton lowering passes
  |
  |  load_binary() → knobs.runtime.kernel_load_end_hook(...)
  |    HookManager.init_handle(...) → InstrumentationHook.init_handle(...)
  |      - parse TTGIR
  |      - compute scope ids / parents
  |      - libproton.init_function_metadata(...)
  |
  |  launcher.run(... launch_enter_hook, launch_exit_hook ...)
  |    launch_enter_hook → HookManager.enter(...) → libproton.enter_instrumented_op(...)
  |    [GPU runs instrumented kernel: circular_store writes records]
  |    launch_exit_hook  → HookManager.exit(...)  → libproton.exit_instrumented_op(...)
  |                                       |
  |                                       v
  |                         InstrumentationProfiler::exitInstrumentedOp
  |                           - sync stream
  |                           - copy buffer → host
  |                           - parse circular layout → events
  v
proton.finalize() → dump .hatchet or .chrome_trace
```

### 2) Flowchart (how data transforms)

```text
pl.scope("X") inside kernel
   |
   v
Proton DSL builtin emits
  proton.record start/end "X"
   |
   v
ScopeIdAllocation:
  name pairs → numeric scope ids
  nesting → parent ids
   |
   v
ConvertProtonToProtonGPU:
  proton.record → (read_counter + circular_store(scope_id,is_start))
   |
   v
ProtonGPUToLLVM:
  read_counter → %clock / %clock_hi
  circular_store → store 2×u32 record into per-warp ring buffer
   |
   v
Runtime:
  copy scratch buffer → host
  decode entries → CycleEntry(isStart, scopeId, cycle)
  match start/end → ProfileEvent
   |
   v
Output:
  - trace: *.chrome_trace
  - tree : *.hatchet
```

### 3) Module / “class” relationship diagram (who calls whom)

```text
example_dsl.py
  |
  | uses
  v
triton.profiler.profile (python)          triton.profiler.language (python)
  - start/finalize                         - enter_scope/exit_scope/scope
  |                                        |
  | registers hooks                        | emits proton.record during compilation
  v                                        v
HookManager (python)                   triton._C.libtriton.proton (C++)
  - connects to knobs.runtime hooks        - create_proton_record
  |
  | activates
  v
InstrumentationHook (python) ---------> Triton backend compiler (python)
  - sets compiler.instrumentation         - calls instrumentation.patch(...) during lowering
  - calls libproton.* at runtime
  |
  v
triton._C.libproton (C++) -> SessionManager -> InstrumentationProfiler
                                   |
                                   v
                         CircularLayoutParser / EntryDecoder (C++)
```

---

# Trace Part I: Triton Vector Add (`add_kernel`)

## Frame 0: Module import + globals

Relevant code:

```python
import triton
import triton.profiler as proton
import triton.profiler.language as pl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_WARPS = 8
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L5)

**State before**: nothing initialized (no profiling, no kernel compiled).

**State after**:

- `DEVICE` is resolved from Triton’s active driver runtime
- The `pl.*` builtins exist, but **won’t emit IR** unless:
  - `flags.instrumentation_on == True` (set by InstrumentationHook.activate)
  - semantic type is allowed (for Triton kernels, you must call `pl.enable_semantic("triton")`)

---

## Frame 1: `config_helper(description)` parses CLI and returns `(op_measure, mode)`

Core logic (annotated):

```python
def config_helper(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--op-measure", action="store_true", default=False)
    parser.add_argument("--warp-sampling", action="store_true", default=False)
    parser.add_argument("--increase-accuracy", action="store_true", default=False)
    parser.add_argument("--warp-ids", type=str, default="0, 2")
    parser.add_argument("--gmem_buffer", action="store_true", default=False)

    args = parser.parse_args()

    # If increase-accuracy: lower overhead per record + post-process to reduce noise.
    opts = "clock32,time_shift" if args.increase_accuracy else ""

    buf = "global" if args.gmem_buffer else "shared"

    if args.warp_sampling:
        mode = proton.mode.Default(
            optimizations=opts,
            sampling_strategy="selective",
            sampling_options=args.warp_ids,
            buffer_type=buf,
        )
    else:
        mode = proton.mode.Default(optimizations=opts, buffer_type=buf)

    return args.op_measure, mode
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L35)

**State before**: Proton not started.

**State after**:

- `mode` is a `proton.mode.Default(...)` instance, stringified later and passed into libproton
- `mode.optimizations` may include `clock32` and `time_shift` (see [mode.py](../third_party/proton/proton/mode.py#L72))

---

## Frame 2: `__main__` (vector add) starts instrumentation profiling

The first `__main__` block (annotated):

```python
if __name__ == "__main__":
    # This must run before compiling any Triton kernels that contain pl.enter_scope.
    pl.enable_semantic("triton")

    op_measure, mode = config_helper(description)

    if op_measure:
        proton.start("vector-add", backend="instrumentation", mode=mode)
    else:
        proton.start("vector-add", data="trace", backend="instrumentation", mode=mode)
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L132)

### 2.1 `pl.enable_semantic("triton")`: why it exists

`pl.enable_semantic` mutates the allowed semantic types in Proton DSL:

```python
_SEMANTICS = {_ALL_SEMANTICS["gluon"]}  # default

def enable_semantic(semantic_name: str):
    _SEMANTICS.add(_ALL_SEMANTICS[semantic_name])
```

Source: [proton/language.py](../third_party/proton/proton/language.py#L14)

**Why you need it for Triton DSL kernels:**

`record(...)` checks that `_semantic` is an instance of an enabled semantic type:

```python
def record(..., semantic):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(semantic)  # TypeError if semantic type not enabled
    ...
```

Source: [proton/language.py](../third_party/proton/proton/language.py#L36)

For Triton kernels, `_semantic` will be a `TritonSemantic` instance (injected by the code generator), so you must add `TritonSemantic` to `_SEMANTICS`.

### 2.2 `proton.start(... backend="instrumentation")`: what happens

In Python:

```python
session = libproton.start(name, context, data, backend, mode_str)
if backend == "instrumentation":
    HookManager.register(InstrumentationHook(mode), session)
```

Source: [proton/profile.py](../third_party/proton/proton/profile.py#L99)

**Key side effects:**

1. Creates a session in the C++ runtime (`libproton.start`)
2. Registers the `InstrumentationHook` into Triton runtime hook chains (`HookManager.register`)

---

## Frame 3: Hook installation (Python) — `HookManager.register(...)`

Registration attaches three “heads” into Triton’s hook chains:

```python
knobs.runtime.kernel_load_end_hook.add(HookManager.init_handle)
knobs.runtime.launch_enter_hook.add(HookManager.enter)
knobs.runtime.launch_exit_hook.add(HookManager.exit)
```

Source: [proton/hooks/hook.py](../third_party/proton/proton/hooks/hook.py#L92)

This is the bridge between:

- **Compilation / binary loading** (kernel_load_end_hook)
- **Kernel launch enter/exit** (launch_enter_hook / launch_exit_hook)

and Proton’s own hook objects (`InstrumentationHook`, `LaunchHook`, etc.).

---

## Frame 4: `InstrumentationHook.activate()` enables instrumentation and injects compiler pass hooks

Key lines (annotated):

```python
flags.instrumentation_on = True

backends[backend_name].compiler.instrumentation = Instrumentation({
    "ttgpuir_to_llvmir": lambda pm: to_llvmir_passes(pm),
    "llvmir_to_llvm":    lambda pm: to_llvm_passes(pm),
})

set_profile_allocator(self.allocator)
triton.knobs.compilation.instrumentation_mode = str(self.mode)
```

Source: [proton/hooks/instrumentation.py](../third_party/proton/proton/hooks/instrumentation.py#L146)

**State before**:

- `flags.instrumentation_on == False` → `pl.enter_scope` would be a no-op
- backend compiler has no Proton instrumentation patches

**State after**:

- `flags.instrumentation_on == True` → `pl.enter_scope` emits `proton.record` ops during compilation
- backend compiler will call `instrumentation.patch(...)` at key lowering stages
- profile allocator is set so the runtime can allocate a profiling scratch buffer

### Where the backend actually calls `instrumentation.patch(...)`

On NVIDIA, the backend checks `CUDABackend.instrumentation`:

```python
if CUDABackend.instrumentation:
    CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
...
if CUDABackend.instrumentation:
    CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
```

Source: [third_party/nvidia/backend/compiler.py](../third_party/nvidia/backend/compiler.py#L344)

---

## Frame 5: Calling `add(x, y)` launches `add_kernel` (and compiles it on first use)

Python wrapper:

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=NUM_WARPS)
    return output
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L123)

The call `add_kernel[grid](...)` goes through `JITFunction.run(...)` and ends up calling the launcher with hooks:

```python
kernel.run(...,
           knobs.runtime.launch_enter_hook,
           knobs.runtime.launch_exit_hook,
           *bound_args.values())
```

Source: [runtime/jit.py](../python/triton/runtime/jit.py#L706)

---

## Frame 6: Compilation inserts `proton.record` ops (this is where `pl.scope` becomes real)

### 6.1 `with pl.scope(...)` is lowered by the compiler, not executed at runtime

The AST code generator translates `with` by explicitly calling `__enter__`/`__exit__`:

```python
cm = fn(*args, _semantic=self.semantic, **kws)  # injects _semantic into constructor
res = cm.__enter__()                            # emits enter_scope builtin
...
cm.__exit__(None, None, None)                   # emits exit_scope builtin
```

Source: [code_generator.py](../python/triton/compiler/code_generator.py#L1012)

### 6.2 How `_semantic` is provided (Triton vs Gluon)

During codegen, Triton kernels use `TritonSemantic(builder)`:

```python
if is_gluon:
    self.semantic = GluonSemantic(self.builder)
else:
    self.semantic = TritonSemantic(self.builder)
```

Source: [code_generator.py](../python/triton/compiler/code_generator.py#L274)

### 6.3 The `pl.enter_scope` builtin emits a `proton.record` op

Proton DSL builtin:

```python
@builtin
def enter_scope(name: tl.constexpr, _semantic=None):
    record(is_start=True, scope_name=name, semantic=_semantic)
```

Source: [proton/language.py](../third_party/proton/proton/language.py#L45)

Under the hood it calls the C++ binding:

```cpp
m.def("create_proton_record",
      [](TritonOpBuilder &opBuilder, bool isStart, const std::string &name) {
        auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(), name);
        opBuilder.create<proton::RecordOp>(isStart, nameAttr);
      });
```

Source: [triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#L76)

At the MLIR level, this is the `proton.record` op:

```mlir
proton.record start "name0"
...
proton.record end "name0"
```

Source: [ProtonOps.td](../third_party/proton/Dialect/include/Dialect/Proton/IR/ProtonOps.td#L16)

---

## Frame 7: `add_kernel` (what the scoped kernel looks like)

Annotated kernel (scopes emphasized):

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pl.enter_scope("kernel")               # (A) start scope "kernel"

    pid = tl.program_id(axis=0)            # program instance index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    with pl.scope("load_and_add"):         # (B) nested scope
        with pl.scope("load_x_issue"):     # (C) nested nested scope
            x = tl.load(x_ptr + offsets, mask=mask)
        with pl.scope("load_y_issue"):     # (D) nested nested scope
            y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
    pl.exit_scope("kernel")                # (E) end scope "kernel"
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L100)

**Important invariants:**

- Scope names must be unique per function (enforced when scope IDs are allocated)
- For Triton DSL, Proton warns about loops because TTIR/TTGIR rewrites can reorder/duplicate IR around the record points

---

## Frame 8: Kernel binary load triggers `kernel_load_end_hook` → scope metadata registration

Triton calls `kernel_load_end_hook` after loading the binary:

```python
if knobs.runtime.kernel_load_end_hook is not None:
    knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
```

Source: [compiler.py](../python/triton/compiler/compiler.py#L440)

`InstrumentationHook.init_handle(...)` then:

1. Finds the compiled `.ttgir` path in `metadata_group`
2. Parses the MLIR module
3. Extracts scope id/name + parent relationships
4. Calls `libproton.init_function_metadata(function, name, scope_id_names, scope_id_parents, metadata_path)`

Source: [instrumentation.py](../third_party/proton/proton/hooks/instrumentation.py#L223)

### How scope IDs + nesting are computed (C++)

Scope IDs are allocated by pairing record ops and validating control-flow constraints:

- Assign IDs and pair start/end by name: [ScopeIdAllocation.cpp](../third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp#L114)
- Validate CFG reachability (“start without close”, “end without open”): [ScopeIdAllocation.cpp](../third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp#L160)
- Infer parent/child nesting using dominance/post-dominance: [ScopeIdAllocation.cpp](../third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp#L236)

---

## Frame 9: Proton lowering: `proton.record` → clock reads + circular buffer stores

The core of `ConvertProtonToProtonGPU`:

```cpp
Value counter = gpu::ReadCounterOp::create(..., clkType, metricType);
int scopeId = scopeInfo.getOpScopeId(record);
gpu::CircularStoreOp::create(..., segment, counter, record.getIsStart(), scopeId);
record.erase();
```

Source: [ProtonToProtonGPUPass.cpp](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#L74)

Then `ReadCounterOp` lowers to the target clock implementation:

```cpp
Value clock = targetInfo.clock(rewriter, op.getLoc(), isClock64);
rewriter.replaceOp(op, clock);
```

Source: [PatternProtonGPUOpToLLVM.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.cpp#L48)

On NVIDIA, `TargetInfo::clock` reads `%clock` / `%clock_hi`:

```cpp
Value clkLow32 = getClockReg("%clock");
Value clkHigh32 = getClockReg("%clock_hi");
Value clock64 = (zext(clkHigh32) << 32) | zext(clkLow32);
```

Source: [TargetInfo.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.cpp#L12)

### How each record is encoded and written (C++)

`lowerCircularStoreOpHelper` packs `(isStart, scopeId, clock)` into 2×u32 and stores only from lane 0 of each warp:

```cpp
// tag: [isStart?][scopeId:8b][clk_high:11b]
Value tag = op.getIsStart() ? i32(scopeId<<23) : i32(1<<31 | scopeId<<23);
...
Value isWarpMaster = (laneId == 0);
...
return {isWriter, valsVec, vecPtr, addrSpace};
```

Source: [Utility.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/Utility.cpp#L68)

On NVIDIA global memory, the lowering emits PTX `st.global.cg` (vectorized when safe):

Source: [NvidiaPatternProtonGPUOpToLLVM.cpp](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.cpp#L27)

---

## Frame 10: Launch enter/exit hooks call into `libproton` (C++) and parse results

### 10.1 Hook invocation at launch sites

The launcher is given:

- `knobs.runtime.launch_enter_hook`
- `knobs.runtime.launch_exit_hook`

Source: [runtime/jit.py](../python/triton/runtime/jit.py#L706)

Those are `HookChain` objects that call each registered hook in order:

Source: [knobs.py](../python/triton/knobs.py#L406)

### 10.2 `InstrumentationHook.enter/exit` call libproton

```python
libproton.enter_instrumented_op(stream, func, self._data_ptr(), alloc_size)
...
libproton.exit_instrumented_op(stream, func, self._data_ptr(), alloc_size)
```

Source: [instrumentation.py](../third_party/proton/proton/hooks/instrumentation.py#L253)

### 10.3 libproton → SessionManager → InstrumentationProfiler

Pybind layer:

```cpp
m.def("exit_instrumented_op", [](uint64_t streamId, uint64_t functionId,
                                 uint64_t buffer, size_t size) {
  SessionManager::instance().exitInstrumentedOp(streamId, functionId,
      reinterpret_cast<uint8_t *>(buffer), size);
});
```

Source: [Proton.cpp](../third_party/proton/csrc/Proton.cpp#L126)

Dispatch:

```cpp
executeInterface(instrumentationInterfaceCounts, [&](auto *iface) {
  iface->exitInstrumentedOp(streamId, functionId, buffer, size);
}, /*isReversed=*/true);
```

Source: [Session.cpp](../third_party/proton/csrc/lib/Session/Session.cpp#L272)

### 10.4 InstrumentationProfiler parses circular layout entries into events

At exit:

```cpp
runtime->synchronizeStream(streamId);
runtime->processHostBuffer(hostBuffer, size, buffer, size, priorityStream,
  [&](uint8_t *bufferPtr, size_t size) {
    CircularLayoutParser parser(byteSpan, *circularLayoutConfig);
    parser.parse();
    for (blockTrace/trace/event) {
      duration = end.cycle - start.cycle;
      ... add CycleMetric ...
    }
  });
```

Source: [InstrumentationProfiler.cpp](../third_party/proton/csrc/lib/Profiler/Instrumentation/InstrumentationProfiler.cpp#L181)

The parser:

- Reads per-CTA metadata + per-warp counters
- Splits the record buffer into per-warp segments
- Decodes 8-byte records into `CycleEntry(scopeId,isStart,cycle)`
- Matches `START` followed by `END` per `scopeId`

Source: [CircularLayoutParser.cpp](../third_party/proton/common/lib/TraceDataIO/CircularLayoutParser.cpp#L38)

Record decoding (matches the encoding in `lowerCircularStoreOpHelper`):

```cpp
entry.isStart = (tagClkUpper & 0x80000000) == 0;
entry.scopeId = (tagClkUpper & 0x7F800000) >> 23;
entry.cycle   = (uint64(tagClkUpper & 0x7FF) << 32) | clkLower;
```

Source: [EntryDecoder.cpp](../third_party/proton/common/lib/TraceDataIO/EntryDecoder.cpp#L27)

---

# Trace Part II: Gluon Hopper Matmul (`blocked_matmul_pipelined_kernel`)

## Frame 11: Second `__main__` block (Hopper-only)

```python
if __name__ == "__main__":
    if not is_hopper():
        raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")
    ...
    proton.start("gemm", data="trace", backend="instrumentation", mode=mode)
    blocked_matmul_pipelined(...)
    proton.finalize()
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L287)

**Important gotcha**: because the file has *two* `__main__` blocks, running it on a non-Hopper GPU will:

1. Run the vector-add section successfully
2. Then raise in the GEMM section

---

## Frame 12: Gluon kernel scopes around TMA and WGMMA

The inner loop is explicitly scoped:

```python
for k in range(0, K, BLOCK_K):
    ...
    with pl.scope("tma_loads_issue"):
        tma.async_copy_global_to_shared(...)
        tma.async_copy_global_to_shared(...)

    with pl.scope("tma_loads_wait"):
        mbarrier.wait(...)

    with pl.scope("wgmma_wait"):
        acc = warpgroup_mma_wait(...)

    with pl.scope("wgmma_issue"):
        acc = warpgroup_mma(..., is_async=True)
```

Source: [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py#L204)

Because Gluon uses `GluonSemantic` (enabled by default in Proton DSL), `pl.enter_scope` and `pl.scope` work without `pl.enable_semantic("triton")`.

---

# Appendix: Why this yields “intra-kernel” timing

The instrumentation backend is not “sampling from outside”; it:

1. Lowers scope boundaries into device instructions that read a clock counter
2. Writes records into a per-block ring buffer
3. Copies and parses that buffer on the host after kernel completion

So the trace duration for a scope is derived from:

`cycle_end - cycle_start` for the same `scopeId` within the same warp/CTA record stream.

---

## Appendix: Where output files come from

In the tutorial:

- `data="trace"` → Chrome trace output (`*.chrome_trace`)
- default `data="tree"` → Hatchet output (`*.hatchet`)

The choice is passed from Python → C++ `SessionManager` → `Data` implementation chosen by name:

```cpp
if (toLower(dataName) == "tree") return TreeData(...)
if (toLower(dataName) == "trace") return TraceData(...)
```

Source: [Session.cpp](../third_party/proton/csrc/lib/Session/Session.cpp#L26)

