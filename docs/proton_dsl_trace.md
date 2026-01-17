# Proton DSL Intra-Kernel Profiling Trace Documentation

This document provides a comprehensive code trace of `example_dsl.py`, which demonstrates Proton's intra-kernel profiling capabilities for both Triton and Gluon kernels.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Key Files Reference](#key-files-reference)
4. [Key Functions Index](#key-functions-index)
5. [Detailed Code Trace](#detailed-code-trace)
   - [Frame 1: Module Initialization](#frame-1-module-initialization)
   - [Frame 2: Proton DSL Language Setup](#frame-2-proton-dsl-language-setup)
   - [Frame 3: Profiling Mode Configuration](#frame-3-profiling-mode-configuration)
   - [Frame 4: Proton Start](#frame-4-proton-start)
   - [Frame 5: Triton Kernel Definition (add_kernel)](#frame-5-triton-kernel-definition-add_kernel)
   - [Frame 6: Kernel Compilation](#frame-6-kernel-compilation)
   - [Frame 7: Proton Scope Recording](#frame-7-proton-scope-recording)
   - [Frame 8: Gluon Kernel (blocked_matmul_pipelined_kernel)](#frame-8-gluon-kernel-blocked_matmul_pipelined_kernel)
   - [Frame 9: Profiling Data Collection](#frame-9-profiling-data-collection)
   - [Frame 10: Finalization](#frame-10-finalization)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Module Relationships](#module-relationships)

---

## Overview

The `example_dsl.py` file demonstrates two approaches to intra-kernel profiling using Proton:

1. **Triton DSL Profiling**: A simple vector addition kernel with explicit scope annotations
2. **Gluon DSL Profiling**: A pipelined matrix multiplication kernel using Hopper-specific features (TMA, warpgroup MMA)

Proton instruments the IR at compile time, inserting clock read and storage operations to measure execution cycles within user-defined scopes.

---

## Architecture Diagrams

### Execution Sequence Diagram

```
┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐
│  User Script    │  │   Proton API   │  │  Triton Compiler │  │     GPU        │
│  example_dsl.py │  │                │  │                  │  │                │
└────────┬────────┘  └───────┬────────┘  └────────┬─────────┘  └───────┬────────┘
         │                   │                    │                    │
         │ pl.enable_semantic("triton")           │                    │
         │──────────────────>│                    │                    │
         │                   │                    │                    │
         │ proton.start(...)│                    │                    │
         │──────────────────>│                    │                    │
         │                   │ Register hooks     │                    │
         │                   │───────────────────>│                    │
         │                   │                    │                    │
         │ add_kernel[grid](...) ─────────────────────────────────────>│
         │                   │                    │                    │
         │                   │ Compile with       │                    │
         │                   │ instrumentation    │                    │
         │                   │<──────────────────>│                    │
         │                   │                    │                    │
         │                   │ pl.enter_scope("kernel")                │
         │                   │ ─ ─ ─ ─ ─ ─ ─ ─ ─>│ read_counter()     │
         │                   │                    │───────────────────>│
         │                   │                    │                    │
         │                   │                    │ circular_store()   │
         │                   │                    │───────────────────>│
         │                   │                    │                    │
         │                   │ kernel execution   │                    │
         │                   │<───────────────────│<───────────────────│
         │                   │                    │                    │
         │                   │ pl.exit_scope("kernel")                 │
         │                   │ ─ ─ ─ ─ ─ ─ ─ ─ ─>│ read_counter()     │
         │                   │                    │───────────────────>│
         │                   │                    │                    │
         │ proton.finalize() │                    │                    │
         │──────────────────>│                    │                    │
         │                   │ Write output files │                    │
         │                   │<───────────────────│                    │
         │                   │                    │                    │
         └───────────────────┴────────────────────┴────────────────────┘
```

### Profiling Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              COMPILATION PHASE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐    ┌───────────────┐    ┌────────────────┐                   │
│   │ Python AST   │───>│  Triton IR    │───>│ TritonGPU IR   │                   │
│   │ @triton.jit  │    │  (TTIR)       │    │   (TTGIR)      │                   │
│   └──────────────┘    └───────────────┘    └───────┬────────┘                   │
│                                                     │                            │
│   pl.enter_scope("name")  ───────────────>   proton.record start "name"         │
│   pl.exit_scope("name")   ───────────────>   proton.record end "name"           │
│                                                     │                            │
│                         ┌───────────────────────────┼───────────────────────┐   │
│                         │    ProtonToProtonGPU Pass │                       │   │
│                         │                           ▼                       │   │
│                         │   ┌─────────────────────────────────────────┐     │   │
│                         │   │  proton.record  ──>  gpu.read_counter   │     │   │
│                         │   │                 ──>  gpu.circular_store │     │   │
│                         │   └─────────────────────────────────────────┘     │   │
│                         └───────────────────────────────────────────────────┘   │
│                                                     │                            │
│                                                     ▼                            │
│                                           ┌────────────────┐                    │
│                                           │     LLVM IR    │                    │
│                                           └───────┬────────┘                    │
│                                                   │                              │
│                                                   ▼                              │
│                                           ┌────────────────┐                    │
│                                           │   PTX / CUBIN  │                    │
│                                           └────────────────┘                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                               RUNTIME PHASE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   GPU Memory Layout (Circular Buffer Strategy):                                 │
│                                                                                  │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │ Profile Scratch Buffer                                                  │    │
│   ├────────────────┬────────────────────────────┬───────────────────────────┤    │
│   │    Header      │   Warp Contexts            │     Profiled Data         │    │
│   │ (circularHdr)  │ (4 bytes × numWarps)       │   (allocBufferSize)       │    │
│   └────────────────┴────────────────────────────┴───────────────────────────┘    │
│                                                                                  │
│   Each profiled entry:                                                          │
│   ┌──────────────┬──────────────┬──────────────┐                                │
│   │   Clock      │   Scope ID   │  Start/End   │                                │
│   │  (32/64 bit) │   (16 bit)   │   (1 bit)    │                                │
│   └──────────────┴──────────────┴──────────────┘                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| [example_dsl.py](../third_party/proton/tutorials/intra_kernel/example_dsl.py) | Main tutorial demonstrating Proton DSL profiling |
| [proton/__init__.py](../third_party/proton/proton/__init__.py) | Proton package entry point, exports public API |
| [proton/language.py](../third_party/proton/proton/language.py#1) | DSL functions: `enter_scope`, `exit_scope`, `scope` |
| [proton/profile.py](../third_party/proton/proton/profile.py#1) | `start()`, `finalize()`, profiling session management |
| [proton/mode.py](../third_party/proton/proton/mode.py#1) | Mode configuration: `Default`, `MMA`, optimizations |
| [proton/hooks/instrumentation.py](../third_party/proton/proton/hooks/instrumentation.py#1) | `InstrumentationHook` - compiler pass registration |
| [proton/hooks/hook.py](../third_party/proton/proton/hooks/hook.py#1) | `HookManager` - lifecycle management |
| [proton/flags.py](../third_party/proton/proton/flags.py#1) | Global profiler state flags |
| [Dialect/triton_proton.cc](../third_party/proton/Dialect/triton_proton.cc#1) | Python bindings for Proton MLIR dialect |
| [ProtonOps.td](../third_party/proton/Dialect/include/Dialect/Proton/IR/ProtonOps.td#1) | MLIR TableGen definition for `proton.record` |
| [ProtonToProtonGPUPass.cpp](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#1) | IR lowering pass implementation |
| [runtime/jit.py](../python/triton/runtime/jit.py#1) | `@triton.jit` decorator and `JITFunction` class |
| [gluon/_runtime.py](../python/triton/experimental/gluon/_runtime.py#1) | `@gluon.jit` decorator and `GluonJITFunction` |
| [gluon/language/__init__.py](../python/triton/experimental/gluon/language/__init__.py#1) | Gluon language exports |
| [gluon/nvidia/hopper/__init__.py](../python/triton/experimental/gluon/language/nvidia/hopper/__init__.py#1) | Hopper-specific ops: `warpgroup_mma`, `tma` |
| [gluon/nvidia/hopper/tma.py](../python/triton/experimental/gluon/language/nvidia/hopper/tma.py#1) | TMA async copy operations |
| [gluon/nvidia/hopper.py](../python/triton/experimental/gluon/nvidia/hopper.py#1) | `TensorDescriptor` host-side helper |

---

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `pl.enable_semantic("triton")` | [language.py#28](../third_party/proton/proton/language.py#28) | Enable Proton DSL for Triton kernels |
| `pl.enter_scope(name)` | [language.py#46](../third_party/proton/proton/language.py#46) | Record scope start marker |
| `pl.exit_scope(name)` | [language.py#51](../third_party/proton/proton/language.py#51) | Record scope end marker |
| `pl.scope(name)` | [language.py#55](../third_party/proton/proton/language.py#55) | Context manager for scopes |
| `proton.start()` | [profile.py#50](../third_party/proton/proton/profile.py#50) | Start profiling session |
| `proton.finalize()` | [profile.py#167](../third_party/proton/proton/profile.py#167) | End session and write output |
| `mode.Default()` | [mode.py#117](../third_party/proton/proton/mode.py#117) | Default instrumentation mode |
| `HookManager.register()` | [hook.py#92](../third_party/proton/proton/hooks/hook.py#92) | Register profiling hooks |
| `InstrumentationHook.activate()` | [instrumentation.py#146](../third_party/proton/proton/hooks/instrumentation.py#146) | Install compiler passes |
| `create_proton_record()` | [triton_proton.cc#76](../third_party/proton/Dialect/triton_proton.cc#76) | Create MLIR RecordOp |
| `@triton.jit` | [jit.py#912](../python/triton/runtime/jit.py#912) | Triton kernel decorator |
| `@gluon.jit` | [_runtime.py#56](../python/triton/experimental/gluon/_runtime.py#56) | Gluon kernel decorator |
| `warpgroup_mma()` | [hopper/__init__.py#76](../python/triton/experimental/gluon/language/nvidia/hopper/__init__.py#76) | Hopper warpgroup MMA |
| `tma.async_copy_global_to_shared()` | [tma.py#124](../python/triton/experimental/gluon/language/nvidia/hopper/tma.py#124) | TMA async load |
| `TensorDescriptor.from_tensor()` | [hopper.py#54](../python/triton/experimental/gluon/nvidia/hopper.py#54) | Create TMA descriptor |

---

## Detailed Code Trace

### Frame 1: Module Initialization

**File:** [example_dsl.py#1-26](../third_party/proton/tutorials/intra_kernel/example_dsl.py#1)

```python
"""
Intra-Kernel Profiling Examples using Proton DSL for Triton and Gluon Kernels
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton           # Main Proton profiler module
import triton.profiler.language as pl      # DSL for scope annotations
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

NUM_WARPS = 8
```

**State BEFORE:** Python interpreter starts, no modules loaded
**State AFTER:**
- `proton` module loaded, provides `start()`, `finalize()` API
- `pl` (proton.language) provides `enter_scope()`, `exit_scope()`, `scope()`
- `gluon` experimental module loaded with Hopper-specific operations
- `DEVICE` set to current CUDA device
- `NUM_WARPS` constant = 8

**Explanation:**
The imports establish the full profiling and kernel development stack. The `proton.profiler` module wraps the native profiling backend (CUPTI/roctracer) and the instrumentation system. The `pl` module provides the in-kernel scope annotation DSL.

---

### Frame 2: Proton DSL Language Setup

**File:** [language.py#1-66](../third_party/proton/proton/language.py#1)

```python
from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton
from triton.language.semantic import TritonSemantic
from triton.experimental.gluon.language._semantic import GluonSemantic

from .flags import flags

# Supported semantics - by default only Gluon is enabled
_ALL_SEMANTICS = {
    "triton": TritonSemantic,
    "gluon": GluonSemantic,
}

# Gluon enabled by default; Triton requires explicit enablement
_SEMANTICS = {_ALL_SEMANTICS["gluon"]}


def enable_semantic(semantic_name: str):
    """Enable profiling for a specific DSL (triton or gluon)"""
    _SEMANTICS.add(_ALL_SEMANTICS[semantic_name])


def record(is_start: tl.constexpr, scope_name: tl.constexpr, semantic):
    """Internal: Create a proton.record IR operation"""
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(semantic)
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    # Create the MLIR proton.record operation
    return tl.tensor(triton_proton.create_proton_record(semantic.builder, is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _semantic=None):
    """Mark the start of a named profiling scope"""
    record(is_start=True, scope_name=name, semantic=_semantic)


@builtin
def exit_scope(name: tl.constexpr, _semantic=None):
    """Mark the end of a named profiling scope"""
    record(is_start=False, scope_name=name, semantic=_semantic)


class scope:
    """Context manager for profiling scopes"""

    def __init__(self, name: str, _semantic=None):
        self.name = name
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, _semantic=self.semantic)
```

**State BEFORE:** Module imported but Triton semantic not enabled
**State AFTER:** When `pl.enable_semantic("triton")` is called, Triton kernels can use scope annotations

**Explanation:**
By default, only Gluon kernels support Proton DSL because Triton's compiler performs aggressive optimizations (loop pipelining, instruction reordering, IR duplication) that can invalidate naive instrumentation. Explicit enablement is required for Triton to acknowledge this limitation.

The `@builtin` decorator marks these as kernel-level operations that are lowered during compilation. The `record()` function calls into the C++ binding to create `proton.record` MLIR operations.

---

### Frame 3: Profiling Mode Configuration

**File:** [example_dsl.py#35-97](../third_party/proton/tutorials/intra_kernel/example_dsl.py#35)

```python
def config_helper(description: str):
    # Configure command line arguments for profiling options
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--op-measure", action="store_true", default=False,
                        help="Enable operation measurement. Otherwise, timeline trace.")
    parser.add_argument("--warp-sampling", action="store_true", default=False,
                        help="Enable warp sampling during profiling")
    parser.add_argument("--increase-accuracy", action="store_true", default=False,
                        help="Enable increased-accuracy during profiling")
    parser.add_argument("--warp-ids", type=str, default="0, 2",
                        help="Comma-separated list of warp IDs for warp sampling")
    parser.add_argument("--gmem_buffer", action="store_true", default=False,
                        help="Use global memory as the internal buffer")

    args = parser.parse_args()

    # Configure optimizations
    opts = ""
    if args.increase_accuracy:
        opts = "clock32,time_shift"  # Lower overhead, post-process to reduce noise

    buf = "global" if args.gmem_buffer else "shared"

    # Configure sampling mode
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

Now trace into the `mode.Default` class:

**File:** [mode.py#54-124](../third_party/proton/proton/mode.py#54)

```python
@dataclass(frozen=True)
class InstrumentationMode(BaseMode):
    """Common base class for instrumentation modes with shared configuration."""
    metric_type: triton_proton.METRIC_TYPE = triton_proton.METRIC_TYPE.CYCLE
    sampling_strategy: triton_proton.SAMPLING_STRATEGY = triton_proton.SAMPLING_STRATEGY.NONE
    sampling_options: str = ""
    granularity: triton_proton.GRANULARITY = triton_proton.GRANULARITY.WARP
    buffer_strategy: triton_proton.BUFFER_STRATEGY = triton_proton.BUFFER_STRATEGY.CIRCULAR
    buffer_type: triton_proton.BUFFER_TYPE = triton_proton.BUFFER_TYPE.SHARED
    buffer_size: int = 0
    optimizations: List[Optimize] = field(default_factory=list)

    def __post_init__(self):
        # Automatically map string inputs to enums
        mappings = [
            ("metric_type", metric_types),
            ("sampling_strategy", sampling_strategies),
            ("granularity", granularities),
            ("buffer_strategy", buffer_strategies),
            ("buffer_type", buffer_types),
        ]
        for field_name, lookup in mappings:
            value = getattr(self, field_name)
            if isinstance(value, str):
                object.__setattr__(self, field_name, lookup[value])

        # Parse optimization string
        values_str = getattr(self, "optimizations")
        if isinstance(values_str, str):
            values = [v.strip() for v in values_str.split(",") if v.strip()]
            object.__setattr__(self, "optimizations",
                              [optimizations[v] for v in values])


@dataclass(frozen=True)
class Default(InstrumentationMode):
    name: str = field(default="default", init=False)
```

**State BEFORE:** No profiling configuration
**State AFTER:** `mode` object created with:
- `metric_type`: CYCLE (GPU clock counter)
- `granularity`: WARP (per-warp profiling)
- `buffer_strategy`: CIRCULAR (ring buffer for events)
- `buffer_type`: SHARED or GLOBAL memory
- `optimizations`: Optional clock32, time_shift

**Explanation:**
The mode configuration determines how profiling data is collected:
- **CYCLE metric**: Uses GPU clock counter for timing
- **WARP granularity**: Each warp has its own buffer segment
- **CIRCULAR buffer**: Events written to ring buffer, oldest overwritten if full
- **clock32**: Use 32-bit clock (lower overhead, 32-bit range)
- **time_shift**: Post-process to reduce timing noise

---

### Frame 4: Proton Start

**File:** [example_dsl.py#143-151](../third_party/proton/tutorials/intra_kernel/example_dsl.py#143)

```python
if op_measure:
    # Operation measurement mode generates scope-level metrics
    # View results with: proton-viewer -m normalized_cycles vector-add.hatchet
    proton.start("vector-add", backend="instrumentation", mode=mode)
else:
    # Timeline trace mode generates Chrome trace format
    # Output file: vector-add.chrome_trace
    proton.start("vector-add", data="trace", backend="instrumentation", mode=mode)
```

Now trace into `proton.start()`:

**File:** [profile.py#50-119](../third_party/proton/proton/profile.py#50)

```python
def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[Union[str, BaseMode]] = None,
    hook: Optional[str] = None,
) -> Optional[int]:
    """
    Start profiling with the given name and backend.

    Args:
        name: Profile output name (path)
        context: "shadow" or "python"
        data: "tree" (hatchet) or "trace" (chrome_trace)
        backend: None (auto), "cupti", "roctracer", or "instrumentation"
        mode: Instrumentation configuration
        hook: Optional hook type

    Returns:
        session: Session ID
    """
    if flags.command_line or triton.knobs.proton.disable:
        return None

    flags.profiling_on = True

    name = DEFAULT_PROFILE_NAME if name is None else name
    backend = _select_backend() if backend is None else backend
    mode_str = _get_mode_str(backend, mode)

    _check_env(backend)

    # Start the native profiling session
    session = libproton.start(name, context, data, backend, mode_str)

    # Register appropriate hooks
    if hook == "triton":
        HookManager.register(LaunchHook(), session)
    if backend == "instrumentation":
        HookManager.register(InstrumentationHook(mode), session)

    return session
```

**State BEFORE:** No profiling active
**State AFTER:**
- `flags.profiling_on = True`
- Native profiling session started via `libproton.start()`
- `InstrumentationHook` registered with `HookManager`

Now trace into `InstrumentationHook.activate()`:

**File:** [instrumentation.py#146-195](../third_party/proton/proton/hooks/instrumentation.py#146)

```python
def activate(self):
    if InstrumentationHook.active_count > 0:
        raise RuntimeError("Only one instrumentation hook can be active")

    InstrumentationHook.active_count += 1
    flags.instrumentation_on = True

    device = triton.runtime.driver.active.get_current_device()
    max_shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
    backend_name = _get_backend_name()  # "nvidia" or "amd"

    def to_llvmir_passes(pm):
        """Register passes for TTGIR -> LLVMIR conversion"""
        is_long_clk = False if mode.Optimize.CLOCK32 in self.mode.optimizations else True

        # Main pass: Convert proton.record to GPU-specific ops
        triton_proton.add_convert_proton_to_protongpu(
            pm,
            self.mode.metric_type,
            self.mode.sampling_strategy,
            self.mode.sampling_options,
            self.mode.granularity,
            self.mode.buffer_strategy,
            self.mode.buffer_type,
            self.mode.buffer_size,
            max_shared_mem,
            self.profile_buffer_size,
            self.profile_buffer_alignment,
            is_long_clk
        )
        triton_passes.common.add_cse(pm)  # Common subexpression elimination

        if mode.Optimize.SCHED_STORES in self.mode.optimizations:
            triton_proton.add_schedule_buffer_store(pm)

        triton_proton.add_allocate_proton_shared_memory(pm)

    def to_llvm_passes(pm):
        """Register passes for LLVMIR -> LLVM conversion"""
        triton_proton.add_allocate_proton_global_scratch_buffer(pm)
        if backend_name == "nvidia":
            triton_proton.add_convert_proton_nvidia_gpu_to_llvm(pm)
        elif backend_name == "amd":
            triton_proton.add_convert_proton_amd_gpu_to_llvm(pm, arch)

    # Register instrumentation passes with the compiler
    backends[backend_name].compiler.instrumentation = Instrumentation({
        "ttgpuir_to_llvmir": lambda pm: to_llvmir_passes(pm),
        "llvmir_to_llvm": lambda pm: to_llvm_passes(pm),
    })

    # Set up profiling allocator
    set_profile_allocator(self.allocator)

    # Configure compilation mode
    triton.knobs.compilation.instrumentation_mode = str(self.mode)
```

**State BEFORE:** No compiler instrumentation
**State AFTER:**
- `flags.instrumentation_on = True` (enables `pl.enter_scope` etc.)
- Compiler passes registered for ProtonToProtonGPU and ProtonGPUToLLVM
- Profile buffer allocator configured

---

### Frame 5: Triton Kernel Definition (add_kernel)

**File:** [example_dsl.py#100-130](../third_party/proton/tutorials/intra_kernel/example_dsl.py#100)

```python
@triton.jit
def add_kernel(x_ptr,              # *Pointer* to first input vector
               y_ptr,              # *Pointer* to second input vector
               output_ptr,         # *Pointer* to output vector
               n_elements,         # Size of the vector
               BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
               ):
    # Enter the outermost profiling scope
    pl.enter_scope("kernel")

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Nested scope for load and compute
    with pl.scope("load_and_add"):
        with pl.scope("load_x_issue"):
            x = tl.load(x_ptr + offsets, mask=mask)
        with pl.scope("load_y_issue"):
            y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

    # Exit the outermost scope
    pl.exit_scope("kernel")


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=NUM_WARPS)
    return output
```

**State BEFORE:** Function defined as Python function
**State AFTER:** `add_kernel` is a `JITFunction` object (not yet compiled)

The `@triton.jit` decorator:

**File:** [jit.py#912-964](../python/triton/runtime/jit.py#912)

```python
def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> KernelInterface[T]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        if knobs.runtime.interpret:
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn, ...)
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                ...
            )

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
```

---

### Frame 6: Kernel Compilation

When `add_kernel[grid](...)` is called:

**File:** [jit.py#706-761](../python/triton/runtime/jit.py#706)

```python
def run(self, *args, grid, warmup, **kwargs):
    kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)

    # Execute pre-run hooks
    for hook in self.pre_run_hooks:
        hook(*args, **kwargs)

    kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)

    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(key, None)

    # Compile if not cached
    if kernel is None:
        options, signature, constexprs, attrs = self._pack_args(
            backend, kwargs, bound_args, specialization, options)

        kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
        if kernel is None:
            return None

    if not warmup:
        # Canonicalize grid
        if callable(grid):
            grid = grid(bound_args)
        grid_0 = grid[0]
        grid_1 = grid[1] if len(grid) > 1 else 1
        grid_2 = grid[2] if len(grid) > 2 else 1

        # Launch kernel with hooks
        launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
        kernel.run(grid_0, grid_1, grid_2, stream, kernel.function,
                   kernel.packed_metadata, launch_metadata,
                   knobs.runtime.launch_enter_hook,  # InstrumentationHook.enter
                   knobs.runtime.launch_exit_hook,   # InstrumentationHook.exit
                   *bound_args.values())
    return kernel
```

The compilation flow:

1. **AST to TTIR**: Python AST → Triton IR
2. **TTIR to TTGIR**: Add GPU-specific layout information
3. **TTGIR to LLVMIR**: Lower to LLVM IR (with Proton passes!)
4. **LLVMIR to PTX/CUBIN**: Final GPU code generation

---

### Frame 7: Proton Scope Recording

When `pl.enter_scope("kernel")` executes during compilation:

**File:** [language.py#36-47](../third_party/proton/proton/language.py#36)

```python
def record(is_start: tl.constexpr, scope_name: tl.constexpr, semantic):
    """Create MLIR proton.record operation"""
    if not flags.instrumentation_on:
        return  # No-op if profiling disabled

    _check_supported_semantic(semantic)
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)

    # Call into C++ to create MLIR operation
    return tl.tensor(
        triton_proton.create_proton_record(semantic.builder, is_start, scope_name),
        tl.void
    )
```

The C++ binding:

**File:** [triton_proton.cc#76-82](../third_party/proton/Dialect/triton_proton.cc#76)

```cpp
m.def("create_proton_record",
      [](TritonOpBuilder &opBuilder, bool isStart,
         const std::string &name) -> void {
        auto nameAttr = mlir::StringAttr::get(opBuilder.getContext(),
                                              llvm::StringRef(name));
        opBuilder.create<proton::RecordOp>(isStart, nameAttr);
      });
```

This creates MLIR IR like:
```mlir
proton.record start "kernel"
...
proton.record end "kernel"
```

The ProtonToProtonGPU pass transforms this:

**File:** [ProtonToProtonGPUPass.cpp#74-140](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#74)

```cpp
LogicalResult replaceProtonRecordOp(OpBuilder &builder, FuncOp func,
                                    Value segment, MetricType metricType,
                                    ModuleScopeIdAllocation &scopeInfo,
                                    bool clockExtension) {
  mlir::IntegerType clkType =
      clockExtension ? mlir::IntegerType::get(builder.getContext(), 64)
                     : mlir::IntegerType::get(builder.getContext(), 32);

  // Replace all proton::RecordOp operations
  func->walk([&](proton::RecordOp record) {
    builder.setInsertionPoint(record);

    // Read the GPU clock counter
    Value counter = gpu::ReadCounterOp::create(builder, record.getLoc(),
                                               clkType, metricType);

    // Get the unique scope ID
    int scopeId = scopeInfo.getOpScopeId(record);

    // Store to circular buffer: (clock, scope_id, is_start)
    gpu::CircularStoreOp::create(builder, record.getLoc(), segment,
                                 counter, record.getIsStart(), scopeId);

    record.erase();  // Remove original operation
  });

  return success();
}
```

**State BEFORE:** `proton.record start "kernel"` MLIR operation
**State AFTER:**
```mlir
%clock = gpu.read_counter : i64 metric=CYCLE
gpu.circular_store %segment, %clock, is_start=true, scope_id=0
```

---

### Frame 8: Gluon Kernel (blocked_matmul_pipelined_kernel)

**File:** [example_dsl.py#204-271](../third_party/proton/tutorials/intra_kernel/example_dsl.py#204)

```python
@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pl.enter_scope("blocked_matmul_pipelined_kernel")

    # Allocate 2 buffers for double-buffering
    a_smem = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)
    index = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Initialize accumulator with WGMMA layout
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = warpgroup_mma_init(gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout))

    # Initialize barrier for TMA synchronization
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    # Main loop with pipelining
    for k in range(0, K, BLOCK_K):
        a = a_smem.index(index)
        b = b_smem.index(index)

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)

        # Profiled scope: TMA loads
        with pl.scope("tma_loads_issue"):
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a)
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b)

        with pl.scope("tma_loads_wait"):
            mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # Profiled scope: WGMMA operations
        with pl.scope("wgmma_wait"):
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

        with pl.scope("wgmma_issue"):
            acc = warpgroup_mma(a, b, acc, is_async=True)

        index ^= 1  # Swap buffers

    # Final wait
    with pl.scope("wgmma_last_wait"):
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

    mbarrier.invalidate(bar)

    # Write result back via TMA
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)

    pl.exit_scope("blocked_matmul_pipelined_kernel")
```

The Gluon `@jit` decorator:

**File:** [_runtime.py#45-102](../python/triton/experimental/gluon/_runtime.py#45)

```python
class GluonJITFunction(JITFunction[T]):

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource  # Use Gluon source, not Triton
        return result

    def is_gluon(self):
        return True


class GluonASTSource(ASTSource):

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        super().__init__(fn, signature, constexprs, attrs)
        self.language = Language.GLUON
        self.ext = "ttgir"  # Start from TTGIR directly

    def make_ir(self, target, options, codegen_fns, module_map, context):
        """Generate TTGIR directly (skip TTIR stage)"""
        builder = ir.builder(context)
        module = builder.create_module()

        # Set module attributes
        backend = make_backend(target)
        target = backend.get_target_name(options)
        module.set_attr("ttg.target", builder.get_string_attr(target))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))

        # Generate TTGIR from AST
        module = ast_to_ttir(self.fn, self, context=context, options=options,
                            codegen_fns=codegen_fns, module_map=module_map, module=module)
        return module
```

**Key difference from Triton:**
- Gluon generates TTGIR directly, skipping the TTIR stage
- Layout information is explicit in the code (not inferred by compiler)
- Hardware-specific operations (TMA, WGMMA) are directly available

---

### Frame 9: Profiling Data Collection

During kernel execution, the instrumentation hooks are invoked:

**File:** [instrumentation.py#253-268](../third_party/proton/proton/hooks/instrumentation.py#253)

```python
def enter(self, metadata: LazyDict) -> None:
    """Called before kernel launch"""
    func = metadata.data.get("function")
    stream = metadata.data.get("stream")
    alloc_size = 0 if self.buffer is None else self.buffer.element_size() * self.buffer.numel()

    # Notify profiler that instrumented kernel is starting
    libproton.enter_instrumented_op(stream, func, self._data_ptr(), alloc_size)

    if InstrumentationHook.enable_host_buffer:
        InstrumentationHook.host_buffer = None


def exit(self, metadata: LazyDict) -> None:
    """Called after kernel completion"""
    func = metadata.data.get("function")
    stream = metadata.data.get("stream")
    alloc_size = 0 if self.buffer is None else self.buffer.element_size() * self.buffer.numel()

    # Collect profiling data from device buffer
    libproton.exit_instrumented_op(stream, func, self._data_ptr(), alloc_size)

    if InstrumentationHook.enable_host_buffer:
        self._populate_host_buffer(func)
```

The HookManager orchestrates these calls:

**File:** [hook.py#44-53](../third_party/proton/proton/hooks/hook.py#44)

```python
@staticmethod
def enter(metadata: LazyDict) -> None:
    """Called by Triton runtime before kernel launch"""
    for hook in HookManager.active_hooks:
        hook.enter(metadata)

@staticmethod
def exit(metadata: LazyDict) -> None:
    """Called by Triton runtime after kernel completion"""
    # Reverse order: FILO semantics
    for hook in reversed(HookManager.active_hooks):
        hook.exit(metadata)
```

---

### Frame 10: Finalization

**File:** [example_dsl.py#160](../third_party/proton/tutorials/intra_kernel/example_dsl.py#160)

```python
proton.finalize()
```

**File:** [profile.py#167-189](../third_party/proton/proton/profile.py#167)

```python
def finalize(session: Optional[int] = None, output_format: Optional[str] = "") -> None:
    """
    Finalizes a profiling session.
    Flush and write the profiling data to the file specified by the session name.

    Args:
        session: Session ID to finalize. If None, all sessions are finalized.
        output_format: Output format ("hatchet", "hatchet_msgpack", "chrome_trace").
    """
    # Unregister all hooks for this session
    HookManager.unregister(session)

    if session is None:
        flags.profiling_on = False
        libproton.finalize_all(output_format)
    else:
        if flags.command_line and session != 0:
            raise ValueError("Only one session can be finalized from command line.")
        libproton.finalize(session, output_format)
```

The `HookManager.unregister()` method:

**File:** [hook.py#106-128](../third_party/proton/proton/hooks/hook.py#106)

```python
@staticmethod
def unregister(session: Optional[int] = None) -> None:
    if session is not None and session not in HookManager.session_hooks:
        return

    if session is None:
        # Deactivate all hooks
        for hook in HookManager.active_hooks:
            hook.deactivate()
        HookManager.active_hooks.clear()
        HookManager.session_hooks.clear()
    else:
        # Deactivate hooks for specific session
        popped_hooks = HookManager.session_hooks.pop(session)
        for hook, active in popped_hooks.items():
            if not active:
                continue
            # Only deactivate if no other session uses this hook
            if not any(session_hooks[hook] for session_hooks in HookManager.session_hooks.values()):
                hook.deactivate()
                HookManager.active_hooks.remove(hook)

    # Unregister the runtime hooks
    if not HookManager.active_hooks:
        knobs.runtime.kernel_load_end_hook.remove(HookManager.init_handle)
        knobs.runtime.launch_enter_hook.remove(HookManager.enter)
        knobs.runtime.launch_exit_hook.remove(HookManager.exit)
```

**State BEFORE:** Profiling active, data in GPU buffers
**State AFTER:**
- All hooks deactivated
- Compiler instrumentation removed
- Data written to output file(s):
  - `vector-add.hatchet` (tree format) or `vector-add.chrome_trace` (timeline)
  - `gemm.hatchet` or `gemm.chrome_trace`

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   USER CODE                     COMPILATION                   EXECUTION          │
│   ─────────                     ───────────                   ─────────          │
│                                                                                  │
│   @triton.jit                                                                    │
│   def kernel(...):                                                               │
│       pl.enter_scope("A")  ──────┐                                              │
│       ...                        │   ┌───────────────────┐                      │
│       pl.exit_scope("A")   ──────┴──>│ proton.record     │                      │
│                                      │ start "A"         │                      │
│                                      │ ...               │                      │
│                                      │ proton.record     │                      │
│                                      │ end "A"           │                      │
│                                      └─────────┬─────────┘                      │
│                                                │                                 │
│                                   ProtonToProtonGPU Pass                        │
│                                                │                                 │
│                                                ▼                                 │
│                                      ┌───────────────────┐                      │
│                                      │ gpu.read_counter  │                      │
│                                      │ gpu.circular_store│ ────┐                │
│                                      └───────────────────┘     │                │
│                                                                │                │
│                                            ProtonGPUToLLVM Pass│                │
│                                                                │                │
│                                                ▼               │                │
│                                      ┌───────────────────┐     │                │
│                                      │ PTX instructions: │     │                │
│                                      │ - mov.u64 %clock  │     │                │
│                                      │ - st.shared [buf] │     │                │
│                                      └───────────────────┘     │                │
│                                                                │                │
│   ┌─────────────────────────────────────────────────────────────┘                │
│   │                                                                              │
│   ▼                                                                              │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │                         GPU EXECUTION                                      │ │
│   │                                                                            │ │
│   │   Warp 0          Warp 1          Warp 2          ...      Warp N-1       │ │
│   │   ┌────┐          ┌────┐          ┌────┐                   ┌────┐         │ │
│   │   │Seg0│          │Seg1│          │Seg2│                   │SegN│         │ │
│   │   │────│          │────│          │────│                   │────│         │ │
│   │   │clk │          │clk │          │clk │                   │clk │         │ │
│   │   │id  │          │id  │          │id  │                   │id  │         │ │
│   │   │... │          │... │          │... │                   │... │         │ │
│   │   └────┘          └────┘          └────┘                   └────┘         │ │
│   │     │               │               │                        │            │ │
│   │     └───────────────┴───────────────┴────────────────────────┘            │ │
│   │                                    │                                       │ │
│   └────────────────────────────────────┼───────────────────────────────────────┘ │
│                                        │                                         │
│                              libproton.exit_instrumented_op()                    │
│                                        │                                         │
│                                        ▼                                         │
│                             ┌─────────────────────┐                             │
│                             │   HOST PROCESSING    │                             │
│                             │                      │                             │
│                             │ - Copy D2H           │                             │
│                             │ - Parse clock values │                             │
│                             │ - Build scope tree   │                             │
│                             │ - Compute cycles     │                             │
│                             └──────────┬──────────┘                             │
│                                        │                                         │
│                              proton.finalize()                                   │
│                                        │                                         │
│                                        ▼                                         │
│                             ┌─────────────────────┐                             │
│                             │   OUTPUT FILES       │                             │
│                             │                      │                             │
│                             │ - .hatchet (metrics) │                             │
│                             │ - .chrome_trace      │                             │
│                             └─────────────────────┘                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODULE DEPENDENCY GRAPH                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                           ┌────────────────────┐                                │
│                           │   User Code        │                                │
│                           │   example_dsl.py   │                                │
│                           └─────────┬──────────┘                                │
│                                     │                                            │
│            ┌────────────────────────┼────────────────────────┐                  │
│            │                        │                        │                  │
│            ▼                        ▼                        ▼                  │
│   ┌────────────────┐    ┌────────────────────┐    ┌───────────────────┐        │
│   │  triton.jit    │    │ triton.profiler    │    │ gluon (exprmtl)   │        │
│   │  runtime/jit   │    │ (proton)           │    │                   │        │
│   └───────┬────────┘    └─────────┬──────────┘    └─────────┬─────────┘        │
│           │                       │                         │                   │
│           │                       │                         │                   │
│           ▼                       ▼                         ▼                   │
│   ┌───────────────────────────────────────────────────────────────────┐        │
│   │                    triton.compiler                                 │        │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐        │        │
│   │  │ ASTSource   │  │ code_gen    │  │ GluonASTSource      │        │        │
│   │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘        │        │
│   │         │                │                    │                    │        │
│   │         └────────────────┼────────────────────┘                    │        │
│   │                          │                                         │        │
│   │                          ▼                                         │        │
│   │                   ┌─────────────┐                                  │        │
│   │                   │ MLIR Passes │                                  │        │
│   │                   └──────┬──────┘                                  │        │
│   └──────────────────────────┼─────────────────────────────────────────┘        │
│                              │                                                   │
│   ┌──────────────────────────┼──────────────────────────────────────────┐       │
│   │    Proton Dialect (C++)  │                                          │       │
│   │                          ▼                                          │       │
│   │   ┌─────────────────────────────────────────────────────────────┐   │       │
│   │   │ proton::RecordOp                                             │   │       │
│   │   │ ────────────────                                             │   │       │
│   │   │ MLIR operation representing scope start/end                  │   │       │
│   │   └──────────────────────────────┬──────────────────────────────┘   │       │
│   │                                  │                                  │       │
│   │                ProtonToProtonGPU Pass                               │       │
│   │                                  │                                  │       │
│   │                                  ▼                                  │       │
│   │   ┌─────────────────────────────────────────────────────────────┐   │       │
│   │   │ proton::gpu::ReadCounterOp  │  proton::gpu::CircularStoreOp │   │       │
│   │   │ ─────────────────────────── │  ─────────────────────────────│   │       │
│   │   │ Read GPU clock counter      │  Store to profiling buffer    │   │       │
│   │   └──────────────────────────────┬──────────────────────────────┘   │       │
│   │                                  │                                  │       │
│   │                ProtonGPUToLLVM Pass                                 │       │
│   │                                  │                                  │       │
│   │                                  ▼                                  │       │
│   │   ┌─────────────────────────────────────────────────────────────┐   │       │
│   │   │ LLVM IR: inline PTX for clock read, memory operations       │   │       │
│   │   └─────────────────────────────────────────────────────────────┘   │       │
│   │                                                                     │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│                              │                                                   │
│                              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    libproton (C++ Runtime)                               │   │
│   │                                                                          │   │
│   │   - Session management                                                   │   │
│   │   - Buffer allocation                                                    │   │
│   │   - Data collection (enter_instrumented_op / exit_instrumented_op)      │   │
│   │   - Output file generation (hatchet / chrome_trace)                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Complete Compilation Pipeline Trace (C++ MLIR Passes)

This section provides a comprehensive frame-by-frame trace of the complete compilation pipeline from Python to PTX, including all C++ MLIR passes.

### Compilation Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE TRITON COMPILATION PIPELINE                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────┐                                                                  │
│   │ Python Source   │  @triton.jit def kernel(...):                                    │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            ▼  ast_to_ttir() (code_generator.py)                                        │
│   ┌─────────────────┐                                                                  │
│   │  Triton IR      │  tt.func, tt.load, tt.store, proton.record                       │
│   │  (TTIR)         │                                                                  │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            ▼  ConvertTritonToTritonGPU pass (TritonToTritonGPUPass.cpp)                │
│   ┌─────────────────┐                                                                  │
│   │  TritonGPU IR   │  + BlockedEncodingAttr, DotOperandEncodingAttr                   │
│   │  (TTGIR)        │  + GPU-specific layout information                               │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            ▼  TTGIR Optimization passes (Coalesce, AccelerateMatmul, etc.)             │
│   ┌─────────────────┐                                                                  │
│   │  Optimized      │  + Memory coalescing, MMA acceleration                           │
│   │  TTGIR          │  + Layout conversions, instruction reordering                    │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  ╔══════════════════════════════════════════════════════════════════╗    │
│            ├──║ PROTON INSTRUMENTATION PASSES (if profiling enabled)             ║    │
│            │  ║                                                                  ║    │
│            │  ║  1. ProtonToProtonGPU: proton.record → gpu.read_counter +        ║    │
│            │  ║                                        gpu.circular_store        ║    │
│            │  ║  2. AllocateProtonSharedMemory: Allocate profiling buffer        ║    │
│            │  ╚══════════════════════════════════════════════════════════════════╝    │
│            │                                                                            │
│            ▼  ConvertTritonGPUToLLVM pass (TritonGPUToLLVM.cpp)                        │
│   ┌─────────────────┐                                                                  │
│   │  LLVM IR        │  LLVM dialect + NVVM dialect (CUDA intrinsics)                   │
│   │  (NVVM)         │                                                                  │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  ╔══════════════════════════════════════════════════════════════════╗    │
│            ├──║ PROTON LLVM LOWERING (if profiling enabled)                      ║    │
│            │  ║                                                                  ║    │
│            │  ║  ProtonGPUToLLVM: gpu.read_counter → PTX clock intrinsics        ║    │
│            │  ║                   gpu.circular_store → shared/global stores      ║    │
│            │  ╚══════════════════════════════════════════════════════════════════╝    │
│            │                                                                            │
│            ▼  LLVM::translateModuleToPTX (make_ptx)                                    │
│   ┌─────────────────┐                                                                  │
│   │  PTX Assembly   │  .version 8.x, .target sm_90a, ...                               │
│   │                 │  mov.u64 %rd0, %clock64;                                         │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            ▼  ptxas / nvcc (make_cubin)                                                │
│   ┌─────────────────┐                                                                  │
│   │  CUBIN          │  Binary ready for execution                                       │
│   │  (GPU binary)   │                                                                  │
│   └─────────────────┘                                                                  │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Frame 11: Backend Compilation Stages (Python → CUBIN)

**File:** [third_party/nvidia/backend/compiler.py#180-350](../third_party/nvidia/backend/compiler.py#180)

```python
class CUDABackend(BaseBackend):

    @staticmethod
    def make_ttir(mod, metadata, options):
        """Stage 1: Convert TTIR to optimized TTIR"""
        passes.ttir.add_combine(pm)           # Fold and simplify
        passes.ttir.add_reorder_broadcast(pm) # Optimize broadcast placement
        passes.common.add_canonicalize(pm)    # Standard MLIR canonicalization
        passes.common.add_cse(pm)             # Common subexpression elimination
        passes.ttir.add_loop_unroll(pm)       # Unroll loops
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        """Stage 2: Convert TTIR to TTGIR with GPU layouts"""
        # Primary conversion pass
        passes.ttir.add_convert_to_ttgpuir(pm,
            target=metadata["target"],
            num_warps=options.num_warps,
            threads_per_warp=options.warp_size,
            num_ctas=options.num_ctas,
        )
        # Layout optimizations
        passes.ttgpuir.add_coalesce(pm)            # Memory coalescing
        passes.ttgpuir.add_f32_dot_tc(pm)          # Tensor core for FP32
        passes.ttgpuir.add_accelerate_matmul(pm)   # WMMA/MMA instruction selection

        if options.num_stages:
            passes.ttgpuir.add_pipeline(pm, options.num_stages)  # Software pipelining

        passes.ttgpuir.add_prefetch(pm)                         # Prefetching
        passes.ttgpuir.add_optimize_dot_operands(pm, True)     # Operand optimization
        passes.ttgpuir.add_remove_layout_conversions(pm)       # Layout cleanup
        passes.ttgpuir.add_reduce_data_duplication(pm)         # Reduce redundant data
        passes.ttgpuir.add_reorder_instructions(pm)            # Instruction scheduling
        return mod

    @staticmethod
    def make_llir(mod, metadata, options):
        """Stage 3: Convert TTGIR to LLVM IR"""
        # INSTRUMENTATION HOOK POINT: ttgpuir_to_llvmir passes injected here
        passes.ttgpuir.add_decompose_unsupported_nvidia_conversions(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)

        # Main conversion to LLVM
        passes.ttgpuir.nvidia.add_to_llvmir(pm,
            compute_capability=metadata["cc"],
            ptx_version=metadata["ptx_version"],
        )

        # INSTRUMENTATION HOOK POINT: llvmir_to_llvm passes injected here
        passes.common.add_canonicalize(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        return mod

    @staticmethod
    def make_ptx(mod, metadata, options):
        """Stage 4: Convert LLVM IR to PTX assembly"""
        return llvm.translate_to_ptx(
            mod,
            compute_capability=metadata["cc"],
            ptx_version=metadata["ptx_version"],
        )

    @staticmethod
    def make_cubin(src, metadata, options):
        """Stage 5: Compile PTX to CUBIN"""
        return llvm.compile_ptx_to_cubin(
            src,
            arch=f"sm_{metadata['cc']}",
            ptxas_path=options.ptxas_path,
        )
```

**Stage Outputs:**
- `make_ttir`: Optimized Triton IR with loop unrolling and CSE
- `make_ttgir`: TritonGPU IR with GPU layouts and MMA acceleration
- `make_llir`: LLVM IR with NVVM intrinsics
- `make_ptx`: PTX assembly text
- `make_cubin`: Binary executable for GPU

---

### Frame 12: TTIR to TTGIR Conversion (C++)

**File:** [lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp#796-840](../lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp#796)

```cpp
class ConvertTritonToTritonGPU
    : public triton::impl::ConvertTritonToTritonGPUBase<ConvertTritonToTritonGPU> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Type converter adds GPU encodings to tensor types
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, enableSourceRemat);
    TritonGPUConversionTarget target(*context, typeConverter);

    RewritePatternSet patterns(context);

    // Register conversion patterns for all operations
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, numCTAs);
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);

    // Set module-level GPU attributes
    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    // Apply partial conversion
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};
```

**Key Conversion Patterns:**

```cpp
// TritonDotPattern: Convert tt.dot to ttg.dot with MMA encodings
struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  LogicalResult matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();

    // Compute optimal sizePerThread for dot products
    SmallVector<unsigned> retSizePerThread(rank, 1);
    auto numElements = product<int64_t>(origShape);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }

    // Create BlockedEncodingAttr for output
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder,
        numWarps, threadsPerWarp, numCTAs);

    // Add DotOperandEncodingAttr for operands A and B
    Attribute aEncoding = triton::gpu::DotOperandEncodingAttr::get(
        getContext(), 0, dEncoding, aEltType);
    Attribute bEncoding = triton::gpu::DotOperandEncodingAttr::get(
        getContext(), 1, dEncoding, bEltType);

    // Insert layout conversions and create new dot
    a = triton::gpu::ConvertLayoutOp::create(rewriter, a.getLoc(), dstTypeA, a);
    b = triton::gpu::ConvertLayoutOp::create(rewriter, b.getLoc(), dstTypeB, b);
    c = triton::gpu::ConvertLayoutOp::create(rewriter, c.getLoc(), retType, c);

    rewriter.replaceOpWithNewOp<triton::DotOp>(op, retType, a, b, c, ...);
    return success();
  }
};
```

**State BEFORE:**
```mlir
%result = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
```

**State AFTER:**
```mlir
%a_conv = ttg.convert_layout %a : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot_op<opIdx=0>>
%b_conv = ttg.convert_layout %b : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot_op<opIdx=1>>
%c_conv = ttg.convert_layout %c : tensor<128x128xf32, #blocked0> -> tensor<128x128xf32, #blocked1>
%result = tt.dot %a_conv, %b_conv, %c_conv : ... -> tensor<128x128xf32, #blocked1>
```

---

### Frame 13: TTGIR to LLVM IR Conversion (NVIDIA)

**File:** [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#79-212](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#79)

```cpp
struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Target-specific info (compute capability, PTX version)
    TargetInfo targetInfo(computeCapability, ptxVersion);

    // Shared memory allocation and memory barrier analysis
    ModuleAllocation allocation(mod,
        mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Type converter with 32-bit index override
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions first
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                                   targetInfo, ...);
    applyPartialConversion(mod, funcTarget, std::move(funcPatterns));

    // Initialize shared memory global
    initSharedMemory(typeConverter);

    // Main conversion patterns
    RewritePatternSet patterns(context);

    // NVIDIA-specific patterns
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(...);
    mlir::triton::NVIDIA::populateTMAToLLVMPatterns(...);
    populateDotOpToLLVMPatterns(...);  // WMMA/MMA lowering
    populateElementwiseOpToLLVMPatterns(...);
    populateLoadStoreOpToLLVMPatterns(...);
    populateBarrierOpToLLVMPatterns(...);

    // Generic patterns
    mlir::triton::populateReduceOpToLLVMPatterns(...);
    mlir::triton::populateScanOpToLLVMPatterns(...);
    mlir::triton::populatePrintOpToLLVMPattern(...);
    mlir::triton::populateControlFlowOpToLLVMPattern(...);
    mlir::triton::populateSPMDOpToLLVMPattern(...);

    // PROTON INSTRUMENTATION PATTERNS
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    // Standard MLIR patterns
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    // Apply conversion
    TritonLLVMConversionTarget convTarget(*context);
    applyPartialConversion(mod, convTarget, std::move(patterns));
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();

    // Create global shared memory allocation with dynamic size
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(b, loc, arrayTy, /*isConstant=*/false,
                           LLVM::Linkage::External, "global_smem",
                           /*value=*/Attribute(), /*alignment=*/16,
                           static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};
```

**Key Conversion - Load Operation to LLVM:**

```cpp
// LoadStoreOpToLLVM.cpp (simplified)
struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  LogicalResult matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Get pointer values for each element
    auto ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);

    // Generate load for each element with vectorization
    SmallVector<Value> loadedVals;
    for (size_t i = 0; i < ptrElems.size(); i += vec) {
      // Create vectorized load
      Value vecPtr = ptrElems[i];
      auto vecTy = vec_ty(elemTy, vec);
      Value loaded = b.load(vecTy, vecPtr, alignment);
      loadedVals.push_back(loaded);
    }

    // Pack results back into struct
    Value result = packLLElements(loc, loadedVals, rewriter, resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

---

### Frame 14: ProtonToProtonGPU Pass (C++)

**File:** [third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#74-140](../third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp#74)

```cpp
// This pass converts high-level proton.record operations to GPU-specific
// profiling operations (read_counter + circular_store)

LogicalResult replaceProtonRecordOp(OpBuilder &builder, FuncOp func,
                                    Value segment, MetricType metricType,
                                    ModuleScopeIdAllocation &scopeInfo,
                                    bool clockExtension) {
  // Choose clock type based on configuration
  mlir::IntegerType clkType =
      clockExtension ? mlir::IntegerType::get(builder.getContext(), 64)
                     : mlir::IntegerType::get(builder.getContext(), 32);

  // Walk all proton::RecordOp in the function
  func->walk([&](proton::RecordOp record) {
    builder.setInsertionPoint(record);

    // Step 1: Read the GPU clock counter
    // Creates: %clock = proton_gpu.read_counter : i32/i64 {metric = CYCLE}
    Value counter = gpu::ReadCounterOp::create(
        builder, record.getLoc(), clkType, metricType);

    // Step 2: Get unique scope ID for this record pair
    // The ScopeIdAllocation analysis pairs start/end records and assigns IDs
    int scopeId = scopeInfo.getOpScopeId(record);

    // Step 3: Store clock value to circular buffer
    // Creates: proton_gpu.circular_store %segment, %clock, is_start, scope_id
    gpu::CircularStoreOp::create(
        builder, record.getLoc(),
        segment,              // Per-warp buffer segment
        counter,              // Clock value
        record.getIsStart(),  // true=start, false=end
        scopeId               // Unique scope identifier
    );

    // Remove original operation
    record.erase();
  });

  return success();
}

// Main pass entry point
void ProtonToProtonGPUPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Run scope analysis to pair start/end records and compute IDs
  ModuleScopeIdAllocation scopeInfo(moduleOp);

  for (auto funcOp : moduleOp.getOps<triton::FuncOp>()) {
    OpBuilder builder(funcOp);

    // Allocate profiling buffer (shared or global memory)
    Value buffer = allocateProfilingBuffer(funcOp, builder, bufferType, bufferSize);

    // Create segment for this warp
    // Creates: %segment = proton_gpu.segment_alloc %buffer
    Value segment = gpu::SegmentAllocOp::create(builder, funcOp.getLoc(), buffer);

    // Initialize profiler at kernel start
    // Creates: proton_gpu.initialize %scratch_ptr
    createInitializeOp(funcOp, builder, scratchPtr);

    // Replace all proton.record ops
    replaceProtonRecordOp(builder, funcOp, segment, metricType,
                          scopeInfo, clockExtension);

    // Finalize profiler at kernel end
    // Creates: proton_gpu.finalize %segment, %scratch_ptr
    createFinalizeOp(funcOp, builder, segment, scratchPtr);
  }
}
```

**Scope ID Allocation Analysis:**

**File:** [third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp#48-112](../third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp#48)

```cpp
void ScopeIdAllocation::run() {
  // Stage 1: liveness() - Pair start/end records with same name
  //
  //   proton.record start @"foo"  // scopeId = 0
  //   ...
  //   proton.record end @"foo"    // scopeId = 0 (matched!)
  //   proton.record start @"foo"  // scopeId = 1 (new instance)
  //   ...
  //   proton.record end @"foo"    // scopeId = 1

  // Stage 2: reachability() - Track scopes across CFG
  //
  //   scf.if %cond {
  //     proton.record start @"foo"  // Error if not closed on this path
  //   }

  // Stage 3: dominance() - Verify start dominates end, compute nesting
  //
  //   proton.record start @"outer"
  //   scf.if %cond {
  //     proton.record start @"inner"  // parent = outer
  //     proton.record end @"inner"
  //   }
  //   proton.record end @"outer"

  liveness();
  reachability();
  dominance();
}

void ScopeIdAllocation::liveness() {
  llvm::DenseMap<StringRef, std::pair<size_t, bool>> nameToIdMap;
  ScopeId scopeId = 0;

  funcOp->walk<WalkOrder::PreOrder>([&](RecordOp recordOp) {
    auto name = recordOp.getName();

    if (!nameToIdMap.contains(name)) {
      // First occurrence of this name - assign new ID
      nameToIdMap[name] = {scopeId, recordOp.getIsStart()};
      opToIdMap[recordOp] = scopeId;
      idToNameMap[scopeId] = name;
      scopeId++;
    } else {
      auto &[existingId, isStart] = nameToIdMap[name];
      if (isStart == recordOp.getIsStart()) {
        // Error: duplicate start or end
        mlir::emitError(recordOp.getLoc(), "The scope name '")
            << name << "' has duplicate "
            << (recordOp.getIsStart() ? "start" : "end") << " record";
      } else {
        // Matching pair found
        opToIdMap[recordOp] = existingId;
        nameToIdMap.erase(name);  // Allow same name to be reused
      }
    }
  });
}
```

**State BEFORE:**
```mlir
proton.record start "kernel"
...
  proton.record start "load_x"
  %x = tt.load %ptr
  proton.record end "load_x"
...
proton.record end "kernel"
```

**State AFTER:**
```mlir
%segment = proton_gpu.segment_alloc %buffer : !proton_gpu.segment
proton_gpu.initialize %scratch : !tt.ptr<i32>

%clk0 = proton_gpu.read_counter : i64 {metric = cycle}
proton_gpu.circular_store start %segment, %clk0 : !proton_gpu.segment, i64 {scopeId = 0}
...
  %clk1 = proton_gpu.read_counter : i64 {metric = cycle}
  proton_gpu.circular_store start %segment, %clk1 : !proton_gpu.segment, i64 {scopeId = 1}
  %x = tt.load %ptr
  %clk2 = proton_gpu.read_counter : i64 {metric = cycle}
  proton_gpu.circular_store end %segment, %clk2 : !proton_gpu.segment, i64 {scopeId = 1}
...
%clk3 = proton_gpu.read_counter : i64 {metric = cycle}
proton_gpu.circular_store end %segment, %clk3 : !proton_gpu.segment, i64 {scopeId = 0}

proton_gpu.finalize %segment, %scratch : !proton_gpu.segment, !tt.ptr<i32>
```

---

### Frame 15: ProtonGPU to LLVM Conversion (NVIDIA Clock Intrinsics)

**File:** [third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.cpp#12-51](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.cpp#12)

```cpp
// NVIDIA clock reading implementation using PTX inline assembly
Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {

  // Helper to read PTX special registers via inline assembly
  auto getClockReg = [&](const std::string &clkName) {
    PTXBuilder builder;
    // PTX instruction: mov.u32 %rd, %clock  (or %clock_hi)
    auto &movLow = builder.create("mov")->o("u32");
    auto *destLowOpr = builder.newOperand("=r");        // Output: 32-bit register
    auto *sRegLowOpr = builder.newConstantOperand(clkName);  // Input: %clock
    movLow(destLowOpr, sRegLowOpr);

    // Generate inline assembly and get result
    Value clkLow32 = builder.launch(rewriter, loc,
                                     rewriter.getIntegerType(32), true);
    return clkLow32;
  };

  // Read low 32 bits of clock
  Value clkLow32 = getClockReg("%clock");

  if (!isClock64)
    return clkLow32;  // 32-bit clock is sufficient

  // For 64-bit clock, also read high 32 bits
  Value clkHigh32 = getClockReg("%clock_hi");

  // Combine into 64-bit value: (high << 32) | low
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value clkLow64 = b.zext(i64_ty, clkLow32);
  Value clkHigh64 = b.zext(i64_ty, clkHigh32);
  Value clock64 = b.or_(b.shl(clkHigh64, b.i64_val(32)), clkLow64);
  return clock64;
}

// Global timer for absolute timestamps (nanoseconds)
Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  // globaltimer is a 64-bit global nanosecond counter
  // Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringRef globalTimeIntrinsicName = "llvm.nvvm.read.ptx.sreg.globaltimer";
  Value globalTimeVal = LLVM::createLLVMIntrinsicCallOp(
                            rewriter, loc, globalTimeIntrinsicName, i64_ty, {})
                            .getResult(0);
  return globalTimeVal;
}

// SM ID for hardware identification
Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return NVVM::SmIdOp::create(rewriter, loc, i32_ty);
}
```

**ReadCounterOp Conversion:**

**File:** [third_party/proton/Dialect/lib/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.cpp#48-71](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.cpp#48)

```cpp
struct ReadCounterOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp> {

  LogicalResult matchAndRewrite(mlir::triton::proton::gpu::ReadCounterOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Determine if 32 or 64 bit clock
    bool isClock64 = false;
    auto intType = mlir::cast<IntegerType>(op.getResult().getType());
    isClock64 = intType.getWidth() == 64;

    // Call target-specific clock reading
    Value clock = targetInfo.clock(rewriter, op.getLoc(), isClock64);
    rewriter.replaceOp(op, clock);
    return success();
  }
};
```

**CircularStore Implementation:**

**File:** [third_party/proton/Dialect/lib/ProtonGPUToLLVM/Utility.cpp#104-135](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/Utility.cpp#104)

```cpp
// Memory layout for each profiling entry (8 bytes total):
//
// tag and upper clock (4 bytes):
//   31:21 scope ID (11 bits, max 2048 scopes)
//   20:11 reserved (10 bits)
//   10:0  64-bit clock bit 32:42 (11 bits)
//
// lower clock (4 bytes):
//   31:0 64-bit clock bit 0:31

void emitCircularStore(proton::gpu::CircularStoreOp op,
                       Value segmentBase, Value indexPtr,
                       ConversionPatternRewriter &rewriter,
                       const TargetInfoBase &targetInfo) {
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value clock = op.getCounter();
  auto clkTy = mlir::cast<IntegerType>(clock.getType());

  // Pack scope ID and is_start flag into tag
  Value scopeId = b.i32_val(op.getScopeId());
  Value isStart = b.i32_val(op.getIsStart() ? 1 : 0);
  Value tag = b.or_(b.shl(scopeId, b.i32_val(21)),
                    b.shl(isStart, b.i32_val(20)));

  // Handle 64-bit clock packing
  Value valsVec;
  if (clkTy.getWidth() == 64) {
    // Pack upper 11 bits of clock into tag
    Value clkHigh = b.trunc(i32_ty, b.lshr(clock, b.i64_val(32)));
    Value clkHighMasked = b.and_(clkHigh, b.i32_val(0x7FF));  // 11 bits
    tag = b.or_(tag, clkHighMasked);
    Value clkLow = b.trunc(i32_ty, clock);
    valsVec = packLLVector(loc, {tag, clkLow}, rewriter);
  } else {
    valsVec = packLLVector(loc, {tag, clock}, rewriter);
  }

  // Get current index and compute buffer offset
  Value index = b.load(i32_ty, indexPtr);
  Value bufOffset = b.add(segmentBase, index);

  // Store to buffer (shared or global memory)
  Value bufPtr = b.gep(bufferType, i32_ty, bufferBase, bufOffset);
  b.store(valsVec, bufPtr);

  // Update circular index: (index + 2) % bufferSize
  Value nextIndex = b.add(index, b.i32_val(2));  // 2 words per entry
  Value wrappedIndex = b.urem(nextIndex, b.i32_val(segmentSize));
  b.store(wrappedIndex, indexPtr);
}
```

**State BEFORE (ProtonGPU IR):**
```mlir
%clk = proton_gpu.read_counter : i64 {metric = cycle}
proton_gpu.circular_store start %segment, %clk {scopeId = 0}
```

**State AFTER (LLVM IR with inline PTX):**
```llvm
; Read clock low 32 bits
%clk_low = call i32 asm sideeffect "mov.u32 $0, %clock;", "=r"()
; Read clock high 32 bits
%clk_high = call i32 asm sideeffect "mov.u32 $0, %clock_hi;", "=r"()
; Combine to 64-bit
%clk_low64 = zext i32 %clk_low to i64
%clk_high64 = zext i32 %clk_high to i64
%clk_high_shifted = shl i64 %clk_high64, 32
%clock64 = or i64 %clk_high_shifted, %clk_low64

; Pack tag: (scopeId << 21) | (isStart << 20) | (clk_high & 0x7FF)
%scope_shifted = shl i32 0, 21      ; scope_id = 0
%start_shifted = shl i32 1, 20      ; is_start = 1
%tag_partial = or i32 %scope_shifted, %start_shifted
%clk_upper = trunc i64 %clock64 to i32  ; get bits 32-42
%clk_upper_masked = and i32 %clk_upper, 2047
%tag = or i32 %tag_partial, %clk_upper_masked

; Store to circular buffer
%buf_ptr = getelementptr i32, ptr addrspace(3) %segment_base, i32 %index
store <2 x i32> <%tag, %clk_low>, ptr addrspace(3) %buf_ptr

; Update circular index
%next_idx = add i32 %index, 2
%wrapped_idx = urem i32 %next_idx, %segment_size
store i32 %wrapped_idx, ptr %index_ptr
```

---

### Frame 16: InitializeOp and FinalizeOp Conversion

**File:** [third_party/proton/Dialect/lib/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.cpp#73-159](../third_party/proton/Dialect/lib/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.cpp#73)

```cpp
struct InitializeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::InitializeOp> {

  LogicalResult matchAndRewrite(mlir::triton::proton::gpu::InitializeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Header layout (total: circularHeaderSize = 40 bytes = 10 words)
    //  +-------------------------------+ 0
    //  | preamble (0xdeadbeef)         |  <- Magic number for validation
    //  +-------------------------------+ 1
    //  | program id (linear CTA id)    |  <- Identifies which CTA
    //  +-------------------------------+ 2
    //  | hw id (SM id)                 |  <- Hardware SM identifier
    //  +-------------------------------+ 3
    //  | buffer size (bytes)           |  <- Total buffer capacity
    //  +-------------------------------+ 4
    //  | init time (64-bit ns)         |  <- Kernel start timestamp
    //  +-------------------------------+ 6
    //  | pre-final time (64-bit ns)    |  <- Before finalize
    //  +-------------------------------+ 8
    //  | post-final time (64-bit ns)   |  <- After finalize
    //  +-------------------------------+ 10

    Value scratchPtr = adaptor.getScratchPtr();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Only first thread in block writes header
    Value threadId = getThreadId(rewriter, loc);
    Value isFirstThread = b.icmp_eq(threadId, b.i32_val(0));

    // Create conditional block
    Block *ifBlock = rewriter.splitBlock(...);
    cf::CondBranchOp::create(rewriter, loc, isFirstThread, ifBlock, thenBlock);

    // Write preamble magic number
    Value preamble = b.i32_val(0xdeadbeef);
    Value gmemPreamblePtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, b.i32_val(0));
    b.store(preamble, gmemPreamblePtr);

    // Write linear program ID
    Value pid = getLinearId(loc, rewriter);  // pidX + pidY*gridDimX + ...
    Value gmemPidPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, b.i32_val(1));
    b.store(pid, gmemPidPtr);

    // Write SM hardware ID
    Value hwid = targetInfo.processorId(rewriter, loc);  // NVVM::SmIdOp
    Value gmemHwidPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, b.i32_val(2));
    b.store(hwid, gmemHwidPtr);

    // Write init timestamp (global nanosecond timer)
    Value initTime = targetInfo.globalTime(rewriter, loc);
    Value gmemInitTimePtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, b.i32_val(4));
    b.store(initTime, gmemInitTimePtr);  // 64-bit store

    rewriter.eraseOp(op);
    return success();
  }
};
```

**FinalizeOp - Copy profiling data from shared to global memory:**

```cpp
struct FinalizeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp> {

  LogicalResult matchAndRewrite(...) const override {
    // Memory layout (per-CTA scratch buffer):
    //  +---------------------------------------+
    //  | header (circularHeaderSize = 40 bytes)|
    //  +---------------------------------------+
    //  | warp index (4 bytes × numWarps)       |  <- Current write position per warp
    //  +---------------------------------------+
    //  | profiled data (allocBufferSize bytes) |  <- Clock + scope records
    //  +---------------------------------------+

    // Step 1: Block leader writes buffer size and pre-final timestamp
    emitBlockLeaderPrologue(op, isBlockFirstThread, scratchPtr, ...);

    // Step 2: Each warp leader writes its current buffer index
    emitWarpIndexWriteback(op, isWarpFirstThread, warpId, ...);

    // Step 3: Copy from shared memory to global memory (if buffer in smem)
    if (segmentBaseTy.getAddressSpace() == 3) {
      emitWarpCopySection(op, laneId, threadsPerWarp, ...);
    }

    // Step 4: Block leader writes post-final timestamp
    emitBlockLeaderEpilogue(op, isBlockFirstThread, scratchPtr, ...);
  }

private:
  Block *emitWarpCopySection(...) const {
    // Parallel copy: each lane copies records in warp-strided pattern
    // Loop: for (idx = laneId*2; idx < effectiveWords; idx += warpSize*2)
    //         load from shared[segment + idx]
    //         store to global[metadata + segment + idx]

    auto copyWord = [&](Value bufOffset, Value gmemOffset, Attribute memory) {
      Value ptr = b.gep(bufferBaseType, i32_ty, segmentObj.base, bufOffset);
      Value load = tritonTargetInfo.loadShared(rewriter, loc, ptr, i32_ty, b.true_val());
      Value gmemPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemOffset);
      b.store(load, gmemPtr);
    };

    // Copy tag and counter words
    copyWord(bufTagOffset, gmemWbTagOffset, memSpace);
    copyWord(bufCounterOffset, gmemWbCounterOffset, memSpace);
  }
};
```

---

### Frame 17: LLVM IR to PTX Translation

**File:** [third_party/nvidia/backend/compiler.py#280-310](../third_party/nvidia/backend/compiler.py#280)

```python
@staticmethod
def make_ptx(mod, metadata, options):
    """Translate LLVM IR to PTX assembly"""
    # Serialize LLVM module to text
    llvm_ir = str(mod)

    # Call LLVM backend for PTX generation
    ptx = llvm.translate_to_ptx(
        llvm_ir,
        compute_capability=metadata["cc"],  # e.g., 90 for H100
        ptx_version=metadata["ptx_version"],  # e.g., 85
        enable_fp_fusion=True,
    )

    # Add PTX metadata
    if options.enable_debug:
        ptx = add_debug_info_to_ptx(ptx)

    return ptx
```

**Example Generated PTX (clock reading):**

```ptx
.version 8.5
.target sm_90a
.address_size 64

.visible .entry add_kernel(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 output_ptr,
    .param .u32 n_elements
)
{
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .pred %p<8>;

    // Read clock registers for profiling
    mov.u32 %r1, %clock;           // Read low 32 bits
    mov.u32 %r2, %clock_hi;        // Read high 32 bits

    // Combine into 64-bit value
    cvt.u64.u32 %rd1, %r1;
    cvt.u64.u32 %rd2, %r2;
    shl.b64 %rd3, %rd2, 32;
    or.b64 %rd4, %rd3, %rd1;       // %rd4 = 64-bit clock

    // Pack tag: (scopeId << 21) | (isStart << 20) | (clockHi & 0x7FF)
    shl.b32 %r3, %r_scopeId, 21;
    shl.b32 %r4, %r_isStart, 20;
    or.b32 %r5, %r3, %r4;
    and.b32 %r6, %r2, 2047;        // Mask 11 bits of clock_hi
    or.b32 %r7, %r5, %r6;          // Final tag

    // Store to circular buffer in shared memory
    st.shared.v2.u32 [%r_buf_ptr], {%r7, %r1};

    // ... kernel body ...
}
```

---

### Frame 18: PTX to CUBIN Compilation

**File:** [third_party/nvidia/backend/compiler.py#312-350](../third_party/nvidia/backend/compiler.py#312)

```python
@staticmethod
def make_cubin(src, metadata, options):
    """Compile PTX to CUBIN using ptxas"""
    import subprocess
    import tempfile

    # Write PTX to temporary file
    with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False) as f:
        f.write(src.encode())
        ptx_path = f.name

    cubin_path = ptx_path.replace('.ptx', '.cubin')

    # Build ptxas command
    arch = f"sm_{metadata['cc']}"  # e.g., sm_90
    cmd = [
        options.ptxas_path or 'ptxas',
        '-arch', arch,
        '-o', cubin_path,
        ptx_path,
    ]

    if options.enable_debug:
        cmd.extend(['-g', '-lineinfo'])

    if options.opt_level:
        cmd.extend(['-O' + str(options.opt_level)])

    # Run ptxas
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ptxas failed: {result.stderr}")

    # Read compiled binary
    with open(cubin_path, 'rb') as f:
        cubin = f.read()

    return cubin
```

**Final CUBIN contains:**
- Machine code for GPU execution
- Embedded profiling instrumentation
- Register allocation
- Shared memory layout

---

### Complete Data Flow with C++ Passes

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE END-TO-END COMPILATION TRACE                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  PYTHON LAYER                                                                          │
│  ────────────                                                                          │
│  @triton.jit                                                                           │
│  def kernel():                                                                         │
│      pl.enter_scope("foo")     ───┐                                                   │
│      x = tl.load(ptr)              │   ast_to_ttir()                                  │
│      pl.exit_scope("foo")      ───┘                                                   │
│                                     │                                                  │
│  ╔═══════════════════════════════════╧════════════════════════════════════════════╗   │
│  ║                           TTIR (Triton IR)                                      ║   │
│  ╠═════════════════════════════════════════════════════════════════════════════════╣   │
│  ║  proton.record start @"foo"                                                     ║   │
│  ║  %x = tt.load %ptr : tensor<1024xf32>                                           ║   │
│  ║  proton.record end @"foo"                                                       ║   │
│  ╚═══════════════════════════════════╤════════════════════════════════════════════╝   │
│                                       │                                                │
│                     ConvertTritonToTritonGPU (C++)                                     │
│                     TritonToTritonGPUPass.cpp:796                                      │
│                                       │                                                │
│  ╔═══════════════════════════════════╧════════════════════════════════════════════╗   │
│  ║                          TTGIR (TritonGPU IR)                                   ║   │
│  ╠═════════════════════════════════════════════════════════════════════════════════╣   │
│  ║  proton.record start @"foo"                                                     ║   │
│  ║  %x = tt.load %ptr : tensor<1024xf32, #blocked<...>>                            ║   │
│  ║  proton.record end @"foo"                                                       ║   │
│  ╚═══════════════════════════════════╤════════════════════════════════════════════╝   │
│                                       │                                                │
│                     ProtonToProtonGPU (C++)                                            │
│                     ProtonToProtonGPUPass.cpp:74                                       │
│                     ScopeIdAllocation.cpp:48                                           │
│                                       │                                                │
│  ╔═══════════════════════════════════╧════════════════════════════════════════════╗   │
│  ║                    TTGIR + ProtonGPU Ops                                        ║   │
│  ╠═════════════════════════════════════════════════════════════════════════════════╣   │
│  ║  %segment = proton_gpu.segment_alloc %buffer                                    ║   │
│  ║  proton_gpu.initialize %scratch_ptr                                             ║   │
│  ║                                                                                 ║   │
│  ║  %clk0 = proton_gpu.read_counter : i64 {metric = cycle}                         ║   │
│  ║  proton_gpu.circular_store start %segment, %clk0 {scopeId = 0}                  ║   │
│  ║  %x = tt.load %ptr : tensor<1024xf32, #blocked<...>>                            ║   │
│  ║  %clk1 = proton_gpu.read_counter : i64 {metric = cycle}                         ║   │
│  ║  proton_gpu.circular_store end %segment, %clk1 {scopeId = 0}                    ║   │
│  ║                                                                                 ║   │
│  ║  proton_gpu.finalize %segment, %scratch_ptr                                     ║   │
│  ╚═══════════════════════════════════╤════════════════════════════════════════════╝   │
│                                       │                                                │
│                     ConvertTritonGPUToLLVM (C++)                                       │
│                     TritonGPUToLLVM.cpp:79                                             │
│                                       │                                                │
│                     ProtonGPUToLLVM (C++)                                              │
│                     PatternProtonGPUOpToLLVM.cpp:48                                    │
│                     TargetInfo.cpp:12 (NVIDIA clock)                                   │
│                                       │                                                │
│  ╔═══════════════════════════════════╧════════════════════════════════════════════╗   │
│  ║                          LLVM IR + NVVM                                         ║   │
│  ╠═════════════════════════════════════════════════════════════════════════════════╣   │
│  ║  ; Read clock via inline PTX assembly                                           ║   │
│  ║  %clk_lo = call i32 asm sideeffect "mov.u32 $0, %clock;", "=r"()                ║   │
│  ║  %clk_hi = call i32 asm sideeffect "mov.u32 $0, %clock_hi;", "=r"()             ║   │
│  ║  %clk64 = or i64 (shl (zext %clk_hi), 32), (zext %clk_lo)                       ║   │
│  ║                                                                                 ║   │
│  ║  ; Pack tag and store to circular buffer                                        ║   │
│  ║  %tag = or (shl %scopeId, 21), (or (shl %isStart, 20), (and %clk_hi, 0x7FF))   ║   │
│  ║  %buf_ptr = getelementptr ptr addrspace(3) %segment, i32 %index                 ║   │
│  ║  store <2 x i32> <%tag, %clk_lo>, ptr addrspace(3) %buf_ptr                     ║   │
│  ║                                                                                 ║   │
│  ║  ; Load operation lowered to vectorized loads                                   ║   │
│  ║  %vec = load <4 x float>, ptr %ptr, align 16                                    ║   │
│  ╚═══════════════════════════════════╤════════════════════════════════════════════╝   │
│                                       │                                                │
│                     llvm.translate_to_ptx                                              │
│                                       │                                                │
│  ╔═══════════════════════════════════╧════════════════════════════════════════════╗   │
│  ║                          PTX Assembly                                           ║   │
│  ╠═════════════════════════════════════════════════════════════════════════════════╣   │
│  ║  .version 8.5                                                                   ║   │
│  ║  .target sm_90a                                                                 ║   │
│  ║                                                                                 ║   │
│  ║  mov.u32 %r1, %clock;                    // Read clock low                      ║   │
│  ║  mov.u32 %r2, %clock_hi;                 // Read clock high                     ║   │
│  ║  cvt.u64.u32 %rd1, %r1;                                                         ║   │
│  ║  cvt.u64.u32 %rd2, %r2;                                                         ║   │
│  ║  shl.b64 %rd3, %rd2, 32;                                                        ║   │
│  ║  or.b64 %rd4, %rd3, %rd1;                // 64-bit clock                        ║   │
│  ║                                                                                 ║   │
│  ║  shl.b32 %r3, %r_scope, 21;              // Pack scope ID                       ║   │
│  ║  or.b32 %r4, %r3, %r_start;              // Add isStart flag                    ║   │
│  ║  st.shared.v2.u32 [%r_buf], {%r4, %r1};  // Store to smem                       ║   │
│  ║                                                                                 ║   │
│  ║  ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd_ptr];  // Vectorized load            ║   │
│  ╚═══════════════════════════════════╤════════════════════════════════════════════╝   │
│                                       │                                                │
│                     ptxas (NVIDIA assembler)                                           │
│                                       │                                                │
│  ╔═══════════════════════════════════╧════════════════════════════════════════════╗   │
│  ║                          CUBIN (GPU Binary)                                     ║   │
│  ╠═════════════════════════════════════════════════════════════════════════════════╣   │
│  ║  • Machine code for execution on GPU                                            ║   │
│  ║  • Embedded clock reading instructions                                          ║   │
│  ║  • Shared memory layout for circular buffer                                     ║   │
│  ║  • Register allocation optimized                                                ║   │
│  ╚═════════════════════════════════════════════════════════════════════════════════╝   │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

The Proton DSL profiling system works through several key mechanisms:

1. **Compile-time instrumentation**: `pl.enter_scope()` / `pl.exit_scope()` calls are converted to `proton.record` MLIR operations, which are then lowered to GPU clock reads and buffer stores.

2. **Hook-based lifecycle**: The `InstrumentationHook` registers compiler passes during `proton.start()` and manages data collection during kernel execution.

3. **Circular buffer strategy**: Profiling data is stored in a per-warp circular buffer in shared or global memory, allowing efficient recording with minimal overhead.

4. **Mode configuration**: Users can tune profiling behavior (clock precision, warp sampling, buffer type) via the `mode` parameter.

5. **Output formats**: Finalization generates either aggregated metrics (hatchet format) or timeline traces (Chrome trace format) for analysis.

6. **Complete C++ pipeline**: The compilation flows through multiple MLIR passes:
   - **TritonToTritonGPU**: Adds GPU layout encodings (`BlockedEncodingAttr`, `DotOperandEncodingAttr`)
   - **ProtonToProtonGPU**: Converts `proton.record` to `gpu.read_counter` + `gpu.circular_store`
   - **TritonGPUToLLVM**: Lowers GPU ops to LLVM IR with NVVM intrinsics
   - **ProtonGPUToLLVM**: Lowers profiling ops to PTX inline assembly (`%clock`, `%clock_hi`)
   - **LLVM→PTX→CUBIN**: Final translation to executable GPU binary

The key distinction between Triton and Gluon profiling is that Gluon generates TTGIR directly with explicit layout control, making instrumentation more reliable since the compiler doesn't reorder operations as aggressively.
