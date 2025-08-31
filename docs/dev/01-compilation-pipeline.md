# Triton Compilation Pipeline to PTX on Hopper (sm_90)

This is a stage-by-stage transcript showing how a `tl.dot` turns into Hopper WGMMA + TMA in PTX. Each step links to the relevant source with VSCode-style file:line references.

## 1) Front-End Triton IR (TT)
- Dot op definition: `tt.dot a, b, c` with `inputPrecision` and `maxNumImpreciseAcc`.
  - include/triton/Dialect/Triton/IR/TritonOps.td:658-692

What it means: `$d = a @ b + c`, with optional TF32 selection (tf32/tf32x3/ieee) used later.

## 2) TT → TritonGPU (TTG): add distributed layouts and dot-operand encodings
- Conversion pattern that rewrites `tt.dot` and inserts `ttg.convert_layout` for A/B/C, assigning dot-operand encodings to A/B and a TTG result encoding:
  - lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:205-272
    - Picks a blocked result encoding (lines 233-239), ensures A/B use `DotOperandEncodingAttr` (lines 252-263), converts C/result (line 264), and recreates `tt.dot` with enriched types (lines 266-270).

- Encodings used downstream:
  - NVMMAShared (SMEM tiles for MMAv3/5): include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:445-510
  - NvidiaMmaEncoding (result accumulator layout; Hopper-aware): include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1242-1326
  - DotOperandEncoding (operand A/B register layout; also used with Hopper to carry `kWidth`): include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1397-1460

Why it matters: Hopper WGMMA expects B in NVMMA SMEM layout and A either in registers (dot_op) or NVMMA SMEM; result carries a Hopper `nvidia_mma` encoding.

## 3) NVIDIA planning and Hopper-specific legalization
- NVIDIA pass menu used around matmul lowering:
  - include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:27-119 (CTA planning, fence insertion, MMA/TMA lowering, descriptor/tmem optimization)

- Hopper WGMMA op in TTNG:
  - Op definitions (async flag and wait): include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:76-110 (warp_group_dot), 112-127 (warp_group_dot_wait)
  - Semantic/shape/type verification for Hopper constraints (M%64, N%8, num warps % 4, A/B layout kinds, accumulator precision limits): lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:65-104

Effect: `tt.dot` in a Hopper-targeted pipeline is represented as `ttng.warp_group_dot` (optionally `isAsync=true`) plus `ttng.warp_group_dot_wait` placements.

## 4) TMA (Tensor Memory Accelerator) for moving tiles + mbarriers
- Lowering of descriptor loads/stores to TMA async copies with mbarriers:
  - lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:26-61
    - Allocates SMEM for destination and a 64-bit mbarrier object (lines 37-48), signals expected byte count (lines 49-55), issues `ttng.async_tma_copy_*` (user-provided lambda), then `wait_barrier` and `inval_barrier` (lines 56-58).
  - TMA load emission: lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:67-79 (creates `ttng.async_tma_copy_global_to_local` at lines 75-76)
  - TMA store emission: lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:99-117 (`ttng.async_tma_copy_local_to_global` and `ttng.tma_store_wait`)

Impact: Global↔Shared transfers occur via PTX `cp.async.bulk.tensor.*` guarded by `mbarrier.*`, ensuring correct overlap with WGMMA.

## 5) Software pipelining and making WGMMA async on Hopper
- WGMMAPipeline converts sync `warp_group_dot` to `isAsync=true` and inserts waits to cap in-flight groups, threading values through waits:
  - lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:656-689 (flip to async and insert appropriate `warp_group_dot_wait`)
  - Example threading and final wait emission: lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:643-654, 680-691

Outcome: Loads (TMA/cp.async) and compute (WGMMA) are overlapped, with explicit limits on outstanding WGMMA groups.

## 6) Shared/Tensor memory allocation, proxy fences, and kernel shape
- Passes that annotate SMEM, allocate global scratch for TMA descriptors, and assign warpgroups:
  - include/triton/Conversion/TritonGPUToLLVM/Passes.td:1-36 (allocate-shared-memory), 18-36 (global scratch), 30-36 (allocate warpgroups)

## 7) TT/TTG/TTNG → LLVM/NVVM → PTX for sm_90
- Test pipeline invoking the exact flags for Hopper and PTX ISA 8.3:
  - test/Conversion/tritongpu_to_ptx.mlir:1-2
    - allocate-shared-memory-nv='compute-capability=90 ptx-version=83'
    - convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83'
    - convert-nv-gpu-to-llvm | mlir-translate | opt -O3 | llc -mtriple nvptx64 -mcpu=sm_90 -mattr=+ptx83

What gets emitted:
- `ttng.warp_group_dot` → NVVM WGMMA intrinsics → PTX `wgmma.mma_async.aligned.*`, `wgmma.wait_group`.
- `ttng.async_tma_copy_*` + barriers → PTX `cp.async.bulk.tensor.*`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`, `mbarrier.inval`.
- `ttg.local_alloc` → `.shared` allocations with annotated offsets from allocation analysis.

## Quick Mental Model
- TT `tt.dot` → TTG dot-operand/result encodings → TTNG `warp_group_dot` (+waits) with TMA-fed tiles → NVVM/PTX: WGMMA + cp.async.bulk.tensor + mbarrier on Hopper (sm_90).

---

If you’d like, I can add a second file that shows a small `tl.dot` kernel and dump its IR at TT → TTG → TTNG to map each transformation 1:1 with the references above.

