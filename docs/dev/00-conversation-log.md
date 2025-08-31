# Conversation Log — Triton Compilation (Hopper) and New-Op Tutorial

Date: 2025-08-31
Author: AI assistant (Codex CLI)

This document captures the key explanations and steps from our session, so you can review the rationale and mapping later.

## 1) Hopper (sm_90) Pipeline Walkthrough for `tl.dot`

Big Picture
- Front-End (TT): Triton parses `tl.dot(a, b, c)` to `tt.dot %a, %b, %c` with attributes like `inputPrecision`.
- Mid-End (TTG): `convert-triton-to-tritongpu` assigns distributed layouts, converts A/B to `ttg.dot_op` encodings, shapes C/result.
- NVIDIA (TTNG): Hopper-only features (WGMMA + TMA/mbarrier) use `ttng.*` ops: `ttng.warp_group_dot`, `ttng.warp_group_dot_wait`, `ttng.async_tma_copy_*`, `ttng.init/wait/inval_barrier`, etc.
- Back-End: `convert-triton-gpu-to-llvm` lowers to LLVM/NVVM IR; `llc -mcpu=sm_90` emits PTX with `wgmma.mma_async.*`, `cp.async.bulk.tensor.*`, and `mbarrier.*`.

Dialects In Play
- Triton (tt): high-level ops/types (TT_DotOp). See include/triton/Dialect/Triton/IR/TritonOps.td.
- TritonGPU (ttg): layouts/encodings + smem/tmem descriptors; encodings include `DotOperandEncoding`, `NVMMASharedEncoding`, `NvidiaMmaEncoding`.
- TritonNvidiaGPU (ttng): Hopper intrinsics surface — WGMMA/TMA/mbarrier ops & waits.

Analyses & Passes (selected)
- AxisInfo, Allocation, Membar analyses inform placement and buffer offsets.
- TT → TTG: `convert-triton-to-tritongpu` (TritonDotPattern) inserts `ttg.convert_layout` and assigns dot encodings.
- TTG optimizations: `tritongpu-accelerate-matmul`, `tritongpu-optimize-dot-operands`, `tritongpu-remove-layout-conversions`, etc.
- NVIDIA passes: `triton-nvidia-tma-lowering`, `triton-nvidia-mma-lowering`, fence/proxy-fence insertion, TMEM optimization.
- Pipelining/warp specialization: `tritongpu-pipeline`, `WGMMAPipeline` converts to `isAsync=true` WGMMA and inserts `ttng.warp_group_dot_wait`.

End-to-end lowering (Hopper)
- `tt.dot` → TTG with `ttg.dot_op` A/B and Hopper `nvidia_mma` result.
- Legalized to `ttng.warp_group_dot` (+ `ttng.warp_group_dot_wait`).
- TMA loads/stores: `ttng.async_tma_copy_*` with `ttng.init_barrier` + `ttng.barrier_expect` + `ttng.wait_barrier` + `ttng.inval_barrier`.
- PTX: `wgmma.mma_async.aligned.*`, `wgmma.wait_group`, `cp.async.bulk.tensor.*`, `mbarrier.*`, `.shared` allocations.

References (examples used during explanation)
- TT_DotOp: include/triton/Dialect/Triton/IR/TritonOps.td:658-692
- TritonDotPattern (TT→TTG): lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:205-272
- Encodings: include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td (NVMMAShared, NvidiaMmaEncoding, DotOperandEncoding)
- WGMMA op + verify: include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:76-127, lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:65-104
- TMA lowering: lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp
- WGMMAPipeline: lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:643-691
- PTX path (test driver): test/Conversion/tritongpu_to_ptx.mlir:1-2

## 2) Artifacts Added — Chapter 1 (Pipeline) and Chapter 2 (New Op)

- Chapter 1: docs/dev/01-compilation-pipeline.md
  - A curated, file:line-linked walkthrough of the Hopper lowering of `tl.dot` to PTX.
- Chapter 2: docs/dev/02-add-new-op.md
  - Tutorial for adding a new op and lowering it to PTX via LLVM inline asm.

## 3) Code Changes Implemented (Dummy Op)

Goal: Provide a small, complete example of adding an op and making it show up in PTX.

- New tutorial op `tt.dummy_sync` (pure, no operands/results):
  - include/triton/Dialect/Triton/IR/TritonOps.td
    - Added TT_DummySyncOp near the end of the file.
- TT→TTG conversion: keep the op through conversion by registering a generic pattern:
  - lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp
    - Added `GenericOpPattern<triton::DummySyncOp>` to `populateTritonPatterns`.
- TTG→LLVM: lower the op to PTX via LLVM inline asm:
  - lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp
    - Added `DummySyncOpConversion` which emits `bar.sync 0;` using `LLVM::InlineAsmOp`.
    - Registered the pattern inside `populateElementwiseOpToLLVMPatterns`.
- Test: prove PTX contains `bar.sync 0;` after the full toolchain:
  - test/Conversion/dummy_sync_to_ptx.mlir
    - RUN pipeline matches Hopper (compute-capability=90, ptx-version=83) and FileCheck asserts `bar.sync 0;`.

Why inline PTX? This keeps the example focused and avoids creating new NVVM intrinsics; the NVPTX backend passes it through to PTX cleanly.

## 4) Docs Restructuring

- Moved the initial pipeline explanation into docs/dev/01-compilation-pipeline.md.
- Wrote a clear how-to for new ops in docs/dev/02-add-new-op.md.
- Removed older duplicates to avoid confusion.

## 5) Suggested Next Steps

- If you want a real barrier model, add MemoryEffects to `tt.dummy_sync` and route to an NVVM/MLIR op instead of inline PTX.
- Add a `triton.language` Python wrapper to exercise the op from Python kernels.
- For more complex ops, consider a TTNG op and a dedicated NVVM lowering path.

---

This log mirrors our explanations and the changes performed, with enough pointers to review the diffs and tests later.

## Verbatim Transcript

> User: explain the various dialects in Triton's compilation pipeline. Walk me through specifically the tl.dot is lowered to PTX, tracing through each dialect, analysis, transformation, optimization, and conversion pass. Do this specifically for Hopper (sm90) CUDA architecture

> Assistant: I’m going to scan the repo for dialects and passes related to Triton’s lowering (dot/mma, NVVM/WGMMA) to ground the walkthrough.

> Assistant (explanation):
Big Picture
- Front-End (Python → TT): Triton parses your `@triton.jit` kernel into MLIR Triton dialect (`tt.*`). Your `tl.dot(a, b, c, ...)` is materialized as `tt.dot %a, %b, %c` with attributes like `inputPrecision` (ieee/tf32/tf32x3) and `maxNumImpreciseAcc`.
- Mid-End (TT → TTG): Pass “convert-triton-to-tritongpu” adds explicit distributed layouts/encodings and memory descriptor types in the TritonGPU dialect (`ttg.*`). This is where dot operands are put into “dot-operand” encodings and where many TritonGPU optimization passes run.
- NVIDIA-Specific (TTNG): Hopper-only features (WGMMA, TMA, mbarrier) are expressed in the TritonNvidiaGPU dialect (`ttng.*`) via ops like `ttng.warp_group_dot`, `ttng.warp_group_dot_wait`. These appear after NVIDIA passes insert/convert them.
- Back-End (LLVM/NVVM → PTX): “convert-triton-gpu-to-llvm” lowers TT/TTG/TTNG to LLVM + NVVM IR and then to PTX for sm_90. Tests in this repo drive `mlir-translate` and `llc -mtriple nvptx64 -mcpu=sm_90` to emit PTX.

Dialects You’ll See
- Triton (tt): high-level ops/types (including `tt.dot`).
- TritonGPU (ttg): distributed layouts, shared-memory/tensor-memory descriptors, convert-layouts, async copies, loop pipelining helpers. Key encodings: `ttg.dot_op`, `ttg.nvidia_mma`, NVMMAShared.
- TritonNvidiaGPU (ttng): Hopper ops for WGMMA and TMA/mbarrier (`ttng.warp_group_dot`, `ttng.warp_group_dot_wait`, `ttng.async_tma_copy_*`, `ttng.init_barrier`...).

Analyses That Inform Lowering
- Axis info/dataflow; Allocation (shared/tmem); Membar/barrier planning.

Key Optimization/Lowering Passes (Hopper sm_90)
- TT → TTG: `convert-triton-to-tritongpu` (adds `ttg.convert_layout`, dot-operand/result encodings).
- TTG passes: `tritongpu-accelerate-matmul`, `tritongpu-optimize-dot-operands`, `tritongpu-remove-layout-conversions`, `tritongpu-F32DotTC`, etc.
- NVIDIA-specific: `triton-nvidia-tma-lowering`, `triton-nvidia-mma-lowering`, TMEM optimizations, fence insertion.
- Software pipelining/warp specialization: `tritongpu-pipeline`, `WGMMAPipeline` (switches to `isAsync=true`, inserts waits).
- Memory allocation & kernel shape: `allocate-shared-memory`, `tritongpu-global-scratch-memory-allocation`, `tritongpu-allocate-warp-groups`.

The tl.dot Walkthrough (Hopper, sm_90)
- Step 0: TT `tt.dot` from `tl.dot`.
- Step 1: TT→TTG: A/B → `ttg.dot_op`; C/result moved to Hopper-friendly result encoding.
- Step 2: Hopper encodings fixed; B in NVMMAShared SMEM; result is Hopper `nvidia_mma`.
- Step 3: Introduce Hopper `ttng.warp_group_dot` (+ `ttng.warp_group_dot_wait`).
- Step 4: Move tiles with TMA + barriers: `ttng.async_tma_copy_*`, `ttng.init/wait/inval_barrier`.
- Step 5: Pipeline: `isAsync=true` WGMMA and `warp_group_dot_wait` placement.
- Step 6: Allocate smem/tmem; finalize shapes and fences.
- Step 7: TT/TTG/TTNG → LLVM/NVVM → PTX (WGMMA/TMA/mbarrier PTX shows up).

> User: Please first create markdown transcript of your explanation with VSCode-formatted pathnames and line numbers linking correlated code sections to each compilation step / stage. Save this markdown as triton-compilation.md.

> Assistant: I’m going to add a new Markdown file with a concise, linked walkthrough and code references.

> Assistant: Done. I added triton-compilation.md with a concise, linkable walkthrough of each stage and the exact file:line references that back it up.

> User: Yes - that would be helpful. I am ultimately interested in adding new ops and modifying various stages of the compilation pipeline... What would be super-helpful is if you could create a tutorial that creates a new op ... and adds the needed wiring through the entire pipeline to lower it to valid PTX.

> Assistant: I’m about to add a developer tutorial and example MLIR file under docs/dev/ that show how to add a new Triton op and wire it through to PTX using inline PTX via LLVM::InlineAsmOp.

> Assistant: Added a developer tutorial with a concrete, end-to-end pattern you can follow (docs/dev/new-op-tutorial.md).

> User: That would be appreciated. Let's restructure the docs/dev so that the previous tutorial is Chapter 1, and your new proposed tutorial is Chapter 2.

> Assistant: Great idea — I restructured docs/dev into Chapter 1 and Chapter 2, and implemented ready-to-apply code changes to add the dummy op and lowering.

> User: Please first generate a markdown of your previous explanation and include it within the docs/dev folder so that I have a log of our conversations.

> Assistant: Added docs/dev/00-conversation-log.md summarizing the pipeline and changes.

> User: Yes please.

> Assistant: Appended the Verbatim Transcript section (this section) with the raw conversation highlights for later review.

## Full Verbatim (Expanded)

> Assistant (preamble): I’m going to scan the repo for dialects and passes related to Triton’s lowering (dot/mma, NVVM/WGMMA) to ground the walkthrough.

> Assistant (Hopper pipeline explanation):
Big Picture
- Front-End (Python → TT): Triton parses your `@triton.jit` kernel into MLIR Triton dialect (`tt.*`). Your `tl.dot(a, b, c, ...)` is materialized as `tt.dot %a, %b, %c` with attributes like `inputPrecision` (ieee/tf32/tf32x3) and `maxNumImpreciseAcc`.
- Mid-End (TT → TTG): Pass “convert-triton-to-tritongpu” adds explicit distributed layouts/encodings and memory descriptor types in the TritonGPU dialect (`ttg.*`). This is where dot operands are put into “dot-operand” encodings and where many TritonGPU optimization passes run.
- NVIDIA-Specific (TTNG): Hopper-only features (WGMMA, TMA, mbarrier) are expressed in the TritonNvidiaGPU dialect (`ttng.*`) via ops like `ttng.warp_group_dot`, `ttng.warp_group_dot_wait`, `ttng.async_tma_copy_*`, and `ttng.wait_barrier`/`ttng.init_barrier`/`ttng.inval_barrier`. These appear after NVIDIA passes insert/convert them.
- Back-End (LLVM/NVVM → PTX): `convert-triton-gpu-to-llvm` lowers TT/TTG/TTNG to LLVM + NVVM IR and then to PTX for sm_90. Tests in this repo drive `mlir-translate` and `llc -mtriple nvptx64 -mcpu=sm_90` to emit PTX.

Dialects You’ll See
- Triton (tt): high-level ops/types (including `tt.dot`).
- TritonGPU (ttg): distributed layouts, shared-memory/tensor-memory descriptors, convert-layouts, async copies, loop pipelining helpers. Key encodings: `ttg.dot_op`, `ttg.nvidia_mma`, `ttg.nvmma_shared`.
- TritonNvidiaGPU (ttng): Hopper ops for WGMMA and TMA/mbarrier (`ttng.warp_group_dot`, `ttng.warp_group_dot_wait`, `ttng.async_tma_copy_*`, `ttng.init_barrier`, `ttng.wait_barrier`, `ttng.inval_barrier`).

Analyses That Inform Lowering
- Axis info/dataflow (include/triton/Analysis/AxisInfo.h), Allocation (shared/tmem) (include/triton/Analysis/Allocation.h), Membar/barrier planning (include/triton/Analysis/Membar.h).

Key Optimization/Lowering Passes (Hopper sm_90)
- TT → TTG: `convert-triton-to-tritongpu` assigns dot-operand/result encodings; TritonDotPattern rewrites `tt.dot` and inserts `ttg.convert_layout`.
- TTG generic: `tritongpu-accelerate-matmul`, `tritongpu-optimize-dot-operands`, `tritongpu-coalesce`, `tritongpu-remove-layout-conversions`, `tritongpu-F32DotTC` (tf32x3 trick), etc.
- NVIDIA-specific: `triton-nvidia-tma-lowering` (descriptor_* to `ttng.async_tma_*` + barriers), `triton-nvidia-mma-lowering` (prep WGMMA), TMEM layout/interleave/remove-token passes, fence insertion and proxy fence insertion.
- Pipelining/warp specialization: `tritongpu-assign-latencies`, `tritongpu-schedule-loops`, `tritongpu-pipeline`, `WGMMAPipeline` (flip `ttng.warp_group_dot` to `isAsync=true`, insert `ttng.warp_group_dot_wait`).
- Memory allocation/kernel shape: `allocate-shared-memory`, `tritongpu-global-scratch-memory-allocation`, `tritongpu-allocate-warp-groups`.

The tl.dot Walkthrough (Hopper, sm_90)
- Step 0: In TT (front-end): `tt.dot` from `tl.dot` (TT_DotOp).
- Step 1: TT → TTG: `convert-triton-to-tritongpu` makes A/B use `ttg.dot_op` encodings and C/result get distributed encoding; emits `ttg.convert_layout` as needed.
- Step 2: Make it Hopper MMA-friendly: ensure B in NVMMA shared layout, A in dot_op or NVMMA shared, result in `NvidiaMmaEncodingAttr isHopper()`.
- Step 3: Introduce Hopper WGMMA: `ttng.warp_group_dot` (+ `ttng.warp_group_dot_wait`), obeying M%64, N%8, warps%4, and supported element types.
- Step 4: Use TMA: `ttng.async_tma_copy_*` guarded by `ttng.barrier_expect`, `ttng.wait_barrier`, `ttng.inval_barrier`.
- Step 5: Pipeline: `isAsync=true` WGMMA with `ttng.warp_group_dot_wait` to bound in-flight groups; overlap data movement and compute.
- Step 6: Allocate SMEM/TMEM; add fences; finalize kernel shape/warpgroups.
- Step 7: Convert to NVVM/PTX: emit `wgmma.mma_async.aligned.*`, `wgmma.wait_group`, `cp.async.bulk.tensor.*`, `mbarrier.*` and `.shared` buffers.

> Assistant: Done. I added triton-compilation.md with a concise, linkable walkthrough of each stage and the exact file:line references that back it up.

> Assistant: Added a developer tutorial with a concrete, end-to-end pattern you can follow.

> Assistant: Great idea — I restructured docs/dev into “Chapter 1” and “Chapter 2,” and I also implemented the complete, ready-to-apply code changes for the dummy op, wiring it through the pipeline all the way to PTX.

> Assistant: Added a conversation log markdown capturing the prior explanations and changes.

## Commands Executed (Repository Queries)

Note: Executed from repo root. Some commands produced empty output or benign errors; they’re included for completeness.

```
rg -n "dialect|TritionGPU|triton_gpu|triton-gpu|wgmma|mma_async|cp.async|TMA|mbarrier|dot|tl.dot|mma.sync|NVVM" -S  # broad grep over repo for relevant keywords
sed -n '1,200p' test/Conversion/tritongpu_to_ptx.mlir  # inspect Hopper test pipeline flags and checks
sed -n '1,260p' test/TritonNvidiaGPU/tma_lowering.mlir  # review TMA/barrier lowering examples
rg -n "Passes.td|triton-nvidia|TritonNvidia|TritonGPU" include lib -S  # find pass definitions and locations
sed -n '1,220p' include/triton/Dialect/TritonGPU/Transforms/Passes.td  # TTG passes (pipelining, specialization)
sed -n '220,640p' include/triton/Dialect/TritonGPU/Transforms/Passes.td  # continuation
ls include/triton/Dialect | sed -n '1,200p'  # list dialect directories
ls include/triton/Dialect/TritonNvidiaGPU | sed -n '1,200p'  # list NVIDIA dialect dirs
ls include/triton/Dialect/TritonNvidiaGPU/IR | sed -n '1,200p'  # list NVIDIA IR files
sed -n '1,280p' include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td  # NVIDIA ops part 1
sed -n '280,860p' include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td  # NVIDIA ops part 2 (TMA, barriers)
sed -n '1,280p' include/triton/Dialect/Triton/IR/TritonOps.td  # Triton ops (TT), locate TT_DotOp region
rg -n "def TT_DotOp|warp_group_dot|mma|wgmma|NvidiaMmaEncoding|dot_op<|dot" include/triton -S  # quick index of key defs
sed -n '640,760p' include/triton/Dialect/Triton/IR/TritonOps.td  # TT_DotOp definition
sed -n '1,160p' include/triton/Conversion/TritonToTritonGPU/Passes.td  # TT→TTG pass decl
sed -n '1,240p' include/triton/Conversion/TritonGPUToLLVM/Passes.td  # TTG→LLVM pass decl
sed -n '1,220p' include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td  # NVIDIA-specific passes
sed -n '1,220p' include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td  # TTG encodings index
rg -n "NVMMASharedEncodingAttr|NvidiaMmaEncodingAttr|dot_op<|mma" include/triton/Dialect/TritonGPU -S  # locate encoding defs
sed -n '420,520p' include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td  # NVMMAShared
sed -n '1240,1520p' include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td  # NvidiaMma, DotOperandEncoding
rg -n "wgmma|WarpGroupDot|WarpGroupDotWait|mma.sync|nvvm|NVVM.*wgmma|cp.async.bulk|tma|mbarrier|nvvm\\.mbarrier|TMA.*LLVM" lib include -S  # impl references
sed -n '180,360p' lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp  # dot-to-FMA utility
sed -n '1,220p' lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp  # TritonDotPattern start
sed -n '220,400p' lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp  # TritonDotPattern continuation
sed -n '460,560p' include/triton/Conversion/TritonGPUToLLVM/Utility.h  # shared mem helpers & layout application
sed -n '1,160p' lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp  # WGMMA verifier
rg -n "ModuleAllocation|AxisInfo|Analyze|analysis|Occupancy|estimate|Resource|Shared memory used|num warps" include lib -S  # analyses
rg -n "Lower.*WarpGroupDot|WarpGroupDot.*LLVM|wgmma.*LLVM|NVVM::Wgmma|nvvm\\.wgmma|createWgmma|mma_async|Lower.*TMA|cp\\.async\\.bulk|mbarrier|nvvm\\.mbarrier|TMA.*LLVM" lib -S  # conversion hooks
sed -n '1,260p' lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp  # descriptor_* → TMA lowering
sed -n '1,260p' lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp  # TMEM layout helpers
nl -ba include/triton/Dialect/Triton/IR/TritonOps.td | sed -n '630,780p'  # numbered TT_DotOp region for citations
nl -ba lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp | sed -n '200,320p'  # numbered TritonDotPattern
nl -ba include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td | sed -n '420,520p'  # numbered NVMMAShared excerpt
nl -ba include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td | sed -n '1240,1460p'  # numbered NvidiaMma/DotOperand
nl -ba include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td | sed -n '70,140p'  # numbered WGMMA ops td
nl -ba lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp | sed -n '40,120p'  # numbered WGMMA verifier
nl -ba include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td | sed -n '1,200p'  # numbered NVIDIA passes
nl -ba test/Conversion/tritongpu_to_ptx.mlir | sed -n '1,40p'  # numbered Hopper PTX test header
nl -ba lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp | sed -n '1,120p'  # numbered TMA lowering prologue
rg -n "populateTritonPatterns\(|populateTritonPatterns\(" lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp -n -S  # locate registration
sed -n '520,620p' lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp  # registration areas
rg -n "populate.*ToLLVMPatterns\(|TritonGPUToLLVM.*Pass|applyPartialConversion|addConversionPass|create.*ToLLVM" include lib -S  # LLVM populator discovery
sed -n '1,140p' include/triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h  # declared populators
rg -n "populate(Elementwise|Memory|View|Histogram|Reduce|Scan|Gather|ConvertLayout|ControlFlow|SPMD|FuncOp|PrintOp|Instrumentation)Op.*\\\(" lib/Conversion/TritonGPUToLLVM -S  # implemented populators
sed -n '1,200p' lib/Conversion/TritonGPUToLLVM/CMakeLists.txt  # sources list
sed -n '640,760p' lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp  # populator tail (insertion point for dummy op)
sed -n '1,40p' lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp  # file includes/top
git status --porcelain  # track modified/added files
ls docs/dev | sed -n '1,200p'  # list docs after changes
test -f triton-compilation.md && echo EXISTS || echo MISSING  # check old file presence
sed -n '1,240p' triton-compilation.md  # read (before moving) if existed
wc -l include/triton/Dialect/Triton/IR/TritonOps.td  # file line count around TT_DotOp
sed -n '1200,1400p' include/triton/Dialect/Triton/IR/TritonOps.td  # TT tail segments
sed -n '1400,1416p' include/triton/Dialect/Triton/IR/TritonOps.td  # TT end
ls -l docs/dev | sed -n '1,200p'  # final docs snapshot
```
