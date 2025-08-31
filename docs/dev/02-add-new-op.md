# Chapter 2 — Add A New Op And Lower It To PTX

This chapter shows a concrete, end-to-end change set that adds a trivial `tt.dummy_sync` op and wires it through to PTX (Hopper). It complements Chapter 1 (pipeline overview).

What you’ll do:
- Define `tt.dummy_sync` (TT dialect).
- Keep it through TT→TTG conversion.
- Lower to LLVM with `LLVM::InlineAsmOp` emitting `bar.sync 0;`.
- Add a lit test that proves the PTX contains `bar.sync`.

Already applied in this repo:
- Op definition: include/triton/Dialect/Triton/IR/TritonOps.td
- TT→TTG conversion registration: lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp
- LLVM lowering: lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp
- Test: test/Conversion/dummy_sync_to_ptx.mlir

You can use these diffs as a reference template when adding real ops.

## Minimal Test

Run:

```
triton-opt test/Conversion/dummy_sync_to_ptx.mlir \
  --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' \
  --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' \
  --convert-nv-gpu-to-llvm \
| mlir-translate --mlir-to-llvmir | opt -O3 -S \
| llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 -o - \
| FileCheck test/Conversion/dummy_sync_to_ptx.mlir --check-prefix=PTX
```

Expected: FileCheck passes and PTX contains `bar.sync 0;`.

## Next Steps
- Replace inline PTX with an NVVM op once a suitable intrinsic exists.
- Add memory effects to `tt.dummy_sync` if you want to model barriers precisely.
- Provide a `triton.language` frontend wrapper if desired.

