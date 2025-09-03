# TMA Gather Rows Lowering: End-to-End Trace

This document traces the `tma_gather_rows_kernel` from Python through MLIR passes to the NVVM/PTX emission of `cp.async.bulk.tensor.2d.tile::gather4` (Tile::gather4) instructions.


## Detailed Step-by-Step With Paths and Lines

1) Python kernel
- python/test/unit/language/test_tensor_descriptor.py:1356 — def tma_gather_rows_kernel(...)
- python/test/unit/language/test_tensor_descriptor.py:1396 — tma_gather_rows_kernel[(1, )](...)

```python
# python/test/unit/language/test_tensor_descriptor.py:1356-1363
@triton.jit
def tma_gather_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr, BLOCK_X: tl.constexpr,
                           BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    out = desc.gather(idx, y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)
```

2) Python -> Triton IR
- API entry: python/triton/language/core.py:2223–2296 — def make_tensor_descriptor(...)
  - Calls semantic: python/triton/language/semantic.py:1899–1961 — def make_tensor_descriptor(...)
    - Emits builder.create_make_tensor_descriptor(...)

```python
# python/triton/language/core.py:2226-2269 (excerpt)
@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object ... """
    padding_option = _unwrap_if_constexpr(padding_option)
    return _semantic.make_tensor_descriptor(base, shape, strides, block_shape, padding_option)
```

```python
# python/triton/language/semantic.py:1899-1931 (excerpt)
def make_tensor_descriptor(self, base: TensorTy, shape: List[TensorTy], strides: List[TensorTy],
                           block_shape: List[tl.constexpr], padding_option: str = "zero") -> tl.tensor_descriptor:
    # validate dims, strides, block shape...
    shape = [self.make_scalar(x, tl.int32) for x in shape]
    strides = [self.make_scalar(tl._unwrap_if_constexpr(x), tl.int64) for x in strides]
    padding = self._str_to_padding_option(padding_option)
    handle = self.builder.create_make_tensor_descriptor(base.handle, [s.handle for s in shape],
                                                        [s.handle for s in strides], block_shape,
                                                        is_signed_int, padding)
    return tl.tensor_descriptor(handle, shape, strides, type)
```

3) Triton Ops (dialect)
- include/triton/Dialect/Triton/IR/TritonOps.td:1040 — def TT_MakeTensorDescOp
- include/triton/Dialect/Triton/IR/TritonOps.td:1356 — def TT_DescriptorGatherOp

```tablegen
# include/triton/Dialect/Triton/IR/TritonOps.td (excerpts)
def TT_MakeTensorDescOp : TT_Op<"make_tensor_descriptor", [Pure, SameVariadicOperandSize]> {
  let arguments = (ins TT_Ptr:$base, Variadic<I32>:$shape, Variadic<I64>:$strides,
                   DefaultValuedAttr<TT_PaddingOptionAttr, "::mlir::triton::PaddingOption::PAD_ZERO">:$padding);
  let results = (outs TT_TensorDescType:$result);
  let assemblyFormat = "$base `,` `[` $shape `]` `,` `[` $strides `]` attr-dict `:` type($base) `,` type($result)";
}

def TT_DescriptorGatherOp : TT_Op<"descriptor_gather", [TT_DescriptorOpInterface]> {
  let arguments = (ins Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
                   RankedTensorOf<[I32]>:$x_offsets, I32:$y_offset);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = "$desc `[` $x_offsets `,` $y_offset `]` attr-dict `:` functional-type(operands, results)";
}
```

4) Triton → NVIDIA-GPU TMA Lowering
- lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:48–67 — TMACreateDescLowering (alloc descriptor + create tensormap)
- lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:83–97 — TMAGatherLowering (descriptor_gather → async_tma_gather)
- lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:26–61 — lowerTMALoad (alloc smem result, barrier expect/wait)

5) TMA utilities (descriptor fields)
- lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:233–321 — createTMADesc (emits ttng.tensormap.create)
- lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:113–141 — getTMABlockShape
- lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:193–231 — getTMAElementType

6) Indices relayout for gather4
- lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:127–169 — getNewIndicesType + convertGatherScatterOp glue
- lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:577–579 — pattern registration GatherScatterOpPattern<DescriptorGatherOp>
- lib/Conversion/TritonToTritonGPU/RelayoutTritonGPU.cpp:116–121 — pattern registration GatherScatterOpPattern<ttng::AsyncTMAGatherOp>

7) NVVM / PTX emission of Tile::gather4
- third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1567–1713 — iterateGatherScatterIndices (quad grouping + smem dst + per-message y)
- third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:145–190 — AsyncTMAGatherOpConversion::matchAndRewrite
- PTX template site (same file): around 160–176 builds string: 
  "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];"

8) Host vs device descriptor
- Descriptor is allocated in device global memory and created on-device:
  - lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:56–65 — GlobalScratchAllocOp (128B, 128B aligned)
  - lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:233–321 — ttng.tensormap.create fills fields
  - Acquire fenceproxy after creation before first use (TMALowering.cpp lines 61–64)
- The interpreter-only host path exists for tests marked @pytest.mark.interpreter:
  - python/triton/runtime/interpreter.py:1414–1461 — create_make_tensor_descriptor
  - python/triton/runtime/interpreter.py:1440–1450 — create_descriptor_gather (CPU-side emulation)

## MLIR Primer (for novices)
- Operation syntax: `dialect.opname operands attr-dict : types`. Dialect is the namespace (e.g., `tt`, `ttng`, `ttg`, `arith`).
- Types: MLIR prints dialect-qualified types like `!tt.tensordesc<tensor<1x128xbf16, #shared>>` or `!ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>`.
- Attributes: key–value metadata (e.g., `cacheModifier = "NONE"`). `attr-dict` is the `{ ... }` section in op assembly.
- AssemblyFormat: TableGen (`.td`) drives how ops print/parse. Backticks in the format denote literal tokens.
- Patterns: Passes transform IR using `OpRewritePattern` / `OpConversionPattern`. `matchAndRewrite` matches an op and uses a `PatternRewriter` to replace it.
- Type conversion: `TypeConverter` + `applyPartialConversion`/`applyPatternsGreedily` orchestrate multi-op rewrites.
- Adaptor objects: Provide converted operands/results during rewrite (`OpAdaptor`, `OneToNOpAdaptor`).
- SSA values: `%0`, `%1`, etc., are value names; operations can have results used by later ops.

## Pass-By-Pass Deep Dive (with inputs/outputs and line-by-line notes)

### Triton → NVIDIA TMA Lowering (TMALowering.cpp)
- High-level input: `tt.make_tensor_descriptor`, `tt.descriptor_{load,store,gather,scatter,reduce}`.
- High-level output: TMA descriptor materialization + async copy/gather/scatter ops in the `ttng` dialect, with correct shared-memory allocations and mbarrier usage.

Line-by-line: `lowerTMALoad` (26–61)
- 26–29: Helper signature with `createLoad` callback.
- 31–37: Create destination `!ttg.memdesc` in shared memory based on descriptor’s encoding.
- 38–47: Allocate and initialize a 1-slot mbarrier in shared memory.
- 49–55: Compute `sizeInBytes` for `barrier.expect` with `pred = 1`.
- 55: Call `createLoad(...)` (specialized per op type) to enqueue the async transaction.
- 56–58: `wait` and `invalidate` the barrier.
- 59–60: Replace original op’s result with a load from the local allocation; erase op.

Line-by-line: `TMAGatherLowering` (83–97)
- 86–93: Emit `ttng.async_tma_gather %desc[%x, %y] %barrier, %result, %pred` via the callback.
- 94: Delegate buffer/barrier handling to `lowerTMALoad`.

Line-by-line: `TMACreateDescLowering` (167–186)
- 175–176: Allocate 128B/128B-aligned device-global space for the tensormap.
- 177–179: Build the tensormap via `createTMADesc`.
- 180: Acquire fenceproxy.
- 181–184: Replace the original `tt.make_tensor_descriptor` with `ttng.reinterpret_tensor_descriptor` to the tensormap object.

### Tensormap Utilities (TMAUtilities.cpp)
- High-level input: Descriptor’s base, shape, strides, block shape, padding.
- High-level output: A fully-populated tensormap via `ttng.tensormap.create`.

Line-by-line: `createTMADesc` (233–321)
- 238–241: Helper to create i32 constants.
- 243–249: Inspect element type and shared encoding; detect FP4 padding.
- 250–255: Compute `blockShape` / `contigDimSize` and validate FP4 constraints.
- 261–263: Build `box_dim`: `[contigDimSize, blockShape[k-1], ...]`.
- 265–274: Get `swizzleBytes` and validate non-MMA encodings.
- 276–281: Get `swizzle_mode`; build `elemSize` value.
- 284–288: Build reversed `global_dim` and `global_stride` (omit last stride).
- 290–294: Adjust FP4 padded last dim.
- 296: `element_strides = [1,...,1]`.
- 298–300: Convert global strides from elements to bytes.
- 302–305: Map element type to TMA enum.
- 307: Select fill mode (0=zero, 1=nan).
- 309–320: Create the tensormap.

### Indices Relayout for gather4 (TritonGPUConversion.cpp)
- High-level input: `x_offsets` tensor with any distributed layout.
- High-level output: `x_offsets` converted so every warp sees groups of 4 consecutive indices.

Line-by-line: `getNewIndicesType` (131–154)
- 137–144: Define a 2D parent layout with packs-of-4 along register dimension; broadcast across warps.
- 149–153: Slice to 1D and clone type if layout changes.

Line-by-line: `impl::convertGatherScatterOp` (171–183)
- 175–180: Modify operands/results to converted types; call `convertGatherScatterIndices` to insert `ConvertLayoutOp` for `x_offsets`.

### NVVM/PTX Emission (LoadStoreOpToLLVM.cpp)
- High-level input: `ttng.async_tma_gather` with descriptor pointer, offsets, y, barrier, result, and predicate.
- High-level output: Inline PTX `cp.async.bulk.tensor.2d.tile::gather4...` invocations covering all rows/segments.

Line-by-line: `iterateGatherScatterIndices` (1566–1693)
- 1591–1602: Validate 4-grouped indices per warp.
- 1604–1618: Validate memdesc shape/encoding and compute base shared pointer.
- 1622–1627: Compute inner/contiguous sizes and number of per-row messages.
- 1634–1643: Build layout mappings and invert to compute shared memory offsets.
- 1645–1661: Build predicate: mask redundant warps and elect one thread per warp.
- 1662–1689: For each register pack and message, compute `(shMemPtr, yOffset)` and invoke callback with 4 x-offsets.

Line-by-line: `AsyncTMAGatherOpConversion::matchAndRewrite` (1695–1749)
- 1710–1714: Get barrier’s shared base pointer.
- 1719–1734: PTX template string and operands: pred (b), dst shmem ptr (r), desc (l), y (r), four x (r,r,r,r), barrier (r).
- 1736–1739: Emit and launch the inline PTX.
- 1741–1745: Call `iterateGatherScatterIndices` to emit all messages.
- 1747–1748: Erase the original op.

## Additional MLIR Notes
- `include/.../*.td` files declare ops and their printing/parsing; `assemblyFormat` shows the textual syntax you’ll see when dumping IR.
- `OpRewritePattern` vs `OpConversionPattern`: the latter is used when types/layouts also change and a `TypeConverter` participates.
- `applyPatternsGreedily` runs until no more rewrites apply; `applyPartialConversion` enforces legality rules while rewriting.
```c++
// lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp (excerpts)
static void lowerTMALoad(Operation *op, RankedTensorType tensorType, Value desc,
                         function_ref<void(Value, Value, Value, Value)> createLoad,
                         PatternRewriter &rewriter) {
  // allocate dst memdesc in shared, setup mbarrier + expect bytes, then call createLoad
}

struct TMAGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  LogicalResult matchAndRewrite(DescriptorGatherOp op, PatternRewriter &rewriter) const override {
    auto createLoad = [&](Value tmaPtr, Value barrierAlloc, Value alloc, Value pred) {
      rewriter.create<triton::nvidia_gpu::AsyncTMAGatherOp>(
          op.getLoc(), tmaPtr, op.getXOffsets(), op.getYOffset(), barrierAlloc, alloc, pred);
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

class TMACreateDescLowering : public OpRewritePattern<MakeTensorDescOp> {
  LogicalResult matchAndRewrite(MakeTensorDescOp op, PatternRewriter &rewriter) const override {
    auto alloc = rewriter.create<triton::gpu::GlobalScratchAllocOp>(
        loc, getPointerType(rewriter.getI8Type()), TMA_SIZE_BYTES, TMA_ALIGN);
    if (failed(createTMADesc(alloc, op, rewriter))) return failure();
    rewriter.create<TensormapFenceproxyAcquireOp>(loc, alloc.getResult());
    auto newDesc = rewriter.create<ReinterpretTensorDescOp>(loc, op.getType(), alloc.getResult());
    rewriter.replaceOp(op, newDesc);
    return success();
  }
};
```
```c++
// lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:233-321 (excerpt)
builder.create<TensormapCreateOp>(
    loc,
    /*desc_ptr=*/tmaPtr,
    /*global_address=*/op.getBase(),
    /*box_dim=*/boxDim,
    /*global_dim=*/globalDim,
    /*global_stride=*/globalStride,
    /*element_strides=*/elementStride,
    /*elem_type*/ builder.getI32IntegerAttr(*elemTypeEnum),
    /*interleave_layout*/ builder.getI32IntegerAttr(0),
    /*swizzle_mode=*/ builder.getI32IntegerAttr(swizzleMode),
    /*fill_mode=*/ builder.getI32IntegerAttr(fillMode));
```
```c++
// lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:127-154 (excerpt)
std::array<unsigned, 2> sizePerThread{1, 4};
std::array<unsigned, 2> threadsPerWarp = {numThreads, 1};
std::array<unsigned, 2> order = {1, 0};
auto parentEncoding = BlockedEncodingAttr::get(ctx, sizePerThread, threadsPerWarp, warpsPerCta, order, ctaLayout);
auto newEncoding = SliceEncodingAttr::get(ctx, /*dim=*/0, parentEncoding);
```
```c++
// lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:171-183 (excerpt)
rewriter.modifyOpInPlace(op, [&] {
  for (auto [operand, value] : llvm::zip(op->getOpOperands(), operands)) operand.set(value);
  for (OpResult result : op->getOpResults()) result.setType(typeConverter.convertType(result.getType()));
  result = convertGatherScatterIndices(op, xOffsetsMutable, rewriter);
});
```
```c++
// third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp (excerpts)
if (xCoordsLayout.getInDimSize(kRegister) < 4)
  return op->emitError("must have at least 4 x offsets per warp");
// ... compute msg layout and shared offsets ...
std::string tmaInst = "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared"
                      "::cluster.global.mbarrier::complete_tx::bytes "
                      "[$1], [$2, {$3, $4, $5, $6, $7}], [$8];";
```
```c++
// AsyncTMAGatherOpConversion callback (1716+)
SmallVector<PTXBuilder::Operand *, 9> operands{
  ptxBuilder.newOperand(pred, "b"),
  ptxBuilder.newOperand(shMemPtr, "r"),
  ptxBuilder.newOperand(adaptor.getDesc(), "l"),
  ptxBuilder.newOperand(yOffset, "r")};
for (Value xOffset : xOffsets) operands.push_back(ptxBuilder.newOperand(xOffset, "r"));
operands.push_back(ptxBuilder.newOperand(barrierMemObj.getBase(), "r"));
auto &tma = *ptxBuilder.create<>(tmaInst);
tma(operands, /*attachOnlyMLIRArgs=*/true);
ptxBuilder.launch(rewriter, loc, voidTy);
```
## Backend Debugging and Tracing Knobs

- Kernel dump/override (python/triton/knobs.py:335+)
  - TRITON_KERNEL_DUMP=1: dump kernel IR/binaries to TRITON_DUMP_DIR (default ~/.triton/dump)
  - TRITON_DUMP_DIR=/path: choose dump directory
  - TRITON_ALWAYS_COMPILE=1: force recompilation each run (disable cache reuse)
  - TRITON_KERNEL_OVERRIDE=1 and TRITON_OVERRIDE_DIR=/path: load user-edited IR/binaries from override dir

- MLIR pipeline dumps (python/src/ir.cc:1818+)
  - MLIR_ENABLE_DUMP=1 or function-name substring: print pass manager IR (before/after on failure)
  - MLIR_DUMP_PATH=/path/file.mlir: redirect MLIR dumps to a file

- NVPTX / PTXAS artifacts (python/triton/knobs.py:160–164)
  - NVPTX_ENABLE_DUMP=1: dump NVPTX (LLVM→NVVM/PTX stage)
  - TRITON_DUMP_PTXAS_LOG=1: emit ptxas logs (register usage, spills)
  - DISABLE_PTXAS_OPT=1: ask ptxas to lower optimizations for readability

- LLVM IR dumps (python/src/llvm.cc:84,334)
  - LLVM_IR_ENABLE_DUMP=1: dump generated LLVM IR modules

- Runtime/debug
  - TRITON_DEBUG=1: enable runtime debug logging
  - TRITON_INTERPRET=1: run kernels in the Python interpreter
  - TRITON_OVERRIDE_ARCH=sm_90: test codegen for a specific arch

Example session to capture everything in ./dump:
```bash
export TRITON_DUMP_DIR=$PWD/dump
export TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1
export MLIR_ENABLE_DUMP=1 MLIR_DUMP_PATH=$TRITON_DUMP_DIR/pipeline.mlir
export LLVM_IR_ENABLE_DUMP=1 NVPTX_ENABLE_DUMP=1 TRITON_DUMP_PTXAS_LOG=1
python -m pytest -k test_tma_gather -q
```

Proton helpers (third_party/proton/scripts):
- dump_ttgir.sh: runs with TRITON_ALWAYS_COMPILE/TRITON_KERNEL_DUMP/TRITON_DUMP_DIR and collects TTGIR dumps.

## Reproducing Concrete TTIR/TTGIR/LLIR/PTX Artifacts (Works on sm90 hosts)

You have two options. Option A uses compile-only (AOT-style) so the host GPU does not need to support TMA gather; Option B uses JIT with an override arch and dump knobs.

### Option A: Compile-only (recommended)
- Script: `tma_gather_exploration/dump_artifacts.py`
- What it does: builds `tma_gather_rows_kernel` and compiles for a chosen arch (default sm_100) using `triton.compile(ASTSource(...), target=GPUTarget("cuda", 100, 32))`. It writes `tma_gather_rows.{ttir,ttgir,llir,ptx}` and `cubin` under `tma_gather_exploration/out/`.

Run:
```bash
python tma_gather_exploration/dump_artifacts.py \
  --out-dir tma_gather_exploration/out \
  --arch sm_100 --cc 100 --warp 32 \
  --X 128 --Y 128 --block-x 32 --block-y 32 --y-offset 0 --dtype fp32
```

Notes:
- The compile-only path does not execute the kernel, so it works on Hopper (sm_90) hosts.
- Ensure `block-y * element_size >= 16` and `strides=[Y,1]` to satisfy descriptor constraints.
- The script prints whether TTGIR contains `ttng.async_tma_gather` and whether PTX contains `tile::gather4`.

### Option B: JIT with override arch and dumps
If you prefer Triton’s normal JIT, you can run any Python script (including the unit test kernel) with:
```bash
export TRITON_OVERRIDE_ARCH=sm_100
export TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$PWD/dump
export NVPTX_ENABLE_DUMP=1 TRITON_DUMP_PTXAS_LOG=1 LLVM_IR_ENABLE_DUMP=1
python - <<'PY'
import triton, triton.language as tl
# Define/compile the same kernel here or import it
PY
```
Artifacts will show up under `$TRITON_DUMP_DIR`. On sm90 hosts, do not launch kernels that actually execute TMA gather; compile-only is safer.

### Reference: AOT tips
See Lei’s “Triton compiler development tips” for AOT/compile-only flows and artifact collection. The compile-only approach above mirrors the AOT style by using `triton.compile` and saving `k.asm[...]` to files.
