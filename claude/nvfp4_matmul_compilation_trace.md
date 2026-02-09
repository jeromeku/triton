# Triton Compilation Pipeline Trace: Block-Scaled NVFP4 Matrix Multiplication

This document provides a comprehensive frame-by-frame trace of the Triton compilation pipeline, from the Python user kernel through MLIR passes to PTX generation. **Each C++ pass is traced at the source level.**

## Source Kernel

**File:** [python/tutorials/block-scaled-matmul-nvfp4.py#46-110](../python/tutorials/block-scaled-matmul-nvfp4.py#46)

```python
@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        output_type: tl.constexpr,
        ELEM_PER_BYTE_A: tl.constexpr, ELEM_PER_BYTE_B: tl.constexpr,
        VEC_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        rep_m: tl.constexpr, rep_n: tl.constexpr, rep_k: tl.constexpr,
        NUM_STAGES: tl.constexpr,
):
    # Output type selection
    if output_type == 0:
        output_dtype = tl.float32
    # ...

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        # Reshape scales for microscaling format
        scale_a = scale_a.reshape(...).trans(...).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(...).trans(...).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # Block-scaled dot product (microscaling format)
        accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)

        # Update offsets
        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))
```

---

## Compilation Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      TRITON COMPILATION PIPELINE (NVIDIA CUDA)                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────┐                                                                  │
│   │ Python Source   │  @triton.jit def block_scaled_matmul_kernel(...):                │
│   │ (tl.language)   │  tl.dot_scaled(), tl.load(), tl.store()                          │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  @triton.jit decorator → JITFunction                                      │
│            │  CodeGenerator.visit(fn.parse())                                          │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  Triton IR      │  tt.func, tt.load, tt.dot_scaled, scf.for                        │
│   │  (TTIR)         │  High-level tensor operations                                    │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  make_ttir() passes:                                                       │
│            │  - Inliner, RewriteTensorPointer, Canonicalizer                           │
│            │  - Combine, ReorderBroadcast, CSE, LoopUnroll                             │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  Optimized      │  Simplified TTIR with unrolled loops                             │
│   │  TTIR           │                                                                  │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  make_ttgir() passes:                                                      │
│            │  - ConvertTritonToTritonGPU (add GPU layouts)                             │
│            │  - Coalesce, F32DotTC, PlanCTA                                            │
│            │  - AccelerateMatmul (select MMA version)                                  │
│            │  - OptimizeDotOperands, Pipeline                                          │
│            │  - FenceInsertion, LowerMMA                                               │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  TritonGPU IR   │  + BlockedEncodingAttr, NvidiaMmaEncodingAttr                    │
│   │  (TTGIR)        │  + DotOperandEncodingAttr, SharedEncodingAttr                    │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  make_llir() passes:                                                       │
│            │  - SCFToCF, AllocateSharedMemory                                          │
│            │  - ConvertTritonGPUToLLVM (main lowering)                                 │
│            │  - NVGPUToLLVM, WarpSpecializeToLLVM                                      │
│            │  - NVVMToLLVM, DIScope                                                    │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  LLVM IR        │  LLVM dialect + NVVM dialect                                     │
│   │  (MLIR)         │  Inline PTX assembly for MMA instructions                        │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  llvm.to_module(), llvm.optimize_module(O3)                               │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  LLVM IR        │  Native LLVM module                                              │
│   │  (Native)       │                                                                  │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  make_ptx(): llvm.translate_to_asm()                                      │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  PTX Assembly   │  .version 8.x, .target sm_90a                                    │
│   │                 │  mma.sync, ldmatrix, cp.async                                    │
│   └────────┬────────┘                                                                  │
│            │                                                                            │
│            │  make_cubin(): ptxas                                                      │
│            ▼                                                                            │
│   ┌─────────────────┐                                                                  │
│   │  CUBIN          │  GPU binary executable                                           │
│   │  (GPU binary)   │                                                                  │
│   └─────────────────┘                                                                  │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Frame 1: @triton.jit Decorator

**File:** [python/triton/runtime/jit.py#34-100](../python/triton/runtime/jit.py#34)

When `@triton.jit` decorates a function, it creates a `JITFunction` object:

```python
class JITFunction(KernelInterface[T]):
    def __init__(self, fn, version=None, do_not_specialize=None, ...):
        self.fn = fn
        self.module = fn.__module__
        self.arg_names = [p.name for p in signature.parameters.values()]

        # Parse function source for dependency tracking
        self.src = textwrap.dedent(inspect.getsource(fn))

        # Build dependency hash for caching
        tree = ast.parse(self.src)
        finder = DependenciesFinder(name=fn.__name__, globals=fn.__globals__, ...)
        finder.visit(tree)
        self.hash = finder.ret  # SHA256 of source + dependencies
```

---

## Frame 2: Kernel Compilation Entry Point

**File:** [python/triton/compiler/compiler.py#52-82](../python/triton/compiler/compiler.py#52)

```python
class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None):
        self.fn = fn
        self.language = Language.TRITON
        self.ext = "ttir"  # Output will be TTIR
        self.name = fn.__name__
        self.signature = signature  # Type signature for arguments
        self.constants = dict()     # Constexpr values

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(self.fn, self, context=context, options=options,
                           codegen_fns=codegen_fns, module_map=module_map)
```

---

## Frame 3: Python AST to TTIR Conversion

**File:** [python/triton/compiler/code_generator.py#1620-1659](../python/triton/compiler/code_generator.py#1620)

```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    # Build argument types from signature
    arg_types = [None] * len(fn.arg_names)
    for k, v in src.signature.items():
        idx = fn.arg_names.index(k)
        arg_types[idx] = str_to_ty(v, None)  # e.g., "*fp16" -> tt.ptr<f16>

    # Create the code generator
    prototype = ASTFunction([], arg_types, src.attrs)
    generator = CodeGenerator(
        context, prototype,
        gscope=fn.get_capture_scope(),
        function_name=fn.repr(proxy),
        jit_fn=fn,
        is_kernel=True,
        options=options,
        codegen_fns=codegen_fns,
        module_map=module_map,
    )

    # Visit the parsed AST to generate MLIR
    generator.visit(fn.parse())  # <-- Main IR generation

    module = generator.module
    return module
```

---

## Frame 4: tl.dot_scaled to MLIR

**File:** [python/triton/language/semantic.py#1620-1671](../python/triton/language/semantic.py#1620)

When the code generator encounters `tl.dot_scaled()`:

```python
class TritonSemantic:
    def dot_scaled(self, lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
                   acc, fast_math, lhs_k_pack, rhs_k_pack, out_dtype):
        # Convert format strings to enum
        lhs_format_enum = self._str_to_fp_type(lhs_format)  # "e4m3" -> FP8E4M3
        rhs_format_enum = self._str_to_fp_type(rhs_format)  # "e2m1" -> FP4E2M1

        # Create tt.dot_scaled operation
        return tensor(
            self.builder.create_dot_scaled(
                lhs.handle, lhs_scale_handle, lhs_format_enum,
                rhs.handle, rhs_scale_handle, rhs_format_enum,
                fast_math, lhs_k_pack, rhs_k_pack, acc_handle
            ),
            ret_ty
        )
```

**Generated TTIR:**
```mlir
%result = tt.dot_scaled %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    {lhs_format = e4m3, rhs_format = e2m1}
    : tensor<128x128xf8E4M3> * tensor<128x128xf4E2M1> -> tensor<128x128xf32>
```

---

## Frame 5: TTIR Optimization Passes (make_ttir)

**File:** [third_party/nvidia/backend/compiler.py#234-249](../third_party/nvidia/backend/compiler.py#234)

```python
@staticmethod
def make_ttir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    if capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)

    pm.run(mod, 'make_ttir')
    return mod
```

---

# DETAILED C++ PASS TRACING

---

## Frame 6: make_ttgir() - Python Entry Point

**File:** [third_party/nvidia/backend/compiler.py#251-323](../third_party/nvidia/backend/compiler.py#251)

```python
@staticmethod
def make_ttgir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)

    # ===== PHASE 1: Convert to GPU IR =====
    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}",
        opt.num_warps, 32, opt.num_ctas)

    # ===== PHASE 2: Memory Optimization =====
    passes.ttgpuir.add_coalesce(pm)

    # ===== PHASE 3: Matmul Acceleration =====
    passes.ttgpuir.add_f32_dot_tc(pm, capability >= 80)
    nvidia.passes.ttnvgpuir.add_plan_cta(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    passes.ttgpuir.add_accelerate_matmul(pm)  # KEY: MMA version selection
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)

    # ===== PHASE 4: Hopper-specific (SM90+) =====
    if capability // 10 in [8, 9]:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)

    # ===== PHASE 5: Final Optimizations =====
    passes.ttgpuir.add_prefetch(pm)
    passes.ttgpuir.add_coalesce_async_copy(pm)
    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)  # TMA for Hopper
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)

    # ===== PHASE 6: Hardware-specific Lowering =====
    nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
    nvidia.passes.ttnvgpuir.add_lower_mma(pm)

    pm.run(mod, 'make_ttgir')
    return mod
```

---

## Frame 7: ConvertTritonToTritonGPU Pass (C++)

**File:** [lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp](../lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp)

This pass converts Triton IR to TritonGPU IR by adding GPU-specific layout encodings.

### Pass Declaration (Lines 14-17)
```cpp
namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
}
```

### Key Conversion Patterns (Lines 32-112)

**GenericOpPattern**: Converts arith and math operations with layout encodings
```cpp
template <typename Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
  LogicalResult matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Add encoding to result type
    Type newType = getTypeConverter()->convertType(op.getType());
    Op::create(rewriter, op.getLoc(), newType, adaptor.getOperands());
  }
};
```

**ArithConstantPattern**: Handles arith::ConstantOp with layout encoding (Lines 85-110)
```cpp
struct ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
  LogicalResult matchAndRewrite(arith::ConstantOp op, ...) const {
    // For tensor constants, add blocked encoding
    auto rankedType = dyn_cast<RankedTensorType>(op.getType());
    Attribute encoding = BlockedEncodingAttr::get(ctx, shape, ...);
    auto newType = RankedTensorType::get(shape, elementType, encoding);
    arith::ConstantOp::create(rewriter, loc, newType, newAttr);
  }
};
```

**TritonExpandDimsPattern** (Lines 133-189): Handles dimension expansion with proper encoding propagation

### Registered Operations (Lines 32-80)
- **Arith ops**: Add, Sub, Mul, Div, CeilDiv, Rem, And, Or, Xor, ShL, ShR
- **Math ops**: Exp, Floor, Ceil, Cos, Sin, Log, Erf, Abs, Sqrt, Rsqrt, Fma
- **Cast ops**: Trunc, Ext, SIToFP, FPToSI, FPToUI, UIToFP
- **Compare/Select ops**: CmpF, CmpI, Select

---

## Frame 8: Coalesce Pass (C++)

**File:** [lib/Dialect/TritonGPU/Transforms/Coalesce.cpp](../lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)

Optimizes memory access patterns for L1 cache efficiency.

### Pass Implementation (Lines 71-121)
```cpp
struct CoalescePass : public impl::TritonGPUCoalesceBase<CoalescePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);  // Run axis info analysis

    // Step 1: Build coalesced layouts for memory operations
    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr) return;
      auto tensorType = cast<RankedTensorType>(ptr.getType());
      // Build coalesced layout based on access pattern
      auto layout = buildCoalescedEncoding(axisInfoAnalysis, tensorType, ...);
      layoutMap[curr] = layout;
    });

    // Step 2: Pick layouts for descriptor load/store ops
    pickDescriptorLoadStoreLayout(moduleOp, layoutMap);

    // Step 3: Convert operands and results with new layouts
    for (auto &kv : layoutMap) {
      convertDistributedOpEncoding(kv.second, kv.first);
    }
  }
};
```

### Key Algorithm
1. Analyzes memory access patterns using `ModuleAxisInfoAnalysis`
2. Builds coalesced memory layouts for optimal L1 cache usage
3. For tensor pointer operations: creates coalesced layout L2, converts operands L1→L2, executes op, converts result L2→L1
4. Special handling for descriptor load/store ops (vectorization up to 16 bytes)

---

## Frame 9: AccelerateMatmul Pass (C++) - **CRITICAL FOR NVFP4**

**File:** [lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](../lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)

This pass selects the appropriate MMA instruction version and converts DotOp/DotScaledOp to use hardware accelerators.

### MMA Version Selection (Lines 34-71)
```cpp
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  SmallVector<int> versionsSupported;
  if (computeCapability < 75) {
    versionsSupported = {1};        // Volta: mma.sync.m8n8k4
  } else if (computeCapability < 90) {
    versionsSupported = {2};        // Turing/Ampere: mma.sync.m16n8k8
  } else if (computeCapability < 100) {
    versionsSupported = {3, 2};     // Hopper: wgmma.mma_async preferred
  } else if (computeCapability < 120) {
    versionsSupported = {5, 2};     // Blackwell: tcgen05.mma
  } else if (computeCapability < 130) {
    versionsSupported = {2};        // Consumer Blackwell
  }

  for (int baseVersion : versionsSupported) {
    if (supportMMA(op, baseVersion))
      return baseVersion;
    // Emit remarks for unsupported configurations
  }
  return 0;
}
```

### Warp Distribution for MMAv2 (Lines 73-135)
```cpp
SmallVector<unsigned> warpsPerTileV2(DotOpInterface dotOp,
                                     const ArrayRef<int64_t> shape,
                                     int numWarps) {
  // Check for chained dots (flash attention pattern)
  auto slices = mlir::getSlice(dotOp, ...);
  for (Operation *op : slices) {
    if (isa<DotOp, DotScaledOp>(op) && (op != dotOp)) {
      if (auto mmaEncoding = dyn_cast<NvidiaMmaEncodingAttr>(...)) {
        return to_vector(mmaEncoding.getWarpsPerCTA());  // Reuse existing
      }
      hasChainedDot = true;
    }
  }

  // Balance repM and repN for register pressure
  SmallVector<int64_t> shapePerWarp = {16, 8};
  while (product(warps) < numWarps) {
    if (reps[0] >= reps[1]) {
      warps[0] *= 2;
    } else {
      warps[1] *= 2;
    }
  }
  return warps;
}
```

### BlockedToMMA Pattern (Lines 360-447) - For Regular DotOp
```cpp
class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  mlir::LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                      mlir::PatternRewriter &rewriter) const {
    auto mmaVersion = getMMAVersionSafe(computeCapability, dotOp);
    auto mmaResult = createMMAEncodingForDot(dotOp, rewriter, computeCapability, mmaVersion);

    if (mmaResult.versionMajor == 3) {
      // MMAv3 (Hopper WGMMA): Put operands in shared memory
      a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose, ...);
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose, ...);
      newDot = triton::nvidia_gpu::WarpGroupDotOp::create(...);
    } else {
      // MMAv2: Convert to DotOperandEncoding
      a = convertDotOperandForMMA(a, 0, minBitwidth, mmaResult.newRetType, rewriter);
      b = convertDotOperandForMMA(b, 1, minBitwidth, mmaResult.newRetType, rewriter);
      newDot = DotOp::create(rewriter, ...);
    }
  }
};
```

### ScaledBlockedToMMAv5 Pattern (Lines 756-895) - **FOR NVFP4 MATMUL**
```cpp
class ScaledBlockedToMMAv5 : public mlir::OpRewritePattern<triton::DotScaledOp> {
  mlir::LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                      mlir::PatternRewriter &rewriter) const {
    // Validate: Blackwell only, 4 or 8 warps, M >= 128, N >= 16
    if (computeCapability < 100 || computeCapability >= 120)
      return failure();
    if (numWarps != 4 && numWarps != 8)
      return failure();
    if (retShapePerCTA[0] < 128 || retShapePerCTA[1] < 16)
      return failure();

    // Handle mixed precision FP4 padding
    bool IsAMixedPrecFp4 = (dotOp.getAElemType() != dotOp.getBElemType()) && isAFP4;
    bool isMMAv5Fp4PaddedLhs = IsAMixedPrecFp4 || !dotOp.getLhsKPack();

    // Put A, B in shared memory with appropriate layouts
    a = getSharedMemoryMMAOperand(a, rewriter, 0, !isAFP4, isMMAv5Fp4PaddedLhs, ...);
    b = getSharedMemoryMMAOperand(b, rewriter, 1, !isBFP4, isMMAv5Fp4PaddedRhs, ...);

    // Create TensorMemoryEncodingAttr for accumulator (TMEM)
    unsigned m = 128;
    unsigned n = retShapePerCTA[1] >= 256 ? 256 : retShapePerCTA[1];
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, m, n, colStride, CTASplitNum[0], CTASplitNum[1], false);

    // Allocate accumulator in tensor memory
    auto acc = triton::nvidia_gpu::TMEMAllocOp::create(rewriter, loc, accMemDescType, tokType, cvtAcc);

    // Create TensorMemoryScalesEncodingAttr for scales
    Attribute scaleEncoding = triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
        context, CTASplitNum[0], CTASplitNum[1]);

    // Add SMEM stage for scale loading (enables pipelining)
    auto lhsScale = addSmemStageToScaleLoad(dotOp.getAScale(), rewriter);
    auto rhsScale = addSmemStageToScaleLoad(dotOp.getBScale(), rewriter);

    // Allocate scales in tensor memory
    auto scaleA = triton::nvidia_gpu::TMEMAllocOp::create(rewriter, loc, scaleAType, ...);
    auto scaleB = triton::nvidia_gpu::TMEMAllocOp::create(rewriter, loc, scaleBType, ...);

    // Create TCGen5MMAScaledOp (Blackwell scaled MMA)
    auto mmaOp = triton::nvidia_gpu::TCGen5MMAScaledOp::create(
        rewriter, loc, tokType, a, b, acc.getResult(), acc.getToken(),
        scaleA.getResult(), scaleB.getResult(),
        dotOp.getAElemType(), dotOp.getBElemType(), useD, pred);

    // Load result from tensor memory
    auto ld = triton::nvidia_gpu::TMEMLoadOp::create(rewriter, loc, newAccType, tokType, acc, mmaOp.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
  }
};
```

### Pass Entry Point (Lines 988-1023)
```cpp
class TritonGPUAccelerateMatmulPass : public impl::TritonGPUAccelerateMatmulBase<...> {
  void runOnOperation() override {
    auto computeCapability = getNVIDIAComputeCapability(m);

    // Transpose scaled dots that have scale only on LHS
    transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    constexpr int benefitMMAv5 = 10;  // Higher priority for MMAv5

    patterns.add<BlockedToMMA>(context, computeCapability, benefitDefault);
    patterns.add<ScaledBlockedToMMA>(context, computeCapability, benefitSM120);
    populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    patterns.add<BlockedToMMAv5, ScaledBlockedToMMAv5>(
        context, computeCapability, benefitMMAv5);

    applyPatternsGreedily(m, std::move(patterns));

    // Decompose mixed-mode dot ops not natively supported
    decomposeMixedModeDotOp(m, computeCapability);
  }
};
```

---

## Frame 10: OptimizeDotOperands Pass (C++)

**File:** [lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp](../lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp)

### SwizzleShmemConvert Pattern (Lines 25-84)
```cpp
// Transforms: dot(convert(trans(src))) → dot(convert(local_load(trans(alloc(src)))))
struct SwizzleShmemConvert : public OpRewritePattern<ConvertLayoutOp> {
  LogicalResult matchAndRewrite(ConvertLayoutOp op, PatternRewriter &rewriter) const {
    // Match pattern: convert(trans(src)) with dot_operand encoding
    auto dstDotOp = dyn_cast<DotOperandEncodingAttr>(op.getType().getEncoding());
    if (!dstDotOp) return failure();

    // Compute swizzled shared memory encoding with transpose
    auto newSharedEnc = CGAEncodingAttr::get(ctx, transposeLinearLayout(...));

    // Create: alloc → trans → local_load → convert
    auto alloc = LocalAllocOp::create(rewriter, loc, allocType, src);
    auto trans = TransOp::create(rewriter, loc, transType, alloc);
    auto load = LocalLoadOp::create(rewriter, loc, loadType, trans);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, op.getType(), load);
  }
};
```

### FuseTransMMAV3Plus Pattern (Lines 92-133)
```cpp
// For MMAv3/v5: Rewrites alloc(trans()) → trans(alloc()) for better efficiency
struct FuseTransMMAV3Plus : public OpRewritePattern<LocalAllocOp> {
  LogicalResult matchAndRewrite(LocalAllocOp op, PatternRewriter &rewriter) const {
    // Only for WarpGroupDotOp or MMAv5OpInterface consumers
    for (auto user : op.getResult().getUsers()) {
      if (!isa<nvidia_gpu::WarpGroupDotOp, nvidia_gpu::MMAv5OpInterface>(user))
        return failure();
    }
    // Fuse transpose into allocation
    auto newEnc = inferTransOpEncoding(oldEnc, order);
    auto newAlloc = LocalAllocOp::create(rewriter, loc, newType, src);
    auto newTrans = TransOp::create(rewriter, loc, transType, newAlloc);
    rewriter.replaceOp(op, newTrans);
  }
};
```

### UseShmemForScales Pattern (Lines 185-318) - **FOR NVFP4 SCALES**
```cpp
// Optimizes scale loading for TCGen5MMAScaledOp
struct UseShmemForScales : public OpRewritePattern<nvidia_gpu::TCGen5MMAScaledOp> {
  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op, PatternRewriter &rewriter) const {
    // Detect sequence: local_load → reshape → transpose → reshape → tmem_alloc
    // Replace with direct local_alloc path for efficiency
    for (auto &operand : {op.getAScaleMutable(), op.getBScaleMutable()}) {
      if (auto tmemAlloc = operand.get().getDefiningOp<TMEMAllocOp>()) {
        // Trace back through reshapes/transposes to find local_load
        // Replace with optimized path if contiguity requirements met
      }
    }
  }
};
```

---

## Frame 11: RemoveLayoutConversions Pass (C++)

**File:** [lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp](../lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)

This is a multi-phase optimization pass with ~1700 lines of code.

### Phase 1: Layout Propagation (Lines 207-363)
```cpp
class LayoutPropagation {
  void initAnchorLayout() {
    // Identify layout anchors: ops that define the required layout
    // - Function arguments
    // - DescriptorOpInterface, LoadOp, StoreOp, DotOp, DotScaledOp
    // - nvidia_gpu::WarpGroupDotOp
    // - AtomicRMWOp, AtomicCASOp, GatherOp
    op->walk([&](Operation *op) {
      if (isa<DotOp, DotScaledOp>(op) || isa<LoadOp, StoreOp>(op) || ...) {
        anchors.insert(op);
      }
    });
  }

  void propagateLayout() {
    // BFS propagation through use-def chain
    while (!worklist.empty()) {
      Value v = worklist.pop_back();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        Attribute newEncoding = inferDstEncoding(user, v.getType().getEncoding());
        setEncoding(user->getResult(0), newEncoding);
        worklist.push_back(user->getResult(0));
      }
    }
  }

  void resolveConflicts() {
    // When multiple layouts are possible, use heuristics
    // - Prefer BlockedEncodingAttr for loads/stores
    // - Prefer MMA encoding for dot operations
  }
};
```

### Phase 2: Backward Rematerialization (Lines 1028-1265)
```cpp
class LayoutRematerialization {
  void backwardRematerialization() {
    // Walk ConvertLayoutOps in IR order
    module.walk([&](ConvertLayoutOp cvtOp) {
      // Get backward slice of operations
      auto slice = getRematerializableSlice(cvtOp.getSrc());

      // Cost analysis
      int convertCost = 32 * convertLayoutBytes;
      int rematCost = 0;
      for (auto op : slice) {
        if (isExpensiveMathOp(op))
          rematCost += 8 * numElements;  // 8x multiplier for expensive ops
        else
          rematCost += numElements;
      }

      // Rematerialize if cheaper
      if (rematCost < convertCost) {
        rematerializeSlice(slice, cvtOp);
      }
    });
  }
};
```

### Phase 3: Layout Hoisting (Lines 1393-1609)
```cpp
// Hoist converts before type extensions to reduce cost
void hoistConvertOnTopOfExtOrBroadcast() {
  module.walk([&](ConvertLayoutOp cvtOp) {
    if (auto extOp = cvtOp.getSrc().getDefiningOp<arith::ExtFOp>()) {
      // Move convert before extend (smaller data = cheaper convert)
      auto newCvt = ConvertLayoutOp::create(rewriter, loc, newType, extOp.getSrc());
      auto newExt = arith::ExtFOp::create(rewriter, loc, cvtOp.getType(), newCvt);
      cvtOp.replaceAllUsesWith(newExt);
    }
  });
}

// Hoist dot operand converts for better pipelining
void hoistConvertDotOperand() {
  module.walk([&](ConvertLayoutOp cvtOp) {
    if (auto dotOpEnc = dyn_cast<DotOperandEncodingAttr>(...)) {
      // Hoist convert out of loop if safe
    }
  });
}
```

---

## Frame 12: ScheduleLoops Pass (C++)

**File:** [lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp](../lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp)

### Safety Checks (Lines 36-48)
```cpp
static bool isSafeToPipeline(scf::ForOp forOp) {
  // Loop distance must be ≤ 1
  if (getLoopDistance(forOp) > 1) return false;
  // Not outer loop
  if (isOuterLoop(forOp)) return false;
  // No barriers, asserts, or prints
  if (containsBarrierOrAssert(forOp)) return false;
  return true;
}
```

### Key Ops Scheduling (Lines 152-247)
```cpp
void scheduleKeyOps(scf::ForOp forOp, CoarseSchedule &schedule) {
  // Identify ops with non-zero latency (loads, TMA, MMA)
  DenseMap<Operation *, int> latencies;
  forOp.walk([&](Operation *op) {
    if (isa<LoadOp, DescriptorLoadOp>(op))
      latencies[op] = 4;  // Memory latency
    if (isa<nvidia_gpu::WarpGroupDotOp, nvidia_gpu::TCGen5MMAOp>(op))
      latencies[op] = 2;  // Compute latency
  });

  // Compute longest path to yield using dominator info
  for (auto &[op, lat] : latencies) {
    int distance = computeDistanceToYield(op);
    int stage = maxDistance - distance;
    schedule.setStage(op, stage);
  }

  // Create clusters in reverse-stage order
  for (int stage = schedule.getNumStages() - 1; stage >= 0; stage--) {
    schedule.createCluster(stage);
  }
}
```

---

## Frame 13: Pipeline Pass (C++)

**File:** [lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp](../lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp)

### Pass Entry Point (Lines 171-224)
```cpp
struct PipelinePass : public impl::TritonGPUPipelineBase<PipelinePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Step 1: Lower loops by introducing async operations
    lowerLoops(moduleOp);

    // Step 2: Apply the pipeline expansion (prologue + main + epilogue)
    expandLoops(moduleOp);

    // Step 3: Cleanup pipeline attributes
    removePipeliningAttributes(moduleOp);

    // Step 4: Pipeline WGMMA operations
    pipelineWgmma(moduleOp, numStages);

    // Step 5: Schedule barrier waits
    mlir::triton::updateWaits(getOperation());

    // Step 6: Pipeline TMA stores
    for (scf::ForOp forOp : loops) {
      mlir::triton::pipelineTMAStores(forOp);
    }
  }
};
```

### WGMMA Pipelining (Lines 37-45)
```cpp
static void pipelineWgmma(ModuleOp moduleOp, unsigned numStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    if (getNumStagesOrDefault(forOp, numStages) >= 1)
      mlir::triton::asyncLaunchDots(forOp);  // Convert dots to async
  }
}
```

### Loop Expansion (Lines 64-169)
```cpp
static void expandLoops(ModuleOp moduleOp) {
  for (scf::ForOp forOp : loops) {
    CoarseSchedule schedule;
    schedule.deSerialize(forOp);

    // Create final schedule from coarse schedule
    auto finalSchedule = schedule.createFinalSchedule(forOp);

    // Configure pipelining options
    triton::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.predicateFn = wrapInMaskOp;

    // Heuristic: peel epilogue for MMAv5 loops with waits in last stage
    bool customEpiloguePeeling = hasMMAv5WaitsInLastStage(forOp, schedule);

    if (customEpiloguePeeling) {
      options.emitPredicateStageFn = [](RewriterBase &rewriter, ...) {
        return triton::gpu::PredicateStageOp::create(rewriter, loc, ...);
      };
    }

    // Apply pipeline expansion
    auto newForOp = triton::pipelineForLoop(rewriter, forOp, options);

    if (customEpiloguePeeling) {
      mlir::triton::peelLoopEpilogue(*newForOp, processPeeledEpilogueOp);
    }
  }
}
```

---

## Frame 14: TMALowering Pass (C++)

**File:** [lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp)

### Pass Entry Point (Lines 185-200)
```cpp
class TritonNvidiaGPUTMALoweringPass : public impl::TritonNvidiaGPUTMALoweringPassBase<...> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMALoadLowering, TMAGatherLowering, TMAStoreLowering,
                 TMAScatterLowering, TMAReduceLowering, TMACreateDescLowering>(context);
    applyPatternsGreedily(m, std::move(patterns));
  }
};
```

### TMALoadLowering Pattern (Lines 64-80)
```cpp
class TMALoadLowering : public OpRewritePattern<DescriptorLoadOp> {
  LogicalResult matchAndRewrite(DescriptorLoadOp op, PatternRewriter &rewriter) const {
    // Lower: DescriptorLoadOp → AsyncTMACopyGlobalToLocal with barriers

    // 1. Allocate shared memory for result
    auto alloc = gpu::LocalAllocOp::create(rewriter, loc, memDescType);

    // 2. Allocate and initialize barrier
    auto barrierAlloc = gpu::LocalAllocOp::create(rewriter, loc, barrierMemDescType);
    InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);

    // 3. Set expected bytes
    BarrierExpectOp::create(rewriter, loc, barrierAlloc, sizeInBytes, pred);

    // 4. Issue async TMA copy
    AsyncTMACopyGlobalToLocalOp::create(rewriter, loc, desc, indices, barrierAlloc, alloc, pred);

    // 5. Wait for completion
    WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase);
    InvalBarrierOp::create(rewriter, loc, barrierAlloc);

    // 6. Replace with local load
    replaceUsesWithLocalLoad(rewriter, op.getResult(), alloc);
  }
};
```

### TMAStoreLowering Pattern (Lines 118-130)
```cpp
struct TMAStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
  LogicalResult matchAndRewrite(DescriptorStoreOp op, PatternRewriter &rewriter) const {
    // Lower: DescriptorStoreOp → AsyncTMACopyLocalToGlobal

    // 1. Allocate shared memory and store source
    Value alloc = gpu::LocalAllocOp::create(rewriter, loc, memDescType, src);

    // 2. Fence before async store
    FenceAsyncSharedOp::create(rewriter, loc, false);

    // 3. Issue async TMA store
    AsyncTMACopyLocalToGlobalOp::create(rewriter, loc, desc, indices, alloc);

    // 4. Wait for completion
    TMAStoreWaitOp::create(rewriter, loc, 0);
  }
};
```

### TMACreateDescLowering Pattern (Lines 161-181)
```cpp
class TMACreateDescLowering : public OpRewritePattern<MakeTensorDescOp> {
  LogicalResult matchAndRewrite(MakeTensorDescOp op, PatternRewriter &rewriter) const {
    // Allocate global scratch memory for TMA descriptor
    auto alloc = GlobalScratchAllocOp::create(rewriter, loc, ptrType, TMA_SIZE_BYTES, TMA_ALIGN);

    // Initialize TMA descriptor
    createTMADesc(alloc, op, rewriter);

    // Fence for descriptor acquisition
    TensormapFenceproxyAcquireOp::create(rewriter, loc, alloc.getResult());

    // Reinterpret as tensor descriptor
    auto newDesc = ReinterpretTensorDescOp::create(rewriter, loc, op.getType(), alloc.getResult());
    rewriter.replaceOp(op, newDesc);
  }
};
```

---

## Frame 15: FenceInsertion Pass (C++)

**File:** [lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp)

### Pass Implementation (Lines 25-147)
```cpp
struct FenceInsertionPass : public impl::TritonGPUFenceInsertionBase<FenceInsertionPass> {
  void runOnOperation() override {
    // Only for SM90+ (Hopper and later)
    if (computeCapability < 90)
      return;

    ModuleOp mod = getOperation();
    mod.walk([&](DotOpInterface dotOp) {
      Value a = dotOp.getA();
      Value b = dotOp.getB();

      // Find register-to-shared copy operations for both operands
      SmallVector<Operation *> copyRegToSharedOpsA = findCopyRegToSharedOps(a);
      SmallVector<Operation *> copyRegToSharedOpsB = findCopyRegToSharedOps(b);

      if (copyRegToSharedOpsA.empty() && copyRegToSharedOpsB.empty())
        return WalkResult::advance();

      // Insert FenceAsyncSharedOp before the dot
      OpBuilder builder(dotOp);
      auto fence = FenceAsyncSharedOp::create(builder, dotOp.getLoc(), /*bCluster=*/false);

      // Hoist fence out of loops if possible
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (hasLoopDependency(copyOps, loopOp))
          break;
        loopOp.moveOutOfLoop(fence);
      }

      // Eliminate duplicate fences
      if (auto lastFence = dyn_cast_or_null<FenceAsyncSharedOp>(fence->getPrevNode())) {
        if (lastFence.getBCluster() == fence.getBCluster())
          fence.erase();
      }
    });
  }

private:
  // Trace back through use-def chain to find LocalAllocOp with src
  SmallVector<Operation *> findCopyRegToSharedOps(Value operand) {
    // Look for: LocalAllocOp with src → LocalStoreOp patterns
    if (auto localAlloc = dyn_cast<LocalAllocOp>(op)) {
      if (localAlloc.getSrc()) {
        result.insert(op);  // Found register-to-shared copy
      }
      // Check for LocalStoreOp users
      for (auto user : localAlloc.getResult().getUsers()) {
        if (isa<LocalStoreOp>(user)) {
          result.insert(user);
        }
      }
    }
  }
};
```

---

## Frame 16: MMALowering Pass (C++) - **CRITICAL FOR NVFP4 SCALES**

**File:** [lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp](../lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp)

### Pass Entry Point (Lines 203-218)
```cpp
class TritonNvidiaGPUMMALoweringPass : public impl::TritonNvidiaGPUMMALoweringPassBase<...> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SyncMMALowering, TCGen5MMAScaleSharedToTmemConversion,
                 MergeCommitIntoMMA>(context);

    applyPatternsGreedily(m, std::move(patterns));
  }
};
```

### SyncMMALowering Pattern (Lines 20-53)
```cpp
class SyncMMALowering : public OpInterfaceRewritePattern<MMAv5OpInterface> {
  LogicalResult matchAndRewrite(MMAv5OpInterface op, PatternRewriter &rewriter) const {
    // Skip if already async
    if (op.isAsync())
      return failure();

    // Create barrier for synchronization
    auto barrierAlloc = LocalAllocOp::create(rewriter, loc, barrierMemDescType, Value());
    InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);

    // Add completion barrier to MMA op
    op.addCompletionBarrier(barrierAlloc, arith::ConstantIntOp::create(rewriter, loc, 1, 1));
    op.setIsAsync(true);

    // Insert wait after MMA
    rewriter.setInsertionPointAfter(op);
    WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase, op.getPredicate());
    InvalBarrierOp::create(rewriter, loc, barrierAlloc);
  }
};
```

### TCGen5MMAScaleSharedToTmemConversion Pattern (Lines 55-102) - **KEY FOR NVFP4**
```cpp
// This pattern moves scale factors from shared memory to tensor memory (TMEM)
// Required for efficient MMAv5 scaled operations (NVFP4/MXFP4/MXFP8)
struct TCGen5MMAScaleSharedToTmemConversion : public OpRewritePattern<TCGen5MMAScaledOp> {
  // Create TMEM copy for scales. `rows` is M for LHS, N for RHS.
  bool lowerScaleToTmem(OpOperand &operand, PatternRewriter &rewriter, int rows) const {
    Location loc = operand.getOwner()->getLoc();
    auto oldType = cast<MemDescType>(operand.get().getType());
    auto numElems = product(oldType.getShape());

    // Distribute scales across MMA rows/columns
    SmallVector<int64_t> shape = {rows, numElems / rows};

    // Create TensorMemoryScalesEncodingAttr
    Attribute scaleEncoding = TensorMemoryScalesEncodingAttr::get(
        context, CTASplitNum[0], CTASplitNum[1]);

    Type scaleType = MemDescType::get(shape, elType, scaleEncoding, tensorMemorySpace,
                                       /*mutableMemory=*/true);

    // Allocate in tensor memory
    auto tmemAlloc = TMEMAllocOp::create(rewriter, loc, scaleType, Value());

    // Copy from shared to tensor memory
    TMEMCopyOp::create(rewriter, loc, operand.get(), tmemAlloc, /*barrier*/ Value());

    // Update the scale operand
    operand.set(tmemAlloc);
    return true;
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op, PatternRewriter &rewriter) const {
    int blockM = op.getBlockM();
    int blockN = op.getBlockN();
    bool anyChanged = false;

    // Lower A scale if in shared memory
    if (isa<SharedMemorySpaceAttr>(op.getAScale().getType().getMemorySpace())) {
      anyChanged = lowerScaleToTmem(op.getAScaleMutable(), rewriter, blockM);
    }

    // Lower B scale if in shared memory
    if (isa<SharedMemorySpaceAttr>(op.getBScale().getType().getMemorySpace())) {
      anyChanged = lowerScaleToTmem(op.getBScaleMutable(), rewriter, blockN);
    }

    return LogicalResult::success(anyChanged);
  }
};
```

### MergeCommitIntoMMA Pattern (Lines 174-199)
```cpp
class MergeCommitIntoMMA : public OpInterfaceRewritePattern<MMAv5OpInterface> {
  LogicalResult matchAndRewrite(MMAv5OpInterface op, PatternRewriter &rewriter) const {
    // Collect TCGen5CommitOp operations following the MMA
    auto [commitOps, predicates] = collectCommitOpsAfter(op);
    if (commitOps.empty())
      return failure();

    for (auto [commit, pred] : llvm::zip(commitOps, predicates)) {
      // Move barrier/predicate definitions before MMA if needed
      if (!moveDefiningOpsBefore(commit.getBarrier(), op))
        continue;

      // Add commit barrier to MMA operation
      op.addCompletionBarrier(commit.getBarrier(), pred);
      rewriter.eraseOp(commit);
    }
    return success();
  }
};
```

---

## Frame 17: make_llir() - Python Entry Point

**File:** [third_party/nvidia/backend/compiler.py#344-438](../third_party/nvidia/backend/compiler.py#344)

```python
def make_llir(self, src, metadata, options, capability):
    ptx_version = get_ptx_version_from_options(options, self.target.arch)
    mod = src

    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    # ===== PHASE 1: Prepare for LLVM lowering =====
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_allocate_warp_groups(pm)
    passes.convert.add_scf_to_cf(pm)  # Structured CF -> Control Flow
    passes.gluon.add_inliner(pm)

    # ===== PHASE 2: Memory allocation =====
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)  # Hopper tensor memory

    # ===== PHASE 3: Main LLVM conversion =====
    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)

    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)

    # ===== PHASE 4: NVIDIA-specific LLVM lowering =====
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
    nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)

    passes.convert.add_nvvm_to_llvm(pm)  # NVVM -> LLVM intrinsics
    passes.llvmir.add_di_scope(pm)       # Debug info

    pm.run(mod, 'make_llir')

    # ===== PHASE 5: Convert MLIR LLVM to native LLVM =====
    llvm.init_targets()
    context = llvm.context()
    llvm_mod = llvm.to_module(mod, context)

    proc = sm_arch_from_capability(capability)  # e.g., "sm_90a"
    features = get_features(options, self.target.arch)  # e.g., "+ptx86"
    triple = 'nvptx64-nvidia-cuda'

    llvm.attach_datalayout(llvm_mod, triple, proc, features)

    if options.extern_libs:
        llvm.link_extern_libs(llvm_mod, paths)

    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    return str(llvm_mod)
```

---

## Frame 18: ConvertTritonGPUToLLVM Pass (C++)

**File:** [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp)

### Pass Implementation (Lines 79-212)
```cpp
struct ConvertTritonGPUToLLVM : public triton::impl::ConvertTritonGPUToLLVMBase<...> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);

    // Shared memory and barrier analysis
    ModuleAllocation allocation(mod, getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Type converter
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions first
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(typeConverter, funcPatterns, targetInfo, ...);
    applyPartialConversion(mod, funcTarget, std::move(funcPatterns));

    // Initialize shared memory
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Register all conversion patterns
    RewritePatternSet patterns(context);

    // Layout conversion patterns
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo, patterns, ...);

    // TMA patterns (Hopper)
    mlir::triton::NVIDIA::populateTMAToLLVMPatterns(typeConverter, targetInfo, patterns, ...);

    // Dot operation patterns (MMA instructions)
    populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability, ...);

    // Elementwise operations
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, ...);

    // Load/Store operations
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, computeCapability, patterns, ...);

    // Reduction and scan
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns, targetInfo, ...);

    // TCGen5 MMA patterns (Blackwell scaled MMA)
    mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(typeConverter, patterns, ...);

    // FP4 conversion patterns
    mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(typeConverter, patterns, ...);

    // Standard MLIR patterns
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    TritonLLVMConversionTarget convTarget(*context);
    applyPartialConversion(mod, convTarget, std::move(patterns));

    // Lower CF ops
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, cfPatterns);
    applyPartialConversion(mod, cfTarget, std::move(cfPatterns));

    // Fold CTAId for single CTA
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        id.replaceAllUsesWith(LLVM::createConstantI32(id->getLoc(), b, 0));
      });
    }
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    // Create global_smem with dynamic shared allocation
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(b, loc, arrayTy, /*isConstant=*/false,
        LLVM::Linkage::External, "global_smem", /*value=*/Attribute(),
        /*alignment=*/16, NVVM::NVVMMemorySpace::Shared);
  }
};
```

---

## Frame 19: DotOpToLLVM Patterns (C++)

**File:** [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp)

### Pattern Registration (Lines 161-169)
```cpp
void mlir::triton::NVIDIA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, computeCapability, benefit);
  patterns.add<WarpGroupDotOpConversion>(typeConverter, benefit);
  patterns.add<WarpGroupDotWaitOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, computeCapability, benefit);
}
```

### ScaledDotOpConversion (Lines 28-45) - **FOR NVFP4**
```cpp
struct ScaledDotOpConversion : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  LogicalResult matchAndRewrite(triton::DotScaledOp op, Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Delegates to MMA-specific conversion for scaled dot
    return convertMMADotScaled(op, adaptor, getTypeConverter(), rewriter);
  }
};
```

### DotOpConversion (Lines 47-87)
```cpp
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  LogicalResult matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Value A = op.getA();
    Value D = op.getResult();

    NvidiaMmaEncodingAttr mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());

    if (mmaLayout) {
      if (mmaLayout.getVersionMajor() == 2) {
        // Ampere: mma.sync.aligned.m16n8k8
        bool isHopperF64 = computeCapability == 90 && A.getType().isF64();
        return convertMMA(op, adaptor, getTypeConverter(), rewriter,
                          mmaLayout.isTuring(), isHopperF64);
      }
      llvm::report_fatal_error("Unsupported MMA kind");
    }

    if (isa<BlockedEncodingAttr>(D.getType().getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error("Unsupported DotOp");
  }
};
```

### WarpGroupDotOpConversion (Lines 89-101)
```cpp
struct WarpGroupDotOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotOp> {
  LogicalResult matchAndRewrite(WarpGroupDotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Converts to Hopper WGMMA instruction
    return convertWGMMA(op, adaptor, getTypeConverter(), rewriter, getThreadId(rewriter, loc));
  }
};
```

---

## Frame 20: PTX Generation (make_ptx)

**File:** [third_party/nvidia/backend/compiler.py#440-464](../third_party/nvidia/backend/compiler.py#440)

```python
def make_ptx(self, src, metadata, opt, capability):
    ptx_version = get_ptx_version_from_options(opt, self.target.arch)

    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)  # "sm_90a"
    features = get_features(opt, self.target.arch)  # "+ptx86"
    flags = ["nvptx-mad-wide-opt"]

    # Translate LLVM IR to PTX assembly
    ret = llvm.translate_to_asm(src, triple, proc, features, flags,
                                 opt.enable_fp_fusion, False)

    # Extract kernel name
    names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
    metadata["name"] = names[0]

    # Update PTX version header
    ptx_version = f'{ptx_version//10}.{ptx_version%10}'
    ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret)
    ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret)

    return ret
```

---

## Frame 21: CUBIN Generation (make_cubin)

**File:** [third_party/nvidia/backend/compiler.py#466-530](../third_party/nvidia/backend/compiler.py#466)

```python
def make_cubin(self, src, metadata, opt, capability):
    ptxas = get_ptxas(self.target.arch).path

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc:
        fsrc.write(src)
        fsrc.flush()
        fbin = fsrc.name + '.o'

        arch = sm_arch_from_capability(capability)  # "sm_90a"

        ptxas_cmd = [
            ptxas,
            '-lineinfo',
            '-v',
            f'--gpu-name={arch}',
            '-o', fbin,
            fsrc.name,
        ]

        if not opt.enable_fp_fusion:
            ptxas_cmd.append('--fmad=false')

        subprocess.run(ptxas_cmd, check=True, capture_output=True)

        with open(fbin, 'rb') as f:
            cubin = f.read()

        return cubin
```

---

## Complete Data Flow for NVFP4 Scaled Matmul

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     BLOCK-SCALED NVFP4 MATMUL COMPILATION DATA FLOW                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  PYTHON                                                                                   │
│  ──────                                                                                   │
│  tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)                    │
│                                                                                           │
│  ╔════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                                  TTIR                                               ║  │
│  ╠════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║  %acc = tt.dot_scaled %a, %scale_a, %b, %scale_b, %acc_prev                         ║  │
│  ║           {lhs_format = e4m3, rhs_format = e2m1}                                    ║  │
│  ╚════════════════════════════════════════════════════════════════════════════════════╝  │
│                                         │                                                 │
│            AccelerateMatmul: ScaledBlockedToMMAv5 (AccelerateMatmul.cpp:756-895)         │
│                                         │                                                 │
│  ╔════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                                  TTGIR                                              ║  │
│  ╠════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║  // A, B in shared memory with NVMMASharedEncodingAttr                              ║  │
│  ║  %a_smem = ttg.local_alloc %a : tensor<128x64xf8, #shared>                          ║  │
│  ║  %b_smem = ttg.local_alloc %b : tensor<64x256xf4, #shared>                          ║  │
│  ║                                                                                     ║  │
│  ║  // Accumulator in tensor memory with TensorMemoryEncodingAttr                      ║  │
│  ║  %acc_tmem = ttng.tmem_alloc %acc : !ttg.memdesc<128x256xf32, #tmem>               ║  │
│  ║                                                                                     ║  │
│  ║  // Scales in tensor memory with TensorMemoryScalesEncodingAttr                     ║  │
│  ║  %scale_a_tmem = ttng.tmem_alloc : !ttg.memdesc<128x4xf8, #tmem_scales>            ║  │
│  ║  %scale_b_tmem = ttng.tmem_alloc : !ttg.memdesc<256x4xf8, #tmem_scales>            ║  │
│  ║                                                                                     ║  │
│  ║  // TCGen5 MMA Scaled Op (Blackwell)                                                ║  │
│  ║  %tok = ttng.tc_gen5_mma_scaled %a_smem, %b_smem, %acc_tmem,                        ║  │
│  ║                                  %scale_a_tmem, %scale_b_tmem                       ║  │
│  ║                                  {a_elem_type = e4m3, b_elem_type = e2m1}           ║  │
│  ║                                                                                     ║  │
│  ║  // Load result from tensor memory                                                  ║  │
│  ║  %result = ttng.tmem_load %acc_tmem, %tok                                           ║  │
│  ╚════════════════════════════════════════════════════════════════════════════════════╝  │
│                                         │                                                 │
│         MMALowering: TCGen5MMAScaleSharedToTmemConversion (MMALowering.cpp:55-102)       │
│         → Copies scales from shared memory to tensor memory                               │
│                                         │                                                 │
│         ConvertTritonGPUToLLVM: populateTCGen5MMAOpToLLVMPattern                         │
│                                         │                                                 │
│  ╔════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                                 LLVM IR                                             ║  │
│  ╠════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║  ; Inline PTX for TCGen5 MMA with scaling                                           ║  │
│  ║  %result = call <128 x float> asm sideeffect                                        ║  │
│  ║    "tcgen05.mma.cta_group::1.kind::f4f4acc32.block_scale.scale_vec::1X             ║  │
│  ║         {$0, ...}, $desc_a, $desc_b, {$acc, ...}, $scale_a, $scale_b;",             ║  │
│  ║    "=f,=f,...,l,l,f,f,..."(...)                                                     ║  │
│  ╚════════════════════════════════════════════════════════════════════════════════════╝  │
│                                         │                                                 │
│                              llvm.translate_to_asm()                                      │
│                                         │                                                 │
│  ╔════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                                   PTX                                               ║  │
│  ╠════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║  .version 8.6                                                                       ║  │
│  ║  .target sm_100a  // Blackwell                                                      ║  │
│  ║                                                                                     ║  │
│  ║  // TMA bulk copy for A, B                                                          ║  │
│  ║  cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx         ║  │
│  ║      [%r_smem], [%rd_desc, {%r_y, %r_x}], [%r_mbar];                               ║  │
│  ║                                                                                     ║  │
│  ║  // TCGen5 MMA with block scaling (NVFP4)                                           ║  │
│  ║  tcgen05.mma.cta_group::1.kind::f4f4acc32.block_scale.scale_vec::1X                ║  │
│  ║      {%f0, %f1, ...}, descA, descB, {%f_acc0, ...}, scaleA, scaleB;                ║  │
│  ║                                                                                     ║  │
│  ║  // Wait for MMA completion                                                         ║  │
│  ║  tcgen05.wait::ld.sync.aligned;                                                    ║  │
│  ╚════════════════════════════════════════════════════════════════════════════════════╝  │
│                                         │                                                 │
│                                       ptxas                                               │
│                                         │                                                 │
│  ╔════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                                  CUBIN                                              ║  │
│  ╠════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║  • Machine code for Blackwell GPU                                                   ║  │
│  ║  • TCGen5 MMA instructions with block scaling                                       ║  │
│  ║  • Tensor Memory (TMEM) allocation for accumulator and scales                       ║  │
│  ║  • TMA engine programming for efficient memory transfers                            ║  │
│  ╚════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Passes Summary (with C++ Source Locations)

| Pass | C++ File | Key Classes/Patterns | Purpose |
|------|----------|---------------------|---------|
| `add_convert_to_ttgpuir` | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` | `GenericOpPattern`, `ArithConstantPattern` | Add GPU layout encodings |
| `add_coalesce` | `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` | `CoalescePass` | Memory coalescing optimization |
| `add_accelerate_matmul` | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` | `BlockedToMMA`, `BlockedToMMAv5`, `ScaledBlockedToMMAv5` | MMA version selection (v2/v3/v5) |
| `add_optimize_dot_operands` | `lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp` | `SwizzleShmemConvert`, `FuseTransMMAV3Plus`, `UseShmemForScales` | Dot operand optimization |
| `add_remove_layout_conversions` | `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` | `LayoutPropagation`, `LayoutRematerialization` | Minimize layout conversions |
| `add_schedule_loops` | `lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp` | `scheduleKeyOps`, `scheduleDistanceOneDependencies` | Latency-based scheduling |
| `add_pipeline` | `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp` | `PipelinePass`, `expandLoops`, `pipelineWgmma` | Software pipelining |
| `add_tma_lowering` | `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp` | `TMALoadLowering`, `TMAStoreLowering`, `TMACreateDescLowering` | TMA descriptor operations |
| `add_fence_insertion` | `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp` | `FenceInsertionPass` | Memory ordering (SM90+) |
| `add_lower_mma` | `lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp` | `SyncMMALowering`, **`TCGen5MMAScaleSharedToTmemConversion`**, `MergeCommitIntoMMA` | MMA instruction lowering |
| `add_to_llvmir` | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` | `ConvertTritonGPUToLLVM` | Main LLVM conversion |
| `populateDotOpToLLVMPatterns` | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp` | `DotOpConversion`, `WarpGroupDotOpConversion`, `ScaledDotOpConversion` | Dot → MMA instructions |

---

## Files Referenced

### Python
- `python/triton/runtime/jit.py` - JIT decorator and JITFunction
- `python/triton/compiler/compiler.py` - ASTSource and compilation entry
- `python/triton/compiler/code_generator.py` - Python AST to TTIR
- `python/triton/language/semantic.py` - tl.* operation semantics
- `third_party/nvidia/backend/compiler.py` - NVIDIA backend passes

### C++ - Core Passes
- `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` - TTIR→TTGIR
- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` - MMA selection
- `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` - Memory coalescing
- `lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp` - Dot optimization
- `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` - Layout optimization
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp` - Loop scheduling
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp` - Pipelining

### C++ - NVIDIA-Specific
- `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp` - TMA lowering
- `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp` - Fence insertion
- `lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp` - MMA lowering
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` - TTGIR→LLVM
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp` - Dot lowering
