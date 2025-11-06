# Frame-by-Frame Execution Trace: test_warp_specialize_basic_ir

This document provides a comprehensive, frame-by-frame execution trace of the `test_warp_specialize_basic_ir` test function, focusing on the `triton.compile()` execution path when compiling MLIR IR files. The trace follows the literate programming style, interleaving code snippets with detailed explanations.

---

## Frame 1: Test Setup and IR Definition

**File**: `[python/test/unit/language/test_warp_specialization.py:24](../../../python/test/unit/language/test_warp_specialization.py#L24)`

```python
@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_hopper_or_blackwell(), reason="Requires Hopper or Blackwell")
def test_warp_specialize_basic_ir(tmp_path: pathlib.Path):
    ir = """
    tt.func @kernel(%arg0: !tt.ptr<i32>) {
      %c42_i32 = arith.constant 42 : i32
      gpu.barrier
      ttg.warp_specialize(%arg0)
      default {
        tt.store %arg0, %c42_i32 : !tt.ptr<i32>
        gpu.barrier
        ttg.warp_yield
      }
      partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
        %c5555_i32 = arith.constant 5555 : i32
        %c1_i32 = arith.constant 1 : i32
        gpu.barrier
        %ptr = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
        tt.store %ptr, %c5555_i32 : !tt.ptr<i32>
        ttg.warp_return
      } : (!tt.ptr<i32>) -> ()
      tt.return
    }
    """
```

### What Happens

The test defines an MLIR IR string in the **Triton dialect** (specifically, the TritonGPU dialect extension). This IR represents a kernel that demonstrates warp specialization - a technique where different warps (groups of 32 threads on NVIDIA GPUs) execute different code paths.

### Understanding the IR Structure

**Function Declaration:**
- `tt.func @kernel(%arg0: !tt.ptr<i32>)`: Declares a Triton function named `@kernel` that takes one argument - a pointer to 32-bit integers (`!tt.ptr<i32>`)

**Warp Specialization Operation:**
- `ttg.warp_specialize(%arg0)`: The key operation! This creates a control flow structure where:
  - One or more warps execute the "default" region (fallback path)
  - Other warps can be assigned to specialized "partition" regions

**Default Region:**
```mlir
default {
  tt.store %arg0, %c42_i32 : !tt.ptr<i32>
  gpu.barrier
  ttg.warp_yield
}
```
- Executed by warps not assigned to any partition
- Stores the constant `42` to the pointer `%arg0`
- Uses `gpu.barrier` for synchronization
- Ends with `ttg.warp_yield` to yield control back

**Partition0 Region:**
```mlir
partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
  %c5555_i32 = arith.constant 5555 : i32
  %c1_i32 = arith.constant 1 : i32
  gpu.barrier
  %ptr = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
  tt.store %ptr, %c5555_i32 : !tt.ptr<i32>
  ttg.warp_return
}
```
- Executed by exactly 1 warp (specified by `num_warps(1)`)
- Takes the input pointer as `%arg1`
- Computes `%ptr = %arg1 + 1` (pointer arithmetic)
- Stores the constant `5555` to `%ptr` (second element of array)
- Ends with `ttg.warp_return`

### Key Observations

1. **Dialect Mix**: The IR uses multiple MLIR dialects:
   - `tt.*`: Triton dialect operations
   - `ttg.*`: TritonGPU dialect operations
   - `arith.*`: Standard arithmetic dialect
   - `gpu.*`: GPU dialect

2. **Warp Specialization Pattern**: This pattern allows heterogeneous workload distribution where specialized warps can perform different tasks (e.g., async memory operations, producer-consumer patterns)

3. **Expected Behavior**: After kernel execution:
   - `input[0]` should be `42` (written by default region)
   - `input[1]` should be `5555` (written by partition0)

---

## Frame 2: File I/O - Writing IR to Disk

**File**: `[python/test/unit/language/test_warp_specialization.py:47](../../../python/test/unit/language/test_warp_specialization.py#L47)`

```python
temp_file = tmp_path / "test_warp_specialize_basic_ir.ttir"
temp_file.write_text(ir)
```

### What Happens

The test writes the MLIR IR string to a temporary file with the `.ttir` extension. The `.ttir` extension is significant:

- **`.ttir`**: Triton IR (highest-level Triton representation after Python AST)
- Other extensions in the compilation pipeline:
  - `.ttgir`: TritonGPU IR (after GPU-specific lowering)
  - `.llir`: LLVM IR (in MLIR format)
  - `.ptx`: PTX assembly (NVIDIA parallel thread execution)
  - `.cubin`: Compiled CUDA binary

The file extension tells the compiler:
1. What dialect to expect when parsing
2. Which compilation stage to start from
3. Which passes to skip (since we're providing pre-lowered IR)

**File Path Example**: `/tmp/pytest-xyz/test_warp_specialize_basic_ir0/test_warp_specialize_basic_ir.ttir`

---

## Frame 3: triton.compile() Entry Point

**File**: `[python/test/unit/language/test_warp_specialization.py:49](../../../python/test/unit/language/test_warp_specialization.py#L49)`

```python
kernel = triton.compile(str(temp_file))
```

**Calls**: `triton.compiler.compiler.compile()`

**Location**: `[python/triton/compiler/compiler.py:226](../../../python/triton/compiler/compiler.py#L226)`

### Function Signature and Initial Setup

```python
def compile(src, target=None, options=None, _env_vars=None):
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
```

### What Happens

**Step 1: Determine Compilation Mode**
- `src` is a string (file path), not an `ASTSource` object
- Sets `ir_source = True` to indicate we're compiling from IR file, not Python AST

**Step 2: Get GPU Target**
- Queries the active driver for current GPU target
- Example: `GPUTarget(backend='cuda', arch=90, warp_size=32)` for Hopper (H100)
- Example: `GPUTarget(backend='cuda', arch=100, warp_size=32)` for Blackwell (B100)

**Step 3: Create Backend**
```python
backend = make_backend(target)  # Returns CUDABackend instance
```
- Scans registered backends to find one supporting the target
- For CUDA GPUs, instantiates `CUDABackend` from `[third_party/nvidia/backend/compiler.py](../../../third_party/nvidia/backend/compiler.py)`

---

## Frame 4: IRSource Creation and MLIR Parsing

**Location**: `[python/triton/compiler/compiler.py:237](../../../python/triton/compiler/compiler.py#L237)`

```python
if ir_source:
    assert isinstance(src, str), "source must be either AST or a filepath"
    context = ir.context()
    src = IRSource(src, context, backend)
```

### Creating MLIR Context

```python
context = ir.context()
```

This creates a new MLIR context via the Python binding in `[python/src/ir.cc:326](../../../python/src/ir.cc#L326)`:

```cpp
py::class_<MLIRContext>(m, "context", py::module_local())
    .def(py::init<>([]() {
        return std::make_unique<MLIRContext>(MLIRContext::Threading::DISABLED);
    }))
```

The MLIR context is the top-level container for:
- All IR objects (types, attributes, operations)
- Loaded dialects
- Configuration settings (threading, debug info)

### IRSource Constructor

**Location**: `[python/triton/compiler/compiler.py:87](../../../python/triton/compiler/compiler.py#L87)`

```python
class IRSource:
    def __init__(self, path, context, backend):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]  # "ttir"
        self.language = Language.TRITON
        self.src = path.read_text()  # Read the IR string from file

        # Load dialects into context
        ir.load_dialects(context)
        backend.load_dialects(context)

        # Parse based on extension
        if self.ext == "ptx":
            # PTX uses regex parsing (no MLIR parser available)
            match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
            self.name = match.group(1)
            signature = match.group(2)
            types = re.findall(arg_type_pattern[self.ext], signature)
            self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}
        else:
            # MLIR-based formats (ttir, ttgir, llir)
            self.module = ir.parse_mlir_module(self.path, context)
            fn_name = self.module.get_entry_func_name()
            self.name = "@" + fn_name
            funcOp = self.module.get_function(fn_name)
            func_ty = self.module.get_function_signature(funcOp)
            self.signature = {k: ty for k, ty in enumerate(func_ty)}
```

### What Happens Step-by-Step

**Step 1: Extract File Information**
- `self.ext = "ttir"` (from file extension)
- `self.src = <IR string>` (file contents)
- `self.language = Language.TRITON`

**Step 2: Load Dialects**

From `[python/src/ir.cc:340](../../../python/src/ir.cc#L340)`:

```cpp
m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<TritonDialect,
                    ::mlir::triton::gpu::TritonGPUDialect,
                    ::mlir::triton::instrument::TritonInstrumentDialect,
                    ::mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    math::MathDialect,
                    arith::ArithDialect,
                    scf::SCFDialect,
                    ::mlir::gpu::GPUDialect,
                    cf::ControlFlowDialect,
                    LLVM::LLVMDialect,
                    mlir::ub::UBDialect,
                    mlir::triton::gluon::GluonDialect>();
    mlir::LLVM::registerInlinerInterface(registry);
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
});
```

This registers all MLIR dialects needed:
- **TritonDialect**: Core Triton operations (`tt.*`)
- **TritonGPUDialect**: GPU-specific operations (`ttg.*`)
- **TritonNvidiaGPUDialect**: NVIDIA-specific operations (`ttng.*`)
- **Standard dialects**: `arith`, `math`, `scf`, `gpu`, `cf`
- **LLVM dialect**: For eventual lowering to LLVM IR

**Step 3: Backend Dialect Loading**

From `[third_party/nvidia/backend/compiler.py:223](../../../third_party/nvidia/backend/compiler.py#L223)`:

```python
def load_dialects(self, ctx):
    nvidia.load_dialects(ctx)
    if CUDABackend.instrumentation:
        CUDABackend.instrumentation.load_dialects(ctx)
```

This loads NVIDIA-specific dialects and instrumentation dialects if enabled.

### Python → C++ Boundary: Parsing MLIR Module

**Python call** (`compiler.py:107`):
```python
self.module = ir.parse_mlir_module(self.path, context)
```

**C++ implementation** (`[python/src/ir.cc:731](../../../python/src/ir.cc#L731)`):

```cpp
m.def("parse_mlir_module",
    [](const std::string &inputFilename, MLIRContext &context) {
        // parse module
        OwningOpRef<ModuleOp> module =
            parseSourceFile<ModuleOp>(inputFilename, &context);
        if (!module)
            throw std::runtime_error("Parse MLIR file failed.");
        return module->clone();
    },
    ret::take_ownership);
```

This calls MLIR's `parseSourceFile` function, which:
1. Opens the file and creates a source buffer
2. Invokes the MLIR parser on the buffer
3. Constructs an MLIR ModuleOp (the top-level container)
4. Validates the parsed IR structure
5. Returns ownership of the ModuleOp to Python

The parser recognizes:
- Operation names (e.g., `tt.func`, `ttg.warp_specialize`, `tt.store`)
- Types (e.g., `!tt.ptr<i32>`, `i32`)
- Attributes (e.g., `num_warps(1)`)
- Regions and blocks (the nested `{}` structures)

**Step 4: Extract Function Metadata**

From `[python/src/ir.cc:647](../../../python/src/ir.cc#L647)` (get_entry_func_name):

```cpp
.def("get_entry_func_name",
    [](ModuleOp &self) -> std::string {
        for (auto &op : self.getOps()) {
            if (auto func = dyn_cast<FuncOp>(op)) {
                if (triton::isKernel(func))
                    return func.getName().str();
            }
        }
        return "";
    })
```

Iterates through module operations to find the kernel function:
- Checks each operation to see if it's a `tt.func`
- Checks if it's marked as a kernel (has public visibility or kernel attribute)
- Returns the function name: `"kernel"`

From `[python/src/ir.cc:663](../../../python/src/ir.cc#L663)` (get_function):

```cpp
.def("get_function",
    [](ModuleOp &self, std::string &funcName) -> FuncOp {
        return self.lookupSymbol<FuncOp>(funcName);
    })
```

Retrieves the `FuncOp` object for the kernel function.

From `[python/src/ir.cc:672](../../../python/src/ir.cc#L672)` (get_function_signature):

```cpp
.def("get_function_signature",
    [](ModuleOp &self, FuncOp &func) -> std::vector<std::string> {
        std::vector<std::string> strVec;
        auto type = func.getFunctionType();
        unsigned numArgs = type.getNumInputs();
        for (unsigned i = 0; i != numArgs; ++i) {
            std::string tempType;
            llvm::raw_string_ostream os(tempType);
            auto ty = type.getInput(i);
            // Convert type to string representation
            os << convert_type_repr(ty);
            strVec.push_back(os.str());
        }
        return strVec;
    })
```

Extracts the function signature:
- Input: `!tt.ptr<i32>`
- Converts to Python-friendly representation: `"*i32"`
- Returns: `{0: "*i32"}` (dictionary mapping argument index to type)

**Final IRSource State:**
```python
self.path = "/tmp/.../test_warp_specialize_basic_ir.ttir"
self.ext = "ttir"
self.language = Language.TRITON
self.src = "<IR string contents>"
self.module = <MLIR ModuleOp object>
self.name = "@kernel"
self.signature = {0: "*i32"}
```

---

## Frame 5: Parse Options and Cache Check

**Location**: `[python/triton/compiler/compiler.py:242](../../../python/triton/compiler/compiler.py#L242)`

### Parsing Source-Specific Options

```python
extra_options = src.parse_options()
options = backend.parse_options(dict(options or dict(), **extra_options))
```

From `[python/triton/compiler/compiler.py:121](../../../python/triton/compiler/compiler.py#L121)` (IRSource.parse_options):

```python
def parse_options(self):
    if self.ext == "ttgir":
        num_warps = self.module.get_int_attr("ttg.num-warps")
        assert num_warps is not None, "Unable to parse ttg.num-warps attribute"
        options = {'num_warps': num_warps}
        num_ctas = self.module.get_int_attr("ttg.num-ctas")
        if num_ctas is not None:
            options['num_ctas'] = num_ctas
        return options
    return dict()
```

For `.ttir` files, this returns an empty dict (no module-level attributes yet).

From `[third_party/nvidia/backend/compiler.py:171](../../../third_party/nvidia/backend/compiler.py#L171)` (CUDABackend.parse_options):

```python
def parse_options(self, opts) -> Any:
    args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
    args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys()
                 if k in opts if opts[k] is not None})
    capability = int(self._parse_arch(args["arch"]))

    # ... validation and default value logic ...

    return CUDAOptions(**args)
```

This creates a `CUDAOptions` object with defaults:
```python
CUDAOptions(
    num_warps=4,
    num_ctas=1,
    num_stages=3,
    warp_size=32,
    maxnreg=None,
    ptx_version=None,
    enable_fp_fusion=True,
    arch="sm90",  # or "sm100" for Blackwell
    # ... other fields ...
)
```

### Cache Key Generation

```python
env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
key = get_cache_key(src, backend, options, env_vars=env_vars)
hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
fn_cache_manager = get_cache_manager(hash)
```

The cache key includes:
- **Source hash**: SHA256 of IR file contents (`src.hash()`)
- **Backend hash**: PTX version and architecture (`backend.hash()`)
- **Options hash**: All compilation options (`options.hash()`)
- **Environment variables**: Version-sensitive env vars (CUDA_HOME, etc.)

Example cache key:
```
source:abcd1234...
backend:12.6-90
options:num_warps-4_num_ctas-1_...
triton_version:3.2.0
env:CUDA_HOME=/usr/local/cuda
```

### Cache Lookup

```python
file_name = src.name[:150]  # "@kernel"
metadata_filename = f"{file_name}.json"
metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
metadata_path = metadata_group.get(metadata_filename)
always_compile = knobs.compilation.always_compile

if not always_compile and metadata_path is not None:
    # cache hit!
    res = CompiledKernel(src, metadata_group, hash)
    return res
```

On **cache miss**, compilation proceeds. The cache would contain:
- `@kernel.json`: Metadata (options, kernel name, shared memory usage, etc.)
- `@kernel.ttir`: Original IR
- `@kernel.ttgir`: After GPU-specific lowering
- `@kernel.llir`: LLVM IR
- `@kernel.ptx`: PTX assembly
- `@kernel.cubin`: Compiled binary

---

## Frame 6: MLIR Context Setup and IR Initialization

**Location**: `[python/triton/compiler/compiler.py:278](../../../python/triton/compiler/compiler.py#L278)`

### Initialize Metadata

```python
metadata = {
    "hash": hash,
    "target": target,
    **options.__dict__,
    **env_vars,
}
metadata["triton_version"] = __version__
```

Creates a metadata dictionary that will be populated throughout compilation:
```python
{
    "hash": "abc123...",
    "target": GPUTarget(backend='cuda', arch=90, warp_size=32),
    "num_warps": 4,
    "num_ctas": 1,
    "num_stages": 3,
    "arch": "sm90",
    "triton_version": "3.2.0",
    # ... more fields added during compilation ...
}
```

### Determine Compilation Stages

```python
stages = dict()
backend.add_stages(stages, options, src.language)
first_stage = list(stages.keys()).index(src.ext)
# when the source is an IR file, don't apply the passes related to this stage
if ir_source:
    first_stage += 1
```

From `[third_party/nvidia/backend/compiler.py:533](../../../third_party/nvidia/backend/compiler.py#L533)` (CUDABackend.add_stages):

```python
def add_stages(self, stages, options, language):
    capability = self._parse_arch(options.arch)
    if language == Language.TRITON:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
    elif language == Language.GLUON:
        stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
    stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
    stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
```

The `stages` dictionary now contains:
```python
{
    "ttir": <function make_ttir>,
    "ttgir": <function make_ttgir>,
    "llir": <function make_llir>,
    "ptx": <function make_ptx>,
    "cubin": <function make_cubin>,
}
```

Since we're starting from `.ttir`:
- `first_stage = 0` (index of "ttir")
- `first_stage += 1` → `1` (skip ttir stage since we already have ttir)
- Compilation will start from `ttgir` stage

### Context and Dialect Loading (Already Done for IRSource)

```python
if not isinstance(src, IRSource):
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
```

This is skipped because we already created the context and loaded dialects in `IRSource.__init__()`.

### Get IR Module

```python
codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()
try:
    module = src.make_ir(target, options, codegen_fns, module_map, context)
except Exception as e:
    filter_traceback(e)
    raise
```

From `[python/triton/compiler/compiler.py:117](../../../python/triton/compiler/compiler.py#L117)` (IRSource.make_ir):

```python
def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
    self.module.context = context
    return self.module
```

Simply returns the already-parsed MLIR module. For `ASTSource`, this would invoke the Python AST → MLIR lowering.

### Save Initial IR to Cache

```python
if ir_source:
    ir_filename = f"{file_name}.{src.ext}"
    metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
```

Saves `@kernel.ttir` to the cache.

### Optional: Create Location Snapshot

```python
use_ir_loc = knobs.compilation.use_ir_loc
if ir_source and use_ir_loc:
    module.create_location_snapshot(src.path)
    print(f"Creating new locations for {src.path}")
```

If enabled, this creates MLIR location information that maps back to the original source file for better debugging.

---

## Frame 7: Compilation Stage 1 - TTGIR (TritonGPU IR)

**Location**: `[python/triton/compiler/compiler.py:323](../../../python/triton/compiler/compiler.py#L323)`

```python
for ext, compile_ir in list(stages.items())[first_stage:]:
    next_module = compile_ir(module, metadata)
    # ... cache and dump logic ...
    module = next_module
```

### Stage: ttir → ttgir

Since we skipped the ttir stage, the first stage executed is **ttgir**.

**Function**: `CUDABackend.make_ttgir()`

**Location**: `[third_party/nvidia/backend/compiler.py:245](../../../third_party/nvidia/backend/compiler.py#L245)`

```python
@staticmethod
def make_ttgir(mod, metadata, opt, capability):
    # Set maxnreg on all kernels, if it was provided.
    if opt.maxnreg is not None:
        mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

    pm = ir.pass_manager(mod.context)
    dump_enabled = pm.enable_debug()
    emuTF32 = (capability // 10 >= 8)
```

Creates a pass manager to orchestrate MLIR transformations.

### GPU-Specific Passes (Hopper/Blackwell - capability >= 90)

```python
# Convert Triton IR to TritonGPU IR
passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)

# Optimize TTGIR
passes.ttgpuir.add_coalesce(pm)
passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
nvidia.passes.ttnvgpuir.add_plan_cta(pm)
passes.ttgpuir.add_remove_layout_conversions(pm)
passes.ttgpuir.add_optimize_thread_locality(pm)
passes.ttgpuir.add_accelerate_matmul(pm)
passes.ttgpuir.add_remove_layout_conversions(pm)
passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
passes.ttir.add_loop_aware_cse(pm)
```

**Key Transformations:**

1. **add_convert_to_ttgpuir**:
   - Converts generic Triton operations to GPU-specific operations
   - Assigns data layouts (blocked, shared, etc.)
   - Adds `ttg.num-warps` and `ttg.num-ctas` attributes to the module

2. **add_coalesce**:
   - Optimizes memory access patterns for coalescing
   - Reorders operations to maximize memory bandwidth

3. **add_accelerate_matmul**:
   - Identifies matrix multiplication patterns
   - Maps to tensor cores when possible

### Blackwell-Specific Pipeline (capability >= 100)

```python
if capability // 10 >= 10:
    passes.ttgpuir.add_fuse_nested_loops(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_triton_licm(pm)
    passes.ttgpuir.add_optimize_accumulator_init(pm)
    passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
    nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
    passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
    passes.ttgpuir.add_schedule_loops(pm)

    # !!! WARP SPECIALIZATION PASS !!!
    passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)

    passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
    passes.ttgpuir.add_optimize_partition_warps(pm)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
    nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
```

**Critical Pass: add_warp_specialize**

This pass analyzes the `ttg.warp_specialize` operation and:
- Determines which warps execute which partitions
- May create additional partitions based on analysis
- Optimizes warp allocation based on workload
- For this simple test case, likely preserves the structure as-is

### Additional Optimization Passes

```python
passes.common.add_canonicalizer(pm)
passes.ttir.add_loop_aware_cse(pm)
passes.ttgpuir.add_prefetch(pm)
passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
passes.ttgpuir.add_coalesce_async_copy(pm)
nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)

if capability // 10 >= 9:
    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)

passes.ttgpuir.add_remove_layout_conversions(pm)
nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
passes.ttgpuir.add_reduce_data_duplication(pm)
passes.ttgpuir.add_reorder_instructions(pm)
passes.ttir.add_loop_aware_cse(pm)
passes.common.add_symbol_dce(pm)
nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
nvidia.passes.ttnvgpuir.add_lower_mma(pm)
passes.common.add_sccp(pm)
passes.common.add_cse(pm)
passes.common.add_canonicalizer(pm)
```

**Notable passes:**
- **add_tma_lowering**: Lowers TMA (Tensor Memory Accelerator) operations for Hopper+
- **add_fence_insertion**: Inserts memory fences for correctness
- **add_lower_mma**: Lowers matrix operations to hardware instructions
- **add_canonicalizer**: Normalizes IR to canonical form (multiple times)
- **add_cse**: Common subexpression elimination
- **add_sccp**: Sparse conditional constant propagation

### Running the Pass Pipeline

```python
pm.run(mod, 'make_ttgir')
metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
return mod
```

Executes all passes in sequence. Each pass:
1. Traverses the MLIR module
2. Applies pattern matching and rewriting
3. May add, remove, or modify operations
4. Verifies IR validity after transformation

**Output**: The module now contains TritonGPU IR with:
- Data layout annotations (e.g., `#blocked`, `#shared`)
- Hardware-specific operations (e.g., `ttng.warp_group_dot`)
- Warp specialization regions preserved and optimized
- Module attributes: `ttg.num-warps`, `ttg.shared` (shared memory usage)

**Saved to cache**: `@kernel.ttgir`

---

## Frame 8: Compilation Stage 2 - LLIR (LLVM IR in MLIR)

**Function**: `CUDABackend.make_llir()`

**Location**: `[third_party/nvidia/backend/compiler.py:337](../../../third_party/nvidia/backend/compiler.py#L337)`

```python
def make_llir(self, src, metadata, options, capability):
    ptx_version = get_ptx_version_from_options(options, self.target.arch)

    mod = src
    # TritonGPU -> LLVM-IR (MLIR)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
```

### Memory Allocation Passes

```python
passes.ttgpuir.add_combine_tensor_select_and_if(pm)
passes.ttgpuir.add_allocate_warp_groups(pm)
passes.convert.add_scf_to_cf(pm)
passes.gluon.add_inliner(pm)

# Allocate shared memory
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)

# Allocate tensor memory (Blackwell TMEM)
nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)

# Concurrency sanitizer (if enabled)
if knobs.compilation.instrumentation_mode == "consan":
    passes.ttgpuir.add_concurrency_sanitizer(pm)

# Allocate global scratch memory
passes.ttgpuir.add_allocate_global_scratch_memory(pm)

# Insert proxy fences
nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
```

**Key allocations:**
- **add_allocate_shared_memory_nv**: Allocates shared memory buffers, calculates total usage
- **add_allocate_tensor_memory**: Allocates Blackwell tensor memory (TMEM)
- **add_allocate_global_scratch_memory**: Allocates global memory for temporary data

### Lowering to LLVM Dialect

```python
# Convert TritonGPU operations to LLVM operations
nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
passes.common.add_canonicalizer(pm)
passes.common.add_cse(pm)

# Convert NVIDIA GPU operations to LLVM
nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)

# !!! WARP SPECIALIZATION TO LLVM !!!
nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)

passes.common.add_canonicalizer(pm)
passes.common.add_cse(pm)
passes.common.add_symbol_dce(pm)

# Convert NVVM dialect to LLVM dialect
passes.convert.add_nvvm_to_llvm(pm)
```

**Critical Pass: add_warp_specialize_to_llvm**

This pass converts the high-level `ttg.warp_specialize` structure into LLVM control flow:
1. Generates warp ID calculation code
2. Creates conditional branches based on warp ID
3. Routes warps to appropriate code regions
4. Inserts synchronization primitives

Pseudocode of generated LLVM IR:
```llvm
%warp_id = call i32 @llvm.nvvm.read.ptx.sreg.warpid()
%is_default = icmp ne i32 %warp_id, 0  ; warp 0 goes to partition0
br i1 %is_default, label %default_region, label %partition0_region

default_region:
  ; Store 42 to input[0]
  store i32 42, ptr %arg0
  call void @llvm.nvvm.barrier0()
  br label %end

partition0_region:
  ; Store 5555 to input[1]
  %ptr1 = getelementptr i32, ptr %arg0, i32 1
  store i32 5555, ptr %ptr1
  call void @llvm.nvvm.barrier0()
  br label %end

end:
  ret void
```

### Debug Information

```python
if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
    passes.llvmir.add_di_scope(pm)
```

Adds debug information for better error messages and debugging with cuda-gdb.

### Run Pass Pipeline

```python
pm.run(mod, 'make_llir')
```

### MLIR → LLVM Translation

```python
# LLVM-IR (MLIR) -> LLVM-IR (LLVM)
llvm.init_targets()
context = llvm.context()
llvm_mod = llvm.to_module(mod, context)
proc = sm_arch_from_capability(capability)  # "sm_90a" or "sm_100a"
features = get_features(options, self.target.arch)  # "+ptx86"
triple = 'nvptx64-nvidia-cuda'
nvidia.set_short_ptr()
llvm.attach_datalayout(llvm_mod, triple, proc, features)
nvidia.set_nvvm_reflect_ftz(llvm_mod)
```

**What happens:**
1. Converts MLIR LLVM dialect to native LLVM IR data structures
2. Sets target triple: `nvptx64-nvidia-cuda`
3. Sets CPU type: `sm_90a` (Hopper) or `sm_100a` (Blackwell)
4. Attaches data layout specification
5. Sets NVVM reflection metadata (flush-to-zero behavior)

### Link External Libraries

```python
if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
    paths = [path for (name, path) in options.extern_libs]
    llvm.link_extern_libs(llvm_mod, paths)
```

Links external libraries like `libdevice.10.bc` (math library for CUDA).

### LLVM Optimization

```python
llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
```

Runs LLVM's standard optimization pipeline at O3 level:
- Function inlining
- Loop optimizations (unrolling, vectorization)
- Scalar optimizations (GVN, SCCP, DCE)
- Code layout optimization

### Extract Metadata

```python
# warp-specialization mutates num_warps
total_num_warps = src.get_int_attr("ttg.total-num-warps")
if total_num_warps is not None:
    metadata["num_warps"] = total_num_warps
metadata["shared"] = src.get_int_attr("ttg.shared")
metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
```

Updates metadata with resource requirements:
- `num_warps`: May be updated if warp specialization changed configuration
- `shared`: Shared memory usage in bytes
- `tmem_size`: Tensor memory usage (Blackwell only)
- `global_scratch_size`: Global scratch buffer size

### Return LLVM IR String

```python
ret = str(llvm_mod)
del llvm_mod
del context
return ret
```

**Output**: LLVM IR as a text string (`.ll` format)

**Saved to cache**: `@kernel.llir`

---

## Frame 9: Compilation Stage 3 - PTX (Parallel Thread Execution)

**Function**: `CUDABackend.make_ptx()`

**Location**: `[third_party/nvidia/backend/compiler.py:431](../../../third_party/nvidia/backend/compiler.py#L431)`

```python
def make_ptx(self, src, metadata, opt, capability):
    ptx_version = get_ptx_version_from_options(opt, self.target.arch)

    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)
    features = get_features(opt, self.target.arch)
    flags = ["nvptx-mad-wide-opt"]
    ret = llvm.translate_to_asm(src, triple, proc, features, flags,
                                opt.enable_fp_fusion, False)
```

### LLVM IR → PTX Translation

`llvm.translate_to_asm()` invokes LLVM's NVPTX backend to generate PTX assembly:
1. Instruction selection: Maps LLVM IR to PTX instructions
2. Register allocation: Assigns virtual registers
3. Instruction scheduling: Reorders for ILP
4. Assembly emission: Generates PTX text

Example PTX output structure:
```ptx
.version 8.6
.target sm_90
.address_size 64

.visible .entry kernel(
    .param .u64 kernel_param_0
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<5>;

    // Get warp ID
    mov.u32 %r1, %warpid;
    setp.ne.s32 %p1, %r1, 0;

    // Branch based on warp
    @%p1 bra DEFAULT_REGION;

PARTITION0_REGION:
    // Load pointer
    ld.param.u64 %rd1, [kernel_param_0];
    // Compute offset
    add.s64 %rd2, %rd1, 4;
    // Store 5555
    mov.u32 %r2, 5555;
    st.global.u32 [%rd2], %r2;
    bar.sync 0;
    bra END;

DEFAULT_REGION:
    // Load pointer
    ld.param.u64 %rd1, [kernel_param_0];
    // Store 42
    mov.u32 %r2, 42;
    st.global.u32 [%rd1], %r2;
    bar.sync 0;

END:
    ret;
}
```

### Extract Kernel Name

```python
# Find kernel names (there should only be one)
names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
assert len(names) == 1
metadata["name"] = names[0]
```

Extracts the kernel entry point name from PTX. Due to name mangling, this might be something like `kernel_01d23abc`.

### Post-Process PTX

```python
# post-process
ptx_version = f'{ptx_version//10}.{ptx_version%10}'
ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)

if not knobs.compilation.dump_ir_extract_di_local_variables:
    # Remove the debug flag that prevents ptxas from optimizing the code
    ret = re.sub(r",\s*debug|debug,\s*", "", ret)
```

- Updates `.version` directive to target PTX version
- Updates `.target` directive to target architecture
- Removes debug flags if not needed (allows more aggressive optimization)

**Output**: PTX assembly as a text string

**Saved to cache**: `@kernel.ptx`

---

## Frame 10: Compilation Stage 4 - CUBIN (CUDA Binary)

**Function**: `CUDABackend.make_cubin()`

**Location**: `[third_party/nvidia/backend/compiler.py:457](../../../third_party/nvidia/backend/compiler.py#L457)`

```python
def make_cubin(self, src, metadata, opt, capability):
    ptxas = get_ptxas(self.target.arch).path
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
        tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
        fsrc.write(src)
        fsrc.flush()
        fbin = fsrc.name + '.o'
```

### Prepare ptxas Command

```python
debug_info = []
if knobs.compilation.disable_line_info:
    debug_info += ["-lineinfo", "-suppress-debug-info"]
elif knobs.nvidia.disable_ptxas_opt:
    debug_info += ["-g"]
else:
    debug_info += ["-lineinfo"]

fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
arch = sm_arch_from_capability(capability)
disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []
ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

ptxas_cmd = [
    ptxas, *debug_info, *fmad, '-v', *disable_opt, *ptx_extra_options,
    f'--gpu-name={arch}', fsrc.name, '-o', fbin
]
```

Example command:
```bash
/usr/local/cuda/bin/ptxas -lineinfo -v --gpu-name=sm_90a /tmp/xyz.ptx -o /tmp/xyz.ptx.o
```

**ptxas flags:**
- `-lineinfo`: Include line number information for profiling
- `-v`: Verbose output (register usage, shared memory)
- `--gpu-name=sm_90a`: Target Hopper architecture
- `--fmad=false`: Disable fused multiply-add (if requested)

### Run ptxas

```python
try:
    subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
    if knobs.nvidia.dump_ptxas_log:
        with open(flog.name) as log_file:
            print(log_file.read())
```

**ptxas output** (written to log file):
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'kernel_01d23abc' for 'sm_90'
ptxas info    : Function properties for kernel_01d23abc
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 0 bytes cmem[0]
```

This tells us:
- **Registers**: 8 32-bit registers per thread
- **Shared memory**: 0 bytes (for this simple kernel)
- **Spills**: 0 (all data fits in registers)

### Error Handling

```python
except subprocess.CalledProcessError as e:
    with open(flog.name) as log_file:
        log = log_file.read()

    if e.returncode == 255:
        error = 'Internal Triton PTX codegen error'
    elif e.returncode == 128 + signal.SIGSEGV:
        error = '`ptxas` raised SIGSEGV'
    else:
        error = f'`ptxas` failed with error code {e.returncode}'

    error = (f"{error}\n"
             f"`ptxas` stderr:\n{log}\n"
             f'Repro command: {" ".join(ptxas_cmd)}\n')

    raise PTXASError(error)
```

Provides detailed error information if ptxas fails.

### Read Binary

```python
with open(fbin, 'rb') as f:
    cubin = f.read()
if os.path.exists(fbin):
    os.remove(fbin)
return cubin
```

**Output**: CUBIN (CUDA Binary) as bytes. This is an ELF file containing:
- SASS (GPU machine code)
- Constant data sections
- Symbol table
- Debug information (if enabled)

**Saved to cache**: `@kernel.cubin`

---

## Frame 11: Write Cache and Return CompiledKernel

**Location**: `[python/triton/compiler/compiler.py:350](../../../python/triton/compiler/compiler.py#L350)`

```python
# write-back metadata
metadata_group[metadata_filename] = fn_cache_manager.put(
    json.dumps(metadata, default=vars), metadata_filename, binary=False)
fn_cache_manager.put_group(metadata_filename, metadata_group)

# notify any listener
if compilation_listener:
    compilation_listener(src=src, metadata=metadata, metadata_group=metadata_group,
                        times=timer.end(), cache_hit=False)

# return handle to compiled kernel
return CompiledKernel(src, metadata_group, hash)
```

### Final Metadata

The complete metadata now includes:
```python
{
    "hash": "abc123...",
    "target": GPUTarget(backend='cuda', arch=90, warp_size=32),
    "num_warps": 4,  # May be modified by warp specialization
    "num_ctas": 1,
    "num_stages": 3,
    "arch": "sm90",
    "triton_version": "3.2.0",
    "name": "kernel_01d23abc",  # Mangled kernel name
    "shared": 0,  # Shared memory bytes
    "tmem_size": None,  # Tensor memory (Blackwell only)
    "global_scratch_size": 0,
    "global_scratch_align": 0,
    "profile_scratch_size": 0,
    "profile_scratch_align": 1,
    "tensordesc_meta": [],
}
```

### CompiledKernel Construction

**Location**: `[python/triton/compiler/compiler.py:404](../../../python/triton/compiler/compiler.py#L404)`

```python
class CompiledKernel:
    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())

        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])

        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)

        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name

        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]  # CUBIN binary

        # binaries are lazily initialized
        self.module = None
        self.function = None
        self._run = None
```

**Key fields:**
- `self.metadata`: Named tuple with all metadata
- `self.asm`: Dictionary with all IR stages (`ttir`, `ttgir`, `llir`, `ptx`, `cubin`)
- `self.kernel`: The CUBIN binary
- `self._run`: Launcher function (initialized lazily)

**Packed metadata** (from `CUDABackend.pack_metadata`):
```python
def pack_metadata(self, metadata):
    return (
        metadata.num_warps,
        metadata.num_ctas,
        metadata.shared,
    )
```

Returns: `(4, 1, 0)` - tuple of critical launch parameters.

---

## Frame 12: Kernel Launch Preparation - __getitem__

**File**: `[python/test/unit/language/test_warp_specialization.py:51](../../../python/test/unit/language/test_warp_specialization.py#L51)`

```python
input = torch.empty(2, dtype=torch.int32, device='cuda')
kernel[(1, 1, 1)](input)
```

### Grid Specification via __getitem__

**Location**: `[python/triton/compiler/compiler.py:490](../../../python/triton/compiler/compiler.py#L490)`

```python
def __getitem__(self, grid):
    self._init_handles()

    def runner(*args, stream=None):
        if stream is None:
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        launch_metadata = self.launch_metadata(grid, stream, *args)
        self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata,
                 launch_metadata, knobs.runtime.launch_enter_hook,
                 knobs.runtime.launch_exit_hook, *args)

    return runner
```

**What happens:**
1. `kernel[(1, 1, 1)]` calls `__getitem__` with `grid = (1, 1, 1)`
2. Initializes GPU handles (loads binary to device)
3. Returns a `runner` function that captures the grid

---

## Frame 13: GPU Handle Initialization - _init_handles

**Location**: `[python/triton/compiler/compiler.py:436](../../../python/triton/compiler/compiler.py#L436)`

```python
def _init_handles(self):
    if self.module is not None:
        return

    device = driver.active.get_current_device()

    # create launcher
    self._run = driver.active.launcher_cls(self.src, self.metadata)

    # not enough shared memory to run the kernel
    max_shared = max_shared_mem(device)
    if self.metadata.shared > max_shared:
        raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))

    if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
        max_tmem_size = 512  # tmem size in number of columns
        if self.metadata.tmem_size > max_tmem_size:
            raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
```

### Step 1: Create Launcher

```python
self._run = driver.active.launcher_cls(self.src, self.metadata)
```

For CUDA, `driver.active.launcher_cls` is `CudaLauncher`.

**Location**: `[third_party/nvidia/backend/driver.py:674](../../../third_party/nvidia/backend/driver.py#L674)`

```python
class CudaLauncher(object):
    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        # Generate C++ launcher code
        src = make_launcher(constants, signature, tensordesc_meta)

        # Compile C++ code to shared library
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )

        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl
```

**What happens:**
1. Generates C++ code that wraps CUDA driver API calls
2. Compiles C++ code to a shared library using JIT compilation
3. Loads the library and extracts the `launch` function
4. The launcher handles argument packing and CUDA kernel launch

**Generated C++ code structure** (simplified):
```cpp
extern "C" void launch(
    CUstream stream, CUfunction function,
    void *arg0,  // kernel arguments
    ...
) {
    void *args[] = {&arg0, ...};

    // Launch kernel
    CUresult err = cuLaunchKernel(
        function,
        gridX, gridY, gridZ,
        num_warps * 32, 1, 1,  // block dimensions
        shared_mem,
        stream,
        args,
        nullptr
    );

    if (err != CUDA_SUCCESS) {
        // error handling
    }
}
```

### Step 2: Load Binary to GPU

```python
self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = \
    driver.active.utils.load_binary(
        self.name, self.kernel, self.metadata.shared, device)
```

This calls CUDA driver APIs:
```cpp
// Pseudocode of load_binary implementation
CUmodule module;
CUfunction function;

// Load CUBIN to device
cuModuleLoadData(&module, cubin_bytes);

// Get function handle
cuModuleGetFunction(&function, module, kernel_name);

// Query function attributes
cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
cuFuncGetAttribute(&n_max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);

// Set shared memory configuration
if (shared_mem > 48 * 1024) {
    cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem);
}
```

**Returns:**
- `module`: CUDA module handle
- `function`: CUDA function handle
- `n_regs`: Number of registers per thread (8 for our kernel)
- `n_spills`: Number of register spills (0 for our kernel)
- `n_max_threads`: Maximum threads per block

### Step 3: Validate Resources

```python
warp_size = driver.active.get_current_target().warp_size
if self.metadata.num_warps * warp_size > self.n_max_threads:
    raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
```

Ensures the kernel doesn't exceed device limits:
- Required: 4 warps × 32 threads = 128 threads
- Maximum: Usually 1024 threads (device-dependent)

---

## Frame 14: Kernel Launch Execution

**File**: `[python/test/unit/language/test_warp_specialization.py:52](../../../python/test/unit/language/test_warp_specialization.py#L52)`

```python
kernel[(1, 1, 1)](input)
```

This invokes the `runner` function created in Frame 12.

### Runner Execution

```python
def runner(*args, stream=None):
    if stream is None:
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
    launch_metadata = self.launch_metadata(grid, stream, *args)
    self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata,
             launch_metadata, knobs.runtime.launch_enter_hook,
             knobs.runtime.launch_exit_hook, *args)
```

**Arguments:**
- `args = (input,)` - the PyTorch tensor
- `stream = None` → uses default CUDA stream
- `grid = (1, 1, 1)` → launch 1 thread block

### CudaLauncher.__call__

**Location**: `[third_party/nvidia/backend/driver.py:700](../../../third_party/nvidia/backend/driver.py#L700)`

```python
def __call__(self, gridX, gridY, gridZ, stream, function, *args):

    def allocate_scratch(size, align, allocator):
        if size > 0:
            grid_size = gridX * gridY * gridZ
            total_size = size * grid_size
            return allocator(total_size, align, stream)
        return None

    # Allocate scratch buffers if needed
    global_scratch = allocate_scratch(
        self.global_scratch_size, self.global_scratch_align, CudaUtils.get_device_global_scratch_allocator())
    profile_scratch = allocate_scratch(
        self.profile_scratch_size, self.profile_scratch_align, lambda size, align, stream: torch.empty(size, device="cuda"))

    # Call the C++ launcher
    self.launch(
        gridX, gridY, gridZ,
        self.num_ctas,
        stream,
        function,
        global_scratch,
        profile_scratch,
        *args  # Kernel arguments
    )
```

### C++ Launcher Invocation

The C++ `launch` function (generated and compiled earlier) makes the CUDA driver API call:

```cpp
CUresult cuLaunchKernel(
    function,
    1, 1, 1,      // grid: 1 block
    128, 1, 1,    // block: 4 warps × 32 threads = 128 threads
    0,            // shared memory: 0 bytes
    stream,
    args,         // [pointer to input tensor data]
    nullptr
);
```

### GPU Execution

The kernel now executes on the GPU:

**Thread/Warp Layout:**
- **Block 0**: 128 threads = 4 warps
  - **Warp 0** (threads 0-31): Executes partition0 region
  - **Warps 1-3** (threads 32-127): Execute default region

**Warp 0 (partition0) execution:**
1. Load pointer: `%arg1 = input.data_ptr()`
2. Compute offset: `%ptr = %arg1 + 4` (bytes) = `input.data_ptr() + 1` (i32 element)
3. Store value: `*%ptr = 5555`
4. Synchronize: `__syncthreads()`

**Warps 1-3 (default) execution:**
1. Load pointer: `%arg0 = input.data_ptr()`
2. Store value: `*%arg0 = 42`
3. Synchronize: `__syncthreads()`

**Result:**
- All threads in warps 1-3 write 42 to `input[0]` (redundant writes, but last write wins)
- All threads in warp 0 write 5555 to `input[1]` (redundant writes, but last write wins)

The kernel returns, synchronizes with the CPU (implicitly via the default stream), and control returns to Python.

---

## Frame 15: Results Verification

**File**: `[python/test/unit/language/test_warp_specialization.py:53](../../../python/test/unit/language/test_warp_specialization.py#L53)`

```python
assert input[0] == 42
assert input[1] == 5555
```

### Memory State After Execution

The `input` tensor (on GPU) now contains:
```python
input = torch.tensor([42, 5555], dtype=torch.int32, device='cuda')
```

### Assertions

```python
assert input[0] == 42     # ✓ Written by default region (warps 1-3)
assert input[1] == 5555   # ✓ Written by partition0 region (warp 0)
```

Both assertions pass, confirming:
1. Warp specialization correctly routed different warps to different code paths
2. The default region executed and wrote 42
3. The partition0 region executed and wrote 5555
4. Memory synchronization worked correctly (no race conditions)

---

## Summary: Complete Compilation Pipeline

### High-Level Flow

```
Python IR String
    ↓
[Frame 1-2] Write to file (.ttir)
    ↓
[Frame 3] triton.compile(path)
    ↓
[Frame 4] IRSource: Parse MLIR, extract metadata
    ↓
[Frame 5] Check cache (miss → continue)
    ↓
[Frame 6] Initialize MLIR context, load dialects
    ↓
[Frame 7] TTIR → TTGIR (GPU lowering, warp specialization pass)
    ↓
[Frame 8] TTGIR → LLIR (allocate memory, lower to LLVM)
    ↓
[Frame 9] LLIR → PTX (LLVM backend generates assembly)
    ↓
[Frame 10] PTX → CUBIN (ptxas assembles to binary)
    ↓
[Frame 11] Write cache, return CompiledKernel
    ↓
[Frame 12-13] kernel[(1,1,1)] creates runner, loads binary to GPU
    ↓
[Frame 14] kernel(input) launches GPU execution
    ↓
[Frame 15] Verify results
```

### Key Transformations

1. **TTIR → TTGIR**:
   - Generic Triton ops → GPU-specific ops
   - Add data layout annotations
   - **Warp specialization pass** analyzes and optimizes partitioning
   - Memory coalescing, tensor core mapping

2. **TTGIR → LLIR**:
   - Allocate shared memory, tensor memory, scratch memory
   - **Warp specialization to LLVM**: Convert high-level regions to control flow
   - Lower GPU-specific ops to LLVM intrinsics

3. **LLIR → PTX**:
   - LLVM optimization (inlining, loop opt, etc.)
   - Instruction selection for NVIDIA GPUs
   - Register allocation

4. **PTX → CUBIN**:
   - ptxas assembler
   - Generate SASS (native GPU machine code)
   - Optimize for target architecture

### Critical Pass: Warp Specialization

The warp specialization transformation occurs in two main stages:

**Stage 1: TTGIR (Frame 7)**
- `passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)`
- Analyzes workload distribution
- May create/modify partitions for optimal performance
- Annotates warp requirements

**Stage 2: LLIR (Frame 8)**
- `nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)`
- Converts to LLVM control flow
- Generates warp ID checks
- Creates conditional branches
- Inserts synchronization

### Python ↔ C++ Boundaries

1. **MLIR Parsing**: Python `ir.parse_mlir_module()` → C++ `parseSourceFile()`
2. **Pass Execution**: Python `pm.run()` → C++ MLIR pass infrastructure
3. **LLVM Translation**: Python `llvm.to_module()` → C++ MLIR-to-LLVM conversion
4. **ptxas Invocation**: Python `subprocess.run()` → External process
5. **Kernel Launch**: Python calls C++ launcher (JIT-compiled) → CUDA driver API

### Resource Management

- **Cache**: Hash-based, stores all IR stages and metadata
- **Shared Memory**: Allocated during TTGIR → LLIR (0 bytes for this kernel)
- **Registers**: Determined by ptxas (8 registers per thread)
- **Tensor Memory**: Blackwell-only, not used in this kernel
- **Scratch Buffers**: Global and profile scratch, not used in this kernel

### Conclusion

The `test_warp_specialize_basic_ir` test demonstrates:
1. Compiling from pre-existing MLIR IR (bypassing Python AST)
2. The full Triton compilation pipeline (TTIR → TTGIR → LLIR → PTX → CUBIN)
3. Warp specialization: Different warps executing different code
4. End-to-end execution on GPU with correct results

This trace provides a detailed look at every step, from parsing MLIR text to executing machine code on the GPU, with special attention to the warp specialization feature that enables heterogeneous workload distribution within a single kernel.
