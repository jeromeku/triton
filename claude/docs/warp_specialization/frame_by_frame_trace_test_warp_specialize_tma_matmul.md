# Frame-by-Frame Execution Trace: test_warp_specialize_tma_matmul

This document provides a comprehensive, frame-by-frame execution trace of `test_warp_specialize_tma_matmul`, focusing on how the `@triton.jit` decorator transforms Python code into GPU kernels. This trace covers the entire compilation pipeline from AST parsing through MLIR generation to GPU execution.

---

## Overview

**Test Location**: [`experiments/test_warp_specialization.py:243-279`](../../../experiments/test_warp_specialization.py#L243)

**Key Focus Areas**:
1. `@triton.jit` decorator mechanics
2. Python AST parsing and transformation
3. `triton.language.standard` vs `triton.language.semantic` distinction
4. Code generation: Python → Triton IR (TTIR) using MLIR builders
5. Warp specialization in loops with TMA operations

---

## Frame 1: Decorator Application - Creating JITFunction

**File**: [`experiments/test_warp_specialization.py:191-229`](../../../experiments/test_warp_specialization.py#L191)

```python
@triton.jit
def matmul_tma_ws_kernel(
        a_ptr, b_ptr, c_ptr,
        a_stride0, a_stride1,
        b_stride0, b_stride1,
        c_stride0, c_stride1,
        M, N, K,
        num_stages: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        USE_FP8: tl.constexpr,
):
    # ... kernel body
```

### What Happens

**Location**: `[python/triton/runtime/jit.py:846](../../../python/triton/runtime/jit.py#L846)`

The `@triton.jit` decorator is actually a function that returns a `JITFunction` instance:

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
) -> Union[JITFunction[T], Callable[[T], JITFunction[T]]]:
    """Decorator for JIT-compiling a function using the Triton compiler."""

    def decorator(fn: T) -> JITFunction[T]:
        if isinstance(fn, JITFunction):
            return fn
        elif isinstance(fn, InterpretedFunction):
            return fn
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
```

### JITFunction Initialization

**Location**: `[python/triton/runtime/jit.py:712](../../../python/triton/runtime/jit.py#L712)`

```python
def __init__(self, fn, version=None, do_not_specialize=None,
             do_not_specialize_on_alignment=None, debug=None,
             noinline=None, repr=None, launch_metadata=None):
    do_not_specialize = do_not_specialize if do_not_specialize else []
    do_not_specialize_on_alignment = do_not_specialize_on_alignment if do_not_specialize_on_alignment else []

    super().__init__(fn)  # Calls JITCallable.__init__
    self.module = fn.__module__
    self.version = version
    self.do_not_specialize = do_not_specialize
    self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
    self._repr = repr
    self.launch_metadata = launch_metadata

    # Parse function signature and create KernelParam objects
    self.params = []
    for i, param in enumerate(self.signature.parameters.values()):
        dns = i in do_not_specialize or param.name in do_not_specialize
        dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
        self.params.append(KernelParam(i, param, dns, dns_oa))

    # Initialize per-device caches
    self.device_caches = defaultdict(self.create_binder)

    # ... more initialization
```

### JITCallable Base Class

**Location**: `[python/triton/runtime/jit.py:455](../../../python/triton/runtime/jit.py#L455)`

The parent `JITCallable` class handles dependency tracking and caching:

```python
class JITCallable:
    def __init__(self, fn):
        self.fn = fn
        self._src = textwrap.dedent(inspect.getsource(fn))
        self._fn_name = get_full_name(fn)
        self.arg_names = [p.name for p in inspect.signature(fn).parameters.values()]
        self.signature = inspect.signature(fn)

        # Compute cache key by finding dependencies
        finder = DependenciesFinder(
            name=self._fn_name,
            globals=fn.__globals__,
            nonlocals={},
            src=self._src
        )
        finder.visit(self.parse())
        self.cache_key = finder.ret
        self.used_global_vals = finder.used_global_vals
```

### DependenciesFinder: AST Hashing

**Location**: `[python/triton/runtime/jit.py:34](../../../python/triton/runtime/jit.py#L34)`

The `DependenciesFinder` is an AST visitor that computes a hash of the function and its dependencies:

```python
class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.
    """

    def __init__(self, name, globals, nonlocals, src) -> None:
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))
        self.globals = globals
        self.nonlocals = nonlocals
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}
        # ...
```

It walks the AST and:
- Tracks global variables referenced
- Recursively hashes called JIT functions
- Computes a final SHA256 hash for cache invalidation

### Key Observations

1. **Decorator Runs at Module Import Time**: When Python imports the module containing `matmul_tma_ws_kernel`, the decorator executes immediately, wrapping the function.

2. **No Compilation Yet**: At this point, no compilation has occurred. The `JITFunction` object just stores metadata about the kernel.

3. **Cache Key Computation**: The AST is parsed and hashed during initialization to enable cache lookups later.

4. **Parameter Analysis**: Each parameter is analyzed for:
   - Type annotations (e.g., `tl.constexpr`)
   - Specialization requirements
   - Alignment constraints

---

## Frame 2: Test Function Entry - Preparing to Launch Kernel

**File**: [`experiments/test_warp_specialization.py:243-279`](../../../experiments/test_warp_specialization.py#L243)

```python
def test_warp_specialize_tma_matmul(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                     num_stages, num_warps, use_fp8):
    # Skip test if too much shared memory
    if exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, use_fp8=use_fp8):
        pytest.skip("uses too much shared memory")

    dtype = torch.float8_e4m3fn if use_fp8 else torch.float16
    GROUP_SIZE_M = 8

    # Create input tensors
    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(dtype)
    B = torch.randn((N, K), dtype=torch.float16, device=device).to(dtype)
    C = torch.randn((M, N), dtype=torch.float16, device=device).to(dtype)

    # Set custom allocator for TMA descriptors
    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)

    # Define grid function
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    # LAUNCH KERNEL - This triggers compilation!
    kernel = matmul_tma_ws_kernel[grid](
        A, B, C,
        *A.stride(), *B.stride(), *C.stride(),
        M, N, K, num_stages,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        num_warps=num_warps,
        USE_FP8=use_fp8
    )
```

### What Happens

The key line is:
```python
kernel = matmul_tma_ws_kernel[grid](A, B, C, ...)
```

This syntax breaks down into two operations:
1. `matmul_tma_ws_kernel[grid]` - calls `JITFunction.__getitem__(grid)`
2. `(...)(A, B, C, ...)` - calls the returned runner function

---

## Frame 3: Grid Specification - __getitem__

**Location**: `[python/triton/runtime/jit.py:358](../../../python/triton/runtime/jit.py#L358)`

```python
def __getitem__(self, grid) -> T:
    """
    A JIT function is launched with: fn[grid](*args, **kwargs).
    Hence JITFunction.__getitem__ returns a callable proxy that
    memorizes the grid.
    """
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

### What Happens

- Returns a lambda that captures the grid specification
- Grid can be:
  - A tuple like `(1024, 1, 1)`
  - A callable that computes grid from kernel metadata

In our case, `grid` is:
```python
lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
```

This computes the number of thread blocks needed to cover the M×N output matrix.

---

## Frame 4: Kernel Launch - JITFunction.run()

**Location**: `[python/triton/runtime/jit.py:656](../../../python/triton/runtime/jit.py#L656)`

```python
def run(self, *args, grid, warmup, **kwargs):
    kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

    # Get current device and stream
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)

    # Execute pre run hooks
    for hook in self.pre_run_hooks:
        hook(*args, **kwargs)

    # Get device-specific cache
    kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]

    # Specialize arguments based on runtime values
    bound_args, specialization, options = binder(*args, **kwargs)

    # Compute cache key for this specialization
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(key, None)

    # Kernel is not cached; we have to compile.
    if kernel is None:
        options, signature, constexprs, attrs = self._pack_args(
            backend, kwargs, bound_args, specialization, options
        )

        kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
        if kernel is None:
            return None

    # Check that used global values have not changed
    # (omitted for brevity)

    if not warmup:
        # Canonicalize grid
        assert grid is not None
        if callable(grid):
            grid = grid(bound_args)
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1

        # Launch kernel
        launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
        kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
                   launch_metadata, knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                   *bound_args.values())
    return kernel
```

### What Happens

1. **Argument Binding and Specialization**: The `binder` function (created by `create_function_from_signature`) processes arguments:
   - Binds arguments to parameters
   - Specializes on shapes, strides, alignments
   - Determines which parameters are `constexpr`

2. **Cache Lookup**: Computes a cache key based on:
   - Argument types and specialization values
   - Compilation options (num_warps, num_stages, etc.)
   - Looks up kernel in per-device cache

3. **Compilation (Cache Miss)**: If kernel not cached, calls `_do_compile()`

4. **Execution**: If not warmup mode, launches the kernel on GPU

---

## Frame 5: Argument Specialization - The Binder

**Location**: `[python/triton/runtime/jit.py:380](../../../python/triton/runtime/jit.py#L380)`

The `create_function_from_signature` generates a Python function using `exec()`:

```python
def create_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions every time.
    """
    # Generate code for each parameter
    specialization = []
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
        else:
            is_const = 'True' if kp.is_const else 'False'
            specialize = 'False' if kp.do_not_specialize else 'True'
            align = 'False' if kp.do_not_specialize_on_alignment else 'True'
            ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"
            if kp.annotation_type:
                # Type override from annotation
                ret = f'("{kp.annotation_type}", {ret}[1])'
            specialization.append(ret)

    # Generate function code
    fn_code = f'''
def binder({", ".join(sig.parameters.keys())}, **kwargs):
    bound_args = {{{", ".join(f'"{k}": {k}' for k in sig.parameters.keys())}}}
    specialization = [{", ".join(specialization)}]
    options = dict(kwargs)
    return bound_args, specialization, options
'''

    # Execute and return the function
    exec(fn_code, {"backend": backend, "specialize_impl": native_specialize_impl})
    return locals()['binder']
```

### What Happens with Our Kernel Arguments

For `matmul_tma_ws_kernel`, the generated binder processes:

**Regular arguments** (a_ptr, b_ptr, c_ptr, strides, M, N, K):
- Calls `native_specialize_impl()` (C++ function via pybind11)
- Determines pointer alignment
- Determines whether value is divisible by powers of 2
- Creates type string like `"*fp16:16"` (pointer to fp16, 16-byte aligned)

**Constexpr arguments** (num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, etc.):
- Marked as `("constexpr", value)`
- These become compile-time constants in the generated code

**Keyword options** (num_warps, USE_FP8):
- Passed through in options dict
- Used during compilation

---

## Frame 6: Compilation Trigger - _do_compile()

**Location**: `[python/triton/runtime/jit.py:786](../../../python/triton/runtime/jit.py#L786)`

```python
def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
    kernel_cache, _, target, backend, _ = self.device_caches[device]

    # Call JIT cache hook if registered
    if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device,
                       constexprs, options, [attrs], warmup):
        return None

    # Create ASTSource from this JIT function
    src = self.ASTSource(self, signature, constexprs, attrs)

    # Check for async compilation mode
    async_mode = _async_compile.active_mode.get()
    if async_mode is not None:
        # Async path (omitted for brevity)
        pass
    else:
        # Synchronous compilation
        kernel = self.compile(src, target=target, options=options.__dict__)
        kernel_cache[key] = kernel
        self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device,
                        constexprs, options, [attrs], warmup)
    return kernel
```

### What Happens

1. **Create ASTSource**: Wraps the JITFunction with specialization info
2. **Call compile()**: This is the main compiler entry point
3. **Cache Result**: Store compiled kernel for future use

The `ASTSource` object contains:
- Reference to the JITFunction
- Signature with concrete types
- Constants (constexpr values)
- Attributes (alignment, divisibility hints)

---

## Frame 7: Compiler Entry Point - compile()

**Location**: `[python/triton/compiler/compiler.py:226](../../../python/triton/compiler/compiler.py#L226)`

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

    # For ASTSource, we need to create MLIR from Python AST
    if not ir_source:
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)

    # Create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)

    # Check cache
    metadata_filename = f"{src.name[:150]}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile

    if not always_compile and metadata_path is not None:
        # Cache hit!
        return CompiledKernel(src, metadata_group, hash)

    # Initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__

    # Get compilation stages from backend
    stages = dict()
    backend.add_stages(stages, options, src.language)
    first_stage = list(stages.keys()).index(src.ext)

    # Get codegen implementation
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()

    try:
        # THIS IS WHERE PYTHON → MLIR CONVERSION HAPPENS!
        module = src.make_ir(target, options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise

    # Cache the source
    ir_filename = f"{src.name[:150]}.source"
    metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    # Run compilation stages (ttir → ttgir → llir → ptx → cubin)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{src.name[:150]}.{ext}"
        # Handle IR overrides if configured
        # ...
        metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)

    # Write metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(
        json.dumps(metadata, default=vars), metadata_filename, binary=False
    )
    fn_cache_manager.put_group(metadata_filename, metadata_group)

    # Return compiled kernel
    return CompiledKernel(src, metadata_group, hash)
```

### What Happens

1. **Target and Backend Setup**: Creates NVIDIA backend for CUDA target
2. **MLIR Context Creation**: Initializes MLIR context and loads dialects
3. **Cache Check**: Uses filesystem-based cache (usually `~/.triton/cache`)
4. **AST → MLIR Conversion**: Calls `src.make_ir()` - **This is the critical step!**
5. **Compilation Stages**: Runs backend stages to lower MLIR through dialects
6. **Caching**: Stores all intermediate representations

---

## Frame 8: AST to TTIR Conversion - make_ir()

**Location**: `[python/triton/compiler/compiler.py:78](../../../python/triton/compiler/compiler.py#L78)`

```python
class ASTSource:
    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(self.fn, self, context=context, options=options,
                           codegen_fns=codegen_fns, module_map=module_map)
```

**Location**: `[python/triton/compiler/code_generator.py:1596](../../../python/triton/compiler/code_generator.py#L1596)`

```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    # Build argument types list
    arg_types = [None] * len(fn.arg_names)

    for k, v in src.signature.items():
        idx = fn.arg_names.index(k)
        arg_types[idx] = str_to_ty(v, None)  # Convert string like "*fp16:16" to type object

    # Apply constexpr types
    def apply_constexpr_types(argument, indices, value):
        index = indices.pop()
        if len(indices) == 0:
            if isinstance(argument, list):
                argument[index] = constexpr(value).type
            else:
                argument.types[index] = constexpr(value).type
        else:
            apply_constexpr_types(argument[index], indices, value)

    for path, value in src.constants.items():
        apply_constexpr_types(arg_types, list(path)[::-1], value)

    # Create prototype
    prototype = ASTFunction([], arg_types, src.constants, src.attrs)
    file_name, begin_line = get_jit_fn_file_line(fn)

    # Get constant values for function naming
    leaves = filter(lambda v: len(v) == 1, src.constants)
    constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
    signature = src.signature
    proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, signature)

    # CREATE CODE GENERATOR - The heart of the compiler frontend!
    generator = CodeGenerator(
        context, prototype,
        gscope=fn.get_capture_scope(),  # Global variables
        function_name=fn.repr(proxy),    # Kernel name
        jit_fn=fn,
        is_kernel=True,
        file_name=file_name,
        begin_line=begin_line,
        options=options,
        codegen_fns=codegen_fns,
        module_map=module_map,
        module=module,
        is_gluon=fn.is_gluon()
    )

    # VISIT THE AST - This generates MLIR!
    generator.visit(fn.parse())

    module = generator.module
    module.context = context

    # Verify MLIR module
    if not module.verify():
        print(module)
        raise RuntimeError("error encountered during parsing")

    return module
```

### What Happens

1. **Type Resolution**: Converts signature strings to type objects
   - `"*fp16:16"` → `pointer_type(float16, address_space=1)`

2. **Constexpr Application**: Marks constexpr arguments with their types

3. **CodeGenerator Creation**: Initializes the main compiler class with:
   - MLIR context
   - MLIR builder (C++ object exposed via pybind11)
   - Semantic implementation (for generating MLIR ops)
   - Global scope (for accessing triton.language functions)

4. **AST Visitation**: Walks the Python AST and generates MLIR

---

## Frame 9: CodeGenerator Initialization

**Location**: `[python/triton/compiler/code_generator.py:295](../../../python/triton/compiler/code_generator.py#L295)`

```python
class CodeGenerator(ast.NodeVisitor):

    def __init__(self, context, prototype, gscope, function_name, jit_fn: JITFunction, *,
                 options, codegen_fns, module_map, is_gluon, module=None, is_kernel=False,
                 function_types: Optional[Dict] = None, noinline=False, caller_context=None,
                 file_name: Optional[str] = None, begin_line=0):
        self.context = context
        self.is_gluon = is_gluon

        if is_gluon:
            from triton.experimental.gluon.language._semantic import GluonSemantic
            self.builder = gluon_ir.GluonOpBuilder(context)
            self.semantic = GluonSemantic(self.builder)
        else:
            from triton.language.semantic import TritonSemantic
            self.builder = ir.builder(context)  # MLIR OpBuilder (C++)
            self.semantic = TritonSemantic(self.builder)  # Semantic implementation

        self.name_loc_as_prefix = None
        self.file_name = file_name
        self.begin_line = begin_line - 1
        self.builder.set_loc(file_name, begin_line, 0)

        # Create module and set insertion point
        if module is None:
            self.module = self.builder.create_module()
        else:
            self.module = module
        self.builder.set_insertion_point_to_end(self.module.get_body())

        # Function prototype info
        self.prototype = prototype
        self.gscope = gscope
        self.function_name = function_name
        self.jit_fn = jit_fn

        # Scope tracking
        self.lscope = {}  # Local variables
        self.local_defs = {}  # Local definitions in current region

        # Control flow tracking
        self.scf_stack = []
        self.current_while_target = None

        # Codegen functions and options
        self.codegen_fns = codegen_fns
        self.module_map = module_map
        self.options = options

        # ... more initialization
```

### Key Components

**self.builder**: `ir.builder` (C++ class exposed to Python)
- Creates MLIR operations
- Manages insertion point
- Handles location information for debugging

**self.semantic**: `TritonSemantic` (Python class)
- Provides high-level operation builders
- Implements type checking and promotion
- Wraps self.builder calls

**self.gscope**: Global scope dictionary
- Contains `triton.language` module functions
- User-defined globals
- Allows kernel to call `tl.load()`, `tl.dot()`, etc.

**self.lscope**: Local scope dictionary
- Maps variable names to Triton values
- Updated as we walk the AST

---

## Frame 10: AST Visitation Begins - visit_FunctionDef

**Location**: `[python/triton/compiler/code_generator.py:380](../../../python/triton/compiler/code_generator.py#L380)`

The CodeGenerator visits the parsed AST of `matmul_tma_ws_kernel`:

```python
def visit_FunctionDef(self, node: ast.FunctionDef):
    # Get argument names and types
    arg_names, arg_types = self._get_fn_signature(node)

    # Create MLIR function
    fn_name = mangle_fn(node.name, arg_types, self.prototype.constants, None)
    fn_ty = self.builder.get_function_ty(
        [ty.to_ir(self.builder) for ty in arg_types],
        self.builder.get_void_ty()  # Return type
    )

    fn = self.builder.get_or_insert_function(
        self.module, fn_name, fn_ty,
        "public", is_kernel=self.is_kernel
    )

    # Set insertion point to function entry
    entry_block = fn.add_entry_block()
    self.builder.set_insertion_point_to_start(entry_block)

    # Map arguments to local scope
    mlir_args = [entry_block.arg(i) for i in range(len(arg_types))]
    for arg_name, arg_ty, mlir_arg in zip(arg_names, arg_types, mlir_args):
        self.lscope[arg_name] = arg_ty(mlir_arg, arg_ty)

    # Visit function body
    for stmt in node.body:
        self.visit(stmt)
        if isinstance(stmt, ast.Return):
            break

    # Add return if not already present
    if not self.builder.insertion_point_has_return():
        self.builder.ret([])
```

### What Happens

1. **Function Signature**: Extracts argument names and types from AST + prototype

2. **Create MLIR Function**: Calls `builder.get_or_insert_function()`
   - **Python → C++ boundary**: This calls C++ MLIR builders
   - Creates `tt.func` operation in MLIR
   - Sets visibility to "public" for kernels

3. **Initialize Locals**: Maps function arguments to `self.lscope`
   - Each argument becomes a `tensor` or `pointer_type` object
   - These objects wrap MLIR `ir.value` handles

4. **Visit Body**: Recursively visits each statement in the function

---

## Frame 11: Kernel Body - First Statement

Let's trace the first meaningful statement in `matmul_tma_ws_kernel`:

```python
a_desc = tl.make_tensor_descriptor(
    a_ptr,
    shape=[M, K],
    strides=[a_stride0, a_stride1],
    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
)
```

### Step 11.1: visit_Assign

**Location**: `[python/triton/compiler/code_generator.py:800](../../../python/triton/compiler/code_generator.py#L800)`

```python
def visit_Assign(self, node: ast.Assign):
    if len(node.targets) != 1:
        raise UnsupportedLanguageConstruct(
            self.jit_fn.src, node,
            "Only single assignment is supported"
        )

    target = node.targets[0]

    # Visit right-hand side (the function call)
    value = self.visit(node.value)

    # Store in local scope
    if isinstance(target, ast.Name):
        self.lscope[target.id] = value
        self.local_defs[target.id] = value
    elif isinstance(target, ast.Tuple):
        # Tuple unpacking
        # ...
    else:
        raise UnsupportedLanguageConstruct(
            self.jit_fn.src, node,
            f"Unsupported assignment target: {type(target)}"
        )
```

### Step 11.2: visit_Call - tl.make_tensor_descriptor

**Location**: `/home/jeromeku/triton/compiler/code_generator.py:1150-1250`

```python
def visit_Call(self, node: ast.Call):
    # Resolve the function being called
    fn = self.visit(node.func)

    # Visit arguments
    args = [self.visit(arg) for arg in node.args]
    kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}

    # Check if this is a builtin function
    if is_builtin(fn):
        # Pass _semantic parameter for builtin functions
        kwargs['_semantic'] = self.semantic
        kwargs['_generator'] = self

    # Check if this is a JIT function
    if isinstance(fn, JITCallable):
        # Inline or call JIT function
        return self._call_jit_function(fn, args, kwargs, node)

    # Regular Python function call
    return fn(*args, **kwargs)
```

### Step 11.3: Executing tl.make_tensor_descriptor

**Location**: `[python/triton/language/semantic.py](../../../python/triton/language/semantic.py)` (method on TritonSemantic)

Actually, `make_tensor_descriptor` is defined in:
**Location**: `[python/triton/language/core.py](../../../python/triton/language/core.py)` (around line 2500+)

```python
@builtin
def make_tensor_descriptor(ptr, shape, strides, block_shape, *,
                          order=None, _semantic=None, _generator=None):
    """
    Create a TMA (Tensor Memory Accelerator) descriptor for efficient memory access.
    """
    # The _semantic parameter is passed by visit_Call

    # Convert arguments to tensors
    ptr = _semantic.to_tensor(ptr)

    # Shape and strides must be constexpr
    if not all(isinstance(s, constexpr) for s in shape):
        raise TypeError("shape must be constexpr")
    if not all(isinstance(s, constexpr) for s in strides):
        raise TypeError("strides must be constexpr")

    # Create TMA descriptor (calls C++ MLIR builder)
    desc_ty = _semantic.lang.tensor_descriptor_type(
        element_ty=ptr.type.element_ty,
        shape=[s.value for s in shape],
        block_shape=[b.value for b in block_shape]
    )

    # Generate MLIR operation
    desc_val = _semantic.builder.create_make_tma_desc(
        ptr.handle,  # MLIR value handle
        [s.value for s in shape],
        [s.value for s in strides],
        [b.value for b in block_shape],
        order=order if order else list(range(len(shape)))
    )

    return tensor(desc_val, desc_ty)
```

### Understanding triton.language.standard vs semantic

At this point, it's crucial to understand the architecture:

**triton.language.core** (and **standard**):
- User-facing API functions
- Decorated with `@builtin`
- Check for `_semantic` parameter
- Handle argument validation
- Call methods on `_semantic` to generate MLIR

**triton.language.semantic.TritonSemantic**:
- Backend implementation
- Contains methods that actually generate MLIR operations
- Uses `self.builder` (C++ MLIR OpBuilder)
- Implements type checking and promotion

**Flow**:
```
User code: tl.make_tensor_descriptor(...)
    ↓
CodeGenerator.visit_Call() adds _semantic=self.semantic
    ↓
triton.language.core.make_tensor_descriptor(..., _semantic=semantic)
    ↓
Calls semantic.builder.create_make_tma_desc()
    ↓
Python → C++ (pybind11)
    ↓
C++ MLIR OpBuilder creates ttng.make_tma_desc operation
    ↓
Returns MLIR value handle
    ↓
Wrapped in tensor() object
    ↓
Stored in lscope["a_desc"]
```

### Python → C++ → MLIR Boundary

The `_semantic.builder.create_make_tma_desc()` call crosses into C++:

**File**: `[python/src/ir.cc](../../../python/src/ir.cc)` (large file, ~2000+ lines)

The C++ side (via pybind11) exposes builder methods:

```cpp
// Simplified example of how builder methods are exposed
py::class_<OpBuilder>(m, "builder")
    .def("create_make_tma_desc", [](OpBuilder &self,
                                     Value ptr,
                                     ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> strides,
                                     ArrayRef<int64_t> block_shape,
                                     ArrayRef<int64_t> order) {
        // Create MLIR operation
        Location loc = self.getInsertionPoint()->getLoc();
        auto op = self.create<triton::nvidia_gpu::MakeTensorDescOp>(
            loc, ptr, shape, strides, block_shape, order
        );
        return op.getResult();
    });
```

The MLIR operation is defined in:
**File**: `[include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td](../../../include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td)` (TableGen format)

This generates the actual MLIR operation class in C++.

---

## Frame 12: More Kernel Statements - Program ID and Grid Math

Next statements:
```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m, pid_n = _compute_pid(pid, num_pid_n, num_pid_m, GROUP_SIZE_M)
```

### Step 12.1: tl.program_id

**Location**: `[python/triton/language/core.py](../../../python/triton/language/core.py)`

```python
@builtin
def program_id(axis: int, _semantic=None) -> tensor:
    """
    Returns the program ID for the specified axis.
    In CUDA, this corresponds to blockIdx.{x,y,z}.
    """
    return _semantic.program_id(axis)
```

**Location**: `[python/triton/language/semantic.py:38](../../../python/triton/language/semantic.py#L38)`

```python
def program_id(self, axis: int) -> TensorTy:
    if axis not in (0, 1, 2):
        raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
    return self.tensor(self.builder.create_get_program_id(axis), tl.int32)
```

**What happens**:
- Calls `builder.create_get_program_id(axis)`
- **Python → C++**: Creates `tt.get_program_id` MLIR operation
- Returns scalar `int32` value representing block index
- Stored in `lscope["pid"]`

### Step 12.2: tl.cdiv - Ceiling Division

**Location**: `[python/triton/language/standard.py](../../../python/triton/language/standard.py)`

Wait! `tl.cdiv` is in **standard**, not **core**:

```python
@triton.jit
def cdiv(x, y):
    """Ceiling division: cdiv(x, y) = ceil(x / y)"""
    return (x + y - 1) // y
```

**Key observation**: `tl.cdiv` is itself a `@triton.jit` function!

**What happens**:
1. When the AST visits the call to `tl.cdiv`, it recognizes it's a JITCallable
2. Instead of executing Python code, it **inlines** the JIT function
3. The body of `cdiv` is parsed and its MLIR is generated inline

**Location**: `[python/triton/compiler/code_generator.py:1400](../../../python/triton/compiler/code_generator.py#L1400)` (approximately)

```python
def _call_jit_function(self, fn, args, kwargs, node):
    # Check if we should inline
    should_inline = not fn.noinline and self._can_inline(fn, args)

    if should_inline:
        # INLINE THE FUNCTION
        # Parse the function's AST
        fn_ast = fn.parse()

        # Create new code generator for the function
        # Map arguments to parameters
        arg_map = {param_name: arg_val
                   for param_name, arg_val in zip(fn.arg_names, args)}

        # Save current scope
        old_lscope = self.lscope
        self.lscope = {**self.lscope, **arg_map}

        # Visit function body
        result = None
        for stmt in fn_ast.body[0].body:  # FunctionDef -> body
            if isinstance(stmt, ast.Return):
                result = self.visit(stmt.value)
                break
            else:
                self.visit(stmt)

        # Restore scope
        self.lscope = old_lscope

        return result
    else:
        # CALL AS SEPARATE FUNCTION (not inline)
        # Create function if not already in module
        # Generate call operation
        # ...
```

So for `tl.cdiv(M, BLOCK_SIZE_M)`:
1. Visits the body: `return (x + y - 1) // y`
2. Generates MLIR for: `(M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M`
3. Since `BLOCK_SIZE_M` is constexpr (compile-time constant), this becomes a constant in MLIR

### Step 12.3: _compute_pid - Calling Another JIT Function

```python
@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n
```

This is also inlined, generating MLIR for the arithmetic operations.

**Resulting MLIR** (conceptual):
```mlir
%pid = tt.get_program_id {axis = 0 : i32} : i32
%num_pid_m = arith.constant 64 : i32  // cdiv(8192, 128) computed at compile time
%num_pid_n = arith.constant 64 : i32
// ... inline expansion of _compute_pid ...
%pid_m = ...
%pid_n = ...
```

---

## Frame 13: The Critical Loop - tl.range with warp_specialize=True

Now we reach the heart of warp specialization:

```python
for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
    off_k = k * BLOCK_SIZE_K
    a = a_desc.load((off_am, off_k))
    b = b_desc.load((off_bn, off_k))
    accumulator = tl.dot(a, b.T, accumulator)
```

### Step 13.1: visit_For - Loop Translation

**Location**: `[python/triton/compiler/code_generator.py:900](../../../python/triton/compiler/code_generator.py#L900)` (approximately)

```python
def visit_For(self, node: ast.For):
    # Get loop iterator (the call to tl.range)
    iter_node = node.iter

    # Check if this is a call to tl.range
    if isinstance(iter_node, ast.Call):
        fn = self.visit(iter_node.func)

        if fn is tl.range or (hasattr(fn, '__name__') and fn.__name__ == 'range'):
            # This is tl.range - handle specially
            return self._visit_range_loop(node, iter_node, fn)

    # Fall back to general iteration
    return self._visit_generic_for(node)
```

### Step 13.2: Handling tl.range

**Location**: `[python/triton/language/core.py](../../../python/triton/language/core.py)` (around line 3000+)

```python
def range(start, end=None, step=None, *, warp_specialize=False, num_stages=None,
          _generator=None, _semantic=None):
    """
    Creates a range for iteration.

    When warp_specialize=True, the loop body may be partitioned across warps
    with specialized roles (e.g., data loading vs compute).
    """
    # Normalize arguments
    if end is None:
        start, end = 0, start
    if step is None:
        step = 1

    # This function doesn't actually execute - it's a marker for the code generator
    # The code generator intercepts calls to tl.range in for loops
    raise RuntimeError("tl.range should only be used in for loops during JIT compilation")
```

Actually, `tl.range` is special-cased in the code generator:

**Location**: `[python/triton/compiler/code_generator.py](../../../python/triton/compiler/code_generator.py)`

```python
def _visit_range_loop(self, node, iter_node, fn):
    # Extract tl.range arguments
    args = [self.visit(arg) for arg in iter_node.args]
    kwargs = {kw.arg: self.visit(kw.value) for kw in iter_node.keywords}

    # Parse start, end, step
    if len(args) == 1:
        start, end, step = self.semantic.to_tensor(0), args[0], self.semantic.to_tensor(1)
    elif len(args) == 2:
        start, end, step = args[0], args[1], self.semantic.to_tensor(1)
    elif len(args) == 3:
        start, end, step = args[0], args[1], args[2]
    else:
        raise ValueError("range() takes 1 to 3 positional arguments")

    # Check for warp specialization
    warp_specialize = kwargs.get('warp_specialize', False)
    num_stages = kwargs.get('num_stages', None)

    if isinstance(warp_specialize, constexpr):
        warp_specialize = warp_specialize.value

    if warp_specialize:
        # GENERATE WARP-SPECIALIZED LOOP
        return self._generate_warp_specialized_loop(
            node, start, end, step, num_stages
        )
    else:
        # GENERATE REGULAR LOOP
        return self._generate_scf_for_loop(node, start, end, step)
```

### Step 13.3: Generating Warp-Specialized Loop

```python
def _generate_warp_specialized_loop(self, node, start, end, step, num_stages):
    """
    Generate a loop with warp specialization.

    This creates a ttg.warp_specialize region where:
    - Some warps act as "producers" (load data asynchronously)
    - Other warps act as "consumers" (perform compute)
    - Synchronization happens via shared memory and barriers
    """

    # Get loop variable name
    loop_var_name = node.target.id

    # Create warp specialization region
    # This generates: ttg.warp_specialize { ... }

    # Extract loop body into regions
    # 1. Default region (consumer warps)
    # 2. Partition region (producer warps)

    # The actual implementation is complex and involves:
    # - Analyzing the loop body for async operations
    # - Splitting operations into producer/consumer
    # - Creating appropriate MLIR regions

    # Simplified conceptual MLIR output:
    """
    scf.for %k = %c0 to %k_tiles step %c1 {
      ttg.warp_specialize(%a_desc, %b_desc) {
        default {
          // Consumer warps: compute using already-loaded data
          %a = ttg.local_load %a_shared : tensor<128x64xf16>
          %b = ttg.local_load %b_shared : tensor<128x64xf16>
          %acc = ttng.warp_group_dot %a, %b, %acc
          ttg.warp_yield
        }
        partition0(%a_desc_arg, %b_desc_arg) num_warps(1) {
          // Producer warps: async load next iteration's data
          %off_k_next = arith.addi %off_k, %c64
          ttng.async_tma_copy_global_to_local %a_desc_arg, %a_shared_next, %off_k_next
          ttng.async_tma_copy_global_to_local %b_desc_arg, %b_shared_next, %off_k_next
          ttg.warp_return
        }
      }
    }
    """

    # The details involve:
    # 1. Hoisting tensor descriptor loads
    # 2. Creating shared memory allocations
    # 3. Inserting sync barriers
    # 4. Scheduling operations across warps

    # Generate loop with warp specialization...
    # (actual implementation is in C++ MLIR passes)
```

**Important**: Much of the warp specialization logic happens in MLIR passes, not in the Python code generator!

The Python code generator creates a representation that marks the loop for warp specialization, then MLIR passes analyze and transform it.

---

## Frame 14: MLIR Pass - Warp Specialization Transformation

After the Python AST is converted to TTIR (Triton IR), the MLIR passes perform the actual warp specialization transformation.

**File**: `[lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp](../../../lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp)` (or similar)

The warp specialization pass:

1. **Analyzes the loop**:
   - Identifies async memory operations (TMA loads)
   - Identifies compute operations (dot products)
   - Determines dependencies

2. **Partitions operations**:
   - Producer operations: TMA async copies
   - Consumer operations: Compute using loaded data
   - Synchronization: Barriers between phases

3. **Creates warp specialization structure**:
   - Default region: Consumer warps (most warps)
   - Partition regions: Producer warps (fewer warps, e.g., 1-2)

4. **Inserts synchronization**:
   - Barriers to ensure data is ready
   - Pipeline stages for overlapping compute and memory

**Resulting MLIR** (simplified):
```mlir
tt.func @matmul_tma_ws_kernel(...) {
  %a_desc = ttng.make_tma_desc ...
  %b_desc = ttng.make_tma_desc ...
  %c_desc = ttng.make_tma_desc ...

  %k_tiles = arith.constant 8 : i32
  %acc_init = arith.constant dense<0.0> : tensor<128x128xf32>

  %acc_final = scf.for %k = %c0 to %k_tiles step %c1 iter_args(%acc = %acc_init) {
    %off_k = arith.muli %k, %c64 : i32

    // Warp specialization begins here
    ttg.warp_specialize(%a_desc, %b_desc, %off_k)
    default {
      // Consumer warps (compute)
      %a_local = ttg.local_load %a_smem : tensor<128x64xf16>
      %b_local = ttg.local_load %b_smem : tensor<128x64xf16>
      %b_t = ttg.trans %b_local {order = [1, 0]}
      %acc_new = ttng.warp_group_dot %a_local, %b_t, %acc
      ttg.warp_yield %acc_new
    }
    partition0(%a_desc_p, %b_desc_p, %off_k_p) num_warps(1) {
      // Producer warps (async memory)
      ttng.async_tma_copy_global_to_local %a_desc_p, %a_smem_next, %off_k_p
      ttng.async_tma_copy_global_to_local %b_desc_p, %b_smem_next, %off_k_p
      ttg.warp_return
    }

    scf.yield %acc_new
  }

  // Store result
  %c_final = arith.truncf %acc_final : tensor<128x128xf32> to tensor<128x128xf16>
  ttng.tma_store_tiled %c_desc, %c_final, ...

  tt.return
}
```

---

## Frame 15: triton.language.standard vs semantic - Complete Explanation

Now we can fully explain the distinction:

### triton.language.core (and standard)

**Purpose**: User-facing API

**Characteristics**:
- Functions decorated with `@builtin`
- Accept `_semantic` and `_generator` parameters (added by code generator)
- Perform argument validation
- Delegate to `_semantic` for actual MLIR generation

**Example**:
```python
@builtin
def load(ptr, mask=None, other=None, cache_modifier="", eviction_policy="",
         is_volatile=False, _semantic=None):
    """Load data from memory."""
    if _semantic is None:
        raise ValueError("load() must be called from within a @triton.jit function")

    # Validate arguments
    ptr = _semantic.to_tensor(ptr)
    if mask is not None:
        mask = _semantic.to_tensor(mask)

    # Delegate to semantic
    return _semantic.load(ptr, mask, other, cache_modifier, eviction_policy, is_volatile)
```

**triton.language.standard**:
- Higher-level functions built on `core`
- Many are themselves `@triton.jit` functions
- Examples: `cdiv`, `swizzle2d`, `softmax`
- Get inlined during compilation

```python
@triton.jit
def cdiv(x, y):
    return (x + y - 1) // y

@triton.jit
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))
```

### triton.language.semantic

**Purpose**: Backend implementation for MLIR generation

**Characteristics**:
- Contains `TritonSemantic` class
- Methods that generate MLIR operations
- Uses `self.builder` (C++ MLIR OpBuilder)
- Implements type checking and promotion

**Example**:
```python
class TritonSemantic:
    def __init__(self, builder):
        self.builder = builder  # ir.builder (C++ object)

    def load(self, ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
        # Type checking
        ptr_ty = ptr.type
        if not ptr_ty.is_ptr():
            raise TypeError(f"Expected pointer, got {ptr_ty}")

        # Generate MLIR operation
        result_ty = ptr_ty.element_ty
        if mask is not None:
            # Masked load
            mlir_val = self.builder.create_masked_load(
                ptr.handle, mask.handle, other.handle if other else None
            )
        else:
            # Regular load
            mlir_val = self.builder.create_load(ptr.handle)

        return tensor(mlir_val, result_ty)

    def program_id(self, axis):
        return tensor(self.builder.create_get_program_id(axis), tl.int32)

    def binary_op_type_checking_impl(self, a_ty, a_is_scalar, b_ty, b_is_scalar, div_or_mod):
        # Complex type promotion logic
        # ... (determines result type for operations like a + b)
        pass
```

### Why the Separation?

1. **Abstraction**: Users interact with high-level `triton.language` functions, not MLIR details

2. **Flexibility**: Multiple backends (NVIDIA, AMD, Intel) can provide different `Semantic` implementations

3. **Type Safety**: Argument validation happens in `core`, MLIR generation in `semantic`

4. **Testability**: Semantic layer can be tested independently

5. **Code Reuse**: Standard library functions use core primitives

### Complete Flow Example: tl.load()

```
User code:
  data = tl.load(ptr)
    ↓
CodeGenerator.visit_Call():
  - Recognizes tl.load is a builtin
  - Adds kwargs['_semantic'] = self.semantic
    ↓
triton.language.core.load(ptr, _semantic=semantic):
  - Validates ptr is a pointer
  - Calls semantic.load(ptr)
    ↓
triton.language.semantic.TritonSemantic.load(ptr):
  - Type checks
  - Calls self.builder.create_load(ptr.handle)
    ↓
Python → C++ (pybind11):
  - ir.builder.create_load() in C++
    ↓
C++ MLIR OpBuilder:
  - Creates tt.load operation
  - Returns mlir::Value handle
    ↓
Back to Python:
  - Wrapped in tensor(mlir_val, element_ty)
  - Stored in lscope["data"]
```

---

## Frame 16: Compilation Stages - TTIR → TTGIR → LLIR → PTX → CUBIN

After the AST is converted to TTIR, the backend runs compilation stages.

**File**: `[third_party/nvidia/backend/compiler.py:533](../../../third_party/nvidia/backend/compiler.py#L533)`

```python
def add_stages(self, stages, options, language):
    if language == Language.TRITON:
        stages["ttir"] = make_ttir     # Triton IR optimizations
        stages["ttgir"] = make_ttgir   # GPU-specific lowering
    stages["llir"] = make_llir         # LLVM IR
    stages["ptx"] = make_ptx           # PTX assembly
    stages["cubin"] = make_cubin       # Binary
```

### Stage 1: TTIR Optimization

**File**: `compiler.py:229-243`

```python
def make_ttir(mod, metadata):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.add_triton_rewrite_tensor_pointer_pass(target)
    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_sccp_pass()
    pm.add_loop_unroll_pass()
    pm.run(mod)
    return mod
```

**Passes**:
- **Inliner**: Inline function calls
- **Rewrite Tensor Pointers**: Optimize pointer arithmetic
- **Canonicalizer**: Simplify operations
- **CSE**: Eliminate common subexpressions
- **SCCP**: Sparse conditional constant propagation
- **Loop Unroll**: Unroll small loops

### Stage 2: TTGIR - GPU Lowering with Warp Specialization

**File**: `compiler.py:246-317`

```python
def make_ttgir(mod, metadata, target):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    # Convert to TritonGPU dialect
    pm.add_convert_triton_to_tritongpu_pass(target, num_warps, num_ctas)

    # Coalesce memory accesses
    pm.add_tritongpu_coalesce_pass()

    # Accelerate matmuls with tensor cores
    pm.add_tritongpu_accelerate_matmul_pass(target)

    # Optimize layouts
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()

    # Software pipelining
    pm.add_tritongpu_pipeline_pass(num_stages)

    # WARP SPECIALIZATION PASS (SM80+)
    if target.arch >= 80:  # Ampere (A100) or newer
        pm.add_tritongpu_optimize_thread_locality_pass()
        pm.add_tritongpu_ws_fixup_missing_attrs_pass()
        pm.add_tritongpu_ws_materialize_pass()
        pm.add_tritongpu_ws_rewrite_matmul_pass()

    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.run(mod)
    return mod
```

**Key Passes**:
- **ConvertTritonToTritonGPU**: Adds layout information (blocked, shared, etc.)
- **Coalesce**: Optimizes memory access patterns
- **AccelerateMatmul**: Maps `tt.dot` to tensor core operations (wgmma, mma)
- **Pipeline**: Implements software pipelining for overlapping compute and memory
- **OptimizeThreadLocality**: Analyzes and creates warp specialization regions
- **WS Passes**: Materialize and optimize warp-specialized code

After this stage, the MLIR contains:
- `ttng.warp_group_dot` or `ttng.tc_gen5_mma` for matmuls
- `ttng.async_tma_copy_global_to_local` for TMA loads
- `ttg.warp_specialize` regions with partitions

### Stage 3: LLIR - LLVM Dialect

**File**: `compiler.py:337-429`

```python
def make_llir(mod, metadata, target):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    # Allocate shared memory
    pm.add_tritongpu_allocate_shared_memory_pass()

    # Lower to LLVM dialect
    pm.add_convert_tritongpu_to_llvm_pass(target)
    pm.add_convert_nv_gpu_to_llvm_pass()
    pm.add_convert_scf_to_cf_pass()
    pm.add_convert_index_to_llvmir_pass()
    pm.add_convert_arith_to_llvmir_pass()

    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)

    # Translate MLIR LLVM dialect to native LLVM IR
    llvm_ir = mod.to_llvm_ir()

    # Link external libraries (libdevice for math functions)
    for lib in extern_libs:
        llvm_ir = link_extern_lib(llvm_ir, lib)

    # Run LLVM optimization passes
    llvm_ir = optimize_llvm_ir(llvm_ir, target)

    return llvm_ir
```

**Key Transformations**:
- **AllocateSharedMemory**: Assigns shared memory addresses
- **ConvertTritonGPUToLLVM**: Lowers high-level ops to LLVM + inline PTX
- **ConvertNVGPUToLLVM**: Lowers NVIDIA-specific ops (TMA, wgmma)
- **Warp Specialization to LLVM**: Converts warp specialization regions to:
  - Warp ID checks: `if (warpId < num_producer_warps)`
  - Divergent control flow
  - Synchronization primitives

### Stage 4: PTX Assembly

**File**: `compiler.py:431-455`

```python
def make_ptx(llvm_ir, metadata, target):
    # Use LLVM backend to generate PTX
    ptx = llvm_ir_to_ptx(llvm_ir, target)

    # Post-process PTX version
    ptx = ptx.replace(".version 7.0", f".version {target.ptx_version}")

    return ptx
```

The LLVM NVPTX backend generates PTX assembly with:
- Warp-level synchronization: `bar.sync`, `bar.warp.sync`
- TMA instructions: `cp.async.bulk.tensor.*`
- Tensor core instructions: `wgmma.*`, `mma.*`
- Conditional execution based on warp ID

### Stage 5: CUBIN Binary

**File**: `compiler.py:457-524`

```python
def make_cubin(ptx, metadata, target):
    # Compile PTX to CUBIN using ptxas
    ptxas_path = find_ptxas()

    # Build command
    cmd = [
        ptxas_path,
        "-arch", f"sm_{target.arch}",
        "--gpu-name", f"sm_{target.arch}",
        "-o", cubin_path,
        ptx_path
    ]

    if debug:
        cmd += ["--device-debug", "--generate-line-info"]

    # Run ptxas
    subprocess.run(cmd, check=True)

    # Read binary
    with open(cubin_path, 'rb') as f:
        cubin = f.read()

    return cubin
```

---

## Frame 17: Kernel Launch and Execution

Back to the test:

```python
kernel = matmul_tma_ws_kernel[grid](A, B, C, ...)
```

At this point, `kernel` is a `CompiledKernel` object.

### Step 17.1: Accessing the Kernel

**File**: `compiler.py:490-501`

```python
def __getitem__(self, grid):
    self._init_handles()

    def runner(*args, stream=None):
        if stream is None:
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        launch_metadata = self.launch_metadata(grid, stream, *args)
        self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata,
                 launch_metadata, knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                 *args)

    return runner
```

### Step 17.2: Initialize GPU Handles

**File**: `compiler.py:436-471`

```python
def _init_handles(self):
    if self.module is not None:
        return

    device = driver.active.get_current_device()

    # Create launcher
    self._run = driver.active.launcher_cls(self.src, self.metadata)

    # Check resource limits
    max_shared = max_shared_mem(device)
    if self.metadata.shared > max_shared:
        raise OutOfResources(self.metadata.shared, max_shared, "shared memory")

    # Load binary to GPU
    self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = \
        driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device
        )

    # Check warp/thread limits
    warp_size = driver.active.get_current_target().warp_size
    if self.metadata.num_warps * warp_size > self.n_max_threads:
        raise OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads")
```

### Step 17.3: Launch Kernel

The `runner` function calls `self.run()`, which is a launcher object:

**File**: `[python/triton/runtime/driver.py](../../../python/triton/runtime/driver.py)` (CUDALauncher)

```python
class CUDALauncher:
    def __call__(self, grid_0, grid_1, grid_2, stream, function, packed_metadata,
                 launch_metadata, enter_hook, exit_hook, *args):
        # Pack arguments
        packed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                packed_args.append(arg.data_ptr())
            else:
                packed_args.append(arg)

        # Call launch hook
        if enter_hook:
            enter_hook(launch_metadata.get())

        # LAUNCH KERNEL via CUDA driver API
        driver.cuLaunchKernel(
            function,              # CUfunction
            grid_0, grid_1, grid_2,  # Grid dimensions
            num_threads_per_block, 1, 1,  # Block dimensions (num_warps * 32)
            shared_mem_bytes,      # Shared memory
            stream,                # CUDA stream
            packed_args,           # Kernel arguments
        )

        # Call exit hook
        if exit_hook:
            exit_hook(launch_metadata.get())
```

### What Happens on the GPU

1. **Grid Launch**: CUDA creates `grid_0 × grid_1 × grid_2` thread blocks

2. **Block Composition**: Each block has `num_warps * 32` threads
   - For `num_warps=8`: 256 threads per block
   - Organized as 8 warps of 32 threads each

3. **Warp Specialization Execution**:
   - **Producer Warps** (e.g., warp 0):
     - Execute partition region
     - Issue TMA async copies
     - Load data into shared memory

   - **Consumer Warps** (e.g., warps 1-7):
     - Execute default region
     - Perform tensor core matmul operations
     - Use data from shared memory

   - **Synchronization**:
     - `bar.sync` ensures all warps reach sync point
     - TMA completion tracked via `wait_group`
     - Double/triple buffering for pipeline

4. **Example Execution Timeline** (3-stage pipeline):
   ```
   Iteration 0:
     Producer: Load A[0], B[0]
     Consumer: Wait...

   Iteration 1:
     Producer: Load A[1], B[1]
     Consumer: Compute C += A[0] @ B[0]

   Iteration 2:
     Producer: Load A[2], B[2]
     Consumer: Compute C += A[1] @ B[1]

   ...

   Iteration k-1:
     Producer: Idle
     Consumer: Compute C += A[k-1] @ B[k-1]
   ```

---

## Frame 18: Result Verification

```python
ttgir = kernel.asm["ttgir"]
if is_blackwell():
    assert "ttng.tc_gen5_mma" in ttgir
    assert "ttng.async_tma_copy_global_to_local" in ttgir
else:
    assert "ttng.warp_group_dot" in ttgir
    assert "ttng.async_tma_copy_global_to_local" in ttgir

if is_hopper() and num_warps == 8:
    assert "ttg.warp_specialize" not in ttgir
else:
    assert "ttg.warp_specialize" in ttgir
```

### What This Checks

1. **Tensor Core Usage**:
   - Blackwell (SM100): Uses 5th-gen tensor cores (`tc_gen5_mma`)
   - Hopper (SM90): Uses warp group matrix multiply (`warp_group_dot`)

2. **TMA (Tensor Memory Accelerator)**:
   - Asynchronous global → shared memory copies
   - Hardware-accelerated, allows overlapping with compute

3. **Warp Specialization**:
   - Present for most configurations
   - Disabled for Hopper + 8 warps (not enough warps to specialize)

Finally, verify correctness:
```python
ref_out = torch.empty((M, N), dtype=dtype, device=device)
cublas.matmul(A, B, ref_out)
torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16),
                           atol=0.03, rtol=0.03)
```

---

## Summary: Complete Compilation Flow

```
User writes kernel:
  @triton.jit
  def matmul_tma_ws_kernel(...):
      a_desc = tl.make_tensor_descriptor(...)
      for k in tl.range(..., warp_specialize=True):
          a = a_desc.load(...)
          accumulator = tl.dot(a, b.T, accumulator)
      ...

Decorator application (module import):
  → JITFunction created
  → AST parsed and hashed
  → Cached for later

Kernel invocation:
  kernel[grid](args)

  → JITFunction.run()
  → Argument specialization
  → Cache lookup (miss)
  → compile() called

compile():
  → ASTSource.make_ir()
  → ast_to_ttir()

ast_to_ttir():
  → CodeGenerator created
    - builder: ir.builder (C++ MLIR OpBuilder)
    - semantic: TritonSemantic (MLIR generation)
  → generator.visit(ast)

AST Visitation:
  → visit_FunctionDef: Create MLIR function
  → visit_Assign: Process assignments
  → visit_Call: Handle function calls
    - Builtins: Add _semantic parameter
    - JIT functions: Inline or call

  For tl.make_tensor_descriptor():
    → core.make_tensor_descriptor(_semantic=semantic)
    → semantic.builder.create_make_tma_desc()
    → Python → C++ (pybind11)
    → MLIR operation created

  For tl.range(..., warp_specialize=True):
    → Recognized by code generator
    → Marks loop for warp specialization
    → Generates scf.for with annotations

MLIR Module (TTIR) returned

Backend Stages:

  Stage 1: TTIR → optimized TTIR
    → Inlining, canonicalization, CSE

  Stage 2: TTIR → TTGIR
    → Add GPU layouts
    → Warp specialization pass:
      - Analyze loop for async ops
      - Create ttg.warp_specialize regions
      - Partition producer/consumer operations
    → Matmul acceleration (tensor cores)
    → Software pipelining

  Stage 3: TTGIR → LLIR
    → Allocate shared memory
    → Lower to LLVM dialect
    → Warp specialization → LLVM:
      - Warp ID checks
      - Divergent control flow

  Stage 4: LLIR → PTX
    → LLVM backend generates PTX
    → TMA instructions, wgmma instructions

  Stage 5: PTX → CUBIN
    → ptxas assembler
    → Machine code binary

CompiledKernel returned, cached

Kernel Launch:
  → Load CUBIN to GPU
  → cuLaunchKernel()
  → GPU execution:
    - Producer warps: TMA loads
    - Consumer warps: Tensor core compute
    - Pipelined with synchronization

Results verified
```

---

## Key Insights

1. **Decorator Pattern**: `@triton.jit` wraps functions without compiling them immediately

2. **Lazy Compilation**: Compilation happens on first invocation with specific argument types

3. **AST-Based**: Python AST is walked to generate MLIR, not direct Python execution

4. **Semantic Separation**: `triton.language.core` (user API) delegates to `triton.language.semantic` (MLIR generation)

5. **Python ↔ C++ Integration**: Seamless via pybind11, MLIR operations created in C++

6. **Multi-Stage Lowering**: Progressive lowering through MLIR dialects (Triton → TritonGPU → LLVM)

7. **Warp Specialization**: Enabled by analysis passes, creates heterogeneous workload distribution

8. **Caching**: Aggressive caching at multiple levels (AST hash, compilation cache)

9. **Standard Library**: Many `tl.*` functions are themselves JIT-compiled and inlined

10. **Hardware Specialization**: Different paths for different GPUs (SM80, SM90, SM100)

This completes the frame-by-frame trace of `test_warp_specialize_tma_matmul`!
