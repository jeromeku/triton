# SwiGLU-Matmul Fusion in Triton Kernels

## Overview

The Triton matmul kernel supports fusing a SwiGLU activation directly into the
matmul epilogue.  Instead of writing the full `[M, N]` accumulator to global
memory and launching a separate SwiGLU kernel, the activation is applied
in-register to each tile of the accumulator before writing a **halved** output
of shape `[M, N/2]`.

Key files:

| File | Role |
|---|---|
| `triton_kernels/matmul.py` | Host-side dispatch; builds `FusedActivation`, allocates output |
| `triton_kernels/specialize.py` | `FnSpecs` / `SpecializationModule` — inlines the activation into the kernel |
| `triton_kernels/swiglu_details/_swiglu.py` | Triton JIT functions: `_swiglu_fn`, `compute_swiglu` |
| `triton_kernels/swiglu.py` | Standalone SwiGLU kernel + `swiglu_torch` reference |
| `triton_kernels/matmul_details/_matmul.py` | Non-persistent matmul kernel |
| `triton_kernels/matmul_details/_p_matmul.py` | Persistent (TMA) matmul kernel |

---

## 1. What is SwiGLU?

SwiGLU is a gated activation from [Shazeer 2020](https://arxiv.org/abs/2002.05202).
Given a vector `x` of width `N`, split it into two halves of width `N/2`:

```
x_gelu   = x[..., 0::2]   # even columns (the "gate" half)
x_linear = x[..., 1::2]   # odd columns  (the "value" half)

swish   = x_gelu * sigmoid(alpha * x_gelu)   # SiLU / Swish gate
output  = swish * (x_linear + 1)              # gated output
```

With optional clamping controlled by `limit`:
- `x_gelu` is clamped to `(-inf, limit]`
- `x_linear` is clamped to `[-limit, limit]`

The output has shape `[M, N/2]` — exactly half the columns of the input.

---

## 2. How the Fusion is Wired

### 2a. Test / call-site (`test_matmul.py:365-368`)

```python
fused_activation = FusedActivation(
    FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
    swiglu_opts   # e.g. (1.1, 1.4) -> alpha=1.1, limit=1.4
)
```

- `FnSpecs.fn` = `_swiglu_fn` — a `@triton.jit` function
- `FnSpecs.fn_arg_names` = `("alpha", "limit")` — names of the closure args
- `FnSpecs.reduction_n` = `2` — the activation **halves** the N dimension
- `FusedActivation.fn_args` = `swiglu_opts` = `(alpha_value, limit_value)` — concrete values

### 2b. Kernel specialization (`matmul.py:479`)

```python
kernels = specializations.get(epilogue=epilogue.specs, activation=matmul_fused_activation.specs)
```

`SpecializationModule` (`specialize.py`) injects:
- `ACTIVATION_FN: tl.constexpr = _swiglu_fn` — the JIT function, inlined at compile time
- `activation_fn_args` — the `(alpha, limit)` tuple, passed as kernel args

This creates a **specialized** version of the matmul kernel with the SwiGLU body
baked into the epilogue path. No function-pointer overhead at runtime.

### 2c. Kernel launch args (`matmul.py:516`)

```python
*matmul_fused_activation.fn_args, matmul_fused_activation.specs.reduction_n,
```

This passes the concrete `alpha`, `limit`, and `ACTIVATION_REDUCTION_N=2` to the kernel.

### 2d. Inside the kernel (`_matmul.py:421-428`, `_p_matmul.py:498-503`)

After the matmul accumulation loop completes and scaling/bias are applied:

```python
# acc shape: [BLOCK_M, BLOCK_N]   (e.g. [128, 256])
if ACTIVATION_FN is not None:
    out = ACTIVATION_FN(acc, *activation_fn_args)
    # out shape: [BLOCK_M, OUT_BLOCK_N]  (e.g. [128, 128])
else:
    out = acc
```

Where:
```python
OUT_BLOCK_N = BLOCK_N // ACTIVATION_REDUCTION_N   # BLOCK_N / 2
```

The activation function `_swiglu_fn` does this in-register:

```python
@triton.jit
def _swiglu_fn(input, alpha, limit):
    # input shape: [BLOCK_M, BLOCK_N]
    # reshape to [BLOCK_M, BLOCK_N/2, 2] then split on last dim
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    # gelu, linear each: [BLOCK_M, BLOCK_N/2]
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)
    # returns: [BLOCK_M, BLOCK_N/2]
```

The output tile is then written to global memory with column offsets scaled by
`OUT_BLOCK_N` instead of `BLOCK_N`.

---

## 3. Accumulator Tile Layout, Reshape, and Split

### 3a. Background: unfused MLP with separate gate and up projections

In a standard SwiGLU MLP (e.g., LLaMA), the first linear layer has two
conceptually separate weight matrices:

```
W_gate [K, N]   — produces the "gate" (SiLU) pathway
W_up   [K, N]   — produces the "value" (linear) pathway
```

Without any fusion, you'd compute:

```python
gate = X @ W_gate          # [M, N]
up   = X @ W_up            # [M, N]
out  = swiglu(gate, up)    # [M, N]
```

That's two separate matmuls, two global memory writes of `[M, N]`, then a
pointwise kernel that reads both back.

### 3b. Fused weight matrix: column-interleaved, not block-concatenated

To fuse these into a single matmul, you combine W_gate and W_up into one
`[K, 2N]` matrix. But **how** you combine them matters enormously.

**Option A — Block-concatenated (NOT what this code does):**
```
W_fused = [W_gate | W_up]     shape [K, 2N]
         columns 0..N-1 = gate
         columns N..2N-1 = up

result = X @ W_fused           shape [M, 2N]
gate = result[:, :N]
up   = result[:, N:]
```

**Option B — Column-interleaved (what this code DOES):**
```
W_fused[:, 0::2] = W_gate     even columns = gate
W_fused[:, 1::2] = W_up       odd columns  = up

result = X @ W_fused           shape [M, 2N]
gate = result[:, 0::2]
up   = result[:, 1::2]
```

The code uses **Option B**. Why? Because of how tiled matmul works on the GPU.

### 3c. Where the interleaving happens (or doesn't)

**The interleaving is a weight preparation convention, not a kernel operation.**
There is no explicit interleaving step in the matmul or SwiGLU kernel code. The
contract is that whoever prepares the weight tensor (model initialization,
checkpoint loading, weight conversion) must store it so that even columns are
gate and odd columns are up:

```python
# Done once during model preparation / checkpoint loading:
W_fused = torch.empty(K, 2 * N, dtype=dtype, device=device)
W_fused[:, 0::2] = W_gate   # even columns = gate
W_fused[:, 1::2] = W_up     # odd columns  = up
```

Evidence from the codebase:

- In the **test** (`test_matmul.py:346`), `b` is simply
  `make_random_tensor(shape=(k, n))` with random data. No interleaving is
  performed because the test is checking numerical correctness: both the fused
  kernel and the reference (`matmul_torch` + standalone `swiglu`) apply the same
  even/odd convention to the same random tensor, so the results match regardless
  of what the column values actually represent.

- In the **benchmark** (`bench_mlp.py:117-118`):
  ```python
  w1_ep_local = torch.randn((n_expts_tot // EP, dim1, dim2), device=dev)      # [E, K, 2N]
  w2_ep_local = torch.randn((n_expts_tot // EP, dim2 // 2, dim1), device=dev)  # [E, N, K]
  ```
  `w1` has width `dim2 = 2N` (the full interleaved width), and `w2` has K
  dimension `dim2 // 2 = N` (the halved width after SwiGLU). Again, random
  values — the benchmark measures performance, not functional correctness.

- In a real deployment, the model serving framework interleaves W_gate and W_up
  into the fused layout at weight-loading time.

### 3d. Why each threadblock automatically gets both halves

Given a column-interleaved `W_fused [K, 2N]`, the standard matmul tiling handles
the rest naturally — no special tiling or scheduling logic is needed.

In a tiled matmul, each CTA (cooperative thread array / threadblock) computes
one tile of the output of shape `[BLOCK_M, BLOCK_N]`. Crucially, each CTA
only has access to its own tile — it cannot cheaply read another CTA's registers.

The matmul tiles over the output N dimension in contiguous chunks of `BLOCK_N`.
Because the interleaving alternates at **every single column**, any contiguous
slice of `BLOCK_N` columns contains exactly `BLOCK_N/2` gate values and
`BLOCK_N/2` up values, perfectly paired for the same output positions.

Consider what happens with **block-concatenated** weights and `BLOCK_N = 128`:

```
CTA for columns [0, 128)    → gets ONLY gate data
CTA for columns [128, 256)  → gets ONLY up data
```

To apply SwiGLU, you need both `gate[m, j]` and `up[m, j]` for every output
position `j`. With block concatenation, these live in **different CTAs**. You'd
have to write both tiles to global memory, then launch a separate SwiGLU kernel
to pair them — which is exactly the unfused approach.

Now consider **column-interleaved** weights with `BLOCK_N = 128`:

```
CTA for columns [0, 128) → has gate AND up for output positions 0..63
  col 0 = gate[0], col 1 = up[0], col 2 = gate[1], col 3 = up[1], ...
```

Every CTA has **both gate and up values for the same output positions**, sitting
in adjacent register slots within the tile. The SwiGLU can be applied entirely
within the CTA's local registers, with no cross-CTA communication and no global
memory round-trip.

### 3d. The reshape + split operation in detail

Here is the exact code path (`_swiglu.py:68-70`):

```python
@triton.jit
def _swiglu_fn(input, alpha, limit):
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)
```

`input` is the matmul accumulator tile with shape `[BLOCK_M, BLOCK_N]`.
Let's trace through with a concrete example: `BLOCK_M=4, BLOCK_N=8`.

**Step 1: The accumulator tile after matmul (the `input`).**

Each element `acc[m, n]` is the dot product of row `m` of X with column `n` of
W_fused. Because the weights are interleaved, even columns hold gate results and
odd columns hold up results:

```
input [4, 8]:
         col0    col1    col2    col3    col4    col5    col6    col7
         gate₀   up₀     gate₁   up₁     gate₂   up₂     gate₃   up₃
row 0 [ g(0,0) u(0,0) g(0,1) u(0,1) g(0,2) u(0,2) g(0,3) u(0,3) ]
row 1 [ g(1,0) u(1,0) g(1,1) u(1,1) g(1,2) u(1,2) g(1,3) u(1,3) ]
row 2 [ g(2,0) u(2,0) g(2,1) u(2,1) g(2,2) u(2,2) g(2,3) u(2,3) ]
row 3 [ g(3,0) u(3,0) g(3,1) u(3,1) g(3,2) u(3,2) g(3,3) u(3,3) ]
```

**Step 2: `tl.reshape(input, (4, 4, 2))`**

Reshape treats the data in row-major (C-contiguous) order. Each row of 8
elements is reinterpreted as 4 groups of 2:

```
reshaped [4, 4, 2]:
                j=0           j=1           j=2           j=3
row 0   [ [g(0,0),u(0,0)] [g(0,1),u(0,1)] [g(0,2),u(0,2)] [g(0,3),u(0,3)] ]
row 1   [ [g(1,0),u(1,0)] [g(1,1),u(1,1)] [g(1,2),u(1,2)] [g(1,3),u(1,3)] ]
row 2   [ [g(2,0),u(2,0)] [g(2,1),u(2,1)] [g(2,2),u(2,2)] [g(2,3),u(2,3)] ]
row 3   [ [g(3,0),u(3,0)] [g(3,1),u(3,1)] [g(3,2),u(3,2)] [g(3,3),u(3,3)] ]

dim 0 = row (M)
dim 1 = output position (N/2)
dim 2 = [gate, up] pair
```

The element at `[m, j, 0]` is `gate(m, j)` and `[m, j, 1]` is `up(m, j)`.
This works because the interleaved layout places each (gate, up) pair in
**adjacent columns** — which are adjacent in the row-major flat order — so the
reshape into groups of 2 naturally groups them as pairs.

**Step 3: `tl.split(...)`**

`tl.split` splits a tensor along its **last dimension**, which must have size 2
(`core.py:1853-1868`). It produces two tensors, each with the last dimension
removed:

```
gelu [4, 4]:                          linear [4, 4]:
       j=0    j=1    j=2    j=3             j=0    j=1    j=2    j=3
  0 [ g(0,0) g(0,1) g(0,2) g(0,3) ]   0 [ u(0,0) u(0,1) u(0,2) u(0,3) ]
  1 [ g(1,0) g(1,1) g(1,2) g(1,3) ]   1 [ u(1,0) u(1,1) u(1,2) u(1,3) ]
  2 [ g(2,0) g(2,1) g(2,2) g(2,3) ]   2 [ u(2,0) u(2,1) u(2,2) u(2,3) ]
  3 [ g(3,0) g(3,1) g(3,2) g(3,3) ]   3 [ u(3,0) u(3,1) u(3,2) u(3,3) ]
```

Each is `[BLOCK_M, BLOCK_N/2]` — exactly the gate and up projections for this
tile's output positions.

**Step 4: `compute_swiglu(gelu, linear, ...)`**

Now the SwiGLU is a simple elementwise operation on two same-shaped tensors:
```
s = gelu / (1 + exp(-alpha * gelu))      # SiLU/Swish
out = s * (linear + 1)                    # gated output
```

Output shape: `[BLOCK_M, BLOCK_N/2]` = `[4, 4]`.

### 3e. Why reshape + split is zero-cost

Both `tl.reshape` and `tl.split` are **metadata-only operations** in Triton.
They change the logical shape/indexing of a tensor but do not move any data in
registers. The underlying flat register layout is unchanged — the compiler just
reinterprets which register corresponds to which logical index.

This is exactly why the interleaved layout is chosen: the (gate, up) pairs for
each output position are already in adjacent register slots from the matmul.
The reshape just tells the compiler "view these 8 registers per row as 4 groups
of 2," and split says "the first of each group is `gelu`, the second is
`linear`." No shuffles, no shared memory, no extra loads.

If the weights were block-concatenated instead, the gate and up values for the
same output position would be `BLOCK_N/2` registers apart — in general, owned by
**different threads within the warp**. Pairing them would require cross-lane
shuffles or shared memory exchanges, adding significant overhead.

### 3f. Comparison: standalone `_swiglu` kernel vs fused

The standalone kernel (`_swiglu.py:86-118`) operates on a `[M, 2N]` tensor
already in global memory. It uses the same interleaved layout but must **load**
the data:

```python
# _swiglu.py:97 — loads 2*BLOCK_N contiguous columns per tile
packed_off_n = pid_n * 2 * BLOCK_N + tl.arange(0, 2 * BLOCK_N)
...
a_packed = tl.load(A + packed_offs, ...)    # [BLOCK_M, 2*BLOCK_N]
a_gelu, a_linear = tl.split(tl.reshape(a_packed, (BLOCK_M, BLOCK_N, 2)))
```

The standalone kernel tiles its grid over the **output** space `[M, N]` (where
`N` is half the input width), but loads `2 * BLOCK_N` contiguous input columns
per tile to get both the gate and up values. The reshape + split is identical.

The fused version avoids this entirely: the accumulator is already in registers
from the matmul — no global memory load needed.

### 3g. SWAP_XW consideration

When `SWAP_XW` is true (e.g., Hopper with MX-scaled weights), the accumulator
is transposed (`acc = acc.trans()`) before the activation is applied. The
interleaving requirement applies to the accumulator **after** this transpose,
which still corresponds to the N (column) dimension of the output.

---

## 4. Split-K Handling

When `split_k > 1`, the activation **cannot** be fused into the matmul tiles
because partial sums must be reduced first. The code swaps the activation to the
reduce phase:

```python
# matmul.py:364-368
matmul_fused_activation = fused_activation
reduce_fused_activation = FusedActivation()
if opt_flags.split_k > 1:
    # swap: matmul writes full [M, N] partials; reduce applies SwiGLU after summation
    matmul_fused_activation, reduce_fused_activation = reduce_fused_activation, matmul_fused_activation
```

In this case:
1. The matmul kernel writes `[split_k, batch, M, N]` partial sums (full width, no activation)
2. The `reduce()` kernel sums along the split-k dimension, then applies SwiGLU,
   producing the final `[M, N/2]` output

---

## 5. Output Shape

The output allocation reflects the halving (`matmul.py:159`):

```python
out_shape = (batch_dim, y_rows, N // fused_activation.specs.reduction_n)
# With SwiGLU: N // 2
```

The test also confirms this (`test_matmul.py:373`):

```python
c_shape += (b.shape[-1] // (1 if fused_activation is None else fused_activation.specs.reduction_n),)
```

---

## 6. Reference PyTorch Implementation

The equivalent unfused PyTorch computation is:

```python
import torch
import torch.nn.functional as F

def matmul_swiglu_reference(
    X: torch.Tensor,       # [M, K] activations
    W: torch.Tensor,       # [K, N] weights (N = 2 * output_width, interleaved)
    alpha: float = 1.0,    # SiLU scaling parameter
    limit: float = None,   # optional clamp value
    bias: torch.Tensor = None,  # [N] optional bias
) -> torch.Tensor:
    """
    Equivalent to: matmul(X, W, fused_activation=SwiGLU(alpha, limit))

    Returns: [M, N/2]
    """
    # Step 1: Standard matmul -> [M, N]
    acc = X.float() @ W.float()
    if bias is not None:
        acc = acc + bias.float()

    # Step 2: Split interleaved columns into gate and value halves
    x_gelu   = acc[..., 0::2]    # [M, N/2] — gate pathway (even cols)
    x_linear = acc[..., 1::2]    # [M, N/2] — value pathway (odd cols)

    # Step 3: Optional clamping
    if limit is not None:
        x_gelu   = x_gelu.clamp(max=limit)            # upper clamp only
        x_linear = x_linear.clamp(min=-limit, max=limit)  # symmetric clamp

    # Step 4: SwiGLU activation
    swish  = x_gelu * torch.sigmoid(alpha * x_gelu)   # SiLU/Swish gate
    output = swish * (x_linear + 1)                    # gated output

    return output  # [M, N/2]
```

### Relationship to the test reference

In `test_matmul.py:413-418`, the reference is computed in two steps:

```python
# Step 1: unfused matmul -> [M, N]
ref_y = matmul_torch(a, b, bias, ...)

# Step 2: standalone SwiGLU -> [M, N/2]
ref_y = swiglu(ref_y, alpha=swiglu_opts[0],
               precision_config=SwiGLUPrecisionConfig(swiglu_opts[1]))
```

The standalone `swiglu()` function (`swiglu.py:85-86`) launches a separate Triton
kernel that reads the full `[M, N]` tensor from global memory, applies the same
`compute_swiglu` math, and writes the `[M, N/2]` result. The fused version
avoids this extra global memory round-trip.

### Pure PyTorch reference (`swiglu.py:89-100`)

```python
def swiglu_torch(a, alpha, precision_config):
    limit = precision_config.limit
    a_gelu = a[..., ::2]                          # even columns
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]                        # odd columns
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    out = out_gelu * (a_linear + 1)
    return out
```

---

## 7. The Specialization Process, End-to-End

> This section uses SwiGLU fusion as a concrete worked example of how
> `SpecializationModule` / `specialize()` work. The same machinery is used for
> any closure-style function injection into Triton kernels (epilogues, fused
> activations, etc.), so this trace generalizes.

### Phase 1: Module-level registration (import time)

At `matmul.py:69-75`, a singleton `SpecializationModule` is created:

```python
specializations = SpecializationModule("matmul",
    kernels=[("_matmul", _matmul), ("_p_matmul", _p_matmul)],
    closure_args={
        "epilogue": ClosureArg("EPILOGUE_FN", "epilogue_fn_args"),
        "activation": ClosureArg("ACTIVATION_FN", "activation_fn_args"),
    },
)
```

This registers two **template** kernels (`_matmul` and `_p_matmul`) and declares
two "slots" that can be filled: `epilogue` and `activation`. Each `ClosureArg`
names:
- `fn_name` — the kernel parameter that holds the **function** (e.g., `"ACTIVATION_FN"`)
- `fn_params_name` — the kernel parameter that holds the **function's argument tuple** (e.g., `"activation_fn_args"`)

The template kernel's original signature contains both of these as explicit parameters:

```python
# _matmul.py:49
ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr,
```

At this point, nothing has been compiled. We just have a registry.

### Phase 2: `.get()` — source-code surgery (first call per combo, then cached)

At `matmul.py:479`:

```python
kernels = specializations.get(epilogue=epilogue.specs, activation=matmul_fused_activation.specs)
```

For a SwiGLU call, `matmul_fused_activation.specs` is:

```python
FnSpecs(name="swiglu", fn=_swiglu_fn, fn_arg_names=("alpha", "limit"), reduction_n=2)
```

Inside `SpecializationModule.get()` (`specialize.py:183-203`):

**Step 2a — Cache check.** A cache key is formed from the `.name` of each spec:
`("dflt", "swiglu")`. If a module was already built for this key, return it
immediately. This means the expensive source rewriting happens only once per
unique combination of epilogue + activation.

**Step 2b — Build the constant and tuple maps.**

```python
# specialize.py:192-193
spec_constants = {
    "EPILOGUE_FN":   None,         # epilogue slot: no epilogue function
    "ACTIVATION_FN": _swiglu_fn,   # activation slot: the SwiGLU JIT function
}
spec_tuples = {
    "epilogue_fn_args":   (),                # epilogue has no args
    "activation_fn_args": ("alpha", "limit") # activation arg *names*
}
```

- `spec_constants` maps each `ClosureArg.fn_name` → the actual `JITFunction` (or `None`).
- `spec_tuples` maps each `ClosureArg.fn_params_name` → the **names** of the
  closure parameters (not their values — just the names that will become kernel
  parameters).

**Step 2c — Call `specialize()` for each template kernel.** A synthetic Python
module is created, and for each of `_matmul` / `_p_matmul`:

```python
# specialize.py:199-201
setattr(module, kernel_name,
        specialize(kernel_fn, module, spec_constants, spec_tuples, ...))
```

### Phase 3: `specialize()` — rewriting the kernel source (`specialize.py:80-166`)

This is where the magic happens. `specialize()` performs **source-level surgery**
on the Triton kernel's Python source code.

**3a. Extract the original source.**

```python
src = inspect.getsource(fn.fn)   # gets the raw Python text of _matmul
```

**3b. Parse the function signature** into individual argument strings.

The original `_matmul` signature contains (among many other args):

```python
def _matmul(...,
            ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr,
            EPILOGUE_FN: tl.constexpr, epilogue_fn_args,
            ...):
```

**3c. Remove specialized parameters, expand tuples.** For each arg in the
original signature (`specialize.py:109-114`):

- If `arg_key in constants` (e.g., `"ACTIVATION_FN"`, `"EPILOGUE_FN"`): **remove
  it** from the new signature entirely. It will become a constexpr local.
- If `arg_key in tuples` (e.g., `"activation_fn_args"`): **replace it** with the
  expanded individual parameter names from the tuple.

So the signature transforms like this:

```
Original:   ..., ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr, ...
                 ^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^
                 removed (in constants)        replaced (in tuples)

New:        ..., alpha, limit, ACTIVATION_REDUCTION_N: tl.constexpr, ...
                 ^^^^^  ^^^^^
                 individual params from fn_arg_names
```

**3d. Inject constexpr assignments at the top of the body.**

These lines are prepended to the kernel body (`specialize.py:120-122`):

```python
    ACTIVATION_FN: tl.constexpr = _swiglu_fn    # ← the JITFunction object
    EPILOGUE_FN: tl.constexpr = None
```

This is the key transformation: `ACTIVATION_FN` becomes a **compile-time
constant** bound directly to the `_swiglu_fn` JIT function object. It is no
longer a runtime parameter — it is hardcoded into this specialized variant of the
kernel.

**3e. Inject tuple reconstruction lines.**

These lines reconstruct the tuples that the kernel body references
(`specialize.py:123-124`):

```python
    activation_fn_args = (alpha, limit,)
    epilogue_fn_args = ()
```

The kernel body — which references `activation_fn_args` as a single tuple — still
works unchanged. But `alpha` and `limit` are now individual kernel parameters
that Triton can see and track independently for autotuning and caching.

**3f. Assemble the new source and create a `JITFunction`.**

```python
# specialize.py:126
new_src = "\n".join(["@triton.jit", new_signature] + constexpr_lines + tuple_lines + body_lines)
```

The assembled source for the specialized kernel looks like:

```python
@triton.jit
def _matmul(..., alpha, limit, ACTIVATION_REDUCTION_N: tl.constexpr, ...):
    ACTIVATION_FN: tl.constexpr = _swiglu_fn     # injected
    EPILOGUE_FN: tl.constexpr = None              # injected
    activation_fn_args = (alpha, limit,)          # injected
    epilogue_fn_args = ()                         # injected
    # ... original kernel body unchanged below ...
```

This is passed to `define_kernel()` (`specialize.py:31-64`) which:

1. Creates a new function via `exec()` of the source string.
2. Injects `_swiglu_fn` into the function's `__globals__` dict so the constexpr
   assignment `ACTIVATION_FN: tl.constexpr = _swiglu_fn` can resolve the name at
   compile time.
3. Wraps the result in `triton.JITFunction(f, **attrs)` — a proper Triton JIT
   kernel ready to be launched.

**3g. Compose the repr for cache keying.**

The specialized kernel gets a repr that includes the activation's repr
(`specialize.py:140-148`). For SwiGLU, `_swiglu_fn` has `repr=lambda _:
"_swiglu"`, so the final cache key includes `..._swiglu`, giving distinct Triton
compilation cache entries per activation variant.

**3h. Cache the result.** Both specialized `_matmul` and `_p_matmul` are stored
on the synthetic module and cached under `("dflt", "swiglu")`.

### Phase 4: Kernel launch (runtime)

At `matmul.py:492-516`, the **specialized** kernel is called:

```python
kernels._matmul[(grid,)](
    ...,
    *matmul_fused_activation.fn_args,                    # unpacks to: 1.1, 1.4
    matmul_fused_activation.specs.reduction_n,           # 2
    ...
)
```

`matmul_fused_activation.fn_args` is `(1.1, 1.4)` — the concrete alpha and limit
values. These land on the `alpha` and `limit` **positional parameters** that the
specialization step spliced into the signature.

Notice what is **not** passed at launch time: `ACTIVATION_FN`. It was removed
from the parameter list and hardcoded as a constexpr inside the function body.
Triton never sees it as a runtime argument.

### Phase 5: Inside the kernel (Triton compilation + execution)

When Triton compiles the specialized kernel, it sees:

```python
    ACTIVATION_FN: tl.constexpr = _swiglu_fn   # known at compile time
    activation_fn_args = (alpha, limit,)         # reconstructed from runtime args
    ...
    if ACTIVATION_FN is not None:                # resolved at compile time → True
        out = ACTIVATION_FN(acc, *activation_fn_args)
```

Because `ACTIVATION_FN` is `tl.constexpr`:

1. **Dead code elimination**: The `if ACTIVATION_FN is not None` branch is
   resolved at compile time to `True`. The else branch (the no-activation path
   with `tl.static_assert(ACTIVATION_REDUCTION_N == 1, ...)`) is eliminated
   entirely from the compiled code.

2. **Direct call + inlining**: `ACTIVATION_FN(acc, *activation_fn_args)` becomes
   `_swiglu_fn(acc, alpha, limit)`. Since `_swiglu_fn` is itself a `@triton.jit`
   function, Triton **inlines its body** into the caller. There is no function
   pointer, no indirect call, no dispatch overhead in the generated PTX/SASS.

3. **Transitive inlining**: `_swiglu_fn` calls `compute_swiglu`, which is also
   `@triton.jit`. This too is inlined, producing a single straight-line block of
   GPU instructions operating on registers:

```python
# What Triton effectively compiles (after all inlining):
gelu, linear = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2)))
gelu = gelu.to(tl.float32)          # already f32 from accumulator
linear = linear.to(tl.float32)
if limit is not None:
    gelu = tl.minimum(gelu, limit)
    linear = tl.clamp(linear, -limit, limit)
s = gelu / (1 + exp_ftz(-alpha * gelu))
out = tl.fma(s, linear, s)          # s * (linear + 1)
```

### Visual summary

```
 import time                    first call to matmul()             kernel launch
 ───────────                    ──────────────────────             ─────────────
 SpecializationModule           .get(activation=swiglu_specs)      specialized._matmul[(grid,)](
   template: _matmul               │                                 ...,
   slots:                          ├─ cache key = ("dflt","swiglu")  alpha=1.1,    ← runtime arg
     ACTIVATION_FN                 │   miss → build                  limit=1.4,    ← runtime arg
     activation_fn_args            │                                 reduction_n=2 ← runtime arg
                                   ├─ specialize(_matmul, ...)       ...
                                   │    ├─ inspect.getsource()     )
                                   │    ├─ parse signature
                                   │    ├─ REMOVE: ACTIVATION_FN param
                                   │    ├─ REPLACE: activation_fn_args → alpha, limit
                                   │    ├─ INJECT constexpr:
                                   │    │    ACTIVATION_FN = _swiglu_fn  ← compile-time const
                                   │    ├─ INJECT tuple rebuild:
                                   │    │    activation_fn_args = (alpha, limit,)
                                   │    ├─ exec() new source
                                   │    ├─ inject _swiglu_fn into __globals__
                                   │    └─ wrap in JITFunction → specialized kernel
                                   │
                                   └─ cache under ("dflt","swiglu")

                      Triton compiler sees:
                      ┌─────────────────────────────────────────────────────┐
                      │ ACTIVATION_FN: tl.constexpr = _swiglu_fn           │
                      │ activation_fn_args = (alpha, limit,)               │
                      │ ...                                                 │
                      │ if ACTIVATION_FN is not None:    # → True (DCE)    │
                      │     out = ACTIVATION_FN(acc, *activation_fn_args)  │
                      │           ↓ inlined ↓                              │
                      │     gelu, linear = tl.split(tl.reshape(acc, ...))  │
                      │     s = gelu / (1 + exp_ftz(-alpha * gelu))        │
                      │     out = fma(s, linear, s)                        │
                      └─────────────────────────────────────────────────────┘
```

### Why this design instead of passing `_swiglu_fn` as a `tl.constexpr` argument?

Passing the function directly as a constexpr kernel argument would work for a
single activation, but the specialization approach provides several advantages:

1. **Caching by name** — the module key is `("dflt", "swiglu")`, not the
   function object identity. Different calls with the same spec name reuse the
   same compiled kernel.

2. **Argument expansion** — the tuple `activation_fn_args` is unpacked into
   individual named parameters (`alpha`, `limit`). Triton's autotuning cache
   keys on individual args; a tuple would be opaque to it.

3. **Composability** — the same mechanism handles both `epilogue` and
   `activation` slots independently. You can mix-and-match any epilogue with any
   activation without a combinatorial explosion of manually written kernels.

4. **`repr` composition** — the specialized kernel's repr includes the activation
   function's repr, giving distinct cache keys in Triton's compilation cache
   (e.g., `_matmul_..._swiglu`).

5. **Source-level transparency** — the rewritten source is a valid `@triton.jit`
   function that can be inspected, debugged, and has correct line-number
   metadata traced back to the original template kernel
   (`specialize.py:156-165`).

---

## 8. Summary

| Aspect | Detail |
|---|---|
| **Activation** | SwiGLU: `swish(x_gate) * (x_value + 1)` |
| **Reduction factor** | 2x — output N is half of accumulator N |
| **Accumulator layout** | Row-major tile `[BLOCK_M, BLOCK_N]` with interleaved gate/value columns |
| **Weight layout** | `W[:, 0::2]` = gate weights, `W[:, 1::2]` = value weights |
| **In-kernel operation** | `tl.reshape` + `tl.split` (zero-cost register reinterpret) then `compute_swiglu` |
| **Split-K** | Activation deferred to reduce kernel; matmul writes full-width partials |
| **Memory saving** | Avoids writing full `[M, N]` intermediate to global memory |
| **Specialization** | `_swiglu_fn` is inlined at Triton compile time via `SpecializationModule` |
