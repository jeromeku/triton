# SwiGLU Matmul Fusion

## Overview

The SwiGLU activation function is **fused directly into the matmul kernel**, meaning it is applied immediately after the matrix multiplication computation within the same GPU kernel, without writing intermediate results to memory. This fusion provides significant performance benefits by:

1. **Reducing memory bandwidth**: The intermediate matmul result is never written to global memory
2. **Improving cache locality**: Data stays in registers/shared memory between matmul and activation
3. **Reducing kernel launch overhead**: Single kernel launch instead of two separate kernels

## Code Location

The fusion is configured in the test file at `python/triton_kernels/tests/test_matmul.py`:

```python
# --- create fused activation ---
fused_activation = None
if swiglu_opts is not None:
    fused_activation = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), swiglu_opts)
```

The `fused_activation` is then passed to the `matmul` function, which applies it directly in the Triton kernel after computing the matrix multiplication.

## How It Works

1. **Input**: The matmul kernel computes `Y = X @ W` where:
   - `X` has shape `(M, K)`
   - `W` has shape `(K, N)` where `N` must be even
   - `Y` has shape `(M, N)`

2. **Fusion**: Instead of writing `Y` to memory, the SwiGLU activation is applied immediately:
   - The output dimension is reduced by 2 (`reduction_n=2`)
   - Final output shape: `(M, N // 2)`

3. **SwiGLU Operation**: The activation splits the output into two parts:
   - **GELU part** (even indices): `Y[..., ::2]` - elements at indices 0, 2, 4, ...
   - **Linear part** (odd indices): `Y[..., 1::2]` - elements at indices 1, 3, 5, ...

## SwiGLU Parameters

The `swiglu_opts` parameter is a tuple `(alpha, limit)`:
- **`alpha`**: Scaling factor for the sigmoid in the GELU computation (typically around 1.0-1.7)
- **`limit`**: Clamping limit applied to both GELU and linear parts (typically 1.4-7.0)

## Reference PyTorch Implementation

Here is an equivalent reference PyTorch implementation that shows what the fused operation computes:

```python
import torch

def swiglu_matmul_fused_reference(X, W, alpha, limit):
    """
    Reference implementation of fused matmul + SwiGLU activation.
    
    This is equivalent to what the Triton kernel computes when fused_activation
    is set to SwiGLU.
    
    Args:
        X: Input tensor of shape (M, K)
        W: Weight tensor of shape (K, N) where N must be even
        alpha: Scaling factor for sigmoid in GELU (from swiglu_opts[0])
        limit: Clamping limit (from swiglu_opts[1])
    
    Returns:
        Output tensor of shape (M, N // 2)
    """
    # Step 1: Matrix multiplication
    Y = torch.matmul(X, W)  # Shape: (M, N)
    
    # Step 2: Split into GELU and linear parts
    a_gelu = Y[..., ::2]   # Even indices: shape (M, N // 2)
    a_linear = Y[..., 1::2]  # Odd indices: shape (M, N // 2)
    
    # Step 3: Apply clamping if limit is provided
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
        a_linear = a_linear.clamp(min=-limit, max=limit)
    
    # Step 4: Apply GELU to the even-indexed part
    # GELU(x) = x * sigmoid(alpha * x)
    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    
    # Step 5: Multiply GELU output with (linear + 1)
    out = out_gelu * (a_linear + 1)
    
    return out  # Shape: (M, N // 2)
```

## Example Usage

```python
import torch
from triton_kernels.matmul import matmul, FusedActivation, FnSpecs
from triton_kernels.swiglu import swiglu_fn

# Setup
M, K, N = 1024, 512, 1024  # N must be even
X = torch.randn(M, K, dtype=torch.float16, device='cuda')
W = torch.randn(K, N, dtype=torch.float16, device='cuda')

# SwiGLU options: (alpha, limit)
swiglu_opts = (1.1, 1.4)

# Create fused activation
fused_activation = FusedActivation(
    FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
    swiglu_opts
)

# Fused matmul + SwiGLU (single kernel)
output = matmul(X, W, bias=None, fused_activation=fused_activation)
# Output shape: (M, N // 2) = (1024, 512)

# Equivalent reference implementation
ref_output = swiglu_matmul_fused_reference(X, W, alpha=swiglu_opts[0], limit=swiglu_opts[1])
```

## Performance Benefits

The fusion provides several advantages over separate kernels:

1. **Memory Bandwidth**: Saves one write and one read of the intermediate matmul result
2. **Latency**: Eliminates kernel launch overhead and synchronization between kernels
3. **Cache Efficiency**: Intermediate results stay in fast memory (registers/shared memory)
4. **Throughput**: Better GPU utilization by combining operations

For typical transformer MLP layers, this fusion can provide 20-40% speedup compared to separate matmul and SwiGLU kernels.
