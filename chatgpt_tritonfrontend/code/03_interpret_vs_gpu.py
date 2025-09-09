
import os, torch, triton, triton.language as tl

@triton.jit
def relu(x_ptr, y_ptr, n: tl.constexpr):
    pid = tl.program_id(0)
    i = pid + tl.arange(0, n)
    x = tl.load(x_ptr + i)
    tl.store(y_ptr + i, tl.maximum(x, 0))

if __name__ == "__main__":
    n = 8
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    print("Running on GPU...")
    relu[(1,)](x, y, n)
    print("y (gpu) =", y)

    print("\nSwitching to interpreter mode...")
    os.environ["TRITON_INTERPRET"] = "1"
    y_i = torch.empty_like(x, device="cpu")  # interpreter runs on host
    relu[(1,)](x.cpu(), y_i, n)
    print("y (interp) =", y_i)
