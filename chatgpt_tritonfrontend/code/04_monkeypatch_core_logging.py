
import functools, triton.language as tl
from triton.language import core as core

def wrap(name):
    fn = getattr(core, name)
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        print(f"[core.{name}] args={args} kwargs={kwargs}")
        return fn(*args, **kwargs)
    setattr(core, name, _wrapped)

# Patch a few primitives
for _n in ["load", "store", "where"]:
    if hasattr(core, _n):
        wrap(_n)

# A toy kernel that exercises these core ops
import torch, triton

@triton.jit
def toy(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
    i = tl.arange(0, n)
    x = tl.load(x_ptr + i)
    y = tl.load(y_ptr + i)
    tl.store(out_ptr + i, tl.where(x > y, x, y))

if __name__ == "__main__":
    n = 8
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.arange(n, 0, -1, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    toy[(1,)](x, y, out, n)
    print("out =", out)
