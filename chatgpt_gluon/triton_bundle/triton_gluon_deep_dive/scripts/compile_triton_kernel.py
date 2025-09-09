
#!/usr/bin/env python3
import torch, triton, triton.language as tl
@triton.jit
def vadd_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.)
    y = tl.load(y_ptr + offs, mask=m, other=0.)
    tl.store(out_ptr + offs, x + y, mask=m)
if __name__ == "__main__":
    n = 4096
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    vadd_kernel[((n+1023)//1024,)](x,y,out,n,BLOCK=1024)
    print("classic triton compiled/run; check MLIR dumps")
