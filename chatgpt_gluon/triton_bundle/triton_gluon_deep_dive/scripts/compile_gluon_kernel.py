
#!/usr/bin/env python3
import torch, triton, triton.language as tl
try:
    import triton.experimental.gluon.language as gl
except Exception:
    gl = None
@triton.jit
def gluon_demo_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.)
    y = tl.load(y_ptr + offs, mask=m, other=0.)
    if gl is not None and hasattr(gl, "set_auto_layout"):
        x = gl.set_auto_layout(x); y = gl.set_auto_layout(y)
    tl.store(out_ptr + offs, x + y, mask=m)
if __name__ == "__main__":
    n = 4096
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    gluon_demo_kernel[((n+1023)//1024,)](x,y,out,n,BLOCK=1024)
    print("gluon demo compiled/run; check MLIR dumps")
