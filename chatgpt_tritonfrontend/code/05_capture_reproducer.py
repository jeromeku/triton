
import os, tempfile, pathlib, torch, triton, triton.language as tl

@triton.jit
def divzero(x_ptr, out_ptr, n: tl.constexpr):
    i = tl.arange(0, n)
    x = tl.load(x_ptr + i)
    y = x / 0.0  # silly example; goal is to capture a reproducer regardless
    tl.store(out_ptr + i, y)

if __name__ == "__main__":
    repro = pathlib.Path(tempfile.mkdtemp(prefix="triton_repro_")) / "stage.mlir"
    os.environ["TRITON_REPRODUCER_PATH"] = str(repro)
    x = torch.ones(16, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    try:
        divzero[(1,)](x, out, 16)
    except Exception as e:
        print("Kernel raised:", type(e).__name__, e)
    finally:
        print("If a pass failed, a reproducer should be at:", repro)
