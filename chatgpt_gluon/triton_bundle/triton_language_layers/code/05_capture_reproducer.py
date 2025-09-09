
import os, tempfile, pathlib, torch, triton, triton.language as tl
@triton.jit
def demo(x_ptr, out_ptr, n: tl.constexpr):
    i = tl.arange(0, n); x = tl.load(x_ptr + i); tl.store(out_ptr + i, x + 1)
if __name__ == "__main__":
    repro = pathlib.Path(tempfile.mkdtemp(prefix="triton_repro_")) / "stage.mlir"
    os.environ["TRITON_REPRODUCER_PATH"] = str(repro)
    x = torch.ones(16, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    demo[(1,)](x, out, 16)
    print("Reproducer path:", repro)
