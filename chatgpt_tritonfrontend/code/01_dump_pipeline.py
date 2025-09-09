
import os, tempfile, pathlib, shutil, torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

if __name__ == "__main__":
    dumpdir = pathlib.Path(tempfile.mkdtemp(prefix="triton_dumps_"))
    os.environ["MLIR_ENABLE_DUMP"] = "1"         # dump before each MLIR pass
    os.environ["MLIR_DUMP_PATH"] = str(dumpdir)  # send dumps to files
    os.environ["LLVM_IR_ENABLE_DUMP"] = "1"      # dump LLVM passes too

    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = ((n + 255) // 256,)
    add_kernel[grid](x, y, out, n, BLOCK=256)
    print("OK. IR/LLVM dumps are under:", dumpdir)
