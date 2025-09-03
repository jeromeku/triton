#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def tma_gather_rows_kernel(out_ptr, in_ptr, idx_ptr, y,
                           X: tl.constexpr, Y: tl.constexpr,
                           BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    out = desc.gather(idx, y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)


def main():
    p = argparse.ArgumentParser(description="Compile-only dump of TTIR/TTGIR/LLIR/PTX for tma_gather_rows_kernel")
    p.add_argument("--out-dir", type=Path, default=Path("tma_gather_exploration/out"), help="Output directory for artifacts")
    p.add_argument("--arch", type=str, default="sm_100", help="CUDA arch to target (e.g., sm_100)")
    p.add_argument("--cc", type=int, default=100, help="Compute capability as integer (e.g., 100 for Blackwell)")
    p.add_argument("--warp", type=int, default=32, help="Threads per warp (typically 32)")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16", "int8"], help="Element dtype of input/output")
    p.add_argument("--X", type=int, default=128, help="Number of rows in input tensor")
    p.add_argument("--Y", type=int, default=128, help="Number of cols in input tensor")
    p.add_argument("--block-x", type=int, default=32, dest="block_x", help="Rows gathered per invocation (must be <= X)")
    p.add_argument("--block-y", type=int, default=32, dest="block_y", help="Cols gathered per invocation (must be <= Y)")
    p.add_argument("--y-offset", type=int, default=0, dest="y_offset", help="Column start offset (0-based)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build signature for compile-only
    sig = {
        "out_ptr": f"*{args.dtype}",
        "in_ptr": f"*{args.dtype}",
        "idx_ptr": "*i32",
        "y": "i32",
    }

    constexprs = {
        "X": args.X,
        "Y": args.Y,
        "BLOCK_X": args.block_x,
        "BLOCK_Y": args.block_y,
    }

    # Target a specific architecture regardless of host (compile-only)
    target = GPUTarget("cuda", args.cc, args.warp)

    print(f"Compiling for CUDA cc={args.cc} ({args.arch}), warp={args.warp} ...")
    h = triton.compile(ASTSource(fn=tma_gather_rows_kernel, signature=sig, constexprs=constexprs), target=target)

    # Dump all available artifacts
    for key in ["ttir", "ttgir", "llir", "ptx"]:
        text = h.asm.get(key)
        if text:
            # Some entries may be bytes; coerce to str
            if isinstance(text, bytes):
                try:
                    text = text.decode("utf-8", errors="ignore")
                except Exception:
                    pass
            out_path = args.out_dir / f"tma_gather_rows.{key}"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(str(text))
            print(f"  wrote {out_path}")

    cubin = h.asm.get("cubin")
    if cubin:
        out_path = args.out_dir / "tma_gather_rows.cubin"
        with open(out_path, "wb") as f:
            f.write(cubin)
        print(f"  wrote {out_path}")

    # Quick sanity message
    ttgir = h.asm.get("ttgir", "")
    if "ttng.async_tma_gather" not in str(ttgir):
        print("[warn] did not find ttng.async_tma_gather in TTGIR; check your block shape and arch")
    else:
        print("[ok] TTGIR contains ttng.async_tma_gather")

    ptx = h.asm.get("ptx", "")
    if "tile::gather4" in str(ptx):
        print("[ok] PTX contains tile::gather4")
    else:
        print("[warn] PTX does not contain tile::gather4; verify arch=sm_100 and args")


if __name__ == "__main__":
    # Allow overriding arch through env for JIT parity/debug; compile-only uses GPUTarget
    os.environ.setdefault("TRITON_OVERRIDE_ARCH", "sm100")
    main()

