#!/usr/bin/env python3
import argparse
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


def compute_launch_config(md, grid):
    gx, gy, gz = grid
    # Apply cluster dims expansion if using clustered CTAs
    if getattr(md, 'num_ctas', 1) != 1:
        cd = getattr(md, 'cluster_dims', (1, 1, 1))
        gx *= cd[0]
        gy *= cd[1]
        gz *= cd[2]
    block_x = 32 * md.num_warps
    block = (block_x, 1, 1)
    shared = md.shared
    attrs = []
    if getattr(md, 'launch_pdl', False):
        attrs.append('PROGRAMMATIC_STREAM_SERIALIZATION')
    if getattr(md, 'launch_cooperative_grid', False):
        attrs.append('COOPERATIVE')
    if getattr(md, 'num_ctas', 1) != 1:
        attrs.append(f'CLUSTER_DIM({md.cluster_dims[0]},{md.cluster_dims[1]},{md.cluster_dims[2]})')
        attrs.append('CLUSTER_SCHEDULING=SPREAD')
    return (gx, gy, gz), block, shared, attrs


def main():
    ap = argparse.ArgumentParser(description='Compute CUDA launch config from Triton metadata (no kernel launch).')
    ap.add_argument('--arch', default='sm_100', help='CUDA arch to target, e.g. sm_100')
    ap.add_argument('--cc', type=int, default=100, help='Compute capability integer, e.g. 100')
    ap.add_argument('--warp', type=int, default=32)
    ap.add_argument('--X', type=int, default=128)
    ap.add_argument('--Y', type=int, default=128)
    ap.add_argument('--block-x', type=int, default=32)
    ap.add_argument('--block-y', type=int, default=32)
    ap.add_argument('--y-offset', type=int, default=0)
    ap.add_argument('--grid', type=int, nargs=3, default=(1, 1, 1), metavar=('GX', 'GY', 'GZ'))
    ap.add_argument('--out', type=Path, default=Path('tma_gather_exploration/out/launch_inspect.txt'))
    args = ap.parse_args()

    # Compile-only to get metadata (and artifacts if ptxas is available)
    target = GPUTarget('cuda', args.cc, args.warp)
    src = ASTSource(
        fn=tma_gather_rows_kernel,
        signature={
            'out_ptr': '*fp32',
            'in_ptr': '*fp32',
            'idx_ptr': '*i32',
            'y': 'i32',
        },
        constexprs={'X': args.X, 'Y': args.Y, 'BLOCK_X': args.block_x, 'BLOCK_Y': args.block_y},
    )
    ck = triton.compile(src, target=target)
    md = ck.metadata

    grid_eff, block, shared, attrs = compute_launch_config(md, tuple(args.grid))

    # Dump human-readable summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        print('# Kernel metadata', file=f)
        print(f'name={md.name}', file=f)
        print(f'num_warps={md.num_warps} num_ctas={md.num_ctas} cluster_dims={md.cluster_dims}', file=f)
        print(f'shared={md.shared}B global_scratch={getattr(md, "global_scratch_size", 0)}B align={getattr(md, "global_scratch_align", 1)}', file=f)
        print(f'profile_scratch={getattr(md, "profile_scratch_size", 0)}B align={getattr(md, "profile_scratch_align", 1)}', file=f)
        print(f'launch_cooperative_grid={getattr(md, "launch_cooperative_grid", False)} launch_pdl={getattr(md, "launch_pdl", False)}', file=f)
        print('', file=f)
        print('# Computed CUlaunchConfig (no launch performed)', file=f)
        print(f'grid={grid_eff} block={block} sharedMemBytes={shared}', file=f)
        print(f'attrs={attrs}', file=f)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

