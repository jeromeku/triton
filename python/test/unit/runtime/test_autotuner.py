import torch

import triton
import triton.language as tl


def test_kwargs():
    N = 1024
    src = torch.empty(N, device='cuda')
    dst = torch.empty(N, device='cuda')

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'])
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    _kernel[grid](dst, src, N)
    _kernel[grid](dst=dst, src=src, N=N)


def test_restore():
    N = 1024
    src = torch.zeros(N, device='cuda')

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], restore_value=['src'])
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))
