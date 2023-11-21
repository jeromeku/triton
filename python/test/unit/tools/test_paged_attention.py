import itertools
import operator
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
import torch

import triton
import triton.language as tl


def generate_medusa_attn_mask(medusa_choices, device="cuda"):
    """
    Generate buffers related to the Medusa structure.

    This function generates various buffers used in the Medusa structure, which is a complex data structure consisting of a tree-like hierarchy. The buffers include indices, attention masks, position identifiers, and more.

    Args:
        medusa_choices (torch.Tensor): A tensor containing choices for the Medusa structure.
        context_len: int for context lengths
        dtype: data type of the mask tensor
        device (str, optional): Target device for the generated buffers. Defaults to "cuda".

    Returns:
        The attention mask designed specifically for the Medusa structure, ensuring proper attention computation.
    """
    medusa_choices = torch.tensor(medusa_choices)
    cumulative_product = torch.cumprod(medusa_choices, dim=0)
    medusa_len = cumulative_product.sum().item()
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)

    # 2. Update the Medusa attention mask
    prev_cumprod_sum = -1
    for i in range(medusa_choices.size(0)):
        cumprod_sum = cumulative_product[:i].sum().item()
        if prev_cumprod_sum != -1:
            parent_idx = (
                torch.arange(prev_cumprod_sum, cumprod_sum)
                .repeat(medusa_choices[i], 1)
                .transpose(0, 1)
                .flatten()
            )
            medusa_attn_mask[
                cumprod_sum : cumprod_sum + parent_idx.size(0)
            ] += medusa_attn_mask[parent_idx]
        prev_cumprod_sum = cumulative_product[:i].sum().item()
    return medusa_attn_mask.to(device)


@dataclass
class DataClassDict:
    def __post_init__(self):
        self.update(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__.values())


@dataclass(kw_only=True)
class PagedFlashAttentionStrideArgs(DataClassDict):
    stride_ob: int
    stride_os: int
    stride_oh: int
    stride_od: int
    stride_qb: int
    stride_qs: int
    stride_qh: int
    stride_qd: int
    stride_kb: int
    stride_kh: int
    stride_kxd: int
    stride_kbs: int
    stride_kx: int
    stride_vb: int
    stride_vh: int
    stride_vd: int
    stride_vbs: int
    stride_bts: int
    stride_btb: int
    stride_mm_row: int
    stride_mm_col: int


@dataclass(kw_only=True)
class PagedFlashAttentionConstants(DataClassDict):
    BLOCK_M: tl.constexpr
    BLOCK_DMODEL: tl.constexpr
    BLOCK_N: tl.constexpr
    BLOCK_SIZE: tl.constexpr
    PAGES_PER_BLOCK_N: tl.constexpr
    BLOCK_DKEYCACHE: tl.constexpr
    X: tl.constexpr


@dataclass(kw_only=True)
class PagedFlashAttentionCacheArgs(DataClassDict):
    key_cache: torch.Tensor  # [num_blocks, num_heads, head_size // x, block_size, x]
    value_cache: torch.Tensor  # [num_blocks, num_heads, head_size, block_size]
    head_mapping: torch.Tensor  # [num_heads]
    block_tables: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor  # [num_seqs]


@dataclass(kw_only=True)
class PagedFlashAttentionIndexArgs(DataClassDict):
    key_block_idx: torch.Tensor
    key_head_idx: torch.Tensor
    value_block_idx: torch.Tensor


@dataclass(kw_only=True)
class PagedFlashAttentionInputOutputArgs(DataClassDict):
    output: torch.Tensor
    query: torch.Tensor  # [num_seqs, num_candidates, num_heads, head_dim]


@dataclass(kw_only=True)
class PagedFlashAttentionMedusaArgs(DataClassDict):
    medusa_attn_mask: torch.Tensor
    scale: float
    num_candidates: int


@dataclass(kw_only=True)
class PagedFlashAttentionArgs(DataClassDict):
    input_output_args: PagedFlashAttentionInputOutputArgs
    cache_args: PagedFlashAttentionCacheArgs
    medusa_args: PagedFlashAttentionMedusaArgs
    stride_args: PagedFlashAttentionStrideArgs
    index_args: PagedFlashAttentionIndexArgs
    constants: PagedFlashAttentionConstants

    def __post_init__(self):
        super().__post_init__(self)

        full_args = OrderedDict(
            **self.input_output_args,
            **self.cache_args,
            **self.medusa_args,
            **self.stride_args,
            **self.index_args,
            **self.constants,
        )
        self.__dict__.update(full_args)

        # self.__dict__.pop("cache_args")
        # self.__dict__.pop("stride_args")
        # self.__dict__.pop("constants")


@dataclass
class SingleQueryArgs(DataClassDict):
    num_sequences: int
    num_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    dtype: torch.dtype
    num_kv_heads: int = None
    medusa_choices: List[int] = None


SINGLE_QUERY_CONFIG = [SingleQueryArgs(7, 40, 16, 128, 10240, torch.float16, [1, 3, 4])]


def construct_paged_fa_args(
    num_sequences,
    num_heads,
    head_size,
    block_size,
    num_blocks,
    dtype,
    num_kv_heads,
    medusa_choices,
    MAX_SEQ_LEN,
    BLOCK_M=16,
    BLOCK_N=128,
):
    medusa_candidates = sum(itertools.accumulate(medusa_choices, operator.mul))
    medusa_attn_mask = generate_medusa_attn_mask(medusa_choices, device="cuda")

    # Input / Output Args
    qkv = torch.empty(
        num_sequences,
        medusa_candidates,
        3,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    qkv.uniform_(-1e-3, 1e-3)

    # query shape: [num_sequences, medusa_candidates, num_heads, head_size]
    query, _, _ = qkv.unbind(dim=2)

    output = torch.empty(
        num_sequences,
        medusa_candidates,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )

    input_output_args = PagedFlashAttentionInputOutputArgs(
        output=output,
        query=query,
    )

    # Cache Args
    X = 16 // torch.tensor([], dtype=query.dtype).element_size()

    key_block_shape = (
        num_heads,
        head_size // X,
        block_size,
        X,
    )

    key_cache = torch.empty(
        size=(num_blocks, *key_block_shape),
        dtype=dtype,
        device="cuda",
    )
    key_cache.uniform_(-1e-3, 1e-3)

    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(
        size=(num_blocks, *value_block_shape), dtype=dtype, device="cuda"
    )
    value_cache.uniform_(-1e-3, 1e-3)

    context_lens = [
        max(random.randint(1, MAX_SEQ_LEN), medusa_candidates + 10)
        for _ in range(num_sequences)
    ]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    cache_args = PagedFlashAttentionCacheArgs(
        key_cache=key_cache,
        value_cache=value_cache,
        head_mapping=head_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
    )

    # Medusa Args
    scale = float(1.0 / (head_size**0.5))
    num_candidates, num_heads, head_dim = query.shape

    medusa_args = PagedFlashAttentionMedusaArgs(
        medusa_attn_mask=medusa_attn_mask,
        scale=scale,
        num_candidates=num_candidates,
    )

    # Stride Args
    # taking medusa candidates into account
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_sequences):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    assert num_heads % num_kv_heads == 0

    num_queries_per_kv = num_heads // num_kv_heads

    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )

    stride_args = PagedFlashAttentionStrideArgs(
        stride_ob=output.stride(0),
        stride_os=output.stride(0),
        stride_oh=output.stride(1),
        stride_od=output.stride(2),
        stride_qb=query.stride(0),
        stride_qs=query.stride(0),
        stride_qh=query.stride(1),
        stride_qd=query.stride(2),
        stride_kb=key_cache.stride(0),
        stride_kh=key_cache.stride(1),
        stride_kxd=key_cache.stride(2),
        stride_kbs=key_cache.stride(3),
        stride_kx=key_cache.stride(4),
        stride_vb=value_cache.stride(0),
        stride_vh=value_cache.stride(1),
        stride_vd=value_cache.stride(2),
        stride_vbs=value_cache.stride(3),
        stride_bts=block_tables.stride(0),
        stride_btb=block_tables.stride(1),
        stride_mm_row=medusa_attn_mask.stride(2),
        stride_mm_col=medusa_attn_mask.stride(3),
    )

    # Index Args
    key_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
    key_head_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
    value_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
    index_args = PagedFlashAttentionIndexArgs(
        key_block_idx=key_block_idx,
        key_head_idx=key_head_idx,
        value_block_idx=value_block_idx,
    )

    # Constants
    assert BLOCK_N % block_size == 0
    constant_args = PagedFlashAttentionConstants(
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=block_size,
        PAGES_PER_BLOCK_N=BLOCK_N // block_size,  # asserted divisible
        BLOCK_DKEYCACHE=head_dim // X,  # asserted divisible
        X=X,
    )

    return PagedFlashAttentionArgs(
        input_output_args=input_output_args,
        cache_args=cache_args,
        medusa_args=medusa_args,
        stride_args=stride_args,
        index_args=index_args,
        constants=constant_args,
    )


@pytest.mark.parametrize("query_config", SINGLE_QUERY_CONFIG)
@pytest.mark.parametrize("MAX_SEQ_LEN", [2048])
def test_paged_attention(query_config, MAX_SEQ_LEN):
    kernel_path = "/notebooks/torch-extensions/extension_builder/triton-aot/python/test/unit/tools/fixtures/kernels/triton-paged-attention/paged_attention.py"
    trace_dir = Path(__file__).parent / kernel_path.name.split(".")[0]

    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    (
        num_sequences,
        num_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        num_kv_heads,
        medusa_choices,
    ) = query_config

    TEST_SEED = 0
    torch.manual_seed(TEST_SEED)
    random.seed(TEST_SEED)

    args = construct_paged_fa_args(
        num_sequences,
        num_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        num_kv_heads,
        medusa_choices,
        MAX_SEQ_LEN,
    )

    num_candidates, num_heads, head_dim = args.query.shape
    batch = 1
    head_dim = args.query.shape[-1]

    grid = (batch, num_heads, triton.cdiv(num_candidates, args.BLOCK_M))
    num_warps = 4 if head_dim <= 64 else 8

    # trace_config = TraceConfig(
    #     trace_dir=trace_dir,
    # )
    # jit_fn = JITFunction(kernel_path)
    # kernel_name = jit_fn.__name__
    # results = jit_fn
    # compiler = AOTCompiler(
    #     kernel_name=kernel_name,
    #     jit_args=args,
    #     jit_fn=jit_fn,
    #     save_dir=codegen_dir,
    # )
    # compiled_result: AOTCompilationResult = compiler.generate()
    # compiled_result: AOTCompilationResult = compiler.generate()
