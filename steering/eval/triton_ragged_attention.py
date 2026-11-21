"""Fused one-token ragged KV-cache attention for pre-Hopper CUDA."""

from __future__ import annotations

import math
from typing import Any, cast

import torch as t
import triton
import triton.language as tl


@triton.jit
def _update_kv_kernel(
    k_cache,
    v_cache,
    k,
    v,
    cache_positions,
    stride_cache_b: tl.constexpr,
    stride_cache_s: tl.constexpr,
    stride_cache_h: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_h: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offsets_d = tl.arange(0, block_d)
    position = tl.load(cache_positions + batch)
    cache_offsets = (
        batch * stride_cache_b
        + position * stride_cache_s
        + head * stride_cache_h
        + offsets_d
    )
    source_offsets = batch * stride_k_b + head * stride_k_h + offsets_d
    mask = offsets_d < head_dim
    tl.store(k_cache + cache_offsets, tl.load(k + source_offsets, mask=mask), mask=mask)
    tl.store(v_cache + cache_offsets, tl.load(v + source_offsets, mask=mask), mask=mask)


@triton.jit
def _ragged_decode_attention_kernel(
    output,
    q,
    k_cache,
    v_cache,
    attn_mask,
    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_cache_b: tl.constexpr,
    stride_cache_s: tl.constexpr,
    stride_cache_h: tl.constexpr,
    stride_mask_b,
    stride_out_b: tl.constexpr,
    stride_out_h: tl.constexpr,
    cache_end,
    q_per_kv: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // q_per_kv
    offsets_d = tl.arange(0, block_d)
    q_values = tl.load(
        q + batch * stride_q_b + q_head * stride_q_h + offsets_d,
        mask=offsets_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_d,), dtype=tl.float32)
    for start_n in range(0, cache_end, block_n):
        offsets_n = start_n + tl.arange(0, block_n)
        valid_n = offsets_n < cache_end
        valid_n &= tl.load(
            attn_mask + batch * stride_mask_b + offsets_n,
            mask=valid_n,
            other=0,
        ).to(tl.int1)
        cache_offsets = (
            batch * stride_cache_b
            + offsets_n[:, None] * stride_cache_s
            + kv_head * stride_cache_h
            + offsets_d[None, :]
        )
        k_values = tl.load(
            k_cache + cache_offsets,
            mask=valid_n[:, None] & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k_values * q_values[None, :], axis=1) * scale
        scores = tl.where(valid_n, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        # A sliding window can mask entire leading tiles. Until the first
        # valid key, both maxima are -inf: subtracting them would poison the
        # accumulator with NaNs. Use a finite subtraction origin for those
        # empty tiles while retaining -inf as the running maximum.
        softmax_origin = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_scale = tl.exp(running_max - softmax_origin)
        probabilities = tl.exp(scores - softmax_origin)
        block_sum = tl.sum(probabilities, axis=0)
        v_values = tl.load(
            v_cache + cache_offsets,
            mask=valid_n[:, None] & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * v_values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + block_sum
        running_max = next_max

    result = accumulator / running_sum
    tl.store(
        output + batch * stride_out_b + q_head * stride_out_h + offsets_d,
        result,
        mask=offsets_d < head_dim,
    )


def triton_ragged_attention_with_kvcache(
    q: t.Tensor,
    k_cache: t.Tensor,
    v_cache: t.Tensor,
    k: t.Tensor,
    v: t.Tensor,
    cache_positions: t.Tensor,
    attn_mask: t.Tensor,
    cache_end: int,
) -> t.Tensor:
    """Update K/V and attend over one masked cache prefix per batch row."""
    batch_size, width, q_heads, head_dim = q.shape
    if width != 1:
        raise ValueError(f"Triton ragged attention requires width 1, got {width}")
    kv_heads = k_cache.size(2)
    if q_heads % kv_heads:
        raise ValueError(
            f"query heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
        )

    compute_dtype = (
        k_cache.dtype
        if k_cache.dtype in (t.bfloat16, t.float16)
        else t.bfloat16
    )
    q_compute = q[:, 0].to(compute_dtype).contiguous()
    k_compute = k[:, 0].to(compute_dtype).contiguous()
    v_compute = v[:, 0].to(compute_dtype).contiguous()
    block_d = triton.next_power_of_2(head_dim)
    update_kv_kernel = cast(Any, _update_kv_kernel)
    update_kv_kernel[(batch_size, kv_heads)](
        k_cache,
        v_cache,
        k_compute,
        v_compute,
        cache_positions,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_compute.stride(0),
        k_compute.stride(1),
        head_dim,
        block_d,
        num_warps=4,
    )
    output = t.empty_like(q_compute)
    attention_kernel = cast(Any, _ragged_decode_attention_kernel)
    attention_kernel[(batch_size, q_heads)](
        output,
        q_compute,
        k_cache,
        v_cache,
        attn_mask,
        q_compute.stride(0),
        q_compute.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        attn_mask.stride(0),
        output.stride(0),
        output.stride(1),
        cache_end,
        q_heads // kv_heads,
        head_dim,
        1.0 / math.sqrt(head_dim),
        block_n=64,
        block_d=block_d,
        num_warps=4,
    )
    return output.unsqueeze(1)


__all__ = ["triton_ragged_attention_with_kvcache"]
