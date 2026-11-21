from __future__ import annotations

import sys

import pytest
import torch as t
import torch.nn.functional as F

if sys.platform == "linux":
    from steering.eval.triton_ragged_attention import (
        triton_ragged_attention_with_kvcache,
    )
else:
    triton_ragged_attention_with_kvcache = None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "window_left", "cache_dtype"),
    [
        (4, 4, -1, t.bfloat16),
        (4, 2, 7, t.bfloat16),
        (4, 4, -1, t.float8_e4m3fn),
    ],
)
def test_triton_ragged_attention_matches_sdpa_and_cache_updates(
    q_heads: int,
    kv_heads: int,
    window_left: int,
    cache_dtype: t.dtype,
) -> None:
    assert triton_ragged_attention_with_kvcache is not None
    device = t.device("cuda")
    if cache_dtype == t.float8_e4m3fn and t.cuda.get_device_capability(device) != (8, 9):
        pytest.skip("E4M3 ragged cache is enabled only on measured Ada CUDA")
    batch_size, cache_end, head_dim = 4, 37, 64
    generator = t.Generator(device=device).manual_seed(1234)
    q = t.randn(
        batch_size,
        1,
        q_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=t.bfloat16,
    )
    k = t.randn(
        batch_size,
        1,
        kv_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=t.bfloat16,
    )
    v = t.randn_like(k)
    cache_positions = t.tensor([5, 12, 21, 30], device=device, dtype=t.long)
    key_positions = t.arange(cache_end, device=device)
    attn_mask = key_positions.unsqueeze(0) <= cache_positions.unsqueeze(1)
    if window_left >= 0:
        attn_mask &= key_positions.unsqueeze(0) >= (cache_positions - window_left).unsqueeze(1)
    base_k = t.randn(
        batch_size,
        cache_end,
        kv_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=t.bfloat16,
    )
    base_v = t.randn_like(base_k)
    base_k, base_v = base_k.to(cache_dtype), base_v.to(cache_dtype)
    reference_k = base_k.clone()
    reference_v = base_v.clone()
    batch_indices = t.arange(batch_size, device=device)
    reference_k[batch_indices, cache_positions] = k[:, 0].to(cache_dtype)
    reference_v[batch_indices, cache_positions] = v[:, 0].to(cache_dtype)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        reference_k.to(t.bfloat16).transpose(1, 2),
        reference_v.to(t.bfloat16).transpose(1, 2),
        attn_mask=attn_mask[:, None, None],
        enable_gqa=q_heads != kv_heads,
    ).transpose(1, 2)

    candidate_k = base_k.clone()
    candidate_v = base_v.clone()
    candidate = triton_ragged_attention_with_kvcache(
        q,
        candidate_k,
        candidate_v,
        k,
        v,
        cache_positions,
        attn_mask,
        cache_end,
    )
    t.cuda.synchronize()

    assert t.equal(candidate_k, reference_k)
    assert t.equal(candidate_v, reference_v)
    tolerance = (0.01, 0.05) if cache_dtype == t.float8_e4m3fn else (0.002, 0.02)
    t.testing.assert_close(candidate, reference, atol=tolerance[0], rtol=tolerance[1])


@pytest.mark.parametrize("cache_dtype", [t.bfloat16, t.float8_e4m3fn])
def test_d14_sliding_window_crosses_fully_masked_tile(cache_dtype: t.dtype) -> None:
    """Cover the D14 7x128 geometry on both sides of cache position 576."""
    assert triton_ragged_attention_with_kvcache is not None
    device = t.device("cuda")
    if cache_dtype == t.float8_e4m3fn and t.cuda.get_device_capability(device) != (8, 9):
        pytest.skip("E4M3 ragged cache is enabled only on measured Ada CUDA")
    t.manual_seed(42)
    batch, heads, head_dim, cache_end = 4, 7, 128, 1001
    q = t.randn(batch, 1, heads, head_dim, device=device, dtype=t.bfloat16)
    k, v = t.randn_like(q), t.randn_like(q)
    positions = t.tensor([575, 576, 640, 990], device=device)
    keys = t.arange(cache_end, device=device)[None, :]
    mask = (keys <= positions[:, None]) & (keys >= positions[:, None] - 512)
    base_k = t.randn(batch, cache_end, heads, head_dim, device=device, dtype=t.bfloat16)
    base_v = t.randn_like(base_k)
    # Poison masked history and the future tail before converting to cache dtype.
    base_k.masked_fill_(~mask[:, :, None, None], t.nan)
    base_v.masked_fill_(~mask[:, :, None, None], t.nan)
    candidate_k, candidate_v = base_k.to(cache_dtype), base_v.to(cache_dtype)
    rows = t.arange(batch, device=device)
    expected_k, expected_v = candidate_k.clone(), candidate_v.clone()
    expected_k[rows, positions] = k[:, 0].to(cache_dtype)
    expected_v[rows, positions] = v[:, 0].to(cache_dtype)
    # Dense SDPA can multiply masked NaNs; its oracle uses finite masked storage.
    reference_k = expected_k.to(t.bfloat16).masked_fill(~mask[:, :, None, None], 0)
    reference_v = expected_v.to(t.bfloat16).masked_fill(~mask[:, :, None, None], 0)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        reference_k.transpose(1, 2),
        reference_v.transpose(1, 2),
        attn_mask=mask[:, None, None, :],
    ).transpose(1, 2)
    actual = triton_ragged_attention_with_kvcache(
        q,
        candidate_k,
        candidate_v,
        k,
        v,
        positions,
        mask,
        cache_end,
    )
    assert t.isfinite(actual).all()
    t.testing.assert_close(actual, reference, atol=0.002, rtol=0.02)
    t.testing.assert_close(candidate_k.to(t.bfloat16), expected_k.to(t.bfloat16), rtol=0, atol=0, equal_nan=True)
    t.testing.assert_close(candidate_v.to(t.bfloat16), expected_v.to(t.bfloat16), rtol=0, atol=0, equal_nan=True)
