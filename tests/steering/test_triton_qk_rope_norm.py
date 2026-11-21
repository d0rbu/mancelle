from __future__ import annotations

import sys

import pytest
import torch as t

from nanochat.gpt import apply_rotary_emb, norm

if sys.platform == "linux":
    from steering.eval.triton_qk_rope_norm import triton_qk_rope_rms_norm
else:
    triton_qk_rope_rms_norm = None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize(
    ("dtype", "atol", "mean_tolerance"),
    [
        (t.bfloat16, 0.03125, 0.0015),
        (t.float16, 0.003, 0.0002),
    ],
)
def test_triton_qk_rope_rms_norm_matches_strided_torch_views(
    dtype: t.dtype,
    atol: float,
    mean_tolerance: float,
) -> None:
    assert triton_qk_rope_rms_norm is not None
    device = t.device("cuda")
    generator = t.Generator(device=device).manual_seed(29290)
    batch_size = 73
    head_dim = 128
    q_heads = 13
    k_heads = 5
    q_dim = q_heads * head_dim
    k_dim = k_heads * head_dim
    qkv = t.randn(
        batch_size,
        1,
        q_dim + 2 * k_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    q, k, _v = qkv.split((q_dim, k_dim, k_dim), dim=-1)
    q = q.view(batch_size, 1, q_heads, head_dim)
    k = k.view(batch_size, 1, k_heads, head_dim)
    angles = t.randn(
        batch_size,
        1,
        1,
        head_dim // 2,
        generator=generator,
        device=device,
        dtype=t.float32,
    )
    cos = t.cos(angles).to(dtype)
    sin = t.sin(angles).to(dtype)
    reference_q = norm(apply_rotary_emb(q, cos, sin))
    reference_k = norm(apply_rotary_emb(k, cos, sin))

    candidate_q, candidate_k = triton_qk_rope_rms_norm(q, k, cos, sin)
    t.cuda.synchronize()

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert candidate_q.is_contiguous()
    assert candidate_k.is_contiguous()
    t.testing.assert_close(candidate_q, reference_q, rtol=0.02, atol=atol)
    t.testing.assert_close(candidate_k, reference_k, rtol=0.02, atol=atol)
    assert (candidate_q - reference_q).abs().float().mean().item() <= mean_tolerance
    assert (candidate_k - reference_k).abs().float().mean().item() <= mean_tolerance
