"""Verification tests for the flash attention abstraction layer (SDPA backend).

Tests cover:
  1. Causal correctness
  2. Sliding window (training path)
  3. Sliding window (single-token decode)
  4. Full-context equivalence
  5. KV cache correctness
  6. GQA support
  7. Sliding window + KV cache
"""

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import torch
import torch.nn.functional as F

import pytest

from nanochat.flash_attention import (
    _sdpa_attention,
    flash_attn_func,
    flash_attn_with_kvcache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Reference attention: softmax(Q K^T / sqrt(d) + mask) V.

    q: (B, H_q, Tq, D)
    k: (B, H_kv, Tk, D)
    v: (B, H_kv, Tk, D)
    mask: (Tq, Tk) bool — True = attend, False = masked out
    enable_gqa: if True, broadcast H_kv to H_q
    """
    B, H_q, Tq, D = q.shape
    H_kv = k.shape[1]

    if enable_gqa and H_q != H_kv:
        # repeat KV heads to match Q heads
        repeats = H_q // H_kv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    # (B, H, Tq, D) @ (B, H, D, Tk) -> (B, H, Tq, Tk)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows with 0
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


def causal_mask(Tq: int, Tk: int) -> torch.Tensor:
    """Return a (Tq, Tk) boolean causal mask (True = can attend)."""
    row = torch.arange(Tq).unsqueeze(1)
    col = torch.arange(Tk).unsqueeze(0)
    return col <= row  # token i attends to tokens 0..i


def sliding_causal_mask(Tq: int, Tk: int, window: int) -> torch.Tensor:
    """Return a (Tq, Tk) boolean causal + sliding window mask.

    Token at position i attends to tokens in [max(0, i-window), i].
    """
    row = torch.arange(Tq).unsqueeze(1)
    col = torch.arange(Tk).unsqueeze(0)
    return (col <= row) & ((row - col) <= window)


# ---------------------------------------------------------------------------
# Test 1: SDPA causal correctness
# ---------------------------------------------------------------------------


class TestCausalCorrectness:
    """Verify that causal attention means token i attends only to tokens 0..i."""

    def test_sdpa_causal_matches_naive(self):
        """SDPA causal output matches manually computed reference."""
        torch.manual_seed(42)
        B, H, T, D = 1, 2, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        # SDPA path (window_size=(-1,0) means no sliding window, just causal)
        out_sdpa = _sdpa_attention(q, k, v, window_size=(-1, 0), enable_gqa=False)

        # Naive reference with causal mask
        mask = causal_mask(T, T)
        out_ref = naive_attention(q, k, v, mask=mask)

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_flash_attn_func_causal_matches_naive(self):
        """flash_attn_func with causal=True matches naive reference."""
        torch.manual_seed(123)
        B, H, T, D = 2, 4, 6, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        mask = causal_mask(T, T)
        out_ref = naive_attention(q, k, v, mask=mask)

        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)

    def test_causal_blocks_future_tokens(self):
        """Directly verify that future tokens have zero influence.

        Set V so each token has a unique value. Then output at position 0
        must equal V[0] exactly, position 1 is a weighted mix of V[0:2], etc.
        """
        B, H, T, D = 1, 1, 4, 8
        q = torch.zeros(B, H, T, D)
        k = torch.zeros(B, H, T, D)
        # Distinct V per position: use one-hot style values
        v = torch.zeros(B, H, T, D)
        for t in range(T):
            v[0, 0, t, t] = 1.0

        out = _sdpa_attention(q, k, v, window_size=(-1, 0), enable_gqa=False)

        # Position 0: can only see V[0] -> output has nonzero only in dim 0
        # With uniform attention (Q=K=0), position i attends equally to 0..i
        for i in range(T):
            # Dimensions beyond i must be zero (those correspond to future tokens)
            for j in range(i + 1, T):
                assert out[0, 0, i, j].item() == pytest.approx(0.0, abs=1e-6), (
                    f"Position {i} has nonzero weight on future token {j}'s value"
                )


# ---------------------------------------------------------------------------
# Test 2: Sliding window (training path)
# ---------------------------------------------------------------------------


class TestSlidingWindowTraining:
    """Verify that window_size=(W, 0) limits attention to tokens max(0, i-W)..i."""

    def test_sliding_window_w2(self):
        """With W=2, token i attends to [max(0, i-2), i]."""
        torch.manual_seed(99)
        B, H, T, D = 1, 2, 8, 16
        W = 2
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out_sdpa = _sdpa_attention(q, k, v, window_size=(W, 0), enable_gqa=False)

        mask = sliding_causal_mask(T, T, W)
        out_ref = naive_attention(q, k, v, mask=mask)

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_sliding_window_w1(self):
        """With W=1, each token can only attend to itself and the previous token."""
        torch.manual_seed(77)
        B, H, T, D = 1, 1, 6, 8
        W = 1
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out_sdpa = _sdpa_attention(q, k, v, window_size=(W, 0), enable_gqa=False)

        mask = sliding_causal_mask(T, T, W)
        out_ref = naive_attention(q, k, v, mask=mask)

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_sliding_window_via_flash_attn_func(self):
        """flash_attn_func routes correctly with sliding window."""
        torch.manual_seed(55)
        B, H, T, D = 1, 2, 8, 16
        W = 3
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = flash_attn_func(q, k, v, causal=True, window_size=(W, 0))

        mask = sliding_causal_mask(T, T, W)
        out_ref = naive_attention(q, k, v, mask=mask)

        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)

    def test_sliding_window_blocks_distant_tokens(self):
        """Directly verify that tokens outside the window are not attended."""
        B, H, T, D = 1, 1, 6, 6
        W = 1
        q = torch.zeros(B, H, T, D)
        k = torch.zeros(B, H, T, D)
        v = torch.zeros(B, H, T, D)
        for t in range(T):
            v[0, 0, t, t] = 1.0

        out = _sdpa_attention(q, k, v, window_size=(W, 0), enable_gqa=False)

        # Position 4 with W=1: can attend to [3, 4] only.
        # So output at position 4 should have nonzero only in dims 3 and 4.
        for j in range(3):  # dims 0, 1, 2 correspond to tokens 0, 1, 2
            assert out[0, 0, 4, j].item() == pytest.approx(0.0, abs=1e-6), (
                f"Position 4 (W=1) should not attend to token {j}"
            )


# ---------------------------------------------------------------------------
# Test 3: Sliding window (single-token decode, Tq=1)
# ---------------------------------------------------------------------------


class TestSlidingWindowDecode:
    """Verify that when Tq=1, the KV is correctly sliced to the window."""

    def test_single_token_window_slice(self):
        """With Tq=1 and W=2, Tk=6: should attend to k[:,:,3:,:] (last 3 tokens)."""
        torch.manual_seed(88)
        B, H, D = 1, 2, 16
        Tq, Tk, W = 1, 6, 2
        q = torch.randn(B, H, Tq, D)
        k = torch.randn(B, H, Tk, D)
        v = torch.randn(B, H, Tk, D)

        out_sdpa = _sdpa_attention(q, k, v, window_size=(W, 0), enable_gqa=False)

        # Reference: manually slice to the last (W+1) tokens, no causal needed
        start = max(0, Tk - (W + 1))
        k_sliced = k[:, :, start:, :]
        v_sliced = v[:, :, start:, :]
        out_ref = naive_attention(q, k_sliced, v_sliced, mask=None)

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_single_token_window_larger_than_seq(self):
        """Window larger than Tk: all tokens should be attended."""
        torch.manual_seed(22)
        B, H, D = 1, 2, 16
        Tq, Tk, W = 1, 4, 10
        q = torch.randn(B, H, Tq, D)
        k = torch.randn(B, H, Tk, D)
        v = torch.randn(B, H, Tk, D)

        out_sdpa = _sdpa_attention(q, k, v, window_size=(W, 0), enable_gqa=False)

        # W >= Tk, so no slicing should occur. Attend to all.
        out_ref = naive_attention(q, k, v, mask=None)

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)

    def test_single_token_no_window(self):
        """window=-1 means unlimited: attend to all tokens."""
        torch.manual_seed(33)
        B, H, D = 1, 2, 16
        Tq, Tk = 1, 8
        q = torch.randn(B, H, Tq, D)
        k = torch.randn(B, H, Tk, D)
        v = torch.randn(B, H, Tk, D)

        out_sdpa = _sdpa_attention(q, k, v, window_size=(-1, 0), enable_gqa=False)

        # No masking at all for single token with unlimited window
        out_ref = naive_attention(q, k, v, mask=None)

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: Full context equivalence
# ---------------------------------------------------------------------------


class TestFullContextEquivalence:
    """window_size=(-1,0) and window_size=(seq_len,0) should equal plain causal."""

    def test_neg1_equals_seqlen_window(self):
        """window=(-1,0) and window=(T,0) produce identical output."""
        torch.manual_seed(44)
        B, H, T, D = 1, 2, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out_neg1 = _sdpa_attention(q, k, v, window_size=(-1, 0), enable_gqa=False)
        out_seqlen = _sdpa_attention(q, k, v, window_size=(T, 0), enable_gqa=False)

        torch.testing.assert_close(out_neg1, out_seqlen, atol=1e-6, rtol=1e-6)

    def test_large_window_equals_no_window(self):
        """A window much larger than T produces the same result as no window."""
        torch.manual_seed(55)
        B, H, T, D = 2, 4, 10, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out_no_win = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
        out_large = flash_attn_func(q, k, v, causal=True, window_size=(1000, 0))

        torch.testing.assert_close(out_no_win, out_large, atol=1e-6, rtol=1e-6)

    def test_all_variants_match_naive_causal(self):
        """All full-context variants match the naive causal reference."""
        torch.manual_seed(66)
        B, H, T, D = 1, 2, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        mask = causal_mask(T, T)
        out_ref = naive_attention(q, k, v, mask=mask)

        out_neg1 = _sdpa_attention(q, k, v, window_size=(-1, 0), enable_gqa=False)
        out_T = _sdpa_attention(q, k, v, window_size=(T, 0), enable_gqa=False)
        out_api = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        torch.testing.assert_close(out_neg1, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_T, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_api, out_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: KV cache correctness (SDPA path)
# ---------------------------------------------------------------------------


class TestKVCacheCorrectness:
    """Verify flash_attn_with_kvcache inserts correctly and matches full forward."""

    def test_cache_insertion(self):
        """New K, V are correctly written into the cache at the right position."""
        B, T_max, H_kv, D = 1, 16, 2, 16
        k_cache = torch.zeros(B, T_max, H_kv, D)
        v_cache = torch.zeros(B, T_max, H_kv, D)

        torch.manual_seed(10)
        k_new = torch.randn(B, 4, H_kv, D)
        v_new = torch.randn(B, 4, H_kv, D)
        q = torch.randn(B, 4, H_kv, D)  # H_q == H_kv for simplicity
        cache_seqlens = torch.tensor([0], dtype=torch.int32)

        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(-1, -1),
        )

        # Verify the cache was written at positions 0..3
        torch.testing.assert_close(k_cache[:, 0:4, :, :], k_new)
        torch.testing.assert_close(v_cache[:, 0:4, :, :], v_new)
        # Verify positions 4+ are still zeros
        assert (k_cache[:, 4:, :, :] == 0).all()
        assert (v_cache[:, 4:, :, :] == 0).all()

    def test_prefill_then_decode_matches_full_forward(self):
        """Prefill 4 tokens, decode 1: last-token output matches full 5-token pass."""
        torch.manual_seed(20)
        B, T_max, H, D = 1, 16, 2, 16
        T_prefill = 4

        # Generate all 5 tokens' Q, K, V
        q_all = torch.randn(B, T_prefill + 1, H, D)
        k_all = torch.randn(B, T_prefill + 1, H, D)
        v_all = torch.randn(B, T_prefill + 1, H, D)

        # ---- Full 5-token forward pass (training-style) ----
        # flash_attn_func expects (B, H, T, D)
        q_full = q_all.transpose(1, 2)
        k_full = k_all.transpose(1, 2)
        v_full = v_all.transpose(1, 2)
        out_full = flash_attn_func(
            q_full, k_full, v_full, causal=True, window_size=(-1, -1)
        )
        # Last token output: (B, H, D)
        out_full_last = out_full[:, :, -1, :]  # (B, H, D)

        # ---- KV cache: prefill + decode ----
        k_cache = torch.zeros(B, T_max, H, D)
        v_cache = torch.zeros(B, T_max, H, D)

        # Prefill: insert tokens 0..3
        cache_seqlens = torch.tensor([0], dtype=torch.int32)
        flash_attn_with_kvcache(
            q_all[:, :T_prefill, :, :],
            k_cache,
            v_cache,
            k=k_all[:, :T_prefill, :, :],
            v=v_all[:, :T_prefill, :, :],
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(-1, -1),
        )

        # Decode: insert token 4, attend to all 5
        cache_seqlens = torch.tensor([T_prefill], dtype=torch.int32)
        out_decode = flash_attn_with_kvcache(
            q_all[:, T_prefill : T_prefill + 1, :, :],
            k_cache,
            v_cache,
            k=k_all[:, T_prefill : T_prefill + 1, :, :],
            v=v_all[:, T_prefill : T_prefill + 1, :, :],
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(-1, -1),
        )
        # out_decode: (B, 1, H, D) -> (B, H, D)
        out_decode_last = out_decode.squeeze(1).transpose(0, 1)  # -> (H, B, D)
        # Reshape to match: (B, H, D)
        out_decode_last = out_decode[:, 0, :, :]  # (B, H, D)

        torch.testing.assert_close(out_decode_last, out_full_last, atol=1e-5, rtol=1e-5)

    def test_multi_step_decode(self):
        """Decode multiple steps sequentially and verify each matches full pass."""
        torch.manual_seed(30)
        B, T_max, H, D = 1, 16, 2, 16
        T_total = 6

        q_all = torch.randn(B, T_total, H, D)
        k_all = torch.randn(B, T_total, H, D)
        v_all = torch.randn(B, T_total, H, D)

        # Full forward reference
        q_f = q_all.transpose(1, 2)
        k_f = k_all.transpose(1, 2)
        v_f = v_all.transpose(1, 2)
        out_full = flash_attn_func(q_f, k_f, v_f, causal=True, window_size=(-1, -1))

        # Step-by-step decode
        k_cache = torch.zeros(B, T_max, H, D)
        v_cache = torch.zeros(B, T_max, H, D)

        for t in range(T_total):
            cache_seqlens = torch.tensor([t], dtype=torch.int32)
            out_t = flash_attn_with_kvcache(
                q_all[:, t : t + 1, :, :],
                k_cache,
                v_cache,
                k=k_all[:, t : t + 1, :, :],
                v=v_all[:, t : t + 1, :, :],
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(-1, -1),
            )
            # Compare to full forward at position t
            out_ref_t = out_full[:, :, t, :]  # (B, H, D)
            out_t_squeezed = out_t[:, 0, :, :]  # (B, H, D)
            torch.testing.assert_close(
                out_t_squeezed,
                out_ref_t,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Mismatch at decode step {t}",
            )


# ---------------------------------------------------------------------------
# Test 6: GQA support
# ---------------------------------------------------------------------------


class TestGQASupport:
    """Verify attention correctness when n_head != n_kv_head (grouped query attention)."""

    def test_gqa_basic(self):
        """With H_q=4, H_kv=2, attention produces correct results."""
        torch.manual_seed(50)
        B, T, D = 1, 6, 16
        H_q, H_kv = 4, 2

        q = torch.randn(B, H_q, T, D)
        k = torch.randn(B, H_kv, T, D)
        v = torch.randn(B, H_kv, T, D)

        out = _sdpa_attention(q, k, v, window_size=(-1, 0), enable_gqa=True)

        mask = causal_mask(T, T)
        out_ref = naive_attention(q, k, v, mask=mask, enable_gqa=True)

        assert out.shape == (B, H_q, T, D)
        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)

    def test_gqa_with_sliding_window(self):
        """GQA + sliding window: H_q=4, H_kv=2, W=2."""
        torch.manual_seed(51)
        B, T, D = 1, 8, 16
        H_q, H_kv = 4, 2
        W = 2

        q = torch.randn(B, H_q, T, D)
        k = torch.randn(B, H_kv, T, D)
        v = torch.randn(B, H_kv, T, D)

        out = _sdpa_attention(q, k, v, window_size=(W, 0), enable_gqa=True)

        mask = sliding_causal_mask(T, T, W)
        out_ref = naive_attention(q, k, v, mask=mask, enable_gqa=True)

        assert out.shape == (B, H_q, T, D)
        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)

    def test_gqa_via_flash_attn_func(self):
        """flash_attn_func auto-detects GQA from mismatched Q/K heads."""
        torch.manual_seed(52)
        B, T, D = 2, 8, 16
        H_q, H_kv = 4, 2

        q = torch.randn(B, H_q, T, D)
        k = torch.randn(B, H_kv, T, D)
        v = torch.randn(B, H_kv, T, D)

        out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        mask = causal_mask(T, T)
        out_ref = naive_attention(q, k, v, mask=mask, enable_gqa=True)

        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)

    def test_gqa_kv_cache(self):
        """GQA with KV cache: H_q=4, H_kv=2."""
        torch.manual_seed(53)
        B, T_max, D = 1, 16, 16
        H_q, H_kv = 4, 2
        T_prefill = 4

        q_all = torch.randn(B, T_prefill + 1, H_q, D)
        k_all = torch.randn(B, T_prefill + 1, H_kv, D)
        v_all = torch.randn(B, T_prefill + 1, H_kv, D)

        # Full forward (training layout: B, H, T, D)
        q_f = q_all.transpose(1, 2)
        k_f = k_all.transpose(1, 2)
        v_f = v_all.transpose(1, 2)
        out_full = flash_attn_func(q_f, k_f, v_f, causal=True, window_size=(-1, -1))
        out_full_last = out_full[:, :, -1, :]  # (B, H_q, D)

        # KV cache path
        k_cache = torch.zeros(B, T_max, H_kv, D)
        v_cache = torch.zeros(B, T_max, H_kv, D)

        # Prefill
        cache_seqlens = torch.tensor([0], dtype=torch.int32)
        flash_attn_with_kvcache(
            q_all[:, :T_prefill, :, :],
            k_cache,
            v_cache,
            k=k_all[:, :T_prefill, :, :],
            v=v_all[:, :T_prefill, :, :],
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(-1, -1),
        )

        # Decode step
        cache_seqlens = torch.tensor([T_prefill], dtype=torch.int32)
        out_decode = flash_attn_with_kvcache(
            q_all[:, T_prefill : T_prefill + 1, :, :],
            k_cache,
            v_cache,
            k=k_all[:, T_prefill : T_prefill + 1, :, :],
            v=v_all[:, T_prefill : T_prefill + 1, :, :],
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(-1, -1),
        )
        out_decode_last = out_decode[:, 0, :, :]  # (B, H_q, D)

        torch.testing.assert_close(out_decode_last, out_full_last, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 7: Sliding window + KV cache
# ---------------------------------------------------------------------------


class TestSlidingWindowKVCache:
    """Verify that during decode with a sliding window, old tokens are excluded."""

    def test_window_excludes_old_tokens_in_cache(self):
        """Step-by-step decode with W=2: verify each step matches windowed full pass."""
        torch.manual_seed(60)
        B, T_max, H, D = 1, 16, 2, 16
        T_total = 6
        W = 2

        q_all = torch.randn(B, T_total, H, D)
        k_all = torch.randn(B, T_total, H, D)
        v_all = torch.randn(B, T_total, H, D)

        # Full forward with sliding window (training layout: B, H, T, D)
        q_f = q_all.transpose(1, 2)
        k_f = k_all.transpose(1, 2)
        v_f = v_all.transpose(1, 2)
        out_full = flash_attn_func(q_f, k_f, v_f, causal=True, window_size=(W, 0))

        # Step-by-step decode with KV cache
        k_cache = torch.zeros(B, T_max, H, D)
        v_cache = torch.zeros(B, T_max, H, D)

        for t in range(T_total):
            cache_seqlens = torch.tensor([t], dtype=torch.int32)
            out_t = flash_attn_with_kvcache(
                q_all[:, t : t + 1, :, :],
                k_cache,
                v_cache,
                k=k_all[:, t : t + 1, :, :],
                v=v_all[:, t : t + 1, :, :],
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(W, 0),
            )
            out_ref_t = out_full[:, :, t, :]  # (B, H, D)
            out_t_squeezed = out_t[:, 0, :, :]  # (B, H, D)
            torch.testing.assert_close(
                out_t_squeezed,
                out_ref_t,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Mismatch at decode step {t} with sliding window W={W}",
            )

    def test_window_ignores_distant_cached_tokens(self):
        """After filling cache with 8 tokens, decode with W=2 must NOT attend to tokens 0..4."""
        torch.manual_seed(70)
        B, T_max, H, D = 1, 16, 1, 8
        W = 2

        # Fill cache with 8 tokens
        k_all = torch.randn(B, 8, H, D)
        v_all = torch.randn(B, 8, H, D)
        k_cache = torch.zeros(B, T_max, H, D)
        v_cache = torch.zeros(B, T_max, H, D)
        k_cache[:, :8, :, :] = k_all
        v_cache[:, :8, :, :] = v_all

        # Now decode token 8 (position 8): should attend to [6, 7, 8] only (W=2)
        q_new = torch.randn(B, 1, H, D)
        k_new = torch.randn(B, 1, H, D)
        v_new = torch.randn(B, 1, H, D)

        cache_seqlens = torch.tensor([8], dtype=torch.int32)
        out_cached = flash_attn_with_kvcache(
            q_new,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(W, 0),
        )

        # Reference: manually construct the windowed KV
        # Position 8 with W=2: attend to tokens [6, 7, 8]
        k_windowed = torch.cat([k_all[:, 6:8, :, :], k_new], dim=1)  # (B, 3, H, D)
        v_windowed = torch.cat([v_all[:, 6:8, :, :], v_new], dim=1)
        # No causal masking needed for single-token query
        q_sdpa = q_new.transpose(1, 2)
        k_sdpa = k_windowed.transpose(1, 2)
        v_sdpa = v_windowed.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=False
        )
        out_ref = out_ref.transpose(1, 2)  # back to (B, T, H, D)

        torch.testing.assert_close(out_cached, out_ref, atol=1e-5, rtol=1e-5)

    def test_sliding_window_kv_cache_gqa(self):
        """Sliding window + KV cache + GQA together."""
        torch.manual_seed(80)
        B, T_max, D = 1, 16, 16
        H_q, H_kv = 4, 2
        T_total = 6
        W = 2

        q_all = torch.randn(B, T_total, H_q, D)
        k_all = torch.randn(B, T_total, H_kv, D)
        v_all = torch.randn(B, T_total, H_kv, D)

        # Full forward with sliding window
        q_f = q_all.transpose(1, 2)
        k_f = k_all.transpose(1, 2)
        v_f = v_all.transpose(1, 2)
        out_full = flash_attn_func(q_f, k_f, v_f, causal=True, window_size=(W, 0))

        # Step-by-step decode
        k_cache = torch.zeros(B, T_max, H_kv, D)
        v_cache = torch.zeros(B, T_max, H_kv, D)

        for t in range(T_total):
            cache_seqlens = torch.tensor([t], dtype=torch.int32)
            out_t = flash_attn_with_kvcache(
                q_all[:, t : t + 1, :, :],
                k_cache,
                v_cache,
                k=k_all[:, t : t + 1, :, :],
                v=v_all[:, t : t + 1, :, :],
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(W, 0),
            )
            out_ref_t = out_full[:, :, t, :]  # (B, H_q, D)
            out_t_squeezed = out_t[:, 0, :, :]  # (B, H_q, D)
            torch.testing.assert_close(
                out_t_squeezed,
                out_ref_t,
                atol=1e-5,
                rtol=1e-5,
                msg=f"GQA+window mismatch at step {t}",
            )
