"""
Comprehensive correctness tests for flash_attention.py (FlexAttention migration).

Tests the flex_attention path (Triton backend on sm89, FLASH backend on sm90+)
and the SDPA fallback against a pure-Python reference implementation.

Run: uv run python -m pytest tests/test_attention_fallback.py -v -s
"""

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.engine import KVCache
from nanochat.flash_attention import (
    HAS_FA3,
    HAS_FA4_BACKEND,
    HAS_FLEX_ATTENTION,
    FlashAttnImpl,
    flash_attn,
)


# =============================================================================
# Helpers
# =============================================================================
def set_impl(impl):
    """Set the implementation override."""
    fa_module._override_impl = impl


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1.float() - t2.float()).abs().max().item()
    mean_diff = (t1.float() - t2.float()).abs().mean().item()
    assert torch.allclose(t1.float(), t2.float(), atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
    return max_diff, mean_diff


# =============================================================================
# Reference implementation (ground truth)
# =============================================================================
def reference_attention(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Pure-Python reference attention. No tricks, no optimizations.
    q: (B, H_q, S_q, D), k: (B, H_kv, S_kv, D), v: (B, H_kv, S_kv, D)
    Returns: (B, H_q, S_q, D)
    """
    B, H_q, S_q, D = q.shape
    _, H_kv, S_kv, _ = k.shape

    # GQA: repeat KV heads to match Q heads
    if H_q != H_kv:
        assert H_q % H_kv == 0
        repeats = H_q // H_kv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    # Compute attention scores in float32 for numerical stability
    scale = D**-0.5
    scores = (
        torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    )  # (B, H, S_q, S_kv)

    # Build mask
    if causal:
        q_pos = torch.arange(S_q, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(S_kv, device=q.device).unsqueeze(0)
        causal_mask = q_pos >= kv_pos
        window_left = window_size[0]
        if window_left >= 0:
            causal_mask = causal_mask & ((q_pos - kv_pos) <= window_left)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def _reset_impl():
    """Save and restore _override_impl across each test.

    Tests in this file mutate fa_module._override_impl directly via
    `set_impl(...)` to exercise specific backends. Save/restore preserves
    the prior value (rather than unconditionally resetting to None) so
    this fixture itself doesn't clobber state another test set up."""
    saved = fa_module._override_impl
    yield
    fa_module._override_impl = saved


# =============================================================================
# Skip conditions
# =============================================================================
requires_flex = pytest.mark.skipif(
    not HAS_FLEX_ATTENTION or not torch.cuda.is_available(),
    reason="Requires flex_attention and CUDA",
)

requires_fa3 = pytest.mark.skipif(
    not HAS_FA3,
    reason="Requires FA3 (Hopper GPU)",
)


# =============================================================================
# Test: FlexAttention vs Reference
# =============================================================================
@requires_flex
class TestFlexAttentionVsReference:
    """Test flex_attention path against reference implementation. Requires CUDA."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def _run_flex(self, q, k, v, causal=True, window_size=(-1, -1)):
        """Run flex_attention via the fa4 override."""
        set_impl("fa4")
        return flash_attn.flash_attn_func(
            q, k, v, causal=causal, window_size=window_size
        )

    def test_causal_full_context(self):
        """Basic causal attention with full context window."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "causal_full_context")
        print(f"causal_full_context: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_causal_sliding_window(self):
        """Sliding window (window=32) with seq_len=128."""
        B, H, S, D = 2, 4, 128, 32
        window = 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(window, 0))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(window, 0))

        max_d, mean_d = assert_close(y_flex, y_ref, "causal_sliding_window")
        print(f"causal_sliding_window: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_gqa(self):
        """GQA: H_q=8, H_kv=2."""
        B, S, D = 2, 64, 32
        H_q, H_kv = 8, 2
        q = torch.randn(B, H_q, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H_kv, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H_kv, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "gqa")
        print(f"gqa: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_127(self):
        """Seq length 127: one less than block size."""
        B, H, S, D = 2, 4, 127, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "seq_len_127")
        print(f"seq_len_127: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_short_transposed_fp32_query_matches_contiguous_on_non_fa4(self):
        """Non-FA4 FlexDecoding reads the standard transposed Q layout correctly."""
        if fa_module.HAS_FA4_BACKEND:
            pytest.skip("Requires the bare non-FA4 FlexAttention fallback")

        B, H, S, D = 4, 6, 96, 128
        torch.manual_seed(1234)
        q = torch.nn.functional.rms_norm(
            torch.randn(B, S, H, D, device=self.DEVICE),
            (D,),
        ).transpose(1, 2)
        k = torch.nn.functional.rms_norm(
            torch.randn(B, S, H, D, device=self.DEVICE),
            (D,),
        ).transpose(1, 2)
        v = (24 * torch.randn(B, S, H, D, device=self.DEVICE, dtype=torch.bfloat16)).transpose(1, 2)
        assert q.dtype == torch.float32
        assert k.dtype == torch.float32
        assert v.dtype == torch.bfloat16
        assert not q.is_contiguous()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            set_impl("fa4")
            y_flex = flash_attn.flash_attn_func(q, k, v, causal=True)
            y_contiguous = flash_attn.flash_attn_func(q.contiguous(), k, v, causal=True)

        # Issue #1601 is a query-layout regression. Identical contiguous-Q
        # execution is an exact oracle, independent of SDPA's numeric mode.
        assert torch.equal(y_flex, y_contiguous)

    def test_seq_len_128(self):
        """Seq length 128: exact block size."""
        B, H, S, D = 2, 4, 128, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "seq_len_128")
        print(f"seq_len_128: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_129(self):
        """Seq length 129: one more than block size."""
        B, H, S, D = 2, 4, 129, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "seq_len_129")
        print(f"seq_len_129: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_255(self):
        """Seq length 255: near Blackwell boundary (256)."""
        B, H, S, D = 2, 4, 255, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "seq_len_255")
        print(f"seq_len_255: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_256(self):
        """Seq length 256: Blackwell block boundary."""
        B, H, S, D = 2, 4, 256, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "seq_len_256")
        print(f"seq_len_256: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_257(self):
        """Seq length 257: one past Blackwell boundary."""
        B, H, S, D = 2, 4, 257, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_ref, "seq_len_257")
        print(f"seq_len_257: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_window_equals_seq_len(self):
        """Window size equals seq_len: should equal full causal."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_window = self._run_flex(q, k, v, causal=True, window_size=(S, 0))
        y_full = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_window, y_full, "window_equals_seq_len")
        print(f"window_equals_seq_len: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_window_larger_than_seq_len(self):
        """Window larger than seq_len: should equal full causal."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_window = self._run_flex(q, k, v, causal=True, window_size=(S * 2, 0))
        y_full = self._run_flex(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_window, y_full, "window_larger_than_seq_len")
        print(
            f"window_larger_than_seq_len: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}"
        )

    def test_window_size_1(self):
        """Minimum window: each query attends to itself and one previous token."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(1, 0))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(1, 0))

        max_d, mean_d = assert_close(y_flex, y_ref, "window_size_1")
        print(f"window_size_1: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_window_size_0(self):
        """Degenerate window: each query attends only to itself."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex = self._run_flex(q, k, v, causal=True, window_size=(0, 0))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(0, 0))

        max_d, mean_d = assert_close(y_flex, y_ref, "window_size_0")
        print(f"window_size_0: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_backward_gradients(self):
        """Forward + backward: check gradients exist and are finite."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(
            B, H, S, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True
        )
        k = torch.randn(
            B, H, S, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True
        )
        v = torch.randn(
            B, H, S, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True
        )

        set_impl("fa4")
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
        loss = y.sum()
        loss.backward()

        for name, tensor in [("q", q), ("k", k), ("v", v)]:
            assert tensor.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(tensor.grad).all(), f"Non-finite gradient for {name}"
        print("backward_gradients: all gradients exist and are finite")


# =============================================================================
# Test: SDPA vs Reference
# =============================================================================
class TestSDPAVsReference:
    """Test SDPA fallback against reference implementation. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _run_sdpa(self, q, k, v, causal=True, window_size=(-1, -1)):
        """Run SDPA via override."""
        set_impl("sdpa")
        return flash_attn.flash_attn_func(
            q, k, v, causal=causal, window_size=window_size
        )

    def test_causal_full_context(self):
        """Basic causal attention with full context window."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_causal_full_context")
        print(f"sdpa_causal_full_context: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_causal_sliding_window(self):
        """Sliding window (window=32) with seq_len=128."""
        B, H, S, D = 2, 4, 128, 32
        window = 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(window, 0))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(window, 0))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_causal_sliding_window")
        print(
            f"sdpa_causal_sliding_window: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}"
        )

    def test_gqa(self):
        """GQA: H_q=8, H_kv=2."""
        B, S, D = 2, 64, 32
        H_q, H_kv = 8, 2
        q = torch.randn(B, H_q, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H_kv, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H_kv, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_gqa")
        print(f"sdpa_gqa: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_127(self):
        """Seq length 127: one less than block size."""
        B, H, S, D = 2, 4, 127, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_seq_len_127")
        print(f"sdpa_seq_len_127: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_128(self):
        """Seq length 128: exact block size."""
        B, H, S, D = 2, 4, 128, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_seq_len_128")
        print(f"sdpa_seq_len_128: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_129(self):
        """Seq length 129: one more than block size."""
        B, H, S, D = 2, 4, 129, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_seq_len_129")
        print(f"sdpa_seq_len_129: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_255(self):
        """Seq length 255: near Blackwell boundary."""
        B, H, S, D = 2, 4, 255, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_seq_len_255")
        print(f"sdpa_seq_len_255: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_256(self):
        """Seq length 256: Blackwell block boundary."""
        B, H, S, D = 2, 4, 256, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_seq_len_256")
        print(f"sdpa_seq_len_256: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_seq_len_257(self):
        """Seq length 257: one past Blackwell boundary."""
        B, H, S, D = 2, 4, 257, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_seq_len_257")
        print(f"sdpa_seq_len_257: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_window_equals_seq_len(self):
        """Window size equals seq_len: should equal full causal."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_window = self._run_sdpa(q, k, v, causal=True, window_size=(S, 0))
        y_full = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_window, y_full, "sdpa_window_equals_seq_len")
        print(
            f"sdpa_window_equals_seq_len: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}"
        )

    def test_window_larger_than_seq_len(self):
        """Window larger than seq_len: should equal full causal."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_window = self._run_sdpa(q, k, v, causal=True, window_size=(S * 2, 0))
        y_full = self._run_sdpa(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(
            y_window, y_full, "sdpa_window_larger_than_seq_len"
        )
        print(
            f"sdpa_window_larger_than_seq_len: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}"
        )

    def test_window_size_1(self):
        """Window size 1: each token attends only to itself."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(1, 0))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(1, 0))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_window_size_1")
        print(f"sdpa_window_size_1: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_window_size_0(self):
        """Window size 0: each token attends only to itself (self-attention only)."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_sdpa = self._run_sdpa(q, k, v, causal=True, window_size=(0, 0))
        y_ref = reference_attention(q, k, v, causal=True, window_size=(0, 0))

        max_d, mean_d = assert_close(y_sdpa, y_ref, "sdpa_window_size_0")
        print(f"sdpa_window_size_0: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_backward(self):
        """Forward + backward: check gradients exist and are finite."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(
            B, H, S, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True
        )
        k = torch.randn(
            B, H, S, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True
        )
        v = torch.randn(
            B, H, S, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True
        )

        set_impl("sdpa")
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
        loss = y.sum()
        loss.backward()

        for name, tensor in [("q", q), ("k", k), ("v", v)]:
            assert tensor.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(tensor.grad).all(), f"Non-finite gradient for {name}"
        print("sdpa_backward: all gradients exist and are finite")


# =============================================================================
# Test: FlexAttention vs SDPA (cross-backend)
# =============================================================================
@requires_flex
class TestFlexVsSDPA:
    """Cross-backend comparison: flex_attention vs SDPA on same inputs."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def _run_both(self, q, k, v, causal=True, window_size=(-1, -1)):
        """Run both backends, return (flex_out, sdpa_out)."""
        set_impl("fa4")
        y_flex = flash_attn.flash_attn_func(
            q, k, v, causal=causal, window_size=window_size
        )
        set_impl("sdpa")
        y_sdpa = flash_attn.flash_attn_func(
            q, k, v, causal=causal, window_size=window_size
        )
        return y_flex, y_sdpa

    def test_cross_backend_causal(self):
        """Both backends produce same output for causal attention."""
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex, y_sdpa = self._run_both(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_sdpa, "cross_backend_causal")
        print(f"cross_backend_causal: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_cross_backend_sliding_window(self):
        """Both backends produce same output for sliding window."""
        B, H, S, D = 2, 4, 128, 32
        window = 32
        q = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex, y_sdpa = self._run_both(q, k, v, causal=True, window_size=(window, 0))

        max_d, mean_d = assert_close(y_flex, y_sdpa, "cross_backend_sliding_window")
        print(
            f"cross_backend_sliding_window: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}"
        )

    def test_cross_backend_gqa(self):
        """Both backends produce same output for GQA."""
        B, S, D = 2, 64, 32
        H_q, H_kv = 8, 2
        q = torch.randn(B, H_q, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H_kv, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H_kv, S, D, device=self.DEVICE, dtype=self.DTYPE)

        y_flex, y_sdpa = self._run_both(q, k, v, causal=True, window_size=(-1, -1))

        max_d, mean_d = assert_close(y_flex, y_sdpa, "cross_backend_gqa")
        print(f"cross_backend_gqa: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

    def test_cross_backend_gradients(self):
        """Both backends produce similar gradients."""
        B, H, S, D = 2, 4, 64, 32
        q_data = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, H, S, D, device=self.DEVICE, dtype=self.DTYPE)

        results = {}
        for impl_name in ("fa4", "sdpa"):
            set_impl(impl_name)
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
            y.sum().backward()
            q_grad = q.grad
            k_grad = k.grad
            v_grad = v.grad
            assert q_grad is not None
            assert k_grad is not None
            assert v_grad is not None
            results[impl_name] = {
                "y": y.detach(),
                "q_grad": q_grad.detach(),
                "k_grad": k_grad.detach(),
                "v_grad": v_grad.detach(),
            }

        max_d, mean_d = assert_close(
            results["fa4"]["y"], results["sdpa"]["y"], "cross_grad_output"
        )
        print(f"cross_grad_output: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")

        for param in ("q_grad", "k_grad", "v_grad"):
            max_d, mean_d = assert_close(
                results["fa4"][param],
                results["sdpa"][param],
                f"cross_{param}",
                atol=0.05,
                rtol=0.05,
            )
            print(f"cross_{param}: max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")


# =============================================================================
# Test: KV Cache (SDPA path, since no FA3 on sm89)
# =============================================================================
class TestKVCache:
    """Test KV cache path (flash_attn_with_kvcache). Uses SDPA since FA3 unavailable on sm89."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_prefill_shape(self):
        """Output shape is correct after prefill."""
        set_impl("sdpa")
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16
        n_layers = 1

        cache = KVCache(
            batch_size=B,
            num_heads=H,
            seq_len=T_max,
            head_dim=D,
            num_layers=n_layers,
            device=self.DEVICE,
            dtype=self.DTYPE,
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D), (
            f"Expected {(B, T_prefill, H, D)}, got {y.shape}"
        )
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert cache.get_pos() == T_prefill
        print(f"prefill_shape: {y.shape}, pos={cache.get_pos()}")

    def test_single_token_decode(self):
        """Single token decode after prefill."""
        set_impl("sdpa")
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16
        n_layers = 1

        cache = KVCache(
            batch_size=B,
            num_heads=H,
            seq_len=T_max,
            head_dim=D,
            num_layers=n_layers,
            device=self.DEVICE,
            dtype=self.DTYPE,
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        cache.advance(T_prefill)

        # Single token decode
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single,
            k_cache,
            v_cache,
            k=k_single,
            v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D), (
            f"Expected {(B, 1, H, D)}, got {y_single.shape}"
        )
        assert not torch.isnan(y_single).any(), "Output contains NaN"
        assert cache.get_pos() == T_prefill + 1
        print(f"single_token_decode: {y_single.shape}, pos={cache.get_pos()}")

    def test_sliding_window_decode(self):
        """Single token with window < cache size."""
        set_impl("sdpa")
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32
        window = 8  # Much smaller than cache
        n_layers = 1

        cache = KVCache(
            batch_size=B,
            num_heads=H,
            seq_len=T_max,
            head_dim=D,
            num_layers=n_layers,
            device=self.DEVICE,
            dtype=self.DTYPE,
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        cache.advance(T_prefill)

        # Single token decode with sliding window
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_window = flash_attn.flash_attn_with_kvcache(
            q_single,
            k_cache,
            v_cache,
            k=k_single,
            v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(window, 0),
        )

        assert y_window.shape == (B, 1, H, D)
        assert not torch.isnan(y_window).any(), "Output contains NaN"

        # Verify sliding window is different from full context
        y_full = flash_attn.flash_attn_with_kvcache(
            q_single,
            k_cache,
            v_cache,
            k=k_single,
            v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        # They should NOT be identical since window < T_prefill
        diff = (y_window.float() - y_full.float()).abs().max().item()
        assert diff > 1e-6, (
            f"Sliding window decode same as full context (diff={diff:.8f}). "
            "Window masking may not be applied."
        )
        print(f"sliding_window_decode: window vs full diff={diff:.6f}")


# =============================================================================
# Test: Override mechanism
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism routes to the correct backend."""

    def test_override_fa4(self):
        """Override to fa4 uses flex_attention when available."""
        if not HAS_FLEX_ATTENTION:
            pytest.skip("flex_attention not available")
        set_impl("fa4")
        assert fa_module._use_flex() is True
        assert fa_module._active_impl() == FlashAttnImpl.FA4

    def test_override_sdpa(self):
        """Override to sdpa skips flex_attention."""
        set_impl("sdpa")
        assert fa_module._use_flex() is False
        assert fa_module._use_fa3() is False
        assert fa_module._active_impl() == FlashAttnImpl.SDPA

    def test_override_auto(self):
        """Override=None uses auto-detection."""
        set_impl(None)
        impl = fa_module._active_impl()
        if HAS_FLEX_ATTENTION:
            assert impl == FlashAttnImpl.FA4
        elif HAS_FA3:
            assert impl == FlashAttnImpl.FA3
        else:
            assert impl == FlashAttnImpl.SDPA

    @requires_fa3
    def test_override_fa3(self):
        """Override to fa3 when FA3 is available."""
        set_impl("fa3")
        assert fa_module._use_fa3() is True
        assert fa_module._active_impl() == FlashAttnImpl.FA3


# =============================================================================
# Test: FA4 detection flags
# =============================================================================
class TestFA4Detection:
    """Test FA4/flex_attention detection flags and active impl."""

    def test_has_fa4_backend_flag_type(self):
        """HAS_FA4_BACKEND flag should be a boolean."""
        assert isinstance(fa_module.HAS_FA4_BACKEND, bool)

    def test_has_flex_attention_flag_type(self):
        """HAS_FLEX_ATTENTION flag should be a boolean."""
        assert isinstance(fa_module.HAS_FLEX_ATTENTION, bool)

    def test_active_impl_returns_string(self):
        """_active_impl() should return the name of the active backend."""
        result = fa_module._active_impl()
        assert result in (FlashAttnImpl.FA4, FlashAttnImpl.FA3, FlashAttnImpl.SDPA)

    def test_flash_attn_impl_enum(self):
        """FlashAttnImpl enum has expected values."""
        assert FlashAttnImpl.FA4 == "fa4"
        assert FlashAttnImpl.FA3 == "fa3"
        assert FlashAttnImpl.SDPA == "sdpa"


# =============================================================================
# Test: CuTe DSL availability check (the actual root cause of the FA4 crash)
# =============================================================================
class TestCuteDSLAvailabilityCheck:
    """Test that HAS_FA4_BACKEND correctly requires flash_attn.cute (CuTe DSL).

    Without this check, Hopper GPUs would try BACKEND='FLASH' and crash with:
    RuntimeError: BACKEND='FLASH' but flash attention cannot be used:
    CUTE flash attention library is not available
    """

    def test_check_cute_dsl_available_returns_bool(self):
        """_check_cute_dsl_available() returns a boolean."""
        result = fa_module._check_cute_dsl_available()
        assert isinstance(result, bool)

    def test_cute_dsl_check_matches_importlib(self):
        """_check_cute_dsl_available() result matches direct importlib check."""
        import importlib.util

        try:
            spec = importlib.util.find_spec("flash_attn.cute")
            expected = spec is not None
        except (ImportError, ModuleNotFoundError):
            expected = False
        assert fa_module._check_cute_dsl_available() == expected

    def test_has_fa4_backend_requires_cute_dsl(self):
        """HAS_FA4_BACKEND must be False when flash_attn.cute is unavailable.

        This is the critical invariant: if CuTe DSL is not installed,
        HAS_FA4_BACKEND MUST be False regardless of GPU arch, because
        BACKEND='FLASH' will crash at kernel compile time without it.
        """
        if fa_module._check_cute_dsl_available():
            pytest.skip("flash_attn.cute is installed — can't test negative case")
        assert fa_module.HAS_FA4_BACKEND is False, (
            "HAS_FA4_BACKEND is True but flash_attn.cute is not installed. "
            "This would cause RuntimeError when using BACKEND='FLASH'."
        )

    def test_has_fa4_backend_implies_cute_dsl(self):
        """If HAS_FA4_BACKEND is True, CuTe DSL must be available."""
        if fa_module.HAS_FA4_BACKEND:
            assert fa_module._check_cute_dsl_available(), (
                "HAS_FA4_BACKEND is True but _check_cute_dsl_available() is False"
            )

    def test_has_fa4_backend_implies_flex_attention(self):
        """If HAS_FA4_BACKEND is True, flex_attention must be available."""
        if fa_module.HAS_FA4_BACKEND:
            assert fa_module.HAS_FLEX_ATTENTION, (
                "HAS_FA4_BACKEND is True but HAS_FLEX_ATTENTION is False"
            )

    def test_has_fa4_backend_implies_hopper_or_later(self):
        """If HAS_FA4_BACKEND is True, GPU must be Hopper (sm90) or later."""
        if fa_module.HAS_FA4_BACKEND:
            assert fa_module._gpu_arch_major >= 9, (
                f"HAS_FA4_BACKEND is True but GPU arch is {fa_module._gpu_arch_major}"
            )

    def test_compiled_flex_does_not_use_flash_without_cute_dsl(self):
        """When CuTe DSL is unavailable, _compiled_flex must not use FLASH backend.

        The _compiled_flex function is compiled at import time. If HAS_FA4_BACKEND
        is correctly False, it uses the Triton backend (no kernel_options).
        """
        if fa_module._check_cute_dsl_available():
            pytest.skip("flash_attn.cute is installed")
        # _compiled_flex should exist (flex_attention is available on modern PyTorch)
        if not HAS_FLEX_ATTENTION:
            pytest.skip("flex_attention not available")
        assert fa_module._compiled_flex is not None, (
            "_compiled_flex should be set when flex_attention is available"
        )
        # On a non-FA4 path, _compiled_flex should NOT have FLASH baked in.
        # We can't easily inspect the compiled function, but we verify the flag
        # is False, which controls the branch at module import time.
        assert fa_module.HAS_FA4_BACKEND is False

    @requires_flex
    def test_flex_attention_works_without_cute_dsl(self):
        """flex_attention with Triton backend works when CuTe DSL is unavailable.

        This is the fallback path: Hopper GPUs without flash-attn-4 installed
        should still work via the Triton backend.
        """
        if fa_module._check_cute_dsl_available():
            pytest.skip("flash_attn.cute is installed — testing Triton fallback path")
        set_impl("fa4")  # uses flex_attention (Triton backend since no CuTe DSL)
        B, H, S, D = 2, 4, 64, 32
        device = "cuda"
        dtype = torch.bfloat16
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        y = flash_attn.flash_attn_func(q, k, v, causal=True)
        assert y.shape == (B, H, S, D)
        assert not torch.isnan(y).any(), "Output contains NaN"


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA4_BACKEND: {HAS_FA4_BACKEND}")
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"HAS_FLEX_ATTENTION: {HAS_FLEX_ATTENTION}")
    print()

    pytest.main([__file__, "-v", "-s"])
