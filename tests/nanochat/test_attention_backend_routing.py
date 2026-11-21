"""Tests for attention backend detection and routing logic (nanochat/flash_attention.py).

Verifies the backend selection chain works correctly:
- NANOCHAT_ATTN_IMPL env var / _override_impl override
- _use_flex() and _use_fa3() routing functions
- _active_impl() returns correct backend
- SDPA fallback always works
- FA3 override rejected on non-Hopper hardware

This is critical after the FA4 dependency and backend-priority changes.
"""

import pytest
import torch

import nanochat.flash_attention as fa_mod
from nanochat.flash_attention import FlashAttnImpl


@pytest.fixture(autouse=True)
def _reset_override():
    """Reset _override_impl after each test."""
    saved = fa_mod._override_impl
    yield
    fa_mod._override_impl = saved


class TestOverrideImpl:
    """Verify _override_impl controls backend selection."""

    def test_sdpa_override_disables_flex(self):
        """SDPA override makes _use_flex() return False."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        assert not fa_mod._use_flex()

    def test_sdpa_override_disables_fa3(self):
        """SDPA override makes _use_fa3() return False."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        assert not fa_mod._use_fa3()

    def test_sdpa_override_active_impl(self):
        """SDPA override makes _active_impl() return SDPA."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        assert fa_mod._active_impl() == FlashAttnImpl.SDPA

    def test_fa3_override_disables_flex(self):
        """FA3 override makes _use_flex() return False."""
        fa_mod._override_impl = FlashAttnImpl.FA3
        assert not fa_mod._use_flex()

    def test_fa3_override_active_impl(self):
        """FA3 override makes _active_impl() return FA3."""
        fa_mod._override_impl = FlashAttnImpl.FA3
        assert fa_mod._active_impl() == FlashAttnImpl.FA3

    def test_fa4_override_enables_flex(self):
        """FA4 override makes _use_flex() return True (if flex_attention available)."""
        if not fa_mod.HAS_FLEX_ATTENTION:
            pytest.skip("flex_attention not available")
        fa_mod._override_impl = FlashAttnImpl.FA4
        assert fa_mod._use_flex()

    def test_none_override_uses_auto(self):
        """None override falls through to auto-detection."""
        fa_mod._override_impl = None
        impl = fa_mod._active_impl()
        # On CPU, flex_attention is available but can't
        # do backward. Auto should select FA4 (flex) since it's available.
        if fa_mod.HAS_FLEX_ATTENTION:
            assert impl == FlashAttnImpl.FA4
        elif fa_mod.HAS_FA3:
            assert impl == FlashAttnImpl.FA3
        else:
            assert impl == FlashAttnImpl.SDPA


class TestSDPAFallback:
    """SDPA fallback works correctly for both training and inference."""

    def test_sdpa_forward_training(self):
        """SDPA backend handles training forward pass on CPU."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        B, H, T, D = 2, 2, 16, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out = fa_mod.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()

    def test_sdpa_forward_sliding_window(self):
        """SDPA backend handles sliding window attention."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        B, H, T, D = 2, 2, 16, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out = fa_mod.flash_attn_func(q, k, v, causal=True, window_size=(8, 0))
        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()

    def test_sdpa_backward(self):
        """SDPA backend supports backward on CPU."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        B, H, T, D = 2, 2, 16, 32
        q = torch.randn(B, H, T, D, requires_grad=True)
        k = torch.randn(B, H, T, D, requires_grad=True)
        v = torch.randn(B, H, T, D, requires_grad=True)
        out = fa_mod.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        out.sum().backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_sdpa_gqa(self):
        """SDPA backend handles GQA (different Q and KV head counts)."""
        fa_mod._override_impl = FlashAttnImpl.SDPA
        B, H_q, H_kv, T, D = 2, 4, 2, 16, 32
        q = torch.randn(B, H_q, T, D)
        k = torch.randn(B, H_kv, T, D)
        v = torch.randn(B, H_kv, T, D)
        out = fa_mod.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        assert out.shape == (B, H_q, T, D)
        assert not torch.isnan(out).any()


class TestBackendDetectionFlags:
    """Verify detection flags are consistent with torch version."""

    def test_flex_attention_available(self):
        """flex_attention should be available on torch >= 2.5."""
        major, minor = torch.__version__.split(".")[:2]
        if int(major) >= 2 and int(minor) >= 5:
            assert fa_mod.HAS_FLEX_ATTENTION, (
                f"flex_attention should be available on torch {torch.__version__}"
            )

    def test_fa4_backend_requires_cute_dsl(self):
        """FA4 (FLASH) backend requires flash_attn.cute package."""
        # Non-Linux development environments do not install the FA4 dependency.
        if not fa_mod._check_cute_dsl_available():
            assert not fa_mod.HAS_FA4_BACKEND, (
                "HAS_FA4_BACKEND should be False without flash_attn.cute"
            )

    def test_flash_attn_impl_enum_values(self):
        """FlashAttnImpl enum has the expected values."""
        assert FlashAttnImpl.FA4 == "fa4"
        assert FlashAttnImpl.FA3 == "fa3"
        assert FlashAttnImpl.SDPA == "sdpa"
        assert FlashAttnImpl.AUTO == "auto"


class TestEnvVarOverride:
    """Verify NANOCHAT_ATTN_IMPL env var behavior."""

    def test_env_var_fa3_parsed(self):
        """NANOCHAT_ATTN_IMPL=fa3 is correctly parsed."""
        result = FlashAttnImpl("fa3")
        assert result == FlashAttnImpl.FA3

    def test_env_var_sdpa_parsed(self):
        """NANOCHAT_ATTN_IMPL=sdpa is correctly parsed."""
        result = FlashAttnImpl("sdpa")
        assert result == FlashAttnImpl.SDPA

    def test_env_var_invalid_raises(self):
        """Invalid NANOCHAT_ATTN_IMPL value raises ValueError."""
        with pytest.raises(ValueError):
            FlashAttnImpl("invalid_backend")


class TestFlexDecodeQueryLayout:
    """Pin the non-FA4 short-query layout guard around PyTorch FlexDecoding."""

    @pytest.mark.parametrize(
        (
            "sequence_length",
            "has_fa4_backend",
            "query_dtype",
            "expects_contiguous_query",
        ),
        [
            (127, False, torch.float32, True),
            (127, False, torch.bfloat16, False),
            (128, False, torch.float32, False),
            (127, True, torch.float32, False),
        ],
    )
    def test_only_short_non_fa4_query_is_made_contiguous(
        self,
        monkeypatch,
        sequence_length,
        has_fa4_backend,
        query_dtype,
        expects_contiguous_query,
    ):
        """Avoid the FlexDecoding stride bug without copying full Flex or FA4 Q."""
        B, H, D = 2, 3, 16
        q = torch.randn(B, sequence_length, H, D).to(query_dtype).transpose(1, 2)
        k = torch.randn(B, sequence_length, H, D).transpose(1, 2)
        v = torch.randn(B, sequence_length, H, D).transpose(1, 2)
        assert not q.is_contiguous()

        captured = {}

        def fake_compiled_flex(query, key, value, **_kwargs):
            captured.update(q=query, k=key, v=value)
            return query

        monkeypatch.setattr(fa_mod, "_use_flex", lambda: True)
        monkeypatch.setattr(fa_mod, "_ensure_flash_compat", lambda: None)
        monkeypatch.setattr(fa_mod, "HAS_FA4_BACKEND", has_fa4_backend)
        monkeypatch.setattr(fa_mod, "_get_block_mask", lambda *_args: object())
        monkeypatch.setattr(fa_mod, "_compiled_flex", fake_compiled_flex)

        output = fa_mod.flash_attn_func(q, k, v, causal=True)

        assert captured["q"].is_contiguous() is expects_contiguous_query
        assert torch.equal(captured["q"], q)
        if expects_contiguous_query:
            assert captured["q"].data_ptr() != q.data_ptr()
        else:
            assert captured["q"] is q
        assert captured["k"] is k
        assert captured["v"] is v
        assert output is captured["q"]
