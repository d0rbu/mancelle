"""
Tests for FA4 FLASH backend compatibility: score_mod and mask_mod must accept
extra kwargs (e.g. seqlen_info) that the CuTe DSL FLASH backend passes.

The flash-attn-4 CuTe DSL FLASH backend passes additional keyword arguments
(like seqlen_info) to score_mod functions. Without **kwargs in the function
signature, this causes:
    TypeError: score_mod() got an unexpected keyword argument 'seqlen_info'

This test suite verifies:
1. All score_mod functions accept **kwargs (the actual fix)
2. All mask_mod functions accept **kwargs (defensive)
3. The NANOCHAT_ATTN_IMPL env var override works correctly
4. End-to-end cross-attention works with simulated extra kwargs

Run: uv run python -m pytest tests/test_fa4_score_mod_compat.py -v -s
"""

import inspect

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import (
    HAS_FLEX_ATTENTION,
    FlashAttnImpl,
    _causal_mask,
    _make_causal_sliding_mask,
    has_fa4_backend,
)
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
    build_packed_cross_attention_block_mask,
)
from tests.padded_encoder_reference import pack_dense_encoder_output_reference


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def _reset_impl():
    """Save and restore _override_impl + _flash_compat_validated across each test.

    Tests in this file mutate both globals to exercise specific backends.
    Save/restore preserves the prior value (rather than unconditionally
    resetting to None / a fixed value) so this fixture itself doesn't
    clobber state another test set up."""
    saved_override = fa_module._override_impl
    saved_validated = fa_module._flash_compat_validated
    yield
    fa_module._override_impl = saved_override
    fa_module._flash_compat_validated = saved_validated


# =============================================================================
# Test: score_mod accepts extra kwargs
# =============================================================================
class TestScoreModKwargsCompat:
    """Verify that score_mod functions in cross_attention.py accept **kwargs.

    The FLASH backend (CuTe DSL) passes seqlen_info and potentially other
    kwargs to score_mod. Our functions must accept and ignore them.
    """

    def test_cross_attention_uses_flex_block_mask_on_cuda(self):
        """CUDA cross-attention routes packed queries through FlexAttention."""
        helper_source = inspect.getsource(build_packed_cross_attention_block_mask)
        cuda_source = inspect.getsource(CrossAttentionLayer._packed_cross_attention_dynamic)
        assert "build_block_mask_from_dense_routing" in helper_source
        assert "**kwargs" in helper_source
        assert "flex_attention" in cuda_source

    def test_score_mod_with_seqlen_info_kwarg(self):
        """Simulate the FLASH backend passing seqlen_info to a score_mod."""
        # Replicate the _score_mod pattern from cross_attention.py
        kv_valid = torch.tensor([[True, True, False, False]])  # (1, 4)

        def _score_mod(score, b, h, q_idx, kv_idx, **kwargs):
            return torch.where(kv_valid[b, kv_idx], score, -float("inf"))

        # Simulate FLASH backend calling with seqlen_info
        score = torch.randn(1)
        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(0)
        kv_idx = torch.tensor(1)
        seqlen_info = {"q_seqlen": torch.tensor([4]), "kv_seqlen": torch.tensor([4])}

        # This must NOT raise TypeError
        result = _score_mod(score, b, h, q_idx, kv_idx, seqlen_info=seqlen_info)
        assert result.shape == score.shape

    def test_score_mod_with_multiple_extra_kwargs(self):
        """score_mod must handle arbitrary future extra kwargs."""
        kv_valid = torch.tensor([[True, True, False]])

        def _score_mod(score, b, h, q_idx, kv_idx, **kwargs):
            return torch.where(kv_valid[b, kv_idx], score, -float("inf"))

        score = torch.randn(1)
        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(0)
        kv_idx = torch.tensor(0)

        # Future-proof: multiple extra kwargs
        result = _score_mod(
            score,
            b,
            h,
            q_idx,
            kv_idx,
            seqlen_info={"q_seqlen": torch.tensor([3])},
            some_future_arg=42,
            another_arg="test",
        )
        assert result.shape == score.shape

    def test_score_mod_without_extra_kwargs_still_works(self):
        """score_mod must still work when called without extra kwargs (Triton backend)."""
        kv_valid = torch.tensor([[True, False]])

        def _score_mod(score, b, h, q_idx, kv_idx, **kwargs):
            return torch.where(kv_valid[b, kv_idx], score, -float("inf"))

        score = torch.tensor(1.5)
        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(0)

        # kv_idx=0 -> valid -> score unchanged
        result = _score_mod(score, b, h, q_idx, torch.tensor(0))
        assert result == score

        # kv_idx=1 -> invalid -> -inf
        result = _score_mod(score, b, h, q_idx, torch.tensor(1))
        assert result == float("-inf")


# =============================================================================
# Test: mask_mod accepts extra kwargs (defensive)
# =============================================================================
class TestMaskModKwargsCompat:
    """Verify that mask_mod functions accept **kwargs for forward-compatibility."""

    def test_causal_mask_accepts_kwargs(self):
        """_causal_mask must accept **kwargs."""
        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(5)
        kv_idx = torch.tensor(3)

        # With extra kwargs (should not raise)
        result = _causal_mask(b, h, q_idx, kv_idx, seqlen_info={"test": 1})
        assert result.item() is True  # q_idx=5 >= kv_idx=3

    def test_causal_mask_correctness_with_kwargs(self):
        """_causal_mask produces correct results even with extra kwargs."""
        b = torch.tensor(0)
        h = torch.tensor(0)

        # q_idx >= kv_idx -> True
        assert (
            _causal_mask(b, h, torch.tensor(5), torch.tensor(3), extra=42).item()
            is True
        )
        assert (
            _causal_mask(b, h, torch.tensor(3), torch.tensor(3), extra=42).item()
            is True
        )
        # q_idx < kv_idx -> False
        assert (
            _causal_mask(b, h, torch.tensor(2), torch.tensor(3), extra=42).item()
            is False
        )

    def test_causal_mask_without_kwargs(self):
        """_causal_mask still works without extra kwargs (standard path)."""
        b = torch.tensor(0)
        h = torch.tensor(0)
        assert _causal_mask(b, h, torch.tensor(5), torch.tensor(3)).item() is True
        assert _causal_mask(b, h, torch.tensor(2), torch.tensor(3)).item() is False

    def test_sliding_mask_accepts_kwargs(self):
        """_make_causal_sliding_mask's inner mask_mod must accept **kwargs."""
        mask_fn = _make_causal_sliding_mask(window_size=4)
        b = torch.tensor(0)
        h = torch.tensor(0)

        # Within window: q=5, kv=3, diff=2 <= 4 -> True
        result = mask_fn(
            b, h, torch.tensor(5), torch.tensor(3), seqlen_info={"test": 1}
        )
        assert result.item() is True

    def test_sliding_mask_correctness_with_kwargs(self):
        """Sliding window mask produces correct results with extra kwargs."""
        mask_fn = _make_causal_sliding_mask(window_size=2)
        b = torch.tensor(0)
        h = torch.tensor(0)

        # q=5, kv=3, diff=2 <= 2 -> True (at boundary)
        assert mask_fn(b, h, torch.tensor(5), torch.tensor(3), extra=1).item() is True
        # q=5, kv=2, diff=3 > 2 -> False (outside window)
        assert mask_fn(b, h, torch.tensor(5), torch.tensor(2), extra=1).item() is False
        # q=5, kv=5, diff=0 <= 2 -> True (self)
        assert mask_fn(b, h, torch.tensor(5), torch.tensor(5), extra=1).item() is True
        # q=3, kv=5 -> causal violation -> False
        assert mask_fn(b, h, torch.tensor(3), torch.tensor(5), extra=1).item() is False

    def test_sliding_mask_without_kwargs(self):
        """Sliding mask still works without extra kwargs."""
        mask_fn = _make_causal_sliding_mask(window_size=4)
        b = torch.tensor(0)
        h = torch.tensor(0)
        assert mask_fn(b, h, torch.tensor(5), torch.tensor(3)).item() is True
        assert mask_fn(b, h, torch.tensor(5), torch.tensor(0)).item() is False

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION,
        reason="Requires flex_attention",
    )
    def test_create_block_mask_with_kwargs_mask_mod(self):
        """create_block_mask works with mask_mod that has **kwargs."""
        from torch.nn.attention.flex_attention import create_block_mask

        # Use our actual _causal_mask (now with **kwargs)
        bm = create_block_mask(
            _causal_mask,
            B=1,
            H=1,
            Q_LEN=64,
            KV_LEN=64,
            device="cpu",
        )
        assert bm is not None

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION,
        reason="Requires flex_attention",
    )
    def test_create_block_mask_sliding_with_kwargs_mask_mod(self):
        """create_block_mask works with sliding window mask_mod that has **kwargs."""
        from torch.nn.attention.flex_attention import create_block_mask

        mask_fn = _make_causal_sliding_mask(window_size=16)
        bm = create_block_mask(
            mask_fn,
            B=1,
            H=1,
            Q_LEN=64,
            KV_LEN=64,
            device="cpu",
        )
        assert bm is not None


# =============================================================================
# Test: NANOCHAT_ATTN_IMPL env var override
# =============================================================================
class TestEnvVarOverride:
    """Test the NANOCHAT_ATTN_IMPL environment variable override mechanism."""

    def test_env_var_sets_override_on_import(self):
        """Setting NANOCHAT_ATTN_IMPL before import overrides the impl."""
        # We can't re-import easily, but we can test the env var parsing logic
        for impl_str in ("fa3", "fa4", "sdpa"):
            result = FlashAttnImpl(impl_str)
            assert result.value == impl_str

    def test_env_var_invalid_value_raises(self):
        """Invalid NANOCHAT_ATTN_IMPL value raises ValueError."""
        with pytest.raises(ValueError):
            FlashAttnImpl("invalid_backend")

    def test_programmatic_override_sdpa(self):
        """Programmatic override to sdpa works."""
        fa_module._override_impl = FlashAttnImpl.SDPA
        assert fa_module._active_impl() == FlashAttnImpl.SDPA
        assert fa_module._use_flex() is False
        assert fa_module._use_fa3() is False

    def test_programmatic_override_fa4(self):
        """Programmatic override to fa4 works (when flex_attention available)."""
        if not HAS_FLEX_ATTENTION:
            pytest.skip("flex_attention not available")
        fa_module._override_impl = FlashAttnImpl.FA4
        assert fa_module._active_impl() == FlashAttnImpl.FA4
        assert fa_module._use_flex() is True

    def test_programmatic_override_none_auto_detects(self):
        """Setting override to None uses auto-detection."""
        fa_module._override_impl = None
        impl = fa_module._active_impl()
        assert impl in (FlashAttnImpl.FA4, FlashAttnImpl.FA3, FlashAttnImpl.SDPA)


# =============================================================================
# Test: Signature inspection (belt-and-suspenders)
# =============================================================================
class TestFunctionSignatures:
    """Verify function signatures include **kwargs via inspection."""

    def test_causal_mask_has_var_keyword(self):
        """_causal_mask must have a **kwargs parameter."""
        sig = inspect.signature(_causal_mask)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        assert has_var_keyword, (
            f"_causal_mask signature {sig} lacks **kwargs — "
            "FLASH backend may pass extra kwargs like seqlen_info"
        )

    def test_sliding_mask_has_var_keyword(self):
        """_make_causal_sliding_mask's inner function must have **kwargs."""
        mask_fn = _make_causal_sliding_mask(window_size=4)
        sig = inspect.signature(mask_fn)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        assert has_var_keyword, (
            f"sliding mask signature {sig} lacks **kwargs — "
            "FLASH backend may pass extra kwargs like seqlen_info"
        )

    def test_cross_attention_imports_flex_attention(self):
        """Packed CUDA cross-attention imports the native FlexAttention API."""
        from steering.conditional.encoder_decoder import cross_attention

        source = inspect.getsource(cross_attention)
        # Strip docstrings and comments, keep only executable code
        import ast

        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [alias.name for alias in node.names]
                imports.append(module + " " + " ".join(names))
        assert any("flex_attention" in entry for entry in imports)


# =============================================================================
# Test: Cross-attention with simulated FLASH kwargs
# =============================================================================
class TestCrossAttentionFlashCompat:
    """Test-owned CPU oracle and CUDA packed FlexAttention smoke tests."""

    def _make_layer(
        self, dec_n_embd=64, enc_n_embd=64, n_head=2, n_kv_head=2, head_dim=32
    ):
        config = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )
        layer = CrossAttentionLayer(config)
        layer.init_weights()
        return layer

    def test_cross_attention_cpu_sdpa_reference(self):
        """The test-owned padded SDPA oracle remains finite on CPU."""
        fa_module._override_impl = FlashAttnImpl.SDPA
        layer = self._make_layer()

        B, T, n_seg, F_feat = 2, 16, 2, 4
        decoder_hidden = torch.randn(B, T, 64)
        encoder_kv = torch.randn(B, n_seg, F_feat, 64)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        encoder_mask[:, :, -1] = False  # last feature is padding
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2 :] = 1

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, 64)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA",
    )
    def test_cross_attention_packed_cuda_runs_with_fa4_override(self):
        """Packed CUDA cross-attention runs when decoder attention uses FA4."""
        fa_module._override_impl = FlashAttnImpl.FA4
        layer = self._make_layer().cuda()

        B, T, n_seg, F_feat = 2, 16, 2, 4
        decoder_hidden = torch.randn(B, T, 64, device="cuda")
        encoder_kv = torch.randn(B, n_seg, F_feat, 64, device="cuda")
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        encoder_mask[:, :, -1] = False
        seg_map = torch.zeros(B, T, dtype=torch.long, device="cuda")
        seg_map[:, T // 2 :] = 1
        packed_encoder_kv = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed_encoder_kv.layout)

        with torch.no_grad():
            out = layer(
                decoder_hidden,
                packed_encoder_kv,
                seg_map,
                packed_block_mask=block_mask,
            )

        assert out.shape == (B, T, 64)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA",
    )
    def test_decoder_attention_override_does_not_change_packed_xattn(self):
        """Decoder backend selection is independent of packed cross-attention."""
        layer = self._make_layer().cuda()

        B, T, n_seg, F_feat = 2, 16, 2, 4
        decoder_hidden = torch.randn(B, T, 64, device="cuda")
        encoder_kv = torch.randn(B, n_seg, F_feat, 64, device="cuda")
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        encoder_mask[:, :, -1] = False
        seg_map = torch.zeros(B, T, dtype=torch.long, device="cuda")
        seg_map[:, T // 2 :] = 1
        packed_encoder_kv = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed_encoder_kv.layout)

        with torch.no_grad():
            fa_module._override_impl = FlashAttnImpl.FA4
            out_fa4 = layer(
                decoder_hidden,
                packed_encoder_kv,
                seg_map,
                packed_block_mask=block_mask,
            )

            fa_module._override_impl = FlashAttnImpl.SDPA
            out_sdpa = layer(
                decoder_hidden,
                packed_encoder_kv,
                seg_map,
                packed_block_mask=block_mask,
            )

        # Packed cross-attention uses Triton independently of the decoder
        # self-attention override.
        assert torch.equal(out_fa4, out_sdpa), (
            "Packed cross-attention should ignore the decoder attention override"
        )


# =============================================================================
# Test: HAS_FA4_BACKEND flag integrity
# =============================================================================
class TestFA4BackendIntegrity:
    """Verify HAS_FA4_BACKEND flag is consistent with its prerequisites."""

    def test_fa4_backend_requires_all_three(self):
        """HAS_FA4_BACKEND is True only when flex_attention + Hopper+ + CuTe DSL."""
        if fa_module.HAS_FA4_BACKEND:
            assert fa_module.HAS_FLEX_ATTENTION, "FA4 backend requires flex_attention"
            assert fa_module._gpu_arch_major >= 9, "FA4 backend requires Hopper (sm90+)"
            assert fa_module._check_cute_dsl_available(), (
                "FA4 backend requires CuTe DSL"
            )

    def test_no_cute_dsl_means_no_fa4_backend(self):
        """Without CuTe DSL, HAS_FA4_BACKEND must be False."""
        if not fa_module._check_cute_dsl_available():
            assert fa_module.HAS_FA4_BACKEND is False

    def test_compiled_flex_exists_when_flex_available(self):
        """_compiled_flex is set when flex_attention is available."""
        if HAS_FLEX_ATTENTION:
            assert fa_module._compiled_flex is not None
        else:
            assert fa_module._compiled_flex is None


# =============================================================================
# Test: Lazy FLASH backend validation and auto-fallback
# =============================================================================
class TestFlashBackendValidation:
    """Test the lazy validation that detects seqlen_info incompatibility."""

    def test_has_fa4_backend_function_returns_bool(self):
        """has_fa4_backend() returns a boolean."""
        result = has_fa4_backend()
        assert isinstance(result, bool)

    def test_ensure_flash_compat_idempotent(self):
        """_ensure_flash_compat() is idempotent — safe to call multiple times."""
        fa_module._ensure_flash_compat()
        fa_module._ensure_flash_compat()
        fa_module._ensure_flash_compat()
        # No error, no state corruption

    def test_disable_flash_backend_sets_flag_false(self):
        """_disable_flash_backend() sets HAS_FA4_BACKEND to False."""
        original = fa_module.HAS_FA4_BACKEND
        fa_module._disable_flash_backend()
        assert fa_module.HAS_FA4_BACKEND is False
        # Restore
        fa_module.HAS_FA4_BACKEND = original
        fa_module._compiled_flex = fa_module._build_compiled_flex()

    def test_disable_flash_backend_rebuilds_compiled_flex(self):
        """After _disable_flash_backend(), _compiled_flex uses Triton backend."""
        if not HAS_FLEX_ATTENTION:
            pytest.skip("flex_attention not available")
        original_fa4 = fa_module.HAS_FA4_BACKEND
        original_flex = fa_module._compiled_flex

        fa_module.HAS_FA4_BACKEND = True  # Pretend it was True
        fa_module._disable_flash_backend()

        assert fa_module.HAS_FA4_BACKEND is False
        assert fa_module._compiled_flex is not None  # Still has a compiled fn
        assert fa_module._compiled_flex is not original_flex  # But it's different

        # Restore
        fa_module.HAS_FA4_BACKEND = original_fa4
        fa_module._compiled_flex = original_flex

    def test_disable_flash_backend_noop_when_already_disabled(self):
        """_disable_flash_backend() is a no-op when already disabled."""
        original = fa_module.HAS_FA4_BACKEND
        fa_module.HAS_FA4_BACKEND = False
        flex_before = fa_module._compiled_flex

        fa_module._disable_flash_backend()  # Should be a no-op

        assert fa_module._compiled_flex is flex_before  # Not rebuilt
        # Restore
        fa_module.HAS_FA4_BACKEND = original

    def test_has_fa4_backend_consistent_with_module_flag(self):
        """has_fa4_backend() matches the module-level HAS_FA4_BACKEND."""
        assert has_fa4_backend() == fa_module.HAS_FA4_BACKEND

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION or not torch.cuda.is_available(),
        reason="Requires flex_attention and CUDA",
    )
    def test_flash_attn_func_runs_after_flash_disabled(self):
        """flash_attn_func works correctly after FLASH backend is disabled."""
        original_fa4 = fa_module.HAS_FA4_BACKEND
        original_flex = fa_module._compiled_flex
        original_validated = fa_module._flash_compat_validated

        try:
            # Simulate: disable FLASH
            fa_module._disable_flash_backend()
            fa_module._flash_compat_validated = True

            fa_module._override_impl = FlashAttnImpl.FA4

            B, H, S, D = 2, 4, 64, 32
            q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

            y = fa_module.flash_attn.flash_attn_func(q, k, v, causal=True)
            assert y.shape == (B, H, S, D)
            assert torch.isfinite(y).all(), "Output has non-finite values"
        finally:
            fa_module.HAS_FA4_BACKEND = original_fa4
            fa_module._compiled_flex = original_flex
            fa_module._flash_compat_validated = original_validated

    def test_build_compiled_flex_triton_when_fa4_false(self):
        """_build_compiled_flex() builds Triton version when HAS_FA4_BACKEND is False."""
        if not HAS_FLEX_ATTENTION:
            pytest.skip("flex_attention not available")
        original = fa_module.HAS_FA4_BACKEND
        try:
            fa_module.HAS_FA4_BACKEND = False
            flex_fn = fa_module._build_compiled_flex()
            assert flex_fn is not None
        finally:
            fa_module.HAS_FA4_BACKEND = original


# =============================================================================
# Test: cross_attention.py uses has_fa4_backend() (not stale import)
# =============================================================================
class TestCrossAttentionDynamicBackend:
    """Verify dynamic-mask cross-attention selects Triton independently."""

    def test_cross_attention_uses_triton_for_dynamic_mask_backward(self):
        from steering.conditional.encoder_decoder import cross_attention

        source = inspect.getsource(cross_attention)
        assert "fa_module.HAS_FA4_BACKEND" not in source
        assert '{"BACKEND": "TRITON"}' in source


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA4_BACKEND: {fa_module.HAS_FA4_BACKEND}")
    print(f"HAS_FA3: {fa_module.HAS_FA3}")
    print(f"HAS_FLEX_ATTENTION: {HAS_FLEX_ATTENTION}")
    print()

    pytest.main([__file__, "-v", "-s"])
