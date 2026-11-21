"""Regression tests for P0 KV-cache / RoPE safety bugs in nanochat/gpt.py and flash_attention.py.

These tests lock in the fixes for three P0 findings from Phase 1 audit 01:

1. SDPA fallback silently corrupts cache for batch elements >=1 when
   ``cache_seqlens`` has varying values (flash_attention.py:521).
2. ``forward`` only asserts ``T <= cos.size(1)`` — fails to catch the common
   long-generation case ``T0 + T > cos.size(1)`` (gpt.py:540).
3. ``CausalSelfAttention.forward`` advances the KV cache at the last layer
   even when an exception is raised mid-forward, corrupting cache position
   for subsequent calls (gpt.py:151-153).
"""

from __future__ import annotations

import os
from unittest.mock import patch

# Force SDPA fallback path before importing flash_attention.
os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.flash_attention import flash_attn_with_kvcache
from nanochat.gpt import GPT, norm
from tests.conftest import (
    TINY_SEQ_LEN,
    TINY_VOCAB,
    _gpt_block,
    _gpt_blocks,
    _gpt_embedding,
    _make_gpt,
    _make_gpt_config,
)


def _reference_gpt_forward_success(
    model: GPT,
    idx: torch.Tensor,
    kv_cache: KVCache | None = None,
) -> torch.Tensor:
    """Reference copy of GPT.forward's successful inference path.

    This intentionally omits the rollback try/except. On non-exceptional
    forwards, logits and KV-cache side effects must be bit-identical to the
    production forward.
    """
    _, T = idx.size()
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    assert T0 + T <= model.cos.size(1)
    cos_sin = (
        model.cos[:, T0 : T0 + T],
        model.sin[:, T0 : T0 + T],
    )

    x = _gpt_embedding(model)(idx)
    x = x.to(COMPUTE_DTYPE)
    x = norm(x)
    x0 = x
    for i, block in enumerate(_gpt_blocks(model)):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        x = x.to(COMPUTE_DTYPE)
        ve = (
            model.value_embeds[str(i)](idx).to(x.dtype)
            if str(i) in model.value_embeds
            else None
        )
        x = block(x, ve, cos_sin, model.window_sizes[i], kv_cache)
    x = norm(x)

    softcap = 15
    logits = model.lm_head(x)
    logits = logits[..., : model.config.vocab_size]
    logits = logits.float()
    return softcap * torch.tanh(logits / softcap)


# ---------------------------------------------------------------------------
# Fix 1: SDPA fallback rejects heterogeneous cache_seqlens
# ---------------------------------------------------------------------------
class TestSDPAFallbackHeterogeneousCacheSeqlens:
    """When ``cache_seqlens`` has varying positions across the batch, the SDPA
    fallback used a single ``cache_seqlens[0].item()`` for the write offset,
    silently corrupting batch elements >=1. The fix raises
    NotImplementedError instead of producing wrong output."""

    def _build_inputs(self, cache_positions: list[int]):
        """Return q, k_cache, v_cache, k, v, cache_seqlens sized for SDPA path."""
        B = len(cache_positions)
        T_new = 1
        H = 2
        H_kv = 2
        D = 32
        T_max = 32
        dtype = torch.float32  # CPU-compatible (matches model default on CPU)
        q = torch.randn(B, T_new, H, D, dtype=dtype)
        k_cache = torch.zeros(B, T_max, H_kv, D, dtype=dtype)
        v_cache = torch.zeros(B, T_max, H_kv, D, dtype=dtype)
        k = torch.randn(B, T_new, H_kv, D, dtype=dtype)
        v = torch.randn(B, T_new, H_kv, D, dtype=dtype)
        cache_seqlens = torch.tensor(cache_positions, dtype=torch.int32)
        return q, k_cache, v_cache, k, v, cache_seqlens

    def test_varying_cache_seqlens_raises_not_implemented(self):
        """Batch with varying cache positions must raise, not silently corrupt."""
        # Force SDPA path: set the override to SDPA explicitly.
        prev_override = fa_module._override_impl
        fa_module._override_impl = fa_module.FlashAttnImpl.SDPA
        try:
            q, k_cache, v_cache, k, v, cache_seqlens = self._build_inputs([3, 7])
            with pytest.raises(
                NotImplementedError,
                match="SDPA fallback requires uniform cache_seqlens",
            ):
                flash_attn_with_kvcache(
                    q,
                    k_cache,
                    v_cache,
                    k=k,
                    v=v,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                    window_size=(-1, -1),
                )
        finally:
            fa_module._override_impl = prev_override

    def test_uniform_cache_seqlens_still_works(self):
        """Uniform cache_seqlens must NOT trigger the error — regression check."""
        prev_override = fa_module._override_impl
        fa_module._override_impl = fa_module.FlashAttnImpl.SDPA
        try:
            q, k_cache, v_cache, k, v, cache_seqlens = self._build_inputs([5, 5])
            out = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(-1, -1),
            )
            assert out.shape == q.shape
            assert torch.isfinite(out).all()
        finally:
            fa_module._override_impl = prev_override

    def test_single_element_batch_unaffected(self):
        """Batch size 1 trivially uniform — must not regress."""
        prev_override = fa_module._override_impl
        fa_module._override_impl = fa_module.FlashAttnImpl.SDPA
        try:
            q, k_cache, v_cache, k, v, cache_seqlens = self._build_inputs([4])
            out = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(-1, -1),
            )
            assert out.shape == q.shape
        finally:
            fa_module._override_impl = prev_override


# ---------------------------------------------------------------------------
# Fix 2: RoPE bounds assert uses T0+T, not just T
# ---------------------------------------------------------------------------
class TestRoPEBoundsAssert:
    """``GPT.forward`` originally asserted ``T <= cos.size(1)`` but then sliced
    ``cos[:, T0:T0+T]``. When ``T0 + T > cos.size(1)`` the slice silently
    truncates to fewer than T rows, leaving RoPE dropping positions. The fix
    asserts the stronger ``T0 + T <= cos.size(1)`` with an informative message.
    """

    def _make_model_with_small_rope_cache(self) -> GPT:
        """Build a tiny GPT and manually shrink the RoPE cache so we can hit
        the overflow with modest token counts."""
        config = _make_gpt_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        model = model.to_empty(device="cpu")
        model.init_weights()
        model.eval()

        # Shrink the RoPE cache to exactly 16 positions.
        head_dim = config.n_embd // config.n_head
        cos, sin = model._precompute_rotary_embeddings(16, head_dim, device=torch.device("cpu"))
        model.cos, model.sin = cos, sin
        assert model.cos.size(1) == 16
        return model

    def _make_kv_cache(self, model: GPT, batch_size: int = 1) -> KVCache:
        config = model.config
        head_dim = config.n_embd // config.n_head
        return KVCache(
            batch_size=batch_size,
            num_heads=config.n_kv_head,
            seq_len=TINY_SEQ_LEN,
            head_dim=head_dim,
            num_layers=config.n_layer,
            device=torch.device("cpu"),
            dtype=next(model.parameters()).dtype,
        )

    def test_asserts_when_T0_plus_T_exceeds_cache(self):
        """With cos.size(1) = 16 and T0 = 10, T = 8 must fail (10 + 8 > 16)."""
        model = self._make_model_with_small_rope_cache()
        cache = self._make_kv_cache(model)

        # Prefill 10 tokens — this is within the cache (0 + 10 <= 16).
        tokens_prefill = torch.randint(2, TINY_VOCAB, (1, 10))
        with torch.inference_mode():
            model(tokens_prefill, kv_cache=cache)
        assert cache.get_pos() == 10

        # Now try decoding 8 more tokens: T0 + T = 10 + 8 = 18 > 16. Must assert.
        tokens_decode = torch.randint(2, TINY_VOCAB, (1, 8))
        with torch.inference_mode():
            with pytest.raises(AssertionError, match=r"RoPE cache too small"):
                model(tokens_decode, kv_cache=cache)

    def test_assert_message_includes_positions(self):
        """Error message must include T0, T, and cache size to aid debugging."""
        model = self._make_model_with_small_rope_cache()
        cache = self._make_kv_cache(model)

        tokens_prefill = torch.randint(2, TINY_VOCAB, (1, 12))
        with torch.inference_mode():
            model(tokens_prefill, kv_cache=cache)

        tokens_decode = torch.randint(2, TINY_VOCAB, (1, 8))
        with torch.inference_mode():
            with pytest.raises(AssertionError) as excinfo:
                model(tokens_decode, kv_cache=cache)
        msg = str(excinfo.value)
        assert "T0" in msg and "T" in msg
        assert "12" in msg  # T0
        assert "8" in msg  # T
        assert "16" in msg  # cache size

    def test_boundary_T0_plus_T_equals_cache_ok(self):
        """Exactly filling the cache (T0+T == cache size) must still succeed."""
        model = self._make_model_with_small_rope_cache()
        cache = self._make_kv_cache(model)

        # Prefill 8, decode 8: 8 + 8 = 16, which is == cos.size(1). Must succeed.
        with torch.inference_mode():
            model(torch.randint(2, TINY_VOCAB, (1, 8)), kv_cache=cache)
            assert cache.get_pos() == 8
            model(torch.randint(2, TINY_VOCAB, (1, 8)), kv_cache=cache)
            assert cache.get_pos() == 16

    def test_T_only_fits_but_T0_plus_T_overflows(self):
        """Regression: the old assert ``T <= cos.size(1)`` passed for T=1 even
        if T0 had advanced the overall position past the cache. Ensure the new
        check catches that."""
        model = self._make_model_with_small_rope_cache()
        cache = self._make_kv_cache(model)

        # Advance to the edge: 15 prefilled, T=1 decode = 16 (ok).
        with torch.inference_mode():
            model(torch.randint(2, TINY_VOCAB, (1, 15)), kv_cache=cache)
            model(torch.randint(2, TINY_VOCAB, (1, 1)), kv_cache=cache)
            assert cache.get_pos() == 16

        # Now one more single-token decode — old assert (T=1 <= 16) would pass,
        # but the slice cos[:, 16:17] is empty. New assert must catch this.
        with torch.inference_mode():
            with pytest.raises(AssertionError, match=r"RoPE cache too small"):
                model(torch.randint(2, TINY_VOCAB, (1, 1)), kv_cache=cache)


# ---------------------------------------------------------------------------
# Fix 3: KV cache rollback on last-layer fault
# ---------------------------------------------------------------------------
class TestKVCacheRollbackOnLayerFault:
    """``CausalSelfAttention.forward`` advances the KV cache inside the
    last-layer forward, so an exception after ``kv_cache.advance(T)`` but
    before the layer finishes leaves the cache at the wrong position.

    The fix wraps the final layer call in a try/except that rolls back via
    a new ``KVCache.rewind(T)`` helper.
    """

    def _make_model(self) -> GPT:
        return _make_gpt(device="cpu")

    def _make_kv_cache(self, model: GPT) -> KVCache:
        config = model.config
        head_dim = config.n_embd // config.n_head
        return KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=TINY_SEQ_LEN,
            head_dim=head_dim,
            num_layers=config.n_layer,
            device=torch.device("cpu"),
            dtype=next(model.parameters()).dtype,
        )

    def test_rewind_helper_exists_and_decrements(self):
        """KVCache.rewind(T) must be available and decrement cache_seqlens by T."""
        model = self._make_model()
        cache = self._make_kv_cache(model)

        cache.advance(5)
        assert cache.get_pos() == 5
        cache.rewind(3)
        assert cache.get_pos() == 2
        cache.rewind(2)
        assert cache.get_pos() == 0

    def test_rewind_full_advance_returns_to_zero(self):
        model = self._make_model()
        cache = self._make_kv_cache(model)
        cache.advance(7)
        cache.rewind(7)
        assert cache.get_pos() == 0

    def test_cache_position_unchanged_after_last_layer_fault(self):
        """If the last layer's MLP raises (i.e., after advance was called in
        the attention block), the cache must roll back so that a re-run starts
        from the same position."""
        model = self._make_model()
        cache = self._make_kv_cache(model)

        tokens_prefill = torch.randint(2, TINY_VOCAB, (1, 4))
        with torch.inference_mode():
            model(tokens_prefill, kv_cache=cache)
        pos_before = cache.get_pos()
        assert pos_before == 4

        # Patch the last block's MLP to raise, so the advance happens inside
        # the attention forward but then the MLP blows up.
        last_block = _gpt_block(model, -1)

        class _Boom(RuntimeError):
            pass

        def bad_forward(x: torch.Tensor) -> torch.Tensor:
            raise _Boom("simulated OOM in last layer MLP")

        with patch.object(last_block.mlp, "forward", bad_forward):
            tokens_decode = torch.randint(2, TINY_VOCAB, (1, 3))
            with torch.inference_mode():
                with pytest.raises(_Boom):
                    model(tokens_decode, kv_cache=cache)
            # Cache position must be unchanged after the fault.
            assert cache.get_pos() == pos_before, (
                f"Cache advanced despite last-layer fault: {cache.get_pos()} != {pos_before}"
            )

    def test_cache_position_unchanged_after_last_layer_proj_fault(self):
        """Fault in the last layer's c_proj (after advance in attention) must
        also roll back cache position."""
        model = self._make_model()
        cache = self._make_kv_cache(model)

        with torch.inference_mode():
            model(torch.randint(2, TINY_VOCAB, (1, 2)), kv_cache=cache)
        pos_before = cache.get_pos()
        assert pos_before == 2

        last_block = _gpt_block(model, -1)
        original_c_proj = last_block.attn.c_proj

        class _Boom(RuntimeError):
            pass

        def bad_forward(x: torch.Tensor) -> torch.Tensor:
            raise _Boom("simulated fault in c_proj")

        # Replace c_proj's __call__ by swapping its forward method.
        with patch.object(original_c_proj, "forward", bad_forward):
            with torch.inference_mode():
                with pytest.raises(_Boom):
                    model(torch.randint(2, TINY_VOCAB, (1, 2)), kv_cache=cache)
            assert cache.get_pos() == pos_before, (
                f"Cache advanced despite last-layer c_proj fault: {cache.get_pos()} != {pos_before}"
            )

    def test_happy_path_still_advances(self):
        """Regression: the rollback guard must NOT prevent advance on success."""
        model = self._make_model()
        cache = self._make_kv_cache(model)
        with torch.inference_mode():
            model(torch.randint(2, TINY_VOCAB, (1, 3)), kv_cache=cache)
        assert cache.get_pos() == 3
        with torch.inference_mode():
            model(torch.randint(2, TINY_VOCAB, (1, 2)), kv_cache=cache)
        assert cache.get_pos() == 5


class TestGPTForwardReferenceParity:
    """Successful forwards must be bit-identical to a reference implementation."""

    def _make_model_pair(self) -> tuple[GPT, GPT]:
        torch.manual_seed(20260421)
        model = _make_gpt(device="cpu")
        ref_model = _make_gpt(device="cpu")
        ref_model.load_state_dict(model.state_dict())
        model.eval()
        ref_model.eval()
        return model, ref_model

    def _make_kv_cache(self, model: GPT) -> KVCache:
        config = model.config
        head_dim = config.n_embd // config.n_head
        return KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=TINY_SEQ_LEN,
            head_dim=head_dim,
            num_layers=config.n_layer,
            device=torch.device("cpu"),
            dtype=next(model.parameters()).dtype,
        )

    def test_no_cache_logits_match_reference_exactly(self):
        model, ref_model = self._make_model_pair()
        tokens = torch.tensor([[2, 7, 11, 19, 23, 31]], dtype=torch.long)

        with torch.inference_mode():
            actual = model(tokens)
            expected = _reference_gpt_forward_success(ref_model, tokens)

        assert torch.equal(actual, expected)

    def test_kv_cache_logits_and_cache_match_reference_exactly_across_chunks(self):
        model, ref_model = self._make_model_pair()
        cache = self._make_kv_cache(model)
        ref_cache = self._make_kv_cache(ref_model)
        chunks = [
            torch.tensor([[2, 7, 11, 19]], dtype=torch.long),
            torch.tensor([[23]], dtype=torch.long),
            torch.tensor([[31, 37, 41]], dtype=torch.long),
        ]

        with torch.inference_mode():
            for chunk in chunks:
                actual = model(chunk, kv_cache=cache)
                expected = _reference_gpt_forward_success(
                    ref_model, chunk, kv_cache=ref_cache,
                )
                assert torch.equal(actual, expected)
                assert cache.get_pos() == ref_cache.get_pos()
                assert torch.equal(cache.cache_seqlens, ref_cache.cache_seqlens)
                assert torch.equal(cache.k_cache, ref_cache.k_cache)
                assert torch.equal(cache.v_cache, ref_cache.v_cache)

    def test_training_loss_matches_reference_exactly(self):
        model, ref_model = self._make_model_pair()
        tokens = torch.tensor([[2, 7, 11, 19, 23, 31]], dtype=torch.long)
        targets = torch.tensor([[7, 11, 19, 23, 31, 43]], dtype=torch.long)

        with torch.inference_mode():
            actual_logits = model(tokens)
            expected_logits = _reference_gpt_forward_success(ref_model, tokens)

        actual_loss = torch.nn.functional.cross_entropy(
            actual_logits.view(-1, actual_logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="mean",
        )
        expected_loss = torch.nn.functional.cross_entropy(
            expected_logits.view(-1, expected_logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="mean",
        )

        assert torch.equal(actual_loss, expected_loss)
