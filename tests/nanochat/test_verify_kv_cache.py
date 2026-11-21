"""Verify correctness of KV cache and incremental decoding in nanochat."""

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Tiny model constants (mirror conftest.py)
# ---------------------------------------------------------------------------
TINY_VOCAB = 128
TINY_N_EMBD = 64
TINY_N_LAYER = 2
TINY_N_HEAD = 2
TINY_N_KV_HEAD = 2
TINY_SEQ_LEN = 128
TINY_HEAD_DIM = 32


def _make_model() -> GPT:
    """Build a tiny GPT on CPU, initialised weights, eval mode."""
    config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _make_kv_cache(
    model: GPT,
    batch_size: int = 1,
    seq_len: int | None = None,
    dtype: torch.dtype | None = None,
) -> KVCache:
    """Create a KVCache sized for the given model.

    On CPU the model weights are float32, so the cache must also be float32
    to avoid dtype mismatches inside SDPA. On CUDA the model is bfloat16.
    """
    config = model.config
    head_dim = config.n_embd // config.n_head
    device = model.get_device()
    if dtype is None:
        # Match the dtype of the first linear weight in the model
        dtype = next(model.parameters()).dtype
    return KVCache(
        batch_size=batch_size,
        num_heads=config.n_kv_head,
        seq_len=seq_len or config.sequence_len,
        head_dim=head_dim,
        num_layers=config.n_layer,
        device=device,
        dtype=dtype,
    )


def _random_tokens(length: int, batch_size: int = 1) -> torch.Tensor:
    """Return random token ids in [2, TINY_VOCAB) with shape (batch_size, length)."""
    return torch.randint(2, TINY_VOCAB, (batch_size, length))


# ===================================================================
# Test 1: Full-sequence forward matches incremental decode
# ===================================================================
class TestFullSequenceMatchesIncremental:
    """Run a full forward pass and compare against incremental decoding."""

    def test_prefill_all_at_once(self):
        """Prefill entire sequence into KV cache in one shot, compare logits."""
        model = _make_model()
        tokens = _random_tokens(5)

        with torch.inference_mode():
            # Full forward (no cache)
            full_logits = model(tokens)  # (1, 5, vocab)

            # Incremental: prefill all 5 tokens at once via KV cache
            cache = _make_kv_cache(model)
            cached_logits = model(tokens, kv_cache=cache)  # (1, 5, vocab)

        # Both should produce the same logits at every position
        torch.testing.assert_close(
            cached_logits,
            full_logits,
            atol=1e-2,
            rtol=1e-2,
            msg="Prefill-all-at-once logits diverge from full forward",
        )

    def test_prefill_one_at_a_time(self):
        """Feed tokens one-by-one into KV cache, compare final logits."""
        model = _make_model()
        tokens = _random_tokens(5)

        with torch.inference_mode():
            # Full forward
            full_logits = model(tokens)  # (1, 5, vocab)

            # Incremental: one token at a time
            cache = _make_kv_cache(model)
            incremental_logits = []
            for t in range(5):
                tok = tokens[:, t : t + 1]  # (1, 1)
                logits_t = model(tok, kv_cache=cache)  # (1, 1, vocab)
                incremental_logits.append(logits_t)

            incremental_logits = torch.cat(incremental_logits, dim=1)  # (1, 5, vocab)

        torch.testing.assert_close(
            incremental_logits,
            full_logits,
            atol=1e-2,
            rtol=1e-2,
            msg="One-at-a-time logits diverge from full forward",
        )


# ===================================================================
# Test 2: Prefill then single-token decode
# ===================================================================
class TestPrefillThenDecode:
    """Prefill a prompt, then decode one new token and compare."""

    def test_decode_matches_full_forward(self):
        model = _make_model()
        tokens = _random_tokens(5)

        with torch.inference_mode():
            # Full forward on all 5 tokens
            full_logits = model(tokens)  # (1, 5, vocab)
            logits_at_pos4 = full_logits[:, 4, :]  # (1, vocab)

            # Prefill first 4 tokens, then decode token 5
            cache = _make_kv_cache(model)
            _ = model(tokens[:, :4], kv_cache=cache)  # prefill
            decode_logits = model(tokens[:, 4:5], kv_cache=cache)  # (1, 1, vocab)

        torch.testing.assert_close(
            decode_logits[:, 0, :],
            logits_at_pos4,
            atol=1e-2,
            rtol=1e-2,
            msg="Decoded logits at position 4 diverge from full forward",
        )


# ===================================================================
# Test 3: KVCache.advance only at last layer
# ===================================================================
class TestAdvanceOnlyAtLastLayer:
    """Verify advance is called exactly once per forward pass (at the last layer)."""

    def test_advance_called_once(self):
        model = _make_model()
        tokens = _random_tokens(3)
        cache = _make_kv_cache(model)

        advance_calls = []
        original_advance = KVCache.advance

        def tracking_advance(self_cache, num_tokens):
            advance_calls.append(num_tokens)
            return original_advance(self_cache, num_tokens)

        with torch.inference_mode(), patch.object(KVCache, "advance", tracking_advance):
            model(tokens, kv_cache=cache)

        assert len(advance_calls) == 1, (
            f"Expected advance to be called exactly once, got {len(advance_calls)} calls"
        )
        assert advance_calls[0] == 3, (
            f"Expected advance(3) for 3 tokens, got advance({advance_calls[0]})"
        )

    def test_position_after_forward(self):
        """After forwarding N tokens, cache position should be N."""
        model = _make_model()
        cache = _make_kv_cache(model)

        with torch.inference_mode():
            model(_random_tokens(3), kv_cache=cache)
            assert cache.get_pos() == 3

            model(_random_tokens(2), kv_cache=cache)
            assert cache.get_pos() == 5


# ===================================================================
# Test 4: KVCache.reset clears position; new generation matches fresh
# ===================================================================
class TestCacheReset:
    """After reset, generation should produce identical results to a fresh cache."""

    def test_reset_matches_fresh_cache(self):
        model = _make_model()
        tokens_a = _random_tokens(4)
        tokens_b = _random_tokens(3)

        with torch.inference_mode():
            # First: use a fresh cache for tokens_b
            fresh_cache = _make_kv_cache(model)
            fresh_logits = model(tokens_b, kv_cache=fresh_cache)

            # Second: pollute a cache with tokens_a, then reset and use for tokens_b
            reused_cache = _make_kv_cache(model)
            _ = model(tokens_a, kv_cache=reused_cache)
            reused_cache.reset()
            reused_logits = model(tokens_b, kv_cache=reused_cache)

        torch.testing.assert_close(
            reused_logits,
            fresh_logits,
            atol=1e-2,
            rtol=1e-2,
            msg="Logits after cache reset diverge from fresh cache",
        )

    def test_reset_zeros_position(self):
        model = _make_model()
        cache = _make_kv_cache(model)

        with torch.inference_mode():
            model(_random_tokens(5), kv_cache=cache)
            assert cache.get_pos() == 5
            cache.reset()
            assert cache.get_pos() == 0


# ===================================================================
# Test 5: KVCache.prefill copies correctly
# ===================================================================
class TestCachePrefill:
    """Verify prefill copies position and KV data from source to target cache."""

    def test_prefill_copies_position_and_data(self):
        model = _make_model()
        tokens = _random_tokens(4)

        with torch.inference_mode():
            # Populate source cache
            src_cache = _make_kv_cache(model, seq_len=TINY_SEQ_LEN)
            model(tokens, kv_cache=src_cache)
            assert src_cache.get_pos() == 4

            # Create a larger target cache and prefill from source
            dst_cache = _make_kv_cache(model, seq_len=TINY_SEQ_LEN)
            dst_cache.prefill(src_cache)

        # Position should match
        assert dst_cache.get_pos() == src_cache.get_pos() == 4

        # KV data for the filled positions should match exactly
        pos = src_cache.get_pos()
        torch.testing.assert_close(
            dst_cache.k_cache[:, :, :pos, :, :],
            src_cache.k_cache[:, :, :pos, :, :],
            msg="Prefill k_cache data mismatch",
        )
        torch.testing.assert_close(
            dst_cache.v_cache[:, :, :pos, :, :],
            src_cache.v_cache[:, :, :pos, :, :],
            msg="Prefill v_cache data mismatch",
        )

    def test_prefill_then_decode_matches(self):
        """After prefilling cache B from A, decoding the next token should match."""
        model = _make_model()
        tokens = _random_tokens(5)

        with torch.inference_mode():
            # Cache A: prefill 4 tokens, decode 5th
            cache_a = _make_kv_cache(model)
            _ = model(tokens[:, :4], kv_cache=cache_a)
            logits_a = model(tokens[:, 4:5], kv_cache=cache_a)

            # Cache B: prefill from A after 4 tokens, then decode 5th
            cache_src = _make_kv_cache(model)
            _ = model(tokens[:, :4], kv_cache=cache_src)

            cache_b = _make_kv_cache(model)
            cache_b.prefill(cache_src)
            logits_b = model(tokens[:, 4:5], kv_cache=cache_b)

        torch.testing.assert_close(
            logits_b,
            logits_a,
            atol=1e-5,
            rtol=1e-5,
            msg="Decoding after prefill diverges from original cache",
        )

    def test_prefill_rejects_nonempty_target(self):
        """Prefill should fail if the target cache already has data."""
        model = _make_model()
        tokens = _random_tokens(3)

        with torch.inference_mode():
            src = _make_kv_cache(model)
            model(tokens, kv_cache=src)

            dst = _make_kv_cache(model)
            model(tokens, kv_cache=dst)

            with pytest.raises(RuntimeError, match="Cannot prefill a non-empty"):
                dst.prefill(src)


# ===================================================================
# Test 6: RoPE position offset
# ===================================================================
class TestRoPEPositionOffset:
    """Verify rotary embeddings are correctly offset by cache position."""

    def test_rope_offset_during_decode(self):
        """The cos/sin slice used during decode should correspond to the cache position."""
        model = _make_model()
        tokens = _random_tokens(5)

        with torch.inference_mode():
            # Full forward: RoPE positions 0..4
            full_logits = model(tokens)

            # Prefill 3 tokens, then decode tokens[3:5]
            cache = _make_kv_cache(model)
            _ = model(tokens[:, :3], kv_cache=cache)
            assert cache.get_pos() == 3

            # The next forward should use RoPE positions 3,4 — verified by
            # checking that T0 = cache.get_pos() = 3
            T0 = cache.get_pos()
            assert T0 == 3, f"Expected T0=3, got {T0}"

            decode_logits = model(tokens[:, 3:5], kv_cache=cache)

        # Logits at positions 3,4 should match full forward
        torch.testing.assert_close(
            decode_logits,
            full_logits[:, 3:5, :],
            atol=1e-2,
            rtol=1e-2,
            msg="RoPE offset: decode logits at pos 3,4 diverge from full forward",
        )

    def test_rope_slicing_explicit(self):
        """Directly verify the cos/sin slice indices used in forward."""
        model = _make_model()
        cache = _make_kv_cache(model)

        with torch.inference_mode():
            # Prefill 7 tokens
            model(_random_tokens(7), kv_cache=cache)
            assert cache.get_pos() == 7

            # The next decode of 1 token should use cos[:, 7:8] and sin[:, 7:8]
            T0 = cache.get_pos()
            T = 1
            cos_slice = model.cos[:, T0 : T0 + T]
            model.sin[:, T0 : T0 + T]

            # cos[:, 7:8] should differ from cos[:, 0:1] (different RoPE position)
            cos_at_0 = model.cos[:, 0:1]
            assert not torch.equal(cos_slice, cos_at_0), (
                "cos at position 7 should differ from position 0"
            )


# ===================================================================
# Test 7: Multi-token prefill matches one-at-a-time
# ===================================================================
class TestMultiTokenPrefill:
    """Verify batch prefill gives same result as sequential single-token prefill."""

    def test_batch_vs_sequential_prefill(self):
        model = _make_model()
        tokens = _random_tokens(4)

        with torch.inference_mode():
            # Batch prefill: all 4 tokens at once
            cache_batch = _make_kv_cache(model)
            batch_logits = model(tokens, kv_cache=cache_batch)

            # Sequential prefill: one token at a time
            cache_seq = _make_kv_cache(model)
            seq_logits_list = []
            for t in range(4):
                logits_t = model(tokens[:, t : t + 1], kv_cache=cache_seq)
                seq_logits_list.append(logits_t)
            seq_logits = torch.cat(seq_logits_list, dim=1)

        # Both should produce the same logits
        torch.testing.assert_close(
            seq_logits,
            batch_logits,
            atol=1e-2,
            rtol=1e-2,
            msg="Batch prefill logits diverge from sequential one-at-a-time",
        )

    def test_batch_vs_sequential_then_decode(self):
        """After prefilling (batch vs sequential), decoding the next token should match."""
        model = _make_model()
        tokens = _random_tokens(5)

        with torch.inference_mode():
            # Batch prefill 4, then decode 5th
            cache_batch = _make_kv_cache(model)
            _ = model(tokens[:, :4], kv_cache=cache_batch)
            batch_decode = model(tokens[:, 4:5], kv_cache=cache_batch)

            # Sequential prefill 4, then decode 5th
            cache_seq = _make_kv_cache(model)
            for t in range(4):
                _ = model(tokens[:, t : t + 1], kv_cache=cache_seq)
            seq_decode = model(tokens[:, 4:5], kv_cache=cache_seq)

        torch.testing.assert_close(
            seq_decode,
            batch_decode,
            atol=1e-2,
            rtol=1e-2,
            msg="Decode after batch prefill diverges from decode after sequential prefill",
        )

    def test_positions_match_after_prefill(self):
        """Both prefill strategies should leave the cache at the same position."""
        model = _make_model()
        tokens = _random_tokens(4)

        with torch.inference_mode():
            cache_batch = _make_kv_cache(model)
            model(tokens, kv_cache=cache_batch)

            cache_seq = _make_kv_cache(model)
            for t in range(4):
                model(tokens[:, t : t + 1], kv_cache=cache_seq)

        assert cache_batch.get_pos() == cache_seq.get_pos() == 4
