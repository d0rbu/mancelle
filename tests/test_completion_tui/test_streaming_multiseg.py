# tests/test_completion_tui/test_streaming_multiseg.py
"""Tests for streaming generation via generate_completion_stream."""

from __future__ import annotations

import threading

import pytest
import torch as t

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.multiseg_generator import (
    MultisegInputs,
    build_multiseg_inputs,
    generate_completion,
    generate_completion_stream,
)
from tests.conftest import _make_encdec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedWidthTokenizer:
    """Mock tokenizer where each token decodes to exactly 4 characters."""

    def decode(self, token_ids):
        return "x" * (len(token_ids) * 4)


_MOCK_TOK = _FixedWidthTokenizer()


def _make_segments(n_segs: int = 2, features_per_seg: int = 3) -> list[SegmentFeatures]:
    """Create synthetic segments with known features (int64 bucket indices)."""
    segs = []
    for i in range(n_segs):
        fids = t.arange(i * 10, i * 10 + features_per_seg, dtype=t.int32)
        mags = t.ones(features_per_seg, dtype=t.int64) * (i + 1)
        segs.append(
            SegmentFeatures(
                segment_idx=i,
                position=i * 128,
                feature_ids=fids,
                feature_mags=mags,
                char_start=i * 512,
                char_end=(i + 1) * 512,
            )
        )
    return segs


TINY_VOCAB = 128


def _build_test_inputs(
    n_segs: int = 2,
    features_per_seg: int = 3,
    n_prefix: int = 20,
    n_gen: int = 10,
) -> tuple[list[int], MultisegInputs]:
    """Build prefix tokens and MultisegInputs for testing.

    Uses small token counts and IDs within [2, TINY_VOCAB) to avoid
    embedding index-out-of-range errors with the tiny test model.
    """
    segs = _make_segments(n_segs, features_per_seg)
    # Keep token IDs in valid range for tiny model (vocab=128, skip 0=BOS, 1=EOS)
    prefix_tokens = [i % (TINY_VOCAB - 2) + 2 for i in range(n_prefix)]
    encoder_inputs = build_multiseg_inputs(
        segments=segs,
        prefix_tokens=prefix_tokens,
        n_gen_tokens=n_gen,
        target_segment_idx=n_segs - 1,
        tokenizer=_MOCK_TOK,
    )
    return prefix_tokens, encoder_inputs


# ---------------------------------------------------------------------------
# Core equivalence test
# ---------------------------------------------------------------------------


class TestStreamBatchEquivalence:
    """The most important test: streaming must produce identical output to batch."""

    @pytest.fixture()
    def model(self):
        return _make_encdec(device="cpu")

    def test_stream_equals_batch_default_params(self, model):
        """Same inputs + seed → identical token sequences."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        device = t.device("cpu")
        max_new = 10  # must match n_gen in _build_test_inputs
        seed = 42

        batch_tokens = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new,
            device=device,
            seed=seed,
        )

        stream_tokens = list(
            generate_completion_stream(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new,
                device=device,
                seed=seed,
            )
        )

        assert stream_tokens == batch_tokens

    def test_stream_equals_batch_greedy(self, model):
        """Greedy (temperature=0) must also match."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        device = t.device("cpu")
        max_new = 10

        batch_tokens = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new,
            device=device,
            temperature=0.0,
            top_k=None,
        )

        stream_tokens = list(
            generate_completion_stream(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new,
                device=device,
                temperature=0.0,
                top_k=None,
            )
        )

        assert stream_tokens == batch_tokens

    def test_stream_equals_batch_different_seeds(self, model):
        """Multiple seeds all produce matching results."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        device = t.device("cpu")
        max_new = 10

        for seed in [0, 1, 123, 9999]:
            batch_tokens = generate_completion(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new,
                device=device,
                seed=seed,
            )

            stream_tokens = list(
                generate_completion_stream(
                    model=model,
                    prefix_tokens=prefix_tokens,
                    encoder_inputs=encoder_inputs,
                    max_new_tokens=max_new,
                    device=device,
                    seed=seed,
                )
            )

            assert stream_tokens == batch_tokens, f"Mismatch for seed={seed}"


# ---------------------------------------------------------------------------
# Cancellation tests
# ---------------------------------------------------------------------------


class TestCancellation:
    """Test cancel_event stops generation correctly."""

    @pytest.fixture()
    def model(self):
        return _make_encdec(device="cpu")

    def test_cancel_before_start_yields_nothing(self, model):
        """Cancel event already set → 0 tokens."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        cancel = threading.Event()
        cancel.set()  # pre-cancelled

        tokens = list(
            generate_completion_stream(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=10,
                device=t.device("cpu"),
                cancel_event=cancel,
            )
        )

        assert tokens == []

    def test_cancel_after_n_tokens(self, model):
        """Cancel after exactly 3 tokens → exactly 3 tokens yielded."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        cancel = threading.Event()
        device = t.device("cpu")
        max_new = 10

        collected: list[int] = []
        gen = generate_completion_stream(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new,
            device=device,
            cancel_event=cancel,
        )

        for tok in gen:
            collected.append(tok)
            if len(collected) == 3:
                cancel.set()

        assert len(collected) == 3

    def test_no_cancel_generates_up_to_max(self, model):
        """No cancel event → generates up to max_new_tokens or bos_id."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        device = t.device("cpu")
        max_new = 10

        tokens = list(
            generate_completion_stream(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new,
                device=device,
                cancel_event=None,
            )
        )

        # Should produce some tokens (up to max_new), same as batch
        batch_tokens = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new,
            device=device,
        )
        assert tokens == batch_tokens


# ---------------------------------------------------------------------------
# Precomputed encoder tests
# ---------------------------------------------------------------------------


class TestPrecomputedEncoder:
    """Test that precomputed_encoder path works identically."""

    @pytest.fixture()
    def model(self):
        return _make_encdec(device="cpu")

    def test_precomputed_matches_fresh(self, model):
        """precomputed_encoder produces same output as fresh encoder computation."""
        prefix_tokens, encoder_inputs = _build_test_inputs()
        device = t.device("cpu")
        max_new = 10
        seed = 42

        # Compute encoder output manually
        enc_fids = encoder_inputs.enc_feature_ids.to(device)
        enc_mags = encoder_inputs.enc_magnitudes.to(device)
        enc_mask = encoder_inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

        fresh_tokens = list(
            generate_completion_stream(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new,
                device=device,
                seed=seed,
            )
        )

        precomputed_tokens = list(
            generate_completion_stream(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new,
                device=device,
                seed=seed,
                precomputed_encoder=encoder_out,
            )
        )

        assert fresh_tokens == precomputed_tokens
