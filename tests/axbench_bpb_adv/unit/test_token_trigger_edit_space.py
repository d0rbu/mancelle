"""Unit tests for TokenTriggerEditSpace.

These tests exercise the TokenTriggerEditSpace adapter that wraps the existing
token-attack helpers for use with the generic run_gcg loop.  No model or forward
pass is required — only pure bookkeeping over lists of token ids.
"""

from __future__ import annotations

import torch as t

from steering.conditional.feature_gcg.edit_primitives import (
    TokenReplacement,
    token_sequence_digest,
)
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.optimize import TokenTriggerEditSpace, _padded_trigger_tokens


def _min_cfg() -> FasterGcgConfig:
    """A minimal FasterGcgConfig for unit tests (no model calls)."""
    return FasterGcgConfig(
        max_steps=1,
        candidate_batch_size=1,
        top_k=1,
        distance_weight=0.0,
        temperature=1.0,
    )


def _embed() -> t.Tensor:
    """Tiny (10, 3) embedding weight table — small vocab, small dim."""
    vocab, d = 10, 3
    return t.arange(vocab * d, dtype=t.float64).reshape(vocab, d) * 0.1 - 3.0


def test_digest_is_in_the_bos_padded_space(mock_tokenizer):
    space = TokenTriggerEditSpace(tokenizer=mock_tokenizer, cfg=_min_cfg(),
                                  embedding_weight=_embed(), allowed_token_ids=[5, 6, 7])
    trigger = [5, 6]
    bos = mock_tokenizer.get_bos_token_id()
    expected = token_sequence_digest(_padded_trigger_tokens(
        trigger=trigger, bos_token_id=bos, context="t"))
    assert space.digest(trigger) == expected
    # The BARE digest must DIFFER (this is the byte-identity-critical contract).
    assert space.digest(trigger) != token_sequence_digest(trigger)


def test_pool_exhaustion_uses_bos_padded_trigger_digest(mock_tokenizer) -> None:
    space = TokenTriggerEditSpace(
        tokenizer=mock_tokenizer,
        cfg=_min_cfg(),
        embedding_weight=_embed(),
        allowed_token_ids=[5, 6, 7],
    )
    trigger = [5, 6]
    proposals: dict[int, tuple[TokenReplacement, ...]] = {
        1: (TokenReplacement(position=1, token_id=7, score=0.0),),
    }
    padded_candidate = _padded_trigger_tokens(
        trigger=[7, 6],
        bos_token_id=mock_tokenizer.get_bos_token_id(),
        context="test candidate",
    )

    assert not space.full_batch_pool_exhausted(
        proposals,
        trigger,
        {token_sequence_digest([7, 6])},
    )
    assert space.full_batch_pool_exhausted(
        proposals,
        trigger,
        {token_sequence_digest(padded_candidate)},
    )
