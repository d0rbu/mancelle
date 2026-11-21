"""Tests that zero-feature docs survive segment accumulation."""

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    _build_row_segments,
    _pad_segments,
)
from steering.extract_features import pool_segments_batched_gpu
from steering.feature_paths import PoolingMode


class SimpleTokenizer:
    """Tokenizer where each ASCII character is a separate token."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids)


def test_pool_segments_rejects_empty():
    """pool_segments_batched_gpu asserts on empty segments — confirming the
    accumulation loop must guard against this."""
    sae_acts = t.randn(10, 100)
    with pytest.raises(AssertionError, match="non-empty"):
        pool_segments_batched_gpu(sae_acts, [], PoolingMode.max_pool)


def test_char_level_assignment_handles_zero_segments():
    """_build_row_segments handles docs with seg_features=None (0 segments)."""
    tokenizer = SimpleTokenizer()
    T = 32

    # One doc at position 0 with no features (None)
    # doc_tokens = list of token ids for the text
    doc_tokens = list(range(ord("A"), ord("A") + T))
    row_docs = [(0, None, doc_tokens)]

    per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map, seg_doc_id, _, _ = _build_row_segments(
        row_docs,
        T,
        tokenizer,
    )
    feature_ids, magnitudes, _, mask, seg_map = _pad_segments(
        per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map,
    )

    # Should produce a single empty segment covering all positions
    assert feature_ids.shape[0] == 1  # one empty segment
    assert (
        feature_ids.shape[1] == 0 or (feature_ids == 0).all()
    )  # no features or padded zeros
    assert seg_map.tolist() == [0] * T  # all positions map to segment 0
