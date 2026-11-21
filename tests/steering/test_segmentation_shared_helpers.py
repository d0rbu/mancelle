from __future__ import annotations

import pytest

from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    batch_truncate_to_training_window,
    segment_document,
    segment_tokens,
    token_to_char_offsets,
    truncate_to_training_window,
)
from tests.cross_tokenizer_helpers import SimpleTokenizer


@pytest.mark.parametrize("text,cw,maxc,expect_ids_len", [
    ("", 128, 256, 0),
    ("abc", 128, 256, 3),
    ("abcdefgh", 4, 256, 4),     # truncated to context_window
    ("abcdefgh", 256, 5, 5),     # truncated to sae_max_context (min wins)
    ("abcdef", 10, None, 6),     # None max-context -> context_window only
])
def test_truncate_matches_min_limit(text, cw, maxc, expect_ids_len):
    tok = SimpleTokenizer()
    ids, n_full = truncate_to_training_window(text, tok, cw, maxc)
    assert len(ids) == expect_ids_len
    assert n_full == len(text)
    # ids are a head slice of the full encoding
    assert ids == tok.encode(text)[: (cw if maxc is None else min(cw, maxc))]


def test_truncate_empty_skips_tokenizer():
    class Boom:
        def __call__(self, *a, **k):  # would raise if consulted
            raise AssertionError("tokenizer consulted for empty text")
    assert truncate_to_training_window("", Boom(), 128, 256) == ([], 0)


def test_batch_truncate_bit_identical_to_per_text():
    tok = SimpleTokenizer()
    texts = ["", "abc", "abcdefghij", "x"]
    batched = batch_truncate_to_training_window(texts, tok, 4, 256, chunk_size=2)
    per = [truncate_to_training_window(t, tok, 4, 256) for t in texts]
    assert batched == per


def test_segment_document_matches_inline_reference():
    tok = SimpleTokenizer()
    kept = tok.encode("abcdefghij")  # 10 single-char tokens
    segment_size = PositiveInt.parse(4)
    segs, cs, ce = segment_document(
        kept,
        tok,
        SegmentMode.fixed_tokens,
        segment_size,
    )
    # reference (the exact inline logic from _precompute_batch_cpu_work)
    ref_segs = segment_tokens(kept, SegmentMode.fixed_tokens, segment_size)
    offs = token_to_char_offsets(tok, kept)
    assert segs == ref_segs
    assert cs == [offs[s] for s, _ in ref_segs]
    assert ce == [offs[e] for _, e in ref_segs]
    # char=token tokenizer => char offsets equal token indices
    assert cs == [0, 4, 8] and ce == [4, 8, 10]
