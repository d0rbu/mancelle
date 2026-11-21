"""Tests for FIX-11: batched round-trip validation in load_source_segments.

Verifies that the batch restructure produces IDENTICAL results/stats/errors vs
the original per-doc serial path, and that the batch call is exercised (1 batch
call for N docs, not N per-doc calls).

Oracle tests (must pass unmodified after restructure):
  tests/steering/test_feature_shard_splice_trim.py:
    - test_nontruncated_unstable_doc_raises
    - test_trim_stops_at_non_byte_fallback_and_raises
    - test_trim_that_would_empty_last_segment_raises
    - test_stats_zero_on_clean_corpus
    - test_marker_norm_still_applies_before_trim
    - test_marker_norm_and_byte_fallback_trim_combined
  tests/steering/test_feature_shard_splice_align.py:
    - test_make_base_shard_fixture_is_valid
    - test_load_source_segments_attaches_features_and_verifies_alignment
    - test_truncated_doc_aligns_on_full_length
    - test_doc_count_mismatch_raises
    - test_doc_token_count_drift_raises
    - test_segment_position_drift_raises
    - test_segment_indices_not_range_raises
    - test_empty_base_segment_rejected
"""

from __future__ import annotations

from steering.feature_shard_splice import load_source_segments
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer, Gemma3LikeMarkerTokenizer

_BASE_TOK = CharMergeTokenizer(bos_id=500)
_BFID = 400  # synthetic byte-fallback id


class _ByteFallbackTok:
    """Simplified byte-fallback tokenizer for FIX-11 equivalence tests.

    BOS=999. Token _BFID encodes '\\xff'; decode(_BFID)='Z' (unstable round-trip).
    convert_ids_to_tokens: _BFID -> '<0xFF>' so _is_byte_fallback_token recognises it.
    """

    bos_token_id = 999
    _BF = _BFID

    def __call__(self, texts, *, padding=False, add_special_tokens=True):
        def encode_one(text):
            ids = [self.bos_token_id]
            for ch in text:
                ids.append(self._BF if ch == "\xff" else ord(ch))
            return ids

        if isinstance(texts, list):
            return {"input_ids": [encode_one(t_) for t_ in texts]}
        return {"input_ids": encode_one(texts)}

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.bos_token_id:
                continue
            out.append("Z" if i == self._BF else chr(i))
        return "".join(out)

    def convert_ids_to_tokens(self, token_id):
        if isinstance(token_id, (list, tuple)):
            return [self.convert_ids_to_tokens(i) for i in token_id]
        return "<0xFF>" if int(token_id) == self._BF else chr(int(token_id))


_BF_TOK = _ByteFallbackTok()


# ---------------------------------------------------------------------------
# Test (i): equivalence — mixed stable + trim-needing truncated + marker doc
# ---------------------------------------------------------------------------


def test_equivalence_stable_and_trim():
    """Mixed stable + byte-fallback-trim docs: stats identical to serial path.

    Doc 0 ('abcde'): stable, fits fully in context_window=10.
    Doc 1 ('fg\xff' + 'X'*50): long, context_window truncates to [999,'f','g',BF=_BFID].
        was_truncated=True. is_roundtrip_stable fails (BF decodes to 'Z').
        _trim_truncated_doc_tail pops 1 byte-fallback token.

    Expected: n_trimmed_docs=1, n_trimmed_tokens=1, n_marker_normalized_docs=0.
    """
    tok = _BF_TOK
    doc0 = "abcde"
    doc1 = "fg\xff" + "X" * 50
    texts = [doc0, doc1]
    segment_size = 2   # last seg for doc1 = ['g', BF] -> 2 tokens, trim ok
    context_window = 4  # just enough to keep BOS + 'f' + 'g' + BF

    shard = make_base_shard(texts, tok, segment_size=segment_size, context_window=context_window)
    segs, stats = load_source_segments(
        shard, texts, tok,
        segment_size=segment_size, context_window=context_window,
        sae_max_context=None, shard_name="test_equiv",
    )

    assert stats.n_trimmed_docs == 1, f"expected 1 trimmed doc, got {stats.n_trimmed_docs}"
    assert stats.n_trimmed_tokens == 1, f"expected 1 trimmed token, got {stats.n_trimmed_tokens}"
    assert stats.n_marker_normalized_docs == 0
    assert len(segs) == 2


def test_equivalence_three_stable_docs():
    """Three stable docs: zero trims and correct segment structure."""
    texts = ["alpha beta", "gamma delta epsilon", "zeta eta"]
    shard = make_base_shard(texts, _BASE_TOK, segment_size=4, context_window=512)
    segs, stats = load_source_segments(
        shard, texts, _BASE_TOK,
        segment_size=4, context_window=512,
        sae_max_context=None, shard_name="test_three_stable",
    )
    assert stats.n_trimmed_docs == 0
    assert stats.n_trimmed_tokens == 0
    assert stats.n_marker_normalized_docs == 0
    assert len(segs) == 3


def test_equivalence_marker_and_trim():
    """Marker-norm doc + trim-needing doc in same shard: stats match expected."""
    tok = Gemma3LikeMarkerTokenizer(byte_fallback_id=_BFID)
    marker = "▁"
    doc0 = marker + "hello"   # marker normalization
    doc1 = "cd\xff" + "X" * 100  # byte-fallback trim (with context_window truncating)
    texts = [doc0, doc1]
    segment_size = 3
    context_window = 3

    shard = make_base_shard(texts, tok, segment_size=segment_size, context_window=context_window)
    segs, stats = load_source_segments(
        shard, texts, tok,
        segment_size=segment_size, context_window=context_window,
        sae_max_context=None, shard_name="test_marker_trim",
    )

    assert stats.n_marker_normalized_docs == 1
    assert stats.n_trimmed_docs == 1
    assert stats.n_trimmed_tokens == 1
    assert len(segs) == 2


# ---------------------------------------------------------------------------
# Test (ii): batch path is exercised (1 batch validation call, not N per-doc)
# ---------------------------------------------------------------------------


class _CountingTokenizer:
    """Wraps a tokenizer and counts how many times it is called with a list arg."""

    def __init__(self, base) -> None:
        self._b = base
        self.list_call_sizes: list[int] = []  # len(texts) for each list call

    def __call__(self, texts, **kwargs):
        if isinstance(texts, list):
            self.list_call_sizes.append(len(texts))
        return self._b(texts, **kwargs)

    def decode(self, ids):
        return self._b.decode(ids)

    @property
    def bos_token_id(self):
        return self._b.bos_id


def test_batch_stability_makes_one_list_call_for_n_docs():
    """For N stable docs, stability check makes exactly 1 batch list call (not N).

    batch_truncate_to_training_window also makes 1 list call (total = 2 list calls).
    The batch stability call covers all N docs at once.
    """
    n = 4
    texts = [f"doc{i} content here" for i in range(n)]
    tok = _CountingTokenizer(_BASE_TOK)
    shard = make_base_shard(texts, _BASE_TOK, segment_size=4, context_window=512)

    tok.list_call_sizes.clear()
    segs, stats = load_source_segments(
        shard, texts, tok,
        segment_size=4, context_window=512,
        sae_max_context=None, shard_name="test_count",
    )

    assert stats.n_trimmed_docs == 0
    # batch_truncate_to_training_window: 1 list call of size n
    # _batch_roundtrip_stable: 1 list call of size n
    # Total: exactly 2 list calls, each of size n
    assert len(tok.list_call_sizes) == 2, (
        f"expected 2 list calls (truncation + batch stability), "
        f"got {len(tok.list_call_sizes)}: {tok.list_call_sizes}"
    )
    assert all(s == n for s in tok.list_call_sizes), (
        f"all list calls should be size {n}: {tok.list_call_sizes}"
    )
