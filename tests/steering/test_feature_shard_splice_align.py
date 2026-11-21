"""Tests for re-tokenize + align + attach-features (issue #590, Task B1)."""

from __future__ import annotations

import copy

import pyarrow as pa
import pytest
import torch as t

from nanochat.atomic_io import atomic_write_parquet
from steering.feature_io import glob_features, save_features
from steering.feature_shard_splice import AlignmentError, load_source_segments, splice_feature_shard
from tests.steering._splice_fixtures import doc_seg_ranges, make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)  # BOS-prepending, no merges -> deterministic


def _load(shard, texts, *, segment_size=4, context_window=2048, sae_max_context=None):
    segs, _ = load_source_segments(
        shard, texts, TOK, segment_size=segment_size, context_window=context_window,
        sae_max_context=sae_max_context, shard_name="shard_00000",
    )
    return segs


def test_make_base_shard_fixture_is_valid():
    shard = make_base_shard(["alpha beta gamma", "one two three"], TOK, 4)
    assert int(shard["n_segments"]) > 0
    assert int(shard["n_docs"]) == 2


def test_load_source_segments_attaches_features_and_verifies_alignment():
    texts = ["alpha beta gamma delta", "one two three four five six"]
    shard = make_base_shard(texts, TOK, 4)
    segs, _ = load_source_segments(
        shard, texts, TOK, segment_size=4, context_window=2048,
        sae_max_context=None, shard_name="shard_00000",
    )
    # one entry per source doc, per-doc segment counts match the base shard
    ranges = doc_seg_ranges(shard)
    assert [len(doc_segs) for doc_segs in segs] == [hi - lo for lo, hi in ranges]
    # tokens concatenate back to each doc's kept_ids
    for d, doc_segs in enumerate(segs):
        flat = tuple(tid for s in doc_segs for tid in s.tokens)
        assert flat == tuple(TOK(texts[d], padding=False)["input_ids"])
    # features are the EXACT base CSR slices (t.equal) — feature preservation root
    offsets, fids, fmags = shard["offsets"], shard["feature_ids"], shard["feature_mags"]
    for d, (lo, hi) in enumerate(ranges):
        for j, gi in enumerate(range(lo, hi)):
            seg = segs[d][j]
            assert t.equal(seg.feature_ids, fids[int(offsets[gi]) : int(offsets[gi + 1])])
            assert t.equal(seg.feature_mags, fmags[int(offsets[gi]) : int(offsets[gi + 1])])


def test_truncated_doc_aligns_on_full_length():
    # context_window=4 truncates a long doc; doc_token_counts stores n_full (11),
    # segments cover only the kept 4 tokens. Alignment must use n_full.
    texts = ["abcdefghij"]  # 1 + 10 chars = 11 tokens
    shard = make_base_shard(texts, TOK, 2, context_window=4)
    assert int(shard["doc_token_counts"][0]) == 11  # n_full, not the kept 4
    segs = _load(shard, texts, segment_size=2, context_window=4)
    flat = tuple(tid for s in segs[0] for tid in s.tokens)
    assert flat == (500, ord("a"), ord("b"), ord("c"))  # only the kept prefix


def test_doc_count_mismatch_raises():
    texts = ["alpha beta", "one two three"]
    shard = make_base_shard(texts, TOK, 4)
    with pytest.raises(AlignmentError):
        _load(shard, texts[:1])  # fewer texts than n_docs


def test_doc_token_count_drift_raises():
    texts = ["alpha beta gamma", "one two three four"]
    shard = make_base_shard(texts, TOK, 4)
    bad = copy.deepcopy(shard)
    bad["doc_token_counts"][0] = int(bad["doc_token_counts"][0]) + 1
    with pytest.raises(AlignmentError):
        _load(bad, texts)


def test_segment_position_drift_raises():
    texts = ["alpha beta gamma", "one two three four"]
    shard = make_base_shard(texts, TOK, 4)
    bad = copy.deepcopy(shard)
    bad["segment_positions"][1] = int(bad["segment_positions"][1]) + 1
    with pytest.raises(AlignmentError):
        _load(bad, texts)


def test_segment_indices_not_range_raises():
    texts = ["alpha beta gamma delta epsilon"]  # >= 2 segments in one doc
    shard = make_base_shard(texts, TOK, 4)
    assert int(shard["n_segments"]) >= 2
    bad = copy.deepcopy(shard)
    bad["segment_indices"][0] = 1
    bad["segment_indices"][1] = 0  # not 0,1,.. for the doc
    with pytest.raises(AlignmentError):
        _load(bad, texts)


def test_empty_base_segment_rejected():
    from steering.feature_shard_splice import EmptyBaseSegmentError

    # Hand-built shard with a zero-width segment 1; "abcde" -> [bos,a,b,c,d,e] ->
    # segs [(bos,a,b,c),(d,e)] with segment_size=4. Bypasses make_base_shard's
    # validator on purpose.
    shard = {
        "offsets": t.tensor([0, 2, 2], dtype=t.int64),
        "feature_ids": t.tensor([3, 7], dtype=t.int32),
        "feature_mags": t.tensor([1.0, 2.0], dtype=t.float32),
        "doc_ids": t.tensor([0, 0], dtype=t.int32),
        "segment_indices": t.tensor([0, 1], dtype=t.int32),
        "segment_positions": t.tensor([0, 4], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 4], dtype=t.int32),
        "segment_char_ends": t.tensor([4, 6], dtype=t.int32),
        "doc_token_counts": t.tensor([6], dtype=t.int32),
        "n_docs": 1, "n_segments": 2, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_mode": "fixed_tokens", "segment_size": 4,
        "context_independent_features": False, "ignore_bos_features": True,
    }
    with pytest.raises(EmptyBaseSegmentError):
        _load(shard, ["abcde"], segment_size=4)

def test_splice_feature_shard_raises_alignment_error_on_doc_count_mismatch(tmp_path):
    """splice_feature_shard() propagates AlignmentError when the parquet doc count
    does not match the base shard.

    This tests the per-shard public entry point directly (not through the
    orchestration layer). The orchestration layer (sfs.run()) catches these and
    aggregates them into a RuntimeError, so the orchestration test alone does not
    confirm the specific exception type that splice_feature_shard() raises.

    Why RED is not possible here: the behaviour (AlignmentError propagation) is
    already implemented in load_source_segments — splice_feature_shard() has no
    try/except, so AlignmentError propagates unchanged. Weakening this to a
    broader exception check would be the only way to make a prior version green
    while a fixed version also passes, so the test adds specification value
    rather than regression-catch value.
    """
    texts = ["alpha beta", "one two three", "gamma delta"]
    shard = make_base_shard(texts, TOK, 4)
    save_features(shard, tmp_path / "shard_00000")
    # Write a parquet with fewer docs than the shard expects -> doc-count mismatch
    corrupt_parquet = tmp_path / "shard_00000.parquet"
    atomic_write_parquet(pa.table({"text": texts[:2]}), corrupt_parquet)  # 2 docs, shard has 3
    paths = glob_features(tmp_path)
    assert paths, "save_features must produce a file"
    with pytest.raises(AlignmentError):
        splice_feature_shard(
            paths[0], corrupt_parquet, TOK, model="test-model",
            cut_fraction=0.6, seed=42, shard_index=0, shard_name="shard_00000",
            segment_size=4, context_window=2048, sae_max_context=None,
        )
