"""Tests for doc index correctness in _raw_document_iterator.

The iterator walks parquet files row-group-by-row-group and must pass a
**global** document index (0-based within the entire shard file) to
``feature_index.lookup()``.  A previous bug used ``enumerate(texts)``
which resets to 0 for each row group.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.preprocess.precompute_packed_rows import _raw_document_iterator
from steering.conditional.feature_index import SegmentFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_segment(doc_idx: int) -> SegmentFeatures:
    """A minimal present-doc segment (extraction guarantees >= 1 per doc)."""
    return SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor([doc_idx % 7], dtype=t.int32),
        feature_mags=t.tensor([1.0], dtype=t.float32),
        char_start=0,
        char_end=1,
        feature_percentiles=None,
        token_start=0,
        token_end=1,
    )


class RecordingFeatureIndex:
    """Mock FeatureIndex that records every (shard_stem, doc_idx) lookup.

    Mirrors the real ``FeatureIndex.lookup`` contract: returns a non-empty
    ``list[SegmentFeatures]`` for a present doc and RAISES ``KeyError`` for an
    absent one. By default every doc is present (lazily synthesized) so the
    doc-index tests below can consume the full iterator; pass ``present_docs``
    to restrict the present set and exercise the corruption (missing-doc) path.
    """

    def __init__(self, present_docs: set[int] | None = None):
        self.calls: list[tuple[str, int]] = []
        # None ⇒ every doc is present (synthesized on demand).
        self._present_docs = present_docs

    def lookup(self, shard_stem: str, doc_idx: int) -> list[SegmentFeatures]:
        self.calls.append((shard_stem, doc_idx))
        if self._present_docs is not None and doc_idx not in self._present_docs:
            raise KeyError(
                f"Document {doc_idx} not found in shard {shard_stem!r} "
                f"(shard has {len(self._present_docs)} docs)"
            )
        return [_one_segment(doc_idx)]


class SimpleTokenizer:
    """Minimal word-level tokenizer matching the one in test_precompute_packed_rows_integration."""

    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens


def _make_multi_rg_parquet(
    path: Path,
    n_docs_per_rg: int,
    n_row_groups: int,
) -> list[str]:
    """Write a parquet file with *n_row_groups* row groups of *n_docs_per_rg* docs each.

    Returns the list of texts in global order.
    """
    total = n_docs_per_rg * n_row_groups
    texts = [
        f"doc{i} " + " ".join(f"w{w}" for w in range(5 + i % 7))
        for i in range(total)
    ]
    table = pa.table({"text": texts})
    pq.write_table(table, str(path), row_group_size=n_docs_per_rg)
    return texts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDocIndexGlobalOffset:
    """Verify _raw_document_iterator passes global doc indices to feature_index.lookup."""

    def test_global_indices_across_three_row_groups(self, tmp_path):
        """With 3 RGs of 10 docs each, lookup must receive indices 0-29 (not 0-9 three times)."""
        n_per_rg = 10
        n_rgs = 3
        pq_path = tmp_path / "shard_00000.parquet"
        _make_multi_rg_parquet(pq_path, n_per_rg, n_rgs)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()

        # Consume the full iterator
        list(_raw_document_iterator([str(pq_path)], tok, bos, rec))

        actual_indices = [doc_idx for _, doc_idx in rec.calls]
        expected_indices = list(range(n_per_rg * n_rgs))
        assert actual_indices == expected_indices, (
            f"Expected global indices {expected_indices[:15]}..., "
            f"got {actual_indices[:15]}..."
        )

    def test_rg1_first_doc_uses_offset(self, tmp_path):
        """Row group 1's first doc must use global index n_per_rg, not 0."""
        n_per_rg = 8
        pq_path = tmp_path / "shard_00000.parquet"
        _make_multi_rg_parquet(pq_path, n_per_rg, 3)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

        indices = [d for _, d in rec.calls]

        # First doc in RG 1 should be index 8, not 0
        assert indices[n_per_rg] == n_per_rg, (
            f"RG 1 first doc: expected global index {n_per_rg}, got {indices[n_per_rg]}"
        )
        # First doc in RG 2 should be index 16, not 0
        assert indices[2 * n_per_rg] == 2 * n_per_rg, (
            f"RG 2 first doc: expected global index {2 * n_per_rg}, got {indices[2 * n_per_rg]}"
        )

    def test_no_duplicate_indices_across_row_groups(self, tmp_path):
        """Each global index must appear exactly once — duplicates mean per-RG reset."""
        n_per_rg = 5
        pq_path = tmp_path / "shard_00000.parquet"
        _make_multi_rg_parquet(pq_path, n_per_rg, 4)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

        indices = [d for _, d in rec.calls]
        assert len(indices) == len(set(indices)), (
            f"Duplicate doc indices found: the index resets per row group. "
            f"First 15 indices: {indices[:15]}"
        )

    def test_single_row_group_no_regression(self, tmp_path):
        """With one RG, local == global. Must not break existing behavior."""
        n_docs = 12
        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} hello world foo" for i in range(n_docs)]
        table = pa.table({"text": texts})
        # row_group_size >= n_docs → single row group
        pq.write_table(table, str(pq_path), row_group_size=n_docs + 100)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

        actual = [d for _, d in rec.calls]
        assert actual == list(range(n_docs))

    def test_multiple_parquet_files_each_reset_to_zero(self, tmp_path):
        """Each parquet file is a separate shard — global offset resets between files."""
        for shard_idx in range(2):
            pq_path = tmp_path / f"shard_{shard_idx:05d}.parquet"
            _make_multi_rg_parquet(pq_path, 5, 3)  # 15 docs, 3 RGs

        paths = sorted(str(p) for p in tmp_path.glob("shard_*.parquet"))
        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator(paths, tok, tok.get_bos_token_id(), rec))

        # Shard 0: indices 0-14, Shard 1: indices 0-14 (reset per shard)
        shard0_indices = [d for stem, d in rec.calls if stem == "shard_00000"]
        shard1_indices = [d for stem, d in rec.calls if stem == "shard_00001"]

        assert shard0_indices == list(range(15)), (
            f"Shard 0 indices wrong: {shard0_indices}"
        )
        assert shard1_indices == list(range(15)), (
            f"Shard 1 indices wrong: {shard1_indices}"
        )

    def test_unequal_row_group_sizes(self, tmp_path):
        """Last RG may be smaller than the rest. Offset must still be correct."""
        total_docs = 23
        rg_size = 10  # produces RGs of 10, 10, 3
        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} " + " ".join(f"w{w}" for w in range(4)) for i in range(total_docs)]
        table = pa.table({"text": texts})
        pq.write_table(table, str(pq_path), row_group_size=rg_size)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

        actual = [d for _, d in rec.calls]
        assert actual == list(range(total_docs)), (
            f"Unequal RG sizes: expected 0-{total_docs - 1}, got {actual}"
        )

    def test_shard_stem_correct_per_call(self, tmp_path):
        """Each lookup call must use the correct shard stem."""
        pq_path = tmp_path / "shard_00042.parquet"
        _make_multi_rg_parquet(pq_path, 4, 2)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

        stems = [stem for stem, _ in rec.calls]
        assert all(s == "shard_00042" for s in stems)
        assert len(stems) == 8  # 4 docs × 2 RGs


class TestMissingDocFailFast:
    """Issue #521: a parquet row with no feature doc is corruption — the
    generator must propagate lookup's KeyError, never yield (tokens, None)."""

    def test_packed_rows_missing_doc_raises(self, tmp_path):
        """One parquet row lacks a feature doc → lookup raises → generator raises."""
        n_docs = 5
        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} hello world foo" for i in range(n_docs)]
        table = pa.table({"text": texts})
        pq.write_table(table, str(pq_path), row_group_size=n_docs + 100)

        # Docs 0-3 present, doc 4 absent → corruption.
        rec = RecordingFeatureIndex(present_docs=set(range(n_docs - 1)))
        tok = SimpleTokenizer()

        with pytest.raises(KeyError, match=r"not found in shard"):
            list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

    def test_packed_rows_all_present_yields_lists_never_none(self, tmp_path):
        """Every yielded packed_seg_features is a non-None, non-empty list when all
        docs are present."""
        n_docs = 6
        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} hello world foo" for i in range(n_docs)]
        table = pa.table({"text": texts})
        pq.write_table(table, str(pq_path), row_group_size=n_docs + 100)

        rec = RecordingFeatureIndex(present_docs=set(range(n_docs)))
        tok = SimpleTokenizer()

        out = list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))
        assert out, "generator produced no rows"
        assert len(out) == n_docs
        for _tokens, packed_seg_features, _stem, _doc in out:
            assert packed_seg_features is not None
            assert isinstance(packed_seg_features, list) and len(packed_seg_features) >= 1
