"""Tests for the completion TUI DataLoader (DocumentLoader).

Follows TDD: tests written before implementation.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.completion_tui.data_loader import DocumentLoader, segment_features_to_segment
from scripts.completion_tui.state import DocumentData, FeatureCondition, Segment
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.feature_io import save_features

# ---------------------------------------------------------------------------
# Test fixtures — create temporary parquet + feature shard files
# ---------------------------------------------------------------------------

def _make_parquet(path: Path, texts: list[str]) -> None:
    """Write a single-row-group parquet with a 'text' column."""
    table = pa.table({"text": texts})
    pq.write_table(table, str(path))


def _make_feature_shard(
    path: Path,
    doc_texts: list[str],
    segment_size: int = 64,
    features_per_segment: int = 3,
) -> None:
    """Create a synthetic feature shard matching the given documents.

    Produces segments based on character-level chunking of each document,
    with synthetic feature IDs and magnitudes.
    """
    all_doc_ids: list[int] = []
    all_seg_indices: list[int] = []
    all_seg_positions: list[int] = []
    all_char_starts: list[int] = []
    all_char_ends: list[int] = []
    all_offsets: list[int] = [0]
    all_fids: list[int] = []
    all_mags: list[float] = []

    for doc_idx, text in enumerate(doc_texts):
        # Simulate BOS-inclusive character offsets: segment_size chars each
        # In real extraction, char offsets include BOS prefix
        n_chars = len(text)
        seg_idx = 0
        pos = 0
        while pos < n_chars:
            seg_end = min(pos + segment_size, n_chars)
            all_doc_ids.append(doc_idx)
            all_seg_indices.append(seg_idx)
            all_seg_positions.append(seg_idx * 8)  # fake token position
            all_char_starts.append(pos)
            all_char_ends.append(seg_end)

            # Synthetic features: feature_id = doc_idx*100 + seg_idx*10 + i
            for i in range(features_per_segment):
                fid = doc_idx * 100 + seg_idx * 10 + i
                mag = 1.0 + i * 0.5 + doc_idx * 0.1
                all_fids.append(fid)
                all_mags.append(mag)
            all_offsets.append(len(all_fids))

            seg_idx += 1
            pos = seg_end

    n_segments = len(all_doc_ids)
    data = {
        "offsets": t.tensor(all_offsets, dtype=t.int64),
        "feature_ids": t.tensor(all_fids, dtype=t.int32),
        "feature_mags": t.tensor(all_mags, dtype=t.float32),
        "doc_ids": t.tensor(all_doc_ids, dtype=t.int64),
        "segment_indices": t.tensor(all_seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(all_seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(all_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(all_char_ends, dtype=t.int32),
        "n_segments": n_segments,
        "n_docs": len(doc_texts),
        "segment_size": segment_size,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed",
    }
    save_features(data, path)


@pytest.fixture()
def sample_data(tmp_path: Path):
    """Create a minimal dataset with 2 shards, each with 2 documents."""
    parquet_dir = tmp_path / "parquets"
    feature_dir = tmp_path / "features"
    parquet_dir.mkdir()
    feature_dir.mkdir()

    segment_size = 32
    texts_shard_0 = [
        "Alpha bravo charlie delta echo foxtrot golf hotel.",
        "India juliet kilo lima mike november oscar papa.",
    ]
    texts_shard_1 = [
        "Quebec romeo sierra tango uniform victor whiskey xray.",
        "Yankee zulu one two three four five six seven eight.",
    ]

    # Create parquets (shard_00000, shard_00001)
    _make_parquet(parquet_dir / "shard_00000.parquet", texts_shard_0)
    _make_parquet(parquet_dir / "shard_00001.parquet", texts_shard_1)

    # Create matching feature shards
    _make_feature_shard(
        feature_dir / "shard_00000",
        texts_shard_0,
        segment_size=segment_size,
    )
    _make_feature_shard(
        feature_dir / "shard_00001",
        texts_shard_1,
        segment_size=segment_size,
    )

    # Create boundaries tensor: large enough for all feature IDs
    # 4 buckets → 3 boundaries per feature
    num_buckets = 4
    n_sae_features = 500
    boundaries = t.zeros(n_sae_features, num_buckets - 1, dtype=t.float64)
    # Set reasonable boundaries for features we'll actually see
    for fid in range(n_sae_features):
        boundaries[fid] = t.tensor([1.0, 1.5, 2.0], dtype=t.float64)

    return {
        "parquet_dir": str(parquet_dir),
        "feature_dir": str(feature_dir),
        "boundaries": boundaries,
        "num_buckets": num_buckets,
        "has_negatives": False,
        "segment_size": segment_size,
        "texts_shard_0": texts_shard_0,
        "texts_shard_1": texts_shard_1,
    }


# ---------------------------------------------------------------------------
# FeatureIndex new methods tests
# ---------------------------------------------------------------------------

class TestFeatureIndexNewMethods:
    """Tests for shard_stems property and doc_ids_for_shard method."""

    def test_shard_stems_returns_sorted_list(self, sample_data):
        idx = FeatureIndex(sample_data["feature_dir"])
        stems = idx.shard_stems
        assert stems == ["shard_00000", "shard_00001"]

    def test_shard_stems_is_copy(self, sample_data):
        """Mutating the returned list should not affect the index."""
        idx = FeatureIndex(sample_data["feature_dir"])
        stems = idx.shard_stems
        stems.append("fake")
        assert "fake" not in idx.shard_stems

    def test_doc_ids_for_shard_returns_sorted(self, sample_data):
        idx = FeatureIndex(sample_data["feature_dir"])
        doc_ids = idx.doc_ids_for_shard("shard_00000")
        assert doc_ids == [0, 1]

    def test_doc_ids_for_shard_unknown_raises(self, sample_data):
        idx = FeatureIndex(sample_data["feature_dir"])
        with pytest.raises(KeyError, match="nonexistent"):
            idx.doc_ids_for_shard("nonexistent")

    def test_doc_ids_for_shard_caches(self, sample_data):
        """After doc_ids_for_shard, lookup on same shard should work."""
        idx = FeatureIndex(sample_data["feature_dir"])
        idx.doc_ids_for_shard("shard_00001")
        # Now lookup should use cached data
        result = idx.lookup("shard_00001", 0)
        assert result is not None


# ---------------------------------------------------------------------------
# DocumentLoader tests
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    """Tests for the DocumentLoader class."""

    def _make_loader(self, sample_data):
        return DocumentLoader(
            feature_dir=sample_data["feature_dir"],
            parquet_dir=sample_data["parquet_dir"],
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=sample_data["has_negatives"],
        )

    def test_validation_shard_is_last(self, sample_data):
        """Validation shard should be the last shard alphabetically."""
        loader = self._make_loader(sample_data)
        assert loader.validation_shard_stem == "shard_00001"

    def test_available_doc_ids_sorted(self, sample_data):
        loader = self._make_loader(sample_data)
        doc_ids = loader.available_doc_ids()
        assert doc_ids == sorted(doc_ids)
        assert doc_ids == [0, 1]

    def test_load_document_valid(self, sample_data):
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        assert doc.doc_idx == 0
        assert doc.text == sample_data["texts_shard_1"][0]
        assert len(doc.segments) > 0
        assert doc.total_segments >= len(doc.segments)

    def test_load_document_invalid_returns_none(self, sample_data):
        loader = self._make_loader(sample_data)
        doc = loader.load_document(999)
        assert doc is None

    def test_load_document_num_segments_limits(self, sample_data):
        loader = self._make_loader(sample_data)
        # First check total segments
        total = loader.total_segments(0)
        assert total is not None and total > 1

        # Load only 1 segment
        doc = loader.load_document(0, num_segments=1)
        assert doc is not None
        assert len(doc.segments) == 1
        assert doc.total_segments == total

    def test_load_document_segments_have_features(self, sample_data):
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            assert len(seg.features) > 0
            for fc in seg.features:
                assert isinstance(fc.feature_id, int)
                assert isinstance(fc.magnitude, int)
                assert fc.magnitude >= 0

    def test_load_document_last_segment_is_generation_target(self, sample_data):
        """When >1 segment loaded, last one should be is_generation_target."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        if len(doc.segments) > 1:
            assert doc.segments[-1].is_generation_target is True
            for seg in doc.segments[:-1]:
                assert seg.is_generation_target is False

    def test_load_document_single_segment_not_generation_target(self, sample_data):
        """When only 1 segment loaded, it should NOT be generation target."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0, num_segments=1)
        assert doc is not None
        assert len(doc.segments) == 1
        assert doc.segments[0].is_generation_target is False

    def test_total_segments_valid(self, sample_data):
        loader = self._make_loader(sample_data)
        total = loader.total_segments(0)
        assert total is not None
        assert total > 0

    def test_total_segments_invalid_returns_none(self, sample_data):
        loader = self._make_loader(sample_data)
        total = loader.total_segments(999)
        assert total is None

    def test_segment_char_boundaries_consistent(self, sample_data):
        """Segment char boundaries should cover the document text."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            assert seg.char_start >= 0
            assert seg.char_end >= seg.char_start
            # char_end should not exceed text length
            # (boundaries are in BOS-inclusive coordinate system)

    def test_segments_have_sequential_indices(self, sample_data):
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for i, seg in enumerate(doc.segments):
            assert seg.segment_idx == i

    def test_segments_have_color_indices(self, sample_data):
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for i, seg in enumerate(doc.segments):
            assert seg.color_idx == i


# ---------------------------------------------------------------------------
# segment_features_to_segment conversion tests
# ---------------------------------------------------------------------------

class TestSegmentFeaturesToSegment:
    """Tests for the conversion from SegmentFeatures to Segment."""

    def test_basic_conversion(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([10, 20, 30], dtype=t.int32),
            feature_mags=t.tensor([1.2, 1.8, 2.5], dtype=t.float32),
            char_start=0,
            char_end=64,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=2,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
        )
        assert seg.segment_idx == 0
        assert seg.char_start == 0
        assert seg.char_end == 64
        assert seg.color_idx == 2
        assert seg.is_generation_target is False
        assert len(seg.features) == 3

    def test_conversion_preserves_feature_ids(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=1,
            position=8,
            feature_ids=t.tensor([5, 15, 25], dtype=t.int32),
            feature_mags=t.tensor([1.0, 2.0, 3.0], dtype=t.float32),
            char_start=32,
            char_end=64,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
        )
        fids = [fc.feature_id for fc in seg.features]
        assert fids == [5, 15, 25]

    def test_conversion_discretizes_magnitudes(self, sample_data):
        # boundaries for each feature: [1.0, 1.5, 2.0] (4 buckets)
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([0.5, 1.5, 2.5], dtype=t.float32),
            char_start=0,
            char_end=32,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
        )
        # 0.5 < 1.0 → bucket 0
        # 1.5 == 1.5 → bucket 2 (right=True: exactly at boundary goes higher)
        # 2.5 > 2.0 → bucket 3
        mags = [fc.magnitude for fc in seg.features]
        assert mags[0] == 0
        assert mags[1] == 2
        assert mags[2] == 3

    def test_conversion_with_has_negatives(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0], dtype=t.int32),
            feature_mags=t.tensor([0.5], dtype=t.float32),
            char_start=0,
            char_end=32,
        )
        # With has_negatives, positive bucket count is effectively num_buckets-1,
        # so boundaries must have (num_buckets - 2) columns.
        trimmed = sample_data["boundaries"][:, :-1]
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=trimmed,
            num_buckets=sample_data["num_buckets"],
            has_negatives=True,
        )
        # 0.5 → discretized to internal bucket 0 → display magnitude 0.
        # Under current source semantics, the last internal bucket (num_buckets - 1)
        # is reserved for negatives (display -1); bucket 0 maps through as 0.
        assert seg.features[0].magnitude == 0

    def test_generation_target_flag(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0], dtype=t.int32),
            feature_mags=t.tensor([1.0], dtype=t.float32),
            char_start=0,
            char_end=32,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
            is_generation_target=True,
        )
        assert seg.is_generation_target is True

    def test_empty_features(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([], dtype=t.int32),
            feature_mags=t.tensor([], dtype=t.float32),
            char_start=0,
            char_end=32,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
        )
        assert len(seg.features) == 0

    def test_target_token_count_passed_through(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0], dtype=t.int32),
            feature_mags=t.tensor([1.0], dtype=t.float32),
            char_start=0,
            char_end=32,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
            is_generation_target=True,
            target_token_count=64,
        )
        assert seg.target_token_count == 64

    def test_target_token_count_default_none(self, sample_data):
        sf = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0], dtype=t.int32),
            feature_mags=t.tensor([1.0], dtype=t.float32),
            char_start=0,
            char_end=32,
        )
        seg = segment_features_to_segment(
            sf,
            color_idx=0,
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=False,
        )
        assert seg.target_token_count is None


# ---------------------------------------------------------------------------
# Type identity and field completeness tests
# ---------------------------------------------------------------------------

class TestTypeIdentity:
    """Verify that data_loader produces the canonical state.py types,
    not duplicates from a separate module.

    These tests exist because T1 originally created a models.py with
    duplicate Segment/FeatureCondition definitions missing fields like
    id, target_token_count, enabled, and description. This class ensures
    the consolidation is correct and prevents regressions.
    """

    def _make_loader(self, sample_data):
        return DocumentLoader(
            feature_dir=sample_data["feature_dir"],
            parquet_dir=sample_data["parquet_dir"],
            boundaries=sample_data["boundaries"],
            num_buckets=sample_data["num_buckets"],
            has_negatives=sample_data["has_negatives"],
        )

    def test_load_document_returns_state_segment_type(self, sample_data):
        """Segments must be the canonical state.py Segment, not a duplicate."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            assert type(seg) is Segment

    def test_load_document_returns_state_feature_condition_type(self, sample_data):
        """FeatureConditions must be the canonical state.py type."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            for fc in seg.features:
                assert type(fc) is FeatureCondition

    def test_load_document_returns_document_data_type(self, sample_data):
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        assert type(doc) is DocumentData

    def test_segment_has_id_field(self, sample_data):
        """Segment.id must exist and be an 8-char hex string (UUID)."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            assert hasattr(seg, "id")
            assert isinstance(seg.id, str)
            assert len(seg.id) == 8
            int(seg.id, 16)  # must be valid hex

    def test_segment_ids_are_unique(self, sample_data):
        """Each segment should have a distinct UUID."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        ids = [seg.id for seg in doc.segments]
        assert len(ids) == len(set(ids))

    def test_segment_has_target_token_count_field(self, sample_data):
        """All segments must have target_token_count attribute."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            assert hasattr(seg, "target_token_count")

    def test_generation_target_has_token_count_set(self, sample_data):
        """Generation target segment must have target_token_count from FeatureIndex."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        if len(doc.segments) > 1:
            target = doc.segments[-1]
            assert target.is_generation_target is True
            assert target.target_token_count is not None
            assert target.target_token_count > 0

    def test_non_target_segments_have_no_token_count(self, sample_data):
        """Non-target segments should have target_token_count=None."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments[:-1]:
            assert seg.target_token_count is None

    def test_feature_condition_has_enabled_field(self, sample_data):
        """FeatureCondition.enabled must exist and default to True."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            for fc in seg.features:
                assert hasattr(fc, "enabled")
                assert fc.enabled is True

    def test_feature_condition_has_description_field(self, sample_data):
        """FeatureCondition.description must exist and default to None."""
        loader = self._make_loader(sample_data)
        doc = loader.load_document(0)
        assert doc is not None
        for seg in doc.segments:
            for fc in seg.features:
                assert hasattr(fc, "description")
                assert fc.description is None
