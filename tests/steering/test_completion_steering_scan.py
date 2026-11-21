"""Tests for completion steering eval data structures and stages 1-2."""

import pytest
import torch as t

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.completion_steering_eval import (
    FeatureSelection,
    ScanResultTensors,
    _activation_to_percentile,
    _build_percentile_lookup,
    load_scan_results,
    save_scan_results,
)
from steering.stats_types import PerFeatureStat
from tests.steering._completion_steering_reference import (
    scan_results_to_dict,
    scan_validation_features,
)


def _make_per_feature_stats(feature_ids: list[int]) -> dict[str, PerFeatureStat]:
    """Create synthetic per-feature stats with p1..p99 percentiles."""
    stats = {}
    for fid in feature_ids:
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=5.0,
            mean_mag_incl_zeros=2.5,
            std_mag=2.0,
            min_mag=0.1,
            max_mag=10.0,
            median_mag=5.0,
            percentiles=pcts,
            histogram=None,
        )
    return stats


class TestFeatureSelectionDataclass:
    def test_feature_selection_fields(self):
        sel = FeatureSelection(
            feature_id=42,
            neuronpedia_description="test",
            n_docs_activated=10,
            selected_docs=[0, 1, 2],
        )
        assert sel.feature_id == 42
        assert len(sel.selected_docs) == 3


class TestActivationToPercentile:
    def test_exact_percentile_match(self):
        """Activation exactly at p50 value should return ~0.50."""
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        lookup = _build_percentile_lookup(pcts)
        result = _activation_to_percentile(5.0, lookup)  # p50 = 5.0
        assert 0.49 <= result <= 0.51

    def test_below_minimum(self):
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        lookup = _build_percentile_lookup(pcts)
        result = _activation_to_percentile(0.0, lookup)
        assert result <= 0.02

    def test_above_maximum(self):
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        lookup = _build_percentile_lookup(pcts)
        result = _activation_to_percentile(100.0, lookup)
        assert result >= 0.98

    def test_interpolation(self):
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        lookup = _build_percentile_lookup(pcts)
        # p25 = 2.5, p26 = 2.6. Activation 2.55 should be ~0.255
        result = _activation_to_percentile(2.55, lookup)
        assert 0.25 <= result <= 0.26

    def test_empty_percentiles_raise(self):
        with pytest.raises(ValueError, match="No valid percentile keys"):
            _build_percentile_lookup({})

    def test_non_percentile_keys_raise(self):
        with pytest.raises(ValueError, match="No valid percentile keys"):
            _build_percentile_lookup({"mean": 5.0, "std": 1.0})


class TestScanValidationFeatures:
    def test_max_activation_across_segments(self):
        doc_segments = {
            0: [
                SegmentFeatures(0, 0, t.tensor([10]), t.tensor([5.0]), 0, 128),
                SegmentFeatures(1, 128, t.tensor([10]), t.tensor([3.0]), 128, 256),
                SegmentFeatures(2, 256, t.tensor([10]), t.tensor([7.0]), 256, 384),
            ]
        }
        stats = _make_per_feature_stats([10])
        results = scan_validation_features(doc_segments, stats)
        assert 10 in results
        assert results[10].n_docs_activated == 1
        act = results[10].doc_activations[0]
        assert act.max_activation == 7.0
        assert act.max_segment_idx == 2
        assert act.total_segments == 3

    def test_first_segment_excluded(self):
        doc_segments = {
            0: [
                SegmentFeatures(0, 0, t.tensor([10]), t.tensor([5.0]), 0, 128),
            ]
        }
        stats = _make_per_feature_stats([10])
        results = scan_validation_features(doc_segments, stats)
        assert 10 not in results

    def test_single_segment_doc_excluded(self):
        doc_segments = {
            0: [
                SegmentFeatures(0, 0, t.tensor([10, 20]), t.tensor([5.0, 3.0]), 0, 128),
            ]
        }
        stats = _make_per_feature_stats([10, 20])
        results = scan_validation_features(doc_segments, stats)
        assert len(results) == 0

    def test_multiple_docs_multiple_features(self):
        doc_segments = {
            0: [
                SegmentFeatures(0, 0, t.tensor([10]), t.tensor([1.0]), 0, 128),
                SegmentFeatures(1, 128, t.tensor([10]), t.tensor([3.0]), 128, 256),
            ],
            1: [
                SegmentFeatures(0, 0, t.tensor([10]), t.tensor([2.0]), 0, 128),
                SegmentFeatures(1, 128, t.tensor([10]), t.tensor([4.0]), 128, 256),
            ],
        }
        stats = _make_per_feature_stats([10])
        results = scan_validation_features(doc_segments, stats)
        assert results[10].n_docs_activated == 2

    def test_percentile_conversion(self):
        doc_segments = {
            0: [
                SegmentFeatures(0, 0, t.tensor([10]), t.tensor([1.0]), 0, 128),
                SegmentFeatures(1, 128, t.tensor([10]), t.tensor([5.0]), 128, 256),
            ]
        }
        stats = _make_per_feature_stats([10])
        results = scan_validation_features(doc_segments, stats)
        act = results[10].doc_activations[0]
        assert 0.45 <= act.percentile <= 0.55

    def test_feature_not_in_stats_raises(self):
        doc_segments = {
            0: [
                SegmentFeatures(0, 0, t.tensor([999]), t.tensor([1.0]), 0, 128),
                SegmentFeatures(1, 128, t.tensor([999]), t.tensor([3.0]), 128, 256),
            ]
        }
        stats = _make_per_feature_stats([10])  # no stats for 999
        with pytest.raises(KeyError, match="Feature 999"):
            scan_validation_features(doc_segments, stats)


class TestScanResultsSafetensors:
    """Tests for save/load roundtrip of scan results via safetensors."""

    def _make_scan_result_tensors(self) -> ScanResultTensors:
        """Two features: 10 (3 docs) and 42 (1 doc), sorted by feature_id."""
        return ScanResultTensors(
            feature_ids=t.tensor([10, 42], dtype=t.int64),
            n_docs_activated=t.tensor([3, 1], dtype=t.int64),
            offsets=t.tensor([0, 3, 4], dtype=t.int64),
            doc_idx=t.tensor([0, 3, 7, 1], dtype=t.int64),
            max_activation=t.tensor([5.0, 3.2, 8.1, 2.0], dtype=t.float32),
            max_segment_idx=t.tensor([2, 1, 3, 1], dtype=t.int64),
            percentile=t.tensor([0.85, 0.60, 0.95, 0.40], dtype=t.float32),
            total_segments=t.tensor([4, 3, 5, 2], dtype=t.int64),
        )

    def _roundtrip(self, srt: ScanResultTensors, tmp_path, filename="scan.safetensors"):
        path = tmp_path / filename
        save_scan_results(srt, path)
        return load_scan_results(path)

    def test_roundtrip_preserves_data(self, tmp_path):
        """All fields must survive a save→load cycle with correct values."""
        original = self._make_scan_result_tensors()
        loaded = self._roundtrip(original, tmp_path)

        assert t.equal(loaded.feature_ids, original.feature_ids)
        assert t.equal(loaded.n_docs_activated, original.n_docs_activated)
        assert t.equal(loaded.offsets, original.offsets)
        assert t.equal(loaded.doc_idx, original.doc_idx)
        assert t.allclose(loaded.max_activation, original.max_activation)
        assert t.equal(loaded.max_segment_idx, original.max_segment_idx)
        assert t.allclose(loaded.percentile, original.percentile, atol=1e-6)
        assert t.equal(loaded.total_segments, original.total_segments)

    def test_roundtrip_to_dict_preserves_data(self, tmp_path):
        """ScanResultTensors → save → load → dict matches expected dataclass values."""
        original = self._make_scan_result_tensors()
        loaded = self._roundtrip(original, tmp_path, "dict.safetensors")
        d = scan_results_to_dict(loaded)

        assert set(d.keys()) == {10, 42}
        assert d[10].n_docs_activated == 3
        assert len(d[10].doc_activations) == 3
        assert d[10].doc_activations[0].doc_idx == 0
        assert d[10].doc_activations[0].max_activation == pytest.approx(5.0)
        assert d[42].doc_activations[0].percentile == pytest.approx(0.40, abs=1e-6)

    def test_empty_scan_results(self, tmp_path):
        """Empty ScanResultTensors roundtrips cleanly."""
        empty = ScanResultTensors(
            feature_ids=t.empty(0, dtype=t.int64),
            n_docs_activated=t.empty(0, dtype=t.int64),
            offsets=t.zeros(1, dtype=t.int64),
            doc_idx=t.empty(0, dtype=t.int64),
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=t.empty(0, dtype=t.int64),
            percentile=t.empty(0, dtype=t.float32),
            total_segments=t.empty(0, dtype=t.int64),
        )
        loaded = self._roundtrip(empty, tmp_path, "empty.safetensors")
        assert len(loaded.feature_ids) == 0
        d = scan_results_to_dict(loaded)
        assert d == {}

    def test_feature_with_no_activations(self, tmp_path):
        """A feature with n_docs_activated=0 and no doc_activations survives."""
        srt = ScanResultTensors(
            feature_ids=t.tensor([99], dtype=t.int64),
            n_docs_activated=t.tensor([0], dtype=t.int64),
            offsets=t.tensor([0, 0], dtype=t.int64),
            doc_idx=t.empty(0, dtype=t.int64),
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=t.empty(0, dtype=t.int64),
            percentile=t.empty(0, dtype=t.float32),
            total_segments=t.empty(0, dtype=t.int64),
        )
        loaded = self._roundtrip(srt, tmp_path, "no_acts.safetensors")
        d = scan_results_to_dict(loaded)
        assert d[99].n_docs_activated == 0
        assert d[99].doc_activations == []

    def test_large_feature_count(self, tmp_path):
        """1000 features × 10 docs each round-trips correctly."""
        n_feats = 1000
        n_docs_per = 10
        total = n_feats * n_docs_per

        fids = t.arange(n_feats, dtype=t.int64)
        n_docs = t.full((n_feats,), n_docs_per, dtype=t.int64)
        offsets = t.arange(0, total + 1, n_docs_per, dtype=t.int64)

        # doc_idx: 0..9 repeated for each feature
        doc_idx = t.arange(n_docs_per, dtype=t.int64).repeat(n_feats)
        max_act = doc_idx.float() * 0.1
        max_seg = (doc_idx % 5).long()
        pct = t.full((total,), 0.5, dtype=t.float32)
        tot_seg = t.full((total,), 5, dtype=t.int64)

        srt = ScanResultTensors(
            feature_ids=fids,
            n_docs_activated=n_docs,
            offsets=offsets,
            doc_idx=doc_idx,
            max_activation=max_act,
            max_segment_idx=max_seg,
            percentile=pct,
            total_segments=tot_seg,
        )
        loaded = self._roundtrip(srt, tmp_path, "large.safetensors")
        d = scan_results_to_dict(loaded)
        assert len(d) == 1000
        assert d[500].n_docs_activated == 10
        assert len(d[500].doc_activations) == 10
        act = d[500].doc_activations[7]
        assert act.doc_idx == 7
        assert act.max_activation == pytest.approx(0.7)
        assert act.max_segment_idx == 2

    def test_feature_ids_sorted_in_output(self, tmp_path):
        """Feature IDs should be stored sorted regardless of construction order."""
        # Build already-sorted (the tensor API enforces this)
        srt = ScanResultTensors(
            feature_ids=t.tensor([5, 50, 100], dtype=t.int64),
            n_docs_activated=t.tensor([1, 1, 1], dtype=t.int64),
            offsets=t.tensor([0, 1, 2, 3], dtype=t.int64),
            doc_idx=t.tensor([0, 0, 0], dtype=t.int64),
            max_activation=t.tensor([2.0, 3.0, 1.0], dtype=t.float32),
            max_segment_idx=t.tensor([0, 0, 0], dtype=t.int64),
            percentile=t.tensor([0.6, 0.7, 0.5], dtype=t.float32),
            total_segments=t.tensor([2, 2, 2], dtype=t.int64),
        )
        loaded = self._roundtrip(srt, tmp_path, "sorted.safetensors")
        d = scan_results_to_dict(loaded)
        assert set(d.keys()) == {5, 50, 100}
        assert d[5].doc_activations[0].max_activation == pytest.approx(2.0)
        assert d[100].doc_activations[0].max_activation == pytest.approx(1.0)

    def test_float32_precision(self, tmp_path):
        """Percentile and max_activation stored as float32 should survive roundtrip
        within float32 precision (not float64)."""
        import struct

        val = 1.0 / 3.0
        val_f32 = struct.unpack("f", struct.pack("f", val))[0]

        srt = ScanResultTensors(
            feature_ids=t.tensor([1], dtype=t.int64),
            n_docs_activated=t.tensor([1], dtype=t.int64),
            offsets=t.tensor([0, 1], dtype=t.int64),
            doc_idx=t.tensor([0], dtype=t.int64),
            max_activation=t.tensor([val], dtype=t.float32),
            max_segment_idx=t.tensor([0], dtype=t.int64),
            percentile=t.tensor([val], dtype=t.float32),
            total_segments=t.tensor([2], dtype=t.int64),
        )
        loaded = self._roundtrip(srt, tmp_path, "precision.safetensors")
        d = scan_results_to_dict(loaded)
        act = d[1].doc_activations[0]
        assert act.max_activation == pytest.approx(val_f32, rel=1e-7)
        assert act.percentile == pytest.approx(val_f32, rel=1e-7)
