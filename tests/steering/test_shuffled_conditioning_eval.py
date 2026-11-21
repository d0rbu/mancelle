"""Tests for the shuffled conditioning eval."""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch as t

from errors import RemovedFeatureError
from steering.conditional.feature_index import SegmentFeatures
from steering.eval.multiseg_generator import (
    _segment_bounds_for_assignment,
    prepare_segments,
)
from steering.eval.shuffled_conditioning_eval import (
    FeatureSelectionStrategy,
    ShuffledConditioningResult,
    _compute_percentiles_for_features,
    aggregate_results,
    assert_no_feature_clustering,
    compute_topk_percentile_activation,
    deserialize_results,
    generate_for_shuffled_document,
    replace_target_segment,
    sample_shuffled_segment,
    select_eval_documents,
    select_random_k_features,
    select_top_k_features,
    serialize_results,
)
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets
from steering.stats_types import PerFeatureStat
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

# ---------------------------------------------------------------------------
# Removed feature-clustering fail-fast
# ---------------------------------------------------------------------------
#
# shuffled_conditioning_eval rejects legacy clustered checkpoint metadata before
# it can affect runtime behavior. Unclustered checkpoints are byte-identical.


class TestAssertNoFeatureClustering:
    def test_clustered_meta_raises(self):
        with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
            assert_no_feature_clustering(
                {"feature_clustering_canonical": {"clusters": [[2, 6, 8]]}}
            )

    def test_unclustered_meta_key_absent_is_noop(self):
        # No exception ⇒ byte-identical for unclustered checkpoints.
        assert_no_feature_clustering({"step": 100})

    def test_unclustered_meta_key_none_is_noop(self):
        assert_no_feature_clustering({"feature_clustering_canonical": None})

    def test_none_meta_is_noop(self):
        assert_no_feature_clustering(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    feature_ids: list[int],
    feature_mags: list[int],
    segment_idx: int = 1,
) -> SegmentFeatures:
    """Build a SegmentFeatures with int64 bucket magnitudes (post-threshold)."""
    return SegmentFeatures(
        segment_idx=segment_idx,
        position=segment_idx * 128,
        feature_ids=t.tensor(feature_ids, dtype=t.int32),
        feature_mags=t.tensor(feature_mags, dtype=t.int64),
        char_start=segment_idx * 512,
        char_end=(segment_idx + 1) * 512,
    )


def _make_raw_segment(
    feature_ids: list[int],
    feature_mags: list[float],
    segment_idx: int = 1,
) -> SegmentFeatures:
    """Build a SegmentFeatures with float32 raw magnitudes (pre-discretization)."""
    return SegmentFeatures(
        segment_idx=segment_idx,
        position=segment_idx * 128,
        feature_ids=t.tensor(feature_ids, dtype=t.int32),
        feature_mags=t.tensor(feature_mags, dtype=t.float32),
        char_start=segment_idx * 512,
        char_end=(segment_idx + 1) * 512,
    )


def _make_mock_feature_index(doc_segments: dict) -> MagicMock:
    """Build a mock FeatureIndex that returns pre-built doc segments."""
    index = MagicMock()
    index.lookup = MagicMock(
        side_effect=lambda stem, doc_idx: doc_segments.get(doc_idx)
    )
    return index


def _make_per_feature_stats(feature_ids: list[int]) -> dict[str, PerFeatureStat]:
    """Build synthetic per-feature stats with known percentile distributions.

    Each feature has p1=1.0, p10=10.0, ..., p99=99.0.
    """
    stats: dict[str, PerFeatureStat] = {}
    for fid in feature_ids:
        percentiles = {f"p{p}": float(p) for p in range(1, 100)}
        stats[str(fid)] = PerFeatureStat(
            count=1000,
            pct_segments_active=0.5,
            mean_mag=50.0,
            mean_mag_incl_zeros=25.0,
            std_mag=25.0,
            min_mag=0.0,
            max_mag=100.0,
            median_mag=50.0,
            percentiles=percentiles,
        )
    return stats


def _make_custom_per_feature_stat(
    percentile_values: dict[str, float],
) -> PerFeatureStat:
    """Build a PerFeatureStat with a custom percentile distribution."""
    return PerFeatureStat(
        count=1000,
        pct_segments_active=0.5,
        mean_mag=50.0,
        mean_mag_incl_zeros=25.0,
        std_mag=25.0,
        min_mag=0.0,
        max_mag=100.0,
        median_mag=50.0,
        percentiles=percentile_values,
    )


# ===========================================================================
# TestFeatureSelectionStrategy — enum round-tripping
# ===========================================================================


class TestFeatureSelectionStrategy:
    """Tests for the FeatureSelectionStrategy enum."""

    def test_from_string_top_k(self) -> None:
        assert FeatureSelectionStrategy("top_k") is FeatureSelectionStrategy.top_k

    def test_from_string_random_k(self) -> None:
        assert FeatureSelectionStrategy("random_k") is FeatureSelectionStrategy.random_k

    def test_value_round_trip(self) -> None:
        for member in FeatureSelectionStrategy:
            assert FeatureSelectionStrategy(member.value) is member

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            FeatureSelectionStrategy("bogus")


# ===========================================================================
# TestSelectTopKFeatures — percentile-based ranking
# ===========================================================================


class TestSelectTopKFeatures:
    """Tests for select_top_k_features with percentile ranking."""

    def test_basic_percentile_ranking(self) -> None:
        """Features with different raw mags map to different percentiles;
        highest percentile should come first."""
        # All features share the same scale (p1=1..p99=99)
        stats = _make_per_feature_stats([100, 101, 102])
        # raw mags: 100→90.0 (p~90), 101→50.0 (p~50), 102→10.0 (p~10)
        seg = _make_raw_segment(
            feature_ids=[100, 101, 102],
            feature_mags=[90.0, 50.0, 10.0],
        )
        result = select_top_k_features(seg, stats, k=3)
        assert result == [100, 101, 102]

    def test_percentile_reverses_raw_order(self) -> None:
        """Feature A has raw mag 100 but p99=200 (~50th pct);
        Feature B has raw mag 5 but p99=5 (99th pct).
        B should rank higher even though A has higher raw mag."""
        # Feature A: p1=1, ..., p50=100, ..., p99=200
        # So activation 100 maps to ~50th percentile
        stats_a = _make_custom_per_feature_stat(
            {f"p{p}": float(p) * 2.0 for p in range(1, 100)}
        )
        # Feature B: p1=0.05, ..., p99=5.0
        # So activation 5 maps to ~99th percentile
        stats_b = _make_custom_per_feature_stat(
            {f"p{p}": float(p) * 5.0 / 99.0 for p in range(1, 100)}
        )
        per_feature_stats = {"10": stats_a, "20": stats_b}

        seg = _make_raw_segment(
            feature_ids=[10, 20],
            feature_mags=[100.0, 5.0],
        )
        result = select_top_k_features(seg, per_feature_stats, k=2)
        # Feature 20 (B) should rank first because its percentile is higher
        assert result[0] == 20
        assert result[1] == 10

    def test_tie_breaking_same_percentile(self) -> None:
        """Features with identical percentiles should be sorted by feature_id ascending."""
        stats = _make_per_feature_stats([500, 200, 300, 100, 400])
        # All at same raw mag → same percentile on same scale
        seg = _make_raw_segment(
            feature_ids=[500, 200, 300, 100, 400],
            feature_mags=[50.0, 50.0, 50.0, 50.0, 50.0],
        )
        result = select_top_k_features(seg, stats, k=3)
        assert result == [100, 200, 300]

    def test_k_larger_than_active(self) -> None:
        """k=16 but only 3 features; returns all 3."""
        stats = _make_per_feature_stats([1, 2, 3])
        seg = _make_raw_segment(
            feature_ids=[1, 2, 3],
            feature_mags=[10.0, 20.0, 30.0],
        )
        result = select_top_k_features(seg, stats, k=16)
        assert set(result) == {1, 2, 3}
        assert len(result) == 3

    def test_k_equals_one(self) -> None:
        """Returns single highest percentile feature."""
        stats = _make_per_feature_stats([10, 20, 30])
        seg = _make_raw_segment(
            feature_ids=[10, 20, 30],
            feature_mags=[1.0, 90.0, 50.0],
        )
        result = select_top_k_features(seg, stats, k=1)
        assert result == [20]

    def test_no_active_features_raises(self) -> None:
        """Empty segment raises ValueError."""
        stats = _make_per_feature_stats([])
        seg = _make_raw_segment(feature_ids=[], feature_mags=[])
        with pytest.raises(ValueError, match="no active features"):
            select_top_k_features(seg, stats)

    def test_missing_stats_for_feature(self) -> None:
        """Feature not in per_feature_stats is skipped with warning (not crash)."""
        # Only have stats for feature 10, not 20
        stats = _make_per_feature_stats([10])
        seg = _make_raw_segment(
            feature_ids=[10, 20],
            feature_mags=[50.0, 90.0],
        )
        result = select_top_k_features(seg, stats, k=2)
        # Feature 20 is skipped, only feature 10 returned
        assert result == [10]

    def test_all_features_missing_stats_raises(self) -> None:
        """All features have missing stats — should raise ValueError."""
        stats = _make_per_feature_stats([999])  # no stats for features 10, 20
        seg = _make_raw_segment(
            feature_ids=[10, 20],
            feature_mags=[50.0, 90.0],
        )
        with pytest.raises(ValueError, match="none have entries in per_feature_stats"):
            select_top_k_features(seg, stats, k=2)

    def test_all_features_at_floor_percentile(self) -> None:
        """All features have activation 0 (below p1); should still return
        them sorted by feature_id."""
        stats = _make_per_feature_stats([5, 10, 15])
        seg = _make_raw_segment(
            feature_ids=[15, 5, 10],
            feature_mags=[0.0, 0.0, 0.0],
        )
        result = select_top_k_features(seg, stats, k=16)
        # All at floor percentile (p1 = 0.01), tie-break by ascending feature_id
        assert result == [5, 10, 15]


# ===========================================================================
# TestSelectRandomKFeatures — random sampling from active set
# ===========================================================================


class TestSelectRandomKFeatures:
    """Tests for select_random_k_features."""

    def test_returns_k_features(self) -> None:
        """Returns exactly k features when segment has >= k with stats."""
        stats = _make_per_feature_stats([1, 2, 3, 4, 5])
        seg = _make_raw_segment(
            feature_ids=[1, 2, 3, 4, 5],
            feature_mags=[10.0, 20.0, 30.0, 40.0, 50.0],
        )
        result = select_random_k_features(seg, stats, k=3, seed=42)
        assert len(result) == 3

    def test_returns_fewer_if_segment_has_fewer(self) -> None:
        """If segment has < k features with stats, returns all of them."""
        stats = _make_per_feature_stats([1, 2])
        seg = _make_raw_segment(
            feature_ids=[1, 2],
            feature_mags=[10.0, 20.0],
        )
        result = select_random_k_features(seg, stats, k=16, seed=42)
        assert len(result) == 2
        assert set(result) == {1, 2}

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces the same selection."""
        stats = _make_per_feature_stats(list(range(50)))
        seg = _make_raw_segment(
            feature_ids=list(range(50)),
            feature_mags=[float(i) for i in range(50)],
        )
        r1 = select_random_k_features(seg, stats, k=10, seed=123)
        r2 = select_random_k_features(seg, stats, k=10, seed=123)
        assert r1 == r2

    def test_different_seed_different_selection(self) -> None:
        """Different seeds produce different selections (given enough candidates)."""
        stats = _make_per_feature_stats(list(range(100)))
        seg = _make_raw_segment(
            feature_ids=list(range(100)),
            feature_mags=[float(i) for i in range(100)],
        )
        r1 = select_random_k_features(seg, stats, k=10, seed=1)
        r2 = select_random_k_features(seg, stats, k=10, seed=2)
        assert r1 != r2

    def test_only_features_with_stats(self) -> None:
        """Features missing from per_feature_stats are excluded."""
        # Only features 1 and 3 have stats
        stats = _make_per_feature_stats([1, 3])
        seg = _make_raw_segment(
            feature_ids=[1, 2, 3, 4],
            feature_mags=[10.0, 20.0, 30.0, 40.0],
        )
        result = select_random_k_features(seg, stats, k=16, seed=42)
        assert set(result) == {1, 3}

    def test_all_features_missing_stats_raises(self) -> None:
        """Raises ValueError when no features have stats."""
        stats = _make_per_feature_stats([999])
        seg = _make_raw_segment(
            feature_ids=[10, 20],
            feature_mags=[50.0, 90.0],
        )
        with pytest.raises(ValueError, match="none have entries in per_feature_stats"):
            select_random_k_features(seg, stats, k=2, seed=42)

    def test_no_active_features_raises(self) -> None:
        """Raises ValueError on empty segment."""
        stats = _make_per_feature_stats([])
        seg = _make_raw_segment(feature_ids=[], feature_mags=[])
        with pytest.raises(ValueError, match="no active features"):
            select_random_k_features(seg, stats, k=2, seed=42)

    def test_selected_features_are_subset_of_active(self) -> None:
        """Every returned feature ID exists in the segment."""
        stats = _make_per_feature_stats([10, 20, 30, 40, 50])
        seg = _make_raw_segment(
            feature_ids=[10, 20, 30, 40, 50],
            feature_mags=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        result = select_random_k_features(seg, stats, k=3, seed=7)
        active_set = set(seg.feature_ids.tolist())
        assert all(fid in active_set for fid in result)


# ===========================================================================
# TestReplaceTargetSegment
# ===========================================================================


class TestReplaceTargetSegment:
    """Tests for replace_target_segment."""

    def test_replaces_only_target(self) -> None:
        seg0 = _make_segment([1, 2], [10, 20], segment_idx=0)
        seg1 = _make_segment([3, 4], [30, 40], segment_idx=1)
        seg2 = _make_segment([5, 6], [50, 60], segment_idx=2)
        donor = _make_segment([99, 100], [90, 91], segment_idx=7)

        result = replace_target_segment(
            [seg0, seg1, seg2], target_seg_idx=1, donor=donor
        )

        # seg0 and seg2 should be untouched
        assert t.equal(result[0].feature_ids, seg0.feature_ids)
        assert t.equal(result[2].feature_ids, seg2.feature_ids)
        # target should have donor features
        assert t.equal(result[1].feature_ids, donor.feature_ids)
        assert t.equal(result[1].feature_mags, donor.feature_mags)

    def test_preserves_target_metadata(self) -> None:
        seg0 = _make_segment([1], [10], segment_idx=0)
        seg1 = _make_segment([3, 4], [30, 40], segment_idx=1)
        donor = _make_segment([99], [90], segment_idx=5)

        result = replace_target_segment([seg0, seg1], target_seg_idx=1, donor=donor)

        replaced = result[1]
        assert replaced.segment_idx == seg1.segment_idx
        assert replaced.position == seg1.position
        assert replaced.char_start == seg1.char_start
        assert replaced.char_end == seg1.char_end

    def test_does_not_mutate_original(self) -> None:
        seg0 = _make_segment([1, 2], [10, 20], segment_idx=0)
        seg1 = _make_segment([3, 4], [30, 40], segment_idx=1)
        donor = _make_segment([99, 100], [90, 91], segment_idx=7)
        original_list = [seg0, seg1]

        result = replace_target_segment(original_list, target_seg_idx=1, donor=donor)

        # Original list should be unchanged
        assert len(original_list) == 2
        assert t.equal(original_list[1].feature_ids, t.tensor([3, 4], dtype=t.int32))
        # Result should be a different list
        assert result is not original_list


# ===========================================================================
# TestSampleShuffledSegment
# ===========================================================================


class TestSampleShuffledSegment:
    """Tests for sample_shuffled_segment."""

    def _make_raw_segments(
        self, n_segments: int, n_features: int = 3
    ) -> list[SegmentFeatures]:
        """Build raw segments with float32 mags (pre-threshold)."""
        segs = []
        for i in range(n_segments):
            segs.append(
                SegmentFeatures(
                    segment_idx=i,
                    position=i * 128,
                    feature_ids=t.tensor(list(range(n_features)), dtype=t.int32),
                    feature_mags=t.tensor([5.0] * n_features, dtype=t.float32),
                    char_start=i * 512,
                    char_end=(i + 1) * 512,
                )
            )
        return segs

    def _make_boundaries(self, n_features: int = 3, n_boundaries: int = 4) -> t.Tensor:
        """Build simple boundaries tensor."""
        return (
            t.linspace(1.0, 10.0, n_boundaries)
            .unsqueeze(0)
            .expand(n_features, -1)
            .to(t.float64)
        )

    @patch("steering.eval.shuffled_conditioning_eval.prepare_segments")
    def test_excludes_current_doc(self, mock_prepare: MagicMock) -> None:
        """Sampled doc should never be the current doc."""
        # prepare_segments returns segments with int64 mags and at least 1 feature
        prepared_seg = _make_segment([1, 2], [3, 4], segment_idx=2)
        mock_prepare.return_value = [prepared_seg]

        raw_segs = self._make_raw_segments(5)
        doc_segments = {
            0: raw_segs,
            1: raw_segs,
            2: raw_segs,
            3: raw_segs,
        }
        index = _make_mock_feature_index(doc_segments)
        boundaries = self._make_boundaries()
        rng = random.Random(42)

        for _ in range(20):
            _, donor_doc_idx = sample_shuffled_segment(
                target_segment_idx=2,
                current_doc_idx=0,
                feature_index=index,
                val_shard_stem="shard_0",
                all_doc_indices=[0, 1, 2, 3],
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                rng=rng,
            )
            assert donor_doc_idx != 0

    @patch("steering.eval.shuffled_conditioning_eval.prepare_segments")
    def test_returns_correct_segment_idx(self, mock_prepare: MagicMock) -> None:
        """Returned segment should have the requested segment_idx."""
        target_idx = 2
        prepared_seg = _make_segment([1], [3], segment_idx=target_idx)
        mock_prepare.return_value = [prepared_seg]

        raw_segs = self._make_raw_segments(5)
        doc_segments = {0: raw_segs, 1: raw_segs}
        index = _make_mock_feature_index(doc_segments)
        boundaries = self._make_boundaries()
        rng = random.Random(42)

        seg, _ = sample_shuffled_segment(
            target_segment_idx=target_idx,
            current_doc_idx=0,
            feature_index=index,
            val_shard_stem="shard_0",
            all_doc_indices=[0, 1],
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            rng=rng,
        )
        assert seg.segment_idx == target_idx

    @patch("steering.eval.shuffled_conditioning_eval.prepare_segments")
    def test_skips_docs_without_enough_segments(self, mock_prepare: MagicMock) -> None:
        """Only docs with enough segments should be selected."""
        prepared_seg = _make_segment([1], [3], segment_idx=2)
        mock_prepare.return_value = [prepared_seg]

        short_segs = self._make_raw_segments(2)  # only 2 segments, need idx=2
        long_segs = self._make_raw_segments(5)  # 5 segments, idx=2 is valid
        doc_segments = {
            0: long_segs,  # current doc
            1: short_segs,  # too short
            2: long_segs,  # valid donor
        }
        index = _make_mock_feature_index(doc_segments)
        boundaries = self._make_boundaries()
        rng = random.Random(42)

        _, donor_doc_idx = sample_shuffled_segment(
            target_segment_idx=2,
            current_doc_idx=0,
            feature_index=index,
            val_shard_stem="shard_0",
            all_doc_indices=[0, 1, 2],
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            rng=rng,
        )
        assert donor_doc_idx == 2

    def test_no_valid_docs_raises(self) -> None:
        """Should raise ValueError when no valid donor doc found."""
        # All docs are current doc or have no segments
        doc_segments = {0: self._make_raw_segments(5)}
        index = _make_mock_feature_index(doc_segments)
        boundaries = self._make_boundaries()
        rng = random.Random(42)

        with pytest.raises(ValueError, match="Could not find"):
            sample_shuffled_segment(
                target_segment_idx=2,
                current_doc_idx=0,
                feature_index=index,
                val_shard_stem="shard_0",
                all_doc_indices=[0],
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                rng=rng,
                max_attempts=10,
            )

    @patch("steering.eval.shuffled_conditioning_eval.prepare_segments")
    def test_returned_segment_is_prepared(self, mock_prepare: MagicMock) -> None:
        """Returned segment should have int64 feature_mags (discretized)."""
        prepared_seg = _make_segment([1, 2], [3, 4], segment_idx=1)
        mock_prepare.return_value = [prepared_seg]

        raw_segs = self._make_raw_segments(5)
        doc_segments = {0: raw_segs, 1: raw_segs}
        index = _make_mock_feature_index(doc_segments)
        boundaries = self._make_boundaries()
        rng = random.Random(42)

        seg, _ = sample_shuffled_segment(
            target_segment_idx=1,
            current_doc_idx=0,
            feature_index=index,
            val_shard_stem="shard_0",
            all_doc_indices=[0, 1],
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            rng=rng,
        )
        assert seg.feature_mags.dtype == t.int64

    @patch("steering.eval.shuffled_conditioning_eval.prepare_segments")
    def test_logs_when_prepare_segments_returns_empty(
        self, mock_prepare: MagicMock,
    ) -> None:
        """When prepare_segments returns an empty list, sample_shuffled_segment
        must log a debug message — not silently skip."""
        import io

        from loguru import logger as _loguru_logger

        # First call returns empty (should log), subsequent calls return valid
        prepared_seg = _make_segment([1], [3], segment_idx=1)
        mock_prepare.side_effect = [[], [prepared_seg]]

        raw_segs = self._make_raw_segments(5)
        doc_segments = {0: raw_segs, 1: raw_segs, 2: raw_segs}
        index = _make_mock_feature_index(doc_segments)
        boundaries = self._make_boundaries()
        rng = random.Random(42)

        # Capture loguru output via a string sink
        log_output = io.StringIO()
        sink_id = _loguru_logger.add(log_output, level="DEBUG", format="{message}")
        try:
            sample_shuffled_segment(
                target_segment_idx=1,
                current_doc_idx=0,
                feature_index=index,
                val_shard_stem="shard_0",
                all_doc_indices=[0, 1, 2],
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                rng=rng,
            )
        finally:
            _loguru_logger.remove(sink_id)

        captured = log_output.getvalue()
        assert "prepare_segments" in captured.lower(), (
            f"Expected a log message about prepare_segments, got: {captured!r}"
        )


# ===========================================================================
# TestPercentileComputation
# ===========================================================================


class TestPercentileComputation:
    """Tests for _compute_percentiles_for_features."""

    def test_known_percentiles(self) -> None:
        """Activations at known values should map to expected percentiles."""
        stats = _make_per_feature_stats([10, 20])
        activations = {10: 50.0, 20: 90.0}
        result = _compute_percentiles_for_features(activations, stats)

        assert abs(result[10] - 0.50) < 0.02
        assert abs(result[20] - 0.90) < 0.02

    def test_zero_activations(self) -> None:
        """Activation of 0.0 should map to the lowest percentile (0.01)."""
        stats = _make_per_feature_stats([5])
        activations = {5: 0.0}
        result = _compute_percentiles_for_features(activations, stats)
        assert abs(result[5] - 0.01) < 0.005

    def test_above_max_activation(self) -> None:
        """Activation above max should map to highest percentile (0.99)."""
        stats = _make_per_feature_stats([5])
        activations = {5: 999.0}
        result = _compute_percentiles_for_features(activations, stats)
        assert abs(result[5] - 0.99) < 0.005

    def test_mean_percentile(self) -> None:
        """Average of two features' percentiles should be correct."""
        stats = _make_per_feature_stats([10, 20])
        # 10.0 -> ~0.10, 90.0 -> ~0.90; mean ~0.50
        activations = {10: 10.0, 20: 90.0}
        result = _compute_percentiles_for_features(activations, stats)
        mean_pct = (result[10] + result[20]) / 2.0
        assert abs(mean_pct - 0.50) < 0.02


# ===========================================================================
# TestComputeTopkPercentileActivation
# ===========================================================================


class TestComputeTopkPercentileActivation:
    """Test the full probe function with mocked model/SAE."""

    def _make_mock_sae(
        self, feature_activations: dict[int, float], n_features: int = 1000
    ):
        sae = MagicMock()

        def fake_encode(act):
            seq_len = act.shape[0]
            result = t.zeros(seq_len, n_features)
            for fid, val in feature_activations.items():
                result[:, fid] = val
            return result

        sae.encode = MagicMock(side_effect=fake_encode)
        return sae

    def _make_mock_gemma_tokenizer(self):
        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda text: list(range(len(text) // 4)))
        tok.decode = MagicMock(side_effect=lambda ids: "x" * (len(ids) * 4))
        return tok

    def test_basic_percentile_measurement(self):
        stats = _make_per_feature_stats([10, 20])
        sae = self._make_mock_sae({10: 50.0, 20: 90.0})
        gemma_tok = self._make_mock_gemma_tokenizer()

        with patch(
            "steering.eval.shuffled_conditioning_eval.get_probe_activations"
        ) as mock_act:
            mock_act.return_value = t.randn(1, 20, 256)
            result = compute_topk_percentile_activation(
                completion_text="x" * 40,
                pre_segment_text="y" * 40,
                top_k_feature_ids=[10, 20],
                per_feature_stats=stats,
                base_model=MagicMock(),
                sae=sae,
                gemma_tokenizer=gemma_tok,
                layer=8,
                hook_name="blocks.8.hook_resid_post",
                device=t.device("cpu"),
            )
            assert abs(result - 0.70) < 0.05  # (0.50 + 0.90) / 2

    def test_empty_completion_returns_nan(self):
        stats = _make_per_feature_stats([10])
        sae = self._make_mock_sae({10: 50.0})
        gemma_tok = MagicMock()
        gemma_tok.encode = MagicMock(return_value=[])

        with patch(
            "steering.eval.shuffled_conditioning_eval.get_probe_activations"
        ) as mock_act:
            mock_act.return_value = t.randn(1, 0, 256)
            result = compute_topk_percentile_activation(
                completion_text="",
                pre_segment_text="prefix",
                top_k_feature_ids=[10],
                per_feature_stats=stats,
                base_model=MagicMock(),
                sae=sae,
                gemma_tokenizer=gemma_tok,
                layer=8,
                hook_name="blocks.8.hook_resid_post",
                device=t.device("cpu"),
            )
            assert math.isnan(result)

    def test_compute_topk_with_monotonic_char_offsets(self):
        """Verify compute_topk_percentile_activation works correctly when
        token_to_char_offsets returns clamped (non-decreasing) offsets,
        simulating byte-fallback where per-token decode lengths shrink."""
        stats = _make_per_feature_stats([10])
        sae = self._make_mock_sae({10: 50.0})

        # Mock tokenizer that produces tokens where per-token decodes
        # don't sum to full text length (simulating byte fallback)
        gemma_tok = MagicMock()
        # 5 tokens for the full text
        gemma_tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        # Per-token decodes are shorter than expected (byte fallback)
        gemma_tok.decode = MagicMock(side_effect=lambda ids: "x" * max(len(ids), 1))

        # Patch token_to_char_offsets to return clamped monotonic offsets
        # pre_segment_text = "pre" (3 chars), completion = "comp" (4 chars)
        # full_text = "precomp" (7 chars)
        # offsets: [0, 1, 2, 3, 5, 7] — 5 tokens + sentinel
        # completion starts at char 3, so tokens 3,4 (0-indexed) are completion
        clamped_offsets = [0, 1, 2, 3, 5, 7]

        with (
            patch(
                "steering.eval.shuffled_conditioning_eval.get_probe_activations"
            ) as mock_act,
            patch(
                "steering.eval.shuffled_conditioning_eval.token_to_char_offsets"
            ) as mock_offsets,
        ):
            mock_act.return_value = t.randn(1, 5, 256)
            mock_offsets.return_value = clamped_offsets

            result = compute_topk_percentile_activation(
                completion_text="comp",
                pre_segment_text="pre",
                top_k_feature_ids=[10],
                per_feature_stats=stats,
                base_model=MagicMock(),
                sae=sae,
                gemma_tokenizer=gemma_tok,
                layer=8,
                hook_name="blocks.8.hook_resid_post",
                device=t.device("cpu"),
            )
            # SAE mock returns 50.0 for feature 10, and stats have p50=50.0
            # so the percentile should be approximately 0.50
            assert not math.isnan(result)
            assert abs(result - 0.50) < 0.05


# ===========================================================================
# TestAggregateResults
# ===========================================================================


class TestAggregateResults:
    """Tests for aggregate_results and related helpers."""

    def _make_result(
        self,
        doc_idx: int,
        real: list[tuple[str, float]],
        shuffled: list[tuple[str, float]],
    ) -> ShuffledConditioningResult:
        return ShuffledConditioningResult(
            doc_idx=doc_idx,
            target_segment_idx=1,
            n_top_k_features=2,
            top_k_feature_ids=[10, 20],
            ground_truth_text="gt",
            pre_segment_text="pre",
            shuffled_doc_idx=99,
            real_samples=real,
            shuffled_samples=shuffled,
        )

    def test_basic_aggregation(self) -> None:
        """Two docs: doc0 real [0.80, 0.90] mean=0.85, doc1 [0.70] mean=0.70."""
        r0 = self._make_result(
            0,
            real=[("a", 0.80), ("b", 0.90)],
            shuffled=[("d", 0.40)],
        )
        r1 = self._make_result(
            1,
            real=[("e", 0.70)],
            shuffled=[("g", 0.60)],
        )
        agg = aggregate_results([r0, r1], k=2)
        assert agg.n_documents == 2
        assert agg.k == 2
        # real: (0.85 + 0.70) / 2 = 0.775
        assert abs(agg.real_mean - 0.775) < 1e-6

    def test_nan_filtering(self) -> None:
        """NaN samples should be filtered out."""
        r = self._make_result(
            0,
            real=[("a", 0.80), ("b", float("nan"))],
            shuffled=[("d", 0.60)],
        )
        agg = aggregate_results([r], k=2)
        assert abs(agg.real_mean - 0.80) < 1e-6

    def test_yaml_round_trip(self) -> None:
        """serialize -> deserialize should preserve all fields."""
        r = self._make_result(
            0,
            real=[("hello world", 0.85)],
            shuffled=[("baz qux", 0.55)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.yaml"
            serialize_results([r], path)
            loaded = deserialize_results(path)

            assert len(loaded) == 1
            lr = loaded[0]
            assert lr.doc_idx == 0
            assert lr.pre_segment_text == "pre"
            assert lr.ground_truth_text == "gt"
            assert lr.top_k_feature_ids == [10, 20]
            assert lr.real_samples == [("hello world", 0.85)]
            assert lr.shuffled_samples == [("baz qux", 0.55)]

    def test_yaml_round_trip_with_descriptions(self) -> None:
        """top_k_feature_descriptions should survive serialization."""
        r = ShuffledConditioningResult(
            doc_idx=0,
            target_segment_idx=1,
            n_top_k_features=2,
            top_k_feature_ids=[10, 20],
            ground_truth_text="gt",
            pre_segment_text="pre",
            shuffled_doc_idx=99,
            top_k_feature_descriptions=["feature about cats", "feature about dogs"],
            real_samples=[("a", 0.80)],
            shuffled_samples=[("c", 0.50)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.yaml"
            serialize_results([r], path)
            loaded = deserialize_results(path)

            assert len(loaded) == 1
            lr = loaded[0]
            assert lr.top_k_feature_descriptions == [
                "feature about cats",
                "feature about dogs",
            ]

    def test_aggregate_all_nan_samples(self) -> None:
        """All samples are NaN -> aggregated means are NaN."""
        r = self._make_result(
            0,
            real=[("a", float("nan")), ("b", float("nan"))],
            shuffled=[("d", float("nan"))],
        )
        agg = aggregate_results([r], k=2)
        assert math.isnan(agg.real_mean)
        assert math.isnan(agg.real_stderr)
        assert math.isnan(agg.shuffled_mean)

    def test_aggregate_single_document(self) -> None:
        """Single document -> stderr is 0."""
        r = self._make_result(
            0,
            real=[("a", 0.80), ("b", 0.90)],
            shuffled=[("d", 0.60)],
        )
        agg = aggregate_results([r], k=2)
        assert agg.n_documents == 1
        # With 1 doc, stderr should be 0
        assert agg.real_stderr == 0.0
        assert agg.shuffled_stderr == 0.0

    def test_aggregate_empty_results(self) -> None:
        """Empty results list -> NaN aggregated values."""
        agg = aggregate_results([], k=2)
        assert agg.n_documents == 0
        assert math.isnan(agg.real_mean)
        assert math.isnan(agg.real_stderr)
        assert math.isnan(agg.shuffled_mean)


# ===========================================================================
# Fake model/tokenizer for generate_for_shuffled_document
# ===========================================================================


class _CharLevelTokenizer:
    """A fake tokenizer that encodes 1 token per character and round-trips text.

    Token IDs are character ordinals (e.g., 'A' -> 65).
    BOS token ID is 2, which decodes to the single character chr(2).
    This ensures ``token_to_char_offsets`` produces offsets that exactly
    match character positions in the original text, allowing
    ``assign_tokens_to_segments`` to work correctly.
    """

    bos_id = 2

    def encode(self, text: str, prepend=None, append=None) -> list[int]:
        tokens = [ord(c) for c in text]
        if prepend is not None:
            tokens = [prepend] + tokens
        if append is not None:
            tokens = tokens + [append]
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class _FakeModel:
    def __init__(self, vocab_size=300, seq_len=1024):
        self.vocab_size = vocab_size
        self.dec_config = MagicMock()
        self.dec_config.sequence_len = seq_len
        self.dec_config.n_kv_head = 4
        self.dec_config.n_head = 8
        self.dec_config.n_embd = 256
        self.dec_config.n_layer = 2
        self.encoder = MagicMock()
        self.encoder.config = MagicMock()
        self.encoder.config.num_buckets = 4
        self.encoder.return_value = pack_dense_encoder_output_reference(
            t.randn(1, 4, 1, 256),
            t.ones(1, 4, 1, dtype=t.bool),
        )

    def __call__(self, idx, **kwargs):
        B, T = idx.shape[0], idx.shape[1]
        logits = t.randn(B, T, self.vocab_size)
        logits[:, :, 50] = 10.0
        return logits


# ===========================================================================
# TestSelectEvalDocuments
# ===========================================================================


class TestSelectEvalDocuments:
    """Tests for select_eval_documents."""

    def test_filters_single_segment_docs(self) -> None:
        """Documents with < 2 segments should be excluded."""
        doc_segments = {
            0: [_make_segment([1], [10], segment_idx=0)],  # 1 segment, excluded
            1: [
                _make_segment([1], [10], segment_idx=0),
                _make_segment([2], [20], segment_idx=1),
            ],  # 2 segments, included
        }
        result = select_eval_documents(doc_segments, n_documents=10)
        assert len(result) == 1
        assert result[0][0] == 1

    def test_respects_n_documents(self) -> None:
        """Should return at most n_documents."""
        doc_segments = {
            i: [
                _make_segment([1], [10], segment_idx=0),
                _make_segment([2], [20], segment_idx=1),
            ]
            for i in range(10)
        }
        result = select_eval_documents(doc_segments, n_documents=3)
        assert len(result) == 3

    def test_target_segment_is_last(self) -> None:
        """Target segment should be the last list position."""
        doc_segments = {
            0: [
                _make_segment([1], [10], segment_idx=0),
                _make_segment([2], [20], segment_idx=1),
                _make_segment([3], [30], segment_idx=2),
            ],
        }
        result = select_eval_documents(doc_segments, n_documents=1)
        assert len(result) == 1
        doc_idx, target_seg_idx = result[0]
        assert doc_idx == 0
        assert target_seg_idx == 2  # last list position

    # -------------------------------------------------------------------
    # Context-window filtering
    # -------------------------------------------------------------------

    def _make_seg_with_chars(
        self,
        segment_idx: int,
        char_start: int,
        char_end: int,
    ) -> SegmentFeatures:
        """Build a segment with explicit char boundaries."""
        return SegmentFeatures(
            segment_idx=segment_idx,
            position=segment_idx * 128,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([10], dtype=t.int64),
            char_start=char_start,
            char_end=char_end,
        )

    def test_filters_docs_exceeding_context_window(self) -> None:
        """Documents whose tokens through the target segment exceed
        max_seq_len must be excluded."""
        tokenizer = _CharLevelTokenizer()
        max_seq_len = 100

        # Doc 0: text through target = 80 chars → 81 tokens (80 + BOS) → fits
        # Doc 1: text through target = 120 chars → 121 tokens → too long
        doc_segments = {
            0: [
                self._make_seg_with_chars(0, 0, 40),
                self._make_seg_with_chars(1, 40, 80),
            ],
            1: [
                self._make_seg_with_chars(0, 0, 60),
                self._make_seg_with_chars(1, 60, 120),
            ],
        }
        doc_texts = ["a" * 80, "b" * 120]

        result = select_eval_documents(
            doc_segments,
            n_documents=10,
            doc_texts=doc_texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        doc_indices = [r[0] for r in result]
        assert 0 in doc_indices, "Doc 0 fits and should be included"
        assert 1 not in doc_indices, "Doc 1 exceeds context window"

    def test_context_filter_boundary_exact_fit(self) -> None:
        """A document whose token count equals max_seq_len exactly should
        be included (the check is > not >=).

        With _CharLevelTokenizer, bos_prefix = chr(2) (1 char).
        text_through_target = (bos_prefix + doc_text)[1 : char_end].
        So char_end=100 → 99 chars → encode with BOS → 100 tokens = max_seq_len.
        """
        tokenizer = _CharLevelTokenizer()
        max_seq_len = 100

        doc_segments = {
            0: [
                self._make_seg_with_chars(0, 0, 50),
                self._make_seg_with_chars(1, 50, 100),
            ],
        }
        doc_texts = ["a" * 100]

        result = select_eval_documents(
            doc_segments,
            n_documents=10,
            doc_texts=doc_texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        assert len(result) == 1, "Document at exactly max_seq_len should be included"

    def test_context_filter_boundary_one_over(self) -> None:
        """A document with token count = max_seq_len + 1 must be excluded.

        With _CharLevelTokenizer, bos_prefix = chr(2) (1 char).
        text_through_target = (bos_prefix + doc_text)[1 : char_end].
        So char_end=101 → 100 chars → encode with BOS → 101 tokens.
        """
        tokenizer = _CharLevelTokenizer()
        max_seq_len = 100

        doc_segments = {
            0: [
                self._make_seg_with_chars(0, 0, 50),
                self._make_seg_with_chars(1, 50, 101),
            ],
        }
        doc_texts = ["a" * 101]

        result = select_eval_documents(
            doc_segments,
            n_documents=10,
            doc_texts=doc_texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        assert len(result) == 0, "Document 1 token over max_seq_len must be excluded"

    def test_no_context_filter_without_tokenizer(self) -> None:
        """When tokenizer args are omitted, no context filtering should
        be applied (backward compatibility)."""
        # Doc with char_end=10000 would exceed any reasonable context
        # window, but without tokenizer info it should pass through.
        doc_segments = {
            0: [
                self._make_seg_with_chars(0, 0, 5000),
                self._make_seg_with_chars(1, 5000, 10000),
            ],
        }

        result = select_eval_documents(doc_segments, n_documents=10)

        assert len(result) == 1, "Without tokenizer info, no filtering should occur"

    def test_context_filter_skips_out_of_range_doc_idx(self) -> None:
        """If doc_idx >= len(doc_texts), the document must be excluded
        (not raise an IndexError)."""
        tokenizer = _CharLevelTokenizer()

        doc_segments = {
            0: [
                self._make_seg_with_chars(0, 0, 20),
                self._make_seg_with_chars(1, 20, 40),
            ],
            999: [
                self._make_seg_with_chars(0, 0, 20),
                self._make_seg_with_chars(1, 20, 40),
            ],
        }
        doc_texts = ["a" * 40]  # only 1 doc, so idx 999 is out of range

        result = select_eval_documents(
            doc_segments,
            n_documents=10,
            doc_texts=doc_texts,
            tokenizer=tokenizer,
            max_seq_len=2048,
        )

        doc_indices = [r[0] for r in result]
        assert 0 in doc_indices, "Valid doc_idx should be included"
        assert 999 not in doc_indices, "Out-of-range doc_idx must be excluded"

    def test_context_filter_mixed_keeps_only_fitting(self) -> None:
        """With a mix of fitting and non-fitting docs, only fitting docs
        should be returned, up to n_documents."""
        tokenizer = _CharLevelTokenizer()
        max_seq_len = 50

        doc_segments = {}
        doc_texts = []
        # Build 10 docs: even-indexed have 30 chars (fit), odd have 60 (don't fit)
        for i in range(10):
            text_len = 30 if i % 2 == 0 else 60
            doc_segments[i] = [
                self._make_seg_with_chars(0, 0, text_len // 2),
                self._make_seg_with_chars(1, text_len // 2, text_len),
            ]
            doc_texts.append("x" * text_len)

        result = select_eval_documents(
            doc_segments,
            n_documents=10,
            doc_texts=doc_texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        result_indices = {r[0] for r in result}
        # Only even indices (0, 2, 4, 6, 8) have 30 chars → 31 tokens ≤ 50
        assert result_indices == {0, 2, 4, 6, 8}, (
            f"Only fitting docs should be selected, got {result_indices}"
        )

    def test_context_filter_preserves_deterministic_seed(self) -> None:
        """Filtering should not break the deterministic ordering from seed."""
        tokenizer = _CharLevelTokenizer()
        max_seq_len = 200

        doc_segments = {
            i: [
                self._make_seg_with_chars(0, 0, 50),
                self._make_seg_with_chars(1, 50, 100),
            ]
            for i in range(20)
        }
        doc_texts = ["a" * 100] * 20

        result_a = select_eval_documents(
            doc_segments, n_documents=5,
            doc_texts=doc_texts, tokenizer=tokenizer,
            max_seq_len=max_seq_len, seed=42,
        )
        result_b = select_eval_documents(
            doc_segments, n_documents=5,
            doc_texts=doc_texts, tokenizer=tokenizer,
            max_seq_len=max_seq_len, seed=42,
        )

        assert result_a == result_b, "Same seed must produce identical results"

    def test_context_filter_different_seed_different_order(self) -> None:
        """Different seeds should produce different orderings."""
        tokenizer = _CharLevelTokenizer()
        max_seq_len = 200

        doc_segments = {
            i: [
                self._make_seg_with_chars(0, 0, 50),
                self._make_seg_with_chars(1, 50, 100),
            ]
            for i in range(20)
        }
        doc_texts = ["a" * 100] * 20

        result_a = select_eval_documents(
            doc_segments, n_documents=10,
            doc_texts=doc_texts, tokenizer=tokenizer,
            max_seq_len=max_seq_len, seed=42,
        )
        result_b = select_eval_documents(
            doc_segments, n_documents=10,
            doc_texts=doc_texts, tokenizer=tokenizer,
            max_seq_len=max_seq_len, seed=99,
        )

        assert result_a != result_b, "Different seeds should produce different order"

    def test_partial_tokenizer_args_raises(self) -> None:
        """Providing only some of doc_texts/tokenizer/max_seq_len must
        raise ValueError — silent no-op filtering is a bug."""
        tokenizer = _CharLevelTokenizer()

        doc_segments = {
            0: [
                self._make_seg_with_chars(0, 0, 500),
                self._make_seg_with_chars(1, 500, 1000),
            ],
        }
        doc_texts = ["a" * 1000]

        # Only doc_texts
        with pytest.raises(ValueError, match="got 1/3"):
            select_eval_documents(
                doc_segments, n_documents=10, doc_texts=doc_texts,
            )

        # Only tokenizer
        with pytest.raises(ValueError, match="got 1/3"):
            select_eval_documents(
                doc_segments, n_documents=10, tokenizer=tokenizer,
            )

        # Only max_seq_len
        with pytest.raises(ValueError, match="got 1/3"):
            select_eval_documents(
                doc_segments, n_documents=10, max_seq_len=10,
            )

        # Two of three
        with pytest.raises(ValueError, match="got 2/3"):
            select_eval_documents(
                doc_segments, n_documents=10,
                doc_texts=doc_texts, tokenizer=tokenizer,
            )


# ===========================================================================
# TestGenerateForShuffledDocument
# ===========================================================================


class TestGenerateForShuffledDocument:
    """Tests for generate_for_shuffled_document."""

    def test_produces_both_conditions(self) -> None:
        """Should produce 4 samples: 2 real and 2 shuffled."""
        model = _FakeModel()
        tokenizer = _CharLevelTokenizer()

        # BOS decode = chr(2), length 1
        bos_prefix_len = len(tokenizer.decode([tokenizer.bos_id]))  # 1
        doc_text = "A" * 200

        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1, 2], dtype=t.int32),
            feature_mags=t.tensor([5.0, 6.0], dtype=t.float32),
            char_start=bos_prefix_len,  # start after BOS prefix
            char_end=bos_prefix_len + 80,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([3, 4], dtype=t.int32),
            feature_mags=t.tensor([7.0, 8.0], dtype=t.float32),
            char_start=bos_prefix_len + 80,
            char_end=bos_prefix_len + 160,
        )

        shuffled_seg = _make_segment([99, 100], [3, 4], segment_idx=1)

        # Boundaries for discretization (2 features, 4 boundaries)
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        pre_text, gt_text, samples = generate_for_shuffled_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=[seg0, seg1],
            target_segment_idx=1,
            shuffled_segment=shuffled_seg,
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            samples_per_condition=2,
            temperature=0.0,
        )

        assert len(samples) == 4
        conditions = [s.condition for s in samples]
        assert conditions.count("real") == 2  # ty: ignore[invalid-argument-type]
        assert conditions.count("shuffled") == 2  # ty: ignore[invalid-argument-type]
        # Verify ground truth and pre-segment text are correct substrings
        # seg1 spans chars [bos_prefix_len+80, bos_prefix_len+160) in BOS space
        # which is doc_text[80:160] = "A" * 80
        assert gt_text == "A" * 80
        # pre_segment_text is doc_text[0:80] = "A" * 80
        assert pre_text == "A" * 80

    def test_uses_direct_target_features_slot(self) -> None:
        model = _FakeModel()
        tokenizer = _CharLevelTokenizer()
        bos_prefix_len = len(tokenizer.decode([tokenizer.bos_id]))
        doc_text = "A" * 200
        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1, 2], dtype=t.int32),
            feature_mags=t.tensor([5.0, 6.0], dtype=t.float32),
            char_start=bos_prefix_len,
            char_end=bos_prefix_len + 80,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([3, 4], dtype=t.int32),
            feature_mags=t.tensor([7.0, 8.0], dtype=t.float32),
            char_start=bos_prefix_len + 80,
            char_end=bos_prefix_len + 160,
        )
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        generate_for_shuffled_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=[seg0, seg1],
            target_segment_idx=1,
            shuffled_segment=_make_segment([99, 100], [3, 4], segment_idx=1),
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            samples_per_condition=1,
            temperature=0.0,
        )

        enc_fids = model.encoder.call_args_list[0].args[0]
        assert enc_fids.shape[1] == 2
        assert enc_fids[0, 1, 0].item() == 3

    def test_generate_shuffled_n_gen_tokens_from_segment_assignment(self) -> None:
        """Verify that n_gen_tokens is derived from character-level segment
        assignment (assign_tokens_to_segments + cummax), not from
        len(tokenizer.encode(ground_truth))."""
        model = _FakeModel()
        tokenizer = _CharLevelTokenizer()

        bos_prefix_len = len(tokenizer.decode([tokenizer.bos_id]))  # 1
        doc_text = "B" * 300

        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_prefix_len,
            char_end=bos_prefix_len + 100,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([2], dtype=t.int32),
            feature_mags=t.tensor([6.0], dtype=t.float32),
            char_start=bos_prefix_len + 100,
            char_end=bos_prefix_len + 200,
        )

        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        # With char-level tokenizer, the expected n_gen_tokens is the number
        # of tokens assigned to the target segment by assign_tokens_to_segments.
        # For a 1-char-per-token tokenizer, that's exactly the char span of seg1.
        expected_n_gen = seg1.char_end - seg1.char_start  # 100

        # Patch _decode_from_prefill to capture max_new_tokens argument
        captured_max_new_tokens: list[int] = []

        with patch(
            "steering.eval.shuffled_conditioning_eval._decode_from_prefill"
        ) as mock_decode:

            def capture_decode(*args, **kwargs):
                captured_max_new_tokens.append(
                    kwargs.get("max_new_tokens", args[5] if len(args) > 5 else None)
                )
                return [50] * kwargs.get("max_new_tokens", 1)

            mock_decode.side_effect = capture_decode

            pre_text, gt_result, samples = generate_for_shuffled_document(
                model=model,
                tokenizer=tokenizer,
                doc_text=doc_text,
                segments=[seg0, seg1],
                target_segment_idx=1,
                shuffled_segment=shuffled_seg,
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                device=t.device("cpu"),
                samples_per_condition=1,
                temperature=0.0,
            )

        # All calls to _decode_from_prefill should use the same n_gen_tokens
        assert len(captured_max_new_tokens) == 2  # real, shuffled
        for n in captured_max_new_tokens:
            assert n == expected_n_gen

    def test_generate_shuffled_empty_ground_truth_raises(self) -> None:
        """Segment with char_start == char_end raises ValueError."""
        model = _FakeModel()
        tokenizer = _CharLevelTokenizer()

        bos_prefix_len = len(tokenizer.decode([tokenizer.bos_id]))  # 1
        doc_text = "C" * 200

        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_prefix_len,
            char_end=bos_prefix_len + 80,
        )
        # Empty segment — char_start == char_end
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([2], dtype=t.int32),
            feature_mags=t.tensor([6.0], dtype=t.float32),
            char_start=bos_prefix_len + 80,
            char_end=bos_prefix_len + 80,  # same as char_start
        )

        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        with pytest.raises(ValueError, match="non-positive char span"):
            generate_for_shuffled_document(
                model=model,
                tokenizer=tokenizer,
                doc_text=doc_text,
                segments=[seg0, seg1],
                target_segment_idx=1,
                shuffled_segment=shuffled_seg,
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                device=t.device("cpu"),
                samples_per_condition=1,
                temperature=0.0,
            )


# ===========================================================================
# TestFullPipelineIntegration
# ===========================================================================


class TestFullPipelineIntegration:
    """End-to-end integration test exercising the full shuffled conditioning pipeline."""

    def test_full_pipeline(self) -> None:
        """Walk through the entire pipeline: select docs, prepare segments,
        select features, generate, aggregate, and YAML round-trip."""
        tokenizer = _CharLevelTokenizer()
        model = _FakeModel()

        # BOS prefix: tokenizer.decode([2]) = chr(2) (1 char)
        bos_prefix = tokenizer.decode([tokenizer.bos_id])
        bos_len = len(bos_prefix)
        assert bos_len == 1  # chr(2)

        # --- Step 1: Build 3 documents with 3 segments each ---
        # Segments use float32 feature_mags (raw, pre-threshold)
        # BOS-adjusted char offsets: char_start = bos_len + i*100
        # With _CharLevelTokenizer, 1 token = 1 char, so keep segments
        # small enough to fit in seq_len=1024.
        n_features = 5
        seg_chars = 100  # chars per segment
        docs: dict[int, list[SegmentFeatures]] = {}
        for doc_idx in range(3):
            segs: list[SegmentFeatures] = []
            for seg_i in range(3):
                segs.append(
                    SegmentFeatures(
                        segment_idx=seg_i,
                        position=seg_i * 128,
                        feature_ids=t.tensor(
                            list(range(doc_idx * 10, doc_idx * 10 + n_features)),
                            dtype=t.int32,
                        ),
                        feature_mags=t.tensor(
                            [float(j + 1 + seg_i) for j in range(n_features)],
                            dtype=t.float32,
                        ),
                        char_start=bos_len + seg_i * seg_chars,
                        char_end=bos_len + (seg_i + 1) * seg_chars,
                    )
                )
            docs[doc_idx] = segs

        # --- Step 2: select_eval_documents ---
        selected = select_eval_documents(docs, n_documents=3)
        assert len(selected) == 3
        # Each doc has 3 segments, so target_seg_idx should be 2 (last)
        for _doc_idx, target_seg_idx in selected:
            assert target_seg_idx == 2

        # --- Step 3: prepare_segments on one doc ---
        # Build boundaries for discretization: shape (max_feature_id + 1, n_boundaries)
        max_fid = max(
            fid
            for segs in docs.values()
            for seg in segs
            for fid in seg.feature_ids.tolist()
        )
        n_boundaries = 4
        boundaries = (
            t.linspace(1.0, 10.0, n_boundaries)
            .unsqueeze(0)
            .expand(max_fid + 1, -1)
            .to(t.float64)
        )

        doc0_segs = docs[0]
        prepared = prepare_segments(
            doc0_segs, boundaries, feature_cfg=None, precomputed_thresholds=None
        )
        assert len(prepared) > 0
        # Prepared segments should have int64 mags (bucket indices)
        for seg in prepared:
            assert seg.feature_mags.dtype == t.int64

        # --- Step 4: select_top_k_features on the raw target segment ---
        # Use raw segment (float mags) with per-feature stats
        raw_target = doc0_segs[2]  # target is last segment
        all_fids = [fid for seg in doc0_segs for fid in seg.feature_ids.tolist()]
        per_feature_stats = _make_per_feature_stats(list(set(all_fids)))
        top_k_ids = select_top_k_features(raw_target, per_feature_stats, k=3)
        assert len(top_k_ids) == 3

        # --- Step 5: Create a shuffled segment (int64 bucket mags, post-threshold) ---
        shuffled_seg = _make_segment(
            feature_ids=[99, 100, 101],
            feature_mags=[2, 3, 1],
            segment_idx=2,
        )
        assert shuffled_seg.feature_mags.dtype == t.int64

        # --- Step 6: generate_for_shuffled_document ---
        # Need doc_text long enough for char slicing
        doc_text = "X" * (3 * seg_chars)  # 3 segments * seg_chars each

        pre_text, gt_text, samples = generate_for_shuffled_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=doc0_segs,
            target_segment_idx=2,
            shuffled_segment=shuffled_seg,
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            samples_per_condition=2,
            temperature=0.0,
        )

        # --- Step 7: Verify 4 samples: 2 real, 2 shuffled ---
        assert len(samples) == 4
        conditions = [s.condition for s in samples]
        assert conditions.count("real") == 2  # ty: ignore[invalid-argument-type]
        assert conditions.count("shuffled") == 2  # ty: ignore[invalid-argument-type]
        assert len(pre_text) > 0
        assert len(gt_text) > 0

        # --- Step 8: Build ShuffledConditioningResult with hardcoded percentiles ---
        result = ShuffledConditioningResult(
            doc_idx=0,
            target_segment_idx=2,
            n_top_k_features=3,
            top_k_feature_ids=top_k_ids,
            ground_truth_text=gt_text,
            pre_segment_text=pre_text,
            shuffled_doc_idx=1,
            real_samples=[(samples[0].text, 0.85), (samples[1].text, 0.90)],
            shuffled_samples=[(samples[2].text, 0.55), (samples[3].text, 0.60)],
        )

        # --- Step 9: aggregate_results ---
        agg = aggregate_results([result], k=3)
        assert agg.n_documents == 1
        assert agg.k == 3
        # real: (0.85 + 0.90) / 2 = 0.875
        assert abs(agg.real_mean - 0.875) < 1e-6
        # shuffled: (0.55 + 0.60) / 2 = 0.575
        assert abs(agg.shuffled_mean - 0.575) < 1e-6

        # --- Step 10: YAML round-trip ---
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "integration_results.yaml"
            serialize_results([result], path)
            loaded = deserialize_results(path)

            assert len(loaded) == 1
            lr = loaded[0]
            assert lr.doc_idx == result.doc_idx
            assert lr.target_segment_idx == result.target_segment_idx
            assert lr.n_top_k_features == result.n_top_k_features
            assert lr.top_k_feature_ids == result.top_k_feature_ids
            assert lr.ground_truth_text == result.ground_truth_text
            assert lr.pre_segment_text == result.pre_segment_text
            assert lr.shuffled_doc_idx == result.shuffled_doc_idx
            assert len(lr.real_samples) == 2
            assert len(lr.shuffled_samples) == 2
            # Verify percentile values survive round-trip
            assert abs(lr.real_samples[0][1] - 0.85) < 1e-6
            assert abs(lr.real_samples[1][1] - 0.90) < 1e-6
            assert abs(lr.shuffled_samples[1][1] - 0.60) < 1e-6


# ===========================================================================
# New tests for character-level segment assignment interaction
# ===========================================================================


class TestCharLevelSegmentAssignment:
    """Tests verifying that generate_for_shuffled_document correctly uses
    character-level segment assignment for prefix/gen split."""

    def test_gen_token_count_derived_from_segment_assignment(self) -> None:
        """n_gen_tokens should match the number of tokens assigned to the
        target segment by assign_tokens_to_segments, not
        len(tokenizer.encode(ground_truth))."""
        tokenizer = _CharLevelTokenizer()
        model = _FakeModel()

        bos_len = len(tokenizer.decode([tokenizer.bos_id]))  # 1
        # seg0: 50 chars, seg1: 70 chars
        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_len,
            char_end=bos_len + 50,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([2], dtype=t.int32),
            feature_mags=t.tensor([6.0], dtype=t.float32),
            char_start=bos_len + 50,
            char_end=bos_len + 120,
        )

        doc_text = "A" * 50 + "B" * 70
        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        # Manually compute expected n_gen via assign_tokens_to_segments
        text_through_target = doc_text[: seg1.char_end - bos_len]
        all_tokens = tokenizer.encode(text_through_target, prepend=tokenizer.bos_id)
        char_offsets = token_to_char_offsets(tokenizer, all_tokens)
        token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
        token_ends = t.tensor(char_offsets[1:], dtype=t.int64)
        seg_char_starts, seg_char_ends = _segment_bounds_for_assignment(
            [seg0, seg1],
            token_starts,
            token_ends,
        )
        assignments = assign_tokens_to_segments(
            token_starts,
            token_ends,
            seg_char_starts,
            seg_char_ends,
        )
        assignments = t.cummax(assignments, dim=0).values
        expected_n_gen = int((assignments == 1).sum().item())
        assert expected_n_gen > 0

        captured_max_new_tokens: list[int] = []
        with patch(
            "steering.eval.shuffled_conditioning_eval._decode_from_prefill"
        ) as mock_decode:
            mock_decode.side_effect = lambda *a, **kw: (
                captured_max_new_tokens.append(kw["max_new_tokens"])
                or [50] * kw["max_new_tokens"]
            )

            generate_for_shuffled_document(
                model=model,
                tokenizer=tokenizer,
                doc_text=doc_text,
                segments=[seg0, seg1],
                target_segment_idx=1,
                shuffled_segment=shuffled_seg,
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                device=t.device("cpu"),
                samples_per_condition=1,
                temperature=0.0,
            )

        assert len(captured_max_new_tokens) == 2
        for n in captured_max_new_tokens:
            assert n == expected_n_gen

    def test_shuffled_condition_uses_different_features(self) -> None:
        """The shuffled condition's encoder inputs should use the donor
        segment's features, not the real segment's features."""
        tokenizer = _CharLevelTokenizer()
        model = _FakeModel()

        bos_len = len(tokenizer.decode([tokenizer.bos_id]))
        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_len,
            char_end=bos_len + 40,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([3], dtype=t.int32),
            feature_mags=t.tensor([7.0], dtype=t.float32),
            char_start=bos_len + 40,
            char_end=bos_len + 80,
        )

        # Donor segment has completely different features
        shuffled_seg = _make_segment([99], [3], segment_idx=1)

        doc_text = "A" * 80
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        # Capture the encoder inputs for each condition
        captured_inputs: list[tuple[str, t.Tensor]] = []
        encoder_return_value = model.encoder.return_value

        def capture_encoder_call(fids, mags, mask):
            captured_inputs.append(("call", fids.clone()))
            return encoder_return_value

        model.encoder.side_effect = capture_encoder_call

        generate_for_shuffled_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=[seg0, seg1],
            target_segment_idx=1,
            shuffled_segment=shuffled_seg,
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            samples_per_condition=1,
            temperature=0.0,
        )

        assert len(captured_inputs) == 2
        # Real (idx 0) and shuffled (idx 1) should have different feature IDs
        # in the target segment position
        real_fids = captured_inputs[0][1]
        shuffled_fids = captured_inputs[1][1]
        # Target segment is segment_idx=1, so check position 1
        assert not t.equal(real_fids[0, 1], shuffled_fids[0, 1])

    def test_all_conditions_produce_correct_sample_count(self) -> None:
        """Verify exactly samples_per_condition samples per condition with
        correct condition labels."""
        tokenizer = _CharLevelTokenizer()
        model = _FakeModel()

        bos_len = len(tokenizer.decode([tokenizer.bos_id]))
        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_len,
            char_end=bos_len + 40,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([2], dtype=t.int32),
            feature_mags=t.tensor([6.0], dtype=t.float32),
            char_start=bos_len + 40,
            char_end=bos_len + 80,
        )

        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        doc_text = "A" * 80
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        for spc in [1, 3, 5]:
            _, _, samples = generate_for_shuffled_document(
                model=model,
                tokenizer=tokenizer,
                doc_text=doc_text,
                segments=[seg0, seg1],
                target_segment_idx=1,
                shuffled_segment=shuffled_seg,
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                device=t.device("cpu"),
                samples_per_condition=spc,
                temperature=0.0,
            )

            assert len(samples) == 2 * spc
            conditions = [s.condition for s in samples]
            assert conditions.count("real") == spc  # ty: ignore[invalid-argument-type]
            assert conditions.count("shuffled") == spc  # ty: ignore[invalid-argument-type]

    def test_context_window_exceeded_raises(self) -> None:
        """Prefix + gen exceeding model.dec_config.sequence_len should raise."""
        tokenizer = _CharLevelTokenizer()
        # Very small context window
        model = _FakeModel(seq_len=20)

        bos_len = len(tokenizer.decode([tokenizer.bos_id]))
        # seg0: 30 chars, seg1: 30 chars -> total = 60 + BOS = 61 > 20
        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_len,
            char_end=bos_len + 30,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([2], dtype=t.int32),
            feature_mags=t.tensor([6.0], dtype=t.float32),
            char_start=bos_len + 30,
            char_end=bos_len + 60,
        )

        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        doc_text = "A" * 60
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        with pytest.raises(ValueError, match="exceeds context window"):
            generate_for_shuffled_document(
                model=model,
                tokenizer=tokenizer,
                doc_text=doc_text,
                segments=[seg0, seg1],
                target_segment_idx=1,
                shuffled_segment=shuffled_seg,
                boundaries=boundaries,
                threshold_cfg=None,
                precomputed_thresholds=None,
                device=t.device("cpu"),
                samples_per_condition=1,
                temperature=0.0,
            )

    def test_generate_with_small_target_segment(self) -> None:
        """A very small target segment (few chars) should still work
        as long as it encodes to at least 1 token."""
        tokenizer = _CharLevelTokenizer()
        model = _FakeModel()

        bos_len = len(tokenizer.decode([tokenizer.bos_id]))  # 1

        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1, 2], dtype=t.int32),
            feature_mags=t.tensor([5.0, 6.0], dtype=t.float32),
            char_start=bos_len,
            char_end=bos_len + 50,
        )
        # Very small target segment — only 3 chars
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([3], dtype=t.int32),
            feature_mags=t.tensor([7.0], dtype=t.float32),
            char_start=bos_len + 50,
            char_end=bos_len + 53,
        )

        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        doc_text = "A" * 53
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        pre_text, gt_text, samples = generate_for_shuffled_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=[seg0, seg1],
            target_segment_idx=1,
            shuffled_segment=shuffled_seg,
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            samples_per_condition=1,
            temperature=0.0,
        )

        assert len(samples) == 2
        assert len(gt_text) == 3  # 3 chars in the target segment
        assert len(pre_text) == 50  # chars before target segment

    def test_monotonic_char_offsets_with_byte_fallback(self) -> None:
        """Test that clamped char offsets from the monotonicity guarantee
        in token_to_char_offsets interact correctly with
        assign_tokens_to_segments."""

        class _NonMonotonicTokenizer:
            """Tokenizer where progressive decode produces non-monotonic
            raw lengths (simulating byte-fallback completion)."""

            bos_id = 2

            def encode(self, text: str, prepend=None, append=None) -> list[int]:
                # 1 token per character
                tokens = list(range(100, 100 + len(text)))
                if prepend is not None:
                    tokens = [prepend] + tokens
                if append is not None:
                    tokens = tokens + [append]
                return tokens

            def decode(self, ids: list[int]) -> str:
                # Simulate a tokenizer where decoding prefixes can shrink:
                # tokens [2, 100, 101] -> "XAB" (3 chars)
                # but token [100] alone -> "ZZ" (2 chars -- e.g., replacement chars)
                # This causes the fast path check to fail, triggering progressive decode.
                # For progressive decode, we just return predictable lengths.
                result = []
                for tok_id in ids:
                    if tok_id == 2:
                        result.append("X")  # BOS
                    else:
                        result.append(chr(tok_id - 100 + ord("A")))
                return "".join(result)

        tokenizer = _NonMonotonicTokenizer()
        model = _FakeModel()

        bos_len = len(tokenizer.decode([tokenizer.bos_id]))  # 1 ("X")
        seg0 = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1], dtype=t.int32),
            feature_mags=t.tensor([5.0], dtype=t.float32),
            char_start=bos_len,
            char_end=bos_len + 10,
        )
        seg1 = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([2], dtype=t.int32),
            feature_mags=t.tensor([6.0], dtype=t.float32),
            char_start=bos_len + 10,
            char_end=bos_len + 20,
        )

        shuffled_seg = _make_segment([99], [3], segment_idx=1)
        doc_text = "".join(chr(ord("A") + i % 26) for i in range(20))
        boundaries = t.linspace(1.0, 10.0, 4).unsqueeze(0).expand(200, -1).to(t.float64)

        # Should not crash — the clamped char offsets should produce valid
        # assignments even if the raw progressive decode had non-monotonic behavior.
        pre_text, gt_text, samples = generate_for_shuffled_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=[seg0, seg1],
            target_segment_idx=1,
            shuffled_segment=shuffled_seg,
            boundaries=boundaries,
            threshold_cfg=None,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            samples_per_condition=1,
            temperature=0.0,
        )

        assert len(samples) == 2
        assert len(gt_text) > 0
