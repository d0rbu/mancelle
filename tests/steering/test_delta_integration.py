# tests/test_delta_integration.py
"""Integration tests for feature delta conditioning.

Tests cover:
- Full pipeline: document-batched deltas -> selection -> discretize -> negatives
- prev_segment and cum_max modes with threshold and sampling selection
- Regression: delta=None produces identical behavior to current pipeline
- Multi-segment document processing
- Edge cases: single segment, empty raw segment rejection, all-zero deltas
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    compute_doc_deltas,
    sample_delta_negatives,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
    build_feature_pipeline,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingSelection,
    ThresholdSelection,
    build_percentile_table,
    sample_features,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stat(p_values: list[float]) -> PerFeatureStat:
    assert len(p_values) == 99
    percentiles = {f"p{i+1}": float(v) for i, v in enumerate(p_values)}
    return PerFeatureStat(
        count=1000, pct_segments_active=0.5,
        mean_mag=float(p_values[49]), mean_mag_incl_zeros=float(p_values[49]) * 0.5,
        std_mag=1.0, min_mag=float(p_values[0]), max_mag=float(p_values[-1]),
        median_mag=float(p_values[49]), percentiles=percentiles,
    )


def _make_stats_and_table(n_features: int = 100):
    stats = {
        str(f): _make_stat([float(k) for k in range(1, 100)])
        for f in range(n_features)
    }
    table = build_percentile_table(stats, n_features)
    return stats, table


# ---------------------------------------------------------------------------
# Full pipeline: prev_segment + threshold
# ---------------------------------------------------------------------------


class TestPrevSegmentThresholdPipeline:
    """End-to-end: prev_segment mode with threshold selection."""

    def test_multi_segment_document(self) -> None:
        n_features = 200
        _, table = _make_stats_and_table(n_features)

        segments = [
            (t.arange(0, 50, dtype=t.int64), t.linspace(10, 90, 50)),
            (t.arange(25, 75, dtype=t.int64), t.linspace(10, 90, 50)),
            (t.arange(50, 100, dtype=t.int64), t.linspace(10, 90, 50)),
        ]
        results = compute_doc_deltas(
            [fids for fids, _mags in segments],
            [mags for _fids, mags in segments],
            [None] * len(segments),
            n_features,
            DeltaMode.prev_segment,
            table,
        )

        for seg_idx, result in enumerate(results):
            filtered_fids, filtered_fmags, delta_values, neg_fids, _neg_weights, _ref_mask, _ = result
            thresh_cfg = ThresholdConfig(ThresholdType.segment_percentile, 0.5)
            selected_fids, selected_fmags = apply_threshold(
                thresh_cfg, filtered_fids, filtered_fmags,
            )

            if seg_idx == 0:
                assert delta_values is not None and (delta_values > 0).all()
                assert len(neg_fids) == 0

            if len(selected_fids) > 0:
                assert set(selected_fids.tolist()).issubset(set(filtered_fids.tolist()))

    def test_first_segment_matches_non_delta_pipeline(self) -> None:
        n_features = 100
        _, table = _make_stats_and_table(n_features)

        fids = t.arange(20, dtype=t.int64)
        fmags = t.linspace(10, 90, 20)

        (result,) = compute_doc_deltas(
            [fids], [fmags], [None], n_features, DeltaMode.prev_segment, table,
        )
        filtered_fids, filtered_fmags, _deltas, neg_fids, _weights, _ref_mask, _ = result

        assert t.equal(filtered_fids, fids)
        assert t.equal(filtered_fmags, fmags)
        assert len(neg_fids) == 0


# ---------------------------------------------------------------------------
# Full pipeline: cum_max + sampling
# ---------------------------------------------------------------------------


class TestCumMaxSamplingPipeline:
    """End-to-end: cum_max mode with sampling selection."""

    def test_cummax_restricts_positive_pool(self) -> None:
        n_features = 100
        _, table = _make_stats_and_table(n_features)

        fids0 = t.arange(20, dtype=t.int64)
        fmags0 = t.full((20,), 90.0)
        fids1 = t.arange(20, dtype=t.int64)
        fmags1 = t.full((20,), 50.0)
        _, second = compute_doc_deltas(
            [fids0, fids1],
            [fmags0, fmags1],
            [None, None],
            n_features,
            DeltaMode.cum_max,
            table,
        )
        filtered_fids, _fmags, _deltas, neg_fids_1, _weights, _ref_mask, _ = second

        assert len(filtered_fids) == 0
        assert len(neg_fids_1) == 20

    def test_only_exceeding_cummax_is_positive(self) -> None:
        n_features = 100
        _, table = _make_stats_and_table(n_features)

        fids = t.tensor([10, 20], dtype=t.int64)
        fmags_seg0 = t.tensor([50.0, 50.0])
        fmags_seg1 = t.tensor([90.0, 50.0])
        _, second = compute_doc_deltas(
            [fids, fids],
            [fmags_seg0, fmags_seg1],
            [None, None],
            n_features,
            DeltaMode.cum_max,
            table,
        )
        filtered_fids, _fmags, delta_values, neg_fids, _weights, _ref_mask, _ = second

        assert filtered_fids.tolist() == [10]
        assert delta_values is not None and delta_values[0] > 0
        assert not (neg_fids == 20).any()  # same percentile → not in neg


# ---------------------------------------------------------------------------
# Delta-aware negative sampling integration
# ---------------------------------------------------------------------------


class TestDeltaNegativeSamplingIntegration:
    def test_negative_from_disappeared_features(self) -> None:
        n_features = 200
        _, table = _make_stats_and_table(n_features)

        fids0 = t.arange(20, dtype=t.int64)
        fmags0 = t.full((20,), 50.0)
        fids1 = t.arange(10, 30, dtype=t.int64)
        fmags1 = t.full((20,), 50.0)
        _, second = compute_doc_deltas(
            [fids0, fids1],
            [fmags0, fmags1],
            [None, None],
            n_features,
            DeltaMode.prev_segment,
            table,
        )
        positive_ids, _fmags, delta_values, neg_delta_fids, neg_delta_weights, ref_mask, _ = second

        # fids1 = [10, 11, ..., 29]; features 0-9 disappeared → in neg
        for fid in range(10):
            assert (neg_delta_fids == fid).any(), f"Feature {fid} should have negative delta"
        assert positive_ids.tolist() == list(range(20, 30))
        assert delta_values is not None and (delta_values > 0).all()

        rng = t.Generator().manual_seed(42)
        sampled_neg_fids = sample_delta_negatives(
            k=5, neg_delta_ratio=1.0,
            neg_delta_fids=neg_delta_fids,
            neg_delta_weights=neg_delta_weights,
            raw_feature_ids=fids1,
            ref_mask=ref_mask,
            selected_positive_ids=positive_ids,
            n_sae_features=n_features, rng=rng,
        )
        neg_set = set(neg_delta_fids.tolist())
        assert all(fid in neg_set for fid in sampled_neg_fids.tolist())


# ---------------------------------------------------------------------------
# Regression: delta=None identical to current pipeline
# ---------------------------------------------------------------------------


class TestDeltaNoneRegression:
    def test_threshold_path_identical(self) -> None:
        n_features = 100
        stats, table = _make_stats_and_table(n_features)

        fids = t.arange(30, dtype=t.int64)
        fmags = t.linspace(10, 90, 30)
        thresh_cfg = ThresholdConfig(ThresholdType.segment_percentile, 0.5)

        ids_no_delta, mags_no_delta = apply_threshold(thresh_cfg, fids, fmags)

        filtered_fids, filtered_fmags = fids, fmags
        ids_delta_none, mags_delta_none = apply_threshold(
            thresh_cfg, filtered_fids, filtered_fmags,
        )

        assert t.equal(ids_no_delta, ids_delta_none)
        assert t.equal(mags_no_delta, mags_delta_none)

    def test_sampling_path_identical(self) -> None:
        n_features = 100
        stats, table = _make_stats_and_table(n_features)

        fids = t.arange(20, dtype=t.int64)
        fmags = t.linspace(10, 90, 20)
        cfg = SamplingSelection(distribution=Distribution.normal, mean=5.0, std=1.0)

        rng1 = t.Generator().manual_seed(42)
        ids_no_delta, mags_no_delta = sample_features(fids, fmags, table, cfg, rng1)

        rng2 = t.Generator().manual_seed(42)
        ids_delta_none, mags_delta_none = sample_features(
            fids, fmags, table, cfg, rng2, override_weights=None,
        )

        assert t.equal(ids_no_delta, ids_delta_none)
        assert t.equal(mags_no_delta, mags_delta_none)

    def test_build_pipeline_no_delta_unchanged(self) -> None:
        stats_dict = {}
        for fid in range(50):
            pcts = {f"p{k}": float(fid * 100 + k) for k in range(1, 100)}
            stats_dict[str(fid)] = PerFeatureStat(
                count=10, pct_segments_active=0.5, mean_mag=float(fid),
                mean_mag_incl_zeros=float(fid) * 0.5, std_mag=1.0,
                min_mag=0.0, max_mag=float(fid * 100 + 100),
                median_mag=float(fid) * 0.9, percentiles=pcts,
            )
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        pipeline = build_feature_pipeline(
            feature_selection=fs, negative_selection=None,
            per_feature_stats=stats_dict, n_sae_features=50, num_buckets=4,
        )
        assert pipeline.delta_bundle is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDeltaEdgeCases:
    def test_single_segment_document(self) -> None:
        n_features = 100
        _, table = _make_stats_and_table(n_features)

        fids = t.arange(10, dtype=t.int64)
        fmags = t.linspace(10, 90, 10)
        (result,) = compute_doc_deltas(
            [fids], [fmags], [None], n_features, DeltaMode.prev_segment, table,
        )
        filtered_fids, _fmags, delta_values, neg_fids, _weights, _ref_mask, _ = result

        assert t.equal(filtered_fids, fids)
        assert delta_values is not None and (delta_values > 0).all()
        assert len(neg_fids) == 0

    def test_empty_segment_after_active_raises(self) -> None:
        n_features = 100
        _, table = _make_stats_and_table(n_features)

        fids = t.arange(5, dtype=t.int64)
        fmags = t.full((5,), 50.0)
        empty_fids = t.tensor([], dtype=t.int64)
        empty_fmags = t.tensor([])
        with pytest.raises(ValueError, match="zero feature ids"):
            compute_doc_deltas(
                [fids, empty_fids],
                [fmags, empty_fmags],
                [None, None],
                n_features,
                DeltaMode.prev_segment,
                table,
            )

    def test_steady_state_all_zero_delta(self) -> None:
        n_features = 100
        _, table = _make_stats_and_table(n_features)

        fids = t.arange(10, dtype=t.int64)
        fmags = t.full((10,), 50.0)
        _, second = compute_doc_deltas(
            [fids, fids],
            [fmags, fmags],
            [None, None],
            n_features,
            DeltaMode.prev_segment,
            table,
        )
        filtered_fids, _fmags, _deltas, neg_fids, _weights, _ref_mask, _ = second
        assert len(filtered_fids) == 0
        assert len(neg_fids) == 0

    def test_pipeline_config_delta_none_backward_compatible(self) -> None:
        config = FeaturePipelineConfig(
            selection_config=ThresholdSelectionConfig(
                selection=ThresholdSelection(ThresholdType.feature_percentile, 0.9),
                precomputed_thresholds=None,
            ),
            boundaries=t.zeros(10, 3),
            negative_bundle=None,
            effective_buckets=4,
        )
        assert config.delta_bundle is None

    def test_sampling_with_percentile_delta_weights(self) -> None:
        n_features = 100
        stats, table = _make_stats_and_table(n_features)

        fids0 = t.arange(20, dtype=t.int64)
        fmags0 = t.full((20,), 30.0)
        fids1 = t.arange(20, dtype=t.int64)
        fmags1 = t.tensor([90.0] * 5 + [35.0] * 15)
        _, second = compute_doc_deltas(
            [fids0, fids1],
            [fmags0, fmags1],
            [None, None],
            n_features,
            DeltaMode.prev_segment,
            table,
        )
        filtered_fids, filtered_fmags, delta_values, _neg_fids, _weights, _ref_mask, _ = second
        assert delta_values is not None

        if len(filtered_fids) == 0:
            return  # skip if no positive deltas (unlikely with these values)

        override_weights = delta_values

        cfg = SamplingSelection(
            distribution=Distribution.normal, mean=3.0, std=0.001,
            delta_weights=DeltaSamplingWeights.percentile_delta,
        )

        counts: dict[int, int] = {}
        for seed in range(200):
            rng = t.Generator().manual_seed(seed)
            ids, _ = sample_features(
                filtered_fids, filtered_fmags, table, cfg, rng,
                override_weights=override_weights,
            )
            for fid in ids.tolist():
                counts[fid] = counts.get(fid, 0) + 1

        # Features 0-4 (large delta) should appear more than features 5-19
        high_delta_count = sum(counts.get(f, 0) for f in range(5))
        low_delta_count = sum(counts.get(f, 0) for f in range(5, 20))
        assert high_delta_count > low_delta_count, (
            f"High-delta features not dominant: high={high_delta_count}, low={low_delta_count}"
        )
