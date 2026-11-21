"""Delta-specific tests for scripts.preprocess.extract_common_features.

These tests cover the PR-specific behavior that is easy to regress:
``baseline_dir=None`` means absolute selection, any explicit ``baseline_dir``
means delta selection, sampled k is honored in both paths, and selected features
are bucketized as absolute steering magnitudes rather than delta values.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scripts.preprocess import extract_common_features as common_features
from scripts.preprocess.extract_common_features import (
    FeaturePercentileMeans,
    FeatureRankMetric,
    auto_output_path,
    build_feature_inputs_for_selected_ids,
    compute_delta_feature_scores,
    compute_feature_percentile_means,
    extract_common_features,
    resolve_selected_feature_count,
    sample_feature_count,
    select_top_feature_ids_by_score,
    validate_delta_stats_compatible,
)
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.eval.feature_config import parse_feature_configs
from steering.feature_io import save_features
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)


def _stat(*, scale: float = 1.0, count: int = 1) -> PerFeatureStat:
    percentiles = {f"p{p}": scale * (p / 100.0) for p in range(1, 100)}
    return PerFeatureStat(
        count=count,
        pct_segments_active=0.0,
        mean_mag=scale * 0.5,
        mean_mag_incl_zeros=scale * 0.25,
        std_mag=scale * 0.1,
        min_mag=percentiles["p1"],
        max_mag=percentiles["p99"],
        median_mag=percentiles["p50"],
        percentiles=percentiles,
        histogram=None,
    )


def _stats(
    per_feature_stats: dict[str, PerFeatureStat],
    *,
    n_sae_features: int | None = None,
    active_per_segment_percentiles: dict[str, float] | None = None,
    metadata: MetadataStats | None = None,
) -> ExtractionStatistics:
    if n_sae_features is None:
        n_sae_features = max((int(fid) for fid in per_feature_stats), default=-1) + 1
    if active_per_segment_percentiles is None:
        active_per_segment_percentiles = {"p50": 2.0, "p90": 3.0}
    if metadata is None:
        metadata = MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=8,
            n_sae_features=n_sae_features,
            generated_at="2026-01-01",
        )
    return ExtractionStatistics(
        metadata=metadata,
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=1,
            n_tokens_gpt2=1024,
            mean_tokens_per_doc=1024.0,
            median_tokens_per_doc=1024,
        ),
        features=SegmentFeatureStats(
            n_segments=4,
            n_features_seen=len(per_feature_stats),
            mean_active_per_segment=1.0,
            median_active_per_segment=1,
            p90_active_per_segment=2,
            max_active_per_segment=2,
            min_active_per_segment=0,
            n_total_sae_features=n_sae_features,
            pct_features_active=100.0 * len(per_feature_stats) / n_sae_features,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles=active_per_segment_percentiles,
        ),
        storage=StorageStats(
            total_bytes=1024,
            mean_bytes_per_doc=1024.0,
            mean_bytes_per_token=1.0,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
    )


def _fixed_negative_selection(k: int) -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.normal,
        mean=float(k),
        std=0.01,
        min_k=k,
        max_k=k + 1,
    )


def _metadata(n_sae_features: int) -> MetadataStats:
    return MetadataStats(
        pooling_mode="max_pool",
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=128,
        context_window=2048,
        layer=8,
        n_sae_features=n_sae_features,
        generated_at="2026-01-01",
    )


def _write_feature_shard(
    feature_dir: Path,
    segments: list[list[tuple[int, float]]],
    *,
    shard_name: str = "shard_00000_features",
) -> None:
    offsets = [0]
    fids: list[int] = []
    mags: list[float] = []
    for segment in segments:
        for fid, mag in segment:
            fids.append(int(fid))
            mags.append(float(mag))
        offsets.append(len(fids))

    n_segments = len(segments)
    seg_positions = t.arange(n_segments, dtype=t.int32)
    save_features(
        {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": t.tensor(fids, dtype=t.int32),
            "feature_mags": t.tensor(mags, dtype=t.float32),
            "doc_ids": t.zeros(n_segments, dtype=t.int32),
            "segment_indices": t.arange(n_segments, dtype=t.int32),
            "segment_positions": seg_positions,
            "segment_char_starts": seg_positions * 128,
            "segment_char_ends": (seg_positions + 1) * 128,
            "doc_token_counts": t.tensor([n_segments * 128], dtype=t.int32),
            "n_docs": 1 if n_segments > 0 else 0,
            "n_segments": n_segments,
            "layer": 8,
            "context_window": 2048,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        },
        feature_dir / shard_name,
    )


def _write_stats_dir(
    root: Path,
    rel: str,
    per_feature_stats: dict[str, PerFeatureStat],
    shard_segments: list[list[tuple[int, float]]],
    *,
    active_per_segment_percentiles: dict[str, float] | None = None,
    metadata: MetadataStats | None = None,
) -> Path:
    out_dir = root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _stats(
        per_feature_stats,
        active_per_segment_percentiles=active_per_segment_percentiles,
        metadata=metadata,
    )
    save_statistics(stats, out_dir / "feature_statistics")
    _write_feature_shard(out_dir, shard_segments)
    return out_dir


class TestFeaturePercentileMeans:
    def test_repeated_ids_across_shards_include_inactive_segments(
        self, tmp_path: Path
    ) -> None:
        _write_feature_shard(
            tmp_path, [[(2, 0.25), (0, 0.5), (2, 0.25)], [(1, 0.75)]]
        )
        _write_feature_shard(
            tmp_path, [[(0, 0.5)], [(3, 0.75)]], shard_name="shard_00001_features"
        )
        table = build_percentile_table({str(i): _stat() for i in range(4)}, 4)

        result = compute_feature_percentile_means(
            tmp_path, table, 4, chunk_size=2, reference_feature_ids={0, 1, 2, 3}
        )

        assert result.n_segments == 4
        assert result.active_counts.tolist() == [2, 1, 2, 1]
        t.testing.assert_close(
            result.percentile_sums,
            t.tensor([1.0, 0.75, 0.5, 0.75], dtype=t.float64),
            rtol=0,
            atol=0,
        )

    def test_empty_shard_preserves_inactive_segments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The standalone scanner accepts empty loaded data; base writers reject
        # empty segments, so exercise this helper contract at the loader boundary.
        monkeypatch.setattr(common_features, "glob_features", lambda _: [tmp_path])
        monkeypatch.setattr(
            common_features,
            "load_features",
            lambda _: {
                "feature_ids": t.empty(0, dtype=t.int64),
                "feature_mags": t.empty(0),
                "n_segments": 2,
            },
        )
        table = build_percentile_table({"0": _stat()}, 4)

        result = compute_feature_percentile_means(tmp_path, table, 4)

        assert result.n_segments == 2
        assert result.active_counts.tolist() == [0, 0, 0, 0]
        assert result.percentile_sums.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_reference_need_only_cover_active_ids(self, tmp_path: Path) -> None:
        _write_feature_shard(tmp_path, [[(2, 0.5), (0, 0.5), (2, 0.5)]])
        table = build_percentile_table({"0": _stat(), "2": _stat()}, 4)

        result = compute_feature_percentile_means(
            tmp_path, table, 4, reference_feature_ids={0, 2}
        )

        assert result.active_counts.tolist() == [1, 0, 2, 0]
        assert result.percentile_sums.tolist() == [0.5, 0.0, 1.0, 0.0]

    def test_missing_reference_ids_report_sorted_unique_ids(self, tmp_path: Path) -> None:
        _write_feature_shard(tmp_path, [[(3, 0.5), (1, 0.5), (0, 0.5), (3, 0.5)]])
        table = build_percentile_table({str(i): _stat() for i in range(4)}, 4)

        with pytest.raises(ValueError, match=r"feature IDs: \[0, 1, 3\]"):
            compute_feature_percentile_means(
                tmp_path, table, 4, reference_feature_ids=set()
            )

    def test_noncontiguous_native_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ids = t.tensor([2, 9, 0, 9, 2, 9, 1, 9, 0, 9])[::2]
        mags = t.tensor([0.25, 9, 0.5, 9, 0.5, 9, 0.75, 9, 0.25, 9])[::2]
        before_ids, before_mags = ids.clone(), mags.clone()
        monkeypatch.setattr(common_features, "glob_features", lambda _: [tmp_path])
        monkeypatch.setattr(
            common_features,
            "load_features",
            lambda _: {"feature_ids": ids, "feature_mags": mags, "n_segments": 3},
        )
        table = build_percentile_table({str(i): _stat() for i in range(4)}, 4)

        result = compute_feature_percentile_means(tmp_path, table, 4, chunk_size=2)

        assert result.active_counts.tolist() == [2, 1, 2, 0]
        assert result.percentile_sums.tolist() == [0.75, 0.75, 0.75, 0.0]
        t.testing.assert_close(ids, before_ids)
        t.testing.assert_close(mags, before_mags)
        assert ids.stride() == mags.stride() == (2,)

    @pytest.mark.parametrize("bad_id", [-1, 4])
    def test_out_of_range_ids_keep_validation_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_id: int
    ) -> None:
        monkeypatch.setattr(common_features, "glob_features", lambda _: [tmp_path])
        monkeypatch.setattr(
            common_features,
            "load_features",
            lambda _: {
                "feature_ids": t.tensor([bad_id]),
                "feature_mags": t.tensor([0.5]),
                "n_segments": 1,
            },
        )
        with pytest.raises(ValueError, match="exceed n_sae_features=4"):
            compute_feature_percentile_means(tmp_path, t.ones(4, 99), 4)

    def test_malformed_rank_keeps_lookup_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(common_features, "glob_features", lambda _: [tmp_path])
        monkeypatch.setattr(
            common_features,
            "load_features",
            lambda _: {
                "feature_ids": t.tensor([[0, 1], [2, 0]]),
                "feature_mags": t.tensor([[0.2, 0.3], [0.4, 0.5]]),
                "n_segments": 2,
            },
        )
        table = build_percentile_table({str(i): _stat() for i in range(4)}, 4)
        with pytest.raises(IndexError, match="too many indices for tensor of dimension 1"):
            compute_feature_percentile_means(tmp_path, table, 4)


class TestDeltaScores:
    def test_scores_are_source_minus_baseline_percentile_means(self) -> None:
        source = FeaturePercentileMeans(
            percentile_sums=t.tensor([1.6, 0.1, 0.8], dtype=t.float64),
            active_counts=t.tensor([2, 1, 1]),
            n_segments=4,
        )
        baseline = FeaturePercentileMeans(
            percentile_sums=t.tensor([0.4, 0.2, 0.0], dtype=t.float64),
            active_counts=t.tensor([1, 1, 0]),
            n_segments=4,
        )
        per_feature = {"0": _stat(), "1": _stat(), "2": _stat()}

        scores = compute_delta_feature_scores(source, baseline, per_feature)

        assert scores == pytest.approx({0: 0.30, 2: 0.20})
        assert 1 not in scores

    def test_negative_pool_scores_use_baseline_minus_source_candidates(self) -> None:
        source = FeaturePercentileMeans(
            t.tensor([0.2, 0.0, 0.1], dtype=t.float64),
            t.ones(3, dtype=t.int64),
            1,
        )
        baseline = FeaturePercentileMeans(
            t.tensor([0.1, 0.7, 0.4], dtype=t.float64),
            t.ones(3, dtype=t.int64),
            1,
        )

        scores = compute_delta_feature_scores(
            baseline,
            source,
            {"0": _stat(), "1": _stat(), "2": _stat()},
        )

        assert scores == pytest.approx({1: 0.7, 2: 0.3})

    def test_shape_mismatch_raises(self) -> None:
        source = FeaturePercentileMeans(t.zeros(2), t.ones(2), 1)
        baseline = FeaturePercentileMeans(t.zeros(3), t.ones(3), 1)

        with pytest.raises(ValueError, match="different score shapes"):
            compute_delta_feature_scores(source, baseline, {"0": _stat()})

    def test_top_score_selection_is_deterministic(self) -> None:
        assert select_top_feature_ids_by_score({3: 0.7, 1: 0.7, 2: 0.9}, 3) == [
            2,
            1,
            3,
        ]


class TestDeltaCompatibilityValidation:
    def test_matching_metadata_passes(self) -> None:
        a = _stats({"0": _stat()})
        b = _stats({"0": _stat()})
        validate_delta_stats_compatible(a, b)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("n_sae_features", 4, "n_sae_features"),
            ("layer", 9, "layer"),
            ("pooling_mode", "mean_pool", "pooling_mode"),
            ("segment_mode", "other_mode", "segment_mode"),
            ("segment_size", 64, "segment_size"),
            ("context_window", 1024, "context_window"),
        ],
    )
    def test_metadata_mismatch_raises(self, field: str, value, message: str) -> None:
        steering = _stats({"0": _stat()})
        baseline_meta = replace(steering.metadata, **{field: value})
        baseline = _stats(
            {"0": _stat()},
            n_sae_features=baseline_meta.n_sae_features,
            metadata=baseline_meta,
        )

        with pytest.raises(ValueError, match=message):
            validate_delta_stats_compatible(steering, baseline)


class TestDeltaExtraction:
    def test_negative_selection_requires_baseline_dir(self, tmp_path: Path) -> None:
        stats = {"0": _stat()}
        sft_dir = _write_stats_dir(tmp_path, "sft", stats, [[(0, 0.90)]])
        base_dir = _write_stats_dir(tmp_path, "base", stats, [[(0, 0.10)]])

        with pytest.raises(ValueError, match="negative_selection requires baseline_dir"):
            extract_common_features(
                str(sft_dir),
                4,
                stats_dir=str(base_dir),
                baseline_dir=None,
                top_k=1,
                negative_selection=_fixed_negative_selection(1),
            )

    def test_negative_selection_requires_num_buckets_at_least_two(
        self,
        tmp_path: Path,
    ) -> None:
        stats = {"0": _stat()}
        sft_dir = _write_stats_dir(tmp_path, "sft", stats, [[(0, 0.90)]])
        base_dir = _write_stats_dir(tmp_path, "base", stats, [[(0, 0.10)]])

        with pytest.raises(ValueError, match="num_buckets >= 2"):
            extract_common_features(
                str(sft_dir),
                1,
                stats_dir=str(base_dir),
                baseline_dir=str(base_dir),
                top_k=1,
                negative_selection=_fixed_negative_selection(1),
            )

    def test_baseline_dir_none_implies_absolute_selection(self, tmp_path: Path) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(3)}
        sft_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/safety_sft",
            stats,
            [[(0, 0.90)], [(1, 0.20)], [(2, 0.80)]],
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/base",
            stats,
            [[(0, 0.99)], [(1, 0.99)], [(2, 0.01)]],
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=None,
            top_k=2,
        )

        assert list(result) == [0, 2]

    def test_explicit_baseline_dir_enables_delta_selection(self, tmp_path: Path) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(3)}
        sft_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/safety_sft",
            stats,
            [[(0, 0.90)], [(1, 0.80)], [(2, 0.20)]],
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/base",
            stats,
            [[(0, 0.89)], [(1, 0.10)], [(2, 0.10)]],
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=2,
        )

        assert list(result) == [1, 2]

    def test_sampled_k_controls_delta_feature_count(self, tmp_path: Path) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(4)}
        sft_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/safety_sft",
            stats,
            [[(0, 0.90)], [(1, 0.80)], [(2, 0.70)], [(3, 0.10)]],
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/base",
            stats,
            [[(0, 0.10)], [(1, 0.10)], [(2, 0.10)], [(3, 0.10)]],
        )
        selection = SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=4,
        )
        expected_k = sample_feature_count(selection, seed=7)

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            feature_selection=selection,
            seed=7,
        )

        assert len(result) == expected_k
        assert list(result) == [0, 1, 2][:expected_k]

    def test_sampled_k_delta_selection_returns_exact_top_delta_ids(
        self,
        tmp_path: Path,
    ) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(4)}
        sft_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/safety_sft",
            stats,
            [[(0, 0.91)], [(1, 0.82)], [(2, 0.73)], [(3, 0.64)]],
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/base",
            stats,
            [[(0, 0.80)], [(1, 0.10)], [(2, 0.50)], [(3, 0.10)]],
        )
        selection = SamplingSelection(
            distribution=Distribution.normal,
            mean=2.0,
            std=0.01,
            min_k=1,
            max_k=4,
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            feature_selection=selection,
            seed=0,
        )

        assert sample_feature_count(selection, seed=0) == 2
        assert list(result) == [1, 3]

    def test_delta_rank_metric_controls_sparse_feature_semantics(
        self,
        tmp_path: Path,
    ) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(2)}
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            stats,
            [[(0, 0.60), (1, 0.95)], [(0, 0.60)], [(0, 0.60)], [(0, 0.60)]],
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            stats,
            [[(0, 0.10), (1, 0.10)], [(0, 0.10)], [(0, 0.10)], [(0, 0.10)]],
        )

        incl_zeros = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=1,
            rank_metric=FeatureRankMetric.MEAN_PCT_INCL_ZEROS,
        )
        active_only = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=1,
            rank_metric=FeatureRankMetric.MEAN_PCT,
        )

        assert list(incl_zeros) == [0]
        assert list(active_only) == [1]

    @pytest.mark.parametrize("baseline_dir_is_set", [False, True])
    def test_top_k_zero_returns_empty_mapping(
        self,
        tmp_path: Path,
        baseline_dir_is_set: bool,
    ) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(2)}
        sft_dir = _write_stats_dir(tmp_path, "sft", stats, [[(0, 0.90)], [(1, 0.80)]])
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            stats,
            [[(0, 0.10)], [(1, 0.10)]],
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir) if baseline_dir_is_set else None,
            top_k=0,
        )

        assert result == {}

    def test_stats_dir_and_baseline_dir_are_independent(self, tmp_path: Path) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(2)}
        wide_stats = {str(i): _stat(scale=10.0) for i in range(2)}
        sft_dir = _write_stats_dir(tmp_path, "sft", stats, [[(0, 0.90)], [(1, 0.80)]])
        baseline_dir = _write_stats_dir(
            tmp_path,
            "baseline",
            stats,
            [[(0, 0.10)], [(1, 0.79)]],
        )
        narrow_dir = _write_stats_dir(tmp_path, "narrow", stats, [[(0, 0.1)]])
        wide_dir = _write_stats_dir(tmp_path, "wide", wide_stats, [[(0, 1.0)]])

        narrow_result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(narrow_dir),
            baseline_dir=str(baseline_dir),
            top_k=1,
            magnitude_percentile=0.5,
        )
        wide_result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(wide_dir),
            baseline_dir=str(baseline_dir),
            top_k=1,
            magnitude_percentile=0.5,
        )

        assert list(narrow_result) == [0]
        assert list(wide_result) == [0]
        assert wide_result[0] < narrow_result[0]

    def test_no_positive_deltas_returns_empty_mapping(self, tmp_path: Path) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(2)}
        sft_dir = _write_stats_dir(tmp_path, "sft", stats, [[(0, 0.10)], [(1, 0.10)]])
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            stats,
            [[(0, 0.90)], [(1, 0.90)]],
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=5,
        )

        assert result == {}

    def test_missing_baseline_stats_raises_only_when_baseline_is_set(
        self,
        tmp_path: Path,
    ) -> None:
        stats = {"0": _stat()}
        sft_dir = _write_stats_dir(tmp_path, "sft", stats, [[(0, 0.5)]])
        boundary_dir = _write_stats_dir(tmp_path, "base", stats, [[(0, 0.5)]])
        missing = tmp_path / "missing"
        missing.mkdir()

        with pytest.raises(FileNotFoundError, match="baseline_dir"):
            extract_common_features(
                str(sft_dir),
                4,
                stats_dir=str(boundary_dir),
                baseline_dir=str(missing),
                top_k=1,
            )

        absolute = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(boundary_dir),
            baseline_dir=None,
            top_k=1,
        )
        assert absolute

    def test_baseline_active_feature_missing_from_reference_stats_raises(
        self,
        tmp_path: Path,
    ) -> None:
        metadata = MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=8,
            n_sae_features=2,
            generated_at="2026-01-01",
        )
        source_stats = {"0": _stat()}
        reference_stats = {"0": _stat()}
        baseline_stats = {"0": _stat()}
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.90)]],
            metadata=metadata,
        )
        boundary_dir = _write_stats_dir(
            tmp_path,
            "boundary",
            reference_stats,
            [[(0, 0.10)]],
            metadata=metadata,
        )
        baseline_dir = _write_stats_dir(
            tmp_path,
            "baseline",
            baseline_stats,
            [[(1, 0.90)]],
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="active shard"):
            extract_common_features(
                str(sft_dir),
                4,
                stats_dir=str(boundary_dir),
                baseline_dir=str(baseline_dir),
                top_k=1,
            )

    def test_baseline_candidate_missing_from_reference_stats_raises_when_negatives_enabled(
        self,
        tmp_path: Path,
    ) -> None:
        metadata = MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=8,
            n_sae_features=2,
            generated_at="2026-01-01",
        )
        source_stats = {"0": _stat()}
        reference_stats = {"0": _stat()}
        baseline_stats = {"0": _stat(), "1": _stat()}
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.90)]],
            metadata=metadata,
        )
        boundary_dir = _write_stats_dir(
            tmp_path,
            "boundary",
            reference_stats,
            [[(0, 0.10)]],
            metadata=metadata,
        )
        baseline_dir = _write_stats_dir(
            tmp_path,
            "baseline",
            baseline_stats,
            [[(0, 0.10)]],
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="baseline candidate"):
            extract_common_features(
                str(sft_dir),
                4,
                stats_dir=str(boundary_dir),
                baseline_dir=str(baseline_dir),
                top_k=1,
                negative_selection=_fixed_negative_selection(1),
            )

    def test_selected_ids_bucketize_absolute_steering_magnitudes(self) -> None:
        stats = {"0": _stat(scale=1.0), "1": _stat(scale=1.0)}
        boundaries = build_boundaries_from_stats(stats, n_sae_features=2, num_buckets=4)

        _fids, mags, mask = build_feature_inputs_for_selected_ids(
            stats,
            boundaries,
            [0, 1],
            "cpu",
            magnitude_percentile="p50",
        )

        assert mask.tolist() == [[[True, True]]]
        assert mags.squeeze().tolist() == [2, 2]

    def test_mixed_delta_extraction_merges_positive_then_negative(
        self,
        tmp_path: Path,
    ) -> None:
        all_stats = {str(i): _stat(scale=1.0) for i in range(4)}
        source_stats = {key: all_stats[key] for key in ("0", "2")}
        baseline_stats = {key: all_stats[key] for key in ("0", "1", "2", "3")}
        metadata = _metadata(4)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.90)], [(2, 0.80)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            baseline_stats,
            [[(0, 0.10)], [(1, 0.90)], [(2, 0.10)], [(3, 0.80)]],
            metadata=metadata,
        )

        result = extract_common_features(
            str(sft_dir),
            5,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=2,
            negative_selection=_fixed_negative_selection(2),
        )

        assert list(result.items()) == [(0, 2), (2, 2), (1, 4), (3, 4)]

    def test_mixed_delta_binary_mode_uses_active_zero_and_inactive_one(
        self,
        tmp_path: Path,
    ) -> None:
        all_stats = {str(i): _stat(scale=1.0) for i in range(2)}
        source_stats = {"0": all_stats["0"]}
        baseline_stats = {"0": all_stats["0"], "1": all_stats["1"]}
        metadata = _metadata(2)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.99)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            baseline_stats,
            [[(0, 0.10)], [(1, 0.99)]],
            metadata=metadata,
        )

        result = extract_common_features(
            str(sft_dir),
            2,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=1,
            magnitude_percentile=0.99,
            negative_selection=_fixed_negative_selection(1),
        )

        assert result == {0: 0, 1: 1}

    def test_positive_buckets_use_effective_bucket_count_when_negatives_enabled(
        self,
        tmp_path: Path,
    ) -> None:
        all_stats = {str(i): _stat(scale=1.0) for i in range(2)}
        source_stats = {"0": all_stats["0"]}
        baseline_stats = {"0": all_stats["0"], "1": all_stats["1"]}
        metadata = _metadata(2)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.99)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            baseline_stats,
            [[(0, 0.10)], [(1, 0.99)]],
            metadata=metadata,
        )

        result = extract_common_features(
            str(sft_dir),
            5,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=1,
            magnitude_percentile=0.99,
            negative_selection=_fixed_negative_selection(1),
        )

        assert result[0] == 3
        assert result[0] != 4
        assert result[1] == 4

    def test_negative_sampling_uses_seed_plus_one_independently_of_positive_sampling(
        self,
        tmp_path: Path,
    ) -> None:
        all_stats = {str(i): _stat(scale=1.0) for i in range(6)}
        source_stats = {key: all_stats[key] for key in ("0", "2", "4")}
        metadata = _metadata(6)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.90)], [(2, 0.80)], [(4, 0.70)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            all_stats,
            [
                [(0, 0.10)],
                [(1, 0.95)],
                [(2, 0.10)],
                [(3, 0.85)],
                [(4, 0.10)],
                [(5, 0.75)],
            ],
            metadata=metadata,
        )
        negative_selection = NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=0,
            max_k=4,
        )
        expected_negative_k = sample_feature_count(negative_selection, seed=12)

        result_a = extract_common_features(
            str(sft_dir),
            5,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=1,
            seed=11,
            negative_selection=negative_selection,
        )
        result_b = extract_common_features(
            str(sft_dir),
            5,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=3,
            seed=11,
            negative_selection=negative_selection,
        )

        neg_bucket = 4
        assert sum(bucket == neg_bucket for bucket in result_a.values()) == expected_negative_k
        assert sum(bucket == neg_bucket for bucket in result_b.values()) == expected_negative_k

    def test_empty_positive_pool_can_still_emit_negative_features(
        self,
        tmp_path: Path,
    ) -> None:
        all_stats = {str(i): _stat(scale=1.0) for i in range(2)}
        source_stats = {"0": all_stats["0"]}
        baseline_stats = {"0": all_stats["0"], "1": all_stats["1"]}
        metadata = _metadata(2)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.10)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            baseline_stats,
            [[(0, 0.90)], [(1, 0.90)]],
            metadata=metadata,
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=3,
            negative_selection=_fixed_negative_selection(1),
        )

        assert result == {1: 3}

    def test_nonempty_positive_pool_can_have_empty_negative_pool(
        self,
        tmp_path: Path,
    ) -> None:
        all_stats = {str(i): _stat(scale=1.0) for i in range(2)}
        source_stats = {"0": all_stats["0"]}
        baseline_stats = {"0": all_stats["0"], "1": all_stats["1"]}
        metadata = _metadata(2)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.90)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            baseline_stats,
            [[(0, 0.10)], [(1, 0.10)]],
            metadata=metadata,
        )

        result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=1,
            negative_selection=_fixed_negative_selection(0),
        )

        assert result == {0: 1}

    def test_mixed_stats_dir_and_baseline_dir_are_independent(
        self,
        tmp_path: Path,
    ) -> None:
        stats = {str(i): _stat(scale=1.0) for i in range(2)}
        wide_stats = {str(i): _stat(scale=10.0) for i in range(2)}
        metadata = _metadata(2)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            {"0": stats["0"]},
            [[(0, 0.90)]],
            metadata=metadata,
        )
        baseline_dir = _write_stats_dir(
            tmp_path,
            "baseline",
            stats,
            [[(0, 0.10)], [(1, 0.90)]],
            metadata=metadata,
        )
        narrow_dir = _write_stats_dir(
            tmp_path,
            "narrow",
            stats,
            [[(0, 0.10)]],
            metadata=metadata,
        )
        wide_dir = _write_stats_dir(
            tmp_path,
            "wide",
            wide_stats,
            [[(0, 1.0)]],
            metadata=metadata,
        )

        narrow_result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(narrow_dir),
            baseline_dir=str(baseline_dir),
            top_k=1,
            negative_selection=_fixed_negative_selection(1),
        )
        wide_result = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(wide_dir),
            baseline_dir=str(baseline_dir),
            top_k=1,
            negative_selection=_fixed_negative_selection(1),
        )

        assert list(narrow_result) == [0, 1]
        assert list(wide_result) == [0, 1]
        assert narrow_result[1] == 3
        assert wide_result[1] == 3
        assert wide_result[0] < narrow_result[0]


class TestDeltaOutputPaths:
    def test_shipped_config_uses_only_supported_selector_schema(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs" / "eval"
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="extract_common_features")
        GlobalHydra.instance().clear()

        assert "feature_percentile" not in cfg
        assert "rank_by" not in cfg
        assert cfg.feature_selection is not None
        assert cfg.rank_metric == "mean_pct_incl_zeros"

    def test_absolute_and_delta_paths_are_distinct(self) -> None:
        absolute = auto_output_path("/features/safety_sft", 0.5, top_k=8)
        delta = auto_output_path(
            "/features/safety_sft",
            0.5,
            top_k=8,
            baseline_dir="/features/base",
        )

        assert absolute == (
            "out/data/common_features/safety_sft/"
            "absolute/k8/seed_42/r_mean_pct_incl_zeros/m0.5/features.yaml"
        )
        assert delta == (
            "out/data/common_features/safety_sft/"
            "delta_base/k8/seed_42/r_mean_pct_incl_zeros/m0.5/features.yaml"
        )

    def test_feature_count_requires_one_supported_selector(self) -> None:
        with pytest.raises(ValueError, match="feature_selection or top_k"):
            resolve_selected_feature_count()

    def test_output_path_requires_one_supported_selector(self) -> None:
        with pytest.raises(ValueError, match="feature_selection or top_k"):
            auto_output_path("/features/safety_sft", 0.5)

    def test_output_path_rejects_negative_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 0"):
            auto_output_path("/features/safety_sft", 0.5, top_k=-1)

    @pytest.mark.parametrize(
        ("removed_key", "value", "replacement"),
        [
            ("feature_percentile", 0.5, "feature_selection"),
            ("rank_by", "mean_pct", "rank_metric"),
        ],
    )
    def test_hydra_entrypoint_rejects_removed_config_keys(
        self,
        removed_key: str,
        value: object,
        replacement: str,
    ) -> None:
        from scripts.preprocess.extract_common_features import main as extract_main

        cfg = OmegaConf.create({removed_key: value})
        with pytest.raises(ValueError, match=f"{removed_key}.*{replacement}"):
            extract_main.__wrapped__(cfg)

    def test_sampled_delta_path_records_sampling_config_and_baseline_name(self) -> None:
        selection = SamplingSelection(
            distribution=Distribution.normal,
            mean=2.0,
            std=0.01,
            min_k=0,
            max_k=5,
        )

        path = auto_output_path(
            "/features/safety_sft",
            0.9,
            feature_selection=selection,
            seed=123,
            rank_metric=FeatureRankMetric.MEAN_PCT,
            baseline_dir="/features/base",
        )

        assert path == (
            "out/data/common_features/safety_sft/"
            "delta_base/dist_normal/mean_2.0/std_0.01/min_0/max_5/"
            "seed_123/r_mean_pct/m0.9/features.yaml"
        )

    def test_mixed_delta_path_records_negative_selection(self) -> None:
        selection = SamplingSelection(
            distribution=Distribution.normal,
            mean=2.0,
            std=0.01,
            min_k=0,
            max_k=5,
        )
        negative_selection = NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=1,
            max_k=8,
        )

        path = auto_output_path(
            "/features/safety_sft",
            0.9,
            feature_selection=selection,
            negative_selection=negative_selection,
            seed=123,
            rank_metric=FeatureRankMetric.MEAN_PCT,
            baseline_dir="/features/base",
        )

        assert path == (
            "out/data/common_features/safety_sft/"
            "delta_base/dist_normal/mean_2.0/std_0.01/min_0/max_5/"
            "neg_dist_geometric/mean_3.0/std_3.0/min_1/max_8/"
            "seed_123/r_mean_pct/m0.9/features.yaml"
        )

    def test_auto_output_path_rejects_negative_selection_without_baseline_dir(self) -> None:
        with pytest.raises(ValueError, match="negative_selection requires baseline_dir"):
            auto_output_path(
                "/features/safety_sft",
                0.9,
                top_k=2,
                negative_selection=_fixed_negative_selection(1),
            )

    def test_delta_yaml_roundtrip_through_eval_config_parser(
        self,
        tmp_path: Path,
    ) -> None:
        stats = {"0": _stat(scale=1.0), "1": _stat(scale=1.0)}
        sft_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/safety_sft",
            stats,
            [[(0, 0.90)], [(1, 0.80)]],
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "m_gemma/base",
            stats,
            [[(0, 0.10)], [(1, 0.10)]],
        )

        features = extract_common_features(
            str(sft_dir),
            4,
            stats_dir=str(base_dir),
            baseline_dir=str(base_dir),
            top_k=2,
        )
        output_dir = (
            tmp_path
            / "common_features"
            / "safety_sft"
            / "delta_base"
            / "k2"
            / "seed_42"
            / "r_mean_pct_incl_zeros"
            / "m0.5"
        )
        output_dir.mkdir(parents=True)
        with open(output_dir / "features.yaml", "w", encoding="utf-8") as f:
            yaml.dump(features, f, default_flow_style=False, sort_keys=False)

        configs = parse_feature_configs(str(tmp_path / "common_features" / "safety_sft"))

        assert len(configs) == 1
        assert configs[0].name == "safety_sft/delta_base/k2/seed_42/r_mean_pct_incl_zeros/m0.5"
        assert configs[0].features == features

    def test_mixed_delta_yaml_roundtrip_through_eval_config_parser(
        self,
        tmp_path: Path,
    ) -> None:
        mixed = {0: 2, 2: 2, 1: 4, 3: 4}
        output_dir = (
            tmp_path
            / "common_features"
            / "safety_sft"
            / "delta_base"
            / "k2"
            / "neg_dist_normal"
            / "mean_2.0"
            / "std_0.01"
            / "min_2"
            / "max_3"
            / "seed_42"
            / "r_mean_pct_incl_zeros"
            / "m0.5"
        )
        output_dir.mkdir(parents=True)
        with open(output_dir / "features.yaml", "w", encoding="utf-8") as f:
            yaml.dump(mixed, f, default_flow_style=False, sort_keys=False)

        configs = parse_feature_configs(str(tmp_path / "common_features" / "safety_sft"))

        assert len(configs) == 1
        assert configs[0].features == mixed

    def test_hydra_entrypoint_accepts_dictconfig_negative_selection(
        self,
        tmp_path: Path,
    ) -> None:
        from scripts.preprocess.extract_common_features import main as extract_main

        all_stats = {str(i): _stat(scale=1.0) for i in range(2)}
        source_stats = {"0": all_stats["0"]}
        metadata = _metadata(2)
        sft_dir = _write_stats_dir(
            tmp_path,
            "sft",
            source_stats,
            [[(0, 0.99)]],
            metadata=metadata,
        )
        base_dir = _write_stats_dir(
            tmp_path,
            "base",
            all_stats,
            [[(0, 0.10)], [(1, 0.99)]],
            metadata=metadata,
        )
        output = tmp_path / "features.yaml"
        cfg = OmegaConf.create(
            {
                "feature_dir": str(sft_dir),
                "stats_dir": str(base_dir),
                "baseline_dir": str(base_dir),
                "num_buckets": 4,
                "feature_selection": None,
                "negative_selection": {
                    "distribution": "normal",
                    "mean": 1.0,
                    "std": 0.01,
                    "min_k": 1,
                    "max_k": 2,
                },
                "magnitude_percentile": 0.99,
                "rank_metric": "mean_pct_incl_zeros",
                "seed": 0,
                "top_k": 1,
                "output": str(output),
            }
        )

        extract_main.__wrapped__(cfg)

        assert yaml.safe_load(output.read_text()) == {0: 2, 1: 3}


class TestStatisticsSnapshots:
    @pytest.mark.parametrize("boundary_kind", ["same", "distinct", "alias"])
    def test_reuses_only_equal_discovered_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary_kind: str
    ) -> None:
        stats = {"0": _stat(), "1": _stat(scale=2.0)}
        source = _write_stats_dir(
            tmp_path, "source", stats, [[(0, 0.25), (1, 0.75)], [(0, 0.8)]]
        )
        boundary = source
        if boundary_kind == "distinct":
            boundary = _write_stats_dir(tmp_path, "boundary", stats, [[(0, 0.25)]])
        elif boundary_kind == "alias":
            boundary = tmp_path / "alias"
            boundary.symlink_to(source, target_is_directory=True)
        native_load = common_features.load_statistics
        loaded = []

        def load(path):
            value = native_load(path)
            loaded.append(value)
            return value

        monkeypatch.setattr(common_features, "load_statistics", load)
        result = extract_common_features(
            str(source), 10, stats_dir=str(boundary), top_k=2
        )
        assert result == {0: 5, 1: 5}
        assert len(loaded) == (1 if boundary_kind == "same" else 2)

    def test_equal_delta_baseline_is_loaded_separately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _write_stats_dir(
            tmp_path, "source", {"0": _stat()}, [[(0, 0.25)]]
        )
        native_load = common_features.load_statistics
        loaded = []

        def load(path):
            value = native_load(path)
            loaded.append(value)
            return value

        monkeypatch.setattr(common_features, "load_statistics", load)
        assert extract_common_features(
            str(source), 10, stats_dir=str(source), baseline_dir=str(source), top_k=1
        ) == {}
        assert len(loaded) == 2
        assert loaded[0] is not loaded[1]

    def test_later_invocation_reads_updated_statistics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _write_stats_dir(
            tmp_path, "source", {"0": _stat()}, [[(0, 0.25)]]
        )
        native_load = common_features.load_statistics
        medians = []

        def load(path):
            value = native_load(path)
            medians.append(value.per_feature_stats["0"].median_mag)
            return value

        monkeypatch.setattr(common_features, "load_statistics", load)
        assert extract_common_features(
            str(source), 10, stats_dir=str(source), top_k=1
        ) == {0: 5}
        save_statistics(_stats({"0": _stat(scale=3.0)}), source / "feature_statistics")
        assert extract_common_features(
            str(source), 10, stats_dir=str(source), top_k=1
        ) == {0: 5}
        assert medians == [0.5, 1.5]
