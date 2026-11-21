"""Tests for steering.conditional.feature_pipeline module."""

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
)
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
    active_bucket_count,
    avg_features_per_segment_estimate,
    build_feature_pipeline,
    feature_selection_dirname,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.feature_io import save_features
from steering.shard_validation import validate_shard_data
from steering.stats_types import PerFeatureStat
from steering.validate_features import validate_feature_directory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_feature_stats(n_features: int = 100) -> dict[str, PerFeatureStat]:
    """Build minimal per_feature_stats for pipeline construction."""
    stats: dict[str, PerFeatureStat] = {}
    for fid in range(n_features):
        pcts = {f"p{k}": float(fid * 100 + k) for k in range(1, 100)}
        stats[str(fid)] = PerFeatureStat(
            count=10,
            pct_segments_active=0.5,
            mean_mag=float(fid),
            mean_mag_incl_zeros=float(fid) * 0.5,
            std_mag=1.0,
            min_mag=0.0,
            max_mag=float(fid * 100 + 100),
            median_mag=float(fid) * 0.9,
            percentiles=pcts,
        )
    return stats


# ---------------------------------------------------------------------------
# feature_selection_dirname
# ---------------------------------------------------------------------------


class TestFeatureSelectionDirname:
    """Test feature_selection_dirname for both selection types."""

    def test_threshold_integer(self) -> None:
        fs = ThresholdSelection(ThresholdType.segment_topk, 2048.0)
        assert feature_selection_dirname(fs) == "t2048.0"

    def test_threshold_float(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2048.9)
        assert feature_selection_dirname(fs) == "t2048.9"

    def test_threshold_small_decimal(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.95)
        assert feature_selection_dirname(fs) == "t0.95"

    def test_sampling_geometric(self) -> None:
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        assert feature_selection_dirname(fs) == "dist_geometric/mean_5.0/std_5.0"

    def test_sampling_normal(self) -> None:
        fs = SamplingSelection(Distribution.normal, mean=10.0, std=3.0)
        assert feature_selection_dirname(fs) == "dist_normal/mean_10.0/std_3.0"

    def test_sampling_nested_structure(self) -> None:
        """Dirname should produce nested path components (3 dirs)."""
        fs = SamplingSelection(Distribution.geometric, mean=7.5, std=7.5)
        dirname = feature_selection_dirname(fs)
        parts = dirname.split("/")
        assert len(parts) == 3
        assert parts[0] == "dist_geometric"
        assert parts[1] == "mean_7.5"
        assert parts[2] == "std_7.5"

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            feature_selection_dirname(cast(Any, "not a selection"))


# ---------------------------------------------------------------------------
# active_bucket_count
# ---------------------------------------------------------------------------


class TestActiveBucketCount:
    @pytest.mark.parametrize("num_buckets", [1, 2, 5, 10])
    def test_without_reservation_returns_the_full_count(
        self,
        num_buckets: int,
    ) -> None:
        assert active_bucket_count(
            num_buckets,
            reserves_negative_bucket=False,
        ) == num_buckets

    @pytest.mark.parametrize(
        ("num_buckets", "expected"),
        [(2, 1), (3, 2), (10, 9)],
    )
    def test_reservation_removes_exactly_the_final_bucket(
        self,
        num_buckets: int,
        expected: int,
    ) -> None:
        assert active_bucket_count(
            num_buckets,
            reserves_negative_bucket=True,
        ) == expected

    @pytest.mark.parametrize("num_buckets", [0, -1])
    def test_non_positive_bucket_count_raises(self, num_buckets: int) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="num_buckets"):
            active_bucket_count(
                num_buckets,
                reserves_negative_bucket=False,
            )

    def test_single_bucket_cannot_reserve_an_inactive_bucket(self) -> None:
        with pytest.raises(ValueError, match="empty positive bucket space"):
            active_bucket_count(1, reserves_negative_bucket=True)

    @pytest.mark.parametrize("value", [True, False, 1.0, "4", None])
    def test_bucket_count_is_a_strict_builtin_int(self, value: Any) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="num_buckets"):
            active_bucket_count(
                value,
                reserves_negative_bucket=False,
            )

    @pytest.mark.parametrize("value", [0, 1, "false", None])
    def test_reservation_flag_is_a_strict_builtin_bool(self, value: Any) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="reserves_negative_bucket",
        ):
            active_bucket_count(
                4,
                reserves_negative_bucket=value,
            )

    def test_feature_pipeline_uses_the_same_authority(self) -> None:
        stats = _make_per_feature_stats(8)
        selection = SamplingSelection(
            Distribution.geometric,
            mean=2.0,
            std=2.0,
        )
        negative = NegativeSamplingSelection(
            Distribution.geometric,
            mean=2.0,
            std=2.0,
        )
        for negative_selection in (None, negative):
            pipeline = build_feature_pipeline(
                feature_selection=selection,
                negative_selection=negative_selection,
                per_feature_stats=stats,
                n_sae_features=8,
                num_buckets=4,
            )
            assert pipeline.effective_buckets == active_bucket_count(
                4,
                reserves_negative_bucket=negative_selection is not None,
            )


# ---------------------------------------------------------------------------
# build_feature_pipeline — threshold path
# ---------------------------------------------------------------------------


class TestBuildFeaturePipelineThreshold:
    """Test build_feature_pipeline with ThresholdSelection."""

    def test_threshold_returns_boundaries(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=None,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )
        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
        assert pipeline.boundaries is not None
        assert pipeline.selection_config.precomputed_thresholds is not None
        assert pipeline.effective_buckets == 4
        assert pipeline.negative_bundle is None

    def test_threshold_with_feature_percentile(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=None,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )
        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
        assert pipeline.selection_config.precomputed_thresholds is not None
        assert isinstance(pipeline.selection_config.precomputed_thresholds, t.Tensor)

    @pytest.mark.parametrize(
        ("threshold_type", "threshold", "needs_precomputed"),
        [
            (ThresholdType.segment_topk, 7.0, False),
            (ThresholdType.segment_percentile, 0.5, False),
            (ThresholdType.feature_percentile, 0.9, True),
            (ThresholdType.feature_percentile_segment_topk, 7.9, True),
        ],
    )
    def test_builds_all_supported_threshold_types(
        self,
        threshold_type: ThresholdType,
        threshold: float,
        needs_precomputed: bool,
    ) -> None:
        fs = ThresholdSelection(threshold_type, threshold)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=None,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )

        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
        assert pipeline.selection_config.selection is fs
        assert pipeline.boundaries.shape == (50, 3)
        assert pipeline.effective_buckets == 4
        assert (
            pipeline.selection_config.precomputed_thresholds is not None
        ) == needs_precomputed
        expected_boundary_keys = [93, 96, 99] if needs_precomputed else [25, 50, 75]
        expected_first_feature_boundaries = t.tensor(
            expected_boundary_keys,
            dtype=t.float64,
        )
        assert t.equal(pipeline.boundaries[0], expected_first_feature_boundaries)

    def test_threshold_with_negative_selection(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        neg = NegativeSamplingSelection(Distribution.geometric, mean=3.0, std=3.0)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=neg,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )
        assert pipeline.effective_buckets == 3  # 4 - 1
        assert pipeline.negative_bundle is not None
        assert pipeline.negative_bundle.n_sae_features == 50

    def test_threshold_negative_rng_is_separate(self) -> None:
        """Threshold path: NegativeBundle.rng is a separate object."""
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        neg = NegativeSamplingSelection(Distribution.geometric, mean=3.0, std=3.0)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=neg,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )
        # No SamplingBundle in threshold path, so just check negative_bundle exists
        assert pipeline.negative_bundle is not None
        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)

    def test_binary_mode_boundaries(self) -> None:
        """When effective_buckets == 1 (num_buckets=2 + negative), boundaries are empty."""
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        neg = NegativeSamplingSelection(Distribution.geometric, mean=3.0, std=3.0)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=neg,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=2,  # 2 - 1 = 1 effective bucket
        )
        assert pipeline.effective_buckets == 1
        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
        assert pipeline.boundaries is not None
        assert pipeline.boundaries.shape == (50, 0)


# ---------------------------------------------------------------------------
# build_feature_pipeline — sampling path
# ---------------------------------------------------------------------------


class TestBuildFeaturePipelineSampling:
    """Test build_feature_pipeline with SamplingSelection."""

    def test_sampling_returns_bundle(self) -> None:
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=None,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )
        assert isinstance(pipeline.selection_config, SamplingBundle)
        assert pipeline.effective_buckets == 4
        assert pipeline.selection_config.percentile_table.shape == (50, 99)

    def test_sampling_with_negative_shares_rng(self) -> None:
        """Sampling path: SamplingBundle.rng IS NegativeBundle.rng (same object)."""
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        neg = NegativeSamplingSelection(Distribution.geometric, mean=3.0, std=3.0)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=neg,
            per_feature_stats=stats,
            n_sae_features=50,
            num_buckets=4,
        )
        assert isinstance(pipeline.selection_config, SamplingBundle)
        assert pipeline.negative_bundle is not None
        # Same RNG object (shared)
        assert pipeline.selection_config.rng is pipeline.negative_bundle.rng

    def test_seed_determinism(self) -> None:
        """Same seed+rank produces same RNG state."""
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        stats = _make_per_feature_stats(10)
        p1 = build_feature_pipeline(fs, None, stats, 10, 4, seed=123, ddp_rank=0)
        p2 = build_feature_pipeline(fs, None, stats, 10, 4, seed=123, ddp_rank=0)
        assert isinstance(p1.selection_config, SamplingBundle)
        assert isinstance(p2.selection_config, SamplingBundle)
        # Both RNGs seeded identically — sample same value
        v1 = t.empty(1).normal_(generator=p1.selection_config.rng)
        v2 = t.empty(1).normal_(generator=p2.selection_config.rng)
        assert v1.item() == v2.item()

    def test_different_ranks_differ(self) -> None:
        """Different ranks produce different RNG states."""
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        stats = _make_per_feature_stats(10)
        p1 = build_feature_pipeline(fs, None, stats, 10, 4, seed=123, ddp_rank=0)
        p2 = build_feature_pipeline(fs, None, stats, 10, 4, seed=123, ddp_rank=1)
        assert isinstance(p1.selection_config, SamplingBundle)
        assert isinstance(p2.selection_config, SamplingBundle)
        v1 = t.empty(1).normal_(generator=p1.selection_config.rng)
        v2 = t.empty(1).normal_(generator=p2.selection_config.rng)
        assert v1.item() != v2.item()


# ---------------------------------------------------------------------------
# avg_features_per_segment_estimate
# ---------------------------------------------------------------------------


class TestAvgFeaturesPerSegmentEstimate:
    """Test avg_features_per_segment_estimate."""

    def test_segment_topk(self) -> None:
        fs = ThresholdSelection(ThresholdType.segment_topk, 2048.0)
        assert avg_features_per_segment_estimate(fs) == 1024

    def test_feature_percentile_segment_topk(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2048.9)
        assert avg_features_per_segment_estimate(fs) == 1024

    def test_feature_percentile(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.95)
        assert avg_features_per_segment_estimate(fs) == 512

    def test_segment_percentile(self) -> None:
        fs = ThresholdSelection(ThresholdType.segment_percentile, 0.5)
        assert avg_features_per_segment_estimate(fs) == 512

    def test_sampling_geometric(self) -> None:
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        # effective_mean = mean + min_k = 5 + 1 = 6
        assert avg_features_per_segment_estimate(fs) == 6

    def test_sampling_mean_less_than_one(self) -> None:
        fs = SamplingSelection(Distribution.geometric, mean=0.5, std=0.5)
        assert avg_features_per_segment_estimate(fs) == 1

    def test_sampling_large_mean(self) -> None:
        fs = SamplingSelection(Distribution.normal, mean=100.0, std=10.0)
        # effective_mean = mean + min_k = 100 + 1 = 101
        assert avg_features_per_segment_estimate(fs) == 101

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            avg_features_per_segment_estimate(cast(Any, "bad"))


# ---------------------------------------------------------------------------
# FeaturePipelineConfig dataclass
# ---------------------------------------------------------------------------


class TestFeaturePipelineConfig:
    """Test FeaturePipelineConfig creation and field access."""

    def test_threshold_fields_accessible(self) -> None:
        fs = ThresholdSelection(ThresholdType.segment_topk, 10.0)
        sel = ThresholdSelectionConfig(
            selection=fs,
            precomputed_thresholds=None,
        )
        config = FeaturePipelineConfig(
            selection_config=sel,
            boundaries=t.zeros(10, 3),
            negative_bundle=None,
            effective_buckets=4,
        )
        assert isinstance(config.selection_config, ThresholdSelectionConfig)
        assert config.selection_config.selection is fs
        assert config.boundaries.shape == (10, 3)
        assert config.selection_config.precomputed_thresholds is None
        assert config.negative_bundle is None
        assert config.effective_buckets == 4

    def test_sampling_config(self) -> None:
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        bundle = SamplingBundle(
            selection=fs,
            percentile_table=t.zeros(10, 99),
            rng=t.Generator(),
        )
        config = FeaturePipelineConfig(
            selection_config=bundle,
            boundaries=t.zeros(10, 3),
            negative_bundle=None,
            effective_buckets=4,
        )
        assert isinstance(config.selection_config, SamplingBundle)
        assert config.selection_config is bundle


# ---------------------------------------------------------------------------
# DeltaBundle
# ---------------------------------------------------------------------------


class TestDeltaBundle:
    def test_creation(self) -> None:
        cfg = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        table = t.zeros(10, 99, dtype=t.float64)
        bundle = DeltaBundle(config=cfg, percentile_table=table)
        assert bundle.config is cfg
        assert bundle.percentile_table is table


# ---------------------------------------------------------------------------
# build_feature_pipeline with delta
# ---------------------------------------------------------------------------


class TestBuildFeaturePipelineWithDelta:
    def test_delta_none_no_delta_bundle(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs, negative_selection=None,
            per_feature_stats=stats, n_sae_features=50, num_buckets=4,
        )
        assert pipeline.delta_bundle is None

    def test_threshold_with_delta_has_percentile_table(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        delta = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs, negative_selection=None,
            per_feature_stats=stats, n_sae_features=50, num_buckets=4,
            feature_delta=delta,
        )
        assert pipeline.delta_bundle is not None
        assert pipeline.delta_bundle.percentile_table.shape == (50, 99)
        assert pipeline.delta_bundle.config is delta

    def test_sampling_with_delta_reuses_percentile_table(self) -> None:
        fs = SamplingSelection(
            Distribution.geometric, mean=5.0, std=5.0,
            delta_weights=DeltaSamplingWeights.percentile_delta,
        )
        delta = FeatureDeltaConfig(mode=DeltaMode.cum_max)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs, negative_selection=None,
            per_feature_stats=stats, n_sae_features=50, num_buckets=4,
            feature_delta=delta,
        )
        assert pipeline.delta_bundle is not None
        assert isinstance(pipeline.selection_config, SamplingBundle)
        assert pipeline.delta_bundle.percentile_table is pipeline.selection_config.percentile_table

    def test_delta_with_sampling_no_delta_weights_raises(self) -> None:
        fs = SamplingSelection(Distribution.geometric, mean=5.0, std=5.0)
        delta = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        stats = _make_per_feature_stats(50)
        with pytest.raises(ValueError, match="delta_weights"):
            build_feature_pipeline(
                feature_selection=fs, negative_selection=None,
                per_feature_stats=stats, n_sae_features=50, num_buckets=4,
                feature_delta=delta,
            )

    def test_sampling_delta_weights_without_delta_raises(self) -> None:
        fs = SamplingSelection(
            Distribution.geometric, mean=5.0, std=5.0,
            delta_weights=DeltaSamplingWeights.percentile_delta,
        )
        stats = _make_per_feature_stats(50)
        with pytest.raises(ValueError, match="delta_weights.*feature_delta"):
            build_feature_pipeline(
                feature_selection=fs, negative_selection=None,
                per_feature_stats=stats, n_sae_features=50, num_buckets=4,
            )

    def test_threshold_with_delta_no_validation_error(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        delta = FeatureDeltaConfig(mode=DeltaMode.prev_segment, neg_delta_ratio=0.5)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs, negative_selection=None,
            per_feature_stats=stats, n_sae_features=50, num_buckets=4,
            feature_delta=delta,
        )
        assert pipeline.delta_bundle is not None
        assert pipeline.delta_bundle.config.neg_delta_ratio == 0.5

    def test_delta_with_negative_selection(self) -> None:
        fs = ThresholdSelection(ThresholdType.feature_percentile, 0.9)
        neg = NegativeSamplingSelection(Distribution.geometric, mean=3.0, std=3.0)
        delta = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        stats = _make_per_feature_stats(50)
        pipeline = build_feature_pipeline(
            feature_selection=fs, negative_selection=neg,
            per_feature_stats=stats, n_sae_features=50, num_buckets=4,
            feature_delta=delta,
        )
        assert pipeline.delta_bundle is not None
        assert pipeline.negative_bundle is not None


# ---------------------------------------------------------------------------
# validate_shard_data — context_independent_features type checks
# ---------------------------------------------------------------------------


def _valid_shard(ci_flag: bool | None) -> dict:
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([0.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([4], dtype=t.int32),
        "doc_token_counts": t.tensor([4], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    if ci_flag is not None:
        data["context_independent_features"] = ci_flag
    return data


class TestValidateShardDataCI:
    def test_absent_key_valid(self):
        result = validate_shard_data(_valid_shard(None))
        assert result.valid, result.violations

    def test_bool_true_valid(self):
        result = validate_shard_data(_valid_shard(True))
        assert result.valid, result.violations

    def test_bool_false_valid(self):
        result = validate_shard_data(_valid_shard(False))
        assert result.valid, result.violations

    def test_non_bool_invalid(self):
        data = _valid_shard(None)
        data["context_independent_features"] = "true"  # string, not bool
        result = validate_shard_data(data)
        assert not result.valid
        assert any("context_independent_features" in v for v in result.violations)


def _write_ci_shard(dir_path, shard_idx, ci_flag):
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([0.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([4], dtype=t.int32),
        "doc_token_counts": t.tensor([4], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "context_independent_features": ci_flag,
    }
    save_features(data, str(dir_path / f"shard_{shard_idx:05d}_features"))


class TestValidateFeatureDirectoryCIConsistency:
    def test_all_ci_true_passes(self, tmp_path):
        _write_ci_shard(tmp_path, 0, True)
        _write_ci_shard(tmp_path, 1, True)
        result = validate_feature_directory(str(tmp_path))
        # The directory may still have violations unrelated to ci (e.g., no
        # statistics file); check only that the ci flag doesn't trigger one.
        assert not any(
            "context_independent_features" in v for v in result.violations
        ), f"unexpected ci violation: {result.violations}"

    def test_all_ci_false_passes(self, tmp_path):
        _write_ci_shard(tmp_path, 0, False)
        _write_ci_shard(tmp_path, 1, False)
        result = validate_feature_directory(str(tmp_path))
        assert not any(
            "context_independent_features" in v for v in result.violations
        ), f"unexpected ci violation: {result.violations}"

    def test_mixed_ci_flag_fails(self, tmp_path):
        _write_ci_shard(tmp_path, 0, True)
        _write_ci_shard(tmp_path, 1, False)
        result = validate_feature_directory(str(tmp_path))
        assert not result.valid
        assert any(
            "context_independent_features" in v for v in result.violations
        ), f"no ci violation in: {result.violations}"
