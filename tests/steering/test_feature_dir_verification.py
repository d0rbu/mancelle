"""Unit tests for feature directory verification.

Run with:
    uv run pytest tests/test_feature_dir_verification.py -v
"""

from pathlib import Path

import pytest
import torch as t

from steering.feature_io import save_features
from steering.feature_paths import (
    DatasetPhase,
    FeatureDirectoryMismatchError,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
)
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    SegmentFeatureStats,
    StorageStats,
)


class TestSAEWidthMapping:
    """SAE_WIDTH_TO_NUM_FEATURES maps every SAEWidth to the correct integer."""

    def test_all_widths_present(self):
        for w in SAEWidth:
            assert w in SAE_WIDTH_TO_NUM_FEATURES, f"Missing mapping for {w}"

    @pytest.mark.parametrize(
        "width,expected",
        [
            (SAEWidth.W_16K, 16384),
            (SAEWidth.W_32K, 32768),
            (SAEWidth.W_65K, 65536),
            (SAEWidth.W_131K, 131072),
            (SAEWidth.W_262K, 262144),
            (SAEWidth.W_524K, 524288),
            (SAEWidth.W_1M, 1048576),
        ],
    )
    def test_width_values(self, width, expected):
        assert SAE_WIDTH_TO_NUM_FEATURES[width] == expected


def _make_shard(
    *,
    n_docs: int = 2,
    n_segments: int = 4,
    segment_size: int = 64,
    layer: int = 8,
    pooling: str = "max_pool",
    context_window: int = 2048,
    segment_mode: str = "fixed_tokens",
) -> dict:
    """Build a minimal valid feature shard dict."""
    n_features_total = n_segments * 2
    offsets = t.tensor([0, 2, 4, 6, 8], dtype=t.int64)
    feature_ids = t.randint(0, 100, (n_features_total,), dtype=t.int32)
    feature_mags = t.rand(n_features_total, dtype=t.float32)
    doc_ids = t.tensor([0, 0, 1, 1], dtype=t.int32)
    segment_indices = t.tensor([0, 1, 0, 1], dtype=t.int32)
    segment_positions = t.tensor([0, 64, 0, 64], dtype=t.int32)
    doc_token_counts = t.tensor([128, 128], dtype=t.int32)

    return {
        "offsets": offsets,
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "doc_ids": doc_ids,
        "segment_indices": segment_indices,
        "segment_positions": segment_positions,
        "doc_token_counts": doc_token_counts,
        "n_docs": n_docs,
        "n_segments": n_segments,
        "segment_size": segment_size,
        "layer": layer,
        "pooling": pooling,
        "context_window": context_window,
        "segment_mode": segment_mode,
    }


def _make_stats(
    *,
    segment_size: int = 64,
    layer: int = 8,
    pooling_mode: str = "max_pool",
    n_sae_features: int = 16384,
    context_window: int = 2048,
    segment_mode: str = "fixed_tokens",
) -> ExtractionStatistics:
    """Build a minimal ExtractionStatistics for testing."""
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode=pooling_mode,
            segment_mode=SegmentMode(segment_mode),
            segment_size=segment_size,
            context_window=context_window,
            layer=layer,
            n_sae_features=n_sae_features,
            generated_at="2026-01-01T00:00:00",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=2,
            n_tokens_gpt2=256,
            mean_tokens_per_doc=128.0,
            median_tokens_per_doc=128,
        ),
        features=SegmentFeatureStats(
            n_segments=4,
            n_features_seen=100,
            mean_active_per_segment=2.0,
            median_active_per_segment=2,
            p90_active_per_segment=2,
            max_active_per_segment=2,
            min_active_per_segment=2,
            n_total_sae_features=n_sae_features,
            pct_features_active=0.6,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram=None,
            active_per_segment_percentiles={},
            mag_histogram_bin_edges=None,
            segment_length_histogram=None,
            segment_length_percentiles={},
            features_per_token_histogram=None,
            features_per_token_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1024, mean_bytes_per_doc=512.0, mean_bytes_per_token=4.0
        ),
        per_feature_stats={},
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1000000,
            estimated_storage_gb=10.0,
            estimated_wall_time_hours=5.0,
        ),
        per_feature_thresholds=None,
    )


def _build_valid_feature_dir(tmp_path: Path, **overrides) -> Path:
    """Create a feature directory with consistent shards + stats."""
    defaults = {
        "segment_size": 64,
        "layer": 8,
        "activation_type": ActivationType.residual,
        "pooling": PoolingMode.max_pool,
        "width": SAEWidth.W_16K,
        "model": "gemma-2-2b",
        "dataset_phase": DatasetPhase.BASE,
    }
    defaults.update(overrides)

    feature_dir = build_feature_dir(root=tmp_path, **defaults)  # ty: ignore[invalid-argument-type]
    feature_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        shard = _make_shard(
            segment_size=defaults["segment_size"],
            layer=defaults["layer"],
            pooling=defaults["pooling"].value,
        )
        save_features(shard, feature_dir / f"shard_{i:05d}_features")

    stats = _make_stats(
        segment_size=defaults["segment_size"],
        layer=defaults["layer"],
        pooling_mode=defaults["pooling"].value,
        n_sae_features=SAE_WIDTH_TO_NUM_FEATURES[defaults["width"]],
    )
    save_statistics(stats, feature_dir / "feature_statistics")

    return feature_dir


class TestVerifyFeatureDirHappyPath:
    """verify_feature_dir should pass when everything is consistent."""

    def test_basic_valid_dir(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        verify_feature_dir(feature_dir)

    @pytest.mark.parametrize("phase", list(DatasetPhase))
    def test_all_phases(self, tmp_path, phase):
        feature_dir = _build_valid_feature_dir(tmp_path, dataset_phase=phase)
        verify_feature_dir(feature_dir)

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_all_pooling_modes(self, tmp_path, pooling):
        feature_dir = _build_valid_feature_dir(tmp_path, pooling=pooling)
        verify_feature_dir(feature_dir)

    @pytest.mark.parametrize("act", list(ActivationType))
    def test_all_activation_types(self, tmp_path, act):
        feature_dir = _build_valid_feature_dir(tmp_path, activation_type=act)
        verify_feature_dir(feature_dir)

    @pytest.mark.parametrize("width", [SAEWidth.W_16K, SAEWidth.W_65K])
    def test_multiple_widths(self, tmp_path, width):
        feature_dir = _build_valid_feature_dir(tmp_path, width=width)
        verify_feature_dir(feature_dir)


class TestVerifyFeatureDirShardMismatch:
    """verify_feature_dir should raise when shard metadata doesn't match path."""

    def test_wrong_layer_in_shard(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_shard = _make_shard(layer=12)
        save_features(bad_shard, feature_dir / "shard_00001_features")
        with pytest.raises(FeatureDirectoryMismatchError, match="layer"):
            verify_feature_dir(feature_dir)

    def test_wrong_segment_size_in_shard(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_shard = _make_shard(segment_size=128)
        save_features(bad_shard, feature_dir / "shard_00001_features")
        with pytest.raises(FeatureDirectoryMismatchError, match="segment_size"):
            verify_feature_dir(feature_dir)

    def test_wrong_pooling_in_shard(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_shard = _make_shard(pooling="mean_pool")
        save_features(bad_shard, feature_dir / "shard_00001_features")
        with pytest.raises(FeatureDirectoryMismatchError, match="pooling"):
            verify_feature_dir(feature_dir)

    def test_multiple_mismatches_reported(self, tmp_path):
        """All mismatches should be listed, not just the first one."""
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_shard = _make_shard(layer=12, segment_size=128, pooling="mean_pool")
        save_features(bad_shard, feature_dir / "shard_00001_features")
        with pytest.raises(FeatureDirectoryMismatchError) as exc_info:
            verify_feature_dir(feature_dir)
        msg = str(exc_info.value)
        assert "layer" in msg
        assert "segment_size" in msg
        assert "pooling" in msg


class TestVerifyFeatureDirStatsMismatch:
    """verify_feature_dir should raise when stats metadata doesn't match path."""

    def test_wrong_layer_in_stats(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_stats = _make_stats(layer=12)
        save_statistics(bad_stats, feature_dir / "feature_statistics")
        with pytest.raises(FeatureDirectoryMismatchError, match="layer"):
            verify_feature_dir(feature_dir)

    def test_wrong_segment_size_in_stats(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_stats = _make_stats(segment_size=128)
        save_statistics(bad_stats, feature_dir / "feature_statistics")
        with pytest.raises(FeatureDirectoryMismatchError, match="segment_size"):
            verify_feature_dir(feature_dir)

    def test_wrong_pooling_in_stats(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_stats = _make_stats(pooling_mode="mean_pool")
        save_statistics(bad_stats, feature_dir / "feature_statistics")
        with pytest.raises(FeatureDirectoryMismatchError, match="pooling"):
            verify_feature_dir(feature_dir)

    def test_wrong_n_sae_features_in_stats(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        bad_stats = _make_stats(n_sae_features=65536)
        save_statistics(bad_stats, feature_dir / "feature_statistics")
        with pytest.raises(FeatureDirectoryMismatchError, match="n_sae_features"):
            verify_feature_dir(feature_dir)


class TestVerifyFeatureDirMissingFiles:
    """verify_feature_dir should raise when expected files are absent."""

    def test_no_shards(self, tmp_path):
        feature_dir = build_feature_dir(
            root=tmp_path,
            segment_size=64,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True)
        stats = _make_stats()
        save_statistics(stats, feature_dir / "feature_statistics")
        with pytest.raises(FeatureDirectoryMismatchError, match="[Nn]o.*shard"):
            verify_feature_dir(feature_dir)

    def test_no_statistics(self, tmp_path):
        feature_dir = build_feature_dir(
            root=tmp_path,
            segment_size=64,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True)
        shard = _make_shard()
        save_features(shard, feature_dir / "shard_00000_features")
        with pytest.raises(FeatureDirectoryMismatchError, match="[Ss]tatistics"):
            verify_feature_dir(feature_dir)

    def test_dir_does_not_exist(self, tmp_path):
        fake = (
            tmp_path
            / "s64"
            / "l8"
            / "a_res"
            / "p_max_pool"
            / "f16k"
            / "m_gemma-2-2b"
            / "base"
        )
        with pytest.raises(FeatureDirectoryMismatchError, match="does not exist"):
            verify_feature_dir(fake)


class TestVerifyFeatureDirLegacyShards:
    """verify_feature_dir should gracefully skip .pt shards (no metadata header)."""

    def test_pt_shards_skipped(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        legacy_shard = {"offsets": t.tensor([0, 2], dtype=t.int64)}
        t.save(legacy_shard, feature_dir / "shard_00099_features.pt")
        verify_feature_dir(feature_dir)


class TestTrainingScriptImport:
    """Verify the import path used by the training script works."""

    def test_import_verify_feature_dir(self):
        from steering.feature_paths import verify_feature_dir

        assert callable(verify_feature_dir)

    def test_import_error_class(self):
        from steering.feature_paths import FeatureDirectoryMismatchError

        assert issubclass(FeatureDirectoryMismatchError, Exception)


class TestVerifyFeatureDirContextIndependent:
    def _ci_feature_dir(self, tmp_path: Path) -> Path:
        """Create a legal ci feature directory with one shard + statistics."""
        d = (
            tmp_path
            / "s128" / "l8" / "a_res" / "p_max_pool" / "f16k"
            / "m_gemma-2-2b" / "base" / "context_independent"
        )
        d.mkdir(parents=True)
        shard = _make_shard(segment_size=128, layer=8, pooling="max_pool")
        shard["context_independent_features"] = True
        save_features(shard, d / "shard_00000_features")
        save_statistics(
            _make_stats(segment_size=128, layer=8, pooling_mode="max_pool",
                        n_sae_features=16384),
            d / "feature_statistics",
        )
        return d

    def _cd_feature_dir(self, tmp_path: Path) -> Path:
        d = (
            tmp_path
            / "s128" / "l8" / "a_res" / "p_max_pool" / "f16k"
            / "m_gemma-2-2b" / "base"
        )
        d.mkdir(parents=True)
        shard = _make_shard(segment_size=128, layer=8, pooling="max_pool")
        shard["context_independent_features"] = False
        save_features(shard, d / "shard_00000_features")
        save_statistics(
            _make_stats(segment_size=128, layer=8, pooling_mode="max_pool",
                        n_sae_features=16384),
            d / "feature_statistics",
        )
        return d

    def test_ci_path_with_ci_metadata_passes(self, tmp_path):
        feature_dir = self._ci_feature_dir(tmp_path)
        verify_feature_dir(feature_dir)  # should not raise

    def test_cd_path_with_cd_metadata_passes(self, tmp_path):
        feature_dir = self._cd_feature_dir(tmp_path)
        verify_feature_dir(feature_dir)  # should not raise

    def test_ci_path_cd_metadata_raises(self, tmp_path):
        """Path ends in /context_independent but shard metadata says False → raise."""
        d = (
            tmp_path
            / "s128" / "l8" / "a_res" / "p_max_pool" / "f16k"
            / "m_gemma-2-2b" / "base" / "context_independent"
        )
        d.mkdir(parents=True)
        shard = _make_shard(segment_size=128, layer=8, pooling="max_pool")
        shard["context_independent_features"] = False
        save_features(shard, d / "shard_00000_features")
        save_statistics(
            _make_stats(segment_size=128, layer=8, pooling_mode="max_pool",
                        n_sae_features=16384),
            d / "feature_statistics",
        )
        with pytest.raises(FeatureDirectoryMismatchError, match="context_independent"):
            verify_feature_dir(d)

    def test_cd_path_ci_metadata_raises(self, tmp_path):
        """Path lacks /context_independent but shard metadata says True → raise."""
        d = (
            tmp_path
            / "s128" / "l8" / "a_res" / "p_max_pool" / "f16k"
            / "m_gemma-2-2b" / "base"
        )
        d.mkdir(parents=True)
        shard = _make_shard(segment_size=128, layer=8, pooling="max_pool")
        shard["context_independent_features"] = True
        save_features(shard, d / "shard_00000_features")
        save_statistics(
            _make_stats(segment_size=128, layer=8, pooling_mode="max_pool",
                        n_sae_features=16384),
            d / "feature_statistics",
        )
        with pytest.raises(FeatureDirectoryMismatchError, match="context_independent"):
            verify_feature_dir(d)


class TestVerifyFeatureDirL0TierRefactor:
    """Behavioural contract for the assert_uniform_l0_tier refactor.

    NOT a byte-identical golden: a self-contained, at-least-as-informative
    message is an acceptable refactor. Asserts: (1) a mixed-tier dir
    surfaces FeatureDirectoryMismatchError, (2) the l0 problem is
    ACCUMULATED with other errors (not early-raised), (3) the message
    names every offending shard and its tier, (4) a uniform-v2 dir passes
    the l0 check.
    """

    def _restamp_shards(self, feature_dir, tiers):
        # Overwrite the dir's 2 shards keeping path-consistent
        # layer/segment/pooling, varying only l0_tier.
        for i, tier in enumerate(tiers):
            shard = _make_shard(segment_size=64, layer=8, pooling="max_pool")
            if tier is not None:
                shard["l0_tier"] = tier
            save_features(shard, feature_dir / f"shard_{i:05d}_features")

    def test_mixed_tier_dir_accumulates_and_raises(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        self._restamp_shards(feature_dir, ["medium", "big"])
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            verify_feature_dir(feature_dir)
        msg = str(ei.value)
        assert "l0_tier" in msg
        assert "medium" in msg and "big" in msg

    def test_uniform_v2_tier_dir_passes_l0_check(self, tmp_path):
        feature_dir = _build_valid_feature_dir(tmp_path)
        self._restamp_shards(feature_dir, ["medium", "medium"])
        # A fully-valid uniform-v2 dir must not raise (l0 reason).
        verify_feature_dir(feature_dir)

    def test_l0_problem_accumulated_with_other_errors(self, tmp_path):
        """An l0 mismatch AND a missing-stats error both appear in ONE raise
        (proves accumulate-then-raise, not early-raise on l0)."""
        feature_dir = _build_valid_feature_dir(tmp_path)
        self._restamp_shards(feature_dir, ["medium", "big"])
        for p in feature_dir.glob("feature_statistics*"):
            p.unlink()
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            verify_feature_dir(feature_dir)
        msg = str(ei.value)
        assert "Statistics file not found" in msg  # the other error
        assert "l0_tier" in msg                     # the l0 error too
