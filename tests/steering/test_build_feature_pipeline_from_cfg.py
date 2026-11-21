"""Tests for ``build_feature_pipeline_from_cfg``: reconstruct the
``FeaturePipelineConfig`` used at training time from ``cfg.conditioning`` +
auto-resolved stats file (or explicit ``cfg.bpb.stats_path``).

This replaces the placeholder pipeline and the `_validate_selection_shard_coverage`
hard-fail. The fused-inline path in the dataloader can now run safely without
selection shards because the real feature_pipeline is reconstructed from cfg.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_extraction_statistics(
    *,
    n_sae_features: int = 16384,
    n_active: int = 8,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> ExtractionStatistics:
    """Build a minimal ExtractionStatistics with `n_active` active features.

    Provides all percentile keys p1..p99 (required by ``build_percentile_table``
    and ``build_boundaries_from_stats``) plus standard "p50/p90/p95/p99" labels.
    """
    # Full 1..99 percentile keys (required by build_percentile_table) plus
    # p95 / p98 (required by some boundary builders).
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    per_feature_stats: dict[str, PerFeatureStat] = {}
    for fid in range(n_active):
        per_feature_stats[str(fid)] = PerFeatureStat(
            count=100 - fid,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(full_percentiles),
            histogram=None,
        )
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=segment_mode,
            segment_size=128,
            context_window=1024,
            layer=8,
            n_sae_features=n_sae_features,
            generated_at="2025-01-01T00:00:00",
            activation_percentile=0.95,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=10000,
            mean_tokens_per_doc=100.0,
            median_tokens_per_doc=100,
        ),
        features=SegmentFeatureStats(
            n_segments=100,
            n_features_seen=n_active,
            mean_active_per_segment=2.0,
            median_active_per_segment=2,
            p90_active_per_segment=4,
            max_active_per_segment=8,
            min_active_per_segment=1,
            n_total_sae_features=n_sae_features,
            pct_features_active=float(n_active) / n_sae_features,
            count_percentiles={"p50": 50, "p90": 90, "p99": 99},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1000,
            mean_bytes_per_doc=10.0,
            mean_bytes_per_token=0.1,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
        per_feature_thresholds=None,
    )


def _write_stats_at(
    stats_dir: Path,
    n_sae_features: int = 16384,
    *,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> Path:
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats = _make_minimal_extraction_statistics(
        n_sae_features=n_sae_features,
        segment_mode=segment_mode,
    )
    out = save_statistics(stats, stats_dir / "feature_statistics")
    return out


def _make_cfg(
    *,
    stats_path: str | None = None,
    feature_root: str | None = None,
    segment_size: int | None = 128,
    bpb_segment_size: int | None = None,
    dataset_phase: str | None = None,
    n_sae_features: int = 16384,
    num_buckets: int = 2,
    feature_selection: dict | None = None,
    negative_selection: dict | None = None,
):
    """Construct a cfg-like OmegaConf object with the fields
    build_feature_pipeline_from_cfg expects."""
    if feature_selection is None:
        feature_selection = {
            "threshold_type": "feature_percentile",
            "threshold": 0.95,
        }
    cfg_dict = {
        "conditioning": {
            "identity": {
                "model_id": "gemma-2-2b",
                "sae_id": "gemmascope-residual-l8-16k",
                "layer": 8,
                "pooling": "max_pool",
                "feature_selection_hash": None,
            },
            "n_sae_features": n_sae_features,
            "num_buckets": num_buckets,
            "feature_selection": feature_selection,
            "negative_selection": negative_selection,
        },
        "bpb": {
            "stats_path": stats_path,
            "feature_root": feature_root,
            "segment_size": bpb_segment_size,
        },
        "segment_size": segment_size,
        "dataset_phase": dataset_phase,
    }
    return OmegaConf.create(cfg_dict)


# ---------------------------------------------------------------------------
# build_feature_pipeline_from_cfg
# ---------------------------------------------------------------------------


def test_build_feature_pipeline_from_cfg_uses_explicit_stats_path(tmp_path):
    """When ``cfg.bpb.stats_path`` is set, it overrides feature_paths
    auto-resolution."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats_explicit"
    stats_path = _write_stats_at(stats_dir)

    cfg = _make_cfg(stats_path=str(stats_path))
    pipeline = build_feature_pipeline_from_cfg(cfg)

    assert isinstance(pipeline, FeaturePipelineConfig)
    # threshold path: selection_config is ThresholdSelectionConfig
    from steering.conditional.feature_pipeline import ThresholdSelectionConfig
    assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
    assert pipeline.selection_config.selection.threshold_type is ThresholdType.feature_percentile
    assert pipeline.selection_config.selection.threshold == 0.95


def test_build_feature_pipeline_from_cfg_rejects_turn_statistics_before_builder(
    tmp_path: Path,
) -> None:
    """The fixed-geometry BPB loader must not build from turn-segmented stats."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_path = _write_stats_at(
        tmp_path / "turn_stats",
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    cfg = _make_cfg(stats_path=str(stats_path))

    with patch(
        "steering.eval.bpb_pipeline.build_feature_pipeline_from_cfg_block",
        side_effect=AssertionError("pipeline builder must not run"),
    ) as builder:
        with pytest.raises(
            ValueError,
            match=(
                r"conditioned BPB evaluation.*feature directory.*"
                r"turn_delimited_fixed_tokens.*fixed_tokens"
            ),
        ):
            build_feature_pipeline_from_cfg(cfg)

    builder.assert_not_called()


def test_build_feature_pipeline_from_cfg_rejects_turn_feature_path_witness(
    tmp_path: Path,
) -> None:
    """The canonical turn directory is a witness even when the stats say fixed."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    turn_dir = build_feature_dir(
        root=tmp_path / "features",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    stats_path = _write_stats_at(turn_dir, segment_mode=SegmentMode.fixed_tokens)
    assert stats_path.parent == turn_dir
    cfg = _make_cfg(stats_path=str(stats_path))

    with patch(
        "steering.eval.bpb_pipeline.build_feature_pipeline_from_cfg_block",
        side_effect=AssertionError("pipeline builder must not run"),
    ) as builder:
        with pytest.raises(
            ValueError,
            match=(
                r"conditioned BPB evaluation feature artifact segment-mode mismatch"
                r".*feature path='turn_delimited_fixed_tokens'"
                r".*feature statistics='fixed_tokens'"
            ),
        ):
            build_feature_pipeline_from_cfg(cfg)

    builder.assert_not_called()


def test_build_feature_pipeline_from_cfg_reads_feature_selection(tmp_path):
    """Pipeline's selection matches cfg.conditioning.feature_selection."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats"
    stats_path = _write_stats_at(stats_dir)

    cfg = _make_cfg(
        stats_path=str(stats_path),
        feature_selection={
            "threshold_type": "feature_percentile",
            "threshold": 0.85,
        },
    )
    pipeline = build_feature_pipeline_from_cfg(cfg)
    from steering.conditional.feature_pipeline import ThresholdSelectionConfig
    assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
    sel = pipeline.selection_config.selection
    assert isinstance(sel, ThresholdSelection)
    assert sel.threshold_type is ThresholdType.feature_percentile
    assert sel.threshold == 0.85


def test_build_feature_pipeline_from_cfg_supports_sampling_selection(tmp_path):
    """SamplingSelection is also reconstructible (e.g. distribution=geometric)."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats"
    stats_path = _write_stats_at(stats_dir)

    cfg = _make_cfg(
        stats_path=str(stats_path),
        feature_selection={
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "min_k": 1,
            "max_k": 25,
        },
    )
    pipeline = build_feature_pipeline_from_cfg(cfg)
    from steering.conditional.feature_pipeline import SamplingBundle
    assert isinstance(pipeline.selection_config, SamplingBundle)
    sel = pipeline.selection_config.selection
    assert isinstance(sel, SamplingSelection)


def test_build_feature_pipeline_from_cfg_includes_negative_selection(tmp_path):
    """negative_selection from cfg is wired into the negative_bundle."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats"
    stats_path = _write_stats_at(stats_dir)

    cfg = _make_cfg(
        stats_path=str(stats_path),
        num_buckets=2,
        negative_selection={
            "distribution": "geometric",
            "mean": 3.0,
            "min_k": 0,
            "max_k": 15,
        },
    )
    pipeline = build_feature_pipeline_from_cfg(cfg)
    assert pipeline.negative_bundle is not None
    # effective_buckets is num_buckets - 1 when negative_selection is present
    assert pipeline.effective_buckets == 1


def test_build_feature_pipeline_from_cfg_no_negative_selection_keeps_buckets(
    tmp_path,
):
    """When negative_selection is None, effective_buckets == num_buckets."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats"
    stats_path = _write_stats_at(stats_dir)

    cfg = _make_cfg(
        stats_path=str(stats_path),
        num_buckets=4,
        negative_selection=None,
    )
    pipeline = build_feature_pipeline_from_cfg(cfg)
    assert pipeline.negative_bundle is None
    assert pipeline.effective_buckets == 4


def test_build_feature_pipeline_from_cfg_auto_resolves_stats_via_feature_paths(
    tmp_path,
):
    """When stats_path is null, resolve via feature_paths convention from
    cfg.conditioning.identity + cfg.segment_size + cfg.dataset_phase."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    feature_root = tmp_path / "features"
    feature_dir = build_feature_dir(
        root=feature_root,
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
    )
    _write_stats_at(feature_dir)

    cfg = _make_cfg(
        stats_path=None,
        feature_root=str(feature_root),
        segment_size=128,
        dataset_phase="base",
    )
    pipeline = build_feature_pipeline_from_cfg(cfg)
    assert isinstance(pipeline, FeaturePipelineConfig)


def test_build_feature_pipeline_from_cfg_raises_when_stats_unresolvable(tmp_path):
    """No explicit path + no stats found in canonical feature_dir →
    operator-actionable ValueError mentioning both fallback options."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    # Construct a feature_root that points to an empty dir.
    feature_root = tmp_path / "empty_features"
    feature_root.mkdir()

    cfg = _make_cfg(
        stats_path=None,
        feature_root=str(feature_root),
        segment_size=128,
        dataset_phase="base",
    )
    with pytest.raises(ValueError) as exc_info:
        build_feature_pipeline_from_cfg(cfg)
    msg = str(exc_info.value).lower()
    assert "stats" in msg
    # Mention both fallback options so operators know how to fix
    assert "stats_path" in msg
    assert "statistics_" in msg or "feature_statistics" in msg


def test_build_feature_pipeline_from_cfg_uses_bpb_segment_size_override(tmp_path):
    """``cfg.bpb.segment_size`` overrides ``cfg.segment_size`` for stats path
    auto-resolution."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    feature_root = tmp_path / "features"
    # Build under segment_size=64 (cfg.bpb.segment_size) NOT 128 (cfg.segment_size)
    feature_dir = build_feature_dir(
        root=feature_root,
        segment_size=64,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
    )
    _write_stats_at(feature_dir)

    cfg = _make_cfg(
        stats_path=None,
        feature_root=str(feature_root),
        segment_size=128,
        bpb_segment_size=64,
        dataset_phase="base",
    )
    # Should resolve via bpb.segment_size=64 path; if it picked up segment_size=128
    # the lookup would fail.
    pipeline = build_feature_pipeline_from_cfg(cfg)
    assert isinstance(pipeline, FeaturePipelineConfig)


def test_build_feature_pipeline_from_cfg_pins_canonical_seed_and_ddp_rank(
    tmp_path,
):
    """seed=0 and ddp_rank=0 are pinned (matches
    ``compute_canonical_feature_selection_hash``)."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats"
    stats_path = _write_stats_at(stats_dir)

    cfg = _make_cfg(
        stats_path=str(stats_path),
        feature_selection={
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "min_k": 1,
            "max_k": 25,
        },
    )

    # Patch build_feature_pipeline to capture call kwargs. The shared
    # helper (build_feature_pipeline_from_cfg_block) imports
    # build_feature_pipeline from steering.conditional.feature_pipeline,
    # so we patch that module-level reference rather than the bpb_pipeline
    # re-export (which no longer exists after the helper migration).
    captured = {}

    from steering.conditional import feature_pipeline as fp_mod
    real_build = fp_mod.build_feature_pipeline

    def _spy_build_feature_pipeline(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)

    with patch(
        "steering.conditional.feature_pipeline.build_feature_pipeline",
        side_effect=_spy_build_feature_pipeline,
    ):
        build_feature_pipeline_from_cfg(cfg)

    assert captured["seed"] == 0
    assert captured["ddp_rank"] == 0
    assert captured["feature_delta"] is None


def test_build_feature_pipeline_from_cfg_dataset_phase_defaults_to_base(tmp_path):
    """When cfg.dataset_phase is null, falls back to 'base'."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    feature_root = tmp_path / "features"
    feature_dir = build_feature_dir(
        root=feature_root,
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
    )
    _write_stats_at(feature_dir)

    cfg = _make_cfg(
        stats_path=None,
        feature_root=str(feature_root),
        segment_size=128,
        dataset_phase=None,  # defaults to base
    )
    pipeline = build_feature_pipeline_from_cfg(cfg)
    assert isinstance(pipeline, FeaturePipelineConfig)


def test_build_feature_pipeline_from_cfg_raises_when_feature_selection_missing(
    tmp_path,
):
    """When cfg.conditioning.feature_selection is missing, the BPB helper
    raises an operator-actionable error naming the new field."""
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    stats_dir = tmp_path / "stats"
    stats_path = _write_stats_at(stats_dir)

    cfg = OmegaConf.create({
        "conditioning": {
            "identity": {
                "model_id": "gemma-2-2b",
                "sae_id": "gemmascope-residual-l8-16k",
                "layer": 8,
                "pooling": "max_pool",
                "feature_selection_hash": None,
            },
            "n_sae_features": 16384,
            "num_buckets": 2,
            # NO feature_selection key
        },
        "bpb": {
            "stats_path": str(stats_path),
            "feature_root": None,
            "segment_size": None,
        },
        "segment_size": 128,
        "dataset_phase": None,
    })
    with pytest.raises(ValueError, match="feature_selection"):
        build_feature_pipeline_from_cfg(cfg)


# ---------------------------------------------------------------------------
# YAML defaults: cfg.bpb has stats_path / feature_root / segment_size knobs
# ---------------------------------------------------------------------------


def test_base_eval_yaml_bpb_has_stats_path_default_null():
    """``cfg.bpb.stats_path`` defaults to null → auto-resolve."""
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs" / "eval"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="base_eval", overrides=["source=base"])
    assert "bpb" in cfg
    assert cfg.bpb.stats_path is None


def test_base_eval_yaml_bpb_selection_hash_is_optional():
    """``cfg.bpb.selection_hash`` defaults to null (selection shards are
    optional, not mandatory)."""
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs" / "eval"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="base_eval", overrides=["source=base"])
    # selection_hash is null (not ???) — operators usually don't have selection shards
    assert cfg.bpb.selection_hash is None
