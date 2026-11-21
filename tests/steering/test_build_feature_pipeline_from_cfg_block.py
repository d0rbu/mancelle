"""Unit tests for ``build_feature_pipeline_from_cfg_block``: ensure the helper
parses feature_selection / negative_selection correctly across cfg layouts and
honours the use_negative_selection / feature_delta flags.
"""

from __future__ import annotations

import pytest
import torch as t
from omegaconf import OmegaConf

from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    SamplingBundle,
    ThresholdSelectionConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
)
from steering.stats_types import PerFeatureStat

N_SAE_FEATURES = 64
NUM_BUCKETS = 4


def _make_per_feature_stats(n: int = N_SAE_FEATURES) -> dict[str, PerFeatureStat]:
    """Build a tiny stats dict with deterministic percentiles."""
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    out: dict[str, PerFeatureStat] = {}
    for i in range(n):
        out[str(i)] = PerFeatureStat(
            count=100 - i,
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
    return out


def _threshold_block_omega():
    return OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile",
            "threshold": 0.95,
        },
    })


def _sampling_block_omega():
    return OmegaConf.create({
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,  # geometric requires std == mean
        },
    })


def _negative_block_dict() -> dict:
    return {"distribution": "geometric", "mean": 2.0, "std": 2.0}


# ---------------------------------------------------------------------------
# Test 1-4: feature_selection parsing
# ---------------------------------------------------------------------------


def test_parses_feature_selection_from_omegaconf_block():
    cfg = _threshold_block_omega()
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert isinstance(pipeline, FeaturePipelineConfig)
    assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
    assert isinstance(pipeline.selection_config.selection, ThresholdSelection)
    assert pipeline.selection_config.selection.threshold == 0.95


def test_parses_feature_selection_from_plain_dict_block():
    cfg = {
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
    }
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
    assert isinstance(pipeline.selection_config.selection, ThresholdSelection)
    assert pipeline.selection_config.selection.threshold == 0.95


def test_threshold_selection_path():
    cfg = _threshold_block_omega()
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
    assert isinstance(pipeline.selection_config.selection, ThresholdSelection)


def test_sampling_selection_path():
    cfg = _sampling_block_omega()
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert isinstance(pipeline.selection_config, SamplingBundle)
    assert isinstance(pipeline.selection_config.selection, SamplingSelection)


# ---------------------------------------------------------------------------
# Test 5-8: negative_selection handling
# ---------------------------------------------------------------------------


def test_negative_selection_read_from_cfg_when_present():
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": _negative_block_dict(),
    })
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert pipeline.negative_bundle is not None
    assert pipeline.effective_buckets == NUM_BUCKETS - 1


def test_negative_selection_none_when_missing():
    cfg = _threshold_block_omega()
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert pipeline.negative_bundle is None
    assert pipeline.effective_buckets == NUM_BUCKETS


def test_negative_selection_none_when_null():
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": None,
    })
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert pipeline.negative_bundle is None


def test_use_negative_selection_false_forces_none():
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": _negative_block_dict(),
    })
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        use_negative_selection=False,
    )
    assert pipeline.negative_bundle is None


# ---------------------------------------------------------------------------
# Test 9-10: feature_delta passthrough
# ---------------------------------------------------------------------------


def test_feature_delta_passed_through():
    cfg = OmegaConf.create({
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,
            "delta_weights": "absolute_percentile",
        },
    })
    fd = FeatureDeltaConfig(
        mode=DeltaMode.prev_segment, neg_delta_ratio=0.5,
    )
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_delta=fd,
    )
    assert pipeline.delta_bundle is not None
    assert pipeline.delta_bundle.config is fd


def test_feature_delta_none_when_unset():
    cfg = _threshold_block_omega()
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert pipeline.delta_bundle is None


# ---------------------------------------------------------------------------
# Test 11-12: required kwargs
# ---------------------------------------------------------------------------


def test_n_sae_features_is_required():
    cfg = _threshold_block_omega()
    # Bypass static type-check: we want to confirm the runtime raises when
    # n_sae_features is omitted.
    fn = build_feature_pipeline_from_cfg_block
    with pytest.raises(TypeError):
        fn(  # ty: ignore[missing-argument]
            cfg,
            per_feature_stats=_make_per_feature_stats(),
            num_buckets=NUM_BUCKETS,
        )


def test_num_buckets_is_required():
    cfg = _threshold_block_omega()
    fn = build_feature_pipeline_from_cfg_block
    with pytest.raises(TypeError):
        fn(  # ty: ignore[missing-argument]
            cfg,
            per_feature_stats=_make_per_feature_stats(),
            n_sae_features=N_SAE_FEATURES,
        )


# ---------------------------------------------------------------------------
# Test 13-14: invalid input handling
# ---------------------------------------------------------------------------


def test_invalid_feature_selection_type_raises():
    cfg = OmegaConf.create({"feature_selection": "not_a_dict"})
    with pytest.raises(TypeError, match="feature_selection must resolve to a mapping"):
        build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_make_per_feature_stats(),
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
        )


def test_invalid_negative_selection_type_raises():
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": "not_a_dict",
    })
    with pytest.raises(TypeError, match="negative_selection must resolve to a mapping"):
        build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_make_per_feature_stats(),
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
        )


# ---------------------------------------------------------------------------
# Test 15-16: seed and ddp_rank threading
# ---------------------------------------------------------------------------


def test_seed_threaded_to_sampling_path():
    cfg = _sampling_block_omega()
    p1 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=1,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=2,
    )
    assert isinstance(p1.selection_config, SamplingBundle)
    assert isinstance(p2.selection_config, SamplingBundle)
    s1 = t.randint(0, 10**9, (1,), generator=p1.selection_config.rng)
    s2 = t.randint(0, 10**9, (1,), generator=p2.selection_config.rng)
    assert s1.item() != s2.item()


def test_ddp_rank_threaded_to_sampling_path():
    cfg = _sampling_block_omega()
    p1 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=42,
        ddp_rank=0,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=42,
        ddp_rank=1,
    )
    assert isinstance(p1.selection_config, SamplingBundle)
    assert isinstance(p2.selection_config, SamplingBundle)
    s1 = t.randint(0, 10**9, (1,), generator=p1.selection_config.rng)
    s2 = t.randint(0, 10**9, (1,), generator=p2.selection_config.rng)
    assert s1.item() != s2.item()
