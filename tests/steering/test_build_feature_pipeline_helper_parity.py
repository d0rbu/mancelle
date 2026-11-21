"""Parity tests: build_feature_pipeline_from_cfg_block must produce the
same FeaturePipelineConfig output as the original inline parse-and-build
dance. One test per migrated call site.

Equality predicate: structural equality on selection_config.selection
(dataclass __eq__), boundaries tensor (torch.equal), effective_buckets,
negative_bundle.selection (when set), delta_bundle.config (when set).
RNG state compared via one-sample draw.
"""

from __future__ import annotations

import torch as t
from omegaconf import OmegaConf

from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_pipeline import (
    SamplingBundle,
    ThresholdSelectionConfig,
    build_feature_pipeline,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.sampling import (
    parse_feature_selection,
    parse_negative_selection,
)
from steering.stats_types import PerFeatureStat

N = 64
B = 4


def _stats() -> dict[str, PerFeatureStat]:
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    out: dict[str, PerFeatureStat] = {}
    for i in range(N):
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


def _assert_pipelines_equal(p1, p2):
    """Structural + tensor equality on FeaturePipelineConfig outputs.

    Equality predicate covers every observable component:
      - selection_config type + dataclass __eq__
      - boundaries tensor (torch.equal — bit-exact)
      - effective_buckets
      - For SamplingBundle: percentile_table tensor + rng output (one draw)
      - For ThresholdSelectionConfig: precomputed_thresholds tensor
      - For NegativeBundle: selection + rng output (one draw)
      - For DeltaBundle: config + percentile_table tensor

    RNG state is compared via a single int64 draw — if the underlying state
    diverges, draws differ. Since builders share rng seeding, draws must
    match bit-for-bit.
    """
    assert type(p1.selection_config) is type(p2.selection_config)
    assert p1.selection_config.selection == p2.selection_config.selection
    assert t.equal(p1.boundaries, p2.boundaries)
    assert p1.effective_buckets == p2.effective_buckets

    # Selection-specific state
    if isinstance(p1.selection_config, SamplingBundle):
        assert isinstance(p2.selection_config, SamplingBundle)
        assert t.equal(
            p1.selection_config.percentile_table,
            p2.selection_config.percentile_table,
        )
        # If no negative_bundle is sharing the rng, also compare rng draw here.
        # Otherwise the negative_bundle compare below covers it (shared rng).
        if p1.negative_bundle is None:
            s1 = t.randint(0, 10**9, (1,), generator=p1.selection_config.rng)
            s2 = t.randint(0, 10**9, (1,), generator=p2.selection_config.rng)
            assert s1.item() == s2.item(), "SamplingBundle rng diverged"
    elif isinstance(p1.selection_config, ThresholdSelectionConfig):
        assert isinstance(p2.selection_config, ThresholdSelectionConfig)
        # precomputed_thresholds: Tensor | set | None
        pt1 = p1.selection_config.precomputed_thresholds
        pt2 = p2.selection_config.precomputed_thresholds
        if pt1 is None:
            assert pt2 is None
        elif isinstance(pt1, t.Tensor):
            assert isinstance(pt2, t.Tensor)
            assert t.equal(pt1, pt2)
        else:
            assert pt1 == pt2

    if p1.negative_bundle is None:
        assert p2.negative_bundle is None
    else:
        assert p2.negative_bundle is not None
        assert p1.negative_bundle.selection == p2.negative_bundle.selection
        s1 = t.randint(0, 10**9, (1,), generator=p1.negative_bundle.rng)
        s2 = t.randint(0, 10**9, (1,), generator=p2.negative_bundle.rng)
        assert s1.item() == s2.item(), "NegativeBundle rng diverged"

    if p1.delta_bundle is None:
        assert p2.delta_bundle is None
    else:
        assert p2.delta_bundle is not None
        assert p1.delta_bundle.config == p2.delta_bundle.config
        assert t.equal(
            p1.delta_bundle.percentile_table, p2.delta_bundle.percentile_table,
        )


def test_parity_build_eval_context_cache_layout():
    """cfg has top-level feature_selection + null negative_selection; no
    feature_delta. Mirrors scripts/eval/build_eval_context_cache.py."""
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": None,
    })
    stats = _stats()

    fs_d = OmegaConf.to_container(cfg.feature_selection, resolve=True)
    fs = parse_feature_selection(fs_d)  # ty: ignore[invalid-argument-type]
    p1 = build_feature_pipeline(
        feature_selection=fs,
        negative_selection=None,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
    )
    _assert_pipelines_equal(p1, p2)


def test_parity_completion_steering_eval_layout():
    """cfg has top-level feature_selection; negative_selection in cfg
    must be IGNORED via use_negative_selection=False. Mirrors
    completion_steering_eval.py."""
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": {
            "distribution": "geometric", "mean": 2.0, "std": 2.0,
        },
    })
    stats = _stats()

    fs_d = OmegaConf.to_container(cfg.feature_selection, resolve=True)
    fs = parse_feature_selection(fs_d)  # ty: ignore[invalid-argument-type]
    p1 = build_feature_pipeline(
        feature_selection=fs,
        negative_selection=None,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        use_negative_selection=False,
    )
    _assert_pipelines_equal(p1, p2)


def test_parity_precompute_selected_features_layout():
    """cfg has top-level fs + neg + feature_delta; explicit seed.
    Mirrors scripts/preprocess/precompute_selected_features.py."""
    cfg = OmegaConf.create({
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0, "std": 4.0,
            "delta_weights": "absolute_percentile",
        },
        "negative_selection": {
            "distribution": "geometric", "mean": 2.0, "std": 2.0,
        },
    })
    fd = FeatureDeltaConfig(mode=DeltaMode.prev_segment, neg_delta_ratio=0.5)
    stats = _stats()

    fs_d = OmegaConf.to_container(cfg.feature_selection, resolve=True)
    fs = parse_feature_selection(fs_d)  # ty: ignore[invalid-argument-type]
    neg_d = OmegaConf.to_container(cfg.negative_selection, resolve=True)
    neg = parse_negative_selection(neg_d)  # ty: ignore[invalid-argument-type]
    p1 = build_feature_pipeline(
        feature_selection=fs,
        negative_selection=neg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        seed=99,
        ddp_rank=0,
        feature_delta=fd,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        seed=99,
        feature_delta=fd,
    )
    _assert_pipelines_equal(p1, p2)


def test_parity_conditional_encdec_train_layout():
    """cfg has top-level fs + neg + feature_delta; DDP-specific seed.
    Mirrors scripts/train/conditional_encdec_train.py."""
    cfg = OmegaConf.create({
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0, "std": 4.0,
            "delta_weights": "absolute_percentile",
        },
        "negative_selection": {
            "distribution": "geometric", "mean": 2.0, "std": 2.0,
        },
    })
    fd = FeatureDeltaConfig(mode=DeltaMode.prev_segment, neg_delta_ratio=0.5)
    stats = _stats()
    rng_seed = (42 * 1000003 + 0) % (2**63)

    fs_d = OmegaConf.to_container(cfg.feature_selection, resolve=True)
    fs = parse_feature_selection(fs_d)  # ty: ignore[invalid-argument-type]
    neg_d = OmegaConf.to_container(cfg.negative_selection, resolve=True)
    neg = parse_negative_selection(neg_d)  # ty: ignore[invalid-argument-type]
    p1 = build_feature_pipeline(
        feature_selection=fs,
        negative_selection=neg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        seed=rng_seed,
        ddp_rank=0,
        feature_delta=fd,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        seed=rng_seed,
        ddp_rank=0,
        feature_delta=fd,
    )
    _assert_pipelines_equal(p1, p2)


def test_parity_bpb_pipeline_layout():
    """cfg.conditioning has feature_selection (renamed from
    positive_selection) + negative_selection; pinned seed=0,
    ddp_rank=0, feature_delta=None. Mirrors steering/eval/bpb_pipeline.py."""
    cond_cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": None,
    })
    stats = _stats()

    fs_d = OmegaConf.to_container(cond_cfg.feature_selection, resolve=True)
    fs = parse_feature_selection(fs_d)  # ty: ignore[invalid-argument-type]
    p1 = build_feature_pipeline(
        feature_selection=fs,
        negative_selection=None,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        seed=0,
        ddp_rank=0,
        feature_delta=None,
    )
    p2 = build_feature_pipeline_from_cfg_block(
        cond_cfg,
        per_feature_stats=stats,
        n_sae_features=N,
        num_buckets=B,
        seed=0,
        ddp_rank=0,
        feature_delta=None,
    )
    _assert_pipelines_equal(p1, p2)
