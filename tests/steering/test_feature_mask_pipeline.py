"""TDD tests for FeatureMask threading through build_feature_pipeline_from_cfg_block.

Reuses the known-good ``_make_per_feature_stats`` / sampling-cfg fixture from
``test_build_feature_pipeline_from_cfg_block.py``, which is verified by 16
passing tests.  The sampling path (geometric) does NOT call
``extract_threshold_percentile``, so it is safe to use without a full stats
percentile dict with p95/p98 keys.
"""

from __future__ import annotations

from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.stats_types import PerFeatureStat

N_SAE_FEATURES = 64
NUM_BUCKETS = 4


def _make_per_feature_stats(n: int = N_SAE_FEATURES) -> dict[str, PerFeatureStat]:
    """Build a tiny stats dict with deterministic percentiles.

    Copied verbatim from test_build_feature_pipeline_from_cfg_block.py —
    this is the known-good fixture used by all 16 passing tests there.
    """
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


def test_pipeline_carries_mask_and_negbundle_mask():
    """FeatureMask threaded: present in FeaturePipelineConfig AND NegativeBundle."""
    cfg = {
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,
        },
        "negative_selection": {"distribution": "geometric", "mean": 2.0, "std": 2.0},
        "feature_mask": {"mode": "blacklist", "ids": [1, 2]},
    }
    p = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert isinstance(p.feature_mask, FeatureMask)
    assert p.feature_mask.ids == (1, 2)
    assert p.negative_bundle is not None
    assert p.negative_bundle.feature_mask is p.feature_mask


def test_pipeline_no_mask_is_none():
    """When feature_mask is absent from cfg, pipeline.feature_mask is None."""
    cfg = {
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,
        },
    }
    p = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    assert p.feature_mask is None
