"""Unit tests for coefficient_grids (Task 10)."""

from __future__ import annotations

from steering.eval.axbench_base.coefficient_grids import (
    active_bucket_grid,
    diffmean_scalar_grid,
    single_condition_grid,
)
from steering.eval.axbench_base.types import TargetModelMetadata
from tests.axbench_base.unit.test_config import make_valid_config


def _encdec_meta(
    *,
    num_buckets: int | None = 8,
    negative_selection: dict | None = None,
) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=num_buckets,
        n_sae_features=16384,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _decoder_only_meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=None,
        n_sae_features=None,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def test_diffmean_scalar_grid_is_fixed_10_values():
    cfg = make_valid_config()
    grid = diffmean_scalar_grid(_encdec_meta(), cfg)
    assert grid == [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_diffmean_scalar_grid_independent_of_model():
    """The scalar grid is the same regardless of target model metadata."""
    cfg = make_valid_config()
    enc = diffmean_scalar_grid(_encdec_meta(num_buckets=1), cfg)
    dec = diffmean_scalar_grid(_decoder_only_meta(), cfg)
    enc_d13 = diffmean_scalar_grid(
        _encdec_meta(num_buckets=8, negative_selection={"strategy": "global"}),
        cfg,
    )
    assert enc == dec == enc_d13


def test_diffmean_scalar_grid_returns_new_list_each_call():
    """Caller-safe: mutating the returned list must not affect subsequent calls."""
    cfg = make_valid_config()
    g1 = diffmean_scalar_grid(_encdec_meta(), cfg)
    g1.append(999.0)
    g2 = diffmean_scalar_grid(_encdec_meta(), cfg)
    assert 999.0 not in g2


def test_single_condition_grid_returns_none_list():
    cfg = make_valid_config()
    grid = single_condition_grid(_encdec_meta(), cfg)
    assert grid == [None]


def test_single_condition_grid_independent_of_model():
    cfg = make_valid_config()
    assert single_condition_grid(_decoder_only_meta(), cfg) == [None]
    assert single_condition_grid(_encdec_meta(num_buckets=8), cfg) == [None]


def test_active_bucket_grid_decoder_only_defaults_to_zero():
    cfg = make_valid_config()
    assert active_bucket_grid(_decoder_only_meta(), cfg) == [0]


def test_active_bucket_grid_d12():
    """d12-style: num_buckets=1, negative_selection=None -> [0]."""
    cfg = make_valid_config()
    meta = _encdec_meta(num_buckets=1, negative_selection=None)
    assert active_bucket_grid(meta, cfg) == [0]


def test_active_bucket_grid_d13_no_negative_sampling():
    """num_buckets=8, negative_selection=None -> [0..7] (8 entries)."""
    cfg = make_valid_config()
    meta = _encdec_meta(num_buckets=8, negative_selection=None)
    assert active_bucket_grid(meta, cfg) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_active_bucket_grid_d13_with_negative_sampling():
    """num_buckets=8, negative_selection=<cfg> -> [0..6] (7 entries, reserves last for inactive)."""
    cfg = make_valid_config()
    meta = _encdec_meta(
        num_buckets=8,
        negative_selection={"strategy": "global", "n": 216},
    )
    assert active_bucket_grid(meta, cfg) == [0, 1, 2, 3, 4, 5, 6]


def test_active_bucket_grid_num_buckets_equals_negative_reservation():
    """num_buckets=1 with negative_selection -> n_active = 0 -> []."""
    cfg = make_valid_config()
    meta = _encdec_meta(
        num_buckets=1,
        negative_selection={"strategy": "global"},
    )
    assert active_bucket_grid(meta, cfg) == []
