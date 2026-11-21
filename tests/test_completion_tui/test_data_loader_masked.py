from __future__ import annotations

import torch as t

from scripts.completion_tui.data_loader import (
    TUIFeatureSamplingConfig,
    sample_segment_features,
    segment_features_to_segment,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType


def _sf(ids: list[int], mags: list[float]) -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor(ids, dtype=t.int32),
        feature_mags=t.tensor(mags, dtype=t.float32),
        char_start=0,
        char_end=10,
        feature_percentiles=t.tensor([min(255, int(mag * 50)) for mag in mags], dtype=t.uint8),
    )


def _boundaries(n: int = 12) -> t.Tensor:
    return t.tensor([[0.5, 1.5, 2.5]] * n, dtype=t.float64)


def _cfg(
    *,
    mask: FeatureMask | None,
    selection: SamplingSelection | ThresholdSelection | None = None,
    negative: NegativeSamplingSelection | None = None,
) -> TUIFeatureSamplingConfig:
    return TUIFeatureSamplingConfig(
        boundaries=_boundaries(),
        n_sae_features=12,
        feature_selection=selection,
        negative_selection=negative,
        percentile_table=t.zeros(12, 99),
        precomputed_thresholds=t.ones(12, dtype=t.float64),
        feature_mask=mask,
    )


def test_mask_none_calls_no_mask_methods(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("mask method called on None path")

    monkeypatch.setattr(FeatureMask, "apply_to_positive", _forbidden)
    rows = sample_segment_features(_sf([1, 2], [1.0, 2.0]), _cfg(mask=None), 4, False)
    assert [row.feature_id for row in rows] == [1, 2]


def test_whitelist_filters_rows():
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3))
    rows = sample_segment_features(
        _sf([1, 2, 3], [1.0, 2.0, 3.0]),
        _cfg(mask=mask),
        4,
        False,
        feature_mask=mask,
    )
    assert [row.feature_id for row in rows] == [1, 3]


def test_blacklist_filters_rows():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    rows = sample_segment_features(
        _sf([1, 2, 3], [1.0, 2.0, 3.0]),
        _cfg(mask=mask),
        4,
        False,
        feature_mask=mask,
    )
    assert [row.feature_id for row in rows] == [1, 3]


def test_threshold_applies_after_mask():
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    seg = segment_features_to_segment(
        _sf([1, 2, 3], [1.0, 2.0, 100.0]),
        color_idx=0,
        boundaries=_boundaries(),
        num_buckets=4,
        has_negatives=False,
        threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1),
        feature_mask=mask,
    )
    assert [row.feature_id for row in seg.features] == [2]


def test_negatives_respect_mask():
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3, 5, 7))
    negative = NegativeSamplingSelection(
        distribution=Distribution.geometric, mean=1.0, min_k=3, max_k=4
    )
    rows = sample_segment_features(
        _sf([1], [1.0]),
        _cfg(mask=mask, negative=negative),
        4,
        True,
        feature_mask=mask,
    )
    negatives = {row.feature_id for row in rows if row.magnitude == -1}
    assert negatives
    assert negatives <= {3, 5, 7}


def test_sampling_selection_over_masked_pool():
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3))
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=1.0,
        min_k=1,
        max_k=2,
        mode=SamplingMode.top_k_percentile,
    )
    rows = sample_segment_features(
        _sf([1, 2, 3], [1.0, 100.0, 3.0]),
        _cfg(mask=mask, selection=selection),
        4,
        False,
        feature_mask=mask,
    )
    assert [row.feature_id for row in rows] == [1, 3]
    assert [row.feature_id for row in rows if row.enabled] == [3]


def test_segment_features_to_segment_masked():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    seg = segment_features_to_segment(
        _sf([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0]),
        color_idx=0,
        boundaries=_boundaries(),
        num_buckets=4,
        has_negatives=False,
        feature_mask=mask,
    )
    assert [row.feature_id for row in seg.features] == [1, 3]
