from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from errors import FeatureClustersError, FeatureMaskError
from scripts.completion_tui.data_loader import (
    TUIFeatureSamplingConfig,
    clustered_sample_segment_features,
)
from scripts.completion_tui.state import display_to_internal
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
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
from steering.conditional.thresholding import ThresholdType


def _consolidation(
    clusters: tuple[tuple[int, ...], ...] = ((2, 3, 5), (7, 9)),
    *,
    n_sae_features: int = 16,
    mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=clusters)
    resolved = ResolvedFeatureClusters(
        source_path=Path("clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=n_sae_features
        ),
    )
    return FeatureClusterConsolidation.build(resolved, mask)


def _sf(
    ids: list[int],
    pcts: list[int] | None,
    *,
    mags: list[float] | None = None,
) -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor(ids, dtype=t.int32),
        feature_mags=t.tensor(mags or [0.1] * len(ids), dtype=t.float32),
        feature_percentiles=(None if pcts is None else t.tensor(pcts, dtype=t.uint8)),
        char_start=0,
        char_end=20,
    )


def _cfg(
    consolidation: FeatureClusterConsolidation,
    *,
    mask: FeatureMask | None = None,
    selection: SamplingSelection | ThresholdSelection | None = None,
    negative: NegativeSamplingSelection | None = None,
    effective_buckets: int | None = 4,
) -> TUIFeatureSamplingConfig:
    if selection is None:
        selection = ThresholdSelection(ThresholdType.feature_percentile, 0.0)
    return TUIFeatureSamplingConfig(
        boundaries=t.full((consolidation.n_sae_features, 3), 100.0),
        n_sae_features=consolidation.n_sae_features,
        feature_selection=selection,
        negative_selection=negative,
        percentile_table=t.zeros(consolidation.n_sae_features, 99),
        feature_mask=mask,
        consolidation=consolidation,
        effective_buckets=effective_buckets,
    )


def test_requires_percentiles():
    consolidation = _consolidation()
    with pytest.raises(ValueError, match="requires the percentile overlay"):
        clustered_sample_segment_features(_sf([2], None), _cfg(consolidation), 4, False)


def test_requires_effective_buckets():
    consolidation = _consolidation()
    with pytest.raises(ValueError, match="consolidation.*effective_buckets"):
        clustered_sample_segment_features(
            _sf([2], [200]),
            _cfg(consolidation, effective_buckets=None),
            4,
            False,
        )


def test_mask_filters_before_consolidation():
    mask = FeatureMask(FeatureMaskMode.whitelist, (3, 4))
    consolidation = _consolidation(mask=mask)
    result = clustered_sample_segment_features(
        _sf([2, 3, 4], [250, 100, 180]),
        _cfg(consolidation, mask=mask),
        4,
        False,
    )
    assert [row.feature_id for row in result.features] == [3, 4]


def test_sibling_max_becomes_rep_percentile():
    consolidation = _consolidation()
    result = clustered_sample_segment_features(
        _sf([3, 5], [64, 230]), _cfg(consolidation), 4, False
    )
    assert [row.feature_id for row in result.features] == [2]
    expected_bucket = int(percentile_to_bucket(t.tensor([230 / 255]), 4)[0])
    assert result.features[0].magnitude == expected_bucket


def test_rows_are_unique_ascending_reps():
    consolidation = _consolidation()
    result = clustered_sample_segment_features(
        _sf([9, 5, 2, 7, 3], [30, 40, 50, 60, 70]),
        _cfg(consolidation),
        4,
        False,
    )
    assert [row.feature_id for row in result.features] == [2, 7]


def test_bucket_is_percentile_to_bucket_not_boundaries():
    consolidation = _consolidation()
    result = clustered_sample_segment_features(
        _sf([2], [230], mags=[0.1]), _cfg(consolidation), 4, False
    )
    assert result.features[0].magnitude == 3
    assert result.features[0].magnitude != 0


def test_threshold_feature_percentile_matches_producer_semantics():
    consolidation = _consolidation()
    selection = ThresholdSelection(ThresholdType.feature_percentile, 0.5)
    result = clustered_sample_segment_features(
        _sf([2, 7, 11], [127, 128, 255]),
        _cfg(consolidation, selection=selection),
        4,
        False,
    )
    assert {row.feature_id for row in result.features if row.enabled} == {7, 11}


def test_threshold_segment_topk_matches_producer_semantics():
    consolidation = _consolidation()
    selection = ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2.5)
    result = clustered_sample_segment_features(
        _sf([2, 7, 11, 12], [200, 130, 250, 100]),
        _cfg(consolidation, selection=selection),
        4,
        False,
    )
    assert {row.feature_id for row in result.features if row.enabled} == {2, 11}


def test_raw_magnitude_threshold_raises():
    consolidation = _consolidation()
    selection = ThresholdSelection(ThresholdType.segment_topk, 1.0)
    with pytest.raises(ValueError, match="requires raw feature magnitudes"):
        clustered_sample_segment_features(
            _sf([2], [200]),
            _cfg(consolidation, selection=selection),
            4,
            False,
        )


def test_sampling_selection_enabled_subset():
    consolidation = _consolidation()
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=1.0,
        min_k=1,
        max_k=2,
        mode=SamplingMode.top_k_percentile,
    )
    result = clustered_sample_segment_features(
        _sf([2, 7, 11], [100, 240, 200]),
        _cfg(consolidation, selection=selection),
        4,
        False,
        rng=t.Generator().manual_seed(7),
    )
    assert [row.feature_id for row in result.features if row.enabled] == [7]


def test_all_zero_percentiles_use_uniform_sampling_and_bucket_zero():
    consolidation = _consolidation()
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        min_k=2,
        max_k=3,
        mode=SamplingMode.weighted,
    )
    result = clustered_sample_segment_features(
        _sf([2, 7, 11], [0, 0, 0]),
        _cfg(consolidation, selection=selection),
        4,
        False,
        rng=t.Generator().manual_seed(19),
    )
    enabled_ids = {row.feature_id for row in result.features if row.enabled}
    assert len(enabled_ids) == 2
    assert enabled_ids <= {2, 7, 11}
    assert {row.magnitude for row in result.features} == {0}


def test_negatives_drawn_from_effective_negative_mask():
    consolidation = _consolidation()
    negative = NegativeSamplingSelection(
        distribution=Distribution.geometric, mean=1.0, min_k=4, max_k=5
    )
    result = clustered_sample_segment_features(
        _sf([2], [200]),
        _cfg(
            consolidation,
            negative=negative,
            effective_buckets=3,
        ),
        4,
        True,
        rng=t.Generator().manual_seed(12),
    )
    negatives = [row.feature_id for row in result.features if row.magnitude == -1]
    assert negatives
    assert not ({3, 5, 9} & set(negatives))


def test_sibling_presence_blocks_rep_as_negative():
    consolidation = _consolidation(((2, 3),), n_sae_features=8)
    negative = NegativeSamplingSelection(
        distribution=Distribution.geometric, mean=1.0, min_k=20, max_k=21
    )
    result = clustered_sample_segment_features(
        _sf([3], [200]),
        _cfg(consolidation, negative=negative, effective_buckets=3),
        4,
        True,
        rng=t.Generator().manual_seed(3),
    )
    negatives = [row.feature_id for row in result.features if row.magnitude == -1]
    assert 2 not in negatives


def test_empty_after_mask_zero_rows_zero_negatives():
    mask = FeatureMask(FeatureMaskMode.whitelist, (15,))
    consolidation = _consolidation(mask=mask)
    negative = NegativeSamplingSelection(min_k=3, max_k=4)
    result = clustered_sample_segment_features(
        _sf([2, 3], [100, 200]),
        _cfg(consolidation, mask=mask, negative=negative, effective_buckets=3),
        4,
        True,
    )
    assert result.features == []
    assert result.evidence.fingerprint == frozenset()
    assert result.evidence.evidence == {}


def test_empty_after_consolidation_raises_internal_error(monkeypatch):
    """consolidate_segment cannot legitimately empty a masked non-empty segment.

    Upstream hardened this impossible state from a silent empty result into a
    loud raise (`select_clustered_segment`, commit b5cada0a); the TUI mirror
    keeps the identical fail-loud contract.
    """
    consolidation = _consolidation()

    def _empty(self, feature_ids, percentiles, *aligned):
        return feature_ids[:0].to(t.int64), percentiles[:0]

    monkeypatch.setattr(FeatureClusterConsolidation, "consolidate_segment", _empty)
    negative = NegativeSamplingSelection(min_k=3, max_k=4)
    with pytest.raises(
        FeatureClustersError,
        match="consolidation emptied a segment that still had mask-surviving",
    ):
        clustered_sample_segment_features(
            _sf([2], [100]),
            _cfg(consolidation, negative=negative, effective_buckets=3),
            4,
            True,
        )


def test_evidence_contents_exact():
    consolidation = _consolidation()
    result = clustered_sample_segment_features(
        _sf([2, 3, 9], [51, 153, 204]), _cfg(consolidation), 4, False
    )
    assert result.evidence.fingerprint == frozenset({2, 7})
    assert result.evidence.evidence == {
        2: {2: pytest.approx(51 / 255), 3: pytest.approx(153 / 255)},
        7: {9: pytest.approx(204 / 255)},
    }


def test_evidence_absent_for_singletons():
    consolidation = _consolidation()
    result = clustered_sample_segment_features(
        _sf([11, 12], [100, 200]), _cfg(consolidation), 4, False
    )
    assert result.evidence.fingerprint == frozenset()
    assert result.evidence.evidence == {}


def test_corrupted_selected_nonrep_raises(monkeypatch):
    consolidation = _consolidation()
    selection = SamplingSelection(min_k=1, max_k=2)

    def _corrupt(*args, **kwargs):
        return [(t.tensor([3], dtype=t.int64), t.tensor([0.0]))]

    monkeypatch.setattr("scripts.completion_tui.data_loader.sample_features_batched", _corrupt)
    with pytest.raises(FeatureClustersError, match="consolidated-away"):
        clustered_sample_segment_features(
            _sf([2, 3], [100, 200]),
            _cfg(consolidation, selection=selection),
            4,
            False,
            rng=t.Generator().manual_seed(1),
        )


def test_corrupted_selected_masked_id_raises(monkeypatch):
    mask = FeatureMask(FeatureMaskMode.blacklist, (15,))
    consolidation = _consolidation(mask=mask)
    selection = SamplingSelection(min_k=1, max_k=2)

    def _corrupt(*args, **kwargs):
        return [(t.tensor([15], dtype=t.int64), t.tensor([0.0]))]

    monkeypatch.setattr("scripts.completion_tui.data_loader.sample_features_batched", _corrupt)
    with pytest.raises(
        FeatureMaskError,
        match="^feature_mask violated: mode=blacklist produced disallowed ids$",
    ):
        clustered_sample_segment_features(
            _sf([2, 3], [100, 200]),
            _cfg(consolidation, mask=mask, selection=selection),
            4,
            False,
            rng=t.Generator().manual_seed(1),
        )


def test_corrupted_negative_masked_id_raises(monkeypatch):
    mask = FeatureMask(FeatureMaskMode.blacklist, (15,))
    consolidation = _consolidation(mask=mask)
    negative = NegativeSamplingSelection(min_k=1, max_k=2)

    def _corrupt(*args, **kwargs):
        return [t.tensor([15], dtype=t.int64)]

    monkeypatch.setattr(
        "scripts.completion_tui.data_loader.sample_inactive_features_batched",
        _corrupt,
    )
    with pytest.raises(
        FeatureMaskError,
        match="^feature_mask violated: mode=blacklist produced disallowed ids$",
    ):
        clustered_sample_segment_features(
            _sf([2], [200]),
            _cfg(
                consolidation,
                mask=mask,
                negative=negative,
                effective_buckets=3,
            ),
            4,
            True,
            rng=t.Generator().manual_seed(1),
        )


def test_display_magnitudes_round_trip_display_to_internal():
    consolidation = _consolidation()
    result = clustered_sample_segment_features(
        _sf([2, 7, 11], [0, 127, 255]),
        _cfg(consolidation, effective_buckets=3),
        4,
        True,
    )
    assert [
        display_to_internal(row.magnitude, 4, True) for row in result.features
    ] == percentile_to_bucket(t.tensor([0.0, 127 / 255, 1.0]), 3).tolist()
