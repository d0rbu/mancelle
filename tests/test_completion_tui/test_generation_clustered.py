from __future__ import annotations

from pathlib import Path

import pytest

from errors import FeatureClustersError, FeatureMaskError
from scripts.completion_tui.generation import _segments_to_features
from scripts.completion_tui.state import FeatureCondition, Segment
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode


def _consolidation(
    mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((1, 3, 5), (8, 9)))
    resolved = ResolvedFeatureClusters(
        source_path=Path("clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=12),
    )
    return FeatureClusterConsolidation.build(resolved, mask)


def _segment(feature_ids: list[int], magnitudes: list[int] | None = None) -> Segment:
    return Segment(
        segment_idx=7,
        char_start=0,
        char_end=10,
        color_idx=0,
        features=[
            FeatureCondition(feature_id=feature_id, magnitude=magnitude)
            for feature_id, magnitude in zip(
                feature_ids,
                magnitudes or [0] * len(feature_ids),
                strict=True,
            )
        ],
    )


def test_masked_id_raises_before_model():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    with pytest.raises(FeatureMaskError, match="feature_mask violated"):
        _segments_to_features(
            [_segment([2])],
            4,
            False,
            feature_mask=mask,
        )


def test_nonrep_id_raises_before_model():
    consolidation = _consolidation()
    # In a valid consolidation, non-representative and consolidated-away are
    # equivalent; this and the next test intentionally exercise the same guard.
    with pytest.raises(FeatureClustersError, match="segment 7.*feature ID 3"):
        _segments_to_features(
            [_segment([3])],
            4,
            False,
            consolidation=consolidation,
        )


def test_consolidated_away_id_raises():
    consolidation = _consolidation()
    # See the equivalence note in test_nonrep_id_raises_before_model.
    with pytest.raises(FeatureClustersError, match="consolidated-away"):
        _segments_to_features(
            [_segment([5])],
            4,
            False,
            consolidation=consolidation,
        )


def test_clean_rep_only_passes_exact_segmentfeatures():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    consolidation = _consolidation(mask)
    result = _segments_to_features(
        [_segment([1, 8, 11], [0, 2, 3])],
        4,
        False,
        feature_mask=mask,
        consolidation=consolidation,
    )
    assert len(result) == 1
    assert result[0].feature_ids.tolist() == [1, 8, 11]
    assert result[0].feature_mags.tolist() == [0, 2, 3]
    assert result[0].segment_idx == 7


def test_none_none_path_unchanged(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("contract assertion called on None path")

    monkeypatch.setattr(FeatureMask, "assert_no_masked", _forbidden)
    monkeypatch.setattr(FeatureClusterConsolidation, "assert_representatives_only", _forbidden)
    result = _segments_to_features([_segment([3], [2])], 4, False)
    assert result[0].feature_ids.tolist() == [3]
    assert result[0].feature_mags.tolist() == [2]


def test_lazy_manager_receives_contract():
    source = (Path(__file__).parents[2] / "scripts" / "completion_tui" / "__main__.py").read_text(
        encoding="utf-8"
    )
    assert source.count("feature_mask=contract.feature_mask") >= 2
    assert source.count("consolidation=contract.consolidation") >= 2
