from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from errors import FeatureClustersError, RemovedFeatureError
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
    resolve_feature_clusters_from_checkpoint_meta,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode


def _resolved(
    clusters: tuple[tuple[int, ...], ...] = ((1, 3), (5, 8, 13)),
    *,
    n_sae_features: int = 16,
    source_path: str = "clusters.json",
) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=clusters)
    descriptor = FeatureClusterDescriptor.from_definition(definition, n_sae_features=n_sae_features)
    return ResolvedFeatureClusters(
        source_path=Path(source_path),
        definition=definition,
        descriptor=descriptor,
    )


@st.composite
def _cluster_cases(draw: st.DrawFn) -> tuple[int, tuple[tuple[int, ...], ...]]:
    n_sae_features = draw(st.integers(min_value=2, max_value=64))
    members = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae_features - 1),
            min_size=2,
            max_size=n_sae_features,
            unique=True,
        )
    )
    if len(members) % 2:
        members = members[:-1]
    if not members:
        members = [0, 1]
    clusters = tuple(
        tuple(sorted(members[offset : offset + 2])) for offset in range(0, len(members), 2)
    )
    return n_sae_features, clusters


def test_all_absent_returns_none():
    assert (
        resolve_feature_clusters_from_checkpoint_meta(
            {}, surface="test checkpoint", source_path="meta.json"
        )
        is None
    )


@given(case=_cluster_cases())
def test_round_trip_from_checkpoint_metadata(
    case: tuple[int, tuple[tuple[int, ...], ...]],
):
    n_sae_features, clusters = case
    original = _resolved(clusters, n_sae_features=n_sae_features)
    restored = resolve_feature_clusters_from_checkpoint_meta(
        feature_clusters_checkpoint_metadata(original),
        surface="test checkpoint",
        source_path="checkpoint/meta.json",
    )
    assert restored is not None
    assert restored.descriptor == original.descriptor
    assert restored.definition == original.definition


@given(case=_cluster_cases(), data=st.data())
def test_consolidated_away_exactly_matches_non_representatives(
    case: tuple[int, tuple[tuple[int, ...], ...]],
    data: st.DataObject,
):
    n_sae_features, clusters = case
    mode = data.draw(st.sampled_from((None, "blacklist", "whitelist")))
    if mode is None:
        mask = None
    else:
        mask_ids = data.draw(
            st.sets(
                st.integers(min_value=0, max_value=n_sae_features - 1),
                min_size=1,
                max_size=n_sae_features,
            )
        )
        mask = FeatureMask(FeatureMaskMode(mode), tuple(mask_ids))
    consolidation = FeatureClusterConsolidation.build(
        _resolved(clusters, n_sae_features=n_sae_features),
        mask,
    )

    for feature_id in range(n_sae_features):
        is_away = bool(consolidation.consolidated_away[feature_id])
        is_non_representative = int(consolidation.rep_of[feature_id].item()) != feature_id
        assert is_away == is_non_representative


def test_descriptor_only_raises():
    metadata = feature_clusters_checkpoint_metadata(_resolved())
    metadata.pop("feature_clusters_canonical")
    with pytest.raises(FeatureClustersError) as exc_info:
        resolve_feature_clusters_from_checkpoint_meta(
            metadata, surface="test surface", source_path="meta.json"
        )
    assert str(exc_info.value) == (
        "test surface has incomplete static feature-cluster metadata "
        "(descriptor/canonical must both be present); start fresh or use a "
        "checkpoint written by the clustered preselection path"
    )
    assert exc_info.value.__cause__ is None


def test_canonical_only_raises():
    metadata = feature_clusters_checkpoint_metadata(_resolved())
    metadata.pop("feature_clusters_descriptor")
    with pytest.raises(FeatureClustersError) as exc_info:
        resolve_feature_clusters_from_checkpoint_meta(
            metadata, surface="test surface", source_path="meta.json"
        )
    assert str(exc_info.value) == (
        "test surface has incomplete static feature-cluster metadata "
        "(descriptor/canonical must both be present); start fresh or use a "
        "checkpoint written by the clustered preselection path"
    )
    assert exc_info.value.__cause__ is None


def test_user_config_only_raises():
    with pytest.raises(FeatureClustersError) as exc_info:
        resolve_feature_clusters_from_checkpoint_meta(
            {"user_config": {"feature_clusters": {"path": "missing.json"}}},
            surface="test surface",
            source_path="meta.json",
        )
    assert str(exc_info.value) == (
        "test surface has incomplete static feature-cluster metadata "
        "(descriptor/canonical must both be present); start fresh or use a "
        "checkpoint written by the clustered preselection path"
    )
    assert exc_info.value.__cause__ is None


def test_legacy_metadata_raises_removed_feature_error():
    with pytest.raises(RemovedFeatureError) as exc_info:
        resolve_feature_clusters_from_checkpoint_meta(
            {"feature_clustering_canonical": {"clusters": []}},
            surface="legacy checkpoint",
            source_path="meta.json",
        )
    assert str(exc_info.value) == (
        "legacy checkpoint: the feature clustering pipeline was removed. Remove "
        "`feature_clustering` config, regenerate unclustered artifacts, or use "
        "an unclustered checkpoint."
    )
    assert exc_info.value.__cause__ is None


def test_invalid_canonical_wrapped_with_surface():
    metadata = feature_clusters_checkpoint_metadata(_resolved())
    metadata["feature_clusters_canonical"] = {
        "schema_version": 1,
        "clusters": [[1, 99]],
    }
    with pytest.raises(FeatureClustersError) as exc_info:
        resolve_feature_clusters_from_checkpoint_meta(
            metadata, surface="test surface", source_path="meta.json"
        )
    assert str(exc_info.value) == (
        "test surface has invalid static feature-cluster metadata: feature "
        "cluster ID 99 is out of range [0, 16)"
    )
    assert isinstance(exc_info.value.__cause__, FeatureClustersError)
    assert str(exc_info.value.__cause__) == ("feature cluster ID 99 is out of range [0, 16)")


def test_descriptor_definition_mismatch_raises():
    metadata = feature_clusters_checkpoint_metadata(_resolved())
    descriptor = dict(metadata["feature_clusters_descriptor"])
    descriptor["content_sha256"] = "0" * 64
    metadata["feature_clusters_descriptor"] = descriptor
    with pytest.raises(FeatureClustersError, match="descriptor does not match"):
        resolve_feature_clusters_from_checkpoint_meta(
            metadata, surface="test surface", source_path="meta.json"
        )


def test_non_mapping_meta_returns_none_when_inactive():
    assert (
        resolve_feature_clusters_from_checkpoint_meta(
            ["not", "metadata"], surface="test surface", source_path="meta.json"
        )
        is None
    )


def test_non_mapping_user_config_is_inactive():
    assert (
        resolve_feature_clusters_from_checkpoint_meta(
            {"user_config": ["not", "a", "mapping"]},
            surface="test surface",
            source_path="meta.json",
        )
        is None
    )


def test_source_path_is_provenance_only():
    metadata = feature_clusters_checkpoint_metadata(_resolved())
    first = resolve_feature_clusters_from_checkpoint_meta(
        metadata, surface="test surface", source_path="host-a/meta.json"
    )
    second = resolve_feature_clusters_from_checkpoint_meta(
        metadata, surface="test surface", source_path="host-b/meta.json"
    )
    assert first is not None and second is not None
    assert first.source_path != second.source_path
    assert first.descriptor == second.descriptor
    assert first.definition == second.definition
