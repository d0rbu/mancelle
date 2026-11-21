from __future__ import annotations

from dataclasses import replace
from operator import setitem
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts.completion_tui.cluster_view import ClusterView
from scripts.completion_tui.data_loader import _composite_representatives
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode


def _consolidation(
    clusters: tuple[tuple[int, ...], ...],
    *,
    n_sae_features: int,
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


def test_fully_allowed_cluster_is_composite():
    view = ClusterView.build(_consolidation(((2, 5, 9),), n_sae_features=12))
    assert view.rep_to_members == {2: (2, 5, 9)}
    assert view.is_composite(2)


def test_mask_collapsed_to_one_survivor_not_composite():
    mask = FeatureMask(FeatureMaskMode.whitelist, (5,))
    view = ClusterView.build(_consolidation(((2, 5, 9),), n_sae_features=12, mask=mask))
    assert dict(view.rep_to_members) == {}
    assert dict(view.member_to_rep) == {}


def test_fully_masked_cluster_absent():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 5, 9))
    view = ClusterView.build(_consolidation(((2, 5, 9),), n_sae_features=12, mask=mask))
    assert not view.is_composite(2)
    assert dict(view.rep_to_members) == {}


def test_identity_after_mask_empty_view():
    consolidation = _consolidation(
        ((2, 5), (7, 9)),
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.whitelist, (2, 7)),
    )
    assert consolidation.is_identity_after_mask
    assert ClusterView.build(consolidation).rep_to_members == {}


def test_rep_is_min_allowed_member():
    view = ClusterView.build(
        _consolidation(
            ((2, 5, 9),),
            n_sae_features=12,
            mask=FeatureMask(FeatureMaskMode.blacklist, (2,)),
        )
    )
    assert view.rep_to_members == {5: (5, 9)}


def test_member_to_rep_includes_rep_itself():
    view = ClusterView.build(_consolidation(((2, 5, 9),), n_sae_features=12))
    assert view.member_to_rep == {2: 2, 5: 2, 9: 2}


def test_n_members_matches_allowed_count():
    view = ClusterView.build(
        _consolidation(
            ((2, 5, 9, 11),),
            n_sae_features=12,
            mask=FeatureMask(FeatureMaskMode.blacklist, (5,)),
        )
    )
    assert view.n_members(2) == 3


def test_n_members_raises_for_singleton():
    view = ClusterView.build(
        _consolidation(
            ((2, 5),),
            n_sae_features=8,
            mask=FeatureMask(FeatureMaskMode.whitelist, (2,)),
        )
    )
    with pytest.raises(KeyError):
        view.n_members(2)


def test_is_composite_false_for_nonclustered_id():
    view = ClusterView.build(_consolidation(((2, 5),), n_sae_features=8))
    assert not view.is_composite(7)


def test_view_mappings_are_immutable():
    view = ClusterView.build(_consolidation(((2, 5, 9),), n_sae_features=12))
    with pytest.raises(TypeError):
        setitem(cast(Any, view.rep_to_members), 2, (2, 5))
    with pytest.raises(TypeError):
        setitem(cast(Any, view.member_to_rep), 5, 5)


def test_inconsistent_member_rep_mapping_raises_value_error():
    consolidation = replace(
        _consolidation(((2, 5),), n_sae_features=8),
        rep_of=t.arange(8, dtype=t.int64),
    )
    with pytest.raises(
        ValueError,
        match="allowed member 5.*representative 5.*expected 2",
    ):
        ClusterView.build(consolidation)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_view_matches_plain_python_oracle(data: st.DataObject):
    n_sae_features = data.draw(st.integers(min_value=4, max_value=64), label="n")
    members = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae_features - 1),
            min_size=2,
            max_size=n_sae_features,
            unique=True,
        ),
        label="members",
    )
    cluster_size = data.draw(st.integers(min_value=2, max_value=5), label="size")
    clusters = tuple(
        tuple(sorted(members[start : start + cluster_size]))
        for start in range(0, len(members), cluster_size)
        if len(members[start : start + cluster_size]) >= 2
    )
    if not clusters:
        clusters = (tuple(sorted(members[:2])),)

    mode = data.draw(st.sampled_from(("none", "blacklist", "whitelist")), label="mode")
    if mode == "none":
        mask = None
        allowed = set(range(n_sae_features))
    else:
        mask_ids = data.draw(
            st.sets(
                st.integers(min_value=0, max_value=n_sae_features - 1),
                min_size=1,
            ),
            label="mask_ids",
        )
        mask = FeatureMask(FeatureMaskMode(mode), tuple(mask_ids))
        allowed = set(range(n_sae_features)) - mask_ids if mode == "blacklist" else mask_ids

    consolidation = _consolidation(
        clusters,
        n_sae_features=n_sae_features,
        mask=mask,
    )
    view = ClusterView.build(consolidation)
    expected_rep_to_members: dict[int, tuple[int, ...]] = {}
    expected_member_to_rep: dict[int, int] = {}
    for cluster in sorted(tuple(sorted(cluster)) for cluster in clusters):
        allowed_members = tuple(fid for fid in cluster if fid in allowed)
        if len(allowed_members) < 2:
            continue
        rep = allowed_members[0]
        expected_rep_to_members[rep] = allowed_members
        expected_member_to_rep.update(dict.fromkeys(allowed_members, rep))

    assert dict(view.rep_to_members) == expected_rep_to_members
    assert dict(view.member_to_rep) == expected_member_to_rep
    assert _composite_representatives(consolidation) == frozenset(view.rep_to_members)
