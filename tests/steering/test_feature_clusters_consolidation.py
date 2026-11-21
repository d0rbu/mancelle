from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from errors import FeatureClustersError
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from tests.reference_feature_clustering import HistoricalFeatureConsolidation


def _resolved(
    tmp_path: Path,
    clusters: list[list[int]],
    *,
    n_sae_features: int,
    name: str = "clusters.json",
) -> ResolvedFeatureClusters:
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": clusters}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        n_sae_features,
    )
    assert resolved is not None
    return resolved


def _state(
    tmp_path: Path,
    clusters: list[list[int]],
    *,
    n_sae_features: int,
    mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    return FeatureClusterConsolidation.build(
        _resolved(tmp_path, clusters, n_sae_features=n_sae_features),
        mask,
    )


def _oracle_state(
    clusters: tuple[tuple[int, ...], ...],
    n_sae_features: int,
    allowed: list[bool],
) -> tuple[list[int], list[bool], list[bool]]:
    rep_of = list(range(n_sae_features))
    away = [False] * n_sae_features
    for cluster in clusters:
        survivors = [feature_id for feature_id in cluster if allowed[feature_id]]
        if not survivors:
            continue
        representative = min(survivors)
        for feature_id in survivors:
            rep_of[feature_id] = representative
        for feature_id in survivors:
            if feature_id != representative:
                away[feature_id] = True
    effective = [keep and not removed for keep, removed in zip(allowed, away, strict=True)]
    return rep_of, away, effective


def _oracle_reduce(
    state: FeatureClusterConsolidation,
    feature_ids: list[int],
    channels: list[list[float]],
) -> tuple[list[int], list[list[float]]]:
    grouped: dict[int, list[list[float]]] = {}
    for row_idx, feature_id in enumerate(feature_ids):
        representative = int(state.rep_of[feature_id].item())
        grouped.setdefault(representative, []).append(
            [channel[row_idx] for channel in channels]
        )
    reps = sorted(grouped)
    reduced = [[] for _ in channels]
    for representative in reps:
        rows = grouped[representative]
        for channel_idx in range(len(channels)):
            reduced[channel_idx].append(max(row[channel_idx] for row in rows))
    return reps, reduced


def test_no_mask_uses_lowest_original_member_as_representative(tmp_path: Path):
    state = _state(
        tmp_path,
        [[40, 443, 9_405], [7, 2]],
        n_sae_features=10_000,
    )
    assert state.rep_of[40].item() == 40
    assert state.rep_of[443].item() == 40
    assert state.rep_of[9_405].item() == 40
    assert state.rep_of[2].item() == 2
    assert state.rep_of[7].item() == 2
    assert state.rep_of[3].item() == 3
    assert t.nonzero(state.consolidated_away).squeeze(1).tolist() == [7, 443, 9_405]
    assert state.is_identity_after_mask is False
    assert bool(state.cluster_allowed_universe[40])
    assert not bool(state.cluster_allowed_universe[443])


def test_blacklist_removes_original_minimum_and_shifts_representative(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=mask,
    )
    assert state.rep_of[2].item() == 2  # Masked IDs remain identity, never active.
    assert state.rep_of[7].item() == 7
    assert state.rep_of[11].item() == 7
    assert not bool(state.consolidated_away[2])
    assert not bool(state.consolidated_away[7])
    assert bool(state.consolidated_away[11])
    assert not bool(state.allowed_universe[2])
    assert bool(state.cluster_allowed_universe[7])


def test_whitelist_derives_same_shift_from_allowed_set(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 7, 11))
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=mask,
    )
    assert state.rep_of[[2, 7, 11]].tolist() == [2, 7, 7]
    assert state.consolidated_away[[2, 7, 11]].tolist() == [False, False, True]
    assert state.cluster_allowed_universe.nonzero().squeeze(1).tolist() == [0, 7]


def test_fully_masked_cluster_is_inert(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 7, 11))
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=mask,
    )
    assert state.rep_of[[2, 7, 11]].tolist() == [2, 7, 11]
    assert not bool(state.consolidated_away.any())
    assert state.is_identity_after_mask is True
    assert t.equal(state.cluster_allowed_universe, state.allowed_universe)


def test_one_survivor_cluster_is_inert(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (4, 8))
    state = _state(
        tmp_path,
        [[2, 4, 7]],
        n_sae_features=9,
        mask=mask,
    )
    assert state.rep_of[[2, 4, 7]].tolist() == [2, 4, 7]
    assert not bool(state.consolidated_away.any())
    assert state.is_identity_after_mask is True
    assert state.cluster_allowed_universe.nonzero().squeeze(1).tolist() == [4, 8]


def test_noncluster_features_are_identity_and_remain_allowed(tmp_path: Path):
    state = _state(tmp_path, [[1, 4]], n_sae_features=8)
    for feature_id in [0, 2, 3, 5, 6, 7]:
        assert state.rep_of[feature_id].item() == feature_id
        assert not bool(state.consolidated_away[feature_id])
        assert bool(state.cluster_allowed_universe[feature_id])


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_mask_derived_state_matches_simple_python_oracle(data, tmp_path: Path):
    n = data.draw(st.integers(min_value=2, max_value=128), label="n")
    cluster_members = sorted(
        data.draw(
            st.sets(st.integers(min_value=0, max_value=n - 1), min_size=2),
            label="cluster_members",
        )
    )
    mode = data.draw(st.sampled_from(["none", "blacklist", "whitelist"]), label="mode")
    if mode == "none":
        mask = None
        allowed = [True] * n
    else:
        ids = sorted(
            data.draw(
                st.sets(st.integers(min_value=0, max_value=n - 1), min_size=1),
                label="mask_ids",
            )
        )
        mask = FeatureMask(FeatureMaskMode(mode), tuple(ids))
        members = set(ids)
        allowed = [
            (feature_id not in members if mode == "blacklist" else feature_id in members)
            for feature_id in range(n)
        ]
    state = _state(
        tmp_path,
        [cluster_members],
        n_sae_features=n,
        mask=mask,
    )
    expected_rep, expected_away, expected_allowed = _oracle_state(
        state.resolved.definition.clusters,
        n,
        allowed,
    )
    assert state.rep_of.tolist() == expected_rep
    assert state.consolidated_away.tolist() == expected_away
    assert state.is_identity_after_mask is (not any(expected_away))
    assert state.cluster_allowed_universe.tolist() == expected_allowed


def test_historical_valid_subset_golden_is_exact(tmp_path: Path):
    """Frozen output of 49fb3cf's mask-first representative semantics."""
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 8))
    state = _state(
        tmp_path,
        [[1, 3, 9], [2, 8], [4, 5, 7]],
        n_sae_features=10,
        mask=mask,
    )
    assert state.rep_of.tolist() == [0, 1, 2, 3, 4, 4, 6, 4, 8, 3]
    assert state.consolidated_away.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert state.cluster_allowed_universe.tolist() == [
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
    ]


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_shared_primitive_is_bit_exact_to_removed_reference(data, tmp_path: Path):
    """Exact differential gate against ``49fb3cf`` over disjoint clusters."""
    n = data.draw(st.integers(min_value=2, max_value=96), label="n")
    assignments = [0, 0] + data.draw(
        st.lists(
            st.integers(min_value=-1, max_value=5),
            min_size=n - 2,
            max_size=n - 2,
        ),
        label="cluster_assignments",
    )
    clusters = [
        [feature_id for feature_id, assignment in enumerate(assignments) if assignment == group]
        for group in range(6)
    ]
    clusters = [cluster for cluster in clusters if len(cluster) >= 2]

    mode = data.draw(st.sampled_from(["none", "blacklist", "whitelist"]), label="mode")
    if mode == "none":
        mask = None
    else:
        mask_ids = sorted(
            data.draw(
                st.sets(
                    st.integers(min_value=0, max_value=n - 1),
                    min_size=1,
                ),
                label="mask_ids",
            )
        )
        mask = FeatureMask(FeatureMaskMode(mode), tuple(mask_ids))

    current = _state(tmp_path, clusters, n_sae_features=n, mask=mask)
    historical = HistoricalFeatureConsolidation.build(
        current.resolved.definition.clusters,
        mask,
        n,
    )
    assert t.equal(current.rep_of, historical.rep_of)
    assert t.equal(current.consolidated_away, historical.consolidated_away)
    assert current.is_identity_after_mask is (
        not bool(historical.consolidated_away.any())
    )
    assert t.equal(
        current.cluster_allowed_universe,
        historical.augmented_allowed_universe,
    )
    assert current.effective_negative_mask() == historical.effective_negative_mask(mask)

    raw_ids = data.draw(
        st.lists(st.integers(min_value=0, max_value=n - 1), max_size=192),
        label="raw_ids",
    )
    raw_percentiles = data.draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=len(raw_ids),
            max_size=len(raw_ids),
        ),
        label="raw_percentiles",
    )
    ids = t.tensor(raw_ids, dtype=t.int64)
    percentiles = t.tensor(raw_percentiles, dtype=t.float32)
    expected_ids, expected_percentiles = historical.consolidate_segment(
        ids,
        percentiles,
        mask,
    )
    if mask is not None and ids.numel() > 0:
        keep = mask.keep_mask(ids)
        ids = ids[keep]
        percentiles = percentiles[keep]
    actual_ids, actual_percentiles = current.consolidate_segment(ids, percentiles)

    assert t.equal(actual_ids, expected_ids)
    assert t.equal(actual_percentiles, expected_percentiles)


def test_real_width_shared_primitive_is_bit_exact_to_removed_reference(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.blacklist, (40, 32_768))
    current = _state(
        tmp_path,
        [[40, 443, 9_405, 65_535], [2, 32_768, 60_000]],
        n_sae_features=65_536,
        mask=mask,
    )
    historical = HistoricalFeatureConsolidation.build(
        current.resolved.definition.clusters,
        mask,
        65_536,
    )
    raw_ids = t.tensor(
        [65_535, 443, 443, 40, 60_000, 32_768, 2, 10],
        dtype=t.int64,
    )
    raw_percentiles = t.tensor(
        [0.7, 0.3, 0.8, 0.99, 0.4, 0.95, 0.9, 0.2],
        dtype=t.float32,
    )
    expected = historical.consolidate_segment(raw_ids, raw_percentiles, mask)
    keep = mask.keep_mask(raw_ids)
    actual = current.consolidate_segment(raw_ids[keep], raw_percentiles[keep])
    assert all(t.equal(got, want) for got, want in zip(actual, expected, strict=True))


def test_reducer_injects_representative_and_max_reduces_percentile(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    reps, pcts = state.consolidate_segment(
        t.tensor([11, 7], dtype=t.int64),
        t.tensor([0.61, 0.92], dtype=t.float32),
    )
    assert reps.tolist() == [2]
    assert t.equal(pcts, t.tensor([0.92], dtype=t.float32))


def test_reducer_orders_cluster_and_noncluster_representatives_ascending(tmp_path: Path):
    state = _state(tmp_path, [[4, 9], [2, 7]], n_sae_features=12)
    reps, pcts = state.consolidate_segment(
        t.tensor([11, 9, 7, 0, 4, 2], dtype=t.int64),
        t.tensor([0.1, 0.3, 0.8, 0.5, 0.9, 0.2], dtype=t.float32),
    )
    assert reps.tolist() == [0, 2, 4, 11]
    assert t.equal(pcts, t.tensor([0.5, 0.8, 0.9, 0.1]))


def test_reducer_coalesces_duplicate_input_ids_by_max(tmp_path: Path):
    state = _state(tmp_path, [[1, 4]], n_sae_features=6)
    reps, pcts = state.consolidate_segment(
        t.tensor([4, 4, 1, 5]),
        t.tensor([0.1, 0.7, 0.3, 0.2]),
    )
    assert reps.tolist() == [1, 5]
    assert t.equal(pcts, t.tensor([0.7, 0.2]))


def test_percentile_and_generic_scores_reduce_independently(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    reps, pcts, delta_scores, auxiliary_scores = state.consolidate_segment(
        t.tensor([2, 7, 11]),
        t.tensor([0.99, 0.40, 0.50]),
        t.tensor([0.01, 0.90, 0.20]),
        t.tensor([-5.0, -3.0, -8.0]),
    )
    assert reps.tolist() == [2]
    assert pcts.tolist() == pytest.approx([0.99])
    assert delta_scores.tolist() == pytest.approx([0.90])
    # include_self=False is load-bearing: zero must not beat all-negative scores.
    assert auxiliary_scores.tolist() == pytest.approx([-3.0])


def test_raw_magnitude_scale_is_not_part_of_cluster_reduction(tmp_path: Path):
    state = _state(tmp_path, [[1, 2]], n_sae_features=3)
    ids = t.tensor([1, 2])
    pcts = t.tensor([0.7, 0.9])
    huge_scale_mags = t.tensor([1.0e30, 1.0e-30])
    reversed_scale_mags = t.tensor([1.0e-30, 1.0e30])
    out_a = state.consolidate_segment(ids, pcts, huge_scale_mags)
    out_b = state.consolidate_segment(ids, pcts, reversed_scale_mags)
    assert out_a[0].tolist() == out_b[0].tolist() == [1]
    assert out_a[1].tolist() == out_b[1].tolist() == pytest.approx([0.9])
    # The generic channel is independent and may differ; it is never treated as pct.
    assert out_a[2].item() == out_b[2].item() == pytest.approx(1.0e30)


def test_empty_reducer_preserves_id_and_channel_dtypes(tmp_path: Path):
    state = _state(tmp_path, [[1, 2]], n_sae_features=3)
    ids = t.empty(0, dtype=t.uint16)
    pcts = t.empty(0, dtype=t.float16)
    scores = t.empty(0, dtype=t.float64)
    out_ids, out_pcts, out_scores = state.consolidate_segment(ids, pcts, scores)
    assert out_ids.dtype is t.int64
    assert out_pcts.dtype is t.float16
    assert out_scores.dtype is t.float64
    assert out_ids.numel() == out_pcts.numel() == out_scores.numel() == 0


def test_consolidate_segment_validates_feature_ids_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(tmp_path, [[1, 2]], n_sae_features=4)
    original = FeatureClusterConsolidation._validate_feature_ids
    calls = 0

    def counted(
        self: FeatureClusterConsolidation,
        feature_ids: t.Tensor,
        *,
        surface: str,
    ) -> t.Tensor:
        nonlocal calls
        calls += 1
        return original(self, feature_ids, surface=surface)

    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "_validate_feature_ids",
        counted,
    )
    for feature_ids, percentiles in (
        (t.tensor([1, 2], dtype=t.int64), t.tensor([0.2, 0.8])),
        (t.empty(0, dtype=t.uint16), t.empty(0, dtype=t.float16)),
    ):
        before = calls
        state.consolidate_segment(feature_ids, percentiles)
        assert calls == before + 1


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_reducer_matches_simple_python_oracle_with_duplicates(data, tmp_path: Path):
    n = data.draw(st.integers(min_value=2, max_value=64), label="n")
    cluster = sorted(
        data.draw(
            st.sets(st.integers(0, n - 1), min_size=2),
            label="cluster",
        )
    )
    state = _state(tmp_path, [cluster], n_sae_features=n)
    fids = data.draw(
        st.lists(st.integers(0, n - 1), min_size=0, max_size=128),
        label="fids",
    )
    pcts = data.draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=len(fids),
            max_size=len(fids),
        ),
        label="pcts",
    )
    scores = data.draw(
        st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=len(fids),
            max_size=len(fids),
        ),
        label="scores",
    )
    expected_reps, expected_channels = _oracle_reduce(state, fids, [pcts, scores])
    reps, reduced_pcts, reduced_scores = state.consolidate_segment(
        t.tensor(fids, dtype=t.int64),
        t.tensor(pcts, dtype=t.float32),
        t.tensor(scores, dtype=t.float32),
    )
    assert reps.tolist() == expected_reps
    assert reduced_pcts.tolist() == pytest.approx(expected_channels[0])
    assert reduced_scores.tolist() == pytest.approx(expected_channels[1])


@pytest.mark.parametrize(
    "ids",
    [
        t.tensor([-1], dtype=t.int64),
        t.tensor([3], dtype=t.int64),
        t.tensor([-1, 1], dtype=t.int16),
        t.tensor([65_536], dtype=t.int64),
    ],
)
def test_reducer_rejects_out_of_range_before_tensor_indexing(tmp_path: Path, ids: t.Tensor):
    n = 65_536 if int(ids.max().item()) >= 65_536 else 3
    state = _state(tmp_path, [[0, 1]], n_sae_features=n)
    with pytest.raises(FeatureClustersError, match="out of range"):
        state.consolidate_segment(ids, t.ones(len(ids)))


@pytest.mark.parametrize(
    "ids",
    [
        t.tensor([0.0, 1.0]),
        t.tensor([False, True]),
        t.tensor([1 + 0j], dtype=t.complex64),
    ],
)
def test_reducer_rejects_noninteger_id_tensors_without_coercion(tmp_path: Path, ids: t.Tensor):
    state = _state(tmp_path, [[0, 1]], n_sae_features=3)
    with pytest.raises(FeatureClustersError, match="integer dtype"):
        state.consolidate_segment(ids, t.ones(len(ids)))


def test_reducer_rejects_unmasked_blacklisted_input(tmp_path: Path):
    state = _state(
        tmp_path,
        [[1, 2]],
        n_sae_features=4,
        mask=FeatureMask(FeatureMaskMode.blacklist, (2,)),
    )
    with pytest.raises(FeatureClustersError, match="apply the fully resolved"):
        state.consolidate_segment(t.tensor([2]), t.tensor([0.5]))


def test_reducer_rejects_nonwhitelisted_input(tmp_path: Path):
    state = _state(
        tmp_path,
        [[1, 2]],
        n_sae_features=4,
        mask=FeatureMask(FeatureMaskMode.whitelist, (1, 3)),
    )
    with pytest.raises(FeatureClustersError, match="mask-disallowed"):
        state.consolidate_segment(t.tensor([0]), t.tensor([0.5]))


def test_reducer_requires_one_dimensional_ids(tmp_path: Path):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    with pytest.raises(FeatureClustersError, match="one-dimensional"):
        state.consolidate_segment(t.tensor([[0, 1]]), t.tensor([0.1, 0.2]))


@pytest.mark.parametrize(
    "pcts",
    [
        t.tensor([[0.1, 0.2]]),
        t.tensor([0.1]),
        t.tensor([0.1, 0.2, 0.3]),
    ],
)
def test_reducer_requires_aligned_one_dimensional_percentile_channel(
    tmp_path: Path,
    pcts: t.Tensor,
):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    with pytest.raises(FeatureClustersError, match="aligned channel 0"):
        state.consolidate_segment(t.tensor([0, 1]), pcts)


def test_reducer_requires_floating_channels(tmp_path: Path):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    with pytest.raises(FeatureClustersError, match="floating point"):
        state.consolidate_segment(t.tensor([0, 1]), t.tensor([0, 1]))
    with pytest.raises(FeatureClustersError, match="aligned channel 1.*floating point"):
        state.consolidate_segment(
            t.tensor([0, 1]),
            t.tensor([0.1, 0.2]),
            t.tensor([1, 2]),
        )


@pytest.mark.parametrize(
    "bad",
    [
        -0.0001,
        1.0001,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_reducer_rejects_invalid_percentiles(tmp_path: Path, bad: float):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    match = "non-finite" if not math.isfinite(bad) else "normalized"
    with pytest.raises(FeatureClustersError, match=match):
        state.consolidate_segment(t.tensor([0, 1]), t.tensor([0.5, bad]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_reducer_rejects_nonfinite_generic_score(tmp_path: Path, bad: float):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    with pytest.raises(FeatureClustersError, match="channel 1 contains a non-finite"):
        state.consolidate_segment(
            t.tensor([0, 1]),
            t.tensor([0.1, 0.2]),
            t.tensor([0.0, bad]),
        )


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_reducer_rejects_cross_device_aligned_channel(tmp_path: Path):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    with pytest.raises(FeatureClustersError, match="channel 0 is on"):
        state.consolidate_segment(
            t.tensor([0, 1], device="cuda"),
            t.tensor([0.1, 0.2], device="cpu"),
        )


def test_project_presence_maps_allowed_siblings_and_preserves_nonclusters(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    presence = t.zeros(12, dtype=t.bool)
    presence[[7, 10]] = True
    projected = state.project_allowed_presence(presence)
    assert projected.nonzero().squeeze(1).tolist() == [2, 10]


def test_project_presence_ignores_masked_sibling_before_mapping(tmp_path: Path):
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.blacklist, (2,)),
    )
    presence = t.zeros(12, dtype=t.bool)
    presence[2] = True
    projected = state.project_allowed_presence(presence)
    assert not bool(projected.any())


def test_project_presence_maps_new_representative_after_mask_shift(tmp_path: Path):
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.blacklist, (2,)),
    )
    presence = t.zeros(12, dtype=t.bool)
    presence[11] = True
    projected = state.project_allowed_presence(presence)
    assert projected.nonzero().squeeze(1).tolist() == [7]


@pytest.mark.parametrize(
    "presence",
    [
        t.zeros(3),
        t.zeros(2, dtype=t.bool),
        t.zeros((1, 3), dtype=t.bool),
    ],
)
def test_project_presence_requires_exact_bool_width(tmp_path: Path, presence: t.Tensor):
    state = _state(tmp_path, [[0, 1]], n_sae_features=3)
    with pytest.raises(FeatureClustersError, match=r"bool\[n_sae_features\]"):
        state.project_allowed_presence(presence)


def test_effective_negative_mask_without_user_mask_blacklists_only_away(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    mask = state.effective_negative_mask()
    assert mask == FeatureMask(FeatureMaskMode.blacklist, (7, 11))
    assert state.effective_negative_mask() is mask
    assert mask is not None
    assert t.equal(mask.allowed_universe_mask(12), state.cluster_allowed_universe)


def test_effective_negative_mask_unions_blacklist_and_away(tmp_path: Path):
    user_mask = FeatureMask(FeatureMaskMode.blacklist, (0, 2))
    state = _state(
        tmp_path,
        [[2, 7, 11], [3, 5]],
        n_sae_features=12,
        mask=user_mask,
    )
    effective = state.effective_negative_mask()
    assert effective == FeatureMask(FeatureMaskMode.blacklist, (0, 2, 5, 11))
    assert state.effective_negative_mask() is effective
    assert effective is not None
    assert t.equal(
        effective.allowed_universe_mask(12),
        state.cluster_allowed_universe,
    )
    assert user_mask.ids == (0, 2)  # Input mask was not mutated.


def test_effective_negative_mask_subtracts_away_from_whitelist(tmp_path: Path):
    user_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 2, 7, 11))
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=user_mask,
    )
    effective = state.effective_negative_mask()
    assert effective == FeatureMask(FeatureMaskMode.whitelist, (0, 2))
    assert state.effective_negative_mask() is effective
    assert effective is not None
    assert t.equal(
        effective.allowed_universe_mask(12),
        state.cluster_allowed_universe,
    )


def test_effective_negative_mask_returns_same_object_when_no_member_collapses(tmp_path: Path):
    user_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 4))
    state = _state(
        tmp_path,
        [[1, 4]],
        n_sae_features=5,
        mask=user_mask,
    )
    assert state.effective_negative_mask() is user_mask


def test_effective_negative_mask_calls_do_no_width_dependent_tensor_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=65_536,
        mask=FeatureMask(FeatureMaskMode.blacklist, (0,)),
    )
    expected = state.effective_negative_mask()

    def unexpected_nonzero(*_args, **_kwargs):
        raise AssertionError("effective_negative_mask rescanned the SAE width")

    monkeypatch.setattr(t, "nonzero", unexpected_nonzero)
    for _ in range(2_000):
        assert state.effective_negative_mask() is expected
def _reference_representative_success(
    state: FeatureClusterConsolidation,
    ids: t.Tensor,
) -> bool:
    if ids.numel() == 0:
        return True
    allowed = state.allowed_universe.to(ids.device)
    away = state.consolidated_away.to(ids.device)
    reps = state.rep_of.to(ids.device)
    return bool(
        allowed[ids].all()
        and not away[ids].any()
        and (reps[ids] == ids).all()
    )


def _effective_mask_representative_success(
    state: FeatureClusterConsolidation,
    ids: t.Tensor,
) -> bool:
    effective = state.effective_negative_mask()
    return effective is None or bool(effective.keep_mask(ids).all())
def _reference_assert_representatives_only(
    state: FeatureClusterConsolidation,
    feature_ids: t.Tensor,
    *,
    surface: str,
) -> None:
    """Frozen pre-SC-003 bounds/allowed/away/representative diagnostics."""
    ids = state._validate_feature_ids(feature_ids, surface=surface)
    if ids.numel() == 0:
        return
    allowed = state.allowed_universe.to(ids.device)
    if not bool(allowed[ids].all()):
        bad = int(ids[~allowed[ids]][0].item())
        raise FeatureClustersError(
            f"{surface} contains mask-disallowed feature ID {bad}"
        )
    away = state.consolidated_away.to(ids.device)
    if bool(away[ids].any()):
        bad = int(ids[away[ids]][0].item())
        raise FeatureClustersError(
            f"{surface} contains consolidated-away feature ID {bad}"
        )
    reps = state.rep_of.to(ids.device)
    non_rep = reps[ids] != ids
    if bool(non_rep.any()):
        bad = int(ids[non_rep][0].item())
        raise FeatureClustersError(
            f"{surface} contains non-representative feature ID {bad}; "
            f"it maps to representative {int(reps[bad].item())}"
        )


def test_representative_rejections_match_frozen_diagnostics_exactly(
    tmp_path: Path,
) -> None:
    active = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    masked = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.blacklist, (0,)),
    )
    identity = _state(
        tmp_path,
        [[2, 7]],
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.blacklist, (7,)),
    )
    forged_rep_of = identity.rep_of.clone()
    forged_rep_of[3] = 2
    forged = replace(
        identity,
        rep_of=forged_rep_of,
        _effective_negative_feature_mask=FeatureMask(
            FeatureMaskMode.blacklist,
            (3, 7),
        ),
    )
    cases = [
        (active, t.tensor([12]), "out-of-range"),
        (masked, t.tensor([0]), "mask-disallowed"),
        (active, t.tensor([7]), "consolidated-away"),
        (forged, t.tensor([3]), "forged-non-representative"),
        (masked, t.tensor([7, 0]), "mask-precedes-away"),
    ]

    for state, ids, _case in cases:
        expected = _validation_outcome(
            lambda state=state, ids=ids: _reference_assert_representatives_only(
                state,
                ids,
                surface="representative parity",
            )
        )
        actual = _validation_outcome(
            lambda state=state, ids=ids: state.assert_representatives_only(
                ids,
                surface="representative parity",
            )
        )
        assert actual == expected
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_representative_assertion_cpu_cuda_outcomes_are_exact(
    tmp_path: Path,
) -> None:
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.blacklist, (0,)),
    )
    for ids in (
        t.tensor([2, 3]),
        t.tensor([0]),
        t.tensor([7]),
        t.tensor([12]),
    ):
        cpu = _validation_outcome(
            lambda ids=ids: state.assert_representatives_only(
                ids,
                surface="device parity",
            )
        )
        cuda_ids = ids.cuda()
        cuda = _validation_outcome(
            lambda cuda_ids=cuda_ids: state.assert_representatives_only(
                cuda_ids,
                surface="device parity",
            )
        )
        assert cuda == cpu


def test_representative_assertion_accepts_allowed_reps_and_nonclusters(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    state.assert_representatives_only(t.tensor([0, 2, 3, 10]))
    state.assert_representatives_only(t.empty(0, dtype=t.int64))


@pytest.mark.parametrize("bad", [-1, 12, 100])
def test_representative_assertion_checks_bounds_before_indexing(tmp_path: Path, bad: int):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    with pytest.raises(FeatureClustersError, match="out of range"):
        state.assert_representatives_only(t.tensor([bad]))


def test_representative_assertion_rejects_allowed_but_consolidated_away(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    assert bool(state.allowed_universe[7])
    with pytest.raises(FeatureClustersError, match="consolidated-away feature ID 7"):
        state.assert_representatives_only(t.tensor([7]))


def test_representative_assertion_rejects_masked_before_away_check(tmp_path: Path):
    state = _state(
        tmp_path,
        [[2, 7, 11]],
        n_sae_features=12,
        mask=FeatureMask(FeatureMaskMode.blacklist, (2,)),
    )
    with pytest.raises(FeatureClustersError, match="mask-disallowed feature ID 2"):
        state.assert_representatives_only(t.tensor([2]))


def test_representative_assertion_surface_is_in_error(tmp_path: Path):
    state = _state(tmp_path, [[0, 1]], n_sae_features=2)
    with pytest.raises(FeatureClustersError, match="negative selection.*feature ID 1"):
        state.assert_representatives_only(
            t.tensor([1]),
            surface="negative selection",
        )


def _valid_selection_shard_contract():
    return {
        "sel_offsets": t.tensor([0, 2, 3], dtype=t.int32),
        "sel_fids": t.tensor([2, 3, 2], dtype=t.uint16),
        "sel_buckets": t.tensor([0, 2, 1], dtype=t.uint8),
        "sel_pcts": t.tensor([50, 200, 100], dtype=t.uint8),
        "neg_offsets": t.tensor([0, 1, 2], dtype=t.int32),
        "neg_fids": t.tensor([4, 5], dtype=t.uint16),
        "expected_segments": 2,
        "effective_buckets": 3,
        "surface": "test shard",
    }


def _selection_contract_from_rows(
    positive_rows: list[list[int]],
    negative_rows: list[list[int]],
    *,
    surface: str = "differential shard",
) -> dict:
    assert len(positive_rows) == len(negative_rows)

    def flatten(rows: list[list[int]]) -> tuple[list[int], list[int]]:
        offsets = [0]
        values: list[int] = []
        for row in rows:
            values.extend(row)
            offsets.append(len(values))
        return offsets, values

    positive_offsets, positive_ids = flatten(positive_rows)
    negative_offsets, negative_ids = flatten(negative_rows)
    return {
        "sel_offsets": t.tensor(positive_offsets, dtype=t.int32),
        "sel_fids": t.tensor(positive_ids, dtype=t.uint16),
        "sel_buckets": t.zeros(len(positive_ids), dtype=t.uint8),
        "sel_pcts": t.zeros(len(positive_ids), dtype=t.uint8),
        "neg_offsets": t.tensor(negative_offsets, dtype=t.int32),
        "neg_fids": t.tensor(negative_ids, dtype=t.uint16),
        "expected_segments": len(positive_rows),
        "effective_buckets": 3,
        "surface": surface,
    }


def _reference_assert_selection_shard_sets(
    state: FeatureClusterConsolidation,
    contract: dict,
) -> None:
    """Frozen pre-SC-002 row loop for exact differential testing."""
    positive_ids = contract["sel_fids"].to(t.int64)
    negative_ids = contract["neg_fids"].to(t.int64)
    surface = contract["surface"]
    state.assert_representatives_only(
        positive_ids,
        surface=f"{surface} positive selection",
    )
    state.assert_representatives_only(
        negative_ids,
        surface=f"{surface} negative selection",
    )

    positive_offsets = contract["sel_offsets"].to(t.int64).tolist()
    negative_offsets = contract["neg_offsets"].to(t.int64).tolist()
    for segment_idx in range(contract["expected_segments"]):
        pos_start, pos_end = positive_offsets[segment_idx : segment_idx + 2]
        neg_start, neg_end = negative_offsets[segment_idx : segment_idx + 2]
        positive = positive_ids[pos_start:pos_end].tolist()
        negative = negative_ids[neg_start:neg_end].tolist()
        if len(positive) != len(set(positive)):
            raise FeatureClustersError(
                f"{surface} segment {segment_idx} contains duplicate positive IDs"
            )
        if len(negative) != len(set(negative)):
            raise FeatureClustersError(
                f"{surface} segment {segment_idx} contains duplicate negative IDs"
            )
        overlap = set(positive).intersection(negative)
        if overlap:
            raise FeatureClustersError(
                f"{surface} segment {segment_idx} contains feature ID "
                f"{min(overlap)} as both positive and negative"
            )


def _validation_outcome(validate) -> tuple[type[Exception], str] | None:
    try:
        validate()
    except Exception as exc:
        return type(exc), str(exc)
    return None


def test_selection_shard_contract_accepts_exact_valid_csr(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    state.assert_selection_shard(**_valid_selection_shard_contract())


def test_selection_shard_set_validation_bounds_tensor_to_list_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_size = 4_096
    n_segments = block_size + 1
    state = _state(tmp_path, [[6, 7]], n_sae_features=8)
    offsets = t.arange(n_segments + 1, dtype=t.int32)
    contract = {
        "sel_offsets": offsets,
        "sel_fids": t.full((n_segments,), 1, dtype=t.uint16),
        "sel_buckets": t.zeros(n_segments, dtype=t.uint8),
        "sel_pcts": t.zeros(n_segments, dtype=t.uint8),
        "neg_offsets": offsets.clone(),
        "neg_fids": t.full((n_segments,), 2, dtype=t.uint16),
        "expected_segments": n_segments,
        "effective_buckets": 3,
        "surface": "tolist-count shard",
    }
    original = t.Tensor.tolist
    calls = 0

    def counted(tensor: t.Tensor):
        nonlocal calls
        calls += 1
        return original(tensor)

    monkeypatch.setattr(t.Tensor, "tolist", counted)
    state.assert_selection_shard(**contract)

    assert calls <= 2 + 2 * math.ceil(n_segments / block_size)


def test_selection_shard_all_empty_set_validation_uses_no_tensor_to_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_segments = 4_097
    state = _state(tmp_path, [[6, 7]], n_sae_features=8)
    contract = {
        "sel_offsets": t.zeros(n_segments + 1, dtype=t.int32),
        "sel_fids": t.empty(0, dtype=t.uint16),
        "sel_buckets": t.empty(0, dtype=t.uint8),
        "sel_pcts": t.empty(0, dtype=t.uint8),
        "neg_offsets": t.zeros(n_segments + 1, dtype=t.int32),
        "neg_fids": t.empty(0, dtype=t.uint16),
        "expected_segments": n_segments,
        "effective_buckets": 3,
        "surface": "all-empty shard",
    }
    original = t.Tensor.tolist
    calls = 0

    def counted(tensor: t.Tensor):
        nonlocal calls
        calls += 1
        return original(tensor)

    monkeypatch.setattr(t.Tensor, "tolist", counted)
    state.assert_selection_shard(**contract)

    assert calls == 0


@pytest.mark.parametrize(
    ("n_segments", "corruption", "segment_idx"),
    [
        pytest.param(0, None, None, id="zero-segments"),
        pytest.param(1, None, None, id="single-segment"),
        pytest.param(4_095, None, None, id="block-size-minus-one"),
        pytest.param(4_096, None, None, id="block-size"),
        pytest.param(4_097, None, None, id="block-size-plus-one"),
        pytest.param(8_195, None, None, id="multiple-blocks"),
        pytest.param(4_098, "empty", 4_096, id="empty-row-at-boundary"),
        pytest.param(4_096, "duplicate-positive", 4_095, id="corrupt-before-boundary"),
        pytest.param(4_097, "duplicate-negative", 4_096, id="corrupt-at-boundary"),
        pytest.param(4_098, "overlap", 4_097, id="corrupt-after-boundary"),
        pytest.param(4_098, "all-three", 4_096, id="same-row-error-order"),
        pytest.param(8_195, "multiple", 4_097, id="multiple-corruptions"),
        pytest.param(4_097, "fid-extremes", 4_095, id="uint32-max-key-and-block-reset"),
        pytest.param(4_097, "fid-extremes-positive", 4_095, id="uint32-max-key-positive-duplicate"),
        pytest.param(
            4_097,
            "fid-extremes-negative",
            4_095,
            id="uint32-max-key-negative-before-overlap",
        ),
        pytest.param(
            4_097,
            "fid-extremes-first",
            4_095,
            id="earlier-overlap-minimum-before-duplicate",
        ),
    ],
)
def test_selection_shard_blocked_set_validation_matches_frozen_row_loop(
    tmp_path: Path,
    n_segments: int,
    corruption: str | None,
    segment_idx: int | None,
) -> None:
    extreme_ids = corruption is not None and corruption.startswith("fid-extremes")
    state = _state(tmp_path, [[6, 7]], n_sae_features=65_536 if extreme_ids else 8)
    positive_rows = [[1] for _ in range(n_segments)]
    negative_rows = [[2] for _ in range(n_segments)]
    if corruption is not None:
        assert segment_idx is not None
        if extreme_ids:
            for row in (4_094, 4_095, 4_096):
                positive_rows[row], negative_rows[row] = [0, 65_535], []
            if corruption == "fid-extremes-positive":
                positive_rows[segment_idx] = [65_535, 65_535]
            elif corruption == "fid-extremes-negative":
                negative_rows[segment_idx] = [65_535, 65_535]
            elif corruption == "fid-extremes-first":
                negative_rows[segment_idx - 1] = [65_535, 0]
                positive_rows[segment_idx] = [65_535, 65_535]
        elif corruption == "empty":
            positive_rows[segment_idx] = []
            negative_rows[segment_idx] = []
        elif corruption == "duplicate-positive":
            positive_rows[segment_idx] = [1, 1]
        elif corruption == "duplicate-negative":
            negative_rows[segment_idx] = [2, 2]
        elif corruption == "overlap":
            negative_rows[segment_idx] = [1]
        elif corruption == "all-three":
            positive_rows[segment_idx] = [1, 1]
            negative_rows[segment_idx] = [1, 2, 2]
        else:
            assert corruption == "multiple"
            negative_rows[segment_idx] = [1, 2, 2]
            positive_rows[segment_idx + 1] = [1, 1]
            negative_rows[segment_idx + 1] = [1]

    contract = _selection_contract_from_rows(positive_rows, negative_rows)
    reference = _validation_outcome(
        lambda: _reference_assert_selection_shard_sets(state, contract)
    )
    actual = _validation_outcome(lambda: state.assert_selection_shard(**contract))

    assert actual == reference


@pytest.mark.benchmark
def test_selection_shard_set_validation_records_production_shape_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    block_size = 4_096
    n_segments = 282_343
    state = _state(tmp_path, [[6, 7]], n_sae_features=8)
    offsets = t.arange(n_segments + 1, dtype=t.int32)
    contract = {
        "sel_offsets": offsets,
        "sel_fids": t.full((n_segments,), 1, dtype=t.uint16),
        "sel_buckets": t.zeros(n_segments, dtype=t.uint8),
        "sel_pcts": t.zeros(n_segments, dtype=t.uint8),
        "neg_offsets": offsets.clone(),
        "neg_fids": t.full((n_segments,), 2, dtype=t.uint16),
        "expected_segments": n_segments,
        "effective_buckets": 3,
        "surface": "production-shaped benchmark shard",
    }
    original = t.Tensor.tolist
    calls = 0

    def counted(tensor: t.Tensor):
        nonlocal calls
        calls += 1
        return original(tensor)

    monkeypatch.setattr(t.Tensor, "tolist", counted)
    started = time.monotonic()
    state.assert_selection_shard(**contract)
    elapsed = time.monotonic() - started

    record_property("wall_seconds", elapsed)
    record_property("tensor_tolist_calls", calls)
    assert elapsed >= 0.0
    assert calls <= 2 + 2 * math.ceil(n_segments / block_size)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sel_offsets", t.tensor([0, 2], dtype=t.int32), "3 for 2 segments"),
        ("sel_offsets", t.tensor([1, 2, 3], dtype=t.int32), "start at zero"),
        (
            "sel_offsets",
            t.tensor([0, 3, 2], dtype=t.int32),
            "monotonically nondecreasing",
        ),
        (
            "sel_offsets",
            t.tensor([0, 2, 2], dtype=t.int32),
            "terminates at 2.*length 3",
        ),
        ("sel_offsets", t.tensor([0, 2, 3]), "dtype torch.int32"),
        (
            "sel_offsets",
            t.tensor([[0, 2, 3]], dtype=t.int32),
            "one-dimensional",
        ),
        ("sel_fids", t.tensor([2, 3, 2]), "dtype torch.uint16"),
        (
            "sel_buckets",
            t.tensor([0, 2], dtype=t.uint8),
            "positive channels are misaligned",
        ),
        (
            "sel_buckets",
            t.tensor([0, 3, 1], dtype=t.uint8),
            "positive bucket 3 is out of range",
        ),
        (
            "sel_fids",
            t.tensor([2, 2, 2], dtype=t.uint16),
            "segment 0 contains duplicate positive IDs",
        ),
        (
            "neg_fids",
            t.tensor([3, 5], dtype=t.uint16),
            "feature ID 3 as both positive and negative",
        ),
    ],
)
def test_selection_shard_contract_rejects_malformed_storage(
    tmp_path: Path,
    field: str,
    value,
    match: str,
):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    contract = _valid_selection_shard_contract()
    contract[field] = value
    with pytest.raises(FeatureClustersError, match=match):
        state.assert_selection_shard(**contract)


def test_selection_shard_contract_rejects_duplicate_negative_ids(tmp_path: Path):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    contract = _valid_selection_shard_contract()
    contract["neg_offsets"] = t.tensor([0, 2, 2], dtype=t.int32)
    contract["neg_fids"] = t.tensor([4, 4], dtype=t.uint16)
    with pytest.raises(
        FeatureClustersError,
        match="segment 0 contains duplicate negative IDs",
    ):
        state.assert_selection_shard(**contract)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("expected_segments", True, "non-negative strict int"),
        ("expected_segments", -1, "non-negative strict int"),
        ("effective_buckets", True, "strict int in"),
        ("effective_buckets", 0, "strict int in"),
        ("effective_buckets", 257, "strict int in"),
    ],
)
def test_selection_shard_contract_rejects_invalid_scalar_contract(
    tmp_path: Path,
    field: str,
    value,
    match: str,
):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    contract = _valid_selection_shard_contract()
    contract[field] = value
    with pytest.raises(FeatureClustersError, match=match):
        state.assert_selection_shard(**contract)


def test_selection_shard_contract_accepts_all_256_uint8_positive_buckets(
    tmp_path: Path,
):
    state = _state(tmp_path, [[2, 7, 11]], n_sae_features=12)
    contract = _valid_selection_shard_contract()
    contract["effective_buckets"] = 256
    contract["sel_buckets"] = t.tensor([255, 0, 1], dtype=t.uint8)
    state.assert_selection_shard(**contract)


def test_metadata_counts_distinguish_source_survivors_and_away(tmp_path: Path):
    state = _state(
        tmp_path,
        [[0, 1, 2], [3, 4], [5, 6]],
        n_sae_features=8,
        mask=FeatureMask(FeatureMaskMode.blacklist, (0, 1, 3, 4, 6)),
    )
    metadata = state.metadata_dict()
    assert metadata["descriptor"] == state.resolved.descriptor_dict()
    assert metadata["canonical"] == state.resolved.canonical_dict()
    assert metadata["source_path"] == str(state.resolved.source_path)
    assert metadata["counts"] == {
        "source_clusters": 3,
        "source_members": 7,
        "surviving_cluster_representatives": 2,
        "consolidated_away": 0,
    }


def test_metadata_counts_collapsed_survivors(tmp_path: Path):
    state = _state(
        tmp_path,
        [[0, 1, 2], [3, 4]],
        n_sae_features=6,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1,)),
    )
    assert state.metadata_dict()["counts"] == {
        "source_clusters": 2,
        "source_members": 5,
        "surviving_cluster_representatives": 2,
        "consolidated_away": 2,
    }


def test_real_width_state_and_high_id_reduction(tmp_path: Path):
    state = _state(
        tmp_path,
        [[40, 9_405, 65_535], [2, 32_768]],
        n_sae_features=65_536,
        mask=FeatureMask(FeatureMaskMode.blacklist, (40,)),
    )
    assert state.rep_of.shape == (65_536,)
    assert state.allowed_universe.shape == (65_536,)
    assert state.rep_of[65_535].item() == 9_405
    reps, pcts = state.consolidate_segment(
        t.tensor([65_535, 32_768, 2, 1], dtype=t.int64),
        t.tensor([0.8, 0.7, 0.2, 0.4], dtype=t.float32),
    )
    assert reps.tolist() == [1, 2, 9_405]
    assert pcts.tolist() == pytest.approx([0.4, 0.7, 0.8])


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_real_width_reducer_is_exact_on_cuda(tmp_path: Path):
    state = _state(
        tmp_path,
        [[40, 9_405, 65_535], [2, 32_768]],
        n_sae_features=65_536,
    )
    ids_cpu = t.tensor([65_535, 40, 32_768, 2, 10], dtype=t.int64)
    pcts_cpu = t.tensor([0.7, 0.3, 0.4, 0.9, 0.2], dtype=t.float32)
    scores_cpu = t.tensor([-1.0, -2.0, 5.0, 3.0, 0.0], dtype=t.float32)
    cpu = state.consolidate_segment(ids_cpu, pcts_cpu, scores_cpu)
    cuda = state.consolidate_segment(
        ids_cpu.cuda(),
        pcts_cpu.cuda(),
        scores_cpu.cuda(),
    )
    for expected, actual in zip(cpu, cuda, strict=True):
        assert t.equal(expected, actual.cpu())
