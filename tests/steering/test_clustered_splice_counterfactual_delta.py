"""Oracle and integration tests for clustered splice-counterfactual deltas."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t

import scripts.preprocess.precompute_selected_features as precompute_selection
from errors import FeatureClustersError, FeatureMaskError
from scripts.preprocess.precompute_selected_features import (
    _materialize_splice_segment_selection,
)
from steering.conditional.delta import (
    ClusteredCounterfactualDeltaSegResult,
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
    compute_doc_clustered_counterfactual_deltas,
    compute_doc_counterfactual_deltas,
    consolidate_counterfactual_percentile_sides,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import DeltaBundle, FeaturePipelineConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)

N_SAE = 16


def _clusters(
    tmp_path: Path,
    *,
    groups: list[list[int]],
    mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    path = tmp_path / f"clusters_{len(list(tmp_path.iterdir()))}.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": groups}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        N_SAE,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _python_side_oracle(
    feature_ids: list[int],
    percentiles: list[float],
    clusters: FeatureClusterConsolidation,
) -> dict[int, float]:
    allowed = clusters.allowed_universe.tolist()
    rep_of = clusters.rep_of.tolist()
    out: dict[int, float] = {}
    for feature_id, percentile in zip(feature_ids, percentiles, strict=True):
        if not allowed[feature_id]:
            continue
        representative = rep_of[feature_id]
        out[representative] = max(out.get(representative, 0.0), percentile)
    return out


def _python_delta_oracle(
    current_ids: list[int],
    current_pcts: list[float],
    reference_ids: list[int],
    reference_pcts: list[float],
    clusters: FeatureClusterConsolidation,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    current = _python_side_oracle(current_ids, current_pcts, clusters)
    reference = _python_side_oracle(reference_ids, reference_pcts, clusters)
    deltas = {
        representative: current.get(representative, 0.0) - reference.get(representative, 0.0)
        for representative in current.keys() | reference.keys()
    }
    return current, reference, deltas


def _compute_one(
    *,
    current_ids: list[int],
    current_pcts: list[float],
    reference_ids: list[int],
    reference_pcts: list[float],
    clusters: FeatureClusterConsolidation,
) -> ClusteredCounterfactualDeltaSegResult:
    result = compute_doc_clustered_counterfactual_deltas(
        [t.tensor(current_ids, dtype=t.int64)],
        [t.ones(len(current_ids), dtype=t.float32)],
        [t.tensor(current_pcts, dtype=t.float32)],
        [
            (
                t.tensor(reference_ids, dtype=t.int64),
                t.ones(len(reference_ids), dtype=t.float32),
                t.tensor(reference_pcts, dtype=t.float32),
            )
        ],
        clusters,
    )[0]
    assert result is not None
    return result


def _sampling_pipeline(
    *,
    mask: FeatureMask | None,
    negative: bool,
    neg_delta_ratio: float = 1.0,
    seed: int = 91,
) -> FeaturePipelineConfig:
    percentile_table = t.arange(1, 100, dtype=t.float64).unsqueeze(0).expand(N_SAE, -1).clone()
    negative_bundle = None
    if negative:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=2,
                max_k=3,
            ),
            n_sae_features=N_SAE,
            rng=t.Generator().manual_seed(seed + 1),
            feature_mask=mask,
        )
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=2,
                max_k=3,
                mode=SamplingMode.top_k_percentile,
                delta_weights=DeltaSamplingWeights.percentile_delta,
            ),
            percentile_table=percentile_table,
            rng=t.Generator().manual_seed(seed),
        ),
        boundaries=t.zeros(N_SAE, 3, dtype=t.float64),
        negative_bundle=negative_bundle,
        effective_buckets=4,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=neg_delta_ratio,
            ),
            percentile_table=percentile_table,
        ),
        feature_mask=mask,
    )


def test_side_consolidation_matches_python_oracle_and_masks_first(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        (1, 3, 6, 7, 10, 11, 12, 13, 14, 15),
    )
    clusters = _clusters(tmp_path, groups=[[1, 3, 5], [6, 7]], mask=mask)
    current_ids = [3, 5, 7, 10]
    current_pcts = [0.4, 0.99, 0.8, 0.2]
    reference_ids = [1, 3, 5, 6, 11]
    reference_pcts = [0.3, 0.9, 1.0, 0.1, 0.7]

    current, reference, _ = _python_delta_oracle(
        current_ids,
        current_pcts,
        reference_ids,
        reference_pcts,
        clusters,
    )
    current_reps, current_vals, reference_reps, reference_vals = (
        consolidate_counterfactual_percentile_sides(
            current_fids=t.tensor(current_ids),
            current_mags=t.ones(len(current_ids)),
            current_pcts=t.tensor(current_pcts),
            reference_fids=t.tensor(reference_ids),
            reference_mags=t.ones(len(reference_ids)),
            reference_pcts=t.tensor(reference_pcts),
            feature_clusters=clusters,
        )
    )

    assert current_reps.tolist() == sorted(current)
    assert reference_reps.tolist() == sorted(reference)
    assert current_vals.tolist() == pytest.approx([current[i] for i in sorted(current)])
    assert reference_vals.tolist() == pytest.approx([reference[i] for i in sorted(reference)])
    # Mask-disallowed ID 5 has percentile 1.0 but contributes to neither side.
    assert current[1] == pytest.approx(0.4)
    assert reference[1] == pytest.approx(0.9)


def test_noncommutativity_counterexample_matches_cluster_then_delta(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3]])
    current_ids = [1, 3]
    current_pcts = [0.9, 0.1]
    reference_ids = [1, 3]
    reference_pcts = [0.1, 0.9]

    raw = compute_doc_counterfactual_deltas(
        [t.tensor(current_ids)],
        [t.ones(2)],
        [t.tensor(current_pcts)],
        [(t.tensor(reference_ids), t.ones(2), t.tensor(reference_pcts))],
        N_SAE,
    )[0]
    assert raw is not None
    assert raw[0].tolist() == [1]
    assert raw[2] is not None
    assert raw[2].tolist() == pytest.approx([0.8])

    clustered = _compute_one(
        current_ids=current_ids,
        current_pcts=current_pcts,
        reference_ids=reference_ids,
        reference_pcts=reference_pcts,
        clusters=clusters,
    )
    assert clustered.filtered_fids.tolist() == []
    assert clustered.delta_values.tolist() == []
    assert clustered.neg_delta_fids.tolist() == []
    assert clustered.neg_delta_weights.tolist() == []
    assert t.nonzero(clustered.ref_mask).squeeze(1).tolist() == [1]


def test_current_reference_only_and_signed_negative_contract(tmp_path: Path) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    result = _compute_one(
        current_ids=[3, 5, 10],
        current_pcts=[0.9, 0.6, 0.8],
        reference_ids=[1, 7, 11],
        reference_pcts=[0.2, 0.9, 0.7],
        clusters=clusters,
    )

    assert result.filtered_fids.tolist() == [1, 10]
    assert result.filtered_pcts.tolist() == pytest.approx([0.9, 0.8])
    assert result.delta_values.tolist() == pytest.approx([0.7, 0.8])
    # Reference-only representatives come first, then current-present negative
    # representatives, exactly matching the established raw implementation.
    assert result.neg_delta_fids.tolist() == [11, 5]
    assert result.neg_delta_weights.tolist() == pytest.approx([0.7, 0.3])
    assert t.nonzero(result.ref_mask).squeeze(1).tolist() == [1, 5, 10, 11]
    assert set(result.filtered_fids.tolist()).isdisjoint(result.neg_delta_fids.tolist())


def test_segments_are_independent_and_none_reference_remains_ordinary(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    results = compute_doc_clustered_counterfactual_deltas(
        [t.tensor([1]), t.tensor([10]), t.tensor([5])],
        [t.ones(1), t.ones(1), t.ones(1)],
        [t.tensor([0.8]), t.tensor([0.6]), t.tensor([0.4])],
        [
            (t.tensor([7]), t.ones(1), t.tensor([0.9])),
            None,
            (t.tensor([3]), t.ones(1), t.tensor([0.1])),
        ],
        clusters,
    )

    first, ordinary, last = results
    assert first is not None
    assert ordinary is None
    assert last is not None
    assert first.filtered_fids.tolist() == [1]
    assert first.neg_delta_fids.tolist() == [5]
    # Representative 5 was a reference-only negative in the first segment.
    # It must not remain in the dense reference buffer across either the next
    # ordinary segment or the final independent counterfactual comparison.
    assert last.filtered_fids.tolist() == [5]
    assert last.delta_values.tolist() == pytest.approx([0.4])
    assert last.neg_delta_fids.tolist() == [1]
    assert last.neg_delta_weights.tolist() == pytest.approx([0.1])


@pytest.mark.parametrize(
    ("current_ids", "current_pcts", "reference_ids", "reference_pcts", "positive", "negative"),
    [
        ([5], [0.9], [3], [0.6], [], [1]),
        ([3], [0.6], [5], [0.9], [1], []),
        ([5], [0.9], [7], [0.8], [], []),
        ([3], [0.4], [1], [0.4], [], []),
        ([10], [0.7], [11], [0.2], [10], [11]),
    ],
)
def test_post_mask_empty_equal_singleton_and_noncluster_edges(
    tmp_path: Path,
    current_ids: list[int],
    current_pcts: list[float],
    reference_ids: list[int],
    reference_pcts: list[float],
    positive: list[int],
    negative: list[int],
) -> None:
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        (1, 3, 10, 11, 12, 13, 14, 15),
    )
    clusters = _clusters(tmp_path, groups=[[1, 3, 5]], mask=mask)
    result = _compute_one(
        current_ids=current_ids,
        current_pcts=current_pcts,
        reference_ids=reference_ids,
        reference_pcts=reference_pcts,
        clusters=clusters,
    )
    assert result.filtered_fids.tolist() == positive
    assert result.neg_delta_fids.tolist() == negative
    assert all(bool(clusters.allowed_universe[i]) for i in positive + negative)
    assert all(not bool(clusters.consolidated_away[i]) for i in positive + negative)


def test_raw_empty_segments_still_fail_before_post_mask_empty_logic(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3))
    clusters = _clusters(tmp_path, groups=[[1, 3]], mask=mask)
    with pytest.raises(ValueError, match="zero feature ids"):
        compute_doc_clustered_counterfactual_deltas(
            [t.tensor([], dtype=t.int64)],
            [t.tensor([], dtype=t.float32)],
            [t.tensor([], dtype=t.float32)],
            [(t.tensor([1]), t.ones(1), t.tensor([0.5]))],
            clusters,
        )
    with pytest.raises(ValueError, match="reference.*zero feature ids"):
        compute_doc_clustered_counterfactual_deltas(
            [t.tensor([1])],
            [t.ones(1)],
            [t.tensor([0.5])],
            [
                (
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.float32),
                    t.tensor([], dtype=t.float32),
                )
            ],
            clusters,
        )


def test_percentile_fallback_masks_before_side_max(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 10, 11))
    clusters = _clusters(tmp_path, groups=[[1, 3]], mask=mask)
    percentile_table = t.arange(1, 100, dtype=t.float64).unsqueeze(0).expand(N_SAE, -1).clone()
    result = compute_doc_clustered_counterfactual_deltas(
        [t.tensor([1, 3])],
        [t.tensor([20.0, 99.0])],
        [None],
        [(t.tensor([1, 3]), t.tensor([10.0, 99.0]), None)],
        clusters,
        percentile_table,
    )[0]
    assert result is not None
    # Disallowed sibling 3 has the largest magnitude on both sides but cannot
    # affect either max. Allowed representative 1 is 20th vs 10th percentile.
    assert result.filtered_fids.tolist() == [1]
    assert result.filtered_pcts.tolist() == pytest.approx([0.2])
    assert result.delta_values.tolist() == pytest.approx([0.1])


def test_precomputed_percentiles_accept_intentional_empty_magnitude_placeholder(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3]])
    result = compute_doc_clustered_counterfactual_deltas(
        [t.tensor([1, 3])],
        [t.tensor([], dtype=t.float32)],
        [t.tensor([0.9, 0.1])],
        [
            (
                t.tensor([1, 3]),
                t.tensor([], dtype=t.float32),
                t.tensor([0.2, 0.3]),
            )
        ],
        clusters,
    )[0]
    assert result is not None
    assert result.filtered_fids.tolist() == [1]
    assert result.delta_values.tolist() == pytest.approx([0.6])


def test_clustered_delta_input_contract_fails_loud(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3))
    clusters = _clusters(tmp_path, groups=[[1, 3]], mask=mask)
    reference = (t.tensor([1]), t.ones(1), t.tensor([0.5]))

    with pytest.raises(ValueError, match="must have equal lengths"):
        compute_doc_clustered_counterfactual_deltas(
            [t.tensor([1])],
            [],
            [t.tensor([0.5])],
            [reference],
            clusters,
        )
    # Bounds are checked before the mask can discard a raw ID.
    with pytest.raises(ValueError, match=r"feature ID 16 is out of range \[0, 16\)"):
        compute_doc_clustered_counterfactual_deltas(
            [t.tensor([1, N_SAE])],
            [t.ones(2)],
            [t.tensor([0.5, 0.9])],
            [reference],
            clusters,
        )
    with pytest.raises(ValueError, match="feature percentiles must have shape"):
        compute_doc_clustered_counterfactual_deltas(
            [t.tensor([1, 3])],
            [t.ones(2)],
            [t.tensor([0.5])],
            [reference],
            clusters,
        )
    with pytest.raises(ValueError, match="percentile_table is required"):
        compute_doc_clustered_counterfactual_deltas(
            [t.tensor([1])],
            [t.ones(1)],
            [None],
            [(t.tensor([1]), t.ones(1), None)],
            clusters,
        )


def test_randomized_differential_oracle_covers_all_result_channels(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        (0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15),
    )
    clusters = _clusters(
        tmp_path,
        groups=[[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]],
        mask=mask,
    )
    rng = random.Random(790)
    universe = list(range(N_SAE))

    for _ in range(200):
        current_ids = sorted(rng.sample(universe, rng.randint(1, N_SAE)))
        reference_ids = sorted(rng.sample(universe, rng.randint(1, N_SAE)))
        current_pcts = [rng.random() for _ in current_ids]
        reference_pcts = [rng.random() for _ in reference_ids]
        current, reference, deltas = _python_delta_oracle(
            current_ids,
            current_pcts,
            reference_ids,
            reference_pcts,
            clusters,
        )
        result = _compute_one(
            current_ids=current_ids,
            current_pcts=current_pcts,
            reference_ids=reference_ids,
            reference_pcts=reference_pcts,
            clusters=clusters,
        )

        positive_ids = sorted(i for i in current if deltas[i] > 0.0)
        reference_only = sorted(reference.keys() - current.keys())
        current_negative = sorted(i for i in current if deltas[i] < 0.0)
        negative_ids = reference_only + current_negative
        assert result.filtered_fids.tolist() == positive_ids
        assert result.filtered_pcts.tolist() == pytest.approx([current[i] for i in positive_ids])
        assert result.delta_values.tolist() == pytest.approx(
            [deltas[i] for i in positive_ids],
            rel=1e-5,
            abs=1e-7,
        )
        assert result.neg_delta_fids.tolist() == negative_ids
        assert result.neg_delta_weights.tolist() == pytest.approx(
            [reference[i] for i in reference_only] + [-deltas[i] for i in current_negative],
            rel=1e-5,
            abs=1e-7,
        )
        assert t.nonzero(result.ref_mask).squeeze(1).tolist() == sorted(
            current.keys() | reference.keys()
        )


def test_materializer_uses_clustered_percentiles_deltas_and_representatives(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    pipeline = _sampling_pipeline(mask=None, negative=True)
    result = _compute_one(
        current_ids=[3, 5, 10],
        current_pcts=[0.9, 0.6, 0.8],
        reference_ids=[1, 7, 11],
        reference_pcts=[0.2, 0.9, 0.7],
        clusters=clusters,
    )

    selected, buckets, encoded_pcts, negatives = _materialize_splice_segment_selection(
        raw_fids=t.tensor([3, 5, 10]),
        raw_mags=t.ones(3),
        raw_pcts=t.tensor([0.9, 0.6, 0.8]),
        delta_result=result,
        pipeline_config=pipeline,
        feature_clusters=clusters,
    )

    assert selected == [10, 1]
    assert buckets == [3, 3]
    assert encoded_pcts == [204, 230]
    assert isinstance(selected, list)
    assert isinstance(negatives, list)
    assert set(negatives) == {5, 11}
    assert set(selected).isdisjoint(negatives)
    clusters.assert_representatives_only(
        t.tensor(selected + negatives),
        surface="test materialized IDs",
    )


@pytest.mark.parametrize("tensor_output", [False, True])
def test_materializer_counts_delta_negatives_when_uniform_pool_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_output: bool,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3, 5, 7))
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]], mask=mask)
    pipeline = _sampling_pipeline(mask=mask, negative=True)
    result = _compute_one(
        current_ids=[3],
        current_pcts=[0.9],
        reference_ids=[7],
        reference_pcts=[0.8],
        clusters=clusters,
    )

    def duplicate_mask_check(*_args, **_kwargs):
        raise AssertionError("clustered output repeated the representative mask check")

    monkeypatch.setattr(FeatureMask, "assert_no_masked", duplicate_mask_check)
    selected, _buckets, _encoded_pcts, negatives = (
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([3]),
            raw_mags=t.ones(1),
            raw_pcts=t.tensor([0.9]),
            delta_result=result,
            pipeline_config=pipeline,
            feature_clusters=clusters,
            tensor_output=tensor_output,
        )
    )

    assert list(selected) == [1]
    # The effective representative universe is exactly {1, 5}; both are
    # current/reference-present, so the uniform inactive pool has capacity
    # zero. The signed reference-only candidate must still count as capacity.
    assert list(negatives) == [5]


@pytest.mark.parametrize("tensor_output", [False, True])
@pytest.mark.parametrize("side", ["positive", "negative"])
@pytest.mark.parametrize(
    ("bad_id", "message"),
    [
        (15, "mask-disallowed feature ID 15"),
        (3, "consolidated-away feature ID 3"),
        (-65_535, "feature ID -65535 is out of range"),
        (N_SAE, "feature ID 16 is out of range"),
    ],
)
def test_materializer_rejects_corrupt_clustered_output_before_narrowing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_output: bool,
    side: str,
    bad_id: int,
    message: str,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (15,))
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]], mask=mask)
    result = _compute_one(
        current_ids=[3, 5, 10],
        current_pcts=[0.9, 0.6, 0.8],
        reference_ids=[1, 7, 11],
        reference_pcts=[0.2, 0.9, 0.7],
        clusters=clusters,
    )
    real_positive_sampler = precompute_selection.sample_features_batched
    real_negative_sampler = precompute_selection.sample_delta_negatives
    positive_ids: list[t.Tensor] = []

    def capture_positive(*args, **kwargs):
        sampled = real_positive_sampler(*args, **kwargs)
        positive_ids.append(sampled[0][0])
        return sampled

    def corrupt_output(*args, **kwargs):
        negatives = real_negative_sampler(*args, **kwargs)
        # Corrupt after positive percentile encoding and real negative sampling,
        # so only the final output guards can reject the injected ID. -65535
        # would wrap to the allowed representative 1 in uint16 storage.
        target = positive_ids[0] if side == "positive" else negatives
        target[0] = bad_id
        return negatives

    monkeypatch.setattr(precompute_selection, "sample_features_batched", capture_positive)
    monkeypatch.setattr(precompute_selection, "sample_delta_negatives", corrupt_output)
    with pytest.raises(FeatureClustersError, match=f"splice {side} selection.*{message}"):
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([3, 5, 10]),
            raw_mags=t.ones(3),
            raw_pcts=t.tensor([0.9, 0.6, 0.8]),
            delta_result=result,
            pipeline_config=_sampling_pipeline(mask=mask, negative=True),
            feature_clusters=clusters,
            tensor_output=tensor_output,
        )


@pytest.mark.parametrize("tensor_output", [False, True])
def test_materializer_keeps_unclustered_mask_output_guard(
    monkeypatch: pytest.MonkeyPatch,
    tensor_output: bool,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (15,))
    monkeypatch.setattr(
        precompute_selection,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.tensor([15])],
    )
    with pytest.raises(FeatureMaskError, match="feature_mask violated"):
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([1]),
            raw_mags=t.ones(1),
            raw_pcts=t.tensor([0.9]),
            delta_result=None,
            pipeline_config=_sampling_pipeline(mask=mask, negative=True),
            tensor_output=tensor_output,
        )


def _legacy_clustered_negative_inputs(
    *,
    raw_fids: t.Tensor,
    result: ClusteredCounterfactualDeltaSegResult,
    clusters: FeatureClusterConsolidation,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Frozen reference for the retired pre-fast-path normalization."""
    negative_ids = result.neg_delta_fids.to(t.int64)
    negative_weights = result.neg_delta_weights.float()
    positive_ids = result.filtered_fids.to(t.int64)
    allowed = clusters.allowed_universe
    away = clusters.consolidated_away
    representatives = clusters.rep_of
    keep = (
        allowed[negative_ids]
        & ~away[negative_ids]
        & (representatives[negative_ids] == negative_ids)
    )
    if positive_ids.numel() > 0:
        positive_mask = t.zeros(clusters.n_sae_features, dtype=t.bool)
        positive_mask[positive_ids] = True
        keep &= ~positive_mask[negative_ids]
    negative_ids = negative_ids[keep]
    negative_weights = negative_weights[keep]
    raw_presence = t.zeros(N_SAE, dtype=t.bool)
    if raw_fids.numel():
        raw_presence[raw_fids.to(t.int64)] = True
    projected_raw = clusters.project_allowed_presence(raw_presence)
    raw_ids = t.nonzero(projected_raw, as_tuple=False).squeeze(1)
    projected_ref = clusters.project_allowed_presence(
        result.ref_mask.to(dtype=t.bool)
    )
    return (
        negative_ids,
        negative_weights,
        raw_ids,
        projected_ref,
        result.filtered_fids.to(t.int64),
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (cast(t.Tensor, object()), "must be a tensor"),
        (t.tensor([[1]], dtype=t.int64), "one-dimensional CPU tensor"),
        (
            t.empty(1, dtype=t.int64, device="meta"),
            "one-dimensional CPU tensor",
        ),
        (t.tensor([True]), "must use an integer dtype"),
        (t.tensor([1.0]), "must use an integer dtype"),
        (t.tensor([1.0 + 0.0j]), "must use an integer dtype"),
    ],
)
def test_clustered_delta_integer_vector_contract_rejects_invalid_inputs(
    value: t.Tensor,
    message: str,
) -> None:
    with pytest.raises(FeatureClustersError, match=message):
        precompute_selection._require_clustered_delta_integer_cpu_vector(
            "test IDs",
            value,
        )


@pytest.mark.parametrize("dtype", [t.uint8, t.int8, t.int16, t.int32, t.int64])
def test_clustered_delta_integer_vector_contract_normalizes_integer_dtypes(
    dtype: t.dtype,
) -> None:
    actual = precompute_selection._require_clustered_delta_integer_cpu_vector(
        "test IDs",
        t.tensor([0, 7], dtype=dtype),
    )
    assert actual.dtype is t.int64
    assert actual.device.type == "cpu"
    assert actual.tolist() == [0, 7]


def _valid_clustered_delta_contract_args() -> dict[str, Any]:
    ref_mask = t.zeros(N_SAE, dtype=t.bool)
    ref_mask[t.tensor([1, 5, 10, 11, 12])] = True
    return {
        "raw_fids": t.tensor([3, 5, 10], dtype=t.int64),
        "positive_candidate_fids": t.tensor([1, 10], dtype=t.int64),
        "negative_candidate_fids": t.tensor([5, 11], dtype=t.int64),
        "negative_candidate_weights": t.tensor([0.25, 0.75]),
        "ref_mask": ref_mask,
        "n_sae_features": N_SAE,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "n_sae_features",
            True,
            "SAE width must be a positive strict int",
        ),
        (
            "n_sae_features",
            0,
            "SAE width must be a positive strict int",
        ),
        (
            "n_sae_features",
            N_SAE + 1,
            "SAE width does not match the cluster descriptor",
        ),
        (
            "negative_candidate_weights",
            object(),
            "negative candidate weights must be a tensor",
        ),
        (
            "negative_candidate_weights",
            t.ones((2, 1)),
            "weights must be a one-dimensional CPU tensor",
        ),
        (
            "negative_candidate_weights",
            t.empty(2, device="meta"),
            "weights must be a one-dimensional CPU tensor",
        ),
        (
            "ref_mask",
            object(),
            "reference presence must be a tensor",
        ),
        (
            "ref_mask",
            t.empty(N_SAE, dtype=t.bool, device="meta"),
            "reference presence must be on CPU",
        ),
    ],
)
def test_clustered_delta_contract_rejects_boundary_corruption(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    arguments = _valid_clustered_delta_contract_args()
    arguments[field] = value
    with pytest.raises(FeatureClustersError, match=message):
        precompute_selection._validate_clustered_delta_negative_contract(
            feature_clusters=clusters,
            **arguments,
        )


def test_clustered_delta_contract_normalizes_without_rewriting_presence(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    arguments = _valid_clustered_delta_contract_args()
    arguments["raw_fids"] = t.tensor([], dtype=t.int32)
    arguments["positive_candidate_fids"] = t.tensor(
        [1, 10],
        dtype=t.int16,
    )
    arguments["negative_candidate_fids"] = t.tensor(
        [5, 11],
        dtype=t.int32,
    )
    arguments["negative_candidate_weights"] = t.tensor(
        [0.0, 0.0],
        dtype=t.float64,
    )
    original_ref_mask = arguments["ref_mask"]

    negative_ids, negative_weights, ref_mask = (
        precompute_selection._validate_clustered_delta_negative_contract(
            feature_clusters=clusters,
            **arguments,
        )
    )

    assert negative_ids.dtype is t.int64
    assert negative_ids.tolist() == [5, 11]
    assert negative_weights.dtype is t.float32
    assert negative_weights.tolist() == [0.0, 0.0]
    assert ref_mask is original_ref_mask
    assert ref_mask[12]


@pytest.mark.parametrize("neg_delta_ratio", [0.0, 0.5, 1.0])
def test_clustered_delta_fast_path_is_exact_against_legacy_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    neg_delta_ratio: float,
) -> None:
    """Randomized full-materializer differential, including both RNG streams.

    The test replaces only the final delta sampler call with the exact
    pre-optimization normalization sequence. Positive selection, negative-k
    sampling, capacity clamping, bucket encoding, and final assertions all run
    through the real materializer in both executions.
    """
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        (0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15),
    )
    clusters = _clusters(
        tmp_path,
        groups=[[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]],
        mask=mask,
    )
    setup = random.Random(79_600 + round(neg_delta_ratio * 10))
    universe = list(range(N_SAE))
    real_sampler = precompute_selection.sample_delta_negatives

    for case_idx in range(80):
        current_ids = sorted(
            setup.sample(universe, setup.randint(1, N_SAE))
        )
        reference_ids = sorted(
            setup.sample(universe, setup.randint(1, N_SAE))
        )
        current_pcts = [setup.random() for _ in current_ids]
        reference_pcts = [setup.random() for _ in reference_ids]
        raw_fids = t.tensor(current_ids, dtype=t.int64)
        raw_pcts = t.tensor(current_pcts, dtype=t.float32)
        result = _compute_one(
            current_ids=current_ids,
            current_pcts=current_pcts,
            reference_ids=reference_ids,
            reference_pcts=reference_pcts,
            clusters=clusters,
        )
        legacy_inputs = _legacy_clustered_negative_inputs(
            raw_fids=raw_fids,
            result=result,
            clusters=clusters,
        )
        seed = 10_000 + case_idx
        fast_pipeline = _sampling_pipeline(
            mask=mask,
            negative=True,
            neg_delta_ratio=neg_delta_ratio,
            seed=seed,
        )
        legacy_pipeline = _sampling_pipeline(
            mask=mask,
            negative=True,
            neg_delta_ratio=neg_delta_ratio,
            seed=seed,
        )

        fast = _materialize_splice_segment_selection(
            raw_fids=raw_fids,
            raw_mags=t.ones(raw_fids.numel()),
            raw_pcts=raw_pcts,
            delta_result=result,
            pipeline_config=fast_pipeline,
            feature_clusters=clusters,
            tensor_output=True,
        )

        def legacy_sampler(
            k: int,
            neg_delta_ratio: float,
            neg_delta_fids: t.Tensor,
            neg_delta_weights: t.Tensor,
            raw_feature_ids: t.Tensor,
            ref_mask: t.Tensor,
            selected_positive_ids: t.Tensor,
            n_sae_features: int,
            rng: t.Generator,
            *,
            feature_mask: FeatureMask | None = None,
            _expected_ref_mask: t.Tensor = result.ref_mask,
            _legacy_inputs: tuple[
                t.Tensor,
                t.Tensor,
                t.Tensor,
                t.Tensor,
                t.Tensor,
            ] = legacy_inputs,
        ) -> t.Tensor:
            # Prove the production call actually took the optimized seam before
            # substituting the old reference inputs.
            del neg_delta_fids, neg_delta_weights
            assert raw_feature_ids.numel() == 0
            assert selected_positive_ids.numel() == 0
            assert ref_mask is _expected_ref_mask
            (
                negative_ids,
                negative_weights,
                raw_ids,
                projected_ref,
                positive_ids,
            ) = _legacy_inputs
            return real_sampler(
                k=k,
                neg_delta_ratio=neg_delta_ratio,
                neg_delta_fids=negative_ids,
                neg_delta_weights=negative_weights,
                raw_feature_ids=raw_ids,
                ref_mask=projected_ref,
                selected_positive_ids=positive_ids,
                n_sae_features=n_sae_features,
                rng=rng,
                feature_mask=feature_mask,
            )

        with monkeypatch.context() as scoped:
            scoped.setattr(
                precompute_selection,
                "sample_delta_negatives",
                legacy_sampler,
            )
            legacy = _materialize_splice_segment_selection(
                raw_fids=raw_fids,
                raw_mags=t.ones(raw_fids.numel()),
                raw_pcts=raw_pcts,
                delta_result=result,
                pipeline_config=legacy_pipeline,
                feature_clusters=clusters,
                tensor_output=True,
            )

        assert all(isinstance(value, t.Tensor) for value in fast)
        assert all(isinstance(value, t.Tensor) for value in legacy)
        fast_tensors = cast(tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor], fast)
        legacy_tensors = cast(
            tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor],
            legacy,
        )
        for actual, expected in zip(fast_tensors, legacy_tensors, strict=True):
            assert t.equal(actual, expected), (
                neg_delta_ratio,
                case_idx,
                actual,
                expected,
            )
        assert isinstance(fast_pipeline.selection_config, SamplingBundle)
        assert isinstance(legacy_pipeline.selection_config, SamplingBundle)
        assert t.equal(
            fast_pipeline.selection_config.rng.get_state(),
            legacy_pipeline.selection_config.rng.get_state(),
        )
        assert fast_pipeline.negative_bundle is not None
        assert legacy_pipeline.negative_bundle is not None
        assert t.equal(
            fast_pipeline.negative_bundle.rng.get_state(),
            legacy_pipeline.negative_bundle.rng.get_state(),
        )


def test_clustered_delta_fast_path_does_not_call_legacy_identity_transforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    result = _compute_one(
        current_ids=[3, 5, 10],
        current_pcts=[0.9, 0.6, 0.8],
        reference_ids=[1, 7, 11],
        reference_pcts=[0.2, 0.9, 0.7],
        clusters=clusters,
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("legacy clustered-delta identity transform ran")

    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "project_allowed_presence",
        unexpected,
    )
    selected, _buckets, _pcts, negatives = (
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([3, 5, 10]),
            raw_mags=t.ones(3),
            raw_pcts=t.tensor([0.9, 0.6, 0.8]),
            delta_result=result,
            pipeline_config=_sampling_pipeline(mask=None, negative=True),
            feature_clusters=clusters,
        )
    )
    assert set(selected) == {1, 10}
    assert set(negatives) == {5, 11}


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("ref_dtype", r"reference presence must be bool\[n_sae_features\]"),
        ("ref_shape", r"reference presence must be bool\[n_sae_features\]"),
        ("ref_nonrepresentative", "not a mask-allowed representative"),
        ("positive_nonrepresentative", "positive candidates contains consolidated-away"),
        ("negative_nonrepresentative", "negative candidates contains consolidated-away"),
        ("positive_absent", "positive candidate feature ID 1 is absent"),
        ("negative_absent", "negative candidate feature ID 11 is absent"),
        ("positive_duplicate", "positive candidates contain duplicate"),
        ("negative_duplicate", "negative candidates contain duplicate"),
        ("positive_negative_overlap", "appears in both positive and negative"),
        ("weight_shape", "IDs and weights must be aligned"),
        ("weight_dtype", "weights must be floating point"),
        ("weight_nan", "weights contain a non-finite value"),
        ("weight_negative", "weights must be non-negative"),
        ("raw_negative", r"raw current feature ID -1 is out of range"),
        ("raw_out_of_range", r"raw current feature ID 16 is out of range"),
    ],
)
def test_clustered_delta_fast_path_rejects_corruption_before_rng(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3], [5, 7]])
    result = _compute_one(
        current_ids=[3, 5, 10],
        current_pcts=[0.9, 0.6, 0.8],
        reference_ids=[1, 7, 11],
        reference_pcts=[0.2, 0.9, 0.7],
        clusters=clusters,
    )
    raw_fids = t.tensor([3, 5, 10])
    if corruption == "ref_dtype":
        result = result._replace(ref_mask=result.ref_mask.float())
    elif corruption == "ref_shape":
        result = result._replace(ref_mask=result.ref_mask[:-1])
    elif corruption == "ref_nonrepresentative":
        ref_mask = result.ref_mask.clone()
        ref_mask[3] = True
        result = result._replace(ref_mask=ref_mask)
    elif corruption == "positive_nonrepresentative":
        filtered_fids = result.filtered_fids.clone()
        filtered_fids[0] = 3
        result = result._replace(filtered_fids=filtered_fids)
    elif corruption == "negative_nonrepresentative":
        negative_fids = result.neg_delta_fids.clone()
        negative_fids[0] = 7
        result = result._replace(neg_delta_fids=negative_fids)
    elif corruption == "positive_absent":
        ref_mask = result.ref_mask.clone()
        ref_mask[1] = False
        result = result._replace(ref_mask=ref_mask)
    elif corruption == "negative_absent":
        ref_mask = result.ref_mask.clone()
        ref_mask[11] = False
        result = result._replace(ref_mask=ref_mask)
    elif corruption == "positive_duplicate":
        result = result._replace(
            filtered_fids=t.cat((result.filtered_fids, result.filtered_fids[:1])),
            filtered_fmags=t.cat((result.filtered_fmags, result.filtered_fmags[:1])),
            delta_values=t.cat((result.delta_values, result.delta_values[:1])),
            filtered_pcts=t.cat((result.filtered_pcts, result.filtered_pcts[:1])),
        )
    elif corruption == "negative_duplicate":
        result = result._replace(
            neg_delta_fids=t.cat(
                (result.neg_delta_fids, result.neg_delta_fids[:1])
            ),
            neg_delta_weights=t.cat(
                (result.neg_delta_weights, result.neg_delta_weights[:1])
            ),
        )
    elif corruption == "positive_negative_overlap":
        result = result._replace(
            neg_delta_fids=t.cat(
                (result.neg_delta_fids, result.filtered_fids[:1])
            ),
            neg_delta_weights=t.cat(
                (result.neg_delta_weights, t.ones(1))
            ),
        )
    elif corruption == "weight_shape":
        result = result._replace(
            neg_delta_weights=result.neg_delta_weights[:1]
        )
    elif corruption == "weight_dtype":
        result = result._replace(
            neg_delta_weights=result.neg_delta_weights.to(t.int64)
        )
    elif corruption == "weight_nan":
        weights = result.neg_delta_weights.clone()
        weights[0] = float("nan")
        result = result._replace(neg_delta_weights=weights)
    elif corruption == "weight_negative":
        weights = result.neg_delta_weights.clone()
        weights[0] = -1.0
        result = result._replace(neg_delta_weights=weights)
    elif corruption == "raw_negative":
        raw_fids = t.tensor([-1, 3])
    elif corruption == "raw_out_of_range":
        raw_fids = t.tensor([3, N_SAE])
    else:
        raise AssertionError(f"unhandled corruption case {corruption}")

    pipeline = _sampling_pipeline(mask=None, negative=True)
    assert isinstance(pipeline.selection_config, SamplingBundle)
    positive_before = pipeline.selection_config.rng.get_state().clone()
    assert pipeline.negative_bundle is not None
    negative_before = pipeline.negative_bundle.rng.get_state().clone()
    with pytest.raises(FeatureClustersError, match=message):
        _materialize_splice_segment_selection(
            raw_fids=raw_fids,
            raw_mags=t.ones(raw_fids.numel()),
            raw_pcts=t.full((raw_fids.numel(),), 0.5),
            delta_result=result,
            pipeline_config=pipeline,
            feature_clusters=clusters,
            tensor_output=True,
        )
    assert t.equal(
        pipeline.selection_config.rng.get_state(),
        positive_before,
    )
    assert t.equal(
        pipeline.negative_bundle.rng.get_state(),
        negative_before,
    )


def test_clustered_delta_fast_path_requires_raw_current_union_coverage(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3]])
    result = _compute_one(
        current_ids=[3],
        current_pcts=[0.5],
        reference_ids=[1],
        reference_pcts=[0.5],
        clusters=clusters,
    )
    assert result.filtered_fids.numel() == 0
    assert result.neg_delta_fids.numel() == 0
    corrupt_ref = result.ref_mask.clone()
    corrupt_ref[1] = False
    pipeline = _sampling_pipeline(mask=None, negative=True)
    assert isinstance(pipeline.selection_config, SamplingBundle)
    positive_before = pipeline.selection_config.rng.get_state().clone()
    assert pipeline.negative_bundle is not None
    negative_before = pipeline.negative_bundle.rng.get_state().clone()

    with pytest.raises(
        FeatureClustersError,
        match="raw current representative 1 is absent",
    ):
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([3]),
            raw_mags=t.ones(1),
            raw_pcts=t.tensor([0.5]),
            delta_result=result._replace(ref_mask=corrupt_ref),
            pipeline_config=pipeline,
            feature_clusters=clusters,
        )
    assert t.equal(
        pipeline.selection_config.rng.get_state(),
        positive_before,
    )
    assert t.equal(
        pipeline.negative_bundle.rng.get_state(),
        negative_before,
    )


def test_materializer_rejects_raw_delta_currency_for_active_clusters(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3]])
    raw_result = compute_doc_counterfactual_deltas(
        [t.tensor([1, 3])],
        [t.ones(2)],
        [t.tensor([0.9, 0.1])],
        [(t.tensor([1, 3]), t.ones(2), t.tensor([0.1, 0.9]))],
        N_SAE,
    )[0]
    assert raw_result is not None
    with pytest.raises(FeatureClustersError, match="wrong delta currency"):
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([1, 3]),
            raw_mags=t.ones(2),
            raw_pcts=t.tensor([0.9, 0.1]),
            delta_result=raw_result,
            pipeline_config=_sampling_pipeline(mask=None, negative=False),
            feature_clusters=clusters,
        )


def test_materializer_rejects_clustered_currency_without_active_clusters(
    tmp_path: Path,
) -> None:
    clusters = _clusters(tmp_path, groups=[[1, 3]])
    clustered_result = _compute_one(
        current_ids=[1, 3],
        current_pcts=[0.9, 0.1],
        reference_ids=[1, 3],
        reference_pcts=[0.2, 0.3],
        clusters=clusters,
    )
    with pytest.raises(FeatureClustersError, match="wrong delta currency"):
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([1, 3]),
            raw_mags=t.ones(2),
            raw_pcts=t.tensor([0.9, 0.1]),
            delta_result=clustered_result,
            pipeline_config=_sampling_pipeline(mask=None, negative=False),
            feature_clusters=None,
        )
