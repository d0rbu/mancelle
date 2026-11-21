"""Reference differential, property, and mutation-trap suite for
``steering.conditional.cluster_segment_selection.select_clustered_segment``.

Three layers of proof:

1. ``reference_select_clustered_segment`` below is an INDEPENDENT
   re-expression of main's pre-A5 inline clustered sequence, built directly
   from the shared primitives (``FeatureMask.apply_to_positive``,
   ``FeatureClusterConsolidation.consolidate_segment``,
   ``sample_features_batched``, a hand-rolled percentile-space threshold
   select, ``percentile_to_bucket``, ``normalized_percentile_to_uint8``,
   ``sample_inactive_features_batched``). It never imports or calls
   ``select_clustered_segment`` (or any of that module's private helpers),
   so an algorithmic bug shared between "the module" and "a copy of the
   module" cannot hide from this differential. A seeded property loop
   (``test_reference_matches_select_clustered_segment_property``, >=200
   trials) feeds both paths identically-seeded fresh ``torch.Generator``
   instances and asserts exact tensor + generator-state equality.
2. Directed semantics tests pin named invariants (empty-segment RNG
   no-op, representative selection, mask-sibling exclusion, the uint16
   storage cap, and the validation guards) precisely rather than relying on
   the property loop to stumble onto them.
3. Mutation traps assert the CORRECT semantics with a comment naming the
   specific mutation each test kills, validated empirically against the
   real ``select_clustered_segment``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from errors import FeatureClustersError
from scripts.preprocess.precompute_selected_features import select_features_for_shard
from steering.conditional.cluster_segment_selection import (
    ClusteredSegmentSelection,
    _assert_result,
    _selected_percentiles,
    select_clustered_segment,
)
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.percentile_table import (
    normalized_percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
    sample_features_batched,
    sample_inactive_features_batched,
)
from steering.conditional.thresholding import ThresholdType
from steering.packed_batch_io import save_packed_row_shard
from tests.steering.goldens.gen_cluster_selection_golden import _shard_from_segments

_PROPERTY_TRIALS = 220
_PROPERTY_MASTER_SEED = 20260721


# ---------------------------------------------------------------------------
# Step 1: independent reference implementation
#
# Deliberately does NOT import steering.conditional.cluster_segment_selection
# except for the ClusteredSegmentSelection/select_clustered_segment names
# used by the *tests* to call the function under test and type its result --
# nothing below is derived from that module's internals. The output shape is
# a locally defined dataclass (not ClusteredSegmentSelection) so there is no
# shared code at all between "the module" and "the check".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReferenceSelection:
    positive_ids: t.Tensor
    positive_buckets: t.Tensor
    positive_percentiles_uint8: t.Tensor
    negative_ids: t.Tensor


def _reference_empty() -> _ReferenceSelection:
    return _ReferenceSelection(
        positive_ids=t.empty(0, dtype=t.int64),
        positive_buckets=t.empty(0, dtype=t.int64),
        positive_percentiles_uint8=t.empty(0, dtype=t.uint8),
        negative_ids=t.empty(0, dtype=t.int64),
    )


def _reference_threshold_select(
    feature_ids: t.Tensor,
    percentiles: t.Tensor,
    selection: ThresholdSelectionConfig,
) -> t.Tensor:
    """Hand-rolled percentile-space threshold filter.

    Re-derived from the ThresholdType.feature_percentile /
    feature_percentile_segment_topk contract documented in
    steering/conditional/thresholding.py: once activations are already
    expressed as per-feature percentile ranks in [0, 1] (as packed-row
    bytes are), "feature_percentile" keeps members at/above a single scalar
    cutoff, and "feature_percentile_segment_topk" additionally caps the
    survivors to the top-k by percentile within the segment (k = int part
    of the combined threshold, percentile = fractional part).
    """
    threshold_type = selection.selection.threshold_type
    threshold = selection.selection.threshold
    if threshold_type is ThresholdType.feature_percentile:
        return feature_ids[percentiles >= float(threshold)]
    if threshold_type is ThresholdType.feature_percentile_segment_topk:
        k = int(threshold)
        percentile_floor = round(float(threshold) - k, 2)
        keep = percentiles >= percentile_floor
        kept_ids = feature_ids[keep]
        kept_percentiles = percentiles[keep]
        actual_k = min(k, int(kept_ids.numel()))
        if actual_k == 0:
            return kept_ids[:0]
        _, top_idx = t.topk(kept_percentiles, actual_k)
        return kept_ids[top_idx]
    raise AssertionError(
        f"reference implementation does not support threshold type {threshold_type}"
    )


def reference_select_clustered_segment(
    raw_feature_ids: t.Tensor,
    percentile_uint8: t.Tensor,
    *,
    clusters: FeatureClusterConsolidation,
    pipeline: FeaturePipelineConfig,
) -> _ReferenceSelection:
    """Independent re-expression of main's pre-A5 inline clustered sequence.

    mask.apply_to_positive -> clusters.consolidate_segment ->
    sample_features_batched(zeros-mags, precomputed_percentiles) / threshold
    select -> percentile lookup -> percentile_to_bucket ->
    normalized_percentile_to_uint8 -> sample_inactive_features_batched
    (clusters.effective_negative_mask()). No validation/error-path logic is
    reproduced here (that is exercised directly against select_clustered_
    segment in the directed semantics tests below) -- only the happy-path
    computation, on inputs already known to be valid.
    """
    feature_ids = raw_feature_ids.to(t.int64)
    percentiles = uint8_to_percentile(percentile_uint8)

    if feature_ids.numel() == 0:
        return _reference_empty()

    if pipeline.feature_mask is not None:
        feature_ids, percentiles = pipeline.feature_mask.apply_to_positive(feature_ids, percentiles)
    if feature_ids.numel() == 0:
        return _reference_empty()

    feature_ids, percentiles = clusters.consolidate_segment(feature_ids, percentiles)
    if feature_ids.numel() == 0:
        return _reference_empty()

    selection = pipeline.selection_config
    if isinstance(selection, SamplingBundle):
        sampled = sample_features_batched(
            all_feature_ids=feature_ids,
            all_feature_mags=t.zeros_like(feature_ids, dtype=t.float32),
            segment_sizes=[int(feature_ids.numel())],
            percentile_table=selection.percentile_table,
            config=selection.selection,
            rng=selection.rng,
            precomputed_percentiles=percentiles,
        )
        selected_ids = sampled[0][0] if sampled else t.empty(0, dtype=t.int64)
    elif isinstance(selection, ThresholdSelectionConfig):
        selected_ids = _reference_threshold_select(feature_ids, percentiles, selection)
    else:
        raise AssertionError(f"unsupported selection config {type(selection).__name__}")

    if selected_ids.numel() == 0:
        selected_percentiles = t.empty(0, dtype=t.float32)
    else:
        values: list[float] = []
        for selected_id in selected_ids.tolist():
            matches = (feature_ids == selected_id).nonzero(as_tuple=True)[0]
            assert len(matches) == 1, "reference candidate lookup out of sync"
            values.append(float(percentiles[matches[0]].item()))
        selected_percentiles = t.tensor(values, dtype=t.float32)

    selected_buckets = percentile_to_bucket(selected_percentiles, pipeline.effective_buckets)
    selected_percentiles_uint8 = normalized_percentile_to_uint8(selected_percentiles)

    negative_ids = t.empty(0, dtype=t.int64)
    if pipeline.negative_bundle is not None:
        negative = pipeline.negative_bundle
        sampled_negative = sample_inactive_features_batched(
            [feature_ids],
            negative.n_sae_features,
            negative.selection,
            negative.rng,
            feature_mask=clusters.effective_negative_mask(),
        )
        if sampled_negative:
            negative_ids = sampled_negative[0].to(t.int64)

    return _ReferenceSelection(
        positive_ids=selected_ids.to(t.int64),
        positive_buckets=selected_buckets.to(t.int64),
        positive_percentiles_uint8=selected_percentiles_uint8,
        negative_ids=negative_ids,
    )


# ---------------------------------------------------------------------------
# Shared test fixtures/helpers (scaffolding only -- no selection algorithm
# lives here).
# ---------------------------------------------------------------------------


def _build_clusters(
    tmp_path: Path,
    groups: list[list[int]],
    mask: FeatureMask | None,
    n_sae_features: int,
    *,
    name: str = "clusters.json",
) -> FeatureClusterConsolidation:
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "clusters": groups}), encoding="utf-8")
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(path)}}, n_sae_features)
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _uint8_bytes(percentiles: list[float]) -> t.Tensor:
    return normalized_percentile_to_uint8(t.tensor(percentiles, dtype=t.float32))


# ---------------------------------------------------------------------------
# Step 1: reference-vs-actual differential (single cases, then property loop)
# ---------------------------------------------------------------------------


def test_reference_matches_actual_single_case_sampling_with_negatives(
    tmp_path: Path,
) -> None:
    n = 24
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(n)))
    clusters = _build_clusters(tmp_path, [[2, 5, 9], [11, 12]], mask, n)
    raw_ids = t.tensor([1, 2, 5, 9, 11, 12, 20], dtype=t.int64)
    pcts = _uint8_bytes([0.2, 0.6, 0.9, 0.3, 0.5, 0.55, 0.8])
    table = t.zeros(n, 99, dtype=t.float64)

    def make_pipeline(seed: int) -> tuple[FeaturePipelineConfig, t.Generator]:
        rng = t.Generator().manual_seed(seed)
        pipeline = FeaturePipelineConfig(
            selection_config=SamplingBundle(
                selection=SamplingSelection(
                    distribution=Distribution.normal,
                    mean=2.0,
                    std=0.01,
                    min_k=2,
                    max_k=4,
                    mode=SamplingMode.top_k_percentile,
                ),
                percentile_table=table,
                rng=rng,
            ),
            boundaries=t.zeros(n, 3, dtype=t.float64),
            negative_bundle=NegativeBundle(
                selection=NegativeSamplingSelection(
                    distribution=Distribution.normal,
                    mean=3.0,
                    std=0.01,
                    min_k=2,
                    max_k=4,
                ),
                n_sae_features=n,
                rng=rng,
                feature_mask=mask,
            ),
            effective_buckets=4,
            feature_mask=mask,
        )
        return pipeline, rng

    expected_pipeline, expected_rng = make_pipeline(seed=4242)
    actual_pipeline, actual_rng = make_pipeline(seed=4242)

    expected = reference_select_clustered_segment(
        raw_ids, pcts, clusters=clusters, pipeline=expected_pipeline
    )
    actual = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=actual_pipeline)

    assert t.equal(expected.positive_ids, actual.positive_ids)
    assert t.equal(expected.positive_buckets, actual.positive_buckets)
    assert t.equal(expected.positive_percentiles_uint8, actual.positive_percentiles_uint8)
    assert t.equal(expected.negative_ids, actual.negative_ids)
    assert t.equal(expected_rng.get_state(), actual_rng.get_state())


def test_reference_matches_actual_single_case_threshold_no_negatives(
    tmp_path: Path,
) -> None:
    n = 16
    clusters = _build_clusters(tmp_path, [[3, 4], [7, 8, 9]], None, n)
    raw_ids = t.tensor([0, 3, 4, 7, 8, 9, 10], dtype=t.int64)
    pcts = _uint8_bytes([0.1, 0.5, 0.95, 0.2, 0.99, 0.3, 0.6])
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.4),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=3,
        feature_mask=None,
    )

    expected = reference_select_clustered_segment(
        raw_ids, pcts, clusters=clusters, pipeline=pipeline
    )
    actual = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)

    assert t.equal(expected.positive_ids, actual.positive_ids)
    assert t.equal(expected.positive_buckets, actual.positive_buckets)
    assert t.equal(expected.positive_percentiles_uint8, actual.positive_percentiles_uint8)
    assert t.equal(expected.negative_ids, actual.negative_ids)


def _property_trial(py_rng: random.Random, trial_idx: int, tmp_path: Path) -> None:
    # Universe size: mostly small (test speed across 220 trials), but every
    # 20th trial pins exactly 65,536 -- the uint16 storage ceiling -- so the
    # differential is exercised at that boundary, not just below it.
    if trial_idx % 20 == 0:
        n_sae_features = 65_536
    else:
        n_sae_features = py_rng.choice([8, 16, 32, 64, 128])

    # Cluster shapes: 1-5 non-overlapping clusters of 2-4 members.
    universe = list(range(n_sae_features))
    py_rng.shuffle(universe)
    n_clusters = py_rng.randint(1, 5)
    groups: list[list[int]] = []
    idx = 0
    for _ in range(n_clusters):
        if idx + 2 > n_sae_features:
            break
        size = py_rng.randint(2, min(4, n_sae_features - idx))
        groups.append(sorted(universe[idx : idx + size]))
        idx += size
    if not groups:
        groups = [sorted(universe[:2])]

    # Mask: none, blacklist, or whitelist, drawn from the full universe (so
    # it sometimes overlaps cluster members -- shifting or emptying them --
    # and sometimes doesn't).
    mask_choice = py_rng.random()
    mask: FeatureMask | None
    if mask_choice < 0.25:
        mask = None
    else:
        n_ids = py_rng.randint(1, max(1, n_sae_features // 2))
        ids = tuple(sorted(py_rng.sample(range(n_sae_features), min(n_ids, n_sae_features))))
        mode = FeatureMaskMode.blacklist if mask_choice < 0.6 else FeatureMaskMode.whitelist
        mask = FeatureMask(mode, ids)

    clusters = _build_clusters(
        tmp_path, groups, mask, n_sae_features, name=f"clusters_{trial_idx}.json"
    )

    # Segment size: 0, 1, or "large" (up to 48 distinct ids), 1/3 each.
    size_choice = py_rng.random()
    if size_choice < 1 / 3:
        size = 0
    elif size_choice < 2 / 3:
        size = 1
    else:
        size = py_rng.randint(2, min(n_sae_features, 48))
    raw_ids_list = py_rng.sample(range(n_sae_features), size)
    raw_ids = t.tensor(raw_ids_list, dtype=t.int64)
    pct_bytes = t.tensor([py_rng.randint(0, 255) for _ in range(size)], dtype=t.uint8)

    effective_buckets = py_rng.randint(1, 6)
    use_sampling = py_rng.random() < 0.5
    use_negative = py_rng.random() < 0.6
    table = t.zeros(n_sae_features, 99, dtype=t.float64)
    seed = _PROPERTY_MASTER_SEED + trial_idx
    neg_seed = seed + 500_000

    def make_pipeline() -> FeaturePipelineConfig:
        if use_sampling:
            rng = t.Generator().manual_seed(seed)
            positive_distribution = py_rng.choice(
                [Distribution.geometric, Distribution.normal]
            )
            positive_mean = py_rng.uniform(1.0, 4.0)
            positive_std = (
                positive_mean
                if positive_distribution is Distribution.geometric
                else py_rng.uniform(0.01, 1.0)
            )
            selection_config: ThresholdSelectionConfig | SamplingBundle = SamplingBundle(
                selection=SamplingSelection(
                    distribution=positive_distribution,
                    mean=positive_mean,
                    std=positive_std,
                    min_k=py_rng.randint(0, 2),
                    max_k=py_rng.randint(3, 6),
                    mode=py_rng.choice([SamplingMode.weighted, SamplingMode.top_k_percentile]),
                ),
                percentile_table=table,
                rng=rng,
            )
            negative_rng = rng
        else:
            threshold_type = py_rng.choice(
                [ThresholdType.feature_percentile, ThresholdType.feature_percentile_segment_topk]
            )
            if threshold_type is ThresholdType.feature_percentile:
                threshold = round(py_rng.uniform(0.0, 0.9), 2)
            else:
                k = py_rng.randint(1, 5)
                pct = round(py_rng.uniform(0.01, 0.9), 2)
                threshold = k + pct
            selection_config = ThresholdSelectionConfig(
                selection=ThresholdSelection(threshold_type, threshold),
                precomputed_thresholds=None,
            )
            negative_rng = t.Generator().manual_seed(neg_seed)

        negative_bundle = None
        if use_negative:
            negative_distribution = py_rng.choice(
                [Distribution.geometric, Distribution.normal]
            )
            negative_mean = py_rng.uniform(1.0, 4.0)
            negative_std = (
                negative_mean
                if negative_distribution is Distribution.geometric
                else py_rng.uniform(0.01, 1.0)
            )
            negative_bundle = NegativeBundle(
                selection=NegativeSamplingSelection(
                    distribution=negative_distribution,
                    mean=negative_mean,
                    std=negative_std,
                    min_k=py_rng.randint(0, 2),
                    max_k=py_rng.randint(3, 6),
                ),
                n_sae_features=n_sae_features,
                rng=negative_rng,
                feature_mask=mask,
            )

        return FeaturePipelineConfig(
            selection_config=selection_config,
            boundaries=t.zeros(n_sae_features, 3, dtype=t.float64),
            negative_bundle=negative_bundle,
            effective_buckets=effective_buckets,
            feature_mask=mask,
        )

    # py_rng.choice/uniform/randint calls inside make_pipeline() must be
    # deterministic across the two builds: snapshot+restore the Python RNG
    # state so both pipelines get IDENTICAL structural parameters.
    py_state = py_rng.getstate()
    expected_pipeline = make_pipeline()
    py_rng.setstate(py_state)
    actual_pipeline = make_pipeline()

    expected = reference_select_clustered_segment(
        raw_ids, pct_bytes, clusters=clusters, pipeline=expected_pipeline
    )
    actual = select_clustered_segment(
        raw_ids, pct_bytes, clusters=clusters, pipeline=actual_pipeline
    )

    case = (
        f"trial={trial_idx} n_sae_features={n_sae_features} groups={groups} "
        f"mask={mask} size={size} use_sampling={use_sampling} use_negative={use_negative}"
    )
    assert t.equal(expected.positive_ids, actual.positive_ids), case
    assert t.equal(expected.positive_buckets, actual.positive_buckets), case
    assert t.equal(expected.positive_percentiles_uint8, actual.positive_percentiles_uint8), case
    assert t.equal(expected.negative_ids, actual.negative_ids), case

    def relevant_rng(pipeline: FeaturePipelineConfig) -> t.Generator | None:
        if isinstance(pipeline.selection_config, SamplingBundle):
            return pipeline.selection_config.rng
        if pipeline.negative_bundle is not None:
            return pipeline.negative_bundle.rng
        return None

    expected_rng = relevant_rng(expected_pipeline)
    actual_rng = relevant_rng(actual_pipeline)
    if expected_rng is not None:
        assert actual_rng is not None, case
        assert t.equal(expected_rng.get_state(), actual_rng.get_state()), case


def test_reference_matches_select_clustered_segment_property(tmp_path: Path) -> None:
    """>=200-trial seeded property loop (Task 3 Step 1).

    Randomizes universe size (including exactly 65,536), cluster shapes,
    masks (including representative-shifting and cluster-emptying overlaps),
    segment sizes (0/1/large), both selection configs, and negatives on/off.
    No hypothesis dependency: a plain seeded ``random.Random`` loop keeps the
    per-trial structural parameters and the torch.Generator RNG streams
    exactly reproducible and independently inspectable on failure.
    """
    py_rng = random.Random(_PROPERTY_MASTER_SEED)
    for trial_idx in range(_PROPERTY_TRIALS):
        trial_dir = tmp_path / f"trial_{trial_idx}"
        trial_dir.mkdir()
        _property_trial(py_rng, trial_idx, trial_dir)


# ---------------------------------------------------------------------------
# Step 2: directed semantics tests
# ---------------------------------------------------------------------------


def test_empty_segment_consumes_no_rng(tmp_path: Path) -> None:
    """An empty segment must return immediately without touching any RNG
    stream -- neither the positive sampler nor the negative sampler."""
    n = 16
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(n)))
    clusters = _build_clusters(tmp_path, [[1, 2]], mask, n)
    rng = t.Generator().manual_seed(777)
    state_before = rng.get_state().clone()
    pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=1,
                max_k=3,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=t.zeros(n, 99, dtype=t.float64),
            rng=rng,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=3.0,
                std=0.01,
                min_k=1,
                max_k=3,
            ),
            n_sae_features=n,
            rng=rng,
            feature_mask=mask,
        ),
        effective_buckets=4,
        feature_mask=mask,
    )

    result = select_clustered_segment(
        t.tensor([], dtype=t.int64),
        t.tensor([], dtype=t.uint8),
        clusters=clusters,
        pipeline=pipeline,
    )

    assert result.positive_ids.numel() == 0
    assert result.negative_ids.numel() == 0
    assert t.equal(rng.get_state(), state_before)


def test_representative_is_min_surviving_member(tmp_path: Path) -> None:
    """Cluster [2, 5, 9] with id 2 masked out: the representative must be 5
    (the numerically smallest SURVIVING member), not 2 (masked) or 9."""
    n = 16
    mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    clusters = _build_clusters(tmp_path, [[2, 5, 9]], mask, n)

    assert clusters.rep_of[5].item() == 5
    assert clusters.rep_of[9].item() == 5
    assert not bool(clusters.consolidated_away[5])
    assert bool(clusters.consolidated_away[9])

    raw_ids = t.tensor([5, 9], dtype=t.int64)
    pcts = _uint8_bytes([0.3, 0.7])
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.01),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )
    result = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)
    assert result.positive_ids.tolist() == [5]


def test_masked_sibling_never_contributes(tmp_path: Path) -> None:
    """A masked sibling with a HIGH percentile must not influence the
    surviving representative's amax -- it is stripped before consolidation
    ever sees it, however large its own percentile byte is."""
    n = 16
    mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    clusters = _build_clusters(tmp_path, [[2, 5]], mask, n)

    raw_ids = t.tensor([2, 5], dtype=t.int64)
    pcts = _uint8_bytes([0.99, 0.20])  # masked id 2 has the higher percentile
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.01),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )
    result = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)
    assert result.positive_ids.tolist() == [5]
    # The representative's reported percentile is id 5's own (~0.20), not
    # the masked sibling's ~0.99.
    assert int(result.positive_percentiles_uint8[0].item()) == int(_uint8_bytes([0.20])[0].item())


def test_uint16_cap_boundary_65536_passes_65537_raises(tmp_path: Path) -> None:
    mask_ok = FeatureMask(FeatureMaskMode.whitelist, (0, 1))
    clusters_ok = _build_clusters(tmp_path, [[0, 1]], mask_ok, 65_536, name="ok.json")
    pipeline_ok = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(65_536, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask_ok,
    )
    result = select_clustered_segment(
        t.tensor([0], dtype=t.int64),
        _uint8_bytes([0.5]),
        clusters=clusters_ok,
        pipeline=pipeline_ok,
    )
    assert result.positive_ids.tolist() == [0]

    mask_over = FeatureMask(FeatureMaskMode.whitelist, (0, 1))
    clusters_over = _build_clusters(tmp_path, [[0, 1]], mask_over, 65_537, name="over.json")
    pipeline_over = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(65_537, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask_over,
    )
    with pytest.raises(FeatureClustersError, match="65,536"):
        select_clustered_segment(
            t.tensor([0], dtype=t.int64),
            _uint8_bytes([0.5]),
            clusters=clusters_over,
            pipeline=pipeline_over,
        )


def test_mask_mismatch_raises(tmp_path: Path) -> None:
    n = 16
    cluster_mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    pipeline_mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    clusters = _build_clusters(tmp_path, [[3, 4]], cluster_mask, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=pipeline_mask,
    )
    with pytest.raises(FeatureClustersError, match="does not match"):
        select_clustered_segment(
            t.tensor([3], dtype=t.int64),
            _uint8_bytes([0.5]),
            clusters=clusters,
            pipeline=pipeline,
        )


def test_delta_bundle_that_cannot_reduce_to_null_raises(tmp_path: Path) -> None:
    """A non-null delta bundle raises unless it provably reduces to ``null``.

    Previously ANY delta bundle raised here, and this test used
    ``splice_counterfactual`` (whose ``neg_delta_ratio`` defaults to ``1.0``) to
    show it. That exact configuration is now ACCEPTED: it emits a delta only for
    segments carrying a counterfactual reference, which clustered evaluation cannot
    produce, so dropping it is a no-op. Rejecting it made clustered evaluation
    impossible against every clustered checkpoint that exists.

    ``prev_segment`` emits a delta for EVERY segment, so it can never reduce to
    absolute-percentile conditioning and still raises — which is what this test now
    pins. Both directions of the accepted slice, including the reverse-direction
    rejections, are covered in ``test_clustered_selection_delta_slice.py``.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(mode=DeltaMode.prev_segment),
            percentile_table=t.zeros(n, 99, dtype=t.float64),
        ),
        feature_mask=None,
    )
    with pytest.raises(FeatureClustersError, match="produces a delta for"):
        select_clustered_segment(
            t.tensor([1], dtype=t.int64),
            _uint8_bytes([0.5]),
            clusters=clusters,
            pipeline=pipeline,
        )


def test_non_uint8_percentiles_raise(tmp_path: Path) -> None:
    n = 16
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    with pytest.raises(FeatureClustersError, match="uint8"):
        select_clustered_segment(
            t.tensor([1], dtype=t.int64),
            t.tensor([0.5], dtype=t.float32),
            clusters=clusters,
            pipeline=pipeline,
        )


def test_duplicate_raw_ids_raise(tmp_path: Path) -> None:
    n = 16
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    with pytest.raises(FeatureClustersError, match="unique"):
        select_clustered_segment(
            t.tensor([1, 1], dtype=t.int64),
            _uint8_bytes([0.5, 0.6]),
            clusters=clusters,
            pipeline=pipeline,
        )


# ---------------------------------------------------------------------------
# Step 3: mutation traps
# ---------------------------------------------------------------------------


def test_mutation_trap_mask_applied_before_cluster_consolidation(
    tmp_path: Path,
) -> None:
    """Kills the mutation: consolidate_segment() invoked before feature_mask
    .apply_to_positive() (mask/cluster call order swapped).

    A masked id must be stripped BEFORE reaching consolidation. Empirically
    verified (see task-a5-report.md): the correct order returns [1, 5]
    (masked id 3 silently excluded); calling consolidate_segment directly on
    the unmasked ids -- what the mutation would do -- raises
    FeatureClustersError from the consolidation input guard
    ("cluster input contains mask-disallowed feature ID 3") instead of
    returning a valid selection.
    """
    n = 8
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    clusters = _build_clusters(tmp_path, [[1, 2]], mask, n)
    raw_ids = t.tensor([1, 3, 5], dtype=t.int64)
    pcts = _uint8_bytes([0.5, 0.99, 0.4])  # masked id 3 has the highest pct
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.01),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )

    result = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)
    assert result.positive_ids.tolist() == [1, 5]
    assert 3 not in result.positive_ids.tolist()

    # The mutation's own effect, isolated: consolidating unmasked ids raises.
    with pytest.raises(FeatureClustersError, match="apply the fully resolved FeatureMask"):
        clusters.consolidate_segment(raw_ids, t.tensor([0.5, 0.99, 0.4]))


def test_mutation_trap_amax_not_sum_or_mean(tmp_path: Path) -> None:
    """Kills the mutation: cluster percentile reduction uses sum or mean
    instead of amax.

    The amax reduction itself is PRE-EXISTING behavior of
    ``FeatureClusterConsolidation.consolidate_segment`` (feature_clusters.py,
    ``channel.new_zeros(...).scatter_reduce(..., reduce="amax", ...)``), not
    something ``select_clustered_segment`` (this module) implements. This
    test pins that reduction only as CONSUMED BY the shared selector: a
    regression that routed a different reduction through this call site
    would be caught here, even though the reduction's own correctness is
    feature_clusters.py's responsibility, not this module's.

    Cluster [5, 9], both survive. id 5 -> pct 0.90 (uint8 230), id 9 -> pct
    0.10 (uint8 26). amax=0.9020 -> uint8 230. sum=1.0039 (clamped to 1.0)
    -> uint8 255. mean=0.5020 -> uint8 128. All three distinct (empirically
    verified). Only 230 is correct.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[5, 9]], None, n)
    raw_ids = t.tensor([5, 9], dtype=t.int64)
    pcts = _uint8_bytes([0.90, 0.10])
    decoded = uint8_to_percentile(pcts)
    amax_uint8 = int(normalized_percentile_to_uint8(decoded.max().unsqueeze(0))[0].item())
    sum_uint8 = int(
        normalized_percentile_to_uint8(decoded.sum().clamp(max=1.0).unsqueeze(0))[0].item()
    )
    mean_uint8 = int(normalized_percentile_to_uint8(decoded.mean().unsqueeze(0))[0].item())
    assert len({amax_uint8, sum_uint8, mean_uint8}) == 3, "probe values must be distinct"

    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.05),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    result = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)
    assert result.positive_ids.tolist() == [5]
    actual_uint8 = int(result.positive_percentiles_uint8[0].item())
    assert actual_uint8 == amax_uint8
    assert actual_uint8 != sum_uint8
    assert actual_uint8 != mean_uint8


def test_mutation_trap_positive_sampling_precedes_negative_sampling(
    tmp_path: Path,
) -> None:
    """Kills the mutation: sample_inactive_features_batched() called before
    sample_features_batched() (negatives drawn first in the shared RNG
    stream).

    Same seed, same inputs: drawing negatives FIRST consumes the generator
    differently than the real positive-first order, so its negative_ids
    differ from the real call's; a positive-first replica matches exactly
    (empirically verified in task-a5-report.md).
    """
    n = 16
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(n)))
    clusters = _build_clusters(tmp_path, [[14, 15]], mask, n)
    table = t.zeros(n, 99, dtype=t.float64)
    raw_ids = t.tensor([1, 2, 3], dtype=t.int64)
    pcts = _uint8_bytes([0.2, 0.5, 0.8])
    pos_sel = SamplingSelection(
        distribution=Distribution.normal,
        mean=2.0,
        std=0.01,
        min_k=2,
        max_k=3,
        mode=SamplingMode.top_k_percentile,
    )
    neg_sel = NegativeSamplingSelection(
        distribution=Distribution.normal,
        mean=3.0,
        std=0.01,
        min_k=3,
        max_k=4,
    )
    seed = 999

    rng_actual = t.Generator().manual_seed(seed)
    pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(selection=pos_sel, percentile_table=table, rng=rng_actual),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=neg_sel, n_sae_features=n, rng=rng_actual, feature_mask=mask
        ),
        effective_buckets=4,
        feature_mask=mask,
    )
    actual = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)

    # The mutation's effect, isolated: negatives drawn FIRST on an
    # identically-seeded fresh generator.
    rng_wrong_order = t.Generator().manual_seed(seed)
    wrong_order_negative = sample_inactive_features_batched(
        [raw_ids], n, neg_sel, rng_wrong_order, feature_mask=clusters.effective_negative_mask()
    )
    assert wrong_order_negative[0].tolist() != actual.negative_ids.tolist()

    # The correct order, replicated independently: positives first.
    rng_replica = t.Generator().manual_seed(seed)
    sample_features_batched(
        all_feature_ids=raw_ids,
        all_feature_mags=t.zeros_like(raw_ids, dtype=t.float32),
        segment_sizes=[3],
        percentile_table=table,
        config=pos_sel,
        rng=rng_replica,
        precomputed_percentiles=uint8_to_percentile(pcts),
    )
    replica_negative = sample_inactive_features_batched(
        [raw_ids], n, neg_sel, rng_replica, feature_mask=clusters.effective_negative_mask()
    )
    assert replica_negative[0].tolist() == actual.negative_ids.tolist()


def test_mutation_trap_sampler_order_not_cosmetically_sorted(tmp_path: Path) -> None:
    """Kills the mutation: positive_ids sorted ascending by id after
    sampling (instead of preserving the sampler's own score-ranked order).

    ids [1,2,3,4,5] with percentiles [.10,.90,.30,.95,.50], top_k_percentile
    mode, seed=7: the real sampler returns ids ranked by percentile
    descending -- [4, 2, 5, 3] (empirically verified) -- which is NOT
    ascending-by-id ([2, 3, 4, 5]). A cosmetic post-hoc sort would silently
    turn one into the other.
    """
    n = 16
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(n)))
    clusters = _build_clusters(tmp_path, [[14, 15]], mask, n)
    ids = t.tensor([1, 2, 3, 4, 5], dtype=t.int64)
    pcts = _uint8_bytes([0.10, 0.90, 0.30, 0.95, 0.50])
    table = t.arange(1, 100, dtype=t.float64).unsqueeze(0).expand(n, -1).clone()
    rng = t.Generator().manual_seed(7)
    pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=4,
                max_k=5,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=table,
            rng=rng,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )
    result = select_clustered_segment(ids, pcts, clusters=clusters, pipeline=pipeline)
    assert result.positive_ids.tolist() == [4, 2, 5, 3]
    assert result.positive_ids.tolist() != sorted(result.positive_ids.tolist())


def test_mutation_trap_own_percentile_wins_not_raw_magnitude(tmp_path: Path) -> None:
    """Kills the mutation: representative selection (or its reported
    percentile) is driven by a raw activation magnitude rather than each
    member's OWN per-feature percentile rank.

    Cluster [1, 2]. id 1's narrative "raw magnitude" is larger, but on
    id 1's own percentile row that magnitude is unremarkable (p10); id 2's
    smaller "raw magnitude" is exceptional on id 2's own row (p99). Since
    select_clustered_segment only ever sees pre-computed percentile_uint8
    (no raw-magnitude channel reaches it at all), id 2 -- the higher
    percentile -- must win, matching
    test_converter_max_pools_member_percentiles_not_raw_magnitudes on the
    oracle (steering.eval.context_cache_build wrapper, deferred to D2 --
    Task 3 Step 4 trim table), verified directly against the shared
    selector itself here instead.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    raw_ids = t.tensor([1, 2], dtype=t.int64)
    # "id 1: bigger raw magnitude, but p10 on its own row" vs
    # "id 2: smaller raw magnitude, but p99 on its own row".
    pcts = _uint8_bytes([0.10, 0.99])
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.01),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    result = select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)
    assert result.positive_ids.tolist() == [1]  # id 1 is the representative (min id)
    assert int(result.positive_percentiles_uint8[0].item()) == int(_uint8_bytes([0.99])[0].item())


# ---------------------------------------------------------------------------
# Step 4: review fix batch -- guard-coverage tests for every previously
# untested raise (lens-3 Important #1), the gradient_join regression probe
# (lens-3 Important #2 / lens-1 Important), following the same "existing 6
# pytest.raises(FeatureClustersError, match=...)" pattern established above.
# ---------------------------------------------------------------------------


def _threshold_pipeline_for_guard_tests(
    n: int, *, effective_buckets: int = 4
) -> FeaturePipelineConfig:
    """Minimal valid threshold pipeline shared by the _validate_inputs /
    _threshold_select / main-body guard tests below -- only the field(s)
    under test differ from this baseline in each test."""
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=effective_buckets,
        feature_mask=None,
    )


def test_validate_inputs_rejects_non_cpu_device(tmp_path: Path) -> None:
    """_validate_inputs (:68): a non-CPU raw_feature_ids tensor is rejected
    before any cluster/selection logic runs. Uses torch's "meta" device (no
    GPU required, empirically verified to yield device.type == "meta") only
    to trip the ``.device.type != "cpu"`` guard -- no values are ever read
    off it since the function raises immediately."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = _threshold_pipeline_for_guard_tests(n)
    raw_ids = t.empty(1, dtype=t.int64, device="meta")
    pcts = _uint8_bytes([0.5])
    with pytest.raises(FeatureClustersError, match="requires CPU input tensors"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_validate_inputs_rejects_non_1d_shapes(tmp_path: Path) -> None:
    """_validate_inputs (:72): a 2-D raw_feature_ids tensor is rejected even
    though percentile_uint8 itself stays 1-D (the two checks are ORed)."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = _threshold_pipeline_for_guard_tests(n)
    raw_ids = t.zeros(2, 2, dtype=t.int64)  # 2-D, not 1-D
    pcts = _uint8_bytes([0.5, 0.5])
    with pytest.raises(FeatureClustersError, match="one-dimensional"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_validate_inputs_rejects_shape_misalignment(tmp_path: Path) -> None:
    """_validate_inputs (:76): differing raw_feature_ids/percentile_uint8
    lengths are rejected once both are already confirmed CPU and 1-D."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = _threshold_pipeline_for_guard_tests(n)
    raw_ids = t.tensor([1, 2, 3], dtype=t.int64)
    pcts = _uint8_bytes([0.5, 0.6])  # length 2, not 3
    with pytest.raises(FeatureClustersError, match="must align"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_validate_inputs_rejects_non_integer_feature_id_dtype(tmp_path: Path) -> None:
    """_validate_inputs (:82): a floating-point raw_feature_ids tensor is
    rejected (same length as percentile_uint8, so only dtype differs)."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = _threshold_pipeline_for_guard_tests(n)
    raw_ids = t.tensor([1.0, 2.0], dtype=t.float32)
    pcts = _uint8_bytes([0.5, 0.6])
    with pytest.raises(FeatureClustersError, match="integer dtype"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_validate_inputs_rejects_effective_buckets_below_one(tmp_path: Path) -> None:
    """_validate_inputs (:111): effective_buckets=0 is rejected once every
    earlier check (device/ndim/shape/dtype/uint8/cap/delta/mask-match) has
    passed."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = _threshold_pipeline_for_guard_tests(n, effective_buckets=0)
    raw_ids = t.tensor([1], dtype=t.int64)
    pcts = _uint8_bytes([0.5])
    with pytest.raises(FeatureClustersError, match="at least one effective positive bucket"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_validate_inputs_rejects_out_of_range_feature_id(tmp_path: Path) -> None:
    """_validate_inputs (:117): a raw feature ID >= n_sae_features is
    rejected once every earlier (device/shape/dtype/bucket) check passes."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = _threshold_pipeline_for_guard_tests(n)
    raw_ids = t.tensor([n], dtype=t.int64)  # == n_sae_features -> out of range
    pcts = _uint8_bytes([0.5])
    with pytest.raises(FeatureClustersError, match="out of range"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_threshold_select_rejects_raw_magnitude_threshold_type(tmp_path: Path) -> None:
    """_threshold_select (:152): a raw-magnitude ThresholdType (segment_topk)
    has no percentile-space meaning for packed clustered selection --
    clustered packed rows only ever carry percentile bytes, never raw
    magnitudes, so only feature_percentile / feature_percentile_segment_topk
    are supported here."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, 3),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    raw_ids = t.tensor([5], dtype=t.int64)  # not part of any cluster
    pcts = _uint8_bytes([0.5])
    with pytest.raises(FeatureClustersError, match="cannot apply raw-magnitude threshold"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


class _BogusSelectionConfig:
    """Neither ThresholdSelectionConfig nor SamplingBundle -- a minimal
    stand-in used only to reach select_clustered_segment's final internal
    ``else`` branch. No supported entry point can construct
    FeaturePipelineConfig.selection_config this way (build_feature_pipeline
    only ever produces the two real shapes) -- the plain dataclass simply
    does not enforce the annotation at runtime, so this exercises the
    guard's own internal-consistency check directly."""


def test_select_clustered_segment_rejects_unsupported_selection_config(
    tmp_path: Path,
) -> None:
    """Main body (:291): a selection_config that is neither
    ThresholdSelectionConfig nor SamplingBundle is rejected once masking and
    cluster consolidation have both produced a non-empty candidate segment
    (an empty segment would short-circuit before ever reaching the
    dispatch)."""
    n = 8
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = FeaturePipelineConfig(
        selection_config=_BogusSelectionConfig(),  # ty: ignore[invalid-argument-type]
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    raw_ids = t.tensor([5], dtype=t.int64)  # not part of any cluster
    pcts = _uint8_bytes([0.5])
    with pytest.raises(FeatureClustersError, match="unsupported clustered selection config"):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)


def test_selected_percentiles_requires_exactly_one_candidate_match() -> None:
    """_selected_percentiles (:170): every selected id must have EXACTLY one
    matching row in the candidate (post-consolidation) tensors -- both zero
    matches (id absent from the candidates) and more than one match
    (duplicated candidate id) raise. Reached directly with crafted
    candidate tensors; no clusters/pipeline fixture is needed since this
    helper only ever sees already-consolidated, already-validated data."""
    candidate_ids = t.tensor([10, 20, 20], dtype=t.int64)  # 20 is duplicated
    candidate_percentiles = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)

    with pytest.raises(FeatureClustersError, match="expected exactly one"):
        _selected_percentiles(
            selected_ids=t.tensor([99], dtype=t.int64),  # absent -> 0 matches
            candidate_ids=candidate_ids,
            candidate_percentiles=candidate_percentiles,
        )

    with pytest.raises(FeatureClustersError, match="expected exactly one"):
        _selected_percentiles(
            selected_ids=t.tensor([20], dtype=t.int64),  # duplicated -> 2 matches
            candidate_ids=candidate_ids,
            candidate_percentiles=candidate_percentiles,
        )


def _assert_result_fixture(
    tmp_path: Path, *, mask: FeatureMask | None = None
) -> tuple[FeatureClusterConsolidation, FeaturePipelineConfig]:
    """Shared minimal clusters/pipeline for the _assert_result postcondition
    tests below: n=8, one 2-member cluster [2, 3] (2 is the representative,
    3 is consolidated away), effective_buckets=4. Each test below builds its
    own hand-crafted ClusteredSegmentSelection violating exactly one
    _assert_result invariant against this shared baseline."""
    n = 8
    clusters = _build_clusters(tmp_path, [[2, 3]], mask, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )
    return clusters, pipeline


def test_assert_result_rejects_wrong_dtypes(tmp_path: Path) -> None:
    """_assert_result: each of the four output tensors has its own required
    dtype (positive_ids/negative_ids/positive_buckets int64, percentiles
    uint8); each is independently enforced, in this declared order."""
    clusters, pipeline = _assert_result_fixture(tmp_path)

    def make(**overrides: t.Tensor) -> ClusteredSegmentSelection:
        fields: dict[str, t.Tensor] = {
            "positive_ids": t.tensor([2, 5], dtype=t.int64),
            "positive_buckets": t.tensor([1, 2], dtype=t.int64),
            "positive_percentiles_uint8": t.tensor([10, 20], dtype=t.uint8),
            "negative_ids": t.tensor([6], dtype=t.int64),
        }
        fields.update(overrides)
        return ClusteredSegmentSelection(**fields)

    with pytest.raises(FeatureClustersError, match="positive IDs must use int64"):
        _assert_result(
            make(positive_ids=t.tensor([2, 5], dtype=t.int32)),
            clusters=clusters,
            pipeline=pipeline,
        )
    with pytest.raises(FeatureClustersError, match="negative IDs must use int64"):
        _assert_result(
            make(negative_ids=t.tensor([6], dtype=t.int32)),
            clusters=clusters,
            pipeline=pipeline,
        )
    with pytest.raises(FeatureClustersError, match="positive buckets must use int64"):
        _assert_result(
            make(positive_buckets=t.tensor([1, 2], dtype=t.int32)),
            clusters=clusters,
            pipeline=pipeline,
        )
    with pytest.raises(FeatureClustersError, match="positive percentiles must use uint8"):
        _assert_result(
            make(positive_percentiles_uint8=t.tensor([10, 20], dtype=t.int64)),
            clusters=clusters,
            pipeline=pipeline,
        )


def test_assert_result_rejects_non_1d_outputs(tmp_path: Path) -> None:
    """_assert_result: any output tensor with ndim != 1 is rejected (checked
    before the shape-alignment check below), even though positive_buckets/
    positive_percentiles_uint8/negative_ids are otherwise well-formed 1-D."""
    clusters, pipeline = _assert_result_fixture(tmp_path)
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([[2], [5]], dtype=t.int64),  # shape (2, 1)
        positive_buckets=t.tensor([1, 2], dtype=t.int64),
        positive_percentiles_uint8=t.tensor([10, 20], dtype=t.uint8),
        negative_ids=t.tensor([6], dtype=t.int64),
    )
    with pytest.raises(FeatureClustersError, match="must be 1-D"):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


def test_assert_result_rejects_misaligned_positive_channels(tmp_path: Path) -> None:
    """_assert_result: positive_ids/positive_buckets/positive_percentiles_uint8
    must all share one shape; a length mismatch on any one of them (all
    still 1-D, so the ndim check above passes) is rejected."""
    clusters, pipeline = _assert_result_fixture(tmp_path)
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([2, 5], dtype=t.int64),
        positive_buckets=t.tensor([1], dtype=t.int64),  # length 1, not 2
        positive_percentiles_uint8=t.tensor([10, 20], dtype=t.uint8),
        negative_ids=t.tensor([6], dtype=t.int64),
    )
    with pytest.raises(FeatureClustersError, match="misaligned"):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


def test_assert_result_rejects_duplicate_positive_ids(tmp_path: Path) -> None:
    """_assert_result: a repeated id within positive_ids is rejected."""
    clusters, pipeline = _assert_result_fixture(tmp_path)
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([2, 2], dtype=t.int64),
        positive_buckets=t.tensor([1, 1], dtype=t.int64),
        positive_percentiles_uint8=t.tensor([10, 10], dtype=t.uint8),
        negative_ids=t.tensor([6], dtype=t.int64),
    )
    with pytest.raises(FeatureClustersError, match="positive selection contains duplicates"):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


def test_assert_result_rejects_duplicate_negative_ids(tmp_path: Path) -> None:
    """_assert_result: a repeated id within negative_ids is rejected."""
    clusters, pipeline = _assert_result_fixture(tmp_path)
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([2, 5], dtype=t.int64),
        positive_buckets=t.tensor([1, 2], dtype=t.int64),
        positive_percentiles_uint8=t.tensor([10, 20], dtype=t.uint8),
        negative_ids=t.tensor([6, 6], dtype=t.int64),
    )
    with pytest.raises(FeatureClustersError, match="negative selection contains duplicates"):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


def test_assert_result_rejects_positive_negative_overlap(tmp_path: Path) -> None:
    """_assert_result: an id appearing in both positive_ids and negative_ids
    is rejected (each individually duplicate-free, but not disjoint)."""
    clusters, pipeline = _assert_result_fixture(tmp_path)
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([2, 5], dtype=t.int64),
        positive_buckets=t.tensor([1, 2], dtype=t.int64),
        positive_percentiles_uint8=t.tensor([10, 20], dtype=t.uint8),
        negative_ids=t.tensor([5], dtype=t.int64),  # overlaps positive id 5
    )
    with pytest.raises(FeatureClustersError, match="must be disjoint"):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


@pytest.mark.parametrize("side", ["positive", "negative"])
def test_assert_result_rejects_mask_disallowed_id(tmp_path: Path, side: str) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (7,))
    clusters, pipeline = _assert_result_fixture(tmp_path, mask=mask)
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([7 if side == "positive" else 2], dtype=t.int64),
        positive_buckets=t.tensor([1], dtype=t.int64),
        positive_percentiles_uint8=t.tensor([10], dtype=t.uint8),
        negative_ids=t.tensor([7] if side == "negative" else [], dtype=t.int64),
    )
    with pytest.raises(
        FeatureClustersError,
        match=f"clustered segment {side} selection contains mask-disallowed feature ID 7$",
    ):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


@pytest.mark.parametrize("mode", [FeatureMaskMode.blacklist, FeatureMaskMode.whitelist])
def test_threshold_result_uses_representative_mask_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: FeatureMaskMode
) -> None:
    mask = FeatureMask(mode, (7,) if mode is FeatureMaskMode.blacklist else (2, 3, 5))
    clusters, pipeline = _assert_result_fixture(tmp_path, mask=mask)

    def duplicate_mask_check(self: FeatureMask, feature_ids: t.Tensor) -> None:
        pytest.fail("representative checks already enforce the result mask")

    monkeypatch.setattr(FeatureMask, "assert_no_masked", duplicate_mask_check)
    result = select_clustered_segment(
        t.tensor([2, 3, 5, 7], dtype=t.int64),
        t.tensor([10, 20, 30, 40], dtype=t.uint8),
        clusters=clusters,
        pipeline=pipeline,
    )
    assert result.positive_ids.tolist() == [2, 5]
    assert result.positive_percentiles_uint8.tolist() == [20, 30]
    assert result.negative_ids.numel() == 0


def test_assert_result_rejects_non_representative_positive_id(tmp_path: Path) -> None:
    """_assert_result: a positive id that does not map to itself under
    clusters.rep_of is rejected by assert_representatives_only's THIRD
    sub-check.

    Real cluster-building (FeatureClusterConsolidation.build) never leaves a
    non-representative id un-consolidated: rep_of[id] != id always implies
    consolidated_away[id] is True there (both are set together, in the same
    cluster iteration), so the SECOND sub-check (consolidated-away) would
    always fire first on any id a normal fixture could produce. This test
    rigs rep_of directly via dataclasses.replace on an otherwise-normal
    FeatureClusterConsolidation to isolate the THIRD sub-check from the
    second one it would otherwise always be shadowed by."""
    clusters, pipeline = _assert_result_fixture(tmp_path)
    assert not bool(clusters.consolidated_away[4])  # id 4 is untouched by [[2, 3]]
    rigged_rep_of = clusters.rep_of.clone()
    rigged_rep_of[4] = 2  # id 4 now claims representative 2, without going "away"
    rigged_clusters = replace(clusters, rep_of=rigged_rep_of)

    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([4], dtype=t.int64),
        positive_buckets=t.tensor([1], dtype=t.int64),
        positive_percentiles_uint8=t.tensor([10], dtype=t.uint8),
        negative_ids=t.empty(0, dtype=t.int64),
    )
    with pytest.raises(FeatureClustersError, match="non-representative feature ID 4"):
        _assert_result(result, clusters=rigged_clusters, pipeline=pipeline)


def test_assert_result_rejects_bucket_out_of_range(tmp_path: Path) -> None:
    """_assert_result: a positive bucket >= pipeline.effective_buckets is
    rejected -- the final postcondition, reached only once every earlier
    channel/representative/mask check has passed."""
    clusters, pipeline = _assert_result_fixture(tmp_path)  # effective_buckets=4
    result = ClusteredSegmentSelection(
        positive_ids=t.tensor([2], dtype=t.int64),
        positive_buckets=t.tensor([4], dtype=t.int64),  # >= effective_buckets
        positive_percentiles_uint8=t.tensor([10], dtype=t.uint8),
        negative_ids=t.empty(0, dtype=t.int64),
    )
    with pytest.raises(FeatureClustersError, match="outside the effective bucket range"):
        _assert_result(result, clusters=clusters, pipeline=pipeline)


def test_clustered_selection_rejects_truthy_gradient_join(tmp_path: Path) -> None:
    """select_features_for_shard's clustered arm (scripts/preprocess/
    precompute_selected_features.py) raises immediately if a truthy
    gradient_join reaches it alongside a real FeatureClusterConsolidation.

    In production this combination is meant to be refused earlier, by
    assert_feature_clusters_precompute_supported's startup gate
    (``grad_spec is not None`` -> "feature_clusters cannot be combined with
    feature_selection.gradient"). But select_features_for_shard is also
    directly callable on its own (as here, bypassing that gate entirely) --
    the review flagged that its OWN internal isinstance-gated guard
    ("internal error: clustered ordinary selection reached the
    gradient-attribution path after the startup support gate") had no
    direct regression test anywhere in the suite.

    ``gradient_join`` only needs a ``.total_segments`` attribute matching
    the base shard's segment count (checked before the per-segment loop);
    the isinstance(feature_clusters, FeatureClusterConsolidation) +
    gradient_join-is-not-None check fires and raises BEFORE
    ``gradient_join.scores_for_segment(...)`` is ever called, so a minimal
    ``SimpleNamespace`` stand-in is enough here -- no real
    ShardGradientJoin/GradScoreJoiner/provenance construction is needed."""
    n = 16
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    shard_path = save_packed_row_shard(
        _shard_from_segments([([1], [128])]), base_dir / "shard_00000"
    )
    out_dir = tmp_path / "sel"
    out_dir.mkdir()
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )
    fake_gradient_join = SimpleNamespace(total_segments=1)

    with pytest.raises(FeatureClustersError, match="reached the gradient-attribution path"):
        select_features_for_shard(
            shard_path,
            pipeline,
            out_dir,
            "shard_00000",
            gradient_join=fake_gradient_join,  # ty: ignore[invalid-argument-type]
            feature_clusters=clusters,
        )


def test_post_consolidation_empty_raises_internal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """select_clustered_segment (:288): a segment that is STILL non-empty
    after mask application but comes back empty from
    ``clusters.consolidate_segment`` is UNREACHABLE for legitimate input --
    ``consolidate_segment`` maps a non-empty tensor through its own
    ``unique()``-backed representative assignment (feature_clusters.py),
    which cannot turn a non-empty input into an empty output
    (review-verified). Unlike the two EARLIER empty returns in this function
    -- both EXPECTED-DOMAIN data, see ``_empty_result``'s docstring
    (raw-empty input, post-mask-empty) -- this third case is therefore a
    hard internal-invariant violation, not ordinary empty-segment data, and
    must raise rather than silently returning ``_empty_result()``'s same
    empty shape.

    Reached by monkeypatching FeatureClusterConsolidation.consolidate_segment
    AT CLASS LEVEL to unconditionally return empty tensors -- no legitimate
    clusters/mask/input combination can trigger this branch on its own, so a
    real (non-monkeypatched) trigger is not constructible."""
    n = 16
    clusters = _build_clusters(tmp_path, [[1, 2]], None, n)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(n, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=None,
    )

    def _fake_consolidate_segment(
        self: FeatureClusterConsolidation,
        feature_ids: t.Tensor,
        percentiles: t.Tensor,
        *aligned_scores: t.Tensor,
    ) -> tuple[t.Tensor, ...]:
        return t.empty(0, dtype=t.int64), t.empty(0, dtype=t.float32)

    monkeypatch.setattr(
        FeatureClusterConsolidation, "consolidate_segment", _fake_consolidate_segment
    )

    raw_ids = t.tensor([1], dtype=t.int64)  # non-empty, mask-surviving (no mask)
    pcts = _uint8_bytes([0.5])
    with pytest.raises(
        FeatureClustersError,
        match="cluster consolidation emptied a segment that still had mask-surviving features",
    ):
        select_clustered_segment(raw_ids, pcts, clusters=clusters, pipeline=pipeline)
