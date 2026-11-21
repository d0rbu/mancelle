"""Cluster-safe selected-feature materialization for one packed segment.

This module is intentionally narrow: it consumes the same packed percentile
bytes as clustered training preselection and owns the mask -> consolidation ->
positive selection -> negative selection ordering.  Evaluation code that must
reproduce the training currency calls this helper instead of maintaining a
second approximation of the selection algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from loguru import logger

from errors import FeatureClustersError
from steering.conditional.delta import DeltaMode
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    SamplingBundle,
    ThresholdSelectionConfig,
)
from steering.conditional.percentile_table import (
    magnitude_to_percentile,
    normalized_percentile_to_uint8,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    sample_features_batched,
    sample_inactive_features_batched,
)
from steering.conditional.thresholding import (
    ThresholdType,
    select_from_packed_percentiles,
)

_PERCENTILE_ONLY_THRESHOLD_TYPES: frozenset[ThresholdType] = frozenset(
    {
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    }
)


@dataclass(frozen=True)
class ClusteredSegmentSelection:
    """Ordered positive and negative conditioning values for one segment."""

    positive_ids: t.Tensor
    positive_buckets: t.Tensor
    positive_percentiles_uint8: t.Tensor
    negative_ids: t.Tensor


def _empty_result() -> ClusteredSegmentSelection:
    """Expected empty selection for raw-empty or post-mask-empty segments.

    Both cases return before positive or negative sampling consumes RNG.
    Consolidation cannot empty a non-empty candidate tensor; that case must
    raise rather than return an expected empty result.
    """
    return ClusteredSegmentSelection(
        positive_ids=t.empty(0, dtype=t.int64),
        positive_buckets=t.empty(0, dtype=t.int64),
        positive_percentiles_uint8=t.empty(0, dtype=t.uint8),
        negative_ids=t.empty(0, dtype=t.int64),
    )


def _validate_inputs(
    raw_feature_ids: t.Tensor,
    percentile_uint8: t.Tensor,
    *,
    clusters: FeatureClusterConsolidation,
    pipeline: FeaturePipelineConfig,
) -> tuple[t.Tensor, t.Tensor]:
    if raw_feature_ids.device.type != "cpu" or percentile_uint8.device.type != "cpu":
        raise FeatureClustersError(
            "clustered segment selection requires CPU input tensors"
        )
    if raw_feature_ids.ndim != 1 or percentile_uint8.ndim != 1:
        raise FeatureClustersError(
            "clustered segment selection inputs must be one-dimensional"
        )
    if raw_feature_ids.shape != percentile_uint8.shape:
        raise FeatureClustersError(
            "clustered segment selection feature IDs and percentile bytes must "
            f"align, got {tuple(raw_feature_ids.shape)} and "
            f"{tuple(percentile_uint8.shape)}"
        )
    if (
        raw_feature_ids.dtype is t.bool
        or raw_feature_ids.is_floating_point()
        or raw_feature_ids.is_complex()
    ):
        raise FeatureClustersError(
            "clustered segment selection feature IDs must use an integer dtype, "
            f"got {raw_feature_ids.dtype}"
        )
    if percentile_uint8.dtype is not t.uint8:
        raise FeatureClustersError(
            "clustered segment selection percentiles must be the canonical uint8 "
            f"training currency, got {percentile_uint8.dtype}"
        )
    if clusters.n_sae_features > 65_536:
        raise FeatureClustersError(
            "clustered selected-feature storage is uint16 and is unsafe for "
            f"n_sae_features={clusters.n_sae_features}; the supported maximum is "
            "65,536"
        )
    if pipeline.delta_bundle is not None:
        # Evaluation has no splice reference, so splice_counterfactual uses
        # ordinary absolute-percentile conditioning. With neg_delta_ratio=1.0,
        # its empty delta pool falls back entirely to ordinary negative sampling.
        # Partial ratios are rejected rather than reinterpreted as this fallback.
        # prev_segment and cum_max produce deltas for every segment and cannot
        # reduce to the ordinary path.
        delta_config = pipeline.delta_bundle.config
        if delta_config.mode is not DeltaMode.splice_counterfactual:
            raise FeatureClustersError(
                "ordinary clustered segment selection requires feature_delta=null "
                "or mode=splice_counterfactual with neg_delta_ratio=1.0; got "
                f"mode={delta_config.mode.value!r}, which produces a delta for "
                "every segment and therefore cannot reduce to absolute-percentile "
                "conditioning at evaluation time"
            )
        if delta_config.neg_delta_ratio != 1.0:
            raise FeatureClustersError(
                "ordinary clustered segment selection accepts "
                "mode=splice_counterfactual only with neg_delta_ratio=1.0 (all "
                "negatives from the delta pool, which is empty without splice "
                f"provenance); got neg_delta_ratio={delta_config.neg_delta_ratio!r}"
            )
    if pipeline.feature_mask != clusters.feature_mask:
        raise FeatureClustersError(
            "clustered segment selection pipeline mask does not match the mask "
            "used to resolve cluster representatives"
        )
    if pipeline.effective_buckets < 1:
        raise FeatureClustersError(
            "clustered segment selection requires at least one effective positive "
            "bucket"
        )

    ids = raw_feature_ids.to(t.int64)
    if ids.numel() > 0:
        invalid = (ids < 0) | (ids >= clusters.n_sae_features)
        if bool(invalid.any()):
            bad = int(ids[invalid][0].item())
            raise FeatureClustersError(
                f"clustered segment selection feature ID {bad} is out of range "
                f"[0, {clusters.n_sae_features})"
            )
        if int(t.unique(ids).numel()) != int(ids.numel()):
            raise FeatureClustersError(
                "clustered segment selection requires unique raw feature IDs"
            )
    return ids, uint8_to_percentile(percentile_uint8)


def _threshold_select(
    *,
    feature_ids: t.Tensor,
    percentiles: t.Tensor,
    selection: ThresholdSelectionConfig,
) -> t.Tensor:
    threshold_type = selection.selection.threshold_type
    threshold = selection.selection.threshold
    if threshold_type not in _PERCENTILE_ONLY_THRESHOLD_TYPES:
        supported = sorted(item.value for item in _PERCENTILE_ONLY_THRESHOLD_TYPES)
        raise FeatureClustersError(
            f"clustered packed selection cannot apply raw-magnitude threshold "
            f"{threshold_type.value!r}; supported percentile-space modes are "
            f"{supported}"
        )
    return select_from_packed_percentiles(
        feature_ids, percentiles, threshold_type, threshold
    )


def _selected_percentiles(
    *,
    selected_ids: t.Tensor,
    candidate_ids: t.Tensor,
    candidate_percentiles: t.Tensor,
) -> t.Tensor:
    values: list[float] = []
    for selected_id in selected_ids.tolist():
        matches = (candidate_ids == selected_id).nonzero(as_tuple=True)[0]
        if len(matches) != 1:
            raise FeatureClustersError(
                f"selected representative {int(selected_id)} has {len(matches)} "
                "matching consolidated percentile entries; expected exactly one"
            )
        values.append(float(candidate_percentiles[matches[0]].item()))
    return t.tensor(values, dtype=t.float32)


def _assert_result(
    result: ClusteredSegmentSelection,
    *,
    clusters: FeatureClusterConsolidation,
    pipeline: FeaturePipelineConfig,
) -> None:
    if result.positive_ids.dtype is not t.int64:
        raise FeatureClustersError("clustered positive IDs must use int64")
    if result.negative_ids.dtype is not t.int64:
        raise FeatureClustersError("clustered negative IDs must use int64")
    if result.positive_buckets.dtype is not t.int64:
        raise FeatureClustersError("clustered positive buckets must use int64")
    if result.positive_percentiles_uint8.dtype is not t.uint8:
        raise FeatureClustersError("clustered positive percentiles must use uint8")
    if not (
        result.positive_ids.ndim
        == result.positive_buckets.ndim
        == result.positive_percentiles_uint8.ndim
        == result.negative_ids.ndim
        == 1
    ):
        raise FeatureClustersError("clustered selection outputs must be 1-D")
    if not (
        result.positive_ids.shape
        == result.positive_buckets.shape
        == result.positive_percentiles_uint8.shape
    ):
        raise FeatureClustersError("clustered positive output channels are misaligned")
    if int(t.unique(result.positive_ids).numel()) != int(result.positive_ids.numel()):
        raise FeatureClustersError("clustered positive selection contains duplicates")
    if int(t.unique(result.negative_ids).numel()) != int(result.negative_ids.numel()):
        raise FeatureClustersError("clustered negative selection contains duplicates")
    if result.positive_ids.numel() and result.negative_ids.numel():
        if bool(t.isin(result.positive_ids, result.negative_ids).any()):
            raise FeatureClustersError(
                "clustered positive and negative selections must be disjoint"
            )
    clusters.assert_representatives_only(
        result.positive_ids,
        surface="clustered segment positive selection",
    )
    clusters.assert_representatives_only(
        result.negative_ids,
        surface="clustered segment negative selection",
    )
    if result.positive_buckets.numel() and bool(
        (
            (result.positive_buckets < 0)
            | (result.positive_buckets >= pipeline.effective_buckets)
        ).any()
    ):
        raise FeatureClustersError(
            "clustered positive bucket is outside the effective bucket range"
        )


def select_clustered_segment(
    raw_feature_ids: t.Tensor,
    percentile_uint8: t.Tensor,
    *,
    clusters: FeatureClusterConsolidation,
    pipeline: FeaturePipelineConfig,
) -> ClusteredSegmentSelection:
    """Materialize one cluster-aware packed segment in training RNG order."""

    feature_ids, percentiles = _validate_inputs(
        raw_feature_ids,
        percentile_uint8,
        clusters=clusters,
        pipeline=pipeline,
    )
    if feature_ids.numel() == 0:
        return _empty_result()

    if pipeline.feature_mask is not None:
        feature_ids, percentiles = pipeline.feature_mask.apply_to_positive(
            feature_ids, percentiles
        )
    if feature_ids.numel() == 0:
        return _empty_result()

    feature_ids, percentiles = clusters.consolidate_segment(
        feature_ids, percentiles
    )
    if feature_ids.numel() == 0:
        raise FeatureClustersError(
            "internal error: cluster consolidation emptied a segment that "
            "still had mask-surviving features"
        )

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
        selected_ids = (
            sampled[0][0]
            if sampled
            else t.empty(0, dtype=t.int64)
        )
    elif isinstance(selection, ThresholdSelectionConfig):
        selected_ids = _threshold_select(
            feature_ids=feature_ids,
            percentiles=percentiles,
            selection=selection,
        )
    else:
        raise FeatureClustersError(
            "unsupported clustered selection config "
            f"{type(selection).__name__}"
        )

    selected_percentiles = _selected_percentiles(
        selected_ids=selected_ids,
        candidate_ids=feature_ids,
        candidate_percentiles=percentiles,
    )
    selected_buckets = percentile_to_bucket(
        selected_percentiles,
        pipeline.effective_buckets,
    )
    selected_percentiles_uint8 = normalized_percentile_to_uint8(
        selected_percentiles
    )

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

    result = ClusteredSegmentSelection(
        positive_ids=selected_ids.to(t.int64),
        positive_buckets=selected_buckets.to(t.int64),
        positive_percentiles_uint8=selected_percentiles_uint8,
        negative_ids=negative_ids,
    )
    _assert_result(result, clusters=clusters, pipeline=pipeline)
    return result


def select_clustered_segment_from_magnitudes(
    feature_ids: t.Tensor,
    feature_magnitudes: t.Tensor,
    *,
    percentile_table: t.Tensor,
    clusters: FeatureClusterConsolidation,
    pipeline: FeaturePipelineConfig,
    source: str,
) -> ClusteredSegmentSelection:
    """Convert raw magnitudes to training currency, then select one segment.

    Callers retain the validation specific to their raw source.  This helper
    owns the one shared magnitude -> percentile rank -> uint8 -> clustered
    selection chain so live eval extraction and clustered CPU replay cannot
    drift apart.

    A masked feature can legitimately fire in an extraction corpus without
    having appeared in the independently configured statistics corpus.  Such
    an all-zero percentile-table row would make ``magnitude_to_percentile``
    raise even though the mask guarantees the feature cannot affect output.
    Drop that exact case with a loud error log.  An unmasked stat-less feature
    remains a fatal percentile-table anomaly.
    """

    score_ids = feature_ids.to(t.int64)
    score_magnitudes = feature_magnitudes.float()
    if pipeline.feature_mask is not None and score_ids.numel() > 0:
        stat_less = (percentile_table[score_ids] == 0).all(dim=1)
        masked_out = ~pipeline.feature_mask.keep_mask(score_ids)
        drop = stat_less & masked_out
        if bool(drop.any()):
            # The trailing "legitimate ... not a fatal error" clause is the
            # operator-facing half of the owner's PASS-but-LOUD ruling: without
            # it an ERROR about missing statistics reads as a broken run. Kept
            # textually identical to the pre-hoist message so the hoist is a
            # code motion in the logs too, not only in the tensors.
            logger.error(
                "{}: masked feature(s) {} fired in the eval extraction corpus "
                "but have no percentile statistics (an all-zero "
                "percentile-table row); excluding them before clustered "
                "percentile scoring since they are masked out regardless -- "
                "the extraction and statistics corpora are independently "
                "configured, so this is a legitimate (if noteworthy) mismatch, "
                "not a fatal error",
                source,
                score_ids[drop].tolist(),
            )
            keep = ~drop
            score_ids = score_ids[keep]
            score_magnitudes = score_magnitudes[keep]

    percentile_ranks = magnitude_to_percentile(
        score_ids,
        score_magnitudes,
        percentile_table,
    )
    return select_clustered_segment(
        score_ids,
        percentile_to_uint8(percentile_ranks),
        clusters=clusters,
        pipeline=pipeline,
    )


__all__ = [
    "ClusteredSegmentSelection",
    "select_clustered_segment",
    "select_clustered_segment_from_magnitudes",
]
