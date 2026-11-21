"""Threshold strategies for filtering SAE feature activations.

Four strategies control which features survive filtering:
- segment_topk: top-k features per segment by magnitude
- segment_percentile: top x% features per segment
- feature_percentile: per-feature percentile-based threshold
- feature_percentile_segment_topk: per-feature percentile filter then segment topk cap

For self-conditioned generation, do not hand-convert per-feature magnitude
thresholds to normalized scores. Pass the original-width training threshold
tensor to ``SelfCondConfig``'s selection bundle: the provider preserves its
finite/``+inf`` availability mask and derives the exact cutoff from the
explicit selection producer in continuous normalized-rank space. Full-rank
selection compares against the configured float32 percentile; raw-magnitude
selection uses this module's legacy persisted statistics key. A single idealized
decimal cutoff cannot reproduce both producers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch as t
from beartype import beartype

from steering.refined_types import (
    NonNegativeFiniteNumberLike,
    PositiveFiniteNumberLike,
    PositiveIntLike,
    UnitIntervalOpenLowerLike,
)
from steering.stats_types import PerFeatureStat

if TYPE_CHECKING:
    _IntFloatPair = tuple[int, float]
    _PerFeatureStats = dict[str, PerFeatureStat]
    _TensorPair = tuple[t.Tensor, t.Tensor]
else:
    # The artifact parser owns item-shape checks. Runtime outer containers
    # avoid beartype's global-RNG-backed element sampling.
    _IntFloatPair = tuple
    _PerFeatureStats = dict
    _TensorPair = tuple


class ThresholdType(Enum):
    segment_topk = "segment_topk"
    segment_percentile = "segment_percentile"
    feature_percentile = "feature_percentile"
    feature_percentile_segment_topk = "feature_percentile_segment_topk"


@beartype
@dataclass(frozen=True)
class ThresholdConfig:
    threshold_type: ThresholdType
    threshold: NonNegativeFiniteNumberLike  # int for topk, float for percentile


@beartype
def apply_segment_topk(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    k: PositiveIntLike,
) -> _TensorPair:
    """Keep top-k features by magnitude within a single segment."""
    assert feature_ids.shape == feature_mags.shape, (
        f"Shape mismatch: {feature_ids.shape} vs {feature_mags.shape}"
    )
    if len(feature_ids) == 0:
        return feature_ids, feature_mags
    actual_k = min(k, len(feature_ids))
    _, topk_idx = t.topk(feature_mags, actual_k)
    return feature_ids[topk_idx], feature_mags[topk_idx]


@beartype
def apply_segment_percentile(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    percentile: UnitIntervalOpenLowerLike,
) -> _TensorPair:
    """Keep top percentile% of features by magnitude within a single segment."""
    assert feature_ids.shape == feature_mags.shape, (
        f"Shape mismatch: {feature_ids.shape} vs {feature_mags.shape}"
    )
    if len(feature_ids) == 0:
        return feature_ids, feature_mags
    k = max(1, math.ceil(len(feature_ids) * percentile))
    return apply_segment_topk(feature_ids, feature_mags, k)


@beartype
def apply_feature_percentile(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    precomputed: t.Tensor,
) -> _TensorPair:
    """Filter features using precomputed per-feature percentile thresholds.

    Inclusive (``>=``) — magnitudes exactly at the per-feature threshold are
    KEPT. This is the boundary contract for all per-feature percentile
    thresholding paths.

    Audit 2026-05-05 S-07-02 fix (was ``>``).
    """
    if len(feature_ids) == 0:
        return feature_ids, feature_mags
    feature_ids_long = feature_ids.long()
    invalid = (feature_ids_long < 0) | (feature_ids_long >= len(precomputed))
    if invalid.any():
        bad = feature_ids_long[invalid]
        raise ValueError(
            f"Feature IDs {bad.tolist()[:5]} are outside valid range "
            f"[0, {len(precomputed) - 1}] for {len(precomputed)} "
            "precomputed thresholds."
        )
    thresholds = precomputed[feature_ids_long]
    mask = feature_mags.to(t.float64) >= thresholds
    return feature_ids[mask], feature_mags[mask]


@beartype
def _parse_combined_threshold(
    threshold: PositiveFiniteNumberLike,
) -> _IntFloatPair:
    """Parse a combined threshold like 2048.7 into (k=2048, percentile=0.7).

    The integer part is the segment topk k, the decimal part is the percentile.
    """
    k = int(threshold)
    percentile = round(threshold - k, 2)
    assert k > 0, (
        f"segment topk k must be positive, got {k} (from threshold={threshold})"
    )
    assert 0 < percentile < 1, (
        f"percentile must be in (0, 1), got {percentile} (from threshold={threshold})"
    )
    return k, percentile


@beartype
def extract_threshold_percentile(config: ThresholdConfig) -> float:
    """Extract the filtering percentile from a threshold config.

    Returns the per-feature percentile used for filtering (0-1 scale).
    Used to compute post-filter bucket boundaries.

    Raises:
        NotImplementedError: If the threshold type doesn't use percentile filtering.
    """
    match config.threshold_type:
        case ThresholdType.feature_percentile:
            return config.threshold
        case ThresholdType.feature_percentile_segment_topk:
            _, pct = _parse_combined_threshold(config.threshold)
            return pct
        case _:
            raise NotImplementedError(
                f"extract_threshold_percentile not supported for {config.threshold_type}"
            )


@beartype
def feature_percentile_key(config: ThresholdConfig) -> int:
    """Return the persisted statistics key used by feature thresholding.

    This owns the historical ``int(percentile * 100)`` projection used by
    :func:`build_thresholds_from_stats`.  It is intentionally *not* equivalent
    to rounding: binary-float hundredths such as ``.29``, ``.57``, and ``.58``
    select the preceding persisted key.  Changing that rule would change
    existing selection/checkpoint identities.

    The returned key is always in the statistics artifact's supported
    ``p1`` ... ``p99`` range.  Combined feature-percentile/top-k thresholds are
    parsed through :func:`_parse_combined_threshold` before applying the legacy
    projection, exactly as they are during training.
    """
    percentile = extract_threshold_percentile(config)
    key = int(percentile * 100)
    if key < 1 or key > 99:
        raise ValueError(
            "feature-percentile threshold resolves outside the persisted "
            f"p1..p99 statistics grid: threshold={config.threshold!r}, "
            f"percentile={percentile!r}, key=p{key}"
        )
    return key


@beartype
def apply_feature_percentile_segment_topk(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    precomputed: t.Tensor,
    k: PositiveIntLike,
) -> _TensorPair:
    """Filter by per-feature percentile, then cap to top-k per segment.

    Two-stage filter: first removes activations below their feature's percentile
    threshold (statistical relevance), then keeps only the top-k remaining
    features by magnitude (segment sparsity control).
    """
    # Stage 1: filter by per-feature percentile
    fids, fmags = apply_feature_percentile(feature_ids, feature_mags, precomputed)
    # Stage 2: cap to top-k by magnitude within segment
    if len(fids) == 0:
        return fids, fmags
    return apply_segment_topk(fids, fmags, k)


@beartype
def apply_threshold(
    config: ThresholdConfig,
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    precomputed: t.Tensor | None = None,
) -> _TensorPair:
    """Apply threshold filtering to a single segment's features.

    Args:
        config: Threshold configuration specifying type and parameter.
        feature_ids: 1D tensor of feature IDs for one segment.
        feature_mags: 1D tensor of feature magnitudes for one segment.
        precomputed: Precomputed per-feature threshold tensor. Required for
            feature-level threshold types.

    Returns:
        Filtered (feature_ids, feature_mags) tuple.
    """
    match config.threshold_type:
        case ThresholdType.segment_topk:
            return apply_segment_topk(feature_ids, feature_mags, int(config.threshold))
        case ThresholdType.segment_percentile:
            return apply_segment_percentile(feature_ids, feature_mags, config.threshold)
        case ThresholdType.feature_percentile:
            assert precomputed is not None, (
                "feature_percentile requires precomputed Tensor thresholds"
            )
            return apply_feature_percentile(feature_ids, feature_mags, precomputed)
        case ThresholdType.feature_percentile_segment_topk:
            assert precomputed is not None, (
                "feature_percentile_segment_topk requires precomputed Tensor thresholds"
            )
            k = int(config.threshold)
            return apply_feature_percentile_segment_topk(
                feature_ids, feature_mags, precomputed, k
            )
        case _:
            raise ValueError(f"Unknown threshold type: {config.threshold_type}")


@beartype
def select_from_packed_percentiles(
    feature_ids: t.Tensor,
    percentiles: t.Tensor,
    threshold_type: ThresholdType,
    threshold: NonNegativeFiniteNumberLike,
) -> t.Tensor:
    """Select feature IDs directly from packed-row percentile bytes.

    Packed row shards store row-aligned per-feature percentile bytes instead
    of raw activation magnitudes, so only the two threshold modes that are
    already expressed in percentile space can be evaluated against them:
    ``feature_percentile`` (a single cutoff) and
    ``feature_percentile_segment_topk`` (the same cutoff, then capped to the
    top-k survivors by percentile within the segment). This is the shared
    math consumed by both packed-percentile selection call sites (ordinary
    clustered segment selection and the precompute script's unclustered
    path) -- callers that also need the percentile of each selected feature
    look it up separately from their own candidate arrays.

    Args:
        feature_ids: 1D tensor of feature IDs for one segment.
        percentiles: 1D tensor of per-feature percentiles (0-1 scale),
            row-aligned with ``feature_ids``.
        threshold_type: Must be ``feature_percentile`` or
            ``feature_percentile_segment_topk``.
        threshold: Percentile threshold, or a combined segment-topk +
            percentile value (see :func:`_parse_combined_threshold`).

    Returns:
        Selected feature IDs. For ``feature_percentile_segment_topk``, in
        percentile-descending order (the ``t.topk`` ranking); for
        ``feature_percentile``, in ``feature_ids`` order.

    Raises:
        ValueError: If ``threshold_type`` is not one of the two supported
            percentile-space modes.
    """
    if threshold_type is ThresholdType.feature_percentile:
        return feature_ids[percentiles >= float(threshold)]
    if threshold_type is ThresholdType.feature_percentile_segment_topk:
        k, percentile_threshold = _parse_combined_threshold(float(threshold))
        keep = percentiles >= percentile_threshold
        kept_ids = feature_ids[keep]
        kept_percentiles = percentiles[keep]
        actual_k = min(k, int(kept_ids.numel()))
        _, top_idx = t.topk(kept_percentiles, actual_k)
        return kept_ids[top_idx]
    supported = sorted(
        item.value
        for item in (
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        )
    )
    raise ValueError(
        f"select_from_packed_percentiles does not support threshold_type "
        f"{threshold_type!r}; supported percentile-space modes are {supported}"
    )


@beartype
def build_thresholds_from_stats(
    per_feature_stats: _PerFeatureStats,
    n_sae_features: PositiveIntLike,
    threshold_type: ThresholdType,
    threshold: NonNegativeFiniteNumberLike,
) -> t.Tensor:
    """Build precomputed thresholds from cached feature statistics.

    Reads from feature statistics instead of raw feature tensors, avoiding
    loading all shard data into memory.

    Args:
        per_feature_stats: Dict from feature_statistics.json per_feature_stats.
        n_sae_features: Total SAE feature count.
        threshold_type: Which threshold type to build.
        threshold: Percentile threshold, or combined segment-topk + percentile.

    Returns:
        Tensor [n_sae_features] of per-feature thresholds (float64).
    """
    assert threshold_type in (
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    ), (
        f"build_thresholds_from_stats only supports feature_percentile "
        f"and feature_percentile_segment_topk, "
        f"got {threshold_type}"
    )

    pct_key = f"p{feature_percentile_key(ThresholdConfig(threshold_type, threshold))}"

    thresholds = t.full((n_sae_features,), float("inf"), dtype=t.float64)

    for fid_str, fstat in per_feature_stats.items():
        fid = int(fid_str)
        percentiles = fstat.percentiles
        assert pct_key in percentiles, (
            f"Required percentile key '{pct_key}' not found in "
            f"feature {fid_str} percentiles (available: {list(percentiles.keys())})"
        )
        thresholds[fid] = percentiles[pct_key]

    return thresholds
