"""Discretize SAE feature activations into bucket indices.

Per-feature percentile boundaries are computed from the corpus, then each
activation magnitude is mapped to a bucket via searchsorted.

Provides:
- discretize_activations: map magnitudes to bucket indices via searchsorted
- compute_percentiles: map magnitudes to percentile values via piecewise linear interpolation
- percentile_to_bucket: map percentile values directly to bucket indices (no per-feature data)
- bucket_percentile_key_grid: reproduce the exact statistics-key grid used for buckets
- build_boundaries_from_stats: build bucket boundaries from pre-computed feature statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch as t
from beartype import beartype

from steering.refined_types import (
    AtLeastTwoIntLike,
    OpenUnitIntervalLike,
    PositiveIntLike,
)

if TYPE_CHECKING:
    _PerFeatureStats = dict[str, Any]
    _PercentileKeyGrid = tuple[int, ...]
else:
    _PerFeatureStats = dict
    # Avoid beartype's global-RNG-backed sampling of variadic tuple elements.
    _PercentileKeyGrid = tuple

DEGENERATE_MIDPOINT = 0.5


def discretize_activations(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    boundaries: t.Tensor,
) -> t.Tensor:
    """Map activation magnitudes to bucket indices via searchsorted.

    Values exactly at a boundary are placed in the higher bucket (right=True).

    Args:
        feature_ids: [N] int32 feature indices.
        feature_mags: [N] float32 magnitudes.
        boundaries: [n_features, n_boundaries] float64 per-feature boundaries.

    Returns:
        [N] int64 bucket indices in [0, num_buckets).
    """
    if len(feature_ids) == 0:
        return t.tensor([], dtype=t.int64)

    assert len(feature_ids) == len(feature_mags), (
        "feature_ids and feature_mags must have same length"
    )

    n_features = boundaries.shape[0]
    mags_f64 = feature_mags.to(t.float64)

    if (feature_ids >= n_features).any():
        bad = feature_ids[feature_ids >= n_features]
        raise ValueError(
            f"Feature IDs {bad.tolist()[:5]} exceed boundaries size {n_features}. "
            f"Boundaries tensor must cover all feature IDs."
        )
    per_element_bounds = boundaries[feature_ids.long()]
    buckets = t.searchsorted(per_element_bounds, mags_f64.unsqueeze(1), right=True).squeeze(1)

    return buckets


def compute_percentiles(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    boundaries: t.Tensor,
) -> t.Tensor:
    """Map activation magnitudes to percentile values in [0, 1] via piecewise linear interpolation.

    Uses the per-feature bucket boundaries (which represent equally-spaced percentile points)
    to interpolate each magnitude's position within the feature's corpus-wide distribution.

    Args:
        feature_ids: [N] int32 feature indices.
        feature_mags: [N] float32 magnitudes.
        boundaries: [n_features, n_boundaries] float64 per-feature boundaries.

    Returns:
        [N] float32 percentile values in [0, 1].
    """
    if len(feature_ids) == 0:
        return t.tensor([], dtype=t.float32)

    n_features = boundaries.shape[0]
    n_boundaries = boundaries.shape[1]
    num_buckets = n_boundaries + 1

    # Single bucket (e.g. binary bucketing with negative sampling): no boundaries
    # to interpolate between, so all features map to the degenerate midpoint.
    if n_boundaries == 0:
        return t.full((len(feature_ids),), DEGENERATE_MIDPOINT, dtype=t.float32)

    mags = feature_mags.to(t.float64)

    if (feature_ids >= n_features).any():
        bad = feature_ids[feature_ids >= n_features]
        raise ValueError(
            f"Feature IDs {bad.tolist()[:5]} exceed boundaries size {n_features}. "
            f"Boundaries tensor must cover all feature IDs."
        )
    per_elem_bounds = boundaries[feature_ids.long()]  # (N, n_boundaries)

    # Percentile values at each boundary: p(1/num_buckets), ..., p((num_buckets-1)/num_buckets)
    boundary_pcts = t.linspace(
        1.0 / num_buckets, 1.0 - 1.0 / num_buckets, n_boundaries, dtype=t.float64
    )

    # Find which interval each magnitude falls in via searchsorted
    bucket_idx = t.searchsorted(per_elem_bounds, mags.unsqueeze(1), right=True).squeeze(1)  # (N,)

    pct_step = 1.0 / num_buckets

    lower_bound = t.zeros_like(mags)
    upper_bound = t.zeros_like(mags)
    lower_pct = t.zeros_like(mags)
    upper_pct = t.zeros_like(mags)

    for i in range(num_buckets):
        mask = bucket_idx == i
        if not mask.any():
            continue
        if i == 0:
            lower_bound[mask] = 0.0
            upper_bound[mask] = per_elem_bounds[mask, 0]
            lower_pct[mask] = 0.0
            upper_pct[mask] = pct_step
        elif i == n_boundaries:
            lower_bound[mask] = per_elem_bounds[mask, n_boundaries - 1]
            upper_bound[mask] = per_elem_bounds[mask, n_boundaries - 1] + (
                per_elem_bounds[mask, n_boundaries - 1]
                - per_elem_bounds[mask, max(0, n_boundaries - 2)]
            )
            lower_pct[mask] = 1.0 - pct_step
            upper_pct[mask] = 1.0
        else:
            lower_bound[mask] = per_elem_bounds[mask, i - 1]
            upper_bound[mask] = per_elem_bounds[mask, i]
            lower_pct[mask] = boundary_pcts[i - 1]
            upper_pct[mask] = boundary_pcts[i]

    span = upper_bound - lower_bound
    safe_span = t.where(span > 0, span, t.ones_like(span))
    frac = (mags - lower_bound) / safe_span
    frac = t.where(span > 0, frac, t.full_like(frac, DEGENERATE_MIDPOINT))  # degenerate -> midpoint
    percentiles = lower_pct + frac * (upper_pct - lower_pct)

    percentiles = percentiles.clamp(0.0, 1.0)

    return percentiles.float()


@beartype
def percentile_to_bucket(
    percentiles: t.Tensor,
    num_buckets: PositiveIntLike,
) -> t.Tensor:
    """Map predicted percentile values to discrete bucket indices.

    Buckets in this codebase are equally-spaced percentile bands by
    construction (the raw-activation reference in
    ``tests/steering/_discretization_reference.py`` uses
    ``linspace(1/B, 1-1/B, B-1)`` percentile points). Therefore the inverse
    map percentile -> bucket is simply ``floor(p * num_buckets)`` clamped to
    ``[0, num_buckets - 1]``.

    Args:
        percentiles: Tensor of percentile values, expected in [0, 1]. Any
            shape; the operation is element-wise. Out-of-range values are
            clamped (no error) so the function is robust to bf16/fp32
            numerical drift around the edges of the [0,1] interval.
        num_buckets: Total number of buckets (must be >= 1).

    Returns:
        Int64 tensor of bucket indices, same shape as ``percentiles``.
    """
    if percentiles.numel() == 0:
        return percentiles.new_zeros(percentiles.shape, dtype=t.int64)
    buckets = (percentiles * num_buckets).floor().to(t.int64)
    return buckets.clamp(min=0, max=num_buckets - 1)


@beartype
def bucket_percentile_key_grid(
    num_buckets: PositiveIntLike,
    *,
    threshold_percentile: OpenUnitIntervalLike | None = None,
) -> _PercentileKeyGrid:
    """Return the exact integer suffixes of the statistics keys for buckets.

    This is the identity-bearing grid historically embedded in
    :func:`build_boundaries_from_stats`. The two branches intentionally use
    different integer conversions: unthresholded boundaries truncate with
    ``int()``, while post-threshold boundaries use Python's ``round()``. Do not
    normalize either branch to an idealized linspace: for example, three
    unthresholded buckets use ``p33`` and ``p66``, not exact thirds.

    The returned integers are percentile-rank keys, not normalized scores. A
    self-conditioned consumer reproducing a *raw-magnitude* producer applies the
    percentile-table endpoint contract (exact p99 has rank 99.5; other integer
    keys keep their rank), then divides by 100 into continuous normalized rank.
    Normalized packed-row producers do not use this grid; they retain uniform
    percentile bucketing.

    A single active bucket has no boundaries and therefore returns an empty
    grid. This case occurs when an encoder has two rows and reserves its final
    row for sampled inactive features.
    """
    n_boundaries = num_buckets - 1
    if n_boundaries == 0:
        return ()

    if threshold_percentile is not None:
        start_pct = threshold_percentile * 100
        end_pct = 99
        spacing = (end_pct - start_pct) / n_boundaries
        assert spacing >= 1.0, (
            f"Post-filter range [{start_pct:.0f}, {end_pct}] is too narrow for "
            f"{n_boundaries} boundaries (spacing={spacing:.2f} < 1 pct point). "
            f"Reduce num_buckets or lower threshold_percentile."
        )
        return tuple(
            round(start_pct + (end_pct - start_pct) * (i + 1) / n_boundaries)
            for i in range(n_boundaries)
        )

    return tuple(int(100 * (i + 1) / num_buckets) for i in range(n_boundaries))


@beartype
def build_boundaries_from_stats(
    per_feature_stats: _PerFeatureStats,
    n_sae_features: PositiveIntLike,
    num_buckets: AtLeastTwoIntLike,
    threshold_percentile: OpenUnitIntervalLike | None = None,
) -> t.Tensor:
    """Build per-feature bucket boundaries from pre-computed feature statistics.

    Uses the percentile values stored in feature statistics instead of
    computing from raw feature data.  For N buckets, uses percentile keys
    p(100/N), p(200/N), ..., p(100*(N-1)/N).

    Args:
        per_feature_stats: Dict mapping feature ID strings to PerFeatureStat
            instances, each containing a ``percentiles`` dict with keys
            like "p10", "p50".
        n_sae_features: Total number of SAE features (determines output rows).
        num_buckets: Number of discrete buckets.
        threshold_percentile: Optional percentile threshold (0-1 scale).
            When provided, bucket boundaries are computed within the
            post-filter percentile range (threshold_pct*100, p99] instead
            of the full [p1, p99] range. This ensures num_buckets usable
            buckets exist after threshold filtering.

    Returns:
        [n_sae_features, num_buckets - 1] float64 boundaries tensor.
        Features not in stats get all-zero boundaries.
    """
    n_boundaries = num_buckets - 1
    pct_keys = [
        f"p{percentile}"
        for percentile in bucket_percentile_key_grid(
            num_buckets,
            threshold_percentile=threshold_percentile,
        )
    ]

    boundaries = t.zeros(n_sae_features, n_boundaries, dtype=t.float64)
    for fid_str, fstats in per_feature_stats.items():
        fid = int(fid_str)
        if fid < n_sae_features:
            pcts = fstats.percentiles
            missing = [k for k in pct_keys if k not in pcts]
            assert not missing, (
                f"Bucket boundary percentile keys {missing} not found for feature {fid}. "
                f"Available: {sorted(pcts.keys())}. Re-run compute_statistics."
            )
            boundaries[fid] = t.tensor([pcts[k] for k in pct_keys], dtype=t.float64)

    return boundaries
