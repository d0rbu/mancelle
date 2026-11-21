"""Health checks over aggregated SAE feature statistics.

A correctly-functioning sparse SAE is content-dependent: most features fire in
a small fraction of segments. A random-weight / mis-loaded model stamps a fixed
~thousands-feature set onto essentially every segment (median per-feature firing
frequency ≈ 1, and a large fraction of the ever-seen features active in nearly
every segment). This module turns that signature into a hard failure at the
``compute_statistics`` aggregation stage, behind the load-time guard. See
issue #549.
"""

from __future__ import annotations

import torch as t

from errors import DegenerateFeatureStatisticsError

MIN_SEGMENTS_FOR_DEGENERACY_CHECK = 1000
DEGENERATE_FIRING_FREQ_THRESHOLD = 0.5
# Fraction of the ever-seen features that, firing in >99% of segments, marks the
# "a fixed feature set fires everywhere" signature. Healthy sparse SAEs have only
# a handful of always-on features (≈0.5% of seen); the random-weight 4b case had
# ≈83%, so 0.5 carries a large margin and catches a median-diluted mis-load.
DEGENERATE_UBIQUITOUS_FEATURE_FRACTION = 0.5


def _interpolated_median(values: t.Tensor) -> float:
    """Linear-interpolated median, byte-identical to ``torch.quantile(values,
    0.5)`` but with NO 2**24-element ceiling (``torch.quantile`` raises
    ``quantile() input tensor is too large`` above that). Two details give exact
    equality: (1) ``torch.quantile`` FLATTENS its input, so we ``reshape(-1)``
    first; (2) at q=0.5 it interpolates the two central order statistics via
    ``torch.lerp(lower, upper, 0.5)`` — so we delegate to that SAME primitive
    rather than hand-inlining the arithmetic (the inline form ``upper -
    (upper-lower)*0.5`` is what lerp computes at weight 0.5, but ``lower +
    (upper-lower)*0.5`` is NOT — it rounds differently by ~1 ULP on ~4% of
    even-count inputs, so delegating removes that footgun). The order statistics
    are selected with ``torch.kthvalue`` (O(n), no full sort or discarded index
    buffer; verified byte-identical to quantile over fuzzed odd/even inputs).
    Issue #549 review. Callers pass a non-empty tensor (``firing_health`` guards
    numel==0)."""
    flat = values.reshape(-1)
    n = flat.numel()
    mid = n // 2
    if n % 2 == 1:
        return float(t.kthvalue(flat, mid + 1).values)
    lower = t.kthvalue(flat, mid).values
    upper = t.kthvalue(flat, mid + 1).values
    return float(t.lerp(lower, upper, 0.5))


def firing_health(counts: t.Tensor, n_segments: int) -> tuple[float, int]:
    """Median per-feature firing frequency and the count of features firing in
    >99% of segments, over SEEN features (``counts > 0``), with one shared
    denominator. Returns ``(0.0, 0)`` if nothing fired. The median is the
    linear-interpolated true median (via :func:`_interpolated_median`), NOT
    ``torch.median`` (which returns the lower of the two middle values for even
    counts)."""
    if n_segments <= 0 or counts.numel() == 0:
        return 0.0, 0
    freq = counts.to(t.float64) / float(n_segments)
    median = _interpolated_median(freq)
    n_firing_gt_99pct = int((freq > 0.99).sum())
    return median, n_firing_gt_99pct


def assert_feature_statistics_not_degenerate(
    *,
    median_firing_frequency: float,
    n_features_firing_gt_99pct: int,
    n_features_seen: int,
    n_sae_features: int,
    n_segments: int,
    allow_degenerate: bool,
) -> None:
    """Raise ``DegenerateFeatureStatisticsError`` on the random-weight signature.

    Checked only on a corpus large enough to be meaningful
    (``n_segments >= MIN_SEGMENTS_FOR_DEGENERACY_CHECK``) and unless explicitly
    overridden (``allow_degenerate``). Two independent signals of the issue #549
    "a fixed feature set fires in ~every segment" signature trigger the failure:

    - the MEDIAN per-feature firing frequency exceeds
      ``DEGENERATE_FIRING_FREQ_THRESHOLD``; or
    - at least ``DEGENERATE_UBIQUITOUS_FEATURE_FRACTION`` of the ever-seen
      features fire in >99% of segments — this catches a mis-load whose median is
      diluted below the first threshold by many rarely-firing random features
      while a large block of features is nonetheless always-on.

    Healthy sparse SAEs sit far below both thresholds, so they carry a large
    margin; ``allow_degenerate_statistics`` covers any genuinely-unusual
    legitimate SAE (e.g. a tiny, low-diversity corpus).
    """
    if allow_degenerate or n_segments < MIN_SEGMENTS_FOR_DEGENERACY_CHECK:
        return
    median_degenerate = median_firing_frequency > DEGENERATE_FIRING_FREQ_THRESHOLD
    ubiquitous_degenerate = (
        n_features_seen > 0
        and n_features_firing_gt_99pct
        >= DEGENERATE_UBIQUITOUS_FEATURE_FRACTION * n_features_seen
    )
    if median_degenerate or ubiquitous_degenerate:
        raise DegenerateFeatureStatisticsError(
            "degenerate feature statistics — median per-feature firing frequency "
            f"{median_firing_frequency:.3f} (threshold {DEGENERATE_FIRING_FREQ_THRESHOLD}); "
            f"{n_features_firing_gt_99pct}/{n_features_seen} ever-seen features fire "
            f">99% of {n_segments:,} segments (threshold "
            f"{DEGENERATE_UBIQUITOUS_FEATURE_FRACTION:.0%} of seen); "
            f"{n_features_seen}/{n_sae_features} features ever fired. "
            "This is the random-weight / mis-loaded-model signature of issue #549. "
            "Set allow_degenerate_statistics=true to override (rarely correct)."
        )
