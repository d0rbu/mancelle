"""Percentile table construction and magnitude-to-percentile lookup.

Provides:
- build_percentile_table: construct [n_features, 99] lookup table from stats
- magnitude_to_percentile: map feature magnitudes to percentile scores [0.5, 99.5]
- compute_percentile_checksum: deterministic hash of a percentile table for shard validation
- percentile_to_uint8 / uint8_to_percentile: encode/decode percentile values as uint8
- normalized_percentile_to_uint8: encode normalized percentile weights [0, 1]

Split into its own module to avoid circular imports between delta.py and sampling.py.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch as t
from beartype import beartype

from steering.refined_types import PositiveIntLike
from steering.stats_types import PerFeatureStat

if TYPE_CHECKING:
    _PerFeatureStats = dict[str, PerFeatureStat]
else:
    _PerFeatureStats = dict


@beartype
def build_percentile_table(
    per_feature_stats: _PerFeatureStats,
    n_sae_features: PositiveIntLike,
) -> t.Tensor:
    """Construct a percentile lookup table from precomputed per-feature statistics.

    Args:
        per_feature_stats: Mapping from feature ID string to PerFeatureStat.
            Each stat must have percentiles dict with keys "p1" through "p99".
        n_sae_features: Total number of SAE features (table row count).

    Returns:
        Float64 tensor of shape [n_sae_features, 99]. Row i contains the p1-p99
        values for feature i. Features not in per_feature_stats get all-zero rows.
    """
    table = t.zeros(n_sae_features, 99, dtype=t.float64)
    percentile_keys = tuple(f"p{k}" for k in range(1, 100))
    for fid_str, fstat in per_feature_stats.items():
        fid = int(fid_str)
        row = t.tensor(
            [fstat.percentiles[key] for key in percentile_keys],
            dtype=t.float64,
        )
        if not (row[1:] >= row[:-1]).all():
            raise ValueError(
                f"Feature {fid} has non-monotonic percentile values. "
                "searchsorted requires sorted rows."
            )
        table[fid] = row
    return table


def magnitude_to_percentile(
    feature_ids: t.Tensor, feature_mags: t.Tensor, percentile_table: t.Tensor
) -> t.Tensor:
    """Map each feature's magnitude to its percentile via linear interpolation.

    The grid x-coordinates are [1, 2, ..., 99] and lookups replicate
    ``t.searchsorted(rows, mags, right=True)`` exactly — a magnitude at p1
    returns idx=1 (in-range), ties go right, NaN sorts above p99 — but via a
    direct-gather binary search over the feature's table row, so no [N, 99]
    row matrix is ever materialized (issue #1269: the row gather moved ~792 MB
    per 1M-value chunk; the binary search reads ~9 float64 values per lookup).

    Requires ``percentile_table`` rows to be monotone non-decreasing, which
    ``build_percentile_table`` enforces at construction. Under that
    precondition a row is all-zero (feature missing from statistics) iff its
    first and last entries are both zero.

    Args:
        feature_ids: [N] integer tensor of feature indices. Normalized internally
            to int64 on the current device.
        feature_mags: [N] float tensor of magnitudes.
        percentile_table: [n_features, 99] float64 lookup table.

    Returns:
        [N] float64 tensor of percentile scores in [0.5, 99.5].
        - Below p1 → 0.5
        - Above p99 (and NaN) → 99.5
        - Otherwise → linearly interpolated in (idx, idx+1)
    """
    return _magnitude_to_percentile(feature_ids, feature_mags, percentile_table)


def _magnitude_to_percentile(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    percentile_table: t.Tensor,
    *,
    table_is_finite: bool = False,
) -> t.Tensor:
    """Shared lookup; finite evidence is private to read-only native pairing tables."""
    n = feature_ids.shape[0]
    if n == 0:
        return t.zeros(0, dtype=t.float64)

    if feature_mags.shape != feature_ids.shape:
        # The pre-#1269 searchsorted rejected mismatched leading dimensions;
        # the gather-based lanes would silently broadcast instead — keep the
        # loud error contract (review finding, issue #1269).
        raise ValueError(
            f"feature_mags shape {tuple(feature_mags.shape)} != "
            f"feature_ids shape {tuple(feature_ids.shape)}; "
            "magnitude_to_percentile requires matched [N] inputs."
        )

    fids = feature_ids.long()
    first_vals = percentile_table[fids, 0]
    last_vals = percentile_table[fids, 98]
    missing_rows = (first_vals == 0) & (last_vals == 0)
    if missing_rows.any():
        # Report the DISTINCT missing population, not just the first few raw
        # hits: a caller that sees "[7, 7, 7, 7, 7]" cannot tell a single
        # unseen feature from a statistics artifact that covers almost
        # nothing, and that is the difference between a one-off data bug and a
        # rebuild of the statistics pass.
        distinct_missing = t.unique(feature_ids[missing_rows].long())
        n_distinct = int(distinct_missing.numel())
        raise ValueError(
            f"Percentile statistics are missing for {n_distinct} distinct "
            f"feature ID(s) (first {min(n_distinct, 5)}: "
            f"{distinct_missing.tolist()[:5]}); all-zero percentile rows "
            "cannot be queried. Percentiles are looked up on the RAW "
            "pre-selection feature ids of each row — before any feature-mask "
            "filtering and before cluster consolidation — so the statistics "
            "artifact must cover every ACTIVE feature in the queried data, "
            "not only the allowed representatives. Rebuild statistics over "
            "the full active feature set; a rebuild restricted to the allowed "
            "set will fail here again."
        )
    if not table_is_finite and (t.isnan(first_vals).any() or t.isnan(last_vals).any()):
        raise ValueError(
            "percentile_table contains NaN in probed entries; NaN rows are "
            "not constructible via build_percentile_table."
        )
    mags = feature_mags.to(t.float64)

    # Binary search replicating searchsorted(right=True): idx = number of row
    # entries <= mag, in [0, 99]. Seven halvings take the interval width
    # 99→50→25→13→7→4→2→1. NaN never satisfies `<=`, which would yield 0;
    # searchsorted places NaN above everything, so force idx=99 afterwards.
    # NaN probed from the TABLE (impossible via build_percentile_table, whose
    # monotonicity check rejects NaN rows) would silently misroute the search
    # relative to searchsorted's predicate — raise instead of diverging.
    lo = t.zeros(n, dtype=t.int64, device=mags.device)
    hi = t.full((n,), 99, dtype=t.int64, device=mags.device)
    for _ in range(7):
        active = lo < hi
        mid = (lo + hi) >> 1
        # Converged lanes may have mid=99; clamp only their unused gather index.
        mid_vals = percentile_table[fids, mid.clamp_max(98)]
        if not table_is_finite and (t.isnan(mid_vals) & active).any():
            raise ValueError(
                "percentile_table contains NaN in probed entries; NaN rows are "
                "not constructible via build_percentile_table."
            )
        go_right = (mid_vals <= mags) & active
        lo = t.where(go_right, mid + 1, lo)
        hi = t.where(~go_right, mid, hi)
    idx = t.where(t.isnan(mags), t.full_like(lo, 99), lo)

    result = t.empty(n, dtype=t.float64, device=feature_ids.device)

    below_mask = idx == 0
    above_mask = idx >= 99
    interp_mask = ~below_mask & ~above_mask

    result[below_mask] = 0.5
    result[above_mask] = 99.5

    if interp_mask.any():
        i_idx = idx[interp_mask]  # values in [1, 98]
        i_fids = fids[interp_mask]
        i_mags = mags[interp_mask]

        # left_val = p_{i_idx} = row[i_idx - 1]; right_val = p_{i_idx + 1} = row[i_idx]
        left_val = percentile_table[i_fids, i_idx - 1]
        right_val = percentile_table[i_fids, i_idx]

        denom = right_val - left_val
        # When left_val == right_val (constant region), use frac=0.5 (midpoint)
        frac = t.where(
            denom == 0.0,
            t.full_like(denom, 0.5),
            (i_mags - left_val) / denom,
        )
        result[interp_mask] = i_idx.to(t.float64) + frac

    return result


def compute_percentile_checksum(percentile_table: t.Tensor) -> str:
    """Compute a deterministic SHA-256 checksum of a percentile table.

    The checksum captures the exact percentile statistics used to build the
    table. If statistics change (new corpus, different computation), the
    checksum changes, invalidating any precomputed percentiles in shards.

    Args:
        percentile_table: [n_features, 99] float64 tensor from build_percentile_table.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).
    """
    # Contiguous float64 bytes ensure deterministic hashing regardless of
    # tensor memory layout.
    data = percentile_table.contiguous().to(t.float64).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def percentile_to_uint8(percentile_scores: t.Tensor) -> t.Tensor:
    """Encode percentile scores ``[0.5, 99.5]`` as uint8 values ``[0, 255]``.

    Encoding: ``uint8 = round(percentile / 100.0 * 255)``.

    NOTE on units: this function pairs with ``uint8_to_percentile``, but the
    pair is **NOT** unit-symmetric — the encoder takes a percent rank in
    ``[0.5, 99.5]`` and the decoder returns a normalized weight in ``[0, 1]``.
    The round-trip ``encode -> decode`` is rank-preserving (and lossy by uint8
    quantization to ≈ 0.39 % bins) but is NOT a unit round-trip in
    ``[0.5, 99.5]`` percentile space. See ``uint8_to_percentile`` for the
    decoder contract.

    Args:
        percentile_scores: [N] float64 tensor from
            :func:`magnitude_to_percentile`, values in ``[0.5, 99.5]``.

    Returns:
        [N] uint8 tensor with values in ``[0, 255]``.
    """
    normalized = percentile_scores.float() / 100.0  # [0.005, 0.995]
    return (normalized * 255.0).round().clamp(0, 255).to(t.uint8)


def normalized_percentile_to_uint8(percentile_weights: t.Tensor) -> t.Tensor:
    """Encode normalized percentile weights ``[0, 1]`` as uint8 values.

    This is the inverse-unit companion to :func:`uint8_to_percentile`, which
    decodes shard bytes to normalized weights rather than percentile-rank units.
    Do not use :func:`percentile_to_uint8` for these values; that encoder takes
    percentile ranks in ``[0.5, 99.5]``.
    """
    return (percentile_weights.float() * 255.0).round().clamp(0, 255).to(t.uint8)


def uint8_to_percentile(encoded: t.Tensor) -> t.Tensor:
    """Decode uint8 codes to a normalized weight in ``[0, 1]``.

    Decoding: ``weight = uint8.float() / 255.0``.

    NOTE on units: the returned values are a **normalized weight** in
    ``[0, 1]`` (suitable for multinomial sampling), NOT the percent rank in
    ``[0.5, 99.5]`` that ``percentile_to_uint8`` accepts as input. The two
    functions are intentionally asymmetric — the slice 04 invariant
    ``m → p → uint8 → p → m`` is rank-preserving (monotone) but lossy by uint8
    quantization. To recover an approximation of the encoded percentile in
    units of percent, multiply by 100.0.

    Args:
        encoded: [N] uint8 tensor.

    Returns:
        [N] float32 tensor with values in ``[0, 1]``.
    """
    return encoded.float() / 255.0
