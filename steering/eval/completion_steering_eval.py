"""Completion Steering Eval — data structures and stages 1-3.

Evaluates whether SAE feature conditioning causally influences the
encoder-decoder model's segment completions on real validation data.

Metric semantics (Wave 3D task 1)
---------------------------------
This evaluator's primary output is the ``completion_scan_select_delta``
metric: a scan-and-select over real validation features to measure
activation differences under conditioning, without any generative step.

This is **not** the canonical "specificity" metric. The canonical
specificity (``real - random_doc_random_seg``) lives in
``steering/eval/inline_steering_eval.py`` and requires generation. A
W&B run reporting ``completion_scan_select_delta`` is NOT comparable to
one reporting ``steering/roundtrip_specificity*``. Downstream consumers
should migrate old dashboard keys to the explicit scan-select metric name.
"""

from __future__ import annotations

import bisect
import random as _random
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch as t
from beartype import beartype
from loguru import logger
from safetensors.torch import load_file, save_file

from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.sampling import (
    SamplingBundle,
    magnitude_to_percentile,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    apply_threshold,
)
from steering.feature_io import load_features
from steering.refined_types import (
    NonNegativeIntLike,
    NonNegativeIntListLike,
    NonNegativeIntMappingLike,
    NonNegativeIntSetLike,
    PositiveIntLike,
    StrictIntLike,
)
from steering.stats_types import PerFeatureStat
from steering.tokenizer_protocols import PrefixedTextTokenizer

if TYPE_CHECKING:
    _BoolFloatPair = tuple[bool, float]
    _RawFeatureSelectionList = list["RawFeatureSelection"]
    _StringList = list[str]
    _TargetSegmentMaps = dict[int, dict[int, int]]
else:
    # Runtime element sampling would perturb the selection RNG. The strings
    # have no refined value domain here, so only the outer list is checked.
    _BoolFloatPair = tuple
    _RawFeatureSelectionList = list
    _StringList = list
    _TargetSegmentMaps = dict

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@beartype
@dataclass(frozen=True)
class RawFeatureSelection:
    """A randomly selected feature with sampled documents (no description yet)."""

    feature_id: NonNegativeIntLike
    n_docs_activated: NonNegativeIntLike
    selected_docs: NonNegativeIntListLike
    target_segments: NonNegativeIntMappingLike = field(default_factory=dict)


@beartype
@dataclass(frozen=True)
class FeatureSelection:
    """A selected feature with Neuronpedia description and document list."""

    feature_id: NonNegativeIntLike
    neuronpedia_description: str
    n_docs_activated: NonNegativeIntLike
    selected_docs: NonNegativeIntListLike
    target_segments: NonNegativeIntMappingLike = field(default_factory=dict)


@dataclass(frozen=True)
class ShardData:
    """Flat tensor representation of a feature shard (segment-indexed).

    Mirrors the dict returned by :func:`load_features` but with typed fields.
    Used as the return type of :func:`apply_threshold_to_shard`.
    """

    offsets: t.Tensor
    doc_ids: t.Tensor
    segment_indices: t.Tensor
    feature_ids: t.Tensor
    feature_mags: t.Tensor
    n_segments: int


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class EvalMode(Enum):
    ABLATION = "ablation"
    INSERTION = "insertion"


class EvalStage(Enum):
    ALL = "all"
    SCAN = "scan"
    THRESHOLD = "threshold"
    SELECT = "select"
    GENERATE = "generate"
    JUDGE = "judge"
    FEATURE_ANALYSIS = "feature_analysis"


# ---------------------------------------------------------------------------
# Percentile lookup helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PercentileLookup:
    """Precomputed sorted arrays for fast percentile interpolation."""

    pcts: list[float]  # sorted percentile values (0.01, 0.02, ...)
    vals: list[float]  # corresponding activation thresholds


def _build_percentile_lookup(percentiles: dict[str, float]) -> PercentileLookup:
    """Extract and sort pXX keys from a percentiles dict once.

    Args:
        percentiles: Dict with keys like "p1".."p99" mapping to float values.

    Returns:
        A :class:`PercentileLookup` with sorted ``pcts`` and ``vals`` arrays.

    Raises:
        ValueError: If no valid percentile keys are found.
    """
    pct_pairs: list[tuple[float, float]] = []
    for key, val in percentiles.items():
        if key.startswith("p") and key[1:].isdigit():
            pct_pairs.append((float(key[1:]) / 100.0, val))
    if not pct_pairs:
        raise ValueError(
            f"No valid percentile keys (p1..p99) found in percentiles dict "
            f"(keys: {list(percentiles.keys())[:10]})"
        )
    pct_pairs.sort(key=lambda x: x[0])
    pcts = [p for p, _ in pct_pairs]
    vals = [v for _, v in pct_pairs]
    return PercentileLookup(pcts=pcts, vals=vals)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _activation_to_percentile(
    activation: float,
    lookup: PercentileLookup,
) -> float:
    """Convert a raw activation value to an approximate percentile (0-1 scale).

    Uses binary search (O(log n)) and linear interpolation between adjacent
    percentile brackets. Returns the lowest available percentile (e.g. 0.01)
    if activation is at or below the lowest threshold, and the highest
    (e.g. 0.99) if at or above the highest.

    Args:
        activation: The raw activation magnitude to convert.
        lookup: Precomputed :class:`PercentileLookup` for this feature.

    Returns:
        Estimated percentile in [lo_pct, hi_pct].
    """
    vals = lookup.vals
    pcts = lookup.pcts

    lo_pct = pcts[0]
    hi_pct = pcts[-1]
    lo_val = vals[0]
    hi_val = vals[-1]

    if activation <= lo_val:
        return lo_pct
    if activation >= hi_val:
        return hi_pct

    # bisect_left gives us the index of the first val >= activation.
    # This means the bracket is [i-1, i], matching the original's behaviour
    # when activation equals a value in vals (including the duplicate-vals case
    # where the original finds the first bracket ending at the shared value).
    i = bisect.bisect_left(vals, activation)
    # Clamp to valid bracket indices (should always be in range given the
    # boundary checks above, but be defensive).
    if i <= 0:
        return lo_pct
    if i >= len(vals):
        return hi_pct

    v0 = vals[i - 1]
    v1 = vals[i]
    p0 = pcts[i - 1]
    p1 = pcts[i]

    if v1 == v0:
        return p0
    frac = (activation - v0) / (v1 - v0)
    return p0 + frac * (p1 - p0)


def _vectorized_percentile(
    feature_ids: t.Tensor,
    activations: t.Tensor,
    per_feature_stats: dict[str, PerFeatureStat],
) -> t.Tensor:
    """Compute percentiles for all activations in one vectorized pass.

    For each (feature_id, activation) pair, looks up the per-feature percentile
    table and interpolates — identical to :func:`_activation_to_percentile` but
    batched over all activations using :func:`torch.searchsorted`.

    Args:
        feature_ids: (M,) int64 — feature ID per activation.
        activations: (M,) float32 — raw activation magnitudes.
        per_feature_stats: Per-feature statistics keyed by str(feature_id).

    Returns:
        (M,) float32 — percentile in [lo_pct, hi_pct] per activation.
    """
    M = feature_ids.numel()
    if M == 0:
        return t.empty(0, dtype=t.float32)

    # Collect unique feature IDs present in this batch
    unique_fids, inverse = t.unique(feature_ids, return_inverse=True)

    # Build a (n_unique, n_pct) table of values and percentiles.
    # All features share the same percentile keys (p1..p99).
    first_stat = per_feature_stats[str(int(unique_fids[0].item()))]
    first_lookup = _build_percentile_lookup(first_stat.percentiles)
    n_pct = len(first_lookup.pcts)
    pcts_1d = t.tensor(first_lookup.pcts, dtype=t.float32)  # (n_pct,) shared

    vals_table = t.empty(len(unique_fids), n_pct, dtype=t.float32)
    for row, ufid in enumerate(unique_fids.tolist()):
        fid_str = str(int(ufid))
        stat = per_feature_stats.get(fid_str)
        if stat is None:
            raise KeyError(f"Feature {int(ufid)} not found in per_feature_stats")
        lookup = _build_percentile_lookup(stat.percentiles)
        assert len(lookup.vals) == n_pct, (
            f"Feature {int(ufid)} has {len(lookup.vals)} percentile keys, expected {n_pct}"
        )
        vals_table[row] = t.tensor(lookup.vals, dtype=t.float32)

    # Sort activations by feature so each feature's data is contiguous.
    # This turns per-feature masking from O(M) boolean ops to O(1) slicing.
    sort_order = t.argsort(inverse, stable=True)
    sorted_acts = activations[sort_order]
    sorted_inverse = inverse[sort_order]

    # Find contiguous boundaries per feature via unique_consecutive
    _, feat_counts = t.unique_consecutive(sorted_inverse, return_counts=True)
    feat_offsets = t.zeros(len(feat_counts) + 1, dtype=t.long)
    feat_offsets[1:] = feat_counts.cumsum(0)

    lo_pct_value = float(pcts_1d[0].item())
    hi_pct_value = float(pcts_1d[-1].item())
    result_sorted = t.empty(M, dtype=t.float32)

    for row in range(len(unique_fids)):
        start = int(feat_offsets[row].item())
        end = int(feat_offsets[row + 1].item())
        if start == end:
            continue

        feat_acts = sorted_acts[start:end]  # contiguous slice — O(1)
        vals = vals_table[row]  # (n_pct,) sorted thresholds

        lo_val = vals[0].item()
        hi_val = vals[-1].item()

        # searchsorted on 1-D sorted vals — O(n log m) per feature
        idx = t.searchsorted(vals, feat_acts)

        # Start with lo_pct default
        feat_result = t.full_like(feat_acts, lo_pct_value)

        # High clamp
        high = feat_acts >= hi_val
        feat_result[high] = hi_pct_value

        # Interpolation region
        interp = (~high) & (feat_acts > lo_val)
        if interp.any():
            ii = idx[interp].clamp(1, n_pct - 1)
            v0 = vals[ii - 1]
            v1 = vals[ii]
            p0 = pcts_1d[ii - 1]
            p1 = pcts_1d[ii]
            denom = v1 - v0
            safe = denom > 0
            frac = t.zeros_like(denom)
            frac[safe] = (feat_acts[interp][safe] - v0[safe]) / denom[safe]
            feat_result[interp] = p0 + frac * (p1 - p0)

        result_sorted[start:end] = feat_result

    # Unsort back to original order
    result = t.empty(M, dtype=t.float32)
    result[sort_order] = result_sorted
    return result


def apply_threshold_to_shard(
    shard_data: dict,
    threshold_cfg: ThresholdConfig,
    precomputed_thresholds: t.Tensor | None,
) -> ShardData:
    """Apply per-segment thresholding to a loaded feature shard.

    Iterates over segments, applies :func:`apply_threshold` to each, and
    rebuilds flat arrays with updated offsets.  ``doc_ids`` and
    ``segment_indices`` pass through unchanged.

    Args:
        shard_data: Dict from :func:`load_features` with keys ``offsets``,
            ``doc_ids``, ``segment_indices``, ``feature_ids``, ``feature_mags``,
            ``n_segments``.
        threshold_cfg: Threshold configuration.
        precomputed_thresholds: Precomputed per-feature threshold tensor.

    Returns:
        :class:`ShardData` with the thresholded features.
    """
    offsets_raw: t.Tensor = shard_data["offsets"]
    feature_ids: t.Tensor = shard_data["feature_ids"]
    feature_mags: t.Tensor = shard_data["feature_mags"]
    n_segments = int(shard_data["n_segments"])

    new_fids_parts: list[t.Tensor] = []
    new_mags_parts: list[t.Tensor] = []
    new_offsets = [0]

    for i in range(n_segments):
        start = int(offsets_raw[i].item())
        end = int(offsets_raw[i + 1].item())
        seg_fids = feature_ids[start:end]
        seg_mags = feature_mags[start:end]

        filtered_fids, filtered_mags = apply_threshold(
            threshold_cfg, seg_fids, seg_mags, precomputed_thresholds
        )
        new_fids_parts.append(filtered_fids)
        new_mags_parts.append(filtered_mags)
        new_offsets.append(new_offsets[-1] + len(filtered_fids))

    if new_fids_parts:
        out_fids = t.cat(new_fids_parts)
        out_mags = t.cat(new_mags_parts)
    else:
        out_fids = t.empty(0, dtype=feature_ids.dtype)
        out_mags = t.empty(0, dtype=feature_mags.dtype)

    return ShardData(
        offsets=t.tensor(new_offsets, dtype=offsets_raw.dtype),
        doc_ids=shard_data["doc_ids"],
        segment_indices=shard_data["segment_indices"],
        feature_ids=out_fids,
        feature_mags=out_mags,
        n_segments=n_segments,
    )


def _scan_core(
    offsets: t.Tensor,
    doc_ids: t.Tensor,
    segment_indices: t.Tensor,
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    n_segments: int,
    per_feature_stats: dict[str, PerFeatureStat],
) -> ScanResultTensors:
    """Pure tensor logic for scanning feature activations (steps 1–6).

    Expands segment-level arrays to feature-level, filters non-first segments,
    computes per-(feature, doc) max activations, total_segments, percentiles,
    and builds CSR-sorted output.

    Args:
        offsets: (n_segments+1,) int — segment offsets into feature arrays.
        doc_ids: (n_segments,) int — document ID per segment.
        segment_indices: (n_segments,) int — segment index within document.
        feature_ids: (total_features,) int — feature ID per activation.
        feature_mags: (total_features,) float — magnitude per activation.
        n_segments: Total number of segments.
        per_feature_stats: Per-feature statistics keyed by ``str(feature_id)``.

    Returns:
        :class:`ScanResultTensors` — compact CSR tensor representation.
    """
    t0 = _time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: Expand segment-level arrays to feature-level
    # ------------------------------------------------------------------
    t1 = _time.monotonic()
    seg_lengths = offsets[1:] - offsets[:-1]
    seg_for_feat = t.repeat_interleave(
        t.arange(n_segments, dtype=t.long),
        seg_lengths.long(),
    )
    doc_for_feat = doc_ids[seg_for_feat]
    segidx_for_feat = segment_indices[seg_for_feat]
    logger.info(f"  Step 1 (expand): {_time.monotonic() - t1:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Filter to non-first segments (segment_idx > 0)
    # ------------------------------------------------------------------
    t2 = _time.monotonic()
    mask = segidx_for_feat > 0
    n_kept = mask.sum().item()
    if n_kept == 0:
        empty = t.empty(0, dtype=t.int64)
        return ScanResultTensors(
            feature_ids=empty,
            n_docs_activated=empty,
            offsets=t.zeros(1, dtype=t.int64),
            doc_idx=empty,
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=empty,
            percentile=t.empty(0, dtype=t.float32),
            total_segments=empty,
        )

    fid = feature_ids[mask].long()
    mag = feature_mags[mask].float()
    doc = doc_for_feat[mask].long()
    segidx = segidx_for_feat[mask].long()

    val = mag
    logger.info(f"  Step 2 (filter): {n_kept:,} non-first entries ({_time.monotonic() - t2:.1f}s)")

    # ------------------------------------------------------------------
    # Step 3: Compute per-(feature_id, doc_id) max selection value
    # ------------------------------------------------------------------
    t3 = _time.monotonic()
    max_doc = int(doc.max().item())
    n_docs_plus_1 = max_doc + 1
    # Compound key encodes (feature, doc) pair as a single int for scatter_reduce.
    # Requires n_docs_plus_1 * max_fid < 2^63 to avoid int64 overflow.
    compound = fid * n_docs_plus_1 + doc

    n_groups = int(compound.max().item()) + 1
    group_max_mag = t.full((n_groups,), float("-inf"))
    group_max_mag.scatter_reduce_(0, compound, val, reduce="amax")

    is_max = val == group_max_mag[compound]

    group_max_segidx = t.full((n_groups,), -1, dtype=t.long)
    max_segidx_candidates = t.where(is_max, segidx, t.tensor(-1, dtype=t.long))
    group_max_segidx.scatter_reduce_(0, compound, max_segidx_candidates, reduce="amax")

    unique_compounds = t.unique(compound)
    group_fid = unique_compounds // n_docs_plus_1
    group_doc = unique_compounds % n_docs_plus_1
    max_mag_vec = group_max_mag[unique_compounds]
    max_segidx_vec = group_max_segidx[unique_compounds]
    logger.info(
        f"  Step 3 (scatter max): {len(unique_compounds):,} unique (feat,doc) pairs "
        f"({_time.monotonic() - t3:.1f}s)"
    )

    # ------------------------------------------------------------------
    # Step 4: Compute total_segments per doc (vectorized)
    # ------------------------------------------------------------------
    t4 = _time.monotonic()
    unique_docs, doc_counts = t.unique(doc_ids, return_counts=True)
    doc_total_seg_map = t.zeros(int(unique_docs.max().item()) + 1, dtype=t.int64)
    doc_total_seg_map[unique_docs.long()] = doc_counts.long()
    total_segs_vec = doc_total_seg_map[group_doc]
    logger.info(f"  Step 4 (total_segments): {_time.monotonic() - t4:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Percentile field
    # ------------------------------------------------------------------
    t5 = _time.monotonic()
    pct_vec = _vectorized_percentile(group_fid, max_mag_vec, per_feature_stats)
    logger.info(f"  Step 5 (percentiles): {_time.monotonic() - t5:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Sort by feature_id and build CSR offsets
    # ------------------------------------------------------------------
    t6 = _time.monotonic()
    sort_order = t.argsort(group_fid, stable=True)
    sorted_fid = group_fid[sort_order]
    sorted_doc = group_doc[sort_order]
    sorted_mag = max_mag_vec[sort_order]
    sorted_segidx = max_segidx_vec[sort_order]
    sorted_pct = pct_vec[sort_order]
    sorted_totseg = total_segs_vec[sort_order]

    # Unique features and counts for CSR offsets
    out_fids, feat_counts = t.unique(sorted_fid, return_counts=True)
    csr_offsets = t.zeros(len(out_fids) + 1, dtype=t.int64)
    csr_offsets[1:] = feat_counts.cumsum(0)
    logger.info(
        f"  Step 6 (sort + CSR): {len(out_fids):,} features "
        f"({_time.monotonic() - t6:.1f}s, total: {_time.monotonic() - t0:.1f}s)"
    )

    return ScanResultTensors(
        feature_ids=out_fids,
        n_docs_activated=feat_counts,
        offsets=csr_offsets,
        doc_idx=sorted_doc,
        max_activation=sorted_mag,
        max_segment_idx=sorted_segidx,
        percentile=sorted_pct,
        total_segments=sorted_totseg,
    )


def scan_validation_features_vectorized(
    shard_path: str | Path,
    per_feature_stats: dict[str, PerFeatureStat],
) -> ScanResultTensors:
    """Vectorized scan of a feature shard — returns tensors directly.

    Loads raw shard tensors via :func:`load_features` and delegates to
    :func:`_scan_core` for the actual computation.

    Args:
        shard_path: Path to the feature shard file (.safetensors or .pt).
        per_feature_stats: Per-feature statistics keyed by ``str(feature_id)``.

    Returns:
        :class:`ScanResultTensors` — compact tensor representation.
    """
    t0 = _time.monotonic()
    data = load_features(shard_path)

    offsets_raw: t.Tensor = data["offsets"]
    doc_ids: t.Tensor = data["doc_ids"]
    segment_indices: t.Tensor = data["segment_indices"]
    feature_ids: t.Tensor = data["feature_ids"]
    feature_mags: t.Tensor = data["feature_mags"]
    n_segments = int(data["n_segments"])
    logger.info(
        f"  Loaded shard: {n_segments} segments, "
        f"{feature_ids.numel():,} feature entries "
        f"({_time.monotonic() - t0:.1f}s)"
    )

    return _scan_core(
        offsets_raw,
        doc_ids,
        segment_indices,
        feature_ids,
        feature_mags,
        n_segments,
        per_feature_stats,
    )


def scan_thresholded_features(
    shard_path: str | Path,
    per_feature_stats: dict[str, PerFeatureStat],
    threshold_cfg: ThresholdConfig,
    precomputed_thresholds: t.Tensor | None,
) -> ScanResultTensors:
    """Load a shard, apply thresholding, then scan — returns thresholded scan.

    This combines :func:`load_features`, :func:`apply_threshold_to_shard`,
    and :func:`_scan_core` into a single pipeline so that feature selection
    operates on post-threshold data.

    Args:
        shard_path: Path to the feature shard file.
        per_feature_stats: Per-feature statistics keyed by ``str(feature_id)``.
        threshold_cfg: Threshold configuration.
        precomputed_thresholds: Precomputed per-feature threshold tensor.

    Returns:
        :class:`ScanResultTensors` — scan results on thresholded features.
    """
    t0 = _time.monotonic()
    data = load_features(shard_path)
    logger.info(
        f"  Loaded shard: {data['n_segments']} segments, "
        f"{data['feature_ids'].numel():,} feature entries "
        f"({_time.monotonic() - t0:.1f}s)"
    )

    t1 = _time.monotonic()
    thresholded = apply_threshold_to_shard(
        data,
        threshold_cfg,
        precomputed_thresholds,
    )
    n_before = data["feature_ids"].numel()
    n_after = thresholded.feature_ids.numel()
    logger.info(
        f"  Thresholding: {n_before:,} → {n_after:,} entries "
        f"({100 * n_after / max(n_before, 1):.1f}% retained, "
        f"{_time.monotonic() - t1:.1f}s)"
    )

    return _scan_core(
        thresholded.offsets,
        thresholded.doc_ids,
        thresholded.segment_indices,
        thresholded.feature_ids,
        thresholded.feature_mags,
        thresholded.n_segments,
        per_feature_stats,
    )


def scan_sampled_features(
    shard_path: str | Path,
    per_feature_stats: dict[str, PerFeatureStat],
    sampling_bundle: SamplingBundle,
) -> ScanResultTensors:
    """Load a shard, generate a percentile-based feature summary, then scan.

    For the sampling path, instead of hard thresholding we use the percentile
    scores from the sampling bundle's percentile table to rank features.
    Features with higher percentile scores are more likely to survive sampling,
    so we use them as a proxy for feature importance in the scan.

    This performs a deterministic scan (no stochastic sampling) — the scan
    stage needs reproducible results across runs.

    Args:
        shard_path: Path to the feature shard file.
        per_feature_stats: Per-feature statistics keyed by ``str(feature_id)``.
        sampling_bundle: Sampling config with percentile table.

    Returns:
        :class:`ScanResultTensors` — scan results using percentile-weighted features.
    """
    t0 = _time.monotonic()
    data = load_features(shard_path)
    logger.info(
        f"  Loaded shard: {data['n_segments']} segments, "
        f"{data['feature_ids'].numel():,} feature entries "
        f"({_time.monotonic() - t0:.1f}s)"
    )

    # For the sampling path, we use percentile scores to filter: keep features
    # whose percentile score is at least 1.0 (i.e. above p1 for their feature).
    # This is a light deterministic filter that approximates what sampling would
    # retain on average, without introducing stochastic variance into the scan.
    #
    # SCALE INVARIANT (map B-17 — verified against
    # ``steering.conditional.percentile_table.magnitude_to_percentile``,
    # which returns values in [0.5, 99.5] (0-100 scale, NOT [0, 1]):
    #   - Below p1  → 0.5
    #   - Above p99 → 99.5
    #   - Otherwise → idx + frac in (1.0, 99.0)
    # So the filter ``pct_scores >= 1.0`` is equivalent to "above p1" — it
    # excludes only the below-p1 values that map to 0.5. A scale change in
    # ``magnitude_to_percentile`` would silently break this invariant; the
    # contract is pinned by ``tests/audit_2026_05/slice_15_steering_evals/
    # test_scan_sampled_features_filter.py``.
    t1 = _time.monotonic()
    offsets_raw: t.Tensor = data["offsets"]
    doc_ids: t.Tensor = data["doc_ids"]
    segment_indices: t.Tensor = data["segment_indices"]
    feature_ids: t.Tensor = data["feature_ids"]
    feature_mags: t.Tensor = data["feature_mags"]
    n_segments = int(data["n_segments"])

    # Compute percentile scores for all features
    pct_scores = magnitude_to_percentile(
        feature_ids,
        feature_mags,
        sampling_bundle.percentile_table,
    )
    # Keep features with percentile >= 1.0 (above p1 threshold)
    keep_mask = pct_scores >= 1.0
    n_before = feature_ids.numel()
    n_after = int(keep_mask.sum().item())
    logger.info(
        f"  Percentile filter: {n_before:,} → {n_after:,} entries "
        f"({100 * n_after / max(n_before, 1):.1f}% retained, "
        f"{_time.monotonic() - t1:.1f}s)"
    )

    if n_after == 0:
        # Return empty scan results
        return ScanResultTensors(
            feature_ids=t.tensor([], dtype=t.int64),
            n_docs_activated=t.tensor([], dtype=t.int64),
            offsets=t.tensor([0], dtype=t.int64),
            doc_idx=t.tensor([], dtype=t.int64),
            max_activation=t.tensor([], dtype=t.float64),
            max_segment_idx=t.tensor([], dtype=t.int64),
            percentile=t.tensor([], dtype=t.float64),
            total_segments=t.tensor([], dtype=t.int64),
        )

    # Rebuild per-segment offsets for filtered data
    filtered_fids = feature_ids[keep_mask]
    filtered_mags = feature_mags[keep_mask]

    # Map each entry back to its segment via the CSR offsets
    n_total = offsets_raw.shape[0] - 1
    seg_lens = offsets_raw[1:] - offsets_raw[:-1]
    seg_idx_per_entry = t.repeat_interleave(t.arange(n_total, dtype=t.int64), seg_lens)
    filtered_seg_idx = seg_idx_per_entry[keep_mask]
    filtered_doc_ids = doc_ids[filtered_seg_idx]
    filtered_segment_indices = segment_indices[filtered_seg_idx]

    # Rebuild CSR offsets for filtered entries
    new_offsets = t.zeros(n_total + 1, dtype=t.int64)
    if n_after > 0:
        filtered_seg_counts = t.zeros(n_total, dtype=t.int64)
        filtered_seg_counts.scatter_add_(0, filtered_seg_idx, t.ones(n_after, dtype=t.int64))
        new_offsets[1:] = filtered_seg_counts.cumsum(0)

    return _scan_core(
        new_offsets,
        filtered_doc_ids,
        filtered_segment_indices,
        filtered_fids,
        filtered_mags,
        n_segments,
        per_feature_stats,
    )


class _ContextFeatureIndex(Protocol):
    def lookup(self, shard_stem: str, doc_idx: int) -> list[SegmentFeatures]: ...


if TYPE_CHECKING:
    _ContextFeatureIndexLike = _ContextFeatureIndex
else:
    _ContextFeatureIndexLike = object


@beartype
@dataclass(frozen=True)
class ContextFilter:
    """Parameters needed for context-length filtering during document selection."""

    doc_texts: _StringList
    feature_index: _ContextFeatureIndexLike
    val_shard_stem: str
    tokenizer: PrefixedTextTokenizer
    max_seq_len: PositiveIntLike
    bos_id: NonNegativeIntLike


@beartype
def doc_fits_context(
    doc_idx: NonNegativeIntLike,
    target_seg_idx: NonNegativeIntLike,
    ctx: ContextFilter,
    bos_prefix: str,
) -> bool:
    """Return True if tokens through *target_seg_idx* fit in the context window.

    Tokenizes text through the end of the target segment and checks that the
    total token count (prefix + target) does not exceed *max_seq_len*.
    """
    segments = ctx.feature_index.lookup(ctx.val_shard_stem, doc_idx)
    if len(segments) < 2:
        return True  # will be handled as a skip downstream
    target_seg_idx = max(1, min(target_seg_idx, len(segments) - 1))
    target_seg = segments[target_seg_idx]
    # char_end is defined relative to the BOS-prefixed text, so we must
    # prepend bos_prefix before slicing to get the correct character range.
    bos_text = bos_prefix + ctx.doc_texts[doc_idx]
    text_through_target = bos_text[len(bos_prefix) : target_seg.char_end]
    all_tokens = ctx.tokenizer.encode(text_through_target, prepend=ctx.bos_id)
    return len(all_tokens) <= ctx.max_seq_len


@beartype
def _segment_contains_feature(
    seg: SegmentFeatures,
    feature_id: NonNegativeIntLike,
) -> _BoolFloatPair:
    """Whether *feature_id* is active in *seg*, and at what selection value.

    Returns ``(present, value)`` where ``value`` is the raw magnitude, or ``-inf``
    when absent.
    """
    fids = seg.feature_ids
    mags = seg.feature_mags
    fid_list = fids.tolist()
    if feature_id not in [int(f) for f in fid_list]:
        return False, float("-inf")
    mag_list = mags.tolist()
    best = max(
        (float(m) for f, m in zip(fid_list, mag_list, strict=True) if int(f) == feature_id),
        default=float("-inf"),
    )
    return True, best


@beartype
def find_fitting_segment(
    doc_idx: NonNegativeIntLike,
    ctx: ContextFilter,
    bos_prefix: str,
    mode: EvalMode,
    feature_id: NonNegativeIntLike,
    preferred_seg_idx: NonNegativeIntLike | None = None,
    seed: StrictIntLike = 42,
) -> int | None:
    """Find a context-fitting segment for *doc_idx*, falling back from preferred.

    **Ablation mode**: returns the fitting segment with the highest selection
    value for *feature_id*.  If *preferred_seg_idx* fits, it is returned
    immediately (fast path).

    **Insertion mode**: returns a random fitting segment that does NOT contain
    *feature_id*.

    Segment 0 is never eligible in either mode.

    Returns ``None`` if no segment qualifies.
    """
    # Fast path (ablation only): preferred fits → use it
    if (
        preferred_seg_idx is not None
        and preferred_seg_idx >= 1
        and doc_fits_context(doc_idx, preferred_seg_idx, ctx, bos_prefix)
    ):
        return preferred_seg_idx

    segments = ctx.feature_index.lookup(ctx.val_shard_stem, doc_idx)
    if len(segments) < 2:
        logger.debug(
            f"find_fitting_segment: doc_idx={doc_idx} has {len(segments)} segments (need >= 2)"
        )
        return None

    if mode is EvalMode.ABLATION:
        # Find fitting segments >= 1 that contain feature_id, pick highest value.
        best_seg: int | None = None
        best_val: float = float("-inf")
        for seg in segments:
            if seg.segment_idx < 1:
                continue
            if not doc_fits_context(doc_idx, seg.segment_idx, ctx, bos_prefix):
                continue
            present, val = _segment_contains_feature(seg, feature_id)
            if present and val > best_val:
                best_val = val
                best_seg = seg.segment_idx
        return best_seg
    # Insertion: fitting segments >= 1 that do NOT contain feature_id.
    candidates: list[int] = []
    for seg in segments:
        if seg.segment_idx < 1:
            continue
        if not doc_fits_context(doc_idx, seg.segment_idx, ctx, bos_prefix):
            continue
        present, _ = _segment_contains_feature(seg, feature_id)
        if present:
            continue
        candidates.append(seg.segment_idx)
    if not candidates:
        logger.debug(
            f"find_fitting_segment: doc_idx={doc_idx} has no "
            f"candidates in insertion mode for feature {feature_id}"
        )
        return None
    return _random.Random(seed ^ doc_idx).choice(candidates)


def assert_selections_within_mask(selections: list, feature_mask: FeatureMask | None) -> None:
    """Defense-in-depth (I4): no selected feature_id violates the mask.

    No-op when feature_mask is None (I5). Used at every point the
    completion-steering pipeline materialises or loads selections,
    so a stale pre-mask feature_selections.yaml or any selection-path
    regression is caught loudly (FeatureMaskError) rather than silently
    steering on a blacklisted feature.
    """
    if feature_mask is None or not selections:
        return
    fids = t.tensor([s.feature_id for s in selections], dtype=t.int64)
    feature_mask.assert_no_masked(fids)


@beartype
def select_from_tensors(
    scan_tensors: ScanResultTensors,
    top_k: NonNegativeIntLike,
    docs_per_feature: NonNegativeIntLike,
    mode: EvalMode,
    all_doc_indices: NonNegativeIntSetLike | None = None,
    seed: StrictIntLike = 42,
    ctx_filter: ContextFilter | None = None,
    feature_mask: FeatureMask | None = None,
) -> _RawFeatureSelectionList:
    """Randomly select top_k features and sample documents from tensors.

    Selects directly from CSR tensors without materialising Python dataclass
    objects for all features; only the selected features' document indices are
    converted to Python lists.

    When *ctx_filter* is provided, candidate documents whose tokens through
    the target segment exceed the context window are excluded **before**
    sampling.

    In insertion mode, a random target segment is selected per document
    (seeded by ``seed ^ doc_idx``) and stored in
    :attr:`RawFeatureSelection.target_segments`.

    Args:
        scan_tensors: CSR-layout :class:`ScanResultTensors` from the scan stage.
        top_k: Number of features to randomly select.
        docs_per_feature: Maximum number of documents to sample per feature.
        mode: ``EvalMode.ABLATION`` — sample docs that activated the feature;
              ``EvalMode.INSERTION`` — sample docs that did NOT activate the feature
              (requires *all_doc_indices*).
        all_doc_indices: Full set of document indices in the validation corpus.
            Required when mode is EvalMode.INSERTION.
        seed: Random seed for reproducible sampling.
        feature_mask: Optional :class:`FeatureMask` restricting the feature-selection
            universe. When None (default), all features are candidates and
            selection is byte-identical to pre-mask runs (invariant I5). When set,
            only mask-allowed feature IDs are eligible; the seeded top-k draw
            samples from the allowed subset only.
        ctx_filter: Optional :class:`ContextFilter` for context-length filtering.

    Returns:
        List of :class:`RawFeatureSelection` objects, one per randomly selected
        feature.  Attach descriptions post-selection to produce :class:`FeatureSelection`.
    """
    if mode is EvalMode.INSERTION and all_doc_indices is None:
        raise ValueError("all_doc_indices is required when mode is EvalMode.INSERTION")

    has_ctx = ctx_filter is not None
    bos_prefix = ctx_filter.tokenizer.decode([ctx_filter.bos_id]) if has_ctx else ""

    n_features = len(scan_tensors.feature_ids)

    effective_mask = feature_mask

    # I5 ANTI-REGRESSION: effective_mask is None ⇒ candidate_pos is None ⇒ this
    # is byte-identical to the pre-mask randperm(n_features) selection (same RNG
    # draw, same order). NEVER restructure so the no-mask path consumes RNG
    # differently. The masked path draws randperm(n_candidates) over the allowed
    # subset only — different RNG draw, but provably correct (subset selection).
    if effective_mask is None:
        candidate_pos = None
        n_candidates = n_features
    else:
        keep = effective_mask.keep_mask(scan_tensors.feature_ids)  # bool[n_features]
        candidate_pos = keep.nonzero(as_tuple=True)[0]  # positions into scan arrays
        n_candidates = int(candidate_pos.numel())
    if n_candidates == 0 or top_k == 0:
        return []

    gen = t.Generator()
    gen.manual_seed(seed)
    perm = t.randperm(n_candidates, generator=gen)
    chosen = perm[: min(top_k, n_candidates)]
    selected_indices = chosen if candidate_pos is None else candidate_pos[chosen]

    rng = _random.Random(seed)
    offsets = scan_tensors.offsets
    selections: list[RawFeatureSelection] = []
    n_ctx_filtered = 0
    n_ctx_fallback = 0

    for idx in selected_indices.tolist():
        fid = int(scan_tensors.feature_ids[idx].item())
        n_docs = int(scan_tensors.n_docs_activated[idx].item())

        # Extract this feature's doc_idx from the CSR layout.
        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        activated_docs: list[int] = scan_tensors.doc_idx[start:end].tolist()

        target_segments: dict[int, int] = {}

        if mode is EvalMode.ABLATION:
            candidate_docs = activated_docs

            if has_ctx:
                assert ctx_filter is not None
                max_seg_indices = scan_tensors.max_segment_idx[start:end].tolist()
                filtered = []
                for doc_idx, max_seg in zip(candidate_docs, max_seg_indices, strict=True):
                    fitting_seg = find_fitting_segment(
                        doc_idx,
                        ctx_filter,
                        bos_prefix,
                        mode=EvalMode.ABLATION,
                        feature_id=fid,
                        preferred_seg_idx=int(max_seg),
                    )
                    if fitting_seg is not None:
                        filtered.append(doc_idx)
                        target_segments[doc_idx] = fitting_seg
                        if fitting_seg != int(max_seg):
                            n_ctx_fallback += 1
                    else:
                        n_ctx_filtered += 1
                candidate_docs = filtered
        else:
            activated_set = set(activated_docs)
            assert all_doc_indices is not None
            candidate_docs = sorted(all_doc_indices - activated_set)

            if has_ctx:
                assert ctx_filter is not None
                filtered = []
                for doc_idx in candidate_docs:
                    fitting_seg = find_fitting_segment(
                        doc_idx,
                        ctx_filter,
                        bos_prefix,
                        mode=EvalMode.INSERTION,
                        feature_id=fid,
                        seed=seed,
                    )
                    if fitting_seg is not None:
                        target_segments[doc_idx] = fitting_seg
                        filtered.append(doc_idx)
                    else:
                        n_ctx_filtered += 1
                candidate_docs = filtered
            else:
                # No context filter — target_segments populated at generation time
                pass

        sampled = rng.sample(candidate_docs, min(docs_per_feature, len(candidate_docs)))

        # Trim target_segments to only sampled docs
        if target_segments:
            target_segments = {d: target_segments[d] for d in sampled if d in target_segments}

        selections.append(
            RawFeatureSelection(
                feature_id=fid,
                n_docs_activated=n_docs,
                selected_docs=sampled,
                target_segments=target_segments,
            )
        )

    if n_ctx_filtered > 0 or n_ctx_fallback > 0:
        logger.info(
            f"Context-length filter: excluded {n_ctx_filtered} candidate doc(s), "
            f"salvaged {n_ctx_fallback} via fallback "
            f"(max_seq_len={ctx_filter.max_seq_len if ctx_filter else '?'})"
        )

    return selections


@beartype
def build_target_segment_maps(
    scan_tensors: ScanResultTensors,
    feature_ids: NonNegativeIntListLike,
) -> _TargetSegmentMaps:
    """Build doc_idx → max_segment_idx maps for selected features from CSR tensors.

    Extracts only the requested features' data from the CSR layout, avoiding
    a full conversion of every activation to Python objects.

    Args:
        scan_tensors: CSR-layout :class:`ScanResultTensors`.
        feature_ids: Feature IDs to extract maps for.

    Returns:
        ``{feature_id: {doc_idx: max_segment_idx}}`` for each requested feature
        present in the scan tensors.  Features not found are omitted.
    """
    result: dict[int, dict[int, int]] = {}
    if len(scan_tensors.feature_ids) == 0:
        return result

    fids_tensor = scan_tensors.feature_ids
    offsets = scan_tensors.offsets

    for fid in feature_ids:
        # Binary search for feature_id in the sorted feature_ids tensor
        idx = int(t.searchsorted(fids_tensor, fid).item())
        if idx >= len(fids_tensor) or int(fids_tensor[idx].item()) != fid:
            continue  # feature not in scan results

        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        doc_idxs = scan_tensors.doc_idx[start:end].tolist()
        seg_idxs = scan_tensors.max_segment_idx[start:end].tolist()
        result[fid] = dict(zip(doc_idxs, seg_idxs, strict=True))

    return result


# ---------------------------------------------------------------------------
# Scan results — tensor representation + I/O
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResultTensors:
    """Compact tensor representation of scan results (ragged CSR layout).

    All per-activation arrays share a common index space: the activations for
    feature ``feature_ids[i]`` span ``doc_idx[offsets[i]:offsets[i+1]]``, etc.

    Attributes:
        feature_ids:      (N,)           int64  — sorted feature IDs
        n_docs_activated: (N,)           int64
        offsets:          (N+1,)         int64  — CSR-style cumulative offsets
        doc_idx:          (total_acts,)  int64
        max_activation:   (total_acts,)  float32
        max_segment_idx:  (total_acts,)  int64
        percentile:       (total_acts,)  float32
        total_segments:   (total_acts,)  int64
    """

    feature_ids: t.Tensor
    n_docs_activated: t.Tensor
    offsets: t.Tensor
    doc_idx: t.Tensor
    max_activation: t.Tensor
    max_segment_idx: t.Tensor
    percentile: t.Tensor
    total_segments: t.Tensor


def save_scan_results(scan_results: ScanResultTensors, path: Path) -> None:
    """Save :class:`ScanResultTensors` to a safetensors file."""
    save_file(
        {
            "feature_ids": scan_results.feature_ids,
            "n_docs_activated": scan_results.n_docs_activated,
            "offsets": scan_results.offsets,
            "doc_idx": scan_results.doc_idx,
            "max_activation": scan_results.max_activation,
            "max_segment_idx": scan_results.max_segment_idx,
            "percentile": scan_results.percentile,
            "total_segments": scan_results.total_segments,
        },
        str(path),
    )


def load_scan_results(path: Path) -> ScanResultTensors:
    """Load scan results from a safetensors file into :class:`ScanResultTensors`."""
    tensors = load_file(str(path))
    return ScanResultTensors(
        feature_ids=tensors["feature_ids"],
        n_docs_activated=tensors["n_docs_activated"],
        offsets=tensors["offsets"],
        doc_idx=tensors["doc_idx"],
        max_activation=tensors["max_activation"],
        max_segment_idx=tensors["max_segment_idx"],
        percentile=tensors["percentile"],
        total_segments=tensors["total_segments"],
    )
