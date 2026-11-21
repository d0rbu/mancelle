"""Feature delta conditioning for cross-segment feature selection.

Provides:
- DeltaMode: enum for delta reference type (prev_segment, cum_max, splice_counterfactual)
- DeltaSamplingWeights: enum for sampling weight source (absolute_percentile, percentile_delta)
- FeatureDeltaConfig: frozen config dataclass for delta conditioning
- parse_feature_delta: parse and validate feature_delta config block
- compute_doc_deltas: document-batched cross-segment delta computation and filtering
- CounterfactualRef: type alias for an external per-segment reference (ref_fids, ref_mags, ref_pcts)
- compute_doc_counterfactual_deltas: per-segment counterfactual deltas against external references
- compute_doc_clustered_counterfactual_deltas: mask/cluster each side before subtraction
- sample_delta_negatives: delta-aware negative feature sampling
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, NamedTuple

import torch as t
from beartype import beartype
from beartype.vale import Is

from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.percentile_table import magnitude_to_percentile
from steering.refined_types import (
    NonBlankStrLike,
    NonNegativeIntLike,
    PositiveIntLike,
    UnitInterval,
    UnitIntervalLike,
)

_PROFILE_DELTA = bool(os.environ.get("ENCDEC_PROFILE", ""))

# Pre-allocated empty tensor constants (avoids ~354K allocations per RG)
_EMPTY_INT64 = t.tensor([], dtype=t.int64)
_EMPTY_FLOAT32 = t.tensor([], dtype=t.float32)
_FEATURE_DELTA_KEYS = frozenset({"mode", "neg_delta_ratio"})

if TYPE_CHECKING:
    _ManifestIdentity = dict[str, int | str]
    _OptionalTensorList = list[t.Tensor | None]
    _TensorList = list[t.Tensor]
else:
    _ManifestIdentity = dict
    _OptionalTensorList = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list)
                and all(item is None or isinstance(item, t.Tensor) for item in value)
            )
        ],
    ]
    _TensorList = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list) and all(isinstance(item, t.Tensor) for item in value)
            )
        ],
    ]

# Conditional identity for the only path whose cluster/delta ordering differs
# from the generic static-cluster contract.  Keep this separate from
# FEATURE_CLUSTERS_SEMANTICS: changing the latter would unnecessarily
# invalidate delta-free clustered selection artifacts.
CLUSTER_DELTA_SEMANTICS_KEY = "cluster_delta_semantics"
CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS = (
    "mask_then_cluster_each_side_max_percentile_then_delta_v1"
)
SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION = 1
CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION = 2


@beartype
def splice_counterfactual_manifest_identity(*, clustered: bool) -> _ManifestIdentity:
    """Return the conditional manifest identity for splice-counterfactual data."""
    if not clustered:
        return {"schema_version": SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION}
    return {
        "schema_version": CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION,
        CLUSTER_DELTA_SEMANTICS_KEY: CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    }


def _raise_empty_delta_segment(context: str) -> None:
    raise ValueError(
        f"{context} has zero feature ids. Delta conditioning requires every "
        "raw extraction segment and counterfactual reference segment to contain "
        "at least one active feature; regenerate or inspect the feature shard."
    )


class DeltaMode(Enum):
    prev_segment = "prev_segment"
    cum_max = "cum_max"
    splice_counterfactual = "splice_counterfactual"


class DeltaSamplingWeights(Enum):
    absolute_percentile = "absolute_percentile"
    percentile_delta = "percentile_delta"


@dataclass(frozen=True, init=False)
class FeatureDeltaConfig:
    """Configuration for feature delta conditioning."""

    mode: DeltaMode
    neg_delta_ratio: UnitInterval

    @beartype
    def __init__(
        self,
        mode: DeltaMode,
        neg_delta_ratio: UnitIntervalLike = 1.0,
    ) -> None:
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "neg_delta_ratio",
            UnitInterval.parse(float(neg_delta_ratio)),
        )


def parse_feature_delta(cfg: dict | None) -> FeatureDeltaConfig | None:
    """Parse and validate the feature_delta config block.

    Returns None if cfg is None (delta conditioning disabled).
    Otherwise validates mode and neg_delta_ratio and returns FeatureDeltaConfig.

    Invalid keys or modes raise ``ValueError``; the beartyped constructor
    rejects an out-of-range ``neg_delta_ratio`` at the owning boundary.
    """
    if cfg is None:
        return None

    unknown = sorted(str(key) for key in cfg if key not in _FEATURE_DELTA_KEYS)
    if unknown:
        raise ValueError(
            "feature_delta config contains unknown keys "
            f"{unknown}. Allowed keys: {sorted(_FEATURE_DELTA_KEYS)}."
        )

    if "mode" not in cfg:
        raise ValueError(
            "feature_delta config must contain 'mode'. "
            f"Supported: {sorted(m.value for m in DeltaMode)}."
        )

    raw_mode = cfg["mode"]
    try:
        mode = DeltaMode(raw_mode)
    except ValueError:
        supported = sorted(m.value for m in DeltaMode)
        raise ValueError(f"Unsupported delta mode '{raw_mode}'. Supported: {supported}.") from None

    return FeatureDeltaConfig(
        mode=mode,
        neg_delta_ratio=cfg.get("neg_delta_ratio", UnitInterval.parse(1.0)),
    )


# ---------------------------------------------------------------------------
# Batched delta computation (processes entire doc in one call)
# ---------------------------------------------------------------------------

# Result tuple type for each segment from compute_doc_deltas
# (filtered_fids, filtered_fmags, delta_values, neg_delta_fids, neg_delta_weights,
#  ref_mask, filtered_pcts)
DeltaSegResult = tuple[
    t.Tensor, t.Tensor, t.Tensor | None, t.Tensor, t.Tensor, t.Tensor, t.Tensor | None
]

if TYPE_CHECKING:
    _DeltaSegResultList = list[DeltaSegResult]
else:
    _DeltaSegResultList = list


@beartype
def compute_doc_deltas(
    seg_fids: _TensorList,
    seg_mags: _TensorList,
    seg_pcts: _OptionalTensorList,
    n_sae: PositiveIntLike,
    mode: DeltaMode,
    percentile_table: t.Tensor | None = None,
) -> _DeltaSegResultList:
    """Compute delta filtering for all segments of one document in a single call.

    Fuses delta computation + positive filtering + negative extraction into a
    tight loop, eliminating per-segment method dispatch overhead. For
    prev_segment mode, each segment's delta depends only on the immediately
    preceding segment (known upfront).

    Args:
        seg_fids: Per-segment feature ID tensors.
        seg_mags: Per-segment feature magnitude tensors.
        seg_pcts: Per-segment precomputed percentiles (float32 [0,1]) or None.
            When None, uses magnitude_to_percentile with percentile_table.
        n_sae: Total SAE feature count (for mask buffer size).
        mode: DeltaMode (prev_segment or cum_max).
        percentile_table: Required when any seg_pcts entry is None.

    Returns:
        List of (filtered_fids, filtered_fmags, delta_values,
                 neg_delta_fids, neg_delta_weights, ref_mask, filtered_pcts)
        per segment. filtered_pcts is the precomputed percentiles for
        positive-delta features (None when seg_pcts was None).
    """
    if mode is DeltaMode.splice_counterfactual:
        raise ValueError(
            "compute_doc_deltas does not support splice_counterfactual (the mode "
            "has no carried state); use compute_doc_counterfactual_deltas."
        )
    n_segs = len(seg_fids)
    results: list[DeltaSegResult] = []

    # Reusable buffers (allocated once, reused across segments)
    ref = t.zeros(n_sae, dtype=t.float32)
    mask_buf = t.zeros(n_sae, dtype=t.bool)
    prev_fids: t.Tensor | None = None
    is_prev_seg = mode is DeltaMode.prev_segment

    for s in range(n_segs):
        fids = seg_fids[s]
        mags = seg_mags[s]
        n = len(fids)

        if n == 0:
            _raise_empty_delta_segment(f"compute_doc_deltas segment {s}")

        # --- Percentile resolution ---
        pcts = seg_pcts[s]
        if pcts is not None:
            pcts = pcts.float()
        else:
            assert percentile_table is not None
            pcts = (magnitude_to_percentile(fids, mags, percentile_table) / 100.0).float()

        fids_long = fids if fids.dtype == t.int64 else fids.long()

        # --- Delta computation + positive filter (fused) ---
        current_deltas = pcts - ref[fids_long]
        pos_mask = current_deltas > 0
        filtered_fids = fids[pos_mask]
        filtered_fmags = mags[pos_mask]
        delta_values = current_deltas[pos_mask]
        # Filtered percentiles for downstream sampling
        filtered_pcts = pcts[pos_mask] if seg_pcts[s] is not None else None

        # --- Reference-only features (using mask_buf, proven fast) ---
        if prev_fids is not None and len(prev_fids) > 0:
            mask_buf.zero_()
            mask_buf[fids_long] = True
            ref_only_mask = ~mask_buf[prev_fids]
            ref_only_fids = prev_fids[ref_only_mask]
        else:
            ref_only_fids = _EMPTY_INT64

        # --- Negative delta tensors (optimized branching) ---
        neg_mask = current_deltas < 0
        has_neg = bool(neg_mask.any())
        has_ref_only = len(ref_only_fids) > 0

        if has_ref_only and has_neg:
            neg_delta_fids = t.cat([ref_only_fids, fids_long[neg_mask]])
            neg_delta_weights = t.cat(
                [ref[ref_only_fids].float(), (-current_deltas[neg_mask]).float()]
            )
        elif has_ref_only:
            neg_delta_fids = ref_only_fids.to(t.int64)
            neg_delta_weights = ref[ref_only_fids].float()
        elif has_neg:
            neg_delta_fids = fids_long[neg_mask].to(t.int64)
            neg_delta_weights = (-current_deltas[neg_mask]).float()
        else:
            neg_delta_fids = _EMPTY_INT64
            neg_delta_weights = _EMPTY_FLOAT32

        # --- ref_mask = union(current, previous) ---
        mask_buf.zero_()
        mask_buf[fids_long] = True
        if prev_fids is not None and len(prev_fids) > 0:
            mask_buf[prev_fids] = True
        ref_mask = mask_buf.clone()

        # --- State update ---
        if is_prev_seg:
            ref.zero_()
            ref[fids_long] = pcts
            prev_fids = fids_long
        else:  # cum_max
            ref[fids_long] = t.maximum(ref[fids_long], pcts)
            if prev_fids is not None and len(prev_fids) > 0:
                prev_fids = t.cat([prev_fids, fids_long]).unique()
            else:
                prev_fids = fids_long

        results.append(
            (
                filtered_fids,
                filtered_fmags,
                delta_values,
                neg_delta_fids,
                neg_delta_weights,
                ref_mask,
                filtered_pcts,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Counterfactual (splice) delta computation — issue #596
# ---------------------------------------------------------------------------

# External per-segment reference: (ref_fids, ref_mags, ref_pcts_or_None).
# ref_pcts, when given, is float32 normalized [0, 1] (same convention as
# seg_pcts); when None the reference percentiles are computed from ref_mags
# via percentile_table exactly like the current segment's.
CounterfactualRef = tuple[t.Tensor, t.Tensor, t.Tensor | None]

if TYPE_CHECKING:
    _CounterfactualRefList = list[CounterfactualRef | None]
else:
    _CounterfactualRefList = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list)
                and all(
                    item is None
                    or (
                        isinstance(item, tuple)
                        and len(item) == 3
                        and isinstance(item[0], t.Tensor)
                        and isinstance(item[1], t.Tensor)
                        and (item[2] is None or isinstance(item[2], t.Tensor))
                    )
                    for item in value
                )
            )
        ],
    ]


class ClusteredCounterfactualDeltaSegResult(NamedTuple):
    """Representative-level splice delta result.

    This remains tuple-compatible with :data:`DeltaSegResult`, while giving
    materialization an unambiguous signal that mask/cluster consolidation has
    already happened and must not be repeated.
    """

    filtered_fids: t.Tensor
    filtered_fmags: t.Tensor
    delta_values: t.Tensor
    neg_delta_fids: t.Tensor
    neg_delta_weights: t.Tensor
    ref_mask: t.Tensor
    filtered_pcts: t.Tensor


if TYPE_CHECKING:
    _ClusteredDeltaSegResultList = list[ClusteredCounterfactualDeltaSegResult | None]
    _CounterfactualDeltaSegResultList = list[DeltaSegResult | None]
else:
    _ClusteredDeltaSegResultList = list
    _CounterfactualDeltaSegResultList = list


@beartype
def _validate_counterfactual_side_inputs(
    *,
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    feature_pcts: t.Tensor | None,
    n_sae_features: PositiveIntLike,
    surface: NonBlankStrLike,
) -> t.Tensor:
    """Validate one raw side before its mask is allowed to discard anything."""
    if feature_ids.ndim != 1:
        raise ValueError(
            f"{surface} feature IDs must be one-dimensional, got shape={tuple(feature_ids.shape)}"
        )
    if feature_ids.dtype is t.bool or feature_ids.is_floating_point() or feature_ids.is_complex():
        raise ValueError(
            f"{surface} feature IDs must use an integer dtype, got {feature_ids.dtype}"
        )
    ids = feature_ids.to(dtype=t.int64)
    if ids.numel() > 0:
        invalid = (ids < 0) | (ids >= n_sae_features)
        if bool(invalid.any()):
            bad = int(ids[invalid][0].item())
            raise ValueError(f"{surface} feature ID {bad} is out of range [0, {n_sae_features})")
    mags_are_required = feature_pcts is None
    mags_are_malformed = feature_mags.ndim != 1 or feature_mags.shape != ids.shape
    if mags_are_malformed and (mags_are_required or feature_mags.numel() != 0):
        raise ValueError(
            f"{surface} feature magnitudes must have shape {tuple(ids.shape)}, "
            f"got {tuple(feature_mags.shape)}"
        )
    if feature_pcts is not None and (feature_pcts.ndim != 1 or feature_pcts.shape != ids.shape):
        raise ValueError(
            f"{surface} feature percentiles must have shape {tuple(ids.shape)}, "
            f"got {tuple(feature_pcts.shape)}"
        )
    return ids


@beartype
def _mask_and_consolidate_counterfactual_side(
    *,
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    feature_pcts: t.Tensor | None,
    feature_clusters: FeatureClusterConsolidation,
    percentile_table: t.Tensor | None,
    surface: NonBlankStrLike,
) -> tuple[t.Tensor, t.Tensor]:
    ids = _validate_counterfactual_side_inputs(
        feature_ids=feature_ids,
        feature_mags=feature_mags,
        feature_pcts=feature_pcts,
        n_sae_features=feature_clusters.n_sae_features,
        surface=surface,
    )
    mags = feature_mags
    pcts = feature_pcts
    feature_mask = feature_clusters.feature_mask
    if feature_mask is not None:
        if pcts is None:
            ids, mags = feature_mask.apply_to_positive(ids, mags)
        else:
            ids, pcts = feature_mask.apply_to_positive(ids, pcts)

    if pcts is not None:
        resolved_pcts = pcts.float()
    else:
        if percentile_table is None:
            raise ValueError(
                f"percentile_table is required when {surface} has no precomputed percentiles"
            )
        resolved_pcts = (magnitude_to_percentile(ids, mags, percentile_table) / 100.0).float()

    consolidated_ids, consolidated_pcts = feature_clusters.consolidate_segment(
        ids,
        resolved_pcts,
    )
    return consolidated_ids, consolidated_pcts


@beartype
def consolidate_counterfactual_percentile_sides(
    *,
    current_fids: t.Tensor,
    current_mags: t.Tensor,
    current_pcts: t.Tensor | None,
    reference_fids: t.Tensor,
    reference_mags: t.Tensor,
    reference_pcts: t.Tensor | None,
    feature_clusters: FeatureClusterConsolidation,
    percentile_table: t.Tensor | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Mask and independently max-consolidate both counterfactual sides.

    Returns ``(current_reps, current_max_pcts, reference_reps,
    reference_max_pcts)``.  Either side may become empty after masking.  Raw
    empty-segment rejection remains the responsibility of the document-level
    delta function so this primitive can also express the post-mask edge case.
    """
    current_reps, current_max_pcts = _mask_and_consolidate_counterfactual_side(
        feature_ids=current_fids,
        feature_mags=current_mags,
        feature_pcts=current_pcts,
        feature_clusters=feature_clusters,
        percentile_table=percentile_table,
        surface="counterfactual current side",
    )
    reference_reps, reference_max_pcts = _mask_and_consolidate_counterfactual_side(
        feature_ids=reference_fids,
        feature_mags=reference_mags,
        feature_pcts=reference_pcts,
        feature_clusters=feature_clusters,
        percentile_table=percentile_table,
        surface="counterfactual reference side",
    )
    return (
        current_reps,
        current_max_pcts,
        reference_reps,
        reference_max_pcts,
    )


@beartype
def compute_doc_clustered_counterfactual_deltas(
    seg_fids: _TensorList,
    seg_mags: _TensorList,
    seg_pcts: _OptionalTensorList,
    seg_refs: _CounterfactualRefList,
    feature_clusters: FeatureClusterConsolidation,
    percentile_table: t.Tensor | None = None,
) -> _ClusteredDeltaSegResultList:
    """Compute splice deltas after independently clustering both masked sides.

    Raw current/reference segments retain the existing fail-loud non-empty
    contract.  Masking may legitimately empty either or both sides.  Positive
    candidates, signed-negative candidates, current percentiles, and presence
    are all emitted exclusively in representative ID space.
    """
    n_segs = len(seg_fids)
    if not (len(seg_mags) == len(seg_pcts) == len(seg_refs) == n_segs):
        raise ValueError(
            "clustered counterfactual document inputs must have equal lengths: "
            f"fids={n_segs}, mags={len(seg_mags)}, pcts={len(seg_pcts)}, "
            f"refs={len(seg_refs)}"
        )

    n_sae = feature_clusters.n_sae_features
    results: list[ClusteredCounterfactualDeltaSegResult | None] = []
    ref = t.zeros(n_sae, dtype=t.float32)
    mask_buf = t.zeros(n_sae, dtype=t.bool)

    for s, (fids, mags, pcts, cf) in enumerate(
        zip(seg_fids, seg_mags, seg_pcts, seg_refs, strict=True)
    ):
        if len(fids) == 0:
            _raise_empty_delta_segment(f"compute_doc_clustered_counterfactual_deltas segment {s}")
        if cf is None:
            results.append(None)
            continue

        ref_fids, ref_mags, ref_pcts = cf
        if len(ref_fids) == 0:
            _raise_empty_delta_segment(
                f"compute_doc_clustered_counterfactual_deltas segment {s} reference"
            )
        (
            current_reps,
            current_vals,
            reference_reps,
            reference_vals,
        ) = consolidate_counterfactual_percentile_sides(
            current_fids=fids,
            current_mags=mags,
            current_pcts=pcts,
            reference_fids=ref_fids,
            reference_mags=ref_mags,
            reference_pcts=ref_pcts,
            feature_clusters=feature_clusters,
            percentile_table=percentile_table,
        )

        ref[reference_reps] = reference_vals
        current_deltas = current_vals - ref[current_reps]
        pos_mask = current_deltas > 0
        filtered_fids = current_reps[pos_mask]
        filtered_pcts = current_vals[pos_mask]
        delta_values = current_deltas[pos_mask]

        mask_buf.zero_()
        mask_buf[current_reps] = True
        ref_only_mask = ~mask_buf[reference_reps]
        ref_only_fids = reference_reps[ref_only_mask]

        neg_mask = current_deltas < 0
        has_neg = bool(neg_mask.any())
        has_ref_only = len(ref_only_fids) > 0
        if has_ref_only and has_neg:
            neg_delta_fids = t.cat([ref_only_fids, current_reps[neg_mask]])
            neg_delta_weights = t.cat(
                [
                    ref[ref_only_fids].float(),
                    (-current_deltas[neg_mask]).float(),
                ]
            )
        elif has_ref_only:
            neg_delta_fids = ref_only_fids.to(t.int64)
            neg_delta_weights = ref[ref_only_fids].float()
        elif has_neg:
            neg_delta_fids = current_reps[neg_mask].to(t.int64)
            neg_delta_weights = (-current_deltas[neg_mask]).float()
        else:
            neg_delta_fids = _EMPTY_INT64
            neg_delta_weights = _EMPTY_FLOAT32

        mask_buf.zero_()
        mask_buf[current_reps] = True
        mask_buf[reference_reps] = True
        ref_mask = mask_buf.clone()
        results.append(
            ClusteredCounterfactualDeltaSegResult(
                filtered_fids=filtered_fids,
                filtered_fmags=t.zeros_like(filtered_pcts, dtype=t.float32),
                delta_values=delta_values,
                neg_delta_fids=neg_delta_fids,
                neg_delta_weights=neg_delta_weights,
                ref_mask=ref_mask,
                filtered_pcts=filtered_pcts,
            )
        )
        ref[reference_reps] = 0.0

    return results


@beartype
def compute_doc_counterfactual_deltas(
    seg_fids: _TensorList,
    seg_mags: _TensorList,
    seg_pcts: _OptionalTensorList,
    seg_refs: _CounterfactualRefList,
    n_sae: PositiveIntLike,
    percentile_table: t.Tensor | None = None,
) -> _CounterfactualDeltaSegResultList:
    """Per-segment counterfactual deltas against external references.

    Every current segment must contain at least one feature id. For each
    segment ``s``: when ``seg_refs[s]`` is None the segment gets NO delta
    treatment — the result slot is None and the caller must apply ordinary
    absolute-percentile conditioning. When a reference is given, both current
    and reference segments must be non-empty and the segment is treated exactly
    like one prev_segment step whose previous segment is the reference:
    ``delta = pct(current) - pct(ref)``; positive deltas are selected; ref-only
    features become negative candidates weighted by their ref percentile;
    negative-delta current features are weighted by ``|delta|``; ``ref_mask``
    is the current∪ref union.

    Unlike prev_segment/cum_max there is NO carried state across segments —
    each reference is installed for its segment and scatter-cleared after.
    """
    n_segs = len(seg_fids)
    if len(seg_refs) != n_segs:
        raise ValueError(f"seg_refs length {len(seg_refs)} != seg_fids length {n_segs}")
    results: list[DeltaSegResult | None] = []
    ref = t.zeros(n_sae, dtype=t.float32)
    mask_buf = t.zeros(n_sae, dtype=t.bool)

    for s in range(n_segs):
        fids = seg_fids[s]
        mags = seg_mags[s]
        n = len(fids)
        if n == 0:
            _raise_empty_delta_segment(f"compute_doc_counterfactual_deltas segment {s}")

        cf = seg_refs[s]
        if cf is None:
            results.append(None)
            continue
        ref_fids, ref_mags, ref_pcts = cf
        ref_fids_long = ref_fids if ref_fids.dtype == t.int64 else ref_fids.long()
        if len(ref_fids_long) == 0:
            _raise_empty_delta_segment(f"compute_doc_counterfactual_deltas segment {s} reference")
        if ref_pcts is not None:
            ref_vals = ref_pcts.float()
        else:
            assert percentile_table is not None, (
                "percentile_table required when a reference has no precomputed percentiles"
            )
            ref_vals = (
                magnitude_to_percentile(ref_fids_long, ref_mags, percentile_table) / 100.0
            ).float()
        ref[ref_fids_long] = ref_vals

        pcts = seg_pcts[s]
        if pcts is not None:
            pcts = pcts.float()
        else:
            assert percentile_table is not None
            pcts = (magnitude_to_percentile(fids, mags, percentile_table) / 100.0).float()
        fids_long = fids if fids.dtype == t.int64 else fids.long()

        current_deltas = pcts - ref[fids_long]
        pos_mask = current_deltas > 0
        filtered_fids = fids[pos_mask]
        filtered_fmags = mags[pos_mask]
        delta_values = current_deltas[pos_mask]
        filtered_pcts = pcts[pos_mask] if seg_pcts[s] is not None else None

        mask_buf.zero_()
        mask_buf[fids_long] = True
        ref_only_mask = ~mask_buf[ref_fids_long]
        ref_only_fids = ref_fids_long[ref_only_mask]

        neg_mask = current_deltas < 0
        has_neg = bool(neg_mask.any())
        has_ref_only = len(ref_only_fids) > 0
        if has_ref_only and has_neg:
            neg_delta_fids = t.cat([ref_only_fids, fids_long[neg_mask]])
            neg_delta_weights = t.cat(
                [
                    ref[ref_only_fids].float(),
                    (-current_deltas[neg_mask]).float(),
                ]
            )
        elif has_ref_only:
            neg_delta_fids = ref_only_fids.to(t.int64)
            neg_delta_weights = ref[ref_only_fids].float()
        elif has_neg:
            neg_delta_fids = fids_long[neg_mask].to(t.int64)
            neg_delta_weights = (-current_deltas[neg_mask]).float()
        else:
            neg_delta_fids = _EMPTY_INT64
            neg_delta_weights = _EMPTY_FLOAT32

        mask_buf.zero_()
        mask_buf[fids_long] = True
        mask_buf[ref_fids_long] = True
        ref_mask = mask_buf.clone()

        results.append(
            (
                filtered_fids,
                filtered_fmags,
                delta_values,
                neg_delta_fids,
                neg_delta_weights,
                ref_mask,
                filtered_pcts,
            )
        )
        ref[ref_fids_long] = 0.0  # scatter-clear for the next segment

    return results


@beartype
def sample_delta_negatives(
    k: NonNegativeIntLike,
    neg_delta_ratio: UnitIntervalLike,
    neg_delta_fids: t.Tensor,
    neg_delta_weights: t.Tensor,
    raw_feature_ids: t.Tensor,
    ref_mask: t.Tensor,
    selected_positive_ids: t.Tensor,
    n_sae_features: PositiveIntLike,
    rng: t.Generator,
    *,
    feature_mask: FeatureMask | None = None,
) -> t.Tensor:
    """Sample negative features with delta-aware allocation.

    Allocates negatives between two pools:
    1. Delta pool: features with negative delta, weighted by abs(delta).
    2. Uniform pool: truly inactive features (not in current or reference segments).

    Args:
        k: Total number of negative features to sample.
        neg_delta_ratio: Fraction of k to target from the delta pool (0-1).
        neg_delta_fids: [M] int64 tensor of feature IDs with negative delta.
        neg_delta_weights: [M] float32 tensor of abs(delta) for each feature.
        raw_feature_ids: [N] int64 — raw extraction features for current segment.
        ref_mask: [n_sae_features] bool tensor, True for features in
            current+reference union (excluded from uniform pool).
        selected_positive_ids: [P] int64 — positively selected features (excluded
            from delta pool).
        n_sae_features: Total SAE feature count.
        rng: Seeded torch.Generator for deterministic sampling.

    Returns:
        [k'] int64 tensor of sampled negative feature IDs (k' <= k if pools exhausted).
    """
    if k == 0:
        return t.tensor([], dtype=t.int64)

    for input_name, feature_ids in (
        ("neg_delta_fids", neg_delta_fids),
        ("raw_feature_ids", raw_feature_ids),
        ("selected_positive_ids", selected_positive_ids),
    ):
        invalid = (feature_ids < 0) | (feature_ids >= n_sae_features)
        if bool(invalid.any()):
            first_bad_id = int(feature_ids[invalid].reshape(-1)[0].item())
            raise ValueError(
                f"{input_name} contains out-of-range feature ID {first_bad_id}; "
                f"valid range is [0, {n_sae_features})."
            )

    # Remove selected positives from delta candidates (vectorized)
    if len(neg_delta_fids) > 0 and len(selected_positive_ids) > 0:
        pos_mask = t.zeros(n_sae_features, dtype=t.bool)
        pos_mask[selected_positive_ids.long()] = True
        keep = ~pos_mask[neg_delta_fids.long()]
        neg_delta_fids = neg_delta_fids[keep]
        neg_delta_weights = neg_delta_weights[keep]

    # I5 guard: all mask logic is skipped when feature_mask is None so RNG draw
    # order, shapes, and pool sizes are bit-identical to the pre-change code path.
    # When a mask is present, narrow both pools: the delta pool drops disallowed
    # candidates; the uniform-pool exclude mask gains the complement of the
    # allowed universe so disallowed ids can never be drawn.
    _universe_mask: t.Tensor | None = (
        feature_mask.allowed_universe_mask(n_sae_features) if feature_mask is not None else None
    )
    if _universe_mask is not None:
        # Narrow delta pool to allowed candidates only
        if len(neg_delta_fids) > 0:
            delta_keep = _universe_mask[neg_delta_fids.long()]
            neg_delta_fids = neg_delta_fids[delta_keep]
            neg_delta_weights = neg_delta_weights[delta_keep]

    target_from_delta = round(k * neg_delta_ratio)
    actual_from_delta = min(target_from_delta, len(neg_delta_fids))
    from_uniform = k - actual_from_delta

    collected: list[t.Tensor] = []
    selected_mask = t.zeros(n_sae_features, dtype=t.bool)

    # --- Delta pool ---
    if actual_from_delta > 0 and len(neg_delta_fids) > 0:
        if actual_from_delta >= len(neg_delta_fids):
            # Take all available
            collected.append(neg_delta_fids)
            selected_mask[neg_delta_fids.long()] = True
        else:
            # Guard against all-zero weights
            w = neg_delta_weights
            if w.sum() == 0:
                w = t.ones_like(w)
            idx = t.multinomial(
                w,
                actual_from_delta,
                replacement=False,
                generator=rng,
            )
            selected = neg_delta_fids[idx]
            collected.append(selected)
            selected_mask[selected.long()] = True

    # --- Uniform pool ---
    if from_uniform > 0:
        exclude = ref_mask.clone()
        # Exclude raw extraction features
        if len(raw_feature_ids) > 0:
            exclude[raw_feature_ids.long()] = True
        # Exclude already-selected negatives
        exclude |= selected_mask
        # Exclude selected positives
        if len(selected_positive_ids) > 0:
            exclude[selected_positive_ids.long()] = True
        # I5 guard: fold disallowed universe into exclude only when mask present
        if _universe_mask is not None:
            exclude |= ~_universe_mask

        eligible = t.nonzero(~exclude, as_tuple=False).squeeze(1)
        actual_from_uniform = min(from_uniform, len(eligible))
        if actual_from_uniform > 0:
            perm = t.randperm(len(eligible), generator=rng)[:actual_from_uniform]
            collected.append(eligible[perm].to(t.int64))

    return t.cat(collected) if collected else t.tensor([], dtype=t.int64)
