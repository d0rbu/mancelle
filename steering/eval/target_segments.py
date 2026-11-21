"""Target-feature segment helpers for PR-C target_real_configs."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING

from beartype import beartype
from phantom.sized import NonEmptyStr

from steering.eval.pooling_op import PoolingOp as _PoolingOp
from steering.eval.target_conditioning import ContextFeatureSegment as _ContextFeatureSegment
from steering.refined_types import NonNegativeIntLike
from steering.segmentation import (
    # Canonical alignment helper — enforces the partial-coverage / excess-
    # coverage window in lockstep with extraction (precompute_token_boundaries)
    # and full-sequence SFT routing. See steering/segmentation.py.
    align_char_ranges_to_decoded_prefix,
)

_TARGET_FILTER_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.target_real_filter"
)

if TYPE_CHECKING:
    _ContextFeatureSegments = Iterable[_ContextFeatureSegment]
else:
    # ContextFeatureSegment owns its element domains. Keeping the runtime outer
    # hint unsubscripted avoids beartype's global-RNG-backed element sampling.
    _ContextFeatureSegments = Iterable

__all__ = [
    "filter_segments_to_target",
    "pool_segments_to_dict",
]


@beartype
def filter_segments_to_target(
    segments: _ContextFeatureSegments,
    *,
    target_char_start: NonNegativeIntLike,
    target_char_end: NonNegativeIntLike,
) -> tuple[_ContextFeatureSegment, ...]:
    """Filter segments to those overlapping [target_char_start, target_char_end);
    rebase chars to target-relative.

    Mirrors the partial/excess-coverage rules in full-sequence SFT routing.

    Raises:
        ValueError: no overlapping segments OR partial coverage > tolerance OR
                    overlapping segments after rebase.
    """
    if target_char_end <= target_char_start:
        raise ValueError(
            f"target range must be positive; got [{target_char_start}, {target_char_end})"
        )
    target_len = target_char_end - target_char_start

    overlapping = [
        seg for seg in segments
        if seg.char_end > target_char_start and seg.char_start < target_char_end
    ]
    if not overlapping:
        raise ValueError(
            f"no segments overlap target range [{target_char_start}, {target_char_end})"
        )

    rebased: list[_ContextFeatureSegment] = []
    for seg in overlapping:
        new_start = max(seg.char_start, target_char_start) - target_char_start
        new_end = min(seg.char_end, target_char_end) - target_char_start
        rebased.append(_ContextFeatureSegment(
            feature_ids=seg.feature_ids,
            buckets=seg.buckets,
            char_start=new_start,
            char_end=new_end,
        ))

    if rebased[0].char_start > 0:
        raise ValueError(
            f"target tiling: first segment starts at {rebased[0].char_start}, "
            f"expected 0 (target_char_start={target_char_start} not covered)"
        )
    for i in range(len(rebased) - 1):
        if rebased[i].char_end < rebased[i + 1].char_start:
            raise ValueError(
                f"target tiling: gap between segment {i} (end={rebased[i].char_end}) "
                f"and segment {i+1} (start={rebased[i+1].char_start})"
            )
        if rebased[i].char_end > rebased[i + 1].char_start:
            raise ValueError(
                f"target tiling: overlap between segment {i} (end={rebased[i].char_end}) "
                f"and segment {i+1} (start={rebased[i+1].char_start})"
            )
    # Reconcile any partial-coverage gap (or excess overhang — a no-op in
    # production today because rebase already clamps char_end to
    # target_char_end, but the primitive defends both directions for free).
    new_starts, new_ends, kept_indices = align_char_ranges_to_decoded_prefix(
        char_starts=[seg.char_start for seg in rebased],
        char_ends=[seg.char_end for seg in rebased],
        decoded_len=target_len,
        counter_name=_TARGET_FILTER_ALIGNMENT_COUNTER,
    )
    return tuple(
        dataclasses.replace(
            rebased[orig_idx],
            char_start=new_starts[new_idx],
            char_end=new_ends[new_idx],
        )
        for new_idx, orig_idx in enumerate(kept_indices)
    )


@beartype
def pool_segments_to_dict(
    segments: _ContextFeatureSegments,
    *,
    op: _PoolingOp,
) -> dict[int, int]:
    """Collapse N target-relative segments into one (feature_id -> bucket) dict.

    BUCKET_MAX: per-feature max bucket across segments.
    BUCKET_MEAN: per-feature mean bucket rounded using round-half-up semantics
        (i.e. 0.5 -> 1, 1.5 -> 2, 2.5 -> 3). Python's built-in ``round()``
        uses banker's rounding (round-half-to-even), which silently rounds 0.5
        to 0 and 2.5 to 2. We use ``int((s + n // 2) // n)`` instead, which
        gives the intuitive "round up on half" result for non-negative inputs.
        ContextFeatureSegment's phantom-backed annotations ensure all bucket
        values are non-negative before this arithmetic runs.
    """
    if op is _PoolingOp.BUCKET_MAX:
        out: dict[int, int] = {}
        for seg in segments:
            for fid, bucket in zip(seg.feature_ids, seg.buckets, strict=True):
                if fid not in out or bucket > out[fid]:
                    out[fid] = int(bucket)
        return out
    if op is _PoolingOp.BUCKET_MEAN:
        sums: dict[int, list[int]] = {}
        for seg in segments:
            for fid, bucket in zip(seg.feature_ids, seg.buckets, strict=True):
                sums.setdefault(int(fid), []).append(int(bucket))
        return {
            fid: int((sum(buckets) + len(buckets) // 2) // len(buckets))
            for fid, buckets in sums.items()
        }
    raise ValueError(f"unknown PoolingOp: {op!r}")
