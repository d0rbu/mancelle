"""Shared types for live feature-token highlighting in the completion TUI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.completion_tui.state import Segment


@dataclass(frozen=True)
class SegmentSpanSnapshot:
    """Immutable segment boundary snapshot for worker-thread highlight requests."""

    list_index: int
    segment_idx: int
    char_start: int
    char_end: int
    color_idx: int


@dataclass(frozen=True)
class FeatureHighlightRequest:
    """Immutable request to compute live activations for one feature/segment."""

    request_key: str
    document_text: str
    segments: tuple[SegmentSpanSnapshot, ...]
    target_segment_index: int
    feature_id: int
    # A composite feature requests all of its source members. ``feature_id``
    # remains the display/representative ID for backwards compatibility.
    feature_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class FeatureHighlightSpan:
    """One rendered span whose token activation lands in a non-gray grade."""

    char_start: int
    char_end: int
    level: int
    percentile: float
    activation: float


@dataclass(frozen=True)
class FeatureHighlightResult:
    """Live highlight result returned by the probe worker."""

    request_key: str
    segment_idx: int
    feature_id: int
    status: str
    spans: tuple[FeatureHighlightSpan, ...] = ()
    max_percentile: float | None = None
    max_activation: float | None = None
    n_tokens: int = 0
    message: str | None = None
    # Per-feature maxima are used by the cluster member panel to show live,
    # current-segment evidence without running a second probe pass.
    per_feature_max_percentile: tuple[tuple[int, float], ...] = ()
    per_feature_max_activation: tuple[tuple[int, float], ...] = ()


def _snapshot_segments(segments: list[Segment]) -> tuple[SegmentSpanSnapshot, ...]:
    return tuple(
        SegmentSpanSnapshot(
            list_index=i,
            segment_idx=seg.segment_idx,
            char_start=seg.char_start,
            char_end=seg.char_end,
            color_idx=seg.color_idx,
        )
        for i, seg in enumerate(segments)
    )


def build_highlight_request(
    document_text: str,
    segments: list[Segment],
    target_segment_index: int,
    feature_id: int | tuple[int, ...],
) -> FeatureHighlightRequest:
    """Build an immutable, fingerprinted live-highlight request.

    The fingerprint includes the document text hash, all segment boundaries,
    the target segment list index, and the feature ID.  App-level result
    application recomputes this key before painting so stale worker results
    cannot repaint edited or newly-loaded text.
    """
    segment_snapshot = _snapshot_segments(segments)
    if isinstance(feature_id, int):
        feature_ids = (feature_id,)
    else:
        feature_ids = tuple(feature_id)
        if not feature_ids:
            raise ValueError("feature_ids must not be empty")
    display_feature_id = feature_ids[0]
    payload = {
        "text_sha256": hashlib.sha256(document_text.encode("utf-8")).hexdigest(),
        "segments": [
            {
                "list_index": seg.list_index,
                "segment_idx": seg.segment_idx,
                "char_start": seg.char_start,
                "char_end": seg.char_end,
                "color_idx": seg.color_idx,
            }
            for seg in segment_snapshot
        ],
        "target_segment_index": target_segment_index,
        "feature_ids": feature_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return FeatureHighlightRequest(
        request_key=hashlib.sha256(encoded).hexdigest(),
        document_text=document_text,
        segments=segment_snapshot,
        target_segment_index=target_segment_index,
        feature_id=display_feature_id,
        feature_ids=feature_ids,
    )
