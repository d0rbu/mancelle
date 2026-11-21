"""Core data model for the completion TUI.

Pure Python dataclasses and functions — no Textual dependency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from uuid import uuid4

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureCondition:
    """A single SAE feature's conditioning signal."""

    feature_id: int
    magnitude: int
    enabled: bool = True
    description: str | None = None


def _make_segment_id() -> str:
    return uuid4().hex[:8]


@dataclass
class Segment:
    """A contiguous text span with feature conditioning."""

    segment_idx: int
    char_start: int
    char_end: int
    color_idx: int
    features: list[FeatureCondition]
    id: str = field(default_factory=_make_segment_id)
    enabled: bool = True
    is_generation_target: bool = False
    target_token_count: int | None = None


@dataclass(frozen=True)
class SegmentClusterEvidence:
    """Per-member evidence bound to the composite rows that produced it."""

    fingerprint: frozenset[int]
    evidence: dict[int, dict[int, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", frozenset(self.fingerprint))
        object.__setattr__(
            self,
            "evidence",
            {
                representative: dict(member_evidence)
                for representative, member_evidence in self.evidence.items()
            },
        )


@dataclass
class DocumentData:
    """Loaded document with text and per-segment features."""

    doc_idx: int
    text: str
    segments: list[Segment]
    total_segments: int  # total available (may be > len(segments))
    evidence_by_segment_index: dict[int, SegmentClusterEvidence] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationSnapshot:
    """Immutable snapshot for undo/redo.

    Defensively deep-copies mutable fields (segments, generated_token_ids)
    on construction so callers cannot corrupt history by mutating shared refs.
    """

    document_text: str
    segments: list[Segment]
    generated_text: str
    generated_token_ids: list[int]
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", copy.deepcopy(self.segments))
        object.__setattr__(self, "generated_token_ids", list(self.generated_token_ids))


@dataclass
class SessionState:
    """The entire app state."""

    document_text: str
    segments: list[Segment]
    num_buckets: int
    has_negatives: bool
    cursor_position: int = 0
    seed: int = 42
    global_conditioning: bool = True
    conditioning_memory: dict[str, bool] = field(default_factory=dict)
    generation_history: list[GenerationSnapshot] = field(default_factory=list)
    generation_redo_stack: list[GenerationSnapshot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Magnitude mapping functions
# ---------------------------------------------------------------------------


def display_magnitude_range(num_buckets: int, has_negatives: bool) -> tuple[int, int]:
    """Return (min_display, max_display) for the given bucket config."""
    if has_negatives:
        return (-1, num_buckets - 2)
    return (0, num_buckets - 1)


def display_to_internal(display_mag: int, num_buckets: int, has_negatives: bool) -> int:
    """Map display magnitude to internal bucket index.

    When has_negatives is True, display -1 maps to the last bucket (null/negative).
    Raises ValueError if display_mag is outside the valid display range.
    """
    lo, hi = display_magnitude_range(num_buckets, has_negatives)
    if display_mag < lo or display_mag > hi:
        msg = f"{display_mag} outside valid display range [{lo}, {hi}] for {num_buckets} buckets"
        raise ValueError(msg)
    if has_negatives and display_mag == -1:
        return num_buckets - 1
    return display_mag


def internal_to_display(internal_mag: int, num_buckets: int, has_negatives: bool) -> int:
    """Map internal bucket index back to display magnitude.

    Raises ValueError if internal_mag is outside [0, num_buckets - 1].
    """
    if internal_mag < 0 or internal_mag >= num_buckets:
        msg = f"{internal_mag} outside valid internal range [0, {num_buckets - 1}] for {num_buckets} buckets"
        raise ValueError(msg)
    if has_negatives and internal_mag == num_buckets - 1:
        return -1
    return internal_mag


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def deep_copy_segments(segments: list[Segment]) -> list[Segment]:
    """Deep copy a list of segments, preserving ids."""
    return copy.deepcopy(segments)
