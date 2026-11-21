"""Segment operations for the completion TUI.

Handles segment enable/disable/takeover, split, retokenize, and text edits.
All functions operate on the state.py dataclasses (Segment, SessionState).
"""

from __future__ import annotations

import copy

import torch as t

from scripts.completion_tui.color_manager import next_color
from scripts.completion_tui.state import Segment, SessionState, _make_segment_id
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets
from steering.tokenizer_protocols import TextTokenizer

# ---------------------------------------------------------------------------
# compute_takeover
# ---------------------------------------------------------------------------


def compute_takeover(segments: list[Segment]) -> list[Segment]:
    """Return a NEW list with disabled segments absorbed by neighbors.

    - Disabled segment → nearest enabled LEFT neighbor absorbs (extends char_end).
    - If no left neighbor → nearest enabled RIGHT neighbor absorbs (extends char_start).
    - If no enabled segments → return empty list.
    - Original segments are NEVER mutated.

    INVARIANT: returned segments' char ranges are contiguous and cover the full
    span from min(char_start) to max(char_end) of the input.
    """
    if not segments:
        return []

    n = len(segments)
    enabled_indices = [
        i for i in range(n)
        if segments[i].enabled or segments[i].is_generation_target
    ]
    if not enabled_indices:
        return []

    # For each segment position, determine which enabled segment absorbs it.
    # Generation targets are never absorbed (they survive even when disabled).
    effective_set = set(enabled_indices)
    absorber: list[int | None] = [None] * n
    for i in range(n):
        if i in effective_set:
            absorber[i] = i
            continue

        # Nearest effective to the left
        left = None
        for j in reversed(range(i)):
            if j in effective_set:
                left = j
                break

        # Nearest effective to the right
        right = None
        for j in range(i + 1, n):
            if j in effective_set:
                right = j
                break

        # Left neighbor preferred
        absorber[i] = left if left is not None else right

    # Build result: for each enabled segment, compute its absorbed span
    result: list[Segment] = []
    for ei in enabled_indices:
        seg = copy.deepcopy(segments[ei])
        absorbed = [j for j in range(n) if absorber[j] == ei]
        seg.char_start = min(segments[j].char_start for j in absorbed)
        seg.char_end = max(segments[j].char_end for j in absorbed)
        result.append(seg)

    return result


# ---------------------------------------------------------------------------
# retokenize_boundaries
# ---------------------------------------------------------------------------


def retokenize_boundaries(
    text: str, segments: list[Segment], tokenizer: TextTokenizer
) -> None:
    """Re-snap segment char boundaries to token edges. MUTATES segments in place.

    Algorithm:
    a. Tokenize full text → tokens
    b. Compute char offsets per token via token_to_char_offsets()
    c. Build per-segment char ranges from current boundaries
    d. Assign tokens to segments via assign_tokens_to_segments() (max char overlap)
    e. Apply monotonicity (cummax) — matches training dataloader
    f. Derive new char boundaries from token assignments
    g. Update segment char_start/char_end

    Edge case: segment gets zero tokens → collapse to zero-width.
    """
    if not segments:
        return

    # a. Tokenize full text
    token_ids = tokenizer.encode(text)

    if not token_ids:
        # No tokens → all segments collapse to zero-width
        for seg in segments:
            seg.char_start = 0
            seg.char_end = 0
        return

    # b. Compute char offsets per token
    offsets = token_to_char_offsets(tokenizer, token_ids)

    # c. Token and segment char ranges as tensors
    n_tokens = len(token_ids)
    token_char_starts = t.tensor([offsets[i] for i in range(n_tokens)], dtype=t.int64)
    token_char_ends = t.tensor(
        [offsets[i + 1] for i in range(n_tokens)], dtype=t.int64
    )

    seg_char_starts = t.tensor([s.char_start for s in segments], dtype=t.int64)
    seg_char_ends = t.tensor([s.char_end for s in segments], dtype=t.int64)

    # Segment char boundaries are in source-model-with-BOS coordinates, so the
    # first segment may start at char > 0 (e.g., char 1 when source BOS = 1 char).
    # Extend the first segment backward to absorb any tokens that fall before it
    # (typically the gap created by the source model's BOS character).
    if len(seg_char_starts) > 0 and len(token_char_starts) > 0:
        if token_char_ends[0] <= seg_char_starts[0]:
            seg_char_starts = seg_char_starts.clone()
            seg_char_starts[0] = token_char_starts[0]

    # d. Assign tokens to segments via max char overlap
    assignments = assign_tokens_to_segments(
        token_char_starts, token_char_ends, seg_char_starts, seg_char_ends
    )

    # e. Apply monotonicity (cummax) — matches training dataloader
    assignments = t.cummax(assignments, dim=0).values

    # f–g. Derive new char boundaries from token assignments
    for seg_i, seg in enumerate(segments):
        mask = assignments == seg_i
        if not mask.any():
            # Zero tokens → collapse to zero-width at predecessor's end.
            # Contiguity is preserved because cummax ensures assignments are
            # non-decreasing, so token offsets form a contiguous chain:
            # predecessor's last token is adjacent to the next segment's first.
            if seg_i > 0:
                seg.char_start = segments[seg_i - 1].char_end
                seg.char_end = segments[seg_i - 1].char_end
            else:
                seg.char_start = 0
                seg.char_end = 0
        else:
            indices = mask.nonzero(as_tuple=True)[0]
            first_tok = int(indices[0].item())
            last_tok = int(indices[-1].item())
            seg.char_start = offsets[first_tok]
            seg.char_end = offsets[last_tok + 1]


# ---------------------------------------------------------------------------
# Text edit helpers
# ---------------------------------------------------------------------------


def _shift_for_delete(boundary: int, position: int, count: int) -> int:
    """Shift a char boundary after a text deletion."""
    del_end = position + count
    if boundary <= position:
        return boundary
    if boundary >= del_end:
        return boundary - count
    return position  # boundary was within the deleted range


# ---------------------------------------------------------------------------
# on_text_insert / on_text_delete
# ---------------------------------------------------------------------------


def on_text_insert(
    state: SessionState, position: int, chars: str, tokenizer: TextTokenizer
) -> None:
    """Handle text insertion.

    Updates document_text, shifts segment boundaries, calls retokenize_boundaries.
    """
    state.document_text = (
        state.document_text[:position] + chars + state.document_text[position:]
    )

    n = len(chars)
    for seg in state.segments:
        if seg.char_start >= position:
            # Segment starts at or after insertion → shift entire segment
            seg.char_start += n
            seg.char_end += n
        elif seg.char_end >= position:
            # Insertion is within or at the end of segment → extend end
            seg.char_end += n
        # Segment entirely before insertion → no change

    retokenize_boundaries(state.document_text, state.segments, tokenizer)
    state.generation_redo_stack.clear()


def on_text_delete(
    state: SessionState, position: int, count: int, tokenizer: TextTokenizer
) -> None:
    """Handle text deletion.

    Updates document_text, shifts segment boundaries, calls retokenize_boundaries.
    Handles deletion spanning segment boundaries.

    If text was deleted from the last segment and it has a preset
    ``target_token_count`` (e.g. from prefill), ``is_generation_target``
    is re-enabled so the user can regenerate to fill the segment back up.
    """
    # Check *before* shifting whether the deletion touches the last segment.
    last_seg_touched = False
    if state.segments:
        del_end = position + count
        if del_end > state.segments[-1].char_start:
            last_seg_touched = True

    state.document_text = (
        state.document_text[:position] + state.document_text[position + count :]
    )

    for seg in state.segments:
        seg.char_start = _shift_for_delete(seg.char_start, position, count)
        seg.char_end = _shift_for_delete(seg.char_end, position, count)

    retokenize_boundaries(state.document_text, state.segments, tokenizer)
    state.generation_redo_stack.clear()

    # Re-enable generation target on the last segment when text was deleted
    # from it and it has a fixed size from prefill (target_token_count).
    if last_seg_touched and state.segments:
        last = state.segments[-1]
        if last.target_token_count is not None and not last.is_generation_target:
            last.is_generation_target = True


# ---------------------------------------------------------------------------
# split_segment
# ---------------------------------------------------------------------------


def split_segment(
    state: SessionState,
    segment_idx: int,
    cursor_pos: int,
    tokenizer: TextTokenizer,
) -> bool:
    """Split a segment at the nearest token boundary to cursor_pos.

    Returns False if cursor is at segment start/end (no-op).
    Both new segments get deep-copied features.
    Second segment gets next color_idx from ColorManager.
    """
    seg = state.segments[segment_idx]

    # No-op if cursor is at or beyond segment edges
    if cursor_pos <= seg.char_start or cursor_pos >= seg.char_end:
        return False

    # Tokenize and get char offsets
    token_ids = tokenizer.encode(state.document_text)
    offsets = token_to_char_offsets(tokenizer, token_ids)

    # Find token boundaries within this segment (exclusive of start/end)
    inner_boundaries = [
        o for o in offsets if seg.char_start < o < seg.char_end
    ]

    if not inner_boundaries:
        return False

    # Find nearest boundary to cursor_pos
    nearest = min(inner_boundaries, key=lambda b: abs(b - cursor_pos))

    # Create the second segment (deep copy features)
    existing_colors = [s.color_idx for s in state.segments]
    new_color = next_color(
        existing_colors,
        left_neighbor_idx=seg.color_idx,
        right_neighbor_idx=(
            state.segments[segment_idx + 1].color_idx
            if segment_idx + 1 < len(state.segments)
            else None
        ),
    )

    new_seg = Segment(
        segment_idx=seg.segment_idx + 1,
        char_start=nearest,
        char_end=seg.char_end,
        color_idx=new_color,
        features=copy.deepcopy(seg.features),
        id=_make_segment_id(),
        enabled=seg.enabled,
    )

    # Trim original segment
    seg.char_end = nearest

    # Insert new segment
    state.segments.insert(segment_idx + 1, new_seg)

    # Re-index segment_idx for all subsequent segments
    for i in range(segment_idx + 2, len(state.segments)):
        state.segments[i].segment_idx += 1

    return True
