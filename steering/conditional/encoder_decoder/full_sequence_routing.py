"""Full-sequence segment routing for encoder-decoder conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import torch as t
from beartype import beartype
from phantom.sized import NonEmptyStr

from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    validate_context_token_assignments,
)
from steering.segmentation import (
    SegmentMode,
    align_char_ranges_to_decoded_prefix,
    assign_tokens_to_segments,
    compute_context_token_segment_assignments,
    token_to_char_offsets,
)

EMPTY_FEATURES = t.tensor([], dtype=t.int64)

_DEFAULT_FULL_SEQUENCE_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.full_sft"
)


@dataclass(frozen=True)
class FullSequenceSegmentRouting:
    """Segment routing over a full rendered decoder token sequence."""

    per_seg_fids: list[t.Tensor]
    per_seg_buckets: list[t.Tensor]
    seg_map: t.Tensor


def source_aligned_char_offsets(
    tokenizer,
    ids: Sequence[int],
    *,
    source_bos_char_len: int = 0,
) -> list[int]:
    """Return decoder token char offsets in the feature-source coordinates.

    Full-sequence SFT extraction text omits the decoder BOS because the
    extraction tokenizer contributes its own BOS.  Shift every boundary after
    the first decoder token by ``source_bos_char_len - nanochat_bos_len`` so
    nanochat token spans line up with the source-model feature shard.
    """
    if source_bos_char_len < 0:
        raise ValueError(
            f"feature_source_bos_char_len must be non-negative, got {source_bos_char_len}"
        )
    offsets = token_to_char_offsets(tokenizer, list(ids))
    if len(offsets) <= 1:
        return offsets
    nanochat_bos_len = offsets[1]
    delta = source_bos_char_len - nanochat_bos_len
    if delta == 0:
        return offsets
    return [offsets[0]] + [offset + delta for offset in offsets[1:]]


def compute_full_coverage_gap(
    ids: Sequence[int],
    seg_features_list,
    tokenizer,
    *,
    source_bos_char_len: int = 0,
) -> int:
    """Return source-coordinate character gap for full-conversation shards."""
    if not seg_features_list:
        raise ValueError("Cannot compute full-sequence coverage for an empty feature row")
    total_chars = source_aligned_char_offsets(
        tokenizer,
        ids,
        source_bos_char_len=source_bos_char_len,
    )[-1]
    coverage_end = max(int(s.char_end) for s in seg_features_list)
    return total_chars - coverage_end


def build_full_sequence_segment_routing(
    *,
    ids: Sequence[int],
    tokenizer,
    seg_features: tuple[list[t.Tensor], list[t.Tensor], list[int], list[int]],
    source_bos_char_len: int = 0,
    counter_name: NonEmptyStr = _DEFAULT_FULL_SEQUENCE_ALIGNMENT_COUNTER,
) -> FullSequenceSegmentRouting:
    """Build segment routing for features extracted over the full sample.

    Segment character ranges are interpreted over the full feature-source text,
    and every token whose character interval overlaps an in-range segment
    receives a non-zero ``seg_map`` entry.
    """
    token_ids = list(ids)
    T = len(token_ids)
    seg_map = t.zeros(T, dtype=t.int64)
    per_seg_fids: list[t.Tensor] = [EMPTY_FEATURES]
    per_seg_buckets: list[t.Tensor] = [EMPTY_FEATURES]

    if seg_features is None:
        raise ValueError("seg_features is required for full-sequence routing")

    fids_list, buckets_list, char_starts, char_ends = seg_features
    n_ext_segs = len(fids_list)
    assert n_ext_segs == len(buckets_list) == len(char_starts) == len(char_ends), (
        "Segment count mismatch"
    )
    if n_ext_segs == 0:
        raise ValueError("No segments provided (need at least one)")
    for i in range(n_ext_segs):
        if char_ends[i] <= char_starts[i]:
            raise ValueError(
                f"Segment {i} has non-positive length: start={char_starts[i]}, end={char_ends[i]}"
            )
    if char_starts[0] != 0:
        raise ValueError(f"First segment must start at 0, got {char_starts[0]}")
    for i in range(n_ext_segs - 1):
        if char_ends[i] < char_starts[i + 1]:
            raise ValueError(
                f"Gap between segment {i} (end={char_ends[i]}) and "
                f"segment {i + 1} (start={char_starts[i + 1]})"
            )
        if char_ends[i] > char_starts[i + 1]:
            raise ValueError(
                f"Overlap between segment {i} (end={char_ends[i]}) and "
                f"segment {i + 1} (start={char_starts[i + 1]})"
            )

    char_offsets = source_aligned_char_offsets(
        tokenizer,
        token_ids,
        source_bos_char_len=source_bos_char_len,
    )
    decoded_len = char_offsets[-1]

    new_starts, new_ends, kept_indices = align_char_ranges_to_decoded_prefix(
        char_starts=char_starts,
        char_ends=char_ends,
        decoded_len=decoded_len,
        counter_name=counter_name,
    )
    char_starts = new_starts
    char_ends = new_ends
    fids_list = [fids_list[i] for i in kept_indices]
    buckets_list = [buckets_list[i] for i in kept_indices]
    n_ext_segs = len(kept_indices)

    tok_starts_full = t.tensor(char_offsets[:-1], dtype=t.int64)
    tok_ends_full = t.tensor(char_offsets[1:], dtype=t.int64)
    zero_width = tok_starts_full == tok_ends_full
    if zero_width.any():
        expanded_ends = tok_ends_full.clone()
        for i in range(len(expanded_ends) - 2, -1, -1):
            if expanded_ends[i] == tok_starts_full[i]:
                expanded_ends[i] = expanded_ends[i + 1]
        tok_ends_full = expanded_ends

    seg_range_end = char_ends[-1]
    in_range = tok_starts_full < seg_range_end
    n_in_range = int(in_range.sum().item())
    if n_in_range == 0:
        raise ValueError(
            "Features provided but no conversation tokens overlap the feature character ranges."
        )

    tok_starts = tok_starts_full[:n_in_range]
    tok_ends = tok_ends_full[:n_in_range]
    seg_char_starts = t.tensor(char_starts, dtype=t.int64)
    seg_char_ends = t.tensor(char_ends, dtype=t.int64)

    assignments = assign_tokens_to_segments(
        tok_starts,
        tok_ends,
        seg_char_starts,
        seg_char_ends,
    )
    assignments = t.cummax(assignments, dim=0).values

    n_segs_used = int(assignments.max().item()) + 1 if len(assignments) > 0 else 0
    assert n_segs_used <= n_ext_segs, (
        f"Assignment produced segment {n_segs_used - 1} but only "
        f"{n_ext_segs} post-alignment segments"
    )

    for pos in range(n_in_range):
        seg_map[pos] = 1 + int(assignments[pos].item())
    for s in range(n_segs_used):
        per_seg_fids.append(fids_list[s])
        per_seg_buckets.append(buckets_list[s])

    return FullSequenceSegmentRouting(per_seg_fids, per_seg_buckets, seg_map)


@beartype
def turn_context_token_assignments(
    *,
    tokenizer,
    context_tokens: Sequence[int],
    context_segments: Sequence[ContextFeatureSegment],
) -> t.Tensor:
    """Assign turn-mode context tokens to their windows exactly as training does.

    Character ranges cannot say which window a zero-width UTF-8 byte piece on
    a boundary belongs to; routing resolves that as training did (the piece
    joins the following window), whereas the fixed-cache assignment keeps it
    with the preceding one.  The entry's ranges are the decoder's own offsets:
    ``source_aligned_char_offsets`` shifts by ``source_bos_char_len`` minus its
    own ``offsets[1]``, so passing ``offsets[1]`` back cancels the shift for any
    leading token.
    """

    ids = list(context_tokens)
    offsets = token_to_char_offsets(tokenizer, ids)
    routing = build_full_sequence_segment_routing(
        ids=ids,
        tokenizer=tokenizer,
        seg_features=(
            [EMPTY_FEATURES] * len(context_segments),
            [EMPTY_FEATURES] * len(context_segments),
            [segment.char_start for segment in context_segments],
            [segment.char_end for segment in context_segments],
        ),
        source_bos_char_len=offsets[1],
    )
    # Only trailing zero-width byte pieces can be left unrouted, and a turn
    # context always ends on a structural special (the prefill's marker check;
    # the fit row's closing marker), so this fires only for a malformed caller.
    unrouted = (routing.seg_map == 0).nonzero().flatten().tolist()
    if unrouted:
        raise ValueError(
            f"turn-mode context windows leave token {unrouted[0]} unrouted "
            f"({len(unrouted)} of {len(ids)} tokens own no window)"
        )
    return routing.seg_map - 1


@beartype
def context_token_assignments(
    *,
    segment_mode: SegmentMode,
    tokenizer,
    context_tokens: Sequence[int],
    context_segments: Sequence[ContextFeatureSegment],
    token_assignments: tuple[int, ...] | None = None,
) -> t.Tensor:
    """Dispatch context ownership using the producing segment geometry."""
    match segment_mode:
        case SegmentMode.fixed_tokens:
            return compute_context_token_segment_assignments(
                tokenizer=tokenizer,
                context_tokens=list(context_tokens),
                context_segments=tuple(context_segments),
            )
        case SegmentMode.turn_delimited_fixed_tokens:
            if token_assignments is not None:
                validate_context_token_assignments(context_tokens, context_segments, token_assignments)
                return t.tensor(token_assignments, dtype=t.int64)
            return turn_context_token_assignments(
                tokenizer=tokenizer,
                context_tokens=context_tokens,
                context_segments=context_segments,
            )
        case _:
            assert_never(segment_mode)
