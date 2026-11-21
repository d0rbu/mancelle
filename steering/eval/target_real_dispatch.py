"""PR-C: build_target_for_real_config dispatcher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch as t
from beartype import beartype

from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.eval.context_cache_io import ContextCacheEntry, ContextFeatureSegment
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    MultiTargetSegments,
    PooledTargetRealConfig,
    SingleTargetSegment,
    TargetRealConfig,
)
from steering.eval.target_segments import (
    filter_segments_to_target,
    pool_segments_to_dict,
)
from steering.eval.turn_context_contract import TurnContextContract
from steering.eval.turn_context_routing import turn_target_start_position
from steering.refined_types import NonNegativeIntLike
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets

if TYPE_CHECKING:
    _ContextFeatureSegmentTuple = tuple[ContextFeatureSegment, ...]
    _IntSequence = Sequence[int]
    _TargetRealConfig = TargetRealConfig
else:
    # The constructors own element domains; avoid beartype's randomized
    # parameterized-container sampling at these RNG-sensitive eval boundaries.
    _ContextFeatureSegmentTuple = tuple
    _IntSequence = Sequence
    _TargetRealConfig = object


@beartype
def build_target_for_real_config(
    *,
    cfg: _TargetRealConfig,
    entry: ContextCacheEntry,
    target_char_start: NonNegativeIntLike,
    target_char_end: NonNegativeIntLike,
    full_token_ids: _IntSequence,
    target_token_loss_mask: _IntSequence,
    tokenizer,
) -> SingleTargetSegment | MultiTargetSegments:
    """Dispatch (cfg, entry) -> TargetConditioning.

    Raises:
        ValueError: filter step rejects (no overlap, partial coverage, etc.)
    """
    target_segments = filter_segments_to_target(
        entry.segments,
        target_char_start=target_char_start,
        target_char_end=target_char_end,
    )
    match cfg:
        case PooledTargetRealConfig(pooling_op=op):
            return SingleTargetSegment(
                features=pool_segments_to_dict(target_segments, op=op),
            )
        case LockstepTargetRealConfig():
            return _build_lockstep_target(
                target_segments=target_segments,
                full_token_ids=full_token_ids,
                target_token_loss_mask=target_token_loss_mask,
                tokenizer=tokenizer,
                target_char_start=target_char_start,
                target_char_end=target_char_end,
            )
        case _:  # exhaustive over union — covered by test_dispatcher_raises_typeerror_on_unknown_variant
            raise TypeError(f"unknown TargetRealConfig variant: {type(cfg).__name__}")


@beartype
def build_turn_target_for_real_config(
    *,
    cfg: _TargetRealConfig,
    entry: ContextCacheEntry,
    target_char_start: NonNegativeIntLike,
    target_char_end: NonNegativeIntLike,
    full_token_ids: _IntSequence,
    target_token_loss_mask: _IntSequence,
    tokenizer,
    turn_contract: TurnContextContract,
    context_token_ids: _IntSequence,
) -> SingleTargetSegment | MultiTargetSegments:
    """Slice SFT's full-document input routing, including marker and tool positions.

    The completion prefix is the exact cache lookup key, so it is as required
    as the contract: neither can be derived from the other here.
    """
    p0 = turn_target_start_position(context_token_ids, tokenizer)
    if tuple(full_token_ids) != entry.token_ids or list(context_token_ids) != list(full_token_ids[:p0 + 1]):
        raise ValueError("turn target-real full row and completion prefix disagree")
    if len(full_token_ids) != len(target_token_loss_mask):
        raise ValueError("full_token_ids and target_token_loss_mask length mismatch")
    if len(context_token_ids) > turn_contract.model_max_seq_len:
        raise ValueError("turn target-real completion prefix exceeds checkpoint model_max_seq_len")
    offsets = token_to_char_offsets(tokenizer, list(full_token_ids))
    ends = [index for index in range(p0 + 2, len(offsets)) if offsets[index] == target_char_end]
    if target_char_start != offsets[p0 + 1] or not ends:
        raise ValueError("turn target-real answer-body character provenance disagrees with the prefix/full row")
    end_position = ends[-1]
    routing = build_full_sequence_segment_routing(
        ids=full_token_ids, tokenizer=tokenizer,
        seg_features=(
            [t.tensor(seg.feature_ids, dtype=t.int64) for seg in entry.segments],
            [t.tensor(seg.buckets, dtype=t.int64) for seg in entry.segments],
            [seg.char_start for seg in entry.segments], [seg.char_end for seg in entry.segments],
        ), source_bos_char_len=turn_contract.semantic["source_bos_char_len"],
    )
    if bool((routing.seg_map <= 0).any()):
        raise ValueError("turn target-real full row has unrouted source slots")
    # Gemma's boundary-straddling ``><`` cannot define decoder slot ownership.
    first_target = int(routing.seg_map[p0]) - 1
    last_target = int(routing.seg_map[end_position - 1]) - 1
    # ``seg_map`` is cummax-monotone once no position is unrouted (checked
    # above), so ``seg_map[p0] >= seg_map[p0 - 1] == keep``: the marker can
    # never route before the last retained prompt slot.
    keep = int(routing.seg_map[:p0].max())
    marker_start = source_aligned_char_offsets(
        tokenizer, full_token_ids, source_bos_char_len=turn_contract.semantic["source_bos_char_len"],
    )[p0]
    # An unrouted straddle before the marker slot belongs to neither payload.
    if any(seg.char_end <= marker_start for seg in entry.segments[keep:first_target]):
        raise ValueError("unused turn source slot before target ends at or before the assistant marker")
    target_segments = entry.segments[first_target:last_target + 1]
    match cfg:
        case PooledTargetRealConfig(pooling_op=op):
            # Pool only slots some target input routes to: a source window inside
            # the span that owns no decoder token (a tool-boundary straddle)
            # conditions nothing in training, so it contributes nothing here.
            routed = sorted({int(slot) - 1 for slot in routing.seg_map[p0:end_position].tolist()})
            return SingleTargetSegment(
                features=pool_segments_to_dict(tuple(entry.segments[slot] for slot in routed), op=op)
            )
        case LockstepTargetRealConfig():
            assignments = routing.seg_map[p0:end_position] - 1 - first_target
            return MultiTargetSegments(
                segments=[dict(zip(seg.feature_ids, seg.buckets, strict=True)) for seg in target_segments],
                token_assignments=assignments,
            )
        case _:
            raise TypeError(f"unknown TargetRealConfig variant: {type(cfg).__name__}")


@beartype
def _build_lockstep_target(
    *,
    target_segments: _ContextFeatureSegmentTuple,
    full_token_ids: _IntSequence,
    target_token_loss_mask: _IntSequence,
    tokenizer,
    target_char_start: NonNegativeIntLike,
    target_char_end: NonNegativeIntLike,
) -> MultiTargetSegments:
    """Build per-token target routing without assistant-turn assumptions.

    ``target_token_loss_mask`` marks autoregressive positions whose logits are
    scored or generated against the target span.  Some callers include the
    position immediately before the target (for example the chat
    ``<assistant_start>`` token or CORE's delimiter token) because that position
    predicts the first target token.  Such leading positions have no character
    overlap with the target span, so they inherit the first target assignment.
    """
    route_positions = [idx for idx, value in enumerate(target_token_loss_mask) if value == 1]
    if not route_positions:
        raise ValueError(
            "lockstep target_real requires at least one target_token_loss_mask=1 "
            "position"
        )
    if len(full_token_ids) != len(target_token_loss_mask):
        raise ValueError(
            "full_token_ids and target_token_loss_mask length mismatch: "
            f"{len(full_token_ids)} vs {len(target_token_loss_mask)}"
        )

    char_offsets = token_to_char_offsets(tokenizer, list(full_token_ids))
    target_len = target_char_end - target_char_start
    if target_len <= 0:
        raise ValueError(
            f"target range must be positive; got [{target_char_start}, {target_char_end})"
        )

    geom_route_indices: list[int] = []
    geom_starts: list[int] = []
    geom_ends: list[int] = []
    for route_idx, pos in enumerate(route_positions):
        if pos + 1 >= len(char_offsets):
            continue
        tok_start_abs = int(char_offsets[pos])
        tok_end_abs = int(char_offsets[pos + 1])
        if tok_end_abs <= tok_start_abs:
            continue
        if tok_end_abs <= target_char_start or tok_start_abs >= target_char_end:
            continue
        geom_route_indices.append(route_idx)
        geom_starts.append(max(0, tok_start_abs - target_char_start))
        geom_ends.append(min(target_len, tok_end_abs - target_char_start))

    if not geom_route_indices:
        raise ValueError(
            "lockstep target_real found no routed tokens overlapping target "
            f"char range [{target_char_start}, {target_char_end})"
        )

    assignments = assign_tokens_to_segments(
        t.tensor(geom_starts, dtype=t.int64),
        t.tensor(geom_ends, dtype=t.int64),
        t.tensor([seg.char_start for seg in target_segments], dtype=t.int64),
        t.tensor([seg.char_end for seg in target_segments], dtype=t.int64),
    )
    assignments = t.cummax(assignments, dim=0).values.to(t.int64)

    n_used = int(assignments.max().item()) + 1
    assert n_used <= len(target_segments), (
        f"Assignment produced segment {n_used - 1} but only "
        f"{len(target_segments)} target segments are available"
    )

    # Assignments are consumed by the decoder at every absolute position after
    # the generation prefill.  Keep the absolute span from the first through
    # the last routed position, rather than compressing it to only routed
    # positions: Engine-forced tool output has mask=0 but still consumes a KV
    # position.  Compressing that gap shifts every post-tool target assignment.
    assignment_by_absolute_position = {
        route_positions[route_idx]: int(assignments[geom_idx].item())
        for geom_idx, route_idx in enumerate(geom_route_indices)
    }
    span_start = route_positions[0]
    span_end = route_positions[-1]
    full_assignments = t.empty(span_end - span_start + 1, dtype=t.int64)
    first_geom_absolute = route_positions[geom_route_indices[0]]
    last_geom_absolute = route_positions[geom_route_indices[-1]]
    # Internal positions without character geometry are zero-width/forced
    # positions. They inherit the preceding assignment. Leading and trailing
    # positions without geometry inherit the nearest valid assignment so the
    # vector remains total over the complete decoder span.
    last_seen = int(assignments[0].item())
    for absolute_position in range(span_start, span_end + 1):
        output_index = absolute_position - span_start
        if absolute_position < first_geom_absolute:
            full_assignments[output_index] = int(assignments[0].item())
        elif absolute_position > last_geom_absolute:
            full_assignments[output_index] = int(assignments[-1].item())
        else:
            if absolute_position in assignment_by_absolute_position:
                last_seen = assignment_by_absolute_position[absolute_position]
            full_assignments[output_index] = last_seen

    segment_dicts: list[dict[int, int]] = []
    for seg in target_segments[:n_used]:
        segment_dicts.append({
            int(fid): int(bucket)
            for fid, bucket in zip(seg.feature_ids, seg.buckets, strict=True)
        })
    return MultiTargetSegments(
        segments=segment_dicts,
        token_assignments=full_assignments,
    )
