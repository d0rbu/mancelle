"""Multi-segment seg_map construction for teacher-forced scoring.

Delegates to existing primitives in
steering.conditional.encoder_decoder.eval_adapter and
steering.segmentation. No char-level overlap logic is reimplemented here.

Callers that need to route prompt-prefix tokens (sitting to the LEFT of
the inputs prefix in a combined prefix) to a no-encoder-context slot
MUST use the split+shift pattern: call ``build_multi_segment_seg_map``
on the inputs-prefix subrange of ``full_prefix``, shift every value up
by 1, prepend ``[0] * n_prompt_surviving``, and prepend an empty slot
to the encoder feature inputs. ``build_multi_segment_seg_map`` does not
handle the combined-prefix coord-space mismatch internally. See
``steering.eval.axbench_bpb.teacher_force._build_encdec_seg_map_and_encoder``
for the canonical caller.
"""
from __future__ import annotations

from typing import Any

import torch
from phantom.sized import NonEmptyStr

from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import (
    align_segments_to_decoded_prefix,
    compute_context_token_segment_assignments,
    token_to_char_offsets,
)

_BPB_SEG_MAP_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.bpb_seg_map"
)


def project_segments_to_truncated_prefix(
    segments: tuple[ContextFeatureSegment, ...],
    *,
    original_prefix_tokens: list[int],
    truncated_prefix_tokens: list[int],
    tokenizer: Any,
) -> tuple[ContextFeatureSegment, ...]:
    """Project ContextFeatureSegment char offsets to a left-truncated prefix.

    When ``truncate_prefix_left`` drops the leftmost ``k`` tokens of
    ``original_prefix_tokens``, every retained context segment's
    ``char_start`` / ``char_end`` (defined against the original prefix
    text) needs to be shifted by ``-dropped_chars`` to match the
    truncated prefix's coordinate space, then clamped to
    ``[0, len_truncated_chars]``. Segments whose projected end is
    ``<= 0`` lie entirely in the dropped portion and are removed.

    This is the **Q11** fix for slice 17 (audit 2026-05-05). Without
    projection, ``compute_context_token_segment_assignments`` would
    compare truncated token char-offsets (which start at 0) against
    untruncated segment char-offsets (which start somewhere > 0), so
    the cummax monotonicity step would either misroute every prefix
    token to slot 0 OR produce an out-of-range slot index in the
    cross-attention scatter.

    Args:
        segments: The original context-cache segments built against
            ``original_prefix_tokens``.
        original_prefix_tokens: The pre-truncation prefix token list
            (with BOS, with prompt-prefix fold-in if any).
        truncated_prefix_tokens: The post-truncation prefix token list
            (suffix of ``original_prefix_tokens``).
        tokenizer: The nanochat-side tokenizer used for both decode
            paths. Required by ``token_to_char_offsets``.

    Returns:
        A new tuple of segments whose ``char_start`` / ``char_end``
        live in the truncated coordinate space. Segments fully dropped
        by the truncation are omitted; ``feature_ids`` and ``buckets``
        are preserved verbatim.

    Raises:
        ValueError: ``truncated_prefix_tokens`` is not a suffix of
            ``original_prefix_tokens`` (i.e. an unexpected drop pattern).
    """
    n_original = len(original_prefix_tokens)
    n_truncated = len(truncated_prefix_tokens)
    if n_truncated > n_original:
        raise ValueError(
            f"truncated_prefix_tokens length {n_truncated} exceeds "
            f"original_prefix_tokens length {n_original}; cannot project "
            f"segment char offsets onto a longer prefix"
        )
    n_dropped = n_original - n_truncated
    if n_dropped == 0:
        # Fast path: no truncation, segments are unchanged.
        return segments
    # Sanity check: truncated must be a suffix of original.
    if (
        n_truncated > 0
        and original_prefix_tokens[n_dropped:] != truncated_prefix_tokens
    ):
        raise ValueError(
            "truncated_prefix_tokens is not a left-suffix of "
            "original_prefix_tokens; cannot project segments. "
            "truncate_prefix_left must drop only the leftmost tokens."
        )

    if not segments:
        return segments

    # Compute char offsets of the original prefix to determine how many
    # characters were dropped on the left.
    original_offsets = token_to_char_offsets(tokenizer, original_prefix_tokens)
    # original_offsets has length n_original + 1; index i is the char position
    # where token i starts. The number of dropped characters equals the char
    # position where the FIRST RETAINED token starts.
    dropped_chars = original_offsets[n_dropped]
    # Char length of the truncated prefix is (total - dropped).
    len_truncated_chars = original_offsets[-1] - dropped_chars

    projected: list[ContextFeatureSegment] = []
    for seg in segments:
        new_start = seg.char_start - dropped_chars
        new_end = seg.char_end - dropped_chars
        if new_end <= 0:
            # Segment lies entirely in the dropped portion.
            continue
        # Clamp to [0, len_truncated_chars].
        new_start = max(0, new_start)
        new_end = min(len_truncated_chars, new_end)
        if new_end <= new_start:
            # Truncation collapsed the segment to zero width; drop it.
            continue
        projected.append(ContextFeatureSegment(
            feature_ids=seg.feature_ids,
            buckets=seg.buckets,
            char_start=new_start,
            char_end=new_end,
        ))
    return tuple(projected)


def build_multi_segment_seg_map(
    *,
    tokenizer: Any,
    prefix_tokens: list[int],
    prefix_segments: tuple[ContextFeatureSegment, ...],
    n_output: int,
    generation_slot: int,
) -> torch.Tensor:
    """Return seg_map of shape (1, n_prefix + n_output).

    Prefix positions are assigned via char-level overlap + cummax monotonicity
    to their ContextFeatureSegment slot; output positions are all set to
    ``generation_slot``.

    This mirrors
    :meth:`TargetAwareEncoderDecoderEvalWrapper._fill_context_assignments`
    in ``steering.conditional.encoder_decoder.eval_adapter``.

    NOTE: ``prefix_segments`` must use char offsets that live in the same
    coordinate space as ``prefix_tokens``. If the caller has left-truncated
    ``prefix_tokens``, it MUST first project the segments via
    :func:`project_segments_to_truncated_prefix` — otherwise the cummax
    monotonicity step will mis-route truncated tokens to slot 0 or produce
    an out-of-range slot for the cross-attention scatter.

    When the nanochat tokenizer's decode of ``prefix_tokens`` disagrees
    with the last segment's ``char_end`` (e.g., due to multi-byte
    handling differences between the Gemma extraction tokenizer and the
    nanochat tokenizer), the segments are reconciled via
    :func:`steering.segmentation.align_segments_to_decoded_prefix`. The
    helper extends the last segment's ``char_end`` (forward direction)
    or filters fully-uncovered segments + clips the last surviving
    segment's ``char_end`` (reverse direction), provided the delta stays
    within ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` (20 chars). A
    delta beyond that bound raises ``ValueError``; each adjustment emits
    the ``segmentation.char_end_extended`` profiler counter with a
    ``direction`` attribute.
    """
    n_prefix = len(prefix_tokens)
    T = n_prefix + n_output
    seg_map = torch.zeros(1, T, dtype=torch.long)

    if n_prefix > 0 and prefix_segments:
        prefix_segments = align_segments_to_decoded_prefix(
            tokenizer=tokenizer,
            prefix_tokens=prefix_tokens,
            segments=prefix_segments,
            counter_name=_BPB_SEG_MAP_ALIGNMENT_COUNTER,
        )

        seg_map[0, :n_prefix] = compute_context_token_segment_assignments(
            tokenizer=tokenizer,
            context_tokens=prefix_tokens,
            context_segments=prefix_segments,
        )

    seg_map[0, n_prefix:] = generation_slot
    return seg_map
