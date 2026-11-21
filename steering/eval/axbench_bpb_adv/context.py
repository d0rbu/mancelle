"""Context projection helpers for prompt-token attacks."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import (
    compute_context_token_segment_assignments,
    token_to_char_offsets,
)


def _token_positions_by_source_segment(
    *,
    tokenizer: Any,
    source_tokens: list[int],
    source_segments: tuple[ContextFeatureSegment, ...],
) -> list[list[int]]:
    assignments = compute_context_token_segment_assignments(
        tokenizer=tokenizer,
        context_tokens=source_tokens,
        context_segments=source_segments,
    ).tolist()
    positions_by_segment: list[list[int]] = [[] for _ in range(len(source_segments))]
    for position, segment_idx in enumerate(assignments):
        idx = int(segment_idx)
        if idx < 0 or idx >= len(source_segments):
            raise AxBenchBpbAdvError(
                f"source context assignment for token position {position} "
                f"returned out-of-range segment index {idx}"
            )
        positions_by_segment[idx].append(position)
    return positions_by_segment


def project_fixed_context_segments(
    *,
    tokenizer: Any,
    source_tokens: Sequence[int],
    attacked_tokens: Sequence[int],
    source_segments: tuple[ContextFeatureSegment, ...],
) -> tuple[ContextFeatureSegment, ...]:
    """Project source SAE segments onto a length-preserving attacked prompt."""

    source = [int(token_id) for token_id in source_tokens]
    attacked = [int(token_id) for token_id in attacked_tokens]
    if len(source) != len(attacked):
        raise AxBenchBpbAdvError(
            "fixed_context projection requires length-preserving token edits, "
            f"got source_len={len(source)} attacked_len={len(attacked)}"
        )
    if not source_segments:
        raise AxBenchBpbAdvError("fixed_context requires non-empty source segments")

    positions_by_segment = _token_positions_by_source_segment(
        tokenizer=tokenizer,
        source_tokens=source,
        source_segments=source_segments,
    )
    attacked_offsets = token_to_char_offsets(tokenizer, attacked)
    decoded_len = attacked_offsets[-1] if attacked_offsets else 0
    projected: list[ContextFeatureSegment] = []
    for segment, positions in zip(source_segments, positions_by_segment, strict=True):
        if not positions:
            raise AxBenchBpbAdvError(
                "source context segment received no token positions during projection"
            )
        start = attacked_offsets[min(positions)]
        end = attacked_offsets[max(positions) + 1]
        if end <= start:
            if decoded_len <= start:
                raise AxBenchBpbAdvError(
                    "fixed_context projection produced a zero-width segment "
                    f"at end of prompt: start={start}, end={end}"
                )
            end = start + 1
        projected.append(replace(segment, char_start=int(start), char_end=int(end)))
    for left, right in itertools.pairwise(projected):
        if left.char_end > right.char_start:
            raise AxBenchBpbAdvError(
                "fixed_context projection produced overlapping segments: "
                f"left_end={left.char_end}, right_start={right.char_start}"
            )
    return tuple(projected)
