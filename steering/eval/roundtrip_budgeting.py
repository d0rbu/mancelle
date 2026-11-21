"""Shared generation-budget helpers for round-trip steering evals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from steering.refined_types import PositiveIntLike
from steering.segmentation import token_to_char_offsets


class RoundtripGenerationBudgetError(ValueError):
    """Raised when a round-trip generation budget cannot be derived."""


@dataclass(frozen=True)
class ReferenceSegmentGenerationBudget:
    """Target-token budget for regenerating the first reference segment.

    ``prefix_tokens + reference_output_tokens`` is decoded as one target-token
    sequence, then the rendered ``prefix_text`` plus contextual reference suffix
    is sliced from that full decoded text and segmented in extraction-token
    space.  Target-token offsets also come from the full target-token sequence,
    so boundary-sensitive target tokenizers are never decoded on the
    continuation in isolation.  The generated completion should fill only the
    first extraction segment touched by the reference continuation.
    """

    output_tokens: tuple[int, ...]
    reference_char_count: int
    prefix_extraction_token_count: int
    segment_end_extraction_token_count: int

    @property
    def max_new_tokens(self) -> int:
        return len(self.output_tokens)


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_nested_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _flatten_token_ids(raw_ids: Any) -> list[int]:
    raw_ids = _to_python(raw_ids)
    if _is_nested_sequence(raw_ids) and raw_ids and _is_nested_sequence(raw_ids[0]):
        raw_ids = raw_ids[0]
    if not _is_nested_sequence(raw_ids):
        raise TypeError("tokenizer input_ids must be a sequence")
    return [int(token_id) for token_id in raw_ids]


def _tokenizer_call_value(encoded: Any, key: str) -> Any:
    if isinstance(encoded, Mapping):
        return encoded[key]
    return getattr(encoded, key)


def _flatten_offset_pairs(raw_offsets: Any) -> list[tuple[int, int]]:
    raw_offsets = _to_python(raw_offsets)
    if (
        _is_nested_sequence(raw_offsets)
        and raw_offsets
        and _is_nested_sequence(raw_offsets[0])
        and raw_offsets[0]
        and _is_nested_sequence(raw_offsets[0][0])
    ):
        raw_offsets = raw_offsets[0]
    if not _is_nested_sequence(raw_offsets):
        raise TypeError("tokenizer offset_mapping must be a sequence")
    return [(int(start), int(end)) for start, end in raw_offsets]


def extraction_token_ids_and_char_ends(
    text: str,
    extraction_tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Tokenize text and return extraction-token ids plus exclusive char ends."""

    try:
        encoded = extraction_tokenizer(
            text,
            padding=False,
            return_offsets_mapping=True,
        )
        token_ids = _flatten_token_ids(_tokenizer_call_value(encoded, "input_ids"))
        offset_pairs = _flatten_offset_pairs(
            _tokenizer_call_value(encoded, "offset_mapping")
        )
        if len(offset_pairs) == len(token_ids):
            return token_ids, [end for _start, end in offset_pairs]
    except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError):
        pass

    try:
        encoded = extraction_tokenizer(text, padding=False)
        token_ids = _flatten_token_ids(_tokenizer_call_value(encoded, "input_ids"))
    except (AttributeError, KeyError, TypeError):
        token_ids = _flatten_token_ids(extraction_tokenizer.encode(text))
    offsets = token_to_char_offsets(extraction_tokenizer, token_ids)
    return token_ids, [min(int(end), len(text)) for end in offsets[1:]]


@beartype
def reference_segment_generation_budget(
    *,
    prefix_text: str,
    prefix_tokens: Sequence[int],
    reference_output_tokens: Sequence[int],
    target_tokenizer: Any,
    extraction_tokenizer: Any,
    reference_segment_size: PositiveIntLike,
) -> ReferenceSegmentGenerationBudget:
    """Return the target-token budget for the first reference segment.

    The computation happens before generation.  ``reference_segment_size`` is
    measured in extraction-token space.  The returned ``max_new_tokens`` is the
    number of target-model output tokens needed to cover that same reference
    span, so the two counts are equal only when both tokenizers align that way.
    ``prefix_tokens`` must be the target-token context that the reference output
    was paired with; the reference continuation is not decoded in isolation.

    Raises:
        BeartypeCallHintParamViolation: An argument violates its annotated
            contract, including a non-positive ``reference_segment_size``.
        RoundtripGenerationBudgetError: A sequence or tokenizer relationship
            prevents a non-empty generation budget from being derived.
    """

    if not prefix_tokens:
        raise RoundtripGenerationBudgetError(
            "prefix_tokens must contain at least one token"
        )
    if not reference_output_tokens:
        raise RoundtripGenerationBudgetError(
            "reference_output_tokens must contain at least one token"
        )

    prefix_target_tokens = tuple(int(token_id) for token_id in prefix_tokens)
    output_tokens = tuple(int(token_id) for token_id in reference_output_tokens)
    full_target_tokens = (*prefix_target_tokens, *output_tokens)
    full_target_text = str(target_tokenizer.decode(list(full_target_tokens)))
    target_char_offsets = token_to_char_offsets(
        target_tokenizer,
        list(full_target_tokens),
    )
    prefix_target_char_count = int(target_char_offsets[len(prefix_target_tokens)])
    prefix_target_text = full_target_text[:prefix_target_char_count]
    if not prefix_target_text.endswith(prefix_text):
        raise RoundtripGenerationBudgetError(
            "prefix_text does not match decoded prefix_tokens in the full target "
            "context; cannot align extraction-token segment boundaries with "
            "target-token context"
        )
    prefix_text_start = prefix_target_char_count - len(prefix_text)
    # Slice one suffix from the full target decode. This drops decoded BOS or
    # control text before ``prefix_text`` while preserving target-token boundary
    # merges in the reference suffix. Tokenize this full suffix once below so
    # extraction-token boundary merges are also accounted for.
    full_text = full_target_text[prefix_text_start:]
    available_reference_char_count = len(full_target_text) - prefix_target_char_count
    full_extraction_ids, full_char_ends = extraction_token_ids_and_char_ends(
        full_text,
        extraction_tokenizer,
    )
    prefix_char_count = len(prefix_text)
    prefix_extraction_len = sum(
        1 for char_end in full_char_ends if int(char_end) <= prefix_char_count
    )
    if prefix_extraction_len >= len(full_extraction_ids):
        raise RoundtripGenerationBudgetError(
            "reference continuation adds no extraction tokens"
        )

    segment_end_token = min(
        len(full_extraction_ids),
        ((prefix_extraction_len // reference_segment_size) + 1)
        * reference_segment_size,
    )
    segment_end_char = int(full_char_ends[segment_end_token - 1])
    reference_char_count = max(
        0,
        min(segment_end_char - len(prefix_text), available_reference_char_count),
    )

    generation_token_count = 0
    n_prefix_tokens = len(prefix_target_tokens)
    output_token_starts = target_char_offsets[
        n_prefix_tokens : n_prefix_tokens + len(output_tokens)
    ]
    for token_index, token_start in enumerate(output_token_starts):
        output_relative_start = int(token_start) - prefix_target_char_count
        if output_relative_start < reference_char_count:
            generation_token_count = token_index + 1
        else:
            break

    budget_tokens = output_tokens[:generation_token_count]
    if not budget_tokens:
        raise RoundtripGenerationBudgetError(
            "no reference output tokens overlap the first generated segment"
        )
    return ReferenceSegmentGenerationBudget(
        output_tokens=budget_tokens,
        reference_char_count=reference_char_count,
        prefix_extraction_token_count=prefix_extraction_len,
        segment_end_extraction_token_count=segment_end_token,
    )
