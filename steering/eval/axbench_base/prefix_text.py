"""Shared decoded-text prefix windowing for AxBench stages."""

from __future__ import annotations

from beartype import beartype

from steering.eval.axbench_base.types import AxBenchBaseError
from steering.refined_types import NonNegativeIntLike, PositiveIntLike


@beartype
def validate_prefix_char_bounds(
    min_prefix_chars: NonNegativeIntLike,
    max_prefix_chars: PositiveIntLike,
) -> None:
    """Validate the relationship between refined decoded-text prefix bounds."""
    if min_prefix_chars > max_prefix_chars:
        raise AxBenchBaseError(
            "min_prefix_chars must be <= max_prefix_chars, got "
            f"{min_prefix_chars} > {max_prefix_chars}"
        )


def window_prefix_text(
    full_text: str,
    end_char: int,
    max_prefix_chars: int,
) -> str:
    """Return the suffix ending at ``end_char`` within ``max_prefix_chars``.

    The operation intentionally uses Python string indexing: the configured
    bound counts decoded Unicode code points, not UTF-8 bytes, tokenizer
    tokens, grapheme clusters, words, or sentences. Callers own eligibility
    checks such as ``min_prefix_chars``.
    """
    start = max(0, end_char - max_prefix_chars)
    return full_text[start:end_char]
