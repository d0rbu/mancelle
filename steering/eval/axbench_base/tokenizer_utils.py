"""Nanochat BOS-ownership and exact labeled-pair tokenization helpers.

AxBench v2 stores BOS-free source content and adds one pipeline-owned BOS only
at a model-input boundary.  :func:`prepend_owned_bos` deliberately does not
inspect or deduplicate payload IDs: source text that naturally tokenizes to the
BOS ID remains source content.  :func:`tokenize_labeled_pair` encodes combined
training text once and locates the first token that overlaps the output text,
including BPE tokens that cross the character boundary.

The stricter :func:`assert_bos_prefix` validator remains available for
callers whose payload is known not to contain BOS.
"""

from __future__ import annotations

from typing import Any

from steering.eval.axbench_base.types import AxBenchBaseError
from steering.segmentation import token_to_char_offsets


def _bos_token_id(tokenizer: Any) -> int:
    bos_id = tokenizer.get_bos_token_id()
    if bos_id is None:
        raise AxBenchBaseError(
            "Tokenizer has no BOS token id (get_bos_token_id() returned "
            "None); cannot construct an owned-BOS model input."
        )
    return int(bos_id)


def prepend_owned_bos(tokenizer: Any, content_tokens: list[int]) -> list[int]:
    """Prepend the pipeline-owned BOS without inspecting the payload.

    A content token equal to ``bos_id`` remains content. Ownership is defined
    solely by construction and position, never by token-value deduplication.
    """

    return [_bos_token_id(tokenizer), *(int(token) for token in content_tokens)]


def tokenize_labeled_pair(
    tokenizer: Any,
    prefix_text: str,
    output_text: str,
) -> tuple[list[int], int]:
    """Tokenize one labeled pair and return its exact output token boundary.

    The concatenated text is encoded once so a BPE token may legitimately
    cross the prefix/output character boundary. The returned boundary is the
    first model-token position whose decoded extent overlaps output text and
    includes the owned BOS at position zero.
    """
    if not isinstance(prefix_text, str):
        raise AxBenchBaseError(
            "prefix_text must be a string for labeled-pair tokenization, "
            f"got {type(prefix_text).__name__}."
        )
    if not isinstance(output_text, str):
        raise AxBenchBaseError(
            "output_text must be a string for labeled-pair tokenization, "
            f"got {type(output_text).__name__}."
        )
    if not output_text:
        raise AxBenchBaseError(
            "output_text must be non-empty for labeled-pair tokenization"
        )

    combined_text = prefix_text + output_text
    content_tokens = [int(token) for token in tokenizer.encode(combined_text)]
    decoded = tokenizer.decode(content_tokens)
    if decoded != combined_text:
        raise AxBenchBaseError(
            "Cannot determine an exact labeled-pair boundary because the "
            "tokenizer does not round-trip prefix_text + output_text: "
            f"expected {combined_text!r}, decoded {decoded!r}."
        )
    if not decoded.startswith(prefix_text):
        raise AxBenchBaseError(
            "Decoded labeled-pair tokens do not have prefix_text as their "
            "actual character prefix."
        )
    offsets = token_to_char_offsets(tokenizer, content_tokens)
    output_char_start = len(prefix_text)
    content_output_index = next(
        (
            index
            for index, token_end in enumerate(offsets[1:])
            if int(token_end) > output_char_start
        ),
        None,
    )
    if content_output_index is None:
        raise AxBenchBaseError(
            "Cannot determine an output token boundary: no encoded token "
            "overlaps the non-empty output_text."
        )
    model_tokens = prepend_owned_bos(tokenizer, content_tokens)
    return model_tokens, 1 + content_output_index


def assert_bos_prefix(tokens: list[int], bos_id: int) -> None:
    """Assert that ``tokens`` starts with exactly one BOS token.

    Three failure modes are caught:

    - Empty list (nothing to prefix).
    - First token is not the BOS id (missing prefix).
    - First TWO tokens are both BOS (double-BOS; usually a bug where
      a caller prepended on top of an already-BOS-prefixed encoding).

    Args:
        tokens: The token id list to validate.
        bos_id: The expected BOS token id (from
            ``tokenizer.get_bos_token_id()``).

    Raises:
        AxBenchBaseError: On any of the three failure modes above.
            The error message includes ``tokens[:5]`` so the caller
            can diagnose the problem from the log alone.
    """
    if len(tokens) == 0:
        raise AxBenchBaseError(
            "Token list is empty; no BOS prefix. Expected at least "
            f"[{bos_id}, ...]."
        )
    if tokens[0] != bos_id:
        raise AxBenchBaseError(
            f"Token list does not start with BOS ({bos_id}): got "
            f"tokens[0]={tokens[0]}, tokens[:5]={tokens[:5]}."
        )
    if len(tokens) >= 2 and tokens[1] == bos_id:
        raise AxBenchBaseError(
            f"Token list has double BOS at positions 0 and 1: "
            f"tokens[:5]={tokens[:5]}. This usually means a caller "
            "prepended BOS on top of an already-BOS-prefixed encoding."
        )
