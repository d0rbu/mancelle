"""Byte-level insertion of a coherent suffix into a supported host prompt."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from beartype import beartype

from steering.eval.axbench_adv.errors import AXBENCH_ADV_BEARTYPE, AxBenchAdvError
from steering.refined_types import NonEmptyStrLike, PositiveIntLike

if TYPE_CHECKING:
    _ForbiddenTokensLike = tuple[str, ...]
    _PromptSplitLike = tuple[str, str]
else:
    # Beartype samples parameterized containers through Python's global RNG.
    _ForbiddenTokensLike = tuple
    _PromptSplitLike = tuple


QWEN_TERMINAL_MARKER = "<|im_end|>\n<|im_start|>assistant\n"
SMOLLM3_NO_THINK_TERMINAL_MARKER = QWEN_TERMINAL_MARKER + "<think>\n\n</think>\n"
PYTHIA_TERMINAL_MARKER = "<|endoftext|>\nassistant:\n"
QWEN_CONTROL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
)
SMOLLM3_CONTROL_TOKENS: tuple[str, ...] = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<think>",
    "</think>",
    "<|finetune_right_pad_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_response>",
    "</tool_response>",
    "<tool_call>",
    "</tool_call>",
    "<code>",
    "</code>",
    "<|reserved_special_token_2|>",
    *(f"<|reserved_special_token_{index}|>" for index in range(11, 248)),
)
PYTHIA_CONTROL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|padding|>",
    "<tool_response>",
    "</tool_response>",
    "<tool_call>",
    "</tool_call>",
)
SUFFIX_SEPARATOR = "\n\n"
CONTROL_TOKENS_BY_MARKER = {
    SMOLLM3_NO_THINK_TERMINAL_MARKER: SMOLLM3_CONTROL_TOKENS,
    QWEN_TERMINAL_MARKER: QWEN_CONTROL_TOKENS,
    PYTHIA_TERMINAL_MARKER: PYTHIA_CONTROL_TOKENS,
}


@beartype(conf=AXBENCH_ADV_BEARTYPE)
def canonicalize_suffix(
    suffix: str,
    *,
    forbidden_tokens: _ForbiddenTokensLike = QWEN_CONTROL_TOKENS,
) -> str:
    """Strip only outer whitespace and reject the selected host's control text."""

    if "\x00" in suffix:
        raise AxBenchAdvError("suffix must not contain NUL")
    if any(token in suffix for token in forbidden_tokens):
        label = "Qwen" if forbidden_tokens == QWEN_CONTROL_TOKENS else "host"
        raise AxBenchAdvError(f"suffix must not contain {label} control tokens")
    try:
        suffix.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise AxBenchAdvError("suffix must be valid UTF-8 without surrogate code points") from exc
    return suffix.strip()


@beartype(conf=AXBENCH_ADV_BEARTYPE)
def split_terminal_marker(prompt: NonEmptyStrLike) -> _PromptSplitLike:
    """Split before the host's unique terminal assistant-generation marker."""

    marker = next((tail for tail in CONTROL_TOKENS_BY_MARKER if prompt.endswith(tail)), None)
    if marker is None:
        if any(tail in prompt for tail in CONTROL_TOKENS_BY_MARKER):
            raise AxBenchAdvError("assistant-generation marker must be the prompt tail")
        raise AxBenchAdvError(
            "prompt must contain a known terminal marker once, observed 0 occurrences; "
            f"tail was {prompt[-64:]!r}"
        )
    occurrences = prompt.count(marker)
    if occurrences != 1:
        raise AxBenchAdvError(
            "prompt must contain its terminal marker exactly once, "
            f"observed {occurrences} occurrences"
        )
    prefix = prompt[: -len(marker)]
    if not prefix:
        raise AxBenchAdvError("prompt has no content before the terminal marker")
    return prefix, marker


def compose_suffix(prompt: str, suffix: str) -> str:
    """Insert a canonical suffix without performing tokenizer-dependent checks."""

    prefix, marker = split_terminal_marker(prompt)
    canonical = canonicalize_suffix(suffix, forbidden_tokens=CONTROL_TOKENS_BY_MARKER[marker])
    return prompt if not canonical else f"{prefix}{SUFFIX_SEPARATOR}{canonical}{marker}"


@beartype(conf=AXBENCH_ADV_BEARTYPE)
def insert_suffix(
    prompt: str,
    suffix: str,
    *,
    token_count: Callable[[str], int],
    max_prompt_tokens: PositiveIntLike,
) -> str:
    """Insert a canonical suffix before the marker, enforcing context capacity.

    The empty suffix is an exact identity operation: it adds no separator and
    returns the original Python string unchanged.
    """

    attacked = compose_suffix(prompt, suffix)
    try:
        observed_count = token_count(attacked)
    except Exception as exc:
        raise AxBenchAdvError("failed to count attacked-prompt tokens") from exc
    if isinstance(observed_count, bool) or not isinstance(observed_count, int):
        raise AxBenchAdvError("token_count must return an integer")
    if observed_count < 0:
        raise AxBenchAdvError("token_count returned a negative value")
    if observed_count > max_prompt_tokens:
        raise AxBenchAdvError(
            f"attacked prompt has {observed_count} tokens, exceeding context limit "
            f"{max_prompt_tokens}"
        )
    return attacked


__all__ = [
    "CONTROL_TOKENS_BY_MARKER",
    "QWEN_CONTROL_TOKENS",
    "QWEN_TERMINAL_MARKER",
    "PYTHIA_CONTROL_TOKENS",
    "PYTHIA_TERMINAL_MARKER",
    "SMOLLM3_CONTROL_TOKENS",
    "SMOLLM3_NO_THINK_TERMINAL_MARKER",
    "SUFFIX_SEPARATOR",
    "canonicalize_suffix",
    "compose_suffix",
    "insert_suffix",
    "split_terminal_marker",
]
