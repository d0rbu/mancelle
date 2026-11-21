"""Chat rendering and next-token targets for encoder-decoder fit rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch as t

from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.sft.loss_mask import apply_completion_only_row_masks


@dataclass(frozen=True)
class RenderedFitRow:
    """One fit pair rendered with the nanochat supervised-token mask."""

    ids: tuple[int, ...]
    mask: tuple[int, ...]
    prompt_token_count: int


def template_prefix_length(tokenizer: Any) -> int:
    """Return the shared template prefix before one-token user texts differ."""

    first = render_prompt_for_chat_completion(tokenizer, "1").token_ids
    second = render_prompt_for_chat_completion(tokenizer, "2").token_ids
    for index, (first_id, second_id) in enumerate(zip(first, second, strict=False)):
        if first_id != second_id:
            return index
    if len(first) != len(second):
        return min(len(first), len(second))
    raise RuntimeError("chat-completion probes have no differing token")


def render_fit_row(
    tokenizer: Any,
    *,
    input_text: str,
    output_text: str,
    sequence_len: int,
    surface: str,
) -> RenderedFitRow:
    """Render a user-assistant fit pair without silently truncating it."""

    if not output_text:
        raise ValueError(f"{surface}: output_text must be nonempty")
    ids, mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]
        },
        max_tokens=sequence_len + 1,
    )
    if len(ids) > sequence_len:
        raise ValueError(
            f"{surface}: rendered fit row has {len(ids)} tokens, "
            f"exceeding sequence_len={sequence_len}"
        )

    ids_tuple = tuple(ids)
    mask_tuple = tuple(mask)
    try:
        prompt_token_count = mask_tuple.index(1)
    except ValueError as exc:
        raise RuntimeError(f"{surface}: rendered fit row has no supervised tokens") from exc
    prompt_tokens, _ = tokenizer.render_conversation(
        {"messages": [{"role": "user", "content": input_text}]}, max_tokens=None,
    )
    prompt_ids = (*prompt_tokens, tokenizer.encode_special("<assistant_start>"))
    if ids_tuple[:prompt_token_count] != prompt_ids:
        raise RuntimeError(f"{surface}: rendered prompt tokens do not match completion prompt")
    return RenderedFitRow(
        ids=ids_tuple,
        mask=mask_tuple,
        prompt_token_count=prompt_token_count,
    )


def targets_from_mask(ids: Sequence[int], mask: Sequence[int]) -> list[int]:
    """Build shifted causal targets supervised only where the next mask is one."""

    if len(ids) != len(mask):
        raise ValueError(f"ids and mask must have equal length, got {len(ids)} and {len(mask)}")
    if not ids:
        raise ValueError("ids and mask must be nonempty")
    targets = t.tensor([[*ids[1:], -1]], dtype=t.int64)
    apply_completion_only_row_masks(targets, [mask])
    return targets[0].tolist()


__all__ = [
    "RenderedFitRow",
    "render_fit_row",
    "targets_from_mask",
    "template_prefix_length",
]
