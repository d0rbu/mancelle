"""Shared prompt rendering and deduplication for AxBench upstream drivers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RenderedPromptPlan:
    """Rendered prompts indexed for source-order generation and canonical preparation."""

    rows: tuple[Any, ...]
    rendered_tokens_by_pair: dict[tuple[int, int], tuple[int, ...]]
    rendered_prompt_by_pair: dict[tuple[int, int], tuple[str, int]]
    prompt_by_tokens: dict[tuple[int, ...], tuple[str, int]]


def build_rendered_prompt_plan(
    frame: pd.DataFrame,
    nanochat_tokenizer: Any,
    *,
    declared_factors: tuple[float, ...],
    render: Callable[[Any, str], Any],
) -> RenderedPromptPlan:
    """Render source prompts and deduplicate token twins in canonical pair order.

    ``render`` is the caller's prompt renderer (production callers pass
    ``render_prompt_for_chat_completion`` by name so their existing
    module-level test seams keep working); it must return an object with
    ``token_ids`` and ``assistant_end_id``.
    """

    rows: tuple[Any, ...] = tuple(frame.itertuples(index=False))
    rendered_tokens_by_pair: dict[tuple[int, int], tuple[int, ...]] = {}
    rendered_prompt_by_pair: dict[tuple[int, int], tuple[str, int]] = {}
    for row in rows:
        concept_id = int(row.concept_id)
        input_id = int(row.input_id)
        factor = float(row.factor)
        if factor not in declared_factors:
            raise ValueError(
                f"SelfConditioning factor {factor!r} is not in the declared factor grid"
            )
        prompt = str(row.original_prompt)
        rendered = render(nanochat_tokenizer, prompt)
        exact_tokens = tuple(int(token_id) for token_id in rendered.token_ids)
        if not exact_tokens:
            raise ValueError(
                "rendered prompt token tuple must be nonempty for "
                f"concept_id={concept_id}, input_id={input_id}"
            )
        pair = (concept_id, input_id)
        prior_tokens = rendered_tokens_by_pair.setdefault(pair, exact_tokens)
        if prior_tokens != exact_tokens:
            raise ValueError(
                "rendered prompt token tuple varies across factors for "
                f"concept_id={concept_id}, input_id={input_id}"
            )
        prior_prompt = rendered_prompt_by_pair.setdefault(
            pair,
            (prompt, int(rendered.assistant_end_id)),
        )
        if prior_prompt != (prompt, int(rendered.assistant_end_id)):
            raise ValueError(
                "rendered prompt or assistant-end ID varies across factors for "
                f"concept_id={concept_id}, input_id={input_id}"
            )

    # Prompt feature cardinalities share the checkpoint's seeded RNG across
    # segments and entries. Consume it in canonical pair order so input-row or
    # factor order cannot change any prepared prompt bag. Exact token twins are
    # still extracted once, at their canonical first pair.
    prompt_by_tokens: dict[tuple[int, ...], tuple[str, int]] = {}
    for pair in sorted(rendered_tokens_by_pair):
        exact_tokens = rendered_tokens_by_pair[pair]
        prompt, assistant_end_id = rendered_prompt_by_pair[pair]
        prior_prompt = prompt_by_tokens.setdefault(
            exact_tokens,
            (prompt, assistant_end_id),
        )
        if prior_prompt[1] != assistant_end_id:
            raise ValueError("deduplicated prompt token tuple has inconsistent assistant-end IDs")

    return RenderedPromptPlan(
        rows=rows,
        rendered_tokens_by_pair=rendered_tokens_by_pair,
        rendered_prompt_by_pair=rendered_prompt_by_pair,
        prompt_by_tokens=prompt_by_tokens,
    )
