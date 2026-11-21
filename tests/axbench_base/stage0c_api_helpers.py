"""Test-only composition helpers for the live Stage 0c parser primitives."""

from __future__ import annotations

from typing import Any

from steering.eval.axbench_base.stage0c_api_generation import (
    _encode_continuation_text,
    _parse_continuation_text,
    _truncate_and_validate_continuation,
)
from steering.eval.axbench_base.synth.continuation_guards import ResponseEnvelope


def parse_continuation_response(
    response: str,
    *,
    envelope: ResponseEnvelope,
    prefix_text: str,
    output_length: int,
    tokenizer: Any,
    concept_definition: str | None = None,
    recurring_peak_text: str | None = None,
    required_source_bridge_text: str = "",
) -> list[int]:
    """Compose the same parser primitives used by ``Stage0cApiGenerator``."""
    continuation = _parse_continuation_text(
        response,
        envelope=envelope,
        prefix_text=prefix_text,
        concept_definition=concept_definition,
        recurring_peak_text=recurring_peak_text,
        required_source_bridge_text=required_source_bridge_text,
    )
    token_ids = _encode_continuation_text(continuation, tokenizer)
    return _truncate_and_validate_continuation(
        token_ids,
        output_length=output_length,
        tokenizer=tokenizer,
        prefix_text=prefix_text,
        concept_definition=concept_definition,
        recurring_peak_text=recurring_peak_text,
        trusted_leading_text=required_source_bridge_text,
    )
