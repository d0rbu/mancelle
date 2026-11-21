"""Convert AxBench synth parquet rows into teacher-forcing inputs."""
from __future__ import annotations

from typing import Any

import torch

from steering.eval.axbench_bpb.sample_loader import (
    make_prefix_hash,
    make_sample_id,
)
from steering.eval.axbench_bpb.tokenization import (
    encode_output,
    encode_prefix,
    truncate_prefix_left,
)
from steering.eval.axbench_bpb.types import ScoringInputs, SteeringMode


def parquet_row_to_scoring_input(
    parquet_row: dict,
    *,
    mode: SteeringMode,
    context_segments_by_hash: dict[str, tuple],
    tokenizer: Any,
    max_seq_len: int,
    token_bytes: torch.Tensor,
) -> ScoringInputs:
    """Convert one synth sample row into ``ScoringInputs``.

    The returned prefix tokens are intentionally untruncated. ``score_sample``
    re-runs truncation against the active target model's sequence length so
    each target owns its own context-window policy.
    """
    prefix_text = parquet_row["prefix_text"]
    output_text = parquet_row["output_text"]
    content_class = parquet_row["content_class"]
    concept_id = int(parquet_row["concept_id"])
    prefix_hash = make_prefix_hash(prefix_text)
    prefix_tokens = encode_prefix(prefix_text, tokenizer)
    output_tokens = encode_output(output_text, tokenizer)
    _, truncated = truncate_prefix_left(
        prefix=prefix_tokens,
        n_output=len(output_tokens),
        max_seq_len=max_seq_len,
    )
    output_byte_weights = token_bytes[
        torch.tensor(output_tokens, dtype=torch.long)
    ]
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=concept_id,
            mode=mode,
            content_class=content_class,
            prefix_text=prefix_text,
            output_text=output_text,
        ),
        concept_id=concept_id,
        mode=mode,
        content_class=content_class,
        prefix_text=prefix_text,
        prefix_hash=prefix_hash,
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=context_segments_by_hash.get(prefix_hash, ()),
        n_prefix=len(prefix_tokens),
        n_output=len(output_tokens),
        output_byte_weights=output_byte_weights,
        truncated=truncated,
    )


__all__ = ["parquet_row_to_scoring_input"]
