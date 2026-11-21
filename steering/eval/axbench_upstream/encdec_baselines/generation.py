"""Pure self-conditioned generation with temporary baseline steering deltas."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch as t
import torch.nn as nn

from steering.eval.axbench_base.baselines.hooks import (
    register_batched_residual_hook,
    register_residual_hook,
)
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_upstream.encdec_baselines.lora import (
    apply_lora_state,
    attach_encdec_lora,
    detach_encdec_lora,
)
from steering.eval.axbench_upstream.methods import (
    TEMPERATURE,
    TOP_K,
    MethodSetup,
    PreparedPromptContext,
    _generate_self_conditioning,
    _validate_self_conditioning_generation_inputs,
    build_self_conditioning_config,
)
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import SingleEntryContextProvider
from steering.segmentation import SegmentMode


@dataclass(frozen=True)
class DirectionGenerationRequest:
    """One independently seeded residual-direction generation row."""

    direction: t.Tensor
    coefficient: float
    layer: int
    selection_seed: int
    token_seed: int


@contextmanager
def steering_delta(
    model: nn.Module,
    *,
    direction: t.Tensor | None = None,
    coefficient: float | None = None,
    layer: int | None = None,
    lora_state: dict[str, t.Tensor] | None = None,
    lora_layers: Sequence[int] | None = None,
) -> Iterator[None]:
    """Apply one temporary residual or LoRA delta.

    The no-argument form is invalid; prompt steering skips this context
    entirely because it has no model delta.
    """

    if (direction is None) != (coefficient is None):
        raise ValueError("direction and coefficient must be provided together")
    if direction is not None and lora_state is not None:
        raise ValueError("direction and lora_state are mutually exclusive")
    if direction is None and lora_state is None:
        raise ValueError("steering_delta requires a direction/coefficient pair or lora_state")

    if direction is not None:
        assert coefficient is not None
        if layer is None:
            raise ValueError("layer is required when direction/coefficient are provided")
        handle = register_residual_hook(
            model,
            layer,
            direction,
            coefficient,
        )
        try:
            yield
        finally:
            handle.remove()
        return

    assert lora_state is not None
    if lora_layers is None:
        raise ValueError("lora_layers is required when lora_state is provided")
    refs = attach_encdec_lora(model, lora_layers)
    try:
        apply_lora_state(refs, lora_state)
        yield
    finally:
        detach_encdec_lora(model, refs)


def generate_encdec_baseline(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    *,
    selection_seed: int,
    seed: int,
    max_new_tokens: int,
) -> str:
    """Generate from prepared prompt features under pure self-conditioning."""
    return _generate_self_conditioning(
        setup,
        model,
        tokenizer,
        prepared,
        None,
        selection_seed=selection_seed,
        seed=seed,
        max_new_tokens=max_new_tokens,
        profile_driver="encdec",
    )


def generate_encdec_direction_batch(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    requests: Sequence[DirectionGenerationRequest],
    *,
    max_new_tokens: int,
) -> tuple[str, ...]:
    """Batch fixed-token direction cells sharing one exact prepared prompt.

    Prefills remain scalar under each row's original intervention, which
    preserves the prompt-conditioned S0 state and KV contents. Decode is
    batched only for a budget that cannot reach the next self-conditioning
    boundary; each row retains its own token RNG and stop state.

    Turn-delimited callers must use :func:`generate_encdec_baseline` once per
    row under that row's :func:`steering_delta` context.
    """
    _validate_self_conditioning_generation_inputs(
        model,
        prepared,
        max_new_tokens=max_new_tokens,
    )
    rows = tuple(requests)
    if len(rows) < 2:
        raise ValueError(
            "generate_encdec_direction_batch requires at least two rows; "
            "use generate_encdec_baseline for a scalar cell"
        )
    layers = {request.layer for request in rows}
    if len(layers) != 1:
        raise ValueError(f"batched direction rows must share one intervention layer: {layers}")
    if any(request.direction.ndim != 1 for request in rows):
        shapes = [tuple(request.direction.shape) for request in rows]
        raise ValueError(f"batched directions must all have shape (C,), got {shapes}")
    widths = {int(request.direction.numel()) for request in rows}
    if len(widths) != 1:
        raise ValueError(f"batched directions must share one width, got {sorted(widths)}")
    if setup.segment_mode is not SegmentMode.fixed_tokens:
        raise ValueError(
            "generate_encdec_direction_batch supports only fixed_tokens; "
            "turn-delimited generation requires the scalar cell dispatcher"
        )

    prefills = []
    engine = None
    with profile_region(
        "axbench_upstream.generation.batch_prefills",
        driver="encdec",
        physical_batch_size=len(rows),
        prompt_tokens=len(prepared.token_ids),
    ):
        for request in rows:
            config = build_self_conditioning_config(
                setup,
                model,
                None,
                selection_seed=request.selection_seed,
            )
            provider = SingleEntryContextProvider(
                prepared.token_ids,
                prepared.segments,
                segment_mode=prepared.segment_mode,
                token_assignments=prepared.token_assignments,
            )
            engine = build_engine_self_conditioned(
                model,
                tokenizer,
                next(model.parameters()).device,
                config,
                context_provider=provider,
            )
            with steering_delta(
                model,
                direction=request.direction,
                coefficient=request.coefficient,
                layer=request.layer,
            ):
                prefills.append(
                    engine.prepare_self_conditioned_prefill(
                        list(prepared.token_ids),
                        max_new_tokens=max_new_tokens,
                    )
                )
    assert engine is not None

    directions = t.stack([request.direction for request in rows], dim=0)
    handle = register_batched_residual_hook(
        model,
        rows[0].layer,
        directions,
        [request.coefficient for request in rows],
    )
    try:
        with profile_region(
            "axbench_upstream.generation.batch_stream",
            driver="encdec",
            physical_batch_size=len(rows),
            prompt_tokens=len(prepared.token_ids),
            max_new_tokens=max_new_tokens,
        ):
            generated = engine.generate_self_conditioned_batch(
                prefills,
                seeds=[request.token_seed for request in rows],
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                stop_token_ids={
                    int(tokenizer.get_bos_token_id()),
                    prepared.assistant_end_id,
                },
            )
    finally:
        handle.remove()

    with profile_region(
        "axbench_upstream.generation.tokenizer_decode_batch",
        driver="encdec",
        physical_batch_size=len(rows),
    ):
        return tuple(tokenizer.decode(tokens) for tokens in generated)


__all__ = (
    "DirectionGenerationRequest",
    "generate_encdec_baseline",
    "generate_encdec_direction_batch",
    "steering_delta",
)
