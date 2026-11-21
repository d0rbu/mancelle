"""Residual counter primitive usable in both Python environments without AxBench imports."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import torch as t


def load_vendor_direction(path: Path, concept_id: int, expected_sha256: str) -> t.Tensor:
    """Read the exact saved AxBench projection row (bias is not a direction)."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"counter direction hash changed: {path}")
    weights = t.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(weights, t.Tensor) or weights.ndim != 2 or not 0 <= concept_id < len(weights):
        raise ValueError(f"missing counter direction for concept {concept_id}: {path}")
    direction = weights[concept_id].detach().clone()
    validate_direction(direction)
    return direction


def validate_direction(direction: t.Tensor) -> None:
    if (
        direction.ndim != 1
        or not direction.numel()
        or not direction.is_floating_point()
        or not bool(t.isfinite(direction).all())
    ):
        raise ValueError("counter direction must be a nonempty finite floating vector")


@contextmanager
def residual_counter(
    model: Any,
    *,
    layer: int,
    direction: t.Tensor,
    magnitude: float,
) -> Iterator[None]:
    """Subtract at a vendored baseline's block output; always remove the hook."""
    validate_direction(direction)
    if type(layer) is not int or layer < 0:
        raise ValueError("counter layer must be a nonnegative integer")
    if isinstance(magnitude, bool) or not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("counter magnitude must be finite and nonnegative")
    if getattr(model.config, "model_type", None) == "gpt_neox":
        blocks = model.gpt_neox.layers
    else:
        blocks = model.model.layers
    if layer >= len(blocks):
        raise ValueError(f"counter layer {layer} outside model with {len(blocks)} layers")
    if magnitude == 0:
        yield
        return

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        if not isinstance(residual, t.Tensor) or residual.shape[-1] != direction.numel():
            raise ValueError(
                f"counter direction width {direction.numel()} does not match layer {layer}"
            )
        shifted = residual - magnitude * direction.to(device=residual.device, dtype=residual.dtype)
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    model_vars = vars(model)
    instance_forward = model_vars.get("forward")
    had_instance_forward = "forward" in model_vars
    original_forward = model.forward
    ordered_hooks = getattr(original_forward, "_axbench_adv_counter_hooks", None)
    # Nested scopes share the outer wrapper so hooks retain scope-entry order.
    if getattr(original_forward, "_axbench_adv_counter_model", None) is model and isinstance(
        ordered_hooks, list
    ):
        entry = (blocks[layer], hook)
        ordered_hooks.append(entry)
        try:
            yield
        finally:
            ordered_hooks.remove(entry)
        return

    scope_hooks = [(blocks[layer], hook)]

    @wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        handles = []
        try:
            for block, callback in tuple(scope_hooks):
                handles.append(block.register_forward_hook(callback))
            return original_forward(*args, **kwargs)
        finally:
            for handle in reversed(handles):
                handle.remove()

    forward.__dict__["_axbench_adv_counter_model"] = model
    forward.__dict__["_axbench_adv_counter_hooks"] = scope_hooks
    model.forward = forward
    try:
        yield
    finally:
        if had_instance_forward:
            model.forward = instance_forward
        else:
            del model.forward
