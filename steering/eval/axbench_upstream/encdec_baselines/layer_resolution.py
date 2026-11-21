"""Depth-relative layer resolution for the encoder-decoder baseline analogs.

The vendored AxBench Qwen2.5-3B sweep pins its DiffMean/LinearProbe/
PromptSteering residual hook at ``layer: 28`` and its LoRA adapters at
``lora_layers: [7, 14, 21, 28]``, out of 36 decoder layers (see
``vendor/axbench/axbench/sweep/nano_gpt_pretrain_steer/qwen2_5_3b_{main,aux}.yaml``).
The encoder-decoder analogs preserve the vendored FRACTION of decoder depth,
not the vendored literal index, so a target checkpoint of any decoder depth
gets a well-defined mapping instead of an out-of-range constant. Round-half-up
on ``target_n_layer * vendored_layer / VENDORED_QWEN_N_LAYER`` reproduces the
originally hand-picked 12-layer target constants (``9`` and ``(2, 5, 7, 9)``)
exactly, so it is a generalization of that choice, not a reinterpretation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

VENDORED_QWEN_N_LAYER = 36
VENDORED_STEERING_LAYER = 28
VENDORED_LORA_LAYERS = (7, 14, 21, 28)


def resolve_vendored_layer(vendored_layer: int, *, target_n_layer: int) -> int:
    """Map one vendored-Qwen layer index onto a ``target_n_layer``-deep decoder."""

    if type(vendored_layer) is not int or not 0 <= vendored_layer < VENDORED_QWEN_N_LAYER:
        raise ValueError(
            f"vendored_layer must be an integer in [0, {VENDORED_QWEN_N_LAYER}), "
            f"got {vendored_layer!r}"
        )
    if type(target_n_layer) is not int or target_n_layer <= 0:
        raise ValueError(f"target_n_layer must be a positive integer, got {target_n_layer!r}")
    resolved = math.floor(target_n_layer * vendored_layer / VENDORED_QWEN_N_LAYER + 0.5)
    if not 0 <= resolved < target_n_layer:
        raise ValueError(
            f"resolved layer {resolved} is out of range for a {target_n_layer}-layer decoder "
            f"(vendored_layer={vendored_layer} of {VENDORED_QWEN_N_LAYER})"
        )
    return resolved


def resolve_vendored_layers(
    vendored_layers: Sequence[int],
    *,
    target_n_layer: int,
) -> tuple[int, ...]:
    """Map several vendored-Qwen layer indices, failing loud on any collision."""

    resolved = tuple(
        resolve_vendored_layer(layer, target_n_layer=target_n_layer) for layer in vendored_layers
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"target_n_layer={target_n_layer} is too shallow to give vendored_layers="
            f"{tuple(vendored_layers)} distinct fractionally-mapped layers: resolved {resolved}"
        )
    return resolved


__all__ = (
    "VENDORED_LORA_LAYERS",
    "VENDORED_QWEN_N_LAYER",
    "VENDORED_STEERING_LAYER",
    "resolve_vendored_layer",
    "resolve_vendored_layers",
)
