"""Scoped eager-evaluation callbacks at the post-cross-attention residual site."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch as t

from nanochat.common import COMPUTE_DTYPE

PostCrossAttentionCallback = Callable[[int, t.Tensor], t.Tensor]


def _callback_slot(model: Any) -> PostCrossAttentionCallback | None:
    if not hasattr(model, "_post_cross_attention_eval_callback"):
        raise TypeError(
            "post-cross-attention callbacks require an EncoderDecoderGPT model"
        )
    callback = model._post_cross_attention_eval_callback
    if callback is not None and not callable(callback):
        raise RuntimeError("post-cross-attention callback slot is corrupt")
    return callback


@contextmanager
def scoped_post_cross_attention_callback(
    model: Any,
    callback: PostCrossAttentionCallback,
) -> Iterator[None]:
    """Install one callback on the exact raw eager model used for the forward.

    Compiled wrappers and in-place compiled modules are rejected. A separately
    cached compiled wrapper around this raw module is outside the contract and
    must not be invoked while the scope is active: Dynamo specializes the
    default-``None`` slot away without guarding later Python-attribute changes.
    """

    if not callable(callback):
        raise TypeError("post-cross-attention callback must be callable")
    if (
        isinstance(model, t._dynamo.eval_frame.OptimizedModule)
        or getattr(model, "_compiled_call_impl", None) is not None
    ):
        raise RuntimeError(
            "post-cross-attention callbacks require the exact raw eager model; "
            "compiled wrappers and in-place compiled modules are unsupported"
        )
    if bool(model.training):
        raise RuntimeError("post-cross-attention callbacks require model.eval()")
    if _callback_slot(model) is not None:
        raise RuntimeError("a post-cross-attention callback is already installed")
    model._post_cross_attention_eval_callback = callback
    try:
        yield
    finally:
        model._post_cross_attention_eval_callback = None


@contextmanager
def post_cross_attention_residual_counter(
    model: Any,
    *,
    layer: int,
    direction: t.Tensor,
    magnitude: float,
) -> Iterator[None]:
    """Subtract one fixed saved direction after cross-attention at one layer.

    Direction validation and device/dtype conversion happen once at scope
    entry. The callback itself performs only a layer comparison, cheap shape
    checks, and the residual subtraction; it has no host synchronization.
    ``magnitude == 0`` is a literal no-installation path.
    """

    if type(layer) is not int or layer < 0:
        raise ValueError("counter layer must be a nonnegative integer")
    n_layers = int(model.dec_config.n_layer)
    if layer >= n_layers:
        raise ValueError(f"counter layer {layer} outside model with {n_layers} layers")
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise ValueError("counter magnitude must be finite and nonnegative")
    coefficient = float(magnitude)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("counter magnitude must be finite and nonnegative")
    if (
        not isinstance(direction, t.Tensor)
        or direction.ndim != 1
        or direction.numel() == 0
        or not direction.is_floating_point()
    ):
        raise ValueError("counter direction must be a nonempty floating vector")
    expected_width = int(model.dec_config.n_embd)
    if direction.numel() != expected_width:
        raise ValueError(
            f"counter direction width {direction.numel()} does not match model width "
            f"{expected_width}"
        )
    detached_direction = direction.detach()
    try:
        input_is_finite = bool(t.isfinite(detached_direction).all().item())
    except (NotImplementedError, RuntimeError):
        input_is_finite = bool(
            t.isfinite(detached_direction.to(dtype=t.float32)).all().item()
        )
    if not input_is_finite:
        raise ValueError("counter direction must contain only finite values")
    if coefficient == 0.0:
        yield
        return

    model_device = model.get_device()
    # Cast the saved direction directly to the residual dtype. Routing through
    # float32 first would double-round accepted float64 inputs and differ from
    # the documented ``saved_direction.to(residual)`` arithmetic.
    resolved_direction = detached_direction.to(
        device=model_device,
        dtype=COMPUTE_DTYPE,
    )
    if not bool(t.isfinite(resolved_direction).all().item()):
        raise ValueError("counter direction overflows the model compute dtype")
    offset = coefficient * resolved_direction
    if not bool(t.isfinite(offset).all().item()):
        raise ValueError("counter magnitude times direction is non-finite in compute dtype")

    def apply(layer_index: int, residual: t.Tensor) -> t.Tensor:
        if layer_index != layer:
            return residual
        if residual.ndim < 1 or residual.shape[-1] != expected_width:
            raise ValueError(
                f"post-cross-attention residual width at layer {layer} does not match "
                f"counter direction width {expected_width}"
            )
        if residual.device != offset.device or residual.dtype != offset.dtype:
            raise RuntimeError(
                "post-cross-attention residual device/dtype changed after counter preflight: "
                f"residual={residual.device}/{residual.dtype}, "
                f"direction={offset.device}/{offset.dtype}"
            )
        return residual - offset

    with scoped_post_cross_attention_callback(model, apply):
        yield


__all__ = (
    "PostCrossAttentionCallback",
    "post_cross_attention_residual_counter",
    "scoped_post_cross_attention_callback",
)
