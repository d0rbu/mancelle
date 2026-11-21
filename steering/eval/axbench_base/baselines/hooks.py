"""Residual-stream forward-hook helper (Task 20, axbench_base Stage 3).

Installs a forward hook on ``model.transformer.h[layer]`` that adds
``coefficient * direction`` to the block's output residual. Works for
both ``nanochat.gpt.GPT`` and
``steering.conditional.encoder_decoder.model.EncoderDecoderGPT``
via the shared ``transformer.h`` attribute path.

Used by the DiffMean and LinearProbe Stage 3 baselines as their single
intervention primitive — both methods produce a steering vector of shape
``(C,)`` and a scalar coefficient, and apply it at every position of the
residual stream at a chosen layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch as t
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from steering.eval.axbench_base.types import AxBenchBaseError


def _get_block(model: nn.Module, layer: int) -> nn.Module:
    """Return ``model.transformer.h[layer]`` as an ``nn.Module``.

    The intermediate ``transformer`` attribute is an ``nn.ModuleDict``
    (whose ``__getattr__`` returns ``Tensor | Module``) and ``h`` is an
    ``nn.ModuleList`` — both need explicit casts because PyTorch's
    ``nn.Module.__getattr__`` is typed as a ``Tensor | Module`` union.

    Raises:
        IndexError: if ``layer`` is not a valid index into
            ``model.transformer.h``.
    """
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    return blocks[layer]


def register_residual_hook(
    model: nn.Module,
    layer: int,
    direction: t.Tensor,
    coefficient: float,
) -> RemovableHandle:
    """Install a forward hook that adds ``coefficient * direction`` to a layer's output.

    The hook fires on every forward pass through ``model.transformer.h[layer]``
    (prefill and decode). Inside the hook, ``direction`` is cast to the
    block output's dtype to preserve the model's residual precision
    (bf16 stays bf16, fp32 stays fp32).

    Contract:
        * ``direction`` must be a 1-D tensor of shape ``(C,)`` where
          ``C`` is ``n_embd``. Any other shape raises
          ``AxBenchBaseError``.
        * ``direction`` is moved to ``next(model.parameters()).device``
          once at registration time — the caller is responsible for
          finalising the model's device BEFORE calling this helper.
        * ``layer`` out of range raises the natural ``IndexError`` from
          ``model.transformer.h[layer]`` — there is no additional guard.
        * ``coefficient = 0.0`` is a well-defined no-op.
        * The caller is responsible for calling ``.remove()`` on the
          returned handle when the intervention is done.

    Args:
        model: A transformer with the ``model.transformer.h[i]``
            attribute path (GPT or EncoderDecoderGPT).
        layer: Index into ``model.transformer.h``.
        direction: 1-D tensor of shape ``(C,)``.
        coefficient: Scalar multiplier applied to ``direction``.

    Returns:
        A ``torch.utils.hooks.RemovableHandle``. Call ``.remove()`` on
        it to restore the original forward path exactly.
    """
    if direction.ndim != 1:
        raise AxBenchBaseError(
            f"direction must be a 1-D tensor of shape (C,), "
            f"got shape {tuple(direction.shape)} (ndim={direction.ndim}).",
        )

    model_device = next(model.parameters()).device
    direction = direction.to(model_device)

    def hook(
        module: nn.Module,
        inputs: tuple[t.Tensor, ...],
        output: t.Tensor,
    ) -> t.Tensor:
        del module, inputs
        addend = (coefficient * direction.to(output.dtype))[None, None, :]
        return output + addend

    return _get_block(model, layer).register_forward_hook(hook)


def register_batched_residual_hook(
    model: nn.Module,
    layer: int,
    directions: t.Tensor,
    coefficients: Sequence[float],
) -> RemovableHandle:
    """Install one independently scaled residual direction per batch row.

    Scaling deliberately mirrors :func:`register_residual_hook`: each Python
    coefficient multiplies a direction only after that direction is cast to
    the live output dtype. Pre-scaling in fp32 and casting the result would
    change the established bf16 rounding boundary.
    """
    if directions.ndim != 2:
        raise AxBenchBaseError(
            "batched residual directions must have shape (B, C), "
            f"got shape {tuple(directions.shape)} (ndim={directions.ndim}).",
        )
    if directions.size(0) < 1 or directions.size(1) < 1:
        raise AxBenchBaseError(
            "batched residual directions must have positive B and C dimensions, "
            f"got shape {tuple(directions.shape)}.",
        )
    row_coefficients = tuple(coefficients)
    if len(row_coefficients) != directions.size(0):
        raise AxBenchBaseError(
            "batched residual hook requires one coefficient per direction; "
            f"got {len(row_coefficients)} coefficients for {directions.size(0)} rows."
        )

    model_device = next(model.parameters()).device
    directions = directions.to(model_device)

    def hook(
        module: nn.Module,
        inputs: tuple[t.Tensor, ...],
        output: t.Tensor,
    ) -> t.Tensor:
        del module, inputs
        if output.ndim != 3:
            raise AxBenchBaseError(
                "batched residual hook expected block output with shape (B, T, C), "
                f"got {tuple(output.shape)}."
            )
        expected = (directions.size(0), directions.size(1))
        actual = (output.size(0), output.size(2))
        if actual != expected:
            raise AxBenchBaseError(
                "batched residual hook output disagrees with directions in (B, C): "
                f"output={actual}, directions={expected}."
            )
        addends = t.stack(
            [
                coefficient * direction.to(output.dtype)
                for coefficient, direction in zip(
                    row_coefficients,
                    directions,
                    strict=True,
                )
            ]
        )
        return output + addends[:, None, :]

    return _get_block(model, layer).register_forward_hook(hook)
