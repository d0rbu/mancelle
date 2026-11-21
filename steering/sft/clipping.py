"""Shared optional gradient clipping for supervised fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch as t
from loguru import logger


def resolve_grad_clip_max_norm(raw: object) -> float | None:
    """Validate and log the nullable positive SFT clipping threshold."""
    if raw is None:
        logger.info(
            "Gradient clipping disabled (default); grad_clip_max_norm=null "
            "skips torch gradient clipping completely. The historical 1.0 "
            "value from commit edc101b remains available as a NaN defense for "
            "garbled U+FFFD batches under the flex-attention fallback."
        )
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"grad_clip_max_norm must be null or a finite positive number; got {raw!r}"
        )
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"grad_clip_max_norm must be null or a finite positive number; got {raw!r}"
        )
    logger.info(f"Gradient clipping enabled: max_norm={value}")
    return value


def apply_gradient_clipping(
    parameters: Iterable[t.nn.Parameter],
    max_norm: float | None,
) -> t.Tensor | None:
    """Clip accumulated local gradients immediately before optimizer reduction.

    Distributed SFT intentionally follows the repository optimizer convention:
    each rank clips its accumulated local gradient first, then ``DistMuonAdamW``
    performs gradient reduction inside ``optimizer.step()``.
    """
    if max_norm is None:
        return None
    return t.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        error_if_nonfinite=True,
    )
