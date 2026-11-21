"""Dependency-light numeric kernels shared by AxBench-Adv GCG runtimes."""

from __future__ import annotations

import math

import torch as t


def response_margin_value(
    negative_bpb: float | t.Tensor,
    positive_bpb: float | t.Tensor,
    *,
    alpha: float,
) -> float | t.Tensor:
    """Return the minimized response margin: negative minus alpha times positive."""

    return negative_bpb - alpha * positive_bpb


def response_margin_gradient(
    negative_gradient: t.Tensor,
    positive_gradient: t.Tensor,
    *,
    alpha: float,
) -> t.Tensor:
    """Differentiate :func:`response_margin_value` without changing tensor order."""

    return negative_gradient - alpha * positive_gradient


def weighted_bpb(
    token_losses: t.Tensor,
    byte_weights: t.Tensor,
    *,
    tensor_ln2: bool,
) -> t.Tensor:
    """Reduce output-token NLLs with AxBench's byte-weighted BPB formula.

    ``tensor_ln2`` preserves the old single-sample reduction's device tensor;
    the batch reducer historically used the Python float expression instead.
    The caller retains surface-specific shape, finiteness, and error handling.
    """

    losses = token_losses.to(t.float32)
    weights = byte_weights.to(t.float32)
    ln2 = t.log(t.tensor(2.0, device=losses.device)) if tensor_ln2 else math.log(2.0)
    return (losses * weights).sum() / (ln2 * weights.sum())


def activation_l2_row(
    attacked_hidden: t.Tensor,
    clean_hidden: t.Tensor,
    direction_offset: t.Tensor,
) -> t.Tensor:
    """Mean tokenwise Euclidean norm for one fixed-response training row."""

    if attacked_hidden.ndim != 2 or clean_hidden.shape != attacked_hidden.shape:
        raise ValueError(
            "clean and attacked hidden states must have the same (tokens, hidden) shape"
        )
    if direction_offset.ndim != 1 or direction_offset.shape[0] != attacked_hidden.shape[1]:
        raise ValueError("direction offset width must match hidden-state width")
    if attacked_hidden.shape[0] == 0:
        raise ValueError("activation_l2 requires at least one aligned token")
    delta = (
        attacked_hidden.to(t.float32)
        + direction_offset.to(device=attacked_hidden.device, dtype=t.float32)
        - clean_hidden.detach().to(device=attacked_hidden.device, dtype=t.float32)
    )
    return t.linalg.vector_norm(delta, ord=2, dim=-1).mean()


def activation_l2_pair(positive_row_loss: t.Tensor, negative_row_loss: t.Tensor) -> t.Tensor:
    """Give both classes equal weight before the population mean."""

    return 0.5 * (positive_row_loss + negative_row_loss)


__all__ = [
    "activation_l2_pair",
    "activation_l2_row",
    "response_margin_gradient",
    "response_margin_value",
    "weighted_bpb",
]
