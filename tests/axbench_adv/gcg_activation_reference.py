"""Original activation-L2 batch evaluation, with reference copies after forward."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch as t

from scripts.eval.axbench_adv.gcg_math import activation_l2_row


def _activation_l2_batch(
    session: Any, items: list[Any], *, gradient: bool
) -> list[tuple[t.Tensor, t.Tensor | None]]:
    """Retain the activation path from ce08487e7 independently of `_side_batch`."""
    sequences = [session._sequence(item) for item in items]
    embeds, attention = session._padded_embeds(sequences, gradient=gradient)
    hidden = session._capture_hidden(embeds, attention)
    if hidden is None or session._direction_offset is None:
        raise RuntimeError("activation_l2 state is incomplete")
    losses: list[t.Tensor] = []
    for row, (item, sequence) in enumerate(zip(items, sequences, strict=True)):
        begin = len(item.layout.fixed_before) + len(item.tokens)
        attacked = hidden[row, begin : len(sequence)]
        clean = session._clean_hidden[item.layout.sample_id].to(attacked.device)
        losses.append(activation_l2_row(attacked, clean, session._direction_offset))
    full_gradient = (
        t.autograd.grad(tuple(losses), embeds, create_graph=False)[0] if gradient else None
    )
    result: list[tuple[t.Tensor, t.Tensor | None]] = []
    for row, (item, loss) in enumerate(zip(items, losses, strict=True)):
        if full_gradient is None:
            suffix_gradient = None
        else:
            begin = len(item.layout.fixed_before)
            suffix_gradient = (
                full_gradient[row, begin : begin + len(item.tokens)]
                .detach()
                .float()
                .cpu()
                .contiguous()
            )
        result.append((loss.detach(), suffix_gradient))
    return result


@contextmanager
def late_clean_transfers(module: Any) -> Iterator[None]:
    """Keep the original per-row clean references in numerical comparisons."""
    original = module.GCGSession._side_batch

    def side_batch(
        session: Any, items: list[Any], *, gradient: bool
    ) -> list[tuple[t.Tensor, t.Tensor | None]]:
        if session.objective == "activation_l2":
            return _activation_l2_batch(session, items, gradient=gradient)
        return original(session, items, gradient=gradient)

    with patch.object(module.GCGSession, "_side_batch", side_batch):
        yield
