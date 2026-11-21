"""Frozen side computation from 5b29e4b0460dc4a339465d6ade23cb5a28efe435.

Only the name/signature and module-helper bindings differ from the parent.
Targets and byte weights are independently rebuilt on every call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch as t


@contextmanager
def fresh_response_tensors(worker: Any) -> Iterator[None]:
    """Use the frozen per-call construction instead of the prepared-cell tensors."""

    def side(session: Any, items: Any, *, gradient: bool) -> Any:
        return fresh_response_side_batch(worker, session, items, gradient=gradient)

    with patch.object(worker.GCGSession, "_side_batch", side):
        yield


def fresh_response_side_batch(
    worker: Any,
    self: Any,
    items: list[Any],
    *,
    gradient: bool,
) -> list[tuple[t.Tensor, t.Tensor | None]]:
    _forward = worker._forward
    F = worker.F
    weighted_bpb = worker.weighted_bpb
    activation_l2_row = worker.activation_l2_row
    sequences = [self._sequence(item) for item in items]
    embeds, attention = self._padded_embeds(sequences, gradient=gradient)
    if self.objective == "response_margin":
        output = _forward(self.runtime, embeds=embeds, attention_mask=attention)
        if output is None or not hasattr(output, "logits"):
            raise RuntimeError("response-margin forward returned no logits")
        hidden = None
    else:
        output = None
        hidden = self._capture_hidden(embeds, attention)
    losses: list[t.Tensor] = []
    for row, (item, sequence) in enumerate(zip(items, sequences, strict=True)):
        layout = item.layout
        response_start = len(layout.fixed_before) + len(item.tokens) + len(layout.fixed_after)
        if self.objective == "response_margin":
            selected_logits = output.logits[row, response_start - 1 : len(sequence) - 1].float()  # ty: ignore[unresolved-attribute]
            targets = t.tensor(layout.response_ids, dtype=t.long, device=selected_logits.device)
            token_losses = F.cross_entropy(selected_logits, targets, reduction="none")
            weights = t.tensor(
                layout.response_byte_weights, dtype=t.float32, device=selected_logits.device
            )
            loss = weighted_bpb(token_losses, weights, tensor_ln2=False)
        else:
            if hidden is None or self._direction_offset is None:
                raise RuntimeError("activation_l2 state is incomplete")
            aligned_start = len(layout.fixed_before) + len(item.tokens)
            attacked = hidden[row, aligned_start : len(sequence)]
            clean = self._clean_hidden[layout.sample_id].to(attacked.device)
            loss = activation_l2_row(attacked, clean, self._direction_offset)
        losses.append(loss)
    # Rows are independent: the joint VJP preserves each row's gradient
    # without traversing the full batch's graph once per scalar loss.
    full_gradient = (
        t.autograd.grad(tuple(losses), embeds, create_graph=False)[0] if gradient else None
    )
    results: list[tuple[t.Tensor, t.Tensor | None]] = []
    for row, (item, loss) in enumerate(zip(items, losses, strict=True)):
        if full_gradient is not None:
            begin = len(item.layout.fixed_before)
            gradient_value = (
                full_gradient[row, begin : begin + len(item.tokens)]
                .detach()
                .float()
                .cpu()
                .contiguous()
            )
        else:
            gradient_value = None
        results.append((loss.detach(), gradient_value))
    return results
