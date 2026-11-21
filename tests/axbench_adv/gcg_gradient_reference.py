"""Test-only rowwise VJP oracle for the batched AxBench-Adv adapters.

The reference at 6b741afce differentiated each scalar row loss against the
entire batch and retained only that row's input-gradient slice. Reconstruct
those slices independently, and check the row-independence precondition.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch as t


@contextmanager
def full_sequence_logits(model: t.nn.Module) -> Iterator[None]:
    """Restore the reference LM-head computation, including unused positions."""

    original = model.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("logits_to_keep", None)
        return original(*args, **kwargs)

    # PyVene clears module hooks, so a forward pre-hook cannot serve as this
    # oracle. Patching only the base model also exercises real wrapper routing.
    with patch.object(model, "forward", forward):
        yield


@contextmanager
def rowwise_vjp() -> Iterator[list[int]]:
    """Replace a joint VJP with the original rowwise derivatives of its graph."""

    original = t.autograd.grad
    calls: list[int] = []

    def backward(
        outputs: t.Tensor | Sequence[t.Tensor],
        inputs: t.Tensor | Sequence[t.Tensor],
        **kwargs: Any,
    ) -> tuple[t.Tensor | None, ...]:
        losses = (outputs,) if isinstance(outputs, t.Tensor) else tuple(outputs)
        roots = (inputs,) if isinstance(inputs, t.Tensor) else tuple(inputs)
        calls.append(len(losses))
        if len(losses) == 1:
            return original(outputs, inputs, **kwargs)
        assert all(root.shape[0] == len(losses) for root in roots)
        diagonals: list[t.Tensor | None] = [None] * len(roots)
        for row, loss in enumerate(losses):
            options: dict[str, Any] = dict(kwargs, retain_graph=row + 1 < len(losses))
            gradients = original(loss, roots, **options)
            for index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                other_rows = t.cat((gradient[:row], gradient[row + 1 :]))
                assert t.equal(other_rows, t.zeros_like(other_rows)), "rows are not independent"
                diagonal = diagonals[index]
                if diagonal is None:
                    diagonal = t.zeros_like(gradient)
                    diagonals[index] = diagonal
                diagonal[row] = gradient[row]
        return tuple(diagonals)

    with patch.object(t.autograd, "grad", backward):
        yield calls
