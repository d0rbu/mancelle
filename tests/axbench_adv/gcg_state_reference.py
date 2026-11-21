"""Frozen GCG guard from commit 85d44829f9ba4201aff9680f24ee5d6089430c09.

Kept independent of the production implementation for numerical and timing
comparisons. This is a benchmark oracle, not a runtime fallback.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch as t


def _unique(items: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[int] = set()
    unique: list[Any] = []
    for item in items:
        if id(item) not in seen:
            seen.add(id(item))
            unique.append(item)
    return tuple(unique)


@contextmanager
def preserve_gcg_model_state(
    roots: Sequence[t.nn.Module], *, padding_owner: Any | None = None
) -> Iterator[None]:
    """Run GCG forwards without changing caller-owned Torch or RNG state."""

    modules = _unique(module for root in roots for module in root.modules())
    modes = tuple(module.training for module in modules)
    parameters = _unique(parameter for root in roots for parameter in root.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    gradients = tuple(parameter.grad for parameter in parameters)
    gradient_values = tuple(
        None if gradient is None else gradient.detach().clone() for gradient in gradients
    )
    hook_attributes = (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
    )
    hook_state = tuple(
        tuple(
            (name, getattr(module, name), getattr(module, name).copy()) for name in hook_attributes
        )
        for module in modules
    )
    full_backward_hook_flags = tuple(module._is_full_backward_hook for module in modules)
    padding_side = getattr(padding_owner, "padding_side", None)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = t.random.get_rng_state()
    cuda_rng = t.cuda.get_rng_state_all() if t.cuda.is_available() else None
    try:
        for root in roots:
            root.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        with t.no_grad():
            for parameter, required, gradient, gradient_value in zip(
                parameters, requires_grad, gradients, gradient_values, strict=True
            ):
                parameter.requires_grad_(required)
                parameter.grad = gradient
                if gradient is not None and gradient_value is not None:
                    gradient.copy_(gradient_value)
        for module, mode, saved_hooks, full_backward_hook_flag in zip(
            modules, modes, hook_state, full_backward_hook_flags, strict=True
        ):
            module.training = mode
            for name, original, saved in saved_hooks:
                if getattr(module, name) is not original:
                    setattr(module, name, original)
                original.clear()
                original.update(saved)
            module._is_full_backward_hook = full_backward_hook_flag
        if padding_owner is not None and padding_side is not None:
            padding_owner.padding_side = padding_side
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        t.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            t.cuda.set_rng_state_all(cuda_rng)


__all__ = ["preserve_gcg_model_state"]
