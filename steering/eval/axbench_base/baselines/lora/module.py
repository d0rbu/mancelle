"""Hand-rolled LoRA adapter for ``nanochat.gpt.Linear`` (Task 23).

``LoRAAdapter`` wraps a frozen ``nn.Linear`` (typically the
``nanochat.gpt.Linear`` subclass used throughout this codebase) with a
rank-r low-rank adaptation::

    adapter(x) = base(x) + (x @ A.T @ B.T) * (alpha / rank)

At initialisation ``lora_B`` is zero-filled and ``lora_A`` is
Kaiming-uniform-initialised, so the adapter forward is bit-exact equal
to the base forward.  The base weights are frozen
(``requires_grad=False``) so gradients flow only to ``lora_A`` and
``lora_B``.

Critical: ``forward`` calls ``self.base(x)`` rather than re-implementing
``F.linear``, so the ``nanochat.gpt.Linear.forward`` dtype-cast
behaviour (which casts master fp32 weights to the activation dtype on
every forward) is preserved.  Re-implementing the matmul here would
break bf16 training with fp32 master weights.
"""

from __future__ import annotations

import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from steering.refined_types import PositiveFiniteNumberLike, PositiveIntLike


class LoRAAdapter(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a rank-r LoRA update.

    Args:
        base: the linear layer to wrap.  Must be an ``nn.Linear``
            subclass (``nanochat.gpt.Linear`` qualifies).  The adapter
            freezes ``base``'s parameters in place.
        rank: rank ``r`` of the low-rank adaptation.  Must be ``>= 1``.
        alpha: scaling numerator.  Must be ``> 0``.  The effective
            scaling applied to the LoRA path is ``alpha / rank``.

    Attributes:
        base: the wrapped (frozen) ``nn.Linear``.
        rank: the LoRA rank.
        alpha: the LoRA alpha.
        scaling: ``alpha / rank``.
        lora_A: ``(rank, in_features)`` Kaiming-uniform-initialised.
        lora_B: ``(out_features, rank)`` zero-initialised.
    """

    base: nn.Linear
    rank: int
    alpha: float
    scaling: float
    lora_A: nn.Parameter
    lora_B: nn.Parameter

    @beartype
    def __init__(
        self,
        base: nn.Linear,
        rank: PositiveIntLike,
        alpha: PositiveFiniteNumberLike,
    ) -> None:
        super().__init__()

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank

        # lora_A: (rank, in_features), Kaiming-uniform (a = sqrt(5) is the
        # PyTorch default for nn.Linear's kaiming_uniform_ initialisation,
        # which is what LoRA reference implementations use).
        self.lora_A = nn.Parameter(
            t.empty(rank, base.in_features, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # lora_B: (out_features, rank), zero-initialised so the adapter
        # forward at t=0 is bit-exact equal to the base forward.
        self.lora_B = nn.Parameter(
            t.zeros(base.out_features, rank, dtype=base.weight.dtype)
        )

        # Freeze the base weights so gradients only flow to lora_A/B.
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass: ``base(x) + scaling * (x @ A.T @ B.T)``."""
        # Delegate to self.base(x) so nanochat.gpt.Linear's dtype-cast
        # behaviour (casting fp32 master weights to the input dtype) is
        # preserved. Bypassing this with F.linear(x, self.base.weight)
        # would break bf16 training with fp32 master weights.
        base_out = self.base(x)

        # Cast lora_A and lora_B to the input dtype for the same reason:
        # master fp32 LoRA parameters should not force the activation
        # pathway into fp32.
        lora_a = self.lora_A.to(dtype=x.dtype)
        lora_b = self.lora_B.to(dtype=x.dtype)

        # LoRA path: x -> F.linear(·, lora_A) -> F.linear(·, lora_B)
        # F.linear(x, W) computes x @ W.T, so:
        #   step1 = x @ lora_A.T    # (..., rank)
        #   step2 = step1 @ lora_B.T         # (..., out_features)
        step1 = F.linear(x, lora_a)
        lora_out = F.linear(step1, lora_b)

        return base_out + lora_out * self.scaling
