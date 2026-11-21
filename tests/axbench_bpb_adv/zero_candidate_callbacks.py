from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch as t


def zero_gradient_batch(
    items: Sequence[tuple[dict[str, Any], Sequence[int]]],
    _step: int,
) -> list[t.Tensor]:
    return [t.zeros((len(tokens), 1), dtype=t.float32) for _row, tokens in items]


def zero_objective(
    items: Sequence[tuple[dict[str, Any], Sequence[int]]],
) -> list[float]:
    return [0.0 for _ in items]
