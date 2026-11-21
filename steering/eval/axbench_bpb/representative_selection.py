"""Shared representative-coefficient selection helpers for BPB evals."""

from __future__ import annotations

import math
from typing import Any

from steering.eval.axbench_base.types import SteeringMode

FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE: dict[str, str] = {
    SteeringMode.POSITIVE.value: "feature_diffmean_encdec",
    SteeringMode.NEGATIVE.value: "neg_feature_diffmean_encdec",
}


def is_no_coefficient(value: Any) -> bool:
    """Return true for both in-memory and persisted no-coefficient sentinels."""

    return value is None or str(value) == "None"


def coefficient_abs(value: Any) -> float:
    """Absolute numeric coefficient, with non-numeric values sorted last."""

    if is_no_coefficient(value):
        return math.inf
    try:
        return abs(float(str(value)))
    except ValueError:
        return math.inf


def coefficient_lex(value: Any) -> str:
    """Stable lexicographic representation for final tie-breaks."""

    return "None" if value is None else str(value)


__all__ = [
    "FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE",
    "coefficient_abs",
    "coefficient_lex",
    "is_no_coefficient",
]
