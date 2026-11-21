"""Canonical percentile-unit conversions for AxBench QC.

Configuration stores percentile thresholds as fractions of percentile space
(``0.3`` means 30 percentile points). SAE QC computes and persists raw,
unquantized percentile points on a nominal 0-100 scale. Every boundary between
those two representations must pass through this module.
"""

from __future__ import annotations

import math
from numbers import Real

from steering.eval.axbench_base.types import AxBenchBaseError

PERCENTILE_POINTS_PER_FRACTION = 100.0


def percentile_fraction_to_points(value: object, *, field_name: str) -> float:
    """Validate a percentile fraction and return raw percentile points.

    Valid fractions are finite, non-boolean real numbers in the inclusive
    interval ``[0, 1]``. The endpoints are intentional: zero can disable a
    threshold while one provides an effectively unsatisfiable threshold for
    the current ``[0.5, 99.5]`` percentile lookup.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise AxBenchBaseError(
            f"{field_name} must be a finite real percentile fraction in [0, 1], "
            f"got {value!r} ({type(value).__name__})"
        )
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise AxBenchBaseError(
            f"{field_name} must be a finite real percentile fraction in [0, 1], "
            f"got {value!r}"
        )
    return fraction * PERCENTILE_POINTS_PER_FRACTION


def percentile_delta_meets_or_exceeds(
    delta_points: object,
    threshold_points: object,
) -> bool:
    """Return the canonical inclusive Stage 1b raw-point decision."""

    resolved: list[float] = []
    for field_name, value in (
        ("delta_points", delta_points),
        ("threshold_points", threshold_points),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise AxBenchBaseError(
                f"{field_name} must be a finite real number, got {value!r} "
                f"({type(value).__name__})"
            )
        as_float = float(value)
        if not math.isfinite(as_float):
            raise AxBenchBaseError(
                f"{field_name} must be a finite real number, got {value!r}"
            )
        resolved.append(as_float)
    return resolved[0] >= resolved[1]


__all__ = [
    "PERCENTILE_POINTS_PER_FRACTION",
    "percentile_delta_meets_or_exceeds",
    "percentile_fraction_to_points",
]
