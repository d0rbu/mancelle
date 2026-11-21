"""Shared statistics helpers for steering eval.

Provides the single canonical :func:`compute_mean_stderr` implementation
shared across :mod:`steering.eval.feature_probe`,
:mod:`steering.eval.shuffled_conditioning_eval`, and
:mod:`scripts.eval.completion_steering_eval`. See manifest §1.4 for the
deduplication rationale (3 prior reimplementations differed only in their
empty/all-NaN policy).

Two thin wrappers preserve the legacy semantic seams:

- :func:`compute_mean_stderr_or_nan` — returns ``(NaN, NaN)`` on empty/all-NaN
  (used by ``shuffled_conditioning_eval``).
- :func:`compute_mean_stderr_or_zero` — returns ``(0.0, 0.0)`` on empty
  (used by ``scripts/eval/completion_steering_eval`` for plotting defaults).
"""

from __future__ import annotations

import math

from errors import ValidationError

__all__ = [
    "compute_mean_stderr",
    "compute_mean_stderr_or_nan",
    "compute_mean_stderr_or_zero",
]


def compute_mean_stderr(vals: list[float]) -> tuple[float, float]:
    """Compute mean and standard error for a list of floats, filtering NaN.

    Returns ``(mean, stderr)``. The standard error is the sample SEM:
    ``sqrt(sample_variance / n)`` for ``n >= 2`` (using Bessel's correction
    in the variance) and ``0.0`` for ``n == 1``.

    Args:
        vals: List of floats. NaN values are filtered before the computation.

    Returns:
        ``(mean, stderr)`` tuple computed over non-NaN values.

    Raises:
        ValidationError: If ``vals`` is empty or all entries are NaN. Callers
            that need a non-raising semantic should use
            :func:`compute_mean_stderr_or_nan` (returns NaN, NaN) or
            :func:`compute_mean_stderr_or_zero` (returns 0, 0).
    """
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        raise ValidationError(
            "compute_mean_stderr: all values are NaN or list is empty"
        )
    n = len(clean)
    m = sum(clean) / n
    if n >= 2:
        var = sum((v - m) ** 2 for v in clean) / (n - 1)
        se = (var / n) ** 0.5
    else:
        se = 0.0
    return m, se


def compute_mean_stderr_or_nan(vals: list[float]) -> tuple[float, float]:
    """Mean/stderr returning ``(NaN, NaN)`` on empty/all-NaN.

    Thin wrapper around :func:`compute_mean_stderr` that swallows
    ``ValidationError`` and returns ``(float('nan'), float('nan'))`` —
    the historical semantic of ``shuffled_conditioning_eval._mean_stderr``.
    """
    try:
        return compute_mean_stderr(vals)
    except ValidationError:
        return float("nan"), float("nan")


def compute_mean_stderr_or_zero(vals: list[float]) -> tuple[float, float]:
    """Mean/stderr returning ``(0.0, 0.0)`` on empty input.

    Thin wrapper around :func:`compute_mean_stderr` that returns
    ``(0.0, 0.0)`` for empty/all-NaN inputs — the historical semantic of
    ``scripts/eval/completion_steering_eval._compute_mean_stderr``. Plotting
    code uses this default to keep zero-baseline bars rendered.
    """
    try:
        return compute_mean_stderr(vals)
    except ValidationError:
        return 0.0, 0.0
