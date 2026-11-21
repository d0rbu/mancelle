"""Property tests for the ``hmean`` aggregation helper.

The ``hmean(a, b)`` helper implements AxBench's zero-bottom harmonic
mean: ``2ab/(a+b)`` when both inputs are positive, ``0`` otherwise.
The public signature takes ``int`` in ``[0, 2]`` (the 0/1/2 judge
scale) but the body is robust to any non-negative int.

Four invariants must hold across the full ``[0, 2]`` input lattice:

1. Zero-bottom: ``hmean(a, b) == 0 iff min(a, b) == 0``.
2. Symmetric: ``hmean(a, b) == hmean(b, a)``.
3. Bounded: ``0 <= hmean(a, b) <= max(a, b)``.
4. Monotonic in a single axis: fixing ``b`` and increasing ``a`` never
   DECREASES the output (strictly: for a in [0, 2], the function is
   non-decreasing).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_base.aggregation import hmean

_judge_axis_strategy = st.integers(min_value=0, max_value=2)


# ---------------------------------------------------------------------------
# Zero-bottom
# ---------------------------------------------------------------------------


@given(a=_judge_axis_strategy, b=_judge_axis_strategy)
def test_hmean_zero_bottom(a: int, b: int) -> None:
    """``hmean(a, b) == 0`` iff ``min(a, b) == 0``."""
    result = hmean(a, b)
    if min(a, b) == 0:
        assert result == 0.0
    else:
        assert result > 0.0


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


@given(a=_judge_axis_strategy, b=_judge_axis_strategy)
def test_hmean_symmetric(a: int, b: int) -> None:
    """``hmean(a, b) == hmean(b, a)`` for all axis values."""
    assert hmean(a, b) == hmean(b, a)


# ---------------------------------------------------------------------------
# Bounded by max(a, b)
# ---------------------------------------------------------------------------


@given(a=_judge_axis_strategy, b=_judge_axis_strategy)
def test_hmean_bounded_by_max(a: int, b: int) -> None:
    """``0 <= hmean(a, b) <= max(a, b)``.

    The harmonic mean is always bounded above by BOTH arguments
    (indeed, by ``min(a, b)`` when both are positive), so a fortiori
    it is bounded above by the max.
    """
    result = hmean(a, b)
    assert 0.0 <= result
    assert result <= float(max(a, b))


# ---------------------------------------------------------------------------
# Monotonic in a single axis
# ---------------------------------------------------------------------------


@given(b=_judge_axis_strategy)
def test_hmean_monotone_in_a(b: int) -> None:
    """Increasing ``a`` with ``b`` fixed never decreases the output.

    This is a weak-monotonicity property: ``hmean(a, b) <= hmean(a + 1, b)``
    for ``a in {0, 1}`` and ``b in {0, 1, 2}``. The zero-bottom jump
    from ``hmean(0, b) = 0`` to ``hmean(1, b) > 0`` (for ``b > 0``)
    is the largest increase; past that, the harmonic mean grows
    monotonically in each argument.
    """
    for a in (0, 1):
        assert hmean(a, b) <= hmean(a + 1, b)


@given(a=_judge_axis_strategy)
def test_hmean_monotone_in_b(a: int) -> None:
    """Symmetric counterpart: monotone in ``b`` with ``a`` fixed."""
    for b in (0, 1):
        assert hmean(a, b) <= hmean(a, b + 1)
