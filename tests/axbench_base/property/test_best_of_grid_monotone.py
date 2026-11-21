"""Property tests for ``best_of_grid`` reduction.

``best_of_grid(scores)`` is a thin wrapper around Python's ``max()``
that raises :class:`AxBenchBaseError` on empty input. Wrapping ``max``
still deserves a property-based guard: downstream aggregation relies
on four invariants that any ``max``-like reduction satisfies and that
a future, more sophisticated reduction (e.g., trimmed-mean) might
silently break:

1. Upper-bound of every element: ``best_of_grid(lst) >= max(lst)``.
2. Upper-bound of the mean: ``best_of_grid(lst) >= mean(lst)``.
3. Appending a larger element strictly raises the result
   (``best_of_grid([*lst, max(lst) + 1]) > best_of_grid(lst)``).
4. Appending a strictly-smaller-than-everything element leaves the
   result unchanged.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_base.aggregation import best_of_grid

_score_strategy = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=32,
    min_value=-1e6,
    max_value=1e6,
)

_scores_strategy = st.lists(_score_strategy, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Element upper-bound
# ---------------------------------------------------------------------------


@given(scores=_scores_strategy)
def test_best_of_grid_ge_any_element(scores: list[float]) -> None:
    """``best_of_grid(scores) >= x`` for every ``x in scores``."""
    result = best_of_grid(scores)
    for x in scores:
        assert result >= x


# ---------------------------------------------------------------------------
# Mean upper-bound (max >= mean for any non-empty list)
# ---------------------------------------------------------------------------


@given(scores=_scores_strategy)
def test_best_of_grid_ge_mean(scores: list[float]) -> None:
    """``best_of_grid(scores) >= sum(scores) / len(scores)``."""
    result = best_of_grid(scores)
    mean = sum(scores) / len(scores)
    assert result >= mean


# ---------------------------------------------------------------------------
# Appending a strictly-larger element raises the result
# ---------------------------------------------------------------------------


@given(scores=_scores_strategy)
def test_best_of_grid_monotone_appending_larger(scores: list[float]) -> None:
    """Appending ``max(scores) + 1`` strictly raises the best."""
    base = best_of_grid(scores)
    larger = best_of_grid([*scores, max(scores) + 1.0])
    assert larger > base


# ---------------------------------------------------------------------------
# Appending a strictly-smaller element leaves the result unchanged
# ---------------------------------------------------------------------------


@given(scores=_scores_strategy)
def test_best_of_grid_stable_appending_smaller(scores: list[float]) -> None:
    """Appending ``min(scores) - 1`` leaves the best unchanged."""
    base = best_of_grid(scores)
    extended = best_of_grid([*scores, min(scores) - 1.0])
    assert extended == base
