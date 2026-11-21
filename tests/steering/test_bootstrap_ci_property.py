"""Property-based invariants for bootstrap_ci.

The 95% CI should always contain the point estimate ``wins / total`` for
any non-degenerate (wins, losses, ties) distribution. Also pin the
empty-input behavior to ``(0.0, 0.0)``.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.eval.steering_eval import bootstrap_ci


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    wins=st.integers(min_value=0, max_value=200),
    losses=st.integers(min_value=0, max_value=200),
    ties=st.integers(min_value=0, max_value=200),
)
def test_ci_contains_point_estimate_when_nonempty(
    wins: int, losses: int, ties: int
) -> None:
    """For total > 0: lo <= point_estimate <= hi."""
    total = wins + losses + ties
    if total == 0:
        # Skip: empty case handled separately.
        return
    point_estimate = wins / total
    lo, hi = bootstrap_ci(wins, losses, ties, n_boot=200, seed=42)
    # Allow small slop because bootstrap is finite-sample.
    eps = 1e-6
    assert lo - eps <= point_estimate <= hi + eps


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    wins=st.integers(min_value=0, max_value=100),
    losses=st.integers(min_value=0, max_value=100),
    ties=st.integers(min_value=0, max_value=100),
)
def test_ci_lo_le_hi(wins: int, losses: int, ties: int) -> None:
    """The lower bound must not exceed the upper bound."""
    lo, hi = bootstrap_ci(wins, losses, ties, n_boot=200, seed=42)
    assert lo <= hi


@settings(
    derandomize=True,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    wins=st.integers(min_value=1, max_value=100),
    losses=st.integers(min_value=1, max_value=100),
    ties=st.integers(min_value=0, max_value=100),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_ci_in_unit_interval(
    wins: int, losses: int, ties: int, seed: int
) -> None:
    """Bootstrap of binary outcomes must produce CI in [0, 1]."""
    lo, hi = bootstrap_ci(wins, losses, ties, n_boot=200, seed=seed)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_empty_input_returns_zero_zero() -> None:
    """All zeros → ``(0.0, 0.0)`` (no bootstrap to do)."""
    lo, hi = bootstrap_ci(0, 0, 0)
    assert lo == 0.0
    assert hi == 0.0


def test_all_wins_ci_includes_one() -> None:
    """100% wins: the CI should be near (1.0, 1.0)."""
    lo, hi = bootstrap_ci(50, 0, 0, n_boot=200)
    assert lo == 1.0
    assert hi == 1.0
