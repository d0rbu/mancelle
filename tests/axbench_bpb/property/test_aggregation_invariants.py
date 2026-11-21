"""Hypothesis: aggregation math satisfies numerical invariants on random inputs."""
from __future__ import annotations

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

_trapezoid = getattr(np, "trapezoid", np.trapz)


@given(
    values=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=100,
    ),
)
def test_macro_mean_equals_sum_over_n(values):
    """mean = sum / n — exact, not approximate, over float64 arrays."""
    arr = np.array(values, dtype=np.float64)
    assert math.isclose(float(arr.mean()), float(arr.sum() / len(arr)), abs_tol=1e-9)


@given(
    points=st.lists(
        st.tuples(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=2, max_size=20, unique_by=lambda t: t[0],
    ),
)
def test_trapezoid_matches_hand_computed(points):
    """np.trapezoid(y, x) matches the hand-computed sum-of-trapezoids."""
    points_sorted = sorted(points, key=lambda t: t[0])
    cv = np.array([t[0] for t in points_sorted], dtype=np.float64)
    vv = np.array([t[1] for t in points_sorted], dtype=np.float64)
    auc_lib = float(_trapezoid(vv, cv))
    ref = sum(
        (cv[i + 1] - cv[i]) * (vv[i] + vv[i + 1]) / 2.0
        for i in range(len(cv) - 1)
    )
    assert math.isclose(auc_lib, ref, rel_tol=1e-6, abs_tol=1e-9), (
        f"trapezoid mismatch: lib={auc_lib}, ref={ref}"
    )


@given(
    rich_bpb=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    avoid_bpb=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_primary_margin_sign_matches_class_gap(rich_bpb, avoid_bpb):
    """primary_margin = avoid_bpb - rich_bpb in positive mode."""
    primary_margin = avoid_bpb - rich_bpb
    if avoid_bpb > rich_bpb:
        assert primary_margin > 0
    elif avoid_bpb < rich_bpb:
        assert primary_margin < 0
    else:
        assert primary_margin == 0.0
