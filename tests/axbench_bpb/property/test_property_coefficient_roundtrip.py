"""Slice 17 audit (2026-05-05) coefficient roundtrip property test.

The leaderboard's ``coefficient_str`` column is the SQLite primary-key
component for Stage 3 BPB rows. ``str(c)`` → ``_coefficient_from_str(s)``
must round-trip cleanly for the values produced by the live coefficient
grids — None, ints, floats.
"""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str


def test_coefficient_roundtrip_none():
    """str(None) → 'None' → None."""
    assert _coefficient_from_str(str(None)) is None


@given(c=st.integers(min_value=-10**6, max_value=10**6))
@settings(max_examples=200, deadline=None)
def test_coefficient_roundtrip_int(c: int):
    """str(int) → int."""
    s = str(c)
    parsed = _coefficient_from_str(s)
    assert parsed == c
    assert isinstance(parsed, int)


@given(c=st.floats(
    allow_nan=False, allow_infinity=False,
    min_value=-1e9, max_value=1e9,
).filter(lambda x: x != int(x) or "." in str(x)))
@settings(max_examples=200, deadline=None)
def test_coefficient_roundtrip_float(c: float):
    """str(float) → float (close)."""
    s = str(c)
    parsed = _coefficient_from_str(s)
    assert isinstance(parsed, float)
    # Round-trip via str loses some precision; tolerance is 1e-9 relative
    # for non-zero values.
    if c == 0.0:
        assert parsed == 0.0
    else:
        assert math.isclose(parsed, c, rel_tol=1e-9, abs_tol=1e-12)


def test_coefficient_roundtrip_scientific():
    """Scientific notation parses as float."""
    parsed = _coefficient_from_str("1.5e2")
    assert isinstance(parsed, float)
    assert parsed == 150.0


def test_coefficient_roundtrip_negative_zero():
    """str(-0.0) round-trips through float."""
    parsed = _coefficient_from_str(str(-0.0))
    assert isinstance(parsed, float)
    assert parsed == 0.0
