"""Unit tests for the small pure-arithmetic helper extracted from
``scripts.preprocess.tok_eval`` for testability (X2 zero-coverage gap).

The original module-load script runs the full evaluation as soon as the
module is imported (it requires real datasets). We extract the
compression-ratio / relative-diff math into ``compute_relative_diff``
so we can test it in isolation.
"""

from __future__ import annotations

import math

from scripts.preprocess.tok_eval import compute_relative_diff, decide_better_label


def test_relative_diff_zero_when_equal():
    assert compute_relative_diff(baseline_tokens=100, ours_tokens=100) == 0.0


def test_relative_diff_positive_when_ours_smaller():
    # baseline=100 ours=80 -> 20% better.
    assert math.isclose(
        compute_relative_diff(baseline_tokens=100, ours_tokens=80), 20.0
    )


def test_relative_diff_negative_when_ours_larger():
    # baseline=100 ours=120 -> -20% worse.
    assert math.isclose(
        compute_relative_diff(baseline_tokens=100, ours_tokens=120), -20.0
    )


def test_relative_diff_zero_baseline_returns_inf_or_nan():
    """We don't pin the exact value but ensure no crash on division by zero."""
    res = compute_relative_diff(baseline_tokens=0, ours_tokens=10)
    # Either NaN, inf, or 0.0 is acceptable; we just ensure it returns a float.
    assert isinstance(res, float)


def test_decide_better_label_baseline_wins():
    # baseline ratio 4.0 > ours ratio 3.0 -> "GPT-2"
    assert decide_better_label("GPT-2", baseline_ratio=4.0, ours_ratio=3.0) == "GPT-2"


def test_decide_better_label_ours_wins():
    assert decide_better_label("GPT-2", baseline_ratio=3.0, ours_ratio=4.0) == "Ours"


def test_decide_better_label_tie():
    assert decide_better_label("GPT-2", baseline_ratio=3.5, ours_ratio=3.5) == "Tie"
