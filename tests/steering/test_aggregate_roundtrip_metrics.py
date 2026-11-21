"""Pin the symmetric NaN-filtering contract for _aggregate_roundtrip_metrics.

Map B-13 surfaced an asymmetry: ``pct_val`` was filtered with
``math.isfinite`` but ``pre_val`` and ``post_val`` were passed through
directly to ``statistics.mean``. A single non-finite pre/post value would
poison the entire condition's mean (yielding NaN), silently breaking W&B
logging.

The fix applies the same finite-filter to all three paths. These tests
red-line the prior asymmetry plus the divide-by-zero guard at line 846
(``n_docs == 0`` → 0.0, not raise).
"""

from __future__ import annotations

import math

import pytest

from steering.eval.inline_steering_eval import (
    _aggregate_roundtrip_metrics,
    _condition_means,
)

# ---------------------------------------------------------------------------
# Symmetric NaN filtering — pre, post, and pct paths must filter identically
# ---------------------------------------------------------------------------


class TestSymmetricNanFiltering:
    def test_finite_pre_finite_post_finite_pct(self) -> None:
        """All-finite scores: every condition gets a non-zero mean."""
        scores = [
            ("real", 1.0, 2.0, 50.0),
            ("real", 3.0, 4.0, 70.0),
            ("random", 0.1, 0.1, 10.0),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=4, n_docs=2)
        assert m["steering/roundtrip_real_mean"] == pytest.approx(2.0)
        assert m["steering/roundtrip_real_mean_post_act"] == pytest.approx(3.0)
        assert m["steering/roundtrip_real_mean_pct"] == pytest.approx(60.0)
        assert m["steering/roundtrip_random_mean"] == pytest.approx(0.1)
        assert m["steering/roundtrip_n_samples"] == 3
        assert m["steering/roundtrip_n_gt_features_avg"] == pytest.approx(2.0)

    def test_nan_pre_filtered_does_not_poison_pre_mean(self) -> None:
        """RED-LINE: the prior bug let nan in pre_val poison the mean.

        After the fix, NaN values in pre/post must be dropped symmetrically
        with the pct path. This test demonstrates the asymmetry would have
        been a NaN in real_mean before the fix; after the fix it's a clean
        finite mean over the surviving values.
        """
        scores = [
            ("real", math.nan, 1.0, 50.0),  # nan pre poisons under old code
            ("real", 2.0, 3.0, 60.0),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=2, n_docs=1)
        # With the fix, the NaN pre_val is filtered before the mean.
        assert math.isfinite(m["steering/roundtrip_real_mean"])
        assert m["steering/roundtrip_real_mean"] == pytest.approx(2.0)

    def test_nan_post_filtered_symmetric(self) -> None:
        """post_val NaN must be filtered identically to pre_val."""
        scores = [
            ("real", 1.0, math.nan, 50.0),
            ("real", 3.0, 5.0, 60.0),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=2, n_docs=1)
        assert math.isfinite(m["steering/roundtrip_real_mean_post_act"])
        assert m["steering/roundtrip_real_mean_post_act"] == pytest.approx(5.0)

    def test_inf_pre_filtered(self) -> None:
        """+inf and -inf must be filtered (math.isfinite returns False)."""
        scores = [
            ("real", math.inf, 1.0, 50.0),
            ("real", -math.inf, 1.0, 50.0),
            ("real", 4.0, 1.0, 50.0),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=1, n_docs=1)
        assert m["steering/roundtrip_real_mean"] == pytest.approx(4.0)

    def test_all_nan_condition_returns_zero(self) -> None:
        """If every value for a condition is NaN, the condition mean is 0.0
        (not NaN). This preserves W&B-friendly behavior — see the
        ``_condition_means`` docstring for rationale."""
        scores = [("real", math.nan, math.nan, math.nan)]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=0, n_docs=0)
        assert m["steering/roundtrip_real_mean"] == 0.0
        assert m["steering/roundtrip_real_mean_post_act"] == 0.0
        assert m["steering/roundtrip_real_mean_pct"] == 0.0


# ---------------------------------------------------------------------------
# Divide-by-zero guard
# ---------------------------------------------------------------------------


class TestDivideByZero:
    def test_n_docs_zero_returns_zero_avg(self) -> None:
        """``n_docs=0`` must NOT raise; the avg goes to 0.0."""
        m = _aggregate_roundtrip_metrics([], total_gt_features=5, n_docs=0)
        assert m["steering/roundtrip_n_gt_features_avg"] == 0.0

    def test_empty_scores_yields_zero_means(self) -> None:
        m = _aggregate_roundtrip_metrics([], total_gt_features=0, n_docs=0)
        assert m["steering/roundtrip_real_mean"] == 0.0
        assert m["steering/roundtrip_random_mean"] == 0.0
        assert m["steering/roundtrip_n_samples"] == 0


# ---------------------------------------------------------------------------
# _condition_means standalone behavior
# ---------------------------------------------------------------------------


class TestConditionMeans:
    def test_missing_condition_returns_zero(self) -> None:
        real, rnd = _condition_means({"real": [1.0, 2.0]})
        assert real == pytest.approx(1.5)
        assert rnd == 0.0

    def test_empty_condition_returns_zero(self) -> None:
        """Empty list (post-filter) must NOT raise — return 0.0."""
        real, rnd = _condition_means({"real": [], "random": []})
        assert real == 0.0
        assert rnd == 0.0

    def test_both_conditions(self) -> None:
        real, rnd = _condition_means({"real": [10.0], "random": [1.0]})
        assert real == 10.0
        assert rnd == 1.0


# ---------------------------------------------------------------------------
# Specificity = real - random invariants
# ---------------------------------------------------------------------------


class TestSpecificityComputation:
    def test_specificity_post_act_is_difference(self) -> None:
        """``steering/roundtrip_specificity_post_act`` = post_real - post_random."""
        scores = [
            ("real", 0.0, 8.0, 80.0),
            ("random", 0.0, 2.0, 20.0),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=1, n_docs=1)
        assert m["steering/roundtrip_specificity_post_act"] == pytest.approx(6.0)
        assert m["steering/roundtrip_specificity_pct"] == pytest.approx(60.0)
