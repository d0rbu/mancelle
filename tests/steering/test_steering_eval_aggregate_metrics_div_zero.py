"""Pin div-zero / empty-input behavior for steering_eval.compute_aggregate_metrics.

Map B-10: ``compute_aggregate_metrics`` divided by ``total_weight`` without
checking for zero. Two failure modes:

- Empty results list → ``ZeroDivisionError`` at runtime.
- All confidences are 0 (e.g. judge produced no usable verdicts) →
  ``ZeroDivisionError`` even with ``len(results) > 0``.

Fix: surface NaN-filled metrics rather than raising.
"""

from __future__ import annotations

import math

from steering.eval.steering_eval import (
    AggregateMetrics,
    FeatureResult,
    compute_aggregate_metrics,
)


class TestEmptyInput:
    def test_empty_results_returns_nan_metrics(self) -> None:
        out = compute_aggregate_metrics([])
        assert isinstance(out, AggregateMetrics)
        assert math.isnan(out.confidence_weighted_win_rate)
        assert math.isnan(out.confidence_weighted_monotonicity)
        assert math.isnan(out.fraction_monotonic)
        assert out.total_features == 0
        assert out.high_confidence_features == 0


class TestZeroConfidence:
    def test_all_zero_confidence_returns_nan_metrics(self) -> None:
        """A non-empty list of zero-confidence features must NOT raise."""
        results = [
            FeatureResult(
                feature_id=0,
                confidence=0,
                standard_win_rate=0.5,
                spearman_rho=0.7,
            ),
            FeatureResult(
                feature_id=1,
                confidence=0,
                standard_win_rate=0.4,
                spearman_rho=0.8,
            ),
        ]
        out = compute_aggregate_metrics(results)
        assert math.isnan(out.confidence_weighted_win_rate)
        assert math.isnan(out.confidence_weighted_monotonicity)
        assert math.isnan(out.fraction_monotonic)
        assert out.total_features == 2
        assert out.high_confidence_features == 0


class TestNonZeroConfidence:
    def test_single_high_confidence_feature(self) -> None:
        results = [
            FeatureResult(
                feature_id=0,
                confidence=4,
                standard_win_rate=0.75,
                spearman_rho=0.9,
            )
        ]
        out = compute_aggregate_metrics(results)
        assert out.confidence_weighted_win_rate == 0.75
        assert out.confidence_weighted_monotonicity == 0.9
        assert out.fraction_monotonic == 1.0
        assert out.total_features == 1
        assert out.high_confidence_features == 1

    def test_mix_low_and_high_confidence(self) -> None:
        """Low-confidence features still contribute to weighted_wr but only
        high-confidence (>= 3) appear in monotonicity stats."""
        results = [
            FeatureResult(0, 1, 0.5, 0.0),  # low conf — drops out of monotonicity
            FeatureResult(1, 4, 1.0, 0.9),  # high conf
        ]
        out = compute_aggregate_metrics(results)
        # weighted_wr = (1*0.5 + 4*1.0) / (1+4) = 4.5/5 = 0.9
        assert out.confidence_weighted_win_rate == 0.9
        # monotonicity only uses confidence>=3
        assert out.confidence_weighted_monotonicity == 0.9
        assert out.fraction_monotonic == 1.0
        assert out.high_confidence_features == 1
