"""Tests for steering module error handling (Tasks 13-17, 20).

Validates that silent returns (None, {}, []) have been replaced with
descriptive exceptions: DataLoadError, ValidationError, NotImplementedError.
"""

from __future__ import annotations

import pytest
import torch as t

from errors import DataLoadError, ValidationError
from steering.eval._stats_helpers import compute_mean_stderr

# ---------------------------------------------------------------------------
# Task 13: stats_io — _load_histogram / _load_percentile_dict
# ---------------------------------------------------------------------------


class TestLoadHistogram:
    """_load_histogram raises DataLoadError for missing keys."""

    def test_valid_load(self) -> None:
        from steering.stats_io import _load_histogram

        tensors = {
            "act_hist_edges": t.tensor([0.0, 1.0, 2.0], dtype=t.float64),
            "act_hist_counts": t.tensor([10, 20], dtype=t.int64),
        }
        result = _load_histogram(tensors, "act")
        assert "bin_edges" in result
        assert "counts" in result
        bin_edges = result["bin_edges"]
        counts = result["counts"]
        assert isinstance(bin_edges, list)
        assert isinstance(counts, list)
        assert len(bin_edges) == 3
        assert len(counts) == 2

    def test_missing_edges_raises(self) -> None:
        from steering.stats_io import _load_histogram

        tensors = {"act_hist_counts": t.tensor([10, 20], dtype=t.int64)}
        with pytest.raises(DataLoadError, match="edges"):
            _load_histogram(tensors, "act")

    def test_missing_counts_raises(self) -> None:
        from steering.stats_io import _load_histogram

        tensors = {"act_hist_edges": t.tensor([0.0, 1.0], dtype=t.float64)}
        with pytest.raises(DataLoadError, match="counts"):
            _load_histogram(tensors, "act")


class TestLoadPercentileDict:
    """_load_percentile_dict raises DataLoadError for missing keys."""

    def test_valid_load(self) -> None:
        from steering.stats_io import _load_percentile_dict

        tensors = {"act_values": t.tensor([0.5, 0.9], dtype=t.float64)}
        metadata = {"act_keys": "p50,p90"}
        result = _load_percentile_dict(tensors, metadata, "act")
        assert "p50" in result
        assert "p90" in result

    def test_missing_keys_metadata_raises(self) -> None:
        from steering.stats_io import _load_percentile_dict

        tensors = {"act_values": t.tensor([0.5], dtype=t.float64)}
        metadata: dict[str, str] = {}
        with pytest.raises(DataLoadError, match="keys metadata"):
            _load_percentile_dict(tensors, metadata, "act")

    def test_missing_values_tensor_raises(self) -> None:
        from steering.stats_io import _load_percentile_dict

        tensors: dict[str, t.Tensor] = {}
        metadata = {"act_keys": "p50"}
        with pytest.raises(DataLoadError, match="values"):
            _load_percentile_dict(tensors, metadata, "act")


# ---------------------------------------------------------------------------
# Task 14: compute_statistics — compute_segment_lengths
# ---------------------------------------------------------------------------


class TestComputeSegmentLengths:
    """compute_segment_lengths must raise ValidationError on invalid config."""

    def test_invalid_segment_mode_raises(self) -> None:
        from steering.compute_statistics import compute_segment_lengths

        shard_data = {"segment_mode": "sentences", "segment_size": 64}
        with pytest.raises(ValidationError, match="segment_mode"):
            compute_segment_lengths(shard_data)

    def test_non_positive_segment_size_raises(self) -> None:
        from steering.compute_statistics import compute_segment_lengths

        shard_data = {"segment_mode": "fixed_tokens", "segment_size": 0}
        with pytest.raises(ValidationError, match="segment_size"):
            compute_segment_lengths(shard_data)

    def test_negative_segment_size_raises(self) -> None:
        from steering.compute_statistics import compute_segment_lengths

        shard_data = {"segment_mode": "fixed_tokens", "segment_size": -1}
        with pytest.raises(ValidationError, match="segment_size"):
            compute_segment_lengths(shard_data)


# ---------------------------------------------------------------------------
# Task 15: thresholding — extract_threshold_percentile
# ---------------------------------------------------------------------------


class TestExtractThresholdPercentile:
    """extract_threshold_percentile must raise NotImplementedError for unhandled types."""

    def test_feature_percentile_works(self) -> None:
        from steering.conditional.thresholding import (
            ThresholdConfig,
            ThresholdType,
            extract_threshold_percentile,
        )

        cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.9)
        assert extract_threshold_percentile(cfg) == 0.9

    def test_combined_threshold_works(self) -> None:
        from steering.conditional.thresholding import (
            ThresholdConfig,
            ThresholdType,
            extract_threshold_percentile,
        )

        cfg = ThresholdConfig(ThresholdType.feature_percentile_segment_topk, 2048.9)
        assert extract_threshold_percentile(cfg) == pytest.approx(0.9)

    def test_segment_topk_raises(self) -> None:
        from steering.conditional.thresholding import (
            ThresholdConfig,
            ThresholdType,
            extract_threshold_percentile,
        )

        cfg = ThresholdConfig(ThresholdType.segment_topk, 10)
        with pytest.raises(NotImplementedError, match="not supported"):
            extract_threshold_percentile(cfg)

    def test_removed_feature_filter_topk_rejected(self) -> None:
        from steering.conditional.sampling import parse_feature_selection

        with pytest.raises(ValueError, match="feature_filter_topk"):
            parse_feature_selection(
                {"threshold_type": "feature_filter_topk", "threshold": 100}
            )


# ---------------------------------------------------------------------------
# Task 16: feature_probe — compute_mean_stderr
# ---------------------------------------------------------------------------


class TestComputeMeanStderr:
    """compute_mean_stderr must raise ValidationError on empty/all-NaN input."""

    def test_valid_input(self) -> None:
        m, se = compute_mean_stderr([1.0, 2.0, 3.0])
        assert m == pytest.approx(2.0)
        assert se > 0

    def test_single_value(self) -> None:
        m, se = compute_mean_stderr([5.0])
        assert m == pytest.approx(5.0)
        assert se == 0.0

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValidationError, match="NaN or list is empty"):
            compute_mean_stderr([float("nan"), float("nan")])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValidationError, match="NaN or list is empty"):
            compute_mean_stderr([])

    def test_mixed_nan_filters(self) -> None:
        m, se = compute_mean_stderr([1.0, float("nan"), 3.0])
        assert m == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Task 17: steering_eval — compute_spearman / load_raw_data
# ---------------------------------------------------------------------------


class TestComputeSpearman:
    """compute_spearman must raise ValidationError on constant data."""

    def test_valid_correlation(self) -> None:
        from steering.eval.steering_eval import compute_spearman

        levels = list(range(10))
        win_rates = [0.1 * (i + 1) for i in range(10)]
        rho, p = compute_spearman(levels, win_rates)
        assert abs(rho - 1.0) < 1e-9
        assert p < 0.05

    def test_constant_data_raises(self) -> None:
        from steering.eval.steering_eval import compute_spearman

        levels = list(range(5))
        win_rates = [0.5] * 5
        with pytest.raises(ValidationError, match="NaN"):
            compute_spearman(levels, win_rates)


class TestLoadRawData:
    """load_raw_data must distinguish missing file from corrupted data."""

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        from steering.eval.steering_eval import load_raw_data

        result = load_raw_data(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_corrupted_yaml_raises(self, tmp_path) -> None:
        from steering.eval.steering_eval import load_raw_data

        p = tmp_path / "corrupted.yaml"
        p.write_text("- this is a list not a dict\n")
        with pytest.raises(DataLoadError, match="expected dict"):
            load_raw_data(p)

    def test_missing_features_key_raises(self, tmp_path) -> None:
        from steering.eval.steering_eval import load_raw_data

        p = tmp_path / "no_features.yaml"
        p.write_text("some_key: some_value\n")
        with pytest.raises(DataLoadError, match="missing expected key"):
            load_raw_data(p)

    def test_none_content_raises(self, tmp_path) -> None:
        from steering.eval.steering_eval import load_raw_data

        p = tmp_path / "empty.yaml"
        p.write_text("")  # yaml.safe_load returns None
        with pytest.raises(DataLoadError, match="expected dict"):
            load_raw_data(p)
