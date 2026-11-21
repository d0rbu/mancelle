"""Tests for the steering.stats_schema histogram dataclass.

Run with:
    uv run pytest tests/test_stats_schema.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from steering.stats_schema import HistogramData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_histogram(**overrides) -> HistogramData:
    """Return a valid HistogramData, with optional field overrides."""
    defaults = {
        "bin_edges": [0.0, 1.0, 2.0, 3.0],
        "counts": [5, 10, 3],
    }
    defaults.update(overrides)
    return HistogramData(**defaults)


# ---------------------------------------------------------------------------
# HistogramData tests
# ---------------------------------------------------------------------------


class TestHistogramData:
    def test_frozen(self):
        h = _make_histogram()
        with pytest.raises(FrozenInstanceError):
            h.bin_edges = [0.0, 1.0]  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_validate_happy_path(self):
        h = _make_histogram()
        h.validate()  # should not raise

    def test_validate_single_bin(self):
        h = HistogramData(bin_edges=[0.0, 1.0], counts=[7])
        h.validate()

    def test_validate_empty_histogram(self):
        h = HistogramData(bin_edges=[0.0], counts=[])
        h.validate()

    def test_validate_equal_adjacent_edges(self):
        """Monotonically *non-decreasing* means equal neighbours are fine."""
        h = HistogramData(bin_edges=[0.0, 0.0, 1.0], counts=[2, 3])
        h.validate()

    def test_mismatched_lengths_too_few_edges(self):
        h = HistogramData(bin_edges=[0.0, 1.0], counts=[1, 2, 3])
        with pytest.raises(ValueError, match="len\\(bin_edges\\)"):
            h.validate()

    def test_mismatched_lengths_too_many_edges(self):
        h = HistogramData(bin_edges=[0.0, 1.0, 2.0, 3.0, 4.0], counts=[1])
        with pytest.raises(ValueError, match="len\\(bin_edges\\)"):
            h.validate()

    def test_non_monotonic_bins(self):
        h = HistogramData(bin_edges=[0.0, 2.0, 1.0, 3.0], counts=[1, 2, 3])
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            h.validate()

    def test_negative_counts(self):
        h = HistogramData(bin_edges=[0.0, 1.0, 2.0], counts=[5, -1])
        with pytest.raises(ValueError, match="non-negative"):
            h.validate()

    def test_zero_counts_are_fine(self):
        h = HistogramData(bin_edges=[0.0, 1.0, 2.0], counts=[0, 0])
        h.validate()
