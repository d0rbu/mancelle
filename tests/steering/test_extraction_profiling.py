"""Tests for extraction pipeline profiling instrumentation.

Tests:
  1. GPUUtilizationMonitor captures samples and reports statistics

Run with:
    uv run pytest tests/test_extraction_profiling.py -v
"""

import time

import pytest
import torch as t

from steering.profiling.gpu_utilization import GPUUtilizationMonitor

# ---------------------------------------------------------------------------
# GPUUtilizationMonitor tests
# ---------------------------------------------------------------------------


class TestGPUUtilizationMonitor:
    def test_captures_samples_over_time(self):
        """Monitor captures multiple utilization samples during active period."""
        monitor = GPUUtilizationMonitor(poll_interval_ms=10)
        with monitor:
            # Simulate some work
            time.sleep(0.1)

        assert monitor.sample_count > 0, "Monitor should capture at least one sample"

    def test_reports_statistics(self):
        """Monitor reports mean, min, max, and count."""
        monitor = GPUUtilizationMonitor(poll_interval_ms=10)
        with monitor:
            time.sleep(0.1)

        stats = monitor.statistics()
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] > 0
        assert 0.0 <= stats["min"] <= stats["max"] <= 100.0
        assert 0.0 <= stats["mean"] <= 100.0

    def test_context_manager_stops_thread(self):
        """Thread stops when context manager exits."""
        monitor = GPUUtilizationMonitor(poll_interval_ms=10)
        with monitor:
            time.sleep(0.05)
            assert monitor._thread is not None
            assert monitor._thread.is_alive()

        # After exiting context, thread should stop
        time.sleep(0.05)
        assert not monitor._thread.is_alive()

    def test_no_samples_before_start(self):
        """Monitor has no samples before entering context."""
        monitor = GPUUtilizationMonitor(poll_interval_ms=10)
        assert monitor.sample_count == 0
        stats = monitor.statistics()
        assert stats["count"] == 0

    def test_statistics_without_cuda_returns_zeros(self):
        """On CPU-only systems, monitor still works but reports 0% utilization."""
        if t.cuda.is_available():
            pytest.skip("Test only valid on CPU-only systems")
        monitor = GPUUtilizationMonitor(poll_interval_ms=10)
        with monitor:
            time.sleep(0.05)
        stats = monitor.statistics()
        # On CPU-only, all samples should be 0
        assert stats["mean"] == 0.0

    def test_samples_are_floats_in_range(self):
        """All captured samples are floats in [0, 100]."""
        monitor = GPUUtilizationMonitor(poll_interval_ms=10)
        with monitor:
            time.sleep(0.1)
        for sample in monitor.samples:
            assert isinstance(sample, float)
            assert 0.0 <= sample <= 100.0

    def test_poll_interval_respected(self):
        """Samples are captured at approximately the specified interval."""
        monitor = GPUUtilizationMonitor(poll_interval_ms=50)
        with monitor:
            time.sleep(0.3)
        # With 50ms interval and 300ms duration, expect roughly 5-7 samples
        # Allow wide tolerance for thread scheduling
        assert 2 <= monitor.sample_count <= 20, (
            f"Expected ~6 samples at 50ms interval over 300ms, got {monitor.sample_count}"
        )

