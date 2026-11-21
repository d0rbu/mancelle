"""Background GPU utilization monitor for profiling extraction pipelines.

Polls torch.cuda.utilization() at configurable intervals in a daemon thread
and records samples for statistical analysis. Useful for identifying GPU
idle gaps during feature extraction.

Usage:
    monitor = GPUUtilizationMonitor(poll_interval_ms=50)
    with monitor:
        process_shard(...)
    stats = monitor.statistics()
    # {'mean': 72.3, 'min': 0.0, 'max': 98.0, 'count': 120}
"""

import threading
import time

import torch as t
from loguru import logger


class GPUUtilizationMonitor:
    """Background thread that polls GPU utilization and records samples.

    Args:
        poll_interval_ms: Polling interval in milliseconds. Default 50.
        device_index: CUDA device index to monitor. Default 0.
    """

    def __init__(self, poll_interval_ms: int = 50, device_index: int = 0):
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._device_index = device_index
        self._samples: list[float] = []
        self._timestamped_samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._samples.clear()
        self._timestamped_samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return False

    def _poll_loop(self):
        """Polling loop that runs in background thread."""
        while not self._stop_event.is_set():
            utilization = self._read_utilization()
            self._samples.append(utilization)
            self._timestamped_samples.append((time.perf_counter(), utilization))
            self._stop_event.wait(self._poll_interval_s)

    def _read_utilization(self) -> float:
        """Read GPU utilization percentage. Returns 0.0 if CUDA unavailable."""
        if not t.cuda.is_available():
            return 0.0
        try:
            return float(t.cuda.utilization(self._device_index))
        except (RuntimeError, AttributeError, ValueError) as e:
            logger.debug(f"GPU utilization read failed: {e}")
            return 0.0

    @property
    def samples(self) -> list[float]:
        """Return a copy of all captured samples."""
        return list(self._samples)

    @property
    def timestamped_samples(self) -> list[tuple[float, float]]:
        """Return ``(perf_counter_s, utilization_percent)`` samples."""
        return list(self._timestamped_samples)

    @property
    def sample_count(self) -> int:
        """Number of samples captured so far."""
        return len(self._samples)

    def statistics(self) -> dict[str, float]:
        """Return summary statistics of captured samples.

        Returns:
            Dict with keys: mean, min, max, count.
            All zeros if no samples captured.
        """
        if not self._samples:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": sum(self._samples) / len(self._samples),
            "min": min(self._samples),
            "max": max(self._samples),
            "count": len(self._samples),
        }
