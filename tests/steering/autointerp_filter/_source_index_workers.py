"""Picklable source-index workers used only by spawned-process tests."""

from __future__ import annotations

import time
from pathlib import Path

from steering.eval.autointerp_filter.source_index import (
    _scan_shard_worker as _production_scan_shard_worker,
)
from steering.eval.autointerp_filter.source_index import _ScanResult, _ScanTask

_REVERSE_ORDER_DELAY_SECONDS = 0.2


def completion_marker_path(temp_dir: Path, shard_index: int) -> Path:
    """Return the test-only completion marker for one spawned shard worker."""
    return temp_dir / f".source-index-test-worker-{shard_index}.complete"


def reverse_completion_order_worker(task: _ScanTask) -> _ScanResult:
    """Delay earlier shards so workers finish in reverse shard order."""
    time.sleep(
        _REVERSE_ORDER_DELAY_SECONDS
        * max(0, task.shard_count - task.shard_index)
    )
    result = _production_scan_shard_worker(task)
    completion_marker_path(task.temp_dir, task.shard_index).write_text(
        str(time.monotonic_ns()),
        encoding="utf-8",
    )
    return result


def fail_second_shard_worker(task: _ScanTask) -> _ScanResult:
    """Inject a failure after the first shard to exercise parent cleanup."""
    if task.shard_index == 1:
        raise RuntimeError("test-injected worker failure on shard index 1")
    return _production_scan_shard_worker(task)
