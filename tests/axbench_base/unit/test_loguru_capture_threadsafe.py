"""Thread-safety tests for ``capture_loguru_info`` (PR-1).

The PR-1 review flagged that ``capture_loguru_info`` uses a StringIO
sink which has no explicit mutex; concurrent emission from multiple
threads could in principle race. While loguru itself serializes
handler writes internally, the helper now wraps the buffer in an
explicit lock so the contract is visible to readers. These tests
verify that under concurrent emission no log lines are dropped or
truncated.
"""

from __future__ import annotations

import threading

from loguru import logger

from tests.axbench_base._loguru_capture import capture_loguru_info


def test_concurrent_emission_preserves_all_lines():
    """4 worker threads each emit 100 unique log lines simultaneously.
    The captured list must contain all 400 unique messages, none of
    them dropped or interleaved into garbage strings."""
    n_threads = 4
    n_per_thread = 100
    barrier = threading.Barrier(n_threads)
    expected: set[str] = set()
    for tid in range(n_threads):
        for i in range(n_per_thread):
            expected.add(f"thread-{tid}-line-{i}")

    def _worker(tid: int) -> None:
        barrier.wait()
        for i in range(n_per_thread):
            logger.info(f"thread-{tid}-line-{i}")

    with capture_loguru_info() as captured:
        threads = [
            threading.Thread(target=_worker, args=(tid,))
            for tid in range(n_threads)
        ]
        for tA in threads:
            tA.start()
        for tA in threads:
            tA.join()

    actual = set(captured)
    missing = expected - actual
    extras = actual - expected
    assert not missing, (
        f"{len(missing)} log lines dropped under concurrent emission; "
        f"first few: {sorted(missing)[:5]}"
    )
    assert not extras, (
        f"unexpected garbage lines in captured output: {sorted(extras)[:5]}"
    )
    assert len(captured) == n_threads * n_per_thread


def test_concurrent_emission_lines_are_intact():
    """No captured line is a partial concatenation of two emissions
    (would happen under a true race even if no drops)."""
    n_threads = 4
    n_per_thread = 50
    barrier = threading.Barrier(n_threads)

    def _worker(tid: int) -> None:
        barrier.wait()
        for i in range(n_per_thread):
            logger.info(f"thread-{tid}-line-{i}")

    with capture_loguru_info() as captured:
        threads = [
            threading.Thread(target=_worker, args=(tid,))
            for tid in range(n_threads)
        ]
        for tA in threads:
            tA.start()
        for tA in threads:
            tA.join()

    for line in captured:
        # Every captured line must match one specific emission template.
        # If a race caused two emissions to interleave, the line would
        # contain "thread-X-line-Y" twice or have a torn middle.
        assert line.count("thread-") == 1, (
            f"torn / interleaved log line under concurrency: {line!r}"
        )
        assert line.count("line-") == 1
