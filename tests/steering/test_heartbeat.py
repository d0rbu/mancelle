"""Heartbeat: periodic liveness lines for long silent phases.

Motivated by the 2026-08-28 hstor013 storage incidents, where selection scans
and checkpoint reads ran 30-60 min with zero log output and liveness had to be
established from /proc on the compute nodes.
"""

import time

import pytest

from steering.profiling.heartbeat import Heartbeat


def test_fast_phase_emits_nothing():
    lines: list[str] = []
    with Heartbeat("fast phase", interval_s=0.2, emit=lines.append):
        time.sleep(0.01)
    assert lines == []


def test_slow_phase_emits_periodic_lines_with_label():
    lines: list[str] = []
    with Heartbeat("slow phase", interval_s=0.05, emit=lines.append):
        time.sleep(0.22)
    assert len(lines) >= 2
    for line in lines:
        assert line.startswith("slow phase: still running after ")


def test_tick_detail_appears_in_subsequent_beats():
    lines: list[str] = []
    with Heartbeat("scan", interval_s=0.05, emit=lines.append) as heartbeat:
        heartbeat.tick("3/10 shards scanned")
        time.sleep(0.12)
    assert lines, "expected at least one beat"
    assert all("(3/10 shards scanned)" in line for line in lines)


def test_exception_propagates_and_thread_stops():
    lines: list[str] = []
    heartbeat = Heartbeat("failing phase", interval_s=0.05, emit=lines.append)
    with pytest.raises(RuntimeError, match="phase exploded"):
        with heartbeat:
            raise RuntimeError("phase exploded")
    assert heartbeat._thread is not None
    heartbeat._thread.join(timeout=1.0)
    assert not heartbeat._thread.is_alive()


def test_thread_stops_after_clean_exit():
    with Heartbeat("clean phase", interval_s=0.05) as heartbeat:
        time.sleep(0.01)
    assert heartbeat._thread is not None
    assert not heartbeat._thread.is_alive()


def test_nonpositive_interval_rejected():
    with pytest.raises(ValueError, match="interval_s must be positive"):
        Heartbeat("bad", interval_s=0.0)
    with pytest.raises(ValueError, match="interval_s must be positive"):
        Heartbeat("bad", interval_s=-1.0)
