"""Unit tests for timed_region + use_recorder in steering.profiling.markers."""

from __future__ import annotations

import time

from steering.profiling.markers import (
    TimingRecorder,
    timed_region,
    use_recorder,
)


def test_noop_when_no_recorder_active():
    # Without use_recorder, timed_region must not crash.
    with timed_region("noop"):
        pass


def test_records_elapsed_ns_when_active():
    r = TimingRecorder()
    with use_recorder(r):
        with timed_region("sleep", sync_cuda=False):
            time.sleep(0.02)
    rows = r.rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "sleep"
    # 20 ms sleep; allow 10–120 ms window to tolerate scheduling jitter.
    assert 10_000_000 < rows[0]["elapsed_ns"] < 120_000_000


def test_nested_regions_stack():
    r = TimingRecorder()
    with use_recorder(r):
        with timed_region("outer", sync_cuda=False):
            with timed_region("inner", sync_cuda=False):
                time.sleep(0.005)
    names = [row["name"] for row in r.rows()]
    # inner closes first (exit order = LIFO)
    assert names == ["inner", "outer"]


def test_use_recorder_isolates_scope():
    r1 = TimingRecorder()
    r2 = TimingRecorder()
    with use_recorder(r1):
        with timed_region("a", sync_cuda=False):
            pass
    with use_recorder(r2):
        with timed_region("b", sync_cuda=False):
            pass
    assert [row["name"] for row in r1.rows()] == ["a"]
    assert [row["name"] for row in r2.rows()] == ["b"]


def test_extra_fields_pass_through():
    r = TimingRecorder()
    with use_recorder(r):
        with timed_region("x", sync_cuda=False, extra={"iter": 3, "path": "B"}):
            pass
    row = r.rows()[0]
    assert row["iter"] == 3
    assert row["path"] == "B"


def test_sync_cuda_true_is_safe_without_cuda():
    # sync_cuda=True must not crash when torch.cuda is unavailable (CPU-only CI).
    r = TimingRecorder()
    with use_recorder(r):
        with timed_region("needs_sync", sync_cuda=True):
            pass
    assert len(r.rows()) == 1


def test_exception_inside_region_still_propagates():
    r = TimingRecorder()
    try:
        with use_recorder(r):
            with timed_region("boom", sync_cuda=False):
                raise RuntimeError("intentional")
    except RuntimeError as exc:
        assert str(exc) == "intentional"
    else:  # pragma: no cover
        raise AssertionError("exception should have propagated")
    # Still recorded the region before the exception bubbled up.
    assert r.rows()[0]["name"] == "boom"
