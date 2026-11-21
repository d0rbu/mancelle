"""log_rank_duration_skew: cross-rank checkpoint I/O skew reporting.

The NCCL watchdog kills on rank skew, not absolute duration — the 2026-08-28
hstor013 incidents produced exactly that (a 664s storage-starved step and
startup-phase skews past the 600s timeout) with the skew itself invisible
until the abort. These tests cover the single-rank fallback and the gathered
path via a monkeypatched process group.
"""

from io import StringIO

from loguru import logger

import steering.profiling.rank_skew as rank_skew_module
from steering.profiling.rank_skew import log_rank_duration_skew


def _capture():
    sink = StringIO()
    handler_id = logger.add(sink, format="{level} | {message}", level="DEBUG")
    return sink, handler_id


def test_single_rank_logs_solo_and_returns_degenerate_stats():
    sink, handler_id = _capture()
    try:
        stats = log_rank_duration_skew("checkpoint save (step 5)", 12.3, rank=0, world_size=1)
    finally:
        logger.remove(handler_id)
    assert stats == {
        "min_s": 12.3,
        "median_s": 12.3,
        "max_s": 12.3,
        "skew_s": 0.0,
        "slowest_rank": 0,
    }
    captured = sink.getvalue()
    assert "checkpoint save (step 5): 12.3s (single rank)" in captured


def test_uninitialized_process_group_falls_back_to_solo(monkeypatch):
    monkeypatch.setattr(rank_skew_module.dist, "is_initialized", lambda: False)
    sink, handler_id = _capture()
    try:
        stats = log_rank_duration_skew("checkpoint load", 3.0, rank=2, world_size=4)
    finally:
        logger.remove(handler_id)
    assert stats["skew_s"] == 0.0
    assert stats["slowest_rank"] == 2
    assert "(single rank)" in sink.getvalue()


def _fake_gather(monkeypatch, durations: list[float]) -> None:
    monkeypatch.setattr(rank_skew_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(rank_skew_module.dist, "is_initialized", lambda: True)

    def fake_all_gather_object(out, _value):
        out[:] = durations

    monkeypatch.setattr(rank_skew_module.dist, "all_gather_object", fake_all_gather_object)


def test_gathered_stats_and_rank0_log_line(monkeypatch):
    _fake_gather(monkeypatch, [0.5, 2.0, 100.0])
    sink, handler_id = _capture()
    try:
        stats = log_rank_duration_skew("checkpoint save (step 35000)", 0.5, rank=0, world_size=3)
    finally:
        logger.remove(handler_id)
    assert stats["min_s"] == 0.5
    assert stats["median_s"] == 2.0
    assert stats["max_s"] == 100.0
    assert stats["skew_s"] == 99.5
    assert stats["slowest_rank"] == 2
    captured = sink.getvalue()
    assert "rank duration min/median/max = 0.5/2.0/100.0s" in captured
    assert "slowest rank 2" in captured


def test_skew_above_threshold_warns(monkeypatch):
    _fake_gather(monkeypatch, [1.0, 700.0])
    sink, handler_id = _capture()
    try:
        stats = log_rank_duration_skew(
            "checkpoint save (step 40000)", 1.0, rank=0, world_size=2, skew_warn_s=300.0
        )
    finally:
        logger.remove(handler_id)
    assert stats["skew_s"] == 699.0
    captured = sink.getvalue()
    assert "WARNING" in captured
    assert "risks the NCCL watchdog" in captured


def test_nonzero_rank_stays_silent_but_computes_stats(monkeypatch):
    _fake_gather(monkeypatch, [1.0, 5.0])
    sink, handler_id = _capture()
    try:
        stats = log_rank_duration_skew("checkpoint load", 5.0, rank=1, world_size=2)
    finally:
        logger.remove(handler_id)
    assert stats["max_s"] == 5.0
    assert stats["slowest_rank"] == 1
    assert sink.getvalue() == ""
