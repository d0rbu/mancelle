import json
import time

from steering.profiling.extraction import (
    ExtractionProfile,
    _idle_fraction,
    _longest_idle_run_s,
    extraction_profiler,
)
from steering.profiling.markers import TimingRecorder, timed_region


def test_disabled_yields_none_and_writes_nothing(tmp_path):
    sidecar = tmp_path / "prof"
    with extraction_profiler(enabled=False, rank=0, local_rank=0, sidecar_path=sidecar) as p:
        assert p is None
        with timed_region("compute"):  # free-when-off: body runs, no recorder
            pass
    assert not (tmp_path / "prof.json").exists()
    assert not (tmp_path / "prof.csv").exists()


def test_enabled_writes_sidecar_with_expected_fields(tmp_path):
    sidecar = tmp_path / "prof"
    with extraction_profiler(enabled=True, rank=3, local_rank=1, sidecar_path=sidecar) as p:
        assert p is not None
        with timed_region("compute", sync_cuda=False):
            pass
        p.mark_shard_done()
        p.mark_shard_done()
    summary = json.loads((tmp_path / "prof.json").read_text())
    for key in ("gpu_idle_fraction", "gpu_longest_idle_s", "gpu_util_mean",
                "shards_completed", "shards_per_hour", "wall_seconds", "rank"):
        assert key in summary
    assert summary["rank"] == 3
    assert summary["shards_completed"] == 2
    assert (tmp_path / "prof.csv").exists()


def test_enabled_local_rank0_drives_monitor(tmp_path):
    sidecar = tmp_path / "prof"
    with extraction_profiler(enabled=True, rank=0, local_rank=0,
                             sidecar_path=sidecar, poll_interval_ms=5) as p:
        assert p is not None
        time.sleep(0.03)  # let the daemon monitor thread poll at least once
        p.mark_shard_done()
    summary = json.loads((tmp_path / "prof.json").read_text())
    assert summary["gpu_sample_count"] >= 1            # monitor actually ran + samples flowed to summary
    assert 0.0 <= summary["gpu_idle_fraction"] <= 1.0  # computed from real samples (CPU util reads 0.0 -> ~1.0)
    assert summary["gpu_util_mean"] >= 0.0
    assert summary["shards_completed"] == 1
    assert (tmp_path / "prof.csv").exists()


def test_sidecar_name_with_dot_is_preserved(tmp_path):
    """A profile_output whose final path component contains a dot must keep its
    full name (incl. the per-rank suffix). with_suffix() replaces everything after
    the last dot, which silently collapsed all ranks onto one file. (Finding #7.)"""
    sidecar = tmp_path / "extraction.profile_rank0"  # dotted stem + rank suffix
    with extraction_profiler(enabled=True, rank=0, local_rank=1, sidecar_path=sidecar) as p:
        assert p is not None
        p.mark_shard_done()
    assert (tmp_path / "extraction.profile_rank0.json").exists()
    assert (tmp_path / "extraction.profile_rank0.csv").exists()
    assert not (tmp_path / "extraction.json").exists()  # the old with_suffix collision
    assert not (tmp_path / "extraction.csv").exists()


def test_idle_fraction_helpers():
    s = [0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 90.0]
    assert _idle_fraction(s, 5.0) == 5 / 7
    assert _longest_idle_run_s(s, 5.0, 0.05) == 3 * 0.05  # last run of 3 zeros
    assert _idle_fraction([], 5.0) == 0.0
    assert _longest_idle_run_s([], 5.0, 0.05) == 0.0          # empty
    assert _longest_idle_run_s([0.0, 0.0, 0.0], 5.0, 0.05) == 3 * 0.05  # all idle


def test_summary_no_monitor_yields_zeroed_gpu_fields():
    """Non-rank-0 ranks run with monitor=None: every GPU field must be 0 and the
    shards/hr math must still be correct (the monitor-None summary branch)."""
    p = ExtractionProfile(TimingRecorder(), None, 0.05, 5.0)
    p.mark_shard_done()
    s = p.summary(wall_seconds=10.0)
    assert s["gpu_sample_count"] == 0
    assert s["gpu_util_mean"] == 0.0
    assert s["gpu_idle_fraction"] == 0.0
    assert s["gpu_longest_idle_s"] == 0.0
    assert s["shards_completed"] == 1
    assert s["shards_per_hour"] == 1 / 10.0 * 3600.0


def test_summary_zero_wall_seconds_guards_division():
    p = ExtractionProfile(TimingRecorder(), None, 0.05, 5.0)
    s = p.summary(wall_seconds=0.0)
    assert s["shards_per_hour"] == 0.0  # guarded, not ZeroDivisionError
