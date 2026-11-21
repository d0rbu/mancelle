"""Unit tests for AxBench-Base profiling instrumentation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
import yaml
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_base import profiler as profiler_mod
from steering.eval.axbench_base.profiler import (
    AxBenchProfiler,
    ProcessMemorySnapshot,
    ProfileConfig,
    get_current_profiler,
    process_memory_snapshot,
    profile_region,
    record_counter,
    resolve_profile_gpu_device_index,
)
from steering.eval.axbench_base.types import Stage


def _install_deferred_end_record_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events_created = 0

    class EndRecordFailureCudaEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            nonlocal events_created
            assert enable_timing is True
            self.index = events_created
            events_created += 1

        def record(self, _stream: str) -> None:
            if self.index == 1:
                raise RuntimeError("synthetic end-event record failure")

    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler_mod.t.cuda, "Event", EndRecordFailureCudaEvent)
    monkeypatch.setattr(profiler_mod.t.cuda, "current_stream", lambda _index: "stream")
    monkeypatch.setattr(profiler_mod, "_cuda_snapshot", lambda: None)


def test_parse_linux_process_status_reports_current_and_anonymous_total() -> None:
    snapshot = profiler_mod._parse_linux_process_status(
        "Name:\tpython\nVmHWM:\t  456 kB\nVmRSS:\t  123 kB\nRssAnon:\t  100 kB\nVmSwap:\t  7 kB"
    )

    assert snapshot == ProcessMemorySnapshot(
        rss_kb=123,
        rss_anon_kb=100,
        swap_kb=7,
        anonymous_total_kb=107,
        peak_rss_kb=456,
    )


@pytest.mark.parametrize(
    "status_text",
    [
        "VmRSS:\t1 kB\nRssAnon:\t1 kB\nVmSwap:\t0 kB",
        "VmRSS:\tbad kB\nRssAnon:\t1 kB\nVmSwap:\t0 kB\nVmHWM:\t2 kB",
        "VmRSS:\t1 MB\nRssAnon:\t1 kB\nVmSwap:\t0 kB\nVmHWM:\t2 kB",
        "VmRSS:\t-1 kB\nRssAnon:\t1 kB\nVmSwap:\t0 kB\nVmHWM:\t2 kB",
        ("VmRSS:\t1 kB\nVmRSS:\t2 kB\nRssAnon:\t1 kB\nVmSwap:\t0 kB\nVmHWM:\t2 kB"),
    ],
)
def test_parse_linux_process_status_rejects_missing_or_malformed_fields(
    status_text: str,
) -> None:
    assert profiler_mod._parse_linux_process_status(status_text) is None


def test_process_memory_snapshot_falls_back_after_malformed_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = ProcessMemorySnapshot(
        rss_kb=12,
        rss_anon_kb=None,
        swap_kb=3,
        anonymous_total_kb=None,
        peak_rss_kb=20,
    )
    monkeypatch.setattr(profiler_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        profiler_mod,
        "_read_linux_process_status",
        lambda: "VmRSS:\tmalformed",
    )
    monkeypatch.setattr(
        profiler_mod,
        "_fallback_process_memory_snapshot",
        lambda: fallback,
    )

    assert process_memory_snapshot() == fallback


def test_process_memory_snapshot_uses_valid_linux_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiler_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        profiler_mod,
        "_read_linux_process_status",
        lambda: "VmRSS:\t1 kB\nRssAnon:\t2 kB\nVmSwap:\t3 kB\nVmHWM:\t4 kB",
    )
    monkeypatch.setattr(
        profiler_mod,
        "_fallback_process_memory_snapshot",
        lambda: pytest.fail("valid procfs must not use the fallback"),
    )

    assert process_memory_snapshot() == ProcessMemorySnapshot(
        rss_kb=1,
        rss_anon_kb=2,
        swap_kb=3,
        anonymous_total_kb=5,
        peak_rss_kb=4,
    )


def test_fallback_process_memory_snapshot_uses_psutil_and_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=8 * 1024),
        memory_full_info=lambda: SimpleNamespace(swap=3 * 1024),
    )
    psutil = SimpleNamespace(Process=lambda: process)
    monkeypatch.setattr(
        profiler_mod.importlib,
        "import_module",
        lambda name: psutil if name == "psutil" else pytest.fail(name),
    )
    monkeypatch.setattr(profiler_mod, "_cpu_rss_kb", lambda: 12)

    assert profiler_mod._fallback_process_memory_snapshot() == (
        ProcessMemorySnapshot(
            rss_kb=8,
            rss_anon_kb=None,
            swap_kb=3,
            anonymous_total_kb=None,
            peak_rss_kb=12,
        )
    )


def test_fallback_process_memory_snapshot_omits_negative_or_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=-1),
        memory_full_info=lambda: SimpleNamespace(),
    )
    psutil = SimpleNamespace(Process=lambda: process)
    monkeypatch.setattr(
        profiler_mod.importlib,
        "import_module",
        lambda name: psutil if name == "psutil" else pytest.fail(name),
    )
    monkeypatch.setattr(profiler_mod, "_cpu_rss_kb", lambda: 12)

    assert profiler_mod._fallback_process_memory_snapshot() == (
        ProcessMemorySnapshot(
            rss_kb=None,
            rss_anon_kb=None,
            swap_kb=None,
            anonymous_total_kb=None,
            peak_rss_kb=12,
        )
    )


def test_fallback_process_memory_snapshot_tolerates_swap_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=8 * 1024),
        memory_full_info=lambda: (_ for _ in ()).throw(RuntimeError("swap unavailable")),
    )
    psutil = SimpleNamespace(Process=lambda: process)
    monkeypatch.setattr(
        profiler_mod.importlib,
        "import_module",
        lambda name: psutil if name == "psutil" else pytest.fail(name),
    )
    monkeypatch.setattr(profiler_mod, "_cpu_rss_kb", lambda: None)

    assert profiler_mod._fallback_process_memory_snapshot() == (
        ProcessMemorySnapshot(
            rss_kb=8,
            rss_anon_kb=None,
            swap_kb=None,
            anonymous_total_kb=None,
            peak_rss_kb=None,
        )
    )


def test_fallback_process_memory_snapshot_returns_none_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profiler_mod.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    monkeypatch.setattr(profiler_mod, "_cpu_rss_kb", lambda: None)

    assert profiler_mod._fallback_process_memory_snapshot() is None


def test_process_memory_snapshot_never_throws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiler_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        profiler_mod,
        "_read_linux_process_status",
        lambda: (_ for _ in ()).throw(OSError("no procfs")),
    )
    monkeypatch.setattr(
        profiler_mod,
        "_fallback_process_memory_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("no fallback")),
    )

    assert process_memory_snapshot() is None


def test_process_memory_snapshot_skips_procfs_off_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = ProcessMemorySnapshot(
        rss_kb=12,
        rss_anon_kb=None,
        swap_kb=None,
        anonymous_total_kb=None,
        peak_rss_kb=20,
    )
    monkeypatch.setattr(profiler_mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        profiler_mod,
        "_read_linux_process_status",
        lambda: pytest.fail("non-Linux platforms must not read procfs"),
    )
    monkeypatch.setattr(
        profiler_mod,
        "_fallback_process_memory_snapshot",
        lambda: fallback,
    )

    assert process_memory_snapshot() == fallback


def test_profile_region_disabled_is_noop(tmp_path: Path) -> None:
    out_dir = tmp_path / "profile"
    seen: list[str] = []

    assert get_current_profiler() is None
    with profile_region("disabled.region", rows=3):
        seen.append("ran")
    record_counter("disabled.counter", 7)

    assert seen == ["ran"]
    assert not out_dir.exists()
    assert get_current_profiler() is None


def test_enabled_profiler_writes_event_and_summary_schemas(tmp_path: Path) -> None:
    out_dir = tmp_path / "profile"

    with AxBenchProfiler(out_dir, metadata={"run_tag": "unit-test"}):
        with profile_region("outer", rows=2):
            record_counter("rows", 2, stage="x")
        with profile_region("outer", rows=3):
            pass

    events_path = out_dir / "events.jsonl"
    summary_yaml = out_dir / "summary.yaml"
    summary_csv = out_dir / "summary.csv"
    assert events_path.is_file()
    assert summary_yaml.is_file()
    assert summary_csv.is_file()

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "profiler_start"
    assert any(e["type"] == "region" and e["name"] == "outer" for e in events)
    assert any(e["type"] == "counter" and e["name"] == "rows" for e in events)
    assert events[-1]["type"] == "profiler_stop"

    summary = yaml.safe_load(summary_yaml.read_text(encoding="utf-8"))
    assert summary["metadata"]["run_tag"] == "unit-test"
    assert summary["status"] == "ok"
    outer = next(r for r in summary["regions"] if r["name"] == "outer")
    assert outer["count"] == 2
    assert outer["rows"] == 5
    assert "peak_cpu_rss_kb" in outer
    assert "gpu_util_mean" in outer
    assert "gpu_sample_count" in outer
    assert "gpu_utilization" in summary
    rows_counter = next(c for c in summary["counters"] if c["name"] == "rows")
    assert rows_counter["count"] == 1
    assert rows_counter["total"] == 2

    csv_text = summary_csv.read_text(encoding="utf-8")
    assert "kind,name,count,total_s" in csv_text
    assert "region,outer,2" in csv_text
    assert "counter,rows,1" in csv_text
    assert "peak_cpu_rss_kb" in csv_text
    assert "gpu_util_mean" in csv_text


def test_profile_config_validation() -> None:
    ProfileConfig()
    ProfileConfig(enabled=True, output_dir="profile")

    with pytest.raises(BeartypeCallHintParamViolation, match="gpu_poll_interval_ms"):
        ProfileConfig(gpu_poll_interval_ms=0)
    with pytest.raises(BeartypeCallHintParamViolation, match="gpu_idle_threshold"):
        ProfileConfig(gpu_idle_threshold=-1.0)
    with pytest.raises(BeartypeCallHintParamViolation, match="output_dir"):
        ProfileConfig(output_dir=" ")


def test_resolve_profile_gpu_device_index_honors_single_process_cuda_index() -> None:
    assert (
        resolve_profile_gpu_device_index(
            "cuda:1",
            ddp_local_rank=0,
            ddp_world_size=1,
        )
        == 1
    )
    assert (
        resolve_profile_gpu_device_index(
            "cuda",
            ddp_local_rank=0,
            ddp_world_size=1,
        )
        == 0
    )


def test_resolve_profile_gpu_device_index_prefers_ddp_local_rank() -> None:
    assert (
        resolve_profile_gpu_device_index(
            "cuda:1",
            ddp_local_rank=3,
            ddp_world_size=8,
        )
        == 3
    )


def test_profiler_gpu_monitor_summary_without_cuda_dependency(tmp_path: Path) -> None:
    out_dir = tmp_path / "profile"

    with AxBenchProfiler(
        out_dir,
        gpu_utilization=True,
        gpu_poll_interval_ms=1,
        synchronize_cuda=False,
    ):
        with profile_region("gpu.region"):
            pass

    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    gpu_summary = summary["gpu_utilization"]
    assert set(gpu_summary) == {
        "gpu_util_mean",
        "gpu_util_min",
        "gpu_util_max",
        "gpu_sample_count",
        "gpu_idle_fraction",
    }


def test_deferred_cuda_events_measure_only_requested_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "profile"
    created_events: list[FakeCudaEvent] = []
    recorded_streams: list[tuple[int, str]] = []
    synchronized_devices: list[int] = []

    class FakeCudaEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.index = len(created_events)
            created_events.append(self)

        def record(self, stream: str) -> None:
            recorded_streams.append((self.index, stream))

        def elapsed_time(self, end_event: FakeCudaEvent) -> float:
            assert end_event.index == self.index + 1
            return float(2 + self.index)

    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler_mod.t.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(
        profiler_mod.t.cuda,
        "current_stream",
        lambda device_index: f"stream-{device_index}",
    )
    monkeypatch.setattr(
        profiler_mod.t.cuda,
        "synchronize",
        lambda device_index: synchronized_devices.append(device_index),
    )
    monkeypatch.setattr(profiler_mod, "_cuda_snapshot", lambda: None)

    with AxBenchProfiler(
        out_dir,
        synchronize_cuda=False,
        gpu_device_index=3,
        cuda_event_regions=("selected.region",),
    ) as profiler:
        with profile_region("unselected.region"):
            pass
        with profile_region("selected.region", rows=2):
            assert len(created_events) == 2
            assert recorded_streams == [(0, "stream-3")]
        with profile_region("selected.region", rows=3):
            assert len(created_events) == 4
            assert recorded_streams == [
                (0, "stream-3"),
                (1, "stream-3"),
                (2, "stream-3"),
            ]
        profiler.require_cuda_event_regions_complete()

    assert len(created_events) == 4
    assert recorded_streams == [
        (0, "stream-3"),
        (1, "stream-3"),
        (2, "stream-3"),
        (3, "stream-3"),
    ]
    assert synchronized_devices == [3]

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cuda_summary_event = next(event for event in events if event["type"] == "cuda_event_timing")
    assert events[-2] == cuda_summary_event
    assert events[-1]["type"] == "profiler_stop"
    assert cuda_summary_event["device_index"] == 3
    assert cuda_summary_event["configured_regions"] == ["selected.region"]
    assert cuda_summary_event["missing_regions"] == []
    assert cuda_summary_event["event_count"] == 2
    assert cuda_summary_event["schema_version"] == 1
    assert cuda_summary_event["regions"] == [
        {
            "name": "selected.region",
            "count": 2,
            "errored_count": 0,
            "total_s": pytest.approx(0.006),
            "mean_s": pytest.approx(0.003),
            "min_s": pytest.approx(0.002),
            "max_s": pytest.approx(0.004),
        }
    ]

    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["cuda_event_timing"] == {
        key: value for key, value in cuda_summary_event.items() if key not in {"type", "time"}
    }
    selected = next(row for row in summary["regions"] if row["name"] == "selected.region")
    assert not any(key.startswith("cuda_event_") for key in selected)
    unselected = next(row for row in summary["regions"] if row["name"] == "unselected.region")
    assert not any(key.startswith("cuda_event_") for key in unselected)

    csv_text = (out_dir / "summary.csv").read_text(encoding="utf-8")
    assert "cuda_event_region,selected.region,2,0.006" in csv_text


def test_disabled_deferred_cuda_events_preserve_legacy_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profiler_mod.t.cuda,
        "Event",
        lambda **_kwargs: pytest.fail("disabled timing must not allocate CUDA events"),
    )
    monkeypatch.setattr(
        AxBenchProfiler,
        "_begin_deferred_cuda_timing",
        lambda *_args: pytest.fail("disabled timing must bypass deferred event setup"),
    )
    monkeypatch.setattr(
        AxBenchProfiler,
        "_finalize_deferred_cuda_timings",
        lambda *_args: pytest.fail("disabled timing must bypass deferred event finalization"),
    )

    out_dir = tmp_path / "profile"
    with AxBenchProfiler(out_dir, synchronize_cuda=False):
        with profile_region("ordinary.region"):
            pass

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not any(event["type"] == "cuda_event_timing" for event in events)
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert "cuda_event_timing" not in summary
    ordinary = next(row for row in summary["regions"] if row["name"] == "ordinary.region")
    assert not any(key.startswith("cuda_event_") for key in ordinary)
    csv_text = (out_dir / "summary.csv").read_text(encoding="utf-8")
    assert "cuda_event_region" not in csv_text


@pytest.mark.parametrize(
    "region_names, expected",
    [
        (("",), "nonempty"),
        ((" padded",), "surrounding whitespace"),
        (("same", "same"), "unique"),
    ],
)
def test_deferred_cuda_event_region_names_are_strict(
    tmp_path: Path,
    region_names: tuple[str, ...],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        AxBenchProfiler(tmp_path / "profile", cuda_event_regions=region_names)


def test_deferred_cuda_event_region_container_rejects_a_bare_string(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="sequence of exact region names"):
        AxBenchProfiler(tmp_path / "string", cuda_event_regions="one.region")


@pytest.mark.parametrize("device_index", [-1, True])
def test_deferred_cuda_event_device_index_is_strict(
    tmp_path: Path,
    device_index: int,
) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        AxBenchProfiler(
            tmp_path / "profile",
            gpu_device_index=device_index,
            cuda_event_regions=("region",),
        )


def test_deferred_cuda_events_require_cuda_and_unsynchronized_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="synchronize_cuda=False"):
        with AxBenchProfiler(
            tmp_path / "synchronized",
            synchronize_cuda=True,
            cuda_event_regions=("region",),
        ):
            pass
    assert not (tmp_path / "synchronized").exists()

    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        with AxBenchProfiler(
            tmp_path / "cpu",
            synchronize_cuda=False,
            cuda_event_regions=("region",),
        ):
            pass
    assert not (tmp_path / "cpu").exists()


def test_deferred_cuda_event_completion_fails_closed_with_error_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: True)
    out_dir = tmp_path / "profile"

    with pytest.raises(RuntimeError, match="'required.region' was not observed"):
        with AxBenchProfiler(
            out_dir,
            synchronize_cuda=False,
            cuda_event_regions=("required.region",),
        ) as profiler:
            with profile_region("different.region"):
                pass
            profiler.require_cuda_event_regions_complete()

    assert get_current_profiler() is None
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-2]["type"] == "cuda_event_timing"
    assert events[-2]["missing_regions"] == ["required.region"]
    assert events[-1]["type"] == "profiler_stop"
    assert events[-1]["attrs"]["status"] == "error"
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["cuda_event_timing"]["missing_regions"] == ["required.region"]


def test_deferred_cuda_event_completion_rejects_a_missing_event_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        AxBenchProfiler,
        "_begin_deferred_cuda_timing",
        lambda *_args: (object(), object(), object()),
    )
    monkeypatch.setattr(
        AxBenchProfiler,
        "_complete_deferred_cuda_timing",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(profiler_mod, "_cuda_snapshot", lambda: None)

    with pytest.raises(RuntimeError, match="0 CUDA event pairs for 1 region events"):
        with AxBenchProfiler(
            tmp_path / "profile",
            synchronize_cuda=False,
            cuda_event_regions=("required.region",),
        ) as profiler:
            with profile_region("required.region"):
                pass
            profiler.require_cuda_event_regions_complete()


def test_invalid_deferred_cuda_duration_fails_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidCudaEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self, _stream: str) -> None:
            pass

        def elapsed_time(self, _end_event: InvalidCudaEvent) -> float:
            return float("nan")

    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler_mod.t.cuda, "Event", InvalidCudaEvent)
    monkeypatch.setattr(profiler_mod.t.cuda, "current_stream", lambda _index: "stream")
    monkeypatch.setattr(profiler_mod.t.cuda, "synchronize", lambda _index: None)
    monkeypatch.setattr(profiler_mod, "_cuda_snapshot", lambda: None)

    out_dir = tmp_path / "profile"
    profiler = AxBenchProfiler(
        out_dir,
        synchronize_cuda=False,
        cuda_event_regions=("measured.region",),
    )
    with pytest.raises(RuntimeError, match="nonfinite or negative"):
        with profiler:
            with profile_region("measured.region"):
                pass

    assert get_current_profiler() is None
    assert profiler._events_fp is None
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["type"] == "profiler_start"
    assert events[-1]["type"] == "region"


def test_body_error_survives_deferred_cuda_finalization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidCudaEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self, _stream: str) -> None:
            pass

        def elapsed_time(self, _end_event: InvalidCudaEvent) -> float:
            return -1.0

    monkeypatch.setattr(profiler_mod.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler_mod.t.cuda, "Event", InvalidCudaEvent)
    monkeypatch.setattr(profiler_mod.t.cuda, "current_stream", lambda _index: "stream")
    monkeypatch.setattr(profiler_mod.t.cuda, "synchronize", lambda _index: None)
    monkeypatch.setattr(profiler_mod, "_cuda_snapshot", lambda: None)

    profiler = AxBenchProfiler(
        tmp_path / "profile",
        synchronize_cuda=False,
        cuda_event_regions=("measured.region",),
    )
    with pytest.raises(ValueError, match="producer failure") as caught:
        with profiler:
            with profile_region("measured.region"):
                raise ValueError("producer failure")

    assert get_current_profiler() is None
    assert profiler._events_fp is None
    assert any(
        "profiler finalization also failed" in note and "nonfinite or negative" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_body_error_survives_deferred_cuda_end_record_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deferred_end_record_failure(monkeypatch)

    out_dir = tmp_path / "profile"
    profiler = AxBenchProfiler(
        out_dir,
        synchronize_cuda=False,
        cuda_event_regions=("measured.region",),
    )
    with pytest.raises(ValueError, match="producer failure") as caught:
        with profiler:
            with profile_region("measured.region"):
                raise ValueError("producer failure")

    assert get_current_profiler() is None
    assert profiler._events_fp is None
    assert any(
        "region 'measured.region' profiling finalization also failed" in note
        and "synthetic end-event record failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["cuda_event_timing"]["missing_regions"] == ["measured.region"]


def test_deferred_cuda_end_record_error_propagates_after_successful_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deferred_end_record_failure(monkeypatch)

    out_dir = tmp_path / "profile"
    profiler = AxBenchProfiler(
        out_dir,
        synchronize_cuda=False,
        cuda_event_regions=("measured.region",),
    )
    with pytest.raises(RuntimeError, match="synthetic end-event record failure"):
        with profiler:
            with profile_region("measured.region"):
                pass

    assert get_current_profiler() is None
    assert profiler._events_fp is None
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["cuda_event_timing"]["missing_regions"] == ["measured.region"]


def test_nested_profiler_context_restores_outer_profiler(tmp_path: Path) -> None:
    outer_dir = tmp_path / "outer"
    inner_dir = tmp_path / "inner"

    with AxBenchProfiler(outer_dir) as outer:
        assert get_current_profiler() is outer
        with AxBenchProfiler(inner_dir) as inner:
            assert get_current_profiler() is inner
            record_counter("inner.counter", 1)
        assert get_current_profiler() is outer
        record_counter("outer.counter", 1)

    assert get_current_profiler() is None
    outer_summary = yaml.safe_load((outer_dir / "summary.yaml").read_text(encoding="utf-8"))
    inner_summary = yaml.safe_load((inner_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert [c["name"] for c in outer_summary["counters"]] == ["outer.counter"]
    assert [c["name"] for c in inner_summary["counters"]] == ["inner.counter"]


def test_profiler_exception_records_error_status_and_resets_context(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "profile"

    with pytest.raises(RuntimeError, match="boom"):
        with AxBenchProfiler(out_dir):
            with profile_region("failing.region"):
                raise RuntimeError("boom")

    assert get_current_profiler() is None
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(e["type"] == "region" and e["name"] == "failing.region" for e in events)
    assert events[-1]["type"] == "profiler_stop"
    assert events[-1]["attrs"]["status"] == "error"
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "error"


def test_profiler_json_normalizes_common_attr_types(tmp_path: Path) -> None:
    out_dir = tmp_path / "profile"

    with AxBenchProfiler(out_dir):
        with profile_region(
            "json.attrs",
            path=Path("x/y"),
            stage=Stage.PREPARE_CONCEPTS,
            scalar=t.tensor(3),
            vector=t.tensor([1, 2]),
        ):
            pass

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event = next(e for e in events if e["type"] == "region")
    assert event["attrs"]["path"] == "x/y"
    assert event["attrs"]["stage"] == "0a"
    assert event["attrs"]["scalar"] == 3
    assert event["attrs"]["vector"] == {
        "shape": [2],
        "dtype": "torch.int64",
        "device": "cpu",
    }
