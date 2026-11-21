"""Real-CUDA integration coverage for deferred profiler event timing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
import yaml

from steering.eval.axbench_base.profiler import AxBenchProfiler, profile_region

pytestmark = pytest.mark.slow


def test_deferred_cuda_events_measure_a_real_matrix_multiply(tmp_path: Path) -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA is required for deferred event integration coverage")

    device_index = t.cuda.current_device()
    device = t.device("cuda", device_index)
    generator = t.Generator(device=device).manual_seed(20260830)
    left = t.randn((2048, 2048), device=device, dtype=t.float16, generator=generator)
    right = t.randn((2048, 2048), device=device, dtype=t.float16, generator=generator)
    stream = t.cuda.current_stream(device_index)
    reference_start = t.cuda.Event(enable_timing=True)
    reference_end = t.cuda.Event(enable_timing=True)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(
        profile_dir,
        synchronize_cuda=False,
        gpu_utilization=False,
        gpu_device_index=device_index,
        cuda_event_regions=("integration.cuda_matmul",),
    ) as profiler:
        with profile_region("integration.unmeasured"):
            unmeasured = left + right
        reference_start.record(stream)
        with profile_region("integration.cuda_matmul", rows=left.shape[0]):
            product = left @ right
        reference_end.record(stream)
        profiler.require_cuda_event_regions_complete()

    assert bool(t.isfinite(product).all().item())
    assert bool(t.isfinite(unmeasured).all().item())

    events = [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    timing_event = next(event for event in events if event["type"] == "cuda_event_timing")
    assert timing_event["schema_version"] == 1
    assert timing_event["device_index"] == device_index
    assert timing_event["configured_regions"] == ["integration.cuda_matmul"]
    assert timing_event["missing_regions"] == []
    assert timing_event["event_count"] == 1
    assert timing_event["regions"][0]["name"] == "integration.cuda_matmul"
    assert timing_event["regions"][0]["count"] == 1
    assert timing_event["regions"][0]["errored_count"] == 0
    assert timing_event["regions"][0]["total_s"] > 0.0
    reference_s = float(reference_start.elapsed_time(reference_end)) / 1000.0
    assert timing_event["regions"][0]["total_s"] <= reference_s
    assert timing_event["regions"][0]["mean_s"] == timing_event["regions"][0]["total_s"]
    assert timing_event["regions"][0]["min_s"] == timing_event["regions"][0]["total_s"]
    assert timing_event["regions"][0]["max_s"] == timing_event["regions"][0]["total_s"]
    assert events[-1]["type"] == "profiler_stop"
    assert events[-1]["attrs"]["status"] == "ok"

    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["cuda_event_timing"] == {
        key: value for key, value in timing_event.items() if key not in {"type", "time"}
    }
    assert not any(
        row["name"] == "integration.unmeasured" for row in summary["cuda_event_timing"]["regions"]
    )
