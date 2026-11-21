"""Tests for the issue-657 AxBench profiling roll-up (CPU-only, no GPU).

The roll-up reads finished ``profile/summary.yaml`` files (the schema written by
``steering.eval.axbench_base.profiler.AxBenchProfiler.write_reports``) and emits a
combined manifest + Markdown report. These tests drive it against synthetic
summaries that mirror that schema exactly (``regions`` is a LIST of dicts keyed by
``name``; memory is raw bytes / KB).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.benchmark.rollup_axbench_profile import (
    build_manifest,
    load_run,
    render_report,
)

_GB = 1024**3
_KB_GB = 1024**2


def _write_summary(
    profile_dir: Path,
    *,
    run_tag: str,
    regions: list[dict],
    counters: list[dict] | None = None,
    elapsed_s: float = 1.0,
) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": {"run_tag": run_tag},
        "status": "ok",
        "elapsed_s": elapsed_s,
        "gpu_utilization": {"mean": 50.0},
        "regions": regions,
        "counters": counters or [],
        "first_verified_candidate_breakdown": [],
    }
    (profile_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    return profile_dir


def _bpb_region(stage: str, *, total_s: float, cuda_gb: float, rss_gb: float) -> dict:
    return {
        "name": f"axbench_bpb_adv.{stage}",
        "count": 1,
        "total_s": total_s,
        "mean_s": total_s,
        "min_s": total_s,
        "max_s": total_s,
        "rows": 0,
        "peak_cuda_allocated_bytes": int(cuda_gb * _GB),
        "peak_cuda_reserved_bytes": int(cuda_gb * _GB),
        "peak_cpu_rss_kb": int(rss_gb * _KB_GB),
        "gpu_util_mean": 42.0,
        "gpu_util_min": 0.0,
        "gpu_util_max": 100.0,
        "gpu_sample_count": 10,
        "gpu_idle_fraction": 0.25,
    }


def test_load_run_converts_units_and_infers_pipeline(tmp_path: Path):
    pdir = _write_summary(
        tmp_path / "cold" / "profile",
        run_tag="issue657-l12-cold",
        regions=[_bpb_region("build_candidates", total_s=120.0, cuda_gb=4.0, rss_gb=8.0)],
    )
    run = load_run("cold", pdir)

    assert run.pipeline == "bpb_adv"
    assert run.run_tag == "issue657-l12-cold"
    stage = run.regions["axbench_bpb_adv.build_candidates"]
    assert stage.wall_s == 120.0
    # bytes -> GB and KB -> GB
    assert stage.peak_cuda_gb == 4.0
    assert stage.cpu_rss_gb == 8.0


def test_load_run_accepts_summary_file_or_parent_dir(tmp_path: Path):
    pdir = _write_summary(
        tmp_path / "p",
        run_tag="t",
        regions=[_bpb_region("score", total_s=1.0, cuda_gb=1.0, rss_gb=1.0)],
    )
    # both the dir and the summary.yaml path resolve to the same run
    by_dir = load_run("a", pdir)
    by_file = load_run("a", pdir / "summary.yaml")
    assert by_dir.regions.keys() == by_file.regions.keys()


def test_manifest_has_cold_warm_comparison_with_speedup(tmp_path: Path):
    cold = load_run(
        "cold",
        _write_summary(
            tmp_path / "cold" / "profile",
            run_tag="cold",
            regions=[_bpb_region("score", total_s=200.0, cuda_gb=4.0, rss_gb=8.0)],
        ),
    )
    warm = load_run(
        "warm",
        _write_summary(
            tmp_path / "warm" / "profile",
            run_tag="warm",
            regions=[_bpb_region("score", total_s=50.0, cuda_gb=4.0, rss_gb=8.0)],
        ),
    )
    manifest = build_manifest([cold, warm], generated_at="2026-07-11", git_sha="abc1234")

    assert manifest["git_sha"] == "abc1234"
    assert {r["label"] for r in manifest["runs"]} == {"cold", "warm"}
    comp = manifest["comparison"]["axbench_bpb_adv.score"]
    assert comp["cold_wall_s"] == 200.0
    assert comp["warm_wall_s"] == 50.0
    # warm reused the cache -> 4x faster
    assert comp["warm_speedup"] == 4.0


def test_render_report_fills_bpb_adv_table_and_flags_missing_stages(tmp_path: Path):
    cold = load_run(
        "cold",
        _write_summary(
            tmp_path / "cold" / "profile",
            run_tag="cold",
            regions=[
                _bpb_region("build_candidates", total_s=300.0, cuda_gb=6.0, rss_gb=9.0),
                _bpb_region("score", total_s=100.0, cuda_gb=4.0, rss_gb=8.0),
            ],
        ),
    )
    report = render_report([cold], generated_at="2026-07-11", git_sha="abc1234")

    # The canonical 8 stages appear in order; present ones carry numbers,
    # absent ones are flagged (not silently blank).
    assert "build_candidates" in report
    assert "300.0" in report  # build_candidates wall
    assert "6.0" in report  # build_candidates peak CUDA GB
    # a stage with no region should be marked, not dropped
    assert "prepare_context_cache" in report
    assert "n/a" in report
    assert "abc1234" in report  # provenance


def test_end_to_end_writes_manifest_and_report(tmp_path: Path):
    # Smoke: two runs -> manifest json + markdown report on disk.
    cold_dir = _write_summary(
        tmp_path / "cold" / "profile",
        run_tag="cold",
        regions=[_bpb_region("score", total_s=200.0, cuda_gb=4.0, rss_gb=8.0)],
        counters=[{"name": "score.rows", "count": 1, "total": 3600}],
    )
    warm_dir = _write_summary(
        tmp_path / "warm" / "profile",
        run_tag="warm",
        regions=[_bpb_region("score", total_s=40.0, cuda_gb=4.0, rss_gb=8.0)],
    )
    runs = [load_run("cold", cold_dir), load_run("warm", warm_dir)]
    manifest = build_manifest(runs, generated_at="2026-07-11", git_sha="abc1234")
    report = render_report(runs, generated_at="2026-07-11", git_sha="abc1234")

    manifest_path = tmp_path / "rollup.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loaded = json.loads(manifest_path.read_text())
    assert loaded["comparison"]["axbench_bpb_adv.score"]["warm_speedup"] == 5.0
    assert "score.rows" in loaded["runs"][0]["counters"]
    assert report.strip().startswith("#")
