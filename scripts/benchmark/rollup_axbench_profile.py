"""Roll up AxBench profiler outputs (issue #657) into a manifest + Markdown report.

The production AxBench-Base / BPB-Adv CLIs, run with ``+profile.enabled=true``,
each write ``<output_root>/<run_tag>/profile/summary.yaml`` (schema:
``steering.eval.axbench_base.profiler.AxBenchProfiler.write_reports`` -- ``regions``
is a LIST of dicts keyed by ``name``; ``peak_cuda_*`` is bytes, ``peak_cpu_rss_kb``
is KB). This tool reads one or more of those per-run summaries -- typically a
cold-cache and a warm-cache pass -- converts the raw memory figures to GB, and
emits:

* a machine-readable JSON manifest combining every run's per-stage wall time,
  peak CUDA/CPU memory and GPU utilisation, plus a cold-vs-warm comparison; and
* a Markdown report with the 8-stage BPB-Adv timing and memory table for
  ad-hoc analysis outside the source tree.

CPU-only: it reads finished profiler outputs, so it never touches a GPU and is
safe to run anywhere.

Usage::

    uv run python -m scripts.benchmark.rollup_axbench_profile \\
        --run cold=out/eval/axbench_bpb_adv/issue657-l12-standard72-cold/profile \\
        --run warm=out/eval/axbench_bpb_adv/issue657-l12-standard72-warm/profile \\
        --run base=out/eval/axbench_base/issue657-l12-src/profile \\
        --out-manifest out/eval/issue657_profile_rollup.json \\
        --out-report out/eval/issue657_profile_rollup.md

Each ``--run LABEL=PATH`` names a run; ``PATH`` is a ``profile/`` dir or a
``summary.yaml``. A cold-vs-warm comparison is produced for any two runs whose
labels contain ``cold`` and ``warm``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_BYTES_PER_GB = 1024**3
_KB_PER_GB = 1024**2

# The 8 BPB-Adv stages in pipeline order; region name is
# ``axbench_bpb_adv.<stage>``.
_BPB_ADV_STAGES: tuple[str, ...] = (
    "validate_source",
    "prepare_context_cache",
    "prepare_attack_rows",
    "build_candidates",
    "score",
    "score_general_val",
    "aggregate",
    "plot",
)
_BPB_ADV_PREFIX = "axbench_bpb_adv."
_BASE_PREFIXES: tuple[str, ...] = (
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
)


def _bytes_to_gb(n: float) -> float:
    return round(float(n) / _BYTES_PER_GB, 3)


def _kb_to_gb(n: float) -> float:
    return round(float(n) / _KB_PER_GB, 3)


@dataclass
class RegionStat:
    name: str
    wall_s: float
    peak_cuda_gb: float
    peak_cuda_reserved_gb: float
    cpu_rss_gb: float
    gpu_util_mean: float
    gpu_idle_fraction: float
    count: int


@dataclass
class RunProfile:
    label: str
    source: str
    run_tag: str
    status: str
    elapsed_s: float
    pipeline: str  # "bpb_adv" | "base" | "mixed" | "unknown"
    regions: dict[str, RegionStat]
    counters: dict[str, float]


def _resolve_summary_path(path: Path) -> Path:
    return path / "summary.yaml" if path.is_dir() else path


def _infer_pipeline(region_names: list[str]) -> str:
    has_bpb = any(n.startswith(_BPB_ADV_PREFIX) for n in region_names)
    has_base = any(
        n.startswith(_BASE_PREFIXES) for n in region_names
    )
    if has_bpb and has_base:
        return "mixed"
    if has_bpb:
        return "bpb_adv"
    if has_base:
        return "base"
    return "unknown"


def load_run(label: str, path: Path) -> RunProfile:
    """Parse a profiler ``summary.yaml`` (or its parent dir) into a RunProfile."""
    summary_path = _resolve_summary_path(path)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No profiler summary at {summary_path}. Expected a "
            "'<run_tag>/profile/summary.yaml' (or the profile dir containing it) "
            "written by a run with '+profile.enabled=true'."
        )
    data = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    regions: dict[str, RegionStat] = {}
    for entry in data.get("regions", []):
        name = str(entry["name"])
        regions[name] = RegionStat(
            name=name,
            wall_s=round(float(entry.get("total_s", 0.0)), 3),
            peak_cuda_gb=_bytes_to_gb(entry.get("peak_cuda_allocated_bytes", 0)),
            peak_cuda_reserved_gb=_bytes_to_gb(
                entry.get("peak_cuda_reserved_bytes", 0)
            ),
            cpu_rss_gb=_kb_to_gb(entry.get("peak_cpu_rss_kb", 0)),
            gpu_util_mean=round(float(entry.get("gpu_util_mean", 0.0)), 1),
            gpu_idle_fraction=round(float(entry.get("gpu_idle_fraction", 0.0)), 3),
            count=int(entry.get("count", 0)),
        )
    counters = {
        str(entry["name"]): float(entry.get("total", 0.0))
        for entry in data.get("counters", [])
    }
    meta = data.get("metadata", {}) or {}
    return RunProfile(
        label=label,
        source=str(summary_path),
        run_tag=str(meta.get("run_tag", "")),
        status=str(data.get("status", "")),
        elapsed_s=round(float(data.get("elapsed_s", 0.0)), 3),
        pipeline=_infer_pipeline(list(regions)),
        regions=regions,
        counters=counters,
    )


def _find_labelled(runs: list[RunProfile], token: str) -> RunProfile | None:
    matches = [r for r in runs if token in r.label.lower()]
    return matches[0] if len(matches) == 1 else None


def _build_comparison(runs: list[RunProfile]) -> dict[str, dict[str, float]]:
    """Per-region cold-vs-warm wall comparison, if exactly one of each exists."""
    cold = _find_labelled(runs, "cold")
    warm = _find_labelled(runs, "warm")
    if cold is None or warm is None:
        return {}
    comparison: dict[str, dict[str, float]] = {}
    for name in sorted(set(cold.regions) & set(warm.regions)):
        cold_wall = cold.regions[name].wall_s
        warm_wall = warm.regions[name].wall_s
        comparison[name] = {
            "cold_wall_s": cold_wall,
            "warm_wall_s": warm_wall,
            "warm_speedup": (
                round(cold_wall / warm_wall, 2) if warm_wall > 0 else 0.0
            ),
        }
    return comparison


def build_manifest(
    runs: list[RunProfile], *, generated_at: str, git_sha: str
) -> dict[str, Any]:
    return {
        "tool": "rollup_axbench_profile",
        "issue": 657,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "runs": [
            {
                "label": r.label,
                "source": r.source,
                "run_tag": r.run_tag,
                "pipeline": r.pipeline,
                "status": r.status,
                "elapsed_s": r.elapsed_s,
                "stages": {
                    name: dataclasses.asdict(rs) for name, rs in r.regions.items()
                },
                "counters": r.counters,
            }
            for r in runs
        ],
        "comparison": _build_comparison(runs),
    }


def _cell(run: RunProfile | None, stage: str, attr: str) -> str:
    if run is None:
        return "n/a"
    region = run.regions.get(_BPB_ADV_PREFIX + stage)
    if region is None:
        return "n/a"
    return str(getattr(region, attr))


def render_report(
    runs: list[RunProfile], *, generated_at: str, git_sha: str
) -> str:
    cold = _find_labelled(runs, "cold")
    warm = _find_labelled(runs, "warm")
    # If no cold/warm split, treat the first bpb_adv run as the single column.
    if cold is None and warm is None:
        cold = next(
            (r for r in runs if r.pipeline in ("bpb_adv", "mixed")), None
        )

    lines: list[str] = []
    lines.append("# AxBench BPB-Adv l12 profile (issue #657)")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append(f"- Commit: `{git_sha}`")
    for r in runs:
        lines.append(
            f"- Run `{r.label}` ({r.pipeline}, run_tag=`{r.run_tag}`): "
            f"elapsed {r.elapsed_s}s, status {r.status}"
        )
    lines.append("")

    # --- BPB-Adv per-stage table (the baseline-doc table, cold + warm) ---
    lines.append("## BPB-Adv per stage")
    lines.append("")
    lines.append(
        "| Stage | Cold wall (s) | Warm wall (s) | Cold peak CUDA (GB) "
        "| Warm peak CUDA (GB) | Cold CPU RSS (GB) | Warm CPU RSS (GB) "
        "| Warm speedup |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage in _BPB_ADV_STAGES:
        cold_wall = _cell(cold, stage, "wall_s")
        warm_wall = _cell(warm, stage, "wall_s")
        speedup = "n/a"
        if cold_wall != "n/a" and warm_wall != "n/a" and float(warm_wall) > 0:
            speedup = f"{round(float(cold_wall) / float(warm_wall), 2)}x"
        lines.append(
            f"| `{stage}` | {cold_wall} | {warm_wall} "
            f"| {_cell(cold, stage, 'peak_cuda_gb')} "
            f"| {_cell(warm, stage, 'peak_cuda_gb')} "
            f"| {_cell(cold, stage, 'cpu_rss_gb')} "
            f"| {_cell(warm, stage, 'cpu_rss_gb')} | {speedup} |"
        )
    lines.append("")

    # --- AxBench-Base top regions (hypotheses 1-2: Stage 0a / 0c) ---
    base_runs = [r for r in runs if r.pipeline in ("base", "mixed")]
    if base_runs:
        lines.append("## AxBench-Base top regions (by wall)")
        lines.append("")
        for r in base_runs:
            lines.append(f"### `{r.label}`")
            lines.append("")
            lines.append("| Region | Wall (s) | Peak CUDA (GB) | CPU RSS (GB) | GPU util % |")
            lines.append("|---|---:|---:|---:|---:|")
            top = sorted(
                r.regions.values(), key=lambda x: x.wall_s, reverse=True
            )[:15]
            for rs in top:
                lines.append(
                    f"| `{rs.name}` | {rs.wall_s} | {rs.peak_cuda_gb} "
                    f"| {rs.cpu_rss_gb} | {rs.gpu_util_mean} |"
                )
            lines.append("")

    lines.append(
        "> Peak CUDA is `peak_cuda_allocated_bytes` / 2^30; CPU RSS is "
        "`peak_cpu_rss_kb` / 2^20. `n/a` = the stage emitted no region in that "
        "run (e.g. a stage not run, or the cold/warm column absent)."
    )
    lines.append("")
    return "\n".join(lines)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--run must be LABEL=PATH, got {raw!r}"
        )
    label, _, path = raw.partition("=")
    label = label.strip()
    if not label or not path.strip():
        raise argparse.ArgumentTypeError(
            f"--run must be LABEL=PATH with both non-empty, got {raw!r}"
        )
    return label, Path(path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A run to include; PATH is a profile/ dir or summary.yaml. Repeat "
        "for cold/warm/base. Labels containing 'cold'/'warm' drive the comparison.",
    )
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Override the timestamp (default: now, UTC). Set for reproducibility.",
    )
    args = parser.parse_args(argv)

    generated_at = args.generated_at or datetime.now(UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    git_sha = _git_sha()

    runs = [load_run(*_parse_run_arg(raw)) for raw in args.run]

    manifest = build_manifest(runs, generated_at=generated_at, git_sha=git_sha)
    report = render_report(runs, generated_at=generated_at, git_sha=git_sha)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report, encoding="utf-8")

    print(f"Wrote manifest: {args.out_manifest}")
    print(f"Wrote report:   {args.out_report}")
    for name, comp in manifest["comparison"].items():
        print(f"  {name}: warm {comp['warm_speedup']}x faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
