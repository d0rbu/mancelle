"""Compare bounded project judge settings through real-GPU publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.benchmark.benchmark_axbench_adv_judge_pair import _LocalJudgeAPI
from scripts.eval.axbench_adv.merge_slices import _load_run_slice
from scripts.eval.axbench_adv.pressures import Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.pipeline import run_pipeline


def _published_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and "logs" not in path.relative_to(root).parts
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", default="SelfConditioningMasked")
    parser.add_argument("--magnitudes", default="256", help="Comma-separated positive raw strengths")
    parser.add_argument("--judge-delay", type=float, default=2.0)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-inflight-depth",
        action="store_true",
        help="Compare three versus two in-flight responses with two judge workers fixed.",
    )
    comparison.add_argument(
        "--compare-final-worker",
        action="store_true",
        help="Compare persistent versus fresh final evaluator processes with project settings fixed.",
    )
    comparison.add_argument(
        "--compare-judge-rebind",
        action="store_true",
        help="Compare reused versus fresh project judge workers with other settings fixed.",
    )
    parser.add_argument("--reverse-order", action="store_true", help="Run the baseline first.")
    args = parser.parse_args()
    try:
        magnitudes = [int(value) for value in args.magnitudes.split(",")]
    except ValueError:
        parser.error("magnitudes must be comma-separated positive integers")
    if not magnitudes or any(value <= 0 for value in magnitudes) or len(set(magnitudes)) != len(magnitudes):
        parser.error("magnitudes must be distinct positive integers")
    if args.judge_delay < 0 or not t.cuda.is_available():
        parser.error("a real CUDA device and nonnegative judge delay are required")
    if args.root.exists():
        raise ValueError(f"replay root already exists: {args.root}")
    args.root.mkdir(parents=True)
    os.environ["AXBENCH_ADV_METHODS"] = args.method
    os.environ["AXBENCH_ADV_CONCEPT_LO"] = "36"
    os.environ["AXBENCH_ADV_CONCEPT_HI"] = "37"
    os.environ["AXBENCH_ADV_PRESSURES"] = "diffmean_counter,linear_probe_counter"
    os.environ["OPENROUTER_API_KEY"] = "offline-benchmark"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy") and name.lower() != "no_proxy":
            del os.environ[name]

    runs: dict[str, dict[str, Any]] = {}
    with _LocalJudgeAPI(args.judge_delay) as server:
        os.environ["OPENROUTER_BASE_URL"] = server.base_url
        if args.compare_judge_rebind:
            settings = (("judge_rebind_on", 2, 3, True, True), ("judge_rebind_off", 2, 3, True, False))
        elif args.compare_final_worker:
            settings = (("final_worker_on", 2, 3, True, False), ("final_worker_off", 2, 3, False, False))
        elif args.compare_inflight_depth:
            settings = (("inflight3", 2, 3, False, False), ("inflight2", 2, 2, False, False))
        else:
            settings = (("workers2", 2, 2, False, False), ("workers1", 1, 2, False, False))
        if args.reverse_order:
            settings = tuple(reversed(settings))
        for name, workers, inflight, persistent_final, reuse_judges in settings:
            raw = OmegaConf.load(args.config)
            if not isinstance(raw, DictConfig):
                raise ValueError("benchmark config must be a mapping")
            run_root = args.root / name
            raw.output_dir = str(run_root / "output")
            raw.judge_cache = str(run_root / "judge.sqlite")
            raw.reflection_cache = str(run_root / "unused-reflection.sqlite")
            raw.api_base_url = server.base_url
            raw.counter_magnitudes = magnitudes
            raw.project_judge_inflight = inflight
            raw.project_judge_workers = workers
            raw.project_judge_pair_concurrency = 2
            raw.project_reuse_judge_workers = reuse_judges
            raw.project_reuse_single_token_mask = True
            raw.project_reuse_diffmean_texts = True
            raw.project_fit_selected_concepts_only = True
            raw.reuse_final_judge_worker = persistent_final
            server.reset()
            started = time.perf_counter()
            manifest_path = run_pipeline(build_config(raw))
            wall = time.perf_counter() - started
            for pressure in (Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER):
                _load_run_slice(manifest_path.parent, pressure)
            manifest = json.loads(manifest_path.read_text())
            runs[name] = {
                "wall_seconds": wall,
                "provider_calls": server.calls,
                "peak_provider_concurrency": server.peak,
                "call_accounting": manifest["call_accounting"],
                "manifest_files": manifest["files"],
                "published_files": _published_files(manifest_path.parent),
            }

    if args.compare_judge_rebind:
        candidate, baseline = runs["judge_rebind_on"], runs["judge_rebind_off"]
    elif args.compare_final_worker:
        candidate, baseline = runs["final_worker_on"], runs["final_worker_off"]
    elif args.compare_inflight_depth:
        candidate, baseline = runs["inflight3"], runs["inflight2"]
    else:
        candidate, baseline = runs["workers2"], runs["workers1"]
    same_files = candidate["published_files"] == baseline["published_files"]
    same_manifest_inventory = candidate["manifest_files"] == baseline["manifest_files"]
    same_accounting = candidate["call_accounting"] == baseline["call_accounting"]
    report = {
        "device": t.cuda.get_device_name(),
        "method": args.method,
        "concept_ids": [36],
        "pressures": ["diffmean_counter", "linear_probe_counter"],
        "counter_magnitudes": magnitudes,
        "input_count_per_point": 10,
        "judge_delay_seconds_per_response_if_sequential": args.judge_delay,
        "comparison": (
            "final_worker"
            if args.compare_final_worker
            else (
                "judge_rebind"
                if args.compare_judge_rebind
                else ("inflight_depth" if args.compare_inflight_depth else "worker_pool")
            )
        ),
        "order": [name for name, _workers, _inflight, _persistent_final, _reuse_judges in settings],
        "runs": {
            name: {key: value for key, value in run.items() if key not in ("manifest_files", "published_files")}
            for name, run in runs.items()
        },
        "same_manifest_inventory": same_manifest_inventory,
        "same_call_accounting": same_accounting,
        "same_published_files_excluding_manifest": same_files,
        "published_file_count": len(candidate["published_files"]),
        "speedup": baseline["wall_seconds"] / candidate["wall_seconds"],
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not (same_files and same_manifest_inventory and same_accounting):
        raise AssertionError("judge-worker replay changed published output or call accounting")


if __name__ == "__main__":
    main()
