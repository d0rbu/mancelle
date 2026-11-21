"""Provider-free, bounded AxBench-Adv publication replay on one real GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.benchmark.benchmark_axbench_adv_judge_pair import _LocalJudgeAPI
from scripts.eval.axbench_adv.merge_slices import _load_run_slice
from scripts.eval.axbench_adv.pressures import Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", choices=("serial", "enabled"), required=True)
    parser.add_argument("--methods", default="SelfConditioningMasked")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.root.exists():
        raise ValueError(f"replay root already exists: {args.root}")
    args.root.mkdir(parents=True)
    os.environ["AXBENCH_ADV_METHODS"] = args.methods
    os.environ["AXBENCH_ADV_CONCEPT_LO"] = "36"
    os.environ["AXBENCH_ADV_CONCEPT_HI"] = "37"
    os.environ["AXBENCH_ADV_PRESSURES"] = "diffmean_counter,linear_probe_counter"
    os.environ["OPENROUTER_API_KEY"] = "offline-benchmark"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise ValueError("benchmark config must be a mapping")
    config.output_dir = str(args.root / "output")
    config.judge_cache = str(args.root / "judge.sqlite")
    config.reflection_cache = str(args.root / "unused-reflection.sqlite")
    config.counter_magnitudes = [256]
    config.project_judge_inflight = 2 if args.mode == "enabled" else 1
    config.project_judge_pair_concurrency = 2 if args.mode == "enabled" else 1
    config.project_reuse_single_token_mask = args.mode == "enabled"
    config.project_reuse_diffmean_texts = args.mode == "enabled"
    config.project_fit_selected_concepts_only = args.mode == "enabled"
    with _LocalJudgeAPI(args.delay) as server:
        config.api_base_url = server.base_url
        os.environ["OPENROUTER_BASE_URL"] = server.base_url
        start = time.perf_counter()
        manifest_path = run_pipeline(build_config(config))
        wall = time.perf_counter() - start
        for pressure in (Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER):
            _load_run_slice(manifest_path.parent, pressure)
        manifest = json.loads(manifest_path.read_text())
        result = {
            "mode": args.mode,
            "wall_seconds": wall,
            "manifest": str(manifest_path),
            "manifest_file_count": len(manifest["files"]),
            "calls": server.calls,
            "peak_api_concurrency": server.peak,
            "manifest_call_accounting": manifest["call_accounting"],
            "methods": manifest["methods"],
            "concept_ids": manifest["concept_ids"],
            "pressures": manifest["pressures"],
            "magnitudes": manifest["counter_magnitudes"],
        }
        (args.root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
