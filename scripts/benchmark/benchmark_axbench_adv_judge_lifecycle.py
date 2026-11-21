"""Profile one published AxBench-Adv slice with a local deterministic judge."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from scripts.benchmark.benchmark_axbench_adv_judge_pair import _LocalJudgeAPI
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.pipeline import run_pipeline
from steering.eval.axbench_adv.self_conditioning import _BoundSelfConditioningCell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--all-three-methods", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    events: list[dict[str, Any]] = []

    def timed(name: str):
        original = getattr(_BoundSelfConditioningCell, name)

        def wrapper(self: _BoundSelfConditioningCell, *values: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(self, *values, **kwargs)
            finally:
                events.append({
                    "method": self._method,
                    "concept_id": self._concept_id,
                    "factor": self._factor,
                    "phase": name,
                    "seconds": time.perf_counter() - started,
                })

        setattr(_BoundSelfConditioningCell, name, wrapper)

    for method in ("__enter__", "evaluate", "close"):
        timed(method)

    os.environ["AXBENCH_ADV_METHODS"] = (
        "SelfConditioning,SelfConditioningMasked,NLSteering"
        if args.all_three_methods else "SelfConditioningMasked"
    )
    os.environ["AXBENCH_ADV_CONCEPT_LO"] = "36"
    os.environ["AXBENCH_ADV_CONCEPT_HI"] = "37"
    os.environ["AXBENCH_ADV_PRESSURES"] = "diffmean_counter,linear_probe_counter"
    os.environ["OPENROUTER_API_KEY"] = "offline-benchmark"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy") and name.lower() != "no_proxy":
            del os.environ[name]

    with _LocalJudgeAPI(1.0) as server:
        os.environ["OPENROUTER_BASE_URL"] = server.base_url
        raw = OmegaConf.load(args.config)
        if not isinstance(raw, DictConfig):
            raise ValueError("benchmark config must be a mapping")
        raw.output_dir = str(args.root / "output")
        raw.judge_cache = str(args.root / "judge.sqlite")
        raw.reflection_cache = str(args.root / "unused-reflection.sqlite")
        raw.api_base_url = server.base_url
        raw.counter_magnitudes = [256]
        raw.project_judge_inflight = 3
        raw.project_judge_workers = 2
        raw.project_judge_pair_concurrency = 2
        raw.project_reuse_single_token_mask = True
        raw.project_reuse_diffmean_texts = True
        raw.project_fit_selected_concepts_only = True
        raw.reuse_final_judge_worker = True
        started = time.perf_counter()
        manifest_path = run_pipeline(build_config(raw))
        wall = time.perf_counter() - started
        report = {
            "wall_seconds": wall,
            "events": events,
            "local_http_calls": server.calls,
            "local_http_peak_concurrency": server.peak,
            "manifest": str(manifest_path),
        }
        (args.root / "lifecycle-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
