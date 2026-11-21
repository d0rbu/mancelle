"""Real-GPU, provider-free paired replay of project AxBench-Adv counter judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.eval.axbench_adv.pressures import Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv import self_conditioning as conditioning_module
from steering.eval.axbench_adv.best_point import point_at_factor
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_counter_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.pipeline import _cell_rows
from steering.eval.axbench_adv.self_conditioning import JudgeFactory, SelfConditioningMethodRuntime
from steering.eval.axbench_adv.source_contract import validate_source_contract
from steering.eval.axbench_adv.types import NL_METHOD


@dataclass
class _Timers:
    generation_seconds: float = 0.0
    judge_seconds: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    generation_index: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


class _Judge:
    def __init__(self, *, init_payload: dict[str, Any], delay: float, timers: _Timers, **_: Any):
        self._concept = init_payload["concept"]
        self._delay = delay
        self._timers = timers
        self._cache: set[tuple[str, str]] = set()

    def reset(self) -> None:
        self._cache.clear()

    def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
        start = time.perf_counter()
        started_unix = time.time()
        misses = 0
        for kind in ("concept", "fluency"):
            self._timers.calls += 1
            key = (kind, self._concept + generation if kind == "concept" else generation)
            if key in self._cache:
                self._timers.cache_hits += 1
            else:
                time.sleep(self._delay / 2)
                self._cache.add(key)
                misses += 1
        self._timers.judge_seconds += time.perf_counter() - start
        self._timers.trace.append(
            {
                "stage": "judge",
                "input_id": input_id,
                "start_unix": started_unix,
                "end_unix": time.time(),
                "paid_misses": misses,
            }
        )
        return RawEvaluation(generation, 1.5, 1.25, "{}", misses)

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-count", type=int, default=5)
    parser.add_argument("--judge-delay", type=float, default=2.0)
    parser.add_argument("--magnitude", type=float, default=256.0)
    parser.add_argument("--mode-only", choices=("serial", "overlap"))
    args = parser.parse_args()
    if not 1 <= args.input_count <= 10 or args.judge_delay < 0:
        parser.error("input-count must be in 1..10 and judge-delay must be nonnegative")
    if not t.cuda.is_available():
        parser.error("this benchmark requires a real CUDA device")
    benchmark_start = time.perf_counter()
    modes = (1, 2, 2, 1) if args.mode_only is None else (1 if args.mode_only == "serial" else 2,)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_config = OmegaConf.load(args.config)
    if not isinstance(raw_config, DictConfig):
        raise ValueError("benchmark config must be a mapping")
    config = build_config(raw_config)
    source_start = time.perf_counter()
    validated = validate_source_contract(
        paths=config.source_paths,
        expectations=config.source_expectations,
        expected_hashes=config.source_hashes,
        output_dir=output,
        reserved_paths=(config.judge_cache, config.reflection_cache),
        nl_attachment_paths=config.nl_attachment_paths,
        nl_attachment_expectations=config.nl_attachment_expectations,
    )
    source_seconds = time.perf_counter() - source_start
    if config.project_counter_direction_root is None or config.project_counter_layer is None:
        raise ValueError("project counter direction root and layer are required")
    if validated.target_meta is None:
        raise ValueError("validated target metadata is missing")
    meta = validated.target_meta["model_config"]
    fit_start = time.perf_counter()
    fit = preflight_project_direction_fit(
        config.project_counter_direction_root,
        ProjectCounterExpectations(
            target_model_sha256=validated.runtime_hashes["target_model"],
            target_metadata_sha256=validated.runtime_hashes["target_meta"],
            concept_texts={key: value.concept for key, value in validated.training_texts.items()},
            decoder_width=meta["n_embd"],
            layer=config.project_counter_layer,
            fit_seed=config.project_counter_fit_seed,
        ),
    )
    fit_seconds = time.perf_counter() - fit_start
    timers = _Timers()
    original_generate = conditioning_module.generate_self_conditioning

    def timed_generate(*call_args: Any, **call_kwargs: Any) -> str:
        t.cuda.synchronize()
        start = time.perf_counter()
        started_unix = time.time()
        result = original_generate(*call_args, **call_kwargs)
        t.cuda.synchronize()
        timers.generation_seconds += time.perf_counter() - start
        timers.trace.append(
            {
                "stage": "generate",
                "input_id": timers.generation_index,
                "start_unix": started_unix,
                "end_unix": time.time(),
            }
        )
        timers.generation_index += 1
        return result

    generation_patch = patch.object(
        conditioning_module, "generate_self_conditioning", timed_generate
    )
    generation_patch.start()
    judges: list[_Judge] = []

    def judge_factory(**kwargs: Any) -> _Judge:
        judge = _Judge(delay=args.judge_delay, timers=timers, **kwargs)
        judges.append(judge)
        return judge

    rows: list[dict[str, Any]] = []
    runtime_start = time.perf_counter()
    try:
        with SelfConditioningMethodRuntime(
            setup=validated.method_setup,
            training=validated.training_texts,
            main_frame=validated.main_frame,
            max_concepts=config.source_expectations.max_concepts,
            max_new_tokens=config.source_expectations.output_length,
            master_seed=config.master_seed,
            device=config.device,
            extraction_batch_size=validated.extraction_batch_size,
            judge_factory=cast(JudgeFactory, judge_factory),
            judge_cache=config.judge_cache,
            judge_model="openai/gpt-5.6-luna",
            api_base_url=config.api_base_url,
            cache_only=True,
            timeout_seconds=config.worker_timeout_seconds,
            stderr_dir=output / "logs",
            nl_target_scores=validated.nl_target_scores,
        ) as runtime:
            runtime_seconds = time.perf_counter() - runtime_start
            for method in ("SelfConditioning", "SelfConditioningMasked", NL_METHOD):
                source_entries = (
                    validated.nl_attachment.steering_entries
                    if method == NL_METHOD and validated.nl_attachment is not None
                    else validated.steering_entries
                )
                factor = point_at_factor(source_entries, method, 6.0).factor
                concept, _workers, examples = _cell_rows(
                    validated, method=method, concept_id=0, factor=factor
                )
                with runtime.bind_cell(
                    method=method, concept_id=0, concept=concept, factor=factor, examples=examples
                ) as cell:
                    for pressure in (
                        None,
                        Pressure.DIFFMEAN_COUNTER,
                        Pressure.LINEAR_PROBE_COUNTER,
                    ):
                        counter = (
                            None
                            if pressure is None
                            else load_counter_direction(
                                pressure=pressure,
                                method=method,
                                concept_id=0,
                                baseline_train=config.source_paths.main_parquet.parent.parent
                                / "train",
                                baseline_layer=config.source_expectations.layer,
                                project_fit=fit,
                            )
                        )
                        reference: tuple[RawEvaluation, ...] | None = None
                        for mode in modes:
                            runtime._judge_inflight = mode
                            judges[-1].reset()
                            timers.generation_seconds = timers.judge_seconds = 0.0
                            timers.calls = timers.cache_hits = 0
                            timers.generation_index = 0
                            timers.trace = []
                            t.cuda.reset_peak_memory_stats(config.device)
                            start = time.perf_counter()
                            ids = list(range(args.input_count))
                            result = (
                                cell.evaluate("", ids)
                                if counter is None
                                else cell.evaluate_counter(counter, args.magnitude, ids)
                            )
                            t.cuda.synchronize()
                            wall = time.perf_counter() - start
                            if reference is None:
                                reference = result
                            elif result != reference:
                                raise AssertionError(
                                    f"serial/overlap parity failed: {method}/{pressure}"
                                )
                            rows.append(
                                {
                                    "method": method,
                                    "pressure": "control" if pressure is None else str(pressure),
                                    "mode": mode,
                                    "wall_seconds": wall,
                                    "generation_seconds": timers.generation_seconds,
                                    "judge_seconds": timers.judge_seconds,
                                    "outputs_per_second": len(result) / wall,
                                    "judge_calls": timers.calls,
                                    "cache_hits": timers.cache_hits,
                                    "paid_misses": sum(
                                        row.paid_judge_cache_misses for row in result
                                    ),
                                    "peak_cuda_allocated_bytes": t.cuda.max_memory_allocated(
                                        config.device
                                    ),
                                    "peak_host_rss_kib": resource.getrusage(
                                        resource.RUSAGE_SELF
                                    ).ru_maxrss,
                                    "generations_sha256": hashlib.sha256(
                                        json.dumps(
                                            [row.generation for row in result], ensure_ascii=False
                                        ).encode()
                                    ).hexdigest(),
                                    "trace": timers.trace,
                                }
                            )
    finally:
        generation_patch.stop()
    summary = []
    serial_total = 0.0
    overlap_total = 0.0
    if args.mode_only is None:
        for method in ("SelfConditioning", "SelfConditioningMasked", NL_METHOD):
            for pressure in (
                "control",
                str(Pressure.DIFFMEAN_COUNTER),
                str(Pressure.LINEAR_PROBE_COUNTER),
            ):
                matched = [
                    row for row in rows if row["method"] == method and row["pressure"] == pressure
                ]
                if [row["mode"] for row in matched] != [1, 2, 2, 1]:
                    raise AssertionError("paired benchmark mode order changed")
                cold_serial = matched[0]["wall_seconds"]
                # The first control can compile kernels; compare the warmed final
                # serial replay with the two intervening overlapped replays.
                serial = matched[-1]["wall_seconds"]
                overlap = statistics.median(
                    row["wall_seconds"] for row in matched if row["mode"] == 2
                )
                serial_total += serial
                overlap_total += overlap
                summary.append(
                    {
                        "method": method,
                        "pressure": pressure,
                        "cold_serial_seconds": cold_serial,
                        "serial_seconds": serial,
                        "overlap_seconds": overlap,
                        "speedup": serial / overlap,
                    }
                )
    report = {
        "config": str(args.config.resolve()),
        "device": t.cuda.get_device_name(config.device),
        "source_seconds": source_seconds,
        "direction_preflight_seconds": fit_seconds,
        "runtime_startup_seconds": runtime_seconds,
        "judge_delay_seconds_per_response": args.judge_delay,
        "input_count": args.input_count,
        "magnitude": args.magnitude,
        "mode_only": args.mode_only,
        "benchmark_wall_seconds": time.perf_counter() - benchmark_start,
        "rows": rows,
        "summary": summary,
        "overall_cell_speedup": serial_total / overlap_total if args.mode_only is None else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
