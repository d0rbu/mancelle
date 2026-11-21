"""Real-GPU AxBench-Adv replay with the real judge worker and a local fake API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.eval.axbench_adv.pressures import Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.best_point import point_at_factor
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_counter_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.pipeline import _cell_rows
from steering.eval.axbench_adv.self_conditioning import SelfConditioningMethodRuntime
from steering.eval.axbench_adv.source_contract import validate_source_contract
from steering.eval.axbench_adv.types import NL_METHOD
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient


class _LocalJudgeAPI:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                size = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(size))
                prompt = request["messages"][0]["content"]
                concept = "[Concept Start]" in prompt
                with endpoint.lock:
                    endpoint.active += 1
                    endpoint.calls += 1
                    endpoint.peak = max(endpoint.peak, endpoint.active)
                try:
                    time.sleep(endpoint.delay / 2)
                    completion = (
                        "concept reason Rating: [[1.5]]"
                        if concept
                        else "fluency reason Rating: [[1.25]]"
                    )
                    response = {
                        "id": "local-axbench-judge",
                        "object": "chat.completion",
                        "created": 0,
                        "model": request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": completion},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                    }
                    encoded = json.dumps(response).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                finally:
                    with endpoint.lock:
                        endpoint.active -= 1

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _LocalJudgeAPI:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def reset(self) -> None:
        with self.lock:
            if self.active:
                raise AssertionError("local judge still has an active request")
            self.calls = self.peak = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-count", type=int, default=10)
    parser.add_argument("--judge-delay", type=float, default=2.0)
    parser.add_argument("--magnitude", type=float, default=256.0)
    parser.add_argument(
        "--method",
        choices=("SelfConditioning", "SelfConditioningMasked", NL_METHOD),
        help="Replay one project method; default replays all three.",
    )
    parser.add_argument(
        "--pressure",
        choices=("control", str(Pressure.DIFFMEAN_COUNTER), str(Pressure.LINEAR_PROBE_COUNTER)),
        help="Replay one control or counter condition; default replays all three.",
    )
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-mask-cache",
        action="store_true",
        help="Compare packed mask reuse 0,1,1,0 with paired judging held fixed.",
    )
    comparison.add_argument(
        "--compare-stack",
        action="store_true",
        help="Compare serial defaults with generation overlap, paired judging, and mask reuse.",
    )
    comparison.add_argument(
        "--compare-worker-pool",
        action="store_true",
        help="Compare one versus two framed workers with other project options fixed.",
    )
    comparison.add_argument(
        "--compare-inflight-depth",
        action="store_true",
        help="Compare two versus three queued responses with two judge workers fixed.",
    )
    args = parser.parse_args()
    if not 1 <= args.input_count <= 10 or args.judge_delay < 0:
        parser.error("input-count must be in 1..10 and judge-delay must be nonnegative")
    methods = (
        (args.method,) if args.method else ("SelfConditioning", "SelfConditioningMasked", NL_METHOD)
    )
    if args.pressure is None:
        pressures = (None, Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER)
    elif args.pressure == "control":
        pressures = (None,)
    else:
        pressures = (Pressure(args.pressure),)
    if not t.cuda.is_available():
        parser.error("this benchmark requires a real CUDA device")
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
    rows: list[dict[str, Any]] = []
    with _LocalJudgeAPI(args.judge_delay) as local_api:
        for name in tuple(os.environ):
            if name.lower().endswith("_proxy") and name.lower() != "no_proxy":
                del os.environ[name]
        os.environ["OPENROUTER_BASE_URL"] = local_api.base_url
        os.environ["OPENROUTER_API_KEY"] = "offline-benchmark"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        runtime_start = time.perf_counter()
        with SelfConditioningMethodRuntime(
            setup=validated.method_setup,
            training=validated.training_texts,
            main_frame=validated.main_frame,
            max_concepts=config.source_expectations.max_concepts,
            max_new_tokens=config.source_expectations.output_length,
            master_seed=config.master_seed,
            device=config.device,
            extraction_batch_size=validated.extraction_batch_size,
            judge_factory=AxBenchAdvJudgeClient,
            judge_cache=output / "unused-cache.sqlite",
            judge_model="openai/gpt-5.6-luna",
            api_base_url=local_api.base_url,
            cache_only=False,
            timeout_seconds=max(float(config.worker_timeout_seconds), 600.0),
            stderr_dir=output / "logs",
            judge_inflight=2,
            reuse_diffmean_texts=(
                True
                if args.compare_worker_pool or args.compare_inflight_depth
                else config.project_reuse_diffmean_texts
            ),
            fit_concept_ids=(0,) if args.compare_worker_pool or args.compare_inflight_depth else None,
            nl_target_scores=validated.nl_target_scores,
        ) as runtime:
            runtime_seconds = time.perf_counter() - runtime_start
            for method in methods:
                source_entries = (
                    validated.nl_attachment.steering_entries
                    if method == NL_METHOD and validated.nl_attachment is not None
                    else validated.steering_entries
                )
                factor = point_at_factor(source_entries, method, 6.0).factor
                concept, _workers, examples = _cell_rows(
                    validated, method=method, concept_id=0, factor=factor
                )
                for pressure in pressures:
                    counter = (
                        None
                        if pressure is None
                        else load_counter_direction(
                            pressure=pressure,
                            method=method,
                            concept_id=0,
                            baseline_train=config.source_paths.main_parquet.parent.parent / "train",
                            baseline_layer=config.source_expectations.layer,
                            project_fit=fit,
                        )
                    )
                    reference: tuple[RawEvaluation, ...] | None = None
                    for ordinal, mode in enumerate((1, 2, 2, 1)):
                        pair_concurrency = (
                            2
                            if args.compare_mask_cache
                            or args.compare_worker_pool
                            or args.compare_inflight_depth
                            else mode
                        )
                        reuse_mask = args.compare_worker_pool or args.compare_inflight_depth or (
                            (args.compare_mask_cache or args.compare_stack) and mode == 2
                        )
                        runtime._judge_inflight = (
                            mode
                            if args.compare_stack
                            else (mode + 1 if args.compare_inflight_depth else 2)
                        )
                        runtime._judge_workers = (
                            mode if args.compare_worker_pool else (2 if args.compare_inflight_depth else 1)
                        )
                        runtime._judge_pair_concurrency = pair_concurrency
                        runtime._reuse_single_token_mask = reuse_mask
                        runtime._judge_cache = (
                            output / f"cache-{method}-{pressure}-{ordinal}.sqlite"
                        )
                        runtime._stderr_dir = output / "logs" / f"{method}-{pressure}-{ordinal}"
                        local_api.reset()
                        with runtime.bind_cell(
                            method=method,
                            concept_id=0,
                            concept=concept,
                            factor=factor,
                            examples=examples,
                        ) as cell:
                            t.cuda.reset_peak_memory_stats(config.device)
                            t.cuda.synchronize()
                            started = time.perf_counter()
                            started_unix = time.time()
                            ids = list(range(args.input_count))
                            result = (
                                cell.evaluate("", ids)
                                if counter is None
                                else cell.evaluate_counter(counter, args.magnitude, ids)
                            )
                            t.cuda.synchronize()
                            wall = time.perf_counter() - started
                            ended_unix = time.time()
                        if reference is None:
                            reference = result
                        elif result != reference:
                            raise AssertionError(
                                f"paired parity failed: {method}/{pressure}/{mode}"
                            )
                        rows.append(
                            {
                                "method": method,
                                "pressure": "control" if pressure is None else str(pressure),
                                "pair_concurrency": pair_concurrency,
                                **(
                                    {"judge_workers": runtime._judge_workers}
                                    if args.compare_worker_pool
                                    else {}
                                ),
                                **(
                                    {"judge_inflight": runtime._judge_inflight}
                                    if args.compare_stack or args.compare_inflight_depth
                                    else {}
                                ),
                                **(
                                    {"reuse_single_token_mask": reuse_mask}
                                    if args.compare_mask_cache or args.compare_stack
                                    else {}
                                ),
                                "wall_seconds": wall,
                                "started_unix": started_unix,
                                "ended_unix": ended_unix,
                                "generations_sha256": hashlib.sha256(
                                    json.dumps(
                                        [row.generation for row in result], ensure_ascii=False
                                    ).encode()
                                ).hexdigest(),
                                "judge_calls": local_api.calls,
                                "peak_provider_concurrency": local_api.peak,
                                "paid_misses": sum(row.paid_judge_cache_misses for row in result),
                                "peak_cuda_allocated_bytes": t.cuda.max_memory_allocated(
                                    config.device
                                ),
                                "peak_host_rss_kib": resource.getrusage(
                                    resource.RUSAGE_SELF
                                ).ru_maxrss,
                            }
                        )
    summary: list[dict[str, Any]] = []
    serial_total = 0.0
    parallel_total = 0.0
    for method in methods:
        for pressure_value in pressures:
            pressure = "control" if pressure_value is None else str(pressure_value)
            matched = [
                row for row in rows if row["method"] == method and row["pressure"] == pressure
            ]
            observed_order = [
                int(row["judge_workers"])
                if args.compare_worker_pool
                else (
                    int(row["judge_inflight"])
                    if args.compare_inflight_depth
                    else (
                        int(row["reuse_single_token_mask"])
                        if args.compare_mask_cache
                        else row["pair_concurrency"]
                    )
                )
                for row in matched
            ]
            expected_order = (
                [0, 1, 1, 0]
                if args.compare_mask_cache
                else ([2, 3, 3, 2] if args.compare_inflight_depth else [1, 2, 2, 1])
            )
            if observed_order != expected_order:
                raise AssertionError("paired order changed")
            serial = matched[-1]["wall_seconds"]
            parallel = statistics.median(row["wall_seconds"] for row in matched[1:3])
            serial_total += serial
            parallel_total += parallel
            summary.append(
                {
                    "method": method,
                    "pressure": pressure,
                    **(
                        {
                            "cold_baseline_seconds": matched[0]["wall_seconds"],
                            "warm_baseline_seconds": serial,
                            "candidate_seconds": parallel,
                        }
                        if args.compare_mask_cache
                        or args.compare_stack
                        or args.compare_worker_pool
                        or args.compare_inflight_depth
                        else {
                            "cold_serial_seconds": matched[0]["wall_seconds"],
                            "warm_serial_seconds": serial,
                            "parallel_seconds": parallel,
                        }
                    ),
                    "speedup": serial / parallel,
                }
            )
    report = {
        "config": str(args.config.resolve()),
        "device": t.cuda.get_device_name(config.device),
        "source_seconds": source_seconds,
        "runtime_startup_seconds": runtime_seconds,
        "judge_delay_seconds_per_response_if_sequential": args.judge_delay,
        "input_count": args.input_count,
        "magnitude": args.magnitude,
        **(
            {
                "comparison": (
                    "worker_pool"
                    if args.compare_worker_pool
                    else (
                        "inflight_depth"
                        if args.compare_inflight_depth
                        else ("mask_cache" if args.compare_mask_cache else "stack")
                    )
                )
            }
            if args.compare_mask_cache
            or args.compare_stack
            or args.compare_worker_pool
            or args.compare_inflight_depth
            else {}
        ),
        "rows": rows,
        "summary": summary,
        "overall_cell_speedup": serial_total / parallel_total,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
