"""Replay AxBench-Adv HF gradients/scores in the vendored Python environment.

This benchmark consumes prepared training rows from the production driver. It
does not generate completions or call a provider. Compare separate revisions
using the same input JSON, physical batches, environment, and CUDA settings.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import torch as t
from axbench.evaluators.judge_cache import JudgeCache
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts.benchmark.gcg_paired import measure_paired
from scripts.eval.axbench_adv.vendored import gcg_worker
from scripts.eval.axbench_adv.vendored.worker import WorkerRuntime
from tests.axbench_adv.gcg_state_reference import (
    preserve_gcg_model_state as reference_model_state,
)


def measure(operation: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, Any]]:
    samples = []
    result = None
    for _ in range(repeats):
        t.cuda.synchronize()
        start = time.perf_counter()
        result = operation()
        t.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return result, {"seconds": samples, "median_seconds": statistics.median(samples)}


def _numerics(runtime: WorkerRuntime, value: dict[str, Any]) -> t.Tensor:
    if "gradient_sidecar" in value:
        directory = runtime._gcg_sidecar_dir
        if directory is None:
            raise ValueError("paired GCG gradients require a sidecar directory")
        return load_file(directory / "gradients.safetensors")["gradients"]
    return t.tensor(value["losses"], dtype=t.float64)


def _measure_paired(
    operation: Callable[[], Any], repeats: int, *, runtime: WorkerRuntime, warmups: int
) -> tuple[Any, dict[str, Any]]:
    return measure_paired(
        operation,
        repeats,
        reference=lambda: patch.object(
            gcg_worker, "preserve_gcg_model_state", reference_model_state
        ),
        numerics=partial(_numerics, runtime),
        warmups=warmups,
    )


@hydra.main(
    version_base=None, config_path="../../configs/eval", config_name="benchmark_axbench_adv_gcg_hf"
)
def main(args: DictConfig) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    if not t.cuda.is_available():
        raise ValueError("this benchmark requires CUDA")
    for name in ("members", "batch_size", "repeats", "warmups"):
        if type(args[name]) is not int or args[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    t.use_deterministic_algorithms(args.deterministic)
    t.manual_seed(args.seed)
    input_path = Path(args.input)
    data = json.loads(input_path.read_text())
    source = data["source"]
    if args.members > len(data["rows"]):
        raise ValueError("members exceeds the prepared training population")
    cache = output_dir / "judge.sqlite3"
    with JudgeCache(cache):
        pass
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "git_sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": platform.python_version(),
        "torch": str(t.__version__),
        "cuda": t.version.cuda,
        "gpu": t.cuda.get_device_name(),
        "deterministic": args.deterministic,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "config": OmegaConf.to_container(args, resolve=True),
        "packages": {
            name: importlib.metadata.version(name) for name in ("transformers", "pyvene", "peft")
        },
        "gcg_worker_sha256": hashlib.sha256(
            Path("scripts/eval/axbench_adv/vendored/gcg_worker.py").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "scripts/benchmark/benchmark_axbench_adv_gcg_hf.py",
                "scripts/benchmark/gcg_paired.py",
                "scripts/eval/axbench_adv/gcg_runtime_state.py",
                "tests/axbench_adv/gcg_state_reference.py",
            )
        },
        "measurements": [],
    }
    tensors = {}
    method = args.method
    sidecars = output_dir / method
    sidecars.mkdir(mode=0o700)
    payload = {
        "api_base_url": "http://127.0.0.1:1",
        "cache_only": True,
        "aux_dump_dir": source["aux_dump_dir"],
        "main_dump_dir": source["main_dump_dir"],
        "prepared_dir": source["prepared_dir"],
        "concept": data["concept"],
        "concept_id": args.concept_id,
        "device": "cuda:0",
        "factor": args.factor,
        "intervene_on_prompt": True,
        "intervention_type": "addition",
        "judge_cache": str(cache),
        "judge_model": "openai/gpt-5.6-luna",
        "layer": args.layer,
        "max_concepts": args.max_concepts,
        "method": method,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "output_length": 128,
        "steering_layers": [args.layer],
        "temperature": 1.0,
        "use_bf16": True,
        "worker_max_prompt_tokens": 1024,
        "gcg_sidecar_dir": str(sidecars),
    }
    runtime = WorkerRuntime(payload)
    try:
        measure_operation = (
            partial(_measure_paired, runtime=runtime, warmups=args.warmups)
            if args.paired_state
            else measure
        )
        for objective in args.objectives:
            for k in args.token_counts:
                case = f"{method}.{objective}.k{k}"
                prepared, prep_time = measure(
                    partial(
                        runtime.gcg_prepare,
                        {
                            "objective": objective,
                            "alpha": 0.01,
                            "rows": data["rows"],
                            "suffix_token_count": k,
                            "device_batch_size": args.batch_size,
                            "separator": "\n\n",
                        },
                    ),
                    1,
                )
                tokens = prepared["seed_token_ids"]
                alternative = runtime.tokenizer.encode(
                    " ".join(["?"] * k), add_special_tokens=False
                )
                candidates = [tokens, alternative]
                filtered = runtime.gcg_filter({"candidate_token_ids": candidates})
                if not all(candidate["valid"] for candidate in filtered["candidates"]):
                    raise ValueError(
                        f"benchmark candidates failed round-trip filtering: {filtered}"
                    )
                gradient_payload = {
                    "token_ids": tokens,
                    "member_indices": list(range(args.members)),
                }
                score_payload = {
                    "candidate_token_ids": candidates,
                    "member_indices": list(range(args.members)),
                    "candidate_eval_device_batch_size": args.batch_size,
                }
                gradient_operation = partial(runtime.gcg_gradient, gradient_payload)
                score_operation = partial(runtime.gcg_score, score_payload)
                gradient_operation()
                score_operation()
                t.cuda.reset_peak_memory_stats()
                _grad, grad_time = measure_operation(gradient_operation, args.repeats)
                with safe_open(
                    sidecars / "gradients.safetensors", framework="pt", device="cpu"
                ) as handle:
                    tensors[case + ".gradient"] = handle.get_tensor("gradients").clone()
                scores, score_time = measure_operation(score_operation, args.repeats)
                tensors[case + ".scores"] = t.tensor(scores["losses"], dtype=t.float32)
                record = {
                    "case": case,
                    "method": method,
                    "objective": objective,
                    "k": k,
                    "batch_size": args.batch_size,
                    "members": args.members,
                    "prepared_pairs": len(data["rows"]),
                    "tokens": tokens,
                    "candidates": candidates,
                    "filtered": filtered,
                    "gradient": grad_time,
                    "score": score_time,
                    "prepare": prep_time,
                    "peak_allocated": t.cuda.max_memory_allocated(),
                    "peak_reserved": t.cuda.max_memory_reserved(),
                    "attention": runtime.benchmark_model.model.config._attn_implementation,
                    "model_commit": getattr(
                        runtime.benchmark_model.model.config, "_commit_hash", None
                    ),
                }
                if args.profile and k == args.token_counts[0]:
                    for label, operation in [
                        ("gradient", gradient_operation),
                        ("score", score_operation),
                    ]:
                        with t.profiler.profile(
                            activities=[
                                t.profiler.ProfilerActivity.CPU,
                                t.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                            profile_memory=True,
                        ) as prof:
                            with t.profiler.record_function(f"gcg.{case}.{label}"):
                                operation()
                            t.cuda.synchronize()
                        prof.export_chrome_trace(str(output_dir / f"{case}.{label}.trace.json"))
                        events = prof.key_averages()
                        (output_dir / f"{case}.{label}.table.txt").write_text(
                            events.table(sort_by="self_cuda_time_total", row_limit=40)
                        )
                        record[label + "_profile"] = [
                            {
                                "name": event.key,
                                "count": event.count,
                                "self_cpu_us": event.self_cpu_time_total,
                                "self_device_us": event.self_device_time_total,
                                "device_us": event.device_time_total,
                            }
                            for event in events
                        ]
                report["measurements"].append(record)
                (output_dir / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False)
                )
                save_file(tensors, output_dir / "numerics.safetensors")
                logger.info(
                    json.dumps(
                        {
                            key: record[key]
                            for key in ["case", "gradient", "score", "peak_allocated"]
                        }
                    )
                )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
