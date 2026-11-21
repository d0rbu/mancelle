"""Pair complete project runtime startup with full and selected DiffMean fits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.benchmark.benchmark_axbench_adv_judge_pair import _LocalJudgeAPI
from scripts.eval.axbench_adv.pressures import Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.best_point import point_at_factor
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_counter_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.pipeline import _cell_rows
from steering.eval.axbench_adv.self_conditioning import (
    SelfConditioningMethodRuntime,
    _prompt_rng,
)
from steering.eval.axbench_adv.source_contract import validate_source_contract
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept-id", type=int, required=True)
    parser.add_argument("--fit-concept-ids", type=int, nargs="+", required=True)
    parser.add_argument("--input-count", type=int, default=10)
    args = parser.parse_args()
    if not t.cuda.is_available():
        parser.error("requires CUDA")
    if not 1 <= args.input_count <= 10:
        parser.error("input-count must be in 1..10")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = OmegaConf.load(args.config)
    if not isinstance(raw, DictConfig):
        raise ValueError("config must be a mapping")
    config = build_config(raw)
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
    if args.concept_id not in validated.training_texts:
        raise ValueError("concept ID is absent from validated source")
    fit_concept_ids = tuple(args.fit_concept_ids)
    if (
        args.concept_id not in fit_concept_ids
        or len(set(fit_concept_ids)) != len(fit_concept_ids)
        or not set(fit_concept_ids).issubset(validated.training_texts)
    ):
        raise ValueError(
            "fit concept IDs must be unique source IDs including the evaluated concept"
        )
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
    counter = load_counter_direction(
        pressure=Pressure.DIFFMEAN_COUNTER,
        method="SelfConditioningMasked",
        concept_id=args.concept_id,
        baseline_train=config.source_paths.main_parquet.parent.parent / "train",
        baseline_layer=config.source_expectations.layer,
        project_fit=fit,
    )
    factor = point_at_factor(validated.steering_entries, "SelfConditioningMasked", 6.0).factor
    concept, _workers, examples = _cell_rows(
        validated, method="SelfConditioningMasked", concept_id=args.concept_id, factor=factor
    )
    rows: list[dict[str, object]] = []
    baseline_directions: dict[int, t.Tensor] | None = None
    baseline_result: object = None
    prompt_rng = _prompt_rng(validated.method_setup)
    initial_prompt_rng_state = prompt_rng.get_state().clone()
    with _LocalJudgeAPI(0.0) as local_api:
        for name in tuple(os.environ):
            if name.lower().endswith("_proxy") and name.lower() != "no_proxy":
                del os.environ[name]
        os.environ["OPENROUTER_BASE_URL"] = local_api.base_url
        os.environ["OPENROUTER_API_KEY"] = "offline-benchmark"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        for ordinal, scope in enumerate(("full", "selected", "selected", "full")):
            prompt_rng.set_state(initial_prompt_rng_state.clone())
            gc.collect()
            t.cuda.empty_cache()
            t.cuda.synchronize()
            t.cuda.reset_peak_memory_stats(config.device)
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
                judge_cache=output / f"cache-{ordinal}.sqlite",
                judge_model="openai/gpt-5.6-luna",
                api_base_url=local_api.base_url,
                cache_only=False,
                timeout_seconds=max(float(config.worker_timeout_seconds), 600.0),
                stderr_dir=output / "logs" / str(ordinal),
                judge_inflight=2,
                judge_pair_concurrency=2,
                reuse_single_token_mask=True,
                reuse_diffmean_texts=True,
                fit_concept_ids=fit_concept_ids if scope == "selected" else None,
                nl_target_scores=validated.nl_target_scores,
            ) as runtime:
                t.cuda.synchronize()
                startup_seconds = time.perf_counter() - runtime_start
                if baseline_directions is None:
                    baseline_directions = {
                        cid: runtime._dense_diffmeans[cid].clone() for cid in fit_concept_ids
                    }
                exact_direction = all(
                    t.equal(runtime._dense_diffmeans[cid], baseline_directions[cid])
                    for cid in fit_concept_ids
                )
                local_api.reset()
                with runtime.bind_cell(
                    method="SelfConditioningMasked",
                    concept_id=args.concept_id,
                    concept=concept,
                    factor=factor,
                    examples=examples,
                ) as cell:
                    t.cuda.synchronize()
                    cell_start = time.perf_counter()
                    result = cell.evaluate_counter(counter, 256.0, list(range(args.input_count)))
                    t.cuda.synchronize()
                    cell_seconds = time.perf_counter() - cell_start
                if baseline_result is None:
                    baseline_result = result
                exact_result = result == baseline_result
                row: dict[str, object] = {
                    "scope": scope,
                    "startup_seconds": startup_seconds,
                    "cell_seconds": cell_seconds,
                    "exact_direction": exact_direction,
                    "exact_raw_evaluations": exact_result,
                    "prompt_rng_after_startup_sha256": hashlib.sha256(
                        prompt_rng.get_state().numpy().tobytes()
                    ).hexdigest(),
                    "generation_sha256": hashlib.sha256(
                        json.dumps(
                            [item.generation for item in result], ensure_ascii=False
                        ).encode()
                    ).hexdigest(),
                    "judge_calls": local_api.calls,
                    "peak_provider_concurrency": local_api.peak,
                    "paid_misses": sum(item.paid_judge_cache_misses for item in result),
                    "peak_cuda_allocated_bytes": t.cuda.max_memory_allocated(config.device),
                    "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
                rows.append(row)
                (output / "report.json").write_text(
                    json.dumps(
                        {
                            "device": t.cuda.get_device_name(config.device),
                            "concept_id": args.concept_id,
                            "fit_concept_ids": list(fit_concept_ids),
                            "input_count": args.input_count,
                            "source_seconds": source_seconds,
                            "rows": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(row), flush=True)
                if not exact_direction or not exact_result:
                    raise AssertionError(
                        "selected runtime changed fitted direction or judged output"
                    )


if __name__ == "__main__":
    main()
