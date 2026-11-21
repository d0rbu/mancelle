"""Compare model logits for one real-GPU latent counter response with mask reuse off/on."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

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


class _Judge:
    def __init__(self, **_: Any) -> None:
        pass

    def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
        return RawEvaluation(generation, 1.5, 1.25, "{}", 0)

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not t.cuda.is_available():
        parser.error("requires CUDA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = OmegaConf.load(args.config)
    if not isinstance(raw, DictConfig):
        raise ValueError("config must be a mapping")
    config = build_config(raw)
    validated = validate_source_contract(
        paths=config.source_paths,
        expectations=config.source_expectations,
        expected_hashes=config.source_hashes,
        output_dir=output,
        reserved_paths=(config.judge_cache, config.reflection_cache),
        nl_attachment_paths=config.nl_attachment_paths,
        nl_attachment_expectations=config.nl_attachment_expectations,
    )
    if validated.target_meta is None or config.project_counter_direction_root is None:
        raise ValueError("project target and directions are required")
    if config.project_counter_layer is None:
        raise ValueError("project counter layer is required")
    fit = preflight_project_direction_fit(
        config.project_counter_direction_root,
        ProjectCounterExpectations(
            target_model_sha256=validated.runtime_hashes["target_model"],
            target_metadata_sha256=validated.runtime_hashes["target_meta"],
            concept_texts={key: value.concept for key, value in validated.training_texts.items()},
            decoder_width=validated.target_meta["model_config"]["n_embd"],
            layer=config.project_counter_layer,
            fit_seed=config.project_counter_fit_seed,
        ),
    )
    method = "SelfConditioning"
    factor = point_at_factor(validated.steering_entries, method, 6.0).factor
    concept, _workers, examples = _cell_rows(
        validated, method=method, concept_id=0, factor=factor
    )
    direction = load_counter_direction(
        pressure=Pressure.DIFFMEAN_COUNTER,
        method=method,
        concept_id=0,
        baseline_train=config.source_paths.main_parquet.parent.parent / "train",
        baseline_layer=config.source_expectations.layer,
        project_fit=fit,
    )
    with SelfConditioningMethodRuntime(
        setup=validated.method_setup,
        training=validated.training_texts,
        main_frame=validated.main_frame,
        max_concepts=config.source_expectations.max_concepts,
        max_new_tokens=config.source_expectations.output_length,
        master_seed=config.master_seed,
        device=config.device,
        extraction_batch_size=validated.extraction_batch_size,
        judge_factory=cast(Any, _Judge),
        judge_cache=output / "unused-judge.sqlite",
        judge_model="openai/gpt-5.6-luna",
        api_base_url=config.api_base_url,
        cache_only=True,
        timeout_seconds=config.worker_timeout_seconds,
        stderr_dir=output / "logs",
        nl_target_scores=validated.nl_target_scores,
    ) as runtime:
        model = runtime._target_model
        captures: list[list[t.Tensor]] = []
        generations: list[str] = []
        with runtime.bind_cell(
            method=method,
            concept_id=0,
            concept=concept,
            factor=factor,
            examples=examples,
        ) as cell:
            for reuse in (False, True):
                runtime._reuse_single_token_mask = reuse
                captured: list[t.Tensor] = []
                original_forward = model.forward

                def capture_forward(
                    *call_args: Any,
                    _forward: Any = original_forward,
                    _captured: list[t.Tensor] = captured,
                    **call_kwargs: Any,
                ) -> Any:
                    result = _forward(*call_args, **call_kwargs)
                    if (
                        tuple(call_args[0].shape) == (1, 1)
                        and not call_kwargs.get("fp_logits_only", False)
                        and not call_kwargs.get("fp_hidden_only", False)
                    ):
                        logits = result[0] if isinstance(result, tuple) else result
                        if isinstance(logits, t.Tensor) and logits.shape[-1] == model.dec_config.vocab_size:
                            _captured.append(logits.detach().cpu().clone())
                    return result

                with patch.object(model, "forward", capture_forward):
                    evaluation = cell.evaluate_counter(direction, 256.0, [0])[0]
                captures.append(captured)
                generations.append(evaluation.generation)
    if len(captures[0]) != len(captures[1]) or not captures[0]:
        raise AssertionError("decode logit capture lengths changed")
    baseline = t.cat(captures[0], dim=1)
    candidate = t.cat(captures[1], dim=1)
    if baseline.shape != candidate.shape:
        raise AssertionError("decode logit shapes changed")
    difference = (baseline - candidate).abs()
    maximum = float(difference.max())
    close = bool(t.allclose(baseline, candidate, rtol=1e-5, atol=1e-5))
    report = {
        "source": "1ed472d5139d6b2afe7e80969acccd32a93ef329",
        "device": t.cuda.get_device_name(config.device),
        "method": method,
        "pressure": "diffmean_counter",
        "input_id": 0,
        "magnitude": 256.0,
        "logit_shape": list(baseline.shape),
        "maximum_absolute_logit_difference": maximum,
        "exact_logits": bool(t.equal(baseline, candidate)),
        "within_1e-5": close,
        "exact_generation": generations[0] == generations[1],
        "captured_at": time.time(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not close or generations[0] != generations[1]:
        raise AssertionError("mask cache changed logits or generated text")


if __name__ == "__main__":
    main()
