"""Source validation through attacked frames, artifacts, aggregation, and plots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.eval.axbench_adv import source_contract as source_contract_module
from steering.eval.axbench_adv.aggregation import build_aggregate
from steering.eval.axbench_adv.artifacts import write_suffix_artifacts
from steering.eval.axbench_adv.attacked_prompts import ATTACK_COLUMN, build_attacked_frame
from steering.eval.axbench_adv.best_point import select_best_points
from steering.eval.axbench_adv.plotting import plot_concept_vs_fluency
from steering.eval.axbench_adv.prompt_surgery import QWEN_TERMINAL_MARKER
from steering.eval.axbench_adv.schemas import UNIVERSAL_SUFFIXES_SCHEMA, suffix_sha256
from steering.eval.axbench_adv.source_contract import (
    revalidate_source_hashes,
    validate_source_contract,
    write_source_contract,
)
from steering.eval.axbench_adv.types import METHODS, PLOT_EFFORT_LEVELS, QWEN_METHODS
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from tests.axbench_adv.unit.test_source_contract import _fixture


def _token_count(text: str) -> int:
    return len(text.encode("utf-8"))


def _judged(method: str, factor: float, level: int) -> list[dict[str, object]]:
    return [
        {
            "concept_id": concept_id,
            "results": {
                "LMJudgeEvaluator": {
                    method: {
                        "factor": [factor],
                        "relevance_concept_ratings": [1.5 - 0.05 * level - 0.1 * concept_id],
                        "fluency_ratings": [1.8 - 0.05 * concept_id],
                    }
                }
            },
        }
        for concept_id in range(2)
    ]


def test_phase2_vertical_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_contract_module,
        "load_method_setup",
        lambda *_args, **_kwargs: SimpleNamespace(
            marker="validated setup",
            n_sae_features=8,
            allowed_ids=None,
            feature_mask=None,
            output_selection=SamplingSelection(mean=2.0, std=2.0),
            output_negative_selection=NegativeSamplingSelection(mean=3.0, std=3.0),
        ),
    )
    monkeypatch.setattr(
        source_contract_module,
        "select_training_texts",
        lambda *_args, **_kwargs: {
            concept_id: ConceptTrainingTexts(
                concept_id,
                f"concept-{concept_id}",
                "genre",
                (f"positive-{concept_id}",),
                (f"negative-{concept_id}",),
            )
            for concept_id in range(2)
        },
    )
    paths, expectations, hashes = _fixture(tmp_path)
    output_dir = tmp_path / "adv-output"
    validated = validate_source_contract(
        paths=paths,
        expectations=expectations,
        expected_hashes=hashes,
        output_dir=output_dir,
    )
    write_source_contract(validated, output_dir / "source_contract.json")
    best = select_best_points(validated.steering_entries)
    assert tuple(best) == METHODS

    for method in QWEN_METHODS:
        point = best[method]
        source = (
            validated.aux_frame.copy()
            if method in {"LoRA", "PromptSteering"}
            else validated.main_frame.copy()
        )
        target = ATTACK_COLUMN[method]
        source[target] = [
            f"<|im_start|>user\nPrompt {concept_id}/{input_id}." + QWEN_TERMINAL_MARKER
            for concept_id, input_id in zip(source["concept_id"], source["input_id"], strict=True)
        ]
        for level in PLOT_EFFORT_LEVELS:
            suffixes = {
                concept_id: "" if level == 0 else f"Avoid concept {concept_id}."
                for concept_id in range(2)
            }
            attacked = build_attacked_frame(
                source_frame=source,
                suffixes=suffixes,
                method=method,
                factor=point.factor,
                proposal_count=level,
                token_count=_token_count,
                max_prompt_tokens=10_000,
            )
            assert attacked.shape[0] == 20
            assert attacked[target].str.endswith(QWEN_TERMINAL_MARKER).all()

    judged = {
        (method, level): _judged(method, best[method].factor, level)
        for method in METHODS
        for level in PLOT_EFFORT_LEVELS
    }
    aggregate = build_aggregate(
        best_factors={method: point.factor for method, point in best.items()},
        judged_points=judged,
        expected_n_concepts=expectations.max_concepts,
        output_path=output_dir / "aggregate.parquet",
    )
    assert aggregate.num_rows == len(METHODS) * len(PLOT_EFFORT_LEVELS)
    assert len(plot_concept_vs_fluency(aggregate, output_dir / "plots")) == 1 + len(METHODS)

    suffix = "Avoid concept 0."
    suffixes = pa.Table.from_pylist(
        [
            {
                "method": "DiffMean",
                "concept_id": 0,
                "proposal_count": 1,
                "coefficient": best["DiffMean"].factor,
                "suffix": suffix,
                "suffix_sha256": suffix_sha256(suffix),
                "master_seed": expectations.seed,
                "best_fitness": 0.5,
                "gepa_nominal_metric_calls": 1,
                "actual_evaluator_calls": 1,
                "actual_generation_calls": 1,
                "paid_judge_cache_misses": 2,
                "paid_reflection_cache_misses": 1,
            }
        ],
        schema=UNIVERSAL_SUFFIXES_SCHEMA,
    )
    revalidate_source_hashes(validated)
    suffix_path, manifest_path = write_suffix_artifacts(
        suffixes=suffixes,
        output_dir=output_dir / "cells/diffmean/0/1",
    )
    assert suffix_path.is_file() and manifest_path.is_file()
