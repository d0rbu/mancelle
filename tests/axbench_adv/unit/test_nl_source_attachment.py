"""Bounded c100-to-c50 NL attachment validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch as t

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.eval.axbench_adv import source_contract as contract
from steering.eval.axbench_adv.best_point import select_best_points
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS, ALL_METHOD_FACTORS
from steering.eval.axbench_adv.source_contract import (
    NLAttachmentExpectations,
    NLAttachmentPaths,
    SourceExpectations,
    SourcePaths,
)
from steering.eval.axbench_adv.types import NL_METHOD
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from tests.axbench_adv.unit import test_source_contract as base

PROJECT_METHODS = (
    "SelfConditioning",
    "SelfConditioningMasked",
    "SelfConditioningPrompt",
    NL_METHOD,
)
JUDGED_METHODS = (
    "DiffMean", "LinearProbe", "PromptSteering", "LoRA", *PROJECT_METHODS,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(concept_id: int, input_id: int, factor: float) -> dict[str, object]:
    return base._input_values(concept_id, input_id, factor)


def _frame() -> pd.DataFrame:
    rows = []
    for concept_id in range(100):
        for input_id in range(10):
            for factor in ADV_FULL_FACTORS:
                row = _input(concept_id, input_id, factor)
                for method in PROJECT_METHODS:
                    row[f"{method}_steered_generation"] = f"{method}-{concept_id}-{input_id}-{factor}"
                    row[f"{method}_strength"] = float(factor)
                rows.append(row)
    columns = [*INPUT_COLUMNS]
    for method in PROJECT_METHODS:
        columns.extend((f"{method}_steered_generation", f"{method}_strength"))
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("concept_id", "input_id", "sae_id"):
        frame[column] = frame[column].astype("int64")
    for column in ("factor", *(f"{method}_strength" for method in PROJECT_METHODS)):
        frame[column] = frame[column].astype("float64")
    return frame


def _block(method: str, *, second_half: bool) -> dict[str, object]:
    factors = ADV_FULL_FACTORS if method == "SelfConditioningPrompt" else ALL_METHOD_FACTORS[method]
    concept = [0.0] * len(factors)
    fluency = [2.0] * len(factors)
    winner = len(factors) - 1 if second_half else 0
    concept[winner] = 2.0 if second_half else 1.0
    declared = [factor in factors for _input in range(10) for factor in ADV_FULL_FACTORS]
    block: dict[str, object] = {
        "factor": list(factors),
        "lm_judge_rating": [1.0] * len(factors),
        "relevance_concept_ratings": concept,
        "relevance_instruction_ratings": [1.0] * len(factors),
        "fluency_ratings": fluency,
    }
    raw_summaries = {
        "raw_aggregated_ratings": block["lm_judge_rating"],
        "raw_relevance_concept_ratings": block["relevance_concept_ratings"],
        "raw_relevance_instruction_ratings": block["relevance_instruction_ratings"],
        "raw_fluency_ratings": block["fluency_ratings"],
    }
    factor_indices = {factor: index for index, factor in enumerate(factors)}
    for field, summaries in raw_summaries.items():
        block[field] = [
            summaries[factor_indices[factor]] if included else None
            for included, factor in zip(
                declared,
                ADV_FULL_FACTORS * 10,
                strict=True,
            )
        ]
    for field in contract._COMPLETION_FIELDS:
        block[field] = ["Rating: [[1]]" if included else None for included in declared]
    return block


def _entries() -> list[dict[str, Any]]:
    return [
        {
            "concept_id": concept_id,
            "results": {"LMJudgeEvaluator": {
                method: _block(method, second_half=concept_id >= 50)
                for method in JUDGED_METHODS
            }},
        }
        for concept_id in range(100)
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "attachment"
    frame_path = root / "inference" / "steering_data.parquet"
    frame_path.parent.mkdir(parents=True)
    full_frame = _frame()
    full_frame.to_parquet(frame_path, index=False)
    base_main = full_frame.loc[full_frame["concept_id"].lt(50), list(INPUT_COLUMNS)].copy()
    description, concepts = root / "description.npz", root / "concept.npz"
    description.write_bytes(b"description")
    concepts.write_bytes(b"concepts")
    space = {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "revision": "7" * 40,
        "encoding": "fixture-normalized-float32",
        "library_versions": {"torch": "fixture"},
    }
    source = {
        "algorithm": "unit_description_cosine_lower_median_maxabs_v1",
        "feature_contract": {"fixture": True},
        "concepts": [
            {"concept_id": concept_id, "text": f"concept-{concept_id}"}
            for concept_id in range(100)
        ],
        "description_embeddings": {
            "path": description.name, "sha256": _sha(description),
            "source_texts_sha256": "8" * 64, "embedding_space": space,
        },
        "concept_embeddings": {
            "path": concepts.name, "sha256": _sha(concepts),
            "source_texts_sha256": "9" * 64, "embedding_space": space,
        },
    }
    manifest_path = root / "run_manifest.json"
    _write_json(manifest_path, {"methods": {NL_METHOD: {
        "factors": list(ADV_FULL_FACTORS), "policy": contract.NL_STEERING_POLICY,
        "target_source": source,
    }}})
    evaluated = root / "evaluated" / "steering.jsonl"
    _write_jsonl(evaluated, _entries())
    runtime_tokenizer = tmp_path / "runtime" / "tokenizer" / "tokenizer.pkl"
    runtime_tokenizer.parent.mkdir(parents=True)
    runtime_tokenizer.write_bytes(b"runtime-tokenizer")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(runtime_tokenizer.parent.parent))
    paths = NLAttachmentPaths(frame_path, manifest_path, evaluated)
    base_paths = SourcePaths(
        tmp_path / "prepared", tmp_path / "main.parquet", tmp_path / "aux.parquet",
        tmp_path / "evaluated.jsonl", tmp_path / "project.parquet",
        tmp_path / "checkpoint" / "meta_001125.json", tmp_path / "statistics",
    )
    base_expectations = SourceExpectations(
        50, base.COMMIT, base.REVISION, base.MODEL, 42, 1.0, 128, "c" * 64,
        "d" * 40, base.PROJECT_COMMIT, "f" * 64, "qwen", 28,
        project_target_step=1125, project_target_model_tag="fixture-target",
    )
    monkeypatch.setattr(contract, "retrieval_feature_identity", lambda _setup: {"fixture": True})
    monkeypatch.setattr(
        contract, "_validate_training_texts",
        lambda **_kwargs: {
            concept_id: ConceptTrainingTexts(
                concept_id, f"concept-{concept_id}", "genre", ("positive",), ("negative",),
            )
            for concept_id in range(100)
        },
    )
    monkeypatch.setattr(
        contract, "build_retrieval_target_scores",
        lambda **_kwargs: {concept_id: t.tensor([float(concept_id)]) for concept_id in range(100)},
    )
    return {
        "paths": paths,
        "expectations": NLAttachmentExpectations(),
        "base_paths": base_paths,
        "base_expectations": base_expectations,
        "base_main_frame": base_main,
        "setup": SimpleNamespace(),
        "output_dir": tmp_path / "output",
        "reserved_paths": (),
    }


def _validate(case: dict[str, Any]):
    return contract._validate_nl_attachment(**case)


def test_attachment_selects_first_fifty_before_nl_factor_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    legacy_before = case["base_main_frame"].copy(deep=True)
    attachment = _validate(case)
    expected = _frame().loc[lambda frame: frame.concept_id.lt(50), [
        *INPUT_COLUMNS, f"{NL_METHOD}_steered_generation", f"{NL_METHOD}_strength",
    ]].reset_index(drop=True)
    pd.testing.assert_frame_equal(attachment.selected_frame, expected)
    assert set(attachment.target_scores) == set(range(50))
    for concept_id, target in attachment.target_scores.items():
        t.testing.assert_close(target, t.tensor([float(concept_id)]), rtol=0, atol=0)
    assert len(attachment.steering_entries) == 50
    for actual, original in zip(attachment.steering_entries, _entries()[:50], strict=True):
        assert actual == {
            "concept_id": original["concept_id"],
            "results": {"LMJudgeEvaluator": {
                NL_METHOD: original["results"]["LMJudgeEvaluator"][NL_METHOD],
            }},
        }
    assert select_best_points(attachment.steering_entries, methods=(NL_METHOD,))[NL_METHOD].factor == 0.0
    assert select_best_points(_entries(), methods=(NL_METHOD,))[NL_METHOD].factor != 0.0
    pd.testing.assert_frame_equal(case["base_main_frame"], legacy_before)


@pytest.mark.parametrize("checkpoint_copy", [False, True])
def test_attachment_records_actual_runtime_tokenizer_not_checkpoint_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_copy: bool,
) -> None:
    case = _case(tmp_path, monkeypatch)
    unused = case["base_paths"].target_meta.parent / "tokenizer.pkl"
    if checkpoint_copy:
        unused.parent.mkdir()
        unused.write_bytes(b"unused-and-different-tokenizer")
    runtime = tmp_path / "runtime" / "tokenizer" / "tokenizer.pkl"
    attachment = _validate(case)
    assert attachment.identity["tokenizer"] == {
        "name": "tokenizer.pkl", "sha256": _sha(runtime),
        "byte_length": runtime.stat().st_size,
    }
    runtime.write_bytes(b"new-runtime-tokenizer")
    changed = _validate(case)
    assert changed.identity["tokenizer"]["sha256"] == _sha(runtime)
    assert changed.identity["tokenizer"] != attachment.identity["tokenizer"]


def test_attachment_needs_no_resume_store_receipt_or_unused_method_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    before = _validate(case)
    manifest = json.loads(case["paths"].project_manifest.read_text())
    manifest.update({
        "schema_version": 999, "repository_git_commit": "different",
        "repository_git_worktree": {"dirty": True}, "package_lock_sha256": "different",
        "configuration": {"device_type": "different"},
    })
    manifest["methods"]["SelfConditioningPrompt"] = {"policy": "unconsumed"}
    _write_json(case["paths"].project_manifest, manifest)
    after = _validate(case)
    pd.testing.assert_frame_equal(before.selected_frame, after.selected_frame)
    assert before.steering_entries == after.steering_entries
    assert not (case["paths"].project_manifest.parent / "_project_cell_resume_v1").exists()
    assert set(case["paths"].hashed_files()) == {
        "project_parquet", "project_manifest", "evaluated_jsonl",
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("input", "selected inputs differ"),
        ("missing_frame_row", "project frame failed"),
        ("duplicate_frame_row", "project frame failed"),
        ("missing_generation", "project frame failed"),
        ("missing_judged", "each concept exactly once"),
        ("duplicate_judged", "each concept exactly once"),
        ("missing_nl_judged", "missing NLSteering"),
        ("missing_nl_manifest", "missing NLSteering"),
        ("retrieval", "feature contract"),
        ("nl_policy", "NLSteering policy or factor grid differs"),
        ("rating_nan", "non-standard JSON numeric constant 'NaN'"),
        ("rating_outside_range", r"\[0, 2\]"),
        ("missing_rating", "wrong length"),
    ],
)
def test_attachment_rejects_scientifically_invalid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    if mutation == "input":
        case["base_main_frame"].loc[0, "original_prompt"] = "changed"
    elif mutation in {"missing_frame_row", "duplicate_frame_row", "missing_generation"}:
        frame = pd.read_parquet(case["paths"].project_parquet)
        if mutation == "missing_frame_row":
            frame = frame.iloc[:-1]
        elif mutation == "duplicate_frame_row":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        else:
            frame = frame.drop(columns=[f"{NL_METHOD}_steered_generation"])
        frame.to_parquet(case["paths"].project_parquet, index=False)
    elif mutation in {"retrieval", "nl_policy", "missing_nl_manifest"}:
        manifest = json.loads(case["paths"].project_manifest.read_text())
        if mutation == "retrieval":
            manifest["methods"][NL_METHOD]["target_source"]["feature_contract"] = {"wrong": True}
        elif mutation == "nl_policy":
            manifest["methods"][NL_METHOD]["policy"] = "wrong"
        else:
            del manifest["methods"][NL_METHOD]
        _write_json(case["paths"].project_manifest, manifest)
    else:
        entries = _entries()
        if mutation == "missing_judged":
            entries = entries[:-1]
        elif mutation == "duplicate_judged":
            entries[-1] = entries[-2]
        elif mutation == "missing_nl_judged":
            del entries[0]["results"]["LMJudgeEvaluator"][NL_METHOD]
        else:
            block = entries[0]["results"]["LMJudgeEvaluator"][NL_METHOD]
            if mutation == "rating_nan":
                block["raw_fluency_ratings"][0] = float("nan")
            elif mutation == "rating_outside_range":
                block["raw_fluency_ratings"][0] = 3.0
            else:
                block["raw_fluency_ratings"].pop()
        _write_jsonl(case["paths"].evaluated_jsonl, entries)
    with pytest.raises((AxBenchAdvError, ValueError), match=match):
        _validate(case)


def test_attachment_ignores_unselected_method_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    entries = _entries()
    entries[0]["results"]["LMJudgeEvaluator"]["SelfConditioningPrompt"] = {"unused": None}
    _write_jsonl(case["paths"].evaluated_jsonl, entries)
    assert len(_validate(case).steering_entries) == 50


@pytest.mark.parametrize(
    ("single_source_nl", "source_count", "message"),
    [(True, 100, "single-source NL"), (False, 49, "fewer concepts")],
)
def test_attachment_rejects_conflicting_source_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    single_source_nl: bool, source_count: int, message: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["base_expectations"] = replace(
        case["base_expectations"], include_nl_steering=single_source_nl,
    )
    case["expectations"] = NLAttachmentExpectations(max_concepts=source_count)
    with pytest.raises(AxBenchAdvError, match=message):
        _validate(case)


@pytest.mark.parametrize("concept_id", [0, 99])
def test_attachment_rejects_strength_factor_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concept_id: int,
) -> None:
    case = _case(tmp_path, monkeypatch)
    frame = pd.read_parquet(case["paths"].project_parquet)
    frame.loc[frame.concept_id.eq(concept_id), f"{NL_METHOD}_strength"] += 0.25
    frame.to_parquet(case["paths"].project_parquet, index=False)
    with pytest.raises(AxBenchAdvError, match="strength values differ from factors"):
        _validate(case)


@pytest.mark.parametrize("surface", ["retrieval", "source"])
def test_attachment_protects_inputs_from_output_and_cache_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    if surface == "retrieval":
        reserved = case["paths"].project_manifest.parent / "description.npz"
    else:
        reserved = tmp_path / "provider-cache.parquet"
        reserved.hardlink_to(case["paths"].project_parquet)
    case["reserved_paths"] = (reserved,)
    with pytest.raises(AxBenchAdvError, match="path-disjoint"):
        _validate(case)
