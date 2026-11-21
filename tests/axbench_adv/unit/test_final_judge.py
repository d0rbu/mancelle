"""Tests for sparse final frames and exact inner/final judge parity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from steering.eval.axbench_adv import final_judge as final_judge_module
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import (
    POINT_FRAME_SCHEMA,
    FinalJudgeResult,
    PersistentFinalJudge,
    build_point_frame,
    run_final_judge,
    validate_final_judge_outputs,
    validate_point_frame,
    write_point_frame,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.types import METHODS

FACTORS = {
    "DiffMean": 1.0,
    "LinearProbe": 3.0,
    "LoRA": 1.0,
    "PromptSteering": 1.0,
    "SelfConditioning": 6.0,
    "SelfConditioningMasked": 5.0,
}
SUFFIXES = (
    "LMJudgeEvaluator",
    "LMJudgeEvaluator_relevance_concept_ratings",
    "LMJudgeEvaluator_relevance_concept_completions",
    "LMJudgeEvaluator_relevance_instruction_ratings",
    "LMJudgeEvaluator_relevance_instruction_completions",
    "LMJudgeEvaluator_fluency_ratings",
    "LMJudgeEvaluator_fluency_completions",
)


def _source(max_concepts: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "AlpacaEval",
                "concept_id": concept_id,
                "input_concept": f"concept-{concept_id}",
                "input_id": input_id,
                "factor": 1.0,
                "original_prompt": f"prompt {concept_id}/{input_id}",
            }
            for concept_id in range(max_concepts)
            for input_id in range(10)
        ]
    )


def _raw(
    method: str, concept_id: int, input_id: int, *, generation: str | None = None
) -> RawEvaluation:
    concept = 0.25 + 0.25 * concept_id
    fluency = 1.5 + 0.05 * (input_id % 2)
    concept_text = f"{method} concept rationale c{concept_id} i{input_id} Rating: [[{concept}]]"
    fluency_text = f"{method} fluency rationale c{concept_id} i{input_id} Rating: [[{fluency}]]"
    return RawEvaluation(
        generation if generation is not None else f"{method} generated c{concept_id} i{input_id}",
        concept,
        fluency,
        json.dumps(
            {"concept": concept_text, "fluency": fluency_text},
            sort_keys=True,
            separators=(",", ":"),
        ),
        0,
    )


def _emissions(max_concepts: int = 2) -> dict[tuple[str, int, int], RawEvaluation]:
    return {
        (method, concept_id, input_id): _raw(method, concept_id, input_id)
        for method in METHODS
        for concept_id in range(max_concepts)
        for input_id in range(10)
    }


def _aggregate(concept: float, instruction: float, fluency: float) -> float:
    values = (concept, instruction, fluency)
    return 0.0 if 0.0 in values else 3.0 / sum(1.0 / value for value in values)


def _outputs(
    output_dir: Path,
    table: pa.Table,
    emissions: dict[tuple[str, int, int], RawEvaluation],
    *,
    max_concepts: int = 2,
    factors: dict[str, float] = FACTORS,
    concept_ids: tuple[int, ...] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = table.to_pandas()
    evaluated = frame.copy()
    entries: list[dict[str, object]] = []
    per_method_columns: dict[str, dict[str, list[object]]] = {
        method: {suffix: [] for suffix in SUFFIXES} for method in factors
    }
    for concept_id in range(max_concepts) if concept_ids is None else concept_ids:
        concept_frame = frame.loc[frame["concept_id"].eq(concept_id)]
        blocks: dict[str, dict[str, object]] = {}
        for method, factor in factors.items():
            raw_concept: list[float | None] = []
            raw_instruction: list[float | None] = []
            raw_fluency: list[float | None] = []
            raw_aggregate: list[float | None] = []
            concept_text: list[str | None] = []
            instruction_text: list[str | None] = []
            fluency_text: list[str | None] = []
            for row in concept_frame.itertuples(index=False):
                if float(row.factor) != factor:
                    raw_concept.append(None)
                    raw_instruction.append(None)
                    raw_fluency.append(None)
                    raw_aggregate.append(None)
                    concept_text.append(None)
                    instruction_text.append(None)
                    fluency_text.append(None)
                    continue
                raw = emissions[(method, concept_id, int(row.input_id))]
                rationale = json.loads(raw.judge_rationale)
                instruction = 1.25
                raw_concept.append(raw.concept_rating)
                raw_instruction.append(instruction)
                raw_fluency.append(raw.fluency_rating)
                raw_aggregate.append(
                    _aggregate(raw.concept_rating, instruction, raw.fluency_rating)
                )
                concept_text.append(rationale["concept"])
                instruction_text.append(
                    f"instruction c{concept_id} i{row.input_id} Rating: [[1.25]]"
                )
                fluency_text.append(rationale["fluency"])
            def present(values: list[float | None]) -> list[float]:
                return [value for value in values if value is not None]

            blocks[method] = {
                "factor": [factor],
                "lm_judge_rating": [sum(present(raw_aggregate)) / 10],
                "relevance_concept_ratings": [sum(present(raw_concept)) / 10],
                "relevance_instruction_ratings": [sum(present(raw_instruction)) / 10],
                "fluency_ratings": [sum(present(raw_fluency)) / 10],
                "raw_relevance_concept_ratings": raw_concept,
                "raw_relevance_instruction_ratings": raw_instruction,
                "raw_fluency_ratings": raw_fluency,
                "raw_aggregated_ratings": raw_aggregate,
                "relevance_concept_completions": concept_text,
                "relevance_instruction_completions": instruction_text,
                "fluency_completions": fluency_text,
            }
            columns = per_method_columns[method]
            columns["LMJudgeEvaluator"].extend(raw_aggregate)
            columns["LMJudgeEvaluator_relevance_concept_ratings"].extend(raw_concept)
            columns["LMJudgeEvaluator_relevance_concept_completions"].extend(concept_text)
            columns["LMJudgeEvaluator_relevance_instruction_ratings"].extend(raw_instruction)
            columns["LMJudgeEvaluator_relevance_instruction_completions"].extend(instruction_text)
            columns["LMJudgeEvaluator_fluency_ratings"].extend(raw_fluency)
            columns["LMJudgeEvaluator_fluency_completions"].extend(fluency_text)
        entries.append({"concept_id": concept_id, "results": {"LMJudgeEvaluator": blocks}})
    for method in factors:
        for suffix in SUFFIXES:
            evaluated[f"{method}_{suffix}"] = per_method_columns[method][suffix]
    evaluated.to_parquet(output_dir / "steering_data.parquet", index=False)
    (output_dir / "steering.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


def _entries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_entries(path: Path, entries: Sequence[object]) -> None:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def _block(output: Path, method: str = "DiffMean") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = _entries(output / "steering.jsonl")
    return entries, entries[0]["results"]["LMJudgeEvaluator"][method]


def test_build_sparse_point_frame_has_exact_union_grid_and_empty_generation_identity() -> None:
    emissions = _emissions()
    emissions[("DiffMean", 0, 0)] = _raw("DiffMean", 0, 0, generation="")
    table = build_point_frame(
        source_frame=_source(),
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=2,
    )
    assert table.schema.equals(POINT_FRAME_SCHEMA)
    assert table.num_rows == 80
    rows = table.to_pylist()
    assert [(row["concept_id"], row["input_id"], row["factor"]) for row in rows[:8]] == [
        (0, 0, 1.0),
        (0, 0, 3.0),
        (0, 0, 5.0),
        (0, 0, 6.0),
        (0, 1, 1.0),
        (0, 1, 3.0),
        (0, 1, 5.0),
        (0, 1, 6.0),
    ]
    assert rows[0]["DiffMean_steered_generation"] == ""
    assert rows[0]["LinearProbe_steered_generation"] is None
    assert rows[1]["LinearProbe_steered_generation"] == "LinearProbe generated c0 i0"
    assert rows[1]["DiffMean_steered_generation"] is None


def test_point_frame_rejects_missing_emission_and_wrong_availability() -> None:
    emissions = _emissions()
    emissions.pop(("LoRA", 1, 9))
    with pytest.raises(AxBenchAdvError, match="coverage"):
        build_point_frame(
            source_frame=_source(),
            best_factors=FACTORS,
            emissions=emissions,
            max_concepts=2,
        )
    valid = build_point_frame(
        source_frame=_source(),
        best_factors=FACTORS,
        emissions=_emissions(),
        max_concepts=2,
    )
    rows = valid.to_pylist()
    rows[0]["LinearProbe_steered_generation"] = "off grid"
    with pytest.raises(AxBenchAdvError, match="availability"):
        validate_point_frame(
            pa.Table.from_pylist(rows, schema=POINT_FRAME_SCHEMA),
            best_factors=FACTORS,
            max_concepts=2,
        )


def test_point_frame_rejects_invalid_dimensions_and_off_grid_factor() -> None:
    with pytest.raises(AxBenchAdvError, match="positive integer"):
        build_point_frame(
            source_frame=_source(),
            best_factors=FACTORS,
            emissions={},
            max_concepts=0,
        )
    with pytest.raises(AxBenchAdvError, match="declared grid"):
        build_point_frame(
            source_frame=_source(),
            best_factors={**FACTORS, "LoRA": 3.0},
            emissions=_emissions(),
            max_concepts=2,
        )


def test_point_frame_boundaries_fail_closed() -> None:
    for factors, match in (
        ({key: value for key, value in FACTORS.items() if key != "LoRA"}, "coverage differs"),
        ({**FACTORS, "LoRA": True}, "numeric"),
        ({**FACTORS, "LoRA": math.inf}, "finite"),
    ):
        with pytest.raises(AxBenchAdvError, match=match):
            build_point_frame(
                source_frame=_source(),
                best_factors=factors,
                emissions=_emissions(),
                max_concepts=2,
            )

    emissions: dict[tuple[str, int, int], object] = dict(_emissions())
    emissions[("LoRA", 0, 0)] = "not an evaluation"
    with pytest.raises(AxBenchAdvError, match="not a RawEvaluation"):
        build_point_frame(
            source_frame=_source(),
            best_factors=FACTORS,
            emissions=emissions,  # ty: ignore[invalid-argument-type]
            max_concepts=2,
        )

    source = _source()
    for malformed, match in (
        (source.drop(columns="dataset_name"), "missing"),
        (source.iloc[:-1], "exact concept/input product"),
        (source.assign(input_concept=None), "null values"),
    ):
        with pytest.raises(AxBenchAdvError, match=match):
            build_point_frame(
                source_frame=malformed,
                best_factors=FACTORS,
                emissions=_emissions(),
                max_concepts=2,
            )

    valid = build_point_frame(
        source_frame=source,
        best_factors=FACTORS,
        emissions=_emissions(),
        max_concepts=2,
    )
    with pytest.raises(AxBenchAdvError, match="positive integer"):
        validate_point_frame(valid, best_factors=FACTORS, max_concepts=False)
    with pytest.raises(AxBenchAdvError, match="POINT_FRAME_SCHEMA"):
        validate_point_frame(pa.table({"wrong": [1]}), best_factors=FACTORS, max_concepts=2)

    rows = valid.to_pylist()
    for position, field, value, match in (
        (0, "input_id", 9, "canonical union-factor product"),
        (0, "original_prompt", None, "base columns contain null"),
        (1, "original_prompt", "changed", "varies across factors"),
    ):
        mutated = [dict(row) for row in rows]
        mutated[position][field] = value
        with pytest.raises(AxBenchAdvError, match=match):
            validate_point_frame(
                pa.Table.from_pylist(mutated, schema=POINT_FRAME_SCHEMA),
                best_factors=FACTORS,
                max_concepts=2,
            )


def test_final_output_validator_rejects_malformed_vendor_artifacts(tmp_path: Path) -> None:
    emissions = _emissions(max_concepts=1)
    table = build_point_frame(
        source_frame=_source(max_concepts=1),
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=1,
    )
    output = tmp_path / "output"

    def validate(match: str) -> None:
        with pytest.raises(AxBenchAdvError, match=match):
            validate_final_judge_outputs(
                input_table=table,
                output_dir=output,
                best_factors=FACTORS,
                emissions=emissions,
                max_concepts=1,
            )

    _outputs(output, table, emissions, max_concepts=1)
    (output / "steering_data.parquet").unlink()
    validate("both required outputs")

    _outputs(output, table, emissions, max_concepts=1)
    (output / "steering.jsonl").write_text("{not json}\n")
    validate("invalid JSON")

    _outputs(output, table, emissions, max_concepts=1)
    _write_entries(output / "steering.jsonl", [])
    validate("steering.jsonl is empty")

    _outputs(output, table, emissions, max_concepts=1)
    entries = _entries(output / "steering.jsonl")
    entries[0]["extra"] = True
    _write_entries(output / "steering.jsonl", entries)
    validate("only concept_id and results")

    _outputs(output, table, emissions, max_concepts=1)
    entries = _entries(output / "steering.jsonl")
    entries[0]["results"] = []
    _write_entries(output / "steering.jsonl", entries)
    validate("method coverage")

    _outputs(output, table, emissions, max_concepts=1)
    entries = _entries(output / "steering.jsonl")
    methods = entries[0]["results"]["LMJudgeEvaluator"]
    entries[0]["results"]["LMJudgeEvaluator"] = dict(reversed(methods.items()))
    _write_entries(output / "steering.jsonl", entries)
    validate("method coverage")

    _outputs(output, table, emissions, max_concepts=1)
    entries, block = _block(output)
    block.pop("fluency_ratings")
    _write_entries(output / "steering.jsonl", entries)
    validate("result fields differ")

    _outputs(output, table, emissions, max_concepts=1)
    entries, block = _block(output)
    block["factor"] = [0.8]
    _write_entries(output / "steering.jsonl", entries)
    validate("factor differs")

    for field, value, match in (
        ("raw_relevance_concept_ratings", [], "wrong length"),
        ("raw_relevance_concept_ratings", "wrong", "wrong length"),
        ("raw_relevance_concept_ratings", [True] * 40, "nonnumeric rating"),
        ("raw_relevance_concept_ratings", [math.inf] * 40, "invalid JSON"),
        ("raw_relevance_concept_ratings", [2.1] * 40, "outside"),
        ("relevance_concept_completions", [], "wrong length"),
        ("relevance_concept_completions", [" "] * 40, "invalid completion"),
    ):
        _outputs(output, table, emissions, max_concepts=1)
        entries, block = _block(output)
        block[field] = value
        _write_entries(output / "steering.jsonl", entries)
        validate(match)

    _outputs(output, table, emissions, max_concepts=1)
    entries, block = _block(output, "LinearProbe")
    block["raw_relevance_concept_ratings"][0] = 1.0
    _write_entries(output / "steering.jsonl", entries)
    validate("populated an off-grid")

    for rationale, match in (
        ("not json", "rationale is not JSON"),
        ('{"concept":"only"}', "rationale fields differ"),
        ('{"concept":"","fluency":"valid"}', "must be nonblank"),
    ):
        _outputs(output, table, emissions, max_concepts=1)
        original = emissions[("DiffMean", 0, 0)]
        emissions[("DiffMean", 0, 0)] = RawEvaluation(
            original.generation,
            original.concept_rating,
            original.fluency_rating,
            rationale,
            original.paid_judge_cache_misses,
        )
        validate(match)
        emissions[("DiffMean", 0, 0)] = original

    _outputs(output, table, emissions, max_concepts=1)
    entries, block = _block(output)
    block["relevance_concept_completions"][0] = "different Rating: [[0.25]]"
    _write_entries(output / "steering.jsonl", entries)
    validate("completion differs from worker")

    for value, match in ((None, "incomplete"), (0.0, "inconsistent")):
        _outputs(output, table, emissions, max_concepts=1)
        entries, block = _block(output)
        block["relevance_concept_ratings"] = [value]
        _write_entries(output / "steering.jsonl", entries)
        validate(match)

    _outputs(output, table, emissions, max_concepts=1)
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    evaluated = evaluated.drop(columns="DiffMean_LMJudgeEvaluator")
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    validate("columns differ")

    _outputs(output, table, emissions, max_concepts=1)
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    columns = list(evaluated.columns)
    columns[0], columns[1] = columns[1], columns[0]
    evaluated.loc[:, columns].to_parquet(output / "steering_data.parquet", index=False)
    validate("preserve point columns")

    _outputs(output, table, emissions, max_concepts=1)
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    evaluated.loc[0, "original_prompt"] = "changed"
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    validate("changed its point-frame prefix")


def test_final_config_fails_on_vendor_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "evaluate.yaml"
    monkeypatch.setattr(final_judge_module, "EVALUATE_CONFIG", source)
    source.write_text("[]\n")
    with pytest.raises(AxBenchAdvError, match="no evaluate mapping"):
        final_judge_module._final_config(tmp_path / "result.yaml")
    source.write_text("evaluate:\n  lm_model: changed/model\n")
    with pytest.raises(AxBenchAdvError, match="judge model drifted"):
        final_judge_module._final_config(tmp_path / "result.yaml")


def test_run_final_judge_preflight_rejects_every_invalid_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emissions = _emissions(max_concepts=1)
    table = build_point_frame(
        source_frame=_source(max_concepts=1),
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=1,
    )
    input_path = tmp_path / "input" / "steering_data.parquet"
    write_point_frame(table, input_path, best_factors=FACTORS, max_concepts=1)
    cache = tmp_path / "judge.sqlite"
    cache.touch()
    base = {
        "input_table": table,
        "input_path": input_path,
        "output_dir": tmp_path / "judged",
        "cache_path": cache,
        "best_factors": FACTORS,
        "emissions": emissions,
        "max_concepts": 1,
        "cache_only": True,
    }

    for updates, match in (
        ({"cache_only": 1}, "cache_only must be a boolean"),
        ({"api_base_url": " "}, "api_base_url must be nonblank"),
        ({"input_path": tmp_path / "missing.parquet"}, "input is missing"),
        ({"cache_path": tmp_path / "missing.sqlite"}, "cache is missing"),
    ):
        with pytest.raises(AxBenchAdvError, match=match):
            run_final_judge(**{**base, **updates})  # ty: ignore[invalid-argument-type]

    changed = _emissions(max_concepts=1)
    changed[("DiffMean", 0, 0)] = _raw("DiffMean", 0, 0, generation="different")
    different_table = build_point_frame(
        source_frame=_source(max_concepts=1),
        best_factors=FACTORS,
        emissions=changed,
        max_concepts=1,
    )
    with pytest.raises(AxBenchAdvError, match="supplied point table"):
        run_final_judge(**{**base, "input_table": different_table})  # ty: ignore[invalid-argument-type]

    output = tmp_path / "judged"
    output.mkdir()
    with pytest.raises(AxBenchAdvError, match="already exists"):
        run_final_judge(**base)  # ty: ignore[invalid-argument-type]
    output.rmdir()

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AxBenchAdvError, match="OPENROUTER_API_KEY"):
        run_final_judge(**{**base, "cache_only": False})  # ty: ignore[invalid-argument-type]


def test_final_output_validator_requires_exact_worker_parity_and_final_instruction(
    tmp_path: Path,
) -> None:
    emissions = _emissions()
    table = build_point_frame(
        source_frame=_source(),
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=2,
    )
    output = tmp_path / "valid"
    _outputs(output, table, emissions)
    validate_final_judge_outputs(
        input_table=table,
        output_dir=output,
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=2,
    )

    entries = [json.loads(line) for line in (output / "steering.jsonl").read_text().splitlines()]
    block = entries[0]["results"]["LMJudgeEvaluator"]["DiffMean"]
    block["raw_relevance_concept_ratings"][0] = 2.0
    (output / "steering.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    with pytest.raises(AxBenchAdvError, match="rating differs from worker"):
        validate_final_judge_outputs(
            input_table=table,
            output_dir=output,
            best_factors=FACTORS,
            emissions=emissions,
            max_concepts=2,
        )

    _outputs(output, table, emissions)
    entries = [json.loads(line) for line in (output / "steering.jsonl").read_text().splitlines()]
    block = entries[0]["results"]["LMJudgeEvaluator"]["DiffMean"]
    block["raw_relevance_instruction_ratings"][0] = None
    block["relevance_instruction_completions"][0] = None
    (output / "steering.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    with pytest.raises(AxBenchAdvError, match="instruction judging is absent"):
        validate_final_judge_outputs(
            input_table=table,
            output_dir=output,
            best_factors=FACTORS,
            emissions=emissions,
            max_concepts=2,
        )

    _outputs(output, table, emissions)
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    evaluated.loc[0, "DiffMean_LMJudgeEvaluator_relevance_concept_ratings"] = 2.0
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    with pytest.raises(AxBenchAdvError, match="differs from its JSONL values"):
        validate_final_judge_outputs(
            input_table=table,
            output_dir=output,
            best_factors=FACTORS,
            emissions=emissions,
            max_concepts=2,
        )


@pytest.mark.parametrize("persistent", [False, True])
def test_run_final_judge_uses_six_method_config_and_publishes_validated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent: bool,
) -> None:
    emissions = _emissions()
    table = build_point_frame(
        source_frame=_source(),
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=2,
    )
    input_path = tmp_path / "input" / "steering_data.parquet"
    input_digest = write_point_frame(
        table,
        input_path,
        best_factors=FACTORS,
        max_concepts=2,
    )
    cache = tmp_path / "judge.sqlite"
    cache.touch()
    seen: dict[str, object] = {}

    def produce(config: Path, output: Path) -> None:
        assert yaml.safe_load(config.read_text())["evaluate"]["models"] == list(METHODS)
        _outputs(output, table, emissions)
        (output / "judge_call_accounting.json").write_text(
            json.dumps({"total_cache_hits": 357, "total_calls": 360}) + "\n"
        )

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = command
        seen["kwargs"] = kwargs
        produce(
            Path(command[command.index("--config") + 1]),
            Path(command[command.index("--overwrite_evaluate_dump_dir") + 1]),
        )
        return SimpleNamespace(returncode=0)

    class FakeProcess:
        def request(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
            assert operation == "evaluate"
            seen["payload"] = payload
            assert payload["cache_only"] is False
            produce(Path(str(payload["config_path"])), Path(str(payload["output_dir"])))
            return {}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("steering.eval.axbench_adv.final_judge.subprocess.run", fake_run)
    result = run_final_judge(
        input_table=table,
        input_path=input_path,
        output_dir=tmp_path / "judged",
        cache_path=cache,
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=2,
        _process=FakeProcess() if persistent else None,  # ty: ignore[invalid-argument-type]
    )
    assert result.steering_jsonl == (tmp_path / "judged" / "steering.jsonl").resolve()
    assert result.actual_evaluator_calls == 360
    assert result.paid_judge_cache_misses == 3
    if persistent:
        assert seen["payload"]["input_dir"] == str(input_path.parent.resolve())  # ty: ignore[not-subscriptable]
        assert "command" not in seen
    else:
        assert seen["kwargs"]["check"] is True  # ty: ignore[not-subscriptable]
    manifest = json.loads((tmp_path / "judged" / "manifest.json").read_text())
    assert manifest["input_logical_sha256"] == input_digest
    assert manifest["best_factors"] == FACTORS
    assert manifest["call_accounting"] == {
        "actual_evaluator_calls": 360,
        "paid_judge_cache_misses": 3,
    }
    accounting = tmp_path / "judged" / "judge_call_accounting.json"
    assert manifest["files"][accounting.name]["sha256"] == hashlib.sha256(
        accounting.read_bytes()
    ).hexdigest()
    validate_final_judge_outputs(
        input_table=table,
        output_dir=tmp_path / "judged",
        best_factors=FACTORS,
        emissions=emissions,
        max_concepts=2,
    )
    assert pq.read_table(tmp_path / "judged" / "steering_data.parquet").num_rows == 80


@pytest.mark.parametrize("fail", [False, True])
def test_persistent_final_judge_reuses_one_process_and_closes_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    processes: list[object] = []
    exits: list[type[BaseException] | None] = []
    seen: list[object] = []

    class FakeProcess:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            processes.append(self)

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object,
        ) -> None:
            exits.append(exc_type)

    def fake_run(**kwargs: object) -> FinalJudgeResult:
        seen.append(kwargs["_process"])
        if fail and len(seen) == 2:
            raise RuntimeError("final evaluator failed")
        return FinalJudgeResult(tmp_path / "judged/steering.jsonl", 0, 0)

    monkeypatch.setattr(final_judge_module, "_FramedProcess", FakeProcess)
    monkeypatch.setattr(final_judge_module, "run_final_judge", fake_run)
    with pytest.raises(RuntimeError, match="final evaluator failed") if fail else nullcontext():
        with PersistentFinalJudge(tmp_path) as judge:
            judge()
            judge()
    assert len(processes) == 1
    assert seen == [processes[0], processes[0]]
    assert exits == [RuntimeError if fail else None]
    with pytest.raises(AxBenchAdvError, match="closed"):
        judge()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (None, "did not write"),
        ("{", "malformed"),
        ("[]", "fields differ"),
        ('{"total_calls": 6}', "fields differ"),
        ('{"total_calls": true, "total_cache_hits": 4}', "must be integers"),
        ('{"total_calls": 6, "total_cache_hits": false}', "must be integers"),
        ('{"total_calls": -1, "total_cache_hits": 0}', "inconsistent"),
        ('{"total_calls": 6, "total_cache_hits": 7}', "inconsistent"),
        ('{"total_calls": 5, "total_cache_hits": 4}', "expected exactly 6"),
    ],
)
def test_call_accounting_rejects_every_invalid_boundary(
    tmp_path: Path,
    payload: str | None,
    match: str,
) -> None:
    path = tmp_path / "accounting.json"
    if payload is not None:
        path.write_text(payload)
    with pytest.raises(AxBenchAdvError, match=match):
        final_judge_module._read_call_accounting(path, expected_calls=6)


def test_call_accounting_returns_exact_paid_misses(tmp_path: Path) -> None:
    path = tmp_path / "accounting.json"
    path.write_text('{"total_calls": 6, "total_cache_hits": 4}')
    assert final_judge_module._read_call_accounting(path, expected_calls=6) == (6, 2)
