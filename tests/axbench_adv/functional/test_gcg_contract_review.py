"""End-to-end GCG pipeline and concept-slice merge parity with fake runtimes."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.eval.axbench_adv.merge_slices import merge_slices
from scripts.eval.axbench_adv.pressures import Pressure
from steering.eval.axbench_adv import pipeline
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective
from steering.eval.axbench_adv.pipeline import run_pipeline
from steering.eval.axbench_adv.types import QWEN_METHODS, ROOT_METHODS
from steering.eval.axbench_adv.worker_client import GcgFilteredCandidate, GcgPrepared
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from tests.axbench_adv.functional.test_counter_pipeline import (
    _write_complete_judge_outputs,
)
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import (
    BEST_FACTORS,
    FakeFinalJudge,
    FakeOptimizer,
    FakeReflection,
    FakeSelfConditioningCell,
    FakeSelfConditioningRuntime,
    FakeWorker,
    _config,
    _patch_source,
    _validated,
)
from tests.axbench_adv.unit.test_gcg_selected_factor import _calibration

_METHODS = (QWEN_METHODS[0], ROOT_METHODS[-1])


class _GcgRuntimeMixin:
    method: str
    concept_id: int
    factor: float

    def prepare_gcg(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[dict[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str,
    ) -> GcgPrepared:
        assert objective in {"response_margin", "activation_l2"}
        assert alpha == (0.01 if objective == "response_margin" else 937.0)
        assert separator == "\n\n"
        assert device_batch_size == 2
        assert rows
        embedding = t.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        if self.method in ROOT_METHODS:
            identity = {
                "runtime": "project_encoder_decoder",
                "concept_id": self.concept_id,
                "rank_sentinel_id": 0,
                "vocab_size": embedding.shape[0],
                "embedding_dim": embedding.shape[1],
                "method": self.method,
                "factor": self.factor,
                "frozen_state": "clean_real_prompt_S0_before_assistant_marker",
            }
        else:
            identity = {
                "concept_id": self.concept_id,
                "factor": self.factor,
                "layer": 28,
                "method": self.method,
                "model_name": "fake/Qwen",
                "model_revision": None,
                "rank_sentinel_id": 0,
                "tokenizer_class": "FakeTokenizer",
                "vocab_size": embedding.shape[0],
            }
        layout_bytes = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        calibration = (
            _calibration(
                project=self.method in ROOT_METHODS,
                factor=self.factor,
                concept_id=self.concept_id,
            )
            if objective == "activation_l2"
            else None
        )
        if calibration is not None and self.method not in ROOT_METHODS:
            calibration.update(layer=28, site="model.layers[28].output_after_actual_intervention")
        return GcgPrepared(
            embedding_weight=embedding,
            allowed_token_ids=(1, 2, 3, 4),
            seed_token_ids=(1,) * suffix_token_count,
            member_count=len(rows),
            identity=identity,
            layouts_sha256=hashlib.sha256(layout_bytes).hexdigest(),
            direction_calibration=calibration,
        )

    def gcg_filter(
        self, candidate_token_ids: Sequence[Sequence[int]]
    ) -> tuple[GcgFilteredCandidate, ...]:
        filtered = []
        for candidate in candidate_token_ids:
            token_ids = tuple(candidate)
            valid = bool(token_ids) and all(token_id in {1, 2, 3, 4} for token_id in token_ids)
            filtered.append(
                GcgFilteredCandidate(
                    token_ids=token_ids,
                    text=f"suffix-p{sum(token_ids)}" if valid else "",
                    valid=valid,
                    error=None if valid else "invalid fake token",
                )
            )
        return tuple(filtered)

    def gcg_score(
        self,
        candidate_token_ids: Sequence[Sequence[int]],
        member_indices: Sequence[int],
        *,
        candidate_eval_device_batch_size: int,
    ) -> list[float]:
        assert candidate_eval_device_batch_size == 2
        return [
            -float(sum(candidate)) + 0.001 * member
            for candidate in candidate_token_ids
            for member in member_indices
        ]

    def gcg_gradient(
        self, token_ids: Sequence[int], member_indices: Sequence[int]
    ) -> list[t.Tensor]:
        return [t.ones((len(token_ids), 2), dtype=t.float32) for _ in member_indices]


class _GcgWorker(_GcgRuntimeMixin, FakeWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.factor = float(kwargs["init_payload"]["factor"])


class _GcgCell(_GcgRuntimeMixin, FakeSelfConditioningCell):
    def __init__(self, method: str, concept_id: int, factor: float) -> None:
        super().__init__(method, concept_id)
        self.factor = factor


class _GcgSelfConditioningRuntime(FakeSelfConditioningRuntime):
    def bind_cell(
        self, *, method: str, concept_id: int, factor: float = 1.0, **_kwargs: Any
    ) -> _GcgCell:
        cell = _GcgCell(method, concept_id, factor)
        self.cells.append(cell)
        return cell


class _GcgFinalJudge(FakeFinalJudge):
    def __call__(self, **kwargs: Any):
        destination = Path(kwargs["output_dir"])
        if destination.parent.name.startswith("k"):
            synthetic = destination.parent / "p0" / "judged"
            result = super().__call__(**{**kwargs, "output_dir": synthetic})
            result.steering_jsonl.parent.rename(destination)
            synthetic.parent.rmdir()
            _write_complete_judge_outputs(destination, **kwargs)
            return replace(result, steering_jsonl=destination / "steering.jsonl")
        result = super().__call__(**kwargs)
        _write_complete_judge_outputs(destination, **kwargs)
        return result


def _write_training_source(prepared_dir: Path) -> None:
    generate = prepared_dir / "generate"
    generate.mkdir(parents=True)
    rows = []
    metadata = []
    for concept_id in range(2):
        concept = f"concept-{concept_id}"
        genre = f"genre-{concept_id}"
        rows.extend(
            [
                {
                    "input": f"training positive {concept_id}",
                    "output": f"positive response {concept_id}",
                    "output_concept": concept,
                    "category": "positive",
                    "concept_genre": genre,
                },
                {
                    "input": f"training negative {concept_id}",
                    "output": f"negative response {concept_id}",
                    "output_concept": "EEEEE",
                    "category": "negative",
                    "concept_genre": genre,
                },
            ]
        )
        metadata.append(
            {
                "concept_id": concept_id,
                "concept": concept,
                "concept_genres_map": {concept: [genre]},
            }
        )
    pd.DataFrame(rows).to_parquet(generate / "train_data.parquet", index=False)
    (generate / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )


def _with_nl(validated: Any) -> Any:
    entries = copy.deepcopy(validated.steering_entries)
    for entry in entries:
        evaluator = entry["results"]["LMJudgeEvaluator"]
        evaluator["NLSteering"] = copy.deepcopy(evaluator["SelfConditioning"])
    project_manifest = copy.deepcopy(validated.project_manifest)
    project_manifest["methods"]["NLSteering"] = {"fixture_identity": "NL retrieval"}
    return replace(
        validated,
        expectations=replace(validated.expectations, include_nl_steering=True),
        steering_entries=entries,
        project_manifest=project_manifest,
        nl_target_scores={0: t.tensor([-1.0, 1.0]), 1: t.tensor([1.0, -1.0])},
    )


def _run_gcg(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validated: Any,
    prepared_dir: Path,
    label: str,
    lo: int,
    hi: int,
    objective: GcgObjective = GcgObjective.RESPONSE_MARGIN,
    final_judge: Any | None = None,
    pressures: tuple[Pressure, ...] = (Pressure.GCG,),
    factor_overrides: dict[str, float] | None = None,
    token_counts: tuple[int, ...] = (0, 1),
) -> Path:
    output = tmp_path / label
    _patch_source(monkeypatch, output)
    monkeypatch.setattr(
        pipeline,
        "validate_source_contract",
        lambda **kwargs: replace(validated, output_dir=kwargs["output_dir"]),
    )
    monkeypatch.setenv("AXBENCH_ADV_CONCEPT_LO", str(lo))
    monkeypatch.setenv("AXBENCH_ADV_CONCEPT_HI", str(hi))
    if objective is GcgObjective.ACTIVATION_L2:
        # Fake runtimes supply their own calibration receipts.
        monkeypatch.setattr(pipeline, "preflight_project_direction_fit", lambda *_args: object())
        monkeypatch.setattr(
            pipeline,
            "load_project_gcg_activation_direction",
            lambda _fit, *, concept_id: object(),
        )
    gcg = AxBenchAdvGcgConfig(
        objective=objective,
        alpha=0.01 if objective is GcgObjective.RESPONSE_MARGIN else 937.0,
        device_batch_size=2,
        search=replace(
            FasterGcgConfig(),
            optimized_token_counts=token_counts,
            max_steps=2,
            candidate_batch_size=4,
            top_k=4,
            distance_weight=0.0,
            universal_minibatch_size=1,
            candidate_eval_device_batch_size=2,
            master_seed=999,
        ),
    )
    config = replace(
        _config(output),
        pressures=pressures,
        gcg=gcg,
        source_expectations=validated.expectations,
        source_paths=replace(_config(output).source_paths, prepared_dir=prepared_dir),
        factor_overrides=factor_overrides or {},
        project_counter_direction_root=Path("/source/project-fit"),
        project_counter_layer=0,
    )
    manifest = run_pipeline(
        config,
        reflection_lm=FakeReflection(),
        worker_factory=_GcgWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=_GcgSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=_GcgFinalJudge() if final_judge is None else final_judge,
    )
    assert manifest == output.resolve() / "manifest.json"
    return manifest


def _judge_paths(root: Path, point: str) -> tuple[Path, Path]:
    base = root / ("points/p0" if point == "k0" else f"gcg/points/{point}") / "judged"
    return base / "steering.jsonl", base / "steering_data.parquet"


def _judge_cells(paths: Sequence[Path]) -> dict[tuple[int, str, str], Any]:
    """Compare method slices without their native factor-union null padding."""
    cells = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            results = row.pop("results")
            assert results
            for evaluator, methods in results.items():
                assert methods
                for method, block in methods.items():
                    if evaluator == "LMJudgeEvaluator":
                        for key, values in block.items():
                            if key.startswith("raw_") or key.endswith("_completions"):
                                block[key] = [value for value in values if value is not None]
                    cells[row["concept_id"], evaluator, method] = (row, block)
    return cells


@pytest.mark.parametrize(
    ("objective", "methods"),
    [
        (GcgObjective.RESPONSE_MARGIN, _METHODS),
        (GcgObjective.ACTIVATION_L2, (QWEN_METHODS[0],)),
    ],
)
def test_gcg_pipeline_slices_merge_to_exact_unsliced_judgments_and_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    objective: GcgObjective,
    methods: tuple[str, ...],
) -> None:
    """The real driver/pipeline/merger path is invariant to concept slicing."""

    prepared_dir = tmp_path / "prepared"
    _write_training_source(prepared_dir)
    validated = _validated(tmp_path / "validated")
    if "NLSteering" in methods:
        validated = _with_nl(validated)
    monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(reversed(methods)))
    _GcgWorker.instances = []
    _GcgSelfConditioningRuntime.instances = []

    full = _run_gcg(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        validated=validated,
        prepared_dir=prepared_dir,
        label="full",
        lo=0,
        hi=2,
        objective=objective,
    )
    document = json.loads(full.read_text(encoding="utf-8"))
    assert document["pressures"] == ["gcg"]
    assert document["methods"] == list(methods)
    assert document["gcg"]["search"]["master_seed"] == document["master_seed"] == 42
    assert ("alpha" in document["gcg"]) is (objective is GcgObjective.RESPONSE_MARGIN)
    assert "counter_directions" not in document
    assert not (full.parent / "aggregate.parquet").exists()
    expected_generations = len(methods) * 2 * 10
    assert document["call_accounting"]["control_generation_calls"] == expected_generations
    assert document["call_accounting"]["actual_generation_calls"] == expected_generations
    assert len(list(full.parent.glob("gcg/*/c*/k0/search.json"))) == len(methods) * 2
    assert len(list(full.parent.glob("gcg/*/c*/k1/search.json"))) == len(methods) * 2

    low = _run_gcg(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        validated=validated,
        prepared_dir=prepared_dir,
        label="low",
        lo=0,
        hi=1,
        objective=objective,
    )
    high = _run_gcg(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        validated=validated,
        prepared_dir=prepared_dir,
        label="high",
        lo=1,
        hi=2,
        objective=objective,
    )
    merged = merge_slices([low.parent, high.parent], tmp_path / "merged")
    assert pq.read_table(merged.parent / "gcg/aggregate.parquet").equals(
        pq.read_table(full.parent / "gcg/aggregate.parquet")
    )

    for point in ("k0", "k1"):
        full_json, full_parquet = _judge_paths(full.parent, point)
        low_json, low_parquet = _judge_paths(low.parent, point)
        high_json, high_parquet = _judge_paths(high.parent, point)
        assert full_json.read_bytes() == low_json.read_bytes() + high_json.read_bytes()
        sliced = pa.concat_tables([pq.read_table(low_parquet), pq.read_table(high_parquet)])
        assert pq.read_table(full_parquet).equals(sliced)

    for concept_id, source in ((0, low.parent), (1, high.parent)):
        for method in methods:
            relative = Path("gcg") / method / f"c{concept_id}" / "k1" / "search.json"
            assert (full.parent / relative).read_bytes() == (source / relative).read_bytes()


def test_mixed_gepa_gcg_pipeline_reuses_control_with_nl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representative additive pressure combination shares its judged control."""

    prepared_dir = tmp_path / "prepared"
    _write_training_source(prepared_dir)
    validated = _with_nl(_validated(tmp_path / "validated"))
    monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(reversed(_METHODS)))

    manifest = _run_gcg(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        validated=validated,
        prepared_dir=prepared_dir,
        label="mixed",
        lo=0,
        hi=1,
        pressures=(Pressure.GEPA, Pressure.GCG),
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["pressures"] == ["gepa", "gcg"]
    assert document["call_accounting"]["control_generation_calls"] == len(_METHODS) * 10
    assert (manifest.parent / "points/p0/judged/steering.jsonl").is_file()
    assert (manifest.parent / "gcg/points/k1/judged/steering.jsonl").is_file()
    assert (manifest.parent / "aggregate.parquet").is_file()
    assert (manifest.parent / "gcg/aggregate.parquet").is_file()


def test_selected_factor_l2_matches_concept_and_method_slices_and_rejects_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_training_source(prepared_dir)
    validated = _with_nl(_validated(tmp_path / "validated"))
    # Concept zero prefers 0.2 locally, while the full population still selects 1.
    local = validated.steering_entries[0]["results"]["LMJudgeEvaluator"]["DiffMean"]
    index = local["factor"].index(0.2)
    local["relevance_concept_ratings"][index] = 2.0
    local["fluency_ratings"][index] = 2.0
    methods = (*QWEN_METHODS, *ROOT_METHODS)
    overrides = {"LinearProbe": 0.6, "SelfConditioningMasked": 0.0, "NLSteering": 1.5}
    expected_factors = {**BEST_FACTORS, "NLSteering": 6.0, **overrides}

    def run(label: str, selected: tuple[str, ...], lo: int, hi: int) -> Path:
        monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(selected))
        return _run_gcg(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            validated=validated,
            prepared_dir=prepared_dir,
            label=label,
            lo=lo,
            hi=hi,
            objective=GcgObjective.ACTIVATION_L2,
            factor_overrides=overrides,
            token_counts=(0, 1, 3),
        ).parent

    full = run("full", methods, 0, 2)
    full_manifest = json.loads((full / "manifest.json").read_text())
    assert full_manifest["best_factors"] == expected_factors
    concept_roots = [run("c0", methods, 0, 1), run("c1", methods, 1, 2)]
    method_roots = [run("hf", QWEN_METHODS, 0, 2), run("project", ROOT_METHODS, 0, 2)]
    for label, roots in (("concept", concept_roots), ("method", method_roots)):
        merged = merge_slices(roots, tmp_path / f"merged-{label}")
        assert pq.read_table(merged.parent / "gcg/aggregate.parquet").equals(
            pq.read_table(full / "gcg/aggregate.parquet")
        )
        documents = [json.loads((root / "manifest.json").read_text()) for root in roots]
        for key in (
            "control_generation_calls",
            "actual_generation_calls",
            "actual_evaluator_calls",
            "paid_judge_cache_misses",
        ):
            assert (
                sum(document["call_accounting"][key] for document in documents)
                == full_manifest["call_accounting"][key]
            )
        for root, document in zip(roots, documents, strict=True):
            for method, factor in document["best_factors"].items():
                assert factor == expected_factors[method]
            for receipt_path in root.glob("gcg/*/c*/k*/search.json"):
                relative = receipt_path.relative_to(root)
                assert receipt_path.read_bytes() == (full / relative).read_bytes()
                receipt = json.loads(receipt_path.read_text())
                assert (
                    receipt["effective_config"]["direction_factor_policy"]
                    == "selected_method_factor"
                )
                if receipt["token_count"] == 0:
                    assert "direction_calibration" not in receipt
                else:
                    calibration = receipt["direction_calibration"]
                    assert (
                        calibration["factor"]
                        == receipt["factor"]
                        == expected_factors[receipt["method"]]
                    )
                    assert calibration["max_activation"] == 2.0 + receipt["concept_id"] * 1.3
                    assert (
                        calibration["resolved_scale"]
                        == calibration["max_activation"] * receipt["factor"]
                    )
        for k in (0, 1, 3):
            full_json, full_parquet = _judge_paths(full, f"k{k}")
            paths = [_judge_paths(root, f"k{k}") for root in roots]
            json_paths = [json_path for json_path, _parquet in paths]
            if label == "concept":
                assert full_json.read_bytes() == b"".join(path.read_bytes() for path in json_paths)
            else:
                assert _judge_cells(json_paths) == _judge_cells([full_json])
            tables = [pq.read_table(parquet_path) for _json, parquet_path in paths]
            expected_table = pq.read_table(full_parquet)
            for method in methods:
                columns = [
                    name
                    for name in expected_table.column_names
                    if name.startswith(f"{method}_") or name in {"concept_id", "input_id", "factor"}
                ]
                present = f"{method}_LMJudgeEvaluator"
                parts = [
                    table.select(columns).filter(table[present].is_valid())
                    for table in tables
                    if present in table.column_names
                ]
                order = [(key, "ascending") for key in ("concept_id", "input_id", "factor")]
                actual = pa.concat_tables(parts).sort_by(order)
                expected = (
                    expected_table.select(columns)
                    .filter(expected_table[present].is_valid())
                    .sort_by(order)
                )
                assert actual.equals(expected)

    # Hashes authenticate the rewritten legacy bytes; semantic validation must reject them.
    for legacy_count, root in enumerate(concept_roots, 1):
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["gcg"].pop("direction_factor_policy")
        for receipt_path in root.glob("gcg/*/c*/k*/search.json"):
            receipt = json.loads(receipt_path.read_text())
            receipt["effective_config"].pop("direction_factor_policy")
            receipt_path.write_text(json.dumps(receipt))
            manifest["files"][str(receipt_path.relative_to(root))] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))
        output = tmp_path / f"reject-legacy-{legacy_count}"
        with pytest.raises(AxBenchAdvError, match="fresh L2 run"):
            merge_slices(concept_roots, output)
        assert not list(output.rglob("*manifest.json"))


def test_gcg_publication_reloads_and_rejects_training_source_changed_after_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_training_source(prepared_dir)
    train_path = prepared_dir / "generate/train_data.parquet"
    validated = _validated(tmp_path / "validated")
    monkeypatch.setenv("AXBENCH_ADV_METHODS", QWEN_METHODS[0])

    class MutatingJudge(_GcgFinalJudge):
        mutated = False

        def __call__(self, **kwargs: Any):
            result = super().__call__(**kwargs)
            destination = Path(kwargs["output_dir"])
            if destination.parent.name == "k1" and not self.mutated:
                frame = pd.read_parquet(train_path)
                frame.loc[0, "output"] = "mutated after search"
                frame.to_parquet(train_path, index=False)
                self.mutated = True
            return result

    output = tmp_path / "mutated"
    with pytest.raises(AxBenchAdvError, match="training source changed"):
        _run_gcg(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            validated=validated,
            prepared_dir=prepared_dir,
            label=output.name,
            lo=0,
            hi=1,
            final_judge=MutatingJudge(),
        )
    assert (output / f"gcg/{QWEN_METHODS[0]}/c0/k1/search.json").is_file()
    assert not (output / "manifest.json").exists()
