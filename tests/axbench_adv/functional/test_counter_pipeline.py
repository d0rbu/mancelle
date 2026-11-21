"""Additive pressure vertical slice with real artifact loading and existing fake boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.eval.axbench_adv.merge_slices import _load_run_slice, merge_slices
from scripts.eval.axbench_adv.pressures import (
    COEFFICIENT_SEMANTICS,
    DIRECTION_FAMILY,
    DIRECTION_METHOD,
    LEGACY_PRESSURES,
    Pressure,
)
from steering.eval.axbench_adv import pipeline
from steering.eval.axbench_adv.counter import CounterDirection, mean_counter_fitness
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.pipeline import run_pipeline
from steering.eval.axbench_adv.types import METHODS, ROOT_METHODS
from steering.safetensors_io import deterministic_safetensors_bytes
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import (
    FakeFinalJudge,
    FakeOptimizer,
    FakeReflection,
    FakeSelfConditioningCell,
    FakeSelfConditioningRuntime,
    FakeWorker,
    _config,
    _patch_source,
    _raw,
)


class _CounterWorker(FakeWorker):
    def evaluate_counter(self, counter, magnitude, input_ids):
        assert counter.provenance["method"] == self.method
        self.calls.append((f"counter-{magnitude}", tuple(input_ids)))
        return tuple(
            replace(
                _raw(self.method, self.concept_id, i, ""),
                generation=f"counter-{magnitude}-{self.method}-{i}",
                concept_rating=1.5 - magnitude / 2,
            )
            for i in input_ids
        )


class _CounterCell(FakeSelfConditioningCell):
    evaluate_counter = _CounterWorker.evaluate_counter


class _CounterRuntime(FakeSelfConditioningRuntime):
    def bind_cell(self, *, method, concept_id, **_kwargs):
        cell = _CounterCell(method, concept_id)
        self.cells.append(cell)
        return cell


class _ChangedControlWorker(_CounterWorker):
    def evaluate(self, suffix, input_ids):
        results = FakeWorker.evaluate(self, suffix, input_ids)
        if suffix:
            return results
        return tuple(replace(raw, generation=f"changed-{raw.generation}") for raw in results)


class _ChangedControlCell(_CounterCell):
    def evaluate(self, suffix, input_ids):
        results = FakeSelfConditioningCell.evaluate(self, suffix, input_ids)
        if suffix:
            return results
        return tuple(replace(raw, generation=f"changed-{raw.generation}") for raw in results)


class _ChangedControlRuntime(FakeSelfConditioningRuntime):
    def bind_cell(self, *, method, concept_id, **_kwargs):
        cell = _ChangedControlCell(method, concept_id)
        self.cells.append(cell)
        return cell


class _CounterJudge(FakeFinalJudge):
    def __call__(self, **kwargs):
        # Existing fake uses a pN name only to synthesize accounting; adapt
        # its path boundary, retaining its real emission-to-JSONL reduction.
        destination = kwargs["output_dir"]
        if destination.parent.name.startswith("m"):
            result = super().__call__(
                **{**kwargs, "output_dir": destination.parent / "p0" / "judged"}
            )
            result.steering_jsonl.parent.rename(destination)
            _write_complete_judge_outputs(destination, **kwargs)
            return replace(result, steering_jsonl=destination / "steering.jsonl")
        result = super().__call__(**kwargs)
        _write_complete_judge_outputs(destination, **kwargs)
        return result


def _aggregate(concept: float, instruction: float, fluency: float) -> float:
    return 0.0 if 0.0 in (concept, instruction, fluency) else 3.0 / (
        1.0 / concept + 1.0 / instruction + 1.0 / fluency
    )


def _write_complete_judge_outputs(destination: Path, **kwargs) -> None:
    """Make the functional fake honor the real final-judge output contract."""

    methods = tuple(kwargs["best_factors"])
    factors = kwargs["best_factors"]
    emissions = kwargs["emissions"]
    frame = kwargs["input_table"].to_pandas()
    evaluated = frame.copy()
    suffixes = (
        "LMJudgeEvaluator",
        "LMJudgeEvaluator_relevance_concept_ratings",
        "LMJudgeEvaluator_relevance_concept_completions",
        "LMJudgeEvaluator_relevance_instruction_ratings",
        "LMJudgeEvaluator_relevance_instruction_completions",
        "LMJudgeEvaluator_fluency_ratings",
        "LMJudgeEvaluator_fluency_completions",
    )
    columns = {method: {suffix: [] for suffix in suffixes} for method in methods}
    entries = []
    for concept_id in kwargs["concept_ids"]:
        concept_frame = frame.loc[frame["concept_id"].eq(concept_id)]
        blocks = {}
        for method in methods:
            values = {
                "concept": [],
                "instruction": [],
                "fluency": [],
                "aggregate": [],
                "concept_text": [],
                "instruction_text": [],
                "fluency_text": [],
            }
            for row in concept_frame.itertuples(index=False):
                if float(row.factor) != factors[method]:
                    for sequence in values.values():
                        sequence.append(None)
                    continue
                raw = emissions[(method, concept_id, int(row.input_id))]
                rationale = json.loads(raw.judge_rationale)
                instruction = 1.25
                values["concept"].append(raw.concept_rating)
                values["instruction"].append(instruction)
                values["fluency"].append(raw.fluency_rating)
                values["aggregate"].append(
                    _aggregate(raw.concept_rating, instruction, raw.fluency_rating)
                )
                values["concept_text"].append(rationale["concept"])
                values["instruction_text"].append(
                    f"instruction c{concept_id} i{row.input_id} Rating: [[1.25]]"
                )
                values["fluency_text"].append(rationale["fluency"])

            means = {
                name: sum(present) / len(present)
                for name in ("aggregate", "concept", "instruction", "fluency")
                if (present := [value for value in values[name] if value is not None])
            }

            blocks[method] = {
                "factor": [factors[method]],
                "lm_judge_rating": [means["aggregate"]],
                "relevance_concept_ratings": [means["concept"]],
                "relevance_instruction_ratings": [means["instruction"]],
                "fluency_ratings": [means["fluency"]],
                "raw_relevance_concept_ratings": values["concept"],
                "raw_relevance_instruction_ratings": values["instruction"],
                "raw_fluency_ratings": values["fluency"],
                "raw_aggregated_ratings": values["aggregate"],
                "relevance_concept_completions": values["concept_text"],
                "relevance_instruction_completions": values["instruction_text"],
                "fluency_completions": values["fluency_text"],
            }
            target = columns[method]
            target["LMJudgeEvaluator"].extend(values["aggregate"])
            target["LMJudgeEvaluator_relevance_concept_ratings"].extend(values["concept"])
            target["LMJudgeEvaluator_relevance_concept_completions"].extend(
                values["concept_text"]
            )
            target["LMJudgeEvaluator_relevance_instruction_ratings"].extend(
                values["instruction"]
            )
            target["LMJudgeEvaluator_relevance_instruction_completions"].extend(
                values["instruction_text"]
            )
            target["LMJudgeEvaluator_fluency_ratings"].extend(values["fluency"])
            target["LMJudgeEvaluator_fluency_completions"].extend(values["fluency_text"])
        entries.append({"concept_id": concept_id, "results": {"LMJudgeEvaluator": blocks}})
    for method in methods:
        for suffix in suffixes:
            evaluated[f"{method}_{suffix}"] = columns[method][suffix]
    evaluated.to_parquet(destination / "steering_data.parquet", index=False)
    (destination / "steering.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


def _run(
    tmp_path,
    monkeypatch,
    name,
    pressures,
    concept_ids=None,
    *,
    factor_overrides=None,
    project_judge_inflight=1,
    project_judge_workers=1,
    project_judge_pair_concurrency=1,
    project_reuse_judge_workers=False,
    project_reuse_single_token_mask=False,
    project_bf16_decoder_linears=False,
    project_reuse_diffmean_texts=False,
    project_weight_free_extraction=False,
    project_native_extraction=False,
    project_fit_selected_concepts_only=False,
    reuse_final_judge_worker=False,
    worker_factory=_CounterWorker,
    self_conditioning_factory=_CounterRuntime,
):
    output = tmp_path / name
    _patch_source(monkeypatch, output)
    if concept_ids is not None:
        monkeypatch.setenv("AXBENCH_ADV_CONCEPT_LO", str(concept_ids[0]))
        monkeypatch.setenv("AXBENCH_ADV_CONCEPT_HI", str(concept_ids[-1] + 1))
    config = _config(output)
    main = tmp_path / "main"
    train = main / "train"
    train.mkdir(parents=True, exist_ok=True)
    prepared = main / "prepared"
    prepared_train = prepared / "generate" / "train_data.parquet"
    prepared_train.parent.mkdir(parents=True, exist_ok=True)
    prepared_train.write_bytes(b"fixture prepared train")
    (prepared / "prepare_manifest.json").write_bytes(b"fixture prepared manifest")
    (prepared / "generate" / "metadata.jsonl").write_bytes(b"fixture prepared metadata")
    statistics = tmp_path / "feature_statistics.safetensors"
    statistics.write_bytes(b"fixture statistics")
    project = tmp_path / "baselines"
    for pressure in pressures:
        if pressure == Pressure.GEPA:
            continue
        t.save(
            t.tensor([[1.0, 2.0], [3.0, 4.0]]), train / f"{DIRECTION_METHOD[pressure]}_weight.pt"
        )
        for concept in range(2):
            path = (
                project
                / DIRECTION_FAMILY[pressure]
                / "encdec/positive"
                / f"concept_{concept:04d}.safetensors"
            )
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                deterministic_safetensors_bytes(
                    {"direction": t.tensor([1.0, 2.0])},
                    metadata={
                        "concept_id": str(concept),
                        "layer": "1",
                        "mode": "positive",
                        "target_model": "encdec",
                    },
                )
            )
    # This vertical slice owns fake model/judge boundaries. Direction binding
    # itself has a separate real serialization/preflight suite; keep this fake
    # focused on counter orchestration and publication.
    monkeypatch.setattr(
        "steering.eval.axbench_adv.pipeline.preflight_project_direction_fit",
        lambda _root, _expectations: object(),
    )

    def fake_counter_direction(**kwargs):
        pressure = kwargs["pressure"]
        method = kwargs["method"]
        concept_id = kwargs["concept_id"]
        is_project = method in ROOT_METHODS
        path = (
            project
            / DIRECTION_FAMILY[pressure]
            / "encdec/positive"
            / f"concept_{concept_id:04d}.safetensors"
            if is_project
            else train / f"{DIRECTION_METHOD[pressure]}_weight.pt"
        ).resolve()
        provenance = {
            "pressure": pressure.value,
            "method": method,
            "concept_id": concept_id,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "layer": 1 if is_project else 28,
            "family": DIRECTION_FAMILY[pressure],
            "target_model": "encdec" if is_project else "vendored_baseline",
            "mode": "positive",
            "coefficient_semantics": COEFFICIENT_SEMANTICS,
            "direction_norm": 1.0,
        }
        return CounterDirection(t.tensor([1.0, 0.0]), provenance)

    monkeypatch.setattr(
        "steering.eval.axbench_adv.pipeline.load_counter_direction",
        fake_counter_direction,
    )
    config = replace(
        config,
        pressures=pressures,
        counter_magnitudes=[0.25, 0.5],
        project_counter_direction_root=project,
        project_counter_layer=1,
        factor_overrides=factor_overrides or {},
        project_judge_inflight=project_judge_inflight,
        project_judge_workers=project_judge_workers,
        project_judge_pair_concurrency=project_judge_pair_concurrency,
        project_reuse_judge_workers=project_reuse_judge_workers,
        project_reuse_single_token_mask=project_reuse_single_token_mask,
        project_bf16_decoder_linears=project_bf16_decoder_linears,
        project_reuse_diffmean_texts=project_reuse_diffmean_texts,
        project_weight_free_extraction=project_weight_free_extraction,
        project_native_extraction=project_native_extraction,
        project_fit_selected_concepts_only=project_fit_selected_concepts_only,
        reuse_final_judge_worker=reuse_final_judge_worker,
        source_paths=replace(
            config.source_paths,
            main_parquet=main / "inference/steering_data.parquet",
            prepared_dir=prepared,
            feature_statistics=statistics,
        ),
    )
    return run_pipeline(
        config,
        reflection_lm=FakeReflection(),
        worker_factory=worker_factory,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=self_conditioning_factory,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=_CounterJudge(),
    )


@pytest.mark.parametrize("pressures", [(Pressure.DIFFMEAN_COUNTER,), LEGACY_PRESSURES])
def test_counter_pipeline_manifest_control_grid_and_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pressures
) -> None:
    control_stores: list[int] = []
    store = pipeline._store_evaluations

    def observe_store(target, **kwargs):
        if all(raw.generation.endswith("suffix=<empty>") for _, raw in kwargs["evaluations"]):
            control_stores.append(id(target))
        store(target, **kwargs)

    monkeypatch.setattr(pipeline, "_store_evaluations", observe_store)
    FakeFinalJudge.calls.clear()
    manifest_path = _run(tmp_path, monkeypatch, "run", pressures)
    # One store per method/concept, not another copy per counter pressure.
    assert len(control_stores) == len(METHODS) * 2
    assert len(set(control_stores)) == 1
    manifest = json.loads(manifest_path.read_text())
    assert manifest["pressures"] == list(pressures)
    assert manifest["counter_magnitudes"] == [0.25, 0.5]
    assert len(manifest["counter_directions"]) == 12 * sum(p != Pressure.GEPA for p in pressures)
    for pressure in pressures:
        if pressure == Pressure.GEPA:
            continue
        root = manifest_path.parent / "counter" / pressure
        table = pq.read_table(root / "aggregate.parquet")
        assert table.num_rows == 18
        assert set(table.column("magnitude").to_pylist()) == {0.0, 0.25, 0.5}
        assert set(json.loads((root / "selection/best_points.json").read_text())) == set(METHODS)
        assert len(list(root.glob("plots/*.png"))) == 7
        assert len(list(root.glob("*/c*/m*/point.json"))) == 36
        assert all(len(row["sha256"]) == 64 for row in manifest["counter_directions"])
    if Pressure.GEPA not in pressures:
        assert not (manifest_path.parent / "gepa").exists()
        assert manifest["call_accounting"]["gepa_nominal_metric_calls"] == 0
        assert len(FakeFinalJudge.calls) == 3
    else:
        # The control is generated and judged once, shared by all pressures.
        assert manifest["call_accounting"]["control_generation_calls"] == 120
        assert len(FakeFinalJudge.calls) == 9


def test_counter_pipeline_publishes_overlap_mode_after_complete_judging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "overlap",
        (Pressure.DIFFMEAN_COUNTER,),
        project_judge_inflight=2,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_judge_inflight"] == 2
    assert manifest["files"]["counter/diffmean_counter/aggregate.parquet"]


def test_counter_pipeline_records_project_pair_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "pair-concurrent",
        (Pressure.DIFFMEAN_COUNTER,),
        project_judge_pair_concurrency=2,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_judge_pair_concurrency"] == 2
    _load_run_slice(manifest_path.parent, Pressure.DIFFMEAN_COUNTER)


def test_counter_pipeline_records_project_mask_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "mask-reuse",
        (Pressure.DIFFMEAN_COUNTER,),
        project_reuse_single_token_mask=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_reuse_single_token_mask"] is True
    _load_run_slice(manifest_path.parent, Pressure.DIFFMEAN_COUNTER)


def test_counter_pipeline_records_bf16_decoder_linears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "bf16-decoder-linears",
        (Pressure.DIFFMEAN_COUNTER,),
        project_bf16_decoder_linears=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_bf16_decoder_linears"] is True
    assert _CounterRuntime.instances[-1].kwargs["bf16_decoder_linears"] is True
    _load_run_slice(manifest_path.parent, Pressure.DIFFMEAN_COUNTER)


def test_counter_pipeline_records_diffmean_text_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "diffmean-text-reuse",
        (Pressure.DIFFMEAN_COUNTER,),
        project_reuse_diffmean_texts=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_reuse_diffmean_texts"] is True
    assert _CounterRuntime.instances[-1].kwargs["reuse_diffmean_texts"] is True
    _load_run_slice(manifest_path.parent, Pressure.DIFFMEAN_COUNTER)


def test_counter_pipeline_passes_selected_concepts_to_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "selected-fit",
        (Pressure.DIFFMEAN_COUNTER,),
        concept_ids=(1,),
        project_fit_selected_concepts_only=True,
        project_weight_free_extraction=True,
        project_reuse_judge_workers=True,
        reuse_final_judge_worker=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_fit_selected_concepts_only"] is True
    assert manifest["project_weight_free_extraction"] is True
    assert manifest["project_reuse_judge_workers"] is True
    assert manifest["reuse_final_judge_worker"] is True
    assert manifest["concept_ids"] == [1]
    assert _CounterRuntime.instances[-1].kwargs["fit_concept_ids"] == (1,)
    assert _CounterRuntime.instances[-1].kwargs["weight_free_extraction"] is True
    assert _CounterRuntime.instances[-1].kwargs["prewarm_judge_cell"][1] == 1
    _load_run_slice(manifest_path.parent, Pressure.DIFFMEAN_COUNTER)


def test_counter_pipeline_publishes_native_extraction_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "native-extraction",
        (Pressure.DIFFMEAN_COUNTER,),
        concept_ids=(1,),
        project_weight_free_extraction=True,
        project_native_extraction=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_native_extraction"] is True
    assert _CounterRuntime.instances[-1].kwargs["native_extraction"] is True
    _load_run_slice(manifest_path.parent, Pressure.DIFFMEAN_COUNTER)


def test_counter_pipeline_publishes_combined_project_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _run(
        tmp_path,
        monkeypatch,
        "combined-options",
        (Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER),
        concept_ids=(1,),
        project_judge_inflight=3,
        project_judge_workers=2,
        project_judge_pair_concurrency=2,
        project_reuse_single_token_mask=True,
        project_reuse_diffmean_texts=True,
        project_fit_selected_concepts_only=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["project_judge_inflight"] == 3
    assert manifest["project_judge_workers"] == 2
    assert manifest["project_judge_pair_concurrency"] == 2
    assert manifest["project_reuse_single_token_mask"] is True
    assert manifest["project_reuse_diffmean_texts"] is True
    assert manifest["project_fit_selected_concepts_only"] is True
    assert _CounterRuntime.instances[-1].kwargs["fit_concept_ids"] == (1,)
    assert _CounterRuntime.instances[-1].kwargs["judge_workers"] == 2
    for pressure in (Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER):
        _load_run_slice(manifest_path.parent, pressure)


def test_counter_merger_disjoint_concepts_and_pressure_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pressure = Pressure.DIFFMEAN_COUNTER
    first = _run(tmp_path, monkeypatch, "first", (pressure,), [0])
    second = _run(tmp_path, monkeypatch, "second", (pressure,), [1])
    merged = merge_slices([first.parent, second.parent], tmp_path / "merged")
    table = pq.read_table(merged.parent / "counter" / pressure / "aggregate.parquet")
    assert set(table.column("n_concepts").to_pylist()) == {2}
    assert table.num_rows == 18
    reverse = merge_slices([second.parent, first.parent], tmp_path / "reverse")
    assert pq.read_table(
        reverse.parent / "counter" / pressure / "aggregate.parquet"
    ).equals(table)
    third = _run(tmp_path, monkeypatch, "third", (pressure,), [0])
    with pytest.raises(RuntimeError, match="overlap"):
        merge_slices([first.parent, third.parent], tmp_path / "overlap")
    other = _run(tmp_path, monkeypatch, "other", (Pressure.LINEAR_PROBE_COUNTER,), [0])
    # Same cells under different pressures are independent experiments.
    manifest = merge_slices([first.parent, other.parent], tmp_path / "different")
    assert json.loads(manifest.read_text())["pressures"] == [
        Pressure.LINEAR_PROBE_COUNTER,
        pressure,
    ]


def test_counter_merger_rejects_grid_provenance_and_point_tampering(tmp_path, monkeypatch):
    pressure = Pressure.DIFFMEAN_COUNTER
    first = _run(tmp_path, monkeypatch, "first", (pressure,), [0])
    second = _run(tmp_path, monkeypatch, "second", (pressure,), [1])
    original = second.read_bytes()
    document = json.loads(original)
    document["counter_magnitudes"] = [0.25]
    second.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="grids differ"):
        merge_slices([first.parent, second.parent], tmp_path / "grid")
    second.write_bytes(original)
    document = json.loads(original)
    document["counter_directions"][0]["coefficient_semantics"] = "incorrect"
    second.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="provenance"):
        merge_slices([first.parent, second.parent], tmp_path / "identity")
    second.write_bytes(original)
    point = next((second.parent / "counter" / pressure).glob("*/c*/m*/point.json"))
    original_point = point.read_bytes()
    point.write_text("{}")
    with pytest.raises(RuntimeError, match="point hash"):
        merge_slices([first.parent, second.parent], tmp_path / "hash")
    second.write_bytes(original)
    point.write_bytes(original_point)
    point = next((second.parent / "counter" / pressure).glob("*/c*/m0.0/point.json"))
    document = json.loads(point.read_text())
    document["evaluations"][0]["input_id"] = False
    point.write_text(json.dumps(document))
    manifest = json.loads(second.read_text())
    manifest["files"][str(point.relative_to(second.parent))] = hashlib.sha256(
        point.read_bytes()
    ).hexdigest()
    second.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="ten ordered inputs"):
        merge_slices([first.parent, second.parent], tmp_path / "bool-input-id")
    for name in ("grid", "identity", "hash", "bool-input-id"):
        assert not (tmp_path / name / "manifest.json").exists()


def test_counter_merger_rejects_authenticated_source_and_judge_parity_tampering(
    tmp_path, monkeypatch
):
    pressure = Pressure.DIFFMEAN_COUNTER
    first = _run(tmp_path, monkeypatch, "first", (pressure,), [0])
    second = _run(tmp_path, monkeypatch, "second", (pressure,), [1])

    original_manifest = second.read_bytes()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    manifest = json.loads(second.read_text())
    manifest["files"]["../outside.json"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    second.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="file path is unsafe"):
        merge_slices([first.parent, second.parent], tmp_path / "unsafe-path")
    second.write_bytes(original_manifest)

    source_contract = second.parent / "source_contract.json"
    source_contract.write_text('{"schema_version":2}\n')
    with pytest.raises(RuntimeError, match="recorded file hash"):
        merge_slices([first.parent, second.parent], tmp_path / "source-tamper")

    # Restore a fresh source, then alter a raw counter emission and authenticate
    # the altered point in the outer manifest.  The raw/input/final-judge parity
    # validator must still reject it.
    second = _run(tmp_path, monkeypatch, "second-fresh", (pressure,), [1])
    point = next((second.parent / "counter" / pressure).glob("*/c*/m0.25/point.json"))
    original_point = point.read_bytes()
    original_manifest = second.read_bytes()
    document = json.loads(point.read_text())
    document["evaluations"][0]["concept_rating"] = 0.25
    document["mean_fitness"] = mean_counter_fitness(
        [
            RawEvaluation(**{key: value for key, value in row.items() if key != "input_id"})
            for row in document["evaluations"]
        ]
    )
    point.write_text(json.dumps(document))
    manifest = json.loads(second.read_text())
    relative = str(point.relative_to(second.parent))
    manifest["files"][relative] = hashlib.sha256(point.read_bytes()).hexdigest()
    second.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="concept/fluency rating differs from worker"):
        merge_slices([first.parent, second.parent], tmp_path / "rating-tamper")

    point.write_bytes(original_point)
    second.write_bytes(original_manifest)
    document = json.loads(point.read_text())
    document["evaluations"][0]["generation"] = "authenticated but inconsistent"
    point.write_text(json.dumps(document))
    manifest = json.loads(second.read_text())
    manifest["files"][relative] = hashlib.sha256(point.read_bytes()).hexdigest()
    second.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="input generation differs from worker"):
        merge_slices([first.parent, second.parent], tmp_path / "parity-tamper")


def test_counter_merger_rejects_cross_pressure_factor_and_shared_zero_drift(
    tmp_path, monkeypatch
):
    first = _run(tmp_path, monkeypatch, "first", (Pressure.DIFFMEAN_COUNTER,), [0])
    different_factor = _run(
        tmp_path,
        monkeypatch,
        "factor",
        (Pressure.LINEAR_PROBE_COUNTER,),
        [0],
        factor_overrides={"DiffMean": 0.6},
    )
    with pytest.raises(RuntimeError, match="selected factor differs across pressures"):
        merge_slices([first.parent, different_factor.parent], tmp_path / "factor-merge")

    changed_control = _run(
        tmp_path,
        monkeypatch,
        "control",
        (Pressure.LINEAR_PROBE_COUNTER,),
        [0],
        worker_factory=_ChangedControlWorker,
        self_conditioning_factory=_ChangedControlRuntime,
    )
    with pytest.raises(RuntimeError, match="shared zero control differs across pressures"):
        merge_slices([first.parent, changed_control.parent], tmp_path / "control-merge")


def test_partial_merge_is_not_a_completed_campaign(tmp_path, monkeypatch):
    first = _run(tmp_path, monkeypatch, "first", (Pressure.DIFFMEAN_COUNTER,), [0])
    second = _run(tmp_path, monkeypatch, "second", (Pressure.DIFFMEAN_COUNTER,), [1])
    with pytest.raises(RuntimeError, match="all three pressure families"):
        merge_slices(
            [first.parent, second.parent],
            tmp_path / "incomplete",
            require_complete_campaign=True,
        )
    assert not (tmp_path / "incomplete" / "manifest.json").exists()


def test_mixed_pressure_merger_keeps_gepa_layout(tmp_path, monkeypatch):
    pressures = (Pressure.GEPA, Pressure.DIFFMEAN_COUNTER)
    first = _run(tmp_path, monkeypatch, "first", pressures, [0])
    second = _run(tmp_path, monkeypatch, "second", pressures, [1])
    merged = merge_slices([first.parent, second.parent], tmp_path / "merged")
    assert (merged.parent / "aggregate.parquet").is_file()
    assert (merged.parent / "plots/concept_vs_fluency.png").is_file()
    assert (merged.parent / "counter" / Pressure.DIFFMEAN_COUNTER / "aggregate.parquet").is_file()
    assert json.loads(merged.read_text())["pressures"] == list(pressures)
