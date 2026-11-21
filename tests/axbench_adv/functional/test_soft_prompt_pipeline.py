"""Real continuous driver, pipeline and merger with deterministic runtime boundaries."""

from __future__ import annotations

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
from safetensors.torch import load_file

from scripts.eval.axbench_adv.merge_slices import merge_slices
from scripts.eval.axbench_adv.pressures import COEFFICIENT_SEMANTICS, DIRECTION_FAMILY, Pressure
from steering.eval.axbench_adv import pipeline
from steering.eval.axbench_adv.counter import CounterDirection
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective
from steering.eval.axbench_adv.pipeline import run_pipeline
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.types import QWEN_METHODS, ROOT_METHODS
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from tests.axbench_adv.functional.test_counter_pipeline import (
    _CounterWorker,
    _write_complete_judge_outputs,
)
from tests.axbench_adv.functional.test_gcg_contract_review import (
    _GcgCell,
    _GcgFinalJudge,
    _GcgRuntimeMixin,
    _GcgWorker,
    _judge_cells,
    _write_training_source,
)
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import (
    FakeOptimizer,
    FakeReflection,
    FakeSelfConditioningRuntime,
    _config,
    _patch_source,
    _raw,
)


class _SoftMixin(_GcgRuntimeMixin):
    calls: list[tuple[str, tuple[int, ...]]]
    def prepare_soft_prompt(self, **kwargs: Any):
        prepared = self.prepare_gcg(**kwargs)
        project = prepared.identity.get("runtime") == "project_encoder_decoder"
        self._calls_before_search = len(self.calls)
        self._soft_count = kwargs["suffix_token_count"]
        return replace(
            prepared,
            identity={
                **prepared.identity,
                "embedding_layout": [
                    {
                        "name": "wte" if project else "input_embeddings",
                        "width": 2,
                        "dtype": "torch.float32",
                    }
                ],
                "generation_conditioning": "clean_prompt_S0_dynamic_decode"
                if project
                else "ordinary_prompt",
                **({"compute_dtype": "torch.float32"} if project else {}),
            },
        )

    def soft_score(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[float]:
        # Search must never reach a generation/judge runtime call.
        assert len(self.calls) == self._calls_before_search
        return [float(embeddings.square().sum()) + 0.001 * member for member in member_indices]

    def soft_gradient(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[t.Tensor]:
        assert len(self.calls) == self._calls_before_search
        return [2 * embeddings for _ in member_indices]

    def evaluate_soft(self, embeddings: t.Tensor, input_ids: Sequence[int]):
        fingerprint = hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()
        self.calls.append(("soft", tuple(input_ids)))
        return tuple(
            replace(
                _raw(self.method, self.concept_id, input_id, "suffix-p1"),
                generation=f"soft {self.method}/{self.concept_id}/{input_id} {fingerprint}",
            )
            for input_id in input_ids
        )

    evaluate_counter = _CounterWorker.evaluate_counter


class _SoftWorker(_SoftMixin, _GcgWorker):
    pass


class _SoftCell(_SoftMixin, _GcgCell):
    pass


class _SoftProjectRuntime(FakeSelfConditioningRuntime):
    def bind_cell(self, *, method: str, concept_id: int, factor: float = 1.0, **kwargs: Any):
        cell = _SoftCell(method, concept_id, factor)
        self.cells.append(cell)
        return cell


class _MixedJudge(_GcgFinalJudge):
    def __call__(self, **kwargs: Any):
        destination = Path(kwargs["output_dir"])
        if destination.parent.name.startswith("m"):
            synthetic = destination.parent / "p0" / "judged"
            result = super().__call__(**{**kwargs, "output_dir": synthetic})
            result.steering_jsonl.parent.rename(destination)
            synthetic.parent.rmdir()
            _write_complete_judge_outputs(destination, **kwargs)
            return replace(result, steering_jsonl=destination / "steering.jsonl")
        return super().__call__(**kwargs)


def _run_soft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: Path,
    label: str,
    *,
    methods: tuple[str, ...] = (QWEN_METHODS[0], ROOT_METHODS[0]),
    concepts: tuple[int, int] = (0, 2),
    objective: GcgObjective = GcgObjective.RESPONSE_MARGIN,
    pressures: tuple[Pressure, ...] = (Pressure.SOFT_PROMPT,),
    counts: tuple[int, ...] = (0, 1, 3),
    learning_rate: float = 0.01,
    judge: Any | None = None,
) -> Path:
    output = tmp_path / label
    _patch_source(monkeypatch, output)
    monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(methods))
    monkeypatch.setenv("AXBENCH_ADV_CONCEPT_LO", str(concepts[0]))
    monkeypatch.setenv("AXBENCH_ADV_CONCEPT_HI", str(concepts[1]))
    monkeypatch.setattr(pipeline, "preflight_project_direction_fit", lambda *_: object())
    monkeypatch.setattr(
        pipeline, "load_project_gcg_activation_direction", lambda *_args, **_kwargs: object()
    )

    def direction(**kwargs: Any):
        pressure = kwargs["pressure"]
        return CounterDirection(
            t.tensor([1.0, 0.0]),
            {
                "pressure": pressure.value,
                "method": kwargs["method"],
                "concept_id": kwargs["concept_id"],
                "path": "/source/fake-direction.pt",
                "sha256": "d" * 64,
                "layer": 28,
                "family": DIRECTION_FAMILY[pressure],
                "target_model": "encdec"
                if kwargs["method"] in ROOT_METHODS
                else "vendored_baseline",
                "mode": "positive",
                "coefficient_semantics": COEFFICIENT_SEMANTICS,
                "direction_norm": 1.0,
            },
        )

    monkeypatch.setattr(pipeline, "load_counter_direction", direction)
    config = _config(output)
    config = replace(
        config,
        pressures=pressures,
        source_paths=replace(config.source_paths, prepared_dir=prepared),
        project_counter_direction_root=Path("/source/fit"),
        project_counter_layer=0,
        counter_magnitudes=(0.25,),
        gcg=AxBenchAdvGcgConfig(
            device_batch_size=2,
            search=replace(
                FasterGcgConfig(),
                optimized_token_counts=(0, 1),
                max_steps=1,
                candidate_batch_size=4,
                top_k=4,
                distance_weight=0.0,
                universal_minibatch_size=1,
                candidate_eval_device_batch_size=2,
            ),
        ),
        soft_prompt=AxBenchAdvSoftPromptConfig(
            objective=objective,
            alpha=0.01 if objective is GcgObjective.RESPONSE_MARGIN else 937.0,
            device_batch_size=2,
            optimized_token_counts=counts,
            learning_rate=learning_rate,
            max_steps=2,
            universal_minibatch_size=1,
        ),
    )
    return run_pipeline(
        config,
        reflection_lm=FakeReflection(),
        worker_factory=_SoftWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=_SoftProjectRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=_MixedJudge() if judge is None else judge,
    ).parent


def _point(root: Path, count: int) -> Path:
    return root / ("points/p0" if count == 0 else f"soft_prompt/points/k{count}") / "judged"


@pytest.mark.parametrize("objective", list(GcgObjective))
@pytest.mark.parametrize("slice_kind", ["concept", "method"])
def test_soft_pipeline_slices_preserve_exact_judgments_tensors_receipts_and_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, objective: GcgObjective, slice_kind: str
) -> None:
    prepared = tmp_path / "prepared"
    _write_training_source(prepared)
    full = _run_soft(tmp_path, monkeypatch, prepared, "full", objective=objective)
    if slice_kind == "concept":
        roots = [
            _run_soft(
                tmp_path,
                monkeypatch,
                prepared,
                f"slice-{index}",
                concepts=(index, index + 1),
                objective=objective,
            )
            for index in range(2)
        ]
    else:
        roots = [
            _run_soft(
                tmp_path,
                monkeypatch,
                prepared,
                f"slice-{index}",
                methods=(method,),
                objective=objective,
            )
            for index, method in enumerate((QWEN_METHODS[0], ROOT_METHODS[0]))
        ]
    merged = merge_slices(roots, tmp_path / "merged")
    assert pq.read_table(merged.parent / "soft_prompt/aggregate.parquet").equals(
        pq.read_table(full / "soft_prompt/aggregate.parquet")
    )
    producer_manifest = json.loads((full / "manifest.json").read_text())
    assert producer_manifest["pressures"] == ["soft_prompt"]
    assert "gcg" not in producer_manifest
    assert producer_manifest["call_accounting"]["control_generation_calls"] == 40
    assert producer_manifest["call_accounting"]["actual_generation_calls"] == 80
    assert producer_manifest["soft_prompt"]["master_seed"] == 42
    merged_inner = json.loads((merged.parent / "soft_prompt/manifest.json").read_text())
    assert merged_inner["input_roots"] == [
        {
            "path": str(root),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        }
        for root in roots
    ]
    assert not list(merged.parent.rglob("*.safetensors"))
    assert not list(merged.parent.rglob("search.json"))
    for root in roots:
        for path in (root / "soft_prompt").glob("*/c*/k*/*"):
            assert path.read_bytes() == (full / path.relative_to(root)).read_bytes()
        for tensor_path in root.glob("soft_prompt/*/c*/k*/embeddings.safetensors"):
            embeddings = load_file(str(tensor_path))["embeddings"]
            fingerprint = hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()
            count = int(tensor_path.parent.name[1:])
            generations = pq.read_table(
                _point(root, count).parent / "input/steering_data.parquet"
            ).to_pandas()
            assert any(
                fingerprint in str(value) for column in generations for value in generations[column]
            )
    for count in (0, 1, 3):
        full_path = _point(full, count) / "steering.jsonl"
        paths = [_point(root, count) / "steering.jsonl" for root in roots]
        assert _judge_cells(paths) == _judge_cells([full_path])
        if slice_kind == "concept":
            assert full_path.read_bytes() == b"".join(path.read_bytes() for path in paths)
            assert pq.read_table(_point(full, count) / "steering_data.parquet").equals(
                pa.concat_tables(
                    [pq.read_table(_point(root, count) / "steering_data.parquet") for root in roots]
                )
            )


@pytest.mark.parametrize(
    "pressures",
    [
        (Pressure.GCG, Pressure.SOFT_PROMPT),
        (Pressure.GEPA, Pressure.DIFFMEAN_COUNTER, Pressure.GCG, Pressure.SOFT_PROMPT),
    ],
)
def test_mixed_pressures_share_exactly_one_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pressures: tuple[Pressure, ...]
) -> None:
    prepared = tmp_path / "prepared"
    _write_training_source(prepared)
    root = _run_soft(
        tmp_path,
        monkeypatch,
        prepared,
        "mixed",
        pressures=pressures,
        methods=(QWEN_METHODS[0],),
        concepts=(0, 1),
        counts=(0, 1),
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["call_accounting"]["control_generation_calls"] == 10
    assert len(list(root.rglob("points/p0/input/steering_data.parquet"))) == 1
    assert manifest["soft_prompt_training_source"] == manifest["gcg_training_source"]
    for pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
        rows = pq.read_table(root / pressure / "aggregate.parquet").to_pylist()
        assert [row["token_count"] for row in rows] == [0, 1]
        assert rows[0]["concept_score"] == 1.5
    assert not (root / "soft_prompt/points/k0").exists()


def test_soft_source_reread_rejects_mutation_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = tmp_path / "prepared"
    _write_training_source(prepared)

    class MutatingJudge(_MixedJudge):
        def __call__(self, **kwargs: Any):
            result = super().__call__(**kwargs)
            if Path(kwargs["output_dir"]).parent.name == "k1":
                path = prepared / "generate/train_data.parquet"
                frame = pd.read_parquet(path)
                frame.loc[0, "output"] = "changed response"
                frame.to_parquet(path, index=False)
            return result

    with pytest.raises(AxBenchAdvError, match="training source changed"):
        _run_soft(
            tmp_path,
            monkeypatch,
            prepared,
            "mutated",
            methods=(QWEN_METHODS[0],),
            concepts=(0, 1),
            counts=(0, 1),
            judge=MutatingJudge(),
        )
    assert (tmp_path / "mutated/soft_prompt/DiffMean/c0/k1/embeddings.safetensors").is_file()
    assert not (tmp_path / "mutated/manifest.json").exists()
