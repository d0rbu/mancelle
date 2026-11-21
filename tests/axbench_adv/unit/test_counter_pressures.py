"""Pressure ingress, authenticated direction identities and exact residual arithmetic."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch as t
from safetensors.torch import save_file

from scripts.eval.axbench_adv.counter_hooks import residual_counter
from scripts.eval.axbench_adv.pressures import (
    COEFFICIENT_SEMANTICS,
    DIRECTION_METHOD,
    Pressure,
    parse_pressures,
    validate_magnitudes,
)
from steering.eval.axbench_adv.aggregation import (
    COUNTER_AGGREGATE_SCHEMA,
    build_counter_aggregate,
    validate_counter_aggregate,
)
from steering.eval.axbench_adv.best_point import write_counter_best_points
from steering.eval.axbench_adv.counter import (
    load_counter_direction,
    mean_counter_fitness,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.types import QWEN_METHODS, SELF_CONDITIONING_METHODS


def test_pressure_defaults_subsets_and_environment() -> None:
    assert parse_pressures(environ={}) == (Pressure.GEPA,)
    assert parse_pressures([Pressure.DIFFMEAN_COUNTER], environ={}) == (Pressure.DIFFMEAN_COUNTER,)
    assert parse_pressures(environ={"AXBENCH_ADV_PRESSURES": " diffmean_counter,gepa "}) == (
        Pressure.GEPA,
        Pressure.DIFFMEAN_COUNTER,
    )


@pytest.mark.parametrize("value", [[], ["oops"], [Pressure.GEPA, Pressure.GEPA], "gepa"])
def test_pressure_invalid_config(value: list[str] | str) -> None:
    with pytest.raises(ValueError):
        parse_pressures(value, environ={})


@pytest.mark.parametrize("value", ["", "oops", "gepa,", "gepa,gepa"])
def test_pressure_invalid_env(value: str) -> None:
    with pytest.raises(ValueError):
        parse_pressures(environ={"AXBENCH_ADV_PRESSURES": value})


@pytest.mark.parametrize(
    "value", [[], [0], [-1], [float("nan")], [float("inf")], [True], [1, 0.5], [1, 1]]
)
def test_invalid_magnitude_grids(value: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_magnitudes(value)


@pytest.mark.parametrize("pressure", list(DIRECTION_METHOD))
def test_vendor_direction_identity_hash_and_concept(tmp_path: Path, pressure: Pressure) -> None:
    path = tmp_path / f"{DIRECTION_METHOD[pressure]}_weight.pt"
    t.save(t.tensor([[1.0, 2.0], [3.0, 4.0]]), path)
    loaded = load_counter_direction(
        pressure=pressure,
        method=QWEN_METHODS[2],
        concept_id=1,
        baseline_train=tmp_path,
        baseline_layer=3,
        project_fit=None,
    )
    t.testing.assert_close(loaded.direction, t.tensor([3.0, 4.0]), rtol=0, atol=0)
    assert loaded.provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.provenance["layer"] == 3
    assert loaded.provenance["coefficient_semantics"] == COEFFICIENT_SEMANTICS


def test_project_requires_completed_checkpoint_bound_preflight(tmp_path: Path) -> None:
    path = tmp_path / "diffmean/encdec/positive/concept_0000.safetensors"
    path.parent.mkdir(parents=True)
    save_file(
        {"direction": t.tensor([1.0, 2.0])},
        str(path),
        metadata={
            "concept_id": "0",
            "layer": "1",
            "mode": "positive",
            "target_model": "encdec",
            "artifact_kind": "diffmean_direction_v1",
        },
    )
    kwargs = {
        "pressure": Pressure.DIFFMEAN_COUNTER,
        "method": SELF_CONDITIONING_METHODS[0],
        "concept_id": 0,
        "baseline_train": tmp_path,
        "baseline_layer": 28,
        "project_fit": None,
    }
    with pytest.raises(AxBenchAdvError, match="preflight is required"):
        load_counter_direction(**kwargs)


def test_missing_direction_names_method_and_concept(tmp_path: Path) -> None:
    with pytest.raises(AxBenchAdvError, match="diffmean_counter/LoRA/c2"):
        load_counter_direction(
            pressure=Pressure.DIFFMEAN_COUNTER,
            method=QWEN_METHODS[2],
            concept_id=2,
            baseline_train=tmp_path,
            baseline_layer=1,
            project_fit=None,
        )


class _Block(t.nn.Module):
    def __init__(self, tuple_output: bool) -> None:
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, value: t.Tensor):
        result = value * 2
        return (result, "cache") if self.tuple_output else result


class _CounterModel:
    def __init__(self, model_type: str, blocks: t.nn.ModuleList) -> None:
        self.config = SimpleNamespace(model_type=model_type)
        container = SimpleNamespace(layers=blocks)
        self.gpt_neox = container
        self.model = container

    def forward(self, value: t.Tensor, layer: int):
        container = self.gpt_neox if self.config.model_type == "gpt_neox" else self.model
        return container.layers[layer](value)

    def __call__(self, value: t.Tensor, layer: int):
        return self.forward(value, layer)


@pytest.mark.parametrize("model_type", ["llama", "gpt_neox"])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_counter_exact_layer_math_composition_and_cleanup(
    model_type: str, tuple_output: bool
) -> None:
    blocks = t.nn.ModuleList([_Block(tuple_output) for _ in range(3)])
    model = _CounterModel(model_type, blocks)
    direction = t.tensor([2.0, -4.0])
    value = t.ones(1, 3, 2)

    def residual(index: int):
        result = model(value, index)
        return result[0] if tuple_output else result

    with residual_counter(model, layer=1, direction=direction, magnitude=0.25):
        for index in (0, 2):
            t.testing.assert_close(residual(index), value * 2, rtol=0, atol=0)
        t.testing.assert_close(residual(1), value * 2 - 0.25 * direction, rtol=0, atol=0)
        if tuple_output:
            assert blocks[1](value)[1] == "cache"
        # Existing additive intervention remains composed with the counter.
        with residual_counter(model, layer=1, direction=-direction, magnitude=2):
            t.testing.assert_close(residual(1), value * 2 + 1.75 * direction, rtol=0, atol=0)
    t.testing.assert_close(residual(1), value * 2, rtol=0, atol=0)
    assert "forward" not in vars(model)
    with (
        pytest.raises(RuntimeError),
        residual_counter(model, layer=1, direction=direction, magnitude=1),
    ):
        raise RuntimeError("generation failure")
    assert not blocks[1]._forward_hooks
    with residual_counter(model, layer=1, direction=direction, magnitude=0):
        assert not blocks[1]._forward_hooks


def test_counter_restores_existing_forward_override_after_failure() -> None:
    blocks = t.nn.ModuleList([_Block(False)])
    model = _CounterModel("llama", blocks)
    existing_forward = model.forward
    signature = inspect.signature(existing_forward)
    vars(model)["forward"] = existing_forward
    with (
        pytest.raises(RuntimeError, match="generation failure"),
        residual_counter(model, layer=0, direction=t.ones(2), magnitude=1),
    ):
        assert inspect.signature(model.forward) == signature
        assert model(t.ones(1, 1, 2), 0).shape == (1, 1, 2)
        raise RuntimeError("generation failure")
    assert vars(model)["forward"] is existing_forward
    assert not blocks[0]._forward_hooks


def test_counter_applies_once_per_block_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = t.nn.ModuleList([_Block(False)])
    model = _CounterModel("llama", blocks)
    value = t.ones(1, 1, 2)
    direction = t.tensor([0.25, 0.5])

    def repeated_forward(value: t.Tensor, layer: int) -> t.Tensor:
        for _ in range(3):
            value = blocks[layer](value)
        return value

    monkeypatch.setattr(model, "forward", repeated_forward)
    expected = value
    for _ in range(3):
        expected = expected * 2 - direction
    with residual_counter(model, layer=0, direction=direction, magnitude=1):
        actual = model(value, 0)
    assert t.equal(actual, expected)
    assert model.forward is repeated_forward
    assert not blocks[0]._forward_hooks


def test_nested_counters_preserve_scope_order_in_bfloat16() -> None:
    blocks = t.nn.ModuleList([_Block(False)])
    model = _CounterModel("llama", blocks)
    residual = t.tensor([0.72265625, 2.25], dtype=t.bfloat16)
    value = (residual / 2).reshape(1, 1, 2)
    outer = t.tensor([3.625, -1.6015625])
    inner = t.tensor([-2.21875, 1.9375])
    expected = (residual - outer.to(t.bfloat16)) - inner.to(t.bfloat16)
    reversed_order = (residual - inner.to(t.bfloat16)) - outer.to(t.bfloat16)
    assert not t.equal(expected, reversed_order)

    with residual_counter(model, layer=0, direction=outer, magnitude=1):
        with residual_counter(model, layer=0, direction=inner, magnitude=1):
            actual = model(value, 0).reshape(-1)
        with residual_counter(model, layer=0, direction=t.ones(3), magnitude=1):
            with pytest.raises(ValueError, match="direction width 3"):
                model(value, 0)
            assert not blocks[0]._forward_hooks
        outer_only = model(value, 0).reshape(-1)

    assert t.equal(actual, expected)
    assert t.equal(outer_only, residual - outer.to(t.bfloat16))
    assert "forward" not in vars(model)
    assert not blocks[0]._forward_hooks


def test_counter_aggregation_and_best_magnitude(tmp_path: Path) -> None:
    method = QWEN_METHODS[0]

    def entries(score: float):
        return [
            {
                "concept_id": 0,
                "results": {
                    "LMJudgeEvaluator": {
                        method: {
                            "factor": [1.0],
                            "relevance_concept_ratings": [score],
                            "fluency_ratings": [1.5],
                        }
                    }
                },
            }
        ]

    aggregate = build_counter_aggregate(
        best_factors={method: 1.0},
        judged_points={
            (method, 0.0): entries(1.5),
            (method, 0.25): entries(0.5),
            (method, 0.5): entries(0.5),
        },
        magnitudes=[0.25, 0.5],
        fitness={(method, 0.0): 0.25, (method, 0.25): 0.75, (method, 0.5): 0.75},
        concept_ids=[0],
        output_path=tmp_path / "aggregate.parquet",
    )
    assert aggregate.schema == COUNTER_AGGREGATE_SCHEMA
    best = json.loads(write_counter_best_points(aggregate, tmp_path / "selection").read_text())
    assert best[method]["magnitude"] == 0.25
    validate_counter_aggregate(aggregate)
    with pytest.raises(AxBenchAdvError):
        validate_counter_aggregate(pa.concat_tables([aggregate, aggregate]))
    with pytest.raises(AxBenchAdvError, match="zero control"):
        validate_counter_aggregate(aggregate.slice(1))


def test_counter_best_point_applies_gepa_adequacy_gate():
    raw = [
        RawEvaluation("I cannot help with that.", 0.0, 2.0, "{}", 0),
        RawEvaluation("Substantive answer", 0.5, 1.5, "{}", 0),
        RawEvaluation("Broken language", 0.0, 0.5, "{}", 0),
    ]
    assert mean_counter_fitness(raw) == 0.25


@pytest.mark.parametrize(
    "weights", [t.ones(2), t.ones(1, 2, dtype=t.int64), t.tensor([[float("nan"), 1.0]])]
)
def test_invalid_saved_directions_fail(tmp_path, weights):
    path = tmp_path / "DiffMean_weight.pt"
    t.save(weights, path)
    with pytest.raises(ValueError):
        load_counter_direction(
            pressure=Pressure.DIFFMEAN_COUNTER,
            method=QWEN_METHODS[0],
            concept_id=0,
            baseline_train=tmp_path,
            baseline_layer=1,
            project_fit=None,
        )
