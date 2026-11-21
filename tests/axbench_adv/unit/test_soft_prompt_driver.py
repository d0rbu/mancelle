"""Independent numerical and retention oracles for the continuous optimizer."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import load_file

from steering.eval.axbench_adv import soft_prompt_driver as driver_module
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.soft_prompt_config import (
    AxBenchAdvSoftPromptConfig,
    parse_soft_prompt_config,
)
from steering.eval.axbench_adv.soft_prompt_driver import (
    execute_soft_prompt_cell,
    optimize_soft_prompt,
    validate_embedding_layout,
)
from steering.eval.axbench_adv.worker_client import GcgPrepared
from steering.eval.axbench_upstream.data import ConceptTrainingPairs


class ToySoftRuntime:
    """Analytic per-pair quadratics; no production numerical kernels are reused."""

    def __init__(
        self,
        *,
        members: int = 5,
        method: str = "DiffMean",
        concept_id: int = 0,
        factor: float = 1.0,
    ):
        self.members = members
        self.method = method
        self.concept_id = concept_id
        self.factor = factor
        self.prepares: list[int] = []
        self.scored: list[t.Tensor] = []
        self.minibatches: list[list[int]] = []
        self.generated: list[t.Tensor] = []
        self.scores: list[float] | None = None
        self.constant_gradient: float | None = None
        self.dtype = "torch.float32"
        self.active_count = 1

    def prepared(self, count: int = 1) -> GcgPrepared:
        self.active_count = count
        return GcgPrepared(
            embedding_weight=t.tensor([[0.0, 0.0], [1.0, -2.0], [3.0, 4.0]]),
            allowed_token_ids=(1, 2),
            seed_token_ids=(1,) * count,
            member_count=self.members,
            identity={
                "concept_id": self.concept_id,
                "factor": self.factor,
                "layer": 0,
                "method": self.method,
                "model_name": "fixture/model",
                "model_revision": "fixture-revision",
                "rank_sentinel_id": 0,
                "tokenizer_class": "FixtureTokenizer",
                "vocab_size": 3,
                "embedding_layout": [{"name": "input_embeddings", "width": 2, "dtype": self.dtype}],
                "generation_conditioning": "ordinary_prompt",
            },
            layouts_sha256="a" * 64,
            direction_calibration=None,
        )

    def prepare_soft_prompt(
        self, *, rows: Sequence[Mapping[str, Any]], suffix_token_count: int, **kwargs: Any
    ) -> GcgPrepared:
        assert kwargs["separator"] == "\n\n"
        self.members = len(rows)
        self.prepares.append(suffix_token_count)
        return self.prepared(suffix_token_count)

    def soft_score(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[float]:
        assert tuple(member_indices) == tuple(range(self.members))
        self.scored.append(embeddings.clone())
        if self.scores is not None:
            return [self.scores[len(self.scored) - 1]] * len(member_indices)
        return [float(((embeddings - (member + 1) * 0.1) ** 2).sum()) for member in member_indices]

    def soft_gradient(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[t.Tensor]:
        self.minibatches.append(list(member_indices))
        return [
            t.full_like(embeddings, self.constant_gradient)
            if self.constant_gradient is not None
            else 2 * (embeddings - (member + 1) * 0.1)
            for member in member_indices
        ]

    def evaluate_soft(
        self, embeddings: t.Tensor, input_ids: Sequence[int]
    ) -> Sequence[RawEvaluation]:
        self.generated.append(embeddings.clone())
        return [
            RawEvaluation(
                generation=f"soft-{self.active_count}-{input_id}-{float(embeddings.sum()):.6f}",
                concept_rating=1.0,
                fluency_rating=1.5,
                judge_rationale="{}",
                paid_judge_cache_misses=0,
            )
            for input_id in input_ids
        ]


def training_pairs(concept_id: int = 0, members: int = 5) -> ConceptTrainingPairs:
    return ConceptTrainingPairs(
        concept_id=concept_id,
        concept=f"topic {concept_id}",
        genre="fixture",
        positive=tuple((f"positive {concept_id}/{i}", f"good {i}") for i in range(members)),
        negative=tuple((f"negative {concept_id}/{i}", f"bad {i}") for i in range(members)),
    )


def _independent_minibatch(step: int, *, members: int, size: int, seed: int, key: str) -> list[int]:
    epoch_length = math.ceil(members / size)
    epoch, offset = divmod(step, epoch_length)
    order = list(range(members))
    random.Random(hashlib.sha256(f"{key}|seed{seed}|epoch{epoch}".encode()).hexdigest()).shuffle(
        order
    )
    return order[offset * size : (offset + 1) * size]


@pytest.mark.parametrize("steps", [1, 7])
def test_adam_matches_independent_scalar_oracle_with_uneven_minibatches(steps: int) -> None:
    runtime = ToySoftRuntime()
    config = AxBenchAdvSoftPromptConfig(
        max_steps=steps, learning_rate=0.01, universal_minibatch_size=2
    )
    result = optimize_soft_prompt(
        runtime=runtime,
        prepared=runtime.prepared(),
        config=config,
        master_seed=19,
        group_key="oracle",
    )
    position, mean, variance = [1.0, -2.0], [0.0, 0.0], [0.0, 0.0]
    expected_batches = []
    for step in range(steps):
        members = _independent_minibatch(step, members=5, size=2, seed=19, key="oracle")
        expected_batches.append(members)
        for j in range(2):
            gradient = sum(2 * (position[j] - (member + 1) * 0.1) for member in members) / len(
                members
            )
            mean[j] = 0.9 * mean[j] + 0.1 * gradient
            variance[j] = 0.999 * variance[j] + 0.001 * gradient**2
            corrected_mean = mean[j] / (1 - 0.9 ** (step + 1))
            corrected_variance = variance[j] / (1 - 0.999 ** (step + 1))
            position[j] -= 0.01 * corrected_mean / (math.sqrt(corrected_variance) + 1e-8)
    t.testing.assert_close(result.embeddings, t.tensor([position]), atol=3e-7, rtol=2e-7)
    assert runtime.minibatches == expected_batches
    if steps > 2:
        assert list(map(len, expected_batches[:3])) == [2, 2, 1]
    assert result.best_step == steps
    assert result.updates == steps
    assert len(runtime.scored) == steps + 1


@pytest.mark.parametrize(
    ("scores", "best_step"),
    [
        ([0.0, 1.0, 2.0, 3.0], 0),
        ([3.0, 1.0, 2.0, 4.0], 1),
        ([3.0, 2.0, 1.0, 0.0], 3),
        ([1.0, 0.0, 0.0, 0.0], 1),
        ([1.0, 1.0, 1.0, 1.0], 0),
        ([1.0], 0),
    ],
)
def test_retention_clones_strict_best_including_initialization_and_last_update(
    scores: list[float], best_step: int
) -> None:
    runtime = ToySoftRuntime()
    runtime.scores, runtime.constant_gradient = scores, 1.0
    result = optimize_soft_prompt(
        runtime=runtime,
        prepared=runtime.prepared(),
        config=AxBenchAdvSoftPromptConfig(max_steps=len(scores) - 1),
        master_seed=0,
        group_key="retention",
    )
    assert result.best_step == best_step
    assert result.objective == scores[best_step]
    assert t.equal(result.embeddings, runtime.scored[best_step])
    assert len(runtime.scored) == len(scores)
    assert len(runtime.minibatches) == len(scores) - 1
    assert result.embeddings.data_ptr() != runtime.scored[best_step].data_ptr()


def test_search_preserves_python_and_torch_global_rng() -> None:
    runtime = ToySoftRuntime()
    python_state, torch_state = random.getstate(), t.random.get_rng_state()
    optimize_soft_prompt(
        runtime=runtime,
        prepared=runtime.prepared(),
        config=AxBenchAdvSoftPromptConfig(max_steps=4),
        master_seed=11,
        group_key="rng",
    )
    assert random.getstate() == python_state
    assert t.equal(t.random.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    ("gradient", "lr", "dtype", "steps", "match"),
    [
        (float("nan"), 0.01, "torch.float32", 1, "gradient"),
        (float("inf"), 0.01, "torch.float32", 1, "gradient"),
        (1e30, 0.01, "torch.float32", 1, "Adam state exp_avg_sq"),
        (1.0, 70000.0, "torch.float16", 1, "native embedding cast"),
        (1.0, 3e37, "torch.float32", 12, "updated embeddings"),
    ],
)
def test_nonfinite_gradient_adam_moments_update_and_native_cast_fail(
    gradient: float, lr: float, dtype: str, steps: int, match: str
) -> None:
    runtime = ToySoftRuntime()
    runtime.constant_gradient, runtime.dtype = gradient, dtype
    runtime.scores = [0.0] * (steps + 1)
    with pytest.raises(AxBenchAdvError, match=match):
        optimize_soft_prompt(
            runtime=runtime,
            prepared=runtime.prepared(),
            config=AxBenchAdvSoftPromptConfig(max_steps=steps, learning_rate=lr),
            master_seed=0,
            group_key="finite",
        )


@pytest.mark.parametrize(
    "scores", [[float("nan")], [float("inf")], [-float("inf")], [1.0, float("nan")]]
)
def test_nonfinite_initial_and_updated_objectives_fail(scores: list[float]) -> None:
    runtime = ToySoftRuntime()
    runtime.scores = scores
    with pytest.raises(AxBenchAdvError, match="objective"):
        optimize_soft_prompt(
            runtime=runtime,
            prepared=runtime.prepared(),
            config=AxBenchAdvSoftPromptConfig(max_steps=len(scores) - 1),
            master_seed=0,
            group_key="finite",
        )


@pytest.mark.parametrize("kind", ["short", "wrong_shape", "wrong_dtype", "wrong_device"])
def test_gradient_shape_dtype_cardinality_are_validated(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ToySoftRuntime(members=1)
    gradient = {
        "short": [],
        "wrong_shape": [t.zeros(2, 2)],
        "wrong_dtype": [t.zeros(1, 2, dtype=t.float64)],
        "wrong_device": [t.zeros(1, 2, device="meta")],
    }[kind]
    monkeypatch.setattr(runtime, "soft_gradient", lambda *_: gradient)
    with pytest.raises(AxBenchAdvError, match="gradient"):
        optimize_soft_prompt(
            runtime=runtime,
            prepared=runtime.prepared(),
            config=AxBenchAdvSoftPromptConfig(max_steps=1),
            master_seed=0,
            group_key="invalid",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "other"),
        ("alpha", True),
        ("alpha", float("nan")),
        ("alpha", -1),
        ("learning_rate", 0),
        ("learning_rate", float("inf")),
        ("max_steps", -1),
        ("max_steps", True),
        ("device_batch_size", 0),
        ("universal_minibatch_size", False),
        ("optimized_token_counts", (1, 2)),
        ("optimized_token_counts", (0, True)),
        ("optimized_token_counts", (0, 2, 2)),
    ],
)
def test_closed_config_rejects_malformed_values(field: str, value: object) -> None:
    with pytest.raises(AxBenchAdvError):
        AxBenchAdvSoftPromptConfig(**{field: value})  # ty: ignore[invalid-argument-type]


def test_config_excludes_discrete_knobs_and_l2_alpha() -> None:
    config = AxBenchAdvSoftPromptConfig()
    assert parse_soft_prompt_config(asdict(config)) == config
    with pytest.raises(AxBenchAdvError, match="exactly"):
        parse_soft_prompt_config({**asdict(config), "top_k": 10})
    first = replace(config, objective="activation_l2", alpha=0.0).effective_identity(master_seed=2)
    second = replace(config, objective="activation_l2", alpha=1.0).effective_identity(master_seed=2)
    assert first == second
    assert first["direction_factor_policy"] == "selected_method_factor"
    assert first["master_seed"] == 2
    assert first["optimizer"]["foreach"] is False


def test_budgets_initialize_independently_and_final_generation_uses_retained_tensor(
    tmp_path: Path,
) -> None:
    runtime = ToySoftRuntime()
    config = AxBenchAdvSoftPromptConfig(optimized_token_counts=(0, 1, 4), max_steps=2)
    totals = dict.fromkeys(
        ("actual_generation_calls", "actual_evaluator_calls", "paid_judge_cache_misses"), 0
    )
    results = execute_soft_prompt_cell(
        config=config,
        runtime=runtime,
        master_seed=3,
        method="DiffMean",
        concept_id=0,
        concept="topic 0",
        factor=1.0,
        training_pairs=training_pairs(),
        evaluation_prompts=("held out",),
        evaluation_input_ids=(0, 1),
        output_dir=tmp_path,
        totals=totals,
    )
    assert set(results) == {1, 4}
    assert runtime.prepares == [1, 4]
    assert t.equal(runtime.scored[0], t.tensor([[1.0, -2.0]]))
    assert t.equal(runtime.scored[3], t.tensor([[1.0, -2.0]]).repeat(4, 1))
    assert totals == {
        "actual_generation_calls": 4,
        "actual_evaluator_calls": 4,
        "paid_judge_cache_misses": 0,
    }
    for index, count in enumerate((1, 4)):
        root = tmp_path / "soft_prompt" / "DiffMean" / "c0" / f"k{count}"
        receipt = json.loads((root / "search.json").read_text())
        retained = load_file(str(root / "embeddings.safetensors"))["embeddings"]
        assert t.equal(retained, runtime.generated[index])
        assert receipt["diagnostics"]["best_step"] == 2
        assert "suffix" not in receipt and "suffix_token_ids" not in receipt
        assert (
            receipt["tensor"]["sha256"]
            == hashlib.sha256((root / "embeddings.safetensors").read_bytes()).hexdigest()
        )
    zero = tmp_path / "soft_prompt" / "DiffMean" / "c0" / "k0"
    assert not (zero / "embeddings.safetensors").exists()
    assert json.loads((zero / "search.json").read_text())["diagnostics"]["control"] is True


def test_training_eval_overlap_fails_before_prepare(tmp_path: Path) -> None:
    runtime = ToySoftRuntime()
    with pytest.raises(AxBenchAdvError, match="overlap"):
        execute_soft_prompt_cell(
            config=AxBenchAdvSoftPromptConfig(),
            runtime=runtime,
            master_seed=0,
            method="DiffMean",
            concept_id=0,
            concept="topic 0",
            factor=1.0,
            training_pairs=training_pairs(),
            evaluation_prompts=("positive 0/0",),
            evaluation_input_ids=(0,),
            output_dir=tmp_path,
            totals={},
        )
    assert runtime.prepares == []


@pytest.mark.parametrize(
    "change",
    [
        {
            "embedding_layout": [
                {"name": "input_embeddings", "width": True, "dtype": "torch.float32"}
            ]
        },
        {"embedding_layout": [{"name": "input_embeddings", "width": 2, "dtype": "torch.int64"}]},
        {"generation_conditioning": "placeholder_text"},
    ],
)
def test_embedding_layout_is_semantic_identity(change: dict[str, object]) -> None:
    identity = {**ToySoftRuntime().prepared().identity, **change}
    with pytest.raises(AxBenchAdvError):
        validate_embedding_layout(identity)


@pytest.mark.parametrize("values", [None, [], [1.0], [True] * 5, ["1"] * 5])
def test_objective_cardinality_and_scalar_types_are_strict(
    values, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ToySoftRuntime()
    monkeypatch.setattr(runtime, "soft_score", lambda *_: values)
    with pytest.raises(AxBenchAdvError, match="objective"):
        optimize_soft_prompt(
            runtime=runtime,
            prepared=runtime.prepared(),
            config=AxBenchAdvSoftPromptConfig(max_steps=0),
            master_seed=0,
            group_key="invalid",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"seed_token_ids": ()},
        {"seed_token_ids": (True,)},
        {"seed_token_ids": (3,)},
        {"member_count": 0},
        {"member_count": True},
        {"embedding_weight": t.zeros(3, 3)},
        {"embedding_weight": t.full((3, 2), float("nan"))},
    ],
)
def test_prepared_state_rejects_invalid_seed_population_and_initial_embeddings(
    change: dict[str, Any],
) -> None:
    runtime = ToySoftRuntime()
    prepared = replace(runtime.prepared(), **change)
    with pytest.raises(AxBenchAdvError):
        optimize_soft_prompt(
            runtime=runtime,
            prepared=prepared,
            config=AxBenchAdvSoftPromptConfig(max_steps=0),
            master_seed=0,
            group_key="invalid",
        )
    assert runtime.scored == []


def test_project_compute_cast_is_checked_beyond_native_table_dtype() -> None:
    runtime = ToySoftRuntime()
    prepared = runtime.prepared()
    prepared = replace(
        prepared,
        embedding_weight=t.full((3, 2), t.finfo(t.float32).max),
        identity={
            **prepared.identity,
            "runtime": "project_encoder_decoder",
            "embedding_dim": 2,
            "embedding_layout": [{"name": "wte", "width": 2, "dtype": "torch.float32"}],
            "generation_conditioning": "clean_prompt_S0_dynamic_decode",
            "compute_dtype": "torch.bfloat16",
        },
    )
    with pytest.raises(AxBenchAdvError, match="decoder compute cast"):
        optimize_soft_prompt(
            runtime=runtime,
            prepared=prepared,
            config=AxBenchAdvSoftPromptConfig(max_steps=0),
            master_seed=0,
            group_key="invalid",
        )
    assert runtime.scored == []


def test_failed_tensor_publication_cannot_authenticate_a_partial_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_save(_tensors, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial tensor")
        raise OSError("injected write failure")

    monkeypatch.setattr(driver_module, "atomic_safetensors_save", broken_save)
    with pytest.raises(OSError, match="injected write failure"):
        execute_soft_prompt_cell(
            config=AxBenchAdvSoftPromptConfig(optimized_token_counts=(0, 1), max_steps=0),
            runtime=ToySoftRuntime(members=1),
            master_seed=0,
            method="DiffMean",
            concept_id=0,
            concept="topic 0",
            factor=1.0,
            training_pairs=training_pairs(members=1),
            evaluation_prompts=("held out",),
            evaluation_input_ids=(0,),
            output_dir=tmp_path,
            totals=dict.fromkeys(
                ("actual_generation_calls", "actual_evaluator_calls", "paid_judge_cache_misses"), 0
            ),
        )
    positive = tmp_path / "soft_prompt/DiffMean/c0/k1"
    assert (positive / "embeddings.safetensors").read_bytes() == b"partial tensor"
    assert not (positive / "search.json").exists()
