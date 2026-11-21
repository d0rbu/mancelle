"""Root-owned FP32 Adam search and retained tensor receipts for soft suffixes."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch as t

from nanochat.atomic_io import atomic_safetensors_save, atomic_write_text
from steering.conditional.feature_gcg.aggregation import minibatch_for_step
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_driver import _training_rows
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.worker_client import GcgPrepared
from steering.eval.axbench_upstream.data import ConceptTrainingPairs
from steering.refined_types import NonNegativeInt, PositiveInt

_NATIVE_DTYPES = {str(dtype): dtype for dtype in (t.float16, t.bfloat16, t.float32, t.float64)}


class SoftPromptRuntime(Protocol):
    def prepare_soft_prompt(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str,
    ) -> GcgPrepared: ...

    def soft_score(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[float]: ...

    def soft_gradient(
        self, embeddings: t.Tensor, member_indices: Sequence[int]
    ) -> list[t.Tensor]: ...

    def evaluate_soft(
        self, embeddings: t.Tensor, input_ids: Sequence[int]
    ) -> Sequence[RawEvaluation]: ...


def validate_embedding_layout(identity: Mapping[str, Any]) -> tuple[tuple[int, t.dtype], ...]:
    """Validate the ordered coordinate partition shared by search and receipt replay."""

    layout = identity.get("embedding_layout")
    if not isinstance(layout, list) or not layout:
        raise AxBenchAdvError("soft-prompt embedding_layout must be a nonempty list")
    names: list[str] = []
    result: list[tuple[int, t.dtype]] = []
    for item in layout:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "width", "dtype"}
            or not isinstance(item["name"], str)
            or type(item["width"]) is not int
            or item["width"] <= 0
            or not isinstance(item["dtype"], str)
            or item["dtype"] not in _NATIVE_DTYPES
        ):
            raise AxBenchAdvError("soft-prompt embedding layout entry is malformed")
        names.append(item["name"])
        result.append((item["width"], _NATIVE_DTYPES[item["dtype"]]))
    if identity.get("runtime") == "project_encoder_decoder":
        layers: list[int] = []
        for name in names[1:]:
            if not name.startswith("value:") or not name[6:].isascii() or not name[6:].isdigit():
                raise AxBenchAdvError("soft-prompt project value embedding name is malformed")
            layer = int(name[6:])
            if name != f"value:{layer}":
                raise AxBenchAdvError("soft-prompt project value embedding name is noncanonical")
            layers.append(layer)
        if (
            names[0] != "wte"
            or layers != sorted(set(layers))
            or identity.get("embedding_dim") != sum(width for width, _dtype in result)
            or identity.get("generation_conditioning") != "clean_prompt_S0_dynamic_decode"
            or not isinstance(identity.get("compute_dtype"), str)
            or identity["compute_dtype"] not in _NATIVE_DTYPES
        ):
            raise AxBenchAdvError("soft-prompt project coordinate or conditioning policy differs")
    elif (
        names != ["input_embeddings"]
        or identity.get("generation_conditioning") != "ordinary_prompt"
    ):
        raise AxBenchAdvError("soft-prompt HF coordinate or conditioning policy differs")
    return tuple(result)


def _validate_tensor(
    value: t.Tensor,
    *,
    shape: tuple[int, int],
    layout: Sequence[tuple[int, t.dtype]],
    label: str,
    compute_dtype: t.dtype | None = None,
) -> None:
    if (
        not isinstance(value, t.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != t.float32
        or value.device.type != "cpu"
        or not bool(t.isfinite(value).all())
    ):
        raise AxBenchAdvError(
            f"soft-prompt {label} must be a finite FP32 CPU tensor of shape {shape}"
        )
    offset = 0
    for width, dtype in layout:
        if not bool(t.isfinite(value[:, offset : offset + width].to(dtype=dtype)).all()):
            raise AxBenchAdvError(f"soft-prompt {label} has a nonfinite native embedding cast")
        offset += width
    if compute_dtype is not None and not bool(t.isfinite(value.to(dtype=compute_dtype)).all()):
        raise AxBenchAdvError(f"soft-prompt {label} has a nonfinite decoder compute cast")


@dataclass(frozen=True)
class SoftPromptResult:
    embeddings: t.Tensor
    objective: float
    best_step: int
    updates: int


def optimize_soft_prompt(
    *,
    runtime: SoftPromptRuntime,
    prepared: GcgPrepared,
    config: AxBenchAdvSoftPromptConfig,
    master_seed: int,
    group_key: str,
) -> SoftPromptResult:
    """Keep the strict best full-population iterate, including initialization and the last update."""

    config.effective_identity(master_seed=master_seed)
    layout = validate_embedding_layout(prepared.identity)
    compute_dtype = (
        _NATIVE_DTYPES[prepared.identity["compute_dtype"]]
        if prepared.identity.get("runtime") == "project_encoder_decoder"
        else None
    )
    width = sum(part_width for part_width, _dtype in layout)
    weight = prepared.embedding_weight
    seed = prepared.seed_token_ids
    if (
        not isinstance(weight, t.Tensor)
        or weight.ndim != 2
        or not weight.is_floating_point()
        or weight.shape[1] != width
        or type(prepared.member_count) is not int
        or prepared.member_count <= 0
        or not seed
        or any(type(token) is not int or not 0 <= token < weight.shape[0] for token in seed)
    ):
        raise AxBenchAdvError("soft-prompt prepared embedding or population is malformed")
    shape = (len(seed), width)
    master = weight[list(seed)].detach().to(device="cpu", dtype=t.float32).clone()
    _validate_tensor(
        master, shape=shape, layout=layout, label="initialization", compute_dtype=compute_dtype
    )
    master.requires_grad_(True)
    members = list(range(prepared.member_count))

    def full_score() -> float:
        values = runtime.soft_score(master.detach(), members)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != len(members)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
        ):
            raise AxBenchAdvError("soft-prompt objective must return one finite value per member")
        try:
            objective = math.fsum(values) / len(values)
        except OverflowError as exc:
            raise AxBenchAdvError("soft-prompt full objective is nonfinite") from exc
        if not math.isfinite(objective):
            raise AxBenchAdvError("soft-prompt full objective is nonfinite")
        return objective

    best_value = full_score()
    best = master.detach().clone()
    best_step = 0
    optimizer = t.optim.Adam(
        [master],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    for step in range(config.max_steps):
        # The scheduler owns a local RNG; its beartype list-return check samples
        # Python's global RNG, so preserve that unrelated caller state too.
        python_rng = random.getstate()
        try:
            indices = minibatch_for_step(
                NonNegativeInt.parse(len(members)),
                step=NonNegativeInt.parse(step),
                minibatch_size=PositiveInt.parse(config.universal_minibatch_size),
                group_seed=f"{group_key}|seed{master_seed}",
            )
        finally:
            random.setstate(python_rng)
        gradients = runtime.soft_gradient(master.detach(), indices)
        if not isinstance(gradients, (list, tuple)) or len(gradients) != len(indices):
            raise AxBenchAdvError("soft-prompt gradient must return one tensor per member")
        for gradient in gradients:
            # Gradients are FP32 derivatives; they are never cast to the native input dtype.
            _validate_tensor(gradient, shape=shape, layout=(), label="gradient")
        master.grad = t.stack(gradients).mean(dim=0)
        _validate_tensor(master.grad, shape=shape, layout=(), label="mean gradient")
        try:
            optimizer.step()
        except (RuntimeError, OverflowError) as exc:
            raise AxBenchAdvError("soft-prompt Adam update failed") from exc
        _validate_tensor(
            master,
            shape=shape,
            layout=layout,
            label="updated embeddings",
            compute_dtype=compute_dtype,
        )
        for name, value in optimizer.state[master].items():
            if isinstance(value, t.Tensor) and not bool(t.isfinite(value).all()):
                raise AxBenchAdvError(f"soft-prompt Adam state {name} is nonfinite")
        current = full_score()
        if current < best_value:
            best, best_value, best_step = master.detach().clone(), current, step + 1
    return SoftPromptResult(best, best_value, best_step, config.max_steps)


def execute_soft_prompt_cell(
    *,
    config: AxBenchAdvSoftPromptConfig,
    runtime: SoftPromptRuntime,
    master_seed: int,
    method: str,
    concept_id: int,
    concept: str,
    factor: float,
    training_pairs: ConceptTrainingPairs,
    evaluation_prompts: Sequence[str],
    evaluation_input_ids: Sequence[int],
    output_dir: Path,
    totals: dict[str, int],
) -> dict[int, Sequence[tuple[int, RawEvaluation]]]:
    rows, training_identity = _training_rows(
        training_pairs, master_seed=master_seed, evaluation_prompts=evaluation_prompts
    )
    effective = config.effective_identity(master_seed=master_seed)
    results: dict[int, Sequence[tuple[int, RawEvaluation]]] = {}
    for count in config.optimized_token_counts:
        cell_dir = Path(output_dir) / "soft_prompt" / method / f"c{concept_id}" / f"k{count}"
        record: dict[str, Any] = {
            "schema_version": 1,
            "objective": config.objective.value,
            "effective_config": effective,
            "training_pair_identity": training_identity,
            "method": method,
            "concept_id": concept_id,
            "concept": concept,
            "factor": factor,
            "token_count": count,
            "diagnostics": {
                "control": True,
                "updates": 0,
                "best_step": 0,
                "full_population_scores": 0,
            },
        }
        if count > 0:
            prepared = runtime.prepare_soft_prompt(
                objective=config.objective.value,
                alpha=config.alpha,
                rows=rows,
                suffix_token_count=count,
                device_batch_size=config.device_batch_size,
                separator="\n\n",
            )
            if prepared.member_count != len(rows) or len(prepared.seed_token_ids) != count:
                raise AxBenchAdvError("soft-prompt prepared population or seed length differs")
            retained = optimize_soft_prompt(
                runtime=runtime,
                prepared=prepared,
                config=config,
                master_seed=master_seed,
                group_key=f"{method}|c{concept_id}|k{count}|{config.objective.value}",
            )
            evaluations = runtime.evaluate_soft(retained.embeddings, evaluation_input_ids)
            if len(evaluations) != len(evaluation_input_ids):
                raise AxBenchAdvError("soft-prompt final evaluation changed input cardinality")
            results[count] = list(zip(evaluation_input_ids, evaluations, strict=True))
            totals["actual_generation_calls"] += len(evaluations)
            totals["actual_evaluator_calls"] += len(evaluations)
            totals["paid_judge_cache_misses"] += sum(
                evaluation.paid_judge_cache_misses for evaluation in evaluations
            )
            tensor_path = cell_dir / "embeddings.safetensors"
            atomic_safetensors_save({"embeddings": retained.embeddings}, tensor_path)
            with tensor_path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            record.update(
                {
                    "model_tokenizer_identity": dict(prepared.identity),
                    "layouts_sha256": prepared.layouts_sha256,
                    "direction_calibration": prepared.direction_calibration,
                    "initialization_token_ids": list(prepared.seed_token_ids),
                    "objective_value": retained.objective,
                    "tensor": {
                        "path": tensor_path.name,
                        "sha256": digest,
                        "byte_length": tensor_path.stat().st_size,
                        "key": "embeddings",
                        "dtype": "torch.float32",
                        "shape": list(retained.embeddings.shape),
                    },
                    "diagnostics": {
                        "control": False,
                        "updates": retained.updates,
                        "best_step": retained.best_step,
                        "full_population_scores": retained.updates + 1,
                    },
                }
            )
        atomic_write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            cell_dir / "search.json",
        )
    return results


__all__ = [
    "SoftPromptRuntime",
    "SoftPromptResult",
    "execute_soft_prompt_cell",
    "optimize_soft_prompt",
    "validate_embedding_layout",
]
