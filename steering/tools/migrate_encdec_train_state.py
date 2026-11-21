"""Offline W31->W32 encoder-decoder training-state migration.

The supported contract deliberately keeps the training configuration fixed
apart from run identity and the one-rank-larger global batch: BF16, B14,
gradient accumulation two, and the raw training step are preserved.  Model
weights are copied byte-for-byte, optimizer tensors are resharded exactly, and
the precomputed dataloader restarts at an audited replay-only shard boundary.

Run with ``dry_run=true`` first.  Execute mode requires a fresh destination and
publishes it atomically from a sibling staging directory.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_json_dump, atomic_torch_save
from nanochat.gpt import OptimizerKind
from scripts.train.conditional_encdec_train import (
    _checkpoint_user_config,
    _scaling_parameter_count,
    build_encdec_meta_model_from_training_cfg,
)
from steering.conditional.checkpoint_conditioning import read_checkpoint_conditioning
from steering.conditional.encoder_decoder.config import feature_prediction_config_from_metadata
from steering.conditional.encoder_decoder.topology_resume import (
    PackedRowInventory,
    ReplayBoundaryPlan,
    build_replay_boundary_envelope,
    validate_selection_inventory,
)
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    feature_vocab_from_state_dict,
)

_BATCH_SIZE_REFERENCE = 2**19
_CANONICAL_CHECKPOINT_COMPUTE_DTYPE = t.bfloat16
_MODEL_RE = re.compile(r"model_(\d{6})\.pt$")
_OPTIM_RE = re.compile(r"optim_(\d{6})_rank(\d+)\.pt$")
_REMOTE_META_KEYS = frozenset({"remote_run_fingerprint", "remote_semantic_identity"})
_ALLOWED_CONFIG_DIFFERENCES = frozenset(
    {"model_tag", "resume_from_step", "run", "total_batch_size"}
)


@dataclass(frozen=True)
class ParameterLayout:
    """One optimizer parameter's stable identity and tensor contract."""

    param_id: int
    name: str
    shape: tuple[int, ...]
    dtype: t.dtype
    numel: int
    kind: str
    group_index: int


@dataclass(frozen=True)
class GroupLayout:
    """One canonical optimizer group in flat state-dict order."""

    kind: str
    param_ids: tuple[int, ...]


@dataclass(frozen=True)
class OptimizerLayout:
    """World-independent mapping from optimizer ids to model parameters."""

    parameters: dict[int, ParameterLayout]
    groups: tuple[GroupLayout, ...]

    @property
    def expected_state_ids(self) -> frozenset[int]:
        ids: set[int] = set()
        for group in self.groups:
            if group.kind == OptimizerKind.ADAMW.value:
                ids.update(group.param_ids)
            else:
                ids.add(group.param_ids[0])
        return frozenset(ids)


@dataclass(frozen=True)
class MigrationPlan:
    """Fully validated dry-run result used by execute mode."""

    source_world_size: int
    destination_world_size: int
    step: int
    source_total_batch_size: int
    destination_total_batch_size: int
    device_batch_size: int
    max_seq_len: int
    grad_accum_steps: int
    lr_scale_ratio: float
    packed_row_fingerprint: str
    dataloader: ReplayBoundaryPlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_world_size": self.source_world_size,
            "destination_world_size": self.destination_world_size,
            "step": self.step,
            "source_total_batch_size": self.source_total_batch_size,
            "destination_total_batch_size": self.destination_total_batch_size,
            "device_batch_size": self.device_batch_size,
            "max_seq_len": self.max_seq_len,
            "grad_accum_steps": self.grad_accum_steps,
            "lr_scale_ratio": self.lr_scale_ratio,
            "packed_row_fingerprint": self.packed_row_fingerprint,
            "dataloader": self.dataloader.as_dict(),
        }


@dataclass
class PreparedMigration:
    """Validated source objects retained for optional execute mode."""

    plan: MigrationPlan
    source_dir: Path
    destination_dir: Path
    model_path: Path
    model_state: Mapping[str, t.Tensor]
    source_meta: dict[str, Any]
    destination_meta: dict[str, Any]
    source_optimizer_paths: tuple[Path, ...]
    source_optimizer_shards: tuple[dict[str, Any], ...]
    target_param_groups: list[dict[str, Any]]
    optimizer_layout: OptimizerLayout
    packed_inventory: PackedRowInventory
    source_file_identities: dict[Path, tuple[int, ...]]


def _kind_value(raw: Any) -> str:
    value = raw.value if isinstance(raw, OptimizerKind) else raw
    if value not in {OptimizerKind.ADAMW.value, OptimizerKind.MUON.value}:
        raise ValueError(f"unknown optimizer group kind {raw!r}")
    return cast(str, value)


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _strict_equal(left: Any, right: Any) -> bool:
    """JSON-like equality that preserves key presence and scalar types."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_json_file_exact(path: Path, expected: Mapping[str, Any]) -> None:
    actual = _load_json(path)
    if not _strict_equal(actual, expected):
        raise ValueError(f"serialized JSON differs from the intended value: {path}")


def _file_identity(path: Path) -> tuple[int, ...]:
    """Capture link and target identity for a quiescent-source fence."""

    link = path.lstat()
    target = path.stat()
    if not stat.S_ISREG(target.st_mode):
        raise ValueError(f"checkpoint source is not a regular file: {path}")

    def fields(result: os.stat_result) -> tuple[int, ...]:
        return (
            int(result.st_mode),
            int(result.st_dev),
            int(result.st_ino),
            int(result.st_size),
            int(result.st_mtime_ns),
            int(result.st_ctime_ns),
        )

    return (*fields(link), *fields(target))


def _assert_files_unchanged(identities: Mapping[Path, tuple[int, ...]]) -> None:
    changed: list[str] = []
    for path, expected in identities.items():
        try:
            actual = _file_identity(path)
        except (OSError, ValueError):
            changed.append(str(path))
            continue
        if actual != expected:
            changed.append(str(path))
    if changed:
        raise ValueError(f"checkpoint source files changed during migration: {changed[:8]}")


def _safe_mmap_load(path: Path) -> dict[str, Any]:
    try:
        value = t.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"cannot memory-map safe checkpoint {path}; migration requires a local "
            "new-format torch.save file"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a dictionary")
    return value


def _discover_optimizer_paths(source_dir: Path, step: int) -> tuple[Path, ...]:
    by_rank: dict[int, Path] = {}
    for path in source_dir.glob(f"optim_{step:06d}_rank*.pt"):
        match = _OPTIM_RE.fullmatch(path.name)
        if match is None or int(match.group(1)) != step:
            continue
        rank = int(match.group(2))
        if rank in by_rank:
            raise ValueError(f"duplicate optimizer shard for rank {rank}")
        by_rank[rank] = path
    if not by_rank:
        raise FileNotFoundError(f"no optimizer shards found for step {step:06d} in {source_dir}")
    expected = list(range(len(by_rank)))
    if sorted(by_rank) != expected:
        raise ValueError(
            "optimizer shard ranks must be contiguous from zero: "
            f"found={sorted(by_rank)}, expected={expected}"
        )
    return tuple(by_rank[rank] for rank in expected)


def _lr_multiplier(cfg: Mapping[str, Any], *, step: int) -> float:
    iterations = _require_int(cfg["num_iterations"], name="num_iterations", minimum=1)
    warmup = round(float(cfg["warmup_ratio"]) * iterations)
    warmdown = round(float(cfg["warmdown_ratio"]) * iterations)
    if step < warmup:
        return (step + 1) / warmup
    if step <= iterations - warmdown:
        return 1.0
    progress = (iterations - step) / warmdown
    return progress + (1 - progress) * float(cfg["final_lr_frac"])


def _muon_momentum(step: int) -> float:
    fraction = min(step / 300, 1)
    return (1 - fraction) * 0.85 + fraction * 0.95


def _setup_optimizer(
    model: Any,
    cfg: DictConfig,
    *,
    total_batch_size: int,
    base_vocab: int,
    n_sae_features: int,
    feature_vocab: FeatureVocab | None,
) -> Any:
    batch_scale = math.sqrt(total_batch_size / _BATCH_SIZE_REFERENCE)
    target_tokens = int(float(cfg.target_param_data_ratio) * _scaling_parameter_count(model))
    d12 = build_encdec_meta_model_from_training_cfg(
        cfg,
        base_vocab=base_vocab,
        n_sae_features=n_sae_features,
        depth=12,
        feature_vocab=feature_vocab,
    )
    d_ref = float(cfg.target_param_data_ratio) * _scaling_parameter_count(d12)
    weight_decay = float(cfg.weight_decay) * batch_scale * (d_ref / target_tokens)
    return model.setup_optimizer(
        unembedding_lr=float(cfg.unembedding_lr) * batch_scale,
        embedding_lr=float(cfg.embedding_lr) * batch_scale,
        scalar_lr=float(cfg.scalar_lr) * batch_scale,
        adam_betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        matrix_lr=float(cfg.matrix_lr) * batch_scale,
        weight_decay=weight_decay,
        encoder_embedding_lr=float(cfg.encoder_embedding_lr) * batch_scale,
        encoder_matrix_lr=float(cfg.encoder_matrix_lr) * batch_scale,
        xattn_matrix_lr=float(cfg.xattn_matrix_lr) * batch_scale,
    )


def _canonical_groups_at_checkpoint(
    optimizer: Any,
    cfg: DictConfig,
    *,
    step: int,
) -> list[dict[str, Any]]:
    groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
    if step == 0:
        return groups
    previous_step = step - 1
    cfg_mapping = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    lrm = _lr_multiplier(cfg_mapping, step=previous_step)
    iterations = _require_int(cfg.num_iterations, name="num_iterations", minimum=1)
    for group in groups:
        group["lr"] = group["initial_lr"] * lrm
        if _kind_value(group["kind"]) == OptimizerKind.MUON.value:
            group["momentum"] = _muon_momentum(previous_step)
            group["weight_decay"] *= 1 - previous_step / iterations
    return groups


def _build_optimizer_layout(
    model: Any,
    model_state: Mapping[str, t.Tensor],
    optimizer: Any,
) -> OptimizerLayout:
    names_by_object = {id(parameter): name for name, parameter in model.named_parameters()}
    state_groups = optimizer.state_dict()["param_groups"]
    if len(state_groups) != len(optimizer.param_groups):
        raise AssertionError("optimizer state/object group counts disagree")
    parameters: dict[int, ParameterLayout] = {}
    groups: list[GroupLayout] = []
    for group_index, (state_group, object_group) in enumerate(
        zip(state_groups, optimizer.param_groups, strict=True)
    ):
        param_ids = tuple(state_group["params"])
        object_params = object_group["params"]
        if len(param_ids) != len(object_params) or not param_ids:
            raise ValueError(f"optimizer group {group_index} has an invalid parameter list")
        kind = _kind_value(state_group["kind"])
        for param_id, parameter in zip(param_ids, object_params, strict=True):
            name = names_by_object.get(id(parameter))
            if name is None or name not in model_state:
                raise ValueError(f"optimizer parameter {param_id} has no model-state identity")
            tensor = model_state[name]
            if not isinstance(tensor, t.Tensor) or tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(f"model-state tensor {name!r} disagrees with the meta model")
            if param_id in parameters:
                raise ValueError(f"optimizer parameter id {param_id} appears in multiple groups")
            parameters[param_id] = ParameterLayout(
                param_id=param_id,
                name=name,
                shape=tuple(parameter.shape),
                dtype=tensor.dtype,
                numel=parameter.numel(),
                kind=kind,
                group_index=group_index,
            )
        groups.append(GroupLayout(kind=kind, param_ids=param_ids))
    return OptimizerLayout(parameters=parameters, groups=tuple(groups))


def _assert_group_structure(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    context: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{context} optimizer param_groups are not canonical")


def _tensor_is_finite(tensor: t.Tensor) -> bool:
    if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
        return True
    flat = tensor.reshape(-1)
    chunk_size = 8 * 1024 * 1024
    return all(
        bool(t.isfinite(flat[start : start + chunk_size]).all())
        for start in range(0, flat.numel(), chunk_size)
    )


def _tensor_equal(left: t.Tensor, right: t.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    chunk_size = 8 * 1024 * 1024
    return all(
        t.equal(left_flat[start : start + chunk_size], right_flat[start : start + chunk_size])
        for start in range(0, left_flat.numel(), chunk_size)
    )


def _assert_optimizer_shard_exact(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require a serialized optimizer shard to equal its in-memory source."""

    if set(actual) != {"state", "param_groups"} or set(expected) != {
        "state",
        "param_groups",
    }:
        raise ValueError(f"{context} optimizer shard has unexpected top-level fields")
    if actual["param_groups"] != expected["param_groups"]:
        raise ValueError(f"{context} serialized optimizer param_groups changed")
    actual_state = actual["state"]
    expected_state = expected["state"]
    if not isinstance(actual_state, Mapping) or not isinstance(expected_state, Mapping):
        raise ValueError(f"{context} serialized optimizer state is malformed")
    if set(actual_state) != set(expected_state):
        raise ValueError(f"{context} serialized optimizer state ids changed")
    for param_id, expected_fields in expected_state.items():
        actual_fields = actual_state[param_id]
        if not isinstance(actual_fields, Mapping) or not isinstance(expected_fields, Mapping):
            raise ValueError(f"{context} optimizer state {param_id} is malformed")
        if set(actual_fields) != set(expected_fields):
            raise ValueError(f"{context} optimizer state {param_id} fields changed")
        for field, expected_value in expected_fields.items():
            actual_value = actual_fields[field]
            if isinstance(expected_value, t.Tensor):
                if not isinstance(actual_value, t.Tensor) or not _tensor_equal(
                    actual_value,
                    expected_value,
                ):
                    raise ValueError(
                        f"{context} optimizer state {param_id}/{field} changed "
                        "during serialization"
                    )
            elif actual_value != expected_value:
                raise ValueError(
                    f"{context} optimizer state {param_id}/{field} changed "
                    "during serialization"
                )


def _adam_is_replicated(parameter: ParameterLayout, world_size: int) -> bool:
    return parameter.numel < 1024 or parameter.shape[0] % world_size != 0


def _validate_optimizer_shards(
    shards: tuple[dict[str, Any], ...],
    *,
    layout: OptimizerLayout,
    canonical_groups: list[dict[str, Any]],
    step: int,
) -> None:
    world_size = len(shards)
    expected_ids = layout.expected_state_ids if step > 0 else frozenset()
    for rank, shard in enumerate(shards):
        if set(shard) != {"state", "param_groups"}:
            raise ValueError(f"optimizer rank {rank} must contain state and param_groups only")
        groups = shard["param_groups"]
        state = shard["state"]
        if not isinstance(groups, list) or not isinstance(state, dict):
            raise ValueError(f"optimizer rank {rank} has malformed state_dict containers")
        _assert_group_structure(
            groups,
            canonical_groups,
            context=f"optimizer rank {rank}",
        )
        if frozenset(state) != expected_ids:
            raise ValueError(
                f"optimizer rank {rank} state ids differ: "
                f"actual={sorted(state)}, expected={sorted(expected_ids)}"
            )

    if step == 0:
        return

    for group in layout.groups:
        if group.kind == OptimizerKind.ADAMW.value:
            for param_id in group.param_ids:
                parameter = layout.parameters[param_id]
                replicated = _adam_is_replicated(parameter, world_size)
                expected_shape = (
                    parameter.shape
                    if replicated
                    else (parameter.shape[0] // world_size, *parameter.shape[1:])
                )
                reference_step: int | None = None
                reference_buffers: dict[str, t.Tensor] = {}
                for rank, shard in enumerate(shards):
                    state = shard["state"][param_id]
                    if not isinstance(state, dict) or set(state) != {
                        "step",
                        "exp_avg",
                        "exp_avg_sq",
                    }:
                        raise ValueError(
                            f"Adam state {parameter.name!r} rank {rank} has wrong fields"
                        )
                    state_step = state["step"]
                    if type(state_step) is not int or state_step != step:
                        raise ValueError(
                            f"Adam state {parameter.name!r} step {state_step!r} does not "
                            f"match checkpoint step {step}"
                        )
                    if reference_step is None:
                        reference_step = state_step
                    elif state_step != reference_step:
                        raise ValueError(f"Adam state {parameter.name!r} step differs across ranks")
                    for field in ("exp_avg", "exp_avg_sq"):
                        tensor = state[field]
                        if (
                            not isinstance(tensor, t.Tensor)
                            or tuple(tensor.shape) != expected_shape
                            or tensor.dtype != parameter.dtype
                        ):
                            raise ValueError(
                                f"Adam state {parameter.name!r}/{field} rank {rank} "
                                "has the wrong shape or dtype"
                            )
                        if not _tensor_is_finite(tensor):
                            raise ValueError(f"Adam state {parameter.name!r}/{field} is non-finite")
                        if replicated:
                            reference = reference_buffers.setdefault(field, tensor)
                            if not _tensor_equal(reference, tensor):
                                raise ValueError(
                                    f"replicated Adam state {parameter.name!r}/{field} "
                                    "differs across source ranks"
                                )
        else:
            first_id = group.param_ids[0]
            parameter = layout.parameters[first_id]
            chunk = math.ceil(len(group.param_ids) / world_size)
            second_tail = (
                (parameter.shape[-2], 1)
                if parameter.shape[-2] >= parameter.shape[-1]
                else (1, parameter.shape[-1])
            )
            for rank, shard in enumerate(shards):
                state = shard["state"][first_id]
                if not isinstance(state, dict) or set(state) != {
                    "momentum_buffer",
                    "second_momentum_buffer",
                }:
                    raise ValueError(f"Muon group {parameter.name!r} rank {rank} has wrong fields")
                expected_shapes = {
                    "momentum_buffer": (chunk, *parameter.shape),
                    "second_momentum_buffer": (chunk, *second_tail),
                }
                owned = min(chunk, max(0, len(group.param_ids) - rank * chunk))
                for field, expected_shape in expected_shapes.items():
                    tensor = state[field]
                    if (
                        not isinstance(tensor, t.Tensor)
                        or tuple(tensor.shape) != expected_shape
                        or tensor.dtype != parameter.dtype
                    ):
                        raise ValueError(
                            f"Muon state {parameter.name!r}/{field} rank {rank} "
                            "has the wrong shape or dtype"
                        )
                    if not _tensor_is_finite(tensor):
                        raise ValueError(f"Muon state {parameter.name!r}/{field} is non-finite")
                    if owned < chunk and bool(tensor[owned:].count_nonzero()):
                        raise ValueError(
                            f"Muon state {parameter.name!r}/{field} rank {rank} "
                            "has nonzero source padding"
                        )


def _slice_logical_chunks(
    chunks: list[t.Tensor],
    *,
    logical_length: int,
    start: int,
    end: int,
) -> t.Tensor:
    if not 0 <= start <= end <= logical_length:
        raise ValueError(f"invalid logical slice [{start}, {end}) of {logical_length}")
    if start == end:
        return chunks[0].new_empty((0, *chunks[0].shape[1:]))
    chunk_size = chunks[0].shape[0]
    parts: list[t.Tensor] = []
    cursor = start
    while cursor < end:
        source_rank = cursor // chunk_size
        offset = cursor % chunk_size
        take = min(end - cursor, chunk_size - offset)
        parts.append(chunks[source_rank][offset : offset + take])
        cursor += take
    if len(parts) == 1:
        return parts[0].clone()
    return t.cat(parts, dim=0)


def _reshard_adam_tensor(
    source_tensors: list[t.Tensor],
    *,
    parameter: ParameterLayout,
    source_world_size: int,
    destination_world_size: int,
    destination_rank: int,
) -> t.Tensor:
    source_replicated = _adam_is_replicated(parameter, source_world_size)
    destination_replicated = _adam_is_replicated(parameter, destination_world_size)
    if source_replicated:
        full = source_tensors[0]
        if destination_replicated:
            return full.clone()
        rank_rows = parameter.shape[0] // destination_world_size
        start = destination_rank * rank_rows
        return full[start : start + rank_rows].clone()
    if destination_replicated:
        return t.cat(source_tensors, dim=0)
    rank_rows = parameter.shape[0] // destination_world_size
    start = destination_rank * rank_rows
    return _slice_logical_chunks(
        source_tensors,
        logical_length=parameter.shape[0],
        start=start,
        end=start + rank_rows,
    )


def _build_destination_optimizer_shard(
    source_shards: tuple[dict[str, Any], ...],
    *,
    layout: OptimizerLayout,
    target_param_groups: list[dict[str, Any]],
    destination_world_size: int,
    destination_rank: int,
) -> dict[str, Any]:
    source_world_size = len(source_shards)
    if not source_shards[0]["state"]:
        return {"state": {}, "param_groups": copy.deepcopy(target_param_groups)}
    target_state: dict[int, dict[str, Any]] = {}
    for group in layout.groups:
        if group.kind == OptimizerKind.ADAMW.value:
            for param_id in group.param_ids:
                parameter = layout.parameters[param_id]
                source_state = [shard["state"][param_id] for shard in source_shards]
                target_state[param_id] = {
                    "step": source_state[0]["step"],
                    "exp_avg": _reshard_adam_tensor(
                        [state["exp_avg"] for state in source_state],
                        parameter=parameter,
                        source_world_size=source_world_size,
                        destination_world_size=destination_world_size,
                        destination_rank=destination_rank,
                    ),
                    "exp_avg_sq": _reshard_adam_tensor(
                        [state["exp_avg_sq"] for state in source_state],
                        parameter=parameter,
                        source_world_size=source_world_size,
                        destination_world_size=destination_world_size,
                        destination_rank=destination_rank,
                    ),
                }
        else:
            first_id = group.param_ids[0]
            parameter = layout.parameters[first_id]
            source_chunk = math.ceil(len(group.param_ids) / source_world_size)
            destination_chunk = math.ceil(len(group.param_ids) / destination_world_size)
            start = destination_rank * destination_chunk
            owned = min(destination_chunk, max(0, len(group.param_ids) - start))
            source_states = [shard["state"][first_id] for shard in source_shards]
            buffers: dict[str, t.Tensor] = {}
            for field in ("momentum_buffer", "second_momentum_buffer"):
                source_buffers = [state[field] for state in source_states]
                output = source_buffers[0].new_zeros(
                    (destination_chunk, *source_buffers[0].shape[1:])
                )
                if owned:
                    output[:owned].copy_(
                        _slice_logical_chunks(
                            source_buffers,
                            logical_length=len(group.param_ids),
                            start=start,
                            end=start + owned,
                        )
                    )
                buffers[field] = output
            if source_chunk <= 0:  # pragma: no cover - guarded by non-empty groups
                raise AssertionError("source Muon chunk must be positive")
            target_state[first_id] = buffers
    return {
        "state": target_state,
        "param_groups": copy.deepcopy(target_param_groups),
    }


def _validate_model_contract(
    source_model: Any,
    destination_model: Any,
    model_state: Mapping[str, t.Tensor],
) -> None:
    source_contract = _checkpoint_model_contract(source_model)
    destination_contract = _checkpoint_model_contract(destination_model)
    if source_contract != destination_contract:
        raise ValueError("destination training config changes the model state-dict contract")
    if set(model_state) != set(source_contract):
        missing = sorted(set(source_contract) - set(model_state))
        extra = sorted(set(model_state) - set(source_contract))
        raise ValueError(
            f"model checkpoint keys differ from the configured model: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    for name, (shape, dtype) in source_contract.items():
        tensor = model_state.get(name)
        if (
            not isinstance(tensor, t.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
        ):
            raise ValueError(
                f"model checkpoint tensor {name!r} does not match configured "
                "shape/dtype"
            )


def _checkpoint_model_contract(model: Any) -> dict[str, tuple[tuple[int, ...], t.dtype]]:
    """Return the exact native checkpoint shape/dtype contract.

    Production constructs parameters in FP32 on ``meta``, materializes them on
    CUDA, and then ``init_weights()`` casts only token/feature/magnitude/value
    embeddings to the compute dtype. The offline migrator reconstructs on a CPU
    host, so its ambient ``COMPUTE_DTYPE`` and raw meta-model dtypes cannot
    describe a BF16 production checkpoint. The supported W31/W32 handoff is
    explicitly BF16; keep that policy local and fail closed on every other
    parameter and buffer dtype.
    """

    state = model.state_dict()
    names_by_parameter = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    compute_dtype_parameters = (
        list(model._decoder_embedding.parameters())
        + list(model.value_embeds.parameters())
        + list(model.encoder.feature_embed.parameters())
        + list(model.encoder.magnitude_embed.parameters())
        + list(model.encoder.value_embeds.parameters())
    )
    parameter_ids = [id(parameter) for parameter in compute_dtype_parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("model compute-dtype parameter groups overlap")
    missing_parameters = [
        parameter_id
        for parameter_id in parameter_ids
        if parameter_id not in names_by_parameter
    ]
    if missing_parameters:
        raise ValueError("model compute-dtype parameter has no state-dict identity")
    compute_dtype_names = {
        names_by_parameter[parameter_id] for parameter_id in parameter_ids
    }
    unknown_names = compute_dtype_names - set(state)
    if unknown_names:
        raise ValueError(
            "model compute-dtype parameter names are absent from the state dict: "
            f"{sorted(unknown_names)[:8]}"
        )
    return {
        name: (
            tuple(tensor.shape),
            (
                _CANONICAL_CHECKPOINT_COMPUTE_DTYPE
                if name in compute_dtype_names
                else tensor.dtype
            ),
        )
        for name, tensor in state.items()
    }


def _validate_config_contract(
    source_meta: Mapping[str, Any],
    destination_cfg: DictConfig,
    *,
    source_world_size: int,
    destination_world_size: int,
    step: int,
) -> tuple[dict[str, Any], int, int, int, int]:
    source_user = source_meta.get("user_config")
    if not isinstance(source_user, Mapping):
        raise ValueError("checkpoint metadata must contain a mapping user_config")
    feature_prediction_config_from_metadata(source_meta)
    source_user = dict(source_user)
    if read_checkpoint_conditioning(source_meta).conditioning_cooldown is not None:
        raise ValueError("canonical topology migration requires conditioning_cooldown disabled")
    target_user = _checkpoint_user_config(destination_cfg)
    differences = {
        key
        for key in set(source_user) | set(target_user)
        if key not in source_user
        or key not in target_user
        or not _strict_equal(source_user[key], target_user[key])
    }
    unexpected = differences - _ALLOWED_CONFIG_DIFFERENCES
    if unexpected:
        detail = {
            key: {
                "source": source_user.get(key, "<missing>"),
                "destination": target_user.get(key, "<missing>"),
            }
            for key in sorted(unexpected)
        }
        raise ValueError(
            "destination config differs from the source beyond topology identity: "
            f"{json.dumps(detail, sort_keys=True, default=str)}"
        )
    if source_user.get("remote_training") is not None:
        raise ValueError("remote-training checkpoints are not supported by topology migration")
    if bool(source_user.get("fp8")) or bool(target_user.get("fp8")):
        raise ValueError("canonical topology migration requires BF16 on both sides")
    if (
        float(source_user.get("label_dropout", -1)) != 0.0
        or float(target_user.get("label_dropout", -1)) != 0.0
    ):
        raise ValueError("canonical topology migration requires label_dropout=0")
    if (
        source_user.get("use_precomputed_batches") is not True
        or target_user.get("use_precomputed_batches") is not True
    ):
        raise ValueError("canonical topology migration requires precomputed batches")

    device_batch_size = _require_int(
        source_user.get("device_batch_size"),
        name="source device_batch_size",
        minimum=1,
    )
    max_seq_len = _require_int(
        source_user.get("max_seq_len"),
        name="source max_seq_len",
        minimum=1,
    )
    if target_user.get("device_batch_size") != device_batch_size:
        raise ValueError("destination device_batch_size must equal the source")
    if target_user.get("max_seq_len") != max_seq_len:
        raise ValueError("destination max_seq_len must equal the source")
    if target_user.get("num_iterations") != source_user.get("num_iterations"):
        raise ValueError("destination num_iterations must equal the source")
    iterations = _require_int(
        source_user.get("num_iterations"),
        name="source num_iterations",
        minimum=1,
    )
    if step > iterations:
        raise ValueError(f"checkpoint step {step} exceeds num_iterations={iterations}")

    source_total = _require_int(
        source_user.get("total_batch_size"),
        name="source total_batch_size",
        minimum=1,
    )
    destination_total = _require_int(
        target_user.get("total_batch_size"),
        name="destination total_batch_size",
        minimum=1,
    )
    source_micro = source_world_size * device_batch_size * max_seq_len
    destination_micro = destination_world_size * device_batch_size * max_seq_len
    if source_total % source_micro or destination_total % destination_micro:
        raise ValueError("source or destination total batch is not topology-divisible")
    source_accum = source_total // source_micro
    destination_accum = destination_total // destination_micro
    if source_accum != destination_accum:
        raise ValueError("destination gradient accumulation must equal the source")
    if source_total * destination_world_size != destination_total * source_world_size:
        raise ValueError("destination total batch must change only by the world-size ratio")

    target_user["resume_from_step"] = step
    return target_user, source_total, destination_total, device_batch_size, source_accum


def prepare_migration(cfg: DictConfig) -> PreparedMigration:
    """Load and validate every source component without writing destination files."""

    source_dir = Path(str(cfg.source_checkpoint_dir)).resolve()
    destination_dir = Path(str(cfg.destination_dir)).resolve()
    packed_row_dir = Path(str(cfg.packed_row_dir)).resolve()
    selection_dir = Path(str(cfg.selection_dir)).resolve()
    step = _require_int(cfg.source_step, name="source_step")
    destination_world_size = _require_int(
        cfg.destination_world_size,
        name="destination_world_size",
        minimum=1,
    )
    for input_directory in (source_dir, packed_row_dir, selection_dir):
        if destination_dir == input_directory or input_directory in destination_dir.parents:
            raise ValueError(
                "destination_dir must not be inside a migration input directory: "
                f"destination={destination_dir}, input={input_directory}"
            )
    if destination_dir.exists():
        raise FileExistsError(f"destination already exists: {destination_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source checkpoint directory does not exist: {source_dir}")

    model_path = source_dir / f"model_{step:06d}.pt"
    meta_path = source_dir / f"meta_{step:06d}.json"
    if _MODEL_RE.fullmatch(model_path.name) is None or not model_path.is_file():
        raise FileNotFoundError(f"source model file does not exist: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"source metadata file does not exist: {meta_path}")
    optimizer_paths = _discover_optimizer_paths(source_dir, step)
    source_file_identities = {
        path: _file_identity(path) for path in (model_path, meta_path, *optimizer_paths)
    }
    source_meta = _load_json(meta_path)
    if type(source_meta.get("step")) is not int or source_meta["step"] != step:
        raise ValueError(f"source metadata step {source_meta.get('step')!r} does not match {step}")
    present_remote = sorted(_REMOTE_META_KEYS.intersection(source_meta))
    if present_remote:
        raise ValueError(f"remote checkpoint metadata is unsupported: {present_remote}")

    source_world_size = len(optimizer_paths)
    destination_cfg = cfg.destination_training
    if not isinstance(destination_cfg, DictConfig):
        raise TypeError("destination_training must be a Hydra DictConfig")
    target_user, source_total, destination_total, device_batch_size, grad_accum = (
        _validate_config_contract(
            source_meta,
            destination_cfg,
            source_world_size=source_world_size,
            destination_world_size=destination_world_size,
            step=step,
        )
    )
    metadata_device_batch_size = _require_int(
        source_meta.get("device_batch_size"),
        name="metadata device_batch_size",
        minimum=1,
    )
    if metadata_device_batch_size != device_batch_size:
        raise ValueError(
            "checkpoint metadata device_batch_size disagrees with user_config: "
            f"{metadata_device_batch_size} != {device_batch_size}"
        )
    metadata_max_seq_len = _require_int(
        source_meta.get("max_seq_len"),
        name="metadata max_seq_len",
        minimum=1,
    )
    if metadata_max_seq_len != target_user.get("max_seq_len"):
        raise ValueError(
            "checkpoint metadata max_seq_len disagrees with user_config: "
            f"{metadata_max_seq_len} != {target_user.get('max_seq_len')!r}"
        )

    model_state_raw = _safe_mmap_load(model_path)
    if not all(
        isinstance(name, str) and isinstance(tensor, t.Tensor)
        for name, tensor in model_state_raw.items()
    ):
        raise ValueError("model checkpoint must be a string-to-tensor state dict")
    model_state = cast(Mapping[str, t.Tensor], model_state_raw)
    model_config = source_meta.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("source metadata must contain model_config")
    base_vocab = _require_int(model_config.get("vocab_size"), name="model vocab_size", minimum=1)
    n_sae_features, feature_vocab = feature_vocab_from_state_dict(model_state)
    source_cfg = OmegaConf.create(source_meta["user_config"])
    source_model = build_encdec_meta_model_from_training_cfg(
        source_cfg,
        base_vocab=base_vocab,
        n_sae_features=n_sae_features,
        feature_vocab=feature_vocab,
    )
    expected_model_config = asdict(source_model.dec_config)
    if not _strict_equal(dict(model_config), expected_model_config):
        raise ValueError(
            "source metadata model_config disagrees with the model rebuilt from "
            "user_config"
        )
    recorded_feature_vocab = source_meta.get(FEATURE_VOCAB_META_KEY, None)
    if feature_vocab is None:
        if FEATURE_VOCAB_META_KEY in source_meta:
            raise ValueError(
                "source metadata records a feature vocabulary but the model checkpoint "
                "has no compact vocabulary buffers"
            )
    elif (
        FEATURE_VOCAB_META_KEY in source_meta
        and not _strict_equal(recorded_feature_vocab, feature_vocab.metadata_dict())
    ):
        raise ValueError(
            "source metadata feature vocabulary disagrees with the model checkpoint buffers"
        )
    destination_model = build_encdec_meta_model_from_training_cfg(
        destination_cfg,
        base_vocab=base_vocab,
        n_sae_features=n_sae_features,
        feature_vocab=feature_vocab,
    )
    _validate_model_contract(source_model, destination_model, model_state)

    source_optimizer = _setup_optimizer(
        source_model,
        source_cfg,
        total_batch_size=source_total,
        base_vocab=base_vocab,
        n_sae_features=n_sae_features,
        feature_vocab=feature_vocab,
    )
    destination_optimizer = _setup_optimizer(
        destination_model,
        destination_cfg,
        total_batch_size=destination_total,
        base_vocab=base_vocab,
        n_sae_features=n_sae_features,
        feature_vocab=feature_vocab,
    )
    layout = _build_optimizer_layout(source_model, model_state, source_optimizer)
    destination_layout = _build_optimizer_layout(
        destination_model,
        model_state,
        destination_optimizer,
    )
    if layout != destination_layout:
        raise ValueError("destination optimizer parameter grouping differs from the source")
    source_groups = _canonical_groups_at_checkpoint(source_optimizer, source_cfg, step=step)
    target_groups = _canonical_groups_at_checkpoint(
        destination_optimizer,
        destination_cfg,
        step=step,
    )

    optimizer_shards = tuple(_safe_mmap_load(path) for path in optimizer_paths)
    _validate_optimizer_shards(
        optimizer_shards,
        layout=layout,
        canonical_groups=source_groups,
        step=step,
    )

    inventory = PackedRowInventory.from_directory(packed_row_dir)
    max_seq_len = metadata_max_seq_len
    if inventory.context_len != max_seq_len:
        raise ValueError(
            f"packed-row context length {inventory.context_len} != checkpoint {max_seq_len}"
        )
    packed_row_fingerprint = validate_selection_inventory(packed_row_dir, selection_dir)
    dataloader_state = source_meta.get("dataloader_state_dict")
    if not isinstance(dataloader_state, Mapping):
        raise ValueError("source checkpoint has no dataloader state envelope")
    target_envelope, replay_plan = build_replay_boundary_envelope(
        dataloader_state,
        inventory=inventory,
        source_world_size=source_world_size,
        destination_world_size=destination_world_size,
        device_batch_size=device_batch_size,
        seed=_require_int(target_user.get("seed"), name="training seed"),
    )
    source_batch_idx = target_envelope["rank_states"][0]["batch_idx"]
    expected_batch_idx = step * grad_accum
    if source_batch_idx != expected_batch_idx:
        raise ValueError(
            "source dataloader batch_idx disagrees with checkpoint step and gradient "
            f"accumulation: {source_batch_idx} != {step} * {grad_accum}"
        )
    if not isinstance(source_meta.get("loop_state"), Mapping):
        raise ValueError("source checkpoint metadata must contain a mapping loop_state")
    max_replay_rows = cfg.get("max_replay_rows")
    if max_replay_rows is not None:
        maximum = _require_int(max_replay_rows, name="max_replay_rows")
        if replay_plan.replay_rows > maximum:
            raise ValueError(
                f"replay plan requires {replay_plan.replay_rows} rows, exceeding "
                f"max_replay_rows={maximum}"
            )

    destination_meta = copy.deepcopy(source_meta)
    destination_meta["user_config"] = target_user
    if target_user.get("feature_prediction") is not None:
        destination_meta["feature_prediction_config"] = dict(target_user["feature_prediction"])
    destination_meta["device_batch_size"] = device_batch_size
    destination_meta["dataloader_state_dict"] = target_envelope
    destination_meta["topology_migration"] = {
        "schema_version": 1,
        "source_world_size": source_world_size,
        "destination_world_size": destination_world_size,
        "step_policy": "unchanged",
        "optimizer_state": "tensor_exact_canonical_destination_groups",
        "packed_row_fingerprint": packed_row_fingerprint,
        "dataloader_state": replay_plan.as_dict(),
    }

    plan = MigrationPlan(
        source_world_size=source_world_size,
        destination_world_size=destination_world_size,
        step=step,
        source_total_batch_size=source_total,
        destination_total_batch_size=destination_total,
        device_batch_size=device_batch_size,
        max_seq_len=max_seq_len,
        grad_accum_steps=grad_accum,
        lr_scale_ratio=math.sqrt(destination_total / source_total),
        packed_row_fingerprint=packed_row_fingerprint,
        dataloader=replay_plan,
    )
    _assert_files_unchanged(source_file_identities)
    return PreparedMigration(
        plan=plan,
        source_dir=source_dir,
        destination_dir=destination_dir,
        model_path=model_path,
        model_state=model_state,
        source_meta=source_meta,
        destination_meta=destination_meta,
        source_optimizer_paths=optimizer_paths,
        source_optimizer_shards=optimizer_shards,
        target_param_groups=target_groups,
        optimizer_layout=layout,
        packed_inventory=inventory,
        source_file_identities=source_file_identities,
    )


def _copy_with_sha256(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(8 * 1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        shutil.copystat(source, destination)
        os.fsync(writer.fileno())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute_migration(prepared: PreparedMigration) -> Path:
    """Write a prepared migration into a fresh destination atomically."""

    destination = prepared.destination_dir
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.migration-",
            dir=destination.parent,
        )
    )
    os.chmod(stage, stat.S_IMODE(prepared.source_dir.stat().st_mode))
    step = prepared.plan.step
    logger.info(f"Writing migration staging directory {stage}")
    _assert_files_unchanged(prepared.source_file_identities)
    for rank in range(prepared.plan.destination_world_size):
        shard = _build_destination_optimizer_shard(
            prepared.source_optimizer_shards,
            layout=prepared.optimizer_layout,
            target_param_groups=prepared.target_param_groups,
            destination_world_size=prepared.plan.destination_world_size,
            destination_rank=rank,
        )
        path = stage / f"optim_{step:06d}_rank{rank:d}.pt"
        atomic_torch_save(shard, path)
        serialized_shard = _safe_mmap_load(path)
        _assert_optimizer_shard_exact(
            serialized_shard,
            shard,
            context=f"optimizer rank {rank}",
        )
        del serialized_shard
        del shard
        gc.collect()

    serialized_optimizer_shards = tuple(
        _safe_mmap_load(stage / f"optim_{step:06d}_rank{rank:d}.pt")
        for rank in range(prepared.plan.destination_world_size)
    )
    _validate_optimizer_shards(
        serialized_optimizer_shards,
        layout=prepared.optimizer_layout,
        canonical_groups=prepared.target_param_groups,
        step=step,
    )
    del serialized_optimizer_shards
    gc.collect()

    meta_destination = stage / f"meta_{step:06d}.json"
    atomic_json_dump(prepared.destination_meta, meta_destination)
    _assert_json_file_exact(meta_destination, prepared.destination_meta)
    model_destination = stage / f"model_{step:06d}.pt"
    model_sha256 = _copy_with_sha256(prepared.model_path, model_destination)
    destination_model_sha256 = _sha256_file(model_destination)
    if destination_model_sha256 != model_sha256:
        raise ValueError(
            "model checkpoint changed during copy: "
            f"source={model_sha256}, destination={destination_model_sha256}"
        )
    manifest = {
        "schema_version": 1,
        "plan": prepared.plan.as_dict(),
        "source": {
            "checkpoint_dir": str(prepared.source_dir),
            "model_file": prepared.model_path.name,
            "model_sha256": model_sha256,
            "optimizer_files": [path.name for path in prepared.source_optimizer_paths],
        },
        "destination": {
            "checkpoint_dir": str(destination),
            "model_file": model_destination.name,
            "optimizer_files": [
                f"optim_{step:06d}_rank{rank:d}.pt"
                for rank in range(prepared.plan.destination_world_size)
            ],
        },
    }
    manifest_destination = stage / "topology_migration_manifest.json"
    atomic_json_dump(manifest, manifest_destination)
    _assert_json_file_exact(manifest_destination, manifest)

    expected_names = {
        f"model_{step:06d}.pt",
        f"meta_{step:06d}.json",
        "topology_migration_manifest.json",
        *{
            f"optim_{step:06d}_rank{rank:d}.pt"
            for rank in range(prepared.plan.destination_world_size)
        },
    }
    actual_names = {path.name for path in stage.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"staged checkpoint inventory mismatch: actual={actual_names}, expected={expected_names}"
        )
    _assert_files_unchanged(prepared.source_file_identities)
    _fsync_directory(stage)
    if destination.exists():
        raise FileExistsError(f"destination appeared during migration: {destination}")
    os.rename(stage, destination)
    _fsync_directory(destination.parent)
    logger.info(f"Published complete migrated checkpoint at {destination}")
    return destination


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="tools/migrate_encdec_train_state",
)
def main(cfg: DictConfig) -> None:
    dry_run = _require_bool(cfg.get("dry_run"), name="dry_run")
    prepared = prepare_migration(cfg)
    logger.info(
        "Validated topology migration plan:\n" + json.dumps(prepared.plan.as_dict(), indent=2)
    )
    if dry_run:
        logger.info("Dry run complete; no destination files were written")
        return
    execute_migration(prepared)


if __name__ == "__main__":
    main()
