"""Fail-closed restart checkpoints for distributed Hugging Face decoder SFT."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
import torch.distributed as dist
from transformers import PreTrainedModel

from nanochat.atomic_io import atomic_json_dump
from nanochat.checkpoint_manager import save_checkpoint
from steering._common.ddp import rank_synchronized_stage
from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.sft.hf_contract import HFDistributedImplementation

HF_RESTART_FORMAT = "huggingface_decoder_sft_restart"
HF_RESTART_FORMAT_VERSION = 1
_RUNTIME_KEYS = {
    "dataloader_state",
    "completed_steps",
    "smooth_train_loss",
    "latest_val_bpb",
    "min_val_bpb",
    "total_training_time",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state",
}
_NON_SEMANTIC_CONFIG_KEYS = {
    "profile_output_dir",
    "profile_timing",
    "profile_trace_ranks",
    "profile_trace_steps",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalized_json(value: object) -> Any:
    """Return a plain strict-JSON copy and reject non-finite/non-JSON state."""
    return json.loads(_canonical_json(value))


@dataclass(frozen=True)
class HFRestartIdentity:
    """Canonical semantic payload and SHA-256 fence for one training run."""

    payload: dict[str, Any]
    sha256: str


def build_hf_restart_identity(
    *,
    resolved_config: Mapping[str, Any],
    repository: str | Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer_contract: Mapping[str, Any],
    train_mixture_fingerprint: Mapping[str, Any],
    val_mixture_fingerprint: Mapping[str, Any],
    world_size: int,
    local_world_size: int,
) -> HFRestartIdentity:
    """Build the exhaustive same-topology restart identity."""
    if world_size <= 0 or local_world_size <= 0 or world_size % local_world_size != 0:
        raise ValueError(
            f"invalid restart topology world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )
    config = _normalized_json(dict(resolved_config))
    if not isinstance(config, dict):
        raise TypeError("resolved_config must normalize to a JSON object")
    huggingface = config.get("huggingface")
    if not isinstance(huggingface, dict):
        raise ValueError("resolved_config.huggingface must be a mapping")
    for key in _NON_SEMANTIC_CONFIG_KEYS:
        config.pop(key, None)
    # Diagnostics do not affect training state and may be disabled on resume.
    # The restore selector changes by definition; all other keys stay fenced.
    huggingface["restart_from_step"] = -1
    payload = _normalized_json(
        {
            "schema_version": 1,
            "repository": dict(repository) if isinstance(repository, Mapping) else repository,
            "resolved_config": config,
            "source": dict(source),
            "tokenizer_contract": dict(tokenizer_contract),
            "mixture_fingerprints": {
                "train": dict(train_mixture_fingerprint),
                "validation": dict(val_mixture_fingerprint),
            },
            "topology": {
                "world_size": world_size,
                "local_world_size": local_world_size,
            },
        }
    )
    if not isinstance(payload, dict):
        raise RuntimeError("restart identity unexpectedly normalized to a non-object")
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return HFRestartIdentity(payload=payload, sha256=digest)


def assert_distributed_hf_restart_identity(
    identity: HFRestartIdentity,
    *,
    device: str | t.device,
) -> None:
    """Require every rank to have built the same canonical semantic digest."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    digest = t.tensor(
        list(bytes.fromhex(identity.sha256)),
        # Keep the collective on the integer dtype with the broadest NCCL and
        # Gloo reduction support. Values remain the exact digest bytes.
        dtype=t.int32,
        device=device,
    )
    minimum = digest.clone()
    maximum = digest.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not t.equal(minimum, maximum):
        raise RuntimeError(
            "HF restart semantic identity differs across distributed ranks"
        )


def restart_dir(output_root: str | Path) -> Path:
    return Path(output_root) / "restart"


def restart_paths(output_root: str | Path, step: int, *, world_size: int) -> dict[str, Path]:
    if step < 0:
        raise ValueError(f"restart step must be non-negative, got {step}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    root = restart_dir(output_root)
    paths = {
        "model": root / f"model_{step:06d}.pt",
        "meta": root / f"meta_{step:06d}.json",
        "complete": root / f"complete_{step:06d}.json",
        "lock": root / f".{step:06d}.lock",
    }
    for rank in range(world_size):
        paths[f"rank_{rank}"] = root / f"optim_{step:06d}_rank{rank}.pt"
    return paths


def _expected_data_paths(output_root: str | Path, step: int, *, world_size: int) -> list[Path]:
    paths = restart_paths(output_root, step, world_size=world_size)
    return [paths["model"], paths["meta"]] + [
        paths[f"rank_{rank}"] for rank in range(world_size)
    ]


def _inventory(paths: list[Path]) -> dict[str, int]:
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"restart checkpoint is incomplete; missing={missing}")
    inventory = {path.name: path.stat().st_size for path in paths}
    empty = [name for name, size in inventory.items() if size <= 0]
    if empty:
        raise RuntimeError(f"restart checkpoint contains empty files: {empty}")
    return inventory


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=strict_json_object,
                parse_constant=reject_nonstandard_json_constant,
            )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"restart JSON is unreadable or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"restart JSON must be an object: {path}")
    return payload


def validate_hf_restart_runtime(
    runtime: Mapping[str, Any],
    *,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Validate the exact rank-local loop/RNG payload before it is trusted."""
    if set(runtime) != _RUNTIME_KEYS:
        raise RuntimeError(
            "HF restart runtime keys differ from the expected schema: "
            f"missing={sorted(_RUNTIME_KEYS - set(runtime))}, "
            f"unexpected={sorted(set(runtime) - _RUNTIME_KEYS)}"
        )
    if not isinstance(runtime["dataloader_state"], Mapping):
        raise RuntimeError("HF restart runtime dataloader_state must be a mapping")
    completed_steps = runtime["completed_steps"]
    if type(completed_steps) is not int or completed_steps < 0:
        raise RuntimeError(
            "HF restart runtime completed_steps must be a non-negative integer"
        )
    if expected_step is not None and completed_steps != expected_step:
        raise RuntimeError(
            f"HF restart runtime step {completed_steps} differs from checkpoint "
            f"step {expected_step}"
        )
    for name in ("smooth_train_loss", "total_training_time"):
        value = runtime[name]
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise RuntimeError(
                f"HF restart runtime {name} must be a non-negative finite float"
            )
    for name in ("latest_val_bpb", "min_val_bpb"):
        value = runtime[name]
        if value is not None and (
            type(value) is not float or not math.isfinite(value) or value < 0
        ):
            raise RuntimeError(
                f"HF restart runtime {name} must be null or a non-negative finite float"
            )
    cpu_state = runtime["torch_cpu_rng_state"]
    if (
        not isinstance(cpu_state, t.Tensor)
        or cpu_state.dtype != t.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() == 0
    ):
        raise RuntimeError("restart runtime has no valid Torch CPU RNG state")
    cuda_state = runtime["torch_cuda_rng_state"]
    if cuda_state is not None and (
        not isinstance(cuda_state, t.Tensor)
        or cuda_state.dtype != t.uint8
        or cuda_state.ndim != 1
        or cuda_state.numel() == 0
    ):
        raise RuntimeError("restart runtime has no valid Torch CUDA RNG state")
    return dict(runtime)


def capture_hf_restart_runtime(
    *,
    dataloader_state: Mapping[str, Any],
    completed_steps: int,
    smooth_train_loss: float,
    latest_val_bpb: float | None,
    min_val_bpb: float | None,
    total_training_time: float,
    device: str | t.device,
) -> dict[str, Any]:
    """Capture the rank-local committed loop and Torch RNG state."""
    if completed_steps < 0:
        raise ValueError(f"completed_steps must be non-negative, got {completed_steps}")
    torch_device = t.device(device)
    cuda_rng_state = (
        t.cuda.get_rng_state(torch_device) if torch_device.type == "cuda" else None
    )
    runtime = {
        "dataloader_state": copy.deepcopy(dict(dataloader_state)),
        "completed_steps": completed_steps,
        "smooth_train_loss": float(smooth_train_loss),
        "latest_val_bpb": None if latest_val_bpb is None else float(latest_val_bpb),
        "min_val_bpb": None if min_val_bpb is None else float(min_val_bpb),
        "total_training_time": float(total_training_time),
        "torch_cpu_rng_state": t.get_rng_state(),
        "torch_cuda_rng_state": cuda_rng_state,
    }
    return validate_hf_restart_runtime(runtime, expected_step=completed_steps)


def restore_hf_restart_rng(runtime: Mapping[str, Any], *, device: str | t.device) -> None:
    """Restore exactly the Torch RNG state captured at the committed boundary."""
    runtime = validate_hf_restart_runtime(runtime)
    cpu_state = cast(t.Tensor, runtime["torch_cpu_rng_state"])
    t.set_rng_state(cpu_state.cpu())
    torch_device = t.device(device)
    cuda_state = runtime["torch_cuda_rng_state"]
    if torch_device.type == "cuda":
        if cuda_state is None:
            raise RuntimeError("restart runtime has no valid Torch CUDA RNG state")
        t.cuda.set_rng_state(cast(t.Tensor, cuda_state).cpu(), torch_device)
    elif cuda_state is not None:
        raise RuntimeError("CPU restart unexpectedly contains a CUDA RNG state")


def save_hf_restart_distributed(
    *,
    output_root: str | Path,
    completed_step: int,
    model_state: Mapping[str, t.Tensor],
    optimizer_state: Mapping[str, Any] | None,
    runtime_state: Mapping[str, Any],
    identity: HFRestartIdentity,
    implementation: HFDistributedImplementation | str,
    world_size: int,
    local_world_size: int,
    rank: int,
    master_process: bool,
    device: str | t.device,
) -> Path:
    """Atomically write rank state and publish a completeness marker last."""
    strategy: HFDistributedImplementation | None = None
    normalized_runtime: dict[str, Any] | None = None
    with rank_synchronized_stage(
        f"HF restart argument validation step {completed_step}", device=device
    ):
        strategy = HFDistributedImplementation(implementation)
        if completed_step < 0:
            raise ValueError(f"completed_step must be non-negative, got {completed_step}")
        if world_size <= 0 or local_world_size <= 0 or world_size % local_world_size:
            raise ValueError(
                f"invalid restart topology world_size={world_size}, "
                f"local_world_size={local_world_size}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world_size={world_size}")
        if master_process != (rank == 0):
            raise ValueError(
                f"master_process={master_process} disagrees with restart rank {rank}"
            )
        if not model_state or not all(
            isinstance(value, t.Tensor) for value in model_state.values()
        ):
            raise ValueError("model_state must be a nonempty tensor mapping")
        if strategy is HFDistributedImplementation.DIST_MUON_ADAMW:
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("DistMuonAdamW restart requires optimizer state on every rank")
        elif rank == 0:
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("DDP fused AdamW restart requires optimizer state on rank zero")
        elif optimizer_state is not None:
            raise ValueError("DDP fused AdamW nonzero rank files must omit optimizer state")
        normalized_runtime = validate_hf_restart_runtime(
            runtime_state,
            expected_step=completed_step,
        )
        expected_identity_sha256 = hashlib.sha256(
            _canonical_json(identity.payload).encode("utf-8")
        ).hexdigest()
        if identity.sha256 != expected_identity_sha256:
            raise ValueError("HF restart identity digest does not match its payload")
    assert strategy is not None and normalized_runtime is not None
    runtime_state = normalized_runtime
    paths = restart_paths(output_root, completed_step, world_size=world_size)
    root = restart_dir(output_root)
    expected = _expected_data_paths(output_root, completed_step, world_size=world_size)

    with rank_synchronized_stage(
        f"HF restart preflight step {completed_step}", device=device
    ):
        if master_process:
            root.mkdir(parents=True, exist_ok=True)
            collisions = [
                path.name
                for path in [*expected, paths["complete"], paths["lock"]]
                if path.exists()
            ]
            if collisions:
                raise FileExistsError(
                    f"refusing to overwrite restart step {completed_step}; existing={collisions}"
                )
            descriptor = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)

    metadata = {
        "format": HF_RESTART_FORMAT,
        "format_version": HF_RESTART_FORMAT_VERSION,
        "completed_optimizer_step": completed_step,
        "semantic_identity_sha256": identity.sha256,
        "semantic_identity": identity.payload,
        "distributed_implementation": strategy.value,
        "world_size": world_size,
        "local_world_size": local_world_size,
    }
    rank_payload = {
        "optimizer": optimizer_state,
        "runtime": runtime_state,
    }
    with rank_synchronized_stage(
        f"HF restart data write step {completed_step}", device=device
    ):
        save_checkpoint(
            str(root),
            completed_step,
            model_state,
            rank_payload,
            metadata,
            rank=rank,
        )

    with rank_synchronized_stage(
        f"HF restart lock cleanup step {completed_step}", device=device
    ):
        if master_process:
            paths["lock"].unlink()

    with rank_synchronized_stage(
        f"HF restart marker publication step {completed_step}", device=device
    ):
        if master_process:
            inventory = _inventory(expected)
            marker = {
                "format": HF_RESTART_FORMAT,
                "format_version": HF_RESTART_FORMAT_VERSION,
                "completed_optimizer_step": completed_step,
                "semantic_identity_sha256": identity.sha256,
                "distributed_implementation": strategy.value,
                "world_size": world_size,
                "local_world_size": local_world_size,
                "files": inventory,
            }
            atomic_json_dump(marker, paths["complete"])
    return paths["complete"]


@dataclass(frozen=True)
class HFRestartDescriptor:
    root: Path
    step: int
    metadata: dict[str, Any]


def validate_hf_restart(
    *,
    output_root: str | Path,
    step: int,
    expected_identity: HFRestartIdentity,
    implementation: HFDistributedImplementation | str,
    world_size: int,
    local_world_size: int,
) -> HFRestartDescriptor:
    """Validate marker, inventory, topology, strategy, and semantic identity."""
    strategy = HFDistributedImplementation(implementation)
    if world_size <= 0 or local_world_size <= 0 or world_size % local_world_size:
        raise ValueError(
            f"invalid restart topology world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )
    paths = restart_paths(output_root, step, world_size=world_size)
    if not paths["complete"].is_file():
        raise FileNotFoundError(
            f"restart step {step} has no complete marker: {paths['complete']}"
        )
    if paths["lock"].exists():
        raise RuntimeError(
            f"restart step {step} has both a complete marker and an in-progress lock"
        )
    marker = _read_json_object(paths["complete"])
    expected_marker = {
        "format": HF_RESTART_FORMAT,
        "format_version": HF_RESTART_FORMAT_VERSION,
        "completed_optimizer_step": step,
        "semantic_identity_sha256": expected_identity.sha256,
        "distributed_implementation": strategy.value,
        "world_size": world_size,
        "local_world_size": local_world_size,
    }
    expected_marker_keys = {*expected_marker, "files"}
    if set(marker) != expected_marker_keys:
        raise RuntimeError(
            "restart marker keys differ from the expected schema: "
            f"missing={sorted(expected_marker_keys - set(marker))}, "
            f"unexpected={sorted(set(marker) - expected_marker_keys)}"
        )
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise RuntimeError(
                f"restart marker {key} mismatch: expected {expected!r}, "
                f"got {marker.get(key)!r}"
            )
    expected_paths = _expected_data_paths(output_root, step, world_size=world_size)
    inventory = _inventory(expected_paths)
    if marker.get("files") != inventory:
        raise RuntimeError(
            f"restart step {step} file inventory differs from its complete marker"
        )

    metadata = _read_json_object(paths["meta"])
    expected_metadata_keys = {*expected_marker, "semantic_identity"}
    if set(metadata) != expected_metadata_keys:
        raise RuntimeError(
            "restart metadata keys differ from the expected schema: "
            f"missing={sorted(expected_metadata_keys - set(metadata))}, "
            f"unexpected={sorted(set(metadata) - expected_metadata_keys)}"
        )
    for key, expected in expected_marker.items():
        if metadata.get(key) != expected:
            if key == "semantic_identity_sha256":
                raise RuntimeError("restart semantic identity mismatch")
            raise RuntimeError(
                f"restart metadata {key} mismatch: expected {expected!r}, "
                f"got {metadata.get(key)!r}"
            )
    if metadata.get("semantic_identity") != expected_identity.payload:
        raise RuntimeError("restart semantic identity payload mismatch")
    return HFRestartDescriptor(root=restart_dir(output_root), step=step, metadata=metadata)


def load_hf_restart_model_in_local_rank_slots(
    descriptor: HFRestartDescriptor,
    *,
    model: PreTrainedModel,
    local_rank: int,
    local_world_size: int,
    device: str | t.device,
) -> None:
    """Restore at most one full model per node at a time."""
    if not 0 <= local_rank < local_world_size:
        raise ValueError(
            f"local_rank={local_rank} is outside local_world_size={local_world_size}"
        )
    model_path = descriptor.root / f"model_{descriptor.step:06d}.pt"
    for slot in range(local_world_size):
        with rank_synchronized_stage(
            f"HF restart model load local-rank slot {slot}", device=device
        ):
            if local_rank == slot:
                state = t.load(model_path, map_location=device, weights_only=True)
                if not isinstance(state, Mapping):
                    raise RuntimeError("HF restart model state must be a mapping")
                model.load_state_dict(state, strict=True)
                del state


def load_hf_restart_rank_payload_in_local_rank_slots(
    descriptor: HFRestartDescriptor,
    *,
    rank: int,
    local_rank: int,
    local_world_size: int,
    device: str | t.device,
) -> dict[str, Any]:
    """Load rank-local runtime plus the optimizer state this rank consumes."""
    payload: dict[str, Any] | None = None
    path = descriptor.root / f"optim_{descriptor.step:06d}_rank{rank}.pt"
    strategy = HFDistributedImplementation(
        descriptor.metadata["distributed_implementation"]
    )
    for slot in range(local_world_size):
        with rank_synchronized_stage(
            f"HF restart rank-state load local-rank slot {slot}", device=device
        ):
            if local_rank == slot:
                loaded = t.load(path, map_location=device, weights_only=True)
                if not isinstance(loaded, dict):
                    raise RuntimeError("HF restart rank payload must be a mapping")
                if (
                    strategy is HFDistributedImplementation.DDP_FUSED_ADAMW
                    and rank != 0
                ):
                    if loaded.get("optimizer") is not None:
                        raise RuntimeError(
                            f"HF restart nonzero DDP rank {rank} unexpectedly "
                            "contains optimizer state"
                        )
                    source = descriptor.root / f"optim_{descriptor.step:06d}_rank0.pt"
                    rank_zero = t.load(source, map_location=device, weights_only=True)
                    if not isinstance(rank_zero, Mapping) or not isinstance(
                        rank_zero.get("optimizer"), Mapping
                    ):
                        raise RuntimeError(
                            "HF restart rank zero is missing its fused AdamW state"
                        )
                    loaded["optimizer"] = rank_zero["optimizer"]
                payload = loaded
    with rank_synchronized_stage("HF restart rank payload validation", device=device):
        if payload is None:
            raise RuntimeError("HF restart rank payload load completed without local state")
        if set(payload) != {"optimizer", "runtime"} or not isinstance(
            payload["runtime"], Mapping
        ):
            raise RuntimeError("HF restart rank payload has an invalid schema")
        payload["runtime"] = validate_hf_restart_runtime(
            payload["runtime"],
            expected_step=descriptor.step,
        )
        optimizer_state = payload["optimizer"]
        if strategy is HFDistributedImplementation.DIST_MUON_ADAMW:
            if not isinstance(optimizer_state, Mapping):
                raise RuntimeError(
                    f"HF restart rank {rank} is missing its DistMuonAdamW optimizer shard"
                )
        elif not isinstance(optimizer_state, Mapping):
            raise RuntimeError("HF restart rank zero is missing its fused AdamW state")
    assert payload is not None
    return payload
