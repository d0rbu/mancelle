"""Crash-safe exact-pass checkpoints for feature-statistics reduction.

The large safetensors generation is immutable.  A small atomically-replaced
manifest is the sole commit point, so interruption can never expose a partial
generation as resumable state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
from loguru import logger
from safetensors import SafetensorError, safe_open

from nanochat.atomic_io import atomic_json_dump, atomic_safetensors_save
from steering._common.strict_json import (
    canonical_json,
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.exact_statistics import effective_available_memory_bytes
from steering.feature_io import MAX_SAFETENSORS_HEADER_SIZE, RegularFileIdentity

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_ALGORITHM_VERSION = "exact_feature_statistics_pass1_v1"
CHECKPOINT_LOCK_FILENAME = "checkpoint.lock"
CHECKPOINT_MANIFEST_FILENAME = "manifest.json"

_STATE_RE = re.compile(r"^state-(?P<processed>[0-9]{8})-(?P<digest>[0-9a-f]{16})\.safetensors$")
_KNOWN_TEMP_RE = re.compile(
    r"^(?:manifest\.json|state-[0-9]{8}-[0-9a-f]{16}\.safetensors)"
    r"\.tmp\.[0-9]+\.[0-9a-f]{8}$"
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "algorithm_version",
        "contract",
        "contract_sha256",
        "processed_shards",
        "state_filename",
        "state_tensor_sha256",
        "state_tensor_bytes",
        "state_scalar_sha256",
        "scalars",
    }
)
_STATE_METADATA_KEYS = frozenset(
    {
        "checkpoint_schema_version",
        "checkpoint_algorithm_version",
        "contract_sha256",
        "processed_shards",
        "state_tensor_sha256",
        "state_scalar_sha256",
    }
)


class StatisticsCheckpointError(RuntimeError):
    """A statistics checkpoint cannot be safely created or resumed."""


@dataclass(frozen=True)
class ShardInventoryEntry:
    path: Path
    identity: RegularFileIdentity

    def to_json(self) -> dict[str, int | str]:
        return self.identity.to_json(name=self.path.name)


@dataclass
class LoadedStatisticsCheckpoint:
    processed_shards: int
    scalars: dict[str, Any]
    tensors: dict[str, t.Tensor]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def capture_shard_inventory(paths: Sequence[Path]) -> tuple[ShardInventoryEntry, ...]:
    """Capture one ordered, direct-regular-file shard inventory."""
    entries = tuple(
        ShardInventoryEntry(path=Path(path), identity=RegularFileIdentity.capture(path))
        for path in paths
    )
    names = [entry.path.name for entry in entries]
    if len(set(names)) != len(names):
        raise StatisticsCheckpointError(f"statistics shard inventory has duplicate names: {names}")
    return entries


def assert_shard_inventory_unchanged(
    expected: Sequence[ShardInventoryEntry],
    *,
    observed_paths: Sequence[Path] | None = None,
) -> None:
    """Fail if shard membership or any descriptor identity changed.

    ``observed_paths`` is optional so callers that already sealed membership
    can perform an identity-only fence.  Publication boundaries pass a fresh
    directory listing to detect additions, removals, and reorderings too.
    """
    if observed_paths is not None:
        current = capture_shard_inventory(observed_paths)
        if current != tuple(expected):
            raise StatisticsCheckpointError(
                "statistics shard inventory changed before output publication: "
                f"expected_count={len(expected)}, observed_count={len(current)}"
            )
        return

    for index, entry in enumerate(expected):
        try:
            observed = RegularFileIdentity.capture(entry.path)
        except (OSError, ValueError) as exc:
            raise StatisticsCheckpointError(
                f"statistics shard inventory changed at index {index} ({entry.path.name}): {exc}"
            ) from exc
        if observed != entry.identity:
            raise StatisticsCheckpointError(
                "statistics shard inventory changed at index "
                f"{index} ({entry.path.name}): expected={entry.identity!r}, "
                f"observed={observed!r}"
            )


def build_checkpoint_contract(
    *,
    inventory: Sequence[ShardInventoryEntry],
    n_sae_features: int,
    exclude_last_shard: bool,
    skip_per_feature_detail: bool,
    l0_tier: str | None,
    ignore_bos_features: bool,
    sft_mixture_fingerprint_json: str | None,
    numeric_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete semantic identity of one exact-pass run."""
    inventory_json = [entry.to_json() for entry in inventory]
    return {
        "inventory": inventory_json,
        "inventory_sha256": _sha256_json(inventory_json),
        "n_sae_features": n_sae_features,
        "exclude_last_shard": exclude_last_shard,
        "skip_per_feature_detail": skip_per_feature_detail,
        "l0_tier": l0_tier,
        "ignore_bos_features": ignore_bos_features,
        "sft_mixture_fingerprint_json": sft_mixture_fingerprint_json,
        "numeric_runtime": dict(numeric_runtime),
    }


def numeric_runtime_fingerprint(
    *,
    device: t.device,
    gpu_chunk_size: int,
) -> dict[str, Any]:
    """Return runtime facts capable of changing moment-accumulator numerics."""
    payload: dict[str, Any] = {
        "device": str(device),
        "gpu_chunk_size": gpu_chunk_size,
        "torch": str(t.__version__),
        "torch_build_sha256": hashlib.sha256(
            t.__config__.show().encode("utf-8")
        ).hexdigest(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_capability": t.backends.cpu.get_cpu_capability(),
        "cuda_runtime": str(t.version.cuda) if t.version.cuda is not None else None,
        "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": t.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": t.get_float32_matmul_precision(),
        "torch_num_threads": t.get_num_threads(),
    }
    if device == t.device("cuda"):
        index = t.cuda.current_device()
        properties = t.cuda.get_device_properties(index)
        payload.update(
            {
                "gpu_name": properties.name,
                "gpu_uuid": str(properties.uuid),
                "gpu_compute_capability": [properties.major, properties.minor],
                "gpu_total_memory": properties.total_memory,
                "gpu_multiprocessor_count": properties.multi_processor_count,
            }
        )
    return payload


def _tensor_digest(tensors: Mapping[str, t.Tensor]) -> tuple[str, int]:
    """Hash logical tensor state without materialising another tensor payload."""
    digest = hashlib.sha256(b"feature-statistics-checkpoint-tensors-v1\0")
    total_bytes = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise StatisticsCheckpointError(
                f"checkpoint tensor {name!r} must be contiguous CPU state, "
                f"got device={tensor.device} contiguous={tensor.is_contiguous()}"
            )
        if tensor.dtype not in {t.int64, t.float64, t.float32}:
            raise StatisticsCheckpointError(
                f"checkpoint tensor {name!r} has unsupported dtype {tensor.dtype}"
            )
        nbytes = tensor.numel() * tensor.element_size()
        header = canonical_json(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "nbytes": nbytes,
            }
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        view = memoryview(tensor.detach().numpy()).cast("B")
        for start in range(0, len(view), 8 * 1024 * 1024):
            digest.update(view[start : start + 8 * 1024 * 1024])
        total_bytes += nbytes
    return digest.hexdigest(), total_bytes


def _first_difference(
    expected: Any,
    observed: Any,
    path: str = "contract",
) -> tuple[str, Any, Any] | None:
    if type(expected) is not type(observed):
        return path, expected, observed
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            return f"{path}.keys", sorted(expected), sorted(observed)
        for key in sorted(expected):
            difference = _first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}.length", len(expected), len(observed)
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if expected == observed else (path, expected, observed)


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=strict_json_object,
                parse_constant=reject_nonstandard_json_constant,
            )
    except (OSError, UnicodeError, ValueError) as exc:
        raise StatisticsCheckpointError(
            f"statistics checkpoint manifest is unreadable or invalid: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise StatisticsCheckpointError("statistics checkpoint manifest must be a JSON object")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class StatisticsCheckpointStore:
    """Exclusive crash-safe store for one exact statistics pass."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / CHECKPOINT_MANIFEST_FILENAME
        self._lock_fd: int | None = None
        self._current_state_path: Path | None = None

    def __enter__(self) -> StatisticsCheckpointStore:
        if self.directory.exists() and (
            self.directory.is_symlink() or not self.directory.is_dir()
        ):
            raise StatisticsCheckpointError(
                f"checkpoint_dir must be a real directory: {self.directory}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / CHECKPOINT_LOCK_FILENAME
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._lock_fd = os.open(lock_path, flags, 0o600)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise StatisticsCheckpointError(
                f"checkpoint directory is already in use or cannot be locked: {self.directory}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def _require_locked(self) -> None:
        if self._lock_fd is None:
            raise StatisticsCheckpointError("checkpoint store is not locked")

    def _enumerate_and_reject_unknown(self) -> list[Path]:
        known_states: list[Path] = []
        for entry in self.directory.iterdir():
            if entry.name in {CHECKPOINT_LOCK_FILENAME, CHECKPOINT_MANIFEST_FILENAME}:
                if entry.is_symlink() or not entry.is_file():
                    raise StatisticsCheckpointError(
                        f"checkpoint control entry is not a direct regular file: {entry}"
                    )
                continue
            if _STATE_RE.fullmatch(entry.name) is not None:
                if entry.is_symlink() or not entry.is_file():
                    raise StatisticsCheckpointError(
                        f"checkpoint state entry is not a direct regular file: {entry}"
                    )
                known_states.append(entry)
                continue
            if _KNOWN_TEMP_RE.fullmatch(entry.name) is not None:
                if entry.is_symlink() or not entry.is_file():
                    raise StatisticsCheckpointError(
                        f"checkpoint temporary entry is not a direct regular file: {entry}"
                    )
                continue
            raise StatisticsCheckpointError(f"unknown entry in checkpoint directory: {entry}")
        return known_states

    def restore(self, contract: Mapping[str, Any]) -> LoadedStatisticsCheckpoint | None:
        self._require_locked()
        known_states = self._enumerate_and_reject_unknown()
        if not self.manifest_path.exists():
            removed = False
            for path in known_states:
                path.unlink()
                removed = True
            for path in self.directory.iterdir():
                if _KNOWN_TEMP_RE.fullmatch(path.name) is not None:
                    path.unlink()
                    removed = True
            if removed:
                _fsync_directory(self.directory)
            return None

        manifest = _read_strict_json(self.manifest_path)
        if set(manifest) != _MANIFEST_KEYS:
            raise StatisticsCheckpointError(
                "checkpoint manifest keys differ: "
                f"expected={sorted(_MANIFEST_KEYS)}, observed={sorted(manifest)}"
            )
        if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise StatisticsCheckpointError(
                f"checkpoint schema version differs: {manifest['schema_version']!r}"
            )
        if manifest["algorithm_version"] != CHECKPOINT_ALGORITHM_VERSION:
            raise StatisticsCheckpointError(
                f"checkpoint algorithm version differs: {manifest['algorithm_version']!r}"
            )
        expected_contract = dict(contract)
        difference = _first_difference(expected_contract, manifest["contract"])
        if difference is not None:
            path, expected, observed = difference
            raise StatisticsCheckpointError(
                f"statistics checkpoint does not match this run at {path}: "
                f"expected={expected!r}, observed={observed!r}"
            )
        contract_sha256 = _sha256_json(expected_contract)
        if manifest["contract_sha256"] != contract_sha256:
            raise StatisticsCheckpointError("checkpoint contract digest is invalid")
        processed = manifest["processed_shards"]
        if type(processed) is not int or processed <= 0 or processed > len(
            expected_contract["inventory"]
        ):
            raise StatisticsCheckpointError(
                f"checkpoint processed_shards is out of range: {processed!r}"
            )
        filename = manifest["state_filename"]
        state_match = _STATE_RE.fullmatch(filename) if isinstance(filename, str) else None
        if state_match is None:
            raise StatisticsCheckpointError(f"checkpoint state filename is invalid: {filename!r}")
        if int(state_match.group("processed")) != processed:
            raise StatisticsCheckpointError(
                "checkpoint state filename progress disagrees with manifest: "
                f"filename={filename!r}, processed_shards={processed}"
            )
        state_digest = manifest["state_tensor_sha256"]
        if (
            not isinstance(state_digest, str)
            or len(state_digest) != 64
            or any(character not in "0123456789abcdef" for character in state_digest)
        ):
            raise StatisticsCheckpointError(
                f"checkpoint tensor digest is invalid: {state_digest!r}"
            )
        state_tensor_bytes = manifest["state_tensor_bytes"]
        if type(state_tensor_bytes) is not int or state_tensor_bytes < 0:
            raise StatisticsCheckpointError(
                f"checkpoint tensor byte count is invalid: {state_tensor_bytes!r}"
            )
        state_scalar_digest = manifest["state_scalar_sha256"]
        if (
            not isinstance(state_scalar_digest, str)
            or len(state_scalar_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in state_scalar_digest
            )
        ):
            raise StatisticsCheckpointError(
                f"checkpoint scalar digest is invalid: {state_scalar_digest!r}"
            )
        scalars = manifest["scalars"]
        if not isinstance(scalars, dict):
            raise StatisticsCheckpointError("checkpoint scalars must be a JSON object")
        observed_scalar_digest = _sha256_json(scalars)
        if observed_scalar_digest != state_scalar_digest:
            raise StatisticsCheckpointError(
                "checkpoint scalar digest mismatch: "
                f"expected={state_scalar_digest}, observed={observed_scalar_digest}"
            )
        state_path = self.directory / filename
        if state_path not in known_states:
            raise StatisticsCheckpointError(f"committed checkpoint state is missing: {state_path}")
        available = effective_available_memory_bytes()
        state_file_bytes = state_path.stat().st_size
        if state_file_bytes > available:
            raise StatisticsCheckpointError(
                "checkpoint state does not fit current memory headroom: "
                f"file={state_file_bytes:,} bytes, available={available:,} bytes"
            )
        expected_metadata = {
            "checkpoint_schema_version": str(CHECKPOINT_SCHEMA_VERSION),
            "checkpoint_algorithm_version": CHECKPOINT_ALGORITHM_VERSION,
            "contract_sha256": contract_sha256,
            "processed_shards": str(processed),
            "state_tensor_sha256": state_digest,
            "state_scalar_sha256": state_scalar_digest,
        }
        try:
            with safe_open(
                state_path,
                framework="pt",
                device="cpu",
                backend="pread",
            ) as reader:
                metadata = dict(reader.metadata() or {})
                if set(metadata) != _STATE_METADATA_KEYS or metadata != expected_metadata:
                    raise StatisticsCheckpointError(
                        "checkpoint state metadata differs from its manifest: "
                        f"expected={expected_metadata!r}, observed={metadata!r}"
                    )
                tensors = reader.get_tensors()
        except (OSError, SafetensorError, RuntimeError) as exc:
            raise StatisticsCheckpointError(
                f"committed checkpoint state is unreadable: {state_path}: {exc}"
            ) from exc
        observed_digest, observed_bytes = _tensor_digest(tensors)
        if observed_digest != state_digest:
            raise StatisticsCheckpointError(
                "checkpoint tensor digest mismatch: "
                f"expected={manifest['state_tensor_sha256']}, observed={observed_digest}"
            )
        if observed_bytes != state_tensor_bytes:
            raise StatisticsCheckpointError(
                "checkpoint tensor byte count mismatch: "
                f"expected={state_tensor_bytes}, observed={observed_bytes}"
            )
        self._current_state_path = state_path
        for path in known_states:
            if path != state_path:
                path.unlink()
        for path in self.directory.iterdir():
            if _KNOWN_TEMP_RE.fullmatch(path.name) is not None:
                path.unlink()
        _fsync_directory(self.directory)
        logger.info(
            f"Restored exact statistics checkpoint after {processed:,}/"
            f"{len(expected_contract['inventory']):,} shards from {state_path}"
        )
        return LoadedStatisticsCheckpoint(
            processed_shards=processed,
            scalars=dict(scalars),
            tensors=tensors,
        )

    def commit(
        self,
        *,
        contract: Mapping[str, Any],
        processed_shards: int,
        scalars: Mapping[str, Any],
        tensors: Mapping[str, t.Tensor],
    ) -> Path:
        self._require_locked()
        if type(processed_shards) is not int or processed_shards <= 0:
            raise StatisticsCheckpointError(
                f"processed_shards must be a positive int, got {processed_shards!r}"
            )
        contract_dict = dict(contract)
        contract_sha256 = _sha256_json(contract_dict)
        scalars_dict = dict(scalars)
        scalar_digest = _sha256_json(scalars_dict)
        state_digest, state_bytes = _tensor_digest(tensors)
        filename = f"state-{processed_shards:08d}-{state_digest[:16]}.safetensors"
        state_path = self.directory / filename
        if state_path.exists() or state_path.is_symlink():
            raise StatisticsCheckpointError(
                f"refusing to replace unexpected checkpoint generation: {state_path}"
            )
        required_bytes = state_bytes + 8 + MAX_SAFETENSORS_HEADER_SIZE
        available_disk = shutil.disk_usage(self.directory).free
        if required_bytes > available_disk:
            raise StatisticsCheckpointError(
                "checkpoint candidate does not fit the checkpoint filesystem: "
                f"required_at_least={required_bytes:,} bytes, free={available_disk:,} bytes"
            )
        metadata = {
            "checkpoint_schema_version": str(CHECKPOINT_SCHEMA_VERSION),
            "checkpoint_algorithm_version": CHECKPOINT_ALGORITHM_VERSION,
            "contract_sha256": contract_sha256,
            "processed_shards": str(processed_shards),
            "state_tensor_sha256": state_digest,
            "state_scalar_sha256": scalar_digest,
        }
        atomic_safetensors_save(dict(tensors), state_path, metadata=metadata)
        _fsync_directory(self.directory)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm_version": CHECKPOINT_ALGORITHM_VERSION,
            "contract": contract_dict,
            "contract_sha256": contract_sha256,
            "processed_shards": processed_shards,
            "state_filename": filename,
            "state_tensor_sha256": state_digest,
            "state_tensor_bytes": state_bytes,
            "state_scalar_sha256": scalar_digest,
            "scalars": scalars_dict,
        }
        atomic_json_dump(manifest, self.manifest_path)
        _fsync_directory(self.directory)
        previous = self._current_state_path
        self._current_state_path = state_path
        if previous is not None and previous != state_path:
            previous.unlink(missing_ok=True)
            _fsync_directory(self.directory)
        logger.info(
            f"Committed exact statistics checkpoint after {processed_shards:,} shards "
            f"({state_bytes / 2**30:.3f} GiB tensor payload)"
        )
        return state_path

    def cleanup_after_output_commit(self) -> None:
        """Remove large state only after the final output directory is durable."""
        self._require_locked()
        # Retire the manifest commit point first. If state cleanup is
        # interrupted, the next run sees only an orphan generation and safely
        # discards it; it never sees a committed manifest whose state vanished.
        self.manifest_path.unlink(missing_ok=True)
        _fsync_directory(self.directory)
        if self._current_state_path is not None:
            self._current_state_path.unlink(missing_ok=True)
            self._current_state_path = None
        for path in self.directory.iterdir():
            if _STATE_RE.fullmatch(path.name) is not None or _KNOWN_TEMP_RE.fullmatch(
                path.name
            ) is not None:
                path.unlink()
        _fsync_directory(self.directory)


def fsync_parent_directory(path: str | Path) -> None:
    """Durably publish an already-fsynced atomic output rename."""
    _fsync_directory(Path(path).parent)


__all__ = [
    "LoadedStatisticsCheckpoint",
    "ShardInventoryEntry",
    "StatisticsCheckpointError",
    "StatisticsCheckpointStore",
    "assert_shard_inventory_unchanged",
    "build_checkpoint_contract",
    "capture_shard_inventory",
    "fsync_parent_directory",
    "numeric_runtime_fingerprint",
]
