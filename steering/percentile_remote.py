"""Bounded HF streaming producer for percentile overlays.

This module is intentionally specific to percentile overlays.  It reuses the
repository's verified bucket transport, but owns its contract, receipts,
staging namespace, resume state machine, and manifest-last publication.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from typing import Any, cast

import hf_xet
import torch as t
import zstandard
from loguru import logger
from omegaconf import DictConfig

from nanochat.atomic_io import atomic_json_dump, atomic_write_bytes
from steering._common.strict_json import strict_json_object
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.exact_statistics import effective_available_memory_bytes
from steering.feature_io import (
    COMPRESSED_EXT,
    RegularFileIdentity,
    TensorKey,
    decode_percentile_feature_safetensors_bytes,
    parse_percentile_feature_safetensors_bytes,
    read_feature_compressed_bytes_stable,
)
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    MANIFEST_FILENAME,
    BaseShardContentHasher,
    OverlayKind,
    load_overlay_shard_bytes,
    overlay_tensors_content_digest,
    save_overlay_shard,
    write_manifest,
)
from steering.percentile_precompute import (
    compute_percentile_overlay_values,
    compute_percentile_overlay_values_from_encoded,
)
from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
    resolve_remote_feature_prefix,
)
from steering.remote_training import (
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    RemoteTrainingError,
    download_verified_object,
    parse_verified_upload_claim_bytes,
    upload_verified_object,
    verified_local_object_info,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.stats_io import (
    find_statistics_path,
    load_statistics,
    load_statistics_and_checksum_from_bytes,
    save_percentile_checksum,
)

REMOTE_CONTRACT_FILENAME = "_remote_contract.json"
REMOTE_RECEIPTS_DIRNAME = "_receipts"
OVERLAY_STATISTICS_FILENAME = "feature_statistics.safetensors"
REMOTE_CONTRACT_SCHEMA_VERSION = 1
REMOTE_RECEIPT_SCHEMA_VERSION = 1
PERCENTILE_ALGORITHM_VERSION = 1
REMOTE_PROFILE_SCHEMA_VERSION = 1
STAGING_OWNER_SCHEMA_VERSION = 1
CAPACITY_RESERVE_BYTES = 1 << 30
DECOMPRESSION_MEMORY_RESERVE_BYTES = 1 << 30

_LOCK_FILENAME = ".percentile-remote.lock"
_OWNER_FILENAME = ".percentile-remote-owner.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_RE = re.compile(rf"^shard_(?P<index>\d{{5}}){re.escape(COMPRESSED_EXT)}$")
_RECEIPT_RE = re.compile(r"^shard_(?P<index>\d{5})\.json$")
_SOURCE_RE = re.compile(rf"^shard_(?P<index>\d{{5}})_features{re.escape(COMPRESSED_EXT)}$")
_OUTPUT_STAGE_RE = re.compile(rf"^output_shard_(?P<index>\d{{5}}){re.escape(COMPRESSED_EXT)}$")
_VERIFY_STAGE_RE = re.compile(rf"^verify_shard_(?P<index>\d{{5}}){re.escape(COMPRESSED_EXT)}$")
_RECEIPT_STAGE_RE = re.compile(r"^receipt_shard_(?P<index>\d{5})\.json$")
_REMOTE_TMP_RE = re.compile(r"^\.(?P<base>.+)\.remote-tmp\.(?P<pid>\d+)\.[0-9a-f]{8}$")
_ATOMIC_TMP_RE = re.compile(r"^(?P<base>.+)\.tmp\.(?P<pid>\d+)\.[0-9a-f]{8}$")


class RemotePercentileError(RuntimeError):
    """Remote percentile configuration, identity, or state is unsafe."""


def _strict_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RemotePercentileError(f"{name} must be a positive integer, got {value!r}")
    return value


def _strict_positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemotePercentileError(f"{name} must be a positive finite number, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise RemotePercentileError(f"{name} must be a positive finite number, got {value!r}")
    return resolved


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise RemotePercentileError(f"{name} must be a bool, got {value!r}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _join_remote(prefix: NormalizedRemotePrefix | str, *parts: str) -> str:
    value = PurePosixPath(str(prefix), *parts).as_posix()
    return str(NormalizedRemotePrefix.parse(value))


def remote_percentiles_enabled(cfg: DictConfig) -> bool:
    """Read only the opt-in switch; disabled mode ignores all other fields."""
    remote = cfg.get("remote", None)
    if remote is None:
        return False
    enabled = remote.get("enabled", False)
    return _strict_bool(enabled, name="remote.enabled")


@dataclass(frozen=True)
class RemotePercentileConfig:
    bucket_id: HfBucketId
    source_prefix: NormalizedRemotePrefix
    output_prefix: NormalizedRemotePrefix
    staging_dir: Path
    expected_shard_count: int
    max_output_shard_bytes: int
    max_decompressed_shard_bytes: int | None
    retry_timeout_seconds: float
    retry_interval_seconds: float
    expected_segment_size: int
    zstd_level: int
    device: t.device
    resume: bool
    dry_run: bool
    profile: bool
    profile_path: Path | None


@dataclass(frozen=True)
class RemoteReceipt:
    contract_sha256: str
    shard_index: int
    source: RemoteObjectInfo
    output: RemoteObjectInfo
    semantic_content_sha256: str
    n_values: int

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": REMOTE_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "shard_index": self.shard_index,
            "source": {
                "path": self.source.path,
                "byte_length": self.source.byte_length,
                "xet_hash": self.source.xet_hash,
            },
            "output": {
                "path": self.output.path,
                "byte_length": self.output.byte_length,
                "xet_hash": self.output.xet_hash,
                "semantic_content_sha256": self.semantic_content_sha256,
            },
            "n_values": self.n_values,
        }


@dataclass
class RemoteOutputState:
    listing: tuple[RemoteObjectInfo, ...]
    contract_info: RemoteObjectInfo | None = None
    contract: dict[str, object] | None = None
    payloads: dict[int, RemoteObjectInfo] = field(default_factory=dict)
    receipt_infos: dict[int, RemoteObjectInfo] = field(default_factory=dict)
    receipts: dict[int, RemoteReceipt] = field(default_factory=dict)
    statistics_info: RemoteObjectInfo | None = None
    manifest_info: RemoteObjectInfo | None = None
    manifest: dict[str, object] | None = None
    claim_infos: dict[str, RemoteObjectInfo] = field(default_factory=dict)
    claimed_objects: dict[str, RemoteObjectInfo] = field(default_factory=dict)

    @property
    def has_canonical_state(self) -> bool:
        return any(
            (
                self.contract_info is not None,
                bool(self.payloads),
                bool(self.receipt_infos),
                self.statistics_info is not None,
                self.manifest_info is not None,
                bool(self.claim_infos),
            )
        )

    def complete(self, expected_count: int) -> bool:
        return (
            self.structurally_complete(expected_count)
            and self.contract is not None
            and self.manifest is not None
            and len(self.receipts) == expected_count
        )

    def structurally_complete(self, expected_count: int) -> bool:
        expected = set(range(expected_count))
        return (
            self.contract_info is not None
            and self.statistics_info is not None
            and self.manifest_info is not None
            and set(self.payloads) == expected
            and set(self.receipt_infos) == expected
        )


@dataclass
class RemotePercentileProfile:
    config: dict[str, object]
    inventory: dict[str, object]
    staging: dict[str, object]
    per_shard: list[dict[str, object]] = field(default_factory=list)
    totals: dict[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
            "mode": "remote",
            "config": self.config,
            "inventory": self.inventory,
            "staging": self.staging,
            "per_shard": self.per_shard,
            "totals": self.totals,
        }


@dataclass(frozen=True)
class _PublishedFiles:
    receipt: RemoteReceipt
    receipt_identity: RegularFileIdentity
    output_identity: RegularFileIdentity | None = None


@dataclass
class _PreparedPercentileSource:
    """One authenticated source after optional bounded decompression."""

    disposition: str
    safetensors_bytes: bytes | None

    def take_safetensors_bytes(self) -> bytes:
        """Transfer ownership of the plain body without retaining a second reference."""
        payload = self.safetensors_bytes
        if payload is None:
            raise RemotePercentileError("prepared percentile source has no plain body")
        self.safetensors_bytes = None
        return payload


@dataclass
class _PendingUpload:
    future: Future[_PublishedFiles]
    receipt_path: Path
    profile_record: dict[str, object]
    output_path: Path | None = None

    def finish(self, staging: PercentileStaging) -> None:
        published = self.future.result()
        self.profile_record["upload_verified"] = True
        if self.output_path is not None:
            if published.output_identity is None:
                raise RemotePercentileError(
                    "payload publication did not bind its staged output identity"
                )
            staging._safe_unlink(
                self.output_path,
                expected=published.output_identity,
            )
        staging._safe_unlink(
            self.receipt_path,
            expected=published.receipt_identity,
        )


def _validate_disjoint(left: Path, right: Path, *, surfaces: str) -> None:
    resolved_left = left.resolve()
    resolved_right = right.resolve()
    try:
        resolved_left.relative_to(resolved_right)
    except ValueError:
        pass
    else:
        raise RemotePercentileError(
            f"{surfaces} overlap: {resolved_left} is within {resolved_right}"
        )
    try:
        resolved_right.relative_to(resolved_left)
    except ValueError:
        return
    raise RemotePercentileError(f"{surfaces} overlap: {resolved_right} is within {resolved_left}")


def _validate_rank_environment() -> None:
    values: list[int] = []
    for name, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        raw = os.environ.get(name, default)
        try:
            value = int(raw)
        except ValueError as exc:
            raise RemotePercentileError(f"{name} must be an integer, got {raw!r}") from exc
        values.append(value)
    world_size, rank, local_rank = values
    if (world_size, rank, local_rank) != (1, 0, 0):
        raise RemotePercentileError(
            "remote percentile v1 requires WORLD_SIZE=1, RANK=0, LOCAL_RANK=0; "
            f"got WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}"
        )


def _resolve_config(
    cfg: DictConfig,
    *,
    feature_dir: Path,
    checksum: str,
) -> RemotePercentileConfig:
    remote = cfg.remote
    _validate_rank_environment()
    if _strict_bool(cfg.get("force", False), name="force"):
        raise RemotePercentileError("remote percentile mode rejects force=true")
    resume = _strict_bool(cfg.get("resume", False), name="resume")
    dry_run = _strict_bool(cfg.get("dry_run", False), name="dry_run")
    profile = _strict_bool(cfg.get("profile", False), name="profile")

    raw_device = cfg.get("device", None)
    if raw_device not in {"cpu", "cuda"}:
        raise RemotePercentileError(
            f"remote percentile device must be explicitly cpu or cuda; got {raw_device!r}"
        )
    device = t.device("cuda:0" if raw_device == "cuda" else "cpu")
    if device.type == "cuda" and not t.cuda.is_available():
        raise RemotePercentileError("remote percentile device=cuda but CUDA is unavailable")

    expected_count = _strict_positive_int(
        remote.get("expected_shard_count", None),
        name="remote.expected_shard_count",
    )
    max_output_bytes = _strict_positive_int(
        remote.get("max_output_shard_bytes", None),
        name="remote.max_output_shard_bytes",
    )
    raw_max_decompressed_bytes = remote.get("max_decompressed_shard_bytes", None)
    max_decompressed_bytes = (
        _strict_positive_int(
            raw_max_decompressed_bytes,
            name="remote.max_decompressed_shard_bytes",
        )
        if device.type == "cuda" or raw_max_decompressed_bytes is not None
        else None
    )
    segment_size = _strict_positive_int(
        cfg.get("segment_size", None),
        name="segment_size",
    )
    zstd_level = _strict_positive_int(cfg.get("zstd_level", None), name="zstd_level")
    if zstd_level > 22:
        raise RemotePercentileError(f"zstd_level must be in [1, 22], got {zstd_level}")
    timeout = _strict_positive_finite(
        remote.get("retry_timeout_seconds", None),
        name="remote.retry_timeout_seconds",
    )
    interval = _strict_positive_finite(
        remote.get("retry_interval_seconds", None),
        name="remote.retry_interval_seconds",
    )
    staging_raw = remote.get("staging_dir", None)
    if staging_raw is None:
        raise RemotePercentileError("remote.staging_dir is required when remote.enabled=true")
    staging_dir = Path(str(staging_raw)).absolute()
    _validate_disjoint(staging_dir, feature_dir, surfaces="remote staging and feature_dir")

    profile_raw = cfg.get("profile_summary_output", None)
    profile_path = None
    if profile:
        profile_path = Path(
            "out/traces/pct_remote_profile.json" if profile_raw is None else str(profile_raw)
        ).absolute()
        _validate_disjoint(
            staging_dir,
            profile_path,
            surfaces="remote staging and profile output",
        )

    bucket_id = HfBucketId.parse(str(remote.get("bucket_id", "")))
    remote_root = NormalizedRemotePrefix.parse(str(remote.get("remote_features_root", "")))
    source_prefix = resolve_remote_feature_prefix(
        feature_dir,
        local_features_root=Path(str(remote.get("local_features_root", ""))),
        remote_features_root=remote_root,
        surface="feature_dir",
    )
    output_prefix = NormalizedRemotePrefix.parse(
        _join_remote(source_prefix, "pct", checksum[:CHECKSUM_PREFIX_LEN])
    )
    return RemotePercentileConfig(
        bucket_id=bucket_id,
        source_prefix=source_prefix,
        output_prefix=output_prefix,
        staging_dir=staging_dir,
        expected_shard_count=expected_count,
        max_output_shard_bytes=max_output_bytes,
        max_decompressed_shard_bytes=max_decompressed_bytes,
        retry_timeout_seconds=timeout,
        retry_interval_seconds=interval,
        expected_segment_size=segment_size,
        zstd_level=zstd_level,
        device=device,
        resume=resume,
        dry_run=dry_run,
        profile=profile,
        profile_path=profile_path,
    )


def _validate_source_family(
    inventory: Sequence[RemoteFeatureShard],
    *,
    expected_count: int,
) -> tuple[RemoteFeatureShard, ...]:
    sealed = tuple(inventory)
    observed = [entry.index for entry in sealed]
    expected = list(range(expected_count))
    if observed != expected:
        observed_set = set(observed)
        expected_set = set(expected)
        raise RemotePercentileError(
            f"remote source is not the exact contiguous family 0..{expected_count - 1}: "
            f"missing={sorted(expected_set - observed_set)[:20]}, "
            f"extra={sorted(observed_set - expected_set)[:20]}"
        )
    return sealed


def _retry_backend_call[T](
    operation: str,
    call: Callable[[], T],
    *,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> T:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while True:
        attempts += 1
        try:
            return call()
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemotePercentileError(
                    f"timed out after {timeout_seconds:.1f}s during {operation}; "
                    f"attempts={attempts}; last_error={exc!r}"
                ) from exc
            logger.warning(
                f"percentile_remote.retry operation={operation!r} attempt={attempts} "
                f"remaining_seconds={remaining:.1f} error={exc!r}"
            )
            time.sleep(min(retry_interval_seconds, remaining))


def _list_with_retry(
    backend: BucketBackend,
    prefix: str,
    *,
    config: RemotePercentileConfig,
    surface: str,
) -> Sequence[RemoteObjectInfo]:
    return _retry_backend_call(
        f"{surface} listing for {prefix}",
        lambda: backend.list(prefix),
        timeout_seconds=config.retry_timeout_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
    )


def _stat_with_retry(
    backend: BucketBackend,
    path: str,
    *,
    config: RemotePercentileConfig,
    surface: str,
) -> RemoteObjectInfo | None:
    return _retry_backend_call(
        f"{surface} stat for {path}",
        lambda: backend.stat(path),
        timeout_seconds=config.retry_timeout_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
    )


def _producer_contract(
    config: RemotePercentileConfig,
    *,
    inventory_sha256: str,
    percentile_checksum: str,
    n_sae_features: int,
) -> dict[str, object]:
    cuda_runtime: str | None = None
    gpu_name: str | None = None
    gpu_capability: list[int] | None = None
    if config.device.type == "cuda":
        cuda_runtime = t.version.cuda
        gpu_name = t.cuda.get_device_name(0)
        gpu_capability = list(t.cuda.get_device_capability(0))
    return {
        "schema_version": REMOTE_CONTRACT_SCHEMA_VERSION,
        "percentile_algorithm_version": PERCENTILE_ALGORITHM_VERSION,
        "source": {
            "bucket_id": str(config.bucket_id),
            "prefix": str(config.source_prefix),
            "shard_count": config.expected_shard_count,
            "inventory_sha256": inventory_sha256,
        },
        "expected_segment_size": config.expected_segment_size,
        "percentile_table": {
            "checksum": percentile_checksum,
            "n_sae_features": n_sae_features,
        },
        "output": {
            "prefix": str(config.output_prefix),
            "zstd_level": config.zstd_level,
        },
        "producer": {
            "device_type": config.device.type,
            "torch_version": t.__version__,
            "cuda_runtime": cuda_runtime,
            "gpu_name": gpu_name,
            "gpu_capability": gpu_capability,
        },
        "serialization": {
            "safetensors_version": package_version("safetensors"),
            "zstandard_version": zstandard.__version__,
        },
    }


def _require_remote_identity(info: RemoteObjectInfo, *, surface: str) -> None:
    if (
        not isinstance(info.path, str)
        or not info.path
        or info.path.startswith("/")
        or "\\" in info.path
        or PurePosixPath(info.path).as_posix() != info.path
        or any(part in {"", ".", ".."} for part in info.path.split("/"))
    ):
        raise RemotePercentileError(f"{surface} path is malformed")
    if (
        isinstance(info.byte_length, bool)
        or not isinstance(info.byte_length, int)
        or info.byte_length <= 0
    ):
        raise RemotePercentileError(f"{surface} has invalid byte length {info.byte_length!r}")
    if not isinstance(info.xet_hash, str) or _SHA256_RE.fullmatch(info.xet_hash) is None:
        raise RemotePercentileError(f"{surface} has invalid Xet hash {info.xet_hash!r}")


def _relative_output_parts(path: str, prefix: NormalizedRemotePrefix) -> tuple[str, ...]:
    try:
        normalized = str(NormalizedRemotePrefix.parse(path))
        relative = PurePosixPath(normalized).relative_to(PurePosixPath(str(prefix)))
    except (TypeError, ValueError) as exc:
        raise RemotePercentileError(
            f"output listing for {prefix} escaped or was non-normalized: {path!r}"
        ) from exc
    return relative.parts


def _known_verified_claim(parts: tuple[str, ...]) -> bool:
    if len(parts) == 3 and parts[0] == "_verified_upload_claims":
        canonical_name = parts[1]
        hash_name = parts[2]
        allowed_name = (
            canonical_name
            in {
                REMOTE_CONTRACT_FILENAME,
                OVERLAY_STATISTICS_FILENAME,
                MANIFEST_FILENAME,
            }
            or _PAYLOAD_RE.fullmatch(canonical_name) is not None
        )
        return allowed_name and re.fullmatch(r"[0-9a-f]{64}\.json", hash_name) is not None
    if (
        len(parts) == 4
        and parts[0] == REMOTE_RECEIPTS_DIRNAME
        and parts[1] == "_verified_upload_claims"
    ):
        return (
            _RECEIPT_RE.fullmatch(parts[2]) is not None
            and re.fullmatch(r"[0-9a-f]{64}\.json", parts[3]) is not None
        )
    return False


def _verified_claim_target(
    parts: tuple[str, ...],
    *,
    prefix: NormalizedRemotePrefix,
) -> tuple[str, str]:
    """Return the canonical target path and claimed Xet hash for one claim path."""
    if not _known_verified_claim(parts):
        raise RemotePercentileError(f"malformed verified-upload claim path parts {parts!r}")
    if parts[0] == "_verified_upload_claims":
        canonical_parts = (parts[1],)
        hash_name = parts[2]
    else:
        canonical_parts = (REMOTE_RECEIPTS_DIRNAME, parts[2])
        hash_name = parts[3]
    return _join_remote(prefix, *canonical_parts), hash_name.removesuffix(".json")


def _strict_json_object(payload: bytes, *, surface: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_json_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RemotePercentileError(f"{surface} is not readable strict JSON") from exc
    if not isinstance(value, dict):
        raise RemotePercentileError(f"{surface} must be a JSON object")
    return cast(dict[str, object], value)


def _download_json_object(
    backend: BucketBackend,
    *,
    info: RemoteObjectInfo,
    config: RemotePercentileConfig,
    surface: str,
    require_canonical: bool = False,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="pct-remote-control-") as raw_dir:
        local_path = Path(raw_dir) / PurePosixPath(info.path).name
        download_verified_object(
            backend,
            expected=info,
            final_path=local_path,
            timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
        )
        try:
            payload = local_path.read_bytes()
        except OSError as exc:
            raise RemotePercentileError(f"{surface} is not readable canonical JSON") from exc
    result = _strict_json_object(payload, surface=surface)
    if require_canonical and payload != _canonical_json_bytes(result):
        raise RemotePercentileError(f"{surface} is not canonical JSON")
    return result


def _download_verified_claim(
    backend: BucketBackend,
    *,
    claim_info: RemoteObjectInfo,
    canonical_path: str,
    claimed_xet_hash: str,
    config: RemotePercentileConfig,
) -> RemoteObjectInfo:
    with tempfile.TemporaryDirectory(prefix="pct-remote-claim-") as raw_dir:
        local_path = Path(raw_dir) / PurePosixPath(claim_info.path).name
        download_verified_object(
            backend,
            expected=claim_info,
            final_path=local_path,
            timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
        )
        try:
            payload = local_path.read_bytes()
        except OSError as exc:
            raise RemotePercentileError(
                f"verified-upload claim for {canonical_path} is unreadable"
            ) from exc
    try:
        return parse_verified_upload_claim_bytes(
            payload,
            expected_path=canonical_path,
            expected_xet_hash=claimed_xet_hash,
        )
    except RemoteTrainingError as exc:
        raise RemotePercentileError(
            f"verified-upload claim for {canonical_path} is invalid"
        ) from exc


def _parse_object_info(raw: object, *, surface: str) -> RemoteObjectInfo:
    if not isinstance(raw, dict) or set(raw) != {"path", "byte_length", "xet_hash"}:
        raise RemotePercentileError(f"{surface} identity is malformed")
    raw = cast(dict[str, object], raw)
    path = raw["path"]
    byte_length = raw["byte_length"]
    xet_hash = raw["xet_hash"]
    if not isinstance(path, str):
        raise RemotePercentileError(f"{surface} path is malformed")
    info = RemoteObjectInfo(
        path=path, byte_length=cast(int, byte_length), xet_hash=cast(str, xet_hash)
    )
    _require_remote_identity(info, surface=surface)
    return info


def parse_remote_percentile_receipt(
    raw: Mapping[str, object],
    *,
    surface: str,
) -> RemoteReceipt:
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "shard_index",
        "source",
        "output",
        "n_values",
    }
    if (
        set(raw) != expected_keys
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != REMOTE_RECEIPT_SCHEMA_VERSION
    ):
        raise RemotePercentileError(f"{surface} schema or keys are malformed")
    contract_sha = raw["contract_sha256"]
    shard_index = raw["shard_index"]
    n_values = raw["n_values"]
    if not isinstance(contract_sha, str) or _SHA256_RE.fullmatch(contract_sha) is None:
        raise RemotePercentileError(f"{surface} contract SHA-256 is malformed")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        raise RemotePercentileError(f"{surface} shard index is malformed")
    if isinstance(n_values, bool) or not isinstance(n_values, int) or n_values < 0:
        raise RemotePercentileError(f"{surface} n_values is malformed")
    source = _parse_object_info(raw["source"], surface=f"{surface} source")
    output_raw = raw["output"]
    if not isinstance(output_raw, dict) or set(output_raw) != {
        "path",
        "byte_length",
        "xet_hash",
        "semantic_content_sha256",
    }:
        raise RemotePercentileError(f"{surface} output identity is malformed")
    output_raw = cast(dict[str, object], output_raw)
    semantic = output_raw["semantic_content_sha256"]
    if not isinstance(semantic, str) or _SHA256_RE.fullmatch(semantic) is None:
        raise RemotePercentileError(f"{surface} semantic digest is malformed")
    output = _parse_object_info(
        {key: output_raw[key] for key in ("path", "byte_length", "xet_hash")},
        surface=f"{surface} output",
    )
    return RemoteReceipt(
        contract_sha256=contract_sha,
        shard_index=shard_index,
        source=source,
        output=output,
        semantic_content_sha256=semantic,
        n_values=n_values,
    )


def validate_remote_percentile_contract(
    contract: Mapping[str, object],
    *,
    surface: str,
) -> None:
    """Validate the deterministic remote-percentile contract schema."""
    if set(contract) != {
        "schema_version",
        "percentile_algorithm_version",
        "source",
        "expected_segment_size",
        "percentile_table",
        "output",
        "producer",
        "serialization",
    }:
        raise RemotePercentileError(f"{surface} has unexpected or missing keys")
    source = contract["source"]
    table = contract["percentile_table"]
    output = contract["output"]
    producer = contract["producer"]
    serialization = contract["serialization"]
    if (
        type(contract["schema_version"]) is not int
        or contract["schema_version"] != REMOTE_CONTRACT_SCHEMA_VERSION
        or type(contract["percentile_algorithm_version"]) is not int
        or contract["percentile_algorithm_version"] != PERCENTILE_ALGORITHM_VERSION
        or isinstance(contract["expected_segment_size"], bool)
        or not isinstance(contract["expected_segment_size"], int)
        or contract["expected_segment_size"] <= 0
        or not isinstance(source, dict)
        or set(source) != {"bucket_id", "prefix", "shard_count", "inventory_sha256"}
        or not isinstance(table, dict)
        or set(table) != {"checksum", "n_sae_features"}
        or not isinstance(output, dict)
        or set(output) != {"prefix", "zstd_level"}
        or not isinstance(producer, dict)
        or set(producer)
        != {"device_type", "torch_version", "cuda_runtime", "gpu_name", "gpu_capability"}
        or not isinstance(serialization, dict)
        or set(serialization) != {"safetensors_version", "zstandard_version"}
    ):
        raise RemotePercentileError(f"{surface} schema is malformed")
    source = cast(dict[str, object], source)
    table = cast(dict[str, object], table)
    output = cast(dict[str, object], output)
    producer = cast(dict[str, object], producer)
    serialization = cast(dict[str, object], serialization)
    try:
        HfBucketId.parse(source["bucket_id"])
        source_prefix = NormalizedRemotePrefix.parse(source["prefix"])
        output_prefix = NormalizedRemotePrefix.parse(output["prefix"])
    except (TypeError, ValueError) as exc:
        raise RemotePercentileError(f"{surface} contains a malformed remote path") from exc
    device_type = producer["device_type"]
    cuda_runtime = producer["cuda_runtime"]
    gpu_name = producer["gpu_name"]
    gpu_capability = producer["gpu_capability"]
    cpu_runtime = all(item is None for item in (cuda_runtime, gpu_name, gpu_capability))
    cuda_runtime_valid = (
        isinstance(cuda_runtime, str)
        and bool(cuda_runtime)
        and isinstance(gpu_name, str)
        and bool(gpu_name)
        and isinstance(gpu_capability, list)
        and len(gpu_capability) == 2
        and all(type(item) is int and item >= 0 for item in gpu_capability)
    )
    if (
        isinstance(source["shard_count"], bool)
        or not isinstance(source["shard_count"], int)
        or source["shard_count"] <= 0
        or not isinstance(source["inventory_sha256"], str)
        or _SHA256_RE.fullmatch(source["inventory_sha256"]) is None
        or not isinstance(table["checksum"], str)
        or _SHA256_RE.fullmatch(table["checksum"]) is None
        or str(output_prefix)
        != _join_remote(source_prefix, "pct", table["checksum"][:CHECKSUM_PREFIX_LEN])
        or isinstance(table["n_sae_features"], bool)
        or not isinstance(table["n_sae_features"], int)
        or table["n_sae_features"] <= 0
        or isinstance(output["zstd_level"], bool)
        or not isinstance(output["zstd_level"], int)
        or not 1 <= output["zstd_level"] <= 22
        or device_type not in {"cpu", "cuda"}
        or not isinstance(producer["torch_version"], str)
        or not producer["torch_version"]
        or (device_type == "cpu" and not cpu_runtime)
        or (device_type == "cuda" and not cuda_runtime_valid)
        or not all(isinstance(item, str) and bool(item) for item in serialization.values())
    ):
        raise RemotePercentileError(f"{surface} identity is malformed")


def validate_remote_percentile_manifest(
    manifest: Mapping[str, object],
    *,
    checksum: str,
    contract: Mapping[str, object],
    contract_sha256: str,
    statistics_info: RemoteObjectInfo,
) -> None:
    expected_keys = {
        "checksum",
        "checksum_prefix",
        "kind",
        "base_shard_content_hash",
        "schema_version",
        "metadata",
    }
    if set(manifest) != expected_keys:
        raise RemotePercentileError("remote manifest has unexpected or missing keys")
    if (
        manifest["checksum"] != checksum
        or manifest["checksum_prefix"] != checksum[:CHECKSUM_PREFIX_LEN]
        or manifest["kind"] != OverlayKind.PERCENTILE.value
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 2
    ):
        raise RemotePercentileError("remote manifest does not identify this percentile overlay")
    base_hash = manifest["base_shard_content_hash"]
    if not isinstance(base_hash, str) or _SHA256_RE.fullmatch(base_hash) is None:
        raise RemotePercentileError("remote manifest base-shard hash is malformed")
    expected_metadata = _manifest_metadata(
        contract=contract,
        contract_sha256=contract_sha256,
        statistics_info=statistics_info,
    )
    if _canonical_json_bytes(manifest["metadata"]) != _canonical_json_bytes(expected_metadata):
        raise RemotePercentileError("remote manifest metadata differs from the run contract")


def _manifest_metadata(
    *,
    contract: Mapping[str, object],
    contract_sha256: str,
    statistics_info: RemoteObjectInfo,
) -> dict[str, object]:
    validate_remote_percentile_contract(contract, surface="remote percentile contract")
    source = cast(dict[str, object], contract["source"])
    table = cast(dict[str, object], contract["percentile_table"])
    output = cast(dict[str, object], contract["output"])
    return {
        "n_sae_features": table["n_sae_features"],
        "stats_file": OVERLAY_STATISTICS_FILENAME,
        "remote_contract_path": PurePosixPath(
            cast(str, output["prefix"]), REMOTE_CONTRACT_FILENAME
        ).as_posix(),
        "remote_contract_sha256": contract_sha256,
        "source_bucket_id": source["bucket_id"],
        "source_prefix": source["prefix"],
        "source_inventory_sha256": source["inventory_sha256"],
        "source_shard_count": source["shard_count"],
        "producer": contract["producer"],
        "statistics_object": {
            "path": statistics_info.path,
            "byte_length": statistics_info.byte_length,
            "xet_hash": statistics_info.xet_hash,
        },
        "receipt_count": source["shard_count"],
    }


def _inspect_output_state(
    backend: BucketBackend,
    *,
    config: RemotePercentileConfig,
    desired_contract: Mapping[str, object],
    contract_sha256: str,
    inventory: Sequence[RemoteFeatureShard],
    checksum: str,
    download_controls: bool,
) -> RemoteOutputState:
    listing = tuple(
        _list_with_retry(
            backend,
            str(config.output_prefix),
            config=config,
            surface="percentile output",
        )
    )
    state = RemoteOutputState(listing=listing)
    seen_paths: set[str] = set()
    claim_paths: dict[str, tuple[RemoteObjectInfo, str]] = {}
    for info in listing:
        if not isinstance(info.path, str):
            raise RemotePercentileError("output object path is malformed")
        if info.path in seen_paths:
            raise RemotePercentileError(f"duplicate output listing entry {info.path!r}")
        seen_paths.add(info.path)
        parts = _relative_output_parts(info.path, config.output_prefix)
        if parts and parts[0] == "tok":
            continue
        _require_remote_identity(info, surface=f"output object {info.path!r}")
        if "_verified_upload_claims" in parts:
            target_path, claimed_xet = _verified_claim_target(
                parts,
                prefix=config.output_prefix,
            )
            if target_path in claim_paths:
                raise RemotePercentileError(
                    f"multiple verified-upload claims exist for {target_path}"
                )
            claim_paths[target_path] = (info, claimed_xet)
            state.claim_infos[target_path] = info
            continue
        if parts == (REMOTE_CONTRACT_FILENAME,):
            state.contract_info = info
            continue
        if parts == (OVERLAY_STATISTICS_FILENAME,):
            state.statistics_info = info
            continue
        if parts == (MANIFEST_FILENAME,):
            state.manifest_info = info
            continue
        if len(parts) == 1:
            match = _PAYLOAD_RE.fullmatch(parts[0])
            if match is not None:
                index = int(match.group("index"))
                state.payloads[index] = info
                continue
            if parts[0].startswith("shard_") or COMPRESSED_EXT in parts[0]:
                raise RemotePercentileError(
                    f"malformed direct percentile payload name {info.path!r}"
                )
        if len(parts) == 2 and parts[0] == REMOTE_RECEIPTS_DIRNAME:
            match = _RECEIPT_RE.fullmatch(parts[1])
            if match is None:
                raise RemotePercentileError(f"malformed receipt path {info.path!r}")
            index = int(match.group("index"))
            state.receipt_infos[index] = info
            continue
        raise RemotePercentileError(f"unknown object in percentile output family: {info.path!r}")

    canonical_without_contract = (
        bool(state.payloads)
        or bool(state.receipt_infos)
        or state.statistics_info is not None
        or state.manifest_info is not None
    )
    contract_path = _join_remote(config.output_prefix, REMOTE_CONTRACT_FILENAME)
    noncontract_claims = set(claim_paths) - {contract_path}
    if state.contract_info is None and (canonical_without_contract or noncontract_claims):
        raise RemotePercentileError(
            "percentile output state exists without a remote contract; only an "
            "interrupted claim for the deterministic contract itself is recoverable"
        )
    expected_indices = set(range(config.expected_shard_count))
    extras = (set(state.payloads) | set(state.receipt_infos)) - expected_indices
    if extras:
        raise RemotePercentileError(
            f"percentile output has unexpected shard indices {sorted(extras)}"
        )
    expected_canonical_paths = {
        _join_remote(config.output_prefix, REMOTE_CONTRACT_FILENAME),
        _join_remote(config.output_prefix, OVERLAY_STATISTICS_FILENAME),
        _join_remote(config.output_prefix, MANIFEST_FILENAME),
        *(_payload_remote_path(config, index) for index in expected_indices),
        *(_receipt_remote_path(config, index) for index in expected_indices),
    }
    unexpected_claim_targets = set(claim_paths) - expected_canonical_paths
    if unexpected_claim_targets:
        raise RemotePercentileError(
            "verified-upload claims target unexpected canonical objects: "
            f"{sorted(unexpected_claim_targets)}"
        )
    canonical_by_path = {
        info.path: info
        for info in (
            state.contract_info,
            *state.payloads.values(),
            *state.receipt_infos.values(),
            state.statistics_info,
            state.manifest_info,
        )
        if info is not None
    }
    missing_claims = set(canonical_by_path) - set(claim_paths)
    if missing_claims:
        raise RemotePercentileError(
            f"canonical percentile objects lack verified-upload claims: {sorted(missing_claims)}"
        )
    for target_path, canonical in canonical_by_path.items():
        _, claimed_xet = claim_paths[target_path]
        if claimed_xet != canonical.xet_hash:
            raise RemotePercentileError(
                f"verified-upload claim Xet hash differs from canonical object {target_path}"
            )
    missing_payloads = set(state.receipt_infos) - set(state.payloads)
    if missing_payloads:
        raise RemotePercentileError(
            f"receipt(s) exist without payload(s): {sorted(missing_payloads)}"
        )
    if state.manifest_info is not None and (
        set(state.payloads) != expected_indices
        or set(state.receipt_infos) != expected_indices
        or state.statistics_info is None
    ):
        raise RemotePercentileError(
            "remote manifest exists without complete payload, receipt, and statistics coverage"
        )
    if not download_controls:
        return state

    for target_path, (claim_info, claimed_xet) in sorted(claim_paths.items()):
        claimed = _download_verified_claim(
            backend,
            claim_info=claim_info,
            canonical_path=target_path,
            claimed_xet_hash=claimed_xet,
            config=config,
        )
        canonical = canonical_by_path.get(target_path)
        if canonical is not None and claimed != canonical:
            raise RemotePercentileError(
                f"verified-upload claim identity differs from canonical object {target_path}"
            )
        state.claimed_objects[target_path] = claimed

    if state.contract_info is not None:
        contract = _download_json_object(
            backend,
            info=state.contract_info,
            config=config,
            surface="remote percentile contract",
            require_canonical=True,
        )
        if _canonical_json_bytes(contract) != _canonical_json_bytes(desired_contract):
            raise RemotePercentileError(
                "immutable percentile output collision: remote contract differs"
            )
        state.contract = contract

    inventory_by_index = {entry.index: entry.info for entry in inventory}
    for index, info in sorted(state.receipt_infos.items()):
        raw = _download_json_object(
            backend,
            info=info,
            config=config,
            surface=f"receipt {index}",
            require_canonical=True,
        )
        receipt = parse_remote_percentile_receipt(raw, surface=f"receipt {index}")
        expected_output_path = _join_remote(
            config.output_prefix, f"shard_{index:05d}{COMPRESSED_EXT}"
        )
        if (
            receipt.contract_sha256 != contract_sha256
            or receipt.shard_index != index
            or receipt.source != inventory_by_index[index]
            or receipt.output.path != expected_output_path
            or receipt.output != state.payloads[index]
        ):
            raise RemotePercentileError(f"receipt {index} disagrees with sealed identities")
        state.receipts[index] = receipt

    if state.manifest_info is not None:
        raw_manifest = _download_json_object(
            backend,
            info=state.manifest_info,
            config=config,
            surface="remote percentile manifest",
        )
        assert state.statistics_info is not None
        validate_remote_percentile_manifest(
            raw_manifest,
            checksum=checksum,
            contract=desired_contract,
            contract_sha256=contract_sha256,
            statistics_info=state.statistics_info,
        )
        state.manifest = raw_manifest
    return state


def _revalidate_remote_identities(
    backend: BucketBackend,
    state: RemoteOutputState,
    *,
    config: RemotePercentileConfig,
) -> None:
    canonical = [
        state.contract_info,
        *state.payloads.values(),
        *state.receipt_infos.values(),
        state.statistics_info,
        state.manifest_info,
        *state.claim_infos.values(),
    ]
    for expected in canonical:
        if expected is None:
            continue
        observed = _stat_with_retry(
            backend,
            expected.path,
            config=config,
            surface="percentile output identity fence",
        )
        if observed != expected:
            raise RemotePercentileError(
                f"remote output identity changed after listing: {expected.path}"
            )


def required_staging_capacity_bytes(
    inventory: Sequence[RemoteFeatureShard],
    *,
    max_output_shard_bytes: int,
    statistics_bytes: int,
) -> int:
    largest_sources = sorted((entry.info.byte_length for entry in inventory), reverse=True)[:2]
    return (
        sum(largest_sources)
        + max_output_shard_bytes
        + 2 * statistics_bytes
        + CAPACITY_RESERVE_BYTES
    )


def required_decompression_lookahead_memory_bytes(
    inventory: Sequence[RemoteFeatureShard],
    *,
    max_decompressed_shard_bytes: int,
) -> int:
    """Conservative host envelope for one CUDA decompression lookahead."""
    largest_compressed = max((entry.info.byte_length for entry in inventory), default=0)
    return (
        3 * max_decompressed_shard_bytes + largest_compressed + DECOMPRESSION_MEMORY_RESERVE_BYTES
    )


def _decompression_lookahead_enabled(
    config: RemotePercentileConfig,
    inventory: Sequence[RemoteFeatureShard],
) -> bool:
    return config.device.type == "cuda" and len(inventory) >= 2


def _require_decompression_lookahead_memory(
    inventory: Sequence[RemoteFeatureShard],
    *,
    max_decompressed_shard_bytes: int,
) -> tuple[int, int]:
    required = required_decompression_lookahead_memory_bytes(
        inventory,
        max_decompressed_shard_bytes=max_decompressed_shard_bytes,
    )
    available = effective_available_memory_bytes()
    if available < required:
        raise RemotePercentileError(
            "remote percentile CUDA decompression lookahead memory admission failed: "
            f"required={required:,}, available={available:,}, "
            "max_decompressed_shard_bytes="
            f"{max_decompressed_shard_bytes:,}"
        )
    return required, available


def _safe_regular_identity(path: Path, *, surface: str) -> RegularFileIdentity:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RemotePercentileError(f"cannot inspect {surface} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RemotePercentileError(f"{surface} must be a direct single-link regular file: {path}")
    return RegularFileIdentity.from_stat(info)


def _pid_live_or_inaccessible(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PercentileStaging:
    """Locked runner-owned staging namespace with fail-closed cleanup."""

    def __init__(
        self,
        *,
        config: RemotePercentileConfig,
        contract_sha256: str,
        required_capacity: int,
        inventory: Sequence[RemoteFeatureShard],
        output_state: RemoteOutputState,
    ) -> None:
        self.config = config
        self.contract_sha256 = contract_sha256
        self.required_capacity = required_capacity
        self.inventory = tuple(inventory)
        self.output_state = output_state
        self.lock_fd: int | None = None
        self.high_water_bytes = 0
        self.high_water_bodies = 0
        self.admission_free_bytes = 0
        self.admission_recoverable_bytes = 0
        self.directory_identity: tuple[int, int] | None = None
        self.directory_fd: int | None = None
        self.lock_identity: tuple[int, int] | None = None

    def _owner_payload(self) -> dict[str, object]:
        return {
            "schema_version": STAGING_OWNER_SCHEMA_VERSION,
            "bucket_id": str(self.config.bucket_id),
            "source_prefix": str(self.config.source_prefix),
            "output_prefix": str(self.config.output_prefix),
            "contract_sha256": self.contract_sha256,
        }

    def _recognized(self, name: str) -> bool:
        if name in {
            _LOCK_FILENAME,
            _OWNER_FILENAME,
            OVERLAY_STATISTICS_FILENAME,
            REMOTE_CONTRACT_FILENAME,
            MANIFEST_FILENAME,
        }:
            return True
        source_match = _SOURCE_RE.fullmatch(name)
        if source_match is not None:
            index = int(source_match.group("index"))
            return (
                index < len(self.inventory)
                and name == PurePosixPath(self.inventory[index].info.path).name
            )
        for pattern in (_OUTPUT_STAGE_RE, _VERIFY_STAGE_RE, _RECEIPT_STAGE_RE):
            match = pattern.fullmatch(name)
            if match is not None:
                return int(match.group("index")) < self.config.expected_shard_count
        return False

    def assert_bound(self) -> None:
        if self.directory_identity is None:
            return
        if self.directory_fd is None or self.lock_fd is None:
            raise RemotePercentileError(
                "remote percentile staging directory or lock descriptor is closed"
            )
        try:
            info = self.config.staging_dir.lstat()
            opened_directory = os.fstat(self.directory_fd)
            opened_lock = os.fstat(self.lock_fd)
        except OSError as exc:
            raise RemotePercentileError(
                f"remote percentile staging path became unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != self.directory_identity
            or not stat.S_ISDIR(opened_directory.st_mode)
            or (opened_directory.st_dev, opened_directory.st_ino) != self.directory_identity
        ):
            raise RemotePercentileError(
                "remote percentile staging path no longer names the locked directory"
            )
        if self.lock_identity is None:
            return
        try:
            anchored_lock = os.stat(
                _LOCK_FILENAME,
                dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
            linked_lock = (self.config.staging_dir / _LOCK_FILENAME).lstat()
        except OSError as exc:
            raise RemotePercentileError(
                f"remote percentile staging lock became unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(anchored_lock.st_mode)
            or not stat.S_ISREG(linked_lock.st_mode)
            or anchored_lock.st_nlink != 1
            or linked_lock.st_nlink != 1
            or not stat.S_ISREG(opened_lock.st_mode)
            or opened_lock.st_nlink != 1
            or (anchored_lock.st_dev, anchored_lock.st_ino) != self.lock_identity
            or (linked_lock.st_dev, linked_lock.st_ino) != self.lock_identity
            or (opened_lock.st_dev, opened_lock.st_ino) != self.lock_identity
        ):
            raise RemotePercentileError(
                "remote percentile staging lock path no longer names the locked inode"
            )

    def _safe_unlink(
        self,
        path: Path,
        *,
        expected: RegularFileIdentity | None = None,
    ) -> None:
        self.assert_bound()
        if self.directory_fd is None or path.parent != self.config.staging_dir:
            raise RemotePercentileError(
                f"remote percentile cleanup path is outside the locked staging directory: {path}"
            )
        try:
            opened = os.stat(
                path.name,
                dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RemotePercentileError(
                f"remote percentile cleanup entry is unavailable: {path}"
            ) from exc
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RemotePercentileError(
                f"remote percentile cleanup entry is not a safe regular file: {path}"
            )
        observed = RegularFileIdentity.from_stat(opened)
        if expected is not None and observed != expected:
            raise RemotePercentileError(f"runner-owned staging file changed before cleanup: {path}")
        quarantine_name = f".{path.name}.quarantine.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        os.rename(
            path.name,
            quarantine_name,
            src_dir_fd=self.directory_fd,
            dst_dir_fd=self.directory_fd,
        )
        quarantined = os.stat(
            quarantine_name,
            dir_fd=self.directory_fd,
            follow_symlinks=False,
        )
        quarantined_identity = RegularFileIdentity.from_stat(quarantined)
        # A rename legitimately changes ctime, which is part of
        # RegularFileIdentity.  Bind the quarantine to the same inode and the
        # same content-relevant attributes instead.
        same_quarantined_file = (
            quarantined_identity.device == observed.device
            and quarantined_identity.inode == observed.inode
            and quarantined_identity.mode == observed.mode
            and quarantined_identity.size == observed.size
            and quarantined_identity.mtime_ns == observed.mtime_ns
            and quarantined.st_nlink == opened.st_nlink
        )
        if not same_quarantined_file:
            raise RemotePercentileError(
                "runner-owned staging file changed during quarantine; no changed "
                f"entry was deleted and quarantine {quarantine_name!r} remains: {path}"
            )
        os.unlink(quarantine_name, dir_fd=self.directory_fd)
        self.assert_bound()

    def _classify(self, *, mutate: bool) -> tuple[int, list[Path]]:
        recoverable = 0
        entries: list[Path] = []
        if not self.config.staging_dir.exists():
            return recoverable, entries
        for entry in self.config.staging_dir.iterdir():
            remote_temp_match = _REMOTE_TMP_RE.fullmatch(entry.name)
            atomic_temp_match = _ATOMIC_TMP_RE.fullmatch(entry.name)
            temp_match = remote_temp_match or atomic_temp_match
            if temp_match is not None:
                if not self._recognized(temp_match.group("base")):
                    raise RemotePercentileError(
                        f"unknown entry in remote percentile staging; preserving it: {entry}"
                    )
                _safe_regular_identity(entry, surface="remote percentile temporary")
                pid = int(temp_match.group("pid"))
                if _pid_live_or_inaccessible(pid):
                    raise RemotePercentileError(
                        f"live or inaccessible staging temporary blocks adoption: {entry}"
                    )
                if mutate:
                    self._safe_unlink(entry)
                else:
                    entries.append(entry)
                    recoverable += entry.stat().st_size
                continue
            if not self._recognized(entry.name):
                raise RemotePercentileError(
                    f"unknown entry in remote percentile staging; preserving it: {entry}"
                )
            identity = _safe_regular_identity(entry, surface="remote percentile staging entry")
            entries.append(entry)
            if entry.name not in {_LOCK_FILENAME, _OWNER_FILENAME}:
                recoverable += identity.size
        return recoverable, entries

    def preflight_without_mutation(self) -> tuple[int, int]:
        staging = self.config.staging_dir
        if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
            raise RemotePercentileError(
                f"remote percentile staging must be a real directory: {staging}"
            )
        recoverable_files, entries = self._classify(mutate=False)
        owner_path = staging / _OWNER_FILENAME
        if owner_path.exists():
            _safe_regular_identity(owner_path, surface="remote percentile owner record")
            try:
                owner = _strict_json_object(
                    owner_path.read_bytes(),
                    surface="remote percentile owner record",
                )
            except OSError as exc:
                raise RemotePercentileError("remote percentile owner record is unreadable") from exc
            if _canonical_json_bytes(owner) != _canonical_json_bytes(self._owner_payload()):
                raise RemotePercentileError("remote percentile staging belongs to another contract")
        elif any(entry.name != _LOCK_FILENAME for entry in entries):
            raise RemotePercentileError("staging has no owner record and is not empty")
        probe = staging
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        recoverable = free + recoverable_files
        if recoverable < self.required_capacity:
            raise RemotePercentileError(
                "remote percentile staging admission failed: "
                f"required={self.required_capacity:,}, free={free:,}, "
                f"recoverable={recoverable:,}"
            )
        return free, recoverable

    def open(self, *, require_capacity: bool = True) -> None:
        staging = self.config.staging_dir
        if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
            raise RemotePercentileError(
                f"remote percentile staging must be a real directory: {staging}"
            )
        staging.mkdir(parents=True, exist_ok=True)
        if staging.is_symlink() or not staging.is_dir():
            raise RemotePercentileError(
                f"remote percentile staging must be a real directory: {staging}"
            )
        staging_info = staging.lstat()
        self.directory_identity = (staging_info.st_dev, staging_info.st_ino)
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            raise RemotePercentileError(
                "remote percentile staging requires os.O_DIRECTORY and os.O_NOFOLLOW"
            )
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            self.directory_fd = os.open(staging, directory_flags)
        except OSError as exc:
            self.close()
            raise RemotePercentileError(
                f"remote percentile staging directory could not be anchored: {staging}"
            ) from exc
        opened_directory = os.fstat(self.directory_fd)
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or (opened_directory.st_dev, opened_directory.st_ino) != self.directory_identity
        ):
            self.close()
            raise RemotePercentileError(
                "remote percentile staging path changed while opening its directory"
            )
        lock_path = staging / _LOCK_FILENAME
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            self.lock_fd = os.open(
                _LOCK_FILENAME,
                flags,
                0o600,
                dir_fd=self.directory_fd,
            )
            opened = os.fstat(self.lock_fd)
            linked = lock_path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
                or opened.st_nlink != 1
                or linked.st_nlink != 1
            ):
                raise RemotePercentileError("remote percentile staging lock is unsafe")
            self.lock_identity = (opened.st_dev, opened.st_ino)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            linked_after_lock = lock_path.lstat()
            if (
                not stat.S_ISREG(linked_after_lock.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (linked_after_lock.st_dev, linked_after_lock.st_ino)
                or linked_after_lock.st_nlink != 1
            ):
                raise RemotePercentileError(
                    "remote percentile staging lock path changed while locking"
                )
        except OSError as exc:
            self.close()
            raise RemotePercentileError(
                f"remote percentile staging is already in use: {staging}"
            ) from exc
        except BaseException:
            self.close()
            raise

        self.assert_bound()
        _, preexisting_entries = self._classify(mutate=False)
        owner_path = staging / _OWNER_FILENAME
        if not owner_path.exists():
            unexpected = [path for path in preexisting_entries if path.name != _LOCK_FILENAME]
            if unexpected:
                raise RemotePercentileError(
                    "staging has no owner record and is not empty: "
                    f"{[str(path) for path in unexpected]}"
                )
            atomic_json_dump(self._owner_payload(), owner_path)
            self.assert_bound()
        _safe_regular_identity(owner_path, surface="remote percentile owner record")
        try:
            owner = _strict_json_object(
                owner_path.read_bytes(),
                surface="remote percentile owner record",
            )
        except OSError as exc:
            raise RemotePercentileError("remote percentile owner record is unreadable") from exc
        if _canonical_json_bytes(owner) != _canonical_json_bytes(self._owner_payload()):
            raise RemotePercentileError(
                "remote percentile staging belongs to another contract: "
                f"expected={self._owner_payload()!r}, observed={owner!r}"
            )

        _, entries = self._classify(mutate=True)
        keep_source_names = {PurePosixPath(entry.info.path).name for entry in self.inventory[:2]}
        first_incomplete_index = next(
            (
                index
                for index in range(self.config.expected_shard_count)
                if index not in self.output_state.receipts
            ),
            None,
        )
        retained_output = False
        retained_bytes = 0
        for entry in entries:
            if entry.name in {_LOCK_FILENAME, _OWNER_FILENAME}:
                continue
            source_match = _SOURCE_RE.fullmatch(entry.name)
            output_match = _OUTPUT_STAGE_RE.fullmatch(entry.name)
            keep = entry.name == OVERLAY_STATISTICS_FILENAME
            if source_match is not None and entry.name in keep_source_names:
                index = int(source_match.group("index"))
                try:
                    observed = verified_local_object_info(
                        local_path=entry,
                        remote_path=self.inventory[index].info.path,
                    )
                except Exception:
                    keep = False
                else:
                    keep = observed == self.inventory[index].info
            elif output_match is not None and not retained_output:
                index = int(output_match.group("index"))
                # A remotely visible payload supersedes the staged body.  A
                # claim-only interruption has no canonical payload and needs
                # the exact retained bytes to finish the immutable upload.
                keep = index == first_incomplete_index and index not in self.output_state.payloads
                retained_output = keep
            if keep:
                retained_bytes += entry.stat().st_size
            else:
                self._safe_unlink(entry)
        free = shutil.disk_usage(staging).free
        # All recognized retained bytes are reusable or safely removable.
        recoverable = free + retained_bytes
        self.admission_free_bytes = free
        self.admission_recoverable_bytes = recoverable
        if require_capacity and recoverable < self.required_capacity:
            raise RemotePercentileError(
                "remote percentile staging admission failed before remote write: "
                f"required={self.required_capacity:,}, free={free:,}, "
                f"retained={retained_bytes:,}, recoverable={recoverable:,}"
            )
        self.observe()

    def observe(self) -> None:
        total = 0
        bodies = 0
        staging = self.config.staging_dir
        self.assert_bound()
        if staging.is_symlink() or not staging.is_dir():
            return
        try:
            entries = list(staging.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.name in {_LOCK_FILENAME, _OWNER_FILENAME}:
                continue
            try:
                info = entry.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                bodies += 1
        self.high_water_bytes = max(self.high_water_bytes, total)
        self.high_water_bodies = max(self.high_water_bodies, bodies)

    def cleanup_success(self) -> None:
        self.assert_bound()
        for entry in list(self.config.staging_dir.iterdir()):
            if entry.name in {_LOCK_FILENAME, _OWNER_FILENAME}:
                continue
            if not self._recognized(entry.name):
                raise RemotePercentileError(
                    f"unknown staging entry appeared during cleanup: {entry}"
                )
            self._safe_unlink(entry)
        self.assert_bound()

    def close(self) -> None:
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self.lock_fd)
                self.lock_fd = None
        self.lock_identity = None
        if self.directory_fd is not None:
            os.close(self.directory_fd)
            self.directory_fd = None


def _source_local_path(config: RemotePercentileConfig, entry: RemoteFeatureShard) -> Path:
    return config.staging_dir / PurePosixPath(entry.info.path).name


def _read_percentile_overlay_stable(
    path: Path,
    *,
    expected_numel: int,
    expected_identity: RegularFileIdentity | None = None,
) -> tuple[t.Tensor, str, RegularFileIdentity, str]:
    """Decode and digest one direct overlay body from a single bound inode."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemotePercentileError("stable overlay reads require os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RemotePercentileError(f"staged overlay could not be opened safely: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RemotePercentileError(
                f"staged overlay is not a direct single-link regular file: {path}"
            )
        identity = RegularFileIdentity.from_stat(opened)
        if expected_identity is not None and identity != expected_identity:
            raise RemotePercentileError(f"staged overlay changed while being opened: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            compressed = handle.read()
        after_read = os.fstat(descriptor)
        if (
            after_read.st_nlink != 1
            or RegularFileIdentity.from_stat(after_read) != identity
            or len(compressed) != identity.size
        ):
            raise RemotePercentileError(f"staged overlay changed during stable read: {path}")
        try:
            tensors = load_overlay_shard_bytes(compressed)
        except Exception as exc:
            raise RemotePercentileError(f"staged overlay is unreadable: {path}") from exc
        key = TensorKey.FEATURE_PERCENTILES.value
        if set(tensors) != {key}:
            raise RemotePercentileError(
                f"staged overlay has keys {sorted(tensors)}, expected exactly {key!r}: {path}"
            )
        values = tensors[key]
        if values.ndim != 1 or values.dtype != t.uint8 or values.numel() != expected_numel:
            raise RemotePercentileError(
                "staged overlay must contain one rank-1 uint8 percentile value "
                f"per feature ID; expected={expected_numel}, shape={tuple(values.shape)}, "
                f"dtype={values.dtype}: {path}"
            )
        semantic_digest = overlay_tensors_content_digest(tensors)
        xet_result = cast(Any, hf_xet).hash_files([f"/proc/self/fd/{descriptor}"])[0]
        after_xet = os.fstat(descriptor)
        try:
            linked = path.lstat()
        except OSError as exc:
            raise RemotePercentileError(
                f"staged overlay path changed after stable read: {path}"
            ) from exc
        if (
            after_xet.st_nlink != 1
            or RegularFileIdentity.from_stat(after_xet) != identity
            or linked.st_nlink != 1
            or RegularFileIdentity.from_stat(linked) != identity
            or int(xet_result.file_size) != identity.size
        ):
            raise RemotePercentileError(f"staged overlay path changed after stable read: {path}")
        return values, semantic_digest, identity, str(xet_result.hash)
    finally:
        os.close(descriptor)


def _download_source(
    backend: BucketBackend,
    *,
    config: RemotePercentileConfig,
    entry: RemoteFeatureShard,
    cancellation: threading.Event,
    assert_staging: Callable[[], None],
    unlink_staged: Callable[[Path, RegularFileIdentity], None],
) -> tuple[Path, RegularFileIdentity, str]:
    assert_staging()
    path = _source_local_path(config, entry)
    if path.exists() or path.is_symlink():
        identity = _safe_regular_identity(path, surface="retained source body")
        try:
            observed = verified_local_object_info(
                local_path=path,
                remote_path=entry.info.path,
            )
        except Exception:
            unlink_staged(path, identity)
        else:
            if (
                observed == entry.info
                and _safe_regular_identity(path, surface="verified retained source body")
                == identity
            ):
                assert_staging()
                return path, identity, "reused"
            unlink_staged(path, identity)
    bound: RegularFileIdentity | None = None

    def bind(published: os.stat_result) -> None:
        nonlocal bound
        if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
            raise RemotePercentileError("downloaded source body is not a safe regular file")
        bound = RegularFileIdentity.from_verified_publication(
            published,
            verified_byte_length=entry.info.byte_length,
            path=path,
        )

    download_verified_object(
        backend,
        expected=entry.info,
        final_path=path,
        timeout_seconds=config.retry_timeout_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
        cancellation_event=cancellation,
        verified_publication_callback=bind,
    )
    if bound is None:
        raise RemotePercentileError(f"downloaded source has no bound identity: {entry.info.path}")
    assert_staging()
    return path, bound, "downloaded"


def _decompress_feature_source(
    compressed: bytes,
    *,
    max_decompressed_shard_bytes: int,
    source_name: str,
) -> bytes:
    """Decompress one authenticated source under an explicit plain-byte cap."""
    try:
        frame_size = zstandard.frame_content_size(compressed)
    except zstandard.ZstdError as exc:
        raise RemotePercentileError(
            f"source shard has an invalid zstd frame: {source_name}"
        ) from exc
    if frame_size in {zstandard.CONTENTSIZE_ERROR, -2}:
        raise RemotePercentileError(f"source shard has an invalid zstd frame: {source_name}")
    if (
        frame_size not in {zstandard.CONTENTSIZE_UNKNOWN, -1}
        and frame_size > max_decompressed_shard_bytes
    ):
        raise RemotePercentileError(
            f"source shard {source_name} declares {frame_size:,} decompressed bytes, "
            "exceeding remote.max_decompressed_shard_bytes="
            f"{max_decompressed_shard_bytes:,}"
        )
    try:
        plain = zstandard.ZstdDecompressor().decompress(
            compressed,
            max_output_size=max_decompressed_shard_bytes,
        )
    except zstandard.ZstdError as exc:
        raise RemotePercentileError(
            f"source shard could not be decompressed within the configured bound: {source_name}"
        ) from exc
    if len(plain) > max_decompressed_shard_bytes:
        raise RemotePercentileError(
            f"source shard {source_name} produced {len(plain):,} decompressed bytes, "
            "exceeding remote.max_decompressed_shard_bytes="
            f"{max_decompressed_shard_bytes:,}"
        )
    return plain


def _prepare_streamed_percentile_source(
    backend: BucketBackend,
    *,
    config: RemotePercentileConfig,
    entry: RemoteFeatureShard,
    cancellation: threading.Event,
    assert_staging: Callable[[], None],
    unlink_staged: Callable[[Path, RegularFileIdentity], None],
    base_hasher: BaseShardContentHasher,
    decompress: bool,
) -> _PreparedPercentileSource:
    """Download, authenticate, hash, and optionally decompress one source."""
    path, identity, disposition = _download_source(
        backend,
        config=config,
        entry=entry,
        cancellation=cancellation,
        assert_staging=assert_staging,
        unlink_staged=unlink_staged,
    )
    compressed = read_feature_compressed_bytes_stable(
        path,
        expected_identity=identity,
    )
    preparation_error: BaseException | None = None
    try:
        base_hasher.update(PurePosixPath(entry.info.path).name, compressed)
        if not decompress:
            return _PreparedPercentileSource(
                disposition=disposition,
                safetensors_bytes=None,
            )
        if config.max_decompressed_shard_bytes is None:
            raise RemotePercentileError(
                "CUDA decompression lookahead requires remote.max_decompressed_shard_bytes"
            )
        return _PreparedPercentileSource(
            disposition=disposition,
            safetensors_bytes=_decompress_feature_source(
                compressed,
                max_decompressed_shard_bytes=config.max_decompressed_shard_bytes,
                source_name=PurePosixPath(entry.info.path).name,
            ),
        )
    except BaseException as exc:
        preparation_error = exc
        raise
    finally:
        try:
            unlink_staged(path, identity)
        except BaseException as cleanup_error:
            if preparation_error is None:
                raise
            if cleanup_error is not preparation_error:
                preparation_error.add_note(
                    f"source cleanup also failed after preparation error: {cleanup_error!r}"
                )


def _compute_streamed_percentile_values_from_safetensors(
    safetensors_bytes: bytes,
    *,
    entry: RemoteFeatureShard,
    config: RemotePercentileConfig,
    percentile_table: t.Tensor,
) -> t.Tensor:
    """Parse and compute one already-decompressed source on the main thread."""
    if config.device.type == "cuda":
        encoded = parse_percentile_feature_safetensors_bytes(safetensors_bytes)
        del safetensors_bytes
        metadata = encoded.metadata
    else:
        data = decode_percentile_feature_safetensors_bytes(safetensors_bytes)
        del safetensors_bytes
        metadata = data
    segment_size = metadata.get("segment_size")
    if segment_size != config.expected_segment_size:
        raise RemotePercentileError(
            f"source shard {entry.info.path} has segment_size={segment_size!r}, "
            f"expected {config.expected_segment_size}"
        )
    if config.device.type == "cuda":
        return compute_percentile_overlay_values_from_encoded(
            encoded,
            percentile_table,
            device=config.device,
            source_name=PurePosixPath(entry.info.path).name,
        )
    return compute_percentile_overlay_values(
        data,
        percentile_table,
        device=config.device,
        source_name=PurePosixPath(entry.info.path).name,
    )


def _compute_streamed_percentile_values(
    compressed: bytes,
    *,
    entry: RemoteFeatureShard,
    config: RemotePercentileConfig,
    percentile_table: t.Tensor,
) -> t.Tensor:
    """Decode and compute one shard without retaining its raw tensors in the loop."""
    return _compute_streamed_percentile_values_from_safetensors(
        zstandard.ZstdDecompressor().decompress(compressed),
        entry=entry,
        config=config,
        percentile_table=percentile_table,
    )


def _receipt_remote_path(config: RemotePercentileConfig, index: int) -> str:
    return _join_remote(
        config.output_prefix,
        REMOTE_RECEIPTS_DIRNAME,
        f"shard_{index:05d}.json",
    )


def _payload_remote_path(config: RemotePercentileConfig, index: int) -> str:
    return _join_remote(config.output_prefix, f"shard_{index:05d}{COMPRESSED_EXT}")


def _write_receipt(path: Path, receipt: RemoteReceipt) -> bytes:
    payload = _canonical_json_bytes(receipt.to_json())
    atomic_write_bytes(payload, path)
    return payload


def _publish_receipt(
    backend: BucketBackend,
    *,
    config: RemotePercentileConfig,
    local_path: Path,
    receipt: RemoteReceipt,
    assert_staging: Callable[[], None],
    expected_claim: RemoteObjectInfo | None = None,
) -> _PublishedFiles:
    assert_staging()
    receipt_payload = _write_receipt(local_path, receipt)
    receipt_identity = _safe_regular_identity(local_path, surface="staged receipt")
    receipt_remote_path = _receipt_remote_path(config, receipt.shard_index)
    receipt_info = verified_local_object_info(
        local_path=local_path,
        remote_path=receipt_remote_path,
    )
    if expected_claim is not None and expected_claim != receipt_info:
        raise RemotePercentileError("staged receipt bytes differ from the immutable upload claim")
    upload_verified_object(
        backend,
        local_path=local_path,
        remote_path=receipt_remote_path,
        timeout_seconds=config.retry_timeout_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
        expected_local=receipt_info,
        expected_bytes=receipt_payload,
    )
    if _safe_regular_identity(local_path, surface="published staged receipt") != receipt_identity:
        raise RemotePercentileError(f"staged receipt changed during upload: {local_path}")
    assert_staging()
    return _PublishedFiles(receipt=receipt, receipt_identity=receipt_identity)


def _publish_payload_and_receipt(
    backend: BucketBackend,
    *,
    config: RemotePercentileConfig,
    contract_sha256: str,
    source: RemoteObjectInfo,
    index: int,
    output_path: Path,
    semantic_digest: str,
    n_values: int,
    output_identity: RegularFileIdentity,
    output_xet_hash: str,
    assert_staging: Callable[[], None],
    expected_payload_claim: RemoteObjectInfo | None = None,
    expected_receipt_claim: RemoteObjectInfo | None = None,
) -> _PublishedFiles:
    assert_staging()
    if (
        _safe_regular_identity(output_path, surface="staged output before upload")
        != output_identity
    ):
        raise RemotePercentileError(f"staged output changed before upload: {output_path}")
    expected_output_info = RemoteObjectInfo(
        path=_payload_remote_path(config, index),
        byte_length=output_identity.size,
        xet_hash=output_xet_hash,
    )
    if expected_payload_claim is not None and expected_payload_claim != expected_output_info:
        raise RemotePercentileError("staged output bytes differ from the immutable upload claim")
    output_info = upload_verified_object(
        backend,
        local_path=output_path,
        remote_path=_payload_remote_path(config, index),
        timeout_seconds=config.retry_timeout_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
        expected_local=expected_output_info,
    )
    if _safe_regular_identity(output_path, surface="staged output after upload") != output_identity:
        raise RemotePercentileError(f"staged output changed during upload: {output_path}")
    receipt = RemoteReceipt(
        contract_sha256=contract_sha256,
        shard_index=index,
        source=source,
        output=output_info,
        semantic_content_sha256=semantic_digest,
        n_values=n_values,
    )
    published = _publish_receipt(
        backend,
        config=config,
        local_path=config.staging_dir / f"receipt_shard_{index:05d}.json",
        receipt=receipt,
        expected_claim=expected_receipt_claim,
        assert_staging=assert_staging,
    )
    return _PublishedFiles(
        receipt=published.receipt,
        receipt_identity=published.receipt_identity,
        output_identity=output_identity,
    )


def _statistics_checksum(path: Path) -> tuple[str, int]:
    statistics = load_statistics(path)
    table = build_percentile_table(
        statistics.per_feature_stats,
        statistics.metadata.n_sae_features,
    )
    return compute_percentile_checksum(table), statistics.metadata.n_sae_features


def _validate_statistics_copy(
    path: Path,
    *,
    checksum: str,
    n_sae_features: int,
) -> tuple[bytes, RegularFileIdentity, str]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemotePercentileError("stable statistics reads require os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RemotePercentileError(f"statistics copy {path} is unreadable") from exc
    try:
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise RemotePercentileError(
                    f"statistics copy {path} is not a direct single-link regular file"
                )
            identity = RegularFileIdentity.from_stat(opened)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            after_read = os.fstat(descriptor)
            if (
                after_read.st_nlink != 1
                or RegularFileIdentity.from_stat(after_read) != identity
                or len(payload) != identity.size
            ):
                raise RemotePercentileError(f"statistics copy {path} changed during read")
            statistics, stored_checksum, _ = load_statistics_and_checksum_from_bytes(payload)
            table = build_percentile_table(
                statistics.per_feature_stats,
                statistics.metadata.n_sae_features,
            )
            observed_checksum = compute_percentile_checksum(table)
            observed_width = statistics.metadata.n_sae_features
            if observed_checksum != checksum or observed_width != n_sae_features:
                raise RemotePercentileError(
                    f"statistics copy {path} does not match the percentile table contract"
                )
            if stored_checksum != checksum:
                raise RemotePercentileError(
                    f"statistics copy {path} lacks the contracted percentile checksum"
                )
            xet_result = cast(Any, hf_xet).hash_files([f"/proc/self/fd/{descriptor}"])[0]
            after_xet = os.fstat(descriptor)
            linked = path.lstat()
            if (
                after_xet.st_nlink != 1
                or RegularFileIdentity.from_stat(after_xet) != identity
                or linked.st_nlink != 1
                or RegularFileIdentity.from_stat(linked) != identity
                or int(xet_result.file_size) != identity.size
            ):
                raise RemotePercentileError(
                    f"statistics copy {path} changed during stable validation"
                )
            return payload, identity, str(xet_result.hash)
        except RemotePercentileError:
            raise
        except Exception as exc:
            raise RemotePercentileError(f"statistics copy {path} is unreadable") from exc
    finally:
        os.close(descriptor)


def _prepare_statistics_copy(
    *,
    stats_path: Path,
    staged_path: Path,
    checksum: str,
    n_sae_features: int,
    staging: PercentileStaging,
) -> tuple[bytes, RegularFileIdentity, str]:
    if staged_path.exists() or staged_path.is_symlink():
        staged_identity = _safe_regular_identity(
            staged_path,
            surface="staged statistics copy",
        )
        try:
            validated = _validate_statistics_copy(
                staged_path,
                checksum=checksum,
                n_sae_features=n_sae_features,
            )
        except Exception:
            staging._safe_unlink(staged_path, expected=staged_identity)
        else:
            return validated
    temporary = staged_path.parent / (
        f".{staged_path.name}.remote-tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )
    staging.assert_bound()
    copy_error: BaseException | None = None
    temporary_identity: RegularFileIdentity | None = None
    try:
        shutil.copyfile(stats_path, temporary)
        temporary_identity = _safe_regular_identity(
            temporary,
            surface="temporary statistics copy",
        )
        os.replace(temporary, staged_path)
    except BaseException as exc:
        copy_error = exc
        raise
    finally:
        if temporary.exists() or temporary.is_symlink():
            try:
                observed_temporary = _safe_regular_identity(
                    temporary,
                    surface="temporary statistics cleanup",
                )
                staging._safe_unlink(
                    temporary,
                    expected=(
                        temporary_identity if temporary_identity is not None else observed_temporary
                    ),
                )
            except BaseException as cleanup_error:
                if copy_error is None:
                    raise
                copy_error.add_note(f"temporary statistics cleanup also failed: {cleanup_error!r}")
    staging.assert_bound()
    save_percentile_checksum(staged_path, checksum)
    return _validate_statistics_copy(
        staged_path,
        checksum=checksum,
        n_sae_features=n_sae_features,
    )


def materialize_overlay_statistics(
    *,
    overlay_statistics_path: Path,
    feature_root_statistics_path: Path,
    percentile_checksum: str,
    expected_statistics_object: RemoteObjectInfo | None = None,
) -> None:
    """Materialize pulled overlay statistics without dropping root metadata."""
    statistics = load_statistics(overlay_statistics_path)
    overlay_width = statistics.metadata.n_sae_features
    overlay_payload, overlay_identity, overlay_xet_hash = _validate_statistics_copy(
        overlay_statistics_path,
        checksum=percentile_checksum,
        n_sae_features=overlay_width,
    )
    if expected_statistics_object is not None and (
        overlay_identity.size != expected_statistics_object.byte_length
        or overlay_xet_hash != expected_statistics_object.xet_hash
    ):
        raise RemotePercentileError("overlay statistics bytes differ from their manifest identity")
    if not feature_root_statistics_path.exists():
        feature_root_statistics_path.parent.mkdir(parents=True, exist_ok=True)
        staged_root = feature_root_statistics_path.parent / (
            f".{feature_root_statistics_path.name}.materialize.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        staged_identity: RegularFileIdentity | None = None
        published = False
        try:
            atomic_write_bytes(overlay_payload, staged_root)
            _, staged_identity, _ = _validate_statistics_copy(
                staged_root,
                checksum=percentile_checksum,
                n_sae_features=overlay_width,
            )
            os.link(staged_root, feature_root_statistics_path, follow_symlinks=False)
            linked = feature_root_statistics_path.lstat()
            if (linked.st_dev, linked.st_ino) != (
                staged_identity.device,
                staged_identity.inode,
            ) or linked.st_size != staged_identity.size:
                raise RemotePercentileError(
                    "new root statistics path changed during atomic publication"
                )
            published = True
        except BaseException as exc:
            root_cleanup_error: BaseException | None = None
            if not published and staged_identity is not None:
                try:
                    linked = feature_root_statistics_path.lstat()
                    if (linked.st_dev, linked.st_ino) == (
                        staged_identity.device,
                        staged_identity.inode,
                    ):
                        feature_root_statistics_path.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as cleanup_error:
                    root_cleanup_error = cleanup_error
            if isinstance(exc, Exception):
                install_error = RemotePercentileError(
                    "validated overlay statistics could not be installed"
                )
                if root_cleanup_error is not None:
                    install_error.add_note(
                        f"root statistics cleanup also failed: {root_cleanup_error!r}"
                    )
                raise install_error from exc
            if root_cleanup_error is not None and root_cleanup_error is not exc:
                exc.add_note(f"root statistics cleanup also failed: {root_cleanup_error!r}")
            raise
        finally:
            active_error = sys.exception()
            if staged_root.exists() or staged_root.is_symlink():
                try:
                    observed = staged_root.lstat()
                    if staged_identity is None or (
                        stat.S_ISREG(observed.st_mode)
                        and (observed.st_dev, observed.st_ino)
                        == (staged_identity.device, staged_identity.inode)
                        and observed.st_size == staged_identity.size
                    ):
                        staged_root.unlink()
                except BaseException as cleanup_error:
                    if active_error is None:
                        raise
                    if cleanup_error is not active_error:  # pragma: no branch
                        active_error.add_note(
                            f"temporary root statistics cleanup also failed: {cleanup_error!r}"
                        )
        _validate_statistics_copy(
            feature_root_statistics_path,
            checksum=percentile_checksum,
            n_sae_features=overlay_width,
        )
        return
    root_checksum, root_width = _statistics_checksum(feature_root_statistics_path)
    if root_checksum != percentile_checksum or root_width != overlay_width:
        raise RemotePercentileError("existing root statistics table differs from overlay")
    save_percentile_checksum(feature_root_statistics_path, percentile_checksum)


def _update_local_statistics(
    stats_path: Path,
    *,
    checksum: str,
    n_sae_features: int,
) -> None:
    current_checksum, current_width = _statistics_checksum(stats_path)
    if current_checksum != checksum or current_width != n_sae_features:
        raise RemotePercentileError(
            "local statistics percentile table changed during the remote run"
        )
    save_percentile_checksum(stats_path, checksum)


def _write_profile(profile: RemotePercentileProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(profile.payload(), path)


def _fast_complete(
    backend: BucketBackend,
    *,
    state: RemoteOutputState,
    config: RemotePercentileConfig,
    stats_path: Path,
    checksum: str,
    n_sae_features: int,
    inventory: Sequence[RemoteFeatureShard],
    staging: PercentileStaging,
) -> int:
    current_inventory = _validate_source_family(
        normalize_remote_feature_inventory(
            _list_with_retry(
                backend,
                str(config.source_prefix),
                config=config,
                surface="complete-fast source fence",
            ),
            prefix=config.source_prefix,
        ),
        expected_count=config.expected_shard_count,
    )
    if tuple(current_inventory) != tuple(inventory):
        raise RemotePercentileError(
            "remote source inventory changed during complete-family validation"
        )
    _revalidate_remote_identities(backend, state, config=config)
    statistics_info = cast(RemoteObjectInfo, state.statistics_info)
    staged_statistics = config.staging_dir / OVERLAY_STATISTICS_FILENAME
    download_verified_object(
        backend,
        expected=statistics_info,
        final_path=staged_statistics,
        timeout_seconds=config.retry_timeout_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
    )
    staging.assert_bound()
    _validate_statistics_copy(
        staged_statistics,
        checksum=checksum,
        n_sae_features=n_sae_features,
    )
    _update_local_statistics(
        stats_path,
        checksum=checksum,
        n_sae_features=n_sae_features,
    )
    logger.info(
        "Remote percentile overlay is already complete; repaired/validated local "
        "statistics without downloading raw shards"
    )
    return 0


def run_remote_percentiles(
    cfg: DictConfig,
    feature_dir: Path,
    *,
    backend: BucketBackend | None = None,
) -> int:
    """Run the remote percentile producer from an already-resolved feature dir."""
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        logger.error(f"No feature_statistics.safetensors found in {feature_dir}")
        return 1
    statistics = load_statistics(stats_path)
    n_sae_features = statistics.metadata.n_sae_features
    percentile_table = build_percentile_table(
        statistics.per_feature_stats,
        n_sae_features,
    )
    checksum = compute_percentile_checksum(percentile_table)
    config = _resolve_config(cfg, feature_dir=feature_dir, checksum=checksum)
    resolved_backend = backend or HfBucketBackend(config.bucket_id)

    source_listing = _list_with_retry(
        resolved_backend,
        str(config.source_prefix),
        config=config,
        surface="startup source inventory",
    )
    inventory = _validate_source_family(
        normalize_remote_feature_inventory(
            source_listing,
            prefix=config.source_prefix,
        ),
        expected_count=config.expected_shard_count,
    )
    inventory_sha256 = remote_feature_inventory_digest(inventory)
    desired_contract = _producer_contract(
        config,
        inventory_sha256=inventory_sha256,
        percentile_checksum=checksum,
        n_sae_features=n_sae_features,
    )
    contract_sha256 = _sha256_json(desired_contract)
    required_capacity = required_staging_capacity_bytes(
        inventory,
        max_output_shard_bytes=config.max_output_shard_bytes,
        statistics_bytes=stats_path.stat().st_size,
    )
    profile = RemotePercentileProfile(
        config={
            "bucket_id": str(config.bucket_id),
            "source_prefix": str(config.source_prefix),
            "output_prefix": str(config.output_prefix),
            "device": config.device.type,
            "resume": config.resume,
            "zstd_level": config.zstd_level,
            "max_decompressed_shard_bytes": config.max_decompressed_shard_bytes,
        },
        inventory={
            "count": len(inventory),
            "bytes": sum(entry.info.byte_length for entry in inventory),
            "sha256": inventory_sha256,
        },
        staging={"required_bytes": required_capacity},
    )
    logger.info(
        "Remote percentile preflight: "
        f"bucket={config.bucket_id} source={config.source_prefix} "
        f"output={config.output_prefix} count={len(inventory):,} "
        f"bytes={sum(entry.info.byte_length for entry in inventory):,} "
        f"inventory_sha256={inventory_sha256} contract_sha256={contract_sha256} "
        f"device={config.device.type} staging_required={required_capacity:,}"
    )

    output_state = _inspect_output_state(
        resolved_backend,
        config=config,
        desired_contract=desired_contract,
        contract_sha256=contract_sha256,
        inventory=inventory,
        checksum=checksum,
        download_controls=not config.dry_run,
    )
    decompression_lookahead = _decompression_lookahead_enabled(config, inventory)
    profile.config["decompression_lookahead_enabled"] = decompression_lookahead
    lookahead_memory: tuple[int, int] | None = None
    if decompression_lookahead and config.dry_run:
        assert config.max_decompressed_shard_bytes is not None
        lookahead_memory = _require_decompression_lookahead_memory(
            inventory,
            max_decompressed_shard_bytes=config.max_decompressed_shard_bytes,
        )
        profile.staging.update(
            {
                "lookahead_memory_required_bytes": lookahead_memory[0],
                "lookahead_memory_available_bytes": lookahead_memory[1],
            }
        )
    staging = PercentileStaging(
        config=config,
        contract_sha256=contract_sha256,
        required_capacity=required_capacity,
        inventory=inventory,
        output_state=output_state,
    )
    if config.dry_run:
        if (
            output_state.has_canonical_state
            and not output_state.structurally_complete(config.expected_shard_count)
            and not config.resume
        ):
            raise RemotePercentileError(
                "partial remote percentile state exists; rerun with resume=true"
            )
        free, recoverable = staging.preflight_without_mutation()
        logger.info(
            "Remote percentile dry-run passed: "
            f"output_objects={len(output_state.listing)} free={free:,} "
            f"recoverable={recoverable:,}; no downloads or writes performed"
        )
        return 0

    if output_state.complete(config.expected_shard_count):
        fast_started = time.monotonic()
        fast_error: BaseException | None = None
        try:
            staging.open(require_capacity=False)
            result = _fast_complete(
                resolved_backend,
                state=output_state,
                config=config,
                stats_path=stats_path,
                checksum=checksum,
                n_sae_features=n_sae_features,
                inventory=inventory,
                staging=staging,
            )
            staging.cleanup_success()
        except BaseException as exc:
            fast_error = exc
            raise
        finally:
            try:
                staging.close()
            except BaseException as close_error:
                if fast_error is None:
                    raise
                fast_error.add_note(
                    f"remote percentile fast-path staging close also failed: {close_error!r}"
                )
        profile.staging.update(
            {
                "high_water_bytes": staging.high_water_bytes,
                "high_water_bodies": staging.high_water_bodies,
            }
        )
        profile.totals = {
            "wall_seconds": time.monotonic() - fast_started,
            "processed_shards": 0,
            "status": "complete",
            "complete_fast_path": True,
        }
        if config.profile and config.profile_path is not None:
            _write_profile(profile, config.profile_path)
        return result
    if output_state.has_canonical_state and not config.resume:
        raise RemotePercentileError(
            "partial remote percentile state exists; rerun with resume=true"
        )
    if decompression_lookahead:
        assert config.max_decompressed_shard_bytes is not None
        lookahead_memory = _require_decompression_lookahead_memory(
            inventory,
            max_decompressed_shard_bytes=config.max_decompressed_shard_bytes,
        )
        profile.staging.update(
            {
                "lookahead_memory_required_bytes": lookahead_memory[0],
                "lookahead_memory_available_bytes": lookahead_memory[1],
            }
        )
        logger.info(
            "Remote percentile CUDA decompression lookahead enabled: "
            f"one plain source, max_plain={config.max_decompressed_shard_bytes:,}, "
            f"memory_required={lookahead_memory[0]:,}, "
            f"memory_available={lookahead_memory[1]:,}"
        )

    started = time.monotonic()
    cancellation = threading.Event()
    download_executor: ThreadPoolExecutor | None = None
    upload_executor: ThreadPoolExecutor | None = None
    pending_download: Future[tuple[Path, RegularFileIdentity, str]] | None = None
    pending_preparation: Future[_PreparedPercentileSource] | None = None
    pending_upload: _PendingUpload | None = None
    success = False
    pipeline_error: BaseException | None = None
    try:
        staging.open()
        staging.assert_bound()
        profile.staging.update(
            {
                "observed_free_bytes": staging.admission_free_bytes,
                "recoverable_bytes": staging.admission_recoverable_bytes,
            }
        )
        contract_local = config.staging_dir / REMOTE_CONTRACT_FILENAME
        contract_payload = _canonical_json_bytes(desired_contract)
        atomic_write_bytes(contract_payload, contract_local)
        staging.assert_bound()
        upload_verified_object(
            resolved_backend,
            local_path=contract_local,
            remote_path=_join_remote(config.output_prefix, REMOTE_CONTRACT_FILENAME),
            timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
            expected_local=output_state.claimed_objects.get(
                _join_remote(config.output_prefix, REMOTE_CONTRACT_FILENAME)
            ),
            expected_bytes=contract_payload,
        )

        download_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="percentile-remote-download",
        )
        upload_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="percentile-remote-upload",
        )

        base_hasher = BaseShardContentHasher()

        def schedule_source(index: int) -> Future[tuple[Path, RegularFileIdentity, str]]:
            return download_executor.submit(
                _download_source,
                resolved_backend,
                config=config,
                entry=inventory[index],
                cancellation=cancellation,
                assert_staging=staging.assert_bound,
                unlink_staged=lambda path, identity: staging._safe_unlink(
                    path,
                    expected=identity,
                ),
            )

        def schedule_preparation(index: int) -> Future[_PreparedPercentileSource]:
            return download_executor.submit(
                _prepare_streamed_percentile_source,
                resolved_backend,
                config=config,
                entry=inventory[index],
                cancellation=cancellation,
                assert_staging=staging.assert_bound,
                unlink_staged=lambda path, identity: staging._safe_unlink(
                    path,
                    expected=identity,
                ),
                base_hasher=base_hasher,
                decompress=index not in output_state.receipts,
            )

        if decompression_lookahead:
            pending_preparation = schedule_preparation(0)
        else:
            pending_download = schedule_source(0)
        for position, entry in enumerate(inventory):
            shard_started = time.monotonic()
            prepared: _PreparedPercentileSource | None = None
            compressed: bytes | None = None
            if decompression_lookahead:
                assert pending_preparation is not None
                prepared = pending_preparation.result()
                pending_preparation = None
                if position + 1 < len(inventory):
                    pending_preparation = schedule_preparation(position + 1)
                disposition = prepared.disposition
            else:
                assert pending_download is not None
                source_path, source_identity, disposition = pending_download.result()
                pending_download = None
                if position + 1 < len(inventory):
                    pending_download = schedule_source(position + 1)

                compressed = read_feature_compressed_bytes_stable(
                    source_path,
                    expected_identity=source_identity,
                )
                base_hasher.update(PurePosixPath(entry.info.path).name, compressed)
                if (
                    _safe_regular_identity(source_path, surface="consumed source body")
                    != source_identity
                ):
                    raise RemotePercentileError(
                        f"source body changed before cleanup: {source_path}"
                    )
                staging._safe_unlink(source_path, expected=source_identity)

            record: dict[str, object] = {
                "index": entry.index,
                "source_path": entry.info.path,
                "source_bytes": entry.info.byte_length,
                "source_disposition": disposition,
                "computed": False,
                "upload_scheduled": False,
                "upload_verified": False,
                "orphan_adopted": False,
            }
            existing_receipt = output_state.receipts.get(entry.index)
            if existing_receipt is not None:
                record["n_values"] = existing_receipt.n_values
                record["elapsed_seconds"] = time.monotonic() - shard_started
                profile.per_shard.append(record)
                staging.observe()
                del compressed, prepared
                continue

            if prepared is not None:
                values = _compute_streamed_percentile_values_from_safetensors(
                    prepared.take_safetensors_bytes(),
                    entry=entry,
                    config=config,
                    percentile_table=percentile_table,
                )
                del prepared
            else:
                assert compressed is not None
                values = _compute_streamed_percentile_values(
                    compressed,
                    entry=entry,
                    config=config,
                    percentile_table=percentile_table,
                )
                del compressed
            tensors = {TensorKey.FEATURE_PERCENTILES.value: values}
            semantic_digest = overlay_tensors_content_digest(tensors)
            n_values = values.numel()
            record["computed"] = True
            record["n_values"] = n_values

            if pending_upload is not None:
                pending_upload.finish(staging)
                pending_upload = None

            orphan = output_state.payloads.get(entry.index)
            if orphan is not None:
                if orphan.byte_length > config.max_output_shard_bytes:
                    raise RemotePercentileError(
                        f"orphan overlay shard {entry.index} is {orphan.byte_length:,} "
                        "bytes, exceeding remote.max_output_shard_bytes="
                        f"{config.max_output_shard_bytes:,}"
                    )
                verify_path = config.staging_dir / f"verify_shard_{entry.index:05d}{COMPRESSED_EXT}"
                orphan_future = download_executor.submit(
                    download_verified_object,
                    resolved_backend,
                    expected=orphan,
                    final_path=verify_path,
                    timeout_seconds=config.retry_timeout_seconds,
                    retry_interval_seconds=config.retry_interval_seconds,
                )
                orphan_future.result()
                verify_identity = _safe_regular_identity(
                    verify_path,
                    surface="downloaded orphan overlay",
                )
                observed_values, observed_digest, verify_identity, _ = (
                    _read_percentile_overlay_stable(
                        verify_path,
                        expected_numel=n_values,
                        expected_identity=verify_identity,
                    )
                )
                if not t.equal(observed_values, values):
                    raise RemotePercentileError(
                        f"orphan remote payload differs from recomputation: {orphan.path}"
                    )
                if observed_digest != semantic_digest:
                    raise RemotePercentileError(
                        f"orphan remote payload semantic digest differs: {orphan.path}"
                    )
                del observed_values
                staging._safe_unlink(verify_path, expected=verify_identity)
                receipt = RemoteReceipt(
                    contract_sha256=contract_sha256,
                    shard_index=entry.index,
                    source=entry.info,
                    output=orphan,
                    semantic_content_sha256=semantic_digest,
                    n_values=n_values,
                )
                receipt_path = config.staging_dir / f"receipt_shard_{entry.index:05d}.json"
                pending_upload = _PendingUpload(
                    future=upload_executor.submit(
                        _publish_receipt,
                        resolved_backend,
                        config=config,
                        local_path=receipt_path,
                        receipt=receipt,
                        expected_claim=output_state.claimed_objects.get(
                            _receipt_remote_path(config, entry.index)
                        ),
                        assert_staging=staging.assert_bound,
                    ),
                    receipt_path=receipt_path,
                    profile_record=record,
                )
                record["upload_scheduled"] = True
                record["orphan_adopted"] = True
            else:
                output_path = config.staging_dir / f"output_shard_{entry.index:05d}{COMPRESSED_EXT}"
                retained_output_is_exact = False
                output_identity: RegularFileIdentity | None = None
                output_xet_hash: str | None = None
                if output_path.exists() or output_path.is_symlink():
                    retained_identity = _safe_regular_identity(
                        output_path,
                        surface="retained output body",
                    )
                    if retained_identity.size > config.max_output_shard_bytes:
                        raise RemotePercentileError(
                            f"retained staged overlay shard {entry.index} is "
                            f"{retained_identity.size:,} bytes, exceeding "
                            "remote.max_output_shard_bytes="
                            f"{config.max_output_shard_bytes:,}"
                        )
                    (
                        retained_values,
                        retained_digest,
                        output_identity,
                        output_xet_hash,
                    ) = _read_percentile_overlay_stable(
                        output_path,
                        expected_numel=n_values,
                        expected_identity=retained_identity,
                    )
                    retained_output_is_exact = t.equal(retained_values, values)
                    if not retained_output_is_exact:
                        raise RemotePercentileError(
                            f"retained staged output differs from recomputation: {output_path}"
                        )
                    if retained_digest != semantic_digest:
                        raise RemotePercentileError(
                            f"retained staged output digest differs: {output_path}"
                        )
                    del retained_values
                if not retained_output_is_exact:
                    staging.assert_bound()
                    save_overlay_shard(
                        tensors,
                        output_path,
                        zstd_level=config.zstd_level,
                    )
                    staging.assert_bound()
                    saved_values, saved_digest, output_identity, output_xet_hash = (
                        _read_percentile_overlay_stable(
                            output_path,
                            expected_numel=n_values,
                        )
                    )
                    if not t.equal(saved_values, values):
                        raise RemotePercentileError(
                            f"saved staged output differs from computation: {output_path}"
                        )
                    if saved_digest != semantic_digest:
                        raise RemotePercentileError(
                            f"saved staged output digest differs: {output_path}"
                        )
                    del saved_values
                assert output_identity is not None
                assert output_xet_hash is not None
                output_bytes = output_identity.size
                if output_bytes > config.max_output_shard_bytes:
                    raise RemotePercentileError(
                        f"saved overlay shard {entry.index} is {output_bytes:,} bytes, "
                        f"exceeding remote.max_output_shard_bytes="
                        f"{config.max_output_shard_bytes:,}"
                    )
                receipt_path = config.staging_dir / f"receipt_shard_{entry.index:05d}.json"
                pending_upload = _PendingUpload(
                    future=upload_executor.submit(
                        _publish_payload_and_receipt,
                        resolved_backend,
                        config=config,
                        contract_sha256=contract_sha256,
                        source=entry.info,
                        index=entry.index,
                        output_path=output_path,
                        semantic_digest=semantic_digest,
                        n_values=n_values,
                        output_identity=output_identity,
                        output_xet_hash=output_xet_hash,
                        expected_payload_claim=output_state.claimed_objects.get(
                            _payload_remote_path(config, entry.index)
                        ),
                        expected_receipt_claim=output_state.claimed_objects.get(
                            _receipt_remote_path(config, entry.index)
                        ),
                        assert_staging=staging.assert_bound,
                    ),
                    output_path=output_path,
                    receipt_path=receipt_path,
                    profile_record=record,
                )
                record["upload_scheduled"] = True
            del tensors, values
            record["elapsed_seconds"] = time.monotonic() - shard_started
            profile.per_shard.append(record)
            staging.observe()

        if pending_upload is not None:
            pending_upload.finish(staging)
            pending_upload = None

        final_source_listing = _list_with_retry(
            resolved_backend,
            str(config.source_prefix),
            config=config,
            surface="post-processing source fence",
        )
        final_inventory = _validate_source_family(
            normalize_remote_feature_inventory(
                final_source_listing,
                prefix=config.source_prefix,
            ),
            expected_count=config.expected_shard_count,
        )
        if tuple(final_inventory) != tuple(inventory):
            raise RemotePercentileError(
                "remote source inventory changed during percentile production; "
                "manifest was not published"
            )

        coverage_state = _inspect_output_state(
            resolved_backend,
            config=config,
            desired_contract=desired_contract,
            contract_sha256=contract_sha256,
            inventory=inventory,
            checksum=checksum,
            download_controls=True,
        )
        expected_indices = set(range(config.expected_shard_count))
        if (
            set(coverage_state.payloads) != expected_indices
            or set(coverage_state.receipts) != expected_indices
        ):
            raise RemotePercentileError("remote payload/receipt coverage is incomplete")

        current_checksum, current_width = _statistics_checksum(stats_path)
        if current_checksum != checksum or current_width != n_sae_features:
            raise RemotePercentileError(
                "local statistics percentile table changed during the remote run"
            )
        staged_stats = config.staging_dir / OVERLAY_STATISTICS_FILENAME
        staging.assert_bound()
        if coverage_state.statistics_info is not None:
            adopt_remote_statistics = True
            if staged_stats.exists() or staged_stats.is_symlink():
                staged_stats_identity = _safe_regular_identity(
                    staged_stats,
                    surface="retained overlay statistics",
                )
                try:
                    staged_info = verified_local_object_info(
                        local_path=staged_stats,
                        remote_path=coverage_state.statistics_info.path,
                    )
                except Exception:
                    staging._safe_unlink(
                        staged_stats,
                        expected=staged_stats_identity,
                    )
                else:
                    if staged_info == coverage_state.statistics_info:
                        adopt_remote_statistics = False
                    else:
                        staging._safe_unlink(
                            staged_stats,
                            expected=staged_stats_identity,
                        )
            if adopt_remote_statistics:
                statistics_future = download_executor.submit(
                    download_verified_object,
                    resolved_backend,
                    expected=coverage_state.statistics_info,
                    final_path=staged_stats,
                    timeout_seconds=config.retry_timeout_seconds,
                    retry_interval_seconds=config.retry_interval_seconds,
                )
                statistics_future.result()
                staging.assert_bound()
            _, staged_stats_identity, staged_stats_xet = _validate_statistics_copy(
                staged_stats,
                checksum=checksum,
                n_sae_features=n_sae_features,
            )
        else:
            _, staged_stats_identity, staged_stats_xet = _prepare_statistics_copy(
                stats_path=stats_path,
                staged_path=staged_stats,
                checksum=checksum,
                n_sae_features=n_sae_features,
                staging=staging,
            )
            staging.assert_bound()
        statistics_remote_path = _join_remote(config.output_prefix, OVERLAY_STATISTICS_FILENAME)
        expected_statistics_info = RemoteObjectInfo(
            path=statistics_remote_path,
            byte_length=staged_stats_identity.size,
            xet_hash=staged_stats_xet,
        )
        claimed_statistics = coverage_state.claimed_objects.get(statistics_remote_path)
        if claimed_statistics is not None and claimed_statistics != expected_statistics_info:
            raise RemotePercentileError(
                "staged statistics bytes differ from the immutable upload claim"
            )
        statistics_info = upload_verified_object(
            resolved_backend,
            local_path=staged_stats,
            remote_path=statistics_remote_path,
            timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
            expected_local=expected_statistics_info,
        )

        current_checksum, current_width = _statistics_checksum(stats_path)
        if current_checksum != checksum or current_width != n_sae_features:
            raise RemotePercentileError(
                "local statistics percentile table changed before manifest publication"
            )

        premanifest_inventory = _validate_source_family(
            normalize_remote_feature_inventory(
                _list_with_retry(
                    resolved_backend,
                    str(config.source_prefix),
                    config=config,
                    surface="pre-manifest source fence",
                ),
                prefix=config.source_prefix,
            ),
            expected_count=config.expected_shard_count,
        )
        if tuple(premanifest_inventory) != tuple(inventory):
            raise RemotePercentileError(
                "remote source inventory changed before manifest publication"
            )

        manifest_stage_dir = config.staging_dir
        staging.assert_bound()
        manifest_metadata = _manifest_metadata(
            contract=desired_contract,
            contract_sha256=contract_sha256,
            statistics_info=statistics_info,
        )
        base_shard_content_hash = base_hasher.hexdigest()
        manifest_path = write_manifest(
            manifest_stage_dir,
            checksum,
            OverlayKind.PERCENTILE,
            metadata=manifest_metadata,
            precomputed_base_hash=base_shard_content_hash,
            include_created_at=False,
        )
        manifest_payload = (
            json.dumps(
                {
                    "checksum": checksum,
                    "kind": OverlayKind.PERCENTILE.value,
                    "checksum_prefix": checksum[:CHECKSUM_PREFIX_LEN],
                    "metadata": manifest_metadata,
                    "base_shard_content_hash": base_shard_content_hash,
                    "schema_version": 2,
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        staging.assert_bound()
        manifest_remote_path = _join_remote(config.output_prefix, MANIFEST_FILENAME)
        upload_verified_object(
            resolved_backend,
            local_path=manifest_path,
            remote_path=manifest_remote_path,
            timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
            expected_local=coverage_state.claimed_objects.get(manifest_remote_path),
            expected_bytes=manifest_payload,
        )

        completed_state = _inspect_output_state(
            resolved_backend,
            config=config,
            desired_contract=desired_contract,
            contract_sha256=contract_sha256,
            inventory=inventory,
            checksum=checksum,
            download_controls=True,
        )
        if not completed_state.complete(config.expected_shard_count):
            raise RemotePercentileError("remote percentile family failed final completion fence")
        if (
            completed_state.manifest is None
            or completed_state.manifest.get("base_shard_content_hash") != base_hasher.hexdigest()
        ):
            raise RemotePercentileError(
                "published remote manifest base hash differs from streamed source bytes"
            )
        post_manifest_inventory = _validate_source_family(
            normalize_remote_feature_inventory(
                _list_with_retry(
                    resolved_backend,
                    str(config.source_prefix),
                    config=config,
                    surface="post-manifest source diagnostic fence",
                ),
                prefix=config.source_prefix,
            ),
            expected_count=config.expected_shard_count,
        )
        if tuple(post_manifest_inventory) != tuple(inventory):
            raise RemotePercentileError(
                "remote source inventory changed during manifest publication; "
                "do not consume this family"
            )
        _revalidate_remote_identities(
            resolved_backend,
            completed_state,
            config=config,
        )
        _update_local_statistics(
            stats_path,
            checksum=checksum,
            n_sae_features=n_sae_features,
        )
        staging.cleanup_success()
        success = True
        logger.info(
            f"Remote percentile overlay committed at {config.output_prefix}; "
            f"checksum={checksum}, base_hash={base_hasher.hexdigest()}"
        )
        return 0
    except BaseException as exc:
        pipeline_error = exc
        if isinstance(exc, RemoteTrainingError) and (
            "verified-upload" in str(exc) or "collision" in str(exc)
        ):
            exc.add_note(
                "A claim-only immutable object may require restoring the exact retained "
                "staged bytes or publishing to a new empty percentile family; never delete "
                "or overwrite the conflicting canonical object."
            )
        raise
    finally:
        cleanup_errors: list[BaseException] = []

        def cleanup(action: Callable[[], object]) -> None:
            try:
                action()
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)

        cancellation.set()
        if pending_download is not None:
            pending_download.cancel()
        if pending_preparation is not None:
            pending_preparation.cancel()
        if pending_upload is not None:
            pending_upload.future.cancel()
        if download_executor is not None:
            cleanup(lambda: download_executor.shutdown(wait=True, cancel_futures=True))
        if upload_executor is not None:
            cleanup(lambda: upload_executor.shutdown(wait=True, cancel_futures=True))
        for label, future in (
            ("download", pending_download),
            ("preparation", pending_preparation),
            ("upload", None if pending_upload is None else pending_upload.future),
        ):
            if future is None or not future.done() or future.cancelled():
                continue
            try:
                future_error = future.exception()
            except BaseException as future_exc:
                cleanup_errors.append(future_exc)
            else:
                if future_error is not None and future_error is not pipeline_error:
                    future_error.add_note(f"background {label} future failed during teardown")
                    cleanup_errors.append(future_error)
        cleanup(staging.observe)
        profile.staging.update(
            {
                "high_water_bytes": staging.high_water_bytes,
                "high_water_bodies": staging.high_water_bodies,
            }
        )
        profile.totals = {
            "wall_seconds": time.monotonic() - started,
            "processed_shards": len(profile.per_shard),
            "status": "complete" if success else "failed",
        }
        cleanup(staging.close)
        if config.profile and config.profile_path is not None:
            profile_path = config.profile_path
            cleanup(lambda: _write_profile(profile, profile_path))
        if pipeline_error is not None:
            for cleanup_error in cleanup_errors:
                if cleanup_error is not pipeline_error:
                    pipeline_error.add_note(
                        f"remote percentile teardown also failed: {cleanup_error!r}"
                    )
        elif cleanup_errors:
            primary_cleanup_error = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                if cleanup_error is not primary_cleanup_error:
                    primary_cleanup_error.add_note(
                        f"additional remote percentile teardown failure: {cleanup_error!r}"
                    )
            raise primary_cleanup_error


__all__ = [
    "CAPACITY_RESERVE_BYTES",
    "DECOMPRESSION_MEMORY_RESERVE_BYTES",
    "OVERLAY_STATISTICS_FILENAME",
    "REMOTE_CONTRACT_FILENAME",
    "REMOTE_RECEIPTS_DIRNAME",
    "RemotePercentileConfig",
    "RemotePercentileError",
    "RemoteReceipt",
    "materialize_overlay_statistics",
    "remote_percentiles_enabled",
    "required_decompression_lookahead_memory_bytes",
    "required_staging_capacity_bytes",
    "run_remote_percentiles",
]
