"""Identity-sealed, depth-one HF transport for exact feature statistics."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from steering.feature_io import (
    COMPRESSED_EXT,
    RegularFileIdentity,
    assert_uniform_ignore_bos_metadata,
    assert_uniform_l0_tier_metadata,
    read_shard_header_stable,
)
from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
)
from steering.remote_training import (
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    RemoteTrainingError,
    download_verified_object,
    verified_local_object_info,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.sae_resolver import L0Tier
from steering.sft.mixture_fingerprint import (
    _uniform_sft_mixture_fingerprint_metadata_json,
    mixture_fingerprint_json,
    parse_mixture_fingerprint_json,
)

_LOCK_FILENAME = ".statistics-remote.lock"
_OWNER_FILENAME = ".statistics-remote-owner.json"
_OWNER_SCHEMA_VERSION = 1
_METADATA_SCHEMA_VERSION = 1
_CAPACITY_RESERVE_BYTES = 1 << 30
_CANONICAL_RE = re.compile(rf"^shard_(?P<index>\d{{5}})_features{re.escape(COMPRESSED_EXT)}$")
_TEMP_RE = re.compile(
    rf"^\.(?P<canonical>shard_\d{{5}}_features{re.escape(COMPRESSED_EXT)})"
    r"\.remote-tmp\.(?P<pid>[1-9][0-9]*)\.(?P<nonce>[0-9a-f]{8})$"
)
_REMOTE_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "l0_tier",
        "ignore_bos_features",
        "sft_mixture_fingerprint_json",
    }
)


class RemoteStatisticsError(RuntimeError):
    """Remote statistics configuration, inventory, or staging is unsafe."""


def _is_cancelled_download(error: BaseException) -> bool:
    return isinstance(error, RemoteTrainingError) and str(error).startswith("download cancelled:")


@dataclass(frozen=True)
class RemoteStatisticsConfig:
    """Fully resolved transport configuration for one statistics run."""

    bucket_id: HfBucketId
    remote_prefix: NormalizedRemotePrefix
    staging_dir: Path
    expected_shard_count: int | None
    retry_timeout_seconds: float
    retry_interval_seconds: float
    expected_segment_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket_id", HfBucketId.parse(str(self.bucket_id)))
        object.__setattr__(
            self,
            "remote_prefix",
            NormalizedRemotePrefix.parse(str(self.remote_prefix)),
        )
        # Preserve the final path component so namespace setup can reject a
        # staging symlink explicitly. Overlap validation resolves this path
        # separately when comparing it with protected locations.
        object.__setattr__(self, "staging_dir", Path(self.staging_dir).absolute())
        expected = self.expected_shard_count
        if expected is not None and (
            isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0
        ):
            raise ValueError(
                f"expected_shard_count must be null or a positive integer, got {expected!r}"
            )
        for name, value in (
            ("retry_timeout_seconds", self.retry_timeout_seconds),
            ("retry_interval_seconds", self.retry_interval_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.expected_segment_size, bool)
            or not isinstance(self.expected_segment_size, int)
            or self.expected_segment_size <= 0
        ):
            raise ValueError(
                "expected_segment_size must be a positive integer, got "
                f"{self.expected_segment_size!r}"
            )


@dataclass(frozen=True)
class PreparedRemoteShard:
    """A verified compressed body ready for stable consumer-side loading."""

    index: int
    logical_path: Path
    local_path: Path
    remote_identity: RemoteObjectInfo
    file_identity: RegularFileIdentity
    disposition: str


@dataclass(frozen=True)
class _TransferResult:
    prepared: PreparedRemoteShard
    transfer_seconds: float
    attempts: int
    materialized_bytes: int


def _inventory_rows(
    inventory: Sequence[RemoteFeatureShard],
) -> list[dict[str, str | int]]:
    return [
        {
            "path": entry.info.path,
            "byte_length": entry.info.byte_length,
            "xet_hash": entry.info.xet_hash,
        }
        for entry in inventory
    ]


def _validate_family(
    inventory: Sequence[RemoteFeatureShard],
    *,
    expected_count: int | None,
    prefix: NormalizedRemotePrefix,
) -> tuple[RemoteFeatureShard, ...]:
    sealed = tuple(inventory)
    if not sealed:
        raise RemoteStatisticsError(f"no direct raw feature shard objects found under {prefix}")
    if expected_count is None:
        expected_indices = list(range(sealed[-1].index + 1))
    else:
        expected_indices = list(range(expected_count))
    observed_indices = [entry.index for entry in sealed]
    if observed_indices != expected_indices:
        expected_set = set(expected_indices)
        observed_set = set(observed_indices)
        raise RemoteStatisticsError(
            "remote statistics inventory is not the exact contiguous family "
            f"0..{len(expected_indices) - 1}: "
            f"missing={sorted(expected_set - observed_set)[:20]}, "
            f"extra={sorted(observed_set - expected_set)[:20]}"
        )
    return sealed


def required_staging_capacity_bytes(
    inventory: Sequence[RemoteFeatureShard],
) -> int:
    """Return the two-largest-bodies admission requirement plus 1 GiB."""
    largest = sorted(
        (entry.info.byte_length for entry in inventory),
        reverse=True,
    )[:2]
    return sum(largest) + _CAPACITY_RESERVE_BYTES


def _capture_safe_regular(path: Path, *, surface: str) -> RegularFileIdentity:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RemoteStatisticsError(f"cannot inspect {surface} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RemoteStatisticsError(f"{surface} must be a direct, single-link regular file: {path}")
    return RegularFileIdentity.from_stat(info)


def _pid_is_live_or_inaccessible(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _DownloadRecordingBackend:
    """Statistics-only delegator that observes each actual download attempt."""

    def __init__(
        self,
        delegate: BucketBackend,
        *,
        observe_staging: Callable[[], None],
    ) -> None:
        self.delegate = delegate
        self.observe_staging = observe_staging
        self.attempts = 0
        self.materialized_bytes = 0

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        return self.delegate.stat(remote_path)

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        return self.delegate.list(prefix)

    def download(self, remote_path: str, local_path: Path) -> None:
        self.attempts += 1
        try:
            self.delegate.download(remote_path, local_path)
        finally:
            try:
                info = local_path.lstat()
            except OSError:
                pass
            else:
                if stat.S_ISREG(info.st_mode):
                    self.materialized_bytes += int(info.st_size)
            self.observe_staging()

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.delegate.upload(local_path, remote_path)


class RemoteStatisticsSource:
    """One-worker, one-lookahead transport bound to a sealed HF inventory."""

    def __init__(
        self,
        *,
        config: RemoteStatisticsConfig,
        backend: BucketBackend,
        full_inventory: tuple[RemoteFeatureShard, ...],
        statistics_inventory: tuple[RemoteFeatureShard, ...],
        excluded_shard: RemoteFeatureShard | None,
        exclude_last_shard: bool,
        profile: Any | None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.full_inventory = full_inventory
        self.statistics_inventory = statistics_inventory
        self.excluded_shard = excluded_shard
        self.exclude_last_shard = exclude_last_shard
        self.full_inventory_sha256 = remote_feature_inventory_digest(full_inventory)
        self.inventory_sha256 = remote_feature_inventory_digest(statistics_inventory)
        self.required_capacity = required_staging_capacity_bytes(statistics_inventory)
        self.profile = profile
        self._started_ns = time.perf_counter_ns()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[_TransferResult] | None = None
        self._future_index: int | None = None
        self._cancel = threading.Event()
        self._lock_fd: int | None = None
        self._staging_initialized = False
        self._finish_done = False
        self._preserve_paths: set[Path] = set()
        self._metadata: dict[str, object] | None = None
        self._profile_records: dict[int, dict[str, Any]] = {}
        self._profile_lock = threading.Lock()
        self._bootstrap_index: int | None = None
        self._staging_high_water_bytes = 0
        self._staging_high_water_bodies = 0
        self._verified_downloaded_bytes = 0
        self._verified_reused_bytes = 0
        self._materialized_bytes = 0
        self._critical_wait_seconds = 0.0
        self._cleanup_join_seconds = 0.0
        self._admission_free_bytes: int | None = None
        self._admission_recoverable_bytes: int | None = None
        if profile is not None:
            profile.remote_transport = self._profile_payload(status="running")

    @classmethod
    def snapshot(
        cls,
        config: RemoteStatisticsConfig,
        *,
        exclude_last_shard: bool,
        profile: Any | None = None,
        backend: BucketBackend | None = None,
    ) -> RemoteStatisticsSource:
        """List and seal the complete family without touching staging."""
        resolved_backend = backend if backend is not None else HfBucketBackend(config.bucket_id)
        listing = resolved_backend.list(str(config.remote_prefix))
        normalized = normalize_remote_feature_inventory(
            listing,
            prefix=config.remote_prefix,
        )
        full = _validate_family(
            normalized,
            expected_count=config.expected_shard_count,
            prefix=config.remote_prefix,
        )
        excluded = full[-1] if exclude_last_shard and len(full) > 1 else None
        domain = full[:-1] if excluded is not None else full
        source = cls(
            config=config,
            backend=resolved_backend,
            full_inventory=full,
            statistics_inventory=domain,
            excluded_shard=excluded,
            exclude_last_shard=exclude_last_shard,
            profile=profile,
        )
        if config.expected_shard_count is None:
            logger.warning(
                "REMOTE STATISTICS EXPECTED COUNT IS NULL: the observed complete "
                f"family of {len(full):,} shards is sealed, but production must pin "
                "remote.expected_shard_count"
            )
        logger.info(
            "Remote statistics snapshot: "
            f"bucket={config.bucket_id} prefix={config.remote_prefix} "
            f"full_count={len(full):,} full_bytes="
            f"{sum(entry.info.byte_length for entry in full):,} "
            f"full_sha256={source.full_inventory_sha256} "
            f"statistics_count={len(domain):,} statistics_bytes="
            f"{sum(entry.info.byte_length for entry in domain):,} "
            f"statistics_sha256={source.inventory_sha256} "
            f"excluded={excluded.info.path if excluded is not None else None}"
        )
        return source

    @property
    def shard_paths(self) -> list[Path]:
        return [Path(entry.info.path) for entry in self.statistics_inventory]

    @property
    def compressed_byte_lengths(self) -> tuple[int, ...]:
        return tuple(entry.info.byte_length for entry in self.statistics_inventory)

    def checkpoint_contract(
        self,
        *,
        n_sae_features: int,
        skip_per_feature_detail: bool,
        numeric_runtime: Mapping[str, Any],
    ) -> dict[str, object]:
        excluded = (
            _inventory_rows((self.excluded_shard,))[0] if self.excluded_shard is not None else None
        )
        return {
            "source_kind": "remote_hf_feature_shards_v1",
            "bucket_id": str(self.config.bucket_id),
            "remote_prefix": str(self.config.remote_prefix),
            "inventory": _inventory_rows(self.statistics_inventory),
            "inventory_sha256": self.inventory_sha256,
            "full_inventory_count": len(self.full_inventory),
            "full_inventory_sha256": self.full_inventory_sha256,
            "excluded_shard": excluded,
            "n_sae_features": n_sae_features,
            "exclude_last_shard": self.exclude_last_shard,
            "skip_per_feature_detail": skip_per_feature_detail,
            "numeric_runtime": dict(numeric_runtime),
        }

    def checkpoint_metadata(self) -> dict[str, object]:
        if self._metadata is None:
            raise RemoteStatisticsError(
                "remote checkpoint progress cannot be committed before shard metadata "
                "is established"
            )
        return {"remote_source_metadata": dict(self._metadata)}

    def restore_checkpoint_metadata(
        self,
        scalars: Mapping[str, object],
    ) -> dict[str, object]:
        restored = dict(scalars)
        if "remote_source_metadata" not in restored:
            raise RemoteStatisticsError(
                "positive-progress remote checkpoint is missing remote_source_metadata"
            )
        raw = restored.pop("remote_source_metadata")
        self._metadata = self._validate_checkpoint_metadata(raw)
        return restored

    def _validate_checkpoint_metadata(self, raw: object) -> dict[str, object]:
        if not isinstance(raw, dict) or set(raw) != _REMOTE_METADATA_KEYS:
            raise RemoteStatisticsError("remote checkpoint source metadata is malformed")
        metadata = cast(dict[str, object], raw)
        if metadata["schema_version"] != _METADATA_SCHEMA_VERSION:
            raise RemoteStatisticsError("remote checkpoint source metadata schema version differs")
        l0_raw = metadata["l0_tier"]
        if l0_raw is not None:
            if type(l0_raw) is not str:
                raise RemoteStatisticsError("remote checkpoint l0_tier is malformed")
            try:
                l0_raw = L0Tier(l0_raw).value
            except ValueError as exc:
                raise RemoteStatisticsError(
                    f"remote checkpoint l0_tier is invalid: {l0_raw!r}"
                ) from exc
        ignore_bos = metadata["ignore_bos_features"]
        if type(ignore_bos) is not bool:
            raise RemoteStatisticsError("remote checkpoint ignore_bos_features is malformed")
        sft_raw = metadata["sft_mixture_fingerprint_json"]
        if sft_raw is not None:
            if type(sft_raw) is not str:
                raise RemoteStatisticsError(
                    "remote checkpoint SFT mixture fingerprint is malformed"
                )
            try:
                parsed = parse_mixture_fingerprint_json(
                    sft_raw,
                    source="remote checkpoint SFT mixture fingerprint",
                )
            except ValueError as exc:
                raise RemoteStatisticsError(
                    "remote checkpoint SFT mixture fingerprint is invalid"
                ) from exc
            canonical = mixture_fingerprint_json(parsed)
            if canonical != sft_raw:
                raise RemoteStatisticsError(
                    "remote checkpoint SFT mixture fingerprint is not canonical"
                )
        return {
            "schema_version": _METADATA_SCHEMA_VERSION,
            "l0_tier": l0_raw,
            "ignore_bos_features": ignore_bos,
            "sft_mixture_fingerprint_json": sft_raw,
        }

    def resolved_metadata(self) -> tuple[L0Tier | None, bool, str | None]:
        if self._metadata is None:
            raise RemoteStatisticsError("remote shard metadata was never established")
        l0_raw = self._metadata["l0_tier"]
        return (
            L0Tier(l0_raw) if isinstance(l0_raw, str) else None,
            bool(self._metadata["ignore_bos_features"]),
            self._metadata["sft_mixture_fingerprint_json"]
            if isinstance(self._metadata["sft_mixture_fingerprint_json"], str)
            else None,
        )

    def _relative_seconds(self) -> float:
        return (time.perf_counter_ns() - self._started_ns) / 1_000_000_000

    def _profile_record(self, index: int) -> dict[str, Any]:
        record = self._profile_records.get(index)
        if record is None:
            entry = self.statistics_inventory[index]
            record = {
                "index": index,
                "path": entry.info.path,
                "byte_length": entry.info.byte_length,
                "disposition": None,
                "scheduled_seconds": None,
                "verified_ready_seconds": None,
                "consumer_requested_seconds": None,
                "consumer_ready_seconds": None,
                "transfer_seconds": 0.0,
                "consumer_wait_seconds": 0.0,
                "attempts": 0,
                "retries": 0,
                "materialized_bytes": 0,
                "terminal_error": None,
            }
            self._profile_records[index] = record
        return record

    def _profile_payload(self, *, status: str) -> dict[str, object]:
        transfer_after_bootstrap = sum(
            float(record["transfer_seconds"])
            for index, record in self._profile_records.items()
            if index != self._bootstrap_index
        )
        hidden = sum(
            max(
                0.0,
                float(record["transfer_seconds"]) - float(record["consumer_wait_seconds"]),
            )
            for index, record in self._profile_records.items()
            if index != self._bootstrap_index
        )
        return {
            "schema_version": 1,
            "status": status,
            "bucket_id": str(self.config.bucket_id),
            "remote_prefix": str(self.config.remote_prefix),
            "full_inventory_count": len(self.full_inventory),
            "full_inventory_bytes": sum(entry.info.byte_length for entry in self.full_inventory),
            "full_inventory_sha256": self.full_inventory_sha256,
            "statistics_inventory_count": len(self.statistics_inventory),
            "statistics_inventory_bytes": sum(
                entry.info.byte_length for entry in self.statistics_inventory
            ),
            "statistics_inventory_sha256": self.inventory_sha256,
            "excluded_shard": (
                _inventory_rows((self.excluded_shard,))[0]
                if self.excluded_shard is not None
                else None
            ),
            "admission_required_bytes": self.required_capacity,
            "admission_observed_free_bytes": self._admission_free_bytes,
            "admission_recoverable_bytes": self._admission_recoverable_bytes,
            "conservative_staging_body_bound_bytes": self.required_capacity
            - _CAPACITY_RESERVE_BYTES,
            "staging_high_water_bytes": self._staging_high_water_bytes,
            "staging_high_water_bodies": self._staging_high_water_bodies,
            "verified_downloaded_bytes": self._verified_downloaded_bytes,
            "verified_reused_bytes": self._verified_reused_bytes,
            "materialized_bytes": self._materialized_bytes,
            "cleanup_join_seconds": self._cleanup_join_seconds,
            "critical_path_remote_wait_seconds": self._critical_wait_seconds,
            "post_bootstrap_transfer_seconds": transfer_after_bootstrap,
            "post_bootstrap_hidden_transfer_seconds": hidden,
            "overlap_fraction": hidden / transfer_after_bootstrap
            if transfer_after_bootstrap
            else 0.0,
            "per_shard": [
                dict(self._profile_records[index]) for index in sorted(self._profile_records)
            ],
        }

    def _refresh_profile(self, *, status: str = "running") -> None:
        if self.profile is not None:
            with self._profile_lock:
                previous = getattr(self.profile, "remote_transport", None)
                payload = self._profile_payload(status=status)
            if isinstance(previous, dict) and "terminal_error" in previous:
                payload["terminal_error"] = previous["terminal_error"]
                payload["status"] = "failed"
            self.profile.remote_transport = payload

    def record_terminal_error(
        self,
        *,
        stage: str,
        error: BaseException,
        index: int | None = None,
        path: str | None = None,
    ) -> None:
        if self.profile is None:
            return
        self._refresh_profile(status="failed")
        assert isinstance(self.profile.remote_transport, dict)
        self.profile.remote_transport.setdefault(
            "terminal_error",
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "index": index,
                "path": path,
            },
        )

    def _owner_payload(self) -> dict[str, object]:
        return {
            "schema_version": _OWNER_SCHEMA_VERSION,
            "bucket_id": str(self.config.bucket_id),
            "remote_prefix": str(self.config.remote_prefix),
        }

    def _open_lock_and_owner(self) -> None:
        staging = self.config.staging_dir
        if staging.exists() and (staging.is_symlink() or not staging.is_dir()):
            raise RemoteStatisticsError(
                f"remote statistics staging must be a real directory: {staging}"
            )
        staging.mkdir(parents=True, exist_ok=True)
        if staging.is_symlink() or not staging.is_dir():
            raise RemoteStatisticsError(
                f"remote statistics staging must be a real directory: {staging}"
            )
        lock_path = staging / _LOCK_FILENAME
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._lock_fd = os.open(lock_path, flags, 0o600)
            opened = os.fstat(self._lock_fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise RemoteStatisticsError(
                    f"remote statistics lock is not a direct single-link file: {lock_path}"
                )
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise RemoteStatisticsError(
                f"remote statistics staging is already in use or cannot be locked: {staging}"
            ) from exc
        except BaseException:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise

        owner_path = staging / _OWNER_FILENAME
        if not owner_path.exists():
            unexpected = [entry for entry in staging.iterdir() if entry.name != _LOCK_FILENAME]
            if unexpected:
                raise RemoteStatisticsError(
                    "remote statistics staging has no ownership marker and is not "
                    f"empty: {[str(path) for path in unexpected]}"
                )
            atomic_json_dump(self._owner_payload(), owner_path)
        _capture_safe_regular(owner_path, surface="remote statistics owner marker")
        try:
            with owner_path.open("r", encoding="utf-8") as handle:
                owner = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RemoteStatisticsError(
                f"remote statistics ownership marker is unreadable: {owner_path}"
            ) from exc
        if owner != self._owner_payload():
            raise RemoteStatisticsError(
                "remote statistics staging belongs to another source: "
                f"expected={self._owner_payload()!r}, observed={owner!r}"
            )

    def _safe_unlink(
        self,
        path: Path,
        *,
        expected_identity: RegularFileIdentity | None = None,
    ) -> None:
        observed = _capture_safe_regular(path, surface="remote statistics cleanup entry")
        if expected_identity is not None and observed != expected_identity:
            self._preserve_paths.add(path)
            raise RemoteStatisticsError(
                f"remote statistics staged path was replaced before unlink: {path}"
            )
        path.unlink()
        logger.info(f"Removed remote statistics staging body: {path}")

    def _classify_staging(self) -> tuple[list[Path], list[tuple[Path, int]], list[Path]]:
        canonicals: list[Path] = []
        temporaries: list[tuple[Path, int]] = []
        unknown: list[Path] = []
        for entry in self.config.staging_dir.iterdir():
            if entry.name in {_LOCK_FILENAME, _OWNER_FILENAME}:
                _capture_safe_regular(entry, surface="remote statistics control entry")
                continue
            if _CANONICAL_RE.fullmatch(entry.name) is not None:
                _capture_safe_regular(entry, surface="remote statistics staged body")
                canonicals.append(entry)
                continue
            match = _TEMP_RE.fullmatch(entry.name)
            if match is not None:
                _capture_safe_regular(entry, surface="remote statistics temporary")
                temporaries.append((entry, int(match.group("pid"))))
                continue
            unknown.append(entry)
        return canonicals, temporaries, unknown

    def _prepare_namespace(self, start_index: int) -> int:
        self._open_lock_and_owner()
        self._staging_initialized = True
        canonicals, temporaries, unknown = self._classify_staging()
        if unknown:
            raise RemoteStatisticsError(
                "unknown entry in remote statistics staging; preserving it and "
                f"refusing adoption: {[str(path) for path in unknown]}"
            )
        for path, pid in temporaries:
            if _pid_is_live_or_inaccessible(pid):
                self._preserve_paths.add(path)
                raise RemoteStatisticsError(
                    f"live or inaccessible download owner PID {pid} blocks staging adoption: {path}"
                )
            self._safe_unlink(path)
        keep_names = {
            self.statistics_inventory[index].info.path.rsplit("/", 1)[-1]
            for index in (start_index, start_index + 1)
            if index < len(self.statistics_inventory)
        }
        retained_bytes = 0
        for path in canonicals:
            if path.name not in keep_names:
                self._safe_unlink(path)
            else:
                retained_bytes += _capture_safe_regular(
                    path,
                    surface="retained staged body admission",
                ).size
        free = shutil.disk_usage(self.config.staging_dir).free
        recoverable = free + retained_bytes
        self._admission_free_bytes = free
        self._admission_recoverable_bytes = recoverable
        if recoverable < self.required_capacity:
            raise RemoteStatisticsError(
                "remote statistics staging admission failed before transport: "
                f"required={self.required_capacity:,}, free={free:,}, "
                f"retained={retained_bytes:,}, recoverable={recoverable:,}"
            )
        logger.info(
            "Remote statistics staging admitted: "
            f"required={self.required_capacity:,} free={free:,} "
            f"retained={retained_bytes:,} recoverable={recoverable:,}"
        )
        self._observe_staging()
        self._refresh_profile()
        return retained_bytes

    def _observe_staging(self) -> None:
        with self._profile_lock:
            if not self.config.staging_dir.exists():
                return
            total = 0
            bodies = 0
            for entry in self.config.staging_dir.iterdir():
                if (
                    _CANONICAL_RE.fullmatch(entry.name) is None
                    and _TEMP_RE.fullmatch(entry.name) is None
                ):
                    continue
                try:
                    info = entry.lstat()
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    total += int(info.st_size)
                    bodies += 1
            self._staging_high_water_bytes = max(self._staging_high_water_bytes, total)
            self._staging_high_water_bodies = max(
                self._staging_high_water_bodies,
                bodies,
            )

    def _ensure_transport(self, start_index: int) -> None:
        if self._executor is not None:
            return
        if not self._staging_initialized:
            try:
                self._prepare_namespace(start_index)
            except BaseException as exc:
                self.record_terminal_error(
                    stage="staging_initialization",
                    error=exc,
                    index=start_index,
                    path=self.statistics_inventory[start_index].info.path,
                )
                try:
                    self.finish_transport()
                except BaseException as cleanup_exc:
                    exc.add_note(
                        f"remote statistics initialization cleanup also failed: {cleanup_exc!r}"
                    )
                raise
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="statistics-remote-download",
        )

    def _local_path(self, index: int) -> Path:
        name = self.statistics_inventory[index].info.path.rsplit("/", 1)[-1]
        return self.config.staging_dir / name

    def _download_or_reuse(self, index: int) -> _TransferResult:
        started = time.perf_counter_ns()
        entry = self.statistics_inventory[index]
        path = self._local_path(index)
        disposition = "downloaded"
        attempts = 0
        materialized = 0
        recording: _DownloadRecordingBackend | None = None
        identity: RegularFileIdentity | None = None
        try:
            if path.exists() or path.is_symlink():
                identity_before = _capture_safe_regular(
                    path,
                    surface="retained staged body",
                )
                try:
                    observed = verified_local_object_info(
                        local_path=path,
                        remote_path=entry.info.path,
                    )
                except Exception:
                    self._safe_unlink(path, expected_identity=identity_before)
                else:
                    identity_after = _capture_safe_regular(
                        path,
                        surface="verified retained staged body",
                    )
                    if identity_after != identity_before:
                        self._preserve_paths.add(path)
                        raise RemoteStatisticsError(
                            f"retained staged body changed during verification: {path}"
                        )
                    if observed == entry.info:
                        disposition = "reused"
                        identity = identity_after
                    else:
                        self._safe_unlink(path, expected_identity=identity_before)
            if disposition == "downloaded":
                recording = _DownloadRecordingBackend(
                    self.backend,
                    observe_staging=self._observe_staging,
                )

                def bind_verified_identity(published: os.stat_result) -> None:
                    nonlocal identity
                    if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
                        raise RemoteStatisticsError(
                            "verified staged body must be a direct, single-link "
                            f"regular file: {path}"
                        )
                    identity = RegularFileIdentity.from_verified_publication(
                        published,
                        verified_byte_length=entry.info.byte_length,
                        path=path,
                    )

                download_verified_object(
                    recording,
                    expected=entry.info,
                    final_path=path,
                    timeout_seconds=self.config.retry_timeout_seconds,
                    retry_interval_seconds=self.config.retry_interval_seconds,
                    cancellation_event=self._cancel,
                    verified_publication_callback=bind_verified_identity,
                )
                attempts = recording.attempts
                materialized = recording.materialized_bytes
            if identity is None:
                raise RemoteStatisticsError(
                    f"verified staged body has no bound file identity: {entry.info.path}"
                )
            prepared = PreparedRemoteShard(
                index=index,
                logical_path=Path(entry.info.path),
                local_path=path,
                remote_identity=entry.info,
                file_identity=identity,
                disposition=disposition,
            )
            with self._profile_lock:
                if disposition == "reused":
                    self._verified_reused_bytes += entry.info.byte_length
                else:
                    self._verified_downloaded_bytes += entry.info.byte_length
        finally:
            if recording is not None:
                attempts = recording.attempts
                materialized = recording.materialized_bytes
            elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
            with self._profile_lock:
                record = self._profile_record(index)
                record["transfer_seconds"] = elapsed
                record["attempts"] = attempts
                record["retries"] = max(0, attempts - 1)
                record["materialized_bytes"] = materialized
                record["disposition"] = disposition
                self._materialized_bytes += materialized
        with self._profile_lock:
            self._profile_record(index)["verified_ready_seconds"] = self._relative_seconds()
        return _TransferResult(
            prepared=prepared,
            transfer_seconds=elapsed,
            attempts=attempts,
            materialized_bytes=materialized,
        )

    def _submit(self, index: int) -> None:
        if index >= len(self.statistics_inventory):
            return
        if self._future is not None:
            raise RemoteStatisticsError("remote statistics already has a pending future")
        assert self._executor is not None
        with self._profile_lock:
            if self._bootstrap_index is None:
                self._bootstrap_index = index
            self._profile_record(index)["scheduled_seconds"] = self._relative_seconds()
        self._future_index = index
        self._future = self._executor.submit(self._download_or_reuse, index)

    def _consume_future(self, index: int) -> _TransferResult:
        if self._future is None or self._future_index != index:
            raise RemoteStatisticsError(
                "remote statistics scheduler order changed: "
                f"requested={index}, pending={self._future_index}"
            )
        requested = self._relative_seconds()
        wait_started = time.perf_counter_ns()
        with self._profile_lock:
            self._profile_record(index)["consumer_requested_seconds"] = requested
        try:
            try:
                result = self._future.result()
            finally:
                wait = (time.perf_counter_ns() - wait_started) / 1_000_000_000
                self._future = None
                self._future_index = None
                with self._profile_lock:
                    self._critical_wait_seconds += wait
                    record = self._profile_record(index)
                    record["consumer_ready_seconds"] = self._relative_seconds()
                    record["consumer_wait_seconds"] = wait
        except BaseException as exc:
            with self._profile_lock:
                self._profile_record(index)["terminal_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            if not (self._cancel.is_set() and _is_cancelled_download(exc)):
                self.record_terminal_error(
                    stage="download",
                    error=exc,
                    index=index,
                    path=self.statistics_inventory[index].info.path,
                )
            raise
        with self._profile_lock:
            record = self._profile_record(index)
            record["transfer_seconds"] = result.transfer_seconds
            record["attempts"] = result.attempts
            record["retries"] = max(0, result.attempts - 1)
            record["materialized_bytes"] = result.materialized_bytes
            record["disposition"] = result.prepared.disposition
        self._observe_staging()
        self._refresh_profile()
        return result

    def _metadata_for_header(
        self,
        prepared: PreparedRemoteShard,
        metadata: Mapping[str, str],
    ) -> dict[str, object]:
        logical_path = prepared.logical_path
        l0_tier = assert_uniform_l0_tier_metadata([(logical_path, metadata)])
        ignore_bos = assert_uniform_ignore_bos_metadata([(logical_path, metadata)])
        sft = _uniform_sft_mixture_fingerprint_metadata_json(
            [(logical_path, metadata)],
            operation="compute_statistics remote streaming",
        )
        raw_segment_size = metadata.get("segment_size")
        try:
            segment_size = int(raw_segment_size) if raw_segment_size is not None else None
        except ValueError as exc:
            raise RemoteStatisticsError(
                f"remote shard {logical_path} has malformed segment_size metadata "
                f"{raw_segment_size!r}"
            ) from exc
        if segment_size != self.config.expected_segment_size:
            raise RemoteStatisticsError(
                f"remote shard {logical_path} segment_size={segment_size!r} differs "
                f"from resolved segment_size={self.config.expected_segment_size}"
            )
        return {
            "schema_version": _METADATA_SCHEMA_VERSION,
            "l0_tier": l0_tier.value if l0_tier is not None else None,
            "ignore_bos_features": ignore_bos,
            "sft_mixture_fingerprint_json": sft,
        }

    def prepare(self, index: int) -> PreparedRemoteShard:
        """Wait for ``index``, schedule ``index+1``, then validate its header."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise RemoteStatisticsError(f"remote shard index must be an int, got {index!r}")
        if index < 0 or index >= len(self.statistics_inventory):
            raise RemoteStatisticsError(f"remote shard index out of range: {index}")
        self._ensure_transport(index)
        if self._future is None:
            self._submit(index)
        result = self._consume_future(index)
        self._submit(index + 1)
        prepared = result.prepared
        try:
            header = read_shard_header_stable(
                prepared.local_path,
                expected_identity=prepared.file_identity,
            )
            observed_metadata = self._metadata_for_header(
                prepared,
                header["__metadata__"],
            )
            if self._metadata is None:
                self._metadata = observed_metadata
            elif observed_metadata != self._metadata:
                raise RemoteStatisticsError(
                    "remote shard metadata differs from the established statistics "
                    f"domain at {prepared.logical_path}: expected={self._metadata!r}, "
                    f"observed={observed_metadata!r}"
                )
        except BaseException as exc:
            self.record_terminal_error(
                stage="header",
                error=exc,
                index=index,
                path=str(prepared.logical_path),
            )
            raise
        return prepared

    def release(self, prepared: PreparedRemoteShard) -> None:
        """Delete a consumed body only if its exact filesystem identity remains."""
        self._safe_unlink(
            prepared.local_path,
            expected_identity=prepared.file_identity,
        )
        self._observe_staging()
        self._refresh_profile()

    def _cleanup_bodies(self) -> None:
        if not self._staging_initialized or not self.config.staging_dir.exists():
            return
        canonicals, temporaries, unknown = self._classify_staging()
        cleanup_error: BaseException | None = None
        for path in canonicals:
            if path in self._preserve_paths:
                continue
            try:
                self._safe_unlink(path)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        for path, pid in temporaries:
            if path in self._preserve_paths:
                continue
            if pid != os.getpid() and _pid_is_live_or_inaccessible(pid):
                cleanup_error = cleanup_error or RemoteStatisticsError(
                    f"cannot clean temporary owned by live/inaccessible PID {pid}: {path}"
                )
                continue
            try:
                self._safe_unlink(path)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if unknown:
            cleanup_error = cleanup_error or RemoteStatisticsError(
                "unknown entries appeared in remote statistics staging and were "
                f"preserved: {[str(path) for path in unknown]}"
            )
        if cleanup_error is not None:
            raise cleanup_error

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def request_cancel(self) -> None:
        """Signal transport cancellation without joining or cleaning staging."""
        self._cancel.set()

    def finish_transport(self) -> None:
        """Cancel/join transport and remove recognized bodies; safe to repeat."""
        if self._finish_done:
            return
        self._finish_done = True
        started = time.perf_counter_ns()
        failure: BaseException | None = None
        self.request_cancel()
        if self._future is not None:
            try:
                cancelled = self._future.cancel()
                if not cancelled:
                    self._future.result()
            except BaseException as exc:
                if not _is_cancelled_download(exc):
                    failure = exc
            finally:
                self._future = None
                self._future_index = None
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                elif exc is not failure:
                    failure.add_note(f"remote statistics executor shutdown also failed: {exc!r}")
            finally:
                self._executor = None
        try:
            self._cleanup_bodies()
        except BaseException as exc:
            if failure is None:
                failure = exc
            elif exc is not failure:
                failure.add_note(f"remote statistics cleanup also failed: {exc!r}")
        try:
            self._release_lock()
        except BaseException as exc:
            if failure is None:
                failure = exc
            elif exc is not failure:
                failure.add_note(f"remote statistics lock release also failed: {exc!r}")
        finally:
            self._cleanup_join_seconds += (time.perf_counter_ns() - started) / 1_000_000_000
            self._refresh_profile(status="failed" if failure is not None else "complete")
        if failure is not None:
            self.record_terminal_error(stage="transport_finish", error=failure)
            raise failure

    def assert_final_inventory_unchanged(self) -> None:
        """Relist and require exact complete-family identity before publication."""
        listing = self.backend.list(str(self.config.remote_prefix))
        observed = _validate_family(
            normalize_remote_feature_inventory(
                listing,
                prefix=self.config.remote_prefix,
            ),
            expected_count=self.config.expected_shard_count,
            prefix=self.config.remote_prefix,
        )
        observed_excluded = observed[-1] if self.exclude_last_shard and len(observed) > 1 else None
        observed_domain = observed[:-1] if observed_excluded is not None else observed
        if (
            observed != self.full_inventory
            or observed_domain != self.statistics_inventory
            or remote_feature_inventory_digest(observed) != self.full_inventory_sha256
        ):
            raise RemoteStatisticsError(
                "remote feature inventory changed before statistics publication: "
                f"expected_count={len(self.full_inventory)}, "
                f"observed_count={len(observed)}, "
                f"expected_sha256={self.full_inventory_sha256}, "
                f"observed_sha256={remote_feature_inventory_digest(observed)}"
            )


__all__ = [
    "PreparedRemoteShard",
    "RemoteStatisticsConfig",
    "RemoteStatisticsError",
    "RemoteStatisticsSource",
    "required_staging_capacity_bytes",
]
