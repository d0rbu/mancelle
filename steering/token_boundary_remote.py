"""Identity-sealed HF authentication and depth-one token-boundary transport."""

from __future__ import annotations

import fcntl
import math
import os
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from steering.feature_io import (
    COMPRESSED_EXT,
    RegularFileIdentity,
    read_feature_compressed_bytes_stable,
)
from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    RemoteShardFamily,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
)
from steering.remote_training import (
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    download_verified_object,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.token_boundary_source import (
    MissingTokenBoundarySourceError,
    TokenBoundarySourceManifest,
    derive_token_boundary_source_prefix,
    parse_token_boundary_source_manifest,
    token_boundary_source_manifest_path,
    validate_token_boundary_source_manifest,
)

_CAPACITY_RESERVE_BYTES = 1 << 30
_RUN_PREFIX = ".token-boundary-remote."


class TokenBoundaryRemoteError(RuntimeError):
    """Remote token-boundary configuration, inventory, or staging is unsafe."""


@dataclass(frozen=True)
class TokenBoundaryRemoteConfig:
    bucket_id: HfBucketId
    remote_prefix: NormalizedRemotePrefix
    staging_dir: Path
    expected_shard_count: int | None
    retry_timeout_seconds: float
    retry_interval_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket_id", HfBucketId.parse(str(self.bucket_id)))
        object.__setattr__(
            self,
            "remote_prefix",
            NormalizedRemotePrefix.parse(str(self.remote_prefix)),
        )
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
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class PreparedTokenBoundaryShard:
    index: int
    logical_path: Path
    local_path: Path
    remote_identity: RemoteObjectInfo
    file_identity: RegularFileIdentity


@dataclass(frozen=True)
class _TransferResult:
    prepared: PreparedTokenBoundaryShard
    transfer_seconds: float
    attempts: int
    materialized_bytes: int


def _validate_family(
    inventory: Sequence[RemoteFeatureShard],
    *,
    expected_count: int | None,
    prefix: NormalizedRemotePrefix,
) -> tuple[RemoteFeatureShard, ...]:
    sealed = tuple(inventory)
    if not sealed:
        raise TokenBoundaryRemoteError(
            f"no direct token-boundary input shard objects found under {prefix}"
        )
    family_size = expected_count if expected_count is not None else sealed[-1].index + 1
    expected_indices = list(range(family_size))
    observed_indices = [entry.index for entry in sealed]
    if observed_indices != expected_indices:
        expected_set = set(expected_indices)
        observed_set = set(observed_indices)
        raise TokenBoundaryRemoteError(
            "remote token-boundary input inventory is not the exact contiguous family "
            f"0..{family_size - 1}: "
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


def _capture_direct_single_link(path: Path, *, surface: str) -> RegularFileIdentity:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise TokenBoundaryRemoteError(f"cannot inspect {surface} {path}: {exc}") from exc
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise TokenBoundaryRemoteError(
            f"{surface} must be a direct, single-link regular file: {path}"
        )
    return RegularFileIdentity.from_stat(observed)


def _staged_identity_may_drift(
    expected: RegularFileIdentity,
    observed: RegularFileIdentity,
    *,
    path: Path,
) -> bool:
    # Lustre can transiently report a stale st_size for a recently written file,
    # so size (like timestamps) is advisory between two stats of one staged body:
    # content authority is the full-read download verification against the sealed
    # inventory plus the zstd/safetensors decode. mode/device/inode drift is fatal.
    if expected == observed:
        return True
    structural_fields = ("mode", "device", "inode")
    if any(getattr(expected, name) != getattr(observed, name) for name in structural_fields):
        return False
    drifted = [
        name
        for name in ("size", "mtime_ns", "ctime_ns")
        if getattr(expected, name) != getattr(observed, name)
    ]
    if not drifted:
        return False
    logger.warning(
        f"staged body identity drift tolerated: {path}; drifted_fields={','.join(drifted)}"
    )
    return True


def _ensure_staging_directory(staging: Path) -> None:
    if os.path.lexists(staging) and (staging.is_symlink() or not staging.is_dir()):
        raise TokenBoundaryRemoteError(
            f"remote token-boundary staging must be a real directory: {staging}"
        )
    if not os.path.lexists(staging):
        staging.mkdir(parents=True, exist_ok=True)
    if staging.is_symlink() or not staging.is_dir():
        raise TokenBoundaryRemoteError(
            f"remote token-boundary staging must be a real directory: {staging}"
        )


class _RecordingBackend:
    def __init__(
        self,
        delegate: BucketBackend,
        *,
        observe: Callable[[], None],
    ) -> None:
        self.delegate = delegate
        self.observe = observe
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
                observed = local_path.lstat()
            except OSError:
                pass
            else:
                if stat.S_ISREG(observed.st_mode):
                    self.materialized_bytes += int(observed.st_size)
            self.observe()

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.delegate.upload(local_path, remote_path)


class RemoteTokenBoundaryTransport:
    """One-worker, one-lookahead stream bound to an immutable bucket snapshot."""

    def __init__(
        self,
        *,
        config: TokenBoundaryRemoteConfig,
        backend: BucketBackend,
        inventory: tuple[RemoteFeatureShard, ...],
        snapshot_listing: tuple[RemoteObjectInfo, ...],
        family: RemoteShardFamily,
    ) -> None:
        self.config = config
        self.backend = backend
        self.inventory = inventory
        self.snapshot_listing = snapshot_listing
        self.family = family
        self.inventory_sha256 = remote_feature_inventory_digest(inventory)
        self.inventory_bytes = sum(entry.info.byte_length for entry in inventory)
        self.required_capacity = required_staging_capacity_bytes(inventory)
        self._run_dir: Path | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[_TransferResult] | None = None
        self._stream_iterator: Generator[PreparedTokenBoundaryShard] | None = None
        self._cancel = threading.Event()
        self._profile_lock = threading.Lock()
        self._downloaded_bytes = 0
        self._transfer_seconds = 0.0
        self._critical_wait_seconds = 0.0
        self._per_body_transport: dict[int, dict[str, object]] = {}
        self._snapshot_list_seconds = 0.0
        self._fence_list_seconds = 0.0
        self._list_calls = 0
        self._staging_high_water_bytes = 0
        self._staging_high_water_bodies = 0
        self._admission_free_bytes: int | None = None
        self._final_fence_status = "not_run"
        self._terminal_error: dict[str, str] | None = None

    @classmethod
    def snapshot(
        cls,
        config: TokenBoundaryRemoteConfig,
        *,
        backend: BucketBackend | None = None,
        family: RemoteShardFamily = RemoteShardFamily.RAW_FEATURES,
    ) -> RemoteTokenBoundaryTransport:
        resolved_backend = backend or HfBucketBackend(config.bucket_id)
        list_started = time.perf_counter()
        listed = resolved_backend.list(str(config.remote_prefix))
        list_seconds = time.perf_counter() - list_started
        normalized = normalize_remote_feature_inventory(
            listed,
            prefix=config.remote_prefix,
            family=family,
        )
        inventory = _validate_family(
            normalized,
            expected_count=config.expected_shard_count,
            prefix=config.remote_prefix,
        )
        transport = cls(
            config=config,
            backend=resolved_backend,
            inventory=inventory,
            snapshot_listing=tuple(listed),
            family=family,
        )
        transport._snapshot_list_seconds = list_seconds
        transport._list_calls = 1
        if config.expected_shard_count is None:
            logger.warning(
                "REMOTE TOKEN BOUNDARIES EXPECTED COUNT IS NULL: the observed "
                f"complete family of {len(inventory):,} shards is sealed, but "
                "production must pin remote.expected_shard_count"
            )
        logger.info(
            "Remote token-boundary snapshot: "
            f"bucket={config.bucket_id} prefix={config.remote_prefix} "
            f"count={len(inventory):,} bytes={transport.inventory_bytes:,} "
            f"sha256={transport.inventory_sha256}"
        )
        return transport

    def __enter__(self) -> RemoteTokenBoundaryTransport:
        try:
            self._start()
        except BaseException as exc:
            # Context-manager ``__exit__`` is not invoked when ``__enter__``
            # raises. Reuse the normal teardown so a partially-created run dir
            # or executor cannot leak and the terminal profile names the error.
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None:
            self._terminal_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        cleanup_error: BaseException | None = None
        try:
            if self._stream_iterator is not None:
                self._stream_iterator.close()
                self._stream_iterator = None
            self._stop_transport()
        except BaseException as caught:
            cleanup_error = caught
        try:
            self._cleanup_run_dir()
        except BaseException as caught:
            if cleanup_error is None:
                cleanup_error = caught
            else:
                cleanup_error.add_note(f"additional staging cleanup failure: {caught!r}")
        if cleanup_error is not None:
            if self._terminal_error is None:
                self._terminal_error = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
            if exc is not None:
                exc.add_note(f"remote transport cleanup also failed: {cleanup_error!r}")
            else:
                raise cleanup_error

    def _start(self) -> None:
        staging = self.config.staging_dir
        _ensure_staging_directory(staging)
        free = shutil.disk_usage(staging).free
        self._admission_free_bytes = free
        if free < self.required_capacity:
            raise TokenBoundaryRemoteError(
                "remote token-boundary staging admission failed before transport: "
                f"required={self.required_capacity:,}, free={free:,}"
            )
        self._run_dir = Path(tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=staging))
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="token-boundary-remote-download",
        )
        logger.info(
            "Remote token-boundary staging admitted: "
            f"required={self.required_capacity:,} free={free:,} run_dir={self._run_dir}"
        )

    def _local_path(self, position: int) -> Path:
        assert self._run_dir is not None
        return self._run_dir / self.inventory[position].info.path.rsplit("/", 1)[-1]

    def _observe_staging(self) -> None:
        run_dir = self._run_dir
        if run_dir is None or not run_dir.exists():
            return
        total = 0
        bodies = 0
        for path in run_dir.iterdir():
            try:
                observed = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(observed.st_mode):
                total += int(observed.st_size)
                bodies += 1
        with self._profile_lock:
            self._staging_high_water_bytes = max(self._staging_high_water_bytes, total)
            self._staging_high_water_bodies = max(
                self._staging_high_water_bodies,
                bodies,
            )

    def _download(self, position: int) -> _TransferResult:
        started = time.perf_counter()
        entry = self.inventory[position]
        path = self._local_path(position)
        recording = _RecordingBackend(self.backend, observe=self._observe_staging)
        phase_timings: dict[str, float] = {}
        identity: RegularFileIdentity | None = None

        def bind_identity(published: os.stat_result) -> None:
            nonlocal identity
            if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
                raise TokenBoundaryRemoteError(
                    f"verified staged body must be a direct single-link file: {path}"
                )
            identity = RegularFileIdentity.from_verified_publication(
                published,
                verified_byte_length=entry.info.byte_length,
                path=path,
            )

        error: BaseException | None = None
        try:
            download_verified_object(
                recording,
                expected=entry.info,
                final_path=path,
                timeout_seconds=self.config.retry_timeout_seconds,
                retry_interval_seconds=self.config.retry_interval_seconds,
                cancellation_event=self._cancel,
                verified_publication_callback=bind_identity,
                phase_timings=phase_timings,
            )
            if identity is None:
                raise TokenBoundaryRemoteError(
                    f"verified staged body has no bound file identity: {entry.info.path}"
                )
            observed = _capture_direct_single_link(path, surface="verified staged body")
            if not _staged_identity_may_drift(identity, observed, path=path):
                drifted = [
                    name
                    for name in (
                        "mode",
                        "device",
                        "inode",
                        "size",
                        "mtime_ns",
                        "ctime_ns",
                    )
                    if getattr(identity, name) != getattr(observed, name)
                ]
                raise TokenBoundaryRemoteError(
                    f"verified staged body changed after publication: {path}; "
                    f"drifted_fields={','.join(drifted)}; expected={identity!r}, "
                    f"observed={observed!r}"
                )
            identity = replace(observed, size=entry.info.byte_length)
        except BaseException as caught:
            error = caught
            raise
        finally:
            # Publish the per-body record in the SAME locked update as the run
            # aggregates so failed/cancelled/unconsumed downloads are recorded
            # too (the failure sidecar's most relevant transfer is the one that
            # failed) and profile_payload() sees cross-field-consistent state.
            elapsed = time.perf_counter() - started
            with self._profile_lock:
                self._downloaded_bytes += recording.materialized_bytes
                self._transfer_seconds += elapsed
                self._per_body_transport[position] = {
                    "position": position,
                    "status": "ok" if error is None else "failed",
                    "error": None if error is None else type(error).__name__,
                    "attempts": recording.attempts,
                    "transfer_seconds": elapsed,
                    "download_seconds": phase_timings.get("download_seconds", 0.0),
                    "sha256_seconds": phase_timings.get("sha256_seconds", 0.0),
                    "xet_seconds": phase_timings.get("xet_seconds", 0.0),
                    "materialized_bytes": recording.materialized_bytes,
                }
            self._observe_staging()
        return _TransferResult(
            prepared=PreparedTokenBoundaryShard(
                index=entry.index,
                logical_path=Path(entry.info.path),
                local_path=path,
                remote_identity=entry.info,
                file_identity=identity,
            ),
            transfer_seconds=elapsed,
            attempts=recording.attempts,
            materialized_bytes=recording.materialized_bytes,
        )

    def _submit(self, position: int) -> Future[_TransferResult]:
        executor = self._executor
        assert executor is not None
        return executor.submit(self._download, position)

    def stream(
        self,
        *,
        start_position: int = 0,
        stop_position: int | None = None,
    ) -> Generator[PreparedTokenBoundaryShard]:
        """Stream bodies start_position..stop_position-1 in sealed order.

        The bounds do not touch the sealed inventory, so positions, profile
        records, and the final inventory fence keep their whole-family meaning.
        """
        if self._executor is None:
            raise TokenBoundaryRemoteError("remote token-boundary transport is not active")
        if self._stream_iterator is not None:
            raise TokenBoundaryRemoteError("remote token-boundary stream is already active")
        stop = len(self.inventory) if stop_position is None else stop_position
        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
            or not (0 <= start_position < stop <= len(self.inventory))
        ):
            raise TokenBoundaryRemoteError(
                "stream bounds must satisfy 0 <= start_position < stop_position <= "
                f"{len(self.inventory)}: got start={start_position!r}, stop={stop_position!r}"
            )
        iterator = self._stream_impl(start_position, stop)
        self._stream_iterator = iterator
        return iterator

    def _stream_impl(self, start: int, stop: int) -> Generator[PreparedTokenBoundaryShard]:
        self._future = self._submit(start)
        try:
            for position in range(start, stop):
                future = self._future
                assert future is not None
                wait_started = time.perf_counter()
                try:
                    result = future.result()
                finally:
                    with self._profile_lock:
                        self._critical_wait_seconds += time.perf_counter() - wait_started
                self._future = self._submit(position + 1) if position + 1 < stop else None
                prepared = result.prepared
                try:
                    yield prepared
                finally:
                    self.release(prepared)
        except BaseException:
            self._cancel.set()
            if self._future is not None:
                self._future.cancel()
            raise
        finally:
            self._stop_transport()

    def release(self, prepared: PreparedTokenBoundaryShard) -> None:
        observed = _capture_direct_single_link(
            prepared.local_path,
            surface="consumed staged body",
        )
        if not _staged_identity_may_drift(
            prepared.file_identity,
            observed,
            path=prepared.local_path,
        ):
            raise TokenBoundaryRemoteError(
                "staged token-boundary input body changed while consumed: "
                f"{prepared.local_path}; expected={prepared.file_identity!r}, "
                f"observed={observed!r}"
            )
        prepared.local_path.unlink()
        self._observe_staging()

    def _stop_transport(self) -> None:
        self._cancel.set()
        if self._future is not None:
            self._future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._future = None

    def _cleanup_run_dir(self) -> None:
        run_dir = self._run_dir
        self._run_dir = None
        if run_dir is None or not run_dir.exists():
            return
        if run_dir.parent != self.config.staging_dir or not run_dir.name.startswith(_RUN_PREFIX):
            raise TokenBoundaryRemoteError(
                f"refusing cleanup outside run-owned staging namespace: {run_dir}"
            )
        shutil.rmtree(run_dir)

    def final_inventory_fence(self) -> None:
        try:
            list_started = time.perf_counter()
            try:
                listed = self.backend.list(str(self.config.remote_prefix))
            finally:
                with self._profile_lock:
                    self._fence_list_seconds += time.perf_counter() - list_started
                    self._list_calls += 1
            observed = _validate_family(
                normalize_remote_feature_inventory(
                    listed,
                    prefix=self.config.remote_prefix,
                    family=self.family,
                ),
                expected_count=self.config.expected_shard_count,
                prefix=self.config.remote_prefix,
            )
            observed_digest = remote_feature_inventory_digest(observed)
            if observed != self.inventory or observed_digest != self.inventory_sha256:
                raise TokenBoundaryRemoteError(
                    "remote token-boundary input inventory changed before publication: "
                    f"expected_count={len(self.inventory)}, observed_count={len(observed)}, "
                    f"expected_sha256={self.inventory_sha256}, "
                    f"observed_sha256={observed_digest}"
                )
        except BaseException:
            self._final_fence_status = "failed"
            raise
        self._final_fence_status = "passed"

    def profile_payload(self) -> dict[str, object]:
        # One lock section for EVERY mutable profile field so a live caller
        # sees a cross-field-consistent snapshot (records, aggregates, and
        # staging high-water values are all updated under this same lock).
        with self._profile_lock:
            per_body_transport: list[dict[str, object]] = []
            total_attempts = 0
            for position in sorted(self._per_body_transport):
                record = dict(self._per_body_transport[position])
                attempts = record["attempts"]
                assert isinstance(attempts, int)
                total_attempts += attempts
                per_body_transport.append(record)
            snapshot = {
                "downloaded_bytes": self._downloaded_bytes,
                "total_transfer_seconds": self._transfer_seconds,
                "critical_path_wait_seconds": self._critical_wait_seconds,
                "snapshot_list_seconds": self._snapshot_list_seconds,
                "fence_list_seconds": self._fence_list_seconds,
                "list_calls": self._list_calls,
                "staging_high_water_bytes": self._staging_high_water_bytes,
                "staging_high_water_bodies": self._staging_high_water_bodies,
            }
        return {
            "inventory_count": len(self.inventory),
            "inventory_bytes": self.inventory_bytes,
            "inventory_sha256": self.inventory_sha256,
            "admission_required_bytes": self.required_capacity,
            "admission_observed_free_bytes": self._admission_free_bytes,
            **snapshot,
            "per_body_transport": per_body_transport,
            "total_attempts": total_attempts,
            "final_fence_status": self._final_fence_status,
            "terminal_error": self._terminal_error,
        }


def _download_source_manifest(
    *,
    backend: BucketBackend,
    manifest_info: RemoteObjectInfo,
    config: TokenBoundaryRemoteConfig,
) -> TokenBoundarySourceManifest:
    _ensure_staging_directory(config.staging_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".token-boundary-source-manifest.",
        suffix=COMPRESSED_EXT,
        dir=config.staging_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        identity: RegularFileIdentity | None = None

        def bind_identity(published: os.stat_result) -> None:
            nonlocal identity
            if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
                raise TokenBoundaryRemoteError(
                    "compact source manifest must be a single-link regular file"
                )
            identity = RegularFileIdentity.from_verified_publication(
                published,
                verified_byte_length=manifest_info.byte_length,
                path=temporary_path,
            )

        download_verified_object(
            backend,
            expected=manifest_info,
            final_path=temporary_path,
            timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
            verified_publication_callback=bind_identity,
        )
        if identity is None:
            raise TokenBoundaryRemoteError("compact source manifest has no file identity")
        return parse_token_boundary_source_manifest(
            read_feature_compressed_bytes_stable(
                temporary_path,
                expected_identity=identity,
            ),
        )
    finally:
        temporary_path.unlink(missing_ok=True)


class AuthenticatedTokenBoundarySource:
    """A compact source authenticated against its live raw HF inventory."""

    def __init__(
        self,
        *,
        raw_transport: RemoteTokenBoundaryTransport,
        compact_prefix: NormalizedRemotePrefix,
        compact_transport: RemoteTokenBoundaryTransport,
        manifest_info: RemoteObjectInfo,
        manifest: TokenBoundarySourceManifest,
    ) -> None:
        self.raw_transport = raw_transport
        self.compact_prefix = compact_prefix
        self.manifest_info = manifest_info
        self.manifest = manifest
        self.transport = compact_transport
        self._final_fence_status = "not_run"

    @classmethod
    def snapshot(
        cls,
        config: TokenBoundaryRemoteConfig,
        *,
        backend: BucketBackend | None = None,
    ) -> AuthenticatedTokenBoundarySource:
        resolved_backend = backend or HfBucketBackend(config.bucket_id)
        raw_transport = RemoteTokenBoundaryTransport.snapshot(
            config,
            backend=resolved_backend,
        )
        raw_digest = raw_transport.inventory_sha256
        compact_prefix = derive_token_boundary_source_prefix(
            config.remote_prefix,
            raw_digest,
        )
        manifest_path = token_boundary_source_manifest_path(compact_prefix)
        manifest_info = resolved_backend.stat(manifest_path)
        if manifest_info is None:
            expected = (
                config.expected_shard_count
                if config.expected_shard_count is not None
                else len(raw_transport.inventory)
            )
            raise MissingTokenBoundarySourceError(
                "compact token-boundary source is not published: "
                f"raw_prefix={config.remote_prefix}, "
                f"expected_compact_prefix={compact_prefix}. Prepare it explicitly with: "
                "uv run python -m scripts.preprocess.publish_token_boundary_source "
                f"bucket_id={config.bucket_id} raw_prefix={config.remote_prefix} "
                "staging_dir=<scratch-dir> "
                f"expected_shard_count={expected}"
            )
        manifest = _download_source_manifest(
            backend=resolved_backend,
            manifest_info=manifest_info,
            config=config,
        )
        compact_config = TokenBoundaryRemoteConfig(
            bucket_id=config.bucket_id,
            remote_prefix=compact_prefix,
            staging_dir=config.staging_dir,
            expected_shard_count=len(raw_transport.inventory),
            retry_timeout_seconds=config.retry_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
        )
        compact_transport = RemoteTokenBoundaryTransport.snapshot(
            compact_config,
            backend=resolved_backend,
        )
        validate_token_boundary_source_manifest(
            manifest,
            expected_shard_count=len(raw_transport.inventory),
            expected_raw_inventory_sha256=raw_digest,
            expected_source_inventory_sha256=compact_transport.inventory_sha256,
        )
        logger.info(
            "Authenticated compact token-boundary source: "
            f"raw_prefix={config.remote_prefix} compact_prefix={compact_prefix} "
            f"count={len(compact_transport.inventory):,} "
            f"compact_bytes={compact_transport.inventory_bytes:,}"
        )
        return cls(
            raw_transport=raw_transport,
            compact_prefix=compact_prefix,
            compact_transport=compact_transport,
            manifest_info=manifest_info,
            manifest=manifest,
        )

    @property
    def inventory(self) -> tuple[RemoteFeatureShard, ...]:
        return self.transport.inventory

    def __enter__(self) -> AuthenticatedTokenBoundarySource:
        self.transport.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.transport.__exit__(exc_type, exc, traceback)

    def stream(self) -> Iterator[PreparedTokenBoundaryShard]:
        return self.transport.stream()

    def final_inventory_fence(self) -> None:
        try:
            self.raw_transport.final_inventory_fence()
            self.transport.final_inventory_fence()
            observed_manifest = self.raw_transport.backend.stat(self.manifest_info.path)
            if observed_manifest != self.manifest_info:
                raise TokenBoundaryRemoteError(
                    "compact source manifest identity changed before token-boundary publication"
                )
        except BaseException:
            self._final_fence_status = "failed"
            raise
        self._final_fence_status = "passed"

    def profile_payload(self) -> dict[str, object]:
        payload = self.transport.profile_payload()
        compact_final_fence_status = payload["final_fence_status"]
        payload.update(
            {
                "raw_inventory_count": len(self.raw_transport.inventory),
                "raw_inventory_bytes": self.raw_transport.inventory_bytes,
                "raw_inventory_sha256": self.raw_transport.inventory_sha256,
                "compact_prefix": str(self.compact_prefix),
                "base_shard_content_hash": self.manifest.base_shard_content_hash,
                "source_manifest_path": self.manifest_info.path,
                "source_manifest_xet_hash": self.manifest_info.xet_hash,
                "compact_final_fence_status": compact_final_fence_status,
                "final_fence_status": self._final_fence_status,
            }
        )
        return payload


@contextmanager
def remote_output_lock(lock_path: Path) -> Iterator[None]:
    """Hold one nonblocking direct-file lock for a logical remote overlay run."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise TokenBoundaryRemoteError(
                f"remote token-boundary output lock is not a direct single-link file: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TokenBoundaryRemoteError(
                f"remote token-boundary output is already in use: {lock_path}"
            ) from exc
        yield
    finally:
        os.close(descriptor)


__all__ = [
    "AuthenticatedTokenBoundarySource",
    "PreparedTokenBoundaryShard",
    "RemoteTokenBoundaryTransport",
    "TokenBoundaryRemoteConfig",
    "TokenBoundaryRemoteError",
    "remote_output_lock",
    "required_staging_capacity_bytes",
]
