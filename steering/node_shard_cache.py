"""Budget-capped node-local cache for compressed packed/selection shards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nanochat.atomic_io import atomic_write_bytes
from steering.refined_types import (
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
)

_CACHE_SCHEMA_VERSION = 1
_CACHE_KEY_HEX_LENGTH = 16
_CACHE_SUMMARY_EVERY = 50
_CACHE_SKIP_WARN_EVERY = 100
_ORPHAN_TMP_MAX_AGE_S = 24 * 60 * 60
_TMP_NAME_RE = re.compile(r"^.+\.tmp\.(?P<pid>[0-9]+)\.(?P<nonce>[0-9a-f]{8})$")


def _tmp_path_for(final_path: str | os.PathLike[str]) -> Path:
    """Build the cache's ``<final>.tmp.<pid>.<uuid8>`` temporary path."""
    final = Path(final_path)
    suffix = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return final.parent / f"{final.name}.tmp.{suffix}"


@dataclass(frozen=True)
class NodeShardCacheConfig:
    """Resolved operational settings for one node-local shard cache."""

    dir: Path
    max_bytes: PositiveInt
    reserve_bytes: NonNegativeInt


@beartype
def resolve_train_node_cache(
    raw_dir: str | Path | None,
    raw_max_bytes: PositiveIntLike | None,
    raw_reserve_bytes: NonNegativeIntLike | None,
) -> NodeShardCacheConfig | None:
    """Resolve node-cache settings at the training config boundary."""
    if raw_dir is None:
        return None
    if isinstance(raw_dir, str) and not raw_dir.strip():
        raise ValueError("train_node_cache_dir must not be blank or whitespace-only")
    if raw_max_bytes is None or raw_reserve_bytes is None:
        raise ValueError(
            "train_node_cache_max_bytes and train_node_cache_reserve_bytes are required "
            "when train_node_cache_dir is set"
        )
    return NodeShardCacheConfig(
        dir=Path(raw_dir),
        max_bytes=PositiveInt.parse(raw_max_bytes),
        reserve_bytes=NonNegativeInt.parse(raw_reserve_bytes),
    )


@dataclass(frozen=True)
class NodeShardCacheFill:
    """One admitted worker-tee fill, bound to the source stat at reservation."""

    source_abspath: str
    source_size: int
    source_mtime_ns: int
    payload_path: Path
    tmp_path: Path


class NodeLocalShardCache:
    """Read-through cache of source shard compressed bytes on local storage."""

    def __init__(self, config: NodeShardCacheConfig):
        self.config = config
        self.dir = config.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sweep_orphan_tmps()
        self._lock = threading.Lock()
        self._events = 0
        self._counters = {
            "hits": 0,
            "misses": 0,
            "fills": 0,
            "fill_skips": 0,
            "verify_failures": 0,
            "fallbacks": 0,
        }

    def counters(self) -> dict[str, int]:
        """Return a consistent snapshot of self-contained cache counters."""
        with self._lock:
            return dict(self._counters)

    def _record(self, **deltas: int) -> dict[str, int]:
        with self._lock:
            for key, delta in deltas.items():
                self._counters[key] += delta
            self._events += 1
            emit_summary = self._events % _CACHE_SUMMARY_EVERY == 0
            snapshot = dict(self._counters)
        if emit_summary:
            logger.info(
                "node shard cache: "
                f"hits={snapshot['hits']} misses={snapshot['misses']} "
                f"fills={snapshot['fills']} fill_skips={snapshot['fill_skips']} "
                f"verify_failures={snapshot['verify_failures']} "
                f"fallbacks={snapshot['fallbacks']}"
            )
        return snapshot

    def record_verify_failure(self) -> None:
        """Record a post-verification cache failure observed by a consumer."""
        self._record(verify_failures=1)

    def record_shared_decode_fallback(self) -> None:
        """Record a shared-decode consumer fallback without loader-health coupling."""
        self._record(fallbacks=1)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OverflowError:
            # A corrupt decimal PID can exceed the platform pid_t range. Keep
            # it conservatively until the ordinary temporary age ceiling.
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            # Conservatively preserve the temporary when liveness cannot be
            # established as dead.
            return True
        return True

    def _sweep_orphan_tmps(self) -> None:
        """Remove temporaries with dead PIDs or beyond the age ceiling."""
        now_ns = time.time_ns()
        swept = 0
        with os.scandir(self.dir) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False) or ".tmp." not in entry.name:
                    continue
                match = _TMP_NAME_RE.fullmatch(entry.name)
                pid = int(match.group("pid")) if match is not None else 0
                should_sweep = pid > 0 and not self._pid_is_alive(pid)
                if not should_sweep:
                    try:
                        age_ns = now_ns - entry.stat(follow_symlinks=False).st_mtime_ns
                    except OSError as exc:
                        logger.warning(
                            f"node shard cache could not inspect temporary {entry.path}: {exc}"
                        )
                        continue
                    should_sweep = age_ns > _ORPHAN_TMP_MAX_AGE_S * 1_000_000_000
                if not should_sweep:
                    continue
                try:
                    Path(entry.path).unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning(
                        f"node shard cache could not sweep temporary {entry.path}: {exc}"
                    )
                    continue
                swept += 1
        if swept:
            logger.info(f"node shard cache swept {swept} orphan temporary file(s)")

    def _entry_paths(self, path: str | os.PathLike[str]) -> tuple[str, Path, Path]:
        source_abspath = os.path.abspath(os.fspath(path))
        digest = hashlib.sha256(source_abspath.encode()).hexdigest()[:_CACHE_KEY_HEX_LENGTH]
        payload = self.dir / f"{digest}_{Path(source_abspath).name}"
        return source_abspath, payload, payload.with_name(f"{payload.name}.meta.json")

    @staticmethod
    def _metadata(
        *,
        source_abspath: str,
        source_size: int,
        source_mtime_ns: int,
        sha256: str,
        byte_length: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "source_abspath": source_abspath,
            "source_size": source_size,
            "source_mtime_ns": source_mtime_ns,
            "sha256": sha256,
            "byte_length": byte_length,
        }

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(f"node shard cache could not unlink {path}: {exc}")

    def _invalidate(self, payload: Path, sidecar: Path) -> None:
        # The sidecar is the commit marker, so remove it before the payload.
        self._unlink(sidecar)
        self._unlink(payload)

    @staticmethod
    def _validate_metadata(metadata: object, source_abspath: str) -> dict[str, Any]:
        expected_keys = {
            "schema_version",
            "source_abspath",
            "source_size",
            "source_mtime_ns",
            "sha256",
            "byte_length",
        }
        if (
            not isinstance(metadata, dict)
            or set(metadata) != expected_keys
            or metadata["schema_version"] != _CACHE_SCHEMA_VERSION
            or metadata["source_abspath"] != source_abspath
            or any(
                type(metadata[key]) is not int or metadata[key] < 0
                for key in ("source_size", "source_mtime_ns", "byte_length")
            )
            or not isinstance(metadata["sha256"], str)
            or len(metadata["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in metadata["sha256"])
        ):
            raise ValueError("invalid node shard cache sidecar metadata")
        return metadata

    def _verified_payload(
        self,
        path: str | os.PathLike[str],
        *,
        return_bytes: bool,
    ) -> bytes | Path | None:
        source_abspath, payload, sidecar = self._entry_paths(path)
        try:
            sidecar.stat()
        except FileNotFoundError:
            self._record(misses=1)
            return None
        except OSError as exc:
            logger.warning(
                f"node shard cache sidecar handling failed for {source_abspath}: {exc}; "
                "invalidating and treating as a miss"
            )
            self._invalidate(payload, sidecar)
            self._record(misses=1, verify_failures=1)
            return None

        try:
            metadata = self._validate_metadata(
                json.loads(sidecar.read_text(encoding="utf-8")),
                source_abspath,
            )
        except Exception as exc:
            logger.warning(
                f"node shard cache sidecar handling failed for {source_abspath}: {exc}; "
                "invalidating and treating as a miss"
            )
            self._invalidate(payload, sidecar)
            self._record(misses=1, verify_failures=1)
            return None

        try:
            source_stat = os.stat(source_abspath)
        except OSError as exc:
            logger.warning(
                f"node shard cache could not stat source {source_abspath}: {exc}; "
                "preserving the cache entry and treating as a miss"
            )
            self._record(misses=1, verify_failures=1)
            return None
        if (
            source_stat.st_size != metadata["source_size"]
            or source_stat.st_mtime_ns != metadata["source_mtime_ns"]
        ):
            logger.warning(
                "node shard cache source changed; invalidating "
                f"{payload} for {source_abspath}"
            )
            self._invalidate(payload, sidecar)
            self._record(misses=1, verify_failures=1)
            return None

        try:
            if return_bytes:
                compressed = payload.read_bytes()
                byte_length = len(compressed)
                digest = hashlib.sha256(compressed).hexdigest()
                result: bytes | Path = compressed
            else:
                byte_length = payload.stat().st_size
                digest_builder = hashlib.sha256()
                with payload.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest_builder.update(chunk)
                digest = digest_builder.hexdigest()
                result = payload
            if byte_length != metadata["byte_length"] or digest != metadata["sha256"]:
                logger.warning(
                    "node shard cache payload verification failed; invalidating "
                    f"{payload}"
                )
                self._invalidate(payload, sidecar)
                self._record(misses=1, verify_failures=1)
                return None
        except Exception as exc:
            logger.warning(
                f"node shard cache payload handling failed for {source_abspath}: {exc}; "
                "invalidating and treating as a miss"
            )
            self._invalidate(payload, sidecar)
            self._record(misses=1, verify_failures=1)
            return None
        self._record(hits=1)
        return result

    def read(self, path: str | os.PathLike[str]) -> bytes | None:
        """Return verified cached compressed bytes, or a loud non-raising miss."""
        result = self._verified_payload(path, return_bytes=True)
        assert result is None or isinstance(result, bytes)
        return result

    def read_path(self, path: str | os.PathLike[str]) -> Path | None:
        """Return a verified local payload path for the shared-decode worker."""
        result = self._verified_payload(path, return_bytes=False)
        assert result is None or isinstance(result, Path)
        return result

    def _cache_bytes(self) -> int:
        with os.scandir(self.dir) as entries:
            return sum(
                entry.stat(follow_symlinks=False).st_size
                for entry in entries
                if entry.is_file(follow_symlinks=False)
            )

    def _skip_fill(self, reason: str) -> None:
        skip_count = self._record(fill_skips=1)["fill_skips"]
        if skip_count == 1 or skip_count % _CACHE_SKIP_WARN_EVERY == 0:
            logger.warning(f"node shard cache skipped fill #{skip_count}: {reason}")

    def _admit(self, incoming_bytes: int, *, payload: Path) -> bool:
        try:
            # This admission pre-counts only the payload; its sidecar enters the next total.
            cache_bytes = self._cache_bytes()
            existing_bytes = payload.stat().st_size if payload.is_file() else 0
            final_cache_bytes = cache_bytes - existing_bytes + incoming_bytes
            if final_cache_bytes > self.config.max_bytes:
                self._skip_fill(
                    f"cache-directory budget would be {final_cache_bytes} bytes, "
                    f"above max_bytes={self.config.max_bytes}"
                )
                return False
            fs = os.statvfs(self.dir)
            free_bytes = fs.f_bavail * fs.f_frsize
            if free_bytes - incoming_bytes < self.config.reserve_bytes:
                self._skip_fill(
                    f"free space after fill would be {free_bytes - incoming_bytes} bytes, "
                    f"below reserve_bytes={self.config.reserve_bytes}"
                )
                return False
        except OSError as exc:
            self._skip_fill(f"admission check failed for {self.dir}: {exc}")
            return False
        return True

    def _publish_prepared_tmp(
        self,
        *,
        tmp_path: Path,
        payload: Path,
        metadata: dict[str, Any],
    ) -> None:
        sidecar = payload.with_name(f"{payload.name}.meta.json")
        self._unlink(sidecar)
        os.replace(tmp_path, payload)
        atomic_write_bytes(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
            sidecar,
        )
        self._record(fills=1)

    def offer(
        self,
        path: str | os.PathLike[str],
        compressed_bytes: bytes,
        source_stat: os.stat_result,
    ) -> None:
        """Best-effort admission and atomic fill using already-read source bytes."""
        source_abspath, payload, _sidecar = self._entry_paths(path)
        tmp_path = _tmp_path_for(payload)
        try:
            if not self._admit(len(compressed_bytes), payload=payload):
                return
            with tmp_path.open("wb") as handle:
                handle.write(compressed_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            after_stat = os.stat(source_abspath)
            identity_matches = (
                after_stat.st_size == source_stat.st_size
                and after_stat.st_mtime_ns == source_stat.st_mtime_ns
            )
            if not identity_matches or len(compressed_bytes) != source_stat.st_size:
                logger.warning(
                    "node shard cache offered source identity changed or byte length "
                    f"mismatched; discarding {tmp_path} for {source_abspath}"
                )
                self._record(verify_failures=1, fill_skips=1)
                return
            metadata = self._metadata(
                source_abspath=source_abspath,
                source_size=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
                sha256=hashlib.sha256(compressed_bytes).hexdigest(),
                byte_length=len(compressed_bytes),
            )
            self._publish_prepared_tmp(
                tmp_path=tmp_path,
                payload=payload,
                metadata=metadata,
            )
        except Exception as exc:
            self._skip_fill(f"fill failed for {source_abspath}: {exc}")
        finally:
            self._unlink(tmp_path)

    def reserve_fill_tmp(
        self,
        path: str | os.PathLike[str],
    ) -> NodeShardCacheFill | None:
        """Admit a worker tee and reserve its unique local temporary path."""
        source_abspath, payload, _sidecar = self._entry_paths(path)
        try:
            source_stat = os.stat(source_abspath)
            if not self._admit(source_stat.st_size, payload=payload):
                return None
            return NodeShardCacheFill(
                source_abspath=source_abspath,
                source_size=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
                payload_path=payload,
                tmp_path=_tmp_path_for(payload),
            )
        except Exception as exc:
            self._skip_fill(f"worker-tee reservation failed for {source_abspath}: {exc}")
            return None

    def discard_fill_tmp(self, fill: NodeShardCacheFill) -> None:
        """Best-effort cleanup for an unsuccessful worker tee."""
        self._unlink(fill.tmp_path)

    def commit_fill_tmp(self, fill: NodeShardCacheFill) -> None:
        """Verify and atomically publish a successful worker tee temporary."""
        try:
            source_stat = os.stat(fill.source_abspath)
            if (
                source_stat.st_size != fill.source_size
                or source_stat.st_mtime_ns != fill.source_mtime_ns
            ):
                logger.warning(
                    "node shard cache worker-tee source changed before commit; "
                    f"discarding {fill.tmp_path}"
                )
                self._record(verify_failures=1, fill_skips=1)
                self._unlink(fill.tmp_path)
                return
            tmp_size = fill.tmp_path.stat().st_size
            if tmp_size != fill.source_size:
                logger.warning(
                    "node shard cache worker-tee size mismatch: "
                    f"tmp={tmp_size}, source={fill.source_size}; discarding {fill.tmp_path}"
                )
                self._record(verify_failures=1, fill_skips=1)
                self._unlink(fill.tmp_path)
                return
            digest_builder = hashlib.sha256()
            with fill.tmp_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_builder.update(chunk)
                os.fsync(handle.fileno())
            metadata = self._metadata(
                source_abspath=fill.source_abspath,
                source_size=fill.source_size,
                source_mtime_ns=fill.source_mtime_ns,
                sha256=digest_builder.hexdigest(),
                byte_length=tmp_size,
            )
            self._publish_prepared_tmp(
                tmp_path=fill.tmp_path,
                payload=fill.payload_path,
                metadata=metadata,
            )
        except Exception as exc:
            self._skip_fill(f"worker-tee commit failed for {fill.source_abspath}: {exc}")
        finally:
            self._unlink(fill.tmp_path)
