"""Shared atomic-write helpers.

Canonical implementation of the tmp + fsync + os.replace pattern originally
introduced in ``nanochat/checkpoint_manager.py`` (A2 audit) and now propagated
across every writer in the repo that persists training / precompute state.

All helpers follow the same contract:

1. Write bytes to a unique tmp path:
   ``<final>.tmp.<pid>.<uuid-suffix>``. The PID + UUID suffix prevents
   concurrent DDP writers from colliding on the same tmp path (a silent
   data-loss vector when rank 0 and rank 1 both try to persist to the same
   path from forked children).
2. fsync the tmp file so bytes reach disk BEFORE the rename.
3. ``os.replace(tmp, final)`` — atomic on POSIX when tmp and final share a
   filesystem (they always do here because tmp is derived from final.parent).
4. On any exception (including ``KeyboardInterrupt``), best-effort unlink the
   tmp file and re-raise. The final path is never torn: either the old file
   survives intact, or the new file appears whole.

This module is used by feature, statistics, checksum, overlay, and training
state writers so they all share the same crash-safe persistence behavior.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file as _safetensors_save_file


def _tmp_path_for(final_path: str | os.PathLike[str]) -> Path:
    """Build a unique tmp path beside *final_path*.

    The ``<pid>.<uuid8>`` suffix guarantees two concurrent writers (e.g. two
    DDP ranks or two Hydra multi-run shards) do not stomp each other's tmp
    file. Without this, rank 0's crash cleanup could unlink rank 1's in-flight
    tmp and vice versa.
    """
    final = Path(final_path)
    suffix = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return final.parent / f"{final.name}.tmp.{suffix}"


def _fsync_path(path: Path) -> None:
    """Open *path* read-only and fsync its file descriptor."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _cleanup_tmp(tmp_path: Path) -> None:
    """Best-effort unlink of a tmp file; swallow OSError."""
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def atomic_torch_save(obj: Any, final_path: str | os.PathLike[str]) -> None:
    """Atomically persist *obj* to *final_path* via ``torch.save``.

    A crash mid-write leaves at most a ``.tmp.<pid>.<uuid>`` debris file, never
    a torn final file. Callers that glob for the final path (e.g.
    ``find_last_step`` globbing ``model_*.pt``) can rely on the invariant that
    any file at the final path is complete.
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(final)
    try:
        torch.save(obj, str(tmp))
        _fsync_path(tmp)
        os.replace(str(tmp), str(final))
    except BaseException:
        _cleanup_tmp(tmp)
        raise


def atomic_json_dump(obj: Any, final_path: str | os.PathLike[str]) -> None:
    """Atomically persist *obj* as JSON (indent=2) to *final_path*."""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(final)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(final))
    except BaseException:
        _cleanup_tmp(tmp)
        raise


def atomic_write_bytes(data: bytes, final_path: str | os.PathLike[str]) -> None:
    """Atomically write raw *data* bytes to *final_path*."""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(final)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(final))
    except BaseException:
        _cleanup_tmp(tmp)
        raise

def atomic_write_text(
    text: str,
    final_path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically write *text* to *final_path* via tmp + fsync + replace."""
    atomic_write_bytes(text.encode(encoding), final_path)


def atomic_safetensors_save(
    tensors: dict[str, torch.Tensor],
    final_path: str | os.PathLike[str],
    metadata: dict[str, str] | None = None,
) -> None:
    """Atomically write *tensors* as a safetensors file to *final_path*.

    Uses ``safetensors.torch.save_file`` directly on the tmp path (rather than
    ``save`` + ``write_bytes``) so zero-copy serialization works on large
    tensor shards without materializing the whole buffer in memory twice.
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(final)
    try:
        _safetensors_save_file(tensors, str(tmp), metadata=metadata or {})
        _fsync_path(tmp)
        os.replace(str(tmp), str(final))
    except BaseException:
        _cleanup_tmp(tmp)
        raise


def atomic_write_parquet(
    table: pa.Table,
    final_path: str | os.PathLike[str],
) -> None:
    """Atomically write a pyarrow ``Table`` to *final_path* as Parquet.

    Uses tmp + fsync + ``os.replace`` so a crash mid-write leaves at most a
    ``.tmp.<pid>.<uuid>`` debris file (cleaned up on the next call's
    ``except`` branch), never a torn final parquet that would fail
    validation on read.

    pyarrow is already a hard dependency of ``nanochat.dataloader`` and
    ``nanochat.dataset`` so the unconditional import here does not
    enlarge the dependency surface.
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(final)
    try:
        pq.write_table(table, str(tmp))
        _fsync_path(tmp)
        os.replace(str(tmp), str(final))
    except BaseException:
        _cleanup_tmp(tmp)
        raise
