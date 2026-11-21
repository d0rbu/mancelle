# steering/staging/relocate.py
"""EXDEV-safe relocation of finished shards from local scratch to the network dir.

Extraction writes each shard atomically to fast LOCAL scratch, then relocate()
moves it to the canonical network output_dir. Scratch and output are on different
filesystems (local NVMe vs MooseFS), so a direct os.replace would raise EXDEV.
relocate() copies bytes to a temp ON THE DESTINATION FS, fsyncs, and atomically
renames within that FS — never a cross-FS rename. Same-FS callers (local dev/CI)
get a pure rename. Destination ``ENOSPC``/``EDQUOT`` failures become
``DestinationSpaceError`` so the destination path is explicit and the source is
left available for diagnosis.
"""
from __future__ import annotations

import errno
import os
import shutil
import time
import uuid
from pathlib import Path

from loguru import logger


class DestinationSpaceError(OSError):
    """A destination filesystem rejected a staging operation for lack of space."""

    def __init__(self, destination: Path, cause: OSError) -> None:
        self.destination = destination
        super().__init__(
            cause.errno,
            f"staging: destination {destination} cannot accept the shard: "
            f"{cause.strerror or cause}",
        )


def raise_destination_space_error(destination: Path, cause: OSError) -> None:
    """Raise a destination-specific error for quota/space failures, if applicable."""
    if cause.errno in (errno.ENOSPC, errno.EDQUOT):
        raise DestinationSpaceError(destination, cause) from cause


def free_gb(path: str | os.PathLike[str]) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def sweep_dir(d: Path, *, swallow_errors: bool = False) -> None:
    """Remove top-level regular files in d (crash leftovers; canonical files live elsewhere).

    A missing directory is always a quiet no-op. Other listing errors (PermissionError,
    a stale NFS/MooseFS handle, d being a file) re-raise by default — so a misconfigured
    scratch fails fast at startup. Pass ``swallow_errors=True`` only in a teardown
    ``finally`` (StagedSaver/StagedPrefetcher.__exit__) where a sweep error must not mask
    the real exception that triggered teardown.
    """
    try:
        entries = list(d.iterdir())
    except FileNotFoundError:
        return
    except OSError as e:
        if not swallow_errors:
            raise
        logger.warning("staging: cannot list scratch {} to sweep: {!r}", d, e)
        return
    for p in entries:
        if p.is_file():
            try:
                p.unlink()
            except OSError as e:
                logger.warning("staging: failed to sweep {}: {!r}", p, e)


def prepare_scratch_dir(scratch_dir: str | os.PathLike[str]) -> Path:
    """Create the scratch directory and sweep any stale crash leftovers (files only).

    Shared startup prep for StagedSaver and StagedPrefetcher. Fails fast (raises) on a
    misconfigured scratch path — a regular file (FileExistsError from mkdir) or an
    unlistable directory (sweep_dir re-raises).
    """
    p = Path(scratch_dir)
    p.mkdir(parents=True, exist_ok=True)
    sweep_dir(p)
    return p


def _tmp_path_for(final: Path) -> Path:
    # Suffix AFTER the full name so the temp never matches '*_features.safetensors.zst'.
    return final.parent / f"{final.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"


def _cleanup_tmp(tmp: Path) -> None:
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("staging: failed to clean temp {}: {!r}", tmp, e)


def _same_filesystem(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def relocate(src_local: Path, final: Path) -> None:
    """Atomically move src_local to final w.r.t. final's filesystem.

    Same-FS: zero-copy os.replace. Cross-FS: copy -> temp on final's FS -> fsync ->
    os.replace -> unlink src. On failure: temp cleaned, src RETAINED, exception re-raised.
    """
    src_local = Path(src_local)
    final = Path(final)
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise_destination_space_error(final.parent, e)
        raise

    if _same_filesystem(src_local, final.parent):
        try:
            os.replace(src_local, final)
            return
        except OSError as e:
            # st_dev matched but the mount still rejects a cross-directory rename
            # (union/overlay/FUSE) — fall through to the copy path rather than abort.
            if e.errno != errno.EXDEV:
                raise_destination_space_error(final.parent, e)
                raise

    tmp = _tmp_path_for(final)
    try:
        shutil.copyfile(src_local, tmp)  # os.sendfile (kernel zero-copy) on Linux
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, final)  # atomic, within the destination FS
    except OSError as e:
        _cleanup_tmp(tmp)
        raise_destination_space_error(final.parent, e)
        raise
    except BaseException:
        _cleanup_tmp(tmp)
        raise
    try:
        src_local.unlink()
    except OSError as e:
        logger.warning("staging: moved {} but failed to remove source {}: {!r}", final, src_local, e)


def assert_startup_free_space(
    scratch_dir: str | os.PathLike[str],
    *,
    output_depth: int,
    est_shard_gb: float,
    local_world_size: int,
    min_free_gb: float,
    resource: str = "scratch",
) -> None:
    """Hard-error at startup if a shared filesystem can't fit one wave + floor.

    ``resource`` labels the filesystem in the diagnostic (scratch or destination).
    """
    need = max(output_depth, 1) * est_shard_gb * max(local_world_size, 1) + min_free_gb
    have = free_gb(scratch_dir)
    if have < need:
        raise RuntimeError(
            f"staging: {resource} {scratch_dir} has {have:.1f} GB free but needs ~{need:.1f} GB "
            f"(output_depth={output_depth} x {est_shard_gb} GB x {local_world_size} ranks + "
            f"{min_free_gb} GB floor). Lower staging.output_depth or free disk."
        )


def wait_for_free_space(
    scratch_dir: str | os.PathLike[str],
    *,
    min_free_gb: float,
    timeout_s: float = 600.0,
    poll_s: float = 2.0,
    warn_interval_s: float = 30.0,
    resource: str = "scratch",
) -> None:
    """Block until free space >= min_free_gb, else raise.

    Logs a warning every ``warn_interval_s`` while blocked so a long stall is visible
    rather than a silent freeze (a stuck relocate can otherwise hold up teardown for
    the full ``timeout_s`` with no output).
    """
    start = time.monotonic()
    deadline = start + timeout_s
    warned_at = start
    while free_gb(scratch_dir) < min_free_gb:
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"staging: {resource} {scratch_dir} below {min_free_gb} GB free for {timeout_s}s — "
                "space did not recover; aborting to avoid disk exhaustion."
            )
        if now - warned_at >= warn_interval_s:
            logger.warning(
                "staging: {} {} below {:.1f} GB free for {:.0f}s — blocking save "
                "until space recovers", resource, scratch_dir, min_free_gb, now - start,
            )
            warned_at = now
        time.sleep(poll_s)


def coordinate_free_space_check(
    run_scratch: Path,
    *,
    local_rank: int,
    run_id: str,
    output_depth: int,
    est_shard_gb: float,
    local_world_size: int,
    min_free_gb: float,
    destination_dir: Path | None = None,
    timeout_s: float = 1800.0,
    poll_s: float = 1.0,
) -> None:
    """Rank-0-gated startup free-space check, coordinated via a file sentinel.

    The extraction ranks are independent processes with no ``torch.distributed`` group,
    so per-rank checks raced on staggered model-load finish times (double-counting
    in-flight scratch and falsely aborting). Instead local rank 0 is the SOLE checker on
    the shared scratch FS and signals the verdict; the other local ranks wait for it and
    fail if it reports a shortage — so all ranks fail fast and cleanly on a genuinely too
    small scratch, with no race. When ``destination_dir`` is provided, rank 0 applies
    the same estimate to that filesystem before signaling success; failures include the
    path-specific diagnosis in the sentinel for waiting ranks.

    The sentinel name carries a per-launch ``run_id`` (e.g. ``TORCHELASTIC_RUN_ID``) so a
    stale sentinel left by a previous run to the same scratch is never read as this
    launch's verdict. ``timeout_s`` is generous because rank 0 may be the model-load
    straggler the others are waiting on.
    """
    ok = run_scratch / f".startup_ok.{run_id}"
    fail = run_scratch / f".startup_fail.{run_id}"
    if local_rank == 0:
        for stale in run_scratch.glob(".startup_*"):  # don't let prior launches' sentinels accumulate
            try:
                stale.unlink()
            except OSError:
                pass
        try:
            assert_startup_free_space(
                run_scratch, output_depth=output_depth, est_shard_gb=est_shard_gb,
                local_world_size=local_world_size, min_free_gb=min_free_gb,
            )
            if destination_dir is not None:
                assert_startup_free_space(
                    destination_dir, output_depth=output_depth, est_shard_gb=est_shard_gb,
                    local_world_size=local_world_size, min_free_gb=min_free_gb,
                    resource="destination",
                )
        except BaseException as e:
            try:
                fail.write_text(str(e))  # signal the others to fail fast with the diagnosis
            except OSError:
                pass
            raise
        ok.write_text("ok")
        return
    deadline = time.monotonic() + timeout_s
    while True:
        if ok.exists():
            return
        if fail.exists():
            try:
                detail = fail.read_text().strip()
            except OSError:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"staging: local rank 0 reported insufficient scratch in {run_scratch}{suffix}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"staging: local rank 0 gave no free-space verdict within {timeout_s:.0f}s "
                f"for {run_scratch}; aborting."
            )
        time.sleep(poll_s)
