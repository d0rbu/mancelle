"""Bounded allocation-local storage recovery gate for Slurm retry launchers.

The driver performs no storage-path metadata operations. Each measurement runs
in worker mode through ``srun`` so login-node cache or routing cannot certify a
compute-node training path.
"""

import argparse
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_MIB = 1024 * 1024
_BULK_READ_BYTES = 64 * _MIB
_SMALL_READ_BYTES = 64 * 1024
_SMALL_FILE_COUNT = 20
_MIN_BULK_MB_S = 100.0
_MAX_SMALL_LOOP_S = 2.0
_PROBE_TIMEOUT_S = 60
_PROBE_KILL_GRACE_S = 15
_RETRY_DELAY_S = 300
_MAX_PROBES = 3
_READ_CHUNK_BYTES = _MIB


class _ProbeConfigError(ValueError):
    """The requested probe cannot produce a meaningful measurement."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file",
        required=True,
        help="Exact model_<step>.pt used by the matching resume attempt.",
    )
    parser.add_argument(
        "--selection-dir",
        required=True,
        help="Selection directory containing at least 20 shard_000*.zst files.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _read_exact(file_obj: Any, byte_count: int, *, path: Path) -> None:
    remaining = byte_count
    while remaining:
        chunk = file_obj.read(min(remaining, _READ_CHUNK_BYTES))
        if not chunk:
            raise _ProbeConfigError(
                f"probe source {path} ended before {byte_count} bytes were readable"
            )
        remaining -= len(chunk)


def _validate_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.stat()
    except (FileNotFoundError, PermissionError) as exc:
        raise _ProbeConfigError(f"{label} is not accessible: {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise _ProbeConfigError(f"{label} is not a regular file: {path}")
    return info


def _selection_probe_files(selection_dir: Path) -> list[Path]:
    try:
        info = selection_dir.stat()
    except (FileNotFoundError, PermissionError) as exc:
        raise _ProbeConfigError(
            f"selection directory is not accessible: {selection_dir}: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise _ProbeConfigError(f"selection directory is not a directory: {selection_dir}")
    paths = sorted(selection_dir.glob("shard_000*.zst"), key=lambda path: path.name)
    if len(paths) < _SMALL_FILE_COUNT:
        raise _ProbeConfigError(
            f"selection directory {selection_dir} has {len(paths)} shard_000*.zst "
            f"files; {_SMALL_FILE_COUNT} are required"
        )
    selected = paths[:_SMALL_FILE_COUNT]
    for path in selected:
        info = _validate_regular_file(path, label="selection shard")
        if info.st_size < _SMALL_READ_BYTES:
            raise _ProbeConfigError(
                f"selection shard {path} has {info.st_size} bytes; "
                f"at least {_SMALL_READ_BYTES} are required"
            )
    return selected


def _measure_storage(checkpoint_file: Path, selection_dir: Path) -> dict[str, Any]:
    checkpoint_info = _validate_regular_file(checkpoint_file, label="checkpoint file")
    if checkpoint_info.st_size < _BULK_READ_BYTES:
        raise _ProbeConfigError(
            f"checkpoint file {checkpoint_file} has {checkpoint_info.st_size} bytes; "
            f"at least {_BULK_READ_BYTES} are required"
        )
    offset_slots = (checkpoint_info.st_size - _BULK_READ_BYTES) // _MIB + 1
    offset = random.randrange(offset_slots) * _MIB
    bulk_started = time.monotonic()
    with checkpoint_file.open("rb", buffering=0) as checkpoint:
        checkpoint.seek(offset)
        _read_exact(checkpoint, _BULK_READ_BYTES, path=checkpoint_file)
    bulk_elapsed_s = time.monotonic() - bulk_started
    bulk_mb_s = _BULK_READ_BYTES / bulk_elapsed_s / 1_000_000

    # Include directory enumeration and per-file metadata in the small-op
    # measurement. Those operations were the incident's dominant failure mode.
    small_started = time.monotonic()
    selection_files = _selection_probe_files(selection_dir)
    for path in selection_files:
        with path.open("rb", buffering=0) as shard:
            _read_exact(shard, _SMALL_READ_BYTES, path=path)
    small_loop_s = time.monotonic() - small_started

    return {
        "status": "measurement",
        "checkpoint_file": os.fspath(checkpoint_file),
        "bulk_offset": offset,
        "bulk_bytes": _BULK_READ_BYTES,
        "bulk_elapsed_s": bulk_elapsed_s,
        "bulk_mb_s": bulk_mb_s,
        "small_files": len(selection_files),
        "small_read_bytes": _SMALL_READ_BYTES,
        "small_loop_s": small_loop_s,
    }


def _worker_main(checkpoint_file: Path, selection_dir: Path) -> int:
    try:
        payload = _measure_storage(checkpoint_file, selection_dir)
    except OSError as exc:
        payload = {
            "status": "probe_error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        exit_code = 1
    except Exception as exc:
        payload = {
            "status": "fatal_error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        exit_code = 2
    else:
        exit_code = 0
    print(json.dumps(payload, sort_keys=True), flush=True)
    return exit_code


def _worker_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("status"), str):
            return payload
    return None


def _run_compute_probe(
    checkpoint_file: Path,
    selection_dir: Path,
) -> dict[str, Any]:
    command = [
        "timeout",
        "--signal=TERM",
        f"--kill-after={_PROBE_KILL_GRACE_S}s",
        f"{_PROBE_TIMEOUT_S}s",
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--overlap",
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--checkpoint-file",
        os.fspath(checkpoint_file),
        "--selection-dir",
        os.fspath(selection_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "status": "fatal_error",
            "type": type(exc).__name__,
            "message": f"could not launch allocation-local storage probe: {exc}",
        }
    payload = _worker_payload(completed.stdout)
    if completed.returncode in {124, 137}:
        return {
            "status": "probe_error",
            "type": "Timeout",
            "message": (
                f"compute-node storage probe exceeded {_PROBE_TIMEOUT_S}s; "
                f"worker_status={payload.get('status') if payload is not None else None}"
            ),
        }
    if payload is None:
        return {
            "status": "fatal_error",
            "message": (
                f"storage probe exited {completed.returncode} without worker JSON; "
                f"stdout_tail={completed.stdout[-2000:]!r}; "
                f"stderr_tail={completed.stderr[-2000:]!r}"
            ),
        }
    if payload["status"] == "measurement" and completed.returncode != 0:
        return {
            "status": "fatal_error",
            "message": (
                "worker emitted a measurement but the allocation-local launcher "
                f"exited {completed.returncode}; stderr_tail={completed.stderr[-2000:]!r}"
            ),
        }
    return payload


def _measurement_is_healthy(payload: dict[str, Any]) -> bool:
    try:
        bulk_raw = payload["bulk_mb_s"]
        small_raw = payload["small_loop_s"]
    except KeyError as exc:
        raise _ProbeConfigError(f"worker measurement is malformed: {payload!r}") from exc
    if type(bulk_raw) not in {int, float} or type(small_raw) not in {int, float}:
        raise _ProbeConfigError(f"worker measurement is malformed: {payload!r}")
    bulk_mb_s = float(bulk_raw)
    small_loop_s = float(small_raw)
    if not math.isfinite(bulk_mb_s) or bulk_mb_s < 0:
        raise _ProbeConfigError(f"worker bulk_mb_s is invalid: {bulk_mb_s!r}")
    if not math.isfinite(small_loop_s) or small_loop_s < 0:
        raise _ProbeConfigError(f"worker small_loop_s is invalid: {small_loop_s!r}")
    return bulk_mb_s >= _MIN_BULK_MB_S and small_loop_s < _MAX_SMALL_LOOP_S


def _require_slurm_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise _ProbeConfigError("storage recovery gate must run inside a Slurm allocation")


def _driver_main(checkpoint_file: Path, selection_dir: Path) -> int:
    try:
        _require_slurm_allocation()
    except _ProbeConfigError as exc:
        print(f"storage_gate_status=fatal_error message={exc}", file=sys.stderr, flush=True)
        return 2

    for attempt in range(1, _MAX_PROBES + 1):
        print(f"storage probe attempt={attempt}/{_MAX_PROBES}", flush=True)
        payload = _run_compute_probe(checkpoint_file, selection_dir)
        status = payload.get("status")
        if status == "measurement":
            try:
                healthy = _measurement_is_healthy(payload)
            except _ProbeConfigError as exc:
                print(
                    f"storage_gate_status=fatal_error message={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            print(json.dumps(payload, sort_keys=True), flush=True)
            if healthy:
                print("storage_gate_status=healthy", flush=True)
                return 0
        elif status == "probe_error":
            print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        elif status == "fatal_error":
            print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
            print(f"storage_gate_status={status}", file=sys.stderr, flush=True)
            return 2
        else:
            print(
                f"storage_gate_status=fatal_error unknown_payload={payload!r}",
                file=sys.stderr,
                flush=True,
            )
            return 2

        if attempt < _MAX_PROBES:
            print(
                f"storage_gate_status=unhealthy retry_in_s={_RETRY_DELAY_S}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(_RETRY_DELAY_S)

    print(
        "storage_gate_status=exhausted_proceeding "
        f"probes={_MAX_PROBES} threshold_bulk_mb_s={_MIN_BULK_MB_S:g} "
        f"threshold_small_s={_MAX_SMALL_LOOP_S:g}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint_file = Path(os.path.abspath(args.checkpoint_file))
    selection_dir = Path(os.path.abspath(args.selection_dir))
    if args.worker:
        return _worker_main(checkpoint_file, selection_dir)
    return _driver_main(checkpoint_file, selection_dir)


if __name__ == "__main__":
    raise SystemExit(main())
