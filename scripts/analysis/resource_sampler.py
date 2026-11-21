"""Linux process-tree resource accounting for reproducible benchmarks.

RSS cannot answer whether a shared-store configuration fits: every process
mapping the same tmpfs pages reports those pages in RSS.  This module records
the process tree's proportional and private memory from ``smaps_rollup``
together with host, cgroup, tmpfs, paging, pressure, and deleted-mapping state.
It intentionally keeps raw samples; summaries are derived data and never the
only retained evidence.

The sampler is diagnostics-only.  It neither mutates the measured process nor
enters selection hashes/manifests.  A missing kernel surface is recorded as an
error in the sample rather than silently replaced with a different metric.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from subprocess import Popen
from typing import IO, Any

import psutil

RESOURCE_SAMPLE_SCHEMA_VERSION = 1
_SMAPS_KB_FIELDS = (
    "Rss",
    "Pss",
    "Pss_Dirty",
    "Shared_Clean",
    "Shared_Dirty",
    "Private_Clean",
    "Private_Dirty",
    "Swap",
    "SwapPss",
)
_STATUS_KB_FIELDS = ("VmRSS", "VmSwap")
_STATUS_INT_FIELDS = (
    "Threads",
    "voluntary_ctxt_switches",
    "nonvoluntary_ctxt_switches",
)
_PROC_IO_FIELDS = (
    "rchar",
    "wchar",
    "syscr",
    "syscw",
    "read_bytes",
    "write_bytes",
    "cancelled_write_bytes",
)
_TERMINAL_PROCESS_STATES = frozenset({"Z", "X", "x"})


def parse_kb_mapping(text: str, fields: Iterable[str]) -> dict[str, int | None]:
    """Parse selected ``Key: N kB`` fields, preserving missing as ``None``."""
    wanted = set(fields)
    parsed: dict[str, int | None] = dict.fromkeys(wanted)
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep or key not in wanted:
            continue
        parts = raw.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise ValueError(f"{key}: expected '<integer> kB', got {raw.strip()!r}")
        parsed[key] = int(parts[0])
    return parsed


def parse_smaps_rollup(text: str) -> dict[str, int | None]:
    """Parse memory fields from one Linux ``smaps_rollup`` file."""
    raw = parse_kb_mapping(text, _SMAPS_KB_FIELDS)
    private_clean = raw["Private_Clean"]
    private_dirty = raw["Private_Dirty"]
    raw["Uss"] = (
        private_clean + private_dirty
        if private_clean is not None and private_dirty is not None
        else None
    )
    return {f"{key}_kB": value for key, value in raw.items()}


def parse_proc_status(text: str) -> dict[str, int | None]:
    """Parse status memory, thread, and context-switch fields."""
    result = {
        f"{key}_kB": value
        for key, value in parse_kb_mapping(text, _STATUS_KB_FIELDS).items()
    }
    ints: dict[str, int | None] = dict.fromkeys(_STATUS_INT_FIELDS)
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if sep and key in ints:
            ints[key] = int(raw.strip())
    result.update(ints)
    return result


def parse_proc_stat(text: str) -> dict[str, int | str]:
    """Parse identity, faults, CPU ticks, and start time from ``/proc/PID/stat``.

    The command name is parenthesized and may itself contain spaces or closing
    parentheses, so splitting the whole line is incorrect.  Linux places the
    final command-name ``)`` immediately before field 3.
    """
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx <= 0 or close_idx <= open_idx or close_idx + 2 > len(text):
        raise ValueError(f"malformed /proc stat line: {text!r}")
    pid = int(text[:open_idx].strip())
    comm = text[open_idx + 1 : close_idx]
    tail = text[close_idx + 2 :].split()
    if len(tail) < 20:
        raise ValueError(f"truncated /proc stat line for pid {pid}: {len(tail)} fields after comm")
    return {
        "pid": pid,
        "comm": comm,
        "state": tail[0],
        "ppid": int(tail[1]),
        "minflt": int(tail[7]),
        "majflt": int(tail[9]),
        "utime_ticks": int(tail[11]),
        "stime_ticks": int(tail[12]),
        "starttime_ticks": int(tail[19]),
    }


def parse_proc_io(text: str) -> dict[str, int | None]:
    """Parse byte/operation counters from ``/proc/PID/io``."""
    result: dict[str, int | None] = dict.fromkeys(_PROC_IO_FIELDS)
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if sep and key in result:
            result[key] = int(raw.strip())
    return result


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse all numeric ``/proc/meminfo`` fields into kB values."""
    result: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        parts = raw.split()
        if not parts:
            continue
        if len(parts) == 2 and parts[1] != "kB":
            continue
        result[f"{key}_kB"] = int(parts[0])
    return result


def parse_vmstat(text: str) -> dict[str, int]:
    """Parse whitespace-separated ``/proc/vmstat`` counters."""
    result: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"malformed vmstat line: {line!r}")
        result[parts[0]] = int(parts[1])
    return result


def parse_proc_cpu_stat(text: str) -> dict[str, int]:
    """Parse aggregate CPU ticks, including iowait, from ``/proc/stat``."""
    first = text.splitlines()[0].split()
    if not first or first[0] != "cpu" or len(first) < 6:
        raise ValueError("/proc/stat does not start with a complete aggregate cpu line")
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    values = [int(value) for value in first[1:]]
    result = {name: values[i] if i < len(values) else 0 for i, name in enumerate(names)}
    result["total"] = sum(values)
    return {f"{key}_ticks": value for key, value in result.items()}


def parse_psi(text: str) -> dict[str, float | int]:
    """Parse Linux PSI ``some``/``full`` averages and cumulative totals."""
    result: dict[str, float | int] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        scope = parts[0]
        if scope not in {"some", "full"}:
            raise ValueError(f"unknown PSI scope {scope!r}")
        for item in parts[1:]:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"malformed PSI field {item!r}")
            result[f"{scope}_{key}"] = int(raw) if key == "total" else float(raw)
    return result


def parse_cgroup_path(text: str) -> str | None:
    """Return the unified cgroup-v2 path from ``/proc/PID/cgroup``."""
    for line in text.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            return path
    return None


def parse_flat_int_mapping(text: str) -> dict[str, int]:
    """Parse ``key integer`` files such as cgroup ``memory.events``."""
    result: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"malformed key/integer line: {line!r}")
        result[parts[0]] = int(parts[1])
    return result


def parse_deleted_mappings(text: str, store_root: Path | None) -> dict[str, Any]:
    """Return deleted file mappings, optionally restricted to one store root.

    Virtual bytes are deliberately labelled as such: the same physical tmpfs
    pages can be mapped by several processes and must not be summed as resident
    memory.  PSS/SwapPss provide the proportional physical accounting.
    """
    root = os.path.abspath(store_root) if store_root is not None else None
    paths: list[str] = []
    virtual_bytes = 0
    for line in text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or not parts[5].endswith(" (deleted)"):
            continue
        path = parts[5][: -len(" (deleted)")]
        if root is not None and not (path == root or path.startswith(root + os.sep)):
            continue
        start_raw, end_raw = parts[0].split("-", 1)
        virtual_bytes += int(end_raw, 16) - int(start_raw, 16)
        paths.append(path)
    return {
        "deleted_mapping_count": len(paths),
        "deleted_mapped_virtual_bytes": virtual_bytes,
        "deleted_mapping_paths": sorted(set(paths)),
    }


def _read_text(
    path: Path,
    errors: list[str],
    label: str,
    *,
    missing_is_error: bool = False,
) -> str | None:
    try:
        return path.read_text()
    except (FileNotFoundError, ProcessLookupError):
        if missing_is_error:
            errors.append(f"{label}: unavailable at {path}")
        return None
    except OSError as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return None


def sample_process(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    store_root: Path | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Sample one process, returning ``None`` when it vanished mid-sample."""
    errors: list[str] = []
    root = proc_root / str(pid)
    stat_text = _read_text(root / "stat", errors, f"pid {pid} stat")
    if stat_text is None:
        return None, errors
    try:
        sample: dict[str, Any] = parse_proc_stat(stat_text)
    except ValueError as exc:
        errors.append(f"pid {pid} stat parse: {exc}")
        return None, errors

    # A subprocess can become a zombie between the process-tree snapshot and
    # Popen.poll() reaping it.  Linux retains /proc/PID/stat for that terminal
    # record but may deny /proc/PID/io (EACCES); the process has already
    # released its mappings and cannot contribute live resource state.  Record
    # the exact terminal identity without probing optional live-process files.
    # This is distinct from suppressing an unavailable surface on a live task.
    if sample["state"] in _TERMINAL_PROCESS_STATES:
        sample.update(
            smaps_rollup=None,
            status=None,
            io=None,
            deleted_mappings=None,
        )
        return sample, errors

    sources: tuple[tuple[str, Any, str], ...] = (
        ("smaps_rollup", parse_smaps_rollup, "smaps_rollup"),
        ("status", parse_proc_status, "status"),
        ("io", parse_proc_io, "io"),
    )
    for filename, parser, key in sources:
        raw = _read_text(root / filename, errors, f"pid {pid} {filename}")
        if raw is None:
            sample[key] = None
            continue
        try:
            sample[key] = parser(raw)
        except (ValueError, IndexError) as exc:
            errors.append(f"pid {pid} {filename} parse: {exc}")
            sample[key] = None

    maps_text = _read_text(root / "maps", errors, f"pid {pid} maps")
    sample["deleted_mappings"] = (
        parse_deleted_mappings(maps_text, store_root) if maps_text is not None else None
    )
    return sample, errors


def process_tree_pids(root_pid: int) -> list[int]:
    """Return a stable root-first snapshot of the live process tree."""
    try:
        root = psutil.Process(root_pid)
        children = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return []
    return [root_pid, *sorted({proc.pid for proc in children})]


def visible_store_usage(store_root: Path) -> dict[str, int | bool]:
    """Stat visible store files without following symlinks."""
    if not store_root.exists():
        return {"exists": False, "visible_bytes": 0, "files": 0, "directories": 0}
    visible_bytes = 0
    files = 0
    directories = 0
    pending = [store_root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directories += 1
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    visible_bytes += entry.stat(follow_symlinks=False).st_size
    return {
        "exists": True,
        "visible_bytes": visible_bytes,
        "files": files,
        "directories": directories,
    }


def _sample_cgroup(
    root_pid: int,
    *,
    proc_root: Path,
    cgroup_root: Path,
    errors: list[str],
) -> dict[str, Any] | None:
    cgroup_text = _read_text(
        proc_root / str(root_pid) / "cgroup",
        errors,
        "root cgroup",
        missing_is_error=True,
    )
    if cgroup_text is None:
        return None
    try:
        relative = parse_cgroup_path(cgroup_text)
    except ValueError as exc:
        errors.append(f"root cgroup parse: {exc}")
        return None
    if relative is None:
        errors.append("root cgroup: no unified cgroup-v2 entry")
        return None
    group_dir = cgroup_root / relative.lstrip("/")
    if not group_dir.is_dir():
        errors.append(f"root cgroup: resolved directory unavailable at {group_dir}")
        return None
    result: dict[str, Any] = {"path": relative}
    for filename in ("memory.current", "memory.swap.current", "memory.max", "memory.high"):
        raw = _read_text(
            group_dir / filename,
            errors,
            f"cgroup {filename}",
            missing_is_error=True,
        )
        if raw is None:
            result[filename.replace(".", "_")] = None
        else:
            value = raw.strip()
            key = filename.replace(".", "_")
            try:
                result[key] = value if value == "max" else int(value)
            except ValueError as exc:
                errors.append(f"cgroup {filename} parse: {exc}")
                result[key] = None
    events = _read_text(
        group_dir / "memory.events",
        errors,
        "cgroup memory.events",
        missing_is_error=True,
    )
    if events is None:
        result["memory_events"] = None
    else:
        try:
            result["memory_events"] = parse_flat_int_mapping(events)
        except ValueError as exc:
            errors.append(f"cgroup memory.events parse: {exc}")
            result["memory_events"] = None
    return result


def sample_resources(
    root_pid: int,
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    store_root: Path | None = None,
    monotonic_ns: int | None = None,
    wall_time_ns: int | None = None,
) -> dict[str, Any]:
    """Capture one process-tree, host, cgroup, and optional-store sample."""
    errors: list[str] = []
    processes: list[dict[str, Any]] = []
    for pid in process_tree_pids(root_pid):
        sample, process_errors = sample_process(pid, proc_root=proc_root, store_root=store_root)
        errors.extend(process_errors)
        if sample is not None:
            processes.append(sample)

    def _sum_smaps(field: str) -> int:
        return sum(
            int(proc["smaps_rollup"][field])
            for proc in processes
            if proc.get("smaps_rollup") is not None
            and proc["smaps_rollup"].get(field) is not None
        )

    aggregate = {
        "process_count": len(processes),
        "Rss_kB_sum": _sum_smaps("Rss_kB"),
        "Pss_kB_sum": _sum_smaps("Pss_kB"),
        "Uss_kB_sum": _sum_smaps("Uss_kB"),
        "Swap_kB_sum": _sum_smaps("Swap_kB"),
        "SwapPss_kB_sum": _sum_smaps("SwapPss_kB"),
        "deleted_mapping_count_sum": sum(
            int(proc["deleted_mappings"]["deleted_mapping_count"])
            for proc in processes
            if proc.get("deleted_mappings") is not None
        ),
    }

    host: dict[str, Any] = {}
    host_sources: tuple[tuple[Path, Any, str], ...] = (
        (proc_root / "meminfo", parse_meminfo, "meminfo"),
        (proc_root / "vmstat", parse_vmstat, "vmstat"),
        (proc_root / "stat", parse_proc_cpu_stat, "cpu"),
        (proc_root / "pressure" / "memory", parse_psi, "memory_psi"),
        (proc_root / "pressure" / "io", parse_psi, "io_psi"),
    )
    for path, parser, key in host_sources:
        raw = _read_text(path, errors, f"host {key}", missing_is_error=True)
        if raw is None:
            host[key] = None
            continue
        try:
            host[key] = parser(raw)
        except (ValueError, IndexError) as exc:
            errors.append(f"host {key} parse: {exc}")
            host[key] = None

    store: dict[str, Any] | None = None
    if store_root is not None:
        try:
            store = visible_store_usage(store_root)
            stat = os.statvfs(store_root)
            store.update(
                filesystem_total_bytes=stat.f_frsize * stat.f_blocks,
                filesystem_free_bytes=stat.f_frsize * stat.f_bavail,
            )
        except OSError as exc:
            errors.append(f"store {store_root}: {type(exc).__name__}: {exc}")
            store = None

    cgroup = (
        _sample_cgroup(
            root_pid,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
            errors=errors,
        )
        if processes
        else None
    )
    return {
        "schema_version": RESOURCE_SAMPLE_SCHEMA_VERSION,
        "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
        "wall_time_ns": time.time_ns() if wall_time_ns is None else wall_time_ns,
        "root_pid": root_pid,
        "processes": processes,
        "aggregate": aggregate,
        "host": host,
        "cgroup": cgroup,
        "store": store,
        "errors": errors,
    }


def sample_process_until_exit(
    process: Popen[Any],
    output: IO[str],
    *,
    interval_seconds: float = 1.0,
    store_root: Path | None = None,
) -> int:
    """Write raw JSONL samples until *process* exits and return its code."""
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be > 0, got {interval_seconds}")
    next_sample = time.monotonic()
    while True:
        sample = sample_resources(process.pid, store_root=store_root)
        output.write(json.dumps(sample, separators=(",", ":")) + "\n")
        output.flush()
        returncode = process.poll()
        if returncode is not None:
            return returncode
        next_sample += interval_seconds
        delay = next_sample - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_sample = time.monotonic()


def _nested_int(sample: Mapping[str, Any], *keys: str) -> int | None:
    current: Any = sample
    for key in keys:
        if not isinstance(current, Mapping) or current.get(key) is None:
            return None
        current = current[key]
    return int(current)


def summarize_resource_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize raw samples while retaining explicit missing/error state."""
    records = list(samples)
    if not records:
        raise ValueError("cannot summarize an empty resource sample sequence")
    for i, sample in enumerate(records):
        if sample.get("schema_version") != RESOURCE_SAMPLE_SCHEMA_VERSION:
            raise ValueError(
                f"sample {i} schema_version={sample.get('schema_version')!r}, "
                f"expected {RESOURCE_SAMPLE_SCHEMA_VERSION}"
            )

    def _values(*keys: str) -> list[int]:
        return [value for sample in records if (value := _nested_int(sample, *keys)) is not None]

    def _peak(*keys: str) -> int | None:
        values = _values(*keys)
        return max(values) if values else None

    def _minimum(*keys: str) -> int | None:
        values = _values(*keys)
        return min(values) if values else None

    def _delta(*keys: str) -> int | None:
        values = _values(*keys)
        return values[-1] - values[0] if len(values) >= 2 else None

    errors = sorted(
        {
            str(error)
            for sample in records
            for error in sample.get("errors", [])
        }
    )
    return {
        "schema_version": RESOURCE_SAMPLE_SCHEMA_VERSION,
        "sample_count": len(records),
        "duration_seconds": max(
            0.0,
            (int(records[-1]["monotonic_ns"]) - int(records[0]["monotonic_ns"])) / 1e9,
        ),
        "peak_process_count": _peak("aggregate", "process_count"),
        "peak_rss_kB_sum": _peak("aggregate", "Rss_kB_sum"),
        "peak_pss_kB_sum": _peak("aggregate", "Pss_kB_sum"),
        "peak_uss_kB_sum": _peak("aggregate", "Uss_kB_sum"),
        "peak_swap_pss_kB_sum": _peak("aggregate", "SwapPss_kB_sum"),
        "peak_deleted_mapping_count_sum": _peak(
            "aggregate", "deleted_mapping_count_sum"
        ),
        "minimum_mem_available_kB": _minimum("host", "meminfo", "MemAvailable_kB"),
        "peak_shmem_kB": _peak("host", "meminfo", "Shmem_kB"),
        "pswpin_delta": _delta("host", "vmstat", "pswpin"),
        "pswpout_delta": _delta("host", "vmstat", "pswpout"),
        "major_fault_delta": _delta("host", "vmstat", "pgmajfault"),
        "cpu_iowait_ticks_delta": _delta("host", "cpu", "iowait_ticks"),
        "peak_store_visible_bytes": _peak("store", "visible_bytes"),
        "minimum_store_filesystem_free_bytes": _minimum(
            "store", "filesystem_free_bytes"
        ),
        "errors": errors,
        "complete": not errors,
    }


def load_resource_samples(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL resource record with line-numbered validation errors."""
    samples: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        samples.append(value)
    return samples
