"""Benchmark driver for splice-preselection resource safety (issues #740/#758).

Builds a scratch symlink-farm subset of a real packed-row dir (no packed-row
or feature data is ever copied -- only symlinks, and only written under
``scratch``), runs ``scripts.preprocess.precompute_selected_features`` as a
subprocess once per requested worker/shared-store/source-cache arm, and records
raw process-tree PSS/USS/swap/tmpfs samples alongside wall-clock metrics and the
``splice_preselection_locality`` analyzer's LRU load counts for the same
subset (and, by default, the whole ``packed_row_dir``). The first arm is the
unchanged serial/private oracle. Every later arm is accepted only if its whole
selection output tree has the same file set and exact shard bytes; manifest
comparison excludes only the unavoidable wall-clock ``created_at`` value.

Every swept key is producer-only and deliberately excluded from
``compute_selection_hash``, so ``sel/`` is reset between arms. Otherwise the
second arm would silently reuse the first arm's output instead of recomputing.
The required issue #758 matrix is the default: serial, W=8/capacity=12,
W=12/capacity=16, and W=16/capacity=20. Nothing runs until the caller provides
the explicit real shard list and command overrides.

Read-only on all production inputs (``packed_row_dir`` and whatever
``feature_dir`` the caller's ``--base-cmd`` points at); every write happens
under ``scratch``. Does not run anything by default -- ``--shards`` and
``--base-cmd`` are required, and even then ``--skip-run`` limits the driver to
building the subset and running the (cheap, read-only) locality analyzer.

Usage::

    uv run python -m scripts.analysis.benchmark_splice_preselection \\
        /path/to/real/packed_row_dir /tmp/scratch/bench740 \\
        --shards shard_00000,shard_00001,shard_00002 \\
        --base-cmd "feature_dir=/path/to/feature_dir segment_size=128 ..." \\
        --worker-store-arms "serial=1:private:0,w8c12=8:shared:12,w12c16=12:shared:16,w16c20=16:shared:20" \\
        --source-cache-shards 5 \\
        --existing-reference-dir /path/to/existing/sel/hash \\
        --out /tmp/scratch/bench740/metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.analysis.resource_sampler import (
    load_resource_samples,
    sample_process_until_exit,
    summarize_resource_samples,
)
from scripts.analysis.splice_preselection_locality import analyze
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import read_manifest

# precompute_selected_features's own per-shard splice_counterfactual log line:
#   "  shard_00000: splice_counterfactual 42 segs, 10 sel, 3 neg (1034.8s)"
_SHARD_LINE_RE = re.compile(
    r"(?P<shard>[\w.-]+):\s*splice_counterfactual\s+(?P<segs>\d+)\s+segs,\s+"
    r"(?P<sel>\d+)\s+sel,\s+(?P<neg>\d+)\s+neg\s+\((?P<seconds>[\d.]+)s\)"
)
# "Done: 3 shards in 2504s"
_DONE_LINE_RE = re.compile(r"Done:\s*(?P<n_shards>\d+)\s+shards\s+in\s+(?P<seconds>\d+)s")
# "Peak RSS this process: 51.2 GB" (resource.getrusage self-report)
_SELF_RSS_RE = re.compile(r"Peak RSS this process:\s*(?P<gb>[\d.]+)\s*GB")
_RESOURCE_PROFILE_RE = re.compile(r"SPLICE_RESOURCE_PROFILE\s+(?P<payload>\{.*\})$")
_ARM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class BenchmarkArm:
    """One explicit worker/shared-store resource shape."""

    name: str
    workers: int
    shared_store: bool
    shared_store_capacity: int

    def hydra_overrides(self) -> list[str]:
        return [
            f"selection_num_workers={self.workers}",
            f"selection_shared_store={'true' if self.shared_store else 'false'}",
            f"selection_shared_store_capacity={self.shared_store_capacity}",
        ]


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_length += len(chunk)
            digest.update(chunk)
    return byte_length, digest.hexdigest()


def _regular_file_identity(path: Path, *, surface: str) -> tuple[int, ...]:
    """Capture a direct regular file without following a symlink."""
    try:
        result = path.lstat()
    except OSError as exc:
        raise ValueError(f"{surface} is unreadable: {path}: {exc}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise ValueError(f"{surface} must be a real regular file: {path}")
    return (
        int(result.st_mode),
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _compare_exact_files(left: Path, right: Path) -> tuple[bool, dict[str, object]]:
    """Hash and directly compare two stable regular files."""
    left_before = _regular_file_identity(left, surface="reference file")
    right_before = _regular_file_identity(right, surface="candidate file")
    left_size, left_sha = _sha256_file(left)
    right_size, right_sha = _sha256_file(right)
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        exact = True
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                exact = False
                break
            if not left_chunk:
                break
    left_after = _regular_file_identity(left, surface="reference file")
    right_after = _regular_file_identity(right, surface="candidate file")
    if left_after != left_before or right_after != right_before:
        raise ValueError(
            "benchmark comparison input changed while it was being read: "
            f"reference={left}, candidate={right}"
        )
    return exact, {
        "reference_bytes": left_size,
        "reference_sha256": left_sha,
        "candidate_bytes": right_size,
        "candidate_sha256": right_sha,
    }


def _selection_output_dir(sel_root: Path) -> Path:
    """Resolve the one selection-hash directory produced by a benchmark arm."""
    if not sel_root.is_dir():
        raise FileNotFoundError(f"selection root was not produced: {sel_root}")
    children = sorted(path for path in sel_root.iterdir() if path.is_dir())
    if len(children) != 1:
        raise ValueError(
            f"expected exactly one selection output directory under {sel_root}, "
            f"got {[path.name for path in children]}"
        )
    if children[0].is_symlink():
        raise ValueError(f"selection output directory must not be a symlink: {children[0]}")
    return children[0]


def resolve_benchmark_output_path(
    scratch: Path,
    requested: Path | None,
) -> Path:
    """Resolve the metrics path and prove it remains below real scratch."""
    scratch_input = Path(scratch).absolute()
    if scratch_input.is_symlink():
        raise ValueError(f"benchmark scratch must not be a symlink: {scratch_input}")
    scratch_resolved = scratch_input.resolve(strict=False)
    output_input = (
        scratch_resolved / "metrics.json"
        if requested is None
        else Path(requested).absolute()
    )
    if output_input.is_symlink():
        raise ValueError(f"benchmark metrics output must not be a symlink: {output_input}")
    output_resolved = output_input.resolve(strict=False)
    if output_resolved == scratch_resolved or scratch_resolved not in output_resolved.parents:
        raise ValueError(
            "benchmark metrics output must be a file below scratch: "
            f"output={output_resolved}, scratch={scratch_resolved}"
        )
    return output_resolved


def resolve_existing_reference_dir(
    scratch: Path,
    requested: Path | None,
) -> Path | None:
    """Resolve an optional read-only reference and prove it is outside scratch."""
    if requested is None:
        return None
    scratch_input = Path(scratch).absolute()
    if scratch_input.is_symlink():
        raise ValueError(f"benchmark scratch must not be a symlink: {scratch_input}")
    scratch_resolved = scratch_input.resolve(strict=False)
    reference_input = Path(requested).absolute()
    if reference_input.is_symlink():
        raise ValueError(
            f"existing selection reference must not be a symlink: {reference_input}"
        )
    try:
        reference = reference_input.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"existing selection reference is unavailable: {reference_input}: {exc}"
        ) from exc
    if not reference.is_dir():
        raise ValueError(
            f"existing selection reference must be a directory: {reference}"
        )
    if (
        reference == scratch_resolved
        or reference in scratch_resolved.parents
        or scratch_resolved in reference.parents
    ):
        raise ValueError(
            "existing selection reference and benchmark scratch must be disjoint: "
            f"reference={reference}, scratch={scratch_resolved}"
        )
    return reference


def compare_selection_output_trees(reference: Path, candidate: Path) -> dict:
    """Compare benchmark outputs without reducing shard equality to a hash claim.

    Every non-manifest file is streamed twice: SHA-256/length are recorded for
    auditability, then the bytes themselves are compared chunk-for-chunk. The
    manifest is also compared byte-for-byte and canonically after removing only
    its wall-clock ``created_at`` field, which necessarily differs across
    multi-minute benchmark arms. Fixed-clock tests separately require exact
    manifest bytes.
    """
    reference = Path(reference)
    candidate = Path(candidate)
    for label, root in (("reference", reference), ("candidate", candidate)):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"benchmark {label} output must be a real directory: {root}")

    def inventory(root: Path) -> tuple[dict[str, Path], set[str], list[str]]:
        files: dict[str, Path] = {}
        directories: set[str] = set()
        unsafe: list[str] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                unsafe.append(f"{relative}:symlink")
            elif path.is_dir():
                directories.add(relative)
            elif path.is_file():
                files[relative] = path
            else:
                unsafe.append(f"{relative}:special")
        return files, directories, unsafe

    reference_files, reference_directories, reference_unsafe = inventory(reference)
    candidate_files, candidate_directories, candidate_unsafe = inventory(candidate)
    missing = sorted(set(reference_files) - set(candidate_files))
    extra = sorted(set(candidate_files) - set(reference_files))
    missing_directories = sorted(reference_directories - candidate_directories)
    extra_directories = sorted(candidate_directories - reference_directories)
    byte_mismatches: list[str] = []
    identities: dict[str, dict[str, object]] = {}
    manifest_raw_equal: bool | None = None
    manifest_normalized_equal: bool | None = None
    manifest_created_at: dict[str, object] | None = None

    for relative in sorted(set(reference_files) & set(candidate_files)):
        left = reference_files[relative]
        right = candidate_files[relative]
        exact, identities[relative] = _compare_exact_files(left, right)
        if relative == "manifest.json":
            manifest_raw_equal = exact
            try:
                left_manifest = json.loads(left.read_text(encoding="utf-8"))
                right_manifest = json.loads(right.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot parse benchmark selection manifest {relative}: {exc}"
                ) from exc
            if not isinstance(left_manifest, dict) or not isinstance(right_manifest, dict):
                raise ValueError("benchmark selection manifests must be JSON objects")
            manifest_created_at = {
                "reference": left_manifest.pop("created_at", None),
                "candidate": right_manifest.pop("created_at", None),
            }
            manifest_normalized_equal = left_manifest == right_manifest
            if not manifest_normalized_equal:
                byte_mismatches.append(relative)
        elif not exact:
            byte_mismatches.append(relative)

    ok = (
        not missing
        and not extra
        and not missing_directories
        and not extra_directories
        and not reference_unsafe
        and not candidate_unsafe
        and not byte_mismatches
        and manifest_normalized_equal is True
    )
    return {
        "ok": ok,
        "comparison": "exact bytes except manifest.created_at; manifest otherwise exact",
        "missing": missing,
        "extra": extra,
        "missing_directories": missing_directories,
        "extra_directories": extra_directories,
        "reference_unsafe_entries": reference_unsafe,
        "candidate_unsafe_entries": candidate_unsafe,
        "byte_mismatches": byte_mismatches,
        "manifest_raw_equal": manifest_raw_equal,
        "manifest_normalized_equal": manifest_normalized_equal,
        "manifest_created_at": manifest_created_at,
        "file_identities": identities,
    }


def compare_selected_shards_to_existing_reference(
    reference_dir: Path,
    candidate_dir: Path,
    shard_names: list[str],
) -> dict:
    """Compare selected output shards to a pre-existing production tree.

    The production manifest describes the full run while ``candidate_dir`` is
    normally a benchmark subset, so only the named compressed shard files are
    comparable. Each is both SHA-256 hashed and read directly byte-for-byte.
    """
    roots: dict[str, Path] = {}
    for label, raw_root in (
        ("existing reference", reference_dir),
        ("candidate", candidate_dir),
    ):
        root = Path(raw_root).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"benchmark {label} must be a real directory: {root}")
        roots[label] = root
    if not shard_names:
        raise ValueError("existing-reference comparison requires at least one shard")
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("existing-reference shard names must not contain duplicates")
    invalid = [name for name in shard_names if not _ARM_NAME_RE.fullmatch(name)]
    if invalid:
        raise ValueError(
            "existing-reference shard names must be plain shard stems: "
            f"{invalid}"
        )

    missing_reference: list[str] = []
    missing_candidate: list[str] = []
    mismatches: list[str] = []
    identities: dict[str, dict[str, object]] = {}
    reference = roots["existing reference"]
    candidate = roots["candidate"]
    for stem in shard_names:
        filename = f"{stem}.safetensors.zst"
        left = reference / filename
        right = candidate / filename
        if not os.path.lexists(left):
            missing_reference.append(filename)
            continue
        if not os.path.lexists(right):
            missing_candidate.append(filename)
            continue
        exact, identity = _compare_exact_files(left, right)
        identities[filename] = identity
        if not exact:
            mismatches.append(filename)
    return {
        "ok": not missing_reference and not missing_candidate and not mismatches,
        "comparison": "selected production shard files: SHA-256 and direct exact bytes",
        "reference_dir": str(reference),
        "candidate_dir": str(candidate),
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
        "byte_mismatches": mismatches,
        "file_identities": identities,
    }


def build_subset(packed_row_dir: Path, scratch: Path, shard_names: list[str]) -> Path:
    """Symlink the chosen packed + provenance shards into a scratch packed dir.

    Writes an edited ``manifest.json`` whose ``metadata.provenance_shards`` is
    the SUBSET count. ``metadata.base_shard_stems`` (and every other manifest
    field) is copied verbatim: packed-segment provenance
    (``PROVENANCE_BASE_SHARD_IDX_KEY``) indexes into ``base_shard_stems`` by
    absolute position, so truncating that list would resolve a subset shard's
    provenance to the WRONG source stem. Read-only on *packed_row_dir* --
    every write lands under *scratch*.

    Raises ``FileNotFoundError`` if *packed_row_dir* has no manifest, or if a
    requested shard's packed or provenance file is missing.
    """
    packed_row_dir = Path(packed_row_dir).resolve(strict=True)
    scratch_input = Path(scratch).absolute()
    if scratch_input.is_symlink():
        raise ValueError(f"scratch subset must not be a symlink: {scratch_input}")
    scratch = scratch_input.resolve(strict=False)
    if (
        packed_row_dir == scratch
        or packed_row_dir in scratch.parents
        or scratch in packed_row_dir.parents
    ):
        raise ValueError(
            "scratch subset and packed_row_dir must be disjoint, got "
            f"scratch={scratch}, packed_row_dir={packed_row_dir}"
        )
    if not shard_names:
        raise ValueError("shard_names must be non-empty")
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("shard_names must not contain duplicates")
    invalid_names = [name for name in shard_names if not _ARM_NAME_RE.fullmatch(name)]
    if invalid_names:
        raise ValueError(
            "shard_names must be plain packed-shard stems without path separators: "
            f"{invalid_names}"
        )

    manifest = read_manifest(packed_row_dir)
    if manifest is None:
        raise FileNotFoundError(f"no packed-row manifest.json under {packed_row_dir}")

    if scratch.exists() and not scratch.is_dir():
        raise ValueError(f"scratch subset must be a real directory path: {scratch}")
    if scratch.exists() and any(scratch.iterdir()):
        raise ValueError(
            f"scratch subset must be absent or empty; refusing to overwrite {scratch}"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    src_prov_dir = provenance_dir(packed_row_dir)
    out_prov_dir = provenance_dir(scratch)
    out_prov_dir.mkdir(parents=True, exist_ok=True)

    for name in shard_names:
        fn = f"{name}.safetensors.zst"
        src_packed = packed_row_dir / fn
        src_prov = src_prov_dir / fn
        if not src_packed.is_file():
            raise FileNotFoundError(f"packed shard not found: {src_packed}")
        if not src_prov.is_file():
            raise FileNotFoundError(f"provenance shard not found: {src_prov}")
        for dst, src in ((scratch / fn, src_packed), (out_prov_dir / fn, src_prov)):
            dst.symlink_to(src.resolve())

    metadata = dict(manifest["metadata"])
    metadata["provenance_shards"] = len(shard_names)
    manifest = {**manifest, "metadata": metadata}
    (scratch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scratch


def parse_instrumented_log(text: str) -> dict:
    """Parse ``precompute_selected_features``'s own loguru instrumentation.

    Returns per-shard wall time (from the ``splice_counterfactual`` per-shard
    log lines) and the script's own reported total elapsed time / peak RSS
    (``resource.getrusage`` -- a separate, in-process figure from the
    OS-level ``/usr/bin/time -v`` numbers ``parse_time_v_output`` reads).
    Missing lines map to ``None``/``{}`` rather than raising: a failed or
    truncated run still returns a well-formed (if partial) dict.
    """
    per_shard_seconds = {
        m.group("shard"): float(m.group("seconds")) for m in _SHARD_LINE_RE.finditer(text)
    }
    done = _DONE_LINE_RE.search(text)
    rss = _SELF_RSS_RE.search(text)
    resource_profiles: list[dict] = []
    resource_profile_errors: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = _RESOURCE_PROFILE_RE.search(line)
        if match is None:
            continue
        try:
            profile = json.loads(match.group("payload"))
        except json.JSONDecodeError as exc:
            resource_profile_errors.append(f"line {line_no}: {exc}")
            continue
        if not isinstance(profile, dict):
            resource_profile_errors.append(
                f"line {line_no}: resource profile is not a JSON object"
            )
            continue
        resource_profiles.append(profile)
    return {
        "per_shard_seconds": per_shard_seconds,
        "total_elapsed_seconds": float(done.group("seconds")) if done else None,
        "total_shards_reported": int(done.group("n_shards")) if done else None,
        "self_reported_peak_rss_gb": float(rss.group("gb")) if rss else None,
        "resource_profiles": resource_profiles,
        "resource_profile_errors": resource_profile_errors,
    }


def parse_time_v_output(text: str) -> dict:
    """Parse GNU ``/usr/bin/time -v`` output for peak RSS + wall clock."""
    peak_rss_kb: int | None = None
    wall_clock: str | None = None
    for line in text.splitlines():
        if "Maximum resident set size" in line:
            peak_rss_kb = int(line.rsplit(":", 1)[-1].strip())
        elif "Elapsed (wall clock)" in line:
            wall_clock = line.split(": ", 1)[-1].strip()
    return {"peak_rss_kb": peak_rss_kb, "wall_clock": wall_clock}


def run_arm(
    cmd: list[str],
    log_path: Path,
    *,
    resource_path: Path | None = None,
    resource_interval_seconds: float = 1.0,
    shared_store_root: Path | None = None,
) -> dict:
    """Run one arm under ``nice -n 19`` + ``/usr/bin/time -v``.

    Captures combined stdout/stderr to *log_path* and returns parsed metrics
    (never raises on a nonzero return code -- callers inspect
    ``returncode``/``ok`` so one failing arm does not abort a multi-capacity
    sweep). Raises ``FileNotFoundError`` up front if GNU ``time`` is not
    installed, rather than failing opaquely inside the subprocess call.
    """
    if shutil.which("/usr/bin/time") is None:
        raise FileNotFoundError(
            "/usr/bin/time not found (GNU time). Install it (e.g. `apt-get "
            "install time`) to run compute arms, or pass --skip-run for "
            "subset/analyzer-only smoke testing."
        )
    full_cmd = ["nice", "-n", "19", "/usr/bin/time", "-v", *cmd]
    if resource_interval_seconds <= 0:
        raise ValueError(
            "resource_interval_seconds must be > 0, got "
            f"{resource_interval_seconds}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        if resource_path is None:
            proc = subprocess.run(
                full_cmd, stdout=f, stderr=subprocess.STDOUT, check=False
            )
            returncode = proc.returncode
        else:
            resource_path.parent.mkdir(parents=True, exist_ok=True)
            with resource_path.open("w") as resource_file:
                running = subprocess.Popen(
                    full_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = sample_process_until_exit(
                        running,
                        resource_file,
                        interval_seconds=resource_interval_seconds,
                        store_root=shared_store_root,
                    )
                except BaseException:
                    # The measured command owns a separate session so terminal
                    # interrupts do not reach it automatically. Never leak a
                    # production process tree when sampling or the driver is
                    # interrupted.
                    if running.poll() is None:
                        os.killpg(running.pid, signal.SIGTERM)
                        try:
                            running.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(running.pid, signal.SIGKILL)
                            running.wait(timeout=10)
                    raise
    text = log_path.read_text()
    metrics: dict = {"returncode": returncode, "ok": returncode == 0}
    metrics.update(parse_time_v_output(text))
    metrics["instrumented"] = parse_instrumented_log(text)
    if resource_path is not None:
        samples = load_resource_samples(resource_path)
        metrics["resource_path"] = str(resource_path)
        metrics["resources"] = summarize_resource_samples(samples)
    return metrics


def _parse_positive_int_list(value: str, *, name: str) -> list[int]:
    parsed = [int(x) for x in value.split(",") if x.strip()]
    if not parsed:
        raise ValueError(f"{name}: empty capacity list {value!r}")
    for v in parsed:
        if v < 1:
            raise ValueError(f"{name}: capacity must be >= 1, got {v}")
    return parsed


def parse_worker_store_arms(value: str) -> list[BenchmarkArm]:
    """Parse ``name=workers:(private|shared):capacity`` benchmark arms."""
    arms: list[BenchmarkArm] = []
    names: set[str] = set()
    for raw_arm in value.split(","):
        arm_text = raw_arm.strip()
        if not arm_text:
            continue
        name, sep, shape = arm_text.partition("=")
        if not sep or not _ARM_NAME_RE.fullmatch(name):
            raise ValueError(
                "--worker-store-arms: each arm must be named "
                f"name=workers:(private|shared):capacity, got {raw_arm!r}"
            )
        if name in names:
            raise ValueError(f"--worker-store-arms: duplicate arm name {name!r}")
        parts = shape.split(":")
        if len(parts) != 3:
            raise ValueError(
                "--worker-store-arms: expected workers:mode:capacity for "
                f"{name!r}, got {shape!r}"
            )
        workers_raw, mode, capacity_raw = parts
        try:
            workers = int(workers_raw)
            capacity = int(capacity_raw)
        except ValueError as exc:
            raise ValueError(
                f"--worker-store-arms: non-integer workers/capacity in {raw_arm!r}"
            ) from exc
        if workers < 1:
            raise ValueError(
                f"--worker-store-arms: {name!r} workers must be >= 1, got {workers}"
            )
        if mode not in {"private", "shared"}:
            raise ValueError(
                f"--worker-store-arms: {name!r} mode must be private or shared, got {mode!r}"
            )
        shared = mode == "shared"
        if shared and capacity < 1:
            raise ValueError(
                f"--worker-store-arms: shared arm {name!r} capacity must be >= 1, got {capacity}"
            )
        if not shared and capacity != 0:
            raise ValueError(
                f"--worker-store-arms: private arm {name!r} capacity must be 0, got {capacity}"
            )
        names.add(name)
        arms.append(
            BenchmarkArm(
                name=name,
                workers=workers,
                shared_store=shared,
                shared_store_capacity=capacity,
            )
        )
    if not arms:
        raise ValueError("--worker-store-arms: no arms were specified")
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("packed_row_dir", type=Path, help="real packed-row dir (READ-ONLY)")
    ap.add_argument("scratch", type=Path, help="scratch output dir; every write lands here")
    ap.add_argument(
        "--shards",
        required=True,
        help="comma-separated packed shard stems, e.g. shard_00000,shard_00001",
    )
    ap.add_argument(
        "--base-cmd",
        required=True,
        help="precompute_selected_features Hydra overrides WITHOUT packed_row_dir (quoted string)",
    )
    ap.add_argument(
        "--source-cache-shards",
        default="5",
        help="comma-separated selection_source_cache_shards values to sweep "
        "inside every worker/store arm (default: shipped value 5)",
    )
    ap.add_argument(
        "--worker-store-arms",
        default=(
            "serial=1:private:0,w8c12=8:shared:12,"
            "w12c16=12:shared:16,w16c20=16:shared:20"
        ),
        help="comma-separated name=workers:(private|shared):capacity arms",
    )
    ap.add_argument(
        "--sidecar-cache-shards",
        type=int,
        default=16,
        help="selection_sidecar_cache_shards, held constant across every arm",
    )
    ap.add_argument(
        "--analyzer-capacities",
        default="1,2,3,5,8,16,24,48",
        help="comma-separated capacities for the splice_preselection_locality report",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="build the subset + run the analyzer only; do not invoke "
        "precompute_selected_features (fast smoke test of the subset builder)",
    )
    ap.add_argument(
        "--skip-full-dir-locality",
        action="store_true",
        help="skip analyzing the WHOLE packed_row_dir (only analyze the built subset)",
    )
    ap.add_argument(
        "--resource-sample-seconds",
        type=float,
        default=1.0,
        help="seconds between Linux process-tree PSS/USS/swap/tmpfs samples",
    )
    ap.add_argument(
        "--shared-store-root",
        type=Path,
        default=Path("/dev/shm"),
        help="tmpfs root to account while an arm runs (default: /dev/shm)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="metrics JSON below scratch (default: <scratch>/metrics.json)",
    )
    ap.add_argument(
        "--existing-reference-dir",
        type=Path,
        default=None,
        help="optional read-only existing production sel/<hash> directory; every "
        "selected shard must match its existing compressed bytes exactly",
    )
    args = ap.parse_args()

    try:
        output_path = resolve_benchmark_output_path(args.scratch, args.out)
        existing_reference_dir = resolve_existing_reference_dir(
            args.scratch,
            args.existing_reference_dir,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    shard_names = [s.strip() for s in args.shards.split(",") if s.strip()]
    if not shard_names:
        raise SystemExit("--shards must name at least one packed shard stem")
    source_caps = _parse_positive_int_list(args.source_cache_shards, name="--source-cache-shards")
    try:
        worker_store_arms = parse_worker_store_arms(args.worker_store_arms)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    analyzer_caps = _parse_positive_int_list(args.analyzer_capacities, name="--analyzer-capacities")
    if args.sidecar_cache_shards < 1:
        raise SystemExit(f"--sidecar-cache-shards must be >= 1, got {args.sidecar_cache_shards}")
    if args.resource_sample_seconds <= 0:
        raise SystemExit(
            f"--resource-sample-seconds must be > 0, got {args.resource_sample_seconds}"
        )
    reference_arm = worker_store_arms[0]
    if (
        reference_arm.workers != 1
        or reference_arm.shared_store
        or reference_arm.shared_store_capacity != 0
    ):
        raise SystemExit(
            "the first --worker-store-arms entry must be the serial private "
            "reference (1:private:0) so later arms have an exact output oracle"
        )

    subset = build_subset(args.packed_row_dir, args.scratch / "subset", shard_names)
    print(f"subset built at {subset} ({len(shard_names)} shard(s))")

    locality: dict = {"subset": analyze(subset, analyzer_caps)}
    if not args.skip_full_dir_locality:
        locality["full_dir"] = analyze(args.packed_row_dir, analyzer_caps)
    for scope, report in locality.items():
        ideal = report["unique_source_stems"]
        print(f"[{scope}] packed shards={report['n_packed_shards']} unique stems={ideal}")
        for cap, loads in report["loads_by_capacity"].items():
            ratio = loads / ideal if ideal else 0.0
            print(f"  [{scope}] capacity {cap:3d}: {loads:8d} loads ({ratio:5.1f}x ideal)")

    results: dict = {
        "packed_row_dir": str(args.packed_row_dir),
        "scratch": str(args.scratch),
        "shards": shard_names,
        "sidecar_cache_shards": args.sidecar_cache_shards,
        "worker_store_arms": [
            {
                "name": arm.name,
                "workers": arm.workers,
                "shared_store": arm.shared_store,
                "shared_store_capacity": arm.shared_store_capacity,
            }
            for arm in worker_store_arms
        ],
        "locality": locality,
        "reference_output": None,
        "existing_reference_dir": (
            str(existing_reference_dir)
            if existing_reference_dir is not None
            else None
        ),
        "arms": {},
    }

    if args.skip_run:
        print("--skip-run: not invoking precompute_selected_features")
    else:
        base_overrides = shlex.split(args.base_cmd)
        reference_output = args.scratch / "reference_output"
        for arm in worker_store_arms:
            for source_cap in source_caps:
                sel_dir = subset / "sel"
                if sel_dir.exists():
                    # All swept keys are producer-only and excluded from the
                    # selection hash. Reset or this arm would reuse prior bytes.
                    shutil.rmtree(sel_dir)
                key = (
                    arm.name
                    if len(source_caps) == 1
                    else f"{arm.name}_source{source_cap}"
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.preprocess.precompute_selected_features",
                    *base_overrides,
                    f"packed_row_dir={subset}",
                    f"selection_source_cache_shards={source_cap}",
                    f"selection_sidecar_cache_shards={args.sidecar_cache_shards}",
                    "selection_resume=false",
                    *arm.hydra_overrides(),
                ]
                log_path = args.scratch / f"{key}.log"
                resource_path = args.scratch / f"{key}.resources.jsonl"
                print(
                    f"running arm {key}: workers={arm.workers}, "
                    f"shared={arm.shared_store}, "
                    f"capacity={arm.shared_store_capacity} -> {log_path}"
                )
                metrics = run_arm(
                    cmd,
                    log_path,
                    resource_path=resource_path,
                    resource_interval_seconds=args.resource_sample_seconds,
                    shared_store_root=args.shared_store_root,
                )
                metrics.update(
                    source_cache_shards=source_cap,
                    sidecar_cache_shards=args.sidecar_cache_shards,
                    workers=arm.workers,
                    shared_store=arm.shared_store,
                    shared_store_capacity=arm.shared_store_capacity,
                    log_path=str(log_path),
                    command=cmd,
                )
                if metrics["ok"]:
                    try:
                        candidate_output = _selection_output_dir(subset / "sel")
                        if results["reference_output"] is None:
                            if arm != reference_arm:
                                raise ValueError(
                                    "serial reference arm did not produce a usable "
                                    "output; refusing to bless a candidate arm as reference"
                                )
                            shutil.copytree(candidate_output, reference_output)
                            results["reference_output"] = str(reference_output)
                            parity = compare_selection_output_trees(
                                reference_output,
                                candidate_output,
                            )
                            parity["is_reference_arm"] = True
                        else:
                            parity = compare_selection_output_trees(
                                reference_output,
                                candidate_output,
                            )
                            parity["is_reference_arm"] = False
                        metrics["parity"] = parity
                        if not parity["ok"]:
                            metrics["ok"] = False
                            metrics["failure_kind"] = "output_parity"
                        if existing_reference_dir is not None:
                            existing_parity = (
                                compare_selected_shards_to_existing_reference(
                                    existing_reference_dir,
                                    candidate_output,
                                    shard_names,
                                )
                            )
                            metrics["existing_reference_parity"] = existing_parity
                            if not existing_parity["ok"]:
                                metrics["ok"] = False
                                metrics["failure_kind"] = "existing_reference_parity"
                    except (OSError, ValueError) as exc:
                        metrics["parity"] = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        metrics["ok"] = False
                        metrics["failure_kind"] = "output_parity"
                else:
                    metrics["parity"] = {
                        "ok": False,
                        "not_compared_reason": "producer command failed",
                    }
                    if existing_reference_dir is not None:
                        metrics["existing_reference_parity"] = {
                            "ok": False,
                            "not_compared_reason": "producer command failed",
                        }
                results["arms"][key] = metrics
                status = (
                    "ok"
                    if metrics["ok"]
                    else f"FAILED (rc={metrics['returncode']})"
                )
                print(f"  arm {key}: {status}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
