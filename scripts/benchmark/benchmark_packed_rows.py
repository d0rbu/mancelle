"""Correctness-gated benchmark harness for packed-row preprocessing.

Each named variant gets a private feature tree under one benchmark run root. The
tree contains links (or private-copy fallbacks) to the raw/percentile/token inputs, while
all packed-row outputs and the mutable base-hash cache stay private to the
variant.  This matters because the producer derives its output directory from
``feature_dir``; running two variants against the production tree would either
clobber the same bounded output or fail the producer's no-clobber preflight.

The harness stat-fences every linked input around every run, samples process-tree
resources, and compares exact packed payload bytes plus timestamp-normalized
manifests against the first variant before publishing the Markdown report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_text
from nanochat.dataset import list_parquet_files
from scripts.benchmark._common import _median, _p90
from scripts.benchmark._resource_sampler import ResourceSample, sample_process_tree
from scripts.preprocess.precompute_packed_rows import load_packed_rows_config
from scripts.preprocess.precompute_packed_rows import run as run_packed
from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.feature_io import glob_features, shard_stem_from_path
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    OverlayKind,
    compute_base_shard_content_hash,
    write_manifest,
)

_DEFAULT_VARIANTS = [{"name": "profile", "overrides": {}}]
_SAFE_VARIANT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FORBIDDEN_VARIANT_OVERRIDE_KEYS = frozenset(
    {
        "feature_dir",
        "data_dir",
        "output_dir",
        "segment_size",
        "sae_width",
        "context_len",
        "max_shards",
        "decoder_segment_size",
        "buffer_size",
        "rows_per_shard",
        "resume",
        "resume_from_shard",
        "profile",
        "profile_summary_output",
        "remote",
    }
)
_BASE_CONFIG_KEYS = (
    "buffer_size",
    "rows_per_shard",
    "zstd_level",
    "compress_workers",
)
_DERIVED_OUTPUT_CACHES = frozenset({".packed-row-fingerprint-cache.json"})
_LOAD_DETAIL_KEYS = (
    "compressed_read",
    "zstd_decompress",
    "safetensors_deserialize",
    "tensor_upcast",
    "feature_id_delta_decode",
    "base_postprocess",
    "percentile_overlay",
    "array_validation",
    "token_boundary_overlay",
    "doc_index",
)


def _validate_variant_overrides(name: str, overrides: object) -> dict:
    if not _SAFE_VARIANT_NAME.fullmatch(name):
        raise ValueError(
            f"variant name {name!r} must match {_SAFE_VARIANT_NAME.pattern!r}."
        )
    if overrides is None:
        return {}
    if OmegaConf.is_config(overrides):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(overrides, dict):
        raise ValueError(
            f"variant {name!r}: overrides must be a mapping, got "
            f"{type(overrides).__name__}."
        )
    blocked = sorted(
        str(key) for key in overrides if str(key) in _FORBIDDEN_VARIANT_OVERRIDE_KEYS
    )
    if blocked:
        raise ValueError(
            f"variant {name!r}: override(s) {blocked} are not allowed. "
            "Benchmark variants must not replace packed-row identity, input, "
            "output, resume, remote, or profiling fields."
        )
    return dict(overrides)


def _validate_resolved_variant_output(cfg: DictConfig, *, variant_name: str) -> None:
    max_shards = cfg.get("max_shards", None)
    try:
        resolved_max_shards = int(max_shards)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"variant {variant_name!r}: max_shards must resolve to a positive integer, "
            f"got {max_shards!r}."
        ) from exc
    if resolved_max_shards <= 0:
        raise ValueError(
            f"variant {variant_name!r}: max_shards must resolve to a positive integer, "
            f"got {resolved_max_shards}."
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve()
    resolved_right = right.resolve()
    return (
        resolved_left == resolved_right
        or resolved_left.is_relative_to(resolved_right)
        or resolved_right.is_relative_to(resolved_left)
    )


def _assert_benchmark_path_isolated(
    path: Path, *, source_dirs: tuple[Path, ...], label: str
) -> None:
    for source_dir in source_dirs:
        if _paths_overlap(path, source_dir):
            raise ValueError(
                f"benchmark_packed_rows: {label} {path} overlaps source directory "
                f"{source_dir}; benchmark writes must be isolated from production inputs."
            )


def _mirror_file(source: Path, destination: Path, *, copy: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_feature_tree(
    source: Path,
    destination: Path,
    *,
    selected_stems: list[str],
) -> tuple[list[Path], list[tuple[Path, dict, OverlayKind]]]:
    """Mirror the selected raw/overlay family and return manifests to rebind."""
    raw_by_stem = {
        shard_stem_from_path(path): path for path in glob_features(source)
    }
    missing_raw = [stem for stem in selected_stems if stem not in raw_by_stem]
    if missing_raw:
        raise ValueError(
            f"selected parquet stems have no matching feature shard in {source}: "
            f"{missing_raw}"
        )
    fenced_sources: list[Path] = []
    for stem in selected_stems:
        shard = raw_by_stem[stem]
        _mirror_file(shard, destination / shard.name)
        fenced_sources.append(shard)

    statistics = source / "feature_statistics.safetensors"
    if statistics.is_file():
        _mirror_file(statistics, destination / statistics.name)
        fenced_sources.append(statistics)

    overlay_manifests: list[tuple[Path, dict, OverlayKind]] = []
    pct_root = source / "pct"
    if pct_root.is_dir():
        for overlay_dir in sorted(path for path in pct_root.iterdir() if path.is_dir()):
            manifest_path = overlay_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest.get("checksum"), str):
                raise ValueError(f"invalid percentile manifest: {manifest_path}")
            fenced_sources.append(manifest_path)
            destination_overlay = destination / "pct" / overlay_dir.name
            for stem in selected_stems:
                leaf = overlay_dir / f"{stem}.safetensors.zst"
                if not leaf.is_file():
                    raise ValueError(
                        f"percentile overlay {overlay_dir} is missing selected leaf {stem}"
                    )
                _mirror_file(leaf, destination / "pct" / overlay_dir.name / leaf.name)
                fenced_sources.append(leaf)
            overlay_manifests.append(
                (destination_overlay, manifest, OverlayKind.PERCENTILE)
            )

    tok_root = source / "tok"
    if tok_root.is_dir():
        for token_dir in sorted(path for path in tok_root.iterdir() if path.is_dir()):
            boundary_dir = token_dir / "seg_boundaries"
            if not boundary_dir.is_dir():
                continue
            manifest_path = boundary_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest.get("checksum"), str):
                raise ValueError(f"invalid token-boundary manifest: {manifest_path}")
            fenced_sources.append(manifest_path)
            destination_overlay = (
                destination / "tok" / token_dir.name / "seg_boundaries"
            )
            for stem in selected_stems:
                leaf = boundary_dir / f"{stem}.safetensors.zst"
                if not leaf.is_file():
                    # Zero-segment base shards deliberately omit token leaves.
                    continue
                _mirror_file(
                    leaf,
                    destination / "tok" / token_dir.name / "seg_boundaries" / leaf.name,
                )
                fenced_sources.append(leaf)
            overlay_manifests.append(
                (destination_overlay, manifest, OverlayKind.SEG_BOUNDARY)
            )
    return fenced_sources, overlay_manifests


def _write_bounded_input_manifests(
    manifests: list[tuple[Path, dict, OverlayKind]], *, base_hash: str
) -> None:
    for destination, source_manifest, kind in manifests:
        write_manifest(
            destination,
            str(source_manifest["checksum"]),
            kind,
            metadata=source_manifest.get("metadata"),
            precomputed_base_hash=base_hash,
        )


def _prepare_parquets(source: Path, destination: Path, *, max_shards: int) -> list[Path]:
    parquet_paths = [Path(path) for path in list_parquet_files(str(source))]
    selected = parquet_paths[:max_shards]
    if len(selected) != max_shards:
        raise ValueError(
            f"benchmark_packed_rows: requested max_shards={max_shards}, but found "
            f"only {len(selected)} parquet shard(s) in {source}."
        )
    for parquet in selected:
        _mirror_file(parquet, destination / parquet.name)
    return selected


def _stat_inventory(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    inventory: dict[str, dict[str, int | str]] = {}
    for path in sorted({path.resolve() for path in paths}):
        stat = path.stat()
        inventory[str(path)] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
        }
    return inventory


def _assert_inventory_unchanged(
    expected: dict[str, dict[str, int | str]], *, variant_name: str, stage: str
) -> None:
    try:
        actual = _stat_inventory([Path(path) for path in expected])
    except OSError as exc:
        raise AssertionError(
            f"variant {variant_name!r}: source input disappeared during {stage}: {exc}"
        ) from exc
    if actual != expected:
        changed = sorted(
            path
            for path in set(expected) | set(actual)
            if expected.get(path) != actual.get(path)
        )
        raise AssertionError(
            f"variant {variant_name!r}: source input inventory changed during {stage}: "
            f"paths={changed}"
        )


def _canonical_json_digest(path: Path) -> str:
    payload = json.loads(path.read_text())

    def remove_timestamps(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: remove_timestamps(child)
                for key, child in value.items()
                if key != "created_at"
            }
        if isinstance(value, list):
            return [remove_timestamps(child) for child in value]
        return value

    canonical = json.dumps(
        remove_timestamps(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_output_dirs(
    feature_dir: Path, *, context_len: int, max_shards: int
) -> tuple[Path, Path]:
    train_dirs = sorted(
        path.parent
        for path in feature_dir.glob(
            f"pct/*/tok/*/c{context_len}/s{max_shards}/manifest.json"
        )
    )
    if len(train_dirs) != 1:
        raise AssertionError(
            "expected exactly one canonical packed train output under "
            f"{feature_dir}, found {train_dirs}"
        )
    train_dir = train_dirs[0]
    val_dir = train_dir.parent / f"val_{train_dir.name}"
    if not (val_dir / "manifest.json").is_file():
        raise AssertionError(f"packed validation output has no manifest: {val_dir}")
    return train_dir, val_dir


def _artifact_digests(
    feature_dir: Path, *, context_len: int, max_shards: int
) -> dict[str, str]:
    train_dir, val_dir = _discover_output_dirs(
        feature_dir, context_len=context_len, max_shards=max_shards
    )
    result: dict[str, str] = {}
    for split, directory in (("train", train_dir), ("val", val_dir)):
        for path in sorted(
            item
            for item in directory.rglob("*")
            if item.is_file() and item.name not in _DERIVED_OUTPUT_CACHES
        ):
            relative = path.relative_to(directory).as_posix()
            key = f"{split}/{relative}"
            result[key] = (
                _canonical_json_digest(path)
                if path.name == "manifest.json"
                else _file_digest(path)
            )
    return result


def _summarize(sidecar: dict, resources: ResourceSample) -> dict:
    per_shard = sidecar.get("per_output_shard", [])
    save = [float(shard.get("save_seconds", 0.0)) for shard in per_shard]
    worker = [float(shard.get("worker_seconds", 0.0)) for shard in per_shard]
    queue_wait = [float(shard.get("queue_wait_seconds", 0.0)) for shard in per_shard]
    resident_loads = [
        record
        for record in sidecar.get("feature_index_loads", [])
        if record.get("phase") == "resident_load"
    ]
    load_detail_totals = {
        key: sum(
            float(record.get("detail_seconds", {}).get(key, 0.0))
            for record in resident_loads
        )
        for key in _LOAD_DETAIL_KEYS
    }
    return {
        "pre_pass_s": sidecar["manifest_pre_pass"]["seconds"],
        "cache_hit": sidecar["manifest_pre_pass"]["cache_hit"],
        "stage_totals": sidecar["stage_totals"],
        "row_build_detail": sidecar.get(
            "row_build_detail",
            {
                "row_to_csr_seconds": 0.0,
                "row_provenance_seconds": 0.0,
                "build_segments_seconds": 0.0,
                "segment_boundaries_seconds": 0.0,
                "feature_concat_seconds": 0.0,
                "feature_ids_concat_seconds": 0.0,
                "percentiles_concat_seconds": 0.0,
                "feature_counts_seconds": 0.0,
                "row_tensor_finalize_seconds": 0.0,
            },
        ),
        "base_loads_s": sidecar["base_shard_loads"]["total_seconds"],
        "save_med_p90": (_median(save), _p90(save)),
        "worker_med_p90": (_median(worker), _p90(worker)),
        "queue_wait_s": sum(queue_wait),
        "feature_index_scan_s": float(
            sidecar.get("startup", {}).get("feature_index_scan_seconds", 0.0)
        ),
        "load_detail_totals": load_detail_totals,
        "resident_load_count": len(resident_loads),
        "resident_compressed_gib": sum(
            int(record.get("compressed_bytes", 0)) for record in resident_loads
        )
        / (1024**3),
        "resident_decompressed_gib": sum(
            int(record.get("decompressed_bytes", 0)) for record in resident_loads
        )
        / (1024**3),
        "n_shards": len(per_shard),
        "wall_s": sidecar["totals"]["wall_seconds"],
        "rows_per_sec": sidecar["totals"]["rows_per_sec"],
        "threads_max": resources.threads_max,
        "cpu_cores_avg": resources.cpu_cores_avg,
        "rss_peak_gib": resources.rss_peak_bytes / (1024**3),
        "pss_peak_gib": (
            None
            if resources.pss_peak_bytes is None
            else resources.pss_peak_bytes / (1024**3)
        ),
        "swap_peak_gib": (
            None
            if resources.swap_peak_bytes is None
            else resources.swap_peak_bytes / (1024**3)
        ),
    }


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _markdown(rows: list[tuple[str, dict]]) -> str:
    stage_keys = list(rows[0][1]["stage_totals"]) if rows else []
    header = (
        "| variant | identity | scan_s | pre_pass_s | cache_hit | base_loads_s | "
        + " | ".join(f"{key}_s" for key in stage_keys)
        + " | row_to_csr_s | row_provenance_s | packed_save med/p90 | worker med/p90 | queue_wait_s | "
        "rows/s | wall_s | cpu cores | threads | "
        "rss GiB | pss GiB | swap GiB |"
    )
    sep = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
    lines = [header, sep]
    for row_index, (name, row) in enumerate(rows):
        stage_cells = " | ".join(
            f"{row['stage_totals'].get(key, 0.0):.3f}" for key in stage_keys
        )
        save_median, save_p90 = row["save_med_p90"]
        worker_median, worker_p90 = row["worker_med_p90"]
        identity = "reference" if row_index == 0 else "exact"
        lines.append(
            f"| {name} | {identity} | {row['feature_index_scan_s']:.3f} | "
            f"{row['pre_pass_s']:.3f} | {row['cache_hit']} | "
            f"{row['base_loads_s']:.3f} | {stage_cells} | "
            f"{row['row_build_detail']['row_to_csr_seconds']:.3f} | "
            f"{row['row_build_detail']['row_provenance_seconds']:.3f} | "
            f"{save_median:.2f}/{save_p90:.2f} | "
            f"{worker_median:.2f}/{worker_p90:.2f} | {row['queue_wait_s']:.2f} | "
            f"{row['rows_per_sec']:.1f} | {row['wall_s']:.1f} | "
            f"{row['cpu_cores_avg']:.2f} | {row['threads_max']} | "
            f"{row['rss_peak_gib']:.2f} | {_format_optional_float(row['pss_peak_gib'])} | "
            f"{_format_optional_float(row['swap_peak_gib'])} |"
        )
    lines.extend(
        [
            "",
            "Row-to-CSR phase totals (contained by row_to_csr_s and csr_build_s):",
            "",
            "| variant | build_segments_s | segment_boundaries_s | feature_concat_s | "
            "row_tensor_finalize_s | row_provenance_s |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, row in rows:
        detail = row["row_build_detail"]
        lines.append(
            f"| {name} | {detail.get('build_segments_seconds', 0.0):.3f} | "
            f"{detail.get('segment_boundaries_seconds', 0.0):.3f} | "
            f"{detail.get('feature_concat_seconds', 0.0):.3f} | "
            f"{detail.get('row_tensor_finalize_seconds', 0.0):.3f} | "
            f"{detail.get('row_provenance_seconds', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "Feature-concatenation phase totals (contained by feature_concat_s):",
            "",
            "| variant | feature_ids_concat_s | percentiles_concat_s | "
            "feature_counts_s |",
            "|---|---|---|---|",
        ]
    )
    for name, row in rows:
        detail = row["row_build_detail"]
        lines.append(
            f"| {name} | {detail.get('feature_ids_concat_seconds', 0.0):.3f} | "
            f"{detail.get('percentiles_concat_seconds', 0.0):.3f} | "
            f"{detail.get('feature_counts_seconds', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "FeatureIndex resident-load phase totals (background-prefetched work can overlap "
            "the main critical path):",
            "",
            "| variant | loads | compressed GiB | decompressed GiB | "
            + " | ".join(f"{key}_s" for key in _LOAD_DETAIL_KEYS)
            + " |",
            "|" + "|".join(["---"] * (4 + len(_LOAD_DETAIL_KEYS))) + "|",
        ]
    )
    for name, row in rows:
        detail_cells = " | ".join(
            f"{row['load_detail_totals'][key]:.3f}" for key in _LOAD_DETAIL_KEYS
        )
        lines.append(
            f"| {name} | {row['resident_load_count']} | "
            f"{row['resident_compressed_gib']:.3f} | "
            f"{row['resident_decompressed_gib']:.3f} | {detail_cells} |"
        )
    return "\n".join(lines) + "\n"


def run_bench(cfg: DictConfig) -> str:
    """Run isolated variants, require parity, and write a Markdown report."""
    source_feature_dir = Path(str(cfg.feature_dir))
    source_data_dir = Path(str(cfg.get("data_dir", "")))
    segment_size = int(cfg.segment_size)
    max_shards_raw = cfg.get("max_shards", None)
    if max_shards_raw is None:
        raise ValueError(
            "benchmark_packed_rows: max_shards is required (e.g. max_shards=4) "
            "so every benchmark uses a bounded, private s{N}/ output."
        )
    try:
        max_shards = int(max_shards_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark_packed_rows: max_shards must be a positive integer") from exc
    if max_shards <= 0:
        raise ValueError("benchmark_packed_rows: max_shards must be a positive integer")
    if "sae_width" not in cfg:
        raise ValueError(
            "benchmark_packed_rows: sae_width is required so packed-row source "
            "certification uses the feature directory's actual SAE width."
        )
    if cfg.get("data_dir", None) is None:
        raise ValueError("benchmark_packed_rows: data_dir is required")
    sae_width = str(cfg.sae_width)
    context_len = int(cfg.get("context_len", DEFAULT_CONTEXT_WINDOW))
    variants = cfg.get("variants", None) or _DEFAULT_VARIANTS

    run_root = Path(
        str(
            cfg.get(
                "run_dir",
                Path(cfg.get("bench_root", "out/bench/packed_rows_runs"))
                / f"run_{time.time_ns()}_{os.getpid()}",
            )
        )
    )
    sidecar_dir = Path(str(cfg.get("sidecar_dir", run_root / "sidecars")))
    report_md = Path(str(cfg.get("report_md", run_root / "packed_rows.md")))
    source_dirs = (source_feature_dir, source_data_dir)
    for path, label in (
        (run_root, "run_dir"),
        (sidecar_dir, "sidecar_dir"),
        (report_md, "report_md"),
    ):
        _assert_benchmark_path_isolated(path, source_dirs=source_dirs, label=label)
    if os.path.lexists(run_root):
        raise ValueError(
            f"benchmark_packed_rows: run_dir must be a fresh nonexistent path: {run_root}"
        )

    prepared: list[tuple[str, Path, Path, DictConfig]] = []
    seen_names: set[str] = set()
    base_cfg = {
        key: cfg[key]
        for key in _BASE_CONFIG_KEYS
        if key in cfg and cfg.get(key, None) is not None
    }
    for variant in variants:
        name = str(variant["name"])
        if name in seen_names:
            raise ValueError(f"duplicate benchmark variant name: {name!r}")
        seen_names.add(name)
        overrides = _validate_variant_overrides(name, variant.get("overrides", {}) or {})
        variant_feature_dir = run_root / "variants" / name / f"s{segment_size}" / "features"
        sidecar = sidecar_dir / f"{name}.json"
        run_cfg = load_packed_rows_config(
            {
                **base_cfg,
                "feature_dir": str(variant_feature_dir),
                "data_dir": str(run_root / "input_parquets"),
                "segment_size": segment_size,
                "sae_width": sae_width,
                "context_len": context_len,
                "max_shards": max_shards,
                "resume": False,
                "resume_from_shard": None,
                "profile": True,
                "profile_summary_output": str(sidecar),
            },
            overrides,
        )
        _validate_resolved_variant_output(run_cfg, variant_name=name)
        if Path(str(run_cfg.feature_dir)) != variant_feature_dir:
            raise ValueError(f"variant {name!r}: feature_dir escaped its private tree")
        if Path(str(run_cfg.data_dir)) != run_root / "input_parquets":
            raise ValueError(f"variant {name!r}: data_dir escaped the mirrored input tree")
        if Path(str(run_cfg.profile_summary_output)) != sidecar:
            raise ValueError(f"variant {name!r}: profile sidecar escaped its private path")
        prepared.append((name, sidecar, variant_feature_dir, run_cfg))

    # No producer call or filesystem mutation happens before every variant has
    # passed the safety preflight above.
    parquet_sources = _prepare_parquets(
        source_data_dir, run_root / "input_parquets", max_shards=max_shards
    )
    selected_stems = [path.stem for path in parquet_sources]
    feature_sources: list[Path] | None = None
    variant_manifests: list[list[tuple[Path, dict, OverlayKind]]] = []
    for _name, _sidecar, feature_dir, _run_cfg in prepared:
        mirrored, manifests = _prepare_feature_tree(
            source_feature_dir,
            feature_dir,
            selected_stems=selected_stems,
        )
        variant_manifests.append(manifests)
        if feature_sources is None:
            feature_sources = mirrored
        elif sorted(path.resolve() for path in mirrored) != sorted(
            path.resolve() for path in feature_sources
        ):
            raise AssertionError("variant feature mirrors do not bind the same source inventory")
    assert feature_sources is not None
    first_feature_dir = prepared[0][2]
    base_hash_started = time.perf_counter()
    bounded_base_hash = compute_base_shard_content_hash(
        glob_features(first_feature_dir),
        cache_dir=first_feature_dir,
    )
    base_hash_seconds = time.perf_counter() - base_hash_started
    _write_bounded_input_manifests(
        variant_manifests[0], base_hash=bounded_base_hash
    )
    first_cache = first_feature_dir / BASE_SHARD_HASH_CACHE_FILENAME
    if not first_cache.is_file():
        raise AssertionError(f"bounded base-hash computation wrote no cache: {first_cache}")
    for (_name, _sidecar, feature_dir, _run_cfg), manifests in zip(
        prepared[1:], variant_manifests[1:], strict=True
    ):
        _mirror_file(
            first_cache,
            feature_dir / BASE_SHARD_HASH_CACHE_FILENAME,
            copy=True,
        )
        _write_bounded_input_manifests(manifests, base_hash=bounded_base_hash)
    source_inventory = _stat_inventory([*feature_sources, *parquet_sources])
    rendered_inventory = json.dumps(source_inventory, indent=2, sort_keys=True) + "\n"
    inventory_sha256 = hashlib.sha256(rendered_inventory.encode()).hexdigest()
    atomic_write_text(
        rendered_inventory,
        run_root / "input_inventory.json",
    )
    atomic_write_text(
        json.dumps(
            {
                "selected_stems": selected_stems,
                "bounded_base_shard_content_hash": bounded_base_hash,
                "bounded_base_hash_seconds": base_hash_seconds,
                "input_stat_inventory_sha256": inventory_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        run_root / "run_metadata.json",
    )
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for name, _sidecar, _feature_dir, run_cfg in prepared:
        resolved = OmegaConf.to_container(run_cfg, resolve=True)
        atomic_write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            sidecar_dir / f"{name}.config.json",
        )

    rows: list[tuple[str, dict]] = []
    digest_maps: list[tuple[str, dict[str, str]]] = []
    for name, sidecar, feature_dir, run_cfg in prepared:
        _assert_inventory_unchanged(source_inventory, variant_name=name, stage="preflight")
        ret, resources = sample_process_tree(lambda cfg=run_cfg: run_packed(cfg))
        _assert_inventory_unchanged(source_inventory, variant_name=name, stage="execution")
        assert ret == 0, f"variant {name!r} failed with exit code {ret}"
        assert sidecar.is_file(), f"variant {name!r} wrote no sidecar at {sidecar}"
        artifact_map = _artifact_digests(
            feature_dir, context_len=context_len, max_shards=max_shards
        )
        artifact_path = sidecar_dir / f"{name}.artifacts.json"
        atomic_write_text(
            json.dumps(artifact_map, indent=2, sort_keys=True) + "\n",
            artifact_path,
        )
        digest_maps.append((name, artifact_map))
        rows.append((name, _summarize(json.loads(sidecar.read_text()), resources)))

    if digest_maps:
        reference_name, reference = digest_maps[0]
        for name, digests in digest_maps[1:]:
            divergent = sorted(
                path
                for path in set(reference) | set(digests)
                if reference.get(path) != digests.get(path)
            )
            if divergent:
                raise AssertionError(
                    f"packed-row artifact mismatch between {reference_name!r} and "
                    f"{name!r}: paths={divergent}"
                )

    report = (
        f"Run root: `{run_root}`  \n"
        f"Input stat-inventory SHA-256: `{inventory_sha256}`\n\n"
        f"Bounded base content SHA-256: `{bounded_base_hash}` "
        f"(one-time setup hash: {base_hash_seconds:.2f}s)\n\n"
        + _markdown(rows)
    )
    atomic_write_text(report, report_md)
    return report


def main() -> None:
    cfg = OmegaConf.from_cli()
    missing = [
        key
        for key in ("feature_dir", "data_dir", "segment_size", "sae_width", "max_shards")
        if key not in cfg
    ]
    if missing:
        raise SystemExit(
            f"benchmark_packed_rows: required arg(s) missing: {'='.join(missing)}=... "
            "(e.g. feature_dir=... data_dir=... segment_size=128 "
            "sae_width=16k max_shards=4)"
        )
    print(run_bench(cfg))


if __name__ == "__main__":
    main()
