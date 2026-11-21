"""Benchmark harness for post-extraction feature-shard splicing.

The harness mirrors a bounded set of base feature shards and matching parquets into
``out/bench`` using symlinks where possible, then runs each variant into a fresh
derived output directory with ``profile=true``. Production feature directories are
never written.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nanochat.dataset import list_parquet_files
from scripts.benchmark._common import _median, _p90
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import glob_features, shard_stem_from_path
from steering.feature_paths import parse_feature_dir
from steering.stats_io import find_statistics_path

_DEFAULT_VARIANTS = [{"name": "profile", "overrides": {}}]
_FORBIDDEN_VARIANT_OVERRIDE_KEYS = frozenset(
    {
        "feature_dir",
        "data_dir",
        "output_dir",
        "skip_existing",
        "write_provenance",
        "profile",
        "profile_summary_output",
    }
)


def _validate_variant_overrides(name: str, overrides: object) -> dict:
    if overrides is None:
        return {}
    if OmegaConf.is_config(overrides):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(overrides, dict):
        raise ValueError(
            f"variant {name!r}: overrides must be a mapping, got {type(overrides).__name__}."
        )
    blocked = sorted(str(key) for key in overrides if str(key) in _FORBIDDEN_VARIANT_OVERRIDE_KEYS)
    if blocked:
        raise ValueError(
            f"variant {name!r}: override(s) {blocked} are not allowed. "
            "benchmark variants must not replace mirrored inputs, output dirs, "
            "profile sidecars, or resume/provenance controls."
        )
    return dict(overrides)


def _assert_isolated_run_cfg(
    cfg: DictConfig,
    *,
    bench_feature_dir: Path,
    bench_data_dir: Path,
    output_dir: Path,
    sidecar: Path,
    run_root: Path,
    variant_name: str,
) -> None:
    if str(cfg.feature_dir) != str(bench_feature_dir):
        raise ValueError(f"variant {variant_name!r}: feature_dir escaped the mirrored input dir.")
    if str(cfg.data_dir) != str(bench_data_dir):
        raise ValueError(f"variant {variant_name!r}: data_dir escaped the mirrored parquet dir.")
    if str(cfg.output_dir) != str(output_dir):
        raise ValueError(f"variant {variant_name!r}: output_dir escaped the per-variant dir.")
    if str(cfg.profile_summary_output) != str(sidecar):
        raise ValueError(f"variant {variant_name!r}: profile_summary_output escaped sidecar dir.")
    if bool(cfg.skip_existing):
        raise ValueError(f"variant {variant_name!r}: skip_existing must stay false.")
    if not bool(cfg.write_provenance):
        raise ValueError(f"variant {variant_name!r}: write_provenance must stay true.")
    if not bool(cfg.profile):
        raise ValueError(f"variant {variant_name!r}: profile must stay true.")
    try:
        Path(cfg.output_dir).resolve().relative_to(run_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"variant {variant_name!r}: benchmark outputs must remain under {run_root}."
        ) from exc


def _mirror_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _resolve_model(cfg: DictConfig, feature_dir: Path) -> str:
    if cfg.get("model", None) is not None:
        return str(cfg.model)
    try:
        return parse_feature_dir(feature_dir).model
    except ValueError as exc:
        raise ValueError(
            "benchmark_splice_feature_shards: could not parse model from feature_dir; "
            "pass model=... explicitly."
        ) from exc


def _prepare_inputs(
    *,
    source_feature_dir: Path,
    source_data_dir: Path,
    run_root: Path,
    max_shards: int,
) -> tuple[Path, Path, list[str]]:
    source_shards = glob_features(source_feature_dir)
    if max_shards <= 0:
        raise ValueError(f"max_shards must be > 0, got {max_shards}")
    selected = source_shards[:max_shards]
    if len(selected) < max_shards:
        raise ValueError(
            f"requested max_shards={max_shards}, but only found {len(selected)} "
            f"feature shard(s) in {source_feature_dir}."
        )
    parquet_by_stem = {
        Path(path).stem: Path(path) for path in list_parquet_files(str(source_data_dir))
    }
    bench_feature_dir = run_root / "input_features"
    bench_data_dir = run_root / "input_parquets"
    stems: list[str] = []
    for shard in selected:
        stem = shard_stem_from_path(shard)
        parquet = parquet_by_stem.get(stem)
        if parquet is None:
            raise ValueError(f"no parquet matching selected shard {stem!r} in {source_data_dir}")
        stems.append(stem)
        _mirror_file(shard, bench_feature_dir / shard.name)
        _mirror_file(parquet, bench_data_dir / parquet.name)
    stats_path = find_statistics_path(source_feature_dir)
    if stats_path is not None:
        _mirror_file(stats_path, bench_feature_dir / stats_path.name)
    return bench_feature_dir, bench_data_dir, stems


def _summarize(sidecar: dict) -> dict:
    per_shard = sidecar.get("per_shard", [])
    phase_keys = list(sidecar.get("totals", {}).get("phase_seconds", {}))
    phase_summary = {}
    for key in phase_keys:
        values = [float(r.get("phase_seconds", {}).get(key, 0.0)) for r in per_shard]
        phase_summary[key] = (_median(values), _p90(values))
    detail_keys = list(sidecar.get("totals", {}).get("detail_seconds", {}))
    detail_summary = {}
    for key in detail_keys:
        values = [float(r.get("detail_seconds", {}).get(key, 0.0)) for r in per_shard]
        detail_summary[key] = {
            "median": _median(values),
            "p90": _p90(values),
            "total": float(sidecar.get("totals", {}).get("detail_seconds", {}).get(key, 0.0)),
        }
    wall = float(sidecar["totals"]["wall_seconds"])
    n_ok = int(sidecar["totals"]["n_ok"])
    return {
        "n_ok": n_ok,
        "n_failed": int(sidecar["totals"]["n_failed"]),
        "wall_s": wall,
        "shards_per_hour": (n_ok / wall * 3600.0) if wall > 0 else 0.0,
        "output_bytes": int(sidecar["totals"]["output_bytes"]),
        "cpu_seconds": float(sidecar["totals"]["cpu_seconds"]),
        "max_hwm_mb": sidecar["totals"]["max_hwm_mb"],
        "phase_summary": phase_summary,
        "detail_summary": detail_summary,
    }


def _markdown(rows: list[tuple[str, dict]], *, stems: list[str]) -> str:
    phase_keys = list(rows[0][1]["phase_summary"]) if rows else []
    header = (
        "| variant | shards | wall_s | shards/hour | output_mb | cpu_s | max_hwm_mb | "
        + " | ".join(f"{key} med/p90" for key in phase_keys)
        + " |"
    )
    sep = "|" + "|".join(["---"] * (7 + len(phase_keys))) + "|"
    lines = [
        "# Feature-Shard Splice Benchmark",
        "",
        f"Shards: {', '.join(stems)}",
        "",
        header,
        sep,
    ]
    for name, row in rows:
        phase_cells = " | ".join(
            f"{row['phase_summary'][key][0]:.3f}/{row['phase_summary'][key][1]:.3f}"
            for key in phase_keys
        )
        hwm = row["max_hwm_mb"]
        hwm_text = "n/a" if hwm is None else f"{float(hwm):.1f}"
        lines.append(
            f"| {name} | {row['n_ok']} | {row['wall_s']:.3f} | "
            f"{row['shards_per_hour']:.1f} | {row['output_bytes'] / 1_000_000:.2f} | "
            f"{row['cpu_seconds']:.3f} | {hwm_text} | {phase_cells} |"
        )
    detail_rows: list[str] = []
    for name, row in rows:
        ranked = sorted(
            row["detail_summary"].items(),
            key=lambda kv: (-float(kv[1]["total"]), kv[0]),
        )
        for key, summary in ranked:
            total = float(summary["total"])
            if total <= 0.0:
                continue
            detail_rows.append(
                f"| {name} | {key} | {summary['median']:.3f} | {summary['p90']:.3f} | {total:.3f} |"
            )
    if detail_rows:
        lines.extend(
            [
                "",
                "## Detail Timings",
                "",
                "| variant | detail | median_s | p90_s | total_s |",
                "|---|---|---:|---:|---:|",
                *detail_rows,
            ]
        )
    return "\n".join(lines) + "\n"


def run_bench(cfg: DictConfig) -> str:
    feature_dir = Path(cfg.feature_dir)
    data_dir = Path(cfg.data_dir)
    max_shards = int(cfg.max_shards)
    model = _resolve_model(cfg, feature_dir)
    variants = cfg.get("variants", None) or _DEFAULT_VARIANTS

    bench_root = Path(cfg.get("bench_root", "out/bench/splice_feature_shards"))
    run_root = bench_root / f"run_{int(time.time())}_{os.getpid()}"
    sidecar_dir = Path(cfg.get("sidecar_dir", run_root / "sidecars"))
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    bench_feature_dir, bench_data_dir, stems = _prepare_inputs(
        source_feature_dir=feature_dir,
        source_data_dir=data_dir,
        run_root=run_root,
        max_shards=max_shards,
    )

    rows: list[tuple[str, dict]] = []
    for variant in variants:
        name = str(variant["name"])
        overrides = _validate_variant_overrides(name, variant.get("overrides", {}) or {})
        sidecar = sidecar_dir / f"{name}.json"
        output_dir = run_root / "outputs" / name
        run_cfg = sfs.load_splice_config(
            {
                "feature_dir": str(bench_feature_dir),
                "data_dir": str(bench_data_dir),
                "output_dir": str(output_dir),
                "model": model,
                "segment_size": cfg.get("segment_size", None),
                "context_window": cfg.get("context_window", None),
                "cut_fraction": float(cfg.get("cut_fraction", 0.15)),
                "seed": int(cfg.get("seed", 42)),
                "skip_existing": False,
                "write_provenance": True,
                "shard_parallelism": int(cfg.get("shard_parallelism", 1)),
                "profile": True,
                "profile_summary_output": str(sidecar),
            },
            overrides,
        )
        _assert_isolated_run_cfg(
            run_cfg,
            bench_feature_dir=bench_feature_dir,
            bench_data_dir=bench_data_dir,
            output_dir=output_dir,
            sidecar=sidecar,
            run_root=run_root,
            variant_name=name,
        )
        sfs.run(run_cfg)
        if not sidecar.is_file():
            raise RuntimeError(f"variant {name!r} wrote no profile sidecar at {sidecar}")
        summary = _summarize(json.loads(sidecar.read_text(encoding="utf-8")))
        if summary["n_failed"]:
            raise RuntimeError(f"variant {name!r} had {summary['n_failed']} failed shard(s)")
        rows.append((name, summary))

    report = _markdown(rows, stems=stems)
    report_md = Path(cfg.get("report_md", bench_root / "splice_feature_shards.md"))
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    cfg = OmegaConf.from_cli()
    missing = [key for key in ("feature_dir", "data_dir", "max_shards") if key not in cfg]
    if missing:
        raise SystemExit(
            "benchmark_splice_feature_shards: required arg(s) missing: "
            + ", ".join(f"{key}=..." for key in missing)
        )
    print(run_bench(cfg))


if __name__ == "__main__":
    main()
