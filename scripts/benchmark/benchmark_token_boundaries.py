"""Benchmark harness for precompute_token_boundaries (issue #529).

Runs the token-boundary pipeline once per named *variant* (a set of config overrides),
each with ``profile=true`` + ``force=true`` so every variant does the full work, then reads
each run's JSON profiling sidecar and emits a Markdown table of per-stage wall-clock
(median, p90 across shards) plus the manifest pre-pass and process-tree resources.

Headline usage (real 370-shard pt corpus) — measure the base-hash cache win cold vs warm. Pass
``workers=N`` to also exercise worker-pool reuse; the default ``workers=1`` runs tokenization
sequentially, so the pool is NOT measured (only the cache + subset-load wins are)::

    uv run python -m scripts.benchmark.benchmark_token_boundaries \\
        feature_dir=<scratch copy> segment_size=128 \\
        'variants=[{name: w8, overrides: {workers: 8}}, {name: w16, overrides: {workers: 16}}, {name: w24, overrides: {workers: 24}}, {name: w32, overrides: {workers: 32}}]'

The default variants are ``cold`` (first run → streams the base hash + writes the cache) and
``warm`` (second run on the same ``feature_dir`` → cache hit, ~0 s pre-pass). The base-hash cache
is cleared once before the first variant, so ``cold`` is genuinely cold even on a ``feature_dir``
that already carries a cache; the warm run reuses the cache the cold run wrote, so variants run in
order on the same ``feature_dir``.
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.benchmark._common import _median, _p90
from scripts.benchmark._resource_sampler import ResourceSample, sample_process_tree
from scripts.preprocess.precompute_token_boundaries import run as run_tok
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    overlay_content_digest,
)

# cold = first run (streams the base hash + writes the cache); warm = second run on the same
# feature_dir (cache hit). Both run the unconditional optimized path — the cache self-populates
# on the cold run, so no config flag is needed to A/B it.
_DEFAULT_VARIANTS = [
    {"name": "cold", "overrides": {}},
    {"name": "warm", "overrides": {}},
]

_STAGES = ("load_features_seconds", "tokenize_seconds", "overlay_save_seconds")


def _summarize(sidecar: dict, resources: ResourceSample) -> dict:
    per_shard = sidecar.get("per_shard", [])
    row = {
        "pre_pass_s": sidecar["manifest_pre_pass"]["seconds"],
        "bytes_read": sidecar["manifest_pre_pass"]["bytes_read"],
        "cache_hit": sidecar["manifest_pre_pass"]["cache_hit"],
        "wall_s": sidecar["totals"]["wall_seconds"],
        "n_shards": len(per_shard),
        "threads_max": resources.threads_max,
        "cpu_cores_avg": resources.cpu_cores_avg,
        "rss_peak_mb": resources.rss_peak_bytes / (1024 * 1024),
    }
    for stage in _STAGES:
        vals = [float(s.get(stage, 0.0)) for s in per_shard]
        row[stage] = (_median(vals), _p90(vals))
    return row


def _markdown(rows: list[tuple[str, dict]]) -> str:
    header = (
        "| variant | pre_pass_s | bytes_read | cache_hit | wall_s | "
        "threads_max | cpu_cores_avg | rss_peak_mb | "
        "load med/p90 | tokenize med/p90 | save med/p90 |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines = [header, sep]
    for name, r in rows:
        load_m, load_p = r["load_features_seconds"]
        tok_m, tok_p = r["tokenize_seconds"]
        sav_m, sav_p = r["overlay_save_seconds"]
        lines.append(
            f"| {name} | {r['pre_pass_s']:.3f} | {r['bytes_read']} | {r['cache_hit']} | "
            f"{r['wall_s']:.2f} | {r['threads_max']} | {r['cpu_cores_avg']:.2f} | "
            f"{r['rss_peak_mb']:.1f} | {load_m:.3f}/{load_p:.3f} | "
            f"{tok_m:.3f}/{tok_p:.3f} | {sav_m:.3f}/{sav_p:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_bench(cfg: DictConfig) -> str:
    """Run each variant, tabulate per-stage timings, write + return the Markdown report."""
    feature_dir = str(cfg.feature_dir)
    data_dir = cfg.get("data_dir", None)
    segment_size = int(cfg.segment_size)
    variants = cfg.get("variants", None) or _DEFAULT_VARIANTS

    sidecar_dir = Path(cfg.get("sidecar_dir", "out/bench/tok_boundary_sidecars"))
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    # Clear any pre-existing base-hash cache so the FIRST variant is a genuine cold miss even on a
    # feature_dir that already carries one (the cold/warm split now comes from the cache
    # self-populating, not the removed base_shard_hash_cache=false flag).
    (Path(feature_dir) / BASE_SHARD_HASH_CACHE_FILENAME).unlink(missing_ok=True)

    rows: list[tuple[str, dict]] = []
    digest_maps: list[tuple[str, dict[str, str]]] = []
    for variant in variants:
        name = str(variant["name"])
        sidecar = sidecar_dir / f"{name}.json"
        run_cfg = OmegaConf.merge(
            OmegaConf.create(
                {
                    "feature_dir": feature_dir,
                    "data_dir": data_dir,
                    "segment_size": segment_size,
                    "profile": True,
                    # Sidecar only — the first-shard chrome trace OOMs on real
                    # shards and forces the first shard sequential, which would
                    # distort the worker sweep (issue #1269).
                    "profile_trace": False,
                    "profile_summary_output": str(sidecar),
                    "force": True,
                    "resume": False,
                    "dry_run": False,
                    "workers": int(cfg.get("workers", 1)),
                }
            ),
            variant.get("overrides", {}) or {},
        )
        assert isinstance(run_cfg, DictConfig), "merged bench cfg must be a DictConfig"
        ret, resources = sample_process_tree(lambda cfg=run_cfg: run_tok(cfg))
        assert ret == 0, f"variant {name!r} failed with exit code {ret}"
        assert sidecar.is_file(), f"variant {name!r} wrote no sidecar at {sidecar}"
        sidecar_payload = json.loads(sidecar.read_text())
        overlay_dirs = sorted((Path(feature_dir) / "tok").glob("*/seg_boundaries"))
        if len(overlay_dirs) != 1:
            raise AssertionError(
                f"variant {name!r} expected one token overlay directory, found {overlay_dirs}"
            )
        digest_map = {
            path.name: overlay_content_digest(path)
            for path in sorted(overlay_dirs[0].glob("*.safetensors.zst"))
        }
        digest_path = sidecar_dir / f"{name}.digests.json"
        digest_path.write_text(json.dumps(digest_map, indent=2, sort_keys=True) + "\n")
        digest_maps.append((name, digest_map))
        rows.append((name, _summarize(sidecar_payload, resources)))

    if digest_maps:
        reference_name, reference = digest_maps[0]
        for name, digests in digest_maps[1:]:
            diverging_stems = sorted(
                stem
                for stem in set(reference) | set(digests)
                if reference.get(stem) != digests.get(stem)
            )
            if diverging_stems:
                raise AssertionError(
                    f"overlay digest mismatch between {reference_name!r} and {name!r}: "
                    f"stems={diverging_stems}"
                )

    report = _markdown(rows)
    report_md = Path(cfg.get("report_md", "out/bench/tok_boundary.md"))
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(report)
    return report


def main() -> None:
    cfg = OmegaConf.from_cli()
    missing = [k for k in ("feature_dir", "segment_size") if k not in cfg]
    if missing:
        raise SystemExit(
            f"benchmark_token_boundaries: required arg(s) missing: {'='.join(missing)}=... "
            "(e.g. feature_dir=... segment_size=128)"
        )
    print(run_bench(cfg))


if __name__ == "__main__":
    main()
