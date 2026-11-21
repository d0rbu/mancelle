"""Profile the full autointerp source-index build (`_build_index`).

The source-index merge uses a vectorized top-k strategy to combine results from
multiple shards. The --scan-workers timing overhead (≥2.5×) is validated
separately on a calm box; see #649 for details.

Run one process per implementation so ``ru_maxrss`` is attributable:

    uv run python -m scripts.benchmark.benchmark_autointerp_source_index_build \
      --feature-dir path/to/features --data-dir path/to/base_data \
      --n-sae-features 65536 --blacklist-file path/to/freq.txt \
      --n-shards 4 --temp-dir /scratch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sqlite3
import time
from pathlib import Path

from errors import ManualFeatureSelectionError
from steering.eval.autointerp_filter.source_index import (
    SourceIndexProgressEvent,
    SourceIndexProgressEventKind,
    SourceIndexRequest,
    fingerprint_paths,
    load_or_build_source_index,
)
from steering.feature_io import glob_features
from steering.manual_feature_ids import parse_feature_ids_txt


def _read_ids(path: Path) -> set[int]:
    try:
        return set(parse_feature_ids_txt(path))
    except ManualFeatureSelectionError as exc:
        raise SystemExit(f"invalid --blacklist-file {path}: {exc}") from exc


def _table_sha(index_path: Path, table: str, order: str) -> str:
    conn = sqlite3.connect(str(index_path))
    try:
        h = hashlib.sha256()
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            h.update(json.dumps(list(row), sort_keys=True).encode())
        return h.hexdigest()
    finally:
        conn.close()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _summarize_progress_events(
    events: list[SourceIndexProgressEvent],
    *,
    scan_workers: int,
    n_shards: int,
) -> dict:
    shard_done = [
        event
        for event in events
        if event.kind == SourceIndexProgressEventKind.SHARD_DONE
    ]
    waits = [
        float(event.wait_seconds)
        for event in shard_done
        if event.wait_seconds is not None
    ]
    merges = [
        float(event.merge_seconds)
        for event in shard_done
        if event.merge_seconds is not None
    ]
    shard_elapsed = [
        float(event.elapsed_seconds)
        for event in shard_done
        if event.elapsed_seconds is not None
    ]
    spools = [
        int(event.spool_bytes)
        for event in events
        if event.kind == SourceIndexProgressEventKind.SHARD_SPOOLED
        and event.spool_bytes is not None
    ]
    build_done = next(
        (
            event
            for event in reversed(events)
            if event.kind == SourceIndexProgressEventKind.BUILD_DONE
        ),
        None,
    )
    max_spool_bytes = max(spools, default=0)
    concurrent_spools = min(scan_workers, n_shards)
    return {
        "progress_event_count": len(events),
        "shard_done_event_count": len(shard_done),
        "shard_spooled_event_count": len(spools),
        "chunk_done_event_count": sum(
            1 for event in events if event.kind == SourceIndexProgressEventKind.CHUNK_DONE
        ),
        "build_progress_elapsed_s": (
            None if build_done is None else build_done.elapsed_seconds
        ),
        "shard_elapsed_sum_s": sum(shard_elapsed),
        "shard_elapsed_max_s": max(shard_elapsed, default=None),
        "shard_elapsed_mean_s": _mean(shard_elapsed),
        "wait_sum_s": sum(waits),
        "wait_max_s": max(waits, default=None),
        "wait_mean_s": _mean(waits),
        "merge_sum_s": sum(merges),
        "merge_max_s": max(merges, default=None),
        "merge_mean_s": _mean(merges),
        "max_spool_bytes": max_spool_bytes,
        "max_spool_gib": max_spool_bytes / 1024**3,
        "estimated_peak_concurrent_spool_bytes": max_spool_bytes * concurrent_spools,
        "estimated_peak_concurrent_spool_gib": (
            max_spool_bytes * concurrent_spools / 1024**3
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--n-sae-features", type=int, required=True)
    p.add_argument("--blacklist-file", type=Path, default=None)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--max-positive-refs", type=int, default=128)
    p.add_argument("--max-intruder-refs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=5_000_000)
    p.add_argument("--scan-workers", type=int, default=1)
    p.add_argument("--temp-dir", type=Path, required=True)
    args = p.parse_args()

    shards = tuple(glob_features(args.feature_dir))[: args.n_shards]
    if len(shards) < args.n_shards:
        raise SystemExit(f"only {len(shards)} shards found, need {args.n_shards}")
    if args.blacklist_file is not None:
        bl = _read_ids(args.blacklist_file)
        targets = tuple(f for f in range(args.n_sae_features) if f not in bl)
    else:
        targets = tuple(range(args.n_sae_features))

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.temp_dir / "build_bench_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    request = SourceIndexRequest(
        feature_dir=args.feature_dir,
        data_dir=args.data_dir,
        index_path=out_dir / "source_doc_index.sqlite",
        manifest_path=out_dir / "source_doc_index.manifest.json",
        shard_paths=shards,
        parquet_paths=(),
        target_feature_ids=targets,
        n_sae_features=args.n_sae_features,
        max_positive_refs_per_feature=args.max_positive_refs,
        max_intruder_refs=args.max_intruder_refs,
        seed=args.seed,
        feature_fingerprint=fingerprint_paths(shards),
        data_fingerprint=fingerprint_paths(()),
        context_independent_features=True,
        ignore_bos_features=True,
        chunk_size=args.chunk_size,
        scan_workers=args.scan_workers,
        force_rebuild=True,
    )
    progress_events: list[SourceIndexProgressEvent] = []
    start = time.perf_counter()
    data = load_or_build_source_index(request, progress_callback=progress_events.append)
    elapsed = time.perf_counter() - start
    payload = {
        "n_shards": args.n_shards,
        "scan_workers": args.scan_workers,
        "target_count": len(targets),
        "elapsed_s": elapsed,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **_summarize_progress_events(
            progress_events,
            scan_workers=args.scan_workers,
            n_shards=args.n_shards,
        ),
        "positive_refs_sha256": _table_sha(
            request.index_path, "positive_refs", "feature_id, rank"
        ),
        "intruder_refs_sha256": _table_sha(
            request.index_path, "intruder_refs", "rank"
        ),
        "n_positive_features": len(data.positives_by_feature),
        "n_intruders": len(data.intruder_refs),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
