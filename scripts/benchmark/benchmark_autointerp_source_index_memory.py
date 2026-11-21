"""Profile autointerp source-index shard scanning memory.

POSIX ``ru_maxrss`` remains attributable to the production scanner:

    uv run python -m scripts.benchmark.benchmark_autointerp_source_index_memory \
      --shard-path /path/to/shard_00000_features.safetensors.zst \
      --n-sae-features 65536 --blacklist-file /path/to/freq_min0.0001_max0.5.txt \
      --target-limit 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import struct
import time
from dataclasses import asdict
from pathlib import Path

import torch as t
import zstandard

from errors import ManualFeatureSelectionError
from steering.eval.autointerp_filter.source_index import (
    SourceIndexProgressEvent,
    SourceIndexProgressEventKind,
    _scan_source_index_shard_file_topk_vectorized,
    segment_token_bounds,
    shard_index_from_stem,
)
from steering.eval.autointerp_filter.types import SourceDocRef
from steering.feature_io import TensorKey, load_features, save_features
from steering.manual_feature_ids import parse_feature_ids_txt

# The chunk-independent brute-force oracle (`_brute_force_topk_rows`) loops
# over every CSR entry in Python, so `--assert-exact` only uses it as the
# ground-truth check on bounded runs at or below this many entries; larger
# real shards run without the brute-force assertion.
_ASSERT_EXACT_BRUTE_FORCE_MAX_ENTRIES = 2_000_000


def _read_id_file(path: Path) -> set[int]:
    try:
        return set(parse_feature_ids_txt(path))
    except ManualFeatureSelectionError as exc:
        raise SystemExit(f"invalid feature-ID file {path}: {exc}") from exc


def _resolve_target_ids(args: argparse.Namespace) -> tuple[int, ...]:
    if args.target_ids_file is not None:
        targets = tuple(sorted(_read_id_file(args.target_ids_file)))
    elif args.blacklist_file is not None:
        blacklisted = _read_id_file(args.blacklist_file)
        targets = tuple(fid for fid in range(args.n_sae_features) if fid not in blacklisted)
    else:
        targets = tuple(range(args.n_sae_features))
    if args.target_offset:
        targets = targets[args.target_offset :]
    if args.target_limit is not None:
        targets = targets[: args.target_limit]
    return targets


class _RowDigest:
    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.row_count = 0

    def update_rows(self, rows: list[tuple]) -> None:
        self.row_count += len(rows)
        h = self._hash
        for row in rows:
            h.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            h.update(b"\n")

    def hexdigest(self, intruders: list) -> str:
        h = self._hash.copy()
        h.update(b"--intruders--\n")
        for ref in intruders:
            h.update(ref.stable_key().encode())
            h.update(b"\n")
        return h.hexdigest()


def _hash_rows(rows: list[tuple], intruders: list) -> str:
    digest = _RowDigest()
    digest.update_rows(rows)
    return digest.hexdigest(intruders)


def _read_exact(reader, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = reader.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _brute_force_topk_rows(
    shard_path: Path,
    target_feature_ids: tuple[int, ...],
    n_sae_features: int,
    max_positive_refs: int,
) -> list[tuple]:
    """Chunk-independent ground-truth reduction over all raw CSR entries.

    Per-``(feature, stable_key)`` max-magnitude dedup over every entry, then
    per-feature top-k by ``(magnitude desc, stable_key asc)``. It loops in
    Python, hence the caller gates it to bounded runs.
    """
    data = load_features(shard_path)
    offsets = data[TensorKey.OFFSETS].to(t.int64)
    fids = data[TensorKey.FEATURE_IDS].to(t.int64)
    mags = data[TensorKey.FEATURE_MAGS].to(t.float32)
    doc_ids = data[TensorKey.DOC_IDS].to(t.int64)
    n_rows = int(offsets.numel()) - 1
    starts, ends = segment_token_bounds(data, shard_path=shard_path, n_rows=n_rows)
    stem = shard_path.stem.removesuffix(".safetensors")
    shard_index = shard_index_from_stem(stem)
    refs = [
        SourceDocRef(
            shard_stem=stem,
            shard_index=shard_index,
            doc_id=int(doc_ids[row].item()),
            segment_token_start=int(starts[row].item()),
            segment_token_end=int(ends[row].item()),
        )
        for row in range(n_rows)
    ]
    allowed = set(target_feature_ids)
    buckets: dict[int, dict[str, tuple[float, SourceDocRef]]] = {}
    row_index = 0
    for entry_index in range(int(fids.numel())):
        while entry_index >= int(offsets[row_index + 1].item()):
            row_index += 1
        fid = int(fids[entry_index].item())
        if fid < 0 or fid >= n_sae_features or fid not in allowed:
            continue
        mag = float(mags[entry_index].item())
        ref = refs[row_index]
        key = ref.stable_key()
        bucket = buckets.setdefault(fid, {})
        old = bucket.get(key)
        if old is None or mag > old[0]:
            bucket[key] = (mag, ref)
    rows: list[tuple] = []
    for fid in sorted(buckets):
        ordered = sorted(buckets[fid].items(), key=lambda item: (-item[1][0], item[0]))
        for key, (mag, ref) in ordered[:max_positive_refs]:
            rows.append(
                (
                    fid,
                    mag,
                    key,
                    ref.shard_stem,
                    ref.shard_index,
                    ref.doc_id,
                    ref.segment_token_start,
                    ref.segment_token_end,
                )
            )
    return rows


def _read_header(path: Path) -> dict:
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as src:
        with dctx.stream_reader(src) as reader:
            size_bytes = _read_exact(reader, 8)
            if len(size_bytes) != 8:
                raise SystemExit(f"Unable to read safetensors header length from {path}")
            header_size = struct.unpack("<Q", size_bytes)[0]
            header_bytes = _read_exact(reader, header_size)
            if len(header_bytes) != header_size:
                raise SystemExit(f"Unable to read full safetensors header from {path}")
    return json.loads(header_bytes)


def _count_entries(path: Path) -> int:
    header = _read_header(path)
    entry_meta = header.get("feature_ids")
    if not isinstance(entry_meta, dict):
        raise SystemExit(f"{path} does not contain feature_ids metadata")
    shape = entry_meta.get("shape")
    if not isinstance(shape, list) or not shape:
        raise SystemExit(f"{path} feature_ids metadata does not contain a valid shape")
    return int(shape[0])


def _dir_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_synthetic_shard(temp_dir: Path, n_entries: int) -> Path:
    generator = t.Generator().manual_seed(0)
    mean_row = 2617
    n_rows = max(1, n_entries // mean_row)
    row_lengths: list[int] = []
    remaining = n_entries
    for row_index in range(n_rows):
        if row_index == n_rows - 1:
            row_lengths.append(remaining)
            break
        length = max(
            1,
            min(
                remaining - (n_rows - 1 - row_index),
                int(
                    t.normal(
                        mean=float(mean_row),
                        std=400.0,
                        size=(1,),
                        generator=generator,
                    )
                    .clamp(min=1)
                    .item()
                ),
                65536,
            ),
        )
        row_lengths.append(length)
        remaining -= length
    if any(length < 1 or length > 65536 for length in row_lengths):
        raise SystemExit(
            "--synthetic produced an out-of-range row length; choose an "
            "N_ENTRIES between 1 and roughly 65536 * (N_ENTRIES // 2617)"
        )
    fid_rows = [t.randperm(65536, generator=generator)[:length] for length in row_lengths]
    mag_rows = []
    for length in row_lengths:
        mags = t.rand(length, generator=generator) * 12.0 + 0.01
        ties = t.rand(length, generator=generator) < 0.10
        mags[ties] = 1.0
        mag_rows.append(mags.to(t.float16).to(t.float32))
    offsets = t.zeros(len(row_lengths) + 1, dtype=t.int64)
    offsets[1:] = t.tensor(row_lengths, dtype=t.int64).cumsum(0)
    n_rows = len(row_lengths)
    data = {
        TensorKey.OFFSETS: offsets,
        TensorKey.FEATURE_IDS: t.cat(fid_rows).to(t.int64),
        TensorKey.FEATURE_MAGS: t.cat(mag_rows),
        TensorKey.DOC_IDS: t.arange(n_rows, dtype=t.int64) // 8,
        TensorKey.SEGMENT_POSITIONS: (t.arange(n_rows, dtype=t.int64) % 8) * 128,
        "n_docs": (n_rows + 7) // 8,
        "n_segments": n_rows,
        "layer": 22,
        "context_window": 128,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "context_independent_features": "true",
        "ignore_bos_features": "true",
    }
    return save_features(data, temp_dir / "synthetic_shard_00000_features.safetensors.zst")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile the production autointerp source-index scanner.",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N_ENTRIES",
        help=(
            "Generate a deterministic production-shaped synthetic shard of "
            "this many entries into --temp-dir and benchmark it instead of "
            "--shard-path."
        ),
    )
    parser.add_argument("--shard-path", type=Path, default=None)
    parser.add_argument("--n-sae-features", type=int, required=True)
    parser.add_argument("--target-ids-file", type=Path, default=None)
    parser.add_argument("--blacklist-file", type=Path, default=None)
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument("--max-positive-refs", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument(
        "--assert-exact",
        action="store_true",
        help=(
            "On a bounded shard, assert the production scanner matches an "
            "independent brute-force true-reduction oracle."
        ),
    )
    args = parser.parse_args()

    if args.n_sae_features <= 0:
        raise SystemExit("--n-sae-features must be positive")
    if args.target_limit is not None and args.target_limit <= 0:
        raise SystemExit("--target-limit must be positive when set")
    if args.target_offset < 0:
        raise SystemExit("--target-offset must be non-negative")
    if args.target_ids_file is not None and args.blacklist_file is not None:
        raise SystemExit("Use either --target-ids-file or --blacklist-file, not both")
    if args.synthetic is not None and args.temp_dir is None:
        raise SystemExit("--synthetic requires --temp-dir")
    if (args.shard_path is None) == (args.synthetic is None):
        raise SystemExit("Specify exactly one of --shard-path or --synthetic")

    shard_path = (
        _write_synthetic_shard(args.temp_dir, args.synthetic)
        if args.synthetic is not None
        else args.shard_path
    )
    assert shard_path is not None
    entries = _count_entries(shard_path)
    if args.assert_exact and entries > _ASSERT_EXACT_BRUTE_FORCE_MAX_ENTRIES:
        raise SystemExit(
            "--assert-exact requires at most "
            f"{_ASSERT_EXACT_BRUTE_FORCE_MAX_ENTRIES:,} entries; "
            f"the shard contains {entries:,}"
        )

    target_ids = _resolve_target_ids(args)
    start = time.perf_counter()
    temp_before = _dir_size(args.temp_dir)
    progress_events: list[SourceIndexProgressEvent] = []
    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=target_ids,
        n_sae_features=args.n_sae_features,
        max_positive_refs=args.max_positive_refs,
        chunk_size=args.chunk_size,
        temp_dir=args.temp_dir,
        shard_index=0,
        shard_count=1,
        progress_callback=progress_events.append,
    )

    reference_payload: dict[str, object] | None = None
    if args.assert_exact:
        oracle_start = time.perf_counter()
        oracle_rows = _brute_force_topk_rows(
            shard_path,
            target_ids,
            args.n_sae_features,
            args.max_positive_refs,
        )
        oracle_elapsed_s = time.perf_counter() - oracle_start
        reference_payload = {
            "oracle_kind": "brute_force_true_reduction",
            "oracle_elapsed_s": oracle_elapsed_s,
            "oracle_entries_per_second": (
                entries / oracle_elapsed_s if oracle_elapsed_s > 0 else None
            ),
            "oracle_rows": len(oracle_rows),
            "oracle_sha256": _hash_rows(oracle_rows, intruders),
        }
        if rows != oracle_rows:
            raise SystemExit(
                "production source-index rows did not match the brute_force_true_reduction oracle"
            )

    elapsed_s = time.perf_counter() - start
    temp_after = _dir_size(args.temp_dir)
    chunk_events = [
        event for event in progress_events if event.kind == SourceIndexProgressEventKind.CHUNK_DONE
    ]
    last_chunk = chunk_events[-1] if chunk_events else None
    spooled_event = next(
        (
            event
            for event in progress_events
            if event.kind == SourceIndexProgressEventKind.SHARD_SPOOLED
        ),
        None,
    )
    payload = {
        "shard_path": str(shard_path),
        "entries": entries,
        "entries_per_second": entries / elapsed_s if elapsed_s > 0 else None,
        "n_sae_features": args.n_sae_features,
        "target_count": len(target_ids),
        "target_offset": args.target_offset,
        "target_limit": args.target_limit,
        "max_positive_refs": args.max_positive_refs,
        "chunk_size": args.chunk_size,
        "rows": len(rows),
        "positive_rows_per_second": (len(rows) / elapsed_s if elapsed_s > 0 else None),
        "intruders": len(intruders),
        "sha256": _hash_rows(rows, intruders),
        "assert_exact": args.assert_exact,
        "synthetic_entries": args.synthetic,
        "elapsed_s": elapsed_s,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "temp_dir": None if args.temp_dir is None else str(args.temp_dir),
        "temp_dir_bytes_before": temp_before,
        "temp_dir_bytes_after": temp_after,
        "progress_event_count": len(progress_events),
        "chunk_event_count": len(chunk_events),
        "spooled_event": None if spooled_event is None else asdict(spooled_event),
        "last_chunk_event": None if last_chunk is None else asdict(last_chunk),
    }
    if reference_payload is not None:
        payload["reference"] = reference_payload
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
