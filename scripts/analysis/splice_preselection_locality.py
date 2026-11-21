"""Reconstruct the splice-preselection source-shard access sequence and simulate
LRU load counts per cache capacity (issue #740).

Reads ONLY the packed manifest + provenance shards — never the multi-GB feature
shards — so it runs in seconds on the full corpus. Replicates the exact group
ordering of select_splice_counterfactual_features_for_shard (sorted by first
packed-segment appearance), which determines FeatureIndex load locality.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import OrderedDict
from pathlib import Path

import torch as t

from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    provenance_dir,
)
from steering.overlay_io import load_overlay_shard, read_manifest

_DOC_KEY_STRIDE = 1_000_000


def _group_stem_order(base_idx: t.Tensor, doc_idx: t.Tensor) -> list[int]:
    """base_shard_idx per (base, doc) group, ordered by first packed-segment index.

    Reproduces production's ``sorted(groups.items(), key=lambda item: min(item[1]))``
    exactly. Groups are keyed by (base, doc); encoding that pair as
    ``base * _DOC_KEY_STRIDE + doc`` keeps the fast 1-D ``torch.unique`` path. The
    guard makes an out-of-range doc index a LOUD error rather than a silent group
    collision — this project does not tolerate silent-wrong-answer (issues #588/#602).
    """
    b = base_idx.to(t.int64)
    d = doc_idx.to(t.int64)
    if d.numel() and int(d.max()) >= _DOC_KEY_STRIDE:
        raise ValueError(
            f"doc_idx {int(d.max())} >= key stride {_DOC_KEY_STRIDE}: the packed "
            f"(base, doc) key would collide two distinct groups. Raise "
            f"_DOC_KEY_STRIDE above the maximum per-shard document count."
        )
    key = b * _DOC_KEY_STRIDE + d
    uniq, inv = t.unique(key, return_inverse=True)
    first = t.full((uniq.numel(),), t.iinfo(t.int64).max, dtype=t.int64)
    first.scatter_reduce_(0, inv, t.arange(key.numel(), dtype=t.int64), reduce="amin")
    order = first.argsort(stable=True)
    return (uniq[order] // _DOC_KEY_STRIDE).tolist()


def simulate_lru_loads(sequence: list[int], capacity: int) -> int:
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    cache: OrderedDict[int, int] = OrderedDict()
    loads = 0
    for stem in sequence:
        if stem in cache:
            cache.move_to_end(stem)
        else:
            loads += 1
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[stem] = 1
    return loads


def analyze(packed_row_dir: Path, capacities: list[int]) -> dict:
    prov_dir = provenance_dir(packed_row_dir)
    per_shard = []
    full_seq: list[int] = []
    for pp in sorted(prov_dir.glob("*.safetensors.zst")):
        prov = load_overlay_shard(pp)
        order = _group_stem_order(
            prov[PROVENANCE_BASE_SHARD_IDX_KEY], prov[PROVENANCE_DOC_IDX_KEY]
        )
        distinct = len(set(order))
        # Keep this adjacent-pair comparison consistent with the Task-1.1
        # worst_case_stem_alternations helper.
        lru1 = (
            1 + sum(1 for a, c in itertools.pairwise(order) if a != c)
            if order
            else 0
        )
        per_shard.append(
            {
                "shard": pp.name.split(".")[0],
                "groups": len(order),
                "distinct_stems": distinct,
                "lru1_loads_isolated": lru1,
            }
        )
        full_seq.extend(order)
    # per_shard's dict literal mixes str ("shard") and int values, so plain indexing
    # infers as str | int; distinct_stems is always a genuine int (len(...)) at
    # runtime, and the explicit cast tells the type checker so without changing
    # any computed value.
    dis: list[int] = sorted(int(s["distinct_stems"]) for s in per_shard) or [0]

    def q(p: float) -> int:
        return dis[min(len(dis) - 1, int(p * len(dis)))]

    ideal = len(set(full_seq))
    return {
        "n_packed_shards": len(per_shard),
        "unique_source_stems": ideal,
        "distinct_stems": {"p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99), "max": dis[-1]},
        "loads_by_capacity": {c: simulate_lru_loads(full_seq, c) for c in capacities},
        "per_shard": per_shard,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("packed_row_dir", type=Path)
    ap.add_argument("--capacities", default="1,2,4,5,8,16", help="comma-separated ints")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    if read_manifest(args.packed_row_dir) is None:
        raise SystemExit(f"no packed-row manifest under {args.packed_row_dir}")
    caps = [int(c) for c in args.capacities.split(",")]
    result = analyze(args.packed_row_dir, caps)
    ideal = result["unique_source_stems"]
    print(f"packed shards: {result['n_packed_shards']}  unique source stems: {ideal}")
    print(f"distinct-stems per shard: {result['distinct_stems']}")
    for cap, loads in result["loads_by_capacity"].items():
        ratio = loads / ideal if ideal else 0.0
        print(f"  capacity {cap:3d}: {loads:8d} loads  ({ratio:5.1f}x ideal)")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
