"""Diff two feature-shard dirs by DECOMPRESSED tensors (the byte-identity oracle).

Box usage:
  uv run python -m steering.extract_features --config-name <cfg> output_dir=/A staging.output_depth=0
  uv run python -m steering.extract_features --config-name <cfg> output_dir=/B   # staged
  uv run python -m scripts.benchmark.verify_staging_identity /A /B
Exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse

import torch as t

from steering.feature_io import glob_features, load_features


def diff_dirs(a: str, b: str) -> list[str]:
    ra = {p.name: p for p in glob_features(a)}
    rb = {p.name: p for p in glob_features(b)}
    errs: list[str] = []
    if set(ra) != set(rb):
        errs.append(f"filename set differs: {sorted(set(ra) ^ set(rb))}")
    for name in sorted(set(ra) & set(rb)):
        da, db = load_features(str(ra[name])), load_features(str(rb[name]))
        off = da["offsets"]
        if not t.equal(off, db["offsets"]):
            errs.append(f"{name}: offsets")
            continue
        for k in ("doc_ids", "segment_indices", "segment_positions",
                  "segment_char_starts", "segment_char_ends", "doc_token_counts"):
            if not t.equal(da[k], db[k]):
                errs.append(f"{name}: {k}")
        for s in range(off.numel() - 1):
            lo, hi = int(off[s]), int(off[s + 1])
            # Compare in STORED order (feature_io co-sorts feature_ids ascending +
            # co-sorts magnitudes), NOT independent per-tensor sorts — sorting ids
            # and mags separately would discard the id<->mag pairing and certify
            # within-segment-mispaired data as identical.
            if not t.equal(da["feature_ids"][lo:hi], db["feature_ids"][lo:hi]):
                errs.append(f"{name}: feature_ids seg {s}")
            if not t.equal(da["feature_mags"][lo:hi], db["feature_mags"][lo:hi]):
                errs.append(f"{name}: feature_mags seg {s}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    args = ap.parse_args()
    errs = diff_dirs(args.dir_a, args.dir_b)
    if errs:
        print("MISMATCH:")
        for e in errs:
            print(" ", e)
        raise SystemExit(1)
    print("OK: decompressed-tensor identical")


if __name__ == "__main__":
    main()
