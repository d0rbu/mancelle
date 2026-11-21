"""Backfill max_n_seg and raw max_seg_size into packed-row manifests.

Note: max_F is NOT backfilled here — base shards store ALL extraction features
(~8k per segment). The training-time max_F is determined by scanning selection
shards at loader init time.

``max_seg_size`` remains the exact observed segment length. Cross-attention
applies FP8 allocation alignment internally; this script must not normalize the
manifest value because it is also a stale-data and resume-identity bound.

Usage:
    uv run python -m scripts.preprocess.backfill_packed_row_shapes path/to/packed_rows
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from errors import PackedRowInvariantError
from nanochat.atomic_io import atomic_write_text
from scripts.preprocess.precompute_packed_rows import _reduce_max_dims_over_shards
from steering.overlay_io import OverlayKind, glob_overlay_shards, read_manifest

_PACKED_ROW_IDENTITY_FIELDS = (
    "type",
    "context_len",
    "buffer_size",
    "rows_per_shard",
    "pct_checksum",
    "tok_checksum",
)


def backfill_packed_row_shapes(packed_row_dir: Path) -> dict:
    """Scan all shards and write raw shape metadata to the manifest.

    Returns the computed shape dict.
    """
    packed_row_dir = Path(packed_row_dir)
    shard_paths = glob_overlay_shards(packed_row_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No packed row shards in {packed_row_dir}")

    manifest_path = packed_row_dir / "manifest.json"
    manifest = read_manifest(packed_row_dir)
    if manifest is None:
        raise PackedRowInvariantError(
            f"Cannot backfill {packed_row_dir}: manifest.json is missing or invalid."
        )
    checksum = manifest.get("checksum")
    if not isinstance(checksum, str) or not checksum:
        raise PackedRowInvariantError(
            f"Cannot backfill {packed_row_dir}: manifest checksum is missing or empty."
        )
    kind = manifest.get("kind")
    if kind not in {OverlayKind.PACKED_ROW.value, OverlayKind.PERCENTILE.value}:
        raise PackedRowInvariantError(
            f"Cannot backfill {packed_row_dir}: manifest kind {kind!r} is not "
            f"{OverlayKind.PACKED_ROW.value!r} or historical "
            f"{OverlayKind.PERCENTILE.value!r}."
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise PackedRowInvariantError(
            f"Cannot backfill {packed_row_dir}: manifest lacks a metadata dict."
        )
    missing = [field for field in _PACKED_ROW_IDENTITY_FIELDS if field not in metadata]
    if missing:
        raise PackedRowInvariantError(
            f"Cannot backfill {packed_row_dir}: manifest metadata lacks packed-row "
            f"identity fields {missing!r}."
        )
    logger.info(f"Scanning {len(shard_paths)} shards for max dims...")
    global_max_n_seg, global_max_seg_size = _reduce_max_dims_over_shards(shard_paths)

    shapes = {
        "max_n_seg": global_max_n_seg,
        "max_seg_size": global_max_seg_size,
    }
    logger.info(f"Global: {shapes}")

    metadata.update(shapes)
    atomic_write_text(json.dumps(manifest, indent=2) + "\n", manifest_path)
    logger.info(f"Updated {manifest_path}")
    return shapes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python -m scripts.preprocess.backfill_packed_row_shapes <packed_row_dir>"
        )
        sys.exit(1)
    backfill_packed_row_shapes(Path(sys.argv[1]))
