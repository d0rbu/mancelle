"""Stage-level profiling sidecar for precompute_packed_rows (issue #531).

A dependency-light data holder the script populates while running and writes atomically as
a JSON sidecar when ``profile=true``. Kept out of the main module so it can be unit-tested
in isolation and imported without importing the whole pipeline. Mirrors
``scripts/preprocess/_tok_boundary_profile.py``.

Packing is one continuous, globally-sequential stream (rows span base-shard boundaries), so
per-stage costs are run-aggregate (summed into ``stage_totals``) rather than per-base-shard;
the save side, which IS per output shard, is captured as a ``per_output_shard`` array.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from nanochat.atomic_io import atomic_json_dump


@dataclass
class ShardSaveTiming:
    """Assemble + save timing for one OUTPUT shard (one entry per flushed shard)."""

    shard_idx: int
    n_rows: int
    n_segments: int
    n_features: int
    assemble_seconds: float = 0.0
    save_seconds: float = 0.0
    worker_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    save_bytes: int = 0


@dataclass
class PackedRowsProfile:
    """Whole-run profile: config, the manifest base-hash pre-pass, run-aggregate per-stage
    wall-clock, the recurring base-shard loads, per-output-shard save timings, resume mode,
    and totals."""

    config: dict = field(default_factory=dict)
    manifest_pre_pass: dict = field(
        default_factory=lambda: {
            "seconds": 0.0,
            "bytes_read": 0,
            "cache_hit": False,
            "n_shards_hashed": 0,
        }
    )
    stage_totals: dict = field(
        default_factory=lambda: {
            "parquet": 0.0,
            "lookup": 0.0,
            "tokenize": 0.0,
            "seg_to_raw": 0.0,
            "truncate_assert_append": 0.0,
            "pack": 0.0,
            "csr_build": 0.0,
        }
    )
    row_build_detail: dict = field(
        default_factory=lambda: {
            "row_to_csr_seconds": 0.0,
            "row_provenance_seconds": 0.0,
            "build_segments_seconds": 0.0,
            "segment_boundaries_seconds": 0.0,
            "feature_concat_seconds": 0.0,
            "feature_ids_concat_seconds": 0.0,
            "percentiles_concat_seconds": 0.0,
            "feature_counts_seconds": 0.0,
            "row_tensor_finalize_seconds": 0.0,
        }
    )
    base_shard_loads: dict = field(default_factory=lambda: {"count": 0, "total_seconds": 0.0})
    startup: dict = field(default_factory=lambda: {"feature_index_scan_seconds": 0.0})
    feature_index_loads: list = field(default_factory=list)
    compression: dict = field(
        default_factory=lambda: {
            "backpressure_seconds": 0.0,
            "max_inflight_shards": 0,
        }
    )
    per_output_shard: list = field(default_factory=list)
    resume: dict = field(
        default_factory=lambda: {
            "mode": "none",
            "replay_seconds": 0.0,
            "rows_replayed": 0,
            "checkpoint_seek_seconds": 0.0,
            "checkpoint_write_seconds": 0.0,
        }
    )
    totals: dict = field(
        default_factory=lambda: {
            "wall_seconds": 0.0,
            "total_rows": 0,
            "rows_per_sec": 0.0,
            "n_shards_written": 0,
        }
    )
    # Remote-only additive transport/materialization counters. ``None`` keeps
    # the local sidecar schema byte-for-byte unchanged.
    remote_transport: dict | None = None

    def add_output_shard(self, timing: ShardSaveTiming) -> None:
        """Append one flushed shard's timing. Asserts the record is well-formed."""
        assert timing.shard_idx >= 0, "ShardSaveTiming.shard_idx must be >= 0"
        assert timing.n_rows >= 0, "ShardSaveTiming.n_rows must be >= 0"
        self.per_output_shard.append(asdict(timing))

    def to_dict(self) -> dict:
        result = {
            "config": self.config,
            "manifest_pre_pass": self.manifest_pre_pass,
            "stage_totals": self.stage_totals,
            "row_build_detail": self.row_build_detail,
            "base_shard_loads": self.base_shard_loads,
            "startup": self.startup,
            "feature_index_loads": self.feature_index_loads,
            "compression": self.compression,
            "per_output_shard": self.per_output_shard,
            "resume": self.resume,
            "totals": self.totals,
        }
        if self.remote_transport is not None:
            result["remote_transport"] = self.remote_transport
        return result

    def write(self, path: Path) -> None:
        """Atomically write the sidecar. Asserts the per-shard count matches the recorded
        ``n_shards_written`` (defaulting to the array length when the caller did not set it)."""
        n_written = self.totals.get("n_shards_written", len(self.per_output_shard))
        assert len(self.per_output_shard) == n_written, (
            f"profile inconsistent: per_output_shard={len(self.per_output_shard)} "
            f"!= n_shards_written={n_written}"
        )
        atomic_json_dump(self.to_dict(), path)
