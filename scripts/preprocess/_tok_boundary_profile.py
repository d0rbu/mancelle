"""Stage-level profiling sidecar for precompute_token_boundaries (issue #529).

A small, dependency-light data holder that the script populates while running and writes
atomically as a JSON sidecar when ``profile=true``. Kept out of the main script module so it
can be unit-tested in isolation and imported without importing the whole pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from nanochat.atomic_io import atomic_json_dump


@dataclass
class ShardTiming:
    """Per-shard wall-clock + byte counters (one entry per processed shard)."""

    stem: str
    n_segments: int
    load_features_seconds: float = 0.0
    load_bytes: int = 0
    parquet_load_seconds: float = 0.0
    parquet_bytes: int = 0
    batch_encode_seconds: float = 0.0
    tokenize_seconds: float = 0.0
    offset_paths: dict = field(default_factory=dict)
    worker_compute_seconds: float = 0.0
    map_wait_seconds: float = 0.0
    scatter_seconds: float = 0.0
    overlay_save_seconds: float = 0.0
    overlay_bytes: int = 0


@dataclass
class TokenBoundaryProfile:
    """Whole-run profile: config, the manifest pre-pass, worker pool, and per-shard timings."""

    config: dict = field(default_factory=dict)
    manifest_pre_pass: dict = field(
        default_factory=lambda: {
            "seconds": 0.0,
            "bytes_read": 0,
            "cache_hit": False,
            "n_shards_hashed": 0,
        }
    )
    worker_pool: dict = field(
        default_factory=lambda: {"spawn_count": 0, "total_init_seconds": 0.0}
    )
    remote_transport: dict = field(default_factory=dict)
    per_shard: list = field(default_factory=list)
    totals: dict = field(
        default_factory=lambda: {"wall_seconds": 0.0, "n_processed": 0, "n_skipped": 0}
    )

    def add_shard(self, timing: ShardTiming) -> None:
        """Append one processed shard's timing. Asserts the record is well-formed."""
        assert timing.stem, "ShardTiming.stem must be non-empty"
        assert timing.n_segments >= 0, "ShardTiming.n_segments must be >= 0"
        self.per_shard.append(asdict(timing))

    def to_dict(self) -> dict:
        payload = {
            "config": self.config,
            "manifest_pre_pass": self.manifest_pre_pass,
            "worker_pool": self.worker_pool,
            "per_shard": self.per_shard,
            "totals": self.totals,
        }
        if self.remote_transport:
            payload["remote_transport"] = self.remote_transport
        return payload

    def write(self, path: Path) -> None:
        """Atomically write the sidecar. Asserts per-shard count matches n_processed."""
        assert len(self.per_shard) == self.totals["n_processed"], (
            f"profile inconsistent: per_shard={len(self.per_shard)} != "
            f"n_processed={self.totals['n_processed']}"
        )
        atomic_json_dump(self.to_dict(), path)
