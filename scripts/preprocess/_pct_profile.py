"""Profiling sidecar for precompute_percentiles (issue #1269).

The producer populates these small dataclasses while processing shards and writes the
result atomically only when ``profile=true``. Keeping the schema separate makes it
available to the percentile benchmark without importing the producer internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from nanochat.atomic_io import atomic_json_dump


@dataclass
class PercentileShardTiming:
    """Per-shard wall-clock and byte counters."""

    stem: str
    n_values: int
    load_seconds: float = 0.0
    load_bytes: int = 0
    compute_seconds: float = 0.0
    save_seconds: float = 0.0
    overlay_bytes: int = 0


@dataclass
class PercentileProfile:
    """Whole-run percentile profile and processed/skipped totals."""

    config: dict = field(default_factory=dict)
    per_shard: list = field(default_factory=list)
    totals: dict = field(
        default_factory=lambda: {
            "wall_seconds": 0.0,
            "n_processed": 0,
            "n_skipped": 0,
        }
    )

    def add_shard(self, timing: PercentileShardTiming) -> None:
        """Append one processed shard's timing record."""
        assert timing.stem, "PercentileShardTiming.stem must be non-empty"
        assert timing.n_values >= 0, "PercentileShardTiming.n_values must be >= 0"
        self.per_shard.append(asdict(timing))

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "per_shard": self.per_shard,
            "totals": self.totals,
        }

    def write(self, path: Path) -> None:
        """Atomically write the sidecar after checking processed-count consistency."""
        assert len(self.per_shard) == self.totals["n_processed"], (
            f"profile inconsistent: per_shard={len(self.per_shard)} != "
            f"n_processed={self.totals['n_processed']}"
        )
        atomic_json_dump(self.to_dict(), path)
