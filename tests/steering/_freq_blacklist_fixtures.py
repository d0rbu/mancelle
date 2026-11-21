"""Deterministic synthetic ``feature_statistics.safetensors`` fixtures.

The freq-blacklist generator reads the precomputed statistics artifact, so
tests construct a real ``ExtractionStatistics`` (controllable per-feature
``count`` and ``n_segments``) and round-trip it through
``steering.stats_io.save_statistics`` — exactly the on-disk format the
generator consumes.

Serde caveats honored (verified against steering/stats_io.py):
- ``per_feature_stats`` MUST be in count-descending order (save preserves
  dict order; ``compute_statistics`` sorts that way).
- Every ``PerFeatureStat.percentiles`` must share one key set — we use ``{}``.
- ``count`` (int64) and ``features.n_segments`` round-trip EXACTLY.
- features with ``count == 0`` are never present (compute_statistics only
  records active features) — callers pass only active ids.
"""
from __future__ import annotations

from pathlib import Path

from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)


def _pfs(count: int) -> PerFeatureStat:
    return PerFeatureStat(
        count=count,
        pct_segments_active=0.0,  # generator never reads this (uses count)
        mean_mag=0.0,
        mean_mag_incl_zeros=0.0,
        std_mag=0.0,
        min_mag=0.0,
        max_mag=0.0,
        median_mag=0.0,
        percentiles={},  # identical (empty) key set for ALL features
        histogram=None,
    )


def make_stats(
    *,
    n_segments: int,
    counts: dict[int, int],
    n_sae_features: int,
    generated_at: str = "2026-05-19T00:00:00+00:00",
) -> ExtractionStatistics:
    """Build an ``ExtractionStatistics`` whose ``per_feature_stats`` has
    exactly ``{fid: count}`` (only active features; all counts must be > 0),
    in count-descending order, with ``features.n_segments == n_segments``."""
    if any(c <= 0 for c in counts.values()):
        raise ValueError("synthetic stats only hold active features (count>0)")
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    per_feature_stats = {str(fid): _pfs(c) for fid, c in ordered}
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=8,
            n_sae_features=n_sae_features,
            generated_at=generated_at,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=10,
            n_tokens_gpt2=1000,
            mean_tokens_per_doc=100.0,
            median_tokens_per_doc=100,
        ),
        features=SegmentFeatureStats(
            n_segments=n_segments,
            n_features_seen=len(counts),
            mean_active_per_segment=1.0,
            median_active_per_segment=1,
            p90_active_per_segment=1,
            max_active_per_segment=1,
            min_active_per_segment=1,
            n_total_sae_features=n_sae_features,
            pct_features_active=0.0,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=12345,
            mean_bytes_per_doc=1234.5,
            mean_bytes_per_token=12.345,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1000,
            estimated_storage_gb=1.5,
            estimated_wall_time_hours=2.0,
        ),
    )


def write_stats(
    directory: Path,
    *,
    n_segments: int,
    counts: dict[int, int],
    n_sae_features: int,
    generated_at: str = "2026-05-19T00:00:00+00:00",
) -> Path:
    """Write ``feature_statistics.safetensors`` into ``directory``; return it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return save_statistics(
        make_stats(
            n_segments=n_segments,
            counts=counts,
            n_sae_features=n_sae_features,
            generated_at=generated_at,
        ),
        directory / "feature_statistics.safetensors",
    )
