"""Shared builder for tiny *real* ``feature_statistics.safetensors`` files with
controllable per-feature ``max_mag`` / percentiles for extraction/statistics tests
(issue #588).

Uses the production ``save_statistics`` writer so the resulting file is a real,
loadable ``ExtractionStatistics`` — not a hand-mocked stand-in. ``max_mags`` maps
feature_index -> max_mag; features in ``absent`` are omitted from
``per_feature_stats`` entirely (so they load as zero rows / missing). Percentiles
are a monotonic ramp scaled by ``max(max_mag, 0)`` (finite), so the percentile
table is always buildable even when ``max_mag`` itself is 0 or NaN.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
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


def _per_feature_stat(max_mag: float) -> PerFeatureStat:
    pos = max_mag if (math.isfinite(max_mag) and max_mag > 0.0) else 0.0
    return PerFeatureStat(
        count=100,
        pct_segments_active=0.1,
        mean_mag=0.5 * pos,
        mean_mag_incl_zeros=0.05 * pos,
        std_mag=0.25 * pos,
        min_mag=0.0,
        max_mag=max_mag,
        median_mag=0.5 * pos,
        percentiles={f"p{k}": (k / 100.0) * pos for k in range(1, 100)},
        histogram=None,
    )


def write_steering_stats(
    path: str | Path,
    *,
    n_sae_features: int,
    max_mags: Mapping[int, float] | None = None,
    absent: Iterable[int] = (),
) -> Path:
    """Write a real ``feature_statistics.safetensors`` to ``path`` and return it."""
    max_mags = dict(max_mags or {})
    absent_set = {int(i) for i in absent}
    per_feature = {
        str(i): _per_feature_stat(float(max_mags.get(i, 1.0)))
        for i in range(n_sae_features)
        if i not in absent_set
    }
    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=1024,
            layer=8,
            n_sae_features=n_sae_features,
            generated_at="2026-06-30",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=1000,
            mean_tokens_per_doc=10.0,
            median_tokens_per_doc=10,
        ),
        features=SegmentFeatureStats(
            n_segments=100,
            n_features_seen=len(per_feature),
            mean_active_per_segment=5.0,
            median_active_per_segment=5,
            p90_active_per_segment=10,
            max_active_per_segment=15,
            min_active_per_segment=0,
            n_total_sae_features=n_sae_features,
            pct_features_active=0.5,
            count_percentiles={"p50": 50},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={"p50": 5.0},
        ),
        storage=StorageStats(
            total_bytes=1000, mean_bytes_per_doc=10.0, mean_bytes_per_token=1.0,
        ),
        per_feature_stats=per_feature,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1000,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
    )
    return save_statistics(stats, path)
