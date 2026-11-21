"""Round-trip tests for ExtractionStatistics.health_metrics via stats_io.

health_metrics is stored as namespaced ``health.<key>`` metadata entries so it
cannot collide with any reserved key.  A file that lacks any ``health.*`` key
must load with ``health_metrics is None``.
"""

import dataclasses

import torch as t  # noqa: F401  (parity with sibling stats_io tests)

from steering.segmentation import SegmentMode
from steering.stats_io import load_statistics, save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    SegmentFeatureStats,
    StorageStats,
)


def _make_minimal_stats() -> ExtractionStatistics:
    """Minimal valid ExtractionStatistics for a save→load round-trip.

    Reuses the exact construction pattern from test_ignore_bos_stats_io.py.
    """
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=8,
            n_sae_features=16,
            generated_at="2026-05-17T00:00:00+00:00",
        ),
        corpus=CorpusStats(1, 1, 1, 1.0, 1),
        features=SegmentFeatureStats(
            n_segments=1,
            n_features_seen=1,
            mean_active_per_segment=1.0,
            median_active_per_segment=1,
            p90_active_per_segment=1,
            max_active_per_segment=1,
            min_active_per_segment=1,
            n_total_sae_features=16,
            pct_features_active=1.0,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(1, 1.0, 1.0),
        per_feature_stats={},
        extrapolation_10b=Extrapolation(1, 1, 1.0, 1.0),
    )


def test_health_metrics_round_trip(tmp_path):
    stats = dataclasses.replace(
        _make_minimal_stats(),
        health_metrics={
            "median_feature_firing_frequency": "0.0050",
            "n_features_firing_gt_99pct": "21",
        },
    )
    p = save_statistics(stats, tmp_path / "feature_statistics.safetensors")
    loaded = load_statistics(p)
    assert loaded.health_metrics == stats.health_metrics
    # Reserved keys must not be clobbered.
    assert loaded.features.n_segments == stats.features.n_segments


def test_health_metrics_absent_loads_none(tmp_path):
    stats = dataclasses.replace(_make_minimal_stats(), health_metrics=None)
    loaded = load_statistics(save_statistics(stats, tmp_path / "s.safetensors"))
    assert loaded.health_metrics is None


def test_health_metrics_namespace_blocks_reserved_key_collision(tmp_path):
    """Adversarial: health_metrics keys deliberately shadow real serialized
    metadata keys. ``n_segments`` and ``n_features_seen`` are written as bare
    top-level metadata entries (stats_io.py:248-249), so without the ``health.``
    namespace a health metric named ``n_segments`` would overwrite the real
    field. The namespace must isolate them in BOTH directions: the reserved
    fields keep their real values, and the health metrics still round-trip.
    """
    stats = dataclasses.replace(
        _make_minimal_stats(),  # features.n_segments == 1, n_features_seen == 1
        health_metrics={"n_segments": "99", "n_features_seen": "12345"},
    )
    p = save_statistics(stats, tmp_path / "feature_statistics.safetensors")
    loaded = load_statistics(p)
    # Reserved fields untouched by the colliding-named health metrics.
    assert loaded.features.n_segments == 1
    assert loaded.features.n_features_seen == 1
    # Health metrics round-trip intact (not clobbered by the real fields).
    assert loaded.health_metrics == {"n_segments": "99", "n_features_seen": "12345"}
