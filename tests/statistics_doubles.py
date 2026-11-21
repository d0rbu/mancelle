"""One ``ExtractionStatistics``-shaped double for the segment-mode guards.

Every fixed-geometry consumer reads its segment-mode witness off
``statistics.metadata.segment_mode``, so the double must carry a real
:class:`~steering.stats_types.MetadataStats` rather than an ad-hoc namespace
that would happily answer any attribute.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Final

from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode
from steering.stats_types import MetadataStats

_METADATA_DEFAULTS: Final[dict[str, Any]] = {
    "pooling_mode": PoolingMode.max_pool.value,
    "segment_size": 128,
    "context_window": 128,
    "layer": 8,
    "n_sae_features": 16,
    "generated_at": "2026-01-01T00:00:00Z",
}


def fixed_statistics_double(
    *,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    **overrides: Any,
) -> SimpleNamespace:
    """Return a statistics double whose metadata witnesses ``segment_mode``.

    ``overrides`` sets ``per_feature_stats`` (default ``{}``) and any
    :class:`MetadataStats` field.
    """

    per_feature_stats = overrides.pop("per_feature_stats", {})
    return SimpleNamespace(
        metadata=MetadataStats(
            segment_mode=segment_mode,
            **{**_METADATA_DEFAULTS, **overrides},
        ),
        per_feature_stats=per_feature_stats,
    )
