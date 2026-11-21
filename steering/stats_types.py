"""Dataclass types for feature extraction statistics.

Shared between compute_statistics (producer) and stats_io (serialization).
"""

from __future__ import annotations

from dataclasses import dataclass

from steering.sae_resolver import L0Tier
from steering.segmentation import SegmentMode


@dataclass(frozen=True)
class PerFeatureStat:
    count: int
    pct_segments_active: float
    mean_mag: float
    mean_mag_incl_zeros: float
    std_mag: float
    min_mag: float
    max_mag: float
    median_mag: float
    percentiles: dict[str, float]
    histogram: dict[str, object] | None = None


@dataclass(frozen=True)
class MetadataStats:
    pooling_mode: str
    segment_mode: SegmentMode
    segment_size: int
    context_window: int
    layer: int
    n_sae_features: int
    generated_at: str
    activation_percentile: float = 0.0


@dataclass(frozen=True)
class CorpusStats:
    n_shards_processed: int
    n_docs: int
    n_tokens_gpt2: int
    mean_tokens_per_doc: float
    median_tokens_per_doc: int


@dataclass(frozen=True)
class SegmentFeatureStats:
    n_segments: int
    n_features_seen: int
    mean_active_per_segment: float
    median_active_per_segment: int
    p90_active_per_segment: int
    max_active_per_segment: int
    min_active_per_segment: int
    n_total_sae_features: int
    pct_features_active: float
    count_percentiles: dict[str, float | int]
    magnitude_stats: dict[str, object]
    active_per_segment_histogram: dict[str, object] | None
    active_per_segment_percentiles: dict[str, float]
    mag_histogram_bin_edges: list[float] | None = None
    segment_length_histogram: dict[str, object] | None = None
    segment_length_percentiles: dict[str, float] | None = None
    features_per_token_histogram: dict[str, object] | None = None
    features_per_token_percentiles: dict[str, float] | None = None


@dataclass(frozen=True)
class StorageStats:
    total_bytes: int
    mean_bytes_per_doc: float
    mean_bytes_per_token: float


@dataclass(frozen=True)
class Extrapolation:
    target_tokens: int
    estimated_docs: int
    estimated_storage_gb: float
    estimated_wall_time_hours: float


@dataclass(frozen=True)
class ExtractionStatistics:
    metadata: MetadataStats
    corpus: CorpusStats
    features: SegmentFeatureStats
    storage: StorageStats
    per_feature_stats: dict[str, PerFeatureStat]
    extrapolation_10b: Extrapolation
    per_feature_thresholds: dict[str, float] | None = None
    # gemma-scope-2 L0 tier (e.g. L0Tier.medium), or None for v1 releases.
    # Serialized to its string ``.value`` in the safetensors metadata so the
    # stats file mirrors the v1/v2 split carried in shard metadata (see
    # steering.feature_io.MetadataKey.L0_TIER); parsed back via L0Tier(...)
    # on load.
    l0_tier: L0Tier | None = None
    # ignore_bos_features the stats were computed from. Always serialized as
    # "true"/"false" in fresh stats files; a stats file written before this
    # field existed simply lacks the key and loads as ``False`` (a pre-feature
    # corpus was extracted with ignore_bos off, so ``False`` is the correct,
    # not merely convenient, default). Plain bool — no nullable/"unknown"
    # state (no consumer needs to distinguish absent from explicit false).
    ignore_bos_features: bool = False
    # Additive metadata entries written verbatim into the safetensors metadata
    # dict. Absent on legacy files. Used for health / diagnostic metrics that
    # don't warrant a typed dataclass field (e.g. issue #549 degeneracy metrics).
    health_metrics: dict[str, str] | None = None
