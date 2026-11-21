"""DataLoader for the completion TUI.

Loads document text from parquet files and pre-extracted features from
FeatureIndex, targeting the validation shard (last shard alphabetically).
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import torch as t
from loguru import logger

from errors import FeatureClustersError
from scripts.completion_tui.state import (
    DocumentData,
    FeatureCondition,
    Segment,
    SegmentClusterEvidence,
    display_magnitude_range,
    internal_to_display,
)
from steering.conditional.discretization import (
    discretize_activations,
    percentile_to_bucket,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.percentile_table import uint8_to_percentile
from steering.conditional.sampling import (
    FeatureSelection,
    NegativeSamplingSelection,
    SamplingSelection,
    ThresholdSelection,
    sample_features,
    sample_features_batched,
    sample_inactive_features,
    sample_inactive_features_batched,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    select_from_packed_percentiles,
)


@dataclass
class TUIFeatureSamplingConfig:
    """Bundled sampling config for the TUI feature randomize/resample operations."""

    boundaries: t.Tensor
    n_sae_features: int
    feature_selection: FeatureSelection | None = None
    negative_selection: NegativeSamplingSelection | None = None
    percentile_table: t.Tensor | None = None
    precomputed_thresholds: t.Tensor | None = None
    feature_mask: FeatureMask | None = field(default=None, kw_only=True)
    consolidation: FeatureClusterConsolidation | None = field(default=None, kw_only=True)
    effective_buckets: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ClusteredSegmentSample:
    """Cluster-aware TUI rows plus per-member percentile evidence."""

    features: list[FeatureCondition]
    evidence: SegmentClusterEvidence


_PERCENTILE_ONLY_THRESHOLD_TYPES: frozenset[ThresholdType] = frozenset(
    {
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    }
)


def _clustered_threshold_select(
    fids: t.Tensor,
    pcts: t.Tensor,
    selection: ThresholdSelection,
) -> t.Tensor:
    """Delegate to the shared packed-percentile threshold selection.

    ``steering.conditional.thresholding.select_from_packed_percentiles`` is the
    single implementation both the training preselection producer and this live
    path consume; the TUI only guards the threshold-type domain first so an
    unsupported raw-magnitude type fails with a TUI-actionable message instead
    of the shared function's generic one.
    """
    if selection.threshold_type not in _PERCENTILE_ONLY_THRESHOLD_TYPES:
        supported = sorted(item.value for item in _PERCENTILE_ONLY_THRESHOLD_TYPES)
        raise ValueError(
            f"Threshold type {selection.threshold_type!r} requires raw feature "
            "magnitudes, but clustered completion TUI selection only has "
            f"percentiles. Supported types: {supported}."
        )
    return select_from_packed_percentiles(
        fids,
        pcts,
        selection.threshold_type,
        selection.threshold,
    )


def _composite_representatives(
    consolidation: FeatureClusterConsolidation,
) -> frozenset[int]:
    representatives: set[int] = set()
    for cluster in consolidation.resolved.definition.clusters:
        allowed = [member for member in cluster if bool(consolidation.allowed_universe[member])]
        if len(allowed) >= 2:
            representatives.add(allowed[0])
    return frozenset(representatives)


def clustered_sample_segment_features(
    sf: SegmentFeatures,
    cfg: TUIFeatureSamplingConfig,
    num_buckets: int,
    has_negatives: bool,
    rng: t.Generator | None = None,
) -> ClusteredSegmentSample:
    """Mirror producer-exact mask, cluster, select, bucket, negative semantics."""
    if cfg.consolidation is None or cfg.effective_buckets is None:
        raise ValueError(
            "clustered_sample_segment_features requires consolidation + effective_buckets"
        )
    if sf.feature_percentiles is None:
        raise ValueError(
            "clustered completion TUI requires the percentile overlay; "
            "this segment has no precomputed percentiles"
        )
    if rng is None:
        rng = t.Generator()
        rng.manual_seed(random.randint(0, 2**32 - 1))

    fids = sf.feature_ids.to(t.int64)
    percentiles = sf.feature_percentiles.float() / 255.0
    if cfg.feature_mask is not None and fids.numel() > 0:
        fids, percentiles = cfg.feature_mask.apply_to_positive(fids, percentiles)
    if fids.numel() == 0:
        return ClusteredSegmentSample(
            features=[],
            evidence=SegmentClusterEvidence(fingerprint=frozenset(), evidence={}),
        )

    representatives, consolidated_percentiles = cfg.consolidation.consolidate_segment(
        fids, percentiles
    )
    if representatives.numel() == 0:
        raise FeatureClustersError(
            "internal error: cluster consolidation emptied a segment that "
            "still had mask-surviving features"
        )

    composite_reps = _composite_representatives(cfg.consolidation)
    evidence: dict[int, dict[int, float]] = {}
    for feature_id, percentile in zip(fids.tolist(), percentiles.tolist(), strict=True):
        representative = int(cfg.consolidation.rep_of[feature_id].item())
        if representative in composite_reps:
            evidence.setdefault(representative, {})[int(feature_id)] = float(percentile)

    if isinstance(cfg.feature_selection, SamplingSelection):
        if cfg.percentile_table is None:
            raise ValueError("clustered SamplingSelection requires percentile_table")
        sampled = sample_features_batched(
            all_feature_ids=representatives,
            all_feature_mags=t.zeros_like(representatives, dtype=t.float32),
            segment_sizes=[int(representatives.numel())],
            percentile_table=cfg.percentile_table,
            config=cfg.feature_selection,
            rng=rng,
            precomputed_percentiles=consolidated_percentiles,
        )
        selected = sampled[0][0] if sampled else t.tensor([], dtype=t.int64)
    elif isinstance(cfg.feature_selection, ThresholdSelection):
        selected = _clustered_threshold_select(
            representatives,
            consolidated_percentiles,
            cfg.feature_selection,
        )
    else:
        raise ValueError("clustered mode requires a feature_selection")

    if cfg.feature_mask is not None:
        cfg.feature_mask.assert_no_masked(selected)
    cfg.consolidation.assert_representatives_only(
        selected, surface="completion_tui clustered selection"
    )
    selected_set = set(selected.tolist())
    buckets = percentile_to_bucket(consolidated_percentiles, cfg.effective_buckets)
    features = [
        FeatureCondition(
            feature_id=int(representative),
            magnitude=internal_to_display(int(bucket), num_buckets, has_negatives),
            enabled=int(representative) in selected_set,
        )
        for representative, bucket in zip(representatives.tolist(), buckets.tolist(), strict=True)
    ]

    if cfg.negative_selection is not None and has_negatives and cfg.n_sae_features > 0:
        negative_samples = sample_inactive_features_batched(
            [representatives],
            cfg.n_sae_features,
            cfg.negative_selection,
            rng,
            feature_mask=cfg.consolidation.effective_negative_mask(),
        )
        negative_ids = negative_samples[0] if negative_samples else t.tensor([], dtype=t.int64)
        if cfg.feature_mask is not None:
            cfg.feature_mask.assert_no_masked(negative_ids)
        cfg.consolidation.assert_representatives_only(
            negative_ids, surface="completion_tui clustered negatives"
        )
        negative_magnitude = display_magnitude_range(num_buckets, has_negatives)[0]
        features.extend(
            FeatureCondition(feature_id=int(feature_id), magnitude=negative_magnitude)
            for feature_id in negative_ids.tolist()
        )

    fingerprint = frozenset(
        feature.feature_id for feature in features if feature.feature_id in composite_reps
    )
    return ClusteredSegmentSample(
        features=features,
        evidence=SegmentClusterEvidence(
            fingerprint=fingerprint,
            evidence=evidence,
        ),
    )


def sample_segment_features(
    sf: SegmentFeatures,
    cfg: TUIFeatureSamplingConfig,
    num_buckets: int,
    has_negatives: bool,
    *,
    feature_mask: FeatureMask | None = None,
) -> list[FeatureCondition]:
    """Convert raw SegmentFeatures to FeatureConditions with sampling applied.

    Active features are selectively *enabled* according to cfg.feature_selection.
    Unselected active features are included but disabled.
    Negative features are sampled from the inactive pool and added as enabled
    with the minimum display magnitude (−1 when has_negatives is True).
    """
    fids = sf.feature_ids  # int32, raw active feature IDs
    fmags = sf.feature_mags  # float32, raw activation magnitudes
    feature_percentiles = sf.feature_percentiles
    if feature_mask is not None and fids.numel() > 0:
        if feature_percentiles is None:
            fids, fmags = feature_mask.apply_to_positive(fids, fmags)
        else:
            fids, fmags, feature_percentiles = feature_mask.apply_to_positive(
                fids, fmags, feature_percentiles
            )

    # Determine which active features survive selection.
    if cfg.feature_selection is not None and fids.numel() > 0:
        rng = t.Generator()
        rng.manual_seed(random.randint(0, 2**32 - 1))

        if isinstance(cfg.feature_selection, SamplingSelection):
            assert cfg.percentile_table is not None, (
                "percentile_table required for SamplingSelection"
            )
            assert cfg.percentile_table is not None, (
                "percentile_table required for SamplingSelection"
            )
            override_weights: t.Tensor | None = None
            if feature_percentiles is not None:
                override_weights = uint8_to_percentile(feature_percentiles)
            sel_fids, _ = sample_features(
                fids.to(t.int64),
                fmags,
                cfg.percentile_table,
                cfg.feature_selection,
                rng,
                override_weights=override_weights,
            )
        else:  # ThresholdSelection
            threshold_cfg = ThresholdConfig(
                cfg.feature_selection.threshold_type,
                cfg.feature_selection.threshold,
            )
            sel_fids, _ = apply_threshold(
                threshold_cfg,
                fids.to(t.int64),
                fmags,
                cfg.precomputed_thresholds,
            )
        selected_set: set[int] = set(sel_fids.tolist())
    else:
        selected_set = set(fids.tolist()) if fids.numel() > 0 else set()

    # Build FeatureConditions for all active features (enabled if selected).
    features: list[FeatureCondition] = []
    if fids.numel() > 0:
        buckets = discretize_activations(fids, fmags, cfg.boundaries)
        for fid_val, bucket_val in zip(fids.tolist(), buckets.tolist(), strict=True):
            mag = internal_to_display(int(bucket_val), num_buckets, has_negatives)
            enabled = int(fid_val) in selected_set
            features.append(
                FeatureCondition(feature_id=int(fid_val), magnitude=mag, enabled=enabled)
            )

    # Sample negative (inactive) features and append as enabled.
    if cfg.negative_selection is not None and has_negatives and cfg.n_sae_features > 0:
        neg_rng = t.Generator()
        neg_rng.manual_seed(random.randint(0, 2**32 - 1))
        neg_fids = sample_inactive_features(
            fids.to(t.int64) if fids.numel() > 0 else t.tensor([], dtype=t.int64),
            cfg.n_sae_features,
            cfg.negative_selection,
            neg_rng,
            feature_mask=feature_mask,
        )
        neg_mag = display_magnitude_range(num_buckets, has_negatives)[0]  # −1 when has_negatives
        for nfid in neg_fids.tolist():
            features.append(FeatureCondition(feature_id=int(nfid), magnitude=neg_mag, enabled=True))

    return features


def segment_features_to_segment(
    sf: SegmentFeatures,
    color_idx: int,
    boundaries: t.Tensor,
    num_buckets: int,
    has_negatives: bool,
    is_generation_target: bool = False,
    target_token_count: int | None = None,
    threshold_cfg: ThresholdConfig | None = None,
    precomputed_thresholds: t.Tensor | None = None,
    *,
    feature_mask: FeatureMask | None = None,
) -> Segment:
    """Convert a FeatureIndex SegmentFeatures to a TUI Segment.

    Optionally filters features using the same threshold strategy as training
    (e.g., per-feature percentile + segment top-k), then discretizes raw
    float32 magnitudes to bucket indices and applies the negatives offset (+1)
    when has_negatives is True.
    """
    expected_n_boundaries = num_buckets - 2 if has_negatives else num_buckets - 1
    assert boundaries.shape[1] == expected_n_boundaries, (
        f"boundaries shape {boundaries.shape} inconsistent with num_buckets={num_buckets}, has_negatives={has_negatives}"
    )

    if sf.feature_ids.numel() == 0:
        return Segment(
            segment_idx=sf.segment_idx,
            char_start=sf.char_start,
            char_end=sf.char_end,
            features=[],
            color_idx=color_idx,
            is_generation_target=is_generation_target,
            target_token_count=target_token_count,
        )

    fids = sf.feature_ids
    fmags = sf.feature_mags
    if feature_mask is not None:
        fids, fmags = feature_mask.apply_to_positive(fids, fmags)

    # Apply training-time threshold filtering (before discretization)
    if threshold_cfg is not None:
        fids, fmags = apply_threshold(threshold_cfg, fids, fmags, precomputed_thresholds)

    if fids.numel() == 0:
        return Segment(
            segment_idx=sf.segment_idx,
            char_start=sf.char_start,
            char_end=sf.char_end,
            features=[],
            color_idx=color_idx,
            is_generation_target=is_generation_target,
            target_token_count=target_token_count,
        )

    buckets = discretize_activations(fids, fmags, boundaries)

    fids_list: list[int] = fids.tolist()
    mags_list: list[int] = [
        internal_to_display(int(b), num_buckets, has_negatives) for b in buckets.tolist()
    ]

    features = [
        FeatureCondition(feature_id=fid, magnitude=mag)
        for fid, mag in zip(fids_list, mags_list, strict=True)
    ]

    return Segment(
        segment_idx=sf.segment_idx,
        char_start=sf.char_start,
        char_end=sf.char_end,
        features=features,
        color_idx=color_idx,
        is_generation_target=is_generation_target,
        target_token_count=target_token_count,
    )


class DocumentLoader:
    """Loads documents and features for the completion TUI.

    Targets the validation shard (last shard alphabetically) and its
    corresponding parquet file.
    """

    def __init__(
        self,
        feature_dir: str,
        parquet_dir: str,
        boundaries: t.Tensor,
        num_buckets: int,
        has_negatives: bool,
        threshold_cfg: ThresholdConfig | None = None,
        precomputed_thresholds: t.Tensor | None = None,
        sampling_config: TUIFeatureSamplingConfig | None = None,
        *,
        percentile_checksum: str | None = None,
        skip_feature_mags: bool = False,
        compact_feature_ids: bool = False,
    ) -> None:
        self._boundaries = boundaries
        self._num_buckets = num_buckets
        self._has_negatives = has_negatives
        self._threshold_cfg = threshold_cfg
        self._precomputed_thresholds = precomputed_thresholds
        self._sampling_config = sampling_config
        # Textual document workers are cancellable at the task level, but a
        # Python thread already inside FeatureIndex.lookup may finish in the
        # background. Serialize the small LRU/cache mutation so stale workers
        # cannot race a newer request.
        self._feature_index_lock = threading.RLock()

        if percentile_checksum is None:
            self._feature_index = FeatureIndex(
                feature_dir,
                skip_feature_mags=skip_feature_mags,
                compact_feature_ids=compact_feature_ids,
            )
        else:
            self._feature_index = FeatureIndex(
                feature_dir,
                percentile_checksum=percentile_checksum,
                skip_feature_mags=skip_feature_mags,
                compact_feature_ids=compact_feature_ids,
            )
        stems = self._feature_index.shard_stems
        if not stems:
            raise ValueError(f"No feature shards found in {feature_dir}")

        # Validation shard = last shard alphabetically
        self._val_stem = stems[-1]

        # Find matching parquet file
        parquet_path = Path(parquet_dir) / f"{self._val_stem}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"No parquet file for validation shard: {parquet_path}")
        self._parquet_path = parquet_path

        # Load parquet texts
        pf = pq.ParquetFile(str(self._parquet_path))
        table = pf.read()
        self._texts: list[str] = table.column("text").to_pylist()

        # Build available doc_ids, filtering to those within parquet range
        n_texts = len(self._texts)
        self._doc_ids = [
            d for d in self._feature_index.doc_ids_for_shard(self._val_stem) if d < n_texts
        ]

        logger.info(
            f"DocumentLoader: validation shard={self._val_stem}, "
            f"{len(self._doc_ids)} docs, {len(self._texts)} parquet rows"
        )

    @property
    def validation_shard_stem(self) -> str:
        return self._val_stem

    def available_doc_ids(self) -> list[int]:
        """Return sorted list of document indices in the validation shard."""
        return list(self._doc_ids)

    def load_document(self, doc_idx: int, num_segments: int | None = None) -> DocumentData | None:
        """Load document text and features.

        Returns None if doc_idx is not in the validation shard's feature index.
        """
        if doc_idx not in self._doc_ids:
            return None
        with self._feature_index_lock:
            seg_features = self._feature_index.lookup(self._val_stem, doc_idx)

        if doc_idx >= len(self._texts):
            return None

        text = self._texts[doc_idx]
        total = len(seg_features)

        # Optionally limit segments
        if num_segments is not None:
            seg_features = seg_features[:num_segments]

        # Mark last segment as generation target when >1 segment loaded
        mark_last = len(seg_features) > 1
        seg_size = self._feature_index.segment_size

        use_sampling = self._sampling_config is not None and (
            self._sampling_config.feature_selection is not None
            or self._sampling_config.negative_selection is not None
        )

        segments = []
        evidence_by_segment_index: dict[int, SegmentClusterEvidence] = {}
        for i, sf in enumerate(seg_features):
            is_target = mark_last and i == len(seg_features) - 1
            target_token_count = seg_size if is_target else None
            if (
                self._sampling_config is not None
                and self._sampling_config.consolidation is not None
            ):
                clustered = clustered_sample_segment_features(
                    sf,
                    self._sampling_config,
                    self._num_buckets,
                    self._has_negatives,
                )
                seg = Segment(
                    segment_idx=sf.segment_idx,
                    char_start=sf.char_start,
                    char_end=sf.char_end,
                    features=clustered.features,
                    color_idx=i,
                    is_generation_target=is_target,
                    target_token_count=target_token_count,
                )
                if clustered.evidence.fingerprint or clustered.evidence.evidence:
                    evidence_by_segment_index[i] = clustered.evidence
            else:
                seg = segment_features_to_segment(
                    sf,
                    color_idx=i,
                    boundaries=self._boundaries,
                    num_buckets=self._num_buckets,
                    has_negatives=self._has_negatives,
                    is_generation_target=is_target,
                    target_token_count=target_token_count,
                    threshold_cfg=self._threshold_cfg,
                    precomputed_thresholds=self._precomputed_thresholds,
                    feature_mask=(
                        self._sampling_config.feature_mask
                        if self._sampling_config is not None
                        else None
                    ),
                )
            if use_sampling and (
                self._sampling_config is not None and self._sampling_config.consolidation is None
            ):
                assert self._sampling_config is not None
                seg.features = sample_segment_features(
                    sf,
                    self._sampling_config,
                    self._num_buckets,
                    self._has_negatives,
                    feature_mask=self._sampling_config.feature_mask,
                )
            segments.append(seg)

        return DocumentData(
            doc_idx=doc_idx,
            text=text,
            segments=segments,
            total_segments=total,
            evidence_by_segment_index=evidence_by_segment_index,
        )

    def raw_lookup(self, doc_idx: int) -> list[SegmentFeatures] | None:
        """Return raw SegmentFeatures for doc_idx, or None if not in the index."""
        if doc_idx not in self._doc_ids:
            return None
        with self._feature_index_lock:
            return self._feature_index.lookup(self._val_stem, doc_idx)

    def total_segments(self, doc_idx: int) -> int | None:
        """Return total segment count for a document without loading all data."""
        if doc_idx not in self._doc_ids:
            return None
        with self._feature_index_lock:
            return len(self._feature_index.lookup(self._val_stem, doc_idx))
