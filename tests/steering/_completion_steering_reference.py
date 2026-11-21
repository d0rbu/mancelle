"""Test-only reference paths for the completion-steering tensor pipeline.

Production uses ``scan_validation_features_vectorized`` and tensor-native
selection. These deliberately straightforward Python scan/conversion
implementations remain as differential-test oracles without expanding the
runtime API or leaving retired dict/dataclass paths in production.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.completion_steering_eval import (
    ScanResultTensors,
    _activation_to_percentile,
    _build_percentile_lookup,
)
from steering.stats_types import PerFeatureStat


@dataclass(frozen=True)
class DocFeatureActivation:
    """Per-document activation record used by the reference path."""

    doc_idx: int
    max_activation: float
    max_segment_idx: int
    percentile: float
    total_segments: int


@dataclass(frozen=True)
class FeatureScanResult:
    """Per-feature aggregate used by the reference path."""

    feature_id: int
    n_docs_activated: int
    doc_activations: list[DocFeatureActivation]


def scan_validation_features(
    doc_segments: dict[int, list[SegmentFeatures]],
    per_feature_stats: dict[str, PerFeatureStat],
) -> dict[int, FeatureScanResult]:
    """Reference scan over Python document and segment objects."""
    pct_lookups = {
        fid_str: _build_percentile_lookup(stat.percentiles)
        for fid_str, stat in per_feature_stats.items()
    }
    feature_activations: dict[int, list[DocFeatureActivation]] = defaultdict(list)

    for doc_idx, segments in doc_segments.items():
        total_segments = len(segments)
        non_first = [seg for seg in segments if seg.segment_idx != 0]
        if not non_first:
            continue

        feature_max: dict[int, tuple[float, int]] = {}
        for seg in non_first:
            ids = seg.feature_ids.tolist()
            mags = seg.feature_mags.tolist()
            for fid, mag in zip(ids, mags, strict=True):
                feature_id = int(fid)
                magnitude = float(mag)
                previous = feature_max.get(feature_id)
                if previous is None or magnitude > previous[0]:
                    feature_max[feature_id] = (magnitude, seg.segment_idx)

        for feature_id, (max_activation, max_segment_idx) in feature_max.items():
            lookup = pct_lookups.get(str(feature_id))
            if lookup is None:
                raise KeyError(f"Feature {feature_id} not found in per_feature_stats")
            feature_activations[feature_id].append(
                DocFeatureActivation(
                    doc_idx=doc_idx,
                    max_activation=max_activation,
                    max_segment_idx=max_segment_idx,
                    percentile=_activation_to_percentile(max_activation, lookup),
                    total_segments=total_segments,
                )
            )

    return {
        feature_id: FeatureScanResult(
            feature_id=feature_id,
            n_docs_activated=len(activations),
            doc_activations=activations,
        )
        for feature_id, activations in feature_activations.items()
    }


def scan_results_to_dict(
    scan_results: ScanResultTensors,
) -> dict[int, FeatureScanResult]:
    """Reference conversion from tensor CSR to dict/dataclass records."""
    results: dict[int, FeatureScanResult] = {}
    for i, feature_id_tensor in enumerate(scan_results.feature_ids):
        feature_id = int(feature_id_tensor.item())
        start = int(scan_results.offsets[i].item())
        end = int(scan_results.offsets[i + 1].item())
        activations = [
            DocFeatureActivation(
                doc_idx=int(scan_results.doc_idx[j].item()),
                max_activation=float(scan_results.max_activation[j].item()),
                max_segment_idx=int(scan_results.max_segment_idx[j].item()),
                percentile=float(scan_results.percentile[j].item()),
                total_segments=int(scan_results.total_segments[j].item()),
            )
            for j in range(start, end)
        ]
        results[feature_id] = FeatureScanResult(
            feature_id=feature_id,
            n_docs_activated=int(scan_results.n_docs_activated[i].item()),
            doc_activations=activations,
        )
    return results
