"""Extract absolute or differential features with sampled cardinality.

Loads feature statistics from an SFT dataset, samples a feature count from the
same distribution config used by encoder-decoder training, selects the top-k
features by mean activation percentile, discretizes their magnitudes into
bucket indices using boundaries from a separate dataset (default: pretraining
base), and outputs a ``{feature_id: bucket_index}`` mapping ready for eval
scripts.

The number of features to select defaults to ``feature_selection`` (a
``SamplingSelection`` config such as geometric or normal).  It can be
overridden by ``top_k``.

Ranking uses activation percentiles computed from the source feature shards
against the boundary/reference statistics table.  This avoids comparing raw SAE
magnitudes across features with different activation scales.  Passing
``baseline_dir`` enables delta mode, subtracting baseline percentile means from
source percentile means using the same reference table for both datasets.

Magnitude percentile defaults to p50 but is configurable via
``magnitude_percentile`` (e.g. 0.9 for p90).  This raw magnitude is taken from
the source/steering dataset statistics and then bucketized against the boundary
statistics source.

Output path is auto-generated from settings when ``output`` is null:
    out/data/common_features/{phase}/absolute/{selection}/seed_{seed}/...
    out/data/common_features/{phase}/delta_{baseline}/{selection}/seed_{seed}/...

Example:
    python -m scripts.preprocess.extract_common_features feature_dir=.../safety_sft
    python -m scripts.preprocess.extract_common_features feature_dir=.../safety_sft \\
        feature_selection.mean=20 feature_selection.max_k=80 magnitude_percentile=0.9
    python -m scripts.preprocess.extract_common_features feature_dir=.../safety_sft \\
        stats_dir=.../base top_k=64

    # Then use the output as a target-delta config with eval scripts that also
    # receive full prompt/context features from conditioning.context_cache_dir.
    python -m steering.eval.dclm_eval source=encdec feature_configs=features.yaml conditioning.context_cache_dir=...
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import hydra
import torch as t
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from steering.conditional.discretization import (
    build_boundaries_from_stats,
    discretize_activations,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    magnitude_to_percentile,
)
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingSelection,
    ThresholdSelection,
    assert_no_gradient_ranking,
    parse_feature_selection,
    parse_negative_selection,
    sample_k,
)
from steering.feature_io import glob_features, load_features
from steering.stats_io import find_statistics_path, load_statistics
from steering.stats_types import ExtractionStatistics, PerFeatureStat

_PERCENTILE_SCAN_CHUNK_SIZE = 250_000


class FeatureRankMetric(StrEnum):
    """Feature ranking metrics for final conditioning-signal extraction."""

    MEAN_PCT = "mean_pct"
    MEAN_PCT_INCL_ZEROS = "mean_pct_incl_zeros"

    @classmethod
    def parse(cls, value) -> "FeatureRankMetric":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            supported = ", ".join(metric.value for metric in cls)
            raise ValueError(
                f"rank_metric must be one of: {supported}; got {value!r}. "
                "Raw magnitude ranking is intentionally unsupported."
            ) from None


@dataclass(frozen=True)
class FeaturePercentileMeans:
    """Per-feature activation percentile sums/counts over a feature directory."""

    percentile_sums: t.Tensor
    active_counts: t.Tensor
    n_segments: int

    def scores(self, metric: FeatureRankMetric) -> t.Tensor:
        """Return dense scores for the requested percentile-mean metric."""
        if metric is FeatureRankMetric.MEAN_PCT_INCL_ZEROS:
            if self.n_segments <= 0:
                return t.zeros_like(self.percentile_sums, dtype=t.float64)
            return self.percentile_sums / float(self.n_segments)
        if metric is FeatureRankMetric.MEAN_PCT:
            out = t.zeros_like(self.percentile_sums, dtype=t.float64)
            mask = self.active_counts > 0
            out[mask] = self.percentile_sums[mask] / self.active_counts[mask].to(
                t.float64
            )
            return out
        raise ValueError(f"Unsupported rank metric: {metric!r}")


def resolve_base_stats_dir(feature_dir: str) -> str:
    """Resolve the default base (pretraining) stats directory.

    Given an SFT feature directory like ``.../m_gemma-2-2b/safety_sft``,
    returns the sibling ``base`` directory: ``.../m_gemma-2-2b/base``.

    Args:
        feature_dir: Path to the SFT feature directory.

    Returns:
        Path to the base sibling directory.

    Raises:
        FileNotFoundError: If the base directory or its statistics file
            does not exist.
    """
    base_dir = Path(feature_dir).parent / "base"
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Default base stats directory not found: {base_dir}\n"
            f"Set stats_dir= explicitly to use a different boundary source."
        )
    stats_path = find_statistics_path(str(base_dir))
    if stats_path is None:
        raise FileNotFoundError(
            f"No feature_statistics.safetensors in base directory: {base_dir}\n"
            f"Set stats_dir= explicitly to use a different boundary source."
        )
    return str(base_dir)


def _format_path_float(value: float) -> str:
    """Format a float for stable, readable output path components."""
    return str(float(value))


def _coerce_sampling_selection(
    raw: SamplingSelection | dict | DictConfig | None,
) -> SamplingSelection | None:
    """Resolve a Hydra/dict/dataclass feature selection into SamplingSelection."""
    if raw is None:
        return None
    if isinstance(raw, SamplingSelection):
        return raw
    if isinstance(raw, DictConfig):
        container = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(
                "feature_selection must resolve to a mapping; "
                f"got {type(container).__name__}"
            )
        raw = cast(dict, container)
    if not isinstance(raw, dict):
        raise TypeError(
            "feature_selection must be a SamplingSelection, mapping, or null; "
            f"got {type(raw).__name__}"
        )
    parsed = parse_feature_selection(raw)
    assert_no_gradient_ranking(parsed, "extract_common_features selection")
    if isinstance(parsed, ThresholdSelection):
        raise ValueError(
            "extract_common_features expects sampling-based feature_selection "
            "(distribution/mean/std). ThresholdSelection is for per-segment "
            "feature shards, not final conditioning-feature YAML generation."
        )
    return parsed


def _coerce_negative_selection(
    raw: NegativeSamplingSelection | dict | DictConfig | None,
) -> NegativeSamplingSelection | None:
    """Resolve a Hydra/dict/dataclass negative selection config."""
    if raw is None:
        return None
    if isinstance(raw, NegativeSamplingSelection):
        return raw
    if isinstance(raw, DictConfig):
        container = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(
                "negative_selection must resolve to a mapping; "
                f"got {type(container).__name__}"
            )
        raw = cast(dict, container)
    if not isinstance(raw, dict):
        raise TypeError(
            "negative_selection must be a NegativeSamplingSelection, mapping, "
            f"or null; got {type(raw).__name__}"
        )
    return parse_negative_selection(raw)


def sample_feature_count(
    selection: SamplingSelection | NegativeSamplingSelection,
    seed: int,
) -> int:
    """Sample k using the repo's shared SamplingSelection semantics."""
    rng = t.Generator().manual_seed(int(seed))
    return sample_k(
        selection.distribution,
        selection.mean,
        selection.std,
        rng,
        min_k=selection.min_k,
        max_k=selection.max_k,
    )


def resolve_selected_feature_count(
    *,
    feature_selection: SamplingSelection | dict | DictConfig | None = None,
    top_k: int | None = None,
    seed: int = 42,
) -> tuple[int, str, SamplingSelection | None]:
    """Resolve the number of features to emit and a human-readable source.

    Priority:
      1. ``top_k`` explicit override.
      2. ``feature_selection`` sampling config (geometric/normal via sample_k).
    """
    parsed_selection = _coerce_sampling_selection(feature_selection)

    if top_k is not None:
        k = int(top_k)
        if k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        return k, "top_k override", parsed_selection

    if parsed_selection is not None:
        k = sample_feature_count(parsed_selection, seed)
        return (
            k,
            (
                f"{parsed_selection.distribution.value} sample "
                f"(mean={parsed_selection.mean}, std={parsed_selection.std}, "
                f"min_k={parsed_selection.min_k}, max_k={parsed_selection.max_k}, "
                f"seed={seed})"
            ),
            parsed_selection,
        )

    raise ValueError(
        "Feature count selection requires feature_selection or top_k"
    )


def validate_stats_compatible(
    left_stats: ExtractionStatistics,
    right_stats: ExtractionStatistics,
    *,
    left_label: str,
    right_label: str,
    purpose: str,
) -> None:
    """Validate that two statistics files describe the same SAE feature space."""
    left_meta = left_stats.metadata
    right_meta = right_stats.metadata
    checks = [
        ("n_sae_features", left_meta.n_sae_features, right_meta.n_sae_features),
        ("layer", left_meta.layer, right_meta.layer),
        ("pooling_mode", left_meta.pooling_mode, right_meta.pooling_mode),
        ("segment_mode", left_meta.segment_mode, right_meta.segment_mode),
        ("segment_size", left_meta.segment_size, right_meta.segment_size),
        ("context_window", left_meta.context_window, right_meta.context_window),
    ]
    mismatches = [
        f"{name}: {left_label}={left!r}, {right_label}={right!r}"
        for name, left, right in checks
        if left != right
    ]
    if mismatches:
        raise ValueError(
            f"Cannot {purpose} from incompatible statistics:\n  "
            + "\n  ".join(mismatches)
        )


def validate_delta_stats_compatible(
    source_stats: ExtractionStatistics,
    baseline_stats: ExtractionStatistics,
) -> None:
    """Validate that two feature directories can be compared by percentile delta."""
    validate_stats_compatible(
        source_stats,
        baseline_stats,
        left_label="source",
        right_label="baseline",
        purpose="compute percentile deltas",
    )


def _stats_feature_id_set(per_feature_stats: dict[str, PerFeatureStat]) -> set[int]:
    """Return feature IDs covered by an ExtractionStatistics per-feature table."""
    return {int(fid) for fid in per_feature_stats}


def _validate_reference_feature_coverage(
    reference_feature_ids: set[int],
    required_feature_ids: Sequence[int],
    *,
    context: str,
) -> None:
    """Ensure every required feature has reference percentiles."""
    missing = sorted({int(fid) for fid in required_feature_ids} - reference_feature_ids)
    if missing:
        preview = missing[:10]
        suffix = "" if len(missing) <= 10 else f" ... ({len(missing)} total)"
        raise ValueError(
            f"Reference statistics do not cover {context} feature IDs: "
            f"{preview}{suffix}. Ranking by activation percentile requires "
            "reference percentiles for every candidate feature."
        )


def _activation_percentiles_chunked(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    percentile_table: t.Tensor,
    *,
    chunk_size: int = _PERCENTILE_SCAN_CHUNK_SIZE,
) -> t.Tensor:
    """Map shard magnitudes to normalized percentiles without raw ranking."""
    n = int(feature_ids.numel())
    if n == 0:
        return t.empty(0, dtype=t.float64)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    parts: list[t.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pct = magnitude_to_percentile(
            feature_ids[start:end].to(t.int64),
            feature_mags[start:end],
            percentile_table,
        )
        parts.append((pct / 100.0).to(t.float64))
    return t.cat(parts) if len(parts) > 1 else parts[0]


def compute_feature_percentile_means(
    feature_dir: str | Path,
    percentile_table: t.Tensor,
    n_sae_features: int,
    *,
    chunk_size: int = _PERCENTILE_SCAN_CHUNK_SIZE,
    reference_feature_ids: set[int] | None = None,
) -> FeaturePercentileMeans:
    """Compute per-feature mean activation percentiles from feature shards.

    Percentiles are computed from every active feature occurrence in the source
    shards against the caller-provided percentile table.  Inactive segments are
    represented only in ``n_segments``; ``MEAN_PCT_INCL_ZEROS`` divides active
    percentile sums by this segment count.
    """
    feature_dir = Path(feature_dir)
    shard_paths = glob_features(feature_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No feature shards found in {feature_dir}")

    percentile_sums = t.zeros(n_sae_features, dtype=t.float64)
    active_counts = t.zeros(n_sae_features, dtype=t.int64)
    n_segments = 0

    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        shard = load_features(shard_path)
        fids = shard["feature_ids"].to(t.int64)
        mags = shard["feature_mags"]
        n_segments += int(shard["n_segments"])
        if fids.numel() == 0:
            continue
        if fids.min().item() < 0 or fids.max().item() >= n_sae_features:
            raise ValueError(
                f"Feature IDs in {shard_path} exceed n_sae_features={n_sae_features}"
            )
        shard_counts = (
            t.bincount(fids, minlength=n_sae_features)
            if fids.ndim == 1 and fids.device.type == "cpu"
            else None
        )
        if reference_feature_ids is not None:
            active_ids = [
                int(fid)
                for fid in (
                    shard_counts.nonzero().flatten().tolist()
                    if shard_counts is not None
                    else fids.unique().tolist()
                )
            ]
            _validate_reference_feature_coverage(
                reference_feature_ids,
                active_ids,
                context=f"active shard {shard_path}",
            )

        pcts = _activation_percentiles_chunked(
            fids,
            mags,
            percentile_table,
            chunk_size=chunk_size,
        )
        percentile_sums.scatter_add_(0, fids, pcts)
        if shard_counts is None:
            active_counts.scatter_add_(
                0,
                fids,
                t.ones_like(fids, dtype=t.int64),
            )
        else:
            active_counts += shard_counts
        if shard_idx == 1 or shard_idx % 10 == 0 or shard_idx == len(shard_paths):
            logger.info(
                f"Scanned {shard_idx}/{len(shard_paths)} shard(s) in {feature_dir} "
                f"for activation percentile means"
            )

    return FeaturePercentileMeans(
        percentile_sums=percentile_sums,
        active_counts=active_counts,
        n_segments=n_segments,
    )


def _candidate_feature_ids(
    per_feature_stats: Mapping[str, PerFeatureStat],
) -> list[int]:
    """Return feature ids in numeric form for bucketizable candidates."""
    return [int(fid) for fid in per_feature_stats]


def _resolve_candidate_feature_ids(
    candidate_features: dict[str, PerFeatureStat] | Sequence[int],
) -> list[int]:
    """Normalize candidate ids from either a stats table or an explicit list."""
    if isinstance(candidate_features, dict):
        return _candidate_feature_ids(cast(dict[str, PerFeatureStat], candidate_features))
    return [int(fid) for fid in candidate_features]


def _scores_for_candidates(
    dense_scores: t.Tensor,
    candidate_feature_ids: Sequence[int],
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for fid in candidate_feature_ids:
        if fid < 0 or fid >= dense_scores.shape[0]:
            raise ValueError(
                f"Candidate feature {fid} is outside score table shape "
                f"{tuple(dense_scores.shape)}"
            )
        score = float(dense_scores[fid].item())
        if not math.isfinite(score):
            raise ValueError(f"Feature {fid} has non-finite score: {score!r}")
        scores[fid] = score
    return scores


def select_top_feature_ids_by_score(
    scores: dict[int, float],
    k: int,
) -> list[int]:
    """Select top-k feature ids by score with deterministic tie-breaking.

    Ties are broken by feature id ascending to keep outputs deterministic
    without introducing arbitrary dict-insertion-order bias.
    """
    if k <= 0 or not scores:
        return []
    for fid, score in scores.items():
        if not math.isfinite(float(score)):
            raise ValueError(f"Feature {fid} has non-finite score: {score!r}")
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [fid for fid, _score in ranked[: min(k, len(ranked))]]


def select_top_features_by_mean_percentile(
    percentile_means: FeaturePercentileMeans,
    per_feature_stats: dict[str, PerFeatureStat],
    k: int,
    *,
    rank_metric: FeatureRankMetric | str = FeatureRankMetric.MEAN_PCT_INCL_ZEROS,
) -> list[int]:
    """Select top-k source features by activation percentile mean."""
    metric = FeatureRankMetric.parse(rank_metric)
    dense_scores = percentile_means.scores(metric)
    candidate_scores = _scores_for_candidates(
        dense_scores,
        _candidate_feature_ids(per_feature_stats),
    )
    return select_top_feature_ids_by_score(candidate_scores, k)


def compute_delta_feature_scores(
    left_means: FeaturePercentileMeans,
    right_means: FeaturePercentileMeans,
    candidate_features: dict[str, PerFeatureStat] | Sequence[int],
    *,
    rank_metric: FeatureRankMetric | str = FeatureRankMetric.MEAN_PCT_INCL_ZEROS,
) -> dict[int, float]:
    """Compute positive ``left_means - right_means`` over candidate features."""
    metric = FeatureRankMetric.parse(rank_metric)
    left_scores = left_means.scores(metric)
    right_scores = right_means.scores(metric)
    if left_scores.shape != right_scores.shape:
        raise ValueError(
            "Cannot compute percentile deltas with different score shapes: "
            f"left={tuple(left_scores.shape)}, right={tuple(right_scores.shape)}"
        )
    deltas = left_scores - right_scores
    candidate_scores = _scores_for_candidates(
        deltas,
        _resolve_candidate_feature_ids(candidate_features),
    )
    return {fid: score for fid, score in candidate_scores.items() if score > 0.0}


def build_feature_inputs_for_selected_ids(
    per_feature_stats: dict[str, PerFeatureStat],
    boundaries: t.Tensor,
    selected_feature_ids: list[int],
    device: str | t.device,
    *,
    magnitude_percentile: str = "p50",
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Build one-segment encoder inputs for already-selected feature ids.

    The caller has already chosen the feature IDs. This helper only looks up the
    requested absolute activation percentile in the steering statistics and
    discretizes those raw magnitudes with the shared bucketizer.
    """
    if not selected_feature_ids:
        return (
            t.empty(1, 1, 0, dtype=t.int64, device=device),
            t.empty(1, 1, 0, dtype=t.int64, device=device),
            t.empty(1, 1, 0, dtype=t.bool, device=device),
        )

    for fid in selected_feature_ids:
        fid_str = str(fid)
        if fid_str not in per_feature_stats:
            raise KeyError(f"Selected feature {fid} not found in per_feature_stats")
        if fid < 0 or fid >= boundaries.shape[0]:
            raise ValueError(
                f"Selected feature {fid} is outside boundary table shape "
                f"{tuple(boundaries.shape)}"
            )
        pcts = per_feature_stats[fid_str].percentiles
        if magnitude_percentile not in pcts:
            raise KeyError(
                f"Percentile key {magnitude_percentile!r} not found for feature {fid}. "
                f"Available: {sorted(pcts.keys())}"
            )

    fids = t.tensor(selected_feature_ids, dtype=t.int64, device=device)
    raw_magnitudes = t.tensor(
        [
            per_feature_stats[str(fid)].percentiles[magnitude_percentile]
            for fid in selected_feature_ids
        ],
        dtype=t.float64,
        device=device,
    )
    bucket_indices = discretize_activations(
        fids,
        raw_magnitudes,
        boundaries.to(device),
    )

    enc_feature_ids = fids.unsqueeze(0).unsqueeze(0)
    enc_magnitudes = bucket_indices.unsqueeze(0).unsqueeze(0)
    enc_mask = t.ones(1, 1, len(selected_feature_ids), dtype=t.bool, device=device)
    return enc_feature_ids, enc_magnitudes, enc_mask


def _selection_path_component(
    *,
    feature_selection: SamplingSelection | None,
    top_k: int | None,
) -> Path:
    """Build the selection-specific output path component."""
    if top_k is not None:
        k = int(top_k)
        if k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        return Path(f"k{k}")
    if feature_selection is None:
        raise ValueError(
            "Output path selection requires feature_selection or top_k"
        )
    max_part = (
        "none" if feature_selection.max_k is None else str(feature_selection.max_k)
    )
    return (
        Path(f"dist_{feature_selection.distribution.value}")
        / f"mean_{_format_path_float(feature_selection.mean)}"
        / f"std_{_format_path_float(feature_selection.std)}"
        / f"min_{feature_selection.min_k}"
        / f"max_{max_part}"
    )


def _negative_selection_path_component(
    negative_selection: NegativeSamplingSelection | None,
) -> Path:
    """Build the negative-selection output path component."""
    if negative_selection is None:
        return Path()
    max_part = (
        "none" if negative_selection.max_k is None else str(negative_selection.max_k)
    )
    return (
        Path(f"neg_dist_{negative_selection.distribution.value}")
        / f"mean_{_format_path_float(negative_selection.mean)}"
        / f"std_{_format_path_float(negative_selection.std)}"
        / f"min_{negative_selection.min_k}"
        / f"max_{max_part}"
    )


def _build_discretization_boundaries(
    per_feature_stats: dict[str, PerFeatureStat],
    n_sae_features: int,
    effective_buckets: int,
) -> t.Tensor:
    """Build positive-feature bucket boundaries with binary-mode handling."""
    if effective_buckets < 1:
        raise ValueError(
            "effective_buckets must be >= 1; "
            f"got {effective_buckets}"
        )
    if effective_buckets == 1:
        return t.zeros(n_sae_features, 0, dtype=t.float64)
    return build_boundaries_from_stats(
        per_feature_stats,
        n_sae_features,
        effective_buckets,
    )


def auto_output_path(
    feature_dir: str,
    magnitude_percentile: float,
    *,
    feature_selection: SamplingSelection | dict | DictConfig | None = None,
    top_k: int | None = None,
    seed: int = 42,
    rank_metric: FeatureRankMetric | str = FeatureRankMetric.MEAN_PCT_INCL_ZEROS,
    baseline_dir: str | None = None,
    negative_selection: NegativeSamplingSelection | dict | DictConfig | None = None,
) -> str:
    """Generate output path from feature directory and selection settings.

    Args:
        feature_dir: Path to the SFT feature directory (last component is phase).
        magnitude_percentile: Magnitude percentile (0-1).
        feature_selection: Optional sampling selection for k.
        top_k: Optional explicit count override.
        seed: Seed used when sampling k.
        rank_metric: Percentile-mean metric used for ranking.
        baseline_dir: Optional baseline feature directory. If set, output is
            named as a delta selector; otherwise it is named ``absolute``.

    Returns:
        Auto-generated output file path.
    """
    phase = Path(feature_dir).name
    metric = FeatureRankMetric.parse(rank_metric)
    parsed_selection = _coerce_sampling_selection(feature_selection)
    parsed_negative_selection = _coerce_negative_selection(negative_selection)
    if parsed_negative_selection is not None and baseline_dir is None:
        raise ValueError(
            "negative_selection requires baseline_dir because negative common "
            "features are only defined for delta extraction."
        )
    mp_str = str(magnitude_percentile)
    selection_path = _selection_path_component(
        feature_selection=parsed_selection,
        top_k=top_k,
    )
    if baseline_dir is None:
        mode_path = Path("absolute")
    else:
        mode_path = Path(f"delta_{Path(baseline_dir).name}")
    path = Path("out/data/common_features") / phase / mode_path / selection_path
    if parsed_negative_selection is not None:
        path = path / _negative_selection_path_component(parsed_negative_selection)
    return str(
        path / f"seed_{seed}" / f"r_{metric.value}" / f"m{mp_str}" / "features.yaml"
    )


def extract_common_features(
    feature_dir: str,
    num_buckets: int,
    *,
    stats_dir: str | None = None,
    baseline_dir: str | None = None,
    magnitude_percentile: float = 0.5,
    top_k: int | None = None,
    feature_selection: SamplingSelection | dict | DictConfig | None = None,
    negative_selection: NegativeSamplingSelection | dict | DictConfig | None = None,
    seed: int = 42,
    rank_metric: FeatureRankMetric | str = FeatureRankMetric.MEAN_PCT_INCL_ZEROS,
) -> dict[int, int]:
    """Extract absolute or differential features with sampled k and percentile ranking.

    Loads SFT feature statistics for feature selection and magnitude lookup,
    loads boundary statistics (default: pretraining base) for discretization,
    and returns a ``{feature_id: bucket_index}`` mapping.

    Args:
        feature_dir: Path to SFT feature directory with
            ``feature_statistics.safetensors``.
        num_buckets: Number of magnitude buckets (must match training config).
        stats_dir: Path to directory with statistics for bucket boundaries.
            If None, resolves to the ``base/`` sibling of ``feature_dir``.
        baseline_dir: Optional baseline/reference feature directory. When set,
            enables delta selection and ranks source-minus-baseline percentile
            mean deltas. When null, uses absolute source percentile means.
        magnitude_percentile: Per-feature magnitude percentile from SFT stats
            (e.g. 0.5 for p50, 0.9 for p90).
        top_k: Explicit feature count override. If set, ``feature_selection``
            is ignored for count selection.
        feature_selection: SamplingSelection or config dict used to sample k.
            Supports the same geometric/normal distributions as training.
        seed: RNG seed for sampling k.
        rank_metric: FeatureRankMetric used to rank features. Raw magnitudes are
            intentionally unsupported; use percentile means only.

    Returns:
        ``{feature_id: bucket_index}`` dict sorted by percentile mean score
        descending (ties by feature id ascending).

    Raises:
        FileNotFoundError: If statistics files are not found.
        KeyError: If required percentile keys are missing from statistics.
    """
    metric = FeatureRankMetric.parse(rank_metric)
    is_delta = baseline_dir is not None
    selection_kind = "delta" if is_delta else "absolute"
    k, count_source, parsed_selection = resolve_selected_feature_count(
        feature_selection=feature_selection,
        top_k=top_k,
        seed=seed,
    )
    parsed_negative_selection = _coerce_negative_selection(negative_selection)
    if parsed_negative_selection is not None:
        if not is_delta:
            raise ValueError(
                "negative_selection requires baseline_dir because negative common "
                "features are only defined for delta extraction."
            )
        if num_buckets < 2:
            raise ValueError(
                "negative_selection requires num_buckets >= 2 so the last bucket "
                "can be reserved for inactive features."
            )

    # Load SFT statistics (feature selection + magnitude percentiles)
    sft_stats_path = find_statistics_path(feature_dir)
    if sft_stats_path is None:
        raise FileNotFoundError(
            f"No feature_statistics.safetensors found in {feature_dir}"
        )
    sft_stats = load_statistics(sft_stats_path)

    default_base_dir: str | None = None
    if stats_dir is None:
        default_base_dir = resolve_base_stats_dir(feature_dir)

    # Resolve boundary source (pretraining base by default)
    if stats_dir is None:
        assert default_base_dir is not None
        stats_dir = default_base_dir

    boundary_stats_path = find_statistics_path(stats_dir)
    if boundary_stats_path is None:
        raise FileNotFoundError(
            f"No feature_statistics.safetensors found in stats_dir={stats_dir}"
        )
    boundary_stats = (
        sft_stats
        if boundary_stats_path == sft_stats_path
        else load_statistics(boundary_stats_path)
    )
    validate_stats_compatible(
        sft_stats,
        boundary_stats,
        left_label="source",
        right_label="boundary",
        purpose="rank and bucketize conditioning features",
    )

    baseline_stats: ExtractionStatistics | None = None
    if is_delta:
        assert baseline_dir is not None
        baseline_stats_path = find_statistics_path(baseline_dir)
        if baseline_stats_path is None:
            raise FileNotFoundError(
                f"No feature_statistics.safetensors found in baseline_dir={baseline_dir}"
            )
        baseline_stats = load_statistics(baseline_stats_path)
        validate_delta_stats_compatible(sft_stats, baseline_stats)

    negative_k = 0
    negative_count_source = "negative_selection disabled"
    if parsed_negative_selection is not None:
        negative_seed = seed + 1
        negative_k = sample_feature_count(parsed_negative_selection, negative_seed)
        negative_count_source = (
            f"{parsed_negative_selection.distribution.value} sample "
            f"(mean={parsed_negative_selection.mean}, std={parsed_negative_selection.std}, "
            f"min_k={parsed_negative_selection.min_k}, "
            f"max_k={parsed_negative_selection.max_k}, seed={negative_seed})"
        )

    logger.info(
        f"Selecting {k} feature(s) by {metric.value} "
        f"({count_source}, selection_kind={selection_kind})"
    )
    if parsed_negative_selection is not None:
        logger.info(
            f"Selecting {negative_k} negative feature(s) by reversed delta "
            f"({negative_count_source})"
        )

    # Build raw-magnitude bucket boundaries and the percentile lookup table from
    # the same boundary/reference source. Ranking never uses raw magnitudes.
    boundary_per_feature = boundary_stats.per_feature_stats
    boundary_n_sae = boundary_stats.metadata.n_sae_features
    reference_feature_ids = _stats_feature_id_set(boundary_per_feature)
    _validate_reference_feature_coverage(
        reference_feature_ids,
        _candidate_feature_ids(sft_stats.per_feature_stats),
        context="source candidate",
    )
    if parsed_negative_selection is not None:
        assert baseline_stats is not None
        _validate_reference_feature_coverage(
            reference_feature_ids,
            _candidate_feature_ids(baseline_stats.per_feature_stats),
            context="baseline candidate",
        )
    effective_buckets = (
        num_buckets - 1 if parsed_negative_selection is not None else num_buckets
    )
    boundaries = _build_discretization_boundaries(
        boundary_per_feature,
        boundary_n_sae,
        effective_buckets,
    )
    percentile_table = build_percentile_table(boundary_per_feature, boundary_n_sae)

    source_means = compute_feature_percentile_means(
        feature_dir,
        percentile_table,
        boundary_n_sae,
        reference_feature_ids=reference_feature_ids,
    )
    selected_positive_fids: list[int]
    selected_negative_fids: list[int] = []
    if not is_delta:
        selected_positive_fids = select_top_features_by_mean_percentile(
            source_means,
            sft_stats.per_feature_stats,
            k,
            rank_metric=metric,
        )
    else:
        assert baseline_dir is not None
        assert baseline_stats is not None
        baseline_means = compute_feature_percentile_means(
            baseline_dir,
            percentile_table,
            boundary_n_sae,
            reference_feature_ids=reference_feature_ids,
        )
        positive_delta_scores = compute_delta_feature_scores(
            source_means,
            baseline_means,
            sft_stats.per_feature_stats,
            rank_metric=metric,
        )
        selected_positive_fids = select_top_feature_ids_by_score(positive_delta_scores, k)
        if len(selected_positive_fids) < k:
            logger.warning(
                f"Positive delta pool underfilled: requested {k}, "
                f"available {len(positive_delta_scores)}"
            )
        if parsed_negative_selection is not None:
            negative_delta_scores = compute_delta_feature_scores(
                baseline_means,
                source_means,
                baseline_stats.per_feature_stats,
                rank_metric=metric,
            )
            selected_negative_fids = select_top_feature_ids_by_score(
                negative_delta_scores,
                negative_k,
            )
            if len(selected_negative_fids) < negative_k:
                logger.warning(
                    f"Negative delta pool underfilled: requested {negative_k}, "
                    f"available {len(negative_delta_scores)}"
                )

    # Build feature inputs using SFT stats for selected magnitudes,
    # but pretraining boundaries for discretization.
    mag_key = f"p{int(magnitude_percentile * 100)}"
    enc_fids, enc_mags, _enc_mask = build_feature_inputs_for_selected_ids(
        sft_stats.per_feature_stats,
        boundaries,
        selected_positive_fids,
        "cpu",
        magnitude_percentile=mag_key,
    )

    # Convert (1, 1, K) tensors to plain ints
    fids = enc_fids.squeeze().tolist()
    mags = enc_mags.squeeze().tolist()

    # Handle single-feature case (squeeze collapses to scalar)
    if isinstance(fids, int):
        fids = [fids]
        mags = [mags]
    if not isinstance(fids, list) or any(
        isinstance(fid, bool) or not isinstance(fid, int) for fid in fids
    ):
        raise TypeError("encoder feature ids must narrow to a list of integers")
    if not isinstance(mags, list) or any(
        isinstance(mag, bool) or not isinstance(mag, int) for mag in mags
    ):
        raise TypeError("encoder feature magnitudes must narrow to a list of integers")
    integer_fids = cast(list[int], fids)
    integer_mags = cast(list[int], mags)

    if parsed_selection is not None and top_k is None:
        logger.info(
            "Sampled selection config: "
            f"distribution={parsed_selection.distribution.value}, "
            f"mean={parsed_selection.mean}, std={parsed_selection.std}, "
            f"min_k={parsed_selection.min_k}, max_k={parsed_selection.max_k}, "
            f"seed={seed}"
        )

    features = dict(zip(integer_fids, integer_mags, strict=True))
    if parsed_negative_selection is not None:
        inactive_bucket = num_buckets - 1
        duplicate_negatives = [
            int(fid) for fid in selected_negative_fids if int(fid) in features
        ]
        if duplicate_negatives:
            preview = duplicate_negatives[:10]
            suffix = (
                "" if len(duplicate_negatives) <= 10
                else f" ... ({len(duplicate_negatives)} total)"
            )
            logger.warning(
                "Positive/negative selection overlap is unexpected; positive "
                f"selection wins for duplicate negative ids {preview}{suffix}"
            )
        for fid in selected_negative_fids:
            if int(fid) in features:
                continue
            features[int(fid)] = inactive_bucket

    return features


@hydra.main(
    config_path="../../configs/eval",
    config_name="extract_common_features",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if "feature_percentile" in cfg:
        raise ValueError(
            "feature_percentile has been removed; configure feature_selection "
            "or top_k instead."
        )
    if "rank_by" in cfg:
        raise ValueError("rank_by has been removed; configure rank_metric instead.")

    feature_selection = _coerce_sampling_selection(cfg.feature_selection)
    negative_selection = _coerce_negative_selection(cfg.negative_selection)
    seed = int(cfg.seed)
    rank_metric = FeatureRankMetric.parse(cfg.rank_metric)
    baseline_dir = cfg.baseline_dir
    selection_kind = "delta" if baseline_dir is not None else "absolute"
    features = extract_common_features(
        feature_dir=cfg.feature_dir,
        num_buckets=cfg.num_buckets,
        stats_dir=cfg.stats_dir,
        baseline_dir=baseline_dir,
        magnitude_percentile=cfg.magnitude_percentile,
        top_k=cfg.top_k,
        feature_selection=feature_selection,
        negative_selection=negative_selection,
        seed=seed,
        rank_metric=rank_metric,
    )

    logger.info(f"Extracted {len(features)} features from {cfg.feature_dir}")
    logger.info(
        f"num_buckets={cfg.num_buckets}, "
        f"magnitude_percentile={cfg.magnitude_percentile}, "
        f"rank_metric={rank_metric.value}, "
        f"selection_kind={selection_kind}, seed={seed}"
    )

    # Resolve output path
    output = cfg.output
    if output is None:
        output = auto_output_path(
            cfg.feature_dir,
            cfg.magnitude_percentile,
            feature_selection=feature_selection,
            negative_selection=negative_selection,
            top_k=cfg.top_k,
            seed=seed,
            rank_metric=rank_metric,
            baseline_dir=baseline_dir,
        )
        logger.info(f"Auto-generated output path: {output}")

    # Print the flat {fid: bucket} dict as YAML
    yaml_str = yaml.dump(features, default_flow_style=False, sort_keys=False)
    print(yaml_str)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(features, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Written to {output}")


if __name__ == "__main__":
    main()
