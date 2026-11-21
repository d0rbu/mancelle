"""Stochastic feature sampling and negative (inactive) feature selection.

Provides:
- Distribution: enum of supported sampling distributions (geometric, normal)
- SamplingSelection: config for positive feature sampling (geometric/normal distribution)
- NegativeSamplingSelection: config for negative (inactive) feature sampling
- ThresholdSelection: deterministic threshold-based feature selection
- SamplingBundle: sampling config + precomputed percentile table + seeded RNG
- NegativeBundle: negative sampling config + feature count + seeded RNG
- FeatureSelectionConfig: union of ThresholdSelection | SamplingBundle
- parse_feature_selection: parse positive feature selection config
- parse_negative_selection: parse negative feature selection config
- build_percentile_table: re-exported from percentile_table.py
- magnitude_to_percentile: re-exported from percentile_table.py
- sample_features: stochastically sample k features weighted by percentile score
- sample_features_batched: batch-optimized version of sample_features for multiple segments
- sample_inactive_features: sample k inactive features with uniform selection
- sample_inactive_features_batched: batch-optimized version of sample_inactive_features
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch as t
from beartype import beartype
from loguru import logger

from errors import GradientRankingUnsupportedError, SelfCondTrainingError
from steering.conditional.delta import DeltaSamplingWeights
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.percentile_table import build_percentile_table, magnitude_to_percentile
from steering.conditional.thresholding import ThresholdType
from steering.gradient_overlay import (
    GradientRankingSpec,
    resolve_gradient_spec_from_feature_selection,
)
from steering.refined_types import (
    FiniteNumber,
    FiniteNumberLike,
    NonNegativeFiniteNumber,
    NonNegativeFiniteNumberLike,
    NonNegativeInt,
    NonNegativeIntLike,
    NonNegativeIntListLike,
    PositiveFiniteNumber,
    PositiveFiniteNumberLike,
    PositiveInt,
    PositiveIntLike,
)

MAX_REJECTION_ITERATIONS = 100
MIN_OVERSAMPLE_FACTOR = 3
GUMBEL_EPS = 1e-30
_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT = 4096
SELFCOND_ACTION_RANKING_POLICY = "pooled_fp_logits_v1"

if TYPE_CHECKING:
    _OptionalTensorList = list[t.Tensor | None]
    _TensorList = list[t.Tensor]
    _TensorPair = tuple[t.Tensor, t.Tensor]
    _TensorPairList = list[tuple[t.Tensor, t.Tensor]]
else:
    # Unsubscripted runtime containers avoid beartype's randomized element
    # sampling. Tensor shapes and cross-container lengths remain explicit
    # relationship checks in the owning sampling functions.
    _OptionalTensorList = list
    _TensorList = list
    _TensorPair = tuple
    _TensorPairList = list

_SAMPLING_SELECTION_KEYS = frozenset(
    {
        "distribution",
        "mean",
        "std",
        "min_k",
        "max_k",
        "delta_weights",
        "mode",
        "gradient",
    }
)
_THRESHOLD_SELECTION_KEYS = frozenset(
    {
        "threshold_type",
        "threshold",
    }
)
_NEGATIVE_SELECTION_KEYS = frozenset(
    {
        "distribution",
        "mean",
        "std",
        "min_k",
        "max_k",
    }
)

# Public surface of this module. ``sample_k`` and ``sample_k_batched`` are
# the canonical names for the geometric/normal k-sampling helpers (slice-07
# audit promoted them from ``_sample_k`` / ``_sample_k_batched`` per the
# user's "no deprecated aliases" directive).
__all__ = [
    "SelectedActions",
    "build_percentile_table",
    "magnitude_to_percentile",
    "sample_action_counts",
    "sample_k",
    "sample_k_batched",
    "select_topk_bottoml",
]


class Distribution(Enum):
    geometric = "geometric"
    normal = "normal"


class SamplingMode(Enum):
    weighted = "weighted"
    top_k_percentile = "top_k_percentile"


class _SamplingRelations:
    """Relationships shared by positive and negative sampling policies."""

    distribution: Distribution
    mean: FiniteNumber
    std: PositiveFiniteNumber
    min_k: NonNegativeInt
    max_k: PositiveInt | None

    def _validate_sampling_relations(self) -> None:
        if self.distribution is Distribution.geometric:
            if self.mean <= 0:
                raise ValueError(f"Geometric distribution requires mean > 0, got {self.mean}.")
            if self.std != self.mean:
                raise ValueError(
                    "Geometric distribution std must equal mean, "
                    f"got std={self.std} != mean={self.mean}."
                )
        if self.max_k is not None and self.max_k <= self.min_k:
            raise ValueError(f"max_k must be > min_k, got min_k={self.min_k}, max_k={self.max_k}.")


@dataclass(frozen=True, init=False)
class SamplingSelection(_SamplingRelations):
    distribution: Distribution
    mean: FiniteNumber
    std: PositiveFiniteNumber
    delta_weights: DeltaSamplingWeights | None
    min_k: NonNegativeInt
    max_k: PositiveInt | None
    mode: SamplingMode
    # Presence of a gradient spec switches the ranking key from activation
    # percentiles to precomputed gradient-attribution scores. OFFLINE-ONLY:
    # every live selection path hard-fails (see sample_features*'s
    # ``gradient_rank_scores_supplied`` guard); only
    # scripts.preprocess.precompute_selected_features supports it.
    gradient: GradientRankingSpec | None

    @beartype
    def __init__(
        self,
        distribution: Distribution = Distribution.geometric,
        mean: FiniteNumberLike = 5.0,
        std: PositiveFiniteNumberLike = 5.0,
        delta_weights: DeltaSamplingWeights | None = None,
        min_k: NonNegativeIntLike = 1,
        max_k: PositiveIntLike | None = None,
        mode: SamplingMode = SamplingMode.weighted,
        gradient: GradientRankingSpec | None = None,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "mean", FiniteNumber.parse(float(mean)))
        object.__setattr__(
            self,
            "std",
            PositiveFiniteNumber.parse(float(std)),
        )
        object.__setattr__(self, "delta_weights", delta_weights)
        object.__setattr__(self, "min_k", NonNegativeInt.parse(min_k))
        object.__setattr__(
            self,
            "max_k",
            PositiveInt.parse(max_k) if max_k is not None else None,
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "gradient", gradient)
        self._validate_sampling_relations()


@dataclass(frozen=True, init=False)
class ThresholdSelection:
    """Deterministic threshold-based feature selection."""

    threshold_type: ThresholdType
    threshold: NonNegativeFiniteNumber

    @beartype
    def __init__(
        self,
        threshold_type: ThresholdType,
        threshold: NonNegativeFiniteNumberLike,
    ) -> None:
        object.__setattr__(self, "threshold_type", threshold_type)
        object.__setattr__(
            self,
            "threshold",
            NonNegativeFiniteNumber.parse(float(threshold)),
        )


@dataclass(frozen=True, init=False)
class NegativeSamplingSelection(_SamplingRelations):
    """Stochastic negative (inactive) feature sampling config."""

    distribution: Distribution
    mean: FiniteNumber
    std: PositiveFiniteNumber
    min_k: NonNegativeInt
    max_k: PositiveInt | None

    @beartype
    def __init__(
        self,
        distribution: Distribution = Distribution.geometric,
        mean: FiniteNumberLike = 3.0,
        std: PositiveFiniteNumberLike = 3.0,
        min_k: NonNegativeIntLike = 0,
        max_k: PositiveIntLike | None = None,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "mean", FiniteNumber.parse(float(mean)))
        object.__setattr__(
            self,
            "std",
            PositiveFiniteNumber.parse(float(std)),
        )
        object.__setattr__(self, "min_k", NonNegativeInt.parse(min_k))
        object.__setattr__(
            self,
            "max_k",
            PositiveInt.parse(max_k) if max_k is not None else None,
        )
        self._validate_sampling_relations()


FeatureSelection = ThresholdSelection | SamplingSelection


@beartype
@dataclass
class SamplingBundle:
    """Sampling config + precomputed percentile table + seeded RNG."""

    selection: SamplingSelection
    percentile_table: t.Tensor
    rng: t.Generator


@beartype
@dataclass
class NegativeBundle:
    """Negative sampling config + feature count + seeded RNG.

    ``feature_mask`` is the user blacklist/whitelist (mirrors
    ``pipeline.feature_mask``) passed to the samplers' single ``feature_mask``
    argument.
    """

    selection: NegativeSamplingSelection
    n_sae_features: PositiveIntLike
    rng: t.Generator
    feature_mask: FeatureMask | None = None


FeatureSelectionConfig = ThresholdSelection | SamplingBundle


@dataclass(frozen=True)
class SelectedActions:
    """Per-row positive/negative head rows, padded with ``-1`` to batch maxima."""

    pos_ids: t.Tensor
    neg_ids: t.Tensor
    k: t.Tensor
    l: t.Tensor  # noqa: E741 - mathematical negative-action count


def sample_action_counts(
    positive: SamplingSelection,
    negative: NegativeSamplingSelection,
    rng: t.Generator,
    batch_size: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Draw positive counts first and negative counts second.

    This is the shared training/inference cardinality primitive. Both policies
    use exclusive ``max_k`` bounds, exactly like :func:`sample_k_batched`.
    """

    positive_counts = sample_k_batched(
        positive.distribution,
        positive.mean,
        positive.std,
        rng,
        n=batch_size,
        min_k=positive.min_k,
        max_k=positive.max_k,
    )
    negative_counts = sample_k_batched(
        negative.distribution,
        negative.mean,
        negative.std,
        rng,
        n=batch_size,
        min_k=negative.min_k,
        max_k=negative.max_k,
    )
    return positive_counts, negative_counts


def select_topk_bottoml(
    scores: t.Tensor,
    k: t.Tensor,
    l: t.Tensor,  # noqa: E741 - mathematical negative-action count
) -> SelectedActions:
    """Select stable score-descending positives and disjoint ascending negatives.

    Exact ties resolve by input row order. Callers align row order with their
    canonical original-feature-ID vocabulary before invoking this primitive.
    """

    if scores.ndim != 2:
        raise SelfCondTrainingError(f"scores must have shape (B, V), got {tuple(scores.shape)}")
    batch_size, n_features = scores.shape
    if tuple(k.shape) != (batch_size,) or tuple(l.shape) != (batch_size,):
        raise SelfCondTrainingError(
            f"k and l must both have shape {(batch_size,)}, got "
            f"{tuple(k.shape)} and {tuple(l.shape)}"
        )
    if k.dtype != t.int64 or l.dtype != t.int64:
        raise SelfCondTrainingError("k and l must be int64 tensors")
    if bool(((k < 0) | (l < 0) | (k + l > n_features)).any()):
        raise SelfCondTrainingError("each row requires nonnegative k/l with k+l <= n_sae_features")

    descending = (-scores).argsort(dim=1, stable=True)
    descending_rank = t.empty_like(descending)
    descending_rank.scatter_(
        1,
        descending,
        t.arange(n_features, device=scores.device).unsqueeze(0).expand(batch_size, -1),
    )
    positive_mask = descending_rank < k.unsqueeze(1)
    max_positive = int(k.max().item()) if batch_size else 0
    pos_ids = descending[:, :max_positive]
    pos_columns = t.arange(max_positive, device=scores.device).unsqueeze(0)
    pos_ids = t.where(pos_columns < k.unsqueeze(1), pos_ids, -t.ones_like(pos_ids))

    negative_scores = scores.masked_fill(positive_mask, float("inf"))
    ascending = negative_scores.argsort(dim=1, stable=True)
    max_negative = int(l.max().item()) if batch_size else 0
    neg_ids = ascending[:, :max_negative]
    neg_columns = t.arange(max_negative, device=scores.device).unsqueeze(0)
    neg_ids = t.where(neg_columns < l.unsqueeze(1), neg_ids, -t.ones_like(neg_ids))
    return SelectedActions(pos_ids=pos_ids, neg_ids=neg_ids, k=k, l=l)


def _resolve_distribution_config(cfg: dict) -> tuple[Distribution, Any, Any]:
    """Resolve distribution/mean/std keys from a config dict.

    The beartyped owning selection class validates the numeric domains and its
    shared relation method retains the distribution-specific constraint.
    """
    if "distribution" not in cfg:
        raise ValueError("Config must contain 'distribution'.")
    raw = cfg["distribution"]
    try:
        dist = Distribution(raw)
    except ValueError:
        supported = sorted(d.value for d in Distribution)
        raise ValueError(f"Unsupported distribution '{raw}'. Supported: {supported}.") from None
    if "mean" not in cfg:
        raise ValueError(
            f"'{dist.value}' distribution requires 'mean' but it is missing from the config."
        )
    mean = cfg["mean"]
    has_std = "std" in cfg

    if dist == Distribution.geometric:
        # With omitted std, preserve the legacy mean-specific relationship
        # error instead of letting a derived non-positive std mask it at the
        # beartype boundary.
        std = (
            cfg["std"]
            if has_std
            else mean
            if isinstance(mean, PositiveFiniteNumber)
            else PositiveFiniteNumber.parse(1.0)
        )
    else:
        # normal
        if not has_std:
            raise ValueError(
                "Normal distribution requires 'std' but it is missing from the config."
            )
        std = cfg["std"]

    return dist, mean, std


def _reject_unknown_selection_keys(
    cfg: dict,
    *,
    variant: str,
    allowed: frozenset[str],
) -> None:
    unknown = sorted(str(key) for key in cfg if key not in allowed)
    if not unknown:
        return
    raise ValueError(
        f"{variant} feature_selection config contains unknown or incompatible "
        f"keys {unknown}. Allowed keys: {sorted(allowed)}."
    )


def parse_feature_selection(cfg: dict) -> FeatureSelection:
    """Parse and validate the feature_selection config block.

    Discriminates on key presence:
    - 'distribution' → SamplingSelection
    - 'threshold_type' → ThresholdSelection
    Both or neither → error.

    Each arm is a closed schema: unknown keys and keys belonging to the other
    arm are rejected even when their value is null. Shape, enum, and relation
    errors raise ``ValueError``; beartype rejects scalar-domain violations at
    the owning constructor boundary.
    """
    has_distribution = "distribution" in cfg
    has_threshold_type = "threshold_type" in cfg

    if has_distribution and has_threshold_type:
        raise ValueError(
            "feature_selection config keys are mutually exclusive: "
            "cannot specify both 'distribution' and 'threshold_type'."
        )

    if not has_distribution and not has_threshold_type:
        raise ValueError(
            "feature_selection config must contain either 'distribution' (for sampling) "
            "or 'threshold_type' (for threshold-based selection)."
        )

    if has_distribution:
        _reject_unknown_selection_keys(
            cfg,
            variant="Sampling",
            allowed=_SAMPLING_SELECTION_KEYS,
        )
        dist, mean, std = _resolve_distribution_config(cfg)
        delta_weights: DeltaSamplingWeights | None = None
        if "delta_weights" in cfg and cfg["delta_weights"] is not None:
            raw_dw = cfg["delta_weights"]
            try:
                delta_weights = DeltaSamplingWeights(raw_dw)
            except ValueError:
                supported = sorted(d.value for d in DeltaSamplingWeights)
                raise ValueError(
                    f"Unsupported delta_weights '{raw_dw}'. Supported: {supported}."
                ) from None
        mode = SamplingMode.weighted
        if "mode" in cfg:
            raw_mode = cfg["mode"]
            try:
                mode = SamplingMode(raw_mode)
            except ValueError:
                supported = sorted(m.value for m in SamplingMode)
                raise ValueError(
                    f"Unsupported sampling mode '{raw_mode}'. Supported: {supported}."
                ) from None
        gradient = resolve_gradient_spec_from_feature_selection(cfg)
        if gradient is not None:
            if mode != SamplingMode.top_k_percentile:
                raise ValueError(
                    "feature_selection.gradient requires mode=top_k_percentile "
                    f"(gradient scores are a ranking key), got mode={mode.value!r}."
                )
            if delta_weights is not None:
                raise ValueError(
                    "feature_selection.gradient cannot be combined with "
                    "delta_weights (delta re-weighting has no gradient channel)."
                )
        return SamplingSelection(
            distribution=dist,
            mean=mean,
            std=std,
            delta_weights=delta_weights,
            min_k=cfg.get("min_k", NonNegativeInt.parse(1)),
            max_k=cfg.get("max_k"),
            mode=mode,
            gradient=gradient,
        )

    # has_threshold_type branch
    if "gradient" in cfg and cfg["gradient"] is not None:
        raise ValueError(
            "feature_selection.gradient is only supported with the sampling "
            "config shape (distribution/mode=top_k_percentile), not with "
            "threshold_type selection."
        )
    _reject_unknown_selection_keys(
        cfg,
        variant="Threshold",
        allowed=_THRESHOLD_SELECTION_KEYS,
    )
    if "threshold" not in cfg:
        raise ValueError(
            "ThresholdSelection requires 'threshold' but it is missing from the config."
        )
    raw_threshold_type = cfg["threshold_type"]
    try:
        threshold_type = ThresholdType(raw_threshold_type)
    except ValueError:
        supported = sorted(t.value for t in ThresholdType)
        raise ValueError(
            f"Unsupported threshold_type '{raw_threshold_type}'. Supported: {supported}."
        ) from None
    return ThresholdSelection(
        threshold_type=threshold_type,
        threshold=cfg["threshold"],
    )


def parse_negative_selection(cfg: dict | None) -> NegativeSamplingSelection | None:
    """Parse the negative_selection config block.

    Returns None if cfg is None (negative sampling disabled).
    Otherwise validates distribution/mean/std and returns NegativeSamplingSelection.
    """
    if cfg is None:
        return None
    unknown = sorted(str(key) for key in cfg if key not in _NEGATIVE_SELECTION_KEYS)
    if unknown:
        raise ValueError(
            "negative_selection config contains unknown keys "
            f"{unknown}. Allowed keys: {sorted(_NEGATIVE_SELECTION_KEYS)}."
        )
    dist, mean, std = _resolve_distribution_config(cfg)
    return NegativeSamplingSelection(
        distribution=dist,
        mean=mean,
        std=std,
        min_k=cfg.get("min_k", NonNegativeInt.parse(0)),
        max_k=cfg.get("max_k"),
    )


@beartype
def sample_k(
    distribution: Distribution,
    mean: FiniteNumberLike,
    std: PositiveFiniteNumberLike,
    rng: t.Generator,
    min_k: NonNegativeIntLike = 0,
    max_k: PositiveIntLike | None = None,
) -> NonNegativeIntLike:
    """Sample a non-negative integer k from the configured distribution.

    Args:
        distribution: Distribution.geometric or Distribution.normal.
        mean: Distribution mean (must be > 0 for geometric).
        std: Distribution std (equals mean for geometric; > 0 for normal).
        rng: Seeded torch.Generator for deterministic sampling.
        min_k: Minimum value (inclusive) for the sample. Default 0.
        max_k: Maximum value (exclusive) for the sample. Default None.
            For geometric with both ``min_k`` / ``max_k``: returns
            ``(Geometric(p) - 1) % R + min_k`` where ``R = max_k - min_k``.
            Mathematically equivalent to a truncated geometric on
            ``{min_k, ..., max_k - 1}``::

                P(out = i) = (1 - p)^{i - min_k} * p / (1 - (1 - p)^R)

            For geometric with only ``min_k``: sample + min_k (offset, no mod).
            For normal: clamp to [min_k, max_k - 1].

    Returns:
        Non-negative integer sample count in ``[min_k, max_k)``
        (or ``[min_k, infinity)`` when ``max_k`` is None).
    """
    if distribution == Distribution.geometric:
        p = 1.0 / (1.0 + mean)
        sample = t.empty(1, dtype=t.float64)
        sample.geometric_(p, generator=rng)
        k = max(0, int(sample.item()) - 1)
        if max_k is not None:
            k = (k % (max_k - min_k)) + min_k
        elif min_k > 0:
            k = k + min_k
        return k
    if distribution == Distribution.normal:
        k_sample = t.normal(
            mean=t.tensor(mean, dtype=t.float64),
            std=t.tensor(std, dtype=t.float64),
            generator=rng,
        )
        k = int(round(k_sample.item()))
        k = max(min_k, k)
        if max_k is not None:
            k = min(max_k - 1, k)
        return k
    raise ValueError(f"Unsupported distribution '{distribution}'.")


def _assert_gradient_scores_supplied(config: SamplingSelection, supplied: bool) -> None:
    if config.gradient is not None and not supplied:
        raise GradientRankingUnsupportedError(
            "feature_selection.gradient (gradient-attribution ranking) is "
            "configured, but this code path performs LIVE feature selection "
            "and has no gradient scores. Gradient ranking is offline-only: "
            "precompute sel/ shards via "
            "scripts.preprocess.precompute_selected_features (after "
            "precompute_gradient_attribution) "
            "and train with use_precomputed_batches=true."
        )


def assert_no_gradient_ranking(selection, context: str) -> None:
    """Fail fast when a live-selection entry point receives a gradient-ranked
    feature selection. ``selection`` may be any FeatureSelection (threshold
    selections never carry a gradient spec)."""
    if getattr(selection, "gradient", None) is not None:
        raise GradientRankingUnsupportedError(
            f"{context} performs live feature selection and does not support "
            "feature_selection.gradient (gradient-attribution ranking is "
            "offline-only — precompute sel/ shards via "
            "scripts.preprocess.precompute_selected_features)."
        )


@beartype
def sample_features(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    percentile_table: t.Tensor,
    config: SamplingSelection,
    rng: t.Generator,
    override_weights: t.Tensor | None = None,
    gradient_rank_scores_supplied: bool = False,
) -> _TensorPair:
    """Stochastically sample k features weighted by activation percentile.

    When config.mode is SamplingMode.top_k_percentile, selects the top-k
    features by percentile (deterministic selection, stochastic cardinality).
    When config.mode is SamplingMode.weighted (default), uses multinomial
    sampling weighted by percentile (stochastic selection and cardinality).

    ``gradient_rank_scores_supplied`` is the central offline-only guard for
    gradient-attribution ranking: when ``config.gradient`` is set, ONLY the
    selection precompute (which loads the grad overlay and passes the scores
    as override weights) may call this — every live caller that merely
    threads the parsed config raises.
    """
    _assert_gradient_scores_supplied(config, gradient_rank_scores_supplied)
    num_active = feature_ids.shape[0]
    if num_active == 0:
        return feature_ids, feature_mags

    if override_weights is not None and override_weights.shape[0] != num_active:
        raise ValueError(
            f"override_weights length {override_weights.shape[0]} != "
            f"feature_ids length {num_active}"
        )

    # Step 1: sample k from configured distribution
    k = sample_k(
        config.distribution, config.mean, config.std, rng, min_k=config.min_k, max_k=config.max_k
    )

    # Step 2: dispatch on mode
    if config.mode == SamplingMode.top_k_percentile:
        return select_top_k_by_percentile(
            feature_ids,
            feature_mags,
            percentile_table,
            k,
            override_weights=override_weights,
            rng=rng,
        )

    # Weighted mode (original behavior)
    # Step 3: compute weights
    if override_weights is not None:
        weights = override_weights.float()
    else:
        percentiles = magnitude_to_percentile(feature_ids, feature_mags, percentile_table)
        weights = percentiles.float()

    # Step 4: clamp k to [0, num_active]
    k = min(k, num_active)

    if k == 0:
        empty_ids = feature_ids[:0]
        empty_mags = feature_mags[:0]
        return empty_ids, empty_mags

    # Step 5: multinomial sampling without replacement using weights
    if weights.sum() == 0:
        weights = t.ones_like(weights)
    selected_idx = t.multinomial(weights, num_samples=k, replacement=False, generator=rng)

    # Step 6: gather selected features (original magnitudes)
    return feature_ids[selected_idx], feature_mags[selected_idx]


def _rand_like_scores(scores: t.Tensor, rng: t.Generator | None) -> t.Tensor:
    """Return reproducible random tie-break scores with the same shape."""
    if rng is None:
        gen_device = scores.device if scores.device.type == "cuda" else t.device("cpu")
        rng = t.Generator(device=gen_device)
        rng.manual_seed(0)
    if rng.device != scores.device:
        return t.rand(
            scores.shape,
            generator=rng,
            device=rng.device,
            dtype=t.float64,
        ).to(scores.device)
    return t.rand(
        scores.shape,
        generator=rng,
        device=scores.device,
        dtype=t.float64,
    )


@beartype
def _top_k_order(
    scores: t.Tensor,
    feature_ids: t.Tensor,
    rng: t.Generator | None,
    *,
    k: NonNegativeIntLike | None = None,
) -> t.Tensor:
    """Order the highest scores with deterministic or seeded tie-breaking.

    When ``rng`` is ``None``, exact score ties resolve by ascending feature id.
    When ``rng`` is provided, exact ties resolve by seeded random order.

    ``k=None`` preserves the full reference sort. For ``0 < k < N`` with
    finite scores, a top-k cutoff first narrows the two stable sorts to every
    item whose score is at least the kth score. Including *all* cutoff ties is
    what makes the prefix exactly equal to the full reference ordering. The
    random path still draws one float64 tie score for every input before
    narrowing, preserving generator state bit-for-bit.
    """
    tie_scores: t.Tensor | None = None
    if rng is None:
        pass
    else:
        tie_scores = _rand_like_scores(scores, rng)

    n_scores = int(scores.numel())
    use_bounded_sort = k is not None and 0 < k < n_scores and bool(t.isfinite(scores).all())
    if use_bounded_sort:
        assert k is not None
        cutoff = t.topk(scores, k, largest=True, sorted=False).values.min()
        eligible = (scores >= cutoff).nonzero(as_tuple=True)[0]
    else:
        eligible = t.arange(n_scores, device=scores.device)

    eligible_scores = scores[eligible]
    if tie_scores is None:
        eligible_fids = feature_ids.to(scores.device)[eligible]
        tie_order = eligible_fids.argsort(stable=True)
    else:
        tie_order = (-tie_scores[eligible]).argsort(stable=True)
    reordered_scores = eligible_scores[tie_order]
    score_order = (-reordered_scores).argsort(stable=True)
    ordered = eligible[tie_order[score_order]]
    return ordered[:k] if use_bounded_sort else ordered


@beartype
def select_top_k_by_percentile(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    percentile_table: t.Tensor,
    k: NonNegativeIntLike,
    override_weights: t.Tensor | None = None,
    rng: t.Generator | None = None,
) -> _TensorPair:
    """Select the top-k features by percentile score (deterministic).

    For each feature, computes its percentile score via the percentile table,
    then returns the top-k features sorted by percentile descending. Exact ties
    are broken by ascending feature id when ``rng`` is ``None`` and by seeded
    random order when ``rng`` is provided.

    Args:
        feature_ids: [N] int64 tensor of feature indices.
        feature_mags: [N] float tensor of magnitudes.
        percentile_table: [n_features, 99] float64 lookup table.
        k: Number of features to select.
        override_weights: optional [N] float tensor to use as ranking scores
            instead of computing percentiles from the table. Used for
            precomputed percentiles or delta-weighted scores.
        rng: Optional seeded generator for deterministic random tie-breaking.
            If omitted, ties are broken by ascending feature id.

    Returns:
        (selected_ids, selected_mags) tuple, sorted by percentile descending.
        Length is min(k, N).
    """
    num_active = feature_ids.shape[0]
    if num_active == 0 or k <= 0:
        return feature_ids[:0], feature_mags[:0]

    k = min(k, num_active)

    # Compute ranking scores
    if override_weights is not None:
        scores = override_weights.to(t.float64)
    else:
        scores = magnitude_to_percentile(feature_ids, feature_mags, percentile_table)

    final_order = _top_k_order(scores, feature_ids, rng, k=k)

    selected = final_order[:k]
    return feature_ids[selected], feature_mags[selected]


@beartype
def select_top_k_by_percentile_batched(
    all_feature_ids: t.Tensor,
    all_feature_mags: t.Tensor,
    segment_sizes: NonNegativeIntListLike,
    percentile_table: t.Tensor,
    ks: t.Tensor,
    precomputed_percentiles: t.Tensor | None = None,
    override_weights_list: _OptionalTensorList | None = None,
    rng: t.Generator | None = None,
) -> _TensorPairList:
    """Batch-optimized top-k-by-percentile selection for multiple segments.

    Computes magnitude_to_percentile ONCE for all segments, then does
    per-segment deterministic top-k selection.

    Args:
        all_feature_ids: [total_N] concatenated feature IDs from all segments.
        all_feature_mags: [total_N] concatenated magnitudes from all segments.
        segment_sizes: list of per-segment feature counts (sum = total_N).
        percentile_table: [n_features, 99] float64 lookup table.
        ks: [n_segments] int64 tensor of per-segment k values.
        precomputed_percentiles: optional [total_N] float tensor of precomputed
            percentile weights. When provided, skips magnitude_to_percentile.
        override_weights_list: optional per-segment override weights.
        rng: Optional seeded generator for deterministic random tie-breaking.
            If omitted, ties are broken by ascending feature id.

    Returns:
        List of (selected_ids, selected_mags) tuples, one per segment.
    """
    n_segments = len(segment_sizes)
    total_n = all_feature_ids.shape[0]

    if total_n == 0:
        return [
            (t.tensor([], dtype=t.int64), t.tensor([], dtype=t.float32)) for _ in range(n_segments)
        ]

    # Compute percentiles once for all elements
    if precomputed_percentiles is not None:
        all_scores = precomputed_percentiles.to(t.float64)
    else:
        # Determine which segments need percentile computation vs have overrides
        needs_pct = [True] * n_segments
        if override_weights_list is not None:
            for i, ow in enumerate(override_weights_list):
                if ow is not None:
                    needs_pct[i] = False

        all_scores = t.empty(total_n, dtype=t.float64)
        pct_ranges = []
        offset = 0
        for i, sz in enumerate(segment_sizes):
            if needs_pct[i] and sz > 0:
                pct_ranges.append((offset, offset + sz))
            offset += sz

        if pct_ranges:
            pct_indices = t.cat([t.arange(s, e) for s, e in pct_ranges])
            pct_vals = magnitude_to_percentile(
                all_feature_ids[pct_indices],
                all_feature_mags[pct_indices],
                percentile_table,
            )
            all_scores[pct_indices] = pct_vals

    # Apply override weights where provided
    if override_weights_list is not None:
        offset = 0
        for i, sz in enumerate(segment_sizes):
            override_weights = override_weights_list[i]
            if sz > 0 and override_weights is not None:
                all_scores[offset : offset + sz] = override_weights.to(t.float64)
            offset += sz

    max_k_val = int(ks.max().item()) if n_segments > 0 else 0

    if max(segment_sizes, default=0) == 0 or max_k_val == 0:
        empty_i = t.tensor([], dtype=t.int64)
        empty_f = t.tensor([], dtype=t.float32)
        # Independent tensors per slot (no aliasing): a shared object would let
        # a future in-place edit of one segment corrupt every other segment.
        return [(empty_i.clone(), empty_f.clone()) for _ in range(n_segments)]

    results: list[tuple[t.Tensor, t.Tensor]] = []
    offset = 0
    for i, sz in enumerate(segment_sizes):
        k = min(int(ks[i].item()), sz)
        if sz == 0 or k <= 0:
            results.append((all_feature_ids[:0], all_feature_mags[:0]))
            offset += sz
            continue
        seg_scores = all_scores[offset : offset + sz]
        seg_fids = all_feature_ids[offset : offset + sz]
        seg_mags = all_feature_mags[offset : offset + sz]
        order = _top_k_order(seg_scores, seg_fids, rng, k=k)[:k]
        results.append((seg_fids[order], seg_mags[order]))
        offset += sz

    return results


@beartype
def sample_features_batched(
    all_feature_ids: t.Tensor,
    all_feature_mags: t.Tensor,
    segment_sizes: NonNegativeIntListLike,
    percentile_table: t.Tensor,
    config: SamplingSelection,
    rng: t.Generator,
    override_weights_list: _OptionalTensorList | None = None,
    precomputed_percentiles: t.Tensor | None = None,
    gradient_rank_scores_supplied: bool = False,
) -> _TensorPairList:
    """Batch-optimized version of sample_features for multiple segments.

    Computes magnitude_to_percentile ONCE for all segments concatenated,
    then splits and does per-segment multinomial sampling (which is cheap).

    Args:
        all_feature_ids: [total_N] concatenated feature IDs from all segments.
        all_feature_mags: [total_N] concatenated magnitudes from all segments.
        segment_sizes: list of per-segment feature counts (sum = total_N).
        percentile_table: [n_features, 99] float64 lookup table.
        config: Sampling configuration.
        rng: Seeded torch.Generator.
        override_weights_list: optional per-segment override weights (same
            length as segment_sizes). If provided, percentile computation is
            skipped for segments with non-None overrides.
        precomputed_percentiles: optional [total_N] float tensor of precomputed
            percentile weights (e.g. decoded from uint8 shard data). When provided,
            skips the expensive magnitude_to_percentile call entirely for segments
            that don't have override weights.

    Returns:
        List of (sampled_ids, sampled_mags) tuples, one per segment.
    """
    _assert_gradient_scores_supplied(config, gradient_rank_scores_supplied)
    n_segments = len(segment_sizes)
    total_n = all_feature_ids.shape[0]

    if total_n == 0:
        return [
            (t.tensor([], dtype=t.int64), t.tensor([], dtype=t.float32)) for _ in range(n_segments)
        ]

    # Use precomputed percentiles if available, otherwise compute from table
    if precomputed_percentiles is not None:
        all_percentiles = precomputed_percentiles.to(t.float64)
    else:
        # Determine which segments need percentile computation vs have overrides
        needs_pct_mask = [True] * n_segments
        if override_weights_list is not None:
            for i, ow in enumerate(override_weights_list):
                if ow is not None:
                    needs_pct_mask[i] = False

        # Compute which elements need percentile lookup
        needs_pct_elements = []
        offset = 0
        for i, sz in enumerate(segment_sizes):
            if needs_pct_mask[i] and sz > 0:
                needs_pct_elements.append((offset, offset + sz))
            offset += sz

        # Batch percentile computation for all elements that need it
        all_percentiles = t.empty(total_n, dtype=t.float64)
        if needs_pct_elements:
            pct_indices = []
            for start, end in needs_pct_elements:
                pct_indices.append(t.arange(start, end))
            pct_idx = t.cat(pct_indices)

            pct_fids = all_feature_ids[pct_idx]
            pct_mags = all_feature_mags[pct_idx]
            pct_vals = magnitude_to_percentile(pct_fids, pct_mags, percentile_table)
            all_percentiles[pct_idx] = pct_vals

    # Build per-segment weights (resolve overrides)
    all_weights = all_percentiles.float()
    if override_weights_list is not None:
        offset = 0
        for i, sz in enumerate(segment_sizes):
            override_weights = override_weights_list[i]
            if sz > 0 and override_weights is not None:
                all_weights[offset : offset + sz] = override_weights.float()
            offset += sz

    # Batch sample all k values
    sizes_t = t.tensor(segment_sizes, dtype=t.int64)
    all_ks = sample_k_batched(
        config.distribution,
        config.mean,
        config.std,
        rng,
        n_segments,
        min_k=config.min_k,
        max_k=config.max_k,
    )
    # Clamp k to segment size
    all_ks = t.minimum(all_ks, sizes_t)

    max_k_val = int(all_ks.max().item()) if n_segments > 0 else 0
    if max_k_val == 0:
        empty = t.tensor([], dtype=t.int64)
        empty_f = t.tensor([], dtype=t.float32)
        # Independent tensors per slot (no aliasing).
        return [(empty.clone(), empty_f.clone()) for _ in range(n_segments)]

    # Dispatch on mode: top-k-by-percentile or weighted Gumbel-max
    if config.mode == SamplingMode.top_k_percentile:
        # Build float64 scores with overrides applied (avoid float32 round-trip
        # that all_weights introduces — top-k tiebreaking benefits from precision).
        topk_scores = all_percentiles.clone()
        if override_weights_list is not None:
            offset = 0
            for i, sz in enumerate(segment_sizes):
                override_weights = override_weights_list[i]
                if sz > 0 and override_weights is not None:
                    topk_scores[offset : offset + sz] = override_weights.to(t.float64)
                offset += sz
        return select_top_k_by_percentile_batched(
            all_feature_ids,
            all_feature_mags,
            segment_sizes,
            percentile_table,
            all_ks,
            precomputed_percentiles=topk_scores,
            rng=rng,
        )

    # Weighted mode: Gumbel-max trick per segment
    results: list[tuple[t.Tensor, t.Tensor]] = []
    offset = 0
    for i, sz in enumerate(segment_sizes):
        seg_fids = all_feature_ids[offset : offset + sz]
        seg_mags = all_feature_mags[offset : offset + sz]

        k = int(all_ks[i].item())
        if k <= 0 or sz == 0:
            results.append((seg_fids[:0], seg_mags[:0]))
            offset += sz
            continue

        weights = all_weights[offset : offset + sz]
        if weights.sum() == 0:
            weights = t.ones(sz, dtype=t.float32)

        # Gumbel-max: add Gumbel noise to log-weights, top-k selects
        log_w = t.log(weights + GUMBEL_EPS)
        gumbel = -t.log(-t.log(t.rand(sz, generator=rng) + GUMBEL_EPS) + GUMBEL_EPS)
        _, selected_idx = (log_w + gumbel).topk(k)
        results.append((seg_fids[selected_idx], seg_mags[selected_idx]))

        offset += sz

    return results


@beartype
def sample_inactive_features(
    active_feature_ids: t.Tensor,
    n_sae_features: PositiveIntLike,
    config: NegativeSamplingSelection,
    rng: t.Generator,
    *,
    feature_mask: FeatureMask | None = None,
) -> t.Tensor:
    """Sample inactive feature IDs uniformly, with count from distribution.

    Selects features that are NOT in active_feature_ids (the raw features from
    extraction, before any positive selection). Uses vectorized rejection
    sampling with adaptive oversampling for efficiency.

    Args:
        active_feature_ids: [N] int64 tensor of active feature IDs to exclude.
        n_sae_features: Total number of SAE features.
        config: Negative sampling configuration (distribution, mean, std).
        rng: Seeded torch.Generator for deterministic sampling.
        feature_mask: Optional feature blacklist/whitelist restricting the
            eligible pool. When None, behavior is bit-identical to the legacy
            (no-mask) path (I5).

    Returns:
        [k] int64 tensor of sampled inactive feature IDs, unique and disjoint
        from active_feature_ids (and from any masked-out ids when feature_mask
        is set).
    """
    if feature_mask is not None:
        sparse_result = _try_sample_masked_inactive_sparse(
            active_ids=active_feature_ids,
            n_sae_features=n_sae_features,
            config=config,
            rng=rng,
            feature_mask=feature_mask,
        )
        if sparse_result is not None:
            return sparse_result

    # I5: when feature_mask is None, use the legacy pool_size computation
    # (n_sae_features - len(active_feature_ids)) rather than (~mask).sum() so
    # that behavior is byte-identical even if active_feature_ids has duplicates
    # (in which case the two forms would differ). When feature_mask is set we
    # build its allowed universe first and derive pool_size from it. Built with
    # no RNG, so sample_k below stays the first RNG draw (I5).
    effective_allowed = (
        feature_mask.allowed_universe_mask(n_sae_features) if feature_mask is not None else None
    )

    if effective_allowed is None:
        pool_size = n_sae_features - len(active_feature_ids)
        if pool_size <= 0:
            return t.tensor([], dtype=t.int64)

        # Sample k and clamp to available pool
        k = sample_k(
            config.distribution,
            config.mean,
            config.std,
            rng,
            min_k=config.min_k,
            max_k=config.max_k,
        )
        k = min(k, pool_size)

        if k <= 0:
            return t.tensor([], dtype=t.int64)

        # Build exclusion mask (active ids only — legacy path)
        mask = t.zeros(n_sae_features, dtype=t.bool)
        if len(active_feature_ids) > 0:
            mask[active_feature_ids.long()] = True
    else:
        # Masked path: build combined exclusion mask first, then derive pool_size.
        # t.zeros / boolean ops consume no RNG, so the sample_k call below
        # remains the first RNG consumption point — identical to the legacy path.
        mask = t.zeros(n_sae_features, dtype=t.bool)
        if len(active_feature_ids) > 0:
            mask[active_feature_ids.long()] = True
        mask |= ~effective_allowed

        pool_size = int((~mask).sum().item())
        if pool_size <= 0:
            return t.tensor([], dtype=t.int64)

        # Sample k and clamp to available pool
        k = sample_k(
            config.distribution,
            config.mean,
            config.std,
            rng,
            min_k=config.min_k,
            max_k=config.max_k,
        )
        k = min(k, pool_size)

        if k <= 0:
            return t.tensor([], dtype=t.int64)

    # Vectorized rejection sampling
    collected: list[t.Tensor] = []
    remaining = k
    # Adaptive factor: higher when pool is small relative to n_sae_features
    factor = max(MIN_OVERSAMPLE_FACTOR, n_sae_features // pool_size + 1)

    for _ in range(MAX_REJECTION_ITERATIONS):  # safety limit
        cands = t.randint(0, n_sae_features, (factor * remaining,), generator=rng)
        valid = cands[~mask[cands]]
        unique_valid = t.unique(valid)
        take = min(len(unique_valid), remaining)
        if take > 0:
            selected = unique_valid[:take]
            collected.append(selected)
            mask[selected] = True
            remaining -= take
        if remaining <= 0:
            break
    else:
        logger.error(
            f"Vectorized rejection sampling did not converge in {MAX_REJECTION_ITERATIONS} iterations "
            f"(remaining={remaining}, pool_size={pool_size}, n_sae_features={n_sae_features}). "
            f"Falling back to enumerate-and-permute."
        )
        # Fallback: enumerate eligible and permute
        eligible = t.nonzero(~mask, as_tuple=False).squeeze(1)
        perm = t.randperm(len(eligible), generator=rng)[:remaining]
        collected.append(eligible[perm])

    return t.cat(collected) if collected else t.tensor([], dtype=t.int64)


def _first_unique_candidate_mask(candidates: t.Tensor) -> t.Tensor:
    """Return True at each row's first occurrence of every candidate ID.

    Negative sampling consumes candidates in RNG draw order, so deduplication
    must keep the first drawn copy of a repeated ID. ``sort(stable=True)`` is
    intentional: equal candidate IDs must preserve original column order before
    duplicate marking, otherwise the kept copy can depend on backend tie order.
    """
    sorted_cands, sort_idx = candidates.sort(dim=1, stable=True)
    is_dup_sorted = t.zeros_like(candidates, dtype=t.bool)
    is_dup_sorted[:, 1:] = sorted_cands[:, 1:] == sorted_cands[:, :-1]
    is_dup = t.zeros_like(is_dup_sorted)
    is_dup.scatter_(1, sort_idx, is_dup_sorted)
    return ~is_dup


def _sorted_ids_contain(sorted_ids: t.Tensor, queries: t.Tensor) -> t.Tensor:
    """Membership in sorted unique int64 IDs without a full-width bitmap."""
    if sorted_ids.numel() == 0:
        return t.zeros_like(queries, dtype=t.bool)
    positions = t.searchsorted(sorted_ids, queries)
    in_range = positions < sorted_ids.numel()
    safe_positions = positions.clamp(max=sorted_ids.numel() - 1)
    return in_range & (sorted_ids[safe_positions] == queries)


def _require_well_formed_batched_negative_row(
    *,
    row_index: int,
    sampled_ids: t.Tensor,
    active_ids: t.Tensor,
    n_sae_features: int,
    config: NegativeSamplingSelection,
    feature_mask: FeatureMask | None,
    pool_size: int,
) -> None:
    """Fail loudly if a final non-empty batched negative row is malformed."""
    if sampled_ids.numel() == 0:
        return
    if sampled_ids.unique().numel() != sampled_ids.numel():
        raise ValueError(
            f"Batched negative sampling row {row_index} violated uniqueness "
            "invariant: ids are not unique."
        )
    if bool(((sampled_ids < 0) | (sampled_ids >= n_sae_features)).any()):
        raise ValueError(
            f"Batched negative sampling row {row_index} violated id range "
            f"invariant: ids must be within [0, {n_sae_features})."
        )
    if feature_mask is not None and not bool(feature_mask.keep_mask(sampled_ids).all()):
        raise ValueError(
            f"Batched negative sampling row {row_index} violated feature-mask "
            "eligibility invariant: ids must be allowed by the feature mask."
        )
    if active_ids.numel() > 0 and bool(t.isin(sampled_ids, active_ids.long()).any()):
        raise ValueError(
            f"Batched negative sampling row {row_index} violated active-id "
            "disjointness invariant: sampled ids overlap active ids."
        )
    maximum_count = pool_size if config.max_k is None else min(config.max_k, pool_size)
    if sampled_ids.numel() > maximum_count:
        bound_name = "pool_size" if config.max_k is None else "min(config.max_k, pool_size)"
        raise ValueError(
            f"Batched negative sampling row {row_index} violated maximum count "
            f"invariant: selected count={sampled_ids.numel()} exceeds "
            f"{bound_name}={maximum_count}."
        )


def _sparse_masked_active_ids(
    active_ids: t.Tensor,
    n_sae_features: int,
) -> t.Tensor | None:
    """Canonical active IDs for the narrow CPU sparse path, else ``None``."""
    if (
        n_sae_features <= 0
        or active_ids.ndim != 1
        or active_ids.device.type != "cpu"
        or t.get_default_device().type != "cpu"
        or active_ids.dtype is t.bool
        or active_ids.is_floating_point()
        or active_ids.is_complex()
        or active_ids.numel() > _SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT
    ):
        return None
    active = t.unique(active_ids.to(t.int64), sorted=True)
    if active.numel() > 0 and (
        int(active[0].item()) < 0 or int(active[-1].item()) >= n_sae_features
    ):
        return None
    return active


def _try_sample_masked_inactive_sparse(
    *,
    active_ids: t.Tensor,
    n_sae_features: int,
    config: NegativeSamplingSelection,
    rng: t.Generator,
    feature_mask: FeatureMask,
) -> t.Tensor | None:
    """Exact masked rejection sampler without a mutable SAE-width bitmap."""
    active = _sparse_masked_active_ids(active_ids, n_sae_features)
    if active is None:
        return None
    allowed = feature_mask._allowed_universe_cached(n_sae_features)
    allowed_ids = feature_mask._allowed_feature_ids_cached(n_sae_features)
    active_allowed = int(allowed[active].sum().item()) if active.numel() > 0 else 0
    pool_size = int(allowed_ids.numel()) - active_allowed
    if pool_size <= 0:
        return t.tensor([], dtype=t.int64)

    k = sample_k(
        config.distribution,
        config.mean,
        config.std,
        rng,
        min_k=config.min_k,
        max_k=config.max_k,
    )
    k = min(k, pool_size)
    if k <= 0:
        return t.tensor([], dtype=t.int64)

    collected: list[t.Tensor] = []
    excluded = active
    remaining = k
    factor = max(MIN_OVERSAMPLE_FACTOR, n_sae_features // pool_size + 1)
    for _ in range(MAX_REJECTION_ITERATIONS):
        candidates = t.randint(
            0,
            n_sae_features,
            (factor * remaining,),
            generator=rng,
        )
        valid = candidates[allowed[candidates] & ~_sorted_ids_contain(excluded, candidates)]
        unique_valid = t.unique(valid)
        take = min(len(unique_valid), remaining)
        if take > 0:
            selected = unique_valid[:take]
            collected.append(selected)
            excluded = t.unique(t.cat((excluded, selected)), sorted=True)
            remaining -= take
        if remaining <= 0:
            break
    else:
        logger.error(
            f"Vectorized rejection sampling did not converge in "
            f"{MAX_REJECTION_ITERATIONS} iterations "
            f"(remaining={remaining}, pool_size={pool_size}, "
            f"n_sae_features={n_sae_features}). Falling back to "
            "enumerate-and-permute."
        )
        eligible = allowed_ids[~_sorted_ids_contain(excluded, allowed_ids)]
        perm = t.randperm(len(eligible), generator=rng)[:remaining]
        collected.append(eligible[perm])

    return t.cat(collected) if collected else t.tensor([], dtype=t.int64)


def _try_sample_masked_inactive_batched_sparse(
    *,
    active_ids_list: list[t.Tensor],
    n_sae_features: int,
    config: NegativeSamplingSelection,
    rng: t.Generator,
    feature_mask: FeatureMask,
    sampled_ks: t.Tensor,
) -> list[t.Tensor] | None:
    """Exact masked batched sampler without ``[segments, SAE width]`` storage.

    This is deliberately restricted to the same small, one-dimensional CPU
    active-ID domain as the audited one-segment sparse path. Outside that
    domain ``None`` routes the caller through the established dense fallback.
    Within it, pool sizes, RNG calls, candidate order, duplicate handling,
    row-order fallback calls, outputs, and final generator state match the
    dense implementation exactly.
    """
    active_rows: list[t.Tensor] = []
    for active_ids in active_ids_list:
        active = _sparse_masked_active_ids(active_ids, n_sae_features)
        if active is None:
            return None
        active_rows.append(active)

    allowed = feature_mask._allowed_universe_cached(n_sae_features)
    allowed_count = int(feature_mask._allowed_feature_ids_cached(n_sae_features).numel())
    pool_sizes = t.tensor(
        [
            allowed_count - (int(allowed[active].sum().item()) if active.numel() > 0 else 0)
            for active in active_rows
        ],
        dtype=t.int64,
    )
    all_ks = t.minimum(sampled_ks, pool_sizes).clamp(min=0)
    max_k = int(all_ks.max().item())
    if max_k == 0:
        return [t.tensor([], dtype=t.int64)] * len(active_ids_list)

    n_cand = max_k * MIN_OVERSAMPLE_FACTOR
    candidates = t.randint(
        0,
        n_sae_features,
        (len(active_ids_list), n_cand),
        generator=rng,
    )
    is_collision = ~allowed[candidates]
    for row_idx, active in enumerate(active_rows):
        is_collision[row_idx] |= _sorted_ids_contain(active, candidates[row_idx])
    is_first_occurrence = _first_unique_candidate_mask(candidates)
    is_valid = ~is_collision & is_first_occurrence
    valid_rank = is_valid.long().cumsum(dim=1)
    selected = is_valid & (valid_rank <= all_ks.unsqueeze(1))

    results: list[t.Tensor] = []
    for row_idx, active_ids in enumerate(active_ids_list):
        k = int(all_ks[row_idx].item())
        sel_mask = selected[row_idx]
        n_selected = int(sel_mask.sum().item())
        if n_selected > 0 and n_selected == k:
            sampled_ids = candidates[row_idx, sel_mask]
        elif k > 0:
            sampled_ids = sample_inactive_features(
                active_ids,
                n_sae_features,
                config,
                rng,
                feature_mask=feature_mask,
            )
        else:
            sampled_ids = t.tensor([], dtype=t.int64)
        if sampled_ids.numel() > 0:
            _require_well_formed_batched_negative_row(
                row_index=row_idx,
                sampled_ids=sampled_ids,
                active_ids=active_ids,
                n_sae_features=n_sae_features,
                config=config,
                feature_mask=feature_mask,
                pool_size=int(pool_sizes[row_idx].item()),
            )
        results.append(sampled_ids)
    return results


@beartype
def sample_k_batched(
    distribution: Distribution,
    mean: FiniteNumberLike,
    std: PositiveFiniteNumberLike,
    rng: t.Generator,
    n: NonNegativeIntLike,
    min_k: NonNegativeIntLike = 0,
    max_k: PositiveIntLike | None = None,
) -> t.Tensor:
    """Batch-sample n values of k from the configured distribution.

    For geometric with both ``min_k`` / ``max_k``: returns
    ``(Geometric(p) - 1) % R + min_k`` where ``R = max_k - min_k``.
    Mathematically equivalent to a truncated geometric on
    ``{min_k, ..., max_k - 1}``::

        P(out = i) = (1 - p)^{i - min_k} * p / (1 - (1 - p)^R)

    Matches ``sample_k`` exactly in distribution.

    Returns:
        (n,) int64 tensor of sampled k values.
    """
    if distribution == Distribution.geometric:
        p = 1.0 / (1.0 + mean)
        samples = t.empty(n, dtype=t.float64).geometric_(p, generator=rng)
        ks = (samples - 1).clamp(min=0).long()
        if max_k is not None:
            ks = (ks % (max_k - min_k)) + min_k
        elif min_k > 0:
            ks = ks + min_k
        return ks
    if distribution == Distribution.normal:
        samples = t.normal(
            mean=t.full((n,), mean, dtype=t.float64),
            std=t.full((n,), std, dtype=t.float64),
            generator=rng,
        )
        ks = samples.round().long()
        ks = ks.clamp(min=min_k)
        if max_k is not None:
            ks = ks.clamp(max=max_k - 1)
        return ks
    raise ValueError(f"Unsupported distribution '{distribution}'.")


@beartype
def sample_inactive_features_batched(
    active_ids_list: _TensorList,
    n_sae_features: PositiveIntLike,
    config: NegativeSamplingSelection,
    rng: t.Generator,
    *,
    feature_mask: FeatureMask | None = None,
) -> _TensorList:
    """Batch-sample inactive features for multiple segments at once.

    Replaces per-segment sample_inactive_features calls with batched
    operations: one geometric/normal sample, one randint, flat-mask
    collision check, sort-based deduplication, cumsum-based selection.

    Args:
        active_ids_list: List of (N_i,) int64 tensors of active feature IDs
            per segment (to exclude from sampling).
        n_sae_features: Total SAE feature space size.
        config: Negative sampling distribution config.
        rng: Seeded torch.Generator.
        feature_mask: Optional feature blacklist/whitelist restricting the
            eligible pool across all segments. When None, behavior is
            bit-identical to the legacy (no-mask) path (I5).

    Returns:
        List of (k_i,) int64 tensors of sampled inactive feature IDs,
        one per segment.
    """
    n_seg = len(active_ids_list)
    if n_seg == 0:
        return []

    # 1. Batch sample all k values
    all_ks = sample_k_batched(
        config.distribution,
        config.mean,
        config.std,
        rng,
        n_seg,
        min_k=config.min_k,
        max_k=config.max_k,
    )

    # Avoid a ``[n_seg, 65_536]`` allocation for the common small CPU active-ID
    # domain while preserving the established dense implementation as the
    # exact fallback for every other shape, device, dtype, and error edge.
    if feature_mask is not None:
        sparse_result = _try_sample_masked_inactive_batched_sparse(
            active_ids_list=active_ids_list,
            n_sae_features=n_sae_features,
            config=config,
            rng=rng,
            feature_mask=feature_mask,
            sampled_ks=all_ks,
        )
        if sparse_result is not None:
            return sparse_result

    # 2. Build flat exclusion mask: (n_seg, n_sae_features)
    # Uses O(n_seg * n_sae_features) memory but O(n_seg * n_cand) lookup
    masks = t.zeros(n_seg, n_sae_features, dtype=t.bool)
    for i, active in enumerate(active_ids_list):
        if len(active) > 0:
            masks[i, active.long()] = True

    # Restrict to the mask's allowed universe across all segments. No RNG is
    # consumed here, so sample_k_batched (step 1) remains the first RNG draw (I5).
    effective_allowed = (
        feature_mask.allowed_universe_mask(n_sae_features) if feature_mask is not None else None
    )
    if effective_allowed is not None:
        masks |= (~effective_allowed).unsqueeze(0)

    # Clamp k to available pool size per segment.
    # I5: when feature_mask is None, keep the legacy pool_sizes computation
    # (n_sae_features - len(a)) so behavior is byte-identical even if any
    # active_ids_list entry contains duplicates (the two forms could differ in
    # that edge case). When feature_mask is set, derive from the actual mask.
    if effective_allowed is None:
        pool_sizes = t.tensor([n_sae_features - len(a) for a in active_ids_list])
    else:
        pool_sizes = (~masks).sum(dim=1)
    all_ks = t.minimum(all_ks, pool_sizes).clamp(min=0)

    max_k = int(all_ks.max().item())
    if max_k == 0:
        return [t.tensor([], dtype=t.int64)] * n_seg

    # 3. Generate candidates with oversampling
    n_cand = max_k * MIN_OVERSAMPLE_FACTOR
    candidates = t.randint(0, n_sae_features, (n_seg, n_cand), generator=rng)

    # 4. Collision check via flat mask indexing (fast O(n_seg * n_cand))
    seg_idx = t.arange(n_seg).unsqueeze(1).expand_as(candidates)
    is_collision = masks[seg_idx, candidates]

    # 5. Deduplication: keep the first RNG-drawn occurrence within each row.
    is_first_occurrence = _first_unique_candidate_mask(candidates)

    # 6. Valid = not collision AND not duplicate
    is_valid = ~is_collision & is_first_occurrence

    # 7. Select first k valid candidates per segment using cumsum ranking
    valid_rank = is_valid.long().cumsum(dim=1)
    selected = is_valid & (valid_rank <= all_ks.unsqueeze(1))

    # 8. Extract results per segment
    results: list[t.Tensor] = []
    for i in range(n_seg):
        k = int(all_ks[i].item())
        sel_mask = selected[i]
        n_selected = int(sel_mask.sum().item())
        if n_selected > 0 and n_selected == k:
            sampled_ids = candidates[i, sel_mask]
        elif k > 0:
            # Rare fallback: not enough valid candidates
            sampled_ids = sample_inactive_features(
                active_ids_list[i],
                n_sae_features,
                config,
                rng,
                feature_mask=feature_mask,
            )
        else:
            sampled_ids = t.tensor([], dtype=t.int64)
        if sampled_ids.numel() > 0:
            _require_well_formed_batched_negative_row(
                row_index=i,
                sampled_ids=sampled_ids,
                active_ids=active_ids_list[i],
                n_sae_features=n_sae_features,
                config=config,
                feature_mask=feature_mask,
                pool_size=n_sae_features - int(masks[i].sum().item()),
            )
        results.append(sampled_ids)

    return results
