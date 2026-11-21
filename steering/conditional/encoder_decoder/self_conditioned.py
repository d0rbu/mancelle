"""Self-conditioned generation: stateful feature provider.

The :class:`SelfConditionedFeatureProvider` owns the encoder state during
self-conditioned generation. After every K decoder tokens, the generation
loop pools the feature-prediction head's output over those K positions and
calls :meth:`SelfConditionedFeatureProvider.step` to:

1. Pool ``fp_logits_tail`` (B, K, fp_width) using the configured pool mode.
2. Sigmoid → per-feature percentile estimates in [0, 1].
3. Apply the same selection pipeline as training. Full-percentile producers
   consume the predictions directly. Raw-magnitude producers are supported only
   for per-feature quantile cutoffs, whose statistics key can be translated into
   the prediction head's percentile currency. Cross-feature raw ordering fails
   closed because per-feature CDF transforms do not preserve it.
4. Convert selected scores to buckets with the independent bucket producer:
   uniform-percentile paths use their exact historical rule, while raw-stat
   paths search the translated statistics-key boundary grid.
5. Encode the new segment standalone via ``model.encoder`` and append its
   packed values and ownership metadata to the running ``encoder_out`` state.
6. Return a fresh :class:`SelfCondState` whose ``target_slot_idx`` points at
   the newly-appended slot, so subsequent decoder tokens attend to it.

Real-feature prefill constructs the initial encoder state. The provider's
:meth:`seed_from_prompt` then appends a slot from the prompt's own fp_logits —
this is what makes the *first* generation segment self-conditioned.

Compact checkpoints use the same downstream contract. Their feature-prediction
head emits one logit per effective-vocabulary row, so row ``j`` scores original
SAE id ``feature_vocab.original_ids[j]`` rather than feature id ``j``. The
provider pairs those compact scores with ``FeatureVocab.new_to_old_tensor()``
before selection. Masks, full-width per-feature threshold tables, bucket
assignment, and encoder inputs therefore continue to operate entirely in
original-ID space; ``FeatureEncoder._to_compact`` remains the only
original-to-compact encoder boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Protocol

import torch
from beartype import beartype
from beartype.vale import Is
from loguru import logger
from phantom import Phantom

from errors import GradientRankingUnsupportedError
from steering.conditional.discretization import (
    bucket_percentile_key_grid,
    percentile_to_bucket,
)
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.feature_prediction import pool_fp_logits
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    append_packed_encoder_segment,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    sample_action_counts,
    sample_features,
    select_topk_bottoml,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    extract_threshold_percentile,
    feature_percentile_key,
)
from steering.eval.target_conditioning import validate_feature_dict
from steering.profiling.markers import timed_region
from steering.refined_types import PositiveInt, UnitInterval
from steering.segmentation import SegmentMode

# Selection union mirrors the training-time pipeline
# (steering.conditional.feature_pipeline.FeaturePipelineConfig.selection_config).
SelfCondSelection = ThresholdSelectionConfig | SamplingBundle


class EmptySelfConditioningSegmentError(RuntimeError):
    """Self-conditioning would create or consume an empty encoder slot."""


def selection_without_delta(selection: SamplingSelection) -> SamplingSelection:
    """Copy a sampling policy while removing unreplayable delta weighting.

    Count draws, selection mode, and every other field are preserved exactly.
    Offline gradient ranking is not a removable live approximation and fails
    closed.
    """

    if selection.gradient is not None:
        raise ValueError("SelfConditioning cannot perform offline gradient ranking")
    return SamplingSelection(
        distribution=selection.distribution,
        mean=selection.mean,
        std=selection.std,
        delta_weights=None,
        min_k=selection.min_k,
        max_k=selection.max_k,
        mode=selection.mode,
        gradient=None,
    )


def _is_fp_logit_bias_tensor(value: object) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 1
        and value.layout is torch.strided
        and value.is_floating_point()
        and not value.is_complex()
        and value.device.type != "meta"
    )


class _FPLogitBiasTensor(
    Phantom,
    bound=torch.Tensor,
    predicate=_is_fp_logit_bias_tensor,
):
    """Materialized, strided, real floating vector in FP-head row space."""


_FPLogitBiasTensorLike = Annotated[
    torch.Tensor,
    Is[lambda value: isinstance(value, _FPLogitBiasTensor)],
]


@beartype
def _snapshot_fp_logit_bias(
    bias: _FPLogitBiasTensorLike,
    *,
    owner: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Snapshot a phantom-validated inference-time FP-head bias vector."""
    target_device = bias.device if device is None else device
    # ``clone`` is required even when ``to`` and ``contiguous`` are no-ops:
    # callers retain their input tensor, while this inference config must not
    # change after its validation boundary. ``detach`` also prevents a supplied
    # Parameter from rebuilding an autograd graph from detached model logits.
    snapshot = bias.detach().to(device=target_device, dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(snapshot).all()):
        raise ValueError(
            f"{owner} must remain finite when represented as float32; "
            "NaN, Inf, or finite higher-precision values that overflow float32 "
            "would make selection stop depending on the model prediction"
        )
    return snapshot


def _snapshot_target_scores(scores: torch.Tensor, *, owner: str) -> torch.Tensor:
    """Validate and freeze a dense target vector in FP-head row space."""

    if not _is_fp_logit_bias_tensor(scores):
        raise TypeError(
            f"{owner}.target_scores must be a materialized rank-1 strided real floating tensor"
        )
    snapshot = (
        scores.detach().to(device=torch.device("cpu"), dtype=torch.float32).contiguous().clone()
    )
    if not bool(torch.isfinite(snapshot).all()):
        raise ValueError(f"{owner}.target_scores must be finite after float32 conversion")
    if not bool(((snapshot >= -1.0) & (snapshot <= 1.0)).all()):
        raise ValueError(f"{owner}.target_scores must lie in [-1, 1]")
    return snapshot


@beartype
@dataclass(frozen=True)
class SelfCondDenseScorePolicy:
    """Shared contract of the dense two-sided AxBench selection policies.

    Every concrete policy pairs a dense ``target_scores`` vector — exact
    FP-head row order, finite float32 in ``[-1, 1]``, snapshotted at
    construction — with the negative (inactive-feature) count policy that the
    two-sided top-k/bottom-l selection draws alongside the config's positive
    ``selection``. The base is not constructible on its own: it carries no
    score semantics.
    """

    target_scores: torch.Tensor
    negative_selection: NegativeSamplingSelection

    def __post_init__(self) -> None:
        if type(self) is SelfCondDenseScorePolicy:
            raise TypeError(
                "SelfCondDenseScorePolicy is the abstract policy contract; "
                "construct SelfCondScoreInterpolation or SelfCondScoreMasking"
            )
        object.__setattr__(
            self,
            "target_scores",
            _snapshot_target_scores(
                self.target_scores,
                owner=type(self).__name__,
            ),
        )


@beartype
@dataclass(frozen=True)
class SelfCondScoreInterpolation(SelfCondDenseScorePolicy):
    """Dense prompt-prediction/target interpolation policy."""

    weight: UnitInterval


@beartype
@dataclass(frozen=True)
class SelfCondScoreMasking(SelfCondDenseScorePolicy):
    """Sign-consistent dense-target ranked candidate-pool policy.

    ``target_scores`` is the dense diff-mean vector ``d``. The model's own
    pooled float32 logit remains the final selection score; the target instead
    defines deterministic sign-consistent eligibility pools. Positive DiffMean
    rows are ranked descending and negative rows ascending, with stable ties in
    canonical head-row order. Exact-zero DiffMeans are eligible for neither
    side.

    ``positive_pool_limit`` and ``negative_pool_limit`` are deterministic caps
    on those ranked pools, not the stochastic feature counts drawn during
    generation. Drawn counts are clamped to the actual ranked-pool sizes.
    """

    positive_pool_limit: PositiveInt
    negative_pool_limit: PositiveInt


SelfCondScorePolicy = SelfCondScoreInterpolation | SelfCondScoreMasking


def _score_policy_owner(policy: SelfCondScorePolicy | None) -> str:
    """Historical per-policy-type name used in validation error messages."""

    return "score_masking" if isinstance(policy, SelfCondScoreMasking) else "score_interpolation"


@beartype
@dataclass(frozen=True)
class SelfCondConfig:
    """Configuration for a self-conditioned generation run.

    Mirrors the training-time selection pipeline so the head's predicted
    percentiles flow through the same selection function that was used to
    train the model on real SAE features.

    ``selection`` is a union over the two training-time pipelines:

    - :class:`ThresholdSelectionConfig`: deterministic threshold-based
      selection. ``precomputed_thresholds`` (when present) must be in
      the original-SAE-width training threshold tensor. The provider preserves
      its finite-versus-``+inf`` availability mask, but replaces every finite
      raw-magnitude value with the exact persisted statistics-key cutoff in the
      provider's continuous normalized-rank space. Do not convert the tensor
      with ``compute_percentiles`` or fill it with an idealized decimal cutoff.
      Cluster-derived checkpoints must additionally pass
      ``cluster_consolidation`` so availability is OR-folded over each
      cluster's mask-surviving members. A thresholds tensor with no finite
      entry at all is rejected at construction.
    - :class:`SamplingBundle`: stochastic sampling. The sampler weights
      features by predicted percentile (sigmoid of fp_logit) directly,
      so no separate ``percentile_table`` is consulted at inference.
      ``rng`` and the ``SamplingSelection`` distribution / mean / std
      come from the bundle.

    The segment size, bucket count, and SAE width are strict positive integers.
    ``@beartype`` enforces those unary contracts and the selection union at
    construction; ``__post_init__`` retains only the cross-field pool-mode
    relationship that cannot be expressed by an individual field type.

    ``n_sae_features`` always names the original SAE dictionary, including
    when the checkpoint's prediction head emits only ``feature_vocab.size``
    logits. ``num_buckets`` is the number of buckets available to predicted
    active features. For a checkpoint trained with a reserved negative bucket,
    derive it with :func:`steering.conditional.feature_pipeline.active_bucket_count`
    instead of passing the encoder's full bucket-table size.

    ``selection_producer`` and ``bucket_producer`` are independent identities.
    This matters for unclustered percentile sampling: selection uses full ranks,
    while the returned raw magnitudes are discretized against statistics-derived
    boundaries.

    ``fp_logit_bias`` is an optional fixed inference control in prediction-head
    row space. On a compact checkpoint, row ``j`` scores original SAE id
    ``feature_vocab.original_ids[j]``. Translate sparse original ids with
    :meth:`FeatureVocab.to_compact`. For an already-dense original-width vector,
    first validate its nonzero support with ``to_compact``, then gather with
    :meth:`FeatureVocab.select_rows`; the gather alone silently drops excluded
    rows. The config snapshots the vector as detached contiguous float32, and
    the provider snapshots it again, so later caller/config mutation cannot
    alter a running generation.
    """

    self_cond_segment_size: PositiveInt
    """Number of nanochat (decoder) tokens per predicted segment."""

    pool_mode: FeaturePredLossMode
    """How to collapse (B, K, fp_width) fp_logits to (B, fp_width). One of
    ``MAX_POOL`` (default), ``LAST_TOKEN``, or ``FULL_SEGMENT``;
    ``BEST_TOKEN`` is forbidden."""

    selection: SelfCondSelection
    """Threshold-mode or sampling-mode selection (union)."""

    selection_producer: FeatureSelectionProducer
    """Exact score producer used to select positive conditioning features."""

    bucket_producer: FeatureBucketProducer
    """Exact producer used to bucket selected positive features."""

    num_buckets: PositiveInt
    n_sae_features: PositiveInt
    feature_mask: FeatureMask | None = None
    cluster_consolidation: FeatureClusterConsolidation | None = None
    """Cluster consolidation of the checkpoint that produced this head.

    Required for cluster-derived compact checkpoints that use per-feature
    threshold availability: clustered training looks up sibling statistics in
    ORIGINAL-id space BEFORE consolidating each cluster into its
    representative, so availability must be folded over the mask-surviving
    cluster members rather than read off the representative row alone. Rebuild
    it from the checkpoint's metadata with
    :func:`steering.conditional.feature_clusters.resolve_feature_clusters_from_checkpoint_meta`
    followed by :meth:`FeatureClusterConsolidation.build` with the checkpoint's
    fully resolved feature mask.

    It is consulted only for that availability fold; it is a validated no-op
    for selection types without per-feature thresholds.
    """

    fp_logit_bias: torch.Tensor | None = None
    """Optional 1-D bias added after FP-logit pooling and before sigmoid.

    Its width is the feature-prediction head's own row count: full SAE width on
    an uncompacted checkpoint and ``feature_vocab.size`` on a compact one. Use
    ``feature_vocab.to_compact(original_ids)`` to place sparse ID controls. For
    a dense original-width vector, first validate
    ``feature_vocab.to_compact(full_bias.nonzero().flatten())``, then call
    ``feature_vocab.select_rows(full_bias)``; never use the gather alone.
    ``None`` executes no bias arithmetic. Values must be real, finite after
    float32 conversion, materialized, and strided; they are detached and copied.
    """

    score_policy: SelfCondScorePolicy | None = None
    """Optional dense two-sided AxBench score policy (one field, one policy).

    - :class:`SelfCondScoreInterpolation`: pooled float32 logits are ranked at
      weight zero; for an interior weight, ``q = 2 * sigmoid(logits) - 1`` is
      interpolated with dense ``target_scores`` before stable selection.
    - :class:`SelfCondScoreMasking`: pooled float32 logits are the ranking
      score, while stable DiffMean rank defines sign-consistent candidate pools
      capped independently for positive and negative rows.

    The union makes the policies structurally mutually exclusive. Either
    policy is mutually exclusive with ``fp_logit_bias``. ``None`` preserves
    the legacy provider and prefill paths exactly.
    """

    # Appending this field preserves every historical optional positional
    # argument binding while adding the new mode as a keyword-capable field.
    segment_mode: SegmentMode = SegmentMode.fixed_tokens
    """Feature-segmentation geometry recorded by the post-training checkpoint.

    ``fixed_tokens`` preserves the historical two-phase boundary replay.
    ``turn_delimited_fixed_tokens`` treats ``<assistant_start>`` as the first
    row of a generated RustBPE window whose cadence is
    ``self_cond_segment_size``.  Callers must resolve both values from the same
    checkpoint metadata; this runtime never guesses a cadence from the source
    Gemma extraction segment size.
    """

    def __post_init__(self) -> None:
        if self.pool_mode == FeaturePredLossMode.BEST_TOKEN:
            raise ValueError(
                "SelfCondConfig.pool_mode=BEST_TOKEN is not supported: "
                "best-token selection requires the next segment's target "
                "features to compute argmin loss, which is unavailable at "
                "inference. Use MAX_POOL, LAST_TOKEN, or FULL_SEGMENT."
            )
        bias = self.fp_logit_bias
        dense_policy = self.score_policy
        owner = _score_policy_owner(dense_policy)
        if bias is not None and dense_policy is not None:
            raise ValueError(f"SelfCondConfig cannot combine fp_logit_bias with {owner}")
        if bias is not None:
            object.__setattr__(
                self,
                "fp_logit_bias",
                _snapshot_fp_logit_bias(
                    bias,
                    owner="SelfCondConfig.fp_logit_bias",
                ),
            )
        if dense_policy is not None:
            if self.pool_mode is not FeaturePredLossMode.MAX_POOL:
                raise ValueError(f"{owner} requires SelfCondConfig.pool_mode=MAX_POOL")
            if not isinstance(self.selection, SamplingBundle):
                raise TypeError(f"{owner} requires a SamplingBundle positive policy")
            if self.selection.selection.mode is not SamplingMode.top_k_percentile:
                raise ValueError(f"{owner} requires SamplingMode.top_k_percentile")
            if self.selection_producer is not FeatureSelectionProducer.FULL_PERCENTILE_RANK:
                raise ValueError(f"{owner} requires FULL_PERCENTILE_RANK selection")
            if self.bucket_producer is not FeatureBucketProducer.UNIFORM_PERCENTILE:
                raise ValueError(f"{owner} requires UNIFORM_PERCENTILE buckets")
            if int(self.num_buckets) != 1:
                raise ValueError(
                    f"{owner} requires exactly one active bucket; "
                    "the encoder's second bucket is reserved for inactive features"
                )
            for label, counts in (
                ("positive selection", self.selection.selection),
                ("negative selection", dense_policy.negative_selection),
            ):
                if counts.max_k is None:
                    raise ValueError(f"{owner} {label} must have bounded max_k")
        if isinstance(self.selection, ThresholdSelectionConfig):
            threshold_type = self.selection.selection.threshold_type
            if (
                self.selection_producer is FeatureSelectionProducer.RAW_MAGNITUDE
                and threshold_type is not ThresholdType.feature_percentile
            ):
                raise NotImplementedError(
                    "selection_producer=RAW_MAGNITUDE is replayable only for "
                    "threshold_type=feature_percentile. Segment top-k, segment "
                    "percentile, and feature_percentile_segment_topk compare raw "
                    "magnitudes across different features; per-feature percentile "
                    "predictions do not preserve that ordering."
                )
        if isinstance(self.selection, SamplingBundle):
            if self.selection_producer is not FeatureSelectionProducer.FULL_PERCENTILE_RANK:
                raise NotImplementedError(
                    "self-conditioned sampling requires "
                    "selection_producer=FULL_PERCENTILE_RANK: a feature-"
                    "prediction head cannot reconstruct raw-magnitude sampling "
                    "weights or cross-feature ordering"
                )
            sampling = self.selection.selection
            if sampling.delta_weights is not None:
                raise NotImplementedError(
                    "self-conditioned sampling cannot reproduce "
                    "feature_selection.delta_weights: percentile-delta and "
                    "absolute-percentile weighting require reference-segment "
                    "provenance that SelfCondConfig does not carry"
                )
            if sampling.gradient is not None:
                raise GradientRankingUnsupportedError(
                    "self-conditioned generation performs live feature selection "
                    "and cannot reproduce offline gradient-attribution ranking"
                )


def rank_masked_candidate_pools(
    target_scores: torch.Tensor,
    *,
    positive_pool_limit: int,
    negative_pool_limit: int,
    keep_rows: torch.Tensor | None,
    positive_min_k: int,
    negative_min_k: int,
    owner: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build stable sign-consistent ranked pools and reject empty required sides."""

    if not _is_fp_logit_bias_tensor(target_scores):
        raise TypeError(f"{owner} target_scores must be a rank-1 real floating tensor")
    if not bool(torch.isfinite(target_scores).all()):
        raise ValueError(f"{owner} target_scores must be finite")
    if not bool(((target_scores >= -1.0) & (target_scores <= 1.0)).all()):
        raise ValueError(f"{owner} target_scores must lie in [-1, 1]")
    for label, value in (
        ("positive_pool_limit", positive_pool_limit),
        ("negative_pool_limit", negative_pool_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{owner} {label} must be a positive integer, got {value!r}")
    for label, value in (
        ("positive_min_k", positive_min_k),
        ("negative_min_k", negative_min_k),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{owner} {label} must be a non-negative integer, got {value!r}")
    if keep_rows is not None:
        if (
            not isinstance(keep_rows, torch.Tensor)
            or keep_rows.ndim != 1
            or keep_rows.dtype is not torch.bool
            or keep_rows.layout is not torch.strided
            or keep_rows.device.type == "meta"
        ):
            raise TypeError(f"{owner} keep_rows must be a materialized rank-1 bool tensor")
        if keep_rows.shape != target_scores.shape:
            raise ValueError(
                f"{owner} keep_rows width {int(keep_rows.numel())} does not match "
                f"target width {int(target_scores.numel())}"
            )
        if keep_rows.device != target_scores.device:
            raise ValueError(
                f"{owner} keep_rows device {keep_rows.device} does not match "
                f"target device {target_scores.device}"
            )

    positive_base = target_scores > 0.0
    negative_base = target_scores < 0.0
    if keep_rows is not None:
        positive_base = positive_base & keep_rows
        negative_base = negative_base & keep_rows

    positive_ids = torch.nonzero(positive_base, as_tuple=False).flatten()
    negative_ids = torch.nonzero(negative_base, as_tuple=False).flatten()
    positive_order = torch.argsort(
        target_scores[positive_ids],
        descending=True,
        stable=True,
    )
    negative_order = torch.argsort(
        target_scores[negative_ids],
        descending=False,
        stable=True,
    )
    positive_selected = positive_ids[positive_order[: min(positive_pool_limit, positive_ids.numel())]]
    negative_selected = negative_ids[negative_order[: min(negative_pool_limit, negative_ids.numel())]]
    positive_rows = torch.zeros_like(target_scores, dtype=torch.bool)
    negative_rows = torch.zeros_like(target_scores, dtype=torch.bool)
    positive_rows[positive_selected] = True
    negative_rows[negative_selected] = True

    has_positive = bool(positive_rows.any())
    has_negative = bool(negative_rows.any())
    if not has_positive and not has_negative:
        raise ValueError(
            f"{owner} has no eligible head rows: after the sign rule, feature "
            "mask, and rank limits, both candidate pools are empty"
        )
    if not has_positive and positive_min_k >= 1:
        raise ValueError(
            f"{owner} has an empty positive candidate pool, but the positive "
            f"selection requires min_k={positive_min_k}"
        )
    if not has_negative and negative_min_k >= 1:
        raise ValueError(
            f"{owner} has an empty negative candidate pool, but the negative "
            f"selection requires min_k={negative_min_k}"
        )
    return positive_rows, negative_rows


@dataclass(frozen=True)
class SelfCondState:
    """Running encoder state during self-conditioned generation.

    Note: ``frozen=True`` prevents field rebinding only; tensor element
    contents remain mutable post-construction. The provider always returns
    a *new* state from :meth:`step` rather than mutating in place.
    """

    encoder_out: PackedEncoderOutput
    target_slot_idx: int  # which slot future decoder tokens should attend to

    def __post_init__(self) -> None:
        _require_nonempty_self_cond_state(self, owner="SelfCondState")


def _require_nonempty_self_cond_state(state: SelfCondState, *, owner: str) -> None:
    """Reject a state before any decoder token can attend to an empty slot."""

    if not isinstance(state.encoder_out, PackedEncoderOutput):
        raise TypeError(
            f"{owner} requires PackedEncoderOutput, got "
            f"{type(state.encoder_out).__name__}"
        )
    layout = state.encoder_out.layout
    if layout.batch_size != 1:
        raise ValueError(
            f"{owner} requires batch-one packed encoder state, got "
            f"batch_size={layout.batch_size}"
        )
    valid_segment_indices = layout.segment_indices[layout.valid_mask]
    counts = torch.bincount(valid_segment_indices, minlength=layout.n_segments)
    empty_slots = (counts == 0).nonzero(as_tuple=False)
    if empty_slots.numel():
        slots = empty_slots.flatten().tolist()
        raise EmptySelfConditioningSegmentError(
            f"{owner} received a SelfCondState containing empty encoder slots "
            f"{slots}; refusing out-of-distribution conditioning"
        )


@dataclass
class SelfCondRuntime:
    """Wrapper-side runtime container for self-conditioned generation.

    All fields are required when this object exists. The wrapper either
    holds an instance of ``SelfCondRuntime`` (self-cond mode) or ``None``
    (every other target type). This avoids the failure mode where some
    subset of self-cond fields could be ``None`` independently.

    Mutated per forward call: ``state``, ``fp_buffer``, ``tokens_in_seg``.
    Constant across the wrapper's lifetime: ``target``, ``config``,
    ``provider``, except that ``swap_self_cond_provider`` is the sanctioned
    provider/config replacement path.
    """

    target: object  # SelfConditionedTarget — typed in eval_adapter to avoid cycle
    config: SelfCondConfig
    provider: SelfConditionedFeatureProvider
    state: SelfCondState | None  # None until first prefill builds initial state
    fp_buffer: tuple[torch.Tensor, ...]
    tokens_in_seg: int


@dataclass(frozen=True)
class SelfCondDecodeStepResult:
    """One successfully committed emitted-token transition.

    ``fp_buffer`` is an immutable snapshot so callers publish the state,
    generated-window rows, and counter together only after every model/provider
    operation succeeds.
    """

    logits: torch.Tensor
    state: SelfCondState
    fp_buffer: tuple[torch.Tensor, ...]
    tokens_in_seg: int


def final_real_prompt_source_mask(
    seg_map: torch.Tensor,
    *,
    encoder_slot_count: int,
) -> torch.Tensor:
    """Validate prompt routing and select the final real slot before the boundary.

    The final prompt token is the assistant boundary.  The preceding token
    identifies the final real source slot; the returned mask excludes the
    boundary itself from feature-prediction pooling.  The boundary may fall in
    the same or a later real context slot because segmentation is independent
    of chat-template token boundaries.

    Turn-mode prefill drops the terminal marker-only assistant feature slot
    before encoder packing, so the ``<assistant_start>`` column keeps the
    zero fill ``_build_conditioning`` allocates.  Slot 0 is a real slot
    whenever ``encoder_slot_count >= 1``, which the check above already
    requires, so the boundary check is satisfied without special-casing the
    marker.
    """

    if seg_map.ndim != 2 or seg_map.shape[0] != 1:
        raise ValueError(
            f"self-conditioned prompt seg_map must have shape (1, T), got {tuple(seg_map.shape)}"
        )
    prompt_length = int(seg_map.shape[1])
    if prompt_length < 2:
        raise ValueError("self-conditioned prefill requires at least two prompt tokens")
    if encoder_slot_count < 1:
        raise ValueError("self-conditioned prefill requires at least one real prompt slot")

    boundary_slot = int(seg_map[0, -1].item())
    if boundary_slot < 0 or boundary_slot >= encoder_slot_count:
        raise RuntimeError(
            "the final self-conditioned prompt position must route to a real "
            f"prompt slot in [0, {encoder_slot_count}); got {boundary_slot}"
        )
    source_slot = int(seg_map[0, -2].item())
    if source_slot < 0 or source_slot >= encoder_slot_count:
        raise RuntimeError(
            "the final self-conditioning source slot must be a real prompt "
            f"slot in [0, {encoder_slot_count}); got {source_slot}"
        )
    source_mask = seg_map[0, :-1] == source_slot
    if not bool(source_mask.any()):
        raise RuntimeError("the final real prompt slot has no routed source positions")
    return source_mask


def _restore_kv_position(kv_cache, expected_position: int) -> None:
    """Rewind speculative cache advancement to ``expected_position``."""

    observed = int(kv_cache.get_pos())
    if observed < expected_position:
        raise RuntimeError(
            "self-conditioned decode moved the KV cache backwards unexpectedly: "
            f"expected at least {expected_position}, got {observed}"
        )
    if observed > expected_position:
        kv_cache.rewind(observed - expected_position)


class _SelfCondSegmentConfig(Protocol):
    @property
    def self_cond_segment_size(self) -> int: ...

    @property
    def segment_mode(self) -> SegmentMode: ...


class SelfConditionedDecodeProvider(Protocol):
    @property
    def cfg(self) -> _SelfCondSegmentConfig: ...

    def step(
        self,
        state: SelfCondState,
        fp_logits_tail: torch.Tensor,
    ) -> SelfCondState: ...


def self_conditioned_decode_step(
    *,
    model,
    provider: SelfConditionedDecodeProvider,
    state: SelfCondState,
    fp_buffer: tuple[torch.Tensor, ...],
    tokens_in_seg: int,
    emitted_token: torch.Tensor,
    kv_cache,
    device: torch.device,
) -> SelfCondDecodeStepResult:
    """Commit one emitted token under its conditioning mode's causal contract.

    Non-boundary tokens are forwarded once under the current state and their FP
    row is retained.  Under ``fixed_tokens`` the assistant boundary sits outside
    its first generated window, so a Kth token is first forwarded speculatively
    under the outgoing state for its FP row, rewound, then committed at the same
    absolute position under the incoming state: its LM logits are the
    distribution for emitted token K+1.  Under ``turn_delimited_fixed_tokens``
    the input and its FP row belong to the same generated assistant window, so
    the Kth input is forwarded exactly once, keeps its outgoing state's logits,
    and only then publishes the provider's incoming state for the next input.
    No caller-visible runtime field is mutated here.
    """

    if emitted_token.ndim != 2 or tuple(emitted_token.shape) != (1, 1):
        raise ValueError(
            "self-conditioned emitted_token must have shape (1, 1), got "
            f"{tuple(emitted_token.shape)}"
        )
    K = int(provider.cfg.self_cond_segment_size)
    if isinstance(tokens_in_seg, bool) or not isinstance(tokens_in_seg, int):
        raise TypeError("tokens_in_seg must be an integer")
    if tokens_in_seg < 0 or tokens_in_seg >= K:
        raise ValueError(f"tokens_in_seg must be in [0, {K}), got {tokens_in_seg}")
    if len(fp_buffer) != tokens_in_seg:
        raise ValueError(
            "self-conditioned FP-buffer cardinality must equal tokens_in_seg: "
            f"buffer={len(fp_buffer)}, counter={tokens_in_seg}"
        )
    _require_nonempty_self_cond_state(state, owner="self-conditioned decode")
    replay_boundary = provider.cfg.segment_mode is SegmentMode.fixed_tokens

    seg_map = torch.full(
        (1, 1),
        state.target_slot_idx,
        dtype=torch.long,
        device=device,
    )
    common = {
        "seg_map": seg_map,
        "kv_cache": kv_cache,
        "encoder_out": state.encoder_out.to(device),
        "encoder_strict": True,
    }
    start_position = int(kv_cache.get_pos())
    completes_window = tokens_in_seg + 1 == K

    if not (completes_window and replay_boundary):
        if replay_boundary:
            forward_region = "selfcond/decode/non_boundary_forward"
        elif completes_window:
            forward_region = "selfcond/turn_decode/boundary_forward"
        else:
            forward_region = "selfcond/turn_decode/forward"
        try:
            with timed_region(forward_region, sync_cuda=False):
                logits, fp_logits = model(
                    emitted_token,
                    None,
                    return_fp_logits=True,
                    **common,
                )
            if int(kv_cache.get_pos()) != start_position + 1:
                raise RuntimeError(
                    "self-conditioned decode forward must advance the KV cache "
                    "by exactly one position"
                )
        except BaseException:
            _restore_kv_position(kv_cache, start_position)
            raise
        completed_buffer = (*fp_buffer, fp_logits.detach())
        if not completes_window:
            return SelfCondDecodeStepResult(
                logits=logits,
                state=state,
                fp_buffer=completed_buffer,
                tokens_in_seg=tokens_in_seg + 1,
            )
        # Turn mode only: the window is complete and its Kth input has already
        # been committed under the outgoing state, so the incoming state is
        # published for the next input without replaying anything.
        try:
            completed_rows = torch.cat(completed_buffer, dim=1)
            with timed_region("selfcond/turn_decode/provider_step", sync_cuda=False):
                incoming_state = provider.step(state, completed_rows)
            _require_nonempty_self_cond_state(
                incoming_state,
                owner="self-conditioned incoming decode",
            )
        except BaseException:
            _restore_kv_position(kv_cache, start_position)
            raise
        return SelfCondDecodeStepResult(
            logits=logits,
            state=incoming_state,
            fp_buffer=(),
            tokens_in_seg=0,
        )

    try:
        with timed_region(
            "selfcond/decode/boundary_measurement_forward",
            sync_cuda=False,
        ):
            boundary_fp_logits = model(
                emitted_token,
                None,
                fp_logits_only=True,
                **common,
            )
        if int(kv_cache.get_pos()) != start_position + 1:
            raise RuntimeError(
                "self-conditioned speculative boundary forward must advance "
                "the KV cache by exactly one position"
            )
    except BaseException:
        _restore_kv_position(kv_cache, start_position)
        raise

    kv_cache.rewind(1)
    if int(kv_cache.get_pos()) != start_position:
        raise RuntimeError("self-conditioned boundary rewind did not restore the KV position")
    completed_rows = torch.cat((*fp_buffer, boundary_fp_logits.detach()), dim=1)
    with timed_region("selfcond/decode/provider_step", sync_cuda=False):
        incoming_state = provider.step(state, completed_rows)
    _require_nonempty_self_cond_state(
        incoming_state,
        owner="self-conditioned incoming decode",
    )

    incoming_seg_map = torch.full(
        (1, 1),
        incoming_state.target_slot_idx,
        dtype=torch.long,
        device=device,
    )
    try:
        with timed_region(
            "selfcond/decode/boundary_commit_forward",
            sync_cuda=False,
        ):
            logits = model(
                emitted_token,
                None,
                seg_map=incoming_seg_map,
                kv_cache=kv_cache,
                encoder_out=incoming_state.encoder_out.to(device),
                encoder_strict=True,
            )
        if int(kv_cache.get_pos()) != start_position + 1:
            raise RuntimeError(
                "self-conditioned committed boundary forward must advance "
                "the KV cache by exactly one position"
            )
    except BaseException:
        _restore_kv_position(kv_cache, start_position)
        raise

    return SelfCondDecodeStepResult(
        logits=logits,
        state=incoming_state,
        fp_buffer=(),
        tokens_in_seg=0,
    )


def assert_self_conditioned_feature_space(model, cfg: SelfCondConfig) -> None:
    """Validate the head, vocabulary, and config currencies used by feedback.

    Self-conditioning pairs prediction-head rows and candidate feature ids
    positionally. A width mismatch would attribute a valid score to the wrong
    valid original id, so neither selection nor the encoder's out-of-vocabulary
    guard could detect it. This check also pins ``SelfCondConfig`` to the
    model's original SAE width and permits only the encoder's full bucket count
    or the one-smaller active count produced when its final bucket is reserved.
    The model does not persist whether training enabled negative sampling, so
    this is necessarily a structural guard over the only two valid shapes; the
    caller remains responsible for choosing between them via
    :func:`steering.conditional.feature_pipeline.active_bucket_count` from the
    checkpoint's selection metadata.

    The caller checks for a missing head first so that headless models retain
    their existing, more actionable error.
    """
    head = model.feature_pred_head
    if head is None:
        return

    model_n_sae = int(model.encoder.config.n_sae_features)
    cfg_n_sae = int(cfg.n_sae_features)
    if cfg_n_sae != model_n_sae:
        raise ValueError(
            "SelfCondConfig.n_sae_features must name this model's original "
            f"SAE dictionary: config has {cfg_n_sae}, model has {model_n_sae}."
        )

    vocab = model.feature_vocab
    expected_head_width = model_n_sae
    if vocab is not None:
        if int(vocab.n_sae_features) != model_n_sae:
            raise RuntimeError(
                "the compact feature vocabulary names a different original "
                f"SAE width than the encoder: vocab has {vocab.n_sae_features}, "
                f"encoder has {model_n_sae}"
            )
        model.assert_vocab_consistent()
        expected_head_width = int(vocab.size)

    head_width = int(head.out_features)
    if head_width != expected_head_width:
        space = "compact feature vocabulary" if vocab is not None else "original SAE dictionary"
        raise RuntimeError(
            "feature-prediction head width does not match the self-conditioned "
            f"candidate space: head emits {head_width} logits but the {space} "
            f"has {expected_head_width} entries (n_sae_features={model_n_sae}). "
            "Prediction row j and candidate id j would refer to different "
            "features."
        )

    encoder_buckets = int(model.encoder.config.num_buckets)
    active_buckets = int(cfg.num_buckets)
    allowed_bucket_counts = {encoder_buckets}
    if encoder_buckets >= 2:
        allowed_bucket_counts.add(encoder_buckets - 1)
    if active_buckets not in allowed_bucket_counts:
        allowed = ", ".join(str(value) for value in sorted(allowed_bucket_counts))
        raise ValueError(
            "SelfCondConfig.num_buckets must be the encoder's full bucket count "
            "or the active count after reserving exactly its final bucket: "
            f"got {active_buckets}, expected one of {{{allowed}}} for an encoder "
            f"with num_buckets={encoder_buckets}."
        )


def assert_cluster_consolidation_matches_checkpoint(model, cfg: SelfCondConfig) -> None:
    """Validate ``cfg.cluster_consolidation`` against the checkpoint's vocabulary.

    A consolidation is only meaningful for a cluster-derived compact
    checkpoint: the effective vocabulary already contains exactly the surviving
    representatives, so every candidate original id must be its own
    mask-surviving representative. A consolidation built from a different
    cluster file, a different mask, or a different SAE width would fold
    availability onto the wrong rows, which is silent — hence a structural
    check at construction rather than a runtime guard.
    """
    consolidation = cfg.cluster_consolidation
    if consolidation is None:
        return

    vocab = model.feature_vocab
    if vocab is None:
        raise ValueError(
            "SelfCondConfig.cluster_consolidation was provided for an "
            "uncompacted checkpoint (model.feature_vocab is None). Clusters "
            "imply a compacted effective vocabulary whose rows are the "
            "surviving representatives; pairing a consolidation with a "
            "full-width head is contradictory. Drop the consolidation or load "
            "the cluster-derived compact checkpoint."
        )

    rep_of = consolidation.rep_of
    n_sae = int(cfg.n_sae_features)
    if int(rep_of.shape[0]) != n_sae:
        raise ValueError(
            "SelfCondConfig.cluster_consolidation names a different original "
            f"SAE width than this run: consolidation has {int(rep_of.shape[0])} "
            f"features, config/model has {n_sae}."
        )

    candidates = vocab.new_to_old_tensor().to(dtype=torch.int64, device=rep_of.device)
    representatives = rep_of[candidates]
    away = consolidation.consolidated_away.to(device=rep_of.device)
    offending = (representatives != candidates) | away[candidates]
    if bool(offending.any()):
        first = int(candidates[offending][0].item())
        raise ValueError(
            "SelfCondConfig.cluster_consolidation does not describe this "
            "checkpoint's effective vocabulary: original feature id "
            f"{first} has a prediction-head row but maps to representative "
            f"{int(rep_of[first].item())} and consolidated_away="
            f"{bool(away[first].item())}. Rebuild the consolidation from this "
            "checkpoint's own cluster metadata and resolved feature mask."
        )


class SelfConditionedFeatureProvider:
    """Stateful provider that grows the encoder by one segment per step."""

    def __init__(
        self,
        model,  # EncoderDecoderGPT (cannot import for type-hint w/o cycle)
        cfg: SelfCondConfig,
        device: torch.device,
    ) -> None:
        if model.feature_pred_head is None:
            raise RuntimeError(
                "SelfConditionedFeatureProvider requires a model trained with "
                "feature_pred_config; got a model with feature_pred_head=None."
            )
        assert_self_conditioned_feature_space(model, cfg)
        assert_cluster_consolidation_matches_checkpoint(model, cfg)
        fp_cfg = model.feature_pred_config
        if fp_cfg is not None and fp_cfg.loss_mode == FeaturePredLossMode.BEST_TOKEN:
            raise RuntimeError(
                "SelfConditionedFeatureProvider cannot consume a checkpoint "
                "trained with feature_pred_config.loss_mode=BEST_TOKEN. "
                "Train or fine-tune with MAX_POOL, LAST_TOKEN, or FULL_SEGMENT "
                "before self-conditioned generation."
            )
        self._model = model
        self.cfg = cfg
        self._device = device
        self.last_appended_segment: dict[int, int] | None = None
        self._queued_explicit: dict[int, int] | None = None
        vocab = model.feature_vocab
        # Keep the uncompacted path literal: it still builds arange(n_sae) in
        # ``step``. Only compact checkpoints cache a candidate vector, whose
        # values are ORIGINAL SAE ids aligned positionally with the V head rows.
        self._compact_candidate_feature_ids: torch.Tensor | None = (
            None if vocab is None else vocab.new_to_old_tensor(device=device).to(torch.int32)
        )
        self._target_scores: torch.Tensor | None = None
        self._masking_positive_rows: torch.Tensor | None = None
        self._masking_negative_rows: torch.Tensor | None = None
        dense_policy = cfg.score_policy
        if dense_policy is not None:
            owner = _score_policy_owner(dense_policy)
            if fp_cfg is None or FeaturePredLossMode(fp_cfg.loss_mode) is not (
                FeaturePredLossMode.MAX_POOL
            ):
                observed = None if fp_cfg is None else fp_cfg.loss_mode
                raise RuntimeError(
                    f"{owner} requires a checkpoint trained with "
                    "feature_pred_config.loss_mode=MAX_POOL; got "
                    f"{observed!r}"
                )
            if int(model.encoder.config.num_buckets) != 2:
                raise ValueError(
                    f"{owner} requires an encoder with exactly two "
                    "buckets (active bucket 0, inactive bucket 1)"
                )
            expected_width = (
                int(cfg.n_sae_features)
                if self._compact_candidate_feature_ids is None
                else int(self._compact_candidate_feature_ids.shape[0])
            )
            target_width = int(dense_policy.target_scores.shape[0])
            if target_width != expected_width:
                raise ValueError(
                    f"{owner} target width must equal the exact "
                    "feature-prediction head row count: "
                    f"got {target_width}, expected {expected_width}"
                )
            row_original_ids = (
                torch.arange(expected_width, dtype=torch.int64, device=device)
                if self._compact_candidate_feature_ids is None
                else self._compact_candidate_feature_ids.to(torch.int64)
            )
            keep_rows = (
                None
                if cfg.feature_mask is None
                else cfg.feature_mask.keep_mask(row_original_ids).to(device=device)
            )
            candidate_ids = row_original_ids if keep_rows is None else row_original_ids[keep_rows]
            if not isinstance(cfg.selection, SamplingBundle):
                raise RuntimeError(f"{owner} sampling invariant was not preserved")
            if isinstance(dense_policy, SelfCondScoreInterpolation):
                positive_max_k = cfg.selection.selection.max_k
                negative_max_k = dense_policy.negative_selection.max_k
                if positive_max_k is None or negative_max_k is None:
                    raise RuntimeError(
                        "score_interpolation bounded-count invariant was not preserved"
                    )
                maximum_positive = int(positive_max_k) - 1
                maximum_negative = int(negative_max_k) - 1
                if maximum_positive + maximum_negative > int(candidate_ids.numel()):
                    raise ValueError(
                        "score_interpolation bounded positive+negative cardinality "
                        "does not fit the eligible feature vocabulary: "
                        f"{maximum_positive}+{maximum_negative} > "
                        f"{int(candidate_ids.numel())}"
                    )
                self._target_scores = dense_policy.target_scores.to(
                    device=device,
                    dtype=torch.float32,
                )
            else:
                target = dense_policy.target_scores.to(device=device, dtype=torch.float32)
                positive_rows, negative_rows = rank_masked_candidate_pools(
                    target,
                    positive_pool_limit=int(dense_policy.positive_pool_limit),
                    negative_pool_limit=int(dense_policy.negative_pool_limit),
                    keep_rows=keep_rows,
                    positive_min_k=int(cfg.selection.selection.min_k),
                    negative_min_k=int(dense_policy.negative_selection.min_k),
                    owner="score_masking",
                )
                self._masking_positive_rows = positive_rows
                self._masking_negative_rows = negative_rows
        self._fp_logit_bias: torch.Tensor | None = None
        if cfg.fp_logit_bias is not None:
            provider_bias = _snapshot_fp_logit_bias(
                cfg.fp_logit_bias,
                owner="SelfConditionedFeatureProvider fp_logit_bias",
                device=device,
            )
            expected_width = (
                int(cfg.n_sae_features)
                if self._compact_candidate_feature_ids is None
                else int(self._compact_candidate_feature_ids.shape[0])
            )
            bias_width = int(provider_bias.shape[0])
            if bias_width != expected_width:
                raise ValueError(
                    "SelfCondConfig.fp_logit_bias has width "
                    f"{bias_width}, but this provider selects over "
                    f"{expected_width} feature-prediction head rows. The bias "
                    "uses head-row space; on a compact checkpoint translate "
                    "sparse ids with feature_vocab.to_compact(). For a dense "
                    "original-width vector, validate its nonzero ids with "
                    "to_compact() before feature_vocab.select_rows()."
                )
            self._fp_logit_bias = provider_bias
        threshold_percentile: float | None = None
        self._selection_percentile_thresholds: torch.Tensor | None = None
        if isinstance(cfg.selection, ThresholdSelectionConfig):
            threshold_type = cfg.selection.selection.threshold_type
            if threshold_type in (
                ThresholdType.feature_percentile,
                ThresholdType.feature_percentile_segment_topk,
            ):
                threshold_cfg = ThresholdConfig(
                    threshold_type,
                    cfg.selection.selection.threshold,
                )
                threshold_percentile = extract_threshold_percentile(threshold_cfg)
                raw_thresholds = cfg.selection.precomputed_thresholds
                if raw_thresholds is None:
                    raise ValueError(
                        f"{threshold_type.value} self-conditioning requires the "
                        "original-width training precomputed_thresholds tensor"
                    )
                if raw_thresholds.ndim != 1 or raw_thresholds.shape[0] != int(cfg.n_sae_features):
                    raise ValueError(
                        "self-conditioned precomputed_thresholds must have shape "
                        f"({cfg.n_sae_features},), got {tuple(raw_thresholds.shape)}"
                    )
                if not raw_thresholds.is_floating_point():
                    raise TypeError(
                        "self-conditioned precomputed_thresholds must have a "
                        f"floating-point dtype, got {raw_thresholds.dtype}"
                    )
                if torch.isnan(raw_thresholds).any() or torch.isneginf(raw_thresholds).any():
                    raise ValueError(
                        "self-conditioned precomputed_thresholds may contain only "
                        "finite training thresholds or +inf for features missing "
                        "statistics"
                    )
                if cfg.selection_producer is FeatureSelectionProducer.FULL_PERCENTILE_RANK:
                    # Full-rank producers compare the head's continuous
                    # float32 score directly with the configured cutoff.
                    target_cutoff = torch.tensor(
                        threshold_percentile,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )
                elif cfg.selection_producer is FeatureSelectionProducer.RAW_MAGNITUDE:
                    threshold_key = feature_percentile_key(threshold_cfg)
                    threshold_rank = 99.5 if threshold_key == 99 else float(threshold_key)
                    target_cutoff = torch.tensor(
                        threshold_rank / 100.0,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )
                else:
                    raise AssertionError(
                        f"unhandled feature-selection producer: {cfg.selection_producer!r}"
                    )
                self._selection_percentile_thresholds = torch.full(
                    (int(cfg.n_sae_features),),
                    float("inf"),
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                available = torch.isfinite(raw_thresholds).to(device="cpu")
                if cfg.cluster_consolidation is not None:
                    # Clustered training reads per-feature statistics on the
                    # SIBLING original ids and only then max-folds a cluster
                    # into its representative, so a representative row is
                    # available iff ANY mask-surviving member has statistics.
                    rep_of = cfg.cluster_consolidation.rep_of.to("cpu")
                    counts = torch.zeros(int(cfg.n_sae_features), dtype=torch.int64)
                    counts.index_add_(0, rep_of, available.to(torch.int64))
                    available = counts > 0
                if not bool(available.any()):
                    raise ValueError(
                        "every per-feature threshold in "
                        "precomputed_thresholds is +inf, so no feature has "
                        "statistics and this provider could never select "
                        "anything. The thresholds were most likely built from "
                        "a statistics artifact that covers none of this "
                        "model's features (wrong feature_dir, empty or "
                        "mismatched per-feature statistics); for a "
                        "cluster-derived checkpoint also pass the matching "
                        "SelfCondConfig.cluster_consolidation."
                    )
                self._selection_percentile_thresholds[available] = target_cutoff
        self._bucket_percentile_keys: tuple[int, ...] = ()
        self._bucket_percentile_cutpoints = torch.empty(0, dtype=torch.float32)
        if cfg.bucket_producer is FeatureBucketProducer.RAW_STATS_BOUNDARIES:
            self._bucket_percentile_keys = bucket_percentile_key_grid(
                cfg.num_buckets,
                threshold_percentile=threshold_percentile,
            )
            # ``magnitude_to_percentile`` returns an integer rank for exact
            # p1..p98 boundaries, but its right-inclusive endpoint maps exact
            # p99 to 99.5. Self-conditioning keeps every cutoff in one
            # continuous normalized-rank currency.
            bucket_target_ranks = tuple(
                99.5 if key == 99 else float(key) for key in self._bucket_percentile_keys
            )
            self._bucket_percentile_cutpoints = (
                torch.tensor(
                    bucket_target_ranks,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                / 100.0
            )

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def validate_explicit_segment(self, features: Mapping[int, int]) -> dict[int, int]:
        """Validate an explicit segment WITHOUT arming it; return the snapshot.

        Pure with respect to provider state: callers (the chat session's
        ``modify``) run this BEFORE tearing down or replaying anything, so an
        invalid edit can never destroy conversation state. Snapshot FIRST so
        validation and the returned value cover the same mapping even if the
        caller mutates ``features`` concurrently.
        """

        snapshot = dict(features)
        if not snapshot:
            raise ValueError("explicit segment requires at least one feature")

        # The strict shared validator: beartype rejects non-int keys/values
        # (bools and floats included), the body range-checks ids and buckets,
        # and a compact vocabulary rejects excluded ids with FeatureVocabError.
        # The bucket bound is the ENCODER's full bucket count — an explicit
        # inactive bucket is legal even when cfg.num_buckets == 1.
        validate_feature_dict(
            snapshot,
            PositiveInt.parse(int(self.cfg.n_sae_features)),
            PositiveInt.parse(int(self._model.encoder.config.num_buckets)),
            name="explicit segment",
            feature_vocab=self._model.feature_vocab,
        )

        ids = torch.tensor(
            sorted(snapshot),
            dtype=torch.int64,
            device=self._device,
        )
        if self.cfg.feature_mask is not None:
            self.cfg.feature_mask.assert_no_masked(ids)
        if self.cfg.cluster_consolidation is not None:
            self.cfg.cluster_consolidation.assert_representatives_only(
                ids,
                surface="explicit segment",
            )
        return snapshot

    def queue_explicit_segment(self, features: Mapping[int, int]) -> None:
        """Queue one validated segment to replace the next live selection."""

        if self._queued_explicit is not None:
            raise RuntimeError("explicit segment override is already queued")
        self._queued_explicit = self.validate_explicit_segment(features)

    def discard_explicit_segment(self) -> None:
        """Discard any queued explicit segment override, if one exists."""

        self._queued_explicit = None

    def seed_from_prompt(
        self,
        state: SelfCondState,
        prompt_fp_logits: torch.Tensor,  # (1, n_prompt, fp_width)
    ) -> SelfCondState:
        """Pool every explicitly supplied row from the final real prompt slot.

        Prompt segmentation is defined by the extractor and may contain more
        nanochat tokens than the generated-token cadence K.  The caller owns the
        exact source-row mask; this method therefore requires only a non-empty
        time dimension and deliberately applies no K upper bound.
        """

        return self._append_from_fp_logits(
            state,
            prompt_fp_logits,
            owner="prompt_fp_logits",
            max_time=None,
        )

    def step(
        self,
        state: SelfCondState,
        fp_logits_tail: torch.Tensor,  # (1, T, fp_width) where T<=K
    ) -> SelfCondState:
        """Append a slot from one strict generated-token window of at most K."""

        return self._append_from_fp_logits(
            state,
            fp_logits_tail,
            owner="fp_logits_tail",
            max_time=int(self.cfg.self_cond_segment_size),
        )

    def append_tool_window(self, state: SelfCondState, fp_logits: torch.Tensor) -> SelfCondState:
        """Pool one complete source-owned tool window; assistant steps stay K-bounded."""
        if self.cfg.segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
            raise ValueError("tool windows require turn self-conditioning")
        return self._append_from_fp_logits(
            state, fp_logits, owner="tool_fp_logits", max_time=self._model.dec_config.sequence_len,
        )

    def _append_from_fp_logits(
        self,
        state: SelfCondState,
        fp_logits_tail: torch.Tensor,
        *,
        owner: str,
        max_time: int | None,
    ) -> SelfCondState:
        """Shared pool/select/bucket/encode/append implementation."""

        _require_nonempty_self_cond_state(state, owner=owner)

        if fp_logits_tail.ndim != 3:
            raise ValueError(
                f"{owner} must have shape (1, T, fp_width), got {tuple(fp_logits_tail.shape)}"
            )
        if fp_logits_tail.shape[0] != 1:
            raise ValueError(
                "SelfConditionedFeatureProvider supports batch size 1; "
                f"{owner} has batch size {fp_logits_tail.shape[0]}"
            )
        tail_length = int(fp_logits_tail.shape[1])
        if tail_length < 1 or (max_time is not None and tail_length > max_time):
            expected = "at least one" if max_time is None else f"in [1, {max_time}]"
            raise ValueError(f"{owner} time dimension must be {expected}, got {tail_length}")
        expected_width = (
            int(self.cfg.n_sae_features)
            if self._compact_candidate_feature_ids is None
            else int(self._compact_candidate_feature_ids.shape[0])
        )
        fp_width = int(fp_logits_tail.shape[-1])
        if fp_width != expected_width:
            raise ValueError(
                f"{owner} has width {fp_width}, but this provider pairs "
                f"scores with {expected_width} candidate original feature ids. "
                "Use logits from this checkpoint's feature-prediction head."
            )
        if not torch.isfinite(fp_logits_tail).all():
            raise ValueError(
                f"{owner} contains non-finite values (NaN/Inf). "
                "This usually indicates a numerics regression in the "
                "decoder forward pass."
            )

        queued = self._queued_explicit
        if queued is not None:
            self._queued_explicit = None
            with timed_region("selfcond/provider/materialize_segment", sync_cuda=False):
                fids_t, mags_t, mask_t = self._build_segment_inputs(queued)
            with timed_region("selfcond/provider/encode_segment", sync_cuda=False):
                new_encoder_out = self._encode_one_segment(
                    fids_t,
                    mags_t,
                    mask_t,
                )
            with timed_region("selfcond/provider/append_segment", sync_cuda=False):
                merged_out = append_packed_encoder_segment(
                    state.encoder_out,
                    new_encoder_out,
                )
            new_state = SelfCondState(
                encoder_out=merged_out,
                target_slot_idx=state.target_slot_idx + 1,
            )
            self.last_appended_segment = dict(queued)
            return new_state

        if isinstance(self.cfg.score_policy, SelfCondScoreInterpolation):
            with timed_region("selfcond/provider/select_features", sync_cuda=False):
                return self._step_score_interpolation(state, fp_logits_tail, owner=owner)
        if isinstance(self.cfg.score_policy, SelfCondScoreMasking):
            with timed_region("selfcond/provider/select_features", sync_cuda=False):
                return self._step_score_masking(state, fp_logits_tail, owner=owner)

        with timed_region("selfcond/provider/pool_and_prepare", sync_cuda=False):
            # 1. Pool over the time dimension
            pooled = pool_fp_logits(fp_logits_tail, self.cfg.pool_mode)
            # pooled: (1, fp_width)

            # 2. Sigmoid → percentiles (cast to fp32 first to keep numerics
            # stable across bf16 inference paths)
            pooled_f32 = pooled.detach().to(torch.float32)
            pooled_f32 = self._apply_fp_logit_bias(pooled_f32)
            percentiles = torch.sigmoid(pooled_f32)  # (1, fp_width)

        with timed_region("selfcond/provider/select_features", sync_cuda=False):
            # 3. Selection: pair each head score with its ORIGINAL SAE id. The
            # uncompacted branch is the exact legacy arange; compact head row j is
            # translated to feature_vocab.original_ids[j].
            if self._compact_candidate_feature_ids is None:
                feature_ids = torch.arange(
                    self.cfg.n_sae_features,
                    dtype=torch.int32,
                    device=percentiles.device,
                )
            else:
                feature_ids = self._compact_feature_ids_on(percentiles.device)
            scores_1d = percentiles[0]  # (fp_width,)

            # Select directly over the full predicted universe; the mask, when set,
            # is applied inside _apply_selection.
            kept_ids, kept_pcts = self._apply_selection(feature_ids, scores_1d)

            # Defense-in-depth (H-assert): assert finalized selected ids contain no
            # masked features. Cheap: boolean membership over kept_ids (tens–hundreds).
            # Strictly mask-gated (I5: no-op when feature_mask is None).
            if self.cfg.feature_mask is not None:
                self.cfg.feature_mask.assert_no_masked(kept_ids.to(torch.int64))

        with timed_region("selfcond/provider/assign_buckets", sync_cuda=False):
            # 4. Bucket assignment through the explicit conditioning producer:
            # uniform normalized scores or translated raw statistics keys.
            kept_buckets = self._bucket_percentiles_like_training(kept_pcts)

        with timed_region("selfcond/provider/materialize_segment", sync_cuda=False):
            # 5. Build the standalone encoder inputs. The device-to-host list
            # conversion is intentionally inside this marker: it is a candidate
            # synchronization cost that the profiler must make visible.
            new_segment = {
                int(fid): int(bucket)
                for fid, bucket in zip(
                    kept_ids.tolist(),
                    kept_buckets.tolist(),
                    strict=True,
                )
            }
            if not new_segment:
                raise EmptySelfConditioningSegmentError(
                    f"{owner} selected zero features under "
                    f"{type(self.cfg.selection).__name__}; refusing to append an "
                    "out-of-distribution empty encoder segment"
                )
            fids_t, mags_t, mask_t = self._build_segment_inputs(new_segment)

        with timed_region("selfcond/provider/encode_segment", sync_cuda=False):
            new_encoder_out = self._encode_one_segment(
                fids_t,
                mags_t,
                mask_t,
            )

        # 6. Append one packed segment and rebuild its owner coordinates.
        with timed_region("selfcond/provider/append_segment", sync_cuda=False):
            merged_out = append_packed_encoder_segment(
                state.encoder_out,
                new_encoder_out,
            )

        new_state = SelfCondState(
            encoder_out=merged_out,
            target_slot_idx=state.target_slot_idx + 1,
        )
        self.last_appended_segment = dict(new_segment)
        return new_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _step_score_interpolation(
        self,
        state: SelfCondState,
        fp_logits_tail: torch.Tensor,
        *,
        owner: str,
    ) -> SelfCondState:
        """Run endpoint-aware model/target scoring and stable two-sided selection."""

        interpolation = self.cfg.score_policy
        target_scores = self._target_scores
        if not isinstance(interpolation, SelfCondScoreInterpolation) or target_scores is None:
            raise RuntimeError("score-interpolation provider state is incomplete")
        if not isinstance(self.cfg.selection, SamplingBundle):
            raise RuntimeError("score-interpolation positive selection is not sampling")

        pooled = pool_fp_logits(fp_logits_tail, self.cfg.pool_mode)
        pooled_f32 = pooled.detach().to(torch.float32)
        weight = float(interpolation.weight)
        if weight == 0.0:
            # At the pure-model endpoint only ordering matters. Rank logits
            # directly so sigmoid endpoint saturation cannot create false ties.
            scores = pooled_f32
        elif weight == 1.0:
            scores = target_scores.unsqueeze(0)
        else:
            q = 2.0 * torch.sigmoid(pooled_f32) - 1.0
            scores = torch.lerp(q, target_scores.unsqueeze(0), weight)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("interpolated SelfConditioning scores contain non-finite values")

        width = int(scores.shape[1])
        if self._compact_candidate_feature_ids is None:
            candidate_ids = torch.arange(
                width,
                dtype=torch.int64,
                device=scores.device,
            )
        else:
            candidate_ids = self._compact_feature_ids_on(scores.device).to(torch.int64)
        scores_1d = scores[0]
        if self.cfg.feature_mask is not None:
            keep = self.cfg.feature_mask.keep_mask(candidate_ids).to(device=candidate_ids.device)
            candidate_ids = candidate_ids[keep]
            scores_1d = scores_1d[keep]

        rng = self.cfg.selection.rng
        sampled_k, sampled_l = sample_action_counts(
            self.cfg.selection.selection,
            interpolation.negative_selection,
            rng,
            1,
        )
        actions = select_topk_bottoml(
            scores_1d.to(rng.device).unsqueeze(0),
            sampled_k,
            sampled_l,
        )
        positive_count = int(actions.k[0].item())
        negative_count = int(actions.l[0].item())
        row_candidates = candidate_ids.to(rng.device)
        positive_rows = actions.pos_ids[0, :positive_count]
        negative_rows = actions.neg_ids[0, :negative_count]
        positive_ids = row_candidates[positive_rows].to(torch.int64)
        negative_ids = row_candidates[negative_rows].to(torch.int64)
        return self._finalize_two_sided_selection(
            state,
            positive_ids,
            negative_ids,
            data_owner=owner,
            policy_owner="score_interpolation",
        )

    def _finalize_two_sided_selection(
        self,
        state: SelfCondState,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        *,
        data_owner: str,
        policy_owner: str,
    ) -> SelfCondState:
        """Shared two-sided finalization: defenses, segment, encode, append.

        Both dense policies end identically once their positive/negative
        original-id selections exist; keeping one tail means a fix to the
        defense checks, the empty-segment refusal, or the encode/concat
        sequence cannot drift between them.
        """

        policy_phrase = policy_owner.replace("_", " ")
        selected_ids = torch.cat([positive_ids, negative_ids])
        if selected_ids.unique().numel() != selected_ids.numel():
            raise RuntimeError(f"{policy_owner} selected duplicate or overlapping feature ids")
        if selected_ids.numel() and bool(
            ((selected_ids < 0) | (selected_ids >= int(self.cfg.n_sae_features))).any()
        ):
            raise RuntimeError(f"{policy_owner} selected an original SAE id outside the dictionary")
        if self.cfg.feature_mask is not None:
            self.cfg.feature_mask.assert_no_masked(selected_ids)
        if self.cfg.cluster_consolidation is not None:
            self.cfg.cluster_consolidation.assert_representatives_only(
                selected_ids,
                surface=f"SelfConditioning {policy_phrase}",
            )

        with timed_region("selfcond/provider/materialize_segment", sync_cuda=False):
            new_segment = {
                **{int(feature_id): 0 for feature_id in positive_ids.tolist()},
                **{int(feature_id): 1 for feature_id in negative_ids.tolist()},
            }
            if not new_segment:
                raise EmptySelfConditioningSegmentError(
                    f"{data_owner} {policy_phrase} selected zero positive and zero "
                    "negative features; refusing to append an out-of-distribution "
                    "empty encoder segment"
                )
            fids_t, mags_t, mask_t = self._build_segment_inputs(new_segment)
        with timed_region("selfcond/provider/encode_segment", sync_cuda=False):
            new_encoder_out = self._encode_one_segment(
                fids_t,
                mags_t,
                mask_t,
            )
        with timed_region("selfcond/provider/append_segment", sync_cuda=False):
            merged_out = append_packed_encoder_segment(
                state.encoder_out,
                new_encoder_out,
            )
        new_state = SelfCondState(
            encoder_out=merged_out,
            target_slot_idx=state.target_slot_idx + 1,
        )
        self.last_appended_segment = dict(new_segment)
        return new_state

    def _step_score_masking(
        self,
        state: SelfCondState,
        fp_logits_tail: torch.Tensor,
        *,
        owner: str,
    ) -> SelfCondState:
        """Rank by pooled model logits inside sign-consistent target pools."""

        masking = self.cfg.score_policy
        positive_rows = self._masking_positive_rows
        negative_rows = self._masking_negative_rows
        if (
            not isinstance(masking, SelfCondScoreMasking)
            or positive_rows is None
            or negative_rows is None
        ):
            raise RuntimeError("score-masking provider state is incomplete")
        if not isinstance(self.cfg.selection, SamplingBundle):
            raise RuntimeError("score-masking positive selection is not sampling")

        pooled = pool_fp_logits(fp_logits_tail, self.cfg.pool_mode)
        pooled_f32 = pooled.detach().to(torch.float32)
        # The dense target defines eligibility, while the model score is used
        # only for ordering within each pool. Raw logits preserve that ordering
        # without sigmoid saturation ties.
        scores = pooled_f32

        width = int(scores.shape[1])
        if self._compact_candidate_feature_ids is None:
            candidate_ids = torch.arange(
                width,
                dtype=torch.int64,
                device=scores.device,
            )
        else:
            candidate_ids = self._compact_feature_ids_on(scores.device).to(torch.int64)
        scores_1d = scores[0]
        positive_eligible = positive_rows.to(device=scores.device)
        negative_eligible = negative_rows.to(device=scores.device)

        rng = self.cfg.selection.rng
        sampled_k, sampled_l = sample_action_counts(
            self.cfg.selection.selection,
            masking.negative_selection,
            rng,
            1,
        )
        positive_pool = int(positive_eligible.sum().item())
        negative_pool = int(negative_eligible.sum().item())
        sampled_positive_count = int(sampled_k[0].item())
        sampled_negative_count = int(sampled_l[0].item())
        positive_count = min(sampled_positive_count, positive_pool)
        negative_count = min(sampled_negative_count, negative_pool)
        if sampled_positive_count != positive_count:
            positive_log = logger.warning if positive_pool == 0 else logger.info
            positive_log(
                "SelfConditioning score masking positive count: configured pool "
                "limit={}, actual ranked pool size={}, sampled count={}, clamped count={}",
                int(masking.positive_pool_limit),
                positive_pool,
                sampled_positive_count,
                positive_count,
            )
        if sampled_negative_count != negative_count:
            negative_log = logger.warning if negative_pool == 0 else logger.info
            negative_log(
                "SelfConditioning score masking negative count: configured pool "
                "limit={}, actual ranked pool size={}, sampled count={}, clamped count={}",
                int(masking.negative_pool_limit),
                negative_pool,
                sampled_negative_count,
                negative_count,
            )

        scores_rng = scores_1d.to(rng.device)
        row_candidates = candidate_ids.to(rng.device)
        positive_eligible_rng = positive_eligible.to(rng.device)
        negative_eligible_rng = negative_eligible.to(rng.device)
        positive_actions = select_topk_bottoml(
            scores_rng[positive_eligible_rng].unsqueeze(0),
            torch.tensor([positive_count], dtype=torch.int64, device=rng.device),
            torch.tensor([0], dtype=torch.int64, device=rng.device),
        )
        negative_actions = select_topk_bottoml(
            scores_rng[negative_eligible_rng].unsqueeze(0),
            torch.tensor([0], dtype=torch.int64, device=rng.device),
            torch.tensor([negative_count], dtype=torch.int64, device=rng.device),
        )
        positive_ids = row_candidates[positive_eligible_rng][
            positive_actions.pos_ids[0, :positive_count]
        ].to(torch.int64)
        negative_ids = row_candidates[negative_eligible_rng][
            negative_actions.neg_ids[0, :negative_count]
        ].to(torch.int64)
        return self._finalize_two_sided_selection(
            state,
            positive_ids,
            negative_ids,
            data_owner=owner,
            policy_owner="score_masking",
        )

    def _compact_feature_ids_on(self, device: torch.device) -> torch.Tensor:
        """Return the compact head's aligned original-id vector on ``device``."""
        feature_ids = self._compact_candidate_feature_ids
        if feature_ids is None:
            raise RuntimeError("_compact_feature_ids_on called for an uncompacted checkpoint")
        if feature_ids.device != device:
            feature_ids = feature_ids.to(device)
            self._compact_candidate_feature_ids = feature_ids
        return feature_ids

    def _apply_fp_logit_bias(self, pooled_f32: torch.Tensor) -> torch.Tensor:
        """Add the private head-row bias without mutating the pooled logits."""
        bias = self._fp_logit_bias
        if bias is None:
            # This identity return pins the default path: no allocation,
            # conversion, arithmetic, tensor version change, or RNG use.
            return pooled_f32
        if bias.device != pooled_f32.device:
            raise RuntimeError(
                "fp_logit_bias and pooled feature-prediction logits are on "
                f"different devices ({bias.device} vs {pooled_f32.device}); "
                "construct the provider on the model's inference device"
            )
        return pooled_f32 + bias

    def _bucket_percentiles_like_training(
        self,
        percentiles: torch.Tensor,
    ) -> torch.Tensor:
        """Search training's exact statistics-key grid in continuous rank space.

        Uniform-percentile bucket producers use the historical
        ``floor(score * B)`` rule. Certified packed pretraining belongs to this
        branch, so it remains exactly compatible with existing compact and
        full packed checkpoints.

        Raw-stat bucket producers instead use raw statistics-key boundaries.
        Construction maps that key grid directly to continuous normalized rank
        before searching. A score equal to its boundary enters the higher bucket,
        exactly like raw-magnitude ``searchsorted(..., right=True)``. Float32
        scores can still alias at extremely close ranks; right-inclusive ties use
        the higher bucket.
        """
        if self.cfg.bucket_producer is FeatureBucketProducer.UNIFORM_PERCENTILE:
            return percentile_to_bucket(percentiles, self.cfg.num_buckets)
        if self._bucket_percentile_cutpoints.numel() == 0:
            return percentiles.new_zeros(percentiles.shape, dtype=torch.int64)
        cutpoints = self._bucket_percentile_cutpoints.to(
            dtype=percentiles.dtype,
            device=percentiles.device,
        )
        return torch.searchsorted(
            cutpoints,
            percentiles.contiguous(),
            right=True,
        )

    def _apply_selection(
        self,
        feature_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch percentile-space selection to the configured pipeline.

        Returns ``(kept_feature_ids, kept_percentile_scores)``. The result
        of either branch is the set of features whose predicted activation
        survives selection, paired with their predicted percentiles
        (which the caller converts to bucket indices).

        For threshold-mode selection we call ``apply_threshold`` with the FP
        score vector in place of magnitudes. Per-feature cutoffs come from the
        provider's canonical tensor: construction preserves the raw training
        table's finite/``+inf`` availability in original-id space, then derives
        each finite score cutoff from the explicit selection producer in
        continuous normalized-rank space.

        The feature-mask filter (when a mask is set) restricts the candidate
        universe to allowed ids BEFORE selection, so disallowed features cannot
        enter regardless of the selection count (segment_topk with
        ``k > |allowed|`` would otherwise pick disallowed features at any score).
        It is applied VALUE-based via :meth:`FeatureMask.keep_mask`, so it works
        for any ``feature_ids`` subset while preserving the no-mask path.

        Subset-safety is the compact-checkpoint contract: compact logits arrive
        paired with effective-vocabulary ORIGINAL ids, and per-feature tables
        continue to be indexed by those ids rather than by compact row number.
        """
        if self.cfg.feature_mask is not None:
            # Value-based, subset-safe filter. On the full universe this equals the
            # positional allowed_universe_mask filter.
            keep = self.cfg.feature_mask.keep_mask(feature_ids).to(device=feature_ids.device)
            feature_ids = feature_ids[keep]
            scores = scores[keep]

        sel = self.cfg.selection
        if isinstance(sel, ThresholdSelectionConfig):
            threshold_cfg = ThresholdConfig(
                threshold_type=sel.selection.threshold_type,
                threshold=sel.selection.threshold,
            )
            pct_precomputed = self._selection_percentile_thresholds
            if isinstance(pct_precomputed, torch.Tensor):
                # The canonical tensor is float32 in the head's exact target
                # currency. ``apply_feature_percentile`` promotes scores to
                # float64, but the represented cutoff remains byte-identical.
                pct_precomputed = pct_precomputed.to(device=scores.device)

            return apply_threshold(
                threshold_cfg,
                feature_ids,
                scores,
                pct_precomputed,
            )
        # SamplingBundle: drive sample_features with predicted percentiles
        # as the weight signal (override_weights skips the percentile_table
        # lookup since we already have percentiles).
        #
        # Device alignment: ``sample_features`` requires the rng's device
        # to match the weight tensors. Bundle RNGs are conventionally
        # constructed on CPU, but scores live on the model device. Sample
        # on CPU then transfer ids/scores back to the model device.
        rng_device = sel.rng.device
        feature_ids_rng = feature_ids.to(rng_device)
        scores_rng = scores.to(rng_device)
        kept_ids, kept_scores = sample_features(
            feature_ids=feature_ids_rng,
            feature_mags=scores_rng,
            percentile_table=sel.percentile_table,
            config=sel.selection,
            rng=sel.rng,
            override_weights=scores_rng,
        )
        return kept_ids.to(scores.device), kept_scores.to(scores.device)

    def _build_segment_inputs(
        self,
        features: dict[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (1, 1, F) encoder inputs for one segment."""
        if not features:
            raise EmptySelfConditioningSegmentError(
                "self-conditioning encoder inputs require at least one feature"
            )
        fid_list = list(features.keys())
        bucket_list = [features[f] for f in fid_list]
        F = len(fid_list)
        fids = torch.zeros((1, 1, F), dtype=torch.long, device=self._device)
        mags = torch.zeros((1, 1, F), dtype=torch.long, device=self._device)
        mask = torch.zeros((1, 1, F), dtype=torch.bool, device=self._device)
        if fid_list:
            fids[0, 0, : len(fid_list)] = torch.tensor(
                fid_list,
                dtype=torch.long,
                device=self._device,
            )
            mags[0, 0, : len(bucket_list)] = torch.tensor(
                bucket_list,
                dtype=torch.long,
                device=self._device,
            )
            mask[0, 0, : len(fid_list)] = True
        return fids, mags, mask

    @torch.no_grad()
    def _encode_one_segment(
        self,
        fids: torch.Tensor,
        mags: torch.Tensor,
        mask: torch.Tensor,
    ) -> PackedEncoderOutput:
        """Encode one non-empty segment as batch-one packed state."""
        if not bool(mask.any()):
            raise EmptySelfConditioningSegmentError(
                "refusing to invoke the encoder for an all-masked self-conditioning segment"
            )
        encoder_out = self._model.encoder(fids, mags, mask)
        if not isinstance(encoder_out, PackedEncoderOutput):
            raise TypeError(
                "self-conditioning encoder must return PackedEncoderOutput, got "
                f"{type(encoder_out).__name__}"
            )
        layout = encoder_out.layout
        if layout.batch_size != 1 or layout.n_segments != 1:
            raise ValueError(
                "self-conditioning segment encoder must return a (batch=1, segments=1) "
                f"layout, got {(layout.batch_size, layout.n_segments)}"
            )
        if not bool(layout.valid_mask.any()):
            raise EmptySelfConditioningSegmentError(
                "the encoder returned an all-masked self-conditioning segment"
            )
        return encoder_out
