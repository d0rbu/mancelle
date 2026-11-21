"""On-policy self-conditioning post-training primitives.

This module keeps rollout policy mechanics separate from the training script:
per-segment FP-logit pooling, deterministic action-count draws, and disjoint
top/bottom feature selection live here so their contracts are directly testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch as t
from beartype import beartype

from errors import SelfCondTrainingError
from nanochat.common import get_dist_info
from nanochat.gpt import OptimizerKind
from nanochat.optim import DistMuonAdamW, MuonAdamW
from steering.conditional.checkpoint_conditioning import (
    assert_conditioning_supported,
    read_checkpoint_conditioning,
)
from steering.conditional.delta import DeltaMode
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm, SteInputs
from steering.conditional.encoder_decoder.feature_prediction import (
    last_token_mask,
    pool_segment_amax,
    segment_lengths,
    source_segment_masks,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingSelection,
    SelectedActions,
    ThresholdSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.sampling import (
    sample_action_counts as _sample_action_counts,
)
from steering.conditional.sampling import (
    select_topk_bottoml as _select_topk_bottoml,
)
from steering.profiling.markers import timed_region, timing_tags
from steering.refined_types import PositiveInt

# Keep the validation data stream and rollout stream on distinct step seeds.
VAL_DATA_SEED_SENTINEL = 2**31 - 2
VAL_ROLLOUT_STEP_SENTINEL = 2**31 - 1


class SelfCondDataSource(StrEnum):
    """Data family consumed by self-conditioning post-training."""

    PRETRAINING = "pretraining"
    SFT = "sft"


class SelfCondTrainScope(StrEnum):
    """Parameters updated by a self-conditioning training step."""

    HEAD_ONLY = "head_only"
    FULL = "full"


@beartype
@dataclass(frozen=True)
class SelfCondTrainConfig:
    """Resolved policy and estimator settings for self-conditioning training.

    The beartyped selection constructors own distribution, scalar-domain, and
    finite-range validity. This boundary retains only the self-conditioning
    requirement that both ranges have a finite upper bound.
    """

    pool_mode: FeaturePredLossMode
    train_scope: SelfCondTrainScope
    ste_form: SteForm
    positive_counts: SamplingSelection
    negative_counts: NegativeSamplingSelection
    seed: int

    def __post_init__(self) -> None:
        if self.pool_mode is FeaturePredLossMode.BEST_TOKEN:
            raise SelfCondTrainingError(
                "self-conditioning pool_mode=best_token is invalid because the "
                "policy does not have the next segment's target features"
            )
        for label, counts in (
            ("positive_counts", self.positive_counts),
            ("negative_counts", self.negative_counts),
        ):
            if counts.max_k is None:
                raise SelfCondTrainingError(
                    f"{label}.max_k must be non-null for a static self-conditioning layout"
                )


def selfcond_step_seed(seed: int, ddp_rank: int, global_step: int) -> int:
    """Return the pinned stateless action seed for one optimizer step."""
    return ((seed * 1_000_003 + ddp_rank) * 1_000_003 + global_step) % 2**63


def resolve_selfcond_count_configs(
    *,
    feature_selection: dict,
    negative_selection: dict | None,
    positive_override: dict | None,
    negative_override: dict | None,
) -> tuple[SamplingSelection, NegativeSamplingSelection]:
    """Resolve count distributions with checkpoint inheritance and overrides."""
    if negative_selection is None:
        raise SelfCondTrainingError(
            "self-conditioning requires checkpoint negative_selection metadata; "
            "bucket 1 is not known to mean negative without it"
        )

    if positive_override is not None:
        parsed_positive = parse_feature_selection(positive_override)
        if not isinstance(parsed_positive, SamplingSelection):
            raise SelfCondTrainingError("selfcond.positive_counts must be a sampling distribution")
        positive = parsed_positive
    else:
        inherited_positive = parse_feature_selection(feature_selection)
        if isinstance(inherited_positive, ThresholdSelection):
            raise SelfCondTrainingError(
                "checkpoint feature_selection is threshold-based and carries NO "
                "positive count distribution to inherit — there is nothing safe to "
                "assume in its place.\n"
                "Set selfcond.positive_counts explicitly, e.g.:\n"
                "  selfcond.positive_counts='{distribution: geometric, mean: 12.0, "
                "std: 12.0, min_k: 1, max_k: 49}'"
            )
        if not isinstance(inherited_positive, SamplingSelection):
            # e.g. a hand-corrupted meta whose feature_selection parses to None:
            # without this the bad value would surface later as a bare
            # AttributeError deep inside count validation instead of a typed
            # error naming the actual problem (review finding).
            raise SelfCondTrainingError(
                "checkpoint feature_selection did not parse to a sampling "
                f"distribution (got {type(inherited_positive).__name__}); set "
                "selfcond.positive_counts explicitly or repair the checkpoint meta"
            )
        positive = inherited_positive

    negative_raw = negative_override if negative_override is not None else negative_selection
    negative = parse_negative_selection(negative_raw)
    assert negative is not None
    return positive, negative


@beartype
def pool_fp_logits_per_segment(
    fp_logits: t.Tensor,
    seg_map: t.Tensor,
    n_seg: PositiveInt,
    mode: FeaturePredLossMode,
) -> t.Tensor:
    """Pool ``(B,T,V)`` FP logits into the batch-global segment axis.

    ``@beartype`` owns the unary positive segment-count contract; tensor shape
    and pooling-mode relationships remain explicit below.
    """
    if fp_logits.ndim != 3:
        raise SelfCondTrainingError(
            f"fp_logits must have shape (B, T, V), got {tuple(fp_logits.shape)}"
        )
    batch_size, time, n_features = fp_logits.shape
    if tuple(seg_map.shape) != (batch_size, time):
        raise SelfCondTrainingError(
            f"seg_map shape must be {(batch_size, time)}, got {tuple(seg_map.shape)}"
        )
    if time == 0:
        raise SelfCondTrainingError("per-segment pooling requires at least one token")

    segment_index = seg_map.unsqueeze(-1).expand(batch_size, time, n_features)
    if mode is FeaturePredLossMode.MAX_POOL:
        pooled = pool_segment_amax(fp_logits, seg_map, n_seg)
        # Zero ONLY the -inf empty-segment sentinel (a slot no token mapped to).
        # A +inf pooled value is a REAL diverged head logit and must propagate
        # so the callers' finiteness guard raises instead of silently ranking
        # an exploded feature at sigmoid(0)=0.5 (review finding).
        return t.where(t.isneginf(pooled), t.zeros_like(pooled), pooled)

    if mode is FeaturePredLossMode.FULL_SEGMENT:
        # Mean pooling is selfcond-specific (the FP loss has no sum-pool
        # counterpart) and its lengths deliberately follow fp_logits.dtype —
        # NOT the shared float32 segment_lengths — so the returned scores keep
        # the head's dtype under autocast instead of promoting to float32.
        pooled = t.zeros(
            batch_size,
            n_seg,
            n_features,
            dtype=fp_logits.dtype,
            device=fp_logits.device,
        )
        pooled.scatter_add_(1, segment_index, fp_logits)
        lengths = t.zeros(
            batch_size,
            n_seg,
            dtype=fp_logits.dtype,
            device=fp_logits.device,
        )
        lengths.scatter_add_(
            1,
            seg_map,
            t.ones(batch_size, time, dtype=fp_logits.dtype, device=fp_logits.device),
        )
        return pooled / lengths.clamp(min=1).unsqueeze(-1)

    if mode is FeaturePredLossMode.LAST_TOKEN:
        is_last = last_token_mask(seg_map)
        pooled = t.zeros(
            batch_size,
            n_seg,
            n_features,
            dtype=fp_logits.dtype,
            device=fp_logits.device,
        )
        pooled.scatter_add_(
            1,
            segment_index,
            t.where(is_last.unsqueeze(-1), fp_logits, t.zeros_like(fp_logits)),
        )
        return pooled

    if mode is FeaturePredLossMode.BEST_TOKEN:
        raise SelfCondTrainingError(
            "self-conditioning cannot pool best_token logits without target features"
        )
    raise SelfCondTrainingError(f"unsupported self-conditioning pool mode: {mode!r}")


def assert_finite_fp_logits(fp_logits: t.Tensor, *, where: str) -> None:
    """Raise loudly on NaN/Inf head logits (numerics regressions must not be
    silently absorbed into rollout scores). Mirrors the inference provider's
    contract in ``self_conditioned.SelfConditionedFeatureProvider.step``."""
    if not bool(t.isfinite(fp_logits).all()):
        raise SelfCondTrainingError(
            f"fp_logits contain non-finite values (NaN/Inf) during {where}. "
            "This usually indicates a numerics regression or a diverged "
            "feature-prediction head; refusing to select actions from "
            "corrupted scores."
        )


def resolve_recorded_num_buckets(meta: Mapping[str, Any]) -> int:
    """Read the checkpoint's recorded bucket count — the ONE precedence rule.

    ``user_config.num_buckets`` wins over top-level ``num_buckets``; a
    checkpoint recording neither raises (C3 must gate on a recorded fact,
    never a default). Both the training script and the C3 validation delegate
    here so the precedence cannot drift between copies (review finding).
    """
    raw = _meta_value_null_falls_through(meta, "num_buckets")
    if raw is None:
        raise SelfCondTrainingError(
            "C3: num_buckets is absent from checkpoint metadata; self-conditioning "
            "cannot infer bucket semantics from a default"
        )
    if not isinstance(raw, (bool, int, float, str)):
        raise SelfCondTrainingError(
            f"C3: recorded num_buckets has unusable type {type(raw).__name__}"
        )
    # int() must not escape this gate's typed-error contract (a corrupted
    # string like "two" raised a bare ValueError), and a non-integral float
    # like 2.9 must not silently truncate to 2 — both are corrupted metadata
    # and the whole point of this resolver is to fail loudly on them
    # (review finding).
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise SelfCondTrainingError(
            f"C3: recorded num_buckets {raw!r} is not an integer"
        ) from error
    if isinstance(raw, float) and raw != value:
        raise SelfCondTrainingError(
            f"C3: recorded num_buckets {raw!r} is not integral; refusing to truncate"
        )
    return value


def sample_action_counts(
    cfg: SelfCondTrainConfig,
    rng: t.Generator,
    batch_size: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Draw positive counts first and negative counts second (R5)."""
    return _sample_action_counts(
        cfg.positive_counts,
        cfg.negative_counts,
        rng,
        batch_size,
    )


@runtime_checkable
class ConditioningBatch(Protocol):
    """Structural tensor subset needed to rebuild encoder conditioning."""

    @property
    def inputs(self) -> t.Tensor: ...

    @property
    def targets(self) -> t.Tensor: ...

    @property
    def enc_fids(self) -> t.Tensor: ...

    @property
    def enc_mags(self) -> t.Tensor: ...

    @property
    def enc_mask(self) -> t.Tensor: ...

    @property
    def seg_map(self) -> t.Tensor: ...

    @property
    def seg_doc_id(self) -> t.Tensor: ...


@dataclass(frozen=True)
class SelfCondBatch:
    """Source-agnostic tensor subset the rollout/loss consume (issue #1438).

    ``from_sft`` deliberately drops ``fp_targets``: the FP head is trained
    through the rollout STE in this stage, never by a supervised FP loss.
    ``targets == -1`` masking passes through untouched.
    """

    inputs: t.Tensor
    targets: t.Tensor
    enc_fids: t.Tensor
    enc_mags: t.Tensor
    enc_mask: t.Tensor
    seg_map: t.Tensor
    seg_doc_id: t.Tensor

    @classmethod
    def from_sft(cls, batch: tuple | list) -> SelfCondBatch:
        """Normalize the explicit eight-item SFT generator contract."""
        if not isinstance(batch, (tuple, list)):
            raise SelfCondTrainingError(
                f"SelfCondBatch.from_sft expected an SFT batch tuple, got {type(batch).__name__}"
            )
        if len(batch) != 8:
            raise SelfCondTrainingError(
                "SelfCondBatch.from_sft expected the 8-item SFT batch "
                "(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, "
                f"fp_targets, seg_doc_id); got {len(batch)} items. "
                "Self-conditioning requires seg_doc_id — construct the "
                "generator with return_seg_doc_id=True."
            )
        seg_doc_id = batch[7]
        if not isinstance(seg_doc_id, t.Tensor):
            raise SelfCondTrainingError(
                "SelfCondBatch.from_sft requires a real seg_doc_id tensor at "
                f"index 7; got {type(seg_doc_id).__name__}. Self-conditioning "
                "cannot gate cross-document rollout actions without it."
            )
        return cls(
            inputs=batch[0],
            targets=batch[1],
            enc_fids=batch[2],
            enc_mags=batch[3],
            enc_mask=batch[4],
            seg_map=batch[5],
            seg_doc_id=seg_doc_id,
        )


@dataclass
class MutableConditioning:
    """Widened conditioning tensors mutated during an on-policy rollout."""

    fids: t.Tensor
    mags: t.Tensor
    mask: t.Tensor
    sign: t.Tensor


@dataclass
class RolloutResult:
    """Final rollout conditioning and the sampled counts for every slot."""

    conditioning: MutableConditioning
    predicted: t.Tensor
    k: t.Tensor
    l: t.Tensor  # noqa: E741 - mathematical negative-action count from the spec


def continuation_mask(seg_map: t.Tensor, seg_doc_id: t.Tensor) -> t.Tensor:
    """Return slots whose immediately preceding slot can predict them."""
    if seg_map.ndim != 2 or seg_doc_id.ndim != 2:
        raise SelfCondTrainingError(
            "seg_map and seg_doc_id must both be rank-2 tensors, got "
            f"{tuple(seg_map.shape)} and {tuple(seg_doc_id.shape)}"
        )
    batch_size, time = seg_map.shape
    if seg_doc_id.shape[0] != batch_size:
        raise SelfCondTrainingError(
            "seg_map and seg_doc_id batch dimensions must match, got "
            f"{batch_size} and {seg_doc_id.shape[0]}"
        )
    n_seg = seg_doc_id.shape[1]
    if n_seg == 0:
        return t.zeros_like(seg_doc_id, dtype=t.bool)
    if seg_map.dtype != t.int64 or seg_doc_id.dtype != t.int64:
        raise SelfCondTrainingError("seg_map and seg_doc_id must be int64 tensors")
    if bool(((seg_map < 0) | (seg_map >= n_seg)).any()):
        raise SelfCondTrainingError(f"seg_map values must lie in [0, {n_seg})")

    # Slot j is live iff the FP loss would let slot j-1 predict it: this is
    # exactly the target-indexed shift of the loss's valid_source_seg gate
    # (source==target doc, both >= 0, source has tokens; the loss's extra
    # target_seg_valid conjunct is vacuous when doc ids are equal and
    # non-negative). Sharing source_segment_masks keeps rollout gating
    # provably aligned with the objective the head was trained under —
    # bit-identity with the pre-consolidation predicate is pinned by
    # tests/steering/selfcond/test_frozen_fp_parity.py.
    masks = source_segment_masks(segment_lengths(seg_map, n_seg), seg_doc_id)
    live = t.zeros_like(seg_doc_id, dtype=t.bool)
    if n_seg > 1:
        live[:, 1:] = masks.valid_source_seg[:, :-1]
    return live


def lm_conditioned_slot_mask(seg_map: t.Tensor, seg_doc_id: t.Tensor) -> t.Tensor:
    """Rollout slot gate: continuation slots that are also LM-conditioned.

    ``continuation_mask`` is the FP-loss-aligned predicate (bit-identity pinned
    by test_frozen_fp_parity) and deliberately stays untouched. SFT rows may
    carry prediction-target-only segments — real doc ids but zero decoder
    tokens (kept past truncation purely as FP supervision targets). No decoder
    position cross-attends to such a slot, so a rollout action written there
    can never reach the LM loss; predicting it would only waste an action and
    corrupt the predicted-slot/count diagnostics. The extra conjunct is the
    identity on every batch whose real segments all have tokens (all
    precomputed packed rows).
    """
    live = continuation_mask(seg_map, seg_doc_id)
    n_seg = seg_doc_id.shape[1]
    if n_seg == 0:
        return live
    return live & (segment_lengths(seg_map, n_seg) > 0)


def build_selfcond_state(
    batch: ConditioningBatch,
    f_prime: int,
) -> MutableConditioning:
    """Return a non-aliasing byte-copy widened to ``f_prime`` columns."""
    if batch.enc_fids.ndim != 3:
        raise SelfCondTrainingError(
            f"enc_fids must have shape (B, n_seg, F), got {tuple(batch.enc_fids.shape)}"
        )
    if batch.enc_mags.shape != batch.enc_fids.shape or (
        batch.enc_mask.shape != batch.enc_fids.shape
    ):
        raise SelfCondTrainingError("enc_fids, enc_mags, and enc_mask must have identical shapes")
    original_width = batch.enc_fids.shape[-1]
    if f_prime < original_width:
        raise SelfCondTrainingError(
            f"F' must be at least the batch width {original_width}, got {f_prime}"
        )
    if batch.enc_fids.dtype != t.int64 or batch.enc_mags.dtype != t.int64:
        raise SelfCondTrainingError("enc_fids and enc_mags must be int64 tensors")
    if batch.enc_mask.dtype != t.bool:
        raise SelfCondTrainingError("enc_mask must be a bool tensor")

    shape = (*batch.enc_fids.shape[:2], f_prime)
    fids = batch.enc_fids.new_zeros(shape)
    mags = batch.enc_mags.new_zeros(shape)
    mask = batch.enc_mask.new_zeros(shape)
    fids[..., :original_width].copy_(batch.enc_fids)
    mags[..., :original_width].copy_(batch.enc_mags)
    mask[..., :original_width].copy_(batch.enc_mask)
    sign = t.zeros(shape, dtype=t.float32, device=batch.enc_fids.device)
    return MutableConditioning(fids=fids, mags=mags, mask=mask, sign=sign)


def write_predicted_slot(
    state: MutableConditioning,
    slot: int,
    actions: SelectedActions,
    live: t.Tensor,
) -> None:
    """Replace one continuation slot for live rows with binarized actions."""
    if not (state.fids.shape == state.mags.shape == state.mask.shape == state.sign.shape):
        raise SelfCondTrainingError("mutable conditioning tensors must have identical shapes")
    if state.fids.ndim != 3:
        raise SelfCondTrainingError("mutable conditioning tensors must be rank 3")
    batch_size, n_seg, width = state.fids.shape
    if not 0 <= slot < n_seg:
        raise SelfCondTrainingError(f"slot index {slot} is outside [0, {n_seg})")
    if tuple(live.shape) != (batch_size,) or live.dtype != t.bool:
        raise SelfCondTrainingError(
            f"live must be a bool tensor with shape {(batch_size,)}, got {tuple(live.shape)}"
        )
    if tuple(actions.k.shape) != (batch_size,) or tuple(actions.l.shape) != (batch_size,):
        raise SelfCondTrainingError("action counts must have one value per batch row")
    if actions.pos_ids.shape[0] != batch_size or actions.neg_ids.shape[0] != batch_size:
        raise SelfCondTrainingError("selected action IDs must have one row per batch row")

    # Batched validation with a bounded number of host syncs (review finding:
    # the previous per-row .item()/.any() pattern cost ~4 device round-trips
    # per row inside the per-slot rollout hot path).
    live_k = t.where(live, actions.k, t.zeros_like(actions.k))
    live_l = t.where(live, actions.l, t.zeros_like(actions.l))
    if bool(((live_k < 0) | (live_l < 0)).any()):
        raise SelfCondTrainingError("action counts must be nonnegative")
    if bool(((live_k > actions.pos_ids.shape[1]) | (live_l > actions.neg_ids.shape[1])).any()):
        raise SelfCondTrainingError("action counts exceed selected-ID tensor widths")
    if bool((live_k + live_l > width).any()):
        raise SelfCondTrainingError(
            f"predicted actions need more columns than F' = {width} on at least one row"
        )
    used_positive = t.arange(actions.pos_ids.shape[1], device=live_k.device).unsqueeze(
        0
    ) < live_k.unsqueeze(1)
    used_negative = t.arange(actions.neg_ids.shape[1], device=live_l.device).unsqueeze(
        0
    ) < live_l.unsqueeze(1)
    if bool(((actions.pos_ids < 0) & used_positive).any()) or bool(
        ((actions.neg_ids < 0) & used_negative).any()
    ):
        raise SelfCondTrainingError("real selected action IDs must be nonnegative")

    live_rows = live.tolist()
    k_values = live_k.tolist()
    l_values = live_l.tolist()
    for row in range(batch_size):
        if not live_rows[row]:
            continue
        positive_count = k_values[row]
        negative_count = l_values[row]

        positive_ids = actions.pos_ids[row, :positive_count]
        negative_ids = actions.neg_ids[row, :negative_count]

        state.fids[row, slot].zero_()
        state.mags[row, slot].zero_()
        state.mask[row, slot].zero_()
        state.sign[row, slot].zero_()
        state.fids[row, slot, :positive_count].copy_(positive_ids)
        negative_start = positive_count
        negative_end = positive_count + negative_count
        state.fids[row, slot, negative_start:negative_end].copy_(negative_ids)
        state.mags[row, slot, negative_start:negative_end] = 1
        state.mask[row, slot, :negative_end] = True
        state.sign[row, slot, :positive_count] = 1
        state.sign[row, slot, negative_start:negative_end] = -1


def select_topk_bottoml(
    scores: t.Tensor,
    k: t.Tensor,
    l: t.Tensor,  # noqa: E741 - mathematical negative-action count from the spec
) -> SelectedActions:
    """Select score-descending positives and disjoint score-ascending negatives."""
    return _select_topk_bottoml(scores, k, l)


@t.no_grad()
def rollout(
    model: Any,
    batch: ConditioningBatch,
    cfg: SelfCondTrainConfig,
    rng: t.Generator,
    static_max_seg_size: int | None,
    *,
    consolidation: FeatureClusterConsolidation | None = None,
) -> RolloutResult:
    """Run the no-grad on-policy conditioning rollout from spec section 4.1."""
    assert cfg.positive_counts.max_k is not None
    assert cfg.negative_counts.max_k is not None
    f_prime = max(
        batch.enc_fids.shape[-1],
        (cfg.positive_counts.max_k - 1) + (cfg.negative_counts.max_k - 1),
    )
    conditioning = build_selfcond_state(batch, f_prime)
    predicted = lm_conditioned_slot_mask(batch.seg_map, batch.seg_doc_id)
    feature_vocab = model.feature_vocab
    compact_to_original = (
        None
        if feature_vocab is None
        else feature_vocab.new_to_old_tensor(device=batch.inputs.device)
    )
    batch_size, n_seg = batch.seg_doc_id.shape
    counts_k = t.zeros(
        batch_size,
        n_seg,
        dtype=t.int64,
        device=batch.seg_doc_id.device,
    )
    counts_l = t.zeros_like(counts_k)

    for slot in range(1, n_seg):
        with timing_tags(slot=slot):
            with timed_region("selfcond/rollout/slot", sync_cuda=False):
                with timed_region("selfcond/rollout/slot_gate", sync_cuda=False):
                    live = predicted[:, slot]
                    slot_is_live = bool(live.any())
                if not slot_is_live:
                    continue

                with timed_region(
                    "selfcond/rollout/model_forward",
                    sync_cuda=False,
                ):
                    fp_logits = model(
                        batch.inputs,
                        None,
                        conditioning.fids,
                        conditioning.mags,
                        conditioning.mask,
                        batch.seg_map,
                        static_max_seg_size=static_max_seg_size,
                        fp_logits_only=True,
                    )
                    assert_finite_fp_logits(
                        fp_logits,
                        where=f"the rollout forward for slot {slot}",
                    )
                with timed_region("selfcond/rollout/pool", sync_cuda=False):
                    pooled = pool_fp_logits_per_segment(
                        fp_logits,
                        batch.seg_map,
                        PositiveInt.parse(n_seg),
                        cfg.pool_mode,
                    )
                with timed_region(
                    "selfcond/rollout/sample_select",
                    sync_cuda=False,
                ):
                    # Selection consumes only ordering, so rank the pooled
                    # float32 logits directly. Applying sigmoid first can
                    # collapse distinct finite logits into exact endpoint ties
                    # and make stable row order choose a different action from
                    # inference.
                    scores = pooled[:, slot - 1].float()
                    sampled_k, sampled_l = sample_action_counts(
                        cfg,
                        rng,
                        batch_size,
                    )
                    sampled_k = sampled_k.to(device=scores.device)
                    sampled_l = sampled_l.to(device=scores.device)
                    actions = select_topk_bottoml(scores, sampled_k, sampled_l)
                    if compact_to_original is not None:
                        positive_ids = actions.pos_ids.clone()
                        negative_ids = actions.neg_ids.clone()
                        real_positive = positive_ids >= 0
                        real_negative = negative_ids >= 0
                        positive_ids[real_positive] = compact_to_original[
                            positive_ids[real_positive]
                        ]
                        negative_ids[real_negative] = compact_to_original[
                            negative_ids[real_negative]
                        ]
                        actions = SelectedActions(
                            pos_ids=positive_ids,
                            neg_ids=negative_ids,
                            k=actions.k,
                            l=actions.l,
                        )
                    if consolidation is not None:
                        selected_ids = t.cat(
                            (
                                actions.pos_ids[actions.pos_ids >= 0],
                                actions.neg_ids[actions.neg_ids >= 0],
                            )
                        )
                        consolidation.assert_representatives_only(
                            selected_ids,
                            surface="self-conditioning rollout actions",
                        )
                with timed_region(
                    "selfcond/rollout/write_slot",
                    sync_cuda=False,
                ):
                    write_predicted_slot(conditioning, slot, actions, live)
                    counts_k[:, slot] = t.where(live, sampled_k, 0)
                    counts_l[:, slot] = t.where(live, sampled_l, 0)

    return RolloutResult(
        conditioning=conditioning,
        predicted=predicted,
        k=counts_k,
        l=counts_l,
    )


def _meta_value(meta: Mapping[str, Any], key: str, default: object = None) -> object:
    """Read a recorded config value with user_config taking precedence.

    A key PRESENT in user_config wins even when its value is ``None`` — the
    C5/C6 guard reads below must never resolve an explicitly-recorded null to
    a permissive default (review finding: null falling through to
    ``use_precomputed_batches``' True default would silently disarm C6).
    The one place that must tolerate an SFT checkpoint's raw ``null`` is
    ``resolve_recorded_num_buckets``, which owns its own fallthrough.
    """
    user_config = meta.get("user_config")
    if isinstance(user_config, dict) and key in user_config:
        return user_config[key]
    return meta.get(key, default)


def _meta_value_null_falls_through(
    meta: Mapping[str, Any], key: str, default: object = None
) -> object:
    """Like ``_meta_value``, but an explicit ``None`` in user_config defers to
    the top-level meta key. SFT checkpoints retain their raw config in
    user_config (where null means "resolve from the base") and publish the
    RESOLVED value at the top level; readers of resolved structural facts use
    this variant."""
    user_config = meta.get("user_config")
    if isinstance(user_config, dict) and user_config.get(key) is not None:
        return user_config[key]
    return meta.get(key, default)


def validate_selfcond_prerequisites(
    model: Any,
    meta: dict,
    cfg: SelfCondTrainConfig,
    n_sae_features: int,
    *,
    data_source: SelfCondDataSource = SelfCondDataSource.PRETRAINING,
) -> None:
    """Fail loudly unless the checkpoint/data family satisfies C1 through C8."""
    feature_vocab = model.feature_vocab
    if (
        data_source is SelfCondDataSource.PRETRAINING
        and feature_vocab is not None
        and model.feature_pred_head is not None
    ):
        raise NotImplementedError(
            "C1: pretraining-source self-conditioning post-training does not "
            "support a compact "
            "feature-prediction head. The checkpoint predicts compact rows whose "
            "meaning depends on its FeatureVocab, while the packed-row rollout/STE "
            "path has no compact target contract. Source-aware SFT post-training has "
            "a separate compact contract; inference-time feedback is separate too. "
            "Use an uncompacted FP checkpoint for pretraining data, or an exact "
            "compact SFT checkpoint with data_source=sft."
        )
    if model.feature_pred_head is None:
        raise SelfCondTrainingError(
            "C1: self-conditioning post-training requires a feature-prediction head, "
            "but this checkpoint loaded without one.\n"
            "Choose a checkpoint pretrained with feature_prediction enabled."
        )

    feature_prediction = _meta_value(meta, "feature_prediction")
    if feature_prediction is None:
        feature_prediction = meta.get("feature_prediction_config")
    if not isinstance(feature_prediction, dict):
        raise SelfCondTrainingError(
            "C1: the feature-prediction head configuration is not recoverable from checkpoint "
            "metadata.\nExpected user_config.feature_prediction or the legacy top-level "
            "feature_prediction_config block."
        )
    if feature_vocab is None:
        if model.feature_pred_head.out_features != n_sae_features:
            raise SelfCondTrainingError(
                "C1: the loaded feature-prediction head width disagrees with n_sae_features: "
                f"{model.feature_pred_head.out_features} != {n_sae_features}.\n"
                "Use the checkpoint's recorded SAE feature universe."
            )
    elif model.feature_pred_head.out_features != feature_vocab.size:
        raise SelfCondTrainingError(
            "C1: the loaded feature-prediction head width disagrees with the compact "
            f"action-space width: {model.feature_pred_head.out_features} != "
            f"{feature_vocab.size}.\nUse the checkpoint's recorded FeatureVocab."
        )

    feature_mask = _meta_value(meta, "feature_mask")
    if data_source is SelfCondDataSource.PRETRAINING and (
        feature_vocab is not None
        or model.feature_mask is not None
        or hasattr(model.encoder, "feature_vocab_ids")
        or feature_mask is not None
    ):
        raise NotImplementedError(
            "C2: pretraining-source self-conditioning post-training requires an "
            "UNCOMPACTED encoder, but a "
            "feature_mask or encoder.feature_vocab_ids resolved.\n"
            "Use an unmasked checkpoint for pretraining data. Source-aware SFT "
            "post-training has a separate exact compact target contract, and "
            "inference-time compact feedback is another supported path."
        )

    recorded_num_buckets = resolve_recorded_num_buckets(meta)
    if recorded_num_buckets != 2 or model.encoder.config.num_buckets != 2:
        raise SelfCondTrainingError(
            "C3: self-conditioning requires num_buckets == 2 in both metadata and the loaded "
            f"model; got metadata={recorded_num_buckets!r}, "
            f"model={model.encoder.config.num_buckets}.\n"
            "Only that layout makes bucket 1 the dedicated negative bucket."
        )

    conditioning = read_checkpoint_conditioning(meta)
    if conditioning.negative_selection is None:
        raise SelfCondTrainingError(
            "C4: checkpoint negative_selection is null.\n"
            "Without negative sampling, bucket 1 means a high positive rather than a negative; "
            "feeding self-conditioned negatives would silently corrupt semantics."
        )

    if (
        data_source is SelfCondDataSource.PRETRAINING
        and _meta_value(meta, "feature_clusters") is not None
    ):
        raise SelfCondTrainingError(
            "C5: feature_clusters is unsupported by the pretraining-source "
            "self-conditioning post-training path.\nChoose a checkpoint trained "
            "without clustered selection, or use source-aware SFT with an exact "
            "cluster/vocabulary contract."
        )
    feature_selection = _meta_value(meta, "feature_selection")
    if isinstance(feature_selection, dict) and feature_selection.get("gradient") is not None:
        raise SelfCondTrainingError(
            "C5: feature_selection.gradient is unsupported by self-conditioning "
            "post-training v1.\nChoose a checkpoint without gradient-attribution ranking."
        )
    if _meta_value(meta, "label_dropout", 0.0) != 0.0:
        raise SelfCondTrainingError(
            "C5: label_dropout must equal 0 for self-conditioning post-training v1.\n"
            "The rollout contract does not define dropped-label interactions."
        )
    if conditioning.has_scheduled_dropout():
        raise SelfCondTrainingError(
            "C5: conditioning_cooldown must not drop feature labels for "
            "self-conditioning post-training v1.\n"
            "The rollout contract does not define dropped-label interactions."
        )
    # Truthiness, not identity: legacy or hand-repaired meta json may record
    # these as 0/1 integers; `is not True` would spuriously reject a valid
    # checkpoint family with a misleading constraint message (review finding).
    if bool(_meta_value(meta, "next_segment_conditioning_offset", False)):
        raise SelfCondTrainingError(
            "C5: next_segment_conditioning_offset must be false for self-conditioning "
            "post-training v1.\nThe rollout already owns the one-segment prediction offset."
        )

    if data_source is SelfCondDataSource.PRETRAINING and not bool(
        _meta_value(meta, "use_precomputed_batches", True)
    ):
        raise SelfCondTrainingError(
            "C6: self-conditioning post-training v1 requires the precomputed packed-row "
            "loader.\nSet use_precomputed_batches=true and provide the matching packed rows."
        )

    if cfg.pool_mode not in {
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.LAST_TOKEN,
        FeaturePredLossMode.FULL_SEGMENT,
    }:
        raise SelfCondTrainingError(
            f"C7: pool_mode={cfg.pool_mode!s} cannot drive a target-free rollout.\n"
            "Use max_pool, last_token, or full_segment; best_token requires target features."
        )

    # C8 must also see a top-level feature_delta: read_checkpoint_conditioning
    # consults user_config and the pretrain_conditioning block only, and a
    # fail-closed guard must not narrow its input domain (review finding).
    feature_delta = (
        conditioning.feature_delta
        if conditioning.feature_delta is not None
        else meta.get("feature_delta")
    )
    sft_delta_mode: DeltaMode | None = None
    if data_source is SelfCondDataSource.SFT:
        sft_conditioning = replace(conditioning, feature_delta=feature_delta)
        try:
            sft_delta_mode = sft_conditioning.delta_mode()
            assert_conditioning_supported(
                sft_conditioning,
                checkpoint_label="Self-conditioning SFT base checkpoint",
            )
        except ValueError as error:
            raise SelfCondTrainingError(f"C8: {error}") from error
    elif isinstance(feature_delta, dict) and feature_delta.get("mode") == "splice_counterfactual":
        raise SelfCondTrainingError(
            "C8: feature_delta.mode=splice_counterfactual is outside self-conditioning "
            "post-training v1.\nUse a non-splice checkpoint and packed-row dataset."
        )
    feature_dir = _meta_value(meta, "feature_dir")
    if feature_dir is not None:
        feature_path = Path(str(feature_dir))
        if feature_path.parent.parent.name == "splice" and not (
            data_source is SelfCondDataSource.SFT
            and sft_delta_mode is DeltaMode.splice_counterfactual
        ):
            raise SelfCondTrainingError(
                "C8: a spliced derived feature_dir is outside self-conditioning "
                f"post-training v1: {feature_path}.\nUse the original non-spliced feature dir."
            )

    # C9: the STE reaches the loss ONLY through encoder_out -> cross-attention.
    # A model with zero cross-attention layers (reachable via
    # cross_attention.frequency > n_layer) leaves encoder_out unused: head-only
    # backward would raise "does not require grad" and full scope would leave
    # every encoder parameter with a None gradient that DistMuonAdamW cannot
    # consume. No conditioned checkpoint has zero cross-attention layers, so this
    # is a defensive startup guard turning a confusing mid-step crash into a
    # clear message.
    if len(model.cross_attn_layers) == 0:
        raise SelfCondTrainingError(
            "C9: the loaded model has no cross-attention layers, so the encoder "
            "output (and therefore the self-conditioning STE) never reaches the "
            "decoder loss. This model cannot be conditioned or self-conditioned; "
            "it was likely built with cross_attention.frequency greater than the "
            "decoder depth."
        )

    assert cfg.positive_counts.max_k is not None
    assert cfg.negative_counts.max_k is not None
    max_selected = (cfg.positive_counts.max_k - 1) + (cfg.negative_counts.max_k - 1)
    if feature_vocab is None:
        if max_selected > n_sae_features:
            raise SelfCondTrainingError(
                "The maximum positive+negative action count exceeds n_sae_features: "
                f"{max_selected} > {n_sae_features}.\n"
                "Reduce the two count max_k bounds so disjoint selection is possible."
            )
    elif max_selected > feature_vocab.size:
        raise SelfCondTrainingError(
            "The maximum positive+negative action count exceeds the compact action-space "
            f"width: {max_selected} > {feature_vocab.size}.\n"
            "Reduce the two count max_k bounds so disjoint selection is possible."
        )


@dataclass(frozen=True)
class SelfCondStepMetrics:
    """Detached scalar diagnostics for one self-conditioning optimizer step."""

    lm_loss: float
    frac_predicted_slots: float
    mean_k: float
    mean_l: float
    mean_selected_pos_score: float


def _step_metrics(
    loss: t.Tensor,
    batch: ConditioningBatch,
    result: RolloutResult,
    shifted_scores: t.Tensor | None,
    *,
    feature_vocab: FeatureVocab | None = None,
) -> SelfCondStepMetrics:
    n_seg = batch.seg_doc_id.shape[1]
    if n_seg == 0:
        # Mirror the n_seg == 0 guard of continuation_mask /
        # lm_conditioned_slot_mask: segment_lengths cannot scatter into a
        # zero-width tensor, and the pre-gate code returned fraction 0.0 here
        # (review finding).
        real_slots = t.zeros_like(batch.seg_doc_id, dtype=t.bool)
    else:
        real_slots = (batch.seg_doc_id >= 0) & (segment_lengths(batch.seg_map, n_seg) > 0)
    real_count = int(real_slots.sum().item())
    predicted_count = int(result.predicted.sum().item())
    fraction = predicted_count / real_count if real_count else 0.0
    if predicted_count:
        mean_k = float(result.k[result.predicted].float().mean().item())
        mean_l = float(result.l[result.predicted].float().mean().item())
    else:
        mean_k = 0.0
        mean_l = 0.0

    mean_positive = 0.0
    if shifted_scores is not None:
        positive_columns = result.conditioning.sign > 0
        if bool(positive_columns.any()):
            if feature_vocab is None:
                selected_scores = shifted_scores.gather(2, result.conditioning.fids)
            else:
                original_to_compact = feature_vocab.old_to_new_tensor(
                    device=result.conditioning.fids.device
                )
                compact_fids = original_to_compact[result.conditioning.fids]
                if bool((compact_fids[positive_columns] < 0).any()):
                    raise SelfCondTrainingError(
                        "a positive self-conditioning feature ID is absent from the "
                        "compact feature vocabulary; rollout must emit original IDs "
                        "represented by the checkpoint's FeatureVocab"
                    )
                selected_scores = shifted_scores.gather(2, compact_fids.clamp_min(0))
            mean_positive = float(selected_scores[positive_columns].detach().mean().item())
    return SelfCondStepMetrics(
        lm_loss=float(loss.detach().item()),
        frac_predicted_slots=fraction,
        mean_k=mean_k,
        mean_l=mean_l,
        mean_selected_pos_score=mean_positive,
    )


def selfcond_loss(
    model: Any,
    batch: ConditioningBatch,
    result: RolloutResult,
    cfg: SelfCondTrainConfig,
    static_max_seg_size: int | None,
) -> tuple[t.Tensor, SelfCondStepMetrics]:
    """Build the specified two-pass, one-hop STE surrogate and return its CE."""
    state = result.conditioning
    if not bool(result.predicted.any()):
        with timed_region("selfcond/pass2/encoder", sync_cuda=False):
            encoder_out = model.encoder(state.fids, state.mags, state.mask)
        with timed_region("selfcond/pass2/lm_forward", sync_cuda=False):
            loss = model(
                batch.inputs,
                batch.targets,
                seg_map=batch.seg_map,
                encoder_out=encoder_out,
                static_max_seg_size=static_max_seg_size,
            )
        if not loss.requires_grad:
            loss = loss.detach().requires_grad_()
        with timed_region("selfcond/metrics", sync_cuda=False):
            metrics = _step_metrics(
                loss,
                batch,
                result,
                None,
                feature_vocab=model.feature_vocab,
            )
        return loss, metrics

    with timed_region("selfcond/pass1/model_forward", sync_cuda=False):
        fp_logits = model(
            batch.inputs,
            None,
            state.fids,
            state.mags,
            state.mask,
            batch.seg_map,
            static_max_seg_size=static_max_seg_size,
            fp_logits_only=True,
        )
        assert_finite_fp_logits(fp_logits, where="the differentiable pass-1 forward")
    with timed_region("selfcond/pass1/pool_and_ste", sync_cuda=False):
        n_seg = batch.seg_doc_id.shape[1]
        pooled = pool_fp_logits_per_segment(
            fp_logits,
            batch.seg_map,
            PositiveInt.parse(n_seg),
            cfg.pool_mode,
        )
        scores = t.sigmoid(pooled.float())
        shifted_scores = t.zeros_like(scores)
        shifted_scores[:, 1:] = scores[:, :-1]
        ste = SteInputs(
            scores=shifted_scores,
            sign=state.sign,
            form=cfg.ste_form,
        )
    with timed_region("selfcond/pass2/encoder", sync_cuda=False):
        encoder_out = model.encoder(
            state.fids,
            state.mags,
            state.mask,
            ste=ste,
        )
    with timed_region("selfcond/pass2/lm_forward", sync_cuda=False):
        loss = model(
            batch.inputs,
            batch.targets,
            seg_map=batch.seg_map,
            encoder_out=encoder_out,
            static_max_seg_size=static_max_seg_size,
        )
    with timed_region("selfcond/metrics", sync_cuda=False):
        metrics = _step_metrics(
            loss,
            batch,
            result,
            shifted_scores,
            feature_vocab=model.feature_vocab,
        )
    return loss, metrics


def freeze_for_head_only(model: Any) -> None:
    """Freeze every parameter except the feature-prediction head."""
    if model.feature_pred_head is None:
        raise SelfCondTrainingError(
            "head-only self-conditioning requires a feature-prediction head"
        )
    model.requires_grad_(False)
    model.feature_pred_head.requires_grad_(True)


def materialize_head_zero_grads(model: Any) -> None:
    """Materialize mathematically exact zero head grads for DDP collectives (R16)."""
    if model.feature_pred_head is None:
        raise SelfCondTrainingError(
            "self-conditioning gradient materialization requires a feature-prediction head"
        )
    for parameter in model.feature_pred_head.parameters():
        if parameter.grad is None:
            parameter.grad = t.zeros_like(parameter)


def build_head_only_optimizer(
    model: Any,
    *,
    unembedding_lr: float,
    adam_betas: tuple[float, float],
) -> t.optim.Optimizer:
    """Build the single complete AdamW group used by head-only training."""
    if model.feature_pred_head is None:
        raise SelfCondTrainingError(
            "head-only self-conditioning requires a feature-prediction head"
        )
    head_parameters = list(model.feature_pred_head.parameters())
    dmodel_lr_scale = (model.dec_config.n_embd / 768) ** -0.5
    param_groups = [
        {
            "kind": OptimizerKind.ADAMW,
            "params": head_parameters,
            "lr": unembedding_lr * dmodel_lr_scale,
            "betas": adam_betas,
            "eps": 1e-10,
            "weight_decay": 0.0,
        }
    ]
    ddp, _rank, _local_rank, _world_size = get_dist_info()
    factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    assert len(optimizer.param_groups) == 1
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in head_parameters
    }
    return optimizer
