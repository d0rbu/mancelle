"""Sequential structured-SFT rollout, detached reward, and sparse replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch as t
from torch.utils.checkpoint import checkpoint

from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.selfcond_training import (
    freeze_for_head_only,
    pool_fp_logits_per_segment,
)
from steering.conditional.encoder_decoder.sft_grpo import (
    ActionMode,
    ConditioningAction,
    SFTGRPOError,
    SparseActionTrace,
    _behavior_checked_replay_ratios_from_finite_log_weights,
    _sample_weighted_without_replacement_batch_from_log_weights,
    _sample_weighted_without_replacement_from_finite_forward,
    _sample_weighted_without_replacement_from_log_weights,
    _weighted_log_weights_from_finite_forward,
    behavior_checked_replay_ratio,
    bernoulli_current_to_reference_kl,
    conversation_group_advantages,
    masked_mean_token_ce,
    sample_bernoulli_topk_bottomk,
    trace_to_conditioning,
    trajectory_mean_ratio,
    validate_count_support,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
    sample_action_counts,
)
from steering.refined_types import PositiveInt
from steering.sft.structured_rollout import ConversationEnvironment, stateless_seed


class InitialConditioningMode(StrEnum):
    """The only explicit first-slot provider choices."""

    NONE = "none"
    ONLINE_SAE = "online_sae"


@dataclass(frozen=True)
class GRPOPolicyConfig:
    """Resolved action and deterministic replay policy."""

    action_mode: ActionMode
    positive_counts: SamplingSelection
    negative_counts: NegativeSamplingSelection
    seed: int
    action_policy_version: int = 1
    replay_atol: float = 2e-5
    replay_rtol: float = 2e-5
    pool_mode: FeaturePredLossMode = FeaturePredLossMode.MAX_POOL

    def validate(self, *, feature_count: int) -> None:
        if (
            self.positive_counts.distribution is not Distribution.geometric
            or self.negative_counts.distribution is not Distribution.geometric
        ):
            raise SFTGRPOError(
                "structured-SFT GRPO requires checkpoint-recorded geometric "
                "positive and negative count distributions"
            )
        if self.pool_mode is not FeaturePredLossMode.MAX_POOL:
            raise SFTGRPOError("structured-SFT GRPO requires feature-prediction pooling=max_pool")
        if self.positive_counts.gradient is not None:
            raise SFTGRPOError(
                "live gradient-attribution selection is unsupported by structured-SFT GRPO"
            )
        if self.positive_counts.max_k is None or self.negative_counts.max_k is None:
            raise SFTGRPOError("positive and negative count maxima must be finite")
        validate_count_support(
            positive_min=int(self.positive_counts.min_k),
            positive_max_exclusive=int(self.positive_counts.max_k),
            negative_min=int(self.negative_counts.min_k),
            negative_max_exclusive=int(self.negative_counts.max_k),
            feature_count=feature_count,
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise SFTGRPOError("policy seed must be a non-negative integer")
        if self.action_policy_version not in {1, 2}:
            raise SFTGRPOError("action_policy_version must be 1 or 2")
        if any(
            not math.isfinite(value) or value < 0 for value in (self.replay_atol, self.replay_rtol)
        ):
            raise SFTGRPOError("replay tolerances must be finite and non-negative")


@dataclass(frozen=True)
class SeedContext:
    """Stateless identity shared by every trajectory decision."""

    namespace: str
    completed_optimizer_step: int
    conversation_index: int

    def __post_init__(self) -> None:
        if self.namespace not in {"train", "validation", "benchmark"}:
            raise SFTGRPOError("seed namespace must be train, validation, or benchmark")
        if self.completed_optimizer_step < 0 or self.conversation_index < 0:
            raise SFTGRPOError("seed step and conversation index must be non-negative")


@dataclass(frozen=True)
class TrajectoryTrace:
    """Sparse decisions plus complete encoder-conditioning history."""

    rollout_index: int
    conditioning: tuple[ConditioningAction, ...]
    actions: tuple[SparseActionTrace, ...]
    reward: float


@dataclass(frozen=True)
class ConversationGroupRollout:
    """One conversation-local group and detached normalized advantages."""

    environment: ConversationEnvironment
    trajectories: tuple[TrajectoryTrace, ...]
    advantages: tuple[float, ...]
    initial_conditioning_mode: InitialConditioningMode
    sae_loaded: bool


@dataclass(frozen=True)
class ReplayMetrics:
    """Detached diagnostics for one replayed conversation group."""

    policy_gradient_loss: float
    kl_loss: float
    total_loss: float
    ratio_min: float
    ratio_max: float
    live_decisions: int


@dataclass(frozen=True)
class NTPObjective:
    """Per-step auxiliary next-token-prediction mixture."""

    trajectories_per_group: int
    loss_weight: float


class NoneInitialConditioningProvider:
    """Canonical all-masked C0 with no resolver, model, RNG, or CUDA activity."""

    mode = InitialConditioningMode.NONE
    sae_loaded = False

    def provide(self, environment: ConversationEnvironment) -> ConditioningAction:
        del environment
        return ConditioningAction(feature_ids=(), magnitudes=(), mask=())


class InitialConditioningProvider(Protocol):
    """Provider contract; implementations are invoked once per conversation."""

    mode: InitialConditioningMode

    @property
    def sae_loaded(self) -> bool: ...

    def provide(self, environment: ConversationEnvironment) -> ConditioningAction: ...


def resolve_train_feature_head_only(cfg: Mapping[str, Any]) -> bool:
    """Validate the shared training scope before model or artifact admission."""
    enabled = cfg.get("train_feature_head_only", False)
    if not isinstance(enabled, bool):
        raise SFTGRPOError(f"train_feature_head_only must be a boolean, got {enabled!r}")
    if enabled and cfg.get("ntp") is not None:
        raise SFTGRPOError(
            "train_feature_head_only=true requires ntp=null: integer/detached conditioning "
            "actions give the NTP loss no differentiable path into the feature head"
        )
    return enabled


def configure_grpo_trainable_parameters(
    model: Any,
    *,
    train_lm_head: bool,
    train_feature_head_only: bool = False,
) -> tuple[str, ...]:
    """Set the trainable set and return the ordered manifest.

    By default, ``train_lm_head=False`` reproduces the policy-only objective:
    every parameter remains trainable except the untied ``lm_head``.
    ``train_lm_head=True`` keeps ``lm_head`` trainable for auxiliary NTP.
    ``train_feature_head_only=True`` instead trains the complete feature head;
    auxiliary NTP cannot train this scope through detached integer actions.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise SFTGRPOError("model must expose an untied lm_head")
    if train_feature_head_only:
        if train_lm_head:
            raise SFTGRPOError("feature-head-only GRPO cannot also train the LM head for NTP")
        if getattr(model, "feature_pred_head", None) is None:
            raise SFTGRPOError("feature-head-only GRPO requires a feature-prediction head")
        freeze_for_head_only(model)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if not train_lm_head:
            lm_head.requires_grad_(False)
    manifest = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not manifest:
        raise SFTGRPOError("policy objective resolved no trainable parameters")
    has_lm_head = any(name == "lm_head" or name.startswith("lm_head.") for name in manifest)
    if train_lm_head and not has_lm_head:
        raise AssertionError("trainable lm_head is absent from GRPO parameter manifest")
    if not train_lm_head and has_lm_head:
        raise AssertionError("frozen lm_head leaked into GRPO parameter manifest")
    return manifest


def assert_grpo_gradients_ready(
    model: Any,
    *,
    expected_manifest: tuple[str, ...],
) -> None:
    """Validate the exact non-None finite-gradient precondition for the optimizer."""
    actual_manifest = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if actual_manifest != expected_manifest:
        raise SFTGRPOError("trainable policy parameter manifest changed after setup")
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(t.isfinite(parameter.grad).all()):
            nonfinite.append(name)
    if missing or nonfinite:
        raise SFTGRPOError(
            f"policy gradients are not optimizer-ready: missing={missing}, nonfinite={nonfinite}"
        )


def _feature_count(model: Any) -> int:
    head = getattr(model, "feature_pred_head", None)
    width = getattr(head, "out_features", None)
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise SFTGRPOError("model must expose a non-empty feature_pred_head")
    return width


def _compact_to_original(model: Any, *, device: t.device) -> t.Tensor | None:
    feature_vocab = getattr(model, "feature_vocab", None)
    if feature_vocab is None:
        return None
    mapping = feature_vocab.new_to_old_tensor(device=device)
    if mapping.dtype != t.int64 or mapping.ndim != 1:
        raise SFTGRPOError("model feature vocabulary returned an invalid row mapping")
    return mapping


def _compact_to_original_host(model: Any) -> tuple[int, ...] | None:
    """Materialize the encoder-boundary map once without touching CUDA."""
    mapping = _compact_to_original(model, device=t.device("cpu"))
    if mapping is None:
        return None
    return tuple(int(value) for value in mapping.tolist())


def _environment_tensors(
    environment: ConversationEnvironment,
    *,
    batch_size: int,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    inputs = t.tensor(environment.inputs, dtype=t.int64, device=device).unsqueeze(0)
    targets = t.tensor(environment.targets, dtype=t.int64, device=device).unsqueeze(0)
    seg_map = t.tensor(environment.seg_map, dtype=t.int64, device=device).unsqueeze(0)
    reward_mask = t.tensor(environment.reward_mask, dtype=t.bool, device=device).unsqueeze(0)
    return (
        inputs.expand(batch_size, -1),
        targets.expand(batch_size, -1),
        seg_map.expand(batch_size, -1),
        reward_mask.expand(batch_size, -1),
    )


def _conditioning_tensors(
    histories: list[tuple[ConditioningAction, ...]],
    *,
    n_segments: int,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    if not histories or any(len(history) != n_segments for history in histories):
        raise SFTGRPOError("every trajectory conditioning history must have one slot per segment")
    width = max(
        1,
        max(len(action.feature_ids) for history in histories for action in history),
    )
    shape = (len(histories), n_segments, width)
    feature_ids = t.zeros(shape, dtype=t.int64, device=device)
    magnitudes = t.zeros(shape, dtype=t.int64, device=device)
    mask = t.zeros(shape, dtype=t.bool, device=device)
    for row, history in enumerate(histories):
        for slot, action in enumerate(history):
            selected = len(action.feature_ids)
            if not (
                len(action.magnitudes) == selected
                and len(action.mask) == selected
                and all(action.mask)
            ):
                raise SFTGRPOError(
                    "conditioning actions must contain aligned selected IDs, buckets, and true masks"
                )
            if selected > width:
                raise AssertionError("conditioning width accounting drifted")
            if selected:
                feature_ids[row, slot, :selected] = t.tensor(
                    action.feature_ids,
                    dtype=t.int64,
                    device=device,
                )
                magnitudes[row, slot, :selected] = t.tensor(
                    action.magnitudes,
                    dtype=t.int64,
                    device=device,
                )
                mask[row, slot, :selected] = True
    return feature_ids, magnitudes, mask


def _generator(seed: int, *, device: t.device | str) -> t.Generator:
    return t.Generator(device=device).manual_seed(seed)


def _decision_seed(
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_index: int,
    segment_index: int,
    draw_namespace: str,
) -> int:
    return stateless_seed(
        {
            "namespace": seed_context.namespace,
            "base_seed": config.seed,
            "completed_optimizer_step": seed_context.completed_optimizer_step,
            "original_index": environment.original_index,
            "conversation_index": seed_context.conversation_index,
            "rollout_index": rollout_index,
            "segment_index": segment_index,
            "draw_namespace": draw_namespace,
            "action_policy_version": config.action_policy_version,
        }
    )


def _sample_counts(
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_index: int,
    segment_index: int,
) -> tuple[int, int]:
    seed = _decision_seed(
        environment,
        seed_context,
        config,
        rollout_index=rollout_index,
        segment_index=segment_index,
        draw_namespace="positive_negative_count",
    )
    positive, negative = sample_action_counts(
        config.positive_counts,
        config.negative_counts,
        _generator(seed, device="cpu"),
        1,
    )
    return int(positive[0].item()), int(negative[0].item())


def _microbatch_seed(
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_start: int,
    rollout_count: int,
    segment_index: int,
    draw_namespace: str,
) -> int:
    return stateless_seed(
        {
            "namespace": seed_context.namespace,
            "base_seed": config.seed,
            "completed_optimizer_step": seed_context.completed_optimizer_step,
            "original_index": environment.original_index,
            "conversation_index": seed_context.conversation_index,
            "rollout_start": rollout_start,
            "rollout_count": rollout_count,
            "segment_index": segment_index,
            "draw_namespace": draw_namespace,
            "action_policy_version": config.action_policy_version,
        }
    )


def _sample_counts_batch(
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_start: int,
    rollout_count: int,
    segment_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    seed = _microbatch_seed(
        environment,
        seed_context,
        config,
        rollout_start=rollout_start,
        rollout_count=rollout_count,
        segment_index=segment_index,
        draw_namespace="positive_negative_count_batch",
    )
    positive, negative = sample_action_counts(
        config.positive_counts,
        config.negative_counts,
        _generator(seed, device="cpu"),
        rollout_count,
    )
    return (
        tuple(int(value) for value in positive.tolist()),
        tuple(int(value) for value in negative.tolist()),
    )


def _batch_action_generator(
    logits: t.Tensor,
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_start: int,
    rollout_count: int,
    segment_index: int,
) -> t.Generator:
    seed = _microbatch_seed(
        environment,
        seed_context,
        config,
        rollout_start=rollout_start,
        rollout_count=rollout_count,
        segment_index=segment_index,
        draw_namespace="weighted_ordered_draw_batch",
    )
    return _generator(seed, device=logits.device)


def _sample_action(
    logits: t.Tensor,
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_index: int,
    segment_index: int,
    positive_count: int,
    negative_count: int,
    original_feature_ids: t.Tensor,
) -> SparseActionTrace:
    generator = _action_generator(
        logits,
        environment,
        seed_context,
        config,
        rollout_index=rollout_index,
        segment_index=segment_index,
    )
    if config.action_mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        return _sample_weighted_without_replacement_from_finite_forward(
            logits,
            positive_count=positive_count,
            negative_count=negative_count,
            generator=generator,
        )
    return sample_bernoulli_topk_bottomk(
        logits,
        positive_count=positive_count,
        negative_count=negative_count,
        generator=generator,
        original_feature_ids=original_feature_ids,
    )


def _action_generator(
    logits: t.Tensor,
    environment: ConversationEnvironment,
    seed_context: SeedContext,
    config: GRPOPolicyConfig,
    *,
    rollout_index: int,
    segment_index: int,
) -> t.Generator:
    draw_namespace = (
        "weighted_ordered_draw"
        if config.action_mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT
        else "bernoulli_classification"
    )
    seed = _decision_seed(
        environment,
        seed_context,
        config,
        rollout_index=rollout_index,
        segment_index=segment_index,
        draw_namespace=draw_namespace,
    )
    return _generator(seed, device=logits.device)


def _fp_forward(
    model: Any,
    inputs: t.Tensor,
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
    mask: t.Tensor,
    seg_map: t.Tensor,
) -> t.Tensor:
    logits = model(
        inputs,
        None,
        feature_ids,
        magnitudes,
        mask,
        seg_map,
        fp_logits_only=True,
    )
    if not isinstance(logits, t.Tensor) or logits.ndim != 3:
        raise SFTGRPOError("fp_logits_only forward must return a (B,T,V) tensor")
    if not bool(t.isfinite(logits).all()):
        raise SFTGRPOError("feature-policy forward produced non-finite logits")
    return logits


def _fp_forward_replay(
    model: Any,
    inputs: t.Tensor,
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
    mask: t.Tensor,
    seg_map: t.Tensor,
) -> t.Tensor:
    """Expose the behavior-mode value while recomputing an autograd graph.

    Reentrant activation checkpointing executes its visible forward under
    ``no_grad`` and recomputes under autograd during backward. That keeps the
    replay value on the same CUDA kernel path as behavior collection without
    detaching the policy gradient. The dummy differentiable input satisfies the
    reentrant API; model parameters used by the closure receive the real grads.
    """
    if not t.is_grad_enabled():
        raise SFTGRPOError("differentiable replay requires autograd to be enabled")
    dummy = t.ones((), dtype=t.float32, device=inputs.device, requires_grad=True)

    def recompute(
        _dummy: t.Tensor,
        replay_inputs: t.Tensor,
        replay_feature_ids: t.Tensor,
        replay_magnitudes: t.Tensor,
        replay_mask: t.Tensor,
        replay_seg_map: t.Tensor,
    ) -> t.Tensor:
        del _dummy
        return _fp_forward(
            model,
            replay_inputs,
            replay_feature_ids,
            replay_magnitudes,
            replay_mask,
            replay_seg_map,
        )

    return checkpoint(
        recompute,
        dummy,
        inputs,
        feature_ids,
        magnitudes,
        mask,
        seg_map,
        use_reentrant=True,
        preserve_rng_state=False,
    )


@t.no_grad()
def collect_conversation_group(
    model: Any,
    environment: ConversationEnvironment,
    *,
    group_size: int,
    config: GRPOPolicyConfig,
    seed_context: SeedContext,
    initial_provider: InitialConditioningProvider,
    trajectory_microbatch_size: int | None = None,
) -> ConversationGroupRollout:
    """Collect one complete on-policy group and detached LM rewards."""
    if getattr(model, "training", False):
        raise SFTGRPOError("behavior collection requires deterministic eval-mode model")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 1:
        raise SFTGRPOError("group_size must be an integer greater than one")
    if trajectory_microbatch_size is None:
        trajectory_microbatch_size = group_size
    if (
        isinstance(trajectory_microbatch_size, bool)
        or not isinstance(trajectory_microbatch_size, int)
        or trajectory_microbatch_size <= 0
        or group_size % trajectory_microbatch_size
    ):
        raise SFTGRPOError(
            "collection trajectory_microbatch_size must be positive and divide group_size"
        )
    device = next(model.parameters()).device
    feature_count = _feature_count(model)
    config.validate(feature_count=feature_count)
    n_segments = len(environment.segments)
    if n_segments <= 1:
        raise SFTGRPOError("accepted environment must contain at least two segments")
    initial = initial_provider.provide(environment)
    histories: list[list[ConditioningAction]] = [[initial] for _ in range(group_size)]
    actions: list[list[SparseActionTrace]] = [[] for _ in range(group_size)]
    compact_to_original = _compact_to_original(model, device=device)
    conditioning_vocab_ids = _compact_to_original_host(model)
    original_feature_ids = (
        t.arange(feature_count, dtype=t.int64, device=device)
        if compact_to_original is None
        else compact_to_original
    )
    inputs_one, targets_one, seg_map_one, reward_mask_one = _environment_tensors(
        environment,
        batch_size=1,
        device=device,
    )

    shared_initial_pooled: t.Tensor | None = None
    for segment_index in range(n_segments - 1):
        padded_histories = [
            (
                *history,
                *([ConditioningAction((), (), ())] * (n_segments - len(history))),
            )
            for history in histories
        ]
        if segment_index == 0 and any(
            history != padded_histories[0] for history in padded_histories[1:]
        ):
            raise AssertionError("shared initial conditioning histories diverged")
        for start in range(0, group_size, trajectory_microbatch_size):
            end = start + trajectory_microbatch_size
            if segment_index == 0 and shared_initial_pooled is not None:
                decision_pooled = shared_initial_pooled
            else:
                feature_ids, magnitudes, conditioning_mask = _conditioning_tensors(
                    padded_histories[start:end],
                    n_segments=n_segments,
                    device=device,
                )
                inputs = inputs_one.expand(end - start, -1)
                seg_map = seg_map_one.expand(end - start, -1)
                fp_logits = _fp_forward(
                    model,
                    inputs,
                    feature_ids,
                    magnitudes,
                    conditioning_mask,
                    seg_map,
                )
                pooled = pool_fp_logits_per_segment(
                    fp_logits,
                    seg_map,
                    PositiveInt.parse(n_segments),
                    config.pool_mode,
                ).float()
                del fp_logits
                decision_pooled = pooled[:, segment_index]
                if segment_index == 0:
                    # C0 is constructed once per conversation and shared across
                    # all trajectories. Inputs and future all-masked slots are
                    # also identical, so every same-shaped microbatch would
                    # repeat this deterministic policy forward byte-for-byte.
                    shared_initial_pooled = decision_pooled
            if decision_pooled.shape != (end - start, feature_count):
                raise AssertionError("shared initial policy cache shape drifted")
            weighted_log_weights = (
                _weighted_log_weights_from_finite_forward(decision_pooled)
                if config.action_mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT
                else None
            )
            batched_weighted_traces: tuple[SparseActionTrace, ...] | None = None
            if weighted_log_weights is not None and config.action_policy_version == 2:
                positive_counts, negative_counts = _sample_counts_batch(
                    environment,
                    seed_context,
                    config,
                    rollout_start=start,
                    rollout_count=end - start,
                    segment_index=segment_index,
                )
                positive_log_weights, negative_log_weights = weighted_log_weights
                batched_weighted_traces = (
                    _sample_weighted_without_replacement_batch_from_log_weights(
                        positive_log_weights,
                        negative_log_weights,
                        positive_counts=positive_counts,
                        negative_counts=negative_counts,
                        generator=_batch_action_generator(
                            decision_pooled,
                            environment,
                            seed_context,
                            config,
                            rollout_start=start,
                            rollout_count=end - start,
                            segment_index=segment_index,
                        ),
                    )
                )
            for row, rollout_index in enumerate(range(start, end)):
                if batched_weighted_traces is not None:
                    trace = batched_weighted_traces[row]
                    actions[rollout_index].append(trace)
                    histories[rollout_index].append(
                        trace_to_conditioning(
                            trace,
                            compact_to_original=conditioning_vocab_ids,
                        )
                    )
                    continue
                positive_count, negative_count = _sample_counts(
                    environment,
                    seed_context,
                    config,
                    rollout_index=rollout_index,
                    segment_index=segment_index,
                )
                if weighted_log_weights is not None:
                    positive_log_weights, negative_log_weights = weighted_log_weights
                    trace = _sample_weighted_without_replacement_from_log_weights(
                        positive_log_weights[row],
                        negative_log_weights[row],
                        positive_count=positive_count,
                        negative_count=negative_count,
                        generator=_action_generator(
                            decision_pooled[row],
                            environment,
                            seed_context,
                            config,
                            rollout_index=rollout_index,
                            segment_index=segment_index,
                        ),
                    )
                else:
                    trace = _sample_action(
                        decision_pooled[row],
                        environment,
                        seed_context,
                        config,
                        rollout_index=rollout_index,
                        segment_index=segment_index,
                        positive_count=positive_count,
                        negative_count=negative_count,
                        original_feature_ids=original_feature_ids,
                    )
                actions[rollout_index].append(trace)
                histories[rollout_index].append(
                    trace_to_conditioning(
                        trace,
                        compact_to_original=conditioning_vocab_ids,
                    )
                )
        if segment_index == 0:
            shared_initial_pooled = None

    complete_histories = [tuple(history) for history in histories]
    reward_chunks = []
    for start in range(0, group_size, trajectory_microbatch_size):
        end = start + trajectory_microbatch_size
        feature_ids, magnitudes, conditioning_mask = _conditioning_tensors(
            complete_histories[start:end],
            n_segments=n_segments,
            device=device,
        )
        inputs = inputs_one.expand(end - start, -1)
        # EncoderDecoderGPT's per-token CE path flattens targets with ``view``.
        # ``expand`` is zero-copy and therefore non-contiguous once a real
        # trajectory microbatch has more than one row. Materialize only this
        # compact int64 target view rather than copying model activations.
        targets = targets_one.expand(end - start, -1).contiguous()
        seg_map = seg_map_one.expand(end - start, -1)
        reward_mask = reward_mask_one.expand(end - start, -1)
        per_token_ce = model(
            inputs,
            targets,
            feature_ids,
            magnitudes,
            conditioning_mask,
            seg_map,
            loss_reduction="none",
        )
        if not isinstance(per_token_ce, t.Tensor) or per_token_ce.numel() != targets.numel():
            raise SFTGRPOError("teacher-forced reward forward must return one CE value per target")
        per_token_ce = per_token_ce.reshape_as(targets).float()
        reward_chunks.append((-masked_mean_token_ce(per_token_ce, reward_mask)).detach())
    rewards = t.cat(reward_chunks)
    if not bool(t.isfinite(rewards).all()):
        raise SFTGRPOError("teacher-forced trajectory reward is non-finite")
    advantages = conversation_group_advantages(rewards)
    trajectories = tuple(
        TrajectoryTrace(
            rollout_index=rollout_index,
            conditioning=complete_histories[rollout_index],
            actions=tuple(actions[rollout_index]),
            reward=float(rewards[rollout_index].item()),
        )
        for rollout_index in range(group_size)
    )
    return ConversationGroupRollout(
        environment=environment,
        trajectories=trajectories,
        advantages=tuple(float(value) for value in advantages.tolist()),
        initial_conditioning_mode=initial_provider.mode,
        sae_loaded=initial_provider.sae_loaded,
    )


def backward_replay_group(
    model: Any,
    group: ConversationGroupRollout,
    *,
    config: GRPOPolicyConfig,
    trajectory_microbatch_size: int,
    loss_scale: float = 1.0,
    kl_beta: float = 0.0,
    reference_model: Any | None = None,
) -> ReplayMetrics:
    """Recompute sparse policy terms and accumulate microbatch backwards locally."""
    if getattr(model, "training", False):
        raise SFTGRPOError("differentiable replay requires deterministic eval-mode model")
    group_size = len(group.trajectories)
    if (
        isinstance(trajectory_microbatch_size, bool)
        or not isinstance(trajectory_microbatch_size, int)
        or trajectory_microbatch_size <= 0
        or group_size % trajectory_microbatch_size
    ):
        raise SFTGRPOError("trajectory_microbatch_size must be positive and divide group_size")
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise SFTGRPOError("loss_scale must be finite and positive")
    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise SFTGRPOError("kl_beta must be finite and non-negative")
    if kl_beta == 0 and reference_model is not None:
        raise SFTGRPOError(
            "kl_beta=0 requires reference_model=None so the no-reference path is exact"
        )
    if kl_beta > 0:
        if reference_model is None:
            raise SFTGRPOError("kl_beta>0 requires a frozen reference model")
        if getattr(reference_model, "training", False):
            raise SFTGRPOError("reference model must be in eval mode")
        if any(parameter.requires_grad for parameter in reference_model.parameters()):
            raise SFTGRPOError("reference model parameters must all be frozen")

    device = next(model.parameters()).device
    feature_count = _feature_count(model)
    config.validate(feature_count=feature_count)
    n_segments = len(group.environment.segments)
    inputs_all, _targets, seg_map_all, _reward_mask = _environment_tensors(
        group.environment,
        batch_size=group_size,
        device=device,
    )
    advantages = t.tensor(group.advantages, dtype=t.float32, device=device)
    if advantages.requires_grad or advantages.shape != (group_size,):
        raise SFTGRPOError("group advantages must be detached with shape (G,)")

    pg_total = 0.0
    kl_total = 0.0
    ratio_values = []
    live_decisions = 0
    for start in range(0, group_size, trajectory_microbatch_size):
        end = start + trajectory_microbatch_size
        micro_trajectories = group.trajectories[start:end]
        inputs = inputs_all[start:end]
        seg_map = seg_map_all[start:end]
        decision_ratios_by_row: list[list[t.Tensor]] = [[] for _ in micro_trajectories]
        decision_kls_by_row: list[list[t.Tensor]] = [[] for _ in micro_trajectories]
        for trajectory in micro_trajectories:
            if len(trajectory.actions) != n_segments - 1:
                raise SFTGRPOError("trajectory must contain one action for every non-final segment")
            if len(trajectory.conditioning) != n_segments:
                raise SFTGRPOError(
                    "trajectory conditioning history must contain every segment slot"
                )

        # Reconstruct each decision with the exact prefix history used during
        # behavior collection.  A single forward over the completed history is
        # semantically causal, but the packed encoder sees a different global
        # packed extent once future slots contain actions.  BF16 kernels may
        # consequently choose different reduction geometry and drift enough to
        # break the required behavior/replay equality even though routing does
        # not leak future features.  Keeping all future slots masked here also
        # makes the replay contract mechanically identical to collection.
        empty = ConditioningAction((), (), ())
        for segment_index in range(n_segments - 1):
            prefix_histories = [
                (
                    *trajectory.conditioning[: segment_index + 1],
                    *([empty] * (n_segments - segment_index - 1)),
                )
                for trajectory in micro_trajectories
            ]
            feature_ids, magnitudes, conditioning_mask = _conditioning_tensors(
                prefix_histories,
                n_segments=n_segments,
                device=device,
            )
            fp_logits = _fp_forward_replay(
                model,
                inputs,
                feature_ids,
                magnitudes,
                conditioning_mask,
                seg_map,
            )
            pooled = pool_fp_logits_per_segment(
                fp_logits,
                seg_map,
                PositiveInt.parse(n_segments),
                config.pool_mode,
            ).float()
            decision_pooled = pooled[:, segment_index]
            weighted_log_weights = (
                _weighted_log_weights_from_finite_forward(decision_pooled)
                if config.action_mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT
                else None
            )
            weighted_ratios = (
                _behavior_checked_replay_ratios_from_finite_log_weights(
                    decision_pooled,
                    *weighted_log_weights,
                    tuple(trajectory.actions[segment_index] for trajectory in micro_trajectories),
                    atol=config.replay_atol,
                    rtol=config.replay_rtol,
                )
                if weighted_log_weights is not None
                else None
            )
            reference_pooled = None
            if kl_beta > 0:
                assert reference_model is not None
                with t.no_grad():
                    reference_logits = _fp_forward(
                        reference_model,
                        inputs,
                        feature_ids,
                        magnitudes,
                        conditioning_mask,
                        seg_map,
                    )
                    reference_pooled = pool_fp_logits_per_segment(
                        reference_logits,
                        seg_map,
                        PositiveInt.parse(n_segments),
                        config.pool_mode,
                    ).float()

            for row, trajectory in enumerate(micro_trajectories):
                trace = trajectory.actions[segment_index]
                decision_logits = decision_pooled[row]
                if weighted_ratios is not None:
                    ratio = weighted_ratios[row]
                else:
                    ratio = behavior_checked_replay_ratio(
                        decision_logits,
                        trace,
                        atol=config.replay_atol,
                        rtol=config.replay_rtol,
                    )
                decision_ratios_by_row[row].append(ratio)
                if reference_pooled is not None:
                    decision_kls_by_row[row].append(
                        bernoulli_current_to_reference_kl(
                            decision_logits,
                            reference_pooled[row, segment_index],
                        )
                    )

        trajectory_ratios = []
        trajectory_kls = []
        for decision_ratios, decision_kls in zip(
            decision_ratios_by_row,
            decision_kls_by_row,
            strict=True,
        ):
            live_decisions += len(decision_ratios)
            trajectory_ratio = trajectory_mean_ratio(decision_ratios)
            trajectory_ratios.append(trajectory_ratio)
            ratio_values.extend(float(ratio.detach().item()) for ratio in decision_ratios)
            if decision_kls:
                trajectory_kls.append(t.stack(decision_kls).mean())
        stacked_ratios = t.stack(trajectory_ratios)
        micro_pg = -(advantages[start:end] * stacked_ratios).sum() / group_size
        micro_kl = t.stack(trajectory_kls).sum() / group_size if trajectory_kls else micro_pg * 0.0
        micro_total = micro_pg + kl_beta * micro_kl
        (micro_total * loss_scale).backward()
        pg_total += float(micro_pg.detach().item())
        kl_total += float(micro_kl.detach().item())

    total = pg_total + kl_beta * kl_total
    return ReplayMetrics(
        policy_gradient_loss=pg_total,
        kl_loss=kl_total,
        total_loss=total,
        ratio_min=min(ratio_values),
        ratio_max=max(ratio_values),
        live_decisions=live_decisions,
    )


def backward_ntp_group(
    model: Any,
    group: ConversationGroupRollout,
    *,
    trajectory_count: int,
    loss_scale: float,
) -> float:
    """Accumulate NTP gradients for the first ``trajectory_count`` rollouts.

    Returns the unweighted mean CE over those rollouts. ``loss_scale`` multiplies
    that mean, so doubling ``trajectory_count`` reduces variance without changing
    gradient magnitude. Each rollout is scored in its own batch-1 forward, which
    keeps only one differentiable LM graph alive at a time.
    """
    if not t.is_grad_enabled():
        raise SFTGRPOError("differentiable NTP backward requires gradients enabled")
    if getattr(model, "training", False):
        raise SFTGRPOError("differentiable NTP backward requires eval-mode model")
    if (
        isinstance(trajectory_count, bool)
        or not isinstance(trajectory_count, int)
        or not 1 <= trajectory_count <= len(group.trajectories)
    ):
        raise SFTGRPOError(
            "trajectory_count must be an integer in "
            f"[1, {len(group.trajectories)}], got {trajectory_count!r}"
        )
    if not math.isfinite(loss_scale) or loss_scale < 0:
        raise SFTGRPOError(f"NTP loss_scale must be finite and non-negative, got {loss_scale!r}")
    device = next(model.parameters()).device
    n_segments = len(group.environment.segments)
    # Loop-invariant: every rollout of a group shares one environment. The
    # conditioning is what varies. ``_conditioning_tensors`` already rejects a
    # history that does not tile the segments, so no separate length check here.
    inputs, targets, seg_map, reward_mask = _environment_tensors(
        group.environment,
        batch_size=1,
        device=device,
    )
    targets = targets.contiguous()
    total = 0.0
    for trajectory in group.trajectories[:trajectory_count]:
        feature_ids, magnitudes, mask = _conditioning_tensors(
            [trajectory.conditioning],
            n_segments=n_segments,
            device=device,
        )
        per_token_ce = model(
            inputs,
            targets,
            feature_ids,
            magnitudes,
            mask,
            seg_map,
            loss_reduction="none",
        )
        if not isinstance(per_token_ce, t.Tensor) or per_token_ce.numel() != targets.numel():
            raise SFTGRPOError("NTP forward must return one CE value per target")
        row = masked_mean_token_ce(per_token_ce.reshape_as(targets), reward_mask)
        ((row.mean() * loss_scale) / trajectory_count).backward()
        total += float(row.detach().item())
    return total / trajectory_count


__all__ = [
    "ConversationGroupRollout",
    "GRPOPolicyConfig",
    "InitialConditioningProvider",
    "InitialConditioningMode",
    "NTPObjective",
    "NoneInitialConditioningProvider",
    "ReplayMetrics",
    "SeedContext",
    "TrajectoryTrace",
    "assert_grpo_gradients_ready",
    "backward_ntp_group",
    "backward_replay_group",
    "collect_conversation_group",
    "configure_grpo_trainable_parameters",
    "resolve_train_feature_head_only",
]
