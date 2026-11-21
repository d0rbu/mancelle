"""Optimizer-step orchestration and exact runtime geometry for SFT GRPO."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch as t

from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_training import (
    GRPOPolicyConfig,
    InitialConditioningProvider,
    NoneInitialConditioningProvider,
    NTPObjective,
    ReplayMetrics,
    SeedContext,
    assert_grpo_gradients_ready,
    backward_ntp_group,
    backward_replay_group,
    collect_conversation_group,
)
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.profiling.markers import timed_region
from steering.sft.clipping import apply_gradient_clipping
from steering.sft.structured_dataset import (
    StagedEnvironmentBatch,
    StructuredEnvironmentCursor,
)


@dataclass(frozen=True)
class GRPOBatchGeometry:
    """Conversation/group/microbatch relationships that define the objective."""

    global_conversations_per_step: int
    group_size: int
    device_conversation_batch_size: int
    trajectory_microbatch_size: int
    world_size: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SFTGRPOError(f"{name} must be a positive integer")
        if self.group_size <= 1:
            raise SFTGRPOError("group_size must be greater than one")
        if self.global_conversations_per_step % self.world_size:
            raise SFTGRPOError("global_conversations_per_step must divide evenly across world_size")
        local = self.local_conversations_per_step
        if local % self.device_conversation_batch_size:
            raise SFTGRPOError(
                "rank-local conversations must divide by device_conversation_batch_size"
            )
        if self.group_size % self.trajectory_microbatch_size:
            raise SFTGRPOError("trajectory_microbatch_size must divide group_size")

    @property
    def local_conversations_per_step(self) -> int:
        return self.global_conversations_per_step // self.world_size

    def identity_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class GRPOStepMetrics:
    """Step diagnostics, rank-local until explicitly reduced for logging."""

    policy_gradient_loss: float
    kl_loss: float
    total_loss: float
    reward_mean: float
    reward_std: float
    zero_variance_groups: int
    ratio_min: float
    ratio_max: float
    live_decisions: int
    requested_positive: int
    requested_negative: int
    realized_positive: int
    realized_negative: int
    shortage_actions: int
    empty_actions: int
    gradient_norm: float
    ntp_loss: float


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _count_selection_identity(
    selection: SamplingSelection | NegativeSamplingSelection,
) -> dict[str, object]:
    """Serialize geometric-count policy fields without Python ``repr`` drift."""
    result: dict[str, object] = {
        "distribution": selection.distribution.value,
        "mean": float(selection.mean),
        "std": float(selection.std),
        "min_k": int(selection.min_k),
        "max_k": (None if selection.max_k is None else int(selection.max_k)),
    }
    if isinstance(selection, SamplingSelection):
        result["mode"] = selection.mode.value
        delta_weights = selection.delta_weights
        result["delta_weights"] = None if delta_weights is None else delta_weights.value
        result["gradient"] = None
    return result


def build_grpo_resume_identity(
    *,
    base_checkpoint_sha256: str,
    train_artifact_sha256: str,
    validation_artifact_sha256: str,
    geometry: GRPOBatchGeometry,
    parameter_manifest: tuple[str, ...],
    policy: GRPOPolicyConfig,
    segmentation: dict[str, object],
    initial_conditioning_mode: str,
    sae_loaded: bool,
    kl_beta: float,
    reference_sha256: str | None,
    optimization: dict[str, object],
    ntp: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return every same-topology field that exact resume pins."""
    if initial_conditioning_mode == "none" and sae_loaded:
        raise SFTGRPOError("initial_conditioning_mode=none cannot record sae_loaded=true")
    if kl_beta == 0 and reference_sha256 is not None:
        raise SFTGRPOError("kl_beta=0 cannot record a reference checkpoint")
    payload: dict[str, object] = {
        "version": 1,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "train_artifact_sha256": train_artifact_sha256,
        "validation_artifact_sha256": validation_artifact_sha256,
        "geometry": geometry.identity_dict(),
        "parameter_manifest": list(parameter_manifest),
        "action_mode": policy.action_mode.value,
        "action_policy_version": policy.action_policy_version,
        "policy_seed": policy.seed,
        "positive_counts": _count_selection_identity(policy.positive_counts),
        "negative_counts": _count_selection_identity(policy.negative_counts),
        "segmentation": segmentation,
        "initial_conditioning_mode": initial_conditioning_mode,
        "sae_loaded": sae_loaded,
        "kl_beta": kl_beta,
        "reference_sha256": reference_sha256,
        "optimization": optimization,
        "kl_direction": "bernoulli_current_to_reference",
        "kl_reduction": "features_decisions_trajectories_conversations_mean",
    }
    if ntp is not None:
        payload["ntp"] = dict(ntp)
    return {**payload, "sha256": _sha256_json(payload)}


def assert_grpo_resume_identity(
    expected: dict[str, object],
    recorded: object,
    *,
    caller: str = "structured-SFT GRPO",
) -> None:
    """Reject any exact-resume identity drift before cursor or optimizer use."""
    if isinstance(recorded, Mapping) and recorded.get("parameter_manifest") != expected.get(
        "parameter_manifest"
    ):
        raise SFTGRPOError(
            f"{caller} resume identity mismatch; trainable parameter manifest "
            "differs. Exact resume requires the same training scope "
            "(train_feature_head_only and NTP settings)."
        )
    if recorded != expected:
        raise SFTGRPOError(
            f"{caller} resume identity mismatch; exact resume requires "
            "identical checkpoint, artifacts, policy, topology, and batch geometry"
        )


def _combine_metrics(metrics: list[ReplayMetrics]) -> tuple[float, float, float, float, float, int]:
    if not metrics:
        raise SFTGRPOError("optimizer step requires at least one local conversation")
    scale = 1.0 / len(metrics)
    return (
        sum(metric.policy_gradient_loss for metric in metrics) * scale,
        sum(metric.kl_loss for metric in metrics) * scale,
        sum(metric.total_loss for metric in metrics) * scale,
        min(metric.ratio_min for metric in metrics),
        max(metric.ratio_max for metric in metrics),
        sum(metric.live_decisions for metric in metrics),
    )


def reduce_grpo_step_metrics(
    metrics: GRPOStepMetrics,
    *,
    device: str | t.device,
) -> GRPOStepMetrics:
    """Return global diagnostics while preserving conversation-equal reductions."""
    if not t.distributed.is_available() or not t.distributed.is_initialized():
        return metrics
    world_size = t.distributed.get_world_size()
    second_reward_moment = metrics.reward_std**2 + metrics.reward_mean**2
    sums = t.tensor(
        [
            metrics.policy_gradient_loss,
            metrics.kl_loss,
            metrics.total_loss,
            metrics.reward_mean,
            second_reward_moment,
            metrics.zero_variance_groups,
            metrics.live_decisions,
            metrics.requested_positive,
            metrics.requested_negative,
            metrics.realized_positive,
            metrics.realized_negative,
            metrics.shortage_actions,
            metrics.empty_actions,
            metrics.ntp_loss,
        ],
        dtype=t.float64,
        device=device,
    )
    t.distributed.all_reduce(sums, op=t.distributed.ReduceOp.SUM)
    means = sums[:5] / world_size
    bounds = t.tensor(
        [metrics.ratio_min, metrics.ratio_max, metrics.gradient_norm],
        dtype=t.float64,
        device=device,
    )
    t.distributed.all_reduce(bounds[:1], op=t.distributed.ReduceOp.MIN)
    t.distributed.all_reduce(bounds[1:], op=t.distributed.ReduceOp.MAX)
    reward_variance = max(0.0, float(means[4].item()) - float(means[3].item()) ** 2)
    return GRPOStepMetrics(
        policy_gradient_loss=float(means[0].item()),
        kl_loss=float(means[1].item()),
        total_loss=float(means[2].item()),
        reward_mean=float(means[3].item()),
        reward_std=math.sqrt(reward_variance),
        zero_variance_groups=int(sums[5].item()),
        ratio_min=float(bounds[0].item()),
        ratio_max=float(bounds[1].item()),
        live_decisions=int(sums[6].item()),
        requested_positive=int(sums[7].item()),
        requested_negative=int(sums[8].item()),
        realized_positive=int(sums[9].item()),
        realized_negative=int(sums[10].item()),
        shortage_actions=int(sums[11].item()),
        empty_actions=int(sums[12].item()),
        gradient_norm=float(bounds[2].item()),
        ntp_loss=float(sums[13].item()) / world_size,
    )


def run_grpo_optimizer_step(
    model: Any,
    optimizer: Any,
    *,
    staged: StagedEnvironmentBatch,
    cursor: StructuredEnvironmentCursor,
    geometry: GRPOBatchGeometry,
    policy: GRPOPolicyConfig,
    completed_optimizer_step: int,
    parameter_manifest: tuple[str, ...],
    device: str | t.device,
    grad_clip_max_norm: float = 1.0,
    kl_beta: float = 0.0,
    reference_model: Any | None = None,
    initial_provider: InitialConditioningProvider | None = None,
    ntp: NTPObjective | None = None,
) -> GRPOStepMetrics:
    """Collect, replay, synchronize, step once, then commit the data cursor."""
    if len(staged.environments) != geometry.local_conversations_per_step:
        raise SFTGRPOError(
            "staged environment count does not equal rank-local conversation geometry"
        )
    if not math.isfinite(grad_clip_max_norm) or grad_clip_max_norm <= 0:
        raise SFTGRPOError("GRPO global gradient clipping must be finite and positive")
    with timed_region("grpo/zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    replay_metrics: list[ReplayMetrics] = []
    ntp_losses: list[float] = []
    rewards: list[float] = []
    zero_variance_groups = 0
    requested_positive = 0
    requested_negative = 0
    realized_positive = 0
    realized_negative = 0
    shortage_actions = 0
    empty_actions = 0
    if initial_provider is None:
        initial_provider = NoneInitialConditioningProvider()
    with rank_synchronized_stage(
        "structured-SFT GRPO collection and replay",
        device=device,
    ):
        for batch_start in range(
            0,
            len(staged.environments),
            geometry.device_conversation_batch_size,
        ):
            batch = staged.environments[
                batch_start : batch_start + geometry.device_conversation_batch_size
            ]
            for batch_offset, environment in enumerate(batch):
                conversation_index = batch_start + batch_offset
                with timed_region("grpo/collection"):
                    group = collect_conversation_group(
                        model,
                        environment,
                        group_size=geometry.group_size,
                        config=policy,
                        seed_context=SeedContext(
                            "train",
                            completed_optimizer_step,
                            conversation_index,
                        ),
                        initial_provider=initial_provider,
                        trajectory_microbatch_size=geometry.trajectory_microbatch_size,
                    )
                group_rewards = [trajectory.reward for trajectory in group.trajectories]
                rewards.extend(group_rewards)
                for trajectory in group.trajectories:
                    for trace in trajectory.actions:
                        requested_positive += trace.requested_positive
                        requested_negative += trace.requested_negative
                        realized_positive += len(trace.positive_rows)
                        realized_negative += len(trace.negative_rows)
                        shortage_actions += int(
                            len(trace.positive_rows) < trace.requested_positive
                            or len(trace.negative_rows) < trace.requested_negative
                        )
                        empty_actions += int(trace.selected_count == 0)
                if all(advantage == 0.0 for advantage in group.advantages):
                    zero_variance_groups += 1
                with timed_region("grpo/replay_and_backward"):
                    replay_metrics.append(
                        backward_replay_group(
                            model,
                            group,
                            config=policy,
                            trajectory_microbatch_size=(geometry.trajectory_microbatch_size),
                            loss_scale=1.0 / geometry.local_conversations_per_step,
                            kl_beta=kl_beta,
                            reference_model=reference_model,
                        )
                    )
                if ntp is not None:
                    with timed_region("grpo/ntp_backward"):
                        ntp_losses.append(
                            backward_ntp_group(
                                model,
                                group,
                                trajectory_count=ntp.trajectories_per_group,
                                loss_scale=(
                                    ntp.loss_weight / geometry.local_conversations_per_step
                                ),
                            )
                        )
    with rank_synchronized_stage(
        "structured-SFT GRPO gradient readiness and clipping",
        device=device,
    ):
        with timed_region("grpo/gradient_readiness_and_clip"):
            assert_grpo_gradients_ready(
                model,
                expected_manifest=parameter_manifest,
            )
            norm = apply_gradient_clipping(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                grad_clip_max_norm,
            )
        assert norm is not None
        gradient_norm = float(norm.detach().item())
        if not math.isfinite(gradient_norm):
            raise SFTGRPOError("structured-SFT GRPO gradient norm is non-finite")
    with timed_region("grpo/optimizer_step"):
        optimizer.step()
    with rank_synchronized_stage(
        "structured-SFT GRPO post-step cursor commit",
        device=device,
    ):
        with timed_region("grpo/cursor_commit"):
            cursor.commit(staged)

    pg, kl, total, ratio_min, ratio_max, live_decisions = _combine_metrics(replay_metrics)
    ntp_loss = sum(ntp_losses) / len(ntp_losses) if ntp_losses else 0.0
    reward_tensor = t.tensor(rewards, dtype=t.float32)
    return GRPOStepMetrics(
        policy_gradient_loss=pg,
        kl_loss=kl,
        total_loss=total,
        reward_mean=float(reward_tensor.mean().item()),
        reward_std=float(reward_tensor.std(unbiased=False).item()),
        zero_variance_groups=zero_variance_groups,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        live_decisions=live_decisions,
        requested_positive=requested_positive,
        requested_negative=requested_negative,
        realized_positive=realized_positive,
        realized_negative=realized_negative,
        shortage_actions=shortage_actions,
        empty_actions=empty_actions,
        gradient_norm=gradient_norm,
        ntp_loss=ntp_loss,
    )


__all__ = [
    "GRPOBatchGeometry",
    "GRPOStepMetrics",
    "assert_grpo_resume_identity",
    "build_grpo_resume_identity",
    "reduce_grpo_step_metrics",
    "run_grpo_optimizer_step",
]
