"""Sequential self-conditioning GRPO over authoritative structured-SFT rows.

This is a separate opt-in trainer. It never packs conversations and never
changes the existing conditioned-SFT or deployment self-conditioning paths.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import beartype
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.checkpoint_manager import (
    find_last_step,
    load_checkpoint,
    save_checkpoint,
)
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
)
from nanochat.flash_attention import has_fa4_backend
from nanochat.gpt import GPTConfig
from nanochat.hydra_cli import parse_config_name_and_overrides
from nanochat.optim import DistMuonAdamW
from nanochat.tokenizer import get_tokenizer
from steering._common.ddp import compute_init_for_checkpoint_resume, rank_synchronized_stage
from steering._common.lr_schedules import flat_then_linear_decay
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    feature_prediction_config_from_metadata,
    reject_removed_feature_prediction_keys,
    serialize_feature_prediction_config,
    validate_feature_prediction_resume,
)
from steering.conditional.encoder_decoder.dataloader import (
    build_dataloader_state_envelope,
    select_dataloader_rank_state,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    resolve_recorded_num_buckets,
)
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_checkpoint import (
    validate_grpo_checkpoint_eligibility,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import (
    OnlineSAEInitialConditioningProvider,
    OnlineSAELineage,
    build_online_selection_context_factory,
)
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    assert_grpo_resume_identity,
    build_grpo_resume_identity,
    reduce_grpo_step_metrics,
    run_grpo_optimizer_step,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    InitialConditioningMode,
    InitialConditioningProvider,
    NoneInitialConditioningProvider,
    NTPObjective,
    SeedContext,
    collect_conversation_group,
    configure_grpo_trainable_parameters,
    resolve_train_feature_head_only,
)
from steering.conditional.encoder_decoder.sm12_fa4 import admit_project_attention
from steering.conditional.feature_vocab import feature_vocab_from_state_dict
from steering.profiling.markers import (
    TimingRecorder,
    timed_region,
    timing_tags,
    use_recorder,
)
from steering.refined_types import PositiveFiniteNumber, PositiveInt, UnitInterval
from steering.sft.structured_dataset import (
    StagedEnvironmentBatch,
    StructuredEnvironmentCursor,
    StructuredRolloutArtifact,
    load_structured_rollout_artifact,
)
from steering.sft.structured_rollout import (
    RewardMaskMode,
    SegmentationMode,
    build_conversation_environment,
    rustbpe_tokenizer_sha256,
)

GRPO_LOOP_META_KEY = "structured_sft_grpo_v1"
DETERMINISTIC_KERNEL_MODE = (
    "hardware_fa4_sm90_nondeterministic_sm100_deterministic_"
    "sm12_native_nondeterministic_else_sdpa_deterministic_checkpoint_reexec_v3"
)
CHECKPOINT_PROCESS_MODE = "exec_after_nonfinal_save_v1"
_BASE_DETERMINISTIC_KERNEL_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


NTP_BLOCK_FIELDS = frozenset({"trajectories_per_group", "loss_weight", "decay_start", "fp_head_lr"})


@beartype
@dataclass(frozen=True)
class NTPSettings:
    """Resolved auxiliary NTP settings; ``None`` means disabled."""

    trajectories_per_group: PositiveInt
    loss_weight: PositiveFiniteNumber
    decay_start: UnitInterval
    fp_head_lr: PositiveFiniteNumber | None

    def identity_dict(self) -> dict[str, object]:
        """Serialize for the resume identity, next to the fields it pins.

        Mirrors ``GRPOBatchGeometry.identity_dict``. Keeping this beside the
        dataclass rather than inline in ``main`` is what makes it directly
        testable: a field routed to the wrong key would otherwise produce a
        self-consistent but wrong hash that only a real cross-resume could catch.
        """
        return {
            "version": 1,
            "trajectories_per_group": int(self.trajectories_per_group),
            "loss_weight": float(self.loss_weight),
            "decay_start": float(self.decay_start),
            "fp_head_lr": None if self.fp_head_lr is None else float(self.fp_head_lr),
        }


def resolve_ntp_settings(
    cfg: Mapping[str, Any],
    *,
    group_size: int,
) -> NTPSettings | None:
    """Resolve the opt-in ``ntp`` block, or ``None`` when it is absent.

    Absence is the whole disable mechanism: there is no separate enable flag and
    nothing inside the block to prove inert.
    """
    raw = cfg["ntp"]
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise SFTGRPOError(f"ntp must be a mapping or null, got {raw!r}")
    if set(raw) != NTP_BLOCK_FIELDS:
        raise SFTGRPOError(
            f"ntp must define exactly {sorted(NTP_BLOCK_FIELDS)}; "
            f"missing={sorted(NTP_BLOCK_FIELDS - set(raw))}, "
            f"unknown={sorted(set(raw) - NTP_BLOCK_FIELDS)}"
        )

    try:
        trajectories_per_group = PositiveInt.parse(raw["trajectories_per_group"])
        loss_weight = PositiveFiniteNumber.parse(raw["loss_weight"])
        decay_start = UnitInterval.parse(raw["decay_start"])
        raw_fp_head_lr = raw["fp_head_lr"]
        fp_head_lr = None if raw_fp_head_lr is None else PositiveFiniteNumber.parse(raw_fp_head_lr)
    except (TypeError, ValueError) as error:
        raise SFTGRPOError(f"invalid NTP setting: {error}") from error
    if trajectories_per_group > group_size:
        raise SFTGRPOError(
            "ntp.trajectories_per_group must be no greater than "
            f"group_size={group_size}, got {trajectories_per_group}"
        )
    return NTPSettings(
        trajectories_per_group=trajectories_per_group,
        loss_weight=loss_weight,
        decay_start=decay_start,
        fp_head_lr=fp_head_lr,
    )


def resolve_lr_decay_start(cfg: Mapping[str, Any]) -> float:
    """Validate the always-active optimizer LR decay start, a run fraction."""
    try:
        return float(UnitInterval.parse(cfg["lr_decay_start"]))
    except (TypeError, ValueError) as error:
        raise SFTGRPOError(f"invalid lr_decay_start: {error}") from error


def objective_weights(
    settings: NTPSettings | None,
    *,
    completed_step: int,
    num_iterations: int,
) -> NTPObjective | None:
    """Resolve the auxiliary objective's weight for one completed step.

    The policy side carries no loss weight of its own. Only the RATIO between the
    two objectives moves the update, so scaling both is redundant with scaling
    NTP alone -- and a loss weight is the wrong instrument for the policy's own
    decay anyway: under Muon/AdamW it barely changes the step size. The RL run's
    decay is an optimizer learning-rate schedule, driven by
    ``lr_decay_start`` in the training loop.
    """
    if settings is None:
        return None
    progress = completed_step / num_iterations
    return NTPObjective(
        trajectories_per_group=int(settings.trajectories_per_group),
        loss_weight=float(settings.loss_weight)
        * flat_then_linear_decay(progress, decay_start=float(settings.decay_start)),
    )


def _resolve_timing_profile(cfg: Mapping[str, Any]) -> tuple[bool, Path]:
    """Validate opt-in attribution settings without creating output state."""
    enabled = cfg.get("profile_timing", False)
    if not isinstance(enabled, bool):
        raise SFTGRPOError(f"profile_timing must be a boolean, got {enabled!r}")
    raw_output_dir = cfg.get(
        "profile_output_dir",
        "out/profiles/encdec_sft_grpo",
    )
    if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
        raise SFTGRPOError(f"profile_output_dir must be a non-empty string, got {raw_output_dir!r}")
    return enabled, Path(raw_output_dir)


@contextmanager
def _grpo_timing_scope(
    recorder: TimingRecorder | None,
    *,
    rank: int,
    step: int,
) -> Iterator[None]:
    """Activate rank/step-tagged timings only for an explicit profile arm."""
    if recorder is None:
        yield
        return
    with use_recorder(recorder), timing_tags(rank=rank, step=step):
        yield


def _target_attention_impl() -> str:
    """Resolve FA4/CuTE on Hopper+ and SDPA on earlier CUDA devices."""
    if t.cuda.is_available() and t.cuda.get_device_capability()[0] >= 9:
        # SM12 uses the native interface, checked by admit_project_attention.
        if t.cuda.get_device_capability()[0] != 12 and not has_fa4_backend():
            raise SFTGRPOError("Hopper-or-newer structured-SFT GRPO requires FA4/CuTE")
        return "fa4"
    return "sdpa"


def _deterministic_kernel_env() -> dict[str, str]:
    return {
        **_BASE_DETERMINISTIC_KERNEL_ENV,
        "NANOCHAT_ATTN_IMPL": _target_attention_impl(),
    }


def _reexec_for_deterministic_kernels() -> None:
    """Start the trainer with hardware-appropriate, resume-pinned attention."""
    required = _deterministic_kernel_env()
    if all(os.environ.get(name) == value for name, value in required.items()):
        return
    environment = os.environ.copy()
    environment.update(required)
    os.execve(
        sys.executable,
        _python_reexec_argv(sys.argv),
        environment,
    )
    raise SFTGRPOError("deterministic-kernel process re-exec unexpectedly returned")


def _python_reexec_argv(arguments: list[str]) -> list[str]:
    """Preserve module execution so fresh workers retain package import paths."""
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if main_spec is not None and main_spec.name:
        return [sys.executable, "-m", main_spec.name, *arguments[1:]]
    return [sys.executable, *arguments]


def _reset_step_compiler_state() -> None:
    """Clear non-checkpointed compiler counters before an optimizer step."""
    t.compiler.reset()


def _checkpoint_reexec_argv(completed_step: int) -> list[str]:
    """Replace the explicit resume override for a checkpoint-boundary exec."""
    if completed_step <= 0:
        raise SFTGRPOError("checkpoint re-exec requires a positive completed step")
    prefix = "resume_from_step="
    return _python_reexec_argv([
        *(argument for argument in sys.argv if not argument.startswith(prefix)),
        f"{prefix}{completed_step}",
    ])


def _reexec_from_checkpoint(completed_step: int) -> None:
    """Reset process-local CUDA state after every durable nonfinal checkpoint."""
    if t.distributed.is_available() and t.distributed.is_initialized():
        t.distributed.barrier()
    compute_cleanup()
    os.execve(
        sys.executable,
        _checkpoint_reexec_argv(completed_step),
        os.environ.copy(),
    )
    raise SFTGRPOError("checkpoint-boundary process re-exec unexpectedly returned")


def _configure_deterministic_algorithms(
    *,
    device_type: str,
    attention_impl: str,
) -> bool:
    """Use hard determinism except where SM90/SM12 FA4 backward forbids it."""
    deterministic = not (
        device_type == "cuda" and attention_impl == "fa4" and t.cuda.get_device_capability()[0] in (9, 12)
    )
    t.use_deterministic_algorithms(deterministic)
    if device_type == "cuda":
        t.set_float32_matmul_precision("highest")
    return deterministic


def _assert_attention_impl(*, device_type: str) -> str:
    """Verify and return the hardware-resolved attention identity."""
    if device_type != "cuda":
        return "cpu"
    expected = _target_attention_impl()
    actual = os.environ.get("NANOCHAT_ATTN_IMPL")
    if actual != expected:
        raise SFTGRPOError(
            f"structured-SFT GRPO requires attention backend {expected!r}, got {actual!r}"
        )
    if expected == "fa4":
        admit_project_attention(require_fa4=True)
    return expected


def _local_rollout_work(
    staged: StagedEnvironmentBatch,
    *,
    group_size: int,
) -> tuple[int, int]:
    """Return rank-local trajectories and scored target-token positions."""
    trajectories = len(staged.environments) * group_size
    target_tokens = (
        sum(len(environment.targets) for environment in staged.environments) * group_size
    )
    return trajectories, target_tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_checkpoint_sha256(checkpoint_dir: str, step: int) -> str:
    model_digest = _sha256_file(Path(checkpoint_dir) / f"model_{step:06d}.pt")
    meta_digest = _sha256_file(Path(checkpoint_dir) / f"meta_{step:06d}.json")
    return hashlib.sha256(f"{model_digest}\n{meta_digest}\n".encode()).hexdigest()


def _architecture_from_meta(
    meta: Mapping[str, Any],
) -> tuple[GPTConfig, EncoderConfig, CrossAttentionConfig]:
    decoder = GPTConfig(**dict(cast(Mapping[str, Any], meta["model_config"])))
    user_config = meta.get("user_config")
    user = user_config if isinstance(user_config, Mapping) else {}
    if isinstance(user.get("encoder"), Mapping) and isinstance(
        user.get("cross_attention"), Mapping
    ):
        encoder = EncoderConfig(**dict(cast(Mapping[str, Any], user["encoder"])))
        cross_attention = CrossAttentionConfig(
            **dict(cast(Mapping[str, Any], user["cross_attention"]))
        )
    elif isinstance(meta.get("encoder_config"), Mapping) and isinstance(
        meta.get("xattn_config"), Mapping
    ):
        encoder = EncoderConfig(**dict(cast(Mapping[str, Any], meta["encoder_config"])))
        cross_attention = CrossAttentionConfig(
            **dict(cast(Mapping[str, Any], meta["xattn_config"]))
        )
    else:
        raise SFTGRPOError("checkpoint does not recover encoder and cross-attention architecture")
    return decoder, encoder, cross_attention


def _feature_prediction_from_meta(
    meta: Mapping[str, Any],
) -> FeaturePredictionConfig:
    try:
        config = feature_prediction_config_from_metadata(meta)
    except (TypeError, ValueError) as error:
        raise SFTGRPOError(str(error)) from error
    if config is None:
        raise SFTGRPOError("checkpoint does not recover feature-prediction lineage")
    return config


def _build_model(
    model_data: Mapping[str, t.Tensor],
    meta: Mapping[str, Any],
    *,
    device: str | t.device,
) -> EncoderDecoderGPT:
    decoder, encoder, cross_attention = _architecture_from_meta(meta)
    n_sae_features, feature_vocab = feature_vocab_from_state_dict(model_data)
    feature_prediction = _feature_prediction_from_meta(meta)
    num_buckets = resolve_recorded_num_buckets(meta)
    with t.device("meta"):
        model = EncoderDecoderGPT(
            decoder,
            encoder,
            cross_attention,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            feature_pred_config=feature_prediction,
            feature_vocab=feature_vocab,
        )
    model.to_empty(device=device)
    model.init_buffers()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.assert_vocab_consistent()
    model.eval()
    return model


def _build_optimizer(
    model: EncoderDecoderGPT,
    cfg: DictConfig,
    *,
    fp_head_lr: float | None,
):
    if float(cfg.weight_decay) != 0.0:
        raise SFTGRPOError("structured-SFT GRPO requires optimizer weight_decay=0")
    optimizer = model.setup_optimizer(
        unembedding_lr=float(cfg.unembedding_lr) * float(cfg.lr_multiplier),
        embedding_lr=float(cfg.embedding_lr) * float(cfg.lr_multiplier),
        matrix_lr=float(cfg.matrix_lr) * float(cfg.lr_multiplier),
        weight_decay=0.0,
        encoder_embedding_lr=(float(cfg.encoder_embedding_lr) * float(cfg.lr_multiplier)),
        encoder_matrix_lr=float(cfg.encoder_matrix_lr) * float(cfg.lr_multiplier),
        xattn_matrix_lr=float(cfg.xattn_matrix_lr) * float(cfg.lr_multiplier),
        trainable_only=True,
        fp_head_lr=(None if fp_head_lr is None else fp_head_lr * float(cfg.lr_multiplier)),
    )
    for group in optimizer.param_groups:
        if float(group.get("weight_decay", 0.0)) != 0.0:
            raise SFTGRPOError("every structured-SFT GRPO optimizer group must have zero decay")
        group["lr"] = float(group["lr"]) * float(cfg.init_lr_frac)
        group["initial_lr"] = group["lr"]
    return optimizer


def _environment_builder(cfg: DictConfig, *, bos_token_id: int, context_length: int):
    segmentation_mode = SegmentationMode(str(cfg.segmentation_mode))
    reward_mask_mode = RewardMaskMode(str(cfg.reward_mask_mode))

    def build(row):
        return build_conversation_environment(
            row,
            bos_token_id=bos_token_id,
            context_length=context_length,
            segment_length=int(cfg.segment_length),
            segmentation_mode=segmentation_mode,
            reward_mask_mode=reward_mask_mode,
            min_reward_tokens=int(cfg.min_reward_tokens),
        )

    return build


def _cursor_geometry(cfg: DictConfig, geometry: GRPOBatchGeometry, context_length: int):
    return {
        **geometry.identity_dict(),
        "context_length": context_length,
        "segment_length": int(cfg.segment_length),
        "segmentation_mode": str(cfg.segmentation_mode),
        "reward_mask_mode": str(cfg.reward_mask_mode),
        "min_reward_tokens": int(cfg.min_reward_tokens),
    }


@t.no_grad()
def _evaluate(
    model: EncoderDecoderGPT,
    artifact: StructuredRolloutArtifact,
    cursor: StructuredEnvironmentCursor,
    *,
    cfg: DictConfig,
    policy,
    initial_provider: InitialConditioningProvider,
    completed_step: int,
    build,
) -> dict[str, float]:
    del artifact
    device = next(model.parameters()).device
    with rank_synchronized_stage(
        "structured-SFT GRPO validation collection",
        device=device,
    ):
        staged = cursor.stage(
            int(cfg.validation_conversations),
            prefetch=0,
            build=build,
        )
        rewards = []
        zero_variance = 0
        valid_reward_targets = 0
        for conversation_index, environment in enumerate(staged.environments):
            group = collect_conversation_group(
                model,
                environment,
                group_size=int(cfg.group_size),
                config=policy,
                seed_context=SeedContext(
                    "validation",
                    completed_step,
                    conversation_index,
                ),
                initial_provider=initial_provider,
                trajectory_microbatch_size=int(cfg.trajectory_microbatch_size),
            )
            rewards.extend(trajectory.reward for trajectory in group.trajectories)
            valid_reward_targets += sum(environment.reward_mask) * len(group.trajectories)
            zero_variance += int(all(value == 0.0 for value in group.advantages))
        reward_tensor = t.tensor(rewards, dtype=t.float64, device=device)
        aggregate = t.tensor(
            [
                reward_tensor.sum().item(),
                reward_tensor.square().sum().item(),
                reward_tensor.numel(),
                zero_variance,
                valid_reward_targets,
            ],
            dtype=t.float64,
            device=device,
        )
    if t.distributed.is_available() and t.distributed.is_initialized():
        t.distributed.all_reduce(aggregate, op=t.distributed.ReduceOp.SUM)
    with rank_synchronized_stage(
        "structured-SFT GRPO validation cursor commit",
        device=device,
    ):
        cursor.commit(staged)
    count = aggregate[2]
    mean = aggregate[0] / count
    variance = (aggregate[1] / count - mean.square()).clamp_min(0.0)
    result = {
        "reward_mean": float(mean.item()),
        "reward_std": float(variance.sqrt().item()),
        "masked_ce": float((-mean).item()),
        "valid_reward_targets": float(aggregate[4].item()),
        "zero_variance_groups": float(aggregate[3].item()),
    }
    if RewardMaskMode(str(cfg.reward_mask_mode)) is RewardMaskMode.ASSISTANT_ONLY:
        result["assistant_ce"] = result["masked_ce"]
    return result


def _checkpoint_meta(
    base_meta: Mapping[str, Any],
    *,
    step: int,
    identity: dict[str, object],
    train_cursor_envelope: Mapping[str, Any],
    validation_cursor_envelope: Mapping[str, Any],
    user_config: Mapping[str, Any],
    initial_conditioning_mode: InitialConditioningMode,
    sae_loaded: bool,
) -> dict[str, Any]:
    result = dict(base_meta)
    fp_config = serialize_feature_prediction_config(feature_prediction_config_from_metadata(base_meta))
    if fp_config is not None:
        result["feature_prediction_config"] = fp_config
        result["user_config"] = {**base_meta.get("user_config", {}), "feature_prediction": fp_config}
    result.update(
        {
            "step": step,
            GRPO_LOOP_META_KEY: {
                "version": 1,
                "completed_optimizer_step": step,
                "resume_identity": identity,
                "train_cursor": dict(train_cursor_envelope),
                "validation_cursor": dict(validation_cursor_envelope),
                "user_config": dict(user_config),
                "initial_conditioning_mode": initial_conditioning_mode.value,
                "sae_loaded": sae_loaded,
            },
        }
    )
    return result


def main() -> None:
    _reexec_for_deterministic_kernels()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "training")
    config_name, overrides = parse_config_name_and_overrides(
        sys.argv[1:],
        default_config_name="conditional_encdec_sft_grpo",
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    user_config_raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(user_config_raw, dict)
    user_config = cast(dict[str, Any], user_config_raw)
    reject_removed_feature_prediction_keys(user_config)
    train_feature_head_only = resolve_train_feature_head_only(user_config)
    profile_timing_enabled, profile_output_dir = _resolve_timing_profile(user_config)
    initial_conditioning_mode = InitialConditioningMode(str(cfg.initial_conditioning_mode))
    expected_sae_loaded = initial_conditioning_mode is InitialConditioningMode.ONLINE_SAE
    if bool(cfg.sae_loaded) is not expected_sae_loaded:
        raise SFTGRPOError(
            "sae_loaded must be false for initial_conditioning_mode=none and true "
            "for initial_conditioning_mode=online_sae"
        )
    if config_name == "conditional_encdec_sft_grpo_pilot" and (
        initial_conditioning_mode is not InitialConditioningMode.NONE or bool(cfg.sae_loaded)
    ):
        raise SFTGRPOError(
            "the experimental pilot requires initial_conditioning_mode=none and sae_loaded=false"
        )
    if float(cfg.kl_beta) < 0:
        raise SFTGRPOError("kl_beta must be non-negative")
    if int(cfg.num_iterations) <= 0:
        raise SFTGRPOError("num_iterations must be a positive optimizer-step count")
    if float(cfg.advantage_std_floor) != 1e-6:
        raise SFTGRPOError("v1 fixes advantage_std_floor at 1e-6")
    ntp_settings = resolve_ntp_settings(user_config, group_size=int(cfg.group_size))
    lr_decay_start = resolve_lr_decay_start(user_config)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    device_type = autodetect_device_type() if cfg.device_type == "" else str(cfg.device_type)
    ddp, rank, _local_rank, world_size, device = compute_init_for_checkpoint_resume(
        device_type,
        resume_step=int(cfg.resume_from_step),
    )
    del ddp
    try:
        attention_impl = _assert_attention_impl(device_type=device_type)
        deterministic_algorithms = _configure_deterministic_algorithms(
            device_type=device_type,
            attention_impl=attention_impl,
        )
        timing_recorder = TimingRecorder() if profile_timing_enabled else None
        geometry = GRPOBatchGeometry(
            int(cfg.global_conversations_per_step),
            int(cfg.group_size),
            int(cfg.device_conversation_batch_size),
            int(cfg.trajectory_microbatch_size),
            world_size,
        )
        base_dir = str(cfg.base_checkpoint_dir)
        base_step = find_last_step(base_dir) if cfg.model_step is None else int(cfg.model_step)
        base_model_data, _, base_meta = load_checkpoint(
            base_dir,
            base_step,
            device,
            load_optimizer=False,
            rank=rank,
        )
        base_sha256 = _resolved_checkpoint_sha256(base_dir, base_step)
        model = _build_model(base_model_data, base_meta, device=device)
        del base_model_data
        eligibility = validate_grpo_checkpoint_eligibility(
            model,
            base_meta,
            checkpoint_dir=base_dir,
            action_mode=ActionMode(str(cfg.action_mode)),
            seed=int(cfg.policy_seed),
            action_policy_version=int(cfg.action_policy_version),
        )
        policy = eligibility.policy
        manifest = configure_grpo_trainable_parameters(
            model,
            train_lm_head=ntp_settings is not None,
            train_feature_head_only=train_feature_head_only,
        )
        optimizer = _build_optimizer(
            model,
            cfg,
            fp_head_lr=None if ntp_settings is None else ntp_settings.fp_head_lr,
        )
        if not isinstance(optimizer, DistMuonAdamW):
            raise SFTGRPOError(
                "structured-SFT GRPO requires DistMuonAdamW on every topology; "
                "launch even a one-rank run with torchrun"
            )

        tokenizer = get_tokenizer()
        tokenizer_sha256 = rustbpe_tokenizer_sha256(tokenizer)
        initial_provider: InitialConditioningProvider
        if initial_conditioning_mode is InitialConditioningMode.NONE:
            initial_provider = NoneInitialConditioningProvider()
        else:
            # Apply the fixed-only policy before touching the training-host
            # feature tree: an unsupported turn base must report the policy
            # error, not a FileNotFoundError from a path this host lacks.
            online_lineage = OnlineSAELineage.from_checkpoint_meta(base_meta)
            online_lineage.assert_grpo_supported()
            online_lineage.validate_paths()
            selection_context_factory = build_online_selection_context_factory(
                online_lineage,
                eligibility.feature_contract,
            )
            initial_provider = OnlineSAEInitialConditioningProvider(
                online_lineage,
                decode_rustbpe=tokenizer.decode,
                selection_context_factory=selection_context_factory,
                device=device,
            )
        train_artifact = load_structured_rollout_artifact(
            str(cfg.train_artifact_dir),
            decode=tokenizer.decode,
            tokenizer_sha256=tokenizer_sha256,
            expected_split="train",
        )
        validation_artifact = load_structured_rollout_artifact(
            str(cfg.validation_artifact_dir),
            decode=tokenizer.decode,
            tokenizer_sha256=tokenizer_sha256,
            expected_split="val",
        )
        context_length = int(model.dec_config.sequence_len)
        segmentation_identity = _cursor_geometry(cfg, geometry, context_length)
        optimization_identity = {
            key: user_config[key]
            for key in (
                "num_iterations",
                "grad_clip_max_norm",
                "lr_multiplier",
                "init_lr_frac",
                "embedding_lr",
                "unembedding_lr",
                "matrix_lr",
                "encoder_embedding_lr",
                "encoder_matrix_lr",
                "xattn_matrix_lr",
                "weight_decay",
                # Scales the optimizer LR schedule for every run, NTP or not, so
                # exact resume must pin it like any other LR field.
                "lr_decay_start",
            )
        }
        optimization_identity["deterministic_kernel_mode"] = DETERMINISTIC_KERNEL_MODE
        optimization_identity["checkpoint_process_mode"] = CHECKPOINT_PROCESS_MODE
        optimization_identity["attention_impl"] = attention_impl
        optimization_identity["deterministic_algorithms"] = deterministic_algorithms
        resume_identity = build_grpo_resume_identity(
            base_checkpoint_sha256=base_sha256,
            train_artifact_sha256=train_artifact.identity_sha256,
            validation_artifact_sha256=validation_artifact.identity_sha256,
            geometry=geometry,
            parameter_manifest=manifest,
            policy=policy,
            segmentation=segmentation_identity,
            initial_conditioning_mode=initial_conditioning_mode.value,
            sae_loaded=expected_sae_loaded,
            kl_beta=float(cfg.kl_beta),
            reference_sha256=base_sha256 if float(cfg.kl_beta) > 0 else None,
            optimization=optimization_identity,
            ntp=None if ntp_settings is None else ntp_settings.identity_dict(),
        )
        output_dir = (
            Path(str(cfg.output_dir))
            if cfg.output_dir is not None
            else Path(base_dir) / "selfcond_grpo_checkpoints"
        )
        completed_step = 0
        train_cursor_state = None
        validation_cursor_state = None
        resume_step = int(cfg.resume_from_step)
        if resume_step != -1:
            if resume_step <= 0:
                raise SFTGRPOError("resume_from_step must be -1 or a positive step")
            resume_model, resume_optimizer, resume_meta = load_checkpoint(
                str(output_dir),
                resume_step,
                device,
                load_optimizer=True,
                rank=rank,
            )
            loop = resume_meta.get(GRPO_LOOP_META_KEY)
            if not isinstance(loop, Mapping):
                raise SFTGRPOError("resume checkpoint lacks structured-SFT GRPO loop state")
            assert_grpo_resume_identity(resume_identity, loop.get("resume_identity"))
            if loop.get("initial_conditioning_mode") != initial_conditioning_mode.value:
                raise SFTGRPOError(
                    "resume checkpoint initial-conditioning mode disagrees with identity"
                )
            if loop.get("sae_loaded") is not expected_sae_loaded:
                raise SFTGRPOError(
                    "resume checkpoint sae_loaded state disagrees with conditioning mode"
                )
            completed_step = int(loop["completed_optimizer_step"])
            if completed_step != resume_step:
                raise SFTGRPOError("resume filename step and loop step disagree")
            # Keep the Parameter objects already referenced by optimizer groups.
            # ``assign=True`` here would replace them after optimizer construction.
            validate_feature_prediction_resume(resume_meta, model.feature_pred_config)
            model.load_state_dict(resume_model, strict=True, assign=False)
            model.assert_vocab_consistent()
            configure_grpo_trainable_parameters(
                model,
                train_lm_head=ntp_settings is not None,
                train_feature_head_only=train_feature_head_only,
            )
            assert resume_optimizer is not None
            optimizer.load_state_dict(resume_optimizer)
            train_cursor_state = select_dataloader_rank_state(
                cast(Mapping[str, Any], loop["train_cursor"]),
                ddp_rank=rank,
                ddp_world_size=world_size,
            )
            validation_cursor_state = select_dataloader_rank_state(
                cast(Mapping[str, Any], loop["validation_cursor"]),
                ddp_rank=rank,
                ddp_world_size=world_size,
            )

        cursor_geometry = _cursor_geometry(cfg, geometry, context_length)
        train_cursor = StructuredEnvironmentCursor(
            train_artifact,
            rank=rank,
            world_size=world_size,
            geometry_identity={"namespace": "train", **cursor_geometry},
            state=train_cursor_state,
        )
        validation_cursor = StructuredEnvironmentCursor(
            validation_artifact,
            rank=rank,
            world_size=world_size,
            geometry_identity={"namespace": "validation", **cursor_geometry},
            state=validation_cursor_state,
        )
        build = _environment_builder(
            cfg,
            bos_token_id=tokenizer.get_bos_token_id(),
            context_length=context_length,
        )

        reference_model = None
        if float(cfg.kl_beta) > 0:
            reference_data, _, reference_meta = load_checkpoint(
                base_dir,
                base_step,
                device,
                load_optimizer=False,
                rank=rank,
            )
            reference_model = _build_model(reference_data, reference_meta, device=device)
            del reference_data
            reference_model.requires_grad_(False)
            reference_model.eval()
        logger.info(
            "structured-SFT GRPO startup: "
            f"initial_conditioning_mode={initial_conditioning_mode.value}, "
            f"sae_loaded={str(initial_provider.sae_loaded).lower()}, "
            f"action={policy.action_mode.value}, "
            f"kernel_mode={DETERMINISTIC_KERNEL_MODE}, "
            f"attention_impl={attention_impl}, "
            f"deterministic_algorithms={str(deterministic_algorithms).lower()}, "
            f"train_feature_head_only={str(train_feature_head_only).lower()}, "
            f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}, "
            f"total_parameters={sum(p.numel() for p in model.parameters())}, "
            f"group_size={geometry.group_size}, rank={rank}/{world_size}"
        )

        while completed_step < int(cfg.num_iterations):
            with _grpo_timing_scope(
                timing_recorder,
                rank=rank,
                step=completed_step,
            ):
                # Packed FlexAttention must remain compiled, but its in-process
                # recompile counters and fallback state are not checkpointed. Start
                # every optimizer step from the same cold compiler state so a
                # freshly resumed step selects the same kernels as an uninterrupted
                # step with identical sparse histories and shapes.
                with timed_region("grpo/compiler_reset"):
                    _reset_step_compiler_state()
                with timed_region("grpo/data_stage"):
                    staged = train_cursor.stage(
                        geometry.local_conversations_per_step,
                        prefetch=int(cfg.cursor_prefetch_conversations),
                        build=build,
                    )
                # The RL run's own LR decay. ``lr_decay_start=0.8`` reproduces
                # the original schedule exactly; it says WHEN decay begins as a
                # fraction of the run, and decay always ends at zero.
                progress = completed_step / int(cfg.num_iterations)
                multiplier = flat_then_linear_decay(progress, decay_start=lr_decay_start)
                for group in optimizer.param_groups:
                    group["lr"] = float(group["initial_lr"]) * multiplier
                ntp_objective = objective_weights(
                    ntp_settings,
                    completed_step=completed_step,
                    num_iterations=int(cfg.num_iterations),
                )
                device_object = t.device(device)
                if device_object.type == "cuda":
                    t.cuda.synchronize(device_object)
                    t.cuda.reset_peak_memory_stats(device_object)
                step_started = time.perf_counter()
                with timed_region("grpo/train_step"):
                    metrics = run_grpo_optimizer_step(
                        model,
                        optimizer,
                        staged=staged,
                        cursor=train_cursor,
                        geometry=geometry,
                        policy=policy,
                        completed_optimizer_step=completed_step,
                        parameter_manifest=manifest,
                        device=device,
                        grad_clip_max_norm=float(cfg.grad_clip_max_norm),
                        kl_beta=float(cfg.kl_beta),
                        reference_model=reference_model,
                        initial_provider=initial_provider,
                        ntp=ntp_objective,
                    )
                if device_object.type == "cuda":
                    t.cuda.synchronize(device_object)
                step_elapsed = time.perf_counter() - step_started
                with timed_region("grpo/metric_reduce"):
                    metrics = reduce_grpo_step_metrics(metrics, device=device)
            completed_step += 1
            logger.info(
                f"GRPO step={completed_step} reward={metrics.reward_mean:.6f} "
                f"pg={metrics.policy_gradient_loss:.8f} kl={metrics.kl_loss:.8f} "
                f"ratio=[{metrics.ratio_min:.6f},{metrics.ratio_max:.6f}] "
                f"grad_norm={metrics.gradient_norm:.6f} zero_var={metrics.zero_variance_groups} "
                f"requested_pos_neg={metrics.requested_positive}/{metrics.requested_negative} "
                f"realized_pos_neg={metrics.realized_positive}/{metrics.realized_negative} "
                f"shortage={metrics.shortage_actions} empty={metrics.empty_actions}"
                f" lrm={multiplier:.6f} ntp={metrics.ntp_loss:.6f} "
                f"w_ntp={0.0 if ntp_objective is None else ntp_objective.loss_weight:.6f}"
            )
            local_trajectories, local_target_tokens = _local_rollout_work(
                staged,
                group_size=geometry.group_size,
            )
            peak_allocated_bytes = (
                t.cuda.max_memory_allocated(device_object) if device_object.type == "cuda" else 0
            )
            peak_reserved_bytes = (
                t.cuda.max_memory_reserved(device_object) if device_object.type == "cuda" else 0
            )
            logger.info(
                f"GRPO runtime step={completed_step} elapsed_s={step_elapsed:.3f} "
                f"local_trajectories={local_trajectories} "
                f"local_target_tokens={local_target_tokens} "
                f"trajectories_per_s={local_trajectories / step_elapsed:.3f} "
                f"target_tokens_per_s={local_target_tokens / step_elapsed:.3f} "
                f"peak_allocated_gib={peak_allocated_bytes / 1024**3:.3f} "
                f"peak_reserved_gib={peak_reserved_bytes / 1024**3:.3f}"
            )
            if int(cfg.eval_every) > 0 and completed_step % int(cfg.eval_every) == 0:
                evaluation = _evaluate(
                    model,
                    validation_artifact,
                    validation_cursor,
                    cfg=cfg,
                    policy=policy,
                    initial_provider=initial_provider,
                    completed_step=completed_step,
                    build=build,
                )
                logger.info(f"GRPO validation step={completed_step}: {evaluation}")
            should_save = completed_step == int(cfg.num_iterations) or (
                int(cfg.save_every) > 0 and completed_step % int(cfg.save_every) == 0
            )
            if should_save:
                train_envelope = build_dataloader_state_envelope(
                    train_cursor.state.to_json_dict(),
                    ddp_rank=rank,
                    ddp_world_size=world_size,
                )
                validation_envelope = build_dataloader_state_envelope(
                    validation_cursor.state.to_json_dict(),
                    ddp_rank=rank,
                    ddp_world_size=world_size,
                )
                meta = _checkpoint_meta(
                    base_meta,
                    step=completed_step,
                    identity=resume_identity,
                    train_cursor_envelope=train_envelope,
                    validation_cursor_envelope=validation_envelope,
                    user_config=user_config,
                    initial_conditioning_mode=initial_conditioning_mode,
                    sae_loaded=initial_provider.sae_loaded,
                )
                save_checkpoint(
                    str(output_dir),
                    completed_step,
                    model.state_dict(),
                    optimizer.state_dict(),
                    meta,
                    rank=rank,
                )
                if completed_step < int(cfg.num_iterations):
                    logger.info(
                        "Resetting the GRPO worker from durable checkpoint "
                        f"step {completed_step} ({CHECKPOINT_PROCESS_MODE})"
                    )
                    _reexec_from_checkpoint(completed_step)
        if timing_recorder is not None:
            timing_path = profile_output_dir / f"grpo_timing_rank{rank}.csv"
            timing_recorder.to_csv(timing_path)
            logger.info(f"Exported structured-SFT GRPO timing CSV to {timing_path}")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
