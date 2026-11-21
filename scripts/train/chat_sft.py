"""Decoder-only supervised fine-tuning for nanochat and pinned HF models."""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
import transformers
import wandb
from hydra import compose, initialize_config_dir
from loguru import logger
from transformers import PreTrainedModel

from nanochat.checkpoint_manager import ModelPhase, load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    configure_wandb_step_metric,
    get_base_dir,
    get_peak_flops,
)
from nanochat.dataset import DatasetSplit
from nanochat.gpt import GPT
from nanochat.hydra_cli import parse_config_name_and_overrides
from nanochat.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.tokenizer import get_token_bytes
from steering._common.ddp import (
    broadcast_bitmap_from_rank_zero,
    rank_synchronized_stage,
    sync_stop_and_progress,
)
from steering._common.lr_schedules import sft_flat_decay
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.config_payloads import resolved_config_dict
from steering.profiling import (
    StartupTrace,
    TimingRecorder,
    timed_region,
    timing_tags,
    use_recorder,
)
from steering.sft.clipping import apply_gradient_clipping, resolve_grad_clip_max_norm
from steering.sft.hf_checkpoint import (
    publish_hf_checkpoint_distributed,
    repository_version,
    validate_export_smoke,
)
from steering.sft.hf_contract import (
    HF_SFT_FORMAT_VERSION,
    DecoderSFTBackend,
    HFAttentionImplementation,
    HFDecoderContract,
    HFDistributedImplementation,
)
from steering.sft.hf_decoder import (
    HFDecoderModelAdapter,
    HFSFTTokenizer,
    build_hf_optimizer,
    configure_generation_contract,
    distributed_loss_backward_scale,
    load_hf_config_and_tokenizer,
    load_hf_model_in_local_rank_slots,
    prepare_hf_training_model,
    reduce_global_loss_mean,
    reduce_global_target_counts,
    resolve_hf_output_root,
    validate_decoder_sft_config,
    validated_local_world_size,
)
from steering.sft.hf_restart import (
    HFRestartDescriptor,
    HFRestartIdentity,
    assert_distributed_hf_restart_identity,
    build_hf_restart_identity,
    capture_hf_restart_runtime,
    load_hf_restart_model_in_local_rank_slots,
    load_hf_restart_rank_payload_in_local_rank_slots,
    restore_hf_restart_rng,
    save_hf_restart_distributed,
    validate_hf_restart,
    validate_hf_restart_runtime,
)
from steering.sft.iteration import (
    SFTGeneratorProgress,
    resolve_sft_loop_position,
    sft_data_generator_bos_bestfit_core,
    should_save_checkpoint,
    validate_decoder_sft_dataloader_state,
)
from steering.sft.loss_mask import parse_loss_mask_mode
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft.token_bytes import compute_hf_token_bytes
from steering.sft_schema import SFTSplit
from tasks.mixtures import (
    MIXTURE_PARAM_KEYS,
    SFTMixture,
    build_task_mixtures,
    select_mixture_overrides,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


@dataclass(frozen=True)
class _HFProfileSettings:
    timing: bool
    trace_steps: frozenset[int]
    trace_ranks: frozenset[int]
    output_dir: Path


def _strict_int_list(value: object, *, name: str, minimum: int) -> frozenset[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of integers; got {value!r}")
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ValueError(f"{name} entries must be integers >= {minimum}; got {item!r}")
        parsed.append(item)
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} must not contain duplicates")
    return frozenset(parsed)


def _resolve_hf_profile_settings(cfg: Any, *, world_size: int) -> _HFProfileSettings:
    if not isinstance(cfg.profile_timing, bool):
        raise ValueError(f"profile_timing must be a boolean; got {cfg.profile_timing!r}")
    steps = _strict_int_list(cfg.profile_trace_steps, name="profile_trace_steps", minimum=1)
    ranks = _strict_int_list(cfg.profile_trace_ranks, name="profile_trace_ranks", minimum=0)
    if any(rank >= world_size for rank in ranks):
        raise ValueError(
            f"profile_trace_ranks must be smaller than world_size={world_size}; got {sorted(ranks)}"
        )
    if int(cfg.num_iterations) > 0 and any(step > int(cfg.num_iterations) for step in steps):
        raise ValueError(
            "profile_trace_steps must not exceed the configured optimizer-step "
            f"horizon {int(cfg.num_iterations)}; got {sorted(steps)}"
        )
    if bool(steps) != bool(ranks):
        raise ValueError(
            "profile_trace_steps and profile_trace_ranks must either both be "
            "empty or both be nonempty"
        )
    output = cfg.profile_output_dir
    if not isinstance(output, str) or not output.strip():
        raise ValueError("profile_output_dir must be a nonempty path string")
    return _HFProfileSettings(
        timing=cfg.profile_timing,
        trace_steps=steps,
        trace_ranks=ranks,
        output_dir=Path(output),
    )


@contextmanager
def _maybe_profile_hf_step(
    settings: _HFProfileSettings,
    *,
    completed_step: int,
    rank: int,
    device: str | t.device,
) -> Iterator[None]:
    if completed_step not in settings.trace_steps:
        yield
        return
    profiler_context: Any | None = None
    profiler: Any | None = None
    with rank_synchronized_stage(f"HF profiler start step {completed_step}", device=device):
        if rank in settings.trace_ranks:
            activities = [t.profiler.ProfilerActivity.CPU]
            if t.cuda.is_available():
                activities.append(t.profiler.ProfilerActivity.CUDA)
            profiler_context = t.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            profiler = profiler_context.__enter__()
    try:
        yield
    finally:
        exception = sys.exc_info()
        with rank_synchronized_stage(
            f"HF profiler publication step {completed_step}", device=device
        ):
            if profiler_context is not None:
                profiler_context.__exit__(*exception)
                assert profiler is not None
                settings.output_dir.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(
                    str(
                        settings.output_dir
                        / f"chat_sft_trace_rank{rank}_step{completed_step}.json"
                    )
                )


@contextmanager
def _optional_startup_phase(
    trace: StartupTrace | None,
    name: str,
) -> Iterator[None]:
    if trace is None:
        yield
        return
    with trace.phase(name):
        yield


def _record_hf_paid_step(
    recorder: TimingRecorder,
    *,
    rank: int,
    completed_step: int,
    step_s: float,
    scheduled_tokens: int,
    nonpadding_tokens: int,
    supervised_tokens: int,
    loss: float,
    algorithmic_mfu_percent: float,
    peak_allocated_bytes: int,
) -> None:
    """Record the already-synchronized outer step without another CUDA sync."""
    recorder.append(
        "chat_sft/paid_step",
        max(1, int(step_s * 1_000_000_000)),
        {
            "rank": rank,
            "step": completed_step,
            "scheduled_tokens": scheduled_tokens,
            "nonpadding_tokens": nonpadding_tokens,
            "supervised_tokens": supervised_tokens,
            "loss": loss,
            "scheduled_tok_per_sec": scheduled_tokens / step_s,
            "nonpadding_tok_per_sec": nonpadding_tokens / step_s,
            "supervised_tok_per_sec": supervised_tokens / step_s,
            "algorithmic_mfu_percent": algorithmic_mfu_percent,
            "peak_allocated_bytes": peak_allocated_bytes,
        },
    )


def _hf_persistence_due(
    *,
    completed_steps: int,
    last_step: bool,
    dry_run: bool,
    portable_save_every: int,
    restart_save_every: int,
    loaded_restart_step: int | None,
) -> tuple[bool, bool]:
    """Return portable/restart save decisions for one committed HF boundary."""
    if dry_run or (loaded_restart_step is not None and completed_steps == loaded_restart_step):
        return False, False
    portable_due = last_step or (
        portable_save_every > 0
        and completed_steps > 0
        and completed_steps % portable_save_every == 0
    )
    restart_due = (
        restart_save_every > 0
        and completed_steps > 0
        and (last_step or completed_steps % restart_save_every == 0)
    )
    return portable_due, restart_due


def _distributed_directory_exists(
    path: Path,
    *,
    device: str | t.device,
) -> bool:
    """Return one rank-zero filesystem decision identically on every rank."""

    def inspect() -> t.Tensor:
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"expected a directory or absent path, got {path}")
        return t.tensor([path.is_dir()], dtype=t.bool)

    result = broadcast_bitmap_from_rank_zero(
        inspect,
        numel=1,
        device=device,
        stage="HF portable checkpoint existence",
    )
    return bool(result.item())


def _hf_metadata(
    *,
    cfg: Any,
    user_config: dict[str, Any],
    completed_step: int,
    val_bpb: float,
    min_val_bpb: float,
    ddp_world_size: int,
    grad_accum_steps: int,
    grad_clip_max_norm: float | None,
    train_fingerprint: Mapping[str, Any],
    val_fingerprint: Mapping[str, Any],
    raw_model: PreTrainedModel,
    contract: HFDecoderContract,
    attention_implementation: HFAttentionImplementation,
    distributed_implementation: HFDistributedImplementation,
) -> dict[str, Any]:
    hf = cfg.huggingface
    return {
        "format": "huggingface_standard_sft",
        "format_version": HF_SFT_FORMAT_VERSION,
        "resumable": False,
        "completed_optimizer_step": completed_step,
        "latest_validation_bpb": val_bpb if math.isfinite(val_bpb) else None,
        "minimum_validation_bpb": min_val_bpb if math.isfinite(min_val_bpb) else None,
        "source": {
            "model_id": str(hf.model_id),
            "revision": str(hf.revision),
            "architecture": contract.architecture,
            "model_type": contract.model_type,
        },
        "output": {
            "architecture": type(raw_model).__name__,
            "model_type": str(raw_model.config.model_type),
        },
        "backend": DecoderSFTBackend.HUGGINGFACE.value,
        "family": contract.family.value,
        "attention_implementation": attention_implementation.value,
        "gradient_checkpointing": bool(hf.gradient_checkpointing),
        "precision": {
            "parameters": "float32",
            "optimizer_moments": "float32",
            "autocast": "bfloat16",
            "export": "bfloat16",
        },
        "optimizer": {
            "name": (
                "DistMuonAdamW"
                if distributed_implementation is HFDistributedImplementation.DIST_MUON_ADAMW
                and ddp_world_size > 1
                else "MuonAdamW"
                if distributed_implementation is HFDistributedImplementation.DIST_MUON_ADAMW
                else "AdamW(fused=True)"
            ),
            "parameter_groups": ["adamw"],
            "learning_rate": float(hf.learning_rate),
            "betas": [float(value) for value in hf.adam_betas],
            "epsilon": float(hf.adam_eps),
            "weight_decay": float(cfg.weight_decay),
            "gradient_clip_max_norm": grad_clip_max_norm,
            "distributed_implementation": distributed_implementation.value,
            "distributed_clip_semantics": (
                "local_before_optimizer_reduction"
                if distributed_implementation is HFDistributedImplementation.DIST_MUON_ADAMW
                else "after_ddp_global_average"
            ),
        },
        "batch": {
            "world_size": ddp_world_size,
            "device_batch_size": int(cfg.device_batch_size),
            "max_seq_len": int(cfg.max_seq_len),
            "total_batch_size": int(cfg.total_batch_size),
            "gradient_accumulation_steps": grad_accum_steps,
        },
        "mixture": str(cfg.mixture),
        "sft_loss_mask": str(cfg.sft_loss_mask),
        "user_config": user_config,
        "mixture_fingerprints": {
            "train": train_fingerprint,
            "validation": val_fingerprint,
        },
        "chat_template_sha256": contract.chat_template_sha256,
        "structural_token_ids": dict(contract.structural_token_ids),
        "vocabulary": {
            "model": contract.model_vocab_size,
            "tokenizer": contract.tokenizer_vocab_size,
        },
        "tied_embeddings": contract.tied_embeddings,
        "versions": {
            "transformers": transformers.__version__,
            "torch": t.__version__,
            "repository": repository_version(),
        },
    }


def _broadcast_hf_token_bytes(
    tokenizer: HFSFTTokenizer,
    *,
    master_process: bool,
    device: str | t.device,
) -> t.Tensor:
    token_bytes_cpu: t.Tensor | None = None
    with rank_synchronized_stage("HF token-byte accounting", device=device):
        if master_process:
            token_bytes_cpu = t.tensor(
                compute_hf_token_bytes(
                    tokenizer,
                    additional_zero_ids=tokenizer.contract.structural_token_ids.values(),
                ),
                dtype=t.int32,
            )
    if master_process:
        assert token_bytes_cpu is not None
        token_bytes = token_bytes_cpu.to(device)
    else:
        token_bytes = t.empty(tokenizer.get_vocab_size(), dtype=t.int32, device=device)
    if t.distributed.is_initialized():
        t.distributed.broadcast(token_bytes, src=0)
    return token_bytes


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
    config_name, overrides = parse_config_name_and_overrides(
        args,
        default_config_name="chat_sft",
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    user_config = resolved_config_dict(cfg)
    backend = validate_decoder_sft_config(cfg)
    loss_mask_mode = parse_loss_mask_mode(cfg.sft_loss_mask)
    grad_clip_max_norm = resolve_grad_clip_max_norm(cfg.grad_clip_max_norm)
    logger.info(f"SFT backend: {backend.value}; loss mask: {loss_mask_mode.value}")

    startup_trace = (
        StartupTrace(event_prefix="chat_sft") if backend is DecoderSFTBackend.HUGGINGFACE else None
    )
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    with _optional_startup_phase(startup_trace, "compute_init"):
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    if backend is DecoderSFTBackend.HUGGINGFACE:
        validate_decoder_sft_config(cfg, world_size=ddp_world_size)
    profile_settings = _resolve_hf_profile_settings(cfg, world_size=ddp_world_size)
    timing_recorder = (
        TimingRecorder()
        if backend is DecoderSFTBackend.HUGGINGFACE and profile_settings.timing
        else None
    )
    if backend is DecoderSFTBackend.HUGGINGFACE and device_type != "cuda":
        raise ValueError("model_backend=huggingface requires a CUDA device for BF16 autocast")
    ptdtype = t.float32 if cfg.dtype == "float32" else t.bfloat16
    autocast_ctx = (
        t.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )
    synchronize = t.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = t.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    if device_type == "cuda":
        gpu_device_name = t.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        logger.info(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")

    native_meta: dict[str, Any] | None = None
    hf_tokenizer: HFSFTTokenizer | None = None
    hf_contract: HFDecoderContract | None = None
    hf_raw_model: PreTrainedModel | None = None
    hf_attention: HFAttentionImplementation | None = None
    hf_distributed: HFDistributedImplementation | None = None
    hf_local_world_size = 1
    hf_output_root: Path | None = None
    hf_restart_identity: HFRestartIdentity | None = None
    hf_restart_descriptor: HFRestartDescriptor | None = None
    hf_restart_rank_payload: dict[str, Any] | None = None
    optimizer: t.optim.Optimizer
    if backend is DecoderSFTBackend.NANOCHAT:
        with rank_synchronized_stage("native decoder source preflight", device=device):
            native_model, tokenizer, native_meta = load_model(
                cfg.source,
                device,
                phase=ModelPhase.TRAIN,
                model_tag=cfg.model_tag,
                step=cfg.model_step,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
            reject_feature_clusters_metadata(native_meta, "chat_sft source checkpoint")
            if not isinstance(native_model, GPT):
                raise TypeError(
                    "chat_sft requires a decoder-only GPT source checkpoint for "
                    f"model_backend=nanochat; got {type(native_model).__name__}"
                )
        orig_model: t.nn.Module = native_model
        model: t.nn.Module = cast(t.nn.Module, t.compile(native_model, dynamic=False))
        pretrain_batch_size = native_meta.get("device_batch_size")
        if pretrain_batch_size is not None and cfg.device_batch_size > pretrain_batch_size:
            logger.warning(
                "FOOTGUN: base model training used device_batch_size "
                f"{pretrain_batch_size}, current SFT uses {cfg.device_batch_size}"
            )
    else:
        hf = cfg.huggingface
        hf_attention = HFAttentionImplementation(str(hf.attention_implementation))
        hf_distributed = HFDistributedImplementation(str(hf.distributed_implementation))
        hf_local_world_size = validated_local_world_size(
            local_rank=ddp_local_rank,
            device=device,
        )
        logger.info(f"HF source pinned to model_id={str(hf.model_id)!r}, revision={hf.revision}")
        with (
            _optional_startup_phase(startup_trace, "source_config_and_tokenizer"),
            rank_synchronized_stage("HF config/tokenizer preflight", device=device),
        ):
            source_config, hf_tokenizer, hf_contract = load_hf_config_and_tokenizer(
                model_id=str(hf.model_id),
                revision=str(hf.revision),
            )
            validate_decoder_sft_config(cfg, source_config=source_config)
        with _optional_startup_phase(startup_trace, "source_model_load"):
            hf_raw_model = load_hf_model_in_local_rank_slots(
                model_id=str(hf.model_id),
                revision=str(hf.revision),
                contract=hf_contract,
                device=device,
                local_rank=ddp_local_rank,
                gradient_checkpointing=bool(hf.gradient_checkpointing),
                attention_implementation=hf_attention,
            )
        configure_generation_contract(hf_raw_model, hf_contract)
        tokenizer = hf_tokenizer
        orig_model = HFDecoderModelAdapter(hf_raw_model, max_seq_len=int(cfg.max_seq_len))
        model = orig_model

    num_flops_per_token = orig_model.estimate_flops()
    tokens_per_fwdbwd = int(cfg.device_batch_size) * int(cfg.max_seq_len)
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    if int(cfg.total_batch_size) % world_tokens_per_fwdbwd != 0:
        raise ValueError(
            f"total_batch_size={cfg.total_batch_size} must be divisible by "
            f"device_batch_size*max_seq_len*world_size={world_tokens_per_fwdbwd}"
        )
    grad_accum_steps = int(cfg.total_batch_size) // world_tokens_per_fwdbwd
    if grad_accum_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    logger.info(
        f"Tokens / micro-batch / rank: {cfg.device_batch_size} x {cfg.max_seq_len} "
        f"= {tokens_per_fwdbwd:,}"
    )
    logger.info(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    logger.info(
        f"Total batch size {cfg.total_batch_size:,} => gradient accumulation "
        f"steps: {grad_accum_steps}"
    )

    if backend is DecoderSFTBackend.NANOCHAT:
        token_bytes = get_token_bytes(device=device)
        native_gpt = cast(GPT, orig_model)
        optimizer = native_gpt.setup_optimizer(
            unembedding_lr=cfg.unembedding_lr,
            embedding_lr=cfg.embedding_lr,
            matrix_lr=cfg.matrix_lr,
            weight_decay=cfg.weight_decay,
        )
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] * cfg.init_lr_frac
            group["initial_lr"] = group["lr"]
    else:
        assert hf_tokenizer is not None
        token_bytes = _broadcast_hf_token_bytes(
            hf_tokenizer,
            master_process=master_process,
            device=device,
        )

    mixture = SFTMixture(cfg.mixture)
    mixture_params = select_mixture_overrides(
        mixture,
        {key: getattr(cfg, key) for key in MIXTURE_PARAM_KEYS},
    )
    with (
        _optional_startup_phase(startup_trace, "mixture_build"),
        rank_synchronized_stage("decoder SFT mixture build", device=device),
    ):
        train_dataset, val_dataset = build_task_mixtures(mixture, **mixture_params)
    if val_dataset is None:
        raise ValueError(
            f"mixture {mixture.value!r} defines no validation tasks; chat_sft requires a val split"
        )
    logger.info(f"[{cfg.mixture}] Training mixture: {len(train_dataset):,} rows")
    logger.info(f"[{cfg.mixture}] Validation mixture: {len(val_dataset):,} rows")
    train_fingerprint: Mapping[str, Any] | None = None
    val_fingerprint: Mapping[str, Any] | None = None
    if backend is DecoderSFTBackend.HUGGINGFACE:
        with (
            _optional_startup_phase(startup_trace, "mixture_identity"),
            rank_synchronized_stage("HF mixture identity", device=device),
        ):
            train_fingerprint = compute_mixture_fingerprint(
                train_dataset,
                mixture_name=mixture.value,
                split=SFTSplit.TRAIN,
            )
            val_fingerprint = compute_mixture_fingerprint(
                val_dataset,
                mixture_name=mixture.value,
                split=SFTSplit.VAL,
            )
        assert hf_tokenizer is not None and hf_contract is not None
        assert hf_raw_model is not None and hf_attention is not None
        assert hf_distributed is not None
        hf = cfg.huggingface
        if (
            ddp_world_size > 1
            and hf_distributed is HFDistributedImplementation.DIST_MUON_ADAMW
            and grad_clip_max_norm is not None
        ):
            raise ValueError(
                "distributed_implementation=dist_muon_adamw requires "
                "grad_clip_max_norm=null because local-before-reduction clipping "
                "is not a global gradient norm"
            )
        hf_output_root = resolve_hf_output_root(
            base_dir=get_base_dir() if hf.output_dir is None else ".",
            model_id=str(hf.model_id),
            revision=str(hf.revision),
            output_dir=hf.output_dir,
        )
        with (
            _optional_startup_phase(startup_trace, "restart_identity"),
            rank_synchronized_stage("HF restart identity", device=device),
        ):
            hf_restart_identity = build_hf_restart_identity(
                resolved_config=user_config,
                repository=repository_version(),
                source={
                    "model_id": str(hf.model_id),
                    "revision": str(hf.revision),
                    "architecture": hf_contract.architecture,
                    "model_type": hf_contract.model_type,
                },
                tokenizer_contract={
                    "chat_template_sha256": hf_contract.chat_template_sha256,
                    "model_vocab_size": hf_contract.model_vocab_size,
                    "tokenizer_vocab_size": hf_contract.tokenizer_vocab_size,
                    "structural_token_ids": dict(hf_contract.structural_token_ids),
                    "tied_embeddings": hf_contract.tied_embeddings,
                },
                train_mixture_fingerprint=train_fingerprint,
                val_mixture_fingerprint=val_fingerprint,
                world_size=ddp_world_size,
                local_world_size=hf_local_world_size,
            )
            assert_distributed_hf_restart_identity(
                hf_restart_identity,
                device=device,
            )
        restart_from_step = int(hf.restart_from_step)
        if restart_from_step >= 0:
            with _optional_startup_phase(startup_trace, "restart_model_load"):
                with rank_synchronized_stage("HF restart checkpoint validation", device=device):
                    hf_restart_descriptor = validate_hf_restart(
                        output_root=hf_output_root,
                        step=restart_from_step,
                        expected_identity=hf_restart_identity,
                        implementation=hf_distributed,
                        world_size=ddp_world_size,
                        local_world_size=hf_local_world_size,
                    )
                load_hf_restart_model_in_local_rank_slots(
                    hf_restart_descriptor,
                    model=hf_raw_model,
                    local_rank=ddp_local_rank,
                    local_world_size=hf_local_world_size,
                    device=device,
                )
        with _optional_startup_phase(startup_trace, "compile_and_ddp"):
            model = prepare_hf_training_model(
                cast(HFDecoderModelAdapter, orig_model),
                ddp=ddp,
                local_rank=ddp_local_rank,
                distributed_implementation=hf_distributed,
                torch_compile=bool(hf.torch_compile),
            )
        with _optional_startup_phase(startup_trace, "optimizer_setup"):
            optimizer = build_hf_optimizer(
                cast(HFDecoderModelAdapter, orig_model),
                ddp=ddp,
                learning_rate=float(hf.learning_rate),
                betas=(float(hf.adam_betas[0]), float(hf.adam_betas[1])),
                eps=float(hf.adam_eps),
                weight_decay=float(cfg.weight_decay),
                init_lr_frac=float(cfg.init_lr_frac),
                distributed_implementation=hf_distributed,
            )
        if hf_restart_descriptor is not None:
            with _optional_startup_phase(startup_trace, "restart_optimizer_load"):
                hf_restart_rank_payload = load_hf_restart_rank_payload_in_local_rank_slots(
                    hf_restart_descriptor,
                    rank=ddp_rank,
                    local_rank=ddp_local_rank,
                    local_world_size=hf_local_world_size,
                    device=device,
                )
                with rank_synchronized_stage("HF restart optimizer restoration", device=device):
                    optimizer.load_state_dict(
                        dict(cast(Mapping[str, Any], hf_restart_rank_payload["optimizer"]))
                    )
    per_rank_size = len(range(ddp_rank, len(train_dataset), ddp_world_size))
    with rank_synchronized_stage("decoder SFT dataset rank coverage", device=device):
        if per_rank_size <= 0:
            raise ValueError(
                f"training dataset size {len(train_dataset)} is smaller than world size "
                f"{ddp_world_size}; rank {ddp_rank} owns no conversations"
            )
        if int(cfg.eval_every) > 0 and len(val_dataset) <= ddp_rank:
            raise ValueError(
                f"validation dataset size {len(val_dataset)} is smaller than world size "
                f"{ddp_world_size}; rank {ddp_rank} owns no conversations"
            )

    def sft_data_generator_bos_bestfit(
        split: DatasetSplit,
        buffer_size: int = 100,
        *,
        resume_state: Mapping[str, Any] | None = None,
    ):
        """Delegate unchanged BOS best-fit packing to the shared core generator."""
        assert split in DatasetSplit, f"split must be one of {list(DatasetSplit)}"
        dataset = train_dataset if split == DatasetSplit.TRAIN else val_dataset
        return sft_data_generator_bos_bestfit_core(
            dataset=dataset,
            tokenizer=tokenizer,
            max_seq_len=int(cfg.max_seq_len),
            device_batch_size=int(cfg.device_batch_size),
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device=t.device(device),
            pin_memory=(device_type == "cuda"),
            buffer_size=buffer_size,
            loss_mask_mode=loss_mask_mode,
            resume_state=resume_state,
        )

    min_val_bpb = float("inf")
    val_bpb = float("nan")
    smooth_train_loss = 0.0
    ema_beta = 0.9
    total_training_time = 0.0
    completed_steps = 0
    committed_consumed = 0
    committed_dataloader_state: dict[str, Any] | None = None
    current_epoch = 1
    final_hf_checkpoint: Path | None = None
    if hf_restart_rank_payload is not None:
        assert hf_restart_descriptor is not None
        with rank_synchronized_stage("HF restart runtime validation", device=device):
            runtime = validate_hf_restart_runtime(
                hf_restart_rank_payload["runtime"],
                expected_step=hf_restart_descriptor.step,
            )
            completed_steps = runtime["completed_steps"]
            smooth_train_loss = runtime["smooth_train_loss"]
            latest_val = runtime["latest_val_bpb"]
            minimum_val = runtime["min_val_bpb"]
            val_bpb = float("nan") if latest_val is None else latest_val
            min_val_bpb = float("inf") if minimum_val is None else minimum_val
            total_training_time = runtime["total_training_time"]
            committed_dataloader_state = validate_decoder_sft_dataloader_state(
                runtime["dataloader_state"],
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                buffer_size=100,
            )
            committed_consumed = int(committed_dataloader_state["consumed"])
            current_epoch = int(committed_dataloader_state["next_cursor"]) // len(train_dataset) + 1

    # External logging begins only after source/model, optimizer coverage,
    # batch geometry, and mixture identity have all passed preflight, and
    # before the first training generator is constructed.
    use_dummy_wandb = cfg.run == "dummy" or not master_process
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat-sft", name=cfg.run, config=user_config)
    )
    configure_wandb_step_metric(wandb_run)
    # Restore at the last setup boundary so third-party logging initialization
    # cannot perturb the exact Torch RNG continuation captured by the restart.
    if hf_restart_rank_payload is not None:
        with rank_synchronized_stage("HF restart RNG restoration", device=device):
            restore_hf_restart_rng(hf_restart_rank_payload["runtime"], device=device)

    train_loader = sft_data_generator_bos_bestfit(
        DatasetSplit.TRAIN,
        resume_state=committed_dataloader_state,
    )

    def build_val_loader():
        for inputs, targets, _progress in sft_data_generator_bos_bestfit(DatasetSplit.VAL):
            yield inputs, targets

    while True:
        position = resolve_sft_loop_position(
            completed_steps=completed_steps,
            num_iterations=int(cfg.num_iterations),
            committed_consumed=committed_consumed,
            dataset_size=per_rank_size,
        )
        last_step, progress = sync_stop_and_progress(
            last_step=position.last_step,
            progress=position.progress,
            device=device,
        )
        flops_so_far = num_flops_per_token * int(cfg.total_batch_size) * completed_steps

        if int(cfg.eval_every) > 0 and (last_step or completed_steps % int(cfg.eval_every) == 0):
            evaluation_model = orig_model if backend is DecoderSFTBackend.HUGGINGFACE else model
            evaluation_model.eval()
            eval_steps = int(cfg.eval_tokens) // (
                int(cfg.device_batch_size) * int(cfg.max_seq_len) * ddp_world_size
            )
            if eval_steps <= 0:
                raise ValueError(
                    f"eval_tokens={cfg.eval_tokens} produces zero evaluation steps for "
                    "the distributed batch geometry"
                )
            with autocast_ctx:
                val_bpb = evaluate_bpb(
                    evaluation_model, build_val_loader(), eval_steps, token_bytes
                )
            logger.info(f"Step {completed_steps:05d} | Validation bpb: {val_bpb:.4f}")
            min_val_bpb = min(min_val_bpb, val_bpb)
            wandb_run.log(
                {
                    "step": completed_steps,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "val/bpb": val_bpb,
                }
            )
            evaluation_model.train()

        save_every = int(cfg.save_every)
        interval_save_due = should_save_checkpoint(
            is_master=master_process,
            is_dry_run=bool(cfg.dry_run),
            is_last_step=False,
            step=completed_steps,
            save_every=save_every,
        )
        final_save_due = last_step and not bool(cfg.dry_run)
        hf_save_due = False
        hf_restart_due = False
        if backend is DecoderSFTBackend.HUGGINGFACE:
            hf_save_due, hf_restart_due = _hf_persistence_due(
                completed_steps=completed_steps,
                last_step=last_step,
                dry_run=bool(cfg.dry_run),
                portable_save_every=save_every,
                restart_save_every=int(cfg.huggingface.restart_save_every),
                loaded_restart_step=(
                    None if hf_restart_descriptor is None else hf_restart_descriptor.step
                ),
            )
            if (
                hf_restart_descriptor is not None
                and completed_steps == hf_restart_descriptor.step
                and last_step
                and not bool(cfg.dry_run)
            ):
                assert hf_output_root is not None
                restored_portable = hf_output_root / f"step_{completed_steps:06d}"
                if _distributed_directory_exists(restored_portable, device=device):
                    final_hf_checkpoint = restored_portable
                else:
                    # A restart can commit before a same-boundary portable
                    # publication fails. Exact-final resume backfills only that
                    # missing portable artifact and never rewrites the restart.
                    hf_save_due = True

        # Publish the resumable artifact first. If a same-boundary portable
        # export then fails, the completed restart can restore/backfill it.
        if backend is DecoderSFTBackend.HUGGINGFACE:
            assert hf_raw_model is not None and hf_distributed is not None
            assert hf_restart_identity is not None and hf_output_root is not None
            if hf_restart_due:
                if committed_dataloader_state is None:
                    raise RuntimeError(
                        "cannot save an HF restart without committed dataloader state"
                    )
                save_optimizer_state = (
                    hf_distributed is HFDistributedImplementation.DIST_MUON_ADAMW or master_process
                )
                with rank_synchronized_stage("HF restart state capture", device=device):
                    runtime_state = capture_hf_restart_runtime(
                        dataloader_state=committed_dataloader_state,
                        completed_steps=completed_steps,
                        smooth_train_loss=smooth_train_loss,
                        latest_val_bpb=val_bpb if math.isfinite(val_bpb) else None,
                        min_val_bpb=min_val_bpb if math.isfinite(min_val_bpb) else None,
                        total_training_time=total_training_time,
                        device=device,
                    )
                    model_state = hf_raw_model.state_dict()
                    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
                save_hf_restart_distributed(
                    output_root=hf_output_root,
                    completed_step=completed_steps,
                    model_state=model_state,
                    optimizer_state=optimizer_state,
                    runtime_state=runtime_state,
                    identity=hf_restart_identity,
                    implementation=hf_distributed,
                    world_size=ddp_world_size,
                    local_world_size=hf_local_world_size,
                    rank=ddp_rank,
                    master_process=master_process,
                    device=device,
                )

        if backend is DecoderSFTBackend.NANOCHAT and (interval_save_due or final_save_due):
            # Native intermediate checkpoints are NOT resume-able and omit
            # optimizer state; the final boundary retains the historical
            # all-rank sharded optimizer checkpoint behavior.
            assert native_meta is not None
            native_gpt = cast(GPT, orig_model)
            checkpoint_dir = os.path.join(
                native_meta["checkpoint_dir"],
                f"{cfg.mixture}_sft_checkpoints",
            )
            meta_dict = {
                "step": completed_steps,
                "val_bpb": val_bpb,
                "sft_loss_mask": loss_mask_mode.value,
                "model_config": {
                    "sequence_len": cfg.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": native_gpt.config.n_layer,
                    "n_head": native_gpt.config.n_head,
                    "n_kv_head": native_gpt.config.n_kv_head,
                    "n_embd": native_gpt.config.n_embd,
                    "window_pattern": native_gpt.config.window_pattern,
                },
                "user_config": user_config,
            }
            if final_save_due:
                save_checkpoint(
                    checkpoint_dir,
                    completed_steps,
                    orig_model.state_dict(),
                    optimizer.state_dict(),
                    meta_dict,
                    rank=ddp_rank,
                )
            else:
                save_checkpoint(
                    checkpoint_dir,
                    completed_steps,
                    orig_model.state_dict(),
                    None,
                    meta_dict,
                )
        elif backend is DecoderSFTBackend.HUGGINGFACE and hf_save_due:
            assert hf_tokenizer is not None and hf_raw_model is not None
            assert hf_contract is not None
            assert train_fingerprint is not None and val_fingerprint is not None
            assert hf_output_root is not None and hf_attention is not None
            assert hf_distributed is not None
            metadata = _hf_metadata(
                cfg=cfg,
                user_config=user_config,
                completed_step=completed_steps,
                val_bpb=val_bpb,
                min_val_bpb=min_val_bpb,
                ddp_world_size=ddp_world_size,
                grad_accum_steps=grad_accum_steps,
                grad_clip_max_norm=grad_clip_max_norm,
                train_fingerprint=train_fingerprint,
                val_fingerprint=val_fingerprint,
                raw_model=hf_raw_model,
                contract=hf_contract,
                attention_implementation=hf_attention,
                distributed_implementation=hf_distributed,
            )
            final_hf_checkpoint = publish_hf_checkpoint_distributed(
                output_root=hf_output_root,
                completed_step=completed_steps,
                model=hf_raw_model,
                tokenizer=hf_tokenizer,
                metadata=metadata,
                master_process=master_process,
                device=device,
            )

        if last_step:
            break

        synchronize()
        t0 = time.perf_counter()
        produced_progress: SFTGeneratorProgress | None = None
        step_train_loss: float | None = None
        global_nonpadding_tokens = int(cfg.total_batch_size)
        global_supervised_tokens = int(cfg.total_batch_size)
        next_completed_step = completed_steps + 1
        recorder_context = (
            use_recorder(timing_recorder) if timing_recorder is not None else nullcontext()
        )
        tags_context = (
            timing_tags(rank=ddp_rank, step=next_completed_step)
            if timing_recorder is not None
            else nullcontext()
        )
        profile_context = (
            _maybe_profile_hf_step(
                profile_settings,
                completed_step=next_completed_step,
                rank=ddp_rank,
                device=device,
            )
            if backend is DecoderSFTBackend.HUGGINGFACE
            else nullcontext()
        )
        with (
            profile_context,
            recorder_context,
            tags_context,
            timed_region("chat_sft/whole_step", sync_cuda=False),
        ):
            if backend is DecoderSFTBackend.HUGGINGFACE:
                micro_batches: list[tuple[t.Tensor, t.Tensor, SFTGeneratorProgress]] = []
                first_batch_context = (
                    _optional_startup_phase(startup_trace, "first_batch")
                    if startup_trace is not None and startup_trace.first_update_step is None
                    else nullcontext()
                )
                with (
                    first_batch_context,
                    timed_region(
                        "chat_sft/batch_construction",
                        sync_cuda=False,
                    ),
                    rank_synchronized_stage(
                        "decoder SFT optimizer-step batch construction",
                        device=device,
                    ),
                ):
                    for _micro_step in range(grad_accum_steps):
                        micro_batches.append(next(train_loader))
                local_nonpadding = sum(
                    batch_progress.nonpadding_target_count
                    for _inputs, _targets, batch_progress in micro_batches
                )
                local_supervised = sum(
                    batch_progress.supervised_target_count
                    for _inputs, _targets, batch_progress in micro_batches
                )
                target_counts = reduce_global_target_counts(
                    local_nonpadding=local_nonpadding,
                    local_supervised=local_supervised,
                    device=device,
                )
                global_nonpadding_tokens = target_counts.nonpadding
                global_supervised_tokens = target_counts.supervised
                loss_scale = distributed_loss_backward_scale(
                    target_counts.supervised,
                    world_size=ddp_world_size,
                )
                local_loss_sum = t.zeros((), dtype=t.float32, device=device)
                with timed_region(
                    "chat_sft/forward_backward",
                    sync_cuda=False,
                ):
                    for micro_step, (
                        inputs,
                        targets,
                        batch_progress,
                    ) in enumerate(micro_batches):
                        produced_progress = batch_progress
                        is_last_micro = micro_step == len(micro_batches) - 1
                        no_sync = getattr(model, "no_sync", None)
                        reduction_context = (
                            no_sync() if not is_last_micro and callable(no_sync) else nullcontext()
                        )
                        with reduction_context:
                            with autocast_ctx:
                                loss_sum = model(
                                    inputs,
                                    targets,
                                    loss_reduction="sum",
                                )
                            local_loss_sum.add_(loss_sum.detach())
                            (loss_sum * loss_scale).backward()
                step_train_loss = reduce_global_loss_mean(
                    local_loss_sum,
                    global_supervised=target_counts.supervised,
                )
            else:
                train_loss: t.Tensor | None = None
                with timed_region(
                    "chat_sft/forward_backward",
                    sync_cuda=False,
                ):
                    for _micro_step in range(grad_accum_steps):
                        with rank_synchronized_stage(
                            "decoder SFT batch construction", device=device
                        ):
                            inputs, targets, produced_progress = next(train_loader)
                        with autocast_ctx:
                            loss = model(inputs, targets)
                        train_loss = loss.detach()
                        (loss / grad_accum_steps).backward()
                assert train_loss is not None
                step_train_loss = float(train_loss.item())

            lrm = sft_flat_decay(progress)
            momentum_progress = min(completed_steps / 300, 1)
            muon_momentum = (1 - momentum_progress) * 0.85 + momentum_progress * 0.95
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
            pre_clip_grad_norm: t.Tensor | None = None
            if grad_clip_max_norm is not None:
                with (
                    timed_region("chat_sft/gradient_clip", sync_cuda=False),
                    rank_synchronized_stage("decoder SFT gradient clipping", device=device),
                ):
                    pre_clip_grad_norm = apply_gradient_clipping(
                        orig_model.parameters(),
                        grad_clip_max_norm,
                    )
            with timed_region("chat_sft/optimizer_step", sync_cuda=False):
                optimizer.step()
            orig_model.zero_grad(set_to_none=True)
        synchronize()
        dt = time.perf_counter() - t0

        assert produced_progress is not None and step_train_loss is not None
        completed_steps += 1
        committed_consumed = produced_progress.consumed
        committed_dataloader_state = dict(produced_progress.dataloader_state)
        current_epoch = produced_progress.current_epoch
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * step_train_loss
        debias_exponent = (
            completed_steps if backend is DecoderSFTBackend.HUGGINGFACE else completed_steps + 1
        )
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**debias_exponent)
        tok_per_sec = int(cfg.total_batch_size / dt)
        nonpadding_tok_per_sec = int(global_nonpadding_tokens / dt)
        supervised_tok_per_sec = int(global_supervised_tokens / dt)
        flops_per_sec = num_flops_per_token * int(cfg.total_batch_size) / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        if timing_recorder is not None:
            _record_hf_paid_step(
                timing_recorder,
                rank=ddp_rank,
                completed_step=completed_steps,
                step_s=dt,
                scheduled_tokens=int(cfg.total_batch_size),
                nonpadding_tokens=global_nonpadding_tokens,
                supervised_tokens=global_supervised_tokens,
                loss=step_train_loss,
                algorithmic_mfu_percent=mfu,
                peak_allocated_bytes=int(get_max_memory()),
            )
        if completed_steps > 10:
            total_training_time += dt
        if startup_trace is not None:
            startup_trace.observe_step(
                update_step=completed_steps,
                step_s=dt,
                compile_configured=bool(cfg.huggingface.torch_compile),
            )
        if backend is DecoderSFTBackend.HUGGINGFACE:
            logger.info(
                f"step {completed_steps:05d} ({100 * progress:.2f}%) | "
                f"loss: {step_train_loss:.6f} | "
                f"smooth loss: {debiased_smooth_loss:.6f} | "
                f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | "
                f"scheduled tok/sec: {tok_per_sec:,} | "
                f"nonpadding tok/sec: {nonpadding_tok_per_sec:,} | "
                f"supervised tok/sec: {supervised_tok_per_sec:,} | "
                f"mfu: {mfu:.2f} | epoch: {current_epoch} | "
                f"total time: {total_training_time / 60:.2f}m"
            )
        else:
            # Preserve native decoder SFT's existing log schema and smoothed
            # loss meaning; the new exact-token metrics belong only to HF.
            logger.info(
                f"step {completed_steps:05d} ({100 * progress:.2f}%) | "
                f"loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
                f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
                f"mfu: {mfu:.2f} | epoch: {current_epoch} | "
                f"total time: {total_training_time / 60:.2f}m"
            )
        if completed_steps % 10 == 0:
            if backend is DecoderSFTBackend.HUGGINGFACE:
                completed_flops = num_flops_per_token * int(cfg.total_batch_size) * completed_steps
                train_metrics: dict[str, float | int] = {
                    "step": completed_steps,
                    "total_training_flops": completed_flops,
                    "total_training_time": total_training_time,
                    "train/loss": step_train_loss,
                    "train/smoothed_loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/scheduled_tokens": int(cfg.total_batch_size),
                    "train/nonpadding_tokens": global_nonpadding_tokens,
                    "train/supervised_tokens": global_supervised_tokens,
                    "train/scheduled_tok_per_sec": tok_per_sec,
                    "train/nonpadding_tok_per_sec": nonpadding_tok_per_sec,
                    "train/supervised_tok_per_sec": supervised_tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": current_epoch,
                }
            else:
                train_metrics = {
                    "step": completed_steps,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": current_epoch,
                }
            if pre_clip_grad_norm is not None:
                train_metrics["train/pre_clip_grad_norm"] = float(pre_clip_grad_norm.item())
            wandb_run.log(train_metrics)

    logger.info(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    logger.info(f"Total training time: {total_training_time / 60:.2f}m")
    if math.isfinite(min_val_bpb):
        logger.info(f"Minimum validation bpb: {min_val_bpb:.4f}")
    else:
        logger.info("Minimum validation bpb: unavailable (validation disabled)")
    profiling_publication_context = (
        rank_synchronized_stage("HF profiling evidence publication", device=device)
        if backend is DecoderSFTBackend.HUGGINGFACE
        else nullcontext()
    )
    with profiling_publication_context:
        if timing_recorder is not None:
            timing_recorder.to_csv(
                profile_settings.output_dir / f"chat_sft_timing_rank{ddp_rank}.csv"
            )
        if startup_trace is not None:
            startup_trace.finish(
                status=(
                    "complete"
                    if startup_trace.first_update_step is not None
                    else "no_training_step"
                )
            )

    if backend is DecoderSFTBackend.NANOCHAT and not cfg.dry_run:
        assert native_meta is not None
        sft_checkpoint_dir = os.path.join(
            native_meta["checkpoint_dir"],
            f"{cfg.mixture}_sft_checkpoints",
        )
        get_report(checkpoint_dir=sft_checkpoint_dir).log(
            section="SFT",
            data=[
                user_config,
                {
                    "Number of iterations": completed_steps,
                    "DDP world size": ddp_world_size,
                },
                {"Minimum validation bpb": min_val_bpb},
            ],
        )

    wandb_run.finish()

    if (
        backend is DecoderSFTBackend.HUGGINGFACE
        and not bool(cfg.dry_run)
        and bool(cfg.huggingface.validate_export)
    ):
        if final_hf_checkpoint is None:
            raise RuntimeError("final HF checkpoint path was not recorded")
        assert hf_contract is not None
        del optimizer, model, orig_model, hf_raw_model
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
        with rank_synchronized_stage(
            "final HF export smoke validation",
            device=device,
        ):
            if master_process:
                validate_export_smoke(
                    final_hf_checkpoint,
                    contract=hf_contract,
                    device=device,
                )

    compute_cleanup()


if __name__ == "__main__":
    main()
