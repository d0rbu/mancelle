"""Verified-reward RL of generated feature conditioning after turn-delimited SFT."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nanochat import flash_attention as native_flash_attention
from nanochat.atomic_io import atomic_json_dump
from nanochat.checkpoint_manager import find_last_step, load_checkpoint, save_checkpoint
from nanochat.common import COMPUTE_DTYPE, compute_cleanup
from nanochat.hydra_cli import parse_config_name_and_overrides
from nanochat.optim import DistMuonAdamW
from nanochat.tokenizer import get_tokenizer
from scripts.train.conditional_encdec_sft_grpo import (
    CHECKPOINT_PROCESS_MODE,
    DETERMINISTIC_KERNEL_MODE,
    _assert_attention_impl,
    _build_model,
    _build_optimizer,
    _configure_deterministic_algorithms,
    _reexec_for_deterministic_kernels,
    _reexec_from_checkpoint,
    _reset_step_compiler_state,
    _sha256_file,
)
from steering._common.ddp import compute_init_for_checkpoint_resume, rank_synchronized_stage
from steering._common.file_inventory import file_signature
from steering._common.lr_schedules import flat_then_linear_decay
from steering.conditional.encoder_decoder.config import (
    reject_removed_feature_prediction_keys,
    serialize_feature_prediction_config,
    validate_feature_prediction_resume,
)
from steering.conditional.encoder_decoder.dataloader import (
    build_dataloader_state_envelope,
    select_dataloader_rank_state,
)
from steering.conditional.encoder_decoder.frozen_decoder import (
    StagingReport,
    frozen_decoder_staging_settings,
    stage_frozen_decoder_linears,
)
from steering.conditional.encoder_decoder.rlvr_data import grade_completion_result, load_rlvr_bundle
from steering.conditional.encoder_decoder.rlvr_policy import (
    backward_rlvr_group,
    collect_rlvr_group,
    rlvr_group_replay_cache,
)
from steering.conditional.encoder_decoder.rlvr_replay_cache import (
    HiddenReplayReport,
    HiddenReplaySettings,
    RLVRReplaySession,
    hidden_replay_report_snapshot,
    hidden_replay_settings,
)
from steering.conditional.encoder_decoder.rlvr_runtime import (
    RLVR_GENERATION_VERSION,
    RLVR_META_KEY,
    RLVR_OBJECTIVE_VERSION,
    RLVR_TASKS,
    RLVRWandb,
    assert_complete_checkpoint,
    capture_rank_rng,
    checkpoint_files,
    clip_and_step,
    configure_rlvr_compiler,
    resolve_rlvr_tasks,
    restore_rank_rng,
    rlvr_seed,
    select_training_row,
    validate_rlvr_settings,
    validation_panel,
)
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_checkpoint import (
    validate_grpo_checkpoint_eligibility,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    _count_selection_identity,
    assert_grpo_resume_identity,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    configure_grpo_trainable_parameters,
    resolve_train_feature_head_only,
)
from steering.eval.chat_eval_result_identity import (
    ChatCacheSnapshot,
    TurnChatEvalCheckpointSnapshot,
    self_conditioned_code_identity,
)
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.self_conditioning_policy import build_pure_model_self_conditioning_config
from steering.eval.target_conditioning import ContextRoutingMode
from steering.eval.turn_context_contract import (
    canonical_turn_payload,
    resolve_turn_context_contract,
    turn_extraction_tokenizer_id,
)
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
)


@dataclass
class RLVRLoaded:
    """One model and immutable inputs; each rollout creates a fresh KV runtime."""

    model: Any
    tokenizer: Any
    source_tokenizer: Any
    base_meta: dict
    base_identity: dict
    bundle: Any
    turn_contract: Any
    provider: Any
    cache_snapshot: ChatCacheSnapshot
    policy: Any
    generation_policy: Any
    cadence: int
    compiler_config: dict[str, int | bool]

    def engine(self, _rollout_index: int):
        return build_engine_self_conditioned(
            self.model, self.tokenizer, self.model.get_device(), self.generation_policy,
            context_provider=self.provider, turn_contract=self.turn_contract,
            context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE,
        )


def _prepare_rlvr_runtime(device: str | t.device) -> None:
    """Apply the observed SM90 token-generation repeatability mitigation."""
    device = t.device(device)
    if (
        device.type == "cuda"
        and t.cuda.get_device_capability(device) == (9, 0)
        and native_flash_attention._active_impl()
        == native_flash_attention.FlashAttnImpl.FA4
        and not t.are_deterministic_algorithms_enabled()
        and _resolved_token_kv_backend() == "sdpa"
    ):
        t.backends.cuda.enable_cudnn_sdp(False)


def load_rlvr_inputs(cfg, *, device, rank: int) -> RLVRLoaded:
    """Admit the original checkpoint and prepared question cache before training."""
    _prepare_rlvr_runtime(device)
    base_dir = Path(cfg.base_checkpoint_dir)
    step = find_last_step(str(base_dir)) if cfg.model_step is None else int(cfg.model_step)
    meta_path = base_dir / f"meta_{step:06d}.json"
    meta_payload = meta_path.read_bytes()
    original_meta = json.loads(meta_payload)
    snapshot = TurnChatEvalCheckpointSnapshot.capture(meta_path, meta_payload, step)
    if "step" in original_meta and (type(original_meta["step"]) is not int or original_meta["step"] != step):
        raise SFTGRPOError("base checkpoint metadata step disagrees with filename")
    mode = resolve_checkpoint_sft_segment_mode(original_meta, checkpoint_dir=str(base_dir))
    if mode is not SegmentMode.turn_delimited_fixed_tokens:
        raise SFTGRPOError("RLVR v1 requires a turn-delimited SFT checkpoint")
    cadence = resolve_turn_self_cond_segment_size_rustbpe(original_meta, checkpoint_dir=str(base_dir))
    if cadence is None:
        raise SFTGRPOError("turn RLVR requires checkpoint-recorded generated-assistant cadence")
    if resolve_next_segment_conditioning_offset({"next_segment_conditioning_offset": "auto"}, original_meta):
        raise SFTGRPOError("RLVR requires current-segment prompt conditioning")
    base_identity = {"step": step, "model_sha256": _sha256_file(base_dir / f"model_{step:06d}.pt"),
                     "metadata_sha256": hashlib.sha256(meta_payload).hexdigest()}
    state, _, meta = load_checkpoint(str(base_dir), step, device, load_optimizer=False, rank=rank)
    snapshot.assert_stable()
    if meta != original_meta:
        raise SFTGRPOError("base checkpoint metadata changed during admission")
    model = _build_model(state, meta, device=device)
    del state
    eligibility = validate_grpo_checkpoint_eligibility(model, meta, checkpoint_dir=str(base_dir),
        action_mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT, seed=int(cfg.seed), action_policy_version=1)
    tokenizer = get_tokenizer()
    lineage = OnlineSAELineage.from_checkpoint_meta(meta)
    source_tokenizer = AutoTokenizer.from_pretrained(turn_extraction_tokenizer_id(lineage.model))
    contract = resolve_turn_context_contract(meta_path, tokenizer=tokenizer,
        extraction_tokenizer=source_tokenizer, statistics_path=cfg.stats_path,
        checkpoint_contract=eligibility.feature_contract)
    bundle = load_rlvr_bundle(Path(cfg.bundle_dir), tokenizer=tokenizer,
        expected_base_model_sha256=base_identity["model_sha256"], expected_turn_context_contract=contract)
    cache_snapshot = ChatCacheSnapshot.capture(bundle.cache_dir)
    provider = bundle.open_prompt_cache(expected_turn_context_contract=contract)
    cache_snapshot.bind_loaded(provider)
    for row in (*bundle.train, *bundle.validation):
        budget = int(cfg.gsm8k_max_tokens if row.task == "GSM8K" else cfg.mc_max_tokens)
        if budget <= 0 or len(row.prompt_token_ids) + budget > model.dec_config.sequence_len:
            raise SFTGRPOError(f"{row.row_id}: prompt plus requested answer budget exceeds the model context")
    generation_policy = build_pure_model_self_conditioning_config(
        model=model, checkpoint_meta=meta, n_sae_features=model.encoder.config.n_sae_features,
        feature_mask=eligibility.feature_contract.feature_mask, consolidation=eligibility.feature_contract.consolidation,
        segment_mode=mode, segment_size=cadence, selection_seed=int(cfg.seed),
        percentile_table=t.empty((0, 0), dtype=t.float64))
    compiler_config = configure_rlvr_compiler(model)
    snapshot.assert_stable()
    return RLVRLoaded(model, tokenizer, source_tokenizer, meta, base_identity, bundle, contract,
                      provider, cache_snapshot, eligibility.policy, generation_policy, cadence,
                      compiler_config)


def _group(loaded: RLVRLoaded, row, *, cfg, group_size: int, seed: int,
           replay_cache: RLVRReplaySession | None = None):
    return collect_rlvr_group(
        loaded.model, engine_factory=loaded.engine, prompt_token_ids=row.prompt_token_ids,
        policy_config=loaded.policy, group_size=group_size, group_seed=seed,
        reward_fn=lambda generation: grade_completion_result(
            row, loaded.tokenizer.decode(list(generation.token_ids[len(row.prompt_token_ids):]))).reward,
        max_tokens=int(cfg.gsm8k_max_tokens if row.task == "GSM8K" else cfg.mc_max_tokens),
        temperature=float(cfg.temperature), top_k=cfg.top_k,
        extraction_tokenizer=loaded.source_tokenizer,
        replay_cache=replay_cache,
    )


def rows_by_task(rows) -> dict[str, tuple]:
    return {task: tuple(row for row in rows if row.task == task) for task in RLVR_TASKS}


def evaluate_rlvr(loaded: RLVRLoaded, *, cfg, rank: int, world_size: int) -> tuple[dict[str, float], list]:
    """Partition one fixed heldout panel across ranks without row duplication."""
    panel = validation_panel(rows_by_task(loaded.bundle.validation), seed=int(cfg.validation_seed),
                             per_task=cfg.validation_per_task)
    device = loaded.model.get_device()
    totals = t.zeros((len(RLVR_TASKS), 3), dtype=t.float64, device=device)
    samples = []
    sample_ids = {row.row_id for task in RLVR_TASKS for row in [r for r in panel if r.task == task][:2]}
    with rank_synchronized_stage("RLVR validation rollouts", device=device):
        for index in range(rank, len(panel), world_size):
            row = panel[index]
            group = _group(loaded, row, cfg=cfg, group_size=1,
                seed=rlvr_seed(int(cfg.validation_seed), "validation", row.task, row.row_id))
            trajectory = group.trajectories[0]
            completion = loaded.tokenizer.decode(list(trajectory.token_ids[len(row.prompt_token_ids):]))
            grade = grade_completion_result(row, completion)
            task_index = RLVR_TASKS.index(row.task)
            totals[task_index] += t.tensor([trajectory.reward, 1, float(grade.valid_format)], device=device)
            if row.row_id in sample_ids:
                samples.append([row.task, row.row_id, loaded.tokenizer.decode(list(row.prompt_token_ids)),
                                completion, row.oracle["answer"], grade.parsed_answer, trajectory.reward, trajectory.stop_reason])
    if t.distributed.is_initialized():
        t.distributed.all_reduce(totals)
        gathered: list[Any] = [None] * world_size
        t.distributed.all_gather_object(gathered, samples)
        samples = [row for rows in gathered for row in rows]
    result = {}
    for task, values in zip(RLVR_TASKS, totals.cpu().tolist(), strict=True):
        reward, count, parsed = values
        result[f"validation/{task}/accuracy"] = reward / count
        result[f"validation/{task}/parse_rate"] = parsed / count
        result[f"validation/{task}/count"] = count
    result["validation/accuracy"] = sum(result[f"validation/{task}/accuracy"] for task in RLVR_TASKS) / len(RLVR_TASKS)
    return result, sorted(samples, key=lambda row: (row[0], row[1]))


def _resolved_token_kv_backend() -> str:
    backend = (
        native_flash_attention.FlashAttnImpl.FA3
        if native_flash_attention.HAS_FA3
        and native_flash_attention._override_impl != native_flash_attention.FlashAttnImpl.SDPA
        else native_flash_attention.FlashAttnImpl.SDPA
    )
    return backend.value


def _token_kv_sdpa_identity(device: str | t.device) -> dict[str, bool] | None:
    if t.device(device).type != "cuda" or _resolved_token_kv_backend() != "sdpa":
        return None
    return {
        "cudnn_enabled": t.backends.cuda.cudnn_sdp_enabled(),
        "flash_enabled": t.backends.cuda.flash_sdp_enabled(),
        "math_enabled": t.backends.cuda.math_sdp_enabled(),
        "memory_efficient_enabled": t.backends.cuda.mem_efficient_sdp_enabled(),
        "fp16_bf16_math_reduction_allowed": (
            t.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        ),
    }


def resume_identity(loaded: RLVRLoaded, cfg: Mapping[str, Any], *, world_size: int,
                    parameter_manifest: tuple[str, ...], attention_impl: str, deterministic: bool) -> dict:
    # The manifest already binds scope; omit this new key for legacy false/absent parity.
    operational = {"base_checkpoint_dir", "model_step", "bundle_dir", "stats_path", "output_dir", "resume_from_step", "resume_checkpoint_dir", "wandb", "train_feature_head_only"}
    body = {"version": 1, "base_checkpoint": loaded.base_identity, "bundle": loaded.bundle.identity_sha256,
            "turn_contract": loaded.turn_contract.semantic, "cadence": loaded.cadence,
            "config": {**{key: value for key, value in cfg.items() if key not in operational},
                       "tasks": resolve_rlvr_tasks(cfg.get("tasks", RLVR_TASKS))},
            "world_size": world_size, "parameter_manifest": parameter_manifest,
            "feature_action": feature_action_identity(loaded.policy),
            "objective": RLVR_OBJECTIVE_VERSION,
            "generation": RLVR_GENERATION_VERSION, "local_clip_before_optimizer_average": True,
            "runtime": {"code": self_conditioned_code_identity(), "torch": t.__version__, "cuda": t.version.cuda,
                        "compute_dtype": str(COMPUTE_DTYPE),
                        "autocast_enabled": t.is_autocast_enabled(loaded.model.get_device().type),
                        "device": str(loaded.model.get_device().type),
                        "device_name": t.cuda.get_device_name() if loaded.model.get_device().type == "cuda" else "cpu",
                        "attention": attention_impl,
                        "token_kv_backend": _resolved_token_kv_backend(),
                        "token_kv_sdpa": _token_kv_sdpa_identity(loaded.model.get_device()),
                        "deterministic": deterministic,
                        "compiler": loaded.compiler_config,
                        "kernel_mode": DETERMINISTIC_KERNEL_MODE, "process_mode": CHECKPOINT_PROCESS_MODE}}
    # Normalize tuples/enums before comparison with checkpoint JSON.
    body = json.loads(canonical_turn_payload(body))
    return {**body, "sha256": hashlib.sha256(canonical_turn_payload(body)).hexdigest()}


def feature_action_identity(policy) -> dict:
    return {"mode": policy.action_mode.value,
            "positive_counts": _count_selection_identity(policy.positive_counts),
            "negative_counts": _count_selection_identity(policy.negative_counts)}


def _close_frozen_decoder_worker(
    stack: ExitStack,
    report: StagingReport | None,
    *,
    setup_seconds: float,
    first_step: int,
    completed: int,
) -> None:
    """Release process-local shadows before an exec skips Python finally blocks."""
    already_closed = report is not None and report.cleaned_up
    started = time.perf_counter()
    stack.close()
    cleanup_seconds = time.perf_counter() - started
    if report is None or already_closed:
        return
    if not report.cleaned_up:
        raise RuntimeError("Frozen decoder staging did not clean up")
    updates = max(0, completed - first_step)
    logger.info("RLVR frozen-decoder worker complete: {}", {
        **asdict(report), "worker_start_step": first_step,
        "worker_optimizer_updates": updates, "setup_seconds": setup_seconds,
        "cleanup_seconds": cleanup_seconds,
        "setup_seconds_per_update": setup_seconds / updates if updates else None,
    })


def _log_hidden_replay_worker(
    report: HiddenReplayReport,
    settings: HiddenReplaySettings | None,
    *,
    first_step: int,
    completed: int,
    completed_successfully: bool = True,
) -> None:
    """Report tensor-free totals after every group session has closed."""
    if settings is None or settings.max_cache_bytes == 0:
        return
    snapshot = hidden_replay_report_snapshot(report)
    if completed_successfully and (not snapshot["cleaned_up"] or snapshot["active_groups"] != 0
            or snapshot["current_bytes"] != 0
            or snapshot["max_owned_tensor_objects_alive_after_cleanup"] != 0):
        raise RuntimeError("RLVR hidden replay retained group state before worker exit")
    logger.info("RLVR hidden-replay worker complete: {}", {
        **snapshot, "requested": asdict(settings), "worker_start_step": first_step,
        "worker_optimizer_updates": max(0, completed - first_step),
        "worker_completed_without_error": completed_successfully,
    })


def train(cfg) -> None:
    _, rank, _, world_size, device = compute_init_for_checkpoint_resume(
        str(cfg.device_type),
        resume_step=int(cfg.resume_from_step),
    )
    device = t.device(device)
    telemetry = None
    exit_code = 1
    frozen_stack = ExitStack()
    frozen_report: StagingReport | None = None
    frozen_setup_seconds = 0.0
    replay_settings: HiddenReplaySettings | None = None
    replay_report = HiddenReplayReport()
    replay_worker_reported = False
    worker_start_step = completed = 0
    try:
        raw_config = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(raw_config, dict)
        user_config = cast(dict[str, Any], raw_config)
        with rank_synchronized_stage("RLVR settings", device=device):
            reject_removed_feature_prediction_keys(user_config)
            train_feature_head_only = resolve_train_feature_head_only(user_config)
            validate_rlvr_settings(user_config, world_size=world_size)
            replay_settings = hidden_replay_settings(user_config)
            tasks = tuple(user_config.setdefault("tasks", list(RLVR_TASKS)))
            attention = _assert_attention_impl(device_type=device.type)
            deterministic = _configure_deterministic_algorithms(device_type=device.type, attention_impl=attention)
        with rank_synchronized_stage("RLVR model and artifact admission", device=device):
            loaded = load_rlvr_inputs(cfg, device=device, rank=rank)
            manifest = configure_grpo_trainable_parameters(
                loaded.model, train_lm_head=False,
                train_feature_head_only=train_feature_head_only,
            )
            optimizer = _build_optimizer(loaded.model, cfg, fp_head_lr=None)
            if not isinstance(optimizer, DistMuonAdamW):
                raise SFTGRPOError("RLVR training requires torchrun, including one-rank runs")
            identity = resume_identity(loaded, user_config, world_size=world_size,
                                       parameter_manifest=manifest, attention_impl=attention, deterministic=deterministic)
            output = Path(cfg.output_dir)
            completed = 0
            resumed_wandb = None
            resume_rng = None
            if cfg.resume_from_step != -1:
                resume_dir = output if cfg.resume_checkpoint_dir is None else Path(cfg.resume_checkpoint_dir)
                assert_complete_checkpoint(resume_dir, cfg.resume_from_step, world_size)
                resume_step = int(cfg.resume_from_step)
                meta_path = resume_dir / f"meta_{resume_step:06d}.json"
                meta_payload = meta_path.read_bytes()
                original_meta = json.loads(meta_payload)
                snapshot = TurnChatEvalCheckpointSnapshot.capture(
                    meta_path, meta_payload, resume_step
                )
                optimizer_path = resume_dir / f"optim_{resume_step:06d}_rank{rank}.pt"
                optimizer_signature = file_signature(optimizer_path)
                state, optim_state, meta = load_checkpoint(str(resume_dir), cfg.resume_from_step, device, load_optimizer=True, rank=rank)
                snapshot.assert_stable()
                if file_signature(optimizer_path) != optimizer_signature:
                    raise ValueError("RLVR checkpoint optimizer changed during admission")
                if meta != original_meta:
                    raise SFTGRPOError("RLVR checkpoint metadata changed during admission")
                loop = meta.get(RLVR_META_KEY)
                if not isinstance(loop, dict):
                    raise SFTGRPOError("RLVR resume identity mismatch: checkpoint, data, policy, runtime or topology changed")
                assert_grpo_resume_identity(identity, loop.get("resume_identity"), caller="RLVR")
                completed = loop["completed_optimizer_step"]
                if completed != cfg.resume_from_step or meta["step"] != completed:
                    raise SFTGRPOError("RLVR checkpoint filename and completed optimizer step disagree")
                rank_rng = loop.get("rank_rng")
                if not isinstance(rank_rng, Mapping) or "rank_states" not in rank_rng:
                    raise SFTGRPOError("RLVR checkpoint rank_rng must be a rank-state envelope")
                resume_rng = select_dataloader_rank_state(
                    rank_rng, ddp_rank=rank, ddp_world_size=world_size
                )
                assert resume_rng is not None
                validate_feature_prediction_resume(meta, loaded.model.feature_pred_config)
                loaded.model.load_state_dict(state, strict=True, assign=False)
                loaded.model.assert_vocab_consistent()
                optimizer.load_state_dict(optim_state)
                del state, optim_state
                resumed_wandb = loop["wandb"]
                later = any(int(path.stem.removeprefix("rlvr_complete_")) > completed for path in resume_dir.glob("rlvr_complete_*.json"))
                if later and (not cfg.wandb.new_run or output.resolve() == resume_dir.resolve()):
                    raise SFTGRPOError("resuming an older checkpoint requires wandb.new_run=true and a fresh output directory")
                if output.resolve() != resume_dir.resolve() and (output.exists() and any(output.iterdir())):
                    raise SFTGRPOError("a forked RLVR run requires an empty output_dir")
            elif output.exists() and any(output.iterdir()):
                raise SFTGRPOError("a new RLVR run requires an empty output_dir; resume explicitly")
            output.mkdir(parents=True, exist_ok=True)
        if t.distributed.is_initialized():
            identities: list[Any] = [None] * world_size
            t.distributed.all_gather_object(identities, identity["sha256"])
            if len(set(identities)) != 1:
                raise SFTGRPOError("RLVR ranks disagree on admitted data, model, configuration or runtime identity")
        logger.info(
            "RLVR startup: train_feature_head_only={}, trainable_parameters={}, total_parameters={}",
            str(train_feature_head_only).lower(),
            sum(p.numel() for p in loaded.model.parameters() if p.requires_grad),
            sum(p.numel() for p in loaded.model.parameters()),
        )
        if completed == cfg.num_iterations:
            logger.info("RLVR checkpoint already completed all {} optimizer steps", completed)
            worker_start_step = completed
            exit_code = 0
            return
        assert replay_settings is not None
        worker_start_step = completed
        staging_settings = frozen_decoder_staging_settings(user_config)
        if staging_settings["max_shadow_bytes"] > 0:
            with rank_synchronized_stage("RLVR frozen decoder staging", device=device):
                staging_started = time.perf_counter()
                frozen_report = frozen_stack.enter_context(stage_frozen_decoder_linears(
                    loaded.model, **staging_settings,
                    train_feature_head_only=train_feature_head_only,
                    world_size=world_size,
                ))
                frozen_setup_seconds = time.perf_counter() - staging_started
                logger.info("RLVR frozen-decoder worker setup: {}", {
                    **asdict(frozen_report), "worker_start_step": completed,
                    "setup_seconds": frozen_setup_seconds,
                })
        with rank_synchronized_stage("RLVR WandB initialization", device=device):
            telemetry = RLVRWandb(cfg.wandb, rank=rank, output_dir=output,
                run_config={"resume_identity": identity}, resumed=resumed_wandb, new_run=bool(cfg.wandb.new_run))
        if resume_rng is not None:
            with rank_synchronized_stage("RLVR RNG restoration", device=device):
                restore_rank_rng(resume_rng, device)
        training_rows = rows_by_task(loaded.bundle.train)
        local_prompts = int(cfg.global_prompts_per_step) // world_size
        if completed == 0:
            metrics, samples = evaluate_rlvr(loaded, cfg=cfg, rank=rank, world_size=world_size)
            with rank_synchronized_stage("RLVR initial telemetry", device=device):
                if rank == 0:
                    for task in RLVR_TASKS:
                        if metrics[f"validation/{task}/parse_rate"] == 0:
                            logger.warning(
                                "RLVR initial validation {}: 0/{:g} completions matched the native "
                                "answer format (training_enabled={}). Inspect validation/samples "
                                "and the reward-format requirements in docs/pipelines/encoder-decoder-rlvr.md.",
                                task, metrics[f"validation/{task}/count"], task in tasks,
                            )
                telemetry.log(0, metrics, samples)
        while completed < cfg.num_iterations:
            # Head-only replay keeps the trunk in collection's no-grad mode.
            # Retain its compiled kernels between checkpoint worker re-execs;
            # compiler-limit exhaustion already raises instead of falling back.
            if not train_feature_head_only:
                _reset_step_compiler_state()
            optimizer.zero_grad(set_to_none=True)
            multiplier = flat_then_linear_decay(completed / cfg.num_iterations, decay_start=float(cfg.lr_decay_start))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = parameter_group["initial_lr"] * multiplier
            if device.type == "cuda":
                t.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            rewards, replay_metrics = [], []
            # Additive counters are reduced across ranks only after the update.
            counters = dict.fromkeys(("zero_variance_groups", "groups", "generated_tokens", "forced_tokens", "live_actions", "unused_actions", "positive_features", "negative_features", "parse_valid", "stop_eos", "stop_budget", "stop_invalid_structure"), 0.0)
            task_totals = t.zeros((len(RLVR_TASKS), 2), dtype=t.float64, device=device)
            collection_seconds = replay_seconds = 0.0
            with rank_synchronized_stage("RLVR collection and feature replay", device=device):
                for local_ordinal in range(local_prompts):
                    ordinal = rank * local_prompts + local_ordinal
                    row = select_training_row(training_rows, seed=int(cfg.seed), step=completed, ordinal=ordinal,
                                              tasks=tasks)
                    collection_started = time.monotonic()
                    with rlvr_group_replay_cache(
                        loaded.model, settings=replay_settings, report=replay_report,
                        train_feature_head_only=train_feature_head_only, world_size=world_size,
                    ) as replay_cache:
                        group = _group(loaded, row, cfg=cfg, group_size=int(cfg.group_size),
                                       seed=rlvr_seed(int(cfg.seed), "train", completed, ordinal, row.task, row.row_id),
                                       replay_cache=replay_cache)
                        collection_seconds += time.monotonic() - collection_started
                        rewards.extend(trajectory.reward for trajectory in group.trajectories)
                        task_totals[RLVR_TASKS.index(row.task)] += t.tensor(
                            [sum(trajectory.reward for trajectory in group.trajectories), len(group.trajectories)], device=device)
                        counters["groups"] += 1
                        counters["zero_variance_groups"] += int(all(value == 0 for value in group.advantages))
                        for trajectory in group.trajectories:
                            counters["generated_tokens"] += len(trajectory.token_ids) - len(row.prompt_token_ids)
                            counters["forced_tokens"] += sum(value == 0 for value in trajectory.sampled_mask[len(row.prompt_token_ids):])
                            counters["live_actions"] += len(trajectory.live_actions)
                            counters["unused_actions"] += len(trajectory.actions) - len(trajectory.live_actions)
                            counters[f"stop_{trajectory.stop_reason}"] += 1
                            completion = loaded.tokenizer.decode(list(trajectory.token_ids[len(row.prompt_token_ids):]))
                            counters["parse_valid"] += grade_completion_result(row, completion).valid_format
                            for action in trajectory.actions:
                                counters["positive_features"] += len(action.policy_trace.positive_rows)
                                counters["negative_features"] += len(action.policy_trace.negative_rows)
                        replay_started = time.monotonic()
                        replay_metrics.append(backward_rlvr_group(
                            loaded.model, group, policy_config=loaded.policy,
                            loss_scale=1 / local_prompts,
                            train_feature_head_only=train_feature_head_only, replay_cache=replay_cache,
                        ))
                    replay_seconds += time.monotonic() - replay_started
            norm = clip_and_step(loaded.model, optimizer, parameter_manifest=manifest,
                                 max_norm=float(cfg.grad_clip_max_norm), device=device)
            completed += 1
            if device.type == "cuda":
                t.cuda.synchronize(device)
            aggregate = t.tensor([sum(rewards), len(rewards), *counters.values(),
                                  sum(metric.loss for metric in replay_metrics)], dtype=t.float64, device=device)
            bounds = t.tensor([min(metric.ratio_min for metric in replay_metrics),
                               max(metric.ratio_max for metric in replay_metrics), norm,
                               time.monotonic() - started, collection_seconds, replay_seconds,
                               t.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0],
                              dtype=t.float64, device=device)
            if t.distributed.is_initialized():
                t.distributed.all_reduce(aggregate)
                t.distributed.all_reduce(task_totals)
                t.distributed.all_reduce(bounds[:1], op=t.distributed.ReduceOp.MIN)
                t.distributed.all_reduce(bounds[1:], op=t.distributed.ReduceOp.MAX)
            reward = float(aggregate[0] / aggregate[1])
            metrics = {"train/reward": reward, "train/reward_std": (reward * (1 - reward))**0.5,
                       "train/trajectories": float(aggregate[1]), "train/gradient_norm_max_local": float(bounds[2]),
                       "train/lr_multiplier": multiplier, "train/policy_loss": float(aggregate[-1]) / cfg.global_prompts_per_step,
                       "train/ratio_min": float(bounds[0]), "train/ratio_max": float(bounds[1]),
                       "runtime/step_seconds": float(bounds[3]), "runtime/collection_seconds": float(bounds[4]),
                       "runtime/replay_seconds": float(bounds[5]), "runtime/peak_memory_gib": float(bounds[6])}
            for index, key in enumerate(counters, start=2):
                metrics[f"train/{key}"] = float(aggregate[index])
            metrics["train/parse_rate"] = metrics["train/parse_valid"] / metrics["train/trajectories"]
            metrics["train/zero_variance_fraction"] = metrics["train/zero_variance_groups"] / metrics["train/groups"]
            for task, (correct, count) in zip(RLVR_TASKS, task_totals.cpu().tolist(), strict=True):
                metrics[f"train/{task}/count"] = count
                if count:
                    metrics[f"train/{task}/accuracy"] = correct / count
            for index, parameter_group in enumerate(optimizer.param_groups):
                metrics[f"train/lr_group_{index}"] = parameter_group["lr"]
            samples = []
            if completed % cfg.eval_every == 0 or completed == cfg.num_iterations:
                evaluation, samples = evaluate_rlvr(loaded, cfg=cfg, rank=rank, world_size=world_size)
                metrics.update(evaluation)
            with rank_synchronized_stage("RLVR telemetry", device=device):
                telemetry.log(completed, metrics, samples)
                logger.info("RLVR step={} reward={} gradient_norm={}", completed, metrics["train/reward"], norm)
            if completed % cfg.save_every == 0 or completed == cfg.num_iterations:
                with rank_synchronized_stage("RLVR RNG capture", device=device):
                    local_rng = {**capture_rank_rng(device), "ddp_world_size": world_size}
                rank_rng = build_dataloader_state_envelope(local_rng, ddp_rank=rank, ddp_world_size=world_size)
                with rank_synchronized_stage("RLVR checkpoint preflight", device=device):
                    loaded.cache_snapshot.assert_stable()
                    if any(path.exists() for path in checkpoint_files(output, completed, world_size)):
                        raise SFTGRPOError("refusing to overwrite an existing RLVR checkpoint")
                with rank_synchronized_stage("RLVR checkpoint save", device=device):
                    meta = {**loaded.base_meta, "step": completed, RLVR_META_KEY: {
                        "version": 1, "completed_optimizer_step": completed, "resume_identity": identity,
                        "rank_rng": rank_rng, "wandb": telemetry.state, "user_config": user_config}}
                    fp_config = serialize_feature_prediction_config(loaded.model.feature_pred_config)
                    meta["feature_prediction_config"] = fp_config
                    meta["user_config"] = {**meta.get("user_config", {}), "feature_prediction": fp_config}
                    save_checkpoint(str(output), completed, loaded.model.state_dict(), optimizer.state_dict(), meta, rank=rank)
                with rank_synchronized_stage("RLVR checkpoint completion", device=device):
                    if rank == 0:
                        atomic_json_dump({"step": completed, "world_size": world_size}, output / f"rlvr_complete_{completed:06d}.json")
                if completed < cfg.num_iterations:
                    with rank_synchronized_stage("RLVR WandB checkpoint flush", device=device):
                        telemetry.finish()
                    telemetry = None
                    # Once a fork has a checkpoint in its own output directory,
                    # subsequent worker restarts resume that checkpoint and run.
                    sys.argv = [argument for argument in sys.argv if not argument.startswith(("resume_checkpoint_dir=", "wandb.new_run="))]
                    sys.argv.extend(["resume_checkpoint_dir=null", "wandb.new_run=false"])
                    if frozen_report is not None:
                        with rank_synchronized_stage("RLVR frozen decoder cleanup", device=device):
                            _close_frozen_decoder_worker(
                                frozen_stack, frozen_report, setup_seconds=frozen_setup_seconds,
                                first_step=worker_start_step, completed=completed,
                            )
                    with rank_synchronized_stage("RLVR hidden replay completion", device=device):
                        _log_hidden_replay_worker(
                            replay_report, replay_settings,
                            first_step=worker_start_step, completed=completed,
                        )
                        replay_worker_reported = True
                    _reexec_from_checkpoint(completed)
                    raise AssertionError("RLVR checkpoint worker restart unexpectedly returned")
        exit_code = 0
    finally:
        try:
            _close_frozen_decoder_worker(
                frozen_stack, frozen_report, setup_seconds=frozen_setup_seconds,
                first_step=worker_start_step, completed=completed,
            )
        finally:
            try:
                if not replay_worker_reported:
                    _log_hidden_replay_worker(
                        replay_report, replay_settings,
                        first_step=worker_start_step, completed=completed,
                        completed_successfully=exit_code == 0,
                    )
            finally:
                try:
                    if telemetry is not None:
                        telemetry.finish(exit_code=exit_code)
                finally:
                    compute_cleanup()


def main() -> None:
    _reexec_for_deterministic_kernels()
    config_name, overrides = parse_config_name_and_overrides(sys.argv[1:], default_config_name="conditional_encdec_rlvr")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs" / "training"), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    train(cfg)


if __name__ == "__main__":
    main()
