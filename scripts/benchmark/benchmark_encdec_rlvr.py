"""Time complete generated-token RLVR updates on one real CUDA worker.

Run with ``torchrun --standalone --nproc_per_node=1 -m
scripts.benchmark.benchmark_encdec_rlvr -- base_checkpoint_dir=...
bundle_dir=... stats_path=... output_dir=...``. Warmup performs real updates;
compare separate fresh processes with identical inputs, seeds and step counts.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch as t
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init
from nanochat.hydra_cli import parse_config_name_and_overrides
from nanochat.optim import DistMuonAdamW
from scripts.train.conditional_encdec_rlvr import RLVRLoaded, _group, load_rlvr_inputs
from scripts.train.conditional_encdec_sft_grpo import (
    _assert_attention_impl,
    _build_optimizer,
    _configure_deterministic_algorithms,
    _reexec_for_deterministic_kernels,
    _reset_step_compiler_state,
)
from steering.conditional.encoder_decoder.frozen_decoder import (
    StagingReport,
    frozen_decoder_staging_settings,
    stage_frozen_decoder_linears,
)
from steering.conditional.encoder_decoder.rlvr_policy import (
    backward_rlvr_group,
    rlvr_group_replay_cache,
)
from steering.conditional.encoder_decoder.rlvr_replay_cache import (
    HiddenReplayReport,
    HiddenReplaySettings,
    hidden_replay_report_snapshot,
    hidden_replay_settings,
)
from steering.conditional.encoder_decoder.rlvr_runtime import clip_and_step, rlvr_seed
from steering.conditional.encoder_decoder.sft_grpo_training import (
    configure_grpo_trainable_parameters,
    resolve_train_feature_head_only,
)
from steering.eval.chat_eval_result_identity import self_conditioned_code_identity


def _cpu_payload(value: Any) -> Any:
    """Make portable parity receipts without serializing application objects."""
    if isinstance(value, t.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _cpu_payload(getattr(value, field.name))
                for field in fields(value) if not field.name.startswith("_")}
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_cpu_payload(item) for item in value)
    return value


def _step(
    loaded: RLVRLoaded,
    optimizer: DistMuonAdamW,
    rows: tuple[Any, ...],
    manifest: tuple[str, ...],
    cfg: DictConfig,
    step: int,
    *,
    synchronize_phases: bool = False,
    replay_settings: HiddenReplaySettings | None = None,
    replay_report: HiddenReplayReport | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    if replay_settings is None:
        replay_settings = hidden_replay_settings(cfg)
    if replay_report is None:
        replay_report = HiddenReplayReport()
    device = loaded.model.get_device()
    t.cuda.synchronize(device)
    t.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    reset_compiler = (
        not cfg.train_feature_head_only if cfg.compiler_reset is None else cfg.compiler_reset
    )
    if reset_compiler:
        _reset_step_compiler_state()
    optimizer.zero_grad(set_to_none=True)
    phase_seconds: dict[str, float] = {}
    groups, replay_metrics = [], []
    for ordinal, row in enumerate(rows):
        phase_start = time.perf_counter()
        with rlvr_group_replay_cache(
            loaded.model, settings=replay_settings, report=replay_report,
            train_feature_head_only=bool(cfg.train_feature_head_only), world_size=1,
        ) as replay_cache:
            with t.profiler.record_function("rlvr/collection"):
                group = _group(
                    loaded, row, cfg=cfg, group_size=int(cfg.group_size),
                    seed=rlvr_seed(int(cfg.seed), "train", step, ordinal, row.task, row.row_id),
                    replay_cache=replay_cache,
                )
            if synchronize_phases:
                t.cuda.synchronize(device)
                phase_seconds["collection"] = phase_seconds.get("collection", 0.0) + time.perf_counter() - phase_start
            groups.append(group)
            phase_start = time.perf_counter()
            with t.profiler.record_function("rlvr/replay_backward"):
                replay_metrics.append(backward_rlvr_group(
                    loaded.model, group, policy_config=loaded.policy,
                    loss_scale=1 / len(rows), train_feature_head_only=bool(cfg.train_feature_head_only),
                    replay_cache=replay_cache,
                ))
        if synchronize_phases:
            t.cuda.synchronize(device)
            phase_seconds["replay_backward"] = phase_seconds.get("replay_backward", 0.0) + time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    with t.profiler.record_function("rlvr/clip_optimizer"):
        norm = clip_and_step(
            loaded.model, optimizer, parameter_manifest=manifest,
            max_norm=float(cfg.grad_clip_max_norm), device=device,
        )
    t.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if synchronize_phases:
        phase_seconds["clip_optimizer"] = time.perf_counter() - phase_start
    # Receipt construction, copies and serialization are outside timed regions.
    trajectories = [trajectory for group in groups for trajectory in group.trajectories]
    return {
        "step": step,
        "seconds": elapsed,
        "phase_seconds": phase_seconds,
        "phase_synchronized": synchronize_phases,
        "gradient_norm_before_clip": norm,
        "peak_allocated_bytes": t.cuda.max_memory_allocated(device),
        "generated_tokens": sum(len(trajectory.token_ids) - len(group.prompt.prompt_token_ids)
                                for group in groups for trajectory in group.trajectories),
        "live_actions": sum(len(trajectory.live_actions) for trajectory in trajectories),
        "rewards": [trajectory.reward for trajectory in trajectories],
        "advantages": [list(group.advantages) for group in groups],
        "replay": [asdict(metric) for metric in replay_metrics],
    }, groups


def benchmark(cfg: DictConfig) -> None:
    train_feature_head_only = resolve_train_feature_head_only(cfg)
    if cfg.compiler_reset is not None and type(cfg.compiler_reset) is not bool:
        raise ValueError("compiler_reset must be null or a boolean")
    if int(cfg.warmup_steps) < 0 or int(cfg.measure_steps) < 1:
        raise ValueError("warmup_steps must be nonnegative and measure_steps positive")
    if int(cfg.group_size) < 2:
        raise ValueError("group_size must be at least two")
    staging_settings = frozen_decoder_staging_settings(cfg)
    replay_settings = hidden_replay_settings(cfg)
    replay_report = HiddenReplayReport()
    output = Path(cfg.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("benchmark output_dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen_stack = ExitStack()
    frozen_report: StagingReport | None = None
    frozen_setup_seconds = 0.0
    result: dict[str, Any] | None = None
    benchmark_completed = False
    _, rank, _, world_size, device = compute_init("cuda")
    try:
        if world_size != 1:
            raise ValueError("this benchmark requires exactly one torchrun worker")
        attention = _assert_attention_impl(device_type="cuda")
        deterministic = _configure_deterministic_algorithms(device_type="cuda", attention_impl=attention)
        loaded = load_rlvr_inputs(cfg, device=device, rank=rank)
        manifest = configure_grpo_trainable_parameters(
            loaded.model, train_lm_head=False,
            train_feature_head_only=train_feature_head_only,
        )
        optimizer = _build_optimizer(loaded.model, cfg, fp_head_lr=None)
        if not isinstance(optimizer, DistMuonAdamW):
            raise ValueError("launch with torchrun --standalone --nproc_per_node=1")
        indices = tuple(cfg.row_indices)
        if not indices or any(type(index) is not int or index < 0 or index >= len(loaded.bundle.train) for index in indices):
            raise ValueError("row_indices must contain valid training-bundle indices")
        rows = tuple(loaded.bundle.train[index] for index in indices)
        parameters = dict(loaded.model.named_parameters())
        frozen_versions = {name: parameter._version for name, parameter in parameters.items() if not parameter.requires_grad}
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "git_sha": git_sha,
            "source_identity": self_conditioned_code_identity(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu": t.cuda.get_device_name(device),
            "gpu_capability": t.cuda.get_device_capability(device),
            "torch": t.__version__, "cuda": t.version.cuda,
            "compute_dtype": str(COMPUTE_DTYPE),
            "autocast_enabled": t.is_autocast_enabled("cuda"),
            "attention": attention, "deterministic": deterministic,
            "compiler": loaded.compiler_config,
            "compiler_reset_effective": (
                not train_feature_head_only
                if cfg.compiler_reset is None else cfg.compiler_reset
            ),
            "base_identity": loaded.base_identity,
            "bundle_sha256": loaded.bundle.identity_sha256,
            "geometry": {
                "effective_prompts_per_update": len(rows),
                "decoder": asdict(loaded.model.dec_config),
                "encoder": asdict(loaded.model.encoder.config),
                "feature_head": asdict(loaded.model.feature_pred_config),
                "cadence": loaded.cadence,
                "parameter_count": sum(parameter.numel() for parameter in parameters.values()),
                "trainable_parameter_count": sum(parameters[name].numel() for name in manifest),
            },
            "rows": [{"index": index, "row_id": row.row_id, "task": row.task,
                      "prompt_tokens": len(row.prompt_token_ids)} for index, row in zip(indices, rows, strict=True)],
            "manifest": manifest,
            "timing_contract": "Outer CUDA synchronization; compiler reset, collection, native reward, replay, clipping and optimizer included. Separate phase/profiler steps are excluded from measured samples.",
            "warmup": [], "measured": [],
        }
        if staging_settings["max_shadow_bytes"] > 0:
            staging_started = time.perf_counter()
            frozen_report = frozen_stack.enter_context(stage_frozen_decoder_linears(
                loaded.model, **staging_settings,
                train_feature_head_only=train_feature_head_only,
                world_size=world_size,
            ))
            frozen_setup_seconds = time.perf_counter() - staging_started
            logger.info("RLVR benchmark frozen-decoder setup: {}", {
                **asdict(frozen_report), "setup_seconds": frozen_setup_seconds,
            })
        parity_steps = set(cfg.parity_steps)
        for step in range(int(cfg.warmup_steps) + int(cfg.measure_steps)):
            measured_index = step - int(cfg.warmup_steps)
            receipt, groups = _step(loaded, optimizer, rows, manifest, cfg, step,
                                    replay_settings=replay_settings, replay_report=replay_report)
            result["warmup" if measured_index < 0 else "measured"].append(receipt)
            if measured_index in parity_steps:
                gradients = {name: parameters[name].grad for name in manifest}
                receipt["nonzero_gradient_parameters"] = [name for name, gradient in gradients.items()
                                                           if gradient is not None and bool(t.count_nonzero(gradient))]
                t.save(_cpu_payload({
                    "groups": groups,
                    "clipped_gradients": gradients,
                    "updated_parameters": {name: parameters[name] for name in manifest},
                    "optimizer": optimizer.state_dict(),
                    "receipt": receipt,
                }), output / f"parity-{measured_index:03d}.pt")
                del gradients
            logger.info("RLVR benchmark step {}: {:.6f}s, rewards={}, live_actions={}",
                        step, receipt["seconds"], receipt["rewards"], receipt["live_actions"])
            del groups
            (output / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        next_step = int(cfg.warmup_steps) + int(cfg.measure_steps)
        if cfg.measure_phases:
            result["phase_diagnostic"], diagnostic_groups = _step(
                loaded, optimizer, rows, manifest, cfg, next_step, synchronize_phases=True,
                replay_settings=replay_settings, replay_report=replay_report,
            )
            del diagnostic_groups
            next_step += 1
        if cfg.profile:
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True, with_stack=False) as profile:
                result["profile_diagnostic"], diagnostic_groups = _step(
                    loaded, optimizer, rows, manifest, cfg, next_step,
                    replay_settings=replay_settings, replay_report=replay_report,
                )
                del diagnostic_groups
            profile.export_chrome_trace(str(output / "trace.json"))
        result["frozen_parameters_unchanged"] = all(
            parameters[name]._version == version and parameters[name].grad is None
            for name, version in frozen_versions.items()
        )
        if not result["frozen_parameters_unchanged"]:
            raise RuntimeError("frozen parameter version or gradient changed")
        seconds = [row["seconds"] for row in result["measured"]]
        result["median_seconds"] = statistics.median(seconds)
        result["mean_seconds"] = statistics.mean(seconds)
        (output / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        logger.info("RLVR median complete update: {:.6f}s; receipts: {}", result["median_seconds"], output)
        benchmark_completed = True
    finally:
        try:
            cleanup_started = time.perf_counter()
            frozen_stack.close()
            cleanup_seconds = time.perf_counter() - cleanup_started
            if result is not None:
                result["hidden_replay_cache"] = {
                    **hidden_replay_report_snapshot(replay_report),
                    "requested": asdict(replay_settings), "benchmark_completed": benchmark_completed,
                    "timing_contract": "Per-group admission, capture, validation, lookup and cleanup are inside complete update timing. Worker totals include warmup, measured and diagnostic updates; no cache persists across groups.",
                }
                (output / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
            if result is not None and frozen_report is not None:
                updates = len(result["warmup"]) + len(result["measured"])
                updates += int("phase_diagnostic" in result) + int("profile_diagnostic" in result)
                result["frozen_decoder_staging"] = {
                    **asdict(frozen_report), "benchmark_completed": benchmark_completed,
                    "requested": staging_settings, "setup_seconds": frozen_setup_seconds,
                    "cleanup_seconds": cleanup_seconds, "completed_worker_updates": updates,
                    "setup_seconds_per_update": frozen_setup_seconds / updates if updates else None,
                    "timing_contract": "Persistent worker staging spans warmup, measured and diagnostic updates. Setup/cleanup excluded from native _step timings and reported separately; per-forward guards remain timed.",
                }
                (output / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        finally:
            if t.distributed.is_initialized():
                compute_cleanup()


def main() -> None:
    _reexec_for_deterministic_kernels()
    name, overrides = parse_config_name_and_overrides(sys.argv[1:], default_config_name="profiling/encdec_rlvr")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    benchmark(cfg)


if __name__ == "__main__":
    main()
