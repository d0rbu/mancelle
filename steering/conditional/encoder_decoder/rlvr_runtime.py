"""Small runtime utilities for generated, feature-policy-only RLVR.

Prompt selection is stateless. The optimizer owns distributed gradient
averaging; callers accumulate local prompt means and clip locally first.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch as t
import wandb
from loguru import logger
from wandb.util import generate_id

from nanochat.atomic_io import atomic_json_dump
from nanochat.common import DummyWandb, configure_wandb_step_metric
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder.frozen_decoder import frozen_decoder_staging_settings
from steering.conditional.encoder_decoder.rlvr_data import RLVR_TASKS
from steering.conditional.encoder_decoder.rlvr_replay_cache import hidden_replay_settings
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_training import assert_grpo_gradients_ready
from steering.sft.clipping import apply_gradient_clipping

RLVR_META_KEY = "encdec_rlvr_v1"
RLVR_SEED_VERSION = "sha256_namespace_global_prompt_v1"
RLVR_OBJECTIVE_VERSION = "prompt_group_live_decision_mean_feature_ratio_v1"
RLVR_GENERATION_VERSION = "canonical_full_prefix_turn_calculator_v2"


def configure_rlvr_compiler(model: Any) -> dict[str, int | bool]:
    """Budget static decoder specializations and fail on any later exhaustion."""

    if t._dynamo.config.suppress_errors:
        raise SFTGRPOError("RLVR requires torch compile suppress_errors=False")
    sequence_len = model.dec_config.sequence_len
    window_variants = len(set(model.window_sizes))
    # Account for the one startup compatibility probe already performed, then
    # every sequence position for two gradient modes and each static window.
    budget = 1 + 2 * window_variants * sequence_len
    t._dynamo.config.recompile_limit = max(
        t._dynamo.config.recompile_limit,
        budget,
    )
    t._dynamo.config.accumulated_recompile_limit = max(
        t._dynamo.config.accumulated_recompile_limit, budget
    )
    # Dynamic packed attention has no proved specialization bound; exhaustion
    # must raise instead of silently changing the execution path to eager.
    t._dynamo.config.fail_on_recompile_limit_hit = True  # ty: ignore[invalid-assignment]
    configured = {
        "derived_minimum": budget,
        "window_variants": window_variants,
        "sequence_len": sequence_len,
        "recompile_limit": int(t._dynamo.config.recompile_limit),
        "accumulated_recompile_limit": int(
            t._dynamo.config.accumulated_recompile_limit
        ),
        "fail_on_recompile_limit_hit": bool(
            t._dynamo.config.fail_on_recompile_limit_hit
        ),
        "suppress_errors": bool(t._dynamo.config.suppress_errors),
    }
    logger.info("RLVR torch compiler configuration: {}", configured)
    return configured


def rlvr_seed(seed: int, *coordinates: str | int) -> int:
    """Separate random streams without Python hash randomization or rank IDs."""
    payload = json.dumps([RLVR_SEED_VERSION, seed, *coordinates], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


def resolve_rlvr_tasks(tasks: object = RLVR_TASKS) -> tuple[str, ...]:
    """Validate an ordered training allow-list without changing sampling order."""
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise SFTGRPOError("tasks must be a nonempty list of supported RLVR task names")
    if any(not isinstance(task, str) or task not in RLVR_TASKS for task in tasks):
        raise SFTGRPOError(f"tasks must contain only exact names from {RLVR_TASKS}")
    if len(set(tasks)) != len(tasks):
        raise SFTGRPOError("tasks must not contain duplicate names")
    return cast(tuple[str, ...], tuple(tasks))


def select_training_row(rows_by_task: Mapping[str, Sequence[Any]], *, seed: int, step: int, ordinal: int,
                        tasks: Sequence[str] = RLVR_TASKS):
    """Sample a task uniformly, then a source row uniformly, with replacement."""
    tasks = resolve_rlvr_tasks(tasks)
    if set(rows_by_task) != set(RLVR_TASKS) or any(not rows_by_task[name] for name in RLVR_TASKS):
        raise SFTGRPOError("RLVR requires nonempty training rows for all four tasks")
    rng = random.Random(rlvr_seed(seed, "prompt", step, ordinal))
    task = tasks[rng.randrange(len(tasks))]
    return rows_by_task[task][rng.randrange(len(rows_by_task[task]))]


def validation_panel(rows_by_task: Mapping[str, Sequence[Any]], *, seed: int, per_task: int | None) -> tuple[Any, ...]:
    """Choose one stable seeded panel; None selects all heldout rows."""
    if set(rows_by_task) != set(RLVR_TASKS):
        raise SFTGRPOError("RLVR validation requires exactly the four configured tasks")
    if per_task is not None and (type(per_task) is not int or per_task <= 0):
        raise SFTGRPOError("validation_per_task must be positive or null for full heldout evaluation")
    result = []
    for task in RLVR_TASKS:
        rows = rows_by_task[task]
        if not rows:
            raise SFTGRPOError(f"RLVR validation has no {task} rows")
        indices = list(range(len(rows)))
        if per_task is not None:
            random.Random(rlvr_seed(seed, "validation_panel", task)).shuffle(indices)
            indices = indices[:per_task]
        result.extend(rows[index] for index in indices)
    return tuple(result)


def validate_rlvr_settings(cfg: Mapping[str, Any], *, world_size: int) -> None:
    """Reject unsupported objectives and invalid run geometry before loading."""
    resolve_rlvr_tasks(cfg.get("tasks", RLVR_TASKS))
    try:
        frozen_decoder_staging_settings(cfg)
        hidden_replay_settings(cfg)
    except ValueError as error:
        raise SFTGRPOError(str(error)) from error
    for key in ("global_prompts_per_step", "group_size", "num_iterations", "eval_every", "save_every",
                "mc_max_tokens", "gsm8k_max_tokens"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise SFTGRPOError(f"{key} must be a positive integer")
    panel_size = cfg["validation_per_task"]
    if panel_size is not None and (type(panel_size) is not int or panel_size <= 0):
        raise SFTGRPOError("validation_per_task must be positive or null for full heldout evaluation")
    if cfg["group_size"] < 2:
        raise SFTGRPOError("group_size must be at least two")
    if cfg["global_prompts_per_step"] % world_size:
        raise SFTGRPOError("global_prompts_per_step must divide evenly across world_size")
    if any(type(cfg[key]) is not int or cfg[key] < 0 for key in ("seed", "validation_seed")):
        raise SFTGRPOError("seed and validation_seed must be nonnegative integers")
    for key in ("temperature", "grad_clip_max_norm", "lr_multiplier", "init_lr_frac",
                "embedding_lr", "unembedding_lr", "matrix_lr", "encoder_embedding_lr",
                "encoder_matrix_lr", "xattn_matrix_lr"):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise SFTGRPOError(f"{key} must be finite and positive")
    if cfg["top_k"] is not None and (type(cfg["top_k"]) is not int or cfg["top_k"] <= 0):
        raise SFTGRPOError("top_k must be null or a positive integer")
    if cfg["weight_decay"] != 0 or not 0 <= cfg["lr_decay_start"] <= 1:
        raise SFTGRPOError("RLVR requires zero weight_decay and lr_decay_start in [0,1]")
    if type(cfg["resume_from_step"]) is not int or cfg["resume_from_step"] == 0 or cfg["resume_from_step"] < -1:
        raise SFTGRPOError("resume_from_step must be -1 or a positive optimizer step")
    if cfg["resume_from_step"] > cfg["num_iterations"]:
        raise SFTGRPOError("resume step exceeds num_iterations")
    if cfg["device_type"] not in ("cpu", "cuda"):
        raise SFTGRPOError("device_type must explicitly select cpu or cuda")
    logging = cfg["wandb"]
    if any(type(logging[key]) is not bool for key in ("enabled", "new_run")):
        raise SFTGRPOError("wandb.enabled and wandb.new_run must be booleans")
    if logging["mode"] not in ("online", "offline"):
        raise SFTGRPOError("wandb.mode must be online or offline")
    if any(cfg.get(key) is not None for key in ("ntp", "kl_beta", "entropy_weight", "train_lm_head")):
        raise SFTGRPOError("RLVR v1 supports only the feature-policy objective with a frozen LM head")


def clip_and_step(model, optimizer, *, parameter_manifest: tuple[str, ...], max_norm: float, device) -> float:
    """Check all connected gradients, clip locally, then let DistMuon average."""
    with rank_synchronized_stage("RLVR gradient readiness and local clipping", device=device):
        assert_grpo_gradients_ready(model, expected_manifest=parameter_manifest)
        norm = apply_gradient_clipping((p for p in model.parameters() if p.requires_grad), max_norm)
        if norm is None or not math.isfinite(float(norm)):
            raise SFTGRPOError("RLVR gradient norm is nonfinite")
    optimizer.step()
    return float(norm)


def capture_rank_rng(device: t.device) -> dict[str, Any]:
    """Save ordinary RNG state too, although policy/data streams are explicit."""
    return {
        "torch_cpu": t.get_rng_state().tolist(),
        "torch_device": None if device.type == "cpu" else t.cuda.get_rng_state(device).tolist(),
        "python": random.getstate(),
    }


def restore_rank_rng(state: Mapping[str, Any], device: t.device) -> None:
    t.set_rng_state(t.tensor(state["torch_cpu"], dtype=t.uint8))
    if device.type == "cuda":
        t.cuda.set_rng_state(t.tensor(state["torch_device"], dtype=t.uint8), device)
    version, values, gaussian = state["python"]
    random.setstate((version, tuple(values), gaussian))


def checkpoint_files(directory: Path, step: int, world_size: int) -> tuple[Path, ...]:
    return (directory / f"model_{step:06d}.pt", directory / f"meta_{step:06d}.json",
            *(directory / f"optim_{step:06d}_rank{rank}.pt" for rank in range(world_size)))


def assert_complete_checkpoint(directory: Path, step: int, world_size: int) -> None:
    marker = directory / f"rlvr_complete_{step:06d}.json"
    if not marker.is_file() or json.loads(marker.read_text()) != {"step": step, "world_size": world_size}:
        raise SFTGRPOError("RLVR resume requires a completed checkpoint for this topology")
    missing = [str(path) for path in checkpoint_files(directory, step, world_size) if not path.is_file()]
    if missing:
        raise SFTGRPOError(f"RLVR checkpoint is incomplete: {missing}")


class RLVRWandb:
    """Rank-zero logging with explicit resume identity and optimizer-step deduplication."""

    def __init__(self, cfg: Mapping[str, Any], *, rank: int, output_dir: Path,
                 run_config: Mapping[str, Any], resumed: Mapping[str, Any] | None = None,
                 new_run: bool = False):
        self.run = DummyWandb()
        self.state: dict[str, Any] = {}
        self.enabled = bool(cfg["enabled"]) and rank == 0
        self.state_path = output_dir / "wandb_state.json"
        if not self.enabled:
            return
        mode = cfg["mode"]
        if mode not in ("online", "offline"):
            raise SFTGRPOError("wandb.mode must be online or offline; use enabled=false to disable")
        parent = None if resumed is None else resumed.get("id")
        reuse = resumed is not None and bool(parent) and not new_run
        if reuse and any(resumed.get(key) != cfg[key] for key in ("project", "entity", "mode")):
            raise SFTGRPOError("WandB project/entity/mode changed; select wandb.new_run=true")
        run_id = parent if reuse else generate_id()
        last_logged = int(resumed.get("last_logged_step", -1)) if reuse else -1
        if reuse and self.state_path.is_file():
            local = json.loads(self.state_path.read_text())
            if local.get("id") == run_id:
                last_logged = max(last_logged, int(local["last_logged_step"]))
        self.run = wandb.init(project=cfg["project"], entity=cfg["entity"], name=cfg["name"],
                              id=run_id, mode=mode, resume=("must" if reuse else "never") if mode == "online" else None,
                              dir=str(output_dir), config={**run_config, "parent_run_id": parent if new_run else None})
        if self.run is None:
            raise SFTGRPOError("WandB initialization returned no run")
        try:
            configure_wandb_step_metric(self.run)
            self.state = {"id": run_id, "project": cfg["project"], "entity": cfg["entity"], "mode": mode,
                          "last_logged_step": max(last_logged, int(self.run.step) - 1), "parent_run_id": parent if new_run else None}
            atomic_json_dump(self.state, self.state_path)
        except BaseException:
            self.run.finish(exit_code=1)
            raise

    def log(self, step: int, metrics: Mapping[str, Any], samples: Sequence[Sequence[Any]] = ()) -> None:
        if not self.enabled or step <= self.state["last_logged_step"]:
            return
        payload = {**metrics, "step": step}
        if samples:
            columns = ["task", "row_id", "question", "completion", "expected", "parsed", "reward", "stop"]
            if len(samples) > 8 or any(len(row) != len(columns) for row in samples):
                raise SFTGRPOError("RLVR samples require at most eight rows with the fixed table schema")
            bounded = [[str(value)[:4096] if isinstance(value, str) else value for value in row] for row in samples]
            payload["validation/samples"] = wandb.Table(columns=columns, data=bounded)
        self.run.log(payload, step=step)
        self.state["last_logged_step"] = step
        atomic_json_dump(self.state, self.state_path)

    def finish(self, *, exit_code: int = 0) -> None:
        if not isinstance(self.run, DummyWandb):
            self.run.finish(exit_code=exit_code)
