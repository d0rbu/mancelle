"""Dump chat-eval prompts and generated responses under full-context conditioning."""

from __future__ import annotations

import os
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import torch as t
import torch.distributed as dist
import yaml
from beartype import beartype
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine import Engine
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.eval.chat_generation import render_chat_completion
from steering.eval.feature_config import (
    FeatureConfig,
    parse_strict_feature_configs,
    require_nonempty_feature_configs,
    resolve_next_segment_conditioning_offset,
    validate_inactive_bucket_feature_configs,
)
from steering.eval.inference_engine import build_wrapper_for_features
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.target_real_runner import build_shared_context_feature_provider
from steering.refined_types import NonNegativeInt, PositiveInt
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

DEFAULT_TASK_NAMES: tuple[str, ...] = (
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
)

_LETTER_BOUNDARY_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


@dataclass(frozen=True)
class DumpRow:
    index: int
    user_message: str
    expected_answer: str
    completion: str
    outcome: object
    eval_error: str | None
    predicted_letter: str | None


@beartype
def iter_local_indices(
    n: NonNegativeInt,
    rank: NonNegativeInt,
    world_size: PositiveInt,
) -> range:
    if rank >= world_size:
        raise ValueError(
            f"rank must satisfy 0 <= rank < world_size; got {rank}/{world_size}"
        )
    return range(rank, n, world_size)


def _task_names_from_config(raw_tasks) -> list[str]:
    if raw_tasks is None:
        return list(DEFAULT_TASK_NAMES)
    names = [str(name) for name in raw_tasks]
    if not names:
        raise ValueError("tasks must contain at least one entry")
    unknown = [name for name in names if name not in DEFAULT_TASK_NAMES]
    if unknown:
        raise ValueError(
            f"unknown chat task(s) {unknown}; supported: {list(DEFAULT_TASK_NAMES)}"
        )
    return names


def build_task(name: str):
    if name in ("ARC-Easy", "ARC-Challenge"):
        return ARC(subset=name, split="test")
    if name == "MMLU":
        return MMLU(subset="all", split="test")
    if name == "GSM8K":
        return GSM8K(subset="main", split="test")
    if name == "HumanEval":
        return HumanEval()
    if name == "SpellingBee":
        return SpellingBee(size=256, split="test")
    raise ValueError(f"unknown chat task {name!r}")


def extract_user_message(conversation: dict) -> str:
    for msg in conversation.get("messages", []) or []:
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def extract_expected_answer(conversation: dict) -> str:
    for msg in reversed(conversation.get("messages", []) or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""


def _match_first_letter(completion: str, letters: list[str]) -> str | None:
    upper_letters = {letter.upper() for letter in letters}
    for match in _LETTER_BOUNDARY_RE.finditer(completion or ""):
        letter = match.group(1).upper()
        if letter in upper_letters:
            return letter
    return None


def evaluate_completion(
    task,
    conversation: dict,
    completion: str,
) -> tuple[object, str | None, str | None]:
    is_categorical = getattr(task, "eval_type", None) == "categorical"
    letters = conversation.get("letters") if is_categorical else None
    if is_categorical and letters:
        predicted = _match_first_letter(completion, list(letters))
        if predicted is None:
            return None, "no answer letter matched in completion", None
        try:
            return task.evaluate(conversation, predicted), None, predicted
        except Exception as exc:  # noqa: BLE001 - captured in YAML for inspection
            return None, f"{type(exc).__name__}: {exc}", predicted

    try:
        return task.evaluate(conversation, completion), None, None
    except Exception as exc:  # noqa: BLE001 - captured in YAML for inspection
        return None, f"{type(exc).__name__}: {exc}", None


@beartype
def gather_rows(
    local_rows: list[DumpRow],
    *,
    n_problems: NonNegativeInt,
    ddp: bool,
    ddp_world_size: PositiveInt,
    all_gather_object,
) -> list[DumpRow]:
    if ddp:
        gathered: list[list[DumpRow] | None] = [None] * ddp_world_size
        all_gather_object(gathered, local_rows)
        rows = [row for chunk in gathered if chunk is not None for row in chunk]
    else:
        rows = list(local_rows)
    rows.sort(key=lambda row: row.index)
    indices = [row.index for row in rows]
    if len(indices) != len(set(indices)):
        raise AssertionError(f"duplicate dumped indices: {indices}")
    expected = set(range(n_problems))
    actual = set(indices)
    if actual != expected:
        raise AssertionError(
            f"dump coverage mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return rows


@beartype
def dump_one_feature_config(
    *,
    feature_config: FeatureConfig,
    model: EncoderDecoderGPT,
    tokenizer,
    context_provider,
    target_mode: TargetConditioningMode,
    task_names: list[str],
    n_problems: NonNegativeInt,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    autocast_ctx,
    ddp: bool,
    ddp_rank: NonNegativeInt,
    ddp_world_size: PositiveInt,
    all_gather_object,
) -> dict[str, list[DumpRow]]:
    results: dict[str, list[DumpRow]] = {}
    for task_name in task_names:
        task = build_task(task_name)
        n = NonNegativeInt.parse(min(n_problems, len(task)))
        local_rows: list[DumpRow] = []
        for idx in iter_local_indices(n, ddp_rank, ddp_world_size):
            conversation = task[idx]
            encoded_prompt = render_chat_completion(tokenizer, conversation).token_ids
            wrapper = build_wrapper_for_features(
                model,
                tokenizer,
                features=feature_config.features,
                context_provider=context_provider,
                target_mode=target_mode,
            )
            engine = Engine(wrapper, tokenizer)
            with autocast_ctx, wrapper.conditioning_context(
                context_token_ids=[encoded_prompt],
            ):
                generated, _masks = engine.generate_batch(
                    encoded_prompt,
                    num_samples=1,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            prefix_len = len(encoded_prompt)
            completion = tokenizer.decode(generated[0][prefix_len:])
            outcome, eval_error, predicted_letter = evaluate_completion(
                task,
                conversation,
                completion,
            )
            local_rows.append(
                DumpRow(
                    index=idx,
                    user_message=extract_user_message(conversation),
                    expected_answer=extract_expected_answer(conversation),
                    completion=completion,
                    outcome=outcome,
                    eval_error=eval_error,
                    predicted_letter=predicted_letter,
                )
            )
        results[task_name] = gather_rows(
            local_rows,
            n_problems=n,
            ddp=ddp,
            ddp_world_size=ddp_world_size,
            all_gather_object=all_gather_object,
        )
    return results


def _write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                data,
                handle,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=10000,
            )
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def dump_chat_responses(
    cfg: DictConfig,
    *,
    ddp: bool,
    ddp_rank: NonNegativeInt,
    ddp_world_size: PositiveInt,
    device,
    all_gather_object=None,
    barrier=None,
) -> dict[str, Any]:
    checkpoint_dir = str(cfg.checkpoint_dir)
    step = int(cfg.step)
    feature_configs = parse_strict_feature_configs(
        OmegaConf.to_container(cfg.feature_configs, resolve=True)
    )
    if not feature_configs:
        raise ValueError(
            "dump_chat_responses requires feature_configs. Empty target-delta "
            "dumps are intentionally unsupported for encoder-decoder models."
        )
    require_nonempty_feature_configs(feature_configs, surface="dump_chat_responses")
    n_problems = NonNegativeInt.parse(cfg.n_problems)

    model, tokenizer, meta = build_encdec_model(
        checkpoint_dir,
        step,
        device,
        ModelPhase.EVAL,
        eval_conditioning=EvalConditioning.deferred("dump_chat_responses"),
    )
    reject_feature_clusters_metadata(meta, "dump_chat_responses checkpoint")
    if not isinstance(model, EncoderDecoderGPT):
        raise TypeError("dump_chat_responses requires an EncoderDecoderGPT checkpoint")
    require_fixed_checkpoint_segment_mode(
        meta,
        surface="dump_chat_responses",
        checkpoint_dir=checkpoint_dir,
    )
    model.eval()
    context_provider = build_shared_context_feature_provider(
        cfg,
        eval_name="dump_chat_responses",
    )
    validate_inactive_bucket_feature_configs(
        feature_configs,
        num_buckets=model.encoder.config.num_buckets,
        checkpoint_meta=meta,
        checkpoint_label=f"{checkpoint_dir} step {step}",
    )
    target_mode = TargetConditioningMode.from_next_segment_offset(
        resolve_next_segment_conditioning_offset(cfg, meta)
    )
    device_type = device.type if hasattr(device, "type") else str(device)
    autocast_ctx = (
        t.amp.autocast(device_type=device_type, dtype=t.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    task_names = _task_names_from_config(cfg.get("tasks"))

    dumped: dict[str, dict[str, list[DumpRow]]] = {}
    for feature_config in feature_configs:
        if ddp_rank == 0:
            logger.info(f"Dumping chat responses for {feature_config.name}")
        dumped[feature_config.name] = dump_one_feature_config(
            feature_config=feature_config,
            model=model,
            tokenizer=tokenizer,
            context_provider=context_provider,
            target_mode=target_mode,
            task_names=task_names,
            n_problems=n_problems,
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            autocast_ctx=autocast_ctx,
            ddp=ddp,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            all_gather_object=all_gather_object,
        )

    output: dict[str, Any] = {
        "checkpoint_dir": checkpoint_dir,
        "step": step,
        "target_conditioning_mode": target_mode.value,
        "n_problems": n_problems,
        "max_new_tokens": int(cfg.max_new_tokens),
        "temperature": float(cfg.temperature),
        "top_k": int(cfg.top_k),
        "variants": {
            name: {
                task_name: [asdict(row) for row in rows]
                for task_name, rows in task_rows.items()
            }
            for name, task_rows in dumped.items()
        },
    }
    if ddp_rank == 0:
        _write_yaml_atomic(Path(str(cfg.output)), output)
    if barrier is not None:
        barrier()
    return output


@hydra.main(
    config_path="../../configs/eval",
    config_name="dump_chat_responses",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    device_type = autodetect_device_type() if not cfg.device_type else cfg.device_type
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    try:
        dump_chat_responses(
            cfg,
            ddp=ddp,
            ddp_rank=NonNegativeInt.parse(ddp_rank),
            ddp_world_size=PositiveInt.parse(ddp_world_size),
            device=device,
            all_gather_object=dist.all_gather_object if ddp else None,
            barrier=dist.barrier if ddp else None,
        )
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
