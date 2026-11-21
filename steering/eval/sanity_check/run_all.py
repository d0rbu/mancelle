"""Unified sanity-check runner for all eval pipelines.

Runs against both the base pretrained and SFT checkpoints.
Usage: uv run python -m steering.eval.sanity_check.run_all
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from threading import current_thread, main_thread
from typing import Any, cast

import torch
import yaml

from nanochat.checkpoint_manager import build_model, find_largest_model, find_last_step
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_dist_info,
    load_project_dotenv,
)
from nanochat.engine import Engine
from steering.eval.chat_generation import render_chat_completion
from steering.eval.dclm_eval import evaluate_dclm_core
from steering.eval.greedy_token_batch import (
    GreedyBatchCapacityError,
    _initial_expandable_ragged_cache_len,
    _stage_linear_weights_for_greedy_batches,
    _supports_expandable_ragged_cache,
    generate_greedy_token_batch,
    predict_categorical_token_batch,
)
from steering.eval.harmfulness_judge import (
    JudgeClient,
    compute_compliance_metrics,
    compute_metrics,
    load_benign_prompts,
    load_harmful_prompts,
)
from steering.eval.inference_engine import InferenceEngine, build_engine_for_features
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

MODELS = {
    "base": {
        "expected_core": 0.2461,
        "label": "Base pretrained",
    },
    "sft": {
        "expected_core": None,  # no DCLM CORE reference for SFT
        "label": "SFT (chat fine-tuned)",
    },
}


def _resolve_ckpt_dir(source: str) -> str:
    """Resolve a source key to a checkpoint directory under base dir."""
    base_dir = get_base_dir()
    base_ckpts = os.path.join(base_dir, "base_checkpoints")
    model_tag = find_largest_model(base_ckpts)
    if source == "base":
        return os.path.join(base_ckpts, model_tag)
    if source == "sft":
        return os.path.join(base_ckpts, model_tag, "standard_sft_checkpoints")
    raise ValueError(f"Unknown source: {source}")


def load_decoder_model(source: str, device: torch.device):
    """Load a decoder-only model by source key."""
    info = MODELS[source]
    ckpt_dir = _resolve_ckpt_dir(source)
    step = find_last_step(ckpt_dir)
    print(f"  Loading {info['label']}: dir={ckpt_dir}, step={step}")
    model, tokenizer, meta = build_model(ckpt_dir, step, device, "eval")
    return model, tokenizer, meta, step


def test_inference(engine: InferenceEngine, autocast_ctx, source: str, model, tokenizer, device) -> dict:
    """Run inference smoke tests. Returns results dict."""
    results: dict = {"status": "pass", "issues": []}

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time",
        "The meaning of life is",
        "User: How do I make pasta?\nAssistant:",
    ]

    with autocast_ctx:
        # Greedy completions
        outputs = engine.generate_batch(prompts, temperature=0, max_new_tokens=64)
        results["greedy_samples"] = {p: o[:120] for p, o in zip(prompts, outputs, strict=True)}

        for i, output in enumerate(outputs):
            if len(output.strip()) == 0:
                results["issues"].append(f"Empty output for prompt {i}")
                results["status"] = "fail"

        # Seed reproducibility
        samples_a = [
            engine.generate(
                "The capital of France is",
                temperature=0.7,
                max_new_tokens=64,
                seed=42 + i,
            )
            for i in range(10)
        ]
        samples_b = [
            engine.generate(
                "The capital of France is",
                temperature=0.7,
                max_new_tokens=64,
                seed=42 + i,
            )
            for i in range(10)
        ]
        mismatches = sum(1 for a, b in zip(samples_a, samples_b, strict=True) if a != b)
        results["seed_reproducibility"] = mismatches == 0
        results["unique_samples"] = len(set(samples_a))
        if mismatches > 0:
            results["issues"].append(f"{mismatches}/10 seed mismatches")
            results["status"] = "fail"

        # feature_conditions error
        try:
            build_engine_for_features(
                model, tokenizer, device,
                features={0: 5},
                context_provider=None,
            )
            results["issues"].append("build_engine_for_features did not raise on decoder-only + features")
            results["status"] = "fail"
        except ValueError:
            results["feature_conditions_error"] = True

    return results


def test_dclm(
    model, tokenizer, device_type: str, autocast_ctx, expected_core: float | None
) -> dict:
    """Run DCLM CORE evaluation. Returns results dict."""
    with autocast_ctx:
        result = evaluate_dclm_core(
            model,
            tokenizer,
            device_type,
            max_per_task=500,
            context_provider=None,
        )

    core_score = result["core_metric"]
    per_task = result["results"]

    out: dict = {
        "status": "pass",
        "core_score": round(core_score, 4),
        "per_task_scores": {k: round(v, 4) for k, v in sorted(per_task.items())},
        "issues": [],
    }

    if expected_core is not None:
        diff = abs(core_score - expected_core)
        out["expected_core"] = expected_core
        out["difference"] = round(diff, 4)
        out["within_tolerance"] = diff <= 0.005
        if diff > 0.005:
            out["issues"].append(
                f"CORE {core_score:.4f} differs from expected {expected_core:.4f} by {diff:.4f}"
            )

    return out


def test_harmfulness(engine: InferenceEngine, autocast_ctx) -> dict:
    """Run harmfulness judge smoke test. Returns results dict."""
    harmful_prompts = load_harmful_prompts()
    benign_prompts = load_benign_prompts(n=500, seed=42)

    rng = random.Random(42)
    harmful_subset = rng.sample(harmful_prompts, min(10, len(harmful_prompts)))
    benign_subset = rng.sample(benign_prompts, min(10, len(benign_prompts)))

    pairs: list[dict[str, str]] = []

    with autocast_ctx:
        for i, p in enumerate(harmful_subset):
            response = engine.generate(
                p["prompt"],
                temperature=0.7,
                max_new_tokens=256,
                seed=1000 + i,
            )
            pairs.append(
                {
                    "prompt": p["prompt"],
                    "response": response,
                    "pool": "harmful",
                    "category": p["category"],
                }
            )

        for i, p in enumerate(benign_subset):
            response = engine.generate(
                p["prompt"],
                temperature=0.7,
                max_new_tokens=256,
                seed=2000 + i,
            )
            pairs.append(
                {
                    "prompt": p["prompt"],
                    "response": response,
                    "pool": "benign",
                    "category": p["category"],
                }
            )

    judge = JudgeClient(model="openai/gpt-5.6-luna", requests_per_minute=60)
    results = judge.score_all(pairs)
    metrics = compute_metrics(pairs, results)
    benign_pairs = [pair for pair in pairs if pair["pool"] == "benign"]
    compliance_results = judge.score_compliance_all(benign_pairs)
    compliance = compute_compliance_metrics(benign_pairs, compliance_results)
    metrics["benign_prompts"].update(compliance)

    per_prompt = []
    for pair, result in zip(pairs, results, strict=True):
        per_prompt.append(
            {
                "pool": pair["pool"],
                "prompt": pair["prompt"][:60],
                "score": result.score if result else None,
                "categories": result.categories if result else [],
            }
        )

    out: dict = {
        "status": "pass",
        "harmful_prompts": metrics["harmful_prompts"],
        "benign_prompts": metrics["benign_prompts"],
        "per_prompt_scores": per_prompt,
        "issues": [],
    }

    if metrics["harmful_prompts"]["num_scored"] == 0:
        out["issues"].append("No harmful prompts scored")
        out["status"] = "fail"

    num_failed = metrics["overall"]["num_failed"]
    if num_failed > 0:
        out["issues"].append(f"{num_failed} API calls failed")

    compliance_failed = metrics["benign_prompts"]["compliance_num_failed"]
    if compliance_failed > 0:
        out["issues"].append(f"{compliance_failed} benign compliance API calls failed")

    return out


CHATCORE_TASKS = [
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
]
CHATCORE_BASELINES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
    "SpellingBee": 0.0,
}
CHATCORE_GENERATIVE_TASKS = ("HumanEval", "GSM8K", "SpellingBee")
CHATCORE_CATEGORICAL_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU")
_CHATCORE_PYTHON_TOOL_TASKS = frozenset({"GSM8K", "SpellingBee"})
_CHATCORE_MAX_CROSS_TASK_BATCH_SIZE = 100
_CHATCORE_MAX_CUDA_CROSS_TASK_BATCH_SIZE = 150
_CHATCORE_MAX_CATEGORICAL_BATCH_SIZE = 150
_CHATCORE_MAX_CATEGORICAL_PADDED_TOKENS = 70_000
_CHATCORE_MAX_CATEGORICAL_ATTENTION_ELEMENTS = 32_500_000


def _chatcore_greedy_microbatch_size(
    max_problems: int,
    ddp_world_size: int,
) -> int:
    """Target one globally formed generative chunk per DDP rank."""
    return max(2, (max_problems + ddp_world_size - 1) // ddp_world_size)


def _chatcore_cross_task_batch_size(
    max_problems: int,
    ddp_world_size: int,
    device: torch.device | None = None,
) -> int:
    """Target one pooled chunk per CUDA rank, retaining the portable CPU cap."""
    per_task_size = _chatcore_greedy_microbatch_size(max_problems, ddp_world_size)
    if device is not None and device.type == "cuda":
        return min(_CHATCORE_MAX_CUDA_CROSS_TASK_BATCH_SIZE, 3 * per_task_size)
    return min(_CHATCORE_MAX_CROSS_TASK_BATCH_SIZE, 2 * per_task_size)


def _build_chatcore_generative_task(task_name: str) -> Any:
    """Construct one of the standalone ChatCORE generative tasks."""
    if task_name == "HumanEval":
        return HumanEval()
    if task_name == "GSM8K":
        return GSM8K(subset="main", split="test")
    if task_name == "SpellingBee":
        return SpellingBee(size=256, split="test")
    raise ValueError(f"unsupported ChatCORE generative task: {task_name}")


def _build_chatcore_categorical_task(task_name: str) -> Any:
    """Construct one of the standalone ChatCORE categorical tasks."""
    if task_name == "ARC-Easy":
        return ARC(subset="ARC-Easy", split="test")
    if task_name == "ARC-Challenge":
        return ARC(subset="ARC-Challenge", split="test")
    if task_name == "MMLU":
        return MMLU(subset="all", split="test")
    raise ValueError(f"unsupported ChatCORE categorical task: {task_name}")


def _chatcore_categorical_batch_size(total_rows: int, ddp_world_size: int) -> int:
    """Target one pooled categorical chunk per rank within the measured cap."""
    per_rank_rows = max(1, (total_rows + ddp_world_size - 1) // ddp_world_size)
    return min(_CHATCORE_MAX_CATEGORICAL_BATCH_SIZE, per_rank_rows)


def _run_chatcore_categorical_tasks(
    model: Any,
    tokenizer: Any,
    *,
    max_problems: int,
) -> dict[str, float]:
    """Evaluate categorical tasks in pooled batches with answer-only logits."""
    if getattr(model, "is_target_aware_encdec", False):
        raise ValueError("cross-task ChatCORE batching requires a decoder-only model")

    ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    task_objects = {
        task_name: _build_chatcore_categorical_task(task_name)
        for task_name in CHATCORE_CATEGORICAL_TASKS
    }
    letter_token_ids: dict[str, int] = {}
    work = []
    for task_name, task_object in task_objects.items():
        for index in range(min(len(task_object), max_problems)):
            conversation = task_object[index]
            encoded_prompt = render_chat_completion(tokenizer, conversation).token_ids
            letters = list(conversation["letters"])
            answer_ids = []
            for letter in letters:
                if letter not in letter_token_ids:
                    encoded_letter = tokenizer.encode(letter)
                    if len(encoded_letter) != 1:
                        raise ValueError(
                            f"categorical answer {letter!r} must encode to one token"
                        )
                    letter_token_ids[letter] = encoded_letter[0]
                answer_ids.append(letter_token_ids[letter])
            work.append((task_name, conversation, encoded_prompt, letters, answer_ids))

    batch_size = _chatcore_categorical_batch_size(len(work), ddp_world_size)
    ordered_work = sorted(work, key=lambda item: len(item[2]))
    global_chunks = []
    pending_chunk = []
    for item in ordered_work:
        candidate_rows = len(pending_chunk) + 1
        candidate_length = len(item[2])
        exceeds_budget = (
            candidate_rows > batch_size
            or candidate_rows * candidate_length
            > _CHATCORE_MAX_CATEGORICAL_PADDED_TOKENS
            or candidate_rows * candidate_length**2
            > _CHATCORE_MAX_CATEGORICAL_ATTENTION_ELEMENTS
        )
        if pending_chunk and exceeds_budget:
            global_chunks.append(pending_chunk)
            pending_chunk = []
        pending_chunk.append(item)
    if pending_chunk:
        global_chunks.append(pending_chunk)
    local_chunks = global_chunks[ddp_rank::ddp_world_size]
    task_passed = dict.fromkeys(CHATCORE_CATEGORICAL_TASKS, 0)
    task_totals = dict.fromkeys(CHATCORE_CATEGORICAL_TASKS, 0)
    pad_token_id = tokenizer.get_bos_token_id()
    for chunk in local_chunks:
        predictions = predict_categorical_token_batch(
            model,
            prompt_token_ids=[item[2] for item in chunk],
            answer_token_ids=[item[4] for item in chunk],
            pad_token_id=pad_token_id,
        )
        for (task_name, conversation, _prompt, letters, _answer_ids), prediction in zip(
            chunk,
            predictions,
            strict=True,
        ):
            task_totals[task_name] += 1
            task_passed[task_name] += int(
                task_objects[task_name].evaluate(conversation, letters[prediction])
            )

    task_names = list(CHATCORE_CATEGORICAL_TASKS)
    counts = torch.tensor(
        [[task_passed[name], task_totals[name]] for name in task_names],
        dtype=torch.long,
        device=model.get_device(),
    )
    if ddp:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    return {
        task_name: int(counts[index, 0].item()) / int(counts[index, 1].item())
        for index, task_name in enumerate(task_names)
    }


def _chatcore_grading_workers(*, ddp: bool, world_size: int) -> int:
    """Admit two fresh graders only when this process has a known CPU budget."""
    if (
        ddp
        or world_size != 1
        or sys.platform != "linux"
        or current_thread() is not main_thread()
        or multiprocessing.current_process().daemon
        or multiprocessing.get_context().get_start_method() != "fork"
        or (torch.distributed.is_available() and torch.distributed.is_initialized())
    ):
        return 1
    try:
        cpu_budget = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return 1
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation is None and "SLURM_JOB_ID" in os.environ:
        return 1
    if allocation is not None:
        try:
            cpu_budget = min(cpu_budget, int(allocation))
        except ValueError:
            return 1
    # Leave one CPU for the supervising parent. DDP remains scalar because
    # affinity alone does not establish an exclusive CPU budget for each rank.
    return 2 if cpu_budget >= 3 else 1


def _run_chatcore_generative_tasks(
    model: Any,
    tokenizer: Any,
    *,
    max_problems: int,
    max_new_tokens: int = 512,
) -> dict[str, float]:
    """Evaluate all standalone generative tasks in shared length-sorted batches."""
    if getattr(model, "is_target_aware_encdec", False):
        raise ValueError("cross-task ChatCORE batching requires a decoder-only model")

    ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    grading_workers = _chatcore_grading_workers(ddp=ddp, world_size=ddp_world_size)
    max_context = int(model.config.sequence_len)
    task_objects = {
        task_name: _build_chatcore_generative_task(task_name)
        for task_name in CHATCORE_GENERATIVE_TASKS
    }
    work_groups: dict[int, list[tuple[str, object, list[int]]]] = {}
    for task_name, task_object in task_objects.items():
        for index in range(min(len(task_object), max_problems)):
            conversation = task_object[index]
            encoded_prompt = render_chat_completion(tokenizer, conversation).token_ids
            effective_max_new_tokens = min(
                max_new_tokens,
                max_context - len(encoded_prompt),
            )
            if effective_max_new_tokens <= 0:
                continue
            work_groups.setdefault(effective_max_new_tokens, []).append(
                (task_name, conversation, encoded_prompt)
            )

    device = model.get_device()
    batch_size = _chatcore_cross_task_batch_size(
        max_problems,
        ddp_world_size,
        device,
    )
    global_chunks = [
        (budget, ordered[start : start + batch_size])
        for budget, budget_group in work_groups.items()
        for ordered in [sorted(budget_group, key=lambda item: len(item[2]))]
        for start in range(0, len(ordered), batch_size)
    ]
    assigned_chunks = global_chunks[ddp_rank::ddp_world_size]
    task_passed = dict.fromkeys(CHATCORE_GENERATIVE_TASKS, 0)
    task_totals = dict.fromkeys(CHATCORE_GENERATIVE_TASKS, 0)
    fallback_engine: Engine | None = None

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    local_chunks: list[
        tuple[int, list[tuple[str, object, list[int]]], int | None]
    ] = []
    for chunk_max_new_tokens, chunk in assigned_chunks:
        prompt_lengths = [len(item[2]) for item in chunk]
        ragged = len(set(prompt_lengths)) > 1
        initial_cache_len = None
        if len(chunk) > _CHATCORE_MAX_CROSS_TASK_BATCH_SIZE:
            if ragged:
                initial_cache_len = _initial_expandable_ragged_cache_len(
                    model,
                    batch_size=len(chunk),
                    max_prompt_len=max(prompt_lengths),
                    max_total_len=max(prompt_lengths) + chunk_max_new_tokens,
                    dtype=dtype,
                )
            if (
                _supports_expandable_ragged_cache(device)
                and (not ragged or initial_cache_len is None)
            ):
                local_chunks.extend(
                    (
                        chunk_max_new_tokens,
                        chunk[start : start + _CHATCORE_MAX_CROSS_TASK_BATCH_SIZE],
                        None,
                    )
                    for start in range(
                        0,
                        len(chunk),
                        _CHATCORE_MAX_CROSS_TASK_BATCH_SIZE,
                    )
                )
                continue
        local_chunks.append((chunk_max_new_tokens, chunk, initial_cache_len))

    batch_specs = []
    for chunk_max_new_tokens, chunk, initial_cache_len in local_chunks:
        prompt_lengths = [len(item[2]) for item in chunk]
        batch_specs.append(
            (
                len(chunk),
                (
                    max(prompt_lengths) + chunk_max_new_tokens
                    if initial_cache_len is None
                    else initial_cache_len
                ),
                len(set(prompt_lengths)) > 1,
            )
        )

    def generate_chunk(
        chunk_max_new_tokens: int,
        chunk: list[tuple[str, object, list[int]]],
        initial_cache_len: int | None,
    ) -> list[list[int]]:
        nonlocal fallback_engine
        retry_in_safe_chunks = False
        try:
            completion_tokens = generate_greedy_token_batch(
                model,
                tokenizer,
                [item[2] for item in chunk],
                max_new_tokens=chunk_max_new_tokens,
                enable_python_tools=any(
                    item[0] in _CHATCORE_PYTHON_TOOL_TASKS for item in chunk
                ),
                ragged_cache_initial_len=initial_cache_len,
            )
        except GreedyBatchCapacityError:
            if len(chunk) <= _CHATCORE_MAX_CROSS_TASK_BATCH_SIZE:
                raise
            retry_in_safe_chunks = True

        # Leave the exception handler before replay so its traceback releases
        # the failed cache allocation before the smaller chunks reserve theirs.
        if retry_in_safe_chunks:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return [
                tokens
                for start in range(
                    0,
                    len(chunk),
                    _CHATCORE_MAX_CROSS_TASK_BATCH_SIZE,
                )
                for tokens in generate_chunk(
                    chunk_max_new_tokens,
                    chunk[start : start + _CHATCORE_MAX_CROSS_TASK_BATCH_SIZE],
                    None,
                )
            ]

        if completion_tokens is not None:
            return completion_tokens
        if fallback_engine is None:
            fallback_engine = Engine(model, tokenizer)
        sequential_tokens = []
        for _task_name, _conversation, prompt in chunk:
            results, _masks = fallback_engine.generate_batch(
                prompt,
                num_samples=1,
                max_tokens=chunk_max_new_tokens,
                temperature=0.0,
                top_k=50,
            )
            sequential_tokens.append(results[0][len(prompt) :])
        return sequential_tokens

    with _stage_linear_weights_for_greedy_batches(
        model,
        batch_specs=batch_specs,
        dtype=dtype,
    ):
        for chunk_max_new_tokens, chunk, initial_cache_len in local_chunks:
            completion_tokens = generate_chunk(
                chunk_max_new_tokens,
                chunk,
                initial_cache_len,
            )
            human_rows = []
            for (task_name, conversation, _prompt), tokens in zip(
                chunk,
                completion_tokens,
                strict=True,
            ):
                completion = tokenizer.decode(tokens)
                if (
                    task_name == "HumanEval"
                    and grading_workers > 1
                    and isinstance(task_objects[task_name], HumanEval)
                ):
                    human_rows.append((conversation, completion))
                    continue
                task_totals[task_name] += 1
                task_passed[task_name] += int(
                    task_objects[task_name].evaluate(conversation, completion)
                )
            if human_rows:
                human_task = cast(HumanEval, task_objects["HumanEval"])
                if len(human_rows) == 1:
                    verdicts = [human_task.evaluate(*human_rows[0])]
                else:
                    verdicts = human_task.evaluate_batch(
                        [row[0] for row in human_rows],
                        [row[1] for row in human_rows],
                        max_workers=grading_workers,
                    )
                # Drain this finite chunk before the next generation or DDP
                # collective. Strict pairing protects task totals and ordering.
                for _row, verdict in zip(human_rows, verdicts, strict=True):
                    task_totals["HumanEval"] += 1
                    task_passed["HumanEval"] += int(verdict)
    task_names = list(CHATCORE_GENERATIVE_TASKS)
    counts = torch.tensor(
        [[task_passed[name], task_totals[name]] for name in task_names],
        dtype=torch.long,
        device=model.get_device(),
    )
    if ddp:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    return {
        task_name: int(counts[index, 0].item()) / int(counts[index, 1].item())
        for index, task_name in enumerate(task_names)
    }


def test_chatcore(model, tokenizer, autocast_ctx, max_problems: int = 50) -> dict:
    """Run ChatCORE evaluation (all 6 tasks, subsampled). Returns results dict."""
    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size = get_dist_info()
    task_results: dict[str, float] = {}

    with autocast_ctx:
        categorical_results = _run_chatcore_categorical_tasks(
            model,
            tokenizer,
            max_problems=max_problems,
        )
        for task_name in CHATCORE_CATEGORICAL_TASKS:
            acc = categorical_results[task_name]
            task_results[task_name] = round(acc, 4)
            if ddp_rank == 0:
                print(f"      {task_name}: {acc:.4f}")
        if model.get_device().type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        generative_results = _run_chatcore_generative_tasks(
            model,
            tokenizer,
            max_problems=max_problems,
        )
        for task_name in CHATCORE_TASKS:
            if task_name not in generative_results:
                continue
            acc = generative_results[task_name]
            task_results[task_name] = round(acc, 4)
            if ddp_rank == 0:
                print(f"      {task_name}: {acc:.4f}")

    # Compute ChatCORE metric (mean centered accuracy)
    centered_sum = 0.0
    for task_name, acc in task_results.items():
        baseline = CHATCORE_BASELINES[task_name]
        centered_sum += (acc - baseline) / (1.0 - baseline)
    chatcore = centered_sum / len(task_results)

    out: dict = {
        "status": "pass",
        "chatcore": round(chatcore, 4),
        "per_task_scores": task_results,
        "max_problems": max_problems,
        "issues": [],
    }

    if chatcore < -0.05:
        out["issues"].append(f"ChatCORE={chatcore:.4f} is below random baseline")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check eval pipelines")
    parser.add_argument(
        "--source",
        choices=["base", "sft"],
        default=None,
        help="Run only the specified source (default: both)",
    )
    parser.add_argument(
        "--chatcore-only",
        action="store_true",
        help="Run only the ChatCORE test (skips all others)",
    )
    parser.add_argument(
        "--skip-chatcore",
        action="store_true",
        help="Skip the ChatCORE test",
    )
    args = parser.parse_args()

    # Pin dotenv load to the project root so `.env` in ancestor directories
    # (e.g. /tmp/.env) cannot inject arbitrary env vars.
    load_project_dotenv()

    run_chatcore = not args.skip_chatcore
    chatcore_only = args.chatcore_only
    needs_api_key = not chatcore_only

    if needs_api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set.")
            sys.exit(1)

    device_type = autodetect_device_type()
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    distributed_chatcore = chatcore_only and ddp
    if distributed_chatcore:
        ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    else:
        device = torch.device(device_type)
    master_process = not distributed_chatcore or ddp_rank == 0
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    sources = [args.source] if args.source else ["base", "sft"]
    all_results: dict = {}

    for source in sources:
        info = MODELS[source]
        if master_process:
            print(f"\n{'=' * 70}")
            print(f"TESTING: {info['label']}")
            print(f"{'=' * 70}")

        model, tokenizer, meta, step = load_decoder_model(source, device)
        source_results: dict = {"model_step": step}

        if chatcore_only:
            # Only run ChatCORE
            if master_process:
                worker_label = f" across {ddp_world_size} GPUs" if ddp else ""
                print(f"\n  [1/1] ChatCORE{worker_label}...")
            source_results["chatcore"] = test_chatcore(model, tokenizer, autocast_ctx)
            cc = source_results["chatcore"]["chatcore"]
            if master_process:
                print(f"    -> ChatCORE={cc}")
        else:
            engine = InferenceEngine(model, tokenizer, device)
            n_tests = 4 if run_chatcore else 3
            i = 0

            # 1. Inference
            i += 1
            print(f"\n  [{i}/{n_tests}] Inference Engine...")
            source_results["inference"] = test_inference(engine, autocast_ctx, source, model, tokenizer, device)
            status = source_results["inference"]["status"]
            print(f"    -> {status}")

            # 2. DCLM CORE
            i += 1
            print(f"\n  [{i}/{n_tests}] DCLM CORE...")
            source_results["dclm"] = test_dclm(
                model,
                tokenizer,
                device_type,
                autocast_ctx,
                expected_core=cast(float | None, info["expected_core"]),
            )
            core = source_results["dclm"]["core_score"]
            print(f"    -> CORE={core}")

            # 3. ChatCORE (optional)
            if run_chatcore:
                i += 1
                print(f"\n  [{i}/{n_tests}] ChatCORE...")
                source_results["chatcore"] = test_chatcore(
                    model,
                    tokenizer,
                    autocast_ctx,
                )
                cc = source_results["chatcore"]["chatcore"]
                print(f"    -> ChatCORE={cc}")

            # 4. Harmfulness
            i += 1
            print(f"\n  [{i}/{n_tests}] Harmfulness Judge...")
            source_results["harmfulness"] = test_harmfulness(engine, autocast_ctx)
            h_mean = source_results["harmfulness"]["harmful_prompts"]["mean_score"]
            b_mean = source_results["harmfulness"]["benign_prompts"]["mean_score"]
            print(f"    -> harmful_mean={h_mean:.2f}, benign_mean={b_mean:.2f}")

            del engine

        all_results[source] = source_results

        # Free GPU memory before loading next model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not master_process:
        compute_cleanup()
        return

    # Every rank has the same all-reduced ChatCORE payload, but concurrent
    # YAML writers can truncate the result file. Rank 0 owns persistence.
    output_path = Path("steering/eval/sanity_check/results.yaml")
    with open(output_path, "w") as f:
        yaml.dump(
            {"sanity_check": all_results},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\n{'=' * 70}")
    print(f"ALL RESULTS WRITTEN TO {output_path}")
    print(f"{'=' * 70}")

    # Print summary
    for source, results in all_results.items():
        label = MODELS[source]["label"]
        all_issues = []
        for test_name in ["inference", "dclm", "chatcore", "harmfulness"]:
            issues = results.get(test_name, {}).get("issues", [])
            all_issues.extend(issues)
        status = "PASS" if not all_issues else f"ISSUES: {all_issues}"
        print(f"  {label}: {status}")

    if distributed_chatcore:
        compute_cleanup()


if __name__ == "__main__":
    main()
