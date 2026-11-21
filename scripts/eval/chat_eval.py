"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.eval.chat_eval source=sft task_name=ARC-Easy
torchrun --nproc_per_node=8 -m scripts.eval.chat_eval -- source=sft task_name=ARC-Easy
"""

import json
import os
from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from nanochat.checkpoint_manager import ModelPhase, find_last_step, get_checkpoint_dir, load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_dist_info,
)
from nanochat.engine import Engine
from nanochat.report import get_report
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.chat_eval_result_identity import (
    SelfConditionedChatEvalCheckpointSnapshot,
    TurnChatEvalCheckpointSnapshot,
    build_turn_chat_eval_run_identity,
)
from steering.eval.chat_generation import render_chat_completion
from steering.eval.chat_self_conditioning import (
    ChatEvalMode,
    ChatSelfConditionedSettings,
    parse_chat_eval_mode,
    run_self_conditioned_chat_eval,
)
from steering.eval.compacted_eval_loading import (
    CHAT_EVAL_SELF_CONDITIONED_SURFACE,
    CHAT_EVAL_TURN_SURFACE,
    assert_conditioned_eval_model,
    conditioned_eval_cache_contract_metadata,
    conditioned_eval_conditioning,
)
from steering.eval.context_enumeration import TargetRealRole
from steering.eval.feature_config import (
    ensure_core_eval_has_work,
    parse_feature_configs,
    resolve_next_segment_conditioning_offset,
    validate_inactive_bucket_feature_configs,
)
from steering.eval.greedy_token_batch import generate_greedy_token_batch
from steering.eval.hf_chat_eval import (
    load_hf_chat_eval_runtime,
    validate_chat_eval_selection,
)
from steering.eval.target_conditioning import (
    MultiTargetSegments,
    SelfConditionedTarget,
    SingleTargetSegment,
    TargetConditioning,
    TargetConditioningMode,
)
from steering.eval.target_real_runner import (
    assert_target_real_requires_encdec,
    build_target_real_artifacts_for_example,
    build_target_real_payload,
    context_tokens_from_target_mask,
    parse_target_real_configs_from_cfg,
    require_single_process_turn_target_real,
    run_target_real_evals,
    validate_turn_consumer_config,
)
from steering.eval.turn_context_contract import (
    TurnContextContract,
    resolve_turn_context_contract,
    turn_extraction_tokenizer_id,
)
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
)
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee


def prepare_model_for_chat_eval(
    model,
    target: TargetConditioning,
):
    """Wrap the model for chat eval, handling encoder-decoder conditioning.

    For decoder-only models, returns the model unchanged unless a feature target
    is requested. Encoder-decoder chat evals must use a cache-backed
    ``target_real`` path so prompt tokens are conditioned too.

    Args:
        model: A GPT or EncoderDecoderGPT instance.
        target: Static target segments or the self-conditioning marker.

    Returns:
        Model ready for chat eval (GPT or TargetAwareEncoderDecoderEvalWrapper).

    Raises:
        ValueError: If a decoder-only model is given with a feature target.
    """
    is_encdec = isinstance(model, EncoderDecoderGPT)
    target_is_active = (
        isinstance(target, (MultiTargetSegments, SelfConditionedTarget))
        or (isinstance(target, SingleTargetSegment) and bool(target.features))
    )

    if not is_encdec and target_is_active:
        raise ValueError(
            "Decoder-only models do not support feature conditioning. "
            "Pass SingleTargetSegment(features={}) or use an EncoderDecoderGPT."
        )

    if not is_encdec:
        return model

    if not target_is_active:
        raise ValueError(
            "Encoder-decoder chat_eval requires target_real with a full chat "
            "context cache."
        )

    raise ValueError(
        "Standalone chat_eval no longer supports target-only feature "
        "conditioning. Build a full chat context cache and use "
        "target_real with a full chat context cache so prompt tokens are "
        "conditioned on real full-sequence features."
    )


# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)


def _shard_greedy_work_chunks[ConversationT](
    work_items: list[tuple[ConversationT, list[int], int]],
    *,
    microbatch_size: int,
    rank: int,
    world_size: int,
    allow_mixed_prompt_lengths: bool = False,
) -> list[list[tuple[ConversationT, list[int], int]]]:
    """Form compatible batches globally, then assign whole batches to ranks."""
    groups: dict[
        tuple[int, int],
        list[tuple[ConversationT, list[int], int]],
    ] = {}
    for item in work_items:
        prompt_length_key = 0 if allow_mixed_prompt_lengths else len(item[1])
        groups.setdefault((prompt_length_key, item[2]), []).append(item)
    chunks = [
        group[start : start + microbatch_size]
        for group in groups.values()
        for start in range(0, len(group), microbatch_size)
    ]
    return chunks[rank::world_size]


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
    greedy_microbatch_size: int = 1,
    greedy_python_tools: bool = False,
):

    if (
        not isinstance(greedy_microbatch_size, int)
        or isinstance(greedy_microbatch_size, bool)
        or greedy_microbatch_size < 1
    ):
        raise ValueError("greedy_microbatch_size must be an int >= 1")
    if not isinstance(greedy_python_tools, bool):
        raise TypeError("greedy_python_tools must be a bool")
    if greedy_python_tools and greedy_microbatch_size == 1:
        raise ValueError("greedy_python_tools requires greedy_microbatch_size > 1")
    target_aware = getattr(model, "is_target_aware_encdec", False)
    if greedy_microbatch_size > 1:
        if temperature != 0.0:
            raise ValueError("greedy microbatching requires temperature=0.0")
        if num_samples != 1:
            raise ValueError("greedy microbatching requires num_samples=1")
        if target_aware:
            raise ValueError("greedy microbatching requires a decoder-only model")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = (
        len(task_object)
        if max_problems is None
        else min(len(task_object), max_problems)
    )

    max_context = getattr(getattr(model, "config", None), "sequence_len", None)
    if not isinstance(max_context, Integral):
        max_context = None
    work_items: list[tuple[object, list[int], int]] = []
    problem_indices = (
        range(num_problems)
        if greedy_microbatch_size > 1
        else range(ddp_rank, num_problems, ddp_world_size)
    )
    for i in problem_indices:
        conversation = task_object[i]
        encoded_prompt = render_chat_completion(tokenizer, conversation).token_ids
        effective_max_new_tokens = int(max_new_tokens)
        if max_context is not None:
            remaining = int(max_context) - len(encoded_prompt)
            if remaining <= 0:
                logger.warning(
                    f"Skipping generative example {i}: prompt length "
                    f"{len(encoded_prompt)} exceeds context window {int(max_context)}"
                )
                continue
            if remaining < effective_max_new_tokens:
                effective_max_new_tokens = remaining
        work_items.append((conversation, encoded_prompt, effective_max_new_tokens))

    def generate_exact(
        conversation: object,
        encoded_prompt: list[int],
        effective_max_new_tokens: int,
    ) -> list[str]:
        ctx = (
            model.conditioning_context(context_token_ids=[encoded_prompt])
            if target_aware
            else nullcontext()
        )
        with ctx:
            results, _masks = engine.generate_batch(
                encoded_prompt,
                num_samples=num_samples,
                max_tokens=effective_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        prefix_length = len(encoded_prompt)
        return [
            tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results
        ]

    num_passed, total = 0, 0

    def record_result(conversation: object, completions: list[str]) -> None:
        nonlocal num_passed, total
        outcomes = [
            task_object.evaluate(conversation, completion) for completion in completions
        ]
        passed = any(outcomes)
        total += 1
        num_passed += int(passed)
        print(
            f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    if greedy_microbatch_size == 1:
        for conversation, encoded_prompt, effective_max_new_tokens in work_items:
            record_result(
                conversation,
                generate_exact(
                    conversation,
                    encoded_prompt,
                    effective_max_new_tokens,
                ),
            )
    else:
        chunks = _shard_greedy_work_chunks(
            work_items,
            microbatch_size=greedy_microbatch_size,
            rank=ddp_rank,
            world_size=ddp_world_size,
            allow_mixed_prompt_lengths=device.type == "cuda",
        )
        for chunk in chunks:
            completion_tokens = generate_greedy_token_batch(
                model,
                tokenizer,
                [item[1] for item in chunk],
                max_new_tokens=chunk[0][2],
                enable_python_tools=greedy_python_tools,
            )
            if completion_tokens is None:
                logger.info(
                    "Greedy token batch emitted <python_start>; "
                    "restarting the chunk through the exact Engine path"
                )
                for conversation, encoded_prompt, effective_max_new_tokens in chunk:
                    record_result(
                        conversation,
                        generate_exact(
                            conversation,
                            encoded_prompt,
                            effective_max_new_tokens,
                        ),
                    )
                continue
            for (conversation, _encoded_prompt, _budget), tokens in zip(
                chunk, completion_tokens, strict=True
            ):
                record_result(conversation, [tokenizer.decode(tokens)])

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    logger.info("=" * 50)
    if total == 0:
        logger.warning("No generative examples fit the model context window")
        return 0.0
    logger.info(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")

    # Return the accuracy
    return num_passed / total


# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = (
        tokenizer.get_bos_token_id()
    )  # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = (
        len(task_object)
        if max_problems is None
        else min(len(task_object), max_problems)
    )

    def ceil_div(x, y):
        return -(-x // y)

    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {}  # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [
            render_chat_completion(tokenizer, conversation).token_ids
            for conversation in conversations
        ]  # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [
            len(ids) - 1 for ids in prompt_ids
        ]  # where the last token is (and the predicted answer)
        padded_prompt_ids = [
            ids + [bos] * (max_length - len(ids)) for ids in prompt_ids
        ]
        context_token_ids = prompt_ids
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)
        target_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
        for row, answer_pos in enumerate(answer_time_positions):
            target_mask[row, answer_pos] = True

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            if getattr(model, "is_target_aware_encdec", False) is True:
                ctx = model.conditioning_context(
                    target_mask=target_mask,
                    context_token_ids=context_token_ids,
                )
            else:
                ctx = nullcontext()
            with ctx:
                logits = model(prompt_ids)  # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, (
                        "Each letter must be a single token"
                    )
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    logger.info(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------


def _create_chat_task(task_name):
    return {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]()


def run_chat_eval(
    task_name,
    model,
    tokenizer,
    engine,
    batch_size=1,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
    greedy_microbatch_size: int = 1,
    greedy_python_tools: bool = False,
):
    task_object = _create_chat_task(task_name)
    # Run the evaluation
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
            greedy_microbatch_size=greedy_microbatch_size,
            greedy_python_tools=greedy_python_tools,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object, tokenizer, model, batch_size, max_problems=max_problems
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


# -----------------------------------------------------------------------------

BASELINE_ACCURACIES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
    "SpellingBee": 0.0,
}

ALL_CHAT_TASKS = [
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
]


def compute_chatcore_metric(results: Mapping[str, float]) -> float | None:
    """Return the existing baseline-centered mean only for a complete suite."""
    if not all(task in results for task in ALL_CHAT_TASKS):
        return None
    centered_sum = 0.0
    for task_name, acc in results.items():
        baseline_acc = BASELINE_ACCURACIES.get(task_name)
        if baseline_acc is None:
            logger.warning(f"No baseline accuracy for task {task_name!r}. Using 0.0.")
            baseline_acc = 0.0
        centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
        centered_sum += centered_acc
    return centered_sum / len(results)


def _target_real_output_dir(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target_real_output_dir must be null or a non-blank path")
    return value


def _chat_eval_result(name, results):
    entry = {"name": name, "results": results}
    metric = compute_chatcore_metric(results)
    if metric is not None:
        entry["chatcore_metric"] = metric
    return entry


def run_multi_config_chat_eval(
    model,
    tokenizer,
    configs,
    task_names,
    batch_size,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems,
):
    """Run chat evaluation across multiple feature-conditioning configs.

    For each config, wraps the model (if encoder-decoder), creates an Engine,
    runs all tasks, and computes per-config ChatCORE metric.

    Args:
        model: A GPT or EncoderDecoderGPT instance.
        tokenizer: Tokenizer with render_for_completion support.
        configs: List of FeatureConfig to evaluate.
        task_names: List of task name strings.
        batch_size: Batch size for categorical eval.
        num_samples: Number of samples for generative eval.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        max_problems: Cap on problems per task (None = all).

    Returns:
        List of dicts, one per config, each with:
            name, results, chatcore_metric (if all tasks evaluated).
    """
    config_results = []

    for cfg in configs:
        logger.info(f"Evaluating config: {cfg.name}")

        target = SingleTargetSegment(features=dict(cfg.features))
        eval_model = prepare_model_for_chat_eval(
            model,
            target,
        )
        engine = Engine(eval_model, tokenizer)

        results = {}
        for task_name in task_names:
            acc = run_chat_eval(
                task_name,
                eval_model,
                tokenizer,
                engine,
                batch_size=batch_size,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                max_problems=max_problems,
            )
            results[task_name] = acc
            logger.info(f"  {task_name} accuracy: {100 * acc:.2f}%")

        config_results.append(_chat_eval_result(cfg.name, results))

    return config_results


# -----------------------------------------------------------------------------
class _SingleExampleTaskAdapter:
    """Wrap one task example so the run_categorical_eval / run_generative_eval
    helpers can drive it as a length-1 task.

    The underlying ``task_object`` is consulted for ``eval_type`` and
    ``evaluate``; ``__getitem__(0)`` returns the wrapped conversation and
    ``__len__`` is always 1.
    """

    def __init__(self, task_object, conversation):
        self._task = task_object
        self._conv = conversation
        self.eval_type = task_object.eval_type

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError(idx)
        return self._conv

    def evaluate(self, conversation, prediction):
        return self._task.evaluate(conversation, prediction)


def _chat_task(task_name):
    return {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]()


def _turn_problem_range(task_name, task_size, problem_start, max_problems) -> range:
    start = 0 if problem_start is None else problem_start
    if type(start) is not int or start < 0:
        raise ValueError("problem_start must be null or a nonnegative integer")
    if max_problems is not None and (type(max_problems) is not int or max_problems <= 0):
        raise ValueError("max_problems must be null or a positive integer")
    stop = task_size if max_problems is None else start + max_problems
    if start >= task_size or stop > task_size:
        raise ValueError(f"{task_name}: problem range [{start}, {stop}) exceeds task size {task_size}")
    return range(start, stop)


@dataclass(frozen=True)
class TurnChatEvalProblems:
    """The turn ChatCORE problem selection, shared by preflight and evaluation.

    Built once per run so the cache preflight and the per-config loop cannot
    disagree about which problems a turn run covers.
    """

    contract: TurnContextContract
    tasks: Mapping[str, Any]
    ranges: Mapping[str, range]

    @classmethod
    def select(cls, contract, task_names, *, problem_start, max_problems) -> "TurnChatEvalProblems":
        # Same constructor and canonical order as the cache builder, including
        # each task's own deterministic dataset shuffle.
        if len(task_names) != len(set(task_names)):
            raise ValueError("duplicate ChatCORE task names")
        tasks = {name: _chat_task(name) for name in task_names}
        ranges = {name: _turn_problem_range(name, len(task), problem_start, max_problems)
                  for name, task in tasks.items()}
        return cls(contract, tasks, ranges)


def _precheck_turn_problems(problems: TurnChatEvalProblems, provider, *, tokenizer) -> None:
    """Check all selected tasks' exact full/prefix rows before any model forward."""
    for name, indices in problems.ranges.items():
        task = problems.tasks[name]
        for index in indices:
            conversation = task[index]
            full_ids, mask = tokenizer.render_conversation(conversation)
            prefix = context_tokens_from_target_mask(list(full_ids), list(mask))
            if prefix != render_chat_completion(tokenizer, conversation).token_ids:
                raise ValueError("turn target-real full row prefix disagrees with independently rendered completion prompt")
            for role, ids in ((TargetRealRole.FULL, full_ids), (TargetRealRole.CONTEXT_PREFIX, prefix)):
                if provider.get(ids) is None:
                    raise ValueError(
                        f"{name}: problem {index} missing {role.value} cache row before evaluation; "
                        "rebuild the turn cache with chat_max_problems covering every requested problem"
                    )


def _chat_eval_target_real_per_config(
    *,
    model,
    tokenizer,
    trc,
    provider,
    task_names: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_problems: int | None,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
    turn: TurnChatEvalProblems | None = None,
) -> dict:
    """Run chat eval for one TargetRealConfig — real per-example loop.

    Fixed runs evaluate each task's leading ``max_problems`` examples. A turn
    run evaluates the preflighted selection in ``turn`` (whose tasks must be
    ``task_names``) and records per-task range/counts for shard merging.

    For each task and each selected example:
      1. Render the FULL conversation -> ``(full_token_ids, mask)``
      2. Derive ``target_char_start/end`` from ``mask==1`` tokens
      3. Look up the target cache row via ``provider.get(full_token_ids)``
         (hard-fail on miss)
      4. Build a target via ``build_target_for_real_config(...)``
      5. Build a fresh wrapper via ``build_wrapper_for_features(...)`` with
         prompt/context features loaded from the exact prefix cache row.
      6. Run ``run_categorical_eval`` or ``run_generative_eval`` on a
         single-example adapter, accumulate accuracy

    Returns the flat eval body. The standalone chat_eval orchestrator wraps
    this with ``build_target_real_payload`` before writing the SuiteResult
    YAML; training_progression consumes the flat body directly.
    """
    if turn is not None and list(turn.tasks) != list(task_names):
        raise ValueError("turn problem selection does not cover the requested tasks")
    turn_contract = None if turn is None else turn.contract
    results: dict[str, float] = {}
    task_details: dict[str, dict[str, int]] = {}
    for task_name in task_names:
        if turn is None:
            task_object = _chat_task(task_name)
            indices = range(len(task_object) if max_problems is None else min(len(task_object), max_problems))
        else:
            task_object, indices = turn.tasks[task_name], turn.ranges[task_name]
        num_passed = 0
        for ex_idx in indices:
            conversation = task_object[ex_idx]
            # Render the FULL conversation (prompt + ground-truth assistant
            # response) for cache lookup and target derivation.
            full_token_ids, mask = tokenizer.render_conversation(conversation)
            full_token_ids = list(full_token_ids)
            mask = list(mask)
            context_token_ids = context_tokens_from_target_mask(
                full_token_ids,
                mask,
            )
            if turn_contract is not None and context_token_ids != render_chat_completion(tokenizer, conversation).token_ids:
                raise ValueError("turn target-real full row prefix disagrees with independently rendered completion prompt")
            artifacts = build_target_real_artifacts_for_example(
                model=model,
                tokenizer=tokenizer,
                trc=trc,
                provider=provider,
                full_token_ids=full_token_ids,
                mask=mask,
                context_token_ids=context_token_ids,
                target_mode=target_mode,
                turn_contract=turn_contract,
            )
            wrapper = artifacts.wrapper
            engine = Engine(wrapper, tokenizer)
            adapter = _SingleExampleTaskAdapter(task_object, conversation)
            if task_object.eval_type == "categorical":
                acc = run_categorical_eval(
                    adapter, tokenizer, wrapper,
                    batch_size=1, max_problems=1,
                )
            elif task_object.eval_type == "generative":
                acc = run_generative_eval(
                    adapter, tokenizer, wrapper, engine,
                    num_samples=num_samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    max_problems=1,
                )
            else:
                raise ValueError(
                    f"Unsupported task evaluation type: {task_object.eval_type}"
                )
            num_passed += int(acc * 1)
        total = len(indices)
        task_acc = (num_passed / total) if total > 0 else 0.0
        results[task_name] = task_acc
        if turn is not None:
            task_details[task_name] = {
                "problem_start": indices.start, "max_problems": total,
                "task_size": len(task_object), "num_passed": num_passed, "total": total,
            }
        logger.info(
            f"target_real['{trc.name}'] {task_name}: {100 * task_acc:.2f}% "
            f"({num_passed}/{total})"
        )
    body: dict = {"results": results}
    if turn is not None:
        body["task_details"] = task_details
    # ChatCORE metric (only when all 6 tasks evaluated).
    metric = compute_chatcore_metric(results)
    if metric is not None:
        body["chatcore_metric"] = metric
    return body


# -----------------------------------------------------------------------------


def _run_hf_chat_eval(cfg: DictConfig) -> None:
    """Own HF process lifecycle and publish a result only after complete coverage."""
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("HF ChatCORE supports CPU or CUDA")
    if get_dist_info()[0] and device_type != "cuda":
        raise ValueError("distributed HF ChatCORE CLI requires CUDA")
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    try:
        _ddp, rank, _local_rank, world, device = compute_init(device_type)
        with rank_synchronized_stage("HF output directory", device=device):
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=False)
        with ExitStack() as stack:
            with rank_synchronized_stage("HF evidence file", device=device):
                records = stack.enter_context(
                    (output_dir / f"records.rank{rank}.jsonl").open("x", encoding="utf-8")
                )
            with rank_synchronized_stage("HF model loading", device=device):
                runtime = load_hf_chat_eval_runtime(cfg, device, records)
            results = {}
            for task_name in ALL_CHAT_TASKS if cfg.task_name is None else cfg.task_name.split("|"):
                with rank_synchronized_stage(f"HF {task_name} dataset", device=device):
                    task = _create_chat_task(task_name)
                results[task_name] = runtime.evaluate(
                    task,
                    task_name,
                    batch_size=cfg.batch_size,
                    num_samples=cfg.num_samples,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    max_problems=cfg.max_problems,
                )
            with rank_synchronized_stage("HF evidence close", device=device):
                stack.close()
        with rank_synchronized_stage("HF result publication", device=device):
            if rank == 0:
                entry = _chat_eval_result("unconditional", results)
                entry.update(
                    model=runtime.identity,
                    task_counts=runtime.task_counts,
                    evaluation_config=OmegaConf.to_container(cfg, resolve=True),
                    evidence_files=[f"records.rank{i}.jsonl" for i in range(world)],
                )
                get_report(checkpoint_dir=str(output_dir)).log(
                    section=f"Chat evaluation {cfg.hf_path}@{cfg.hf_revision}",
                    data=[entry],
                )
                temporary = output_dir / "unconditional.yaml.tmp"
                with temporary.open("x", encoding="utf-8") as stream:
                    yaml.safe_dump(entry, stream, sort_keys=False, allow_unicode=True)
                temporary.replace(output_dir / "unconditional.yaml")
                logger.info(f"HF ChatCORE results written to {output_dir / 'unconditional.yaml'}")
    finally:
        compute_cleanup()


def _resolve_chat_turn_contract(
    cfg, model, tokenizer, meta, *, checkpoint_contract: CheckpointFeatureContract,
    rank: int, world_size: int,
) -> TurnContextContract:
    """Admit only checkpoint-bound static targets; optional cadence has no routing role."""
    require_single_process_turn_target_real(rank=rank, world_size=world_size)
    if not cfg.get("target_real_configs") or cfg.get("feature_configs"):
        raise ValueError("chat_eval turn_delimited_fixed_tokens requires target_real_configs and empty feature_configs")
    if type(cfg.get("num_samples")) is not int or cfg.num_samples != 1 or cfg.get("temperature") != 0.0:
        raise ValueError("turn target-real requires num_samples=1 and temperature=0")
    if cfg.get("greedy_microbatch_size", 1) != 1 or cfg.get("greedy_python_tools", False):
        raise ValueError("turn target-real requires the scalar Engine path; greedy microbatching is unsupported")
    # CURRENT/NEXT timing is checkpoint-derived; an explicit value is an equality
    # assertion like every other turn setting, never an override that shifts
    # the prompt schedule off training's.
    derived = resolve_next_segment_conditioning_offset({"next_segment_conditioning_offset": "auto"}, meta)
    if resolve_next_segment_conditioning_offset(cfg, meta) != derived:
        raise ValueError(
            "next_segment_conditioning_offset disagrees with checkpoint metadata "
            f"(checkpoint timing: {'next' if derived else 'current'}); use auto"
        )
    if TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY in meta:
        resolve_turn_self_cond_segment_size_rustbpe(meta, checkpoint_dir=str(meta.get("checkpoint_dir", "<checkpoint>")))
    lineage = OnlineSAELineage.from_checkpoint_meta(meta)
    tokenizer_id = turn_extraction_tokenizer_id(lineage.model, cfg.get("extraction_tokenizer_id"))
    extraction_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    contract = resolve_turn_context_contract(meta, tokenizer=tokenizer, extraction_tokenizer=extraction_tokenizer,
                                            statistics_path=cfg.get("stats_path"), checkpoint_contract=checkpoint_contract)
    validate_turn_consumer_config(cfg, contract)
    if model.dec_config.sequence_len != contract.model_max_seq_len:
        raise ValueError("model sequence_len must equal model_max_seq_len and checkpoint metadata")
    for field in ("n_sae_features", "num_buckets"):
        if getattr(model.encoder.config, field) != getattr(contract.lineage, field):
            raise ValueError(f"model {field} disagrees with turn checkpoint metadata")
    if model.feature_vocab != contract.feature_vocab:
        raise ValueError("model feature vocabulary disagrees with turn checkpoint metadata")
    cond = cfg.get("conditioning") or {}
    cache_identity_from_config(cond.get("identity"), cond, turn_contract=contract)
    return contract


@hydra.main(config_path="../../configs/eval", config_name="chat_eval", version_base=None)
def main(cfg: DictConfig):
    evaluation_mode = parse_chat_eval_mode(cfg)
    self_conditioned_settings = (
        ChatSelfConditionedSettings.parse(cfg) if evaluation_mode is ChatEvalMode.SELF_CONDITIONED else None)
    if validate_chat_eval_selection(cfg, ALL_CHAT_TASKS):
        _run_hf_chat_eval(cfg)
        return
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if self_conditioned_settings is not None and (ddp or ddp_rank != 0 or ddp_world_size != 1):
        raise ValueError("self_conditioned ChatCORE requires one process")
    ptdtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # Pin the receipt to the physical directory; keep the configured path for
    # loading and outputs so fixed runs still write under the operator's tree.
    configured_dir = get_checkpoint_dir(cfg.source, model_tag=cfg.model_tag, step=cfg.step)
    checkpoint_dir = str(Path(configured_dir).resolve(strict=True))
    step = cfg.step if cfg.step is not None else find_last_step(checkpoint_dir)
    checkpoint_path = Path(checkpoint_dir) / f"meta_{step:06d}.json"
    checkpoint_payload = checkpoint_path.read_bytes()
    checkpoint_meta = json.loads(checkpoint_payload)
    mode = resolve_checkpoint_sft_segment_mode(checkpoint_meta, checkpoint_dir=checkpoint_dir)
    if self_conditioned_settings is None and mode is not SegmentMode.turn_delimited_fixed_tokens and cfg.get("problem_start") is not None:
        raise ValueError("problem_start is only supported on the turn path")
    eval_conditioning = (
        conditioned_eval_conditioning(CHAT_EVAL_SELF_CONDITIONED_SURFACE)
        if self_conditioned_settings is not None else
        conditioned_eval_conditioning(CHAT_EVAL_TURN_SURFACE)
        if mode is SegmentMode.turn_delimited_fixed_tokens
        else EvalConditioning.deferred("chat_eval")
    )
    checkpoint_snapshot = (
        SelfConditionedChatEvalCheckpointSnapshot.capture(checkpoint_path, checkpoint_payload, step)
        if self_conditioned_settings is not None else
        TurnChatEvalCheckpointSnapshot.capture(checkpoint_path, checkpoint_payload, step)
        if mode is SegmentMode.turn_delimited_fixed_tokens else None
    )
    model, tokenizer, meta = load_model(
        cfg.source,
        device,
        phase=ModelPhase.EVAL,
        checkpoints_dir=str(Path(configured_dir).parent),
        model_tag=Path(configured_dir).name,
        step=step,
        eval_conditioning=eval_conditioning,
    )
    if resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=checkpoint_dir) is not mode:
        raise ValueError("chat_eval checkpoint segment_mode changed during model load")
    checkpoint_contract = None
    if self_conditioned_settings is not None:
        # This early return preserves the fixed/static admission path below.
        checkpoint_contract = assert_conditioned_eval_model(model, meta, checkpoint_dir=checkpoint_dir,
            eval_name=CHAT_EVAL_SELF_CONDITIONED_SURFACE)
        if checkpoint_snapshot is None:
            raise ValueError("self_conditioned requires a checkpoint snapshot")
        checkpoint_identity = checkpoint_snapshot.bind_loaded(meta)
        try:
            # The model owns inference precision, including packed encoder state.
            # Disable inherited autocast as well as this entrypoint's CUDA scope.
            with torch.amp.autocast(device_type=device_type, enabled=False):
                run_self_conditioned_chat_eval(cfg, settings=self_conditioned_settings, model=model,
                    tokenizer=tokenizer, meta=meta, mode=mode, feature_contract=checkpoint_contract,
                    checkpoint_snapshot=checkpoint_snapshot, checkpoint_identity=checkpoint_identity,
                    device_type=device_type, task_factory=_chat_task, all_task_names=ALL_CHAT_TASKS,
                    task_names=ALL_CHAT_TASKS if cfg.task_name is None else cfg.task_name.split("|"),
                    problem_range=_turn_problem_range, metric=compute_chatcore_metric)
        finally:
            compute_cleanup()
        return
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        checkpoint_contract = assert_conditioned_eval_model(
            model, meta, checkpoint_dir=checkpoint_dir, eval_name=CHAT_EVAL_TURN_SURFACE,
        )
    if mode is not SegmentMode.turn_delimited_fixed_tokens:
        reject_feature_clusters_metadata(meta, "chat_eval checkpoint")
    is_encdec = isinstance(model, EncoderDecoderGPT)
    turn_contract = None
    if is_encdec:
        if cfg.get("segment_mode") is not None and SegmentMode(cfg.segment_mode) is not mode:
            raise ValueError("chat_eval segment_mode disagrees with checkpoint metadata")
        if mode is SegmentMode.turn_delimited_fixed_tokens:
            if checkpoint_contract is None:
                raise ValueError("turn chat_eval requires a loaded checkpoint feature contract")
            turn_contract = _resolve_chat_turn_contract(
                cfg, model, tokenizer, meta, checkpoint_contract=checkpoint_contract,
                rank=ddp_rank, world_size=ddp_world_size,
            )
    elif mode is SegmentMode.turn_delimited_fixed_tokens or (
        cfg.get("segment_mode") is not None and SegmentMode(cfg.segment_mode) is SegmentMode.turn_delimited_fixed_tokens
    ):
        raise ValueError("turn target-real requires an encoder-decoder model")

    # Unsupported model/config combinations retain their admission diagnostics;
    # every admitted turn run binds the snapshot before task or cache evaluation.
    checkpoint_identity = checkpoint_snapshot.bind_loaded(meta) if checkpoint_snapshot is not None else None

    # Parse task names and feature configs
    task_names = ALL_CHAT_TASKS if cfg.task_name is None else cfg.task_name.split("|")
    configs = parse_feature_configs(cfg.feature_configs)
    target_mode = TargetConditioningMode.from_next_segment_offset(
        resolve_next_segment_conditioning_offset(cfg, meta)
    )

    # Suite name picks the cache subdirectory for null-cache_dir auto-resolution
    # (Task 2). chat_eval consumes target caches built from full chat conversations.
    target_real_configs = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", turn_contract=turn_contract
    )
    assert_target_real_requires_encdec(target_real_configs, is_encdec=is_encdec)
    # Empty feature_configs must not silently run zero passes: decoder-only models
    # fall back to one unconditioned baseline; encoder-decoder runs require either
    # static configs or target_real work (otherwise hard error).
    configs = ensure_core_eval_has_work(
        configs,
        is_encdec=is_encdec,
        has_target_real=bool(target_real_configs),
        surface="chat_eval",
    )
    model_type = "encoder_decoder" if is_encdec else "decoder_only"
    logger.info(f"Model type: {model_type}, source: {cfg.source}")
    logger.info(f"Feature configs: {[c.name for c in configs]}")
    if is_encdec:
        validate_inactive_bucket_feature_configs(
            configs,
            num_buckets=model.encoder.config.num_buckets,
            checkpoint_meta=meta,
        )
    # Run multi-config evaluation
    with autocast_ctx:
        config_results = run_multi_config_chat_eval(
            model,
            tokenizer,
            configs,
            task_names=task_names,
            batch_size=cfg.batch_size,
            num_samples=cfg.num_samples,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            max_problems=cfg.max_problems,
        )

    # Write per-config result files and log to report
    output_dir = os.path.join(meta["checkpoint_dir"], "chat_eval")
    for entry in config_results:
        config_name = entry["name"]
        section_name = f"Chat evaluation {cfg.source}"
        if len(configs) > 1:
            section_name += f" ({config_name})"

        # Determine the kind discriminator for this config entry so report
        # consumers can filter by variant without modifying nanochat/report.py
        # (read-only per CODEOWNERS).
        entry_kind = "static_yaml"

        report_data = [
            OmegaConf.to_container(cfg, resolve=True),
            entry["results"],
            {"kind": entry_kind},
        ]
        if "chatcore_metric" in entry:
            report_data.append({"ChatCORE metric": entry["chatcore_metric"]})

        get_report(checkpoint_dir=meta["checkpoint_dir"]).log(
            section=section_name,
            data=report_data,
        )

        # Write per-config YAML result file
        if ddp_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{config_name}.yaml")
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(entry, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Results written to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Chat Evaluation Summary")
    logger.info("=" * 60)
    for entry in config_results:
        config_label = entry["name"]
        for task_name, acc in entry["results"].items():
            logger.info(f"  [{config_label}] {task_name}: {100 * acc:.2f}%")
        if "chatcore_metric" in entry:
            logger.info(f"  [{config_label}] ChatCORE: {entry['chatcore_metric']:.4f}")

    # PR-C: per-target-real-config eval loop. Orchestrated by the shared
    # ``run_target_real_evals`` helper — rank-0 barrier, per-config provider
    # construction, output dir creation, and yaml writing all live there.
    if target_real_configs and ddp_rank == 0:
        turn_problems = None if turn_contract is None else TurnChatEvalProblems.select(
            turn_contract, task_names,
            problem_start=cfg.get("problem_start"), max_problems=cfg.max_problems,
        )

        def evaluate_target_real_config(trc, provider):
            body = _chat_eval_target_real_per_config(
                model=model, tokenizer=tokenizer, trc=trc, provider=provider,
                task_names=task_names, num_samples=cfg.num_samples,
                max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                top_k=cfg.top_k, max_problems=cfg.max_problems, target_mode=target_mode,
                turn=turn_problems,
            )
            if turn_contract is not None:
                if checkpoint_snapshot is None or checkpoint_identity is None:
                    raise ValueError("turn ChatCORE receipt requires a loaded checkpoint snapshot")
                checkpoint_snapshot.assert_stable()
                body["run_identity"] = build_turn_chat_eval_run_identity(
                    cfg, checkpoint_identity=checkpoint_identity, model=model,
                    turn_contract=turn_contract, device_type=device_type, provider=provider,
                )
            return build_target_real_payload(trc, body, target_mode=target_mode)

        run_target_real_evals(
            target_real_configs=target_real_configs,
            cfg=cfg,
            output_dir_base=(
                _target_real_output_dir(cfg.target_real_output_dir)
                if cfg.get("target_real_output_dir") is not None
                else os.path.join(meta["checkpoint_dir"], "chat_eval")
            ),
            ddp_rank=ddp_rank,
            per_config_eval_fn=evaluate_target_real_config,
            checkpoint_dir=meta["checkpoint_dir"],
            report_section_prefix=f"Chat evaluation {cfg.source}",
            expected_contract_metadata=(
                None if checkpoint_contract is None
                else conditioned_eval_cache_contract_metadata(checkpoint_contract)
            ),
            turn_contract=turn_contract,
            preflight_fn=(
                None if turn_problems is None
                else partial(_precheck_turn_problems, turn_problems, tokenizer=tokenizer)
            ),
        )

    compute_cleanup()


if __name__ == "__main__":
    main()
