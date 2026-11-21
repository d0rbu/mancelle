"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.distributed as dist
from jinja2 import Template
from loguru import logger

# -----------------------------------------------------------------------------
# Task type enum and metadata


class TaskType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    SCHEMA = "schema"
    LANGUAGE_MODELING = "language_modeling"


@dataclass(frozen=True)
class TaskMeta:
    task_type: TaskType
    num_fewshot: int
    continuation_delimiter: str


# -----------------------------------------------------------------------------
# Prompt rendering utilities


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [template.render(choice=choice, **context) for choice in item["choices"]]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [
        template.render(context=context_option, **context)
        for context_option in item["context_options"]
    ]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], (
        "prompt without is supposed to be a prefix of prompt with"
    )
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids, target_mask=None, context_token_ids=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    if getattr(model, "is_target_aware_encdec", False) is True:
        ctx = model.conditioning_context(
            target_mask=target_mask,
            context_token_ids=context_token_ids,
        )
    else:
        ctx = nullcontext()

    with ctx:
        outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float("nan")
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


def _sample_fewshot_indices(
    data_len: int, current_idx: int, num_fewshot: int, seed: int
) -> list[int]:
    """Sample few-shot indices, clamped to the available population.

    Returns up to ``num_fewshot`` distinct indices in ``range(data_len)``,
    never including ``current_idx``. If fewer indices are available than
    ``num_fewshot`` (e.g. tiny smoke-test ``max_per_task``), this draws all
    of them without raising.
    """
    if num_fewshot <= 0 or data_len <= 1:
        return []
    rng = random.Random(seed)
    available_indices = [i for i in range(data_len) if i != current_idx]
    k = min(num_fewshot, len(available_indices))
    return rng.sample(available_indices, k)


def core_model_max_seq_len(model) -> int | None:
    """Best-effort CORE context length for decoder and encoder-decoder wrappers."""

    max_tokens = getattr(model, "max_seq_len", None)
    if max_tokens is None:
        config = getattr(model, "config", None)
        max_tokens = getattr(config, "sequence_len", None)
    if max_tokens is None:
        dec_config = getattr(model, "dec_config", None)
        max_tokens = getattr(dec_config, "sequence_len", None)
    if max_tokens is None:
        return None
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError(f"model.max_seq_len must be positive when set; got {max_tokens}")
    return max_tokens


def should_skip_oversized_core_example(
    tokens,
    *,
    max_tokens: int | None,
    label: str,
    idx: int,
) -> bool:
    """Return True when any rendered row exceeds the CORE context contract.

    Multiple-choice/schema examples render several rows. Dropping only the
    oversized row would change the answer set, so the whole example is skipped
    when any row is too long.
    """

    if max_tokens is None:
        return False
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError(f"max_seq_len/max_tokens must be positive when set; got {max_tokens}")
    max_row_len = max((len(t) for t in tokens), default=0)
    if max_row_len <= max_tokens:
        return False
    severity = (
        logger.error
        if max_row_len > int(max_tokens * 1.2)
        else logger.warning
        if max_row_len > int(max_tokens * 1.1)
        else logger.info
    )
    severity(
        f"core: {label} example {idx} has a row of {max_row_len} "
        f"tokens > model_max_seq_len ({max_tokens}); skipping the entire example"
    )
    return True


def _crop_core_rows_to_max_seq_len(tokens, start_idxs, end_idxs, max_tokens: int):
    """Legacy opt-out path for callers that explicitly request cropping."""

    new_tokens, new_start_idxs, new_end_idxs = [], [], []
    cropped_count = 0
    for t, s, e in zip(tokens, start_idxs, end_idxs, strict=True):
        if len(t) > max_tokens:
            num_to_crop = len(t) - max_tokens
            new_tokens.append(t[-max_tokens:])
            new_start_idxs.append(s - num_to_crop)
            new_end_idxs.append(e - num_to_crop)
            assert s - num_to_crop >= 0, "cropping removed target start"
            assert e - num_to_crop >= 0, "cropping removed target end"
            cropped_count += 1
        else:
            new_tokens.append(t)
            new_start_idxs.append(s)
            new_end_idxs.append(e)
    if cropped_count > 0:
        logger.debug(
            f"Cropped {cropped_count}/{len(new_tokens)} sequences exceeding max_tokens={max_tokens}"
        )
    return new_tokens, new_start_idxs, new_end_idxs


@torch.no_grad()
def evaluate_example(
    idx,
    model,
    tokenizer,
    data,
    device,
    task_meta: TaskMeta,
    seed=1234,
    *,
    label: str = "unknown",
    crop_oversized: bool = False,
) -> bool | None:
    """Evaluate one example.

    Returns True/False when scored, or None when the rendered rows exceed the
    model context length and the example must be excluded from aggregation.
    """
    item = data[idx]
    task_type = task_meta.task_type
    num_fewshot = task_meta.num_fewshot
    continuation_delimiter = task_meta.continuation_delimiter

    # Sample few-shot examples (excluding current item). Clamped so tiny-N
    # dev runs (where len(data) may be smaller than num_fewshot) don't crash.
    fewshot_indices = _sample_fewshot_indices(
        data_len=len(data),
        current_idx=idx,
        num_fewshot=num_fewshot,
        seed=seed + idx,
    )
    fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == TaskType.MULTIPLE_CHOICE:
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == TaskType.SCHEMA:
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == TaskType.LANGUAGE_MODELING:
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    max_tokens = core_model_max_seq_len(model)
    if max_tokens is not None:
        if crop_oversized:
            tokens, start_idxs, end_idxs = _crop_core_rows_to_max_seq_len(
                tokens, start_idxs, end_idxs, max_tokens
            )
        elif should_skip_oversized_core_example(
            tokens,
            max_tokens=max_tokens,
            label=label,
            idx=idx,
        ):
            return None

    # Stack up all the sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id()  # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    context_token_ids = []
    for row, (tok_seq, si, ei) in enumerate(zip(tokens, start_idxs, end_idxs, strict=True)):
        target_start = max(si - 1, 0)
        target_end = max(target_start + 1, ei - 1)
        target_end = min(target_end, input_ids.size(1))
        target_mask[row, target_start:target_end] = True
        context_token_ids.append(tok_seq[: target_start + 1])

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions = forward_model(
        model,
        input_ids,
        target_mask=target_mask,
        context_token_ids=context_token_ids,
    )

    # See if the losses/predictions come out correctly
    if task_type == TaskType.LANGUAGE_MODELING:
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = bool(torch.all(predicted_tokens == actual_tokens).item())
    elif task_type in (TaskType.MULTIPLE_CHOICE, TaskType.SCHEMA):
        # For MC/schema: find the option with lowest average loss
        mean_losses = [
            losses[i, si - 1 : ei - 1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs, strict=True))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = bool(pred_idx == item["gold"])
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(
    model,
    tokenizer,
    data,
    device,
    task_meta: TaskMeta,
    seed=1234,
    *,
    label: str = "unknown",
    crop_oversized: bool = False,
):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    # Empty datasets used to return NaN silently, poisoning the CORE aggregate
    # downstream (``scripts/base_eval.py`` forwards NaN into ``centered_results``).
    # Fail loudly so misconfigurations (``max_per_task=0``, failed shard
    # downloads, etc.) surface immediately.
    if len(data) == 0:
        raise ValueError("core_eval: empty dataset")
    t0 = time.perf_counter()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    evaluated = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(
            idx,
            model,
            tokenizer,
            data,
            device,
            task_meta,
            seed=seed,
            label=label,
            crop_oversized=crop_oversized,
        )
        if is_correct is None:
            continue
        correct[idx] = float(is_correct)
        evaluated[idx] = 1.0
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(evaluated, op=dist.ReduceOp.SUM)
    # compute the mean
    n_evaluated = int(evaluated.sum().item())
    if n_evaluated == 0:
        logger.warning(
            f"evaluate_task: {label} skipped all {len(data)} examples; "
            "no evaluable CORE examples remain"
        )
        return None
    mean_correct = (correct.sum() / evaluated.sum()).item()
    elapsed = time.perf_counter() - t0
    examples_per_sec = n_evaluated / elapsed if elapsed > 0 else 0
    logger.info(
        f"evaluate_task: {elapsed:.1f}s | {n_evaluated} examples scored/"
        f"{len(data)} total | {examples_per_sec:.1f} ex/sec"
    )
    return mean_correct
