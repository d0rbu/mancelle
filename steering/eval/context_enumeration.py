"""Exact eval-context enumeration for target-aware conditioning caches."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from nanochat.core_eval import (
    TaskMeta,
    TaskType,
    _sample_fewshot_indices,
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
    should_skip_oversized_core_example,
)
from steering.eval.chat_generation import empty_assistant_conversation, render_chat_completion
from steering.eval.harmfulness_prompt_pool import HarmfulnessPromptPool
from steering.eval.target_conditioning import token_context_key
from steering.segmentation import SegmentMode, token_to_char_offsets
from steering.sft.turn_spans import (
    ChatSpecialTokens,
    TurnRole,
    legacy_turn_spans,
    rendered_text_and_turn_char_starts,
)

if TYPE_CHECKING:
    from steering.conditional.encoder_decoder.rlvr_data import RLVRRow


class TargetRealRole(StrEnum):
    FULL = "full"
    CONTEXT_PREFIX = "context_prefix"


@dataclass(frozen=True)
class EvalContextRecord:
    """One exact context token sequence that must exist in the feature cache."""

    suite: str
    token_ids: tuple[int, ...]
    context_text: str
    task: str | None = None
    seed: int | None = None
    example_index: int | None = None
    row_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    turn_char_starts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        starts = self.turn_char_starts
        if starts is not None and (
            not starts
            or any(type(start) is not int for start in starts)
            or starts[0] != 0
            or any(right <= left for left, right in pairwise(starts))
            or starts[-1] >= len(self.context_text)
        ):
            raise ValueError("turn_char_starts must start at zero, strictly increase, and lie within text")

    @property
    def key(self) -> str:
        return token_context_key(self.token_ids)


def context_text_for_extraction(tokenizer, token_ids) -> str:
    """Decode eval context tokens into the text Gemma should process.

    Eval token sequences are keyed by exact nanochat decoder tokens and usually
    begin with the nanochat BOS token.  Gemma feature extraction adds its own BOS
    prefix, so the decoder BOS must not also become a literal ``"<bos>"`` string
    in the extracted text.  Other nanochat special tokens remain in the decoded
    text because they are real prompt-template content from the decoder's view.
    """

    tokens = [int(tok) for tok in token_ids]
    if tokens and tokens[0] == int(tokenizer.get_bos_token_id()):
        tokens = tokens[1:]
    return tokenizer.decode(tokens)


def dedupe_context_records(
    records: list[EvalContextRecord],
) -> list[EvalContextRecord]:
    """Return records with duplicate token sequences removed, preserving order."""

    seen: dict[str, EvalContextRecord] = {}
    deduped: list[EvalContextRecord] = []
    for record in records:
        if record.key in seen:
            previous = seen[record.key]
            if (record.turn_char_starts is not None or previous.turn_char_starts is not None) and (
                record.context_text != previous.context_text
                or record.turn_char_starts != previous.turn_char_starts
            ):
                raise ValueError(f"Conflicting duplicate turn annotations for {record.key}")
            continue
        seen[record.key] = record
        deduped.append(record)
    return deduped


def enumerate_core_example_contexts(
    *,
    tokenizer,
    data: list[dict],
    idx: int,
    task_meta: TaskMeta,
    seed: int,
    task_label: str | None = None,
    max_seq_len: int | None = None,
) -> list[EvalContextRecord]:
    """Enumerate CORE context_token_ids exactly as ``evaluate_example`` does."""

    item = data[idx]
    fewshot_indices = _sample_fewshot_indices(
        data_len=len(data),
        current_idx=idx,
        num_fewshot=task_meta.num_fewshot,
        seed=seed + idx,
    )
    fewshot_examples = [data[i] for i in fewshot_indices]

    if task_meta.task_type == TaskType.MULTIPLE_CHOICE:
        prompts = render_prompts_mc(
            item,
            task_meta.continuation_delimiter,
            fewshot_examples,
        )
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_meta.task_type == TaskType.SCHEMA:
        prompts = render_prompts_schema(
            item,
            task_meta.continuation_delimiter,
            fewshot_examples,
        )
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_meta.task_type == TaskType.LANGUAGE_MODELING:
        prompts = render_prompts_lm(
            item,
            task_meta.continuation_delimiter,
            fewshot_examples,
        )
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported CORE task type: {task_meta.task_type}")

    if max_seq_len is not None and should_skip_oversized_core_example(
        tokens,
        max_tokens=max_seq_len,
        label=task_label or "unknown",
        idx=idx,
    ):
        return []

    records: list[EvalContextRecord] = []
    for row, (tok_seq, si, ei) in enumerate(zip(tokens, start_idxs, end_idxs, strict=True)):
        del ei
        target_start = max(si - 1, 0)
        context_tokens = tuple(int(tok) for tok in tok_seq[: target_start + 1])
        records.append(
            EvalContextRecord(
                suite="core",
                task=task_label,
                seed=seed,
                example_index=idx,
                row_index=row,
                token_ids=context_tokens,
                context_text=context_text_for_extraction(tokenizer, context_tokens),
                metadata={
                    "task_type": str(task_meta.task_type),
                    "target_start": target_start,
                    "num_fewshot": task_meta.num_fewshot,
                },
            )
        )
    return records


def enumerate_core_task_contexts(
    *,
    tokenizer,
    data: list[dict],
    task_meta: TaskMeta,
    seed: int,
    task_label: str | None = None,
    max_per_task: int = -1,
    max_seq_len: int | None = None,
) -> list[EvalContextRecord]:
    """Enumerate all CORE contexts for one task under progression settings."""

    task_data = list(data)
    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(task_data)
    if max_per_task > 0:
        task_data = task_data[:max_per_task]

    return enumerate_selected_core_task_contexts(
        tokenizer=tokenizer,
        data=task_data,
        task_meta=task_meta,
        seed=seed,
        task_label=task_label,
        max_seq_len=max_seq_len,
    )


def enumerate_selected_core_task_contexts(
    *,
    tokenizer,
    data: list[dict],
    task_meta: TaskMeta,
    seed: int,
    task_label: str | None = None,
    max_seq_len: int | None = None,
) -> list[EvalContextRecord]:
    """Enumerate CORE contexts from rows already shuffled and capped."""

    records: list[EvalContextRecord] = []
    for idx in range(len(data)):
        records.extend(
            enumerate_core_example_contexts(
                tokenizer=tokenizer,
                data=data,
                idx=idx,
                task_meta=task_meta,
                seed=seed,
                task_label=task_label,
                max_seq_len=max_seq_len,
            )
        )
    return records


def enumerate_chat_categorical_contexts(
    *,
    tokenizer,
    task_object,
    batch_size: int,
    max_problems: int | None = None,
    task_name: str | None = None,
) -> list[EvalContextRecord]:
    """Enumerate chat categorical contexts exactly as ``run_categorical_eval``."""

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    records: list[EvalContextRecord] = []
    for batch_idx, start in enumerate(range(0, num_problems, batch_size)):
        end = min(start + batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(start, end)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        for row, ids in enumerate(prompt_ids):
            tokens = tuple(int(tok) for tok in ids)
            records.append(
                EvalContextRecord(
                    suite="chat",
                    task=task_name,
                    example_index=start + row,
                    row_index=row,
                    token_ids=tokens,
                    context_text=context_text_for_extraction(tokenizer, tokens),
                    metadata={
                        "eval_type": "categorical",
                        "batch_index": batch_idx,
                        "answer_position": len(ids) - 1,
                    },
                )
            )
    return records


def enumerate_chat_generative_contexts(
    *,
    tokenizer,
    task_object,
    max_problems: int | None = None,
    task_name: str | None = None,
) -> list[EvalContextRecord]:
    """Enumerate chat generative prompt tokens exactly as ``run_generative_eval``."""

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    records: list[EvalContextRecord] = []
    for idx in range(num_problems):
        conversation = task_object[idx]
        ids = tokenizer.render_for_completion(conversation)
        tokens = tuple(int(tok) for tok in ids)
        records.append(
            EvalContextRecord(
                suite="chat",
                task=task_name,
                example_index=idx,
                token_ids=tokens,
                context_text=context_text_for_extraction(tokenizer, tokens),
                metadata={"eval_type": "generative"},
            )
        )
    return records


def enumerate_harmfulness_pool_contexts(
    *,
    tokenizer,
    pool: HarmfulnessPromptPool,
    task_name: str = "harmfulness",
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> list[EvalContextRecord]:
    """Enumerate exact generation contexts for a resolved prompt pool."""
    records: list[EvalContextRecord] = []
    for idx, prompt in enumerate(pool.rows):
        conversation = empty_assistant_conversation(prompt["prompt"])
        ids = tokenizer.render_for_completion(conversation)
        tokens = tuple(int(tok) for tok in ids)
        records.append(
            EvalContextRecord(
                suite="harmfulness",
                task=task_name,
                seed=pool.seed,
                example_index=idx,
                token_ids=tokens,
                context_text=context_text_for_extraction(tokenizer, tokens),
                metadata={"pool": prompt["pool"]},
            )
        )
    if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        records = [annotate_turn_context_record(record, tokenizer) for record in records]
        validate_turn_context_records(records, tokenizer)
    return records


def enumerate_chat_rlvr_contexts(
    *,
    tokenizer,
    rows: Sequence[RLVRRow],
) -> list[EvalContextRecord]:
    """Enumerate exact question-only RLVR prompts from an authenticated bundle."""

    records = [
        EvalContextRecord(
            suite="chat_rlvr",
            task=row.task,
            example_index=index,
            token_ids=row.prompt_token_ids,
            context_text=context_text_for_extraction(tokenizer, row.prompt_token_ids),
            metadata={"row_id": row.row_id, "split": row.split},
        )
        for index, row in enumerate(rows)
    ]
    records = [annotate_turn_context_record(record, tokenizer) for record in records]
    validate_turn_context_records(records, tokenizer)
    return records


def enumerate_sample_prompt_contexts(
    *,
    prompts: Sequence[str],
    tokenizer,
    max_seq_len: int,
) -> list[EvalContextRecord]:
    """One record per raw-text sample prompt (BOS-prepended, no chat template).

    Used by the in-training sample-generation eval in
    ``scripts/train/conditional_encdec_train.py``: each prompt becomes one
    EvalContextRecord whose token sequence matches what the training
    script tokenizes at runtime.

    Args:
        prompts: Sequence of raw text strings (no chat-template wrapping).
        tokenizer: Nanochat tokenizer.
        max_seq_len: Hard cap on token-id sequence length per prompt.

    Returns:
        list[EvalContextRecord] of length ``len(prompts)``, with
        ``suite="samples"`` and ``metadata={"prompt": <text>}``.

    Raises:
        ValueError: If any tokenized prompt exceeds ``max_seq_len``.
    """
    records: list[EvalContextRecord] = []
    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise ValueError(
                f"sample_prompts[{idx}] must be a string, got "
                f"{type(prompt).__name__}: {prompt!r}. Check the sample_prompts "
                f"list in your config."
            )
        token_list = tokenizer(prompt, prepend=tokenizer.get_bos_token_id())
        if len(token_list) > max_seq_len:
            raise ValueError(
                f"sample prompt index {idx} ({prompt!r}) tokenizes to "
                f"{len(token_list)} tokens, exceeding max_seq_len={max_seq_len}. "
                f"Edit the prompt or raise max_seq_len."
            )
        token_ids = tuple(int(tok) for tok in token_list)
        records.append(
            EvalContextRecord(
                suite="samples",
                task="conditional_encdec_train_samples",
                seed=None,
                example_index=idx,
                row_index=0,
                token_ids=token_ids,
                context_text=context_text_for_extraction(tokenizer, token_ids),
                metadata={"prompt": str(prompt)},
            )
        )
    return records


def annotate_turn_context_record(record: EvalContextRecord, tokenizer) -> EvalContextRecord:
    """Bind extraction text and logical turns to the same exact token sequence."""
    text, starts = rendered_text_and_turn_char_starts(record.token_ids, tokenizer)
    return replace(record, context_text=text, turn_char_starts=starts)


def count_assistant_turns(token_ids: Sequence[int], tokenizer) -> int:
    """Count rendered assistant turns, not the spans a tool output splits.

    ``legacy_turn_spans`` opens a fresh assistant span after every tool
    ``<output_end>``, so only spans that open on the marker itself are turns.
    Spans are BOS-free, hence the ``+ 1`` back into the full render.
    """
    specials = ChatSpecialTokens.from_tokenizer(tokenizer)
    return sum(
        TurnRole(span.role) is TurnRole.ASSISTANT
        and token_ids[span.start + 1] == specials.assistant_start
        for span in legacy_turn_spans(token_ids, specials)
    )


def turn_supervision_interval(token_ids: Sequence[int], tokenizer) -> tuple[int, int]:
    """Recover the renderer's first/last supervised positions, including truncation.

    Assistant text, Python calls and the assistant terminator are supervised;
    assistant-start markers and tool-output spans are not. Logical turn spans
    are BOS-free, so translate their indices back into the full render.
    """
    specials = ChatSpecialTokens.from_tokenizer(tokenizer)
    supervised = []
    for span in legacy_turn_spans(token_ids, specials):
        if TurnRole(span.role) is TurnRole.ASSISTANT:
            start, end = span.start + 1, span.end + 1
            if token_ids[start] == specials.assistant_start:
                start += 1
            if start < end:
                supervised.append((start, end))
    if not supervised:
        raise ValueError("turn full row has no renderer-supervised reference tokens")
    return supervised[0][0], supervised[-1][1]


def validate_turn_context_records(records: Sequence[EvalContextRecord], tokenizer) -> None:
    """Reconstruct annotations and require independently extracted full/prefix pairs."""
    unique = dedupe_context_records(list(records))
    rows = {record.token_ids: record for record in unique}
    by_key = {record.key: record for record in unique}
    paired_prefixes: set[tuple[int, ...]] = set()
    for record in unique:
        if record.suite not in {"chat_target", "harmfulness", "chat_rlvr"}:
            raise ValueError(
                "turn context caches support chat_target, harmfulness, and chat_rlvr only; "
                "CORE stays fixed-only"
            )
        expected = annotate_turn_context_record(record, tokenizer)
        if record.turn_char_starts is None or (record.context_text, record.turn_char_starts) != (
            expected.context_text, expected.turn_char_starts
        ):
            raise ValueError(f"Turn text/boundaries disagree with exact tokens for {record.key}")
        role = record.metadata.get("target_real_role")
        if record.suite in {"harmfulness", "chat_rlvr"}:
            marker = ChatSpecialTokens.from_tokenizer(tokenizer).assistant_start
            if (role is not None or record.token_ids[-1] != marker
                    or count_assistant_turns(record.token_ids, tokenizer) != 1
                    or record.context_text[record.turn_char_starts[-1]:] != tokenizer.decode([marker])):
                raise ValueError(
                    f"{record.suite} turn context must be an independent marker-only assistant prefix"
                )
            continue
        if role == TargetRealRole.FULL:
            target_start = record.metadata.get("target_start")
            if type(target_start) is not int or not 1 < target_start < len(record.token_ids):
                raise ValueError("turn full row has invalid target_start")
            expected_start, expected_end = turn_supervision_interval(record.token_ids, tokenizer)
            offsets = token_to_char_offsets(tokenizer, list(record.token_ids))
            expected_interval = {"target_start": expected_start, "target_char_start": offsets[expected_start],
                                 "target_char_end": offsets[expected_end]}
            for field, value in expected_interval.items():
                if type(record.metadata.get(field)) is not int or record.metadata[field] != value:
                    raise ValueError(f"{record.key}: {field} disagrees with renderer supervision interval")
            prefix = rows.get(record.token_ids[:target_start])
            # Mixed suites may deduplicate to an identical independently
            # enumerated prompt, which receives the marker-only turn and
            # geometry checks above and can supply the exact same prefix.
            if prefix is None or not (
                prefix.metadata.get("target_real_role") == TargetRealRole.CONTEXT_PREFIX
                or (prefix.suite in {"harmfulness", "chat_rlvr"} and prefix.metadata.get("target_real_role") is None)
            ):
                raise ValueError(f"Missing independent context_prefix for {record.key}")
            paired_prefixes.add(prefix.token_ids)
        elif role == TargetRealRole.CONTEXT_PREFIX:
            if record.turn_char_starts is None:
                raise ValueError("turn context_prefix requires turn_char_starts")
            paired_full = by_key.get(record.metadata.get("paired_target_key"))
            if paired_full is None or paired_full.metadata.get("target_real_role") != TargetRealRole.FULL or paired_full.token_ids[:len(record.token_ids)] != record.token_ids:
                raise ValueError(f"Missing matching full row for context_prefix {record.key}")
            marker = ChatSpecialTokens.from_tokenizer(tokenizer).assistant_start
            if record.token_ids[-1] != marker or record.context_text[record.turn_char_starts[-1]:] != tokenizer.decode([marker]):
                raise ValueError("context_prefix must end with a marker-only assistant turn")
        else:
            raise ValueError(f"Unknown turn target_real_role: {role!r}")
    for record in unique:
        if record.metadata.get("target_real_role") == TargetRealRole.CONTEXT_PREFIX and record.token_ids not in paired_prefixes:
            raise ValueError(f"Missing full row for context_prefix {record.key}")


def enumerate_chat_target_contexts(
    *,
    tokenizer,
    task_object,
    max_problems: int | None,
    task_name: str | None = None,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> list[EvalContextRecord]:
    """For each example, render target-real full and prompt/context rows.

    The full row gives target-real dispatch the ground-truth target span. The
    prompt/context row gives the eval wrapper exact observed-token features
    without deriving them from prompt+target extraction.

    Conversations must be in {"messages": [...]} format (same as accepted by
    tokenizer.render_conversation).
    """
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    records: list[EvalContextRecord] = []
    for idx in range(num_problems):
        conv = task_object[idx]
        ids, mask = tokenizer.render_conversation(conv)
        # Char offsets: char_offsets[i] = start of token i, char_offsets[-1] = total chars.
        char_offsets = token_to_char_offsets(tokenizer, ids)
        # Target = positions where mask==1 (assistant content + assistant_end).
        mask_1_indices = [i for i, m in enumerate(mask) if m == 1]
        if not mask_1_indices:
            raise ValueError(
                f"chat_target: example {idx} has no mask=1 tokens (no assistant content); "
                f"cannot derive target_char range"
            )
        target_start_idx = int(mask_1_indices[0])
        target_char_start = int(char_offsets[mask_1_indices[0]])
        assert mask_1_indices[-1] + 1 <= len(ids), (
            f"chat_target: mask_1_indices[-1]+1={mask_1_indices[-1] + 1} > len(ids)={len(ids)}; "
            f"token_to_char_offsets returns offsets of length len(ids)+1"
        )
        target_char_end = int(char_offsets[mask_1_indices[-1] + 1])
        full_token_ids = tuple(int(tok) for tok in ids)
        records.append(
            EvalContextRecord(
                suite="chat_target",
                task=task_name,
                example_index=idx,
                token_ids=full_token_ids,
                context_text=context_text_for_extraction(tokenizer, full_token_ids),
                metadata={
                    "target_real_role": TargetRealRole.FULL,
                    "target_start": target_start_idx,
                    "target_char_start": target_char_start,
                    "target_char_end": target_char_end,
                },
            )
        )
        context_token_ids = tuple(int(tok) for tok in ids[:target_start_idx])
        if not context_token_ids:
            raise ValueError(
                f"chat_target: example {idx} has empty prompt/context prefix; "
                "cannot build exact context-cache row"
            )
        if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
            assistant_turns = count_assistant_turns(ids, tokenizer)
            if assistant_turns != 1:
                raise ValueError(
                    "turn chat_target supports a single assistant turn; "
                    f"example {idx} has {assistant_turns}"
                )
            if context_token_ids != tuple(render_chat_completion(tokenizer, conv).token_ids):
                raise ValueError(f"chat_target: example {idx} completion-prefix disagreement")
        records.append(
            EvalContextRecord(
                suite="chat_target",
                task=task_name,
                example_index=idx,
                token_ids=context_token_ids,
                context_text=context_text_for_extraction(tokenizer, context_token_ids),
                metadata={
                    "target_real_role": TargetRealRole.CONTEXT_PREFIX,
                    "paired_target_key": token_context_key(full_token_ids),
                    "target_start": max(len(context_token_ids) - 1, 0),
                },
            )
        )
    if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        records = [annotate_turn_context_record(record, tokenizer) for record in records]
        validate_turn_context_records(records, tokenizer)
    return records


def enumerate_core_target_contexts(
    *,
    tokenizer,
    data,
    task_meta: TaskMeta,
    seed: int,
    task_label: str,
    max_per_task: int,
    max_seq_len: int,
) -> list[EvalContextRecord]:
    """Per CORE example, enumerate one record per row (MC) or per example (Schema/LM)
    with target_char_* metadata pointing at the continuation/answer span."""
    task_data = list(data)
    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(task_data)
    if max_per_task > 0:
        task_data = task_data[:max_per_task]

    return enumerate_selected_core_target_contexts(
        tokenizer=tokenizer,
        data=task_data,
        task_meta=task_meta,
        seed=seed,
        task_label=task_label,
        max_seq_len=max_seq_len,
    )


def enumerate_selected_core_target_contexts(
    *,
    tokenizer,
    data: list[dict],
    task_meta: TaskMeta,
    seed: int,
    task_label: str,
    max_seq_len: int,
) -> list[EvalContextRecord]:
    """Enumerate CORE target rows from data already shuffled and capped."""

    records: list[EvalContextRecord] = []
    n = len(data)
    for idx in range(n):
        item = data[idx]
        fewshot_idxs = _sample_fewshot_indices(
            len(data),
            idx,
            task_meta.num_fewshot,
            seed + idx,
        )
        fewshot = [data[i] for i in fewshot_idxs]
        if task_meta.task_type is TaskType.MULTIPLE_CHOICE:
            prompts = render_prompts_mc(item, task_meta.continuation_delimiter, fewshot)
            tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
        elif task_meta.task_type is TaskType.SCHEMA:
            prompts = render_prompts_schema(item, task_meta.continuation_delimiter, fewshot)
            tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
        else:  # LM
            prompts = render_prompts_lm(item, task_meta.continuation_delimiter, fewshot)
            tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        if should_skip_oversized_core_example(
            tokens,
            max_tokens=max_seq_len,
            label=task_label,
            idx=idx,
        ):
            continue
        for row, (tok_seq, si, ei) in enumerate(zip(tokens, start_idxs, end_idxs, strict=True)):
            tok_list = list(tok_seq)
            prefix_text = tokenizer.decode(tok_list[:si])
            full_text = tokenizer.decode(tok_list[:ei])
            target_start_safe = max(si - 1, 0)
            full_token_ids = tuple(int(tok) for tok in tok_list)
            records.append(
                EvalContextRecord(
                    suite="core_target",
                    task=task_label,
                    seed=seed,
                    example_index=idx,
                    row_index=row,
                    token_ids=full_token_ids,
                    context_text=context_text_for_extraction(tokenizer, full_token_ids),
                    metadata={
                        "target_real_role": TargetRealRole.FULL,
                        "target_start": target_start_safe,
                        "target_char_start": len(prefix_text),
                        "target_char_end": len(full_text),
                        "task_type": task_meta.task_type.value,
                    },
                )
            )
            context_token_ids = tuple(int(tok) for tok in tok_list[: target_start_safe + 1])
            if not context_token_ids:
                raise ValueError(
                    f"core_target: task={task_label} example={idx} row={row} "
                    "has empty prompt/context prefix; cannot build exact "
                    "context-cache row"
                )
            records.append(
                EvalContextRecord(
                    suite="core_target",
                    task=task_label,
                    seed=seed,
                    example_index=idx,
                    row_index=row,
                    token_ids=context_token_ids,
                    context_text=context_text_for_extraction(tokenizer, context_token_ids),
                    metadata={
                        "target_real_role": TargetRealRole.CONTEXT_PREFIX,
                        "paired_target_key": token_context_key(full_token_ids),
                        "target_start": max(len(context_token_ids) - 1, 0),
                        "task_type": task_meta.task_type.value,
                    },
                )
            )
    return records
