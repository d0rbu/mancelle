from __future__ import annotations

import torch

from nanochat.core_eval import TaskMeta, TaskType, evaluate_example, evaluate_task


class _ChoiceTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def __call__(self, prompts, prepend=None):
        rows = []
        for prompt in prompts:
            is_long = "OVERSIZED" in prompt
            prefix = [1] * (199 if is_long else 4)
            choice_token = 10 if prompt.endswith("A") else 11
            row = prefix + [choice_token]
            if prepend is not None:
                row = [prepend] + row
            rows.append(row)
        return rows


class _ZeroChoiceModel:
    max_seq_len = 50

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1
        return torch.zeros(input_ids.size(0), input_ids.size(1), 256)


def _mc_meta() -> TaskMeta:
    return TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )


def _schema_meta() -> TaskMeta:
    return TaskMeta(
        task_type=TaskType.SCHEMA,
        num_fewshot=0,
        continuation_delimiter=" ",
    )


def _lm_meta() -> TaskMeta:
    return TaskMeta(
        task_type=TaskType.LANGUAGE_MODELING,
        num_fewshot=0,
        continuation_delimiter=" ",
    )


class _LmTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def __call__(self, prompts, prepend=None):
        rows = []
        for prompt in prompts:
            prefix = [1] * (199 if "OVERSIZED" in prompt else 4)
            continuation = [2, 3] if prompt.endswith(" answer") else []
            row = prefix + continuation
            if prepend is not None:
                row = [prepend] + row
            rows.append(row)
        return rows


def test_evaluate_example_skips_oversized_rows_before_forward():
    model = _ZeroChoiceModel()
    tokenizer = _ChoiceTokenizer()
    data = [{"query": "OVERSIZED", "choices": ["A", "B"], "gold": 0}]

    result = evaluate_example(
        0,
        model,
        tokenizer,
        data,
        torch.device("cpu"),
        _mc_meta(),
        label="oversized_mc",
    )

    assert result is None
    assert model.calls == 0


def test_evaluate_example_skips_oversized_schema_rows_before_forward():
    model = _ZeroChoiceModel()
    tokenizer = _ChoiceTokenizer()
    data = [
        {
            "context_options": ["OVERSIZED context", "short context"],
            "continuation": "answer",
            "gold": 0,
        }
    ]

    result = evaluate_example(
        0,
        model,
        tokenizer,
        data,
        torch.device("cpu"),
        _schema_meta(),
        label="oversized_schema",
    )

    assert result is None
    assert model.calls == 0


def test_evaluate_example_skips_oversized_lm_rows_before_forward():
    model = _ZeroChoiceModel()
    tokenizer = _LmTokenizer()
    data = [{"context": "OVERSIZED passage", "continuation": "answer"}]

    result = evaluate_example(
        0,
        model,
        tokenizer,
        data,
        torch.device("cpu"),
        _lm_meta(),
        label="oversized_lm",
    )

    assert result is None
    assert model.calls == 0


def test_evaluate_task_denominator_excludes_skipped_examples():
    model = _ZeroChoiceModel()
    tokenizer = _ChoiceTokenizer()
    data = [
        {"query": "OVERSIZED", "choices": ["A", "B"], "gold": 1},
        {"query": "short", "choices": ["A", "B"], "gold": 0},
    ]

    accuracy = evaluate_task(
        model,
        tokenizer,
        data,
        torch.device("cpu"),
        _mc_meta(),
        label="mixed_mc",
    )

    assert accuracy == 1.0
    assert model.calls == 1


def test_evaluate_task_returns_none_when_all_examples_skip():
    model = _ZeroChoiceModel()
    tokenizer = _ChoiceTokenizer()
    data = [
        {"query": "OVERSIZED one", "choices": ["A", "B"], "gold": 0},
        {"query": "OVERSIZED two", "choices": ["A", "B"], "gold": 1},
    ]

    accuracy = evaluate_task(
        model,
        tokenizer,
        data,
        torch.device("cpu"),
        _mc_meta(),
        label="all_oversized_mc",
    )

    assert accuracy is None
    assert model.calls == 0
