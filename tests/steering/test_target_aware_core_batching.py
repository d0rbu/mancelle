from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch as t

from nanochat import flash_attention as nanochat_flash_attention
from nanochat.core_eval import TaskMeta, TaskType, evaluate_task
from scripts.eval.base_eval import evaluate_target_aware_core_task
from tests.conftest import MockTokenizer


class _UniformTargetAwareModel:
    is_target_aware_encdec = True
    max_seq_len = 128

    def __init__(self) -> None:
        self.forward_shapes: list[tuple[int, int]] = []
        self.context_batch_sizes: list[int] = []
        self.attention_impls: list[nanochat_flash_attention.FlashAttnImpl] = []
        self.context_rows: list[tuple[int, ...]] = []
        self.target_positions: list[tuple[int, ...]] = []

    @contextmanager
    def conditioning_context(self, *, target_mask, context_token_ids):
        assert tuple(target_mask.shape)[0] == len(context_token_ids)
        self.context_batch_sizes.append(len(context_token_ids))
        self.context_rows.extend(tuple(row) for row in context_token_ids)
        self.target_positions.extend(
            tuple(mask_row.nonzero(as_tuple=True)[0].tolist()) for mask_row in target_mask
        )
        yield self

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        self.forward_shapes.append((batch_size, seq_len))
        self.attention_impls.append(nanochat_flash_attention._active_impl())
        return t.zeros(batch_size, seq_len, 128, dtype=t.float32)


def _task_case(task_type: TaskType) -> tuple[TaskMeta, list[dict]]:
    meta = TaskMeta(
        task_type=task_type,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    if task_type is TaskType.MULTIPLE_CHOICE:
        data = [
            {
                "query": f"Question {index}?",
                "choices": ["alpha", "beta", "gamma"],
                "gold": index % 3,
            }
            for index in range(5)
        ]
    elif task_type is TaskType.SCHEMA:
        data = [
            {
                "context_options": [f"left {index}", f"right {index}"],
                "continuation": "continues",
                "gold": index % 2,
            }
            for index in range(5)
        ]
    else:
        data = [{"context": f"prefix {index}", "continuation": "suffix"} for index in range(5)]
    return meta, data


@pytest.mark.parametrize("task_type", list(TaskType))
def test_batched_task_matches_vendored_per_example_scores(task_type: TaskType) -> None:
    tokenizer = MockTokenizer()
    task_meta, data = _task_case(task_type)
    reference_model = _UniformTargetAwareModel()
    batched_model = _UniformTargetAwareModel()

    reference = evaluate_task(
        reference_model,
        tokenizer,
        data,
        "cpu",
        task_meta,
        seed=1234,
        label=f"test_{task_type}",
    )
    previous_impl = nanochat_flash_attention._active_impl()
    batched = evaluate_target_aware_core_task(
        batched_model,
        tokenizer,
        data,
        "cpu",
        task_meta,
        seed=1234,
        label=f"test_{task_type}",
        example_batch_size=2,
    )

    assert batched == reference
    assert len(reference_model.forward_shapes) == 5
    assert len(batched_model.forward_shapes) == 3
    assert sum(batched_model.context_batch_sizes) == sum(reference_model.context_batch_sizes)
    assert batched_model.context_rows == reference_model.context_rows
    assert batched_model.target_positions == reference_model.target_positions
    assert batched_model.attention_impls == [previous_impl] * 3


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "2"])
def test_batched_task_rejects_invalid_example_batch_size(invalid) -> None:
    task_meta, data = _task_case(TaskType.MULTIPLE_CHOICE)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_target_aware_core_task(
            _UniformTargetAwareModel(),
            MockTokenizer(),
            data,
            "cpu",
            task_meta,
            example_batch_size=invalid,
        )


def test_batched_task_preserves_decoder_attention_override(monkeypatch) -> None:
    task_meta, data = _task_case(TaskType.MULTIPLE_CHOICE)
    monkeypatch.setattr(
        nanochat_flash_attention,
        "_override_impl",
        nanochat_flash_attention.FlashAttnImpl.FA4,
    )

    evaluate_target_aware_core_task(
        _UniformTargetAwareModel(),
        MockTokenizer(),
        data,
        "cpu",
        task_meta,
        example_batch_size=2,
    )

    assert nanochat_flash_attention._override_impl is nanochat_flash_attention.FlashAttnImpl.FA4


def test_batched_task_preserves_decoder_attention_override_on_error(monkeypatch) -> None:
    class _FailingModel(_UniformTargetAwareModel):
        def __call__(self, input_ids):
            raise RuntimeError("model forward failed")

    task_meta, data = _task_case(TaskType.MULTIPLE_CHOICE)
    monkeypatch.setattr(nanochat_flash_attention, "_override_impl", None)

    with pytest.raises(RuntimeError, match="model forward failed"):
        evaluate_target_aware_core_task(
            _FailingModel(),
            MockTokenizer(),
            data,
            "cpu",
            task_meta,
            example_batch_size=2,
        )

    assert nanochat_flash_attention._override_impl is None
