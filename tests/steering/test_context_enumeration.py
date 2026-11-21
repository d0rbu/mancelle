from __future__ import annotations

import random
from contextlib import contextmanager
from typing import cast
from unittest.mock import patch

import pytest
import torch

from nanochat.core_eval import TaskMeta, TaskType, evaluate_example
from scripts.eval.chat_eval import run_categorical_eval
from steering.eval.context_enumeration import (
    context_text_for_extraction,
    dedupe_context_records,
    enumerate_chat_categorical_contexts,
    enumerate_chat_generative_contexts,
    enumerate_core_example_contexts,
    enumerate_core_task_contexts,
    enumerate_harmfulness_pool_contexts,
)
from steering.eval.harmfulness_prompt_pool import build_harmfulness_prompt_pool
from tests.conftest import MockTokenizer


class VisibleBosTokenizer(MockTokenizer):
    def decode(self, ids: list[int]) -> str:
        parts = []
        for tok in ids:
            if tok == self._bos_id:
                parts.append("<bos>")
            elif tok == self._eos_id:
                parts.append("<eos>")
            else:
                parts.append(chr(tok - 2))
        return "".join(parts)


def test_context_text_for_extraction_strips_only_leading_bos():
    tokenizer = VisibleBosTokenizer()

    assert context_text_for_extraction(tokenizer, [0, 99, 100]) == "ab"
    assert context_text_for_extraction(tokenizer, [99, 0, 100]) == "a<bos>b"


def test_core_enumerator_matches_runtime_context_token_ids():
    tokenizer = MockTokenizer()
    data = [
        {"query": "Q0?", "choices": ["A", "B"], "gold": 0},
        {"query": "Q1?", "choices": ["C", "D"], "gold": 1},
        {"query": "Q2?", "choices": ["E", "F"], "gold": 0},
        {"query": "Q3?", "choices": ["G", "H"], "gold": 1},
    ]
    task_meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=1,
        continuation_delimiter=" ",
    )
    captured = []

    def fake_forward(model, input_ids, target_mask=None, context_token_ids=None):
        del model, target_mask
        assert context_token_ids is not None
        captured.extend([tuple(row) for row in context_token_ids])
        losses = torch.zeros_like(input_ids, dtype=torch.float32)
        predictions = input_ids.clone()
        return losses, predictions

    with patch("nanochat.core_eval.forward_model", side_effect=fake_forward):
        evaluate_example(
            1,
            object(),
            tokenizer,
            data,
            torch.device("cpu"),
            task_meta,
            seed=1234,
        )

    enumerated = enumerate_core_example_contexts(
        tokenizer=tokenizer,
        data=data,
        idx=1,
        task_meta=task_meta,
        seed=1234,
        task_label="toy_mc",
    )

    assert [record.token_ids for record in enumerated] == captured


def test_core_enumerator_oversized_skip_matches_runtime_logic():
    tokenizer = MockTokenizer()
    data = [{"context": "abcdef", "continuation": "XYZ"}]
    task_meta = TaskMeta(
        task_type=TaskType.LANGUAGE_MODELING,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    captured = []

    class ShortContextModel:
        max_seq_len = 7

    def fake_forward(model, input_ids, target_mask=None, context_token_ids=None):
        del model, target_mask
        assert context_token_ids is not None
        captured.extend([tuple(row) for row in context_token_ids])
        losses = torch.zeros_like(input_ids, dtype=torch.float32)
        predictions = input_ids.clone()
        return losses, predictions

    with patch("nanochat.core_eval.forward_model", side_effect=fake_forward):
        result = evaluate_example(
            0,
            ShortContextModel(),
            tokenizer,
            data,
            torch.device("cpu"),
            task_meta,
            seed=1,
            label="toy_lm",
        )
    assert result is None
    assert captured == []

    enumerated = enumerate_core_example_contexts(
        tokenizer=tokenizer,
        data=data,
        idx=0,
        task_meta=task_meta,
        seed=1,
        task_label="toy_lm",
        max_seq_len=ShortContextModel.max_seq_len,
    )

    assert enumerated == []


def test_core_enumerator_covers_schema_and_task_level_subsampling():
    tokenizer = MockTokenizer()
    data = [
        {
            "context_options": [f"left {i}", f"right {i}"],
            "continuation": "answer",
            "gold": i % 2,
        }
        for i in range(4)
    ]
    task_meta = TaskMeta(
        task_type=TaskType.SCHEMA,
        num_fewshot=0,
        continuation_delimiter=" ",
    )

    records = enumerate_core_task_contexts(
        tokenizer=tokenizer,
        data=data,
        task_meta=task_meta,
        seed=5,
        task_label="schema",
        max_per_task=2,
    )

    assert len(records) == 4
    assert all(record.metadata["task_type"].endswith("schema") for record in records)


def test_core_enumerator_rejects_unsupported_and_skips_oversized():
    tokenizer = MockTokenizer()
    bad_meta = TaskMeta(
        task_type=cast(TaskType, "bogus"),
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    with pytest.raises(ValueError, match="Unsupported CORE"):
        enumerate_core_example_contexts(
            tokenizer=tokenizer,
            data=[{"query": "Q", "choices": ["A"], "gold": 0}],
            idx=0,
            task_meta=bad_meta,
            seed=1,
        )

    lm_meta = TaskMeta(
        task_type=TaskType.LANGUAGE_MODELING,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    with pytest.raises(ValueError, match="max_seq_len"):
        enumerate_core_example_contexts(
            tokenizer=tokenizer,
            data=[{"context": "abc", "continuation": "def"}],
            idx=0,
            task_meta=lm_meta,
            seed=1,
            max_seq_len=0,
        )
    assert (
        enumerate_core_example_contexts(
            tokenizer=tokenizer,
            data=[{"context": "abc", "continuation": "def"}],
            idx=0,
            task_meta=lm_meta,
            seed=1,
            max_seq_len=1,
        )
        == []
    )


def test_core_enumerator_clamps_fewshot_population_like_runtime():
    tokenizer = MockTokenizer()
    data = [
        {"query": "Q0?", "choices": ["A", "B"], "gold": 0},
        {"query": "Q1?", "choices": ["C", "D"], "gold": 1},
    ]
    task_meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=2,
        continuation_delimiter=" ",
    )
    captured = []

    def fake_forward(model, input_ids, target_mask=None, context_token_ids=None):
        del model, target_mask
        assert context_token_ids is not None
        captured.extend([tuple(row) for row in context_token_ids])
        losses = torch.zeros_like(input_ids, dtype=torch.float32)
        predictions = input_ids.clone()
        return losses, predictions

    with patch("nanochat.core_eval.forward_model", side_effect=fake_forward):
        evaluate_example(
            0,
            object(),
            tokenizer,
            data,
            torch.device("cpu"),
            task_meta,
            seed=5,
        )

    enumerated = enumerate_core_example_contexts(
        tokenizer=tokenizer,
        data=data,
        idx=0,
        task_meta=task_meta,
        seed=5,
        task_label="toy_fewshot",
    )

    assert [record.token_ids for record in enumerated] == captured

    task_records = enumerate_core_task_contexts(
        tokenizer=tokenizer,
        data=data,
        task_meta=task_meta,
        seed=5,
        task_label="toy_fewshot",
        max_per_task=2,
    )
    assert len(task_records) == 4


def test_chat_categorical_enumerator_matches_runtime_contexts():
    tokenizer = MockTokenizer()

    class ToyTask:
        eval_type = "categorical"

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"text": f"question {idx} ", "letters": ["A", "B"]}

        def evaluate(self, conversation, predicted_letter):
            del conversation
            return predicted_letter == "A"

    class TargetAwareModel:
        is_target_aware_encdec = True

        def __init__(self):
            self.captured = []

        def get_device(self):
            return torch.device("cpu")

        @contextmanager
        def conditioning_context(self, target_mask=None, context_token_ids=None):
            del target_mask
            assert context_token_ids is not None
            self.captured.extend([tuple(row) for row in context_token_ids])
            yield self

        def __call__(self, prompt_ids):
            logits = torch.zeros(
                prompt_ids.shape[0],
                prompt_ids.shape[1],
                tokenizer.get_vocab_size(),
            )
            a_id = tokenizer.encode("A")[0]
            logits[:, :, a_id] = 1.0
            return logits

    task = ToyTask()
    model = TargetAwareModel()
    run_categorical_eval(task, tokenizer, model, batch_size=2, max_problems=2)
    enumerated = enumerate_chat_categorical_contexts(
        tokenizer=tokenizer,
        task_object=task,
        batch_size=2,
        max_problems=2,
        task_name="toy",
    )

    assert [record.token_ids for record in enumerated] == model.captured


def test_chat_generative_enumerator_uses_render_for_completion():
    tokenizer = MockTokenizer()

    class ToyTask:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return {"text": f"generate {idx}"}

    records = enumerate_chat_generative_contexts(
        tokenizer=tokenizer,
        task_object=ToyTask(),
        max_problems=2,
        task_name="toy_gen",
    )

    assert len(records) == 2
    assert records[0].metadata["eval_type"] == "generative"
    assert records[0].token_ids == tuple(tokenizer.render_for_completion({"text": "generate 0"}))


def test_harmfulness_enumerator_matches_seeded_prompt_selection():
    tokenizer = MockTokenizer()
    harmful = [{"prompt": "h0"}, {"prompt": "h1"}, {"prompt": "h2"}]
    benign = [{"prompt": "b0"}]

    pool = build_harmfulness_prompt_pool(
        harmful=harmful,
        benign=benign,
        seed=7,
        max_harmful=2,
    )
    records = enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool)

    assert [record.metadata["pool"] for record in records] == [
        "harmful",
        "harmful",
        "benign",
    ]
    selected_harmful = list(harmful)
    random.Random(7).shuffle(selected_harmful)
    expected_rows = selected_harmful[:2] + benign
    for record, row in zip(records, expected_rows, strict=True):
        expected = tokenizer.render_for_completion(
            {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": ""},
                ]
            }
        )
        assert record.token_ids == tuple(expected)
    assert len({record.key for record in records}) == 3


def test_dedupe_context_records_preserves_first_occurrence():
    tokenizer = MockTokenizer()
    pool = build_harmfulness_prompt_pool(
        harmful=[{"prompt": "same"}, {"prompt": "same"}],
        benign=[],
        seed=1,
        max_harmful=None,
    )
    records = enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool)

    deduped = dedupe_context_records(records)

    assert len(records) == 2
    assert len(deduped) == 1
    assert deduped[0].example_index == 0
