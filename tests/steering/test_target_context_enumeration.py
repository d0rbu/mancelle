"""PR-C: chat_target / core_target context enumeration."""

from __future__ import annotations

import random

import pytest

from steering.eval.context_enumeration import (
    enumerate_chat_target_contexts,
    enumerate_core_target_contexts,
)
from tests.conftest import build_tiny_tokenizer


class _FakeChatTask:
    """Wraps a list of {messages: [...]} conversation dicts as a task object."""

    def __init__(self, convs):
        self._convs = convs

    def __getitem__(self, idx):
        return self._convs[idx]

    def __len__(self):
        return len(self._convs)


def test_chat_target_records_carry_target_char_metadata():
    tokenizer = build_tiny_tokenizer()
    task = _FakeChatTask(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q1?"},
                    {"role": "assistant", "content": "A1"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q2?"},
                    {"role": "assistant", "content": "A2A"},
                ]
            },
        ]
    )
    records = enumerate_chat_target_contexts(
        tokenizer=tokenizer,
        task_object=task,
        max_problems=None,
        task_name="fake_chat",
    )
    assert len(records) == 4
    full_records = [r for r in records if r.metadata["target_real_role"] == "full"]
    prefix_records = [
        r for r in records if r.metadata["target_real_role"] == "context_prefix"
    ]
    assert len(full_records) == 2
    assert len(prefix_records) == 2
    for r in full_records:
        assert r.suite == "chat_target"
        assert "target_start" in r.metadata
        assert "target_char_start" in r.metadata
        assert "target_char_end" in r.metadata
        # target span must be positive
        assert r.metadata["target_char_end"] > r.metadata["target_char_start"]
    for r in prefix_records:
        assert r.suite == "chat_target"
        assert "paired_target_key" in r.metadata
        assert "target_char_start" not in r.metadata


def test_chat_target_token_ids_include_full_response():
    tokenizer = build_tiny_tokenizer()
    task = _FakeChatTask(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q?"},
                    {"role": "assistant", "content": "Hello"},
                ]
            },
        ]
    )
    records = enumerate_chat_target_contexts(
        tokenizer=tokenizer,
        task_object=task,
        max_problems=None,
        task_name="t",
    )
    # The full conversation tokenization must include the response chars, while
    # the paired context-prefix row must not.
    full_record = next(r for r in records if r.metadata["target_real_role"] == "full")
    prefix_record = next(
        r for r in records if r.metadata["target_real_role"] == "context_prefix"
    )
    decoded = tokenizer.decode(list(full_record.token_ids))
    assert "Hello" in decoded
    assert "Hello" not in tokenizer.decode(list(prefix_record.token_ids))


def test_chat_target_max_problems_caps_records():
    """max_problems caps examples; each example emits full + prefix rows."""
    tokenizer = build_tiny_tokenizer()
    task = _FakeChatTask(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q?"},
                    {"role": "assistant", "content": "A"},
                ]
            },
        ]
        * 5
    )
    records = enumerate_chat_target_contexts(
        tokenizer=tokenizer,
        task_object=task,
        max_problems=2,
        task_name="t",
    )
    assert len(records) == 4


def test_core_target_per_task_type_dispatch_mc():
    """MULTIPLE_CHOICE: per-row records (one per choice)."""
    from nanochat.core_eval import TaskMeta, TaskType

    tokenizer = build_tiny_tokenizer()
    data = [
        {"query": "Q1", "choices": ["A", "B", "C"], "gold": 0},
        {"query": "Q2", "choices": ["X", "Y"], "gold": 1},
    ]
    meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    records = enumerate_core_target_contexts(
        tokenizer=tokenizer,
        data=data,
        task_meta=meta,
        seed=1234,
        task_label="mc_test",
        max_per_task=2,
        max_seq_len=128,
    )
    # MC: 3 choices for Q1 + 2 choices for Q2 = 5 rows, each full + prefix.
    assert len(records) == 10
    full_records = [r for r in records if r.metadata["target_real_role"] == "full"]
    prefix_records = [
        r for r in records if r.metadata["target_real_role"] == "context_prefix"
    ]
    assert len(full_records) == 5
    assert len(prefix_records) == 5
    for r in full_records:
        assert r.suite == "core_target"
        assert "target_start" in r.metadata
        assert "target_char_start" in r.metadata
        assert "target_char_end" in r.metadata
        assert r.metadata["target_char_end"] > r.metadata["target_char_start"]
    for r in prefix_records:
        assert r.suite == "core_target"
        assert "paired_target_key" in r.metadata
        assert "target_char_start" not in r.metadata


def test_core_target_skips_entire_example_when_any_choice_row_is_oversized():
    """Dropping one MC row would change the choice set, so skip the example."""
    from nanochat.core_eval import TaskMeta, TaskType

    class PartialLongTokenizer:
        def get_bos_token_id(self):
            return 0

        def __call__(self, prompts, prepend=None):
            rows = []
            for prompt in prompts:
                row = [1] * (80 if prompt.endswith("A") else 8)
                row.append(10 if prompt.endswith("A") else 11)
                if prepend is not None:
                    row = [prepend] + row
                rows.append(row)
            return rows

        def decode(self, ids):
            return "x" * len(list(ids))

    data = [{"query": "Q", "choices": ["A", "B"], "gold": 0}]
    meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )

    records = enumerate_core_target_contexts(
        tokenizer=PartialLongTokenizer(),
        data=data,
        task_meta=meta,
        seed=1234,
        task_label="partial_long_mc",
        max_per_task=1,
        max_seq_len=50,
    )

    assert records == []


def test_core_target_max_per_task_caps_examples():
    from nanochat.core_eval import TaskMeta, TaskType

    tokenizer = build_tiny_tokenizer()
    data = [{"query": f"Q{i}", "choices": ["A", "B"], "gold": 0} for i in range(10)]
    meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    records = enumerate_core_target_contexts(
        tokenizer=tokenizer,
        data=data,
        task_meta=meta,
        seed=1234,
        task_label="mc_test",
        max_per_task=3,
        max_seq_len=128,
    )
    # 3 examples * 2 choices * (full + prefix) = 12 records
    assert len(records) == 12


@pytest.mark.parametrize("max_per_task", [3, 10])
def test_core_target_shuffles_before_truncation_and_fewshot_sampling(max_per_task):
    from nanochat.core_eval import (
        TaskMeta,
        TaskType,
        _sample_fewshot_indices,
        batch_sequences_mc,
        render_prompts_mc,
    )

    tokenizer = build_tiny_tokenizer()
    data = [{"query": f"Question {i}", "choices": [f"A{i}", f"B{i}"], "gold": 0} for i in range(5)]
    meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=1,
        continuation_delimiter=" ",
    )

    records = enumerate_core_target_contexts(
        tokenizer=tokenizer,
        data=data,
        task_meta=meta,
        seed=1234,
        task_label="mc_test",
        max_per_task=max_per_task,
        max_seq_len=1024,
    )

    task_data = list(data)
    random.Random(1337).shuffle(task_data)
    if max_per_task > 0:
        task_data = task_data[:max_per_task]

    expected_token_ids = []
    for idx, item in enumerate(task_data):
        fewshot_idxs = _sample_fewshot_indices(
            len(task_data),
            idx,
            meta.num_fewshot,
            1234 + idx,
        )
        fewshot = [task_data[i] for i in fewshot_idxs]
        prompts = render_prompts_mc(item, meta.continuation_delimiter, fewshot)
        tokens, start_idxs, _end_idxs = batch_sequences_mc(tokenizer, prompts)
        for row, start_idx in zip(tokens, start_idxs, strict=True):
            target_start_safe = max(start_idx - 1, 0)
            expected_token_ids.append(tuple(int(tok) for tok in row))
            expected_token_ids.append(
                tuple(int(tok) for tok in row[: target_start_safe + 1])
            )

    assert [record.token_ids for record in records] == expected_token_ids
