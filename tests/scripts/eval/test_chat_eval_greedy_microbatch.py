from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest
import torch as t

from scripts.eval.chat_eval import _shard_greedy_work_chunks, run_generative_eval
from steering.eval.sanity_check import run_all


class _Tokenizer:
    def render_for_completion(self, conversation) -> list[int]:
        return list(conversation["encoded_prompt"])

    def encode_special(self, name: str) -> int:
        assert name == "<assistant_end>"
        return 9

    def decode(self, tokens: list[int]) -> str:
        return "".join({50: "A", 51: "B", 52: "C"}.get(token, "") for token in tokens)

    def encode(self, text: str) -> list[int]:
        return [{"A": 50, "B": 51, "C": 52}[text]]

    def get_bos_token_id(self) -> int:
        return 0


class _Task:
    eval_type = "generative"

    def __init__(self, prompts: list[list[int]]) -> None:
        self.items = [{"encoded_prompt": prompt} for prompt in prompts]
        self.completions: list[str] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def evaluate(self, _conversation, completion: str) -> bool:
        self.completions.append(completion)
        return completion in {"A", "B", "C"}


class _Model:
    is_target_aware_encdec = False
    config = SimpleNamespace(sequence_len=32)

    def get_device(self) -> t.device:
        return t.device("cpu")


class _ExactEngine:
    def __init__(self, generated_tokens: list[int]) -> None:
        self.generated_tokens = iter(generated_tokens)
        self.calls: list[list[int]] = []

    def generate_batch(self, prompt_tokens: list[int], **_kwargs):
        self.calls.append(prompt_tokens)
        result = [*prompt_tokens, next(self.generated_tokens)]
        return [result], [[0] * len(result)]


def test_equal_length_chunks_are_formed_before_rank_sharding() -> None:
    items = [
        ({"index": 0}, [1], 4),
        ({"index": 1}, [2, 3], 4),
        ({"index": 2}, [4], 4),
        ({"index": 3}, [5, 6], 4),
    ]

    rank_zero = _shard_greedy_work_chunks(
        items,
        microbatch_size=2,
        rank=0,
        world_size=2,
    )
    rank_one = _shard_greedy_work_chunks(
        items,
        microbatch_size=2,
        rank=1,
        world_size=2,
    )

    assert [[item[0]["index"] for item in chunk] for chunk in rank_zero] == [[0, 2]]
    assert [[item[0]["index"] for item in chunk] for chunk in rank_one] == [[1, 3]]


def test_cuda_ragged_chunks_join_mixed_prompt_lengths() -> None:
    items = [
        ({"index": 0}, [1], 4),
        ({"index": 1}, [2, 3], 4),
        ({"index": 2}, [4, 5, 6], 4),
    ]

    chunks = _shard_greedy_work_chunks(
        items,
        microbatch_size=3,
        rank=0,
        world_size=1,
        allow_mixed_prompt_lengths=True,
    )

    assert [[item[0]["index"] for item in chunk] for chunk in chunks] == [[0, 1, 2]]


def test_microbatches_pairs_and_routes_trailing_singleton_through_greedy_path() -> None:
    task = _Task([[1], [2], [4, 5]])
    engine = _ExactEngine([])

    with (
        patch("scripts.eval.chat_eval.get_dist_info", return_value=(False, 0, 0, 1)),
        patch(
            "scripts.eval.chat_eval.generate_greedy_token_batch",
            side_effect=[[[50], [51]], [[52]]],
        ) as generate_batch,
    ):
        accuracy = run_generative_eval(
            task,
            _Tokenizer(),
            _Model(),
            engine,
            num_samples=1,
            max_new_tokens=4,
            temperature=0.0,
            top_k=50,
            greedy_microbatch_size=2,
        )

    assert accuracy == 1.0
    assert task.completions == ["A", "B", "C"]
    assert generate_batch.call_args_list == [
        (
            (ANY, ANY, [[1], [2]]),
            {"max_new_tokens": 4, "enable_python_tools": False},
        ),
        (
            (ANY, ANY, [[4, 5]]),
            {"max_new_tokens": 4, "enable_python_tools": False},
        ),
    ]
    assert engine.calls == []


def test_tool_signal_restarts_every_row_through_exact_engine() -> None:
    task = _Task([[1], [2]])
    engine = _ExactEngine([50, 51])

    with (
        patch("scripts.eval.chat_eval.get_dist_info", return_value=(False, 0, 0, 1)),
        patch(
            "scripts.eval.chat_eval.generate_greedy_token_batch",
            return_value=None,
        ),
    ):
        accuracy = run_generative_eval(
            task,
            _Tokenizer(),
            _Model(),
            engine,
            num_samples=1,
            max_new_tokens=4,
            temperature=0.0,
            top_k=50,
            greedy_microbatch_size=2,
        )

    assert accuracy == 1.0
    assert task.completions == ["A", "B"]
    assert engine.calls == [[1], [2]]


def test_tool_batching_forwards_explicit_helper_opt_in() -> None:
    task = _Task([[1], [2]])
    engine = _ExactEngine([])

    with (
        patch("scripts.eval.chat_eval.get_dist_info", return_value=(False, 0, 0, 1)),
        patch(
            "scripts.eval.chat_eval.generate_greedy_token_batch",
            return_value=[[50], [51]],
        ) as generate_batch,
    ):
        accuracy = run_generative_eval(
            task,
            _Tokenizer(),
            _Model(),
            engine,
            num_samples=1,
            max_new_tokens=4,
            temperature=0.0,
            top_k=50,
            greedy_microbatch_size=2,
            greedy_python_tools=True,
        )

    assert accuracy == 1.0
    generate_batch.assert_called_once_with(
        ANY,
        ANY,
        [[1], [2]],
        max_new_tokens=4,
        enable_python_tools=True,
    )
    assert engine.calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 0.7, "num_samples": 1}, "temperature=0.0"),
        ({"temperature": 0.0, "num_samples": 2}, "num_samples=1"),
    ],
)
def test_microbatch_rejects_non_greedy_or_multi_sample_calls(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        run_generative_eval(
            _Task([[1], [2]]),
            _Tokenizer(),
            _Model(),
            _ExactEngine([]),
            max_new_tokens=4,
            top_k=50,
            greedy_microbatch_size=2,
            **kwargs,
        )


def test_python_tools_require_the_greedy_microbatch_path() -> None:
    with pytest.raises(ValueError, match="greedy_microbatch_size > 1"):
        run_generative_eval(
            _Task([[1]]),
            _Tokenizer(),
            _Model(),
            _ExactEngine([]),
            num_samples=1,
            max_new_tokens=4,
            temperature=0.0,
            top_k=50,
            greedy_microbatch_size=1,
            greedy_python_tools=True,
        )


def test_standalone_chatcore_batches_supported_generative_tasks(monkeypatch) -> None:
    generated_batches: list[tuple[list[list[int]], dict[str, object]]] = []
    staged_batch_specs: list[tuple[int, int, bool]] = []
    tasks = {
        "HumanEval": _Task([[10], [10, 11, 12]]),
        "GSM8K": _Task([[20], [20, 21]]),
        "SpellingBee": _Task([[30], [30, 31, 32, 33]]),
    }

    monkeypatch.setattr(run_all, "Engine", lambda *_args: object())
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        run_all,
        "_run_chatcore_categorical_tasks",
        lambda *_args, **_kwargs: {
            "ARC-Easy": 0.0,
            "ARC-Challenge": 0.0,
            "MMLU": 0.0,
        },
    )
    monkeypatch.setattr(
        run_all,
        "_build_chatcore_generative_task",
        lambda task_name: tasks[task_name],
    )

    def fake_generate_batch(_model, _tokenizer, prompts, **kwargs):
        generated_batches.append((prompts, kwargs))
        return [[50] for _prompt in prompts]

    monkeypatch.setattr(
        run_all,
        "generate_greedy_token_batch",
        fake_generate_batch,
    )

    def fake_stage_batches(_model, *, batch_specs, dtype):
        assert dtype == t.float32
        staged_batch_specs.extend(batch_specs)
        return nullcontext(False)

    monkeypatch.setattr(
        run_all,
        "_stage_linear_weights_for_greedy_batches",
        fake_stage_batches,
    )

    model = _Model()
    model.config = SimpleNamespace(sequence_len=2048)
    result = run_all.test_chatcore(model, _Tokenizer(), nullcontext(), max_problems=2)

    assert [len(prompts) for prompts, _kwargs in generated_batches] == [4, 2]
    assert staged_batch_specs == [(4, 514, True), (2, 516, True)]
    assert {
        (kwargs["max_new_tokens"], kwargs["enable_python_tools"])
        for _prompts, kwargs in generated_batches
    } == {(512, True)}
    assert result["per_task_scores"] == {
        "ARC-Easy": 0.0,
        "ARC-Challenge": 0.0,
        "MMLU": 0.0,
        "GSM8K": 1.0,
        "HumanEval": 1.0,
        "SpellingBee": 1.0,
    }


def test_standalone_chatcore_collects_categorical_cuda_tensors_before_generative(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        run_all,
        "_run_chatcore_categorical_tasks",
        lambda *_args, **_kwargs: (
            events.append("categorical")
            or {"ARC-Easy": 0.0, "ARC-Challenge": 0.0, "MMLU": 0.0}
        ),
    )
    monkeypatch.setattr(
        run_all,
        "_run_chatcore_generative_tasks",
        lambda *_args, **_kwargs: (
            events.append("generative")
            or {"HumanEval": 0.0, "GSM8K": 0.0, "SpellingBee": 0.0}
        ),
    )
    monkeypatch.setattr(run_all.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(
        t.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    model = _Model()
    model.get_device = lambda: t.device("cuda")  # ty: ignore[invalid-assignment]

    run_all.test_chatcore(model, _Tokenizer(), nullcontext(), max_problems=1)

    assert events == ["categorical", "collect", "empty_cache", "generative"]


def test_standalone_chatcore_pools_categorical_tasks_before_sharding(
    monkeypatch,
) -> None:
    class CategoricalTask:
        def __init__(self, prefix: int) -> None:
            self.items = [
                {
                    "encoded_prompt": [prefix],
                    "letters": ["A", "B"],
                    "answer": "A",
                },
                {
                    "encoded_prompt": [prefix, prefix + 1, prefix + 2],
                    "letters": ["A", "B"],
                    "answer": "A",
                },
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            return self.items[index]

        def evaluate(self, conversation, prediction: str) -> bool:
            return prediction == conversation["answer"]

    tasks = {
        "ARC-Easy": CategoricalTask(10),
        "ARC-Challenge": CategoricalTask(20),
        "MMLU": CategoricalTask(30),
    }
    calls: list[tuple[list[list[int]], list[list[int]], int]] = []
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        run_all,
        "_build_chatcore_categorical_task",
        lambda task_name: tasks[task_name],
    )

    def fake_predict(
        _model,
        *,
        prompt_token_ids: list[list[int]],
        answer_token_ids: list[list[int]],
        pad_token_id: int,
    ) -> list[int]:
        calls.append((prompt_token_ids, answer_token_ids, pad_token_id))
        return [0] * len(prompt_token_ids)

    monkeypatch.setattr(run_all, "predict_categorical_token_batch", fake_predict)

    result = run_all._run_chatcore_categorical_tasks(
        _Model(),
        _Tokenizer(),
        max_problems=2,
    )

    assert result == {"ARC-Easy": 1.0, "ARC-Challenge": 1.0, "MMLU": 1.0}
    assert len(calls) == 1
    prompts, answer_ids, pad_token_id = calls[0]
    assert [len(prompt) for prompt in prompts] == [1, 1, 1, 3, 3, 3]
    assert answer_ids == [[50, 51]] * 6
    assert pad_token_id == 0


def test_standalone_chatcore_splits_categorical_chunks_at_padding_budget(
    monkeypatch,
) -> None:
    class CategoricalTask:
        def __init__(self, prefix: int) -> None:
            self.items = [
                {"encoded_prompt": [prefix], "letters": ["A"], "answer": "A"},
                {
                    "encoded_prompt": [prefix, prefix, prefix],
                    "letters": ["A"],
                    "answer": "A",
                },
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            return self.items[index]

        def evaluate(self, conversation, prediction: str) -> bool:
            return prediction == conversation["answer"]

    tasks = {
        "ARC-Easy": CategoricalTask(10),
        "ARC-Challenge": CategoricalTask(20),
        "MMLU": CategoricalTask(30),
    }
    chunk_sizes: list[int] = []
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        run_all,
        "_build_chatcore_categorical_task",
        lambda task_name: tasks[task_name],
    )
    monkeypatch.setattr(run_all, "_CHATCORE_MAX_CATEGORICAL_PADDED_TOKENS", 6)

    def fake_predict(_model, *, prompt_token_ids, **_kwargs) -> list[int]:
        chunk_sizes.append(len(prompt_token_ids))
        return [0] * len(prompt_token_ids)

    monkeypatch.setattr(run_all, "predict_categorical_token_batch", fake_predict)

    result = run_all._run_chatcore_categorical_tasks(
        _Model(),
        _Tokenizer(),
        max_problems=2,
    )

    assert result == {"ARC-Easy": 1.0, "ARC-Challenge": 1.0, "MMLU": 1.0}
    assert chunk_sizes == [3, 2, 1]


def test_cross_task_chatcore_restarts_tool_free_signal_sequentially(monkeypatch) -> None:
    tasks = {
        "HumanEval": _Task([[10]]),
        "GSM8K": _Task([[20]]),
        "SpellingBee": _Task([[30]]),
    }
    exact_engine = _ExactEngine([50, 51, 52])
    monkeypatch.setattr(run_all, "Engine", lambda *_args: exact_engine)
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        run_all,
        "_build_chatcore_generative_task",
        lambda task_name: tasks[task_name],
    )
    monkeypatch.setattr(
        run_all,
        "generate_greedy_token_batch",
        lambda *_args, **_kwargs: None,
    )
    model = _Model()
    model.config = SimpleNamespace(
        sequence_len=2048,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=4,
    )

    result = run_all._run_chatcore_generative_tasks(
        model,
        _Tokenizer(),
        max_problems=1,
    )

    assert result == {"HumanEval": 1.0, "GSM8K": 1.0, "SpellingBee": 1.0}
    assert exact_engine.calls == [[10], [20], [30]]


def test_cross_task_chatcore_retries_expandable_batch_in_safe_chunks(
    monkeypatch,
) -> None:
    tasks = {
        "HumanEval": _Task([[10], [11, 12]]),
        "GSM8K": _Task([[20, 21, 22], [23, 24, 25, 26]]),
        "SpellingBee": _Task([[30, 31, 32, 33, 34], [35, 36, 37, 38, 39, 40]]),
    }
    generated: list[tuple[int, int | None]] = []
    released_failed_batches: list[int] = []

    class FailedBatch:
        def __del__(self) -> None:
            released_failed_batches.append(1)
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        run_all,
        "_build_chatcore_generative_task",
        lambda task_name: tasks[task_name],
    )
    monkeypatch.setattr(run_all, "_CHATCORE_MAX_CROSS_TASK_BATCH_SIZE", 2)
    monkeypatch.setattr(
        run_all,
        "_chatcore_cross_task_batch_size",
        lambda *_args, **_kwargs: 6,
    )
    monkeypatch.setattr(
        run_all,
        "_supports_expandable_ragged_cache",
        lambda _device: True,
    )
    monkeypatch.setattr(
        run_all,
        "_initial_expandable_ragged_cache_len",
        lambda *_args, **_kwargs: 16,
    )
    monkeypatch.setattr(
        run_all,
        "_stage_linear_weights_for_greedy_batches",
        lambda *_args, **_kwargs: nullcontext(False),
    )

    def fake_generate(_model, _tokenizer, prompts, **kwargs):
        initial_cache_len = kwargs["ragged_cache_initial_len"]
        generated.append((len(prompts), initial_cache_len))
        if initial_cache_len is not None:
            failed_batch = FailedBatch()
            assert failed_batch is not None
            raise run_all.GreedyBatchCapacityError("probe")
        assert released_failed_batches == [1]
        return [[50] for _prompt in prompts]

    monkeypatch.setattr(run_all, "generate_greedy_token_batch", fake_generate)
    model = _Model()
    model.config = SimpleNamespace(sequence_len=2048)

    result = run_all._run_chatcore_generative_tasks(
        model,
        _Tokenizer(),
        max_problems=2,
    )

    assert generated == [(6, 16), (2, None), (2, None), (2, None)]
    assert result == {"HumanEval": 1.0, "GSM8K": 1.0, "SpellingBee": 1.0}


@pytest.mark.parametrize(
    ("max_problems", "world_size", "expected"),
    [
        (50, 1, 50),
        (50, 2, 25),
        (50, 8, 7),
        (2, 8, 2),
    ],
)
def test_chatcore_greedy_microbatch_targets_one_chunk_per_rank(
    max_problems: int,
    world_size: int,
    expected: int,
) -> None:
    assert run_all._chatcore_greedy_microbatch_size(max_problems, world_size) == expected


@pytest.mark.parametrize(
    ("max_problems", "world_size", "expected"),
    [
        (50, 1, 100),
        (50, 2, 50),
        (50, 8, 14),
        (2, 8, 4),
        (100, 1, 100),
    ],
)
def test_chatcore_cross_task_batch_pools_two_task_chunks(
    max_problems: int,
    world_size: int,
    expected: int,
) -> None:
    assert run_all._chatcore_cross_task_batch_size(max_problems, world_size) == expected


@pytest.mark.parametrize(
    ("max_problems", "world_size", "expected"),
    [
        (50, 1, 150),
        (50, 2, 75),
        (50, 8, 21),
        (2, 8, 6),
        (100, 1, 150),
    ],
)
def test_chatcore_cuda_cross_task_batch_pools_three_task_chunks(
    max_problems: int,
    world_size: int,
    expected: int,
) -> None:
    assert (
        run_all._chatcore_cross_task_batch_size(
            max_problems,
            world_size,
            t.device("cuda"),
        )
        == expected
    )


@pytest.mark.parametrize(
    ("total_rows", "world_size", "expected"),
    [(150, 1, 150), (150, 2, 75), (150, 8, 19), (300, 1, 150)],
)
def test_chatcore_categorical_batch_targets_one_chunk_per_rank(
    total_rows: int,
    world_size: int,
    expected: int,
) -> None:
    assert run_all._chatcore_categorical_batch_size(total_rows, world_size) == expected
