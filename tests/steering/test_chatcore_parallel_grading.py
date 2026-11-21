"""Admission and chunk-order contracts for bounded HumanEval grading."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

from steering.eval.sanity_check import run_all
from tasks import humaneval


@pytest.fixture
def admitted_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Give admission an explicit, isolated Linux/fork/single-rank CPU budget."""
    thread = object()
    state = SimpleNamespace(
        environ={},
        sched_getaffinity=lambda _pid: {0, 1, 2, 3},
    )
    monkeypatch.setattr(run_all, "os", state)
    monkeypatch.setattr(run_all, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(run_all, "current_thread", lambda: thread)
    monkeypatch.setattr(run_all, "main_thread", lambda: thread)
    monkeypatch.setattr(
        run_all,
        "multiprocessing",
        SimpleNamespace(
            current_process=lambda: SimpleNamespace(daemon=False),
            get_context=lambda: SimpleNamespace(get_start_method=lambda: "fork"),
        ),
    )
    monkeypatch.setattr(t.distributed, "is_available", lambda: True)
    monkeypatch.setattr(t.distributed, "is_initialized", lambda: False)
    return state


@pytest.mark.parametrize(
    ("ddp", "world_size"),
    [(True, 1), (True, 2), (False, 2), (False, 0)],
)
def test_cpu_admission_is_serial_without_single_rank_ownership(
    admitted_runtime: SimpleNamespace,
    ddp: bool,
    world_size: int,
) -> None:
    assert admitted_runtime.environ == {}
    assert run_all._chatcore_grading_workers(ddp=ddp, world_size=world_size) == 1


@pytest.mark.parametrize(
    "restriction",
    ["platform", "thread", "daemon", "spawn", "forkserver", "distributed"],
)
def test_cpu_admission_rejects_unsafe_process_contexts(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
    restriction: str,
) -> None:
    assert admitted_runtime.environ == {}
    if restriction == "platform":
        monkeypatch.setattr(run_all.sys, "platform", "darwin")
    elif restriction == "thread":
        monkeypatch.setattr(run_all, "current_thread", object)
    elif restriction == "daemon":
        monkeypatch.setattr(
            run_all.multiprocessing,
            "current_process",
            lambda: SimpleNamespace(daemon=True),
        )
    elif restriction in {"spawn", "forkserver"}:
        monkeypatch.setattr(
            run_all.multiprocessing,
            "get_context",
            lambda: SimpleNamespace(get_start_method=lambda: restriction),
        )
    else:
        monkeypatch.setattr(t.distributed, "is_initialized", lambda: True)
    assert run_all._chatcore_grading_workers(ddp=False, world_size=1) == 1


@pytest.mark.parametrize("missing", [True, False])
def test_cpu_admission_requires_readable_affinity(
    admitted_runtime: SimpleNamespace,
    missing: bool,
) -> None:
    def unreadable(_pid: int) -> set[int]:
        raise OSError("affinity unavailable")

    if missing:
        del admitted_runtime.sched_getaffinity
    else:
        admitted_runtime.sched_getaffinity = unreadable
    assert run_all._chatcore_grading_workers(ddp=False, world_size=1) == 1


@pytest.mark.parametrize(
    ("affinity_count", "allocation", "in_slurm", "expected"),
    [
        (0, None, False, 1),
        (1, None, False, 1),
        (2, None, False, 1),
        (3, None, False, 2),
        (64, None, False, 2),
        (64, None, True, 1),
        (64, "", True, 1),
        (64, "unknown", True, 1),
        (64, "3.0", True, 1),
        (64, "-1", True, 1),
        (64, "0", True, 1),
        (64, "2", True, 1),
        (64, "3", True, 2),
        (2, "64", True, 1),
        (3, "64", True, 2),
        (64, "2", False, 1),
    ],
)
def test_cpu_admission_reserves_parent_cpu_and_honors_allocation(
    admitted_runtime: SimpleNamespace,
    affinity_count: int,
    allocation: str | None,
    in_slurm: bool,
    expected: int,
) -> None:
    admitted_runtime.sched_getaffinity = lambda _pid: set(range(affinity_count))
    if in_slurm:
        admitted_runtime.environ["SLURM_JOB_ID"] = "123"
    if allocation is not None:
        admitted_runtime.environ["SLURM_CPUS_PER_TASK"] = allocation
    assert run_all._chatcore_grading_workers(ddp=False, world_size=1) == expected


def test_cpu_admission_allows_build_without_distributed_support(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
) -> None:
    assert admitted_runtime.environ == {}
    monkeypatch.setattr(t.distributed, "is_available", lambda: False)

    def unavailable() -> bool:
        raise AssertionError("unavailable distributed module must not be queried")

    monkeypatch.setattr(t.distributed, "is_initialized", unavailable)
    assert run_all._chatcore_grading_workers(ddp=False, world_size=1) == 2


def _row(row_id: int, prompt_length: int) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "encoded_prompt": [row_id] * prompt_length,
        "messages": [{"role": "user", "content": f"def entry_{row_id}():"}],
        "entry_point": f"entry_{row_id}",
        "test": f"test_{row_id}",
    }


def _expected_program(row_id: int) -> str:
    return f"\n\ncompletion_{row_id}\n\ntest_{row_id}\ncheck(entry_{row_id})"


class _HumanTask(humaneval.HumanEval):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class _OtherTask:
    def __init__(self, rows: list[dict[str, Any]], events: list) -> None:
        self.rows = rows
        self.events = events

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def evaluate(self, conversation: dict, completion: str) -> bool:
        row_id = conversation["row_id"]
        assert completion == f"completion_{row_id}"
        self.events.append(("other", row_id))
        return row_id % 2 == 0


class _Model:
    config = SimpleNamespace(sequence_len=1024)

    @staticmethod
    def get_device() -> t.device:
        return t.device("cpu")


class _Tokenizer:
    @staticmethod
    def decode(tokens: list[int]) -> str:
        assert len(tokens) == 1
        return f"completion_{tokens[0]}"


def _install_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    human_rows: list[dict[str, Any]],
    math_rows: list[dict[str, Any]],
    spelling_rows: list[dict[str, Any]],
    batch_size: int,
    ddp: bool = False,
    world_size: int = 1,
) -> list:
    events: list = []
    tasks = {
        "HumanEval": _HumanTask(human_rows),
        "GSM8K": _OtherTask(math_rows, events),
        "SpellingBee": _OtherTask(spelling_rows, events),
    }
    monkeypatch.setattr(run_all, "get_dist_info", lambda: (ddp, 0, 0, world_size))
    monkeypatch.setattr(run_all, "_build_chatcore_generative_task", tasks.__getitem__)
    monkeypatch.setattr(
        run_all,
        "render_chat_completion",
        lambda _tokenizer, row: SimpleNamespace(token_ids=row["encoded_prompt"]),
    )
    monkeypatch.setattr(run_all, "_chatcore_cross_task_batch_size", lambda *_args: batch_size)
    monkeypatch.setattr(
        run_all,
        "_stage_linear_weights_for_greedy_batches",
        lambda *_args, **_kwargs: nullcontext(),
    )

    def generate(_model, _tokenizer, prompts, **_kwargs) -> list[list[int]]:
        events.append(("generate", [prompt[0] for prompt in prompts]))
        return [[prompt[0]] for prompt in prompts]

    def grade_one(program: str) -> bool:
        events.append(("scalar", program))
        return program != _expected_program(11)

    def grade_batch(programs: list[str], *, max_workers: int) -> list[bool]:
        events.append(("batch_start", programs, max_workers))
        results = [program != _expected_program(11) for program in programs]
        events.append(("batch_drained", programs))
        return results

    monkeypatch.setattr(run_all, "generate_greedy_token_batch", generate)
    monkeypatch.setattr(humaneval, "python_program_passes", grade_one)
    monkeypatch.setattr(humaneval, "python_programs_pass", grade_batch)
    return events


def test_mixed_chunk_preserves_rows_scores_and_drains_before_next_generation(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
) -> None:
    assert admitted_runtime.environ == {}
    events = _install_runner(
        monkeypatch,
        human_rows=[_row(10, 3), _row(11, 1)],
        math_rows=[_row(20, 2), _row(21, 5)],
        spelling_rows=[_row(30, 4), _row(31, 6)],
        batch_size=4,
    )

    result = run_all._run_chatcore_generative_tasks(_Model(), _Tokenizer(), max_problems=2)

    assert result == {"HumanEval": 0.5, "GSM8K": 0.5, "SpellingBee": 0.5}
    human_programs = [_expected_program(11), _expected_program(10)]
    assert events == [
        ("generate", [11, 20, 10, 30]),
        ("other", 20),
        ("other", 30),
        ("batch_start", human_programs, 2),
        ("batch_drained", human_programs),
        ("generate", [21, 31]),
        ("other", 21),
        ("other", 31),
    ]


def test_singleton_human_chunks_and_nonhuman_chunks_keep_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
) -> None:
    assert admitted_runtime.environ == {}
    events = _install_runner(
        monkeypatch,
        human_rows=[_row(10, 1), _row(11, 4)],
        math_rows=[_row(20, 2), _row(21, 5)],
        spelling_rows=[_row(30, 3), _row(31, 6)],
        batch_size=2,
    )

    result = run_all._run_chatcore_generative_tasks(_Model(), _Tokenizer(), max_problems=2)

    assert result == {"HumanEval": 0.5, "GSM8K": 0.5, "SpellingBee": 0.5}
    assert [event for event in events if event[0] == "scalar"] == [
        ("scalar", _expected_program(10)),
        ("scalar", _expected_program(11)),
    ]
    assert [event for event in events if event[0].startswith("batch")] == []
    assert [event for event in events if event[0] == "generate"] == [
        ("generate", [10, 20]),
        ("generate", [30, 11]),
        ("generate", [21, 31]),
    ]


def test_ddp_grades_serially_and_collects_exact_counts_after_grading(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
) -> None:
    assert admitted_runtime.environ == {}
    events = _install_runner(
        monkeypatch,
        human_rows=[_row(10, 3), _row(11, 1)],
        math_rows=[_row(20, 2), _row(21, 5)],
        spelling_rows=[_row(30, 4), _row(31, 6)],
        batch_size=6,
        ddp=True,
        world_size=2,
    )

    def reduce_counts(counts: t.Tensor, *, op: Any) -> None:
        assert op == t.distributed.ReduceOp.SUM
        assert counts.tolist() == [[1, 2], [1, 2], [1, 2]]
        assert len([event for event in events if event[0] == "scalar"]) == 2
        assert len([event for event in events if event[0] == "other"]) == 4
        events.append(("reduce",))

    monkeypatch.setattr(t.distributed, "all_reduce", reduce_counts)
    result = run_all._run_chatcore_generative_tasks(_Model(), _Tokenizer(), max_problems=2)

    assert result == {"HumanEval": 0.5, "GSM8K": 0.5, "SpellingBee": 0.5}
    assert events[-1] == ("reduce",)
    assert [event for event in events if event[0].startswith("batch")] == []


@pytest.mark.parametrize("verdicts", [[], [True], [True, False, True]])
def test_batch_verdict_count_mismatch_fails_before_next_generation(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
    verdicts: list[bool],
) -> None:
    assert admitted_runtime.environ == {}
    events = _install_runner(
        monkeypatch,
        human_rows=[_row(10, 1), _row(11, 2)],
        math_rows=[_row(20, 3)],
        spelling_rows=[_row(30, 4)],
        batch_size=2,
    )
    monkeypatch.setattr(humaneval, "python_programs_pass", lambda *_args, **_kwargs: verdicts)

    with pytest.raises(ValueError, match="zip"):
        run_all._run_chatcore_generative_tasks(_Model(), _Tokenizer(), max_problems=2)

    assert events == [("generate", [10, 11])]


@pytest.mark.parametrize("completion_count", [0, 1, 3])
def test_generation_count_mismatch_cannot_silently_drop_or_add_human_rows(
    monkeypatch: pytest.MonkeyPatch,
    admitted_runtime: SimpleNamespace,
    completion_count: int,
) -> None:
    assert admitted_runtime.environ == {}
    events = _install_runner(
        monkeypatch,
        human_rows=[_row(10, 1), _row(11, 2)],
        math_rows=[_row(20, 3)],
        spelling_rows=[_row(30, 4)],
        batch_size=2,
    )
    monkeypatch.setattr(
        run_all,
        "generate_greedy_token_batch",
        lambda *_args, **_kwargs: [[10]] * completion_count,
    )

    with pytest.raises(ValueError, match="zip"):
        run_all._run_chatcore_generative_tasks(_Model(), _Tokenizer(), max_problems=2)

    assert events == []
