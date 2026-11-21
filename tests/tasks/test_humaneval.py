"""Direct tests for `tasks.humaneval.HumanEval` loader.

Wave 3C task 6: `tasks/humaneval.py` had zero direct test coverage.

We test the loader (`get_example`) and the pure helpers (`extract_imports`,
`extract_program`). Loader tests use an injected fake dataset to avoid network.
"""

from __future__ import annotations

import pytest

from tasks import humaneval
from tasks.humaneval import HumanEval, extract_imports, extract_program


class _FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]


def _make_task(rows: list[dict]) -> HumanEval:
    task = HumanEval.__new__(HumanEval)
    task.start = 0
    task.stop = None
    task.step = 1
    task.ds = _FakeDataset(rows)
    return task


FIXTURE_ROWS = [
    {
        "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
        "canonical_solution": "    return a + b",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
    },
    {
        "prompt": "import math\n\ndef sq(x):\n    \"\"\"Square.\"\"\"\n",
        "canonical_solution": "    return x * x",
        "entry_point": "sq",
        "test": "def check(candidate):\n    assert candidate(4) == 16",
    },
    {
        "prompt": "def greet(name):\n    \"\"\"Hi.\"\"\"\n",
        "canonical_solution": "    return 'hi ' + name",
        "entry_point": "greet",
        "test": "def check(candidate):\n    assert candidate('x') == 'hi x'",
    },
]


class TestHumanEvalLoader:
    def test_num_examples(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        assert task.num_examples() == 3
        assert len(task) == 3

    def test_eval_type_generative(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        assert task.eval_type == "generative"

    def test_exact_rendered_output_all_rows(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        got = [task.get_example(i) for i in range(3)]
        expected = [
            {
                "messages": [
                    {"role": "user", "content": FIXTURE_ROWS[i]["prompt"]},
                    {
                        "role": "assistant",
                        "content": f"{FIXTURE_ROWS[i]['prompt']}\n{FIXTURE_ROWS[i]['canonical_solution']}",
                    },
                ],
                "entry_point": FIXTURE_ROWS[i]["entry_point"],
                "test": FIXTURE_ROWS[i]["test"],
            }
            for i in range(3)
        ]
        assert got == expected

    def test_schema_shape(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        conv = task.get_example(0)
        assert set(conv.keys()) == {"messages", "entry_point", "test"}

    def test_single_turn_structure(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        for i in range(3):
            msgs = task.get_example(i)["messages"]
            assert [m["role"] for m in msgs] == ["user", "assistant"]


class TestExtractImports:
    def test_extracts_import_and_from(self) -> None:
        prompt = "import math\nfrom typing import List\n\ndef foo():\n    pass\n"
        assert extract_imports(prompt) == "import math\nfrom typing import List"

    def test_stops_at_first_non_import_non_comment(self) -> None:
        prompt = "import os\ndef f(): pass\nimport sys\n"
        # Should stop at the `def` line; the trailing `import sys` is NOT captured.
        assert extract_imports(prompt) == "import os"

    def test_skips_comments_but_keeps_imports(self) -> None:
        prompt = "# header\nimport re\n# another\nfrom x import y\n"
        assert extract_imports(prompt) == "import re\nfrom x import y"

    def test_empty_when_no_imports(self) -> None:
        assert extract_imports("def f(): pass\n") == ""


class TestExtractProgram:
    def test_extracts_from_python_fence(self) -> None:
        completion = "Here's the code:\n```python\ndef f():\n    return 1\n```\nDone."
        assert extract_program(completion) == "def f():\n    return 1"

    def test_extracts_from_plain_fence(self) -> None:
        completion = "```\nx = 1\n```"
        assert extract_program(completion) == "x = 1"

    def test_first_block_wins(self) -> None:
        completion = "```python\nA = 1\n```\nand\n```python\nB = 2\n```"
        assert extract_program(completion) == "A = 1"

    def test_no_fence_returns_whole_stripped(self) -> None:
        completion = "   def f(): return 1   "
        assert extract_program(completion) == "def f(): return 1"


def test_scalar_and_batch_grading_preserve_exact_programs_and_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task([
        {
            "prompt": "# heading\nimport math\nfrom typing import List\n\ndef sq(x):\n",
            "canonical_solution": "    return x * x",
            "entry_point": "sq",
            "test": "def check(candidate):\n    assert candidate(4) == 16\n",
        },
        FIXTURE_ROWS[0],
    ])
    conversations = [task[0], task[1], task[0]]
    completions = [
        "Explanation\n```python\ndef sq(x):\n    return math.pow(x, 2)\n```\nIgnored",
        "  def add(a, b): return a - b  ",
        "Explanation\n```python\ndef sq(x):\n    return math.pow(x, 2)\n```\nIgnored",
    ]
    squared = (
        "import math\nfrom typing import List\n\n"
        "def sq(x):\n    return math.pow(x, 2)\n\n"
        "def check(candidate):\n    assert candidate(4) == 16\n\ncheck(sq)"
    )
    added = (
        "\n\ndef add(a, b): return a - b\n\n"
        "def check(candidate):\n    assert candidate(2, 3) == 5\ncheck(add)"
    )
    expected = [squared, added, squared]
    scalar_programs: list[str] = []
    batch_calls: list[tuple[list[str], int]] = []

    def grade_one(program: str) -> bool:
        scalar_programs.append(program)
        return program == squared

    def grade_batch(programs: list[str], *, max_workers: int) -> list[bool]:
        batch_calls.append((programs, max_workers))
        return [program == squared for program in programs]

    monkeypatch.setattr(humaneval, "python_program_passes", grade_one)
    monkeypatch.setattr(humaneval, "python_programs_pass", grade_batch)

    assert [
        task.evaluate(conversation, completion)
        for conversation, completion in zip(conversations, completions, strict=True)
    ] == [True, False, True]
    assert task.evaluate_batch(conversations, completions, max_workers=2) == [True, False, True]
    assert scalar_programs == expected
    assert batch_calls == [(expected, 2)]


@pytest.mark.parametrize("completion_count", [0, 2])
def test_batch_rejects_row_count_mismatch_before_starting_any_grader(
    monkeypatch: pytest.MonkeyPatch,
    completion_count: int,
) -> None:
    task = _make_task(FIXTURE_ROWS)
    called = []
    monkeypatch.setattr(
        humaneval,
        "python_programs_pass",
        lambda *_args, **_kwargs: called.append(True),
    )

    with pytest.raises(ValueError, match="zip"):
        task.evaluate_batch([task[0]], ["completion"] * completion_count, max_workers=2)

    assert called == []
