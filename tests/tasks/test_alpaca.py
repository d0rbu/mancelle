"""Direct tests for `tasks.alpaca.Alpaca` loader.

Wave 3C task 6: `tasks/alpaca.py` had zero direct test coverage.

Because `Alpaca.__init__` calls `datasets.load_dataset("yahma/alpaca-cleaned", ...)`
which requires network access and a HuggingFace cache, these tests build instances
via `__new__` and inject a synthetic in-memory dataset. This exercises the entire
`get_example` rendering path without hitting the network.
"""

from __future__ import annotations

import pytest

from tasks.alpaca import Alpaca


class _FakeDataset:
    """Minimal stand-in for a HuggingFace `Dataset` — supports `len` and indexing."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]


def _make_task(rows: list[dict]) -> Alpaca:
    task = Alpaca.__new__(Alpaca)
    # Task base class fields
    task.start = 0
    task.stop = None
    task.step = 1
    task.ds = _FakeDataset(rows)
    task.length = len(rows)
    return task


FIXTURE_ROWS = [
    {
        "instruction": "Translate to French",
        "input": "Hello world",
        "output": "Bonjour le monde",
    },
    {
        "instruction": "What is 2 plus 2?",
        "input": "",
        "output": "4",
    },
    {
        "instruction": "Name a color",
        "input": "   ",  # whitespace only — should be treated as empty
        "output": "blue",
    },
]


class TestAlpacaLoader:
    def test_num_examples_matches_fixture(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        assert task.num_examples() == 3
        assert len(task) == 3

    def test_exact_rendered_output_all_three_rows(self) -> None:
        """Lock in the exact rendered conversation for all 3 fixture rows."""
        task = _make_task(FIXTURE_ROWS)

        expected = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Translate to French\n\nHello world",
                    },
                    {"role": "assistant", "content": "Bonjour le monde"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 2 plus 2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Name a color"},
                    {"role": "assistant", "content": "blue"},
                ]
            },
        ]
        got = [task.get_example(i) for i in range(3)]
        assert got == expected

    def test_schema_shape(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        conv = task.get_example(0)
        assert set(conv.keys()) == {"messages"}
        msgs = conv["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_single_turn_structure(self) -> None:
        """Alpaca is single-turn: exactly 1 user + 1 assistant message."""
        task = _make_task(FIXTURE_ROWS)
        for i in range(3):
            msgs = task.get_example(i)["messages"]
            roles = [m["role"] for m in msgs]
            assert roles == ["user", "assistant"]

    def test_empty_instruction_raises(self) -> None:
        """Filter/error path: empty instruction violates precondition assertion."""
        task = _make_task([{"instruction": "", "input": "", "output": "hi"}])
        with pytest.raises(AssertionError, match="instruction must be a non-empty"):
            task.get_example(0)

    def test_empty_output_raises(self) -> None:
        task = _make_task([{"instruction": "Say hi", "input": "", "output": ""}])
        with pytest.raises(AssertionError, match="output must be a non-empty"):
            task.get_example(0)

    def test_non_string_instruction_raises(self) -> None:
        task = _make_task([{"instruction": 123, "input": "", "output": "hi"}])
        with pytest.raises(AssertionError, match="instruction must be a non-empty"):
            task.get_example(0)
