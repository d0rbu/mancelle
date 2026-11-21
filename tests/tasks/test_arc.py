"""Direct tests for `tasks.arc.ARC` loader.

Wave 3C task 6: `tasks/arc.py` had zero direct test coverage.

ARC is a multiple-choice task. We build instances via `__new__` and inject a
fake in-memory dataset so we never hit the network. The rendered user prompt
uses `tasks.common.render_mc`, which we also lock down by asserting on the
exact rendered string for a fixed fixture.
"""

from __future__ import annotations

import pytest

from tasks.arc import ARC
from tasks.common import render_mc


class _FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]


def _make_task(rows: list[dict]) -> ARC:
    task = ARC.__new__(ARC)
    task.start = 0
    task.stop = None
    task.step = 1
    task.ds = _FakeDataset(rows)
    return task


FIXTURE_ROWS = [
    {
        "question": "What color is the sky on a clear day?",
        "choices": {"text": ["red", "blue", "green", "yellow"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    },
    {
        "question": "How many legs does a spider have?",
        "choices": {"text": ["six", "eight", "ten"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    },
    {
        "question": "Which planet is closest to the Sun?",
        "choices": {
            "text": ["Earth", "Venus", "Mercury", "Mars"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
    },
]


class TestARCLoader:
    def test_num_examples(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        assert task.num_examples() == 3
        assert len(task) == 3

    def test_eval_type_categorical(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        assert task.eval_type == "categorical"

    def test_exact_rendered_output_row0(self) -> None:
        """Lock in the full rendered conversation for row 0."""
        task = _make_task(FIXTURE_ROWS)
        conv = task.get_example(0)
        expected_user = render_mc(
            "What color is the sky on a clear day?",
            ["A", "B", "C", "D"],
            ["red", "blue", "green", "yellow"],
        )
        assert conv == {
            "messages": [
                {"role": "user", "content": expected_user},
                {"role": "assistant", "content": "B"},
            ],
            "letters": ["A", "B", "C", "D"],
        }
        # Sanity: rendered prompt contains the choice/letter pairs in expected form.
        assert "- red=A\n" in expected_user
        assert "- blue=B\n" in expected_user
        assert expected_user.endswith("Respond only with the letter of the correct answer.")

    def test_all_three_rows_render(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        for i, row in enumerate(FIXTURE_ROWS):
            conv = task.get_example(i)
            assert conv["messages"][-1]["content"] == row["answerKey"]
            choices = row["choices"]
            assert isinstance(choices, dict)
            assert conv["letters"] == choices["label"]

    def test_single_turn_structure(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        for i in range(3):
            msgs = task.get_example(i)["messages"]
            assert [m["role"] for m in msgs] == ["user", "assistant"]

    def test_schema_shape(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        conv = task.get_example(0)
        assert set(conv.keys()) == {"messages", "letters"}

    def test_bad_answer_key_raises(self) -> None:
        """Filter/error path: answerKey not in labels -> AssertionError."""
        bad = [
            {
                "question": "Q?",
                "choices": {"text": ["a", "b"], "label": ["A", "B"]},
                "answerKey": "Z",
            }
        ]
        task = _make_task(bad)
        with pytest.raises(AssertionError, match="ARC answer Z must be one of"):
            task.get_example(0)

    def test_evaluate_correct_and_wrong(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        conv = task.get_example(0)  # correct answer is "B"
        assert task.evaluate(conv, "B") is True
        assert task.evaluate(conv, "A") is False

    def test_evaluate_response_outside_letters_raises(self) -> None:
        task = _make_task(FIXTURE_ROWS)
        conv = task.get_example(0)
        with pytest.raises(AssertionError, match="expected to be one of"):
            task.evaluate(conv, "Z")

    def test_invalid_subset_rejected(self) -> None:
        """ARC-Middle is not a valid subset."""
        with pytest.raises(AssertionError, match="ARC subset"):
            ARC(subset="ARC-Middle", split="train")

    def test_invalid_split_rejected(self) -> None:
        with pytest.raises(AssertionError, match="ARC split"):
            ARC(subset="ARC-Easy", split="dev")
