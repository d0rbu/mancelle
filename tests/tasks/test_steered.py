"""Unit tests for the steered SFT primitives.

Covers the JSONL loader's rejection surface, the exactly-even interleaving of
``SteeredTaskMixture`` (and its fingerprint identity), and the largest-remainder
apportionment used by the balanced mixture.
"""

from __future__ import annotations

import json
import random
import re
from itertools import pairwise
from pathlib import Path

import pytest

from errors import DataLoadError
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft_schema import SFTSplit
from tasks.common import Task, TaskMixture
from tasks.steered import (
    SteeredConversations,
    SteeredTaskMixture,
    balanced_standard_caps,
)


class _FakeTask(Task):
    """A base task with deterministic, identifiable examples."""

    def __init__(self, count: int, tag: str) -> None:
        super().__init__()
        self.count = count
        self.tag = tag

    def num_examples(self) -> int:
        return self.count

    def get_example(self, index: int) -> dict:
        return {"tag": self.tag, "index": index}


def _row(index: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"prompt {index}"},
        {"role": "assistant", "content": f"answer {index}"},
    ]


def write_steered(path: Path, rows: int, *, blank_lines: bool = False) -> Path:
    lines = [json.dumps(_row(index)) for index in range(rows)]
    if blank_lines:
        lines = ["", *lines, "   ", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def steered_file(tmp_path: Path) -> Path:
    return write_steered(tmp_path / "steered.jsonl", 4)


# ---------------------------------------------------------------------------
# SteeredConversations
# ---------------------------------------------------------------------------


class TestSteeredConversationsLoader:
    def test_loads_every_row_and_returns_messages_shape(self, steered_file: Path) -> None:
        task = SteeredConversations(str(steered_file))
        assert len(task) == 4
        assert task.num_examples() == 4
        assert task.get_example(2) == {"messages": _row(2)}
        assert task[0] == {"messages": _row(0)}

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = write_steered(tmp_path / "blanks.jsonl", 3, blank_lines=True)
        assert len(SteeredConversations(str(path))) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.jsonl"
        pattern = re.escape(f"steered: {missing} not found")
        with pytest.raises(DataLoadError, match=pattern):
            SteeredConversations(str(missing))

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("\n  \n", encoding="utf-8")
        with pytest.raises(DataLoadError, match=r"holds zero steered conversations"):
            SteeredConversations(str(path))

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ("{not json", r"line 2 is not valid JSON"),
            ('{"role": "user"}', r"line 2 must be a JSON list of exactly 2 messages"),
            (json.dumps([_row(0)[0]]), r"line 2 must be a JSON list of exactly 2 messages"),
            (
                json.dumps([*_row(0), _row(0)[1]]),
                r"line 2 must be a JSON list of exactly 2 messages",
            ),
            (json.dumps(["user", "assistant"]), r"line 2 message 0 must be an object"),
            (
                json.dumps([{"role": "assistant", "content": "a"}, _row(0)[1]]),
                r"line 2 message 0 must have role='user', got 'assistant'",
            ),
            (
                json.dumps([_row(0)[0], {"role": "user", "content": "a"}]),
                r"line 2 message 1 must have role='assistant', got 'user'",
            ),
            (
                json.dumps([{"role": "user", "content": 7}, _row(0)[1]]),
                r"line 2 message 0 content must be a string, got 7",
            ),
            (
                json.dumps([_row(0)[0], {"role": "assistant", "content": ["a"]}]),
                r"line 2 message 1 content must be a string",
            ),
            (
                json.dumps([_row(0)[0], {"role": "assistant", "content": "   "}]),
                r"line 2 assistant content must be non-blank",
            ),
        ],
    )
    def test_invalid_row_raises_naming_the_line(
        self, tmp_path: Path, payload: str, reason: str
    ) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(f"{json.dumps(_row(0))}\n{payload}\n", encoding="utf-8")
        pattern = f"{re.escape(f'steered: {path} ')}{reason}"
        with pytest.raises(DataLoadError, match=pattern):
            SteeredConversations(str(path))

    def test_extra_message_fields_are_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.jsonl"
        messages = [
            {"role": "user", "content": "hi", "feature_id": 12},
            {"role": "assistant", "content": "yo", "scale": 0.5},
        ]
        path.write_text(json.dumps(messages) + "\n", encoding="utf-8")
        assert SteeredConversations(str(path)).get_example(0) == {"messages": messages}

    def test_object_lines_with_a_messages_key_load_like_bare_lists(
        self, tmp_path: Path
    ) -> None:
        """The exported dataset shape ``{"id": ..., "messages": [...]}``."""
        as_lists = tmp_path / "lists.jsonl"
        as_objects = tmp_path / "objects.jsonl"
        rows = [_row(index) for index in range(5)]
        as_lists.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        as_objects.write_text(
            "".join(
                json.dumps({"id": f"{i}_{i * 7}", "messages": r, "concept": "x"}) + "\n"
                for i, r in enumerate(rows)
            ),
            encoding="utf-8",
        )
        lists_task = SteeredConversations(str(as_lists))
        objects_task = SteeredConversations(str(as_objects))
        assert len(objects_task) == len(lists_task) == 5
        for index in range(5):
            assert objects_task.get_example(index) == lists_task.get_example(index)
            # Provenance keys stay OUT of the example: only the messages train.
            assert set(objects_task.get_example(index)) == {"messages"}

    def test_list_and_object_lines_may_be_mixed_in_one_file(self, tmp_path: Path) -> None:
        """The shape is decided per line, so a file may carry both."""
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            json.dumps(_row(0)) + "\n" + json.dumps({"id": "x", "messages": _row(1)}) + "\n",
            encoding="utf-8",
        )
        task = SteeredConversations(str(path))
        assert [task.get_example(i) for i in range(2)] == [
            {"messages": _row(0)},
            {"messages": _row(1)},
        ]

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            (
                json.dumps({"id": "7", "conversation": _row(0)}),
                r"line 2 must be a JSON list of exactly 2 messages .*or an object whose "
                r"'messages' key holds that list.*, got \{'id': '7'",
            ),
            (
                json.dumps({"messages": {"role": "user"}}),
                r"line 2 must be a JSON list of exactly 2 messages",
            ),
            (
                json.dumps({"messages": [_row(0)[0]]}),
                r"line 2 must be a JSON list of exactly 2 messages",
            ),
            (
                json.dumps({"messages": [_row(0)[0], {"role": "assistant", "content": " "}]}),
                r"line 2 assistant content must be non-blank",
            ),
        ],
    )
    def test_invalid_object_lines_raise_naming_the_line(
        self, tmp_path: Path, payload: str, reason: str
    ) -> None:
        path = tmp_path / "bad_objects.jsonl"
        path.write_text(f"{json.dumps(_row(0))}\n{payload}\n", encoding="utf-8")
        pattern = f"{re.escape(f'steered: {path} ')}{reason}"
        with pytest.raises(DataLoadError, match=pattern):
            SteeredConversations(str(path))


# ---------------------------------------------------------------------------
# SteeredTaskMixture
# ---------------------------------------------------------------------------


class TestSteeredTaskMixture:
    def test_is_a_task_mixture_with_the_steered_task_last(self, steered_file: Path) -> None:
        steered = SteeredConversations(str(steered_file))
        base = [_FakeTask(6, "a"), _FakeTask(6, "b")]
        mixture = SteeredTaskMixture(base, steered)

        assert isinstance(mixture, TaskMixture)
        assert mixture.tasks == [*base, steered]
        assert mixture.lengths == [6, 6, 4]
        assert mixture.num_conversations == 16
        assert len(mixture) == 16
        assert len(mixture.index_map) == 16
        assert mixture.steered_task_index == 2

    def test_positions_are_exactly_evenly_spaced_and_increasing(self, steered_file: Path) -> None:
        steered = SteeredConversations(str(steered_file))
        mixture = SteeredTaskMixture([_FakeTask(20, "a")], steered)
        positions = sorted(mixture.steered_positions)

        # Hand-computed: L=24, M=4 -> (2k+1)*24 // 8 = 24//8, 72//8, 120//8, 168//8.
        assert positions == [3, 9, 15, 21]
        assert all(later > earlier for earlier, later in pairwise(positions))
        assert all(position < len(mixture) for position in positions)
        assert all(mixture.is_steered(position) for position in positions)
        assert not any(
            mixture.is_steered(position)
            for position in range(len(mixture))
            if position not in mixture.steered_positions
        )

    @pytest.mark.parametrize(
        ("base_len", "steered_len", "expected"),
        [
            # Every pair below divides INEXACTLY, so floor/round/ceil disagree.
            # L=12, M=4: (2k+1)*12 // 8 = 12//8, 36//8, 60//8, 84//8.
            (8, 4, [1, 4, 7, 10]),
            # L=13, M=4: 13//8, 39//8, 65//8, 91//8.
            (9, 4, [1, 4, 8, 11]),
            # L=17, M=5: 17//10, 51//10, 85//10, 119//10, 153//10.
            (12, 5, [1, 5, 8, 11, 15]),
            # L=7, M=2: 7//4, 21//4.
            (5, 2, [1, 5]),
        ],
    )
    def test_positions_are_the_floored_midpoints_on_inexact_divisions(
        self, tmp_path: Path, base_len: int, steered_len: int, expected: list[int]
    ) -> None:
        """Pin the ROUNDING, not just the spacing.

        Every position literal here is hand-computed, never produced by the
        production expression. The pairs are deliberately inexact so a
        ``//`` -> ``round()`` (or ``ceil``) mutation moves at least one row;
        the ``rounded`` guard below asserts each case really discriminates.
        """
        total = base_len + steered_len
        rounded = [
            round((2 * k + 1) * total / (2 * steered_len)) for k in range(steered_len)
        ]
        assert rounded != expected, "this (N, M) pair does not discriminate floor"

        steered = SteeredConversations(str(write_steered(tmp_path / "s.jsonl", steered_len)))
        mixture = SteeredTaskMixture([_FakeTask(base_len, "a")], steered)

        assert sorted(mixture.steered_positions) == expected
        assert [
            position
            for position, pair in enumerate(mixture.index_map)
            if pair[0] == mixture.steered_task_index
        ] == expected

    def test_one_steered_row_per_aligned_block_when_ratio_is_one_fifth(
        self, tmp_path: Path
    ) -> None:
        steered = SteeredConversations(str(write_steered(tmp_path / "s.jsonl", 10)))
        mixture = SteeredTaskMixture([_FakeTask(50, "a")], steered)
        assert len(mixture) == 60
        flags = [mixture.is_steered(position) for position in range(len(mixture))]
        # EVERY window of six, not only the aligned blocks: a left-aligned
        # [0, 6, 12, ...] scheme and a within-block jitter both satisfy the
        # aligned-block equality while breaking the owner's property.
        assert all(sum(flags[start : start + 6]) == 1 for start in range(len(flags) - 5))
        positions = sorted(mixture.steered_positions)
        assert positions == [3, 9, 15, 21, 27, 33, 39, 45, 51, 57]
        assert {later - earlier for earlier, later in pairwise(positions)} == {6}

    def test_equal_counts_alternate_with_steered_on_odd_positions(self, tmp_path: Path) -> None:
        steered = SteeredConversations(str(write_steered(tmp_path / "s.jsonl", 8)))
        mixture = SteeredTaskMixture([_FakeTask(8, "a")], steered)
        assert sorted(mixture.steered_positions) == list(range(1, 16, 2))
        assert [pair[0] == mixture.steered_task_index for pair in mixture.index_map] == [
            position % 2 == 1 for position in range(16)
        ]

    def test_base_and_steered_rows_keep_their_seeded_orders(self, steered_file: Path) -> None:
        steered = SteeredConversations(str(steered_file))
        base = [_FakeTask(6, "a"), _FakeTask(6, "b")]
        mixture = SteeredTaskMixture(base, steered)

        expected_base = TaskMixture([_FakeTask(6, "a"), _FakeTask(6, "b")]).index_map
        expected_steered = list(range(4))
        random.Random(42).shuffle(expected_steered)

        actual_base = [pair for pair in mixture.index_map if pair[0] != mixture.steered_task_index]
        actual_steered = [
            pair[1] for pair in mixture.index_map if pair[0] == mixture.steered_task_index
        ]
        assert actual_base == expected_base
        assert actual_steered == expected_steered
        # Non-vacuity: the seeded steered order is NOT the identity here.
        assert expected_steered != list(range(4))

    def test_get_example_routes_through_the_index_map(self, steered_file: Path) -> None:
        steered = SteeredConversations(str(steered_file))
        mixture = SteeredTaskMixture([_FakeTask(8, "a")], steered)
        for position in range(len(mixture)):
            task_index, local_index = mixture.index_map[position]
            assert mixture[position] == mixture.tasks[task_index][local_index]
        steered_position = min(mixture.steered_positions)
        assert "messages" in mixture[steered_position]

    def test_two_constructions_are_identical(self, steered_file: Path) -> None:
        first = SteeredTaskMixture([_FakeTask(9, "a")], SteeredConversations(str(steered_file)))
        second = SteeredTaskMixture([_FakeTask(9, "a")], SteeredConversations(str(steered_file)))
        assert first.index_map == second.index_map
        assert first.steered_positions == second.steered_positions

    def test_zero_steered_rows_raises(self, steered_file: Path) -> None:
        empty = SteeredConversations(str(steered_file))
        empty.stop = 0
        assert len(empty) == 0
        with pytest.raises(ValueError, match="requires at least one steered conversation"):
            SteeredTaskMixture([_FakeTask(4, "a")], empty)


class TestSteeredTaskMixtureFingerprint:
    def _fingerprint(self, task) -> dict:
        return compute_mixture_fingerprint(
            task, mixture_name="standard_steered", split=SFTSplit.TRAIN
        )

    def test_fingerprint_enumerates_every_task_with_the_steered_one_last(
        self, steered_file: Path
    ) -> None:
        steered = SteeredConversations(str(steered_file))
        mixture = SteeredTaskMixture([_FakeTask(6, "a"), _FakeTask(5, "b")], steered)
        fingerprint = self._fingerprint(mixture)

        assert fingerprint["tasks"] == [
            {"class_name": "_FakeTask", "length": 6},
            {"class_name": "_FakeTask", "length": 5},
            {"class_name": "SteeredConversations", "length": 4},
        ]
        assert fingerprint["len_mixture"] == 15

    def test_fingerprint_differs_from_the_plain_mixture_and_tracks_the_row_count(
        self, tmp_path: Path
    ) -> None:
        four = SteeredConversations(str(write_steered(tmp_path / "four.jsonl", 4)))
        five = SteeredConversations(str(write_steered(tmp_path / "five.jsonl", 5)))
        base = _FakeTask(12, "a")

        steered_fingerprint = self._fingerprint(SteeredTaskMixture([base], four))
        plain_fingerprint = self._fingerprint(TaskMixture([base, four]))
        assert steered_fingerprint["len_mixture"] == plain_fingerprint["len_mixture"]
        assert steered_fingerprint["index_map_sha256"] != plain_fingerprint["index_map_sha256"]
        assert (
            self._fingerprint(SteeredTaskMixture([base], five))["index_map_sha256"]
            != steered_fingerprint["index_map_sha256"]
        )

    def test_same_length_different_content_is_the_accepted_fingerprint_limit(
        self, tmp_path: Path
    ) -> None:
        """Owner-accepted limit: identity is row COUNT plus order, not content.

        The conversion records no content hash, so two steered JSONLs of the
        same length are indistinguishable to the fingerprint gate. Operators
        must point the trainer at the same file the parquet was converted from.
        """
        original = write_steered(tmp_path / "a.jsonl", 4)
        other = tmp_path / "b.jsonl"
        other.write_text(
            "\n".join(
                json.dumps(
                    [
                        {"role": "user", "content": f"totally different {index}"},
                        {"role": "assistant", "content": f"other answer {index}"},
                    ]
                )
                for index in range(4)
            )
            + "\n",
            encoding="utf-8",
        )
        base = _FakeTask(12, "a")
        assert self._fingerprint(
            SteeredTaskMixture([base], SteeredConversations(str(original)))
        ) == self._fingerprint(SteeredTaskMixture([base], SteeredConversations(str(other))))


# ---------------------------------------------------------------------------
# balanced_standard_caps
# ---------------------------------------------------------------------------


class TestBalancedStandardCaps:
    def test_worked_apportionment_example(self) -> None:
        # 3 seats over [17, 3, 3, 6000, 7000, 200000, 80000] (total 293023):
        # floors are all 0 except 200000 -> 2; the single remainder seat goes to
        # 80000 (fraction 0.819), which beats 7000's 0.072.
        assert balanced_standard_caps([17, 3, 3, 6000, 7000, 200000, 80000], 3) == [
            0,
            0,
            0,
            0,
            0,
            2,
            1,
        ]

    @pytest.mark.parametrize("target", [1, 2, 7, 13, 40])
    def test_caps_sum_to_target_and_never_exceed_their_length(self, target: int) -> None:
        lengths = [40, 0, 7, 13, 1]
        caps = balanced_standard_caps(lengths, target)
        assert sum(caps) == target
        assert all(0 <= cap <= length for cap, length in zip(caps, lengths, strict=True))

    def test_ties_go_to_the_lower_index(self) -> None:
        # Four equal lengths, three seats: every remainder ties, so the first
        # three indices win.
        assert balanced_standard_caps([10, 10, 10, 10], 3) == [1, 1, 1, 0]
        assert balanced_standard_caps([10, 10, 10, 10], 2) == [1, 1, 0, 0]

    def test_exact_total_gives_every_row(self) -> None:
        assert balanced_standard_caps([3, 5, 2], 10) == [3, 5, 2]

    def test_target_above_total_raises(self) -> None:
        with pytest.raises(ValueError, match=r"target=11 exceeds the total available rows \(10\)"):
            balanced_standard_caps([3, 5, 2], 11)

    @pytest.mark.parametrize("target", [0, -1])
    def test_non_positive_target_raises(self, target: int) -> None:
        with pytest.raises(ValueError, match=r"target must be >= 1"):
            balanced_standard_caps([3, 5, 2], target)
