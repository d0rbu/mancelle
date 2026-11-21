"""Direct tests for `tasks.pku_saferlhf.PKUSafeRLHF` loader.

Wave 3C task 6: `tasks/pku_saferlhf.py` had zero direct test coverage.

The loader selects the "safe" response from PKU-SafeRLHF-10K preference pairs,
preferring `response_0` when it's safe, falling back to `response_1`, and
skipping rows where neither is safe. If the skip rate exceeds 50 percent, it
raises `DataLoadError`. These tests monkeypatch `load_dataset` to feed in
synthetic rows that exercise all three branches plus the skip-threshold.
"""

from __future__ import annotations

import pytest

from errors import DataLoadError
from tasks import pku_saferlhf as pku_module
from tasks.pku_saferlhf import PKUSafeRLHF


def _make_row(
    prompt: str,
    response_0: str,
    response_1: str,
    safe_0: bool,
    safe_1: bool,
) -> dict:
    return {
        "prompt": prompt,
        "response_0": response_0,
        "response_1": response_1,
        "is_response_0_safe": safe_0,
        "is_response_1_safe": safe_1,
    }


FIXTURE_ROWS = [
    # response_0 is safe -> use response_0
    _make_row("What is 2+2?", "It is 4.", "Kill everyone.", True, False),
    # response_0 unsafe, response_1 safe -> use response_1
    _make_row("How to bake bread?", "Use poison.", "Mix flour and water.", False, True),
    # both safe -> use response_0 (preferred)
    _make_row("Say hello.", "Hello!", "Hi there!", True, True),
]


@pytest.fixture
def patch_load_dataset(monkeypatch: pytest.MonkeyPatch):
    def _patch(rows: list[dict]) -> None:
        monkeypatch.setattr(pku_module, "load_dataset", lambda *a, **kw: list(rows))

    return _patch


class TestPKUSafeRLHFLoader:
    def test_num_examples_matches_safe_rows(self, patch_load_dataset) -> None:
        patch_load_dataset(FIXTURE_ROWS)
        task = PKUSafeRLHF(split="train")
        assert task.num_examples() == 3

    def test_exact_rendered_output_all_rows(self, patch_load_dataset) -> None:
        """Lock down the user+assistant content selection logic.

        The loader deterministically shuffles the examples with `random.Random(42)`.
        Rather than depending on that shuffle, we check the multiset of
        expected conversations equals the multiset of produced conversations.
        """
        patch_load_dataset(FIXTURE_ROWS)
        task = PKUSafeRLHF(split="train")

        expected_pairs = {
            ("What is 2+2?", "It is 4."),  # row 0 -> response_0
            ("How to bake bread?", "Mix flour and water."),  # row 1 -> response_1
            ("Say hello.", "Hello!"),  # row 2 -> response_0 (preferred)
        }
        got_pairs = {
            (
                task.get_example(i)["messages"][0]["content"],
                task.get_example(i)["messages"][1]["content"],
            )
            for i in range(task.num_examples())
        }
        assert got_pairs == expected_pairs

    def test_schema_shape(self, patch_load_dataset) -> None:
        patch_load_dataset(FIXTURE_ROWS)
        task = PKUSafeRLHF(split="train")
        conv = task.get_example(0)
        assert set(conv.keys()) == {"messages"}
        msgs = conv["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_skips_both_unsafe(self, patch_load_dataset) -> None:
        """Row where neither response is safe is dropped — but below 50% skip threshold."""
        rows = [
            *FIXTURE_ROWS,  # 3 kept
            _make_row("Bad?", "Bad A", "Bad B", False, False),  # 1 skipped
            _make_row("Also bad?", "Bad A", "Bad B", False, False),  # 1 skipped
        ]
        # 2 skipped / 5 total = 40% — under threshold
        patch_load_dataset(rows)
        task = PKUSafeRLHF(split="train")
        assert task.num_examples() == 3

    def test_high_skip_rate_raises_dataloaderror(self, patch_load_dataset) -> None:
        """>50% skip rate must raise DataLoadError loud, not silently drop data."""
        rows = [
            _make_row("Q1", "x", "y", True, True),  # kept
            _make_row("Q2", "x", "y", False, False),  # skipped
            _make_row("Q3", "x", "y", False, False),  # skipped
            _make_row("Q4", "x", "y", False, False),  # skipped
        ]
        # 3/4 = 75% skipped
        patch_load_dataset(rows)
        with pytest.raises(DataLoadError, match="skip rate"):
            PKUSafeRLHF(split="train")

    def test_non_train_split_rejected(self, patch_load_dataset) -> None:
        patch_load_dataset(FIXTURE_ROWS)
        with pytest.raises(AssertionError, match="only has a train split"):
            PKUSafeRLHF(split="validation")

    def test_empty_prompt_raises(self, patch_load_dataset) -> None:
        """Malformed row: empty prompt -> AssertionError (error path, not silent drop)."""
        bad = [_make_row("", "answer", "answer", True, False)]
        patch_load_dataset(bad)
        with pytest.raises(AssertionError, match="Prompt must be a non-empty"):
            PKUSafeRLHF(split="train")

    def test_get_example_postconditions_hold(self, patch_load_dataset) -> None:
        patch_load_dataset(FIXTURE_ROWS)
        task = PKUSafeRLHF(split="train")
        for i in range(task.num_examples()):
            conv = task.get_example(i)
            msgs = conv["messages"]
            assert [m["role"] for m in msgs] == ["user", "assistant"]
            for m in msgs:
                assert isinstance(m["content"], str) and len(m["content"]) > 0

    def test_out_of_range_index_raises(self, patch_load_dataset) -> None:
        patch_load_dataset(FIXTURE_ROWS)
        task = PKUSafeRLHF(split="train")
        with pytest.raises(AssertionError, match="out of range"):
            task.get_example(99)
