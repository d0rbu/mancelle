"""Regression test: few-shot sampling must not crash when data is small.

When ``max_per_task`` in ``scripts.eval.base_eval.evaluate_core`` is smaller than
``num_fewshot + 1`` for some task (e.g. tiny smoke-test configs with 3–5
examples per task vs. tasks that request 10-shot context), ``evaluate_example``
previously called ``random.sample(available, num_fewshot)`` and crashed with
``ValueError: Sample larger than population or is negative``.

The fix clamps the request to the actual population size. These tests pin
the clamped contract without booting a model.
"""

from __future__ import annotations

import pytest

from nanochat.core_eval import _sample_fewshot_indices


class TestSampleFewshotIndicesClamps:
    def test_returns_empty_when_num_fewshot_is_zero(self):
        assert _sample_fewshot_indices(
            data_len=10, current_idx=3, num_fewshot=0, seed=1234
        ) == []

    def test_returns_empty_when_data_has_only_current(self):
        """Only one example in the dataset — nothing else to draw from."""
        assert _sample_fewshot_indices(
            data_len=1, current_idx=0, num_fewshot=5, seed=1234
        ) == []

    def test_draws_exactly_num_fewshot_when_population_is_large(self):
        out = _sample_fewshot_indices(
            data_len=100, current_idx=3, num_fewshot=5, seed=1234
        )
        assert len(out) == 5
        assert 3 not in out
        assert all(0 <= i < 100 for i in out)
        assert len(set(out)) == 5  # distinct indices

    def test_clamps_when_num_fewshot_exceeds_available(self):
        """2 available indices but 5 requested — draw 2 without raising."""
        out = _sample_fewshot_indices(
            data_len=3, current_idx=0, num_fewshot=5, seed=1234
        )
        assert len(out) == 2
        assert 0 not in out
        assert set(out) == {1, 2}

    def test_clamps_when_num_fewshot_equals_available(self):
        out = _sample_fewshot_indices(
            data_len=3, current_idx=1, num_fewshot=2, seed=1234
        )
        assert len(out) == 2
        assert 1 not in out
        assert set(out) == {0, 2}

    def test_clamps_when_num_fewshot_one_over_available(self):
        out = _sample_fewshot_indices(
            data_len=4, current_idx=2, num_fewshot=4, seed=1234
        )
        # population = 3 (0,1,3); request = 4 → clamp to 3
        assert len(out) == 3
        assert 2 not in out
        assert set(out) == {0, 1, 3}

    def test_deterministic_given_same_seed(self):
        a = _sample_fewshot_indices(
            data_len=50, current_idx=7, num_fewshot=10, seed=42
        )
        b = _sample_fewshot_indices(
            data_len=50, current_idx=7, num_fewshot=10, seed=42
        )
        assert a == b

    def test_different_seeds_differ(self):
        a = _sample_fewshot_indices(
            data_len=50, current_idx=7, num_fewshot=10, seed=42
        )
        b = _sample_fewshot_indices(
            data_len=50, current_idx=7, num_fewshot=10, seed=43
        )
        assert a != b

    def test_never_includes_current_idx(self):
        for current in (0, 5, 9):
            for nfewshot in (1, 5, 9, 15):
                out = _sample_fewshot_indices(
                    data_len=10, current_idx=current, num_fewshot=nfewshot, seed=1
                )
                assert current not in out

    def test_negative_num_fewshot_returns_empty(self):
        """Defensive: unexpected negative num_fewshot should NOT crash."""
        assert _sample_fewshot_indices(
            data_len=10, current_idx=3, num_fewshot=-1, seed=1
        ) == []

    @pytest.mark.parametrize(
        "data_len,current_idx,num_fewshot",
        [
            (2, 0, 1),
            (2, 0, 10),  # dramatic over-request
            (5, 4, 5),
            (1000, 500, 5),
        ],
    )
    def test_never_raises(self, data_len, current_idx, num_fewshot):
        """Broad smoke — never raise across a range of params."""
        _sample_fewshot_indices(
            data_len=data_len,
            current_idx=current_idx,
            num_fewshot=num_fewshot,
            seed=1,
        )
