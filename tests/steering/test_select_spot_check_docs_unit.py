"""Unit tests for _select_spot_check_docs (slice-3 map §7.2 unit gap)."""

from __future__ import annotations

import torch as t

from steering.validate_features import _select_spot_check_docs


def test_n_zero_returns_empty() -> None:
    counts = t.tensor([10, 20, 30])
    assert _select_spot_check_docs(counts, n_docs=3, n_spot_check=0) == []


def test_n_negative_returns_empty() -> None:
    counts = t.tensor([10, 20, 30])
    assert _select_spot_check_docs(counts, n_docs=3, n_spot_check=-1) == []


def test_n_one_returns_first_doc() -> None:
    counts = t.tensor([10, 20, 30])
    assert _select_spot_check_docs(counts, n_docs=3, n_spot_check=1) == [0]


def test_zero_docs_returns_empty() -> None:
    counts = t.tensor([], dtype=t.int32)
    assert _select_spot_check_docs(counts, n_docs=0, n_spot_check=5) == []


def test_one_doc_returns_first() -> None:
    counts = t.tensor([100])
    assert _select_spot_check_docs(counts, n_docs=1, n_spot_check=5) == [0]


def test_two_docs_returns_first_and_last() -> None:
    counts = t.tensor([100, 200])
    assert _select_spot_check_docs(counts, n_docs=2, n_spot_check=5) == [0, 1]


def test_n_two_returns_first_and_last() -> None:
    counts = t.tensor([100, 200, 300, 400, 500])
    assert _select_spot_check_docs(counts, n_docs=5, n_spot_check=2) == [0, 4]


def test_n_three_includes_first_last_plus_one_sample() -> None:
    """With n=3 and 5 docs, we get first + last + 1 sampled middle doc."""
    counts = t.tensor([100, 200, 300, 400, 500])
    selected = _select_spot_check_docs(counts, n_docs=5, n_spot_check=3)

    assert len(selected) == 3
    assert 0 in selected
    assert 4 in selected
    # The third must be in [1, 2, 3].
    middle = [i for i in selected if i not in (0, 4)]
    assert len(middle) == 1
    assert middle[0] in {1, 2, 3}


def test_n_greater_than_docs_capped_to_docs() -> None:
    counts = t.tensor([100, 200, 300])
    selected = _select_spot_check_docs(counts, n_docs=3, n_spot_check=10)

    assert len(selected) == 3
    assert set(selected) == {0, 1, 2}


def test_output_is_sorted_and_deduplicated() -> None:
    counts = t.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    selected = _select_spot_check_docs(counts, n_docs=8, n_spot_check=5)

    assert selected == sorted(selected)
    assert len(selected) == len(set(selected))


def test_zero_token_doc_can_still_be_picked_via_first_or_last() -> None:
    """A doc with zero tokens is still picked when it lands at index 0 or
    n_docs-1 (because those endpoints are unconditionally included).

    The clamp(min=1.0) on weights also lets zero-token docs participate in
    sampling at lower-bound weight; the behavior is documented (slice-3
    P2-4), not a bug.
    """
    counts = t.tensor([0, 100, 200])
    selected = _select_spot_check_docs(counts, n_docs=3, n_spot_check=2)

    assert 0 in selected
    assert 2 in selected
