"""T-U3: deterministic, disjoint top-k/bottom-l action selection."""

from __future__ import annotations

import torch as t

from steering.conditional.encoder_decoder.selfcond_training import select_topk_bottoml
from steering.conditional.sampling import select_top_k_by_percentile


def _reference(
    scores: t.Tensor,
    positive_counts: t.Tensor,
    negative_counts: t.Tensor,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    feature_ids = t.arange(scores.shape[1], dtype=t.int64)
    feature_mags = t.zeros_like(feature_ids, dtype=t.float32)
    percentile_table = t.zeros(scores.shape[1], 1, dtype=t.float64)
    positives: list[t.Tensor] = []
    negatives: list[t.Tensor] = []
    for row in range(scores.shape[0]):
        pos_ids, _ = select_top_k_by_percentile(
            feature_ids,
            feature_mags,
            percentile_table,
            int(positive_counts[row]),
            override_weights=scores[row],
            rng=None,
        )
        # Independent negative oracle (reviewer finding: the previous
        # +inf-mask-then-stable-argsort mirror shared its core mechanism with
        # select_topk_bottoml itself). Plain python instead: sort every feature
        # ascending by (score, id), drop the positive SET, keep the first l.
        positive_set = set(pos_ids.tolist())
        ascending = sorted(
            range(scores.shape[1]),
            key=lambda fid: (float(scores[row, fid]), fid),
        )
        survivors = [fid for fid in ascending if fid not in positive_set]
        positives.append(pos_ids)
        negatives.append(t.tensor(survivors[: int(negative_counts[row])], dtype=t.int64))
    return positives, negatives


def test_selection_matches_independent_loop_reference_including_ties() -> None:
    scores = t.tensor(
        [
            [0.8, 0.2, 0.8, 0.1, 0.5, 0.5, 0.3],
            [0.4, 0.4, 0.9, 0.2, 0.2, 0.8, 0.1],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=t.float32,
    )
    k = t.tensor([3, 1, 4], dtype=t.int64)
    negative_counts = t.tensor([2, 0, 3], dtype=t.int64)
    expected_positive, expected_negative = _reference(scores, k, negative_counts)

    actions = select_topk_bottoml(scores, k, negative_counts)

    assert t.equal(actions.k, k)
    assert t.equal(actions.l, negative_counts)
    for row in range(scores.shape[0]):
        assert t.equal(actions.pos_ids[row, : k[row]], expected_positive[row])
        assert t.equal(actions.neg_ids[row, : negative_counts[row]], expected_negative[row])
        assert (actions.pos_ids[row, k[row] :] == -1).all()
        assert (actions.neg_ids[row, negative_counts[row] :] == -1).all()


def test_all_equal_scores_pin_ascending_ties_and_disjointness() -> None:
    scores = t.ones(2, 10, dtype=t.float32)
    k = t.tensor([3, 5], dtype=t.int64)
    negative_counts = t.tensor([4, 2], dtype=t.int64)

    actions = select_topk_bottoml(scores, k, negative_counts)

    for row in range(2):
        expected_positive = t.arange(int(k[row]), dtype=t.int64)
        expected_negative = t.arange(int(k[row]), int(k[row] + negative_counts[row]), dtype=t.int64)
        actual_positive = actions.pos_ids[row, : k[row]]
        actual_negative = actions.neg_ids[row, : negative_counts[row]]
        assert t.equal(actual_positive, expected_positive)
        assert t.equal(actual_negative, expected_negative)
        assert set(actual_positive.tolist()).isdisjoint(actual_negative.tolist())


def test_zero_negative_counts_and_heterogeneous_counts_are_supported() -> None:
    scores = t.tensor([[0.1, 0.9, 0.3, 0.7], [0.8, 0.2, 0.6, 0.4]], dtype=t.float32)
    k = t.tensor([1, 3], dtype=t.int64)
    negative_counts = t.zeros(2, dtype=t.int64)

    first = select_topk_bottoml(scores, k, negative_counts)
    second = select_topk_bottoml(scores, k, negative_counts)

    assert first.neg_ids.shape == (2, 0)
    assert t.equal(first.pos_ids, second.pos_ids)
    assert t.equal(first.neg_ids, second.neg_ids)
    assert first.pos_ids.tolist() == [[1, -1, -1], [0, 2, 3]]
