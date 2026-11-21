"""The FP loss under a SPLIT prediction group, against a pure-Python oracle.

The steered SFT producer widens what ``seg_doc_id`` means: it is no longer "one
id per packed conversation" but "one id per prediction group", and a steered
conversation contributes two groups. No loss code changed for that — this file
is the evidence that none had to.

For every ``(loss_mode, loss_fn)`` pair it recomputes the loss with a naive
``for b, s`` loop that applies the documented gate (same group, both ``>= 0``,
source has tokens, target valid) and each mode's own weighting, and compares
against ``compute_feature_pred_loss``. It then pins the two consequences that
matter for steering: the first assistant segment of a split row receives
EXACTLY zero gradient, and self-conditioning's rollout gate marks that slot as
not live — both flipping when the split is removed.
"""

from __future__ import annotations

import math

import pytest
import torch as t

from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    compute_feature_pred_loss,
    segment_lengths,
    source_segment_masks,
)
from steering.conditional.encoder_decoder.selfcond_training import continuation_mask

N_SEG = 6
N_SAE = 5

# Row 0 packs one steered conversation: segment 0 is the shared empty slot,
# segments 1-2 are its user turn, segments 3-4 its (perturbed) response, and
# segment 5 is batch padding. Row 1 packs two plain conversations, so its only
# boundary is a conversation boundary. Segment 4 of row 1 is a
# prediction-target-only segment: a real target with no decoder tokens.
SPLIT_DOC_IDS = [
    [-1, 0, 0, 1, 1, -1],
    [-1, 0, 0, 1, 1, -1],
]
# The same rows without the steered split: the response segments stay in the
# user turn's group, so segment 2 may predict segment 3.
UNSPLIT_DOC_IDS = [
    [-1, 0, 0, 0, 0, -1],
    [-1, 0, 0, 1, 1, -1],
]
FIRST_ASSISTANT_SEG = 3


def _random_case(seed: int) -> tuple[t.Tensor, t.Tensor]:
    """Return ``(logits, seg_map)`` for a small batch with realistic geometry.

    ``seg_map`` is non-decreasing per row, leaves segment 5 empty (padding) and,
    in row 1, also leaves segment 4 empty so the prediction-target-only branch
    of ``_target_segment_validity`` is exercised.
    """
    generator = t.Generator().manual_seed(seed)
    seg_map = t.tensor(
        [
            [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ],
        dtype=t.int64,
    )
    logits = t.randn(2, seg_map.shape[1], N_SAE, generator=generator) * 2.0
    return logits, seg_map


def _make_targets(seed: int) -> t.Tensor:
    generator = t.Generator().manual_seed(seed + 1000)
    return t.rand(2, N_SEG, N_SAE, generator=generator)


def _per_vector_loss(
    logit_vector: t.Tensor,
    target_vector: t.Tensor,
    loss_fn: FeaturePredLossFn,
) -> t.Tensor:
    """Mean over the feature dimension of the configured per-feature loss."""
    if loss_fn == FeaturePredLossFn.BCE:
        per_feature = t.nn.functional.binary_cross_entropy_with_logits(
            logit_vector, target_vector, reduction="none"
        )
    else:
        per_feature = (t.sigmoid(logit_vector) - target_vector) ** 2
    return per_feature.mean()


def _oracle_gate(
    seg_map: t.Tensor,
    seg_doc_id: t.Tensor,
) -> tuple[list[list[bool]], list[list[float]]]:
    """The documented gate and the effective target lengths, by hand."""
    B, n_seg = seg_doc_id.shape
    lengths = [[0.0] * n_seg for _ in range(B)]
    for b in range(B):
        for position in range(seg_map.shape[1]):
            lengths[b][int(seg_map[b, position])] += 1.0
    effective = [
        [
            lengths[b][s] if lengths[b][s] > 0 else float(int(seg_doc_id[b, s]) >= 0)
            for s in range(n_seg)
        ]
        for b in range(B)
    ]
    valid = [[False] * n_seg for _ in range(B)]
    for b in range(B):
        for s in range(n_seg - 1):
            target_valid = lengths[b][s + 1] > 0 or int(seg_doc_id[b, s + 1]) >= 0
            valid[b][s] = (
                int(seg_doc_id[b, s]) == int(seg_doc_id[b, s + 1])
                and int(seg_doc_id[b, s]) >= 0
                and lengths[b][s] > 0
                and target_valid
            )
    return valid, effective


def _oracle_loss(
    logits: t.Tensor,
    seg_map: t.Tensor,
    targets: t.Tensor,
    seg_doc_id: t.Tensor,
    loss_mode: FeaturePredLossMode,
    loss_fn: FeaturePredLossFn,
) -> t.Tensor:
    """Recompute the loss with an explicit ``for b, s`` loop."""
    B, T = seg_map.shape
    n_seg = seg_doc_id.shape[1]
    valid, effective = _oracle_gate(seg_map, seg_doc_id)

    def target_of(b: int, s: int) -> t.Tensor:
        return targets[b, min(s + 1, n_seg - 1)]

    if loss_mode is FeaturePredLossMode.FULL_SEGMENT:
        total = t.zeros((), dtype=logits.dtype)
        count = 0
        for b in range(B):
            for position in range(T):
                s = int(seg_map[b, position])
                if valid[b][s]:
                    total = total + _per_vector_loss(logits[b, position], target_of(b, s), loss_fn)
                    count += 1
        return total / max(count, 1)

    # Every weighted mode shares one normaliser: the total number of target
    # tokens reachable through the gate.
    total_target_tokens = max(
        sum(effective[b][s + 1] for b in range(B) for s in range(n_seg - 1) if valid[b][s]),
        1.0,
    )

    if loss_mode is FeaturePredLossMode.LAST_TOKEN:
        total = t.zeros((), dtype=logits.dtype)
        for b in range(B):
            for position in range(T):
                s = int(seg_map[b, position])
                is_last = position == T - 1 or int(seg_map[b, position + 1]) != s
                if is_last and valid[b][s]:
                    weight = effective[b][min(s + 1, n_seg - 1)] / total_target_tokens
                    total = total + weight * _per_vector_loss(
                        logits[b, position], target_of(b, s), loss_fn
                    )
        return total

    total = t.zeros((), dtype=logits.dtype)
    for b in range(B):
        for s in range(n_seg - 1):
            if not valid[b][s]:
                continue
            weight = effective[b][s + 1] / total_target_tokens
            positions = [p for p in range(T) if int(seg_map[b, p]) == s]
            if loss_mode is FeaturePredLossMode.BEST_TOKEN:
                pooled = min(
                    (_per_vector_loss(logits[b, p], target_of(b, s), loss_fn) for p in positions),
                    key=float,
                    default=t.zeros((), dtype=logits.dtype),
                )
            else:
                stacked = t.stack([logits[b, p] for p in positions])
                pooled = _per_vector_loss(stacked.amax(dim=0), target_of(b, s), loss_fn)
            total = total + weight * pooled
    return total


ALL_MODES = list(FeaturePredLossMode)
ALL_FNS = list(FeaturePredLossFn)


@pytest.mark.parametrize("doc_ids", [SPLIT_DOC_IDS, UNSPLIT_DOC_IDS], ids=["split", "unsplit"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_gate_matches_the_documented_predicate_exactly(
    doc_ids: list[list[int]], seed: int
) -> None:
    """``source_segment_masks`` IS the documented gate, bit for bit."""
    _logits, seg_map = _random_case(seed)
    seg_doc_id = t.tensor(doc_ids, dtype=t.int64)
    expected, _effective = _oracle_gate(seg_map, seg_doc_id)

    masks = source_segment_masks(segment_lengths(seg_map, N_SEG), seg_doc_id)

    assert t.equal(masks.valid_source_seg, t.tensor(expected, dtype=t.bool))


@pytest.mark.parametrize("loss_fn", ALL_FNS)
@pytest.mark.parametrize("loss_mode", ALL_MODES)
@pytest.mark.parametrize("doc_ids", [SPLIT_DOC_IDS, UNSPLIT_DOC_IDS], ids=["split", "unsplit"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_mode_agrees_with_the_python_oracle(
    loss_mode: FeaturePredLossMode,
    loss_fn: FeaturePredLossFn,
    doc_ids: list[list[int]],
    seed: int,
) -> None:
    """A split group is just another group: no mode needed a change."""
    logits, seg_map = _random_case(seed)
    targets = _make_targets(seed)
    seg_doc_id = t.tensor(doc_ids, dtype=t.int64)

    actual = compute_feature_pred_loss(logits, seg_map, targets, seg_doc_id, loss_mode, loss_fn)
    expected = _oracle_loss(logits, seg_map, targets, seg_doc_id, loss_mode, loss_fn)

    t.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("loss_fn", ALL_FNS)
@pytest.mark.parametrize("loss_mode", ALL_MODES)
def test_the_split_and_unsplit_losses_actually_differ(
    loss_mode: FeaturePredLossMode, loss_fn: FeaturePredLossFn
) -> None:
    """Non-vacuity for the oracle: the two doc-id rows are not the same input."""
    logits, seg_map = _random_case(0)
    targets = _make_targets(0)

    split = compute_feature_pred_loss(
        logits,
        seg_map,
        targets,
        t.tensor(SPLIT_DOC_IDS, dtype=t.int64),
        loss_mode,
        loss_fn,
    )
    unsplit = compute_feature_pred_loss(
        logits,
        seg_map,
        targets,
        t.tensor(UNSPLIT_DOC_IDS, dtype=t.int64),
        loss_mode,
        loss_fn,
    )

    assert not math.isclose(float(split), float(unsplit), rel_tol=1e-4)


@pytest.mark.parametrize("loss_fn", ALL_FNS)
@pytest.mark.parametrize("loss_mode", ALL_MODES)
def test_the_split_zeroes_the_gradient_on_the_first_assistant_target(
    loss_mode: FeaturePredLossMode, loss_fn: FeaturePredLossFn
) -> None:
    """The point of the whole feature, stated as a gradient.

    With the split, nothing in the loss can pull the head toward the steered
    response's features from the user turn — EXACTLY zero, not merely small.
    Without it, the same slot carries real gradient.
    """
    logits, seg_map = _random_case(0)

    def target_gradient(doc_ids: list[list[int]]) -> t.Tensor:
        targets = _make_targets(0).requires_grad_(True)
        compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            t.tensor(doc_ids, dtype=t.int64),
            loss_mode,
            loss_fn,
        ).backward()
        assert targets.grad is not None
        return targets.grad[0, FIRST_ASSISTANT_SEG]

    assert float(target_gradient(SPLIT_DOC_IDS).abs().max()) == 0.0
    assert float(target_gradient(UNSPLIT_DOC_IDS).abs().max()) > 0.0


def test_the_rollout_gate_follows_the_split_without_being_told_about_it() -> None:
    """``continuation_mask`` is the target-indexed shift of the loss gate.

    Self-conditioning therefore stops treating the first assistant slot as a
    continuation for free — no steering-aware code in the rollout.
    """
    _logits, seg_map = _random_case(0)

    split = continuation_mask(seg_map, t.tensor(SPLIT_DOC_IDS, dtype=t.int64))
    unsplit = continuation_mask(seg_map, t.tensor(UNSPLIT_DOC_IDS, dtype=t.int64))

    assert not bool(split[0, FIRST_ASSISTANT_SEG])
    assert bool(unsplit[0, FIRST_ASSISTANT_SEG])
    # Row 1 is identical in both, so nothing else moved.
    assert t.equal(split[1], unsplit[1])
