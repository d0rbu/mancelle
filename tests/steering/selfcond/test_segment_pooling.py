"""T-U4 plus Task-3 helper coverage for per-segment FP-logit pooling."""

from __future__ import annotations

from dataclasses import fields
from typing import cast

import pytest
import torch as t

from errors import SelfCondTrainingError
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.feature_prediction import pool_fp_logits
from steering.conditional.encoder_decoder.selfcond_training import (
    pool_fp_logits_per_segment as _pool_fp_logits_per_segment,
)
from steering.refined_types import PositiveInt
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch, tiny_head_model


def pool_fp_logits_per_segment(
    fp_logits: t.Tensor,
    seg_map: t.Tensor,
    n_seg: int,
    mode: FeaturePredLossMode,
) -> t.Tensor:
    """Pass raw test values through the production beartype boundary."""
    return _pool_fp_logits_per_segment(
        fp_logits,
        seg_map,
        cast(PositiveInt, n_seg),
        mode,
    )


@pytest.mark.parametrize(
    "mode",
    [
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.FULL_SEGMENT,
        FeaturePredLossMode.LAST_TOKEN,
    ],
)
def test_segment_pooling_equals_per_slice_reference_for_all_supported_modes(
    mode: FeaturePredLossMode,
) -> None:
    batch = make_batch([[3, 1], [2]], T=12, F=4, n_sae=16, seed=3)
    generator = t.Generator().manual_seed(19)
    # Small integers keep addition exact, so the assertion isolates pooling
    # semantics rather than reduction-kernel rounding order.
    logits = t.randint(-8, 9, (2, 12, 7), generator=generator).float()

    actual = pool_fp_logits_per_segment(logits, batch.seg_map, batch.seg_doc_id.shape[1], mode)

    for row in range(logits.shape[0]):
        for segment in range(batch.seg_doc_id.shape[1]):
            token_mask = batch.seg_map[row] == segment
            if token_mask.any():
                expected = pool_fp_logits(logits[row : row + 1, token_mask], mode)[0]
            else:
                expected = t.zeros(logits.shape[-1], dtype=logits.dtype)
            assert t.equal(actual[row, segment], expected)


def _loop_oracle(
    logits: t.Tensor,
    seg_map: t.Tensor,
    n_seg: int,
    mode: FeaturePredLossMode,
) -> t.Tensor:
    """Pure-python per-(row, segment) oracle — no scatter machinery at all.

    Reviewer finding: the earlier versions of the two tests below re-pasted
    ``pool_fp_logits_per_segment``'s own scatter code as their "expected"
    values, comparing the implementation to itself. This oracle is independent:
    plain slicing, ``max``/``mean``/last-index over the token positions of each
    segment.
    """
    batch_size, time, n_features = logits.shape
    expected = t.zeros(batch_size, n_seg, n_features, dtype=logits.dtype)
    for row in range(batch_size):
        for segment in range(n_seg):
            positions = [p for p in range(time) if int(seg_map[row, p]) == segment]
            if not positions:
                continue
            span = logits[row, positions]
            if mode is FeaturePredLossMode.MAX_POOL:
                expected[row, segment] = span.max(dim=0).values
            elif mode is FeaturePredLossMode.FULL_SEGMENT:
                expected[row, segment] = span.mean(dim=0)
            else:
                expected[row, segment] = span[-1]
    return expected


def test_max_pool_matches_the_independent_loop_oracle_exactly() -> None:
    batch = make_batch([[4], [2, 1]], T=13, F=3, n_sae=12, seed=4)
    logits = t.randn(2, 13, 5, generator=t.Generator().manual_seed(20))
    n_seg = batch.seg_doc_id.shape[1]
    expected = _loop_oracle(logits, batch.seg_map, n_seg, FeaturePredLossMode.MAX_POOL)

    actual = pool_fp_logits_per_segment(logits, batch.seg_map, n_seg, FeaturePredLossMode.MAX_POOL)

    assert t.equal(actual, expected)


def test_last_token_matches_the_independent_loop_oracle_exactly() -> None:
    batch = make_batch([[3, 1], [2]], T=11, F=3, n_sae=12, seed=5)
    logits = t.randn(2, 11, 6, generator=t.Generator().manual_seed(21))
    n_seg = batch.seg_doc_id.shape[1]
    expected = _loop_oracle(logits, batch.seg_map, n_seg, FeaturePredLossMode.LAST_TOKEN)

    actual = pool_fp_logits_per_segment(
        logits, batch.seg_map, n_seg, FeaturePredLossMode.LAST_TOKEN
    )

    assert t.equal(actual, expected)


def test_full_segment_matches_the_independent_loop_oracle() -> None:
    batch = make_batch([[2, 2], [3]], T=10, F=3, n_sae=12, seed=6)
    logits = t.randn(2, 10, 4, generator=t.Generator().manual_seed(22))
    n_seg = batch.seg_doc_id.shape[1]
    expected = _loop_oracle(logits, batch.seg_map, n_seg, FeaturePredLossMode.FULL_SEGMENT)

    actual = pool_fp_logits_per_segment(
        logits, batch.seg_map, n_seg, FeaturePredLossMode.FULL_SEGMENT
    )

    # mean via scatter_add/divide vs slice-mean can differ by float accumulation
    # order; bound is machine-epsilon scale, real bugs differ at O(1).
    t.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "mode",
    [
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.FULL_SEGMENT,
        FeaturePredLossMode.LAST_TOKEN,
    ],
)
def test_single_token_segments_are_identity_pooling(mode: FeaturePredLossMode) -> None:
    batch = make_batch([[6]], T=6, F=2, n_sae=8, seed=6)
    logits = t.randn(1, 6, 4, generator=t.Generator().manual_seed(22))

    actual = pool_fp_logits_per_segment(logits, batch.seg_map, 6, mode)

    assert t.equal(actual, logits)


def test_best_token_pooling_raises() -> None:
    batch = make_batch([[2]], T=4, F=2, n_sae=8, seed=7)
    with pytest.raises(SelfCondTrainingError, match="best_token"):
        pool_fp_logits_per_segment(
            t.zeros(1, 4, 8),
            batch.seg_map,
            2,
            FeaturePredLossMode.BEST_TOKEN,
        )


def test_task3_helpers_are_deterministic_and_cover_padding_contract() -> None:
    first = make_batch([[3, 1], [2]], T=12, F=5, n_sae=16, seed=99)
    second = make_batch([[3, 1], [2]], T=12, F=5, n_sae=16, seed=99)
    assert [field.name for field in fields(SelfcondBatch)] == [
        "inputs",
        "targets",
        "enc_fids",
        "enc_mags",
        "enc_mask",
        "enc_pcts",
        "seg_map",
        "seg_doc_id",
    ]
    for field in fields(SelfcondBatch):
        assert t.equal(getattr(first, field.name), getattr(second, field.name))
    assert first.seg_doc_id.tolist() == [[0, 0, 0, 1], [0, 0, -1, -1]]
    assert not first.enc_mask[1, 2:].any()
    assert (first.enc_mags[first.enc_mask] <= 1).all()
    for row in range(first.seg_map.shape[0]):
        real_segments = int((first.seg_doc_id[row] >= 0).sum())
        assert t.equal(t.unique(first.seg_map[row]), t.arange(real_segments))

    model_a = tiny_head_model(FeaturePredLossMode.LAST_TOKEN, seed=44)
    model_b = tiny_head_model(FeaturePredLossMode.LAST_TOKEN, seed=44)
    assert model_a.feature_pred_config is not None
    assert model_a.feature_pred_config.loss_mode is FeaturePredLossMode.LAST_TOKEN
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert t.equal(tensor_a, tensor_b), name_a


def test_max_pool_propagates_positive_inf_instead_of_silently_zeroing() -> None:
    """Only the -inf empty-slot sentinel may be zeroed; a +inf pooled value is
    a REAL diverged head logit and must survive so the callers' finiteness
    guard raises (review finding: `.isinf()` also swallowed +inf)."""
    seg_map = t.tensor([[0, 0, 1, 1]], dtype=t.int64)
    logits = t.zeros(1, 4, 3)
    logits[0, 1, 2] = float("inf")
    pooled = pool_fp_logits_per_segment(
        seg_map=seg_map, fp_logits=logits, n_seg=3, mode=FeaturePredLossMode.MAX_POOL
    )
    assert pooled[0, 0, 2] == float("inf")  # +inf propagates
    assert t.equal(pooled[0, 2], t.zeros(3))  # empty slot 2: -inf sentinel zeroed


def test_assert_finite_fp_logits_raises_on_nonfinite() -> None:
    from steering.conditional.encoder_decoder.selfcond_training import assert_finite_fp_logits

    assert_finite_fp_logits(t.zeros(1, 2, 3), where="test")  # finite: no raise
    for bad in (float("inf"), float("-inf"), float("nan")):
        corrupted = t.zeros(1, 2, 3)
        corrupted[0, 1, 1] = bad
        with pytest.raises(SelfCondTrainingError, match="non-finite"):
            assert_finite_fp_logits(corrupted, where="test")
