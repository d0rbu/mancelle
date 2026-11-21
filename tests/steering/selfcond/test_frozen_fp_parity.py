"""Bit-exact parity of feature_prediction against its frozen known-good copy.

The primitive-consolidation refactor (review finding #12) may not move the
FP-loss surface by a single bit: ``frozen_feature_prediction_v1`` is the
verbatim origin/main implementation, and every test here demands
``torch.equal`` between it and the live module across all four loss modes,
both loss functions, masked and unmasked, fp32 and fp64, on batches covering
every structural edge the loss gates on (multi-document rows, ``-1`` padding
slots, SFT-style zero-token target-only segments, all-padding rows, and
non-finite logits).
"""

from __future__ import annotations

import pytest
import torch as t

import steering.conditional.encoder_decoder.feature_prediction as live
import tests.steering.selfcond.frozen_feature_prediction_v1 as frozen
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.selfcond_training import continuation_mask
from steering.conditional.feature_mask import resolve_feature_mask

N_SAE = 12

_BIT_VIEW = {t.float32: t.int32, t.float64: t.int64}


def _assert_bit_identical(actual: t.Tensor, expected: t.Tensor, context: object) -> None:
    """True bit-level equality — stricter than torch.equal, which treats any
    NaN as unequal even to a bit-identical NaN. The FP loss legitimately
    produces NaN for non-finite BCE logits under full_segment/last_token (a
    pre-existing origin/main property this refactor must PRESERVE, not fix),
    so parity must compare raw bits."""
    assert actual.dtype == expected.dtype, context
    assert actual.shape == expected.shape, context
    assert t.equal(
        actual.reshape(-1).contiguous().view(_BIT_VIEW[actual.dtype]),
        expected.reshape(-1).contiguous().view(_BIT_VIEW[expected.dtype]),
    ), context


def _case_batches(dtype: t.dtype) -> list[dict]:
    """Structurally adversarial (logits, seg_map, targets, seg_doc_id) cases."""
    cases: list[dict] = []
    generator = t.Generator().manual_seed(1234)

    def logits(batch: int, time: int) -> t.Tensor:
        return t.randn(batch, time, N_SAE, generator=generator, dtype=dtype)

    def targets(batch: int, n_seg: int) -> t.Tensor:
        return t.rand(batch, n_seg, N_SAE, generator=generator, dtype=t.float32)

    # 1. Plain packed-row shape: exact tiling, two docs + one doc.
    seg_map = t.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 1, 1, 1, 1, 1]])
    seg_doc_id = t.tensor([[0, 0, 1, 1], [0, 0, -1, -1]])
    cases.append(
        {
            "logits": logits(2, 8),
            "seg_map": seg_map,
            "targets": targets(2, 4),
            "seg_doc_id": seg_doc_id,
        }
    )

    # 2. SFT shape: zero-token prediction-target-only segment (index 2 has no
    #    tokens but doc_id >= 0) plus a trailing padding slot.
    seg_map = t.tensor([[0, 0, 0, 1, 1, 1]])
    seg_doc_id = t.tensor([[0, 0, 0, -1]])  # segment 2: no tokens, real doc id
    seg_doc_id[0, 2] = 0
    cases.append(
        {
            "logits": logits(1, 6),
            "seg_map": seg_map,
            "targets": targets(1, 4),
            "seg_doc_id": seg_doc_id,
        }
    )

    # 3. Adjacent padding slots (the -1 == -1 leak case the losses guard) and
    #    a row whose every slot is padding.
    seg_map = t.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
    seg_doc_id = t.tensor([[0, 0, -1, -1], [-1, -1, -1, -1]])
    cases.append(
        {
            "logits": logits(2, 4),
            "seg_map": seg_map,
            "targets": targets(2, 4),
            "seg_doc_id": seg_doc_id,
        }
    )

    # 4. Non-finite logits: +inf and -inf in real segments (the max-pool
    #    isinf handling must stay byte-for-byte).
    corrupted = logits(1, 6)
    corrupted[0, 1, 3] = float("inf")
    corrupted[0, 4, 5] = float("-inf")
    cases.append(
        {
            "logits": corrupted,
            "seg_map": t.tensor([[0, 0, 1, 1, 2, 2]]),
            "targets": targets(1, 3),
            "seg_doc_id": t.tensor([[0, 0, 0]]),
        }
    )
    return cases


@pytest.mark.parametrize("dtype", [t.float32, t.float64])
@pytest.mark.parametrize("loss_fn", list(FeaturePredLossFn))
@pytest.mark.parametrize("mode", list(FeaturePredLossMode))
@pytest.mark.parametrize("masked", [False, True])
def test_compute_feature_pred_loss_is_bit_identical_to_frozen(
    mode: FeaturePredLossMode,
    loss_fn: FeaturePredLossFn,
    dtype: t.dtype,
    masked: bool,
) -> None:
    feature_mask = (
        resolve_feature_mask({"mode": "blacklist", "ids": [2, 7]}, N_SAE) if masked else None
    )
    for case in _case_batches(dtype):
        live_loss = live.compute_feature_pred_loss(
            case["logits"],
            case["seg_map"],
            case["targets"],
            case["seg_doc_id"],
            mode,
            loss_fn,
            feature_mask=feature_mask,
        )
        frozen_loss = frozen.compute_feature_pred_loss(
            case["logits"],
            case["seg_map"],
            case["targets"],
            case["seg_doc_id"],
            mode,
            loss_fn,
            feature_mask=feature_mask,
        )
        _assert_bit_identical(live_loss, frozen_loss, (mode, loss_fn, dtype, masked))


def test_pool_fp_logits_is_bit_identical_to_frozen() -> None:
    generator = t.Generator().manual_seed(77)
    fp_logits = t.randn(2, 9, N_SAE, generator=generator)
    for mode in (
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.LAST_TOKEN,
        FeaturePredLossMode.FULL_SEGMENT,
    ):
        assert t.equal(live.pool_fp_logits(fp_logits, mode), frozen.pool_fp_logits(fp_logits, mode))


def _vendored_pre_refactor_continuation_mask(seg_map: t.Tensor, seg_doc_id: t.Tensor) -> t.Tensor:
    """The original continuation_mask body (pre-consolidation), verbatim."""
    batch_size, time = seg_map.shape
    n_seg = seg_doc_id.shape[1]
    seg_lengths = t.zeros(batch_size, n_seg, dtype=t.float32, device=seg_map.device)
    seg_lengths.scatter_add_(
        1,
        seg_map,
        t.ones(batch_size, time, dtype=t.float32, device=seg_map.device),
    )
    live_mask = t.zeros_like(seg_doc_id, dtype=t.bool)
    if n_seg > 1:
        live_mask[:, 1:] = (
            (seg_doc_id[:, 1:] >= 0)
            & (seg_doc_id[:, 1:] == seg_doc_id[:, :-1])
            & (seg_lengths[:, :-1] > 0)
        )
    return live_mask


def test_continuation_mask_is_bit_identical_to_pre_refactor_body() -> None:
    """Includes randomized doc layouts plus every structural edge case; the
    equivalence proof (the FP-loss target_seg_valid conjunct is vacuous when
    adjacent doc ids are equal and non-negative) must hold empirically."""
    generator = t.Generator().manual_seed(4321)
    for case in _case_batches(t.float32):
        expected = _vendored_pre_refactor_continuation_mask(case["seg_map"], case["seg_doc_id"])
        actual = continuation_mask(case["seg_map"], case["seg_doc_id"])
        assert t.equal(actual, expected)
    for _ in range(50):
        n_seg = int(t.randint(1, 6, (1,), generator=generator))
        time = int(t.randint(n_seg, 3 * n_seg + 1, (1,), generator=generator))
        seg_map = t.sort(t.randint(0, n_seg, (2, time), generator=generator), dim=1).values
        seg_doc_id = t.randint(-1, 3, (2, n_seg), generator=generator)
        expected = _vendored_pre_refactor_continuation_mask(seg_map, seg_doc_id)
        actual = continuation_mask(seg_map, seg_doc_id)
        assert t.equal(actual, expected)
