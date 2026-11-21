"""T-U5: continuation masks and byte-preserving conditioning rebuilds."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

from steering.conditional.encoder_decoder.selfcond_training import (
    MutableConditioning,
    SelectedActions,
    build_selfcond_state,
    continuation_mask,
    write_predicted_slot,
)
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch


def _widened_reference(batch: SelfcondBatch, width: int) -> MutableConditioning:
    shape = (*batch.enc_fids.shape[:2], width)
    fids = t.zeros(shape, dtype=batch.enc_fids.dtype)
    mags = t.zeros(shape, dtype=batch.enc_mags.dtype)
    mask = t.zeros(shape, dtype=batch.enc_mask.dtype)
    original_width = batch.enc_fids.shape[-1]
    fids[..., :original_width] = batch.enc_fids
    mags[..., :original_width] = batch.enc_mags
    mask[..., :original_width] = batch.enc_mask
    return MutableConditioning(
        fids=fids,
        mags=mags,
        mask=mask,
        sign=t.zeros(shape, dtype=t.float32),
    )


def _actions(batch_size: int = 2) -> SelectedActions:
    return SelectedActions(
        pos_ids=t.tensor([[9, 7], [8, -1]], dtype=t.int64)[:batch_size],
        neg_ids=t.tensor([[3, -1], [2, 4]], dtype=t.int64)[:batch_size],
        k=t.tensor([2, 1], dtype=t.int64)[:batch_size],
        l=t.tensor([1, 2], dtype=t.int64)[:batch_size],  # noqa: E741 - spec notation
    )


def test_continuation_mask_and_noncontinuation_slots_are_byte_preserved() -> None:
    batch = make_batch([[3, 1], [1, 2]], T=12, F=3, n_sae=16, seed=31)
    expected_live = t.tensor(
        [[False, True, True, False], [False, False, True, False]],
        dtype=t.bool,
    )
    live = continuation_mask(batch.seg_map, batch.seg_doc_id)
    state = build_selfcond_state(batch, 5)
    original = _widened_reference(batch, 5)

    assert t.equal(live, expected_live)
    actions = _actions()
    for slot in range(1, batch.seg_doc_id.shape[1]):
        write_predicted_slot(state, slot, actions, live[:, slot])

    for row in range(batch.seg_doc_id.shape[0]):
        for slot in range(batch.seg_doc_id.shape[1]):
            if not bool(expected_live[row, slot]):
                assert t.equal(state.fids[row, slot], original.fids[row, slot])
                assert t.equal(state.mags[row, slot], original.mags[row, slot])
                assert t.equal(state.mask[row, slot], original.mask[row, slot])
                assert t.equal(state.sign[row, slot], original.sign[row, slot])


def test_predicted_slot_layout_is_pos_then_neg_with_exact_buckets_mask_and_sign() -> None:
    batch = make_batch([[2], [2]], T=8, F=2, n_sae=16, seed=32)
    state = build_selfcond_state(batch, 5)
    live = continuation_mask(batch.seg_map, batch.seg_doc_id)[:, 1]

    write_predicted_slot(state, 1, _actions(), live)

    assert t.equal(
        state.fids[:, 1],
        t.tensor([[9, 7, 3, 0, 0], [8, 2, 4, 0, 0]], dtype=t.int64),
    )
    assert t.equal(
        state.mags[:, 1],
        t.tensor([[0, 0, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=t.int64),
    )
    assert t.equal(
        state.mask[:, 1],
        t.tensor(
            [[True, True, True, False, False], [True, True, True, False, False]],
            dtype=t.bool,
        ),
    )
    assert t.equal(
        state.sign[:, 1],
        t.tensor(
            [[1.0, 1.0, -1.0, 0.0, 0.0], [1.0, -1.0, -1.0, 0.0, 0.0]],
            dtype=t.float32,
        ),
    )


def test_single_segment_documents_leave_whole_state_equal_to_widened_batch() -> None:
    batch = make_batch([[1, 1], [1]], T=8, F=3, n_sae=16, seed=33)
    state = build_selfcond_state(batch, 5)
    expected = _widened_reference(batch, 5)
    live = continuation_mask(batch.seg_map, batch.seg_doc_id)

    for slot in range(1, batch.seg_doc_id.shape[1]):
        write_predicted_slot(state, slot, _actions(), live[:, slot])

    assert not live.any()
    assert t.equal(state.fids, expected.fids)
    assert t.equal(state.mags, expected.mags)
    assert t.equal(state.mask, expected.mask)
    assert t.equal(state.sign, expected.sign)


def test_zero_token_source_keeps_following_real_slot_unchanged() -> None:
    base = make_batch([[3]], T=4, F=3, n_sae=16, seed=34)
    batch = replace(
        base,
        seg_map=t.tensor([[0, 0, 2, 2]], dtype=t.int64),
        seg_doc_id=t.tensor([[0, 0, 0]], dtype=t.int64),
    )
    state = build_selfcond_state(batch, 5)
    expected = _widened_reference(batch, 5)
    live = continuation_mask(batch.seg_map, batch.seg_doc_id)

    assert t.equal(live, t.tensor([[False, True, False]], dtype=t.bool))
    write_predicted_slot(state, 1, _actions(1), live[:, 1])
    write_predicted_slot(state, 2, _actions(1), live[:, 2])

    assert t.equal(state.fids[:, 2], expected.fids[:, 2])
    assert t.equal(state.mags[:, 2], expected.mags[:, 2])
    assert t.equal(state.mask[:, 2], expected.mask[:, 2])
    assert t.equal(state.sign[:, 2], expected.sign[:, 2])


@pytest.mark.parametrize(
    ("batch_width", "predicted_need", "expected_width"),
    [(7, 5, 7), (2, 5, 5)],
)
def test_state_width_is_max_of_batch_width_and_predicted_need_without_aliasing(
    batch_width: int,
    predicted_need: int,
    expected_width: int,
) -> None:
    batch = make_batch([[2]], T=6, F=batch_width, n_sae=16, seed=35)

    state = build_selfcond_state(batch, max(batch_width, predicted_need))

    assert state.fids.shape[-1] == expected_width
    assert state.mags.shape[-1] == expected_width
    assert state.mask.shape[-1] == expected_width
    assert state.sign.shape[-1] == expected_width
    assert state.fids.data_ptr() != batch.enc_fids.data_ptr()
    assert state.mags.data_ptr() != batch.enc_mags.data_ptr()
    assert state.mask.data_ptr() != batch.enc_mask.data_ptr()
    original_fids = batch.enc_fids.clone()
    original_mags = batch.enc_mags.clone()
    original_mask = batch.enc_mask.clone()
    state.fids.zero_()
    state.mags.zero_()
    state.mask.zero_()
    assert t.equal(batch.enc_fids, original_fids)
    assert t.equal(batch.enc_mags, original_mags)
    assert t.equal(batch.enc_mask, original_mask)
