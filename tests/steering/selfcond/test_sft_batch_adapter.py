"""Source-specific batch normalization contracts for self-conditioning SFT."""

from __future__ import annotations

import pytest
import torch as t

from errors import SelfCondTrainingError
from steering.conditional.encoder_decoder.selfcond_training import (
    ConditioningBatch,
    SelfCondBatch,
)
from tests.steering.selfcond.helpers import make_batch


def test_from_sft_maps_the_eight_tuple_and_ignores_fp_targets() -> None:
    batch = make_batch([[2], [1, 1]], T=16, F=4, n_sae=8, seed=5)
    fp_targets = t.zeros(
        batch.enc_fids.shape[0],
        batch.enc_fids.shape[1],
        8,
        dtype=t.float32,
    )
    eight = (
        batch.inputs.to(t.int32),
        batch.targets,
        batch.enc_fids,
        batch.enc_mags,
        batch.enc_mask,
        batch.seg_map,
        fp_targets,
        batch.seg_doc_id,
    )

    view = SelfCondBatch.from_sft(eight)

    assert view.inputs is eight[0]
    assert view.targets is batch.targets
    assert view.enc_fids is batch.enc_fids
    assert view.enc_mags is batch.enc_mags
    assert view.enc_mask is batch.enc_mask
    assert view.seg_map is batch.seg_map
    assert view.seg_doc_id is batch.seg_doc_id
    assert not hasattr(view, "fp_targets")

    without_fp_targets = eight[:6] + (None, batch.seg_doc_id)
    view_without_fp_targets = SelfCondBatch.from_sft(without_fp_targets)
    assert view_without_fp_targets.seg_doc_id is batch.seg_doc_id


def test_from_sft_rejects_seven_tuple_and_none_doc_id() -> None:
    batch = make_batch([[2]], T=8, F=3, n_sae=8, seed=7)
    inputs = batch.inputs.to(t.int32)
    targets = batch.targets.clone()
    targets[:, -2:] = -1
    fids = batch.enc_fids
    mags = batch.enc_mags
    mask = batch.enc_mask
    seg_map = batch.seg_map

    seven = (inputs, targets, fids, mags, mask, seg_map, None)
    with pytest.raises(SelfCondTrainingError, match="return_seg_doc_id=True"):
        SelfCondBatch.from_sft(seven)

    eight_without_doc_ids = (inputs, targets, fids, mags, mask, seg_map, None, None)
    with pytest.raises(
        SelfCondTrainingError, match="requires a real seg_doc_id tensor at index 7"
    ):
        SelfCondBatch.from_sft(eight_without_doc_ids)


def test_adapter_satisfies_conditioning_batch_protocol() -> None:
    batch = make_batch([[2]], T=8, F=3, n_sae=8, seed=11)
    adapter = SelfCondBatch(
        inputs=batch.inputs,
        targets=batch.targets,
        enc_fids=batch.enc_fids,
        enc_mags=batch.enc_mags,
        enc_mask=batch.enc_mask,
        seg_map=batch.seg_map,
        seg_doc_id=batch.seg_doc_id,
    )

    assert isinstance(adapter, ConditioningBatch)
