import math

import pytest
import torch

from steering.conditional.encoder_decoder.loss_eval import (
    ConditionedBpbMetrics,
    evaluate_encdec_bpb_conditioned,
)


class RecordingModel:
    def __init__(self) -> None:
        self.masks: list[torch.Tensor] = []

    def __call__(
        self,
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        *,
        loss_reduction,
    ):
        assert loss_reduction == "none"
        self.masks.append(enc_mask.detach().clone())
        base = torch.arange(1, targets.numel() + 1, dtype=torch.float32)
        return (base * math.log(2)).view_as(targets)


def test_evaluate_encdec_bpb_conditioned_uses_real_encoder_mask_only() -> None:
    inputs = torch.tensor([[10, 11, 12]], dtype=torch.long)
    targets = torch.tensor([[1, 2, -1]], dtype=torch.long)
    enc_fids = torch.tensor([[[7, 0], [8, 9]]], dtype=torch.long)
    enc_mags = torch.tensor([[[2, 0], [3, 4]]], dtype=torch.long)
    enc_mask = torch.tensor([[[True, False], [True, True]]])
    enc_pcts = torch.zeros((1, 2, 2), dtype=torch.float32)
    seg_map = torch.tensor([[0, 1, 1]], dtype=torch.long)
    seg_doc_id = torch.tensor([[0, 0]], dtype=torch.long)
    batch = (
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        enc_pcts,
        seg_map,
        seg_doc_id,
        None,
        {},
    )

    model = RecordingModel()
    token_bytes = torch.tensor([0, 2, 4, 8], dtype=torch.int64)

    metrics = evaluate_encdec_bpb_conditioned(
        model,
        iter([batch]),
        eval_steps=1,
        token_bytes=token_bytes,
        device=torch.device("cpu"),
    )

    assert isinstance(metrics, ConditionedBpbMetrics)
    assert metrics.bpb_conditioned == pytest.approx(0.5)
    assert metrics.features_per_segment == [1, 2]
    assert metrics.avg_features_per_segment == pytest.approx(1.5)
    assert metrics.avg_features_per_batch == pytest.approx(3.0)
    assert metrics.magnitude_buckets == [2, 3, 4]
    assert len(model.masks) == 1
    assert torch.equal(model.masks[0], enc_mask)
