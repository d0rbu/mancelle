"""Tests for encoder input staging in the SFT data path."""

import torch as t

from scripts.train.conditional_encdec_sft import _stack_padded_encoder_inputs


def test_stack_padded_encoder_inputs_pads_rows_on_cpu() -> None:
    row_enc_fids = [
        t.tensor([[1, 2], [3, 0]], dtype=t.int64),
        t.tensor([[4]], dtype=t.int64),
    ]
    row_enc_mags = [
        t.tensor([[10, 20], [30, 0]], dtype=t.int64),
        t.tensor([[40]], dtype=t.int64),
    ]
    row_enc_masks = [
        t.tensor([[True, True], [True, False]], dtype=t.bool),
        t.tensor([[True]], dtype=t.bool),
    ]

    enc_fids, enc_mags, enc_mask = _stack_padded_encoder_inputs(
        row_enc_fids,
        row_enc_mags,
        row_enc_masks,
        device="cpu",
        use_cuda=False,
    )

    assert enc_fids.device.type == "cpu"
    assert enc_mags.device.type == "cpu"
    assert enc_mask.device.type == "cpu"

    expected_fids = t.tensor(
        [
            [[1, 2], [3, 0]],
            [[4, 0], [0, 0]],
        ],
        dtype=t.int64,
    )
    expected_mags = t.tensor(
        [
            [[10, 20], [30, 0]],
            [[40, 0], [0, 0]],
        ],
        dtype=t.int64,
    )
    expected_mask = t.tensor(
        [
            [[True, True], [True, False]],
            [[True, False], [False, False]],
        ],
        dtype=t.bool,
    )

    t.testing.assert_close(enc_fids, expected_fids)
    t.testing.assert_close(enc_mags, expected_mags)
    t.testing.assert_close(enc_mask, expected_mask)


def test_stack_padded_encoder_inputs_preserves_single_feature_width() -> None:
    row_enc_fids = [t.empty(1, 0, dtype=t.int64)]
    row_enc_mags = [t.empty(1, 0, dtype=t.int64)]
    row_enc_masks = [t.empty(1, 0, dtype=t.bool)]

    enc_fids, enc_mags, enc_mask = _stack_padded_encoder_inputs(
        row_enc_fids,
        row_enc_mags,
        row_enc_masks,
        device="cpu",
        use_cuda=False,
    )

    assert enc_fids.shape == (1, 1, 1)
    assert enc_mags.shape == (1, 1, 1)
    assert enc_mask.shape == (1, 1, 1)
    assert not enc_mask.any()


def test_stack_padded_encoder_inputs_accepts_torch_device_cpu() -> None:
    row_enc_fids = [t.tensor([[1]], dtype=t.int64)]
    row_enc_mags = [t.tensor([[2]], dtype=t.int64)]
    row_enc_masks = [t.tensor([[True]], dtype=t.bool)]

    enc_fids, enc_mags, enc_mask = _stack_padded_encoder_inputs(
        row_enc_fids,
        row_enc_mags,
        row_enc_masks,
        device=t.device("cpu"),
        use_cuda=False,
    )

    t.testing.assert_close(enc_fids, row_enc_fids[0].unsqueeze(0))
    t.testing.assert_close(enc_mags, row_enc_mags[0].unsqueeze(0))
    t.testing.assert_close(enc_mask, row_enc_masks[0].unsqueeze(0))
