"""Pre-#1286 full-sequence target goldens for both SFT trainers."""

from __future__ import annotations

import hashlib
import json

import torch as t

from tests.steering.goldens.gen_sft_loss_mask_off_golden import (
    GOLDEN_PATH,
    build_capture,
)

_DTYPE_BY_NAME = {
    "torch.int32": t.int32,
    "torch.int64": t.int64,
}
_GOLDEN_SHA256 = "32b7f57fc8ca4142e78c26093ff3f68e2853ecb3119a3fe5539a2b0b79afb1d5"


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_off_mode_golden_file_matches_pre_feature_capture() -> None:
    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == _GOLDEN_SHA256


def test_both_sft_trainers_reconstruct_pre_mask_batches_with_torch_equal() -> None:
    golden = _golden()
    actual = build_capture()
    for trainer in ("decoder_only", "encoder_decoder"):
        for field in ("inputs", "targets"):
            expected_record = golden[trainer][field]
            actual_record = actual[trainer][field]
            assert actual_record["sha256"] == expected_record["sha256"], (
                trainer,
                field,
            )
            dtype = _DTYPE_BY_NAME[expected_record["dtype"]]
            assert t.equal(
                t.tensor(actual_record["values"], dtype=dtype),
                t.tensor(expected_record["values"], dtype=dtype),
            ), (trainer, field)
