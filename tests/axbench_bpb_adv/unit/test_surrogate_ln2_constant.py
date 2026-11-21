"""Parity guards for the shared BPB nats-to-bits reduction."""

from __future__ import annotations

import torch as t

from scripts.eval.axbench_adv.gcg_math import weighted_bpb
from steering.eval.axbench_bpb_adv.surrogate import _output_byte_weights


def test_output_byte_weights_are_float32() -> None:
    """The surrogate supplies the shared reducer with float32 byte weights."""
    weights = _output_byte_weights(
        output_tokens=[1, 2, 3],
        token_bytes=t.tensor([0, 1, 2, 3, 4], dtype=t.long),
        device=t.device("cpu"),
    )
    assert weights.dtype == t.float32


def test_bpb_reduction_matches_device_tensor_ln2_bitwise() -> None:
    """End-to-end parity of the reduction formula under float32."""
    generator = t.Generator().manual_seed(507)
    in_mask_loss = t.rand(257, generator=generator, dtype=t.float32) + 0.25
    byte_weights = t.rand(257, generator=generator, dtype=t.float32) + 0.5

    ln2_tensor = t.log(t.tensor(2.0))  # the removed expression, on CPU
    old = (in_mask_loss.to(t.float32) * byte_weights).sum() / (ln2_tensor * byte_weights.sum())
    new = weighted_bpb(in_mask_loss, byte_weights, tensor_ln2=False)

    assert t.equal(old, new), "float32 BPB reduction must be bitwise unchanged"
    assert old.dtype == new.dtype == t.float32


def test_shared_reducer_normalizes_bfloat16_weights_to_float32() -> None:
    """Lower-precision inputs retain the reducer's explicit float32 contract."""
    generator = t.Generator().manual_seed(1)
    in_mask_loss = t.rand(129, generator=generator, dtype=t.float32) + 0.25
    byte_weights = (t.rand(129, generator=generator) + 0.5).to(t.bfloat16)

    normalized_weights = byte_weights.to(t.float32)
    expected = (in_mask_loss.to(t.float32) * normalized_weights).sum() / (
        t.log(t.tensor(2.0)) * normalized_weights.sum()
    )
    actual = weighted_bpb(in_mask_loss, byte_weights, tensor_ln2=False)

    assert t.equal(actual, expected)
    assert actual.dtype == t.float32
