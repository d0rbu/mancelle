"""Host-staged batching for target-aware encoder inputs."""

from types import SimpleNamespace

import pytest
import torch as t

from steering.conditional.encoder_decoder.eval_adapter import (
    ContextConditioningRow,
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.target_conditioning import ContextFeatureSegment

requires_cuda = pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")


def _payloads() -> list[ContextConditioningRow]:
    return [
        ContextConditioningRow(
            context_tokens=[1, 2],
            context_segments=(
                ContextFeatureSegment(
                    feature_ids=(3, 7),
                    buckets=(1, 2),
                    char_start=0,
                    char_end=1,
                ),
                ContextFeatureSegment(
                    feature_ids=(),
                    buckets=(),
                    char_start=1,
                    char_end=2,
                ),
            ),
        ),
        ContextConditioningRow(
            context_tokens=[4],
            context_segments=(
                ContextFeatureSegment(
                    feature_ids=(5,),
                    buckets=(2,),
                    char_start=0,
                    char_end=1,
                ),
            ),
        ),
    ]


def _feature_slots(
    segments: tuple[ContextFeatureSegment, ...],
) -> list[dict[int, int]]:
    return [
        dict(zip(segment.feature_ids, segment.buckets, strict=True))
        for segment in segments
    ] + [{9: 3}]


def _stub(write_feature_slot):
    return SimpleNamespace(
        _target_segments=[{9: 3}],
        _num_encoder_segments_for_context=lambda segments: len(
            _feature_slots(segments)
        ),
        _encoder_feature_slots=_feature_slots,
        _write_feature_slot=write_feature_slot,
        turn_contract=None,  # the fixed packing path; turn packing carries a contract
        model=SimpleNamespace(
            encoder=SimpleNamespace(
                config=SimpleNamespace(n_sae_features=20, num_buckets=4)
            )
        ),
    )


def _pack(stub, *, device: t.device | str):
    pack = TargetAwareEncoderDecoderEvalWrapper._pack_encoder_inputs.__get__(
        stub,
        TargetAwareEncoderDecoderEvalWrapper,
    )
    return pack(_payloads(), device=device)


def test_host_staged_encoder_inputs_preserve_values_and_padding() -> None:
    actual = _pack(
        _stub(TargetAwareEncoderDecoderEvalWrapper._write_feature_slot),
        device="cpu",
    )
    expected_ids = t.tensor(
        [
            [[3, 7], [0, 0], [9, 0]],
            [[5, 0], [9, 0], [0, 0]],
        ]
    )
    expected_magnitudes = t.tensor(
        [
            [[1, 2], [0, 0], [3, 0]],
            [[2, 0], [3, 0], [0, 0]],
        ]
    )
    expected_mask = t.tensor(
        [
            [[True, True], [False, False], [True, False]],
            [[True, False], [True, False], [False, False]],
        ]
    )
    for tensor in actual:
        assert tensor.device.type == "cpu"
    assert t.equal(actual[0], expected_ids)
    assert t.equal(actual[1], expected_magnitudes)
    assert t.equal(actual[2], expected_mask)


def test_context_token_lengths_are_validated_on_host() -> None:
    wrapper = object.__new__(TargetAwareEncoderDecoderEvalWrapper)
    object.__setattr__(wrapper, "_active_target_mask", None)
    object.__setattr__(
        wrapper,
        "_active_context_token_ids",
        [[1, 2, 3], []],
    )

    with pytest.raises(ValueError, match=r"row 1 has 0"):
        wrapper._validate_active_context(batch_size=2, seq_len=3)


def test_context_tokens_supply_target_start_without_reading_device_mask() -> None:
    def reject_target_mask_read(_row: int):
        raise AssertionError("target mask should not be read")

    wrapper = object.__new__(TargetAwareEncoderDecoderEvalWrapper)
    object.__setattr__(
        wrapper,
        "_active_context_token_ids",
        [[11, 12, 13], [21, 22]],
    )
    object.__setattr__(wrapper, "_target_mask_for_row", reject_target_mask_read)

    assert wrapper._final_segment_start(row=0, T=5) == 2
    assert wrapper._final_segment_start(row=1, T=5) == 1


@requires_cuda
def test_cuda_encoder_inputs_are_filled_on_host_then_bulk_transferred() -> None:
    write_devices: list[tuple[t.device, t.device, t.device]] = []

    def recording_write(*args) -> None:
        write_devices.append((args[0].device, args[1].device, args[2].device))
        TargetAwareEncoderDecoderEvalWrapper._write_feature_slot(*args)

    expected = _pack(
        _stub(TargetAwareEncoderDecoderEvalWrapper._write_feature_slot),
        device="cpu",
    )
    actual = _pack(_stub(recording_write), device="cuda")

    assert write_devices
    assert all(
        devices == (t.device("cpu"), t.device("cpu"), t.device("cpu"))
        for devices in write_devices
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.device.type == "cuda"
        assert t.equal(actual_tensor.cpu(), expected_tensor)
