"""Canonical history recomputation preserves semantics without retaining weight casts.

CPU cases use the existing test-owned padded encoder reference; CUDA cases run
the compiled packed encoder. Neither path replaces the encoder's real layers.
"""

from __future__ import annotations

import copy
import weakref
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import steering.conditional.encoder_decoder.encoder as encoder_module
import steering.conditional.encoder_decoder.rlvr_policy as rlvr_policy
from steering.conditional.encoder_decoder.encoder import FeatureEncoder, FeatureEncoderConfig
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    append_packed_encoder_segment,
)
from steering.conditional.encoder_decoder.rlvr_policy import RLVRPromptReplay
from steering.conditional.encoder_decoder.sft_grpo import ConditioningAction
from tests.padded_encoder_reference import feature_encoder_forward_with_cpu_reference


class _EncoderModel(t.nn.Module):
    def __init__(self, encoder: FeatureEncoder) -> None:
        super().__init__()
        self.encoder = encoder


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> FeatureEncoder:
    # Match the live mixed-precision model: FP32 linear weights, BF16 embeddings.
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.bfloat16)
    with t.random.fork_rng(devices=[]):
        t.manual_seed(1687)
        result = FeatureEncoder(
            FeatureEncoderConfig(
                n_sae_features=32,
                num_buckets=2,
                n_embd=64,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                head_dim=32,
            )
        ).eval()
    for embedding in (
        result.feature_embed,
        result.magnitude_embed,
        *result.value_embeds.values(),
    ):
        embedding.to(dtype=t.bfloat16)
    return result


def _inputs(history_count: int) -> tuple[RLVRPromptReplay, tuple[ConditioningAction, ...]]:
    ids = t.tensor([[[1, 2, 0, 3], [4, 0, 5, 0]]])
    prompt = RLVRPromptReplay(
        prompt_token_ids=(1, 2),
        enc_feature_ids=ids,
        enc_magnitudes=t.tensor([[[0, 1, 0, 1], [1, 0, 0, 0]]]),
        enc_mask=ids.ne(0),
        prefix_seg_map=t.tensor([[0, 1]]),
        encoder_slot_count=2,
    )
    history = tuple(
        ConditioningAction(
            feature_ids=tuple(1 + (3 * index + column) % 31 for column in range(1 + index % 5)),
            magnitudes=tuple((index + column) % 2 for column in range(1 + index % 5)),
            mask=tuple(column == 0 or (index + column) % 3 != 0 for column in range(1 + index % 5)),
        )
        for index in range(history_count)
    )
    return prompt, history


def _original_encode_state(
    model: _EncoderModel,
    prompt: RLVRPromptReplay,
    feature_history: Sequence[ConditioningAction],
    *,
    device: t.device,
) -> PackedEncoderOutput:
    """The pre-checkpoint loop, including independent action tensorization."""
    packed = model.encoder(
        prompt.enc_feature_ids.to(device),
        prompt.enc_magnitudes.to(device),
        prompt.enc_mask.to(device),
    )
    for action in feature_history:
        segment = model.encoder(
            t.tensor(action.feature_ids, device=device, dtype=t.long)[None, None, :],
            t.tensor(action.magnitudes, device=device, dtype=t.long)[None, None, :],
            t.tensor(action.mask, device=device, dtype=t.bool)[None, None, :],
        )
        packed = append_packed_encoder_segment(packed, segment)
    return packed


def _assert_packed_equal(actual: PackedEncoderOutput, expected: PackedEncoderOutput) -> None:
    t.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    for field in fields(expected.layout):
        left, right = getattr(actual.layout, field.name), getattr(expected.layout, field.name)
        if isinstance(right, t.Tensor):
            t.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right, field.name


def _replay(
    monkeypatch: pytest.MonkeyPatch,
    encoder: FeatureEncoder,
    history_count: int,
    encode_state: Callable[..., PackedEncoderOutput],
) -> tuple[tuple[PackedEncoderOutput, PackedEncoderOutput], dict[str, t.Tensor], list[bool]]:
    model = _EncoderModel(copy.deepcopy(encoder))
    prompt, history = _inputs(history_count)
    marker = ContextVar("history_replay_context", default="outside")
    calls: list[bool] = []
    packed_outputs: list[PackedEncoderOutput] = []

    def forward(ids: t.Tensor, magnitudes: t.Tensor, mask: t.Tensor) -> PackedEncoderOutput:
        assert marker.get() == "caller", "compiler context lost in nested recomputation"
        calls.append(t.is_grad_enabled())
        return feature_encoder_forward_with_cpu_reference(model.encoder, ids, magnitudes, mask)

    def canonical(model: _EncoderModel, prompt: RLVRPromptReplay, **kwargs: Any) -> t.Tensor:
        packed = encode_state(
            model, prompt, kwargs["feature_history"], device=next(model.parameters()).device
        )
        packed_outputs.append(packed.detach())
        return packed.values

    monkeypatch.setattr(model.encoder, "forward", forward)
    monkeypatch.setattr(rlvr_policy, "canonical_fp_logits", canonical)
    action = cast(
        rlvr_policy.RLVRActionTrace,
        SimpleNamespace(
            committed_prefix=(), outgoing_seg_map=(), pool_positions=(), feature_history=history
        ),
    )
    token = marker.set("caller")
    try:
        output = rlvr_policy._checkpointed_canonical_fp_logits(model, prompt, action)
        assert marker.get() == "caller"
    finally:
        marker.reset(token)
    # Autograd recomputation must restore the captured context even when backward
    # runs outside it, as happens with CUDA autograd worker ContextVars.
    weights = t.linspace(-1, 1, output.numel(), device=output.device).view_as(output)
    (output.float() * weights).sum().backward()
    assert marker.get() == "outside"
    assert len(packed_outputs) == 2
    # CUDA inference and autograd graphs can round BF16 intermediates differently.
    # The caller compares each mode against the original loop.
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert t.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero() > 0, name
        gradients[name] = parameter.grad.detach().clone()
    return (packed_outputs[0], packed_outputs[1]), gradients, calls


@pytest.mark.parametrize("history_count", [0, 1, 4, 16])
def test_history_replay_matches_original_outputs_gradients_layout_and_context(
    monkeypatch: pytest.MonkeyPatch, encoder: FeatureEncoder, history_count: int
) -> None:
    expected_outputs, expected_gradients, original_calls = _replay(
        monkeypatch, encoder, history_count, _original_encode_state
    )
    actual_outputs, actual_gradients, calls = _replay(
        monkeypatch, encoder, history_count, rlvr_policy._encode_canonical_state
    )
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        _assert_packed_equal(actual, expected)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, gradient in actual_gradients.items():
        t.testing.assert_close(gradient, expected_gradients[name], rtol=0, atol=0, msg=name)
    assert original_calls == [False] * (1 + history_count) + [True] * (1 + history_count)
    assert calls == [False] * (1 + history_count) + [True] * (1 + 2 * history_count)


@pytest.mark.parametrize("history_count", [0, 1, 4, 16])
def test_collection_does_not_checkpoint_or_repeat_encoder_calls(
    monkeypatch: pytest.MonkeyPatch, encoder: FeatureEncoder, history_count: int
) -> None:
    prompt, history = _inputs(history_count)
    model = _EncoderModel(encoder)
    calls = []

    def forward(ids: t.Tensor, magnitudes: t.Tensor, mask: t.Tensor) -> PackedEncoderOutput:
        calls.append(t.is_grad_enabled())
        return feature_encoder_forward_with_cpu_reference(encoder, ids, magnitudes, mask)

    def unexpected_checkpoint(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("collection must not use activation checkpointing")

    monkeypatch.setattr(encoder, "forward", forward)
    monkeypatch.setattr(rlvr_policy, "checkpoint", unexpected_checkpoint)
    with t.no_grad():
        actual = rlvr_policy._encode_canonical_state(model, prompt, history, device=t.device("cpu"))
        assert calls == [False] * (1 + history_count)
        expected = _original_encode_state(model, prompt, history, device=t.device("cpu"))
    _assert_packed_equal(actual, expected)
    assert not actual.values.requires_grad


@pytest.mark.parametrize("history_count", [0, 16])
def test_history_does_not_retain_additional_bf16_mlp_weight_casts(
    monkeypatch: pytest.MonkeyPatch, encoder: FeatureEncoder, history_count: int
) -> None:
    # Observe actual saved weight storages, not a mock checkpoint invocation.
    # Inner checkpoints own their hooks; this measures what remains saved by the
    # outer replay, not total CUDA peak allocation during a segment recomputation.
    expected_weights = {
        name: parameter.detach().to(t.bfloat16).T
        for name, parameter in encoder.named_parameters()
        if ".mlp." in name
    }

    def saved_cast_peak(encode_state: Callable[..., PackedEncoderOutput]) -> int:
        saved: list[weakref.ReferenceType[t.Tensor]] = []
        matched = set()
        peak = 0

        def pack(tensor: t.Tensor) -> t.Tensor:
            nonlocal peak
            if tensor.dtype == t.bfloat16 and tuple(tensor.shape) in {(64, 256), (256, 64)}:
                matches = [
                    name
                    for name, expected in expected_weights.items()
                    if expected.shape == tensor.shape and t.equal(expected, tensor)
                ]
                assert len(matches) == 1
                matched.add(matches[0])
                saved.append(weakref.ref(tensor))
                live_storages = {
                    value.untyped_storage().data_ptr()
                    for ref in saved
                    if (value := ref()) is not None
                }
                peak = max(peak, len(live_storages))
            return tensor

        with t.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            _replay(monkeypatch, encoder, history_count, encode_state)
        assert matched == expected_weights.keys()
        assert all(ref() is None for ref in saved)
        return peak

    peaks = [
        saved_cast_peak(encode_state)
        for encode_state in (_original_encode_state, rlvr_policy._encode_canonical_state)
    ]
    assert peaks == [len(expected_weights) * (1 + history_count), len(expected_weights)]


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="compiled packed encoder requires CUDA")
@pytest.mark.parametrize("history_count", [0, 4])
def test_compiled_cuda_history_replay_matches_original(
    monkeypatch: pytest.MonkeyPatch, encoder: FeatureEncoder, history_count: int
) -> None:
    # The shared CPU-reference wrapper delegates CUDA tensors to the real
    # production forward, including the compiled packed encoder and FlexAttention.
    encoder = encoder.to("cuda")
    test_history_replay_matches_original_outputs_gradients_layout_and_context(
        monkeypatch, encoder, history_count
    )
