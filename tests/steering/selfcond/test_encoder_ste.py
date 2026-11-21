"""T-U6: value-zero STE injection preserves the encoder's forward contract."""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from errors import SelfCondTrainingError
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
    SteForm,
    SteInputs,
)
from steering.conditional.encoder_decoder.packed_encoder import PackedEncoderOutput
from steering.conditional.feature_vocab import FeatureVocab


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because the CPU backward contract cannot use FlexAttention."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _encoder(*, feature_vocab: FeatureVocab | None = None, seed: int = 17) -> FeatureEncoder:
    cfg = FeatureEncoderConfig(
        n_sae_features=8,
        num_buckets=2,
        n_embd=4,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        head_dim=4,
        ve_gate_channels=4,
    )
    enc = FeatureEncoder(cfg, feature_vocab=feature_vocab).double()
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for parameter in enc.parameters():
            parameter.copy_(t.randn(parameter.shape, generator=generator, dtype=t.float64) * 0.2)
    enc.init_vocab_buffers()
    return enc


def _batch() -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    feature_ids = t.tensor([[[1, 5, 0], [2, 7, 3]], [[4, 6, 0], [0, 1, 2]]], dtype=t.int64)
    magnitudes = t.tensor([[[0, 1, 0], [1, 0, 1]], [[1, 0, 0], [0, 1, 0]]], dtype=t.int64)
    mask = t.tensor(
        [[[True, True, False], [True, True, True]], [[True, True, False], [True, True, True]]]
    )
    return feature_ids, magnitudes, mask


def _loss_weights(output: t.Tensor) -> t.Tensor:
    return t.arange(output.numel(), dtype=output.dtype).reshape_as(output) / output.numel()


def _backward_parameter_grads(
    encoder: FeatureEncoder,
    *,
    explicit_none: bool,
) -> tuple[t.Tensor, t.Tensor, dict[str, t.Tensor | None]]:
    feature_ids, magnitudes, mask = _batch()
    if explicit_none:
        packed = encoder(feature_ids, magnitudes, mask, ste=None)
    else:
        packed = encoder(feature_ids, magnitudes, mask)
    output = packed.values
    (output * _loss_weights(output)).sum().backward()
    grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
    }
    return output.detach(), packed.layout.valid_mask.detach(), grads


def test_ste_none_is_byte_identical_in_outputs_and_parameter_gradients() -> None:
    """T-U6a/R12: omitted and explicit None traverse the literal same graph."""
    omitted = _encoder()
    explicit = copy.deepcopy(omitted)

    output_a, mask_a, grads_a = _backward_parameter_grads(omitted, explicit_none=False)
    output_b, mask_b, grads_b = _backward_parameter_grads(explicit, explicit_none=True)

    assert t.equal(output_a, output_b)
    assert t.equal(mask_a, mask_b)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        grad_a, grad_b = grads_a[name], grads_b[name]
        assert (grad_a is None) == (grad_b is None), name
        if grad_a is not None and grad_b is not None:
            assert t.equal(grad_a, grad_b), name


@pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
def test_present_ste_is_value_zero_and_bitwise_forward_identical(form: SteForm) -> None:
    """T-U6b: both estimator forms add an exact forward zero."""
    encoder = _encoder()
    feature_ids, magnitudes, mask = _batch()
    baseline = encoder(feature_ids, magnitudes, mask)
    scores = t.randn(2, 2, 8, dtype=t.float64, requires_grad=True)
    sign = t.tensor(
        [[[1.0, -1.0, 0.0], [1.0, 1.0, -1.0]], [[-1.0, 1.0, 0.0], [1.0, -1.0, 1.0]]],
        dtype=t.float64,
    )

    actual = encoder(
        feature_ids,
        magnitudes,
        mask,
        ste=SteInputs(scores=scores, sign=sign, form=form),
    )

    assert isinstance(actual, PackedEncoderOutput)
    assert t.equal(actual.values, baseline.values)
    assert t.equal(actual.layout.valid_mask, baseline.layout.valid_mask)
    assert t.equal(actual.layout.flat_indices, baseline.layout.flat_indices)
    assert t.equal(actual.layout.batch_indices, baseline.layout.batch_indices)
    assert t.equal(actual.layout.segment_indices, baseline.layout.segment_indices)


@pytest.mark.parametrize(
    ("scores_shape", "sign_shape", "message"),
    [
        ((2, 2, 7), (2, 2, 3), "ste.scores shape"),
        ((2, 2, 8), (2, 2, 4), "ste.sign shape"),
    ],
)
def test_ste_shapes_are_validated_eagerly(
    scores_shape: tuple[int, ...],
    sign_shape: tuple[int, ...],
    message: str,
) -> None:
    """T-U6c: carrier and sign shapes are exact contracts, not broadcastable hints."""
    encoder = _encoder()
    feature_ids, magnitudes, mask = _batch()
    ste = SteInputs(
        scores=t.zeros(scores_shape, dtype=t.float64, requires_grad=True),
        sign=t.zeros(sign_shape, dtype=t.float64),
        form=SteForm.DENSE,
    )

    with pytest.raises(SelfCondTrainingError, match=message):
        encoder(feature_ids, magnitudes, mask, ste=ste)


def test_ste_on_compacted_encoder_rejects_uncompacted_score_width() -> None:
    """T-U6c/C2 defense in depth: a compacted encoder scores in COMPACT space.

    Compacted self-conditioning is supported, so the axis check is no longer
    "reject any compacted encoder" but "reject scores on the wrong axis": the
    embedding table has ``feature_vocab.size`` rows, so full-SAE-width scores
    are the error case.
    """
    vocab = FeatureVocab(n_sae_features=8, original_ids=(0, 1, 2, 3))
    encoder = _encoder(feature_vocab=vocab)
    feature_ids = t.tensor([[[0, 1]]], dtype=t.int64)
    magnitudes = t.tensor([[[0, 1]]], dtype=t.int64)
    mask = t.ones_like(feature_ids, dtype=t.bool)
    ste = SteInputs(
        scores=t.zeros(1, 1, 8, dtype=t.float64, requires_grad=True),
        sign=t.ones(1, 1, 2, dtype=t.float64),
        form=SteForm.DENSE,
    )

    with pytest.raises(SelfCondTrainingError, match=r"\(1, 1, 4\)"):
        encoder(feature_ids, magnitudes, mask, ste=ste)


def test_every_ste_use_in_forward_is_none_gated() -> None:
    """T-U6a/R12: source-level pin against work on the packed non-STE path."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(FeatureEncoder._encode_packed)))
    guarded_uses: set[int] = set()
    guard_test_uses: set[int] = set()
    guards = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_text = ast.unparse(node.test)
        if "ste is not None" not in test_text:
            continue
        guards += 1
        guard_test_uses.update(
            id(child)
            for child in ast.walk(node.test)
            if isinstance(child, ast.Name) and child.id == "ste"
        )
        for statement in node.body:
            guarded_uses.update(
                id(child)
                for child in ast.walk(statement)
                if isinstance(child, ast.Name) and child.id == "ste"
            )

    all_uses = {
        id(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "ste" and isinstance(node.ctx, ast.Load)
    }
    assert guards >= 1
    assert all_uses == guarded_uses | guard_test_uses
