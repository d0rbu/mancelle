"""OPT #2 loss parity: the loss-only path matches the gradient path exactly.

Faster-GCG ranks ~64 candidates per step by LOSS only, but the
old ``objective()`` went through ``_evaluate`` which always evaluated a gradient.
``teacher_forced_bpb_loss`` is a forward-only path (no hooks or VJP) sharing the
EXACT forward + BPB-reduction code with ``teacher_forced_bpb_gradient``. The
embedding-only VJP only reads the recorded graph and never mutates the forward
``bpb`` scalar, so the loss must be byte-identical.

These tests pin that byte-identity across decoder-only and encdec, multiple token
sequences, and assert the loss-only path raises the SAME guards (it shares them).
"""

from __future__ import annotations

import pytest
import torch as t

from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.surrogate import (
    teacher_forced_bpb_gradient,
)
from tests.axbench_bpb_adv.full_sequence_reference import teacher_forced_bpb_loss
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _bundle,
    _encdec_bundle,
    _encdec_inputs,
    _generic_decoder_bundle,
    _inputs,
    _token_bytes,
    _ToyDecoder,
    _ToyEncDec,
    _ToyNonFinite,
)


@pytest.mark.parametrize(
    "prompt_tokens,output_tokens",
    [
        ([0, 1, 2, 3], [4, 5]),
        ([0, 7, 8], [9]),
        ([2, 3, 4, 5, 6], [7, 8, 9]),
    ],
)
@pytest.mark.parametrize("value_dims", [(), (2,)])
def test_loss_only_matches_gradient_loss_decoder(
    prompt_tokens: list[int],
    output_tokens: list[int],
    value_dims: tuple[int, ...],
) -> None:
    model = _ToyDecoder(value_dims=value_dims)
    bundle = _bundle(model)
    grad = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=list(output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )
    loss_only = teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=list(output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )
    # Byte-identical (exact ==), not approx.
    assert loss_only == grad.loss


def test_loss_only_matches_gradient_loss_encdec() -> None:
    model = _ToyEncDec()
    bundle = _encdec_bundle(model)
    inputs = _encdec_inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])
    intervention = Intervention.ours(feature_conditions={11: 3})
    grad = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=inputs.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    loss_only = teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=inputs.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    assert loss_only == grad.loss


def test_loss_only_matches_gradient_loss_after_left_truncation() -> None:
    model = _ToyDecoder()
    bundle = _bundle(model, max_seq_len=4)
    inputs = _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])
    grad = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )
    loss_only = teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )
    assert loss_only == grad.loss


def test_loss_only_does_not_mutate_param_grads() -> None:
    """The loss-only path must not leave param grads populated (no backward)."""

    model = _ToyDecoder()
    bundle = _bundle(model)
    # Seed a sentinel grad to prove the loss-only path leaves grads untouched.
    sentinel = t.full_like(model.transformer.wte.weight, 3.0)
    model.transformer.wte.weight.grad = sentinel.clone()

    teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    assert model.transformer.wte.weight.grad is not None
    assert t.equal(model.transformer.wte.weight.grad, sentinel)


def test_loss_only_raises_same_finite_guard() -> None:
    """The loss-only path shares ``_bpb_tensor_from_loss`` -> same finite guard."""

    with pytest.raises(AxBenchBpbAdvError, match="non-finite"):
        teacher_forced_bpb_loss(
            target_bundle=_generic_decoder_bundle(_ToyNonFinite()),
            prompt_tokens=[0, 1, 2, 3],
            output_tokens=[4, 5],
            prefix_context_segments=(),
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


def test_loss_only_restores_training_mode() -> None:
    """The model's training flag is restored exactly like the gradient path."""

    model = _ToyDecoder()
    model.train(True)
    bundle = _bundle(model)
    teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )
    assert model.training is True
