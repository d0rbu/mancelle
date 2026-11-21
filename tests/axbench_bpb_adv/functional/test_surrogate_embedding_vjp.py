"""Exact and stateful regressions for the embedding-only BPB surrogate VJP."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.surrogate as surrogate
from nanochat import flash_attention as flash_attention_module
from nanochat.flash_attention import FlashAttnImpl
from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb_adv.surrogate import (
    BpbGradientResult,
    bpb_surrogate_embedding_weight,
    teacher_forced_bpb_gradient,
    teacher_forced_bpb_gradient_batch,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.axbench_bpb_adv.functional.test_generate_completions_flow import (
    _LORA_ADAPTER,
    _LORA_ALPHA,
    _LORA_LAYER,
    _LORA_RANK,
    _tiny_model,
)
from tests.axbench_bpb_adv.functional.test_generate_completions_flow import (
    _TinyTokenizer as _RealTinyTokenizer,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _bundle,
    _encdec_bundle,
    _generic_decoder_bundle,
    _TinyTokenizer,
    _token_bytes,
    _ToyBadEmbeddingOutput,
    _ToyDecoder,
    _ToyEncDec,
    _ToyNoValueUse,
    _ToyNoWteUse,
)
from tests.axbench_bpb_adv.legacy_surrogate_oracle import (
    legacy_teacher_forced_bpb_gradient,
    legacy_teacher_forced_bpb_gradient_batch,
)

_SAMPLES = [
    ([3, 7, 11], [13, 1]),
    ([5, 9], [4, 8, 6]),
    ([2, 6, 10, 14], [12]),
    ([21], [22, 23]),
]


def _assert_result_equal(got: BpbGradientResult, want: BpbGradientResult) -> None:
    assert got.loss == want.loss
    assert t.equal(got.token_gradient, want.token_gradient)
    assert t.equal(got.embedding_weight, want.embedding_weight)


def _single_call(
    model: t.nn.Module,
    *,
    intervention: Intervention | None = None,
    max_seq_len: int = 16,
    embedding_weight: t.Tensor | None = None,
) -> BpbGradientResult:
    return teacher_forced_bpb_gradient(
        target_bundle=_generic_decoder_bundle(model, max_seq_len=max_seq_len),
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=intervention or Intervention.none(),
        device=t.device("cpu"),
        embedding_weight=embedding_weight,
    )


@pytest.mark.parametrize(
    ("value_dims", "max_seq_len", "intervention"),
    [
        ((), 16, Intervention.none()),
        ((2,), 16, Intervention.none()),
        (
            (2, 1),
            5,
            Intervention.prompting(
                target_model="decoder_only", prompt_prefix_tokens=[30, 31]
            ),
        ),
    ],
)
def test_single_vjp_matches_legacy_backward_exactly(
    value_dims: tuple[int, ...],
    max_seq_len: int,
    intervention: Intervention,
) -> None:
    old_model = _ToyDecoder(value_dims=value_dims)
    new_model = copy.deepcopy(old_model)
    old = legacy_teacher_forced_bpb_gradient(
        target_bundle=_bundle(old_model, max_seq_len=max_seq_len),
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    new = teacher_forced_bpb_gradient(
        target_bundle=_bundle(new_model, max_seq_len=max_seq_len),
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    _assert_result_equal(new, old)
    assert new.token_gradient.abs().sum() > 0


def test_single_vjp_matches_legacy_encdec_ours_with_context_exactly() -> None:
    old_model = _ToyEncDec(value_dims=(2, 1))
    new_model = copy.deepcopy(old_model)
    prompt = [1, 2, 3]
    segments = (
        ContextFeatureSegment(
            feature_ids=(7,),
            buckets=(2,),
            char_start=0,
            char_end=len(_TinyTokenizer().decode(prompt)),
        ),
    )
    intervention = Intervention.ours(feature_conditions={11: 3})
    old = legacy_teacher_forced_bpb_gradient(
        target_bundle=_encdec_bundle(old_model),
        prompt_tokens=prompt,
        output_tokens=[4, 5],
        prefix_context_segments=segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    new = teacher_forced_bpb_gradient(
        target_bundle=_encdec_bundle(new_model),
        prompt_tokens=prompt,
        output_tokens=[4, 5],
        prefix_context_segments=segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    _assert_result_equal(new, old)


def test_single_vjp_preserves_cached_embedding_identity() -> None:
    old_model = _ToyDecoder(value_dims=(2, 1))
    new_model = copy.deepcopy(old_model)
    old_cache = bpb_surrogate_embedding_weight(old_model)
    new_cache = bpb_surrogate_embedding_weight(new_model)
    old = legacy_teacher_forced_bpb_gradient(
        target_bundle=_bundle(old_model),
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        embedding_weight=old_cache,
    )
    new = teacher_forced_bpb_gradient(
        target_bundle=_bundle(new_model),
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        embedding_weight=new_cache,
    )
    _assert_result_equal(new, old)
    assert old.embedding_weight is old_cache
    assert new.embedding_weight is new_cache


@pytest.mark.parametrize("value_dims", [(), (2, 1)])
@pytest.mark.parametrize("device_batch_size", [1, 2, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_batched_vjp_matches_legacy_backward_exactly(
    value_dims: tuple[int, ...],
    device_batch_size: int,
    reverse: bool,
) -> None:
    samples = list(reversed(_SAMPLES)) if reverse else list(_SAMPLES)
    old_model = _ToyDecoder(value_dims=value_dims)
    new_model = copy.deepcopy(old_model)
    old_cache = bpb_surrogate_embedding_weight(old_model)
    new_cache = bpb_surrogate_embedding_weight(new_model)
    prompts = [prompt for prompt, _ in samples]
    outputs = [output for _, output in samples]
    segments: list[tuple[ContextFeatureSegment, ...]] = [() for _ in samples]
    token_bytes = _token_bytes()
    intervention = Intervention.none()
    device = t.device("cpu")
    old = legacy_teacher_forced_bpb_gradient_batch(
        target_bundle=_bundle(old_model, max_seq_len=4),
        prompts=prompts,
        outputs=outputs,
        prefix_context_segments_list=segments,
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=old_cache,
        device_batch_size=device_batch_size,
    )
    new = teacher_forced_bpb_gradient_batch(
        target_bundle=_bundle(new_model, max_seq_len=4),
        prompts=prompts,
        outputs=outputs,
        prefix_context_segments_list=segments,
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=new_cache,
        device_batch_size=PositiveInt.parse(device_batch_size),
    )
    for got, want in zip(new, old, strict=True):
        _assert_result_equal(got, want)
        assert got.embedding_weight is new_cache
        assert want.embedding_weight is old_cache
    assert new[0].token_gradient.abs().sum() > 0


def _seed_parameter_gradients(
    model: t.nn.Module,
    *,
    flag: Callable[[int], bool],
) -> dict[t.nn.Parameter, tuple[t.Tensor, t.Tensor, bool]]:
    snapshot: dict[t.nn.Parameter, tuple[t.Tensor, t.Tensor, bool]] = {}
    for index, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(flag(index))
        gradient = t.full_like(parameter, float(index + 1))
        parameter.grad = gradient
        snapshot[parameter] = (gradient, gradient.detach().clone(), flag(index))
    return snapshot


def _assert_parameter_state_unchanged(
    snapshot: dict[t.nn.Parameter, tuple[t.Tensor, t.Tensor, bool]],
) -> None:
    for parameter, (gradient_object, gradient_value, requires_grad) in snapshot.items():
        assert parameter.grad is gradient_object
        assert t.equal(parameter.grad, gradient_value)
        assert parameter.requires_grad is requires_grad


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param(lambda _index: True, id="all-trainable"),
        pytest.param(lambda index: index % 2 == 0, id="mixed"),
        pytest.param(lambda _index: False, id="all-frozen"),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_vjp_preserves_parameter_grad_objects_flags_and_training_mode(
    flag: Callable[[int], bool],
    batched: bool,
) -> None:
    model = _ToyDecoder(value_dims=(2, 1))
    model.train(True)
    snapshot = _seed_parameter_gradients(model, flag=flag)
    saw_frozen_forward = False

    def _assert_frozen(module: t.nn.Module, _inputs: tuple[Any, ...]) -> None:
        nonlocal saw_frozen_forward
        saw_frozen_forward = True
        assert all(not parameter.requires_grad for parameter in module.parameters())

    handle = model.register_forward_pre_hook(_assert_frozen)
    try:
        if batched:
            [result] = teacher_forced_bpb_gradient_batch(
                target_bundle=_bundle(model),
                prompts=[[0, 1, 2, 3]],
                outputs=[[4, 5]],
                prefix_context_segments_list=[()],
                output_token_bytes=_token_bytes(),
                intervention=Intervention.none(),
                device=t.device("cpu"),
            )
        else:
            result = _single_call(model)
    finally:
        handle.remove()
    assert saw_frozen_forward
    assert result.token_gradient.abs().sum() > 0
    _assert_parameter_state_unchanged(snapshot)
    assert model.training is True


@pytest.mark.parametrize("batched", [False, True])
def test_vjp_creates_no_parameter_gradients(
    batched: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyDecoder(value_dims=(2,))
    for parameter in model.parameters():
        parameter.grad = None

    def _backward_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("surrogate VJP must not call Tensor.backward")

    monkeypatch.setattr(t.Tensor, "backward", _backward_forbidden)
    if batched:
        results = teacher_forced_bpb_gradient_batch(
            target_bundle=_bundle(model),
            prompts=[sample[0] for sample in _SAMPLES[:2]],
            outputs=[sample[1] for sample in _SAMPLES[:2]],
            prefix_context_segments_list=[(), ()],
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )
    else:
        results = [_single_call(model)]
    assert all(result.token_gradient.abs().sum() > 0 for result in results)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not hasattr(surrogate, "_snapshot_and_clear_grads")


class _ToyForwardError(_ToyDecoder):
    def forward(self, *_args: Any, **_kwargs: Any) -> t.Tensor:
        raise RuntimeError("synthetic forward failure")


class _TupleValueEmbedding(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.zeros(32, 2))

    def forward(self, idx: t.Tensor) -> tuple[t.Tensor]:
        return (t.nn.functional.embedding(idx, self.weight),)


class _ToyBadValueOutput(_ToyDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.value_embeds = t.nn.ModuleDict({"0": _TupleValueEmbedding()})

    def _decoder_signal(self, idx: t.Tensor) -> t.Tensor:
        value = self.value_embeds["0"](idx)
        return t.cat([self.transformer.wte(idx), value[0]], dim=-1)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_ToyForwardError(), "synthetic forward failure"),
        (_ToyNoWteUse(), "failed to capture gradient from transformer.wte"),
        (_ToyNoValueUse(value_dims=(2,)), r"value_embeds\[0\]"),
        (_ToyBadEmbeddingOutput(), "embedding hook wte returned non-Tensor"),
        (_ToyBadValueOutput(), "embedding hook value:0 returned non-Tensor"),
    ],
)
def test_single_vjp_restores_state_after_error(
    model: t.nn.Module,
    match: str,
) -> None:
    model.train(True)
    snapshot = _seed_parameter_gradients(model, flag=lambda index: index % 2 == 0)
    with pytest.raises(Exception, match=match):
        _single_call(model)
    _assert_parameter_state_unchanged(snapshot)
    assert model.training is True


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_ToyForwardError(), "synthetic forward failure"),
        (_ToyNoWteUse(), "failed to capture gradient from transformer.wte"),
        (_ToyNoValueUse(value_dims=(2,)), r"value_embeds\[0\]"),
        (_ToyBadEmbeddingOutput(), "embedding hook wte returned non-Tensor"),
        (_ToyBadValueOutput(), "embedding hook value:0 returned non-Tensor"),
    ],
)
def test_batched_vjp_restores_state_after_error(
    model: t.nn.Module,
    match: str,
) -> None:
    model.train(True)
    snapshot = _seed_parameter_gradients(model, flag=lambda index: index % 2 == 0)
    with pytest.raises(Exception, match=match):
        teacher_forced_bpb_gradient_batch(
            target_bundle=_generic_decoder_bundle(model),
            prompts=[[0, 1, 2, 3], [0, 1]],
            outputs=[[4, 5], [2, 3]],
            prefix_context_segments_list=[(), ()],
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )
    _assert_parameter_state_unchanged(snapshot)
    assert model.training is True


def _real_tiny_bundle(model: t.nn.Module) -> TargetModelBundle:
    return TargetModelBundle(
        model=model,
        tokenizer=_RealTinyTokenizer(),
        max_seq_len=16,
        target_name="decoder_only",
    )


@pytest.mark.parametrize("with_counter", [False, True])
def test_real_tiny_residual_intervention_matches_legacy_exactly(
    with_counter: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flash_attention_module, "_override_impl", FlashAttnImpl.SDPA
    )
    old_model = _tiny_model()
    new_model = _tiny_model()
    n_embd = int(old_model.config.n_embd)
    intervention = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=t.linspace(-1.0, 1.0, n_embd),
        direction_layer=1,
        coefficient=0.5,
    )
    if with_counter:
        intervention = intervention.with_counter_direction(
            direction=t.linspace(1.0, -1.0, n_embd),
            layer=1,
            coefficient=-0.25,
        )
    old = legacy_teacher_forced_bpb_gradient(
        target_bundle=_real_tiny_bundle(old_model),
        prompt_tokens=[0, 5, 6, 7],
        output_tokens=[11, 12],
        prefix_context_segments=(),
        output_token_bytes=t.ones(128, dtype=t.int64),
        intervention=intervention,
        device=t.device("cpu"),
    )
    new = teacher_forced_bpb_gradient(
        target_bundle=_real_tiny_bundle(new_model),
        prompt_tokens=[0, 5, 6, 7],
        output_tokens=[11, 12],
        prefix_context_segments=(),
        output_token_bytes=t.ones(128, dtype=t.int64),
        intervention=intervention,
        device=t.device("cpu"),
    )
    _assert_result_equal(new, old)


def test_real_tiny_lora_attaches_before_freeze_and_detaches_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flash_attention_module, "_override_impl", FlashAttnImpl.SDPA
    )
    old_model = _tiny_model()
    new_model = _tiny_model()
    original_state = {
        name: tensor.detach().clone() for name, tensor in new_model.state_dict().items()
    }
    original_structure = tuple(
        (name, type(module)) for name, module in new_model.named_modules()
    )
    snapshot = _seed_parameter_gradients(
        new_model, flag=lambda index: index % 3 != 0
    )
    intervention = Intervention.lora(adapter_path=_LORA_ADAPTER)
    lora = LoraParams(
        cfg=LoraConfig(rank=_LORA_RANK, alpha=_LORA_ALPHA),
        layer=_LORA_LAYER,
    )
    attached_adapters: list[LoRAAdapter] = []

    def _assert_attached_and_frozen(
        module: t.nn.Module, _inputs: tuple[Any, ...]
    ) -> None:
        adapters = [
            child for child in module.modules() if isinstance(child, LoRAAdapter)
        ]
        assert adapters
        assert all(not parameter.requires_grad for parameter in module.parameters())
        for adapter in adapters:
            assert not adapter.base.weight.requires_grad
            assert not adapter.lora_A.requires_grad
            assert not adapter.lora_B.requires_grad
        attached_adapters.extend(adapters)

    hook = new_model.register_forward_pre_hook(_assert_attached_and_frozen)
    try:
        new = teacher_forced_bpb_gradient(
            target_bundle=_real_tiny_bundle(new_model),
            prompt_tokens=[0, 5, 6, 7],
            output_tokens=[11, 12],
            prefix_context_segments=(),
            output_token_bytes=t.ones(128, dtype=t.int64),
            intervention=intervention,
            device=t.device("cpu"),
            lora=lora,
        )
    finally:
        hook.remove()
    old = legacy_teacher_forced_bpb_gradient(
        target_bundle=_real_tiny_bundle(old_model),
        prompt_tokens=[0, 5, 6, 7],
        output_tokens=[11, 12],
        prefix_context_segments=(),
        output_token_bytes=t.ones(128, dtype=t.int64),
        intervention=intervention,
        device=t.device("cpu"),
        lora=lora,
    )
    _assert_result_equal(new, old)
    assert attached_adapters
    assert all(
        parameter.grad is None
        for adapter in attached_adapters
        for parameter in adapter.parameters()
        if parameter not in snapshot
    )
    _assert_parameter_state_unchanged(snapshot)
    assert tuple((name, type(module)) for name, module in new_model.named_modules()) == (
        original_structure
    )
    after_state = new_model.state_dict()
    assert set(after_state) == set(original_state)
    assert all(t.equal(after_state[name], value) for name, value in original_state.items())

    no_lora = teacher_forced_bpb_gradient(
        target_bundle=_real_tiny_bundle(new_model),
        prompt_tokens=[0, 5, 6, 7],
        output_tokens=[11, 12],
        prefix_context_segments=(),
        output_token_bytes=t.ones(128, dtype=t.int64),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )
    assert new.loss != no_lora.loss
