"""Conditioned rsLoRA training for the AxBench encoder-decoder baselines."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig, Linear
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.axbench_base.baselines.hooks import _get_block
from steering.eval.axbench_upstream.encdec_baselines import lora as lora_module
from steering.eval.axbench_upstream.encdec_baselines.capture import (
    FitRowConditioning,
    conditioned_forward_inputs,
    conditioned_loss_forward,
)
from steering.eval.axbench_upstream.encdec_baselines.lora import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_GRAD_ACCUM,
    LORA_RANK,
    LoRATrainingRow,
    RsLoRAAdapter,
    apply_lora_state,
    attach_encdec_lora,
    detach_encdec_lora,
    load_lora,
    save_lora,
    train_encdec_lora,
    validate_lora_state,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    RenderedFitRow,
    targets_from_mask,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from steering.safetensors_io import canonicalize_safetensors_bytes
from tests.conftest import build_tiny_decoder

FIXTURE_PATH = Path("tests/fixtures/axbench_encdec/lora_fixture.safetensors")
_LORA_LAYERS = (2, 5, 7, 9)


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def _build_lora_encdec() -> EncoderDecoderGPT:
    """Mirror conftest's tiny encdec with enough decoder layers for LoRA."""

    width = 32
    model = EncoderDecoderGPT(
        dec_config=GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=10,
            n_head=1,
            n_kv_head=1,
            n_embd=width,
            window_pattern="L",
        ),
        enc_config=EncoderConfig(
            n_embd=width,
            depth=1,
            n_head=1,
            n_kv_head=1,
            head_dim=width,
        ),
        xattn_config=CrossAttentionConfig(frequency=1),
        n_sae_features=PositiveInt.parse(64),
        num_buckets=PositiveInt.parse(4),
        pad_vocab_size_to=128,
    )
    model.init_weights()
    model.eval()
    return model


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Use the repository's CPU-backward pattern for tiny model tests."""

    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _fit_conditioning(token_ids: tuple[int, ...]) -> FitRowConditioning:
    return FitRowConditioning(
        token_ids=token_ids,
        segments=(
            ContextFeatureSegment(
                feature_ids=(2, 5),
                buckets=(1, 2),
                char_start=0,
                char_end=len(token_ids),
            ),
        ),
        segment_assignments=(0,) * len(token_ids),
    )


def _training_row(offset: int = 0) -> LoRATrainingRow:
    token_ids = (10 + offset, 11 + offset, 12 + offset, 13 + offset)
    return LoRATrainingRow(
        rendered=RenderedFitRow(
            ids=token_ids,
            mask=(0, 0, 1, 1),
            prompt_token_count=2,
        ),
        conditioning=_fit_conditioning(token_ids),
    )


def _conditioned_logits(model: nn.Module, row: FitRowConditioning) -> t.Tensor:
    device = next(model.parameters()).device
    enc_ids, enc_mags, enc_mask, seg_map = conditioned_forward_inputs(row, device)
    idx = t.tensor([row.token_ids], dtype=t.int64, device=device)
    return model(
        idx,
        None,
        enc_feature_ids=enc_ids,
        enc_magnitudes=enc_mags,
        enc_mask=enc_mask,
        seg_map=seg_map,
        encoder_strict=True,
    )


def _clone_state_dict(model: nn.Module) -> dict[str, t.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_state_dict_equal(
    actual: dict[str, t.Tensor],
    expected: dict[str, t.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert t.equal(value, expected[name]), name


def test_adapter_closed_form_and_peft_fixture_are_bitwise_exact() -> None:
    fixture = load_file(FIXTURE_PATH, device="cpu")
    base = nn.Linear(32, 16, bias=False)
    with t.no_grad():
        base.weight.copy_(fixture["base_weight"])
    adapter = RsLoRAAdapter(base).eval()
    with t.no_grad():
        adapter.lora_A.copy_(fixture["lora_A"])
        adapter.lora_B.copy_(fixture["lora_B"])

    # Clone out of safetensors' mmap-backed storage. Torch 2.13 selects a
    # different CPU GEMM kernel for that mapping than fixture-generation
    # Torch 2.5 did; ordinary owned storage reproduces the PEFT bytes exactly.
    fixture_input = fixture["input"].clone()
    actual = adapter(fixture_input)
    expected_closed_form = base(fixture_input) + (LORA_ALPHA / math.sqrt(LORA_RANK)) * F.linear(
        F.linear(fixture_input, fixture["lora_A"]),
        fixture["lora_B"],
    )

    assert t.equal(actual, expected_closed_form)
    assert t.equal(actual, fixture["output"])


def test_adapter_initialization_matches_stored_peft_distribution() -> None:
    fixture = load_file(FIXTURE_PATH, device="cpu")
    base = nn.Linear(32, 16, bias=False)
    t.manual_seed(1234)
    adapter = RsLoRAAdapter(base)

    # PEFT constructs nn.Linear A and B (both running reset_parameters and
    # consuming RNG) before reset_lora_parameters initializes A again. Our
    # raw Parameters consume no RNG before the same Kaiming initialization,
    # so manual_seed(1234) cannot produce byte-identical tensors.
    stored_initial = fixture["lora_A_initial"]
    bound = 1.0 / math.sqrt(32)
    expected_std = bound / math.sqrt(3)
    assert adapter.lora_A.shape == stored_initial.shape == (LORA_RANK, 32)
    assert not t.equal(adapter.lora_A, stored_initial)
    # EXACT oracle for our side: under the same seed, our adapter's A must be
    # byte-identical to a direct kaiming_uniform_(a=sqrt(5)) draw with no
    # other RNG consumption — pinning both the initializer family and its
    # scale exactly (a rescaled or non-Kaiming bounded init fails here).
    expected_ours = t.empty(LORA_RANK, 32)
    t.manual_seed(1234)
    nn.init.kaiming_uniform_(expected_ours, a=math.sqrt(5))
    assert t.equal(adapter.lora_A.detach(), expected_ours)
    # PEFT's stored draw keeps the distribution-level sanity bounds only
    # (its extra pre-init RNG consumption forbids an exact replay).
    stored_value = stored_initial.detach()
    assert float(stored_value.abs().max()) <= bound
    assert abs(float(stored_value.mean())) <= bound / 4
    assert 0.75 * expected_std <= float(stored_value.std()) <= 1.25 * expected_std
    assert t.count_nonzero(adapter.lora_B).item() == 0
    assert adapter.lora_A.dtype == t.float32
    assert adapter.lora_B.dtype == t.float32


def test_adapter_bfloat16_boundary_matches_peft_fp32_path_and_falsifies_downcast() -> None:
    base = Linear(4, 3, bias=False)
    adapter = RsLoRAAdapter(base).eval()
    with t.no_grad():
        base.weight.copy_(t.linspace(-0.7, 0.8, steps=12).reshape(3, 4))
        adapter.lora_A.copy_(t.linspace(-1.3, 1.1, steps=16).reshape(4, 4))
        adapter.lora_B.copy_(t.linspace(-0.9, 1.4, steps=12).reshape(3, 4))
    inputs = t.tensor(
        [[0.333, -1.75, 2.125, -0.0625], [-0.875, 0.15625, 1.375, 2.75]],
        dtype=t.bfloat16,
    )

    base_result = base(inputs)
    fp32_delta = (
        F.linear(
            F.linear(inputs.to(t.float32), adapter.lora_A),
            adapter.lora_B,
        )
        * adapter.scaling
    )
    expected = base_result + fp32_delta.to(base_result.dtype)
    downcast_delta = (
        F.linear(
            F.linear(inputs, adapter.lora_A.to(t.bfloat16)),
            adapter.lora_B.to(t.bfloat16),
        )
        * adapter.scaling
    )
    downcast_variant = base_result + downcast_delta

    actual = adapter(inputs)

    assert actual.dtype == t.bfloat16
    assert t.equal(actual, expected)
    assert not t.equal(actual, downcast_variant)


def test_adapter_dropout_is_stochastic_only_in_train_mode_and_eval_matches_fixture() -> None:
    fixture = load_file(FIXTURE_PATH, device="cpu")
    base = nn.Linear(32, 16, bias=False)
    with t.no_grad():
        base.weight.copy_(fixture["base_weight"])
    adapter = RsLoRAAdapter(base)
    with t.no_grad():
        adapter.lora_A.copy_(fixture["lora_A"])
        adapter.lora_B.copy_(fixture["lora_B"])
    inputs = fixture["input"].clone()
    base_result = base(inputs)

    adapter.train()
    t.manual_seed(101)
    first_delta = adapter(inputs) - base_result
    t.manual_seed(202)
    second_delta = adapter(inputs) - base_result
    assert not t.equal(first_delta, second_delta)

    adapter.eval()
    t.manual_seed(303)
    first_eval = adapter(inputs)
    t.manual_seed(404)
    second_eval = adapter(inputs)
    assert t.equal(first_eval, second_eval)
    assert t.equal(first_eval, fixture["output"])


def test_attach_is_eval_noop_and_targets_exact_layers() -> None:
    t.manual_seed(0)
    model = _build_lora_encdec().eval()
    row = _training_row().conditioning
    assert not [module for module in model.modules() if isinstance(module, nn.Dropout)]
    originals = {layer: _get_block(model, layer).attn.c_proj for layer in _LORA_LAYERS}
    before = _conditioned_logits(model, row)

    refs = attach_encdec_lora(model, _LORA_LAYERS)
    after = _conditioned_logits(model, row)

    assert [ref.layer for ref in refs] == list(_LORA_LAYERS)
    assert all(not ref.adapter.training for ref in refs)
    assert t.equal(after, before)
    assert len([module for module in model.modules() if isinstance(module, nn.Dropout)]) == len(
        _LORA_LAYERS
    )
    for ref in refs:
        assert ref.original is originals[ref.layer]
        assert _get_block(model, ref.layer).attn.c_proj is ref.adapter


def test_detach_restores_original_module_identity_and_state_bytes() -> None:
    model = _build_lora_encdec()
    before = _clone_state_dict(model)
    originals = {layer: _get_block(model, layer).attn.c_proj for layer in _LORA_LAYERS}

    refs = attach_encdec_lora(model, _LORA_LAYERS)
    detach_encdec_lora(model, refs)

    for layer in _LORA_LAYERS:
        assert _get_block(model, layer).attn.c_proj is originals[layer]
    _assert_state_dict_equal(model.state_dict(), before)


def test_attach_failure_is_transactional(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_lora_encdec()
    originals = {layer: _get_block(model, layer).attn.c_proj for layer in _LORA_LAYERS}
    real_get_block = lora_module._get_block

    def fail_third_layer(candidate: nn.Module, layer: int) -> nn.Module:
        if layer == _LORA_LAYERS[2]:
            raise RuntimeError("third layer access failed")
        return real_get_block(candidate, layer)

    monkeypatch.setattr(lora_module, "_get_block", fail_third_layer)
    with pytest.raises(RuntimeError, match="third layer access failed"):
        attach_encdec_lora(model, _LORA_LAYERS)

    for layer in _LORA_LAYERS:
        assert _get_block(model, layer).attn.c_proj is originals[layer]


def test_wrong_layer_attachment_falsifies_target_layer_delta() -> None:
    model = _build_lora_encdec().eval()
    row = _training_row().conditioning
    target_original = _get_block(model, _LORA_LAYERS[0]).attn.c_proj

    wrong_refs = attach_encdec_lora(model, (1,))
    with t.no_grad():
        wrong_refs[0].adapter.lora_B.fill_(0.1)
    wrong_layer_logits = _conditioned_logits(model, row)

    assert _get_block(model, _LORA_LAYERS[0]).attn.c_proj is target_original
    assert wrong_refs[0].layer == 1
    assert _get_block(model, 1).attn.c_proj is wrong_refs[0].adapter

    wrong_A = wrong_refs[0].adapter.lora_A.detach().clone()
    wrong_B = wrong_refs[0].adapter.lora_B.detach().clone()
    detach_encdec_lora(model, wrong_refs)
    correct_refs = attach_encdec_lora(model, (_LORA_LAYERS[0],))
    with t.no_grad():
        correct_refs[0].adapter.lora_A.copy_(wrong_A)
        correct_refs[0].adapter.lora_B.copy_(wrong_B)
    correct_layer_logits = _conditioned_logits(model, row)

    assert not t.equal(wrong_layer_logits, correct_layer_logits)


def test_six_row_sum_over_tokens_matches_padded_decoder_batch_mean() -> None:
    """The two loss forms differ only by harmless kernel accumulation order."""

    t.manual_seed(0)
    model = build_tiny_decoder().eval()
    lengths = (4, 5, 6, 7, 8, 9)
    rows = [tuple(range(10, 10 + length)) for length in lengths]
    masks = [(0, 0, *([1] * (length - 2))) for length in lengths]
    targets = [targets_from_mask(ids, mask) for ids, mask in zip(rows, masks, strict=True)]
    row_sum = sum(
        model(
            t.tensor([ids], dtype=t.int64),
            t.tensor([row_targets], dtype=t.int64),
            loss_reduction="sum",
        )
        for ids, row_targets in zip(rows, targets, strict=True)
    )
    token_count = sum(target != -1 for row_targets in targets for target in row_targets)
    accumulated_mean = row_sum / token_count

    max_length = max(lengths)
    padded_ids = [list(ids) + [0] * (max_length - len(ids)) for ids in rows]
    padded_targets = [
        row_targets + [-1] * (max_length - len(row_targets)) for row_targets in targets
    ]
    batched_mean = model(
        t.tensor(padded_ids, dtype=t.int64),
        t.tensor(padded_targets, dtype=t.int64),
        loss_reduction="mean",
    )

    # CPU SDPA and cross-entropy see different batch shapes, so their fp32
    # reduction order is not promised bitwise; the token-mean identity holds.
    assert t.allclose(accumulated_mean, batched_mean, rtol=0.0, atol=1e-6)


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_proj = nn.Linear(4, 4, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()


class _TrainerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict({"h": nn.ModuleList([_Block() for _ in range(10)])})

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:
        assert kwargs["loss_reduction"] == "sum"
        x = t.tensor([[float(idx[0, 0].item() % 7), 1.0, -0.5, 0.25]])
        total = t.zeros((), dtype=t.float32)
        for layer in _LORA_LAYERS:
            projection = cast(RsLoRAAdapter, _get_block(self, layer).attn.c_proj)
            total = total + projection(x).square().sum()
        return total * targets.ne(-1).sum()


class _FakeAcceleratorParameter:
    """Minimal parameter surface for accelerator-dispatch failure tests."""

    def __init__(self, device_type: str) -> None:
        self.device = t.device(device_type)
        self.requires_grad = True
        self.grad = None

    def requires_grad_(self, value: bool) -> _FakeAcceleratorParameter:
        self.requires_grad = value
        return self


class _FakeAcceleratorModel:
    """Stop before tensor work while exercising RNG ownership and cleanup."""

    def __init__(self, device_type: str) -> None:
        self.training = False
        self.parameter = _FakeAcceleratorParameter(device_type)

    def parameters(self):
        return iter((self.parameter,))

    def named_parameters(self):
        return iter((("parameter", self.parameter),))

    def train(self, mode: bool = True) -> _FakeAcceleratorModel:
        self.training = mode
        return self


def test_trainer_normalizes_batch_loss_before_optimizer_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_losses = {
        10: 1.0,
        11: 2.0,
        12: 4.0,
        13: 8.0,
        14: 16.0,
        15: 32.0,
    }
    observed_token_counts: list[int] = []

    def controlled_forward(
        model: nn.Module,
        idx: t.Tensor,
        targets: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:
        assert kwargs["loss_reduction"] == "sum"
        observed_token_counts.append(int(targets.ne(-1).sum()))
        adapter = cast(RsLoRAAdapter, _get_block(model, _LORA_LAYERS[0]).attn.c_proj)
        anchor = adapter.lora_A[0, 0]
        value = controlled_losses[int(idx[0, 0].item())]
        return (anchor - anchor.detach() + 1.0) * value

    real_adamw = t.optim.AdamW
    optimizer_visible_losses: list[float] = []

    class _RecordingAdamW(real_adamw):
        def step(self, closure=None):
            first_parameter = self.param_groups[0]["params"][0]
            assert first_parameter.grad is not None
            optimizer_visible_losses.append(float(first_parameter.grad[0, 0]))
            return super().step(closure)

    monkeypatch.setattr(_TrainerModel, "forward", controlled_forward)
    monkeypatch.setattr(t.optim, "AdamW", _RecordingAdamW)
    rows = [_training_row(index) for index in range(6)]

    train_encdec_lora(
        _TrainerModel().eval(),
        rows,
        layers=_LORA_LAYERS,
        base_seed=5,
        concept_id=2,
        surface="batch normalization",
        epochs=1,
    )

    expected = sum(controlled_losses.values()) / sum(observed_token_counts) / LORA_GRAD_ACCUM
    assert observed_token_counts == [2] * 6
    assert optimizer_visible_losses == [expected]


def test_training_is_seed_deterministic_and_preserves_base_bytes(
    tmp_path: Path,
) -> None:
    t.manual_seed(90)
    template = _TrainerModel().eval()
    template_state = _clone_state_dict(template)
    saved_bytes: list[bytes] = []
    trained_states: list[dict[str, t.Tensor]] = []

    for run_index, seed in enumerate((17, 17, 18)):
        model = _TrainerModel().eval()
        model.load_state_dict(template_state)
        before = _clone_state_dict(model)
        requires_grad_before = {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        }
        state = train_encdec_lora(
            model,
            [_training_row(index) for index in range(8)],
            layers=_LORA_LAYERS,
            base_seed=seed,
            concept_id=3,
            surface="determinism",
            epochs=2,
        )
        path = tmp_path / f"run_{run_index}.safetensors"
        save_lora(path, state, {"seed": seed})
        saved_bytes.append(path.read_bytes())
        trained_states.append(state)
        _assert_state_dict_equal(model.state_dict(), before)
        assert {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        } == requires_grad_before
        assert not model.training

    assert saved_bytes[0] == saved_bytes[1]
    assert saved_bytes[0] != saved_bytes[2]
    assert all(t.equal(trained_states[0][key], trained_states[1][key]) for key in trained_states[0])
    assert any(
        not t.equal(trained_states[0][key], trained_states[2][key]) for key in trained_states[0]
    )


def test_training_builds_conditioned_forward_inputs_once_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = lora_module.conditioned_forward_inputs
    built_tokens: list[tuple[int, ...]] = []

    def recording_builder(
        row: FitRowConditioning,
        device: t.device | str,
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        built_tokens.append(row.token_ids)
        return real_builder(row, device)

    monkeypatch.setattr(lora_module, "conditioned_forward_inputs", recording_builder)
    rows = [_training_row(index) for index in range(3)]

    train_encdec_lora(
        _TrainerModel().eval(),
        rows,
        layers=_LORA_LAYERS,
        base_seed=17,
        concept_id=3,
        surface="precomputed conditioning",
        epochs=4,
    )

    assert built_tokens == [row.conditioning.token_ids for row in rows]


@pytest.mark.parametrize("entry_training", [False, True])
def test_training_restores_mode_requires_grad_and_leaves_no_base_grads(
    monkeypatch: pytest.MonkeyPatch,
    entry_training: bool,
) -> None:
    model = _TrainerModel()
    model.train(entry_training)
    base_parameters = list(model.named_parameters())
    base_parameters[0][1].requires_grad_(False)
    requires_grad_before = {name: parameter.requires_grad for name, parameter in base_parameters}
    before = _clone_state_dict(model)
    adapter_parameter_ids: set[int] = set()
    real_detach = lora_module.detach_encdec_lora

    def recording_detach(
        candidate: nn.Module,
        refs: list[lora_module.AttachedAdapterRef],
    ) -> None:
        adapter_parameter_ids.update(
            id(parameter) for ref in refs for parameter in (ref.adapter.lora_A, ref.adapter.lora_B)
        )
        real_detach(candidate, refs)

    monkeypatch.setattr(lora_module, "detach_encdec_lora", recording_detach)

    train_encdec_lora(
        model,
        [_training_row(index) for index in range(6)],
        layers=_LORA_LAYERS,
        base_seed=19,
        concept_id=6,
        surface="state hygiene",
        epochs=1,
    )

    assert model.training is entry_training
    _assert_state_dict_equal(model.state_dict(), before)
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == requires_grad_before
    for name, parameter in model.named_parameters():
        assert id(parameter) not in adapter_parameter_ids, name
        assert parameter.grad is None, name


def test_epoch_end_partial_accumulation_scheduler_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_scheduler = t.optim.lr_scheduler.LambdaLR
    lr_trace: list[float] = []

    class _RecordingScheduler:
        def __init__(self, optimizer: t.optim.Optimizer, *args: Any, **kwargs: Any) -> None:
            self._optimizer = optimizer
            self._inner = real_scheduler(optimizer, *args, **kwargs)

        def step(self) -> None:
            self._inner.step()
            lr_trace.append(float(self._optimizer.param_groups[0]["lr"]))

    monkeypatch.setattr(t.optim.lr_scheduler, "LambdaLR", _RecordingScheduler)
    model = _TrainerModel().eval()

    train_encdec_lora(
        model,
        [_training_row(index) for index in range(7 * 6)],
        layers=_LORA_LAYERS,
        base_seed=12,
        concept_id=1,
        surface="scheduler trace",
        epochs=2,
    )

    assert len(lr_trace) == 2 * 2
    assert lr_trace == sorted(lr_trace, reverse=True)
    assert all(lr > 0.0 for lr in lr_trace[:-1])
    assert lr_trace[-1] == 0.0


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), float("-inf")))
def test_non_finite_loss_raises_with_surface_and_restores_model(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    def nan_forward(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor(bad_value, requires_grad=True)

    monkeypatch.setattr(_TrainerModel, "forward", nan_forward)
    model = _TrainerModel().train()
    before = _clone_state_dict(model)

    with pytest.raises(ValueError, match="named surface: non-finite rsLoRA loss"):
        train_encdec_lora(
            model,
            [_training_row()],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="named surface",
            epochs=1,
        )

    assert model.training
    _assert_state_dict_equal(model.state_dict(), before)


def test_training_rejects_non_tensor_row_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_TrainerModel, "forward", lambda *_args, **_kwargs: "not-a-tensor")

    with pytest.raises(
        TypeError,
        match=_exact("AxBench encdec-baseline fit row loss forward must return a tensor, got str"),
    ):
        train_encdec_lora(
            _TrainerModel(),
            [_training_row()],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="non-tensor loss",
            epochs=1,
        )


def test_training_rejects_batch_without_supervised_tokens() -> None:
    token_ids = (10, 11, 12, 13)
    row = LoRATrainingRow(
        rendered=RenderedFitRow(
            ids=token_ids,
            mask=(0, 0, 0, 0),
            prompt_token_count=4,
        ),
        conditioning=_fit_conditioning(token_ids),
    )

    with pytest.raises(
        ValueError,
        match=_exact("all-prompt row: LoRA batch has no supervised tokens"),
    ):
        train_encdec_lora(
            _TrainerModel(),
            [row],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="all-prompt row",
            epochs=1,
        )


def test_training_raises_and_clears_any_non_adapter_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contaminated_name = "transformer.h.0.attn.c_proj.weight"
    original_forward = _TrainerModel.forward

    def contaminating_forward(
        model: nn.Module,
        idx: t.Tensor,
        targets: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:
        base_weight = cast(nn.Linear, _get_block(model, 0).attn.c_proj).weight
        base_weight.grad = t.ones_like(base_weight)
        return original_forward(model, idx, targets, **kwargs)

    monkeypatch.setattr(_TrainerModel, "forward", contaminating_forward)
    model = _TrainerModel().eval()
    base_weight = dict(model.named_parameters())[contaminated_name]

    with pytest.raises(
        RuntimeError,
        match=(
            r"^non-adapter parameter retained a gradient after frozen LoRA training: "
            + contaminated_name.replace(".", r"\.")
            + r"$"
        ),
    ):
        train_encdec_lora(
            model,
            [_training_row()],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="gradient contamination",
            epochs=1,
        )

    assert base_weight.grad is None
    assert not model.training


def test_inflight_error_wins_over_base_gradient_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contaminated_name = "transformer.h.0.attn.c_proj.weight"
    warnings: list[str] = []

    def contaminating_error(
        model: nn.Module,
        _idx: t.Tensor,
        targets: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:
        assert targets.shape == (1, 4)
        assert kwargs["loss_reduction"] == "sum"
        base_weight = cast(nn.Linear, _get_block(model, 0).attn.c_proj).weight
        base_weight.grad = t.ones_like(base_weight)
        raise ValueError("primary training failure")

    monkeypatch.setattr(_TrainerModel, "forward", contaminating_error)
    sink = lora_module.logger.add(warnings.append, format="{message}", level="WARNING")
    model = _TrainerModel().eval()
    base_weight = dict(model.named_parameters())[contaminated_name]
    try:
        with pytest.raises(ValueError, match="primary training failure"):
            train_encdec_lora(
                model,
                [_training_row()],
                layers=_LORA_LAYERS,
                base_seed=0,
                concept_id=0,
                surface="in-flight contamination",
                epochs=1,
            )
    finally:
        lora_module.logger.remove(sink)

    assert base_weight.grad is None
    assert warnings == [
        "non-adapter parameter retained a gradient after frozen LoRA training: "
        + contaminated_name
        + "\n"
    ]


def test_success_path_contamination_raises_even_inside_an_outer_except(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leaked-grad raise must key on THIS call's outcome, not ambient state.

    ``sys.exc_info()`` also reflects an exception the CALLER is currently
    handling; a guard based on it downgraded a success-path failure to a
    warning whenever training ran inside an ``except`` block (final merge-gate
    R1 finding). The fix tracks the function body's own success.
    """

    contaminated_name = "transformer.h.0.attn.c_proj.weight"

    original_forward = _TrainerModel.forward

    def contaminating_success(
        model: nn.Module,
        idx: t.Tensor,
        targets: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:
        base_weight = cast(nn.Linear, _get_block(model, 0).attn.c_proj).weight
        base_weight.grad = t.ones_like(base_weight)
        return original_forward(model, idx, targets, **kwargs)

    monkeypatch.setattr(_TrainerModel, "forward", contaminating_success)
    model = _TrainerModel().eval()

    try:
        raise KeyError("ambient exception being handled by the caller")
    except KeyError:
        with pytest.raises(
            RuntimeError,
            match=(
                r"^non-adapter parameter retained a gradient after frozen LoRA training: "
                + contaminated_name.replace(".", r"\.")
                + r"$"
            ),
        ):
            train_encdec_lora(
                model,
                [_training_row()],
                layers=_LORA_LAYERS,
                base_seed=0,
                concept_id=0,
                surface="ambient-exception contamination",
                epochs=1,
            )


def test_training_restores_the_callers_global_rng_state() -> None:
    """Training seeds the global RNG for dropout but must restore it (R2 seal)."""

    model = _TrainerModel().eval()  # constructed FIRST — init consumes RNG
    t.manual_seed(20260826)
    entry_state = t.get_rng_state().clone()
    train_encdec_lora(
        model,
        [_training_row()],
        layers=_LORA_LAYERS,
        base_seed=0,
        concept_id=0,
        surface="rng seal",
        epochs=1,
    )
    assert t.equal(t.get_rng_state(), entry_state)
    # And the next draw matches what an untouched process would produce.
    expected_next = t.rand(3)
    t.manual_seed(20260826)
    assert t.equal(t.rand(3), expected_next)


@pytest.mark.parametrize("device_type", ("cuda", "mps"))
def test_training_restores_only_the_active_accelerator_rng_after_failure(
    device_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accelerator seeding and restoration target exactly the model device."""

    model = _FakeAcceleratorModel(device_type)
    device_state = t.tensor([11, 22, 33], dtype=t.uint8)
    expected_seed = lora_module._derive_lora_seed(23, 7)
    calls: list[tuple[str, object]] = []

    def fail_after_seeding(_model: nn.Module, _layers: tuple[int, ...]):
        raise RuntimeError("stop after accelerator seeding")

    monkeypatch.setattr(lora_module, "attach_encdec_lora", fail_after_seeding)
    if device_type == "cuda":

        def get_cuda_state(device: t.device) -> t.Tensor:
            calls.append(("get", device))
            return device_state.clone()

        def enter_cuda_device(device: t.device):
            calls.append(("device", device))
            return nullcontext()

        def seed_cuda(seed: int) -> None:
            calls.append(("seed", seed))

        def restore_cuda(state: t.Tensor, *, device: t.device) -> None:
            assert t.equal(state, device_state)
            calls.append(("restore", device))

        monkeypatch.setattr(t.cuda, "get_rng_state", get_cuda_state)
        monkeypatch.setattr(t.cuda, "device", enter_cuda_device)
        monkeypatch.setattr(t.cuda, "manual_seed", seed_cuda)
        monkeypatch.setattr(t.cuda, "set_rng_state", restore_cuda)
    else:

        def get_mps_state() -> t.Tensor:
            calls.append(("get", "mps"))
            return device_state.clone()

        def seed_mps(seed: int) -> None:
            calls.append(("seed", seed))

        def restore_mps(state: t.Tensor) -> None:
            assert t.equal(state, device_state)
            calls.append(("restore", "mps"))

        monkeypatch.setattr(t.mps, "get_rng_state", get_mps_state)
        monkeypatch.setattr(t.mps, "manual_seed", seed_mps)
        monkeypatch.setattr(t.mps, "set_rng_state", restore_mps)

    cpu_state = t.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="stop after accelerator seeding"):
        train_encdec_lora(
            cast(nn.Module, model),
            [_training_row()],
            layers=_LORA_LAYERS,
            base_seed=23,
            concept_id=7,
            surface=f"{device_type} RNG dispatch",
            epochs=1,
        )

    assert t.equal(t.get_rng_state(), cpu_state)
    assert model.parameter.requires_grad
    assert not model.training
    if device_type == "cuda":
        assert calls == [
            ("get", t.device("cuda")),
            ("device", t.device("cuda")),
            ("seed", expected_seed),
            ("restore", t.device("cuda")),
        ]
    else:
        assert calls == [
            ("get", "mps"),
            ("seed", expected_seed),
            ("restore", "mps"),
        ]


def test_save_lora_publishes_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A torn serialization leaves no final-path artifact and no stale temp."""

    state = {"layer_2_attn_c_proj_lora_A": t.ones(LORA_RANK, 8)}
    path = tmp_path / "concept_00000_lora.safetensors"

    def torn_save(_tensors: Any, *, metadata: Any) -> bytes:
        del _tensors, metadata
        raise OSError("disk full mid-serialize")

    monkeypatch.setattr(lora_module, "deterministic_safetensors_bytes", torn_save)
    with pytest.raises(OSError, match="disk full mid-serialize"):
        save_lora(path, state, {"rank": LORA_RANK})
    assert not path.exists()

    monkeypatch.undo()
    save_lora(path, state, {"rank": LORA_RANK})
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_apply_lora_state_rejects_non_finite_tensors() -> None:
    model = _TrainerModel()
    refs = attach_encdec_lora(model, _LORA_LAYERS)
    try:
        state = {
            f"layer_{ref.layer}_attn_c_proj_lora_{matrix}": t.zeros_like(
                ref.adapter.lora_A if matrix == "A" else ref.adapter.lora_B
            )
            for ref in refs
            for matrix in ("A", "B")
        }
        bad_key = f"layer_{refs[0].layer}_attn_c_proj_lora_A"
        state[bad_key][0, 0] = float("inf")
        with pytest.raises(
            ValueError,
            match=rf"^LoRA state tensor '{bad_key}' contains non-finite values$",
        ):
            apply_lora_state(refs, state)
    finally:
        detach_encdec_lora(model, refs)


def test_save_load_apply_round_trip_is_exact(tmp_path: Path) -> None:
    source = _TrainerModel()
    refs = attach_encdec_lora(source, _LORA_LAYERS)
    with t.no_grad():
        for index, ref in enumerate(refs):
            ref.adapter.lora_A.fill_(index + 0.25)
            ref.adapter.lora_B.fill_(-index - 0.5)
    state = {
        f"layer_{ref.layer}_attn_c_proj_lora_A": ref.adapter.lora_A.detach().clone() for ref in refs
    } | {
        f"layer_{ref.layer}_attn_c_proj_lora_B": ref.adapter.lora_B.detach().clone() for ref in refs
    }
    path = tmp_path / "lora.safetensors"
    metadata = {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "scaling": LORA_ALPHA / math.sqrt(LORA_RANK),
        "dropout": LORA_DROPOUT,
        "layers": ",".join(map(str, _LORA_LAYERS)),
        "seed": 42,
        "n_rows": 8,
        "steps": 24,
    }

    save_lora(path, state, metadata)
    loaded_state, loaded_metadata = load_lora(path)
    destination = _TrainerModel()
    destination_refs = attach_encdec_lora(destination, _LORA_LAYERS)
    apply_lora_state(destination_refs, loaded_state)

    assert loaded_metadata == {key: str(value) for key, value in metadata.items()}
    assert loaded_state.keys() == state.keys()
    for key in state:
        assert t.equal(loaded_state[key], state[key])
    assert path.read_bytes() == canonicalize_safetensors_bytes(path.read_bytes())

    reordered_path = tmp_path / "lora_reordered.safetensors"
    save_lora(
        reordered_path,
        dict(reversed(list(state.items()))),
        dict(reversed(list(metadata.items()))),
    )
    assert reordered_path.read_bytes() == path.read_bytes()
    for ref in destination_refs:
        assert t.equal(
            ref.adapter.lora_A,
            state[f"layer_{ref.layer}_attn_c_proj_lora_A"],
        )
        assert t.equal(
            ref.adapter.lora_B,
            state[f"layer_{ref.layer}_attn_c_proj_lora_B"],
        )


def test_lora_bytes_are_stable_across_fresh_processes(tmp_path: Path) -> None:
    script = """
from pathlib import Path
import torch as t
from steering.eval.axbench_upstream.encdec_baselines.lora import save_lora

save_lora(
    Path(__import__('sys').argv[1]),
    {
        'layer_2_attn_c_proj_lora_A': t.arange(8, dtype=t.float32).reshape(2, 4),
        'layer_2_attn_c_proj_lora_B': t.arange(8, dtype=t.float32).reshape(4, 2),
    },
    {'rank': 2, 'alpha': 8.0, 'layers': [2], 'seed': 42},
)
"""
    first = tmp_path / "process-1.safetensors"
    second = tmp_path / "process-2.safetensors"

    subprocess.run([sys.executable, "-c", script, str(first)], check=True)
    subprocess.run([sys.executable, "-c", script, str(second)], check=True)

    assert first.read_bytes() == second.read_bytes()


def _zero_adapter_state(
    refs: list[lora_module.AttachedAdapterRef],
) -> dict[str, t.Tensor]:
    return {
        f"layer_{ref.layer}_attn_c_proj_lora_{matrix}": t.zeros_like(parameter)
        for ref in refs
        for matrix, parameter in (("A", ref.adapter.lora_A), ("B", ref.adapter.lora_B))
    }


def test_apply_lora_state_rejects_missing_key() -> None:
    refs = attach_encdec_lora(_TrainerModel(), _LORA_LAYERS)
    state = _zero_adapter_state(refs)
    missing_key = "layer_2_attn_c_proj_lora_A"
    del state[missing_key]

    with pytest.raises(
        ValueError,
        match=_exact(
            "LoRA state keys do not match attached adapters: "
            f"missing=['{missing_key}'], unexpected=[]"
        ),
    ):
        apply_lora_state(refs, state)


def test_apply_lora_state_rejects_unexpected_key() -> None:
    refs = attach_encdec_lora(_TrainerModel(), _LORA_LAYERS)
    state = _zero_adapter_state(refs)
    unexpected_key = "layer_99_attn_c_proj_lora_A"
    state[unexpected_key] = t.zeros(1)

    with pytest.raises(
        ValueError,
        match=_exact(
            "LoRA state keys do not match attached adapters: "
            f"missing=[], unexpected=['{unexpected_key}']"
        ),
    ):
        apply_lora_state(refs, state)


def test_apply_lora_state_rejects_shape_mismatch() -> None:
    refs = attach_encdec_lora(_TrainerModel(), _LORA_LAYERS)
    state = _zero_adapter_state(refs)
    key = "layer_2_attn_c_proj_lora_A"
    state[key] = t.zeros(1, 1)

    with pytest.raises(
        ValueError,
        match=_exact(f"{key} shape (1, 1) does not match adapter shape ({LORA_RANK}, 4)"),
    ):
        apply_lora_state(refs, state)


def test_real_conditioned_loss_decreases_over_thirty_optimizer_steps() -> None:
    t.manual_seed(0)
    model = _build_lora_encdec().eval()
    row = _training_row()
    targets = t.tensor(
        [targets_from_mask(row.rendered.ids, row.rendered.mask)],
        dtype=t.int64,
    )
    initial = (
        conditioned_loss_forward(
            model,
            row.conditioning,
            targets=targets,
            loss_reduction="sum",
        )
        / targets.ne(-1).sum()
    )

    state = train_encdec_lora(
        model,
        [row],
        layers=_LORA_LAYERS,
        base_seed=41,
        concept_id=2,
        surface="loss decrease",
        epochs=30,
    )
    refs = attach_encdec_lora(model, _LORA_LAYERS)
    apply_lora_state(refs, state)
    model.eval()
    final = (
        conditioned_loss_forward(
            model,
            row.conditioning,
            targets=targets,
            loss_reduction="sum",
        )
        / targets.ne(-1).sum()
    )
    detach_encdec_lora(model, refs)

    assert t.isfinite(final)
    assert final < initial


def test_training_rejects_mismatched_rendered_and_conditioning_tokens() -> None:
    model = _TrainerModel()
    row = _training_row()
    mismatched = LoRATrainingRow(
        rendered=row.rendered,
        conditioning=_fit_conditioning((20, 21, 22, 23)),
    )

    with pytest.raises(
        ValueError, match="mismatch: rendered ids must equal conditioning token_ids"
    ):
        train_encdec_lora(
            model,
            [mismatched],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="mismatch",
            epochs=1,
        )


def test_training_rejects_empty_rows() -> None:
    with pytest.raises(
        ValueError,
        match=_exact("empty: LoRA training rows must not be empty"),
    ):
        train_encdec_lora(
            _TrainerModel(),
            [],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="empty",
            epochs=1,
        )


def test_training_rejects_nonpositive_epochs() -> None:
    with pytest.raises(
        ValueError,
        match=_exact("epochs: LoRA training epochs must be positive"),
    ):
        train_encdec_lora(
            _TrainerModel(),
            [_training_row()],
            layers=_LORA_LAYERS,
            base_seed=0,
            concept_id=0,
            surface="epochs",
            epochs=0,
        )


def test_gradient_accumulation_constant_is_vendored_value() -> None:
    assert LORA_GRAD_ACCUM == 6


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "must contain at least one adapter pair"),
        ({"unexpected": t.zeros(2, 2)}, "invalid LoRA state key"),
        ({"layer_2_attn_c_proj_lora_A": t.zeros(4, 4)}, "incomplete A/B pairs"),
        (
            {
                "layer_2_attn_c_proj_lora_A": t.zeros(4, 4, dtype=t.float64),
                "layer_2_attn_c_proj_lora_B": t.zeros(4, 4),
            },
            "finite contiguous CPU float32 matrix",
        ),
        (
            {
                "layer_2_attn_c_proj_lora_A": t.zeros(4),
                "layer_2_attn_c_proj_lora_B": t.zeros(4, 4),
            },
            "finite contiguous CPU float32 matrix",
        ),
        (
            {
                "layer_2_attn_c_proj_lora_A": t.zeros(4, 0),
                "layer_2_attn_c_proj_lora_B": t.zeros(0, 4),
            },
            "finite contiguous CPU float32 matrix",
        ),
        (
            {
                "layer_2_attn_c_proj_lora_A": t.full((4, 4), float("nan")),
                "layer_2_attn_c_proj_lora_B": t.zeros(4, 4),
            },
            "finite contiguous CPU float32 matrix",
        ),
    ],
)
def test_load_lora_rejects_corrupted_tensor_contracts(
    tmp_path: Path,
    state: dict[str, t.Tensor],
    message: str,
) -> None:
    path = tmp_path / "corrupt.safetensors"
    save_file(state, path, metadata={"format": "test"})

    with pytest.raises(ValueError, match=message):
        load_lora(path)


def test_validate_lora_state_binds_exact_target_layers_and_shapes() -> None:
    model = _TrainerModel()
    refs = attach_encdec_lora(model, _LORA_LAYERS)
    state = _zero_adapter_state(refs)
    detach_encdec_lora(model, refs)

    validate_lora_state(model, state, layers=_LORA_LAYERS)

    wrong_shape = {key: value.clone() for key, value in state.items()}
    wrong_shape["layer_2_attn_c_proj_lora_A"] = t.zeros(1, 1)
    with pytest.raises(ValueError, match="violates the target-model contract"):
        validate_lora_state(model, wrong_shape, layers=_LORA_LAYERS)

    missing = {key: value for key, value in state.items() if not key.endswith("lora_B")}
    with pytest.raises(ValueError, match="keys do not match the target model"):
        validate_lora_state(model, missing, layers=_LORA_LAYERS)
