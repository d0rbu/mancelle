"""Vendored-Python CPU oracles for AxBench GCG pyvene, PEFT, and dtype behavior."""

from __future__ import annotations

import hashlib
import random
import runpy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch as t
from axbench.models.interventions import AdditionIntervention
from peft import LoraConfig, get_peft_model
from pyvene import CollectIntervention, IntervenableConfig, IntervenableModel
from safetensors import safe_open
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    SmolLM3Config,
    SmolLM3ForCausalLM,
)

from scripts.eval.axbench_adv.counter_hooks import residual_counter
from scripts.eval.axbench_adv.vendored import gcg_worker
from tests.axbench_adv.gcg_activation_reference import late_clean_transfers
from tests.axbench_adv.gcg_gradient_reference import full_sequence_logits, rowwise_vjp
from tests.axbench_adv.gcg_layer_reference import full_decoder_depth
from tests.axbench_adv.gcg_response_reference import fresh_response_tensors
from tests.axbench_adv.integration.gcg_worker_reference import device_collector_dimensions

# The grid is dependency-free; importing steering would load root-only packages.
ADV_FULL_FACTORS = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "steering/eval/axbench_adv/method_grid.py")
)["ADV_FULL_FACTORS"]


def test_tensor_sidecar_atomic_replacement_preserves_prior_views(tmp_path: Path) -> None:
    path = tmp_path / "gradients.safetensors"
    first = t.arange(24, dtype=t.float32).reshape(2, 3, 4)
    descriptor = gcg_worker._write_tensor_sidecar(path, "gradients", first)
    assert descriptor == {
        "shape": [2, 3, 4],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert path.stat().st_mode & 0o077 == 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        old = handle.get_tensor("gradients")
    old_member = old[0]
    old_inode = path.stat().st_ino
    path.unlink()
    second = -first
    gcg_worker._write_tensor_sidecar(path, "gradients", second)
    with safe_open(path, framework="pt", device="cpu") as handle:
        new = handle.get_tensor("gradients")
    assert path.stat().st_ino != old_inode
    assert t.equal(old, first)
    assert t.equal(old_member, first[0])
    assert t.equal(new, second)
    path.unlink()
    assert t.equal(old_member, first[0])


def _tiny_llama(*, layers: int = 1) -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    )


def _tiny_gpt_neox(*, layers: int = 1) -> GPTNeoXForCausalLM:
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=4,
            max_position_embeddings=64,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _tiny_qwen2(*, layers: int = 1) -> Qwen2ForCausalLM:
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _tiny_smollm3(*, layers: int = 1) -> SmolLM3ForCausalLM:
    return SmolLM3ForCausalLM(
        SmolLM3Config(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _layer_residual(output: Any) -> t.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_layer_residual(output: Any, residual: t.Tensor) -> Any:
    return (residual, *output[1:]) if isinstance(output, tuple) else residual


@contextmanager
def _manual_positive_hook(
    model: Any,
    addition: AdditionIntervention,
    subspace: dict[str, t.Tensor],
) -> Iterator[None]:
    def add(_module: Any, _inputs: Any, output: Any) -> Any:
        residual = _layer_residual(output)
        positive = addition(residual, subspaces=subspace).to(residual.dtype)
        return _replace_layer_residual(output, positive)

    handle = gcg_worker.decoder_layers(model)[0].register_forward_hook(add)
    try:
        yield
    finally:
        handle.remove()


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


@pytest.mark.parametrize("model_factory", [_tiny_llama, _tiny_gpt_neox, _tiny_qwen2, _tiny_smollm3])
@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe"])
@pytest.mark.parametrize("factor", [0.0, 0.6, 1.0, 3.0])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.slow)])
@pytest.mark.parametrize("inference_prepare", [False, True])
def test_prepared_collector_host_dimension_preserves_residuals_and_vjps(
    tmp_path: Path,
    model_factory: Callable[..., Any],
    method: str,
    factor: float,
    device: str,
    inference_prepare: bool,
) -> None:
    if device.startswith("cuda") and not t.cuda.is_available():
        pytest.skip("collector device qualification requires CUDA")
    t.manual_seed(891)
    model = model_factory(layers=2).to(device=device, dtype=t.bfloat16).eval()
    hidden = model.config.hidden_size
    actual = AdditionIntervention(embed_dim=hidden, low_rank_dimension=1).to(
        device=device, dtype=t.bfloat16
    )
    (tmp_path / "train").mkdir()
    (tmp_path / "inference").mkdir()
    t.save(t.linspace(-0.7, 0.7, hidden).unsqueeze(0), tmp_path / "train/DiffMean_weight.pt")
    t.save(t.zeros(1), tmp_path / "train/DiffMean_bias.pt")
    pd.DataFrame({"concept_id": [0], "DiffMean_max_act": [2.0]}).to_parquet(
        tmp_path / "inference/latent_data.parquet"
    )
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model, ax=actual, max_activations={0: 2.0}),
        concept_id=0,
        device=device,
        factor=factor,
        method=method,
        layer=0,
        max_concepts=1,
        tokenizer=SimpleNamespace(padding_side="right", pad_token_id=0),
        _gcg_main_dump_dir=tmp_path,
    )
    session = gcg_worker.GCGSession(runtime)
    positive = gcg_worker._CompletionLayout(11, "positive", (1, 5, 2), (1, 5), (2,), (3, 4), (1, 1))
    negative = gcg_worker._CompletionLayout(
        12, "negative", (1, 6, 7, 2), (1, 6, 7), (2,), (8, 9), (1, 1)
    )
    session.layouts = (gcg_worker._PairLayout(0, positive, negative),)
    session.device_batch_size = 2
    session.objective = "activation_l2"
    with t.inference_mode(inference_prepare), gcg_worker._preserve_runtime(runtime):
        calibration = session._prepare_activation()
    assert calibration["max_activation"] == 2.0
    assert calibration["factor"] == factor
    assert calibration["resolved_scale"] == 2.0 * factor
    oracle = AdditionIntervention(embed_dim=hidden, low_rank_dimension=1).to(
        device=device, dtype=t.bfloat16
    )
    oracle.proj.weight.data.copy_(t.load(tmp_path / "train/DiffMean_weight.pt", weights_only=True))
    expected_offset = oracle(
        t.zeros(1, 1, hidden, dtype=t.bfloat16, device=device),
        subspaces={
            "idx": t.tensor([0], device=device),
            "max_act": t.tensor([2.0], device=device),
            "mag": t.tensor([factor], device=device),
        },
    )[0, 0]
    assert session._direction_offset is not None
    assert session._direction_offset.dtype is t.float32
    assert t.equal(session._direction_offset, expected_offset)
    assert session._activation_model is not None
    [collector] = [
        item
        for item in session._activation_model.interventions.values()
        if isinstance(item, CollectIntervention)
    ]
    bound = collector.interchange_dim
    assert bound.device.type == "cpu" and bound.dtype == t.long and bound.ndim == 0
    assert int(bound) == hidden
    assert collector.embed_dim.device == t.device(device)
    assert all(parameter.device == t.device(device) for parameter in model.parameters())
    clean = dict(session._clean_hidden)
    assert set(clean) == {11, 12}
    with device_collector_dimensions(session):
        assert collector.interchange_dim.device == t.device(device)
        session._compute_clean_hidden()
    assert collector.interchange_dim is bound
    for sample_id, expected in clean.items():
        t.testing.assert_close(session._clean_hidden[sample_id], expected, rtol=0, atol=0)
    session._clean_hidden = clean
    clean_values = {sample_id: value.clone() for sample_id, value in clean.items()}
    for tokens in ((10,), (10, 11, 12)):
        items = [gcg_worker._SideItem(side, tokens) for side in (negative, positive)]
        for compute_gradient in (False, True):
            with gcg_worker._preserve_runtime(runtime), t.set_grad_enabled(compute_gradient):
                optimized = session._side_batch(items, gradient=compute_gradient)
                with device_collector_dimensions(session), late_clean_transfers(gcg_worker):
                    reference = session._side_batch(items, gradient=compute_gradient)
            for (loss, gradient), (ref_loss, ref_gradient) in zip(
                optimized, reference, strict=True
            ):
                t.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
                if compute_gradient:
                    assert gradient is not None and ref_gradient is not None
                    t.testing.assert_close(gradient, ref_gradient, rtol=0, atol=0)
                    assert gradient.shape == (len(tokens), hidden)
                    assert bool(t.isfinite(gradient).all()) and bool(gradient.ne(0).any())
                else:
                    assert gradient is ref_gradient is None
            assert session._clean_hidden is clean
            for sample_id, expected in clean_values.items():
                assert session._clean_hidden[sample_id].device.type == "cpu"
                t.testing.assert_close(session._clean_hidden[sample_id], expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="collector reference restoration"):
        with device_collector_dimensions(session):
            raise RuntimeError("collector reference restoration")
    assert collector.interchange_dim is bound
    session.close()
    assert session._activation_model is None and not session._clean_hidden


@pytest.mark.parametrize("fail_after_backward", [False, True])
def test_runtime_guard_restores_existing_gradient_contents(fail_after_backward: bool) -> None:
    first_layer = t.nn.Linear(3, 3)
    model = t.nn.Sequential(first_layer, t.nn.Tanh(), t.nn.Linear(3, 1))
    parameter = first_layer.weight
    sentinel = t.full_like(parameter, 0.125)
    expected = sentinel.clone()
    parameter.grad = sentinel
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model),
        tokenizer=SimpleNamespace(padding_side="left"),
    )

    def mutate_existing_gradient(
        _module: object, _grad_input: object, _grad_output: object
    ) -> None:
        with t.no_grad():
            assert parameter.grad is sentinel
            parameter.grad.add_(4.0)

    handle = model[-1].register_full_backward_hook(mutate_existing_gradient)

    def run() -> None:
        inputs = t.randn(2, 3, requires_grad=True)
        with gcg_worker._preserve_runtime(runtime):
            output = model(inputs).sum()
            t.autograd.grad(output, inputs)
            assert not t.equal(sentinel, expected)
            if fail_after_backward:
                raise RuntimeError("failure after backward")

    try:
        if fail_after_backward:
            with pytest.raises(RuntimeError, match="failure after backward"):
                run()
        else:
            run()
        assert parameter.grad is sentinel
        assert t.equal(parameter.grad, expected)
    finally:
        handle.remove()


@pytest.mark.parametrize(
    ("model_factory", "component", "attention_attribute"),
    [
        pytest.param(_tiny_llama, "model.layers[0].output", "self_attn", id="llama"),
        pytest.param(_tiny_gpt_neox, "block_output", "attention", id="gpt_neox"),
    ],
)
def test_real_pyvene_additive_vjp_and_capture_restore_hooks_and_primary_failure(
    model_factory: Callable[..., Any],
    component: str,
    attention_attribute: str,
) -> None:
    model = model_factory(layers=2).eval()
    original_layers = gcg_worker.decoder_layers(model)
    actual = AdditionIntervention(embed_dim=16, low_rank_dimension=1)
    collector = CollectIntervention(embed_dim=16, keep_last_dim=True)
    ax_model = IntervenableModel(
        IntervenableConfig(
            representations=[
                {"layer": 0, "component": component, "intervention": actual},
            ]
        ),
        model,
    )
    ax_model.set_device("cpu")
    activation_model = IntervenableModel(
        IntervenableConfig(
            representations=[
                {
                    "layer": 0,
                    "component": component,
                    "intervention": actual,
                },
                {
                    "layer": 0,
                    "component": component,
                    "intervention": collector,
                },
            ]
        ),
        model,
    )
    activation_model.set_device("cpu")
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(
            model=model,
            ax=actual,
            ax_model=ax_model,
            max_activations={0: 2.0},
        ),
        concept_id=0,
        device="cpu",
        factor=1.0,
        method="DiffMean",
        layer=0,
        tokenizer=SimpleNamespace(padding_side="left"),
    )
    session = gcg_worker.GCGSession(runtime)
    session._activation_model = activation_model
    layer = gcg_worker.decoder_layers(model)[0]
    model.train()
    getattr(layer, attention_attribute).eval()
    actual.train()
    actual.proj.eval()
    base_modules = tuple(model.modules())
    known_modules = set(base_modules)
    mode_modules = base_modules + tuple(
        child for child in actual.modules() if child not in known_modules
    )
    expected_modes = tuple(child.training for child in mode_modules)
    frozen_parameter = model.get_input_embeddings().weight
    frozen_parameter.requires_grad_(False)
    grad_parameter = model.get_output_embeddings().weight
    grad_sentinel = t.full_like(grad_parameter, 19.0)
    grad_parameter.grad = grad_sentinel
    calls: list[int] = []

    def external_hook(_module: Any, _inputs: Any, _output: Any) -> None:
        calls.append(1)

    handle = layer.register_forward_hook(external_hook, always_call=True)
    original_hook_mapping = layer._forward_hooks
    original_hooks = tuple(layer._forward_hooks.items())
    ids = t.tensor([[1, 2, 3]], dtype=t.long)
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    attention = t.ones_like(ids)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = t.random.get_rng_state()

    with gcg_worker._preserve_runtime(runtime), t.enable_grad():
        logits = gcg_worker._forward(runtime, embeds=embeds, attention_mask=attention).logits
        hidden = session._capture_hidden(embeds, attention)
        [logit_gradient] = t.autograd.grad(
            logits[0, -1].square().sum(), [embeds], retain_graph=True
        )
        [hidden_gradient] = t.autograd.grad(hidden.square().sum(), [embeds])
    assert hidden.shape == (1, 3, 16)
    assert gcg_worker.decoder_layers(model) is original_layers
    assert hidden.shape == embeds.shape
    for gradient in (logit_gradient, hidden_gradient):
        assert bool(t.isfinite(gradient).all())
        assert bool(gradient.abs().sum() > 0)
    assert layer._forward_hooks is original_hook_mapping
    assert tuple(layer._forward_hooks.items()) == original_hooks
    assert tuple(child.training for child in mode_modules) == expected_modes
    assert frozen_parameter.requires_grad is False
    assert grad_parameter.grad is grad_sentinel
    assert t.equal(grad_parameter.grad, t.full_like(grad_parameter, 19.0))
    assert calls == []
    model(inputs_embeds=embeds.detach(), attention_mask=attention, use_cache=False)
    assert calls == [1]
    assert random.getstate() == python_rng
    assert _numpy_state_equal(np.random.get_state(), numpy_rng)
    assert t.equal(t.random.get_rng_state(), torch_rng)

    class _PrimaryFailure(RuntimeError):
        pass

    class _RaiseAfterForward:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            activation_model(*args, **kwargs)
            random.random()
            np.random.random()
            t.rand(1)
            runtime.tokenizer.padding_side = "right"
            model.train()
            actual.train()
            raise _PrimaryFailure("primary forward failure")

    session._activation_model = _RaiseAfterForward()
    with pytest.raises(_PrimaryFailure, match="primary forward failure"):
        with gcg_worker._preserve_runtime(runtime):
            session._capture_hidden(embeds, attention)
    assert gcg_worker.decoder_layers(model) is original_layers
    assert layer._forward_hooks is original_hook_mapping
    assert tuple(layer._forward_hooks.items()) == original_hooks
    assert tuple(child.training for child in mode_modules) == expected_modes
    assert frozen_parameter.requires_grad is False
    assert grad_parameter.grad is grad_sentinel
    assert t.equal(grad_parameter.grad, t.full_like(grad_parameter, 19.0))
    assert runtime.tokenizer.padding_side == "left"
    assert random.getstate() == python_rng
    assert _numpy_state_equal(np.random.get_state(), numpy_rng)
    assert t.equal(t.random.get_rng_state(), torch_rng)
    model(inputs_embeds=embeds.detach(), attention_mask=attention, use_cache=False)
    assert calls == [1, 1]
    handle.remove()
    assert not layer._forward_hooks
    model(inputs_embeds=embeds.detach(), attention_mask=attention, use_cache=False)
    assert calls == [1, 1]


@pytest.mark.parametrize("factor", ADV_FULL_FACTORS)
def test_actual_addition_intervention_matches_calibration_multiplication_order(
    factor: float,
) -> None:
    intervention = AdditionIntervention(embed_dim=3, low_rank_dimension=1).to(t.bfloat16)
    direction = t.tensor([[1.1171875, -0.62109375, 0.30859375]], dtype=t.bfloat16)
    intervention.proj.weight.data.copy_(direction)
    base = t.tensor([[[0.5, -1.0, 2.0]]], dtype=t.bfloat16)
    subspaces = {
        "idx": t.tensor([0], dtype=t.long),
        "max_act": t.tensor([3.3], dtype=t.float32),
        "mag": t.tensor([factor], dtype=t.float32),
    }
    output = intervention(base, subspaces=subspaces)
    expected_offset = (
        subspaces["max_act"].unsqueeze(-1) * subspaces["mag"].unsqueeze(-1) * direction
    )
    assert expected_offset.dtype is t.float32
    assert output.dtype is t.float32
    assert t.equal(output, base + expected_offset.unsqueeze(1))
    wrongly_scalarized = (
        t.tensor(3.3, dtype=t.float32) * t.tensor(factor, dtype=t.float32) * direction[0]
    )
    assert wrongly_scalarized.dtype is t.bfloat16
    assert t.equal(expected_offset[0], wrongly_scalarized.float()) is (factor == 0.0)


@pytest.mark.parametrize("make_model", [_tiny_llama, _tiny_qwen2, _tiny_smollm3, _tiny_gpt_neox])
def test_counter_runs_after_real_pyvene_addition_in_forward_and_generate(
    make_model: Callable[..., Any],
) -> None:
    t.manual_seed(919)
    model = make_model().eval().to(t.bfloat16)
    addition = AdditionIntervention(embed_dim=16, low_rank_dimension=1).eval().to(t.bfloat16)
    steering_direction = t.tensor(
        [[1.1171875, -0.62109375, 0.30859375, 0.94921875] * 4], dtype=t.bfloat16
    )
    with t.no_grad():
        addition.proj.weight.copy_(steering_direction)
        addition.proj.bias.zero_()
    ax_model = IntervenableModel(
        IntervenableConfig(
            representations=[
                {
                    "layer": 0,
                    "component": (
                        "block_output"
                        if model.config.model_type == "gpt_neox"
                        else "model.layers[0].output"
                    ),
                    "intervention": addition,
                }
            ]
        ),
        model,
    )
    ax_model.set_device("cpu")
    ids = t.tensor([[1, 7, 9]], dtype=t.long)
    base = {"input_ids": ids, "attention_mask": t.ones_like(ids)}
    subspace = {
        "idx": t.tensor([0]),
        "max_act": t.tensor([3.3], dtype=t.float32),
        "mag": t.tensor([1.0], dtype=t.float32),
    }
    direction = t.tensor([0.34375, -0.7109375, 1.0390625, -0.28515625] * 4, dtype=t.float32)
    magnitude = 0.7

    with residual_counter(model, layer=0, direction=direction, magnitude=magnitude):
        with t.no_grad():
            _, pyvene_forward = ax_model(
                base,
                unit_locations=None,
                subspaces=[subspace],
                use_cache=False,
            )
            _, pyvene_generate = ax_model.generate(
                base,
                unit_locations=None,
                intervene_on_prompt=True,
                subspaces=[subspace],
                max_new_tokens=2,
                do_sample=False,
            )
        with _manual_positive_hook(model, addition, subspace), t.no_grad():
            manual_forward = model(**base, use_cache=False)
            manual_generate = model.generate(**base, max_new_tokens=2, do_sample=False)

    assert t.equal(pyvene_forward.logits, manual_forward.logits)
    assert t.equal(pyvene_generate, manual_generate)
    assert "forward" not in vars(model)
    assert not gcg_worker.decoder_layers(model)[0]._forward_hooks

    positive_offset = (
        subspace["max_act"].unsqueeze(-1) * subspace["mag"].unsqueeze(-1) * steering_direction
    )
    counter_offset = magnitude * direction.to(t.bfloat16)
    clean_hidden: list[t.Tensor] = []
    handle = gcg_worker.decoder_layers(model)[0].register_forward_hook(
        lambda _module, _inputs, output: clean_hidden.append(
            _layer_residual(output).detach().clone()
        )
    )
    with t.no_grad():
        model(**base, use_cache=False)
    handle.remove()
    positive_then_counter = (clean_hidden[0].float() + positive_offset.unsqueeze(1)).to(
        t.bfloat16
    ) - counter_offset
    counter_then_positive = (
        (clean_hidden[0] - counter_offset).float() + positive_offset.unsqueeze(1)
    ).to(t.bfloat16)
    assert not t.equal(positive_then_counter, counter_then_positive)


@pytest.mark.parametrize("use_peft", [False, True])
@pytest.mark.parametrize("make_model", [_tiny_llama, _tiny_qwen2, _tiny_smollm3, _tiny_gpt_neox])
def test_counter_changes_real_plain_and_peft_generate_scores(
    use_peft: bool, make_model: Callable[..., Any]
) -> None:
    t.manual_seed(143)
    model = make_model().eval()
    generation_model: Any = model
    if use_peft:
        generation_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=(
                    ["dense"] if model.config.model_type == "gpt_neox" else ["q_proj", "v_proj"]
                ),
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        ).eval()
        with t.no_grad():
            for module in generation_model.modules():
                lora_b = getattr(module, "lora_B", None)
                if lora_b is not None:
                    lora_b["default"].weight.fill_(0.03125)
    ids = t.tensor([[1, 5, 9]], dtype=t.long)
    generate = {
        "input_ids": ids,
        "attention_mask": t.ones_like(ids),
        "max_new_tokens": 2,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    with t.no_grad():
        clean = generation_model.generate(**generate)
    direction = t.linspace(-1.0, 1.0, 16)
    with residual_counter(model, layer=0, direction=direction, magnitude=3.0), t.no_grad():
        countered = generation_model.generate(**generate)
    assert len(clean.scores) == len(countered.scores) == 2
    assert any(
        not t.equal(left, right) for left, right in zip(clean.scores, countered.scores, strict=True)
    )
    assert "forward" not in vars(model)
    assert not gcg_worker.decoder_layers(model)[0]._forward_hooks


def test_real_peft_lora_right_padding_matches_individual_float32_forwards() -> None:
    model = _tiny_llama().eval()
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ),
    ).eval()
    with t.no_grad():
        for module in peft_model.modules():
            lora_b = getattr(module, "lora_B", None)
            if lora_b is not None:
                lora_b["default"].weight.fill_(0.03125)
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model, ax_model=peft_model),
        device="cpu",
        layer=0,
        method="LoRA",
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="right"),
    )
    session = gcg_worker.GCGSession(runtime)
    short = (1, 3, 5, 7)
    long = (2, 4, 6, 8, 10, 12, 14)

    def forward(sequences: list[tuple[int, ...]]) -> t.Tensor:
        embeds, attention = session._padded_embeds(sequences, gradient=False)
        with gcg_worker._preserve_runtime(runtime), t.no_grad():
            return gcg_worker._forward(
                runtime,
                embeds=embeds,
                attention_mask=attention,
            ).logits

    short_logits = forward([short])[0]
    long_logits = forward([long])[0]
    batched = forward([short, long])
    t.testing.assert_close(batched[0, : len(short)], short_logits, rtol=1e-6, atol=1e-6)
    t.testing.assert_close(batched[1, : len(long)], long_logits, rtol=1e-6, atol=1e-6)

    layer = model.model.layers[0]
    model.train()
    layer.self_attn.eval()
    modules = tuple(model.modules())
    expected_modes = tuple(module.training for module in modules)
    adapter_parameter = next(
        parameter for name, parameter in peft_model.named_parameters() if "lora_A" in name
    )
    grad_sentinel = t.full_like(adapter_parameter, 23.0)
    adapter_parameter.grad = grad_sentinel
    expected_requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())
    calls: list[int] = []

    def external_hook(_module: Any, _inputs: Any, _output: Any) -> None:
        calls.append(1)

    handle = layer.register_forward_hook(external_hook, always_call=True)
    original_hook_mapping = layer._forward_hooks
    original_hooks = tuple(original_hook_mapping.items())
    embeds, attention = session._padded_embeds([short], gradient=False)
    with gcg_worker._preserve_runtime(runtime), t.no_grad():
        first = session._capture_hidden(embeds, attention)
        second = session._capture_hidden(embeds, attention)
    assert t.equal(first, second)
    assert calls == [1, 1]
    assert layer._forward_hooks is original_hook_mapping
    assert tuple(layer._forward_hooks.items()) == original_hooks
    assert tuple(module.training for module in modules) == expected_modes
    assert (
        tuple(parameter.requires_grad for parameter in model.parameters()) == expected_requires_grad
    )
    assert adapter_parameter.grad is grad_sentinel
    assert t.equal(adapter_parameter.grad, t.full_like(adapter_parameter, 23.0))
    handle.remove()
    assert not layer._forward_hooks


@pytest.mark.parametrize(
    ("model_factory", "target_modules"),
    [
        pytest.param(_tiny_qwen2, None, id="qwen2-plain"),
        pytest.param(_tiny_smollm3, None, id="smollm3-plain"),
        pytest.param(_tiny_gpt_neox, None, id="gpt_neox-plain"),
        pytest.param(_tiny_smollm3, ["q_proj", "v_proj"], id="smollm3-lora"),
        pytest.param(_tiny_gpt_neox, ["query_key_value"], id="gpt_neox-lora"),
    ],
)
def test_real_host_vjp_and_right_padding(
    model_factory: Any,
    target_modules: list[str] | None,
) -> None:
    model = model_factory().eval()
    benchmark_model = SimpleNamespace(model=model)
    if target_modules is not None:
        benchmark_model.ax_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=target_modules,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        ).eval()
    runtime = SimpleNamespace(
        benchmark_model=benchmark_model,
        device="cpu",
        layer=0,
        method="PromptSteering" if target_modules is None else "LoRA",
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="right"),
    )
    session = gcg_worker.GCGSession(runtime)
    short = (1, 3, 5, 7)
    long = (2, 4, 6, 8, 10, 12, 14)

    def forward(sequences: list[tuple[int, ...]]) -> t.Tensor:
        embeds, attention = session._padded_embeds(sequences, gradient=False)
        with gcg_worker._preserve_runtime(runtime), t.no_grad():
            return gcg_worker._forward(
                runtime,
                embeds=embeds,
                attention_mask=attention,
            ).logits

    short_logits = forward([short])[0]
    long_logits = forward([long])[0]
    batched = forward([short, long])
    t.testing.assert_close(batched[0, : len(short)], short_logits, rtol=1e-6, atol=1e-6)
    t.testing.assert_close(batched[1, : len(long)], long_logits, rtol=1e-6, atol=1e-6)

    embeds, attention = session._padded_embeds([short], gradient=True)
    with gcg_worker._preserve_runtime(runtime), t.enable_grad():
        logits = gcg_worker._forward(
            runtime,
            embeds=embeds,
            attention_mask=attention,
        ).logits
        [gradient] = t.autograd.grad(logits[0, -1].square().sum(), [embeds])
    assert gradient.shape == embeds.shape
    assert bool(t.isfinite(gradient).all())
    assert bool(gradient.abs().sum() > 0)


@pytest.mark.parametrize(
    ("model_factory", "decoder_path"),
    [
        pytest.param(_tiny_smollm3, "model", id="smollm3-layer-28-analogue"),
        pytest.param(_tiny_gpt_neox, "gpt_neox", id="gpt-neox-layer-25-analogue"),
    ],
)
def test_non_qwen_activation_capture_uses_configured_decoder_layer(
    model_factory: Any,
    decoder_path: str,
) -> None:
    model = model_factory(layers=2).eval()
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model),
        device="cpu",
        layer=1,
        method="PromptSteering",
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="right"),
    )
    session = gcg_worker.GCGSession(runtime)
    embeds, attention = session._padded_embeds([(1, 3, 5, 7)], gradient=True)
    decoder = getattr(model, decoder_path)
    layers = decoder.layers
    observed: list[t.Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: observed.append(_layer_residual(output))
        )
        for layer in layers
    ]
    try:
        with gcg_worker._preserve_runtime(runtime), t.enable_grad():
            captured = session._capture_hidden(embeds, attention)
            [gradient] = t.autograd.grad(captured.square().sum(), [embeds])
    finally:
        for handle in handles:
            handle.remove()
    assert len(observed) == 2
    assert captured is observed[1]
    assert not t.equal(captured, observed[0])
    assert bool(t.isfinite(gradient).all())
    assert bool(gradient.abs().sum() > 0)


@pytest.mark.parametrize(
    ("config_class", "model_class", "component", "lora_targets"),
    [
        (LlamaConfig, LlamaForCausalLM, "model.layers[0].output", ["q_proj", "v_proj"]),
        (Qwen2Config, Qwen2ForCausalLM, "model.layers[0].output", ["q_proj", "v_proj"]),
        (SmolLM3Config, SmolLM3ForCausalLM, "model.layers[0].output", ["q_proj", "v_proj"]),
        (GPTNeoXConfig, GPTNeoXForCausalLM, "block_output", ["query_key_value"]),
    ],
)
@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe", "LoRA", "PromptSteering"])
@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_real_hf_joint_vjp_matches_independent_rowwise_reference(
    config_class: Any,
    model_class: Any,
    component: str,
    lora_targets: list[str],
    method: str,
    objective: str,
    dtype: t.dtype,
) -> None:
    t.manual_seed(314)
    model = (
        model_class(
            config_class(
                vocab_size=37,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attention_dropout=0.0,
                pad_token_id=0,
                rotary_pct=1.0,
            )
        )
        .to(dtype=dtype)
        .eval()
    )
    benchmark = SimpleNamespace(model=model, max_activations={0: 1.25})
    if method in ("DiffMean", "LinearProbe"):
        benchmark.ax = AdditionIntervention(embed_dim=16, low_rank_dimension=1).to(dtype)
        benchmark.ax_model = IntervenableModel(
            IntervenableConfig(
                representations=[
                    {
                        "layer": 0,
                        "component": component,
                        "intervention": benchmark.ax,
                    }
                ]
            ),
            model,
        )
        benchmark.ax_model.set_device("cpu")
    elif method == "LoRA":
        benchmark.ax_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=lora_targets,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        ).eval()
        with t.no_grad():
            for module in benchmark.ax_model.modules():
                lora_b = getattr(module, "lora_B", None)
                if lora_b is not None:
                    lora_b["default"].weight.fill_(0.03125)
    runtime = SimpleNamespace(
        benchmark_model=benchmark,
        method=method,
        concept_id=0,
        factor=1.0,
        device="cpu",
        layer=0,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="left"),
    )
    session = gcg_worker.GCGSession(runtime)
    session.objective = objective
    layouts = [
        gcg_worker._CompletionLayout(
            sample_id=index,
            content="test",
            clean_prompt_ids=prefix + (4, 5),
            fixed_before=prefix,
            fixed_after=(4, 5),
            response_ids=response,
            response_byte_weights=tuple(range(1, len(response) + 1)),
        )
        for index, (prefix, response) in enumerate(
            [
                ((1, 3), (7, 8)),
                ((1, 2, 3, 6), (9, 10, 11, 12)),
                ((1,), (8,)),
            ]
        )
    ]
    if objective == "activation_l2":
        if method in ("DiffMean", "LinearProbe"):
            session._activation_model = IntervenableModel(
                IntervenableConfig(
                    representations=[
                        {
                            "layer": 0,
                            "component": component,
                            "intervention": benchmark.ax,
                        },
                        {
                            "layer": 0,
                            "component": component,
                            "intervention": CollectIntervention(embed_dim=16, keep_last_dim=True),
                        },
                    ]
                ),
                model,
            )
            session._activation_model.set_device("cpu")
        session._direction_offset = t.linspace(-0.2, 0.3, 16, dtype=dtype)
        sequences = [layout.clean_prompt_ids + layout.response_ids for layout in layouts]
        with gcg_worker._preserve_runtime(runtime), t.no_grad():
            embeds, attention = session._padded_embeds(sequences, gradient=False)
            with patch.object(
                model.get_output_embeddings(),
                "forward",
                wraps=model.get_output_embeddings().forward,
            ) as head:
                clean = session._capture_hidden(embeds, attention)
                assert head.call_count == 1
                assert head.call_args.args[0].shape[1] == 0
            with full_sequence_logits(model), full_decoder_depth(gcg_worker):
                reference_clean = session._capture_hidden(embeds, attention)
            assert t.equal(clean, reference_clean)
            for index, (layout, sequence) in enumerate(zip(layouts, sequences, strict=True)):
                session._clean_hidden[index] = (
                    clean[index, len(layout.fixed_before) : len(sequence)].detach().float().clone()
                )
    items = [gcg_worker._SideItem(layout, (13, 14, 15)) for layout in layouts]
    head = model.get_output_embeddings()
    full_length = max(len(session._sequence(item)) for item in items)
    expected_length = 0 if objective == "activation_l2" else full_length
    with gcg_worker._preserve_runtime(runtime), t.enable_grad():
        with patch.object(head, "forward", wraps=head.forward) as actual_head:
            actual = session._side_batch(items, gradient=True)
        assert actual_head.call_count == 1
        assert actual_head.call_args.args[0].shape[1] == expected_length
    with (
        gcg_worker._preserve_runtime(runtime),
        t.enable_grad(),
        full_sequence_logits(model),
        full_decoder_depth(gcg_worker),
        rowwise_vjp() as calls,
        fresh_response_tensors(gcg_worker),
        late_clean_transfers(gcg_worker),
        patch.object(head, "forward", wraps=head.forward) as reference_head,
    ):
        reference = session._side_batch(items, gradient=True)
        assert reference_head.call_count == 1
        assert reference_head.call_args.args[0].shape[1] == full_length
    with gcg_worker._preserve_runtime(runtime), t.no_grad():
        scores = session._side_batch(items, gradient=False)
        with (
            full_sequence_logits(model),
            full_decoder_depth(gcg_worker),
            fresh_response_tensors(gcg_worker),
            late_clean_transfers(gcg_worker),
        ):
            reference_scores = session._side_batch(items, gradient=False)
    assert calls == [3]
    for (actual_loss, actual_gradient), (
        reference_loss,
        reference_gradient,
    ), score, reference_score in zip(actual, reference, scores, reference_scores, strict=True):
        assert t.equal(actual_loss, reference_loss)
        assert t.equal(score[0], reference_score[0])
        assert score[1] is reference_score[1] is None
        assert actual_gradient is not None and reference_gradient is not None
        assert actual_gradient.shape == (3, 16)
        assert bool(actual_gradient.ne(0).any())
        assert t.equal(actual_gradient, reference_gradient)
