"""CPU-only contract checks; never fake native CUDA cache admission."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch as t
from torch._dynamo import mutation_guard

from nanochat.gpt import GPTConfig, Linear
from steering.conditional.encoder_decoder import _rlvr_replay_guard as guard
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sft_grpo_training import (
    configure_grpo_trainable_parameters,
)


@pytest.fixture(autouse=True)
def _install_test_owned_cpu_encoder_reference():
    # These tests inspect the native class contract without forwarding a CPU
    # encoder. The global CPU-reference monkeypatch would be a foreign method.
    yield


@pytest.fixture
def model(request):
    result = EncoderDecoderGPT(
        GPTConfig(sequence_len=32, vocab_size=128, n_layer=2, n_head=4, n_kv_head=2, n_embd=64),
        EncoderConfig(n_embd=64, depth=1, n_head=4, n_kv_head=2, head_dim=16),
        CrossAttentionConfig(frequency=1), 8, 4,
        feature_pred_config=FeaturePredictionConfig(
            head_layers=2, block=getattr(request, "param", FeaturePredictionBlock.MLP)),
    ).eval()
    configure_grpo_trainable_parameters(result, train_lm_head=False, train_feature_head_only=True)
    yield result
    assert not t.cuda.is_initialized()


@pytest.mark.parametrize("model", [FeaturePredictionBlock.MLP, FeaturePredictionBlock.TRANSFORMER], indirect=True)
def test_native_metadata_deepcopy_and_cpu_admission(model):
    copied = copy.deepcopy(model)
    assert guard._execution_reason() is None
    for name, module in copied.named_modules():
        state = guard._ModuleState.capture(name, module)
        assert state.matches(), name
    state, reason = guard.NativeFrozenState.capture(copied, policy_dependencies=())
    assert state is None and ("cuda" in reason or reason == "compute_dtype_not_bf16")


@pytest.mark.parametrize("kind", ["descriptor", "getattribute", "getattr"])
def test_class_changes_rejected_before_a_descriptor_can_execute(model, monkeypatch, kind):
    def explode(*_args):
        raise AssertionError("Changed lookup must never execute")

    if kind == "descriptor":
        monkeypatch.setattr(Linear, "weight", property(explode), raising=False)
    else:
        monkeypatch.setattr(Linear, "__" + kind + "__", explode)
    state, reason = guard.NativeFrozenState.capture(model, policy_dependencies=())
    assert state is None and reason == "native_class_changed:Linear"


def test_native_module_metadata_rejects_new_override_and_config_mutation(model):
    linear = model._decoder_blocks[0].mlp.c_fc
    linear_state = guard._ModuleState.capture("linear", linear)
    linear.forward = lambda value: value
    assert not linear_state.matches()
    model_state = guard._ModuleState.capture("model", model)
    model.feature_pred_config.layer = 0
    assert not model_state.matches()


def test_embedding_max_norm_is_not_an_admissible_frozen_forward(model):
    embedding = model._decoder_embedding
    embedding.max_norm = 1.0
    with pytest.raises(ValueError, match="embedding_max_norm"):
        guard._ModuleState.capture("embedding", embedding)


def test_head_value_updates_leave_layout_live_but_storage_and_scope_changes_do_not(model):
    parameter = model.feature_pred_head.weight
    initial = guard._tensor_layout(parameter)
    with t.no_grad():
        parameter.add_(0.125)
    assert guard._tensor_layout(parameter) == initial
    parameter.requires_grad_(False)
    assert guard._tensor_layout(parameter) != initial
    parameter.requires_grad_(True)
    with t.no_grad():
        parameter.set_(parameter.detach().clone())
    assert guard._tensor_layout(parameter) != initial


def test_saved_tensor_and_module_hooks_are_detected(model):
    assert guard._execution_reason() is None
    with t.autograd.graph.saved_tensors_hooks(lambda value: value, lambda value: value):
        assert guard._execution_reason() == "saved_tensor_hooks"
    with model.feature_pred_head.register_forward_pre_hook(lambda _module, _inputs: None):
        with pytest.raises(ValueError, match="module_hooks"):
            guard._ModuleState.capture("head", model.feature_pred_head)
    assert guard._ModuleState.capture("head", model.feature_pred_head).matches()


def test_frozen_tensor_version_detects_update_and_storage_replacement(model):
    parameter = model._decoder_blocks[0].mlp.c_fc.weight
    state = guard._ParameterState.capture(parameter)
    assert state.matches(parameter)
    with t.no_grad():
        parameter.add_(1)
    assert not state.matches(parameter)
    current = guard._ParameterState.capture(parameter)
    with t.no_grad():
        parameter.set_(parameter.detach().clone())
    assert not current.matches(parameter)


@pytest.mark.parametrize("registry", ["parameter", "buffer"])
def test_same_head_tensor_registered_in_trunk_is_rejected_before_device_admission(model, registry):
    trunk = model.encoder
    tensor = model.feature_pred_head.weight
    if registry == "parameter":
        trunk.register_parameter("head_alias", tensor)
    else:
        trunk.register_buffer("head_alias", tensor)
    state = guard.NativeFrozenState(model, policy_dependencies=())
    # The same-object alias deduplicates storage. Its conflicting role must
    # still fail before this CPU model reaches native CUDA device admission.
    with pytest.raises(ValueError, match="^head_parameter_registered_in_trunk$"):
        state._capture()


@pytest.mark.parametrize("module,name", [
    (guard.models, "ResidualFeaturePredictionHead"),
    (guard.heads, "Block"),
    (guard.packed, "_pack_tensor"),
    (guard.packed, "_mark_packed_inputs_dynamic"),
    (guard.packed, "_validate_packed_output_shape"),
    (guard.packed, "_build_packed_encoder_output"),
    (guard.flash, "_causal_mask"),
    (guard.flash, "_make_causal_sliding_mask"),
    (guard.flash, "_create_block_mask"),
])
def test_direct_packed_and_attention_dependency_replacement_is_rejected(model, monkeypatch, module, name):
    monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    state, reason = guard.NativeFrozenState.capture(model, policy_dependencies=())
    assert state is None and reason == "native_callable_changed:" + module.__name__ + "." + name


def _fresh_guard(monkeypatch):
    name = "_rlvr_guard_lazy_admission_canary"
    spec = importlib.util.spec_from_file_location(name, guard.__file__)
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    return fresh


def test_native_dynamo_generation_tagging_preserves_existing_forward_and_guard(monkeypatch):
    # Exercise real Dynamo lifecycle changes without a CUDA or Inductor compile.
    model = t.nn.Linear(4, 3).eval()
    value = t.ones(2, 4)
    expected = model(value).detach()
    parameters = tuple((id(parameter), parameter._version) for parameter in model.parameters())
    before_compile = _fresh_guard(monkeypatch)
    assert before_compile._execution_reason() is None

    compiled = t.compile(model, backend="eager", fullgraph=True)
    assert t.equal(compiled(value), expected)
    assert tuple((id(parameter), parameter._version) for parameter in model.parameters()) == parameters
    assert guard._execution_reason() is None
    assert before_compile._execution_reason() is None
    after_compile = _fresh_guard(monkeypatch)
    assert after_compile._execution_reason() is None

    # Ignoring lifecycle-only fields does not change native construction,
    # deserialization, or Dynamo's bookkeeping for newly created modules.
    copied = copy.deepcopy(model)
    assert t.equal(copied(value), expected)
    assert mutation_guard.GenerationTracker.check(copied)
    assert mutation_guard.GenerationTracker.check(t.nn.Linear(4, 3))
    assert not t.cuda.is_initialized()


def test_import_and_admission_do_not_read_installed_source(model, monkeypatch):
    def unreadable(_path, *_args, **_kwargs):
        raise AssertionError("Guard must not read installed framework source")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    monkeypatch.setattr(Path, "read_text", unreadable)
    fresh = _fresh_guard(monkeypatch)
    assert fresh._execution_reason() is None
    state, reason = fresh.NativeFrozenState.capture(model, policy_dependencies=())
    assert state is None and ("cuda" in reason or reason == "compute_dtype_not_bf16")
    assert not t.cuda.is_initialized()


@pytest.mark.parametrize("module_name,attribute", [
    ("torch._C._autograd", "_top_saved_tensors_default_hooks"),
    ("torch.overrides", "_get_current_function_mode_stack"),
    ("torch.utils._python_dispatch", "_get_current_dispatch_mode_stack"),
])
def test_missing_private_inspector_does_not_break_import_and_falls_back_on_admission(monkeypatch, module_name, attribute):
    module = sys.modules[module_name]
    monkeypatch.delattr(module, attribute)
    fresh = _fresh_guard(monkeypatch)
    state, reason = fresh.NativeFrozenState.capture(object(), policy_dependencies=())
    assert state is None
    assert reason == "unsupported_framework_capability:" + module_name + "." + attribute
    assert not t.cuda.is_initialized()


def test_missing_private_module_does_not_get_imported_by_guard(monkeypatch):
    name = "torch.overrides"
    monkeypatch.delitem(sys.modules, name)
    fresh = _fresh_guard(monkeypatch)
    assert name not in sys.modules
    state, reason = fresh.NativeFrozenState.capture(object(), policy_dependencies=())
    assert state is None and reason == "unsupported_framework_module:" + name


@pytest.mark.parametrize("mutation", ["global", "class_method", "class_method_code", "module_call"])
def test_pre_admission_mutation_does_not_change_import_time_baseline(monkeypatch, mutation):
    fresh = _fresh_guard(monkeypatch)

    def foreign(*_args, **_kwargs):
        raise AssertionError("Foreign computation must not execute during admission")

    if mutation == "global":
        monkeypatch.setattr(guard.packed, "_pack_tensor", foreign)
        expected = "native_callable_changed:" + guard.packed.__name__ + "._pack_tensor"
    elif mutation == "class_method":
        monkeypatch.setattr(Linear, "forward", foreign)
        expected = "native_class_changed:Linear"
    elif mutation == "module_call":
        monkeypatch.setattr(t.nn.Module, "_call_impl", foreign)
        expected = "native_class_changed:Module"
    else:
        monkeypatch.setattr(Linear.forward, "__code__", foreign.__code__)
        expected = "native_class_changed:Linear"
    state, reason = fresh.NativeFrozenState.capture(object(), policy_dependencies=())
    assert state is None and reason == expected


def test_verified_private_inspector_replacement_is_detected_without_calling_it(monkeypatch):
    fresh = _fresh_guard(monkeypatch)
    assert fresh._execution_reason() is None
    module = sys.modules["torch.overrides"]

    def foreign():
        raise AssertionError("Do not invoke a replaced private inspector")

    monkeypatch.setattr(module, "_get_current_function_mode_stack", foreign)
    state, reason = fresh.NativeFrozenState.capture(object(), policy_dependencies=())
    assert state is None and reason == "framework_binding_changed:torch.overrides._get_current_function_mode_stack"


def test_explicit_policy_binding_keeps_definition_time_object_and_code(monkeypatch):
    namespace = ModuleType("policy_canary")

    def original():
        return 1

    def foreign():
        return 2

    namespace.__dict__["compute"] = original
    binding = guard.NativeBinding.capture(namespace, "compute", original)
    assert binding.matches()
    monkeypatch.setattr(namespace, "compute", foreign)
    assert not binding.matches()
    monkeypatch.setattr(namespace, "compute", original)
    monkeypatch.setattr(original, "__code__", foreign.__code__)
    assert not binding.matches()
