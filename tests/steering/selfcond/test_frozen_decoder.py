"""Admission, numerical parity and lifetime contracts for frozen decoder shadows."""

from __future__ import annotations

import copy
import gc
import weakref
from contextlib import contextmanager

import pytest
import torch as t

from nanochat.gpt import Linear
from steering.conditional.encoder_decoder import frozen_decoder as candidate


class TinyEncoderDecoder(t.nn.Module):
    def __init__(self, *, bias: bool = True):
        super().__init__()
        self.transformer = t.nn.ModuleDict({
            "h": t.nn.ModuleList([Linear(8, 8, bias=bias), Linear(8, 8, bias=bias)])
        })
        self.lm_head = Linear(8, 5, bias=bias)
        self.encoder = Linear(8, 8)
        self.cross_attn_layers = t.nn.ModuleDict({"0": Linear(8, 8)})
        self.feature_pred_head = t.nn.Sequential(Linear(8, 8), t.nn.SiLU(), Linear(8, 4))
        self.requires_grad_(False)
        self.feature_pred_head.requires_grad_(True)

    def forward(self, x):
        layers = self.transformer["h"]
        assert isinstance(layers, t.nn.ModuleList)
        for layer in layers:
            x = layer(x).relu()
        return self.lm_head(x), self.feature_pred_head(x)


@pytest.fixture(scope="module")
def require_real_cuda():
    assert t.cuda.is_available(), "Qualification requires an accessible real CUDA device"
    assert candidate.COMPUTE_DTYPE == t.bfloat16
    deterministic = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    t.use_deterministic_algorithms(True)
    yield
    t.cuda.synchronize()
    t.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture
def model(require_real_cuda):
    t.manual_seed(173)
    return TinyEncoderDecoder().cuda()


@contextmanager
def staged(model, **overrides):
    with candidate.stage_frozen_decoder_linears(
        model,
        max_shadow_bytes=overrides.get("max_shadow_bytes", 1024**2),
        min_free_bytes=overrides.get("min_free_bytes", 0),
        train_feature_head_only=overrides.get("train_feature_head_only", True),
        world_size=overrides.get("world_size", 1),
    ) as report:
        yield report


def selected(model):
    return [*model.transformer["h"], model.lm_head]


def assert_restored(model):
    assert all("forward" not in module.__dict__ for module in model.modules())
    assert model not in candidate._OWNERS
    assert all(module not in candidate._OWNERS for module in selected(model))


def loss(outputs):
    return outputs[0].float().square().mean() + outputs[1].float().square().mean()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.slow
def test_bf16_native_outputs_input_and_head_gradients_are_exact(require_real_cuda, bias, noncontiguous):
    t.manual_seed(37)
    model = TinyEncoderDecoder(bias=bias).cuda()
    reference = copy.deepcopy(model)
    x = t.randn((8, 6) if noncontiguous else (6, 8), device="cuda", dtype=t.bfloat16)
    if noncontiguous:
        x = x.t()
        assert not x.is_contiguous()
    actual_x = x.detach().clone().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    expected = reference(reference_x)
    loss(expected).backward()
    with staged(model) as report:
        assert report.applied
        actual = model(actual_x)
        assert all(t.equal(a, b) for a, b in zip(actual, expected, strict=True))
        loss(actual).backward()
        assert actual_x.grad is not None and reference_x.grad is not None
        assert t.equal(actual_x.grad, reference_x.grad)
        actual_parameters = dict(model.named_parameters())
        for name, parameter in reference.named_parameters():
            observed = actual_parameters[name]
            if parameter.requires_grad:
                assert parameter.grad is not None and bool(t.count_nonzero(parameter.grad))
                assert observed.grad is not None
                assert t.equal(observed.grad, parameter.grad)
            else:
                assert observed.grad is None and parameter.grad is None
        assert report.fallback_calls == report.invalidated_modules == 0
    assert report.cleaned_up
    assert_restored(model)


@pytest.mark.slow
def test_parameter_state_dict_optimizer_identity_and_frozen_bytes(model):
    reference = copy.deepcopy(model)
    parameters = dict(model.named_parameters())
    parameter_ids = {name: id(value) for name, value in parameters.items()}
    state_identity = {name: (value.data_ptr(), value.dtype, tuple(value.stride()))
                      for name, value in model.state_dict().items()}
    frozen = {name: (value.detach().clone(), value._version)
              for name, value in parameters.items() if not value.requires_grad}
    optimizer = t.optim.SGD(model.feature_pred_head.parameters(), lr=0.1, momentum=0.9)
    reference_optimizer = t.optim.SGD(reference.feature_pred_head.parameters(), lr=0.1, momentum=0.9)
    optimizer_ids = [id(value) for group in optimizer.param_groups for value in group["params"]]
    native_forward = Linear.forward
    x = t.randn((6, 8), device="cuda", dtype=t.bfloat16)
    with staged(model) as report:
        assert report.applied
        assert Linear.forward is native_forward
        assert {name: id(value) for name, value in model.named_parameters()} == parameter_ids
        assert {name: (value.data_ptr(), value.dtype, tuple(value.stride()))
                for name, value in model.state_dict().items()} == state_identity
        assert all("forward" not in module.__dict__ for module in model.feature_pred_head.modules())
        assert "forward" not in model.encoder.__dict__
        assert "forward" not in model.cross_attn_layers["0"].__dict__
        for _ in range(2):
            for current, opt in ((reference, reference_optimizer), (model, optimizer)):
                opt.zero_grad(set_to_none=True)
                loss(current(x)).backward()
                assert all(value.grad is not None and bool(t.count_nonzero(value.grad))
                           for value in current.feature_pred_head.parameters())
                opt.step()
            assert all(t.equal(value, reference.state_dict()[name])
                       for name, value in model.state_dict().items())
            assert all(t.equal(optimizer.state[p]["momentum_buffer"],
                               reference_optimizer.state[q]["momentum_buffer"])
                       for p, q in zip(model.feature_pred_head.parameters(),
                                       reference.feature_pred_head.parameters(), strict=True))
        assert report.fallback_calls == report.invalidated_modules == 0
    assert [id(value) for group in optimizer.param_groups for value in group["params"]] == optimizer_ids
    assert {name: id(value) for name, value in model.named_parameters()} == parameter_ids
    assert {name: (value.data_ptr(), value.dtype, tuple(value.stride()))
            for name, value in model.state_dict().items()} == state_identity
    for name, (value, version) in frozen.items():
        assert t.equal(parameters[name], value)
        assert parameters[name]._version == version and parameters[name].grad is None
    assert Linear.forward is native_forward
    assert_restored(model)


@pytest.mark.slow
def test_exception_cleanup_releases_shadows_and_allows_reentry(model):
    references = []
    with pytest.raises(ValueError, match="injected body failure"):
        with staged(model) as report:
            assert report.applied
            for layer in selected(model):
                entry = layer.forward.__self__
                references.append(weakref.ref(entry.compute_weight))
                if entry.compute_bias is not None:
                    references.append(weakref.ref(entry.compute_bias))
            raise ValueError("injected body failure")
    assert report.cleaned_up
    gc.collect()
    assert all(reference() is None for reference in references)
    assert_restored(model)
    with staged(model) as second:
        assert second.applied
    assert_restored(model)


@pytest.mark.slow
@pytest.mark.parametrize("change", ["delete", "replace", "wrap"])
@pytest.mark.parametrize("body_raises", [False, True])
def test_cleanup_preserves_changed_forward_and_body_error(model, change, body_raises):
    body_error = RuntimeError("native body failure")
    foreign_calls = []
    saved_forward = model.lm_head.forward

    def foreign_forward(x):
        foreign_calls.append(x)
        if change == "wrap":
            return saved_forward(x)
        return Linear.forward(model.lm_head, x)

    retained_forwards = []
    references = []
    try:
        with staged(model) as report:
            assert report.applied
            saved_forward = model.lm_head.forward
            for layer in selected(model):
                retained_forwards.append(layer.forward)
                entry = layer.forward.__self__
                references.append(weakref.ref(entry.compute_weight))
                if entry.compute_bias is not None:
                    references.append(weakref.ref(entry.compute_bias))
            if change == "delete":
                del model.lm_head.forward
            else:
                model.lm_head.forward = foreign_forward
            if body_raises:
                raise body_error
    except RuntimeError as error:
        assert body_raises and error is body_error
    else:
        assert not body_raises

    assert report.cleaned_up
    assert model not in candidate._OWNERS
    assert all(layer not in candidate._OWNERS for layer in selected(model))
    assert all("forward" not in layer.__dict__ for layer in model.transformer["h"])
    assert all(forward.__self__.invalidated
               and forward.__self__.compute_weight is None and forward.__self__.compute_bias is None
               for forward in retained_forwards)
    gc.collect()
    assert all(reference() is None for reference in references)
    if change != "delete":
        assert model.lm_head.forward is foreign_forward
        x = t.randn((2, 8), device="cuda", dtype=t.bfloat16)
        assert t.equal(model.lm_head(x), Linear.forward(model.lm_head, x))
        assert len(foreign_calls) == 1 and foreign_calls[0] is x
        assert report.fallback_calls == (1 if change == "wrap" else 0)
        # The caller owns the replacement and removes it before native reentry.
        del model.lm_head.forward
    assert_restored(model)
    with staged(model) as retried:
        assert retried.applied
    assert retried.cleaned_up
    assert_restored(model)


@pytest.mark.slow
def test_float32_input_falls_back_without_poisoning_bf16_shadow(model):
    layer = model.transformer["h"][0]
    x = t.randn((3, 8), device="cuda", dtype=t.float32, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    expected = Linear.forward(layer, expected_x)
    expected.sum().backward()
    with staged(model) as report:
        actual = layer(x)
        actual.sum().backward()
        assert x.grad is not None and expected_x.grad is not None
        assert t.equal(actual, expected) and t.equal(x.grad, expected_x.grad)
        assert report.fallback_calls == 1 and report.invalidated_modules == 0
        bf16 = x.detach().to(t.bfloat16)
        assert t.equal(layer(bf16), Linear.forward(layer, bf16))
        assert report.fallback_calls == 1 and report.invalidated_modules == 0


@pytest.mark.parametrize("mutation", [
    "weight_add", "bias_add", "load_state_dict", "parameter_replacement",
    "requires_grad", "dtype_change", "same_pointer_transpose", "storage_offset",
])
@pytest.mark.slow
def test_selected_tensor_mutation_invalidates_and_uses_current_weights(model, mutation):
    layer = model.transformer["h"][0]
    backing = None
    if mutation == "storage_offset":
        backing = t.randn(128, device="cuda")
        layer.weight = t.nn.Parameter(backing[:64].view(8, 8), requires_grad=False)
    x = t.randn((3, 8), device="cuda", dtype=t.bfloat16)
    with staged(model) as report:
        entry = layer.forward.__self__
        with t.no_grad():
            if mutation == "weight_add":
                layer.weight.add_(0.25)
            elif mutation == "bias_add":
                layer.bias.add_(0.25)
            elif mutation == "load_state_dict":
                layer.load_state_dict({name: value + 0.25 for name, value in layer.state_dict().items()})
            elif mutation == "parameter_replacement":
                layer.weight = t.nn.Parameter(layer.weight.detach().clone() + 0.25, requires_grad=False)
            elif mutation == "requires_grad":
                layer.weight.requires_grad_(True)
            elif mutation == "dtype_change":
                layer.weight.data = layer.weight.data.double()
            elif mutation == "same_pointer_transpose":
                layer.weight.data = layer.weight.data.t()
            elif mutation == "storage_offset":
                assert backing is not None
                layer.weight.data = backing[64:].view(8, 8)
        expected = Linear.forward(layer, x)
        actual = layer(x)
        assert t.equal(actual, expected)
        assert report.invalidated_modules == report.fallback_calls == 1
        assert entry.compute_weight is None and entry.compute_bias is None
        if mutation == "requires_grad":
            actual.sum().backward()
            assert layer.weight.grad is not None and bool(t.count_nonzero(layer.weight.grad))
        assert t.equal(layer(x), Linear.forward(layer, x))
        assert report.invalidated_modules == 1 and report.fallback_calls == 2
    assert_restored(model)


@pytest.mark.parametrize("disjoint_views", [False, True])
@pytest.mark.slow
def test_trainable_storage_alias_with_independent_version_is_rejected(model, disjoint_views):
    frozen = model.transformer["h"][0]
    if disjoint_views:
        backing = t.randn(128, device="cuda")
        frozen.weight = t.nn.Parameter(backing[:64].view(8, 8), requires_grad=False)
        alias_data = backing[64:].view(8, 8)
    else:
        alias_data = frozen.weight.detach()
    alias = t.nn.Parameter(t.empty_like(alias_data))
    alias.data = alias_data
    model.feature_pred_head[0].weight = alias
    frozen_version = frozen.weight._version
    with t.no_grad():
        alias.add_(0.25)
    assert frozen.weight._version == frozen_version
    with staged(model) as report:
        assert not report.applied
        assert report.reason == "selected_storage_overlaps_registered_tensor"
        assert report.storage_overlap_registered_names == ("parameter:feature_pred_head.0.weight",)
        assert report.staged_shadow_bytes == 0
        assert all("forward" not in module.__dict__ for module in model.modules())
    assert_restored(model)


@pytest.mark.slow
def test_budget_refusal_precedes_allocator_queries(model, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Budget rejection must precede allocator queries")
    monkeypatch.setattr(t.cuda, "mem_get_info", unexpected)
    with staged(model, max_shadow_bytes=1) as report:
        assert not report.applied and report.reason == "shadow_budget_exceeded"
        assert report.planned_shadow_bytes > 0 and report.staged_shadow_bytes == 0
    assert_restored(model)


@pytest.mark.slow
def test_native_non_bf16_compute_is_a_noop(model, monkeypatch):
    monkeypatch.setattr(candidate, "COMPUTE_DTYPE", t.float32)
    with staged(model) as report:
        assert not report.applied and report.reason == "native_compute_dtype_not_bf16"
        assert report.staged_shadow_bytes == 0
    assert_restored(model)


@pytest.mark.parametrize("admit", [False, True])
@pytest.mark.slow
def test_allocator_reusable_reserve_estimate_and_headroom(model, monkeypatch, admit):
    reserved = 1024**2 if admit else 1024
    monkeypatch.setattr(t.cuda, "mem_get_info", lambda device: (0, 2 * 1024**2))
    monkeypatch.setattr(t.cuda, "memory_reserved", lambda device: reserved)
    monkeypatch.setattr(t.cuda, "memory_allocated", lambda device: 512)
    with staged(model, min_free_bytes=1024) as report:
        assert report.admission_free_bytes == 0
        assert report.admission_allocator_reserved_bytes == reserved
        assert report.admission_allocator_allocated_bytes == 512
        assert report.admission_reusable_reserved_bytes == reserved - 512
        assert report.admission_estimated_available_bytes == reserved - 512
        assert report.applied is admit
        assert report.reason == ("staged" if admit else "insufficient_estimated_memory_with_headroom")
    assert_restored(model)


@pytest.mark.slow
def test_shared_frozen_parameter_deduplicates_shadow_storage(model):
    first, second = model.transformer["h"]
    second.weight = first.weight
    unique = {id(parameter): parameter for layer in selected(model)
              for parameter in (layer.weight, layer.bias) if parameter is not None}
    with staged(model) as report:
        assert report.applied
        assert report.planned_shadow_bytes == report.staged_shadow_bytes == 2 * sum(p.numel() for p in unique.values())
        assert first.forward.__self__.compute_weight is second.forward.__self__.compute_weight
        x = t.randn((2, 8), device="cuda", dtype=t.bfloat16)
        assert t.equal(first(x), Linear.forward(first, x))
        assert t.equal(second(x), Linear.forward(second, x))
    assert_restored(model)


@pytest.mark.slow
def test_partial_staging_oom_releases_shadows_ownership_and_restores(model, monkeypatch):
    original_to = t.Tensor.to
    calls = 0
    references = []

    def injected_to(tensor, *args, **kwargs):
        nonlocal calls
        if kwargs.get("dtype") == t.bfloat16 and tensor.device.type == "cuda":
            calls += 1
            if calls == 2:
                raise t.OutOfMemoryError("injected partial staging OOM")
            result = original_to(tensor, *args, **kwargs)
            references.append(weakref.ref(result))
            return result
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(t.Tensor, "to", injected_to)
        with staged(model) as report:
            assert not report.applied and report.reason == "shadow_allocation_oom"
            assert calls == 2 and report.staged_shadow_bytes == 0
            assert all("forward" not in module.__dict__ for module in model.modules())
    gc.collect()
    assert all(reference() is None for reference in references)
    assert_restored(model)
    with staged(model) as retried:
        assert retried.applied
    assert_restored(model)


@pytest.mark.parametrize("shared_model", [False, True])
@pytest.mark.slow
def test_overlap_rejection_preserves_outer_context_and_later_reentry(model, shared_model):
    other = TinyEncoderDecoder().cuda() if shared_model else model
    if shared_model:
        other_layers = other.transformer["h"]
        assert isinstance(other_layers, t.nn.ModuleList)
        other_layers[0] = model.transformer["h"][0]
    with staged(model) as outer:
        installed = model.transformer["h"][0].forward
        with pytest.raises((RuntimeError, ValueError), match="already owns|existing instance forward"):
            with staged(other):
                pytest.fail("Overlapping context was admitted")
        assert model.transformer["h"][0].forward is installed
        x = t.randn((2, 8), device="cuda", dtype=t.bfloat16)
        assert t.equal(model.transformer["h"][0](x), Linear.forward(model.transformer["h"][0], x))
        assert outer.fallback_calls == outer.invalidated_modules == 0
    with staged(other) as retried:
        assert retried.applied
    assert_restored(model)
    assert_restored(other)


@pytest.mark.slow
def test_partial_forward_installation_releases_every_prepared_shadow(model, monkeypatch):
    original_init = candidate._StagedLinear.__init__
    original_setattr = Linear.__setattr__
    failure = RuntimeError("injected forward installation failure")
    entries, references = [], []
    installations = 0

    def observed_init(entry, *args, **kwargs):
        original_init(entry, *args, **kwargs)
        entries.append(entry)
        references.append(weakref.ref(entry.compute_weight))
        if entry.compute_bias is not None:
            references.append(weakref.ref(entry.compute_bias))

    def fail_second_installation(module, name, value):
        nonlocal installations
        if name == "forward":
            # All adapters must be prepared before the first installation.
            assert len(entries) == len(selected(model))
            installations += 1
            if installations == 2:
                raise failure
        original_setattr(module, name, value)

    with monkeypatch.context() as patcher:
        patcher.setattr(candidate._StagedLinear, "__init__", observed_init)
        patcher.setattr(Linear, "__setattr__", fail_second_installation)
        with pytest.raises(RuntimeError) as raised:
            with staged(model):
                pytest.fail("Partial installation must not enter the body")
        assert raised.value is failure and installations == 2
    assert all(entry.invalidated and entry.compute_weight is None and entry.compute_bias is None
               and entry.report.cleaned_up for entry in entries)
    gc.collect()
    assert references and all(reference() is None for reference in references)
    assert_restored(model)
    with staged(model) as retried:
        assert retried.applied
    assert retried.cleaned_up
    assert_restored(model)


@pytest.mark.slow
def test_inference_mode_entry_builds_normal_shadows_usable_for_gradients(model):
    x = t.randn((3, 8), device="cuda", dtype=t.bfloat16, requires_grad=True)
    with t.inference_mode():
        with staged(model) as report:
            for layer in selected(model):
                entry = layer.forward.__self__
                assert not t.is_inference(entry.compute_weight)
                assert entry.compute_bias is None or not t.is_inference(entry.compute_bias)
            with t.inference_mode(False), t.enable_grad():
                loss(model(x)).backward()
                assert x.grad is not None and bool(t.count_nonzero(x.grad))
                assert all(parameter.grad is not None for parameter in model.feature_pred_head.parameters())
            assert report.fallback_calls == report.invalidated_modules == 0
    assert_restored(model)


@pytest.mark.slow
def test_cross_stream_consume_survives_context_exit_and_allocator_pressure(model):
    layer = model.transformer["h"][0]
    x = t.randn((3, 8), device="cuda", dtype=t.bfloat16)
    expected = Linear.forward(layer, x)
    stream = t.cuda.Stream()
    t.cuda.synchronize()
    with staged(model) as report:
        with t.cuda.stream(stream):
            t.cuda._sleep(2_000_000)
            actual = layer(x)
        assert report.fallback_calls == report.invalidated_modules == 0
    pressure = [t.full((8, 8), float("nan"), device="cuda", dtype=t.bfloat16) for _ in range(64)]
    t.cuda.synchronize()
    assert t.equal(actual, expected)
    assert len(pressure) == 64
    assert_restored(model)


@pytest.mark.slow
@pytest.mark.parametrize("alias_kind", ["parameter", "buffer", "selected_parameter"])
@pytest.mark.parametrize("disjoint_views", [False, True])
def test_registered_frozen_independent_version_storage_alias_rejects_before_allocation(
    model, monkeypatch, alias_kind, disjoint_views
):
    layer = model.transformer["h"][0]
    if disjoint_views:
        backing = t.randn(128, device="cuda")
        layer.weight = t.nn.Parameter(backing[:64].view(8, 8), requires_grad=False)
        alias_data = backing[64:].view(8, 8)
    else:
        alias_data = layer.weight.detach()
    alias = t.empty_like(alias_data)
    alias.data = alias_data
    if alias_kind == "buffer":
        model.register_buffer("frozen_alias", alias)
        expected_names = {"buffer:frozen_alias"}
    else:
        alias = t.nn.Parameter(alias, requires_grad=False)
        if alias_kind == "selected_parameter":
            model.transformer["h"][1].weight = alias
            expected_names = {
                "parameter:transformer.h.0.weight", "parameter:transformer.h.1.weight"
            }
        else:
            model.register_parameter("frozen_alias", alias)
            expected_names = {"parameter:frozen_alias"}
    version = layer.weight._version
    old_values = layer.weight.detach().clone()
    with t.no_grad():
        alias.add_(0.5)
    assert layer.weight._version == version
    assert t.equal(layer.weight, old_values) is disjoint_views

    def unexpected(*args, **kwargs):
        pytest.fail("Registered alias rejection must precede allocator admission")

    monkeypatch.setattr(t.cuda, "mem_get_info", unexpected)
    with staged(model) as report:
        assert not report.applied
        assert report.reason == "selected_storage_overlaps_registered_tensor"
        assert set(report.storage_overlap_registered_names) == expected_names
        assert report.staged_shadow_bytes == 0
        assert all("forward" not in module.__dict__ for module in model.modules())
    assert report.cleaned_up
    assert_restored(model)


@pytest.mark.slow
@pytest.mark.parametrize("registration_kind", ["parameter", "buffer"])
def test_identical_tensor_registration_is_allowed_and_shares_invalidation(
    model, registration_kind
):
    layer = model.transformer["h"][0]
    if registration_kind == "parameter":
        model.register_parameter("same_tensor_alias", layer.weight)
    else:
        model.register_buffer("same_tensor_alias", layer.weight)
    assert model.same_tensor_alias is layer.weight
    x = t.randn((3, 8), device="cuda", dtype=t.bfloat16)
    with staged(model) as report:
        assert report.applied and report.storage_overlap_registered_names == ()
        assert t.equal(layer(x), Linear.forward(layer, x))
        assert report.fallback_calls == report.invalidated_modules == 0
        with t.no_grad():
            model.same_tensor_alias.add_(0.25)
        assert t.equal(layer(x), Linear.forward(layer, x))
        assert report.fallback_calls == report.invalidated_modules == 1
    assert report.cleaned_up
    assert_restored(model)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"max_shadow_bytes": 0}, "disabled"),
        ({"train_feature_head_only": False}, "full_policy"),
        ({"world_size": 2}, "unsupported_world_size"),
        ({}, "decoder_not_on_one_cuda_device"),
    ],
)
def test_explicit_inactive_policies_preserve_cpu_model_and_native_results(
    monkeypatch, options, reason
):
    cpu_model = TinyEncoderDecoder()
    if options.get("train_feature_head_only") is False:
        cpu_model.requires_grad_(True)
    monkeypatch.setattr(candidate, "COMPUTE_DTYPE", t.bfloat16)

    def unexpected(*args, **kwargs):
        pytest.fail("Inactive CPU/policy admission must not query CUDA memory")

    monkeypatch.setattr(t.cuda, "mem_get_info", unexpected)
    before = {
        name: (id(parameter), parameter.data_ptr(), parameter.requires_grad, parameter._version)
        for name, parameter in cpu_model.named_parameters()
    }
    x = t.randn((3, 8))
    expected = cpu_model(x)
    with staged(cpu_model, **options) as report:
        assert not report.applied and report.reason == reason
        assert report.staged_shadow_bytes == 0
        assert report.fallback_calls == report.invalidated_modules == 0
        actual = cpu_model(x)
        assert all(t.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert all("forward" not in module.__dict__ for module in cpu_model.modules())
    assert {
        name: (id(parameter), parameter.data_ptr(), parameter.requires_grad, parameter._version)
        for name, parameter in cpu_model.named_parameters()
    } == before
    assert report.cleaned_up
    assert_restored(cpu_model)


def test_production_settings_default_to_disabled_and_resolve_explicit_budget():
    assert candidate.frozen_decoder_staging_settings({}) == {
        "max_shadow_bytes": 0,
        "min_free_bytes": 4 * 1024**3,
    }
    assert candidate.frozen_decoder_staging_settings({
        "frozen_decoder_max_shadow_bytes": 1234,
        "frozen_decoder_min_free_bytes": 0,
    }) == {"max_shadow_bytes": 1234, "min_free_bytes": 0}


@pytest.mark.parametrize("name", ["max_shadow_bytes", "min_free_bytes"])
@pytest.mark.parametrize("invalid", [-1, True, 1.0, "1024", None])
def test_production_settings_reject_noninteger_or_negative_limits(name, invalid):
    with pytest.raises(ValueError, match="nonnegative integer"):
        candidate.frozen_decoder_staging_settings({f"frozen_decoder_{name}": invalid})


@pytest.mark.parametrize(
    "invalid_options",
    [
        {"max_shadow_bytes": -1},
        {"min_free_bytes": True},
        {"train_feature_head_only": "true"},
        {"world_size": True},
        {"world_size": 0},
    ],
)
def test_context_rejects_malformed_scope_and_limits_before_model_admission(invalid_options):
    cpu_model = TinyEncoderDecoder()
    with pytest.raises(ValueError):
        with staged(cpu_model, **invalid_options):
            pytest.fail("Malformed staging controls were admitted")
    assert_restored(cpu_model)
