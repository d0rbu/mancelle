"""Exact input contracts and delayed-copy host allocation lifetime checks."""

from __future__ import annotations

import copy
import gc
import sys
import weakref
from contextlib import nullcontext
from importlib.util import find_spec
from types import MethodType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch as t
from torch.nn.modules import module as module_hooks
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM

from tests.axbench_adv.unit.test_gcg_vendor_worker import GCGSession as IsolatedSession
from tests.axbench_adv.unit.test_gcg_vendor_worker import (
    _prepare,
    _runtime,
    _write_activation_artifacts,
    _write_manifest,
)

if find_spec("pyvene") is not None and find_spec("axbench") is not None:
    from pyvene import IntervenableConfig, IntervenableModel  # ty: ignore[unresolved-import]

    from scripts.eval.axbench_adv.vendored import gcg_worker as worker
else:
    IntervenableConfig = None
    IntervenableModel = None
    worker = sys.modules[IsolatedSession.__module__]

if find_spec("peft") is not None:
    from peft import LoraConfig, get_peft_model  # ty: ignore[unresolved-import]
else:
    LoraConfig = None
    get_peft_model = None


# Frozen from PR #1721 (3909cac4e), independent of the production padding helper.
def _reference_padded_embeds(
    self, sequences: list[tuple[int, ...]], *, gradient: bool
) -> tuple[t.Tensor, t.Tensor]:
    embedding = self.runtime.benchmark_model.model.get_input_embeddings()
    vocab = int(embedding.weight.shape[0])
    pad_id = getattr(self.runtime.tokenizer, "pad_token_id", None)
    if type(pad_id) is not int or not 0 <= pad_id < vocab:
        pad_id = getattr(self.runtime.tokenizer, "eos_token_id", None)
    if type(pad_id) is not int or not 0 <= pad_id < vocab:
        pad_id = 0
    maximum = max(map(len, sequences))
    ids = t.full(
        (len(sequences), maximum), pad_id, dtype=t.long, device=t.device(self.runtime.device)
    )
    attention = t.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = t.tensor(sequence, dtype=t.long, device=ids.device)
        attention[row, : len(sequence)] = 1
    embeds = embedding(ids).detach()
    embeds.requires_grad_(gradient)
    return (embeds, attention)


class IntegerSubclass(int):
    pass


class RecordingEmbedding(t.nn.Embedding):
    def forward(self, input: t.Tensor) -> t.Tensor:
        self.last_ids = input.detach().clone()
        return super().forward(input)


def sessions(device, dtype, pad, eos):
    weight = t.arange(257 * 16, dtype=t.float32).reshape(257, 16).to(dtype)
    embedding = RecordingEmbedding.from_pretrained(weight.to(device), freeze=False)
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(
            model=SimpleNamespace(get_input_embeddings=lambda: embedding)
        ),
        tokenizer=SimpleNamespace(pad_token_id=pad, eos_token_id=eos),
        device=device,
    )
    reference = SimpleNamespace(runtime=runtime)
    candidate = worker.GCGSession(runtime)
    reference._padded_embeds = MethodType(_reference_padded_embeds, reference)
    return reference, candidate, embedding


CASES = [
    ([(1, 2, 3), (4,)], 7, 8, 7),
    ([(1, 2), (3,)], None, 9, 9),
    ([(1, 2), ()], -1, None, 0),
    ([(), (1, 2, 3)], 257, 6, 6),
    ([(), ()], 4, 8, 4),
    ([(1,)], 2, None, 2),
    ([(1, 2), (3, 4)], None, 999, 0),
    ([(1, 2), (3,)], True, 5, 5),
    ([(1, 2), (3,)], IntegerSubclass(2), False, 0),
    ([(1, 2), (3,)], "2", -1, 0),
]


@pytest.mark.parametrize("sequences,pad,eos,expected_pad", CASES)
@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float32])
@pytest.mark.parametrize(
    "mode",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.slow),
        pytest.param("cuda-stream", marks=pytest.mark.slow),
    ],
)
def test_inputs_and_leaf_contract(sequences, pad, eos, expected_pad, gradient, dtype, mode):
    device = "cpu" if mode == "cpu" else "cuda:0"
    if mode != "cpu" and not t.cuda.is_available():
        pytest.skip("CUDA required")
    reference, candidate, embedding = sessions(device, dtype, pad, eos)
    initial_sequences = copy.deepcopy(sequences)
    parameter_version = embedding.weight._version
    initial_cpu_rng = t.random.get_rng_state().clone()
    initial_cuda_rng = t.cuda.get_rng_state().clone() if mode != "cpu" else None
    stream = t.cuda.Stream() if mode == "cuda-stream" else None
    if stream is not None:
        stream.wait_stream(t.cuda.current_stream())
    context = t.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        old_embeds, old_attention = reference._padded_embeds(sequences, gradient=gradient)
        old_ids = embedding.last_ids
        with t.no_grad():
            new_embeds, new_attention = candidate._padded_embeds(
                sequences, gradient=gradient, stage_inputs=mode != "cpu"
            )
        new_ids = embedding.last_ids
    if stream is not None:
        stream.synchronize()
    maximum = max(map(len, sequences))
    expected_ids = t.tensor(
        [list(row) + [expected_pad] * (maximum - len(row)) for row in sequences], dtype=t.long
    )
    expected_attention = t.tensor(
        [[1] * len(row) + [0] * (maximum - len(row)) for row in sequences], dtype=t.long
    )
    assert t.equal(old_ids.cpu(), expected_ids)
    assert t.equal(new_ids.cpu(), expected_ids)
    assert t.equal(new_attention.cpu(), expected_attention)
    for old, new in ((old_embeds, new_embeds), (old_attention, new_attention)):
        assert old.shape == new.shape and old.dtype == new.dtype and old.device == new.device
        assert old.stride() == new.stride() and t.equal(old, new)
        assert old.is_contiguous() and new.is_contiguous()
    assert old_embeds.requires_grad == new_embeds.requires_grad == gradient
    assert old_embeds.is_leaf and new_embeds.is_leaf
    assert old_embeds.grad_fn is None and new_embeds.grad_fn is None
    if gradient:
        old_embeds.sum().backward()
        new_embeds.sum().backward()
        assert t.equal(old_embeds.grad, new_embeds.grad)
    assert embedding.weight.grad is None and embedding.weight._version == parameter_version
    assert sequences == initial_sequences
    assert t.equal(initial_cpu_rng, t.random.get_rng_state())
    if initial_cuda_rng is not None:
        assert t.equal(initial_cuda_rng, t.cuda.get_rng_state())


@pytest.mark.parametrize("mode", ["cpu", pytest.param("cuda:0", marks=pytest.mark.slow)])
def test_empty_collection_still_raises(mode):
    if mode != "cpu" and not t.cuda.is_available():
        pytest.skip("CUDA required")
    reference, candidate, _ = sessions(mode, t.float32, None, None)
    with pytest.raises(ValueError, match="max.*empty"):
        reference._padded_embeds([], gradient=False)
    with pytest.raises(ValueError, match="max.*empty"):
        candidate._padded_embeds([], gradient=False, stage_inputs=mode != "cpu")


@pytest.mark.slow
@pytest.mark.parametrize("nondefault", [False, True])
@pytest.mark.parametrize("gradient", [False, True])
def test_released_pinned_allocations_survive_delayed_stream(nondefault, gradient):
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    reference, candidate, embedding = sessions("cuda:0", t.bfloat16, 7, None)
    sequences = [tuple(range(129)), tuple(range(81))]
    expected_embeds, expected_attention = reference._padded_embeds(sequences, gradient=gradient)
    expected_ids = embedding.last_ids
    candidate._padded_embeds(sequences, gradient=gradient, stage_inputs=True)
    t.cuda.synchronize()
    # Populate the host allocator before the delay so churn does not need new
    # cudaHostAlloc calls, which could themselves synchronize the device.
    reserves = [t.empty((2, 129), dtype=t.long, device="cpu", pin_memory=True) for _ in range(96)]
    del reserves
    gc.collect()
    original_tensor = t.tensor
    allocations = []

    def observe_tensor(*args, **kwargs):
        result = original_tensor(*args, **kwargs)
        if kwargs.get("pin_memory"):
            assert result.device.type == "cpu" and result.is_pinned()
            allocations.append((result.data_ptr(), weakref.ref(result)))
        return result

    stream = t.cuda.Stream() if nondefault else t.cuda.current_stream()
    stream.wait_stream(t.cuda.current_stream())
    done = t.cuda.Event()
    with t.cuda.stream(stream):
        t.cuda._sleep(300_000_000)
        with patch.object(t, "tensor", observe_tensor):
            actual_embeds, actual_attention = candidate._padded_embeds(
                sequences, gradient=gradient, stage_inputs=True
            )
        actual_ids = embedding.last_ids
        done.record(stream)
    assert len(allocations) == 2
    assert all(reference() is None for _, reference in allocations)
    assert not done.query(), "The adversarial check requires a pending DMA copy."
    protected = {pointer for pointer, _ in allocations}
    churn = []
    for index in range(64):
        buffer = t.empty((2, 129), dtype=t.long, device="cpu", pin_memory=True)
        assert buffer.data_ptr() not in protected
        buffer.fill_(1000 + index)
        churn.append(buffer)
    assert not done.query(), "The delayed stream completed before allocator churn finished."
    stream.synchronize()
    assert t.equal(actual_ids, expected_ids)
    assert t.equal(actual_attention, expected_attention)
    assert t.equal(actual_embeds, expected_embeds)
    assert actual_embeds.is_leaf and actual_embeds.requires_grad == gradient
    assert len(churn) == 64


def _plain_runtime(device: str = "cuda:0", dtype: t.dtype = t.float32) -> Any:
    config = SmolLM3Config(
        vocab_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
        sliding_window=None,
    )
    config._attn_implementation = "sdpa"
    model = cast(Any, SmolLM3ForCausalLM)(config).to(dtype=dtype).eval()
    return SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model),
        method="PromptSteering",
        device=device,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )


@pytest.fixture
def native_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    # These structural checks need no CUDA allocation. The slow tests below
    # exercise the actual installed API, CUDA copies, masks and model forwards.
    monkeypatch.setattr(worker, "_EXPLICIT_CAUSAL_MASK", True)
    return _plain_runtime()


def _wrap(runtime: Any, method: str) -> Any:
    model = runtime.benchmark_model.model
    if method in {"DiffMean", "LinearProbe"}:
        if IntervenableModel is None:
            pytest.skip("PyVene wrapper checks run in the vendored environment")
        assert IntervenableConfig is not None
        wrapper = IntervenableModel(IntervenableConfig(representations=[]), model)
    else:
        if get_peft_model is None:
            pytest.skip("PEFT wrapper checks run in the vendored environment")
        assert LoraConfig is not None
        wrapper = get_peft_model(
            model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type="CAUSAL_LM")
        )
    runtime.method = method
    runtime.benchmark_model.ax_model = wrapper
    return wrapper


@pytest.mark.parametrize("method", ["PromptSteering", "DiffMean", "LinearProbe", "LoRA"])
def test_native_wrappers_are_eligible(native_runtime: Any, method: str) -> None:
    if method != "PromptSteering":
        _wrap(native_runtime, method)
    with t.no_grad():
        assert worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])


@pytest.mark.parametrize(
    "condition",
    [
        "api_absent",
        "grad_enabled",
        "autocast",
        "cpu",
        "equal_lengths",
        "single_row",
        "empty_row",
        "empty_batch",
        "unknown_model",
        "unknown_backbone",
        "eager",
        "noncausal",
        "sliding",
        "window",
        "distinct_config",
        "embedding_subclass",
        "embedding_weight_subclass",
        "embedding_getter",
        "factory_replaced",
        "model_factory_replaced",
        "registry_replaced",
        "registry_missing",
        "unknown_method",
        "tracing",
        "compiling",
    ],
)
def test_unsupported_inputs_fall_back(
    native_runtime: Any,
    condition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = native_runtime.benchmark_model.model
    sequences = [(1, 2), (3,)]
    if condition == "api_absent":
        monkeypatch.setattr(worker, "_EXPLICIT_CAUSAL_MASK", False)
    elif condition == "autocast":
        monkeypatch.setattr(t, "is_autocast_enabled", lambda: True)
    elif condition == "cpu":
        native_runtime.device = "cpu"
    elif condition == "equal_lengths":
        sequences = [(1, 2), (3, 4)]
    elif condition == "single_row":
        sequences = [(1, 2)]
    elif condition == "empty_row":
        sequences = [(1, 2), ()]
    elif condition == "empty_batch":
        sequences = []
    elif condition == "unknown_model":
        native_runtime.benchmark_model.model = t.nn.Linear(1, 1)
    elif condition == "unknown_backbone":
        model.model = t.nn.Linear(1, 1)
    elif condition == "eager":
        model.config._attn_implementation = "eager"
    elif condition == "noncausal":
        model.config.is_causal = False
    elif condition == "sliding":
        model.model.has_sliding_layers = True
    elif condition == "window":
        model.config.sliding_window = 8
    elif condition == "distinct_config":
        model.model.config = copy.deepcopy(model.config)
    elif condition == "embedding_subclass":
        model.model.embed_tokens = RecordingEmbedding(64, 32)
    elif condition == "embedding_weight_subclass":

        class ParameterSubclass(t.nn.Parameter):
            pass

        model.model.embed_tokens.weight = ParameterSubclass(model.model.embed_tokens.weight)
    elif condition == "embedding_getter":
        model.get_input_embeddings = lambda: model.model.embed_tokens
    elif condition == "factory_replaced":
        monkeypatch.setattr(worker.masking_utils, "create_causal_mask", lambda **_: None)
    elif condition == "model_factory_replaced":
        monkeypatch.setitem(worker._INPUT_MODEL_GLOBALS, "create_causal_mask", lambda **_: None)
    elif condition in {"registry_replaced", "registry_missing"}:
        registry = {"sdpa": lambda **_: None} if condition == "registry_replaced" else {}
        monkeypatch.setattr(worker.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", registry)
    elif condition == "unknown_method":
        native_runtime.method = "custom"
    elif condition == "tracing":
        monkeypatch.setattr(t.jit, "is_tracing", lambda: True)
    elif condition == "compiling":
        monkeypatch.setattr(t.compiler, "is_compiling", lambda: True)
    with t.set_grad_enabled(condition == "grad_enabled"):
        assert not worker._can_stage_scoring_inputs(native_runtime, sequences)


@pytest.mark.parametrize("location", ["model", "backbone", "embedding"])
@pytest.mark.parametrize("change", ["forward_instance", "forward_class", "compiled"])
def test_modified_modules_fall_back(
    native_runtime: Any,
    location: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = native_runtime.benchmark_model.model
    module = {"model": model, "backbone": model.model, "embedding": model.get_input_embeddings()}[
        location
    ]
    if change == "forward_instance":
        monkeypatch.setattr(module, "forward", lambda *_: None)
    elif change == "forward_class":
        monkeypatch.setattr(type(module), "forward", lambda *_: None)
    else:
        monkeypatch.setattr(module, "_compiled_call_impl", lambda *_: None)
    with t.no_grad():
        assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])


@pytest.mark.parametrize("location", ["model", "backbone", "embedding"])
@pytest.mark.parametrize(
    "registration",
    [
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
    ],
)
def test_local_observers_keep_original_inputs(
    native_runtime: Any, location: str, registration: str
) -> None:
    model = native_runtime.benchmark_model.model
    module = {"model": model, "backbone": model.model, "embedding": model.get_input_embeddings()}[
        location
    ]
    handle = getattr(module, registration)(lambda *_: None)
    try:
        with t.no_grad():
            assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])
    finally:
        handle.remove()
    with t.no_grad():
        assert worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])


@pytest.mark.parametrize(
    "registration",
    [
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
    ],
)
def test_global_observers_keep_original_inputs(native_runtime: Any, registration: str) -> None:
    handle = getattr(module_hooks, registration)(lambda *_: None)
    try:
        with t.no_grad():
            assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])
    finally:
        handle.remove()


@pytest.mark.parametrize("mode", [TorchFunctionMode, TorchDispatchMode])
def test_torch_modes_keep_original_inputs(native_runtime: Any, mode: type) -> None:
    with t.no_grad(), mode():
        assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])


@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe", "LoRA"])
@pytest.mark.parametrize(
    "change",
    ["forward", "pre_hook", "post_hook", "backward_hook", "compiled", "wrong_model", "subclass"],
)
def test_modified_wrappers_keep_original_inputs(
    native_runtime: Any,
    method: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrap(native_runtime, method)
    if change == "forward":
        monkeypatch.setattr(wrapper, "forward", lambda *_: None)
    elif change == "pre_hook":
        wrapper.register_forward_pre_hook(lambda *_: None)
    elif change == "post_hook":
        wrapper.register_forward_hook(lambda *_: None)
    elif change == "backward_hook":
        wrapper.register_full_backward_hook(lambda *_: None)
    elif change == "compiled":
        monkeypatch.setattr(wrapper, "_compiled_call_impl", lambda *_: None)
    elif change == "wrong_model":
        if method == "LoRA":
            wrapper.base_model.model = t.nn.Linear(1, 1)
        else:
            wrapper.model = t.nn.Linear(1, 1)
    else:
        wrapper.__class__ = type("CustomWrapper", (type(wrapper),), {})
    with t.no_grad():
        assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])


@pytest.mark.parametrize(
    "change",
    [
        "tuner_type",
        "tuner_forward",
        "tuner_hook",
        "wrapper_context",
        "tuner_context",
        "prompt_learning",
    ],
)
def test_lora_input_rewriters_fall_back(
    native_runtime: Any,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrap(native_runtime, "LoRA")
    tuner = wrapper.base_model
    if change == "tuner_type":
        tuner.__class__ = type("CustomTuner", (type(tuner),), {})
    elif change == "tuner_forward":
        monkeypatch.setattr(tuner, "forward", lambda *_: None)
    elif change == "tuner_hook":
        tuner.register_forward_pre_hook(lambda *_: None)
    elif change == "wrapper_context":
        monkeypatch.setattr(wrapper, "_enable_peft_forward_hooks", nullcontext)
    elif change == "tuner_context":
        monkeypatch.setattr(tuner, "_enable_peft_forward_hooks", nullcontext)
    else:
        wrapper.peft_config[wrapper.active_adapter] = SimpleNamespace(is_prompt_learning=True)
    with t.no_grad():
        assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])


class _ReferenceInputSession(worker.GCGSession):
    _padded_embeds = _reference_padded_embeds


def _items(lengths: tuple[int, ...]) -> list[Any]:
    return [
        worker._SideItem(
            worker._CompletionLayout(
                sample_id=index,
                content="fixture",
                clean_prompt_ids=(1,),
                fixed_before=(1,) * (length - 4),
                fixed_after=(4,),
                response_ids=(5, 6),
                response_byte_weights=(1, 2),
            ),
            (3,),
        )
        for index, length in enumerate(lengths)
    ]


@pytest.mark.slow
@pytest.mark.parametrize("lengths", [(5, 6), (21, 9), (37, 36), (11, 5, 8, 11)])
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float32])
@pytest.mark.parametrize("nondefault", [False, True])
def test_actual_scoring_preserves_native_masks_and_losses(
    lengths: tuple[int, ...],
    dtype: t.dtype,
    nondefault: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    runtime = _plain_runtime(dtype=dtype)
    runtime.benchmark_model.model.to(runtime.device)
    reference, candidate = _ReferenceInputSession(runtime), worker.GCGSession(runtime)
    reference.objective = candidate.objective = "response_margin"
    seen = []
    forward = worker._forward

    def observe(runtime: Any, *, embeds: t.Tensor, attention_mask: t.Tensor) -> Any:
        seen.append((embeds, attention_mask))
        return forward(runtime, embeds=embeds, attention_mask=attention_mask)

    monkeypatch.setattr(worker, "_forward", observe)
    stream = t.cuda.Stream() if nondefault else None
    if stream is not None:
        stream.wait_stream(t.cuda.current_stream())
    with t.no_grad(), t.cuda.stream(stream) if stream is not None else nullcontext():
        expected = reference._side_batch(_items(lengths), gradient=False)
        actual = candidate._side_batch(_items(lengths), gradient=False)
        old_embeds, old_attention = seen[0]
        new_embeds, new_attention = seen[1]
        assert t.equal(old_embeds, new_embeds)
        assert old_attention.ndim == 2
        if worker._EXPLICIT_CAUSAL_MASK:
            expected_mask = worker.masking_utils.create_causal_mask(
                config=runtime.benchmark_model.model.config,
                inputs_embeds=old_embeds,
                attention_mask=old_attention,
                past_key_values=None,
                position_ids=t.arange(max(lengths), device=old_embeds.device).unsqueeze(0),
            )
            assert new_attention.ndim == 4
            assert expected_mask.shape == new_attention.shape
            assert expected_mask.dtype == new_attention.dtype == t.bool
            assert expected_mask.stride() == new_attention.stride()
            assert t.equal(expected_mask, new_attention)
        else:
            assert t.equal(old_attention, new_attention)
        for (old_loss, old_gradient), (new_loss, new_gradient) in zip(
            expected, actual, strict=True
        ):
            assert old_gradient is new_gradient is None
            assert t.isfinite(old_loss) and t.equal(old_loss, new_loss)
    if stream is not None:
        stream.synchronize()


@pytest.mark.parametrize("scope", ["prepare_and_gradient", "l2_score_and_gradient"])
def test_unoptimized_operations_do_not_enter_staging(
    tmp_path: Any,
    scope: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_worker = sys.modules[IsolatedSession.__module__]

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("This operation must retain its original input path")

    monkeypatch.setattr(isolated_worker, "_can_stage_scoring_inputs", fail)
    _write_manifest(tmp_path)
    session = IsolatedSession(_runtime(tmp_path))
    if scope == "prepare_and_gradient":
        _prepare(session)
    else:
        _write_activation_artifacts(tmp_path)
        session.prepare(
            {
                "alpha": 0.01,
                "device_batch_size": 2,
                "objective": "activation_l2",
                "rows": [
                    {
                        "pair_id": 0,
                        "positive": {"sample_id": 11, "prompt": "P", "response": "A"},
                        "negative": {"sample_id": 12, "prompt": "N", "response": "B"},
                    }
                ],
                "separator": "\n\n",
                "suffix_token_count": 1,
            }
        )
        session.score(
            {
                "candidate_eval_device_batch_size": 2,
                "candidate_token_ids": [[ord("?")]],
                "member_indices": [0],
            }
        )
    session.gradient({"member_indices": [0], "token_ids": [ord("?")]})
    session.close()


@pytest.mark.parametrize("location", ["instance", "class"])
def test_replaced_padding_method_keeps_its_original_call_contract(
    tmp_path: Any,
    location: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(tmp_path)
    session = IsolatedSession(_runtime(tmp_path))
    _prepare(session)
    seen = []
    original = IsolatedSession._padded_embeds

    def replacement(self: Any, sequences: Any, *, gradient: bool) -> Any:
        seen.append(gradient)
        return original(self, sequences, gradient=gradient)

    if location == "instance":
        monkeypatch.setattr(session, "_padded_embeds", MethodType(replacement, session))
    else:
        monkeypatch.setattr(IsolatedSession, "_padded_embeds", replacement)
    isolated_worker = sys.modules[IsolatedSession.__module__]
    monkeypatch.setattr(
        isolated_worker,
        "_can_stage_scoring_inputs",
        lambda *_: pytest.fail("padding override must fall back before eligibility"),
    )
    session.score(
        {
            "candidate_eval_device_batch_size": 2,
            "candidate_token_ids": [[ord("?")]],
            "member_indices": [0],
        }
    )
    assert seen == [False]
    session.close()


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float32])
@pytest.mark.parametrize("nondefault", [False, True])
def test_inference_scoring_then_gradients_preserves_inputs_and_state(
    dtype: t.dtype,
    nondefault: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    runtime = _plain_runtime(dtype=dtype)
    model = runtime.benchmark_model.model.to(runtime.device)
    reference, candidate = _ReferenceInputSession(runtime), worker.GCGSession(runtime)
    reference.objective = candidate.objective = "response_margin"
    items = _items((21, 9))
    seen = []
    forward = worker._forward
    versions = [parameter._version for parameter in model.parameters()]
    cpu_rng = t.random.get_rng_state().clone()
    cuda_rng = t.cuda.get_rng_state().clone()

    def observe(runtime: Any, *, embeds: t.Tensor, attention_mask: t.Tensor) -> Any:
        seen.append((t.is_inference_mode_enabled(), attention_mask.ndim))
        return forward(runtime, embeds=embeds, attention_mask=attention_mask)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A later gradient call must keep its original input path")

    monkeypatch.setattr(worker, "_forward", observe)
    stream = t.cuda.Stream() if nondefault else None
    if stream is not None:
        stream.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(stream) if stream is not None else nullcontext():
        with t.inference_mode():
            old_scores = reference._side_batch(items, gradient=False)
            new_scores = candidate._side_batch(items, gradient=False)
        for session in (reference, candidate):
            assert session._response_tensors
            assert all(
                not value.is_inference()
                for values in session._response_tensors.values()
                for value in values
            )
        monkeypatch.setattr(worker, "_can_stage_scoring_inputs", fail)
        with t.enable_grad():
            old_backward = reference._side_batch(items, gradient=True)
            new_backward = candidate._side_batch(items, gradient=True)
    if stream is not None:
        stream.synchronize()
    assert seen == [
        (True, 2),
        (True, 4 if worker._EXPLICIT_CAUSAL_MASK else 2),
        (False, 2),
        (False, 2),
    ]
    for (old_loss, old_gradient), (new_loss, new_gradient) in zip(
        old_scores, new_scores, strict=True
    ):
        assert old_gradient is new_gradient is None
        assert old_loss.is_inference() and new_loss.is_inference()
        assert t.isfinite(old_loss) and t.equal(old_loss, new_loss)
    for (old_loss, old_gradient), (new_loss, new_gradient) in zip(
        old_backward, new_backward, strict=True
    ):
        assert old_gradient is not None and new_gradient is not None
        assert not old_loss.is_inference() and not new_loss.is_inference()
        assert t.isfinite(old_loss) and t.equal(old_loss, new_loss)
        assert t.isfinite(old_gradient).all() and t.equal(old_gradient, new_gradient)
    assert versions == [parameter._version for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training
    assert t.equal(cpu_rng, t.random.get_rng_state())
    assert t.equal(cuda_rng, t.cuda.get_rng_state())


def test_class_embedding_getter_observer_falls_back_before_it_is_called(
    native_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = native_runtime.benchmark_model.model
    original = type(model).get_input_embeddings
    calls = []

    def observe(self: Any) -> Any:
        calls.append(self)
        return original(self)

    monkeypatch.setattr(type(model), "get_input_embeddings", observe)
    with t.no_grad():
        assert not worker._can_stage_scoring_inputs(native_runtime, [(1, 2), (3,)])
    assert calls == []
