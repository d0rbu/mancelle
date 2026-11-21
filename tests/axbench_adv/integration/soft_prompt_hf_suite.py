"""Real tiny-HF/PyVene/PEFT soft-prompt oracles in the frozen vendor environment."""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import axbench
import numpy as np
import pandas as pd
import pytest
import torch as t
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from scripts.eval.axbench_adv.soft_prompt_tensors import write_soft_embeddings
from scripts.eval.axbench_adv.vendored import gcg_worker, worker
from scripts.eval.axbench_adv.vendored.prompt_recovery import PromptWrapper
from tests.axbench_adv.integration.gcg_hf_review_suite import (
    _numpy_state_equal,
    _tiny_gpt_neox,
    _tiny_llama,
    _tiny_qwen2,
    _tiny_smollm3,
)

_HOSTS = {
    "llama": _tiny_llama,
    "qwen2": _tiny_qwen2,
    "smollm3": _tiny_smollm3,
    "gpt_neox": _tiny_gpt_neox,
}
_METHODS = ("DiffMean", "LinearProbe", "LoRA", "PromptSteering")
_PREFIX = "<|im_start|>user\n"
_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
_ROWS = [
    {
        "pair_id": 0,
        "positive": {"sample_id": 0, "prompt": "alpha", "response": "yes yes"},
        "negative": {"sample_id": 1, "prompt": "beta gamma delta", "response": "no"},
    },
    {
        "pair_id": 1,
        "positive": {"sample_id": 2, "prompt": "gamma beta", "response": "alpha yes beta"},
        "negative": {"sample_id": 3, "prompt": "delta", "response": "no gamma"},
    },
]


@pytest.fixture
def deterministic_cuda(device: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    if device != "cpu":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        t.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        t.use_deterministic_algorithms(previous, warn_only=warn_only)


def _tokenizer() -> PreTrainedTokenizerFast:
    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "<|im_start|>",
        "<|im_end|>",
        "user",
        "assistant",
        "!",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "yes",
        "no",
        "steer",
        "Question:",
    ]
    words.extend(f"word{index}" for index in range(len(words), 37))
    backend = Tokenizer(
        models.WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        model_max_length=64,
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + "
        "message['content'] + '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    return tokenizer


class _Judge:
    @staticmethod
    def judge(generation: str) -> tuple[float, float, str, int]:
        return 1.0, 2.0, json.dumps({"generation": generation}), 0

    @staticmethod
    def close() -> None:
        pass


def _runtime(
    root: Path,
    *,
    host: str = "qwen2",
    method: str = "PromptSteering",
    dtype: t.dtype = t.float32,
    device: str = "cpu",
    factor: float = 0.6,
    sidecar_dir: Path | None = None,
) -> worker.WorkerRuntime:
    root.mkdir(parents=True, exist_ok=True)
    t.manual_seed(239)
    model = _HOSTS[host](layers=2).to(device=device, dtype=dtype).eval()
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    tokenizer = _tokenizer()
    benchmark = getattr(axbench, method)(
        model, tokenizer, layer=0, device=device, steering_layers=[0]
    )
    if method in ("DiffMean", "LinearProbe"):
        benchmark.make_model(mode="steering", low_rank_dimension=1)
        benchmark.ax.to(device=device, dtype=dtype).eval()
        with t.no_grad():
            benchmark.ax.proj.weight.copy_(t.linspace(-0.08, 0.12, 16).unsqueeze(0))
        benchmark.max_activations = {0: 1.25}
    elif method == "LoRA":
        benchmark.ax_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["query_key_value"] if host == "gpt_neox" else ["q_proj", "v_proj"],
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        ).eval()
        with t.no_grad():
            for module in model.modules():
                lora_b = getattr(module, "lora_B", None)
                if lora_b is not None:
                    lora_b["default"].weight.fill_(0.03125)
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.benchmark_model = benchmark
    runtime.tokenizer = tokenizer
    runtime.method = method
    runtime.concept_id = 0
    runtime.concept = "tiny concept"
    runtime.factor = factor
    runtime.layer = 0
    runtime.max_concepts = 1
    runtime.device = device
    runtime.worker_max_prompt_tokens = 64
    runtime.temperature = 0.75
    runtime.output_length = 3
    runtime.prefix_length = 1
    runtime.model_name = {
        "qwen2": "Qwen/Qwen2.5-3B-Instruct",
        "smollm3": "HuggingFaceTB/SmolLM3-3B-checkpoints",
        "gpt_neox": "/tiny-pythia-sft",
        "llama": "tiny-llama-kernel-fixture",
    }[host]
    runtime.model_revision = "it-soup-APO" if host == "smollm3" else None
    runtime._gcg_main_dump_dir = root
    runtime._gcg_aux_dump_dir = root / "aux"
    runtime._gcg_sidecar_dir = root / "sidecars" if sidecar_dir is None else sidecar_dir
    runtime._gcg_sidecar_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime._gcg_session = None
    runtime.inner_judge = _Judge()
    (root / "train").mkdir(exist_ok=True)
    (root / "inference").mkdir(exist_ok=True)
    (runtime._gcg_aux_dump_dir / "inference").mkdir(parents=True, exist_ok=True)
    t.save(t.linspace(-0.2, 0.3, 16).unsqueeze(0), root / "train/DiffMean_weight.pt")
    t.save(t.zeros(1), root / "train/DiffMean_bias.pt")
    pd.DataFrame({"concept_id": [0], "DiffMean_max_act": [1.25]}).to_parquet(
        root / "inference/latent_data.parquet", index=False
    )
    pd.DataFrame(
        {
            "concept_id": [0],
            "input_id": [0],
            "original_prompt": ["alpha"],
            "steered_input": [f"{_PREFIX}steer\n\nQuestion: alpha{_TAIL}"],
        }
    ).to_parquet(runtime._gcg_aux_dump_dir / "inference/steering_data.parquet", index=False)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "target_model": runtime.model_name,
                "config": {
                    "train": {
                        "model_name": runtime.model_name,
                        "model_revision": runtime.model_revision,
                    },
                    "inference": {
                        "model_name": runtime.model_name,
                        "steering_model_name": runtime.model_name,
                        "model_revision": runtime.model_revision,
                        "steering_model_revision": runtime.model_revision,
                    },
                },
            }
        )
    )
    return runtime


def _prepare_payload(
    objective: str, *, k: int = 3, batch: int = 3, alpha: float = 0.01
) -> dict[str, Any]:
    return {
        "objective": objective,
        "alpha": alpha,
        "rows": _ROWS,
        "suffix_token_count": k,
        "device_batch_size": batch,
        "separator": "\n\n",
    }


def _session(
    runtime: worker.WorkerRuntime,
    objective: str,
    *,
    k: int = 3,
    batch: int = 3,
    alpha: float = 0.01,
) -> gcg_worker.GCGSession:
    session = runtime._gcg()
    if runtime.benchmark_model.model.config.model_type != "llama":
        session.prepare_soft_prompt(_prepare_payload(objective, k=k, batch=batch, alpha=alpha))
        return session
    # Llama has no admitted campaign source wrapper. Its real model/PyVene/PEFT
    # kernels still receive the identical layout construction and calibration.
    session.objective, session.alpha, session.device_batch_size = objective, alpha, batch
    session.seed_token_ids = tuple(runtime.tokenizer(" ".join(["!"] * k))["input_ids"])
    session._allowed_token_id_set = frozenset(range(7, 37))
    session._wrapper = PromptWrapper(
        model="tiny-llama",
        revision=None,
        prefix=_PREFIX,
        suffix=_TAIL,
        error_model_name="tiny-llama",
        source_model_name=runtime.model_name,
    )
    pairs = []
    for row in _ROWS:
        sides = [
            gcg_worker._completion_layout(
                runtime,
                row[side],
                wrapper=session._wrapper,
                steering_prefix="steer",
                separator="\n\n",
                seed_text=" ".join(["!"] * k),
                seed_ids=session.seed_token_ids,
                label=side,
            )
            for side in ("positive", "negative")
        ]
        pairs.append(gcg_worker._PairLayout(row["pair_id"], *sides))
    session.layouts = tuple(pairs)
    if objective == "activation_l2":
        with gcg_worker._preserve_runtime(runtime):
            session._prepare_activation()
    session._soft_prepared = True
    return session


def _seed(session: gcg_worker.GCGSession) -> t.Tensor:
    weight = session.runtime.benchmark_model.model.get_input_embeddings().weight
    return weight.detach()[list(session.seed_token_ids)].float().cpu().clone()


def _soft_call(
    session: gcg_worker.GCGSession, name: str, embeddings: t.Tensor, indices: list[int]
) -> Any:
    path = session.runtime._gcg_sidecar_dir / "soft_embeddings.safetensors"
    write_soft_embeddings(path, embeddings)
    result = getattr(session, name)({"member_indices": indices})
    assert not path.exists()
    if name == "soft_gradient":
        return load_file(session.runtime._gcg_sidecar_dir / "gradients.safetensors")["gradients"]
    return result["losses"]


@pytest.mark.parametrize("operation", ["soft_score", "soft_gradient"])
def test_soft_input_is_consumed_once(tmp_path: Path, operation: str) -> None:
    runtime = _runtime(tmp_path)
    session = _session(runtime, "response_margin")
    _soft_call(session, operation, _seed(session), [0])
    with patch.object(
        runtime.benchmark_model.model, "forward", side_effect=AssertionError("forward ran")
    ):
        with pytest.raises(ValueError, match="unavailable"):
            getattr(session, operation)({"member_indices": [0]})


def _rng() -> tuple[Any, Any, t.Tensor, list[t.Tensor] | None]:
    return (
        random.getstate(),
        np.random.get_state(),
        t.random.get_rng_state(),
        t.cuda.get_rng_state_all() if t.cuda.is_initialized() else None,
    )


def _assert_rng(
    left: tuple[Any, Any, t.Tensor, list[t.Tensor] | None],
    right: tuple[Any, Any, t.Tensor, list[t.Tensor] | None],
) -> None:
    assert left[0] == right[0]
    assert _numpy_state_equal(left[1], right[1])
    assert t.equal(left[2], right[2])
    # The existing GCG guard can initialize the available CUDA RNG lazily on
    # its first CPU call. Compare CUDA states when the caller already had one.
    if left[3] is not None:
        assert right[3] is not None
        assert all(t.equal(a, b) for a, b in zip(left[3], right[3], strict=True))


@pytest.mark.parametrize("host", _HOSTS)
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.slow)])
@pytest.mark.parametrize("setting", [(1, 1, 0.0, 0.0), (3, 3, 1.0, 1.0), (3, 4, 0.6, 0.01)])
def test_exact_seed_objective_and_gradient_parity(
    tmp_path: Path,
    host: str,
    method: str,
    objective: str,
    dtype: t.dtype,
    device: str,
    deterministic_cuda: None,
    setting: tuple[int, int, float, float],
) -> None:
    k, batch, factor, alpha = setting
    runtime = _runtime(
        tmp_path, host=host, method=method, dtype=dtype, factor=factor, device=device
    )
    session = _session(runtime, objective, k=k, batch=batch, alpha=alpha)
    seed = _seed(session)
    indices = [1, 0]
    rng = _rng()
    model = runtime.benchmark_model.model
    parameters = [
        (parameter, parameter.detach().clone(), parameter.requires_grad)
        for parameter in model.parameters()
    ]
    expected = session.score(
        {
            "candidate_token_ids": [list(session.seed_token_ids)],
            "member_indices": indices,
            "candidate_eval_device_batch_size": batch,
        }
    )["losses"]
    session.gradient({"token_ids": list(session.seed_token_ids), "member_indices": indices})
    gradient = load_file(runtime._gcg_sidecar_dir / "gradients.safetensors")["gradients"]
    assert _soft_call(session, "soft_score", seed, indices) == expected
    assert t.equal(_soft_call(session, "soft_gradient", seed, indices), gradient)
    assert not gradient.requires_grad and gradient.dtype is t.float32
    _assert_rng(rng, _rng())
    for parameter, value, requires_grad in parameters:
        assert t.equal(parameter, value)
        assert parameter.requires_grad is requires_grad
        assert parameter.grad is None
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("host", _HOSTS)
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
def test_nonseed_fp32_finite_difference(
    tmp_path: Path, host: str, method: str, objective: str
) -> None:
    runtime = _runtime(tmp_path, host=host, method=method)
    session = _session(runtime, objective)
    embeddings = _seed(session) + t.linspace(-0.15, 0.1, 48).reshape(3, 16)
    direction = t.linspace(0.4, -0.7, 48).reshape(3, 16)
    direction /= t.linalg.vector_norm(direction)
    gradient = _soft_call(session, "soft_gradient", embeddings, [1, 0])
    epsilon = 0.001
    plus = t.tensor(_soft_call(session, "soft_score", embeddings + epsilon * direction, [1, 0]))
    minus = t.tensor(_soft_call(session, "soft_score", embeddings - epsilon * direction, [1, 0]))
    numerical = (plus - minus) / (2 * epsilon)
    analytical = (gradient * direction).sum(dim=(1, 2))
    t.testing.assert_close(analytical, numerical, rtol=0.015, atol=3e-4)
    assert bool(analytical.ne(0).any())


def _direct_embedded_forward(
    runtime: worker.WorkerRuntime, embeddings: t.Tensor, *, logits_to_keep: int = 0
) -> tuple[t.Tensor, t.Tensor]:
    """Independent native-model oracle, including the actual additive method.

    It bypasses GCG's scoring, collectors, layer-prefix optimization and PyVene.
    PEFT's LoRA modules are already installed in the underlying native model.
    """
    model = runtime.benchmark_model.model
    captured: list[t.Tensor] = []

    def intervene_and_capture(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if runtime.method in {"DiffMean", "LinearProbe"}:
            weight = runtime.benchmark_model.ax.proj.weight.detach()[[runtime.concept_id]]
            maximum = runtime.benchmark_model.max_activations[runtime.concept_id]
            direction = (
                t.tensor([[maximum]], dtype=t.float32)
                * t.tensor([[runtime.factor]], dtype=t.float32)
                * weight
            )
            hidden = (hidden + direction.unsqueeze(1)).to(hidden.dtype)
        captured.append(hidden.clone())
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = gcg_worker.decoder_layers(model)[runtime.layer].register_forward_hook(
        intervene_and_capture
    )
    try:
        output = model(
            inputs_embeds=embeddings,
            attention_mask=t.ones(embeddings.shape[:2], dtype=t.long),
            logits_to_keep=logits_to_keep,
            use_cache=False,
        )
    finally:
        handle.remove()
    assert len(captured) == 1
    return output.logits, captured[0]


def _direct_side_loss(
    runtime: worker.WorkerRuntime, side: dict[str, Any], soft: t.Tensor, objective: str
) -> t.Tensor:
    content = side["prompt"]
    if runtime.method == "PromptSteering":
        content = f"steer\n\nQuestion: {content}"
    tokenizer = runtime.tokenizer
    clean = f"{_PREFIX}{content}{_TAIL}"
    seed_text = " ".join(["!"] * len(soft))
    prompt = tokenizer(f"{_PREFIX}{content}\n\n{seed_text}{_TAIL}")["input_ids"]
    tail = tokenizer(_TAIL, add_special_tokens=False)["input_ids"]
    response = tokenizer(side["response"], add_special_tokens=False)["input_ids"]
    begin = len(prompt) - len(tail) - len(soft)
    embedding = runtime.benchmark_model.model.get_input_embeddings()
    base = embedding(t.tensor([prompt + response])).detach()
    inputs = t.cat((base[:, :begin], soft.unsqueeze(0), base[:, begin + len(soft) :]), dim=1)
    logits, hidden = _direct_embedded_forward(runtime, inputs)
    if objective == "response_margin":
        selected = logits[0, len(prompt) - 1 : len(prompt + response) - 1].float()
        losses = F.cross_entropy(selected, t.tensor(response), reduction="none")
        sizes = t.tensor(
            [
                len(tokenizer.decode([token], clean_up_tokenization_spaces=False).encode("utf-8"))
                for token in response
            ],
            dtype=t.float32,
        )
        return (losses * sizes).sum() / (math.log(2.0) * sizes.sum())
    clean_ids = tokenizer(clean)["input_ids"]
    with t.no_grad():
        _, clean_hidden = _direct_embedded_forward(
            runtime, embedding(t.tensor([clean_ids + response])).detach()
        )
    # Read calibration inputs independently, without prepared offsets/references.
    root = runtime._gcg_main_dump_dir
    saved = t.load(root / "train/DiffMean_weight.pt", weights_only=True)[0].to(t.bfloat16)
    maximum = float(
        pd.read_parquet(root / "inference/latent_data.parquet")["DiffMean_max_act"].max()
    )
    offset = (
        t.tensor([[maximum]], dtype=t.float32)
        * t.tensor([[runtime.factor]], dtype=t.float32)
        * saved.unsqueeze(0)
    )[0]
    delta = (
        hidden[0, begin + len(soft) :].float()
        + offset
        - clean_hidden[0, len(clean_ids) - len(tail) :].float()
    )
    return delta.square().sum(-1).sqrt().mean()


@pytest.mark.parametrize("host", _HOSTS)
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
def test_nonseed_score_and_vjp_against_independent_native_model(
    tmp_path: Path, host: str, method: str, objective: str
) -> None:
    runtime = _runtime(tmp_path, host=host, method=method, factor=0.6)
    session = _session(runtime, objective, batch=1, alpha=0.37)
    soft = (_seed(session) + t.linspace(-0.25, 0.7, 48).reshape(3, 16)).requires_grad_(True)
    scores = _soft_call(session, "soft_score", soft.detach(), [1, 0])
    gradients = _soft_call(session, "soft_gradient", soft.detach(), [1, 0])
    expected_values, expected_gradients = [], []
    for index in [1, 0]:
        row = _ROWS[index]
        positive = _direct_side_loss(runtime, row["positive"], soft, objective)
        negative = _direct_side_loss(runtime, row["negative"], soft, objective)
        value = (
            0.5 * (positive + negative)
            if objective == "activation_l2"
            else negative - 0.37 * positive
        )
        expected_values.append(value.detach())
        expected_gradients.append(t.autograd.grad(value, soft)[0])
    # Independent reduction/backprop grouping can differ in its last FP32 bits.
    t.testing.assert_close(t.tensor(scores), t.stack(expected_values), rtol=1e-6, atol=1e-7)
    t.testing.assert_close(gradients, t.stack(expected_gradients), rtol=2e-5, atol=1e-7)
    assert t.count_nonzero(gradients) > 0


def _generation_payload(k: int) -> dict[str, Any]:
    suffix = " ".join(["!"] * k)
    return {
        "input": f"{_PREFIX}alpha\n\n{suffix}{_TAIL}",
        "steered_input": f"{_PREFIX}steer\n\nQuestion: alpha\n\n{suffix}{_TAIL}",
        "input_id": 0,
        "row_seed": 4583,
    }


@pytest.mark.parametrize("host", _HOSTS)
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.slow)])
def test_seed_generation_parity_and_nonseed_embedding_logit_oracle(
    tmp_path: Path, host: str, method: str, dtype: t.dtype, device: str, deterministic_cuda: None
) -> None:
    runtime = _runtime(tmp_path, host=host, method=method, dtype=dtype, device=device)
    session = _session(runtime, "response_margin")
    seed = _seed(session)
    payload = _generation_payload(len(session.seed_token_ids))
    model = runtime.benchmark_model.model
    embedding = model.get_input_embeddings()
    original = model.forward
    observations: list[tuple[t.Tensor, t.Tensor, t.Tensor]] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        captured: list[t.Tensor] = []
        handle = embedding.register_forward_hook(lambda _m, _a, out: captured.append(out.clone()))
        try:
            result = original(*args, **kwargs)
        finally:
            handle.remove()
        observations.append((kwargs["input_ids"].clone(), captured[0], result.logits.clone()))
        return result

    with patch.object(model, "forward", recording):
        ordinary = runtime.evaluate(payload)
        ordinary_observations = observations[:]
        ordinary_rng = _rng()
        observations.clear()
        seeded = runtime.evaluate(payload, soft_embeddings=seed)
        assert seeded == ordinary
        _assert_rng(ordinary_rng, _rng())
        assert len(observations) == len(ordinary_observations) == 3
        for actual, expected in zip(observations, ordinary_observations, strict=True):
            assert all(t.equal(a, e) for a, e in zip(actual, expected, strict=True))
        assert model.forward is recording
        observations.clear()
        perturbed = seed + t.linspace(-0.7, 0.9, seed.numel()).reshape(seed.shape)
        runtime.evaluate(payload, soft_embeddings=perturbed)
        first_ids, first_embeddings, first_logits = observations[0]
        tail = runtime.tokenizer(_TAIL, add_special_tokens=False)["input_ids"]
        begin = first_ids.shape[1] - len(tail) - len(seed)
        assert t.equal(
            first_embeddings[0, begin : begin + len(seed)],
            perturbed.to(device=first_embeddings.device, dtype=dtype),
        )
        assert t.equal(first_embeddings[:, :begin], ordinary_observations[0][1][:, :begin])
        assert t.equal(
            first_embeddings[:, begin + len(seed) :],
            ordinary_observations[0][1][:, begin + len(seed) :],
        )
        assert not t.equal(first_logits, ordinary_observations[0][2])
        for ids, values, _logits in observations[1:]:
            assert ids.shape[1] == 1
            with t.no_grad():
                assert t.equal(values, embedding(ids))
        assert model.forward is recording and not embedding._forward_hooks
        observations.clear()
        assert runtime.evaluate(payload) == ordinary
    assert "forward" not in vars(model)


@pytest.mark.parametrize(
    ("host", "method"),
    [
        ("qwen2", "DiffMean"),
        ("smollm3", "LoRA"),
        ("gpt_neox", "LinearProbe"),
        ("llama", "PromptSteering"),
    ],
)
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
@pytest.mark.parametrize("cached", [False, True])
def test_nonseed_generation_matches_direct_embedding_inputs(
    tmp_path: Path, host: str, method: str, dtype: t.dtype, cached: bool
) -> None:
    runtime = _runtime(tmp_path, host=host, method=method, dtype=dtype)
    session = _session(runtime, "response_margin")
    model = runtime.benchmark_model.model
    model.generation_config.use_cache = cached
    soft = _seed(session) + t.linspace(-0.7, 0.9, 48).reshape(3, 16)
    observed: list[tuple[t.Tensor, t.Tensor]] = []
    original = model.forward

    def capture(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        observed.append((kwargs["input_ids"].clone(), output.logits.clone()))
        return output

    with patch.object(model, "forward", capture):
        runtime.evaluate(_generation_payload(3), soft_embeddings=soft)
        assert model.forward is capture
    assert "forward" not in vars(model)
    tail_count = len(runtime.tokenizer(_TAIL, add_special_tokens=False)["input_ids"])
    begin = observed[0][0].shape[1] - tail_count - len(soft)
    with t.inference_mode():
        for ids, logits in observed[:1] if cached else observed:
            base = model.get_input_embeddings()(ids)
            embeds = t.cat(
                (base[:, :begin], soft.to(dtype).unsqueeze(0), base[:, begin + len(soft) :]), dim=1
            )
            # PyVene with unit_locations=None applies the positive additive
            # method on every forward, including full-prefix recurrence.
            direct, _ = _direct_embedded_forward(runtime, embeds, logits_to_keep=logits.shape[1])
            t.testing.assert_close(logits, direct, rtol=0, atol=0)
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
def test_independent_oracle_rejects_shared_objective_mutation(
    tmp_path: Path, objective: str
) -> None:
    original = gcg_worker.GCGSession._prepare_activation

    def wrong_scale(self: gcg_worker.GCGSession) -> Any:
        result = original(self)
        self._direction_offset *= 2
        return result

    def wrong_sign(negative: t.Tensor, positive: t.Tensor, *, alpha: float) -> t.Tensor:
        return negative + alpha * positive

    # A shared bug would pass GCG-versus-soft seed parity. The independent
    # source-backed oracle must detect both of these concrete mutations.
    target, name, replacement = (
        (gcg_worker.GCGSession, "_prepare_activation", wrong_scale)
        if objective == "activation_l2"
        else (gcg_worker, "response_margin_gradient", wrong_sign)
    )
    with patch.object(target, name, replacement), pytest.raises(AssertionError):
        test_nonseed_score_and_vjp_against_independent_native_model(
            tmp_path, "qwen2", "DiffMean", objective
        )


@pytest.mark.parametrize("method", _METHODS)
def test_generation_hook_deletion_and_forward_failure_restore_exact_callable(
    tmp_path: Path, method: str
) -> None:
    runtime = _runtime(tmp_path, method=method)
    session = _session(runtime, "response_margin")
    seed, payload = _seed(session), _generation_payload(3)
    model = runtime.benchmark_model.model
    embedding = model.get_input_embeddings()
    original = model.forward

    def deleting(*args: Any, **kwargs: Any) -> Any:
        embedding._forward_hooks.clear()
        return original(*args, **kwargs)

    with patch.object(model, "forward", deleting):
        with pytest.raises(RuntimeError, match="did not inject"):
            runtime.evaluate(payload, soft_embeddings=seed)
        assert model.forward is deleting
        assert not embedding._forward_hooks
    assert "forward" not in vars(model)

    def failing(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise RuntimeError("injected forward failure")

    with patch.object(model, "forward", failing):
        with pytest.raises(RuntimeError, match="injected forward failure"):
            runtime.evaluate(payload, soft_embeddings=seed)
        assert model.forward is failing
        assert not embedding._forward_hooks
    assert "forward" not in vars(model)
    assert runtime.evaluate(payload, soft_embeddings=seed) == runtime.evaluate(payload)


@pytest.mark.parametrize("method", _METHODS)
def test_no_cache_full_prefix_recurrence(tmp_path: Path, method: str) -> None:
    runtime = _runtime(tmp_path, method=method)
    session = _session(runtime, "response_margin")
    runtime.benchmark_model.model.generation_config.use_cache = False
    seed, payload = _seed(session), _generation_payload(3)
    assert runtime.evaluate(payload, soft_embeddings=seed) == runtime.evaluate(payload)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
@pytest.mark.parametrize("failure", ["forward", "backward"])
def test_search_failure_restores_gradients_modes_hooks_and_rng(
    tmp_path: Path, method: str, objective: str, failure: str
) -> None:
    runtime = _runtime(tmp_path, method=method)
    session = _session(runtime, objective)
    seed = _seed(session)
    model = runtime.benchmark_model.model
    model.train()
    model.get_input_embeddings().eval()
    parameter = model.get_input_embeddings().weight
    parameter.requires_grad_(True)
    old_grad = t.full_like(parameter, 13.0)
    parameter.grad = old_grad
    modules = list(model.modules())
    if hasattr(runtime.benchmark_model, "ax"):
        modules.extend(runtime.benchmark_model.ax.modules())
    modes = [module.training for module in modules]
    layer = gcg_worker.decoder_layers(model)[0]
    handle = layer.register_forward_hook(lambda _m, _a, _o: None)
    hooks = layer._forward_hooks
    saved_hooks = tuple(hooks.items())
    rng, padding = _rng(), runtime.tokenizer.padding_side
    original_forward, original_grad = model.forward, t.autograd.grad

    def fail_after_call(*args: Any, **kwargs: Any) -> Any:
        (original_forward if failure == "forward" else original_grad)(*args, **kwargs)
        random.random()
        np.random.random()
        t.rand(1)
        runtime.tokenizer.padding_side = "left"
        model.train()
        raise RuntimeError(f"injected {failure} failure")

    target, attribute = (model, "forward") if failure == "forward" else (t.autograd, "grad")
    try:
        with patch.object(target, attribute, fail_after_call):
            with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
                _soft_call(session, "soft_gradient", seed, [1, 0])
        assert parameter.requires_grad and parameter.grad is old_grad
        assert t.equal(parameter.grad, t.full_like(parameter, 13.0))
        assert [module.training for module in modules] == modes
        assert layer._forward_hooks is hooks and tuple(hooks.items()) == saved_hooks
        assert runtime.tokenizer.padding_side == padding
        _assert_rng(rng, _rng())
        assert not (runtime._gcg_sidecar_dir / "soft_embeddings.safetensors").exists()
        assert bool(t.isfinite(_soft_call(session, "soft_gradient", seed, [1, 0])).all())
    finally:
        handle.remove()


@pytest.mark.parametrize("dtype", [t.float16, t.bfloat16])
@pytest.mark.parametrize("operation", ["soft_score", "soft_gradient"])
def test_native_cast_overflow_fails_before_forward(
    tmp_path: Path, dtype: t.dtype, operation: str
) -> None:
    runtime = _runtime(tmp_path, dtype=dtype)
    session = _session(runtime, "response_margin")
    seed = _seed(session)
    with patch.object(
        runtime.benchmark_model.model, "forward", side_effect=AssertionError("forward ran")
    ):
        with pytest.raises(ValueError, match="native dtype"):
            _soft_call(session, operation, t.full_like(seed, t.finfo(t.float32).max), [0])
    assert not (runtime._gcg_sidecar_dir / "soft_embeddings.safetensors").exists()


def test_generation_requires_injection_and_rejects_unverified_geometry(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    session = _session(runtime, "response_margin")
    seed, payload = _seed(session), _generation_payload(3)
    prompt = payload["steered_input"]
    ids = runtime.tokenizer(prompt, add_special_tokens=True)["input_ids"]
    model = runtime.benchmark_model.model
    with pytest.raises(RuntimeError, match="did not inject"):
        with session.soft_generation(seed, prompt=prompt, token_ids=ids):
            pass
    wrong = t.tensor([ids])
    wrong[0, 0] = 12
    with pytest.raises(ValueError, match="verified prompt prefix"):
        with session.soft_generation(seed, prompt=prompt, token_ids=ids):
            model(input_ids=wrong)
    with pytest.raises(ValueError, match="unsupported cache"):
        with session.soft_generation(seed, prompt=prompt, token_ids=ids):
            model(input_ids=t.tensor([[7]]), past_key_values=object())
    with pytest.raises(ValueError, match="cached-decode geometry"):
        with session.soft_generation(seed, prompt=prompt, token_ids=ids), t.no_grad():
            result = model(input_ids=t.tensor([ids]), use_cache=True)
            model(input_ids=t.tensor([ids]), past_key_values=result.past_key_values)
    bad_span = list(ids)
    bad_span[-len(runtime.tokenizer(_TAIL)["input_ids"]) - 1] = 12
    with pytest.raises(ValueError, match="suffix span"):
        with session.soft_generation(seed, prompt=prompt, token_ids=bad_span):
            pass
    assert "forward" not in vars(model)
    assert not model.get_input_embeddings()._forward_hooks


@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
def test_prepare_identity_and_failed_reprepare_invalidates_soft_state(
    tmp_path: Path, objective: str
) -> None:
    runtime = _runtime(tmp_path)
    session = runtime._gcg()
    payload = _prepare_payload(objective)
    ordinary = session.prepare(payload)
    soft = session.prepare_soft_prompt(payload)
    identity = dict(soft["identity"])
    assert identity.pop("embedding_layout") == [
        {"name": "input_embeddings", "width": 16, "dtype": "torch.float32"}
    ]
    assert identity.pop("generation_conditioning") == "ordinary_prompt"
    assert {**soft, "identity": identity} == ordinary
    seed = _seed(session)
    with pytest.raises(ValueError, match="fields must be exactly"):
        session.prepare_soft_prompt({})
    with pytest.raises(ValueError, match="soft_prepare must succeed"):
        _soft_call(session, "soft_score", seed, [0])
    session.close()
    assert not list(runtime._gcg_sidecar_dir.glob("soft*.safetensors"))
    session.prepare(payload)
    with pytest.raises(ValueError, match="soft_prepare must succeed"):
        _soft_call(session, "soft_gradient", seed, [0])


def _worker_fixture(payload: dict[str, Any]) -> worker.WorkerRuntime:
    return _runtime(
        Path(payload["fixture_dir"]),
        host=payload["host"],
        method=payload["method"],
        factor=payload["factor"],
        sidecar_dir=Path(payload["gcg_sidecar_dir"]),
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("expected --worker")
    protocol_writer = sys.stdout.buffer
    sys.stdout = sys.stderr
    raise SystemExit(
        worker.serve(sys.stdin.buffer, protocol_writer, runtime_factory=_worker_fixture)
    )
