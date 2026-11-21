"""CPU contract tests for the dependency-isolated vendored GCG worker math."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch as t
import torch.nn.functional as F
from safetensors import safe_open


def _load_gcg_session() -> type[Any]:
    module_name = "_test_axbench_gcg_worker"
    axbench = ModuleType("axbench")
    utils = ModuleType("axbench.utils")
    model_utils = ModuleType("axbench.utils.model_utils")
    cast(Any, model_utils).decoder_layers = lambda model: model.layers
    pyvene = ModuleType("pyvene")
    cast(Any, pyvene).CollectIntervention = object
    cast(Any, pyvene).IntervenableConfig = object
    cast(Any, pyvene).IntervenableModel = object

    class _FakeDirection(t.nn.Module):
        def __init__(self, weight: t.Tensor) -> None:
            super().__init__()
            self.proj = t.nn.Linear(weight.shape[1], weight.shape[0], bias=True)
            self.proj.weight.data.copy_(weight)

    class _FakeDiffMean:
        def __init__(self, model: t.nn.Module, tokenizer: object, **_kwargs: object) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.ax: _FakeDirection
            self.max_activations: dict[int, float] = {}

        def load(self, dump_dir: Path, **_kwargs: object) -> None:
            weight = t.load(Path(dump_dir) / "DiffMean_weight.pt", weights_only=True)
            self.ax = _FakeDirection(weight)

        def to(self, device: str) -> None:
            self.ax.to(device)

        def pre_compute_mean_activations(self, dump_dir: Path) -> None:
            frame = pd.read_parquet(Path(dump_dir) / "latent_data.parquet")
            value = float(frame.loc[frame["concept_id"].eq(0), "DiffMean_max_act"].max())
            self.max_activations = {0: value if value > 0 else 50.0}

    cast(Any, axbench).DiffMean = _FakeDiffMean
    names = ("axbench", "axbench.utils", "axbench.utils.model_utils", "pyvene")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["axbench"] = axbench
    sys.modules["axbench.utils"] = utils
    sys.modules["axbench.utils.model_utils"] = model_utils
    sys.modules["pyvene"] = pyvene
    path = Path(__file__).resolve().parents[3] / "scripts/eval/axbench_adv/vendored/gcg_worker.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, prior in saved.items():
            if prior is None:
                del sys.modules[name]
            else:
                sys.modules[name] = prior
    return module.GCGSession


GCGSession = _load_gcg_session()


class _Tokenizer:
    all_special_ids = [0]
    added_tokens_decoder: dict[int, object] = {}
    eos_token_id = 0
    padding_side = "right"

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        token_ids: list[int] = []
        position = 0
        while position < len(text):
            if text.startswith(" !", position):
                token_ids.append(127)
                position += 2
            else:
                token_ids.append(ord(text[position]))
                position += 1
        return {"input_ids": token_ids}

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "".join(" !" if token_id == 127 else chr(token_id) for token_id in token_ids)

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs: object,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        return (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\n"
            f"{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        )


class _PythiaTokenizer(_Tokenizer):
    """Reproduces GPT-NeoX BPE's separator quirk: ``"\\n\\n"`` is one token unless ``"!"`` follows."""

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        token_ids: list[int] = []
        position = 0
        while position < len(text):
            if text.startswith("\n\n", position) and not text.startswith("\n\n!", position):
                token_ids.append(200)
                position += 2
            elif text.startswith(" !", position):
                token_ids.append(127)
                position += 2
            else:
                token_ids.append(ord(text[position]))
                position += 1
        return {"input_ids": token_ids}

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "".join(
            "\n\n" if token_id == 200 else " !" if token_id == 127 else chr(token_id)
            for token_id in token_ids
        )

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert not kwargs
        return f"<|endoftext|>\nuser:\n{messages[0]['content']}<|endoftext|>\nassistant:\n"


class _SmolNoThinkTokenizer(_Tokenizer):
    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = True,
        **kwargs: object,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert not kwargs
        return (
            "<|im_start|>system\nSmolLM<|im_start|>user\n"
            f"{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n"
        )


class _TinyCausalLm(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2", hidden_size=5)
        self.embedding = t.nn.Embedding(128, 5)
        self.lm_head = t.nn.Linear(5, 128, bias=False)
        self.layers = t.nn.ModuleList([t.nn.Identity()])

    def get_input_embeddings(self) -> t.nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: t.Tensor,
        attention_mask: t.Tensor,
        use_cache: bool,
        logits_to_keep: int | t.Tensor = 0,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.layers[0](inputs_embeds.cumsum(dim=1))
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        return SimpleNamespace(logits=self.lm_head(hidden[:, indices, :]))


def _runtime(
    tmp_path: Path,
    *,
    method: str = "PromptSteering",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    model_revision: str | None = None,
    model_type: str = "qwen2",
    tokenizer: _Tokenizer | None = None,
) -> SimpleNamespace:
    model = _TinyCausalLm()
    model.config.model_type = model_type
    generator = t.Generator().manual_seed(391)
    with t.no_grad():
        model.embedding.weight.copy_(t.randn(model.embedding.weight.shape, generator=generator))
        model.lm_head.weight.copy_(t.randn(model.lm_head.weight.shape, generator=generator))
    aux_dir = tmp_path / "aux"
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir(exist_ok=True)
    (aux_dir / "inference").mkdir(parents=True, exist_ok=True)
    prefix = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
        "assistant.<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    pd.DataFrame(
        {
            "concept_id": [0],
            "input_id": [0],
            "original_prompt": ["source"],
            "steered_input": [f"{prefix}steer\n\nQuestion: source{suffix}"],
        }
    ).to_parquet(aux_dir / "inference" / "steering_data.parquet", index=False)
    return SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model, ax_model=model),
        concept_id=0,
        device="cpu",
        factor=1.25,
        layer=0,
        method=method,
        model_name=model_name,
        model_revision=model_revision,
        max_concepts=1,
        tokenizer=_Tokenizer() if tokenizer is None else tokenizer,
        worker_max_prompt_tokens=512,
        _gcg_main_dump_dir=tmp_path,
        _gcg_aux_dump_dir=aux_dir,
        _gcg_sidecar_dir=sidecar_dir,
    )


def _write_manifest(
    directory: Path,
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    model_revision: str | None = None,
) -> None:
    revision = "null" if model_revision is None else f'"{model_revision}"'
    directory.joinpath("run_manifest.json").write_text(
        f'{{"target_model":"{model_name}","config":{{"train":{{"model_name":"{model_name}",'
        f'"model_revision":{revision}}},"inference":{{"model_name":"{model_name}",'
        f'"model_revision":{revision},"steering_model_name":"{model_name}",'
        f'"steering_model_revision":{revision}}}}}}}',
        encoding="utf-8",
    )


def _write_activation_artifacts(directory: Path) -> None:
    (directory / "train").mkdir(exist_ok=True)
    (directory / "inference").mkdir(exist_ok=True)
    t.save(t.tensor([[1.0, -0.5, 0.25, 0.75, -1.0]]), directory / "train/DiffMean_weight.pt")
    t.save(t.zeros(1), directory / "train/DiffMean_bias.pt")
    pd.DataFrame({"concept_id": [0], "DiffMean_max_act": [2.0]}).to_parquet(
        directory / "inference/latent_data.parquet", index=False
    )


def _prepare(
    session: Any, *, suffix_token_count: int = 1, separator: str = "\n\n"
) -> dict[str, Any]:
    return session.prepare(
        {
            "alpha": 0.25,
            "objective": "response_margin",
            "rows": [
                {
                    "pair_id": 0,
                    "positive": {"sample_id": 11, "prompt": "P", "response": "A"},
                    "negative": {"sample_id": 12, "prompt": "N", "response": "B"},
                }
            ],
            "separator": separator,
            "suffix_token_count": suffix_token_count,
            "device_batch_size": 2,
        }
    )


def test_prepare_sidecar_filter_score_and_gradient_are_exact(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    runtime = _runtime(tmp_path)
    session = GCGSession(runtime)
    prepared = _prepare(session)
    assert set(prepared) == {
        "allowed_token_ids",
        "direction_calibration",
        "embedding_sidecar",
        "identity",
        "layouts_sha256",
        "member_count",
        "seed_token_ids",
    }
    assert prepared["member_count"] == 1
    assert prepared["seed_token_ids"] == [ord("!")]
    assert prepared["direction_calibration"] is None
    sidecar = prepared["embedding_sidecar"]
    path = runtime._gcg_sidecar_dir / "input_embeddings.safetensors"
    raw = path.read_bytes()
    assert path.stat().st_mode & 0o077 == 0
    assert sidecar["shape"] == [128, 5]
    assert hashlib.sha256(raw).hexdigest() == sidecar["sha256"]
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"embedding_weight"}
        first_embedding = handle.get_tensor("embedding_weight")
    first_embedding_copy = first_embedding.clone()
    first_embedding_row = first_embedding[0]
    first_inode = path.stat().st_ino

    candidate = [ord("?")]
    filtered = session.filter({"candidate_token_ids": [candidate, [0]]})["candidates"]
    assert filtered[0] == {"error": None, "text": "?", "token_ids": candidate, "valid": True}
    assert filtered[1]["valid"] is False
    assert filtered[1]["error"]

    spaced = session.filter({"candidate_token_ids": [[127]]})["candidates"][0]
    assert spaced["valid"] is False
    assert "outer-spaced" in spaced["error"]

    result = session.score(
        {
            "candidate_eval_device_batch_size": 1,
            "candidate_token_ids": [candidate],
            "member_indices": [0],
        }
    )
    model = runtime.benchmark_model.model
    side_losses = []
    for side in (session.layouts[0].negative, session.layouts[0].positive):
        sequence = side.fixed_before + tuple(candidate) + side.fixed_after + side.response_ids
        ids = t.tensor([sequence])
        output = model(
            inputs_embeds=model.embedding(ids), attention_mask=t.ones_like(ids), use_cache=False
        )
        response_start = len(side.fixed_before) + 1 + len(side.fixed_after)
        logits = output.logits[0, response_start - 1 : -1].float()
        side_losses.append(
            F.cross_entropy(logits, t.tensor(side.response_ids), reduction="sum") / math.log(2.0)
        )
    expected = side_losses[0] - 0.25 * side_losses[1]
    assert result == {
        "candidate_count": 1,
        "losses": pytest.approx([float(expected.detach())]),
        "member_count": 1,
    }

    gradient_result = session.gradient({"member_indices": [0], "token_ids": candidate})
    assert gradient_result["member_indices"] == [0]
    gradient_sidecar = gradient_result["gradient_sidecar"]
    gradient_path = runtime._gcg_sidecar_dir / "gradients.safetensors"
    assert hashlib.sha256(gradient_path.read_bytes()).hexdigest() == gradient_sidecar["sha256"]
    with safe_open(gradient_path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"gradients"}
        gradient = handle.get_tensor("gradients")[0]
    assert gradient_sidecar["shape"] == [1, 1, 5]
    assert t.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0

    with t.no_grad():
        runtime.benchmark_model.model.embedding.weight.add_(1.0)
    replacement = _prepare(session, suffix_token_count=2)
    assert len(replacement["seed_token_ids"]) == 2
    assert path.exists()
    assert path.stat().st_ino != first_inode
    with safe_open(path, framework="pt", device="cpu") as handle:
        second_embedding = handle.get_tensor("embedding_weight")
    assert t.equal(first_embedding, first_embedding_copy)
    assert t.equal(first_embedding_row, first_embedding_copy[0])
    assert not t.equal(first_embedding, second_embedding)
    session.close()
    assert not path.exists()
    assert t.equal(first_embedding, first_embedding_copy)


@pytest.mark.parametrize(
    ("host", "revision", "model_type", "tokenizer", "tail", "seed_token_ids"),
    (
        (
            "pythia_sft",
            None,
            "gpt_neox",
            _PythiaTokenizer(),
            "<|endoftext|>\nassistant:\n",
            [ord("!"), 127],
        ),
        (
            "HuggingFaceTB/SmolLM3-3B",
            "a07cc9a04f16550a088caea529712d1d335b0ac1",
            "smollm3",
            _SmolNoThinkTokenizer(),
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n",
            [ord("!"), 127],
        ),
    ),
)
def test_non_qwen_prepare_uses_exact_tail_and_rendered_token_budget(
    tmp_path: Path,
    host: str,
    revision: str | None,
    model_type: str,
    tokenizer: _Tokenizer,
    tail: str,
    seed_token_ids: list[int],
) -> None:
    model_name = str(tmp_path / host) if model_type == "gpt_neox" else host
    _write_manifest(tmp_path, model_name=model_name, model_revision=revision)
    runtime = _runtime(
        tmp_path,
        method="LoRA",
        model_name=model_name,
        model_revision=revision,
        model_type=model_type,
        tokenizer=tokenizer,
    )
    session = GCGSession(runtime)
    prepared = _prepare(session, suffix_token_count=2)
    candidate = prepared["seed_token_ids"]
    [filtered] = session.filter({"candidate_token_ids": [candidate]})["candidates"]
    assert filtered["valid"], filtered["error"]
    scored = session.score(
        {
            "candidate_eval_device_batch_size": 2,
            "candidate_token_ids": [candidate],
            "member_indices": [0],
        }
    )
    assert math.isfinite(scored["losses"][0])
    gradient = session.gradient({"member_indices": [0], "token_ids": candidate})
    assert gradient["gradient_sidecar"]["shape"] == [1, 2, 5]
    assert prepared["seed_token_ids"] == seed_token_ids
    for pair in session.layouts:
        for side in (pair.positive, pair.negative):
            assert tokenizer.decode(list(side.fixed_after)) == tail
            assert tokenizer.decode(list(side.fixed_before)).endswith("\n\n")

    exact_limit = max(
        max(
            len(side.fixed_before)
            + len(session.seed_token_ids)
            + len(side.fixed_after)
            + len(side.response_ids),
            len(side.clean_prompt_ids) + len(side.response_ids),
        )
        for pair in session.layouts
        for side in (pair.positive, pair.negative)
    )
    runtime.worker_max_prompt_tokens = exact_limit
    assert _prepare(session, suffix_token_count=2)["member_count"] == 1
    runtime.worker_max_prompt_tokens = exact_limit - 1
    with pytest.raises(ValueError, match=rf"token limit {exact_limit - 1}"):
        _prepare(session, suffix_token_count=2)
    session.close()


def test_prepare_and_operations_fail_closed_on_shape_and_order_mutations(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    session = GCGSession(_runtime(tmp_path))
    with pytest.raises(ValueError, match="exactly"):
        session.prepare(
            {
                "alpha": 1.0,
                "objective": "response_margin",
                "rows": [],
                "separator": "\n\n",
                "suffix_token_count": 1,
                "unknown": True,
            }
        )
    prepared = _prepare(session)
    assert prepared["member_count"] == 1
    with pytest.raises(ValueError, match="exactly"):
        session.score(
            {
                "candidate_eval_device_batch_size": 1,
                "candidate_token_ids": [[ord("?")]],
            }
        )
    with pytest.raises(ValueError, match="unique"):
        session.gradient({"member_indices": [0, 0], "token_ids": [ord("?")]})
    with pytest.raises(ValueError, match="outside"):
        session.score(
            {
                "candidate_eval_device_batch_size": 1,
                "candidate_token_ids": [[ord("?")]],
                "member_indices": [1],
            }
        )
    session.close()


@pytest.mark.parametrize("separator", ["", "\n", "\n\n\n", " \n\n"])
def test_prepare_rejects_noncanonical_separator(tmp_path: Path, separator: str) -> None:
    _write_manifest(tmp_path)
    session = GCGSession(_runtime(tmp_path))
    with pytest.raises(ValueError, match="separator must be exactly"):
        _prepare(session, separator=separator)


@pytest.mark.parametrize("fault", ["model_name", "model_revision", "seed_boundary"])
def test_failed_reprepare_invalidates_prior_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _write_manifest(tmp_path)
    runtime = _runtime(tmp_path)
    session = GCGSession(runtime)
    _prepare(session)
    if fault == "seed_boundary":
        encode = _Tokenizer.__call__

        def merge_boundary(
            tokenizer: _Tokenizer, text: str, **kwargs: object
        ) -> dict[str, list[int]]:
            result = encode(tokenizer, text, **kwargs)
            if "\n\n!" in text:
                tokens = result["input_ids"]
                start = tokens.index(ord("!"))
                tokens[start : start + 2] = [126]
            return result

        monkeypatch.setattr(_Tokenizer, "__call__", merge_boundary)
        expected_error = "requested token span before the tail"
    else:
        setattr(runtime, fault, "different-source")
        expected_error = "runtime model identity differs"

    with pytest.raises(ValueError, match=expected_error):
        _prepare(session)
    assert list(runtime._gcg_sidecar_dir.iterdir()) == []
    with pytest.raises(ValueError, match="gcg_prepare must succeed"):
        session.filter({"candidate_token_ids": [[ord("!")]]})


def test_score_and_gradient_restore_model_state_and_existing_gradients(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    runtime = _runtime(tmp_path)
    model = runtime.benchmark_model.model
    model.train()
    parameters = tuple(model.parameters())
    parameters[0].requires_grad_(False)
    parameters[1].grad = t.full_like(parameters[1], 7.0)
    expected_grad = parameters[1].grad.clone()
    session = GCGSession(runtime)
    _prepare(session)
    request = {
        "candidate_eval_device_batch_size": 1,
        "candidate_token_ids": [[ord("?")]],
        "member_indices": [0],
    }
    session.score(request)
    session.gradient({"member_indices": [0], "token_ids": [ord("?")]})
    assert model.training is True
    assert parameters[0].requires_grad is False
    assert parameters[1].requires_grad is True
    assert t.equal(parameters[1].grad, expected_grad)
    session.close()


@pytest.mark.parametrize("method", ["PromptSteering", "LoRA"])
def test_activation_l2_plain_and_lora_use_scoped_actual_forward(
    tmp_path: Path, method: str
) -> None:
    _write_manifest(tmp_path)
    _write_activation_artifacts(tmp_path)
    session = GCGSession(_runtime(tmp_path, method=method))
    prepared = session.prepare(
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
    calibration = prepared["direction_calibration"]
    assert calibration["max_activation"] == 2.0
    assert calibration["factor"] == session.runtime.factor
    assert calibration["resolved_scale"] == 2.0 * session.runtime.factor
    assert session._activation_model is None
    assert set(session._clean_hidden) == {11, 12}

    score = session.score(
        {
            "candidate_eval_device_batch_size": 2,
            "candidate_token_ids": [[ord("?")]],
            "member_indices": [0],
        }
    )
    assert len(score["losses"]) == 1
    assert math.isfinite(score["losses"][0])
    result = session.gradient({"member_indices": [0], "token_ids": [ord("?")]})
    assert result["member_indices"] == [0]
    path = session.runtime._gcg_sidecar_dir / "gradients.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        gradient = handle.get_tensor("gradients")
    assert gradient.shape == (1, 1, 5)
    assert t.isfinite(gradient).all()
    session.close()
