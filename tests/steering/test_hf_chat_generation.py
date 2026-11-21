"""Real local GPT-NeoX export tests for harmfulness HF generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch as t
import yaml
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GenerationConfig,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

import steering.eval.hf_chat_generation as hf_chat_generation
from scripts.eval import harmfulness_eval
from steering.eval.harmfulness_artifact import (
    HARMFULNESS_EVAL_SCHEMA_VERSION,
    HARMFULNESS_RESPONSES_SCHEMA_VERSION,
)
from steering.eval.harmfulness_generation import absolute_prompt_batch_seed
from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult
from steering.eval.hf_chat_generation import (
    HuggingFaceChatGenerationEngine,
    RepositoryHFCheckpoint,
    load_repository_hf_checkpoint,
)
from steering.sft.hf_contract import HF_SFT_FORMAT_VERSION

_CHAT_TEMPLATE = (
    "{{ bos_token }} "
    "{% for message in messages %}"
    "{{ message['role'] }} {{ message['content'] }} {{ eos_token }} "
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant {% endif %}"
)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<bos>": 0,
        "<eos>": 1,
        "<pad>": 2,
        "<unk>": 3,
        "user": 4,
        "assistant": 5,
        "hello": 6,
        "danger": 7,
        "safe": 8,
        "answer": 9,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = _CHAT_TEMPLATE
    return tokenizer


def _write_tiny_export(
    root: Path,
    *,
    step: int = 3,
    model_vocab_padding: int = 0,
) -> Path:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir()
    tokenizer = _tiny_tokenizer()
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer) + model_vocab_padding,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    t.manual_seed(1234)
    model = GPTNeoXForCausalLM(config)
    t.nn.Module.to(model, dtype=t.bfloat16)
    model.eval()
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    model.generation_config.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    metadata = {
        "format": "huggingface_standard_sft",
        "format_version": HF_SFT_FORMAT_VERSION,
        "resumable": False,
        "completed_optimizer_step": step,
        "backend": "huggingface",
        "family": "pythia",
        "source": {
            "model_id": "EleutherAI/pythia-tiny-test",
            "revision": "a" * 40,
            "architecture": "GPTNeoXForCausalLM",
            "model_type": "gpt_neox",
        },
        "output": {
            "architecture": "GPTNeoXForCausalLM",
            "model_type": "gpt_neox",
        },
        "attention_implementation": "sdpa",
        "precision": {"export": "bfloat16"},
        "batch": {"max_seq_len": 16},
        "chat_template_sha256": hashlib.sha256(_CHAT_TEMPLATE.encode("utf-8")).hexdigest(),
        "vocabulary": {
            "model": len(tokenizer) + model_vocab_padding,
            "tokenizer": len(tokenizer),
        },
        "backend_specific": {"test_export": True},
    }
    (checkpoint_dir / "sft_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


@pytest.fixture()
def tiny_export(tmp_path: Path) -> Path:
    return _write_tiny_export(tmp_path)


@pytest.fixture()
def loaded(tiny_export: Path) -> RepositoryHFCheckpoint:
    return load_repository_hf_checkpoint(tiny_export, device="cpu")


def _direct_generate(
    checkpoint: RepositoryHFCheckpoint,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int | None,
) -> str:
    prompt_ids = cast(
        Any,
        checkpoint.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ),
    )
    if isinstance(prompt_ids, Mapping):
        prompt_ids = prompt_ids["input_ids"]
    assert isinstance(prompt_ids, t.Tensor)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_k": 0 if top_k is None else top_k,
        "top_p": 1.0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "use_cache": True,
        "bos_token_id": checkpoint.model.config.bos_token_id,
        "eos_token_id": [
            checkpoint.model.config.eos_token_id,
            checkpoint.model.config.bos_token_id,
        ],
        "pad_token_id": checkpoint.model.config.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    generation_config = GenerationConfig(**generation_kwargs)

    def generate() -> t.Tensor:
        with t.inference_mode():
            return cast(Any, checkpoint.model).generate(
                input_ids=prompt_ids,
                attention_mask=t.ones_like(prompt_ids),
                generation_config=generation_config,
            )

    if temperature > 0:
        assert seed is not None
        with t.random.fork_rng(devices=[]):
            t.manual_seed(seed)
            generated = generate()
    else:
        generated = generate()
    continuation = generated[0, prompt_ids.shape[1] :].tolist()
    for index, token_id in enumerate(continuation):
        if token_id in {
            checkpoint.model.config.eos_token_id,
            checkpoint.model.config.bos_token_id,
        }:
            continuation = continuation[:index]
            break
    return cast(
        str,
        checkpoint.tokenizer.decode(
            continuation,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )


def test_tiny_gpt_neox_export_loads_with_exact_identity(
    tiny_export: Path,
    loaded: RepositoryHFCheckpoint,
) -> None:
    identity = loaded.model_identity
    assert identity == {
        "format": "repository_hf_sft",
        "backend": "pythia",
        "architecture": "GPTNeoXForCausalLM",
        "model_type": "gpt_neox",
        "step": 3,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
    }
    assert all(len(str(identity[key])) == 64 for key in ("checkpoint_sha256", "tokenizer_sha256"))
    assert {p.dtype for p in loaded.model.parameters() if p.is_floating_point()} == {t.bfloat16}
    logits = loaded.model(t.tensor([[0, 4, 6]], dtype=t.long)).logits
    assert t.isfinite(logits.float()).all()


def test_pythia_style_padded_model_vocabulary_loads(tmp_path: Path) -> None:
    checkpoint = load_repository_hf_checkpoint(
        _write_tiny_export(tmp_path, model_vocab_padding=2),
        device="cpu",
    )

    assert checkpoint.model.config.vocab_size == len(checkpoint.tokenizer) + 2


def test_report_tree_is_excluded_but_other_files_change_identity(
    tiny_export: Path,
) -> None:
    first = load_repository_hf_checkpoint(tiny_export, device="cpu")
    report = tiny_export / "report" / "harmfulness_results"
    report.mkdir(parents=True)
    (report / "unconditional.yaml").write_text("result: 1\n", encoding="utf-8")
    second = load_repository_hf_checkpoint(tiny_export, device="cpu")
    assert second.model_identity == first.model_identity
    (tiny_export / "identity-bearing.txt").write_text("new\n", encoding="utf-8")
    third = load_repository_hf_checkpoint(tiny_export, device="cpu")
    assert third.model_identity["checkpoint_sha256"] != first.model_identity["checkpoint_sha256"]


def test_engine_matches_direct_transformers_greedy_and_sampled_oracles(
    loaded: RepositoryHFCheckpoint,
) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    greedy = engine.generate_batch(
        ["hello"],
        max_new_tokens=4,
        temperature=0.0,
        top_k=None,
        seed=7,
        render_as_chat=True,
    )
    assert greedy == [
        _direct_generate(
            loaded,
            "hello",
            max_new_tokens=4,
            temperature=0.0,
            top_k=None,
            seed=None,
        )
    ]
    sampled = engine.generate_batch(
        ["danger"],
        max_new_tokens=4,
        temperature=0.8,
        top_k=3,
        seed=91,
        render_as_chat=True,
    )
    assert sampled == [
        _direct_generate(
            loaded,
            "danger",
            max_new_tokens=4,
            temperature=0.8,
            top_k=3,
            seed=91,
        )
    ]


def test_sampling_restores_rng_and_is_repeatable(loaded: RepositoryHFCheckpoint) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    before = t.random.get_rng_state().clone()
    first = engine.generate_batch(
        ["hello", "danger"],
        max_new_tokens=3,
        temperature=1.0,
        top_k=None,
        seed=17,
        render_as_chat=True,
    )
    assert t.equal(t.random.get_rng_state(), before)
    second = engine.generate_batch(
        ["hello", "danger"],
        max_new_tokens=3,
        temperature=1.0,
        top_k=None,
        seed=17,
        render_as_chat=True,
    )
    assert second == first
    assert t.equal(t.random.get_rng_state(), before)


def test_batched_sampler_is_repeatable_and_partition_dependent() -> None:
    scores = cast(t.FloatTensor, t.linspace(-2.0, 2.0, 50).reshape(5, 10))

    def sample(batch_size: int) -> list[list[int]]:
        sampled: list[list[int]] = [[] for _ in range(5)]
        for start in range(0, len(scores), batch_size):
            batch_scores = scores[start : start + batch_size]
            processor = hf_chat_generation._BatchedSamplingLogitsProcessor(
                device=t.device("cpu"),
                seed=50 + start,
                temperature=0.9,
                top_k=5,
            )
            input_ids = cast(
                t.LongTensor,
                t.zeros((len(batch_scores), 1), dtype=t.long),
            )
            for _ in range(3):
                selected = processor(
                    cast(t.LongTensor, input_ids),
                    cast(t.FloatTensor, batch_scores),
                ).argmax(dim=-1)
                for row, token_id in enumerate(selected.tolist()):
                    sampled[start + row].append(token_id)
                input_ids = t.cat([input_ids, selected[:, None]], dim=1)
        return sampled

    assert sample(2) == sample(2)
    assert sample(1) != sample(5)


def test_batched_sampler_matches_vectorized_transform_reference() -> None:
    scores = cast(t.FloatTensor, t.linspace(-3.0, 3.0, 120).reshape(4, 30))
    processor = hf_chat_generation._BatchedSamplingLogitsProcessor(
        device=t.device("cpu"),
        seed=91,
        temperature=0.8,
        top_k=11,
    )
    generator = t.Generator().manual_seed(91)
    temperature = TemperatureLogitsWarper(0.8)
    top_k = TopKLogitsWarper(top_k=11, min_tokens_to_keep=1)
    input_ids = cast(t.LongTensor, t.zeros((4, 1), dtype=t.long))

    for _ in range(5):
        warped = temperature(input_ids, scores)
        warped = top_k(input_ids, warped)
        probabilities = t.nn.functional.softmax(warped, dim=-1)
        expected_tokens = t.multinomial(probabilities, 1, generator=generator).squeeze(1)
        actual_tokens = processor(cast(t.LongTensor, input_ids), scores).argmax(dim=-1)
        assert t.equal(actual_tokens, expected_tokens)
        input_ids = t.cat([input_ids, actual_tokens[:, None]], dim=1)


def test_batched_sampler_uses_one_multinomial_per_decode_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = hf_chat_generation._BatchedSamplingLogitsProcessor(
        device=t.device("cpu"),
        seed=17,
        temperature=0.7,
        top_k=5,
    )
    input_ids = cast(t.LongTensor, t.zeros((32, 1), dtype=t.long))
    scores = cast(t.FloatTensor, t.randn((32, 50)))
    original_multinomial = t.multinomial
    multinomial = MagicMock(wraps=original_multinomial)
    monkeypatch.setattr(hf_chat_generation.t, "multinomial", multinomial)

    processor(input_ids, scores)

    multinomial.assert_called_once()
    assert multinomial.call_args.args[0].shape == (32, 50)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_tiny_cpu_sampled_responses_repeat_for_fixed_outer_batch_size(
    loaded: RepositoryHFCheckpoint,
    batch_size: int,
) -> None:
    prompts = ["hello", "danger", "safe", "answer", "hello"]
    engine = HuggingFaceChatGenerationEngine(loaded)
    def generate() -> list[str]:
        responses: list[str] = []
        for start in range(0, len(prompts), batch_size):
            responses.extend(
                engine.generate_batch(
                    prompts[start : start + batch_size],
                    max_new_tokens=2,
                    temperature=0.9,
                    top_k=None,
                    seed=absolute_prompt_batch_seed(50, start),
                    render_as_chat=True,
                )
            )
        return responses

    assert generate() == generate()


def test_multi_prompt_generation_uses_one_left_padded_model_call(
    loaded: RepositoryHFCheckpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    generate = MagicMock()

    def generate_batch_once(*, input_ids: t.Tensor, attention_mask: t.Tensor, **_kwargs: Any):
        assert input_ids.shape[0] == 2
        assert attention_mask.shape == input_ids.shape
        assert int(attention_mask[0].sum()) < int(attention_mask[1].sum())
        assert attention_mask[0, 0].item() == 0
        assert input_ids[0, 0].item() == loaded.model.config.pad_token_id
        continuation = t.full((2, 1), 9, dtype=t.long)
        return t.cat([input_ids, continuation], dim=1)

    generate.side_effect = generate_batch_once
    monkeypatch.setattr(loaded.model, "generate", generate)

    assert engine.generate_batch(
        ["hello", "hello danger"],
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        render_as_chat=True,
    ) == ["answer", "answer"]
    generate.assert_called_once()


def test_zero_tokens_and_generation_shape_guards(
    loaded: RepositoryHFCheckpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    generate = MagicMock(side_effect=AssertionError("generate must not run"))
    monkeypatch.setattr(loaded.model, "generate", generate)
    assert engine.generate_batch(
        ["hello"],
        max_new_tokens=0,
        temperature=0.0,
        top_k=None,
        render_as_chat=True,
    ) == [""]
    generate.assert_not_called()


def test_stop_slicing_preserves_nonterminal_special_tokens(
    loaded: RepositoryHFCheckpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    prompt_ids = engine._render_prompt("hello", prompt_index=0)
    generated = t.cat(
        [
            prompt_ids,
            t.tensor([[loaded.model.config.pad_token_id, 9, loaded.model.config.eos_token_id, 8]]),
        ],
        dim=1,
    )
    monkeypatch.setattr(loaded.model, "generate", MagicMock(return_value=generated))
    response = engine.generate_batch(
        ["hello"],
        max_new_tokens=4,
        temperature=0.0,
        top_k=None,
        render_as_chat=True,
    )[0]
    assert response == loaded.tokenizer.decode(
        [loaded.model.config.pad_token_id, 9],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def test_prefix_corruption_and_context_overflow_fail_loudly(
    loaded: RepositoryHFCheckpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    prompt_ids = engine._render_prompt("hello", prompt_index=0)
    corrupted = t.cat([prompt_ids.clone(), t.tensor([[9]])], dim=1)
    corrupted[0, 0] = 3
    monkeypatch.setattr(loaded.model, "generate", MagicMock(return_value=corrupted))
    with pytest.raises(ValueError, match="prompt prefix"):
        engine.generate_batch(
            ["hello"],
            max_new_tokens=1,
            temperature=0.0,
            top_k=None,
            render_as_chat=True,
        )
    with pytest.raises(ValueError, match="prompt_index=0.*prompt_tokens=.*max_new_tokens=32"):
        engine.generate_batch(
            ["hello"],
            max_new_tokens=32,
            temperature=0.0,
            top_k=None,
            render_as_chat=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"render_as_chat": False},
        {"temperature": 0.0, "top_k": 3},
        {"temperature": 0.8, "seed": None},
        {"temperature": 0.8, "seed": hf_chat_generation.MAX_TORCH_SEED + 1},
        {"temperature": float("nan"), "seed": 1},
        {"max_new_tokens": -1},
    ],
)
def test_generation_contract_rejects_unsupported_values(
    loaded: RepositoryHFCheckpoint,
    kwargs: dict[str, Any],
) -> None:
    engine = HuggingFaceChatGenerationEngine(loaded)
    call = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_k": None,
        "seed": 1,
        "render_as_chat": True,
        **kwargs,
    }
    with pytest.raises(ValueError):
        engine.generate_batch(["hello"], **call)


def test_loader_rejects_pickle_weight_and_step_mismatch(tiny_export: Path) -> None:
    (tiny_export / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="safetensors-only"):
        load_repository_hf_checkpoint(tiny_export, device="cpu")
    (tiny_export / "pytorch_model.bin").unlink()
    metadata_path = tiny_export / "sft_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_optimizer_step"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="directory step"):
        load_repository_hf_checkpoint(tiny_export, device="cpu")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_weights", "no safetensors"),
        ("template_hash", "chat template does not match"),
        ("empty_template", "nonempty saved chat template"),
        ("token_ids", "token identities disagree"),
        ("context", "max_position_embeddings"),
        ("class", "model class/config architectures disagree"),
        ("symlink", "symlink"),
    ],
)
def test_loader_rejects_load_contract_violations(
    tiny_export: Path,
    mutation: str,
    match: str,
) -> None:
    metadata_path = tiny_export / "sft_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "missing_weights":
        for path in tiny_export.glob("model*.safetensors"):
            path.unlink()
    elif mutation == "template_hash":
        metadata["chat_template_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif mutation == "empty_template":
        for path in tiny_export.glob("*chat_template*"):
            path.unlink()
        tokenizer_config_path = tiny_export / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["chat_template"] = ""
        tokenizer_config_path.write_text(json.dumps(tokenizer_config), encoding="utf-8")
    elif mutation == "token_ids":
        generation_path = tiny_export / "generation_config.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        generation["eos_token_id"] = 9
        generation_path.write_text(json.dumps(generation), encoding="utf-8")
    elif mutation == "context":
        config_path = tiny_export / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["max_position_embeddings"] = 0
        config_path.write_text(json.dumps(config), encoding="utf-8")
    elif mutation == "class":
        metadata["output"]["architecture"] = "OtherCausalLM"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif mutation == "symlink":
        (tiny_export / "linked-config.json").symlink_to("config.json")
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        load_repository_hf_checkpoint(tiny_export, device="cpu")


def test_loader_detects_inventory_mutation_during_transformers_load(
    tiny_export: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = hf_chat_generation.AutoModelForCausalLM.from_pretrained

    def mutating_load(*args: Any, **kwargs: Any):
        model = real_load(*args, **kwargs)
        (tiny_export / "changed-during-load.txt").write_text("changed\n", encoding="utf-8")
        return model

    monkeypatch.setattr(
        hf_chat_generation.AutoModelForCausalLM,
        "from_pretrained",
        mutating_load,
    )
    with pytest.raises(ValueError, match="inventory membership changed"):
        load_repository_hf_checkpoint(tiny_export, device="cpu")


def test_real_hf_harmfulness_main_fresh_resume_and_judge_only(
    tiny_export: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [
        {"prompt": "danger", "pool": "harmful", "category": "Violence"},
        {"prompt": "hello", "pool": "benign", "category": "benign"},
    ]
    cfg = OmegaConf.create(
        {
            "source": None,
            "hf_checkpoint_dir": str(tiny_export),
            "model_tag": None,
            "step": None,
            "feature_configs": [],
            "segment_size": None,
            "feature_selection": None,
            "feature_mask": None,
            "next_segment_conditioning_offset": "auto",
            "max_new_tokens": 1,
            "temperature": 0.0,
            "top_k": None,
            "batch_size": 1,
            "repository_hf_batch_size": 2,
            "seed": 42,
            "judge_model": "fake-judge",
            "requests_per_minute": 100,
            "cache_path": None,
            "judge_only": False,
            "num_harmful": 1,
            "num_benign": 1,
            "responses_file": None,
            "output_file": None,
            "conditioning": {
                "context_cache_dir": None,
                "identity": {
                    "model_id": None,
                    "sae_id": None,
                    "layer": None,
                    "pooling": None,
                    "feature_selection_hash": None,
                },
            },
        }
    )
    client = MagicMock()
    client.score_all.return_value = [
        JudgeResult([], 1, "safe"),
        JudgeResult([], 1, "benign"),
    ]
    client.score_compliance_all.return_value = [
        ComplianceJudgeResult(5, False, "helpful"),
    ]
    monkeypatch.setattr(harmfulness_eval, "_compose_cli_config", lambda _argv: cfg)
    monkeypatch.setattr(
        harmfulness_eval,
        "_load_expected_prompt_pairs",
        lambda _cfg: [dict(row) for row in prompts],
    )
    monkeypatch.setattr(harmfulness_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **_kwargs: client)

    harmfulness_eval.main()

    response_path = tiny_export / "report" / "harmfulness_results_responses" / "unconditional.yaml"
    result_path = tiny_export / "report" / "harmfulness_results" / "unconditional.yaml"
    response_artifact = yaml.safe_load(response_path.read_text(encoding="utf-8"))
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert response_artifact["harmfulness_responses_schema_version"] == (
        HARMFULNESS_RESPONSES_SCHEMA_VERSION
    )
    assert result["harmfulness_eval_schema_version"] == HARMFULNESS_EVAL_SCHEMA_VERSION
    assert response_artifact["conditioning"] == {
        "conditioning_mode": "unconditioned",
        "eval_name": "harmfulness_eval",
    }
    generation = response_artifact["generation"]
    assert generation["model"]["format"] == "repository_hf_sft"
    assert generation["model"]["backend"] == "pythia"
    assert generation["feature_conditions"] is None
    assert generation["target_conditioning_mode"] == "none"
    assert generation["batch_size"] == 2
    assert generation["seed_schedule"] == "base_plus_batch_start_hf_batch_v2"
    assert [row["prompt"] for row in response_artifact["responses"]] == [
        "danger",
        "hello",
    ]
    assert result["config"]["generation"] == generation
    assert result["harmful_prompts"]["asr"] == 0.0
    assert result["benign_prompts"]["compliance_rate"] == 1.0

    no_regeneration = MagicMock(side_effect=AssertionError("cache must be reused"))
    monkeypatch.setattr(harmfulness_eval, "_generate_responses", no_regeneration)
    harmfulness_eval.main()
    no_regeneration.assert_not_called()

    cfg.judge_only = True
    no_model_load = MagicMock(side_effect=AssertionError("judge-only must not load weights"))
    monkeypatch.setattr(harmfulness_eval, "load_repository_hf_checkpoint", no_model_load)
    harmfulness_eval.main()
    no_model_load.assert_not_called()
