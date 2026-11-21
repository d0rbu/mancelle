"""Offline contracts and real tiny-architecture oracles for HF ChatCORE."""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch as t
import torch.distributed as dist
import yaml
from hydra import compose, initialize_config_dir
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    GenerationConfig,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    SmolLM3Config,
    SmolLM3ForCausalLM,
)

from scripts.eval import chat_eval
from steering._common.ddp import AlignedRankFailure
from steering.eval import hf_chat_eval
from steering.eval.hf_chat_eval import (
    HFChatEvalRuntime,
    validate_chat_eval_selection,
)
from steering.eval.hf_chat_generation import RepositoryHFCheckpoint
from steering.sft.pythia import PythiaModelId
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.spellingbee import SpellingBee

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/eval"
TEMPLATE = (
    "{% if enable_thinking is defined %}"
    "{{ strftime_now('%d %B %Y') }} {{ 'think' if enable_thinking else 'no_think' }}\n"
    "{% endif %}"
    "{% for message in messages %}{{ message.role }}\n{{ message.content }}<eos>\n{% endfor %}"
    "{% if add_generation_prompt %}assistant\n{% endif %}"
)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({"<eos>": 256, "<pad>": 257, "<bos>": 258, "<unk>": 259})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = TEMPLATE
    return tokenizer


def tiny_runtime(family: str = "qwen2", *, pad_is_eos: bool = False) -> HFChatEvalRuntime:
    config_class, model_class = (
        (Qwen2Config, Qwen2ForCausalLM)
        if family == "qwen2"
        else (SmolLM3Config, SmolLM3ForCausalLM)
    )
    config = config_class(
        vocab_size=264,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=258,
        eos_token_id=256,
        pad_token_id=257,
    )
    config._attn_implementation = "sdpa"
    with t.random.fork_rng(devices=[]):
        t.manual_seed(1234)
        model = model_class(config)
    return HFChatEvalRuntime(
        model,
        tiny_tokenizer(),
        (256, 258),
        256 if pad_is_eos else 257,
        "2026-09-10" if family == "smollm3" else None,
        {},
        io.StringIO(),
    )


def conversation(text: str = "Choose A or B.", answer: Any = "A") -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": answer}],
        "letters": ["A", "B"],
    }


def hf_config(tmp_path: Path, preset: str = "chat_eval_qwen2_5_3b_instruct"):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=preset, overrides=[f"output_dir={tmp_path / 'run'}"])


@pytest.fixture(params=["qwen2", "smollm3"])
def runtime(request) -> HFChatEvalRuntime:
    return tiny_runtime(request.param)


@pytest.mark.parametrize("preset", ["chat_eval_qwen2_5_3b_instruct", "chat_eval_smollm3_3b"])
def test_real_hydra_presets(tmp_path, preset):
    cfg = hf_config(tmp_path, preset)
    assert validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)
    assert len(cfg.hf_revision) == 40
    assert cfg.source is None
    assert cfg.dtype == "bfloat16"
    assert cfg.temperature == 0
    assert cfg.max_new_tokens == 512


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("source", "sft", "exactly one"),
        ("hf_path", None, "exactly one"),
        ("hf_path", "Qwen/Qwen2.5-3B", "hf_path"),
        ("hf_path", [], "hf_path"),
        ("hf_revision", "main", "commit SHA"),
        ("hf_revision", "A" * 40, "commit SHA"),
        ("hf_revision", None, "commit SHA"),
        ("model_tag", "d12", "model_tag"),
        ("step", 0, "step"),
        ("feature_configs", ["ignored.yaml"], "feature_configs"),
        ("target_real_configs", [{"name": "bad"}], "target_real_configs"),
        ("next_segment_conditioning_offset", True, "next_segment"),
        ("dtype", "float16", "dtype"),
        ("output_dir", "", "output_dir"),
        ("output_dir", None, "output_dir"),
        ("task_name", "gsm8k", "exact names"),
        ("task_name", "GSM8K,MMLU", "exact names"),
        ("task_name", "GSM8K|GSM8K", "distinct"),
        ("task_name", "", "exact names"),
        ("task_name", True, "pipe-separated"),
        ("batch_size", 0, "positive integer"),
        ("batch_size", True, "positive integer"),
        ("num_samples", -1, "positive integer"),
        ("max_new_tokens", 1.5, "positive integer"),
        ("max_problems", 0, "positive integer"),
        ("top_k", 0, "positive integer"),
        ("temperature", True, "finite"),
        ("temperature", -0.1, "finite"),
        ("temperature", float("nan"), "finite"),
        ("temperature", float("inf"), "finite"),
        ("hf_template_date", "2026-09-10", "only supported"),
    ],
)
def test_config_errors_precede_loading(tmp_path, key, value, message):
    cfg = hf_config(tmp_path)
    cfg[key] = value
    with (
        patch.object(chat_eval, "compute_init") as init,
        patch.object(chat_eval, "load_model") as load,
    ):
        with pytest.raises(ValueError, match=message):
            chat_eval.main.__wrapped__(cfg)
        init.assert_not_called()
        load.assert_not_called()


@pytest.mark.parametrize("value", [None, "2026-2-1", "2026-02-30", 20260910])
def test_smollm3_requires_a_valid_fixed_date(tmp_path, value):
    cfg = hf_config(tmp_path, "chat_eval_smollm3_3b")
    cfg.hf_template_date = value
    with pytest.raises(ValueError):
        validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)


def test_native_selection_and_hf_only_settings(tmp_path):
    cfg = hf_config(tmp_path)
    cfg.source, cfg.hf_path, cfg.hf_revision, cfg.output_dir = "sft", None, None, None
    assert not validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)
    cfg.hf_revision = "a" * 40
    with pytest.raises(ValueError, match="require hf_path"):
        validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)


def test_local_repository_export_selection_is_explicit_and_fail_closed(tmp_path):
    cfg = hf_config(tmp_path)
    export = tmp_path / "step_000003"
    export.mkdir()
    cfg.hf_path = str(export)
    cfg.hf_revision = None

    assert validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)
    cfg.hf_revision = "a" * 40
    with pytest.raises(ValueError, match="hf_revision=null"):
        validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)
    cfg.hf_revision = None
    cfg.dtype = "float32"
    with pytest.raises(ValueError, match="dtype=bfloat16"):
        validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)


def _tiny_pythia_repository_checkpoint(*, hidden_size: int = 2048) -> RepositoryHFCheckpoint:
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=264,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=64,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=1,
            tie_word_embeddings=False,
            use_parallel_residual=True,
        )
    ).bfloat16()
    # Keep the executable model tiny while presenting the exact immutable
    # source-config fields that the already-validated repository loader read.
    model.config.vocab_size = 50304
    model.config.hidden_size = hidden_size
    model.config.intermediate_size = 8192
    model.config.num_hidden_layers = 24
    model.config.num_attention_heads = 16
    model.config.max_position_embeddings = 2048
    model.config.architectures = ["GPTNeoXForCausalLM"]
    return RepositoryHFCheckpoint(
        model=model,
        tokenizer=tiny_tokenizer(),
        model_identity={
            "format": "repository_hf_sft",
            "backend": "pythia",
            "architecture": "GPTNeoXForCausalLM",
            "model_type": "gpt_neox",
            "step": 3,
            "checkpoint_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
        },
        source_model_id=PythiaModelId.PYTHIA_1_4B.value,
        source_revision="c" * 40,
    )


def test_local_pythia_1_4b_export_uses_family_geometry_in_chatcore(tmp_path, monkeypatch):
    cfg = hf_config(tmp_path)
    export = tmp_path / "step_000003"
    export.mkdir()
    cfg.hf_path = str(export)
    cfg.hf_revision = None
    checkpoint = _tiny_pythia_repository_checkpoint()
    monkeypatch.setattr(hf_chat_eval, "load_repository_hf_checkpoint", lambda *a, **kw: checkpoint)

    runtime = hf_chat_eval.load_hf_chat_eval_runtime(cfg, t.device("cpu"), io.StringIO())

    assert runtime.model is checkpoint.model
    assert runtime.identity["source_model_id"] == PythiaModelId.PYTHIA_1_4B.value
    assert runtime.identity["family"] == "pythia"
    assert runtime.eos_token_ids == (0,)
    assert runtime.pad_token_id == 1


def test_local_pythia_1_4b_export_rejects_mismatched_geometry(tmp_path, monkeypatch):
    cfg = hf_config(tmp_path)
    export = tmp_path / "step_000003"
    export.mkdir()
    cfg.hf_path = str(export)
    cfg.hf_revision = None
    checkpoint = _tiny_pythia_repository_checkpoint(hidden_size=2560)
    monkeypatch.setattr(hf_chat_eval, "load_repository_hf_checkpoint", lambda *a, **kw: checkpoint)

    with pytest.raises(ValueError, match="hidden_size"):
        hf_chat_eval.load_hf_chat_eval_runtime(cfg, t.device("cpu"), io.StringIO())


@pytest.mark.parametrize("name,value", [
    ("problem_start", 200), ("segment_mode", "turn_delimited_fixed_tokens"),
    ("extraction_tokenizer_id", "google/gemma-3-4b-pt"), ("stats_path", "stats.safetensors"),
    ("target_real_output_dir", "out"),
])
def test_hf_rejects_native_only_turn_settings(tmp_path, name, value):
    # kills: silently running the full HF suite when a shard or turn setting was requested.
    cfg = hf_config(tmp_path)
    cfg[name] = value
    with pytest.raises(ValueError, match=f"does not support {name}"):
        validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)


def test_existing_output_cannot_overwrite_evidence(tmp_path):
    cfg = hf_config(tmp_path)
    Path(cfg.output_dir).mkdir()
    with pytest.raises(ValueError, match="already exists"):
        validate_chat_eval_selection(cfg, chat_eval.ALL_CHAT_TASKS)


@pytest.mark.parametrize("reference", ["SECRET ANSWER", [{"type": "python", "text": "SECRET"}], ""])
def test_render_matches_direct_template_without_reference_or_mutation(runtime, reference):
    conv = conversation("日本語 café 🐝\nWhitespace  stays.\n", reference)
    original = copy.deepcopy(conv)
    text, ids = runtime.render(conv)
    kwargs = {}
    if runtime.template_date:
        kwargs = {"enable_thinking": False, "strftime_now": lambda _: "10 September 2026"}
    expected = runtime.tokenizer.apply_chat_template(
        conv["messages"][:-1],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **kwargs,
    )
    assert ids == expected
    assert runtime.tokenizer.decode(ids) == text
    assert "SECRET" not in text
    assert text.endswith("assistant\n")
    assert conv == original


def test_multiturn_prefix_is_preserved(runtime):
    conv = conversation()
    conv["messages"][:0] = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    text, _ = runtime.render(conv)
    assert "Previous answer" in text and "Be concise." in text


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{}],
        [{"role": "user"}, {"role": "user"}],
        [{"role": "user", "content": []}, {"role": "assistant", "content": "answer"}],
        [{"role": "tool", "content": "tool"}, {"role": "assistant", "content": "answer"}],
        [None, {"role": "assistant"}],
    ],
)
def test_invalid_prefix_fails(runtime, messages):
    with pytest.raises(ValueError):
        runtime.render({"messages": messages})


def test_clock_is_fixed_locale_independent_and_format_checked():
    rt = tiny_runtime("smollm3")
    assert rt._template_clock("%d %B %Y") == "10 September 2026"
    assert rt.render(conversation()) == rt.render(conversation())
    with pytest.raises(ValueError, match="clock format"):
        rt._template_clock("%Y")


@pytest.mark.parametrize("pad_is_eos", [False, True])
def test_batched_scores_match_full_forward_and_unpadded_reference(runtime, pad_is_eos):
    if pad_is_eos:
        runtime.pad_token_id = runtime.eos_token_ids[0]
    prompts = [runtime.render(conversation(text))[1] for text in ["A?", "A much longer prompt\nB?"]]
    answer_ids = [
        [runtime.tokenizer.encode(label, add_special_tokens=False)[0] for label in ["A", "B"]]
    ] * 2
    actual = t.tensor(runtime.score_choices(prompts, answer_ids))
    width = max(map(len, prompts))
    padded = t.tensor([[runtime.pad_token_id] * (width - len(p)) + p for p in prompts])
    mask = t.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    with t.inference_mode():
        full = runtime.model(
            padded, attention_mask=mask, position_ids=positions, use_cache=False
        ).logits[:, -1]
        direct_last = runtime.model(
            padded, attention_mask=mask, position_ids=positions, use_cache=False, logits_to_keep=1
        ).logits[:, 0]
        solo = t.stack(
            [
                runtime.model(t.tensor([p]), use_cache=False).logits[0, -1, answer_ids[i]]
                for i, p in enumerate(prompts)
            ]
        )
    expected = t.stack([full[i, choices] for i, choices in enumerate(answer_ids)])
    assert t.equal(
        actual, t.stack([direct_last[i, choices] for i, choices in enumerate(answer_ids)])
    )
    t.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    t.testing.assert_close(actual, solo, atol=1e-6, rtol=1e-5)
    assert t.equal(actual.argmax(-1), expected.argmax(-1))
    assert t.equal(actual.argmax(-1), solo.argmax(-1))


@pytest.mark.parametrize(
    ("prompts", "answers"), [([], []), ([[1]], []), ([[1]], [[]]), ([[]], [[1]]), ([[1]], [[999]])]
)
def test_invalid_categorical_inputs(runtime, prompts, answers):
    with pytest.raises(ValueError):
        runtime.score_choices(prompts, answers)


def test_nonfinite_and_wrong_shaped_logits_fail(runtime):
    with patch.object(runtime.model, "forward") as forward:
        forward.return_value.logits = t.full((1, 1, 264), float("nan"))
        with pytest.raises(FloatingPointError):
            runtime.score_choices([[1]], [[2]])
        forward.return_value.logits = t.zeros(1, 2, 264)
        with pytest.raises(ValueError, match="shape"):
            runtime.score_choices([[1]], [[2]])


@pytest.mark.parametrize(
    ("temperature", "num_samples", "top_k"),
    [(0.0, 1, 50), (0.0, 3, None), (0.8, 3, 12), (0.8, 2, None)],
)
def test_generation_exact_direct_transformers_oracle_and_rng_restore(
    runtime, temperature, num_samples, top_k
):
    _, prompt = runtime.render(conversation("Hello"))
    state = t.get_rng_state().clone()
    actual = runtime.generate(
        prompt, num_samples=num_samples, max_new_tokens=5, temperature=temperature, top_k=top_k
    )
    assert t.equal(t.get_rng_state(), state)
    sampling = temperature > 0
    config = GenerationConfig(
        max_new_tokens=5,
        do_sample=sampling,
        num_beams=1,
        num_return_sequences=num_samples if sampling else 1,
        use_cache=True,
        eos_token_id=list(runtime.eos_token_ids),
        pad_token_id=runtime.pad_token_id,
        repetition_penalty=1.0,
        **({"temperature": temperature, "top_k": top_k or 0, "top_p": 1.0} if sampling else {}),
    )
    with t.random.fork_rng(devices=[]), t.inference_mode():
        t.manual_seed(42)
        generated = runtime.model.generate(
            input_ids=t.tensor([prompt]),
            attention_mask=t.ones(1, len(prompt), dtype=t.long),
            generation_config=config,
        )
    expected = []
    for row in generated[:, len(prompt) :].tolist():
        end = next((i for i, token in enumerate(row) if token in runtime.eos_token_ids), len(row))
        expected.append(row[:end])
    if not sampling:
        expected *= num_samples
    assert [item["token_ids"] for item in actual] == expected
    assert actual == runtime.generate(
        prompt, num_samples=num_samples, max_new_tokens=5, temperature=temperature, top_k=top_k
    )
    assert t.equal(t.get_rng_state(), state)


def test_saved_generation_defaults_are_neutralized(runtime):
    runtime.model.generation_config = GenerationConfig(
        do_sample=True,
        repetition_penalty=1.05,
        top_p=0.8,
        temperature=0.7,
        suppress_tokens=[1, 2],
        min_new_tokens=123,
    )
    runtime.__post_init__()
    config, _ = runtime.model._prepare_generation_config(
        GenerationConfig(max_new_tokens=4, do_sample=False)
    )
    assert config.repetition_penalty == 1.0
    assert config.suppress_tokens is None
    assert not config.do_sample
    assert config.min_new_tokens in (None, 0)


def test_incremental_cache_matches_fresh_full_prefix_forward(runtime):
    sequence = t.tensor([[2, 8, 14]])
    with t.inference_mode():
        cached = runtime.model(sequence, use_cache=True, logits_to_keep=1)
        for _ in range(5):
            fresh = runtime.model(sequence, use_cache=False, logits_to_keep=1)
            assert cached.past_key_values.get_seq_length() == sequence.shape[1]
            t.testing.assert_close(cached.logits, fresh.logits, atol=1e-6, rtol=1e-5)
            assert t.equal(cached.logits.argmax(-1), fresh.logits.argmax(-1))
            next_token = cached.logits[:, -1].argmax(-1, keepdim=True)
            sequence = t.cat([sequence, next_token], dim=1)
            cached = runtime.model(
                next_token,
                attention_mask=t.ones_like(sequence),
                past_key_values=cached.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )


@pytest.mark.parametrize("stop_id", [256, 258])
def test_stop_slicing_preserves_nonterminal_specials_and_whitespace(runtime, stop_id):
    prompt = [1, 2]
    tokens = runtime.tokenizer.encode(" A\n  B ", add_special_tokens=False) + [runtime.pad_token_id]
    generated = t.tensor([prompt + tokens + [stop_id, 7, 8]])
    with patch.object(runtime.model, "generate", return_value=generated):
        result = runtime.generate(
            prompt, num_samples=1, max_new_tokens=30, temperature=0, top_k=50
        )[0]
    assert result["token_ids"] == tokens
    assert result["text"] == " A\n  B <pad>"
    assert result["stop_token_id"] == stop_id
    assert not result["length_limited"]


def test_immediate_eos_and_generation_failure_restore_rng(runtime):
    state = t.get_rng_state().clone()
    with patch.object(runtime.model, "generate", return_value=t.tensor([[1, 256]])):
        assert (
            runtime.generate([1], num_samples=1, max_new_tokens=2, temperature=0, top_k=None)[0][
                "text"
            ]
            == ""
        )
    with patch.object(runtime.model, "generate", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            runtime.generate([1], num_samples=1, max_new_tokens=2, temperature=0.7, top_k=5)
    assert t.equal(state, t.get_rng_state())


@pytest.mark.parametrize(
    "generated",
    [None, t.tensor([1, 2]), t.tensor([[9, 2]]), t.tensor([[1]]), t.tensor([[1, 2], [1, 2]])],
)
def test_generated_prefix_and_shape_are_checked(runtime, generated):
    with patch.object(runtime.model, "generate", return_value=generated):
        with pytest.raises(ValueError, match="prefix"):
            runtime.generate([1, 2], num_samples=1, max_new_tokens=3, temperature=0, top_k=None)


def test_architecture_context_limit_is_authoritative(runtime):
    runtime.model.config.max_position_embeddings = 3
    runtime.tokenizer.model_max_length = 100000
    runtime._check_context([1, 2, 3])
    runtime._check_context([1, 2], 1)
    with pytest.raises(ValueError, match="architecture_limit=3"):
        runtime.generate([1, 2], num_samples=1, max_new_tokens=2, temperature=0, top_k=None)
    with pytest.raises(ValueError, match="context overflow"):
        runtime.score_choices([[1, 2, 3, 4]], [[2]])


class FixtureTask:
    def __init__(self, real_task_class=ARC, rows=3):
        self.real_task = object.__new__(real_task_class)
        self.eval_type = self.real_task.eval_type
        self.items = [conversation(f"Choose A or B. Example {i}.", "A") for i in range(rows)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return copy.deepcopy(self.items[index])

    def evaluate(self, conv, response):
        return self.real_task.evaluate(conv, response)


def test_real_categorical_graders_counts_ties_and_evidence(runtime):
    task = FixtureTask(ARC, rows=5)
    with patch.object(
        runtime, "score_choices", side_effect=lambda prompts, answers: [[0.0, 0.0] for _ in prompts]
    ):
        accuracy = runtime.evaluate(
            task,
            "ARC-Easy",
            batch_size=2,
            num_samples=1,
            max_new_tokens=5,
            temperature=0,
            top_k=50,
            max_problems=3,
        )
    assert accuracy == 1.0
    assert runtime.task_counts["ARC-Easy"]["total"] == 3
    rows = [json.loads(line) for line in runtime.records.getvalue().splitlines()]
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert all(row["prediction"] == "A" and row["passed"] for row in rows)
    assert len({row["prompt_sha256"] for row in rows}) == 3


@pytest.mark.parametrize("real_class", [GSM8K, SpellingBee, HumanEval])
def test_real_generative_graders_and_pass_at_k(runtime, real_class):
    task = FixtureTask(real_class, rows=1)
    if real_class is HumanEval:
        task.items = [conversation("def add(a, b):\n    pass", "def add(a, b):\n    return a+b")]
        task.items[0].update(test="def check(fn):\n    assert fn(1, 2) == 3", entry_point="add")
        outputs = ["def add(a, b):\n    return 0", "```python\ndef add(a, b):\n    return a+b\n```"]
    else:
        task.items = [conversation("What is one plus two?", [{"type": "text", "text": "#### 3"}])]
        outputs = ["The answer is 3.", "#### 3"]
    with patch.object(runtime, "generate", return_value=[{"text": text} for text in outputs]):
        accuracy = runtime.evaluate(
            task,
            real_class.__name__,
            batch_size=8,
            num_samples=2,
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            max_problems=None,
        )
    row = json.loads(runtime.records.getvalue())
    assert accuracy == 1.0
    assert row["outcomes"] == [False, True]
    assert runtime.task_counts[real_class.__name__]["passed"] == 1


def test_multi_token_label_and_empty_task_fail(runtime):
    task = FixtureTask()
    task.items[0]["letters"] = ["long answer", "B"]
    with pytest.raises(ValueError, match="one continuation token"):
        runtime.evaluate(
            task,
            "ARC-Easy",
            batch_size=2,
            num_samples=1,
            max_new_tokens=5,
            temperature=0,
            top_k=50,
            max_problems=None,
        )
    with pytest.raises(ValueError, match="no examples"):
        runtime.evaluate(
            FixtureTask(rows=0),
            "ARC-Easy",
            batch_size=2,
            num_samples=1,
            max_new_tokens=5,
            temperature=0,
            top_k=50,
            max_problems=None,
        )


def test_shared_orchestration_all_tasks_and_exact_aggregate(runtime, tmp_path, monkeypatch):
    cfg = hf_config(tmp_path)
    cfg.device_type = "cpu"
    expected_scores = dict(
        zip(chat_eval.ALL_CHAT_TASKS, [0.25, 1.0, 0.5, 0.0, 0.5, 1.0], strict=True)
    )
    factories: dict[str, Any] = {
        name: lambda **kw: object() for name in ["ARC", "MMLU", "GSM8K", "SpellingBee", "HumanEval"]
    }
    monkeypatch.setattr(chat_eval, "load_hf_chat_eval_runtime", lambda *args: runtime)
    for name in ("Engine", "prepare_model_for_chat_eval", "ensure_core_eval_has_work"):
        monkeypatch.setattr(
            chat_eval, name, lambda *a, **kw: pytest.fail("native conditioning path entered")
        )
    with patch.object(
        runtime, "evaluate", side_effect=lambda task, name, **kw: expected_scores[name]
    ):
        with patch.multiple(chat_eval, **factories):
            chat_eval.main.__wrapped__(cfg)
    result = yaml.safe_load((Path(cfg.output_dir) / "unconditional.yaml").read_text())
    # Preserve native sequential addition: Python 3.13 sum uses compensation.
    expected = 0.47222222222222215
    assert result["name"] == "unconditional"
    assert result["results"] == expected_scores
    assert result["chatcore_metric"] == expected


def test_pinned_loader_and_functional_main_use_real_local_weights(tmp_path, monkeypatch):
    original = tiny_runtime()
    export = tmp_path / "model"
    original.model.save_pretrained(export)
    GenerationConfig(
        eos_token_id=[256, 258], pad_token_id=257, do_sample=True, repetition_penalty=1.05
    ).save_pretrained(export)
    original.tokenizer.save_pretrained(export)
    cfg = hf_config(tmp_path)
    cfg.dtype, cfg.task_name, cfg.max_problems = "float32", "ARC-Easy", 2
    calls = []
    for factory in [
        chat_eval.load_hf_chat_eval_runtime.__globals__[name]
        for name in ["AutoConfig", "AutoTokenizer", "AutoModelForCausalLM", "GenerationConfig"]
    ]:
        real = factory.from_pretrained

        def local_load(path, *args, _real=real, **kwargs):
            if path != cfg.hf_path:
                return _real(path, *args, **kwargs)
            calls.append((path, kwargs.copy()))
            assert path == cfg.hf_path
            assert kwargs.pop("revision") == cfg.hf_revision
            assert kwargs["trust_remote_code"] is False
            return _real(export, *args, local_files_only=True, **kwargs)

        monkeypatch.setattr(factory, "from_pretrained", local_load)
    monkeypatch.setattr(chat_eval, "ARC", lambda **kwargs: FixtureTask(rows=2))
    monkeypatch.setattr(chat_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        chat_eval, "load_model", lambda *args, **kw: pytest.fail("native loader called")
    )
    chat_eval.main.__wrapped__(cfg)
    root = Path(cfg.output_dir)
    result = yaml.safe_load((root / "unconditional.yaml").read_text())
    assert result["name"] == "unconditional"
    assert "chatcore_metric" not in result
    assert result["model"]["revision"] == cfg.hf_revision
    assert result["model"]["eos_token_ids"] == [256, 258]
    assert result["model"]["pad_token_id"] == 257
    assert result["task_counts"]["ARC-Easy"]["total"] == 2
    assert len((root / "records.rank0.jsonl").read_text().splitlines()) == 2
    assert len(calls) == 4
    assert not (export / "unconditional.yaml").exists()


def test_inference_failure_leaves_no_completed_result(tmp_path, monkeypatch):
    cfg = hf_config(tmp_path)
    runtime = tiny_runtime()
    monkeypatch.setattr(chat_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(chat_eval, "load_hf_chat_eval_runtime", lambda *args: runtime)
    monkeypatch.setattr(chat_eval, "ARC", lambda **kwargs: FixtureTask(rows=1))
    monkeypatch.setattr(
        runtime, "evaluate", lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("injected"))
    )
    with patch.object(chat_eval, "compute_cleanup") as cleanup:
        with pytest.raises(RuntimeError, match="injected"):
            chat_eval.main.__wrapped__(cfg)
        cleanup.assert_called_once()
    assert not (Path(cfg.output_dir) / "unconditional.yaml").exists()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("model_type", "model_type"),
        ("eos_none", "EOS"),
        ("eos_bounds", "EOS"),
        ("eos_empty", "at least one EOS"),
        ("pad_none", "PAD"),
        ("pad_bool", "PAD"),
        ("tokenizer_vocab", "tokenizer vocabulary"),
        ("template_empty", "saved chat template"),
        ("template_mapping", "saved chat template"),
        ("context_zero", "max_position_embeddings"),
        ("missing_keys", "did not load exactly"),
        ("unexpected_keys", "did not load exactly"),
        ("mismatched_keys", "did not load exactly"),
        ("error_msgs", "did not load exactly"),
        ("architecture", "architecture"),
        ("embedding", "embedding vocabulary"),
        ("dtype", "requested dtype"),
    ],
)
def test_checkpoint_contract_rejects_inconsistent_artifacts(tmp_path, monkeypatch, fault, message):
    original = tiny_runtime()
    cfg = hf_config(tmp_path)
    cfg.dtype = "float32"
    model, tokenizer, config = original.model, original.tokenizer, original.model.config
    config.architectures = [type(model).__name__]
    generation = GenerationConfig(eos_token_id=[256, 258], pad_token_id=257)
    loading_info = {}
    if fault == "model_type":
        cast(Any, config).model_type = "smollm3"
    elif fault == "eos_none":
        generation.eos_token_id = None
    elif fault == "eos_bounds":
        generation.eos_token_id = [config.vocab_size]
    elif fault == "eos_empty":
        generation.eos_token_id = []
    elif fault == "pad_none":
        generation.pad_token_id = None
    elif fault == "pad_bool":
        generation.pad_token_id = True
    elif fault == "tokenizer_vocab":
        config.vocab_size = 259
    elif fault == "template_empty":
        tokenizer.chat_template = "  "
    elif fault == "template_mapping":
        tokenizer.chat_template = {"default": TEMPLATE}
    elif fault == "context_zero":
        config.max_position_embeddings = 0
    elif fault in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}:
        loading_info[fault] = ["injected mismatch"]
    elif fault == "architecture":
        config.architectures = ["GPT2LMHeadModel"]
    elif fault == "embedding":
        config.vocab_size += 1
    elif fault == "dtype":
        model.bfloat16()
    monkeypatch.setattr(hf_chat_eval.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(hf_chat_eval.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(
        hf_chat_eval.GenerationConfig, "from_pretrained", lambda *a, **kw: generation
    )
    monkeypatch.setattr(
        hf_chat_eval.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: (model, loading_info)
    )
    with pytest.raises(ValueError, match=message):
        hf_chat_eval.load_hf_chat_eval_runtime(cfg, t.device("cpu"), io.StringIO())


def test_loader_rejects_unsupported_device_before_io(tmp_path):
    with patch.object(hf_chat_eval.AutoConfig, "from_pretrained") as load:
        with pytest.raises(ValueError, match="CPU and CUDA"):
            hf_chat_eval.load_hf_chat_eval_runtime(
                hf_config(tmp_path), t.device("mps"), io.StringIO()
            )
        load.assert_not_called()


@pytest.mark.parametrize(
    ("device_type", "ddp", "message"),
    [("mps", False, "CPU or CUDA"), ("cpu", True, "requires CUDA")],
)
def test_cli_device_rejection_precedes_process_group_init(tmp_path, device_type, ddp, message):
    cfg = hf_config(tmp_path)
    cfg.device_type = device_type
    with patch.object(chat_eval, "get_dist_info", return_value=(ddp, 0, 0, 2 if ddp else 1)):
        with patch.object(chat_eval, "compute_init") as init:
            with pytest.raises(ValueError, match=message):
                chat_eval.main.__wrapped__(cfg)
            init.assert_not_called()


@pytest.mark.parametrize("invalid", [None, [], [[1]], [True], [999999]])
def test_template_token_output_is_validated(runtime, invalid):
    with patch.object(runtime.tokenizer, "encode", return_value=invalid):
        with pytest.raises(ValueError):
            runtime.render(conversation())


def test_template_must_render_text(runtime):
    with patch.object(runtime.tokenizer, "apply_chat_template", return_value=None):
        with pytest.raises(ValueError, match="render text"):
            runtime.render(conversation())


@pytest.mark.parametrize("failure", ["open", "close", "publish"])
def test_evidence_failure_never_publishes_a_completed_result(tmp_path, monkeypatch, failure):
    cfg = hf_config(tmp_path)
    cfg.task_name = "ARC-Easy"
    runtime = tiny_runtime()
    monkeypatch.setattr(chat_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(chat_eval, "load_hf_chat_eval_runtime", lambda *a: runtime)
    monkeypatch.setattr(chat_eval, "ARC", lambda **kw: FixtureTask(rows=1))
    original_open = Path.open

    class FailingClose(io.StringIO):
        def close(self):
            super().close()
            raise OSError("injected close")

    def open_evidence(path, *args, **kwargs):
        if path.name == "records.rank0.jsonl":
            if failure == "open":
                raise OSError("injected open")
            if failure == "close":
                return FailingClose()
        if path.name == "unconditional.yaml.tmp" and failure == "publish":
            raise OSError("injected publish")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_evidence)
    with pytest.raises(OSError, match="injected"):
        chat_eval.main.__wrapped__(cfg)
    assert not (Path(cfg.output_dir) / "unconditional.yaml").exists()


def test_two_process_gloo_coverage_and_aligned_failures(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(CONFIG_DIR.parent.parent), OMP_NUM_THREADS="1")
    worker_command = [str(Path(__file__).resolve()), str(tmp_path)]
    if os.environ.get("COVERAGE_FILE"):
        # pytest-cov does not automatically instrument torchrun's subprocesses.
        worker_command = [
            "--no-python",
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=steering/eval",
            *worker_command,
        ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            *worker_command,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        evidence = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert evidence == {
            "success_cases": 5,
            "aligned_failures": 5,
            "identity_failure": True,
            "coverage_failure": True,
        }


def _ddp_worker(output_dir: Path) -> None:
    """A real two-process test, including ranks with no assigned examples."""
    runtime = tiny_runtime()
    cases = [(ARC, 5, 2), (ARC, 1, 2), (ARC, 3, 8), (GSM8K, 3, 8), (GSM8K, 1, 8)]
    settings: dict[str, Any] = {
        "num_samples": 1,
        "max_new_tokens": 3,
        "temperature": 0,
        "top_k": 50,
        "max_problems": None,
    }
    expected = []
    for task_class, rows, batch in cases:
        runtime.records = io.StringIO()
        task = FixtureTask(task_class, rows=rows)
        if task_class is GSM8K:
            for conv in task.items:
                conv["messages"][-1]["content"] = [{"type": "text", "text": "#### 3"}]
        accuracy = runtime.evaluate(task, "test", batch_size=batch, **settings)
        expected.append(
            (
                task,
                accuracy,
                json.loads("[" + ",".join(runtime.records.getvalue().splitlines()) + "]"),
            )
        )
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    try:
        for (task, accuracy, records), (_, _, batch) in zip(expected, cases, strict=True):
            runtime.records = io.StringIO()
            assert runtime.evaluate(task, "test", batch_size=batch, **settings) == accuracy
            local = [json.loads(line) for line in runtime.records.getvalue().splitlines()]
            gathered: list[Any] = [None, None]
            dist.all_gather_object(gathered, local)
            combined = sorted(
                [row for part in gathered for row in part], key=lambda row: row["index"]
            )
            assert len(combined) == len(task)
            for actual, reference in zip(combined, records, strict=True):
                # Different categorical batch shapes may differ in float rounding.
                actual.pop("choice_logits", None)
                reference.pop("choice_logits", None)
                assert actual == reference
        for failure in ("metadata", "render", "score_choices", "write", "eval_type"):
            task = FixtureTask(rows=3)
            runtime.records = io.StringIO()
            original_render, original_scores = runtime.render, runtime.score_choices
            if rank == 1:
                if failure == "metadata":
                    cast(Any, task).words = {object()}
                elif failure == "eval_type":
                    task.eval_type = "unsupported"
                elif failure == "write":
                    runtime.records.close()
                else:
                    setattr(
                        runtime,
                        failure,
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError("injected")),
                    )
            try:
                with pytest.raises(
                    AlignedRankFailure, match="first failing rank 1|failed on rank 1"
                ):
                    runtime.evaluate(task, "injected", batch_size=1, **settings)
            finally:
                cast(Any, runtime).render = original_render
                cast(Any, runtime).score_choices = original_scores
        task = FixtureTask(rows=3)
        cast(Any, task).words = [str(rank)]
        runtime.records = io.StringIO()
        with pytest.raises(ValueError, match="identity differs"):
            runtime.evaluate(task, "identity", batch_size=1, **settings)
        real_all_reduce = dist.all_reduce

        def corrupt_counts(tensor, *args, **kwargs):
            real_all_reduce(tensor, *args, **kwargs)
            if tensor.dtype == t.long and tensor.shape == (2,):
                tensor[1] += 1

        runtime.records = io.StringIO()
        with patch.object(dist, "all_reduce", side_effect=corrupt_counts):
            with pytest.raises(RuntimeError, match="coverage mismatch"):
                runtime.evaluate(FixtureTask(rows=3), "counts", batch_size=1, **settings)
        (output_dir / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "success_cases": len(cases),
                    "aligned_failures": 5,
                    "identity_failure": True,
                    "coverage_failure": True,
                }
            )
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _ddp_worker(Path(sys.argv[1]))
