"""Real public 3B checkpoints and benchmark rows; requires a CUDA GPU and HF cache."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
import yaml
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer, GenerationConfig

from steering.eval.hf_chat_eval import load_hf_chat_eval_runtime
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

pytestmark = pytest.mark.slow
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/eval"


@pytest.fixture
def full_precision_fp32():
    previous = t.get_float32_matmul_precision()
    t.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        t.set_float32_matmul_precision(previous)


@pytest.mark.usefixtures("full_precision_fp32")
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize(
    ("preset", "architecture", "kv_heads", "vocab", "eos", "pad"),
    [
        ("chat_eval_qwen2_5_3b_instruct", "Qwen2ForCausalLM", 2, 151936, (151645, 151643), 151643),
        ("chat_eval_smollm3_3b", "SmolLM3ForCausalLM", 4, 128256, (128012,), 128004),
    ],
)
def test_public_3b_models_against_direct_transformers(
    tmp_path, preset, architecture, kv_heads, vocab, eos, pad, dtype
):
    if not t.cuda.is_available():
        pytest.skip("real 3B checkpoint validation requires CUDA")
    started = time.monotonic()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=preset, overrides=[f"output_dir={tmp_path / 'run'}"])
    cfg.dtype = dtype
    runtime = load_hf_chat_eval_runtime(cfg, t.device("cuda:0"), io.StringIO())
    model = runtime.model
    assert type(model).__name__ == architecture
    assert model.config.num_hidden_layers == 36
    assert model.config.hidden_size == 2048
    assert model.config.num_attention_heads == 16
    assert model.config.num_key_value_heads == kv_heads
    assert model.config.vocab_size == vocab
    assert runtime.identity["parameter_count"] == (3_085_938_688 if kv_heads == 2 else 3_075_098_624)
    assert runtime.eos_token_ids == eos
    assert runtime.pad_token_id == pad
    assert not model.training
    expected_dtype = t.bfloat16 if dtype == "bfloat16" else t.float32
    assert all(p.dtype == expected_dtype and p.device.type == "cuda" for p in model.parameters())
    assert runtime.tokenizer.bos_token_id is None
    max_logit_drift = 0.0
    max_cache_logit_drift = 0.0
    max_cache_kl = 0.0
    task_rows = {}
    with t.inference_mode():
        for name, task in [
            ("ARC-Easy", ARC(subset="ARC-Easy", split="test")),
            ("ARC-Challenge", ARC(subset="ARC-Challenge", split="test")),
            ("MMLU", MMLU(subset="all", split="test")),
        ]:
            conversations = [task[i] for i in range(3)]
            prompts = [runtime.render(conv)[1] for conv in conversations]
            choices = [
                [
                    runtime.tokenizer.encode(label, add_special_tokens=False)[0]
                    for label in c["letters"]
                ]
                for c in conversations
            ]
            scores = runtime.score_choices(prompts, choices)
            ids, mask, positions = runtime._batch(prompts)
            direct = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                logits_to_keep=1,
            ).logits
            for i, (prompt, labels) in enumerate(zip(prompts, choices, strict=True)):
                assert scores[i] == direct[i, 0, labels].float().tolist()
                unpadded = t.tensor([prompt], device=model.device)
                full = model(input_ids=unpadded, use_cache=False).logits[0, -1, labels].float()
                drift = (full - t.tensor(scores[i], device=model.device)).abs().max().item()
                max_logit_drift = max(max_logit_drift, drift)
                # BF16 GEMM shapes can round differently. Decisions must agree.
                assert int(full.argmax()) == max(range(len(labels)), key=scores[i].__getitem__)
                t.testing.assert_close(
                    full.cpu(),
                    t.tensor(scores[i]),
                    atol=0.25 if dtype == "bfloat16" else 1e-4,
                    rtol=0.02 if dtype == "bfloat16" else 1e-5,
                )
            task_rows[name] = len(task)
        for name, task in [
            ("GSM8K", GSM8K(subset="main", split="test")),
            ("HumanEval", HumanEval()),
            ("SpellingBee", SpellingBee(size=256, split="test")),
        ]:
            conv = task[0]
            text, prompt = runtime.render(conv)
            kwargs: dict[str, Any] = (
                {}
                if kv_heads == 2
                else {"enable_thinking": False, "strftime_now": lambda _: "10 September 2026"}
            )
            assert text == runtime.tokenizer.apply_chat_template(
                conv["messages"][:-1], tokenize=False, add_generation_prompt=True, **kwargs
            )
            if kv_heads == 4:
                assert text.endswith("<think>\n\n</think>\n")
                assert "10 September 2026" in text
            ids = t.tensor([prompt], device=model.device)
            mask = t.ones_like(ids)
            sequence = ids
            cached = model(sequence, use_cache=True, logits_to_keep=1)
            for _ in range(4):
                fresh = model(sequence, use_cache=False, logits_to_keep=1)
                assert cached.past_key_values.get_seq_length() == sequence.shape[1]
                drift = (cached.logits.float() - fresh.logits.float()).abs().max().item()
                max_cache_logit_drift = max(max_cache_logit_drift, drift)
                cached_log_probs = cached.logits.float().log_softmax(-1)
                fresh_log_probs = fresh.logits.float().log_softmax(-1)
                kl = (cached_log_probs.exp() * (cached_log_probs - fresh_log_probs)).sum().item()
                max_cache_kl = max(max_cache_kl, kl)
                if dtype == "bfloat16":
                    # The FP32 counterpart checks cache positions without BF16
                    # accumulation across matrix shapes.
                    # Bound the whole distribution and require exact decisions.
                    assert drift <= 1.0
                    assert abs(kl) < 0.01
                else:
                    t.testing.assert_close(cached.logits, fresh.logits, atol=1e-4, rtol=1e-5)
                assert t.equal(cached.logits.argmax(-1), fresh.logits.argmax(-1))
                next_token = cached.logits[:, -1].argmax(-1, keepdim=True)
                sequence = t.cat([sequence, next_token], dim=1)
                cached = model(
                    next_token,
                    attention_mask=t.ones_like(sequence),
                    past_key_values=cached.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            for temperature, samples in [(0.0, 2), (0.7, 2)]:
                cpu_rng, gpu_rng = t.get_rng_state(), t.cuda.get_rng_state()
                actual = runtime.generate(
                    prompt,
                    num_samples=samples,
                    max_new_tokens=32,
                    temperature=temperature,
                    top_k=50,
                )
                assert t.equal(cpu_rng, t.get_rng_state())
                assert t.equal(gpu_rng, t.cuda.get_rng_state())
                settings = {
                    "max_new_tokens": 32,
                    "do_sample": temperature > 0,
                    "num_beams": 1,
                    "num_return_sequences": samples if temperature else 1,
                    "use_cache": True,
                    "eos_token_id": list(eos),
                    "pad_token_id": pad,
                    "repetition_penalty": 1.0,
                }
                if temperature:
                    settings.update(temperature=temperature, top_k=50, top_p=1.0)
                with t.random.fork_rng(devices=[0]):
                    t.random.default_generator.manual_seed(42)
                    t.cuda.manual_seed(42)
                    expected = (
                        cast(Any, model)
                        .generate(
                            input_ids=ids,
                            attention_mask=mask,
                            generation_config=GenerationConfig(**settings),
                        )[:, len(prompt) :]
                        .tolist()
                    )
                if not temperature:
                    expected *= samples
                for row, output in zip(expected, actual, strict=True):
                    stop = next((i for i, token in enumerate(row) if token in eos), len(row))
                    assert output["token_ids"] == row[:stop]
                    assert output["text"] == runtime.tokenizer.decode(
                        row[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    assert output["stop_token_id"] == (row[stop] if stop < len(row) else None)
                    assert bool(task.evaluate(conv, output["text"])) == bool(
                        task.evaluate(
                            conv,
                            runtime.tokenizer.decode(
                                row[:stop],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                        )
                    )
            task_rows[name] = len(task)
    evidence = {
        "model": runtime.identity,
        "rows": task_rows,
        "max_logit_drift": max_logit_drift,
        "max_cache_logit_drift": max_cache_logit_drift,
        "max_cache_kl": max_cache_kl,
        "elapsed_seconds": time.monotonic() - started,
        "process_peak_allocated_bytes": t.cuda.max_memory_allocated(),
    }
    (tmp_path / "real_model_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, sort_keys=True))


@pytest.mark.parametrize("preset", ["chat_eval_qwen2_5_3b_instruct", "chat_eval_smollm3_3b"])
def test_public_3b_nccl_cli_matches_single_gpu(tmp_path, preset):
    if t.cuda.device_count() < 2:
        pytest.skip("NCCL parity requires two distinct CUDA devices")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for world in (1, 2):
        output = tmp_path / f"world{world}"
        command = [sys.executable]
        if world == 2:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command += [
            "-m",
            "scripts.eval.chat_eval",
            f"--config-name={preset}",
            f"output_dir={output}",
            "batch_size=2",
            "max_problems=3",
            "max_new_tokens=32",
        ]
        result = subprocess.run(
            command, env=env, check=False, capture_output=True, text=True, timeout=240
        )
        (tmp_path / f"world{world}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(output.glob("records.rank*.jsonl"))) == world
        assert len(list(output.glob("unconditional.yaml"))) == 1
        assert not list(output.glob("*.tmp"))
    single = yaml.safe_load((tmp_path / "world1/unconditional.yaml").read_text())
    distributed = yaml.safe_load((tmp_path / "world2/unconditional.yaml").read_text())
    assert single["model"] == distributed["model"]
    assert single["results"] == distributed["results"]
    assert single["chatcore_metric"] == distributed["chatcore_metric"]
    assert single["task_counts"] == distributed["task_counts"]
    assert all(counts["total"] == 3 for counts in single["task_counts"].values())
    records = []
    for world in (1, 2):
        rows = []
        for path in (tmp_path / f"world{world}").glob("records.rank*.jsonl"):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        assert len(rows) == 18
        assert len({(row["task"], row["index"]) for row in rows}) == 18
        for row in rows:
            row.pop("choice_logits", None)
        records.append(sorted(rows, key=lambda row: (row["task"], row["index"])))
    assert records[0] == records[1]


@pytest.mark.very_slow
def test_complete_run_evidence_matches_every_real_dataset_row():
    """Re-render and re-grade a completed full run, independent of the runtime."""
    run_dir = os.environ.get("CHATCORE_FULL_RUN_DIR")
    if run_dir is None:
        pytest.skip("set CHATCORE_FULL_RUN_DIR to a completed full 3B evaluation")
    root = Path(run_dir)
    result = yaml.safe_load((root / "unconditional.yaml").read_text())
    config, identity = result["evaluation_config"], result["model"]
    assert config["max_problems"] is None
    tokenizer = AutoTokenizer.from_pretrained(
        identity["model_id"], revision=identity["revision"], trust_remote_code=False
    )
    assert tokenizer is not None
    assert isinstance(tokenizer.chat_template, str)
    assert (
        identity["chat_template_sha256"]
        == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    )
    kwargs: dict[str, Any] = {}
    if identity["enable_thinking"] is False:
        # Test the checked-in SmolLM3 preset's fixed date independently.
        assert identity["template_date"] == "2026-09-10"
        kwargs = {"enable_thinking": False, "strftime_now": lambda _: "10 September 2026"}
    rows = {}
    for name in result["evidence_files"]:
        for line in (root / name).read_text().splitlines():
            record = json.loads(line)
            key = record["task"], record["index"]
            assert key not in rows, f"duplicate evidence: {key}"
            rows[key] = record
    expected_total = 0
    for name, task in [
        ("ARC-Easy", ARC(subset="ARC-Easy", split="test")),
        ("ARC-Challenge", ARC(subset="ARC-Challenge", split="test")),
        ("MMLU", MMLU(subset="all", split="test")),
        ("GSM8K", GSM8K(subset="main", split="test")),
        ("HumanEval", HumanEval()),
        ("SpellingBee", SpellingBee(size=256, split="test")),
    ]:
        expected_total += len(task)
        counts = result["task_counts"][name]
        assert counts["total"] == counts["rows"] == len(task)
        assert counts["dataset_fingerprint"] == getattr(
            getattr(task, "ds", None), "_fingerprint", None
        )
        if hasattr(task, "words"):
            assert (
                counts["word_list_sha256"]
                == hashlib.sha256(
                    json.dumps(
                        task.words, ensure_ascii=False, sort_keys=True, allow_nan=False
                    ).encode()
                ).hexdigest()
            )
        passed = 0
        for index in range(len(task)):
            conv = task[index]
            record = rows[name, index]
            expected_prompt = tokenizer.apply_chat_template(
                conv["messages"][:-1],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **kwargs,
            )
            assert record["prompt_token_ids"] == expected_prompt, (name, index)
            assert len(expected_prompt) <= identity["max_position_embeddings"]
            for value, digest in [
                (conv, record["conversation_sha256"]),
                (expected_prompt, record["prompt_sha256"]),
            ]:
                assert (
                    hashlib.sha256(
                        json.dumps(
                            value, ensure_ascii=False, sort_keys=True, allow_nan=False
                        ).encode()
                    ).hexdigest()
                    == digest
                )
            if task.eval_type == "categorical":
                assert record["choices"] == list(conv["letters"])
                logits = record["choice_logits"]
                assert len(logits) == len(conv["letters"])
                assert all(t.isfinite(t.tensor(logits)))
                winner = max(range(len(logits)), key=logits.__getitem__)
                assert record["prediction"] == conv["letters"][winner]
                outcome = bool(task.evaluate(conv, record["prediction"]))
            else:
                assert len(record["completions"]) == config["num_samples"]
                outcomes = []
                for completion in record["completions"]:
                    ids = completion["token_ids"]
                    assert all(
                        type(token) is int and 0 <= token < identity["model_vocab_size"]
                        for token in ids
                    )
                    assert not set(ids).intersection(identity["eos_token_ids"])
                    assert completion["text"] == tokenizer.decode(
                        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    if completion["length_limited"]:
                        assert len(ids) == config["max_new_tokens"]
                        assert completion["stop_token_id"] is None
                    else:
                        assert completion["stop_token_id"] in identity["eos_token_ids"]
                        assert len(ids) < config["max_new_tokens"]
                    outcomes.append(bool(task.evaluate(conv, completion["text"])))
                assert record["outcomes"] == outcomes, (name, index)
                outcome = any(outcomes)
            assert record["passed"] is outcome, (name, index)
            passed += outcome
        assert counts["passed"] == passed
        assert result["results"][name] == passed / len(task)
    assert len(rows) == expected_total
    centered = 0.0
    for name, accuracy in result["results"].items():
        baseline = 0.25 if name in {"ARC-Easy", "ARC-Challenge", "MMLU"} else 0.0
        centered += (accuracy - baseline) / (1.0 - baseline)
    assert result["chatcore_metric"] == centered / 6
