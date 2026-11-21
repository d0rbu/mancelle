"""Native SFT prompt policy, provenance, context, and crash/resume contracts."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest
import tiktoken
import torch as t
from omegaconf import OmegaConf

from nanochat import flash_attention
from nanochat.gpt import GPT, GPTConfig
from nanochat.sampling import sample_next_token
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.conditional.encoder_decoder import sm12_fa4
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.native_decoder import validate_native_source
from steering.eval.axbench_upstream import native_decoder as native
from steering.eval.axbench_upstream.native_contract import (
    METHOD,
    frozen_user_prompt,
    load_native_output,
    native_input_frame,
    sha256,
)
from steering.immutable_io import ImmutableArtifactError


@pytest.mark.parametrize("capability,override,expected", [
    ((12, 0), None, "native_fa4"),
    ((12, 0), flash_attention.FlashAttnImpl.FA4, "native_fa4"),
    ((12, 0), flash_attention.FlashAttnImpl.SDPA, "sdpa"),
    ((9, 0), None, "flex_fa4"),
    ((8, 9), None, "flex_triton"),
])
def test_fit_identity_admits_full_prefix_and_distinguishes_cached_backend(
    monkeypatch, capability, override, expected,
):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: capability)
    monkeypatch.setattr(sm12_fa4, "_native_fa4", object())
    monkeypatch.setattr(flash_attention, "_override_impl", override)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    monkeypatch.setattr(flash_attention, "HAS_FLEX_ATTENTION", True)
    monkeypatch.setattr(flash_attention, "HAS_FA4_BACKEND", capability[0] >= 9)
    monkeypatch.setattr(flash_attention.flash_attn, "flash_attn_func", flash_attention.flash_attn_func)
    cached = flash_attention.flash_attn.flash_attn_with_kvcache
    settings = native.native_runtime_settings(full_prefix_device=t.device("cuda"))
    assert settings["full_prefix_attention_backend"] == expected
    assert settings["attention_backend"] == "sdpa"
    assert flash_attention.flash_attn.flash_attn_with_kvcache is cached
    assert "full_prefix_attention_backend" not in native.native_runtime_settings()


def test_full_prefix_admission_requires_native_fa4(monkeypatch):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    monkeypatch.setattr(flash_attention.flash_attn, "flash_attn_func", flash_attention.flash_attn_func)
    monkeypatch.setattr(sm12_fa4, "_native_fa4", None)
    with pytest.raises(SFTGRPOError, match="requires native FA4"):
        native.native_runtime_settings(full_prefix_device=t.device("cuda:0"))
    monkeypatch.setattr(sm12_fa4, "_native_fa4", object())
    native.native_runtime_settings(full_prefix_device=t.device("cuda:0"))
    assert sm12_fa4.uses_native_sm12_fa4()


def test_native_loader_admits_selected_device_before_build(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "checkpoint_identity", lambda *_args: {})
    monkeypatch.setattr(native, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        t.cuda,
        "get_device_capability",
        lambda device=None: (12, 0) if device == t.device("cuda:1") else (9, 0),
    )
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    monkeypatch.setattr(
        flash_attention.flash_attn, "flash_attn_func", flash_attention.flash_attn_func
    )
    monkeypatch.setattr(sm12_fa4, "_native_fa4", None)
    build = Mock(side_effect=RuntimeError("admitted before loading"))
    monkeypatch.setattr(native, "build_model", build)
    tokenizer = tmp_path / "tokenizer" / "tokenizer.pkl"
    with pytest.raises(SFTGRPOError, match="requires native FA4"):
        native.NativeDecoderAdapter(tmp_path, 1003, "cuda:1", tokenizer)
    build.assert_not_called()
    monkeypatch.setattr(sm12_fa4, "_native_fa4", object())
    with pytest.raises(RuntimeError, match="admitted before loading"):
        native.NativeDecoderAdapter(tmp_path, 1003, "cuda:1", tokenizer)
    assert sm12_fa4.uses_native_sm12_fa4(t.device("cuda:1"))
    build.assert_called_once()


def test_full_prefix_admits_selected_cuda_device(monkeypatch):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    seen = []

    def capability(device=None):
        seen.append(device)
        return (12, 0) if device == t.device("cuda:1") else (9, 0)

    monkeypatch.setattr(t.cuda, "get_device_capability", capability)
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    monkeypatch.setattr(sm12_fa4, "_native_fa4", None)
    with pytest.raises(SFTGRPOError, match="requires native FA4"):
        native.native_runtime_settings(full_prefix_device=t.device("cuda:1"))
    assert seen == [t.device("cuda:1")]


def test_mixed_cuda_runtime_rejects_imported_fa3_on_wrong_device(monkeypatch):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        t.cuda, "get_device_capability",
        lambda device=None: (12, 0) if device == t.device("cuda:1") else (9, 0),
    )
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(flash_attention, "HAS_FA3", True)
    monkeypatch.setattr(sm12_fa4, "_native_fa4", object())
    monkeypatch.setattr(flash_attention.flash_attn, "flash_attn_func", flash_attention.flash_attn_func)
    with pytest.raises(ValueError, match="imported FA3 cache backend"):
        native.native_runtime_settings(full_prefix_device=t.device("cuda:1"))
    assert flash_attention.HAS_FA3
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    selected = native.native_runtime_settings(full_prefix_device=t.device("cuda:1"))
    assert selected["full_prefix_attention_backend"] == "native_fa4"
    assert selected["attention_backend"] == "sdpa"
    assert native.native_runtime_settings(full_prefix_device=t.device("cuda:0"))[
        "full_prefix_attention_backend"
    ] != "native_fa4"


def test_cached_runtime_rejects_unavailable_explicit_fa3(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.FA3)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    with pytest.raises(ValueError, match="explicit FA3 attention override is unavailable"):
        native.native_runtime_settings(cached_device=t.device("cuda:1"))


def source_frame() -> pd.DataFrame:
    rows = []
    for input_id in range(10):
        prompt = f"Question {input_id}?"
        row: dict[str, object] = dict.fromkeys(INPUT_COLUMNS, "")
        row.update(dataset_name="AlpacaEval", concept_id=0, input_id=input_id,
                   input_concept="rental equipment", factor=1.0, original_prompt=prompt,
                   steered_input="<|im_start|>system\nprovider metadata\n<|im_start|>user\n"
                   + "Always mention rental equipment.\n\nQuestion: " + prompt
                   + "<|im_end|>\n<|im_start|>assistant\n", sae_id=42)
        rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_boosted_policy_preserves_exact_user_and_drops_provider_template():
    source = source_frame()
    frame = native_input_frame(source, max_concepts=1)
    assert frame.native_user_prompt[0] == "Always mention rental equipment.\n\nQuestion: Question 0?"
    assert frame.steering_prefix.nunique() == 1
    assert "provider metadata" not in frame.native_user_prompt[0]
    pd.testing.assert_series_equal(frame.original_prompt, source.original_prompt)


@pytest.mark.parametrize("mutation", ["duplicate", "factor", "missing", "extra", "concept_id", "order", "concept", "prefix", "question",
                                    "missing_column", "blank_concept", "prompt_type", "blank_prefix", "delimiter"])
def test_source_fails_closed_on_grid_or_prompt_corruption(mutation):
    frame = source_frame()
    if mutation == "duplicate":
        frame.loc[1, "input_id"] = 0
    elif mutation == "factor":
        frame.loc[1, "factor"] = 2.0
    elif mutation == "missing":
        frame = frame.iloc[:-1]
    elif mutation == "extra":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == "concept_id":
        frame.loc[1, "concept_id"] = 1
    elif mutation == "order":
        frame = frame.iloc[::-1]
    elif mutation == "concept":
        frame.loc[1, "input_concept"] = "different concept"
    elif mutation == "prefix":
        frame.loc[1, "steered_input"] = frame.loc[1, "steered_input"].replace("Always", "Never")
    elif mutation == "missing_column":
        frame = frame.drop(columns="original_prompt")
    elif mutation == "blank_concept":
        frame.loc[1, "input_concept"] = ""
    elif mutation == "prompt_type":
        frame.loc[1, "original_prompt"] = None
    elif mutation == "blank_prefix":
        frame["steered_input"] = frame.steered_input.str.replace("Always mention rental equipment.", "", regex=False)
    elif mutation == "delimiter":
        frame.loc[1, "steered_input"] = frame.loc[1, "steered_input"].replace("Always", "<|im_end|>Always")
    else:
        frame.loc[1, "original_prompt"] = "Changed question"
    with pytest.raises(ValueError):
        native_input_frame(frame, max_concepts=1)


def test_role_injection_is_not_parsed_as_one_user():
    text = source_frame().steered_input[0].replace("Always", "<|im_start|>user\nAlways")
    with pytest.raises(ValueError, match="exactly one user"):
        frozen_user_prompt(text, "Question 0?")


@pytest.mark.parametrize("count", [0, True])
def test_source_rejects_invalid_concept_count(count):
    with pytest.raises(ValueError, match="max_concepts"):
        native_input_frame(source_frame(), max_concepts=count)


def test_native_adapter_uses_training_renderer():
    adapter = native.NativeDecoderAdapter.__new__(native.NativeDecoderAdapter)
    adapter.model = SimpleNamespace(config=SimpleNamespace(sequence_len=12))
    adapter.tokenizer = SimpleNamespace(render_conversation=Mock(return_value=([1, 2, 3], [0, 0, 0])),
                                        encode_special=Mock(return_value=99), get_bos_token_id=lambda: 1,
                                        decode=Mock(return_value="generated"))
    stream = Mock(return_value=[7])
    adapter.engine = cast(native.InferenceEngine, SimpleNamespace(generate_stream=stream))
    assert adapter.generate_user_text("exact user", seed=42, max_new_tokens=8) == "generated"
    adapter.tokenizer.render_conversation.assert_called_once_with(
        {"messages": [{"role": "user", "content": "exact user"}]}, max_tokens=None)
    assert stream.call_args.args[0] == [1, 2, 3, 99]
    assert stream.call_args.kwargs["seed"] == 42


def test_checkpoint_rejects_base_and_binds_native_model(tmp_path):
    (tmp_path / "model_001003.pt").write_bytes(b"native model")
    tokenizer = tmp_path / "tokenizer.pkl"
    tokenizer.write_bytes(b"trained tokenizer")
    metadata = {"step": 1003, "model_config": {"n_layer": 12},
                "user_config": {"mixture": "standard_steered", "model_backend": "nanochat"}}
    meta = tmp_path / "meta_001003.json"
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SFT checkpoint"):
        native.checkpoint_identity(tmp_path, 1003, tokenizer)
    metadata["sft_loss_mask"] = "completion_only"
    meta.write_text(json.dumps(metadata))
    receipt = native.checkpoint_identity(tmp_path, 1003, tokenizer)
    assert receipt["model_sha256"] == sha256(tmp_path / "model_001003.pt")
    metadata["encoder_config"] = {}
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="decoder-only"):
        native.checkpoint_identity(tmp_path, 1003, tokenizer)
    metadata.pop("encoder_config")
    metadata["user_config"]["model_backend"] = "hf"
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="native SFT"):
        native.checkpoint_identity(tmp_path, 1003, tokenizer)
    with pytest.raises(ValueError, match="step"):
        native.checkpoint_identity(tmp_path, -1, tokenizer)


@pytest.fixture
def producer(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    source_frame().to_parquet(source)
    (tmp_path / "model_001003.pt").write_bytes(b"native model")
    (tmp_path / "tokenizer.pkl").write_bytes(b"trained tokenizer")
    (tmp_path / "meta_001003.json").write_text(json.dumps({
        "step": 1003, "model_config": {"sequence_len": 4096},
        "sft_loss_mask": "completion_only", "user_config": {"mixture": "standard_steered"},
    }))
    checkpoint = native.checkpoint_identity(tmp_path, 1003, tmp_path / "tokenizer.pkl")
    calls = []

    class FakeAdapter:
        identity = checkpoint

        def __init__(self, *args):
            pass

        def generate_user_text(self, text, **kwargs):
            calls.append((text, kwargs["seed"]))
            return f"sample {kwargs['seed']}"

    monkeypatch.setattr(native, "NativeDecoderAdapter", FakeAdapter)
    cfg = OmegaConf.create({"source_path": str(source), "source_sha256": sha256(source),
                           "checkpoint_dir": str(tmp_path), "step": 1003,
                           "tokenizer_path": str(tmp_path / "tokenizer.pkl"),
                           "output_dir": str(tmp_path / "result"), "max_concepts": 1,
                           "device": "cpu", "seed": 42, "max_new_tokens": 128,
                           "temperature": 1.0, "top_k": 50, "max_cells": 2})
    return cfg, calls


def test_pilot_resumes_exact_cells_without_repetition_and_rejects_identity_drift(tmp_path, producer):
    cfg, calls = producer
    assert native.run_native_decoder(cfg)["complete"] is False
    assert len(calls) == 2
    cfg.max_cells = None
    assert native.run_native_decoder(cfg)["complete"] is True
    assert len(calls) == 10
    manifest, output = load_native_output(tmp_path / "result/run_manifest.json")
    assert manifest["output"]["rows"] == 10
    assert output[f"{METHOD}_steered_generation"].tolist() == [f"sample {seed}" for _, seed in calls]
    native.run_native_decoder(cfg)
    assert len(calls) == 10
    cfg.temperature = 0.9
    with pytest.raises(ImmutableArtifactError):
        native.run_native_decoder(cfg)
    relocated = tmp_path / "relocated.parquet"
    relocated.write_bytes((tmp_path / "result/steering_data.parquet").read_bytes())
    load_native_output(tmp_path / "result/run_manifest.json", relocated)
    relocated.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        load_native_output(tmp_path / "result/run_manifest.json", relocated)


def test_output_cannot_replace_frozen_source(tmp_path, producer):
    cfg, calls = producer
    source = tmp_path / "steering_data.parquet"
    source_frame().assign(baseline_generation="keep this").to_parquet(source)
    before = source.read_bytes()
    cfg.source_path = str(source)
    cfg.source_sha256 = sha256(source)
    cfg.output_dir = str(tmp_path)
    cfg.max_cells = None
    with pytest.raises(ValueError, match="disjoint"):
        native.run_native_decoder(cfg)
    assert source.read_bytes() == before
    assert not calls
    assert not (tmp_path / "cells").exists()
    assert not (tmp_path / "identity.json").exists()


@pytest.mark.parametrize("field,value", [("identity_sha256", "wrong"), ("cell", {}), ("generation", None)])
def test_producer_rejects_corrupt_cell_before_generating(tmp_path, producer, field, value):
    cfg, calls = producer
    native.run_native_decoder(cfg)
    path = next((tmp_path / "result/cells").glob("*.json"))
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cell receipt"):
        native.run_native_decoder(cfg)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", ["source_hash", "cell_budget", "loaded_checkpoint"])
def test_producer_rejects_invalid_run_before_generating(producer, monkeypatch, mutation):
    cfg, calls = producer
    if mutation == "source_hash":
        cfg.source_sha256 = "0" * 64
    elif mutation == "cell_budget":
        cfg.max_cells = 0
    else:
        monkeypatch.setattr(native.NativeDecoderAdapter, "identity", {"wrong": True})
    with pytest.raises(ValueError):
        native.run_native_decoder(cfg)
    assert not calls


@pytest.mark.parametrize("stage", ["producer", "loader"])
def test_parquet_changed_during_read_is_rejected(tmp_path, producer, monkeypatch, stage):
    cfg, _ = producer
    cfg.max_cells = None
    native.run_native_decoder(cfg)
    read = pd.read_parquet

    def changed(path, *args, **kwargs):
        frame = read(path, *args, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed")
        return frame

    monkeypatch.setattr(pd, "read_parquet", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        if stage == "producer":
            native.run_native_decoder(cfg)
        else:
            load_native_output(tmp_path / "result/run_manifest.json")


@pytest.mark.parametrize("mutation", ["incomplete", "rows", "generation", "prompt_policy"])
def test_loader_rejects_invalid_completed_output(tmp_path, producer, mutation):
    cfg, _ = producer
    cfg.max_cells = None
    native.run_native_decoder(cfg)
    manifest_path = tmp_path / "result/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = tmp_path / "result/steering_data.parquet"
    if mutation == "incomplete":
        manifest["complete"] = False
    elif mutation == "rows":
        manifest["output"]["rows"] = 9
    else:
        frame = pd.read_parquet(output)
        frame.loc[0, f"{METHOD}_steered_generation" if mutation == "generation" else "native_user_prompt"] = None
        frame.to_parquet(output)
        manifest["output"]["sha256"] = sha256(output)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_native_output(manifest_path)


@pytest.mark.parametrize("field,value", [("max_new_tokens", 0), ("max_new_tokens", True),
                                       ("temperature", 0.0), ("temperature", float("nan")),
                                       ("top_k", 0), ("top_k", True)])
def test_native_adapter_rejects_invalid_sampling_before_model_use(field, value):
    adapter = native.NativeDecoderAdapter.__new__(native.NativeDecoderAdapter)
    with pytest.raises(ValueError):
        adapter.generate_user_text("prompt", seed=42, **{field: value})


@pytest.mark.parametrize("prompt_length", [32, 2500, 5000])
def test_native_context_uses_full_rustbpe_prompt(prompt_length, monkeypatch):
    tokenizer = RustBPETokenizer(tiktoken.Encoding(
        name="native-byte-test", pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
    ), "<bos>")
    adapter = native.NativeDecoderAdapter.__new__(native.NativeDecoderAdapter)
    adapter.model = SimpleNamespace(config=SimpleNamespace(sequence_len=4096))
    adapter.tokenizer = tokenizer
    prompt = "x" * prompt_length + " question suffix"
    if prompt_length == 5000:
        adapter.engine = native.InferenceEngine(adapter.model, tokenizer, t.device("cpu"))
        allocate = Mock(side_effect=AssertionError("overflow must fail before KV-cache allocation"))
        monkeypatch.setattr(adapter.engine, "_make_kv_cache", allocate)
        with pytest.raises(ValueError, match="context window"):
            adapter.generate_user_text(prompt, seed=42, max_new_tokens=128)
        allocate.assert_not_called()
    else:
        stream = Mock(return_value=[ord("y")])
        adapter.engine = cast(native.InferenceEngine, SimpleNamespace(generate_stream=stream))
        assert adapter.generate_user_text(prompt, seed=42, max_new_tokens=128) == "y"
        tokens = stream.call_args.args[0]
        assert tokens == [tokenizer.get_bos_token_id(), tokenizer.encode_special("<user_start>"),
                          *tokenizer.encode(prompt), tokenizer.encode_special("<user_end>"),
                          tokenizer.encode_special("<assistant_start>")]
        assert stream.call_args.kwargs["stop_token_ids"] == {
            tokenizer.get_bos_token_id(), tokenizer.encode_special("<assistant_end>")}
        if prompt_length == 32:
            assert tokens == tokenizer.render_for_completion({"messages": [
                {"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]})


def test_native_source_can_be_relocated(tmp_path, producer):
    cfg, _ = producer
    cfg.max_cells = None
    native.run_native_decoder(cfg)
    staged = tmp_path / "staged.parquet"
    (tmp_path / "result/steering_data.parquet").rename(staged)
    _, frame = validate_native_source(tmp_path / "result/run_manifest.json", staged)
    assert len(frame) == 10


@pytest.mark.parametrize("filename", ["model_001003.pt", "meta_001003.json", "tokenizer.pkl"])
def test_adv_source_checks_actual_checkpoint_bytes(tmp_path, producer, filename):
    cfg, _ = producer
    cfg.max_cells = None
    native.run_native_decoder(cfg)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(AxBenchAdvError, match="native runtime source changed"):
        validate_native_source(tmp_path / "result/run_manifest.json", tmp_path / "result/steering_data.parquet")


@pytest.mark.parametrize("mutation", ["frozen_input", "implementation"])
def test_adv_rejects_changed_source_or_implementation(tmp_path, producer, mutation):
    cfg, _ = producer
    cfg.max_cells = None
    native.run_native_decoder(cfg)
    path = tmp_path / "result/run_manifest.json"
    if mutation == "frozen_input":
        (tmp_path / "source.parquet").write_bytes(b"changed")
    else:
        manifest = json.loads(path.read_text())
        hashes = manifest["implementation"]["source_sha256"]
        hashes[next(iter(hashes))] = "0" * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(AxBenchAdvError):
        validate_native_source(path, tmp_path / "result/steering_data.parquet")


def test_native_resume_and_adv_reject_changed_resolved_attention(tmp_path, producer, monkeypatch):
    cfg, _ = producer
    cfg.max_cells = None
    monkeypatch.delenv("NANOCHAT_ATTN_IMPL", raising=False)
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    native.run_native_decoder(cfg)
    monkeypatch.setattr(flash_attention, "HAS_FA3", True)
    with pytest.raises(ValueError, match="imported FA3 cache backend"):
        native.run_native_decoder(cfg)
    with pytest.raises(AxBenchAdvError, match="numerical runtime settings differ"):
        validate_native_source(tmp_path / "result/run_manifest.json", tmp_path / "result/steering_data.parquet")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.slow)])
@pytest.mark.parametrize("prompt_length", [32, 2100])
@t.inference_mode()
def test_real_checkpoint_native_generation_matches_full_prefix(tmp_path, monkeypatch, device, prompt_length):
    if device == "cuda" and not t.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tokenizer = RustBPETokenizer(tiktoken.Encoding(
        name="native-checkpoint-test", pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
    ), "<bos>")
    tokenizer.save(tmp_path / "tokenizer")
    config = GPTConfig(sequence_len=2200, vocab_size=tokenizer.get_vocab_size(),
                       n_layer=2, n_head=2, n_kv_head=1, n_embd=32, window_pattern="L")
    t.manual_seed(1745)
    model = GPT(config).to(device)
    model.init_weights()
    # Nonzero projections make the oracle sensitive to the entire prompt and KV positions.
    for name, parameter in model.named_parameters():
        if name.endswith(".c_proj.weight"):
            t.nn.init.normal_(parameter, std=0.02)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    t.save(model.state_dict(), checkpoint / "model_001003.pt")
    (checkpoint / "meta_001003.json").write_text(json.dumps({
        "step": 1003, "model_config": asdict(config), "sft_loss_mask": "completion_only",
        "user_config": {"mixture": "standard_steered", "model_backend": "nanochat"},
    }))
    adapter = native.NativeDecoderAdapter(checkpoint, 1003, device, tmp_path / "tokenizer/tokenizer.pkl")
    prompt = "x" * prompt_length + " question suffix"
    tokens = [tokenizer.get_bos_token_id(), tokenizer.encode_special("<user_start>"),
              *tokenizer.encode(prompt), tokenizer.encode_special("<user_end>"),
              tokenizer.encode_special("<assistant_start>")]
    assert adapter.prompt_tokens(prompt) == tokens
    sm12_fa4.admit_project_attention(device=t.device(device))
    rng = t.Generator(device=device).manual_seed(42)
    expected = []
    for _ in range(4):
        logits = adapter.model(t.tensor([tokens + expected], device=device))[:, -1, :]
        token = sample_next_token(logits, rng, temperature=0.7, top_k=20).item()
        if token in {tokenizer.get_bos_token_id(), tokenizer.encode_special("<assistant_end>")}:
            break
        expected.append(token)
    assert expected
    first = adapter.generate_user_text(prompt, seed=42, max_new_tokens=4, temperature=0.7, top_k=20)
    assert first == tokenizer.decode(expected)
    assert adapter.generate_user_text(prompt, seed=42, max_new_tokens=4, temperature=0.7, top_k=20) == first
    alternate = tmp_path / "alternate.pkl"
    alternate.write_bytes((tmp_path / "tokenizer/tokenizer.pkl").read_bytes())
    with pytest.raises(ValueError, match="tokenizer_path"):
        native.NativeDecoderAdapter(checkpoint, 1003, device, alternate)
    build = native.build_model
    monkeypatch.setattr(native, "build_model", lambda *_args: (adapter.model, object(), {}))
    with pytest.raises(ValueError, match="RustBPE"):
        native.NativeDecoderAdapter(checkpoint, 1003, device, tmp_path / "tokenizer/tokenizer.pkl")

    def changed_checkpoint(*args):
        result = build(*args)
        meta_path = checkpoint / "meta_001003.json"
        meta_path.write_text(meta_path.read_text() + "\n")
        return result

    monkeypatch.setattr(native, "build_model", changed_checkpoint)
    with pytest.raises(ValueError, match="changed while loading"):
        native.NativeDecoderAdapter(checkpoint, 1003, device, tmp_path / "tokenizer/tokenizer.pkl")
