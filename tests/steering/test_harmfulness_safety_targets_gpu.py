"""Opt-in real Gemma/SAE fit followed by turn-aware harmfulness generation.

Set HARMFULNESS_SAFETY_GPU_CONFIG to YAML with `builder` and `harmfulness_eval`
overrides pointing to real checkpoint/statistics/tokenizers and a prebuilt cache
for one harmful and one benign prompt, seed 42. NANOCHAT_BASE_DIR supplies the
native tokenizer/checkpoint tree. This test calls no judge or paid provider.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from scripts.eval import build_harmfulness_safety_target as builder
from scripts.eval import harmfulness_eval
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.self_conditioned import SelfConditionedFeatureProvider
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.harmfulness_self_conditioning import (
    generate_harmfulness_response_self_conditioned,
    harmfulness_completion_provenance,
    prepare_harmfulness_self_conditioning,
)

ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.slow


def wait_for_shared_gpu():
    deadline = time.monotonic() + 15 * 60
    while True:
        completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True)
        others = {int(line.strip()) for line in completed.stdout.splitlines() if line.strip()} - {os.getpid()}
        if not others:
            return
        if time.monotonic() >= deadline:
            pytest.skip(f"shared GPU remained busy with other PIDs: {sorted(others)}")
        time.sleep(60)


def tensor_digest(value):
    tensor = value.detach().contiguous().cpu()
    return hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode()
                          + tensor.view(t.uint8).numpy().tobytes()).hexdigest()


def test_real_tiny_fit_then_steered_harmfulness_generation(tmp_path, monkeypatch):
    config_path = os.environ.get("HARMFULNESS_SAFETY_GPU_CONFIG")
    if not config_path:
        pytest.skip("set HARMFULNESS_SAFETY_GPU_CONFIG for real fit and cached harmfulness generation")
    overrides = OmegaConf.load(config_path)
    fitting = OmegaConf.merge(OmegaConf.load(ROOT / "configs/eval/build_harmfulness_safety_target.yaml"), overrides.builder,
        {"output_dir": str(tmp_path / "target"), "positive_cap": 2, "background_count": 2,
         "extraction_batch_size": 1, "device": "cuda", "dry_run": True})
    assert isinstance(fitting, DictConfig)
    preparation = builder.run(fitting)
    assert preparation.is_file()
    wait_for_shared_gpu()
    assert t.cuda.is_available(), "explicit GPU qualification requested but CUDA is unavailable"
    total_memory = t.cuda.get_device_properties(0).total_memory
    t.cuda.set_per_process_memory_fraction(min(1.0, 14 * 1024**3 / total_memory), 0)
    t.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    fitting.dry_run = False
    target_path = builder.run(fitting)
    fit_seconds = time.monotonic() - started
    fit_peak = t.cuda.max_memory_allocated()
    assert fit_seconds < 15 * 60 and fit_peak < 16 * 1024**3
    published = load_file(target_path)
    assert set(published) == {"feature_ids", "target.safety"}
    assert bool(t.isfinite(published["target.safety"]).all())
    assert float(published["target.safety"].norm()) > 0
    gc.collect()
    t.cuda.empty_cache()

    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/eval/harmfulness_eval.yaml"), overrides.harmfulness_eval,
        {"num_harmful": 1, "num_benign": 1, "seed": 42, "batch_size": 1,
         "max_new_tokens": 48, "temperature": 0, "top_k": 50, "feature_configs": None,
         "self_cond_segment_size": None, "self_conditioned": {"safety_steering": None}})
    assert isinstance(cfg, DictConfig)
    wait_for_shared_gpu()
    t.cuda.reset_peak_memory_stats()
    model, tokenizer, meta, feature_contract, _checkpoint_identity = harmfulness_eval._load_model_for_harmfulness(cfg, t.device("cuda"))
    assert isinstance(model, EncoderDecoderGPT)
    # Match harmfulness_eval's explicit CUDA parameter/KV-cache dtype contract.
    model.to(dtype=t.bfloat16)
    pairs = harmfulness_eval._load_expected_prompt_pairs(cfg)
    assert len(pairs) == 2 and {pair["pool"] for pair in pairs} == {"harmful", "benign"}
    prompt_ids = [render_prompt_for_chat_completion(tokenizer, pair["prompt"]).token_ids for pair in pairs]
    traces, selections, policies = [], [], []
    forward = model.forward

    def captured_forward(*args, **kwargs):
        output = forward(*args, **kwargs)
        logits = output[0] if isinstance(output, tuple) else output
        assert bool(t.isfinite(logits).all())
        traces.append(tensor_digest(logits))
        return output

    monkeypatch.setattr(model, "forward", captured_forward)
    finalize = SelfConditionedFeatureProvider._finalize_two_sided_selection

    def captured_selection(self, *args, **kwargs):
        result = finalize(self, *args, **kwargs)
        selections.append((args[1].tolist(), args[2].tolist()))
        policies.append(float(self.cfg.score_policy.weight))
        return result

    monkeypatch.setattr(SelfConditionedFeatureProvider, "_finalize_two_sided_selection", captured_selection)
    records = {}
    for strength in (None, 0, 2.5):
        wait_for_shared_gpu()
        started = time.monotonic()
        traces.clear()
        selections.clear()
        policies.clear()
        cfg.self_conditioned.safety_steering = None if strength is None else {"targets": str(target_path), "strength": strength}
        setup = prepare_harmfulness_self_conditioning(cfg, model=model, tokenizer=tokenizer, device=t.device("cuda"),
            meta=meta, feature_contract=feature_contract, prompt_token_ids=prompt_ids)
        responses = [generate_harmfulness_response_self_conditioned(setup=setup, stable_pool_index=index,
            prompt=pair["prompt"], pool=pair["pool"], category=pair["category"], base_seed=cfg.seed,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
            for index, pair in enumerate(pairs)]
        assert len(responses) == 2 and all(isinstance(response, str) for response in responses)
        assert traces and selections
        assert set(policies) == {0.0 if strength is None else strength / 6}
        receipt = harmfulness_completion_provenance(setup)
        if strength is not None:
            assert receipt["safety_steering"] == {"targets_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                                                  "strength": strength}
        elapsed = time.monotonic() - started
        assert elapsed < 15 * 60 and t.cuda.max_memory_allocated() < 16 * 1024**3
        records[str(strength)] = {"responses": responses, "trace": list(traces), "selections": list(selections),
                                  "receipt": receipt, "seconds": elapsed}
    for field in ("responses", "trace", "selections"):
        assert records["None"][field] == records["0"][field], field
    evidence = {"fit_seconds": fit_seconds, "fit_peak_bytes": fit_peak,
                "generation_peak_bytes": t.cuda.max_memory_allocated(), "records": records,
                "target_l2": float(published["target.safety"].norm()), "prompts": pairs}
    (tmp_path / "qualification.json").write_text(json.dumps(evidence, sort_keys=True))
