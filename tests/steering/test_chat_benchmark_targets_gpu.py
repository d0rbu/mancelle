"""Opt-in qualification on real Gemma/SAE and >=2.5B-parameter ChatCORE models.

Set CHATCORE_BENCHMARK_GPU_CONFIG to an operator YAML containing `chat_eval`
overrides, `builder` overrides, and `baseline_tree` (an untouched pre-change
checkout). Source fitting and decoder tests use separate model lifetimes.
These tests never replace weights, tokenizers, datasets or GPU kernels.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch as t
import yaml
from omegaconf import OmegaConf
from safetensors.torch import load_file

from scripts.eval import build_chat_benchmark_targets as builder
from scripts.eval import chat_eval
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.self_conditioned import SelfConditionedFeatureProvider
from steering.eval.axbench_upstream import methods
from steering.eval.chat_eval_result_identity import validate_self_conditioned_run_identity

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


@pytest.fixture
def real_config():
    path = os.environ.get("CHATCORE_BENCHMARK_GPU_CONFIG")
    if not path:
        pytest.skip("set CHATCORE_BENCHMARK_GPU_CONFIG to stage real checkpoint, statistics and cache")
    assert t.cuda.is_available(), "qualification explicitly requested but CUDA is unavailable"
    # Collect after the previous test's monkeypatch teardown has released model
    # methods; collecting inside that test still leaves those methods alive.
    gc.collect()
    t.cuda.empty_cache()
    return OmegaConf.load(path)


def tensor_digest(value):
    tensor = value.detach().contiguous().cpu()
    return hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode()
                          + tensor.view(t.uint8).numpy().tobytes()).hexdigest()


def frozen_module(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task_name,count,strengths", [
    (None, 2, (None, 0, 2.5, 6)), ("MMLU", 500, (None, 0)),
], ids=["all-six-shapes", "mmlu-500"])
def test_real_3b_all_tasks_zero_exact_and_both_nonzero_regimes(real_config, tmp_path, monkeypatch, task_name, count, strengths):
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/eval/chat_eval.yaml"), {
        "source": "encdec", "evaluation_mode": "self_conditioned", "batch_size": 1,
        "num_samples": 1, "temperature": 0, "next_segment_conditioning_offset": False,
        "device_type": "cuda", "max_problems": 2, "max_new_tokens": 96,
    }, real_config.chat_eval)
    cfg.task_name, cfg.max_problems = task_name, count
    task_names = chat_eval.ALL_CHAT_TASKS if task_name is None else [task_name]
    target_path = cfg.self_conditioned.benchmark_steering.targets
    published = load_file(target_path)
    feature_ids = published["feature_ids"].tolist()
    traces, selections, predictions = [], [], []
    loaded = []
    original_load = chat_eval.load_model

    def load_model(*args, **kwargs):
        if not loaded:
            model, tokenizer, meta = original_load(*args, **kwargs)
            assert isinstance(model, EncoderDecoderGPT)
            assert sum(parameter.numel() for parameter in model.parameters()) >= 2_500_000_000
            assert model.get_device().type == "cuda"
            assert model.encoder.config.n_sae_features == 65_536
            forward = model.forward
            def capture(*a, **kw):
                output = forward(*a, **kw)
                logits = output[0] if isinstance(output, tuple) else output
                assert bool(t.isfinite(logits).all())
                traces.append((tensor_digest(a[0]), tensor_digest(kw["seg_map"]), tensor_digest(logits)))
                return output
            monkeypatch.setattr(model, "forward", capture)
            loaded.append((model, tokenizer, copy.deepcopy(meta)))
        model, tokenizer, meta = loaded[0]
        return model, tokenizer, copy.deepcopy(meta)

    monkeypatch.setattr(chat_eval, "load_model", load_model)
    select = SelfConditionedFeatureProvider._finalize_two_sided_selection
    def selected(self, *args, **kwargs):
        result = select(self, *args, **kwargs)
        positive, negative = args[1].tolist(), args[2].tolist()
        assert not set(positive) & set(negative)
        assert set(positive + negative) <= set(feature_ids)
        assert result.target_slot_idx == args[0].target_slot_idx + 1
        assert self.last_appended_segment == {**dict.fromkeys(positive, 0), **dict.fromkeys(negative, 1)}
        policy = self.cfg.score_policy
        assert policy is not None
        assert float(policy.weight) == (0 if strength is None else strength / 6)
        if strength is not None:
            name = task_names[len(predictions) // count]
            expected_target = published[f"target.{name}"]
            assert t.equal(policy.target_scores.cpu(), expected_target)
            if strength == 6:
                # Independent Python stable ordering verifies original-ID mapping
                # and both polarities at every S0 and subsequent refresh.
                scores = dict(zip(feature_ids, expected_target.tolist(), strict=True))
                assert positive == sorted(feature_ids, key=lambda fid: -scores[fid])[:len(positive)]
                remaining = [fid for fid in feature_ids if fid not in positive]
                assert negative == sorted(remaining, key=lambda fid: scores[fid])[:len(negative)]
        selections.append((positive, negative))
        return result
    monkeypatch.setattr(SelfConditionedFeatureProvider, "_finalize_two_sided_selection", selected)
    factory = chat_eval._chat_task
    def task_factory(name):
        task = factory(name)
        evaluate = task.evaluate
        def evaluated(conversation, prediction):
            result = evaluate(conversation, prediction)
            predictions.append((name, prediction, bool(result)))
            return result
        monkeypatch.setattr(task, "evaluate", evaluated)
        return task
    monkeypatch.setattr(chat_eval, "_chat_task", task_factory)

    original_runner = chat_eval.run_self_conditioned_chat_eval
    frozen = frozen_module(Path(real_config.baseline_tree) / "steering/eval/chat_self_conditioning.py",
                           "_chatcore_frozen_runtime", monkeypatch)
    def original_runtime(config, **kwargs):
        kwargs.pop("all_task_names")
        return frozen.run_self_conditioned_chat_eval(config, **kwargs)

    records: dict[str, Any] = {}
    for strength in strengths:
        traces.clear()
        selections.clear()
        predictions.clear()
        monkeypatch.setattr(chat_eval, "run_self_conditioned_chat_eval", original_runtime if strength is None else original_runner)
        cfg.self_conditioned.benchmark_steering = None if strength is None else {"targets": target_path, "strength": strength}
        cfg.self_conditioned_output_dir = str(tmp_path / f"strength-{strength}")
        t.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        chat_eval.main.__wrapped__(cfg)
        t.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = yaml.safe_load((Path(cfg.self_conditioned_output_dir) / "self_conditioned/result.yaml").read_text())["seeds"]["adhoc"]
        validate_self_conditioned_run_identity(result["run_identity"])
        assert set(result["results"]) == set(task_names)
        assert len(predictions) == len(task_names) * cfg.max_problems
        if task_name is None:
            assert len(selections) > len(predictions), "real trajectories must exercise later refreshes"
        else:
            assert len(selections) == len(predictions)
        record = {"trace": copy.deepcopy(traces), "selections": copy.deepcopy(selections),
            "predictions": copy.deepcopy(predictions), "result": result,
            "peak_cuda_bytes": t.cuda.max_memory_allocated(), "seconds_with_trace_capture": elapsed}
        records[str(strength)] = record
        if strength == 0:
            baseline = records["None"]
            for key in ("trace", "selections", "predictions"):
                assert record[key] == baseline[key], key
            assert result["results"] == baseline["result"]["results"]
            assert result["task_details"] == baseline["result"]["task_details"]
        elif strength is not None:
            assert record["selections"] != records["0"]["selections"]
            assert result["run_identity"]["conditioning"]["score_policy"]["weight"] == strength / 6
    (tmp_path / "real-3b-qualification.json").write_text(json.dumps(records, sort_keys=True))
    # An external qualification run can preserve the same evidence independently
    # of pytest's temporary-directory retention policy.
    evidence = os.environ.get("CHATCORE_BENCHMARK_GPU_EVIDENCE")
    if evidence:
        path = Path(evidence)
        if task_name is not None:
            path = path.with_stem(f"{path.stem}-{task_name}-{count}")
        path.write_text(json.dumps(records, sort_keys=True))
    loaded.clear()
    gc.collect()
    t.cuda.empty_cache()


def test_real_gemma_sae_targets_equal_frozen_axbench_exactly(real_config, monkeypatch, tmp_path):
    config = real_config.builder
    setup = methods.load_method_setup(Path(config.checkpoint_meta_path), Path(config.stats_path))
    assert setup.n_sae_features == 65_536
    corpus = json.loads((Path(config.output_dir) / "corpus.json").read_text())
    assert all(corpus[split] for split in ("fit", "calibration", "background"))
    assert len(corpus["background"]) == 1024
    assert len(corpus["fit"]["MMLU"]) >= 250
    assert len(corpus["fit"]["HumanEval"]) >= 100
    # Import the entire untouched implementation, so equality does not compare
    # two new callers of the same shared reducer.
    reference_path = Path(real_config.baseline_tree) / "steering/eval/axbench_upstream/methods.py"
    reference_source = reference_path.read_text()
    assert "def _mean_percentiles(" in reference_source
    assert "dense_feature_targets" not in reference_source
    reference = frozen_module(reference_path, "_chatcore_frozen_axbench", monkeypatch)
    model, sae = methods.load_extraction_stack(setup, "cuda")
    model.eval()
    sae.eval()
    if getattr(model, "nnterp_model", None) is not None:
        model.nnterp_model.model.eval()
    positive = corpus["fit"]
    negative = corpus["background"]
    extract = builder._batched_dense_sae_max_pool
    activations = {}
    def captured(texts, *args, **kwargs):
        result = extract(texts, *args, **kwargs)
        assert result.shape == (len(texts), 65_536)
        assert bool(t.isfinite(result).all())
        activations[tuple(texts)] = result.detach().clone()
        return result
    monkeypatch.setattr(builder, "_batched_dense_sae_max_pool", captured)
    allowed = t.nonzero(setup.consolidation.allowed_universe, as_tuple=False).squeeze(1) if setup.consolidation else setup.allowed_ids
    if allowed is None:
        allowed = t.arange(65_536)
    with t.inference_mode():
        targets = builder.fit_targets(positive, negative, model=model, sae=sae, sae_model=setup.sae_model,
            sae_layer=setup.sae_layer, percentile_table=setup.percentile_table, allowed_ids=allowed,
            consolidation=setup.consolidation, missing=methods._diffmean_missing_percentile(setup, allowed),
            feature_vocab=setup.feature_vocab, batch_size=1)
    assert len(activations) == 7
    monkeypatch.setattr(reference, "_batched_dense_sae_max_pool", lambda texts, *args, **kwargs: activations[tuple(texts)].clone())
    digests = {}
    published = load_file(Path(config.output_dir) / "targets.safetensors")
    for index, name in enumerate(builder.ALL_CHAT_TASKS):
        original = reference.fit_dense_diffmean(setup, model, sae, positive[name], negative, concept_id=index, extraction_batch_size=1)
        assert t.equal(targets[name], original), name
        assert t.equal(targets[name], published[f"target.{name}"]), f"published {name}"
        digests[name] = tensor_digest(original)
    (tmp_path / "real-source-parity.json").write_text(json.dumps({"target_sha256": digests,
        "reference_source_sha256": hashlib.sha256(reference_source.encode()).hexdigest()}, sort_keys=True))
    del model, sae
    gc.collect()
    t.cuda.empty_cache()
