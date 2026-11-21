"""Checkpoint-scoped dense steering, real CPU forwards, and resume receipts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors import safe_open

import scripts.train.training_progression as progression
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import model as model_module
from steering.eval.chat_benchmark_targets import TARGET_METADATA_KEY, benchmark_fit_contract
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.context_cache_io import MANIFEST_FILENAME, ContextCacheEntry, save_context_cache
from steering.eval.harmfulness_generation import STRENGTH_SWEEP_SPEC_SHA256_KEY
from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult
from steering.eval.harmfulness_progression import load_safety_progression_runs
from steering.eval.harmfulness_safety_targets import write_safety_targets
from steering.eval.target_conditioning import ContextFeatureSegment, token_context_key
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.overlay_io import COMPRESSED_EXT
from steering.refined_types import PositiveInt
from steering.safetensors_io import deterministic_safetensors_bytes
from tests.scripts.eval.test_build_harmfulness_safety_target import (
    fitting_provenance,
    populations,
    prepare,
    source_provenance,
)
from tests.steering.test_chat_eval_self_conditioned import fixed_fixture


def _cfg(tmp_path: Path):
    return OmegaConf.create({
        "checkpoint_source": "local", "cleanup_after_eval": False,
        "model_family": "encoder_decoder", "model_tag": "tiny",
        "checkpoint_root": "conditional_encdec_checkpoints", "checkpoint_prefix": "",
        "checkpoints_dir": str(tmp_path), "evals": "harmfulness", "dtype": "float32",
        "device_type": "cpu", "harm_temperature": 0.0, "harm_top_k": 10,
        "harm_max_new_tokens": 3, "harm_num_harmful": 1, "harm_num_benign": 1,
        "harm_judge_model": "test/judge", "stats_path": None,
        "conditioning": {"conditioned_configs": None, "target_real_configs": None,
                         "context_cache_dir": str(tmp_path / "cache")},
        "self_conditioned": {"safety_steering": None}, "self_conditioned_overrides": [],
    })


def _checkpoint(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True)
    family = fixed_fixture(tmp_path, monkeypatch)
    prompts = [{"prompt": "harmful tiny", "category": "Violence", "pool": "harmful"},
               {"prompt": "benign tiny", "category": "benign", "pool": "benign"}]
    ids = [render_prompt_for_chat_completion(family.tokenizer, row["prompt"]).token_ids for row in prompts]
    root = tmp_path / "harm-cache"
    cache = root / family.cache.relative_to(tmp_path / "cache")
    save_context_cache(cache, [ContextCacheEntry(tuple(tokens), (
        ContextFeatureSegment((1, 2), (0, 1), 0, len(family.tokenizer.decode(tokens))),
    )) for tokens in ids], identity=family.contract.identity,
        n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(2),
        metadata=family.contract.metadata)
    contract = resolve_checkpoint_feature_contract(family.meta, n_sae_features=PositiveInt.parse(8),
        surface="test", source_path=str(family.checkpoint_path))
    fit_contract, feature_ids = benchmark_fit_contract(feature_contract=contract, n_sae_features=8,
        tokenizer=family.tokenizer, fixed_contract=family.contract)
    training, _, background = populations()
    _, audit, evaluation_keys = prepare((training, prompts, background), tokenizer=family.tokenizer)
    audit["sources"] = source_provenance()
    target_path = tmp_path / "targets.safetensors"
    target = write_safety_targets(target_path, target=t.linspace(-0.8, 0.8, 8), feature_ids=feature_ids,
        tokenizer=family.tokenizer,
        manifest={"format_version": 1, "fit_contract": fit_contract, "preparation": audit,
                  "fit_provenance": fitting_provenance(fit_contract),
                  "evaluation_prompt_keys": evaluation_keys})
    family.meta["checkpoint_dir"] = str(family.checkpoint_path.parent)
    return family, contract, prompts, root, target_path, target


@pytest.mark.parametrize("shared_bundle", [False, True])
def test_two_tiny_checkpoints_generate_through_steered_progression(tmp_path, monkeypatch, request, shared_bundle):
    previous = t.get_num_threads()
    t.set_num_threads(1)
    request.addfinalizer(lambda: t.set_num_threads(previous))
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    cells = [_checkpoint(tmp_path / phase, monkeypatch) for phase in ("instruct", "safety")]
    if shared_bundle:
        # Distinct weights/checkpoint bytes with the same fitted feature space.
        first, contract, prompts, cache, path, target = cells[0]
        second = copy.copy(first)
        second.model, second.meta = copy.deepcopy(first.model), copy.deepcopy(first.meta)
        with t.no_grad():
            next(second.model.parameters()).add_(0.01)
        second.checkpoint_path = cells[1][0].checkpoint_path
        weight_path = second.checkpoint_path.parent / "model_001116.pt"
        t.save(second.model.state_dict(), weight_path)
        second.meta["checkpoint_dir"] = str(second.checkpoint_path.parent)
        second.meta["fixed_self_cond_cadence_provenance"]["original_checkpoint"] = [{
            "path": str(weight_path), "sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest()}]
        second.checkpoint_path.write_text(json.dumps(second.meta))
        cells[1] = (second, contract, prompts, cache, path, target)
    cfg = _cfg(tmp_path)
    cfg.self_conditioned_overrides = [{"phase": phase, "step": 1116, "config": {
        "conditioning": {"context_cache_dir": str(cell[3])},
        "self_conditioned": {"safety_steering": {
            "targets": str(cells[0][4] if shared_bundle else cell[4]), "strength": 2.5}},
    }} for phase, cell in zip(("instruct", "safety"), cells, strict=True)]
    judge = MagicMock()
    judge.score_all.return_value = [JudgeResult([], 1, "Safe."), JudgeResult([], 1, "Helpful.")]
    judge.score_compliance_all.return_value = [ComplianceJudgeResult(5, False, "Helpful.")]
    monkeypatch.setattr(progression.harmfulness_judge, "JudgeClient", lambda **kwargs: judge)
    all_results = []
    for phase, (family, contract, prompts, _, _, target) in zip(("instruct", "safety"), cells, strict=True):
        calls = []
        family.model.register_forward_hook(lambda *args, calls=calls: calls.append(1))
        monkeypatch.setattr(progression, "resolve_local_checkpoint_dir", lambda *args, family=family: str(family.checkpoint_path.parent))
        monkeypatch.setattr(progression, "load_progression_model", lambda *args, family=family, contract=contract: (
            family.model, family.tokenizer, family.meta, contract))
        monkeypatch.setattr(progression, "resolve_harmfulness_prompt_pool", lambda prompts=prompts, **kwargs: SimpleNamespace(rows=prompts))
        spec = progression.CheckpointSpec(1116, phase, "")
        result = progression.evaluate_single_checkpoint(spec, t.device("cpu"), t.float32,
            ["harmfulness"], [42], cfg)
        judged = judge.score_all.call_count
        resumed = progression.evaluate_single_checkpoint(spec, t.device("cpu"), t.float32,
            ["harmfulness"], [42], cfg, existing_entry={"evals": result})
        assert resumed == result
        assert judge.score_all.call_count == judged, "completed seed must not be regenerated or rejudged"
        split = progression.evaluate_single_checkpoint(spec, t.device("cpu"), t.float32,
            ["harmfulness"], [42, 43], cfg, existing_entry={"evals": result}, work_keys={
                progression._eval_work_key("harmfulness", "conditioned", "self_conditioned", 43)})
        assert judge.score_all.call_count == judged + 1
        assert set(split["harmfulness"]["conditioned"]["configs"]["self_conditioned"]["seeds"]) == {"42", "43"}
        seed = result["harmfulness"]["conditioned"]["configs"]["self_conditioned"]["seeds"]["42"]
        assert calls, "the tiny checkpoint must perform real FP/KV forwards"
        expected_target = cells[0][5] if shared_bundle else target
        assert seed["completion"]["safety_steering"] == {"targets_sha256": expected_target.sha256, "strength": 2.5}
        assert progression._eval_seed_payload_done(seed, eval_name="harmfulness", seed=42,
            cfg=progression._cfg_for_checkpoint(cfg, spec))
        corrupted = copy.deepcopy(seed)
        corrupted["completion"]["safety_steering"]["targets_sha256"] = "e" * 64
        assert not progression._eval_seed_payload_done(corrupted, eval_name="harmfulness", seed=42,
            cfg=progression._cfg_for_checkpoint(cfg, spec))
        assert all(isinstance(row["response"], str) for row in judge.score_all.call_args.args[0])
        all_results.append(seed)
    assert len(all_results) == 2


def test_progression_resume_binds_target_bytes_strength_and_statistics(tmp_path):
    cfg = _cfg(tmp_path)
    cache = Path(cfg.conditioning.context_cache_dir)
    cache.mkdir()
    (cache / MANIFEST_FILENAME).write_text("{}")
    target = tmp_path / "targets.safetensors"
    target.write_bytes(b"first bundle")
    stats = tmp_path / "statistics.safetensors"
    stats.write_bytes(b"first stats")
    cfg.stats_path = str(stats)
    cfg.self_conditioned.safety_steering = {"targets": str(target), "strength": 2.5}
    original = progression._harmfulness_progression_provenance(cfg, 42)
    cfg.self_conditioned.safety_steering["strength"] = 0
    assert progression._harmfulness_progression_provenance(cfg, 42) != original
    cfg.self_conditioned.safety_steering["strength"] = 2.5
    target.write_bytes(b"changed bundle")
    assert progression._harmfulness_progression_provenance(cfg, 42) != original
    target.write_bytes(b"first bundle")
    stats.write_bytes(b"changed stats")
    assert progression._harmfulness_progression_provenance(cfg, 42) != original


def test_same_checkpoint_three_strengths_save_separate_collectable_receipts(tmp_path, monkeypatch, request):
    previous = t.get_num_threads()
    t.set_num_threads(1)
    request.addfinalizer(lambda: t.set_num_threads(previous))
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    family, contract, prompts, cache, path, target = _checkpoint(tmp_path / "checkpoint", monkeypatch)
    cfg = _cfg(tmp_path)
    cfg.conditioning.context_cache_dir = str(cache)
    judge = MagicMock()
    judge.score_all.return_value = [JudgeResult([], 1, "Safe."), JudgeResult([], 1, "Helpful.")]
    judge.score_compliance_all.return_value = [ComplianceJudgeResult(5, False, "Helpful.")]
    monkeypatch.setattr(progression.harmfulness_judge, "JudgeClient", lambda **kwargs: judge)
    monkeypatch.setattr(progression, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=prompts))
    paths, payloads = [], []
    for strength in (0, 3, 6):
        cfg.self_conditioned.safety_steering = {"targets": str(path), "strength": strength}
        cfg.results_file = str(tmp_path / f"strength-{strength}" / "results.json")
        payload = progression.run_self_conditioned_harmfulness_eval(model=family.model, tokenizer=family.tokenizer,
            device=t.device("cpu"), seed=42, cfg=cfg, meta=family.meta, feature_contract=contract)
        assert payload["completion"]["safety_steering"] == {"targets_sha256": target.sha256, "strength": strength}
        assert progression._eval_seed_payload_done(payload, eval_name="harmfulness", seed=42, cfg=cfg)
        if payloads:
            assert not progression._eval_seed_payload_done(payloads[-1], eval_name="harmfulness", seed=42, cfg=cfg)
        progression.save_checkpoint_result(cfg, 1116, "safety_sft", {"harmfulness": {"conditioned": {"configs": {
            "self_conditioned": {"kind": "self_conditioned", "cache_dir": str(cache), "seeds": {"42": payload}}}}}},
            checkpoint_identity={"model_sha256": hashlib.sha256(family.checkpoint_path.with_name("model_001116.pt").read_bytes()).hexdigest()})
        progression.save_merged_results(cfg)
        paths.append(Path(cfg.results_file))
        payloads.append(payload)
    records, stages = load_safety_progression_runs(paths, model_tag=cfg.model_tag)
    assert len(records) == 3 and len(stages) == 1
    assert len({record.signal for record in records}) == 3
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize("field, value", [
    ("harm_temperature", 0.5), ("harm_top_k", 25), ("harm_max_new_tokens", 5),
    ("harm_num_harmful", 2), ("harm_num_benign", 2), ("harm_judge_model", "other/judge"),
    ("dtype", "bfloat16"), ("self_cond_segment_size", 64),
])
def test_sweep_digest_binds_every_control_except_strength(tmp_path, field, value):
    cfg = _cfg(tmp_path)
    cache = Path(cfg.conditioning.context_cache_dir)
    cache.mkdir()
    (cache / MANIFEST_FILENAME).write_text("{}")
    target = tmp_path / "target.safetensors"
    target.write_bytes(b"target")
    cfg.self_conditioned.safety_steering = {"targets": str(target), "strength": 0}
    original = progression._harmfulness_progression_provenance(cfg, 42)[STRENGTH_SWEEP_SPEC_SHA256_KEY]
    cfg.self_conditioned.safety_steering["strength"] = 6
    assert progression._harmfulness_progression_provenance(cfg, 42)[STRENGTH_SWEEP_SPEC_SHA256_KEY] == original
    assert progression._harmfulness_progression_provenance(cfg, 43)[STRENGTH_SWEEP_SPEC_SHA256_KEY] != original
    cfg[field] = value
    assert progression._harmfulness_progression_provenance(cfg, 42)[STRENGTH_SWEEP_SPEC_SHA256_KEY] != original


def test_sweep_digest_uses_cache_and_statistics_bytes_independent_of_paths(tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "target.safetensors"
    target.write_bytes(b"target")
    cfg.self_conditioned.safety_steering = {"targets": str(target), "strength": 3}
    digests = []
    for directory in (tmp_path / "first", tmp_path / "second"):
        directory.mkdir()
        (directory / MANIFEST_FILENAME).write_text("{}")
        stats = directory / "stats.safetensors"
        stats.write_bytes(b"stats")
        cfg.conditioning.context_cache_dir = str(directory)
        cfg.stats_path = str(stats)
        digests.append(progression._harmfulness_progression_provenance(cfg, 42)[STRENGTH_SWEEP_SPEC_SHA256_KEY])
    assert digests[0] == digests[1]
    stats.write_bytes(b"changed statistics")
    assert progression._harmfulness_progression_provenance(cfg, 42)[STRENGTH_SWEEP_SPEC_SHA256_KEY] != digests[1]


@pytest.mark.parametrize("fault", ["statistics", "population"])
def test_each_checkpoint_rejects_mismatched_target_before_forward(tmp_path, monkeypatch, fault):
    family, contract, prompts, cache, path, _ = _checkpoint(tmp_path / "checkpoint", monkeypatch)
    with safe_open(path, framework="pt") as source:
        tensors = {key: source.get_tensor(key) for key in source.keys()}
        metadata = source.metadata()
    manifest = json.loads(metadata[TARGET_METADATA_KEY])
    if fault == "statistics":
        manifest["fit_contract"]["statistics_sha256"] = "e" * 64
    else:
        manifest["evaluation_prompt_keys"]["safety"] = [token_context_key([999])]
    metadata[TARGET_METADATA_KEY] = canonical_turn_payload(manifest).decode()
    path.write_bytes(deterministic_safetensors_bytes(tensors, metadata=metadata))
    cfg = _cfg(tmp_path)
    cfg.conditioning.context_cache_dir = str(cache)
    cfg.self_conditioned.safety_steering = {"targets": str(path), "strength": 0}
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setattr(progression, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=prompts))
    monkeypatch.setattr(family.model, "forward", lambda *args, **kwargs: pytest.fail("mismatch reached forward"))
    with pytest.raises(ValueError, match="contract|excluded|evaluation|prompt"):
        progression.run_self_conditioned_harmfulness_eval(model=family.model,
            tokenizer=family.tokenizer, device=t.device("cpu"), seed=42, cfg=cfg,
            meta=family.meta, feature_contract=contract)


def test_checkpoint_override_is_explicit_and_does_not_mutate_global_cfg(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.self_conditioned_overrides = [{"phase": "safety", "step": 9, "config": {
        "self_conditioned": {"safety_steering": {"targets": "/new", "strength": 2.5}}}}]
    changed = progression._cfg_for_checkpoint(cfg, progression.CheckpointSpec(9, "safety", ""))
    assert changed.self_conditioned.safety_steering.targets == "/new"
    assert cfg.self_conditioned.safety_steering is None
    assert progression._cfg_for_checkpoint(cfg, progression.CheckpointSpec(8, "safety", "")) is cfg
    cfg.self_conditioned_overrides.append(copy.deepcopy(cfg.self_conditioned_overrides[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        progression._cfg_for_checkpoint(cfg, progression.CheckpointSpec(9, "safety", ""))


@pytest.mark.parametrize("override", [
    {}, {"phase": "safety", "step": 9, "config": []},
    {"phase": "safety", "step": 9, "config": {"temperature": 1}},
    {"phase": "safety", "step": True, "config": {}},
    {"phase": "", "step": 9, "config": {}},
])
def test_malformed_checkpoint_overrides_fail_loud(tmp_path, override):
    cfg = _cfg(tmp_path)
    cfg.self_conditioned_overrides = [override]
    with pytest.raises((TypeError, ValueError), match="self_conditioned_overrides"):
        progression._cfg_for_checkpoint(cfg, progression.CheckpointSpec(9, "safety", ""))


def test_unconfigured_checkpoint_override_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.phases = [{"phase": "safety", "steps": [9], "subdir": ""}]
    cfg.self_conditioned_overrides = [{"phase": "typo", "step": 9, "config": {}}]
    with pytest.raises(ValueError, match="unconfigured checkpoints"):
        progression.build_checkpoint_specs(cfg)


@pytest.mark.parametrize("block", [None, {"safety_steering": None},
    {"safety_steering": {"targets": "/targets", "strength": 2.5}}])
def test_sparse_harmfulness_progression_rejected_before_loading(tmp_path, monkeypatch, block):
    cfg = _cfg(tmp_path)
    cfg.self_conditioned = block
    cfg.conditioning.conditioned_configs = [{"name": "legacy", "features": {1: 1}}]
    monkeypatch.setattr(progression, "load_progression_model", lambda *args: pytest.fail("must reject before load"))
    with pytest.raises(ValueError, match="no longer accepts hand-supplied"):
        progression.evaluate_single_checkpoint(progression.CheckpointSpec(1, "safety", ""),
            t.device("cpu"), t.float32, ["harmfulness"], [42], cfg)


def test_self_conditioned_variant_supports_progression_plot_dispatch(tmp_path):
    linestyle, _ = progression._variant_linestyle_resolver("self_conditioned", {
        "kind": "self_conditioned", "cache_dir": str(tmp_path), "seeds": {}})
    assert linestyle == "-"


@pytest.mark.parametrize("dtype, expected", [("bfloat16", t.bfloat16), ("float32", t.float32)])
def test_progression_cuda_loading_stages_weights_on_cpu_then_transfers_once(tmp_path, monkeypatch, dtype, expected):
    cfg = _cfg(tmp_path)
    cfg.dtype = dtype
    model = MagicMock()
    model.to.return_value = model
    build = MagicMock(return_value=(model, object(), {}))
    monkeypatch.setattr(progression, "build_encdec_model", build)
    monkeypatch.setattr(progression, "assert_conditioned_eval_model", lambda *args, **kwargs: object())
    progression.load_progression_model(str(tmp_path), 9, t.device("cuda"), cfg)
    assert build.call_args.args[2] == t.device("cpu")
    model.to.assert_called_once_with(device=t.device("cuda"), dtype=expected)


@pytest.mark.parametrize("artifact", [MANIFEST_FILENAME, f"contexts_00000{COMPRESSED_EXT}"])
def test_progression_receipt_binds_cache_bytes_but_ignores_logs(tmp_path, artifact):
    cfg = _cfg(tmp_path)
    cache = Path(cfg.conditioning.context_cache_dir)
    cache.mkdir()
    (cache / MANIFEST_FILENAME).write_bytes(b"{}")
    shard = cache / f"contexts_00000{COMPRESSED_EXT}"
    shard.write_bytes(b"first")
    original = progression._harmfulness_progression_provenance(cfg, 42)
    path = cache / artifact
    previous = path.read_bytes()
    path.write_bytes(b"x" * len(previous))
    changed = progression._harmfulness_progression_provenance(cfg, 42)
    assert changed != original, "even same-size cache changes must invalidate resume"
    path.write_bytes(previous)
    assert progression._harmfulness_progression_provenance(cfg, 42) == original
    (cache / "validation.log").write_text("new validation output")
    assert progression._harmfulness_progression_provenance(cfg, 42) == original
    shard.unlink()
    assert progression._harmfulness_progression_provenance(cfg, 42) != original


def test_cache_without_manifest_fails_loud_before_provenance(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg.conditioning.context_cache_dir).mkdir()
    with pytest.raises(ValueError, match="cache has no manifest"):
        progression._harmfulness_progression_provenance(cfg, 42)
    cfg.conditioning.context_cache_dir = None
    with pytest.raises(ValueError, match="requires conditioning.context_cache_dir"):
        progression._harmfulness_progression_provenance(cfg, 42)


@pytest.mark.parametrize("fault", ["preparation", "generation", "judging", "completion"])
def test_progression_refuses_changed_inputs_before_recording_seed(tmp_path, monkeypatch, fault):
    family, contract, prompts, cache, path, _ = _checkpoint(tmp_path / "checkpoint", monkeypatch)
    cfg = _cfg(tmp_path)
    cfg.conditioning.context_cache_dir = str(cache)
    cfg.self_conditioned.safety_steering = {"targets": str(path), "strength": 0}
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setattr(progression, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=prompts))
    manifest = next(cache.rglob(MANIFEST_FILENAME))

    def change_cache():
        manifest.write_bytes(manifest.read_bytes() + b" ")

    original_prepare = progression.prepare_harmfulness_self_conditioning

    def prepare_runtime(*args, **kwargs):
        setup = original_prepare(*args, **kwargs)
        if fault == "preparation":
            change_cache()
        return setup

    monkeypatch.setattr(progression, "prepare_harmfulness_self_conditioning", prepare_runtime)

    def generate_response(**kwargs):
        if fault == "generation":
            change_cache()
        return "response"

    generation = MagicMock(side_effect=generate_response)
    monkeypatch.setattr(progression, "generate_harmfulness_response_self_conditioned", generation)
    judge = MagicMock()

    def score_all(rows):
        if fault == "judging":
            change_cache()
        return [JudgeResult([], 1, "Safe.") for row in rows]

    judge.score_all.side_effect = score_all
    judge.score_compliance_all.return_value = [ComplianceJudgeResult(5, False, "Helpful.")]
    client = MagicMock(return_value=judge)
    monkeypatch.setattr(progression.harmfulness_judge, "JudgeClient", client)
    if fault == "completion":
        original_completion = progression.harmfulness_completion_provenance

        def corrupt_completion(setup):
            receipt = original_completion(setup)
            receipt["safety_steering"]["targets_sha256"] = "e" * 64
            return receipt

        monkeypatch.setattr(progression, "harmfulness_completion_provenance", corrupt_completion)
    with pytest.raises(ValueError, match="changed"):
        progression.run_self_conditioned_harmfulness_eval(model=family.model, tokenizer=family.tokenizer,
            device=t.device("cpu"), seed=42, cfg=cfg, meta=family.meta, feature_contract=contract)
    if fault in {"preparation", "completion"}:
        generation.assert_not_called()
    if fault != "judging":
        client.assert_not_called()


def test_self_conditioned_progression_renders_all_prompts_and_preserves_runtime_inputs(tmp_path, monkeypatch):
    family, contract, prompts, cache, _, _ = _checkpoint(tmp_path / "checkpoint", monkeypatch)
    cfg = _cfg(tmp_path)
    cfg.harm_top_k = None
    cfg.conditioning.context_cache_dir = str(cache)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setattr(progression, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=prompts))
    renderer = MagicMock(wraps=family.tokenizer.render_for_completion)
    monkeypatch.setattr(family.tokenizer, "render_for_completion", renderer)
    generation = MagicMock(side_effect=["harmful response", "benign response"])
    monkeypatch.setattr(progression, "generate_harmfulness_response_self_conditioned", generation)
    judge = MagicMock()
    judge.score_all.return_value = [JudgeResult([], 1, "Safe."), JudgeResult([], 1, "Helpful.")]
    judge.score_compliance_all.return_value = [ComplianceJudgeResult(5, False, "Helpful.")]
    monkeypatch.setattr(progression.harmfulness_judge, "JudgeClient", lambda **kwargs: judge)
    payload = progression.run_self_conditioned_harmfulness_eval(model=family.model, tokenizer=family.tokenizer,
        device=t.device("cpu"), seed=42, cfg=cfg, meta=family.meta, feature_contract=contract)
    assert [call.args[0] for call in renderer.call_args_list] == [
        {"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": ""}]}
        for row in prompts]
    for index, call in enumerate(generation.call_args_list):
        assert call.kwargs["stable_pool_index"] == index
        assert call.kwargs["base_seed"] == 42
        assert call.kwargs["prompt"] == prompts[index]["prompt"]
        assert call.kwargs["pool"] == prompts[index]["pool"]
        assert call.kwargs["category"] == prompts[index]["category"]
        assert call.kwargs["top_k"] is None
    assert [row["response"] for row in judge.score_all.call_args.args[0]] == ["harmful response", "benign response"]
    assert all("response" not in row for row in prompts), "prompt-pool records must remain immutable"
    assert progression._eval_seed_payload_done(payload, eval_name="harmfulness", seed=42, cfg=cfg)
    manifest = next(cache.rglob(MANIFEST_FILENAME))
    original = manifest.read_bytes()
    manifest.write_bytes(original + b" ")
    assert not progression._eval_seed_payload_done(payload, eval_name="harmfulness", seed=42, cfg=cfg)
    manifest.write_bytes(original)
    assert progression._eval_seed_payload_done(payload, eval_name="harmfulness", seed=42, cfg=cfg)
    entry = {"evals": {"harmfulness": {"conditioned": {"configs": {
        "self_conditioned": {"seeds": {"42": payload}}}}}}}
    assert progression._pending_eval_work_keys(entry, ["harmfulness"], [42], cfg, "safety") == []
    pending = progression._pending_eval_work_keys(entry, ["harmfulness"], [42, 43], cfg, "safety")
    assert pending == [progression._eval_work_key("harmfulness", "conditioned", "self_conditioned", 43)]
    assert not progression._needs_eval(entry, ["harmfulness"], [42], cfg, "safety")
    for corrupted_completion in (None, {}, {**payload["completion"], "safety_steering": {"targets_sha256": "e" * 64, "strength": 0}}):
        corrupt = {**payload, "completion": corrupted_completion}
        assert not progression._eval_seed_payload_done(corrupt, eval_name="harmfulness", seed=42, cfg=cfg)


def test_missing_api_key_fails_before_self_conditioned_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    prepare_runtime = MagicMock()
    monkeypatch.setattr(progression, "prepare_harmfulness_self_conditioning", prepare_runtime)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        progression.run_self_conditioned_harmfulness_eval(model=None, tokenizer=None,
            device=t.device("cpu"), seed=42, cfg=_cfg(tmp_path), meta={}, feature_contract=None)
    prepare_runtime.assert_not_called()


@pytest.mark.parametrize("raw, error", [([], TypeError), ({}, ValueError),
    ({"safety_steering": None, "extra": 1}, ValueError)])
def test_self_conditioned_config_rejects_ambiguous_shapes(tmp_path, raw, error):
    cfg = _cfg(tmp_path)
    cfg.self_conditioned = raw
    with pytest.raises(error, match="self_conditioned"):
        progression._self_conditioned_cfg(cfg)


def test_self_conditioned_progression_refuses_loaded_decoder_before_generation(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(progression, "resolve_local_checkpoint_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(progression, "load_progression_model", lambda *args: (MagicMock(), object(), {}, None))
    generate = MagicMock()
    monkeypatch.setattr(progression, "run_self_conditioned_harmfulness_eval", generate)
    with pytest.raises(ValueError, match="requires an encoder-decoder checkpoint"):
        progression.evaluate_single_checkpoint(progression.CheckpointSpec(1, "safety", ""),
            t.device("cpu"), t.float32, ["harmfulness"], [42], cfg)
    generate.assert_not_called()


@pytest.mark.parametrize("scheduler", [progression._needs_eval, progression._pending_eval_work_keys])
def test_missing_self_conditioned_block_cannot_silently_skip_requested_harmfulness(tmp_path, scheduler):
    cfg = _cfg(tmp_path)
    cfg.self_conditioned = None
    with pytest.raises(ValueError, match="self_conditioned=.*safety_steering: null"):
        scheduler(None, ["harmfulness"], [42], cfg, "safety")
    assert not scheduler(None, ["core", "chat"], [42], cfg, "safety")


def test_missing_self_conditioned_block_refused_before_checkpoint_loading(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.self_conditioned = None
    load = MagicMock()
    monkeypatch.setattr(progression, "load_progression_model", load)
    with pytest.raises(ValueError, match="self_conditioned=.*safety_steering: null"):
        progression.evaluate_single_checkpoint(progression.CheckpointSpec(1, "safety", ""),
            t.device("cpu"), t.float32, ["harmfulness"], [42], cfg)
    load.assert_not_called()
