"""Artifact admission and real fixed/turn ChatCORE behavior for benchmark targets."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.eval import chat_eval
from scripts.eval.merge_chat_eval_target_real import merge_target_real_results
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.sampling import SamplingBundle
from steering.eval import chat_self_conditioned_generation
from steering.eval.chat_benchmark_targets import (
    TARGET_METADATA_KEY,
    BenchmarkSteeringSettings,
    benchmark_fit_contract,
    decode_benchmark_targets,
    load_benchmark_targets,
    write_benchmark_targets,
)
from steering.eval.chat_eval_result_identity import validate_self_conditioned_run_identity
from steering.eval.chat_generation import render_chat_completion
from steering.eval.chat_self_conditioning import ChatSelfConditionedSettings, parse_chat_eval_mode
from steering.eval.target_conditioning import token_context_key
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.immutable_io import ImmutableArtifactError
from steering.safetensors_io import deterministic_safetensors_bytes
from tests.steering.test_chat_eval_self_conditioned import (
    _assert_same_trace,
    _result,
    _trace,
    fixed_fixture,
    selfcond_turn_fixture,
)
from tests.steering.test_chat_self_conditioned_generation import script_tokens
from tests.steering.test_self_conditioned_interpolation import _config


@pytest.fixture
def artifact_parts():
    names = ("MMLU", "GSM8K")
    ids = t.tensor([1, 3, 5], dtype=t.int64)
    tensors = {"feature_ids": ids, "target.MMLU": t.tensor([0.5, 0.0, -0.5]),
               "target.GSM8K": t.tensor([-0.2, 0.3, 0.6])}
    manifest = {"format_version": 1, "fit_contract": {"statistics_sha256": "a" * 64},
        "preparation": {"source": "local fixture"}, "fit_provenance": {"code": "frozen"},
        "evaluation_prompt_keys": {name: [token_context_key([1, 2, 3])] for name in names}}
    return names, ids, tensors, manifest


def encoded(tensors, manifest):
    return deterministic_safetensors_bytes(tensors,
        metadata={TARGET_METADATA_KEY: canonical_turn_payload(manifest).decode()})


def decode(parts, *, tensors=None, manifest=None):
    names, ids, original_tensors, original_manifest = parts
    return decode_benchmark_targets(encoded(original_tensors if tensors is None else tensors,
        original_manifest if manifest is None else manifest), expected_contract=original_manifest["fit_contract"],
        expected_feature_ids=ids, task_names=names)


@pytest.mark.parametrize("strength", [0, -0.0, 0.125, 2.5, 6, 6.0])
def test_continuous_strength(strength):
    settings = BenchmarkSteeringSettings.parse(OmegaConf.create({"targets": "/targets", "strength": strength}))
    assert settings.targets == Path("/targets")
    assert settings.strength == float(strength)
    if strength == 0:
        assert str(settings.strength) == "0.0"


@pytest.mark.parametrize("strength", [True, False, "2.5", None, float("nan"), float("inf"), -float("inf"), -0.1, 6.01, 10**1000])
def test_invalid_strength(strength):
    with pytest.raises(ValueError, match="finite number"):
        BenchmarkSteeringSettings.parse({"targets": "/targets", "strength": strength})


@pytest.mark.parametrize("value", [None, [], "target", {}, {"targets": "a"}, {"strength": 0},
    {"targets": "a", "strength": 0, "typo": 1}, {"targets": "", "strength": 0},
    {"targets": "  ", "strength": 0}, {"targets": 3, "strength": 0}])
def test_invalid_setting_shape(value):
    with pytest.raises(ValueError):
        BenchmarkSteeringSettings.parse(value)


@pytest.mark.parametrize("mode", [None, "unconditioned", "target_real"])
def test_steering_cannot_be_ignored_by_legacy_or_hf_dispatch(mode, monkeypatch):
    cfg = OmegaConf.load(Path(__file__).parents[2] / "configs/eval/chat_eval.yaml")
    cfg.evaluation_mode = mode
    cfg.self_conditioned.benchmark_steering = {"targets": "/missing", "strength": 1}
    monkeypatch.setattr(chat_eval, "load_model", lambda *a, **kw: pytest.fail("model load before settings validation"))
    monkeypatch.setattr(chat_eval, "_run_hf_chat_eval", lambda *a, **kw: pytest.fail("HF dispatch ignored steering"))
    with pytest.raises(ValueError, match="requires evaluation_mode"):
        chat_eval.main.__wrapped__(cfg)


def test_null_block_preserves_legacy_mode():
    cfg = OmegaConf.create({"evaluation_mode": None, "self_conditioned": {"benchmark_steering": None}})
    assert parse_chat_eval_mode(cfg).value == "unconditioned"


@pytest.mark.parametrize("strength", ["0", "2.5", "6"])
def test_public_hydra_preset_resolves_existing_mode_and_continuous_override(strength):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[2] / "configs/eval"), version_base=None):
        cfg = compose(config_name="chat_eval_benchmark_steering", overrides=[
            "self_conditioned.cache_dir=/cache", "self_conditioned.benchmark_steering.targets=/targets",
            f"self_conditioned.benchmark_steering.strength={strength}"])
    settings = ChatSelfConditionedSettings.parse(cfg)
    assert parse_chat_eval_mode(cfg).value == "self_conditioned"
    assert settings.benchmark_steering is not None
    assert settings.benchmark_steering.targets == Path("/targets")
    assert settings.benchmark_steering.strength == float(strength)
    assert cfg.batch_size == cfg.num_samples == 1
    assert cfg.target_real_configs == cfg.feature_configs == []


def test_actual_bytes_roundtrip_replay_and_snapshot(tmp_path, artifact_parts):
    names, ids, tensors, manifest = artifact_parts
    targets = {name: tensors[f"target.{name}"] for name in names}
    path = tmp_path / "targets.safetensors"
    bundle = write_benchmark_targets(path, targets=targets, feature_ids=ids, manifest=manifest, task_names=names)
    original = path.read_bytes()
    assert bundle.sha256 == hashlib.sha256(original).hexdigest()
    assert write_benchmark_targets(path, targets=targets, feature_ids=ids, manifest=manifest, task_names=names).sha256 == bundle.sha256
    assert path.read_bytes() == original
    link = tmp_path / "linked.safetensors"
    link.symlink_to(path)
    loaded = load_benchmark_targets(link, expected_contract=manifest["fit_contract"], expected_feature_ids=ids, task_names=names)
    path.write_bytes(b"changed after loading")
    tensors["target.MMLU"].zero_()
    assert t.equal(loaded.targets["MMLU"], t.tensor([0.5, 0.0, -0.5]))
    loaded.validate_example("MMLU", 1, 0, [1, 2, 3])
    with pytest.raises(ImmutableArtifactError):
        write_benchmark_targets(path, targets=targets, feature_ids=ids, manifest=manifest, task_names=names)


@pytest.mark.parametrize("fault", ["missing", "extra", "ids_order", "ids_dtype", "ids_rank", "width", "rank", "dtype", "nan", "inf", "lower", "upper"])
def test_tensor_contract_rejects_independent_corruption(artifact_parts, fault):
    tensors = {key: value.clone() for key, value in artifact_parts[2].items()}
    if fault == "missing":
        del tensors["target.GSM8K"]
    elif fault == "extra":
        tensors["target.Unknown"] = t.zeros(3)
    elif fault == "ids_order":
        tensors["feature_ids"] = tensors["feature_ids"].flip(0)
    elif fault == "ids_dtype":
        tensors["feature_ids"] = tensors["feature_ids"].to(t.int32)
    elif fault == "ids_rank":
        tensors["feature_ids"] = tensors["feature_ids"].unsqueeze(0)
    elif fault == "width":
        tensors["target.MMLU"] = t.zeros(2)
    elif fault == "rank":
        tensors["target.MMLU"] = t.zeros(1, 3)
    elif fault == "dtype":
        tensors["target.MMLU"] = tensors["target.MMLU"].double()
    else:
        tensors["target.MMLU"][0] = {"nan": float("nan"), "inf": float("inf"), "lower": -1.0, "upper": 1.0}[fault]
    with pytest.raises(ValueError, match="benchmark target"):
        decode(artifact_parts, tensors=tensors)


@pytest.mark.parametrize("fault", ["section", "version_bool", "version_unknown", "contract", "empty_preparation", "empty_provenance", "tasks", "empty_keys", "key_type", "key_hash"])
def test_manifest_contract_rejects_independent_corruption(artifact_parts, fault):
    manifest = copy.deepcopy(artifact_parts[3])
    if fault == "section":
        del manifest["fit_provenance"]
    elif fault == "version_bool":
        manifest["format_version"] = True
    elif fault == "version_unknown":
        manifest["format_version"] = 2
    elif fault == "contract":
        manifest["fit_contract"]["statistics_sha256"] = "b" * 64
    elif fault == "empty_preparation":
        manifest["preparation"] = {}
    elif fault == "empty_provenance":
        manifest["fit_provenance"] = None
    elif fault == "tasks":
        del manifest["evaluation_prompt_keys"]["GSM8K"]
    else:
        manifest["evaluation_prompt_keys"]["MMLU"] = {"empty_keys": [], "key_type": [3], "key_hash": ["a" * 64]}[fault]
    with pytest.raises(ValueError):
        decode(artifact_parts, manifest=manifest)


def test_nested_json_duplicates_and_missing_header_rejected(artifact_parts):
    names, ids, tensors, manifest = artifact_parts
    arguments = {"expected_contract": manifest["fit_contract"], "expected_feature_ids": ids, "task_names": names}
    for metadata in (None, {}, {TARGET_METADATA_KEY: canonical_turn_payload(manifest).decode().replace('"format_version":1', '"format_version":0,"format_version":1')}):
        with pytest.raises(ValueError):
            decode_benchmark_targets(deterministic_safetensors_bytes(tensors, metadata=metadata), **arguments)
    with pytest.raises(ValueError):
        decode_benchmark_targets(b"broken", **arguments)


@pytest.mark.parametrize("size,index,tokens", [(2, 0, [1, 2, 3]), (1, -1, [1, 2, 3]), (1, 1, [1, 2, 3]), (1, 0, [1, 2, 4])])
def test_original_index_population_guard(artifact_parts, size, index, tokens):
    with pytest.raises(ValueError, match="exclusion population"):
        decode(artifact_parts).validate_example("MMLU", size, index, tokens)


def test_task_policy_snapshots_target_and_preserves_base_rng(artifact_parts):
    bundle = decode(artifact_parts)
    base = _config(t.zeros(6), 0)
    assert isinstance(base.selection, SamplingBundle)
    state = base.selection.rng.get_state().clone()
    policy = bundle.policy(base, "MMLU", 2.5)
    assert float(policy.score_policy.weight) == 2.5 / 6
    assert policy.selection is base.selection
    assert t.equal(state, base.selection.rng.get_state())
    bundle.targets["MMLU"].zero_()
    assert t.equal(policy.score_policy.target_scores, t.tensor([0.5, 0.0, -0.5]))
    with pytest.raises(ValueError, match="existing interpolation"):
        bundle.policy(replace(base, score_policy=None), "MMLU", 2.5)


def runtime_bundle(fixture, path, *, sign=1):
    width = fixture.model.encoder.config.n_sae_features
    features = resolve_checkpoint_feature_contract(fixture.meta, n_sae_features=width, surface="test", source_path=str(fixture.checkpoint_path))
    is_fixed = hasattr(fixture.contract, "source")
    contract, ids = benchmark_fit_contract(feature_contract=features, n_sae_features=width, tokenizer=fixture.tokenizer,
        fixed_contract=fixture.contract if is_fixed else None, turn_contract=None if is_fixed else fixture.contract)
    actual_tasks = {name: chat_eval._chat_task(name) for name in chat_eval.ALL_CHAT_TASKS}
    population = {name: [token_context_key(render_chat_completion(fixture.tokenizer, task[index]).token_ids)
        for index in range(len(task))] for name, task in actual_tasks.items()}
    manifest = {"format_version": 1, "fit_contract": contract, "evaluation_prompt_keys": population,
                "preparation": {"source": "native fixture tasks"}, "fit_provenance": {"source": "known targets"}}
    targets = {name: sign * t.linspace(-0.8, 0.8, len(ids)) for name in fixture.tasks}
    write_benchmark_targets(path, targets=targets, feature_ids=ids, manifest=manifest, task_names=chat_eval.ALL_CHAT_TASKS)
    return manifest, ids, targets


@pytest.fixture(params=["fixed", "turn"])
def family(request, tmp_path, monkeypatch):
    return (fixed_fixture if request.param == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch, gsm_problems=3)


def test_zero_strength_exact_real_evaluator_trace_all_tasks(family, tmp_path, monkeypatch):
    traces, slots = _trace(family, monkeypatch)
    chat_eval.main.__wrapped__(family.cfg)
    baseline_trace, baseline_slots = list(traces), copy.deepcopy(slots)
    baseline = yaml.safe_load(_result(family).read_text())["seeds"]["adhoc"]
    for sign in (1, -1):
        path = tmp_path / f"targets-{sign}.safetensors"
        runtime_bundle(family, path, sign=sign)
        family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": 0}
        steering = ChatSelfConditionedSettings.parse(family.cfg).benchmark_steering
        assert steering is not None and steering.strength == 0.0
        family.cfg.self_conditioned_output_dir = str(tmp_path / f"zero-{sign}")
        traces.clear()
        slots.clear()
        chat_eval.main.__wrapped__(family.cfg)
        _assert_same_trace(baseline_trace, traces)
        assert baseline_slots == slots
        result = yaml.safe_load(_result(family).read_text())["seeds"]["adhoc"]
        assert result["results"] == baseline["results"]
        assert result["task_details"] == baseline["task_details"]
        cond = result["run_identity"]["conditioning"]
        assert cond["score_policy"]["weight"] == 0
        assert cond["score_policy"]["ranking"] == "raw_float32_logits"
        assert cond["benchmark_steering"]["targets_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("strength", [2.5, 6])
def test_nonzero_policy_reaches_s0_and_every_refresh(family, tmp_path, monkeypatch, strength):
    path = tmp_path / "targets.safetensors"
    runtime_bundle(family, path)
    baseline_traces, baseline_slots = _trace(family, monkeypatch)
    chat_eval.main.__wrapped__(family.cfg)
    old = list(baseline_traces)
    baseline_traces.clear()
    baseline_slots.clear()
    family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": strength}
    family.cfg.self_conditioned_output_dir = str(tmp_path / "steered")
    chat_eval.main.__wrapped__(family.cfg)
    assert len(baseline_slots) > len(chat_eval.ALL_CHAT_TASKS)
    assert any(not t.equal(a[2], b[2]) for a, b in zip(old, baseline_traces, strict=False))
    if strength == 6:
        for positive, negative in baseline_slots:
            assert set(positive) == set(range(8 - len(positive), 8))
            assert set(negative) == set(range(len(negative)))
    body = yaml.safe_load(_result(family).read_text())["seeds"]["adhoc"]
    assert body["run_identity"]["conditioning"]["score_policy"]["weight"] == strength / 6


def test_each_task_uses_its_own_target_for_every_generated_slot(family, tmp_path, monkeypatch):
    manifest, ids, targets = runtime_bundle(family, tmp_path / "original.safetensors")
    targets["GSM8K"].neg_()
    path = tmp_path / "opposing.safetensors"
    write_benchmark_targets(path, targets=targets, feature_ids=ids, manifest=manifest,
                            task_names=chat_eval.ALL_CHAT_TASKS)
    _, slots = _trace(family, monkeypatch)
    family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": 6}
    family.cfg.max_problems = 1
    family.cfg.task_name = "MMLU|GSM8K"
    chat_eval.main.__wrapped__(family.cfg)
    assert len(slots) >= 2
    # MMLU scores one categorical marker (one S0). All subsequent slots belong
    # to GSM8K, including any calculator/generative refreshes in its trajectory.
    for index, (positive, negative) in enumerate(slots):
        high, low = (positive, negative) if index == 0 else (negative, positive)
        assert set(high) == set(range(8 - len(high), 8)), index
        assert set(low) == set(range(len(low))), index


def test_benchmark_target_survives_calculator_exchange_and_resumed_generation(family, tmp_path, monkeypatch):
    path = tmp_path / "targets.safetensors"
    runtime_bundle(family, path)
    tok = family.tokenizer
    special = tok.encode_special
    scripted = [special("<python_start>"), *tok.encode("1+1"), special("<python_end>"),
        special("<output_start>"), *tok.encode("2"), special("<output_end>"),
        *tok.encode(" resumed" * 10), special("<assistant_end>")]
    script_tokens(monkeypatch, chat_self_conditioned_generation, scripted)
    traces, slots = _trace(family, monkeypatch)
    predictions = []
    task_type = type(family.tasks["GSM8K"])
    evaluate = task_type.evaluate
    def capture(self, conversation, prediction):
        predictions.append(prediction)
        return evaluate(self, conversation, prediction)
    monkeypatch.setattr(task_type, "evaluate", capture)
    family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": 6}
    family.cfg.task_name, family.cfg.max_problems = "GSM8K", 1
    family.cfg.max_new_tokens = len(scripted) + 1
    chat_eval.main.__wrapped__(family.cfg)
    assert predictions == [tok.decode(scripted[:-1])]
    assert len(slots) > 3
    assert any(special("<output_end>") in row for ids, _, _ in traces for row in ids)
    for positive, negative in slots:
        assert set(positive) == set(range(8 - len(positive), 8))
        assert set(negative) == set(range(len(negative)))


@pytest.mark.parametrize("fault", ["population", "contract", "feature_ids", "tensor"])
def test_admission_failure_before_any_real_forward(family, tmp_path, monkeypatch, fault):
    path = tmp_path / "targets.safetensors"
    manifest, ids, targets = runtime_bundle(family, path)
    if fault == "population":
        manifest["evaluation_prompt_keys"]["MMLU"][0] = token_context_key([3])
    elif fault == "contract":
        manifest["fit_contract"]["statistics_sha256"] = "e" * 64
    elif fault == "feature_ids":
        ids = ids.flip(0)
    else:
        targets["MMLU"][0] = float("nan")
    path.write_bytes(encoded({"feature_ids": ids, **{f"target.{name}": value for name, value in targets.items()}}, manifest))
    family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": 0}
    monkeypatch.setattr(family.model, "forward", lambda *a, **kw: pytest.fail("invalid artifact reached model"))
    with pytest.raises(ValueError):
        chat_eval.main.__wrapped__(family.cfg)
    assert not _result(family).exists()


def test_steered_shards_merge_exactly_and_strength_mismatch_fails(family, tmp_path):
    path = tmp_path / "targets.safetensors"
    runtime_bundle(family, path)
    family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": 2.5}
    family.cfg.task_name, family.cfg.max_problems = "GSM8K", 3
    chat_eval.main.__wrapped__(family.cfg)
    baseline = yaml.safe_load(_result(family).read_text())
    paths = []
    for start, count in ((2, 1), (0, 2)):
        family.cfg.problem_start, family.cfg.max_problems = start, count
        family.cfg.self_conditioned_output_dir = str(tmp_path / f"shard-{start}")
        chat_eval.main.__wrapped__(family.cfg)
        paths.append(_result(family))
    merged = merge_target_real_results(paths, tmp_path / "merged.yaml", task_totals={"GSM8K": 3})
    assert merged["seeds"]["adhoc"] == baseline["seeds"]["adhoc"]
    altered = yaml.safe_load(paths[0].read_text())
    cond = altered["seeds"]["adhoc"]["run_identity"]["conditioning"]
    cond["benchmark_steering"]["strength"] = 3.0
    cond["score_policy"]["weight"] = 0.5
    paths[0].write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError, match="disagree"):
        merge_target_real_results(paths, tmp_path / "bad.yaml", task_totals={"GSM8K": 3})
    assert not (tmp_path / "bad.yaml").exists()


@pytest.mark.parametrize("fault", ["weight", "ranking", "digest", "missing", "pure", "bool"])
def test_receipt_cannot_misdescribe_effective_policy(family, tmp_path, fault):
    path = tmp_path / "targets.safetensors"
    runtime_bundle(family, path)
    family.cfg.self_conditioned.benchmark_steering = {"targets": str(path), "strength": 2.5}
    family.cfg.task_name = "MMLU"
    chat_eval.main.__wrapped__(family.cfg)
    identity = yaml.safe_load(_result(family).read_text())["seeds"]["adhoc"]["run_identity"]
    cond = identity["conditioning"]
    if fault in ("weight", "ranking"):
        cond["score_policy"][fault] = 0.0 if fault == "weight" else "raw_float32_logits"
    elif fault == "digest":
        cond["benchmark_steering"]["targets_sha256"] = "bad"
    elif fault == "missing":
        del cond["benchmark_steering"]
    elif fault == "pure":
        cond["selection_policy"] = "axbench_pure_model"
    else:
        cond["benchmark_steering"]["strength"] = True
    with pytest.raises(ValueError, match="benchmark"):
        validate_self_conditioned_run_identity(identity)
