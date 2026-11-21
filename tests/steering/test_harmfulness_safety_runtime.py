"""Real fixed/turn harmlessness runtime admission, receipts, and exact zero traces."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch as t

from scripts.eval import harmfulness_eval
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.encoder_decoder import self_conditioned
from steering.conditional.encoder_decoder.self_conditioned import SelfCondScoreInterpolation
from steering.eval import chat_self_conditioned_generation
from steering.eval import harmfulness_self_conditioning as runtime
from steering.eval.chat_benchmark_targets import benchmark_fit_contract
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    save_context_cache,
)
from steering.eval.harmfulness_artifact import (
    build_harmfulness_responses_artifact,
    write_harmfulness_responses_artifact,
)
from steering.eval.harmfulness_generation import validate_generation_provenance
from steering.eval.harmfulness_safety_targets import write_safety_targets
from steering.eval.harmfulness_self_conditioning import (
    build_harmfulness_context_runtime,
    generate_harmfulness_response_self_conditioned,
    harmfulness_completion_provenance,
    prepare_harmfulness_self_conditioning,
)
from tests.scripts.eval.test_build_harmfulness_safety_target import (
    fitting_provenance,
    populations,
    prepare,
    source_provenance,
)
from tests.scripts.test_harmfulness_eval_compliance import CONDITIONED_PROVENANCE
from tests.steering.test_chat_eval_self_conditioned import (
    _assert_same_trace,
    _trace,
    fixed_fixture,
    selfcond_turn_fixture,
)
from tests.steering.test_chat_self_conditioned_generation import script_tokens
from tests.steering.test_harmfulness_generation_provenance import _self_conditioned_generation


def _runtime_fixture(tmp_path, monkeypatch, family, *, compacted=False):
    fixture = (
        fixed_fixture(tmp_path, monkeypatch, masked=compacted)
        if family == "fixed"
        else selfcond_turn_fixture(tmp_path, monkeypatch, clustered=compacted)
    )
    # Untrained tiny models can emit role markers. Script ordinary tokens so
    # the test observes full real forward/feature-refresh traces for both arms.
    script_tokens(monkeypatch, chat_self_conditioned_generation, fixture.tokenizer.encode("a"))
    prompt = fixture.tasks["GSM8K"][0]["messages"][0]["content"]
    ids = render_prompt_for_chat_completion(fixture.tokenizer, prompt).token_ids
    meta = {**fixture.meta, "checkpoint_dir": str(fixture.checkpoint_path.parent)}
    feature_contract = resolve_checkpoint_feature_contract(meta,
        n_sae_features=fixture.model.encoder.config.n_sae_features,
        surface="test", source_path=str(fixture.checkpoint_path))
    contract, feature_ids = benchmark_fit_contract(feature_contract=feature_contract,
        n_sae_features=fixture.model.encoder.config.n_sae_features, tokenizer=fixture.tokenizer,
        fixed_contract=fixture.contract if family == "fixed" else None,
        turn_contract=fixture.contract if family == "turn" else None)
    cfg = {
        "conditioning": {"context_cache_dir": fixture.cfg.self_conditioned.cache_dir},
        "self_conditioned": {"safety_steering": None},
        "self_cond_segment_size": None, "stats_path": fixture.cfg.get("stats_path"),
    }
    arguments = {"model": fixture.model, "tokenizer": fixture.tokenizer, "meta": meta,
        "feature_contract": feature_contract,
        "device": t.device("cpu"), "prompt_token_ids": [ids]}
    training, _, background = populations()
    _, preparation, population = prepare((training, [{"prompt": prompt}, {"prompt": prompt}], background),
        tokenizer=fixture.tokenizer)
    preparation["sources"] = source_provenance()
    target_path = tmp_path / "targets.safetensors"
    write_safety_targets(target_path, target=t.linspace(-0.8, 0.8, len(feature_ids)),
        feature_ids=feature_ids, manifest={"format_version": 1, "fit_contract": contract,
            "preparation": preparation, "fit_provenance": fitting_provenance(contract),
            "evaluation_prompt_keys": population}, tokenizer=fixture.tokenizer)
    return fixture, cfg, arguments, prompt, target_path


def _generate(setup, prompt):
    return generate_harmfulness_response_self_conditioned(setup=setup,
        stable_pool_index=0, prompt=prompt, pool="harmful", category="test",
        base_seed=42, max_new_tokens=5, temperature=0, top_k=None)


@pytest.mark.parametrize("family", ["fixed", "turn"])
@pytest.mark.parametrize("compacted", [False, True])
def test_zero_strength_preserves_real_forward_logits_and_selections(tmp_path, monkeypatch, family, compacted):
    fixture, cfg, arguments, prompt, path = _runtime_fixture(tmp_path, monkeypatch, family, compacted=compacted)
    setup = prepare_harmfulness_self_conditioning(cfg, **arguments)
    traces, slots = _trace(fixture, monkeypatch)
    baseline = _generate(setup, prompt)
    baseline_trace, baseline_slots = list(traces), copy.deepcopy(slots)
    cfg["self_conditioned"]["safety_steering"] = {"targets": str(path), "strength": 0}
    steered = prepare_harmfulness_self_conditioning(cfg, **arguments)
    traces.clear()
    slots.clear()
    assert _generate(steered, prompt) == baseline
    _assert_same_trace(baseline_trace, traces)
    assert baseline_slots == slots
    assert harmfulness_completion_provenance(setup)["safety_steering"] is None
    bound = harmfulness_completion_provenance(steered)["safety_steering"]
    assert bound["strength"] == 0
    assert steered.safety_steering is not None
    assert bound["targets_sha256"] == steered.safety_steering.targets.sha256
    assert setup.policy.segment_mode == steered.policy.segment_mode
    setup.assert_target_unchanged()


@pytest.mark.parametrize("family", ["fixed", "turn"])
@pytest.mark.parametrize("strength", [2.5, 6.0])
def test_nonzero_policy_runs_through_real_cache_and_refresh(tmp_path, monkeypatch, family, strength):
    fixture, cfg, arguments, prompt, path = _runtime_fixture(tmp_path, monkeypatch, family, compacted=True)
    cfg["self_conditioned"]["safety_steering"] = {"targets": str(path), "strength": strength}
    setup = prepare_harmfulness_self_conditioning(cfg, **arguments)
    traces, slots = _trace(fixture, monkeypatch)
    ranked_scores = []
    select = self_conditioned.select_topk_bottoml

    def record_scores(scores, *args, **kwargs):
        ranked_scores.append(scores.clone())
        return select(scores, *args, **kwargs)

    monkeypatch.setattr(self_conditioned, "select_topk_bottoml", record_scores)
    assert isinstance(_generate(setup, prompt), str)
    assert traces and slots
    assert isinstance(setup.policy.score_policy, SelfCondScoreInterpolation)
    assert float(setup.policy.score_policy.weight) == strength / 6
    assert len(ranked_scores) == len(slots)
    if strength == 6:
        targets = setup.policy.score_policy.target_scores
        original_ids = fixture.model.feature_vocab.original_ids
        for scores, (positive, negative) in zip(ranked_scores, slots, strict=True):
            assert t.equal(scores, targets.unsqueeze(0))
            assert positive == [original_ids[index] for index in
                t.argsort(targets, descending=True, stable=True).tolist()[:len(positive)]]
            assert negative == [original_ids[index] for index in
                t.argsort(targets, stable=True).tolist()[:len(negative)]]
    generation = _self_conditioned_generation()
    generation["completion"] = harmfulness_completion_provenance(setup)
    assert validate_generation_provenance(generation)["completion"]["safety_steering"]["strength"] == strength


@pytest.mark.parametrize("fault", ["overlap", "identity", "changed_bytes", "cadence"])
def test_safety_admission_and_mutation_fail_before_forward(tmp_path, monkeypatch, fault):
    fixture, cfg, arguments, prompt, path = _runtime_fixture(tmp_path, monkeypatch, "turn")
    cfg["self_conditioned"]["safety_steering"] = {"targets": str(path), "strength": 2.5}
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("forward before admission"))
    if fault == "overlap":
        arguments["prompt_token_ids"] = [[9, 8, 7]]
    elif fault == "identity":
        arguments["feature_contract"] = replace(arguments["feature_contract"], feature_mask=None)
        # A changed checkpoint statistic is authoritative before bundle admission.
        wrong = tmp_path / "wrong-statistics"
        wrong.write_bytes(b"changed statistics")
        cfg["stats_path"] = str(wrong)
    elif fault == "cadence":
        cfg["self_cond_segment_size"] = 32
    else:
        setup = prepare_harmfulness_self_conditioning(cfg, **arguments)
        path.write_bytes(b"changed target bytes")
        with pytest.raises(ValueError, match="changed"):
            _generate(setup, prompt)
        return
    with pytest.raises((ValueError, RuntimeError)):
        prepare_harmfulness_self_conditioning(cfg, **arguments)


@pytest.mark.parametrize("fault", ["missing_digest", "changed_strength", "mixed_features", "nonfinite", "unsteered_strength",
    "invalid_completion", "invalid_segment_size", "out_of_range", "boolean_strength", "invalid_safety", "boolean_safety_strength"])
def test_receipt_rejects_incomplete_or_mixed_safety_binding(fault):
    generation = _self_conditioned_generation()
    completion = generation["completion"]
    completion.update(strength=2.5, safety_steering={"targets_sha256": "a" * 64, "strength": 2.5})
    if fault == "missing_digest":
        del completion["safety_steering"]["targets_sha256"]
    elif fault == "changed_strength":
        completion["safety_steering"]["strength"] = 0
    elif fault == "mixed_features":
        generation["feature_conditions"] = {1: 1}
    elif fault == "nonfinite":
        completion["strength"] = float("nan")
    elif fault == "invalid_completion":
        generation["completion"] = []
    elif fault == "invalid_segment_size":
        completion["self_cond_segment_size"] = True
    elif fault == "out_of_range":
        completion["strength"] = 7
    elif fault == "boolean_strength":
        completion["strength"] = True
    elif fault == "invalid_safety":
        completion["safety_steering"] = []
    elif fault == "boolean_safety_strength":
        completion["strength"] = 1
        completion["safety_steering"]["strength"] = True
    else:
        completion["safety_steering"] = None
    with pytest.raises(ValueError):
        validate_generation_provenance(generation)


@pytest.mark.parametrize("changed", ["digest", "strength"])
def test_saved_response_cache_binds_safety_digest_and_strength(tmp_path, monkeypatch, changed):
    generation = _self_conditioned_generation()
    generation["completion"].update(strength=2.5,
        safety_steering={"targets_sha256": "a" * 64, "strength": 2.5})
    pairs: list[dict[str, object]] = [{"prompt": "p", "response": "r", "pool": "harmful", "category": "test"}]
    artifact = build_harmfulness_responses_artifact(responses=pairs, conditioning=CONDITIONED_PROVENANCE,
        generation=generation)
    path = tmp_path / "responses.yaml"
    write_harmfulness_responses_artifact(artifact, path)
    monkeypatch.setattr(harmfulness_eval, "_load_expected_prompt_pairs", lambda cfg: pairs)
    assert harmfulness_eval._load_responses(str(path), {}, expected_generation=generation)[2] == generation
    changed_generation = copy.deepcopy(generation)
    if changed == "digest":
        changed_generation["completion"]["safety_steering"]["targets_sha256"] = "b" * 64
    else:
        changed_generation["completion"]["strength"] = 0
        changed_generation["completion"]["safety_steering"]["strength"] = 0
    with pytest.raises(ValueError, match="generation provenance does not match"):
        harmfulness_eval._load_responses(str(path), {}, expected_generation=changed_generation)


@pytest.mark.parametrize("fault", ["next_routing", "missing_conditioning", "missing_directory", "short_prompt", "overlength_prompt", "missing_prompt"])
def test_prompt_and_cache_admission_fail_before_forward(tmp_path, monkeypatch, fault):
    fixture, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, "fixed")
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("forward before admission"))
    if fault == "next_routing":
        arguments["meta"]["user_config"]["next_segment_conditioning_offset"] = True
        match = "current-segment prompt routing"
    elif fault == "missing_conditioning":
        cfg.pop("conditioning")
        match = "conditioning.context_cache_dir"
    elif fault == "missing_directory":
        cfg["conditioning"].pop("context_cache_dir")
        match = "conditioning.context_cache_dir"
    elif fault == "short_prompt":
        arguments["prompt_token_ids"] = [[1]]
        match = "empty or overlength"
    elif fault == "overlength_prompt":
        arguments["prompt_token_ids"] = [[1] * (fixture.model.dec_config.sequence_len + 1)]
        match = "empty or overlength"
    else:
        arguments["prompt_token_ids"] = [[1, 2, 3]]
        match = "no real context cache row"
    with pytest.raises(ValueError, match=match):
        prepare_harmfulness_self_conditioning(cfg, **arguments)


@pytest.mark.parametrize("field", ["feature_vocab", "feature_clusters_canonical"])
def test_compact_turn_cache_rejects_changed_contract_before_forward(tmp_path, monkeypatch, field):
    fixture, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, "turn", compacted=True)
    path = fixture.cache / "manifest.json"
    manifest = json.loads(path.read_text())
    # Change only the cache's authority; checkpoint and shard data stay intact.
    if field == "feature_vocab":
        manifest["turn_contract"][field]["sha256"] = "f" * 64
    else:
        manifest["turn_contract"][field] = None
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("forward with changed cache contract"))
    with pytest.raises(ContextCacheValidationError, match="turn context contract"):
        prepare_harmfulness_self_conditioning(cfg, **arguments)


def test_fixed_cache_rejects_ids_outside_checkpoint_mask_before_forward(tmp_path, monkeypatch):
    fixture, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, "fixed", compacted=True)
    context = build_harmfulness_context_runtime(cfg, model=fixture.model,
        tokenizer=fixture.tokenizer, meta=arguments["meta"], feature_contract=arguments["feature_contract"])
    assert fixture.model.feature_vocab is not None
    excluded = next(index for index in range(fixture.model.encoder.config.n_sae_features)
        if index not in fixture.model.feature_vocab.original_ids)
    entries = [ContextCacheEntry(tuple(ids), tuple(replace(segment,
        feature_ids=(excluded, *segment.feature_ids[1:])) for segment in segments))
        for ids, segments in context.provider._rows.values()]
    assert context.fixed_contract is not None
    save_context_cache(fixture.cache, entries, identity=context.fixed_contract.identity,
        n_sae_features=fixture.model.encoder.config.n_sae_features,
        num_buckets=fixture.model.encoder.config.num_buckets,
        metadata=context.provider.manifest["metadata"], overwrite=True)
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("forward with excluded cache IDs"))
    with pytest.raises(ValueError, match="outside checkpoint vocabulary"):
        prepare_harmfulness_self_conditioning(cfg, **arguments)


@pytest.mark.parametrize("family", ["fixed", "turn"])
def test_generated_prompt_cannot_escape_admitted_population(tmp_path, monkeypatch, family):
    fixture, cfg, arguments, prompt, path = _runtime_fixture(tmp_path, monkeypatch, family)
    cfg["self_conditioned"]["safety_steering"] = {"targets": str(path), "strength": 0}
    setup = prepare_harmfulness_self_conditioning(cfg, **arguments)
    monkeypatch.setattr(runtime, "build_engine_self_conditioned",
        lambda *args, **kwargs: pytest.fail("engine before held-out prompt validation"))
    with pytest.raises(ValueError, match="not excluded from fitting"):
        _generate(setup, prompt + "different prompt")


@pytest.mark.parametrize("family", ["fixed", "turn"])
@pytest.mark.parametrize("field,changed", [
    ("model_id", "different-model"), ("sae_id", "different-sae"),
    ("layer", -1), ("layer", True), ("pooling", "different-pooling"),
    ("feature_selection_hash", "f" * 64),
])
def test_explicit_cache_identity_conflicts_fail_before_provider_or_forward(
    tmp_path, monkeypatch, family, field, changed,
):
    fixture, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, family)
    cfg["conditioning"]["identity"] = {field: changed}
    monkeypatch.setattr(runtime, "SafetensorsContextFeatureProvider",
        lambda *args, **kwargs: pytest.fail("cache opened before explicit identity assertion"))
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("forward before admission"))
    with pytest.raises(ValueError, match=rf"conditioning\.identity\.{field} conflicts"):
        prepare_harmfulness_self_conditioning(cfg, **arguments)


@pytest.mark.parametrize("family", ["fixed", "turn"])
@pytest.mark.parametrize("assertions", [[], "cache", {"typo": None}])
def test_malformed_cache_identity_assertions_fail_loud(tmp_path, monkeypatch, family, assertions):
    _, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, family)
    cfg["conditioning"]["identity"] = assertions
    with pytest.raises(ValueError, match="conditioning.identity must contain only cache identity fields"):
        prepare_harmfulness_self_conditioning(cfg, **arguments)


@pytest.mark.parametrize("family", ["fixed", "turn"])
def test_matching_partial_and_null_cache_identity_assertions_are_accepted(tmp_path, monkeypatch, family):
    _, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, family)
    baseline = prepare_harmfulness_self_conditioning(cfg, **arguments)
    assert isinstance(baseline.context_provider, runtime.SafetensorsContextFeatureProvider)
    expected = baseline.context_provider.identity.as_dict()
    for assertions in (expected, {"model_id": expected["model_id"]}, dict.fromkeys(expected), {}):
        cfg["conditioning"]["identity"] = assertions
        setup = prepare_harmfulness_self_conditioning(cfg, **arguments)
        assert isinstance(setup.context_provider, runtime.SafetensorsContextFeatureProvider)
        assert setup.context_provider.identity == baseline.context_provider.identity


@pytest.mark.parametrize("family", ["fixed", "turn"])
@pytest.mark.parametrize("remaining_capacity", [0, 1])
def test_cached_prompt_requires_room_for_generation_before_forward(
    tmp_path, monkeypatch, family, remaining_capacity,
):
    fixture, cfg, arguments, _, _ = _runtime_fixture(tmp_path, monkeypatch, family)
    context = build_harmfulness_context_runtime(cfg, model=fixture.model,
        tokenizer=fixture.tokenizer, meta=arguments["meta"], feature_contract=arguments["feature_contract"])
    ids = arguments["prompt_token_ids"][0]
    assert context.provider.get(ids)
    # Keep the independently authenticated real cache while exercising the
    # model's exact admission boundary, without rebuilding unrelated artifacts.
    fixture.model.dec_config.sequence_len = len(ids) + remaining_capacity
    monkeypatch.setattr(runtime, "build_harmfulness_context_runtime", lambda *args, **kwargs: context)
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("forward before admission"))
    if remaining_capacity == 0:
        with pytest.raises(ValueError, match="no generation budget"):
            prepare_harmfulness_self_conditioning(cfg, **arguments)
    else:
        setup = prepare_harmfulness_self_conditioning(cfg, **arguments)
        assert setup.context_provider is context.provider
