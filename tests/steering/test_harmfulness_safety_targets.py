"""Safety bundle admission, immutable snapshots, and shared interpolation policy."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.conditional.encoder_decoder.self_conditioned as selfcond_module
from steering.conditional.encoder_decoder.self_conditioned import SelfConditionedFeatureProvider
from steering.conditional.sampling import SamplingBundle
from steering.eval.chat_benchmark_targets import TARGET_METADATA_KEY
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.harmfulness_safety_targets import (
    PromptHarmLabel,
    ResponseRefusalLabel,
    SafetySteeringSettings,
    SafetySubpopulation,
    decode_safety_targets,
    load_safety_targets,
    validate_safety_preparation,
    write_safety_targets,
)
from steering.eval.target_conditioning import token_context_key
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.immutable_io import ImmutableArtifactError
from steering.safetensors_io import deterministic_safetensors_bytes
from steering.segmentation import compute_tokenizer_checksum
from tests.scripts.eval.test_build_harmfulness_safety_target import (
    fitting_provenance,
    populations,
    prepare,
    safe_row,
    source_provenance,
)
from tests.steering._self_conditioned_state import build_test_self_cond_state
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.test_self_conditioned_interpolation import _config, _model


@pytest.fixture
def artifact_parts():
    _, preparation, population = prepare(populations())
    preparation["sources"] = source_provenance()
    ids = t.arange(6, dtype=t.int64)
    target = t.tensor([0.9, 0.7, 0.5, -0.5, -0.7, -0.9])
    sae = {"model": "gemma-3-4b-pt", "release": "gemmascope_2", "activation_type": "residual",
           "layer": 22, "width": "65k", "l0": "medium"}
    manifest = {"format_version": 1,
                "fit_contract": {"statistics_sha256": "a" * 64, "segment_mode": "turn_delimited_fixed_tokens",
                                 "source": {"source_segment_size": 32, "rustbpe_cadence": 33,
                                            "source_lineage": sae, "source_context_window": 4096,
                                            "extraction_tokenizer_checksum": "d" * 64},
                                 "decoder_tokenizer_checksum": compute_tokenizer_checksum(ByteChatTokenizer()), "n_sae_features": 6,
                                 "feature_mask": None, "feature_clusters": None, "feature_vocab": None,
                                 "missing_percentile_policy": "error"},
                "preparation": preparation, "evaluation_prompt_keys": population}
    manifest["fit_provenance"] = fitting_provenance(manifest["fit_contract"])
    return ids, target, manifest


def encoded(ids, target, manifest):
    return deterministic_safetensors_bytes({"feature_ids": ids, "target.safety": target},
        metadata={TARGET_METADATA_KEY: canonical_turn_payload(manifest).decode()})


def decode(parts, *, ids=None, target=None, manifest=None):
    original_ids, original_target, original_manifest = parts
    return decode_safety_targets(encoded(original_ids if ids is None else ids,
        original_target if target is None else target, original_manifest if manifest is None else manifest),
        expected_contract=original_manifest["fit_contract"], expected_feature_ids=original_ids, tokenizer=ByteChatTokenizer())


@pytest.mark.parametrize("strength", [0, -0.0, 0.125, 2.5, 6])
def test_settings_use_shared_strength_contract_and_safety_namespace(strength):
    settings = SafetySteeringSettings.parse(OmegaConf.create({"targets": "/targets", "strength": strength}))
    assert isinstance(settings, SafetySteeringSettings)
    assert settings.targets == Path("/targets")
    assert settings.strength == float(strength)
    if strength == 0:
        assert str(settings.strength) == "0.0"


@pytest.mark.parametrize("value", [None, [], {}, {"targets": "a"}, {"targets": "a", "strength": 0, "extra": 1},
    {"targets": " ", "strength": 0}, *({"targets": "a", "strength": strength}
        for strength in (True, None, "2.5", float("nan"), float("inf"), -0.1, 6.01))])
def test_invalid_settings_name_the_safety_block(value):
    with pytest.raises(ValueError, match="safety_steering"):
        SafetySteeringSettings.parse(value)


def test_bundle_bytes_roundtrip_replay_and_mutation_detection(tmp_path, artifact_parts):
    ids, target, manifest = artifact_parts
    path = tmp_path / "targets.safetensors"
    bundle = write_safety_targets(path, target=target, feature_ids=ids, manifest=manifest, tokenizer=ByteChatTokenizer())
    original = path.read_bytes()
    assert bundle.sha256 == hashlib.sha256(original).hexdigest()
    assert write_safety_targets(path, target=target, feature_ids=ids, manifest=manifest, tokenizer=ByteChatTokenizer()).sha256 == bundle.sha256
    bundle.assert_matches_file(path)
    link = tmp_path / "alias.safetensors"
    link.symlink_to(path)
    loaded = load_safety_targets(
        link,
        expected_contract=manifest["fit_contract"],
        expected_feature_ids=ids,
        tokenizer=ByteChatTokenizer(),
    )
    assert loaded.sha256 == bundle.sha256
    assert loaded.manifest == manifest == bundle.manifest
    with pytest.raises(ImmutableArtifactError):
        write_safety_targets(path, target=-target, feature_ids=ids, manifest=manifest, tokenizer=ByteChatTokenizer())
    path.write_bytes(encoded(ids, -target, manifest))
    with pytest.raises(ValueError, match="bytes changed"):
        bundle.assert_matches_file(path)
    # A validated in-memory snapshot stays bound to the original captured bytes.
    assert t.equal(bundle.bundle.targets["safety"], target)
    assert bundle.sha256 == hashlib.sha256(original).hexdigest()


def test_v1_bundle_is_rejected_with_explicit_refit_message(artifact_parts):
    manifest = copy.deepcopy(artifact_parts[2])
    report = manifest["preparation"]
    report["source_kind"] = "wildguard_train_safe_responses_v1"
    for field in ("fit_order", "shuffle_seed", "subpopulation_counts"):
        del report[field]
    for row in report["accepted"]["safety"]:
        for field in ("prompt_harm_label", "response_refusal_label", "subpopulation"):
            del row[field]
    with pytest.raises(ValueError, match="stale.*refit.*v2"):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("fault", ["count", "missing_count", "boolean_count", "population", "benign_refusal",
                                 "unknown_prompt", "unknown_refusal", "order", "seed"])
def test_v2_provenance_revalidates_rule_counts_and_order(artifact_parts, fault):
    manifest = copy.deepcopy(artifact_parts[2])
    report = manifest["preparation"]
    row = report["accepted"]["safety"][0]
    if fault == "count":
        report["subpopulation_counts"][SafetySubpopulation.SAFE_HARMFUL] += 1
    elif fault == "missing_count":
        del report["subpopulation_counts"][SafetySubpopulation.BENIGN_NON_REFUSAL]
    elif fault == "boolean_count":
        report["subpopulation_counts"][SafetySubpopulation.SAFE_HARMFUL] = True
    elif fault == "population":
        row["subpopulation"] = SafetySubpopulation.BENIGN_NON_REFUSAL
    elif fault == "benign_refusal":
        row.update(prompt_harm_label=PromptHarmLabel.BENIGN, response_refusal_label=ResponseRefusalLabel.REFUSAL,
                   subpopulation=SafetySubpopulation.BENIGN_NON_REFUSAL)
        report["subpopulation_counts"] = {SafetySubpopulation.SAFE_HARMFUL: 0, SafetySubpopulation.BENIGN_NON_REFUSAL: 1}
    elif fault == "unknown_prompt":
        row["prompt_harm_label"] = "benign"
    elif fault == "unknown_refusal":
        row["response_refusal_label"] = "not_refusal"
    elif fault == "order":
        report["fit_order"] = "stratified"
    else:
        report["shuffle_seed"] = 43
    with pytest.raises(ValueError):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("population", ["accepted", "excluded", "background", "background_excluded"])
def test_audit_rejects_source_order_that_disagrees_with_claimed_scan(population):
    _, evaluation, background = populations()
    training = [safe_row(f"Candidate {index}") for index in range(12)]
    background[:0] = [
        {"messages": [{"role": "user", "content": evaluation[0]["prompt"]},
                      {"role": "assistant", "content": "Background response"}]}
        for _ in range(2)
    ]
    _, preparation, _ = prepare((training, evaluation, background), positive_cap=4)
    rows = (preparation["accepted"]["safety"] if population == "accepted" else
            preparation["accepted"]["background"] if population == "background" else
            preparation[population])
    rows.reverse()
    with pytest.raises(ValueError, match="source order"):
        validate_safety_preparation(preparation)


@pytest.mark.parametrize("field,value", [("statistics_sha256", "c" * 64),
    ("decoder_tokenizer_checksum", "d" * 64), ("source", {"segment_size": 64, "rustbpe_cadence": 33}),
    ("segment_mode", "fixed_tokens"), ("feature_mask", {"allow": [1, 2]}),
    ("feature_clusters", {"sha256": "e" * 64}), ("feature_vocab", {"original_ids": [5, 4, 3, 2, 1, 0]}),
    ("n_sae_features", 7), ("missing_percentile_policy", "zero")])
def test_every_checkpoint_identity_dimension_is_admitted_before_policy(artifact_parts, field, value):
    manifest = copy.deepcopy(artifact_parts[2])
    manifest["fit_contract"][field] = value
    with pytest.raises(ValueError, match="fit contract disagrees"):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("fault", ["reversed_ids", "wrong_ids", "float_ids", "empty_ids", "target_width",
    "target_rank", "target_dtype", "nan", "infinite", "positive_endpoint", "negative_endpoint"])
def test_tensor_shape_dtype_finiteness_bounds_and_row_order_are_strict(artifact_parts, fault):
    ids, target, _ = artifact_parts
    if fault == "reversed_ids":
        ids = ids.flip(0)
    elif fault == "wrong_ids":
        ids = ids + 1
    elif fault == "float_ids":
        ids = ids.float()
    elif fault == "empty_ids":
        ids = ids[:0]
    elif fault == "target_width":
        target = target[:-1]
    elif fault == "target_rank":
        target = target.unsqueeze(0)
    elif fault == "target_dtype":
        target = target.double()
    else:
        target = target.clone()
        target[0] = {"nan": float("nan"), "infinite": float("inf"), "positive_endpoint": 1,
                     "negative_endpoint": -1}[fault]
    with pytest.raises(ValueError):
        decode(artifact_parts, ids=ids, target=target)


@pytest.mark.parametrize("fault", ["source_kind", "empty_eval", "nonnormalized_eval", "missing_safety",
    "empty_safety", "empty_background", "unknown_population", "missing_source_id", "blank_source_id",
    "bad_text_hash", "bad_response_hash", "unsafe_label", "nonnormalized_fit", "evaluation_overlap",
    "background_eval_overlap", "background_fit_overlap", "duplicate_positive", "duplicate_background"])
def test_preparation_rejects_unsafe_empty_overlapping_or_unattested_rows(artifact_parts, fault):
    manifest = copy.deepcopy(artifact_parts[2])
    report = manifest["preparation"]
    positive, background = report["accepted"]["safety"], report["accepted"]["background"]
    if fault == "source_kind":
        report["source_kind"] = "unattested"
    elif fault == "empty_eval":
        report["evaluation_input_keys"] = []
    elif fault == "nonnormalized_eval":
        report["evaluation_input_keys"][0] = "NOT normalized"
    elif fault == "missing_safety":
        del report["accepted"]["safety"]
    elif fault == "empty_safety":
        positive.clear()
    elif fault == "empty_background":
        background.clear()
    elif fault == "unknown_population":
        report["accepted"]["unexpected"] = copy.deepcopy(positive)
    elif fault == "missing_source_id":
        del positive[0]["source_id"]
    elif fault == "blank_source_id":
        positive[0]["source_id"] = " "
    elif fault == "bad_text_hash":
        positive[0]["text_sha256"] = "abc"
    elif fault == "bad_response_hash":
        positive[0]["response_sha256"] = "g" * 64
    elif fault == "unsafe_label":
        positive[0]["response_harm_label"] = "harmful"
    elif fault == "nonnormalized_fit":
        positive[0]["input_key"] = "Not normalized"
    elif fault == "evaluation_overlap":
        positive[0]["input_key"] = report["evaluation_input_keys"][0]
    elif fault == "background_eval_overlap":
        background[0]["input_key"] = report["evaluation_input_keys"][0]
    elif fault == "background_fit_overlap":
        background[0]["input_key"] = positive[0]["input_key"]
    elif fault == "duplicate_positive":
        positive.append(copy.deepcopy(positive[0]))
    else:
        background.append(copy.deepcopy(background[0]))
    with pytest.raises(ValueError):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("reverse", [False, True])
def test_positive_background_overlap_is_refused_regardless_of_json_key_order(artifact_parts, reverse):
    preparation = copy.deepcopy(artifact_parts[2]["preparation"])
    rows = preparation["accepted"]
    rows["background"][0]["input_key"] = rows["safety"][0]["input_key"]
    if reverse:
        preparation["accepted"] = dict(reversed(list(rows.items())))
    with pytest.raises(ValueError, match="overlap"):
        validate_safety_preparation(preparation)


@pytest.mark.parametrize("fault", ["missing", "missing_evaluation", "extra", "revision", "empty_revision", "row_count",
    "bool_count", "extra_field", "training_dataset", "training_split", "training_config", "background_dataset"])
def test_source_provenance_requires_pinned_train_evaluation_and_shared_background(artifact_parts, fault):
    manifest = copy.deepcopy(artifact_parts[2])
    sources = manifest["preparation"]["sources"]
    if fault == "missing":
        del manifest["preparation"]["sources"]
    elif fault == "missing_evaluation":
        del sources["evaluation_wildguard"]
    elif fault == "extra":
        sources["extra"] = copy.deepcopy(sources["training"])
    elif fault in ("revision", "empty_revision"):
        sources["training"]["revision"] = "main" if fault == "revision" else ""
    elif fault in ("row_count", "bool_count"):
        sources["training"]["rows"] = 0 if fault == "row_count" else True
    elif fault == "extra_field":
        sources["training"]["extra"] = "unexpected"
    elif fault.startswith("training_"):
        sources["training"][fault.removeprefix("training_")] = "wrong"
    else:
        sources["background"]["dataset"] = "wrong"
    with pytest.raises(ValueError, match="source|training"):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("source", ["evaluation_beavertails", "evaluation_wildguard", "background", "training"])
@pytest.mark.parametrize("field,value", [("dataset", "other/unrelated"), ("config", "wrong"), ("split", "wrong"),
                                        ("revision", "b" * 40)])
def test_every_source_coordinate_and_revision_must_match_registry(artifact_parts, source, field, value):
    manifest = copy.deepcopy(artifact_parts[2])
    manifest["preparation"]["sources"][source][field] = value
    with pytest.raises(ValueError, match="pinned dataset"):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("fault", ["omitted_audit", "missing_raw_prompts", "raw_count", "normalized_keys", "token_keys",
    "hidden_eval_overlap", "training_count", "background_count", "cap", "wrong_exclusion_count", "false_exclusion_count",
    "missing_source_row", "source_row_alias", "source_row_wrong_split", "wrong_overlap_policy", "source_count"])
def test_complete_audit_and_raw_evaluation_membership_cannot_diverge(artifact_parts, fault):
    manifest = copy.deepcopy(artifact_parts[2])
    report = manifest["preparation"]
    if fault == "omitted_audit":
        manifest["preparation"] = {key: report[key] for key in ("source_kind", "accepted", "evaluation_input_keys", "sources")}
    elif fault == "missing_raw_prompts":
        del report["evaluation_prompts"]
    elif fault == "raw_count":
        report["evaluation_rows"] += 1
    elif fault == "normalized_keys":
        report["evaluation_input_keys"] = ["unrelated excluded prompt"]
    elif fault == "token_keys":
        manifest["evaluation_prompt_keys"] = {"safety": [token_context_key([1, 2, 3])]}
    elif fault == "hidden_eval_overlap":
        report["accepted"]["safety"][0]["input_key"] = report["evaluation_input_keys"][0]
        report["evaluation_prompts"] = ["unrelated excluded prompt"] * 2
        report["evaluation_input_keys"] = ["unrelated excluded prompt"]
    elif fault in ("training_count", "background_count", "cap"):
        field = {"training_count": "training_rows", "background_count": "background_count", "cap": "positive_cap"}[fault]
        report[field] = 2 if fault == "training_count" else 1 if fault == "background_count" else 0
    elif fault in ("wrong_exclusion_count", "false_exclusion_count"):
        report["exclusion_counts"] = {"unsafe_response": 1 if fault == "wrong_exclusion_count" else True}
    elif fault in ("missing_source_row", "source_row_alias", "source_row_wrong_split"):
        report["accepted"]["safety"][0]["source_id"] = {"missing_source_row": "wildguardtrain/train/1",
            "source_row_alias": "wildguardtrain/train/00", "source_row_wrong_split": "wildguardtrain/test/0"}[fault]
    elif fault == "wrong_overlap_policy":
        report["background_overlap_policy"] = "embedded"
    else:
        report["sources"]["training"]["rows"] += 1
    with pytest.raises(ValueError):
        decode(artifact_parts, manifest=manifest)


@pytest.mark.parametrize("fault,match", [
    ("malformed_positive", "missing provenance"),
    ("exceeded_cap", "exceeds positive cap"),
    ("malformed_exclusion", "malformed exclusions"),
    ("tokenizer", "decoder tokenizer disagrees"),
    ("missing_source", "requires source geometry"),
    ("missing_lineage", "requires source lineage"),
    ("unsupported_mode", "segment mode is unsupported"),
])
def test_bundle_rechecks_fit_audit_and_runtime_tokenizer(artifact_parts, fault, match):
    parts = copy.deepcopy(artifact_parts)
    manifest = parts[2]
    preparation = manifest["preparation"]
    contract = manifest["fit_contract"]
    if fault == "malformed_positive":
        preparation["accepted"]["safety"][0] = None
    elif fault == "exceeded_cap":
        training = [safe_row(f"Eligible prompt {index}") for index in range(3)]
        _, preparation, _ = prepare((training, populations()[1], populations()[2]), positive_cap=3)
        preparation["positive_cap"] = 2
        manifest["preparation"] = preparation
    elif fault == "malformed_exclusion":
        preparation["excluded"] = [{"source_id": "wildguardtrain/train/1", "reason": "unreviewed"}]
    elif fault == "tokenizer":
        contract["decoder_tokenizer_checksum"] = "0" * 64
    elif fault == "missing_source":
        contract["source"] = None
    elif fault == "missing_lineage":
        del contract["source"]["source_lineage"]
    else:
        contract["segment_mode"] = "sentences"
    with pytest.raises(ValueError, match=match):
        decode(parts)


@pytest.mark.parametrize("fault", ["missing", "missing_digest", "bad_digest", "unattested_code", "bad_git", "batch_size",
    "window", "window_geometry", "source_tokenizer", "backend", "device", "reduction", "sae", "resolved_sae", "weights"])
def test_fit_receipt_requires_complete_extraction_and_source_code_provenance(artifact_parts, fault):
    manifest = copy.deepcopy(artifact_parts[2])
    provenance = manifest["fit_provenance"]
    if fault == "missing":
        manifest["fit_provenance"] = {"unattested": True}
    elif fault == "missing_digest":
        del provenance["corpus_sha256"]
    elif fault == "bad_digest":
        provenance["checkpoint_metadata_sha256"] = "missing"
    elif fault == "unattested_code":
        provenance["code"] = {"unknown": True}
    elif fault == "bad_git":
        provenance["code"]["git_commit"] = "main"
    elif fault in ("batch_size", "window"):
        provenance["extraction_batch_size" if fault == "batch_size" else "source_window"] = True
    elif fault == "window_geometry":
        provenance["source_window"] += 1
    elif fault == "source_tokenizer":
        provenance["extraction_tokenizer_checksum"] = "b" * 64
    elif fault in ("backend", "device", "reduction"):
        field = "render_reduction" if fault == "reduction" else fault
        provenance[field] = "" if fault == "device" else "unknown"
    elif fault == "sae":
        provenance["sae"]["layer"] += 1
    elif fault == "resolved_sae":
        provenance["resolved_sae"]["sae_id"] = "different SAE"
    else:
        provenance["source_weight_revision"] = "main"
    with pytest.raises(ValueError):
        decode(artifact_parts, manifest=manifest)


def test_evaluation_subset_admission_uses_exact_rendered_prompt_keys(artifact_parts):
    parts = copy.deepcopy(artifact_parts)
    prompts = ["one", "two"]
    ids = [render_prompt_for_chat_completion(ByteChatTokenizer(), prompt).token_ids for prompt in prompts]
    parts[2]["preparation"]["evaluation_prompts"] = prompts
    parts[2]["preparation"]["evaluation_input_keys"] = prompts
    parts[2]["evaluation_prompt_keys"] = {"safety": sorted(token_context_key(row) for row in ids)}
    bundle = decode(parts)
    bundle.validate_prompts([ids[1], ids[0], ids[1]])
    for prompts in ([], [[1, 3]], [ids[0], [3]]):
        with pytest.raises(ValueError, match="not excluded from fitting"):
            bundle.validate_prompts(prompts)


@pytest.mark.parametrize("fault", ["malformed_bytes", "missing_metadata", "wrong_tensor_name", "unknown_tensor",
    "extra_task", "empty_prompt_keys", "invalid_prompt_key", "unknown_manifest", "bool_version"])
def test_malformed_bundle_fails_before_policy(artifact_parts, fault):
    ids, target, manifest = copy.deepcopy(artifact_parts)
    tensors = {"feature_ids": ids, "target.safety": target}
    if fault == "malformed_bytes":
        payload = b"not a safetensors file"
    elif fault == "missing_metadata":
        payload = deterministic_safetensors_bytes(tensors)
    else:
        if fault == "wrong_tensor_name":
            tensors["target.harmfulness"] = tensors.pop("target.safety")
        elif fault == "unknown_tensor":
            tensors["unexpected"] = target.clone()
        elif fault == "extra_task":
            manifest["evaluation_prompt_keys"]["unexpected"] = [token_context_key([1])]
        elif fault == "empty_prompt_keys":
            manifest["evaluation_prompt_keys"]["safety"] = []
        elif fault == "invalid_prompt_key":
            manifest["evaluation_prompt_keys"]["safety"] = ["not a digest"]
        elif fault == "unknown_manifest":
            manifest["extra"] = True
        else:
            manifest["format_version"] = True
        payload = deterministic_safetensors_bytes(tensors,
            metadata={TARGET_METADATA_KEY: canonical_turn_payload(manifest).decode()})
    with pytest.raises(ValueError):
        decode_safety_targets(payload, expected_contract=artifact_parts[2]["fit_contract"], expected_feature_ids=ids,
            tokenizer=ByteChatTokenizer())


def test_safety_policy_reuses_score_interpolation_and_preserves_selection_rng(artifact_parts):
    bundle = decode(artifact_parts)
    base = _config(t.zeros(6), 0)
    assert isinstance(base.selection, SamplingBundle)
    state = base.selection.rng.get_state().clone()
    policy = bundle.policy(base, 2.5)
    assert float(policy.score_policy.weight) == 2.5 / 6
    assert policy.selection is base.selection
    assert t.equal(state, base.selection.rng.get_state())
    assert t.equal(policy.score_policy.target_scores, artifact_parts[1])
    bundle.bundle.targets["safety"].zero_()
    assert t.equal(policy.score_policy.target_scores, artifact_parts[1])
    with pytest.raises(ValueError, match="existing interpolation"):
        bundle.policy(replace(base, score_policy=None), 2.5)


def test_strength_zero_is_bit_exact_for_scores_selected_ids_and_buckets(artifact_parts, monkeypatch):
    bundle = decode(artifact_parts)
    observed, segments = [], []
    original = selfcond_module.select_topk_bottoml

    def capture(scores, positive_counts, negative_counts):
        observed.append(scores.clone())
        return original(scores, positive_counts, negative_counts)

    monkeypatch.setattr(selfcond_module, "select_topk_bottoml", capture)
    monkeypatch.setattr(selfcond_module, "sample_action_counts", lambda *a, **kw: (t.tensor([2]), t.tensor([2])))
    logits = t.tensor([[[-20.0, -21.0, 20.0, 21.0, 0.0, 1.0]]], dtype=t.bfloat16)
    for steering in (False, True):
        base = _config(t.zeros(6), 0)
        cfg = bundle.policy(base, 0) if steering else base
        provider = SelfConditionedFeatureProvider(_model(), cfg, t.device("cpu"))
        state = build_test_self_cond_state(provider)
        provider.step(state, logits)
        assert provider.last_appended_segment is not None
        segments.append(dict(provider.last_appended_segment))
    assert t.equal(observed[0], observed[1])
    assert t.equal(observed[0], logits.squeeze(1).float())
    assert segments[0] == segments[1] == {3: 0, 2: 0, 1: 1, 0: 1}
