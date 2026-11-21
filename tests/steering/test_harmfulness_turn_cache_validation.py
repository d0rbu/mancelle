"""Harmfulness prompt caches must pass the same source replay as target caches."""

import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from scripts.eval import build_eval_context_cache as builder
from scripts.eval import validate_eval_context_cache as validator
from steering.eval.cache_builder_stage import CacheBuilderStage
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.context_cache_validation import validate_turn_context_cache_records
from steering.eval.context_enumeration import (
    TargetRealRole,
    enumerate_chat_target_contexts,
    enumerate_harmfulness_pool_contexts,
    validate_turn_context_records,
)
from steering.eval.harmfulness_prompt_pool import build_harmfulness_prompt_pool
from steering.eval.prompt_coverage import assert_exact_prompt_coverage
from steering.segmentation import SegmentMode
from tests.steering.chatcore_cache_helpers import ChatTokenizer
from tests.steering.test_turn_context_cache_build import build_fixture

TURN = SegmentMode.turn_delimited_fixed_tokens


class EntryPoint(StrEnum):
    BUILD_STAGE = "build_stage"
    STANDALONE = "standalone"


def harmfulness_records(tokenizer):
    pool = build_harmfulness_prompt_pool(
        harmful=[{"prompt": "harmful question"}],
        benign=[{"prompt": "benign question"}], seed=42, max_harmful=1,
    )
    return enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool, segment_mode=TURN)


def invoke_validation(cfg, tokenizer, records, monkeypatch, entrypoint):
    if entrypoint is EntryPoint.BUILD_STAGE:
        cfg.stage = CacheBuilderStage.VALIDATE.value
        builder.main.__wrapped__(cfg)
        return json.loads(Path(cfg.summary_file).read_text())
    cfg = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": True})
    assert isinstance(cfg, DictConfig)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(validator, "enumerate_expected_eval_contexts", lambda *_: records)
    return validator.validate_cache_from_config(cfg)


@pytest.mark.parametrize("entrypoint", list(EntryPoint))
@pytest.mark.parametrize("mask", [None, [0, 2, 3, 5, 6]])
@pytest.mark.parametrize("include_targets", [False, True])
def test_prompt_cache_replays_through_both_validation_entrypoints(
    tmp_path, monkeypatch, entrypoint, mask, include_targets,
):
    tokenizer = ChatTokenizer()
    records = harmfulness_records(tokenizer)
    if include_targets:
        targets = enumerate_chat_target_contexts(
            tokenizer=tokenizer, task_object=[{"messages": [
                {"role": "user", "content": "reference question"},
                {"role": "assistant", "content": "reference answer"},
            ]}], task_name="reference", max_problems=None, segment_mode=TURN,
        )
        # A prompt both before and after the reference pair guards loop-local state.
        records[1:1] = targets
    cfg, records, _, _, _, _ = build_fixture(tmp_path, monkeypatch, records=records, mask=mask)
    cfg.summary_file = str(tmp_path / "summary.json")
    builder.build_cache_from_config(cfg)
    summary = invoke_validation(cfg, tokenizer, records, monkeypatch, entrypoint)
    assert summary["cached_contexts"] == summary["expected_contexts"] == len(records)
    assert summary["total_context_features"] > 0
    assert summary["target_feature_configs"] == (3 if include_targets else 0)
    assert len(summary.get("turn_target_checks", [])) == int(include_targets)
    if include_targets:
        assert summary["turn_target_checks"][0]["key"] == targets[0].key
    for record, check in zip(records, summary["routing_checks"], strict=True):
        assert check["key"] == record.key
        assert check["context_feature_count"] > 0
        assert check["context_assigned_token_count"] + check["final_assigned_token_count"] == len(record.token_ids)
        if record.metadata.get("target_real_role") is None:
            assert check["final_start"] == len(record.token_ids) - 1
            assert check["final_assigned_token_count"] == 1
    assert json.loads((tmp_path / "summary.json").read_text()) == json.loads(json.dumps(summary))


def test_prompt_only_validation_rejects_cached_bucket_corruption(tmp_path, monkeypatch):
    tokenizer = ChatTokenizer()
    cfg, records, paths, contract, tokenizer, source = build_fixture(
        tmp_path, monkeypatch, records=harmfulness_records(tokenizer), mask=[0, 2, 3, 5, 6],
    )
    manifest = builder.build_cache_from_config(cfg)
    provider = SafetensorsContextFeatureProvider(
        manifest.parent, identity=cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=TURN, expected_turn_contract=contract,
    )
    key = next(iter(provider._rows))
    tokens, segments = provider._rows[key]
    assert segments[0].buckets
    changed = replace(segments[0], buckets=tuple((bucket + 1) % contract.lineage.num_buckets
                                                for bucket in segments[0].buckets))
    provider._rows[key] = (tokens, (changed, *segments[1:]))
    with pytest.raises(ValueError, match="cached selection/raw spans disagree with source replay"):
        validate_turn_context_cache_records(
            records=records, provider=provider, tokenizer=tokenizer, turn_contract=contract,
            extraction_tokenizer=source, feature_shard_paths=paths,
        )


def test_prompt_support_does_not_admit_missing_chat_target_role(tmp_path, monkeypatch):
    cfg, records, _, _, _, _ = build_fixture(tmp_path, monkeypatch)
    full = records[0]
    assert full.metadata["target_real_role"] == TargetRealRole.FULL
    records[0] = replace(full, metadata={key: value for key, value in full.metadata.items()
                                       if key != "target_real_role"})
    builder.write_context_records(cfg.contexts_file, records)
    with pytest.raises(ValueError, match="Unknown turn target_real_role: None"):
        builder.build_cache_from_config(cfg)


def test_harmfulness_prompt_requires_explicit_turn_boundaries():
    tokenizer = ChatTokenizer()
    row = replace(harmfulness_records(tokenizer)[0], turn_char_starts=None)
    with pytest.raises(ValueError, match="Turn text/boundaries disagree with exact tokens"):
        validate_turn_context_records([row], tokenizer)


@pytest.mark.parametrize("suite", ["core_target", "chat", "samples"])
def test_turn_cache_rejects_unsupported_persisted_suite(suite):
    tokenizer = ChatTokenizer()
    row = replace(harmfulness_records(tokenizer)[0], suite=suite)
    with pytest.raises(ValueError, match="turn context caches support chat_target and harmfulness only"):
        validate_turn_context_records([row], tokenizer)


@pytest.mark.parametrize("entrypoint", list(EntryPoint))
@pytest.mark.parametrize("harmfulness_first", [False, True])
def test_shared_harmfulness_and_chat_prefix_replays_in_either_order(
    tmp_path, monkeypatch, entrypoint, harmfulness_first,
):
    tokenizer = ChatTokenizer()
    prompts = harmfulness_records(tokenizer)
    targets = enumerate_chat_target_contexts(
        tokenizer=tokenizer, task_object=[{"messages": [
            {"role": "user", "content": "harmful question"},
            {"role": "assistant", "content": "reference answer"},
        ]}], task_name="reference", max_problems=None, segment_mode=TURN,
    )
    records = [*prompts, *targets] if harmfulness_first else [*targets, *prompts]
    validate_turn_context_records(records, tokenizer)
    cfg, records, _, contract, _, _ = build_fixture(tmp_path, monkeypatch, records=records)
    assert len(records) == 3
    cfg.summary_file = str(tmp_path / "summary.json")
    manifest = builder.build_cache_from_config(cfg)
    summary = invoke_validation(cfg, tokenizer, records, monkeypatch, entrypoint)
    assert summary["cached_contexts"] == summary["expected_contexts"] == 3
    assert summary["target_feature_configs"] == 3
    assert [check["key"] for check in summary["turn_target_checks"]] == [targets[0].key]
    provider = SafetensorsContextFeatureProvider(
        manifest.parent, identity=cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=TURN, expected_turn_contract=contract,
    )
    coverage = assert_exact_prompt_coverage(provider, prompts, surface="mixed turn prefix test")
    assert coverage.expected_prompt_count == 2
    assert coverage.expected_unique_keys == 2


@pytest.mark.parametrize("fault", ["missing", "different_tokens", "conflicting_geometry"])
def test_shared_prefix_support_preserves_exact_independent_context_requirement(fault):
    tokenizer = ChatTokenizer()
    prompts = harmfulness_records(tokenizer)
    targets = enumerate_chat_target_contexts(
        tokenizer=tokenizer, task_object=[{"messages": [
            {"role": "user", "content": "harmful question"},
            {"role": "assistant", "content": "reference answer"},
        ]}], task_name="reference", max_problems=None, segment_mode=TURN,
    )
    if fault == "conflicting_geometry":
        prompts[0] = replace(prompts[0], context_text=prompts[0].context_text + "changed")
        records = [*prompts, *targets]
        match = "Conflicting duplicate turn annotations"
    else:
        records = [targets[0]] if fault == "missing" else [prompts[1], targets[0]]
        match = "Missing independent context_prefix"
    with pytest.raises(ValueError, match=match):
        validate_turn_context_records(records, tokenizer)
