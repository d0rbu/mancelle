"""Checkpoint-bound clustered turn selection, cache admission and deep replay."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from errors import FeatureVocabError
from scripts.eval import build_eval_context_cache as builder
from scripts.eval import context_cache_common as common
from scripts.eval import validate_eval_context_cache as validator
from steering.conditional.encoder_decoder.sft_selection import select_conversation_segments
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval import context_cache_validation as validation
from steering.eval.context_cache_build import turn_source_segments
from steering.eval.context_cache_io import (
    CLUSTER_SELECTION_SEMANTICS,
    CLUSTER_SELECTION_SEMANTICS_KEY,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.turn_context_contract import (
    build_turn_selection_context,
    resolve_turn_context_contract,
)
from steering.feature_io import load_features
from steering.refined_types import PositiveInt
from tests.steering.test_turn_context_cache_build import TURN, build_fixture, convert


def provider_for(path, contract):
    return SafetensorsContextFeatureProvider(path,
        identity=common.cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=TURN, expected_turn_contract=contract)


def test_clustered_conversion_matches_sft_selector(tmp_path, monkeypatch):
    # kills: dropping consolidation from the selection context, mask-only negatives, or cluster percentile tables.
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch, clustered=True)
    entries = convert(records, paths, contract, tokenizer)
    lineage = contract.lineage
    pipeline = build_feature_pipeline_from_cfg_block(
        {"feature_selection": dict(lineage.feature_selection), "negative_selection": lineage.negative_selection},
        per_feature_stats=contract.statistics.per_feature_stats, n_sae_features=lineage.n_sae_features,
        num_buckets=lineage.num_buckets, seed=lineage.selection_seed, feature_mask=contract.feature_mask)
    # Independent construction bypasses TurnContextContract.build_selection_context and its wiring.
    expected_context = build_turn_selection_context(pipeline=pipeline,
        missing_percentile_policy=lineage.missing_percentile_policy, consolidation=contract.consolidation)
    assert contract.feature_vocab == FeatureVocab(8, (0, 2, 3, 6, 7))
    seen = set()
    for index, (path, entry) in enumerate(zip(paths, entries, strict=True)):
        raw = turn_source_segments(load_features(path), 0)
        selected = select_conversation_segments(raw, ctx=expected_context, example_idx=index)
        for segment, expected in zip(entry.segments, selected, strict=True):
            assert segment.feature_ids == tuple(expected.feature_ids.tolist())
            assert segment.buckets == tuple(expected.buckets.tolist())
            assert set(segment.feature_ids) <= set(contract.feature_vocab.original_ids)
            assert 3 in segment.buckets  # The fixture exercises actual negative selection too.
            seen.update(segment.feature_ids)
    assert {2, 3} <= seen
    got_context = contract.build_selection_context()
    assert got_context.consolidation is not None
    assert got_context.feature_mask == contract.feature_mask
    assert expected_context.cluster_threshold_percentile_table is not None
    assert got_context.cluster_threshold_percentile_table is not None
    assert t.equal(got_context.cluster_threshold_percentile_table, expected_context.cluster_threshold_percentile_table)
    assert expected_context.negative_bundle is not None
    assert got_context.negative_bundle is not None
    assert got_context.negative_bundle.feature_mask == expected_context.negative_bundle.feature_mask
    # The clustered positive-only branch uses the same consolidation without a negative bundle.
    positive_only = replace(contract, lineage=replace(lineage, negative_selection=None)).build_selection_context()
    assert positive_only.negative_bundle is None
    assert positive_only.consolidation is contract.consolidation


@pytest.mark.parametrize("mutation", [None, "missing-marker", "stale-marker", "wrong-canonical", "wrong-descriptor"])
def test_clustered_turn_provider_requires_semantics_and_identity(tmp_path, monkeypatch, mutation):
    # kills: missing/stale cluster marker admission, ignoring canonical content, or trusting only the vocabulary.
    cfg, records, _, contract, _, _ = build_fixture(tmp_path, monkeypatch, clustered=True)
    path = builder.build_cache_from_config(cfg)
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 5
    assert manifest["metadata"][CLUSTER_SELECTION_SEMANTICS_KEY] == CLUSTER_SELECTION_SEMANTICS
    if mutation is None:
        provider = provider_for(path.parent, contract)
        assert all(provider.get(record.token_ids) for record in records)
        return
    if mutation == "missing-marker":
        del manifest["metadata"][CLUSTER_SELECTION_SEMANTICS_KEY]
    elif mutation == "stale-marker":
        manifest["metadata"][CLUSTER_SELECTION_SEMANTICS_KEY] = "post_mask_representative_max_percentile_v1"
    else:
        key = "feature_clusters_canonical" if mutation == "wrong-canonical" else "feature_clusters_descriptor"
        manifest["metadata"][key] = {}
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(SafetensorsContextFeatureProvider, "_load_all_shards",
                        lambda _: pytest.fail("invalid cluster metadata reached shard IO"))
    with pytest.raises(ContextCacheValidationError, match="clustered selection|disagrees"):
        provider_for(path.parent, contract)


@pytest.mark.parametrize("cache_clustered", [False, True])
def test_clustered_and_unclustered_turn_contracts_reject_each_others_cache(tmp_path, monkeypatch, cache_clustered):
    # kills: one-way cluster admission, hash-leaf-only identity, or comparing vocabulary without clusters.
    cfg, records, paths, clustered, tokenizer, source = build_fixture(tmp_path, monkeypatch, clustered=True)
    meta = json.loads(Path(cfg.checkpoint_meta_path).read_text())
    unclustered_meta = copy.deepcopy(meta)
    del unclustered_meta["feature_clusters_canonical"]
    del unclustered_meta["feature_clusters_descriptor"]
    del unclustered_meta["user_config"]["feature_clusters"]
    # A mask-only checkpoint can have exactly the same effective vocabulary.
    unclustered_meta["feature_mask_canonical"]["ids"] = list(clustered.feature_vocab.original_ids)
    unclustered_meta["user_config"]["feature_mask"] = unclustered_meta["feature_mask_canonical"]
    unclustered = resolve_turn_context_contract(unclustered_meta, tokenizer=tokenizer, extraction_tokenizer=source)
    assert unclustered.feature_vocab == clustered.feature_vocab
    produced, consumed = (clustered, unclustered) if cache_clustered else (unclustered, clustered)
    identity = common.cache_identity_from_config({}, turn_contract=produced)
    cache = tmp_path / "cross-admission" / "c2048" / identity.feature_selection_hash[:16]
    manifest = save_context_cache(cache, convert(records, paths, produced, tokenizer), identity=identity,
        n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
        source_segment_mode=TURN, turn_contract=produced)
    if not cache_clustered:
        assert CLUSTER_SELECTION_SEMANTICS_KEY not in json.loads(manifest.read_text())["metadata"]
    renamed = cache.with_name(common.cache_identity_from_config({}, turn_contract=consumed).feature_selection_hash[:16])
    cache.rename(renamed)
    monkeypatch.setattr(SafetensorsContextFeatureProvider, "_load_all_shards",
                        lambda _: pytest.fail("incompatible cluster contract reached shard IO"))
    with pytest.raises(ContextCacheValidationError, match="disagrees|clustered selection"):
        provider_for(renamed, consumed)


def test_clustered_turn_deep_validation_uses_effective_vocabulary(tmp_path, monkeypatch):
    # kills: deep replay skipping consolidated selection or validating against the mask-only vocabulary.
    cfg, records, paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch, clustered=True)
    manifest = builder.build_cache_from_config(cfg)
    provider = provider_for(manifest.parent, contract)
    observed_vocabs = []
    original = validation._validate_context_segment_features

    def capture(**kwargs):
        observed_vocabs.append(kwargs["feature_vocab"])
        original(**kwargs)

    monkeypatch.setattr(validation, "_validate_context_segment_features", capture)
    standalone = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": False})
    assert isinstance(standalone, DictConfig)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    builder.validate_cache_from_config(cfg)
    summary = validator.validate_cache_from_config(standalone)
    assert summary["turn_target_checks"]
    assert observed_vocabs and all(vocab == contract.feature_vocab for vocab in observed_vocabs)
    segment = provider.get(records[0].token_ids)[0]
    bad = replace(segment, feature_ids=(4,), buckets=(0,))  # Allowed raw member, not a representative.
    with pytest.raises(FeatureVocabError, match="vocab|universe"):
        original(record=records[0], segment=bad, seg_idx=0, n_sae_features=8,
                 num_buckets=4, feature_vocab=contract.feature_vocab)
    key = next(iter(provider._rows))
    tokens, segments = provider._rows[key]
    provider._rows[key] = (tokens, (bad, *segments[1:]))
    with pytest.raises(ValueError, match="source replay"):
        validation.validate_turn_context_cache_records(records=records, provider=provider, tokenizer=tokenizer,
            turn_contract=contract, extraction_tokenizer=source, feature_shard_paths=paths)


def test_clustered_turn_provider_rejects_raw_member_with_valid_checksum(tmp_path, monkeypatch):
    # kills: trusting manifest/vocabulary metadata while admitting nonrepresentative payload IDs.
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch, clustered=True)
    entries = convert(records, paths, contract, tokenizer)
    entry = entries[0]
    bad_segment = replace(entry.segments[0], feature_ids=(4,), buckets=(0,))
    entries[0] = replace(entry, segments=(bad_segment, *entry.segments[1:]))
    identity = common.cache_identity_from_config({}, turn_contract=contract)
    cache = tmp_path / "bad-payload" / "c2048" / identity.feature_selection_hash[:16]
    save_context_cache(cache, entries, identity=identity, n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(4), source_segment_mode=TURN, turn_contract=contract)
    with pytest.raises(ContextCacheValidationError, match="vocab|universe"):
        provider_for(cache, contract)


def test_clustered_turn_writer_rejects_stale_marker_override(tmp_path, monkeypatch):
    # kills: allowing caller metadata to replace the required clustered selection marker.
    cfg, _, _, _, _, _ = build_fixture(tmp_path, monkeypatch, clustered=True)
    cfg.metadata[CLUSTER_SELECTION_SEMANTICS_KEY] = "stale"
    with pytest.raises(ValueError, match="cluster_selection_semantics disagrees"):
        builder.build_cache_from_config(cfg)


@pytest.mark.parametrize("field", ["feature_clusters", "feature_delta"])
def test_clustered_builder_still_rejects_cli_selection_overrides(tmp_path, monkeypatch, field):
    # kills: granting CLI authority over the checkpoint's clustered selection.
    cfg, _, _, _, _, _ = build_fixture(tmp_path, monkeypatch, clustered=True)
    cfg[field] = {"path": "/operator/override.json"}
    with pytest.raises(ValueError, match="overrides"):
        builder.build_cache_from_config(cfg)
