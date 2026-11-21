"""Schema-5 opt-in, identity completeness and early cross-mode rejection."""

import copy
import json
import re
from dataclasses import replace
from typing import Any

import pytest

from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.cache_identity import cache_identity_from_config, resolve_feature_selection_hash
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.eval.turn_context_contract import turn_context_contract_hash
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from tests.conftest import TEST_CACHE_IDENTITY
from tests.steering.test_turn_context_contract import checkpoint_fixture, resolve

TURN = SegmentMode.turn_delimited_fixed_tokens


def cache_fixture(tmp_path):
    meta, _, _ = checkpoint_fixture(tmp_path)
    contract = resolve(meta)
    identity = cache_identity_from_config({}, turn_contract=contract)
    cache = tmp_path / "cache" / "c2048" / identity.feature_selection_hash[:16]
    entries = [ContextCacheEntry.from_sequences((0, 1, 2), [ContextFeatureSegment((2,), (1,), 0, 37)],
                                                segment_mode=TURN)]
    path = save_context_cache(cache, entries, identity=identity, n_sae_features=PositiveInt.parse(8),
                              num_buckets=PositiveInt.parse(4), source_segment_mode=TURN,
                              turn_contract=contract)
    return contract, identity, path, entries


def test_turn_roundtrip(tmp_path):
    # kills: snapping schema-5 coordinates or losing the explicit consumer contract.
    contract, identity, path, entries = cache_fixture(tmp_path)
    provider = SafetensorsContextFeatureProvider(path.parent, identity=identity,
        expected_source_segment_mode=TURN, expected_turn_contract=contract)
    assert provider.manifest["schema_version"] == 5
    assert provider.manifest["source_segment_size"] == 4
    assert provider.turn_contract is contract
    assert provider.get(entries[0].token_ids) == entries[0].segments


# "fixed-consumer", "schema4" and "schema6" share the schema-version guard by construction; the
# pinned observed/expected pair keeps the three cases distinguishable. "hash" and "identity" share
# the identity-disagreement guard, which enumerates the offending keys.
@pytest.mark.parametrize("mutation,message", [
    ("fixed-consumer", "Context cache schema has changed: manifest schema_version=5, expected 4"),
    ("missing-contract", "turn cache consumption requires a complete checkpoint-derived contract"),
    ("schema4", "Context cache schema has changed: manifest schema_version=4, expected 5"),
    ("schema6", "Context cache schema has changed: manifest schema_version=6, expected 5"),
    ("mode", "Context cache source_segment_mode='fixed_tokens', expected 'turn_delimited_fixed_tokens'"),
    ("hash", "turn cache identity disagrees with contract: ['feature_selection_hash']"),
    ("identity", "turn cache identity disagrees with contract: ['sae_id']"),
    # Key presence is identity: an unexpected key (even null) or a missing key disagrees.
    ("extra-null-key", "turn cache identity disagrees with contract: ['unexpected']"),
    ("extra-key", "turn cache identity disagrees with contract: ['unexpected']"),
    ("missing-key", "turn cache identity disagrees with contract: ['sae_id']"),
    ("coordinate", "turn cache source_bos_char_len disagrees with contract"),
    ("contract", "turn context contract disagrees with checkpoint-derived expectations"),
    ("metadata", "turn cache metadata model_max_seq_len disagrees with contract"),
])
def test_wrong_cache_rejected_before_shard_reads(tmp_path, monkeypatch, mutation, message):
    # kills: mode relabeling, direct-leaf bypass, incomplete contract admission, late shard rejection.
    contract, identity, path, _ = cache_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    args: dict[str, Any] = {"expected_source_segment_mode": TURN, "expected_turn_contract": contract}
    if mutation == "fixed-consumer":
        args: dict[str, Any] = {}
    elif mutation == "missing-contract":
        del args["expected_turn_contract"]
    elif mutation in {"schema4", "schema6"}:
        manifest["schema_version"] = int(mutation[-1])
    elif mutation == "mode":
        manifest["source_segment_mode"] = SegmentMode.fixed_tokens.value
    elif mutation == "hash":
        manifest["identity"]["feature_selection_hash"] = "0" * 64
    elif mutation == "identity":
        manifest["identity"]["sae_id"] = "wrong"
    elif mutation == "extra-null-key":
        manifest["identity"]["unexpected"] = None
    elif mutation == "extra-key":
        manifest["identity"]["unexpected"] = "value"
    elif mutation == "missing-key":
        del manifest["identity"]["sae_id"]
    elif mutation == "coordinate":
        manifest["source_bos_char_len"] = 0
    elif mutation == "contract":
        del manifest["turn_contract"]["selection_semantics"]
    elif mutation == "metadata":
        manifest["metadata"]["model_max_seq_len"] = 1024
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(SafetensorsContextFeatureProvider, "_load_all_shards",
                        lambda _: pytest.fail("wrong-cache rejection occurred after shard reads"))
    with pytest.raises(ContextCacheValidationError, match=re.escape(message)):
        SafetensorsContextFeatureProvider(path.parent, identity=identity, **args)


def test_turn_consumer_rejects_fixed_cache_before_shard_reads(tmp_path, monkeypatch):
    # kills: accepting schema 4 with turn opt-in or trusting the mode label alone.
    meta, _, _ = checkpoint_fixture(tmp_path)
    contract = resolve(meta)
    cache = tmp_path / "fixed" / "c2048" / ("0" * 16)
    save_context_cache(cache, [], identity=TEST_CACHE_IDENTITY, n_sae_features=PositiveInt.parse(8),
                       num_buckets=PositiveInt.parse(4), metadata={"model_max_seq_len": 2048})
    monkeypatch.setattr(SafetensorsContextFeatureProvider, "_load_all_shards",
                        lambda _: pytest.fail("fixed cache reached shard reads"))
    with pytest.raises(ContextCacheValidationError, match=re.escape(
            "Context cache schema has changed: manifest schema_version=4, expected 5")):
        SafetensorsContextFeatureProvider(cache, identity=TEST_CACHE_IDENTITY,
            expected_source_segment_mode=TURN, expected_turn_contract=contract)


def test_hash_includes_every_semantic_field(tmp_path):
    # kills: omitting mode, coordinates, geometry, seed, vocabulary or tokenizer identity from hash.
    meta, _, _ = checkpoint_fixture(tmp_path)
    contract = resolve(meta)
    required_semantic = frozenset({
        "contract_version", "source_segment_mode", "source_segment_size", "source_context_window",
        "segment_char_coordinate_system", "source_bos_char_len", "pooling", "context_independent_features",
        "ignore_bos_features", "statistics_raw_sha256", "source_lineage", "feature_mask", "feature_vocab",
        "decoder_tokenizer_checksum", "extraction_tokenizer_checksum", "enumeration_semantics", "selection_semantics",
    })
    required_lineage = frozenset({
        "model", "release", "activation_type", "layer", "width", "l0", "source_segment_size",
        "statistics_sha256", "n_sae_features", "num_buckets", "feature_selection", "negative_selection",
        "missing_percentile_policy", "selection_seed", "source_segment_mode",
    })
    assert frozenset(contract.semantic) == required_semantic
    assert frozenset(contract.semantic["source_lineage"]) == required_lineage
    baseline = turn_context_contract_hash(contract)
    for field in required_semantic:
        changed = copy.deepcopy(contract.semantic)
        del changed[field]
        assert turn_context_contract_hash(replace(contract, semantic=changed)) != baseline, field
    for field in required_lineage:
        changed = copy.deepcopy(contract.semantic)
        del changed["source_lineage"][field]
        assert turn_context_contract_hash(replace(contract, semantic=changed)) != baseline, field
    with pytest.raises(ValueError, match=re.escape(
            "explicit turn feature_selection_hash does not match checkpoint contract")):
        resolve_feature_selection_hash({"feature_selection_hash": "0" * 64}, None, turn_contract=contract)
    with pytest.raises(ValueError, match=re.escape(
            "turn cache identity requires a complete checkpoint-derived contract")):
        resolve_feature_selection_hash({"feature_selection_hash": baseline}, {"segment_mode": TURN.value})
    assert resolve_feature_selection_hash({"feature_selection_hash": baseline}, None, turn_contract=contract) == baseline


@pytest.mark.parametrize("field", ["model_id", "sae_id", "layer", "pooling"])
def test_identity_override_is_an_assertion(tmp_path, field):
    # kills: CLI identity overriding the checkpoint-bound SAE namespace.
    meta, _, _ = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match=re.escape(
            f"conditioning.identity.{field} conflicts with checkpoint turn contract")):
        cache_identity_from_config({field: 9 if field == "layer" else "wrong"}, turn_contract=resolve(meta))


# "width" and "buckets" share one production guard: it validates both geometry fields together and
# reports them jointly, so the message cannot say which of the two disagreed.
@pytest.mark.parametrize("mutation,message", [
    ("missing-contract", "schema-5 turn cache requires a checkpoint contract"),
    ("fixed-contract", "fixed cache cannot carry a turn contract"),
    ("width", "turn cache feature width/buckets disagree with contract"),
    ("buckets", "turn cache feature width/buckets disagree with contract"),
    ("metadata", "turn cache metadata source_bos_char_len disagrees with contract"),
    ("identity", "turn cache identity disagrees with contract: ['feature_selection_hash', 'layer', 'model_id', 'pooling', 'sae_id']"),
])
def test_writer_rejects_wrong_turn_contract_before_creating_files(tmp_path, mutation, message):
    # kills: publishing schema-5 files before checking the source contract and identity.
    meta, _, _ = checkpoint_fixture(tmp_path)
    contract = resolve(meta)
    identity = cache_identity_from_config({}, turn_contract=contract)
    args: dict[str, Any] = {"identity": identity, "n_sae_features": PositiveInt.parse(8), "num_buckets": PositiveInt.parse(4),
            "source_segment_mode": TURN, "turn_contract": contract}
    if mutation == "missing-contract":
        args["turn_contract"] = None
    elif mutation == "fixed-contract":
        args["source_segment_mode"] = SegmentMode.fixed_tokens
    elif mutation == "width":
        args["n_sae_features"] = PositiveInt.parse(7)
    elif mutation == "buckets":
        args["num_buckets"] = PositiveInt.parse(3)
    elif mutation == "metadata":
        args["metadata"] = {"source_bos_char_len": 0}
    else:
        args["identity"] = TEST_CACHE_IDENTITY
    with pytest.raises(ValueError, match=re.escape(message)):
        save_context_cache(tmp_path / "bad", [], **args)
    assert not (tmp_path / "bad").exists()


def test_schema5_writer_requires_turn_labelled_entries(tmp_path):
    # kills: persisting schema-5 rows that claim the fixed-token producing geometry.
    contract, identity, path, entries = cache_fixture(tmp_path)
    assert entries[0].segment_mode is TURN
    fixed = replace(entries[0], segment_mode=SegmentMode.fixed_tokens)
    relabelled = tmp_path / "relabelled" / "c2048" / identity.feature_selection_hash[:16]
    with pytest.raises(ValueError, match="persistent turn_delimited_fixed_tokens context caches"):
        save_context_cache(relabelled, [fixed], identity=identity,
                           n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
                           source_segment_mode=TURN, turn_contract=contract)
    assert not (relabelled / "manifest.json").exists()


def test_explicit_provider_vocabulary_must_match_contract(tmp_path, monkeypatch):
    contract, identity, path, _ = cache_fixture(tmp_path)
    monkeypatch.setattr(SafetensorsContextFeatureProvider, "_load_all_shards",
                        lambda _: pytest.fail("vocabulary mismatch reached shard reads"))
    with pytest.raises(ContextCacheValidationError, match="explicit feature vocabulary"):
        SafetensorsContextFeatureProvider(path.parent, identity=identity,
            expected_source_segment_mode=TURN, expected_turn_contract=contract,
            expected_feature_vocab=FeatureVocab(8, (1, 2)))


def test_mode_mismatch_reports_observed_and_expected(tmp_path):
    contract, identity, path, _ = cache_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["source_segment_mode"] = SegmentMode.fixed_tokens.value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContextCacheValidationError,
                       match="source_segment_mode='fixed_tokens', expected 'turn_delimited_fixed_tokens'"):
        SafetensorsContextFeatureProvider(path.parent, identity=identity,
            expected_source_segment_mode=TURN, expected_turn_contract=contract)
