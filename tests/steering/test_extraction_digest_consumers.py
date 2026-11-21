"""The optional documents digest is additive outside self-conditioned preparation."""

import hashlib
import json

import pytest
import torch as t

from scripts.eval import prepare_chat_self_conditioned_cache as preparation
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.eval.axbench_base import seed_pools
from steering.extraction_identity import extraction_documents_sha256
from steering.feature_io import MetadataKey, feature_path, load_features, save_features
from steering.validate_features import validate_feature_directory
from tests.axbench_base.unit.test_seed_pools import _write_merge_fixture
from tests.steering.test_prepare_chat_self_conditioned_cache import finish_extraction
from tests.steering.test_prepare_chat_self_conditioned_cache import legacy as legacy
from tests.steering.test_sft_features import _write_parquet, _write_shard
from tests.steering.test_turn_context_cache_build import build_fixture, convert
from tests.steering.test_validate_features import _make_valid_shard_data


@pytest.mark.parametrize("digest_present", [False, True], ids=["legacy", "digest"])
def test_axbench_seed_merge_accepts_optional_documents_digest(tmp_path, digest_present):
    paths, root, expectation, tokenizer, fingerprint, runtime = _write_merge_fixture(tmp_path)
    path = feature_path(root, paths[0].stem)
    data = load_features(path)
    assert MetadataKey.EXTRACTION_DOCUMENTS_SHA256 not in data
    if digest_present:
        data[MetadataKey.EXTRACTION_DOCUMENTS_SHA256] = extraction_documents_sha256(["first", "second"])
        save_features(data, path)
        # A producer binds its sidecar to its actual shard, including added metadata.
        sidecar_path = seed_pools._feature_sidecar_path(path)
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["canonical_output_sha256"] = seed_pools._canonical_feature_digest(path)[0]
        sidecar_path.write_text(json.dumps(sidecar))
    rows = seed_pools._merge_candidate_features(
        candidate_paths=paths, feature_root=root, expectation=expectation,
        nanochat_tokenizer=tokenizer, extraction_fingerprint=fingerprint, runtime_identity=runtime,
    )
    assert [row.sae_feature_ids for row in rows] == [[1, 5, 7], [3, 5]]
    assert rows[0].sae_feature_mags == pytest.approx([0.9, 0.2, 0.3], abs=1e-3)
    assert rows[1].sae_feature_mags == pytest.approx([0.5, 0.6], abs=1e-3)


def test_historical_dictionary_header_preserves_canonical_sidecar(tmp_path):
    paths, root, expectation, tokenizer, fingerprint, runtime = _write_merge_fixture(tmp_path)
    path = feature_path(root, paths[0].stem)
    data = load_features(path)
    data[MetadataKey.N_SAE_FEATURES] = 16
    save_features(data, path)
    historical_bytes = path.read_bytes()
    # Replay the old digest loop over exactly the fields the old reader exposed.
    old_reader_fields = load_features(path)
    assert old_reader_fields.pop(MetadataKey.N_SAE_FEATURES) == 16
    old_digest = hashlib.sha256()
    for key in sorted(old_reader_fields):
        old_digest.update(key.encode("utf-8"))
        value = old_reader_fields[key]
        if isinstance(value, t.Tensor):
            seed_pools._hash_tensor(old_digest, value)
        else:
            old_digest.update(seed_pools.canonical_json(value).encode("utf-8"))
    sidecar_path = seed_pools._feature_sidecar_path(path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["canonical_output_sha256"] = old_digest.hexdigest()
    sidecar_path.write_text(json.dumps(sidecar))
    sidecar_bytes = sidecar_path.read_bytes()
    assert seed_pools._canonical_feature_digest(path)[0] == old_digest.hexdigest()
    rows = seed_pools._merge_candidate_features(
        candidate_paths=paths, feature_root=root, expectation=expectation,
        nanochat_tokenizer=tokenizer, extraction_fingerprint=fingerprint, runtime_identity=runtime,
    )
    assert [row.sae_feature_ids for row in rows] == [[1, 5, 7], [3, 5]]
    assert path.read_bytes() == historical_bytes
    assert sidecar_path.read_bytes() == sidecar_bytes


@pytest.mark.parametrize("digest_present", [False, True], ids=["legacy", "digest"])
def test_turn_cache_conversion_accepts_optional_documents_digest(tmp_path, monkeypatch, digest_present):
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    expected = convert(records, paths, contract, tokenizer)
    for path, record in zip(paths, records, strict=True):
        data = load_features(path)
        assert MetadataKey.EXTRACTION_DOCUMENTS_SHA256 not in data
        if digest_present:
            data[MetadataKey.EXTRACTION_DOCUMENTS_SHA256] = extraction_documents_sha256([record.context_text])
            save_features(data, path)
    assert convert(records, paths, contract, tokenizer) == expected


@pytest.mark.parametrize("digest_present", [False, True], ids=["legacy", "digest"])
def test_sft_feature_loading_accepts_optional_documents_digest(tmp_path, digest_present):
    features, parquets = tmp_path / "features", tmp_path / "parquets"
    features.mkdir()
    parquets.mkdir()
    metadata = ({MetadataKey.EXTRACTION_DOCUMENTS_SHA256: extraction_documents_sha256(["doc-100", "doc-200"])}
                if digest_present else {})
    expected = _write_shard(features, "shard_00000", doc_ids=[0, 1], metadata=metadata)
    _write_parquet(parquets, "shard_00000", [100, 200])
    store = load_sft_features(str(features), str(parquets))
    assert sorted(store) == [100, 200]
    for doc, original_index in enumerate([100, 200]):
        assert store[original_index][0].feature_ids.tolist() == expected[doc]


@pytest.mark.parametrize("digests", [(False, False), (True, True), (False, True)],
                         ids=["legacy", "digest", "mixed"])
def test_directory_validation_accepts_optional_per_shard_documents_digest(tmp_path, digests):
    for index, present in enumerate(digests):
        data = _make_valid_shard_data()
        data["doc_token_counts"].fill_(2 * data["segment_size"])
        if present:
            data[MetadataKey.EXTRACTION_DOCUMENTS_SHA256] = extraction_documents_sha256(
                [f"shard {index} document {doc}" for doc in range(data["n_docs"])]
            )
        save_features(data, tmp_path / f"shard_{index:05d}_features")
    result = validate_feature_directory(tmp_path)
    assert result.valid, result.violations


@pytest.mark.parametrize("digest_present", [False, True], ids=["legacy", "digest"])
def test_selfcond_preparation_rebuilds_only_unprovable_legacy_rows(legacy, digest_present):
    cfg, _, source, _ = legacy
    path = preparation.glob_features(cfg.legacy_feature_dir)[0]
    if not digest_present:
        data = load_features(path)
        del data[MetadataKey.EXTRACTION_DOCUMENTS_SHA256]
        save_features(data, path)
    plan = json.loads(preparation.run(cfg).read_text())
    assert len(plan["extraction_keys"]) == (2 if digest_present else 3)
    if not digest_present:
        assert plan["replay"] == "rebuild_requested_prefixes"
        assert "extraction documents" in plan["legacy_replay_failure"]["message"]
    finish_extraction(cfg, source)
    manifest = json.loads(preparation.run(cfg).read_text())
    assert manifest["n_contexts"] == 3
