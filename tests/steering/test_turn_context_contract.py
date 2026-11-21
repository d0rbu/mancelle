"""Checkpoint-lineage and single-snapshot contract mutation tests."""

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

import nanochat.tokenizer as decoder_tokenizers
from errors import FeatureClustersError
from scripts.eval import chat_eval, context_cache_common
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    feature_clusters_checkpoint_metadata,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode, canonical_dict
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import MissingPercentilePolicy
from steering.eval import target_real_runner
from steering.eval import turn_context_contract as contracts
from steering.eval.cache_identity import cache_identity_from_config
from steering.models import ExtractionModelConfig
from steering.refined_types import PositiveInt
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY
from tests.steering.chatcore_cache_helpers import ChatTokenizer
from tests.steering.test_context_cache_in_memory_conversion import _DifferentSourceTokenizer


def checkpoint_fixture(tmp_path, *, size=4, window=4096, seed=19, mask=None, width=8, buckets=4):
    stats_path = tmp_path / "statistics.safetensors"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: t.ones(width, dtype=t.float64) for name in (
        "counts", "pct_segments_active", "mean_mag", "mean_mag_incl_zeros", "std_mag",
        "min_mag", "max_mag", "median_mag",
    )}
    tensors.update(feature_ids=t.arange(width), percentile_values=t.linspace(0.01, 0.99, 99).repeat(width, 1))
    save_file(tensors, str(stats_path), metadata={
        "pooling_mode": "max_pool", "segment_mode": "turn_delimited_fixed_tokens",
        "segment_size": str(size), "context_window": str(window), "layer": "22",
        "n_sae_features": str(width), "ignore_bos_features": "true", "l0_tier": "medium",
        "percentile_names": ",".join(f"p{i}" for i in range(1, 100)),
    })
    selection = {"distribution": "geometric", "mean": 2.0, "std": 2.0,
                 "min_k": 1, "max_k": 4, "mode": "weighted"}
    negative = {"distribution": "geometric", "mean": 1.0, "std": 1.0,
                "min_k": 1, "max_k": 2}
    sae = {"model": "gemma-3-4b-pt", "release": "gemmascope_2",
           "activation_type": "residual", "layer": 22, "width": "65k", "l0": "medium"}
    meta = {
        "step": 100, "sft_segment_mode": "turn_delimited_fixed_tokens",
        "model_config": {"sequence_len": 2048}, "num_buckets": buckets,
        "feature_vocab": None, "user_config": {
            "sae": {"release": "gemmascope", "model": "gemma-2-2b", "activation_type": "residual",
                    "layer": 8, "width": "16k"},
            "num_buckets": None,
        },
        "pretrain_conditioning": {},
        ONLINE_SAE_LINEAGE_META_KEY: {
            "schema_version": 1, **sae, "source_segment_size": size,
            "source_segment_mode": "turn_delimited_fixed_tokens", "pooling": "max_pool",
            "context_independent_features": True, "ignore_bos_features": True,
            "feature_dir": str(tmp_path / "training"), "statistics_path": str(stats_path),
            "statistics_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "n_sae_features": width, "num_buckets": buckets,
            "feature_selection": selection, "negative_selection": negative,
            "missing_percentile_policy": "error", "selection_seed": seed,
        },
    }
    if mask is not None:
        meta["feature_mask"] = mask
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta, meta_path, stats_path


def resolve(meta, **kwargs):
    return contracts.resolve_turn_context_contract(
        meta, tokenizer=ChatTokenizer(), extraction_tokenizer=_DifferentSourceTokenizer(), **kwargs,
    )


def cluster_checkpoint(meta, tmp_path, *, clusters=((1, 3, 5), (2, 4)), mask=(0, 2, 3, 4, 5, 6, 7)):
    width = PositiveInt.parse(meta[ONLINE_SAE_LINEAGE_META_KEY]["n_sae_features"])
    """Real cluster JSON plus a mask removing the first raw representative."""
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(json.dumps({"schema_version": 1, "clusters": clusters}))
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, tuple(mask))
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, width)
    assert resolved is not None
    consolidation = FeatureClusterConsolidation.build(resolved, feature_mask)
    meta.update(feature_clusters_checkpoint_metadata(resolved))
    meta["feature_mask_canonical"] = canonical_dict(feature_mask)
    vocab = effective_feature_vocab(consolidation, width)
    assert vocab is not None
    meta["feature_vocab"] = vocab.metadata_dict()
    meta["user_config"].update(feature_clusters={"path": str(cluster_path)},
                               feature_mask=canonical_dict(feature_mask))
    return consolidation


def test_complete_contract_and_relocated_snapshot(tmp_path):
    # kills: requiring cadence, hashing paths/step, using evaluation statistics, zero source BOS.
    meta, path, stats = checkpoint_fixture(tmp_path)
    contract = resolve(path)
    assert contract.semantic["source_bos_char_len"] == 3
    assert contract.semantic["source_context_window"] == 4096
    assert contract.provenance["checkpoint_metadata_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    relocated = tmp_path / "relocated.safetensors"
    relocated.write_bytes(stats.read_bytes())
    moved = copy.deepcopy(meta)
    moved[ONLINE_SAE_LINEAGE_META_KEY]["feature_dir"] = "/elsewhere/training"
    moved[ONLINE_SAE_LINEAGE_META_KEY]["statistics_path"] = "/elsewhere/statistics.safetensors"
    moved["step"] = 999
    moved["turn_self_cond_segment_size_rustbpe"] = 7
    other = resolve(moved, statistics_path=relocated)
    contracts.validate_turn_context_contract(other.semantic, contract)
    assert contracts.turn_context_contract_hash(other) == contracts.turn_context_contract_hash(contract)
    assert cache_identity_from_config({}, turn_contract=other) == cache_identity_from_config({}, turn_contract=contract)
    context = contract.build_selection_context()
    assert context.sampling_bundle.rng.initial_seed() == 19
    assert context.negative_bundle is not None
    assert context.boundaries.shape == (8, 2)


# Each case names the distinct guard it must trip; every message below is unique, so a
# case that trips a neighbouring guard fails instead of passing vacuously.
@pytest.mark.parametrize("field,value,message", [
    ("model", "wrong", "Unsupported model 'wrong' for gemma-scope-2"),
    ("release", "wrong", "'wrong' is not a valid SAERelease"),
    ("activation_type", "wrong", "'wrong' is not a valid ActivationType"),
    ("layer", 4, "Layer 4 is not published for residual/65k/medium on gemma-3-4b-pt"),
    ("width", "wrong", "'wrong' is not a valid SAEWidth"),
    ("l0", "small", "turn statistics L0 tier conflicts with checkpoint lineage"),
    ("num_buckets", 2, "checkpoint num_buckets conflicts with online-SAE lineage"),
    ("source_segment_size", 32, "turn statistics segment_size conflicts with checkpoint lineage"),
    ("n_sae_features", 16, "turn statistics n_sae_features conflicts with checkpoint lineage"),
    ("pooling", "mean_pool", "online_sae requires checkpoint-recorded max_pool extraction"),
    ("context_independent_features", False, "online_sae requires checkpoint-recorded context-independent extraction"),
    ("ignore_bos_features", False, "online_sae requires checkpoint-recorded no-BOS extraction"),
    ("statistics_sha256", "f" * 64, "Turn-context statistics raw SHA-256 does not match the checkpoint's online-SAE lineage"),
    ("selection_seed", True, "online-SAE lineage selection_seed must be an integer >= 0"),
    ("missing_percentile_policy", "silent", "Unknown missing_percentile_policy 'silent'"),
    ("source_segment_mode", "fixed_tokens", "conflicting segment-mode witnesses: sft_segment_mode='turn_delimited_fixed_tokens', structured_sft_online_sae_lineage_v1.source_segment_mode='fixed_tokens'"),
    ("feature_selection", {}, "feature_selection config must contain either 'distribution' (for sampling) or 'threshold_type'"),
])
def test_lineage_mismatch(tmp_path, field, value, message):
    # kills: trusting one mode witness, CLI/default policy, wrong SAE/size/buckets, digest bypass.
    meta, _, _ = checkpoint_fixture(tmp_path)
    meta[ONLINE_SAE_LINEAGE_META_KEY][field] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve(meta)


# "stats-l0" shares its message with test_lineage_mismatch[l0-small] (one L0 guard reached from
# either witness); within this matrix all nine messages are distinct.
@pytest.mark.parametrize("mutation,message", [
    ("missing", "online_sae requires checkpoint metadata 'structured_sft_online_sae_lineage_v1'"),
    ("incomplete", "online-SAE lineage fields are incomplete or unknown"),
    ("vocab", "checkpoint feature vocabulary conflicts with canonical mask/clusters"),
    ("buckets", "checkpoint num_buckets conflicts with online-SAE lineage"),
    ("sequence", "checkpoint model_config.sequence_len must be a positive integer"),
    ("stats-l0", "turn statistics L0 tier conflicts with checkpoint lineage"),
    ("stats-window", "turn statistics context_window must be a positive integer"),
    ("stats-mode", "turn statistics segment_mode conflicts with checkpoint lineage"),
    ("stats-bos", "turn statistics require ignore_bos_features=true"),
])
def test_incomplete_and_statistics_contracts(tmp_path, mutation, message):
    # kills: incomplete lineage admission, unbound vocabulary, unaudited statistics geometry.
    meta, _, path = checkpoint_fixture(tmp_path)
    if mutation == "missing":
        del meta[ONLINE_SAE_LINEAGE_META_KEY]
    elif mutation == "incomplete":
        del meta[ONLINE_SAE_LINEAGE_META_KEY]["selection_seed"]
    elif mutation == "vocab":
        meta["feature_vocab"] = {"vocab_size": 8}
    elif mutation == "buckets":
        meta["num_buckets"] = 5
    elif mutation == "sequence":
        meta["model_config"]["sequence_len"] = True
    else:
        field, value = {"stats-l0": ("l0_tier", "small"), "stats-window": ("context_window", "0"),
                        "stats-mode": ("segment_mode", "fixed_tokens"), "stats-bos": ("ignore_bos_features", "false")}[mutation]
        raw = path.read_bytes()
        header_len = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:8 + header_len])
        header["__metadata__"][field] = value
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw[8 + header_len:])
        meta[ONLINE_SAE_LINEAGE_META_KEY]["statistics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve(meta)


@pytest.mark.parametrize("cadence", [True, 0, -1, 1.5, "4"])
def test_invalid_optional_cadence(tmp_path, cadence):
    # kills: accepting malformed optional cadence or coercing it into an integer.
    # One cadence guard for all five cases; the echoed repr pins which value reached it.
    meta, _, _ = checkpoint_fixture(tmp_path)
    meta["turn_self_cond_segment_size_rustbpe"] = cadence
    with pytest.raises(ValueError, match=re.escape(
            f"turn_self_cond_segment_size_rustbpe must be a positive integer, got {cadence!r}")):
        resolve(meta)


def test_snapshot_single_read(tmp_path, monkeypatch):
    # kills: hashing one statistics snapshot and then parsing a second filesystem read.
    meta, _, stats = checkpoint_fixture(tmp_path)
    original = contracts.load_statistics_and_checksum_from_bytes
    payload = stats.read_bytes()

    def parse_and_replace(raw):
        assert raw == payload
        stats.write_bytes(b"replaced after digest")
        return original(raw)

    monkeypatch.setattr(contracts, "load_statistics_and_checksum_from_bytes", parse_and_replace)
    resolve(meta)
    assert stats.read_bytes() == b"replaced after digest"


def test_complete_contract_comparison(tmp_path):
    # kills: partial contract comparisons and bool/integer equality in metadata.
    meta, _, _ = checkpoint_fixture(tmp_path)
    contract = resolve(meta)
    for field in contract.semantic:
        changed = dict(contract.semantic)
        del changed[field]
        # Production reports one message for every dropped field; it does not name the field.
        with pytest.raises(ValueError, match=re.escape(
                "turn context contract disagrees with checkpoint-derived expectations")):
            contracts.validate_turn_context_contract(changed, contract)


def production_checkpoint_fixture(tmp_path, *, user_buckets=2):
    """SFT save conventions for the 3B step-1116 checkpoint; synthetic statistics only."""
    meta, path, stats = checkpoint_fixture(tmp_path, size=32, seed=42, width=65536, buckets=2)
    meta.update(step=1116, turn_self_cond_segment_size_rustbpe=33)
    meta["user_config"]["num_buckets"] = user_buckets
    meta[ONLINE_SAE_LINEAGE_META_KEY]["missing_percentile_policy"] = "lowest"
    meta["feature_mask_canonical"] = {"mode": "whitelist", "ids": list(range(844))}
    meta["feature_vocab"] = FeatureVocab(65536, tuple(range(844))).metadata_dict()
    path.write_text(json.dumps(meta))
    return meta, path, stats


def production_clustered_checkpoint_fixture(tmp_path):
    """Production dimensions with sparse, lineage-authenticated synthetic statistics."""
    meta, path, stats = production_checkpoint_fixture(tmp_path)
    # Raw id 0 is deliberately missing, despite firing in every extraction row.
    with safe_open(stats, framework="pt") as handle:
        stats_metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key)[1:847] for key in handle.keys()}
    save_file(tensors, stats, metadata=stats_metadata)
    lineage = meta[ONLINE_SAE_LINEAGE_META_KEY]
    lineage["statistics_sha256"] = hashlib.sha256(stats.read_bytes()).hexdigest()
    lineage["missing_percentile_policy"] = MissingPercentilePolicy.LOWEST.value
    # Select every active representative, including the uncovered feature.
    lineage["feature_selection"].update(min_k=4, max_k=5)
    meta["user_config"]["seed"] = 999  # SFT defaults are not selection authority.
    cluster_checkpoint(meta, tmp_path, mask=tuple(i for i in range(847) if i != 1))
    path.write_text(json.dumps(meta))
    return meta, path, stats


@pytest.mark.parametrize("user_buckets", [2, None])
def test_producer_shaped_checkpoint_resolves_lineage_everywhere(tmp_path, monkeypatch, capsys, user_buckets):
    meta, path, _ = production_checkpoint_fixture(tmp_path, user_buckets=user_buckets)
    assert meta["user_config"]["sae"] == {
        "release": "gemmascope", "model": "gemma-2-2b", "activation_type": "residual",
        "layer": 8, "width": "16k",
    }  # The real SFT default has no l0 field.
    tokenizer, source = ChatTokenizer(), _DifferentSourceTokenizer()
    tokenizer_ids = []

    def load_tokenizer(identifier):
        tokenizer_ids.append(identifier)
        return source

    monkeypatch.setattr(context_cache_common.AutoTokenizer, "from_pretrained", load_tokenizer)
    cfg = OmegaConf.create({"checkpoint_meta_path": str(path), "model_max_seq_len": 2048,
        "feature_configs": [], "target_real_configs": [{"name": "pooled", "mode": "pooled", "cache_dir": None}],
        "num_samples": 1, "temperature": 0.0, "conditioning": {"identity": {}}})
    built, _ = context_cache_common.turn_contract_from_config(cfg, tokenizer)
    model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=2048), feature_vocab=built.feature_vocab,
        encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=65536, num_buckets=2)))
    checkpoint_contract = resolve_checkpoint_feature_contract(meta, n_sae_features=PositiveInt.parse(65536),
        surface="test", source_path=str(path))
    consumed = chat_eval._resolve_chat_turn_contract(cfg, model, tokenizer, meta,
        checkpoint_contract=checkpoint_contract, rank=0, world_size=1)
    expected_sae = {"model": "gemma-3-4b-pt", "release": "gemmascope_2", "activation_type": "residual",
                    "layer": 22, "width": "65k", "l0": "medium"}
    for contract in (built, consumed):
        assert {key: getattr(contract.lineage, key) for key in expected_sae} == expected_sae
        assert contract.feature_vocab == FeatureVocab(65536, tuple(range(844)))
        assert contract.lineage.num_buckets == 2
        assert contract.lineage.missing_percentile_policy == "lowest"
    assert tokenizer_ids == ["google/gemma-3-4b-pt"] * 2
    resolved = []

    def cache_dir(**kwargs):
        resolved.append(kwargs)
        return tmp_path / "resolved"

    monkeypatch.setattr(target_real_runner, "default_target_cache_dir", cache_dir)
    target_real_runner.parse_target_real_configs_from_cfg(cfg, suite="chat_target", turn_contract=consumed)
    assert resolved[0]["activation_label"] == "res"
    assert resolved[0]["sae_width"] == 65536
    identity = resolved[0]["sae_identity"]
    assert identity.model_id == "gemma-3-4b-pt"
    assert identity.layer == 22
    assert identity.sae_id == "layer_22_width_65k_l0_medium"
    # Execute the runbook's actual extraction-coordinate snippet on the same metadata.
    runbook = (Path(__file__).resolve().parents[2] / "docs/pipelines/evaluation.md").read_text()
    snippet = runbook.split("<<'PYMETA'\n", 1)[1].split("\nPYMETA", 1)[0]
    monkeypatch.setattr(sys, "argv", ["runbook", str(path)])
    monkeypatch.setattr(decoder_tokenizers, "get_tokenizer", lambda: tokenizer)
    namespace = {}
    exec(compile(snippet, "evaluation.md:PYMETA", "exec"), namespace)
    assert namespace["sae_coordinates"] == ("gemma-3-4b-pt", "gemmascope_2", "residual", 22, "65k", "medium")
    assert namespace["geometry"] == (32, 4096, 2, 65536, 2048)
    assert capsys.readouterr().out.strip() == "gemma-3-4b-pt gemmascope_2 residual 22 65k medium"


def test_tokenizer_id_uses_registry(monkeypatch):
    monkeypatch.setitem(contracts.EXTRACTION_MODELS, "gemma-3-4b-pt", ExtractionModelConfig("registry/custom"))
    assert contracts.turn_extraction_tokenizer_id("gemma-3-4b-pt") == "registry/custom"
    assert contracts.turn_extraction_tokenizer_id("gemma-3-4b-pt", "gemma-3-4b-pt") == "registry/custom"
    with pytest.raises(ValueError, match=re.escape(
            "extraction_tokenizer_id conflicts with checkpoint SAE model")):
        contracts.turn_extraction_tokenizer_id("gemma-3-4b-pt", "google/gemma-3-4b-pt")


def test_both_cli_paths_use_the_shared_tokenizer_registry(tmp_path, monkeypatch):
    meta, path, _ = checkpoint_fixture(tmp_path)
    tokenizer, source = ChatTokenizer(), _DifferentSourceTokenizer()
    monkeypatch.setitem(contracts.EXTRACTION_MODELS, "gemma-3-4b-pt", ExtractionModelConfig("registry/custom"))
    loaded = []

    def load_tokenizer(identifier):
        loaded.append(identifier)
        return source

    monkeypatch.setattr(context_cache_common.AutoTokenizer, "from_pretrained", load_tokenizer)
    cfg = OmegaConf.create({"checkpoint_meta_path": str(path), "model_max_seq_len": 2048,
        "feature_configs": [], "target_real_configs": [{"name": "pooled", "mode": "pooled"}],
        "num_samples": 1, "temperature": 0.0, "conditioning": {"identity": {}}})
    built, _ = context_cache_common.turn_contract_from_config(cfg, tokenizer)
    model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=2048), feature_vocab=built.feature_vocab,
        encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=8, num_buckets=4)))
    checkpoint_contract = resolve_checkpoint_feature_contract(meta, n_sae_features=PositiveInt.parse(8),
        surface="test", source_path=str(path))
    consumed = chat_eval._resolve_chat_turn_contract(cfg, model, tokenizer, meta,
        checkpoint_contract=checkpoint_contract, rank=0, world_size=1)
    assert loaded == ["registry/custom", "registry/custom"]
    assert consumed.semantic == built.semantic


# The three num_buckets cases share one guard by design: both witness blocks and the bool/int
# distinction must reach the same reconciliation, which names only the conflicting field.
@pytest.mark.parametrize("block,field,value", [
    ("user_config", "num_buckets", 3), ("user_config", "num_buckets", True),
    ("pretrain_conditioning", "num_buckets", 3), ("pretrain_conditioning", "n_sae_features", 7),
])
def test_resolved_checkpoint_witness_conflicts(tmp_path, block, field, value):
    meta, _, _ = checkpoint_fixture(tmp_path)
    meta[block][field] = value
    with pytest.raises(ValueError, match=re.escape(f"checkpoint {field} conflicts with online-SAE lineage")):
        resolve(meta)


@pytest.mark.parametrize("field,value", [("activation_type", "mlp"), ("width", "16k")])
def test_valid_alternative_lineage_changes_identity(tmp_path, field, value):
    meta, _, _ = checkpoint_fixture(tmp_path)
    before = cache_identity_from_config({}, turn_contract=resolve(meta))
    meta[ONLINE_SAE_LINEAGE_META_KEY][field] = value
    after = cache_identity_from_config({}, turn_contract=resolve(meta))
    assert after != before
    assert after.feature_selection_hash != before.feature_selection_hash
    assert meta["user_config"]["sae"]["model"] == "gemma-2-2b"


def test_clustered_contract_identity_and_checkpoint_authority(tmp_path):
    # kills: rejecting clusters, using mask-only vocabulary, or dropping cluster identity from the hash.
    meta, _, _ = checkpoint_fixture(tmp_path)
    unclustered = resolve(meta)
    consolidation = cluster_checkpoint(meta, tmp_path)
    clustered = resolve(meta)
    assert clustered.consolidation is not None
    assert clustered.consolidation.resolved.definition == consolidation.resolved.definition
    assert clustered.feature_vocab == FeatureVocab(8, (0, 2, 3, 6, 7))
    assert contracts.turn_context_contract_hash(clustered) != contracts.turn_context_contract_hash(unclustered)
    assert "feature_clusters_canonical" not in unclustered.semantic
    assert "feature_clusters_descriptor" not in unclustered.semantic
    # Both partitions have the SAME effective representatives; vocabulary hashing alone is insufficient.
    cluster_checkpoint(meta, tmp_path, clusters=((1, 3), (2, 4, 5)))
    changed = resolve(meta)
    assert changed.feature_vocab == clustered.feature_vocab
    assert contracts.turn_context_contract_hash(changed) != contracts.turn_context_contract_hash(clustered)
    # Paths are provenance. Neither the original files nor mutable derived tensors are checkpoint authority.
    (tmp_path / "clusters.json").unlink()
    meta["user_config"]["feature_clusters"]["path"] = "/relocated/clusters.json"
    assert resolve(meta).semantic == changed.semantic


def test_turn_contract_binds_loaded_consolidation_even_when_vocabulary_is_equal(tmp_path):
    meta, _, _ = checkpoint_fixture(tmp_path)
    cluster_checkpoint(meta, tmp_path)
    loaded_contract = resolve_checkpoint_feature_contract(meta, n_sae_features=PositiveInt.parse(8),
        surface="test", source_path="loaded checkpoint")
    bound = resolve(meta, checkpoint_contract=loaded_contract)
    assert bound.feature_vocab == effective_feature_vocab(loaded_contract.universe, 8)
    assert bound.consolidation is not None
    assert bound.consolidation.resolved is loaded_contract.resolved_clusters
    cluster_checkpoint(meta, tmp_path, clusters=((1, 3), (2, 4, 5)))
    assert resolve(meta).feature_vocab == bound.feature_vocab
    with pytest.raises(ValueError, match=re.escape(
            "turn mask/consolidation disagrees with loaded checkpoint feature contract")):
        resolve(meta, checkpoint_contract=loaded_contract)


# "descriptor" and "canonical" deliberately share one guard: it compares the two recordings, so
# corrupting either side reports the same mismatch. The other three hit distinct guards, and
# "mask-vocab" is the only ValueError -- FeatureClustersError is not a ValueError subclass.
@pytest.mark.parametrize("mutation,error,message", [
    ("mask-vocab", ValueError, "checkpoint feature vocabulary conflicts with canonical mask/clusters"),
    ("descriptor", FeatureClustersError, "feature cluster descriptor does not match the canonical definition"),
    ("canonical", FeatureClustersError, "feature cluster descriptor does not match the canonical definition"),
    ("width", FeatureClustersError, "turn ChatCORE cache cluster descriptor has n_sae_features=16, but the loaded model has n_sae_features=8"),
    ("partial", FeatureClustersError, "turn ChatCORE cache has incomplete static feature-cluster metadata (descriptor/canonical must both be present)"),
])
def test_clustered_contract_rejects_inconsistent_checkpoint(tmp_path, mutation, error, message):
    # kills: trusting cached/partial cluster definitions, descriptors, or a mask-only checkpoint vocabulary.
    meta, _, _ = checkpoint_fixture(tmp_path)
    cluster_checkpoint(meta, tmp_path)
    if mutation == "mask-vocab":
        meta["feature_vocab"] = FeatureVocab(8, (0, 2, 3, 4, 5, 6, 7)).metadata_dict()
    elif mutation == "descriptor":
        meta["feature_clusters_descriptor"]["content_sha256"] = "f" * 64
    elif mutation == "canonical":
        meta["feature_clusters_canonical"]["clusters"] = [[1, 3], [2, 4, 5]]
    elif mutation == "width":
        meta["feature_clusters_descriptor"]["n_sae_features"] = 16
    else:
        del meta["feature_clusters_canonical"]
    with pytest.raises(error, match=re.escape(message)):
        resolve(meta)


def test_contract_constructed_once_with_final_semantics(tmp_path, monkeypatch):
    meta, _, _ = checkpoint_fixture(tmp_path)
    original = contracts.TurnContextContract
    constructed_payloads = []

    def capture(**kwargs):
        constructed_payloads.append(contracts.canonical_turn_payload(kwargs["semantic"]))
        return original(**kwargs)

    monkeypatch.setattr(contracts, "TurnContextContract", capture)
    contract = resolve(meta)
    assert constructed_payloads == [contracts.canonical_turn_payload(contract.semantic)]
