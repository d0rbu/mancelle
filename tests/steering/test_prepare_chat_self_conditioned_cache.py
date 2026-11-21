"""Real legacy serialization, CPU planning, extraction boundary and atomic publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.eval import prepare_chat_self_conditioned_cache as preparation
from scripts.eval.build_eval_context_cache import write_context_parquet, write_context_records
from scripts.eval.context_cache_common import load_context_records
from steering.eval.context_cache_io import save_context_cache
from steering.extraction_identity import extraction_documents_sha256
from steering.feature_io import MetadataKey, save_features
from steering.overlay_io import load_overlay_shard, save_overlay_shard
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, segment_document
from tests.steering.test_chat_eval_self_conditioned import ROOT, fixed_fixture
from tests.steering.test_context_cache_in_memory_conversion import _DifferentSourceTokenizer


def extracted_rows(directory, records, source, *, identity):
    directory.mkdir(parents=True, exist_ok=True)
    docs, indices, positions, starts, ends, counts = [], [], [], [], [], []
    for doc, row in enumerate(records):
        ids = source.encode(row.context_text)
        spans, cs, ce = segment_document(ids, source, SegmentMode.fixed_tokens, PositiveInt.parse(128))
        counts.append(len(ids))
        for ordinal, ((start, _), lo, hi) in enumerate(zip(spans, cs, ce, strict=True)):
            docs.append(doc)
            indices.append(ordinal)
            positions.append(start)
            starts.append(lo)
            ends.append(hi)
    n = len(docs)
    data = {"offsets": t.arange(0, 5 * n + 1, 5, dtype=t.int64),
        "feature_ids": t.tensor([0, 1, 2, 3, 4] * n, dtype=t.int32),
        "feature_mags": t.tensor([0.11, 0.25, 0.53, 0.71, 0.92] * n),
        "doc_ids": t.tensor(docs, dtype=t.int32), "segment_indices": t.tensor(indices, dtype=t.int32),
        "segment_positions": t.tensor(positions, dtype=t.int32), "segment_char_starts": t.tensor(starts, dtype=t.int32),
        "segment_char_ends": t.tensor(ends, dtype=t.int32), "doc_token_counts": t.tensor(counts, dtype=t.int32),
        "n_docs": len(records), "n_segments": n, "n_sae_features": 8,
        "segment_mode": "fixed_tokens", "segment_size": 128, "context_window": 4096,
        "pooling": "max_pool", "layer": 22, "context_independent_features": True,
        "ignore_bos_features": True, "l0_tier": "medium", **identity,
        MetadataKey.EXTRACTION_DOCUMENTS_SHA256: extraction_documents_sha256(row.context_text for row in records)}
    path = save_features(data, directory / "shard_00000_features")
    return [Path(path)]


@pytest.fixture(params=[(False, False), (True, False), (False, True), (True, True)],
                ids=["plain", "clustered", "raw-plain", "raw-clustered"])
def legacy(tmp_path, monkeypatch, request):
    clustered, raw_pretrain = request.param
    fixture = fixed_fixture(tmp_path, monkeypatch, gsm_problems=3, clustered=clustered)
    if raw_pretrain:
        meta = json.loads(fixture.checkpoint_path.read_text())
        meta.pop("n_sae_features")
        meta.pop("num_buckets")
        fixture.checkpoint_path.write_text(json.dumps(meta))
    source = _DifferentSourceTokenizer()
    records = [row for row in fixture.records if row.task == "GSM8K"]
    old_records = records[:1]
    jsonl = write_context_records(tmp_path / "legacy.jsonl", old_records)
    write_context_parquet(tmp_path / "legacy-parquet", old_records)
    paths = extracted_rows(tmp_path / "legacy-features", old_records, source, identity=fixture.contract.source["extraction_identity"])
    entries = preparation._convert(old_records, paths, contract=fixture.contract, tokenizer=fixture.tokenizer,
        source_tokenizer=source, pipeline=fixture.contract.pipeline())
    root = tmp_path / "legacy-cache/c2048" / fixture.contract.identity.feature_selection_hash[:16]
    manifest_path = save_context_cache(root, entries, identity=fixture.contract.identity,
        n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(2), metadata=fixture.contract.metadata)
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 3
    for index, shard in enumerate(manifest["shards"]):
        path = root / shard["file"]
        tensors = load_overlay_shard(path)
        save_overlay_shard(tensors, path, metadata={"schema_version": "3", "format": manifest["format"], "shard_index": str(index)})
        shard["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    cfg = OmegaConf.load(ROOT / "configs/eval/prepare_chat_self_conditioned_cache.yaml")
    cfg.checkpoint_meta_path = str(fixture.checkpoint_path)
    cfg.legacy_cache_dir, cfg.legacy_contexts_file = str(root), str(jsonl)
    cfg.legacy_feature_dir, cfg.legacy_parquet_dir = str(tmp_path / "legacy-features"), str(tmp_path / "legacy-parquet")
    cfg.output_dir, cfg.chat_tasks, cfg.chat_max_problems = str(tmp_path / "published"), ["GSM8K"], 3
    monkeypatch.setattr(preparation, "get_tokenizer", lambda: fixture.tokenizer)
    monkeypatch.setattr(preparation.AutoTokenizer, "from_pretrained", lambda *a: source)
    return cfg, fixture, source, entries


def finish_extraction(cfg, source):
    work = Path(str(cfg.output_dir) + ".prepare")
    rows = load_context_records(work / "extraction.jsonl")
    extracted_rows(work / "extracted", rows, source,
        identity=json.loads((work / "plan.json").read_text())["contract"]["source_geometry"]["extraction_identity"])
    cfg.stage = "publish"
    return work


def test_cpu_plan_missing_enumeration_replay_and_publication(legacy):
    cfg, fixture, source, old_entries = legacy
    plan_path = preparation.run(cfg)
    plan = json.loads(plan_path.read_text())
    assert len(plan["requested_keys"]) == 3
    assert len(plan["missing_keys"]) == len(plan["extraction_keys"]) == 2
    assert "steering.extract_features" in plan["extraction_command"]
    with initialize_config_dir(config_dir=str(ROOT / "configs/extraction"), version_base=None):
        extraction_cfg = compose(config_name="extract_features", overrides=plan["extraction_command"][6:])
    assert extraction_cfg.segment_size == 128
    assert extraction_cfg.context_independent_features is True
    assert extraction_cfg.device == "cuda"
    assert not Path(cfg.output_dir).exists()
    finish_extraction(cfg, source)
    manifest_path = preparation.run(cfg)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 4
    provenance = manifest["metadata"]["preparation_provenance"]
    assert len(provenance["preserved_keys"]) == 1
    assert provenance["content_changes"] == []
    assert manifest["n_contexts"] == 3
    assert json.loads((Path(cfg.legacy_cache_dir) / "manifest.json").read_text())["schema_version"] == 3


@pytest.mark.parametrize("fault", ["manifest", "shard", "jsonl", "parquet"])
def test_legacy_authentication_precedes_plan_publication(legacy, fault):
    cfg, _, _, _ = legacy
    manifest_path = Path(cfg.legacy_cache_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if fault == "manifest":
        manifest["identity"]["layer"] = 8
        manifest_path.write_text(json.dumps(manifest))
    elif fault == "shard":
        (manifest_path.parent / manifest["shards"][0]["file"]).write_bytes(b"bad")
    elif fault == "jsonl":
        Path(cfg.legacy_contexts_file).write_text("")
    else:
        Path(cfg.legacy_parquet_dir, "shard_00000.parquet").unlink()
    with pytest.raises(ValueError):
        preparation.run(cfg)
    assert not Path(cfg.output_dir).exists()


def test_same_shape_different_extraction_is_rejected(legacy, monkeypatch):
    cfg, fixture, source, _ = legacy
    cfg.legacy_feature_dir = None
    preparation.run(cfg)
    work = finish_extraction(cfg, source)
    records = load_context_records(work / "extraction.jsonl")
    # A real serialized alternative extraction with identical counts/geometry:
    # change text content while preserving every character and token length.
    alternatives = [replace(row, context_text=row.context_text.replace("1", "9")) for row in records]
    assert any(a.context_text != b.context_text for a, b in zip(alternatives, records, strict=True))
    path = next((work / "extracted").glob("*.zst"))
    before = preparation.load_features(path)
    extracted_rows(work / "extracted", alternatives, source, identity=fixture.contract.source["extraction_identity"])
    after = preparation.load_features(path)
    for key in ("n_docs", "n_segments", "doc_ids", "segment_char_starts", "segment_char_ends", "doc_token_counts"):
        assert t.equal(before[key], after[key]) if isinstance(before[key], t.Tensor) else before[key] == after[key]
    monkeypatch.setattr(preparation, "build_cache_entries_from_extracted_feature_shards",
                        lambda **kwargs: pytest.fail("converted different documents"))
    with pytest.raises(ValueError, match="documents do not authenticate"):
        preparation.run(cfg)
    assert not Path(cfg.output_dir).exists()
