"""Both deep validators authenticate producer files while permitting relocation."""

import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.eval import build_eval_context_cache as builder
from scripts.eval import context_cache_common as common
from scripts.eval import validate_eval_context_cache as validator
from steering.eval import turn_checkpoint_provenance as provenance
from steering.eval.turn_checkpoint_provenance import turn_checkpoint_provenance
from steering.eval.turn_context_contract import turn_context_contract_hash
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY
from tests.steering.chatcore_checkpoint_helpers import PRODUCTION_MODEL_TAG
from tests.steering.test_turn_context_cache_build import build_fixture


def provenance_fixture(tmp_path, monkeypatch, *, model_present=True):
    cfg, _, _, _, tokenizer, _ = build_fixture(tmp_path, monkeypatch, clustered=True, seed=42)
    meta = json.loads(Path(cfg.checkpoint_meta_path).read_bytes())
    meta["step"] = 1116
    meta["user_config"]["seed"] = 999
    leaf = tmp_path / "original/conditional_encdec_checkpoints" / PRODUCTION_MODEL_TAG
    leaf.mkdir(parents=True)
    cfg.checkpoint_meta_path = str(leaf / "meta_001116.json")
    Path(cfg.checkpoint_meta_path).write_text(json.dumps(meta))
    if model_present:
        t.save({"tiny": t.arange(5)}, leaf / "model_001116.pt")
    contract, _ = common.turn_contract_from_config(cfg, tokenizer)
    manifest = builder.build_cache_from_config(cfg)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    standalone = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": False})
    assert isinstance(standalone, DictConfig)
    return cfg, standalone, manifest, contract, tokenizer


@pytest.mark.parametrize("entrypoint", ["builder", "standalone"])
@pytest.mark.parametrize("field", [
    "checkpoint_metadata_sha256", "checkpoint_resolved_tag",
    "checkpoint_step", "checkpoint_model_sha256", "checkpoint_model_sha256_reason",
    "checkpoint_preselection_seed", "checkpoint_preselection_seed_source",
])
@pytest.mark.parametrize("delete", [False, True], ids=["falsified", "deleted"])
def test_checkpoint_provenance_tampering(tmp_path, monkeypatch, entrypoint, field, delete):
    cfg, standalone, manifest_path, contract, _ = provenance_fixture(tmp_path, monkeypatch)
    metadata = json.loads(manifest_path.read_bytes())
    recorded = metadata["metadata"]
    assert recorded["checkpoint_resolved_tag"] == PRODUCTION_MODEL_TAG
    assert recorded["checkpoint_step"] == 1116
    assert recorded["checkpoint_preselection_seed"] == 42
    assert recorded["checkpoint_preselection_seed_source"] == f"{ONLINE_SAE_LINEAGE_META_KEY}.selection_seed"
    assert recorded["checkpoint_model_sha256"] == hashlib.sha256(
        Path(cfg.checkpoint_meta_path).with_name("model_001116.pt").read_bytes()).hexdigest()
    metadata_digest = hashlib.sha256(Path(cfg.checkpoint_meta_path).read_bytes()).hexdigest()
    # Exactly one recorded field carries the producer metadata digest.
    assert [key for key, value in recorded.items() if value == metadata_digest] == [
        "checkpoint_metadata_sha256"]
    assert not any(key.startswith("checkpoint_") for key in contract.semantic)
    if delete:
        del recorded[field]
    else:
        recorded[field] = "falsified"
    manifest_path.write_text(json.dumps(metadata))
    invoke, config = (builder.validate_cache_from_config, cfg) if entrypoint == "builder" else (
        validator.validate_cache_from_config, standalone)
    with pytest.raises(ValueError, match=f"checkpoint provenance disagrees: {field}"):
        invoke(config)


@pytest.mark.parametrize("model_present", [False, True])
def test_checkpoint_provenance_relocation_and_exact_files(tmp_path, monkeypatch, model_present):
    cfg, standalone, manifest_path, contract, tokenizer = provenance_fixture(
        tmp_path, monkeypatch, model_present=model_present)
    metadata = json.loads(manifest_path.read_bytes())["metadata"]
    if not model_present:
        assert metadata["checkpoint_model_sha256"] is None
        assert metadata["checkpoint_model_sha256_reason"] == "model_file_not_present_next_to_metadata"
    leaf = Path(cfg.checkpoint_meta_path).parent
    relocated = tmp_path / "relocated/conditional_encdec_checkpoints" / PRODUCTION_MODEL_TAG
    shutil.copytree(leaf, relocated)
    # Keep embedded paths and metadata bytes identical; only the staging root moves.
    cfg.checkpoint_meta_path = standalone.checkpoint_meta_path = str(relocated / "meta_001116.json")
    moved, _ = common.turn_contract_from_config(cfg, tokenizer)
    assert moved.semantic == contract.semantic
    assert turn_context_contract_hash(moved) == turn_context_contract_hash(contract)
    assert common.cache_identity_from_config({}, turn_contract=moved) == common.cache_identity_from_config({}, turn_contract=contract)
    for invoke, config in ((builder.validate_cache_from_config, cfg), (validator.validate_cache_from_config, standalone)):
        invoke(config)
    # Changed or newly staged weights are a different producer snapshot.
    (relocated / "model_001116.pt").write_bytes(b"different weights")
    for invoke, config in ((builder.validate_cache_from_config, cfg), (validator.validate_cache_from_config, standalone)):
        with pytest.raises(ValueError, match="checkpoint_model_sha256"):
            invoke(config)


@pytest.mark.parametrize("step", [True, 1117, None])
def test_checkpoint_provenance_rejects_filename_step_disagreement(tmp_path, step):
    path = tmp_path / "meta_001116.json"
    meta = {"step": step, ONLINE_SAE_LINEAGE_META_KEY: {"selection_seed": 42}}
    payload = json.dumps(meta).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="step disagrees"):
        turn_checkpoint_provenance(path, payload, meta)


@pytest.mark.parametrize("entrypoint", ["builder", "standalone"])
@pytest.mark.parametrize("field", ["checkpoint_step", "checkpoint_preselection_seed"])
def test_checkpoint_provenance_rejects_equal_float(tmp_path, monkeypatch, entrypoint, field):
    cfg, standalone, manifest_path, _, _ = provenance_fixture(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_bytes())
    expected = manifest["metadata"][field]
    assert type(expected) is int
    manifest["metadata"][field] = float(expected)
    assert manifest["metadata"][field] == expected
    manifest_path.write_text(json.dumps(manifest))
    invoke, config = (builder.validate_cache_from_config, cfg) if entrypoint == "builder" else (
        validator.validate_cache_from_config, standalone)
    with pytest.raises(ValueError, match=f"checkpoint provenance disagrees: {field}"):
        invoke(config)


def metadata_receipt_fixture(directory, *, name="meta_001116.json", step=1116):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    meta = {"step": step, ONLINE_SAE_LINEAGE_META_KEY: {"selection_seed": 42}}
    payload = json.dumps(meta).encode()
    path.write_bytes(payload)
    return path, payload, meta


@pytest.mark.parametrize("replace_file", [False, True], ids=["in-place-write", "path-replacement"])
def test_checkpoint_provenance_rejects_change_during_hash(tmp_path, monkeypatch, replace_file):
    path, payload, meta = metadata_receipt_fixture(tmp_path)
    model = path.with_name("model_001116.pt")
    model.write_bytes(b"original weights")
    before = model.stat()
    original_digest = hashlib.file_digest

    def change_during_hash(handle, algorithm):
        digest = original_digest(handle, algorithm)
        if replace_file:
            replacement = model.with_suffix(".replacement")
            replacement.write_bytes(model.read_bytes())
            # Same bytes, length and mtime still cannot authenticate a replaced inode.
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(model)
            assert model.stat().st_ino != before.st_ino
        else:
            # Hashing used the original bytes, but the open file changes before
            # the final fstat. A stable path/inode alone is insufficient.
            model.write_bytes(b"changed weights!")
            os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
            assert model.stat().st_ino == before.st_ino
        assert model.stat().st_size == before.st_size
        return digest

    monkeypatch.setattr(provenance.hashlib, "file_digest", change_during_hash)
    with pytest.raises(ValueError, match="model changed while hashing"):
        turn_checkpoint_provenance(path, payload, meta)


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO], ids=["permission-denied", "io-error"])
def test_checkpoint_provenance_propagates_model_open_errors(tmp_path, monkeypatch, error_number):
    path, payload, meta = metadata_receipt_fixture(tmp_path)
    model = path.with_name("model_001116.pt")
    model.write_bytes(b"staged model")
    original_open = Path.open
    error = OSError(error_number, "injected model read failure", str(model))

    def fail_model_open(self, *args, **kwargs):
        if self == model:
            raise error
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_model_open)
    with pytest.raises(OSError) as caught:
        turn_checkpoint_provenance(path, payload, meta)
    assert caught.value is error


@pytest.mark.parametrize("root", ["conditional_encdec_checkpoints", "base_checkpoints"])
def test_checkpoint_provenance_recognizes_nested_root(tmp_path, root):
    logical_tag = "parent-model/standard_sft_checkpoints"
    receipts = []
    for staging in ("original", "relocated"):
        path, payload, meta = metadata_receipt_fixture(tmp_path / staging / root / logical_tag)
        receipt = turn_checkpoint_provenance(path, payload, meta)
        assert receipt["checkpoint_resolved_tag"] == logical_tag
        receipts.append({key: value for key, value in receipt.items() if key != "checkpoint_meta_path"})
    assert receipts[0] == receipts[1]


@pytest.mark.parametrize("step", [-1, 0, 1116], ids=["negative", "zero", "positive"])
def test_checkpoint_provenance_fallback_step_domain(tmp_path, step):
    path, payload, meta = metadata_receipt_fixture(tmp_path, name="meta.json", step=step)
    if step < 0:
        with pytest.raises(ValueError, match="requires a numeric"):
            turn_checkpoint_provenance(path, payload, meta)
    else:
        assert turn_checkpoint_provenance(path, payload, meta)["checkpoint_step"] == step
