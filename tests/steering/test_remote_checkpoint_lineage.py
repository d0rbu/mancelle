from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch as t

from nanochat.checkpoint_manager import save_checkpoint
from steering.remote_training import (
    REMOTE_RUN_META_KEY,
    REMOTE_SEMANTIC_META_KEY,
    RemoteCheckpointStore,
    RemoteTrainingError,
    build_remote_run_fingerprint,
    build_remote_semantic_fingerprint_v2,
    build_remote_semantic_identity,
)
from tests.steering.test_remote_training import FakeBucket, _remote_config, _xet_hash


def _contracts() -> dict[str, dict[str, str]]:
    return {
        label: {"selection_checksum": char * 64}
        for label, char in zip(("train", "derived_val", "base_val"), "abc", strict=True)
    }


def _config(**updates: Any) -> dict[str, Any]:
    config = {
        "max_seq_len": 1024,
        "depth": 12,
        "device_batch_size": 16,
        "core_metric_every": 250,
        "sample_every": 250,
        "save_every": 250,
        "decoder_segment_size": None,
        "initialize_encoder_embeddings_cluster_vectors": "mean",
    }
    config.update(updates)
    return config


def _save_group(
    root: Path,
    *,
    step: int,
    lineage: str,
    user_config: dict[str, Any],
    semantic: str | None = None,
) -> Path:
    checkpoint_dir = root / "model" / "leaf"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "step": step,
        REMOTE_RUN_META_KEY: lineage,
        "user_config": user_config,
    }
    if semantic is not None:
        metadata[REMOTE_SEMANTIC_META_KEY] = build_remote_semantic_identity(semantic)
    save_checkpoint(
        str(checkpoint_dir),
        step,
        {"weight": t.ones(1)},
        {"state": {}},
        metadata,
        rank=0,
    )
    return checkpoint_dir


def _store(
    *,
    bucket: FakeBucket,
    root: Path,
    fingerprint: str,
    config: dict[str, Any] | None = None,
    contracts: dict[str, dict[str, str]] | None = None,
) -> RemoteCheckpointStore:
    return RemoteCheckpointStore(
        backend=bucket,
        remote=_remote_config(),
        checkpoint_dir=root / "model" / "leaf",
        checkpoint_root=root,
        remote_run_fingerprint=fingerprint,
        world_size=1,
        semantic_fingerprint_v2=(fingerprint if config is not None else None),
        resolved_training_config=config,
        selection_contracts=contracts,
    )


def test_direct_v2_publish_restore_authenticates_metadata_before_bodies(tmp_path):
    config = _config()
    contracts = _contracts()
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=1,
        selection_contracts=contracts,
    )
    source_root = tmp_path / "source"
    _save_group(
        source_root,
        step=10,
        lineage=semantic,
        semantic=semantic,
        user_config=config,
    )
    bucket = FakeBucket(tmp_path / "bucket")
    _store(
        bucket=bucket,
        root=source_root,
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    ).publish(10)

    target_root = tmp_path / "target"
    target = _store(
        bucket=bucket,
        root=target_root,
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    )
    target.restore(10)

    assert target.compatibility_mode == "direct_v2"
    assert (target.checkpoint_dir / "model_000010.pt").is_file()
    assert (target.checkpoint_dir / "optim_000010_rank0.pt").is_file()


@pytest.mark.parametrize("disabled_cooldown", [False, True])
def test_legacy_cadence_migration_retains_lineage_across_next_resume(tmp_path, disabled_cooldown):
    saved_config = _config()
    current_config = _config(
        core_metric_every=0,
        sample_every=0,
        save_every=50,
    )
    if disabled_cooldown:
        current_config["conditioning_cooldown"] = {"enabled": False}
    contracts = _contracts()
    legacy = build_remote_run_fingerprint(
        resolved_training_config=saved_config,
        world_size=1,
        selection_contracts=contracts,
    )
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=current_config,
        world_size=1,
        selection_contracts=contracts,
    )
    assert legacy != semantic

    source_root = tmp_path / "source"
    _save_group(
        source_root,
        step=10,
        lineage=legacy,
        user_config=saved_config,
    )
    bucket = FakeBucket(tmp_path / "bucket")
    _store(bucket=bucket, root=source_root, fingerprint=legacy).publish(10)

    migrated_root = tmp_path / "migrated"
    migrated = _store(
        bucket=bucket,
        root=migrated_root,
        fingerprint=semantic,
        config=current_config,
        contracts=contracts,
    )
    migrated.restore(10)
    assert migrated.remote_run_fingerprint == legacy
    assert migrated.compatibility_mode == "legacy_v1_migrated"

    _save_group(
        migrated_root,
        step=11,
        lineage=legacy,
        semantic=semantic,
        user_config=current_config,
    )
    marker = migrated.publish(11)
    assert marker["remote_run_fingerprint"] == legacy

    second = _store(
        bucket=bucket,
        root=tmp_path / "second",
        fingerprint=semantic,
        config=current_config,
        contracts=contracts,
    )
    assert second.latest_complete_step() == 11
    assert second.remote_run_fingerprint == legacy
    assert second.compatibility_mode == "legacy_v1_migrated"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("selection", "reconstruction mismatch"),
        ("semantic", "not semantically compatible"),
        ("missing_config", "requires checkpoint user_config"),
        ("wrong_stored_v2", "semantic fingerprint v2 mismatch"),
    ],
)
def test_legacy_migration_rejects_drift_before_body_download(tmp_path, mutation, error):
    saved_config = _config()
    contracts = _contracts()
    legacy = build_remote_run_fingerprint(
        resolved_training_config=saved_config,
        world_size=1,
        selection_contracts=contracts,
    )
    source_root = tmp_path / "source"
    checkpoint_dir = _save_group(
        source_root,
        step=10,
        lineage=legacy,
        user_config=saved_config,
        semantic=("f" * 64 if mutation == "wrong_stored_v2" else None),
    )
    if mutation == "missing_config":
        meta_path = checkpoint_dir / "meta_000010.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        del metadata["user_config"]
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    bucket = FakeBucket(tmp_path / "bucket")
    _store(bucket=bucket, root=source_root, fingerprint=legacy).publish(10)

    current_config = copy.deepcopy(saved_config)
    current_contracts = copy.deepcopy(contracts)
    if mutation == "semantic":
        current_config["depth"] = 13
    elif mutation == "selection":
        current_contracts["train"]["selection_checksum"] = "f" * 64
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=current_config,
        world_size=1,
        selection_contracts=current_contracts,
    )
    target = _store(
        bucket=bucket,
        root=tmp_path / "target",
        fingerprint=semantic,
        config=current_config,
        contracts=current_contracts,
    )

    with pytest.raises(RemoteTrainingError, match=error):
        target.restore(10)
    assert not (target.checkpoint_dir / "model_000010.pt").exists()
    assert not (target.checkpoint_dir / "optim_000010_rank0.pt").exists()


def test_tampered_authenticated_metadata_fails_before_body_download(tmp_path):
    config = _config()
    contracts = _contracts()
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=1,
        selection_contracts=contracts,
    )
    source_root = tmp_path / "source"
    _save_group(
        source_root,
        step=10,
        lineage=semantic,
        semantic=semantic,
        user_config=config,
    )
    bucket = FakeBucket(tmp_path / "bucket")
    source = _store(
        bucket=bucket,
        root=source_root,
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    )
    source.publish(10)
    marker_path = bucket._path(source.remote_dir + "/remote_complete_000010.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    remote_meta = bucket._path(
        source.remote_dir
        + f"/_groups/{marker['aggregate_sha256']}/meta_000010.json"
    )
    remote_meta.write_bytes(remote_meta.read_bytes() + b"tamper")

    target = _store(
        bucket=bucket,
        root=tmp_path / "target",
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    )
    with pytest.raises(RemoteTrainingError, match="identity drift"):
        target.restore(10)
    assert not (target.checkpoint_dir / "model_000010.pt").exists()


@pytest.mark.parametrize("missing", ["fingerprint", "config", "contracts"])
def test_v2_store_requires_all_compatibility_inputs(tmp_path, missing):
    config = _config()
    contracts = _contracts()
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=1,
        selection_contracts=contracts,
    )
    kwargs: dict[str, Any] = {
        "semantic_fingerprint_v2": semantic,
        "resolved_training_config": config,
        "selection_contracts": contracts,
    }
    kwargs[
        {
            "fingerprint": "semantic_fingerprint_v2",
            "config": "resolved_training_config",
            "contracts": "selection_contracts",
        }[missing]
    ] = None
    with pytest.raises(RemoteTrainingError, match="requires semantic_fingerprint_v2"):
        RemoteCheckpointStore(
            backend=FakeBucket(tmp_path / "bucket"),
            remote=_remote_config(),
            checkpoint_dir=tmp_path / "target" / "model" / "leaf",
            checkpoint_root=tmp_path / "target",
            remote_run_fingerprint=semantic,
            world_size=1,
            **kwargs,
        )


def test_v2_store_recomputes_identity_before_remote_io(tmp_path):
    config = _config()
    contracts = _contracts()
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=1,
        selection_contracts=contracts,
    )
    with pytest.raises(RemoteTrainingError, match="does not match the supplied"):
        _store(
            bucket=FakeBucket(tmp_path / "bucket"),
            root=tmp_path / "target",
            fingerprint=semantic,
            config={**config, "depth": 13},
            contracts=contracts,
        )


def test_publish_rejects_mixed_lineage_leaf(tmp_path):
    config = _config()
    contracts = _contracts()
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=1,
        selection_contracts=contracts,
    )
    source_root = tmp_path / "source"
    _save_group(
        source_root,
        step=10,
        lineage=semantic,
        semantic=semantic,
        user_config=config,
    )
    bucket = FakeBucket(tmp_path / "bucket")
    source = _store(
        bucket=bucket,
        root=source_root,
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    )
    source.publish(10)

    wrong_lineage = "f" * 64
    _save_group(
        source_root,
        step=11,
        lineage=wrong_lineage,
        semantic=semantic,
        user_config=config,
    )
    source.remote_run_fingerprint = wrong_lineage
    with pytest.raises(RemoteTrainingError, match="immutable|lineage"):
        source.publish(11)
    assert bucket.stat(source.remote_dir + "/remote_complete_000011.json") is None


def test_marker_meta_lineage_disagreement_is_authenticated_and_rejected(tmp_path):
    config = _config()
    contracts = _contracts()
    semantic = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=1,
        selection_contracts=contracts,
    )
    source_root = tmp_path / "source"
    _save_group(
        source_root,
        step=10,
        lineage=semantic,
        semantic=semantic,
        user_config=config,
    )
    bucket = FakeBucket(tmp_path / "bucket")
    source = _store(
        bucket=bucket,
        root=source_root,
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    )
    source.publish(10)

    marker_path = bucket._path(source.remote_dir + "/remote_complete_000010.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    original_generation = bucket._path(
        source.remote_dir + f"/_groups/{marker['aggregate_sha256']}"
    )
    remote_meta = original_generation / "meta_000010.json"
    metadata = json.loads(remote_meta.read_text(encoding="utf-8"))
    metadata[REMOTE_RUN_META_KEY] = "f" * 64
    remote_meta.write_text(json.dumps(metadata), encoding="utf-8")
    meta_entry = next(entry for entry in marker["files"] if entry["name"] == "meta_000010.json")
    meta_bytes = remote_meta.read_bytes()
    meta_entry["byte_length"] = len(meta_bytes)
    meta_entry["content_sha256"] = hashlib.sha256(meta_bytes).hexdigest()
    meta_entry["xet_hash"] = _xet_hash(remote_meta)
    signed = {key: value for key, value in marker.items() if key != "aggregate_sha256"}
    marker["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            signed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    rewritten_generation = bucket._path(
        source.remote_dir + f"/_groups/{marker['aggregate_sha256']}"
    )
    shutil.copytree(original_generation, rewritten_generation)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    target = _store(
        bucket=bucket,
        root=tmp_path / "target",
        fingerprint=semantic,
        config=config,
        contracts=contracts,
    )
    with pytest.raises(RemoteTrainingError, match="marker/meta lineage disagreement"):
        target.restore(10)
    assert not (target.checkpoint_dir / "model_000010.pt").exists()
