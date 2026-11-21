"""Issue #806 signed inventory and remote packed-row cache contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.remote_training as remote_module
from errors import PackedRowInvariantError
from scripts.preprocess.publish_remote_packed_inventory import (
    run as run_publish_remote_packed_inventory,
)
from steering.conditional.encoder_decoder.fp_target_source import (
    FP_TARGET_SOURCE_INTEGRITY_KEY,
    certify_fp_target_source,
)
from steering.conditional.splice_counterfactual import (
    _file_fingerprint,
    validate_packed_provenance,
)
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import load_overlay_shard, save_overlay_shard
from steering.packed_batch_io import (
    PackedRowShard,
    load_packed_row_shard_bytes,
    save_packed_row_shard,
)
from steering.refined_types import PositiveInt
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    REMOTE_PACKED_INVENTORY_FILENAME,
    RemotePackedRowMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
    build_remote_loader_certificate,
    build_remote_packed_inventory,
    parse_remote_packed_inventory,
    publish_remote_packed_inventory,
)
from tests.steering.test_remote_training import FakeBucket

_SELECTION_CHECKSUM = "a" * 64


@dataclass(frozen=True)
class _Family:
    root: Path
    packed_dir: Path
    inventory: dict[str, Any]


class _CountingBucket(FakeBucket):
    def __init__(self, root: Path):
        super().__init__(root)
        self.download_counts: Counter[str] = Counter()
        self._download_count_lock = threading.Lock()

    def download(self, remote_path: str, local_path: Path) -> None:
        with self._download_count_lock:
            self.download_counts[remote_path] += 1
        super().download(remote_path, local_path)


def _packed_shard(index: int) -> PackedRowShard:
    return PackedRowShard(
        tokens=(
            t.arange(18, dtype=t.int64).reshape(2, 9).add(index).remainder(31).add(1).to(t.uint16)
        ),
        row_seg_offsets=t.tensor([0, 1, 2], dtype=t.int32),
        seg_token_start=t.tensor([0, 0], dtype=t.uint16),
        seg_token_end=t.tensor([8, 8], dtype=t.uint16),
        seg_doc_id=t.tensor([0, 0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, 3, 5], dtype=t.int32),
        feature_ids=t.tensor(
            [1 + index, 5 + index, 9 + index, 13 + index, 17 + index],
            dtype=t.uint16,
        ),
        feature_pcts=t.tensor([20, 80, 140, 200, 240], dtype=t.uint8),
    )


def _write_family(
    root: Path,
    *,
    n_shards: int = 3,
    publish: bool = True,
) -> _Family:
    packed_dir = root / "family" / "c1024" / "sall"
    packed_dir.mkdir(parents=True)
    prov_dir = provenance_dir(packed_dir)
    prov_dir.mkdir()
    packed_paths: list[Path] = []
    for index in range(n_shards):
        stem = f"shard_{index:05d}"
        packed_paths.append(save_packed_row_shard(_packed_shard(index), packed_dir / stem))
        save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.tensor([0, 0], dtype=t.int32),
                PROVENANCE_DOC_IDX_KEY: t.tensor([2 * index, 2 * index + 1], dtype=t.int32),
                PROVENANCE_SEG_ORDINAL_KEY: t.tensor([0, 0], dtype=t.int32),
            },
            prov_dir / stem,
        )
    (packed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": "packed-test",
                "metadata": {
                    "type": "packed_rows",
                    "context_len": 1024,
                    "base_shard_stems": ["source_00000"],
                    "provenance_shards": n_shards,
                    "max_n_seg": 1,
                    "max_seg_size": 8,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    certify_fp_target_source(packed_dir, n_sae_features=PositiveInt.parse(64))
    packed_info = validate_packed_provenance(
        packed_dir,
        cache_dir=packed_dir,
    )
    selection_dir = packed_dir / "sel" / _SELECTION_CHECKSUM
    selection_dir.mkdir(parents=True)
    loader = build_remote_loader_certificate(
        selection_checksum=_SELECTION_CHECKSUM,
        selection_inventory_aggregate="b" * 64,
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        max_features_per_segment=5,
    )
    (selection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": _SELECTION_CHECKSUM,
                "metadata": {
                    REMOTE_LOADER_KEY: loader,
                    "packed_provenance": {
                        "packed_stem_fingerprint": (packed_info.packed_stem_fingerprint),
                        "packed_row_fingerprint": (packed_info.packed_row_fingerprint),
                        "provenance_shard_count": len(packed_info.provenance_shard_paths),
                        "provenance_fingerprint": (packed_info.provenance_fingerprint),
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    inventory = (
        publish_remote_packed_inventory(packed_dir)
        if publish
        else build_remote_packed_inventory(packed_dir)
    )
    assert [entry["name"] for entry in inventory["shards"]] == [path.name for path in packed_paths]
    return _Family(root=root, packed_dir=packed_dir, inventory=inventory)


def _remote_config(
    *,
    timeout: float = 0.2,
    cache_shards: int = 2,
    trust_existing_cache: bool = False,
):
    parsed = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {
                "lookahead_shards_per_rank": 2,
                "cache_shards": cache_shards,
            },
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints/conditional_encdec",
            "selection_wait_timeout_seconds": timeout,
            "checkpoint_retry_timeout_seconds": timeout,
            "retry_interval_seconds": 0.001,
            "resume_latest": True,
            "train_packed": {
                "trust_existing_cache": trust_existing_cache,
            },
        }
    )
    assert parsed is not None
    assert parsed.train_packed is not None
    return parsed


def _upload_family(family: _Family, bucket: FakeBucket) -> str:
    relative = family.packed_dir.relative_to(family.root).as_posix()
    remote_dir = f"features/{relative}"
    for name in (
        "manifest.json",
        REMOTE_PACKED_INVENTORY_FILENAME,
        *[entry["name"] for entry in family.inventory["shards"]],
    ):
        source = family.packed_dir / name
        target = bucket._path(f"{remote_dir}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return remote_dir


def _fresh_cache(family: _Family, fresh_root: Path) -> Path:
    local_dir = fresh_root / family.packed_dir.relative_to(family.root)
    local_provenance = provenance_dir(local_dir)
    local_provenance.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(provenance_dir(family.packed_dir), local_provenance)
    return local_dir


def _discover(
    family: _Family,
    bucket: FakeBucket,
    fresh_root: Path,
    *,
    timeout: float = 0.2,
    cache_shards: int = 2,
    trust_existing_cache: bool = False,
) -> RemotePackedRowMaterializer:
    local_dir = _fresh_cache(family, fresh_root)
    return RemotePackedRowMaterializer.discover(
        backend=bucket,
        remote=_remote_config(
            timeout=timeout,
            cache_shards=cache_shards,
            trust_existing_cache=trust_existing_cache,
        ),
        local_packed_dir=local_dir,
        feature_root=fresh_root,
        cache_shards=cache_shards,
        label="test train packed",
    )


def _resign_inventory(inventory: dict[str, Any]) -> None:
    signed = {key: value for key, value in inventory.items() if key != "aggregate_sha256"}
    inventory["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            signed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _inventory_v1(inventory: dict[str, Any]) -> dict[str, Any]:
    legacy = copy.deepcopy(inventory)
    legacy["schema_version"] = 1
    legacy.pop("base_shard_content_hash")
    for entry in legacy["shards"]:
        entry.pop("layout")
    _resign_inventory(legacy)
    return legacy


def test_inventory_build_parse_round_trip_and_digest_domains(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    parsed = parse_remote_packed_inventory(
        json.loads((family.packed_dir / REMOTE_PACKED_INVENTORY_FILENAME).read_text()),
        surface="test inventory",
    )
    assert parsed == family.inventory
    first = parsed["shards"][0]
    assert first["content_sha256"] != first["file_fingerprint"]

    renamed = tmp_path / "renamed.safetensors.zst"
    source = family.packed_dir / first["name"]
    shutil.copyfile(source, renamed)
    _, renamed_content, renamed_file_fingerprint = remote_module._dual_digest(renamed)
    assert renamed_content == first["content_sha256"]
    assert renamed_file_fingerprint != first["file_fingerprint"]


@pytest.mark.parametrize("schema_version", [1, 2])
def test_remote_packed_training_discovers_v1_and_v2_equivalently(
    tmp_path: Path,
    schema_version: int,
) -> None:
    family = _write_family(tmp_path / "producer")
    inventory = family.inventory if schema_version == 2 else _inventory_v1(family.inventory)
    parsed = parse_remote_packed_inventory(inventory, surface=f"schema-v{schema_version} inventory")
    bucket = FakeBucket(tmp_path / "bucket")
    remote_dir = _upload_family(family, bucket)
    bucket._path(f"{remote_dir}/{REMOTE_PACKED_INVENTORY_FILENAME}").write_text(
        json.dumps(inventory), encoding="utf-8"
    )

    materializer = _discover(family, bucket, tmp_path / "fresh")

    assert materializer.inventory == parsed
    assert materializer.packed_row_fingerprint == family.inventory["packed_row_fingerprint"]
    assert materializer.total_segments_by_stem == {
        entry["name"].removesuffix(".safetensors.zst"): entry["total_segments"]
        for entry in family.inventory["shards"]
    }


def test_publish_remote_packed_inventory_cli_run_writes_signed_artifact(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer", publish=False)
    inventory_path = family.packed_dir / REMOTE_PACKED_INVENTORY_FILENAME
    assert not inventory_path.exists()

    result = run_publish_remote_packed_inventory(
        OmegaConf.create(
            {
                "packed_row_dir": str(family.packed_dir),
                "selection_checksum": _SELECTION_CHECKSUM,
            }
        )
    )

    assert result == 0
    parsed = parse_remote_packed_inventory(
        json.loads(inventory_path.read_text(encoding="utf-8")),
        surface=str(inventory_path),
    )
    assert parsed == family.inventory


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda raw: raw.__setitem__("extra", 1),
            "contain exactly",
        ),
        (
            lambda raw: raw["shards"][0].__setitem__("extra", 1),
            "must contain exactly",
        ),
        (
            lambda raw: raw["shards"].reverse(),
            "strictly name-sorted",
        ),
        (
            lambda raw: raw["shards"].insert(1, copy.deepcopy(raw["shards"][0])),
            "strictly name-sorted",
        ),
        (
            lambda raw: raw["shards"][0].__setitem__("name", "../shard.safetensors.zst"),
            "direct filename",
        ),
        (
            lambda raw: raw["shards"][0].__setitem__("name", "bad\\shard.safetensors.zst"),
            "direct filename",
        ),
        (
            lambda raw: raw["shards"][0].__setitem__("content_sha256", "A" * 64),
            "lowercase hexadecimal",
        ),
        (
            lambda raw: raw["shards"][0].__setitem__("file_fingerprint", "0" * 64),
            "does not match.*fold",
        ),
        (
            lambda raw: raw.__setitem__("packed_row_fingerprint", "0" * 64),
            "does not match.*fold",
        ),
        (
            lambda raw: raw.__setitem__("aggregate_sha256", "0" * 64),
            "aggregate_sha256 mismatch",
        ),
    ],
)
def test_inventory_parser_rejects_every_malformed_contract_row(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    raw = copy.deepcopy(_write_family(tmp_path / "producer").inventory)
    mutate(raw)
    with pytest.raises(RemoteTrainingError, match=match):
        parse_remote_packed_inventory(raw, surface="malformed inventory")


def test_inventory_rejects_content_and_name_salted_digest_swap(
    tmp_path: Path,
) -> None:
    raw = copy.deepcopy(_write_family(tmp_path / "producer").inventory)
    entry = raw["shards"][0]
    entry["content_sha256"], entry["file_fingerprint"] = (
        entry["file_fingerprint"],
        entry["content_sha256"],
    )
    with pytest.raises(RemoteTrainingError, match="does not match.*fold"):
        parse_remote_packed_inventory(raw, surface="swapped digests")


@pytest.mark.parametrize(
    "field",
    [
        "packed_manifest_file_fingerprint",
        "provenance_shard_count",
        "provenance_fingerprint",
    ],
)
def test_inventory_aggregate_detects_each_signed_field_tamper(
    tmp_path: Path,
    field: str,
) -> None:
    raw = copy.deepcopy(_write_family(tmp_path / "producer").inventory)
    raw[field] = raw[field] + 1 if field == "provenance_shard_count" else "0" * 64
    with pytest.raises(RemoteTrainingError, match="aggregate_sha256 mismatch"):
        parse_remote_packed_inventory(raw, surface="tampered inventory")


def test_inventory_builder_refuses_quiescence_violation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    family = _write_family(tmp_path / "producer")
    real_dual_digest = remote_module._dual_digest
    mutated = False

    def mutate_after_hash(path: Path, **kwargs):
        nonlocal mutated
        result = real_dual_digest(path, **kwargs)
        if not mutated:
            mutated = True
            path.write_bytes(path.read_bytes() + b"mutation")
        return result

    monkeypatch.setattr(remote_module, "_dual_digest", mutate_after_hash)
    with pytest.raises(RemoteTrainingError, match="changed while building"):
        build_remote_packed_inventory(family.packed_dir)


def test_inventory_builder_refuses_selection_manifest_disagreement(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    manifest_path = family.packed_dir / "sel" / _SELECTION_CHECKSUM / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["packed_provenance"]["provenance_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RemoteTrainingError, match="packed_provenance"):
        build_remote_packed_inventory(family.packed_dir, selection_checksum=_SELECTION_CHECKSUM)


def test_inventory_builder_is_selection_independent_by_default(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    shutil.rmtree(family.packed_dir / "sel")

    rebuilt = build_remote_packed_inventory(family.packed_dir)

    assert rebuilt == family.inventory
    assert rebuilt["schema_version"] == 2
    assert "base_shard_content_hash" in rebuilt
    assert all("layout" in entry for entry in rebuilt["shards"])


def test_inventory_builder_requires_fp_certificate_for_c1024(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    manifest_path = family.packed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"].pop(FP_TARGET_SOURCE_INTEGRITY_KEY)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RemoteTrainingError, match="requires a completed"):
        build_remote_packed_inventory(family.packed_dir)


def test_discover_downloads_metadata_only_and_reader_requires_worker(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = _CountingBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    materializer = _discover(family, bucket, tmp_path / "fresh")

    assert materializer.shard_paths()
    assert not any(path.exists() for path in materializer.shard_paths())
    assert "backend" not in vars(materializer)
    with pytest.raises(RemoteTrainingError, match="strict rank-striped"):
        materializer.read(materializer.shard_paths()[0])


def test_discover_records_exact_preexisting_packed_descriptor(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = _CountingBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    fresh_root = tmp_path / "fresh"
    local_dir = _fresh_cache(family, fresh_root)
    first_name = family.inventory["shards"][0]["name"]
    local_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(family.packed_dir / first_name, local_dir / first_name)

    materializer = RemotePackedRowMaterializer.discover(
        backend=bucket,
        remote=_remote_config(),
        local_packed_dir=local_dir,
        feature_root=fresh_root,
        cache_shards=2,
        label="warm packed",
    )

    assert materializer.preexisting_descriptors == {
        first_name: remote_module._capture_cache_body_descriptor(local_dir / first_name)
    }


def test_discover_waits_for_inventory_arrival(tmp_path: Path) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    remote_dir = _upload_family(family, bucket)
    remote_inventory = bucket._path(f"{remote_dir}/{REMOTE_PACKED_INVENTORY_FILENAME}")
    payload = remote_inventory.read_bytes()
    remote_inventory.unlink()

    def publish_later() -> None:
        time.sleep(0.02)
        remote_inventory.write_bytes(payload)

    thread = threading.Thread(target=publish_later)
    thread.start()
    try:
        materializer = _discover(
            family,
            bucket,
            tmp_path / "fresh",
            timeout=0.2,
        )
    finally:
        thread.join()
    assert materializer.inventory == family.inventory


def test_discover_times_out_when_inventory_never_arrives(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    remote_dir = _upload_family(family, bucket)
    bucket._path(f"{remote_dir}/{REMOTE_PACKED_INVENTORY_FILENAME}").unlink()
    with pytest.raises(RemoteTrainingError, match="timed out"):
        _discover(
            family,
            bucket,
            tmp_path / "fresh",
            timeout=0.01,
        )


def test_discover_rejects_local_inventory_conflict(tmp_path: Path) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    fresh_root = tmp_path / "fresh"
    local_dir = _fresh_cache(family, fresh_root)
    (local_dir / REMOTE_PACKED_INVENTORY_FILENAME).write_text("{}")
    with pytest.raises(RemoteTrainingError, match="conflicts"):
        RemotePackedRowMaterializer.discover(
            backend=bucket,
            remote=_remote_config(),
            local_packed_dir=local_dir,
            feature_root=fresh_root,
            cache_shards=2,
            label="conflict",
        )


def test_discover_rejects_manifest_restamped_after_inventory(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    remote_dir = _upload_family(family, bucket)
    manifest_path = bucket._path(f"{remote_dir}/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["restamped"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        RemoteTrainingError,
        match="publish_remote_packed_inventory",
    ):
        _discover(family, bucket, tmp_path / "fresh")


@pytest.mark.parametrize("mode", ["missing", "count", "name", "fold"])
def test_discover_rejects_every_provenance_failure(
    tmp_path: Path,
    mode: str,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    fresh_root = tmp_path / "fresh"
    local_dir = _fresh_cache(family, fresh_root)
    prov_dir = provenance_dir(local_dir)
    paths = sorted(prov_dir.glob("*.safetensors.zst"))
    if mode == "missing":
        shutil.rmtree(prov_dir)
        match = "pre-staged packed provenance"
    elif mode == "count":
        paths[-1].unlink()
        match = "names must pair"
    elif mode == "name":
        paths[-1].rename(prov_dir / "wrong.safetensors.zst")
        match = "names must pair"
    else:
        paths[0].write_bytes(paths[0].read_bytes() + b"corrupt")
        match = "fingerprint mismatch"
    with pytest.raises(RemoteTrainingError, match=match):
        RemotePackedRowMaterializer.discover(
            backend=bucket,
            remote=_remote_config(),
            local_packed_dir=local_dir,
            feature_root=fresh_root,
            cache_shards=2,
            label="provenance failure",
        )


def test_discover_rejects_fp_certificate_remote_inventory_mismatch(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    remote_dir = _upload_family(family, bucket)
    manifest_path = bucket._path(f"{remote_dir}/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"][FP_TARGET_SOURCE_INTEGRITY_KEY]["packed_row_fingerprint"] = "0" * 64
    manifest["metadata"][FP_TARGET_SOURCE_INTEGRITY_KEY][
        "ordered_shard_content_aggregate_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inventory_path = bucket._path(f"{remote_dir}/{REMOTE_PACKED_INVENTORY_FILENAME}")
    inventory = json.loads(inventory_path.read_text())
    inventory["packed_manifest_file_fingerprint"] = _file_fingerprint(manifest_path)
    _resign_inventory(inventory)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(PackedRowInvariantError, match="fingerprint.*stale"):
        _discover(family, bucket, tmp_path / "fresh")


@pytest.mark.parametrize("mode", ["valid", "corrupt", "unknown"])
def test_namespace_validation_is_strict(
    tmp_path: Path,
    mode: str,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    fresh_root = tmp_path / "fresh"
    local_dir = _fresh_cache(family, fresh_root)
    first_name = family.inventory["shards"][0]["name"]
    if mode == "valid":
        shutil.copyfile(
            family.packed_dir / first_name,
            local_dir / first_name,
        )
    elif mode == "corrupt":
        (local_dir / first_name).write_bytes(b"corrupt")
    else:
        (local_dir / "unknown.safetensors.zst").write_bytes(b"unknown")

    if mode == "valid":
        materializer = RemotePackedRowMaterializer.discover(
            backend=bucket,
            remote=_remote_config(),
            local_packed_dir=local_dir,
            feature_root=fresh_root,
            cache_shards=2,
            label="namespace",
        )
        assert materializer.shard_paths()[0].is_file()
    else:
        with pytest.raises(RemoteTrainingError, match="integrity|unexpected"):
            RemotePackedRowMaterializer.discover(
                backend=bucket,
                remote=_remote_config(),
                local_packed_dir=local_dir,
                feature_root=fresh_root,
                cache_shards=2,
                label="namespace",
            )


def test_deferred_provenance_check_fails_before_use(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    materializer = _discover(family, bucket, tmp_path / "fresh")
    first = materializer.shard_paths()[0]
    shard = load_packed_row_shard_bytes((family.packed_dir / first.name).read_bytes())
    materializer.validate_loaded_shard(
        first,
        total_segments=shard.total_segments,
    )

    provenance_path = provenance_dir(materializer.local_dir) / first.name
    provenance = load_overlay_shard(provenance_path)
    provenance[PROVENANCE_DOC_IDX_KEY] = provenance[PROVENANCE_DOC_IDX_KEY][:-1]
    save_overlay_shard(provenance, provenance_path)
    with pytest.raises(RemoteTrainingError, match="provenance has lengths"):
        materializer.validate_loaded_shard(
            first,
            total_segments=shard.total_segments,
        )


def test_deferred_inventory_total_segments_mismatch_fails(
    tmp_path: Path,
) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    materializer = _discover(family, bucket, tmp_path / "fresh")
    first = materializer.shard_paths()[0]
    with pytest.raises(RemoteTrainingError, match="inventory certifies"):
        materializer.validate_loaded_shard(first, total_segments=999)


def test_discover_preserves_process_owned_discovery_transports(tmp_path: Path) -> None:
    family = _write_family(tmp_path / "producer")
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_family(family, bucket)
    local_dir = _fresh_cache(family, tmp_path / "fresh")
    local_dir.mkdir(parents=True, exist_ok=True)
    body_name = family.inventory["shards"][0]["name"]
    discovery_transports = [
        local_dir / f".{body_name}.remote-tmp.99999998.bbbbbbbb",
        local_dir
        / f".{REMOTE_PACKED_INVENTORY_FILENAME}.remote-discovery.99999999.cccccccc",
    ]
    lock_file = local_dir / f".{body_name}.lock"
    unrelated_hidden = local_dir / ".unrelated-cache.json"
    for path in (*discovery_transports, lock_file, unrelated_hidden):
        path.write_bytes(b"debris")

    RemotePackedRowMaterializer.discover(
        backend=bucket,
        remote=_remote_config(timeout=0.2, cache_shards=2),
        local_packed_dir=local_dir,
        feature_root=tmp_path / "fresh",
        cache_shards=2,
        label="discovery transport test",
    )
    for path in discovery_transports:
        assert path.exists(), "another node may still own the discovery transport"
    assert lock_file.exists(), "lock files must never be swept"
    assert unrelated_hidden.exists(), "non-staging hidden files must be preserved"
