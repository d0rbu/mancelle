"""Adversarial unit coverage for the remote-percentile state machine."""

from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import steering.percentile_remote as remote
from steering.feature_io import COMPRESSED_EXT, RegularFileIdentity
from steering.remote_feature_inventory import RemoteFeatureShard
from steering.remote_training import BucketBackend, RemoteObjectInfo
from steering.remote_types import NormalizedRemotePrefix
from steering.stats_io import load_statistics, save_statistics
from tests.scripts.test_production_manifest_v2 import _make_env
from tests.steering.test_percentile_remote import (
    FilesystemBucket,
    _remote_cfg,
    _stage_sources,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _config(tmp_path: Path, *, expected_count: int = 3) -> remote.RemotePercentileConfig:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    cfg.remote.expected_shard_count = expected_count
    return remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)


def _inventory(count: int = 3) -> tuple[RemoteFeatureShard, ...]:
    return tuple(
        RemoteFeatureShard(
            index,
            RemoteObjectInfo(
                f"raw/shard_{index:05d}_features{COMPRESSED_EXT}",
                100 + index,
                f"{index + 1:064x}",
            ),
        )
        for index in range(count)
    )


def _staging(
    tmp_path: Path,
    *,
    output_state: remote.RemoteOutputState | None = None,
    required_capacity: int = 1,
) -> remote.PercentileStaging:
    config = _config(tmp_path)
    return remote.PercentileStaging(
        config=config,
        contract_sha256=_HASH_B,
        required_capacity=required_capacity,
        inventory=_inventory(),
        output_state=output_state or remote.RemoteOutputState(listing=()),
    )


def _info(path: str, *, size: int = 1, xet: str = _HASH_A) -> RemoteObjectInfo:
    return RemoteObjectInfo(path, size, xet)


def _claim_info(
    config: remote.RemotePercentileConfig,
    canonical_name: str,
    *,
    xet: str = _HASH_A,
) -> RemoteObjectInfo:
    return _info(
        remote._join_remote(
            config.output_prefix,
            "_verified_upload_claims",
            canonical_name,
            f"{xet}.json",
        )
    )


@pytest.mark.parametrize("value", [True, "1", None, 0.0, -1.0, float("inf")])
def test_strict_positive_finite_rejects_every_nonpositive_or_wrong_type(value: object) -> None:
    with pytest.raises(remote.RemotePercentileError, match="positive finite"):
        remote._strict_positive_finite(value, name="value")
    assert remote._strict_positive_finite(2, name="value") == 2.0


def test_config_path_rank_cuda_and_profile_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    cfg = _remote_cfg(feature_dir, feature_dir / "nested")
    with pytest.raises(remote.RemotePercentileError, match="overlap"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)

    cfg.remote.staging_dir = str(tmp_path)
    with pytest.raises(remote.RemotePercentileError, match="overlap"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)

    cfg.remote.staging_dir = str(tmp_path / "staging")
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "staging" / "profile.json")
    with pytest.raises(remote.RemotePercentileError, match="profile output overlap"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)

    cfg.profile = False
    cfg.zstd_level = 23
    with pytest.raises(remote.RemotePercentileError, match="zstd_level"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)

    cfg.zstd_level = 12
    cfg.remote.staging_dir = None
    with pytest.raises(remote.RemotePercentileError, match="staging_dir"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)

    cfg.remote.staging_dir = str(tmp_path / "staging")
    monkeypatch.setenv("WORLD_SIZE", "not-an-int")
    with pytest.raises(remote.RemotePercentileError, match="must be an integer"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)
    monkeypatch.setenv("WORLD_SIZE", "1")
    cfg.device = "cuda"
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    with pytest.raises(remote.RemotePercentileError, match="CUDA is unavailable"):
        remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)


def test_profile_default_and_source_family_gap(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    cfg.profile = True
    cfg.profile_summary_output = None
    resolved = remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_HASH_A)
    assert resolved.profile_path is not None
    assert resolved.profile_path.name == "pct_remote_profile.json"
    with pytest.raises(remote.RemotePercentileError, match="not the exact contiguous family"):
        remote._validate_source_family((_inventory()[1],), expected_count=1)


def test_retry_timeout_and_base_exception_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter((0.0, 1.0))
    monkeypatch.setattr(remote.time, "monotonic", lambda: next(times))
    with pytest.raises(remote.RemotePercentileError, match="timed out"):
        remote._retry_backend_call(
            "test",
            lambda: (_ for _ in ()).throw(OSError("permanent")),
            timeout_seconds=0.1,
            retry_interval_seconds=0.01,
        )
    monkeypatch.setattr(remote.time, "monotonic", lambda: 0.0)
    with pytest.raises(KeyboardInterrupt, match="stop"):
        remote._retry_backend_call(
            "test",
            lambda: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
            timeout_seconds=1,
            retry_interval_seconds=0.01,
        )


def test_cuda_contract_and_manifest_metadata_are_runtime_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        device=t.device("cuda:0"),
        output_prefix=NormalizedRemotePrefix.parse(
            f"raw/pct/{_HASH_B[: remote.CHECKSUM_PREFIX_LEN]}"
        ),
    )
    monkeypatch.setattr(t.cuda, "get_device_name", lambda _index: "test-gpu")
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _index: (8, 9))
    contract = remote._producer_contract(
        config,
        inventory_sha256=_HASH_A,
        percentile_checksum=_HASH_B,
        n_sae_features=17,
    )
    assert cast(dict, contract["producer"])["gpu_name"] == "test-gpu"
    statistics = _info(
        remote._join_remote(config.output_prefix, remote.OVERLAY_STATISTICS_FILENAME)
    )
    metadata = remote._manifest_metadata(
        contract=contract,
        contract_sha256=_HASH_A,
        statistics_info=statistics,
    )
    assert cast(dict, metadata["producer"])["gpu_capability"] == [8, 9]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.pop("source"), "unexpected or missing keys"),
        (lambda value: value.__setitem__("schema_version", True), "schema is malformed"),
        (
            lambda value: value.__setitem__("percentile_algorithm_version", True),
            "schema is malformed",
        ),
        (
            lambda value: value.__setitem__("expected_segment_size", True),
            "schema is malformed",
        ),
        (lambda value: value.__setitem__("expected_segment_size", 0), "schema is malformed"),
        (lambda value: value.__setitem__("source", []), "schema is malformed"),
        (
            lambda value: cast(dict, value["source"]).__setitem__("extra", 1),
            "schema is malformed",
        ),
        (lambda value: value.__setitem__("percentile_table", []), "schema is malformed"),
        (
            lambda value: cast(dict, value["percentile_table"]).__setitem__("extra", 1),
            "schema is malformed",
        ),
        (lambda value: value.__setitem__("output", []), "schema is malformed"),
        (
            lambda value: cast(dict, value["output"]).__setitem__("extra", 1),
            "schema is malformed",
        ),
        (lambda value: value.__setitem__("producer", []), "schema is malformed"),
        (
            lambda value: cast(dict, value["producer"]).__setitem__("extra", 1),
            "schema is malformed",
        ),
        (lambda value: value.__setitem__("serialization", []), "schema is malformed"),
        (
            lambda value: cast(dict, value["serialization"]).__setitem__("extra", 1),
            "schema is malformed",
        ),
        (
            lambda value: cast(dict, value["source"]).__setitem__("bucket_id", 1),
            "malformed remote path",
        ),
        (
            lambda value: cast(dict, value["source"]).__setitem__("prefix", "../raw"),
            "malformed remote path",
        ),
        (
            lambda value: cast(dict, value["output"]).__setitem__("prefix", ""),
            "malformed remote path",
        ),
        (
            lambda value: cast(dict, value["output"]).__setitem__(
                "prefix", "raw/pct/ffffffffffffffff"
            ),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["source"]).__setitem__("shard_count", True),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["source"]).__setitem__("shard_count", 0),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["source"]).__setitem__("inventory_sha256", "bad"),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["percentile_table"]).__setitem__("checksum", "bad"),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["percentile_table"]).__setitem__("n_sae_features", True),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["percentile_table"]).__setitem__("n_sae_features", 0),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["output"]).__setitem__("zstd_level", True),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["output"]).__setitem__("zstd_level", 0),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["output"]).__setitem__("zstd_level", 23),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["producer"]).__setitem__("device_type", "tpu"),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["producer"]).__setitem__("torch_version", ""),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["producer"]).__setitem__("cuda_runtime", "12.8"),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["serialization"]).__setitem__("safetensors_version", ""),
            "identity is malformed",
        ),
        (
            lambda value: cast(dict, value["serialization"]).__setitem__("zstandard_version", 1),
            "identity is malformed",
        ),
    ],
)
def test_remote_contract_parser_mutation_table(tmp_path: Path, mutate, match: str) -> None:
    config = _config(tmp_path)
    contract = remote._producer_contract(
        config,
        inventory_sha256=_HASH_A,
        percentile_checksum=_HASH_A,
        n_sae_features=17,
    )
    mutate(contract)
    with pytest.raises(remote.RemotePercentileError, match=match):
        remote.validate_remote_percentile_contract(contract, surface="contract")


@pytest.mark.parametrize(
    "producer_mutation",
    [
        lambda value: value.__setitem__("cuda_runtime", None),
        lambda value: value.__setitem__("cuda_runtime", ""),
        lambda value: value.__setitem__("gpu_name", None),
        lambda value: value.__setitem__("gpu_name", ""),
        lambda value: value.__setitem__("gpu_capability", None),
        lambda value: value.__setitem__("gpu_capability", [8]),
        lambda value: value.__setitem__("gpu_capability", [True, 9]),
        lambda value: value.__setitem__("gpu_capability", [-1, 9]),
    ],
)
def test_remote_contract_parser_rejects_malformed_cuda_runtime(
    tmp_path: Path,
    producer_mutation,
) -> None:
    config = _config(tmp_path)
    contract = remote._producer_contract(
        config,
        inventory_sha256=_HASH_A,
        percentile_checksum=_HASH_A,
        n_sae_features=17,
    )
    producer = cast(dict[str, object], contract["producer"])
    producer.update(
        {
            "device_type": "cuda",
            "cuda_runtime": "12.8",
            "gpu_name": "test-gpu",
            "gpu_capability": [8, 9],
        }
    )
    remote.validate_remote_percentile_contract(contract, surface="contract")
    producer_mutation(producer)
    with pytest.raises(remote.RemotePercentileError, match="identity is malformed"):
        remote.validate_remote_percentile_contract(contract, surface="contract")


@pytest.mark.parametrize(
    "info,match",
    [
        (RemoteObjectInfo(cast(str, None), 1, _HASH_A), "path"),
        (_info("path", size=0), "byte length"),
        (_info("path", size=cast(int, True)), "byte length"),
        (_info("path", xet="A" * 64), "Xet hash"),
    ],
)
def test_remote_identity_is_strict(info: RemoteObjectInfo, match: str) -> None:
    with pytest.raises(remote.RemotePercentileError, match=match):
        remote._require_remote_identity(info, surface="object")


def test_remote_paths_claim_shapes_and_strict_json(tmp_path: Path) -> None:
    prefix = NormalizedRemotePrefix.parse("raw/pct/hash")
    with pytest.raises(remote.RemotePercentileError, match="escaped"):
        remote._relative_output_parts("other/object", prefix)
    assert remote._known_verified_claim(
        ("_verified_upload_claims", remote.REMOTE_CONTRACT_FILENAME, f"{_HASH_A}.json")
    )
    assert remote._known_verified_claim(
        (
            remote.REMOTE_RECEIPTS_DIRNAME,
            "_verified_upload_claims",
            "shard_00000.json",
            f"{_HASH_A}.json",
        )
    )
    assert not remote._known_verified_claim(("_verified_upload_claims", "bad", "bad"))
    with pytest.raises(remote.RemotePercentileError, match="malformed verified-upload"):
        remote._verified_claim_target(("bad",), prefix=prefix)
    with pytest.raises(remote.RemotePercentileError, match="strict JSON"):
        remote._strict_json_object(b'{"a":1,"a":2}', surface="test")
    with pytest.raises(remote.RemotePercentileError, match="JSON object"):
        remote._strict_json_object(b"[]", surface="test")
    assert remote._strict_json_object(b'{"a":1}', surface="test") == {"a": 1}


def test_download_json_requires_canonical_bytes(tmp_path: Path) -> None:
    bucket = FilesystemBucket(tmp_path / "bucket")
    path = bucket._path("control.json")
    path.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    info = bucket.stat("control.json")
    assert info is not None
    config = _config(tmp_path / "cfg")
    assert remote._download_json_object(
        bucket,
        info=info,
        config=config,
        surface="control",
    ) == {"a": 1, "b": 2}
    with pytest.raises(remote.RemotePercentileError, match="not canonical"):
        remote._download_json_object(
            bucket,
            info=info,
            config=config,
            surface="control",
            require_canonical=True,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "extra": 1},
        lambda value: {**value, "schema_version": True},
        lambda value: {**value, "remote_path": "wrong"},
        lambda value: {**value, "byte_length": 0},
    ],
)
def test_verified_claim_parser_rejects_mutations(
    tmp_path: Path,
    mutation,
) -> None:
    bucket = FilesystemBucket(tmp_path / "bucket")
    config = _config(tmp_path / "cfg")
    canonical = remote._join_remote(config.output_prefix, remote.REMOTE_CONTRACT_FILENAME)
    value = {
        "schema_version": 1,
        "remote_path": canonical,
        "byte_length": 5,
        "xet_hash": _HASH_A,
    }
    payload = remote._canonical_json_bytes(mutation(value)) + b"\n"
    path = f"claim-{len(payload)}.json"
    bucket._path(path).write_bytes(payload)
    info = bucket.stat(path)
    assert info is not None
    with pytest.raises(remote.RemotePercentileError, match="is invalid"):
        remote._download_verified_claim(
            bucket,
            claim_info=info,
            canonical_path=canonical,
            claimed_xet_hash=_HASH_A,
            config=config,
        )


def test_verified_claim_requires_canonical_newline(tmp_path: Path) -> None:
    bucket = FilesystemBucket(tmp_path / "bucket")
    config = _config(tmp_path / "cfg")
    canonical = remote._join_remote(config.output_prefix, remote.REMOTE_CONTRACT_FILENAME)
    value = {
        "schema_version": 1,
        "remote_path": canonical,
        "byte_length": 5,
        "xet_hash": _HASH_A,
    }
    bucket._path("claim.json").write_bytes(remote._canonical_json_bytes(value))
    info = bucket.stat("claim.json")
    assert info is not None
    with pytest.raises(remote.RemotePercentileError, match="is invalid"):
        remote._download_verified_claim(
            bucket,
            claim_info=info,
            canonical_path=canonical,
            claimed_xet_hash=_HASH_A,
            config=config,
        )


def _receipt_json() -> dict[str, object]:
    return remote.RemoteReceipt(
        contract_sha256=_HASH_A,
        shard_index=0,
        source=_info(f"raw/shard_00000_features{COMPRESSED_EXT}"),
        output=_info(f"raw/pct/hash/shard_00000{COMPRESSED_EXT}"),
        semantic_content_sha256=_HASH_B,
        n_values=0,
    ).to_json()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.pop("source"), "schema or keys"),
        (lambda value: value.__setitem__("schema_version", True), "schema or keys"),
        (lambda value: value.__setitem__("contract_sha256", "bad"), "contract SHA"),
        (lambda value: value.__setitem__("shard_index", True), "shard index"),
        (lambda value: value.__setitem__("shard_index", -1), "shard index"),
        (lambda value: value.__setitem__("n_values", True), "n_values"),
        (lambda value: value.__setitem__("source", []), "identity is malformed"),
        (lambda value: cast(dict, value["source"]).__setitem__("path", 1), "path is malformed"),
        (
            lambda value: cast(dict, value["source"]).__setitem__(
                "path", "../../shard_00000_features.safetensors.zst"
            ),
            "path is malformed",
        ),
        (
            lambda value: cast(dict, value["output"]).__setitem__(
                "path", r"raw\shard_00000.safetensors.zst"
            ),
            "path is malformed",
        ),
        (lambda value: value.__setitem__("output", []), "output identity"),
        (
            lambda value: cast(dict, value["output"]).__setitem__("semantic_content_sha256", "bad"),
            "semantic digest",
        ),
    ],
)
def test_receipt_parser_mutation_table(mutate, match: str) -> None:
    value = _receipt_json()
    mutate(value)
    with pytest.raises(remote.RemotePercentileError, match=match):
        remote.parse_remote_percentile_receipt(value, surface="receipt")


def test_manifest_parser_mutation_table(tmp_path: Path) -> None:
    config = _config(tmp_path)
    contract = remote._producer_contract(
        config,
        inventory_sha256=_HASH_B,
        percentile_checksum=_HASH_A,
        n_sae_features=17,
    )
    statistics = _info(
        remote._join_remote(config.output_prefix, remote.OVERLAY_STATISTICS_FILENAME)
    )
    manifest: dict[str, object] = {
        "checksum": _HASH_A,
        "checksum_prefix": _HASH_A[:16],
        "kind": "pct",
        "base_shard_content_hash": _HASH_B,
        "schema_version": 2,
        "metadata": remote._manifest_metadata(
            contract=contract,
            contract_sha256=_HASH_A,
            statistics_info=statistics,
        ),
    }
    remote.validate_remote_percentile_manifest(
        manifest,
        checksum=_HASH_A,
        contract=contract,
        contract_sha256=_HASH_A,
        statistics_info=statistics,
    )
    mutations = (
        (lambda value: value.pop("kind"), "unexpected or missing"),
        (lambda value: value.__setitem__("kind", "tok"), "does not identify"),
        (lambda value: value.__setitem__("schema_version", True), "does not identify"),
        (lambda value: value.__setitem__("base_shard_content_hash", "bad"), "hash is malformed"),
        (
            lambda value: cast(dict, value["metadata"]).__setitem__("receipt_count", 99),
            "metadata differs",
        ),
    )
    for mutate, match in mutations:
        changed = copy.deepcopy(manifest)
        mutate(changed)
        with pytest.raises(remote.RemotePercentileError, match=match):
            remote.validate_remote_percentile_manifest(
                changed,
                checksum=_HASH_A,
                contract=contract,
                contract_sha256=_HASH_A,
                statistics_info=statistics,
            )


def test_output_layout_parser_rejects_malformed_combinations(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_count=1)
    contract_path = remote._join_remote(config.output_prefix, remote.REMOTE_CONTRACT_FILENAME)
    contract = _info(contract_path)
    contract_claim = _claim_info(config, remote.REMOTE_CONTRACT_FILENAME)

    class ListingBucket:
        def __init__(self, listing: Sequence[RemoteObjectInfo]) -> None:
            self.listing = listing

        def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
            return self.listing

        def stat(self, remote_path: str) -> RemoteObjectInfo | None:
            raise AssertionError

        def download(self, remote_path: str, local_path: Path) -> None:
            raise AssertionError

        def upload(self, local_path: Path, remote_path: str) -> None:
            raise AssertionError

    def inspect(listing: list[RemoteObjectInfo]) -> remote.RemoteOutputState:
        return remote._inspect_output_state(
            ListingBucket(listing),
            config=config,
            desired_contract={},
            contract_sha256=_HASH_A,
            inventory=_inventory(1),
            checksum=_HASH_A,
            download_controls=False,
        )

    tok = _info(
        remote._join_remote(config.output_prefix, "tok", "consumer.bin"),
        size=cast(int, 0),
        xet="bad",
    )
    assert inspect([tok]).listing == (tok,)

    malformed = (
        ([_info(cast(str, None))], "path is malformed"),
        ([contract, contract], "duplicate output listing"),
        ([_info("outside/object")], "escaped"),
        (
            [_info(remote._join_remote(config.output_prefix, "shard_bad.safetensors.zst"))],
            "malformed direct",
        ),
        (
            [_info(remote._join_remote(config.output_prefix, "_receipts", "bad"))],
            "malformed receipt",
        ),
        ([_info(remote._join_remote(config.output_prefix, "unknown"))], "unknown object"),
        (
            [_claim_info(config, "shard_00000.safetensors.zst")],
            "without a remote contract",
        ),
        ([contract], "lack verified-upload claims"),
        (
            [
                contract,
                contract_claim,
                _info(remote._join_remote(config.output_prefix, f"shard_00001{COMPRESSED_EXT}")),
                _claim_info(config, f"shard_00001{COMPRESSED_EXT}"),
            ],
            "unexpected shard indices",
        ),
        (
            [
                contract,
                contract_claim,
                _info(remote._join_remote(config.output_prefix, "_receipts", "shard_00000.json")),
                _info(
                    remote._join_remote(
                        config.output_prefix,
                        "_receipts",
                        "_verified_upload_claims",
                        "shard_00000.json",
                        f"{_HASH_A}.json",
                    )
                ),
            ],
            "without payload",
        ),
        (
            [
                contract,
                contract_claim,
                _info(remote._join_remote(config.output_prefix, remote.MANIFEST_FILENAME)),
                _claim_info(config, remote.MANIFEST_FILENAME),
            ],
            "without complete",
        ),
        (
            [
                contract,
                contract_claim,
                _claim_info(config, remote.REMOTE_CONTRACT_FILENAME, xet=_HASH_B),
            ],
            "multiple verified-upload claims",
        ),
    )
    for listing, match in malformed:
        with pytest.raises(remote.RemotePercentileError, match=match):
            inspect(listing)


def test_direct_file_identity_pid_and_cleanup_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(remote.RemotePercentileError, match="cannot inspect"):
        remote._safe_regular_identity(missing, surface="test")
    body = tmp_path / "body"
    body.write_bytes(b"body")
    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(body)
    with pytest.raises(remote.RemotePercentileError, match="single-link"):
        remote._safe_regular_identity(body, surface="test")

    monkeypatch.setattr(
        os, "kill", lambda _pid, _signal: (_ for _ in ()).throw(ProcessLookupError())
    )
    assert not remote._pid_live_or_inaccessible(123)
    monkeypatch.setattr(os, "kill", lambda _pid, _signal: (_ for _ in ()).throw(PermissionError()))
    assert remote._pid_live_or_inaccessible(123)
    monkeypatch.setattr(os, "kill", lambda _pid, _signal: None)
    assert remote._pid_live_or_inaccessible(123)


def test_staging_lock_contention_directory_swap_and_safe_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    staging.open()
    contender = _staging(tmp_path)
    try:
        with pytest.raises(remote.RemotePercentileError, match="already in use"):
            contender.open()
        body = staging.config.staging_dir / "output_shard_00000.safetensors.zst"
        body.write_bytes(b"body")
        identity = remote._safe_regular_identity(body, surface="body")
        replacement = body.with_name("replacement")
        replacement.write_bytes(b"replacement")
        replacement.replace(body)
        with pytest.raises(remote.RemotePercentileError, match="changed before cleanup"):
            staging._safe_unlink(body, expected=identity)

        body.write_bytes(b"runner-owned")
        identity = remote._safe_regular_identity(body, surface="body")
        real_rename = os.rename

        def replace_between_stat_and_rename(src, dst, **kwargs):
            attacker = staging.config.staging_dir / "attacker-replacement"
            attacker.write_bytes(b"must survive")
            os.replace(attacker, body)
            return real_rename(src, dst, **kwargs)

        monkeypatch.setattr(os, "rename", replace_between_stat_and_rename)
        with pytest.raises(remote.RemotePercentileError, match="changed during quarantine"):
            staging._safe_unlink(body, expected=identity)
        assert not body.exists()
        first_quarantines = list(staging.config.staging_dir.glob(".*.quarantine.*"))
        assert len(first_quarantines) == 1 and first_quarantines[0].read_bytes() == b"must survive"
        first_quarantines[0].unlink()
        monkeypatch.setattr(os, "rename", real_rename)

        body.write_bytes(b"runner-owned-again")
        identity = remote._safe_regular_identity(body, surface="body")

        def replace_and_recreate_between_stat_and_rename(src, dst, **kwargs):
            attacker = staging.config.staging_dir / "second-attacker-replacement"
            attacker.write_bytes(b"quarantined replacement")
            os.replace(attacker, body)
            result = real_rename(src, dst, **kwargs)
            body.write_bytes(b"concurrent replacement")
            return result

        monkeypatch.setattr(os, "rename", replace_and_recreate_between_stat_and_rename)
        with pytest.raises(remote.RemotePercentileError, match="changed during quarantine"):
            staging._safe_unlink(body, expected=identity)
        assert body.read_bytes() == b"concurrent replacement"
        quarantines = list(staging.config.staging_dir.glob(".*.quarantine.*"))
        assert len(quarantines) == 1
        quarantines[0].unlink()
        monkeypatch.setattr(os, "rename", real_rename)

        original_dir = staging.config.staging_dir
        moved_dir = original_dir.with_name("moved-staging")
        original_dir.rename(moved_dir)
        original_dir.mkdir()
        replacement_body = original_dir / body.name
        replacement_body.write_bytes(b"must survive")
        with pytest.raises(remote.RemotePercentileError, match="no longer names"):
            staging._safe_unlink(replacement_body)
        assert replacement_body.read_bytes() == b"must survive"
    finally:
        staging.close()
        contender.close()


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_staging_lock_rejects_indirect_file(tmp_path: Path, kind: str) -> None:
    staging = _staging(tmp_path)
    staging.config.staging_dir.mkdir(parents=True)
    lock = staging.config.staging_dir / remote._LOCK_FILENAME
    target = tmp_path / "target"
    target.write_bytes(b"")
    if kind == "symlink":
        lock.symlink_to(target)
    else:
        lock.hardlink_to(target)
    with pytest.raises((remote.RemotePercentileError, OSError)):
        staging.open()
    assert staging.lock_fd is None


def test_staging_detects_lock_path_replacement_after_acquisition(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    staging.open()
    lock_path = staging.config.staging_dir / remote._LOCK_FILENAME
    moved_lock = lock_path.with_name("original-lock")
    try:
        lock_path.rename(moved_lock)
        lock_path.write_bytes(b"replacement")
        with pytest.raises(remote.RemotePercentileError, match="locked inode"):
            staging.assert_bound()
    finally:
        staging.close()


def test_staging_bound_and_cleanup_defensive_failures(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    staging.open()
    directory = staging.config.staging_dir
    lock_path = directory / remote._LOCK_FILENAME
    original_lock_identity = staging.lock_identity
    try:
        original_directory_fd = staging.directory_fd
        staging.directory_fd = None
        with pytest.raises(remote.RemotePercentileError, match="descriptor is closed"):
            staging.assert_bound()
        staging.directory_fd = original_directory_fd

        original_lock_fd = staging.lock_fd
        staging.lock_fd = None
        with pytest.raises(remote.RemotePercentileError, match="descriptor is closed"):
            staging.assert_bound()
        staging.lock_fd = original_lock_fd

        staging.lock_identity = None
        staging.assert_bound()
        staging.lock_identity = original_lock_identity

        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        with pytest.raises(remote.RemotePercentileError, match="outside the locked"):
            staging._safe_unlink(outside)

        missing = directory / "output_shard_00000.safetensors.zst"
        with pytest.raises(remote.RemotePercentileError, match="entry is unavailable"):
            staging._safe_unlink(missing)
        target = tmp_path / "link-target"
        target.write_bytes(b"target")
        missing.symlink_to(target)
        with pytest.raises(remote.RemotePercentileError, match="not a safe regular file"):
            staging._safe_unlink(missing)

        lock_path.unlink()
        with pytest.raises(remote.RemotePercentileError, match="lock became unavailable"):
            staging.assert_bound()
        lock_path.write_bytes(b"replacement")
    finally:
        staging.close()


def test_staging_directory_anchor_open_and_identity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = _staging(tmp_path / "unavailable")
    real_open = os.open

    def fail_directory_open(path, flags, *args, **kwargs):
        if Path(path) == unavailable.config.staging_dir:
            raise OSError("anchor failed")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_directory_open)
    with pytest.raises(remote.RemotePercentileError, match="could not be anchored"):
        unavailable.open()
    assert unavailable.directory_fd is None
    monkeypatch.undo()

    changed = _staging(tmp_path / "changed")
    other = tmp_path / "other-directory"
    other.mkdir()

    def open_other_directory(path, flags, *args, **kwargs):
        if Path(path) == changed.config.staging_dir:
            return real_open(other, flags, *args, **kwargs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_other_directory)
    with pytest.raises(remote.RemotePercentileError, match="changed while opening"):
        changed.open()
    assert changed.directory_fd is None


def test_staging_preflight_owner_capacity_and_unknown_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path, required_capacity=100)
    directory = staging.config.staging_dir
    directory.mkdir(parents=True)
    remote.atomic_json_dump(staging._owner_payload(), directory / remote._OWNER_FILENAME)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=100, free=0),
    )
    with pytest.raises(remote.RemotePercentileError, match="admission failed"):
        staging.preflight_without_mutation()

    monkeypatch.undo()
    staging.open(require_capacity=False)
    try:
        unknown = directory / "unknown"
        unknown.write_bytes(b"preserve")
        with pytest.raises(remote.RemotePercentileError, match="unknown staging entry"):
            staging.cleanup_success()
        assert unknown.read_bytes() == b"preserve"
    finally:
        staging.close()


def test_capacity_formula_profile_and_state_properties(tmp_path: Path) -> None:
    assert (
        remote.required_staging_capacity_bytes(
            _inventory(),
            max_output_shard_bytes=7,
            statistics_bytes=11,
        )
        == remote.CAPACITY_RESERVE_BYTES + 102 + 101 + 7 + 22
    )
    profile = remote.RemotePercentileProfile(config={}, inventory={}, staging={})
    assert profile.payload()["mode"] == "remote"
    empty = remote.RemoteOutputState(listing=())
    assert not empty.has_canonical_state
    assert not empty.complete(0)
    empty.claim_infos["claim"] = _info("claim")
    assert empty.has_canonical_state


def test_revalidate_remote_identity_detects_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    expected = _info("object")

    class Backend:
        def stat(self, _path: str) -> RemoteObjectInfo:
            return _info("object", size=2)

    state = remote.RemoteOutputState(listing=(expected,), contract_info=expected)
    with pytest.raises(remote.RemotePercentileError, match="identity changed"):
        remote._revalidate_remote_identities(
            cast(BucketBackend, Backend()),
            state,
            config=config,
        )


def test_revalidate_remote_identity_includes_verified_claims(tmp_path: Path) -> None:
    config = _config(tmp_path)
    claim = _info("claim")

    class Backend:
        def stat(self, _path: str) -> RemoteObjectInfo:
            return _info("claim", size=2)

    state = remote.RemoteOutputState(listing=(claim,), claim_infos={"object": claim})
    with pytest.raises(remote.RemotePercentileError, match="identity changed"):
        remote._revalidate_remote_identities(
            cast(BucketBackend, Backend()),
            state,
            config=config,
        )


def test_download_source_rejects_unsafe_download_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "cfg", expected_count=1)
    config.staging_dir.mkdir(parents=True)
    entry = _inventory(1)[0]

    def unsafe_download(_backend, **kwargs) -> None:
        path = kwargs["final_path"]
        path.write_bytes(b"body")
        info = path.stat()
        kwargs["verified_publication_callback"](
            os.stat_result(
                (
                    stat.S_IFDIR | 0o700,
                    info.st_ino,
                    info.st_dev,
                    2,
                    info.st_uid,
                    info.st_gid,
                    info.st_size,
                    info.st_atime,
                    info.st_mtime,
                    info.st_ctime,
                )
            )
        )

    monkeypatch.setattr(remote, "download_verified_object", unsafe_download)
    with pytest.raises(remote.RemotePercentileError, match="not a safe regular file"):
        remote._download_source(
            cast(BucketBackend, None),
            config=config,
            entry=entry,
            cancellation=remote.threading.Event(),
            assert_staging=lambda: None,
            unlink_staged=lambda _path, _identity: None,
        )


def test_download_source_binds_verified_length_over_stale_stat_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(tmp_path / "cfg", expected_count=1)
    config.staging_dir.mkdir(parents=True)
    entry = _inventory(1)[0]

    def stale_size_download(_backend, **kwargs) -> None:
        path = kwargs["final_path"]
        path.write_bytes(b"x" * entry.info.byte_length)
        info = path.stat()
        kwargs["verified_publication_callback"](
            os.stat_result(
                (
                    info.st_mode,
                    info.st_ino,
                    info.st_dev,
                    1,
                    info.st_uid,
                    info.st_gid,
                    entry.info.byte_length - 5,
                    info.st_atime,
                    info.st_mtime,
                    info.st_ctime,
                ),
                {
                    "st_atime_ns": info.st_atime_ns,
                    "st_mtime_ns": info.st_mtime_ns,
                    "st_ctime_ns": info.st_ctime_ns,
                },
            )
        )

    monkeypatch.setattr(remote, "download_verified_object", stale_size_download)
    _path, bound, disposition = remote._download_source(
        cast(BucketBackend, None),
        config=config,
        entry=entry,
        cancellation=remote.threading.Event(),
        assert_staging=lambda: None,
        unlink_staged=lambda _path, _identity: None,
    )

    assert disposition == "downloaded"
    assert bound.size == entry.info.byte_length
    assert any("binding the verified length" in record.message for record in caplog.records)


def test_write_receipt_is_timestamp_free_and_canonical(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = remote.parse_remote_percentile_receipt(_receipt_json(), surface="receipt")
    remote._write_receipt(path, receipt)
    assert path.read_bytes() == remote._canonical_json_bytes(receipt.to_json())
    identity = remote._safe_regular_identity(path, surface="receipt")
    assert isinstance(identity, RegularFileIdentity)


def _canonical_manifests(bucket: FilesystemBucket) -> list[Path]:
    return [
        path
        for path in bucket.root.rglob(remote.MANIFEST_FILENAME)
        if "_verified_upload_claims" not in path.parts
    ]


@pytest.mark.parametrize(
    "surface",
    ["download_source", "stable_read", "decode", "compute", "save"],
)
def test_fault_boundary_retains_resumable_state_and_joins_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    target_name = {
        "download_source": "_download_source",
        "stable_read": "read_feature_compressed_bytes_stable",
        "decode": "decode_percentile_feature_safetensors_bytes",
        "compute": "compute_percentile_overlay_values",
        "save": "save_overlay_shard",
    }[surface]
    original = getattr(remote, target_name)
    injected = False

    def fail_once(*args, **kwargs):
        nonlocal injected
        result = original(*args, **kwargs)
        if not injected:
            injected = True
            raise RuntimeError(f"injected after {surface}")
        return result

    monkeypatch.setattr(remote, target_name, fail_once)
    with pytest.raises(RuntimeError, match=surface):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert _canonical_manifests(bucket) == []
    monkeypatch.setattr(remote, target_name, original)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert not any(thread.name.startswith("percentile-remote-") for thread in threading.enumerate())


def test_streaming_workers_are_singleton_and_download_upload_really_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    staging_dir = tmp_path / "staging"
    cfg = _remote_cfg(feature_dir, staging_dir, resume=True)
    original_download = bucket.download
    original_upload = bucket.upload
    state_lock = threading.Lock()
    active = {"download": 0, "upload": 0}
    maximum = {"download": 0, "upload": 0}
    second_download_started = threading.Event()
    allow_second_copy = threading.Event()
    second_copy_ready = threading.Event()
    allow_second_finish = threading.Event()
    observed_overlap = False
    staged_body_counts: tuple[int, int] | None = None

    def enter(kind: str) -> None:
        with state_lock:
            active[kind] += 1
            maximum[kind] = max(maximum[kind], active[kind])

    def leave(kind: str) -> None:
        with state_lock:
            active[kind] -= 1

    def tracked_download(remote_path: str, local_path: Path) -> None:
        is_second_source = remote_path.endswith("shard_00001_features.safetensors.zst")
        enter("download")
        try:
            if is_second_source:
                second_download_started.set()
                assert allow_second_copy.wait(timeout=5)
            original_download(remote_path, local_path)
            if is_second_source:
                second_copy_ready.set()
                assert allow_second_finish.wait(timeout=5)
        finally:
            leave("download")

    def tracked_upload(local_path: Path, remote_path: str) -> None:
        nonlocal observed_overlap, staged_body_counts
        is_first_payload = (
            remote_path.endswith("shard_00000.safetensors.zst")
            and "_verified_upload_claims" not in remote_path
        )
        enter("upload")
        try:
            if is_first_payload:
                assert second_download_started.wait(timeout=5)
                allow_second_copy.set()
                assert second_copy_ready.wait(timeout=5)
                with state_lock:
                    observed_overlap = active == {"download": 1, "upload": 1}
                staged_names = [path.name for path in staging_dir.iterdir()]
                staged_body_counts = (
                    sum("_features.safetensors.zst" in name for name in staged_names),
                    sum(name.startswith("output_shard_") for name in staged_names),
                )
                allow_second_finish.set()
            original_upload(local_path, remote_path)
        finally:
            if is_first_payload:
                allow_second_copy.set()
                allow_second_finish.set()
            leave("upload")

    monkeypatch.setattr(bucket, "download", tracked_download)
    monkeypatch.setattr(bucket, "upload", tracked_upload)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert maximum == {"download": 1, "upload": 1}
    assert observed_overlap
    assert staged_body_counts is not None
    source_bodies, output_bodies = staged_body_counts
    assert 1 <= source_bodies <= 2
    assert output_bodies == 1


def test_completed_receipt_resume_hashes_all_raw_but_skips_its_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    checksum, _ = remote._statistics_checksum(stats_path)
    prefix = f"raw/pct/{checksum[:16]}"
    bucket.interrupt_after_upload = f"{prefix}/_receipts/shard_00000.json"
    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)

    original = remote.compute_percentile_overlay_values
    original_stable_read = remote.read_feature_compressed_bytes_stable
    computed: list[str] = []
    stable_reads: list[str] = []

    def record_compute(*args, **kwargs):
        computed.append(cast(str, kwargs["source_name"]))
        return original(*args, **kwargs)

    def record_stable_read(path: Path, **kwargs) -> bytes:
        stable_reads.append(path.name)
        return original_stable_read(path, **kwargs)

    monkeypatch.setattr(remote, "compute_percentile_overlay_values", record_compute)
    monkeypatch.setattr(
        remote,
        "read_feature_compressed_bytes_stable",
        record_stable_read,
    )
    bucket.operations.clear()
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert "shard_00000_features.safetensors.zst" not in computed
    assert len(computed) == 2
    assert set(stable_reads) == {
        f"shard_{index:05d}_features.safetensors.zst" for index in range(3)
    }


def test_stats_present_manifest_claim_resume_computes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    checksum, _ = remote._statistics_checksum(stats_path)
    prefix = f"raw/pct/{checksum[:16]}"
    bucket.interrupt_before_upload = f"{prefix}/{remote.MANIFEST_FILENAME}"
    with pytest.raises(KeyboardInterrupt, match="injected before"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert bucket._path(f"{prefix}/{remote.OVERLAY_STATISTICS_FILENAME}").is_file()
    assert not bucket._path(f"{prefix}/{remote.MANIFEST_FILENAME}").exists()

    def unexpected_compute(*_args, **_kwargs):
        raise AssertionError("completed receipts must skip recomputation")

    monkeypatch.setattr(remote, "compute_percentile_overlay_values", unexpected_compute)
    bucket.operations.clear()
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert (
        len(
            [
                path
                for operation, path in bucket.operations
                if operation == "download" and path.startswith("raw/shard_")
            ]
        )
        == 3
    )


def test_statistics_claim_mismatch_fails_before_another_immutable_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=1)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    cfg.remote.expected_shard_count = 1
    checksum, _ = remote._statistics_checksum(stats_path)
    statistics_remote_path = f"raw/pct/{checksum[:16]}/{remote.OVERLAY_STATISTICS_FILENAME}"
    original_inspect = remote._inspect_output_state

    def inject_conflicting_claim(*args, **kwargs):
        state = original_inspect(*args, **kwargs)
        if len(state.receipts) == 1 and state.statistics_info is None:
            state.claimed_objects[statistics_remote_path] = _info(
                statistics_remote_path,
                xet=_HASH_A,
            )
        return state

    monkeypatch.setattr(remote, "_inspect_output_state", inject_conflicting_claim)
    with pytest.raises(remote.RemotePercentileError, match="staged statistics bytes differ"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert _canonical_manifests(bucket) == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("segment_size", 17, "segment_size"),
        ("remote.max_output_shard_bytes", 1, "max_output_shard_bytes"),
    ],
)
def test_runner_fails_before_payload_publication_on_decode_or_size_contract(
    tmp_path: Path,
    field: str,
    value: int,
    match: str,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    OmegaConf.update(cfg, field, value)
    with pytest.raises(remote.RemotePercentileError, match=match):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert not any(
        operation == "upload" and path.endswith(COMPRESSED_EXT)
        for operation, path in bucket.operations
    )
    assert _canonical_manifests(bucket) == []


def test_staging_directory_swap_before_first_remote_write_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    staging_dir = tmp_path / "staging"
    cfg = _remote_cfg(feature_dir, staging_dir, resume=True)
    original = remote.atomic_write_bytes

    def swap_after_contract(payload: bytes, path: Path) -> None:
        original(payload, path)
        if path.name == remote.REMOTE_CONTRACT_FILENAME:
            moved = staging_dir.with_name("moved-staging")
            staging_dir.rename(moved)
            staging_dir.mkdir()

    monkeypatch.setattr(remote, "atomic_write_bytes", swap_after_contract)
    with pytest.raises(remote.RemotePercentileError, match="no longer names"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert [operation for operation, _path in bucket.operations if operation == "upload"] == []


@pytest.mark.parametrize("surface", ["control", "claim"])
def test_downloaded_control_read_failure_is_attributed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    bucket = FilesystemBucket(tmp_path / "bucket")
    config = _config(tmp_path / "cfg")
    canonical = remote._join_remote(config.output_prefix, remote.REMOTE_CONTRACT_FILENAME)
    if surface == "control":
        remote_path = "control.json"
        bucket._path(remote_path).write_bytes(b"{}")
    else:
        value = {
            "schema_version": 1,
            "remote_path": canonical,
            "byte_length": 2,
            "xet_hash": _HASH_A,
        }
        remote_path = "claim.json"
        bucket._path(remote_path).write_bytes(remote._canonical_json_bytes(value) + b"\n")
    info = bucket.stat(remote_path)
    assert info is not None
    original = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path.parent.name.startswith("pct-remote-"):
            raise OSError("injected unreadable control")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    if surface == "control":
        with pytest.raises(remote.RemotePercentileError, match="not readable canonical JSON"):
            remote._download_json_object(
                bucket,
                info=info,
                config=config,
                surface="control",
            )
    else:
        with pytest.raises(remote.RemotePercentileError, match="is unreadable"):
            remote._download_verified_claim(
                bucket,
                claim_info=info,
                canonical_path=canonical,
                claimed_xet_hash=_HASH_A,
                config=config,
            )


def test_output_state_claim_identity_contract_and_receipt_mismatches(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "cfg", expected_count=1)
    inventory = _inventory(1)
    common = {
        "config": config,
        "inventory": inventory,
        "checksum": _HASH_A,
        "download_controls": True,
    }

    # A claim whose filename Xet matches the canonical body but whose JSON size
    # does not is authenticated and then rejected against the canonical listing.
    claim_bucket = FilesystemBucket(tmp_path / "claim-bucket")
    contract_path = remote._join_remote(config.output_prefix, remote.REMOTE_CONTRACT_FILENAME)
    contract_body = claim_bucket._path(contract_path)
    contract_body.parent.mkdir(parents=True)
    contract_body.write_bytes(b"x")
    canonical_info = claim_bucket.stat(contract_path)
    assert canonical_info is not None
    claim_value = {
        "schema_version": 1,
        "remote_path": contract_path,
        "byte_length": canonical_info.byte_length + 1,
        "xet_hash": canonical_info.xet_hash,
    }
    claim_path = remote._join_remote(
        config.output_prefix,
        "_verified_upload_claims",
        remote.REMOTE_CONTRACT_FILENAME,
        f"{canonical_info.xet_hash}.json",
    )
    claim_body = claim_bucket._path(claim_path)
    claim_body.parent.mkdir(parents=True)
    claim_body.write_bytes(remote._canonical_json_bytes(claim_value) + b"\n")
    with pytest.raises(remote.RemotePercentileError, match="claim identity differs"):
        remote._inspect_output_state(
            claim_bucket,
            desired_contract={},
            contract_sha256=remote._sha256_json({}),
            **common,
        )

    # Publish a valid claimed contract so control-parser mismatches are reached.
    contract_bucket = FilesystemBucket(tmp_path / "contract-bucket")
    desired_contract = {"contract": "desired"}
    contract_local = tmp_path / "contract.json"
    contract_local.write_bytes(remote._canonical_json_bytes(desired_contract))
    remote.upload_verified_object(
        contract_bucket,
        local_path=contract_local,
        remote_path=contract_path,
        timeout_seconds=1,
        retry_interval_seconds=0.001,
    )
    with pytest.raises(remote.RemotePercentileError, match="contract differs"):
        remote._inspect_output_state(
            contract_bucket,
            desired_contract={"contract": "other"},
            contract_sha256=remote._sha256_json({"contract": "other"}),
            **common,
        )


def test_output_state_rejects_unexpected_claim_target(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_count=1)
    contract = _info(remote._join_remote(config.output_prefix, remote.REMOTE_CONTRACT_FILENAME))
    listing = [
        contract,
        _claim_info(config, remote.REMOTE_CONTRACT_FILENAME),
        _claim_info(config, f"shard_00001{COMPRESSED_EXT}"),
    ]

    class Bucket:
        def list(self, _prefix: str):
            return listing

    with pytest.raises(remote.RemotePercentileError, match="unexpected canonical objects"):
        remote._inspect_output_state(
            cast(BucketBackend, Bucket()),
            config=config,
            desired_contract={},
            contract_sha256=_HASH_A,
            inventory=_inventory(1),
            checksum=_HASH_A,
            download_controls=False,
        )


def _replace_claimed_json(
    bucket: FilesystemBucket,
    remote_path: str,
    value: dict[str, object],
) -> None:
    canonical = bucket._path(remote_path)
    canonical.write_bytes(remote._canonical_json_bytes(value))
    observed = bucket.stat(remote_path)
    assert observed is not None
    relative = Path(remote_path)
    if relative.parent.name == remote.REMOTE_RECEIPTS_DIRNAME:
        claim_dir = bucket._path(
            f"{relative.parent.as_posix()}/_verified_upload_claims/{relative.name}"
        )
    else:
        claim_dir = bucket._path(
            f"{relative.parent.as_posix()}/_verified_upload_claims/{relative.name}"
        )
    if claim_dir.exists():
        shutil.rmtree(claim_dir)
    claim_dir.mkdir(parents=True)
    claim_value = {
        "schema_version": 1,
        "remote_path": remote_path,
        "byte_length": observed.byte_length,
        "xet_hash": observed.xet_hash,
    }
    (claim_dir / f"{observed.xet_hash}.json").write_bytes(
        remote._canonical_json_bytes(claim_value) + b"\n"
    )


def test_complete_family_rejects_consistently_reclaimed_receipt_tamper(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    manifest_path = _canonical_manifests(bucket)[0]
    prefix = manifest_path.parent.relative_to(bucket.root).as_posix()
    receipt_path = f"{prefix}/_receipts/shard_00000.json"
    receipt = json.loads(bucket._path(receipt_path).read_text(encoding="utf-8"))
    receipt["contract_sha256"] = _HASH_A
    _replace_claimed_json(bucket, receipt_path, receipt)
    with pytest.raises(remote.RemotePercentileError, match="disagrees with sealed identities"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_complete_fast_path_semantically_validates_scoped_statistics(
    tmp_path: Path,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    manifest_path = _canonical_manifests(bucket)[0]
    prefix = manifest_path.parent.relative_to(bucket.root).as_posix()
    statistics_remote_path = f"{prefix}/{remote.OVERLAY_STATISTICS_FILENAME}"
    statistics_path = bucket._path(statistics_remote_path)
    statistics_path.write_bytes(b"fully claimed but unreadable statistics")
    observed = bucket.stat(statistics_remote_path)
    assert observed is not None
    statistics_claim_dir = bucket._path(
        f"{prefix}/_verified_upload_claims/{remote.OVERLAY_STATISTICS_FILENAME}"
    )
    shutil.rmtree(statistics_claim_dir)
    statistics_claim_dir.mkdir(parents=True)
    (statistics_claim_dir / f"{observed.xet_hash}.json").write_bytes(
        remote._canonical_json_bytes(
            {
                "schema_version": 1,
                "remote_path": statistics_remote_path,
                "byte_length": observed.byte_length,
                "xet_hash": observed.xet_hash,
            }
        )
        + b"\n"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["statistics_object"] = {
        "path": observed.path,
        "byte_length": observed.byte_length,
        "xet_hash": observed.xet_hash,
    }
    manifest_remote_path = f"{prefix}/{remote.MANIFEST_FILENAME}"
    _replace_claimed_json(bucket, manifest_remote_path, manifest)

    bucket.operations.clear()
    with pytest.raises(remote.RemotePercentileError, match="statistics copy .* is unreadable"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert ("download", statistics_remote_path) in bucket.operations
    assert not [
        path
        for operation, path in bucket.operations
        if operation == "download" and path.startswith("raw/shard_")
    ]


def test_revalidate_empty_state_is_a_noop(tmp_path: Path) -> None:
    class Backend:
        def stat(self, _path: str):
            raise AssertionError("no canonical objects should be stated")

    remote._revalidate_remote_identities(
        cast(BucketBackend, Backend()),
        remote.RemoteOutputState(listing=()),
        config=_config(tmp_path),
    )


def test_staging_unbound_missing_special_and_owner_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path / "unbound")
    staging.assert_bound()
    staging.open()
    moved = staging.config.staging_dir.with_name("moved")
    staging.config.staging_dir.rename(moved)
    try:
        with pytest.raises(remote.RemotePercentileError, match="became unavailable"):
            staging.assert_bound()
    finally:
        staging.close()

    special = _staging(tmp_path / "special")
    special.config.staging_dir.parent.mkdir(parents=True, exist_ok=True)
    special.config.staging_dir.write_bytes(b"file")
    with pytest.raises(remote.RemotePercentileError, match="real directory"):
        special.preflight_without_mutation()
    with pytest.raises(remote.RemotePercentileError, match="real directory"):
        special.open()

    owner = _staging(tmp_path / "owner")
    owner.config.staging_dir.mkdir(parents=True)
    remote.atomic_json_dump({"wrong": True}, owner.config.staging_dir / remote._OWNER_FILENAME)
    with pytest.raises(remote.RemotePercentileError, match="belongs to another contract"):
        owner.preflight_without_mutation()
    with pytest.raises(remote.RemotePercentileError, match="belongs to another contract"):
        owner.open()
    owner.close()

    unreadable = _staging(tmp_path / "unreadable")
    unreadable.config.staging_dir.mkdir(parents=True)
    owner_path = unreadable.config.staging_dir / remote._OWNER_FILENAME
    remote.atomic_json_dump(unreadable._owner_payload(), owner_path)
    original = Path.read_bytes

    def fail_owner(path: Path) -> bytes:
        if path == owner_path:
            raise OSError("unreadable owner")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_owner)
    with pytest.raises(remote.RemotePercentileError, match="owner record is unreadable"):
        unreadable.preflight_without_mutation()
    with pytest.raises(remote.RemotePercentileError, match="owner record is unreadable"):
        unreadable.open()
    unreadable.close()


def test_staging_temporary_liveness_mutation_and_ownerless_preservation(
    tmp_path: Path,
) -> None:
    live = _staging(tmp_path / "live")
    live.config.staging_dir.mkdir(parents=True)
    remote.atomic_json_dump(live._owner_payload(), live.config.staging_dir / remote._OWNER_FILENAME)
    live_temp = live.config.staging_dir / (
        f".{remote.REMOTE_CONTRACT_FILENAME}.remote-tmp.{os.getpid()}.deadbeef"
    )
    live_temp.write_bytes(b"live")
    with pytest.raises(remote.RemotePercentileError, match="live or inaccessible"):
        live.preflight_without_mutation()

    dead = _staging(tmp_path / "dead")
    dead.config.staging_dir.mkdir(parents=True)
    remote.atomic_json_dump(dead._owner_payload(), dead.config.staging_dir / remote._OWNER_FILENAME)
    dead_temp = dead.config.staging_dir / (
        f".{remote.REMOTE_CONTRACT_FILENAME}.remote-tmp.99999999.deadbeef"
    )
    dead_temp.write_bytes(b"dead")
    assert dead.preflight_without_mutation()[1] >= 4
    dead.open()
    try:
        assert not dead_temp.exists()
    finally:
        dead.close()

    ownerless = _staging(tmp_path / "ownerless")
    ownerless.config.staging_dir.mkdir(parents=True)
    known = ownerless.config.staging_dir / remote.REMOTE_CONTRACT_FILENAME
    known.write_bytes(b"preserve")
    with pytest.raises(remote.RemotePercentileError, match="no owner record"):
        ownerless.open()
    assert known.read_bytes() == b"preserve"
    ownerless.close()


def test_staging_open_capacity_nofollow_postmkdir_and_lock_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = _staging(tmp_path / "capacity", required_capacity=100)
    capacity.config.staging_dir.mkdir(parents=True)
    remote.atomic_json_dump(
        capacity._owner_payload(),
        capacity.config.staging_dir / remote._OWNER_FILENAME,
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=100, free=0),
    )
    with pytest.raises(remote.RemotePercentileError, match="admission failed before remote write"):
        capacity.open()
    capacity.close()
    monkeypatch.undo()

    nofollow = _staging(tmp_path / "nofollow")
    monkeypatch.delattr(remote.os, "O_NOFOLLOW")
    with pytest.raises(remote.RemotePercentileError, match="requires .*O_NOFOLLOW"):
        nofollow.open()
    monkeypatch.undo()

    raced = _staging(tmp_path / "mkdir-race")
    staging_path = raced.config.staging_dir
    real_mkdir = Path.mkdir

    def symlink_after_mkdir(path: Path, *args, **kwargs) -> None:
        real_mkdir(path, *args, **kwargs)
        if path == staging_path:
            actual = staging_path.with_name("actual-staging")
            staging_path.rename(actual)
            staging_path.symlink_to(actual, target_is_directory=True)

    monkeypatch.setattr(Path, "mkdir", symlink_after_mkdir)
    with pytest.raises(remote.RemotePercentileError, match="real directory"):
        raced.open()
    monkeypatch.undo()

    replaced = _staging(tmp_path / "lock-replaced")
    lock_path = replaced.config.staging_dir / remote._LOCK_FILENAME
    real_lstat = Path.lstat
    lock_lstat_calls = 0

    def replace_before_second_lock_lstat(path: Path):
        nonlocal lock_lstat_calls
        if path == lock_path:
            lock_lstat_calls += 1
            if lock_lstat_calls == 2:
                moved_lock = lock_path.with_name("old-lock")
                lock_path.rename(moved_lock)
                lock_path.write_bytes(b"replacement")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replace_before_second_lock_lstat)
    with pytest.raises(remote.RemotePercentileError, match="changed while locking"):
        replaced.open()
    assert replaced.lock_fd is None


def test_staging_retained_corrupt_source_and_observe_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path / "retained")
    directory = staging.config.staging_dir
    directory.mkdir(parents=True)
    remote.atomic_json_dump(staging._owner_payload(), directory / remote._OWNER_FILENAME)
    source = directory / Path(staging.inventory[0].info.path).name
    source.write_bytes(b"wrong")
    staging.open()
    try:
        assert not source.exists()
    finally:
        staging.close()
    absent = _staging(tmp_path / "absent-observe")
    absent.observe()
    directory = absent.config.staging_dir
    directory.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def fail_iterdir(path: Path):
        if path == directory:
            raise OSError("iterdir failed")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)
    absent.observe()
    monkeypatch.undo()

    ignored_directory = directory / remote.MANIFEST_FILENAME
    ignored_directory.mkdir()
    absent.observe()
    assert absent.high_water_bodies == 0


def test_staging_discards_verified_sources_outside_initial_lookahead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path / "staging")
    directory = staging.config.staging_dir
    directory.mkdir(parents=True)
    remote.atomic_json_dump(staging._owner_payload(), directory / remote._OWNER_FILENAME)
    later = staging.inventory[2]
    retained = directory / Path(later.info.path).name
    retained.write_bytes(b"verified later lookahead")
    monkeypatch.setattr(
        remote,
        "verified_local_object_info",
        lambda **_kwargs: later.info,
    )
    staging.open()
    try:
        assert not retained.exists()
    finally:
        staging.close()


def test_staging_retains_only_initial_sources_and_first_missing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path / "bounded-retention")
    directory = staging.config.staging_dir
    directory.mkdir(parents=True)
    remote.atomic_json_dump(staging._owner_payload(), directory / remote._OWNER_FILENAME)
    source_paths = [directory / Path(entry.info.path).name for entry in staging.inventory]
    for index, source_path in enumerate(source_paths):
        source_path.write_bytes(f"source-{index}".encode())
    output_paths = [directory / f"output_shard_{index:05d}{COMPRESSED_EXT}" for index in range(2)]
    for index, output_path in enumerate(output_paths):
        output_path.write_bytes(f"output-{index}".encode())

    def verify_retained(*, local_path: Path, remote_path: str) -> RemoteObjectInfo:
        index = source_paths.index(local_path)
        assert remote_path == staging.inventory[index].info.path
        return staging.inventory[index].info

    monkeypatch.setattr(remote, "verified_local_object_info", verify_retained)
    staging.open()
    try:
        assert all(path.exists() for path in source_paths[:2])
        assert not source_paths[2].exists()
        assert output_paths[0].exists()
        assert not output_paths[1].exists()
    finally:
        staging.close()

    orphan_state = remote.RemoteOutputState(
        listing=(),
        payloads={0: _info("raw/pct/hash/shard_00000.safetensors.zst")},
    )
    orphan_staging = _staging(
        tmp_path / "orphan-before-future-output",
        output_state=orphan_state,
    )
    orphan_directory = orphan_staging.config.staging_dir
    orphan_directory.mkdir(parents=True)
    remote.atomic_json_dump(
        orphan_staging._owner_payload(),
        orphan_directory / remote._OWNER_FILENAME,
    )
    future_output = orphan_directory / f"output_shard_00001{COMPRESSED_EXT}"
    future_output.write_bytes(b"future output")
    orphan_staging.open()
    try:
        assert not future_output.exists()
    finally:
        orphan_staging.close()


def test_stable_overlay_read_rejects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / f"output_shard_00000{COMPRESSED_EXT}"
    remote.save_overlay_shard(
        {remote.TensorKey.FEATURE_PERCENTILES.value: t.tensor([1, 2], dtype=t.uint8)},
        path.with_name(path.name.removesuffix(COMPRESSED_EXT)),
    )
    original = path.read_bytes()
    real_lstat = Path.lstat
    replaced = False

    def replace_before_final_link_check(target: Path):
        nonlocal replaced
        if target == path and not replaced:
            replaced = True
            replacement = path.with_name("replacement")
            replacement.write_bytes(original)
            replacement.replace(path)
        return real_lstat(target)

    monkeypatch.setattr(Path, "lstat", replace_before_final_link_check)
    with pytest.raises(remote.RemotePercentileError, match="changed after stable read"):
        remote._read_percentile_overlay_stable(path, expected_numel=2)


def test_stable_overlay_read_rejects_open_identity_and_read_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / f"output_shard_00000{COMPRESSED_EXT}"
    remote.save_overlay_shard(
        {remote.TensorKey.FEATURE_PERCENTILES.value: t.tensor([1], dtype=t.uint8)},
        path,
    )
    identity = remote._safe_regular_identity(path, surface="test overlay")
    wrong_identity = replace(identity, size=identity.size + 1)
    with pytest.raises(remote.RemotePercentileError, match="changed while being opened"):
        remote._read_percentile_overlay_stable(
            path,
            expected_numel=1,
            expected_identity=wrong_identity,
        )

    hardlink = path.with_name("hardlink.safetensors.zst")
    hardlink.hardlink_to(path)
    with pytest.raises(remote.RemotePercentileError, match="single-link regular file"):
        remote._read_percentile_overlay_stable(path, expected_numel=1)
    hardlink.unlink()

    real_fdopen = os.fdopen

    def unlink_before_read(descriptor: int, *args, **kwargs):
        handle = real_fdopen(descriptor, *args, **kwargs)
        path.unlink()
        return handle

    monkeypatch.setattr(os, "fdopen", unlink_before_read)
    with pytest.raises(remote.RemotePercentileError, match="changed during stable read"):
        remote._read_percentile_overlay_stable(path, expected_numel=1)


def test_stable_overlay_read_rejects_platform_open_decode_and_schema_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.safetensors.zst"
    with pytest.raises(remote.RemotePercentileError, match="could not be opened safely"):
        remote._read_percentile_overlay_stable(missing, expected_numel=1)

    monkeypatch.delattr(remote.os, "O_NOFOLLOW")
    with pytest.raises(remote.RemotePercentileError, match="require os.O_NOFOLLOW"):
        remote._read_percentile_overlay_stable(missing, expected_numel=1)
    monkeypatch.undo()

    unreadable = tmp_path / "unreadable.safetensors.zst"
    unreadable.write_bytes(b"not an overlay")
    with pytest.raises(remote.RemotePercentileError, match="is unreadable"):
        remote._read_percentile_overlay_stable(unreadable, expected_numel=1)

    wrong_key = tmp_path / "wrong-key.safetensors.zst"
    remote.save_overlay_shard({"other": t.tensor([1], dtype=t.uint8)}, wrong_key)
    with pytest.raises(remote.RemotePercentileError, match="expected exactly"):
        remote._read_percentile_overlay_stable(wrong_key, expected_numel=1)

    wrong_tensor = tmp_path / "wrong-tensor.safetensors.zst"
    remote.save_overlay_shard(
        {remote.TensorKey.FEATURE_PERCENTILES.value: t.tensor([1], dtype=t.int16)},
        wrong_tensor,
    )
    with pytest.raises(remote.RemotePercentileError, match="rank-1 uint8"):
        remote._read_percentile_overlay_stable(wrong_tensor, expected_numel=1)


def test_stable_overlay_read_rejects_disappearing_final_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "overlay.safetensors.zst"
    remote.save_overlay_shard(
        {remote.TensorKey.FEATURE_PERCENTILES.value: t.tensor([1], dtype=t.uint8)},
        path,
    )
    real_lstat = Path.lstat

    def remove_before_final_link_check(target: Path):
        if target == path:
            target.unlink()
        return real_lstat(target)

    monkeypatch.setattr(Path, "lstat", remove_before_final_link_check)
    with pytest.raises(remote.RemotePercentileError, match="changed after stable read"):
        remote._read_percentile_overlay_stable(path, expected_numel=1)


def test_download_source_retained_cleanup_and_missing_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "cfg", expected_count=1)
    config.staging_dir.mkdir(parents=True)
    entry = _inventory(1)[0]
    path = remote._source_local_path(config, entry)
    path.write_bytes(b"retained")
    removed: list[Path] = []

    def remove(staged_path: Path, _identity: RegularFileIdentity) -> None:
        removed.append(staged_path)
        staged_path.unlink()

    def publish_download(_backend, **kwargs) -> None:
        final_path = kwargs["final_path"]
        final_path.write_bytes(b"downloaded")
        kwargs["verified_publication_callback"](final_path.stat())

    monkeypatch.setattr(
        remote,
        "verified_local_object_info",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("invalid retained")),
    )
    monkeypatch.setattr(remote, "download_verified_object", publish_download)
    result_path, _, disposition = remote._download_source(
        cast(BucketBackend, None),
        config=config,
        entry=entry,
        cancellation=threading.Event(),
        assert_staging=lambda: None,
        unlink_staged=remove,
    )
    assert result_path == path and disposition == "downloaded" and removed == [path]

    path.write_bytes(b"retained-again")
    monkeypatch.setattr(
        remote,
        "verified_local_object_info",
        lambda **_kwargs: _info(entry.info.path, size=999),
    )
    remote._download_source(
        cast(BucketBackend, None),
        config=config,
        entry=entry,
        cancellation=threading.Event(),
        assert_staging=lambda: None,
        unlink_staged=remove,
    )
    assert removed == [path, path]

    path.unlink()
    monkeypatch.setattr(remote, "download_verified_object", lambda *_args, **_kwargs: None)
    with pytest.raises(remote.RemotePercentileError, match="no bound identity"):
        remote._download_source(
            cast(BucketBackend, None),
            config=config,
            entry=entry,
            cancellation=threading.Event(),
            assert_staging=lambda: None,
            unlink_staged=remove,
        )


def test_publication_detects_staged_receipt_and_payload_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, expected_count=1)
    config.staging_dir.mkdir(parents=True)
    receipt = remote.parse_remote_percentile_receipt(_receipt_json(), surface="receipt")
    receipt_path = config.staging_dir / "receipt.json"

    def replace_receipt(_backend, **kwargs) -> RemoteObjectInfo:
        path = kwargs["local_path"]
        replacement = path.with_name("replacement-receipt")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return _info(kwargs["remote_path"], size=path.stat().st_size)

    monkeypatch.setattr(remote, "upload_verified_object", replace_receipt)
    with pytest.raises(remote.RemotePercentileError, match="receipt changed during upload"):
        remote._publish_receipt(
            cast(BucketBackend, None),
            config=config,
            local_path=receipt_path,
            receipt=receipt,
            assert_staging=lambda: None,
        )
    with pytest.raises(remote.RemotePercentileError, match="immutable upload claim"):
        remote._publish_receipt(
            cast(BucketBackend, None),
            config=config,
            local_path=receipt_path,
            receipt=receipt,
            expected_claim=_info(
                remote._receipt_remote_path(config, receipt.shard_index),
                xet=_HASH_B,
            ),
            assert_staging=lambda: None,
        )

    output_path = config.staging_dir / f"output_shard_00000{COMPRESSED_EXT}"
    output_path.write_bytes(b"payload")
    identity = remote._safe_regular_identity(output_path, surface="payload")
    wrong_identity = replace(identity, size=identity.size + 1)
    with pytest.raises(remote.RemotePercentileError, match="changed before upload"):
        remote._publish_payload_and_receipt(
            cast(BucketBackend, None),
            config=config,
            contract_sha256=_HASH_A,
            source=_inventory(1)[0].info,
            index=0,
            output_path=output_path,
            semantic_digest=_HASH_B,
            n_values=1,
            output_identity=wrong_identity,
            output_xet_hash=_HASH_A,
            assert_staging=lambda: None,
        )

    def replace_payload(_backend, **kwargs) -> RemoteObjectInfo:
        path = kwargs["local_path"]
        replacement = path.with_name("replacement-payload")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return _info(kwargs["remote_path"], size=path.stat().st_size)

    monkeypatch.setattr(remote, "upload_verified_object", replace_payload)
    identity = remote._safe_regular_identity(output_path, surface="payload")
    with pytest.raises(remote.RemotePercentileError, match="immutable upload claim"):
        remote._publish_payload_and_receipt(
            cast(BucketBackend, None),
            config=config,
            contract_sha256=_HASH_A,
            source=_inventory(1)[0].info,
            index=0,
            output_path=output_path,
            semantic_digest=_HASH_B,
            n_values=1,
            output_identity=identity,
            output_xet_hash=_HASH_A,
            expected_payload_claim=_info(
                remote._payload_remote_path(config, 0),
                xet=_HASH_B,
            ),
            assert_staging=lambda: None,
        )
    with pytest.raises(remote.RemotePercentileError, match="changed during upload"):
        remote._publish_payload_and_receipt(
            cast(BucketBackend, None),
            config=config,
            contract_sha256=_HASH_A,
            source=_inventory(1)[0].info,
            index=0,
            output_path=output_path,
            semantic_digest=_HASH_B,
            n_values=1,
            output_identity=identity,
            output_xet_hash=_HASH_A,
            assert_staging=lambda: None,
        )


def test_statistics_validation_prepare_materialize_and_update_fail_closed(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "stats", n_shards=0)
    checksum, width = remote._statistics_checksum(stats_path)
    with pytest.raises(remote.RemotePercentileError, match="table contract"):
        remote._validate_statistics_copy(
            stats_path,
            checksum=_HASH_A,
            n_sae_features=width,
        )
    with pytest.raises(remote.RemotePercentileError, match="lacks the contracted"):
        remote._validate_statistics_copy(
            stats_path,
            checksum=checksum,
            n_sae_features=width,
        )

    staging = _staging(tmp_path / "staging")
    staging.open()
    staged = staging.config.staging_dir / remote.OVERLAY_STATISTICS_FILENAME
    try:
        shutil.copyfile(stats_path, staged)
        remote._prepare_statistics_copy(
            stats_path=stats_path,
            staged_path=staged,
            checksum=checksum,
            n_sae_features=width,
            staging=staging,
        )
        remote._validate_statistics_copy(staged, checksum=checksum, n_sae_features=width)
        before = staged.read_bytes()
        remote._prepare_statistics_copy(
            stats_path=stats_path,
            staged_path=staged,
            checksum=checksum,
            n_sae_features=width,
            staging=staging,
        )
        assert staged.read_bytes() == before
    finally:
        staging.close()

    root_dir, root_stats = _make_env(tmp_path / "root", n_shards=0)
    root_statistics = load_statistics(root_stats)
    first_key = next(iter(root_statistics.per_feature_stats))
    first = root_statistics.per_feature_stats[first_key]
    changed_percentiles = dict(first.percentiles)
    changed_percentiles["p1"] -= 0.01
    changed_features = dict(root_statistics.per_feature_stats)
    changed_features[first_key] = replace(first, percentiles=changed_percentiles)
    changed_statistics = replace(root_statistics, per_feature_stats=changed_features)
    save_statistics(changed_statistics, root_stats)
    with pytest.raises(
        remote.RemotePercentileError, match="existing root statistics table differs"
    ):
        remote.materialize_overlay_statistics(
            overlay_statistics_path=staged,
            feature_root_statistics_path=root_stats,
            percentile_checksum=checksum,
        )
    with pytest.raises(
        remote.RemotePercentileError, match="local statistics percentile table changed"
    ):
        remote._update_local_statistics(
            root_stats,
            checksum=checksum,
            n_sae_features=width,
        )
    assert root_dir.is_dir() and feature_dir.is_dir()


def test_stable_statistics_validation_rejects_every_file_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stats_path = _make_env(tmp_path / "stats-races", n_shards=0)
    checksum, width = remote._statistics_checksum(stats_path)
    remote.save_percentile_checksum(stats_path, checksum)

    with monkeypatch.context() as scoped:
        scoped.delattr(remote.os, "O_NOFOLLOW")
        with pytest.raises(remote.RemotePercentileError, match="require os.O_NOFOLLOW"):
            remote._validate_statistics_copy(
                stats_path,
                checksum=checksum,
                n_sae_features=width,
            )

    with pytest.raises(remote.RemotePercentileError, match="is unreadable"):
        remote._validate_statistics_copy(
            tmp_path / "missing-statistics",
            checksum=checksum,
            n_sae_features=width,
        )

    hardlink = stats_path.with_name("hardlinked-statistics")
    hardlink.hardlink_to(stats_path)
    try:
        with pytest.raises(remote.RemotePercentileError, match="single-link regular file"):
            remote._validate_statistics_copy(
                stats_path,
                checksum=checksum,
                n_sae_features=width,
            )
    finally:
        hardlink.unlink()

    unreadable = tmp_path / "unreadable-statistics"
    unreadable.write_bytes(b"not safetensors")
    with pytest.raises(remote.RemotePercentileError, match="is unreadable"):
        remote._validate_statistics_copy(
            unreadable,
            checksum=checksum,
            n_sae_features=width,
        )

    real_fdopen = os.fdopen

    class MutatingHandle:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self.handle.__enter__()

        def __exit__(self, *args):
            result = self.handle.__exit__(*args)
            current = stats_path.stat()
            os.utime(
                stats_path,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
            )
            return result

    with monkeypatch.context() as scoped:
        scoped.setattr(
            os,
            "fdopen",
            lambda *args, **kwargs: MutatingHandle(real_fdopen(*args, **kwargs)),
        )
        with pytest.raises(remote.RemotePercentileError, match="changed during read"):
            remote._validate_statistics_copy(
                stats_path,
                checksum=checksum,
                n_sae_features=width,
            )

    replacement_target = tmp_path / "xet-race-statistics"
    shutil.copyfile(stats_path, replacement_target)
    real_hash_files = cast(Any, remote.hf_xet).hash_files

    def replace_after_hash(paths):
        result = real_hash_files(paths)
        replacement = replacement_target.with_name("xet-race-replacement")
        shutil.copyfile(replacement_target, replacement)
        replacement.replace(replacement_target)
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.hf_xet, "hash_files", replace_after_hash)
        with pytest.raises(remote.RemotePercentileError, match="stable validation"):
            remote._validate_statistics_copy(
                replacement_target,
                checksum=checksum,
                n_sae_features=width,
            )

    opened_descriptors: list[int] = []
    real_open = os.open

    def track_open(path, flags):
        descriptor = real_open(path, flags)
        opened_descriptors.append(descriptor)
        return descriptor

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "open", track_open)
        scoped.setattr(
            remote,
            "load_statistics_and_checksum_from_bytes",
            lambda _payload: (_ for _ in ()).throw(
                KeyboardInterrupt("interrupted statistics decode")
            ),
        )
        with pytest.raises(KeyboardInterrupt, match="interrupted statistics decode"):
            remote._validate_statistics_copy(
                stats_path,
                checksum=checksum,
                n_sae_features=width,
            )
    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


def test_prepare_statistics_copy_cleanup_failure_is_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stats_path = _make_env(tmp_path / "stats", n_shards=0)
    checksum, width = remote._statistics_checksum(stats_path)
    staging = _staging(tmp_path / "staging")
    staging.open()
    staged = staging.config.staging_dir / remote.OVERLAY_STATISTICS_FILENAME
    original_copy = shutil.copyfile

    def partial_copy(source: Path, destination: Path):
        original_copy(source, destination)
        raise RuntimeError("copy failed after materialization")

    def fail_unlink(_path: Path, *, expected=None) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(shutil, "copyfile", partial_copy)
    monkeypatch.setattr(staging, "_safe_unlink", fail_unlink)
    try:
        with pytest.raises(RuntimeError, match="copy failed") as caught:
            remote._prepare_statistics_copy(
                stats_path=stats_path,
                staged_path=staged,
                checksum=checksum,
                n_sae_features=width,
                staging=staging,
            )
        assert any("cleanup also failed" in note for note in getattr(caught.value, "__notes__", ()))
    finally:
        staging.close()


def test_runner_without_statistics_returns_nonzero(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert (
        remote.run_remote_percentiles(
            cfg,
            feature_dir,
            backend=cast(BucketBackend, None),
        )
        == 1
    )


def test_staging_retained_source_verifier_exception_and_observe_lstat_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path / "staging")
    directory = staging.config.staging_dir
    directory.mkdir(parents=True)
    remote.atomic_json_dump(staging._owner_payload(), directory / remote._OWNER_FILENAME)
    source = directory / Path(staging.inventory[0].info.path).name
    source.write_bytes(b"body")
    monkeypatch.setattr(
        remote,
        "verified_local_object_info",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("verification failed")),
    )
    staging.open()
    try:
        assert not source.exists()
        transient = directory / remote.MANIFEST_FILENAME
        transient.write_bytes(b"manifest")
        real_lstat = Path.lstat

        def disappear(path: Path):
            if path == transient:
                transient.unlink()
                raise FileNotFoundError(transient)
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", disappear)
        staging.observe()
    finally:
        staging.close()


def test_statistics_temporary_cleanup_failure_without_primary_is_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stats_path = _make_env(tmp_path / "stats", n_shards=0)
    checksum, width = remote._statistics_checksum(stats_path)
    staging = _staging(tmp_path / "staging")
    staging.open()
    staged = staging.config.staging_dir / remote.OVERLAY_STATISTICS_FILENAME
    monkeypatch.setattr(os, "replace", lambda _source, _destination: None)
    monkeypatch.setattr(
        staging,
        "_safe_unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    try:
        with pytest.raises(OSError, match="cleanup failed"):
            remote._prepare_statistics_copy(
                stats_path=stats_path,
                staged_path=staged,
                checksum=checksum,
                n_sae_features=width,
                staging=staging,
            )
    finally:
        staging.close()


def _make_orphan_run(tmp_path: Path) -> tuple[Path, FilesystemBucket, DictConfig]:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    bucket.interrupt_first_payload = True
    with pytest.raises(KeyboardInterrupt, match="first payload"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    return feature_dir, bucket, cfg


@pytest.mark.parametrize("failure", ["size", "tensor", "digest"])
def test_orphan_payload_recovery_rejects_every_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    feature_dir, bucket, cfg = _make_orphan_run(tmp_path)
    if failure == "size":
        cfg.remote.max_output_shard_bytes = 1
        expected = "max_output_shard_bytes"
    elif failure == "tensor":
        original_stable_read = remote._read_percentile_overlay_stable

        def different_tensor(*args, **kwargs):
            values, digest, identity, xet_hash = original_stable_read(*args, **kwargs)
            if Path(args[0]).name.startswith("verify_shard_"):
                values = values + 1
            return values, digest, identity, xet_hash

        monkeypatch.setattr(remote, "_read_percentile_overlay_stable", different_tensor)
        expected = "differs from recomputation"
    else:
        original_digest = remote.overlay_tensors_content_digest
        calls = 0

        def different_digest(tensors):
            nonlocal calls
            calls += 1
            result = original_digest(tensors)
            return result if calls == 1 else _HASH_A

        monkeypatch.setattr(remote, "overlay_tensors_content_digest", different_digest)
        expected = "semantic digest differs"
    with pytest.raises(remote.RemotePercentileError, match=expected):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert _canonical_manifests(bucket) == []


def _retain_saved_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, FilesystemBucket, DictConfig]:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original_save = remote.save_overlay_shard
    failed = False

    def fail_after_save(*args, **kwargs):
        nonlocal failed
        result = original_save(*args, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("after save")
        return result

    monkeypatch.setattr(remote, "save_overlay_shard", fail_after_save)
    with pytest.raises(RuntimeError, match="after save"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    monkeypatch.setattr(remote, "save_overlay_shard", original_save)
    return feature_dir, bucket, cfg


@pytest.mark.parametrize("failure", ["size", "tensor", "digest"])
def test_retained_output_recovery_rejects_every_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    feature_dir, bucket, cfg = _retain_saved_output(tmp_path, monkeypatch)
    if failure == "size":
        cfg.remote.max_output_shard_bytes = 1
        expected = "retained staged overlay"
    elif failure == "tensor":
        original_compute = remote.compute_percentile_overlay_values

        def different_compute(*args, **kwargs):
            values = original_compute(*args, **kwargs)
            return values.add(1)

        monkeypatch.setattr(remote, "compute_percentile_overlay_values", different_compute)
        expected = "retained staged output differs"
    else:
        original_digest = remote.overlay_tensors_content_digest
        calls = 0

        def different_digest(tensors):
            nonlocal calls
            calls += 1
            result = original_digest(tensors)
            return result if calls == 1 else _HASH_A

        monkeypatch.setattr(remote, "overlay_tensors_content_digest", different_digest)
        expected = "retained staged output digest differs"
    with pytest.raises(remote.RemotePercentileError, match=expected):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert _canonical_manifests(bucket) == []


def test_saved_output_semantic_digest_is_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original_digest = remote.overlay_tensors_content_digest
    calls = 0

    def inconsistent_digest(tensors):
        nonlocal calls
        calls += 1
        result = original_digest(tensors)
        return result if calls == 1 else _HASH_A

    monkeypatch.setattr(remote, "overlay_tensors_content_digest", inconsistent_digest)
    with pytest.raises(remote.RemotePercentileError, match="saved staged output digest differs"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert _canonical_manifests(bucket) == []


@pytest.mark.parametrize("n_shards", [1, 3])
def test_missing_published_output_identity_is_rejected_at_both_drain_sites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_shards: int,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=n_shards)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    cfg.remote.expected_shard_count = n_shards
    placeholder = remote.parse_remote_percentile_receipt(
        _receipt_json(),
        surface="receipt",
    )

    def missing_identity(_backend, **kwargs):
        output_path = kwargs["output_path"]
        return remote._PublishedFiles(
            receipt=placeholder,
            receipt_identity=remote._safe_regular_identity(output_path, surface="output"),
            output_identity=None,
        )

    monkeypatch.setattr(remote, "_publish_payload_and_receipt", missing_identity)
    with pytest.raises(remote.RemotePercentileError, match="did not bind"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)


def _completed_run(tmp_path: Path) -> tuple[Path, FilesystemBucket, DictConfig]:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    return feature_dir, bucket, cfg


def test_complete_fast_path_profiles_and_cleans_retained_staging(tmp_path: Path) -> None:
    feature_dir, bucket, cfg = _completed_run(tmp_path)
    retained = Path(cfg.remote.staging_dir) / remote.MANIFEST_FILENAME
    retained.write_bytes(b"recognized stale body")
    profile_path = tmp_path / "fast-profile.json"
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["totals"]["complete_fast_path"] is True
    assert not retained.exists()


@pytest.mark.parametrize("primary_failure", [False, True])
def test_complete_fast_path_close_failure_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_failure: bool,
) -> None:
    feature_dir, bucket, cfg = _completed_run(tmp_path)
    original_close = remote.PercentileStaging.close

    def close_then_fail(staging) -> None:
        original_close(staging)
        raise OSError("fast close failed")

    monkeypatch.setattr(remote.PercentileStaging, "close", close_then_fail)
    if not primary_failure:
        with pytest.raises(OSError, match="fast close failed"):
            remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
        return
    bucket.source_list_calls = 0
    bucket.mutate_source_on_list_call = 2
    with pytest.raises(remote.RemotePercentileError, match="source inventory changed") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert any("fast-path staging close also failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("root_call", [1, 2])
def test_statistics_drift_at_both_finalization_fences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_call: int,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    stats_path = feature_dir / remote.OVERLAY_STATISTICS_FILENAME
    original = remote._statistics_checksum
    calls = 0

    def drift(path: Path):
        nonlocal calls
        result = original(path)
        if path == stats_path:
            calls += 1
            if calls == root_call:
                return _HASH_A, result[1]
        return result

    monkeypatch.setattr(remote, "_statistics_checksum", drift)
    expected = "during the remote run" if root_call == 1 else "before manifest publication"
    with pytest.raises(remote.RemotePercentileError, match=expected):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert _canonical_manifests(bucket) == []


@pytest.mark.parametrize("staged_mode", ["missing", "different", "exception"])
def test_stats_present_resume_adopts_or_redownloads_exact_remote_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staged_mode: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    checksum, _ = remote._statistics_checksum(stats_path)
    prefix = f"raw/pct/{checksum[:16]}"
    bucket.interrupt_before_upload = f"{prefix}/{remote.MANIFEST_FILENAME}"
    with pytest.raises(KeyboardInterrupt):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    staged_stats = Path(cfg.remote.staging_dir) / remote.OVERLAY_STATISTICS_FILENAME
    if staged_mode == "missing":
        staged_stats.unlink()
    elif staged_mode == "different":
        staged_stats.write_bytes(b"different retained body")
    else:
        original_verify = remote.verified_local_object_info

        def fail_staged_stats(*, local_path: Path, remote_path: str):
            if local_path == staged_stats:
                raise OSError("retained verification failed")
            return original_verify(local_path=local_path, remote_path=remote_path)

        monkeypatch.setattr(remote, "verified_local_object_info", fail_staged_stats)
    bucket.operations.clear()
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert (
        "download",
        f"{prefix}/{remote.OVERLAY_STATISTICS_FILENAME}",
    ) in bucket.operations


@pytest.mark.parametrize(("list_call", "manifest_exists"), [(3, False), (4, True)])
def test_source_mutation_before_commit_blocks_manifest_and_during_commit_is_diagnosed(
    tmp_path: Path,
    list_call: int,
    manifest_exists: bool,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    bucket.mutate_source_on_list_call = list_call
    match = "before manifest publication" if list_call == 3 else "during manifest publication"
    with pytest.raises(remote.RemotePercentileError, match=match):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert bool(_canonical_manifests(bucket)) is manifest_exists


@pytest.mark.parametrize("final_fault", ["coverage", "base_hash"])
def test_final_output_fence_rejects_incomplete_or_wrong_manifest_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_fault: str,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original = remote._inspect_output_state
    calls = 0

    def corrupt_final(*args, **kwargs):
        nonlocal calls
        calls += 1
        state = original(*args, **kwargs)
        if calls == 3:
            if final_fault == "coverage":
                state.receipts.pop(0)
            else:
                assert state.manifest is not None
                state.manifest["base_shard_content_hash"] = _HASH_A
        return state

    monkeypatch.setattr(remote, "_inspect_output_state", corrupt_final)
    expected = "completion fence" if final_fault == "coverage" else "base hash differs"
    with pytest.raises(remote.RemotePercentileError, match=expected):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_intermediate_output_coverage_fence_rejects_missing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original = remote._inspect_output_state
    calls = 0

    def incomplete_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        state = original(*args, **kwargs)
        if calls == 2:
            state.receipts.pop(0)
        return state

    monkeypatch.setattr(remote, "_inspect_output_state", incomplete_second)
    with pytest.raises(remote.RemotePercentileError, match="coverage is incomplete"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_remote_training_collision_receives_exact_repair_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    monkeypatch.setattr(
        remote,
        "upload_verified_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            remote.RemoteTrainingError("verified-upload collision")
        ),
    )
    with pytest.raises(remote.RemoteTrainingError, match="collision") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert any("claim-only immutable object" in note for note in caught.value.__notes__)


def test_source_path_replacement_after_stable_read_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original = remote.read_feature_compressed_bytes_stable
    replaced = False

    def replace_after_read(path: Path, **kwargs) -> bytes:
        nonlocal replaced
        payload = original(path, **kwargs)
        if not replaced:
            replaced = True
            replacement = path.with_name("same-bytes-replacement")
            replacement.write_bytes(payload)
            replacement.replace(path)
        return payload

    monkeypatch.setattr(remote, "read_feature_compressed_bytes_stable", replace_after_read)
    with pytest.raises(remote.RemotePercentileError, match="changed before cleanup"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)


@pytest.mark.parametrize("multiple_cleanup_failures", [False, True])
def test_successful_pipeline_surfaces_cleanup_failures_without_losing_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multiple_cleanup_failures: bool,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original_close = remote.PercentileStaging.close

    def close_then_fail(staging) -> None:
        original_close(staging)
        raise OSError("primary cleanup failure")

    monkeypatch.setattr(remote.PercentileStaging, "close", close_then_fail)
    if multiple_cleanup_failures:
        cfg.profile = True
        cfg.profile_summary_output = str(tmp_path / "profile.json")
        monkeypatch.setattr(
            remote,
            "_write_profile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("secondary cleanup failure")
            ),
        )
    with pytest.raises(OSError, match="primary cleanup failure") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    if multiple_cleanup_failures:
        assert any(
            "additional remote percentile teardown failure" in note
            for note in caught.value.__notes__
        )


def test_processing_error_remains_primary_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original_close = remote.PercentileStaging.close

    def close_then_fail(staging) -> None:
        original_close(staging)
        raise OSError("cleanup also failed")

    monkeypatch.setattr(remote.PercentileStaging, "close", close_then_fail)
    monkeypatch.setattr(
        remote,
        "compute_percentile_overlay_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("processing failed")),
    )
    with pytest.raises(RuntimeError, match="processing failed") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert any("teardown also failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("future_exception_raises", [False, True])
def test_background_future_teardown_errors_are_retrieved_and_not_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    future_exception_raises: bool,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    processing_error = RuntimeError("future result failed")
    background_error = OSError("future exception retrieval failed")

    class FailedFuture:
        def result(self):
            raise processing_error

        def cancel(self) -> bool:
            return False

        def done(self) -> bool:
            return True

        def cancelled(self) -> bool:
            return False

        def exception(self):
            if future_exception_raises:
                raise background_error
            return background_error

    class FakeExecutor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            assert max_workers == 1
            self.thread_name_prefix = thread_name_prefix

        def submit(self, *_args, **_kwargs):
            if self.thread_name_prefix == "percentile-remote-download":
                return FailedFuture()
            raise AssertionError("upload worker must not be scheduled")

        def shutdown(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(remote, "ThreadPoolExecutor", FakeExecutor)
    with pytest.raises(RuntimeError, match="future result failed") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert any("future exception retrieval failed" in note for note in caught.value.__notes__)


def test_single_orphan_receipt_is_drained_without_an_output_body(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=1)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    cfg.remote.expected_shard_count = 1
    bucket.interrupt_first_payload = True
    with pytest.raises(KeyboardInterrupt, match="first payload"):
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0


def test_identical_processing_and_cleanup_error_is_not_self_noted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    shared_error = RuntimeError("shared processing cleanup failure")
    original_close = remote.PercentileStaging.close

    def fail_processing(*_args, **_kwargs):
        raise shared_error

    def close_with_same_error(staging) -> None:
        original_close(staging)
        raise shared_error

    monkeypatch.setattr(remote, "compute_percentile_overlay_values", fail_processing)
    monkeypatch.setattr(remote.PercentileStaging, "close", close_with_same_error)
    with pytest.raises(RuntimeError, match="shared processing cleanup failure") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert getattr(caught.value, "__notes__", ()) == ()


def test_identical_cleanup_error_is_not_added_as_its_own_secondary_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "profile.json")
    shared_error = OSError("shared cleanup failure")
    original_close = remote.PercentileStaging.close

    def close_with_shared_error(staging) -> None:
        original_close(staging)
        raise shared_error

    def profile_with_shared_error(*_args, **_kwargs) -> None:
        raise shared_error

    monkeypatch.setattr(remote.PercentileStaging, "close", close_with_shared_error)
    monkeypatch.setattr(remote, "_write_profile", profile_with_shared_error)
    with pytest.raises(OSError, match="shared cleanup failure") as caught:
        remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert getattr(caught.value, "__notes__", ()) == ()
