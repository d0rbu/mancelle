"""Packed-row source certificate for compact feature prediction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import DictConfig, OmegaConf

import steering.conditional.encoder_decoder.fp_target_source as fp_target_source
from errors import PackedRowInvariantError
from scripts.preprocess.validate_fp_target_source import run as run_validation
from steering.conditional.encoder_decoder.fp_target_source import (
    FP_TARGET_SOURCE_INTEGRITY_KEY,
    certify_fp_target_source,
    fp_target_source_certificate_digest,
    validate_fp_target_source_certificate,
)
from steering.overlay_io import OverlayKind, read_manifest, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    save_packed_row_shard,
)
from steering.refined_types import PositiveInt

_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_N_SAE_FEATURES = PositiveInt.parse(8)


def _shard(feature_ids: list[int]) -> PackedRowShard:
    return PackedRowShard(
        tokens=t.arange(9, dtype=t.int32).to(t.uint16).reshape(1, 9),
        row_seg_offsets=t.tensor([0, 2], dtype=t.int32),
        seg_token_start=t.tensor([0, 4], dtype=t.uint16),
        seg_token_end=t.tensor([4, 8], dtype=t.uint16),
        seg_doc_id=t.tensor([0, 0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, len(feature_ids) - 1, len(feature_ids)], dtype=t.int32),
        feature_ids=t.tensor(feature_ids, dtype=t.uint16),
        feature_pcts=t.arange(1, len(feature_ids) + 1, dtype=t.int32).to(t.uint8),
    )


def _write_family(tmp_path: Path, feature_ids: list[int]) -> Path:
    packed = tmp_path / "packed"
    packed.mkdir()
    save_packed_row_shard(
        _shard(feature_ids), packed / "shard_00000.safetensors.zst"
    )
    write_manifest(
        packed,
        "a" * 64,
        OverlayKind.PACKED_ROW,
        metadata={
            "type": "packed_rows",
            "max_n_seg": 2,
        },
    )
    return packed


def test_complete_scan_writes_and_validates_certificate(tmp_path: Path):
    packed = _write_family(tmp_path, [1, 3, 7])
    certificate = certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    loaded, digest = validate_fp_target_source_certificate(
        packed, n_sae_features=8
    )

    assert loaded == certificate
    assert certificate["rows_checked"] == 1
    assert certificate["segments_checked"] == 2
    assert certificate["feature_entries_checked"] == 3
    assert certificate["unique_original_ids_per_segment"] is True
    assert digest == fp_target_source_certificate_digest(certificate)
    manifest = read_manifest(packed)
    assert manifest is not None
    assert manifest["metadata"][FP_TARGET_SOURCE_INTEGRITY_KEY] == certificate


def test_validation_script_certifies_with_parsed_positive_width(tmp_path: Path) -> None:
    packed = _write_family(tmp_path, [1, 3, 7])
    cfg = OmegaConf.create(
        {
            "packed_row_dir": str(packed),
            "n_sae_features": int(_N_SAE_FEATURES),
        }
    )
    assert isinstance(cfg, DictConfig)

    assert run_validation(cfg) == 0

    manifest = read_manifest(packed)
    assert manifest is not None
    assert (
        manifest["metadata"][FP_TARGET_SOURCE_INTEGRITY_KEY]["n_sae_features"]
        == _N_SAE_FEATURES
    )


def test_duplicate_original_id_fails_and_leaves_manifest_unchanged(tmp_path: Path):
    packed = _write_family(tmp_path, [2, 2, 7])
    manifest_path = packed / "manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(PackedRowInvariantError, match="duplicate.*ID 2"):
        certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    assert manifest_path.read_bytes() == before


def test_distinct_cluster_siblings_are_not_source_duplicates(tmp_path: Path):
    packed = _write_family(tmp_path, [1, 3, 5])
    certificate = certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    assert certificate["unique_original_ids_per_segment"] is True


def test_bounds_failure_leaves_manifest_unchanged(tmp_path: Path):
    packed = _write_family(tmp_path, [1, 9, 3])
    manifest_path = packed / "manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(PackedRowInvariantError, match=r"outside \[0, 8\)"):
        certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    assert manifest_path.read_bytes() == before


def test_wrong_packed_value_dtype_fails_before_certificate_write(tmp_path: Path):
    packed = _write_family(tmp_path, [1, 3, 5])
    malformed = replace(
        _shard([1, 3, 5]),
        feature_pcts=t.tensor([1, 2, 3], dtype=t.int16),
    )
    save_packed_row_shard(
        malformed,
        packed / "shard_00000.safetensors.zst",
    )
    manifest_path = packed / "manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(PackedRowInvariantError, match="feature_pcts.*uint8"):
        certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    assert manifest_path.read_bytes() == before


def test_missing_wrong_width_and_stale_source_fail_loudly(tmp_path: Path):
    packed = _write_family(tmp_path, [1, 3, 5])
    with pytest.raises(PackedRowInvariantError, match="requires a completed"):
        validate_fp_target_source_certificate(packed, n_sae_features=8)

    certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    with pytest.raises(PackedRowInvariantError, match="SAE-width mismatch"):
        validate_fp_target_source_certificate(packed, n_sae_features=7)

    save_packed_row_shard(
        _shard([1, 3, 6]), packed / "shard_00000.safetensors.zst"
    )
    with pytest.raises(PackedRowInvariantError, match="fingerprint is stale"):
        validate_fp_target_source_certificate(packed, n_sae_features=8)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("semantic_version", 2, "unknown FP target source semantic_version"),
        ("scan_complete", False, "scan_complete must be the literal boolean true"),
        (
            "unique_original_ids_per_segment",
            False,
            "unique_original_ids_per_segment must be the literal boolean true",
        ),
    ],
)
def test_incomplete_or_unknown_certificate_is_rejected(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
):
    packed = _write_family(tmp_path, [1, 3, 5])
    certify_fp_target_source(packed, n_sae_features=_N_SAE_FEATURES)
    manifest = read_manifest(packed)
    assert manifest is not None
    manifest["metadata"][FP_TARGET_SOURCE_INTEGRITY_KEY][key] = value
    (packed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(PackedRowInvariantError, match=match):
        validate_fp_target_source_certificate(packed, n_sae_features=8)


@_PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    )
)
def test_certification_rejects_every_invalid_sae_width(invalid_width: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        certify_fp_target_source(
            Path("unused-packed-row-source"),
            n_sae_features=cast(Any, invalid_width),
        )


def test_invalid_sae_width_fails_before_manifest_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_reads = 0

    def forbidden_manifest_read(_packed_dir: Path) -> None:
        nonlocal manifest_reads
        manifest_reads += 1
        raise AssertionError("invalid width must fail before artifact access")

    monkeypatch.setattr(fp_target_source, "read_manifest", forbidden_manifest_read)

    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        certify_fp_target_source(
            Path("unused-packed-row-source"),
            n_sae_features=cast(Any, 0),
        )

    assert manifest_reads == 0
