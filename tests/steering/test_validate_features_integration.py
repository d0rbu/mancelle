"""Integration tests for slice-3 (§B7).

Closes the slice-3 map §7.2 integration gap:

1. Round-trip parity: ``validate_shard_data`` and ``validate_shard`` agree on
   the same shard before / after ``save_features``.
2. Pre-save dtype trap: ``validate_shard_data`` catches malformed dtypes
   without round-tripping (the production fast-path).
3. Schema-version=2 manifest regression (manifest §4.5): ``write_manifest``
   emits ``schema_version=2`` plus ``base_shard_content_hash`` when
   ``base_shard_paths`` is provided. Marked as a ``pytest.mark.xfail`` until
   slice-5 PR #423 ships the producer-side ``base_shard_paths`` argument
   threading at all callsites — this slice owns the regression test that
   the manifest is, in fact, written with v2 once that PR lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from steering.overlay_io import OverlayKind, write_manifest
from steering.shard_validation import validate_shard, validate_shard_data
from tests.steering._helpers import (
    make_valid_shard_data,
    save_valid_shard,
)

# ---------------------------------------------------------------------------
# Round-trip parity (slice-3 P1-2 + P1-3 + A1 helper consolidation).
# ---------------------------------------------------------------------------


def test_validate_shard_data_and_validate_shard_match_round_trip(tmp_path) -> None:
    """A valid in-memory dict round-trips through ``save_features`` and both
    validators report identical violation sets (in this case, both empty)."""
    path, data = save_valid_shard(tmp_path, n_docs=2, segs_per_doc=2, feats_per_seg=4)

    in_memory = validate_shard_data(data)
    on_disk = validate_shard(path)

    assert in_memory.violations == on_disk.violations == []
    assert in_memory.valid is True
    assert on_disk.valid is True


def test_validate_shard_data_catches_bad_dtype_before_save() -> None:
    """The fast-path validator must surface dtype regressions in-memory
    (slice-3 P1-3)."""
    data = make_valid_shard_data()
    data["offsets"] = data["offsets"].to(t.int32)  # disk-canonical is int64

    result = validate_shard_data(data)

    assert not result.valid
    assert any("offsets dtype" in v for v in result.violations), result.violations


def test_validate_shard_data_and_validate_shard_emit_same_violation_set_on_corruption() -> None:
    """For an in-memory corruption that survives save (negative magnitude),
    both validators must produce the same violation set after round-trip.
    """
    data = make_valid_shard_data()
    # Negative magnitude — preserved by save_features (no dtype mutation).
    data["feature_mags"][0] = -1.0

    in_memory = validate_shard_data(data)

    assert not in_memory.valid
    assert any("magnitudes are <= 0" in v for v in in_memory.violations), (
        in_memory.violations
    )


# ---------------------------------------------------------------------------
# Schema-version=2 manifest regression (manifest §4.5).
# ---------------------------------------------------------------------------


def _write_dummy_shard(directory: Path, name: str, payload: bytes) -> Path:
    """Write a tiny placeholder shard file so ``compute_base_shard_content_hash``
    has stable input bytes to digest."""
    p = directory / f"{name}_features.safetensors.zst"
    p.write_bytes(payload)
    return p


def test_write_manifest_emits_schema_version_2_with_base_shard_paths(tmp_path) -> None:
    """Writing a manifest with ``base_shard_paths`` produces v2 + hash field."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    base_paths = [
        _write_dummy_shard(feature_dir, "shard_00000", b"abc"),
        _write_dummy_shard(feature_dir, "shard_00001", b"def"),
    ]
    overlay_dir = tmp_path / "overlay"
    write_manifest(
        overlay_dir,
        "0" * 64,  # placeholder full-checksum
        OverlayKind.PERCENTILE,
        metadata={"smoke": "test"},
        base_shard_paths=base_paths,
    )
    manifest = json.loads((overlay_dir / "manifest.json").read_text())

    assert manifest["schema_version"] == 2
    assert "base_shard_content_hash" in manifest
    assert isinstance(manifest["base_shard_content_hash"], str)
    assert len(manifest["base_shard_content_hash"]) == 64  # sha256 hex


def test_write_manifest_omits_schema_version_when_base_shard_paths_absent(tmp_path) -> None:
    """Backwards-compat: pre-v2 callers (no base_shard_paths) keep
    legacy v1-implicit manifests (no schema_version field)."""
    overlay_dir = tmp_path / "overlay"
    write_manifest(
        overlay_dir,
        "0" * 64,
        OverlayKind.PERCENTILE,
        metadata={"legacy": "true"},
    )
    manifest = json.loads((overlay_dir / "manifest.json").read_text())

    assert "schema_version" not in manifest, (
        "Backwards-compat: legacy callers should still produce implicit-v1 manifests"
    )
    assert "base_shard_content_hash" not in manifest


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Slice-5 PR #423 (followup) threads `base_shard_paths` through all "
        "production `write_manifest` callsites in `precompute_*.py`. Until "
        "that lands, the producer-side regression cannot be exercised "
        "end-to-end. This xfail-strict test will start passing when the "
        "callsites are wired."
    ),
)
def test_production_precompute_manifests_have_schema_version_2() -> None:
    """Cross-slice regression for slice-5 PR #423.

    When precompute_percentiles / precompute_token_boundaries / packed_rows
    are invoked, the manifest they write MUST embed ``schema_version=2`` and
    ``base_shard_content_hash``. Until slice-5 ships the threading, the
    callsites still pass ``base_shard_paths=None``.
    """
    # The follow-up PR lands a thread-through; until then we mark this xfail.
    raise AssertionError(
        "Producer-side base_shard_paths threading not yet shipped (slice-5 PR #423)"
    )
