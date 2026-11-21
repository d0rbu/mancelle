"""Base-shard content-hash integrity for overlay manifests (Wave 3D task 3).

Overlays are keyed by a hash of *inputs* (pct table, tokenizer corpus). If a
base shard is silently re-extracted with different content, the overlay pairs
with a stale base shard. These tests pin the mitigation: overlay manifests
record a SHA-256 of the base shard bytes, and ``discover_overlay`` rejects
overlays whose base shard has since changed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch as t

from steering.overlay_io import (
    MANIFEST_FILENAME,
    BaseShardContentHasher,
    OverlayKind,
    OverlayStaleError,
    compute_base_shard_content_hash,
    discover_overlay,
    overlay_dir,
    save_overlay_shard,
    write_manifest,
)

FAKE_CHECKSUM = "a1b2c3d4e5f6a7b8" + "9" * 48  # 64-char hex
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _write_base_shard(feature_dir: Path, payload: dict[str, t.Tensor]) -> Path:
    """Write a base shard in *feature_dir* and return its path."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    return save_overlay_shard(payload, feature_dir / "shard_00000")


def _write_overlay(
    feature_dir: Path,
    base_shard_paths: list[Path],
    *,
    checksum: str = FAKE_CHECKSUM,
    kind: OverlayKind = OverlayKind.PERCENTILE,
) -> Path:
    """Write an overlay manifest that records the base shard content hash."""
    d = overlay_dir(feature_dir, kind, checksum)
    d.mkdir(parents=True, exist_ok=True)
    write_manifest(d, checksum, kind, base_shard_paths=base_shard_paths)
    return d


class TestBaseShardHashRecorded:
    def test_overlay_write_records_base_shard_hash(self, tmp_path):
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir, {"x": t.tensor([1, 2, 3])})
        d = _write_overlay(feature_dir, [base])

        manifest = json.loads((d / MANIFEST_FILENAME).read_text())
        h = manifest.get("base_shard_content_hash")
        assert h is not None, "manifest must record base_shard_content_hash"
        assert isinstance(h, str)
        assert HEX64_RE.match(h), f"expected 64-char lowercase hex, got {h!r}"


class TestStaleBaseShardRejected:
    def test_overlay_read_rejects_stale_base_shard(self, tmp_path):
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir, {"x": t.tensor([1, 2, 3])})
        _write_overlay(feature_dir, [base])

        # Overwrite the base shard with DIFFERENT content (same path).
        base.unlink()
        save_overlay_shard({"x": t.tensor([9, 9, 9, 9])}, feature_dir / "shard_00000")

        with pytest.raises(OverlayStaleError) as exc_info:
            discover_overlay(
                feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM,
                base_shard_paths=[base],
            )
        msg = str(exc_info.value).lower()
        assert "re-run extraction" in msg or "re-extract" in msg, (
            f"remediation text missing, got: {exc_info.value}"
        )


class TestUnchangedBaseShardAccepted:
    def test_overlay_read_accepts_unchanged_base_shard(self, tmp_path):
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir, {"x": t.tensor([1, 2, 3])})
        d = _write_overlay(feature_dir, [base])

        result = discover_overlay(
            feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM,
            base_shard_paths=[base],
        )
        assert result == d


class TestLegacyManifestWithoutHash:
    def test_legacy_overlay_without_hash_logs_warning(self, tmp_path, caplog):
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir, {"x": t.tensor([1, 2, 3])})
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        # Legacy manifest: no base_shard_content_hash field.
        (d / MANIFEST_FILENAME).write_text(
            json.dumps({"checksum": FAKE_CHECKSUM, "kind": "pct"}) + "\n"
        )

        # Loguru → caplog bridge: propagate to stdlib logging for caplog.
        import logging

        from loguru import logger as _logger

        handler_id = _logger.add(
            lambda m: logging.getLogger("loguru-bridge").warning(m.rstrip()),
            level="WARNING",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="loguru-bridge"):
                result = discover_overlay(
                    feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM,
                    base_shard_paths=[base],
                )
        finally:
            _logger.remove(handler_id)

        assert result == d, "legacy manifest must still be accepted"
        assert any(
            "base_shard_content_hash" in r.message or "legacy" in r.message.lower()
            for r in caplog.records
        ), f"expected warning about legacy manifest, got: {[r.message for r in caplog.records]}"


class TestSchemaVersionDowngradeAttack:
    def test_schema_v2_without_hash_is_rejected(self, tmp_path):
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir, {"x": t.tensor([1, 2, 3])})
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        (d / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "checksum": FAKE_CHECKSUM,
                    "kind": "pct",
                    "schema_version": 2,
                }
            )
            + "\n"
        )

        with pytest.raises(OverlayStaleError):
            discover_overlay(
                feature_dir,
                OverlayKind.PERCENTILE,
                FAKE_CHECKSUM,
                base_shard_paths=[base],
            )


class TestComputeBaseShardContentHash:
    def test_incremental_hasher_golden_digest(self):
        hasher = BaseShardContentHasher()
        hasher.update("shard_00000_features.safetensors.zst", b"")
        hasher.update("shard_00001_features.safetensors.zst", b"\x00\xffabc")
        hasher.update("shard_00002_features.safetensors.zst", b"payload")
        assert hasher.bytes_read == 12
        assert (
            hasher.hexdigest()
            == "f3c8978901e4ae9f76a120fdabad583bb9265417e2e9e2236c8b3543ead50aa2"
        )

    def test_deterministic(self, tmp_path):
        base = _write_base_shard(tmp_path, {"x": t.tensor([1, 2, 3])})
        h1 = compute_base_shard_content_hash([base])
        h2 = compute_base_shard_content_hash([base])
        assert h1 == h2
        assert HEX64_RE.match(h1)

    def test_incremental_hasher_matches_local_in_both_input_orders(self, tmp_path):
        a = save_overlay_shard({"x": t.tensor([1])}, tmp_path / "shard_00000")
        b = save_overlay_shard({"x": t.tensor([2])}, tmp_path / "shard_00001")
        for paths in ([a, b], [b, a]):
            incremental = BaseShardContentHasher()
            for path in sorted(paths):
                incremental.update(path.name, path.read_bytes())
            assert incremental.hexdigest() == compute_base_shard_content_hash(paths)

    def test_changes_with_content(self, tmp_path):
        d = tmp_path / "f"
        d.mkdir()
        a = save_overlay_shard({"x": t.tensor([1, 2, 3])}, d / "shard_00000")
        h1 = compute_base_shard_content_hash([a])
        a.unlink()
        save_overlay_shard({"x": t.tensor([4, 5, 6])}, d / "shard_00000")
        h2 = compute_base_shard_content_hash([d / "shard_00000.safetensors.zst"])
        assert h1 != h2

    def test_order_independent(self, tmp_path):
        d = tmp_path / "f"
        d.mkdir()
        a = save_overlay_shard({"x": t.tensor([1])}, d / "shard_00000")
        b = save_overlay_shard({"x": t.tensor([2])}, d / "shard_00001")
        assert compute_base_shard_content_hash([a, b]) == compute_base_shard_content_hash([b, a])


def test_discover_overlay_accepts_mutually_exclusive_precomputed_hash(tmp_path):
    feature_dir = tmp_path / "features"
    base = _write_base_shard(feature_dir, {"x": t.tensor([1, 2, 3])})
    digest = compute_base_shard_content_hash([base])
    overlay = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
    write_manifest(
        overlay,
        FAKE_CHECKSUM,
        OverlayKind.PERCENTILE,
        precomputed_base_hash=digest,
    )
    assert discover_overlay(
        feature_dir,
        OverlayKind.PERCENTILE,
        FAKE_CHECKSUM,
        precomputed_base_hash=digest,
    ) == overlay
    with pytest.raises(ValueError, match="mutually exclusive"):
        discover_overlay(
            feature_dir,
            OverlayKind.PERCENTILE,
            FAKE_CHECKSUM,
            base_shard_paths=[base],
            precomputed_base_hash=digest,
        )

    with pytest.raises(OverlayStaleError):
        discover_overlay(
            feature_dir,
            OverlayKind.PERCENTILE,
            FAKE_CHECKSUM,
            precomputed_base_hash="f" * 64,
        )


def test_discover_overlay_precomputed_hash_preserves_legacy_v1_acceptance(tmp_path):
    feature_dir = tmp_path / "features"
    overlay = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
    write_manifest(overlay, FAKE_CHECKSUM, OverlayKind.PERCENTILE)
    assert discover_overlay(
        feature_dir,
        OverlayKind.PERCENTILE,
        FAKE_CHECKSUM,
        precomputed_base_hash="a" * 64,
    ) == overlay


@pytest.mark.parametrize("bad_hash", ["f" * 63, "F" * 64, "g" * 64])
def test_precomputed_hash_validation_precedes_manifest_write(tmp_path, bad_hash):
    output = tmp_path / "overlay"
    with pytest.raises(ValueError, match="64 lowercase hex"):
        write_manifest(
            output,
            FAKE_CHECKSUM,
            OverlayKind.PERCENTILE,
            precomputed_base_hash=bad_hash,
        )
    assert not output.exists()
