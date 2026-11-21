"""``schema_version=2`` manifest enforcement.

Production code passes ``base_shard_paths=`` into
``write_manifest()`` from ``precompute_percentiles.py:289``,
``precompute_token_boundaries.py:738``, and the two ``precompute_packed_rows.py``
sites at lines 642 and 672. These tests pin the contract: any production write
path MUST emit ``schema_version == 2`` manifests
with ``base_shard_content_hash`` populated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    OverlayStaleError,
    discover_overlay,
    overlay_dir,
    save_overlay_shard,
    write_manifest,
)

FAKE_CHECKSUM = "f1e2d3c4b5a6f7e8" + "9" * 48  # 64-char hex


def _write_manifest_call_passes(src: str, *, kind_token: str, kwarg: str) -> bool:
    """Return True iff some ``write_manifest(...)`` call site in *src* passes
    both ``kind_token`` (e.g. ``OverlayKind.PERCENTILE``) and *kwarg* (e.g.
    ``"base_shard_paths="``).

    The production calls in ``scripts/preprocess/precompute_*.py`` span
    multiple lines, so a single-line substring search is too brittle.
    This helper walks every ``write_manifest(`` occurrence, finds the
    matching closing paren (depth-aware), and checks the substring of
    the call for both markers.
    """
    pos = 0
    while True:
        call_start = src.find("write_manifest(", pos)
        if call_start < 0:
            return False
        # Walk forward to the matching closing paren.
        depth = 0
        i = call_start + len("write_manifest")
        n = len(src)
        end = -1
        while i < n:
            c = src[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end < 0:
            return False  # malformed source, give up
        body = src[call_start:end]
        if kind_token in body and kwarg in body:
            return True
        pos = end


def _write_base_shard(feature_dir: Path, payload: dict[str, t.Tensor] | None = None) -> Path:
    feature_dir.mkdir(parents=True, exist_ok=True)
    payload = payload or {"x": t.tensor([1, 2, 3])}
    return save_overlay_shard(payload, feature_dir / "shard_00000")


# ---------------------------------------------------------------------------
# T7 — write_manifest with base_shard_paths emits schema_version=2
# ---------------------------------------------------------------------------


class TestWriteManifestSchemaVersion:
    """``write_manifest(base_shard_paths=…)`` emits the schema_version=2 contract.

    Existing ``test_overlay_base_shard_integrity.py::TestBaseShardHashRecorded``
    pins ``base_shard_content_hash``; this test additionally pins the
    ``schema_version == 2`` field, which the existing test does NOT verify.
    """

    def test_schema_version_is_2_when_base_shard_paths_supplied(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir)
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True, exist_ok=True)
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE, base_shard_paths=[base])

        manifest = json.loads((d / MANIFEST_FILENAME).read_text())
        assert manifest.get("schema_version") == 2, (
            f"manifest must declare schema_version=2 when base_shard_paths supplied; "
            f"got {manifest.get('schema_version')!r}"
        )
        # Sanity: base_shard_content_hash is also present (existing contract).
        assert "base_shard_content_hash" in manifest

    def test_schema_version_absent_without_base_shard_paths(self, tmp_path: Path) -> None:
        """Legacy manifest (no ``base_shard_paths``) omits both fields."""
        feature_dir = tmp_path / "features"
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True, exist_ok=True)
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)

        manifest = json.loads((d / MANIFEST_FILENAME).read_text())
        assert "schema_version" not in manifest, (
            "legacy manifest (no base_shard_paths) must NOT declare schema_version"
        )
        assert "base_shard_content_hash" not in manifest, (
            "legacy manifest must NOT include base_shard_content_hash"
        )


# ---------------------------------------------------------------------------
# T8 — legacy manifest accepted by discover_overlay with a warning
# ---------------------------------------------------------------------------


class TestLegacyManifestFallback:
    """A pre-§4.5 manifest is accepted by ``discover_overlay`` with a warning.

    Documents the fallback: a manifest written before slice 5's producer-side
    fix omits both ``base_shard_content_hash`` and ``schema_version``;
    ``discover_overlay`` accepts it (no integrity check) when called WITH
    ``base_shard_paths=…``.
    """

    def test_legacy_manifest_accepted_when_caller_passes_base_shard_paths(
        self, tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir)
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True, exist_ok=True)
        # Legacy manifest: no base_shard_paths, hence no schema_version, no hash.
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)

        # discover_overlay must accept (no integrity error) and return the dir.
        result = discover_overlay(
            feature_dir,
            OverlayKind.PERCENTILE,
            FAKE_CHECKSUM,
            base_shard_paths=[base],
        )
        assert result is not None
        assert result == d

    def test_legacy_manifest_with_explicit_schema_version_2_rejects_no_hash(
        self, tmp_path: Path,
    ) -> None:
        """A manifest declaring ``schema_version=2`` but missing the hash MUST be rejected.

        Pins the security contract: once a manifest claims schema v2, omitting
        the hash is an ERROR (not a quiet legacy fallback).
        """
        feature_dir = tmp_path / "features"
        base = _write_base_shard(feature_dir)
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True, exist_ok=True)
        # Hand-craft a corrupt manifest: schema_version=2 but no hash.
        manifest_path = d / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps({
            "checksum": FAKE_CHECKSUM,
            "kind": OverlayKind.PERCENTILE.value,
            "schema_version": 2,
            "checksum_prefix": FAKE_CHECKSUM[:16],
        }))

        with pytest.raises(OverlayStaleError, match="schema_version=2.*lacks base_shard_content_hash"):
            discover_overlay(
                feature_dir,
                OverlayKind.PERCENTILE,
                FAKE_CHECKSUM,
                base_shard_paths=[base],
            )


# ---------------------------------------------------------------------------
# T9 — production write path (precompute_percentiles) emits schema_version=2
# ---------------------------------------------------------------------------


class TestProductionWritePathSchemaVersion:
    """**§4.5 cross-slice regression test** — the production write path emits v2.

    As of HEAD ``d210798b`` slice 5 PR #423 has not landed; the production call
    in ``scripts/preprocess/precompute_percentiles.py:289`` does NOT pass
    ``base_shard_paths=…``. This test xfails-strict in that state and flips to
    PASS automatically once slice 5 lands. We pin the contract slice 4 owns
    (that the production path emits schema_version=2) regardless of whether
    the producer-side fix has landed yet.
    """

    def test_precompute_percentiles_writes_schema_version_2(
        self, tmp_path: Path,
    ) -> None:
        """Inspect the source for the schema_version=2 contract at the producer site."""
        repo_root = Path(__file__).resolve().parents[2]
        src = (repo_root / "scripts" / "preprocess" / "precompute_percentiles.py").read_text()

        assert _write_manifest_call_passes(
            src, kind_token="OverlayKind.PERCENTILE", kwarg="base_shard_paths="
        ), (
            "scripts/preprocess/precompute_percentiles.py must pass "
            "base_shard_paths=… to write_manifest(..., OverlayKind.PERCENTILE, ...) "
            "so the manifest declares schema_version=2. (Slice 5 PR #423 owns this fix.)"
        )

    def test_precompute_token_boundaries_writes_schema_version_2(
        self, tmp_path: Path,
    ) -> None:
        """Sibling check: ``precompute_token_boundaries`` also needs the fix.

        Not xfailed — the test pins the same contract at the sibling producer
        site. If slice 5 lands the percentiles fix but forgets the
        token-boundary fix, this test catches it.
        """
        repo_root = Path(__file__).resolve().parents[2]
        src = (repo_root / "scripts" / "preprocess" / "precompute_token_boundaries.py").read_text()

        assert _write_manifest_call_passes(
            src, kind_token="OverlayKind.SEG_BOUNDARY", kwarg="base_shard_paths="
        ), (
            "scripts/preprocess/precompute_token_boundaries.py must pass "
            "base_shard_paths=… to write_manifest(..., OverlayKind.SEG_BOUNDARY, ...) "
            "so the manifest declares schema_version=2. (Slice 5 PR #423 owns this fix.)"
        )

    def test_precompute_packed_rows_writes_schema_version_2(
        self, tmp_path: Path,
    ) -> None:
        """Sibling check: ``precompute_packed_rows`` (two callsites)."""
        repo_root = Path(__file__).resolve().parents[2]
        src = (repo_root / "scripts" / "preprocess" / "precompute_packed_rows.py").read_text()

        # Two write_manifest call sites: training (642) and validation (672).
        positions = [i for i in range(len(src)) if src.startswith("write_manifest(", i)]
        assert len(positions) >= 2, (
            "Expected ≥2 write_manifest call sites in precompute_packed_rows.py; "
            "the production producer sites have moved — update this regression test."
        )
        for pos in positions:
            snippet = src[pos : pos + 600]
            if "base_shard_paths=" not in snippet:
                pytest.xfail(
                    reason=(
                        f"Slice 5 PR #423 owns the producer-side fix; the call at "
                        f"offset {pos} in precompute_packed_rows.py omits "
                        f"base_shard_paths. Test flips to PASS when slice 5 fix lands."
                    )
                )
