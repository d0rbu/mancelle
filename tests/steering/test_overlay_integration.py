"""Integration tests for overlay-based precomputation pipeline.

Tests the full cascade: base shards → overlay discovery → FeatureIndex loading.
Also tests the migration script from embedded to overlay format.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

import steering.conditional.feature_index as feature_index_module
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.percentile_table import (
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
)
from steering.feature_io import (
    CorruptFeatureShardError,
    TensorKey,
    load_features,
    save_features,
)
from steering.overlay_io import (
    COMPRESSED_EXT,
    OverlayIncompleteError,
    OverlayKind,
    create_mirror_symlinks,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)

# Constants for test data
N_SAE = 200
N_DOCS = 4
N_SEGS_PER_DOC = 3
N_FEATS_PER_SEG = 5
SEGMENT_SIZE = 16


def _inventory(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _make_pct_table() -> t.Tensor:
    return t.linspace(0.01, 5.0, N_SAE * 99).reshape(N_SAE, 99).double()


def _make_base_shard(feature_dir: Path, shard_stem: str = "shard_00000") -> Path:
    """Create a minimal base feature shard (no embedded precomputed data)."""
    total_segs = N_DOCS * N_SEGS_PER_DOC
    total_feats = total_segs * N_FEATS_PER_SEG

    all_fids = t.zeros(total_feats, dtype=t.int32)
    all_mags = t.zeros(total_feats, dtype=t.float32)
    offsets = t.zeros(total_segs + 1, dtype=t.int64)
    doc_ids = t.zeros(total_segs, dtype=t.int32)
    seg_indices = t.zeros(total_segs, dtype=t.int32)
    seg_positions = t.zeros(total_segs, dtype=t.int32)
    seg_char_starts = t.zeros(total_segs, dtype=t.int32)
    seg_char_ends = t.zeros(total_segs, dtype=t.int32)

    feat_offset = 0
    seg_global = 0
    for doc_idx in range(N_DOCS):
        for seg_idx in range(N_SEGS_PER_DOC):
            doc_ids[seg_global] = doc_idx
            seg_indices[seg_global] = seg_idx
            seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
            offsets[seg_global] = feat_offset
            seg_char_starts[seg_global] = seg_idx * SEGMENT_SIZE * 6
            seg_char_ends[seg_global] = (seg_idx + 1) * SEGMENT_SIZE * 6
            for f in range(N_FEATS_PER_SEG):
                all_fids[feat_offset + f] = (doc_idx * 10 + seg_idx * 5 + f) % N_SAE
                all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx
            feat_offset += N_FEATS_PER_SEG
            seg_global += 1
    offsets[total_segs] = feat_offset

    shard_data = {
        "offsets": offsets,
        "feature_ids": all_fids,
        "feature_mags": all_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": t.full((N_DOCS,), 50, dtype=t.int32),
        "segment_char_starts": seg_char_starts,
        "segment_char_ends": seg_char_ends,
        "n_docs": N_DOCS,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }
    return save_features(shard_data, feature_dir / f"{shard_stem}_features")


def _make_pct_overlay(
    feature_dir: Path, pct_table: t.Tensor, shard_stem: str = "shard_00000",
) -> tuple[Path, str]:
    """Create percentile overlay for a base shard. Returns (overlay_dir, checksum)."""
    checksum = compute_percentile_checksum(pct_table)
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    write_manifest(pct_dir, checksum, OverlayKind.PERCENTILE)

    data = load_features(feature_dir / f"{shard_stem}_features{COMPRESSED_EXT}")
    fids = data["feature_ids"]
    fmags = data["feature_mags"]
    pcts = magnitude_to_percentile(fids.to(t.int64), fmags, pct_table)
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: percentile_to_uint8(pcts)},
        overlay_shard_path(pct_dir, shard_stem),
    )
    return pct_dir, checksum


def _make_tok_overlay(
    feature_dir: Path, tok_checksum: str, shard_stem: str = "shard_00000",
) -> Path:
    """Create token boundary overlay. Returns overlay_dir."""
    data = load_features(feature_dir / f"{shard_stem}_features{COMPRESSED_EXT}")
    n_segs = int(data["n_segments"])
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tok_checksum)
    write_manifest(tok_dir, tok_checksum, OverlayKind.SEG_BOUNDARY)
    save_overlay_shard(
        {
            TensorKey.SEGMENT_TOKEN_STARTS: t.arange(n_segs, dtype=t.int32),
            TensorKey.SEGMENT_TOKEN_ENDS: t.arange(1, n_segs + 1, dtype=t.int32),
        },
        overlay_shard_path(tok_dir, shard_stem),
    )
    return tok_dir


# ---------------------------------------------------------------------------
# FeatureIndex overlay integration
# ---------------------------------------------------------------------------


class TestFeatureIndexOverlayLoading:
    """FeatureIndex discovers and loads from overlay directories."""

    def test_loads_percentile_overlay(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _, checksum = _make_pct_overlay(feature_dir, pct_table)

        idx = FeatureIndex(str(feature_dir), percentile_checksum=checksum)
        assert idx.has_precomputed_percentiles is True

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        for seg in segs:
            assert seg.feature_percentiles is not None
            assert seg.feature_percentiles.dtype == t.uint8

    def test_loads_token_boundary_overlay(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        tok_checksum = "tok_test_hash" + "0" * 51
        _make_tok_overlay(feature_dir, tok_checksum)

        idx = FeatureIndex(str(feature_dir), tokenizer_checksum=tok_checksum)
        assert idx.has_precomputed_token_boundaries is True

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        for seg in segs:
            assert seg.token_start is not None
            assert seg.token_end is not None

    def test_overlay_discovery_uses_feature_dir_base_hash_cache(
        self, tmp_path, monkeypatch
    ):
        """Both expensive base-hash validations share the existing cache.

        This is an argument-level regression: the overlay hash-cache itself has
        exhaustive cold/hit/stale coverage in ``test_base_shard_hash_cache.py``.
        Here we pin the previously missing FeatureIndex wiring for both overlay
        kinds, including the exact cache location.
        """
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _, pct_checksum = _make_pct_overlay(feature_dir, pct_table)
        tok_checksum = "tok_test_hash" + "0" * 51
        _make_tok_overlay(feature_dir, tok_checksum)

        real_discover = feature_index_module.discover_overlay
        calls: list[tuple[OverlayKind, Path | None]] = []

        def recording_discover(
            feature_dir_arg,
            kind,
            checksum,
            *,
            base_shard_paths=None,
            cache_dir=None,
        ):
            calls.append((kind, cache_dir))
            return real_discover(
                feature_dir_arg,
                kind,
                checksum,
                base_shard_paths=base_shard_paths,
                cache_dir=cache_dir,
            )

        monkeypatch.setattr(
            feature_index_module, "discover_overlay", recording_discover
        )
        idx = FeatureIndex(
            str(feature_dir),
            percentile_checksum=pct_checksum,
            tokenizer_checksum=tok_checksum,
        )

        assert idx.has_precomputed_percentiles is True
        assert idx.has_precomputed_token_boundaries is True
        assert calls == [
            (OverlayKind.PERCENTILE, feature_dir),
            (OverlayKind.SEG_BOUNDARY, feature_dir),
        ]

    def test_missing_requested_percentile_overlay_fails_loud(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)

        before_failure = _inventory(feature_dir)
        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(
                str(feature_dir),
                percentile_checksum="nonexistent" + "0" * 55,
            )
        assert _inventory(feature_dir) == before_failure

    def test_wrong_checksum_not_discovered(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _, real_checksum = _make_pct_overlay(feature_dir, pct_table)

        # Pass a different checksum
        wrong = "wrong_checksum" + "0" * 50
        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(str(feature_dir), percentile_checksum=wrong)

    def test_missing_requested_token_overlay_fails_loud(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)

        with pytest.raises(OverlayIncompleteError, match="token"):
            FeatureIndex(
                str(feature_dir),
                tokenizer_checksum="nonexistent" + "0" * 55,
            )

    def test_requested_token_overlay_requires_every_base_shard_by_default(
        self, tmp_path
    ):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir, "shard_00000")
        _make_base_shard(feature_dir, "shard_00001")
        checksum = "tok_partial" + "0" * 53
        _make_tok_overlay(feature_dir, checksum, "shard_00001")

        with pytest.raises(OverlayIncompleteError, match="missing 1"):
            FeatureIndex(str(feature_dir), tokenizer_checksum=checksum)

    def test_token_overlay_can_require_one_explicit_validation_stem(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir, "shard_00000")
        _make_base_shard(feature_dir, "shard_00001")
        checksum = "tok_partial" + "0" * 53
        _make_tok_overlay(feature_dir, checksum, "shard_00001")

        idx = FeatureIndex(
            str(feature_dir),
            tokenizer_checksum=checksum,
            required_token_boundary_stems={"shard_00001"},
        )
        segments = idx.lookup("shard_00001", 0)
        assert segments
        assert all(segment.token_start is not None for segment in segments)
        assert all(segment.token_end is not None for segment in segments)
        with pytest.raises(OverlayIncompleteError, match="out-of-scope shard"):
            idx.lookup("shard_00000", 0)

    def test_required_token_overlay_stem_must_belong_to_feature_family(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir, "shard_00000")
        checksum = "tok_partial" + "0" * 53
        _make_tok_overlay(feature_dir, checksum, "shard_00000")

        with pytest.raises(OverlayIncompleteError, match="absent from the feature family"):
            FeatureIndex(
                str(feature_dir),
                tokenizer_checksum=checksum,
                required_token_boundary_stems={"shard_99999"},
            )

    def test_requested_percentile_overlay_requires_every_base_shard(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir, "shard_00000")
        _make_base_shard(feature_dir, "shard_00001")
        pct_table = _make_pct_table()
        _, checksum = _make_pct_overlay(feature_dir, pct_table, "shard_00000")

        before_failure = _inventory(feature_dir)
        with pytest.raises(OverlayIncompleteError, match="missing 1"):
            FeatureIndex(str(feature_dir), percentile_checksum=checksum)
        assert _inventory(feature_dir) == before_failure

    def test_corrupt_requested_percentile_payload_fails_at_load(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_dir, checksum = _make_pct_overlay(feature_dir, _make_pct_table())
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: t.zeros(1, dtype=t.uint8)},
            overlay_shard_path(pct_dir, "shard_00000"),
        )

        idx = FeatureIndex(str(feature_dir), percentile_checksum=checksum)
        before_failure = _inventory(feature_dir)
        with pytest.raises(CorruptFeatureShardError):
            idx.lookup("shard_00000", 0)
        assert _inventory(feature_dir) == before_failure

    def test_corrupt_requested_token_payload_fails_at_load(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        checksum = "tok_corrupt" + "0" * 53
        tok_dir = _make_tok_overlay(feature_dir, checksum)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(12, dtype=t.int64),
                TensorKey.SEGMENT_TOKEN_ENDS: t.ones(12, dtype=t.int32),
            },
            overlay_shard_path(tok_dir, "shard_00000"),
        )

        idx = FeatureIndex(str(feature_dir), tokenizer_checksum=checksum)
        with pytest.raises(CorruptFeatureShardError):
            idx.lookup("shard_00000", 0)

    def test_both_overlays_simultaneously(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _, pct_checksum = _make_pct_overlay(feature_dir, pct_table)
        tok_checksum = "tok_both_test" + "0" * 52
        _make_tok_overlay(feature_dir, tok_checksum)

        idx = FeatureIndex(
            str(feature_dir),
            percentile_checksum=pct_checksum,
            tokenizer_checksum=tok_checksum,
        )
        assert idx.has_precomputed_percentiles is True
        assert idx.has_precomputed_token_boundaries is True

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        for seg in segs:
            assert seg.feature_percentiles is not None
            assert seg.token_start is not None


# ---------------------------------------------------------------------------
# Mirror symlinks
# ---------------------------------------------------------------------------


class TestMirrorSymlinkIntegration:
    """Mirror symlinks cross-reference overlays correctly."""

    def test_symlinks_are_files_not_dirs(self, tmp_path):
        """Ensure no directory symlinks are created."""
        import os
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        pct_ov_dir, pct_checksum = _make_pct_overlay(feature_dir, pct_table)

        m = tmp_path / "mirror"
        create_mirror_symlinks(pct_ov_dir, m)

        for entry in m.iterdir():
            if entry.is_symlink():
                assert not os.path.isdir(entry), f"Directory symlink found: {entry}"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full cascade
# ---------------------------------------------------------------------------


class TestFullCascade:
    """Test the complete resolution cascade from base to overlays."""

    def test_no_overlays_no_precomputed(self, tmp_path):
        """Base shard only → no precomputed data."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)

        idx = FeatureIndex(str(feature_dir))
        assert idx.has_precomputed_percentiles is False
        assert idx.has_precomputed_token_boundaries is False

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert all(s.feature_percentiles is None for s in segs)
        assert all(s.token_start is None for s in segs)

    def test_pct_overlay_only(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _, pct_checksum = _make_pct_overlay(feature_dir, pct_table)

        idx = FeatureIndex(str(feature_dir), percentile_checksum=pct_checksum)
        assert idx.has_precomputed_percentiles is True
        assert idx.has_precomputed_token_boundaries is False

    def test_tok_overlay_only(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        tok_checksum = "tok_cascade" + "0" * 54
        _make_tok_overlay(feature_dir, tok_checksum)

        idx = FeatureIndex(str(feature_dir), tokenizer_checksum=tok_checksum)
        assert idx.has_precomputed_percentiles is False
        assert idx.has_precomputed_token_boundaries is True

    def test_stale_overlay_not_discovered(self, tmp_path):
        """Changing statistics → new checksum → old overlay not found."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)

        pct_table = _make_pct_table()
        _, old_checksum = _make_pct_overlay(feature_dir, pct_table)

        # New percentile table → different checksum
        new_pct_table = pct_table * 2.0
        new_checksum = compute_percentile_checksum(new_pct_table)
        assert old_checksum != new_checksum

        # A non-null requested checksum is required, never a silent fallback.
        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(str(feature_dir), percentile_checksum=new_checksum)


# ---------------------------------------------------------------------------
# Prefix matching integration
# ---------------------------------------------------------------------------


class TestPrefixMatchIntegration:
    """Integration tests for prefix-based overlay discovery through FeatureIndex."""

    def test_feature_index_with_prefix_only_checksum(self, tmp_path):
        """FeatureIndex should discover overlays when given only the 16-char prefix."""
        from steering.overlay_io import CHECKSUM_PREFIX_LEN

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _, full_checksum = _make_pct_overlay(feature_dir, pct_table)

        # Pass only the prefix
        prefix = full_checksum[:CHECKSUM_PREFIX_LEN]
        idx = FeatureIndex(str(feature_dir), percentile_checksum=prefix)
        assert idx.has_precomputed_percentiles is True

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        for seg in segs:
            assert seg.feature_percentiles is not None

    def test_feature_index_with_empty_checksum_still_requires_overlay(self, tmp_path):
        """Only None disables an overlay; an empty non-null checksum is required."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        _make_pct_overlay(feature_dir, pct_table)

        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(str(feature_dir), percentile_checksum="")

    def test_feature_index_prefix_tok_overlay(self, tmp_path):
        """Token boundary overlay discovered via prefix-only checksum."""
        from steering.overlay_io import CHECKSUM_PREFIX_LEN

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        tok_checksum = "dd" * 32
        _make_tok_overlay(feature_dir, tok_checksum)

        prefix = tok_checksum[:CHECKSUM_PREFIX_LEN]
        idx = FeatureIndex(str(feature_dir), tokenizer_checksum=prefix)
        assert idx.has_precomputed_token_boundaries is True

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        for seg in segs:
            assert seg.token_start is not None
            assert seg.token_end is not None

    def test_truncated_manifest_checksum_not_matched(self, tmp_path):
        """If the manifest has a truncated checksum (corruption), FeatureIndex
        should not discover the overlay even with the correct full hash."""
        import json

        from steering.overlay_io import (
            CHECKSUM_PREFIX_LEN,
            MANIFEST_FILENAME,
            OverlayKind,
            overlay_dir,
        )

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _make_base_shard(feature_dir)
        pct_table = _make_pct_table()
        full_checksum = compute_percentile_checksum(pct_table)

        # Create the overlay directory and shard
        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, full_checksum)
        pct_dir.mkdir(parents=True)

        # Write a CORRUPT manifest with truncated checksum
        manifest_path = pct_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps({
                "checksum": full_checksum[:CHECKSUM_PREFIX_LEN],
                "kind": "pct",
            }) + "\n"
        )

        # Create a dummy overlay shard
        from steering.overlay_io import overlay_shard_path, save_overlay_shard
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: t.zeros(10, dtype=t.uint8)},
            overlay_shard_path(pct_dir, "shard_00000"),
        )

        # The requested full checksum must fail loud rather than fall back.
        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(str(feature_dir), percentile_checksum=full_checksum)

    def test_pipeline_checksum_roundtrip_e2e(self, tmp_path):
        """Full pipeline roundtrip: compute checksum → create overlay → discover via FeatureIndex.

        This mirrors the real flow:
        1. precompute_percentiles computes checksum, writes overlay + manifest
        2. precompute_packed_rows loads checksum from stats, creates FeatureIndex
        3. FeatureIndex discovers overlay and loads precomputed data
        """
        from steering.overlay_io import (
            CHECKSUM_PREFIX_LEN,
            OverlayKind,
            overlay_dir,
            overlay_shard_path,
            save_overlay_shard,
            write_manifest,
        )

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shard_path = _make_base_shard(feature_dir)

        # Step 1: Compute percentile table and checksum
        pct_table = _make_pct_table()
        checksum = compute_percentile_checksum(pct_table)
        assert len(checksum) == 64, "SHA-256 hex should be 64 chars"

        # Step 2: Create overlay (what precompute_percentiles does)
        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        write_manifest(pct_dir, checksum, OverlayKind.PERCENTILE)

        data = load_features(shard_path)
        fids = data["feature_ids"]
        fmags = data["feature_mags"]
        pcts = magnitude_to_percentile(fids.to(t.int64), fmags, pct_table)
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: percentile_to_uint8(pcts)},
            overlay_shard_path(pct_dir, "shard_00000"),
        )

        # Step 3: Discover via FeatureIndex with full hash
        idx = FeatureIndex(str(feature_dir), percentile_checksum=checksum)
        assert idx.has_precomputed_percentiles is True

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert all(s.feature_percentiles is not None for s in segs)

        # Step 4: Also verify discovery via prefix
        idx2 = FeatureIndex(
            str(feature_dir),
            percentile_checksum=checksum[:CHECKSUM_PREFIX_LEN],
        )
        assert idx2.has_precomputed_percentiles is True

        segs2 = idx2.lookup("shard_00000", 0)
        assert segs2 is not None
        assert all(s.feature_percentiles is not None for s in segs2)

        # Step 5: Verify data matches between full hash and prefix discovery
        for s1, s2 in zip(segs, segs2, strict=True):
            p1 = s1.feature_percentiles
            p2 = s2.feature_percentiles
            assert p1 is not None
            assert p2 is not None
            assert t.equal(p1, p2)
