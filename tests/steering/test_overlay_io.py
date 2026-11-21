"""Unit tests for overlay I/O, discovery, manifest, and symlink management."""

from __future__ import annotations

import json
import os

import pytest
import torch as t
import zstandard

from steering.feature_io import CorruptFeatureShardError, TensorKey, save_features
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    COMPRESSED_EXT,
    MANIFEST_FILENAME,
    OverlayKind,
    create_mirror_symlinks,
    discover_overlay,
    expected_overlay_shard_stems,
    glob_overlay_shards,
    load_overlay_shard,
    load_overlay_shard_bytes,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    validate_percentile_overlay_shard,
    validate_token_boundary_overlay_shard,
    write_manifest,
)
from tests.steering._helpers import make_valid_shard_data

FAKE_CHECKSUM = "a1b2c3d4e5f6a7b8" + "9" * 48  # 64-char hex


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


class TestOverlayDir:
    def test_percentile_path(self, tmp_path):
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert d == tmp_path / "pct" / FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN]

    def test_seg_boundary_path(self, tmp_path):
        d = overlay_dir(tmp_path, OverlayKind.SEG_BOUNDARY, FAKE_CHECKSUM)
        assert d == tmp_path / "tok" / FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN] / "seg_boundaries"

    def test_different_checksums_different_dirs(self, tmp_path):
        d1 = overlay_dir(tmp_path, OverlayKind.PERCENTILE, "aa" * 32)
        d2 = overlay_dir(tmp_path, OverlayKind.PERCENTILE, "bb" * 32)
        assert d1 != d2


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------


class TestOverlayShardIO:
    def test_round_trip(self, tmp_path):
        """Save and load preserves tensors exactly."""
        tensors = {
            "feature_percentiles": t.randint(0, 255, (100,), dtype=t.uint8),
            "counts": t.tensor([1, 2, 3], dtype=t.int32),
        }
        path = save_overlay_shard(tensors, tmp_path / "shard_00000")
        assert path.name.endswith(COMPRESSED_EXT)
        assert path.exists()

        loaded = load_overlay_shard(path)
        assert set(loaded.keys()) == set(tensors.keys())
        for key, tensor in tensors.items():
            assert t.equal(loaded[key], tensor), f"{key} mismatch"

    def test_forces_extension(self, tmp_path):
        """Path gets .safetensors.zst extension even if not provided."""
        path = save_overlay_shard({"x": t.zeros(1)}, tmp_path / "test")
        assert path.name == f"test{COMPRESSED_EXT}"

    def test_load_follows_symlinks(self, tmp_path):
        """Loading via symlink reads the correct data."""
        tensors = {"data": t.tensor([42.0])}
        actual = save_overlay_shard(tensors, tmp_path / "actual" / "shard")
        link_dir = tmp_path / "link"
        link_dir.mkdir()
        link = link_dir / actual.name
        link.symlink_to(os.path.relpath(actual, link_dir))

        loaded = load_overlay_shard(link)
        assert t.equal(loaded["data"], tensors["data"])

    def test_metadata_preserved(self, tmp_path):
        """Optional metadata survives round-trip (read via safetensors header)."""
        tensors = {"x": t.zeros(1)}
        save_overlay_shard(tensors, tmp_path / "shard", metadata={"version": "1"})
        # We don't expose metadata in load_overlay_shard, but the file should be valid
        loaded = load_overlay_shard(tmp_path / f"shard{COMPRESSED_EXT}")
        assert "x" in loaded

    def test_path_and_bytes_decoders_are_tensor_layout_exact(self, tmp_path):
        """The remote byte seam must be the exact local decoder, not a reimplementation."""
        tensors = {
            "bools": t.tensor([[True, False], [False, True]], dtype=t.bool),
            "empty": t.empty((2, 0, 3), dtype=t.float32),
            "f64": t.tensor([1.25, -0.0, 2048.5], dtype=t.float64),
            "i32": t.arange(24, dtype=t.int32).reshape(2, 3, 4),
            "scalar": t.tensor(17, dtype=t.int64),
            "u8": t.arange(255, dtype=t.uint8),
        }
        path = save_overlay_shard(tensors, tmp_path / "layout")
        from_path = load_overlay_shard(path)
        from_bytes = load_overlay_shard_bytes(path.read_bytes())

        assert type(from_path) is type(from_bytes) is dict
        assert set(from_path) == set(from_bytes)
        for key in sorted(from_path):
            path_tensor = from_path[key]
            bytes_tensor = from_bytes[key]
            assert type(path_tensor) is type(bytes_tensor) is t.Tensor
            assert path_tensor.dtype == bytes_tensor.dtype
            assert path_tensor.shape == bytes_tensor.shape
            assert path_tensor.stride() == bytes_tensor.stride()
            assert path_tensor.storage_offset() == bytes_tensor.storage_offset()
            assert path_tensor.element_size() == bytes_tensor.element_size()
            assert t.equal(path_tensor, bytes_tensor)
            assert t.equal(
                path_tensor.contiguous().reshape(-1).view(t.uint8),
                bytes_tensor.contiguous().reshape(-1).view(t.uint8),
            )

    def test_symlink_path_and_target_bytes_decode_exactly(self, tmp_path):
        tensors = {
            "data": t.arange(30, dtype=t.int16).reshape(3, 10),
        }
        target = save_overlay_shard(tensors, tmp_path / "target" / "body")
        link = tmp_path / target.name
        link.symlink_to(target)

        from_link = load_overlay_shard(link)
        from_target_bytes = load_overlay_shard_bytes(target.read_bytes())
        assert list(from_link) == list(from_target_bytes)
        assert t.equal(from_link["data"], from_target_bytes["data"])
        assert from_link["data"].stride() == from_target_bytes["data"].stride()

    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            b"not-a-zstd-frame",
            # Valid zstd is generated in the test body for the safetensors case.
            None,
        ],
    )
    def test_path_and_bytes_decoders_raise_identical_corruption_error(
        self,
        tmp_path,
        payload,
    ):
        if payload is None:
            payload = zstandard.ZstdCompressor().compress(b"not-safetensors")
        path = tmp_path / f"corrupt{COMPRESSED_EXT}"
        path.write_bytes(payload)

        with pytest.raises(Exception) as path_error:
            load_overlay_shard(path)
        with pytest.raises(Exception) as bytes_error:
            load_overlay_shard_bytes(payload)
        assert type(path_error.value) is type(bytes_error.value)
        assert str(path_error.value) == str(bytes_error.value)

    def test_wrong_extension_still_fails_before_path_read(self, tmp_path):
        path = tmp_path / "wrong.bin"
        path.write_bytes(b"not-zstd")
        with pytest.raises(ValueError, match=r"Expected \.safetensors\.zst"):
            load_overlay_shard(path)


class TestOverlayShardValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"wrong": t.zeros(3, dtype=t.uint8)},
            {
                TensorKey.FEATURE_PERCENTILES: t.zeros(3, dtype=t.uint8),
                "extra": t.zeros(1),
            },
            {TensorKey.FEATURE_PERCENTILES: t.zeros(3, dtype=t.int32)},
            {TensorKey.FEATURE_PERCENTILES: t.zeros((3, 1), dtype=t.uint8)},
            {TensorKey.FEATURE_PERCENTILES: t.zeros(2, dtype=t.uint8)},
        ],
    )
    def test_percentile_rejects_wrong_structure(self, tmp_path, payload):
        path = save_overlay_shard(payload, tmp_path / "pct")
        with pytest.raises(CorruptFeatureShardError):
            validate_percentile_overlay_shard(path, expected_numel=3)

    def test_percentile_rejects_unreadable_payload(self, tmp_path):
        path = tmp_path / f"pct{COMPRESSED_EXT}"
        path.write_bytes(b"not zstd")
        with pytest.raises(CorruptFeatureShardError, match="unreadable"):
            validate_percentile_overlay_shard(path, expected_numel=3)

    @pytest.mark.parametrize(
        "payload",
        [
            {TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(3, dtype=t.int32)},
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(3, dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.ones(3, dtype=t.int32),
                "extra": t.zeros(1),
            },
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(3, dtype=t.int64),
                TensorKey.SEGMENT_TOKEN_ENDS: t.ones(3, dtype=t.int32),
            },
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.zeros((3, 1), dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.ones(3, dtype=t.int32),
            },
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(2, dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.ones(3, dtype=t.int32),
            },
        ],
    )
    def test_token_boundaries_reject_wrong_structure(self, tmp_path, payload):
        path = save_overlay_shard(payload, tmp_path / "tok")
        with pytest.raises(CorruptFeatureShardError):
            validate_token_boundary_overlay_shard(path, expected_n_segments=3)

    def test_token_structural_validator_has_no_value_policy(self, tmp_path):
        starts = t.tensor([-1, 8, 2], dtype=t.int32)
        ends = t.tensor([-1, 7, 2], dtype=t.int32)
        path = save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: starts,
                TensorKey.SEGMENT_TOKEN_ENDS: ends,
            },
            tmp_path / "tok",
        )
        actual_starts, actual_ends = validate_token_boundary_overlay_shard(
            path, expected_n_segments=3
        )
        assert t.equal(actual_starts, starts)
        assert t.equal(actual_ends, ends)


class TestGlobOverlayShards:
    def test_finds_shards(self, tmp_path):
        save_overlay_shard({"x": t.zeros(1)}, tmp_path / "shard_00000")
        save_overlay_shard({"x": t.zeros(1)}, tmp_path / "shard_00001")
        shards = glob_overlay_shards(tmp_path)
        assert len(shards) == 2
        assert shards[0].name < shards[1].name

    def test_empty_dir(self, tmp_path):
        assert glob_overlay_shards(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert glob_overlay_shards(tmp_path / "nope") == []


class TestOverlayShardPath:
    def test_construction(self, tmp_path):
        p = overlay_shard_path(tmp_path, "shard_00042")
        assert p == tmp_path / f"shard_00042{COMPRESSED_EXT}"


def test_expected_coverage_includes_empty_percentile_but_not_empty_token(tmp_path):
    nonempty = save_features(
        make_valid_shard_data(n_docs=1, segs_per_doc=1, feats_per_seg=2),
        tmp_path / "shard_00000_features",
    )
    empty = save_features(
        make_valid_shard_data(n_docs=0, segs_per_doc=0),
        tmp_path / "shard_00001_features",
    )
    bases = [nonempty, empty]
    assert expected_overlay_shard_stems(bases, OverlayKind.PERCENTILE) == [
        "shard_00000",
        "shard_00001",
    ]
    assert expected_overlay_shard_stems(bases, OverlayKind.SEG_BOUNDARY) == [
        "shard_00000"
    ]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_round_trip(self, tmp_path):
        write_manifest(tmp_path, FAKE_CHECKSUM, OverlayKind.PERCENTILE)
        m = read_manifest(tmp_path)
        assert m is not None
        assert m["checksum"] == FAKE_CHECKSUM
        assert m["kind"] == "pct"
        assert m["checksum_prefix"] == FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN]
        assert "created_at" in m

    def test_with_metadata(self, tmp_path):
        write_manifest(tmp_path, FAKE_CHECKSUM, OverlayKind.SEG_BOUNDARY,
                        metadata={"source": "test"})
        m = read_manifest(tmp_path)
        assert m is not None
        assert m["metadata"]["source"] == "test"

    def test_missing_returns_none(self, tmp_path):
        assert read_manifest(tmp_path) is None

    def test_invalid_json_returns_none(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("not json")
        assert read_manifest(tmp_path) is None

    def test_missing_checksum_field_returns_none(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps({"kind": "pct"}))
        assert read_manifest(tmp_path) is None

    def test_creates_directory(self, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)
        assert (d / MANIFEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverOverlay:
    def test_finds_matching_overlay(self, tmp_path):
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)

        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert result == d

    def test_returns_none_for_missing(self, tmp_path):
        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert result is None

    def test_returns_none_for_wrong_checksum(self, tmp_path):
        """Dir exists with prefix match but manifest has different full checksum."""
        wrong_checksum = FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN] + "x" * 48
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        write_manifest(d, wrong_checksum, OverlayKind.PERCENTILE)

        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert result is None

    def test_returns_none_for_no_manifest(self, tmp_path):
        """Dir exists but no manifest.json."""
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert result is None

    def test_seg_boundary_discovery(self, tmp_path):
        tok_hash = "bb" * 32
        d = overlay_dir(tmp_path, OverlayKind.SEG_BOUNDARY, tok_hash)
        d.mkdir(parents=True)
        write_manifest(d, tok_hash, OverlayKind.SEG_BOUNDARY)

        result = discover_overlay(tmp_path, OverlayKind.SEG_BOUNDARY, tok_hash)
        assert result == d

    # --- Prefix matching edge cases ---

    def test_prefix_only_matches_full_manifest_hash(self, tmp_path):
        """A 16-char prefix should match a manifest with the full 64-char hash."""
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)

        result = discover_overlay(
            tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN],
        )
        assert result == d

    def test_empty_checksum_returns_none(self, tmp_path):
        """Empty checksum must never match any overlay."""
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)

        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, "")
        assert result is None

    def test_empty_manifest_checksum_returns_none(self, tmp_path):
        """A manifest with an empty checksum field must not match anything."""
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        # Write manifest.json with empty checksum (simulates corruption)
        manifest_path = d / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps({"checksum": "", "kind": "pct"}) + "\n")

        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert result is None

    def test_manifest_with_truncated_checksum_does_not_match(self, tmp_path):
        """A manifest with only a prefix (not full hash) must not match a full hash.

        This guards against the unsafe reverse-direction prefix match where a
        truncated manifest checksum would accept any full hash starting with it.
        """
        truncated = FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN]
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        # Write manifest with only the prefix (simulates a corruption or bug)
        manifest_path = d / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps({"checksum": truncated, "kind": "pct"}) + "\n"
        )

        # The full 64-char checksum starts with the manifest's 16-char prefix,
        # but the reverse direction match should be rejected.
        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        assert result is None, (
            "A manifest with only a truncated prefix should not match a full hash — "
            "the reverse direction of prefix matching is unsafe"
        )

    def test_prefix_match_is_one_directional(self, tmp_path):
        """Only manifest.startswith(caller) is accepted, not the reverse."""
        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM)
        d.mkdir(parents=True)
        write_manifest(d, FAKE_CHECKSUM, OverlayKind.PERCENTILE)

        # Caller passes prefix → manifest.startswith(prefix) → match
        assert discover_overlay(
            tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN],
        ) == d

        # Caller passes full hash → manifest.startswith(full) → match (strings equal)
        assert discover_overlay(
            tmp_path, OverlayKind.PERCENTILE, FAKE_CHECKSUM,
        ) == d

    def test_seg_boundary_prefix_match(self, tmp_path):
        """Prefix matching also works for SEG_BOUNDARY overlays."""
        tok_hash = "cc" * 32
        d = overlay_dir(tmp_path, OverlayKind.SEG_BOUNDARY, tok_hash)
        d.mkdir(parents=True)
        write_manifest(d, tok_hash, OverlayKind.SEG_BOUNDARY)

        result = discover_overlay(
            tmp_path, OverlayKind.SEG_BOUNDARY, tok_hash[:CHECKSUM_PREFIX_LEN],
        )
        assert result == d

    def test_colliding_prefix_different_full_hash_rejected(self, tmp_path):
        """Two full hashes sharing a 16-char prefix: discover rejects the wrong one."""
        hash_a = FAKE_CHECKSUM  # "a1b2c3d4e5f6a7b8" + "9" * 48
        hash_b = FAKE_CHECKSUM[:CHECKSUM_PREFIX_LEN] + "0" * 48  # same prefix, different tail

        d = overlay_dir(tmp_path, OverlayKind.PERCENTILE, hash_a)
        d.mkdir(parents=True)
        write_manifest(d, hash_a, OverlayKind.PERCENTILE)

        # hash_b maps to the same directory (same prefix), but should be rejected
        result = discover_overlay(tmp_path, OverlayKind.PERCENTILE, hash_b)
        assert result is None


# ---------------------------------------------------------------------------
# Symlink management
# ---------------------------------------------------------------------------


class TestCreateMirrorSymlinks:
    def test_creates_file_symlinks(self, tmp_path):
        """Symlinks are files, not directories."""
        source = tmp_path / "source"
        source.mkdir()
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00000")
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00001")

        mirror = tmp_path / "mirror"
        n = create_mirror_symlinks(source, mirror)
        assert n == 2

        for entry in mirror.iterdir():
            assert entry.is_symlink(), f"{entry} should be a symlink"
            assert not os.path.isdir(entry), f"{entry} should not be a directory symlink"
            assert entry.resolve().is_file(), f"{entry} should resolve to a file"

    def test_symlinks_resolve_to_source(self, tmp_path):
        """Symlinked files contain the same data as source."""
        source = tmp_path / "source"
        source.mkdir()
        tensors = {"data": t.tensor([1.0, 2.0, 3.0])}
        save_overlay_shard(tensors, source / "shard_00000")

        mirror = tmp_path / "mirror"
        create_mirror_symlinks(source, mirror)

        link = list(mirror.iterdir())[0]
        loaded = load_overlay_shard(link)
        assert t.equal(loaded["data"], tensors["data"])

    def test_specific_stems(self, tmp_path):
        """Only specified shard stems are linked."""
        source = tmp_path / "source"
        source.mkdir()
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00000")
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00001")
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00002")

        mirror = tmp_path / "mirror"
        n = create_mirror_symlinks(source, mirror, shard_stems=["shard_00000", "shard_00002"])
        assert n == 2
        names = sorted(e.name for e in mirror.iterdir())
        assert len(names) == 2
        assert "shard_00001" not in names[0]

    def test_idempotent(self, tmp_path):
        """Running twice doesn't create duplicates or fail."""
        source = tmp_path / "source"
        source.mkdir()
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00000")

        mirror = tmp_path / "mirror"
        n1 = create_mirror_symlinks(source, mirror)
        n2 = create_mirror_symlinks(source, mirror)
        assert n1 == 1
        assert n2 == 0  # Already exists
        assert len(list(mirror.iterdir())) == 1

    def test_relative_symlinks(self, tmp_path):
        """Symlinks use relative paths for portability."""
        source = tmp_path / "source"
        source.mkdir()
        save_overlay_shard({"x": t.zeros(1)}, source / "shard_00000")

        mirror = tmp_path / "mirror"
        create_mirror_symlinks(source, mirror)

        link = list(mirror.iterdir())[0]
        target = os.readlink(link)
        assert not os.path.isabs(target), f"Symlink should be relative, got: {target}"
