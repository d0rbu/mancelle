"""Packed-row manifests are labeled ``packed_rows`` and use schema v2.

The packed-row precompute callsites in ``precompute_packed_rows.run`` must:

1. Use ``OverlayKind.PACKED_ROW`` (not the historical mis-label
   ``OverlayKind.PERCENTILE``).
2. Pass ``base_shard_paths=glob_features(feature_dir)`` so the manifest is
   ``schema_version=2`` with a ``base_shard_content_hash``.

A full ``run(cfg)`` invocation requires the nanochat tokenizer + parquet
fixtures, which are exercised by ``tests/steering/test_precompute_packed_rows_integration.py``.
This test surface is focused: we verify the contract directly via
``write_manifest`` to pin the regression.
"""

from __future__ import annotations

import json

import torch as t

from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    MANIFEST_FILENAME,
    OverlayKind,
    _compute_base_hash,
    save_overlay_shard,
    write_manifest,
)

FAKE_PCT = "ab" * 32  # 64-char hex
FAKE_TOK = "cd" * 32
COMBINED = f"{FAKE_PCT}:{FAKE_TOK}"  # matches precompute_packed_rows pattern


class TestPackedRowsManifestLabel:
    """``write_manifest`` with ``OverlayKind.PACKED_ROW`` writes
    ``"kind": "packed_rows"`` to the JSON.
    """

    def test_kind_is_packed_rows(self, tmp_path):
        d = tmp_path / "packed"
        d.mkdir()
        write_manifest(
            d, COMBINED, OverlayKind.PACKED_ROW,
            metadata={"context_len": 2048, "max_n_seg": 4, "max_seg_size": 512},
        )
        manifest = json.loads((d / MANIFEST_FILENAME).read_text())
        assert manifest["kind"] == "packed_rows"
        assert manifest["checksum"] == COMBINED


class TestPackedRowsManifestV2WithBaseShards:
    """When base_shard_paths is passed (post-fix), the manifest is v2 and
    carries the base-shard content hash.
    """

    def test_v2_with_base_shard_paths(self, tmp_path):
        # Create a fake feature_dir + base shard.
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        base_shard = save_overlay_shard(
            {"x": t.tensor([1, 2, 3], dtype=t.int32)},
            feature_dir / "shard_00000",
        )

        d = tmp_path / "packed"
        d.mkdir()
        write_manifest(
            d, COMBINED, OverlayKind.PACKED_ROW,
            metadata={"max_n_seg": 4},
            base_shard_paths=[base_shard],
        )
        manifest = json.loads((d / MANIFEST_FILENAME).read_text())
        assert manifest["schema_version"] == 2
        assert "base_shard_content_hash" in manifest
        assert manifest["kind"] == "packed_rows"


class TestPackedRowsManifestO6BaseHashCache:
    """O6 (issue #531): run() computes the base-shard hash ONCE via the #529 stat-validated
    cache (``cache_dir=feature_dir``) and threads the SAME ``precomputed_base_hash`` into both
    the train and val manifests — byte-identical to the streaming hash, and a warm second
    computation reads zero base-shard bytes.
    """

    def _base_shards(self, feature_dir, n=3):
        feature_dir.mkdir(parents=True, exist_ok=True)
        return [
            save_overlay_shard(
                {"x": t.tensor([i, i + 1, i + 2], dtype=t.int32)},
                feature_dir / f"shard_{i:05d}",
            )
            for i in range(n)
        ]

    def test_precomputed_hash_equals_streamed_and_writes_cache(self, tmp_path):
        feature_dir = tmp_path / "features"
        paths = self._base_shards(feature_dir)

        streamed = _compute_base_hash(paths, cache_dir=None).hexdigest  # reference
        result = _compute_base_hash(paths, cache_dir=feature_dir)  # O6 path (cold)
        assert result.hexdigest == streamed
        assert result.cache_hit is False
        assert (feature_dir / BASE_SHARD_HASH_CACHE_FILENAME).is_file()

        d = tmp_path / "packed"
        d.mkdir()
        write_manifest(
            d, COMBINED, OverlayKind.PACKED_ROW, metadata={"max_n_seg": 1},
            base_shard_paths=paths, precomputed_base_hash=result.hexdigest,
        )
        manifest = json.loads((d / MANIFEST_FILENAME).read_text())
        assert manifest["base_shard_content_hash"] == streamed

    def test_warm_cache_hit_reads_zero_bytes(self, tmp_path):
        feature_dir = tmp_path / "features"
        paths = self._base_shards(feature_dir, n=1)
        cold = _compute_base_hash(paths, cache_dir=feature_dir)
        warm = _compute_base_hash(paths, cache_dir=feature_dir)
        assert warm.cache_hit is True
        assert warm.bytes_read == 0
        assert warm.hexdigest == cold.hexdigest
