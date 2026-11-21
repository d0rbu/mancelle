"""``backfill_packed_row_shapes`` golden test.

Pin functional behavior of ``scripts/preprocess/backfill_packed_row_shapes.py``
(the script scans existing packed-row shards and writes ``max_n_seg`` /
``max_seg_size`` into the manifest's metadata).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch as t

import nanochat.atomic_io as atomic_io
import scripts.preprocess.backfill_packed_row_shapes as backfill_module
from errors import PackedRowInvariantError
from scripts.preprocess.backfill_packed_row_shapes import backfill_packed_row_shapes
from steering.overlay_io import MANIFEST_FILENAME
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard


def _make_packed_shard(path: Path, n_rows: int, T: int, n_seg_per_row: int):
    seg_size = T // n_seg_per_row
    total_segs = n_rows * n_seg_per_row
    shard = PackedRowShard(
        tokens=t.zeros(n_rows, T + 1, dtype=t.uint16),
        row_seg_offsets=t.tensor(
            [i * n_seg_per_row for i in range(n_rows + 1)], dtype=t.int32,
        ),
        seg_token_start=t.tensor(
            [i * seg_size for r in range(n_rows) for i in range(n_seg_per_row)],
            dtype=t.uint16,
        ),
        seg_token_end=t.tensor(
            [(i + 1) * seg_size for r in range(n_rows) for i in range(n_seg_per_row)],
            dtype=t.uint16,
        ),
        seg_doc_id=t.zeros(total_segs, dtype=t.uint8),
        seg_feat_offsets=t.tensor([0] * (total_segs + 1), dtype=t.int32),
        feature_ids=t.tensor([], dtype=t.uint16),
        feature_pcts=t.tensor([], dtype=t.uint8),
    )
    return save_packed_row_shard(shard, path)


IDENTITY_METADATA = {
    "type": "packed_rows",
    "context_len": 2048,
    "buffer_size": 1000,
    "rows_per_shard": 16384,
    "pct_checksum": "pct-checksum",
    "tok_checksum": "tok-checksum",
    "preserved": "value",
}


def _valid_manifest(kind: str = "packed_rows") -> dict:
    return {
        "checksum": "pct-checksum:tok-checksum",
        "kind": kind,
        "created_at": "2026-07-21T00:00:00Z",
        "checksum_prefix": "pct-checksum:tok",
        "metadata": dict(IDENTITY_METADATA),
        "base_shard_content_hash": "ab" * 32,
        "schema_version": 2,
    }


def _digest_map(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class TestBackfillNoExistingManifest:
    def test_missing_manifest_fails_before_decompression(self, tmp_path, monkeypatch):
        packed_dir = tmp_path / "packed"
        packed_dir.mkdir()
        _make_packed_shard(packed_dir / "shard_00000", n_rows=2, T=8, n_seg_per_row=2)
        _make_packed_shard(packed_dir / "shard_00001", n_rows=3, T=12, n_seg_per_row=3)
        before = _digest_map(packed_dir)
        monkeypatch.setattr(
            backfill_module,
            "_reduce_max_dims_over_shards",
            lambda _paths: pytest.fail("invalid manifest must fail before decompression"),
        )

        with pytest.raises(PackedRowInvariantError, match="manifest"):
            backfill_packed_row_shapes(packed_dir)
        assert _digest_map(packed_dir) == before
        assert not (packed_dir / MANIFEST_FILENAME).exists()


class TestBackfillExistingManifest:
    """If a manifest already exists, the existing metadata is preserved and
    only the shape fields are merged in.
    """

    @pytest.mark.parametrize("kind", ["packed_rows", "pct"])
    def test_preserves_complete_manifest_bytes_except_shapes(self, tmp_path, kind):
        packed_dir = tmp_path / "packed"
        packed_dir.mkdir()
        _make_packed_shard(
            packed_dir / "shard_00000", n_rows=2, T=8, n_seg_per_row=2
        )

        manifest_path = packed_dir / MANIFEST_FILENAME
        existing = _valid_manifest(kind)
        manifest_path.write_text(json.dumps(existing, indent=2) + "\n")
        expected = json.loads(json.dumps(existing))
        expected["metadata"].update({"max_n_seg": 2, "max_seg_size": 4})
        expected_bytes = (json.dumps(expected, indent=2) + "\n").encode()

        assert backfill_packed_row_shapes(packed_dir) == {
            "max_n_seg": 2,
            "max_seg_size": 4,
        }
        assert manifest_path.read_bytes() == expected_bytes
        assert json.loads(manifest_path.read_text()) == expected

    def test_fabricated_stub_is_rejected_without_mutation(self, tmp_path):
        packed_dir = tmp_path / "packed"
        packed_dir.mkdir()
        _make_packed_shard(
            packed_dir / "shard_00000", n_rows=2, T=8, n_seg_per_row=2
        )
        stub = {
            "checksum": "",
            "kind": "percentile",
            "metadata": {"max_n_seg": 2, "max_seg_size": 4},
        }
        manifest_path = packed_dir / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(stub, indent=2) + "\n")
        before = _digest_map(packed_dir)

        with pytest.raises(PackedRowInvariantError):
            backfill_packed_row_shapes(packed_dir)
        assert _digest_map(packed_dir) == before

    @pytest.mark.parametrize(
        "invalid",
        ["empty_checksum", "wrong_kind", "missing_identity", "invalid_metadata"],
    )
    def test_invalid_existing_manifest_is_rejected(self, tmp_path, invalid):
        packed_dir = tmp_path / "packed"
        packed_dir.mkdir()
        _make_packed_shard(
            packed_dir / "shard_00000", n_rows=2, T=8, n_seg_per_row=2
        )
        manifest = _valid_manifest()
        if invalid == "empty_checksum":
            manifest["checksum"] = ""
        elif invalid == "wrong_kind":
            manifest["kind"] = "percentile"
        elif invalid == "missing_identity":
            del manifest["metadata"]["rows_per_shard"]
        else:
            manifest["metadata"] = []
        manifest_path = packed_dir / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        before = _digest_map(packed_dir)

        with pytest.raises(PackedRowInvariantError):
            backfill_packed_row_shapes(packed_dir)
        assert _digest_map(packed_dir) == before

    def test_atomic_replace_failure_preserves_original_manifest(
        self, tmp_path, monkeypatch
    ):
        packed_dir = tmp_path / "packed"
        packed_dir.mkdir()
        _make_packed_shard(
            packed_dir / "shard_00000", n_rows=2, T=8, n_seg_per_row=2
        )
        manifest_path = packed_dir / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(_valid_manifest(), indent=2) + "\n")
        before = _digest_map(packed_dir)
        real_replace = atomic_io.os.replace

        def fail_manifest_replace(source, destination):
            if Path(destination) == manifest_path:
                raise OSError("injected atomic replace failure")
            return real_replace(source, destination)

        monkeypatch.setattr(atomic_io.os, "replace", fail_manifest_replace)
        with pytest.raises(OSError, match="injected atomic replace failure"):
            backfill_packed_row_shapes(packed_dir)
        assert _digest_map(packed_dir) == before


class TestBackfillEmptyDirRaises:
    def test_no_shards_raises(self, tmp_path):
        packed_dir = tmp_path / "empty_packed"
        packed_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No packed row shards"):
            backfill_packed_row_shapes(packed_dir)
