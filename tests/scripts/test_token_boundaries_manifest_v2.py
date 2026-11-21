"""``precompute_token_boundaries.run`` emits ``schema_version=2``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.preprocess.precompute_token_boundaries import run as run_token_boundaries
from steering.feature_io import TensorKey, glob_features, load_features, save_features
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    MANIFEST_FILENAME,
    OverlayKind,
    OverlayStaleError,
    discover_overlay,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
)
from steering.segmentation import compute_tokenizer_checksum

SEGMENT_SIZE = 16
N_SAE = 64


class _SimpleTokenizer:
    """Deterministic char-level tokenizer (matches existing fixture pattern)."""

    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        parts = []
        for tid in token_ids:
            if tid == 0:
                parts.append("<bos>")
            else:
                parts.append(f"t{tid:03d} ")
        return "".join(parts)


def _make_env(
    tmp_path: Path, *, n_shards: int = 1
) -> tuple[Path, Path, _SimpleTokenizer]:
    feature_dir = tmp_path / "features" / f"s{SEGMENT_SIZE}"
    feature_dir.mkdir(parents=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    tokenizer = _SimpleTokenizer()

    n_docs = 2
    n_segs_per_doc = 2
    n_feats_per_seg = 3
    texts = []
    for doc_idx in range(n_docs):
        words = [f"word{w}" for w in range(20 + doc_idx * 5)]
        texts.append(" ".join(words))

    total_segs = n_docs * n_segs_per_doc
    total_feats = total_segs * n_feats_per_seg

    all_fids = t.zeros(total_feats, dtype=t.int32)
    all_mags = t.zeros(total_feats, dtype=t.float32)
    offsets = t.zeros(total_segs + 1, dtype=t.int64)
    doc_ids = t.zeros(total_segs, dtype=t.int32)
    seg_indices = t.zeros(total_segs, dtype=t.int32)
    seg_positions = t.zeros(total_segs, dtype=t.int32)

    feat_offset = 0
    seg_global = 0
    for doc_idx in range(n_docs):
        for seg_idx in range(n_segs_per_doc):
            doc_ids[seg_global] = doc_idx
            seg_indices[seg_global] = seg_idx
            seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
            offsets[seg_global] = feat_offset
            for f in range(n_feats_per_seg):
                fid = (doc_idx * 10 + seg_idx * 5 + f) % N_SAE
                all_fids[feat_offset + f] = fid
                all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx
            feat_offset += n_feats_per_seg
            seg_global += 1
    offsets[total_segs] = feat_offset

    segment_char_starts = seg_positions * 6
    segment_char_ends = (seg_positions + SEGMENT_SIZE) * 6

    shard_data = {
        "offsets": offsets,
        "feature_ids": all_fids,
        "feature_mags": all_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": t.full((n_docs,), 41, dtype=t.int32),
        "segment_char_starts": segment_char_starts,
        "segment_char_ends": segment_char_ends,
        "n_docs": n_docs,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }

    for shard_index in range(n_shards):
        shard_stem = f"shard_{shard_index:05d}"
        pq.write_table(
            pa.table({"text": texts}), str(data_dir / f"{shard_stem}.parquet")
        )
        save_features(shard_data, feature_dir / f"{shard_stem}_features")
    return feature_dir, data_dir, tokenizer


def _inventory(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _cfg(feature_dir: Path, data_dir: Path, **overrides) -> DictConfig:
    values = {
        "feature_dir": str(feature_dir),
        "segment_size": SEGMENT_SIZE,
        "data_dir": str(data_dir),
        "dry_run": False,
        "resume": False,
        "force": False,
        "workers": 1,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _run(cfg: DictConfig, tokenizer: _SimpleTokenizer) -> int:
    with patch(
        "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
        return_value=tokenizer,
    ):
        return run_token_boundaries(cfg)


class TestTokenBoundaryManifestSchemaVersion2:
    def test_run_writes_v2_manifest_with_base_shard_hash(self, tmp_path):
        feature_dir, data_dir, tokenizer = _make_env(tmp_path)
        cfg = OmegaConf.create({
            "feature_dir": str(feature_dir),
            "segment_size": SEGMENT_SIZE,
            "data_dir": str(data_dir),
            "dry_run": False,
            "resume": False,
            "force": False,
            "workers": 1,
        })
        with patch(
            "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
            return_value=tokenizer,
        ):
            ret = run_token_boundaries(cfg)
        assert ret == 0

        checksum = compute_tokenizer_checksum(tokenizer)
        d = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
        manifest = json.loads((d / MANIFEST_FILENAME).read_text())

        assert manifest.get("schema_version") == 2
        assert "base_shard_content_hash" in manifest
        assert manifest["kind"] == OverlayKind.SEG_BOUNDARY.value

    def test_discover_overlay_with_base_shard_paths_succeeds(self, tmp_path):
        from steering.feature_io import glob_features

        feature_dir, data_dir, tokenizer = _make_env(tmp_path)
        cfg = OmegaConf.create({
            "feature_dir": str(feature_dir),
            "segment_size": SEGMENT_SIZE,
            "data_dir": str(data_dir),
            "dry_run": False,
            "resume": False,
            "force": False,
            "workers": 1,
        })
        with patch(
            "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
            return_value=tokenizer,
        ):
            run_token_boundaries(cfg)

        checksum = compute_tokenizer_checksum(tokenizer)
        result = discover_overlay(
            feature_dir, OverlayKind.SEG_BOUNDARY, checksum,
            base_shard_paths=glob_features(feature_dir),
        )
        assert result == overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)


def test_incomplete_token_overlay_requires_resume(tmp_path):
    """F1: a valid token manifest with a missing nonempty shard fails loud."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    cfg = _cfg(feature_dir, data_dir)
    assert _run(cfg, tokenizer) == 0

    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    missing = overlay_shard_path(output, "shard_00000")
    missing.unlink()

    before_failure = _inventory(feature_dir)
    with pytest.raises(RuntimeError, match="incomplete"):
        _run(cfg, tokenizer)
    assert _inventory(feature_dir) == before_failure
    assert not missing.exists()

    cfg.resume = True
    assert _run(cfg, tokenizer) == 0
    assert missing.is_file()


def test_token_producer_rejects_stale_v2_before_mutation(tmp_path):
    """F2: producer discovery authenticates v2 before touching outputs."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    cfg = _cfg(feature_dir, data_dir)
    assert _run(cfg, tokenizer) == 0

    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    target = glob_features(feature_dir)[0]
    data = load_features(target)
    data["feature_mags"] = data["feature_mags"] + 1
    save_features(data, target)

    watched = [
        output / MANIFEST_FILENAME,
        overlay_shard_path(output, "shard_00000"),
        feature_dir / BASE_SHARD_HASH_CACHE_FILENAME,
    ]
    before = {path: path.read_bytes() for path in watched}
    with pytest.raises(OverlayStaleError):
        _run(cfg, tokenizer)
    assert {path: path.read_bytes() for path in watched} == before


@pytest.mark.parametrize("legacy_v1", [False, True])
def test_token_resume_preserves_existing_manifest(tmp_path, legacy_v1):
    """F2: resume fills payloads without restamping either v1 or v2."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    cfg = _cfg(feature_dir, data_dir)
    assert _run(cfg, tokenizer) == 0

    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    manifest_path = output / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = "2001-02-03T04:05:06Z"
    if legacy_v1:
        manifest.pop("base_shard_content_hash")
        manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_before = manifest_path.read_bytes()
    overlay_shard_path(output, "shard_00000").unlink()

    cfg.resume = True
    assert _run(cfg, tokenizer) == 0
    assert overlay_shard_path(output, "shard_00000").is_file()
    assert manifest_path.read_bytes() == manifest_before


def test_token_resume_rejects_corrupt_present_payload(tmp_path):
    """F4: resume validates a present payload instead of blindly skipping it."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    cfg = _cfg(feature_dir, data_dir)
    assert _run(cfg, tokenizer) == 0

    checksum = compute_tokenizer_checksum(tokenizer)
    payload = overlay_shard_path(
        overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum),
        "shard_00000",
    )
    save_overlay_shard(
        {
            TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(4, dtype=t.int64),
            TensorKey.SEGMENT_TOKEN_ENDS: t.ones(4, dtype=t.int32),
        },
        payload,
    )
    cfg.resume = True
    before_failure = _inventory(feature_dir)
    with pytest.raises(ValueError, match="torch.int32"):
        _run(cfg, tokenizer)
    assert _inventory(feature_dir) == before_failure


@pytest.mark.parametrize("missing_indexes", [(0,), (1, 2)])
def test_missing_pending_parquet_fails_before_any_output(
    tmp_path, missing_indexes
):
    """F5: missing pending parquet is one fail-loud, write-free preflight."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path, n_shards=3)
    for index in missing_indexes:
        (data_dir / f"shard_{index:05d}.parquet").unlink()
    before = _inventory(feature_dir)
    cfg = _cfg(feature_dir, data_dir)
    with pytest.raises(
        FileNotFoundError,
        match=rf"{len(missing_indexes)} pending nonempty.*shard_{missing_indexes[0]:05d}",
    ):
        _run(cfg, tokenizer)
    assert _inventory(feature_dir) == before


def test_two_range_token_resume_is_range_local_and_preserves_manifest(tmp_path):
    """Two explicit ranges can complete one v2 overlay without restamping."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path, n_shards=2)
    cfg = _cfg(feature_dir, data_dir, shard_start=0, shard_end=1)
    assert _run(cfg, tokenizer) == 0

    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    first = overlay_shard_path(output, "shard_00000")
    second = overlay_shard_path(output, "shard_00001")
    assert first.is_file()
    assert not second.exists()
    manifest_path = output / MANIFEST_FILENAME
    manifest_before = manifest_path.read_bytes()

    cfg.resume = True
    cfg.shard_start = 1
    cfg.shard_end = 2
    assert _run(cfg, tokenizer) == 0
    assert second.is_file()
    assert manifest_path.read_bytes() == manifest_before


def test_missing_parquet_outside_selected_range_is_ignored(tmp_path):
    """F5 preflight checks only selected nonempty token shards."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path, n_shards=2)
    (data_dir / "shard_00001.parquet").unlink()
    cfg = _cfg(feature_dir, data_dir, shard_start=0, shard_end=1)
    assert _run(cfg, tokenizer) == 0

    output = overlay_dir(
        feature_dir, OverlayKind.SEG_BOUNDARY, compute_tokenizer_checksum(tokenizer)
    )
    assert overlay_shard_path(output, "shard_00000").is_file()
    assert not overlay_shard_path(output, "shard_00001").exists()


def test_valid_resume_present_payload_needs_no_parquet(tmp_path):
    """F5 preflight excludes a structurally valid resume-present payload."""
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    cfg = _cfg(feature_dir, data_dir)
    assert _run(cfg, tokenizer) == 0
    (data_dir / "shard_00000.parquet").unlink()
    before_resume = _inventory(feature_dir)

    cfg.resume = True
    assert _run(cfg, tokenizer) == 0
    assert _inventory(feature_dir) == before_resume
