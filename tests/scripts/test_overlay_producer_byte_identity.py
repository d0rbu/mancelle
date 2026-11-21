"""Reference-vs-candidate byte-identity gate for healthy overlay production."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import zstandard
from safetensors.torch import load as safetensors_load

from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import glob_features, save_features
from steering.overlay_io import OverlayKind, discover_overlay
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)

_REFERENCE_COMMIT = "2c3dd533ee8c3f17094cb22beee9457b045f8f8b"
_N_SAE_FEATURES = 16
_RUNNER = Path(__file__).with_name("_overlay_producer_parity_runner.py")


def _nonempty_shard(shard_index: int) -> dict[str, object]:
    feature_offset = shard_index * 6
    return {
        "offsets": t.tensor([0, 3, 6], dtype=t.int64),
        "feature_ids": t.tensor(
            [
                feature_offset,
                feature_offset + 2,
                feature_offset + 4,
                feature_offset + 1,
                feature_offset + 3,
                feature_offset + 5,
            ],
            dtype=t.int32,
        ),
        "feature_mags": t.tensor(
            [0.11, 0.23, 0.37, 0.41, 0.59, 0.83], dtype=t.float32
        )
        + shard_index * 0.01,
        "doc_ids": t.tensor([0, 0], dtype=t.int32),
        "segment_indices": t.tensor([0, 1], dtype=t.int32),
        "segment_positions": t.tensor([0, 8], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 12], dtype=t.int32),
        "segment_char_ends": t.tensor([12, 20], dtype=t.int32),
        "doc_token_counts": t.tensor([16], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 2,
        "layer": 8,
        "context_window": 64,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 8,
    }


def _empty_shard() -> dict[str, object]:
    return {
        "offsets": t.zeros(1, dtype=t.int64),
        "feature_ids": t.empty(0, dtype=t.int32),
        "feature_mags": t.empty(0, dtype=t.float32),
        "doc_ids": t.empty(0, dtype=t.int32),
        "segment_indices": t.empty(0, dtype=t.int32),
        "segment_positions": t.empty(0, dtype=t.int32),
        "segment_char_starts": t.empty(0, dtype=t.int32),
        "segment_char_ends": t.empty(0, dtype=t.int32),
        "doc_token_counts": t.empty(0, dtype=t.int32),
        "n_docs": 0,
        "n_segments": 0,
        "layer": 8,
        "context_window": 64,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 8,
    }


def _feature_stat(feature_id: int) -> PerFeatureStat:
    percentiles = {
        f"p{percentile}": feature_id * 0.001 + percentile * 0.01
        for percentile in range(1, 100)
    }
    return PerFeatureStat(
        count=4,
        pct_segments_active=1.0,
        mean_mag=0.5,
        mean_mag_incl_zeros=0.5,
        std_mag=0.1,
        min_mag=percentiles["p1"],
        max_mag=percentiles["p99"],
        median_mag=percentiles["p50"],
        percentiles=percentiles,
        histogram=None,
    )


def _statistics() -> ExtractionStatistics:
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=8,
            context_window=64,
            layer=8,
            n_sae_features=_N_SAE_FEATURES,
            generated_at="2026-07-20T00:00:00+00:00",
        ),
        corpus=CorpusStats(
            n_shards_processed=3,
            n_docs=2,
            n_tokens_gpt2=32,
            mean_tokens_per_doc=16.0,
            median_tokens_per_doc=16,
        ),
        features=SegmentFeatureStats(
            n_segments=4,
            n_features_seen=_N_SAE_FEATURES,
            mean_active_per_segment=3.0,
            median_active_per_segment=3,
            p90_active_per_segment=3,
            max_active_per_segment=3,
            min_active_per_segment=3,
            n_total_sae_features=_N_SAE_FEATURES,
            pct_features_active=1.0,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1,
            mean_bytes_per_doc=0.5,
            mean_bytes_per_token=0.03125,
        ),
        per_feature_stats={
            str(feature_id): _feature_stat(feature_id)
            for feature_id in range(_N_SAE_FEATURES)
        },
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=625_000_000,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
    )


def _build_neutral_fixture(root: Path) -> None:
    feature_dir = root / "features" / "s8"
    data_dir = root / "data"
    feature_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    texts = ["abcdefghijklmno"]
    for shard_index in range(2):
        stem = f"shard_{shard_index:05d}"
        save_features(
            _nonempty_shard(shard_index),
            feature_dir / f"{stem}_features",
            zstd_level=12,
        )
        pq.write_table(pa.table({"text": texts}), data_dir / f"{stem}.parquet")
    save_features(
        _empty_shard(),
        feature_dir / "shard_00002_features",
        zstd_level=12,
    )
    save_statistics(_statistics(), feature_dir / "feature_statistics.safetensors")


def _clone_reference(repo_root: Path, destination: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(repo_root),
            str(destination),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", _REFERENCE_COMMIT],
        check=True,
        text=True,
        capture_output=True,
    )


def _run_producers(
    *,
    source_root: Path,
    fixture_root: Path,
    resume: bool = False,
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    command = [
        sys.executable,
        str(_RUNNER),
        "--feature-dir",
        str(fixture_root / "features" / "s8"),
        "--data-dir",
        str(fixture_root / "data"),
    ]
    if resume:
        command.append("--resume")
    result = subprocess.run(
        command,
        cwd=source_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"isolated producer failed in {source_root}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _rewrite_reference_manifests_as_v1(
    *, source_root: Path, fixture_root: Path
) -> None:
    """Use the pinned reference write_manifest to create legacy v1 manifests."""
    script = """
import json
import sys
from pathlib import Path
from steering.overlay_io import OverlayKind, write_manifest

feature_dir = Path(sys.argv[1]) / "features" / "s8"
for pattern, kind in (
    ("pct/*/manifest.json", OverlayKind.PERCENTILE),
    ("tok/*/seg_boundaries/manifest.json", OverlayKind.SEG_BOUNDARY),
):
    manifest_path, = feature_dir.glob(pattern)
    manifest = json.loads(manifest_path.read_text())
    write_manifest(
        manifest_path.parent,
        manifest["checksum"],
        kind,
        metadata=manifest.get("metadata"),
    )
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script, str(fixture_root)],
        cwd=source_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _header_and_tensors(path: Path, *, compressed: bool) -> tuple[dict[str, str], dict[str, t.Tensor]]:
    payload = path.read_bytes()
    if compressed:
        payload = zstandard.ZstdDecompressor().decompress(payload)
    header_size = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_size])
    metadata = header.get("__metadata__", {})
    tensors = safetensors_load(payload)
    return metadata, tensors


def _tensor_content_digest(metadata: dict[str, str], tensors: dict[str, t.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tensor.ndim).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.view(t.uint8).numpy().tobytes())
        digest.update(b"\0")
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    return digest.hexdigest()


def _overlay_files(feature_dir: Path) -> tuple[list[Path], list[Path]]:
    percentile = sorted(feature_dir.glob("pct/*/shard_*.safetensors.zst"))
    token = sorted(feature_dir.glob("tok/*/seg_boundaries/shard_*.safetensors.zst"))
    return percentile, token


def _single_manifest(feature_dir: Path, pattern: str) -> Path:
    paths = sorted(feature_dir.glob(pattern))
    assert len(paths) == 1, paths
    return paths[0]


def _assert_tensor_file_equal(reference: Path, candidate: Path) -> None:
    reference_metadata, reference_tensors = _header_and_tensors(reference, compressed=True)
    candidate_metadata, candidate_tensors = _header_and_tensors(candidate, compressed=True)
    assert reference_metadata == candidate_metadata
    assert reference_tensors.keys() == candidate_tensors.keys()
    for key in reference_tensors:
        reference_tensor = reference_tensors[key]
        candidate_tensor = candidate_tensors[key]
        assert reference_tensor.dtype == candidate_tensor.dtype
        assert reference_tensor.shape == candidate_tensor.shape
        assert t.equal(reference_tensor, candidate_tensor)
    assert _tensor_content_digest(
        reference_metadata, reference_tensors
    ) == _tensor_content_digest(candidate_metadata, candidate_tensors)


def _assert_manifest_equal(reference: Path, candidate: Path) -> None:
    reference_bytes = reference.read_bytes()
    candidate_bytes = candidate.read_bytes()
    reference_manifest = json.loads(reference_bytes)
    candidate_manifest = json.loads(candidate_bytes)
    assert reference_manifest == candidate_manifest
    reference_canonical = json.dumps(
        reference_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    candidate_canonical = json.dumps(
        candidate_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(reference_canonical).digest() == hashlib.sha256(
        candidate_canonical
    ).digest()
    assert hashlib.sha256(reference_bytes).digest() == hashlib.sha256(candidate_bytes).digest()


def _raw_overlay_digests(feature_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(feature_dir.glob("pct/*/*")) + sorted(
        feature_dir.glob("tok/*/seg_boundaries/*")
    ):
        if path.is_file():
            result[str(path.relative_to(feature_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_producers_match_origin_main_on_frozen_healthy_fixture(tmp_path: Path) -> None:
    """Three pct shards, two token shards, and v2 manifests match origin/main."""
    repo_root = Path(__file__).resolve().parents[2]
    neutral = tmp_path / "neutral"
    reference_fixture = tmp_path / "reference-fixture"
    candidate_fixture = tmp_path / "candidate-fixture"
    reference_source = tmp_path / "origin-main"
    _build_neutral_fixture(neutral)
    shutil.copytree(neutral, reference_fixture)
    shutil.copytree(neutral, candidate_fixture)
    _clone_reference(repo_root, reference_source)

    _run_producers(source_root=reference_source, fixture_root=reference_fixture)
    _run_producers(source_root=repo_root, fixture_root=candidate_fixture)

    reference_feature_dir = reference_fixture / "features" / "s8"
    candidate_feature_dir = candidate_fixture / "features" / "s8"
    reference_pct, reference_tok = _overlay_files(reference_feature_dir)
    candidate_pct, candidate_tok = _overlay_files(candidate_feature_dir)
    assert [path.name for path in reference_pct] == [
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
        "shard_00002.safetensors.zst",
    ]
    assert [path.name for path in reference_tok] == [
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
    ]
    assert [path.name for path in candidate_pct] == [path.name for path in reference_pct]
    assert [path.name for path in candidate_tok] == [path.name for path in reference_tok]
    for reference_path, candidate_path in zip(reference_pct, candidate_pct, strict=True):
        _assert_tensor_file_equal(reference_path, candidate_path)
    for reference_path, candidate_path in zip(reference_tok, candidate_tok, strict=True):
        _assert_tensor_file_equal(reference_path, candidate_path)

    reference_pct_manifest = _single_manifest(reference_feature_dir, "pct/*/manifest.json")
    candidate_pct_manifest = _single_manifest(candidate_feature_dir, "pct/*/manifest.json")
    reference_tok_manifest = _single_manifest(
        reference_feature_dir, "tok/*/seg_boundaries/manifest.json"
    )
    candidate_tok_manifest = _single_manifest(
        candidate_feature_dir, "tok/*/seg_boundaries/manifest.json"
    )
    _assert_manifest_equal(reference_pct_manifest, candidate_pct_manifest)
    _assert_manifest_equal(reference_tok_manifest, candidate_tok_manifest)

    percentile_manifest = json.loads(candidate_pct_manifest.read_bytes())
    token_manifest = json.loads(candidate_tok_manifest.read_bytes())
    assert percentile_manifest["schema_version"] == 2
    assert token_manifest["schema_version"] == 2
    assert percentile_manifest["metadata"] == {
        "n_sae_features": _N_SAE_FEATURES,
        "stats_file": "feature_statistics.safetensors",
    }
    assert "metadata" not in token_manifest
    forbidden = {
        "status",
        "expected_shard_stems",
        "source_bos_char_len",
        "build_manifest",
    }
    assert forbidden.isdisjoint(percentile_manifest)
    assert forbidden.isdisjoint(token_manifest)
    assert not list(candidate_feature_dir.rglob("build_manifest.json"))
    assert not [path for path in candidate_feature_dir.rglob("*") if ".staging" in path.name]

    reference_stats = reference_feature_dir / "feature_statistics.safetensors"
    candidate_stats = candidate_feature_dir / "feature_statistics.safetensors"
    reference_metadata, reference_tensors = _header_and_tensors(reference_stats, compressed=False)
    candidate_metadata, candidate_tensors = _header_and_tensors(candidate_stats, compressed=False)
    assert reference_metadata == candidate_metadata
    assert reference_metadata["percentile_checksum"] == percentile_manifest["checksum"]
    assert reference_metadata["tokenizer_checksum"] == token_manifest["checksum"]
    assert _tensor_content_digest(
        reference_metadata, reference_tensors
    ) == _tensor_content_digest(candidate_metadata, candidate_tensors)

    before_noop = _raw_overlay_digests(candidate_feature_dir)
    _run_producers(source_root=repo_root, fixture_root=candidate_fixture)
    assert _raw_overlay_digests(candidate_feature_dir) == before_noop
    _run_producers(source_root=repo_root, fixture_root=candidate_fixture, resume=True)
    assert _raw_overlay_digests(candidate_feature_dir) == before_noop


def test_reference_v1_v2_overlays_remain_accepted_and_byte_stable(
    tmp_path: Path,
) -> None:
    """Discovery, FeatureIndex, no-op, and resume preserve reference v1/v2."""
    repo_root = Path(__file__).resolve().parents[2]
    neutral = tmp_path / "neutral"
    reference_fixture = tmp_path / "reference"
    reference_source = tmp_path / "origin-main"
    _build_neutral_fixture(neutral)
    shutil.copytree(neutral, reference_fixture)
    _clone_reference(repo_root, reference_source)
    _run_producers(source_root=reference_source, fixture_root=reference_fixture)

    fixtures: list[tuple[int, Path]] = []
    for schema_version in (2, 1):
        fixture = tmp_path / f"candidate-v{schema_version}"
        shutil.copytree(reference_fixture, fixture)
        if schema_version == 1:
            _rewrite_reference_manifests_as_v1(
                source_root=reference_source, fixture_root=fixture
            )
        fixtures.append((schema_version, fixture))

    for schema_version, fixture in fixtures:
        feature_dir = fixture / "features" / "s8"
        pct_manifest_path = _single_manifest(feature_dir, "pct/*/manifest.json")
        tok_manifest_path = _single_manifest(
            feature_dir, "tok/*/seg_boundaries/manifest.json"
        )
        pct_manifest = json.loads(pct_manifest_path.read_bytes())
        tok_manifest = json.loads(tok_manifest_path.read_bytes())
        if schema_version == 1:
            assert "schema_version" not in pct_manifest
            assert "base_shard_content_hash" not in pct_manifest
            assert "schema_version" not in tok_manifest
            assert "base_shard_content_hash" not in tok_manifest
        else:
            assert pct_manifest["schema_version"] == 2
            assert tok_manifest["schema_version"] == 2

        bases = glob_features(feature_dir)
        assert discover_overlay(
            feature_dir,
            OverlayKind.PERCENTILE,
            pct_manifest["checksum"],
            base_shard_paths=bases,
        ) == pct_manifest_path.parent
        assert discover_overlay(
            feature_dir,
            OverlayKind.SEG_BOUNDARY,
            tok_manifest["checksum"],
            base_shard_paths=bases,
        ) == tok_manifest_path.parent
        index = FeatureIndex(
            str(feature_dir),
            percentile_checksum=pct_manifest["checksum"],
            tokenizer_checksum=tok_manifest["checksum"],
        )
        segments = index.lookup("shard_00000", 0)
        assert all(segment.feature_percentiles is not None for segment in segments)
        assert all(segment.token_start is not None for segment in segments)

        complete_before = _raw_overlay_digests(feature_dir)
        _run_producers(source_root=repo_root, fixture_root=fixture)
        assert _raw_overlay_digests(feature_dir) == complete_before
        _run_producers(source_root=repo_root, fixture_root=fixture, resume=True)
        assert _raw_overlay_digests(feature_dir) == complete_before

        (pct_manifest_path.parent / "shard_00002.safetensors.zst").unlink()
        (tok_manifest_path.parent / "shard_00001.safetensors.zst").unlink()
        preserved_before = _raw_overlay_digests(feature_dir)
        _run_producers(source_root=repo_root, fixture_root=fixture, resume=True)
        after_resume = _raw_overlay_digests(feature_dir)
        for relative_path, digest in preserved_before.items():
            assert after_resume[relative_path] == digest
        pct_files, tok_files = _overlay_files(feature_dir)
        assert len(pct_files) == 3
        assert len(tok_files) == 2
