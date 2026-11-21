"""Cross-stage ``segment_size`` consistency enforcement (Wave 3D task 4).

Verifies that the ``segment_size`` declared in a Hydra config matches:
    1. The value parsed from the hierarchical feature directory (``s{N}``).
    2. The value stored inside each shard safetensors file's metadata.

Tests construct REAL mismatched on-disk artifacts via ``tmp_path`` + real
safetensors writes — no mocking.

Run with:
    uv run pytest tests/test_segment_size_enforcement.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.feature_io import load_features, read_shard_metadata, save_features
from steering.feature_paths import (
    SegmentSizeMismatch,
    verify_segment_size,
)


def _make_minimal_shard_dict(segment_size: int) -> dict:
    return {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([7], dtype=t.int32),
        "feature_mags": t.tensor([0.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([10], dtype=t.int32),
        "doc_token_counts": t.tensor([10], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
    }


def _write_shard_at(
    tmp_path: Path,
    dir_segment_size: int,
    shard_segment_size: int,
) -> Path:
    feature_dir = (
        tmp_path
        / f"s{dir_segment_size}"
        / "l12"
        / "a_res"
        / "p_mean_pool"
        / "f65k"
        / "m_gemma-2-2b"
        / "base"
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    shard = _make_minimal_shard_dict(shard_segment_size)
    return save_features(shard, feature_dir / "shard_00000_features")


def test_shard_metadata_records_segment_size(tmp_path: Path) -> None:
    out = _write_shard_at(tmp_path, dir_segment_size=128, shard_segment_size=128)

    header = read_shard_metadata(out)
    assert "segment_size" in header, (
        f"shard metadata missing 'segment_size'; keys={sorted(header)}"
    )
    assert header["segment_size"] == "128"

    loaded = load_features(out)
    assert loaded["segment_size"] == 128


def test_verify_segment_size_accepts_match(tmp_path: Path) -> None:
    out = _write_shard_at(tmp_path, dir_segment_size=128, shard_segment_size=128)
    verify_segment_size(out, expected_segment_size=128)
    verify_segment_size(out.parent, expected_segment_size=128)


def test_verify_segment_size_rejects_cfg_vs_dir_mismatch(tmp_path: Path) -> None:
    out = _write_shard_at(tmp_path, dir_segment_size=64, shard_segment_size=64)
    with pytest.raises(SegmentSizeMismatch) as excinfo:
        verify_segment_size(out, expected_segment_size=128)
    msg = str(excinfo.value)
    assert "128" in msg and "64" in msg


def test_verify_segment_size_rejects_cfg_vs_metadata_mismatch(
    tmp_path: Path,
) -> None:
    out = _write_shard_at(tmp_path, dir_segment_size=128, shard_segment_size=64)
    with pytest.raises(SegmentSizeMismatch) as excinfo:
        verify_segment_size(out, expected_segment_size=128)
    msg = str(excinfo.value)
    assert "128" in msg and "64" in msg


def test_verify_segment_size_rejects_dir_vs_metadata_mismatch(
    tmp_path: Path,
) -> None:
    out = _write_shard_at(tmp_path, dir_segment_size=128, shard_segment_size=64)

    with pytest.raises(SegmentSizeMismatch):
        verify_segment_size(out, expected_segment_size=128)

    with pytest.raises(SegmentSizeMismatch):
        verify_segment_size(out, expected_segment_size=64)
