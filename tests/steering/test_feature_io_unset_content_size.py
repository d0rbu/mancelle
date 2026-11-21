"""Regression coverage for feature shards with optional zstd headers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
import zstandard
from safetensors.torch import save as safetensors_save_bytes

from steering.feature_io import (
    COMPRESSED_EXT,
    RegularFileIdentity,
    load_features,
    load_features_stable,
)


def _minimal_safetensors_bytes() -> bytes:
    tensors = {
        "feature_ids": t.zeros(0, dtype=t.int64),
        "feature_mags": t.zeros(0, dtype=t.float32),
        "offsets": t.zeros(1, dtype=t.int64),
        "doc_ids": t.zeros(0, dtype=t.int64),
        "segment_char_starts": t.zeros(0, dtype=t.int64),
        "segment_char_ends": t.zeros(0, dtype=t.int64),
    }
    metadata = {
        "n_docs": "0",
        "n_segments": "0",
        "layer": "22",
        "segment_size": "128",
        "context_window": "2048",
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "context_independent_features": "true",
        "ignore_bos_features": "true",
        "l0_tier": "medium",
    }
    return safetensors_save_bytes(tensors, metadata=metadata)


@pytest.mark.parametrize("write_content_size", [True, False], ids=["known-size", "unknown-size"])
@pytest.mark.parametrize("stable_read", [False, True], ids=["ordinary", "stable"])
def test_load_features_accepts_optional_zstd_content_size(
    tmp_path: Path,
    *,
    write_content_size: bool,
    stable_read: bool,
) -> None:
    payload = _minimal_safetensors_bytes()
    compressed = zstandard.ZstdCompressor(
        level=3,
        write_content_size=write_content_size,
    ).compress(payload)
    frame_size = zstandard.frame_content_size(compressed)
    if write_content_size:
        assert frame_size == len(payload)
    else:
        assert frame_size in {zstandard.CONTENTSIZE_UNKNOWN, -1}

    path = tmp_path / f"shard_99999_features{COMPRESSED_EXT}"
    path.write_bytes(compressed)
    if stable_read:
        data = load_features_stable(
            path,
            expected_identity=RegularFileIdentity.capture(path),
        )
    else:
        data = load_features(path)

    assert data["n_docs"] == 0
    assert data["n_segments"] == 0
    assert data["layer"] == 22
