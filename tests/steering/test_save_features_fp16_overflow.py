"""Regression tests for save_features fp16 range guards.

Bug context: PR #348 Wave 3C.7 spilled Pass-1 sorted magnitudes as fp32, but
the information was already lost upstream: save_features clips feature_mags
to fp16 at shard-write time. Any magnitude > 65504 became +Inf on disk before
compute_statistics ever read it. Tiny positive magnitudes can likewise
underflow to zero and violate the persisted shard contract. This test file
locks in loud failures at save_features instead of silent corruption.
"""

import hashlib

import pytest
import torch as t

from errors import ValidationError
from steering.feature_io import FP16_SAFE_MAX, load_features, save_features


def _minimal_shard(mags: t.Tensor) -> dict:
    """Build a minimal shard dict with a single segment holding the given mags."""
    n = mags.numel()
    return {
        "offsets": t.tensor([0, n], dtype=t.int64),
        "feature_ids": t.arange(n, dtype=t.int32),
        "feature_mags": mags.to(t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([512], dtype=t.int32),
        "doc_token_counts": t.tensor([256], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


def test_save_features_raises_on_fp16_overflow(tmp_path):
    """A magnitude > FP16_SAFE_MAX must raise ValueError, not silently become +Inf."""
    mags = t.tensor([1.0, 100000.0, 2.0], dtype=t.float32)
    shard = _minimal_shard(mags)
    with pytest.raises(ValueError, match="fp16 safe threshold"):
        save_features(shard, tmp_path / "shard_00000_features")


def test_save_features_accepts_fp16_safe_mags(tmp_path):
    """Magnitudes below the safe threshold should round-trip without error."""
    mags = t.tensor([0.1, 1.5, 12.0, 1000.0, 50000.0], dtype=t.float32)
    shard = _minimal_shard(mags)
    out = save_features(shard, tmp_path / "shard_00000_features")
    loaded = load_features(out)
    # After sort within segment (ascending by feature_id), the order is preserved.
    # Compare within fp16 precision (~0.1% relative).
    loaded_mags = loaded["feature_mags"].to(t.float32)
    assert t.allclose(loaded_mags, mags, rtol=1e-3, atol=1e-3)
    assert t.isfinite(loaded_mags).all()


def test_fp16_safe_threshold_margin_below_fp16_max():
    """The safe threshold must leave margin below fp16's true max (65504)."""
    assert FP16_SAFE_MAX < float(t.finfo(t.float16).max)
    # And we want *meaningful* margin, not just 1 ULP.
    assert FP16_SAFE_MAX <= 65000.0


def test_save_features_raises_on_negative_overflow(tmp_path):
    """abs() check: a large-magnitude negative value should also raise."""
    mags = t.tensor([1.0, -80000.0, 2.0], dtype=t.float32)
    shard = _minimal_shard(mags)
    with pytest.raises(ValueError, match="fp16 safe threshold"):
        save_features(shard, tmp_path / "shard_00000_features")


def test_save_features_fp16_underflow_raises_before_overwrite(tmp_path):
    """A positive magnitude that becomes zero in fp16 must not replace a shard."""
    path = tmp_path / "shard_00000_features"
    output_path = save_features(_minimal_shard(t.tensor([1.0])), path)
    before = hashlib.sha256(output_path.read_bytes()).hexdigest()

    underflow = t.tensor([1e-8], dtype=t.float32)
    assert underflow.item() > 0
    assert underflow.to(t.float16).item() == 0

    with pytest.raises(
        ValidationError,
        match=r"shard_00000_features\.safetensors\.zst.*positive.*float16.*zero",
    ):
        save_features(_minimal_shard(underflow), path)

    after = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert after == before
