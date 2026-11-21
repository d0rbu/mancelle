"""compute_statistics stamps the uniform ignore_bos_features into the stats file."""

import pytest
import torch as t

from steering.compute_statistics import compute_statistics
from steering.feature_io import FeatureDirectoryMismatchError, save_features
from steering.stats_io import load_statistics


def _shard(d, name, ignore_bos):
    save_features({
        "offsets": t.tensor([0, 1, 2], dtype=t.int64),
        "feature_ids": t.tensor([1, 2], dtype=t.int32),
        "feature_mags": t.tensor([1.0, 2.0], dtype=t.float32),
        "doc_ids": t.tensor([0, 0], dtype=t.int32),
        "segment_indices": t.tensor([0, 1], dtype=t.int32),
        "segment_positions": t.tensor([0, 128], dtype=t.int32),
        "doc_token_counts": t.tensor([256], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 5], dtype=t.int32),
        "segment_char_ends": t.tensor([5, 9], dtype=t.int32),
        "n_docs": 1, "n_segments": 2, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_mode": "fixed_tokens",
        "segment_size": 128, "ignore_bos_features": ignore_bos,
    }, d / name)


def test_stats_carries_ignore_bos_true(tmp_path):
    fdir = tmp_path / "f"
    fdir.mkdir()
    _shard(fdir, "shard_00000_features", True)
    _shard(fdir, "shard_00001_features", True)
    out = tmp_path / "feature_statistics.safetensors"
    compute_statistics(str(fdir), device="cpu", output_path=str(out),
                       exclude_last_shard=False)
    assert load_statistics(out).ignore_bos_features is True


def test_stats_carries_ignore_bos_false_for_legacy(tmp_path):
    fdir = tmp_path / "f"
    fdir.mkdir()
    save_features({
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([1], dtype=t.int32),
        "feature_mags": t.tensor([1.0], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "n_docs": 1, "n_segments": 1, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_mode": "fixed_tokens",
        "segment_size": 128,  # no ignore_bos_features key (legacy)
    }, fdir / "shard_00000_features")
    out = tmp_path / "feature_statistics.safetensors"
    compute_statistics(str(fdir), device="cpu", output_path=str(out),
                       exclude_last_shard=False)
    assert load_statistics(out).ignore_bos_features is False


def test_compute_statistics_disagreeing_ignore_bos_raises(tmp_path):
    """Two shards in one dir with conflicting ignore_bos_features raise FeatureDirectoryMismatchError.

    assert_uniform_ignore_bos() runs after glob_features (exclude_last_shard=False
    means both shards are in the uniformity domain), before stats are written.
    """
    fdir = tmp_path / "f"
    fdir.mkdir()
    _shard(fdir, "shard_00000_features", True)
    _shard(fdir, "shard_00001_features", False)
    out = tmp_path / "feature_statistics.safetensors"
    with pytest.raises(FeatureDirectoryMismatchError):
        compute_statistics(str(fdir), device="cpu", output_path=str(out),
                           exclude_last_shard=False)


@pytest.mark.parametrize("streaming", [False, True])
def test_compute_statistics_disagreeing_ignore_bos_raises_before_body_load(
    tmp_path, monkeypatch, streaming
):
    fdir = tmp_path / "f"
    fdir.mkdir()
    _shard(fdir, "shard_00000_features", True)
    _shard(fdir, "shard_00001_features", False)
    out = tmp_path / "feature_statistics.safetensors"

    def _fail_body_load(*_args, **_kwargs):
        raise AssertionError("compute_statistics loaded a shard body before fail-fast")

    monkeypatch.setattr("steering.compute_statistics.load_features", _fail_body_load)
    with pytest.raises(FeatureDirectoryMismatchError):
        compute_statistics(
            str(fdir),
            device="cpu",
            output_path=str(out),
            exclude_last_shard=False,
            streaming=streaming,
        )
