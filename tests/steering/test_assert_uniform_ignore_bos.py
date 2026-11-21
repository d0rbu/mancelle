"""Unit tests for read_ignore_bos_from_shard / assert_uniform_ignore_bos.

Mirrors tests/steering/test_assert_uniform_l0_tier.py but with bool semantics
(absent == False, not a distinct "unknown").
"""

import pytest
import torch as t

from steering.feature_io import (
    CorruptFeatureShardError,
    FeatureDirectoryMismatchError,
    assert_uniform_ignore_bos,
    glob_features,
    read_ignore_bos_from_shard,
    save_features,
)


def _shard(tmp_path, name, **extra):
    d = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "n_docs": 1, "n_segments": 1, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_mode": "fixed_tokens", "segment_size": 128,
    }
    d.update(extra)
    return save_features(d, tmp_path / name)


def test_read_true(tmp_path):
    p = _shard(tmp_path, "a", ignore_bos_features=True)
    assert read_ignore_bos_from_shard(p) is True


def test_read_false(tmp_path):
    p = _shard(tmp_path, "a", ignore_bos_features=False)
    assert read_ignore_bos_from_shard(p) is False


def test_read_absent_is_false(tmp_path):
    p = _shard(tmp_path, "a")  # legacy: no key written
    assert read_ignore_bos_from_shard(p) is False


def test_uniform_all_true(tmp_path):
    _shard(tmp_path, "shard_00000", ignore_bos_features=True)
    _shard(tmp_path, "shard_00001", ignore_bos_features=True)
    assert assert_uniform_ignore_bos(glob_features(tmp_path)) is True


def test_uniform_all_legacy_is_false(tmp_path):
    _shard(tmp_path, "shard_00000")
    _shard(tmp_path, "shard_00001")
    assert assert_uniform_ignore_bos(glob_features(tmp_path)) is False


def test_empty_iterable_is_false(tmp_path):
    assert assert_uniform_ignore_bos([]) is False


def test_disagreement_raises(tmp_path):
    a = _shard(tmp_path, "shard_00000", ignore_bos_features=True)
    b = _shard(tmp_path, "shard_00001", ignore_bos_features=False)
    with pytest.raises(FeatureDirectoryMismatchError):
        assert_uniform_ignore_bos([a, b])


def test_legacy_absent_and_explicit_false_no_mismatch(tmp_path):
    _shard(tmp_path, "shard_00000")  # legacy: no ignore_bos_features key
    _shard(tmp_path, "shard_00001", ignore_bos_features=False)  # explicit false
    # absent reads as False, explicit False is False → uniform, no raise
    assert assert_uniform_ignore_bos(glob_features(tmp_path)) is False


# ---------------------------------------------------------------------------
# Post-#466: corrupt-shard surface — new tests
# ---------------------------------------------------------------------------


def test_read_ignore_bos_corrupt_shard_raises(tmp_path):
    """A corrupt shard (empty or non-zstd) MUST raise CorruptFeatureShardError,
    never be silently read as ignore_bos=False — that would hide corruption
    behind a wrong uniformity result."""
    empty_path = tmp_path / "shard_00000_features.safetensors.zst"
    notzstd_path = tmp_path / "shard_00001_features.safetensors.zst"

    empty_path.write_bytes(b"")
    notzstd_path.write_bytes(b"not a zstd frame at all")

    with pytest.raises(CorruptFeatureShardError):
        read_ignore_bos_from_shard(empty_path)

    with pytest.raises(CorruptFeatureShardError):
        read_ignore_bos_from_shard(notzstd_path)


def test_read_ignore_bos_invalid_value_raises(tmp_path, monkeypatch):
    """A shard whose ignore_bos_features metadata is a garbage string must
    raise ValueError (specifically the 'Invalid ignore_bos_features' branch),
    not silently coerce to True or False."""
    import steering.feature_io as feature_io_mod

    p = _shard(tmp_path, "a", ignore_bos_features=True)
    monkeypatch.setattr(
        feature_io_mod,
        "read_shard_metadata",
        lambda _path: {"ignore_bos_features": "yes"},
    )
    with pytest.raises(ValueError, match="Invalid ignore_bos_features"):
        read_ignore_bos_from_shard(p)


def test_assert_uniform_ignore_bos_propagates_corrupt_shard(tmp_path):
    """assert_uniform_ignore_bos must NOT swallow CorruptFeatureShardError —
    it propagates (mirrors assert_uniform_l0_tier behavior).  A corrupt shard
    must not be silently treated as ignore_bos=False."""
    good = _shard(tmp_path, "shard_00000", ignore_bos_features=True)
    corrupt = tmp_path / "shard_00001_features.safetensors.zst"
    corrupt.write_bytes(b"")

    with pytest.raises(CorruptFeatureShardError):
        assert_uniform_ignore_bos([good, corrupt])
