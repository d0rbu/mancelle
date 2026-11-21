"""shard_validation rejects a non-bool ignore_bos_features in both validators."""

import torch as t

from steering.feature_io import save_features
from steering.shard_validation import validate_shard, validate_shard_data


def _data(**extra):
    d = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "n_docs": 1, "n_segments": 1, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    d.update(extra)
    return d


def test_bool_ignore_bos_ok():
    assert validate_shard_data(_data(ignore_bos_features=True)).valid


def test_nonbool_ignore_bos_flagged():
    r = validate_shard_data(_data(ignore_bos_features="true"))
    assert not r.valid
    assert any("ignore_bos_features" in v for v in r.violations)


def test_nonbool_context_independent_still_flagged():
    r = validate_shard_data(_data(context_independent_features="true"))
    assert not r.valid
    assert any("context_independent_features" in v for v in r.violations)


def test_validate_shard_file_path_valid_ignore_bos_passes(tmp_path):
    """Pin: file-path validate_shard accepts a well-formed ignore_bos_features=True shard.

    The non-bool rejection tested by test_nonbool_ignore_bos_flagged is exercised
    in-memory via validate_shard_data. It is structurally unreachable through the
    on-disk validate_shard path because load_features coerces the raw string via
    _parse_metadata_value (raw.lower() == "true") before validate_shard_data sees
    it, so the value is always a proper bool after a disk round-trip.
    """
    p = save_features(_data(ignore_bos_features=True), tmp_path / "s")
    r = validate_shard(str(p))
    assert r.valid
    assert not any("ignore_bos_features" in v for v in r.violations)
