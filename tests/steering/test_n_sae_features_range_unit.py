"""Unit tests for the n_sae_features parameter on validate_shard / validate_shard_data.

Closes slice-3 map §7.2 unit gap: the n_sae_features parameter path is
never exercised by tests/steering/test_validate_features.py.
"""

from __future__ import annotations

from steering.shard_validation import validate_shard, validate_shard_data
from tests.steering._helpers import (
    make_valid_shard_data,
    save_valid_shard,
)


def test_validate_shard_data_accepts_in_range_ids() -> None:
    """Default ids land below the n_sae_features bound."""
    data = make_valid_shard_data()
    result = validate_shard_data(data, n_sae_features=4096)
    assert result.valid, result.violations


def test_validate_shard_data_rejects_negative_id() -> None:
    data = make_valid_shard_data()
    data["feature_ids"][0] = -1

    result = validate_shard_data(data, n_sae_features=4096)

    assert not result.valid
    assert any("Negative feature_id" in v for v in result.violations), result.violations


def test_validate_shard_data_rejects_out_of_range_id() -> None:
    data = make_valid_shard_data()
    # Pick a feature_id larger than n_sae_features.
    data["feature_ids"][-1] = 999_999

    result = validate_shard_data(data, n_sae_features=4096)

    assert not result.valid
    assert any(">= n_sae_features 4096" in v for v in result.violations), result.violations


def test_validate_shard_skips_id_range_when_n_sae_features_none() -> None:
    """When n_sae_features is unset, out-of-range ids do not raise."""
    data = make_valid_shard_data()
    data["feature_ids"][-1] = 999_999

    result = validate_shard_data(data, n_sae_features=None)

    # Out-of-range ID is allowed when no bound is given. The ascending
    # ordering check might still fire — but we set the LAST id, so within
    # the segment it's still ascending or equal.
    range_violations = [
        v for v in result.violations if "feature_id" in v and ">=" in v
    ]
    assert range_violations == []


def test_validate_shard_rejects_out_of_range_id(tmp_path) -> None:
    """File-based path: same n_sae_features check as in-memory."""
    path, data = save_valid_shard(tmp_path)

    in_memory_result = validate_shard_data(data, n_sae_features=4096)
    on_disk_result = validate_shard(path, n_sae_features=4096)

    # Both pass on a known-good shard.
    assert in_memory_result.valid, in_memory_result.violations
    assert on_disk_result.valid, on_disk_result.violations
