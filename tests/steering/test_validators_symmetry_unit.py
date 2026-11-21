"""Unit tests for the validate_shard / validate_shard_data symmetry fix.

Closes slice-3 P1-2 (validate_shard skipped context_independent_features bool
check) and slice-3 P1-3 (validate_shard_data skipped dtype check). Both
validators now share ``_check_csr_shapes``, ``_check_dtypes``,
``_check_feature_id_range``, and ``_check_bool_flag``.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.shard_validation import validate_shard, validate_shard_data
from tests.steering._helpers import (
    make_valid_shard_data,
    save_raw_shard,
    save_valid_shard,
)

# ---------------------------------------------------------------------------
# P1-3 — validate_shard_data must enforce dtypes
# ---------------------------------------------------------------------------


def test_validate_shard_data_rejects_wrong_offsets_dtype() -> None:
    """validate_shard_data MUST reject offsets dtype != int64 (slice-3 P1-3)."""
    data = make_valid_shard_data()
    data["offsets"] = data["offsets"].to(t.int32)

    result = validate_shard_data(data)

    assert not result.valid, "Expected violation on int32 offsets"
    assert any("offsets dtype" in v for v in result.violations), result.violations


def test_validate_shard_data_rejects_wrong_feature_ids_dtype() -> None:
    """validate_shard_data MUST reject feature_ids dtype != int32 (slice-3 P1-3)."""
    data = make_valid_shard_data()
    data["feature_ids"] = data["feature_ids"].to(t.int64)

    result = validate_shard_data(data)

    assert not result.valid
    assert any("feature_ids dtype" in v for v in result.violations), result.violations


def test_validate_shard_data_rejects_wrong_feature_mags_dtype() -> None:
    """validate_shard_data MUST reject feature_mags dtype != float32 (slice-3 P1-3)."""
    data = make_valid_shard_data()
    data["feature_mags"] = data["feature_mags"].to(t.float64)

    result = validate_shard_data(data)

    assert not result.valid
    assert any("feature_mags dtype" in v for v in result.violations), result.violations


def test_validate_shard_data_rejects_wrong_doc_ids_dtype() -> None:
    """validate_shard_data MUST reject doc_ids dtype != int32 (slice-3 P1-3)."""
    data = make_valid_shard_data()
    data["doc_ids"] = data["doc_ids"].to(t.int64)

    result = validate_shard_data(data)

    assert not result.valid
    assert any("doc_ids dtype" in v for v in result.violations), result.violations


@pytest.mark.parametrize(
    "field",
    [
        "segment_indices",
        "segment_positions",
        "segment_char_starts",
        "segment_char_ends",
        "doc_token_counts",
    ],
)
def test_validate_shard_data_rejects_wrong_segmentation_dtype(field: str) -> None:
    """Segmentation metadata tensors are part of the int32 shard contract."""
    data = make_valid_shard_data()
    data[field] = data[field].to(t.int64)

    result = validate_shard_data(data)

    assert not result.valid
    assert any(f"{field} dtype" in v for v in result.violations), result.violations


@pytest.mark.parametrize("field", ["segment_char_starts", "segment_char_ends"])
def test_validate_shard_data_requires_char_boundary_fields(field: str) -> None:
    """Base shards must carry character boundaries for overlay alignment."""
    data = make_valid_shard_data()
    data.pop(field)

    result = validate_shard_data(data)

    assert not result.valid
    assert any(f"Missing required field: {field}" == v for v in result.violations)


def test_validate_shard_data_accepts_canonical_dtypes() -> None:
    """The canonical dtypes (extraction-side) must continue to pass."""
    data = make_valid_shard_data()

    result = validate_shard_data(data)

    assert result.valid, result.violations


# ---------------------------------------------------------------------------
# P1-2 — validate_shard must enforce context_independent_features bool
# ---------------------------------------------------------------------------


def test_validate_shard_rejects_non_bool_context_independent_features(monkeypatch) -> None:
    """validate_shard MUST reject non-bool context_independent_features (slice-3 P1-2).

    Defense-in-depth: in production, ``feature_io.load_features`` coerces
    ``context_independent_features`` back to ``bool`` via
    ``_parse_metadata_value`` (so the asymmetry is unreachable through the
    canonical I/O path). However, the parity check guards against future
    regressions in the loader OR in any non-canonical caller. This test
    pins the defensive contract by stubbing the loader inside
    ``shard_validation`` to return a dict whose ``context_independent_features``
    is a string — the file-based validator must reject it.
    """
    fake_data = make_valid_shard_data()
    fake_data["context_independent_features"] = "yes"  # type: ignore[assignment]

    import steering.shard_validation as sv

    monkeypatch.setattr(sv, "load_features", lambda _path: fake_data)
    result = validate_shard("/dev/null/ignored.safetensors.zst")

    assert not result.valid
    assert any("context_independent_features" in v for v in result.violations), result.violations


def test_validate_shard_data_rejects_non_bool_context_independent_features() -> None:
    """validate_shard_data continues to reject non-bool (regression check)."""
    data = make_valid_shard_data()
    data["context_independent_features"] = 1  # int, not bool

    result = validate_shard_data(data)

    assert not result.valid
    assert any("context_independent_features" in v for v in result.violations), result.violations


def test_validate_shard_accepts_bool_context_independent_features(tmp_path) -> None:
    """A valid shard with bool context_independent_features must pass.

    Round-tripped through ``save_features`` → ``load_features`` so the
    metadata is parsed canonically (``_parse_metadata_value`` returns
    ``True``).
    """
    data = make_valid_shard_data()
    data["context_independent_features"] = True

    raw_path = save_raw_shard(tmp_path, "ci_bool", data)
    result = validate_shard(raw_path)

    assert result.valid, result.violations


# ---------------------------------------------------------------------------
# Symmetry — both validators agree on the same shard
# ---------------------------------------------------------------------------


def test_both_validators_agree_on_valid_shard(tmp_path) -> None:
    """Round-trip a valid shard; both validators should report valid=True."""
    path, data = save_valid_shard(tmp_path)

    in_memory = validate_shard_data(data)
    on_disk = validate_shard(path)

    assert in_memory.valid, in_memory.violations
    assert on_disk.valid, on_disk.violations


def test_both_validators_reject_non_bool_context_independent_features(monkeypatch) -> None:
    """Symmetric rejection of non-bool context_independent_features.

    The in-memory path naturally surfaces a non-bool. The file-based path
    is exercised via a loader stub (production ``load_features`` always
    coerces back to ``bool``; see the defense-in-depth note on
    ``test_validate_shard_rejects_non_bool_context_independent_features``).
    """
    data = make_valid_shard_data()
    data["context_independent_features"] = "true"

    in_memory_violations = validate_shard_data(data).violations

    import steering.shard_validation as sv

    monkeypatch.setattr(sv, "load_features", lambda _path: data)
    on_disk_violations = validate_shard("/dev/null/ignored.safetensors.zst").violations

    in_memory_msg = next(
        (v for v in in_memory_violations if "context_independent_features" in v), None
    )
    on_disk_msg = next((v for v in on_disk_violations if "context_independent_features" in v), None)

    assert in_memory_msg is not None, in_memory_violations
    assert on_disk_msg is not None, on_disk_violations
    # Both should reject; messages share the canonical phrasing.
    assert "must be bool" in in_memory_msg
    assert "must be bool" in on_disk_msg
