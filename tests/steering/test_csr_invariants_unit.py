"""Unit tests for assert_csr_invariants (slice-3 map §7.2 unit gap)."""

from __future__ import annotations

import pytest
import torch as t

from steering.shard_validation import assert_csr_invariants


def test_accepts_valid_segments() -> None:
    ids = [t.tensor([0, 5, 9], dtype=t.int32), t.tensor([1, 4, 7], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.5, 0.9], dtype=t.float32), t.tensor([0.2, 0.4, 0.7], dtype=t.float32)]
    # No assertion error.
    assert_csr_invariants(ids, mags)


def test_rejects_length_mismatch() -> None:
    ids = [t.tensor([0, 1], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.5], dtype=t.float32), t.tensor([0.2], dtype=t.float32)]

    with pytest.raises(AssertionError, match="Length mismatch"):
        assert_csr_invariants(ids, mags)


def test_rejects_wrong_id_dtype() -> None:
    ids = [t.tensor([0, 1], dtype=t.int64)]
    mags = [t.tensor([0.1, 0.5], dtype=t.float32)]

    with pytest.raises(AssertionError, match=r"ids dtype torch.int64, expected int32"):
        assert_csr_invariants(ids, mags)


def test_rejects_wrong_mag_dtype() -> None:
    ids = [t.tensor([0, 1], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.5], dtype=t.float64)]

    with pytest.raises(AssertionError, match=r"mags dtype torch.float64, expected float32"):
        assert_csr_invariants(ids, mags)


def test_rejects_per_segment_length_mismatch() -> None:
    ids = [t.tensor([0, 1, 2], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.5], dtype=t.float32)]

    with pytest.raises(AssertionError, match=r"Segment 0: len mismatch"):
        assert_csr_invariants(ids, mags)


def test_rejects_non_positive_magnitude() -> None:
    ids = [t.tensor([0, 1], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.0], dtype=t.float32)]

    with pytest.raises(AssertionError, match=r"non-positive magnitudes"):
        assert_csr_invariants(ids, mags)


def test_rejects_negative_magnitude() -> None:
    ids = [t.tensor([0, 1], dtype=t.int32)]
    mags = [t.tensor([0.1, -0.2], dtype=t.float32)]

    with pytest.raises(AssertionError, match=r"non-positive magnitudes"):
        assert_csr_invariants(ids, mags)


def test_rejects_descending_feature_ids() -> None:
    ids = [t.tensor([5, 3, 1], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.2, 0.3], dtype=t.float32)]

    with pytest.raises(AssertionError, match=r"Segment 0: feature IDs not sorted ascending"):
        assert_csr_invariants(ids, mags)


def test_accepts_equal_consecutive_feature_ids() -> None:
    """Strictly ascending is NOT required; equal consecutive ids OK."""
    ids = [t.tensor([0, 1, 1, 2], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32)]

    # No assertion error — ids[i] <= ids[i+1] passes.
    assert_csr_invariants(ids, mags)


def test_rejects_out_of_range_feature_id() -> None:
    ids = [t.tensor([0, 1, 100], dtype=t.int32)]
    mags = [t.tensor([0.1, 0.2, 0.3], dtype=t.float32)]

    with pytest.raises(AssertionError, match=r"feature ID 100 >= n_features 50"):
        assert_csr_invariants(ids, mags, n_features=50)


def test_rejects_negative_feature_id() -> None:
    ids = [t.tensor([-1, 0, 1], dtype=t.int32)]
    # Use positive mags so we don't trigger the magnitude check first.
    mags = [t.tensor([0.1, 0.2, 0.3], dtype=t.float32)]

    # When ids start with -1, the IDs-ascending check must NOT fire (since
    # they're already ascending), and the negative-id check must fire.
    with pytest.raises(AssertionError, match=r"negative feature ID"):
        assert_csr_invariants(ids, mags, n_features=10)


def test_rejects_empty_segments() -> None:
    with pytest.raises(AssertionError, match="empty|zero features"):
        assert_csr_invariants(
            [t.tensor([], dtype=t.int32), t.tensor([1], dtype=t.int32)],
            [t.tensor([], dtype=t.float32), t.tensor([0.5], dtype=t.float32)],
        )


def test_first_violator_segment_is_named_in_message() -> None:
    """Mixed valid/invalid; the violating segment index appears in the message."""
    ids = [
        t.tensor([0, 1], dtype=t.int32),
        t.tensor([5, 3], dtype=t.int32),  # descending — should fail at segment 1
    ]
    mags = [
        t.tensor([0.1, 0.2], dtype=t.float32),
        t.tensor([0.1, 0.2], dtype=t.float32),
    ]

    with pytest.raises(AssertionError, match=r"Segment 1: feature IDs not sorted ascending"):
        assert_csr_invariants(ids, mags)
