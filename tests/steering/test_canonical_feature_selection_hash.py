from __future__ import annotations

from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)


def _base():
    return {
        "positive_selection": {"threshold_type": "feature_percentile_segment_topk", "threshold": 2048.9},
        "negative_selection": None,
        "num_buckets": 10,
        "stats_source_sha256": "a" * 64,
    }


def test_same_inputs_same_hash():
    a = compute_canonical_feature_selection_hash(**_base())
    b = compute_canonical_feature_selection_hash(**_base())
    assert a == b


def test_dict_insertion_order_irrelevant():
    a = compute_canonical_feature_selection_hash(
        positive_selection={"threshold": 2048.9, "threshold_type": "feature_percentile_segment_topk"},
        negative_selection=None, num_buckets=10, stats_source_sha256="a" * 64,
    )
    b = compute_canonical_feature_selection_hash(**_base())
    assert a == b


def test_different_threshold_different_hash():
    a = compute_canonical_feature_selection_hash(**_base())
    b = compute_canonical_feature_selection_hash(**{**_base(), "positive_selection": {"threshold_type": "feature_percentile_segment_topk", "threshold": 1024.5}})  # ty: ignore[invalid-argument-type]
    assert a != b


def test_different_num_buckets_different_hash():
    a = compute_canonical_feature_selection_hash(**_base())
    b = compute_canonical_feature_selection_hash(**{**_base(), "num_buckets": 8})  # ty: ignore[invalid-argument-type]
    assert a != b


def test_different_stats_sha_different_hash():
    a = compute_canonical_feature_selection_hash(**_base())
    b = compute_canonical_feature_selection_hash(**{**_base(), "stats_source_sha256": "b" * 64})  # ty: ignore[invalid-argument-type]
    assert a != b


def test_negative_selection_field_changes_hash():
    a = compute_canonical_feature_selection_hash(**_base())
    b = compute_canonical_feature_selection_hash(**{**_base(), "negative_selection": {"distribution": "geometric", "mean": 5.0}})  # ty: ignore[invalid-argument-type]
    assert a != b


def test_returned_hash_is_hex_64():
    h = compute_canonical_feature_selection_hash(**_base())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_canonical_hash_matches_known_snapshot():
    """Pin the on-the-wire hash for a fixed input. If a future refactor
    changes seed=0/ddp_rank=0 to something else, the hash will shift and
    this test catches it. The hash value below was computed with
    seed=0/ddp_rank=0 against the v1 schema."""
    h = compute_canonical_feature_selection_hash(
        positive_selection={"threshold_type": "feature_percentile_segment_topk", "threshold": 2048.9},
        negative_selection=None,
        num_buckets=10,
        stats_source_sha256="a" * 64,
    )
    expected = "f8bfb3a1e8c58a47f91bca43938359842186d911a3d9d9371353d95907edea6e"
    assert h == expected, f"hash drift detected: got {h}"
