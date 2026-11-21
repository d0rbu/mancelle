"""Slice 07 — S-07-11 P0: feature_selection_hash must distinguish delta configs.

Manifest §9 Q7 + map §6 S-07-11.

Two pipelines that differ only in feature_delta.mode (prev_segment vs cum_max)
or in neg_delta_ratio MUST produce different canonical hashes — otherwise the
eval-context-cache identity collides and runs read each other's cached
artifacts.

These tests also lock in the backwards-compatibility property: when no
feature_delta is provided (None / omitted), the hash must MATCH the pre-fix
canonical value, so existing non-delta caches stay valid.

Seed source: deterministic — these are pure-hash tests with no RNG.
"""

from __future__ import annotations

import pytest

from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)

_POS = {
    "threshold_type": "feature_percentile_segment_topk",
    "threshold": 2048.9,
}


def _base() -> dict:
    return {
        "positive_selection": _POS,
        "negative_selection": None,
        "num_buckets": 10,
        "stats_source_sha256": "a" * 64,
    }


# Pin: this is the on-disk hash from the pre-fix codebase. It MUST stay equal
# after the fix, because we only add a new optional kwarg with a None default
# that maps onto the same delta_config dict the wrapper used to emit.
PRE_FIX_NO_DELTA_HASH = (
    "f8bfb3a1e8c58a47f91bca43938359842186d911a3d9d9371353d95907edea6e"
)


def test_default_no_delta_hash_unchanged():
    """REGRESSION GUARD — non-delta caches survive the Q7 fix.

    The canonical hash for a config that omits feature_delta must equal the
    pre-fix value to preserve existing on-disk caches.
    """
    h = compute_canonical_feature_selection_hash(**_base())
    assert h == PRE_FIX_NO_DELTA_HASH, (
        f"non-delta hash drifted: got {h}, expected {PRE_FIX_NO_DELTA_HASH}. "
        "This breaks backwards compatibility for ALL existing eval caches "
        "that were built without a feature_delta block."
    )


def test_delta_none_explicit_matches_omitted():
    """Passing feature_delta=None must equal omitting feature_delta entirely."""
    h_omitted = compute_canonical_feature_selection_hash(**_base())
    h_explicit_none = compute_canonical_feature_selection_hash(
        **_base(), feature_delta=None,
    )
    assert h_omitted == h_explicit_none


def test_delta_mode_changes_hash():
    """S-07-11 P0 — prev_segment vs cum_max MUST produce different hashes.

    Pre-fix bug: same hash, leading to silent eval-cache collision.
    """
    h_prev = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "prev_segment", "neg_delta_ratio": 1.0},
    )
    h_cummax = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "cum_max", "neg_delta_ratio": 1.0},
    )
    assert h_prev != h_cummax, (
        "prev_segment and cum_max yielded the same hash — eval cache identity "
        "collision (S-07-11 P0)"
    )


def test_delta_neg_ratio_changes_hash():
    """S-07-11 P0 — neg_delta_ratio MUST be hashed."""
    h_full = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "cum_max", "neg_delta_ratio": 1.0},
    )
    h_half = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "cum_max", "neg_delta_ratio": 0.5},
    )
    assert h_full != h_half


def test_delta_dict_insertion_order_irrelevant():
    """Hash must remain order-stable on dict iteration order."""
    h_a = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "prev_segment", "neg_delta_ratio": 0.7},
    )
    h_b = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"neg_delta_ratio": 0.7, "mode": "prev_segment"},
    )
    assert h_a == h_b


def test_delta_distinct_from_no_delta():
    """Adding ANY non-None feature_delta block must shift the hash."""
    h_none = compute_canonical_feature_selection_hash(**_base())
    h_with = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "prev_segment", "neg_delta_ratio": 1.0},
    )
    assert h_none != h_with


@pytest.mark.parametrize(
    "mode_a, mode_b",
    [
        ("prev_segment", "cum_max"),
        ("cum_max", "prev_segment"),
    ],
)
def test_modes_pairwise_distinct(mode_a: str, mode_b: str):
    """Both delta modes must hash distinctly from each other."""
    h_a = compute_canonical_feature_selection_hash(
        **_base(), feature_delta={"mode": mode_a, "neg_delta_ratio": 1.0},
    )
    h_b = compute_canonical_feature_selection_hash(
        **_base(), feature_delta={"mode": mode_b, "neg_delta_ratio": 1.0},
    )
    assert h_a != h_b


def test_returned_hash_is_hex_64_with_delta():
    h = compute_canonical_feature_selection_hash(
        **_base(),
        feature_delta={"mode": "cum_max", "neg_delta_ratio": 0.25},
    )
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


class TestSpliceCounterfactualHash:
    def test_new_mode_changes_hash(self) -> None:
        base = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=None,
            num_buckets=10,
            stats_source_sha256="s" * 64,
            feature_delta={"mode": "prev_segment", "neg_delta_ratio": 1.0},
        )
        cf = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=None,
            num_buckets=10,
            stats_source_sha256="s" * 64,
            feature_delta={"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
        )
        assert base != cf

    def test_absent_delta_unchanged_by_new_mode_existing(self) -> None:
        a = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=None,
            num_buckets=10,
            stats_source_sha256="s" * 64,
        )
        b = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=None,
            num_buckets=10,
            stats_source_sha256="s" * 64,
            feature_delta=None,
        )
        assert a == b

    def test_neg_delta_ratio_still_sensitive_under_new_mode(self) -> None:
        def hash_with_ratio(r: float) -> str:
            return compute_canonical_feature_selection_hash(
                positive_selection=_POS,
                negative_selection=None,
                num_buckets=10,
                stats_source_sha256="s" * 64,
                feature_delta={"mode": "splice_counterfactual", "neg_delta_ratio": r},
            )

        assert hash_with_ratio(0.1) != hash_with_ratio(0.9)
