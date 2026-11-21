"""``compute_selection_hash`` determinism and ``ddp_rank`` identity.

The hash is a deterministic SHA-256 of the canonical selection config, with ``ddp_rank``
intentionally part of the canonical input. The precompute path always passes
``ddp_rank=0``; document this contract via tests so a future seeding-rank
change is visible.
"""

from __future__ import annotations

import hashlib
import json

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.packed_batch_io import compute_selection_hash

_SAMPLE_POSITIVE: dict = {
    "type": "ThresholdSelection",
    "threshold_type": "feature_percentile",
    "threshold": 0.95,
}
_SAMPLE_DELTA = None
_SAMPLE_NEGATIVE = None


class TestDeterminism:
    def test_same_inputs_same_hash(self):
        h1 = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=42,
        )
        h2 = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=42,
        )
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_different_seed_different_hash(self):
        h1 = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=42,
        )
        h2 = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=43,
        )
        assert h1 != h2

    def test_different_num_buckets_different_hash(self):
        h_low = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=2,
            seed=42,
        )
        h_high = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=32,
            seed=42,
        )
        assert h_low != h_high


class TestDdpRankIsCanonicalInput:
    """``ddp_rank`` is part of the canonical hash input — different ranks
    produce different hashes. The precompute pipeline (single process) always
    passes ``ddp_rank=0``; this test pins the contract so any future change
    to the seeding rank surfaces.
    """

    def test_default_ddp_rank_zero(self):
        h_default = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=42,
        )
        h_explicit = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=42,
            ddp_rank=0,
        )
        assert h_default == h_explicit, "default ddp_rank must be 0"

    def test_different_ranks_different_hashes(self):
        hashes = {
            rank: compute_selection_hash(
                positive_selection=_SAMPLE_POSITIVE,
                negative_selection=_SAMPLE_NEGATIVE,
                delta_config=_SAMPLE_DELTA,
                num_buckets=16,
                seed=42,
                ddp_rank=rank,
            )
            for rank in [0, 1, 7, 15]
        }
        # All four ranks must produce distinct hashes.
        assert len(set(hashes.values())) == 4


class TestCanonicalJsonForm:
    """The canonical form is ``json.dumps(..., sort_keys=True, separators=(",", ":"))``
    over a fixed key order. Verify the function produces the same hash as a
    hand-computed canonical SHA-256.
    """

    def test_matches_hand_computed_sha256(self):
        canonical = json.dumps(
            {
                "positive_selection": _SAMPLE_POSITIVE,
                "negative_selection": _SAMPLE_NEGATIVE,
                "delta_config": _SAMPLE_DELTA,
                "num_buckets": 16,
                "seed": 42,
                "ddp_rank": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        actual = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=16,
            seed=42,
        )
        assert actual == expected


class TestPropertyHypothesis:
    @settings(derandomize=True, max_examples=200)
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        num_buckets=st.integers(min_value=2, max_value=64),
        ddp_rank=st.integers(min_value=0, max_value=128),
    )
    def test_deterministic_for_given_args(self, seed, num_buckets, ddp_rank):
        h1 = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=num_buckets,
            seed=seed,
            ddp_rank=ddp_rank,
        )
        h2 = compute_selection_hash(
            positive_selection=_SAMPLE_POSITIVE,
            negative_selection=_SAMPLE_NEGATIVE,
            delta_config=_SAMPLE_DELTA,
            num_buckets=num_buckets,
            seed=seed,
            ddp_rank=ddp_rank,
        )
        assert h1 == h2
        assert len(h1) == 64
