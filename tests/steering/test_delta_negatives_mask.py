"""Tests for feature_mask integration in sample_delta_negatives.

Mirrors the call setup from tests/steering/test_delta.py
(TestSampleDeltaNegatives) — specifically the _make_ref_mask helper and
the known-good kwarg-based call pattern used throughout that class.

Three tests:
- blacklist: a blacklisted feature never appears in the returned negatives.
- whitelist universe: every returned negative is inside the whitelist.
- no-mask bit-identical (I5): feature_mask=None is byte-identical to the
  baseline call with no feature_mask argument at all.
"""

from __future__ import annotations

import torch as t

from steering.conditional.delta import sample_delta_negatives
from steering.conditional.feature_mask import FeatureMask

# ---------------------------------------------------------------------------
# Helpers — mirrored verbatim from test_delta.py::_make_ref_mask
# ---------------------------------------------------------------------------


def _make_ref_mask(fids: set[int], n: int) -> t.Tensor:
    """Helper: build a ref_mask bool tensor from a set of feature IDs."""
    mask = t.zeros(n, dtype=t.bool)
    for fid in fids:
        mask[fid] = True
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSampleDeltaNegativesMask:
    """feature_mask enforcement in sample_delta_negatives."""

    def test_delta_negatives_respect_blacklist(self) -> None:
        """Blacklisted feature must never appear in the sampled negatives.

        Setup mirrors test_delta.py::TestSampleDeltaNegatives.test_all_from_deltas_not_enough —
        5 negatives requested, 2 available in delta pool, 3 from uniform pool.
        Feature 150 sits in the uniform pool (not excluded by ref_mask) and
        would be sampled normally.  After blacklisting it, it must never appear
        across multiple seeds.
        """
        # Blacklist a feature that would normally be sampleable from the uniform pool
        blacklisted_id = 150
        mask = FeatureMask("blacklist", (blacklisted_id,))

        ndf = t.tensor([100, 101], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3], dtype=t.float32)
        # n_sae_features large enough that feature 150 is well within universe
        n_sae = 300

        for seed in range(30):
            rng = t.Generator().manual_seed(seed)
            result = sample_delta_negatives(
                k=5, neg_delta_ratio=1.0,
                neg_delta_fids=ndf, neg_delta_weights=ndw,
                raw_feature_ids=t.tensor([0, 1], dtype=t.int64),
                ref_mask=_make_ref_mask({0, 1, 100, 101}, n_sae),
                selected_positive_ids=t.tensor([0], dtype=t.int64),
                n_sae_features=n_sae,
                rng=rng,
                feature_mask=mask,
            )
            assert blacklisted_id not in result.tolist(), (
                f"Blacklisted feature {blacklisted_id} appeared in negatives "
                f"at seed {seed}: {result.tolist()}"
            )

    def test_delta_negatives_whitelist_universe(self) -> None:
        """Every returned negative must be inside the whitelist set.

        Setup mirrors test_delta.py::TestSampleDeltaNegatives.test_partial_ratio —
        uses a partial neg_delta_ratio so both delta and uniform pools are
        exercised, verifying both paths respect the whitelist.

        The whitelist includes features 100-139 (delta pool candidates) plus a
        selection of higher IDs for the uniform pool.  Any ID outside the
        whitelist appearing in the result is a violation.
        """
        whitelist_ids = tuple(range(100, 140)) + tuple(range(200, 230))
        mask = FeatureMask("whitelist", whitelist_ids)

        fids_list = list(range(100, 140))
        ndf = t.tensor(fids_list, dtype=t.int64)
        ndw = t.full((40,), 0.5, dtype=t.float32)
        n_sae = 500

        for seed in range(20):
            rng = t.Generator().manual_seed(seed)
            result = sample_delta_negatives(
                k=16, neg_delta_ratio=0.5,
                neg_delta_fids=ndf, neg_delta_weights=ndw,
                raw_feature_ids=t.arange(10, dtype=t.int64),
                ref_mask=_make_ref_mask(set(range(10)) | set(range(100, 140)), n_sae),
                selected_positive_ids=t.tensor([0], dtype=t.int64),
                n_sae_features=n_sae,
                rng=rng,
                feature_mask=mask,
            )
            result_set = set(result.tolist())
            allowed = set(whitelist_ids)
            # Also exclude positives (per existing semantics — the function
            # itself handles that; we assert the whitelist universe constraint).
            bad = result_set - allowed
            assert not bad, (
                f"Non-whitelisted features in negatives at seed {seed}: {bad}"
            )

    def test_delta_negatives_no_mask_bit_identical(self) -> None:
        """I5: feature_mask=None must produce bit-identical output to no arg at all.

        Two independently seeded generators with identical state are used — one
        call passes no feature_mask argument (the pre-change code path), the
        other passes feature_mask=None explicitly.  The results must be equal
        in both tensor values and dtype.
        """
        ndf = t.tensor([100, 101, 102], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3, 0.8], dtype=t.float32)

        common_kwargs: dict = {
            "k": 2, "neg_delta_ratio": 1.0,
            "neg_delta_fids": ndf, "neg_delta_weights": ndw,
            "raw_feature_ids": t.tensor([0], dtype=t.int64),
            "ref_mask": _make_ref_mask({0, 100, 101, 102}, 200),
            "selected_positive_ids": t.tensor([], dtype=t.int64),
            "n_sae_features": 200,
        }

        for seed in range(20):
            rng_baseline = t.Generator().manual_seed(seed)
            rng_none = t.Generator().manual_seed(seed)

            # Baseline: no feature_mask argument (original call style)
            result_baseline = sample_delta_negatives(**common_kwargs, rng=rng_baseline)
            # I5 path: feature_mask=None explicitly passed
            result_none = sample_delta_negatives(**common_kwargs, rng=rng_none,
                                                 feature_mask=None)

            assert t.equal(result_baseline, result_none), (
                f"seed {seed}: feature_mask=None produced different output than no arg. "
                f"baseline={result_baseline.tolist()}, none={result_none.tolist()}"
            )
