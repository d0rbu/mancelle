"""Unit tests for the explicit missing-percentile policy (PR #1311 / #1316).

The pre-#1311 behaviour was "die on whichever uncovered feature id happened to
be looked up first, minutes into setup". These tests pin the replacement: an
up-front coverage bitmap that reports the COMPLETE uncovered population, an
explicit two-valued policy, and a resume guard that stops a pilot's numbers
being continued as production numbers.
"""

from __future__ import annotations

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger

from steering.conditional.missing_percentile import (
    LOWEST_MAGNITUDE_BUCKET,
    LOWEST_NORMALIZED_RANK,
    LOWEST_PERCENTILE_RANK,
    MISSING_PERCENTILE_POLICY_META_KEY,
    MissingPercentileError,
    MissingPercentilePolicy,
    MissingPercentileResolver,
    assert_resume_missing_percentile_policy_matches,
    check_percentile_coverage,
    combined_coverage,
    enforce_percentile_coverage,
    parse_missing_percentile_policy,
    percentile_table_coverage,
)
from steering.conditional.percentile_table import magnitude_to_percentile


def _table(n_features: int, covered: set[int]) -> t.Tensor:
    """A [n, 99] table whose rows are monotone for ``covered`` and zero elsewhere."""
    table = t.zeros(n_features, 99, dtype=t.float64)
    row = t.arange(1, 100, dtype=t.float64)
    for fid in covered:
        table[fid] = row
    return table


# ---------------------------------------------------------------------------
# The bitmap must agree EXACTLY with the primitive it is an early guard for
# ---------------------------------------------------------------------------


class TestCoveragePredicateMatchesThePrimitive:
    def test_bitmap_and_primitive_agree_id_for_id(self):
        """Non-vacuity: the bitmap clearing an id must mean the primitive accepts it.

        If the two predicates ever drift, the startup check would wave through
        ids the primitive then rejects mid-run — the exact failure this module
        exists to remove.
        """
        n = 12
        covered = {0, 3, 4, 9, 11}
        table = _table(n, covered)
        bitmap = percentile_table_coverage(table)
        assert bitmap.tolist() == [fid in covered for fid in range(n)]
        for fid in range(n):
            ids = t.tensor([fid], dtype=t.int64)
            mags = t.tensor([5.0])
            if bitmap[fid]:
                assert magnitude_to_percentile(ids, mags, table).numel() == 1
            else:
                with pytest.raises(ValueError, match=r"missing for 1 distinct"):
                    magnitude_to_percentile(ids, mags, table)

    def test_rejects_a_table_of_the_wrong_width(self):
        with pytest.raises(ValueError, match=r"requires a \[n_features, 99\] table"):
            percentile_table_coverage(t.zeros(4, 50, dtype=t.float64))


# ---------------------------------------------------------------------------
# Policy parsing
# ---------------------------------------------------------------------------


class TestParsePolicy:
    def test_accepts_both_values_and_is_idempotent(self):
        assert (
            parse_missing_percentile_policy("error") is MissingPercentilePolicy.ERROR
        )
        assert (
            parse_missing_percentile_policy(" LOWEST ") is MissingPercentilePolicy.LOWEST
        )
        assert (
            parse_missing_percentile_policy(MissingPercentilePolicy.LOWEST)
            is MissingPercentilePolicy.LOWEST
        )

    def test_unknown_value_names_the_accepted_set(self):
        with pytest.raises(ValueError, match=r"Unknown missing_percentile_policy 'skip'"):
            parse_missing_percentile_policy("skip")

    def test_non_string_is_rejected_separately(self):
        with pytest.raises(ValueError, match=r"must be a string, got 3"):
            parse_missing_percentile_policy(3)


# ---------------------------------------------------------------------------
# The report: complete count, not the first few hits
# ---------------------------------------------------------------------------


class TestCoverageReport:
    def test_reports_the_complete_distinct_uncovered_count(self):
        table = _table(40, set(range(20)))
        # 25 distinct required ids, 5 of them covered, each repeated.
        required = t.tensor(list(range(15, 40)) * 3, dtype=t.int64)
        report = check_percentile_coverage(
            percentile_table=table,
            required_ids=required,
            table_name="fp_percentile_table",
            required_set_name="raw pre-selection segment feature ids",
        )
        assert report.n_required == 25
        assert report.n_uncovered == 20
        assert report.examples == tuple(range(20, 30))
        assert "20 of the 25 distinct feature ids" in report.describe()

    def test_clean_table_reports_zero(self):
        table = _table(8, set(range(8)))
        report = check_percentile_coverage(
            percentile_table=table,
            required_ids=t.tensor([1, 2, 3]),
            table_name="fp_percentile_table",
            required_set_name="raw ids",
        )
        assert report.n_uncovered == 0
        assert "covers all 3 ids" in report.describe()

    def test_id_beyond_the_table_is_a_width_mismatch_not_a_coverage_hole(self):
        table = _table(8, set(range(8)))
        with pytest.raises(MissingPercentileError, match=r"outside .*8 rows"):
            check_percentile_coverage(
                percentile_table=table,
                required_ids=t.tensor([1, 9]),
                table_name="fp_percentile_table",
                required_set_name="raw ids",
            )


class TestEnforcePolicy:
    def test_error_policy_aborts_with_every_failing_table(self):
        reports = [
            check_percentile_coverage(
                percentile_table=_table(10, set(range(4))),
                required_ids=t.arange(10),
                table_name="fp_percentile_table",
                required_set_name="raw pre-selection segment feature ids",
            ),
            check_percentile_coverage(
                percentile_table=_table(10, set(range(8))),
                required_ids=t.arange(10),
                table_name="sampling_bundle.percentile_table",
                required_set_name="post-feature-mask selection candidates",
            ),
        ]
        with pytest.raises(
            MissingPercentileError,
            match=(
                r"fp_percentile_table is missing percentile statistics for 6 of "
                r"the 10 distinct.*sampling_bundle\.percentile_table is missing "
                r"percentile statistics for 2 of the 10 distinct"
            ),
        ):
            enforce_percentile_coverage(reports, MissingPercentilePolicy.ERROR)

    def test_lowest_policy_proceeds_and_says_it_is_not_comparable(self):
        """Proceeding silently would be the whole bug; the warning is the contract."""
        reports = [
            check_percentile_coverage(
                percentile_table=_table(10, set(range(4))),
                required_ids=t.arange(10),
                table_name="fp_percentile_table",
                required_set_name="raw ids",
            )
        ]
        emitted: list[str] = []
        sink_id = logger.add(lambda message: emitted.append(str(message)), level="WARNING")
        try:
            enforce_percentile_coverage(reports, MissingPercentilePolicy.LOWEST)
        finally:
            logger.remove(sink_id)
        joined = "\n".join(emitted)
        assert "missing percentile statistics for 6 of the 10" in joined, joined
        assert "NOT comparable" in joined, joined
        assert "rank 0.5" in joined and "0.005" in joined and "bucket 0" in joined, joined

    def test_clean_coverage_never_raises_under_error(self):
        reports = [
            check_percentile_coverage(
                percentile_table=_table(10, set(range(10))),
                required_ids=t.arange(10),
                table_name="fp_percentile_table",
                required_set_name="raw ids",
            )
        ]
        enforce_percentile_coverage(reports, MissingPercentilePolicy.ERROR)


# ---------------------------------------------------------------------------
# The runtime resolver — one contract for every consumer
# ---------------------------------------------------------------------------


class TestResolverErrorPolicy:
    """``error`` substitutes nothing: the primitive is still the guard."""

    def test_ranks_delegates_straight_to_the_primitive(self):
        table = _table(6, {0, 1})
        resolver = MissingPercentileResolver.error()
        got = resolver.ranks(t.tensor([0, 1]), t.tensor([10.0, 90.0]), table)
        expected = magnitude_to_percentile(t.tensor([0, 1]), t.tensor([10.0, 90.0]), table)
        assert t.equal(got, expected)

    def test_uncovered_id_still_raises(self):
        table = _table(6, {0, 1})
        resolver = MissingPercentileResolver.error()
        with pytest.raises(ValueError, match=r"missing for 1 distinct feature ID"):
            resolver.ranks(t.tensor([0, 5]), t.tensor([10.0, 10.0]), table)

    def test_pins_are_identity(self):
        resolver = MissingPercentileResolver.error()
        fids = t.tensor([0, 5])
        buckets = t.tensor([3, 3])
        pcts = t.tensor([0.9, 0.9])
        thresholds = t.zeros(6, dtype=t.float64)
        assert resolver.pin_buckets(fids, buckets) is buckets
        assert resolver.pin_normalized(fids, pcts) is pcts
        assert resolver.mask_precomputed_thresholds(thresholds) is thresholds


class TestResolverLowestPolicy:
    def _resolver(self, n: int, covered: set[int]) -> MissingPercentileResolver:
        bitmap = t.zeros(n, dtype=t.bool)
        for fid in covered:
            bitmap[fid] = True
        return MissingPercentileResolver(
            policy=MissingPercentilePolicy.LOWEST, covered=bitmap
        )

    def test_requires_the_bitmap(self):
        with pytest.raises(ValueError, match=r"policy=lowest\) requires the coverage bitmap"):
            MissingPercentileResolver(policy=MissingPercentilePolicy.LOWEST)

    def test_rejects_a_non_bool_bitmap(self):
        with pytest.raises(ValueError, match=r"covered must be a bool bitmap"):
            MissingPercentileResolver(
                policy=MissingPercentilePolicy.LOWEST,
                covered=t.ones(4, dtype=t.int64),
            )

    def test_uncovered_rank_is_the_floor_and_covered_is_untouched(self):
        table = _table(6, {0, 1})
        resolver = self._resolver(6, {0, 1})
        fids = t.tensor([0, 5, 1])
        mags = t.tensor([10.0, 999.0, 90.0])
        got = resolver.ranks(fids, mags, table)
        assert got[1].item() == LOWEST_PERCENTILE_RANK
        covered_only = magnitude_to_percentile(
            t.tensor([0, 1]), t.tensor([10.0, 90.0]), table
        )
        assert t.equal(got[[0, 2]], covered_only)

    def test_normalized_target_is_the_documented_0_005(self):
        table = _table(6, {0})
        resolver = self._resolver(6, {0})
        ranks = resolver.ranks(t.tensor([4]), t.tensor([7.0]), table)
        normalized = (ranks / 100.0).clamp(0.0, 1.0)
        assert normalized.item() == pytest.approx(LOWEST_NORMALIZED_RANK)

    def test_pin_normalized_replaces_only_uncovered_entries(self):
        resolver = self._resolver(6, {0, 1})
        fids = t.tensor([0, 4, 1])
        pcts = t.tensor([0.7, 0.99, 0.2])
        pinned = resolver.pin_normalized(fids, pcts)
        assert pinned.tolist() == pytest.approx([0.7, LOWEST_NORMALIZED_RANK, 0.2])

    def test_pin_buckets_forces_bucket_zero(self):
        resolver = self._resolver(6, {0, 1})
        fids = t.tensor([0, 4, 1])
        buckets = t.tensor([3, 7, 5])
        assert resolver.pin_buckets(fids, buckets).tolist() == [
            3,
            LOWEST_MAGNITUDE_BUCKET,
            5,
        ]

    def test_thresholds_for_uncovered_features_become_infinite(self):
        """A 0.0 threshold ADMITS every positive magnitude; the floor must not."""
        resolver = self._resolver(6, {0, 1})
        thresholds = t.zeros(6, dtype=t.float64)
        masked = resolver.mask_precomputed_thresholds(thresholds)
        assert masked[0].item() == 0.0
        assert masked[1].item() == 0.0
        assert t.isinf(masked[2:]).all()

    def test_threshold_width_mismatch_is_loud(self):
        resolver = self._resolver(6, {0})
        with pytest.raises(ValueError, match=r"4 thresholds for a 6-row coverage bitmap"):
            resolver.mask_precomputed_thresholds(t.zeros(4, dtype=t.float64))

    def test_all_covered_takes_the_untouched_path(self):
        table = _table(6, set(range(6)))
        resolver = self._resolver(6, set(range(6)))
        fids = t.tensor([0, 3, 5])
        mags = t.tensor([2.0, 50.0, 98.0])
        assert t.equal(
            resolver.ranks(fids, mags, table),
            magnitude_to_percentile(fids, mags, table),
        )


class TestCombinedCoverage:
    def test_is_the_conjunction(self):
        a = t.tensor([True, True, False, False])
        b = t.tensor([True, False, True, False])
        assert combined_coverage([a, b]).tolist() == [True, False, False, False]

    def test_does_not_mutate_its_inputs(self):
        a = t.tensor([True, True])
        b = t.tensor([True, False])
        combined_coverage([a, b])
        assert a.tolist() == [True, True]

    def test_requires_at_least_one_bitmap(self):
        with pytest.raises(ValueError, match=r"requires at least one coverage bitmap"):
            combined_coverage([])


# ---------------------------------------------------------------------------
# Resume guard
# ---------------------------------------------------------------------------


class TestResumeGuard:
    def test_matching_policy_passes(self):
        assert_resume_missing_percentile_policy_matches(
            MissingPercentilePolicy.LOWEST,
            {MISSING_PERCENTILE_POLICY_META_KEY: "lowest"},
            checkpoint_dir="/ckpt",
        )

    def test_historical_checkpoint_without_the_key_is_error(self):
        assert_resume_missing_percentile_policy_matches(
            MissingPercentilePolicy.ERROR, {"step": 3}, checkpoint_dir="/ckpt"
        )
        with pytest.raises(
            MissingPercentileError,
            match=r"checkpoint records 'error', configured 'lowest'",
        ):
            assert_resume_missing_percentile_policy_matches(
                MissingPercentilePolicy.LOWEST, {"step": 3}, checkpoint_dir="/ckpt"
            )

    def test_pilot_checkpoint_cannot_be_resumed_as_production(self):
        with pytest.raises(
            MissingPercentileError,
            match=r"checkpoint records 'lowest', configured 'error'",
        ):
            assert_resume_missing_percentile_policy_matches(
                MissingPercentilePolicy.ERROR,
                {MISSING_PERCENTILE_POLICY_META_KEY: "lowest"},
                checkpoint_dir="/ckpt",
            )

    def test_none_meta_is_treated_as_error(self):
        assert_resume_missing_percentile_policy_matches(
            MissingPercentilePolicy.ERROR, None, checkpoint_dir="/ckpt"
        )

    def test_malformed_recorded_policy_is_rejected_by_the_parser(self):
        with pytest.raises(ValueError, match=r"Unknown missing_percentile_policy 'wat'"):
            assert_resume_missing_percentile_policy_matches(
                MissingPercentilePolicy.ERROR,
                {MISSING_PERCENTILE_POLICY_META_KEY: "wat"},
                checkpoint_dir="/ckpt",
            )

    def test_configured_policy_argument_is_beartyped(self):
        with pytest.raises(BeartypeCallHintParamViolation, match=r"configured_policy"):
            assert_resume_missing_percentile_policy_matches(
                "error",  # ty: ignore[invalid-argument-type]
                None,
                checkpoint_dir="/ckpt",
            )
