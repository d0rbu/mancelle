"""Unit tests for production equivalence kernels and test-only oracles.

(a) NON-VACUOUS arithmetic-pin cases: reuse concrete (ids, mags) fixtures from
    ``test_parity_helpers_unit.py``, re-pointed at ``steering.feature_equivalence``.
(b) Re-export identity check: ``conftest_parity`` must re-export each
    production kernel and test oracle rather than redefining it.
"""

from __future__ import annotations

import pytest
import torch as t

import steering.feature_equivalence as fe
import tests.steering.native._feature_equivalence_oracles as test_oracles
import tests.steering.native._parity_thresholds as thresholds
import tests.steering.native.conftest_parity as cp

# --------------------------------------------------------------------------- #
# (b) Re-export identity / value equality                                      #
# --------------------------------------------------------------------------- #

_MOVED_CONSTANTS = [
    "FEATURE_COUNT_RTOL",
    "JACCARD_THRESHOLD",
    "MAG_RTOL",
    "MAG_ATOL",
    "FP32_NATIVE_VS_TL_MAG_RTOL",
    "FP32_NATIVE_VS_TL_MAG_ATOL",
    "BF16_NATIVE_VS_TL_MAG_RTOL",
    "BF16_NATIVE_VS_TL_MAG_ATOL",
]

_MOVED_CALLABLES = [
    "_seg_jaccard_and_count_rdiff",
    "feature_equivalence_metrics",
]

_TEST_CALLABLES = [
    "_seg_common_mag_max_abs_and_rel",
    "feature_equivalence_passed",
    "feature_magnitude_metrics",
]


@pytest.mark.parametrize("name", _MOVED_CONSTANTS)
def test_constant_value_equal(name: str) -> None:
    """The parity facade must re-export the test-owned threshold values."""
    assert getattr(cp, name) == getattr(thresholds, name), (
        f"{name}: conftest_parity value {getattr(cp, name)!r} != "
        f"test threshold value {getattr(thresholds, name)!r}"
    )


@pytest.mark.parametrize("name", _MOVED_CALLABLES)
def test_callable_is_identical(name: str) -> None:
    """Every moved callable must be the EXACT SAME object (identity, not just
    equality) when imported from both modules — proving conftest_parity
    re-exports rather than re-defines."""
    assert getattr(cp, name) is getattr(fe, name), (
        f"{name}: conftest_parity object is not the same as "
        f"steering.feature_equivalence object (re-definition drift)"
    )


@pytest.mark.parametrize("name", _TEST_CALLABLES)
def test_test_oracle_is_identical(name: str) -> None:
    """The parity conftest re-exports, rather than redefines, test oracles."""
    assert getattr(cp, name) is getattr(test_oracles, name)


# --------------------------------------------------------------------------- #
# (a) NON-VACUOUS arithmetic-pin cases (faithful reproductions of the concrete #
#     fixtures in test_parity_helpers_unit.py, re-pointed at fe.*)            #
# --------------------------------------------------------------------------- #

# --- Structural: jaccard + count-rdiff ---

def test_seg_jaccard_perfect_overlap() -> None:
    """Identical id sets -> jaccard 1.0, count_rdiff 0.0."""
    ids = t.arange(10)
    jac, rdiff = fe._seg_jaccard_and_count_rdiff(ids, ids)
    assert jac == 1.0
    assert rdiff == 0.0


def test_seg_jaccard_partial_overlap() -> None:
    """The _jaccard_about_985 fixture: arange(0,133) vs arange(1,134) ->
    132 shared, union 134 -> jaccard = 132/134 ≈ 0.98507; count rdiff = 0."""
    a_ids = t.arange(0, 133)
    b_ids = t.arange(1, 134)
    jac, rdiff = fe._seg_jaccard_and_count_rdiff(a_ids, b_ids)
    assert abs(jac - 132 / 134) < 1e-9
    assert rdiff == 0.0
    # must be strictly below the default JACCARD_THRESHOLD
    assert jac < thresholds.JACCARD_THRESHOLD


def test_seg_jaccard_count_mismatch() -> None:
    """100 vs 90 -> rdiff = 10/100 = 0.1, jaccard for disjoint = 0/100 if
    no overlap; here arange(100) vs arange(90) -> 90 overlap, union 100,
    jaccard 0.9, rdiff 0.1."""
    a_ids = t.arange(100)
    b_ids = t.arange(90)
    jac, rdiff = fe._seg_jaccard_and_count_rdiff(a_ids, b_ids)
    assert abs(rdiff - 0.1) < 1e-9      # 10/100
    assert abs(jac - 0.9) < 1e-9       # 90/100
    assert rdiff > thresholds.FEATURE_COUNT_RTOL  # exceeds the tight 0.01 gate


# --- Magnitude: max_abs + max_rel ---

def _bf16_envelope_ids_mags() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Exact copy of _bf16_ulp_envelope_pair from test_parity_helpers_unit:
    feat 7 (huge: a=1312 b=1328, |Δ|=16) + feat 42 (small: a=7 b=4, |Δ|=3).
    Common ids identical both sides (jaccard 1.0)."""
    ids = t.tensor([7, 42])
    a_mags = t.tensor([1312.0, 7.0])
    b_mags = t.tensor([1328.0, 4.0])
    return ids, a_mags, ids, b_mags


def test_seg_common_mag_exact_values() -> None:
    """max_abs must be 16.0 (global max, huge feature); max_rel must be
    3.0/4.0=0.75 (the small-|b| feature dominates relative error)."""
    aid, amg, bid, bmg = _bf16_envelope_ids_mags()
    max_abs, max_rel = test_oracles._seg_common_mag_max_abs_and_rel(
        aid, amg, bid, bmg
    )
    assert max_abs == 16.0
    assert abs(max_rel - 3.0 / 4.0) < 1e-6


def test_seg_common_mag_no_common_features() -> None:
    """Disjoint id sets -> no common features -> (0.0, 0.0) vacuously."""
    a_ids = t.tensor([1, 2])
    b_ids = t.tensor([7, 8])
    a_mags = t.tensor([9.0, 9.0])
    b_mags = t.tensor([9.0, 9.0])
    max_abs, max_rel = test_oracles._seg_common_mag_max_abs_and_rel(
        a_ids, a_mags, b_ids, b_mags
    )
    assert max_abs == 0.0
    assert max_rel == 0.0


# --- feature_equivalence_metrics ---

def test_feature_equivalence_metrics_worst_segment() -> None:
    """Multi-segment: seg0 perfect (jac 1.0, rdiff 0); seg1: arange(0,4) vs
    arange(0,2) -> |∩|=2 |∪|=4 -> 0.5; rdiff 2/4=0.5. Reports worst."""
    a0, b0 = t.arange(0, 10), t.arange(0, 10)
    a1, b1 = t.arange(0, 4), t.arange(0, 2)
    min_j, max_rd = fe.feature_equivalence_metrics([a0, a1], [b0, b1])
    assert min_j == 0.5
    assert max_rd == 0.5


def test_feature_equivalence_metrics_matches_jaccard_fixture() -> None:
    """The _jaccard_about_985 analogue: min_jaccard ≈ 132/134, max_rdiff 0."""
    a_ids = t.arange(0, 133)
    b_ids = t.arange(1, 134)
    min_j, max_rd = fe.feature_equivalence_metrics([a_ids], [b_ids])
    assert 0.98 < min_j < thresholds.JACCARD_THRESHOLD
    assert abs(min_j - 132 / 134) < 1e-9
    assert max_rd == 0.0


# --- feature_magnitude_metrics ---

def test_feature_magnitude_metrics_exact_bf16_envelope() -> None:
    """Re-pins the bf16-ULP envelope fixture: max_abs=16, max_rel=3/4."""
    aid, amg, bid, bmg = _bf16_envelope_ids_mags()
    max_abs, max_rel = test_oracles.feature_magnitude_metrics(
        [aid], [amg], [bid], [bmg]
    )
    assert max_abs == 16.0
    assert abs(max_rel - 3.0 / 4.0) < 1e-6
    assert max_abs == pytest.approx(abs(1312.0 - 1328.0))


def test_feature_magnitude_metrics_worst_segment() -> None:
    """Multi-segment: seg0 disjoint (0,0); seg1: common feature id=5,
    |Δ|=20 on |b|=10 -> rel 2.0. Reports the worst."""
    a0_id, b0_id = t.tensor([1, 2]), t.tensor([7, 8])
    a0_mg, b0_mg = t.tensor([9.0, 9.0]), t.tensor([9.0, 9.0])
    a1_id, b1_id = t.tensor([5, 3]), t.tensor([5, 99])
    a1_mg, b1_mg = t.tensor([30.0, 1.0]), t.tensor([10.0, 1.0])
    max_abs, max_rel = test_oracles.feature_magnitude_metrics(
        [a0_id, a1_id], [a0_mg, a1_mg], [b0_id, b1_id], [b0_mg, b1_mg],
    )
    assert max_abs == 20.0
    assert max_rel == pytest.approx(20.0 / 10.0)


# --------------------------------------------------------------------------- #
# feature_equivalence_passed — THE single pass/fail predicate. Direct          #
# arithmetic pins (the asserter-equivalence battery lives in                   #
# test_native_compile_gate_unit.py). Each case nails one short-circuit.        #
# --------------------------------------------------------------------------- #

_TIGHT = {
    "mag_rtol": 1e-2, "mag_atol": 1e-5, "jaccard_threshold": 0.99, "count_rtol": 0.01
}


def test_passed_true_on_exact_match() -> None:
    """Identical structure + magnitudes within tol -> True."""
    ids = [t.tensor([1, 2, 3])]
    a = [t.tensor([1.0, 2.0, 3.0])]
    b = [t.tensor([1.005, 2.0, 3.0])]  # |Δ|=5e-3 < 1e-2*1
    assert test_oracles.feature_equivalence_passed(ids, a, ids, b, **_TIGHT) is True


def test_passed_false_on_segment_count_mismatch_no_raise() -> None:
    """The len-precondition mismatch returns False (fails CLOSED) — it does
    NOT raise (assert_feature_equivalent keeps its own raising precondition)."""
    out = test_oracles.feature_equivalence_passed(
        [t.tensor([1])], [t.tensor([1.0])],
        [t.tensor([1]), t.tensor([2])], [t.tensor([1.0]), t.tensor([2.0])],
        **_TIGHT,
    )
    assert out is False


def test_passed_false_on_count_rdiff_exceeds_rtol() -> None:
    """100 vs 90 -> rdiff 0.1 > 0.01 -> False."""
    assert test_oracles.feature_equivalence_passed(
        [t.arange(100)], [t.ones(100)],
        [t.arange(90)], [t.ones(90)], **_TIGHT,
    ) is False


def test_passed_false_on_low_jaccard() -> None:
    """Equal counts, only 50/150 shared -> jaccard 0.333 < 0.99 -> False."""
    assert test_oracles.feature_equivalence_passed(
        [t.arange(0, 100)], [t.ones(100)],
        [t.arange(50, 150)], [t.ones(100)], **_TIGHT,
    ) is False


def test_passed_false_on_magnitude_outside_tol() -> None:
    """Structure OK, one shared feature |Δ|=0.5 >> tol -> False."""
    a_ids = t.arange(0, 200)
    b_ids = t.arange(1, 201)  # 199 shared, counts equal, jaccard≈0.990
    a_mags = t.full((200,), 1.0)
    b_mags = t.full((200,), 1.0)
    b_mags[(b_ids == 1).nonzero(as_tuple=True)[0]] = 1.5  # shared id=1
    assert test_oracles.feature_equivalence_passed(
        [a_ids], [a_mags], [b_ids], [b_mags], **_TIGHT
    ) is False


def test_passed_bf16_ulp_envelope_fp32_fail_bf16_pass() -> None:
    """The bf16-ULP envelope (huge |Δ|=16 + small |Δ|=3, jaccard 1.0): the
    SAME inputs FAIL under the tight fp32 mag tol but PASS under the
    data-grounded bf16 mag tol — the dtype-aware tol is the discriminator."""
    ids = t.tensor([7, 42])
    a = t.tensor([1312.0, 7.0])
    b = t.tensor([1328.0, 4.0])
    assert test_oracles.feature_equivalence_passed(
        [ids], [a], [ids], [b],
        mag_rtol=thresholds.FP32_NATIVE_VS_TL_MAG_RTOL,
        mag_atol=thresholds.FP32_NATIVE_VS_TL_MAG_ATOL,
        jaccard_threshold=0.99, count_rtol=0.01,
    ) is False
    assert test_oracles.feature_equivalence_passed(
        [ids], [a], [ids], [b],
        mag_rtol=thresholds.BF16_NATIVE_VS_TL_MAG_RTOL,
        mag_atol=thresholds.BF16_NATIVE_VS_TL_MAG_ATOL,
        jaccard_threshold=0.99, count_rtol=0.01,
    ) is True


def test_passed_empty_common_skips_magnitude() -> None:
    """Disjoint ids, equal COUNT, jaccard_threshold=0 -> structural gates pass
    and magnitude is SKIPPED (no common feature) -> True even with grossly
    different magnitudes. Pins the empty-common short-circuit."""
    assert test_oracles.feature_equivalence_passed(
        [t.tensor([1, 2])], [t.tensor([9.0, 9.0])],
        [t.tensor([7, 8])], [t.tensor([1.0, 1.0])],
        mag_rtol=0.0, mag_atol=0.0, jaccard_threshold=0.0, count_rtol=0.01,
    ) is True


def test_passed_jaccard_count_mag_straddles() -> None:
    """±1-unit straddles around each gate, pinning the exact comparison
    operators (jaccard ``<``, count ``>``, mag ``t.allclose``)."""
    # jaccard: union 201 share 199 -> 0.99005; 0.99 INSIDE, 0.991 OUTSIDE.
    aj, bj = [t.arange(0, 200)], [t.arange(1, 201)]
    mj = [t.ones(200)]
    assert test_oracles.feature_equivalence_passed(
        aj, mj, bj, mj, mag_rtol=1e-2, mag_atol=1e-5,
        jaccard_threshold=0.99, count_rtol=0.01,
    ) is True
    assert test_oracles.feature_equivalence_passed(
        aj, mj, bj, mj, mag_rtol=1e-2, mag_atol=1e-5,
        jaccard_threshold=0.991, count_rtol=0.01,
    ) is False
    # count: 100 vs 99 -> rdiff 0.01; ==rtol PASS (<=), <rtol FAIL.
    ac, bc = [t.arange(100)], [t.arange(99)]
    assert test_oracles.feature_equivalence_passed(
        ac, [t.ones(100)], bc, [t.ones(99)], mag_rtol=1e-2, mag_atol=1e-5,
        jaccard_threshold=0.0, count_rtol=0.01,
    ) is True
    assert test_oracles.feature_equivalence_passed(
        ac, [t.ones(100)], bc, [t.ones(99)], mag_rtol=1e-2, mag_atol=1e-5,
        jaccard_threshold=0.0, count_rtol=0.009,
    ) is False
    # mag: shared |Δ|=0.5 on |b|=1, rtol=0 -> atol-only; 0.5001 PASS, 0.4999
    # FAIL.
    am, bm = [t.tensor([1, 2])], [t.tensor([1, 2])]
    assert test_oracles.feature_equivalence_passed(
        am, [t.tensor([1.0, 2.0])], bm, [t.tensor([1.5, 2.0])],
        mag_rtol=0.0, mag_atol=0.5001, jaccard_threshold=0.99, count_rtol=0.01,
    ) is True
    assert test_oracles.feature_equivalence_passed(
        am, [t.tensor([1.0, 2.0])], bm, [t.tensor([1.5, 2.0])],
        mag_rtol=0.0, mag_atol=0.4999, jaccard_threshold=0.99, count_rtol=0.01,
    ) is False
