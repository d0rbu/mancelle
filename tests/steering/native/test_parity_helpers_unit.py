"""NON-VACUOUS unit tests for the two cross-backend parity assertions in
``tests/steering/native/conftest_parity.py``.

Fast, synthetic tensors only — no model. Each test asserts the SPECIFIC
contract of the branch it covers:

* the pass branch returns ``None`` silently;
* every raise branch raises ``AssertionError`` whose message contains the
  exact diagnostic substring the helper emits (asserted via
  ``pytest.raises(AssertionError, match=...)``).

The strongest oracle covered here (gap H2): ``assert_bit_identical`` must
reject a difference of a SINGLE float32 ULP — proving it enforces
``max_abs == 0.0`` EXACTLY (``torch.equal``), not an approximate tolerance.

Module constants are imported from ``conftest_parity`` (never hardcoded) so
these tests track the helper's real thresholds.
"""

from __future__ import annotations

import pytest
import torch as t

from tests.steering.native.conftest_parity import (
    BF16_NATIVE_VS_TL_MAG_ATOL,
    BF16_NATIVE_VS_TL_MAG_RTOL,
    FEATURE_COUNT_RTOL,
    FP32_NATIVE_VS_TL_MAG_ATOL,
    FP32_NATIVE_VS_TL_MAG_RTOL,
    JACCARD_THRESHOLD,
    MAG_ATOL,
    MAG_RTOL,
    assert_bit_identical,
    assert_feature_equivalent,
    assert_features_nonempty,
    feature_equivalence_metrics,
    feature_magnitude_metrics,
)


# --------------------------------------------------------------------------- #
# assert_bit_identical                                                          #
# --------------------------------------------------------------------------- #
def test_bit_identical_passes_on_exact_match():
    """Identical IDs + magnitudes -> returns None silently (no raise)."""
    ids = [t.tensor([3, 1, 7]), t.tensor([5])]
    mags = [t.tensor([0.5, 0.25, 0.125]), t.tensor([9.0])]
    assert assert_bit_identical(ids, mags, ids, mags, label="X") is None


def test_bit_identical_order_independent_pass():
    """Same multiset of IDs in a DIFFERENT order with co-permuted mags still
    passes — the helper sorts by feature ID before comparing."""
    a_ids = [t.tensor([7, 1, 3])]
    a_mags = [t.tensor([70.0, 10.0, 30.0])]
    b_ids = [t.tensor([1, 3, 7])]
    b_mags = [t.tensor([10.0, 30.0, 70.0])]
    assert assert_bit_identical(a_ids, a_mags, b_ids, b_mags) is None


def test_bit_identical_one_ulp_magnitude_raises():
    """H2 gap: a difference of exactly ONE float32 ULP in a magnitude MUST
    raise — proving max_abs == 0.0 is enforced EXACTLY (torch.equal), not an
    epsilon tolerance. Quote: the raised message must contain
    'magnitude mismatch (max_abs=' and '> 0.0)'."""
    base = t.tensor([1.0, 2.0, 3.0], dtype=t.float32)
    bumped = base.clone()
    # nextafter towards +inf == the very next representable float32 (1 ULP).
    one_ulp_up = t.nextafter(base[1], t.tensor(float("inf")))
    bumped[1] = one_ulp_up
    assert one_ulp_up != base[1], "sanity: nextafter must change the value"
    assert (bumped - base).abs().max().item() > 0.0
    ids = [t.tensor([0, 1, 2])]
    with pytest.raises(
        AssertionError,
        match=r"seg 0: magnitude mismatch \(max_abs=.* > 0\.0\)",
    ):
        assert_bit_identical(ids, [base], ids, [bumped])


def test_bit_identical_tiny_epsilon_magnitude_raises():
    """A sub-ULP-scale tiny epsilon (1e-9) added to one magnitude STILL must
    raise — assert_bit_identical tolerates ZERO deviation. (Distinct from
    the 1-ULP case: this is an arbitrary tiny epsilon, the typical
    'almost-equal' that approximate oracles would wrongly pass.)"""
    base = t.tensor([10.0, 20.0], dtype=t.float32)
    eps = base.clone()
    eps[0] = base[0] + 1e-4  # representable, tiny, but != 0 deviation
    ids = [t.tensor([4, 9])]
    with pytest.raises(
        AssertionError, match=r"seg 0: magnitude mismatch \(max_abs="
    ):
        assert_bit_identical(ids, [base], ids, [eps])


def test_bit_identical_different_feature_ids_raises():
    """Different feature IDs -> the 'feature IDs differ' branch, with the
    set-difference diagnostic."""
    a_ids = [t.tensor([1, 2, 3])]
    b_ids = [t.tensor([1, 2, 4])]
    mags = [t.tensor([1.0, 2.0, 3.0])]
    with pytest.raises(
        AssertionError,
        match=r"seg 0: feature IDs differ — n_a=3, n_b=3",
    ):
        assert_bit_identical(a_ids, mags, b_ids, mags)


def test_bit_identical_different_length_raises():
    """Different per-segment feature COUNT also trips 'feature IDs differ'
    (the sorted-ID equal check fails on length first)."""
    a_ids = [t.tensor([1, 2, 3])]
    a_mags = [t.tensor([1.0, 2.0, 3.0])]
    b_ids = [t.tensor([1, 2])]
    b_mags = [t.tensor([1.0, 2.0])]
    with pytest.raises(
        AssertionError,
        match=r"seg 0: feature IDs differ — n_a=3, n_b=2",
    ):
        assert_bit_identical(a_ids, a_mags, b_ids, b_mags)


def test_bit_identical_segment_count_mismatch_raises():
    """Top-level segment-count guard: differing number of segments."""
    a_ids = [t.tensor([1]), t.tensor([2])]
    a_mags = [t.tensor([1.0]), t.tensor([2.0])]
    b_ids = [t.tensor([1])]
    b_mags = [t.tensor([1.0])]
    with pytest.raises(
        AssertionError,
        match=r"L: segment count mismatch \(2,2\) vs \(1,1\)",
    ):
        assert_bit_identical(a_ids, a_mags, b_ids, b_mags, label="L")


def test_bit_identical_empty_segment_passes():
    """Both segments empty -> max_abs short-circuits to 0.0, equal IDs of
    length 0 -> passes (covers the ``len(amg_s) > 0`` False arm)."""
    empty_id = [t.empty(0, dtype=t.long)]
    empty_mg = [t.empty(0, dtype=t.float32)]
    assert assert_bit_identical(empty_id, empty_mg, empty_id, empty_mg) is None


# --------------------------------------------------------------------------- #
# assert_feature_equivalent                                                     #
# --------------------------------------------------------------------------- #
def test_feature_equivalent_passes_within_all_tolerances():
    """Identical structure + magnitudes within default tol -> None."""
    ids = [t.tensor([1, 2, 3])]
    a_mags = [t.tensor([1.0, 2.0, 3.0])]
    # within default MAG_RTOL=1e-2: 1.0 -> 1.005 is |Δ| 5e-3 < 1e-2*1
    b_mags = [t.tensor([1.005, 2.0, 3.0])]
    assert assert_feature_equivalent(ids, a_mags, ids, b_mags) is None


def test_feature_equivalent_segment_count_mismatch_raises():
    with pytest.raises(
        AssertionError,
        match=r"FE: segment count mismatch \(1,1\) vs \(2,2\)",
    ):
        assert_feature_equivalent(
            [t.tensor([1])], [t.tensor([1.0])],
            [t.tensor([1]), t.tensor([2])],
            [t.tensor([1.0]), t.tensor([2.0])],
            label="FE",
        )


def test_feature_equivalent_count_rdiff_exceeds_rtol_raises():
    """feature-count rel-diff > FEATURE_COUNT_RTOL -> 'feature-count rdiff'.
    100 vs 90 -> rdiff = 10/100 = 0.1 > 0.01."""
    assert FEATURE_COUNT_RTOL == 0.01
    a_ids = [t.arange(100)]
    a_mags = [t.ones(100)]
    b_ids = [t.arange(90)]
    b_mags = [t.ones(90)]
    with pytest.raises(
        AssertionError,
        match=rf"seg 0: feature-count rdiff=0\.1000 > "
              rf"rtol={FEATURE_COUNT_RTOL} \(n_a=100, n_b=90\)",
    ):
        assert_feature_equivalent(a_ids, a_mags, b_ids, b_mags)


def test_feature_equivalent_low_jaccard_raises():
    """count rdiff OK (equal counts) but Jaccard < JACCARD_THRESHOLD ->
    'jaccard' branch. 100 ids each, only 50 shared -> jaccard = 50/150 =
    0.333 < 0.99."""
    assert JACCARD_THRESHOLD == 0.99
    a_ids = [t.arange(0, 100)]
    b_ids = [t.arange(50, 150)]  # same count (100); 50 overlap
    a_mags = [t.ones(100)]
    b_mags = [t.ones(100)]
    with pytest.raises(
        AssertionError,
        match=rf"seg 0: jaccard=0\.3333 < {JACCARD_THRESHOLD}",
    ):
        assert_feature_equivalent(a_ids, a_mags, b_ids, b_mags)


def _almost_full_overlap():
    """200-feature segments sharing 199 ids (jaccard 199/201 ≈ 0.990 ≥ 0.99,
    count rdiff 0) so ONLY the magnitude gate can fail."""
    a_ids = t.arange(0, 200)
    b_ids = t.arange(1, 201)  # 199 shared (1..199), counts equal
    return a_ids, b_ids


def test_feature_equivalent_magnitude_outside_default_tol_raises():
    """Structure OK but a shared-feature magnitude is outside default
    MAG_RTOL/MAG_ATOL -> 'magnitude mismatch' branch."""
    a_ids, b_ids = _almost_full_overlap()
    a_mags = t.full((200,), 1.0)
    b_mags = t.full((200,), 1.0)
    # bump one SHARED feature (id=1, present in both) far beyond
    # MAG_RTOL*|a| + MAG_ATOL  (1e-2*1 + 1e-5 ≈ 0.01001) -> +0.5
    shared_pos_a = (a_ids == 1).nonzero(as_tuple=True)[0]
    shared_pos_b = (b_ids == 1).nonzero(as_tuple=True)[0]
    a_mags[shared_pos_a] = 1.0
    b_mags[shared_pos_b] = 1.5
    with pytest.raises(
        AssertionError, match=r"seg 0: magnitude mismatch max_abs="
    ):
        assert_feature_equivalent([a_ids], [a_mags], [b_ids], [b_mags])


def test_feature_equivalent_loose_mag_kwargs_make_same_case_pass():
    """Asymmetric-oracle contract (the slow Layer-4 native-vs-TL gate):
    the SAME magnitude divergence that fails under default tol PASSES when
    called with a looser ``mag_rtol=`` / ``mag_atol=`` — while count/jaccard
    stay tight (unaffected). Proves the kwargs are honored."""
    a_ids, b_ids = _almost_full_overlap()
    a_mags = t.full((200,), 1.0)
    b_mags = t.full((200,), 1.0)
    shared_pos_a = (a_ids == 1).nonzero(as_tuple=True)[0]
    shared_pos_b = (b_ids == 1).nonzero(as_tuple=True)[0]
    a_mags[shared_pos_a] = 1.0
    b_mags[shared_pos_b] = 1.5  # |Δ|=0.5

    # default tol -> raises (sanity, ties this test to the one above)
    with pytest.raises(AssertionError, match=r"magnitude mismatch"):
        assert_feature_equivalent([a_ids], [a_mags], [b_ids], [b_mags])

    # looser magnitude envelope -> the SAME inputs now pass.
    # 0.5 <= mag_rtol*|b| + mag_atol -> with mag_rtol=0.5 (>> MAG_RTOL):
    mag_loose = 0.5
    assert mag_loose > MAG_RTOL  # genuinely looser than the default
    assert assert_feature_equivalent(
        [a_ids], [a_mags], [b_ids], [b_mags],
        mag_rtol=mag_loose, mag_atol=MAG_ATOL,
    ) is None

    # ...and count/jaccard remain TIGHT regardless of the loose mag kwargs:
    # break jaccard while keeping the loose mag tol -> still raises on
    # jaccard (NOT silently passed by the loosened magnitude envelope).
    bad_b_ids = t.arange(120, 320)  # 80 overlap of 200 -> jaccard ≈ 0.286
    with pytest.raises(
        AssertionError, match=rf"jaccard=.* < {JACCARD_THRESHOLD}"
    ):
        assert_feature_equivalent(
            [a_ids], [a_mags], [bad_b_ids], [b_mags],
            mag_rtol=mag_loose, mag_atol=1e9,
        )


# --------------------------------------------------------------------------- #
# bf16-ULP-aware native-vs-TL MAGNITUDE tolerance (the slow Layer-4 gate's     #
# data-grounded, root-caused bf16 mag_rtol/mag_atol — single-source from       #
# conftest_parity.BF16/FP32_NATIVE_VS_TL_MAG_*). NON-VACUOUS: a 1–2-bf16-ULP   #
# envelope FAILS the tight fp32 tol but PASSES the bf16 tol, while a GROSS     #
# magnitude error STILL fails even at the loosened bf16 tol.                   #
# --------------------------------------------------------------------------- #
def _bf16_ulp_envelope_pair():
    """A common-feature pair reproducing the REAL bf16 native-vs-TL magnitude
    envelope (gemma-3-1b-pt@7 GemmaScope-2, 4090, 2026-05-18), faithful to the
    ACTUAL element-wise ``t.allclose`` semantics:

    * a HUGE feature  b=1328, a=1312  (|Δ|=16, rel 16/1328≈0.0121) — the exact
      feat-118 bf16-ULP signature (bf16 relative ULP 2⁻⁷≈0.0078; |Δ|≈1.5–2
      ULP at magnitude ~1300). It is the GLOBAL max_abs (=16) but it ALONE
      satisfies even the fp32 bound (16 ≤ 0.15+0.04·1328 = 53.27) — so it is
      NOT what trips fp32 allclose;
    * a SMALL-|b| feature  b=4.0, a=7.0  (|Δ|=3.0) — the JumpReLU
      near-threshold / sub-p99 floor. rtol·|b| is negligible here, so this is
      the element that ACTUALLY fails the fp32 tol (3.0 > 0.15+0.04·4 = 0.31)
      while the global max_abs lands on the huge feature — the precise reason
      the realistic 1215-tok input fails fp32 0.04/0.15 even though feat 118
      alone would pass it.

    Co-located ids (same ids both sides, jaccard 1.0, count rdiff 0) so ONLY
    the magnitude gate is in play. Returns ``([ids],[a],[ids],[b])``.
    """
    ids = t.tensor([7, 42])  # 2 common features; identical id sets
    a_mags = t.tensor([1312.0, 7.0])  # huge: |Δ|16 (rel .0121) ; small: |Δ|3
    b_mags = t.tensor([1328.0, 4.0])  # b = the oracle (TL) side
    # Pin the analytic facts the test relies on (so a future edit can't
    # silently make the fixture vacuous):
    assert abs(16.0 / 1328.0 - 0.012048) < 1e-5  # huge-feat rel == real 0.0121
    assert 16.0 <= FP32_NATIVE_VS_TL_MAG_ATOL + FP32_NATIVE_VS_TL_MAG_RTOL * 1328.0
    assert 3.0 > FP32_NATIVE_VS_TL_MAG_ATOL + FP32_NATIVE_VS_TL_MAG_RTOL * 4.0
    assert 3.0 <= BF16_NATIVE_VS_TL_MAG_ATOL + BF16_NATIVE_VS_TL_MAG_RTOL * 4.0
    assert 16.0 <= BF16_NATIVE_VS_TL_MAG_ATOL + BF16_NATIVE_VS_TL_MAG_RTOL * 1328.0
    return [ids], [a_mags], [ids], [b_mags]


def test_bf16_ulp_envelope_fails_fp32_tol_passes_bf16_tol():
    """NON-VACUOUS (a): the realistic 1–2-bf16-ULP native-vs-TL envelope (a
    huge feat with rel≈0.0121 + a small-|b| near-threshold feat) FAILS at the
    tight fp32 ``mag_rtol=0.04/mag_atol=0.15`` — and the failure's reported
    ``max_abs`` is the huge feature's 16 even though the SMALL feature is what
    trips the element-wise allclose — yet PASSES at the data-grounded bf16
    ``mag_rtol=0.05/mag_atol=5.0``. This is exactly the slow-gate bf16 vs fp32
    behaviour, on a synthetic pair faithful to the real allclose semantics."""
    a_ids, a_mags, b_ids, b_mags = _bf16_ulp_envelope_pair()

    # fp32 tol (the UNCHANGED tight native-vs-TL fp32 gate) -> raises; the
    # printed max_abs is the global max (the huge feature's 16), proving the
    # diagnostic reports max_abs while a different small-|b| element fails.
    with pytest.raises(
        AssertionError, match=r"native-vs-TL seg 0: magnitude mismatch max_abs=16"
    ):
        assert_feature_equivalent(
            a_ids, a_mags, b_ids, b_mags, label="native-vs-TL",
            mag_rtol=FP32_NATIVE_VS_TL_MAG_RTOL,
            mag_atol=FP32_NATIVE_VS_TL_MAG_ATOL,
        )

    # data-grounded bf16-ULP-aware tol -> the SAME inputs now pass (the
    # relative term carries the huge feature, the minimal atol the small one).
    assert assert_feature_equivalent(
        a_ids, a_mags, b_ids, b_mags, label="native-vs-TL",
        mag_rtol=BF16_NATIVE_VS_TL_MAG_RTOL,
        mag_atol=BF16_NATIVE_VS_TL_MAG_ATOL,
    ) is None


def test_bf16_mag_tol_still_fails_gross_magnitude_error():
    """NON-VACUOUS (b): loosening to the bf16 tol is NOT disabling. A GROSS
    magnitude error STILL fails even at ``mag_rtol=0.05/mag_atol=5.0``:

    * a HUGE feature off by ~38% (b=1300, a=800, |Δ|=500 ≫ 5.0+0.05·1300=70)
      — far beyond the bf16-ULP envelope (a real capture/scale bug);
    * AND, separately, a SMALL feature off by an order of magnitude (b=8,
      a=88, |Δ|=80 ≫ 5.0+0.05·8=5.4) — proving the minimal atol=5.0 does NOT
      swallow a gross small-feature error (it is ≪ max_abs=16, non-vacuous).

    Both raise at the loosened bf16 tol, so a real bug is still caught."""
    ids = t.tensor([3, 9])

    # (1) gross error on a HUGE feature (38% off) — still caught at bf16 tol.
    a_big = t.tensor([800.0, 5.0])
    b_big = t.tensor([1300.0, 5.0])
    assert 500.0 > BF16_NATIVE_VS_TL_MAG_ATOL + BF16_NATIVE_VS_TL_MAG_RTOL * 1300.0
    with pytest.raises(
        AssertionError, match=r"seg 0: magnitude mismatch max_abs=500"
    ):
        assert_feature_equivalent(
            [ids], [a_big], [ids], [b_big],
            mag_rtol=BF16_NATIVE_VS_TL_MAG_RTOL,
            mag_atol=BF16_NATIVE_VS_TL_MAG_ATOL,
        )

    # (2) gross error on a SMALL feature (10x) — atol=5.0 must NOT mask it.
    a_small = t.tensor([5.0, 88.0])
    b_small = t.tensor([5.0, 8.0])
    assert 80.0 > BF16_NATIVE_VS_TL_MAG_ATOL + BF16_NATIVE_VS_TL_MAG_RTOL * 8.0
    with pytest.raises(
        AssertionError, match=r"seg 0: magnitude mismatch max_abs=80"
    ):
        assert_feature_equivalent(
            [ids], [a_small], [ids], [b_small],
            mag_rtol=BF16_NATIVE_VS_TL_MAG_RTOL,
            mag_atol=BF16_NATIVE_VS_TL_MAG_ATOL,
        )


def test_bf16_mag_tol_still_fails_wrong_site_shaped_magnitudes():
    """NON-VACUOUS (b, wrong-site flavour): even when the feature SET is a
    near-perfect overlap (so jaccard/count pass), a wrong-capture-site-shaped
    magnitude pattern — EVERY common feature scaled by ~2× (a transposed/
    wrong-hook reshape signature) — STILL fails the loosened bf16 mag tol,
    because doubling is ≫ the bf16-ULP envelope on every feature. Proves the
    bf16 mag relaxation catches a systematic magnitude corruption, not just a
    single outlier."""
    a_ids, b_ids = _almost_full_overlap()  # 199/201 jaccard, count rdiff 0
    base = t.full((200,), 50.0)
    doubled = t.full((200,), 100.0)  # every common feature 2x (|Δ|=50 ≫ tol)
    # 50 ≫ 5.0 + 0.05·100 = 10  ⇒ every common feature violates the bf16 tol.
    assert 50.0 > BF16_NATIVE_VS_TL_MAG_ATOL + BF16_NATIVE_VS_TL_MAG_RTOL * 100.0
    with pytest.raises(
        AssertionError, match=r"seg 0: magnitude mismatch max_abs=50"
    ):
        assert_feature_equivalent(
            [a_ids], [base], [b_ids], [doubled],
            mag_rtol=BF16_NATIVE_VS_TL_MAG_RTOL,
            mag_atol=BF16_NATIVE_VS_TL_MAG_ATOL,
        )


def test_bf16_native_vs_tl_mag_constants_are_loose_but_tight():
    """Pin the data-grounded relationships the chosen bf16 mag tol must keep
    (so a future edit can't silently loosen it into vacuity or tighten it
    below the measured bf16-ULP envelope), all against the single-source
    conftest constants — NOT hardcoded:

    * the bf16 rtol is genuinely LOOSER than the tight fp32 rtol, and ≥ ~4×
      the measured max rel 0.0121 (data-grounded safety, the bf16-ULP model);
    * the bf16 atol is genuinely LOOSER than the tight fp32 atol, ≥ the
      measured p99(abs)=4.0 small-feature floor, yet ≪ the measured
      max_abs=16 (so it is MINIMAL / non-vacuous — the relative term, not a
      blanket atol, must carry the huge features);
    * the fp32 tol is UNCHANGED at exactly 0.04 / 0.15 (the tight path)."""
    # fp32 stays exactly the tight, unchanged native-vs-TL fp32 tol.
    assert FP32_NATIVE_VS_TL_MAG_RTOL == 0.04
    assert FP32_NATIVE_VS_TL_MAG_ATOL == 0.15
    # bf16 is genuinely looser than fp32 on BOTH terms (asymmetric oracle).
    assert BF16_NATIVE_VS_TL_MAG_RTOL > FP32_NATIVE_VS_TL_MAG_RTOL
    assert BF16_NATIVE_VS_TL_MAG_ATOL > FP32_NATIVE_VS_TL_MAG_ATOL
    # rtol is data-grounded: ≥ ~4× the measured max rel 0.0121 (bf16-ULP),
    # and not absurdly loose (< 0.10 — still a tight 5%-scale envelope).
    assert BF16_NATIVE_VS_TL_MAG_RTOL >= 4.0 * 0.0121
    assert BF16_NATIVE_VS_TL_MAG_RTOL < 0.10
    # atol is MINIMAL / non-vacuous: it covers the measured p99(abs)=4.0
    # small-feature floor but stays WELL below the measured max_abs=16, so a
    # blanket "atol ≥ max_abs" vacuous gate is provably avoided (the relative
    # term must do the work for the huge features).
    assert BF16_NATIVE_VS_TL_MAG_ATOL >= 4.0          # ≥ measured p99(abs)
    assert BF16_NATIVE_VS_TL_MAG_ATOL < 16.0          # ≪ measured max_abs
    assert BF16_NATIVE_VS_TL_MAG_ATOL <= 0.5 * 16.0   # comfortably minimal


def _jaccard_about_985():
    """Two equal-count segments whose jaccard is ~0.985 — strictly BELOW the
    default JACCARD_THRESHOLD (0.99) yet comfortably ABOVE a 0.98 envelope.

    133 ids each, offset by 1 ⇒ 132 shared, union 134 → jaccard = 132/134 ≈
    0.98507 (for two equal-size-N sets offset by k: |∩|=N−k, |∪|=N+k, so
    jaccard=(N−k)/(N+k); N=133,k=1). Count rdiff is 0 (equal counts) so ONLY
    the jaccard gate is in play. This is the synthetic analogue of the real
    bf16-native-vs-fp32-TL case where a few near-threshold SAE features flip
    across the dtype/kernel boundary (worst observed real jaccard 0.9895).
    """
    a_ids = t.arange(0, 133)
    b_ids = t.arange(1, 134)  # 132 shared (1..132); counts equal (133)
    set_a, set_b = set(a_ids.tolist()), set(b_ids.tolist())
    jac = len(set_a & set_b) / len(set_a | set_b)
    assert 0.98 < jac < JACCARD_THRESHOLD, (
        f"fixture sanity: jaccard {jac} must sit strictly in (0.98, 0.99)"
    )
    return a_ids, b_ids


def test_feature_equivalent_loose_jaccard_kwarg_makes_same_case_pass():
    """Asymmetric-oracle contract for the NEW ``jaccard_threshold=`` kwarg
    (the slow bf16 native-vs-TL gate): a ~0.985-jaccard pair FAILS at the
    default tight JACCARD_THRESHOLD (0.99) but PASSES with the data-grounded
    looser ``jaccard_threshold=0.98`` — proving the kwarg is HONORED — while
    a grossly different feature SET (a wrong-site bug, jaccard ≪ 0.5) STILL
    fails even with the loosened threshold, proving loosening does NOT disable
    the structural gate."""
    a_ids, b_ids = _jaccard_about_985()
    mags_a = [t.ones(len(a_ids))]
    mags_b = [t.ones(len(b_ids))]

    # default tight threshold (0.99) -> raises on jaccard (sanity / anchor):
    # the diagnostic reports the real ~0.9853 jaccard against the default.
    with pytest.raises(
        AssertionError, match=rf"jaccard=0\.985\d* < {JACCARD_THRESHOLD}"
    ):
        assert_feature_equivalent([a_ids], mags_a, [b_ids], mags_b)

    # data-grounded looser structural envelope -> the SAME inputs now pass.
    loose_jac = 0.98
    assert loose_jac < JACCARD_THRESHOLD  # genuinely looser than the default
    assert assert_feature_equivalent(
        [a_ids], mags_a, [b_ids], mags_b, jaccard_threshold=loose_jac,
    ) is None

    # ...and the loosened gate is NOT disabled: a wrong-site-shaped feature
    # SET (only 80 of 200 shared → union 320 → jaccard 0.25 ≪ 0.5) STILL
    # raises even at jaccard_threshold=0.98 (loosening 0.99→0.98 does NOT
    # turn the structural gate off — a wrong capture/layer site is caught).
    wrong_site_b = t.arange(120, 320)  # |∩|=80 |∪|=320 -> jaccard=0.25
    with pytest.raises(
        AssertionError, match=r"jaccard=0\.2500 < 0\.98"
    ):
        assert_feature_equivalent(
            [t.arange(0, 200)], [t.ones(200)],
            [wrong_site_b], [t.ones(200)],
            jaccard_threshold=loose_jac,
        )


def test_feature_equivalent_loose_count_rtol_kwarg_makes_same_case_pass():
    """Same asymmetric-oracle contract for the NEW ``count_rtol=`` kwarg
    (kept at the tight default in the slow gate, but the override must still
    be honored + non-disabling). 100 vs 95 → rdiff = 5/100 = 0.05: FAILS at
    the default FEATURE_COUNT_RTOL (0.01), PASSES with count_rtol=0.06; a
    gross count mismatch (100 vs 50, rdiff 0.5) STILL fails at count_rtol=0.06
    so loosening does not disable the count gate.

    The count gate is checked FIRST in ``assert_feature_equivalent`` (before
    jaccard), so the default-tight and gross cases trip count exactly as
    asserted. The PASS case additionally relaxes ``jaccard_threshold`` to
    isolate the count gate (``arange(100)`` ⊃ ``arange(95)`` ⇒ jaccard
    0.95 < the 0.99 default would otherwise mask the count-PASS) — the SAME
    "isolate the gate under test" idiom the mag-kwarg test uses with
    ``mag_atol=1e9``."""
    a_ids = [t.arange(100)]
    a_mags = [t.ones(100)]
    near_b = [t.arange(95)]  # count rdiff = 5/100 = 0.05; jaccard = 95/100
    near_bm = [t.ones(95)]

    # default tight count rtol (0.01) -> raises on feature-count rdiff
    # (count is gated BEFORE jaccard, so this is the count diagnostic).
    with pytest.raises(
        AssertionError,
        match=rf"feature-count rdiff=0\.0500 > rtol={FEATURE_COUNT_RTOL} "
              rf"\(n_a=100, n_b=95\)",
    ):
        assert_feature_equivalent(a_ids, a_mags, near_b, near_bm)

    # data-grounded looser count envelope -> the SAME inputs now pass the
    # count gate; relax jaccard (=0.0) to isolate the count gate under test
    # (arange(100) ⊃ arange(95) → jaccard 0.95 would otherwise fail).
    loose_count = 0.06
    assert loose_count > FEATURE_COUNT_RTOL  # genuinely looser
    assert assert_feature_equivalent(
        a_ids, a_mags, near_b, near_bm,
        count_rtol=loose_count, jaccard_threshold=0.0,
    ) is None

    # ...and NOT disabled: a gross count mismatch (rdiff 0.5) STILL raises on
    # COUNT even at the loosened count_rtol=0.06 (count gated before jaccard;
    # jaccard relaxed so the raise is unambiguously the count gate).
    gross_b = [t.arange(50)]
    gross_bm = [t.ones(50)]
    with pytest.raises(
        AssertionError,
        match=r"feature-count rdiff=0\.5000 > rtol=0\.06 \(n_a=100, n_b=50\)",
    ):
        assert_feature_equivalent(
            a_ids, a_mags, gross_b, gross_bm,
            count_rtol=loose_count, jaccard_threshold=0.0,
        )


def test_feature_equivalent_default_thresholds_unchanged_by_kwargs():
    """Regression guard: the NEW kwargs default to the tight module constants,
    so an existing caller that passes NEITHER override is byte-behaviour-
    unchanged — a 0.985-jaccard pair STILL fails at the default (proving the
    fp32 path / every legacy caller keeps the tight 0.99 / 0.01)."""
    a_ids, b_ids = _jaccard_about_985()
    # No jaccard_threshold/count_rtol passed -> JACCARD_THRESHOLD (0.99) used.
    with pytest.raises(
        AssertionError, match=rf"jaccard=0\.985\d* < {JACCARD_THRESHOLD}"
    ):
        assert_feature_equivalent(
            [a_ids], [t.ones(len(a_ids))], [b_ids], [t.ones(len(b_ids))],
        )


# --------------------------------------------------------------------------- #
# feature_equivalence_metrics (measure-only sibling powering the slow gate's   #
# permanent envelope instrumentation — same per-seg arithmetic, asserts none)  #
# --------------------------------------------------------------------------- #
def test_feature_equivalence_metrics_matches_assert_arithmetic():
    """The reported ``min_jaccard`` is the WORST per-segment jaccard and uses
    the IDENTICAL arithmetic as ``assert_feature_equivalent`` — the SAME
    0.985-fixture the assert reports as ``jaccard=0.985…`` is reported here as
    ``min_jaccard ≈ 0.98529`` (count rdiff 0). It asserts NOTHING itself."""
    a_ids, b_ids = _jaccard_about_985()
    min_j, max_rd = feature_equivalence_metrics([a_ids], [b_ids])
    assert 0.98 < min_j < JACCARD_THRESHOLD
    assert abs(min_j - 132 / 134) < 1e-9  # exact same formula as the assert
    assert max_rd == 0.0  # equal counts


def test_feature_equivalence_metrics_reports_worst_segment():
    """Multi-segment: returns the MIN jaccard and the MAX count-rdiff across
    ALL segments (so a single bad segment is what gets surfaced/logged)."""
    # seg0: perfect (jaccard 1.0, rdiff 0); seg1: jaccard 0.5, rdiff 0.5.
    a0, b0 = t.arange(0, 10), t.arange(0, 10)
    a1, b1 = t.arange(0, 4), t.arange(0, 2)  # |∩|=2 |∪|=4 -> 0.5; rdiff 2/4
    min_j, max_rd = feature_equivalence_metrics([a0, a1], [b0, b1])
    assert min_j == 0.5
    assert max_rd == 0.5


def test_feature_equivalence_metrics_segment_count_mismatch_raises():
    """Defensive guard mirrors the asserts' top-level segment-count check."""
    with pytest.raises(
        AssertionError,
        match=r"feature_equivalence_metrics: segment count mismatch",
    ):
        feature_equivalence_metrics([t.arange(3)], [t.arange(3), t.arange(2)])


# --------------------------------------------------------------------------- #
# feature_magnitude_metrics (the MAGNITUDE measure-only sibling powering the   #
# slow gate's mag_max_abs/mag_max_rel instrumentation — SAME per-seg common-   #
# feature arithmetic the assert's t.allclose runs against, asserts none).      #
# --------------------------------------------------------------------------- #
def test_feature_magnitude_metrics_matches_assert_arithmetic():
    """The reported ``mag_max_abs`` is IDENTICAL to the number
    ``assert_feature_equivalent`` prints in its ``magnitude mismatch
    max_abs=…`` diagnostic, and ``mag_max_rel`` is the worst |Δ|/|b| over
    common features (b = the oracle side, matching ``t.allclose``'s
    asymmetric ``rtol·|b|``). Same single-source kernel as the assert — pinned
    on the real bf16-ULP envelope pair (huge feat max_abs=16, rel 16/1328)."""
    a_ids, a_mags, b_ids, b_mags = _bf16_ulp_envelope_pair()
    max_abs, max_rel = feature_magnitude_metrics(a_ids, a_mags, b_ids, b_mags)
    # max_abs is the huge feature's 16 (the global max — exactly what the
    # assert's diagnostic prints, verified by the matched fp32-raise test).
    assert max_abs == 16.0
    # max_rel: huge feat 16/1328≈0.01205 vs small feat 3.0/4.0=0.75 → the
    # small-|b| feature dominates the RELATIVE metric (b is the oracle side),
    # which is precisely why a small-|b| feature, not feat 118, trips fp32.
    assert abs(max_rel - 3.0 / 4.0) < 1e-6
    # exact formula tie: max over common features of |a-b| and |a-b|/|b|.
    assert max_abs == pytest.approx(abs(1312.0 - 1328.0))


def test_feature_magnitude_metrics_reports_worst_segment_and_skips_no_common():
    """Multi-segment: returns the MAX |Δ| and MAX rel across ALL segments;
    a segment with NO common features contributes (0.0, 0.0) (vacuously
    perfect — the structural gate + non-vacuity guard handle a disjoint set),
    so a single bad segment is what gets surfaced/logged."""
    # seg0: no common features (disjoint ids) -> (0,0) contribution.
    a0_id, b0_id = t.tensor([1, 2]), t.tensor([7, 8])
    a0_mg, b0_mg = t.tensor([9.0, 9.0]), t.tensor([9.0, 9.0])
    # seg1: one common feature id=5, |Δ|=20 on |b|=10 -> rel 2.0.
    a1_id, b1_id = t.tensor([5, 3]), t.tensor([5, 99])
    a1_mg, b1_mg = t.tensor([30.0, 1.0]), t.tensor([10.0, 1.0])
    max_abs, max_rel = feature_magnitude_metrics(
        [a0_id, a1_id], [a0_mg, a1_mg], [b0_id, b1_id], [b0_mg, b1_mg],
    )
    assert max_abs == 20.0          # only seg1 has a common feature
    assert max_rel == pytest.approx(20.0 / 10.0)


def test_feature_magnitude_metrics_segment_count_mismatch_raises():
    """Defensive guard mirrors the asserts' / structural metric's top-level
    segment-count check (its OWN distinct diagnostic)."""
    with pytest.raises(
        AssertionError,
        match=r"feature_magnitude_metrics: segment count mismatch",
    ):
        feature_magnitude_metrics(
            [t.arange(3)], [t.ones(3)],
            [t.arange(3), t.arange(2)], [t.ones(3), t.ones(2)],
        )


def test_feature_equivalent_no_common_features_skips_magnitude():
    """Both-empty inputs: union=0 → jaccard defaults to 1.0, count rdiff 0,
    no common features → magnitude block skipped → passes silently."""
    empty_id = [t.empty(0, dtype=t.long)]
    empty_mg = [t.empty(0, dtype=t.float32)]
    assert assert_feature_equivalent(
        empty_id, empty_mg, empty_id, empty_mg
    ) is None


# --------------------------------------------------------------------------- #
# assert_features_nonempty (non-vacuity guard for the slow native-vs-nnterp    #
# gate: assert_bit_identical / assert_feature_equivalent both pass VACUOUSLY   #
# when both sides are empty/all-zero — see the two both-empty tests above).    #
# --------------------------------------------------------------------------- #
def test_features_nonempty_passes_on_real_features():
    """At least one segment with ≥1 feature id AND a non-zero magnitude →
    returns None silently (no raise)."""
    ids = [t.tensor([3, 1, 7], dtype=t.int32), t.tensor([5], dtype=t.int32)]
    mags = [t.tensor([0.5, 0.25, 0.125]), t.tensor([9.0])]
    assert assert_features_nonempty(ids, mags, label="native") is None


def test_features_nonempty_all_segments_empty_raises():
    """Every segment length-0 / ``shape (0,)`` (the exact shape ``_sparse``
    emits when no feature pooled > 0) → vacuous → AssertionError mentioning
    the label and the 'vacuous feature set' diagnostic."""
    empty_id = [t.empty(0, dtype=t.int32), t.empty(0, dtype=t.int32)]
    empty_mg = [t.empty(0, dtype=t.float32), t.empty(0, dtype=t.float32)]
    with pytest.raises(
        AssertionError,
        match=r"native: vacuous feature set — no segment has ≥1 feature id "
              r"with a non-zero magnitude",
    ):
        assert_features_nonempty(empty_id, empty_mg, label="native")


def test_features_nonempty_zero_segments_raises():
    """No segments at all (empty per-segment list) → vacuous (``any`` over an
    empty iterable is False) → raises."""
    with pytest.raises(
        AssertionError, match=r"nnterp: vacuous feature set"
    ):
        assert_features_nonempty([], [], label="nnterp")


def test_features_nonempty_all_magnitudes_zero_raises():
    """Feature IDs PRESENT but every magnitude exactly 0.0 → still vacuous
    (the oracle would compare two all-zero magnitude vectors and pass). The
    ``sum(|mags|) == 0`` arm must reject this independently of id count."""
    ids = [t.tensor([1, 2, 3], dtype=t.int32), t.tensor([4, 5], dtype=t.int32)]
    mags = [t.zeros(3, dtype=t.float32), t.zeros(2, dtype=t.float32)]
    with pytest.raises(
        AssertionError,
        match=r"native-vs-nnterp: vacuous feature set — no segment has ≥1 "
              r"feature id with a non-zero magnitude",
    ):
        assert_features_nonempty(ids, mags, label="native-vs-nnterp")


def test_features_nonempty_mixed_one_empty_one_real_passes():
    """Mixed: one empty segment (``shape (0,)``) + one real segment with a
    non-zero magnitude → NOT vacuous → passes silently. (Mirrors the real
    multi-doc case where some docs may have no active features.)"""
    ids = [t.empty(0, dtype=t.int32), t.tensor([2, 9], dtype=t.int32)]
    mags = [t.empty(0, dtype=t.float32), t.tensor([0.0, 4.0])]
    assert assert_features_nonempty(ids, mags, label="native") is None


def test_features_nonempty_segment_count_mismatch_raises():
    """Defensive guard: ids/mags must be aligned per-segment lists (they are,
    by construction in ``_sparse``); a mismatch is a structural bug, not a
    vacuity result, and gets its own distinct diagnostic."""
    with pytest.raises(
        AssertionError,
        match=r"native: vacuity-guard got mismatched segment counts "
              r"\(len\(ids\)=2, len\(mags\)=1\)",
    ):
        assert_features_nonempty(
            [t.tensor([1], dtype=t.int32), t.tensor([2], dtype=t.int32)],
            [t.tensor([1.0])],
            label="native",
        )
