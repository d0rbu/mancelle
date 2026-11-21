"""Reusable cross-backend feature-equivalence assertions.

Tolerances mirror steering/validate_features.py:validate_shard_activations
(feature_count_rtol=0.01, jaccard_threshold=0.99, mags rtol=1e-2/atol=1e-5).

Two assertions are provided:

assert_feature_equivalent
    The magnitude t.allclose tolerances are parameterizable via optional keyword
    arguments ``mag_rtol`` / ``mag_atol`` (defaults = MAG_RTOL / MAG_ATOL, so
    existing callers are byte-unchanged).  The structural gates are likewise
    parameterizable via ``jaccard_threshold`` / ``count_rtol`` (defaults =
    JACCARD_THRESHOLD / FEATURE_COUNT_RTOL).  All four default to the tight
    module constants, so every existing caller is byte-behaviour-unchanged.
    Use the optional params ONLY for a documented, data-grounded cross-backend
    numerical envelope (e.g., native-vs-TL Gemma-2 eager-vs-SDPA magnitude
    drift, or a bf16-native-vs-fp32-TL structural-jaccard envelope where a few
    near-threshold SAE features flip across the dtype/kernel boundary).  This
    is the SAME asymmetric-oracle pattern as the mag tolerances: the strong
    anchor (native ≡ nnterp, bit-identical, same raw-HF/SDPA model) stays
    absolute via assert_bit_identical; only the weaker native-vs-TL oracle —
    which deliberately crosses an attention-kernel/dtype boundary — gets the
    data-grounded looser, still-very-tight gate.  count stays tight unless data
    specifically shows otherwise, so a wrong capture/layer site (which gives a
    grossly different feature SET, jaccard ≪ threshold) is still caught.

assert_bit_identical
    The strongest possible cross-backend assertion: per-segment feature IDs and
    magnitudes must be exactly equal (torch.equal, max_abs == 0.0).  Use when
    both backends run the same raw-HF/SDPA model (native vs nnterp) — any
    divergence here is a real bug in capture site, reshape, or pooling.
"""

from __future__ import annotations

import torch as t

from steering.feature_equivalence import (
    _seg_jaccard_and_count_rdiff,
    feature_equivalence_metrics,  # noqa: F401
)
from tests.steering.native._feature_equivalence_oracles import (
    _seg_common_mag_max_abs_and_rel,
    feature_equivalence_passed,
    feature_magnitude_metrics,  # noqa: F401
)
from tests.steering.native._parity_thresholds import (
    BF16_NATIVE_VS_TL_MAG_ATOL,  # noqa: F401
    BF16_NATIVE_VS_TL_MAG_RTOL,  # noqa: F401
    FEATURE_COUNT_RTOL,
    FP32_NATIVE_VS_TL_MAG_ATOL,  # noqa: F401
    FP32_NATIVE_VS_TL_MAG_RTOL,  # noqa: F401
    JACCARD_THRESHOLD,
    MAG_ATOL,
    MAG_RTOL,
)


def assert_feature_equivalent(
    a_ids: list[t.Tensor],
    a_mags: list[t.Tensor],
    b_ids: list[t.Tensor],
    b_mags: list[t.Tensor],
    *,
    label: str = "A-vs-B",
    mag_rtol: float = MAG_RTOL,
    mag_atol: float = MAG_ATOL,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    count_rtol: float = FEATURE_COUNT_RTOL,
) -> None:
    """Assert two per-segment (ids, mags) sequences are feature-equivalent.

    Structural gates: feature-count rdiff ≤ ``count_rtol`` and jaccard ≥
    ``jaccard_threshold``.  These catch wrong-hook/layer-site bugs (a wrong
    capture site yields a grossly different feature SET → jaccard ≪ any
    sensible threshold) regardless of magnitude tolerance.

    All four numerical tolerances — ``mag_rtol`` / ``mag_atol`` (t.allclose
    envelope) and ``jaccard_threshold`` / ``count_rtol`` (structural gates) —
    are parameterizable so a caller can document a wider, DATA-GROUNDED
    envelope for a specific cross-backend mismatch without touching the module
    defaults or any other caller.  The defaults are the existing tight module
    constants (``MAG_RTOL`` / ``MAG_ATOL`` / ``JACCARD_THRESHOLD`` /
    ``FEATURE_COUNT_RTOL``), so every existing caller is byte-behaviour-
    unchanged.

    WHY a parameterizable structural gate (asymmetric oracle): the strong
    anchor is ``assert_bit_identical`` (native ≡ nnterp — both the SAME
    raw-HF/SDPA model — must match bit-for-bit, no tolerance, even at bf16).
    The native-vs-TL comparison is a deliberately WEAKER oracle: TL forces
    eager attention and is fp32-only, so native at the production bf16 dtype
    vs TL fp32 crosses BOTH an attention-kernel and a precision boundary,
    which can flip a few near-threshold SAE features and pull jaccard a hair
    under the fp32-sized 0.99 constant.  Like the existing mag tolerances,
    the cure is a data-grounded looser (but still extremely tight — a wrong
    site gives jaccard ≪ 0.5) gate on THAT call ONLY; ``count_rtol`` stays at
    the tight default unless data specifically shows otherwise.

    Raises AssertionError with a diagnostic message on the first violation.

    The pass/fail boolean is delegated to the test-only
    ``feature_equivalence_passed`` oracle. When that predicate is ``True`` this
    returns silently. When it is ``False`` the per-segment loop below is re-run
    purely to build the rich
    diagnostic message — it raises on the SAME first violating segment with the
    SAME message text it always did (the loop's gate expressions are the
    structural complement of the predicate's, so the first failing segment and
    the failing gate within it are identical).  The len-equality precondition
    stays an explicit assert here (raised before the boolean is meaningful) so
    the segment-count-mismatch diagnostic is byte-unchanged.
    """
    assert len(a_ids) == len(b_ids) == len(a_mags) == len(b_mags), (
        f"{label}: segment count mismatch "
        f"({len(a_ids)},{len(a_mags)}) vs ({len(b_ids)},{len(b_mags)})"
    )
    if feature_equivalence_passed(
        a_ids, a_mags, b_ids, b_mags,
        mag_rtol=mag_rtol, mag_atol=mag_atol,
        jaccard_threshold=jaccard_threshold, count_rtol=count_rtol,
    ):
        return
    # The single-source predicate said FAIL — re-walk the segments ONLY to
    # raise the SAME detailed AssertionError on the SAME first violation (the
    # asserts below are the exact structural complement of the predicate's
    # short-circuits, so they fire on the identical segment/gate).
    for seg, (aid, amg, bid, bmg) in enumerate(
        zip(a_ids, a_mags, b_ids, b_mags, strict=True)
    ):
        n_a, n_b = len(aid), len(bid)
        jaccard, count_rdiff = _seg_jaccard_and_count_rdiff(aid, bid)
        assert count_rdiff <= count_rtol, (
            f"{label} seg {seg}: feature-count rdiff={count_rdiff:.4f} "
            f"> rtol={count_rtol} (n_a={n_a}, n_b={n_b})"
        )
        assert jaccard >= jaccard_threshold, (
            f"{label} seg {seg}: jaccard={jaccard:.4f} < {jaccard_threshold}"
        )
        common = sorted(set(aid.tolist()) & set(bid.tolist()))
        if common:
            ct = t.tensor(common, dtype=t.long)
            a_mask = t.isin(aid, ct)
            b_mask = t.isin(bid, ct)
            a_sel = amg[a_mask][aid[a_mask].argsort()]
            b_sel = bmg[b_mask][bid[b_mask].argsort()]
            # t.allclose is the element-wise combined predicate
            # ∀i: |aᵢ-bᵢ| ≤ atol + rtol·|bᵢ| (CORRECT per-element semantics —
            # NOT a scalar max_abs ≤ atol+rtol·max|b|; a single small-|b|
            # feature can violate the bound while the global max_abs is on a
            # different large-|b| feature, the exact bf16 GemmaScope-2 case).
            # The printed max_abs comes from the shared single-source kernel so
            # the diagnostic and the measure-only envelope sibling can never
            # drift (the kernel computes max_abs over the SAME co-sorted
            # common-feature vectors this gate runs t.allclose against).
            max_abs, _ = _seg_common_mag_max_abs_and_rel(aid, amg, bid, bmg)
            assert t.allclose(
                a_sel.float(), b_sel.float(), rtol=mag_rtol, atol=mag_atol
            ), (
                f"{label} seg {seg}: magnitude mismatch "
                f"max_abs={max_abs:.6g}"
            )
    # The predicate said FAIL but the diagnostic re-walk found no violating
    # segment — this is structurally impossible (the asserts are the exact
    # complement of the predicate's short-circuits). Guard against a silent
    # pass on a future predicate/asserter divergence.
    raise AssertionError(
        f"{label}: feature_equivalence_passed returned False but the "
        f"per-segment diagnostic re-walk found no violation — the "
        f"single-source predicate and this asserter have drifted"
    )


def assert_features_nonempty(
    ids: list[t.Tensor],
    mags: list[t.Tensor],
    *,
    label: str,
) -> None:
    """Assert a per-segment (ids, mags) sequence is NOT vacuous.

    ``assert_bit_identical`` / ``assert_feature_equivalent`` provably PASS when
    BOTH sides are empty / all-zero (both-empty: equal length-0 ID tensors and a
    short-circuited ``max_abs == 0.0``; no-common-features: union 0 → jaccard
    1.0).  A regression making BOTH backends emit empty / all-zero features would
    therefore pass the headline native-vs-nnterp gate VACUOUSLY.  Call this on
    EACH side first so the comparison only certifies real features.

    Raises AssertionError (consistent with this module's diagnostic style,
    mentioning ``label``) iff the feature set is vacuous, i.e. there is NO
    segment that has at least one feature id AND a non-zero magnitude — covering
    zero segments, every segment ``len`` 0 / ``shape (0,)``, and the
    all-magnitudes-exactly-zero case.  Passes silently when at least one segment
    has ≥1 feature with a non-zero magnitude.
    """
    assert len(ids) == len(mags), (
        f"{label}: vacuity-guard got mismatched segment counts "
        f"(len(ids)={len(ids)}, len(mags)={len(mags)})"
    )
    total_features = sum(int(seg_ids.numel()) for seg_ids in ids)
    total_abs_mag = sum(float(seg_mags.abs().sum().item()) for seg_mags in mags)
    nonempty = any(
        seg_ids.numel() > 0 and float(seg_mags.abs().sum().item()) > 0.0
        for seg_ids, seg_mags in zip(ids, mags, strict=True)
    )
    assert nonempty, (
        f"{label}: vacuous feature set — no segment has ≥1 feature id with a "
        f"non-zero magnitude ({len(ids)} segments, total_features="
        f"{total_features}, total_abs_mag={total_abs_mag:.6g}); the cross-"
        f"backend oracle would pass VACUOUSLY on this side"
    )


def assert_bit_identical(
    a_ids: list[t.Tensor],
    a_mags: list[t.Tensor],
    b_ids: list[t.Tensor],
    b_mags: list[t.Tensor],
    *,
    label: str = "A-vs-B",
) -> None:
    """Assert two per-segment (ids, mags) sequences are exactly equal.

    Both the sorted feature IDs and the co-sorted magnitudes must satisfy
    torch.equal (equivalently, max_abs == 0.0).  This is the strongest possible
    cross-backend assertion and should be used whenever both runs share the same
    raw-HF SDPA model (e.g., native vs nnterp) — any divergence is a real bug.

    Prints a clear diagnostic on the first mismatching segment (segment index,
    max_abs, which check failed) before raising AssertionError.
    """
    assert len(a_ids) == len(b_ids) == len(a_mags) == len(b_mags), (
        f"{label}: segment count mismatch "
        f"({len(a_ids)},{len(a_mags)}) vs ({len(b_ids)},{len(b_mags)})"
    )
    for seg, (aid, amg, bid, bmg) in enumerate(
        zip(a_ids, a_mags, b_ids, b_mags, strict=True)
    ):
        # Sort both by feature ID so comparison is order-independent
        a_order = aid.argsort()
        b_order = bid.argsort()
        aid_s = aid[a_order]
        bid_s = bid[b_order]
        assert t.equal(aid_s, bid_s), (
            f"{label} seg {seg}: feature IDs differ — "
            f"n_a={len(aid)}, n_b={len(bid)}, "
            f"in_a_not_b={set(aid.tolist())-set(bid.tolist())}, "
            f"in_b_not_a={set(bid.tolist())-set(aid.tolist())}"
        )
        amg_s = amg[a_order]
        bmg_s = bmg[b_order]
        max_abs = (amg_s.float() - bmg_s.float()).abs().max().item() if len(amg_s) > 0 else 0.0
        assert t.equal(amg_s.float(), bmg_s.float()), (
            f"{label} seg {seg}: magnitude mismatch (max_abs={max_abs:.6g} > 0.0)"
        )
