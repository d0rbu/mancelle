"""Unit tests for DeltaMode.splice_counterfactual (issue #596)."""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.delta import (
    DeltaMode,
    compute_doc_counterfactual_deltas,
    compute_doc_deltas,
    parse_feature_delta,
)


class TestModeParsing:
    def test_enum_value(self) -> None:
        assert DeltaMode.splice_counterfactual.value == "splice_counterfactual"
        assert DeltaMode("splice_counterfactual") is DeltaMode.splice_counterfactual

    def test_parse_feature_delta_accepts_new_mode(self) -> None:
        cfg = parse_feature_delta({"mode": "splice_counterfactual", "neg_delta_ratio": 0.3})
        assert cfg is not None
        assert cfg.mode is DeltaMode.splice_counterfactual
        assert cfg.neg_delta_ratio == 0.3

    def test_parse_default_neg_delta_ratio(self) -> None:
        cfg = parse_feature_delta({"mode": "splice_counterfactual"})
        assert cfg is not None and cfg.neg_delta_ratio == 1.0


class TestCarriedStateEntryPointRejectsCounterfactualMode:
    def test_compute_doc_deltas_rejects(self) -> None:
        table = t.linspace(0.0, 1.0, 99).unsqueeze(0).expand(8, -1).to(t.float64).contiguous()
        with pytest.raises(ValueError, match="splice_counterfactual"):
            compute_doc_deltas(
                [t.tensor([1], dtype=t.int64)], [t.tensor([1.0])], [None],
                8, DeltaMode.splice_counterfactual, table,
            )


_TABLE = t.linspace(0.0, 1.0, 99).unsqueeze(0).expand(64, -1).to(t.float64).contiguous()


def _seg(fids, mags, pcts=None):
    return (
        t.tensor(fids, dtype=t.int64),
        t.tensor(mags, dtype=t.float32),
        None if pcts is None else t.tensor(pcts, dtype=t.float32),
    )


class TestCounterfactualDeltas:
    def test_none_ref_slots_return_none(self) -> None:
        fids, mags, pcts = _seg([1, 2], [1.0, 2.0], [0.5, 0.6])
        out = compute_doc_counterfactual_deltas(
            [fids, fids], [mags, mags], [pcts, pcts], [None, None], 64,
        )
        assert out == [None, None]

    def test_none_ref_slot_with_empty_current_segment_raises(self) -> None:
        with pytest.raises(ValueError, match="zero feature ids"):
            compute_doc_counterfactual_deltas(
                [t.tensor([], dtype=t.int64)],
                [t.tensor([], dtype=t.float32)],
                [None],
                [None],
                64,
                _TABLE,
            )

    def test_hand_computed_delta_with_precomputed_pcts(self) -> None:
        # current: f1=0.8, f2=0.3, f3=0.5 ; ref: f2=0.7, f4=0.4
        cur = _seg([1, 2, 3], [9.0, 9.0, 9.0], [0.8, 0.3, 0.5])
        ref = _seg([2, 4], [9.0, 9.0], [0.7, 0.4])
        (res,) = compute_doc_counterfactual_deltas(
            [cur[0]], [cur[1]], [cur[2]], [ref], 64,
        )
        assert res is not None
        f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask, f_pcts = res
        # deltas: f1 = 0.8-0 = +0.8 ; f2 = 0.3-0.7 = -0.4 ; f3 = 0.5-0 = +0.5
        assert f_fids.tolist() == [1, 3]
        assert t.allclose(d_vals, t.tensor([0.8, 0.5]))  # ty: ignore[invalid-argument-type]
        assert f_pcts is not None and t.allclose(f_pcts, t.tensor([0.8, 0.5]))
        # negatives: ref-only f4 (weight 0.4) + negative-delta f2 (weight 0.4)
        assert sorted(neg_fids.tolist()) == [2, 4]
        w = dict(zip(neg_fids.tolist(), neg_w.tolist(), strict=True))
        assert abs(w[4] - 0.4) < 1e-6 and abs(w[2] - 0.4) < 1e-6
        # ref_mask = union {1,2,3} ∪ {2,4}
        assert set(t.nonzero(ref_mask).squeeze(1).tolist()) == {1, 2, 3, 4}

    def test_matches_two_segment_carried_reference(self) -> None:
        """O3: one counterfactual step equals segment 1 of a carried-state document."""
        cur = _seg([3, 5, 9], [1.0, 2.0, 3.0], [0.6, 0.1, 0.9])
        ref = _seg([5, 7], [1.0, 1.0], [0.2, 0.8])
        (res,) = compute_doc_counterfactual_deltas(
            [cur[0]], [cur[1]], [cur[2]], [ref], 64,
        )
        _, oracle = compute_doc_deltas(
            [ref[0], cur[0]],
            [ref[1], cur[1]],
            [ref[2], cur[2]],
            64,
            DeltaMode.prev_segment,
        )
        f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask, _ = res  # ty: ignore[not-iterable]
        o_fids, o_mags, o_dvals, o_neg_fids, o_neg_w, o_ref_mask, _ = oracle
        assert t.equal(f_fids, o_fids) and t.equal(f_mags, o_mags)
        assert t.equal(d_vals, o_dvals)  # ty: ignore[invalid-argument-type]
        assert t.equal(neg_fids, o_neg_fids) and t.equal(neg_w, o_neg_w)
        assert t.equal(ref_mask, o_ref_mask)

    def test_table_path_when_pcts_missing(self) -> None:
        cur = (t.tensor([1, 2], dtype=t.int64), t.tensor([5.0, 5.0]), None)
        ref = (t.tensor([2], dtype=t.int64), t.tensor([5.0]), None)
        (res,) = compute_doc_counterfactual_deltas(
            [cur[0]], [cur[1]], [cur[2]], [ref], 64, _TABLE,
        )
        assert res is not None
        assert res[6] is None  # filtered_pcts None on the table path

    def test_table_required_when_pcts_missing(self) -> None:
        cur = (t.tensor([1], dtype=t.int64), t.tensor([5.0]), None)
        ref = (t.tensor([2], dtype=t.int64), t.tensor([5.0]), None)
        with pytest.raises(AssertionError):
            compute_doc_counterfactual_deltas([cur[0]], [cur[1]], [cur[2]], [ref], 64)

    def test_ref_state_does_not_leak_between_segments(self) -> None:
        """Segment 1's ref must not see segment 0's ref values."""
        cur0 = _seg([1], [1.0], [0.5])
        cur1 = _seg([2], [1.0], [0.5])   # feature 2: ref'd in seg0, NOT in seg1's ref
        ref0 = _seg([2], [1.0], [0.9])
        ref1 = _seg([3], [1.0], [0.9])
        out = compute_doc_counterfactual_deltas(
            [cur0[0], cur1[0]], [cur0[1], cur1[1]], [cur0[2], cur1[2]],
            [ref0, ref1], 64,
        )
        res1 = out[1]
        assert res1 is not None
        # f2 delta in seg1 = 0.5 - 0 (NOT 0.5 - 0.9): positive, so selected.
        assert res1[0].tolist() == [2]

    def test_seg_refs_length_mismatch_raises(self) -> None:
        cur = _seg([1], [1.0], [0.5])
        with pytest.raises(ValueError, match="seg_refs"):
            compute_doc_counterfactual_deltas([cur[0]], [cur[1]], [cur[2]], [], 64)

    def test_empty_current_segment_raises(self) -> None:
        ref = _seg([4], [1.0], [0.7])
        with pytest.raises(ValueError, match="zero feature ids"):
            compute_doc_counterfactual_deltas(
                [t.tensor([], dtype=t.int64)],
                [t.tensor([], dtype=t.float32)],
                [None],
                [ref],
                64,
                _TABLE,
            )

    def test_empty_reference_segment_raises(self) -> None:
        cur = _seg([4], [1.0], [0.7])
        ref = _seg([], [], [])
        with pytest.raises(ValueError, match="reference"):
            compute_doc_counterfactual_deltas(
                [cur[0]], [cur[1]], [cur[2]], [ref], 64, _TABLE,
            )


@given(
    n=st.integers(min_value=1, max_value=12),
    m=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
def test_counterfactual_properties(n, m, seed):
    g = t.Generator().manual_seed(seed)
    fids = t.randperm(64, generator=g)[:n].to(t.int64)
    pcts = t.rand(n, generator=g)
    ref_fids = t.randperm(64, generator=g)[:m].to(t.int64)
    ref_pcts = t.rand(m, generator=g)
    mags = t.rand(n, generator=g) * 10
    (res,) = compute_doc_counterfactual_deltas(
        [fids], [mags], [pcts], [(ref_fids, t.ones(m), ref_pcts)], 64,
    )
    assert res is not None
    f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask, f_pcts = res
    cur_set = set(fids.tolist())
    assert set(f_fids.tolist()) <= cur_set                     # positives ⊆ current
    assert bool((d_vals > 0).all()) if len(d_vals) else True   # strictly positive  # ty: ignore[invalid-argument-type, unsupported-operator]
    assert bool((d_vals <= 1.0).all()) if len(d_vals) else True  # ty: ignore[invalid-argument-type, unsupported-operator]
    assert bool((neg_w >= 0).all()) if len(neg_w) else True    # weights non-negative
    assert set(f_fids.tolist()).isdisjoint(set(neg_fids.tolist()))
    union = cur_set | set(ref_fids.tolist())
    assert set(t.nonzero(ref_mask).squeeze(1).tolist()) == union
    # Determinism
    (res2,) = compute_doc_counterfactual_deltas(
        [fids], [mags], [pcts], [(ref_fids, t.ones(m), ref_pcts)], 64,
    )
    assert all(
        (a is None and b is None) or t.equal(a, b)
        for a, b in zip(res, res2, strict=True)  # ty: ignore[invalid-argument-type, not-iterable]
    )
