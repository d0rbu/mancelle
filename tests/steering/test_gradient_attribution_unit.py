"""Unit tests for gradient-attribution building blocks.

Covers the pure schedule function's edge geometry (final-unit merging,
single-segment docs, 1-token docs, segment_size=1), the self-mode
last-row-zero theorem, BOS keep-mask composition, empty read windows,
collector input validation, and the gradient_overlay identity helpers.
The heavyweight production<->oracle parity lives in
``tests/reference_gradient_attribution.py``.
"""

from __future__ import annotations

import weakref
from typing import Any

import pytest
import torch as t

from errors import GradientAttributionError
from steering.gradient_attribution import (
    GradientAttributionCollector,
    SegmentRead,
    build_backward_schedule,
)
from steering.gradient_overlay import (
    GRAD_SCORE_REDUCTION,
    GRAD_SPLIT_POLICY_VERSION,
    GradientAttributionMode,
    GradientRankingSpec,
    canonical_gradient_descriptor,
    canonical_gradient_selection_dict,
    compute_gradient_overlay_checksum,
    parse_gradient_ranking_spec,
    provenance_dir,
    resolve_gradient_spec_from_feature_selection,
)
from tests.reference_gradient_attribution import (
    TINY_BOS,
    TINY_LAYER,
    fixed_segments,
    full_fids,
    make_tiny_sae,
    ref_unit_gradient,
    run_collector,
    seeded_ids,
    tiny_gemma2,
)

FHS = GradientAttributionMode.future_half_split
FSF = GradientAttributionMode.future_self_full
SELF = GradientAttributionMode.self_


# ---------------------------------------------------------------------------
# Schedule geometry — hand-computed cases
# ---------------------------------------------------------------------------


def _reads(units) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for u in units:
        for r in u.reads:
            assert r.seg_idx not in out, "segment read twice"
            out[r.seg_idx] = (r.row_lo, r.row_hi)
    return out


class TestFinalUnitGeometry:
    def test_half_split_normal_final(self):
        # doc 24, segs of 8: final [16, 24) len 8 >= threshold 8 => own unit,
        # mid = 16 + 4 = 20: backprop preds of [20,24), read rows [16,20).
        units = build_backward_schedule(fixed_segments(24, 8), 24, FHS, 8)
        assert units[0].key == ("final", 0)
        assert (units[0].loss_lo, units[0].loss_hi) == (19, 23)
        assert units[0].reads == (SegmentRead(2, 16, 20),)
        assert units[1].key == ("final", 1)
        assert (units[1].loss_lo, units[1].loss_hi) == (15, 19)
        assert units[1].reads == (SegmentRead(1, 8, 16),)
        assert units[2].key == ("chunk", 1)
        assert (units[2].loss_lo, units[2].loss_hi) == (7, 15)
        assert units[2].reads == (SegmentRead(0, 0, 8),)

    def test_self_full_normal_final(self):
        # Phase 1 has no loss under self_full (src_lo == u), so it is NOT
        # emitted: the below-segment read rides on phase 0.
        units = build_backward_schedule(fixed_segments(24, 8), 24, FSF, 8)
        assert units[0].key == ("final", 0)
        assert (units[0].loss_lo, units[0].loss_hi) == (15, 23)
        assert units[0].reads == (
            SegmentRead(2, 16, 24),
            SegmentRead(1, 8, 16),
        )
        assert units[1].key == ("chunk", 1)

    def test_merged_final_half_split(self):
        # doc 26, segs of 8: final [24, 26) len 2 < 8 => merged unit [16, 26),
        # mid = 16 + 5 = 21.
        units = build_backward_schedule(fixed_segments(26, 8), 26, FHS, 8)
        assert (units[0].loss_lo, units[0].loss_hi) == (20, 25)
        assert units[0].reads == (
            SegmentRead(2, 16, 21),
            SegmentRead(3, 16, 21),
        )
        assert (units[1].loss_lo, units[1].loss_hi) == (15, 20)
        assert units[1].reads == (SegmentRead(1, 8, 16),)

    def test_merged_final_self_full(self):
        units = build_backward_schedule(fixed_segments(26, 8), 26, FSF, 8)
        assert (units[0].loss_lo, units[0].loss_hi) == (15, 25)
        assert units[0].reads == (
            SegmentRead(2, 16, 26),
            SegmentRead(3, 16, 26),
            SegmentRead(1, 8, 16),  # phase-1 fold (self_full has no phase 1)
        )

    def test_threshold_boundaries(self):
        # final length >= threshold => own unit; < threshold => merged.
        own = build_backward_schedule(fixed_segments(24, 8), 24, FHS, 8)
        assert len(own[0].reads) == 1
        merged = build_backward_schedule(fixed_segments(23, 8), 23, FHS, 8)
        assert len(merged[0].reads) == 2
        # min_final=2 merges ONLY 1-token finals: a 2-token final stays own.
        own2 = build_backward_schedule(fixed_segments(10, 8), 10, FHS, 2)
        assert len(own2[0].reads) == 1  # final [8,10) len 2 >= 2 => own unit
        merged2 = build_backward_schedule(fixed_segments(9, 8), 9, FHS, 2)
        assert len(merged2[0].reads) == 2  # final [8,9) len 1 < 2 => merged

    def test_single_segment_short_doc_scored_unmerged(self):
        # 5-token single-segment doc below threshold 8: nothing to merge with
        # — the mode's final-unit treatment applies UNCHANGED. FHS: mid = 2,
        # backprop preds of [2,5), read rows [0,2); no phase 1 is emitted
        # (nothing below the unit consumes it).
        units = build_backward_schedule(fixed_segments(5, 8), 5, FHS, 8)
        assert [u.key for u in units] == [("final", 0)]
        assert (units[0].loss_lo, units[0].loss_hi) == (1, 4)
        assert units[0].reads == (SegmentRead(0, 0, 2),)
        units_self = build_backward_schedule(fixed_segments(5, 8), 5, SELF, 8)
        assert units_self[-1].reads == (SegmentRead(0, 0, 5),)

    def test_single_segment_normal_doc(self):
        # 8-token single segment at threshold: final-unit policy applies to
        # segment 0 (mid = 4). Phase 1 (preds of [0,4)) would be consumed by
        # nothing — not emitted.
        units = build_backward_schedule(fixed_segments(8, 8), 8, FHS, 8)
        assert [u.key for u in units] == [("final", 0)]
        assert units[0].reads == (SegmentRead(0, 0, 4),)
        assert (units[0].loss_lo, units[0].loss_hi) == (3, 7)

    def test_one_token_doc_raises(self):
        # A 1-token doc has no LM loss at all — the schedule RAISES (never a
        # zero placeholder); the producer rejects such docs upstream too.
        for mode in (FHS, FSF, SELF):
            with pytest.raises(GradientAttributionError, match="no LM loss"):
                build_backward_schedule(fixed_segments(1, 8), 1, mode, 8)

    def test_unmerged_one_token_final_raises(self):
        # A 1-token final that escapes merging (min_final < 2) is unscoreable
        # — the schedule RAISES naming the unit. min_final >= 2 (enforced by
        # the producer) makes this impossible; here we drive it directly.
        for mode in (FHS, FSF, SELF):
            with pytest.raises(GradientAttributionError, match="1 token"):
                build_backward_schedule(fixed_segments(9, 8), 9, mode, 1)

    def test_self_mode_first_token_segment_raises(self):
        # Self mode, a 1-token first segment [0,1) (only with segment_size==1,
        # which the producer forbids): its single token has no predictor, so
        # the schedule RAISES rather than emit a zero row.
        with pytest.raises(GradientAttributionError, match="no own predictions"):
            build_backward_schedule(fixed_segments(4, 1), 4, SELF, 2)

    def test_segment_size_one_future_mode_valid(self):
        # segment_size == 1 in a FUTURE mode IS scoreable at the schedule
        # level (1-row reads) when min_final >= 2 merges the 1-token final;
        # the producer still forbids segment_size < 2, but the schedule's
        # contract is "raise only when unscoreable". 4 single-token segments,
        # min_final=2 merges the final => unit [2,4); FHS splits at mid=3, so
        # the merged slots read rows [2,3) and the below segments [1,2)/[0,1).
        units = build_backward_schedule(fixed_segments(4, 1), 4, FHS, 2)
        assert _reads(units) == {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (2, 3)}

    def test_pred_range_rejects_malformed_spans(self):
        from steering.gradient_attribution import _pred_range

        assert _pred_range(0, 1) == (0, 0)  # empty RESULT is representable
        assert _pred_range(8, 16) == (7, 15)
        for a, b in [(3, 3), (5, 4), (-1, 2)]:
            with pytest.raises(GradientAttributionError, match="malformed"):
                _pred_range(a, b)

    def test_segments_must_tile(self):
        with pytest.raises(ValueError, match="tile"):
            build_backward_schedule([(0, 8)], 10, FHS, 8)
        with pytest.raises(ValueError, match=">= 1 segment"):
            build_backward_schedule([], 0, FHS, 8)


def test_self_mode_isolation_keys():
    units = build_backward_schedule(fixed_segments(26, 8), 26, SELF, 8)
    assert all(u.reset_grad_before for u in units)
    assert [u.key for u in units] == [("self", 0), ("self", 1), ("self", 2)]
    # merged unit reads both trailing segments over the merged window
    assert units[-1].reads == (
        SegmentRead(2, 16, 26),
        SegmentRead(3, 16, 26),
    )


# ---------------------------------------------------------------------------
# Self-mode last-row-zero theorem
# ---------------------------------------------------------------------------


def test_self_mode_last_row_gradient_is_exactly_zero():
    """A segment's last token influences no prediction INSIDE its own chunk
    (its first influenced position predicts the next segment's first token),
    so its self-mode gradient row is exactly zero. A one-off error in the
    chunk ranges breaks this immediately."""
    model = tiny_gemma2(softcap=True)
    doc_len, seg_size = 24, 8
    ids = seeded_ids(doc_len, seed=44)
    segs = fixed_segments(doc_len, seg_size)
    for b, e in segs[:-1]:  # plain chunks
        positions = [j for j in range(doc_len - 1) if b <= j + 1 < e]
        g = ref_unit_gradient(model, ids, TINY_LAYER, positions)
        assert t.equal(g[e - 1], t.zeros_like(g[e - 1])), (
            f"segment [{b},{e}): last row has non-zero self-chunk gradient"
        )
        if e - b > 1:
            assert not t.equal(g[e - 2], t.zeros_like(g[e - 2])), (
                "penultimate row should receive gradient (non-vacuity)"
            )


# ---------------------------------------------------------------------------
# BOS keep-mask
# ---------------------------------------------------------------------------


def test_bos_rows_excluded_from_max():
    """With ignore_bos_features, a BOS row carrying the would-be max must not
    win the reduction. Construct it by planting BOS mid-segment."""
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("standard", seed=3)
    with t.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.fill_(1.0)  # every feature always fires => pure grad ranking
    doc_len, seg_size = 16, 8
    ids = seeded_ids(doc_len, seed=6)
    ids[3] = TINY_BOS  # a BOS inside segment 0
    segs = fixed_segments(doc_len, seg_size)
    fids = [full_fids() for _ in segs]

    kwargs: dict[str, Any] = {
        "layer": TINY_LAYER, "mode": FSF, "min_final_segment_tokens": 8
    }
    with_mask = GradientAttributionCollector(
        model, sae, bos_token_id=TINY_BOS, ignore_bos_features=True, **kwargs
    ).score_documents([ids], [segs], [fids])[0]
    without = GradientAttributionCollector(
        model, sae, **kwargs
    ).score_documents([ids], [segs], [fids])[0]

    assert not t.equal(with_mask[0], without[0]), (
        "masking BOS rows must change segment-0 scores (non-vacuity)"
    )
    # Reference: recompute the unmasked candidate set with rows 0 and 3
    # removed and check the masked result equals the max over kept rows only.
    positions = [j for j in range(doc_len - 1) if j + 1 >= 8]
    g = ref_unit_gradient(model, ids, TINY_LAYER, positions)
    vals = t.relu(-(g[0:8].float() @ sae.W_dec.float().T))
    keep = t.tensor([tok != TINY_BOS for tok in ids[0:8]])
    expected = vals.masked_fill(~keep.unsqueeze(1), 0.0).amax(0).to(t.float16)
    assert t.equal(with_mask[0], expected)


def test_ignore_bos_requires_bos_id():
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("standard")
    with pytest.raises(GradientAttributionError, match="bos_token_id"):
        GradientAttributionCollector(
            model, sae, layer=TINY_LAYER, mode=FHS,
            min_final_segment_tokens=8, ignore_bos_features=True,
        )


# ---------------------------------------------------------------------------
# Collector validation + degenerate inputs
# ---------------------------------------------------------------------------


def test_one_token_doc_in_batch_raises():
    """A 1-token doc anywhere in a batch makes the whole call RAISE (no LM
    loss) — never a silent zero row for it."""
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("jumprelu")
    docs = [seeded_ids(5, seed=2), seeded_ids(1, seed=1)]
    seg_lists = [fixed_segments(5, 8), fixed_segments(1, 8)]
    fid_lists = [[full_fids()] for _ in docs]
    for mode in (FHS, FSF, SELF):
        collector = GradientAttributionCollector(
            model, sae, layer=TINY_LAYER, mode=mode, min_final_segment_tokens=8
        )
        with pytest.raises(GradientAttributionError, match="no LM loss"):
            collector.score_documents(docs, seg_lists, fid_lists)


def test_short_single_segment_doc_scored():
    """A short (>= 2 token) single-segment doc gets the mode's REAL final-unit
    treatment and matches the oracle (never zeroed)."""
    from tests.reference_gradient_attribution import ref_score_document

    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("jumprelu")
    ids, segs, fids = seeded_ids(5, seed=2), fixed_segments(5, 8), [full_fids()]
    for mode in (FHS, FSF, SELF):
        collector = GradientAttributionCollector(
            model, sae, layer=TINY_LAYER, mode=mode, min_final_segment_tokens=8
        )
        out = collector.score_documents([ids], [segs], [fids])[0]
        ref = ref_score_document(model, sae, ids, segs, fids, mode, 8)
        assert t.equal(out[0], ref[0]), f"{mode.value}: diverges from oracle"
        assert bool((out[0].float() > 0).any()), "must carry real signal"


def test_layer_out_of_range():
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("standard")
    with pytest.raises(GradientAttributionError, match="out of range"):
        GradientAttributionCollector(
            model, sae, layer=99, mode=FHS, min_final_segment_tokens=8
        )


def test_empty_fid_row_raises():
    """Base shards never store zero-feature segments (issue #522 producer
    guard); a read with no stored features is corruption and must raise, not
    silently return an empty row."""
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("standard")
    ids = seeded_ids(16, seed=3)
    segs = fixed_segments(16, 8)
    fids = [t.zeros(0, dtype=t.int64), full_fids()]
    with pytest.raises(GradientAttributionError, match="never emit empty reads"):
        run_collector(model, sae, [ids], [segs], [fids], FHS, 8)


# ---------------------------------------------------------------------------
# gradient_overlay identity helpers
# ---------------------------------------------------------------------------


class TestGradientOverlayHelpers:
    def test_sae_l0_normalized_in_descriptor(self):
        """Producer and consumer derive l0 from DIFFERENT config files; the
        descriptor must hash 68, "68", and " 68 " identically (and keep tier
        strings / None distinct) so a byte-valid overlay is never re-addressed
        by a spelling difference."""

        def _checksum(l0):
            return compute_gradient_overlay_checksum(
                canonical_gradient_descriptor(
                    attribution_mode="self",
                    model="gemma-2-2b",
                    model_dtype="bfloat16",
                    layer=8,
                    sae_release="gemmascope",
                    sae_width="16k",
                    sae_l0=l0,
                    activation_type="residual",
                    ignore_bos_features=False,
                    min_final_segment_tokens=8,
                    percentile_checksum="pctfixed",
                )
            )

        assert _checksum(68) == _checksum("68") == _checksum(" 68 ")
        assert len({_checksum(68), _checksum("medium"), _checksum(None)}) == 3
        assert _checksum("medium") == _checksum("medium")

    def test_parse_round_trip(self):
        spec = parse_gradient_ranking_spec(
            {"mode": "future_half_split", "model_dtype": "float32",
             "min_final_segment_tokens": 4}
        )
        assert spec == GradientRankingSpec(
            mode=FHS, model_dtype="float32", min_final_segment_tokens=4
        )

    def test_parse_defaults(self):
        spec = parse_gradient_ranking_spec({"mode": "self"})
        assert spec.mode is SELF
        assert spec.model_dtype == "bfloat16"
        assert spec.min_final_segment_tokens == 8

    @pytest.mark.parametrize(
        "raw,match",
        [
            ({}, "requires 'mode'"),
            ({"mode": "future"}, "Unsupported gradient attribution mode"),
            ({"mode": "self", "model_dtype": "float16"}, "model_dtype"),
            ({"mode": "self", "min_final_segment_tokens": 1}, ">= 2"),
            ({"mode": "self", "min_final_segment_tokens": 0}, ">= 2"),
            ({"mode": "self", "min_final_segment_tokens": 1.5}, "int"),
            ({"mode": "self", "min_final_segment_tokens": True}, "int"),
            ({"mode": "self", "typo_key": 1}, "unknown keys"),
            ("self", "must be a mapping"),
        ],
    )
    def test_parse_rejections(self, raw, match):
        with pytest.raises(ValueError, match=match):
            parse_gradient_ranking_spec(raw)

    def test_resolve_absent_and_null(self):
        assert resolve_gradient_spec_from_feature_selection(None) is None
        assert resolve_gradient_spec_from_feature_selection({}) is None
        assert (
            resolve_gradient_spec_from_feature_selection({"gradient": None}) is None
        )
        spec = resolve_gradient_spec_from_feature_selection(
            {"gradient": {"mode": "self"}}
        )
        assert spec is not None and spec.mode is SELF

    def test_canonical_selection_dict(self):
        spec = GradientRankingSpec(mode=FHS)
        assert canonical_gradient_selection_dict(spec) == {
            "mode": "future_half_split",
            "model_dtype": "bfloat16",
            "min_final_segment_tokens": 8,
            "score_reduction": GRAD_SCORE_REDUCTION,
            "split_policy_version": GRAD_SPLIT_POLICY_VERSION,
        }

    def test_descriptor_checksum_sensitivity(self):
        base: dict[str, Any] = {
            "attribution_mode": "future_half_split",
            "model": "gemma-2-2b",
            "model_dtype": "bfloat16",
            "layer": 8,
            "sae_release": "gemmascope",
            "sae_width": "16k",
            "sae_l0": None,
            "activation_type": "residual",
            "ignore_bos_features": False,
            "min_final_segment_tokens": 8,
            "percentile_checksum": "pct_aaaaaaaa",
        }
        ref = compute_gradient_overlay_checksum(canonical_gradient_descriptor(**base))
        assert len(ref) == 64
        for key, value in [
            ("attribution_mode", "self"),
            ("model_dtype", "float32"),
            ("layer", 9),
            ("min_final_segment_tokens", 4),
            ("ignore_bos_features", True),
            # The pct identity is part of the content hash: a different
            # percentile overlay MUST re-address grad/<hash>/ (finding #1).
            ("percentile_checksum", "pct_bbbbbbbb"),
        ]:
            variant: dict[str, Any] = {**base, key: value}
            other = compute_gradient_overlay_checksum(
                canonical_gradient_descriptor(**variant)
            )
            assert other != ref, f"checksum insensitive to {key}"
        # Determinism
        again = compute_gradient_overlay_checksum(canonical_gradient_descriptor(**base))
        assert again == ref

    def test_provenance_dir(self):
        from pathlib import Path

        assert provenance_dir("/x/packed") == Path("/x/packed/provenance")


class TestResumeGradientAssert:
    """assert_resume_gradient_matches — the mask/clustering resume-assert
    twin. The checkpoint-dir leaf omits the code-versioned constants; this
    assert is what stops a v1-scored lineage from continuing on re-scored
    sel/ data after a split-policy bump."""

    def _spec(self):
        return GradientRankingSpec(
            mode=FHS, model_dtype="bfloat16", min_final_segment_tokens=8
        )

    def test_matching_meta_passes(self):
        from steering.gradient_overlay import assert_resume_gradient_matches

        spec = self._spec()
        meta = {
            "gradient_attribution_canonical": canonical_gradient_selection_dict(spec)
        }
        assert_resume_gradient_matches(spec, meta, checkpoint_dir="ckpt")

    def test_absent_on_both_sides_passes(self):
        from steering.gradient_overlay import assert_resume_gradient_matches

        assert_resume_gradient_matches(None, {"step": 1}, checkpoint_dir="ckpt")
        assert_resume_gradient_matches(None, None, checkpoint_dir="ckpt")

    def test_version_bump_refused(self):
        """A checkpoint written under a different split_policy_version must
        not resume (its sel/ data was re-scored)."""
        from steering.gradient_overlay import assert_resume_gradient_matches

        spec = self._spec()
        stale = dict(canonical_gradient_selection_dict(spec))
        stale["split_policy_version"] = stale["split_policy_version"] - 1
        with pytest.raises(GradientAttributionError, match="resume mismatch"):
            assert_resume_gradient_matches(
                spec,
                {"gradient_attribution_canonical": stale},
                checkpoint_dir="ckpt",
            )

    def test_configured_vs_legacy_checkpoint_refused(self):
        from steering.gradient_overlay import assert_resume_gradient_matches

        with pytest.raises(GradientAttributionError, match="resume mismatch"):
            assert_resume_gradient_matches(
                self._spec(), {"step": 1}, checkpoint_dir="ckpt"
            )

    def test_gradient_checkpoint_without_config_refused(self):
        from steering.gradient_overlay import assert_resume_gradient_matches

        meta = {
            "gradient_attribution_canonical": canonical_gradient_selection_dict(
                self._spec()
            )
        }
        with pytest.raises(GradientAttributionError, match="resume mismatch"):
            assert_resume_gradient_matches(None, meta, checkpoint_dir="ckpt")


# ---------------------------------------------------------------------------
# VRAM-leak regression: the forward graph must not outlive score_documents
# ---------------------------------------------------------------------------


class TestNoRetainedGraphLeak:
    """``score_documents`` must not leave the trunk forward graph resident past
    the call.

    Autograd graphs are cyclic, so a retained graph is reclaimed only by
    Python's cyclic GC — never by refcount when the locals drop at return.
    The producer calls ``score_documents`` once per ``_flush`` batch, so a
    graph that survives the call piles up between GC passes and climbs to OOM
    (the PR #528 4090 leak: monotonic 2.7 -> 24 GiB). The fix frees the one
    shared trunk on the LAST unit-group (``retain_graph=False``). This guards
    it on CPU, hardware-independently: we assert the saved activations are
    released EAGERLY when the call returns, WITHOUT forcing ``gc.collect()``.
    """

    def test_forward_graph_freed_eagerly(self):
        model = tiny_gemma2(softcap=True)
        sae = make_tiny_sae("standard")
        # Params/buffers are held by the model regardless of graph retention;
        # only intermediate ACTIVATIONS signal a leak.
        owned = {id(p) for p in model.parameters()} | {id(b) for b in model.buffers()}
        owned |= {id(p) for p in sae.parameters()} | {id(b) for b in sae.buffers()}

        # >= 2 docs over several segments => multiple unit-groups => multiple
        # trunk-VJP calls, exercising the "retain for all but the last" path.
        docs = [seeded_ids(40, seed=i) for i in range(3)]
        segs = [fixed_segments(40, 8) for _ in docs]
        fids = [[full_fids() for _ in s] for s in segs]

        refs: list[weakref.ref] = []

        def pack(x: t.Tensor) -> t.Tensor:
            if id(x) not in owned and x.is_floating_point():
                refs.append(weakref.ref(x))
            return x

        collector = GradientAttributionCollector(
            model, sae, layer=TINY_LAYER, mode=FHS, min_final_segment_tokens=8
        )
        with t.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            collector.score_documents(docs, segs, fids)

        assert refs, "saved-tensor hook captured nothing — guard would be vacuous"
        # No gc.collect(): the contract is EAGER release, not eventual collection.
        alive = [r for r in refs if r() is not None]
        assert not alive, (
            f"{len(alive)}/{len(refs)} forward-graph activations still resident "
            f"after score_documents returned — the trunk graph leaked and would "
            f"accumulate one per producer _flush batch until OOM."
        )

    def test_scores_are_detached_even_with_unfrozen_sae(self):
        """Returned score rows must carry NO grad_fn.

        The REAL 4090 leak: an unfrozen SAE (``SAE.from_pretrained`` does not
        freeze) made ``_score_read``'s decoder pull-back ``grad_rows @
        W_dec[fids].T`` build an autograd graph, so each returned row escaped
        with a grad_fn. The producer writes every row into the persistent
        ``flat_scores`` buffer (``flat_scores[f_lo:f_hi] = row``), and assigning
        a grad-carrying tensor into a buffer PINS its saved tensors for the
        buffer's whole lifetime (the entire shard) — ~23 MiB of live GPU
        tensors per batch, climbing to OOM. The collector must freeze the SAE
        so scores are pure data. (The within-call detector above missed this:
        it discarded the rows instead of writing them into a buffer.)
        """
        model = tiny_gemma2(softcap=True)
        sae = make_tiny_sae("standard")
        # Simulate a caller that hands over an UNFROZEN SAE (the producer's
        # SAE.from_pretrained path).
        for p in sae.parameters():
            p.requires_grad_(True)

        collector = GradientAttributionCollector(
            model, sae, layer=TINY_LAYER, mode=FHS, min_final_segment_tokens=8
        )
        # The collector must re-freeze it at the source.
        assert not any(p.requires_grad for p in sae.parameters()), (
            "collector must freeze the SAE — an unfrozen W_dec makes scores "
            "carry a grad_fn"
        )

        docs = [seeded_ids(40, seed=1)]
        segs = [fixed_segments(40, 8)]
        fids = [[full_fids() for _ in segs[0]]]
        out = collector.score_documents(docs, segs, fids)
        for doc_rows in out:
            for row in doc_rows:
                assert not row.requires_grad and row.grad_fn is None, (
                    "score row carries a grad_fn — writing it into the "
                    "persistent flat_scores buffer pins its saved GPU tensors "
                    "for the whole shard (the cross-batch VRAM leak)."
                )


class TestBatchTokenization:
    """#10: the producer batch-tokenizes its docs (one tokenizer call per chunk)
    instead of one per doc. The batched path must be BIT-IDENTICAL to per-doc
    _truncate_to_training_window, or the per-doc alignment asserts would fire."""

    class _FakeTok:
        """Deterministic per-text encoding: ids = [0 .. len(text)-1]. Batch input
        returns one such list per text (HF fast-tokenizer shape, padding=False),
        with NO cross-text interaction — the property the batching relies on."""

        def __call__(self, text, padding=False):
            if isinstance(text, list):
                return {"input_ids": [list(range(len(s))) for s in text]}
            return {"input_ids": list(range(len(text)))}

    def test_bit_identical_to_per_text(self):
        from steering.extract_features import (
            _batch_truncate_to_training_window,
            _truncate_to_training_window,
        )

        tok = self._FakeTok()
        # mixed: short, empty, over-limit (truncated), exact-limit, trailing empty
        texts = ["abc", "", "x" * 50, "de", "y" * 8, "", "f" * 9]
        cw, mc = 8, 100  # limit = min(8, 100) = 8
        ref = [_truncate_to_training_window(t, tok, cw, mc) for t in texts]
        for chunk in (1, 2, 3, 7, 100):  # chunk boundaries incl. < and > len
            got = _batch_truncate_to_training_window(
                texts, tok, cw, mc, chunk_size=chunk
            )
            assert got == ref, f"batched != per-text at chunk_size={chunk}"
        # the interesting cases actually exercised:
        assert ref[2] == (list(range(8)), 50)  # truncated to the limit
        assert ref[1] == ([], 0) and ref[5] == ([], 0)  # empties, no tokenize
        assert ref[4] == (list(range(8)), 8)  # exact-limit kept whole
