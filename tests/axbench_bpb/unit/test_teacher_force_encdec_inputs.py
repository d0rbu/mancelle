"""Unit tests for the checkpoint-vocabulary (V) filter applied to context-
cache features in ``_build_encdec_seg_map_and_encoder`` (issue #695 PR-1,
"AxBench-BPB on a compacted encoder-decoder checkpoint", piece 3).

Context-cache features (``prefix_segments`` -- threshold-filtered raw SAE
activations extracted from the prompt, the same pipeline used at training)
are NOT pre-selected against a compacted checkpoint's vocabulary V the way
a conditioning method's bag is (see the feature-DiffMean / general-val
restriction added by the prior task in this plan). This module filters
them at the encoder-input boundary:

* ids outside V are DROPPED, with one ``logger.warning`` per prefix naming
  the total dropped count (aggregated across all of that prefix's
  segments -- NOT one warning per segment).
* a segment that had feature ids but is fully emptied BY the filter (every
  one of its ids was outside V) RAISES ``AxBenchBpbError`` -- a
  zero-conditionable-feature segment is an error, not a silently-empty
  one (owner-directed). A segment that was ALREADY empty before filtering
  (no feature ids at all) is unaffected -- that is pre-existing, unrelated
  behaviour, not something "emptied by the filter".
* ``allowed_ids=None`` (the default; legacy/uncompacted checkpoints) is a
  byte-identical no-op.

``TestEndToEndWiring`` proves the filter is actually wired into the two
``teacher_force.py`` production callers (``score_sample`` and the batched
``_pad_encoder_inputs`` via ``score_triple``) -- a primitive that is only
ever invoked with ``allowed_ids=None`` is a silent no-op at runtime, which
is exactly the failure mode these tests guard against. The third caller,
``steering.eval.axbench_bpb_adv.surrogate``, has its own wiring proof in
``tests/axbench_bpb_adv/functional/test_surrogate_parity.py``
(``test_encdec_surrogate_threads_allowed_ids_from_model_feature_vocab``).

``TestAssertEncIdsInVocab`` / ``TestPreForwardVocabAssertWiring`` (piece 5,
a later task in this same plan) test a DIFFERENT, complementary guard:
``assert_enc_ids_in_vocab``, an eager backstop called immediately before
every encoder-decoder forward that catches ANY out-of-V id in a real
encoder slot -- not just the ones the filter above can itself produce --
so a future call site that forgets to restrict its own candidate ids is
still caught loudly, in Python, rather than only by
``FeatureEncoder._to_compact``'s async ``torch._assert_async``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    _build_encdec_seg_map_and_encoder,
    assert_enc_ids_in_vocab,
    score_sample,
    score_triple,
)
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    feature_ids: tuple[int, ...],
    buckets: tuple[int, ...],
    char_start: int = 0,
    char_end: int = 1,
) -> ContextFeatureSegment:
    return ContextFeatureSegment(
        feature_ids=feature_ids, buckets=buckets,
        char_start=char_start, char_end=char_end,
    )


def _warnings(caplog: Any) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


def _make_inputs(
    tok: Any,
    prefix_text: str,
    output_text: str,
    *,
    concept_id: int,
    segments: tuple[ContextFeatureSegment, ...],
) -> ScoringInputs:
    prefix_tokens = encode_prefix(prefix_text, tok)
    output_tokens = encode_output(output_text, tok)
    byte_weights = torch.ones(len(output_tokens), dtype=torch.int64)
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=concept_id, mode=SteeringMode.POSITIVE,
            content_class="concept_rich",
            prefix_text=prefix_text, output_text=output_text,
        ),
        concept_id=concept_id, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_text, prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix_tokens, output_tokens=output_tokens,
        prefix_context_segments=segments,
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=byte_weights, truncated=False,
    )


def _cluster_contract(
    *,
    n_features: int = 4,
    mask_ids: tuple[int, ...] = (0, 1, 2),
    mask_mode: FeatureMaskMode = FeatureMaskMode.whitelist,
    clusters: tuple[tuple[int, ...], ...] = ((0, 1),),
) -> Any:
    """Build a ``SimpleNamespace`` standing in for ``ClusterFeatureContract``.

    Mirrors the oracle's ``TestClusterContract`` fixture pattern: a REAL
    ``FeatureClusterConsolidation.build`` + REAL ``FeatureMask`` + a REAL
    ``FeatureVocab`` (via ``FeatureVocab.from_mask``, which -- per
    ``feature_vocab.py``'s documented invariant, confirmed empirically by
    ``test_from_mask_normalizes_full_coverage_to_none`` below -- can never
    itself produce a full-range/identity vocab; only a genuinely masked
    subset), wrapped in a ``SimpleNamespace`` so tests don't pay the cost of
    constructing full checkpoint pins / statistics identity / SAE identity /
    semantic-JSON fingerprints they don't need. Defaults reproduce the
    oracle's own fixture exactly (``n_features=4``, whitelist ``(0, 1, 2)``,
    one cluster ``(0, 1)``) so the oracle-ported tests below can call this
    with no arguments and get byte-identical behaviour.

    Note: ``checkpoint.feature_vocab`` here is deliberately the MASK-ONLY
    ``FeatureVocab.from_mask`` result (matching the oracle's own fixture),
    not the mask-AND-cluster-aware ``effective_feature_vocab`` a real
    post-#759 checkpoint's persisted vocab would equal (see
    ``cluster_checkpoint_pin.py``'s ``_assert_checkpoint_agreement`` /
    ``steering/conditional/effective_feature_vocab.py``). Every value this
    helper can produce is still a genuine, non-identity ``FeatureVocab``
    (never full-range) -- it is not a byte-precise stand-in for what a
    real, fully-validated ``ClusterFeatureContract`` would carry.
    """
    mask = FeatureMask(mask_mode, mask_ids)
    definition = FeatureClusterDefinition.from_mapping(
        {"schema_version": 1, "clusters": [list(cluster) for cluster in clusters]},
        n_sae_features=n_features,
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_features,
        ),
    )
    vocab = FeatureVocab.from_mask(mask, n_features)
    assert vocab is not None
    return SimpleNamespace(
        checkpoint=SimpleNamespace(feature_vocab=vocab),
        consolidation=FeatureClusterConsolidation.build(resolved, mask),
        feature_mask=mask,
    )


# ---------------------------------------------------------------------------
# Drop non-V ids, warn once per prefix with the (aggregated) dropped count.
# ---------------------------------------------------------------------------


class TestDropsOutOfVocabIds:
    def test_out_of_v_id_dropped_in_v_ids_survive(self, caplog: Any) -> None:
        """[in_V, out_of_V, in_V] -> enc_feature_ids contains ONLY the two
        in_V ids. The out-of-V id must be ACTUALLY absent from the tensor,
        not merely outranked/unselected."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b"), ord("c")]
        seg = _seg(feature_ids=(1, 99, 2), buckets=(10, 20, 30), char_start=0, char_end=3)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = _build_encdec_seg_map_and_encoder(
                tokenizer=tok, full_prefix=full_prefix,
                prefix_segments=(seg,), feature_conditions=None,
                n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
                allowed_ids=allowed_ids,
            )

        slot_0_ids = enc_fids[0, 0, :].tolist()
        assert 99 not in slot_0_ids, (
            f"out-of-V id 99 must be dropped from enc_feature_ids, found in {slot_0_ids}"
        )
        active = enc_mask[0, 0, :].nonzero(as_tuple=True)[0]
        assert set(enc_fids[0, 0, active].tolist()) == {1, 2}

        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
        assert "dropped 1 " in warnings[0].message

    def test_dropped_count_aggregates_across_segments_into_one_warning(self, caplog: Any) -> None:
        """2 segments -- 1 dropped id in the first, 2 in the second -- must
        produce ONE combined warning naming the TOTAL (3), not one warning
        per segment."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b"), ord("c"), ord("d")]
        seg_a = _seg(feature_ids=(1, 50), buckets=(1, 1), char_start=0, char_end=2)
        seg_b = _seg(feature_ids=(2, 60, 70), buckets=(1, 1, 1), char_start=2, char_end=4)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok, full_prefix=full_prefix,
                prefix_segments=(seg_a, seg_b), feature_conditions=None,
                n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
                allowed_ids=allowed_ids,
            )

        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one COMBINED warning, got {len(warnings)}"
        assert "dropped 3 " in warnings[0].message

    def test_no_drop_no_warning(self, caplog: Any) -> None:
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b")]
        seg = _seg(feature_ids=(1, 2), buckets=(1, 1), char_start=0, char_end=2)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok, full_prefix=full_prefix,
                prefix_segments=(seg,), feature_conditions=None,
                n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
                allowed_ids=allowed_ids,
            )
        assert _warnings(caplog) == []

    def test_out_of_v_dropped_in_shifted_branch(self, caplog: Any) -> None:
        """The filter fires identically in the prompt-prefix (shifted)
        branch: the surviving ids land at the shifted slot (index 1, not
        0), and the reserved empty slot 0 stays fully masked."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("p"), ord("p"), ord("a"), ord("b"), ord("c")]
        seg = _seg(feature_ids=(1, 99, 2), buckets=(10, 20, 30), char_start=0, char_end=3)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = _build_encdec_seg_map_and_encoder(
                tokenizer=tok, full_prefix=full_prefix,
                prefix_segments=(seg,), feature_conditions=None,
                n_output=2, n_prompt_surviving=3, device=torch.device("cpu"),
                allowed_ids=allowed_ids,
            )

        assert enc_mask[0, 0, :].eq(False).all().item()  # reserved empty slot untouched
        slot_1_ids = enc_fids[0, 1, :].tolist()
        assert 99 not in slot_1_ids
        active = enc_mask[0, 1, :].nonzero(as_tuple=True)[0]
        assert set(enc_fids[0, 1, active].tolist()) == {1, 2}
        assert len(_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# A segment fully emptied BY the filter raises.
# ---------------------------------------------------------------------------


class TestEmptySegmentRaises:
    def test_segment_fully_out_of_v_raises(self) -> None:
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b")]
        seg = _seg(feature_ids=(50, 60), buckets=(1, 2), char_start=0, char_end=2)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok, full_prefix=full_prefix,
                prefix_segments=(seg,), feature_conditions=None,
                n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
                allowed_ids=allowed_ids,
            )

    def test_one_of_two_segments_fully_out_of_v_still_raises(self) -> None:
        """Fails fast even when a later segment would have been fine."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b"), ord("c"), ord("d")]
        seg_a = _seg(feature_ids=(50,), buckets=(1,), char_start=0, char_end=2)  # all OOV
        seg_b = _seg(feature_ids=(1,), buckets=(1,), char_start=2, char_end=4)   # fine
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok, full_prefix=full_prefix,
                prefix_segments=(seg_a, seg_b), feature_conditions=None,
                n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
                allowed_ids=allowed_ids,
            )

    def test_already_empty_segment_does_not_raise(self) -> None:
        """A segment with NO feature ids to begin with was not "emptied BY
        the filter" -- it was already empty. Must not raise."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b")]
        seg = _seg(feature_ids=(), buckets=(), char_start=0, char_end=2)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok, full_prefix=full_prefix,
            prefix_segments=(seg,), feature_conditions=None,
            n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
            allowed_ids=allowed_ids,
        )
        assert enc_mask[0, 0, :].eq(False).all().item()
        assert enc_fids.shape[1] == 2  # 1 segment + 1 gen slot; no raise


# ---------------------------------------------------------------------------
# allowed_ids=None (legacy / uncompacted) is a byte-identical no-op.
# ---------------------------------------------------------------------------


class TestLegacyNoOp:
    def test_allowed_ids_none_byte_identical_to_direct_reconstruction(self) -> None:
        """Independently rebuilds the expected tensors via the untouched
        public helpers (NOT by re-deriving through the filter code path),
        so this cannot pass merely because the filter happens to no-op
        correctly on its own terms."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b"), ord("c"), ord("d")]
        seg_a = ContextFeatureSegment(feature_ids=(1, 99), buckets=(5, 6), char_start=0, char_end=2)
        seg_b = ContextFeatureSegment(feature_ids=(2,), buckets=(7,), char_start=2, char_end=4)
        prefix_segments = (seg_a, seg_b)
        n_output = 2

        enc_fids, enc_mags, enc_mask, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok, full_prefix=full_prefix,
            prefix_segments=prefix_segments, feature_conditions=None,
            n_output=n_output, n_prompt_surviving=0, device=torch.device("cpu"),
            allowed_ids=None,
        )

        expected_conditions = [
            {int(fid): int(bkt) for fid, bkt in zip(seg.feature_ids, seg.buckets, strict=True)}
            for seg in prefix_segments
        ] + [{}]
        expected_fids, expected_mags, expected_mask = build_segmented_feature_encoder_inputs(
            expected_conditions, torch.device("cpu"),
        )
        expected_seg_map = build_multi_segment_seg_map(
            tokenizer=tok, prefix_tokens=full_prefix,
            prefix_segments=prefix_segments, n_output=n_output,
            generation_slot=len(prefix_segments),
        )

        assert torch.equal(enc_fids, expected_fids)
        assert torch.equal(enc_mags, expected_mags)
        assert torch.equal(enc_mask, expected_mask)
        assert torch.equal(seg_map, expected_seg_map)
        assert gen_slot == len(prefix_segments)

    def test_allowed_ids_omitted_matches_explicit_none(self) -> None:
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b")]
        seg = _seg(feature_ids=(1, 99), buckets=(5, 6), char_start=0, char_end=2)

        r_omitted = _build_encdec_seg_map_and_encoder(
            tokenizer=tok, full_prefix=full_prefix,
            prefix_segments=(seg,), feature_conditions=None,
            n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
        )
        r_explicit_none = _build_encdec_seg_map_and_encoder(
            tokenizer=tok, full_prefix=full_prefix,
            prefix_segments=(seg,), feature_conditions=None,
            n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
            allowed_ids=None,
        )
        for a, b in zip(r_omitted[:4], r_explicit_none[:4], strict=True):
            assert torch.equal(a, b)
        assert r_omitted[4] == r_explicit_none[4]

    def test_allowed_ids_none_never_raises_regardless_of_ids(self) -> None:
        """A segment whose ids would ALL be dropped under some V must NOT
        raise when allowed_ids=None -- the raise is gated on an actual V,
        never on some intrinsic property of the segment's ids."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b")]
        seg = _seg(feature_ids=(50, 60), buckets=(1, 2), char_start=0, char_end=2)

        _build_encdec_seg_map_and_encoder(
            tokenizer=tok, full_prefix=full_prefix,
            prefix_segments=(seg,), feature_conditions=None,
            n_output=1, n_prompt_surviving=0, device=torch.device("cpu"),
            allowed_ids=None,
        )  # must not raise


# ---------------------------------------------------------------------------
# Clustered-checkpoint packing (PR #739 Track F1b). The oracle's own
# TestClusterContract coverage stops at one case (a multi-member cluster
# under a whitelist mask keeping both members -- representative never
# shifts); the fixture matrix below extends it per the F1b plan's binding
# Test Design section: cluster_contract=None backward-compat is covered in
# tests/axbench_bpb/functional/test_runtime_state_encdec_attributes.py
# (RuntimeState's own reject/accept fork), mask-shifted-representative /
# singleton-after-mask / inert-cluster each re-run under BOTH effective-
# vocab regimes (compacted-subset default vs. full-width-checkpoint-vocab),
# and the generation-slot conditioning path (not just prefix segments) also
# gets an explicit reject case.
# ---------------------------------------------------------------------------


class TestClusterContract:
    def test_exact_checkpoint_vocab_accepts_representative_only_inputs(self) -> None:
        contract = _cluster_contract()
        allowed_ids = torch.tensor(
            contract.checkpoint.feature_vocab.original_ids,
            dtype=torch.int64,
        )

        enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(
                        feature_ids=(0, 2),
                        buckets=(0, 0),
                        char_start=0,
                        char_end=2,
                    ),
                ),
                feature_conditions={2: 0},
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )
        )

        active = enc_mask[0, 0].nonzero(as_tuple=True)[0]
        assert enc_fids[0, 0, active].tolist() == [0, 2]

    def test_checkpoint_vocab_mismatch_raises_before_packing(self) -> None:
        contract = _cluster_contract()

        with pytest.raises(AxBenchBpbError, match="vocabulary differs"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(0,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=torch.tensor([0, 1], dtype=torch.int64),
                cluster_contract=contract,
            )

    def test_checkpoint_feature_vocab_none_raises_before_packing(self) -> None:
        """DIRECT regression coverage for the ``contract_vocab is None``
        half of teacher_force.py's ``if contract_vocab is None or
        actual_vocab != contract_vocab.original_ids: raise`` (the FIRST
        sub-condition of the ``or`` -- the SECOND is what
        ``test_checkpoint_vocab_mismatch_raises_before_packing`` above
        already covers).

        This state is UNREACHABLE for any real, fully-validated
        ``ClusterFeatureContract``: every static cluster has >= 2 members
        (``feature_clusters.py``'s cluster-membership validation), so
        ``FeatureClusterConsolidation.build`` always consolidates at least
        one id away for any surviving cluster, so the checkpoint's
        persisted vocab can never be full-range/identity (confirmed
        directly by ``test_from_mask_normalizes_full_coverage_to_none``:
        the ONLY way ``FeatureVocab.from_mask``/``effective_feature_vocab``
        ever return ``None`` is for a full-coverage input, and a
        full-coverage mask+cluster combination can never survive
        consolidation intact). ``cluster_checkpoint_pin.py``'s
        ``_assert_checkpoint_agreement`` additionally enforces
        ``checkpoint.feature_vocab == effective_feature_vocab(
        consolidation, n)`` at contract-construction time, which can never
        be ``None`` either.

        Still, the branch is real, load-bearing, fail-loud defensive code.
        This pins its behaviour directly with a raw ``SimpleNamespace``
        double whose ``checkpoint.feature_vocab`` is forced to ``None`` --
        the oracle's own idiom for tests that don't need a fully-realized,
        constructible contract -- rather than relying on constructibility,
        so a future change to the cluster-definition invariant (e.g. if
        singleton "clusters" ever became legal) can't silently leave this
        branch reachable-but-untested."""
        contract = _cluster_contract()
        contract.checkpoint.feature_vocab = None

        with pytest.raises(AxBenchBpbError, match="vocabulary differs"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(0,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=torch.tensor([0, 1], dtype=torch.int64),
                cluster_contract=contract,
            )

    def test_from_mask_normalizes_full_coverage_to_none(self) -> None:
        """Documents the invariant behind the "vocab-size-agnostic" cases
        below (``test_mask_shifted_representative`` etc.) and behind
        ``test_checkpoint_feature_vocab_none_raises_before_packing`` above:
        ``FeatureVocab.from_mask`` can NEVER itself produce a non-``None``
        full-range vocab -- a whitelist covering every id normalizes to
        ``None`` (feature_vocab.py: "a non-``None`` FeatureVocab ALWAYS
        means genuinely compacted"). Combined with the fact that every
        surviving cluster consolidates >= 1 id away (see the ``None``-raise
        test above), this is why NO real ``ClusterFeatureContract`` can
        ever carry a full-width/identity ``checkpoint.feature_vocab`` --
        there are exactly two reachable states, "genuinely compacted
        (narrower than n_sae_features)" and "no persisted vocab at all
        (None)"; "full-width but non-None" is not a legal third state.
        """
        full_whitelist = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2, 3))
        assert FeatureVocab.from_mask(full_whitelist, 4) is None

    # ------------------------------------------------------------------
    # Vocab-SIZE-agnostic packing (NOT "compacted vs. uncompacted" --
    # review follow-up to task-f1b-report.md's original framing, which
    # overstated what a fabricated full-width/identity checkpoint.
    # feature_vocab could prove: that combination is UNCONSTRUCTIBLE for
    # any real, validated ClusterFeatureContract per the invariant
    # documented on test_checkpoint_feature_vocab_none_raises_before_
    # packing/test_from_mask_normalizes_full_coverage_to_none above. The
    # two regimes below are both genuinely compacted (non-identity, real
    # FeatureVocab.from_mask outputs -- never fabricated), differing ONLY
    # in absolute SIZE (V=3 vs. V=999), to isolate exactly the "does this
    # code hardcode a width assumption" question without ever exercising
    # an impossible state.
    # ------------------------------------------------------------------

    _NARROW_VOCAB = {"n_features": 4, "mask_mode": FeatureMaskMode.whitelist}
    _WIDE_VOCAB = {"n_features": 1000, "mask_mode": FeatureMaskMode.blacklist}

    @pytest.mark.parametrize(
        ("vocab_kwargs", "mask_ids"),
        [(_NARROW_VOCAB, (0, 1, 3)), (_WIDE_VOCAB, (2,))],
        ids=["narrow_vocab", "wide_vocab"],
    )
    def test_mask_shifted_representative(
        self, vocab_kwargs: dict[str, Any], mask_ids: tuple[int, ...],
    ) -> None:
        """Cluster ``[2, 0, 1]`` (unsorted) under a mask excluding id 2:
        the surviving representative is 0 (the first ALLOWED member in
        cluster-definition order), not 2 (the cluster's first-listed
        member). Proves packing uses the *resolved* representative
        (``rep_of``), not raw cluster-definition order -- and that this
        holds regardless of the checkpoint's compacted vocabulary SIZE
        (``narrow_vocab``: whitelist, V=3 of 4; ``wide_vocab``: blacklist,
        V=999 of 1000 -- both exclude exactly id 2, the id this scenario
        needs excluded, and neither is full-width)."""
        contract = _cluster_contract(
            mask_ids=mask_ids,
            clusters=((2, 0, 1),),
            **vocab_kwargs,
        )
        allowed_ids = torch.tensor(
            contract.checkpoint.feature_vocab.original_ids, dtype=torch.int64,
        )

        # The shifted representative (0) is accepted.
        enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(0,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )
        )
        active = enc_mask[0, 0].nonzero(as_tuple=True)[0]
        assert enc_fids[0, 0, active].tolist() == [0]

        # The consolidated-away sibling (1) is rejected -- it is no longer
        # its own representative once 0 survives in its place.
        with pytest.raises(AxBenchBpbError, match="representative-only"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(1,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )

    @pytest.mark.parametrize(
        ("vocab_kwargs", "mask_ids"),
        [(_NARROW_VOCAB, (0, 2, 3)), (_WIDE_VOCAB, (1,))],
        ids=["narrow_vocab", "wide_vocab"],
    )
    def test_singleton_after_mask(
        self, vocab_kwargs: dict[str, Any], mask_ids: tuple[int, ...],
    ) -> None:
        """Cluster ``[0, 1]`` under a mask excluding id 1 leaves 0 as a
        legitimately-unconsolidated singleton (only one survivor, so no
        consolidation happens at all). Proves the representative-only
        assertion does not spuriously reject a survivor merely because it
        was cluster-defined alongside a now-excluded sibling; the excluded
        sibling itself is still correctly rejected (by the mask, not by any
        consolidation logic) -- across two differently-SIZED compacted
        vocabs (V=3 of 4 vs. V=999 of 1000), both excluding exactly id 1."""
        contract = _cluster_contract(
            mask_ids=mask_ids,
            clusters=((0, 1),),
            **vocab_kwargs,
        )
        allowed_ids = torch.tensor(
            contract.checkpoint.feature_vocab.original_ids, dtype=torch.int64,
        )

        enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(0,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )
        )
        active = enc_mask[0, 0].nonzero(as_tuple=True)[0]
        assert enc_fids[0, 0, active].tolist() == [0]

        with pytest.raises(AxBenchBpbError, match="representative-only"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(1,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )

    @pytest.mark.parametrize(
        ("vocab_kwargs", "mask_ids"),
        [(_NARROW_VOCAB, (0, 3)), (_WIDE_VOCAB, (1, 2))],
        ids=["narrow_vocab", "wide_vocab"],
    )
    def test_inert_cluster(
        self, vocab_kwargs: dict[str, Any], mask_ids: tuple[int, ...],
    ) -> None:
        """A cluster ``[1, 2]`` fully outside the mask (both members
        excluded) never resolves a representative at all -- neither id is
        marked consolidated-away (there is nothing to consolidate INTO).
        Proves an inert cluster definition (i) does not interfere with
        packing an unrelated, mask-allowed id, and (ii) does not silently
        admit its own (still mask-disallowed) members -- across two
        differently-SIZED compacted vocabs, both excluding exactly ids
        1 and 2."""
        contract = _cluster_contract(
            mask_ids=mask_ids,
            clusters=((1, 2),),
            **vocab_kwargs,
        )
        allowed_ids = torch.tensor(
            contract.checkpoint.feature_vocab.original_ids, dtype=torch.int64,
        )

        enc_fids, _enc_mags, enc_mask, _seg_map, _gen_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(0,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )
        )
        active = enc_mask[0, 0].nonzero(as_tuple=True)[0]
        assert enc_fids[0, 0, active].tolist() == [0]

        with pytest.raises(AxBenchBpbError, match="representative-only"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(1,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions=None,
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )

    def test_generation_slot_conditioning_rejects_consolidated_away_id(self) -> None:
        """The oracle's own coverage only ever passes a VALID generation-slot
        ``feature_conditions`` dict; no test proves the SEPARATE generation-
        slot ``assert_representatives_only``/``assert_no_masked`` pair
        (teacher_force.py's second such block) can actually reject. A
        consolidated-away id placed there must raise, just like in a prefix
        segment."""
        contract = _cluster_contract(
            mask_ids=(0, 1, 3),  # excludes 2
            clusters=((2, 0, 1),),
        )
        allowed_ids = torch.tensor(
            contract.checkpoint.feature_vocab.original_ids, dtype=torch.int64,
        )

        with pytest.raises(AxBenchBpbError, match="generation bag"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=SimpleTokenizer(),
                full_prefix=[0, ord("a"), ord("b")],
                prefix_segments=(
                    _seg(feature_ids=(0,), buckets=(0,), char_start=0, char_end=2),
                ),
                feature_conditions={1: 0},  # consolidated-away sibling of 0
                n_output=1,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
                allowed_ids=allowed_ids,
                cluster_contract=contract,
            )


# ---------------------------------------------------------------------------
# End-to-end wiring: score_sample / score_triple (-> _pad_encoder_inputs)
# must thread allowed_ids from target_bundle.model.feature_vocab, not just
# leave the primitive ready-but-unwired (a Task-2 review finding: the bag
# restriction was fully implemented+tested, but its ONE production caller
# never actually passed a real V -- silent no-op at runtime).
# ---------------------------------------------------------------------------


class _FeatureVocabSentinel:
    """Stands in for ``steering.conditional.feature_vocab.FeatureVocab``:
    ``feature_vocab_allowed_ids`` only checks ``model.feature_vocab is
    None``, so any non-None sentinel signals "compacted"."""


class _VocabAwareCountingEncdecModel:
    """Deterministic encdec stub whose loss depends only on the COUNT of
    active (masked-in) features routed to each decoder position -- so
    dropping an out-of-V id from a segment (fewer active features at that
    slot) changes the loss, while an id that was never there to begin with
    does not. No parameters; exercises the real score_sample / score_triple
    encdec kernel end to end (mirrors
    test_general_val_roofline_orchestration.py's ``_CountingEncdecModel``).
    """

    # Explicit ``Any`` (mirrors test_surrogate_parity.py's ``_ToyEncDec``):
    # Task 4's legacy-no-op wiring test flips this back to ``None`` post-
    # construction to stand in for an uncompacted checkpoint without a
    # second stub class; ty would otherwise narrow the attribute's type to
    # ``_FeatureVocabSentinel`` from the constructor assignment below.
    feature_vocab: Any

    def __init__(self, allowed_ids: torch.Tensor) -> None:
        self.transformer = SimpleNamespace(h=[])  # score_triple's teardown fingerprint
        self.feature_vocab = _FeatureVocabSentinel()
        self.encoder = SimpleNamespace(feature_vocab_ids=allowed_ids)
        # Task 4: counts real __call__ invocations so wiring tests can prove
        # the eager pre-forward assert raises BEFORE the forward runs, not
        # merely "somewhere during scoring" (e.g. the model's own guard).
        self.call_count = 0

    def eval(self) -> _VocabAwareCountingEncdecModel:
        return self

    def train(self, mode: bool = True) -> _VocabAwareCountingEncdecModel:
        return self

    def __call__(
        self,
        dec_input: torch.Tensor,
        dec_target: torch.Tensor,
        *,
        enc_feature_ids: torch.Tensor,
        enc_magnitudes: torch.Tensor,
        enc_mask: torch.Tensor,
        seg_map: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        self.call_count += 1
        del dec_input, dec_target, enc_feature_ids, enc_magnitudes
        assert loss_reduction == "none"
        counts = enc_mask.sum(dim=-1).to(torch.float32)  # (B, n_seg)
        per_pos = torch.gather(counts, 1, seg_map)  # (B, T-1)
        return (0.1 * per_pos + 0.05).reshape(-1)  # strictly positive, finite


class TestEndToEndWiring:
    """A compacted-checkpoint stub (`.feature_vocab` non-None,
    `.encoder.feature_vocab_ids = allowed`) fed a context-cache segment
    carrying an out-of-V id must produce the SAME score as the identical
    call with that segment already pre-filtered down to the in-V id --
    which is only possible if `allowed_ids` was actually threaded from the
    model into `_build_encdec_seg_map_and_encoder` at runtime."""

    def test_score_sample_threads_allowed_ids_from_model_feature_vocab(self) -> None:
        tok = SimpleTokenizer()
        allowed = torch.tensor([1], dtype=torch.int64)  # V excludes 99
        model = _VocabAwareCountingEncdecModel(allowed)
        bundle = TargetModelBundle(model=model, tokenizer=tok, max_seq_len=32, target_name="encdec")

        seg_with_oov = _seg(feature_ids=(1, 99), buckets=(2, 3), char_start=0, char_end=3)
        seg_prefiltered = _seg(feature_ids=(1,), buckets=(2,), char_start=0, char_end=3)
        inputs_with_oov = _make_inputs(tok, "abc", "de", concept_id=1, segments=(seg_with_oov,))
        inputs_prefiltered = _make_inputs(tok, "abc", "de", concept_id=2, segments=(seg_prefiltered,))

        row_with_oov = score_sample(
            inputs=inputs_with_oov, target_bundle=bundle,
            intervention=Intervention.ours(feature_conditions={}),
            device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
        )
        row_prefiltered = score_sample(
            inputs=inputs_prefiltered, target_bundle=bundle,
            intervention=Intervention.ours(feature_conditions={}),
            device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
        )
        # If allowed_ids were NOT threaded (silent no-op), "with_oov" would
        # have one MORE active feature at that segment's slot than
        # "prefiltered" (99 stays in), giving a DIFFERENT BPB. Equality
        # here is only possible if the OOV id was actually dropped upstream.
        assert row_with_oov.bpb_intervened == pytest.approx(row_prefiltered.bpb_intervened)

    def test_score_triple_threads_allowed_ids_from_model_feature_vocab(self) -> None:
        """Same proof through the batched path (`_pad_encoder_inputs`)."""
        tok = SimpleTokenizer()
        allowed = torch.tensor([1], dtype=torch.int64)
        model = _VocabAwareCountingEncdecModel(allowed)
        bundle = TargetModelBundle(model=model, tokenizer=tok, max_seq_len=32, target_name="encdec")

        seg_with_oov = _seg(feature_ids=(1, 99), buckets=(2, 3), char_start=0, char_end=3)
        seg_prefiltered = _seg(feature_ids=(1,), buckets=(2,), char_start=0, char_end=3)
        inputs_with_oov = _make_inputs(tok, "abc", "de", concept_id=1, segments=(seg_with_oov,))
        inputs_prefiltered = _make_inputs(tok, "abc", "de", concept_id=2, segments=(seg_prefiltered,))

        rows = score_triple(
            samples=[inputs_with_oov, inputs_prefiltered],
            target_bundle=bundle,
            intervention=Intervention.ours(feature_conditions={}),
            device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
            device_batch_size=2,
        )
        assert rows[0].bpb_intervened == pytest.approx(rows[1].bpb_intervened)

    def test_score_sample_drops_out_of_v_id_and_raises_when_fully_emptied(self) -> None:
        """A stronger end-to-end variant of the empty-segment raise: on a
        compacted model, a segment whose only feature is out-of-V raises
        through the REAL score_sample call path, not just the private
        helper called directly."""
        tok = SimpleTokenizer()
        allowed = torch.tensor([1], dtype=torch.int64)
        model = _VocabAwareCountingEncdecModel(allowed)
        bundle = TargetModelBundle(model=model, tokenizer=tok, max_seq_len=32, target_name="encdec")
        seg_all_oov = _seg(feature_ids=(99,), buckets=(2,), char_start=0, char_end=3)
        inputs = _make_inputs(tok, "abc", "de", concept_id=1, segments=(seg_all_oov,))

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            score_sample(
                inputs=inputs, target_bundle=bundle,
                intervention=Intervention.ours(feature_conditions={}),
                device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
            )


# ---------------------------------------------------------------------------
# assert_enc_ids_in_vocab: the eager pre-forward V-membership backstop
# (issue #695 PR-1, "AxBench-BPB on a compacted encoder-decoder checkpoint",
# piece 5). Unlike the filter above (which restricts ONLY the context-cache
# `prefix_segments`), this is a correctness backstop that must catch ANY
# out-of-V id in ANY real encoder slot -- including one this module's own
# filter cannot itself produce, such as an unfiltered `intervention.
# feature_conditions` id at the generation slot (see
# `_build_encdec_seg_map_and_encoder`'s docstring: "`feature_conditions` ...
# is NOT filtered here"). ``TestAssertEncIdsInVocab`` unit-tests the helper
# directly; ``TestPreForwardVocabAssertWiring`` proves it is actually wired
# into both teacher_force.py forward call sites (`score_sample`,
# `_score_microbatch` via `score_triple`) and fires EAGERLY -- before the
# model is ever invoked. The third call site,
# ``steering.eval.axbench_bpb_adv.surrogate._teacher_forced_model_forward``,
# has its own wiring proof in
# ``tests/axbench_bpb_adv/functional/test_surrogate_parity.py``
# (``test_encdec_surrogate_raises_before_forward_on_oov_generation_slot_feature``).
# ---------------------------------------------------------------------------


class TestAssertEncIdsInVocab:
    """Direct unit tests of the helper, independent of any seg-map
    construction. Hand-crafted tensors let these exercise scenarios
    `_build_encdec_seg_map_and_encoder`'s own filter cannot itself produce
    (e.g. an OOV id surviving into a real, non-context-cache slot)."""

    def test_oov_id_in_real_slot_raises(self) -> None:
        enc_feature_ids = torch.tensor([[[1, 99]]], dtype=torch.int64)  # (1, 1, 2)
        enc_mask = torch.tensor([[[True, True]]], dtype=torch.bool)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)

    def test_oov_id_in_padding_slot_does_not_raise(self) -> None:
        """99 sits in a masked-OUT (padding) slot -- must be ignored even
        though it is genuinely outside allowed_ids."""
        enc_feature_ids = torch.tensor([[[1, 99]]], dtype=torch.int64)
        enc_mask = torch.tensor([[[True, False]]], dtype=torch.bool)
        allowed_ids = torch.tensor([1], dtype=torch.int64)

        assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)  # must not raise

    def test_pad_value_zero_not_in_allowed_ids_is_ignored(self) -> None:
        """The realistic padding pattern: unused slots literally hold id 0
        (torch.zeros init), and V need not contain 0."""
        enc_feature_ids = torch.zeros((1, 2, 3), dtype=torch.int64)
        enc_feature_ids[0, 0, 0] = 1
        enc_mask = torch.zeros((1, 2, 3), dtype=torch.bool)
        enc_mask[0, 0, 0] = True
        allowed_ids = torch.tensor([1], dtype=torch.int64)  # excludes 0

        assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)  # must not raise

    def test_mixed_real_and_padding_oov_raises_only_for_the_real_one(self) -> None:
        """id 50 (real, masked True) is OOV and must raise; id 60 (padding,
        masked False) is ALSO OOV but must be silently ignored -- proving
        the mask, not mere OOV-ness, gates the raise."""
        enc_feature_ids = torch.tensor([[[1, 50, 60]]], dtype=torch.int64)
        enc_mask = torch.tensor([[[True, True, False]]], dtype=torch.bool)
        allowed_ids = torch.tensor([1], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V") as exc_info:
            assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)
        assert "50" in str(exc_info.value)
        assert "60" not in str(exc_info.value)

    def test_allowed_ids_none_never_raises_even_with_oov_looking_ids(self) -> None:
        enc_feature_ids = torch.tensor([[[1, 99]]], dtype=torch.int64)
        enc_mask = torch.tensor([[[True, True]]], dtype=torch.bool)

        assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, None)  # must not raise

    def test_all_real_ids_in_vocab_does_not_raise(self) -> None:
        enc_feature_ids = torch.tensor([[[1, 2]]], dtype=torch.int64)
        enc_mask = torch.tensor([[[True, True]]], dtype=torch.bool)
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)  # must not raise

    def test_no_real_slots_at_all_does_not_raise(self) -> None:
        """A fully-padded encoder input (e.g. an all-empty conditioning
        slot) has zero True positions in enc_mask -- vacuously fine."""
        enc_feature_ids = torch.zeros((1, 1, 3), dtype=torch.int64)
        enc_mask = torch.zeros((1, 1, 3), dtype=torch.bool)
        allowed_ids = torch.tensor([1], dtype=torch.int64)

        assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)  # must not raise

    def test_message_names_the_offending_ids(self) -> None:
        enc_feature_ids = torch.tensor([[[5, 99, 2]]], dtype=torch.int64)
        enc_mask = torch.tensor([[[True, True, True]]], dtype=torch.bool)
        allowed_ids = torch.tensor([2], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError) as exc_info:
            assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)
        message = str(exc_info.value)
        assert "5" in message
        assert "99" in message

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_safe_cross_device_allowed_ids(self) -> None:
        """Mirrors restrict_to_vocab's C1 cross-device fix (issue #695 PR-1
        review): enc_feature_ids/enc_mask on one device, allowed_ids on
        another, must not raise a device-mismatch RuntimeError -- and must
        still correctly detect the OOV id. Skipped (not failed) on this
        CPU-only env; proves the fix on a GPU box."""
        enc_feature_ids = torch.tensor([[[1, 99]]], dtype=torch.int64, device="cpu")
        enc_mask = torch.tensor([[[True, True]]], dtype=torch.bool, device="cpu")
        allowed_ids = torch.tensor([1], dtype=torch.int64, device="cuda")

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            assert_enc_ids_in_vocab(enc_feature_ids, enc_mask, allowed_ids)


class TestPreForwardVocabAssertWiring:
    """Proves the assert is actually wired into `score_sample` and
    `_score_microbatch` (via `score_triple`), and fires EAGERLY.

    `_build_encdec_seg_map_and_encoder`'s own filter already keeps
    context-cache `prefix_segments` ids within V on every production path,
    so it cannot be used to reach this backstop. `intervention.
    feature_conditions` (the generation slot) is the one real production
    input this filter does NOT restrict (see its docstring) -- an
    out-of-V id placed there is exactly the "missed restriction site"
    class of bug this backstop exists to catch.
    """

    def test_score_sample_raises_before_forward_on_oov_generation_slot_feature(self) -> None:
        tok = SimpleTokenizer()
        allowed = torch.tensor([1], dtype=torch.int64)  # V excludes 99
        model = _VocabAwareCountingEncdecModel(allowed)
        bundle = TargetModelBundle(model=model, tokenizer=tok, max_seq_len=32, target_name="encdec")
        seg = _seg(feature_ids=(1,), buckets=(2,), char_start=0, char_end=3)
        inputs = _make_inputs(tok, "abc", "de", concept_id=1, segments=(seg,))

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            score_sample(
                inputs=inputs, target_bundle=bundle,
                intervention=Intervention.ours(feature_conditions={99: 3}),
                device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
            )
        assert model.call_count == 0, "assert must fire BEFORE the forward, not after"

    def test_score_triple_raises_before_forward_on_oov_generation_slot_feature(self) -> None:
        """Same proof through the batched _score_microbatch path."""
        tok = SimpleTokenizer()
        allowed = torch.tensor([1], dtype=torch.int64)
        model = _VocabAwareCountingEncdecModel(allowed)
        bundle = TargetModelBundle(model=model, tokenizer=tok, max_seq_len=32, target_name="encdec")
        seg = _seg(feature_ids=(1,), buckets=(2,), char_start=0, char_end=3)
        inputs = _make_inputs(tok, "abc", "de", concept_id=1, segments=(seg,))

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            score_triple(
                samples=[inputs],
                target_bundle=bundle,
                intervention=Intervention.ours(feature_conditions={99: 3}),
                device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
                device_batch_size=1,
            )
        assert model.call_count == 0, "assert must fire BEFORE the forward, not after"

    def test_score_sample_legacy_allowed_ids_none_unaffected_by_oov_looking_id(self) -> None:
        """An uncompacted checkpoint (`model.feature_vocab is None`) has no
        V at all: the SAME feature_conditions id that would be "out of V"
        on a compacted model must NOT raise here, and the forward must
        actually run (call_count == 1) -- proving the assert is a true
        no-op on the legacy path, not merely "usually doesn't raise"."""
        tok = SimpleTokenizer()
        # Reuse the vocab-aware stub but flip feature_vocab back to None --
        # feature_vocab_allowed_ids only branches on `is None`, so this is a
        # faithful legacy/uncompacted stand-in without a second model class.
        model = _VocabAwareCountingEncdecModel(torch.tensor([1], dtype=torch.int64))
        model.feature_vocab = None
        bundle = TargetModelBundle(model=model, tokenizer=tok, max_seq_len=32, target_name="encdec")
        seg = _seg(feature_ids=(1,), buckets=(2,), char_start=0, char_end=3)
        inputs = _make_inputs(tok, "abc", "de", concept_id=1, segments=(seg,))

        row = score_sample(
            inputs=inputs, target_bundle=bundle,
            intervention=Intervention.ours(feature_conditions={99: 3}),
            device=torch.device("cpu"), coefficient_str="1.0", method_name="probe",
        )
        assert model.call_count == 1
        assert row.bpb_intervened > 0.0  # _VocabAwareCountingEncdecModel is strictly positive
