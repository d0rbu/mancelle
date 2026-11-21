"""Vocab-restriction unit tests for build_class_margin_members's FIXED context
bags (issue #720 PR-2 Task 2).

A compacted checkpoint's allowed feature vocabulary ``V`` restricts each
context segment's ``{feature_id: bucket}`` conditioning exactly like
``teacher_force._build_encdec_seg_map_and_encoder``'s context-cache filter
(``teacher_force.py:306-340``, the semantic reference this mirrors): ids
outside ``V`` are dropped from each FIXED context bag, ONE aggregated
``logger.warning`` is emitted per call naming the total dropped count (not one
per segment), and a segment that had feature ids but is left with NONE after
the filter raises ``AxBenchBpbError``.  ``allowed_ids=None`` (the default; an
uncompacted checkpoint) is a STRICT no-op.

The restriction is implemented by the shared helper
``steering.eval.axbench_bpb.vocab_restriction.restrict_segment_conditions_to_vocab``
(directly unit-tested in ``tests/steering/test_vocab_restriction.py``); these
tests pin that ``build_class_margin_members`` actually wires it in for its
FIXED context bags (never the empty generation bag, which the GCG optimizer
substitutes into later and which the method's own bag already restricts
upstream).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

_BIG_SEQ_LEN = 100_000


def _seg(
    char_start: int,
    char_end: int,
    *,
    feature_ids: tuple[int, ...],
    buckets: tuple[int, ...],
) -> ContextFeatureSegment:
    return ContextFeatureSegment(
        feature_ids=tuple(feature_ids),
        buckets=tuple(buckets),
        char_start=char_start,
        char_end=char_end,
    )


def _make_prefix(
    *,
    prefix_tokens: list[int] | None = None,
    rich: list[list[int]] | None = None,
    avoid: list[list[int]] | None = None,
    core_prefix_id: str = "vocab_prefix",
) -> BpbAttackPrefix:
    prefix_tokens = prefix_tokens if prefix_tokens is not None else [5, 6]
    rich = rich if rich is not None else [[7, 8]]
    avoid = avoid if avoid is not None else [[9, 10]]
    return BpbAttackPrefix(
        core_prefix_id=core_prefix_id,
        concept_id=0,
        mode=SteeringMode.POSITIVE,
        prefix_text="test",
        prefix_hash="abc123",
        prefix_tokens=prefix_tokens,
        concept_rich_sample_ids=[f"rich_{i}" for i in range(len(rich))],
        concept_avoiding_sample_ids=[f"avoid_{i}" for i in range(len(avoid))],
        concept_rich_output_tokens=rich,
        concept_avoiding_output_tokens=avoid,
    )


def _build(
    prefix: BpbAttackPrefix,
    *,
    prefix_segments: tuple[ContextFeatureSegment, ...],
    allowed_ids: torch.Tensor | None,
    max_seq_len: int = _BIG_SEQ_LEN,
) -> list[Any]:
    return build_class_margin_members(
        prefix,
        prefix_segments=prefix_segments,
        tokenizer=SimpleTokenizer(),
        max_seq_len=max_seq_len,
        device=torch.device("cpu"),
        allowed_ids=allowed_ids,
    )


def _warnings(caplog: Any) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


# ─────────────────────────────────────────────────────────────────────────────
# Compacted (allowed_ids given): OOV ids dropped, in-V ids survive.
# ─────────────────────────────────────────────────────────────────────────────


class TestDropsOutOfVocabIds:
    def test_out_of_v_id_dropped_in_v_ids_survive(self, caplog: Any) -> None:
        """[in_V, out_of_V, in_V] -> the context bag contains ONLY the two
        in_V ids. The out-of-V id must be ACTUALLY absent, not merely
        outranked/unselected -- and the generation bag is untouched."""
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(1, 99, 2), buckets=(10, 20, 30))
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            m = _build(prefix, prefix_segments=(seg,), allowed_ids=allowed_ids)[0]

        assert m.rich.segment_conditions[0].entries == ((1, 10), (2, 30))
        assert m.avoid.segment_conditions[0].entries == ((1, 10), (2, 30))
        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
        assert "dropped 1 " in warnings[0].message

    def test_dropped_count_aggregates_across_segments_into_one_warning(
        self, caplog: Any
    ) -> None:
        """2 segments -- 1 dropped id in the first, 2 in the second -- must
        produce ONE combined warning naming the TOTAL (3), not one warning
        per segment."""
        prefix = _make_prefix(
            prefix_tokens=[5, 6, 3, 7], rich=[[7, 8]], avoid=[[9, 10]]
        )
        seg_a = _seg(0, 2, feature_ids=(1, 50), buckets=(1, 1))
        seg_b = _seg(2, 4, feature_ids=(2, 60, 70), buckets=(1, 1, 1))
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            _build(prefix, prefix_segments=(seg_a, seg_b), allowed_ids=allowed_ids)

        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one COMBINED warning, got {len(warnings)}"
        assert "dropped 3 " in warnings[0].message

    def test_no_drop_no_warning(self, caplog: Any) -> None:
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(1, 2), buckets=(1, 1))
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with caplog.at_level(logging.WARNING):
            _build(prefix, prefix_segments=(seg,), allowed_ids=allowed_ids)
        assert _warnings(caplog) == []

    def test_generation_bag_never_restricted(self) -> None:
        """The generation bag is always empty at build time (the GCG optimizer
        substitutes into it later; the method's own bag is restricted
        upstream) -- the filter only ever touches FIXED context bags."""
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(1,), buckets=(1,))
        allowed_ids = torch.tensor([1], dtype=torch.int64)

        m = _build(prefix, prefix_segments=(seg,), allowed_ids=allowed_ids)[0]
        assert m.rich.segment_conditions[-1].entries == ()
        assert m.avoid.segment_conditions[-1].entries == ()


# ─────────────────────────────────────────────────────────────────────────────
# A segment fully emptied BY the filter raises.
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptySegmentRaises:
    def test_segment_fully_out_of_v_raises(self) -> None:
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(50, 60), buckets=(1, 2))
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            _build(prefix, prefix_segments=(seg,), allowed_ids=allowed_ids)

    def test_one_of_two_segments_fully_out_of_v_still_raises(self) -> None:
        """Fails fast even when the OTHER segment would have been fine."""
        prefix = _make_prefix(
            prefix_tokens=[5, 6, 3, 7], rich=[[7, 8]], avoid=[[9, 10]]
        )
        seg_a = _seg(0, 2, feature_ids=(50,), buckets=(1,))  # all OOV
        seg_b = _seg(2, 4, feature_ids=(1,), buckets=(1,))  # fine
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            _build(prefix, prefix_segments=(seg_a, seg_b), allowed_ids=allowed_ids)

    def test_already_empty_segment_does_not_raise(self) -> None:
        """A segment with NO feature ids to begin with was not "emptied BY
        the filter" -- it was already empty. Must not raise."""
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(), buckets=())
        allowed_ids = torch.tensor([1, 2], dtype=torch.int64)

        m = _build(prefix, prefix_segments=(seg,), allowed_ids=allowed_ids)[0]
        assert m.rich.segment_conditions[0].entries == ()


# ─────────────────────────────────────────────────────────────────────────────
# allowed_ids=None (legacy / uncompacted) is a byte-identical no-op.
# ─────────────────────────────────────────────────────────────────────────────


class TestLegacyNoOp:
    def test_allowed_ids_none_never_drops_or_raises(self) -> None:
        """A segment whose ids would ALL be dropped under some V must NOT be
        emptied/raise when allowed_ids=None -- proves the filter is SKIPPED
        entirely, not merely "V happens to allow everything"."""
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(50, 60), buckets=(1, 2))

        m = _build(prefix, prefix_segments=(seg,), allowed_ids=None)[0]
        assert m.rich.segment_conditions[0].entries == ((50, 1), (60, 2))

    def test_allowed_ids_omitted_matches_explicit_none(self) -> None:
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(1, 99), buckets=(5, 6))

        omitted = build_class_margin_members(
            prefix,
            prefix_segments=(seg,),
            tokenizer=SimpleTokenizer(),
            max_seq_len=_BIG_SEQ_LEN,
            device=torch.device("cpu"),
        )
        explicit_none = _build(prefix, prefix_segments=(seg,), allowed_ids=None)
        assert (
            omitted[0].rich.segment_conditions
            == explicit_none[0].rich.segment_conditions
        )

    def test_allowed_ids_none_no_warning(self, caplog: Any) -> None:
        prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
        seg = _seg(0, 2, feature_ids=(50, 60), buckets=(1, 2))

        with caplog.at_level(logging.WARNING):
            _build(prefix, prefix_segments=(seg,), allowed_ids=None)
        assert _warnings(caplog) == []
