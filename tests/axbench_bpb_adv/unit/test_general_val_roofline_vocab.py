"""Vocab-restriction unit tests for roofline general-val selection.

Two independent restriction sites, tested in two sections below:

* (issue #695 PR-1 task 2) ``derive_real_output_signal`` / ``_select_signal_indices`` /
  ``_top_signal_indices_for_bucket`` restricted to a compacted checkpoint's allowed
  feature vocabulary ``V`` -- the doc's own OUTPUT-WINDOW SIGNAL selection.

* (issue #720 PR-2 Task 2) ``build_general_val_doc_member``'s FIXED per-segment
  CONTEXT bags restricted to ``V``, mirroring ``teacher_force.py:306-340``'s
  drop/warn/raise semantics exactly (via the shared
  ``vocab_restriction.restrict_segment_conditions_to_vocab`` helper). This is a
  DIFFERENT restriction site from the signal selection above: the segments come
  from the context cache (pre-extracted, unrelated to the output-window signal),
  and the ``output_bag`` parameter (the method's own -- already restricted
  upstream) is never touched here.

Mirrors the non-vacuous teeth-test pattern used for feature-DiffMean's
``_select_bag`` (``tests/axbench_bpb_adv/unit/test_feature_diffmean.py``): a
global-top candidate OUTSIDE ``V`` must be EXCLUDED and a lower-ranked in-``V``
candidate chosen instead -- not merely "every returned id happens to be in V".
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.axbench_bpb_adv import general_val_roofline
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.general_val_roofline import (
    GeneralValSignalBudget,
    build_general_val_doc_member,
    derive_real_output_signal,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer


class _FakeTokenizer:
    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(chr(65 + int(token_id) % 26) for token_id in token_ids)


def _doc() -> Any:
    return SimpleNamespace(
        sample_id="general_val_vocab_0",
        prefix_tokens=[0, 1, 2],
        output_tokens=[3, 4],
        prefix_text="ABC",
    )


def _percentile_table(n_features: int) -> t.Tensor:
    """Uniform ``[n_features, 99]`` table: percentile ranking == magnitude ranking."""
    row = t.linspace(1.0 / 99.0, 1.0, 99, dtype=t.float64)
    return row.unsqueeze(0).repeat(n_features, 1)


# ─────────────────────────────────────────────────────────────────────────────
# `_top_signal_indices_for_bucket` -- the lowest-level selection primitive
# ─────────────────────────────────────────────────────────────────────────────


def test_top_signal_indices_restricts_ranking_to_allowed_vocabulary() -> None:
    """The non-vacuous core: a high-percentile candidate OUTSIDE `V` is EXCLUDED
    and a lower-percentile in-`V` candidate is chosen instead."""
    fids = t.tensor([10, 20, 30, 40], dtype=t.int64)
    buckets = t.tensor([0, 0, 0, 0], dtype=t.int64)
    percentiles = t.tensor([50.0, 90.0, 10.0, 99.0], dtype=t.float64)
    allowed = t.tensor([10, 20, 30], dtype=t.int64)  # V excludes fid 40 (idx 3)

    unrestricted = general_val_roofline._top_signal_indices_for_bucket(
        fids, buckets, percentiles, bucket=0, limit=1, allowed_ids=None,
    )
    assert unrestricted == [3]  # fid 40 (idx3, pct=99.0) is the global top

    restricted = general_val_roofline._top_signal_indices_for_bucket(
        fids, buckets, percentiles, bucket=0, limit=1, allowed_ids=allowed,
    )
    # (a) the excluded index (3, fid 40) never appears.
    assert 3 not in restricted
    # (b) the V-internal top (idx 1, fid 20, pct=90.0) wins instead of the
    # global top (idx 3, fid 40, pct=99.0).
    assert restricted == [1]
    assert restricted != unrestricted


def test_top_signal_indices_v_smaller_than_limit_returns_all_of_v_no_crash() -> None:
    """|V| < limit -> return all of V's in-bucket candidates, no pad/crash."""
    fids = t.tensor([1, 2, 3, 4], dtype=t.int64)
    buckets = t.tensor([0, 0, 0, 0], dtype=t.int64)
    percentiles = t.tensor([10.0, 90.0, 50.0, 30.0], dtype=t.float64)
    allowed = t.tensor([2, 4], dtype=t.int64)  # only 2 candidates in V

    selected = general_val_roofline._top_signal_indices_for_bucket(
        fids, buckets, percentiles, bucket=0, limit=10, allowed_ids=allowed,
    )
    assert sorted(selected) == [1, 3]  # idx1 (fid2), idx3 (fid4) -- ALL of V, no crash
    assert {int(fids[i].item()) for i in selected} <= {2, 4}


def test_top_signal_indices_empty_vocabulary_returns_empty_no_crash() -> None:
    """|V| == 0 -> no candidates selected, not a crash."""
    fids = t.tensor([1, 2, 3], dtype=t.int64)
    buckets = t.tensor([0, 0, 0], dtype=t.int64)
    percentiles = t.tensor([10.0, 90.0, 50.0], dtype=t.float64)
    allowed = t.tensor([], dtype=t.int64)

    selected = general_val_roofline._top_signal_indices_for_bucket(
        fids, buckets, percentiles, bucket=0, limit=2, allowed_ids=allowed,
    )
    assert selected == []


def test_top_signal_indices_allowed_ids_none_is_a_no_op() -> None:
    """``allowed_ids=None`` is unrestricted -- byte-identical to omitting it."""
    fids = t.tensor([1, 2, 3], dtype=t.int64)
    buckets = t.tensor([0, 0, 0], dtype=t.int64)
    percentiles = t.tensor([10.0, 90.0, 50.0], dtype=t.float64)

    explicit_none = general_val_roofline._top_signal_indices_for_bucket(
        fids, buckets, percentiles, bucket=0, limit=2, allowed_ids=None,
    )
    omitted = general_val_roofline._top_signal_indices_for_bucket(
        fids, buckets, percentiles, bucket=0, limit=2,
    )
    assert explicit_none == omitted == [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# `_select_signal_indices` -- combines positive/negative partitions
# ─────────────────────────────────────────────────────────────────────────────


def test_select_signal_indices_restricts_both_partitions_to_allowed_vocabulary() -> None:
    fids = t.tensor([10, 20, 30, 40, 50], dtype=t.int64)
    buckets = t.tensor([0, 0, 1, 1, 1], dtype=t.int64)
    percentiles = t.tensor([50.0, 99.0, 20.0, 95.0, 40.0], dtype=t.float64)
    # V excludes fid 20 (idx1, global positive-bucket top) and fid 40 (idx3,
    # global negative-bucket top).
    allowed = t.tensor([10, 30, 50], dtype=t.int64)

    selected = general_val_roofline._select_signal_indices(
        fids, buckets, percentiles,
        signal_budget=GeneralValSignalBudget(
            k_pos=1, k_neg=1, positive_bucket=0, negative_bucket=1,
        ),
        allowed_ids=allowed,
    )
    # idx0 (fid10, the ONLY in-V positive-bucket candidate) and idx4 (fid50,
    # the highest-percentile in-V negative-bucket candidate) -- NOT idx1/idx3.
    assert selected == [0, 4]


def test_select_signal_indices_allowed_ids_none_matches_pre_change_snapshot() -> None:
    """``allowed_ids=None`` reproduces the pre-change (unrestricted) selection."""
    fids = t.tensor([30, 10, 20, 40, 50], dtype=t.int64)
    buckets = t.tensor([0, 0, 0, 1, 1], dtype=t.int64)
    percentiles = t.tensor([0.8, 0.7, 0.8, 0.6, 0.9], dtype=t.float64)

    selected = general_val_roofline._select_signal_indices(
        fids, buckets, percentiles,
        signal_budget=GeneralValSignalBudget(
            k_pos=2, k_neg=1, positive_bucket=0, negative_bucket=1,
        ),
        allowed_ids=None,
    )
    # Bucket 0 contributes indices 2 and 0 (tied percentiles at idx0/idx2 break
    # toward the lower feature id 20 < 30); bucket 1 contributes index 4.
    assert selected == [0, 2, 4]


# ─────────────────────────────────────────────────────────────────────────────
# `derive_real_output_signal` -- the public entry point
# ─────────────────────────────────────────────────────────────────────────────


def test_derive_real_output_signal_restricts_selection_to_allowed_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature OUTSIDE `V` that would otherwise win the top slot is excluded
    end-to-end through the public entry point; a lower-ranked in-`V` feature
    is chosen instead."""

    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        # feature 3 fires highest (0.9) -- the unrestricted top-1 -- but sits
        # OUTSIDE V. feature 2 fires second-highest (0.7) and IS in V.
        return t.tensor([[0.3, 0.5, 0.7, 0.9]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    boundaries = t.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=t.float64)
    table = _percentile_table(4)
    budget = GeneralValSignalBudget(k_pos=1, k_neg=0, positive_bucket=0, negative_bucket=1)

    unrestricted = derive_real_output_signal(
        _doc(), base_model=object(), base_sae=object(), sae_layer=8,
        boundaries=boundaries, percentile_table=table, extraction_threshold=0.1,
        signal_budget=budget, tokenizer=_FakeTokenizer(), device=t.device("cpu"),
    )
    assert dict(unrestricted.entries) == {3: 0}  # sanity: global top wins unrestricted

    allowed = t.tensor([0, 1, 2], dtype=t.int64)  # V excludes feature 3
    restricted = derive_real_output_signal(
        _doc(), base_model=object(), base_sae=object(), sae_layer=8,
        boundaries=boundaries, percentile_table=table, extraction_threshold=0.1,
        signal_budget=budget, tokenizer=_FakeTokenizer(), device=t.device("cpu"),
        allowed_ids=allowed,
    )
    restricted_dict = dict(restricted.entries)
    assert set(restricted_dict) <= {0, 1, 2}
    assert 3 not in restricted_dict
    assert restricted_dict == {2: 0}  # V-internal top (feature 2), not feature 3
    assert restricted_dict != dict(unrestricted.entries)


def test_derive_real_output_signal_v_smaller_than_k_selects_only_v(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """|V ∩ bucket| < k_pos -> select only the available in-V features, no crash."""

    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[0.3, 0.5, 0.7, 0.9]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    boundaries = t.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=t.float64)
    table = _percentile_table(4)
    allowed = t.tensor([0], dtype=t.int64)  # only feature 0 is in V

    bag = derive_real_output_signal(
        _doc(), base_model=object(), base_sae=object(), sae_layer=8,
        boundaries=boundaries, percentile_table=table, extraction_threshold=0.1,
        signal_budget=GeneralValSignalBudget(
            k_pos=3, k_neg=0, positive_bucket=0, negative_bucket=1,
        ),
        tokenizer=_FakeTokenizer(), device=t.device("cpu"), allowed_ids=allowed,
    )
    assert dict(bag.entries) == {0: 0}


def test_derive_real_output_signal_empty_after_vocab_filter_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing zero-selected guard doubles as the empty-after-`V`-filter
    behavior: if `V` excludes every surviving candidate, this raises rather
    than inventing an empty bag."""

    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[0.3, 0.5, 0.7, 0.9]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    boundaries = t.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=t.float64)
    table = _percentile_table(4)
    allowed = t.tensor([99], dtype=t.int64)  # V contains no surviving candidate

    with pytest.raises(AxBenchBpbAdvError, match="selected zero features"):
        derive_real_output_signal(
            _doc(), base_model=object(), base_sae=object(), sae_layer=8,
            boundaries=boundaries, percentile_table=table, extraction_threshold=0.1,
            signal_budget=GeneralValSignalBudget(
                k_pos=3, k_neg=3, positive_bucket=0, negative_bucket=1,
            ),
            tokenizer=_FakeTokenizer(), device=t.device("cpu"), allowed_ids=allowed,
        )


def test_derive_real_output_signal_allowed_ids_none_matches_omitted_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``allowed_ids=None`` is byte-identical to omitting the parameter
    entirely -- the legacy no-op contract at the public entry point."""

    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[0.3, 0.5, 0.7, 0.9]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    boundaries = t.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=t.float64)
    table = _percentile_table(4)
    budget = GeneralValSignalBudget(k_pos=1, k_neg=0, positive_bucket=0, negative_bucket=1)
    kwargs: dict[str, Any] = {
        "base_model": object(), "base_sae": object(), "sae_layer": 8,
        "boundaries": boundaries, "percentile_table": table, "extraction_threshold": 0.1,
        "signal_budget": budget, "tokenizer": _FakeTokenizer(), "device": t.device("cpu"),
    }

    omitted = derive_real_output_signal(_doc(), **kwargs)
    explicit_none = derive_real_output_signal(_doc(), allowed_ids=None, **kwargs)
    assert dict(omitted.entries) == dict(explicit_none.entries) == {3: 0}


# ─────────────────────────────────────────────────────────────────────────────
# `build_general_val_doc_member` -- FIXED context segments restricted to V
# (issue #720 PR-2 Task 2). Mirrors teacher_force.py:306-340's drop/warn/raise
# semantics: ids outside V dropped, ONE aggregated logger.warning per call,
# AxBenchBpbError if a segment is fully emptied by the filter. allowed_ids=None
# is a strict no-op. The output_bag (the method's OWN bag, already restricted
# upstream) is never touched by this filter.
# ─────────────────────────────────────────────────────────────────────────────


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


def _context_entries(member: Any) -> list[tuple[tuple[int, int], ...]]:
    """The member's FIXED context bags only -- segment_conditions[-1] is
    always the untouched output/generation bag."""
    return [c.entries for c in member.segment_conditions[:-1]]


class TestBuildGeneralValDocMemberDropsOutOfVocabIds:
    def test_out_of_v_id_dropped_in_v_ids_survive(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(1, 99, 2), buckets=(10, 20, 30), char_start=0, char_end=3
        )
        allowed_ids = t.tensor([1, 2], dtype=t.int64)

        with caplog.at_level(logging.WARNING):
            member = build_general_val_doc_member(
                _doc(),
                segments=(seg,),
                tokenizer=_FakeTokenizer(),
                output_bag=FeatureConditioning.from_dict({9: 0}),
                max_seq_len=16,
                device=t.device("cpu"),
                allowed_ids=allowed_ids,
            )

        assert _context_entries(member) == [((1, 10), (2, 30))]
        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
        assert "dropped 1 " in warnings[0].message

    def test_dropped_count_aggregates_across_segments_into_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """2 segments over a 2-token prefix -- 1 dropped id in the first, 2 in
        the second -- must produce ONE combined warning naming the TOTAL
        (3), not one warning per segment."""
        tok = SimpleTokenizer()
        seg_a = ContextFeatureSegment(
            feature_ids=(1, 50), buckets=(1, 1), char_start=0, char_end=1
        )
        seg_b = ContextFeatureSegment(
            feature_ids=(2, 60, 70), buckets=(1, 1, 1), char_start=1, char_end=2
        )
        allowed_ids = t.tensor([1, 2], dtype=t.int64)

        with caplog.at_level(logging.WARNING):
            build_general_val_doc_member(
                SimpleNamespace(
                    sample_id="general_val_2seg",
                    prefix_tokens=[5, 6],
                    output_tokens=[7, 8],
                ),
                segments=(seg_a, seg_b),
                tokenizer=tok,
                output_bag=FeatureConditioning.from_dict({9: 0}),
                max_seq_len=16,
                device=t.device("cpu"),
                allowed_ids=allowed_ids,
            )

        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one COMBINED warning, got {len(warnings)}"
        assert "dropped 3 " in warnings[0].message

    def test_no_drop_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(1, 2), buckets=(1, 1), char_start=0, char_end=3
        )
        allowed_ids = t.tensor([1, 2], dtype=t.int64)

        with caplog.at_level(logging.WARNING):
            build_general_val_doc_member(
                _doc(),
                segments=(seg,),
                tokenizer=_FakeTokenizer(),
                output_bag=FeatureConditioning.from_dict({9: 0}),
                max_seq_len=16,
                device=t.device("cpu"),
                allowed_ids=allowed_ids,
            )
        assert _warnings(caplog) == []

    def test_output_bag_not_restricted(self) -> None:
        """The output/generation bag is the method's OWN bag -- already
        restricted upstream -- so it must pass through UNCHANGED, even if it
        (synthetically, for this test) carries an id outside allowed_ids."""
        seg = ContextFeatureSegment(
            feature_ids=(1,), buckets=(1,), char_start=0, char_end=3
        )
        output_bag = FeatureConditioning.from_dict({999: 0})
        allowed_ids = t.tensor([1], dtype=t.int64)

        member = build_general_val_doc_member(
            _doc(),
            segments=(seg,),
            tokenizer=_FakeTokenizer(),
            output_bag=output_bag,
            max_seq_len=16,
            device=t.device("cpu"),
            allowed_ids=allowed_ids,
        )
        assert member.segment_conditions[-1] == output_bag


class TestBuildGeneralValDocMemberEmptySegmentRaises:
    def test_segment_fully_out_of_v_raises(self) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(50, 60), buckets=(1, 2), char_start=0, char_end=3
        )
        allowed_ids = t.tensor([1, 2], dtype=t.int64)

        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            build_general_val_doc_member(
                _doc(),
                segments=(seg,),
                tokenizer=_FakeTokenizer(),
                output_bag=FeatureConditioning.from_dict({9: 0}),
                max_seq_len=16,
                device=t.device("cpu"),
                allowed_ids=allowed_ids,
            )

    def test_already_empty_segment_does_not_raise(self) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(), buckets=(), char_start=0, char_end=3
        )
        allowed_ids = t.tensor([1, 2], dtype=t.int64)

        member = build_general_val_doc_member(
            _doc(),
            segments=(seg,),
            tokenizer=_FakeTokenizer(),
            output_bag=FeatureConditioning.from_dict({9: 0}),
            max_seq_len=16,
            device=t.device("cpu"),
            allowed_ids=allowed_ids,
        )
        assert _context_entries(member) == [()]


class TestBuildGeneralValDocMemberLegacyNoOp:
    def test_allowed_ids_none_never_drops_or_raises(self) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(50, 60), buckets=(1, 2), char_start=0, char_end=3
        )

        member = build_general_val_doc_member(
            _doc(),
            segments=(seg,),
            tokenizer=_FakeTokenizer(),
            output_bag=FeatureConditioning.from_dict({9: 0}),
            max_seq_len=16,
            device=t.device("cpu"),
            allowed_ids=None,
        )
        assert _context_entries(member) == [((50, 1), (60, 2))]

    def test_allowed_ids_omitted_matches_explicit_none(self) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(1, 99), buckets=(5, 6), char_start=0, char_end=3
        )
        kwargs: dict[str, Any] = {
            "segments": (seg,),
            "tokenizer": _FakeTokenizer(),
            "output_bag": FeatureConditioning.from_dict({9: 0}),
            "max_seq_len": 16,
            "device": t.device("cpu"),
        }

        omitted = build_general_val_doc_member(_doc(), **kwargs)
        explicit_none = build_general_val_doc_member(_doc(), allowed_ids=None, **kwargs)
        assert omitted.segment_conditions == explicit_none.segment_conditions

    def test_allowed_ids_none_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        seg = ContextFeatureSegment(
            feature_ids=(50, 60), buckets=(1, 2), char_start=0, char_end=3
        )

        with caplog.at_level(logging.WARNING):
            build_general_val_doc_member(
                _doc(),
                segments=(seg,),
                tokenizer=_FakeTokenizer(),
                output_bag=FeatureConditioning.from_dict({9: 0}),
                max_seq_len=16,
                device=t.device("cpu"),
                allowed_ids=None,
            )
        assert _warnings(caplog) == []
