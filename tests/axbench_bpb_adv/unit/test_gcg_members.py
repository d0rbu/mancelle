"""Unit tests for build_class_margin_members in gcg_members.py.

Phase 3 (Bug 2): the builder conditions each prefix under its REAL multi-segment
context — one FIXED bag per context-cache segment (the extracted
``{feature_id: bucket}``) plus one EMPTY generation bag the GCG optimizer
substitutes into — and builds the per-member seg_map with the SAME router the
score stage uses (``build_multi_segment_seg_map``).  These tests pin:

* the shift / -1-masking convention matching teacher_force.py score_sample
  (lines 742-774) so the optimizer scores the SAME tokens the eval scores;
* the per-segment seg_map routing (prefix tokens → their own context slot,
  output → generation slot) after the ``[:, :-1]`` trim;
* the FIXED per-segment context conditioning (context bags carry the segment
  features; only the generation bag is empty);
* the score-time ``max_seq_len`` guard (a prefix + completion the score stage
  would left-truncate RAISES rather than optimizing a divergent layout).

``SimpleTokenizer`` maps each character to one token (id == ord(c)), so a
prefix's decoded text has one char per token and a segment ``[char_start,
char_end)`` covers token positions ``char_start..char_end-1`` — a 1:1 alignment
that makes the seg_map expectations exact.
"""

from __future__ import annotations

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

# A max_seq_len comfortably above every fixture's prefix+completion length; the
# dedicated guard tests below set it small on purpose.
_BIG_SEQ_LEN = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────


def _seg(
    char_start: int,
    char_end: int,
    *,
    feature_ids: tuple[int, ...] = (10,),
    buckets: tuple[int, ...] = (3,),
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
) -> BpbAttackPrefix:
    prefix_tokens = prefix_tokens if prefix_tokens is not None else [5, 6]
    rich = rich if rich is not None else [[7, 8]]
    avoid = avoid if avoid is not None else [[9, 10]]
    return BpbAttackPrefix(
        core_prefix_id="test_prefix",
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
    prefix_segments: tuple[ContextFeatureSegment, ...] | None = None,
    tokenizer: object | None = None,
    max_seq_len: int = _BIG_SEQ_LEN,
):
    """Call build_class_margin_members with the multi-segment API.

    The default single segment spans the WHOLE prefix (chars ``0..n_prefix``
    under SimpleTokenizer's 1-char-per-token mapping) — the common
    single-context case the masking/shape tests reuse.
    """
    tok = tokenizer if tokenizer is not None else SimpleTokenizer()
    if prefix_segments is None:
        prefix_segments = (_seg(0, len(prefix.prefix_tokens)),)
    return build_class_margin_members(
        prefix,
        prefix_segments=prefix_segments,
        tokenizer=tok,
        max_seq_len=max_seq_len,
        device=torch.device("cpu"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core tensor-shape / masking tests (the spec's "known example")
# ─────────────────────────────────────────────────────────────────────────────


def test_basic_one_rich_one_avoid_tensors():
    """prefix=[5,6], rich=[[7,8]], avoid=[[9,10]], ONE context segment.

    Pins the exact dec_input / dec_target (-1 mask) / seg_map required to match
    teacher_force.py score_sample (lines 742-774), AND the Phase-3 conditioning:
    the single context bag carries the segment's features, the generation bag is
    empty.
    """
    prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
    members = _build(
        prefix, prefix_segments=(_seg(0, 2, feature_ids=(10,), buckets=(3,)),)
    )

    assert len(members) == 1, f"expected 1 member, got {len(members)}"
    m = members[0]

    # ── Rich member ────────────────────────────────────────────────────────
    # all_tokens=[5,6,7,8]; dec_input=all_tokens[:-1]=[5,6,7]
    assert torch.equal(m.rich.dec_input, torch.tensor([[5, 6, 7]], dtype=torch.long))
    # dec_target=all_tokens[1:]=[6,7,8]; position 0 (prefix) masked to -1.
    assert torch.equal(m.rich.dec_target, torch.tensor([[-1, 7, 8]], dtype=torch.long))
    # seg_map: prefix positions 0..1 → context slot 0; output → generation_slot=1.
    assert torch.equal(m.rich.seg_map, torch.tensor([[0, 0, 1]], dtype=torch.long))

    # generation_slot is derived (== n_context == 1).
    assert m.rich.generation_slot == 1
    # segment_conditions: [context bag (segment features), empty generation bag].
    assert len(m.rich.segment_conditions) == 2
    assert m.rich.segment_conditions[0].entries == ((10, 3),), (
        "context bag must carry the segment's extracted features"
    )
    assert m.rich.segment_conditions[1].entries == (), (
        "the generation bag (the only one GCG substitutes into) must be empty"
    )

    # ── Avoid member ───────────────────────────────────────────────────────
    assert torch.equal(m.avoid.dec_input, torch.tensor([[5, 6, 9]], dtype=torch.long))
    assert torch.equal(m.avoid.dec_target, torch.tensor([[-1, 9, 10]], dtype=torch.long))
    assert torch.equal(m.avoid.seg_map, torch.tensor([[0, 0, 1]], dtype=torch.long))
    assert m.avoid.generation_slot == 1
    assert m.avoid.segment_conditions[0].entries == ((10, 3),)
    assert m.avoid.segment_conditions[1].entries == ()


def test_scored_tokens_match_teacher_force_convention():
    """dec_target != -1 selects the same positions as build_output_only_mask.

    teacher_force.py build_output_only_mask(n_full_prefix=2, n_output=2):
      mask[n_full_prefix-1:] = True → [False, True, True]
    """
    prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10]])
    m = _build(prefix)[0]

    expected_mask = [False, True, True]
    assert (m.rich.dec_target != -1).squeeze(0).tolist() == expected_mask
    assert (m.avoid.dec_target != -1).squeeze(0).tolist() == expected_mask


def test_seg_map_trimmed_to_dec_input_width():
    """seg_map is the full (1, n_prefix+n_output) router output trimmed [:, :-1].

    The trim (teacher_force.py:774) is load-bearing: the seg_map must align with
    dec_input = all_tokens[:-1]. Pin shape equality so a dropped/duplicated trim
    is caught.
    """
    prefix = _make_prefix(prefix_tokens=[5, 6, 3], rich=[[7, 8, 9]], avoid=[[9, 10]])
    m = _build(prefix)[0]
    assert m.rich.seg_map.shape == m.rich.dec_input.shape
    assert m.avoid.seg_map.shape == m.avoid.dec_input.shape


def test_single_prefix_token():
    """n_prefix=1: no prefix positions masked in dec_target (no -1)."""
    prefix = _make_prefix(prefix_tokens=[5], rich=[[7, 8]], avoid=[[9, 10]])
    m = _build(prefix)[0]
    # all_tokens=[5,7,8]; dec_input=[5,7]; dec_target=[7,8] (0 positions masked).
    assert torch.equal(m.rich.dec_target, torch.tensor([[7, 8]], dtype=torch.long))
    assert torch.equal(m.rich.dec_input, torch.tensor([[5, 7]], dtype=torch.long))
    # T=1+2-1=2; position 0 → context(0), position 1 → generation_slot(1).
    assert torch.equal(m.rich.seg_map, torch.tensor([[0, 1]], dtype=torch.long))


def test_longer_prefix():
    """n_prefix=3: first 2 dec_target positions masked, first 3 seg_map → context."""
    prefix = _make_prefix(prefix_tokens=[5, 6, 3], rich=[[7, 8]], avoid=[[9, 10]])
    m = _build(prefix)[0]
    # all_tokens=[5,6,3,7,8]; dec_input=[5,6,3,7]; dec_target=[-1,-1,7,8]
    assert torch.equal(m.rich.dec_input, torch.tensor([[5, 6, 3, 7]], dtype=torch.long))
    assert torch.equal(
        m.rich.dec_target, torch.tensor([[-1, -1, 7, 8]], dtype=torch.long)
    )
    assert torch.equal(m.rich.seg_map, torch.tensor([[0, 0, 0, 1]], dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Cartesian count tests
# ─────────────────────────────────────────────────────────────────────────────


def test_cartesian_two_rich_one_avoid():
    """2-rich × 1-avoid → 2 ClassMarginMember objects."""
    prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8], [11, 12]], avoid=[[9, 10]])
    assert len(_build(prefix)) == 2


def test_cartesian_one_rich_two_avoid():
    """1-rich × 2-avoid → 2 ClassMarginMember objects."""
    prefix = _make_prefix(prefix_tokens=[5, 6], rich=[[7, 8]], avoid=[[9, 10], [11, 12]])
    assert len(_build(prefix)) == 2


def test_cartesian_three_rich_two_avoid():
    """3-rich × 2-avoid → 6 ClassMarginMember objects."""
    prefix = _make_prefix(
        prefix_tokens=[5, 6],
        rich=[[7, 8], [11, 12], [13, 14]],
        avoid=[[9, 10], [15, 16]],
    )
    assert len(_build(prefix)) == 6


# ─────────────────────────────────────────────────────────────────────────────
# Multi-segment context (Bug 2 core) — the layout the single-segment builder
# used to raise NotImplementedError on now routes + conditions per segment.
# ─────────────────────────────────────────────────────────────────────────────


def test_two_context_segments_route_and_condition():
    """A 2-segment prefix routes each token to its own slot + conditions per segment.

    prefix=[5,6,3,7] (4 chars under SimpleTokenizer), segments [0,2) and [2,4).
    seg_map (before trim) = [0,0,1,1, gen,gen] with generation_slot=2; each
    context slot carries ITS OWN segment's features; the generation bag is empty.
    """
    prefix = _make_prefix(prefix_tokens=[5, 6, 3, 7], rich=[[7, 8]], avoid=[[9, 10]])
    segs = (
        _seg(0, 2, feature_ids=(10, 11), buckets=(3, 4)),
        _seg(2, 4, feature_ids=(20,), buckets=(5,)),
    )
    m = _build(prefix, prefix_segments=segs)[0]

    # generation_slot sits AFTER the two context slots.
    assert m.rich.generation_slot == 2
    # all_tokens=[5,6,3,7,7,8]; dec_input width = 4+2-1 = 5.
    # full seg_map [0,0,1,1,2,2] trimmed [:, :-1] → [0,0,1,1,2].
    assert torch.equal(
        m.rich.seg_map, torch.tensor([[0, 0, 1, 1, 2]], dtype=torch.long)
    )
    # segment_conditions: [seg0 bag, seg1 bag, empty generation bag].
    assert len(m.rich.segment_conditions) == 3
    assert m.rich.segment_conditions[0].entries == ((10, 3), (11, 4))
    assert m.rich.segment_conditions[1].entries == ((20, 5),)
    assert m.rich.segment_conditions[2].entries == ()


def test_three_context_segments_generation_slot_and_length():
    """A 3-segment prefix → generation_slot=3 and 4 conditioning bags."""
    prefix = _make_prefix(
        prefix_tokens=[5, 6, 3, 7, 4, 8], rich=[[7, 8]], avoid=[[9, 10]]
    )
    segs = (
        _seg(0, 2, feature_ids=(10,), buckets=(1,)),
        _seg(2, 4, feature_ids=(20,), buckets=(2,)),
        _seg(4, 6, feature_ids=(30,), buckets=(3,)),
    )
    m = _build(prefix, prefix_segments=segs)[0]
    assert m.rich.generation_slot == 3
    assert len(m.rich.segment_conditions) == 4
    # Prefix tokens route [0,0,1,1,2,2]; the one surviving output position → 3.
    assert torch.equal(
        m.rich.seg_map, torch.tensor([[0, 0, 1, 1, 2, 2, 3]], dtype=torch.long)
    )
    assert m.rich.segment_conditions[3].entries == ()


def test_only_the_generation_bag_is_empty():
    """Every context bag is non-empty; exactly the last (generation) bag is empty.

    Guards against a regression that leaves context slots unconditioned (the
    pre-Bug-2 behavior) or that optimizes into a context slot.
    """
    prefix = _make_prefix(prefix_tokens=[5, 6, 3, 7], rich=[[7, 8]], avoid=[[9, 10]])
    segs = (
        _seg(0, 2, feature_ids=(10,), buckets=(1,)),
        _seg(2, 4, feature_ids=(20,), buckets=(2,)),
    )
    m = _build(prefix, prefix_segments=segs)[0]
    empties = [i for i, c in enumerate(m.rich.segment_conditions) if c.entries == ()]
    assert empties == [m.rich.generation_slot] == [2]


# ─────────────────────────────────────────────────────────────────────────────
# Guards / edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_rich_raises():
    """Empty rich list fails at the typed prefix-record boundary."""
    with pytest.raises(BeartypeCallHintParamViolation, match="concept_rich_sample_ids"):
        _make_prefix(rich=[], avoid=[[9, 10]])


def test_empty_avoid_raises():
    """Empty avoid list fails at the typed prefix-record boundary."""
    with pytest.raises(BeartypeCallHintParamViolation, match="concept_avoiding_sample_ids"):
        _make_prefix(rich=[[7, 8]], avoid=[])


def test_both_empty_raises():
    """Both empty fail at the first typed side boundary."""
    with pytest.raises(BeartypeCallHintParamViolation, match="concept_rich_sample_ids"):
        _make_prefix(rich=[], avoid=[])


def test_zero_context_segments_raises():
    """An empty prefix_segments tuple → FeatureGcgError (context cache must be >= 1).

    A prefix with no context segments is a real pipeline bug (the score stage
    always conditions >= 1 segment), never a skippable case.
    """
    prefix = _make_prefix()
    with pytest.raises(FeatureGcgError, match="zero context segments"):
        _build(prefix, prefix_segments=())


def test_max_seq_len_guard_raises():
    """prefix + longest completion exceeding max_seq_len → FeatureGcgError.

    Mirrors the score stage's left-truncation trigger
    (len(prefix) + n_output > max_seq_len). The derive does not mirror
    truncation, so it RAISES rather than optimizing an untruncated layout the
    score stage would truncate.
    """
    # n_prefix=2, longest completion=9 → 11 tokens; max_seq_len=10 ⇒ 11 > 10.
    prefix = _make_prefix(
        prefix_tokens=[5, 6], rich=[list(range(20, 29))], avoid=[[9, 10]]
    )
    with pytest.raises(FeatureGcgError, match="exceeds max_seq_len"):
        _build(prefix, max_seq_len=10)


def test_max_seq_len_guard_boundary_builds():
    """prefix + longest completion EXACTLY at max_seq_len builds (score truncates > only).

    Pins the ``>`` boundary: a mutation to ``>=`` makes this case raise (RED),
    matching score's ``len(prefix) + n_output <= max_seq_len`` no-truncation rule.
    """
    # n_prefix=2, longest completion=8 → 10 tokens; max_seq_len=10 ⇒ 10 > 10 is False.
    prefix = _make_prefix(
        prefix_tokens=[5, 6], rich=[list(range(20, 28))], avoid=[[9, 10]]
    )
    members = _build(prefix, max_seq_len=10)
    assert len(members) == 1


def test_long_completion_builds_members():
    """Completion length never raises on its own: all output positions route to the
    single generation slot, so a long completion (within max_seq_len) is fine."""
    long_completion = list(range(20, 220))  # 200 output tokens
    prefix = _make_prefix(prefix_tokens=[5, 6], rich=[long_completion], avoid=[[9, 10]])
    m = _build(prefix)[0]
    # Every output position (n_prefix..) sits in the generation slot (== 1).
    assert m.rich.seg_map[0, 2:].eq(1).all()


def test_segment_conditions_length_is_context_plus_generation():
    """segment_conditions has len(prefix_segments) + 1 bags (contexts + generation)."""
    prefix = _make_prefix(prefix_tokens=[5, 6, 3, 7], rich=[[7, 8]], avoid=[[9, 10]])
    segs = (_seg(0, 2, feature_ids=(10,)), _seg(2, 4, feature_ids=(20,)))
    m = _build(prefix, prefix_segments=segs)[0]
    assert len(m.rich.segment_conditions) == len(segs) + 1
    assert len(m.avoid.segment_conditions) == len(segs) + 1
