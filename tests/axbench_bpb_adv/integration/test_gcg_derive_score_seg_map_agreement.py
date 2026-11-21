"""Golden agreement: the feature-GCG derive's per-member seg_map is byte-identical
to the SCORE stage's seg_map for the same ``(prefix, segments, completion)``.

This is the core Bug-2 correctness property.  The derive optimizes the generation
bag under a per-member seg_map built by ``build_class_margin_members`` /
``_build_member``; the score stage applies that bag under a seg_map built by
``teacher_force._build_encdec_seg_map_and_encoder`` (existing, no-prompt branch).
Both must:

* route each prefix token to its OWN context segment slot (char-overlap),
* route every output token to ``generation_slot == len(prefix_segments)``,
* apply the same ``[:, :-1]`` trim (score_sample line 774).

If any of those diverge between derive and score — a different generation_slot, a
dropped/duplicated trim, a segment-ordering mismatch — the optimized bag would be
scored under a different layout, and THIS test goes red.  ``SimpleTokenizer`` maps
each char to one token so segment char-spans line up 1:1 with token positions.
"""

from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.teacher_force import _build_encdec_seg_map_and_encoder
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

_CPU = torch.device("cpu")


def _prefix(
    prefix_tokens: list[int], rich: list[int], avoid: list[int]
) -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id="p",
        concept_id=0,
        mode=SteeringMode.POSITIVE,
        prefix_text="x",
        prefix_hash="h",
        prefix_tokens=prefix_tokens,
        concept_rich_sample_ids=["r"],
        concept_avoiding_sample_ids=["a"],
        concept_rich_output_tokens=[rich],
        concept_avoiding_output_tokens=[avoid],
    )


def _score_seg_map(
    tok: SimpleTokenizer,
    prefix_tokens: list[int],
    segs: tuple[ContextFeatureSegment, ...],
    n_output: int,
) -> torch.Tensor:
    """The score stage's seg_map for the SAME inputs (feature-GCG has no prompt).

    Mirrors score_sample: build via _build_encdec_seg_map_and_encoder with
    n_prompt_surviving=0 and an empty generation bag (feature_conditions={} — the
    derive's placeholder the optimizer later substitutes into), then trim the last
    column (teacher_force.py:774).
    """
    _fids, _mags, _mask, seg_map, _gen = _build_encdec_seg_map_and_encoder(
        tokenizer=tok,
        full_prefix=list(prefix_tokens),
        prefix_segments=segs,
        feature_conditions={},
        n_output=n_output,
        n_prompt_surviving=0,
        device=_CPU,
    )
    return seg_map[:, :-1]


@pytest.mark.parametrize(
    "spans",
    [
        ((0, 6),),  # single context segment (the common short-prefix case)
        ((0, 3), (3, 6)),  # two segments
        ((0, 2), (2, 4), (4, 6)),  # three segments
    ],
)
def test_derive_member_seg_map_matches_score(spans: tuple[tuple[int, int], ...]):
    """Derive member seg_map == score seg_map, for 1/2/3-segment prefixes.

    Both the rich and avoid completions are checked (their differing lengths give
    different n_output, so this pins the generation-slot routing at two lengths).
    """
    tok = SimpleTokenizer()
    prefix_tokens = list(tok.encode("abcdef"))  # 6 tokens, chars 0..5
    segs = tuple(
        ContextFeatureSegment(
            feature_ids=(10 * (i + 1),),
            buckets=(i + 1,),
            char_start=start,
            char_end=end,
        )
        for i, (start, end) in enumerate(spans)
    )
    rich = list(tok.encode("xy"))  # 2 output tokens
    avoid = list(tok.encode("zw q"))  # 4 output tokens (different length)
    prefix = _prefix(prefix_tokens, rich, avoid)

    member = build_class_margin_members(
        prefix,
        prefix_segments=segs,
        tokenizer=tok,
        max_seq_len=1000,
        device=_CPU,
    )[0]

    assert torch.equal(
        member.rich.seg_map, _score_seg_map(tok, prefix_tokens, segs, len(rich))
    ), "derive rich seg_map diverged from the score stage's seg_map"
    assert torch.equal(
        member.avoid.seg_map, _score_seg_map(tok, prefix_tokens, segs, len(avoid))
    ), "derive avoid seg_map diverged from the score stage's seg_map"


def test_derive_generation_slot_matches_score_generation_slot():
    """The derive's generation_slot equals the score stage's returned generation_slot.

    Both must be ``len(prefix_segments)`` (the slot after the context segments);
    a mismatch would substitute/apply the GCG bag into different encoder slots.
    """
    tok = SimpleTokenizer()
    prefix_tokens = list(tok.encode("abcdef"))
    segs = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=3, char_end=6),
    )
    prefix = _prefix(prefix_tokens, list(tok.encode("xy")), list(tok.encode("zw")))
    member = build_class_margin_members(
        prefix, prefix_segments=segs, tokenizer=tok, max_seq_len=1000, device=_CPU
    )[0]

    _f, _m, _k, _seg, score_generation_slot = _build_encdec_seg_map_and_encoder(
        tokenizer=tok,
        full_prefix=list(prefix_tokens),
        prefix_segments=segs,
        feature_conditions={},
        n_output=2,
        n_prompt_surviving=0,
        device=_CPU,
    )
    assert member.rich.generation_slot == score_generation_slot == len(segs)
