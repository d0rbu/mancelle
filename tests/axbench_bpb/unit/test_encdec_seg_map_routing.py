"""Unit tests for ``_build_encdec_seg_map_and_encoder``.

The helper owns the full seg_map + encoder feature-input construction for the
encdec teacher-forcing path in ``_pad_encoder_inputs`` and ``score_sample``.

When ``n_prompt_surviving > 0`` AND ``prefix_segments`` is non-empty, the
helper prepends an empty conditioning slot at encoder index 0 and shifts
every non-zero seg_map value up by 1 so that:

* Prompt-prefix tokens route to slot 0 (empty, fully masked) and receive
  zero contribution from cross-attention.
* Inputs-prefix tokens route to slots 1..k (originally 0..k-1 in the
  ``build_multi_segment_seg_map`` output, +1 after shift).
* Output tokens route to ``generation_slot = k + 1``.

Without a cached prefix segment, the helper raises instead of building an
unconditioned prompt route. Otherwise it falls through to the existing
``build_multi_segment_seg_map`` behaviour with
``generation_slot = len(prefix_segments)``.

These tests pin both the shape/value semantics of the helper AND the two
downstream invariants the cross-attention layer enforces:
- Monotonic-up-from-zero (cross_attention.py:185-196).
- G9 unique ``padded_idx = seg_map * max_seg_size + pos_in_seg`` per row
  (``_padded_idx_unique_per_row``).
"""
from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_bpb.teacher_force import (
    _build_encdec_seg_map_and_encoder,
    _padded_idx_unique_per_row,
)
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(char_start: int, char_end: int, fid: int = 7, bkt: int = 3) -> ContextFeatureSegment:
    """Build a one-feature ContextFeatureSegment over the given char range."""
    return ContextFeatureSegment(
        feature_ids=(fid,),
        buckets=(bkt,),
        char_start=char_start,
        char_end=char_end,
    )


def _check_non_zero_monotonic(seg_map: torch.Tensor) -> bool:
    """Mirror cross_attention.py's monotonicity check.

    Returns True iff every non-zero entry in ``seg_map`` is >= the
    running max of all earlier non-zero entries. ``seg_map`` shape: (1, T).
    """
    sm = seg_map[0]
    nonzero_cummax = torch.where(sm > 0, sm, torch.zeros_like(sm)).cummax(dim=0).values
    return bool(((sm == 0) | (sm >= nonzero_cummax)).all().item())


# ---------------------------------------------------------------------------
# Branch: missing prefix segments fail
# ---------------------------------------------------------------------------


class TestExistingPath:
    """The helper's no-shift branch reproduces the existing behaviour."""

    def test_no_prompt_no_segments_raises(self) -> None:
        """No cached prefix segments is invalid for encdec BPB scoring."""
        tok = SimpleTokenizer()
        full_prefix = encode_prefix("abc", tok)  # 4 tokens (BOS + 3 chars)
        with pytest.raises(AxBenchBpbError, match="prefix_context_segments"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok,
                full_prefix=full_prefix,
                prefix_segments=(),
                feature_conditions=None,
                n_output=2,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
            )

    def test_no_prompt_with_segments(self) -> None:
        """No prompt prefix, 1 segment: inputs assigned to slot 0, output to slot 1."""
        tok = SimpleTokenizer()
        full_prefix = encode_prefix("abc", tok)  # 4 tokens, chars [0, 4)
        segments = (_seg(0, 4),)
        enc_fids, _enc_mags, _enc_mask, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=2,
            n_prompt_surviving=0,
            device=torch.device("cpu"),
        )
        assert seg_map.shape == (1, len(full_prefix) + 2)
        # 4 prefix tokens at slot 0; 2 output tokens at slot 1.
        assert seg_map[0, :len(full_prefix)].eq(0).all().item()
        assert seg_map[0, len(full_prefix):].eq(1).all().item()
        assert enc_fids.shape[1] == 2  # 1 segment + 1 generation slot
        assert gen_slot == 1
        assert _check_non_zero_monotonic(seg_map)
        _padded_idx_unique_per_row(seg_map, n_seg_max=2)

    def test_prompt_surviving_no_segments_raises(self) -> None:
        """Prompting cannot bypass missing cached prefix segments."""
        tok = SimpleTokenizer()
        full_prefix = encode_prefix("pp", tok) + encode_prefix("abc", tok)[1:]  # 5 tokens
        with pytest.raises(AxBenchBpbError, match="prefix_context_segments"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok,
                full_prefix=full_prefix,
                prefix_segments=(),
                feature_conditions=None,
                n_output=2,
                n_prompt_surviving=3,  # 2 prompt-prefix + BOS, conservatively
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------------------
# Branch: shifted (the bug case being fixed)
# ---------------------------------------------------------------------------


class TestShiftedPath:
    """The helper's shifted branch prepends an empty slot and shifts indices."""

    def test_prompt_surviving_one_segment(self) -> None:
        """2 prompt-prefix + 4 inputs-prefix (1 segment) + 2 output.

        Expected seg_map: [0, 0, 1, 1, 1, 1, 2, 2]
        (n_prompt=2 zeros) + (inputs assignments [0,0,0,0]+1=[1,1,1,1]) + (output [1]+1=[2,2])
        """
        tok = SimpleTokenizer()
        # Build a 6-token combined prefix: BOS + "pp" (3 tokens) + inputs (3 tokens).
        # For tractability use raw integer ids; SimpleTokenizer's encode_prefix
        # prepends BOS=0. We just need len consistency for the helper.
        full_prefix = [0, ord("p"), ord("p"), ord("a"), ord("b"), ord("c")]  # 6 tokens
        # Inputs-prefix segments: char_end=3 covers "abc" (3 chars).
        segments = (_seg(0, 3),)
        n_prompt_surviving = 3  # BOS + 2 prompt tokens
        enc_fids, _enc_mags, enc_mask, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=2,
            n_prompt_surviving=n_prompt_surviving,
            device=torch.device("cpu"),
        )
        assert seg_map.shape == (1, len(full_prefix) + 2)
        # First n_prompt_surviving positions are zero (empty slot).
        assert seg_map[0, :n_prompt_surviving].eq(0).all().item()
        # Inputs-prefix positions are shifted to slot 1.
        n_inputs = len(full_prefix) - n_prompt_surviving
        assert seg_map[0, n_prompt_surviving:n_prompt_surviving + n_inputs].eq(1).all().item()
        # Output positions at the shifted generation slot (k+1 = 2).
        assert seg_map[0, -2:].eq(2).all().item()
        # Encoder has 3 slots: [empty, seg_0, gen_slot].
        assert enc_fids.shape[1] == 3
        # Slot 0 must be fully masked.
        assert enc_mask[0, 0, :].eq(False).all().item()
        assert gen_slot == 2
        assert _check_non_zero_monotonic(seg_map)
        _padded_idx_unique_per_row(seg_map, n_seg_max=3)

    def test_prompt_surviving_two_segments(self) -> None:
        """2 prompt-prefix + 4 inputs-prefix (2 segments) + 2 output.

        Inputs char range: [0, 4). Two segments: [0, 2) and [2, 4) cover chars 0..3.
        Expected: prompt at 0, inputs at 1..2 (shifted from 0..1), output at 3.
        """
        tok = SimpleTokenizer()
        # Combined prefix: 2 prompt + 4 inputs = 6 tokens.
        full_prefix = [0, ord("p"), ord("a"), ord("b"), ord("c"), ord("d")]
        # Inputs-prefix is "abcd" (chars [0,4)); split into two segs.
        segments = (_seg(0, 2), _seg(2, 4))
        n_prompt_surviving = 2  # BOS + "p"
        _enc_fids, _enc_mags, _enc_mask, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=2,
            n_prompt_surviving=n_prompt_surviving,
            device=torch.device("cpu"),
        )
        assert seg_map[0, :n_prompt_surviving].eq(0).all().item()
        # Output positions at slot k+1 = 3.
        assert seg_map[0, -2:].eq(3).all().item()
        # Non-zero values monotonic.
        assert _check_non_zero_monotonic(seg_map)
        # G9 safe with 4 slots (empty + 2 segs + gen).
        _padded_idx_unique_per_row(seg_map, n_seg_max=4)
        assert gen_slot == 3

    def test_prompt_surviving_n_inputs_zero(self) -> None:
        """Edge case G: n_inputs_surviving == 0.

        Only prompt-prefix tokens remain after truncation, but segments
        are non-empty. Expected: ``[0]*n_prompt_surviving + [gen_slot]*n_output``.
        """
        tok = SimpleTokenizer()
        full_prefix = [0, ord("p"), ord("p")]  # 3 prompt-prefix tokens
        segments = (_seg(0, 4), _seg(4, 8))  # 2 segments
        n_prompt_surviving = 3
        _enc_fids, _enc_mags, _enc_mask, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=2,
            n_prompt_surviving=n_prompt_surviving,
            device=torch.device("cpu"),
        )
        assert seg_map.shape == (1, 5)  # 3 + 2
        assert seg_map[0, :3].eq(0).all().item()
        # generation_slot = k+1 = 3.
        assert seg_map[0, 3:].eq(3).all().item()
        assert _check_non_zero_monotonic(seg_map)
        _padded_idx_unique_per_row(seg_map, n_seg_max=4)
        assert gen_slot == 3

    def test_helper_dtypes(self) -> None:
        """All returned tensors have the expected dtypes."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("p"), ord("a")]
        segments = (_seg(0, 1),)
        enc_fids, enc_mags, enc_mask, seg_map, _ = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=1,
            n_prompt_surviving=2,
            device=torch.device("cpu"),
        )
        assert enc_fids.dtype == torch.int64
        assert enc_mags.dtype == torch.int64
        assert enc_mask.dtype == torch.bool
        assert seg_map.dtype == torch.long

    def test_helper_device(self) -> None:
        """Returned tensors live on the requested device."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("p"), ord("a")]
        segments = (_seg(0, 1),)
        enc_fids, enc_mags, enc_mask, seg_map, _ = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=1,
            n_prompt_surviving=2,
            device=torch.device("cpu"),
        )
        assert enc_fids.device.type == "cpu"
        assert enc_mags.device.type == "cpu"
        assert enc_mask.device.type == "cpu"
        assert seg_map.device.type == "cpu"

    def test_helper_returns_correct_generation_slot(self) -> None:
        """``generation_slot`` return value matches the highest seg_map value
        on the output positions in the shifted path."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("p"), ord("a"), ord("b"), ord("c")]
        segments = (_seg(0, 3),)  # 1 segment
        _, _, _, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=2,
            n_prompt_surviving=2,
            device=torch.device("cpu"),
        )
        assert int(seg_map[0, -2:].max().item()) == gen_slot

    def test_feature_conditions_pass_through(self) -> None:
        """``feature_conditions`` dict gets installed in the generation slot."""
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b")]
        segments = (_seg(0, 2),)
        feature_conditions = {42: 5}  # one feature with bucket 5
        enc_fids, enc_mags, enc_mask, _, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=feature_conditions,
            n_output=1,
            n_prompt_surviving=1,
            device=torch.device("cpu"),
        )
        # Generation slot index == 2 (k+1 = 1+1). enc_fids[0, 2, :] should
        # contain {42}, enc_mags[0, 2, :] should contain {5}, mask True.
        assert int(enc_fids[0, gen_slot, 0].item()) == 42
        assert int(enc_mags[0, gen_slot, 0].item()) == 5
        assert bool(enc_mask[0, gen_slot, 0].item()) is True

    def test_helper_idempotent_under_repeated_calls(self) -> None:
        """Calling the helper twice with identical args yields tensors
        that compare equal element-wise. Defends against future
        nondeterminism (e.g., dict-ordering issues in feature_conditions)."""
        tok = SimpleTokenizer()
        full_prefix = [0, 1, 2, 3, 4, 5]
        prefix_segments = (_seg(0, 3),)
        feature_conditions: dict[int, int] = {42: 5, 17: 0, 99: 3}  # multi-key
        n_output = 2
        n_prompt_surviving = 2
        device = torch.device("cpu")
        r1 = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=prefix_segments,
            feature_conditions=feature_conditions,
            n_output=n_output,
            n_prompt_surviving=n_prompt_surviving,
            device=device,
        )
        r2 = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=prefix_segments,
            feature_conditions=feature_conditions,
            n_output=n_output,
            n_prompt_surviving=n_prompt_surviving,
            device=device,
        )
        for t1, t2 in zip(r1[:4], r2[:4], strict=True):
            assert torch.equal(t1, t2), (
                f"helper produced non-identical tensors across calls: "
                f"diff = {(t1 - t2).abs().max() if t1.dtype != torch.bool else (t1 != t2).any()}"
            )
        assert r1[4] == r2[4]  # generation_slot

    def test_shifted_branch_preserves_prefix_segment_features(self) -> None:
        """In the shifted branch, prefix segment N's feature_ids and
        buckets must land at encoder slot N+1 (not slot N, not 0) and
        must equal the segment's original values verbatim. A bug that
        permuted or reordered segments during the shift would still
        produce a valid seg_map but corrupt the encoder features."""
        seg_a = ContextFeatureSegment(
            feature_ids=(10, 11), buckets=(2, 3),
            char_start=0, char_end=2,
        )
        seg_b = ContextFeatureSegment(
            feature_ids=(20,), buckets=(1,),
            char_start=2, char_end=4,
        )
        enc_fids, enc_mags, enc_mask, _, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=SimpleTokenizer(),
            full_prefix=[0, ord("p"), ord("a"), ord("b"), ord("c"), ord("d")],
            prefix_segments=(seg_a, seg_b),
            feature_conditions=None,
            n_output=1,
            n_prompt_surviving=2,
            device=torch.device("cpu"),
        )
        # Encoder layout: slot 0 = empty (masked), slot 1 = seg_a,
        # slot 2 = seg_b, slot 3 = generation.
        assert enc_fids.shape[1] == 4
        # Slot 0 must be fully masked.
        assert enc_mask[0, 0, :].eq(False).all().item()
        # Slot 1 = seg_a. The exact column layout depends on
        # build_segmented_feature_encoder_inputs's padding, but the SET of
        # feature_ids/buckets at slot 1 must equal seg_a's.
        slot_1_active = enc_mask[0, 1, :].nonzero(as_tuple=True)[0]
        slot_1_fids = set(enc_fids[0, 1, slot_1_active].tolist())
        slot_1_mags = set(enc_mags[0, 1, slot_1_active].tolist())
        assert slot_1_fids == {10, 11}, (
            f"slot 1 should carry seg_a feature_ids {{10, 11}}; got {slot_1_fids}"
        )
        assert slot_1_mags == {2, 3}, (
            f"slot 1 should carry seg_a buckets {{2, 3}}; got {slot_1_mags}"
        )
        # Slot 2 = seg_b.
        slot_2_active = enc_mask[0, 2, :].nonzero(as_tuple=True)[0]
        slot_2_fids = set(enc_fids[0, 2, slot_2_active].tolist())
        slot_2_mags = set(enc_mags[0, 2, slot_2_active].tolist())
        assert slot_2_fids == {20}, (
            f"slot 2 should carry seg_b feature_ids {{20}}; got {slot_2_fids}"
        )
        assert slot_2_mags == {1}, (
            f"slot 2 should carry seg_b buckets {{1}}; got {slot_2_mags}"
        )
        # gen_slot index = 3.
        assert gen_slot == 3


# ---------------------------------------------------------------------------
# Regression: G9 + monotonicity checks
# ---------------------------------------------------------------------------


class TestContractRegression:
    """Cross the new seg_map shape against the two production invariants."""

    @pytest.mark.parametrize("n_prompt,k_segs,n_inputs,n_output", [
        (1, 1, 1, 1),
        (1, 1, 2, 2),
        (2, 1, 3, 2),
        (2, 2, 4, 3),
        (3, 3, 5, 4),
        (1, 5, 5, 5),
        (5, 1, 1, 1),
    ])
    def test_padded_idx_unique_per_row_holds(self, n_prompt, k_segs, n_inputs, n_output) -> None:
        tok = SimpleTokenizer()
        # Char layout: prompt-prefix chars [0, n_prompt), inputs chars [n_prompt, n_prompt+n_inputs).
        # SimpleTokenizer is char-level: 1 char == 1 token + BOS prefix handled by encode_prefix.
        # For this property test, ignore BOS — we build raw ids directly.
        full_prefix = list(range(n_prompt + n_inputs))  # token ids 0..P+N-1
        # Each segment covers (n_inputs // k_segs) chars; spread evenly.
        seg_width = max(1, n_inputs // k_segs)
        segments = tuple(
            _seg(i * seg_width, min(n_inputs, (i + 1) * seg_width))
            for i in range(k_segs)
        )
        _, _, _, seg_map, _ = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=n_output,
            n_prompt_surviving=n_prompt,
            device=torch.device("cpu"),
        )
        _padded_idx_unique_per_row(seg_map, n_seg_max=k_segs + 2)

    @pytest.mark.parametrize("n_prompt,k_segs,n_inputs,n_output", [
        (1, 1, 1, 1),
        (2, 2, 4, 3),
        (3, 3, 5, 4),
        (1, 5, 5, 5),
    ])
    def test_non_zero_monotonic(self, n_prompt, k_segs, n_inputs, n_output) -> None:
        tok = SimpleTokenizer()
        full_prefix = list(range(n_prompt + n_inputs))
        seg_width = max(1, n_inputs // k_segs)
        segments = tuple(
            _seg(i * seg_width, min(n_inputs, (i + 1) * seg_width))
            for i in range(k_segs)
        )
        _, _, _, seg_map, _ = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=n_output,
            n_prompt_surviving=n_prompt,
            device=torch.device("cpu"),
        )
        assert _check_non_zero_monotonic(seg_map)


# ---------------------------------------------------------------------------
# No-shift equivalence: the existing branch must be bit-identical to a direct
# call to build_multi_segment_seg_map.
# ---------------------------------------------------------------------------


class TestNoShiftEquivalence:
    """The existing branch must produce a seg_map BIT-IDENTICAL to a
    direct call to ``build_multi_segment_seg_map``. Pins the no-shift
    contract as a regression guard."""

    def test_no_shift_seg_map_bit_identical_to_direct_call(self) -> None:
        from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b"), ord("c"), ord("d")]
        segments = (_seg(0, 4),)
        _, _, _, helper_seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
            tokenizer=tok,
            full_prefix=full_prefix,
            prefix_segments=segments,
            feature_conditions=None,
            n_output=2,
            n_prompt_surviving=0,  # no shift
            device=torch.device("cpu"),
        )
        direct_seg_map = build_multi_segment_seg_map(
            tokenizer=tok,
            prefix_tokens=full_prefix,
            prefix_segments=segments,
            n_output=2,
            generation_slot=gen_slot,
        )
        assert torch.equal(helper_seg_map.cpu(), direct_seg_map.cpu())

    def test_no_shift_seg_map_rejects_empty_segments(self) -> None:
        tok = SimpleTokenizer()
        full_prefix = [0, ord("a"), ord("b"), ord("c")]
        with pytest.raises(AxBenchBpbError, match="prefix_context_segments"):
            _build_encdec_seg_map_and_encoder(
                tokenizer=tok,
                full_prefix=full_prefix,
                prefix_segments=(),
                feature_conditions=None,
                n_output=2,
                n_prompt_surviving=0,
                device=torch.device("cpu"),
            )
