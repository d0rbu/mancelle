"""Audit-fixup 2026-05-11 + 2026-05-11 fix: prompting+encdec routing.

Background
----------
``_pad_encoder_inputs`` (and the sister branch in ``score_sample``) takes
``inputs.prefix_context_segments`` whose ``char_start`` / ``char_end``
are RELATIVE TO ``inputs.prefix_tokens``. When
``intervention.prompt_prefix_tokens`` is non-empty, the model receives
``full_prefix = prompt_prefix_tokens + inputs.prefix_tokens`` — a
combined prefix.

The 2026-05-11 audit-fixup added a raise to prevent silent mis-routing
of surviving prompt-prefix tokens into segment 0. The 2026-05-11 fix
replaced that raise with a shared helper
``_build_encdec_seg_map_and_encoder`` that prepends an empty masked
encoder slot at index 0 and shifts every non-zero seg_map value up by
1. See
``docs/superpowers/specs/2026-05-11-prompting-encdec-encoder-conditioning-design.md``.

Four branches are exercised:

1. Surviving prompt-prefix tokens + non-empty segments → no raise; the
   helper produces a shifted seg_map: prompt-prefix tokens at slot 0
   (empty masked), inputs-prefix at slots 1..k, output at slot k+1.
2. All prompt-prefix tokens dropped (no surviving) + non-empty segments
   → no raise (existing inputs-prefix projection path).
3. Non-empty prompt-prefix + missing segments → hard fail; encdec BPB
   scoring must never silently run with unconditioned prompt text.
4. No prompt prefix + non-empty segments → no raise (canonical path).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from steering.eval.axbench_bpb.sample_loader import (
    make_prefix_hash,
    make_sample_id,
)
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    _pad_encoder_inputs,
    score_sample,
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
# Fixtures
# ---------------------------------------------------------------------------


def _make_inputs(
    tok,
    *,
    prefix_text: str,
    output_text: str,
    prefix_context_segments: tuple[ContextFeatureSegment, ...] = (),
) -> ScoringInputs:
    """Build a ``ScoringInputs`` with optional context segments.

    With ``SimpleTokenizer`` (char-level), each char maps to exactly one
    token and the char offset 0 lines up with token 0.
    """
    prefix_tokens = encode_prefix(prefix_text, tok)
    output_tokens = encode_output(output_text, tok)
    byte_weights = torch.ones(len(output_tokens), dtype=torch.int64)
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=0,
            mode=SteeringMode.POSITIVE,
            content_class="concept_rich",
            prefix_text=prefix_text,
            output_text=output_text,
        ),
        concept_id=0,
        mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_text,
        prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=prefix_context_segments,
        n_prefix=len(prefix_tokens),
        n_output=len(output_tokens),
        output_byte_weights=byte_weights,
        truncated=False,
    )


def _make_bundle(max_seq_len: int) -> TargetModelBundle:
    return TargetModelBundle(
        # _pad_encoder_inputs reads `.feature_vocab` (via
        # `feature_vocab_allowed_ids`) to V-restrict context-cache
        # features; a bare `None` model would AttributeError there. These
        # tests exercise seg_map/padding structure only (never call
        # `model(...)`), so a `.feature_vocab`-only stand-in suffices.
        model=SimpleNamespace(feature_vocab=None),
        tokenizer=SimpleTokenizer(),
        max_seq_len=max_seq_len,
        target_name="encdec",
    )


def _make_segments(char_start: int, char_end: int) -> tuple[ContextFeatureSegment, ...]:
    return (
        ContextFeatureSegment(
            feature_ids=(7,),
            buckets=(3,),
            char_start=char_start,
            char_end=char_end,
        ),
    )


# ---------------------------------------------------------------------------
# _pad_encoder_inputs: dispatch path used by the batched microbatch kernel.
# ---------------------------------------------------------------------------


class TestPadEncoderInputsGuard:
    """``_pad_encoder_inputs`` routes surviving prompt-prefix tokens to the
    empty slot (index 0) via the shared helper, shifting inputs-prefix slots.
    """

    def test_surviving_prompt_prefix_with_segments_routes_to_empty_slot(self) -> None:
        """Surviving prompt-prefix + non-empty segments routes prompt-prefix
        tokens to the empty slot (index 0), inputs-prefix to shifted slots
        1..k, output to slot k+1. Encoder gains one extra slot (the empty
        prompt slot)."""
        bundle = _make_bundle(max_seq_len=128)
        tok = bundle.tokenizer
        sample = _make_inputs(
            tok,
            prefix_text="abc",
            output_text="xy",
            # ``encode_prefix`` prepends BOS, so "abc" → 4 tokens spanning
            # chars [0, 4); cover the whole prefix.
            prefix_context_segments=_make_segments(char_start=0, char_end=4),
        )
        prompt_prefix = list(encode_prefix("pp", tok))
        intervention = Intervention.prompting(
            target_model="encdec",
            prompt_prefix_tokens=prompt_prefix,
        )
        full_prefix = prompt_prefix + list(sample.prefix_tokens)
        n_prompt_surviving = len(prompt_prefix)
        n_inputs_surviving = len(sample.prefix_tokens)
        n_output = sample.n_output
        padded_encoder = _pad_encoder_inputs(
            [sample],
            [full_prefix],
            intervention,
            bundle,
            len(full_prefix) + n_output,
            torch.device("cpu"),
        )
        # seg_map shape: (1, T_max - 1) where T_max = len(full_prefix) + n_output.
        # Inside _pad_encoder_inputs the helper returns shape (1, T) and a
        # `[:, :-1]` slice drops the final position.
        seg_map = padded_encoder.seg_map  # (1, T_max - 1)
        # Prompt-prefix positions route to slot 0 (empty).
        assert seg_map[0, :n_prompt_surviving].eq(0).all().item(), (
            f"prompt-prefix positions should route to slot 0; got "
            f"{seg_map[0, :n_prompt_surviving].tolist()}"
        )
        # Inputs-prefix positions route to slot 1 (single-segment case: shifted
        # from original slot 0 to slot 1).
        inputs_slice = seg_map[0, n_prompt_surviving:n_prompt_surviving + n_inputs_surviving]
        assert inputs_slice.eq(1).all().item(), (
            f"inputs-prefix positions should route to slot 1; got {inputs_slice.tolist()}"
        )
        # Output positions route to slot k+1 = 2. The seg_map's tail
        # (after truncation drops the final position via [:, :-1]) covers
        # n_output - 1 real output positions; all must be at slot 2.
        # Additionally, the per-row pad-tail (filled with generation_slot
        # via _pad_encoder_inputs) must also be at slot 2.
        # In this microbatch with B=1, T_max == n_full_prefix + n_output,
        # so no padding occurs — all output positions are real.
        n_output_real = n_output - 1  # the [:, :-1] truncation drops the last
        if n_output_real > 0:
            output_slice = seg_map[0, -n_output_real:]
            assert output_slice.eq(2).all().item(), (
                f"output positions should all route to slot 2; got "
                f"{output_slice.tolist()}"
            )
        # Empty prompt slot at encoder index 0 is fully masked.
        assert padded_encoder.enc_mask[0, 0, :].eq(False).all().item()

    def test_all_prompt_prefix_dropped_with_segments_does_not_raise(self) -> None:
        """When the truncation dropped EVERY prompt-prefix token, the
        existing inputs-prefix projection path still applies; no guard
        should fire."""
        bundle = _make_bundle(max_seq_len=128)
        tok = bundle.tokenizer
        sample = _make_inputs(
            tok,
            prefix_text="abc",
            output_text="xy",
            # ``encode_prefix`` prepends BOS, so "abc" → 4 tokens spanning
            # chars [0, 4); cover the whole prefix.
            prefix_context_segments=_make_segments(char_start=0, char_end=4),
        )
        prompt_prefix = list(encode_prefix("pp", tok))
        intervention = Intervention.prompting(
            target_model="encdec",
            prompt_prefix_tokens=prompt_prefix,
        )
        # Full_prefix has only the inputs prefix surviving (all 2 prompt-
        # prefix tokens dropped, 0 inputs-prefix tokens dropped).
        full_prefix_inputs_only = list(sample.prefix_tokens)
        # Should not raise; build_segmented_feature_encoder_inputs will run.
        _pad_encoder_inputs(
            [sample],
            [full_prefix_inputs_only],
            intervention,
            bundle,
            len(full_prefix_inputs_only) + sample.n_output,
            torch.device("cpu"),
        )

    def test_empty_segments_with_prompt_prefix_raises(self) -> None:
        """Prompt-prefix + missing segments fails instead of scoring
        unconditioned prompt text."""
        bundle = _make_bundle(max_seq_len=128)
        tok = bundle.tokenizer
        sample = _make_inputs(
            tok,
            prefix_text="abc",
            output_text="xy",
            prefix_context_segments=(),  # explicitly empty
        )
        prompt_prefix = list(encode_prefix("pp", tok))
        intervention = Intervention.prompting(
            target_model="encdec",
            prompt_prefix_tokens=prompt_prefix,
        )
        full_prefix = prompt_prefix + list(sample.prefix_tokens)
        with pytest.raises(AxBenchBpbError, match="prefix_context_segments"):
            _pad_encoder_inputs(
                [sample],
                [full_prefix],
                intervention,
                bundle,
                len(full_prefix) + sample.n_output,
                torch.device("cpu"),
            )

    def test_no_prompt_prefix_with_segments_does_not_raise(self) -> None:
        """Without prompt-prefix, the guard is scoped out (canonical path)."""
        bundle = _make_bundle(max_seq_len=128)
        tok = bundle.tokenizer
        sample = _make_inputs(
            tok,
            prefix_text="abc",
            output_text="xy",
            # ``encode_prefix`` prepends BOS, so "abc" → 4 tokens spanning
            # chars [0, 4); cover the whole prefix.
            prefix_context_segments=_make_segments(char_start=0, char_end=4),
        )
        intervention = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
        full_prefix = list(sample.prefix_tokens)
        _pad_encoder_inputs(
            [sample],
            [full_prefix],
            intervention,
            bundle,
            len(full_prefix) + sample.n_output,
            torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# score_sample: per-sample kernel that uses the same guard.
# ---------------------------------------------------------------------------


class TestScoreSampleGuard:
    """``score_sample``'s encdec branch routes surviving prompt-prefix tokens
    to the empty slot via the shared helper; no raise on any supported config.
    """

    def test_surviving_prompt_prefix_with_segments_routes_to_empty_slot(
        self, tiny_encdec,
    ) -> None:
        """score_sample's encdec branch with prompt-prefix + segments
        returns a finite BPB without raising. The shared helper produces
        a valid seg_map; downstream cross-attention does not corrupt."""
        model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
        bundle = TargetModelBundle(
            model=model,
            tokenizer=SimpleTokenizer(),
            max_seq_len=dec_cfg.sequence_len,
            target_name="encdec",
        )
        sample = _make_inputs(
            bundle.tokenizer,
            prefix_text="abc",
            output_text="xy",
            prefix_context_segments=_make_segments(char_start=0, char_end=4),
        )
        prompt_prefix = list(encode_prefix("pp", bundle.tokenizer))
        intervention = Intervention.prompting(
            target_model="encdec",
            prompt_prefix_tokens=prompt_prefix,
        )
        row = score_sample(
            inputs=sample,
            target_bundle=bundle,
            intervention=intervention,
            device=torch.device("cpu"),
            coefficient_str="None",
            method_name="prompting_encdec",
        )
        # No exception; BPB is finite; the row's bookkeeping is intact.
        assert math.isfinite(row.bpb_intervened), f"row.bpb_intervened is not finite: {row.bpb_intervened}"
        assert row.n_output_tokens == sample.n_output
        assert row.method_name == "prompting_encdec"
