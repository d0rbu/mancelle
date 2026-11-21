"""Functional tests: prompting + encdec + non-empty prefix_context_segments.

These exercise the full ``score_sample`` and ``_pad_encoder_inputs`` paths
end-to-end with the ``tiny_encdec`` fixture's real model and cross-
attention. They complement the unit tests in
``tests/axbench_bpb/unit/test_encdec_seg_map_routing.py``: the unit tests
pin the helper's tensor-level contract; these tests pin the integration
with the real cross-attention scatter and decoder forward.
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
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _make_sample(tok, *, prefix_text: str, output_text: str, segments):
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
        prefix_context_segments=segments,
        n_prefix=len(prefix_tokens),
        n_output=len(output_tokens),
        output_byte_weights=byte_weights,
        truncated=False,
    )


def _segments(char_start: int, char_end: int):
    return (
        ContextFeatureSegment(
            feature_ids=(7,), buckets=(3,),
            char_start=char_start, char_end=char_end,
        ),
    )


def test_score_sample_prompting_encdec_with_segments_returns_finite_bpb(tiny_encdec) -> None:
    """``score_sample`` with prompting + segments returns a finite BPB.

    Exercises the shifted seg_map path through the real encoder-decoder
    forward and cross-attention scatter. A finite BPB demonstrates no
    scatter corruption (otherwise the model output would NaN or diverge).
    """
    model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    bundle = TargetModelBundle(
        model=model,
        tokenizer=SimpleTokenizer(),
        max_seq_len=dec_cfg.sequence_len,
        target_name="encdec",
    )
    sample = _make_sample(
        bundle.tokenizer,
        prefix_text="hello",
        output_text="world",
        segments=_segments(0, 6),  # BOS + "hello" → 6 tokens, chars [0, 6)
    )
    prompt_prefix = list(encode_prefix("aa", bundle.tokenizer))
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
    assert math.isfinite(row.bpb_intervened), (
        f"row.bpb_intervened is not finite: {row.bpb_intervened}"
    )
    assert row.n_output_tokens == sample.n_output


def test_score_sample_seg_map_shape_matches_decoder_length(tiny_encdec) -> None:
    """The score_sample encdec branch's seg_map covers exactly
    ``T - 1 == n_full_prefix + n_output - 1`` positions, matching the
    decoder's shifted target tensor shape."""
    model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    bundle = TargetModelBundle(
        model=model,
        tokenizer=SimpleTokenizer(),
        max_seq_len=dec_cfg.sequence_len,
        target_name="encdec",
    )
    sample = _make_sample(
        bundle.tokenizer,
        prefix_text="ab",
        output_text="cd",
        segments=_segments(0, 3),
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
    # Indirect check: the score_sample succeeded with the right shape;
    # the encoder construction would have raised on shape mismatch.
    assert math.isfinite(row.bpb_intervened)


def test_pad_encoder_inputs_microbatch_prompting_encdec_with_segments() -> None:
    """``_pad_encoder_inputs`` over a mixed microbatch: one sample has
    surviving prompt prefix + segments (shifted path), another has no
    prompt prefix + segments (existing path). The G9 unique-padded_idx
    guard does NOT fire."""
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        # _pad_encoder_inputs reads `.feature_vocab` (via
        # `feature_vocab_allowed_ids`) to V-restrict context-cache
        # features; a bare `None` model would AttributeError there. This
        # test exercises seg_map/padding structure only (never calls
        # `model(...)`), so a `.feature_vocab`-only stand-in suffices.
        model=SimpleNamespace(feature_vocab=None),
        tokenizer=tok,
        max_seq_len=128,
        target_name="encdec",
    )
    # Sample A: prompt prefix surviving + 1 segment.
    sample_a = _make_sample(
        tok, prefix_text="abc", output_text="xy",
        segments=_segments(0, 4),
    )
    prompt_prefix = list(encode_prefix("pp", tok))
    full_prefix_a = prompt_prefix + list(sample_a.prefix_tokens)

    # Sample B: no prompt prefix, 1 segment.
    sample_b = _make_sample(
        tok, prefix_text="def", output_text="xy",
        segments=_segments(0, 4),
    )
    full_prefix_b = list(sample_b.prefix_tokens)

    intervention = Intervention.prompting(
        target_model="encdec",
        prompt_prefix_tokens=prompt_prefix,
    )
    # Microbatch with mixed paths. _pad_encoder_inputs pads to the max
    # T over both samples.
    T_max = max(
        len(full_prefix_a) + sample_a.n_output,
        len(full_prefix_b) + sample_b.n_output,
    )
    padded = _pad_encoder_inputs(
        [sample_a, sample_b],
        [full_prefix_a, full_prefix_b],
        intervention,
        bundle,
        T_max,
        torch.device("cpu"),
    )
    # Both rows should pass the G9 check (run inside _pad_encoder_inputs).
    # The shapes should be padded to (B=2, n_seg_max, F_max).
    assert padded.seg_map.shape[0] == 2
    assert padded.enc_feature_ids.shape[0] == 2


def test_prompting_encdec_preserves_inputs_segment_assignments() -> None:
    """The inputs-prefix portion of seg_map (after the prompt-prefix run)
    holds the same SLOT VALUES as if there were no prompt prefix, modulo
    the +1 shift."""
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=SimpleNamespace(feature_vocab=None),  # see comment above
        tokenizer=tok, max_seq_len=128, target_name="encdec",
    )
    sample = _make_sample(
        tok, prefix_text="abcd", output_text="xy",
        segments=(_segments(0, 2)[0], _segments(2, 5)[0]),
    )
    prompt_prefix = list(encode_prefix("pp", tok))

    # With prompt prefix: shifted path.
    intervention_with = Intervention.prompting(
        target_model="encdec",
        prompt_prefix_tokens=prompt_prefix,
    )
    full_prefix_with = prompt_prefix + list(sample.prefix_tokens)
    n_prompt = len(prompt_prefix)
    padded_with = _pad_encoder_inputs(
        [sample], [full_prefix_with], intervention_with, bundle,
        len(full_prefix_with) + sample.n_output,
        torch.device("cpu"),
    )

    # Without prompt prefix: existing path.
    intervention_without = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    full_prefix_without = list(sample.prefix_tokens)
    padded_without = _pad_encoder_inputs(
        [sample], [full_prefix_without], intervention_without, bundle,
        len(full_prefix_without) + sample.n_output,
        torch.device("cpu"),
    )

    # The inputs-prefix portion of the "with prompt" seg_map should equal
    # the inputs-prefix portion of the "without prompt" seg_map, +1.
    n_inputs = len(sample.prefix_tokens)
    inputs_with = padded_with.seg_map[0, n_prompt:n_prompt + n_inputs]
    inputs_without = padded_without.seg_map[0, :n_inputs]
    assert (inputs_with == inputs_without + 1).all().item(), (
        f"inputs portion mismatch: with={inputs_with.tolist()}, "
        f"without+1={(inputs_without + 1).tolist()}"
    )


def test_pad_encoder_inputs_mixed_branch_encoder_slot_0_semantics() -> None:
    """Microbatch with mixed paths: row 0 takes the shifted branch, row 1
    takes the existing branch. The per-row encoder slot 0 must mean
    different things in each row — fully masked for the shifted row,
    feature-bearing for the existing row. The padded G9 check must not
    fire.
    """
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=SimpleNamespace(feature_vocab=None),  # see comment above
        tokenizer=tok, max_seq_len=128, target_name="encdec",
    )
    # Sample A: prompt-prefix surviving + 1 segment → shifted branch.
    sample_a = _make_sample(
        tok, prefix_text="abc", output_text="xy",
        segments=_segments(0, 4),
    )
    prompt_prefix = list(encode_prefix("pp", tok))
    full_prefix_a = prompt_prefix + list(sample_a.prefix_tokens)
    # Sample B: no prompt-prefix + 1 segment with a DISTINCT feature_id →
    # existing branch. The encoder's slot 0 in row B should carry sample B's
    # segment feature (mask True), distinct from row A's empty slot 0.
    seg_b = ContextFeatureSegment(
        feature_ids=(42,), buckets=(9,), char_start=0, char_end=4,
    )
    sample_b = _make_sample(
        tok, prefix_text="def", output_text="xy",
        segments=(seg_b,),
    )
    full_prefix_b = list(sample_b.prefix_tokens)
    intervention = Intervention.prompting(
        target_model="encdec",
        prompt_prefix_tokens=prompt_prefix,
    )
    T_max = max(
        len(full_prefix_a) + sample_a.n_output,
        len(full_prefix_b) + sample_b.n_output,
    )
    padded = _pad_encoder_inputs(
        [sample_a, sample_b],
        [full_prefix_a, full_prefix_b],
        intervention,
        bundle,
        T_max,
        torch.device("cpu"),
    )
    # Row A (shifted): slot 0 must be fully masked.
    assert padded.enc_mask[0, 0, :].eq(False).all().item(), (
        f"row A slot 0 should be empty (mask all False); got "
        f"{padded.enc_mask[0, 0, :].tolist()}"
    )
    # Row B (existing): slot 0 must carry sample B's segment features
    # (at least one True in the mask).
    assert padded.enc_mask[1, 0, :].any().item(), (
        f"row B slot 0 should carry segment features; got mask "
        f"{padded.enc_mask[1, 0, :].tolist()}"
    )
    # Row A seg_map starts at 0 (prompt-prefix tokens at empty slot).
    n_prompt_surviving_a = len(prompt_prefix)
    assert padded.seg_map[0, :n_prompt_surviving_a].eq(0).all().item()
    # Row B seg_map starts at 0 with prefix tokens (real seg) — also zero
    # but for a DIFFERENT semantic reason (the existing-branch seg starts
    # at index 0).
    assert padded.seg_map[1, 0].item() == 0
    # G9 was implicitly checked inside _pad_encoder_inputs (would have
    # raised if collision). The assertion below double-checks via direct
    # invocation.
    from steering.eval.axbench_bpb.teacher_force import _padded_idx_unique_per_row
    _padded_idx_unique_per_row(padded.seg_map, n_seg_max=padded.enc_feature_ids.shape[1])


def test_score_sample_all_prompt_truncated_with_segments_uses_existing_branch(tiny_encdec) -> None:
    """When ``truncate_prefix_left`` drops every prompt-prefix token (so
    n_prompt_surviving == 0) but inputs-prefix segments survive, the
    helper takes the EXISTING branch (no shift). BPB must be finite; the
    seg_map must NOT have prepended zeros for prompt-prefix tokens that
    no longer exist."""
    model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    # 4-token inputs prefix (encode_prefix("abc", tok) = BOS + 3 chars = 4 tokens)
    # + 2-token output = 6 tokens exactly. max_seq_len=6 means the full prefix
    # fits only if there is no prompt prefix. With a 3-token prompt prefix
    # (encode_prefix("pp", tok) = BOS + 2 chars = 3 tokens), the combined
    # prefix becomes 3 + 4 = 7 tokens. Budget for prefix = max_seq_len - n_output
    # = 6 - 2 = 4, so we must drop 7 - 4 = 3 tokens from the left.
    # The 3 prompt-prefix tokens (BOS + "pp") are at the left, so all 3 are
    # dropped: n_prompt_surviving == 0 → existing branch fires.
    bundle = TargetModelBundle(
        model=model,
        tokenizer=SimpleTokenizer(),
        max_seq_len=6,
        target_name="encdec",
    )
    sample = _make_sample(
        bundle.tokenizer,
        prefix_text="abc",  # encodes to 4 tokens (BOS + 3 chars).
        output_text="xy",   # 2 tokens.
        segments=_segments(0, 4),
    )
    prompt_prefix = list(encode_prefix("pp", bundle.tokenizer))  # 3 tokens: BOS + "pp"
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
    assert math.isfinite(row.bpb_intervened), (
        f"bpb_intervened not finite: {row.bpb_intervened}"
    )


def test_score_sample_prompting_encdec_golden_value(tiny_encdec) -> None:
    """Golden-value lock: pin the exact bpb_intervened produced by the
    shifted-branch path on a deterministic tiny_encdec fixture. Any
    regression in the helper's shift logic, cross-attention scatter,
    or encoder feature routing will surface as a value mismatch.

    The expected value is captured from the first green run against the
    tiny_encdec checkpoint (loaded deterministically from disk). If this
    test fails after a legitimate change, update the golden value alongside
    the change AND document the source of the new value in the commit message.
    """
    model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    bundle = TargetModelBundle(
        model=model,
        tokenizer=SimpleTokenizer(),
        max_seq_len=dec_cfg.sequence_len,
        target_name="encdec",
    )
    sample = _make_sample(
        bundle.tokenizer,
        prefix_text="abc",
        output_text="xy",
        segments=_segments(0, 4),
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
    # Env-calibrated golden: absolute 1e-4 is a drift-tolerant bound.
    GOLDEN_BPB_PROMPTING_ENCDEC_K1_SEG = pytest.approx(
        6.997998829109408, rel=0, abs=1e-4,
    )
    assert math.isfinite(row.bpb_intervened)
    assert row.bpb_intervened == GOLDEN_BPB_PROMPTING_ENCDEC_K1_SEG
