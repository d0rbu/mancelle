"""Unit tests for the private padding helpers in teacher_force.py."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
import torch

from steering.eval.axbench_bpb import teacher_force as _tf
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    _DecoderPaddedInputs,
    _EncoderPaddedInputs,
    _extract_per_sample_bpb,
    _pad_decoder_inputs,
    _pad_encoder_inputs,
    _padded_idx_unique_per_row,
    _score_microbatch,
    _take_intervention_fingerprint,
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
from tests.runtime_attribute_helpers import set_runtime_attribute


def test_decoder_padded_inputs_is_frozen():
    """_DecoderPaddedInputs must be a frozen dataclass."""
    obj = _DecoderPaddedInputs(
        dec_input=torch.zeros(1, 1, dtype=torch.long),
        dec_target=torch.zeros(1, 1, dtype=torch.long),
        loss_mask=torch.zeros(1, 1, dtype=torch.bool),
        n_full_prefix=[1],
        n_output=[1],
        truncated=[False],
        full_prefixes=[[0]],
        T_max=2,
    )
    with pytest.raises(AttributeError):  # FrozenInstanceError is a subclass
        set_runtime_attribute(obj, "T_max", 3)


def test_encoder_padded_inputs_is_frozen():
    """_EncoderPaddedInputs must be a frozen dataclass."""
    obj = _EncoderPaddedInputs(
        enc_feature_ids=torch.zeros(1, 1, 1, dtype=torch.int64),
        enc_magnitudes=torch.zeros(1, 1, 1, dtype=torch.int64),
        enc_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        seg_map=torch.zeros(1, 1, dtype=torch.long),
        allowed_ids=None,  # uncompacted stand-in; irrelevant to this frozen-ness probe
    )
    with pytest.raises(AttributeError):
        set_runtime_attribute(
            obj,
            "seg_map",
            torch.ones(1, 1, dtype=torch.long),
        )


def test_model_ignore_index_is_minus_one(tiny_decoder_only):
    """Probe: GPT.forward / EncoderDecoderGPT.forward must use ignore_index=-1
    in cross_entropy. Pad-with-(-1) strategy depends on this; if any future
    change flips it to -100 (or anything else), our padded targets become
    valid class indices and cross_entropy returns nonzero loss at padded
    positions.

    Strategy: run a 1-token forward with target = -1 and assert the
    returned loss is exactly 0.0. Per-position reduction='none' so we
    can read the value directly.
    """
    model, cfg = tiny_decoder_only
    # Build a 2-token sequence so dec_input has shape (1, 1) post-shift.
    idx = torch.tensor([[0, 1]], dtype=torch.long)        # (1, 2)
    targets = torch.tensor([[1, -1]], dtype=torch.long)   # (1, 2)
    with torch.no_grad():
        loss = model(idx, targets=targets, loss_reduction="none")
    # loss shape is (B*T,) = (2,). Position 0 is real; position 1 is padded.
    assert loss.shape == (2,), f"unexpected loss shape: {loss.shape}"
    assert loss[1].item() == 0.0, (
        f"cross_entropy(target=-1) returned {loss[1].item()} (expected 0.0). "
        f"Model's ignore_index has likely changed; the pad-with-(-1) strategy "
        f"in _pad_decoder_inputs no longer suppresses padded positions."
    )


def _mk_inputs(tok, prefix_text: str, output_text: str, *, pcs=None) -> ScoringInputs:
    """Minimal ScoringInputs builder for unit tests."""
    prefix_tokens = encode_prefix(prefix_text, tok)
    output_tokens = encode_output(output_text, tok)
    byte_weights = torch.ones(len(output_tokens), dtype=torch.int64)
    if pcs is None:
        pcs = (
            ContextFeatureSegment(
                feature_ids=(),
                buckets=(),
                char_start=0,
                char_end=len(tok.decode(prefix_tokens)),
            ),
        )
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=0, mode=SteeringMode.POSITIVE,
            content_class="concept_rich",
            prefix_text=prefix_text, output_text=output_text,
        ),
        concept_id=0, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_text, prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix_tokens, output_tokens=output_tokens,
        prefix_context_segments=pcs,
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=byte_weights, truncated=False,
    )


def _mk_bundle(
    model, cfg, tok,
    target_name: Literal["decoder_only", "encdec"] = "decoder_only",
) -> TargetModelBundle:
    return TargetModelBundle(
        model=model, tokenizer=tok,
        max_seq_len=cfg.sequence_len, target_name=target_name,
    )


def test_pad_decoder_inputs_single_sample_shapes(tiny_decoder_only):
    """U1: B=1; T_max == T_1; all output tensors shaped (1, T_1-1)."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    # prefix_tokens for "abc" is [BOS, 'a', 'b', 'c'] = 4; output is "def" = 3 tokens.
    # T_1 = 4 + 3 = 7; T_max - 1 = 6.
    assert pdi.T_max == 7
    assert pdi.dec_input.shape == (1, 6)
    assert pdi.dec_target.shape == (1, 6)
    assert pdi.loss_mask.shape == (1, 6)
    assert pdi.n_full_prefix == [4]
    assert pdi.n_output == [3]
    assert pdi.truncated == [False]
    assert pdi.full_prefixes == [[0, 97, 98, 99]]


def test_pad_decoder_inputs_two_samples_pad_to_longer(tiny_decoder_only):
    """U2: B=2; T_max = max(T_1, T_2); shorter sample row gets padding tail."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    short = _mk_inputs(tok, "ab", "cd")          # T_1 = 3 + 2 = 5
    long_  = _mk_inputs(tok, "abcdef", "ghi")    # T_2 = 7 + 3 = 10
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    pdi = _pad_decoder_inputs([short, long_], iv, bundle, torch.device("cpu"))
    assert pdi.T_max == 10
    assert pdi.dec_input.shape == (2, 9)
    assert pdi.dec_target.shape == (2, 9)
    assert pdi.loss_mask.shape == (2, 9)
    # Row 0 (short): real positions 0..3, padded 4..8.
    # Row 1 (long_): real positions 0..8, no padding.
    assert pdi.n_full_prefix == [3, 7]
    assert pdi.n_output == [2, 3]


def test_pad_decoder_target_pads_with_neg_one(tiny_decoder_only):
    """U3: dec_target padding positions are exactly -1."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    short = _mk_inputs(tok, "ab", "cd")          # T = 5
    long_ = _mk_inputs(tok, "abcdef", "ghi")     # T = 10
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    pdi = _pad_decoder_inputs([short, long_], iv, bundle, torch.device("cpu"))
    # short row's real positions are 0..3 (T-1 == 4), padded 4..8 (T_max-1 == 9).
    short_row_target = pdi.dec_target[0]
    assert short_row_target[4:].tolist() == [-1, -1, -1, -1, -1], (
        f"dec_target padding positions must be -1; got {short_row_target[4:].tolist()}"
    )


def test_pad_decoder_input_pads_with_zero(tiny_decoder_only):
    """U4: dec_input padding positions are exactly 0."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    short = _mk_inputs(tok, "ab", "cd")
    long_ = _mk_inputs(tok, "abcdef", "ghi")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    pdi = _pad_decoder_inputs([short, long_], iv, bundle, torch.device("cpu"))
    short_row_input = pdi.dec_input[0]
    assert short_row_input[4:].tolist() == [0, 0, 0, 0, 0], (
        f"dec_input padding positions must be 0; got {short_row_input[4:].tolist()}"
    )


def test_pad_decoder_loss_mask_false_at_padding(tiny_decoder_only):
    """U5: loss_mask is False at prefix and padded positions, True at output positions."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    # Single sample so the row is straightforward to inspect.
    sample = _mk_inputs(tok, "ab", "xyz")  # n_full_prefix=3, n_output=3, T-1=5
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    row = pdi.loss_mask[0]
    # Output target positions occupy [n_full_prefix - 1, T - 2] = [2, 4].
    assert row.tolist() == [False, False, True, True, True]


def test_pad_decoder_with_prompting_intervention_folds_in_tokens(tiny_decoder_only):
    """U6: intervention.prompt_prefix_tokens prepends to each sample's prefix."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    sample = _mk_inputs(tok, "abc", "def")  # raw prefix len = 4 (BOS+abc)
    iv = Intervention.prompting(
        target_model="decoder_only",
        prompt_prefix_tokens=[ord("X"), ord("Y")],  # 2 tokens
    )

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    assert pdi.n_full_prefix == [6]  # 2 prompt + 4 raw prefix
    assert pdi.full_prefixes[0][:2] == [ord("X"), ord("Y")]  # prompt is at the front
    assert pdi.full_prefixes[0][2:] == [0, 97, 98, 99]       # then raw prefix
    # Output mask shifts: True at [n_full_prefix - 1, T - 2] = [5, 7]
    expected_mask = [False, False, False, False, False, True, True, True]
    assert pdi.loss_mask[0].tolist() == expected_mask


def test_pad_decoder_truncates_overflow(tiny_decoder_only):
    """U7: when n_full_prefix + n_output > max_seq_len, prefix is left-truncated;
    truncated flag propagates."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    # cfg.sequence_len is 64 (tiny model). Build an oversized prefix.
    long_prefix = "a" * 70  # 71 prefix tokens after BOS
    sample = _mk_inputs(tok, long_prefix, "xy")  # n_output=2; total=73 > 64
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    assert pdi.truncated == [True]
    assert pdi.n_full_prefix == [62]  # 64 - 2 (n_output) = 62
    assert pdi.n_output == [2]
    # Verify the truncated prefix is the rightmost 62 tokens of the original 71.
    expected_prefix = (
        list(encode_prefix(long_prefix, tok))[-62:]
    )
    assert pdi.full_prefixes[0] == expected_prefix


def test_pad_decoder_inputs_raises_on_zero_n_output(tiny_decoder_only):
    """U8: sample with n_output=0 raises AxBenchBpbError; message includes sample_id."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    # Construct a sample with output_tokens=[] (n_output=0) directly.
    bad = ScoringInputs(
        sample_id="bad-sample-id",
        concept_id=0, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="abc", prefix_hash=make_prefix_hash("abc"),
        prefix_tokens=encode_prefix("abc", tok), output_tokens=[],
        prefix_context_segments=(),
        n_prefix=4, n_output=0,
        output_byte_weights=torch.zeros(0, dtype=torch.int64),
        truncated=False,
    )
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    with pytest.raises(AxBenchBpbError, match="bad-sample-id"):
        _pad_decoder_inputs([bad], iv, bundle, torch.device("cpu"))


def test_pad_decoder_inputs_raises_on_zero_full_prefix(tiny_decoder_only):
    """U9: when truncation drives n_full_prefix to 0, raises with sample_id."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    # n_output == max_seq_len → allowed_prefix == 0 in truncate_prefix_left.
    huge_output = "y" * cfg.sequence_len  # 64 tokens
    sample = _mk_inputs(tok, "anything", huge_output)
    # Rewrap with a known-good sample_id for matching. Use dataclasses.replace
    # for type-safe field substitution (vs. **{**__dict__, ...} which ty can't
    # unify with the typed dataclass constructor).
    sample = replace(sample, sample_id="zero-prefix-id")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    with pytest.raises(AxBenchBpbError, match="zero-prefix-id"):
        _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))


def test_padded_idx_unique_per_row_passes_for_gen_slot_padding():
    """U17a: per-row generation_slot padding produces unique padded_idx values
    (mirrors the cross-attention scatter index derivation)."""
    # Sample seg_map [0,0,1,1,2,2] (T_real=6, gen_slot=2), padded with 2 to T_max=10.
    seg_map = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 2, 2, 2]], dtype=torch.long)
    n_seg_max = 3
    # Should NOT raise.
    _padded_idx_unique_per_row(seg_map, n_seg_max)


def test_padded_idx_unique_per_row_detects_collision_with_zero_padding():
    """U17b: padding the same row with 0 (gen_slot != 0) creates two non-contiguous
    runs of slot 0; pos_in_seg resets and padded_idx collides with the first run."""
    seg_map = torch.tensor([[0, 0, 1, 1, 2, 2, 0, 0, 0, 0]], dtype=torch.long)
    n_seg_max = 3
    with pytest.raises(AxBenchBpbError, match="padded_idx"):
        _padded_idx_unique_per_row(seg_map, n_seg_max)


def test_padded_idx_unique_per_row_detects_collision_with_slot_one():
    """U17c: padding with slot 1 (also < gen_slot=2) creates a second run of
    slot 1 that collides with the first."""
    seg_map = torch.tensor([[0, 0, 1, 1, 2, 2, 1, 1, 1, 1]], dtype=torch.long)
    n_seg_max = 3
    with pytest.raises(AxBenchBpbError, match="padded_idx"):
        _padded_idx_unique_per_row(seg_map, n_seg_max)


def test_pad_encoder_inputs_empty_feature_context_segment(tiny_encdec):
    """U10: one empty-feature context segment plus generation slot."""
    model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.ours(feature_conditions={42: 1, 17: 0})

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    # 2 features in feature_conditions → F_max = 2; 1 context segment + generation slot.
    assert pei.enc_feature_ids.shape == (1, 2, 2)
    assert pei.enc_magnitudes.shape == (1, 2, 2)
    assert pei.enc_mask.shape == (1, 2, 2)
    assert pei.enc_mask[0, 0].any().item() is False
    assert pei.enc_feature_ids[0, 1].tolist() == [42, 17]
    assert pei.enc_magnitudes[0, 1].tolist() == [1, 0]
    assert pei.enc_mask[0, 1].tolist() == [True, True]
    # Prefix decoder positions point to the context slot; output positions point to generation.
    assert pei.seg_map.shape == (1, pdi.T_max - 1)
    assert pei.seg_map[0, :len(sample.prefix_tokens)].tolist() == [0, 0, 0, 0]
    assert pei.seg_map[0, len(sample.prefix_tokens):].tolist() == [1, 1]


def test_pad_encoder_inputs_one_pcs_segment_two_slots(tiny_encdec):
    """U11: k=1 → enc shape (B, 2, F_max); slot 0 has prefix segment, slot 1 has gen-slot."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    seg = ContextFeatureSegment(
        feature_ids=(10, 11), buckets=(2, 3), char_start=0, char_end=4,
    )
    sample = _mk_inputs(tok, "abc", "def", pcs=(seg,))
    iv = Intervention.ours(feature_conditions={42: 1})  # 1 feature in gen slot

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    # F_max = max(2, 1) = 2; n_seg = 2.
    assert pei.enc_feature_ids.shape == (1, 2, 2)
    # Slot 0 is the prefix segment (fids 10, 11; buckets 2, 3).
    assert pei.enc_feature_ids[0, 0, :2].tolist() == [10, 11]
    assert pei.enc_magnitudes[0, 0, :2].tolist() == [2, 3]
    assert pei.enc_mask[0, 0, :2].tolist() == [True, True]
    # Slot 1 is the generation slot (fid 42; bucket 1; one feature; second slot zeros).
    assert pei.enc_feature_ids[0, 1, 0].item() == 42
    assert pei.enc_magnitudes[0, 1, 0].item() == 1
    assert pei.enc_mask[0, 1, 0].item() is True
    assert pei.enc_mask[0, 1, 1].item() is False  # padded slot in F dim


def test_pad_encoder_inputs_two_pcs_segments_three_slots(tiny_encdec):
    """U12: k=2 → enc shape (B, 3, F_max); slot ordering preserved."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    seg_a = ContextFeatureSegment(
        feature_ids=(10, 11), buckets=(2, 3), char_start=0, char_end=5,
    )
    seg_b = ContextFeatureSegment(
        feature_ids=(20,), buckets=(1,), char_start=5, char_end=11,
    )
    sample = _mk_inputs(tok, "abcdefghij", "xy", pcs=(seg_a, seg_b))
    iv = Intervention.ours(feature_conditions={})

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    assert pei.enc_feature_ids.shape == (1, 3, 2)  # F_max = max(2, 1, 1) = 2
    # Slot 0: seg_a
    assert pei.enc_feature_ids[0, 0, :2].tolist() == [10, 11]
    assert pei.enc_magnitudes[0, 0, :2].tolist() == [2, 3]
    # Slot 1: seg_b (1 feature, 1 padded)
    assert pei.enc_feature_ids[0, 1, 0].item() == 20
    assert pei.enc_mask[0, 1, 0].item() is True
    assert pei.enc_mask[0, 1, 1].item() is False
    # Slot 2: gen slot (empty)
    assert pei.enc_mask[0, 2].any().item() is False  # all-False


def test_pad_encoder_inputs_five_pcs_segments_six_slots(tiny_encdec):
    """U13: stress test with k=5."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    pcs = tuple(
        ContextFeatureSegment(
            feature_ids=(10 * (i + 1),), buckets=(i + 1,),
            char_start=i * 4, char_end=(i + 1) * 4,
        )
        for i in range(5)
    )
    sample = _mk_inputs(tok, "x" * 19, "y", pcs=pcs)
    iv = Intervention.ours(feature_conditions={})

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    assert pei.enc_feature_ids.shape == (1, 6, 1)
    for i in range(5):
        assert pei.enc_feature_ids[0, i, 0].item() == 10 * (i + 1)
        assert pei.enc_magnitudes[0, i, 0].item() == i + 1
        assert pei.enc_mask[0, i, 0].item() is True
    # gen slot (slot 5) is empty
    assert pei.enc_mask[0, 5].any().item() is False


def test_pad_encoder_inputs_mixed_pcs_lengths(tiny_encdec):
    """U14: sample[0] has k=0, sample[1] has k=2 → n_seg_max=3.
    Sample[0]'s extra slots (1, 2) are all zero with mask False."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    s0 = _mk_inputs(tok, "ab", "cd")  # k=0
    seg_a = ContextFeatureSegment(
        feature_ids=(10,), buckets=(1,), char_start=0, char_end=4,
    )
    seg_b = ContextFeatureSegment(
        feature_ids=(20,), buckets=(2,), char_start=4, char_end=8,
    )
    s1 = _mk_inputs(tok, "abcdefg", "xy", pcs=(seg_a, seg_b))  # k=2
    iv = Intervention.ours(feature_conditions={})

    pdi = _pad_decoder_inputs([s0, s1], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [s0, s1], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    assert pei.enc_feature_ids.shape == (2, 3, 1)  # n_seg_max=3, F_max=1
    # Sample 0 (k=0) has only its single (gen) slot at idx 0; slots 1, 2 are pad.
    assert pei.enc_mask[0, 1].any().item() is False
    assert pei.enc_mask[0, 2].any().item() is False
    # Sample 1 (k=2) has prefix segments at 0 and 1.
    assert pei.enc_feature_ids[1, 0, 0].item() == 10
    assert pei.enc_feature_ids[1, 1, 0].item() == 20


def test_pad_encoder_inputs_F_max_handles_varying_feature_counts(tiny_encdec):
    """U15: F_max is the max feature count across all (sample, segment) pairs."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    seg_3f = ContextFeatureSegment(
        feature_ids=(1, 2, 3), buckets=(1, 1, 1),
        char_start=0, char_end=4,
    )
    seg_1f = ContextFeatureSegment(
        feature_ids=(7,), buckets=(2,), char_start=0, char_end=4,
    )
    s0 = _mk_inputs(tok, "abc", "de", pcs=(seg_3f,))   # 3 features in seg 0
    s1 = _mk_inputs(tok, "abc", "de", pcs=(seg_1f,))   # 1 feature in seg 0
    iv = Intervention.ours(feature_conditions={})

    pdi = _pad_decoder_inputs([s0, s1], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [s0, s1], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    assert pei.enc_feature_ids.shape == (2, 2, 3)  # F_max = 3
    assert pei.enc_mask[0, 0, :3].tolist() == [True, True, True]
    assert pei.enc_mask[1, 0, :3].tolist() == [True, False, False]


def test_pad_encoder_inputs_seg_map_pads_with_generation_slot(tiny_encdec):
    """U16: per-row seg_map tail (positions ≥ T_i - 1) filled with gen_slot_i."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    s_short = _mk_inputs(tok, "ab", "cd")  # T_1 = 5
    seg_a = ContextFeatureSegment(
        feature_ids=(1,), buckets=(1,), char_start=0, char_end=5,
    )
    seg_b = ContextFeatureSegment(
        feature_ids=(2,), buckets=(2,), char_start=5, char_end=8,
    )
    s_long = _mk_inputs(tok, "abcdefg", "xy", pcs=(seg_a, seg_b))  # T_2 = 10, gen_slot=2
    iv = Intervention.ours(feature_conditions={})

    pdi = _pad_decoder_inputs([s_short, s_long], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [s_short, s_long], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )
    # T_max - 1 = 9. s_short has T_1 - 1 = 4 real positions; positions [4, 9) are pad.
    assert pei.seg_map.shape == (2, 9)
    # s_short's gen_slot is 1 (one empty-feature context segment); padded tail is 1.
    assert pei.seg_map[0, 4:].tolist() == [1, 1, 1, 1, 1]
    # s_long's gen_slot is 2; padded tail (none — T_long-1 == 9) is empty.
    assert pei.seg_map[1, 8].item() == 2  # last real position is gen slot
    # s_long has no padded tail (already T_max-1 long).


def test_pad_encoder_inputs_raises_on_seg_map_out_of_range(tiny_encdec, monkeypatch):
    """U18: a malformed scenario where seg_map.max() ≥ n_seg_max raises G10.

    Hard to trigger via real inputs (build_multi_segment_seg_map is well-
    behaved), so we monkeypatch it to return an out-of-range seg_map and
    confirm the guard fires.
    """
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.ours(feature_conditions={42: 1})

    # Force build_multi_segment_seg_map to return a seg_map with values >= 1
    # while the encoder still has n_seg_max == 1 (k=0 → gen_slot=0).
    def fake_seg_map(*, tokenizer, prefix_tokens, prefix_segments, n_output, generation_slot):
        T = len(prefix_tokens) + n_output
        return torch.full((1, T), 5, dtype=torch.long)

    monkeypatch.setattr(_tf, "build_multi_segment_seg_map", fake_seg_map)

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    with pytest.raises(AxBenchBpbError, match="seg_map.max"):
        _pad_encoder_inputs(
            [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Line-coverage tests: defensive raise branches (tests 1–8)
# ---------------------------------------------------------------------------


def test_padded_idx_unique_per_row_raises_on_non_2d_seg_map():
    """U17d: 1-D or 3-D seg_map raises clear error (not silently produces wrong result)."""
    with pytest.raises(AxBenchBpbError, match="must be 2-D"):
        _padded_idx_unique_per_row(torch.zeros(10, dtype=torch.long), n_seg_max=1)
    with pytest.raises(AxBenchBpbError, match="must be 2-D"):
        _padded_idx_unique_per_row(torch.zeros(2, 3, 4, dtype=torch.long), n_seg_max=1)


def test_pad_decoder_inputs_raises_on_empty_samples(tiny_decoder_only):
    """Internal-only guard: _pad_decoder_inputs([], ...) raises loudly."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    with pytest.raises(AxBenchBpbError, match="samples must be non-empty"):
        _pad_decoder_inputs([], iv, bundle, torch.device("cpu"))


def test_pad_encoder_inputs_raises_on_decoder_only_target(tiny_decoder_only):
    """Defensive: _pad_encoder_inputs called with target_name='decoder_only' raises."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, target_name="decoder_only")
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    with pytest.raises(AxBenchBpbError, match="only 'encdec' is supported"):
        _pad_encoder_inputs(
            [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
        )


def test_pad_encoder_inputs_raises_on_full_prefixes_length_mismatch(tiny_encdec):
    """Defensive: samples and full_prefixes lengths must match."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    s = _mk_inputs(tok, "abc", "def")
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    # Pass full_prefixes with WRONG length (2 instead of 1).
    bad_full_prefixes = [[0, 1, 2], [0, 3, 4]]  # length 2
    with pytest.raises(AxBenchBpbError, match="length mismatch"):
        _pad_encoder_inputs(
            [s], bad_full_prefixes, iv, bundle, T_max=10, device=torch.device("cpu"),
        )


def test_extract_per_sample_bpb_raises_on_non_2d_loss(tiny_decoder_only):
    """Defensive: loss_b must be 2-D (B, T_max-1). 1-D or 3-D raises."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    bad_loss = torch.zeros(6, dtype=torch.float32)  # 1-D
    with pytest.raises(AxBenchBpbError, match="loss_b must be 2-D"):
        _extract_per_sample_bpb(
            bad_loss, pdi, [sample], "decoder_only", "X", "name",
        )


def test_extract_per_sample_bpb_raises_on_batch_dim_mismatch(tiny_decoder_only):
    """Defensive: loss_b.shape[0] must match len(samples)."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    # pdi expects 1 sample; pass loss_b with B=2.
    bad_loss = torch.zeros(2, 6, dtype=torch.float32)
    with pytest.raises(AxBenchBpbError, match="batch dim"):
        _extract_per_sample_bpb(
            bad_loss, pdi, [sample], "decoder_only", "X", "name",
        )


def test_extract_per_sample_bpb_raises_on_T_dim_mismatch(tiny_decoder_only):
    """Defensive: loss_b.shape[1] must match loss_mask.shape[1]."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    sample = _mk_inputs(tok, "abc", "def")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    # "abc" prefix → 4 tokens; "def" output → 3 tokens; T_max=7; T_max-1=6.
    # Pass wrong T-dim of 5.
    bad_loss = torch.zeros(1, 5, dtype=torch.float32)
    with pytest.raises(AxBenchBpbError, match="T-dim"):
        _extract_per_sample_bpb(
            bad_loss, pdi, [sample], "decoder_only", "X", "name",
        )


def test_score_microbatch_raises_on_empty_samples(tiny_decoder_only):
    """Defensive: direct call to _score_microbatch with empty samples raises."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok)
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    with pytest.raises(AxBenchBpbError, match="_score_microbatch: samples must be non-empty"):
        _score_microbatch(
            samples=[], target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
        )


# ---------------------------------------------------------------------------
# Robustness tests: A, B, D
# ---------------------------------------------------------------------------


def test_pad_decoder_inputs_truncates_overflow_encdec(tiny_encdec):
    """U7b: truncation works on encdec target too (mirrors U7's decoder_only check)."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, target_name="encdec")
    long_prefix = "a" * (dec_cfg.sequence_len + 6)
    sample = _mk_inputs(tok, long_prefix, "xy")
    iv = Intervention.ours(feature_conditions={})

    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    assert pdi.truncated == [True]
    assert pdi.n_full_prefix == [dec_cfg.sequence_len - 2]
    assert pdi.n_output == [2]


def test_padded_idx_unique_per_row_detects_collision_in_one_of_many_rows():
    """U17e: B=3 with rows [valid, invalid, valid] — guard fires on row index 1."""
    seg_map = torch.tensor([
        [0, 0, 1, 1, 2, 2, 2, 2, 2, 2],  # valid (gen_slot=2 padding)
        [0, 0, 1, 1, 2, 2, 0, 0, 0, 0],  # invalid (slot 0 collision)
        [0, 0, 1, 1, 2, 2, 2, 2, 2, 2],  # valid
    ], dtype=torch.long)
    n_seg_max = 3
    with pytest.raises(AxBenchBpbError, match="row 1"):
        _padded_idx_unique_per_row(seg_map, n_seg_max)


def test_take_intervention_fingerprint_requires_transformer_h():
    """G16 fingerprint helper assumes model.transformer.h exists. If the model
    structure changes in the future, the fingerprint breaks — but loudly, not
    silently. This test locks the assumption."""
    import torch.nn as nn

    bare_model = nn.Linear(4, 4)
    with pytest.raises(AttributeError, match="transformer"):
        _take_intervention_fingerprint(bare_model)
