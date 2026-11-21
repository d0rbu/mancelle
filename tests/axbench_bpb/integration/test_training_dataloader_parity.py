"""Byte-exact parity of the seg_map builder against the training
dataloader's segmenter for identical inputs."""
from __future__ import annotations

from pathlib import Path

import torch

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_index import SegmentFeatures
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    _pad_decoder_inputs,
    _pad_encoder_inputs,
)
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import Intervention, ScoringInputs, SteeringMode
from steering.eval.multiseg_generator import build_multiseg_inputs
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer


def test_seg_map_parity_with_build_multiseg_inputs():
    """Ensure our seg_map matches build_multiseg_inputs' seg_map
    for identical inputs - they are supposed to be the same routine."""
    tok = SimpleTokenizer()
    prefix_tokens = list(tok.encode("abcdefghij"))  # 10 tokens
    # Two segments: chars [0, 4) and [4, 10)
    context_segments = (
        ContextFeatureSegment(feature_ids=(10, 11), buckets=(3, 2),
                              char_start=0, char_end=4),
        ContextFeatureSegment(feature_ids=(20,), buckets=(4,),
                              char_start=4, char_end=10),
    )
    n_output = 3

    # axbench-bpb seg_map
    seg_map_bpb = build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=prefix_tokens,
        prefix_segments=context_segments,
        n_output=n_output,
        generation_slot=2,  # two prefix segs + one target
    )

    # Reference path: translate context_segments to SegmentFeatures and
    # call build_multiseg_inputs for parity.
    segment_features = [
        SegmentFeatures(
            segment_idx=i,
            position=0,
            feature_ids=torch.tensor(cs.feature_ids, dtype=torch.int64),
            feature_mags=torch.tensor(cs.buckets, dtype=torch.int64),
            char_start=cs.char_start,
            char_end=cs.char_end,
        )
        for i, cs in enumerate(context_segments)
    ]
    # Append an empty "target" segment so build_multiseg_inputs has a
    # target_segment_idx to route generated tokens to.
    segment_features.append(
        SegmentFeatures(
            segment_idx=len(context_segments),
            position=0,
            feature_ids=torch.tensor([], dtype=torch.int64),
            feature_mags=torch.tensor([], dtype=torch.int64),
            char_start=10, char_end=10,
        )
    )
    ref_inputs = build_multiseg_inputs(
        segments=segment_features,
        prefix_tokens=prefix_tokens,
        n_gen_tokens=n_output,
        target_segment_idx=len(context_segments),
        tokenizer=tok,
    )
    assert torch.equal(seg_map_bpb, ref_inputs.seg_map), (
        f"seg_map mismatch:\n  axbench-bpb: {seg_map_bpb}\n  reference:   {ref_inputs.seg_map}"
    )


def test_batched_encoder_tensors_parity_with_dataloader():
    """Byte-exact parity on (enc_fids, enc_mags, enc_mask, seg_map) between
    axbench-bpb's _pad_encoder_inputs (batched path) and the training dataloader's
    build_multiseg_inputs (single-row reference).

    For each sample we (a) build the per-row encoder tensors via
    build_multiseg_inputs (which the training dataloader uses for live data)
    and (b) build the per-microbatch tensors via _pad_encoder_inputs. The
    per-row b=0 slice of (b) must equal (a) exactly (after F-padding via
    F_max).
    """
    # Load tiny_encdec checkpoint inline (no fixture dependency from this file).
    ckpt_path = Path(__file__).resolve().parent.parent / "fixtures" / "tiny_encdec" / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec_cfg = GPTConfig(**state["dec_config"])
    enc_cfg = EncoderConfig(**state["enc_config"])
    xattn_cfg = CrossAttentionConfig(**state["xattn_config"])
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config=dec_cfg, enc_config=enc_cfg, xattn_config=xattn_cfg,
            n_sae_features=state["n_sae_features"],
            num_buckets=state["num_buckets"],
            pad_vocab_size_to=state["pad_vocab_size_to"],
        )
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.eval()

    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model, tokenizer=tok, max_seq_len=dec_cfg.sequence_len,
        target_name="encdec",
    )

    # Build a sample with prefix_context_segments populated.
    # encode_prefix prepends a 1-char BOS (chr(0)) so the decoded prefix is
    # chr(0) + "abcdefghij" = 11 chars [0, 11).  Segment char ranges must
    # cover the full decoded extent; seg_b therefore ends at char 11 and the
    # generation slot is anchored at char 11.
    seg_a = ContextFeatureSegment(
        feature_ids=(10, 11), buckets=(2, 3), char_start=0, char_end=5,
    )
    seg_b = ContextFeatureSegment(
        feature_ids=(20,), buckets=(1,), char_start=5, char_end=11,
    )
    prefix_text = "abcdefghij"
    output_text = "xy"
    prefix_tokens = encode_prefix(prefix_text, tok)
    output_tokens = encode_output(output_text, tok)
    sample = ScoringInputs(
        sample_id=make_sample_id(
            concept_id=0, mode=SteeringMode.POSITIVE, content_class="concept_rich",
            prefix_text=prefix_text, output_text=output_text,
        ),
        concept_id=0, mode=SteeringMode.POSITIVE, content_class="concept_rich",
        prefix_text=prefix_text, prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix_tokens, output_tokens=output_tokens,
        prefix_context_segments=(seg_a, seg_b),
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=torch.ones(len(output_tokens), dtype=torch.int64),
        truncated=False,
    )
    iv = Intervention.ours(feature_conditions={42: 2})

    # axbench-bpb path
    pdi = _pad_decoder_inputs([sample], iv, bundle, torch.device("cpu"))
    pei = _pad_encoder_inputs(
        [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, torch.device("cpu"),
    )

    # Reference path: build_multiseg_inputs with the same structure.
    segment_features = [
        SegmentFeatures(
            segment_idx=0, position=0,
            feature_ids=torch.tensor((10, 11), dtype=torch.int64),
            feature_mags=torch.tensor((2, 3), dtype=torch.int64),
            char_start=0, char_end=5,
        ),
        SegmentFeatures(
            segment_idx=1, position=0,
            feature_ids=torch.tensor((20,), dtype=torch.int64),
            feature_mags=torch.tensor((1,), dtype=torch.int64),
            char_start=5, char_end=11,
        ),
        # Generation slot
        SegmentFeatures(
            segment_idx=2, position=0,
            feature_ids=torch.tensor((42,), dtype=torch.int64),
            feature_mags=torch.tensor((2,), dtype=torch.int64),
            char_start=11, char_end=11,
        ),
    ]
    ref = build_multiseg_inputs(
        segments=segment_features,
        prefix_tokens=prefix_tokens,
        n_gen_tokens=len(output_tokens),
        target_segment_idx=2,
        tokenizer=tok,
    )

    # The reference produces (1, 3, F_ref); _pad_encoder_inputs produces
    # (1, 3, F_ours). The two F_max values may differ if axbench-bpb path
    # doesn't see all features; assert the populated entries match.
    F_ref = ref.enc_feature_ids.shape[2]
    F_ours = pei.enc_feature_ids.shape[2]
    assert F_ref == F_ours, (
        f"F_max differs: ref={F_ref}, axbench-bpb={F_ours}. "
        f"Encoder packing logic diverged."
    )
    assert torch.equal(pei.enc_feature_ids[:, :, :F_ref], ref.enc_feature_ids), (
        f"enc_feature_ids drift:\n  bpb={pei.enc_feature_ids}\n  ref={ref.enc_feature_ids}"
    )
    assert torch.equal(pei.enc_magnitudes[:, :, :F_ref], ref.enc_magnitudes)
    assert torch.equal(pei.enc_mask[:, :, :F_ref], ref.enc_mask)
    # seg_map: the reference shape is (1, T) including the final position;
    # axbench-bpb's _pad_encoder_inputs returns (1, T-1) (post-shift slice).
    # Compare on overlap.
    T_minus_1 = pei.seg_map.shape[1]
    assert torch.equal(pei.seg_map, ref.seg_map[:, :T_minus_1])
