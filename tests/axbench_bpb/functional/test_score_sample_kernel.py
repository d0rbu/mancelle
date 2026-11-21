"""End-to-end tests for score_sample with tiny model fixtures.

Wires the fixtures built by Task 6.4 into real teacher-force BPB runs.
decoder_only + prompting branches are exercised here; encdec is more
involved (needs prefix_context_segments + encoder features) and uses
an empty target-delta intervention as the simplest encdec smoke.
"""
from __future__ import annotations

import math

import pytest
import torch

from steering.eval.axbench_bpb.sample_loader import (
    make_prefix_hash,
    make_sample_id,
)
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
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
from tests.runtime_attribute_helpers import set_runtime_attribute


def test_target_model_bundle_construction():
    """TargetModelBundle is a frozen dataclass — basic construction + immutability."""
    bundle = TargetModelBundle(
        model=None, tokenizer=None, max_seq_len=128, target_name="decoder_only",
    )
    assert bundle.target_name == "decoder_only"
    assert bundle.max_seq_len == 128
    with pytest.raises(AttributeError):  # FrozenInstanceError is an AttributeError subclass
        set_runtime_attribute(bundle, "max_seq_len", 256)


def _make_inputs(tok, prefix_text, output_text, concept_id=0, prefix_context_segments=()):
    """Build a minimal ScoringInputs for tests. One byte per output token
    (SimpleTokenizer is char-level)."""
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
        prefix_context_segments=prefix_context_segments,
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=byte_weights, truncated=False,
    )


def test_score_sample_decoder_only_end_to_end(tiny_decoder_only):
    """End-to-end: fixture → score_sample → BpbScoreRow with finite BPB."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    inputs = _make_inputs(bundle.tokenizer, "abc", "def")
    row = score_sample(
        inputs=inputs, target_bundle=bundle,
        intervention=Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[]),
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    )
    assert math.isfinite(row.bpb_intervened)
    # Bound BPB to a realistic range for a tiny small-vocab model:
    # tighter than the trivial `>= 0` (which accepts 0, 1, 100, inf).
    assert 0.0 <= row.bpb_intervened < 10.0, (
        f"bpb_intervened={row.bpb_intervened} outside sanity range [0, 10)"
    )
    assert row.n_output_tokens == 3
    assert row.n_output_tokens > 0
    assert row.n_output_bytes == 3
    assert row.n_output_bytes > 0
    assert row.target == "decoder_only"
    assert row.method_name == "context_only_reference"
    assert row.coefficient_str == "None"
    assert row.truncated is False


def test_score_sample_prompting_prepends_tokens(tiny_decoder_only):
    """Prompting intervention prepends prompt_prefix_tokens to the prefix.
    The intervention itself is handled inside score_sample's step 1 —
    verify the row reflects the underlying forward without errors and
    finite BPB. (Quantitative diff vs. non-prompting is a Phase 15 slow test.)"""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    inputs = _make_inputs(bundle.tokenizer, "abc", "def")
    iv = Intervention.prompting(
        target_model="decoder_only", prompt_prefix_tokens=[ord("x"), ord("y"), ord("z")],
    )
    row = score_sample(
        inputs=inputs, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="prompting_decoder_only",
    )
    assert math.isfinite(row.bpb_intervened)
    # Bounded sanity range for a tiny small-vocab model.
    assert 0.0 <= row.bpb_intervened < 10.0, (
        f"bpb_intervened={row.bpb_intervened} outside sanity range [0, 10)"
    )
    assert row.n_output_tokens == 3  # output tokens unchanged by prompting
    assert row.n_output_tokens > 0
    assert row.n_output_bytes == 3
    assert row.n_output_bytes > 0
    assert row.truncated is False
    assert row.method_name == "prompting_decoder_only"


def test_score_sample_encdec_context_only_end_to_end(tiny_encdec):
    """Encdec smoke: empty target-delta intervention. Exercises the full encdec forward
    including encoder inputs + seg_map."""
    model, dec_cfg, enc_cfg, xattn_cfg = tiny_encdec
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=dec_cfg.sequence_len, target_name="encdec",
    )
    inputs = _make_inputs(
        bundle.tokenizer,
        "abc",
        "def",
        prefix_context_segments=(
            ContextFeatureSegment(
                feature_ids=(1,),
                buckets=(1,),
                char_start=0,
                char_end=4,
            ),
        ),
    )
    iv = Intervention.ours(feature_conditions={})
    row = score_sample(
        inputs=inputs, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="0", method_name="ours_context_only_encdec",
    )
    assert math.isfinite(row.bpb_intervened)
    # Bounded sanity range for a tiny small-vocab model.
    assert 0.0 <= row.bpb_intervened < 10.0, (
        f"bpb_intervened={row.bpb_intervened} outside sanity range [0, 10)"
    )
    assert row.n_output_tokens == 3
    assert row.n_output_tokens > 0
    assert row.n_output_bytes == 3
    assert row.n_output_bytes > 0
    assert row.truncated is False
    assert row.target == "encdec"
    assert row.method_name == "ours_context_only_encdec"


def test_score_sample_encdec_missing_prefix_segments_raises(tiny_encdec):
    """Encdec scoring must not fall back to unconditioned prompt text."""
    model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=dec_cfg.sequence_len, target_name="encdec",
    )
    inputs = _make_inputs(bundle.tokenizer, "abc", "def")
    with pytest.raises(Exception, match="prefix_context_segments"):
        score_sample(
            inputs=inputs, target_bundle=bundle,
            intervention=Intervention.ours(feature_conditions={}),
            device=torch.device("cpu"),
            coefficient_str="0", method_name="ours_context_only_encdec",
        )
