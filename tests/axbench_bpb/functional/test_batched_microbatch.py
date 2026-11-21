"""Functional tests for the batched _score_microbatch path."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from steering.eval.axbench_base.baselines.hooks import register_residual_hook
from steering.eval.axbench_base.baselines.lora.loader import attach_lora_from_disk
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb import teacher_force as tf
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    score_sample,
    score_triple,
)
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    LoraParams,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

# Locked golden BPB values from the spec's Section 2 reproducer (tiny_encdec, seed=12345).
GOLDEN_BPB_EMPTY_PCS_CONTEXT_ONLY = 6.996944002184116
GOLDEN_BPB_K1_PCS_CONTEXT_ONLY   = 6.994442227889565
GOLDEN_BPB_K2_PCS_CONTEXT_ONLY   = 7.0046236008206435
GOLDEN_BPB_K2_PCS_OURS_POPULATED = 7.00536243828875

# Match fixture constants from regenerate_fixtures.py
TINY_LORA_LAYER = 1
TINY_LORA_RANK = 4
TINY_LORA_ALPHA = 8.0


def _adapter_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "fixtures"
        / "tiny_decoder_only" / "lora_adapter_concept_0000.pt"
    )


def _direction(dim: int = 32, *, seed: int = 0) -> torch.Tensor:
    """Reproducible direction vector matching tiny_decoder_only n_embd=32."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g, dtype=torch.float32)


def _mk_inputs(tok, prefix_text: str, output_text: str, *, pcs=None) -> ScoringInputs:
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


def _mk_bundle(model, cfg, tok, target_name) -> TargetModelBundle:
    return TargetModelBundle(
        model=model, tokenizer=tok,
        max_seq_len=cfg.sequence_len, target_name=target_name,
    )


def _solo_bpbs(samples, bundle, iv, **kwargs):
    """Reference: per-sample score_sample loop."""
    return {
        s.sample_id: score_sample(
            inputs=s, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            **kwargs,
        ).bpb_intervened
        for s in samples
    }


def test_encdec_context_only_empty_pcs_cross_batch_size_determinism(tiny_encdec):
    """F11: encdec + empty-feature context segments — bitwise across B ∈ {1,2,4,8}."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])

    results: dict[int, dict[str, float]] = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}

    ref = results[1]
    for bs, got in results.items():
        for sid, bpb in ref.items():
            assert got[sid] == bpb, (
                f"batch_size={bs} sample {sid}: {got[sid]} vs {bpb} (must be byte-equal)"
            )


def test_encdec_context_only_empty_pcs_padding_isolation(tiny_encdec):
    """F12: short sample alone vs batched with long sample → byte-equal BPB."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])

    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_encdec_ours_empty_fc_cross_batch_size_determinism(tiny_encdec):
    """F13."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.ours(feature_conditions={})

    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="0", method_name="ours_empty",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_encdec_ours_empty_fc_padding_isolation(tiny_encdec):
    """F14."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.ours(feature_conditions={})
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="0", method_name="ours_empty",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="0", method_name="ours_empty",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_encdec_ours_populated_fc_cross_batch_size_determinism(tiny_encdec):
    """F15."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.ours(feature_conditions={42: 1, 17: 0})
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="1", method_name="ours_populated",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_encdec_ours_populated_fc_padding_isolation(tiny_encdec):
    """F16."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.ours(feature_conditions={42: 1, 17: 0})
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="1", method_name="ours_populated",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="1", method_name="ours_populated",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_encdec_prompting_cross_batch_size_determinism(tiny_encdec):
    """F17."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.prompting(
        target_model="encdec", prompt_prefix_tokens=[ord("X"), ord("Y")],
    )
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="prompting_encdec",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_encdec_prompting_padding_isolation(tiny_encdec):
    """F18."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.prompting(
        target_model="encdec", prompt_prefix_tokens=[ord("X"), ord("Y")],
    )
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="prompting_encdec",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="prompting_encdec",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_encdec_context_only_nonempty_pcs_finite_bpb_known_value(tiny_encdec):
    """F19: 2-PCS sample, Intervention.prompting → BPB == GOLDEN_BPB_K2_PCS_CONTEXT_ONLY.
    Locked from the spec's Section 2 reproducer."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    seg_a = ContextFeatureSegment(
        feature_ids=(10, 11), buckets=(2, 3), char_start=0, char_end=5,
    )
    seg_b = ContextFeatureSegment(
        feature_ids=(20,), buckets=(1,), char_start=5, char_end=11,
    )
    sample = _mk_inputs(tok, "abcdefghij", "xy", pcs=(seg_a, seg_b))
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])

    row = score_sample(
        inputs=sample, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    )
    # Env-calibrated golden: absolute 1e-4 is a drift-tolerant bound.
    assert row.bpb_intervened == pytest.approx(
        GOLDEN_BPB_K2_PCS_CONTEXT_ONLY, rel=0, abs=1e-4,
    ), (
        f"BPB drift: got {row.bpb_intervened}, expected {GOLDEN_BPB_K2_PCS_CONTEXT_ONLY}"
    )


def test_encdec_context_only_nonempty_pcs_cross_batch_size_determinism(tiny_encdec):
    """F20: k=2 PCS samples scored at B ∈ {1,2,4,8} → byte-equal."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    pcs = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=4),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=4, char_end=8),
    )
    samples = [
        _mk_inputs(tok, "abcdefg", f"o{i}", pcs=pcs) for i in range(8)
    ]
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_encdec_context_only_nonempty_pcs_padding_isolation(tiny_encdec):
    """F21: k=2 short sample alone vs batched with k=0 long sample → byte-equal."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    pcs = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=3, char_end=5),
    )
    short = _mk_inputs(tok, "abcd", "yz", pcs=pcs)  # short sample with PCS
    long_ = _mk_inputs(tok, "z" * 30, "x" * 5)      # long sample without PCS
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_encdec_context_only_equivalent_to_ours_empty_fc_with_pcs(tiny_encdec):
    """F22: Doc invariant lock — Intervention.prompting ≡ Intervention.ours({}) BPB byte-equal."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    pcs = (
        ContextFeatureSegment(feature_ids=(10, 11), buckets=(2, 3), char_start=0, char_end=5),
        ContextFeatureSegment(feature_ids=(20,), buckets=(1,), char_start=5, char_end=11),
    )
    sample = _mk_inputs(tok, "abcdefghij", "xy", pcs=pcs)

    bpb_context_only = score_sample(
        inputs=sample, target_bundle=bundle,
        intervention=Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[]),
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    ).bpb_intervened
    bpb_ours_empty = score_sample(
        inputs=sample, target_bundle=bundle,
        intervention=Intervention.ours(feature_conditions={}),
        device=torch.device("cpu"),
        coefficient_str="0", method_name="ours_empty",
    ).bpb_intervened
    assert bpb_context_only == bpb_ours_empty
    # Env-calibrated golden: absolute 1e-4 is a drift-tolerant bound.
    assert bpb_context_only == pytest.approx(
        GOLDEN_BPB_K2_PCS_CONTEXT_ONLY, rel=0, abs=1e-4,
    )


def test_encdec_ours_populated_with_pcs_changes_bpb(tiny_encdec):
    """F23: BPB(ours_populated) != BPB(ours_empty) by > 1e-5."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    pcs = (
        ContextFeatureSegment(feature_ids=(10, 11), buckets=(2, 3), char_start=0, char_end=5),
        ContextFeatureSegment(feature_ids=(20,), buckets=(1,), char_start=5, char_end=11),
    )
    sample = _mk_inputs(tok, "abcdefghij", "xy", pcs=pcs)
    bpb_pop = score_sample(
        inputs=sample, target_bundle=bundle,
        intervention=Intervention.ours(feature_conditions={42: 1}),
        device=torch.device("cpu"),
        coefficient_str="1", method_name="ours_populated",
    ).bpb_intervened
    bpb_empty = score_sample(
        inputs=sample, target_bundle=bundle,
        intervention=Intervention.ours(feature_conditions={}),
        device=torch.device("cpu"),
        coefficient_str="0", method_name="ours_empty",
    ).bpb_intervened
    assert abs(bpb_pop - bpb_empty) > 1e-5
    # Env-calibrated golden: absolute 1e-4 is a drift-tolerant bound.
    assert bpb_pop == pytest.approx(
        GOLDEN_BPB_K2_PCS_OURS_POPULATED, rel=0, abs=1e-4,
    )


def test_encdec_pcs_perturbation_changes_bpb(tiny_encdec):
    """F24: changing one feature_id in prefix segment 1 changes the BPB —
    locks that prefix-slot wiring is real."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    pcs_a = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(2,), char_start=0, char_end=5),
        ContextFeatureSegment(feature_ids=(20,), buckets=(1,), char_start=5, char_end=11),
    )
    pcs_b = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(2,), char_start=0, char_end=5),
        ContextFeatureSegment(feature_ids=(50,), buckets=(1,), char_start=5, char_end=11),  # changed fid 20→50
    )
    sample_a = _mk_inputs(tok, "abcdefghij", "xy", pcs=pcs_a)
    sample_b = _mk_inputs(tok, "abcdefghij", "xy", pcs=pcs_b)
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    bpb_a = score_sample(
        inputs=sample_a, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    ).bpb_intervened
    bpb_b = score_sample(
        inputs=sample_b, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    ).bpb_intervened
    assert bpb_a != bpb_b


def test_encdec_mixed_pcs_lengths_in_batch_match_solo(tiny_encdec):
    """F25: batch of [k=0, k=1, k=2] samples → each BPB byte-equal to solo."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    # Distinct texts so each sample gets a unique sample_id.
    # encode_prefix prepends BOS (chr(0)) so "abcd" decodes to "\x00abcd" (5 chars).
    # PCS char ranges must cover the full decoded prefix including the BOS char.
    s0 = _mk_inputs(tok, "abc", "de")
    pcs_1 = (ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=5),)
    s1 = _mk_inputs(tok, "abcd", "fg", pcs=pcs_1)
    pcs_2 = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=4),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=4, char_end=8),
    )
    s2 = _mk_inputs(tok, "abcdefg", "yz", pcs=pcs_2)
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    solo_bpbs = _solo_bpbs([s0, s1, s2], bundle, iv)
    rows = score_triple(
        samples=[s0, s1, s2], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=3,
    )
    for r in rows:
        assert r.bpb_intervened == solo_bpbs[r.sample_id], (
            f"sample {r.sample_id}: batched {r.bpb_intervened} != solo {solo_bpbs[r.sample_id]}"
        )


def test_encdec_max_pcs_segments_in_batch_works(tiny_encdec):
    """F26: stress — 5-segment PCS in a batch of 4 → all finite, all match solo."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    # buckets must be in [0, num_buckets-1=3]; use modular wrap to stay in range.
    pcs_5 = tuple(
        ContextFeatureSegment(
            feature_ids=(10 * (i + 1),), buckets=(i % 4,),
            char_start=i * 4, char_end=(i + 1) * 4,
        )
        for i in range(5)
    )
    samples = [_mk_inputs(tok, "x" * 19, f"y{i}", pcs=pcs_5) for i in range(4)]
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    solo_bpbs = _solo_bpbs(samples, bundle, iv)
    rows = score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=4,
    )
    for r in rows:
        assert r.bpb_intervened == solo_bpbs[r.sample_id]


def test_decoder_only_context_only_cross_batch_size_determinism(tiny_decoder_only):
    """F1."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_decoder_only_context_only_padding_isolation(tiny_decoder_only):
    """F2."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_decoder_only_prompting_cross_batch_size_determinism(tiny_decoder_only):
    """F3."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.prompting(
        target_model="decoder_only", prompt_prefix_tokens=[ord("X"), ord("Y")],
    )
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="prompting_decoder_only",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_decoder_only_prompting_padding_isolation(tiny_decoder_only):
    """F4."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.prompting(
        target_model="decoder_only", prompt_prefix_tokens=[ord("X"), ord("Y")],
    )
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="prompting_decoder_only",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="prompting_decoder_only",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_decoder_only_diffmean_cross_batch_size_determinism(tiny_decoder_only):
    """F5."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=_direction(cfg.n_embd, seed=1),
        direction_layer=1, coefficient=1.0,
    )
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="1.0", method_name="diffmean",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_decoder_only_diffmean_padding_isolation(tiny_decoder_only):
    """F6."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=_direction(cfg.n_embd, seed=1),
        direction_layer=1, coefficient=1.0,
    )
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="1.0", method_name="diffmean",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="1.0", method_name="diffmean",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_decoder_only_linear_probe_cross_batch_size_determinism(tiny_decoder_only):
    """F7."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.from_direction(
        kind="linear_probe", target_model="decoder_only",
        direction=_direction(cfg.n_embd, seed=2),
        direction_layer=1, coefficient=0.5,
    )
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="0.5", method_name="linear_probe",
            device_batch_size=bs,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_decoder_only_linear_probe_padding_isolation(tiny_decoder_only):
    """F8."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.from_direction(
        kind="linear_probe", target_model="decoder_only",
        direction=_direction(cfg.n_embd, seed=2),
        direction_layer=1, coefficient=0.5,
    )
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="0.5", method_name="linear_probe",
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="0.5", method_name="linear_probe",
        device_batch_size=2,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_decoder_only_lora_cross_batch_size_determinism(tiny_decoder_only):
    """F9."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.lora(adapter_path=_adapter_path())
    lora = LoraParams(
        cfg=LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA),
        layer=TINY_LORA_LAYER,
    )
    results = {}
    for bs in [1, 2, 4, 8]:
        rows = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="lora_decoder_only",
            device_batch_size=bs,
            lora=lora,
        )
        results[bs] = {r.sample_id: r.bpb_intervened for r in rows}
    ref = results[1]
    for got in results.values():
        for sid, bpb in ref.items():
            assert got[sid] == bpb


def test_decoder_only_lora_padding_isolation(tiny_decoder_only):
    """F10."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    short = _mk_inputs(tok, "x", "y")
    long_ = _mk_inputs(tok, "a" * 30, "b" * 5)
    iv = Intervention.lora(adapter_path=_adapter_path())
    lora = LoraParams(
        cfg=LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA),
        layer=TINY_LORA_LAYER,
    )
    solo = score_sample(
        inputs=short, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="lora_decoder_only",
        lora=lora,
    )
    batched = score_triple(
        samples=[long_, short], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="lora_decoder_only",
        device_batch_size=2,
        lora=lora,
    )
    got = next(r for r in batched if r.sample_id == short.sample_id)
    assert got.bpb_intervened == solo.bpb_intervened


def test_score_triple_detects_intervention_leaked_hook(
    tiny_decoder_only, monkeypatch,
):
    """F30: a monkeypatched apply_intervention that 'forgets' to remove the hook
    must trigger the G16 fingerprint check and raise AxBenchBpbError."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(2)]
    iv = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=_direction(cfg.n_embd, seed=1),
        direction_layer=1, coefficient=1.0,
    )

    @contextmanager
    def leaky_apply_intervention(model, intervention, *, lora=None):
        # Register a hook but FORGET to remove it.
        _ = register_residual_hook(
            model, intervention.direction_layer,
            intervention.direction, intervention.direction_coefficient,
        )
        try:
            yield
        finally:
            pass  # leak: don't call handle.remove()

    monkeypatch.setattr(tf, "apply_intervention", leaky_apply_intervention)

    with pytest.raises(AxBenchBpbError, match="left state behind"):
        score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="1.0", method_name="diffmean",
            device_batch_size=2,
        )


def test_score_triple_detects_intervention_leaked_lora(
    tiny_decoder_only, monkeypatch,
):
    """F31: a monkeypatched apply_intervention that skips LoRA detach must
    trigger the G16 fingerprint check."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(2)]
    iv = Intervention.lora(adapter_path=_adapter_path())
    lora = LoraParams(
        cfg=LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA),
        layer=TINY_LORA_LAYER,
    )

    @contextmanager
    def leaky_apply_intervention(model, intervention, *, lora=None):
        assert lora is not None
        attach_lora_from_disk(
            model=model, adapter_path=intervention.lora_adapter_path,
            lora_cfg=lora.cfg, layer=lora.layer,
        )
        try:
            yield
        finally:
            pass  # leak: don't call detach_fn

    monkeypatch.setattr(tf, "apply_intervention", leaky_apply_intervention)

    with pytest.raises(AxBenchBpbError, match="left state behind"):
        score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="lora_decoder_only",
            device_batch_size=2,
            lora=lora,
        )


def test_microbatch_raises_with_sample_id_on_non_finite_loss(
    tiny_decoder_only, monkeypatch,
):
    """F27: monkeypatch model.forward to return inf loss → AxBenchBpbError
    mentions sample_id, method_name, coefficient_str."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"abc{i}", "def") for i in range(2)]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    real_forward = type(model).forward

    def inf_forward(self, idx, *args, **kwargs):
        out = real_forward(self, idx, *args, **kwargs)
        # Replace ALL positions with inf to guarantee in-mask non-finiteness.
        return torch.full_like(out, float("inf"))

    monkeypatch.setattr(type(model), "forward", inf_forward)

    with pytest.raises(AxBenchBpbError, match=samples[0].sample_id):
        score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="my_coef", method_name="my_method",
            device_batch_size=2,
        )


def test_microbatch_raises_on_unknown_loss_shape(
    tiny_decoder_only, monkeypatch,
):
    """F28: monkeypatch model.forward to return a tuple → AxBenchBpbError
    mentions 'model API drift' or returned type."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, "abc", "def") for _ in range(2)]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    def tuple_forward(self, idx, *args, **kwargs):
        return ("not", "a", "tensor")

    monkeypatch.setattr(type(model), "forward", tuple_forward)

    with pytest.raises(AxBenchBpbError, match="tuple|expected torch.Tensor"):
        score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=2,
        )


def test_microbatch_raises_on_wrong_shape_tensor_loss(tiny_decoder_only, monkeypatch):
    """F28b: monkeypatch model.forward to return a Tensor of wrong shape.
    Hits the loss.shape != expected_shape branch (separate from the isinstance check)."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(2)]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    # Return a Tensor of wrong shape (e.g., a 1-element tensor).
    def wrong_shape_forward(self, idx, *args, **kwargs):
        return torch.zeros(7, dtype=torch.float32)  # arbitrary wrong shape

    monkeypatch.setattr(type(model), "forward", wrong_shape_forward)

    with pytest.raises(AxBenchBpbError, match="loss.shape"):
        score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=2,
        )


def test_encdec_diffmean_with_pcs_runs_and_amortizes(tiny_encdec):
    """Cross-coverage: encdec target + non-empty PCS + diffmean intervention.
    Combines the multi-segment encoder fix with the residual-hook intervention
    install — exercises both at once. Asserts batched matches solo byte-equal."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    pcs = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=4),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=4, char_end=8),
    )
    samples = [_mk_inputs(tok, "abcdefg", f"o{i}", pcs=pcs) for i in range(4)]
    iv = Intervention.from_direction(
        kind="diffmean", target_model="encdec",
        direction=_direction(dec_cfg.n_embd, seed=3),
        direction_layer=1, coefficient=0.7,
    )
    solo = {
        s.sample_id: score_sample(
            inputs=s, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="0.7", method_name="diffmean_encdec",
        ).bpb_intervened
        for s in samples
    }
    rows = score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="0.7", method_name="diffmean_encdec",
        device_batch_size=2,
    )
    for r in rows:
        assert r.bpb_intervened == solo[r.sample_id]


# ---------------------------------------------------------------------------
# Stronger correctness gates: per-position loss equality across batch sizes
# ---------------------------------------------------------------------------
#
# The cross-batch determinism + padding-isolation tests above all assert that
# per-sample BPB is byte-equal across batch sizes. BPB is a sum-reduction over
# output positions, so in principle errors at individual positions could cancel
# and produce a byte-equal sum despite differing per-position losses. The
# probability is vanishingly small (sub-double-precision ULP-level cancellation
# would be required across heterogeneous positions), but we still want a direct
# check for absolute peace of mind.
#
# These tests reach inside the helpers and assert that the per-position model
# loss tensor at the REAL output positions is byte-equal between B=1 and B>1
# runs. This is the strongest possible correctness gate — it makes "but maybe
# the cancellation is masking a bug" impossible.


_CPU_DEVICE = torch.device("cpu")


def _per_position_loss_for_one_sample(
    sample: ScoringInputs,
    bundle: TargetModelBundle,
    iv: Intervention,
    *,
    device: torch.device = _CPU_DEVICE,
    lora: LoraParams | None = None,
) -> torch.Tensor:
    """Run a B=1 forward through _score_microbatch's helpers and return the
    extracted per-position output-loss tensor (shape: (n_output_i,)).

    Uses the SAME helpers as the production batched path (so a bug in the
    helpers would show up here too); compares this against per-position loss
    extracted from a B>1 microbatch run on a different test below.
    """
    from steering.eval.axbench_bpb.intervention import apply_intervention
    from steering.eval.axbench_bpb.teacher_force import (
        _pad_decoder_inputs,
        _pad_encoder_inputs,
    )

    pdi = _pad_decoder_inputs([sample], iv, bundle, device)
    if bundle.target_name == "encdec":
        pei = _pad_encoder_inputs(
            [sample], pdi.full_prefixes, iv, bundle, pdi.T_max, device,
        )
    with torch.no_grad():
        with apply_intervention(
            bundle.model, iv, lora=lora,
        ):
            if bundle.target_name == "encdec":
                loss = bundle.model(
                    pdi.dec_input, pdi.dec_target,
                    enc_feature_ids=pei.enc_feature_ids,
                    enc_magnitudes=pei.enc_magnitudes,
                    enc_mask=pei.enc_mask,
                    seg_map=pei.seg_map,
                    loss_reduction="none",
                )
            else:
                loss = bundle.model(
                    pdi.dec_input, targets=pdi.dec_target, loss_reduction="none",
                )
    loss_b = loss.view(1, pdi.T_max - 1)
    return loss_b[0][pdi.loss_mask[0]].clone()  # only at real output positions


def _per_position_loss_for_sample_in_batch(
    target_sample: ScoringInputs,
    other_samples: list[ScoringInputs],
    bundle: TargetModelBundle,
    iv: Intervention,
    *,
    device: torch.device = _CPU_DEVICE,
    lora: LoraParams | None = None,
) -> torch.Tensor:
    """Run a B=len(other_samples)+1 forward and extract the output-position
    loss tensor for `target_sample` only. Returns shape (n_output,)."""
    from steering.eval.axbench_bpb.intervention import apply_intervention
    from steering.eval.axbench_bpb.teacher_force import (
        _pad_decoder_inputs,
        _pad_encoder_inputs,
    )

    samples = [*other_samples, target_sample]
    target_idx = len(samples) - 1
    pdi = _pad_decoder_inputs(samples, iv, bundle, device)
    if bundle.target_name == "encdec":
        pei = _pad_encoder_inputs(
            samples, pdi.full_prefixes, iv, bundle, pdi.T_max, device,
        )
    with torch.no_grad():
        with apply_intervention(
            bundle.model, iv, lora=lora,
        ):
            if bundle.target_name == "encdec":
                loss = bundle.model(
                    pdi.dec_input, pdi.dec_target,
                    enc_feature_ids=pei.enc_feature_ids,
                    enc_magnitudes=pei.enc_magnitudes,
                    enc_mask=pei.enc_mask,
                    seg_map=pei.seg_map,
                    loss_reduction="none",
                )
            else:
                loss = bundle.model(
                    pdi.dec_input, targets=pdi.dec_target, loss_reduction="none",
                )
    loss_b = loss.view(len(samples), pdi.T_max - 1)
    return loss_b[target_idx][pdi.loss_mask[target_idx]].clone()


def test_per_position_loss_byte_equal_across_batch_sizes_decoder_only(tiny_decoder_only):
    """Strongest correctness gate (decoder_only): for each sample, the
    per-position loss tensor at REAL output positions must be byte-equal
    between B=1 and B=4 runs. Detects per-position cancellation that BPB-only
    tests would mask."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"prefix{i}" + "a" * (i % 3), f"o{i}") for i in range(4)]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    for i, target in enumerate(samples):
        solo = _per_position_loss_for_one_sample(target, bundle, iv)
        in_batch = _per_position_loss_for_sample_in_batch(
            target, [s for j, s in enumerate(samples) if j != i], bundle, iv,
        )
        assert torch.equal(solo, in_batch), (
            f"sample {i} ({target.sample_id!r}): per-position loss differs "
            f"between solo and in-batch runs.\n  solo:    {solo.tolist()}\n  "
            f"batched: {in_batch.tolist()}"
        )


def test_per_position_loss_byte_equal_across_batch_sizes_encdec_empty_pcs(tiny_encdec):
    """Strongest correctness gate (encdec, empty-feature context): per-position loss
    byte-equal between B=1 and B=4 runs."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    samples = [_mk_inputs(tok, f"prefix{i}" + "a" * (i % 3), f"o{i}") for i in range(4)]
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])

    for i, target in enumerate(samples):
        solo = _per_position_loss_for_one_sample(target, bundle, iv)
        in_batch = _per_position_loss_for_sample_in_batch(
            target, [s for j, s in enumerate(samples) if j != i], bundle, iv,
        )
        assert torch.equal(solo, in_batch), (
            f"sample {i} ({target.sample_id!r}): per-position encdec loss "
            f"differs between solo and in-batch runs.\n  solo:    {solo.tolist()}\n"
            f"  batched: {in_batch.tolist()}"
        )


def test_per_position_loss_byte_equal_across_batch_sizes_encdec_context_onlympty_pcs(tiny_encdec):
    """Strongest correctness gate (encdec, NON-empty PCS): per-position
    loss byte-equal between B=1 and B=4 runs WITH multi-segment encoder.
    This is the highest-stakes scenario — the multi-segment scatter must
    be free of any cross-row contamination."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    # prefix_text "abcdefg{i}" = 8 chars, BOS-prepended decoded length = 9.
    # PCS must cover [0, 9): split [0, 5) | [5, 9).
    pcs = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=5),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=5, char_end=9),
    )
    samples = [_mk_inputs(tok, f"abcdefg{i}", f"o{i}", pcs=pcs) for i in range(4)]
    iv = Intervention.ours(feature_conditions={42: 1})

    for i, target in enumerate(samples):
        solo = _per_position_loss_for_one_sample(target, bundle, iv)
        in_batch = _per_position_loss_for_sample_in_batch(
            target, [s for j, s in enumerate(samples) if j != i], bundle, iv,
        )
        assert torch.equal(solo, in_batch), (
            f"sample {i} ({target.sample_id!r}): per-position multi-segment "
            f"encdec loss differs between solo and in-batch runs — possible "
            f"cross-attention scatter contamination.\n  solo:    {solo.tolist()}\n"
            f"  batched: {in_batch.tolist()}"
        )


def test_per_position_loss_byte_equal_with_short_padded_alongside_long(tiny_encdec):
    """Targeted: a short sample's per-position loss must be byte-equal whether
    scored solo or alongside a much longer sample (which forces heavy padding
    on the short row). Catches any scatter / mask leak introduced by padding."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    short = _mk_inputs(tok, "x", "y")          # T_short = 3
    long_ = _mk_inputs(tok, "a" * 25, "b" * 5) # T_long = 32
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])

    solo = _per_position_loss_for_one_sample(short, bundle, iv)
    in_batch = _per_position_loss_for_sample_in_batch(
        short, [long_], bundle, iv,
    )
    assert torch.equal(solo, in_batch), (
        f"short sample's per-position loss diverged when batched with a long "
        f"sample.\n  solo (T_max=3):  {solo.tolist()}\n"
        f"  batched (T_max=32): {in_batch.tolist()}"
    )
