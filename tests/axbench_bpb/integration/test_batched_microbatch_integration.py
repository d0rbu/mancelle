"""Integration tests — score_triple end-to-end through the public API."""
from __future__ import annotations

import random as _random
from pathlib import Path

import torch

from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb import intervention as iv_mod
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    score_sample,
    score_triple,
)
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import (
    Intervention,
    LoraParams,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

TINY_LORA_LAYER = 1
TINY_LORA_RANK = 4
TINY_LORA_ALPHA = 8.0


def _adapter_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "fixtures"
        / "tiny_decoder_only" / "lora_adapter_concept_0000.pt"
    )


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


def _mk_bundle(model, cfg, tok, target_name):
    return TargetModelBundle(
        model=model, tokenizer=tok,
        max_seq_len=cfg.sequence_len, target_name=target_name,
    )


def _direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g, dtype=torch.float32)


def test_score_triple_decoder_only_byte_equal_to_score_sample_loop(tiny_decoder_only):
    """I1: 16 samples; all 5 decoder_only kinds produce byte-equal BPB
    when scored via score_triple(B=4) vs per-sample score_sample loop."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}" + "a" * (i % 5), f"o{i}") for i in range(16)]

    interventions: list[tuple[str, Intervention, dict]] = [
        ("context_only", Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[]), {}),
        ("prompting", Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[ord("X"), ord("Y")],
        ), {}),
        ("diffmean", Intervention.from_direction(
            kind="diffmean", target_model="decoder_only",
            direction=_direction(cfg.n_embd, 1),
            direction_layer=1, coefficient=1.0,
        ), {}),
        ("linear_probe", Intervention.from_direction(
            kind="linear_probe", target_model="decoder_only",
            direction=_direction(cfg.n_embd, 2),
            direction_layer=1, coefficient=0.5,
        ), {}),
        ("lora", Intervention.lora(adapter_path=_adapter_path()),
         {"lora": LoraParams(
             cfg=LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA),
             layer=TINY_LORA_LAYER,
         )}),
    ]

    for name, iv, extra in interventions:
        # Solo loop
        solo = {
            s.sample_id: score_sample(
                inputs=s, target_bundle=bundle, intervention=iv,
                device=torch.device("cpu"),
                coefficient_str="X", method_name=name,
                **extra,
            ).bpb_intervened
            for s in samples
        }
        batched = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="X", method_name=name,
            device_batch_size=4,
            **extra,
        )
        for r in batched:
            assert r.bpb_intervened == solo[r.sample_id], (
                f"intervention {name}: sample {r.sample_id} solo={solo[r.sample_id]} "
                f"batched={r.bpb_intervened}"
            )


def test_score_triple_encdec_byte_equal_to_score_sample_loop_empty_pcs(tiny_encdec):
    """I2: encdec kinds × empty-feature context segments — score_triple == loop."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    samples = [_mk_inputs(tok, f"p{i}" + "a" * (i % 5), f"o{i}") for i in range(16)]

    interventions: list[tuple[str, Intervention]] = [
        ("context_only", Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])),
        ("ours_empty", Intervention.ours(feature_conditions={})),
        ("ours_populated", Intervention.ours(feature_conditions={42: 1, 17: 0})),
        ("prompting_encdec", Intervention.prompting(
            target_model="encdec", prompt_prefix_tokens=[ord("X"), ord("Y")],
        )),
    ]
    for name, iv in interventions:
        solo = {
            s.sample_id: score_sample(
                inputs=s, target_bundle=bundle, intervention=iv,
                device=torch.device("cpu"),
                coefficient_str="X", method_name=name,
            ).bpb_intervened
            for s in samples
        }
        batched = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="X", method_name=name,
            device_batch_size=4,
        )
        for r in batched:
            assert r.bpb_intervened == solo[r.sample_id]


def test_score_triple_encdec_byte_equal_to_score_sample_loop_nonempty_pcs(tiny_encdec):
    """I3: encdec kinds × samples with mixed context segment lengths — score_triple == loop."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    # pcs_1 is used with prefix "abcde" (5 chars + 1 BOS char = 6 chars total).
    # char_end=6 must cover all 6 decoded chars so no token falls outside [0, 6).
    pcs_1 = (ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=6),)
    # pcs_2 is used with prefix "abcdefg" (7 chars + 1 BOS char = 8 chars total).
    # Segment 0 covers [0, 4) and segment 1 covers [4, 8) — together they span all 8 chars.
    pcs_2 = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=4),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=4, char_end=8),
    )
    samples = []
    for i in range(12):
        prefix = "abcdefg"[: 3 + (i % 3) * 2]
        pcs = [None, pcs_1, pcs_2][i % 3]
        samples.append(_mk_inputs(tok, prefix, f"o{i}", pcs=pcs))

    interventions: list[tuple[str, Intervention]] = [
        ("context_only", Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])),
        ("ours_populated", Intervention.ours(feature_conditions={42: 1})),
    ]
    for name, iv in interventions:
        solo = {
            s.sample_id: score_sample(
                inputs=s, target_bundle=bundle, intervention=iv,
                device=torch.device("cpu"),
                coefficient_str="X", method_name=name,
            ).bpb_intervened
            for s in samples
        }
        batched = score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="X", method_name=name,
            device_batch_size=4,
        )
        for r in batched:
            assert r.bpb_intervened == solo[r.sample_id]


def test_model_state_dict_unchanged_after_microbatch(tiny_decoder_only):
    """I4: snapshot state_dict before/after score_triple; assert byte-equal."""
    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    interventions: list[tuple[str, Intervention, dict]] = [
        ("context_only", Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[]), {}),
        ("diffmean", Intervention.from_direction(
            kind="diffmean", target_model="decoder_only",
            direction=_direction(cfg.n_embd, 1),
            direction_layer=1, coefficient=1.0,
        ), {}),
        ("lora", Intervention.lora(adapter_path=_adapter_path()),
         {"lora": LoraParams(
             cfg=LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA),
             layer=TINY_LORA_LAYER,
         )}),
    ]
    for name, iv, extra in interventions:
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        score_triple(
            samples=samples, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="X", method_name=name,
            device_batch_size=4, **extra,
        )
        after = model.state_dict()
        for k, before_value in before.items():
            assert torch.equal(before_value, after[k]), (
                f"intervention {name}: state_dict[{k}] changed after score_triple"
            )


def test_intervention_install_count_amortized_diffmean(
    tiny_decoder_only, monkeypatch,
):
    """I5: register_residual_hook called exactly once per microbatch."""
    call_count = {"n": 0}
    real_register = iv_mod.register_residual_hook

    def counting_register(model, layer, direction, coefficient):
        call_count["n"] += 1
        return real_register(model, layer, direction, coefficient)

    monkeypatch.setattr(iv_mod, "register_residual_hook", counting_register)

    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=_direction(cfg.n_embd, 1),
        direction_layer=1, coefficient=1.0,
    )
    score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="1.0", method_name="diffmean",
        device_batch_size=4,
    )
    # 8 samples / 4 device_batch_size = 2 microbatches → 2 hook registrations.
    assert call_count["n"] == 2, (
        f"register_residual_hook called {call_count['n']} times; "
        f"expected 2 (once per microbatch)."
    )


def test_lora_attach_called_once_per_microbatch(tiny_decoder_only, monkeypatch):
    """I6: attach_lora_from_disk called once per microbatch."""
    call_count = {"n": 0}
    real_attach = iv_mod.attach_lora_from_disk

    def counting_attach(*args, **kwargs):
        call_count["n"] += 1
        return real_attach(*args, **kwargs)

    monkeypatch.setattr(iv_mod, "attach_lora_from_disk", counting_attach)

    model, cfg = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, cfg, tok, "decoder_only")
    samples = [_mk_inputs(tok, f"p{i}", f"o{i}") for i in range(8)]
    iv = Intervention.lora(adapter_path=_adapter_path())
    lora = LoraParams(
        cfg=LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA),
        layer=TINY_LORA_LAYER,
    )
    score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="lora",
        device_batch_size=4,
        lora=lora,
    )
    assert call_count["n"] == 2, (
        f"attach_lora_from_disk called {call_count['n']} times; expected 2"
    )


def test_score_triple_encdec_byte_equal_to_score_sample_loop_prompting_with_segments(tiny_encdec):
    """I3.5: prompting_encdec × samples with mixed PCS lengths — score_triple == loop.

    Validates that the new shifted-encoder-slot path in
    _build_encdec_seg_map_and_encoder produces byte-identical BPBs in both
    the batched (score_triple → _pad_encoder_inputs → _score_microbatch)
    and per-sample (score_sample) code paths."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    # Mix of context segment configurations as in the existing test.
    pcs_1 = (ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=6),)
    pcs_2 = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=4),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=4, char_end=8),
    )
    samples = []
    for i in range(12):
        prefix = "abcdefg"[: 3 + (i % 3) * 2]
        pcs = [None, pcs_1, pcs_2][i % 3]
        samples.append(_mk_inputs(tok, prefix, f"o{i}", pcs=pcs))

    intervention = Intervention.prompting(
        target_model="encdec",
        prompt_prefix_tokens=[ord("X"), ord("Y")],
    )
    solo = {
        s.sample_id: score_sample(
            inputs=s, target_bundle=bundle, intervention=intervention,
            device=torch.device("cpu"),
            coefficient_str="X", method_name="prompting_encdec",
        ).bpb_intervened
        for s in samples
    }
    batched = score_triple(
        samples=samples, target_bundle=bundle, intervention=intervention,
        device=torch.device("cpu"),
        coefficient_str="X", method_name="prompting_encdec",
        device_batch_size=4,
    )
    for r in batched:
        assert r.bpb_intervened == solo[r.sample_id], (
            f"score_triple/score_sample diverged for sample_id={r.sample_id}: "
            f"batched={r.bpb_intervened} vs solo={solo[r.sample_id]}"
        )


def test_score_triple_preserves_input_order_with_mixed_pcs(tiny_encdec):
    """I8: shuffled mixed-PCS samples → output rows in input order."""
    model, dec_cfg, _e, _x = tiny_encdec
    tok = SimpleTokenizer()
    bundle = _mk_bundle(model, dec_cfg, tok, "encdec")
    # "abcde" has 5 chars + 1 BOS char = 6 chars total.
    # Segments must together span [0, 6) so all 6 tokens are assigned.
    pcs_2 = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(1,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(20,), buckets=(2,), char_start=3, char_end=6),
    )
    samples = [_mk_inputs(tok, f"abc{i}", f"o{i}") for i in range(5)]
    samples += [_mk_inputs(tok, "abcde", f"o5+{j}", pcs=pcs_2) for j in range(5)]
    rng = _random.Random(42)
    rng.shuffle(samples)

    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    rows = score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=3,
    )
    assert [r.sample_id for r in rows] == [s.sample_id for s in samples]
