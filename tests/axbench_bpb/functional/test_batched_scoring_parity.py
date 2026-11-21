"""GOLDEN tests: batch-1 vs batch-N parity for score_triple.

These are v1-ship gates: if they fail, the batching implementation diverges
from the single-sample kernel. Task 7.2's true-batched forward must preserve
this invariant.
"""
from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    score_sample,
    score_triple,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention
from tests.axbench_bpb.functional.test_score_sample_kernel import _make_inputs
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _score_each_solo(samples, bundle, iv):
    """Reference: call score_sample individually for each sample."""
    return [
        score_sample(
            inputs=s, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
        )
        for s in samples
    ]


def test_batch_1_equals_solo_decoder_only(tiny_decoder_only):
    """score_triple(device_batch_size=3, 3 samples) matches solo score_sample."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    samples = [
        _make_inputs(bundle.tokenizer, "hello", "world"),
        _make_inputs(bundle.tokenizer, "foo", "bar"),
        _make_inputs(bundle.tokenizer, "abc", "defghij"),
    ]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    solo_rows = _score_each_solo(samples, bundle, iv)
    batch_rows = score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=3,
    )
    assert len(solo_rows) == len(batch_rows)
    for s, b in zip(solo_rows, batch_rows, strict=True):
        assert s.sample_id == b.sample_id
        assert abs(s.bpb_intervened - b.bpb_intervened) < 1e-5


def test_batch_size_stability_varies_bs(tiny_decoder_only):
    """Identical per-sample BPB across device_batch_size ∈ {1, 2, 4, 8}."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    samples = [_make_inputs(bundle.tokenizer, f"p{i}", "out") for i in range(8)]
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
    for bs, got in results.items():
        for sid, bpb in ref.items():
            assert abs(got[sid] - bpb) < 1e-5, (
                f"batch_size={bs} sample {sid}: {got[sid]} vs {bpb}"
            )


def test_score_triple_empty_samples_raises_loudly(tiny_decoder_only):
    """Empty samples must raise loudly (Silent-mask audit fix; see design spec §6).

    Was: silent `return []` (a caller-bug-masking no-op).
    Now: raises AxBenchBpbError with a hint about caller-side filter bug.
    """
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    with pytest.raises(AxBenchBpbError, match="samples must be non-empty"):
        score_triple(
            samples=[], target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=4,
        )


def test_score_triple_device_batch_size_zero_raises(tiny_decoder_only):
    """Loud error when device_batch_size < 1."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    with pytest.raises(AxBenchBpbError, match="device_batch_size must be >= 1"):
        score_triple(
            samples=[_make_inputs(bundle.tokenizer, "a", "b")],
            target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
            device_batch_size=0,
        )


def test_score_triple_preserves_sample_order(tiny_decoder_only):
    """Returned rows are in the SAME order as input samples (regardless of batch size)."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    samples = [_make_inputs(bundle.tokenizer, f"pref{i}", f"out{i}") for i in range(5)]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    rows = score_triple(
        samples=samples, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=2,  # 2+2+1 chunking
    )
    assert [r.sample_id for r in rows] == [s.sample_id for s in samples]


def test_padding_isolation_short_sample_matches_solo(tiny_decoder_only):
    """GATE: a short sample batched alongside a long sample must produce
    identical BPB vs. solo score_sample. This is the padding-isolation
    invariant — when true batching lands in Phase 9, padded positions
    must not leak into the short sample's loss via cross-batch attention
    or dangling mask bits.

    Today the fallback path satisfies this trivially (per-sample forward);
    the test becomes load-bearing once ``_score_microbatch`` does real
    padded-batch forward."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    long_s = _make_inputs(bundle.tokenizer, "a" * 30, "b" * 5)
    short_s = _make_inputs(bundle.tokenizer, "x", "y")
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    solo = score_sample(
        inputs=short_s, target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
    )
    batched = score_triple(
        samples=[long_s, short_s], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=2,
    )
    got_short = next(r for r in batched if r.sample_id == short_s.sample_id)
    assert abs(solo.bpb_intervened - got_short.bpb_intervened) < 1e-5


def test_padding_isolation_multiple_short_samples_match_solo(tiny_decoder_only):
    """Stronger version: multiple short samples batched with one long
    sample; each one's BPB matches its solo score. Ensures the fallback
    (or future batched impl) doesn't silently reuse across samples."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    long_s = _make_inputs(bundle.tokenizer, "a" * 30, "b" * 5)
    shorts = [
        _make_inputs(bundle.tokenizer, f"p{i}", f"o{i}") for i in range(3)
    ]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])

    solos = {
        s.sample_id: score_sample(
            inputs=s, target_bundle=bundle, intervention=iv,
            device=torch.device("cpu"),
            coefficient_str="None", method_name="context_only_reference",
        ).bpb_intervened
        for s in shorts
    }

    batched = score_triple(
        samples=[long_s, *shorts], target_bundle=bundle, intervention=iv,
        device=torch.device("cpu"),
        coefficient_str="None", method_name="context_only_reference",
        device_batch_size=4,
    )
    for row in batched:
        if row.sample_id in solos:
            assert abs(solos[row.sample_id] - row.bpb_intervened) < 1e-5
