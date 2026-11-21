"""Determinism: same (sample, intervention, target_bundle) produces a
byte-identical bpb_intervened across multiple invocations of score_sample.

Locks against latent torch nondeterminism or RNG leakage. If a future
change introduces stochastic data augmentation or non-deterministic
backend ops in the scoring path, this test breaks.

Mirrors the fixture / inputs setup used in
``tests/axbench_bpb/functional/test_score_sample_kernel.py``.
"""
from __future__ import annotations

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
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _make_inputs(tok, prefix_text: str, output_text: str, concept_id: int = 0) -> ScoringInputs:
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
        prefix_context_segments=(),
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=byte_weights, truncated=False,
    )


def test_score_sample_is_bit_identical_across_invocations(tiny_decoder_only) -> None:
    """3x score_sample under the same inputs returns bit-identical bpb_intervened.

    Uses ``==``, not ``approx`` — any drift indicates a source of
    nondeterminism in the scoring path that needs investigation, not
    accommodation.
    """
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    inputs = _make_inputs(bundle.tokenizer, "abc", "def")
    intervention = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[])
    device = torch.device("cpu")

    results = [
        score_sample(
            inputs=inputs, target_bundle=bundle, intervention=intervention,
            device=device,
            coefficient_str="None", method_name="determinism_check",
        )
        for _ in range(3)
    ]

    assert results[0].bpb_intervened == results[1].bpb_intervened, (
        f"Determinism violated between call 0 and call 1: "
        f"{results[0].bpb_intervened!r} != {results[1].bpb_intervened!r}"
    )
    assert results[1].bpb_intervened == results[2].bpb_intervened, (
        f"Determinism violated between call 1 and call 2: "
        f"{results[1].bpb_intervened!r} != {results[2].bpb_intervened!r}"
    )
    # Lock the rest of the row too — n_output_tokens/n_output_bytes/truncated
    # should also be byte-identical (they're integer/bool, but contracted to
    # be invariant across calls).
    assert results[0].n_output_tokens == results[1].n_output_tokens == results[2].n_output_tokens
    assert results[0].n_output_bytes == results[1].n_output_bytes == results[2].n_output_bytes
    assert results[0].truncated == results[1].truncated == results[2].truncated


def test_score_sample_prompting_is_bit_identical(tiny_decoder_only) -> None:
    """Determinism holds under a non-trivial intervention (prompting)."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    inputs = _make_inputs(bundle.tokenizer, "abc", "def")
    intervention = Intervention.prompting(
        target_model="decoder_only",
        prompt_prefix_tokens=[ord("x"), ord("y"), ord("z")],
    )
    device = torch.device("cpu")

    results = [
        score_sample(
            inputs=inputs, target_bundle=bundle, intervention=intervention,
            device=device,
            coefficient_str="None", method_name="determinism_check_prompting",
        )
        for _ in range(3)
    ]

    assert results[0].bpb_intervened == results[1].bpb_intervened
    assert results[1].bpb_intervened == results[2].bpb_intervened
