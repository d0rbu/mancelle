"""score_sample wraps internal errors raised by accumulate_bpb in a single
AxBenchBpbError with sample_id, method_name, and coefficient_str surfaced
in the message and ``__cause__`` chained back to the original exception.

Covers teacher_force.py:193-199 — the try/except around the call to
``accumulate_bpb``. Without this wrapper, a shape mismatch / zero-byte /
non-finite-loss error would surface as a bare ``AxBenchBpbError`` from
deep inside ``accumulate_bpb`` with no breadcrumbs to which sample or
method triggered it. The wrapper is operator-diagnostic: it folds the
identity of the failing unit into the message.

Approach: monkeypatch ``accumulate_bpb`` to raise a synthetic
``AxBenchBpbError``. The wrapper at lines 193-199 catches it, re-raises
with augmented context, chained via ``raise ... from exc``.
"""
from __future__ import annotations

import math

import pytest
import torch

from steering.eval.axbench_bpb import teacher_force as teacher_force_module
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
    AxBenchBpbError,
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _make_inputs(tok, prefix_text, output_text, *, sample_id_override=None):
    """Build a minimal ScoringInputs (mirrors test_score_sample_kernel helper)."""
    prefix_tokens = encode_prefix(prefix_text, tok)
    output_tokens = encode_output(output_text, tok)
    byte_weights = torch.ones(len(output_tokens), dtype=torch.int64)
    sid = sample_id_override or make_sample_id(
        concept_id=0, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_text, output_text=output_text,
    )
    return ScoringInputs(
        sample_id=sid,
        concept_id=0, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_text, prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix_tokens, output_tokens=output_tokens,
        prefix_context_segments=(),
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=byte_weights, truncated=False,
    )


def test_score_sample_wraps_accumulate_bpb_error_with_context(
    tiny_decoder_only, monkeypatch,
):
    """A synthetic ``AxBenchBpbError`` raised by ``accumulate_bpb`` is wrapped
    with sample_id, method_name, and coefficient_str in the message; the
    original exception chains via ``__cause__``."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    inputs = _make_inputs(
        bundle.tokenizer, "abc", "def",
        sample_id_override="sample-xyz-42",
    )

    inner_message = "synthetic-inner-bpb-failure"
    injected = AxBenchBpbError(inner_message)

    def _raising_accumulate(*args, **kwargs):
        raise injected

    monkeypatch.setattr(teacher_force_module, "accumulate_bpb", _raising_accumulate)

    with pytest.raises(AxBenchBpbError) as exc_info:
        score_sample(
            inputs=inputs, target_bundle=bundle,
            intervention=Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[]),
            device=torch.device("cpu"),
            coefficient_str="2.5", method_name="diffmean_decoder_only",
        )

    msg = str(exc_info.value)
    # Operator diagnostics: sample_id, method_name, coefficient_str all present.
    assert "sample-xyz-42" in msg, f"sample_id missing from wrapper message: {msg!r}"
    assert "diffmean_decoder_only" in msg, f"method_name missing from wrapper message: {msg!r}"
    assert "2.5" in msg, f"coefficient_str missing from wrapper message: {msg!r}"
    # Inner error message folded in.
    assert inner_message in msg, f"inner error message lost: {msg!r}"
    # __cause__ chains the original.
    assert exc_info.value.__cause__ is injected, (
        "wrapper must chain the original exception via raise ... from exc"
    )


def test_score_sample_wrapper_preserves_axbench_bpb_type(
    tiny_decoder_only, monkeypatch,
):
    """The wrapper raises ``AxBenchBpbError`` (the package's single user-visible
    exception type), not the bare original — even when the original is also
    an ``AxBenchBpbError``. Callers can rely on a single except clause."""
    model, cfg = tiny_decoder_only
    bundle = TargetModelBundle(
        model=model, tokenizer=SimpleTokenizer(),
        max_seq_len=cfg.sequence_len, target_name="decoder_only",
    )
    inputs = _make_inputs(bundle.tokenizer, "abc", "def")

    injected = AxBenchBpbError("inner failure")

    def _raising_accumulate(*args, **kwargs):
        raise injected

    monkeypatch.setattr(teacher_force_module, "accumulate_bpb", _raising_accumulate)

    with pytest.raises(AxBenchBpbError) as exc_info:
        score_sample(
            inputs=inputs, target_bundle=bundle,
            intervention=Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=[]),
            device=torch.device("cpu"),
            coefficient_str="None", method_name="any_method",
        )
    # The raised exception is a NEW AxBenchBpbError, not the original.
    assert exc_info.value is not injected, (
        "wrapper must raise a new AxBenchBpbError, not re-raise the original"
    )
    assert isinstance(exc_info.value, AxBenchBpbError)


def test_score_sample_succeeds_when_accumulate_bpb_does_not_raise(
    tiny_decoder_only,
):
    """Sanity check: the try/except is transparent on the happy path —
    a successful ``accumulate_bpb`` does not get wrapped or otherwise
    altered. Guards against a future refactor that accidentally swallows
    valid returns."""
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
        coefficient_str="None", method_name="happy_path",
    )
    assert math.isfinite(row.bpb_intervened)
    assert row.method_name == "happy_path"
