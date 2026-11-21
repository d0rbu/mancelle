"""Batched teacher-forced gradient == the single-prompt path (per sample).

``teacher_forced_bpb_gradient_batch`` pads N decoder samples into one forward plus
embedding-only VJP so
the universal GCG step's per-pair gradients run at a usable batch size instead of N
serialized single-prompt gradient evaluations. On ``_ToyDecoder`` -- an EXACT,
per-position
independent float32 model -- each batched per-sample ``BpbGradientResult`` must be
BYTE-IDENTICAL to the single-prompt ``teacher_forced_bpb_gradient``: same loss, same
``token_gradient`` (``t.equal``), same ``embedding_weight``. The samples are
independent (right-padding + causal/per-position structure), so a wrong per-sample
VJP extraction, a wrong padding offset, a leaked cross-sample gradient, or a
wrong scalar reduction would change a sample's gradient and fail loudly here.

This is the crown jewel for the gradient batching: exact equivalence to the trusted
single-prompt path on a model with NO bf16 rounding. The real-model (tolerance)
counterpart lives in ``test_gcg_exact_speedup_integration.py``.
"""

from __future__ import annotations

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.surrogate as surrogate
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.surrogate import (
    BpbGradientResult,
    bpb_surrogate_embedding_weight,
    teacher_forced_bpb_gradient,
    teacher_forced_bpb_gradient_batch,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _bundle,
    _encdec_bundle,
    _generic_decoder_bundle,
    _TinyTokenizer,
    _token_bytes,
    _ToyBadEmbeddingOutput,
    _ToyBadValueDim,
    _ToyDecoder,
    _ToyEncDec,
    _ToyFiniteLossNonFiniteGrad,
    _ToyNoValueUse,
    _ToyNoWteUse,
)


def _encdec_segments(prompt: list[int]) -> tuple[ContextFeatureSegment, ...]:
    decoded_len = len(_TinyTokenizer().decode(prompt))
    return (
        ContextFeatureSegment(
            feature_ids=(7,), buckets=(2,), char_start=0, char_end=decoded_len
        ),
    )

# (prompt, output) pairs of MIXED length so padding offsets matter; ids < vocab=32.
_SAMPLES: list[tuple[list[int], list[int]]] = [
    ([3, 7, 11], [13, 1]),
    ([5, 9], [4, 8, 6]),
    ([2, 6, 10, 14], [12]),
    ([21], [22, 23]),
]


def _single(
    bundle: TargetModelBundle,
    prompt: list[int],
    output: list[int],
    *,
    embedding_weight: t.Tensor | None = None,
) -> BpbGradientResult:
    return teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=prompt,
        output_tokens=output,
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        embedding_weight=embedding_weight,
    )


def _batched(
    bundle: TargetModelBundle,
    samples: list[tuple[list[int], list[int]]],
    *,
    device_batch_size: int = 8,
    embedding_weight: t.Tensor | None = None,
) -> list[BpbGradientResult]:
    return teacher_forced_bpb_gradient_batch(
        target_bundle=bundle,
        prompts=[p for p, _ in samples],
        outputs=[o for _, o in samples],
        prefix_context_segments_list=[() for _ in samples],
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        device_batch_size=PositiveInt.parse(device_batch_size),
        embedding_weight=embedding_weight,
    )


def _assert_result_equal(got: BpbGradientResult, want: BpbGradientResult) -> None:
    assert got.loss == want.loss
    assert t.equal(got.token_gradient, want.token_gradient)
    assert t.equal(got.embedding_weight, want.embedding_weight)


def test_batched_gradient_matches_single_per_sample() -> None:
    # Value embeds present so the gradient has a non-trivial concatenated width.
    bundle = _bundle(_ToyDecoder(value_dims=(2,)))
    single = [_single(bundle, p, o) for p, o in _SAMPLES]
    batched = _batched(bundle, _SAMPLES)
    assert len(batched) == len(single)
    for got, want in zip(batched, single, strict=True):
        _assert_result_equal(got, want)
    # Non-trivial: gradient actually has the wte(3) + value(2) width and is non-zero.
    assert batched[0].token_gradient.shape[1] == 5
    assert batched[0].token_gradient.abs().sum() > 0


def test_batched_gradient_at_batch_one_is_bit_identical() -> None:
    # B=1 has no padding, so the batched path must reproduce the single path EXACTLY.
    bundle = _bundle(_ToyDecoder(value_dims=(2,)))
    prompt, output = _SAMPLES[1]
    [got] = _batched(bundle, [(prompt, output)])
    _assert_result_equal(got, _single(bundle, prompt, output))


def test_batched_gradient_order_independent_of_neighbours() -> None:
    # A sample's gradient must not depend on which others share its batch.
    bundle = _bundle(_ToyDecoder(value_dims=(2,)))
    forward = _batched(bundle, _SAMPLES)
    reverse = _batched(bundle, list(reversed(_SAMPLES)))
    for f, r in zip(forward, reversed(reverse), strict=True):
        _assert_result_equal(f, r)


def test_batched_gradient_chunking_does_not_change_results() -> None:
    # device_batch_size only bounds the live graph; per-sample gradients are
    # independent of how the samples are chunked across forwards.
    bundle = _bundle(_ToyDecoder(value_dims=(2,)))
    one_forward = _batched(bundle, _SAMPLES, device_batch_size=100)
    per_sample = _batched(bundle, _SAMPLES, device_batch_size=1)
    for a, b in zip(one_forward, per_sample, strict=True):
        _assert_result_equal(a, b)


def test_batched_gradient_cached_embedding_matches_rebuilt() -> None:
    # OPT#1: passing the cached static embedding must not change the gradient; the
    # returned embedding_weight is the passed tensor verbatim.
    bundle = _bundle(_ToyDecoder(value_dims=(2,)))
    cached = bpb_surrogate_embedding_weight(bundle.model)
    rebuilt = _batched(bundle, _SAMPLES)
    with_cache = _batched(bundle, _SAMPLES, embedding_weight=cached)
    for r, c in zip(rebuilt, with_cache, strict=True):
        assert r.loss == c.loss
        assert t.equal(r.token_gradient, c.token_gradient)
        assert t.equal(r.embedding_weight, c.embedding_weight)
    assert all(c.embedding_weight is cached for c in with_cache)


def test_batched_gradient_handles_left_truncation_per_sample() -> None:
    # A sample whose prompt is left-truncated (prompt+output > max_seq_len) must
    # zero the dropped prompt positions, exactly like the single path -- batched
    # alongside an untruncated sample to prove the offset is per-sample.
    bundle = _bundle(_ToyDecoder(value_dims=(2,)), max_seq_len=4)
    samples = [([2, 6, 10, 14], [12]), ([21], [22, 23])]
    batched = _batched(bundle, samples)
    for (prompt, output), got in zip(samples, batched, strict=True):
        _assert_result_equal(got, _single(bundle, prompt, output))
    # The truncated sample really did drop leading positions (zeroed gradient rows).
    assert batched[0].token_gradient[:2].abs().sum() == 0


def test_batched_gradient_empty_returns_empty() -> None:
    assert _batched(_bundle(_ToyDecoder()), []) == []


def test_gradient_path_never_constructs_a_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_cache(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("gradient path constructed a KV cache")

    monkeypatch.setattr(surrogate, "KVCache", _forbidden_cache)
    results = _batched(
        _bundle(_ToyDecoder(value_dims=(2,))),
        _SAMPLES[:2],
        device_batch_size=2,
    )
    assert len(results) == 2


def test_batched_gradient_rejects_length_mismatch() -> None:
    with pytest.raises(Exception, match="length mismatch"):
        teacher_forced_bpb_gradient_batch(
            target_bundle=_bundle(_ToyDecoder()),
            prompts=[[1, 2]],
            outputs=[[3], [4]],
            prefix_context_segments_list=[()],
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


def test_batched_gradient_encdec_loops_single_path() -> None:
    # encdec is not batched (encoder tensors are per-sample); the batch function
    # must loop the single primitive and return per-sample-identical results. The
    # encdec teacher-forcing path requires non-empty context segments + an `ours`
    # intervention (mirrors test_surrogate_parity's encdec coverage).
    bundle = _encdec_bundle(_ToyEncDec())
    intervention = Intervention.ours(feature_conditions={11: 3})
    samples = [([1, 2, 3], [4, 5]), ([1, 2], [6, 7, 8])]
    segments = [_encdec_segments(p) for p, _ in samples]
    batched = teacher_forced_bpb_gradient_batch(
        target_bundle=bundle,
        prompts=[p for p, _ in samples],
        outputs=[o for _, o in samples],
        prefix_context_segments_list=segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    for (prompt, output), seg, got in zip(samples, segments, batched, strict=True):
        _assert_result_equal(
            got,
            teacher_forced_bpb_gradient(
                target_bundle=bundle,
                prompt_tokens=prompt,
                output_tokens=output,
                prefix_context_segments=seg,
                output_token_bytes=_token_bytes(),
                intervention=intervention,
                device=t.device("cpu"),
            ),
        )


# --- ERROR/EDGE branches reached only through the *_batch decoder microbatch ---
# The crown-jewel tests above prove exact per-sample equivalence on a benign model.
# These route bad-model toys / pathological gradients through the batched VJP so
# the error and clamp branches in _decoder_microbatch_gradient are covered, not only
# the single-prompt counterparts in test_surrogate_parity. The batched non-finite
# clamp also emits a DISTINCT warning ("teacher_forced_bpb_gradient_batch: ...").


def _batched_raw(
    bundle: TargetModelBundle,
    samples: list[tuple[list[int], list[int]]],
) -> list[BpbGradientResult]:
    return teacher_forced_bpb_gradient_batch(
        target_bundle=bundle,
        prompts=[p for p, _ in samples],
        outputs=[o for _, o in samples],
        prefix_context_segments_list=[() for _ in samples],
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_ToyNoWteUse(), "failed to capture gradient"),
        (_ToyNoValueUse(value_dims=(2,)), "value_embeds"),
        (_ToyBadEmbeddingOutput(), "non-Tensor"),
        (_ToyBadValueDim(), "gradient/embedding dimension mismatch"),
    ],
)
def test_batched_gradient_rejects_invalid_model_paths(
    model: t.nn.Module,
    match: str,
) -> None:
    # Route each bad-model toy through the BATCHED gradient (1- or 2-sample batch);
    # the same guards in _decoder_microbatch_gradient must raise as on the single
    # path. Two samples so the per-sample loop (not just B=1) is exercised.
    with pytest.raises(AxBenchBpbAdvError, match=match):
        _batched_raw(
            _generic_decoder_bundle(model),
            [([0, 1, 2, 3], [4, 5]), ([0, 1], [2, 3])],
        )


def test_batched_gradient_clamps_non_finite_and_warns(caplog) -> None:
    # _ToyFiniteLossNonFiniteGrad has a finite loss but a non-finite wte gradient.
    # The batched path must clamp it to finite AND emit the batch-specific warning
    # ("teacher_forced_bpb_gradient_batch: clamped ...") -- genuinely new coverage,
    # the single path's warning text differs.
    with caplog.at_level("WARNING"):
        results = _batched_raw(
            _generic_decoder_bundle(_ToyFiniteLossNonFiniteGrad()),
            [([0, 1, 2, 3], [4, 5]), ([0, 1], [2, 3])],
        )

    for result in results:
        assert t.isfinite(result.token_gradient).all()
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "teacher_forced_bpb_gradient_batch" in m and "non-finite" in m and "gradient" in m
        for m in messages
    ), messages


def test_batched_gradient_with_zero_value_modules_matches_single() -> None:
    # No value_embeds -> the batched gradient assembly skips the value-module loop
    # entirely (wte-only width). Pin per-sample byte-identity to the single path and
    # that the gradient width is exactly the wte d_model (3).
    bundle = _bundle(_ToyDecoder())
    single = [_single(bundle, p, o) for p, o in _SAMPLES]
    batched = _batched(bundle, _SAMPLES)
    assert len(batched) == len(single)
    for got, want in zip(batched, single, strict=True):
        _assert_result_equal(got, want)
    assert all(r.token_gradient.shape[1] == 3 for r in batched)
