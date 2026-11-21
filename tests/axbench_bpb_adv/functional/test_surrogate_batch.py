"""Batched teacher-forced BPB loss == the single-prompt path (per sample).

The test-only full-sequence reference pads N decoder-only samples into one forward so
the GCG candidate ranking runs on the GPU at a usable batch size. This pins that
its per-sample BPB equals the single-prompt ``teacher_forced_bpb_loss`` (exact at
batch size 1; per-sample equal on a deterministic CPU model with mixed
lengths/byte-weights). The toy model's per-token loss is a function of the INPUT
token, so a wrong per-sample mask, a wrong padding offset, or mis-pairing samples
across the batch would change a sample's BPB and fail loudly here.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.eval.axbench_bpb_adv.surrogate as surrogate
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.surrogate import (
    teacher_forced_bpb_gradient_batch,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.axbench_bpb_adv.full_sequence_reference import (
    teacher_forced_bpb_loss,
    teacher_forced_bpb_loss_full_sequence_reference,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _bundle as _parity_bundle,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _encdec_bundle as _parity_encdec_bundle,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _TinyTokenizer,
    _ToyDecoder,
    _ToyEncDec,
    _ToyNonFinite,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _token_bytes as _parity_token_bytes,
)


class _TokenLossDecoder(t.nn.Module):
    """Per-token loss is a deterministic function of the INPUT token id.

    Decoder-only loss-only path only calls ``model(idx, targets, loss_reduction)``
    (no embedding hooks), so the toy needs no transformer internals. Returning a
    token-dependent ``(B, T)`` tensor makes per-sample extraction bugs observable.
    """

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        del targets, loss_reduction
        return (idx.to(t.float32) % 5.0) + 0.5


class _TinyTok:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) != 0)


def _bundle() -> TargetModelBundle:
    return TargetModelBundle(
        model=_TokenLossDecoder(),
        tokenizer=_TinyTok(),
        max_seq_len=64,
        target_name="decoder_only",
    )


def _token_bytes() -> t.Tensor:
    # Distinct positive byte counts so per-sample byte-weighting differs by token.
    table = t.arange(256, dtype=t.int64) % 4 + 1
    return table


_SAMPLES: list[tuple[list[int], list[int]]] = [
    ([3, 7, 11], [13]),
    ([5, 9], [4, 8, 6]),
    ([2, 6, 10, 14], [12]),
    ([21], [22, 23]),
]


def _single(bundle: TargetModelBundle, prompt: list[int], output: list[int]) -> float:
    return teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=prompt,
        output_tokens=output,
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )


def _batched(
    bundle: TargetModelBundle,
    samples: list[tuple[list[int], list[int]]],
    *,
    device_batch_size: int = 32,
) -> list[float]:
    return teacher_forced_bpb_loss_full_sequence_reference(
        target_bundle=bundle,
        prompts=[p for p, _ in samples],
        outputs=[o for _, o in samples],
        prefix_context_segments_list=[() for _ in samples],
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        device_batch_size=PositiveInt.parse(device_batch_size),
    )


def test_batched_loss_matches_single_prompt_per_sample() -> None:
    bundle = _bundle()
    single = [_single(bundle, p, o) for p, o in _SAMPLES]
    batched = _batched(bundle, _SAMPLES)
    assert len(batched) == len(single)
    for got, want in zip(batched, single, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def test_batched_loss_at_batch_one_is_bit_identical() -> None:
    # At B=1 there is no padding, so the batched path must be EXACTLY the single
    # path (not just approx).
    bundle = _bundle()
    prompt, output = _SAMPLES[1]
    assert _batched(bundle, [(prompt, output)]) == [_single(bundle, prompt, output)]


def test_batched_loss_order_independent_of_neighbours() -> None:
    # A sample's BPB must not depend on which other samples share its batch (no
    # cross-sample leakage): reversing the batch reverses the results, unchanged.
    bundle = _bundle()
    forward = _batched(bundle, _SAMPLES)
    reverse = _batched(bundle, list(reversed(_SAMPLES)))
    assert forward == list(reversed(reverse))


def test_batched_loss_empty_returns_empty() -> None:
    assert _batched(_bundle(), []) == []


def test_batched_loss_chunking_does_not_change_results() -> None:
    # device_batch_size only bounds the live microbatch; per-sample BPB is
    # independent of how the samples are chunked across forwards.
    bundle = _bundle()
    one_forward = _batched(bundle, _SAMPLES, device_batch_size=100)
    per_sample = _batched(bundle, _SAMPLES, device_batch_size=1)
    assert one_forward == pytest.approx(per_sample, abs=1e-6)


_INVALID_POSITIVE_INT = st.one_of(
    st.integers(max_value=0),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)


@settings(max_examples=500, derandomize=True)
@given(
    function_name=st.sampled_from(("loss", "gradient")),
    value=_INVALID_POSITIVE_INT,
)
def test_batched_surrogates_reject_every_invalid_device_batch_size(
    function_name: str,
    value: Any,
) -> None:
    function = (
        teacher_forced_bpb_loss_full_sequence_reference
        if function_name == "loss"
        else teacher_forced_bpb_gradient_batch
    )
    with pytest.raises(BeartypeCallHintParamViolation, match="device_batch_size"):
        function(
            target_bundle=_bundle(),
            prompts=[],
            outputs=[],
            prefix_context_segments_list=[],
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
            device_batch_size=value,
        )


# --- ERROR/EDGE branches reached only through the *_batch decoder microbatch ---
# The tests above pin the batched loss against the single path on a benign model;
# these route inputs that trip the batched reduction's guards (a DIFFERENT message
# from the single path, "(batched)") so the batched microbatch branches are
# covered, not just the single-prompt ones in test_surrogate_parity.


def _encdec_segments(prompt: list[int]) -> tuple[ContextFeatureSegment, ...]:
    decoded_len = len(_TinyTokenizer().decode(prompt))
    return (
        ContextFeatureSegment(
            feature_ids=(7,), buckets=(2,), char_start=0, char_end=decoded_len
        ),
    )


def test_batched_loss_raises_on_non_finite_in_mask_loss() -> None:
    # _ToyNonFinite makes every per-token loss non-finite; routed through the
    # decoder microbatch this must hit the batched finite-loss guard
    # (_microbatch_bpb_tensors), distinct from the single path's "non-finite".
    with pytest.raises(AxBenchBpbAdvError, match=r"non-finite.*\(batched\)"):
        teacher_forced_bpb_loss_full_sequence_reference(
            target_bundle=_parity_bundle(_ToyNonFinite()),
            prompts=[[0, 1, 2, 3]],
            outputs=[[4, 5]],
            prefix_context_segments_list=[()],
            output_token_bytes=_parity_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


def test_batched_loss_raises_on_byte_weight_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the per-sample byte weights to be a wrong length so the batched
    # microbatch reduction's length-mismatch guard fires ("(batched)" variant).
    monkeypatch.setattr(
        surrogate,
        "_output_byte_weights",
        lambda **_kwargs: t.ones(1, dtype=t.float32),
    )
    with pytest.raises(
        AxBenchBpbAdvError, match=r"byte-weight/output loss length mismatch \(batched\)"
    ):
        teacher_forced_bpb_loss_full_sequence_reference(
            target_bundle=_parity_bundle(_ToyDecoder()),
            prompts=[[0, 1, 2, 3]],
            outputs=[[4, 5]],
            prefix_context_segments_list=[()],
            output_token_bytes=_parity_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


def test_batched_loss_encdec_loops_single_path() -> None:
    # encdec is not batched (encoder tensors are per-sample); the batch function
    # must loop the single primitive. Drives the encdec branch of
    # The full-sequence reference with non-empty context segments + an `ours`
    # intervention and asserts per-sample equality to the single loss.
    bundle = _parity_encdec_bundle(_ToyEncDec())
    intervention = Intervention.ours(feature_conditions={11: 3})
    samples = [([1, 2, 3], [4, 5]), ([1, 2], [6, 7, 8])]
    segments = [_encdec_segments(p) for p, _ in samples]
    batched = teacher_forced_bpb_loss_full_sequence_reference(
        target_bundle=bundle,
        prompts=[p for p, _ in samples],
        outputs=[o for _, o in samples],
        prefix_context_segments_list=segments,
        output_token_bytes=_parity_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    for (prompt, output), seg, got in zip(samples, segments, batched, strict=True):
        assert got == teacher_forced_bpb_loss(
            target_bundle=bundle,
            prompt_tokens=prompt,
            output_tokens=output,
            prefix_context_segments=seg,
            output_token_bytes=_parity_token_bytes(),
            intervention=intervention,
            device=t.device("cpu"),
        )


def test_batched_loss_rejects_length_mismatch() -> None:
    # prompts/outputs length mismatch must fail loudly at the top-level guard.
    with pytest.raises(Exception, match="length mismatch"):
        teacher_forced_bpb_loss_full_sequence_reference(
            target_bundle=_bundle(),
            prompts=[[1, 2]],
            outputs=[[3], [4]],
            prefix_context_segments_list=[()],
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )
