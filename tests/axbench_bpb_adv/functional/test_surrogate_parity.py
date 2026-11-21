from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.surrogate as surrogate
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle, score_sample
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.optimize import objective_gradient, objective_value
from steering.eval.axbench_bpb_adv.surrogate import (
    _concatenate_embedding_parts,
    _prompt_gradient_from_full_gradient,
    bpb_surrogate_embedding_weight,
    teacher_forced_bpb_gradient,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.axbench_bpb_adv.full_sequence_reference import teacher_forced_bpb_loss


class _TinyTokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


class _Transformer(t.nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.wte = t.nn.Embedding(vocab_size, d_model)


class _InnerModelWrapper(t.nn.Module):
    def __init__(self, model: t.nn.Module) -> None:
        super().__init__()
        self.model = model


class _ToyDecoder(t.nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 32,
        d_model: int = 3,
        value_dims: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.transformer = _Transformer(vocab_size, d_model)
        self.value_embeds = t.nn.ModuleDict(
            {
                str(index): t.nn.Embedding(vocab_size, dim)
                for index, dim in enumerate(value_dims)
            }
        )
        with t.no_grad():
            values = t.arange(vocab_size * d_model, dtype=t.float32)
            self.transformer.wte.weight.copy_(values.view(vocab_size, d_model) / 100.0)
            for index, module in enumerate(self.value_embeds.values()):
                assert isinstance(module, t.nn.Embedding)
                n_values = vocab_size * module.embedding_dim
                value_weights = t.arange(n_values, dtype=t.float32)
                module.weight.copy_(
                    (value_weights.view(vocab_size, module.embedding_dim) + index) / 200.0
                )

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        del loss_reduction
        if targets is None:
            raise AssertionError("teacher-forced tests always pass targets")
        embeddings = self._decoder_signal(idx)
        target_signal = targets.to(t.float32).unsqueeze(-1) / 10.0
        return (embeddings - target_signal).square().sum(dim=-1)

    def _decoder_signal(self, idx: t.Tensor) -> t.Tensor:
        parts = [self.transformer.wte(idx)]
        parts.extend(module(idx) for module in self.value_embeds.values())
        return t.cat(parts, dim=-1)


class _ToyEncDec(_ToyDecoder):
    # `_build_teacher_forced_forward_inputs` reads `target_bundle.model.feature_vocab`
    # (via `feature_vocab_allowed_ids`) to V-restrict context-cache features.
    # `None` == uncompacted (legacy no-op), matching every real, unmodified
    # `EncoderDecoderGPT` this toy stands in for. `_CompactedToyEncDec` below
    # overrides this to simulate a compacted checkpoint.
    feature_vocab: Any = None

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        enc_feature_ids: t.Tensor | None = None,
        enc_magnitudes: t.Tensor | None = None,
        enc_mask: t.Tensor | None = None,
        seg_map: t.Tensor | None = None,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        del loss_reduction
        if targets is None:
            raise AssertionError("teacher-forced tests always pass targets")
        if (
            enc_feature_ids is None
            or enc_magnitudes is None
            or enc_mask is None
            or seg_map is None
        ):
            raise AssertionError("encdec tests must pass encoder tensors and seg_map")
        embeddings = self._decoder_signal(idx)
        slot_signal = (
            (enc_feature_ids.to(t.float32) + enc_magnitudes.to(t.float32))
            * enc_mask.to(t.float32)
        ).sum(dim=-1)
        routed = slot_signal.gather(1, seg_map).unsqueeze(-1) / 100.0
        target_signal = targets.to(t.float32).unsqueeze(-1) / 10.0
        return (embeddings + routed - target_signal).square().sum(dim=-1)


class _CompactedToyEncDec(_ToyEncDec):
    """`_ToyEncDec` with a non-``None`` ``feature_vocab``, simulating a
    compacted checkpoint. `feature_vocab_allowed_ids` only branches on
    `model.feature_vocab is None`, so any non-``None`` sentinel plus
    `.encoder.feature_vocab_ids` is sufficient to stand in for a real
    compacted `EncoderDecoderGPT`."""

    def __init__(self, allowed: t.Tensor) -> None:
        super().__init__()
        self.feature_vocab = object()
        self.encoder = SimpleNamespace(feature_vocab_ids=allowed)


class _ToyNoValueUse(_ToyDecoder):
    def _decoder_signal(self, idx: t.Tensor) -> t.Tensor:
        return self.transformer.wte(idx)


class _ToyNoWteUse(t.nn.Module):
    def __init__(self, *, vocab_size: int = 32, d_model: int = 3) -> None:
        super().__init__()
        self.transformer = _Transformer(vocab_size, d_model)
        self.bias = t.nn.Parameter(t.ones(()))

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        del targets, loss_reduction
        return t.ones_like(idx, dtype=t.float32) * self.bias


class _ToyBadShape(_ToyDecoder):
    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        loss = super().forward(idx, targets, loss_reduction=loss_reduction)
        return loss[:, :-1]


class _ToyNonFinite(_ToyDecoder):
    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        loss = super().forward(idx, targets, loss_reduction=loss_reduction)
        return loss * t.inf


class _ToyFiniteLossNonFiniteGrad(_ToyDecoder):
    """Forward whose loss is finite-valued but whose gradient w.r.t. the captured
    ``transformer.wte`` output is non-finite.

    ``spike`` is identically 0 in value (so the loss is unperturbed and passes the
    finite-loss guard at surrogate.py:307), but its derivative w.r.t. the embedding
    is nan (``d/dx sqrt(|x|)`` at ``x == 0``), so the captured ``wte`` gradient is
    non-finite. This reaches the gradient assembled at surrogate.py:354 unchecked.
    """

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        base = super().forward(idx, targets, loss_reduction=loss_reduction)
        emb = self.transformer.wte(idx)
        spike = t.sqrt((emb.sum(-1) - emb.sum(-1)).abs())
        return base + spike


class _BadEmbedding(t.nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 3) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.zeros(vocab_size, d_model))

    def forward(self, idx: t.Tensor) -> tuple[t.Tensor]:
        return (t.nn.functional.embedding(idx, self.weight),)


class _ToyBadEmbeddingOutput(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Transformer(vocab_size=32, d_model=3)
        transformer: Any = self.transformer
        transformer.wte = _BadEmbedding()

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "none",
    ) -> t.Tensor:
        del targets, loss_reduction
        embedded = self.transformer.wte(idx)
        return t.zeros_like(idx, dtype=t.float32) + embedded[0].sum() * 0.0


class _BadValueEmbedding(t.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.zeros(vocab_size, 2))

    def forward(self, idx: t.Tensor) -> t.Tensor:
        padded = t.cat([self.weight, t.zeros(self.weight.shape[0], 1)], dim=1)
        return t.nn.functional.embedding(idx, padded)


class _ToyBadValueDim(_ToyDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.value_embeds = t.nn.ModuleDict({"0": _BadValueEmbedding()})

    def _decoder_signal(self, idx: t.Tensor) -> t.Tensor:
        return t.cat([self.transformer.wte(idx), self.value_embeds["0"](idx)], dim=-1)


def _bundle(model: _ToyDecoder, *, max_seq_len: int = 16) -> TargetModelBundle:
    return TargetModelBundle(
        model=model,
        tokenizer=_TinyTokenizer(),
        max_seq_len=max_seq_len,
        target_name="decoder_only",
    )


def _encdec_bundle(model: _ToyEncDec, *, max_seq_len: int = 16) -> TargetModelBundle:
    return TargetModelBundle(
        model=model,
        tokenizer=_TinyTokenizer(),
        max_seq_len=max_seq_len,
        target_name="encdec",
    )


def _generic_decoder_bundle(model: t.nn.Module, *, max_seq_len: int = 16) -> TargetModelBundle:
    return TargetModelBundle(
        model=model,
        tokenizer=_TinyTokenizer(),
        max_seq_len=max_seq_len,
        target_name="decoder_only",
    )


def _inputs(*, prefix_tokens: list[int], output_tokens: list[int]) -> ScoringInputs:
    return ScoringInputs(
        sample_id="sample",
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="prefix",
        prefix_hash="hash",
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=(),
        n_prefix=len(prefix_tokens),
        n_output=len(output_tokens),
        output_byte_weights=t.ones(len(output_tokens), dtype=t.int64),
        truncated=False,
    )


def _encdec_inputs(*, prefix_tokens: list[int], output_tokens: list[int]) -> ScoringInputs:
    decoded_len = len(_TinyTokenizer().decode(prefix_tokens))
    return replace(
        _inputs(prefix_tokens=prefix_tokens, output_tokens=output_tokens),
        prefix_context_segments=(
            ContextFeatureSegment(
                feature_ids=(7,),
                buckets=(2,),
                char_start=0,
                char_end=decoded_len,
            ),
        ),
    )


def _token_bytes(vocab_size: int = 32) -> t.Tensor:
    return t.ones(vocab_size, dtype=t.int64)


def _canonical_bpb(
    *,
    bundle: TargetModelBundle,
    inputs: ScoringInputs,
    intervention: Intervention | None = None,
) -> float:
    return score_sample(
        inputs=inputs,
        target_bundle=bundle,
        intervention=intervention or Intervention.none(),
        device=t.device("cpu"),
        coefficient_str="1.0",
        method_name="diffmean_decoder_only",
    ).bpb_intervened


def test_decoder_surrogate_loss_matches_canonical_score_sample() -> None:
    model = _ToyDecoder()
    bundle = _bundle(model)
    inputs = _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])

    gradient_result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    assert gradient_result.loss == pytest.approx(
        _canonical_bpb(bundle=bundle, inputs=inputs)
    )
    assert gradient_result.token_gradient.shape == (len(inputs.prefix_tokens), 3)
    assert t.isfinite(gradient_result.token_gradient).all()
    assert t.equal(
        gradient_result.embedding_weight,
        model.transformer.wte.weight.detach().float().cpu(),
    )


def test_surrogate_gradient_matches_finite_difference_for_prompt_embedding() -> None:
    model = _ToyDecoder()
    bundle = _bundle(model)
    inputs = _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])
    result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    epsilon = 1.0e-3
    token_id = 3
    dim = 1
    with t.no_grad():
        model.transformer.wte.weight[token_id, dim] += epsilon
    plus = _canonical_bpb(bundle=bundle, inputs=inputs)
    with t.no_grad():
        model.transformer.wte.weight[token_id, dim] -= 2 * epsilon
    minus = _canonical_bpb(bundle=bundle, inputs=inputs)

    finite_difference = (plus - minus) / (2 * epsilon)
    assert result.token_gradient[-1, dim].item() == pytest.approx(
        finite_difference,
        rel=2.0e-3,
        abs=2.0e-3,
    )


def test_surrogate_zeroes_prompt_gradients_for_truncated_prompt_positions() -> None:
    model = _ToyDecoder()
    bundle = _bundle(model, max_seq_len=4)
    inputs = _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])
    truncated_inputs = replace(inputs, truncated=True)

    result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    assert result.loss == pytest.approx(
        _canonical_bpb(bundle=bundle, inputs=truncated_inputs)
    )
    assert t.equal(result.token_gradient[:2], t.zeros_like(result.token_gradient[:2]))
    assert t.isfinite(result.token_gradient[2:]).all()


def test_decoder_surrogate_concatenates_value_embedding_gradients() -> None:
    model = _ToyDecoder(value_dims=(2,))
    bundle = _bundle(model)
    inputs = _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])

    result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    assert result.token_gradient.shape == (len(inputs.prefix_tokens), 5)
    value_embed = model.value_embeds["0"]
    assert isinstance(value_embed, t.nn.Embedding)
    expected_weight = t.cat(
        [
            model.transformer.wte.weight.detach().float().cpu(),
            value_embed.weight.detach().float().cpu(),
        ],
        dim=1,
    )
    assert t.equal(result.embedding_weight, expected_weight)
    assert t.isfinite(result.token_gradient).all()


def test_surrogate_embedding_weight_handles_wrappers_and_value_embed_order() -> None:
    wrapped = _InnerModelWrapper(_ToyDecoder(value_dims=(2,)))

    weight = bpb_surrogate_embedding_weight(wrapped)

    assert weight.shape == (32, 5)

    non_numeric = _ToyDecoder()
    non_numeric.value_embeds = t.nn.ModuleDict({"z": t.nn.Embedding(32, 1)})
    assert bpb_surrogate_embedding_weight(non_numeric).shape == (32, 4)

    no_items = _ToyDecoder()
    no_items_any: Any = no_items
    no_items_any.value_embeds = t.nn.Linear(1, 1)
    assert bpb_surrogate_embedding_weight(no_items).shape == (32, 3)


def test_surrogate_embedding_weight_rejects_missing_or_incompatible_weights() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="does not expose"):
        bpb_surrogate_embedding_weight(t.nn.Linear(1, 1))
    with pytest.raises(AxBenchBpbAdvError, match="no embedding weights"):
        _concatenate_embedding_parts([])
    with pytest.raises(AxBenchBpbAdvError, match="same vocab row count"):
        _concatenate_embedding_parts([t.zeros(2, 1), t.zeros(3, 1)])


def test_prompt_gradient_helper_zeroes_out_of_range_positions() -> None:
    gradient = _prompt_gradient_from_full_gradient(
        full_gradient=t.tensor([[1.0, 2.0]]),
        prompt_len=3,
        prompt_prefix_len=0,
        tokens_dropped_total=0,
    )

    assert t.equal(gradient[0], t.tensor([1.0, 2.0]))
    assert t.equal(gradient[1:], t.zeros(2, 2))


def test_encdec_surrogate_loss_matches_canonical_score_sample_with_context_segments() -> None:
    model = _ToyEncDec()
    bundle = _encdec_bundle(model)
    inputs = _encdec_inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])
    intervention = Intervention.ours(feature_conditions={11: 3})

    result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=inputs.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )

    assert result.loss == pytest.approx(
        _canonical_bpb(bundle=bundle, inputs=inputs, intervention=intervention)
    )
    assert result.token_gradient.shape == (len(inputs.prefix_tokens), 3)
    assert t.isfinite(result.token_gradient).all()


def test_encdec_surrogate_projects_context_segments_after_left_truncation() -> None:
    model = _ToyEncDec()
    bundle = _encdec_bundle(model, max_seq_len=4)
    inputs = _encdec_inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])
    truncated_inputs = replace(inputs, truncated=True)
    intervention = Intervention.ours(feature_conditions={11: 3})

    result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=inputs.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )

    assert result.loss == pytest.approx(
        _canonical_bpb(bundle=bundle, inputs=truncated_inputs, intervention=intervention)
    )
    assert t.equal(result.token_gradient[:2], t.zeros_like(result.token_gradient[:2]))
    assert t.isfinite(result.token_gradient[2:]).all()


def test_encdec_surrogate_threads_allowed_ids_from_model_feature_vocab() -> None:
    """`_build_teacher_forced_forward_inputs` (surrogate.py's own caller of
    `_build_encdec_seg_map_and_encoder`, used by the faster-GCG / GEPA
    adversarial attack layer on top of any encdec conditioning method) must
    thread `allowed_ids` from `target_bundle.model.feature_vocab`, exactly
    like teacher_force.py's two callers -- not leave the V-filter primitive
    ready but never invoked with a real V at runtime.

    Proof: a context-cache segment carrying an out-of-V id (13, excluded by
    `allowed=[7]`) must produce the IDENTICAL loss to the same segment
    pre-filtered down to just the in-V id (7) -- `_ToyEncDec.forward`'s
    `slot_signal` sums `(enc_feature_ids + enc_magnitudes) * enc_mask`, so
    an unfiltered out-of-V id would change the loss. If `allowed_ids` were
    NOT threaded here, this equality would fail (the "with_oov" loss would
    include feature 13's contribution).
    """
    allowed = t.tensor([7], dtype=t.int64)
    model = _CompactedToyEncDec(allowed)
    bundle = _encdec_bundle(model)
    intervention = Intervention.ours(feature_conditions={})

    inputs_with_oov = replace(
        _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5]),
        prefix_context_segments=(
            ContextFeatureSegment(
                feature_ids=(7, 13), buckets=(2, 9), char_start=0, char_end=4,
            ),
        ),
    )
    inputs_prefiltered = replace(
        _inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5]),
        prefix_context_segments=(
            ContextFeatureSegment(
                feature_ids=(7,), buckets=(2,), char_start=0, char_end=4,
            ),
        ),
    )

    result_with_oov = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs_with_oov.prefix_tokens),
        output_tokens=list(inputs_with_oov.output_tokens),
        prefix_context_segments=inputs_with_oov.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    result_prefiltered = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs_prefiltered.prefix_tokens),
        output_tokens=list(inputs_prefiltered.output_tokens),
        prefix_context_segments=inputs_prefiltered.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )

    assert result_with_oov.loss == pytest.approx(result_prefiltered.loss)


def test_unclustered_surrogate_passes_explicit_none_cluster_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the oracle-faithful surrogate exception to the U-invariant.

    Unlike ``model_optimize.py``, the unclustered surrogate call includes the
    ``cluster_contract`` kwarg with value ``None``. The callee accepts this
    explicit value, and the oracle has the same call shape.
    """

    observed: list[dict[str, Any]] = []

    def _build_encoder(**kwargs: Any) -> tuple[t.Tensor, ...]:
        observed.append(dict(kwargs))
        return (
            t.zeros((1, 1, 1), dtype=t.int64),
            t.zeros((1, 1, 1)),
            t.ones((1, 1, 1), dtype=t.bool),
            t.zeros((1, 3), dtype=t.int64),
            t.zeros((), dtype=t.int64),
        )

    monkeypatch.setattr(
        surrogate,
        "_build_encdec_seg_map_and_encoder",
        _build_encoder,
    )
    bundle = _encdec_bundle(_ToyEncDec())
    layout = surrogate._build_teacher_forced_layout(
        target_bundle=bundle,
        prompt_tokens=[0, 1],
        output_tokens=[2, 3],
        prefix_context_segments=(),
        intervention=Intervention.none(),
    )

    surrogate._build_teacher_forced_forward_inputs(
        target_bundle=bundle,
        layout=layout,
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    assert len(observed) == 1
    assert "cluster_contract" in observed[0]
    assert observed[0]["cluster_contract"] is None


def test_encdec_surrogate_gradient_raises_before_forward_on_oov_generation_slot_feature() -> None:
    """`_teacher_forced_model_forward`'s eager `assert_enc_ids_in_vocab` call
    (issue #695 PR-1 Task 4) must fire BEFORE the shared model forward.

    `intervention.feature_conditions` (the generation slot) is NOT filtered
    by `_build_encdec_seg_map_and_encoder` -- only the context-cache
    `prefix_segments` are (see that function's own docstring) -- so an
    out-of-V id placed there is exactly the "missed restriction site" class
    of bug this backstop exists to catch. Proven via a forward-pre-hook call
    counter registered directly on the stub model, independent of the
    stub's own internals, so this cannot pass merely because the stub
    happens to raise on its own for some unrelated reason.
    """
    allowed = t.tensor([7], dtype=t.int64)
    model = _CompactedToyEncDec(allowed)
    calls = 0

    def _count_calls(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1

    model.register_forward_pre_hook(_count_calls)
    bundle = _encdec_bundle(model)
    inputs = _encdec_inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])

    with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
        teacher_forced_bpb_gradient(
            target_bundle=bundle,
            prompt_tokens=list(inputs.prefix_tokens),
            output_tokens=list(inputs.output_tokens),
            prefix_context_segments=inputs.prefix_context_segments,
            output_token_bytes=_token_bytes(),
            intervention=Intervention.ours(feature_conditions={99: 3}),
            device=t.device("cpu"),
        )
    assert calls == 0, "assert must fire BEFORE the forward, not after"


def test_encdec_surrogate_loss_raises_before_forward_on_oov_generation_slot_feature() -> None:
    """Same proof for the loss-only path (`teacher_forced_bpb_loss`) -- a
    separate production entry point that funnels through the SAME
    `_teacher_forced_model_forward`, and the Faster-GCG candidate-ranking
    hot path this backstop must also cover."""
    allowed = t.tensor([7], dtype=t.int64)
    model = _CompactedToyEncDec(allowed)
    calls = 0

    def _count_calls(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1

    model.register_forward_pre_hook(_count_calls)
    bundle = _encdec_bundle(model)
    inputs = _encdec_inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])

    with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
        teacher_forced_bpb_loss(
            target_bundle=bundle,
            prompt_tokens=list(inputs.prefix_tokens),
            output_tokens=list(inputs.output_tokens),
            prefix_context_segments=inputs.prefix_context_segments,
            output_token_bytes=_token_bytes(),
            intervention=Intervention.ours(feature_conditions={99: 3}),
            device=t.device("cpu"),
        )
    assert calls == 0, "assert must fire BEFORE the forward, not after"


def test_encdec_surrogate_legacy_allowed_ids_none_unaffected_by_oov_looking_id() -> None:
    """An uncompacted checkpoint (`model.feature_vocab is None`, i.e. the
    plain `_ToyEncDec`, NOT `_CompactedToyEncDec`) has no V at all: the SAME
    feature_conditions id that would be "out of V" on a compacted model must
    NOT raise here, and the forward must actually run -- proving the assert
    is a true no-op on the legacy path, not merely "usually doesn't raise".
    """
    model = _ToyEncDec()
    bundle = _encdec_bundle(model)
    inputs = _encdec_inputs(prefix_tokens=[0, 1, 2, 3], output_tokens=[4, 5])

    result = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(inputs.prefix_tokens),
        output_tokens=list(inputs.output_tokens),
        prefix_context_segments=inputs.prefix_context_segments,
        output_token_bytes=_token_bytes(),
        intervention=Intervention.ours(feature_conditions={99: 3}),
        device=t.device("cpu"),
    )
    assert t.isfinite(t.tensor(result.loss))


@pytest.mark.parametrize(
    ("prompt_tokens", "output_tokens", "max_seq_len", "match"),
    [
        ([], [4], 2, "prompt_tokens"),
        ([0, 1], [], 2, "output_tokens"),
        ([0, 1], [2, 3, 4], 3, "fully truncated"),
    ],
)
def test_surrogate_rejects_invalid_layout_inputs(
    prompt_tokens: list[int],
    output_tokens: list[int],
    max_seq_len: int,
    match: str,
) -> None:
    model = _ToyDecoder()
    bundle = _bundle(model, max_seq_len=max_seq_len)

    with pytest.raises(Exception, match=match):
        teacher_forced_bpb_gradient(
            target_bundle=bundle,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefix_context_segments=(),
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_ToyBadShape(), "loss/mask shape mismatch"),
        (_ToyNonFinite(), "non-finite"),
        (_ToyNoWteUse(), "failed to capture gradient"),
        (_ToyNoValueUse(value_dims=(2,)), "value_embeds"),
        (_ToyBadEmbeddingOutput(), "non-Tensor"),
        (_ToyBadValueDim(), "gradient/embedding dimension mismatch"),
    ],
)
def test_surrogate_rejects_invalid_model_gradient_paths(
    model: t.nn.Module,
    match: str,
) -> None:
    with pytest.raises(Exception, match=match):
        teacher_forced_bpb_gradient(
            target_bundle=_generic_decoder_bundle(model),
            prompt_tokens=[0, 1, 2, 3],
            output_tokens=[4, 5],
            prefix_context_segments=(),
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


@pytest.mark.parametrize(
    ("token_bytes", "match"),
    [
        (t.ones(5, dtype=t.int64), "outside tokenizer vocab"),
        (t.zeros(32, dtype=t.int64), "zero bytes"),
    ],
)
def test_surrogate_rejects_invalid_token_byte_weights(
    token_bytes: t.Tensor,
    match: str,
) -> None:
    model = _ToyDecoder()
    bundle = _bundle(model)

    with pytest.raises(Exception, match=match):
        teacher_forced_bpb_gradient(
            target_bundle=bundle,
            prompt_tokens=[0, 1, 2, 3],
            output_tokens=[4, 5],
            prefix_context_segments=(),
            output_token_bytes=token_bytes,
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


def test_objective_gradient_matches_finite_difference_of_g() -> None:
    """``objective_gradient`` is the gradient of the dual-completion margin:
    two completions of the SAME prompt scored under ONE intervention.

    g = loss_contrast - loss_desired (the search objective to MINIMIZE), i.e.
    primary_margin_y. The desired/contrast pair here is the positive-mode
    mapping (desired = concept_rich, contrast = concept_avoiding).
    """

    model = _ToyDecoder()
    bundle = _bundle(model)
    prompt_tokens = [0, 1, 2, 3]
    rich_inputs = _inputs(prefix_tokens=prompt_tokens, output_tokens=[4, 5])
    avoid_inputs = replace(
        rich_inputs,
        content_class="concept_avoiding",
        sample_id="avoid",
        output_tokens=[6, 7],
    )
    intervention = Intervention.none()

    def _losses() -> tuple[float, float]:
        desired = _canonical_bpb(
            bundle=bundle, inputs=rich_inputs, intervention=intervention
        )
        contrast = _canonical_bpb(
            bundle=bundle, inputs=avoid_inputs, intervention=intervention
        )
        return desired, contrast

    # Two teacher-forced gradients under the SAME intervention, DIFFERENT outputs.
    grad_desired = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=list(rich_inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )
    grad_contrast = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=list(avoid_inputs.output_tokens),
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=intervention,
        device=t.device("cpu"),
    )

    analytic = objective_gradient(
        grad_loss_desired=grad_desired.token_gradient,
        grad_loss_contrast=grad_contrast.token_gradient,
    )

    # Verify g equals the positive-mode margin (avoid - rich).
    loss_desired, loss_contrast = _losses()
    g_value = objective_value(
        loss_desired=loss_desired,
        loss_contrast=loss_contrast,
    )
    assert g_value == pytest.approx(loss_contrast - loss_desired)

    # Finite-difference of g w.r.t. the final (uniquely occurring) prompt token's
    # embedding entry.
    epsilon = 1.0e-3
    token_id = prompt_tokens[-1]
    dim = 1
    with t.no_grad():
        model.transformer.wte.weight[token_id, dim] += epsilon
    plus_desired, plus_contrast = _losses()
    with t.no_grad():
        model.transformer.wte.weight[token_id, dim] -= 2 * epsilon
    minus_desired, minus_contrast = _losses()
    with t.no_grad():
        model.transformer.wte.weight[token_id, dim] += epsilon

    g_plus = objective_value(
        loss_desired=plus_desired,
        loss_contrast=plus_contrast,
    )
    g_minus = objective_value(
        loss_desired=minus_desired,
        loss_contrast=minus_contrast,
    )
    finite_difference = (g_plus - g_minus) / (2 * epsilon)

    assert analytic[-1, dim].item() == pytest.approx(
        finite_difference,
        rel=2.0e-3,
        abs=2.0e-3,
    )


def test_surrogate_rejects_internal_byte_weight_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyDecoder()
    bundle = _bundle(model)
    monkeypatch.setattr(
        surrogate,
        "_output_byte_weights",
        lambda **_kwargs: t.ones(1, dtype=t.float32),
    )

    with pytest.raises(AxBenchBpbAdvError, match="byte-weight/output loss length"):
        teacher_forced_bpb_gradient(
            target_bundle=bundle,
            prompt_tokens=[0, 1, 2, 3],
            output_tokens=[4, 5],
            prefix_context_segments=(),
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )


def test_surrogate_clamps_non_finite_token_gradient_to_finite() -> None:
    """A finite loss with a non-finite gradient must NOT propagate non-finite
    values out of ``teacher_forced_bpb_gradient``.

    The forward loss is finite (so the finite-loss guard at surrogate.py:307
    passes), but its derivative w.r.t. the captured wte output is nan. The
    producer must clamp non-finite gradient entries to zero so downstream
    Faster-GCG ranking/sampling degrades gracefully instead of crashing.
    """

    result = teacher_forced_bpb_gradient(
        target_bundle=_generic_decoder_bundle(_ToyFiniteLossNonFiniteGrad()),
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
    )

    assert t.isfinite(result.token_gradient).all()
    assert t.isfinite(t.tensor(result.loss))


def test_surrogate_logs_warning_when_clamping_non_finite_gradient(caplog) -> None:
    """Clamping a non-finite gradient must be OBSERVABLE, not silent: the loss
    guard RAISES on a non-finite loss, so a non-finite gradient (which we clamp
    rather than raise, to keep the search alive) must at least emit a warning so a
    real numerical blowup is not masked for the whole GCG run."""

    with caplog.at_level("WARNING"):
        teacher_forced_bpb_gradient(
            target_bundle=_generic_decoder_bundle(_ToyFiniteLossNonFiniteGrad()),
            prompt_tokens=[0, 1, 2, 3],
            output_tokens=[4, 5],
            prefix_context_segments=(),
            output_token_bytes=_token_bytes(),
            intervention=Intervention.none(),
            device=t.device("cpu"),
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any("non-finite" in m and "gradient" in m for m in messages), messages
