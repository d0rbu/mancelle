from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
from steering.eval.axbench_bpb_adv.config import GepaConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import (
    BpbObjectiveCallbacks,
    _gepa_optimizer_for_objective,
)
from steering.eval.axbench_bpb_adv.surrogate import BpbGradientResult
from steering.eval.target_conditioning import ContextFeatureSegment


class _CharTokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


class _PairScorer:
    concepts_by_id = {1: SimpleNamespace(concept_id=1)}

    def _base_steered_intervention(
        self,
        *,
        method: Any,
        concept: Any,
        coefficient: Any,
        candidate_row: dict[str, Any],
    ) -> str:
        del method, concept, coefficient, candidate_row
        return "steered-intervention"


def _make_objective(
    *,
    mode: str,
    lora: Any,
    fake_gradient: Any,
    monkeypatch,
    fake_loss: Any | None = None,
    device_batch_size: int = 32,
    candidate_eval_device_batch_size: int | None = None,
) -> BpbObjectiveCallbacks:
    monkeypatch.setattr(
        model_optimize,
        "bpb_surrogate_embedding_weight",
        lambda _model: t.zeros(5, 1),
    )
    def fake_gradient_batch(
        *,
        prompts: list[list[int]],
        outputs: list[list[int]],
        intervention: Any,
        lora: Any,
        embedding_weight: t.Tensor,
        **_kwargs: Any,
    ) -> list[BpbGradientResult]:
        return [
            fake_gradient(
                prompt_tokens=prompt,
                output_tokens=output,
                intervention=intervention,
                lora=lora,
                embedding_weight=embedding_weight,
            )
            for prompt, output in zip(prompts, outputs, strict=True)
        ]

    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_gradient_batch",
        fake_gradient_batch,
    )
    if fake_loss is None:
        # Default BATCHED loss stub: derives each sample's loss from the gradient
        # stub's result (keyed by completion), mirroring the batched candidate eval.
        def fake_loss(*, outputs: list[list[int]], **_kwargs: Any) -> list[float]:
            return [float(fake_gradient(output_tokens=out).loss) for out in outputs]

    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_loss_shared_prefix_kv_batch",
        fake_loss,
    )
    monkeypatch.setattr(
        model_optimize,
        "get_method_by_name",
        lambda name: SimpleNamespace(name=name),
    )
    return BpbObjectiveCallbacks(
        target_bundle=cast(
            Any,
            SimpleNamespace(
                model=object(),
                tokenizer=None,
                max_seq_len=16,
                target_name="decoder_only",
            ),
        ),
        target_model_meta=None,
        tokenizer=None,
        token_bytes=t.ones(256, dtype=t.int64),
        source_paths=cast(Any, SimpleNamespace()),
        pair_scorer=cast(Any, _PairScorer()),
        context_segments_by_prefix_hash={},
        device=t.device("cpu"),
        lora=lora,
        device_batch_size=device_batch_size,
        candidate_eval_device_batch_size=candidate_eval_device_batch_size,
    )


def _row(*, mode: str) -> dict[str, Any]:
    return {
        "attack_row_id": "row",
        "core_prefix_id": "prefix",
        "method_name": "diffmean_encdec",
        "concept_id": 1,
        "coefficient_str": "1.0",
        "target_model": "decoder_only",
        "context_signal_mode": "not_applicable",
        "mode": mode,
        "concept_rich_output_tokens": [98],
        "concept_avoiding_output_tokens": [99],
        "source_prompt_tokens": [0, 10, 11],
        "editable_token_start": 1,
        "editable_token_count": 2,
    }


def _pair_row(
    *,
    attack_row_id: str,
    rich_output: list[int],
    avoid_output: list[int],
    mode: str = "positive",
) -> dict[str, Any]:
    """A distinct minibatch pair (its own attack_row_id + completions)."""

    return {
        "attack_row_id": attack_row_id,
        "core_prefix_id": attack_row_id,
        "method_name": "diffmean_encdec",
        "concept_id": 1,
        "coefficient_str": "1.0",
        "target_model": "decoder_only",
        "context_signal_mode": "not_applicable",
        "mode": mode,
        "concept_rich_output_tokens": rich_output,
        "concept_avoiding_output_tokens": avoid_output,
        "source_prompt_tokens": [0, 10, 11],
        "editable_token_start": 1,
        "editable_token_count": 2,
    }


def _per_pair_loss(loss_by_output: dict[int, float], loss_calls: list[dict[str, Any]]):
    """A BATCHED loss stub keyed by each sample's FIRST output token (so distinct
    pairs differ). Records one entry PER SAMPLE so ``len(loss_calls)`` counts the
    underlying forward-pass work (a batched call of B samples == B computations),
    not the number of batched calls."""

    def fake_loss(*, outputs: list[list[int]], **_kwargs: Any) -> list[float]:
        result: list[float] = []
        for out in outputs:
            loss_calls.append({"output_tokens": out})
            result.append(float(loss_by_output[int(out[0])]))
        return result

    return fake_loss


def test_universal_cache_correct_and_retained_across_interleaved_minibatch(
    monkeypatch,
) -> None:
    """v2 C3: under a universal minibatch, pairs interleave within a step, so the
    per-row cache MUST NOT clear on ``attack_row_id`` change (that thrash would
    recompute every interleaved eval). The cache is keyed by
    ``(attack_row_id, token_digest)`` so distinct pairs never collide even on
    identical prompt tokens, and a re-query of an earlier pair is served from
    cache (no recompute).
    """

    loss_calls: list[dict[str, Any]] = []
    # Two pairs with DISTINCT completions but the SAME prompt tokens.
    pair_a = _pair_row(attack_row_id="A", rich_output=[10], avoid_output=[11])
    pair_b = _pair_row(attack_row_id="B", rich_output=[20], avoid_output=[21])
    tokens = [0, 30, 31]
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(0.0, 0.0, []),
        fake_loss=_per_pair_loss(
            {10: 7.0, 11: 4.0, 20: 5.0, 21: 9.0}, loss_calls
        ),
        monkeypatch=monkeypatch,
    )

    # Interleave A -> B -> A (the minibatch thrash pattern).
    a_first = objective.objective(pair_a, tokens)
    b_value = objective.objective(pair_b, tokens)
    a_again = objective.objective(pair_a, tokens)

    # positive g(a=1) = loss_avoid - loss_rich, computed per pair from its OWN
    # completions (so the two pairs are genuinely different despite equal tokens).
    assert a_first == pytest.approx(4.0 - 7.0)
    assert b_value == pytest.approx(9.0 - 5.0)
    # A is served from cache the second time: identical value AND no recompute.
    assert a_again == a_first
    # Exactly 4 loss computations (2 for A, 2 for B); the re-query of A added none,
    # proving the cache was NOT cleared when B (a different attack_row_id) ran.
    assert len(loss_calls) == 4

    # Both pairs' entries coexist in the cache (distinct keys), so the per-row
    # clear is gone and the (attack_row_id, digest) keying prevents collision.
    key_a = objective._cache_key(pair_a, tokens)
    key_b = objective._cache_key(pair_b, tokens)
    assert key_a != key_b
    assert key_a in objective._cache
    assert key_b in objective._cache


def test_clear_eval_cache_drops_all_group_entries(monkeypatch) -> None:
    """The new explicit group-boundary eviction (replacing the per-row clear)
    empties the cache so a fresh group starts cold (bounded memory)."""

    loss_calls: list[dict[str, Any]] = []
    pair_a = _pair_row(attack_row_id="A", rich_output=[10], avoid_output=[11])
    pair_b = _pair_row(attack_row_id="B", rich_output=[20], avoid_output=[21])
    tokens = [0, 30, 31]
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(0.0, 0.0, []),
        fake_loss=_per_pair_loss(
            {10: 7.0, 11: 4.0, 20: 5.0, 21: 9.0}, loss_calls
        ),
        monkeypatch=monkeypatch,
    )
    objective.objective(pair_a, tokens)
    objective.objective(pair_b, tokens)
    assert objective._cache

    objective.clear_eval_cache()
    assert objective._cache == {}


def _completion_keyed_gradient(rich_loss: float, avoid_loss: float, calls: list[dict[str, Any]]):
    def fake_gradient(**kwargs):
        calls.append(kwargs)
        if kwargs["output_tokens"] == [98]:
            return BpbGradientResult(
                loss=rich_loss,
                token_gradient=t.tensor([[1.0], [2.0], [3.0]]),
                embedding_weight=t.zeros(5, 1),
            )
        return BpbGradientResult(
            loss=avoid_loss,
            token_gradient=t.tensor([[10.0], [20.0], [30.0]]),
            embedding_weight=t.zeros(5, 1),
        )

    return fake_gradient


def test_gradient_batch_preserves_nograd_objective_loss(monkeypatch) -> None:
    """The universal accept gate compares candidate objectives (no_grad
    shared-prefix scorer) against the current trigger's objective.
    ``gradient_provider_batch`` runs FIRST each step over the current trigger; it
    must attach the gradients WITHOUT overwriting the cached no_grad losses with its
    own grad-enabled forward's losses (the two reduction regimes diverge ~1e-1 on the
    compiled flex-attention model), or the accept comparison straddles regimes. Pin:
    after a gradient eval, the objective served from cache is still the NO_GRAD loss.
    """

    NOGRAD = {98: 7.0, 99: 4.0}  # shared-prefix no_grad ranking path
    GRAD = {98: 7.5, 99: 4.9}  # teacher_forced_bpb_gradient_batch (grad-enabled, DIFFERENT)

    def fake_loss_batch(*, outputs: list[list[int]], **_kwargs: Any) -> list[float]:
        return [NOGRAD[int(o[0])] for o in outputs]

    def fake_grad_batch(*, outputs: list[list[int]], **_kwargs: Any) -> list[Any]:
        return [
            BpbGradientResult(
                loss=GRAD[int(o[0])],
                token_gradient=t.tensor([[1.0], [2.0], [3.0]]),
                embedding_weight=t.zeros(5, 1),
            )
            for o in outputs
        ]

    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(0.0, 0.0, []),
        fake_loss=fake_loss_batch,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(model_optimize, "teacher_forced_bpb_gradient_batch", fake_grad_batch)
    row = _row(mode="positive")  # rich=[98], avoid=[99]
    tokens = [0, 10, 11]

    # 1. Objective path establishes the no_grad loss: positive g = avoid - rich.
    obj_before = objective.objective_batch([(row, tokens)])[0]
    assert obj_before == pytest.approx(4.0 - 7.0)

    # 2. Gradient eval for the SAME key (would overwrite losses with the GRAD regime
    #    if the preserve-fix were absent).
    objective.gradient_provider_batch([(row, tokens)], 0)

    # 3. Objective re-served from cache MUST still be the no_grad value, NOT the
    #    grad-path 4.9 - 7.5 = -2.6 that the unfixed overwrite would yield.
    obj_after = objective.objective_batch([(row, tokens)])[0]
    assert obj_after == pytest.approx(4.0 - 7.0)
    entry = objective._cache[objective._cache_key(row, tokens)]
    assert entry.grad_desired is not None
    assert entry.grad_contrast is not None
    assert entry.loss_desired == pytest.approx(7.0)  # no_grad rich, not 7.5
    assert entry.loss_contrast == pytest.approx(4.0)  # no_grad avoid, not 4.9


def _loss_keyed(rich_loss: float, avoid_loss: float, loss_calls: list[dict[str, Any]]):
    """BATCHED loss stub: ``[98]`` -> rich, else avoid; records one entry per
    sample (a B-sample batched call == B forward computations)."""

    def fake_loss(*, outputs: list[list[int]], **_kwargs: Any) -> list[float]:
        result: list[float] = []
        for out in outputs:
            loss_calls.append({"output_tokens": out})
            result.append(rich_loss if out == [98] else avoid_loss)
        return result

    return fake_loss


def test_objective_uses_loss_only_path_not_gradient(monkeypatch) -> None:
    """OPT #2: ``objective()`` must use the forward-only loss path (no backward)."""

    grad_calls: list[dict[str, Any]] = []
    loss_calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, grad_calls),
        fake_loss=_loss_keyed(7.0, 4.0, loss_calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="positive")
    value = objective.objective(row, [0, 10, 11])

    # Loss-only path ran (both completions); the gradient path did NOT.
    assert len(loss_calls) == 2
    assert grad_calls == []
    assert value == pytest.approx(4.0 - 7.0)


def test_objective_uses_only_shared_prefix_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_calls: list[dict[str, Any]] = []

    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.5, 4.5, []),
        monkeypatch=monkeypatch,
    )

    def shared_loss(
        *,
        outputs: list[list[int]],
        source_prompts: list[list[int]],
        editable_spans: list[Any],
        **_kwargs: Any,
    ) -> list[float]:
        shared_calls.append(
            {
                "source_prompts": source_prompts,
                "editable_spans": editable_spans,
            }
        )
        return [7.25 if output == [98] else 4.75 for output in outputs]

    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_loss_shared_prefix_kv_batch",
        shared_loss,
    )
    row = _row(mode="positive")
    tokens = [0, 20, 21]

    assert objective.objective(row, tokens) == pytest.approx(4.75 - 7.25)
    assert len(shared_calls) == 1
    assert shared_calls[0]["source_prompts"] == [
        row["source_prompt_tokens"],
        row["source_prompt_tokens"],
    ]
    assert [span.count for span in shared_calls[0]["editable_spans"]] == [2, 2]

    assert objective._cache_key(row, tokens) in objective._cache

    # GEPA uses the same scorer, with the top-level batch-size contract.
    assert objective.gepa_objective(row, [0, 22, 23]) == pytest.approx(4.75 - 7.25)
    assert len(shared_calls) == 2


def test_candidate_batch_override_is_scoped_away_from_gradient_and_gepa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_batches: list[int] = []
    gradient_batches: list[int] = []

    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(0.0, 0.0, []),
        device_batch_size=16,
        candidate_eval_device_batch_size=32,
        monkeypatch=monkeypatch,
    )

    def shared_loss(
        *, outputs: list[list[int]], device_batch_size: int, **_kwargs: Any
    ) -> list[float]:
        candidate_batches.append(int(device_batch_size))
        return [7.0 if output == [98] else 4.0 for output in outputs]

    def gradient_batch(
        *, outputs: list[list[int]], device_batch_size: int, **kwargs: Any
    ) -> list[BpbGradientResult]:
        assert "kv_cache" not in kwargs
        gradient_batches.append(int(device_batch_size))
        return [
            BpbGradientResult(
                loss=7.5 if output == [98] else 4.5,
                token_gradient=t.ones(3, 1),
                embedding_weight=t.zeros(5, 1),
            )
            for output in outputs
        ]

    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_loss_shared_prefix_kv_batch",
        shared_loss,
    )
    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_gradient_batch",
        gradient_batch,
    )
    row = _row(mode="positive")
    tokens = [0, 20, 21]
    objective.objective(row, tokens)
    objective.gradient_provider_batch([(row, tokens)], 0)
    objective.gepa_objective(row, [0, 22, 23])

    assert candidate_batches == [32, 16]
    assert gradient_batches == [16]
    selected_entry = objective._cache[objective._cache_key(row, tokens)]
    assert selected_entry.loss_desired == pytest.approx(7.0)
    assert selected_entry.loss_contrast == pytest.approx(4.0)


def test_shared_gradient_first_establishes_matching_objective_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_calls: list[int] = []
    grad_calls: list[int] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(0.0, 0.0, []),
        monkeypatch=monkeypatch,
    )

    def shared_loss(*, outputs: list[list[int]], **_kwargs: Any) -> list[float]:
        shared_calls.append(len(outputs))
        return [7.0 if output == [98] else 4.0 for output in outputs]

    def gradient_batch(*, outputs: list[list[int]], **_kwargs: Any) -> list[Any]:
        grad_calls.append(len(outputs))
        return [
            BpbGradientResult(
                loss=70.0 if output == [98] else 40.0,
                token_gradient=t.ones(3, 1),
                embedding_weight=t.zeros(5, 1),
            )
            for output in outputs
        ]

    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_loss_shared_prefix_kv_batch",
        shared_loss,
    )
    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_gradient_batch",
        gradient_batch,
    )
    row = _row(mode="positive")
    tokens = [0, 20, 21]

    objective.gradient_provider_batch([(row, tokens)], 0)

    assert shared_calls == [2]
    assert grad_calls == [2]
    entry = objective._cache[objective._cache_key(row, tokens)]
    assert entry.loss_desired == 7.0
    assert entry.loss_contrast == 4.0


def test_gradient_after_loss_only_recomputes_full_not_stale(monkeypatch) -> None:
    """After a loss-only ``objective()`` on a key, ``gradient_provider_batch()``
    on the SAME key must recompute the full (fwd+bwd) eval -- never serve a
    stale/absent gradient -- and the losses must be byte-identical across both."""

    grad_calls: list[dict[str, Any]] = []
    loss_calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, grad_calls),
        fake_loss=_loss_keyed(7.0, 4.0, loss_calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="positive")
    tokens = [0, 10, 11]

    # 1) loss-only populates a loss-only cache entry.
    obj_value = objective.objective(row, tokens)
    assert len(loss_calls) == 2 and grad_calls == []
    key = objective._cache_key(row, tokens)
    assert objective._cache[key].grad_desired is None
    assert objective._cache[key].grad_contrast is None

    # 2) gradient on the SAME key recomputes the full eval (no stale grad).
    grad = objective.gradient_provider_batch([(row, tokens)], 0)[0]
    assert len(grad_calls) == 2
    grad_rich = t.tensor([[1.0], [2.0], [3.0]])
    grad_avoid = t.tensor([[10.0], [20.0], [30.0]])
    assert t.equal(grad, grad_avoid - grad_rich)

    # 3) cache is now a FULL entry; loss is byte-identical to the loss-only value.
    full = objective._cache[key]
    assert full.grad_desired is not None
    assert full.grad_contrast is not None
    assert (full.loss_contrast - full.loss_desired) == obj_value


def test_full_entry_serves_objective_without_recompute(monkeypatch) -> None:
    """Once a FULL entry is cached, ``objective()`` serves it without any new
    loss-only OR gradient computation."""

    grad_calls: list[dict[str, Any]] = []
    loss_calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, grad_calls),
        fake_loss=_loss_keyed(7.0, 4.0, loss_calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="positive")
    tokens = [0, 10, 11]

    objective.gradient_provider_batch([(row, tokens)], 0)
    assert len(grad_calls) == 2 and len(loss_calls) == 2

    # objective() on the same key serves the FULL entry -> no new computation.
    objective.objective(row, tokens)
    assert len(grad_calls) == 2
    assert len(loss_calls) == 2


def test_gradient_served_from_full_cache_without_recompute(monkeypatch) -> None:
    grad_calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, grad_calls),
        fake_loss=_loss_keyed(7.0, 4.0, []),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="positive")
    tokens = [0, 10, 11]

    objective.gradient_provider_batch([(row, tokens)], 0)
    objective.gradient_provider_batch([(row, tokens)], 1)
    # Second call hits the cached FULL entry -> no second pair of grad computations.
    assert len(grad_calls) == 2


def test_objective_positive_mode_equals_primary_margin(monkeypatch) -> None:
    # positive: desired=rich, contrast=avoid.
    # g = loss_avoid - loss_rich (the unweighted steering margin).
    calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="positive")
    value = objective.objective(row, [0, 10, 11])
    assert value == pytest.approx(4.0 - 7.0)


def test_objective_negative_mode_flips_desired_and_contrast(monkeypatch) -> None:
    # negative: desired=avoid, contrast=rich.
    # g = loss_rich - loss_avoid = negative primary_margin_y.
    calls: list[dict[str, Any]] = []
    rich_loss, avoid_loss = 7.0, 4.0
    objective = _make_objective(
        mode="negative",
        lora=None,
        fake_gradient=_completion_keyed_gradient(rich_loss, avoid_loss, calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="negative")
    value = objective.objective(row, [0, 10, 11])
    assert value == pytest.approx(rich_loss - avoid_loss)


def test_gradient_positive_mode_uses_g_form(monkeypatch) -> None:
    # positive: grad g = grad_avoid - grad_rich (the unweighted steering margin grad).
    calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="positive")
    grad = objective.gradient_provider_batch([(row, [0, 10, 11])], 0)[0]
    grad_rich = t.tensor([[1.0], [2.0], [3.0]])
    grad_avoid = t.tensor([[10.0], [20.0], [30.0]])
    assert t.equal(grad, grad_avoid - grad_rich)


def test_gradient_negative_mode_uses_g_form(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    objective = _make_objective(
        mode="negative",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, calls),
        monkeypatch=monkeypatch,
    )
    row = _row(mode="negative")
    grad = objective.gradient_provider_batch([(row, [0, 10, 11])], 0)[0]
    grad_rich = t.tensor([[1.0], [2.0], [3.0]])
    grad_avoid = t.tensor([[10.0], [20.0], [30.0]])
    # negative: desired=avoid, contrast=rich -> grad_rich - grad_avoid
    assert t.equal(grad, grad_rich - grad_avoid)


def test_context_segments_rejects_recomputed_context(monkeypatch) -> None:
    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, []),
        monkeypatch=monkeypatch,
    )
    # Seed a non-None source_segments tuple so the cache-miss raise (a DIFFERENT
    # message) is not hit and execution reaches the context_signal_mode dispatch.
    objective.context_segments_by_prefix_hash = {
        "deadbeef": (
            ContextFeatureSegment(
                feature_ids=(3,),
                buckets=(4,),
                char_start=0,
                char_end=1,
            ),
        )
    }
    row = {
        "target_model": "encdec",
        "prefix_hash": "deadbeef",
        "source_prompt_tokens": [1, 2, 3],
        "context_signal_mode": "recomputed_context",
    }
    with pytest.raises(AxBenchBpbAdvError, match="unsupported context_signal_mode"):
        objective._context_segments(row, [1, 2, 3])


def _seeded_encdec_objective(monkeypatch) -> BpbObjectiveCallbacks:
    """An objective wired for the encdec context-dispatch branches.

    Seeds a non-empty source-segment tuple keyed by ``deadbeef`` and installs a
    char tokenizer so ``project_fixed_context_segments`` can run model-free.
    """

    objective = _make_objective(
        mode="positive",
        lora=None,
        fake_gradient=_completion_keyed_gradient(7.0, 4.0, []),
        monkeypatch=monkeypatch,
    )
    objective.tokenizer = _CharTokenizer()
    objective.context_segments_by_prefix_hash = {
        "deadbeef": (
            ContextFeatureSegment(
                feature_ids=(3,),
                buckets=(4,),
                char_start=0,
                char_end=2,
            ),
        )
    }
    return objective


def test_context_segments_encdec_fixed_context_projects_segments(monkeypatch) -> None:
    # encdec + fixed_context -> project the source segment onto the (length-
    # preserving) attacked prompt. Source "ab" (tokens [0, 97, 98]); the attacked
    # prompt replaces 'b'(98) with 'x'(120), so the projection keeps the same
    # char span / feature payload.
    objective = _seeded_encdec_objective(monkeypatch)
    row = {
        "target_model": "encdec",
        "prefix_hash": "deadbeef",
        "source_prompt_tokens": [0, 97, 98],
        "context_signal_mode": "fixed_context",
    }

    out = objective._context_segments(row, [0, 97, 120])

    assert len(out) == 1
    projected = out[0]
    assert projected.feature_ids == (3,)
    assert projected.buckets == (4,)
    assert (projected.char_start, projected.char_end) == (0, 2)


def test_context_segments_encdec_not_applicable_passes_through(monkeypatch) -> None:
    # encdec + not_applicable -> the source segments are returned unchanged.
    objective = _seeded_encdec_objective(monkeypatch)
    source_segments = objective.context_segments_by_prefix_hash["deadbeef"]
    row = {
        "target_model": "encdec",
        "prefix_hash": "deadbeef",
        "source_prompt_tokens": [0, 97, 98],
        "context_signal_mode": "not_applicable",
    }

    out = objective._context_segments(row, [0, 97, 120])

    assert out is source_segments


def test_context_segments_encdec_cache_miss_raises(monkeypatch) -> None:
    # encdec with a prefix_hash absent from the cache -> the distinct cache-miss
    # raise (NOT the unsupported-mode message). The cache is seeded for
    # "deadbeef" only.
    objective = _seeded_encdec_objective(monkeypatch)
    row = {
        "target_model": "encdec",
        "prefix_hash": "missing",
        "source_prompt_tokens": [0, 97, 98],
        "context_signal_mode": "not_applicable",
    }

    with pytest.raises(AxBenchBpbAdvError, match="context cache miss for prefix_hash"):
        objective._context_segments(row, [0, 97, 120])


# ---------------------------------------------------------------------------
# Multi-pair-group coverage for objective_batch / gradient_provider_batch.
#
# Every committed batch test drives a SINGLE pair-group (one intervention) with a
# SINGLE item. The dominant universal-GCG path (optimize._objectives_of_triggers)
# instead flattens EVERY (candidate-trigger, eval-pair) into ONE
# objective_batch/gradient_provider_batch call. Inside, misses are grouped by
# _pair_intervention_key: items sharing a pair (same row, DIFFERENT candidate
# trigger tokens) land in ONE group and are scored together via the
# 2*offset/2*offset+1 rich-then-avoid map, while items of a DIFFERENT pair form a
# SECOND group under its OWN intervention.
#
# These tests reproduce that flattening with TWO eval pairs (rowA, rowB --
# distinct concept_id => distinct intervention marker + distinct key) and TWO
# candidate triggers (T1, T2), so a single call yields exactly two groups, each
# holding two items (offset 0 = T1, offset 1 = T2). They pin:
#   (1) per-pair, per-trigger correctness with the 2*offset / 2*offset+1 map at
#       offset 0 AND offset 1 (a swapped map mixes a pair's rich/avoid across
#       triggers and yields the wrong g),
#   (2) NO cross-intervention contamination (group A's forward saw ONLY A's
#       intervention + A's samples; B's saw ONLY B's),
#   (3) a second identical call serves from cache with ZERO new forwards.
# ---------------------------------------------------------------------------


class _ConceptKeyedScorer:
    """A pair-scorer whose base intervention is DISTINCT per concept group.

    ``concepts_by_id`` registers the two concept ids the multi-group tests use,
    and ``_base_steered_intervention`` returns a marker keyed on the candidate
    row's ``concept_id`` -- so group A (concept 1) and group B (concept 2) get
    genuinely different interventions. The batch stubs record the marker they
    were called with, so a forward that mixed two groups (or built one shared
    intervention from the wrong item) is detectable.
    """

    concepts_by_id = {
        1: SimpleNamespace(concept_id=1),
        2: SimpleNamespace(concept_id=2),
    }

    def _base_steered_intervention(
        self,
        *,
        method: Any,
        concept: Any,
        coefficient: Any,
        candidate_row: dict[str, Any],
    ) -> str:
        del method, concept, coefficient
        return f"intervention-concept-{int(candidate_row['concept_id'])}"


def _eval_pair_row(
    *,
    attack_row_id: str,
    core_prefix_id: str,
    concept_id: int,
    rich_output: list[int],
    avoid_output: list[int],
    mode: str = "positive",
) -> dict[str, Any]:
    """One eval-batch pair. ``concept_id`` + ``core_prefix_id`` select its
    pair-intervention group; every candidate trigger of this pair reuses this row
    (only the attacked PROMPT tokens vary), exactly as
    ``optimize._objectives_of_triggers`` flattens (trigger, pair)."""

    return {
        "attack_row_id": attack_row_id,
        "core_prefix_id": core_prefix_id,
        "method_name": "diffmean_encdec",
        "concept_id": concept_id,
        "coefficient_str": "1.0",
        "target_model": "decoder_only",
        "context_signal_mode": "not_applicable",
        "mode": mode,
        "concept_rich_output_tokens": rich_output,
        "concept_avoiding_output_tokens": avoid_output,
        "source_prompt_tokens": [0, 10, 11],
        "editable_token_start": 1,
        "editable_token_count": 2,
    }


def _recording_sample_loss_batch(
    loss_by_sample: dict[Any, float],  # key: (prompt_tuple, output_tuple)
    forwards: list[dict[str, Any]],
):
    """Batched loss stub keyed on the FULL (prompt, output) sample identity.

    Within a pair-group the two items share the SAME completions but DIFFERENT
    candidate-trigger prompts, so keying on the prompt too is what lets offset 0
    and offset 1 carry DISTINCT losses (a swapped 2*offset/2*offset+1 map then
    fails). Records one ``forwards`` entry per batched call (the intervention
    marker + the ordered (prompt, output) samples it spanned).
    """

    def fake_loss(
        *,
        prompts: list[list[int]],
        outputs: list[list[int]],
        intervention: Any,
        **_kwargs: Any,
    ) -> list[float]:
        samples = [
            (tuple(int(x) for x in p), tuple(int(y) for y in o))
            for p, o in zip(prompts, outputs, strict=True)
        ]
        forwards.append({"intervention": intervention, "samples": samples})
        return [float(loss_by_sample[s]) for s in samples]

    return fake_loss


def _recording_sample_grad_batch(
    grad_by_sample: dict[Any, tuple[float, t.Tensor]],  # key: (prompt_tuple, output_tuple)
    forwards: list[dict[str, Any]],
):
    """Batched gradient stub keyed on the FULL (prompt, output) sample identity.

    The fwd+bwd mirror of ``_recording_sample_loss_batch``: one ``forwards`` entry
    per batched call, returning a per-sample ``BpbGradientResult`` with a DISTINCT
    loss AND token_gradient per (prompt, output) so the offset map is pinned.
    """

    def fake_grad(
        *,
        prompts: list[list[int]],
        outputs: list[list[int]],
        intervention: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        samples = [
            (tuple(int(x) for x in p), tuple(int(y) for y in o))
            for p, o in zip(prompts, outputs, strict=True)
        ]
        forwards.append({"intervention": intervention, "samples": samples})
        results: list[Any] = []
        for s in samples:
            loss, grad = grad_by_sample[s]
            results.append(
                BpbGradientResult(
                    loss=loss,
                    token_gradient=grad,
                    embedding_weight=t.zeros(5, 1),
                )
            )
        return results

    return fake_grad


def _make_multigroup_objective(
    *, loss_batch: Any, monkeypatch
) -> BpbObjectiveCallbacks:
    """``_make_objective`` with the concept-keyed scorer (two registered concepts)
    and a caller-supplied recording loss batch stub, so the multi-group tests can
    assert per-forward intervention + samples."""

    monkeypatch.setattr(
        model_optimize,
        "bpb_surrogate_embedding_weight",
        lambda _model: t.zeros(5, 1),
    )
    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_loss_shared_prefix_kv_batch",
        loss_batch,
    )
    monkeypatch.setattr(
        model_optimize,
        "get_method_by_name",
        lambda name: SimpleNamespace(name=name),
    )
    return BpbObjectiveCallbacks(
        target_bundle=cast(
            Any,
            SimpleNamespace(
                model=object(),
                tokenizer=None,
                max_seq_len=16,
                target_name="decoder_only",
            ),
        ),
        target_model_meta=None,
        tokenizer=None,
        token_bytes=t.ones(256, dtype=t.int64),
        source_paths=cast(Any, SimpleNamespace()),
        pair_scorer=cast(Any, _ConceptKeyedScorer()),
        context_segments_by_prefix_hash={},
        device=t.device("cpu"),
        lora=None,
    )


# Two eval pairs (distinct concept_id => distinct intervention + distinct key).
# rowA's completions: rich=[10] avoid=[11]; rowB's: rich=[30] avoid=[31].
_ROW_A = _eval_pair_row(
    attack_row_id="A", core_prefix_id="pa", concept_id=1,
    rich_output=[10], avoid_output=[11],
)
_ROW_B = _eval_pair_row(
    attack_row_id="B", core_prefix_id="pb", concept_id=2,
    rich_output=[30], avoid_output=[31],
)
# Two candidate triggers => two DISTINCT attacked prompts (the only thing that
# varies across a pair's candidates). _objectives_of_triggers order:
#   [(A,T1), (B,T1), (A,T2), (B,T2)]  ->  group A = {offset0:T1, offset1:T2},
#                                          group B = {offset0:T1, offset1:T2}.
_T1 = [0, 40, 41]
_T2 = [0, 42, 43]


def _flattened_items() -> list[tuple[dict[str, Any], list[int]]]:
    return [(_ROW_A, _T1), (_ROW_B, _T1), (_ROW_A, _T2), (_ROW_B, _T2)]


def test_objective_batch_two_groups_correct_no_mixing_then_cached(monkeypatch) -> None:
    """ONE objective_batch flattening TWO pairs x TWO candidate triggers:

    * two pair-groups (distinct concept_id), each holding two items (offset 0 = T1,
      offset 1 = T2) so the 2*offset / 2*offset+1 rich-then-avoid map runs past
      offset 0 with DISTINCT per-trigger losses;
    * each item's g comes from ITS pair's rich/avoid at ITS trigger;
    * group A's forward saw ONLY A's intervention + A's four samples, group B's
      ONLY B's (no cross-contamination);
    * a second identical call performs ZERO new forwards and returns equal values.
    """

    forwards: list[dict[str, Any]] = []
    # Per (prompt, output) loss. Distinct across trigger so offset 0 != offset 1.
    loss_by_sample = {
        (tuple(_T1), (10,)): 7.0, (tuple(_T1), (11,)): 4.0,  # A @ T1 (rich, avoid)
        (tuple(_T2), (10,)): 6.5, (tuple(_T2), (11,)): 5.5,  # A @ T2 (rich, avoid)
        (tuple(_T1), (30,)): 2.5, (tuple(_T1), (31,)): 8.0,  # B @ T1 (rich, avoid)
        (tuple(_T2), (30,)): 3.0, (tuple(_T2), (31,)): 9.5,  # B @ T2 (rich, avoid)
    }
    objective = _make_multigroup_objective(
        loss_batch=_recording_sample_loss_batch(loss_by_sample, forwards),
        monkeypatch=monkeypatch,
    )

    items = _flattened_items()
    values = objective.objective_batch(items)

    # (1) per-(pair, trigger) g = loss_avoid - loss_rich. The T2 entries (offset 1)
    # prove the 2*offset+1 map is not stuck at offset 0.
    assert values[0] == pytest.approx(4.0 - 7.0)  # A @ T1
    assert values[1] == pytest.approx(8.0 - 2.5)  # B @ T1
    assert values[2] == pytest.approx(5.5 - 6.5)  # A @ T2  (offset 1 in group A)
    assert values[3] == pytest.approx(9.5 - 3.0)  # B @ T2  (offset 1 in group B)

    # (2) exactly two forwards (one per group), no cross-mixing.
    assert len(forwards) == 2
    by_intervention = {f["intervention"]: f["samples"] for f in forwards}
    assert set(by_intervention) == {
        "intervention-concept-1",
        "intervention-concept-2",
    }
    # Group A: offset0 (T1 rich,avoid) then offset1 (T2 rich,avoid), ONLY A's rows.
    assert by_intervention["intervention-concept-1"] == [
        (tuple(_T1), (10,)), (tuple(_T1), (11,)),
        (tuple(_T2), (10,)), (tuple(_T2), (11,)),
    ]
    assert by_intervention["intervention-concept-2"] == [
        (tuple(_T1), (30,)), (tuple(_T1), (31,)),
        (tuple(_T2), (30,)), (tuple(_T2), (31,)),
    ]

    # (3) cache serve: a second identical call does ZERO new forwards.
    again = objective.objective_batch(items)
    assert len(forwards) == 2
    assert again == values


def test_gradient_batch_two_groups_correct_no_mixing_then_cached(monkeypatch) -> None:
    """``gradient_provider_batch`` mirror of the two-group objective test.

    ONE call flattening two pairs x two candidate triggers pins: per-(pair,
    trigger) gradient via the 2*offset / 2*offset+1 map at offset 0 AND 1, no
    cross-intervention contamination, and a cached second call with ZERO new
    fwd+bwd.
    """

    grad_forwards: list[dict[str, Any]] = []
    g = lambda v: t.tensor([[v], [2.0 * v], [3.0 * v]])  # noqa: E731
    # Distinct (loss, token_gradient) per (prompt, output) sample.
    grad_by_sample = {
        (tuple(_T1), (10,)): (7.0, g(1.0)), (tuple(_T1), (11,)): (4.0, g(5.0)),
        (tuple(_T2), (10,)): (6.5, g(2.0)), (tuple(_T2), (11,)): (5.5, g(6.0)),
        (tuple(_T1), (30,)): (2.5, g(3.0)), (tuple(_T1), (31,)): (8.0, g(7.0)),
        (tuple(_T2), (30,)): (3.0, g(4.0)), (tuple(_T2), (31,)): (9.5, g(8.0)),
    }
    objective = _make_multigroup_objective(
        # Gradient-first evaluation establishes the objective-facing no_grad losses
        # before attaching gradients from the uncached path.
        loss_batch=_recording_sample_loss_batch(
            {sample: values[0] for sample, values in grad_by_sample.items()}, []
        ),
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_gradient_batch",
        _recording_sample_grad_batch(grad_by_sample, grad_forwards),
    )

    items = _flattened_items()
    grads = objective.gradient_provider_batch(items, 0)

    # (1) per-(pair, trigger) g-grad = grad_avoid - grad_rich; T2 proves offset 1.
    assert t.equal(grads[0], g(5.0) - g(1.0))  # A @ T1
    assert t.equal(grads[1], g(7.0) - g(3.0))  # B @ T1
    assert t.equal(grads[2], g(6.0) - g(2.0))  # A @ T2  (offset 1 in group A)
    assert t.equal(grads[3], g(8.0) - g(4.0))  # B @ T2  (offset 1 in group B)

    # (2) exactly two fwd+bwd (one per group), no cross-mixing.
    assert len(grad_forwards) == 2
    by_intervention = {f["intervention"]: f["samples"] for f in grad_forwards}
    assert set(by_intervention) == {
        "intervention-concept-1",
        "intervention-concept-2",
    }
    assert by_intervention["intervention-concept-1"] == [
        (tuple(_T1), (10,)), (tuple(_T1), (11,)),
        (tuple(_T2), (10,)), (tuple(_T2), (11,)),
    ]
    assert by_intervention["intervention-concept-2"] == [
        (tuple(_T1), (30,)), (tuple(_T1), (31,)),
        (tuple(_T2), (30,)), (tuple(_T2), (31,)),
    ]

    # (3) cache serve: a second identical call does ZERO new fwd+bwd.
    again = objective.gradient_provider_batch(items, 1)
    assert len(grad_forwards) == 2
    for first, second in zip(grads, again, strict=True):
        assert t.equal(first, second)


def test_gradient_batch_two_groups_preserve_nograd_loss(monkeypatch) -> None:
    """Multi-group form of the preserve-no_grad-loss invariant.

    ONE objective_batch (no_grad ranking losses) then ONE gradient_provider_batch
    over the SAME flattened two-pair item list: the gradient forward attaches each
    sample's gradients WITHOUT clobbering the cached no_grad losses (the regimes
    diverge ~1e-1 on the real model). After the grad call, every item's objective
    re-served from cache is still its no_grad g, for BOTH groups and BOTH offsets.
    """

    NOGRAD = {
        (tuple(_T1), (10,)): 7.0, (tuple(_T1), (11,)): 4.0,
        (tuple(_T2), (10,)): 6.5, (tuple(_T2), (11,)): 5.5,
        (tuple(_T1), (30,)): 2.5, (tuple(_T1), (31,)): 8.0,
        (tuple(_T2), (30,)): 3.0, (tuple(_T2), (31,)): 9.5,
    }
    # Grad-regime losses differ in EVERY sample (would change g if they leaked).
    GRAD = {key: value + 0.7 for key, value in NOGRAD.items()}
    loss_forwards: list[dict[str, Any]] = []
    grad_forwards: list[dict[str, Any]] = []
    objective = _make_multigroup_objective(
        loss_batch=_recording_sample_loss_batch(NOGRAD, loss_forwards),
        monkeypatch=monkeypatch,
    )
    grad_by_sample = {
        key: (GRAD[key], t.tensor([[1.0], [2.0], [3.0]])) for key in GRAD
    }
    monkeypatch.setattr(
        model_optimize,
        "teacher_forced_bpb_gradient_batch",
        _recording_sample_grad_batch(grad_by_sample, grad_forwards),
    )

    items = _flattened_items()
    before = objective.objective_batch(items)
    assert before[0] == pytest.approx(4.0 - 7.0)  # A @ T1
    assert before[1] == pytest.approx(8.0 - 2.5)  # B @ T1
    assert before[2] == pytest.approx(5.5 - 6.5)  # A @ T2
    assert before[3] == pytest.approx(9.5 - 3.0)  # B @ T2
    assert len(loss_forwards) == 2  # one no_grad forward per group

    objective.gradient_provider_batch(items, 0)

    after = objective.objective_batch(items)
    # No_grad g preserved for BOTH groups + BOTH offsets (NOT the GRAD regime).
    assert after == before
    # The objective re-serve added NO new no_grad forward (served from FULL entries).
    assert len(loss_forwards) == 2
    for row, tokens in items:
        entry = objective._cache[objective._cache_key(row, tokens)]
        assert entry.grad_desired is not None
        assert entry.grad_contrast is not None
    # Spot-check one entry's losses are the no_grad values, not the +0.7 grad ones.
    entry_a_t1 = objective._cache[objective._cache_key(_ROW_A, _T1)]
    assert entry_a_t1.loss_desired == pytest.approx(7.0)  # no_grad rich, not 7.7
    assert entry_a_t1.loss_contrast == pytest.approx(4.0)  # no_grad avoid, not 4.7


def test_gepa_evaluator_returns_negative_g(monkeypatch) -> None:
    # GEPA MAXIMIZES; the adversary MINIMIZES the search objective g. The
    # evaluator built inside _gepa_optimizer_for_objective must therefore return
    # -g so that GEPA-maximize == minimize-g.
    known_g = 4.5
    objective_calls: list[tuple[dict[str, Any], list[int]]] = []

    def _objective(row: dict[str, Any], tokens: Any) -> float:
        objective_calls.append((row, list(tokens)))
        return known_g

    objective = cast(
        Any,
        SimpleNamespace(
            objective=_objective,
            # vocab_size = 256 so token id 5 (the replacement) is in range.
            embedding_weight=lambda: t.zeros(256, 1),
        ),
    )

    captured_evaluator: dict[str, Any] = {}

    def _fake_optimize_anything(
        *,
        seed_candidate: str,
        evaluator: Any,
        objective: str,
        config: Any,
    ) -> Any:
        del objective, config
        captured_evaluator["fn"] = evaluator
        # GEPA returns the seed unchanged (the optimizer only needs a string).
        return SimpleNamespace(best_candidate=seed_candidate)

    monkeypatch.setattr(model_optimize, "optimize_anything", _fake_optimize_anything)

    optimizer = _gepa_optimizer_for_objective(
        cfg=GepaConfig(reflection_model="dummy-model"),
        objective=objective,
    )

    # One editable token at position 1 (after BOS); a single-token replacement.
    row = {
        "source_prompt_tokens": [0, 97],
        "editable_token_start": 1,
        "editable_token_count": 1,
    }
    result = optimizer(row, [97], 8)

    # The optimizer parses GEPA's best string back into token ids (seed echoed).
    assert result == [97]

    evaluator = captured_evaluator["fn"]
    # A valid single-token candidate JSON -> evaluator dispatches to objective().
    score, info = evaluator("[5]")

    # GEPA-maximize == minimize-g: the evaluator NEGATES g.
    assert score == pytest.approx(-known_g)
    assert info["accepted"] is True
    assert info["search_objective_g"] == pytest.approx(known_g)
    # Objective was called on the attacked prompt (BOS + the replacement token).
    assert objective_calls[-1][1] == [0, 5]
