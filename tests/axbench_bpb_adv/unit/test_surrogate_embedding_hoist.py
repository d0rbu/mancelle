"""OPT #1 bit-identity: hoisting the static embedding copy out of the gradient.

``teacher_forced_bpb_gradient`` rebuilt a fresh GPU->CPU float32 copy of the
STATIC embedding tables (``wte`` + value-embeds) on EVERY call and returned it as
``BpbGradientResult.embedding_weight`` -- ~720MB / ~260ms of a ~266ms gradient
eval on d12_base, even though the model params do not change during GCG (it
optimizes input tokens, not weights). The optimization hoists that copy: when an
``embedding_weight`` tensor is passed in (the cached
``BpbObjectiveCallbacks._embedding_weight``), the call SKIPS the rebuild and
returns the passed tensor.

These tests pin the exact-equality contract:
  * the cached tensor is byte-identical to a fresh ``bpb_surrogate_embedding_weight``;
  * passing it produces byte-identical ``.loss`` and ``.token_gradient`` to the
    no-kwarg (rebuild) path, and ``.embedding_weight IS`` the passed tensor;
  * the contract holds with value-embeds present (concatenated layout); and
  * a wrong-width passed tensor still trips the existing dimension guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import BpbObjectiveCallbacks
from steering.eval.axbench_bpb_adv.surrogate import (
    bpb_surrogate_embedding_weight,
    teacher_forced_bpb_gradient,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import (
    _bundle,
    _token_bytes,
    _ToyDecoder,
)


def _call(bundle, *, embedding_weight=None):
    return teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=[0, 1, 2, 3],
        output_tokens=[4, 5],
        prefix_context_segments=(),
        output_token_bytes=_token_bytes(),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        embedding_weight=embedding_weight,
    )


def test_cached_embedding_weight_is_byte_identical_to_rebuild() -> None:
    """OPT #1 hoist exactness: the cached embedding matrix equals a fresh rebuild."""

    model = _ToyDecoder(value_dims=(2,))
    rebuild_a = bpb_surrogate_embedding_weight(model)
    rebuild_b = bpb_surrogate_embedding_weight(model)
    assert t.equal(rebuild_a, rebuild_b)


def _objective_callbacks_for(model: _ToyDecoder) -> BpbObjectiveCallbacks:
    return BpbObjectiveCallbacks(
        target_bundle=cast(
            Any,
            SimpleNamespace(
                model=model,
                tokenizer=None,
                max_seq_len=16,
                target_name="decoder_only",
            ),
        ),
        target_model_meta=None,
        tokenizer=None,
        token_bytes=t.ones(32, dtype=t.int64),
        source_paths=cast(Any, SimpleNamespace()),
        pair_scorer=cast(Any, SimpleNamespace()),
        context_segments_by_prefix_hash={},
        device=t.device("cpu"),
        lora=None,
    )


@pytest.mark.parametrize("value_dims", [(), (2,), (2, 1)])
def test_callbacks_cached_embedding_is_byte_identical_to_fresh_surrogate(
    value_dims: tuple[int, ...],
) -> None:
    """Test #2: ``BpbObjectiveCallbacks._embedding_weight`` (cached once in
    ``__post_init__``) is byte-identical to a fresh ``bpb_surrogate_embedding_weight``
    -- proving the hoist serves exactly what the rebuild would have produced."""

    model = _ToyDecoder(value_dims=value_dims)
    callbacks = _objective_callbacks_for(model)
    fresh = bpb_surrogate_embedding_weight(model)

    assert t.equal(callbacks.embedding_weight(), fresh)
    assert t.equal(callbacks._embedding_weight, fresh)


@pytest.mark.parametrize("value_dims", [(), (2,), (2, 1)])
def test_passed_embedding_weight_yields_byte_identical_loss_and_gradient(
    value_dims: tuple[int, ...],
) -> None:
    model = _ToyDecoder(value_dims=value_dims)
    bundle = _bundle(model)
    cached = bpb_surrogate_embedding_weight(model)

    reference = _call(bundle)
    hoisted = _call(bundle, embedding_weight=cached)

    # The loss scalar and token gradient must be byte-identical (no math changed).
    assert hoisted.loss == reference.loss
    assert t.equal(hoisted.token_gradient, reference.token_gradient)
    # The returned embedding_weight IS the passed tensor (no rebuild/copy).
    assert hoisted.embedding_weight is cached
    # And it is byte-identical to what the rebuild path returns.
    assert t.equal(hoisted.embedding_weight, reference.embedding_weight)


def test_none_embedding_weight_preserves_rebuild_behavior() -> None:
    """Backward-compatible: with no kwarg the rebuild path is unchanged."""

    model = _ToyDecoder(value_dims=(2,))
    bundle = _bundle(model)
    result = _call(bundle, embedding_weight=None)
    assert t.equal(
        result.embedding_weight,
        bpb_surrogate_embedding_weight(model),
    )


def test_passed_embedding_weight_wrong_width_trips_dimension_guard() -> None:
    """A mismatched passed tensor must still trip the gradient/embedding guard."""

    model = _ToyDecoder(value_dims=(2,))
    bundle = _bundle(model)
    # wte d_model=3 + value dim 2 == 5; pass a 4-wide tensor to force mismatch.
    wrong = t.zeros(32, 4, dtype=t.float32)
    with pytest.raises(AxBenchBpbAdvError, match="gradient/embedding dimension mismatch"):
        _call(bundle, embedding_weight=wrong)
