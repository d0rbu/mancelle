"""Tests for the custom batched-sample generation kernel (PR-2).

The kernel must produce token-for-token identical output to a sequential
loop of single-row HF generate() calls when both are run with the same
per-row seeds. This is the load-bearing correctness invariant; parity
tests live in test_batched_vs_sequential_parity.py.
"""

from __future__ import annotations

import inspect
import json
import unittest.mock as um
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from transformers import Gemma2Config, Gemma2ForCausalLM

from steering.eval.axbench_base import batched_generation as bg_mod
from steering.eval.axbench_base.batched_generation import (
    _apply_top_k,
    _build_prefix_indices,
    _flatten_prefix_seed_rows,
    _generation_pad_token_id,
    _hf_generation_model,
    _module_device,
    _unflatten_continuations,
    batched_generate_seeds,
    cross_prefix_batched_generate_seeds,
)
from steering.eval.axbench_base.config import VerificationPrefillMode
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base._mock_models import (
    MockBaseGemma,
    MockHFGenerateModel,
    _MockKVCache,
    _MockModelOutput,
)


class _RecordingSelectableCache:
    def __init__(self, batch_rows: int) -> None:
        self.batch_rows = batch_rows
        self.selection_calls: list[t.Tensor] = []

    def batch_select_indices(self, indices: t.Tensor) -> None:
        self.selection_calls.append(indices.detach().cpu().clone())
        self.batch_rows = int(indices.numel())


class _RecordingCrossPrefixModel:
    """Small deterministic HF-style model with a public selectable cache."""

    def __init__(
        self,
        *,
        vocab_size: int = 16,
        prefill_token_from_batch_size: bool = False,
    ) -> None:
        self._vocab_size = vocab_size
        self._prefill_token_from_batch_size = prefill_token_from_batch_size
        self.config = SimpleNamespace(pad_token_id=0, eos_token_id=15)
        self.calls: list[dict[str, Any]] = []
        self.prefill_cache: _RecordingSelectableCache | None = None
        self._anchor = t.nn.Parameter(t.zeros(()), requires_grad=False)

    def parameters(self):
        return iter([self._anchor])

    def __call__(
        self,
        input_ids: t.Tensor,
        *,
        past_key_values: object = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> _MockModelOutput:
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "attention_mask": (
                    attention_mask.detach().cpu().clone()
                    if isinstance(attention_mask, t.Tensor)
                    else None
                ),
                "position_ids": (
                    position_ids.detach().cpu().clone()
                    if isinstance(position_ids, t.Tensor)
                    else None
                ),
                "past_key_values": past_key_values,
                "logits_to_keep": kwargs.get("logits_to_keep"),
                "use_cache": use_cache,
            }
        )
        batch, seq = input_ids.shape
        if past_key_values is None:
            cache = _RecordingSelectableCache(batch)
            self.prefill_cache = cache
        else:
            cache = past_key_values
            assert isinstance(cache, _RecordingSelectableCache)
            assert cache.batch_rows == batch
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            seq = min(seq, logits_to_keep)
        logits = t.zeros(batch, seq, self._vocab_size, dtype=t.float32)
        if self._prefill_token_from_batch_size and past_key_values is None:
            logits[..., min(batch, self._vocab_size - 1)] = 1e9
        return _MockModelOutput(logits=logits, past_key_values=cache)


def _base_with_hf_model(model: Any) -> MockBaseGemma:
    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = model
    return base_model


def test_batched_generate_seeds_signature():
    """The public function exists with the expected keyword-only kwargs."""
    sig = inspect.signature(batched_generate_seeds)
    params = sig.parameters
    assert "base_model" in params
    assert "token_ids" in params
    for kw in (
        "seeds",
        "max_new_tokens",
        "temperature",
        "top_k",
        "stop_token_ids",
    ):
        assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY


def test_cross_prefix_batched_generate_seeds_signature():
    """The cross-prefix function exposes K prefix rows and seed rows."""
    sig = inspect.signature(cross_prefix_batched_generate_seeds)
    params = sig.parameters
    assert "base_model" in params
    assert "token_id_rows" in params
    for kw in (
        "seed_rows",
        "max_new_tokens",
        "temperature",
        "top_k",
        "stop_token_ids",
        "prefill_mode",
    ):
        assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY


def test_batched_generate_seeds_empty_prompt_raises():
    with pytest.raises(AxBenchBaseError, match="empty prompt"):
        batched_generate_seeds(
            base_model=object(),
            token_ids=[],
            seeds=[1, 2],
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
        )


def test_cross_prefix_batched_generate_empty_prompt_raises():
    with pytest.raises(AxBenchBaseError, match="empty prompt"):
        cross_prefix_batched_generate_seeds(
            base_model=object(),
            token_id_rows=[[1, 2], []],
            seed_rows=[[1], [2]],
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
            prefill_mode="repeated",
        )


def test_cross_prefix_batched_generate_mismatched_rows_raises():
    with pytest.raises(AxBenchBaseError, match="length mismatch"):
        cross_prefix_batched_generate_seeds(
            base_model=object(),
            token_id_rows=[[1, 2], [3]],
            seed_rows=[[1]],
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
            prefill_mode="repeated",
        )


def test_cross_prefix_batched_generate_empty_token_rows_mismatch_raises():
    with pytest.raises(AxBenchBaseError, match="length mismatch"):
        cross_prefix_batched_generate_seeds(
            base_model=object(),
            token_id_rows=[],
            seed_rows=[[1]],
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
            prefill_mode="repeated",
        )


def test_cross_prefix_batched_generate_zero_max_tokens_returns_nested_empty():
    out = cross_prefix_batched_generate_seeds(
        base_model=object(),
        token_id_rows=[[1, 2, 3], [4]],
        seed_rows=[[7, 11], [13]],
        max_new_tokens=0,
        temperature=0.7,
        top_k=50,
        stop_token_ids=set(),
        prefill_mode="repeated",
    )
    assert out == [[[], []], [[]]]


def test_cross_prefix_batched_generate_empty_seed_row_raises_like_serial():
    with pytest.raises(AxBenchBaseError, match="every prefix needs"):
        cross_prefix_batched_generate_seeds(
            base_model=MockBaseGemma({}),
            token_id_rows=[[1, 2], [3]],
            seed_rows=[[], [13]],
            max_new_tokens=2,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
            prefill_mode="repeated",
        )


def test_batched_generate_seeds_negative_temp_raises():
    with pytest.raises(AxBenchBaseError, match="non-negative"):
        batched_generate_seeds(
            base_model=object(),
            token_ids=[1, 2, 3],
            seeds=[1],
            max_new_tokens=10,
            temperature=-0.1,
            top_k=50,
            stop_token_ids=set(),
        )


def test_cross_prefix_k1_delegates_to_existing_batched_path(monkeypatch: Any):
    token_ids = [1, 2, 3]
    seeds = [7, 11, 13]
    calls: list[dict[str, Any]] = []

    def fake_batched(
        base_model: object,
        token_ids_arg: list[int],
        *,
        seeds: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        stop_token_ids: set[int],
    ) -> list[list[int]]:
        calls.append(
            {
                "base_model": base_model,
                "token_ids": token_ids_arg,
                "seeds": seeds,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "stop_token_ids": stop_token_ids,
            }
        )
        return [[101], [102], [103]]

    monkeypatch.setattr(bg_mod, "batched_generate_seeds", fake_batched)
    base_model = object()
    stop_token_ids = {99}
    actual = bg_mod.cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=[token_ids],
        seed_rows=[seeds],
        max_new_tokens=3,
        temperature=0.7,
        top_k=50,
        stop_token_ids=stop_token_ids,
        prefill_mode="repeated",
    )
    assert actual == [[[101], [102], [103]]]
    assert calls == [
        {
            "base_model": base_model,
            "token_ids": token_ids,
            "seeds": seeds,
            "max_new_tokens": 3,
            "temperature": 0.7,
            "top_k": 50,
            "stop_token_ids": stop_token_ids,
        }
    ]


def test_generation_pad_token_id_prefers_pad_then_eos_then_zero():
    assert _generation_pad_token_id(
        type("_M", (), {"config": type("_C", (), {"pad_token_id": 9})()})()
    ) == 9
    assert _generation_pad_token_id(
        type(
            "_M",
            (),
            {"config": type("_C", (), {"pad_token_id": None, "eos_token_id": 5})()},
        )()
    ) == 5
    assert _generation_pad_token_id(
        type(
            "_M",
            (),
            {
                "config": type(
                    "_C",
                    (),
                    {"pad_token_id": None, "eos_token_id": [7, 8]},
                )()
            },
        )()
    ) == 7
    assert _generation_pad_token_id(type("_M", (), {"config": object()})()) == 0
    assert _generation_pad_token_id(object()) == 0


def test_flatten_prefix_seed_rows_preserves_prefix_major_order():
    flat_rows, flat_seeds, counts = _flatten_prefix_seed_rows(
        [[10, 11], [20], [30, 31, 32]],
        [[1, 2], [3], [4, 5]],
    )
    assert flat_rows == [
        [10, 11],
        [10, 11],
        [20],
        [30, 31, 32],
        [30, 31, 32],
    ]
    assert flat_seeds == [1, 2, 3, 4, 5]
    assert counts == [2, 1, 2]


def test_unflatten_continuations_restores_prefix_seed_shape():
    got = _unflatten_continuations(
        [[101], [102], [201], [301], [302]],
        [2, 1, 2],
    )
    assert got == [
        [[101], [102]],
        [[201]],
        [[301], [302]],
    ]


@pytest.mark.parametrize(
    ("seed_counts", "expected"),
    [
        ([5, 5, 5, 5], [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5),
        ([2, 1, 3], [0, 0, 1, 2, 2, 2]),
        ([5], [0, 0, 0, 0, 0]),
        ([1, 1, 1, 1], [0, 1, 2, 3]),
        ([2, 3], [0, 0, 1, 1, 1]),
    ],
)
def test_build_prefix_indices_exact_prefix_major_layout(
    seed_counts: list[int],
    expected: list[int],
) -> None:
    indices = _build_prefix_indices(seed_counts, device=t.device("cpu"))
    assert indices.tolist() == expected
    assert indices.dtype is t.long
    assert indices.device == t.device("cpu")


def test_shared_prefill_expands_every_decode_state_in_prefix_major_order():
    model = _RecordingCrossPrefixModel()
    output = cross_prefix_batched_generate_seeds(
        base_model=_base_with_hf_model(model),
        token_id_rows=[[10, 11, 12], [7]],
        seed_rows=[[101, 103], [107, 109, 113]],
        max_new_tokens=2,
        temperature=0.0,
        top_k=None,
        stop_token_ids=set(),
        prefill_mode=VerificationPrefillMode.SHARED,
    )

    assert [[len(row) for row in prefix_rows] for prefix_rows in output] == [
        [2, 2],
        [2, 2, 2],
    ]
    assert all(
        isinstance(token, int)
        for prefix_rows in output
        for continuation in prefix_rows
        for token in continuation
    )
    assert len(model.calls) == 3
    prefill = model.calls[0]
    assert tuple(prefill["input_ids"].shape) == (2, 3)
    assert t.equal(
        prefill["input_ids"],
        t.tensor([[10, 11, 12], [0, 0, 7]]),
    )
    assert t.equal(
        prefill["attention_mask"],
        t.tensor([[1, 1, 1], [0, 0, 1]]),
    )
    assert t.equal(
        prefill["position_ids"],
        t.tensor([[0, 1, 2], [0, 0, 0]]),
    )
    assert model.prefill_cache is not None
    assert len(model.prefill_cache.selection_calls) == 1
    assert t.equal(
        model.prefill_cache.selection_calls[0],
        t.tensor([0, 0, 1, 1, 1]),
    )

    first_decode, second_decode = model.calls[1:]
    assert tuple(first_decode["input_ids"].shape) == (5, 1)
    assert t.equal(
        first_decode["attention_mask"],
        t.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        ),
    )
    assert t.equal(
        first_decode["position_ids"],
        t.tensor([[3], [3], [1], [1], [1]]),
    )
    assert t.equal(
        second_decode["position_ids"],
        t.tensor([[4], [4], [2], [2], [2]]),
    )


def test_repeated_prefill_retains_m_row_prefill_without_cache_selection():
    model = _RecordingCrossPrefixModel()
    output = cross_prefix_batched_generate_seeds(
        base_model=_base_with_hf_model(model),
        token_id_rows=[[10, 11, 12], [8, 9], [7]],
        seed_rows=[[101, 103], [107], [109, 113, 127]],
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        stop_token_ids=set(),
        prefill_mode=VerificationPrefillMode.REPEATED,
    )

    assert [len(rows) for rows in output] == [2, 1, 3]
    assert tuple(model.calls[0]["input_ids"].shape) == (6, 3)
    assert all(tuple(call["input_ids"].shape) == (6, 1) for call in model.calls[1:])
    assert model.prefill_cache is not None
    assert model.prefill_cache.selection_calls == []


def test_shared_k1_prefills_once_and_does_not_delegate(monkeypatch: Any):
    model = _RecordingCrossPrefixModel()

    def unexpected_delegate(*_args: Any, **_kwargs: Any) -> list[list[int]]:
        raise AssertionError("shared K=1 must not delegate")

    monkeypatch.setattr(bg_mod, "batched_generate_seeds", unexpected_delegate)
    output = bg_mod.cross_prefix_batched_generate_seeds(
        base_model=_base_with_hf_model(model),
        token_id_rows=[[1, 2, 3]],
        seed_rows=[[7, 11, 13, 17, 19]],
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        stop_token_ids=set(),
        prefill_mode="shared",
    )

    assert [len(rows) for rows in output] == [5]
    assert tuple(model.calls[0]["input_ids"].shape) == (1, 3)
    assert all(tuple(call["input_ids"].shape) == (5, 1) for call in model.calls[1:])
    assert model.prefill_cache is not None
    assert model.prefill_cache.selection_calls[0].tolist() == [0, 0, 0, 0, 0]


class _FixedPrefillCacheModel(_RecordingCrossPrefixModel):
    def __init__(self, cache: object) -> None:
        super().__init__()
        self._cache = cache

    def __call__(
        self,
        input_ids: t.Tensor,
        *,
        past_key_values: object = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> _MockModelOutput:
        del use_cache
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "past_key_values": past_key_values,
            }
        )
        batch = int(input_ids.shape[0])
        logits = t.zeros(batch, 1, self._vocab_size)
        return _MockModelOutput(logits=logits, past_key_values=self._cache)


@pytest.mark.parametrize(
    ("cache", "type_pattern"),
    [
        (None, r"NoneType"),
        (object(), r"builtins\.object"),
        (
            type("_NonCallableCache", (), {"batch_select_indices": 7})(),
            r"_NonCallableCache",
        ),
    ],
)
def test_shared_prefill_rejects_unsupported_cache_without_decode_or_retry(
    cache: object,
    type_pattern: str,
) -> None:
    model = _FixedPrefillCacheModel(cache)
    with pytest.raises(
        AxBenchBaseError,
        match=type_pattern,
    ) as exc_info:
        cross_prefix_batched_generate_seeds(
            base_model=_base_with_hf_model(model),
            token_id_rows=[[1, 2, 3], [4]],
            seed_rows=[[7, 11], [13]],
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
            prefill_mode="shared",
        )

    message = str(exc_info.value)
    assert "prefill_mode='shared'" in message
    assert "Transformers 5.6.2" in message
    assert "verification_prefill_mode=repeated" in message
    assert len(model.calls) == 1
    assert tuple(model.calls[0]["input_ids"].shape) == (2, 3)


def test_shared_prefill_wraps_public_cache_selection_failure_as_cause() -> None:
    class _RaisingCache:
        def __init__(self) -> None:
            self.calls = 0

        def batch_select_indices(self, _indices: t.Tensor) -> None:
            self.calls += 1
            raise RuntimeError("selection exploded")

    cache = _RaisingCache()
    model = _FixedPrefillCacheModel(cache)
    with pytest.raises(
        AxBenchBaseError,
        match=r"source_rows=2 to rows=3",
    ) as exc_info:
        cross_prefix_batched_generate_seeds(
            base_model=_base_with_hf_model(model),
            token_id_rows=[[1, 2, 3], [4]],
            seed_rows=[[7, 11], [13]],
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
            prefill_mode="shared",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "selection exploded"
    assert cache.calls == 1
    assert len(model.calls) == 1


def test_shared_prefill_rejects_malformed_expansion_index_before_selection(
    monkeypatch: Any,
) -> None:
    model = _RecordingCrossPrefixModel()
    monkeypatch.setattr(
        bg_mod,
        "_build_prefix_indices",
        lambda _counts, *, device: t.tensor([0], dtype=t.long, device=device),
    )
    with pytest.raises(
        AxBenchBaseError,
        match=r"expansion index has 1 rows; expected 3",
    ):
        bg_mod.cross_prefix_batched_generate_seeds(
            base_model=_base_with_hf_model(model),
            token_id_rows=[[1, 2, 3], [4]],
            seed_rows=[[7, 11], [13]],
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
            prefill_mode="shared",
        )

    assert len(model.calls) == 1
    assert model.prefill_cache is not None
    assert model.prefill_cache.selection_calls == []


def test_invalid_prefill_mode_fails_with_accepted_values_and_cause() -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"prefill_mode must be 'shared' or 'repeated', got 'auto'",
    ) as exc_info:
        cross_prefix_batched_generate_seeds(
            base_model=object(),
            token_id_rows=[[1]],
            seed_rows=[[7]],
            max_new_tokens=1,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
            prefill_mode="auto",
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("token_rows", "seed_rows"),
    [
        ([[1, 2, 3]], [[7, 11, 13, 17, 19]]),
        (
            [[1], [2, 3, 4], [5, 6]],
            [[7, 11], [13], [17, 19, 23]],
        ),
    ],
)
@pytest.mark.parametrize(
    ("temperature", "top_k"),
    [
        (0.0, None),
        (0.7, None),
        (0.7, 0),
        (0.7, 1),
        (0.7, 50),
        (0.7, 16),
    ],
)
@pytest.mark.parametrize("prefill_mode", ["shared", "repeated"])
def test_cross_prefix_same_mode_determinism_sampling_matrix(
    token_rows: list[list[int]],
    seed_rows: list[list[int]],
    temperature: float,
    top_k: int | None,
    prefill_mode: str,
) -> None:
    def run_once() -> list[list[list[int]]]:
        model = _RecordingCrossPrefixModel(vocab_size=16)
        return cross_prefix_batched_generate_seeds(
            base_model=_base_with_hf_model(model),
            token_id_rows=token_rows,
            seed_rows=seed_rows,
            max_new_tokens=3,
            temperature=temperature,
            top_k=top_k,
            stop_token_ids=set(),
            prefill_mode=prefill_mode,
        )

    first = run_once()
    second = run_once()
    assert first == second
    assert [len(rows) for rows in first] == [len(seeds) for seeds in seed_rows]
    assert all(len(continuation) == 3 for prefix_rows in first for continuation in prefix_rows)
    assert all(
        isinstance(token, int)
        for prefix_rows in first
        for continuation in prefix_rows
        for token in continuation
    )


@pytest.mark.parametrize("prefill_mode", ["shared", "repeated"])
def test_cross_prefix_stopped_rows_stop_consuming_rng(
    monkeypatch: Any,
    prefill_mode: str,
) -> None:
    class _ScheduledStopModel(_RecordingCrossPrefixModel):
        def __init__(self) -> None:
            super().__init__(vocab_size=16)
            self.decode_forwards = 0

        def __call__(
            self,
            input_ids: t.Tensor,
            *,
            past_key_values: object = None,
            use_cache: bool = False,
            **kwargs: Any,
        ) -> _MockModelOutput:
            output = super().__call__(
                input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            output.logits[..., 1] = 1e9
            if past_key_values is not None:
                row_to_stop = self.decode_forwards
                if row_to_stop < input_ids.shape[0]:
                    output.logits[row_to_stop, -1, 1] = 0.0
                    output.logits[row_to_stop, -1, 15] = 1e9
                self.decode_forwards += 1
            return output

    observed_generators: list[t.Generator | None] = []
    real_multinomial = t.multinomial

    def recording_multinomial(
        probs: t.Tensor,
        num_samples: int,
        *,
        generator: t.Generator | None = None,
        **kwargs: Any,
    ) -> t.Tensor:
        observed_generators.append(generator)
        return real_multinomial(
            probs,
            num_samples,
            generator=generator,
            **kwargs,
        )

    monkeypatch.setattr(t, "multinomial", recording_multinomial)
    model = _ScheduledStopModel()
    output = cross_prefix_batched_generate_seeds(
        base_model=_base_with_hf_model(model),
        token_id_rows=[[1, 2, 3], [4]],
        seed_rows=[[7, 11], [13, 17]],
        max_new_tokens=6,
        temperature=0.7,
        top_k=1,
        stop_token_ids={15},
        prefill_mode=prefill_mode,
    )

    flat = [continuation for rows in output for continuation in rows]
    assert [len(continuation) for continuation in flat] == [1, 2, 3, 4]
    assert all(15 not in continuation for continuation in flat)
    assert len(observed_generators) == 14
    generator_counts: dict[int, int] = {}
    for generator in observed_generators:
        assert generator is not None
        generator_counts[id(generator)] = generator_counts.get(id(generator), 0) + 1
    assert sorted(generator_counts.values()) == [2, 3, 4, 5]
    assert model.decode_forwards == 4


def test_batch_shape_sensitive_model_proves_modes_are_independently_deterministic():
    def run(mode: str) -> list[list[list[int]]]:
        model = _RecordingCrossPrefixModel(
            vocab_size=16,
            prefill_token_from_batch_size=True,
        )
        return cross_prefix_batched_generate_seeds(
            base_model=_base_with_hf_model(model),
            token_id_rows=[[1, 2, 3], [4]],
            seed_rows=[[7, 11], [13, 17]],
            max_new_tokens=1,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
            prefill_mode=mode,
        )

    shared_first = run("shared")
    shared_second = run("shared")
    repeated_first = run("repeated")
    repeated_second = run("repeated")

    assert shared_first == shared_second == [[[2], [2]], [[2], [2]]]
    assert repeated_first == repeated_second == [[[4], [4]], [[4], [4]]]
    assert shared_first != repeated_first


@pytest.mark.parametrize(
    ("token_rows", "seed_rows", "temperature"),
    [
        ([[2, 3, 4]], [[7]], 0.0),
        ([[2, 3, 4]], [[7, 11, 13, 17, 19]], 0.7),
        (
            [[2, 3, 4], [2, 5], [2, 6, 7, 8], [2, 9]],
            [[7], [11], [13], [17]],
            0.0,
        ),
        (
            [[2, 3, 4], [2, 5], [2, 6, 7, 8], [2, 9]],
            [[7], [11, 13, 17, 19, 23], [29, 31], [37, 41, 43]],
            0.7,
        ),
    ],
)
def test_tiny_real_gemma2_public_cache_shared_prefill(
    token_rows: list[list[int]],
    seed_rows: list[list[int]],
    temperature: float,
) -> None:
    class _RecordingGemma2(t.nn.Module):
        def __init__(self, inner: Gemma2ForCausalLM) -> None:
            super().__init__()
            self.inner = inner
            self.config = inner.config
            self.prefill_caches: list[object] = []
            self.calls: list[tuple[tuple[int, ...], bool]] = []

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            input_ids = kwargs["input_ids"]
            is_prefill = kwargs.get("past_key_values") is None
            self.calls.append((tuple(input_ids.shape), is_prefill))
            output = self.inner(*args, **kwargs)
            if is_prefill:
                self.prefill_caches.append(output.past_key_values)
            return output

    config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    assert config.layer_types == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    t.manual_seed(0)
    model = _RecordingGemma2(Gemma2ForCausalLM(config).eval())
    kwargs = {
        "base_model": _base_with_hf_model(model),
        "token_id_rows": token_rows,
        "seed_rows": seed_rows,
        "max_new_tokens": 3,
        "temperature": temperature,
        "top_k": 20,
        "stop_token_ids": set(),
        "prefill_mode": "shared",
    }
    first = cross_prefix_batched_generate_seeds(**kwargs)
    second = cross_prefix_batched_generate_seeds(**kwargs)

    assert first == second
    assert [len(rows) for rows in first] == [len(seeds) for seeds in seed_rows]
    decode_rows = sum(len(seeds) for seeds in seed_rows)
    prefill_calls = [shape for shape, is_prefill in model.calls if is_prefill]
    assert prefill_calls == [
        (len(token_rows), max(map(len, token_rows))),
        (len(token_rows), max(map(len, token_rows))),
    ]
    assert len(model.prefill_caches) == 2
    for cache in model.prefill_caches:
        assert type(cache).__module__ == "transformers.cache_utils"
        assert callable(getattr(cache, "batch_select_indices", None))
        layers = cast(Any, cache).layers
        assert len(layers) == 4
        for layer in layers:
            assert layer.keys is not None
            assert layer.values is not None
            assert int(layer.keys.shape[0]) == decode_rows
            assert int(layer.values.shape[0]) == decode_rows


def test_cross_prefix_prefill_profile_regions_and_aggregate_counters(
    tmp_path: Any,
) -> None:
    with AxBenchProfiler(
        output_dir=tmp_path,
        synchronize_cuda=False,
        gpu_utilization=False,
    ):
        for mode in ("shared", "repeated"):
            model = _RecordingCrossPrefixModel()
            cross_prefix_batched_generate_seeds(
                base_model=_base_with_hf_model(model),
                token_id_rows=[[1, 2, 3], [4]],
                seed_rows=[[7, 11], [13]],
                max_new_tokens=1,
                temperature=0.0,
                top_k=None,
                stop_token_ids=set(),
                prefill_mode=mode,
            )

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    prefill_regions = [
        event
        for event in events
        if event.get("name") == "stage0c.cross_prefix_batched_generate.prefill"
    ]
    assert [event["attrs"] for event in prefill_regions] == [
        {
            "avoided_prefill_rows": 1,
            "decode_rows": 3,
            "max_prompt_len": 3,
            "prefill_mode": "shared",
            "prefixes": 2,
            "rows": 2,
        },
        {
            "avoided_prefill_rows": 0,
            "decode_rows": 3,
            "max_prompt_len": 3,
            "prefill_mode": "repeated",
            "prefixes": 2,
            "rows": 3,
        },
    ]
    expansion_regions = [
        event
        for event in events
        if event.get("name") == "stage0c.cross_prefix_batched_generate.expand_prefill_state"
    ]
    assert len(expansion_regions) == 1
    assert expansion_regions[0]["attrs"] == {
        "prefill_mode": "shared",
        "prefixes": 2,
        "rows": 3,
        "source_rows": 2,
    }

    expected_counter_values = {
        "prefill_rows": [2.0, 3.0],
        "decode_rows": [3.0, 3.0],
        "avoided_prefill_rows": [1.0, 0.0],
        "expansion_calls": [1.0, 0.0],
    }
    for suffix, values in expected_counter_values.items():
        matching = [
            event
            for event in events
            if event.get("name") == f"stage0c.cross_prefix_batched_generate.{suffix}"
        ]
        assert [event["value"] for event in matching] == values
        assert [event["attrs"]["prefill_mode"] for event in matching] == [
            "shared",
            "repeated",
        ]


def test_cross_prefix_matches_per_prefix_batched_with_mock_model():
    token_rows = [[1, 2, 3], [4], [5, 6]]
    seed_rows = [[7, 11], [13, 17], [19, 23]]
    expected = [
        batched_generate_seeds(
            base_model=MockBaseGemma({}),
            token_ids=token_ids,
            seeds=seeds,
            max_new_tokens=4,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
        )
        for token_ids, seeds in zip(token_rows, seed_rows, strict=True)
    ]
    actual = cross_prefix_batched_generate_seeds(
        base_model=MockBaseGemma({}),
        token_id_rows=token_rows,
        seed_rows=seed_rows,
        max_new_tokens=4,
        temperature=0.7,
        top_k=50,
        stop_token_ids=set(),
        prefill_mode="repeated",
    )
    assert actual == expected


def test_cross_prefix_left_padding_attention_mask_and_position_ids():
    invocations: list[dict[str, Any]] = []

    class _RecordingModel(MockHFGenerateModel):
        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            invocations.append({
                "input_ids": input_ids.detach().cpu().clone(),
                "attention_mask": kwargs["attention_mask"].detach().cpu().clone(),
                "position_ids": kwargs["position_ids"].detach().cpu().clone(),
                "logits_to_keep": kwargs.get("logits_to_keep"),
                "past_key_values": past_key_values is not None,
                "use_cache": use_cache,
            })
            batch, seq = input_ids.shape
            logits_to_keep = kwargs.get("logits_to_keep", 0)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                seq = min(seq, int(logits_to_keep))
            logits = t.zeros(batch, seq, self._vocab_size)
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _RecordingModel({})

    cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=[[1, 2, 3], [4]],
        seed_rows=[[7], [11]],
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        stop_token_ids=set(),
        prefill_mode="repeated",
    )

    assert len(invocations) == 2
    prefill = invocations[0]
    assert t.equal(
        prefill["input_ids"],
        t.tensor([[1, 2, 3], [0, 0, 4]], dtype=t.long),
    )
    assert t.equal(
        prefill["attention_mask"],
        t.tensor([[1, 1, 1], [0, 0, 1]], dtype=t.long),
    )
    assert t.equal(
        prefill["position_ids"],
        t.tensor([[0, 1, 2], [0, 0, 0]], dtype=t.long),
    )
    assert prefill["past_key_values"] is False
    assert prefill["use_cache"] is True
    assert prefill["logits_to_keep"] == 1

    decode = invocations[1]
    assert t.equal(decode["input_ids"], t.tensor([[0], [0]], dtype=t.long))
    assert t.equal(
        decode["attention_mask"],
        t.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=t.long),
    )
    assert t.equal(decode["position_ids"], t.tensor([[3], [1]], dtype=t.long))
    assert decode["past_key_values"] is True
    assert decode["use_cache"] is True
    assert decode["logits_to_keep"] is None


def test_batched_generate_prefill_uses_logits_to_keep():
    invocations: list[dict[str, Any]] = []

    class _RecordingModel(MockHFGenerateModel):
        def __init__(self):
            super().__init__({}, vocab_size=8)

        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            invocations.append({
                "shape": tuple(input_ids.shape),
                "logits_to_keep": kwargs.get("logits_to_keep"),
                "past_key_values": past_key_values is not None,
            })
            batch, seq = input_ids.shape
            logits_to_keep = kwargs.get("logits_to_keep", 0)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                seq = min(seq, int(logits_to_keep))
            logits = t.zeros(batch, seq, self._vocab_size)
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _RecordingModel()

    out = batched_generate_seeds(
        base_model=base_model,
        token_ids=[1, 2, 3],
        seeds=[7, 11],
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        stop_token_ids=set(),
    )

    assert out == [[0], [0]]
    assert invocations[0] == {
        "shape": (2, 3),
        "logits_to_keep": 1,
        "past_key_values": False,
    }
    assert invocations[1]["shape"] == (2, 1)
    assert invocations[1]["logits_to_keep"] is None
    assert invocations[1]["past_key_values"] is True


def test_cross_prefix_stop_token_per_flattened_row():
    class _RowZeroStopsAtStep1(MockHFGenerateModel):
        def __init__(self):
            super().__init__({})
            self._step = 0

        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            if self._step == 1:
                logits[0, -1, 88] = 1e9
            self._step += 1
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _RowZeroStopsAtStep1()

    out = cross_prefix_batched_generate_seeds(
        base_model=base_model,
        token_id_rows=[[1, 2, 3], [4]],
        seed_rows=[[7], [11]],
        max_new_tokens=5,
        temperature=0.0,
        top_k=None,
        stop_token_ids={88},
        prefill_mode="repeated",
    )
    assert len(out[0][0]) == 1
    assert 88 not in out[0][0]
    assert len(out[1][0]) == 5
    assert 88 not in out[1][0]


def test_batched_generate_seeds_zero_max_tokens_returns_empty():
    out = batched_generate_seeds(
        base_model=object(),
        token_ids=[1, 2, 3],
        seeds=[7, 11, 13],
        max_new_tokens=0,
        temperature=0.7,
        top_k=50,
        stop_token_ids=set(),
    )
    assert out == [[], [], []]


def test_batched_generate_seeds_seed_count_must_match():
    """Passing an empty seed list with a positive max_new_tokens raises."""
    with pytest.raises(AxBenchBaseError, match="at least one seed"):
        batched_generate_seeds(
            base_model=object(),
            token_ids=[1, 2, 3],
            seeds=[],
            max_new_tokens=4,
            temperature=0.7,
            top_k=50,
            stop_token_ids=set(),
        )


def test_batched_generate_seeds_returns_correct_shapes(mock_base_gemma_for_generation):
    """Returned list has len == n_seeds; each inner list is at most max_new_tokens long.

    Different seeds with random sampling on a 256k vocab → rows are pairwise different
    with probability ~ 1 - 1/256000² (effectively certain).
    """
    seed_to_outputs = {7: [10, 11, 12], 11: [20, 21], 13: [30]}
    base_model = mock_base_gemma_for_generation(seed_to_outputs)

    out = batched_generate_seeds(
        base_model=base_model,
        token_ids=[1, 2, 3],
        seeds=[7, 11, 13],
        max_new_tokens=8,
        temperature=0.7,
        top_k=50,
        stop_token_ids=set(),
    )
    assert len(out) == 3
    for inner in out:
        assert len(inner) <= 8
        assert len(inner) > 0, "kernel produced empty continuation despite max_new_tokens > 0"
        for tok in inner:
            assert isinstance(tok, int)
            assert 0 <= tok < 256_000, f"token {tok} out of vocab range"
    # Distinct seeds + uniform-from-zero-logits sampling → rows differ with overwhelming probability
    assert not (out[0] == out[1] == out[2]), (
        f"All rows identical despite distinct seeds — generator isolation broken: {out}"
    )


def test_batched_generate_seeds_per_row_rng_isolation(monkeypatch):
    """Each row's multinomial draw consumes from its own Generator."""
    observed_generators: list[object] = []
    real_multinomial = t.multinomial

    def fake_multinomial(probs, num_samples, *, generator=None, **kwargs):
        observed_generators.append(generator)
        return real_multinomial(probs, num_samples, generator=generator, **kwargs)

    monkeypatch.setattr(t, "multinomial", fake_multinomial)

    base_model = MockBaseGemma({})  # empty seed map; we won't call generate()

    batched_generate_seeds(
        base_model=base_model,
        token_ids=[1, 2, 3],
        seeds=[7, 11, 13],
        max_new_tokens=2,
        temperature=0.7,
        top_k=50,
        stop_token_ids=set(),
    )

    # Each step of the sample loop uses N multinomial calls (one per active row).
    # For 2 steps × 3 rows: 6 multinomial calls expected.
    assert len(observed_generators) == 6
    first_step = observed_generators[:3]
    second_step = observed_generators[3:]
    # Three distinct Generator instances per step
    assert len({id(g) for g in first_step}) == 3
    # Same instances reused on second step
    assert [id(g) for g in first_step] == [id(g) for g in second_step]


def test_cross_prefix_batched_generate_per_row_rng_isolation(monkeypatch: Any):
    """Flattened cross-prefix rows each keep their own Generator."""
    observed_generators: list[object] = []
    real_multinomial = t.multinomial

    def fake_multinomial(
        probs: t.Tensor,
        num_samples: int,
        *,
        generator: t.Generator | None = None,
        **kwargs: Any,
    ) -> t.Tensor:
        observed_generators.append(generator)
        return real_multinomial(probs, num_samples, generator=generator, **kwargs)

    monkeypatch.setattr(t, "multinomial", fake_multinomial)

    cross_prefix_batched_generate_seeds(
        base_model=MockBaseGemma({}),
        token_id_rows=[[1, 2, 3], [4], [5, 6]],
        seed_rows=[[7], [11, 13], [17, 19, 23]],
        max_new_tokens=2,
        temperature=0.7,
        top_k=50,
        stop_token_ids=set(),
        prefill_mode="repeated",
    )

    assert len(observed_generators) == 12
    first_step = observed_generators[:6]
    second_step = observed_generators[6:]
    assert len({id(g) for g in first_step}) == 6
    assert [id(g) for g in first_step] == [id(g) for g in second_step]


def test_batched_generate_seeds_distinct_seeds_produce_distinct_first_tokens():
    """Distinct seeds must produce distinct first tokens (per-row generators
    are seeded individually, not all with seeds[0]).

    P(all-3-identical) ≤ 1/V² ≈ 1.5e-11 for V=256000. Effectively impossible.
    """
    base_model = MockBaseGemma({})
    out = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[7, 11, 13],
        max_new_tokens=1, temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    # First tokens should differ
    assert not (out[0] == out[1] == out[2]), (
        f"Identical first tokens from distinct seeds → generators not seeded individually: {out}"
    )


def test_batched_generate_seeds_temperature_zero_deterministic():
    """temperature=0 -> all rows produce identical (greedy) tokens."""
    base_model = MockBaseGemma({})  # forward returns zero logits

    out = batched_generate_seeds(
        base_model=base_model,
        token_ids=[1, 2, 3],
        seeds=[7, 11, 13],
        max_new_tokens=4,
        temperature=0.0,
        top_k=None,
        stop_token_ids=set(),
    )
    # All rows greedy-pick from zero logits -> all identical
    assert out[0] == out[1] == out[2]


def test_batched_generate_seeds_top_k_zero_means_disabled():
    """top_k=0 and top_k=None should produce identical samples.

    Note: kernel uses per-row Generator instances, so global rng state is
    irrelevant. Both calls use seeds=[42] which seeds the per-row generator.
    """
    base_model = MockBaseGemma({})

    out_zero = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2], seeds=[42],
        max_new_tokens=3, temperature=1.0, top_k=0, stop_token_ids=set(),
    )
    out_none = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2], seeds=[42],
        max_new_tokens=3, temperature=1.0, top_k=None, stop_token_ids=set(),
    )
    assert out_zero == out_none


def test_batched_generate_seeds_stop_token_per_row_truncation():
    """When a stop token is sampled, that row terminates and trailing
    tokens are not in the output."""

    class _StopForcingModel(MockHFGenerateModel):
        def __init__(self):
            super().__init__({})
            self._step = 0

        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            # Force row 0 to pick token 99 on step 1
            if self._step == 1:
                logits[0, -1, 99] = 1e9
            self._step += 1
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _StopForcingModel()

    out = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[7, 11], max_new_tokens=5, temperature=1.0, top_k=None,
        stop_token_ids={99},
    )
    # Row 0: step 0 gives some token, step 1 would be 99 (stop) -> not in output
    assert len(out[0]) == 1
    assert 99 not in out[0]
    # Row 1 was unaffected — step 1 logits for row 1 are all-zero, so it
    # samples uniformly. It should NOT have hit the stop token (chance is
    # 1/vocab_size); at most it has 5 tokens.
    assert len(out[1]) <= 5
    assert 99 not in out[1]


def test_batched_generate_upcasts_logits_to_float32():
    """Production runs Gemma in bf16 but HF generate upcasts logits to fp32
    before sampling. Verify our kernel does the same upcast — otherwise
    bf16 rounding would silently diverge from sequential per-seed output."""
    captured_dtypes: list[t.dtype] = []

    class _BF16ForwardModel(MockHFGenerateModel):
        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            batch, seq = input_ids.shape
            # Return bf16 logits to simulate production
            logits = t.zeros(batch, seq, self._vocab_size, dtype=t.bfloat16)
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _BF16ForwardModel({})

    # Capture the dtype passed to multinomial
    real_multinomial = t.multinomial

    def capture_multinomial(probs, num_samples, *, generator=None, **kwargs):
        captured_dtypes.append(probs.dtype)
        return real_multinomial(probs, num_samples, generator=generator, **kwargs)

    with um.patch.object(t, "multinomial", capture_multinomial):
        batched_generate_seeds(
            base_model=base_model, token_ids=[1, 2], seeds=[42],
            max_new_tokens=2, temperature=0.7, top_k=50, stop_token_ids=set(),
        )

    # All multinomial calls should receive fp32 probs (after softmax of fp32 logits)
    assert all(dt == t.float32 for dt in captured_dtypes), (
        f"multinomial received non-fp32 probs: {captured_dtypes}. "
        f"Kernel must upcast bf16 logits to fp32 before sampling."
    )


def test_apply_top_k_masks_below_threshold():
    """_apply_top_k masks all but the top-k logits to -inf per row."""
    logits = t.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = _apply_top_k(logits, top_k=2)
    # Top 2 are 5.0 (idx 1) and 4.0 (idx 3); rest must be -inf
    assert t.isinf(out[0, 0]) and out[0, 0] < 0
    assert out[0, 1].item() == 5.0
    assert t.isinf(out[0, 2]) and out[0, 2] < 0
    assert out[0, 3].item() == 4.0
    assert t.isinf(out[0, 4]) and out[0, 4] < 0


def test_apply_top_k_passthrough_when_disabled():
    """_apply_top_k returns input unchanged when top_k is 0 or >= vocab size."""
    logits = t.tensor([[1.0, 5.0, 3.0]])
    # top_k=0 → passthrough
    assert t.equal(_apply_top_k(logits, top_k=0), logits)
    # top_k=-1 → passthrough
    assert t.equal(_apply_top_k(logits, top_k=-1), logits)
    # top_k>=vocab → passthrough
    assert t.equal(_apply_top_k(logits, top_k=3), logits)
    assert t.equal(_apply_top_k(logits, top_k=100), logits)


def test_top_k_one_forces_argmax():
    """top_k=1 with high logit on token 42 forces all rows to sample 42."""
    base_model = MockBaseGemma({})

    class _ArgmaxForcedModel(MockHFGenerateModel):
        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            logits[..., 42] = 100.0  # argmax always 42
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model.nnterp_model._model = _ArgmaxForcedModel({})

    out = batched_generate_seeds(
        base_model=base_model, token_ids=[1], seeds=[7, 11, 13],
        max_new_tokens=3, temperature=1.0, top_k=1, stop_token_ids=set(),
    )
    # top_k=1 + extreme logit forces argmax even with sampling
    assert out == [[42, 42, 42], [42, 42, 42], [42, 42, 42]]


def test_hf_generation_model_returns_standardized_transformer_model():
    """The generation model is the real StandardizedTransformer ``_model``."""
    fake_inner = MockHFGenerateModel({})

    class _StandardizedTransformerShape:
        _model = fake_inner

    class _BaseModel:
        nnterp_model = _StandardizedTransformerShape()

    assert _hf_generation_model(_BaseModel()) is fake_inner


def test_hf_generation_model_raises_when_model_is_none():
    """A wrapper without its underlying Hugging Face model fails loudly."""
    class _BadWrapper:
        _model = None

    class _BaseModel:
        nnterp_model = _BadWrapper()

    with pytest.raises(AxBenchBaseError, match="StandardizedTransformer _model"):
        _hf_generation_model(_BaseModel())


def test_module_device_falls_back_to_cpu_when_no_parameters_attr():
    """_module_device returns CPU when model has no parameters attribute."""
    class _NoParams:
        pass

    assert _module_device(_NoParams()) == t.device("cpu")


def test_module_device_falls_back_to_cpu_on_stop_iteration():
    """_module_device returns CPU when parameters() yields nothing."""
    class _EmptyParams:
        def parameters(self):
            return iter([])

    assert _module_device(_EmptyParams()) == t.device("cpu")


def test_batched_generate_seeds_all_rows_stop_at_step_zero():
    """When all rows hit stop on step 0, return empty continuations and break early."""

    class _AlwaysStopModel(MockHFGenerateModel):
        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            logits[..., 99] = 1e9  # Always pick token 99
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _AlwaysStopModel({})

    out = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[7, 11, 13], max_new_tokens=10, temperature=1.0, top_k=None,
        stop_token_ids={99},
    )
    # All rows stopped before any token added
    assert out == [[], [], []]


# -------- Path coverage gaps --------


def test_hf_generation_model_raises_when_nnterp_model_is_none():
    """base_model.nnterp_model = None must raise with clear message
    (not crash with AttributeError on the nested getattr)."""
    class _NoNnterp:
        nnterp_model = None

    with pytest.raises(AxBenchBaseError, match="StandardizedTransformer _model"):
        _hf_generation_model(_NoNnterp())


def test_hf_generation_model_raises_when_no_nnterp_attr_at_all():
    """base_model with no nnterp_model attribute at all also raises clearly."""
    class _BareBase:
        pass

    with pytest.raises(AxBenchBaseError, match="StandardizedTransformer _model"):
        _hf_generation_model(_BareBase())


def test_module_device_returns_first_param_device():
    """When model has at least one parameter, _module_device returns its device."""
    class _ModelWithParams:
        def parameters(self):
            yield t.zeros(3, 4, device="cpu")

    assert _module_device(_ModelWithParams()) == t.device("cpu")


def test_apply_top_k_keeps_tied_values_at_threshold():
    """Ties at the kth-largest position are KEPT (not masked).
    Matches HF TopKLogitsWarper semantics."""
    # Logits with explicit ties at the top-k boundary.
    # top_k=2: top 2 are 5.0 (idx 1) and 5.0 (idx 3) — both should be kept.
    logits = t.tensor([[1.0, 5.0, 3.0, 5.0, 2.0]])
    out = _apply_top_k(logits, top_k=2)
    # Both 5.0s preserved
    assert out[0, 1].item() == 5.0
    assert out[0, 3].item() == 5.0
    # Strictly-below entries masked
    assert t.isinf(out[0, 0]) and out[0, 0] < 0
    assert t.isinf(out[0, 2]) and out[0, 2] < 0
    assert t.isinf(out[0, 4]) and out[0, 4] < 0


def test_apply_top_k_negative_logits_handled():
    """top-k masking works with negative logits (no overflow assumptions)."""
    logits = t.tensor([[-100.0, -1.0, -50.0, -2.0, -75.0]])
    out = _apply_top_k(logits, top_k=2)
    # Top 2 are -1.0 (idx 1) and -2.0 (idx 3)
    assert out[0, 1].item() == -1.0
    assert out[0, 3].item() == -2.0
    # Rest masked to -inf (was already negative; verify it's now -inf, not unchanged)
    assert t.isinf(out[0, 0]) and out[0, 0] < 0
    assert t.isinf(out[0, 2]) and out[0, 2] < 0
    assert t.isinf(out[0, 4]) and out[0, 4] < 0


def test_skip_final_forward_after_all_rows_done():
    """When all rows hit stop tokens mid-loop, no further forward calls
    are issued (production code's `if not done_mask.all()` guard at line 209)."""
    call_count = [0]

    class _CountingModel(MockHFGenerateModel):
        def __init__(self):
            super().__init__({})
            self._step = 0

        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            call_count[0] += 1
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            # Force ALL rows to pick stop token 99 at step 1 (after step 0)
            if self._step >= 1:
                logits[:, -1, 99] = 1e9
            self._step += 1
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _CountingModel()

    out = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[7, 11, 13], max_new_tokens=10, temperature=1.0, top_k=None,
        stop_token_ids={99},
    )

    # Trace:
    #   - prefill (call 1): _step=0 → zero logits, _step=1 after
    #   - step=0 in loop: sample uniform (no stop), append, advance forward
    #     (call 2): _step=1 → forces 99, _step=2 after
    #   - step=1: sample → all rows pick 99, all marked done.
    #     done_mask.all() is True → skip the final forward.
    #   - step=2: done_mask.all() → break.
    # Total: 2 forward calls.
    assert call_count[0] == 2, (
        f"Expected exactly 2 forward calls (prefill + 1 advance) when all "
        f"rows stop after step 1; got {call_count[0]}"
    )
    # Each row produced exactly 1 token (step 0's output, then step 1 stopped)
    for inner in out:
        assert len(inner) == 1, f"Expected 1 token before stop; got {len(inner)}"


# -------- Determinism guards --------


def test_batched_generate_deterministic_across_calls():
    """Two calls with identical arguments produce byte-identical output."""
    base_model_1 = MockBaseGemma({})
    base_model_2 = MockBaseGemma({})

    out_1 = batched_generate_seeds(
        base_model=base_model_1, token_ids=[1, 2, 3],
        seeds=[42, 100, 200], max_new_tokens=4,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    out_2 = batched_generate_seeds(
        base_model=base_model_2, token_ids=[1, 2, 3],
        seeds=[42, 100, 200], max_new_tokens=4,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    assert out_1 == out_2, (
        f"Same args → different output:\n  call 1: {out_1}\n  call 2: {out_2}"
    )


def test_batched_generate_seed_order_independence():
    """A given seed's per-row output is independent of where it appears in
    the seeds list (per-row generators don't share state across rows)."""
    seeds_a = [42, 100, 200]
    seeds_b = [200, 42, 100]  # same set, different order

    out_a = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=seeds_a, max_new_tokens=3,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    out_b = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=seeds_b, max_new_tokens=3,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    # Map each seed to its output in both calls; they should match
    map_a = dict(zip(seeds_a, out_a, strict=True))
    map_b = dict(zip(seeds_b, out_b, strict=True))
    for s in seeds_a:
        assert map_a[s] == map_b[s], (
            f"Seed {s} produced different output depending on its position: "
            f"order A → {map_a[s]}, order B → {map_b[s]}"
        )


def test_batched_generate_does_not_advance_global_rng():
    """The kernel uses per-row torch.Generator, so it should NOT advance the
    global rng. Two consecutive calls with the same seeds, separated by
    arbitrary global rng draws, must produce identical output."""
    base_model = MockBaseGemma({})

    # First call
    t.manual_seed(0)
    out_1 = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[42, 100], max_new_tokens=3,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )

    # Advance global rng arbitrarily
    _ = t.rand(100)

    # Second call with same seeds — must be identical despite global rng changes
    t.manual_seed(999)  # change global rng entirely
    out_2 = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[42, 100], max_new_tokens=3,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )

    assert out_1 == out_2, (
        f"Kernel output depends on global rng state — per-row generators "
        f"are not isolated:\n  call 1: {out_1}\n  call 2: {out_2}"
    )


# -------- Boundary cases --------


def test_batched_generate_single_seed_works():
    """n_rows=1 is a valid degenerate case — should work without dimension errors."""
    out = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=[42], max_new_tokens=4,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    assert len(out) == 1
    assert len(out[0]) <= 4
    for tok in out[0]:
        assert isinstance(tok, int)


def test_batched_generate_large_batch_50_rows():
    """50-row batch must work without exponential slowdown or shape errors."""
    seeds = list(range(1000, 1050))
    out = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=seeds, max_new_tokens=4,
        temperature=0.7, top_k=50, stop_token_ids=set(),
    )
    assert len(out) == 50
    for inner in out:
        assert len(inner) <= 4


def test_batched_generate_stop_token_never_sampled():
    """Stop tokens with vanishingly small probability never trigger early
    termination — all rows produce max_new_tokens."""
    # vocab_size=256k, uniform sampling from zero logits — chance of hitting
    # any specific token id (e.g. 255_999) is 1/256_000, so 5 steps × 3 rows
    # has total stop probability ≈ 15/256_000 ≈ 6e-5. In practice we observe
    # all rows at full length.
    out = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=[7, 11, 13], max_new_tokens=5,
        temperature=0.7, top_k=50,
        stop_token_ids={255_999},  # extremely unlikely under uniform-from-zero-logits
    )
    # All rows should produce exactly 5 tokens (stop never triggers)
    for inner in out:
        assert len(inner) == 5, (
            f"Expected 5 tokens (stop never sampled); got {len(inner)}: {inner}"
        )


def test_batched_generate_extreme_high_temperature_no_overflow():
    """Very high temperature (1000.0) must not produce NaN or crash."""
    out = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=[7, 11, 13], max_new_tokens=3,
        temperature=1000.0, top_k=50, stop_token_ids=set(),
    )
    assert len(out) == 3
    for inner in out:
        assert len(inner) == 3
        for tok in inner:
            assert 0 <= tok < 256_000


def test_batched_generate_low_temperature_does_not_crash():
    """Very low temperature (0.0001) must produce stable output (numerical
    stability of softmax with large pre-softmax magnitudes)."""
    out = batched_generate_seeds(
        base_model=MockBaseGemma({}), token_ids=[1, 2, 3],
        seeds=[7, 11, 13], max_new_tokens=3,
        temperature=0.0001, top_k=50, stop_token_ids=set(),
    )
    # All rows should produce 3 valid tokens
    for inner in out:
        assert len(inner) == 3
        for tok in inner:
            assert isinstance(tok, int)
            assert 0 <= tok < 256_000


def test_batched_generate_makes_one_prefill_plus_one_per_step():
    """Pin the kernel's call pattern. For ``max_new_tokens=4`` with no stops:

    - prefill (line 156): always 1 call with shape (N, prompt_len)
    - per-step (line 215): runs at the end of each iteration unless all
      rows are done. With no stops, this fires for every step in
      ``range(max_new_tokens)``.

    Total: ``1 + max_new_tokens`` forward calls.
    """
    invocations: list[dict] = []

    class _RecordingModel(MockHFGenerateModel):
        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            invocations.append({
                "input_shape": tuple(input_ids.shape),
                "past_key_values": past_key_values is not None,
                "use_cache": use_cache,
            })
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _RecordingModel({})

    seeds = [7, 11, 13]
    max_tokens = 4
    batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3, 4],
        seeds=seeds, max_new_tokens=max_tokens,
        temperature=1.0, top_k=None, stop_token_ids=set(),
    )

    expected_call_count = 1 + max_tokens  # prefill + one advance per step
    assert len(invocations) == expected_call_count, (
        f"Expected {expected_call_count} forward calls (1 prefill + "
        f"{max_tokens} advances) for max_new_tokens={max_tokens} with no "
        f"stops; got {len(invocations)}. Calls: {invocations}"
    )
    # Prefill: shape (N, prompt_len) — no past_key_values, use_cache=True
    assert invocations[0]["input_shape"] == (3, 4)
    assert invocations[0]["past_key_values"] is False
    assert invocations[0]["use_cache"] is True
    # Subsequent: shape (N, 1) with KV cache
    for inv in invocations[1:]:
        assert inv["input_shape"] == (3, 1)
        assert inv["past_key_values"] is True
        assert inv["use_cache"] is True


def test_batched_generate_done_rows_filled_with_placeholder():
    """When row 0 hits stop, its placeholder token (0) is fed to the next
    forward, but never appended to its output. This test indirectly verifies
    the placeholder logic by checking that row 0's output stops AT the stop
    token while row 1 continues normally."""
    class _RowZeroStopsAtStep1(MockHFGenerateModel):
        def __init__(self):
            super().__init__({})
            self._step = 0

        def __call__(self, input_ids, *, past_key_values=None,
                     use_cache=False, **kwargs):
            batch, seq = input_ids.shape
            logits = t.zeros(batch, seq, self._vocab_size)
            if self._step == 1:
                # Row 0 picks stop token 88 at step 1
                logits[0, -1, 88] = 1e9
            self._step += 1
            return _MockModelOutput(
                logits=logits,
                past_key_values=_MockKVCache() if use_cache else None,
            )

    base_model = MockBaseGemma({})
    base_model.nnterp_model._model = _RowZeroStopsAtStep1()

    out = batched_generate_seeds(
        base_model=base_model, token_ids=[1, 2, 3],
        seeds=[7, 11], max_new_tokens=5, temperature=1.0, top_k=None,
        stop_token_ids={88},
    )
    # Row 0: 1 token (step 0), then stops at step 1 → no token from step 1
    assert len(out[0]) == 1
    assert 88 not in out[0]
    # Row 1: never hits stop, gets all 5 tokens
    assert len(out[1]) == 5
    assert 88 not in out[1]
