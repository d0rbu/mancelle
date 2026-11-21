"""Oracles for single-RNG batched generation and decode/KV compaction."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
import yaml

from nanochat import sampling as nanochat_sampling
from nanochat.sampling import sample_next_token as _real_sample_next_token
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    stack_packed_encoder_outputs,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.axbench_base.baselines.hooks import (
    register_batched_residual_hook,
    register_residual_hook,
)
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_upstream.methods import (
    build_self_conditioning_config,
    build_self_conditioning_masked_policy,
)
from steering.eval.chat_generation import empty_assistant_conversation
from steering.eval.inference_engine import (
    InferenceEngine,
    KVCache,
    SelfConditionedPrefill,
    _sample_next_token_from_finite_logits,
    build_engine_for_features,
    build_engine_self_conditioned,
    build_wrapper_for_features,
    stack_prefilled_kv_caches,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    MultiTargetSegments,
    SingleEntryContextProvider,
)
from steering.profiling.markers import TimingRecorder, use_recorder
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, token_to_char_offsets
from tests.conftest import MockTokenizer, _make_encdec, _make_gpt
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_model_with_feature_prediction,
    build_tiny_model,
)

_PROMPT_TEXT = "shared prompt"


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    ("temperature", "top_k"),
    [
        (0.0, None),
        (0.0, 20),
        (0.0, -1),
        (1.0, None),
        (0.7, 0),
        (0.7, -2),
        (0.7, 1),
        (0.7, 20),
        (0.7, 37),
    ],
)
def test_prevalidated_row_sampler_exactly_matches_canonical(
    temperature: float,
    top_k: int | None,
    tied: bool,
) -> None:
    logits = t.randn(1, 32, generator=t.Generator().manual_seed(1954))
    if tied:
        logits[:, :8] = 2.0
    pinned = logits.clone()
    if temperature == 0.0:
        expected_generator = None
        actual_generator = None
    else:
        expected_generator = t.Generator().manual_seed(913)
        actual_generator = t.Generator().manual_seed(913)

    expected = _real_sample_next_token(
        logits,
        expected_generator,
        temperature,
        top_k,
    )
    actual = _sample_next_token_from_finite_logits(
        logits,
        actual_generator,
        temperature,
        top_k,
    )

    assert t.equal(actual, expected)
    assert actual.shape == (1, 1)
    assert actual.dtype == t.int64
    assert t.equal(logits, pinned)
    if expected_generator is not None:
        assert actual_generator is not None
        assert t.equal(actual_generator.get_state(), expected_generator.get_state())


def test_prevalidated_row_sampler_preserves_nan_temperature_assertion() -> None:
    logits = t.ones(1, 4)
    generator = t.Generator().manual_seed(1)
    with pytest.raises(AssertionError, match="temperature must be non-negative"):
        _real_sample_next_token(logits, generator, float("nan"), None)
    with pytest.raises(AssertionError, match="temperature must be non-negative"):
        _sample_next_token_from_finite_logits(logits, generator, float("nan"), None)


def test_prevalidated_row_sampler_preserves_bool_top_k_type_error() -> None:
    logits = t.ones(1, 4)
    expected_generator = t.Generator().manual_seed(1)
    actual_generator = t.Generator().manual_seed(1)
    expected_state = expected_generator.get_state().clone()
    with pytest.raises(TypeError, match="must be int, not bool"):
        _real_sample_next_token(logits, expected_generator, 0.7, True)
    with pytest.raises(TypeError, match="must be int, not bool"):
        _sample_next_token_from_finite_logits(logits, actual_generator, 0.7, True)
    assert t.equal(expected_generator.get_state(), expected_state)
    assert t.equal(actual_generator.get_state(), expected_state)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_cuda_production_shape_active_guard_and_row_sampler_are_exact(
    dtype: t.dtype,
) -> None:  # pragma: no cover
    device = t.device("cuda")
    steps = 128
    batch_size = 24
    vocab_size = 32_768
    logits_bank = t.randn(
        steps,
        batch_size,
        vocab_size,
        device=device,
        dtype=dtype,
        generator=t.Generator(device=device).manual_seed(20260830),
    )
    pinned = logits_bank.clone()
    expected_generators = [
        t.Generator(device=device).manual_seed(1511_000 + row) for row in range(batch_size)
    ]
    actual_generators = [
        t.Generator(device=device).manual_seed(1511_000 + row) for row in range(batch_size)
    ]
    expected_tokens = []
    actual_tokens = []
    with t.inference_mode():
        for logits in logits_bank:
            expected_tokens.append(
                t.cat(
                    [
                        _real_sample_next_token(
                            logits[row : row + 1],
                            expected_generators[row],
                            1.0,
                            50,
                        )
                        for row in range(batch_size)
                    ]
                )
            )
            nanochat_sampling._assert_finite_logits(logits)
            actual_tokens.append(
                t.cat(
                    [
                        _sample_next_token_from_finite_logits(
                            logits[row : row + 1],
                            actual_generators[row],
                            1.0,
                            50,
                        )
                        for row in range(batch_size)
                    ]
                )
            )

    assert t.equal(t.stack(actual_tokens), t.stack(expected_tokens))
    assert t.equal(logits_bank, pinned)
    for actual, expected in zip(actual_generators, expected_generators, strict=True):
        assert t.equal(actual.get_state(), expected.get_state())


def _assert_packed_equal(
    actual: PackedEncoderOutput,
    expected: PackedEncoderOutput,
) -> None:
    assert t.equal(actual.values, expected.values)
    assert t.equal(actual.layout.flat_indices, expected.layout.flat_indices)
    assert t.equal(actual.layout.valid_mask, expected.layout.valid_mask)
    assert t.equal(actual.layout.batch_indices, expected.layout.batch_indices)
    assert t.equal(actual.layout.segment_indices, expected.layout.segment_indices)
    assert actual.layout.batch_size == expected.layout.batch_size
    assert actual.layout.n_segments == expected.layout.n_segments
    assert actual.layout.feature_count == expected.layout.feature_count


def _prompt(tokenizer: MockTokenizer) -> list[int]:
    encoded = tokenizer.encode(
        _PROMPT_TEXT,
        prepend=tokenizer.get_bos_token_id(),
    )
    assert isinstance(encoded[0], int)
    return cast(list[int], encoded)


def _greedy_sequential_samples(
    engine: InferenceEngine,
    token_ids: list[int],
    n_samples: int,
    **kwargs: Any,
) -> list[list[int]]:
    return [list(engine.generate_stream(token_ids, seed=None, **kwargs)) for _ in range(n_samples)]


class _FullPrefixContextProvider:
    """Return one real full-prefix context segment for tiny encdec tests."""

    def __init__(self, tokenizer: MockTokenizer) -> None:
        self._tokenizer = tokenizer

    def get(self, token_ids):
        normalized = [int(token) for token in token_ids]
        if not normalized:
            return None
        char_end = token_to_char_offsets(self._tokenizer, normalized)[-1]
        if char_end <= 0:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


class _PromptFeatureContextProvider:
    """Return a distinct full-prefix feature segment for each known prompt."""

    def __init__(
        self,
        tokenizer: MockTokenizer,
        features_by_prompt: dict[tuple[int, ...], tuple[int, int]],
    ) -> None:
        self._tokenizer = tokenizer
        self._features_by_prompt = features_by_prompt

    def get(self, token_ids):
        normalized = tuple(int(token) for token in token_ids)
        feature = self._features_by_prompt.get(normalized)
        if feature is None:
            return None
        char_end = token_to_char_offsets(self._tokenizer, list(normalized))[-1]
        feature_id, bucket = feature
        return (
            ContextFeatureSegment(
                feature_ids=(feature_id,),
                buckets=(bucket,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


class _ForwardRecorder:
    """Delegate to a real model while recording physical decode batch sizes."""

    def __init__(self, model: Any) -> None:
        self._wrapped = model
        self.prefill_batch_sizes: list[int] = []
        self.decode_batch_sizes: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, idx: t.Tensor, *, kv_cache) -> t.Tensor:
        batch_size = int(idx.size(0))
        if int(kv_cache.get_pos()) == 0:
            self.prefill_batch_sizes.append(batch_size)
        else:
            self.decode_batch_sizes.append(batch_size)
        return self._wrapped(idx, kv_cache=kv_cache)

    def reset(self) -> None:
        self.prefill_batch_sizes.clear()
        self.decode_batch_sizes.clear()


class _ForcedFirstThenGreedy:
    """Force one heterogeneous draw, then use the real greedy sampler."""

    def __init__(self, first_tokens: list[int]) -> None:
        self._first_tokens = first_tokens
        self.calls = 0

    def __call__(
        self,
        logits: t.Tensor,
        rng: t.Generator | None,
        temperature: float,
        top_k: int | None,
    ) -> t.Tensor:
        if self.calls == 0:
            assert logits.size(0) == len(self._first_tokens)
            result = t.tensor(
                self._first_tokens,
                dtype=t.long,
                device=logits.device,
            ).unsqueeze(1)
        else:
            result = _real_sample_next_token(logits, rng, temperature, top_k)
        self.calls += 1
        return result


class _HeterogeneousGreedyModel:
    """Batch-invariant greedy schedules with deliberately different stop steps.

    A batched instance assigns schedules after the shared first token. A
    sequential instance receives the schedule explicitly. Token transitions
    make the surviving identity recoverable after each compaction, and the
    model checks the copied KV marker before the final one-row decode.
    """

    def __init__(self, *, single_schedule: int | None = None) -> None:
        self.config = SimpleNamespace(
            sequence_len=16,
            vocab_size=6,
            n_kv_head=1,
            n_embd=4,
            n_head=1,
            n_layer=1,
        )
        self.single_schedule = single_schedule
        self.prefill_calls = 0
        self.decode_batch_sizes: list[int] = []
        self.previous_k_markers: list[list[float]] = []

    def __call__(self, idx: t.Tensor, *, kv_cache) -> t.Tensor:
        position = int(kv_cache.get_pos())
        batch_size = int(idx.size(0))
        logits = t.full(
            (batch_size, idx.size(1), self.config.vocab_size),
            -1000.0,
            dtype=t.float32,
            device=idx.device,
        )
        if position == 0:
            self.prefill_calls += 1
            logits[:, :, 1] = 0.0
        else:
            self.decode_batch_sizes.append(batch_size)
            if position > idx.size(1):
                markers = kv_cache.k_cache[0, :, position - 1, 0, 0]
                self.previous_k_markers.append(markers.tolist())
            tokens = idx[:, -1].tolist()
            if self.single_schedule is not None:
                schedules = [self.single_schedule]
            elif tokens == [1, 1, 1]:
                schedules = [0, 1, 2]
            else:
                schedules = [{2: 1, 3: 2, 4: 2}[token] for token in tokens]
            next_tokens = {0: 0, 1: 2, 2: 3}
            for row, (token, schedule) in enumerate(zip(tokens, schedules, strict=True)):
                if token == 1:
                    next_token = next_tokens[schedule]
                elif token == 2:
                    next_token = 0
                elif token == 3:
                    next_token = 4
                elif token == 4:
                    next_token = 0
                else:  # pragma: no cover - the oracle's transitions are exhaustive
                    raise AssertionError(f"unexpected decode token {token}")
                logits[row, :, next_token] = 0.0

        kv_cache.k_cache[:, :, position : position + idx.size(1)].fill_(0)
        kv_cache.v_cache[:, :, position : position + idx.size(1)].fill_(0)
        for row, token in enumerate(idx[:, -1].tolist()):
            kv_cache.k_cache[:, row, position : position + idx.size(1)].fill_(token)
            kv_cache.v_cache[:, row, position : position + idx.size(1)].fill_(-token)
        kv_cache.advance(idx.size(1))
        return logits


class _UniformLogitModel:
    def __init__(self, *, sequence_len: int = 32) -> None:
        self.config = SimpleNamespace(
            sequence_len=sequence_len,
            vocab_size=5,
            n_kv_head=1,
            n_embd=4,
            n_head=1,
            n_layer=1,
        )
        self.prefill_calls = 0
        self.decode_batch_sizes: list[int] = []

    def __call__(self, idx: t.Tensor, *, kv_cache) -> t.Tensor:
        if kv_cache.get_pos() == 0:
            self.prefill_calls += 1
        else:
            self.decode_batch_sizes.append(int(idx.size(0)))
        kv_cache.advance(idx.size(1))
        return t.zeros(
            idx.size(0),
            idx.size(1),
            self.config.vocab_size,
            dtype=t.float32,
            device=idx.device,
        )


def _uniform_engine(
    tokenizer: MockTokenizer,
    *,
    sequence_len: int = 32,
) -> tuple[InferenceEngine, _UniformLogitModel]:
    model = _UniformLogitModel(sequence_len=sequence_len)
    return InferenceEngine(model, tokenizer, t.device("cpu")), model


def _counter_totals(profile_dir: Path) -> dict[str, float]:
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    return {str(row["name"]): float(row["total"]) for row in summary["counters"]}


def _region_names(profile_dir: Path) -> set[str]:
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    return {str(row["name"]) for row in summary["regions"]}


def _timing_counts(recorder: TimingRecorder) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in recorder.rows():
        name = str(row["name"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def _self_cond_config() -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(4),
        pool_mode=FeaturePredLossMode.FULL_SEGMENT,
        selection=ThresholdSelectionConfig(
            selection=ThresholdSelection(
                threshold_type=ThresholdType.segment_topk,
                threshold=2,
            ),
            precomputed_thresholds=None,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=NUM_BUCKETS,
        n_sae_features=N_SAE_FEATURES,
    )


def test_stack_prefilled_kv_caches_preserves_every_row_and_source() -> None:
    caches = tuple(
        KVCache(
            batch_size=1,
            num_heads=2,
            seq_len=9,
            head_dim=4,
            num_layers=3,
            device=t.device("cpu"),
            dtype=t.float32,
        )
        for _ in range(3)
    )
    pinned: list[tuple[t.Tensor, t.Tensor]] = []
    for row, cache in enumerate(caches):
        cache.k_cache[:, :, :5].fill_(float(row + 1))
        cache.v_cache[:, :, :5].fill_(float(-(row + 1)))
        cache.advance(5)
        pinned.append((cache.k_cache.clone(), cache.v_cache.clone()))

    actual = stack_prefilled_kv_caches(caches)

    assert actual.batch_size == 3
    assert actual.max_seq_len == 9
    assert actual.cache_seqlens.tolist() == [5, 5, 5]
    for row in range(3):
        assert t.equal(actual.k_cache[:, row : row + 1, :5], caches[row].k_cache[:, :, :5])
        assert t.equal(actual.v_cache[:, row : row + 1, :5], caches[row].v_cache[:, :, :5])
    assert not bool(actual.k_cache[:, :, 5:].any())
    assert not bool(actual.v_cache[:, :, 5:].any())
    actual.k_cache.zero_()
    actual.v_cache.zero_()
    for cache, (expected_k, expected_v) in zip(caches, pinned, strict=True):
        assert t.equal(cache.k_cache, expected_k)
        assert t.equal(cache.v_cache, expected_v)


def test_stack_prefilled_kv_caches_rejects_nonuniform_sources() -> None:
    def cache(*, batch_size: int = 1, seq_len: int = 8) -> KVCache:
        result = KVCache(
            batch_size=batch_size,
            num_heads=1,
            seq_len=seq_len,
            head_dim=4,
            num_layers=2,
            device=t.device("cpu"),
            dtype=t.float32,
        )
        result.advance(3)
        return result

    with pytest.raises(ValueError, match="at least one cache"):
        stack_prefilled_kv_caches(())
    with pytest.raises(ValueError, match="batch-one sources"):
        stack_prefilled_kv_caches((cache(), cache(batch_size=2)))
    with pytest.raises(ValueError, match="identical shapes"):
        stack_prefilled_kv_caches((cache(), cache(seq_len=9)))
    moved = cache()
    moved.advance(1)
    with pytest.raises(ValueError, match="one decode position"):
        stack_prefilled_kv_caches((cache(), moved))
    mismatched_dtype = cache()
    mismatched_dtype.v_cache = mismatched_dtype.v_cache.double()
    with pytest.raises(TypeError, match="one K/V dtype"):
        stack_prefilled_kv_caches((mismatched_dtype,))
    mismatched_device = cache()
    mismatched_device.v_cache = mismatched_device.v_cache.to("meta")
    with pytest.raises(ValueError, match="K and V on one device"):
        stack_prefilled_kv_caches((mismatched_device,))
    out_of_capacity = cache()
    out_of_capacity.advance(out_of_capacity.max_seq_len)
    with pytest.raises(ValueError, match="in-capacity decode position"):
        stack_prefilled_kv_caches((out_of_capacity,))


def test_tiny_gpt_greedy_batch_is_bitwise_sequential_oracle(
    gpt_model,
    mock_tokenizer,
) -> None:
    engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    prompt = _prompt(mock_tokenizer)
    kwargs: dict[str, Any] = {
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_k": None,
        "stop_token_ids": set(),
    }

    expected = _greedy_sequential_samples(engine, prompt, 4, **kwargs)
    actual = engine.generate_samples(prompt, seed=None, n_samples=4, **kwargs)

    assert actual == expected


@pytest.mark.parametrize("target_kind", ["tiny_gpt", "wrapped_encdec"])
@pytest.mark.parametrize(
    ("first_tokens", "survivor_count"),
    [
        pytest.param([-1, 1, 2], 2, id="one-row-stops"),
        pytest.param([-1, -1, 2], 1, id="two-rows-stop"),
    ],
)
def test_real_forward_forced_divergence_compaction_is_bitwise_oracle(
    target_kind: str,
    first_tokens: list[int],
    survivor_count: int,
    gpt_model,
    encdec_model,
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if target_kind == "tiny_gpt":
        model = gpt_model
    else:
        target = MultiTargetSegments(
            segments=[{5: 1}, {6: 2}, {7: 3}],
            token_assignments=t.tensor([0, 1, 1, 2], dtype=t.int64),
        )
        model = build_wrapper_for_features(
            encdec_model,
            mock_tokenizer,
            target=target,
            context_provider=_FullPrefixContextProvider(mock_tokenizer),
        )
    recorder = _ForwardRecorder(model)
    engine = InferenceEngine(recorder, mock_tokenizer, t.device("cpu"))
    prompt = _prompt(mock_tokenizer)
    budget = 8
    kwargs: dict[str, Any] = {
        "max_new_tokens": budget,
        "temperature": 0.0,
        "top_k": None,
        # -1 is outside the real vocabulary, so later argmax draws cannot stop
        # accidentally; it is filtered before any embedding lookup.
        "stop_token_ids": {-1},
    }

    # Each N=1 run forces that row's first token and then delegates every later
    # draw to the real argmax sampler. N=1 cannot compact, so these are the
    # independent non-compacting references over the same real model forwards.
    expected: list[list[int]] = []
    for first_token in first_tokens:
        recorder.reset()
        monkeypatch.setattr(
            "steering.eval.inference_engine.sample_next_token",
            _ForcedFirstThenGreedy([first_token]),
        )
        expected.append(
            engine.generate_samples(
                prompt,
                seed=None,
                n_samples=1,
                **kwargs,
            )[0]
        )

    recorder.reset()
    schedule = _ForcedFirstThenGreedy(first_tokens)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        schedule,
    )
    profile_dir = tmp_path / f"profile-{target_kind}-{survivor_count}"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(
            prompt,
            seed=None,
            n_samples=3,
            **kwargs,
        )

    assert actual == expected
    assert actual[0] == []
    if survivor_count == 1:
        assert actual[1] == []
    assert recorder.prefill_batch_sizes == [1]
    assert recorder.decode_batch_sizes == [survivor_count] * (budget - 1)
    assert schedule.calls == budget
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.batch_compactions"] >= 1
    assert "axbench_base.generation.compact_decode_batch" in _region_names(profile_dir)


def test_stub_greedy_compaction_matches_sequential_and_copies_survivor_kv(
    mock_tokenizer,
) -> None:
    prompt = [2, 3]
    kwargs: dict[str, Any] = {"max_new_tokens": 8, "temperature": 0.0}
    expected = [
        list(
            InferenceEngine(
                _HeterogeneousGreedyModel(single_schedule=schedule),
                mock_tokenizer,
                t.device("cpu"),
            ).generate_stream(prompt, seed=None, **kwargs)
        )
        for schedule in range(3)
    ]
    model = _HeterogeneousGreedyModel()
    engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))

    actual = engine.generate_samples(
        prompt,
        seed=None,
        n_samples=3,
        **kwargs,
    )

    assert actual == expected == [[1], [1, 2], [1, 3, 4]]
    assert model.decode_batch_sizes == [3, 2, 1]
    assert model.previous_k_markers[-1] == [3.0]


def test_forced_compactions_each_read_kv_position_exactly_once(
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the known Case-B host read without counting nanochat forward syncs."""
    model = _HeterogeneousGreedyModel()
    engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))
    real_get_pos = KVCache.get_pos
    get_pos_calls = 0

    def _get_pos_spy(cache: KVCache) -> int:
        nonlocal get_pos_calls
        get_pos_calls += 1
        return int(real_get_pos(cache))

    real_compact = engine._compact_kv_cache
    calls_per_compaction: list[int] = []

    def _compact_spy(kv_cache: KVCache, keep_indices: t.Tensor) -> KVCache:
        before = get_pos_calls
        compacted = real_compact(kv_cache, keep_indices)
        calls_per_compaction.append(get_pos_calls - before)
        return compacted

    monkeypatch.setattr(KVCache, "get_pos", _get_pos_spy)
    monkeypatch.setattr(engine, "_compact_kv_cache", _compact_spy)

    actual = engine.generate_samples(
        [2, 3],
        seed=None,
        n_samples=3,
        max_new_tokens=8,
        temperature=0.0,
    )

    assert actual == [[1], [1, 2], [1, 3, 4]]
    assert calls_per_compaction == [1, 1]


def test_single_sampled_row_exactly_matches_generate_stream(
    gpt_model,
    mock_tokenizer,
) -> None:
    engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    prompt = _prompt(mock_tokenizer)
    kwargs: dict[str, Any] = {
        "max_new_tokens": 24,
        "temperature": 0.8,
        "top_k": 16,
        "stop_token_ids": set(),
    }

    expected = list(engine.generate_stream(prompt, seed=23, **kwargs))
    actual = engine.generate_samples(prompt, seed=23, n_samples=1, **kwargs)

    assert actual == [expected]


@pytest.mark.parametrize("top_k", [None, 32])
def test_sampled_batch_is_deterministic_and_seed_sensitive(
    gpt_model,
    mock_tokenizer,
    top_k: int | None,
) -> None:
    engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    prompt = _prompt(mock_tokenizer)
    kwargs: dict[str, Any] = {
        "n_samples": 4,
        "max_new_tokens": 32,
        "temperature": 1.1,
        "top_k": top_k,
        "stop_token_ids": set(),
    }

    first = engine.generate_samples(prompt, seed=123, **kwargs)
    repeated = engine.generate_samples(prompt, seed=123, **kwargs)
    changed = engine.generate_samples(prompt, seed=124, **kwargs)

    assert first == repeated
    assert first != changed


def test_wrapped_encdec_multitarget_sampled_is_seed_deterministic(
    encdec_model,
    mock_tokenizer,
) -> None:
    target = MultiTargetSegments(
        segments=[{5: 1}, {6: 2}, {7: 3}],
        token_assignments=t.tensor([0, 1, 1, 2], dtype=t.int64),
    )
    wrapper = build_wrapper_for_features(
        encdec_model,
        mock_tokenizer,
        target=target,
        context_provider=_FullPrefixContextProvider(mock_tokenizer),
    )
    engine = InferenceEngine(wrapper, mock_tokenizer, t.device("cpu"))
    kwargs: dict[str, Any] = {
        "n_samples": 3,
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_k": 32,
        "stop_token_ids": set(),
    }

    first = engine.generate_samples(_prompt(mock_tokenizer), seed=7, **kwargs)
    repeated = engine.generate_samples(_prompt(mock_tokenizer), seed=7, **kwargs)
    changed = engine.generate_samples(_prompt(mock_tokenizer), seed=8, **kwargs)

    assert first == repeated
    assert first != changed


def test_wrapped_engine_reuse_overwrites_prompt_conditioning_state(
    encdec_model,
    mock_tokenizer,
) -> None:
    prompt_a = cast(
        list[int],
        mock_tokenizer.encode(
            "context alpha",
            prepend=mock_tokenizer.get_bos_token_id(),
        ),
    )
    prompt_b = cast(
        list[int],
        mock_tokenizer.encode(
            "context beta",
            prepend=mock_tokenizer.get_bos_token_id(),
        ),
    )
    features_by_prompt = {
        tuple(prompt_a): (0, 1),
        tuple(prompt_b): (1, 3),
    }
    provider = _PromptFeatureContextProvider(mock_tokenizer, features_by_prompt)
    reused = build_engine_for_features(
        encdec_model,
        mock_tokenizer,
        t.device("cpu"),
        features={5: 2},
        context_provider=provider,
    )
    kwargs: dict[str, Any] = {
        "seed": 7,
        "n_samples": 3,
        "max_new_tokens": 6,
        "temperature": 0.7,
        "top_k": 32,
        "stop_token_ids": set(),
    }

    reused.generate_samples(prompt_a, **kwargs)
    state_a = reused._model._generation_encoder_state
    assert state_a is not None
    encoder_out_a = state_a.encoder_out.values.clone()
    reused_b = reused.generate_samples(prompt_b, **kwargs)
    state_b = reused._model._generation_encoder_state
    assert state_b is not None

    fresh = build_engine_for_features(
        encdec_model,
        mock_tokenizer,
        t.device("cpu"),
        features={5: 2},
        context_provider=provider,
    )
    fresh_b = fresh.generate_samples(prompt_b, **kwargs)
    fresh_state_b = fresh._model._generation_encoder_state
    assert fresh_state_b is not None

    assert reused_b == fresh_b
    assert not t.equal(encoder_out_a, state_b.encoder_out.values)
    _assert_packed_equal(state_b.encoder_out, fresh_state_b.encoder_out)


@pytest.mark.parametrize(
    ("max_new_tokens", "bos_only"),
    [
        pytest.param(1, False, id="one-token-budget"),
        pytest.param(12, True, id="bos-only-prompt"),
    ],
)
def test_real_gpt_sampled_shape_sweep_is_deterministic(
    gpt_model,
    mock_tokenizer,
    max_new_tokens: int,
    bos_only: bool,
) -> None:
    engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    prompt = [mock_tokenizer.get_bos_token_id()] if bos_only else _prompt(mock_tokenizer)
    kwargs: dict[str, Any] = {
        "n_samples": 4,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_k": 32,
        "stop_token_ids": set(),
    }

    first = engine.generate_samples(prompt, seed=101, **kwargs)
    repeated = engine.generate_samples(prompt, seed=101, **kwargs)
    changed = engine.generate_samples(prompt, seed=102, **kwargs)

    assert first == repeated
    assert first != changed
    assert all(len(row) == max_new_tokens for row in first)


def test_compaction_bookkeeping_first_stop_and_one_survivor_to_budget(
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, model = _uniform_engine(mock_tokenizer)
    draws = iter([[0, 1, 2], [0, 3], [4]])
    sampled_shapes: list[tuple[int, int]] = []

    def _sample(logits, _rng, _temperature, _top_k):
        sampled_shapes.append(tuple(logits.shape))
        return t.tensor(next(draws), dtype=t.long).unsqueeze(1)

    monkeypatch.setattr("steering.eval.inference_engine.sample_next_token", _sample)
    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(
            [2, 3],
            seed=7,
            n_samples=3,
            max_new_tokens=3,
            temperature=1.0,
        )

    assert actual == [[], [1], [2, 3, 4]]
    assert sampled_shapes == [(3, 5), (2, 5), (1, 5)]
    assert model.decode_batch_sizes == [2, 1]
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 6
    assert counters["axbench_base.generation.batched_decode_iterations"] == 3
    assert counters["axbench_base.generation.batch_compactions"] == 2
    regions = _region_names(profile_dir)
    assert "axbench_base.generation.expand_prefill_cache" in regions
    assert "axbench_base.generation.compact_decode_batch" in regions


def test_custom_stop_token_override_is_honored_per_row_and_compacts(
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt = [2, 3]
    kwargs: dict[str, Any] = {
        "max_new_tokens": 4,
        "temperature": 1.0,
        "stop_token_ids": {3},
    }

    def run_stream(draws: list[int]) -> list[int]:
        engine, _model = _uniform_engine(mock_tokenizer)
        scheduled = iter(draws)
        monkeypatch.setattr(
            "steering.eval.inference_engine.sample_next_token",
            lambda logits, *_args: t.tensor(
                [[next(scheduled)]], dtype=t.long, device=logits.device
            ),
        )
        return list(engine.generate_stream(prompt, seed=7, **kwargs))

    expected = [run_stream([3]), run_stream([1, 2, 3])]
    engine, model = _uniform_engine(mock_tokenizer)
    batch_draws = iter([[3, 1], [2], [3]])

    def sample_batch(logits, *_args):
        return t.tensor(
            next(batch_draws),
            dtype=t.long,
            device=logits.device,
        ).unsqueeze(1)

    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        sample_batch,
    )
    profile_dir = tmp_path / "custom-stop-profile"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(
            prompt,
            seed=7,
            n_samples=2,
            **kwargs,
        )

    assert actual == expected == [[], [1, 2]]
    assert [len(row) for row in actual] == [0, 2]
    assert model.decode_batch_sizes == [1, 1]
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.batch_compactions"] == 1
    assert "axbench_base.generation.compact_decode_batch" in _region_names(profile_dir)


def test_final_budgeted_partial_stop_skips_useless_compaction(
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, model = _uniform_engine(mock_tokenizer)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        lambda logits, *_args: t.tensor([[0], [1], [2]], dtype=t.long, device=logits.device),
    )
    profile_dir = tmp_path / "final-draw-profile"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(
            [2, 3],
            seed=None,
            n_samples=3,
            max_new_tokens=1,
            temperature=0.0,
        )

    assert actual == [[], [1], [2]]
    assert model.decode_batch_sizes == []
    counters = _counter_totals(profile_dir)
    assert "axbench_base.generation.batch_compactions" not in counters
    assert "axbench_base.generation.compact_decode_batch" not in _region_names(profile_dir)


def test_all_rows_finishing_simultaneously_has_no_forward_or_compaction(
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, model = _uniform_engine(mock_tokenizer)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        lambda logits, *_args: t.zeros((logits.size(0), 1), dtype=t.long),
    )
    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(
            [2, 3], seed=7, n_samples=3, max_new_tokens=8, temperature=1.0
        )

    assert actual == [[], [], []]
    assert model.decode_batch_sizes == []
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 3
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1
    assert "axbench_base.generation.batch_compactions" not in counters
    assert "axbench_base.generation.compact_decode_batch" not in _region_names(profile_dir)


def test_no_finished_rows_has_no_compaction_and_no_final_forward(
    mock_tokenizer,
    tmp_path: Path,
) -> None:
    engine, model = _uniform_engine(mock_tokenizer)
    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(
            [2, 3],
            seed=None,
            n_samples=3,
            max_new_tokens=4,
            temperature=0.0,
            stop_token_ids=set(),
        )

    assert [len(row) for row in actual] == [4, 4, 4]
    assert model.decode_batch_sizes == [3, 3, 3]
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.prefill_tokens"] == 2
    assert counters["axbench_base.generation.decode_steps"] == 12
    assert counters["axbench_base.generation.batched_decode_iterations"] == 4
    assert "axbench_base.generation.batch_compactions" not in counters
    assert "axbench_base.generation.compact_decode_batch" not in _region_names(profile_dir)


def test_quiet_path_has_one_loop_local_stop_sync_and_one_final_bulk_conversion(
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort lint for loop-local quiet-path syncs, not an end-to-end oracle."""
    engine, _model = _uniform_engine(mock_tokenizer)
    budget = 5
    real_isin = t.isin
    isin_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    sampled_strides: list[tuple[int, ...]] = []

    def _isin_spy(elements: t.Tensor, test_elements: t.Tensor) -> t.Tensor:
        isin_shapes.append((tuple(elements.shape), tuple(test_elements.shape)))
        return real_isin(elements, test_elements)

    def _sample_passthrough(logits, rng, temperature, top_k):
        sampled_strides.append(logits.stride())
        return _real_sample_next_token(logits, rng, temperature, top_k)

    monkeypatch.setattr("steering.eval.inference_engine.t.isin", _isin_spy)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        _sample_passthrough,
    )
    actual = engine.generate_samples(
        [2, 3],
        seed=None,
        n_samples=4,
        max_new_tokens=budget,
        temperature=0.0,
        stop_token_ids=set(),
    )

    assert all(len(row) == budget for row in actual)
    assert isin_shapes == [((4,), (0,))] * budget
    assert len(sampled_strides) == budget
    assert sampled_strides[0][0] == 0  # expanded shared-prefill logits view

    # Tensor's C-level methods cannot be monkeypatched portably. This attribute-
    # name-level AST lint is a best-effort regression proxy for the quiet branch,
    # not an end-to-end sync oracle: it permits exactly the loop-local
    # ``bool(finished_now.any())`` stop check, rejects common sync-shaped calls in
    # the quiet path, and pins one bulk ``tolist`` after the loop. Runtime spies
    # above prove both t.isin and sampling execute K times.
    source = textwrap.dedent(inspect.getsource(InferenceEngine.generate_samples))
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert len(loops) == 1
    loop = loops[0]
    assert loop.end_lineno is not None
    loop_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
    ]
    any_calls = [
        node
        for node in loop_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "any"
    ]
    bool_calls = [
        node
        for node in loop_calls
        if isinstance(node.func, ast.Name) and node.func.id == "bool"
    ]
    assert len(any_calls) == 1
    assert loop.lineno <= any_calls[0].lineno <= loop.end_lineno
    assert len(bool_calls) == 2
    stop_sync_bool_calls = [call for call in bool_calls if call.args == [any_calls[0]]]
    assert len(stop_sync_bool_calls) == 1

    quiet_branch_index, quiet_branch = next(
        (index, node)
        for index, node in enumerate(loop.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
    )
    quiet_path_statements = [
        *loop.body[:quiet_branch_index],
        *quiet_branch.body,
    ]
    quiet_path_calls = [
        node
        for statement in quiet_path_statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    forbidden_attributes = {"all", "cpu", "item", "synchronize", "tolist"}
    forbidden_names = {"bool", "int"}
    assert [
        node
        for node in quiet_path_calls
        if node not in {any_calls[0], stop_sync_bool_calls[0]}
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_attributes
        )
        or (isinstance(node.func, ast.Name) and node.func.id in forbidden_names)
    ] == []

    method_attributes = [
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    tolist_calls = [
        attribute for attribute in method_attributes if attribute.attr == "tolist"
    ]
    assert len(tolist_calls) == 1
    assert tolist_calls[0].lineno > loop.end_lineno


@pytest.mark.parametrize(
    ("stop_ids", "scheduled_draws", "expected"),
    [
        pytest.param(
            set(),
            [[1, 2], [3, 4]],
            [[1, 3], [2, 4]],
            id="empty-stop-set",
        ),
        pytest.param(
            {0},
            [[0, 1], [0]],
            [[], [1]],
            id="single-stop-set",
        ),
    ],
)
def test_empty_and_single_stop_sets_use_tensor_isin(
    stop_ids: set[int],
    scheduled_draws: list[list[int]],
    expected: list[list[int]],
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _model = _uniform_engine(mock_tokenizer)
    draws = iter(scheduled_draws)
    real_isin = t.isin
    observed_stop_ids: list[list[int]] = []

    def _isin_spy(elements: t.Tensor, test_elements: t.Tensor) -> t.Tensor:
        observed_stop_ids.append(test_elements.tolist())
        return real_isin(elements, test_elements)

    monkeypatch.setattr("steering.eval.inference_engine.t.isin", _isin_spy)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        lambda logits, *_args: t.tensor(
            next(draws),
            dtype=t.long,
            device=logits.device,
        ).unsqueeze(1),
    )

    actual = engine.generate_samples(
        [2, 3],
        seed=None,
        n_samples=2,
        max_new_tokens=2,
        temperature=0.0,
        stop_token_ids=stop_ids,
    )

    assert actual == expected
    assert observed_stop_ids == [sorted(stop_ids)] * 2


@pytest.mark.parametrize(
    ("n_samples", "budget", "stop_ids", "scheduled_draws", "expected"),
    [
        pytest.param(1, 0, set(), [], [[]], id="one-row-zero-budget"),
        pytest.param(2, 1, {9}, [[9, 3]], [[], [3]], id="final-draw-stop"),
        pytest.param(
            4,
            4,
            {0, 5},
            [[1, 0, 2, 5], [3, 0], [4], [5]],
            [[1, 3, 4], [], [2], []],
            id="heterogeneous-multi-stop",
        ),
    ],
)
def test_tensor_buffer_fresh_shape_oracles_are_byte_identical(
    n_samples: int,
    budget: int,
    stop_ids: set[int],
    scheduled_draws: list[list[int]],
    expected: list[list[int]],
    mock_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _model = _uniform_engine(mock_tokenizer)
    draws = iter(scheduled_draws)

    def _sample(logits, *_args):
        row = next(draws)
        assert len(row) == logits.size(0)
        return t.tensor(row, dtype=t.long, device=logits.device).unsqueeze(1)

    monkeypatch.setattr("steering.eval.inference_engine.sample_next_token", _sample)
    actual = engine.generate_samples(
        [2, 3],
        seed=None,
        n_samples=n_samples,
        max_new_tokens=budget,
        temperature=0.0,
        stop_token_ids=stop_ids,
    )

    assert actual == expected
    with pytest.raises(StopIteration):
        next(draws)


@pytest.mark.parametrize("seed", [None, True, 1.5])
def test_sampled_seed_guard_is_pinned(
    mock_tokenizer,
    seed: object,
) -> None:
    engine, _model = _uniform_engine(mock_tokenizer)
    with pytest.raises(
        ValueError,
        match=(
            r"seed must be a non-None, non-bool int when temperature > 0\.0; "
            rf"got seed={re.escape(repr(seed))}, temperature=0\.7"
        ),
    ):
        engine.generate_samples(
            [2, 3],
            seed=seed,  # ty: ignore[invalid-argument-type]
            n_samples=1,
            temperature=0.7,
        )


@pytest.mark.parametrize("n_samples", [0, -1, True, 1.5])
def test_n_samples_guard_is_pinned(
    mock_tokenizer,
    n_samples: object,
) -> None:
    engine, _model = _uniform_engine(mock_tokenizer)
    with pytest.raises(
        ValueError,
        match=rf"n_samples must be an int >= 1, got {re.escape(repr(n_samples))}",
    ):
        engine.generate_samples(
            [2, 3],
            seed=None,
            n_samples=n_samples,  # ty: ignore[invalid-argument-type]
            temperature=0.0,
        )


def test_greedy_ignores_seed(mock_tokenizer) -> None:
    engine, _model = _uniform_engine(mock_tokenizer)
    kwargs: dict[str, Any] = {
        "n_samples": 2,
        "max_new_tokens": 3,
        "temperature": 0.0,
        "stop_token_ids": set(),
    }
    assert engine.generate_samples([2, 3], seed=None, **kwargs) == engine.generate_samples(
        [2, 3], seed=True, **kwargs
    )


def test_exact_context_boundary_succeeds_and_overflow_message_matches_stream(
    mock_tokenizer,
) -> None:
    engine, _model = _uniform_engine(mock_tokenizer, sequence_len=8)
    prompt = [2, 3, 2]
    boundary_kwargs: dict[str, Any] = {
        "max_new_tokens": 5,
        "temperature": 0.0,
        "stop_token_ids": set(),
    }
    expected_boundary = list(engine.generate_stream(prompt, seed=None, **boundary_kwargs))
    actual_boundary = engine.generate_samples(
        prompt,
        seed=None,
        n_samples=1,
        **boundary_kwargs,
    )

    assert actual_boundary == [expected_boundary]
    assert len(expected_boundary) == 5

    expected = "Prompt length (3) + max_new_tokens (6) exceeds context window (8)"
    with pytest.raises(ValueError, match=re.escape(expected)):
        list(engine.generate_stream(prompt, seed=None, max_new_tokens=6, temperature=0.0))
    with pytest.raises(ValueError, match=re.escape(expected)):
        engine.generate_samples(
            prompt,
            seed=None,
            n_samples=1,
            max_new_tokens=6,
            temperature=0.0,
        )


def test_self_conditioned_rejects_multiple_samples_and_one_matches_stream(
    mock_tokenizer,
) -> None:
    t.manual_seed(0)
    model = build_model_with_feature_prediction()
    model.eval()
    engine = build_engine_self_conditioned(
        model,
        mock_tokenizer,
        t.device("cpu"),
        _self_cond_config(),
        context_provider=_FullPrefixContextProvider(mock_tokenizer),
    )
    prompt = [2, 3]
    kwargs: dict[str, Any] = {
        "max_new_tokens": 3,
        "temperature": 0.0,
        "stop_token_ids": set(),
    }
    with pytest.raises(ValueError, match=r"self-conditioned mode with N=2"):
        engine.generate_samples(prompt, seed=None, n_samples=2, **kwargs)

    expected = list(engine.generate_stream(prompt, seed=None, **kwargs))
    actual = engine.generate_samples(prompt, seed=None, n_samples=1, **kwargs)
    assert actual == [expected]


def _self_conditioned_engine(
    model: Any,
    tokenizer: MockTokenizer,
) -> InferenceEngine:
    return build_engine_self_conditioned(
        model,
        tokenizer,
        t.device("cpu"),
        _self_cond_config(),
        context_provider=_FullPrefixContextProvider(tokenizer),
    )


def _fixed_and_turn_engines(
    model: Any,
    tokenizer: MockTokenizer,
    *,
    segment_size: int,
) -> tuple[list[int], InferenceEngine, InferenceEngine]:
    prompt = tokenizer.render_for_completion(empty_assistant_conversation("q"))
    offsets = token_to_char_offsets(tokenizer, prompt)
    marker_pos = len(prompt) - 1
    segments = (
        ContextFeatureSegment(
            feature_ids=(0,),
            buckets=(1,),
            char_start=int(offsets[0]),
            char_end=int(offsets[marker_pos]),
        ),
        ContextFeatureSegment(
            feature_ids=(int(N_SAE_FEATURES) - 1,),
            buckets=(1,),
            char_start=int(offsets[marker_pos]),
            char_end=int(offsets[marker_pos + 1]),
        ),
    )
    provider = SingleEntryContextProvider(prompt, segments)
    fixed_config = replace(
        _self_cond_config(),
        self_cond_segment_size=PositiveInt.parse(segment_size),
    )
    turn_config = replace(
        fixed_config,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    fixed_engine = build_engine_self_conditioned(
        model,
        tokenizer,
        t.device("cpu"),
        fixed_config,
        context_provider=provider,
    )
    turn_engine = build_engine_self_conditioned(
        model,
        tokenizer,
        t.device("cpu"),
        turn_config,
        # kills: weakening the AxBench ownership guard to admit an unstamped fixture.
        context_provider=SingleEntryContextProvider(
            prompt, segments, segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            token_assignments=(0,) * marker_pos + (1,),
        ),
    )
    return prompt, fixed_engine, turn_engine


def test_turn_generate_stream_applies_the_wrappers_structural_stops(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: dropping the wrapper's ``generation_stop_token_ids`` union in
    # generate_stream, which would yield a chat structural token as ordinary
    # assistant text (and then forward it into the turn decode step).
    model = build_model_with_feature_prediction()
    model.eval()
    prompt, _fixed_engine, turn_engine = _fixed_and_turn_engines(
        model,
        mock_tokenizer,
        segment_size=2,
    )
    structural_stop = mock_tokenizer.encode_special("<assistant_end>")
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        lambda *_args: t.tensor([[structural_stop]], dtype=t.long),
    )

    assert (
        list(
            turn_engine.generate_stream(
                prompt,
                max_new_tokens=3,
                temperature=0.0,
                stop_token_ids=set(),
            )
        )
        == []
    )


def test_turn_generate_samples_single_row_uses_structural_stops_and_cadence(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model_with_feature_prediction()
    model.eval()
    prompt, _fixed_engine, turn_engine = _fixed_and_turn_engines(
        model,
        mock_tokenizer,
        segment_size=2,
    )
    structural_stop = mock_tokenizer.encode_special("<assistant_end>")
    scheduled = iter((10, structural_stop))

    def scheduled_sample(_logits, _rng, _temperature, _top_k):
        return t.tensor([[next(scheduled)]], dtype=t.long)

    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        scheduled_sample,
    )
    generated = turn_engine.generate_samples(
        prompt,
        seed=None,
        n_samples=1,
        max_new_tokens=3,
        temperature=0.0,
        stop_token_ids=set(),
    )

    assert generated == [[10]]
    wrapper = turn_engine._model
    snapshot = wrapper.self_cond_runtime_snapshot()
    # Marker + ordinary token complete K=2. The sampled structural stop is
    # neither returned nor forwarded into a third row.
    assert snapshot.state.target_slot_idx == 2
    assert snapshot.state.encoder_out.layout.n_segments == 3
    assert snapshot.tokens_in_seg == 0
    assert snapshot.fp_buffer == ()


def _ranked_v3_setup() -> Any:
    return SimpleNamespace(
        n_sae_features=int(N_SAE_FEATURES),
        allowed_ids=None,
        feature_mask=None,
        output_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=4,
            mode=SamplingMode.top_k_percentile,
        ),
        output_negative_selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=4,
        ),
        percentile_table=t.zeros((int(N_SAE_FEATURES), 99), dtype=t.float64),
        consolidation=None,
        # ``MethodSetup.segment_mode`` is required provenance, so the stub must
        # declare it rather than lean on a removed getattr default.
        segment_mode=SegmentMode.fixed_tokens,
    )


def _ranked_v3_engine(
    model: Any,
    tokenizer: MockTokenizer,
    setup: Any,
    target_scores: t.Tensor,
    *,
    factor: float,
    selection_seed: int,
) -> tuple[InferenceEngine, SelfCondConfig]:
    policy = build_self_conditioning_masked_policy(
        setup,
        target_scores.clone(),
        factor,
    )
    production_config = build_self_conditioning_config(
        setup,
        model,
        policy,
        selection_seed=selection_seed,
    )
    # A short test-only window exercises a completed ranked-v3 feedback boundary
    # without changing any policy, model, or engine construction semantics.
    config = replace(
        production_config,
        self_cond_segment_size=PositiveInt.parse(2),
    )
    return (
        build_engine_self_conditioned(
            model,
            tokenizer,
            t.device("cpu"),
            config,
            context_provider=_FullPrefixContextProvider(tokenizer),
        ),
        config,
    )


@t.inference_mode()
def _frozen_ranked_v3_scalar_generate_stream(
    engine: InferenceEngine,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> tuple[list[int], KVCache]:
    """Frozen scalar loop from before self-conditioned prefill extraction."""

    engine._validate_prompt_length(len(token_ids), max_new_tokens)
    total_len = len(token_ids) + max_new_tokens
    kv_cache = engine._make_kv_cache(batch_size=1, seq_len=total_len)
    idx = t.tensor([token_ids], dtype=t.long, device=engine._device)
    with engine._model.conditioning_context(context_token_ids=[token_ids]):
        logits = engine._model(idx, kv_cache=kv_cache)
    logits = logits[:, -1, :]

    rng = t.Generator(device=engine._device)
    rng.manual_seed(seed)
    generated: list[int] = []
    for token_index in range(max_new_tokens):
        next_id = _real_sample_next_token(logits, rng, temperature, top_k)
        generated.append(int(next_id.item()))
        if token_index + 1 == max_new_tokens:
            break
        logits = engine._model(next_id, kv_cache=kv_cache)
        logits = logits[:, -1, :]
    return generated, kv_cache


def _current_ranked_v3_scalar_generate_stream(
    engine: InferenceEngine,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], KVCache]:
    caches: list[KVCache] = []
    make_kv_cache = engine._make_kv_cache

    def capture_kv_cache(*, batch_size: int, seq_len: int) -> KVCache:
        cache = make_kv_cache(batch_size=batch_size, seq_len=seq_len)
        caches.append(cache)
        return cache

    monkeypatch.setattr(engine, "_make_kv_cache", capture_kv_cache)
    generated = list(
        engine.generate_stream(
            token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            stop_token_ids=set(),
        )
    )
    assert len(caches) == 1
    return generated, caches[0]


def _ranked_v3_post_state(
    engine: InferenceEngine,
    config: SelfCondConfig,
    cache: KVCache,
) -> dict[str, Any]:
    wrapper = engine._model
    runtime = wrapper._self_cond
    snapshot = wrapper.self_cond_runtime_snapshot()
    provider = runtime.provider
    selection = cast(Any, config.selection)
    return {
        "cache": cache,
        "encoder_out": snapshot.state.encoder_out,
        "target_slot_idx": snapshot.state.target_slot_idx,
        "tokens_in_seg": snapshot.tokens_in_seg,
        "fp_buffer": tuple(value.clone() for value in snapshot.fp_buffer),
        "last_appended_segment": provider.last_appended_segment,
        "queued_explicit": provider._queued_explicit,
        "positive_rows": provider._masking_positive_rows.clone(),
        "negative_rows": provider._masking_negative_rows.clone(),
        "selection_rng": selection.rng.get_state().clone(),
    }


def _assert_ranked_v3_post_state_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    actual_cache = cast(KVCache, actual["cache"])
    expected_cache = cast(KVCache, expected["cache"])
    assert t.equal(actual_cache.k_cache, expected_cache.k_cache)
    assert t.equal(actual_cache.v_cache, expected_cache.v_cache)
    assert t.equal(actual_cache.cache_seqlens, expected_cache.cache_seqlens)
    assert actual_cache.get_pos() == expected_cache.get_pos()
    _assert_packed_equal(actual["encoder_out"], expected["encoder_out"])
    assert actual["target_slot_idx"] == expected["target_slot_idx"]
    assert actual["tokens_in_seg"] == expected["tokens_in_seg"]
    assert len(actual["fp_buffer"]) == len(expected["fp_buffer"])
    for actual_logits, expected_logits in zip(
        actual["fp_buffer"],
        expected["fp_buffer"],
        strict=True,
    ):
        assert t.equal(actual_logits, expected_logits)
    assert actual["last_appended_segment"] == expected["last_appended_segment"]
    assert actual["queued_explicit"] == expected["queued_explicit"]
    assert t.equal(actual["positive_rows"], expected["positive_rows"])
    assert t.equal(actual["negative_rows"], expected["negative_rows"])
    assert t.equal(actual["selection_rng"], expected["selection_rng"])


@pytest.mark.parametrize("execution_order", [("current", "frozen"), ("frozen", "current")])
@pytest.mark.parametrize(
    ("factor", "selection_seed", "token_seed"),
    [(0.0, 17, 101), (6.0, 29, 211)],
)
def test_ranked_v3_generate_stream_matches_frozen_scalar_loop_exactly(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    execution_order: tuple[str, str],
    factor: float,
    selection_seed: int,
    token_seed: int,
) -> None:
    t.manual_seed(20260830)
    model = build_tiny_model(
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )
    model.eval()
    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    setup = _ranked_v3_setup()
    half_width = int(N_SAE_FEATURES) // 2
    target_scores = t.cat(
        (
            t.linspace(0.1, 1.0, half_width),
            t.linspace(-1.0, -0.1, int(N_SAE_FEATURES) - half_width),
        )
    )
    original_target_scores = target_scores.clone()
    engines_and_configs = {
        name: _ranked_v3_engine(
            model,
            mock_tokenizer,
            setup,
            target_scores,
            factor=factor,
            selection_seed=selection_seed,
        )
        for name in ("current", "frozen")
    }
    initial_rng_states = {
        name: cast(Any, config.selection).rng.get_state().clone()
        for name, (_engine, config) in engines_and_configs.items()
    }
    assert t.equal(initial_rng_states["current"], initial_rng_states["frozen"])
    assert cast(Any, engines_and_configs["current"][1].selection).rng is not cast(
        Any,
        engines_and_configs["frozen"][1].selection,
    ).rng

    prompt = [2, 3]
    generation_kwargs = {
        "max_new_tokens": 4,
        "temperature": 0.8,
        "top_k": 20,
        "seed": token_seed,
    }
    results: dict[str, tuple[list[int], dict[str, Any]]] = {}
    for index, name in enumerate(execution_order):
        engine, config = engines_and_configs[name]
        if name == "current":
            generated, cache = _current_ranked_v3_scalar_generate_stream(
                engine,
                prompt,
                monkeypatch=monkeypatch,
                **generation_kwargs,
            )
        else:
            generated, cache = _frozen_ranked_v3_scalar_generate_stream(
                engine,
                prompt,
                **generation_kwargs,
            )
        results[name] = (generated, _ranked_v3_post_state(engine, config, cache))
        if index == 0:
            pending = execution_order[1]
            pending_config = engines_and_configs[pending][1]
            assert t.equal(
                cast(Any, pending_config.selection).rng.get_state(),
                initial_rng_states[pending],
            )

    current_tokens, current_state = results["current"]
    frozen_tokens, frozen_state = results["frozen"]
    assert current_tokens == frozen_tokens
    assert len(current_tokens) == 4
    _assert_ranked_v3_post_state_equal(current_state, frozen_state)

    cache = cast(KVCache, current_state["cache"])
    assert cache.get_pos() == len(prompt) + len(current_tokens) - 1
    assert current_state["tokens_in_seg"] == 1
    assert len(current_state["fp_buffer"]) == 1
    assert current_state["fp_buffer"][0].shape == (1, 1, int(N_SAE_FEATURES))
    assert t.isfinite(current_state["fp_buffer"][0]).all()
    appended = current_state["last_appended_segment"]
    assert appended
    assert set(appended.values()) == {0, 1}
    assert current_state["queued_explicit"] is None
    assert not t.equal(current_state["selection_rng"], initial_rng_states["current"])
    assert t.equal(target_scores, original_target_scores)
    for name, expected in model_state.items():
        assert t.equal(model.state_dict()[name], expected)


def test_self_conditioned_prefill_and_batch_validation_is_exhaustive(
    gpt_model: Any,
    mock_tokenizer: MockTokenizer,
) -> None:
    plain_engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    with pytest.raises(ValueError, match="requires a self-conditioned"):
        plain_engine.prepare_self_conditioned_prefill([2, 3], max_new_tokens=1)
    with pytest.raises(ValueError, match="requires a self-conditioned"):
        plain_engine.generate_self_conditioned_batch(
            (),
            seeds=(),
            max_new_tokens=1,
            temperature=0.0,
            top_k=None,
        )

    model = build_model_with_feature_prediction()
    model.eval()
    engine = _self_conditioned_engine(model, mock_tokenizer)
    for tokens, budget, error, message in (
        ([], 1, ValueError, "token_ids must be nonempty"),
        ([2, 3], True, TypeError, "max_new_tokens must be an integer"),
        ([2, 3], -1, ValueError, "max_new_tokens must be nonnegative"),
    ):
        with pytest.raises(error, match=message):
            engine.prepare_self_conditioned_prefill(tokens, max_new_tokens=budget)

    prefill = engine.prepare_self_conditioned_prefill([2, 3], max_new_tokens=4)
    assert prefill.segment_mode is SegmentMode.fixed_tokens

    # kills: restoring a SegmentMode default on the provenance field, which
    # would let a future producer claim fixed provenance by omission.
    with pytest.raises(TypeError, match="segment_mode"):
        SelfConditionedPrefill(  # ty: ignore[missing-argument]
            logits=prefill.logits,
            kv_cache=prefill.kv_cache,
            runtime=prefill.runtime,
            decoder=prefill.decoder,
            segment_size=prefill.segment_size,
        )

    def generate(
        rows: tuple[SelfConditionedPrefill, ...],
        *,
        seeds: tuple[Any, ...] = (1,),
        max_new_tokens: Any = 4,
        temperature: float = 0.0,
        stop_token_ids: set[int] | None = None,
    ) -> list[list[int]]:
        return engine.generate_self_conditioned_batch(
            rows,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            stop_token_ids=stop_token_ids,
        )

    with pytest.raises(ValueError, match="at least one prefill"):
        generate((), seeds=())
    with pytest.raises(ValueError, match="one seed per prefill"):
        generate((prefill,), seeds=())
    with pytest.raises(TypeError, match="max_new_tokens must be an integer"):
        generate((prefill,), max_new_tokens=True)
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        generate((prefill,), max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature must be nonnegative"):
        generate((prefill,), temperature=-0.1)
    with pytest.raises(ValueError, match="non-bool integer seed"):
        generate((prefill,), seeds=(True,), temperature=0.7)

    bad_decoder = replace(prefill, decoder=build_model_with_feature_prediction())
    with pytest.raises(ValueError, match="active decoder"):
        generate((bad_decoder,))
    bad_window = replace(prefill, segment_size=prefill.segment_size + 1)
    with pytest.raises(ValueError, match="active segment window"):
        generate((bad_window,))
    bad_shape = replace(prefill, logits=prefill.logits[0])
    with pytest.raises(ValueError, match=r"shape \(1, V\)"):
        generate((bad_shape,))
    bad_device = replace(prefill, logits=prefill.logits.to("meta"))
    with pytest.raises(ValueError, match="engine device"):
        generate((bad_device,))
    bad_dtype = replace(prefill, logits=prefill.logits.double())
    with pytest.raises(ValueError, match="one vocabulary and dtype"):
        generate((prefill, bad_dtype), seeds=(1, 2))

    actual = generate((prefill,), seeds=(None,), max_new_tokens=1, stop_token_ids=None)
    assert len(actual) == 1

    target_aware = build_wrapper_for_features(
        model,
        mock_tokenizer,
        features={},
        context_provider=_FullPrefixContextProvider(mock_tokenizer),
    )
    with pytest.raises(RuntimeError, match="self-conditioned mode"):
        target_aware.self_cond_segment_size()
    target_engine = InferenceEngine(target_aware, mock_tokenizer, t.device("cpu"))
    generated = list(
        target_engine.generate_stream(
            [2, 3],
            max_new_tokens=1,
            temperature=0.0,
            stop_token_ids=set(),
        )
    )
    assert len(generated) == 1


# kills: deleting the turn-mode batch rejection, or moving it below the
# max_new_tokens-vs-segment_size budget check (the budget below deliberately
# exceeds the engine's K, so a reordered guard reports the wrong defect).
def test_turn_self_conditioned_batch_rejects_before_any_batch_work(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model_with_feature_prediction()
    model.eval()
    prompt, _fixed_engine, engine = _fixed_and_turn_engines(
        model,
        mock_tokenizer,
        segment_size=4,
    )
    prefill = engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=5)

    def unexpected_batch_work(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("turn mode entered optimized batch work")

    monkeypatch.setattr(
        "steering.eval.inference_engine.stack_packed_encoder_outputs",
        unexpected_batch_work,
    )
    monkeypatch.setattr(
        "steering.eval.inference_engine.stack_prefilled_kv_caches",
        unexpected_batch_work,
    )
    monkeypatch.setattr(
        "steering.eval.inference_engine._sample_next_token_from_finite_logits",
        unexpected_batch_work,
    )

    with pytest.raises(
        ValueError,
        match="does not support turn-delimited.*scalar generate_stream calls",
    ):
        engine.generate_self_conditioned_batch(
            (prefill,),
            seeds=(17,),
            max_new_tokens=5,
            temperature=0.7,
            top_k=20,
            stop_token_ids=set(),
        )


def test_fixed_engine_rejects_genuine_turn_prefill_provenance_before_batch_work(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model_with_feature_prediction()
    model.eval()
    prompt, fixed_engine, turn_engine = _fixed_and_turn_engines(
        model,
        mock_tokenizer,
        segment_size=1,
    )
    turn_prefill = turn_engine.prepare_self_conditioned_prefill(
        prompt,
        max_new_tokens=1,
    )
    fixed_prefill = fixed_engine.prepare_self_conditioned_prefill(
        prompt,
        max_new_tokens=1,
    )
    assert turn_prefill.segment_mode is SegmentMode.turn_delimited_fixed_tokens
    assert fixed_prefill.segment_mode is SegmentMode.fixed_tokens
    assert turn_prefill.decoder is fixed_engine._model.model
    assert turn_prefill.segment_size == fixed_engine._model.self_cond_segment_size()

    def unexpected_batch_work(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("turn prefill provenance reached optimized batch work")

    monkeypatch.setattr(
        "steering.eval.inference_engine.stack_packed_encoder_outputs",
        unexpected_batch_work,
    )
    monkeypatch.setattr(
        "steering.eval.inference_engine.stack_prefilled_kv_caches",
        unexpected_batch_work,
    )
    monkeypatch.setattr(
        "steering.eval.inference_engine._sample_next_token_from_finite_logits",
        unexpected_batch_work,
    )

    with pytest.raises(
        ValueError,
        match=r"requires fixed_tokens prefill provenance; row 0 has segment_mode=.*turn",
    ):
        fixed_engine.generate_self_conditioned_batch(
            (turn_prefill,),
            seeds=(17,),
            max_new_tokens=1,
            temperature=0.7,
            top_k=20,
            stop_token_ids=set(),
        )

    with pytest.raises(
        ValueError,
        match=r"requires fixed_tokens prefill provenance; row 1 has segment_mode=.*turn",
    ):
        fixed_engine.generate_self_conditioned_batch(
            (fixed_prefill, turn_prefill),
            seeds=(13, 17),
            max_new_tokens=1,
            temperature=0.7,
            top_k=20,
            stop_token_ids=set(),
        )


@pytest.mark.parametrize(
    ("temperature", "top_k"),
    [
        (0.0, None),
        (0.0, 20),
        (0.7, None),
        (0.7, 0),
        (0.7, 1),
        (0.7, 20),
        (0.7, 200),
    ],
)
@pytest.mark.parametrize("batch_size", [3, 6, 24])
def test_independent_rng_self_conditioned_batch_equals_scalar_streams(
    mock_tokenizer: MockTokenizer,
    temperature: float,
    top_k: int | None,
    batch_size: int,
) -> None:
    t.manual_seed(1949)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    seeds = tuple(11 + 18 * row for row in range(batch_size))
    kwargs: dict[str, Any] = {
        "max_new_tokens": 4,
        "temperature": temperature,
        "top_k": top_k,
        "stop_token_ids": set(),
    }
    expected = [
        list(
            _self_conditioned_engine(model, mock_tokenizer).generate_stream(
                prompt,
                seed=seed,
                **kwargs,
            )
        )
        for seed in seeds
    ]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in seeds)
    prefills = tuple(
        engine.prepare_self_conditioned_prefill(
            prompt,
            max_new_tokens=kwargs["max_new_tokens"],
        )
        for engine in engines
    )
    pinned = tuple(
        (
            prefill.kv_cache.k_cache.clone(),
            prefill.kv_cache.v_cache.clone(),
            prefill.runtime.state.encoder_out.values.clone(),
        )
        for prefill in prefills
    )

    actual = engines[-1].generate_self_conditioned_batch(
        prefills,
        seeds=seeds,
        **kwargs,
    )

    assert actual == expected
    for prefill, (expected_k, expected_v, expected_encoder) in zip(
        prefills,
        pinned,
        strict=True,
    ):
        assert t.equal(prefill.kv_cache.k_cache, expected_k)
        assert t.equal(prefill.kv_cache.v_cache, expected_v)
        assert t.equal(prefill.runtime.state.encoder_out.values, expected_encoder)


def test_self_conditioned_batch_stop_rows_keep_rng_and_outputs_isolated(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t.manual_seed(1950)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(3))
    prefills = tuple(
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=4) for engine in engines
    )
    scheduled = iter((9, 5, 6, 9, 7, 9))
    calls: list[int] = []
    generator_calls: dict[int, int] = {}
    observed_generators: dict[int, t.Generator] = {}
    guard_shapes: list[tuple[int, ...]] = []
    emitted_steps: list[list[int]] = []
    decode_calls = 0
    real_forward = model.forward
    real_guard = nanochat_sampling._assert_finite_logits

    def sample(
        logits: t.Tensor,
        rng: t.Generator | None,
        temperature: float,
        top_k: int | None,
    ) -> t.Tensor:
        del temperature, top_k
        assert rng is not None
        seed = int(rng.initial_seed())
        generator_calls[seed] = generator_calls.get(seed, 0) + 1
        observed_generators[seed] = rng
        t.rand((), generator=rng)
        calls.append(int(logits.size(0)))
        return t.tensor([[next(scheduled)]], dtype=t.long, device=logits.device)

    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal decode_calls
        emitted_steps.append(args[0][:, 0].tolist())
        logits, fp_logits = real_forward(*args, **kwargs)
        logits = logits.clone()
        logits[: decode_calls + 1] = float("nan")
        decode_calls += 1
        return logits, fp_logits

    def guard(logits: t.Tensor) -> None:
        guard_shapes.append(tuple(logits.shape))
        real_guard(logits)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(nanochat_sampling, "_assert_finite_logits", guard)
    monkeypatch.setattr(
        "steering.eval.inference_engine._sample_next_token_from_finite_logits",
        sample,
    )
    profile_dir = tmp_path / "self-conditioned-stop-rows"
    recorder = TimingRecorder()
    with AxBenchProfiler(profile_dir, metadata={"test": True}), use_recorder(recorder):
        actual = engines[-1].generate_self_conditioned_batch(
            prefills,
            seeds=(1, 2, 3),
            max_new_tokens=4,
            temperature=0.7,
            top_k=20,
            stop_token_ids={9},
        )

    assert actual == [[], [5], [6, 7]]
    assert calls == [1] * 6
    assert generator_calls == {1: 1, 2: 2, 3: 3}
    for seed, draw_count in generator_calls.items():
        reference = t.Generator().manual_seed(seed)
        for _ in range(draw_count):
            t.rand((), generator=reference)
        assert t.equal(observed_generators[seed].get_state(), reference.get_state())
    vocab_size = int(prefills[0].logits.size(-1))
    assert guard_shapes == [(3, vocab_size), (2, vocab_size), (1, vocab_size)]
    assert decode_calls == 2
    assert emitted_steps == [[9, 5, 6], [0, 9, 7]]
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 6
    assert counters["axbench_base.generation.batched_decode_iterations"] == 3
    timing_counts = _timing_counts(recorder)
    assert timing_counts["generation/sample_next_token_batch"] == 3
    assert timing_counts["generation/sample_next_token"] == 6
    with pytest.raises(StopIteration):
        next(scheduled)


@pytest.mark.parametrize(
    ("temperature", "top_k"),
    [(0.0, 1), (0.7, 1)],
)
def test_self_conditioned_batch_real_greedy_helper_ignores_finished_row_nans(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    temperature: float,
    top_k: int,
) -> None:
    t.manual_seed(1956)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(3))
    prefills = [
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=4) for engine in engines
    ]
    first_tokens = (9, 5, 6)
    for row, token in enumerate(first_tokens):
        logits = t.full_like(prefills[row].logits, -100.0)
        logits[0, token] = 100.0
        prefills[row] = replace(prefills[row], logits=logits)

    real_forward = model.forward
    emitted_steps: list[list[int]] = []
    decode_call = 0

    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal decode_call
        emitted_steps.append(args[0][:, 0].tolist())
        logits, fp_logits = real_forward(*args, **kwargs)
        logits = t.full_like(logits, -100.0)
        if decode_call == 0:
            logits[0] = float("nan")
            logits[1, 0, 9] = 100.0
            logits[2, 0, 7] = 100.0
        elif decode_call == 1:
            logits[:2] = float("nan")
            logits[2, 0, 9] = 100.0
        else:
            raise AssertionError("all active rows should have stopped")
        decode_call += 1
        return logits, fp_logits

    monkeypatch.setattr(model, "forward", forward)
    actual = engines[-1].generate_self_conditioned_batch(
        tuple(prefills),
        seeds=(101, 102, 103),
        max_new_tokens=4,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids={9},
    )

    assert actual == [[], [5], [6, 7]]
    assert emitted_steps == [[9, 5, 6], [0, 9, 7]]
    assert decode_call == 2


@pytest.mark.parametrize("bad_row", [0, 1, 2])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_self_conditioned_batch_nonfinite_preserves_scalar_diagnostic_and_counter(
    mock_tokenizer: MockTokenizer,
    tmp_path: Path,
    bad_row: int,
    bad_value: float,
) -> None:
    t.manual_seed(1955)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(3))
    prefills = [
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=1) for engine in engines
    ]
    bad_logits = prefills[bad_row].logits.clone()
    bad_logits[0, 7] = bad_value
    prefills[bad_row] = replace(prefills[bad_row], logits=bad_logits)

    profile_dir = tmp_path / f"bad-row-{bad_row}"
    expected = rf"\(1/{bad_logits.numel()}; first positions=\[\[0, 7\]\]\)"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        with pytest.raises(FloatingPointError, match=expected) as exc_info:
            engines[-1].generate_self_conditioned_batch(
                tuple(prefills),
                seeds=(11, 12, 13),
                max_new_tokens=1,
                temperature=0.0,
                top_k=None,
                stop_token_ids=set(),
            )

    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == bad_row
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None


def test_self_conditioned_batch_nonfinite_replay_advances_prior_rngs_exactly(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t.manual_seed(1960)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    seeds = (11, 29, 47)
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in seeds)
    prefills = [
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=1) for engine in engines
    ]
    invalid = prefills[-1].logits.clone()
    invalid[0, 7] = float("nan")
    prefills[-1] = replace(prefills[-1], logits=invalid)
    observed_states: dict[int, t.Tensor] = {}

    def record_scalar_replay(
        logits: t.Tensor,
        rng: t.Generator | None,
        temperature: float,
        top_k: int | None,
    ) -> t.Tensor:
        sampled = _real_sample_next_token(logits, rng, temperature, top_k)
        assert rng is not None
        observed_states[int(rng.initial_seed())] = rng.get_state().clone()
        return sampled

    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        record_scalar_replay,
    )
    profile_dir = tmp_path / "stochastic-nonfinite-replay"
    expected = rf"\(1/{invalid.numel()}; first positions=\[\[0, 7\]\]\)"
    recorder = TimingRecorder()
    with AxBenchProfiler(profile_dir, metadata={"test": True}), use_recorder(recorder):
        with pytest.raises(FloatingPointError, match=expected) as exc_info:
            engines[-1].generate_self_conditioned_batch(
                tuple(prefills),
                seeds=seeds,
                max_new_tokens=1,
                temperature=0.7,
                top_k=20,
                stop_token_ids=set(),
            )

    assert set(observed_states) == set(seeds[:2])
    for row, seed in enumerate(seeds[:2]):
        reference = t.Generator().manual_seed(seed)
        _real_sample_next_token(prefills[row].logits, reference, 0.7, 20)
        assert t.equal(observed_states[seed], reference.get_state())
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 2
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1
    timing_counts = _timing_counts(recorder)
    assert timing_counts["generation/sample_next_token_batch"] == 1
    assert timing_counts["generation/sample_next_token"] == 3


@pytest.mark.parametrize("bad_logits", [False, True])
def test_self_conditioned_batch_nan_temperature_precedes_finite_guard(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_logits: bool,
) -> None:
    t.manual_seed(1959)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(3))
    prefills = [
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=1) for engine in engines
    ]
    if bad_logits:
        invalid = prefills[-1].logits.clone()
        invalid[0, 7] = float("nan")
        prefills[-1] = replace(prefills[-1], logits=invalid)
    reference_rng = t.Generator().manual_seed(11)
    with pytest.raises(AssertionError) as reference_exc_info:
        _real_sample_next_token(
            prefills[0].logits,
            reference_rng,
            float("nan"),
            None,
        )
    guard_called = False
    canonical_calls: list[
        tuple[t.Tensor, int, t.Tensor, t.Tensor, float, int | None]
    ] = []

    def observe_guard(logits: t.Tensor) -> None:
        nonlocal guard_called
        del logits
        guard_called = True

    def observe_canonical_sample(
        logits: t.Tensor,
        rng: t.Generator | None,
        temperature: float,
        top_k: int | None,
    ) -> t.Tensor:
        assert rng is not None
        before = rng.get_state().clone()
        try:
            return _real_sample_next_token(logits, rng, temperature, top_k)
        finally:
            canonical_calls.append(
                (
                    logits.clone(),
                    rng.initial_seed(),
                    before,
                    rng.get_state().clone(),
                    temperature,
                    top_k,
                )
            )

    monkeypatch.setattr(nanochat_sampling, "_assert_finite_logits", observe_guard)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        observe_canonical_sample,
    )
    profile_dir = tmp_path / f"nan-temperature-bad-logits-{bad_logits}"
    recorder = TimingRecorder()
    with AxBenchProfiler(profile_dir, metadata={"test": True}), use_recorder(recorder):
        with pytest.raises(
            AssertionError,
            match="temperature must be non-negative",
        ) as exc_info:
            engines[-1].generate_self_conditioned_batch(
                tuple(prefills),
                seeds=(11, 12, 13),
                max_new_tokens=1,
                temperature=float("nan"),
                top_k=None,
                stop_token_ids=set(),
            )

    assert guard_called is False
    assert len(canonical_calls) == 1
    called_logits, called_seed, before_state, after_state, called_temperature, called_top_k = (
        canonical_calls[0]
    )
    assert t.equal(called_logits, prefills[0].logits)
    assert called_seed == 11
    assert t.equal(before_state, after_state)
    assert t.equal(before_state, t.Generator().manual_seed(11).get_state())
    assert called_temperature != called_temperature
    assert called_top_k is None
    assert type(exc_info.value) is type(reference_exc_info.value)
    assert exc_info.value.args == reference_exc_info.value.args
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert reference_exc_info.value.__context__ is None
    assert reference_exc_info.value.__cause__ is None
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 0
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1
    timing_counts = _timing_counts(recorder)
    assert timing_counts["generation/sample_next_token_batch"] == 1
    assert timing_counts["generation/sample_next_token"] == 1


def test_self_conditioned_batch_rejects_canonical_nan_temperature_return(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t.manual_seed(1960)
    model = build_model_with_feature_prediction()
    model.eval()
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(2))
    prefills = tuple(
        engine.prepare_self_conditioned_prefill([2, 3], max_new_tokens=1)
        for engine in engines
    )
    guard_called = False
    canonical_calls: list[tuple[t.Tensor, int, float, int | None]] = []

    def reject_guard(logits: t.Tensor) -> None:
        nonlocal guard_called
        del logits
        guard_called = True
        raise AssertionError("finite guard must remain unreachable")

    def accept_nan_temperature(
        logits: t.Tensor,
        rng: t.Generator | None,
        temperature: float,
        top_k: int | None,
    ) -> t.Tensor:
        assert rng is not None
        canonical_calls.append((logits.clone(), rng.initial_seed(), temperature, top_k))
        return t.zeros((1, 1), dtype=t.long, device=logits.device)

    monkeypatch.setattr(nanochat_sampling, "_assert_finite_logits", reject_guard)
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        accept_nan_temperature,
    )
    profile_dir = tmp_path / "nan-temperature-canonical-return"
    recorder = TimingRecorder()
    with AxBenchProfiler(profile_dir, metadata={"test": True}), use_recorder(recorder):
        with pytest.raises(
            AssertionError,
            match="canonical sampler accepted a non-finite temperature",
        ) as exc_info:
            engines[-1].generate_self_conditioned_batch(
                prefills,
                seeds=(21, 22),
                max_new_tokens=1,
                temperature=float("nan"),
                top_k=17,
                stop_token_ids=set(),
            )

    assert guard_called is False
    assert len(canonical_calls) == 1
    called_logits, called_seed, called_temperature, called_top_k = canonical_calls[0]
    assert t.equal(called_logits, prefills[0].logits)
    assert called_seed == 21
    assert called_temperature != called_temperature
    assert called_top_k == 17
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 0
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1
    timing_counts = _timing_counts(recorder)
    assert timing_counts["generation/sample_next_token_batch"] == 1
    assert timing_counts["generation/sample_next_token"] == 1


def test_self_conditioned_batch_counts_successful_rows_before_sampler_failure(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t.manual_seed(1957)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(3))
    prefills = tuple(
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=1) for engine in engines
    )
    sentinel = RuntimeError("row sampler failed")
    helper_calls = 0

    def fail_second_row(
        logits: t.Tensor,
        rng: t.Generator | None,
        temperature: float,
        top_k: int | None,
    ) -> t.Tensor:
        nonlocal helper_calls
        del rng, temperature, top_k
        helper_calls += 1
        if helper_calls == 2:
            raise sentinel
        return t.zeros((1, 1), dtype=t.long, device=logits.device)

    monkeypatch.setattr(
        "steering.eval.inference_engine._sample_next_token_from_finite_logits",
        fail_second_row,
    )
    profile_dir = tmp_path / "row-sampler-failure"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        with pytest.raises(RuntimeError, match="row sampler failed") as exc_info:
            engines[-1].generate_self_conditioned_batch(
                prefills,
                seeds=(11, 12, 13),
                max_new_tokens=1,
                temperature=0.7,
                top_k=20,
                stop_token_ids=set(),
            )

    assert exc_info.value is sentinel
    assert helper_calls == 2
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 1
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1


def test_self_conditioned_batch_false_positive_batch_guard_is_explicit(
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t.manual_seed(1958)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engines = tuple(_self_conditioned_engine(model, mock_tokenizer) for _ in range(3))
    prefills = tuple(
        engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=1) for engine in engines
    )
    real_guard = nanochat_sampling._assert_finite_logits
    sentinel = FloatingPointError("false-positive batch guard")

    def false_positive_for_batch(logits: t.Tensor) -> None:
        if logits.size(0) > 1:
            raise sentinel
        real_guard(logits)

    monkeypatch.setattr(nanochat_sampling, "_assert_finite_logits", false_positive_for_batch)
    profile_dir = tmp_path / "false-positive-batch-guard"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        with pytest.raises(
            AssertionError,
            match="batched finite guard failed without a bad active row",
        ) as exc_info:
            engines[-1].generate_self_conditioned_batch(
                prefills,
                seeds=(11, 12, 13),
                max_new_tokens=1,
                temperature=0.0,
                top_k=None,
                stop_token_ids=set(),
            )

    assert exc_info.value.__cause__ is sentinel
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.decode_steps"] == 3
    assert counters["axbench_base.generation.batched_decode_iterations"] == 1


def test_self_conditioned_batch_has_one_vector_token_transfer_syntax_per_iteration() -> None:
    source = textwrap.dedent(inspect.getsource(InferenceEngine.generate_self_conditioned_batch))
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    token_loop = next(
        node
        for node in loops
        if isinstance(node.target, ast.Name) and node.target.id == "token_index"
    )
    calls = [node for node in ast.walk(token_loop) if isinstance(node, ast.Call)]
    item_calls = [
        node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "item"
    ]
    tolist_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "tolist"
    ]
    helper_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_sample_next_token_from_finite_logits"
    ]

    assert item_calls == []
    assert len(tolist_calls) == 1
    assert len(helper_calls) == 1
    helper_source = inspect.getsource(_sample_next_token_from_finite_logits)
    assert ".item(" not in helper_source
    assert ".tolist(" not in helper_source


def test_self_conditioned_batch_rejects_boundary_and_nonboundary_snapshots(
    mock_tokenizer: MockTokenizer,
) -> None:
    t.manual_seed(1951)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engine = _self_conditioned_engine(model, mock_tokenizer)
    prefill = engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=5)

    with pytest.raises(ValueError, match="cannot cross a conditioning boundary"):
        engine.generate_self_conditioned_batch(
            (prefill,),
            seeds=(1,),
            max_new_tokens=5,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
        )

    invalid = replace(
        prefill,
        runtime=type(prefill.runtime)(
            state=prefill.runtime.state,
            fp_buffer=(t.zeros(1, 1, 1),),
            tokens_in_seg=1,
        ),
    )
    with pytest.raises(ValueError, match="boundary prefills"):
        engine.generate_self_conditioned_batch(
            (invalid,),
            seeds=(1,),
            max_new_tokens=4,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
        )


def test_self_conditioned_batch_rejects_incompatible_prefill_payloads(
    mock_tokenizer: MockTokenizer,
) -> None:
    t.manual_seed(1953)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    engine = _self_conditioned_engine(model, mock_tokenizer)
    prefill = engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=4)

    def generate(rows: tuple[SelfConditionedPrefill, ...]) -> None:
        engine.generate_self_conditioned_batch(
            rows,
            seeds=tuple(range(len(rows))),
            max_new_tokens=4,
            temperature=0.0,
            top_k=None,
            stop_token_ids=set(),
        )

    bad_vocab = replace(prefill, logits=prefill.logits[:, :-1])
    with pytest.raises(ValueError, match="one vocabulary and dtype"):
        generate((prefill, bad_vocab))

    bad_slot = replace(
        prefill,
        runtime=type(prefill.runtime)(
            state=type(prefill.runtime.state)(
                encoder_out=prefill.runtime.state.encoder_out,
                target_slot_idx=prefill.runtime.state.encoder_out.layout.n_segments,
            ),
            fp_buffer=(),
            tokens_in_seg=0,
        ),
    )
    with pytest.raises(ValueError, match="target slot"):
        generate((bad_slot,))

    short_cache = KVCache(
        batch_size=1,
        num_heads=prefill.kv_cache.n_heads,
        seq_len=int(prefill.kv_cache.get_pos()) + 3,
        head_dim=prefill.kv_cache.head_dim,
        num_layers=prefill.kv_cache.n_layers,
        device=t.device("cpu"),
        dtype=prefill.kv_cache.k_cache.dtype,
    )
    short_cache.advance(prefill.kv_cache.get_pos())
    bad_capacity = replace(prefill, kv_cache=short_cache)
    with pytest.raises(ValueError, match="insufficient decode capacity"):
        generate((bad_capacity,))


@t.inference_mode()
def test_teacher_forced_batched_direction_decode_matches_every_scalar_state(
    mock_tokenizer: MockTokenizer,
) -> None:
    t.manual_seed(1952)
    model = build_model_with_feature_prediction()
    model.eval()
    prompt = [2, 3]
    directions = (t.randn(model.dec_config.n_embd), t.randn(model.dec_config.n_embd))
    coefficients = (-1.25, 2.5)
    forced_rows = ((5, 7, 9), (6, 8, 10))

    def prepared_rows() -> tuple[tuple[InferenceEngine, SelfConditionedPrefill], ...]:
        result: list[tuple[InferenceEngine, SelfConditionedPrefill]] = []
        for direction, coefficient in zip(directions, coefficients, strict=True):
            engine = _self_conditioned_engine(model, mock_tokenizer)
            handle = register_residual_hook(model, 1, direction, coefficient)
            try:
                prefill = engine.prepare_self_conditioned_prefill(prompt, max_new_tokens=4)
            finally:
                handle.remove()
            result.append((engine, prefill))
        return tuple(result)

    scalar_rows = prepared_rows()
    scalar_logits: list[list[t.Tensor]] = [[], []]
    scalar_fp_logits: list[list[t.Tensor]] = [[], []]
    scalar_k_caches: list[list[t.Tensor]] = [[], []]
    scalar_v_caches: list[list[t.Tensor]] = [[], []]
    for row, ((engine, prefill), direction, coefficient, forced) in enumerate(
        zip(scalar_rows, directions, coefficients, forced_rows, strict=True)
    ):
        handle = register_residual_hook(model, 1, direction, coefficient)
        try:
            for token in forced:
                logits = engine._model(
                    t.tensor([[token]], dtype=t.long),
                    kv_cache=prefill.kv_cache,
                )
                scalar_logits[row].append(logits)
                snapshot = engine._model.self_cond_runtime_snapshot()
                scalar_fp_logits[row].append(snapshot.fp_buffer[-1])
                scalar_k_caches[row].append(prefill.kv_cache.k_cache.clone())
                scalar_v_caches[row].append(prefill.kv_cache.v_cache.clone())
        finally:
            handle.remove()

    batch_rows = prepared_rows()
    batch_prefills = tuple(prefill for _engine, prefill in batch_rows)
    batch_cache = stack_prefilled_kv_caches(tuple(prefill.kv_cache for prefill in batch_prefills))
    batch_encoder = stack_packed_encoder_outputs(
        tuple(prefill.runtime.state.encoder_out for prefill in batch_prefills)
    )
    seg_map = t.tensor(
        [[prefill.runtime.state.target_slot_idx] for prefill in batch_prefills],
        dtype=t.long,
    )
    direction_matrix = t.stack(directions, dim=0)
    handle = register_batched_residual_hook(model, 1, direction_matrix, coefficients)
    try:
        for step in range(3):
            emitted = t.tensor(
                [[forced_rows[row][step]] for row in range(2)],
                dtype=t.long,
            )
            batched_logits, batched_fp_logits = model(
                emitted,
                None,
                seg_map=seg_map,
                kv_cache=batch_cache,
                encoder_out=batch_encoder,
                encoder_strict=True,
                return_fp_logits=True,
            )
            for row in range(2):
                lm_delta = (
                    batched_logits[row : row + 1] - scalar_logits[row][step]
                ).abs()
                fp_delta = (
                    batched_fp_logits[row : row + 1] - scalar_fp_logits[row][step]
                ).abs()
                assert float(lm_delta.max()) <= 2e-4
                assert float(fp_delta.max()) <= 2e-4
                t.testing.assert_close(
                    batch_cache.k_cache[:, row : row + 1],
                    scalar_k_caches[row][step],
                    rtol=0.0,
                    atol=3e-2,
                )
                t.testing.assert_close(
                    batch_cache.v_cache[:, row : row + 1],
                    scalar_v_caches[row][step],
                    rtol=0.0,
                    atol=3e-2,
                )
    finally:
        handle.remove()


def test_zero_budget_returns_empty_rows_without_decode_forwards(
    mock_tokenizer,
) -> None:
    engine, model = _uniform_engine(mock_tokenizer)
    actual = engine.generate_samples(
        [2, 3],
        seed=7,
        n_samples=3,
        max_new_tokens=0,
        temperature=1.0,
    )
    assert actual == [[], [], []]
    assert model.prefill_calls == 1
    assert model.decode_batch_sizes == []


@t.inference_mode()
def _teacher_forced_batch_shape_drifts(
    engine: InferenceEngine,
    prompt: list[int],
    forced_rows: list[list[int]],
) -> list[list[float]]:
    """Measure B=1 versus B=N logits on the same forced prefixes."""
    batch_size = len(forced_rows)
    budget = len(forced_rows[0])
    assert all(len(row) == budget for row in forced_rows)
    total_len = len(prompt) + budget
    model = engine._model
    device = engine._device

    def prefill(cache):
        context = (
            model.conditioning_context(context_token_ids=[prompt])
            if getattr(model, "is_target_aware_encdec", False)
            else nullcontext()
        )
        with context:
            return model(
                t.tensor([prompt], dtype=t.long, device=device),
                kv_cache=cache,
            )[:, -1, :]

    single_caches = [engine._make_kv_cache(1, total_len) for _ in forced_rows]
    single_logits = [prefill(cache) for cache in single_caches]
    prefill_cache = engine._make_kv_cache(1, total_len)
    batch_logits = prefill(prefill_cache).expand(batch_size, -1)
    batch_cache = engine._make_kv_cache(batch_size, total_len)
    batch_cache.prefill(prefill_cache)

    drifts = [[] for _ in forced_rows]
    for step in range(budget):
        for row in range(batch_size):
            delta = (single_logits[row] - batch_logits[row : row + 1]).abs().max()
            drifts[row].append(float(delta.item()))
        if step + 1 == budget:
            break
        for row in range(batch_size):
            single_logits[row] = model(
                t.tensor(
                    [[forced_rows[row][step]]],
                    dtype=t.long,
                    device=device,
                ),
                kv_cache=single_caches[row],
            )[:, -1, :]
        batch_logits = model(
            t.tensor(
                [forced_rows[row][step] for row in range(batch_size)],
                dtype=t.long,
                device=device,
            ).unsqueeze(1),
            kv_cache=batch_cache,
        )[:, -1, :]
    return drifts


@t.inference_mode()
def _teacher_forced_post_compaction_drifts(
    engine: InferenceEngine,
    prompt: list[int],
    forced_rows: list[list[int]],
) -> list[list[float]]:
    """Measure survivor logits after a forced first token and B=3->B=N copy."""
    batch_size = len(forced_rows)
    budget = len(forced_rows[0])
    assert budget >= 2
    assert all(len(row) == budget for row in forced_rows)
    total_len = len(prompt) + budget
    model = engine._model
    device = engine._device

    def prefill(cache):
        context = (
            model.conditioning_context(context_token_ids=[prompt])
            if getattr(model, "is_target_aware_encdec", False)
            else nullcontext()
        )
        with context:
            return model(
                t.tensor([prompt], dtype=t.long, device=device),
                kv_cache=cache,
            )[:, -1, :]

    single_caches = [engine._make_kv_cache(1, total_len) for _ in forced_rows]
    single_logits = [prefill(cache) for cache in single_caches]
    prefill_cache = engine._make_kv_cache(1, total_len)
    prefill(prefill_cache)
    batch_cache = engine._make_kv_cache(3, total_len)
    batch_cache.prefill(prefill_cache)
    keep_indices = t.arange(1, 1 + batch_size, device=device)
    batch_cache = engine._compact_kv_cache(batch_cache, keep_indices)

    for row in range(batch_size):
        single_logits[row] = model(
            t.tensor([[forced_rows[row][0]]], dtype=t.long, device=device),
            kv_cache=single_caches[row],
        )[:, -1, :]
    batch_logits = model(
        t.tensor(
            [row[0] for row in forced_rows],
            dtype=t.long,
            device=device,
        ).unsqueeze(1),
        kv_cache=batch_cache,
    )[:, -1, :]

    drifts = [[] for _ in forced_rows]
    for step in range(1, budget):
        for row in range(batch_size):
            delta = (single_logits[row] - batch_logits[row : row + 1]).abs().max()
            drifts[row].append(float(delta.item()))
        if step + 1 == budget:
            break
        for row in range(batch_size):
            single_logits[row] = model(
                t.tensor([[forced_rows[row][step]]], dtype=t.long, device=device),
                kv_cache=single_caches[row],
            )[:, -1, :]
        batch_logits = model(
            t.tensor(
                [row[step] for row in forced_rows],
                dtype=t.long,
                device=device,
            ).unsqueeze(1),
            kv_cache=batch_cache,
        )[:, -1, :]
    return drifts


def _assert_cuda_exact_or_bounded(
    engine: InferenceEngine,
    prompt: list[int],
    expected: list[list[int]],
    actual: list[list[int]],
    *,
    label: str,
    post_compaction: bool = False,
) -> None:
    if actual == expected:
        print(f"CUDA bf16 {label}: token streams bitwise exact")
        return

    drift_rows = (
        _teacher_forced_post_compaction_drifts(engine, prompt, expected)
        if post_compaction
        else _teacher_forced_batch_shape_drifts(engine, prompt, expected)
    )
    divergences: list[tuple[int, int, float]] = []
    for row, (reference, observed) in enumerate(zip(expected, actual, strict=True)):
        if reference == observed:
            continue
        first = next(
            index
            for index, (left, right) in enumerate(zip(reference, observed, strict=True))
            if left != right
        )
        drift_index = first - 1 if post_compaction else first
        divergences.append((row, first, drift_rows[row][drift_index]))
    max_delta = max(delta for _row, _step, delta in divergences)
    print(
        f"CUDA bf16 {label}: first_divergences={divergences}, max_abs_logit_delta={max_delta:.6g}"
    )
    # Compaction changes the physical batch size mid-decode. As documented by
    # issue #1054, bf16 kernel tiling can depend on B; the old constant-size
    # lockstep path never changed B and could therefore require bitwise output.
    assert max_delta <= 5e-2


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("target", ["decoder", "wrapped_encdec"])
def test_cuda_bf16_greedy_batch_is_exact_or_bounded(  # pragma: no cover
    target: str,
) -> None:
    device = t.device("cuda")
    tokenizer = MockTokenizer()
    if target == "decoder":
        model = _make_gpt(device="cuda").to(dtype=t.bfloat16)
        engine = InferenceEngine(model, tokenizer, device)
    else:
        model = _make_encdec(device="cuda").to(dtype=t.bfloat16)
        engine = build_engine_for_features(
            model,
            tokenizer,
            device,
            features={5: 2},
            context_provider=_FullPrefixContextProvider(tokenizer),
        )
    prompt = _prompt(tokenizer)
    kwargs: dict[str, Any] = {
        "max_new_tokens": 32,
        "temperature": 0.0,
        "top_k": None,
        "stop_token_ids": set(),
    }
    expected = _greedy_sequential_samples(engine, prompt, 4, **kwargs)
    actual = engine.generate_samples(prompt, seed=None, n_samples=4, **kwargs)
    _assert_cuda_exact_or_bounded(
        engine,
        prompt,
        expected,
        actual,
        label=target,
    )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("target", ["decoder", "wrapped_encdec"])
def test_cuda_forced_divergence_compaction_is_exact_or_bounded(  # pragma: no cover
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    device = t.device("cuda")
    tokenizer = MockTokenizer()
    if target == "decoder":
        model: Any = _make_gpt(device="cuda").to(dtype=t.bfloat16)
    else:
        target_segments = MultiTargetSegments(
            segments=[{5: 1}, {6: 2}, {7: 3}],
            token_assignments=t.tensor([0, 1, 1, 2], dtype=t.int64),
        )
        model = build_wrapper_for_features(
            _make_encdec(device="cuda").to(dtype=t.bfloat16),
            tokenizer,
            target=target_segments,
            context_provider=_FullPrefixContextProvider(tokenizer),
        )
    recorder = _ForwardRecorder(model)
    engine = InferenceEngine(recorder, tokenizer, device)
    prompt = _prompt(tokenizer)
    budget = 8
    kwargs: dict[str, Any] = {
        "max_new_tokens": budget,
        "temperature": 0.0,
        "top_k": None,
        "stop_token_ids": {-1},
    }
    first_tokens = [-1, 1, 2]
    expected: list[list[int]] = []
    for first_token in first_tokens:
        monkeypatch.setattr(
            "steering.eval.inference_engine.sample_next_token",
            _ForcedFirstThenGreedy([first_token]),
        )
        expected.append(engine.generate_samples(prompt, seed=None, n_samples=1, **kwargs)[0])

    recorder.reset()
    monkeypatch.setattr(
        "steering.eval.inference_engine.sample_next_token",
        _ForcedFirstThenGreedy(first_tokens),
    )
    profile_dir = tmp_path / f"cuda-forced-{target}"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = engine.generate_samples(prompt, seed=None, n_samples=3, **kwargs)

    assert actual[0] == []
    assert recorder.prefill_batch_sizes == [1]
    assert recorder.decode_batch_sizes == [2] * (budget - 1)
    assert _counter_totals(profile_dir)["axbench_base.generation.batch_compactions"] >= 1
    assert "axbench_base.generation.compact_decode_batch" in _region_names(profile_dir)
    _assert_cuda_exact_or_bounded(
        engine,
        prompt,
        expected[1:],
        actual[1:],
        label=f"forced-compaction-{target}",
        post_compaction=True,
    )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_cuda_sampled_batch_is_deterministic() -> None:  # pragma: no cover
    device = t.device("cuda")
    tokenizer = MockTokenizer()
    engine = InferenceEngine(
        _make_gpt(device="cuda").to(dtype=t.bfloat16),
        tokenizer,
        device,
    )
    kwargs: dict[str, Any] = {
        "seed": 123,
        "n_samples": 4,
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_k": 32,
        "stop_token_ids": set(),
    }
    assert engine.generate_samples(_prompt(tokenizer), **kwargs) == engine.generate_samples(
        _prompt(tokenizer), **kwargs
    )
