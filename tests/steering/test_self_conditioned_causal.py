"""Observable causal and failure-isolation tests for self-conditioning."""

from __future__ import annotations

import re
from collections.abc import Generator
from types import SimpleNamespace

import pytest
import torch as t
from torch.profiler import ProfilerActivity, profile

from steering.conditional.encoder_decoder.packed_encoder import (
    append_packed_encoder_segment,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondState,
    final_real_prompt_source_mask,
    self_conditioned_decode_step,
)
from steering.eval.inference_engine import InferenceEngine
from steering.segmentation import SegmentMode
from tests.padded_encoder_reference import pack_dense_encoder_output_reference


class _Cache:
    def __init__(self, length: int = 8) -> None:
        self.pos = 0
        self.k = t.full((length,), -1, dtype=t.int64)
        self.v = t.full((length,), -1, dtype=t.int64)

    def get_pos(self) -> int:
        return self.pos

    def advance(self, amount: int) -> None:
        self.pos += amount

    def rewind(self, amount: int) -> None:
        self.pos -= amount


class _Provider:
    def __init__(
        self,
        K: int,
        *,
        fail: bool = False,
        segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    ) -> None:
        self.cfg = SimpleNamespace(self_cond_segment_size=K, segment_mode=segment_mode)
        self.fail = fail
        self.calls: list[t.Tensor] = []

    def step(self, state: SelfCondState, fp_logits_tail: t.Tensor) -> SelfCondState:
        self.calls.append(fp_logits_tail.clone())
        if self.fail:
            raise RuntimeError("provider failed")
        new_segment = pack_dense_encoder_output_reference(
            t.zeros((1, 1, 1, 1)),
            t.ones((1, 1, 1), dtype=t.bool),
        )
        return SelfCondState(
            encoder_out=append_packed_encoder_segment(
                state.encoder_out,
                new_segment,
            ),
            target_slot_idx=state.target_slot_idx + 1,
        )


class _SlotModel:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[int, str, int]] = []

    def __call__(self, token: t.Tensor, _targets=None, **kwargs):
        cache = kwargs["kv_cache"]
        slot = int(kwargs["seg_map"].item())
        mode = (
            "fp_only"
            if kwargs.get("fp_logits_only")
            else "both"
            if kwargs.get("return_fp_logits")
            else "lm"
        )
        position = cache.get_pos()
        self.calls.append((slot, mode, position))
        cache.k[position] = 10 * slot + 1
        cache.v[position] = 10 * slot + 2
        cache.advance(1)
        if self.fail_on == mode:
            raise RuntimeError(f"{mode} failed")
        fp = t.full((1, 1, 3), float(slot))
        logits = t.full((1, 1, 7), float(slot))
        if mode == "fp_only":
            return fp
        if mode == "both":
            return logits, fp
        return logits


def _state(slot: int = 5) -> SelfCondState:
    return SelfCondState(
        encoder_out=pack_dense_encoder_output_reference(
            t.zeros((1, slot + 1, 1, 1)),
            t.ones((1, slot + 1, 1), dtype=t.bool),
        ),
        target_slot_idx=slot,
    )


def _turn_provider(K: int, *, fail: bool = False) -> _Provider:
    """A provider whose config selects the turn-delimited decode contract."""

    return _Provider(K, fail=fail, segment_mode=SegmentMode.turn_delimited_fixed_tokens)


def _exact_message(message: str) -> str:
    """Anchor a ``pytest.raises`` match so a longer or shorter message cannot
    satisfy it: several of these guards differ only by a slot number or a
    trailing clause."""

    return rf"^{re.escape(message)}$"


def test_final_real_prompt_source_mask_selects_all_and_only_final_slot_rows() -> None:
    seg_map = t.tensor([[0, 0, 1, 2, 2, 2, 3]])
    mask = final_real_prompt_source_mask(seg_map, encoder_slot_count=4)
    assert mask.tolist() == [False, False, False, True, True, True]


# Turn prefill drops the marker's real feature slot before packing, so the
# marker column keeps ``_build_conditioning``'s zero fill and must pass the
# boundary check as an ordinary slot.
# kills: rejecting (or special-casing) a boundary column that repeats slot 0.
def test_prompt_source_mask_accepts_a_zero_filled_marker_column() -> None:
    seg_map = t.tensor([[0, 0, 1, 2, 2, 0]])
    mask = final_real_prompt_source_mask(seg_map, encoder_slot_count=3)
    assert mask.tolist() == [False, False, False, True, True]


# kills: deleting any of the shape/length/slot-count/boundary/source guards,
# and reusing one guard's wording or slot bound for another.
@pytest.mark.parametrize(
    ("seg_map", "slots", "error", "message"),
    [
        (
            t.zeros((1, 1, 2), dtype=t.long),
            2,
            ValueError,
            "self-conditioned prompt seg_map must have shape (1, T), got (1, 1, 2)",
        ),
        (
            t.tensor([[0, 0], [0, 0]]),
            2,
            ValueError,
            "self-conditioned prompt seg_map must have shape (1, T), got (2, 2)",
        ),
        (
            t.tensor([[0]]),
            2,
            ValueError,
            "self-conditioned prefill requires at least two prompt tokens",
        ),
        (
            t.tensor([[0, 0]]),
            0,
            ValueError,
            "self-conditioned prefill requires at least one real prompt slot",
        ),
        (
            t.tensor([[0, 2]]),
            2,
            RuntimeError,
            "the final self-conditioned prompt position must route to a real "
            "prompt slot in [0, 2); got 2",
        ),
        (
            t.tensor([[3, 0]]),
            2,
            RuntimeError,
            "the final self-conditioning source slot must be a real prompt slot in [0, 2); got 3",
        ),
    ],
)
def test_final_real_prompt_source_mask_rejects_malformed_routing(
    seg_map: t.Tensor,
    slots: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=_exact_message(message)):
        final_real_prompt_source_mask(seg_map, encoder_slot_count=slots)


def test_k2_s0_owns_distributions_one_and_two_and_s1_owns_three() -> None:
    cache = _Cache()
    model = _SlotModel()
    provider = _Provider(2)
    state = _state()

    first = self_conditioned_decode_step(
        model=model,
        provider=provider,
        state=state,
        fp_buffer=(),
        tokens_in_seg=0,
        emitted_token=t.tensor([[11]]),
        kv_cache=cache,
        device=t.device("cpu"),
    )
    assert first.logits.unique().item() == 5
    assert first.state.target_slot_idx == 5

    second = self_conditioned_decode_step(
        model=model,
        provider=provider,
        state=first.state,
        fp_buffer=first.fp_buffer,
        tokens_in_seg=first.tokens_in_seg,
        emitted_token=t.tensor([[12]]),
        kv_cache=cache,
        device=t.device("cpu"),
    )
    assert second.logits.unique().item() == 6
    assert second.state.target_slot_idx == 6
    assert provider.calls[0].shape == (1, 2, 3)
    assert provider.calls[0].unique().item() == 5
    assert model.calls == [(5, "both", 0), (5, "fp_only", 1), (6, "lm", 1)]
    assert cache.pos == 2


@pytest.mark.parametrize("K", [1, 2, 3, 256])
@pytest.mark.parametrize("sample_count", [0, 1, 2, 3, 4, 5, 255, 256, 257])
def test_turn_distribution_and_measurement_ownership_matrix(
    K: int,
    sample_count: int,
) -> None:
    """The marker anchors row one and no final sampled token is consumed."""

    cache = _Cache(length=max(8, sample_count + 1))
    model = _SlotModel()
    provider = _turn_provider(K)
    state = _state()
    fp_buffer: tuple[t.Tensor, ...] = ()
    tokens_in_seg = 0
    distribution_slots: list[int] = []

    # A zero-token generation skips prefill entirely. Otherwise the terminal
    # assistant marker is the first consumed input; only the first N-1 sampled
    # tokens are then forwarded because the final draw controls no successor.
    consumed_inputs = [] if sample_count == 0 else [66, *range(1, sample_count)]
    for token_id in consumed_inputs:
        result = self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=fp_buffer,
            tokens_in_seg=tokens_in_seg,
            emitted_token=t.tensor([[token_id]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )
        distribution_slots.append(int(result.logits.unique().item()))
        state = result.state
        fp_buffer = result.fp_buffer
        tokens_in_seg = result.tokens_in_seg

    assert distribution_slots == [
        5 + distribution_index // K for distribution_index in range(sample_count)
    ]
    assert len(model.calls) == sample_count
    assert len(provider.calls) == sample_count // K
    assert all(rows.shape == (1, K, 3) for rows in provider.calls)
    # Every input is forwarded exactly once, under its own outgoing slot, and
    # the rows handed to the provider carry that same slot's FP value.
    # kills: replaying the Kth input, publishing the incoming state before the
    # Kth forward, and attributing buffered FP rows to the wrong slot.
    assert [mode for _slot, mode, _position in model.calls] == ["both"] * sample_count
    assert [slot for slot, _mode, _position in model.calls] == distribution_slots
    assert [int(rows.unique().item()) for rows in provider.calls] == [
        5 + window for window in range(len(provider.calls))
    ]
    assert state.target_slot_idx == 5 + sample_count // K
    assert tokens_in_seg == sample_count % K
    assert len(fp_buffer) == tokens_in_seg
    assert cache.pos == sample_count


def test_turn_boundary_provider_failure_rewinds_single_committed_forward() -> None:
    cache = _Cache()
    model = _SlotModel()
    provider = _turn_provider(1, fail=True)
    state = _state()

    with pytest.raises(RuntimeError, match="provider failed"):
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[1]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    assert model.calls == [(5, "both", 0)]
    assert cache.pos == 0
    assert state.target_slot_idx == 5


def test_turn_model_failure_rewinds_without_calling_provider() -> None:
    cache = _Cache()
    model = _SlotModel(fail_on="both")
    provider = _turn_provider(1)
    state = _state()

    with pytest.raises(RuntimeError, match="both failed"):
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[1]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    assert model.calls == [(5, "both", 0)]
    assert provider.calls == []
    assert cache.pos == 0
    assert state.target_slot_idx == 5


class _MisadvancingModel(_SlotModel):
    """A model whose forward leaves the KV cache off by one position."""

    def __init__(self, *, advance: int) -> None:
        super().__init__()
        self._advance = advance

    def __call__(self, token: t.Tensor, _targets=None, **kwargs):
        output = super().__call__(token, _targets, **kwargs)
        kwargs["kv_cache"].advance(self._advance - 1)
        return output


# One decode step serves both conditioning modes; these pin every branch of its
# shared validation preamble, which runs before any mode-specific work.
# kills: deleting any of the emitted_token/tokens_in_seg/FP-buffer guards, and
# doing model, KV, or provider work before they run.
@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"emitted_token": t.tensor([[1, 2]])},
            ValueError,
            "self-conditioned emitted_token must have shape (1, 1), got (1, 2)",
        ),
        (
            {"emitted_token": t.tensor([1])},
            ValueError,
            "self-conditioned emitted_token must have shape (1, 1), got (1,)",
        ),
        ({"tokens_in_seg": True}, TypeError, "tokens_in_seg must be an integer"),
        ({"tokens_in_seg": 2}, ValueError, "tokens_in_seg must be in [0, 2), got 2"),
        ({"tokens_in_seg": -1}, ValueError, "tokens_in_seg must be in [0, 2), got -1"),
        (
            {"tokens_in_seg": 1},
            ValueError,
            "self-conditioned FP-buffer cardinality must equal tokens_in_seg: buffer=0, counter=1",
        ),
        (
            {"fp_buffer": (t.zeros((1, 1, 3)),)},
            ValueError,
            "self-conditioned FP-buffer cardinality must equal tokens_in_seg: buffer=1, counter=0",
        ),
    ],
)
def test_decode_step_preamble_rejects_an_inconsistent_runtime_tuple(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    cache = _Cache()
    model = _SlotModel()
    provider = _Provider(2)
    arguments: dict[str, object] = {
        "model": model,
        "provider": provider,
        "state": _state(),
        "fp_buffer": (),
        "tokens_in_seg": 0,
        "emitted_token": t.tensor([[1]]),
        "kv_cache": cache,
        "device": t.device("cpu"),
    }
    arguments.update(overrides)

    with pytest.raises(error, match=_exact_message(message)):
        self_conditioned_decode_step(**arguments)

    assert model.calls == []
    assert provider.calls == []
    assert cache.pos == 0


# kills: dropping the preamble's empty-outgoing-state check, or running it
# after the first model forward.
def test_decode_step_preamble_rejects_an_empty_outgoing_state() -> None:
    cache = _Cache()
    model = _SlotModel()
    provider = _Provider(2)
    state = _state()
    state.encoder_out.layout.valid_mask[
        state.encoder_out.layout.segment_indices == state.target_slot_idx
    ] = False
    message = (
        "self-conditioned decode received a SelfCondState containing empty encoder "
        "slots [5]; refusing out-of-distribution conditioning"
    )

    with pytest.raises(EmptySelfConditioningSegmentError, match=_exact_message(message)):
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[11]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    assert model.calls == []
    assert cache.pos == 0


# kills: dropping the one-position KV-advance assertion, or rewinding to a
# hard-coded 0 instead of the observed start position.
@pytest.mark.parametrize("advance", [0, 2])
def test_decode_step_rejects_a_forward_that_moves_the_cache_by_other_than_one(
    advance: int,
) -> None:
    cache = _Cache()
    cache.advance(3)
    model = _MisadvancingModel(advance=advance)
    provider = _Provider(2)
    state = _state()
    message = "self-conditioned decode forward must advance the KV cache by exactly one position"

    with pytest.raises(RuntimeError, match=_exact_message(message)):
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[1]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    assert cache.pos == 3
    assert provider.calls == []
    assert state.target_slot_idx == 5


# kills: rewinding to 0 rather than to the observed start position, and
# publishing (or mutating) any runtime field when provider.step raises.
def test_turn_provider_failure_rewinds_exactly_one_position_from_a_nonzero_start() -> None:
    cache = _Cache()
    cache.advance(4)
    model = _SlotModel()
    provider = _turn_provider(2, fail=True)
    state = _state()
    buffered = t.full((1, 1, 3), 5.0)

    with pytest.raises(RuntimeError, match="provider failed"):
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(buffered,),
            tokens_in_seg=1,
            emitted_token=t.tensor([[7]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    assert model.calls == [(5, "both", 4)]
    assert cache.pos == 4
    assert len(provider.calls) == 1
    assert provider.calls[0].shape == (1, 2, 3)
    assert state.target_slot_idx == 5
    assert t.equal(buffered, t.full((1, 1, 3), 5.0))


def test_decode_phase_markers_cover_non_boundary_and_boundary_paths() -> None:
    cache = _Cache()
    model = _SlotModel()
    provider = _Provider(2)
    state = _state()

    with profile(activities=[ProfilerActivity.CPU], with_stack=False) as profiler:
        first = self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[11]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=first.state,
            fp_buffer=first.fp_buffer,
            tokens_in_seg=first.tokens_in_seg,
            emitted_token=t.tensor([[12]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    names = {event.name for event in profiler.events()}
    assert {
        "selfcond/decode/non_boundary_forward",
        "selfcond/decode/boundary_measurement_forward",
        "selfcond/decode/provider_step",
        "selfcond/decode/boundary_commit_forward",
    } <= names


def test_turn_decode_markers_separate_the_window_completing_forward() -> None:
    # kills: reusing one marker name for both turn forwards, which would make
    # the window-completing forward indistinguishable from an ordinary one.
    cache = _Cache()
    model = _SlotModel()
    provider = _turn_provider(2)

    with profile(activities=[ProfilerActivity.CPU], with_stack=False) as profiler:
        first = self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=_state(),
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[11]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=first.state,
            fp_buffer=first.fp_buffer,
            tokens_in_seg=first.tokens_in_seg,
            emitted_token=t.tensor([[12]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )

    names = {event.name for event in profiler.events()}
    assert {
        "selfcond/turn_decode/forward",
        "selfcond/turn_decode/boundary_forward",
        "selfcond/turn_decode/provider_step",
    } <= names
    assert "selfcond/decode/non_boundary_forward" not in names


def test_decode_rejects_empty_incoming_state_before_commit_forward() -> None:
    class _EmptyIncomingProvider(_Provider):
        def step(self, state, fp_logits_tail):
            return SelfCondState(
                encoder_out=pack_dense_encoder_output_reference(
                    t.zeros((1, state.encoder_out.layout.n_segments, 1, 1)),
                    t.zeros(
                        (1, state.encoder_out.layout.n_segments, 1),
                        dtype=t.bool,
                    ),
                ),
                target_slot_idx=state.target_slot_idx,
            )

    cache = _Cache()
    model = _SlotModel()
    with pytest.raises(EmptySelfConditioningSegmentError, match="empty encoder slots"):
        self_conditioned_decode_step(
            model=model,
            provider=_EmptyIncomingProvider(1),
            state=_state(),
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[11]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )
    assert model.calls == [(5, "fp_only", 0)]
    assert cache.pos == 0


@pytest.mark.parametrize(
    ("K", "emitted_count"),
    [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 6),
        (3, 7),
        (256, 0),
        (256, 1),
        (256, 255),
        (256, 256),
        (256, 257),
        (256, 512),
    ],
)
def test_distribution_and_measurement_ownership_matrix(
    K: int,
    emitted_count: int,
) -> None:
    """Only emitted tokens followed by another distribution are forwarded."""

    cache = _Cache(length=max(8, emitted_count + 1))
    model = _SlotModel()
    provider = _Provider(K)
    state = _state()
    fp_buffer: tuple[t.Tensor, ...] = ()
    tokens_in_seg = 0
    distribution_slots = [] if emitted_count == 0 else [state.target_slot_idx]

    # The final emitted token is never forwarded: it controls no later sampled
    # distribution and therefore cannot construct an unused state.
    for emitted_index in range(max(0, emitted_count - 1)):
        result = self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=fp_buffer,
            tokens_in_seg=tokens_in_seg,
            emitted_token=t.tensor([[emitted_index + 1]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )
        state = result.state
        fp_buffer = result.fp_buffer
        tokens_in_seg = result.tokens_in_seg
        distribution_slots.append(int(result.logits.unique().item()))

    expected_distribution_slots = [5 + index // K for index in range(emitted_count)]
    assert distribution_slots == expected_distribution_slots
    assert len(provider.calls) == max(0, emitted_count - 1) // K
    assert (0 if emitted_count == 0 else 1 + len(provider.calls)) == (
        0 if emitted_count == 0 else (emitted_count + K - 1) // K
    )
    assert tokens_in_seg == max(0, emitted_count - 1) % K
    assert len(fp_buffer) == tokens_in_seg
    assert cache.pos == max(0, emitted_count - 1)

    measured_slots = [slot for slot, mode, _position in model.calls if mode != "lm"]
    assert measured_slots == [
        5 + index // K for index in range(max(0, emitted_count - 1))
    ]


def test_boundary_commit_overwrites_speculative_kv_exactly() -> None:
    outgoing = _state()
    provider = _Provider(1)
    cache = _Cache()
    model = _SlotModel()
    result = self_conditioned_decode_step(
        model=model,
        provider=provider,
        state=outgoing,
        fp_buffer=(),
        tokens_in_seg=0,
        emitted_token=t.tensor([[9]]),
        kv_cache=cache,
        device=t.device("cpu"),
    )

    reference_cache = _Cache()
    reference_model = _SlotModel()
    incoming = result.state
    reference_model(
        t.tensor([[9]]),
        None,
        seg_map=t.tensor([[incoming.target_slot_idx]]),
        kv_cache=reference_cache,
        encoder_out=incoming.encoder_out,
        encoder_strict=True,
    )
    assert t.equal(cache.k, reference_cache.k)
    assert t.equal(cache.v, reference_cache.v)
    assert cache.pos == reference_cache.pos == 1
    assert int(cache.k[0]) != 10 * outgoing.target_slot_idx + 1


@pytest.mark.parametrize("failure", ["fp_only", "provider", "lm"])
def test_boundary_failure_restores_cache_position_and_publishes_nothing(failure: str) -> None:
    cache = _Cache()
    state = _state()
    provider = _Provider(1, fail=failure == "provider")
    model = _SlotModel(fail_on=None if failure == "provider" else failure)
    with pytest.raises(RuntimeError, match="failed"):
        self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=t.tensor([[4]]),
            kv_cache=cache,
            device=t.device("cpu"),
        )
    assert cache.pos == 0
    assert state.target_slot_idx == 5


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 0


class _EngineModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            sequence_len=16,
            n_kv_head=1,
            n_embd=1,
            n_head=1,
            n_layer=1,
        )
        self.calls = 0

    def __call__(self, idx: t.Tensor, kv_cache=None) -> t.Tensor:
        self.calls += 1
        return t.zeros((1, idx.shape[1], 4))


def test_generate_stream_zero_budget_skips_prefill_and_final_budget_skips_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _EngineModel()
    engine = InferenceEngine(model, _Tokenizer(), t.device("cpu"))
    monkeypatch.setattr(engine, "_make_kv_cache", lambda **_kwargs: _Cache())

    assert list(engine.generate_stream([1, 2], max_new_tokens=0, temperature=0.0)) == []
    assert model.calls == 0
    assert list(
        engine.generate_stream(
            [1, 2],
            max_new_tokens=1,
            temperature=0.0,
            stop_token_ids=set(),
        )
    ) == [0]
    assert model.calls == 1


def test_stop_and_consumer_cancellation_never_forward_the_yielded_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _EngineModel()
    engine = InferenceEngine(model, _Tokenizer(), t.device("cpu"))
    monkeypatch.setattr(engine, "_make_kv_cache", lambda **_kwargs: _Cache())

    # Greedy token 0 is the tokenizer BOS sentinel, so it is never yielded or
    # forwarded after the one prefill call.
    assert list(engine.generate_stream([1, 2], max_new_tokens=3, temperature=0.0)) == []
    assert model.calls == 1

    stream = engine.generate_stream(
        [1, 2],
        max_new_tokens=3,
        temperature=0.0,
        stop_token_ids=set(),
    )
    assert next(stream) == 0
    assert isinstance(stream, Generator)
    stream.close()
    assert model.calls == 2
