"""Rewind, modify, rollback, and steering-swap behavior for ChatSession."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

import scripts.chat_tui.session as session_module
import steering.conditional.encoder_decoder.self_conditioned as self_conditioned_module
from nanochat.tokenizer import get_tokenizer
from scripts.chat_tui.session import (
    ConversationText,
    GenerationEvent,
    GenerationEventKind,
    LedgerSegment,
    SegmentKind,
    SegmentProvenance,
    SteeringStamp,
    TurnState,
)
from scripts.chat_tui.steering import (
    SignalSource,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.test_chat_tui.test_session_driver import (
    _build_session,
    _collect_events,
    _script_samples,
)
from tests.test_chat_tui.test_session_tokens import _shared_nanochat_base_dir


def _set_deterministic_rewind(session, enabled: bool) -> None:
    session._config = replace(session._config, deterministic_rewind=enabled)


def _state_for_tokens(tokenizer, tokens: tuple[int, ...]) -> TurnState:
    marker_states = {
        tokenizer.encode_special("<user_start>"): TurnState.IN_USER,
        tokenizer.encode_special("<user_end>"): TurnState.BETWEEN,
        tokenizer.encode_special("<assistant_start>"): TurnState.IN_ASSISTANT,
        tokenizer.encode_special("<assistant_end>"): TurnState.BETWEEN,
    }
    state = TurnState.BETWEEN
    for token_id in tokens:
        state = marker_states.get(token_id, state)
    return state


def _selfcond_entry(session, ordinal: int):
    return next(
        entry
        for entry in session.ledger()
        if entry.kind is SegmentKind.SELFCOND and entry.selfcond_ordinal == ordinal
    )


def test_deterministic_rewind_regenerates_bit_identical_tail() -> None:
    session, _tokenizer, _extraction = _build_session(max_new_tokens=6)
    _set_deterministic_rewind(session, True)
    session.send_user_turn("1", lambda _event: None)
    original = session.conversation_tokens()
    snap = session._snapshots[0]

    session.rewind(0, lambda _event: None)

    assert session.conversation_tokens() == original
    assert session.conversation_tokens()[: snap.token_count] == original[: snap.token_count]
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == len(original)


def test_fresh_take_rewind_keeps_prefix_and_reseeds_both_rngs() -> None:
    session, _tokenizer, _extraction = _build_session(max_new_tokens=1)
    _set_deterministic_rewind(session, False)
    session.send_user_turn("1", lambda _event: None)
    before = session.conversation_tokens()
    snap = session._snapshots[0]

    session.rewind(0, lambda _event: None)

    assert session.conversation_tokens()[: snap.token_count] == before[: snap.token_count]
    assert not t.equal(session._token_rng.get_state(), snap.token_rng_state)
    assert not t.equal(session._selection_rng.get_state(), snap.selection_rng_state)
    assert session._attempt == 1


def test_mid_assistant_rewind_auto_continues_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=9)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [*([10] * 9), stop])
    session.send_user_turn("1", lambda _event: None)
    ordinal = next(
        snap.selfcond_ordinal
        for snap in session._snapshots[1:]
        if _state_for_tokens(tokenizer, session.conversation_tokens()[: snap.token_count])
        is TurnState.IN_ASSISTANT
    )
    close_count_before = session.conversation_tokens().count(stop)
    events, on_event = _collect_events()

    session.rewind(ordinal, on_event)

    assert session.turn_state() is TurnState.BETWEEN
    assert session.conversation_tokens().count(stop) == close_count_before
    assert session.conversation_tokens()[-1] == stop
    assert sum(event.kind is GenerationEventKind.TURN_CLOSED for event in events) == 1
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == len(session.conversation_tokens())


def test_rewind_emits_truncated_transcript_before_regeneration_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=5)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, 11, 12, 13, 14, stop])
    session.send_user_turn("1", lambda _event: None)
    snapshot = session._snapshots[1]
    events, on_event = _collect_events()

    session.rewind(1, on_event)

    transcript_index = next(
        index for index, event in enumerate(events) if event.kind is GenerationEventKind.TRANSCRIPT
    )
    first_token_index = next(
        index for index, event in enumerate(events) if event.kind is GenerationEventKind.TOKEN
    )
    assert transcript_index < first_token_index
    assert events[transcript_index].detail == tokenizer.decode(
        list(session.conversation_tokens()[: snapshot.token_count])
    )


@pytest.mark.parametrize("target_state", [TurnState.BETWEEN, TurnState.IN_USER])
def test_rewind_to_non_assistant_snapshot_does_not_generate(
    monkeypatch: pytest.MonkeyPatch,
    target_state: TurnState,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=2, max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, stop])
    session.send_user_turn("1", lambda _event: None)
    session.send_user_turn("xyz", lambda _event: None)
    ordinal = next(
        snap.selfcond_ordinal
        for snap in session._snapshots
        if _state_for_tokens(tokenizer, session.conversation_tokens()[: snap.token_count])
        is target_state
    )
    snap = session._snapshots[ordinal]
    expected_tokens = session.conversation_tokens()[: snap.token_count]
    events, on_event = _collect_events()

    session.rewind(ordinal, on_event)

    assert session.turn_state() is target_state
    assert session.conversation_tokens() == expected_tokens
    assert len(session.conversation_tokens()) == snap.token_count
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == snap.kv_pos
    assert not any(event.kind is GenerationEventKind.TURN_CLOSED for event in events)


def test_rewind_to_prefix_is_token_exact_reprefills_assistant_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=1)
    session.load_prefix(
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "bc"},
            {"role": "user", "content": "2"},
        ]
    )
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, stop])
    session.send_user_turn("3", lambda _event: None)
    old_wrapper = session._wrapper
    old_tokens = session.conversation_tokens()
    ledger_index = next(
        index
        for index, entry in enumerate(session.ledger())
        if entry.kind is SegmentKind.PREFIX_REAL
        and _state_for_tokens(tokenizer, old_tokens[: entry.token_end]) is TurnState.IN_ASSISTANT
    )
    token_end = session.ledger()[ledger_index].token_end
    calls_before = len(extraction.calls)
    events, on_event = _collect_events()

    session.rewind_to_prefix(ledger_index, on_event)

    assert extraction.calls[-1] == old_tokens[:token_end]
    assert len(extraction.calls) == calls_before + 1
    assert session._wrapper is not None
    assert session._wrapper is not old_wrapper
    assert session.turn_state() is TurnState.BETWEEN
    assert session.conversation_tokens()[:token_end] == old_tokens[:token_end]
    assert session.conversation_tokens()[-1] == stop
    assert any(event.kind is GenerationEventKind.TURN_CLOSED for event in events)


def test_rewind_to_prefix_rejects_selfcond_ledger_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    ledger_index = next(
        index for index, entry in enumerate(session.ledger()) if entry.kind is SegmentKind.SELFCOND
    )

    with pytest.raises(ValueError, match="not a prefix segment"):
        session.rewind_to_prefix(ledger_index, lambda _event: None)


def test_modify_later_boundary_replays_exact_window_and_consumes_no_selection_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment_size = 4
    session, tokenizer, _extraction = _build_session(
        segment_size=segment_size,
        max_new_tokens=9,
    )
    _set_deterministic_rewind(session, True)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [*range(10, 19), stop])
    session.send_user_turn("1", lambda _event: None)
    ordinal = 1
    prev = session._snapshots[ordinal - 1]
    original = session.conversation_tokens()
    replay = original[prev.token_count : prev.token_count + segment_size]
    features = {2: 0, 3: 1}
    state_when_armed: list[t.Tensor] = []
    original_queue = SelfConditionedFeatureProvider.queue_explicit_segment

    def queue_spy(provider, queued):
        original_queue(provider, queued)
        state_when_armed.append(session._selection_rng.get_state().clone())

    monkeypatch.setattr(SelfConditionedFeatureProvider, "queue_explicit_segment", queue_spy)

    session.modify(ordinal, features, lambda _event: None)

    edited = _selfcond_entry(session, ordinal)
    assert edited.features == features
    assert edited.provenance is SegmentProvenance.OVERRIDDEN
    assert (
        session.conversation_tokens()[prev.token_count : prev.token_count + segment_size] == replay
    )
    assert len(state_when_armed) == 1
    assert t.equal(state_when_armed[0], session._snapshots[ordinal].selection_rng_state)
    assert session._snapshots[ordinal].ledger_len == prev.ledger_len + 1
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == len(session.conversation_tokens())


def test_modify_s0_reprefills_same_prompt_with_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=2)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, 11, stop])
    session.send_user_turn("1", lambda _event: None)
    prompt = extraction.calls[0]
    features = {2: 0, 3: 1}

    session.modify(0, features, lambda _event: None)

    edited = _selfcond_entry(session, 0)
    assert edited.features == features
    assert edited.provenance is SegmentProvenance.OVERRIDDEN
    assert extraction.calls == [prompt, prompt]
    assert session._snapshots[0].token_count == len(prompt)
    assert session.turn_state() is TurnState.BETWEEN


def test_error_rollback_mid_window_restores_last_snapshot_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=3)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, 11, 12, stop])
    session.send_user_turn("1", lambda _event: None)
    snap = session._snapshots[-1]
    expected_tokens = session.conversation_tokens()[: snap.token_count]
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward
    calls = 0
    failed = False

    def fail_second_forward(wrapper, input_ids, *args, **kwargs):
        nonlocal calls, failed
        calls += 1
        if not failed and calls == 2:
            failed = True
            raise RuntimeError("mid-window boom")
        return original_forward(wrapper, input_ids, *args, **kwargs)

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        fail_second_forward,
    )
    events, on_event = _collect_events()

    with pytest.raises(RuntimeError, match="mid-window boom"):
        session.send_user_turn("2", on_event)

    assert events[-1] == GenerationEvent(
        GenerationEventKind.ERROR_ROLLBACK,
        None,
        f"rolled back to boundary {snap.selfcond_ordinal} (kv {snap.kv_pos}): mid-window boom",
    )
    assert session.conversation_tokens() == expected_tokens
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == snap.kv_pos
    assert len(session._ledger) == snap.ledger_len
    assert len(session._snapshots) == snap.selfcond_ordinal + 1
    assert t.equal(session._token_rng.get_state(), snap.token_rng_state)
    assert t.equal(session._selection_rng.get_state(), snap.selection_rng_state)

    session.send_user_turn("2", lambda _event: None)
    assert session.turn_state() is TurnState.BETWEEN
    assert session._kv_cache.get_pos() == len(session.conversation_tokens())


def test_error_rollback_during_first_prefill_preserves_prompt_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop])
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward
    failed = False

    def fail_first_prefill(wrapper, input_ids, *args, **kwargs):
        nonlocal failed
        if not failed and kwargs["kv_cache"].get_pos() == 0:
            failed = True
            raise RuntimeError("prefill boom")
        return original_forward(wrapper, input_ids, *args, **kwargs)

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        fail_first_prefill,
    )
    events, on_event = _collect_events()
    before_send = session.conversation_tokens()

    with pytest.raises(RuntimeError, match="prefill boom"):
        session.send_user_turn("1", on_event)

    # The failed turn's provisionally appended tokens are rolled back too, so a
    # retry is an ordinary fresh send and may carry DIFFERENT text.
    assert session.conversation_tokens() == before_send
    assert session._wrapper is None
    assert session._kv_cache is None
    assert session._prefill_conversation_base is None
    assert session.turn_state() is TurnState.BETWEEN
    assert events[-1].kind is GenerationEventKind.ERROR_ROLLBACK
    assert events[-1].detail == "discarded prefill: prefill boom"

    session.send_user_turn("2", lambda _event: None)
    assert session.conversation_tokens()[: len(before_send)] == before_send
    assert session.conversation_tokens().count(tokenizer.encode_special("<user_start>")) == 1
    assert session.turn_state() is TurnState.BETWEEN


def test_set_steering_swaps_provider_without_changing_runtime_state_and_stamps_next_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=2, max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, stop])
    session.send_user_turn("1", lambda _event: None)
    assert session._wrapper is not None
    runtime = session._wrapper._self_cond
    assert runtime is not None
    state = runtime.state
    old_config = runtime.config
    first_new_ordinal = len(session._snapshots)
    signal = SteeringSignal(
        name="concept",
        target_scores=t.linspace(-1.0, 1.0, int(session._setup.n_sae_features)),
        source=SignalSource.AXBENCH_CONCEPT,
        identity="concept-v1",
    )
    selection = SteeringSelection(
        SteeringMethod.INTERPOLATION,
        ((signal, 3.0),),
    )

    session.set_steering(selection)

    assert runtime.state is state
    assert runtime.config is not old_config
    assert runtime.config.score_policy is session._score_policy
    session.send_user_turn("2", lambda _event: None)
    steered = _selfcond_entry(session, first_new_ordinal)
    assert steered.provenance is SegmentProvenance.STEERED
    assert steered.steering == SteeringStamp(
        SteeringMethod.INTERPOLATION,
        ("concept",),
        3.0,
    )


def _dense_steering_selection(session) -> SteeringSelection:
    signal = SteeringSignal(
        name="concept",
        target_scores=t.linspace(-1.0, 1.0, int(session._setup.n_sae_features)),
        source=SignalSource.AXBENCH_CONCEPT,
        identity="concept-v1",
    )
    return SteeringSelection(SteeringMethod.INTERPOLATION, ((signal, 3.0),))


def test_resteer_regenerates_the_whole_current_turn_and_steers_its_governing_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resteer must re-draw the conditioning the VISIBLE response was generated
    under — the segment holding its <assistant_start> — and rewrite the answer.

    Regression: targeting the most recent boundary instead re-crossed a segment
    covering only the already-closed turn's tail, so the visible answer stayed
    unsteered while a fresh continuation was appended after it.
    """

    session, tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=9)
    stop = tokenizer.encode_special("<assistant_end>")
    assistant_start = tokenizer.encode_special("<assistant_start>")
    _script_samples(monkeypatch, [10] * 9 + [11] * 9 + [stop])
    session.send_user_turn("1", lambda _event: None)
    before = session.conversation_tokens()
    turn_start = max(index for index, token in enumerate(before) if token == assistant_start) + 1
    assert len(session._snapshots) >= 2, "fixture did not cross a boundary"
    assert _selfcond_entry(session, 0).provenance is SegmentProvenance.PREDICTED

    session.set_steering(_dense_steering_selection(session))
    session.resteer(lambda _event: None)

    after = session.conversation_tokens()
    # The prompt through <assistant_start> is preserved; the response is not.
    assert after[:turn_start] == before[:turn_start]
    assert after[turn_start:] != before[turn_start:]
    assert set(after[turn_start:-1]) == {11}
    governing = _selfcond_entry(session, 0)
    assert governing.provenance is SegmentProvenance.STEERED
    assert governing.steering is not None
    assert session._kv_cache is not None
    assert int(session._kv_cache.get_pos()) == len(after)
    assert session.turn_state() is TurnState.BETWEEN


def test_resteer_preserves_user_text_and_regenerates_only_final_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the live boundary fell inside or before a user region, everything
    through the final <assistant_start> is forced verbatim."""

    session, tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=2)
    stop = tokenizer.encode_special("<assistant_end>")
    assistant_start = tokenizer.encode_special("<assistant_start>")
    _script_samples(monkeypatch, [stop, 12, stop, 13, stop])
    session.send_user_turn("1", lambda _event: None)
    session.send_user_turn("22", lambda _event: None)
    before = session.conversation_tokens()
    live = len(session._snapshots) - 1
    assert live >= 1
    last_start = max(index for index, token in enumerate(before) if token == assistant_start)

    session.set_steering(_dense_steering_selection(session))
    session.resteer(lambda _event: None)

    after = session.conversation_tokens()
    assert after[: last_start + 1] == before[: last_start + 1]
    assert after[-1] == stop
    assert _selfcond_entry(session, live).provenance is SegmentProvenance.STEERED
    assert session.turn_state() is TurnState.BETWEEN


def test_resteer_with_only_prefill_boundary_reuses_alignment_and_steers_s0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(segment_size=4, max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop, 11, stop])
    session.send_user_turn("1", lambda _event: None)
    assert len(session._snapshots) == 1
    prompt = session.conversation_tokens()[: session._prefill_len]
    calls_before = len(extraction.calls)

    session.set_steering(_dense_steering_selection(session))
    session.resteer(lambda _event: None)

    assert len(extraction.calls) == calls_before, "resteer must reuse the prompt alignment"
    after = session.conversation_tokens()
    assert after[: len(prompt)] == prompt
    assert after[len(prompt) :] == (11, stop)
    assert _selfcond_entry(session, 0).provenance is SegmentProvenance.STEERED


def test_resteer_guards_wrapper_and_open_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    with pytest.raises(RuntimeError, match="before any generation"):
        session.resteer(lambda _event: None)

    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    session._conversation.append_tokens([tokenizer.encode_special("<user_start>")])
    tokens = session.conversation_tokens()

    with pytest.raises(ValueError, match="open partial user turn"):
        session.resteer(lambda _event: None)
    assert session.conversation_tokens() == tokens


def test_rewind_bounds_and_empty_modify_fail_without_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    tokens = session.conversation_tokens()
    snapshot_count = len(session._snapshots)

    with pytest.raises(ValueError, match=r"valid range.*0\.\.[0-9]+"):
        session.rewind(-1, lambda _event: None)
    with pytest.raises(ValueError, match=r"valid range.*0\.\.[0-9]+"):
        session.rewind(snapshot_count, lambda _event: None)
    with pytest.raises(ValueError, match="requires at least one feature"):
        session.modify(0, {}, lambda _event: None)

    assert session.conversation_tokens() == tokens
    assert len(session._snapshots) == snapshot_count


def test_rewind_and_modify_before_generation_fail_loudly() -> None:
    session, _tokenizer, _extraction = _build_session()

    with pytest.raises(RuntimeError, match="before any generation"):
        session.rewind(0, lambda _event: None)
    with pytest.raises(RuntimeError, match="before any generation"):
        session.modify(0, {2: 0}, lambda _event: None)


def test_invalid_modify_dict_fails_before_any_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """modify pre-validates against the live provider: an invalid edit costs
    nothing — wrapper, tokens, ledger, snapshots, and pending override are all
    untouched, and the session keeps working."""
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop, stop])
    session.send_user_turn("1", lambda _event: None)
    wrapper_before = session._wrapper
    tokens_before = session.conversation_tokens()
    ledger_before = session.ledger()
    snapshot_count_before = len(session._snapshots)

    out_of_range = int(session._setup.n_sae_features)
    with pytest.raises(ValueError, match="out of range"):
        session.modify(0, {out_of_range: 0}, lambda _event: None)

    assert session._wrapper is wrapper_before
    assert session.conversation_tokens() == tokens_before
    assert session.ledger() == ledger_before
    assert len(session._snapshots) == snapshot_count_before
    assert session._pending_s0_override is None

    session.send_user_turn("2", lambda _event: None)
    assert session.turn_state() is TurnState.BETWEEN
    assert session._ledger[-1].provenance is not SegmentProvenance.OVERRIDDEN


def test_first_prefill_failure_restores_rng_and_retry_matches_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session(max_new_tokens=4)
    token_state = session._token_rng.get_state().clone()
    selection_state = session._selection_rng.get_state().clone()
    original_seed = SelfConditionedFeatureProvider.seed_from_prompt
    failed = False

    def select_then_fail(provider, state, prompt_fp_logits):
        nonlocal failed
        result = original_seed(provider, state, prompt_fp_logits)
        if not failed:
            failed = True
            raise RuntimeError("seed draw boom")
        return result

    monkeypatch.setattr(SelfConditionedFeatureProvider, "seed_from_prompt", select_then_fail)
    with pytest.raises(RuntimeError, match="seed draw boom"):
        session.send_user_turn("1", lambda _event: None)

    assert t.equal(session._token_rng.get_state(), token_state)
    assert t.equal(session._selection_rng.get_state(), selection_state)
    session.send_user_turn("1", lambda _event: None)
    fresh, _fresh_tokenizer, _fresh_extraction = _build_session(max_new_tokens=4)
    fresh.send_user_turn("1", lambda _event: None)
    assert session.conversation_tokens() == fresh.conversation_tokens()


def test_modify_auto_continue_failure_keeps_landed_override_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=9)
    _script_samples(monkeypatch, [10] * 9)
    session.send_user_turn("1", lambda _event: None)
    ordinal = 1
    features = {2: 0, 3: 1}
    original_tokens = session.conversation_tokens()
    expected_boundary_tokens = original_tokens[: session._snapshots[ordinal].token_count]

    def fail_auto_continue(_logits, _on_event, _cancel_check=None):
        edited = _selfcond_entry(session, ordinal)
        assert edited.features == features
        assert edited.provenance is SegmentProvenance.OVERRIDDEN
        raise RuntimeError("auto-continue boom")

    monkeypatch.setattr(session, "_sample_until_close", fail_auto_continue)
    events, on_event = _collect_events()

    with pytest.raises(RuntimeError, match="auto-continue boom"):
        session.modify(ordinal, features, on_event)

    boundary = session._snapshots[ordinal]
    edited = _selfcond_entry(session, ordinal)
    assert session._pending_modify_retry is None
    assert edited.features == features
    assert edited.provenance is SegmentProvenance.OVERRIDDEN
    assert session.conversation_tokens() == expected_boundary_tokens
    assert len(session.conversation_tokens()) == boundary.token_count
    assert session._kv_cache is not None
    assert int(session._kv_cache.get_pos()) == boundary.kv_pos
    assert session._runtime_provider()._queued_explicit is None
    assert events[-1].detail.endswith(
        "; the assistant turn is open — rewind via /steer to continue"
    )


def test_modify_replay_forward_failure_restores_previous_boundary_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=9)
    _script_samples(monkeypatch, [10] * 9)
    session.send_user_turn("1", lambda _event: None)
    ordinal = 1
    features = {2: 0, 3: 1}
    previous = session._snapshots[ordinal - 1]
    expected_tokens = session.conversation_tokens()[: previous.token_count]
    expected_ledger = tuple(session._ledger[: previous.ledger_len])
    original_commit = session._commit_token
    commit_calls = 0

    def fail_second_replay_commit(token_id, on_event, provenance_override=None):
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 2:
            raise RuntimeError("replay forward boom")
        return original_commit(token_id, on_event, provenance_override)

    monkeypatch.setattr(session, "_commit_token", fail_second_replay_commit)

    with pytest.raises(RuntimeError, match="replay forward boom"):
        session.modify(ordinal, features, lambda _event: None)

    assert session._runtime_provider()._queued_explicit is None
    assert session.conversation_tokens() == expected_tokens
    assert tuple(session._ledger) == expected_ledger
    assert len(session._snapshots) == ordinal
    assert session._kv_cache is not None
    assert int(session._kv_cache.get_pos()) == previous.kv_pos
    assert t.equal(session._token_rng.get_state(), previous.token_rng_state)
    assert t.equal(session._selection_rng.get_state(), previous.selection_rng_state)
    assert session._pending_modify_retry is not None
    assert session._pending_modify_retry[0] == ordinal

    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.modify(ordinal, features, lambda _event: None)
    assert _selfcond_entry(session, ordinal).features == features
    assert session._pending_modify_retry is None


def test_modify_boundary_append_failure_discards_override_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=9)
    _script_samples(monkeypatch, [10] * 9)
    session.send_user_turn("1", lambda _event: None)
    ordinal = 1
    features = {2: 0, 3: 1}
    provider = session._runtime_provider()
    original_append = self_conditioned_module.append_packed_encoder_segment
    failed = False

    def fail_once(existing, new):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("append boom")
        return original_append(existing, new)

    monkeypatch.setattr(self_conditioned_module, "append_packed_encoder_segment", fail_once)
    with pytest.raises(RuntimeError, match="append boom"):
        session.modify(ordinal, features, lambda _event: None)

    assert provider._queued_explicit is None
    assert session._pending_modify_retry == (ordinal, features, session._pending_modify_retry[2])
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.modify(ordinal, {4: 0, 5: 1}, lambda _event: None)
    edited = _selfcond_entry(session, ordinal)
    assert edited.features == {4: 0, 5: 1}
    assert edited.provenance is SegmentProvenance.OVERRIDDEN
    assert session._pending_modify_retry is None


def test_failed_boundary_override_cannot_hijack_later_predicted_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=9)
    _script_samples(monkeypatch, [10] * 9)
    session.send_user_turn("1", lambda _event: None)
    discarded = {2: 0, 3: 1}
    original_append = self_conditioned_module.append_packed_encoder_segment
    failed = False

    def fail_once(existing, new):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("append boom")
        return original_append(existing, new)

    monkeypatch.setattr(self_conditioned_module, "append_packed_encoder_segment", fail_once)
    with pytest.raises(RuntimeError, match="append boom"):
        session.modify(1, discarded, lambda _event: None)

    _script_samples(monkeypatch, [10, 10, 10, tokenizer.encode_special("<assistant_end>")])
    session.rewind(0, lambda _event: None)
    predicted = _selfcond_entry(session, 1)
    assert predicted.provenance is SegmentProvenance.PREDICTED
    assert predicted.features != discarded


def test_speculative_boundary_failure_restores_snapshot_rng_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session(segment_size=4, max_new_tokens=3)
    _script_samples(monkeypatch, [10, 11, 12])
    session.send_user_turn("1", lambda _event: None)
    snapshot = session._snapshots[0]
    expected_tokens = session.conversation_tokens()[: snapshot.token_count]
    original_forward = session._model.forward
    failed = False

    def fail_speculative(*args, **kwargs):
        nonlocal failed
        if kwargs.get("fp_logits_only") and not failed:
            failed = True
            raise RuntimeError("speculative boom")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(session._model, "forward", fail_speculative)
    _script_samples(monkeypatch, [10, 11, 12])
    with pytest.raises(RuntimeError, match="speculative boom"):
        session.rewind(0, lambda _event: None)

    assert len(session._snapshots) == 1
    assert session.conversation_tokens() == expected_tokens
    assert t.equal(session._token_rng.get_state(), snapshot.token_rng_state)
    assert t.equal(session._selection_rng.get_state(), snapshot.selection_rng_state)
    _script_samples(monkeypatch, [10, 11, 12])
    session.rewind(0, lambda _event: None)
    assert session.turn_state() is TurnState.BETWEEN


def test_modify_s0_prefill_failure_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop])
    session.send_user_turn("1", lambda _event: None)
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward
    failed = False

    def fail_prefill_once(wrapper, input_ids, *args, **kwargs):
        nonlocal failed
        if not failed and kwargs["kv_cache"].get_pos() == 0:
            failed = True
            raise RuntimeError("modify s0 prefill boom")
        return original_forward(wrapper, input_ids, *args, **kwargs)

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        fail_prefill_once,
    )
    with pytest.raises(RuntimeError, match="modify s0 prefill boom"):
        session.modify(0, {2: 0}, lambda _event: None)

    assert session._wrapper is None
    assert session._pending_modify_retry == (0, {2: 0}, ())
    _script_samples(monkeypatch, [stop])
    session.modify(0, {3: 1}, lambda _event: None)
    assert _selfcond_entry(session, 0).features == {3: 1}


def test_intervening_send_clears_failed_modify_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=8, max_new_tokens=7)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10] * 7 + [11] * 7)
    session.send_user_turn("1", lambda _event: None)
    session.send_user_turn("2", lambda _event: None)
    assert (
        _state_for_tokens(
            tokenizer,
            session.conversation_tokens()[: session._snapshots[1].token_count],
        )
        is TurnState.BETWEEN
    )
    original_append = self_conditioned_module.append_packed_encoder_segment
    failed = False

    def fail_once(existing, new):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("retry poison")
        return original_append(existing, new)

    monkeypatch.setattr(self_conditioned_module, "append_packed_encoder_segment", fail_once)
    with pytest.raises(RuntimeError, match="retry poison"):
        session.modify(2, {2: 0}, lambda _event: None)
    assert session._pending_modify_retry is not None

    _script_samples(monkeypatch, [stop])
    session.send_user_turn("3", lambda _event: None)
    assert session._pending_modify_retry is None
    with pytest.raises(ValueError, match="valid range"):
        session.modify(2, {2: 0}, lambda _event: None)


def test_modify_prefix_rebuilds_from_patched_segments_without_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop, stop])
    session.send_user_turn("1", lambda _event: None)
    prompt = session.conversation_tokens()[: session._prefill_len]
    calls_before = len(extraction.calls)
    features = {2: 0, 3: 1}

    session.modify_prefix(0, features, lambda _event: None)

    entry = session.ledger()[0]
    assert entry.features == features
    assert entry.provenance is SegmentProvenance.OVERRIDDEN
    assert extraction.calls == extraction.calls[:calls_before]
    assert len(extraction.calls) == calls_before
    assert session.conversation_tokens()[: len(prompt)] == prompt
    assert session.turn_state() is TurnState.BETWEEN


def test_modify_prefix_build_failure_leaves_live_world_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replacement world is forwarded before teardown: a failed build must
    leave the live wrapper, ledger, conversation, alignment, and RNG untouched,
    and an immediate retry of the same edit must land."""

    session, _tokenizer, extraction = _build_session(max_new_tokens=1)
    session._config = replace(session._config, deterministic_rewind=False)
    session.load_prefix(
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "bc"},
        ]
    )
    with t.inference_mode():
        session._prefill(lambda _event: None)
    original_wrapper = session._wrapper
    original_aligned = session._aligned_segments
    assert original_wrapper is not None
    assert original_aligned is not None
    original_ledger = session.ledger()
    prompt = session.conversation_tokens()
    entry_token_state = session._token_rng.get_state().clone()
    entry_attempt = session._attempt
    calls_before = len(extraction.calls)
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward
    failed = False

    def fail_reprefill_once(wrapper, input_ids, *args, **kwargs):
        nonlocal failed
        if not failed and kwargs["kv_cache"].get_pos() == 0:
            failed = True
            raise RuntimeError("prefix edit prefill boom")
        return original_forward(wrapper, input_ids, *args, **kwargs)

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        fail_reprefill_once,
    )
    events, on_event = _collect_events()

    with pytest.raises(RuntimeError, match="prefix edit prefill boom"):
        session.modify_prefix(0, {2: 0}, on_event)

    assert session._wrapper is original_wrapper
    assert session._aligned_segments is original_aligned
    assert session.conversation_tokens() == prompt
    assert session.ledger() == original_ledger
    assert t.equal(session._token_rng.get_state(), entry_token_state)
    assert session._attempt == entry_attempt, (
        "a transient build failure must not consume a fresh-take attempt"
    )
    assert events[-1].kind is GenerationEventKind.ERROR_ROLLBACK
    assert events[-1].detail.startswith("prefix edit failed before any state change")

    session.modify_prefix(0, {2: 0}, lambda _event: None)

    assert len(extraction.calls) == calls_before
    assert session.ledger()[0].features == {2: 0}
    assert session.ledger()[0].provenance is SegmentProvenance.OVERRIDDEN
    assert session.conversation_tokens() == prompt


def test_modify_prefix_install_failure_restores_the_previous_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install is bookkeeping, but its invariant checks can raise; a failure
    there must restore the live world wholesale (wrapper, ledger, snapshots,
    conversation, alignment, RNG streams and attempt counter) rather than
    strand an assistant-open session with no wrapper."""

    session, _tokenizer, extraction = _build_session(max_new_tokens=1)
    session._config = replace(session._config, deterministic_rewind=False)
    session.load_prefix(
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "bc"},
        ]
    )
    with t.inference_mode():
        session._prefill(lambda _event: None)
    original_wrapper = session._wrapper
    original_kv = session._kv_cache
    original_aligned = session._aligned_segments
    original_ledger_list = session._ledger
    original_snapshots = session._snapshots
    original_prefill_len = session._prefill_len
    prompt = session.conversation_tokens()
    entry_token_state = session._token_rng.get_state().clone()
    entry_selection_state = session._selection_rng.get_state().clone()
    entry_attempt = session._attempt
    calls_before = len(extraction.calls)
    original_append = session._append_prefix_ledger
    calls = {"count": 0}

    def flaky_append(append_prompt, append_aligned):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("install boom")
        return original_append(append_prompt, append_aligned)

    monkeypatch.setattr(session, "_append_prefix_ledger", flaky_append)
    events, on_event = _collect_events()

    with pytest.raises(RuntimeError, match="install boom"):
        session.modify_prefix(0, {2: 0}, on_event)

    assert session._wrapper is original_wrapper
    assert session._kv_cache is original_kv
    assert session._aligned_segments is original_aligned
    assert session._ledger is original_ledger_list
    assert session._snapshots is original_snapshots
    assert session._prefill_len == original_prefill_len
    assert session.conversation_tokens() == prompt
    assert t.equal(session._token_rng.get_state(), entry_token_state)
    assert t.equal(session._selection_rng.get_state(), entry_selection_state)
    assert session._attempt == entry_attempt
    assert session.turn_state() is TurnState.BETWEEN
    assert events[-1].kind is GenerationEventKind.ERROR_ROLLBACK
    assert events[-1].detail.startswith("prefix edit failed during install")

    session.modify_prefix(0, {2: 0}, lambda _event: None)

    assert calls["count"] == 2
    assert len(extraction.calls) == calls_before
    assert session.ledger()[0].features == {2: 0}
    assert session.ledger()[0].provenance is SegmentProvenance.OVERRIDDEN


def test_rewind_to_prefix_build_failure_leaves_live_world_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed re-prefill during a mid-assistant prefix rewind must not strand
    the session (previously: empty ledger + open assistant turn + no wrapper,
    with every mutating entry point dead until /clear)."""

    session, tokenizer, extraction = _build_session(max_new_tokens=1)
    session.load_prefix(
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "bc"},
            {"role": "user", "content": "2"},
        ]
    )
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [10, stop, stop])
    session.send_user_turn("3", lambda _event: None)
    old_wrapper = session._wrapper
    old_tokens = session.conversation_tokens()
    old_ledger = session.ledger()
    ledger_index = next(
        index
        for index, entry in enumerate(session.ledger())
        if entry.kind is SegmentKind.PREFIX_REAL
        and _state_for_tokens(tokenizer, old_tokens[: entry.token_end]) is TurnState.IN_ASSISTANT
    )
    token_end = session.ledger()[ledger_index].token_end
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward
    failed = False

    def fail_reprefill_once(wrapper, input_ids, *args, **kwargs):
        nonlocal failed
        if not failed and kwargs["kv_cache"].get_pos() == 0 and input_ids.shape[1] == token_end:
            failed = True
            raise RuntimeError("prefix rewind prefill boom")
        return original_forward(wrapper, input_ids, *args, **kwargs)

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        fail_reprefill_once,
    )
    events, on_event = _collect_events()

    with pytest.raises(RuntimeError, match="prefix rewind prefill boom"):
        session.rewind_to_prefix(ledger_index, on_event)

    assert session._wrapper is old_wrapper
    assert session.conversation_tokens() == old_tokens
    assert session.ledger() == old_ledger
    assert events[-1].kind is GenerationEventKind.ERROR_ROLLBACK
    assert events[-1].detail.startswith("prefix rewind failed before any state change")

    session.rewind_to_prefix(ledger_index, lambda _event: None)

    assert extraction.calls[-1] == old_tokens[:token_end]
    assert session._wrapper is not None
    assert session._wrapper is not old_wrapper
    assert session.turn_state() is TurnState.BETWEEN
    assert session.conversation_tokens()[:token_end] == old_tokens[:token_end]
    assert session.conversation_tokens()[-1] == stop


def test_modify_prefix_validates_before_state_change_and_rejects_nonprefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    wrapper = session._wrapper
    tokens = session.conversation_tokens()
    out_of_range = int(session._setup.n_sae_features)

    with pytest.raises(ValueError, match="out of range"):
        session.modify_prefix(0, {out_of_range: 0}, lambda _event: None)
    assert session._wrapper is wrapper
    assert session.conversation_tokens() == tokens

    selfcond_index = next(
        index for index, entry in enumerate(session.ledger()) if entry.kind is SegmentKind.SELFCOND
    )
    with pytest.raises(ValueError, match="not a prefix segment"):
        session.modify_prefix(selfcond_index, {2: 0}, lambda _event: None)


def test_rewind_to_prefix_requires_at_least_two_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    session._ledger[0] = replace(session._ledger[0], token_end=1)

    with pytest.raises(ValueError, match="at least two"):
        session.rewind_to_prefix(0, lambda _event: None)


def test_prefix_ledger_allows_only_one_final_empty_slot_with_final_token_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session()
    prompt = (1, 2, 3)
    aligned = (
        ContextFeatureSegment((0,), (0,), 0, 1),
        ContextFeatureSegment((1,), (0,), 1, 2),
    )
    assignments = t.tensor([0, 0, 1])
    monkeypatch.setattr(
        session_module,
        "compute_context_token_segment_assignments",
        lambda **_kwargs: assignments,
    )

    session._append_prefix_ledger(prompt, aligned)

    assert int(assignments[-1]) == 1
    assert session._ledger[-1].token_start == session._ledger[-1].token_end == 2


def test_prefix_ledger_rejects_two_trailing_empty_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session()
    prompt = (1, 2, 3)
    aligned = tuple(
        ContextFeatureSegment((feature_id,), (0,), feature_id, feature_id + 1)
        for feature_id in range(3)
    )
    monkeypatch.setattr(
        session_module,
        "compute_context_token_segment_assignments",
        lambda **_kwargs: t.tensor([0, 0, 2]),
    )

    with pytest.raises(RuntimeError, match="at most one empty slot"):
        session._append_prefix_ledger(prompt, aligned)


def test_prefix_ledger_rejects_middle_empty_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session()
    prompt = (1, 2, 3, 4)
    aligned = tuple(
        ContextFeatureSegment((feature_id,), (0,), feature_id, feature_id + 1)
        for feature_id in range(3)
    )
    monkeypatch.setattr(
        session_module,
        "compute_context_token_segment_assignments",
        lambda **_kwargs: t.tensor([0, 2, 2, 2]),
    )

    with pytest.raises(RuntimeError, match="before a later non-empty slot"):
        session._append_prefix_ledger(prompt, aligned)


def test_forced_turn_assembly_matches_two_turn_completion_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(_shared_nanochat_base_dir()))
    tokenizer = get_tokenizer()
    completed_prefix = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    second_user = "Second question"
    rendered = tokenizer.render_for_completion(
        {
            "messages": [
                *completed_prefix,
                {"role": "user", "content": second_user},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    conversation = ConversationText(tokenizer, sequence_len=1024)
    conversation.load_prefix(completed_prefix)
    assembled = [*conversation.tokens, *conversation.forced_turn_tokens(second_user)]

    assert assembled == rendered


def test_set_steering_failure_does_not_publish_or_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(segment_size=2, max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop])
    session.send_user_turn("1", lambda _event: None)
    active = session._active_steering
    policy = session._score_policy
    signal_a = SteeringSignal(
        "a",
        t.linspace(-1.0, 1.0, int(session._setup.n_sae_features)),
        SignalSource.AXBENCH_CONCEPT,
        "a-v1",
    )
    signal_b = replace(signal_a, name="b", identity="b-v1")
    over_budget = SteeringSelection(
        SteeringMethod.INTERPOLATION,
        ((signal_a, 4.0), (signal_b, 4.0)),
    )

    with pytest.raises(ValueError, match="budget exceeded"):
        session.set_steering(over_budget)

    assert session._active_steering is active
    assert session._score_policy is policy
    _script_samples(monkeypatch, [stop])
    session.send_user_turn("2", lambda _event: None)
    assert session.ledger()[-1].provenance is SegmentProvenance.PREDICTED


def test_discard_preview_pins_all_four_operation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, _extraction = _build_session(
        segment_size=2,
        max_new_tokens=5,
    )
    _script_samples(monkeypatch, [10, 11, 12, 13, 14])
    session.send_user_turn("1", lambda _event: None)
    prefix = next(entry for entry in session.ledger() if entry.kind is SegmentKind.PREFIX_REAL)
    selfcond = _selfcond_entry(session, 1)

    assert session.discard_preview(prefix) == session_module.DiscardPreview(
        retained_tokens=2,
        discarded_tokens=9,
        discarded_segments=5,
        operation="rewind",
    )
    assert session.discard_preview(prefix, is_modify=True) == session_module.DiscardPreview(
        retained_tokens=5,
        discarded_tokens=6,
        discarded_segments=3,
        operation="modify",
    )
    assert session.discard_preview(selfcond) == session_module.DiscardPreview(
        retained_tokens=7,
        discarded_tokens=4,
        discarded_segments=2,
        operation="rewind",
    )
    assert session.discard_preview(selfcond, is_modify=True) == session_module.DiscardPreview(
        retained_tokens=7,
        discarded_tokens=4,
        discarded_segments=2,
        operation="modify",
    )


def test_transcript_spans_tile_authoritative_utf8_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(_shared_nanochat_base_dir()))
    tokenizer = get_tokenizer()
    conversation = ConversationText(tokenizer, sequence_len=1024)
    emoji_ids = tokenizer.encode("✅")
    assert len(emoji_ids) == 2
    conversation.append_tokens(
        [
            tokenizer.encode_special("<user_start>"),
            *tokenizer.encode("marker-heavy "),
            *emoji_ids,
            *tokenizer.encode(" question"),
            tokenizer.encode_special("<user_end>"),
            tokenizer.encode_special("<assistant_start>"),
            *tokenizer.encode("answer"),
            tokenizer.encode_special("<assistant_end>"),
        ]
    )
    tokens = conversation.tokens
    emoji_start = next(
        index
        for index in range(len(tokens) - len(emoji_ids) + 1)
        if tuple(tokens[index : index + len(emoji_ids)]) == tuple(emoji_ids)
    )
    boundaries = (0, emoji_start, emoji_start + 1, len(tokens))
    session, _mock_tokenizer, _extraction = _build_session()
    session._conversation = conversation
    session._ledger = [
        LedgerSegment(
            kind=SegmentKind.PREFIX_REAL if index == 0 else SegmentKind.SELFCOND,
            encoder_slot_idx=index,
            selfcond_ordinal=None if index == 0 else index - 1,
            token_start=start,
            token_end=end,
            features={index: 0},
            provenance=(SegmentProvenance.REAL if index == 0 else SegmentProvenance.PREDICTED),
            steering=None,
            color_idx=index,
        )
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False))
    ]

    full_text, spans = session.transcript_spans()

    assert full_text == tokenizer.decode(list(tokens))
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(full_text)
    assert all(
        left.char_end == right.char_start for left, right in zip(spans, spans[1:], strict=False)
    )
    assert full_text[spans[0].char_start : spans[0].char_end] == tokenizer.decode(
        list(tokens[boundaries[0] : boundaries[1]])
    )
    assert "".join(full_text[span.char_start : span.char_end] for span in spans) == full_text


def test_set_seed_applies_at_next_reset_only(monkeypatch: pytest.MonkeyPatch) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop, stop])
    session.send_user_turn("1", lambda _event: None)
    mid_conversation_state = session._token_rng.get_state().clone()

    session.set_seed(777)
    # The running streams are untouched until a reset.
    assert t.equal(session._token_rng.get_state(), mid_conversation_state)

    session.clear()
    expected = t.Generator(device=session._token_rng.device)
    expected.manual_seed(777)
    assert t.equal(session._token_rng.get_state(), expected.get_state())

    with pytest.raises(ValueError, match="seed must be a positive int"):
        session.set_seed(0)
    with pytest.raises(ValueError, match="seed must be a positive int"):
        session.set_seed(True)


@pytest.mark.parametrize("segment_size", [4, 32])
@pytest.mark.parametrize("deterministic", [True, False])
def test_regenerate_preserves_prior_turns_and_changes_rng(
    monkeypatch: pytest.MonkeyPatch, segment_size: int, deterministic: bool
) -> None:
    session, tokenizer, extraction = _build_session(segment_size=segment_size, max_new_tokens=2)
    session._config = replace(session._config, deterministic_rewind=deterministic)
    stop = tokenizer.encode_special("<assistant_end>")
    start = tokenizer.encode_special("<assistant_start>")
    _script_samples(monkeypatch, [stop, 12, stop, 13, stop, 14, stop])
    session.send_user_turn("1", lambda _event: None)
    session.send_user_turn("22", lambda _event: None)
    before = session.conversation_tokens()
    prefix_end = max(i for i, token in enumerate(before) if token == start) + 1
    calls = len(extraction.calls)
    token_rng = session._token_rng.get_state().clone()
    selection_rng = session._selection_rng.get_state().clone()
    session.regenerate(18, lambda _event: None)
    assert session.conversation_tokens() == before[:prefix_end] + (13, stop)
    assert session._config.seed == 18
    assert session._config.deterministic_rewind is deterministic
    assert not t.equal(token_rng, session._token_rng.get_state())
    assert not t.equal(selection_rng, session._selection_rng.get_state())
    assert len(extraction.calls) == calls
    session.regenerate(19, lambda _event: None)
    assert session.conversation_tokens() == before[:prefix_end] + (14, stop)


def test_regenerate_invalid_state_retains_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    session, tokenizer, _ = _build_session()
    config = session._config
    with pytest.raises(RuntimeError, match="before any generation"):
        session.regenerate(18, lambda _event: None)
    assert session._config == config
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    session._conversation.append_tokens([tokenizer.encode_special("<user_start>")])
    before = session.conversation_tokens()
    with pytest.raises(ValueError, match="open partial user turn"):
        session.regenerate(18, lambda _event: None)
    assert session._config == config
    assert session.conversation_tokens() == before


def test_regenerate_can_cancel_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    session, tokenizer, _ = _build_session()
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [12, stop, 13, stop])
    session.send_user_turn("1", lambda _event: None)
    session.regenerate(18, lambda _event: None, cancel_check=lambda: True)
    assert session.turn_state() is TurnState.BETWEEN
    assert session._config.seed == 18
    session.regenerate(19, lambda _event: None)
    assert session.conversation_tokens()[-2:] == (13, stop)


def test_regenerate_prefill_failure_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    session, tokenizer, _ = _build_session(segment_size=32)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("1", lambda _event: None)
    before = session.conversation_tokens()
    config = session._config
    token_rng = session._token_rng.get_state().clone()
    selection_rng = session._selection_rng.get_state().clone()

    def fail(*args, **kwargs):
        raise RuntimeError("prefill failed")

    monkeypatch.setattr(session, "_build_prefill_world", fail)
    with pytest.raises(RuntimeError, match="prefill failed"):
        session.regenerate(18, lambda _event: None)
    assert session.conversation_tokens() == before
    assert session._config == config
    assert t.equal(token_rng, session._token_rng.get_state())
    assert t.equal(selection_rng, session._selection_rng.get_state())
