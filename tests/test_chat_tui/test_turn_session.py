"""Turn Chat's epoch ownership and parity with the scalar one-shot engine."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import nanochat.flash_attention as flash_attention
import scripts.chat_tui.turn_session as turn_module
from nanochat.flash_attention import FlashAttnImpl
from scripts.chat_tui.session import GenerationEventKind, SegmentKind, SegmentProvenance, TurnState
from scripts.chat_tui.steering import (
    SignalSource,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)
from scripts.chat_tui.turn_session import (
    StaleTurnOperation,
    TurnChatSession,
    _render_prompt,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    turn_context_token_assignments,
)
from steering.eval import inference_engine
from steering.eval.axbench_upstream.methods import PreparedPromptContext
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import ContextFeatureSegment, SingleEntryContextProvider
from steering.segmentation import SegmentMode, token_to_char_offsets
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans
from tests.steering.test_context_cache_in_memory_conversion import (
    _selected_entry,
    _turn_render_and_source_ranges,
    _ZeroWidthTokenizer,
)
from tests.test_chat_tui.test_session_driver import _build_session


class TurnExtraction:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []
        self.failure = None

    def prepare(self, token_ids):
        self.calls.append(token_ids)
        if self.failure is not None:
            raise self.failure
        offsets = token_to_char_offsets(self.tokenizer, token_ids)
        spans = legacy_turn_spans(token_ids, ChatSpecialTokens.from_tokenizer(self.tokenizer))
        starts = [0, *(offsets[span.start + 1] for span in spans[1:])]
        ends = [*starts[1:], offsets[-1]]
        segments = tuple(
            ContextFeatureSegment((index % 4,), (0,), start, end)
            for index, (start, end) in enumerate(zip(starts, ends, strict=True))
        )
        return PreparedPromptContext(
            token_ids,
            self.tokenizer.encode_special("<assistant_end>"),
            segments,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            token_assignments=tuple(
                turn_context_token_assignments(
                    tokenizer=self.tokenizer, context_tokens=token_ids, context_segments=segments
                ).tolist()
            ),
        )


@pytest.fixture(autouse=True)
def sdpa(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)


def build_turn(*, cadence=4, budget=7, sequence_len=128, deterministic=True, device="cpu"):
    fixed, tokenizer, _ = _build_session(
        segment_size=97, max_new_tokens=budget, sequence_len=sequence_len
    )
    setup = SimpleNamespace(
        **{
            **vars(fixed._setup),
            "segment_mode": SegmentMode.turn_delimited_fixed_tokens,
            "turn_self_cond_segment_size_rustbpe": cadence,
        }
    )
    extraction = TurnExtraction(tokenizer)
    session = TurnChatSession(
        setup=setup,
        model=fixed._model.to(device=device),
        tokenizer=tokenizer,
        device=device,
        config=replace(fixed._config, deterministic_rewind=deterministic),
        extraction=extraction,
    )
    return session, tokenizer, extraction


def script(monkeypatch, tokens):
    pending = iter(tokens)
    monkeypatch.setattr(turn_module, "sample_next_token", lambda *_: t.tensor([[next(pending)]]))


def owned_state(session):
    driver = session._driver
    cache = driver._kv_cache
    runtime = None if driver._wrapper is None else driver._wrapper.self_cond_runtime_snapshot()
    encoder = None if runtime is None else runtime.state.encoder_out
    return (
        (
            driver,
            driver.prepared_context,
            driver._wrapper,
            driver._aligned_segments,
            None if cache is None else int(cache.get_pos()),
            None if runtime is None else (runtime.state.target_slot_idx, runtime.tokens_in_seg),
        ),
        session.identity,
        session.conversation_tokens(),
        session.ledger(),
        driver._token_rng.get_state().clone(),
        driver._selection_rng.get_state().clone(),
        None if cache is None else cache.k_cache.clone(),
        None if cache is None else cache.v_cache.clone(),
        None if driver.prefill_rng is None else driver.prefill_rng[0].clone(),
        None if driver.prefill_rng is None else driver.prefill_rng[1].clone(),
        None if encoder is None else encoder.values.clone(),
        None if encoder is None else encoder.layout.flat_indices.clone(),
        None if encoder is None else encoder.layout.valid_mask.clone(),
        None if encoder is None else encoder.layout.batch_indices.clone(),
        None if encoder is None else encoder.layout.segment_indices.clone(),
        *(row.clone() for row in (() if runtime is None else runtime.fp_buffer)),
    )


def assert_unchanged(session, before):
    after = owned_state(session)
    assert after[:4] == before[:4]
    for actual, expected in zip(after[4:], before[4:], strict=True):
        if expected is None:
            assert actual is None
        else:
            assert t.equal(actual, expected)


@pytest.mark.parametrize("cadence", [1, 2, 4, 17])
def test_new_user_reextracts_all_history_and_replaces_predicted_epoch(monkeypatch, cadence):
    session, tokenizer, extraction = build_turn(cadence=cadence)
    stop = tokenizer.encode_special("<assistant_end>")
    script(monkeypatch, [12, 13, stop, 14, stop])
    session.send_user_turn("hi", lambda _: None)
    old = session._driver
    old_row = session.ledger()[-1]
    session.send_user_turn("next", lambda _: None)
    assert len(extraction.calls) == 2
    assert extraction.calls[1] == _render_prompt(tokenizer, session._driver.prompt_messages)
    assert session._driver.prompt_messages[1] == {
        "role": "assistant",
        "content": tokenizer.decode([12, 13]),
    }
    assert session._driver._wrapper is not old._wrapper
    assert session._driver._kv_cache is not old._kv_cache
    assert session._driver._config.self_cond_segment_size == cadence
    real = [row for row in session.ledger() if row.kind is SegmentKind.PREFIX_REAL]
    predicted = [row for row in session.ledger() if row.kind is SegmentKind.SELFCOND]
    assert len(real) == 3  # user A, now-real assistant A, user B
    assert predicted[0].token_start == len(extraction.calls[1]) - 1
    assert {row.epoch_id for row in session.ledger()} == {2}
    with pytest.raises(StaleTurnOperation, match="earlier Chat epoch/revision"):
        session.assert_current_entry(old_row)
    session.transcript_spans()


@pytest.mark.parametrize("cadence", [1, 2, 4])
@pytest.mark.parametrize("force_full_budget", [False, True])
def test_turn_chat_exact_scalar_engine_parity(monkeypatch, cadence, force_full_budget):
    session, tokenizer, extraction = build_turn(cadence=cadence, budget=9)
    driver = session._driver
    prompt = _render_prompt(tokenizer, [{"role": "user", "content": "hi"}])
    prepared = extraction.prepare(prompt)
    config = driver._build_self_cond_config(driver._score_policy)
    reference_rng = t.Generator(device="cpu").set_state(driver._selection_rng.get_state())
    config = replace(config, selection=replace(config.selection, rng=reference_rng))
    engine = build_engine_self_conditioned(
        driver._model,
        tokenizer,
        t.device("cpu"),
        config,
        context_provider=SingleEntryContextProvider(
            prompt,
            prepared.segments,
            segment_mode=prepared.segment_mode,
            token_assignments=prepared.token_assignments,
        ),
    )
    calls = []
    real_forward = driver._model.forward

    def observe(input_ids, *args, **kwargs):
        result = real_forward(input_ids, *args, **kwargs)
        calls.append(
            (
                input_ids.clone(),
                kwargs["seg_map"].clone(),
                tuple(value.clone() for value in result),
            )
        )
        return result

    monkeypatch.setattr(driver._model, "forward", observe)
    if force_full_budget:
        scheduled = iter([12] * 9)
        monkeypatch.setattr(
            inference_engine, "sample_next_token", lambda *_: t.tensor([[next(scheduled)]])
        )
        script(monkeypatch, [12] * 9)
    expected = list(
        engine.generate_stream(
            list(prompt),
            max_new_tokens=9,
            temperature=driver._config.temperature,
            top_k=driver._config.top_k,
            seed=driver._config.seed,
        )
    )
    reference_calls = list(calls)
    calls.clear()
    session.send_user_turn("hi", lambda _: None)
    actual = session.conversation_tokens()[len(prompt) : -1]
    assert actual == tuple(expected)
    assert len(calls) == len(reference_calls)
    for actual_call, expected_call in zip(calls, reference_calls, strict=True):
        for a, b in zip(actual_call[:2], expected_call[:2], strict=True):
            assert t.equal(a, b)
        for a, b in zip(actual_call[2], expected_call[2], strict=True):
            assert t.equal(a, b)
    if force_full_budget:
        assert len(actual) == 9
        assert len(calls) == 10  # real prefix, marker, eight ordinary forwards
        for index, call in enumerate(calls[1:]):
            assert call[1].item() == 1 + index // cadence
    reference = engine._model.self_cond_runtime_snapshot()
    assert session._driver._wrapper is not None
    result = session._driver._wrapper.self_cond_runtime_snapshot()
    assert result.tokens_in_seg == reference.tokens_in_seg
    assert result.state.target_slot_idx == reference.state.target_slot_idx
    assert t.equal(result.state.encoder_out.values, reference.state.encoder_out.values)
    assert len(result.fp_buffer) == len(reference.fp_buffer)
    for a, b in zip(result.fp_buffer, reference.fp_buffer, strict=True):
        assert t.equal(a, b)
    assert t.equal(session._driver._selection_rng.get_state(), reference_rng.get_state())


@pytest.mark.parametrize("cadence", [1, 2, 4])
def test_deterministic_rewinds_restore_exact_tokens_and_rng(monkeypatch, cadence):
    session, _, extraction = build_turn(cadence=cadence, budget=9)
    session.send_user_turn("hi", lambda _: None)
    tokens = session.conversation_tokens()
    rng = session._driver._token_rng.get_state().clone()
    selection_rng = session._driver._selection_rng.get_state().clone()
    ordinals = list(range(len(session._driver._snapshots)))
    for ordinal in reversed(ordinals):
        session.rewind(ordinal, lambda _: None)
        assert session.conversation_tokens() == tokens
        assert t.equal(session._driver._token_rng.get_state(), rng)
        assert t.equal(session._driver._selection_rng.get_state(), selection_rng)
    assert len(extraction.calls) == 1


@pytest.mark.parametrize(
    "stage", ["extract", "prefill", "decode", "provider", "cancel", "cancel_close", "event"]
)
def test_failed_next_epoch_keeps_entire_old_world(monkeypatch, stage):
    session, tokenizer, extraction = build_turn()
    stop = tokenizer.encode_special("<assistant_end>")
    script(monkeypatch, [12, stop, 13, 14, stop])
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    if stage == "extract":
        extraction.failure = RuntimeError("injected extraction")
    elif stage == "provider":
        original_append = turn_module._LedgerProvider._append_from_fp_logits

        def fail_after_selection(*args, **kwargs):
            original_append(*args, **kwargs)
            raise RuntimeError("injected provider failure after selection and encoder append")

        monkeypatch.setattr(
            turn_module._LedgerProvider, "_append_from_fp_logits", fail_after_selection
        )
    elif stage in {"prefill", "decode"}:
        real = session._driver._model.forward

        def fail(input_ids, *args, **kwargs):
            result = real(input_ids, *args, **kwargs)
            if (stage == "prefill" and input_ids.shape[1] > 1) or (
                stage == "decode" and input_ids.shape[1] == 1 and int(input_ids[0, 0]) == 13
            ):
                raise RuntimeError("injected model failure after KV write")
            return result

        monkeypatch.setattr(session._driver._model, "forward", fail)

    def event(event):
        if event.kind is GenerationEventKind.TURN_CLOSED and stage == "cancel_close":
            session.cancel()
        if event.kind is GenerationEventKind.TOKEN:
            if stage == "cancel":
                session.cancel()
            elif stage == "event":
                raise RuntimeError("injected event failure")

    if stage in {"cancel", "cancel_close"}:
        with pytest.raises(RuntimeError, match="cancelled"):
            session.send_user_turn("next", event)
    else:
        with pytest.raises(RuntimeError, match="injected"):
            session.send_user_turn("next", event)
    assert_unchanged(session, before)


@pytest.mark.parametrize("mismatch", ["token_ids", "assistant_end"])
def test_extraction_entry_mismatch_rejected_before_prefill(monkeypatch, mismatch):
    session, _, extraction = build_turn()
    before = owned_state(session)
    original_prepare = extraction.prepare

    def wrong_entry(tokens):
        entry = original_prepare(tokens)
        return (
            replace(entry, token_ids=entry.token_ids[:-1])
            if mismatch == "token_ids"
            else replace(entry, assistant_end_id=entry.assistant_end_id + 1)
        )

    monkeypatch.setattr(extraction, "prepare", wrong_entry)
    monkeypatch.setattr(session._driver._model, "forward", lambda *_a, **_k: pytest.fail("prefill"))
    with pytest.raises(ValueError, match="exact rendered prompt"):
        session.send_user_turn("hi", lambda _: None)
    assert_unchanged(session, before)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"segment_mode": SegmentMode.fixed_tokens}, "turn-delimited segment mode"),
        ({"token_assignments": None}, "persisted token ownership"),
        ({"token_assignments": (0,)}, "ownership length"),
        ({"token_assignments": (0, 0, 0, 0, 0, 2)}, "invalid segment index"),
        ({"token_assignments": (-1, 0, 0, 0, 0, 1)}, "invalid segment index"),
        ({"token_assignments": (False, 0, 0, 0, 0, 1)}, "invalid segment index"),
        ({"token_assignments": (0.0, 0, 0, 0, 0, 1)}, "invalid segment index"),
        ({"token_assignments": (0, 0, 0, 0, 0, 0)}, "only the assistant marker"),
        ({"token_assignments": (0, 1, 0, 0, 0, 1)}, "monotone"),
        ({"token_assignments": (0, 0, 0, 0, 1, 1)}, "only the assistant marker"),
    ],
)
def test_invalid_producer_identity_fails_before_forward_and_preserves_epoch(
    monkeypatch, change, message
):
    session, _, extraction = build_turn()
    before = owned_state(session)
    prepare = extraction.prepare
    monkeypatch.setattr(extraction, "prepare", lambda tokens: replace(prepare(tokens), **change))
    monkeypatch.setattr(session._driver._model, "forward", lambda *_a, **_k: pytest.fail("prefill"))
    with pytest.raises(ValueError, match=message):
        session.send_user_turn("hi", lambda _: None)
    assert_unchanged(session, before)


@pytest.mark.parametrize("field", ["token_assignments", "segments"])
def test_mutable_prepared_context_containers_cannot_become_restart_anchors(monkeypatch, field):
    session, _, extraction = build_turn()
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    prepare = extraction.prepare

    def mutable_entry(tokens):
        entry = prepare(tokens)
        # frozen=True protects field assignment, not a list retained from a
        # malformed producer. Such an alias could change future restart inputs.
        return replace(entry, **{field: list(getattr(entry, field))})

    monkeypatch.setattr(extraction, "prepare", mutable_entry)
    monkeypatch.setattr(session._driver._model, "forward", lambda *_a, **_k: pytest.fail("prefill"))
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        session.send_user_turn("next", lambda _: None)
    assert_unchanged(session, before)


@pytest.mark.parametrize("cadence", [1, 2, 4])
@pytest.mark.parametrize("operation", ["rewind", "modify", "resteer", "steering", "fork_rewind"])
def test_restart_and_ledger_retain_exact_producer_ownership(monkeypatch, operation, cadence):
    session, tokenizer, extraction = build_turn(cadence=cadence, budget=9)
    prepare = extraction.prepare
    entries = []

    def split_user_context(tokens):
        entry = prepare(tokens)
        user, marker = entry.segments
        segments = (
            replace(user, char_end=2),
            replace(user, char_start=2, feature_ids=(2,)),
            marker,
        )
        # Producer ownership can differ from a reconstruction over projected
        # character ranges (e.g. source tokens straddling a boundary).
        owners = (0, 0, 1, 1, 1, 2)
        reconstructed = tuple(
            turn_context_token_assignments(
                tokenizer=tokenizer, context_tokens=tokens, context_segments=segments
            ).tolist()
        )
        assert owners != reconstructed
        entry = replace(entry, segments=segments, token_assignments=owners)
        entries.append(entry)
        return entry

    monkeypatch.setattr(extraction, "prepare", split_user_context)
    script(monkeypatch, [12] * 40)
    forwarded = []
    forward = session._driver._model.forward

    def observe(input_ids, *args, **kwargs):
        forwarded.append((input_ids.clone(), kwargs["seg_map"].clone()))
        return forward(input_ids, *args, **kwargs)

    monkeypatch.setattr(session._driver._model, "forward", observe)
    session.send_user_turn("hi", lambda _: None)
    prepared = entries[0]
    first_inputs, first_owners = forwarded[0]
    assert first_inputs.tolist() == [list(prepared.token_ids[:-1])]
    assert first_owners.tolist() == [list(prepared.token_assignments[:-1])]
    forwarded.clear()
    extraction.failure = AssertionError("restart must reuse the prepared context")
    monkeypatch.setattr(extraction, "prepare", lambda _: pytest.fail("restart re-extracted"))
    if operation == "rewind":
        session.rewind(0, lambda _: None)
    elif operation == "modify":
        session.modify(0, {3: 0}, lambda _: None)
    elif operation == "steering":
        session.apply_steering(SteeringSelection(SteeringMethod.OFF, ()), lambda _: None)
    elif operation == "fork_rewind":
        session.rewind(1, lambda _: None)
    else:
        session.resteer(lambda _: None)
    if operation != "fork_rewind":
        assert t.equal(forwarded[0][0], first_inputs)
        assert t.equal(forwarded[0][1], first_owners)
    driver = session._driver
    assert driver.prepared_context is prepared
    provider = driver._wrapper.context_provider
    assert provider.segment_mode is prepared.segment_mode
    assert provider.token_assignments == prepared.token_assignments
    displayed = tuple(
        row.encoder_slot_idx
        for row in session.ledger()
        if row.kind is SegmentKind.PREFIX_REAL
        for _ in range(row.token_start, row.token_end)
    )
    assert displayed == prepared.token_assignments[:-1]
    session.clear()
    assert session._driver.prepared_context is None


@pytest.mark.parametrize("cadence", [None, True, False, 0, -1, 1.5, "4"])
def test_direct_turn_session_rejects_invalid_metadata_cadence(cadence):
    with pytest.raises(ValueError, match="positive integer checkpoint"):
        build_turn(cadence=cadence)


@pytest.mark.parametrize("cadence,ordinal", [(1, 0), (1, 1), (1, 3), (2, 1), (4, 1)])
def test_edits_override_only_current_epoch_and_use_no_boundary_replay(
    monkeypatch, cadence, ordinal
):
    session, tokenizer, _ = build_turn(cadence=cadence, budget=9)
    script(monkeypatch, [12] * 18)
    session.send_user_turn("hi", lambda _: None)
    session.modify(ordinal, {3: 0}, lambda _: None)
    row = next(row for row in session.ledger() if row.selfcond_ordinal == ordinal)
    assert row.features == {3: 0}
    assert row.provenance.value == "overridden"
    assert session.conversation_tokens()[-1] == tokenizer.encode_special("<assistant_end>")
    session.transcript_spans()


def test_target_budget_rejects_before_extraction_or_state_changes():
    session, _, extraction = build_turn(sequence_len=16, budget=9)
    before = owned_state(session)
    with pytest.raises(ValueError, match="RustBPE"):
        session.send_user_turn("lengthy", lambda _: None)
    assert extraction.calls == []
    assert_unchanged(session, before)


def test_stale_candidate_cannot_replace_a_new_conversation(monkeypatch):
    session, _, _ = build_turn()
    script(monkeypatch, [12, 13])

    def event(event):
        if event.kind is GenerationEventKind.TOKEN:
            session.load_prefix([])

    with pytest.raises(RuntimeError, match="cancelled"):
        session.send_user_turn("hi", event)
    assert session.conversation_tokens() == (0,)
    assert session.ledger() == ()
    assert session._driver.messages == []


def test_earlier_turn_rewind_reextracts_logical_prefix(monkeypatch):
    session, tokenizer, extraction = build_turn()
    stop = tokenizer.encode_special("<assistant_end>")
    script(monkeypatch, [12, stop, 13, stop, 14, stop])
    session.send_user_turn("hi", lambda _: None)
    session.send_user_turn("next", lambda _: None)
    session.rewind_to_prefix(1, lambda _: None)
    assert session._driver.prompt_messages == [{"role": "user", "content": "hi"}]
    assert len(extraction.calls) == 3
    assert len(session._driver.messages) == 2
    assert session.identity[0] == 3


@pytest.mark.parametrize(
    "special",
    [
        "bos",
        "<user_start>",
        "<user_end>",
        "<assistant_start>",
        "<assistant_end>",
        "<python_start>",
        "<python_end>",
        "<output_start>",
        "<output_end>",
    ],
)
@pytest.mark.parametrize("cadence", [1, 2])
def test_all_structural_ids_stop_before_display_or_forward(monkeypatch, special, cadence):
    session, tokenizer, extraction = build_turn(cadence=cadence)
    token = tokenizer.get_bos_token_id() if special == "bos" else tokenizer.encode_special(special)
    script(monkeypatch, [token])
    events = []
    session.send_user_turn("hi", events.append)
    prompt = extraction.calls[0]
    assert session.conversation_tokens() == (*prompt, tokenizer.encode_special("<assistant_end>"))
    assert session._driver._kv_cache.get_pos() == len(prompt)
    assert [e.detail for e in events if e.kind is GenerationEventKind.TOKEN] == [
        str(tokenizer.encode_special("<assistant_end>"))
    ]


def test_failed_attempt_identity_cannot_be_reused_by_retry(monkeypatch):
    session, tokenizer, _ = build_turn()
    script(monkeypatch, [12, tokenizer.encode_special("<assistant_end>")])
    abandoned = []

    def cancel_preview(event):
        abandoned.append(event)
        session.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        session.send_user_turn("hi", cancel_preview)
    assert abandoned[0].revision != session.identity[1]

    def retry_preview(_event):
        assert not session.accepts_event(abandoned[0])

    session.send_user_turn("hi", retry_preview)
    assert not session.accepts_event(abandoned[0])


def test_superseded_attempt_cannot_emit_rollback_after_newer_failed_attempt():
    session, _, extraction = build_turn()
    abandoned = []
    newer = []

    def supersede(event):
        abandoned.append(event)
        extraction.failure = RuntimeError("newer attempt fails")
        with pytest.raises(RuntimeError, match="newer attempt"):
            session.send_user_turn("newer", newer.append)

    with pytest.raises(RuntimeError, match="cancelled"):
        session.send_user_turn("older", supersede)
    assert len(abandoned) == 1  # no late rollback accepted as committed UI state
    assert newer[-1].kind is GenerationEventKind.ERROR_ROLLBACK
    assert session.conversation_tokens() == (0,)
    assert not session.accepts_event(abandoned[0])


def test_cancel_before_prefill_and_midstream_restore_visible_committed_transcript(monkeypatch):
    session, _, extraction = build_turn()
    before = owned_state(session)
    events = []
    with pytest.raises(RuntimeError, match="cancelled"):
        session.send_user_turn("hi", events.append, cancel_check=lambda: True)
    assert not extraction.calls
    assert events[-1].detail == session.transcript_spans()[0]
    assert_unchanged(session, before)

    script(monkeypatch, [12] * 7)
    events.clear()

    def cancel_token(event):
        events.append(event)
        if event.kind is GenerationEventKind.TOKEN:
            session.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        session.send_user_turn("hi", cancel_token)
    assert events[-1].kind is GenerationEventKind.TRANSCRIPT
    assert events[-1].detail == session.transcript_spans()[0]
    assert session.accepts_event(events[-1])
    assert_unchanged(session, before)


@pytest.mark.parametrize("deterministic", [True, False])
def test_successful_steering_is_atomic_and_survives_clear(monkeypatch, deterministic):
    session, _, extraction = build_turn(deterministic=deterministic)
    scores = t.ones(session._driver._setup.n_sae_features, dtype=t.float32)
    signal = SteeringSignal("all", scores, SignalSource.AXBENCH_CONCEPT, "all-v1")
    selection = SteeringSelection(SteeringMethod.INTERPOLATION, ((signal, 6.0),))
    session.apply_steering(selection, lambda _: None)
    assert not extraction.calls
    assert session._driver._active_steering == selection
    script(monkeypatch, [12] * 14)
    session.send_user_turn("hi", lambda _: None)
    old = session._driver
    old_row = session.ledger()[-1]
    off = SteeringSelection(SteeringMethod.OFF, ())

    def preview(_event):
        assert session._driver is old
        assert session._driver._active_steering == selection

    session.apply_steering(off, preview)
    assert session._driver._active_steering == off
    assert len(extraction.calls) == 1
    with pytest.raises(StaleTurnOperation, match="earlier Chat epoch/revision"):
        session.assert_current_entry(old_row)
    session.clear()
    assert session._driver._active_steering == off


def test_loaded_prefix_is_owned_and_clear_restarts_it(monkeypatch):
    session, tokenizer, extraction = build_turn()
    prefix = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    session.load_prefix(prefix)
    tokens = session.conversation_tokens()
    prefix[0]["content"] = "external mutation"
    script(monkeypatch, [12, tokenizer.encode_special("<assistant_end>")])
    session.send_user_turn("next", lambda _: None)
    assert session._driver.prompt_messages[0]["content"] == "hi"
    assert len(extraction.calls) == 1
    session.clear()
    assert session.conversation_tokens() == tokens
    assert session._driver.messages[0]["content"] == "hi"
    assert session.ledger() == ()


@pytest.mark.parametrize(
    "prefix",
    [
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "x" * 128}, {"role": "assistant", "content": "ok"}],
    ],
)
def test_invalid_prefix_leaves_committed_conversation_unchanged(prefix):
    session, _, _ = build_turn()
    before = owned_state(session)
    with pytest.raises(ValueError, match="prefix"):
        session.load_prefix(prefix)
    assert_unchanged(session, before)


@pytest.mark.parametrize("ordinal", [True, 1.0, "1", -1, 100])
def test_invalid_rewind_and_edit_controls_are_atomic(monkeypatch, ordinal):
    session, _, _ = build_turn()
    script(monkeypatch, [12] * 7)
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    for action in (
        lambda: session.rewind(ordinal, lambda _: None),
        lambda: session.modify(ordinal, {3: 0}, lambda _: None),
        lambda: session.rewind_to_prefix(ordinal, lambda _: None),
    ):
        with pytest.raises(ValueError, match="ordinal|prefix ledger entry"):
            action()
        assert_unchanged(session, before)


@pytest.mark.parametrize("deterministic", [True, False])
def test_k1_s1_edit_keeps_committed_s0_features_and_provenance(monkeypatch, deterministic):
    # kills: restarting prefill for a K=1 S1 edit with an override for S1 only,
    # which re-predicts S0 and silently discards an earlier S0 edit.
    session, _, _ = build_turn(cadence=1, budget=6, deterministic=deterministic)
    script(monkeypatch, [12] * 40)
    session.send_user_turn("hi", lambda _: None)
    predicted_s0 = next(row for row in session.ledger() if row.selfcond_ordinal == 0)
    session.modify(1, {5: 1}, lambda _: None)
    kept_s0 = next(row for row in session.ledger() if row.selfcond_ordinal == 0)
    assert (kept_s0.features, kept_s0.provenance) == (
        predicted_s0.features,
        predicted_s0.provenance,
    )
    session.modify(0, {3: 0}, lambda _: None)
    session.modify(1, {7: 1}, lambda _: None)
    rows = {row.selfcond_ordinal: row for row in session.ledger() if row.selfcond_ordinal in {0, 1}}
    assert rows[0].features == {3: 0}
    assert rows[0].provenance is SegmentProvenance.OVERRIDDEN
    assert rows[1].features == {7: 1}
    assert rows[1].provenance is SegmentProvenance.OVERRIDDEN
    session.transcript_spans()


def test_edit_that_does_not_land_on_the_requested_segment_fails_loud(monkeypatch):
    # kills: dropping the post-condition on the edited ledger row.
    session, _, _ = build_turn(cadence=2, budget=9)
    script(monkeypatch, [12] * 18)
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    provider_cls = turn_module._LedgerProvider
    original = provider_cls.queue_explicit_segment

    def swallow(self, features):
        del features
        return original(self, {9: 0})

    monkeypatch.setattr(provider_cls, "queue_explicit_segment", swallow)
    with pytest.raises(
        RuntimeError, match=r"turn edit did not land on segment 1: overridden \{9: 0\}"
    ):
        session.modify(1, {3: 0}, lambda _: None)
    assert_unchanged(session, before)
    # The explicit segment landing on a different ordinal leaves the requested row absent.
    monkeypatch.setattr(provider_cls, "queue_explicit_segment", original)
    script(monkeypatch, [12] * 18)
    original_receipt = turn_module._TurnDriver._append_receipt

    def shifted(self, receipt, ordinal, start):
        return original_receipt(self, receipt, ordinal + 1 if ordinal == 1 else ordinal, start)

    monkeypatch.setattr(turn_module._TurnDriver, "_append_receipt", shifted)
    with pytest.raises(RuntimeError, match="turn edit did not land on segment 1: absent"):
        session.modify(1, {3: 0}, lambda _: None)
    assert_unchanged(session, before)


def test_discard_previews_and_real_history_edit_guard(monkeypatch):
    session, tokenizer, _ = build_turn(cadence=2)
    script(monkeypatch, [12] * 14)
    session.send_user_turn("hi", lambda _: None)
    session.send_user_turn("next", lambda _: None)
    real = session.ledger()[1]  # earlier assistant; rewinds its complete user
    predicted = next(row for row in session.ledger() if row.selfcond_ordinal == 1)
    prefix_preview = session.discard_preview(real)
    assert prefix_preview.retained_tokens == len(
        _render_prompt(tokenizer, [{"role": "user", "content": "hi"}])
    )
    predicted_preview = session.discard_preview(predicted, is_modify=True)
    assert predicted_preview.retained_tokens == session._driver._snapshots[1].token_count
    before = owned_state(session)
    with pytest.raises(ValueError, match="read-only"):
        session.discard_preview(real, is_modify=True)
    with pytest.raises(ValueError, match="read-only"):
        session.modify_prefix(1, {3: 0}, lambda _: None)
    with pytest.raises(StaleTurnOperation, match="no longer live"):
        session.assert_current_entry(replace(predicted, encoder_slot_idx=10000))
    assert_unchanged(session, before)


def test_real_prefix_ledger_uses_training_ownership_at_zero_width_turn_boundary():
    tokenizer = _ZeroWidthTokenizer()
    _, prompt, _, bos, starts, ends = _turn_render_and_source_ranges()
    boundary = prompt.index(tokenizer.encode_special("<assistant_start>"))
    prompt = (*prompt[:boundary], tokenizer.ZERO_WIDTH, *prompt[boundary:])
    entry, _ = _selected_entry(prompt, tokenizer, starts, ends, bos)
    session, _, _ = build_turn()
    driver = session._driver
    driver._tokenizer = tokenizer
    driver._append_prefix_ledger(
        prompt, entry.segments, assignments=t.tensor(entry.token_assignments)
    )
    expected = turn_context_token_assignments(
        tokenizer=tokenizer, context_tokens=prompt[:-1], context_segments=entry.segments[:-1]
    ).tolist()
    actual = [
        row.encoder_slot_idx
        for row in driver.ledger()
        for _ in range(row.token_start, row.token_end)
    ]
    assert actual == expected
    assert actual[boundary] == actual[boundary + 1] != actual[boundary - 1]
    assert driver.ledger()[-1].token_start == driver.ledger()[-1].token_end == len(prompt) - 1


@pytest.mark.parametrize("operation", ["rewind", "modify", "resteer", "steering"])
def test_failed_current_epoch_operations_do_not_overwrite_old_kv_or_rng(monkeypatch, operation):
    session, _, _ = build_turn(cadence=2, budget=7, deterministic=False)
    script(monkeypatch, [12] * 30)
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    attempts = session._driver._attempt
    policy = session._driver._score_policy
    real = session._driver._model.forward

    def fail_after_write(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("injected after writing candidate KV")

    monkeypatch.setattr(session._driver._model, "forward", fail_after_write)
    with pytest.raises(RuntimeError, match="injected"):
        if operation == "rewind":
            session.rewind(1, lambda _: None)
        elif operation == "modify":
            session.modify(1, {3: 0}, lambda _: None)
        elif operation == "steering":
            session.apply_steering(SteeringSelection(SteeringMethod.OFF, ()), lambda _: None)
        else:
            session.resteer(lambda _: None)
    assert_unchanged(session, before)
    assert session._driver._attempt == attempts
    assert session._driver._score_policy is policy


def test_pending_seed_and_clear_invalidate_old_actions_without_advancing_rng(monkeypatch):
    session, _, _ = build_turn()
    script(monkeypatch, [12] * 14)
    session.send_user_turn("hi", lambda _: None)
    row = session.ledger()[-1]
    before = session._driver._token_rng.get_state().clone()
    session.set_seed(123)
    assert t.equal(before, session._driver._token_rng.get_state())
    with pytest.raises(StaleTurnOperation, match="earlier Chat epoch/revision"):
        session.assert_current_entry(row)
    session.clear()
    assert session.conversation_tokens() == (0,)
    assert session._driver._token_rng.initial_seed() == 123
    assert session._driver._selection_rng.initial_seed() == 124


@pytest.mark.parametrize("extra", [0, 1])
def test_exact_target_window_and_one_over(monkeypatch, extra):
    session, tokenizer, extraction = build_turn(sequence_len=16, budget=9)
    prompt = _render_prompt(tokenizer, [{"role": "user", "content": "hi"}])
    assert len(prompt) + 9 + 1 == 16
    script(monkeypatch, [12] * 9)
    if extra:
        with pytest.raises(ValueError, match="RustBPE"):
            session.send_user_turn("hi!", lambda _: None)
        assert extraction.calls == []
    else:
        session.send_user_turn("hi", lambda _: None)
        assert session.token_budget() == (16, 16)


@pytest.mark.parametrize("state", ["new", "cleared", "missing_rng"])
def test_restart_requires_complete_prefill_anchor_and_keeps_committed_state(monkeypatch, state):
    session, _, extraction = build_turn()
    if state != "new":
        session.send_user_turn("hi", lambda _: None)
        if state == "cleared":
            session.clear()
        else:
            session._driver.prefill_rng = None
    before = owned_state(session)
    calls = len(extraction.calls)
    monkeypatch.setattr(session._driver._model, "forward", lambda *_a, **_k: pytest.fail("prefill"))
    with pytest.raises(RuntimeError, match="cannot regenerate before a turn"):
        session.resteer(lambda _: None)
    assert len(extraction.calls) == calls
    assert_unchanged(session, before)


@pytest.mark.parametrize(
    "field", ["token_ids", "assistant_end_id", "segment_mode", "token_assignments"]
)
def test_invalid_next_context_preserves_restart_anchor_and_retry_matches_clean_run(
    monkeypatch, field
):
    session, _, extraction = build_turn()
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    prepare = extraction.prepare

    def corrupt(tokens):
        entry = prepare(tokens)
        changes = {
            "token_ids": entry.token_ids[:-1],
            "assistant_end_id": entry.assistant_end_id + 1,
            "segment_mode": SegmentMode.fixed_tokens,
            "token_assignments": (0,),
        }
        return replace(entry, **{field: changes[field]})

    with monkeypatch.context() as patch:
        patch.setattr(extraction, "prepare", corrupt)
        patch.setattr(session._driver._model, "forward", lambda *_a, **_k: pytest.fail("prefill"))
        with pytest.raises(ValueError):
            session.send_user_turn("next", lambda _: None)
    assert_unchanged(session, before)
    session.rewind(0, lambda _: None)
    assert session.conversation_tokens() == before[2]
    session.send_user_turn("next", lambda _: None)
    control, _, _ = build_turn()
    control.send_user_turn("hi", lambda _: None)
    control.send_user_turn("next", lambda _: None)
    assert session.conversation_tokens() == control.conversation_tokens()
    assert session._driver.prepared_context == control._driver.prepared_context
    for actual, expected in zip(owned_state(session)[4:], owned_state(control)[4:], strict=True):
        assert t.equal(actual, expected)


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "hi"}]])
def test_prompt_renderer_rejects_missing_complete_user(messages):
    _, tokenizer, _ = build_turn()
    with pytest.raises(ValueError, match="complete user message"):
        _render_prompt(tokenizer, messages)


def test_runtime_fork_requires_prefill_and_fixed_checkpoint_is_rejected():
    session, tokenizer, extraction = build_turn()
    driver = session._driver
    with pytest.raises(RuntimeError, match="cannot fork an epoch before prefill"):
        driver.fork(epoch_id=1, revision=1, copy_runtime=True)
    setup = SimpleNamespace(**{**vars(driver._setup), "segment_mode": SegmentMode.fixed_tokens})
    with pytest.raises(ValueError, match="turn-delimited checkpoint"):
        TurnChatSession(
            setup=setup,
            config=driver._config,
            tokenizer=tokenizer,
            model=driver._model,
            device="cpu",
            extraction=extraction,
        )
    assert session.turn_state() is TurnState.BETWEEN


@pytest.mark.parametrize(
    "stage", ["prefill_receipt", "decode_receipt", "snapshot_position", "marker_slot"]
)
def test_inconsistent_runtime_receipts_fail_atomically(monkeypatch, stage):
    # Simulate a shared-runtime contract regression after the model writes its
    # candidate cache; none of those incompatible receipts may be published.
    session, _, _ = build_turn(cadence=2)
    script(monkeypatch, [12] * 30)
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    if stage in {"prefill_receipt", "decode_receipt"}:
        append = turn_module._LedgerProvider._append_from_fp_logits

        def duplicate_receipt(provider, *args, **kwargs):
            result = append(provider, *args, **kwargs)
            if (stage == "prefill_receipt") == (provider._append_ordinal == 1):
                provider.receipts.append(provider.receipts[-1])
            return result

        monkeypatch.setattr(
            turn_module._LedgerProvider, "_append_from_fp_logits", duplicate_receipt
        )
        message = "prefill append count" if stage == "prefill_receipt" else "more than one slot"
    elif stage == "snapshot_position":
        snapshot = turn_module._TurnDriver._record_snapshot

        def shifted_position(driver, *args, **kwargs):
            driver._conversation.append_tokens([12])
            return snapshot(driver, *args, **kwargs)

        monkeypatch.setattr(turn_module._TurnDriver, "_record_snapshot", shifted_position)
        message = "KV position differs"
    else:
        ledger = turn_module._TurnDriver._append_prefix_ledger

        def bad_marker(driver, *args, **kwargs):
            ledger(driver, *args, **kwargs)
            driver._ledger[-1] = replace(driver._ledger[-1], token_start=0)

        monkeypatch.setattr(turn_module._TurnDriver, "_append_prefix_ledger", bad_marker)
        message = "terminal slot is not marker-only"
    with pytest.raises(RuntimeError, match=message):
        session.send_user_turn("next", lambda _: None)
    assert_unchanged(session, before)


@pytest.mark.parametrize("stage", ["last_token", "during_prefill", "superseded_failure"])
def test_late_cancellation_and_superseded_failure_cannot_publish_context(monkeypatch, stage):
    session, _, _ = build_turn(budget=1)
    script(monkeypatch, [12] * 3)
    session.send_user_turn("hi", lambda _: None)
    before = owned_state(session)
    events = []
    if stage == "during_prefill":
        forward = session._driver._model.forward

        def cancel_after_write(*args, **kwargs):
            result = forward(*args, **kwargs)
            session.cancel()
            return result

        monkeypatch.setattr(session._driver._model, "forward", cancel_after_write)

    def event(event):
        events.append(event)
        if event.kind is GenerationEventKind.TOKEN:
            if stage == "superseded_failure":
                session.load_prefix([])
                raise RuntimeError("superseded worker failure")
            session.cancel()

    message = "superseded worker" if stage == "superseded_failure" else "cancelled"
    with pytest.raises(RuntimeError, match=message):
        session.send_user_turn("next", event)
    if stage == "superseded_failure":
        assert session._driver.prepared_context is None
        assert session.conversation_tokens() == (0,)
        assert not any(row.kind is GenerationEventKind.ERROR_ROLLBACK for row in events)
        assert not session.accepts_event(events[-1])
    else:
        assert_unchanged(session, before)
        assert events[-1].kind is GenerationEventKind.TRANSCRIPT
        assert session.accepts_event(events[-1])


@pytest.mark.parametrize("corruption", ["empty_replay", "missing_user"])
def test_inconsistent_committed_history_rejects_edit_or_rewind(monkeypatch, corruption):
    # A future snapshot/transcript bug must fail closed at the edit boundary,
    # rather than silently editing the wrong window or rendering a wrong prefix.
    session, _, _ = build_turn(cadence=2)
    script(monkeypatch, [12] * 20)
    session.send_user_turn("hi", lambda _: None)
    if corruption == "empty_replay":
        driver = session._driver
        driver._snapshots[1] = replace(
            driver._snapshots[1], token_count=driver._snapshots[0].token_count
        )
        message = "no outgoing input rows"
    else:
        session._driver.messages = session._driver.messages[1:]
        message = "logical transcript and rendered user-turn boundaries disagree"
    before = owned_state(session)
    with pytest.raises((RuntimeError, ValueError), match=message):
        if corruption == "empty_replay":
            session.modify(1, {3: 0}, lambda _: None)
        else:
            session.rewind_to_prefix(0, lambda _: None)
    assert_unchanged(session, before)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cadence", [1, 2, 4])
def test_cuda_turn_rewind_and_failed_next_epoch_retry_equality(monkeypatch, cadence):
    # This is a real tiny-model CUDA runtime gate, NOT Gemma artifact acceptance.
    session, _, extraction = build_turn(cadence=cadence, budget=9, device="cuda")
    session.send_user_turn("hi", lambda _: None)
    first = session.conversation_tokens()
    session.rewind(0, lambda _: None)
    assert session.conversation_tokens() == first
    for ordinal in range(1, len(session._driver._snapshots)):
        session.rewind(ordinal, lambda _: None)
        assert session.conversation_tokens() == first
    before = owned_state(session)
    extraction.failure = RuntimeError("injected GPU epoch extraction failure")
    with pytest.raises(RuntimeError, match="injected"):
        session.send_user_turn("next", lambda _: None)
    assert_unchanged(session, before)
    extraction.failure = None
    session.send_user_turn("next", lambda _: None)
    control, _, _ = build_turn(cadence=cadence, budget=9, device="cuda")
    control.send_user_turn("hi", lambda _: None)
    control.send_user_turn("next", lambda _: None)
    assert session.conversation_tokens() == control.conversation_tokens()
    assert t.equal(
        session._driver._selection_rng.get_state(), control._driver._selection_rng.get_state()
    )
