"""Headless Textual/worker checks over the real turn session driver."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as flash_attention
from nanochat.flash_attention import FlashAttnImpl
from scripts.chat_tui.app import ChatApp, ErrorModal
from scripts.chat_tui.session import GenerationEvent, GenerationEventKind, SegmentKind
from scripts.chat_tui.steering import SteeringMethod, SteeringSelection
from scripts.chat_tui.widgets.steer_menu import SteerMenu
from tests.test_chat_tui.test_app_shell import _wait_for
from tests.test_chat_tui.test_steer_menu import StubSignalsSource
from tests.test_chat_tui.test_turn_session import build_turn, script

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def sdpa(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)


async def test_turn_ui_worker_send_seed_clear_and_stale_preview(monkeypatch):
    session, tokenizer, extraction = build_turn()
    stop = tokenizer.encode_special("<assistant_end>")
    script(monkeypatch, [12, stop, 13, stop])
    app = ChatApp(session=session, tokenizer=tokenizer, checkpoint_label="turn-tiny", seed=17)
    async with app.run_test(size=(120, 40)) as pilot:
        app._send("hi")
        await _wait_for(pilot, lambda: not app.busy)
        first_identity = session.identity
        app._send("next")
        await _wait_for(pilot, lambda: not app.busy)
        assert len(extraction.calls) == 2
        assert app._tokens == session.conversation_tokens()
        assert not app.query(ErrorModal)
        text = app._transcript_text
        app._active_sequence = 999
        app._apply_event(
            999,
            GenerationEvent(
                GenerationEventKind.TRANSCRIPT,
                None,
                "stale transcript",
                *first_identity,
            ),
        )
        assert app._transcript_text == text
        app._active_sequence = None
        app._handle_command("/seed 123")
        await _wait_for(pilot, lambda: not app.busy)
        assert app._pending_seed == 123
        app._handle_command("/clear")
        await _wait_for(pilot, lambda: not app.busy)
        assert app._pending_seed is None
        assert session.conversation_tokens() == (0,)
        assert session._driver._token_rng.initial_seed() == 123


async def test_failed_turn_steering_preserves_committed_policy_and_pending_seed(monkeypatch):
    session, tokenizer, _ = build_turn(deterministic=False)
    script(monkeypatch, [12] * 14)
    session.send_user_turn("hi", lambda _: None)
    session.set_seed(123)
    before = session._driver
    app = ChatApp(session=session, tokenizer=tokenizer, checkpoint_label="turn-tiny", seed=17)
    app._pending_seed = 123
    outcomes = []
    menu: Any = SimpleNamespace(
        on_done=lambda success, message: outcomes.append((success, message)),
        enabled_state=lambda: (SteeringMethod.OFF, ()),
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected steering prefill failure")

    monkeypatch.setattr(session._driver._model, "forward", fail)
    async with app.run_test(size=(120, 40)) as pilot:
        app._submit_steering(menu, SteeringSelection(SteeringMethod.OFF, ()))
        await _wait_for(pilot, lambda: not app.busy)
        assert outcomes and outcomes[-1][0] is False
        assert "injected" in outcomes[-1][1]
        assert session._driver is before
        assert app._pending_seed == 123
        assert app._transcript_text == session.transcript_spans()[0]


async def test_deterministic_turn_s0_edit_does_not_consume_pending_seed(monkeypatch):
    session, tokenizer, _ = build_turn()
    script(monkeypatch, [12] * 14)
    session.send_user_turn("hi", lambda _: None)
    session.set_seed(123)
    row = next(row for row in session.ledger() if row.selfcond_ordinal == 0)
    app = ChatApp(
        session=session,
        tokenizer=tokenizer,
        checkpoint_label="turn-tiny",
        seed=17,
        deterministic_rewind=True,
    )
    app._pending_seed = 123
    outcomes = []
    menu: Any = SimpleNamespace(
        on_done=lambda success, message: outcomes.append((success, message))
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app._modify_segment(menu, replace(row), {3: 0})
        await _wait_for(pilot, lambda: not app.busy)
        assert outcomes == [(True, None)]
        assert app._pending_seed == 123


async def test_editing_a_real_history_row_reports_read_only_and_keeps_the_app_alive(
    monkeypatch,
):
    # kills: letting a real-history row enter the editor, whose confirmation
    # render then raises outside any handler and terminates the Textual app.
    session, tokenizer, _ = build_turn()
    script(monkeypatch, [12] * 14)
    session.send_user_turn("hi", lambda _: None)
    assert session.ledger()[0].kind is SegmentKind.PREFIX_REAL
    source: Any = StubSignalsSource()
    app = ChatApp(
        session=session,
        tokenizer=tokenizer,
        checkpoint_label="turn-tiny",
        seed=17,
        signals_source=source,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_command("/steer")
        await _wait_for(pilot, lambda: bool(app.query(SteerMenu)))
        menu = app.query_one(SteerMenu)
        await _wait_for(pilot, lambda: app.focused is menu)
        await pilot.press("e")
        await pilot.pause()
        assert menu._edited_features is None
        assert menu._status_message is not None and "read-only" in menu._status_message
        # Enter now offers the row's rewind, never a modify confirmation.
        await pilot.press("enter", "n")
        await pilot.pause()
        assert app.is_running and menu.is_mounted
        assert menu._edited_features is None
        # A legitimate edit on a predicted row clears the refusal message.
        await pilot.press("down", "e")
        await pilot.pause()
        assert menu._edited_features is not None
        assert menu._status_message is None
        assert session.conversation_tokens()[-1] == tokenizer.encode_special("<assistant_end>")


async def test_turn_steer_menu_drops_fit_results_and_submissions_after_revision_change():
    session, tokenizer, _ = build_turn()
    source: Any = StubSignalsSource()
    app = ChatApp(
        session=session,
        tokenizer=tokenizer,
        checkpoint_label="turn-tiny",
        seed=17,
        signals_source=source,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        menu = app._new_steer_menu()
        menu._enabled.add(10)
        status = menu._status_message
        session.set_seed(123)
        menu._fit_ready(10, source.signals[10])
        menu._fit_error(10, "stale error")
        assert not menu._ready_signals
        assert menu._enabled == {10}
        assert menu._status_message == status
        before = session._driver
        outcomes = []
        monkey_menu: Any = SimpleNamespace(
            on_done=lambda success, message: outcomes.append((success, message)),
            enabled_state=lambda: (SteeringMethod.OFF, ()),
        )
        app._submit_steering(
            monkey_menu, SteeringSelection(SteeringMethod.OFF, ()), turn_identity=(0, 0)
        )
        await _wait_for(pilot, lambda: not app.busy)
        assert session._driver is before
        assert outcomes and outcomes[-1][0] is False
        assert "earlier Chat epoch/revision" in outcomes[-1][1]


@pytest.mark.parametrize("fail", [False, True])
async def test_turn_signal_menu_is_hidden_while_regeneration_streams(monkeypatch, fail):
    session, tokenizer, _ = build_turn()
    script(monkeypatch, [12] * 14)
    session.send_user_turn("Hi, write me a poem", lambda _: None)
    source = StubSignalsSource(cached=True)
    source.signals[10] = replace(
        source.signals[10], target_scores=t.ones(session._driver._setup.n_sae_features)
    )
    app = ChatApp(
        session=session,
        tokenizer=tokenizer,
        checkpoint_label="turn-tiny",
        seed=17,
        signals_source=source,
    )
    started = threading.Event()
    release = threading.Event()
    original = session.apply_steering
    visible_at_start = []
    committed = session._driver

    def stream(selection, on_event, *, cancel_check=None):
        visible_at_start.append(
            app.call_from_thread(lambda: menu in app.screen._compositor.visible_widgets)
        )

        def emit(event):
            on_event(event)
            if event.kind is GenerationEventKind.TOKEN and not started.is_set():
                started.set()
                if not release.wait(timeout=10):
                    raise RuntimeError("test did not release generation")
                if fail:
                    raise RuntimeError("injected mid-stream steering failure")

        original(selection, emit, cancel_check=cancel_check)

    monkeypatch.setattr(session, "apply_steering", stream)
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_command("/steer")
        await pilot.pause()
        menu = app.query_one(".chat-modal")
        await pilot.press("tab", "m", "space", *(["right"] * 11))
        assert menu.strengths[10] == 6.0
        try:
            await pilot.press("enter")
            await _wait_for(pilot, started.is_set)
            assert app.busy
            assert menu not in app.screen._compositor.visible_widgets
        finally:
            release.set()
            await _wait_for(pilot, lambda: not app.busy)
        assert visible_at_start == [False]
        if fail:
            assert menu.is_mounted and menu.display
            assert menu.strengths[10] == 6.0
            assert session._driver is committed
            assert "injected mid-stream" in str(app.query_one(ErrorModal).content)
        else:
            # busy=False is published synchronously; Textual unmounts the
            # successful menu on a later message-loop tick.
            await _wait_for(pilot, lambda: not app.query(".chat-modal"))
            assert not app.query(ErrorModal)
            assert session._driver is not committed
            assert "Σf=6.0" in app.status_text
