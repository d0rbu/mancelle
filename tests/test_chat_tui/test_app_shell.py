"""Textual shell tests for the chat TUI."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

import scripts.chat_tui.app as app_module
from scripts.chat_tui.app import ChatApp, ErrorModal, HelpModal, ModelThread
from scripts.chat_tui.session import (
    GenerationEvent,
    GenerationEventKind,
    LedgerSegment,
    SegmentKind,
    SegmentProvenance,
    TextDelta,
    TranscriptSpan,
    TurnState,
)
from scripts.completion_tui.color_manager import color_for_idx


class AsciiTokenizer:
    """chr-per-token double; special ids live beyond the Unicode range and
    decode to their real-tokenizer-style literals."""

    _SPECIALS = {
        "<bos>": 0x110000,
        "<user_start>": 0x110001,
        "<user_end>": 0x110002,
        "<assistant_start>": 0x110003,
        "<assistant_end>": 0x110004,
    }

    def decode(self, tokens: tuple[int, ...] | list[int]) -> str:
        by_id = {token_id: name for name, token_id in self._SPECIALS.items()}
        return "".join(by_id.get(token) or chr(token) for token in tokens)

    def get_bos_token_id(self) -> int:
        return self._SPECIALS["<bos>"]

    def encode_special(self, name: str) -> int:
        return self._SPECIALS[name]


class FakeSession:
    """Tensors-free session double at the public ``ChatSession`` boundary."""

    def __init__(self) -> None:
        self.tokens: tuple[int, ...] = ()
        self.segments: tuple[LedgerSegment, ...] = ()
        self.total_tokens = 32
        self.scripted_events: list[GenerationEvent] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.cancelled = threading.Event()
        self.cancel_calls = 0
        self.open_user_text_value: str | None = None
        self.sent_texts: list[str] = []
        self.sampled_tokens = 0
        self.send_threads: list[str] = []
        self.clear_threads: list[str] = []
        self.seed_calls: list[tuple[int, str]] = []
        self.callbacks: list[Callable[[GenerationEvent], None]] = []
        self.send_error: Exception | None = None
        self.authoritative: tuple[str, tuple[TranscriptSpan, ...]] | None = None

    def send_user_turn(
        self,
        text: str,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.sent_texts.append(text)
        self.send_threads.append(threading.current_thread().name)
        self.callbacks.append(on_event)
        self.started.set()
        for event in self.scripted_events:
            if event.text_delta is not None:
                self.tokens += tuple(ord(char) for char in event.text_delta.appended)
            on_event(event)
        for _ in range(200):
            if self.release.wait(timeout=0.01):
                break
            if cancel_check is not None and cancel_check():
                break
        else:
            raise AssertionError("test did not release fake generation")
        if cancel_check is not None and cancel_check():
            on_event(_event(GenerationEventKind.TURN_CLOSED))
        else:
            self.sampled_tokens += 1
        if self.send_error is not None:
            raise self.send_error

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.cancelled.set()
        self.release.set()

    def clear(self) -> None:
        self.clear_threads.append(threading.current_thread().name)
        self.tokens = ()
        self.segments = ()

    def regenerate(self, seed, on_event, *, cancel_check=None) -> None:
        self.set_seed(seed)
        self.send_user_turn("", on_event, cancel_check=cancel_check)

    def set_seed(self, seed: int) -> None:
        if seed <= 0:
            raise ValueError(f"seed must be a positive int, got {seed!r}")
        self.seed_calls.append((seed, threading.current_thread().name))

    def conversation_tokens(self) -> tuple[int, ...]:
        return self.tokens

    def ledger(self) -> tuple[LedgerSegment, ...]:
        return self.segments

    def token_budget(self) -> tuple[int, int]:
        return len(self.tokens), self.total_tokens

    def transcript_spans(self) -> tuple[str, tuple[TranscriptSpan, ...]]:
        if self.authoritative is not None:
            return self.authoritative
        text = AsciiTokenizer().decode(self.tokens)
        spans = tuple(
            TranscriptSpan(
                char_start=segment.token_start,
                char_end=segment.token_end,
                color_idx=segment.color_idx,
                kind=segment.kind,
                is_live=index == len(self.segments) - 1 and segment.kind is SegmentKind.SELFCOND,
            )
            for index, segment in enumerate(self.segments)
        )
        return text, spans

    def turn_state(self) -> TurnState:
        return TurnState.BETWEEN

    def open_user_text(self) -> str | None:
        return self.open_user_text_value


def _event(
    kind: GenerationEventKind,
    appended: str | None = None,
    *,
    detail: str = "",
) -> GenerationEvent:
    return GenerationEvent(
        kind=kind,
        text_delta=None if appended is None else TextDelta(appended, None),
        detail=detail,
    )


def _app(session: FakeSession, *, seed: int = 11) -> ChatApp:
    return ChatApp(
        session=session,  # type: ignore[arg-type]
        tokenizer=AsciiTokenizer(),
        checkpoint_label="tiny.ckpt",
        seed=seed,
    )


async def _wait_for(pilot: Any, predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not reached")


async def _enter(pilot: Any, text: str) -> None:
    await pilot.press(*text, "enter")


def _style_at(rendered: Text, position: int) -> str:
    styles = [str(span.style) for span in rendered._spans if span.start <= position < span.end]
    return " ".join(styles)


def test_model_thread_serializes_jobs_and_survives_leaked_exception() -> None:
    model_thread = ModelThread()
    timeline: list[tuple[str, str, float]] = []
    finished = threading.Event()

    def job(name: str, *, fail: bool = False) -> Callable[[], None]:
        def run() -> None:
            timeline.append((name, "enter", time.monotonic()))
            time.sleep(0.02)
            timeline.append((name, "exit", time.monotonic()))
            if fail:
                raise RuntimeError("intentional leaked job error")
            if name == "second":
                finished.set()

        return run

    model_thread.start()
    model_thread.submit(job("first", fail=True))
    model_thread.submit(job("second"))
    assert finished.wait(timeout=2)
    model_thread.shutdown()

    first_exit = next(at for name, phase, at in timeline if (name, phase) == ("first", "exit"))
    second_enter = next(at for name, phase, at in timeline if (name, phase) == ("second", "enter"))
    assert first_exit <= second_enter
    assert not model_thread.is_alive


@pytest.mark.anyio
@pytest.mark.parametrize("size", [(80, 24), (130, 48)])
async def test_status_bar_is_visible_above_input_without_overlap(size) -> None:
    app = _app(FakeSession())
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        status = app.query_one("#status", Static)
        input_widget = app.query_one("#chat-input", Input)
        transcript = app.query_one("#transcript-scroll", VerticalScroll)
        assert transcript.region.bottom <= status.region.y
        assert status.region.bottom <= input_widget.region.y
        assert input_widget.region.bottom <= size[1]
        assert "tiny.ckpt" in app.export_screenshot()
        await _enter(pilot, "/seed 7")
        await _wait_for(pilot, lambda: not app.busy)
        await pilot.pause()
        assert "next reset" in app.status_text
        assert "next&#160;reset" in app.export_screenshot()


@pytest.mark.anyio
async def test_send_flow_streams_in_order_on_model_thread_and_reenables_input() -> None:
    session = FakeSession()
    session.scripted_events = [
        _event(GenerationEventKind.TOKEN, "A"),
        _event(GenerationEventKind.TOKEN, "B"),
        _event(GenerationEventKind.TURN_CLOSED),
    ]
    session.release.clear()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, session.started.is_set)
        assert app.query_one("#chat-input", Input).disabled
        assert app.transcript.plain == "AB"
        assert session.send_threads == ["chat-model"]

        session.release.set()
        await _wait_for(pilot, lambda: not app.busy)
        assert not app.query_one("#chat-input", Input).disabled
        assert app.transcript.plain == "AB"


@pytest.mark.anyio
async def test_enter_while_busy_requests_cancel_once_and_waits_for_terminal_event() -> None:
    session = FakeSession()
    session.release.clear()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, session.started.is_set)
        await pilot.press("enter", "enter")
        await _wait_for(pilot, lambda: not app.busy)
        await _wait_for(pilot, lambda: not app.busy)

        assert session.cancel_calls == 0
        assert session.sampled_tokens == 0
        assert not app.query_one("#chat-input", Input).disabled


@pytest.mark.anyio
async def test_cancel_while_send_is_queued_is_observed_before_first_sample() -> None:
    session = FakeSession()
    app = _app(session)
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def slow_fit() -> None:
        blocker_started.set()
        assert release_blocker.wait(timeout=2), "test did not release queued fit"

    async with app.run_test() as pilot:
        app.model_thread.submit(slow_fit)
        await _wait_for(pilot, blocker_started.is_set)
        await _enter(pilot, "hello")
        assert not session.started.is_set()
        await pilot.press("enter")
        release_blocker.set()
        await _wait_for(pilot, lambda: not app.busy)

        assert session.sampled_tokens == 0
        assert session.cancel_calls == 0


@pytest.mark.anyio
async def test_events_from_a_finished_generation_are_stale() -> None:
    session = FakeSession()
    session.scripted_events = [_event(GenerationEventKind.TOKEN, "A")]
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, lambda: not app.busy and bool(session.callbacks))
        before = app.transcript.plain

        stale_done = threading.Event()
        stale_thread = threading.Thread(
            target=lambda: (
                session.callbacks[0](_event(GenerationEventKind.TOKEN, "STALE")),
                stale_done.set(),
            )
        )
        stale_thread.start()
        await _wait_for(pilot, stale_done.is_set)
        stale_thread.join()
        assert app.transcript.plain == before


@pytest.mark.anyio
async def test_clear_and_seed_are_serialized_and_seed_stays_pending_until_clear() -> None:
    session = FakeSession()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "/seed 7")
        await _wait_for(pilot, lambda: session.seed_calls != [])
        assert session.seed_calls == [(7, "chat-model")]
        assert "seed 7 (next reset)" in app.status_text

        await _enter(pilot, "/clear")
        await _wait_for(pilot, lambda: session.clear_threads != [])
        await _wait_for(pilot, lambda: not app.busy)
        assert session.clear_threads == ["chat-model"]
        assert "seed 7 (next reset)" not in app.status_text
        assert "seed 7" in app.status_text


@pytest.mark.anyio
async def test_unavailable_and_unknown_slash_commands_notify_errors() -> None:
    session = FakeSession()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "/convo")
        await _enter(pilot, "/steer")
        await _enter(pilot, "/xyz")
        messages = [notification.message for notification in app._notifications]
        assert "convo unavailable" in messages
        assert "steer unavailable" in messages
        assert "unknown command: /xyz (see /help)" in messages


@pytest.mark.anyio
async def test_factory_modals_cleanly_remount_after_open_close_churn() -> None:
    session = FakeSession()
    app = ChatApp(
        session=session,  # type: ignore[arg-type]
        tokenizer=AsciiTokenizer(),
        checkpoint_label="tiny.ckpt",
        seed=11,
        convo_modal_factory=lambda: Static("convo modal"),
        steer_modal_factory=lambda: Static("steer modal"),
    )

    async with app.run_test() as pilot:
        for command, content in (("/convo", "convo modal"), ("/steer", "steer modal")):
            for _ in range(2):
                await _enter(pilot, command)
                modals = list(app.query(".chat-modal"))
                assert len(modals) == 1
                assert str(modals[0].content) == content
                await pilot.press("escape")
                await pilot.pause()
                assert len(app.query(".chat-modal")) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("command", ["/quit", "/exit"])
async def test_quit_command_exits_app(command: str) -> None:
    session = FakeSession()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, command)
        await _wait_for(pilot, lambda: not app.is_running)

    assert not app.model_thread.is_alive


@pytest.mark.anyio
async def test_colored_transcript_dims_prefix_underlines_live_and_toggles_plain() -> None:
    session = FakeSession()
    session.tokens = tuple(ord(char) for char in "abcdef")
    session.segments = (
        LedgerSegment(
            kind=SegmentKind.PREFIX_REAL,
            encoder_slot_idx=0,
            selfcond_ordinal=None,
            token_start=0,
            token_end=3,
            features={1: 0},
            provenance=SegmentProvenance.REAL,
            steering=None,
            color_idx=2,
        ),
        LedgerSegment(
            kind=SegmentKind.SELFCOND,
            encoder_slot_idx=1,
            selfcond_ordinal=0,
            token_start=3,
            token_end=6,
            features={2: 1},
            provenance=SegmentProvenance.PREDICTED,
            steering=None,
            color_idx=4,
        ),
    )
    app = _app(session)

    async with app.run_test() as pilot:
        rendered = app.transcript
        assert rendered.plain == "abcdef"
        assert color_for_idx(2) in _style_at(rendered, 1)
        assert "dim" in _style_at(rendered, 1)
        assert color_for_idx(4) in _style_at(rendered, 4)
        assert "underline" in _style_at(rendered, 4)

        await pilot.press("ctrl+t")
        plain = app.transcript
        assert plain.plain == "abcdef"
        assert plain._spans == []


@pytest.mark.anyio
async def test_utf8_rendering_uses_authoritative_text_and_span_offsets() -> None:
    session = FakeSession()
    full_text = "A✅B"
    session.authoritative = (
        full_text,
        (
            TranscriptSpan(0, 1, 2, SegmentKind.PREFIX_REAL, False),
            TranscriptSpan(1, 2, 4, SegmentKind.SELFCOND, False),
            TranscriptSpan(2, 3, 6, SegmentKind.SELFCOND, True),
        ),
    )
    app = _app(session)
    # The authoritative-passthrough contract under test is the RAW pipeline;
    # pretty mode re-derives text from tokens, which this fake divorces.
    app._raw_mode = True

    async with app.run_test():
        rendered = app.transcript
        assert rendered.plain == full_text
        assert color_for_idx(2) in _style_at(rendered, 0)
        assert "dim" in _style_at(rendered, 0)
        assert color_for_idx(4) in _style_at(rendered, 1)
        assert color_for_idx(6) in _style_at(rendered, 2)
        assert "underline" in _style_at(rendered, 2)


@pytest.mark.anyio
async def test_status_bar_tracks_budget_and_busy_generation() -> None:
    session = FakeSession()
    session.tokens = (ord("x"), ord("y"))
    session.release.clear()
    app = _app(session)

    async with app.run_test() as pilot:
        assert app.status_text == "tiny.ckpt · tokens 2/32 · steer: off · seed 11"

        await _enter(pilot, "hello")
        await _wait_for(pilot, session.started.is_set)
        assert "· generating…" in app.status_text

        session.release.set()
        await _wait_for(pilot, lambda: not app.busy)
        assert "generating…" not in app.status_text


@pytest.mark.anyio
async def test_error_rollback_detail_is_notified_and_flashes_in_status() -> None:
    session = FakeSession()
    detail = "rolled back to boundary 2 (kv 17): empty segment"
    session.scripted_events = [
        _event(GenerationEventKind.ERROR_ROLLBACK, detail=detail),
    ]
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, lambda: not app.busy)
        assert detail in app.status_text
        assert detail in [notification.message for notification in app._notifications]
        modal = app.query_one(ErrorModal)
        assert detail in str(modal.content)
        assert "esc to dismiss" in str(modal.content)
        assert app.query_one("#status", Static)


@pytest.mark.anyio
async def test_error_rollback_detail_survives_the_terminal_error_modal() -> None:
    """When a rollback event is followed by the job's exception, the final modal
    must keep the rollback point instead of the bare exception text."""

    session = FakeSession()
    detail = "rolled back to boundary 2 (kv 17): decode exploded"
    session.scripted_events = [
        _event(GenerationEventKind.ERROR_ROLLBACK, detail=detail),
    ]
    session.send_error = RuntimeError("decode exploded")
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, lambda: not app.busy)
        assert detail in str(app.query_one(ErrorModal).content)
        assert detail in app.status_text
        assert not app.query_one("#chat-input", Input).disabled


@pytest.mark.anyio
async def test_open_user_text_prefills_input_and_send_forces_only_the_suffix() -> None:
    """Spec: a rewind landing inside a user turn prefills the input line with
    the partial turn text; submitting the extended text forces only the suffix
    (the committed partial lives in the KV at the rewind point)."""

    session = FakeSession()
    session.open_user_text_value = "par"
    app = _app(session)

    async with app.run_test() as pilot:
        assert app.query_one("#chat-input", Input).value == "par"
        session.open_user_text_value = "par"
        await _enter(pilot, "tial")
        await _wait_for(pilot, lambda: session.sent_texts != [])
        assert session.sent_texts == ["tial"]


@pytest.mark.anyio
async def test_non_extending_edit_of_the_partial_user_turn_is_rejected() -> None:
    session = FakeSession()
    session.open_user_text_value = "par"
    app = _app(session)

    async with app.run_test() as pilot:
        input_widget = app.query_one("#chat-input", Input)
        assert input_widget.value == "par"
        input_widget.value = "xyz"
        await pilot.press("enter")
        await pilot.pause()
        assert session.sent_texts == []
        assert input_widget.value == "xyz"
        assert any(
            "can only be extended" in notification.message for notification in app._notifications
        )


@pytest.mark.anyio
async def test_session_exception_is_notified_and_reenables_input() -> None:
    session = FakeSession()
    session.send_error = RuntimeError("decode exploded")
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, lambda: not app.busy)
        assert "decode exploded" in app.status_text
        assert "decode exploded" in [notification.message for notification in app._notifications]
        assert "decode exploded" in str(app.query_one(ErrorModal).content)
        assert not app.query_one("#chat-input", Input).disabled


def _pretty_fixture_session() -> tuple[FakeSession, int, int]:
    """One settled exchange with real special ids and token-aligned spans.

    AsciiTokenizer decodes specials to their multi-char literals, so span char
    offsets are computed from the decoded literal widths, exactly as the
    session derives them from token char offsets.
    """

    tokenizer = AsciiTokenizer()
    session = FakeSession()
    tokens = (
        tokenizer.encode_special("<bos>"),
        tokenizer.encode_special("<user_start>"),
        *(ord(char) for char in "hi"),
        tokenizer.encode_special("<user_end>"),
        tokenizer.encode_special("<assistant_start>"),
        *(ord(char) for char in "yo"),
        tokenizer.encode_special("<assistant_end>"),
    )
    session.tokens = tokens
    offsets = [0]
    for token in tokens:
        offsets.append(offsets[-1] + len(tokenizer.decode([token])))
    assistant_start_offset = offsets[tokens.index(tokenizer.encode_special("<assistant_start>"))]
    session.segments = (
        LedgerSegment(
            kind=SegmentKind.PREFIX_REAL,
            encoder_slot_idx=0,
            selfcond_ordinal=None,
            token_start=0,
            token_end=assistant_start_offset,
            features={},
            provenance=SegmentProvenance.PREDICTED,
            steering=None,
            color_idx=2,
        ),
        LedgerSegment(
            kind=SegmentKind.SELFCOND,
            encoder_slot_idx=1,
            selfcond_ordinal=0,
            token_start=assistant_start_offset,
            token_end=offsets[-1],
            features={},
            provenance=SegmentProvenance.PREDICTED,
            steering=None,
            color_idx=5,
        ),
    )
    return session, assistant_start_offset, offsets[-1]


@pytest.mark.anyio
async def test_default_pretty_mode_replaces_markers_and_keeps_segment_colors() -> None:
    session, _assistant_start_offset, _end = _pretty_fixture_session()
    app = _app(session)

    async with app.run_test():
        rendered = app.transcript
        assert rendered.plain == "› hi\n\n• yo"
        assert "raw" not in app.status_text
        # Prefix chars (marker + user text) carry the dim prefix color; the
        # assistant marker inherits the S0 segment color it stands in for.
        assert color_for_idx(2) in _style_at(rendered, 0)
        assert "dim" in _style_at(rendered, 0)
        assistant_marker_position = rendered.plain.index("•")
        assert color_for_idx(5) in _style_at(rendered, assistant_marker_position)
        assert color_for_idx(5) in _style_at(rendered, len(rendered.plain) - 1)


@pytest.mark.anyio
async def test_raw_command_toggles_token_rendering_both_ways() -> None:
    session, _assistant_start_offset, _end = _pretty_fixture_session()
    app = _app(session)
    raw_text = "<bos><user_start>hi<user_end><assistant_start>yo<assistant_end>"

    async with app.run_test() as pilot:
        assert app.transcript.plain == "› hi\n\n• yo"

        await _enter(pilot, "/raw")
        assert app.transcript.plain == raw_text
        assert "· raw" in app.status_text
        rendered = app.transcript
        assistant_literal_position = raw_text.index("<assistant_start>")
        assert color_for_idx(5) in _style_at(rendered, assistant_literal_position)

        await _enter(pilot, "/raw")
        assert app.transcript.plain == "› hi\n\n• yo"
        assert "· raw" not in app.status_text


@pytest.mark.anyio
async def test_streaming_display_substitutes_marker_literals() -> None:
    session = FakeSession()
    session.scripted_events = [
        _event(GenerationEventKind.TOKEN, "<user_start>hi<user_end><assistant_start>"),
        _event(GenerationEventKind.TOKEN, "yo"),
        _event(GenerationEventKind.TURN_CLOSED),
    ]
    session.release.clear()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "hello")
        await _wait_for(pilot, session.started.is_set)
        assert app.transcript.plain == "› hi\n\n• yo"
        session.release.set()
        await _wait_for(pilot, lambda: not app.busy)


@pytest.mark.anyio
@pytest.mark.parametrize("invocation", ["command", "binding"])
async def test_copy_transcript_copies_displayed_text_and_mirrors_to_a_file(
    invocation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """/copy and ctrl+y copy exactly what is on screen (pretty markers included)
    and mirror it to a file, because OSC 52 can be declined silently."""

    session, _assistant_start_offset, _end = _pretty_fixture_session()
    mirror = tmp_path / "chat_tui" / "transcript.txt"
    monkeypatch.setattr(app_module, "TRANSCRIPT_COPY_PATH", str(mirror))
    app = _app(session)
    copied: list[str] = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        if invocation == "command":
            await _enter(pilot, "/copy")
        else:
            await pilot.press("ctrl+y")
        await pilot.pause()

    assert copied == ["› hi\n\n• yo"]
    assert mirror.read_text() == "› hi\n\n• yo"
    assert any(
        f"copied {len('› hi\n\n• yo')} characters" in notification.message
        and str(mirror) in notification.message
        for notification in app._notifications
    )


@pytest.mark.anyio
async def test_copy_transcript_in_raw_mode_copies_the_raw_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session, _assistant_start_offset, _end = _pretty_fixture_session()
    monkeypatch.setattr(app_module, "TRANSCRIPT_COPY_PATH", str(tmp_path / "transcript.txt"))
    app = _app(session)
    copied: list[str] = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        await _enter(pilot, "/raw")
        await _enter(pilot, "/copy")
        await pilot.pause()

    assert copied == ["<bos><user_start>hi<user_end><assistant_start>yo<assistant_end>"]


@pytest.mark.anyio
async def test_transcript_pane_scrolls_to_the_end_of_a_long_conversation() -> None:
    """A bare Static clips instead of scrolling, which left everything past the
    pane height unreachable in the chat view."""

    session = FakeSession()
    session.authoritative = ("\n".join(f"line {index}" for index in range(200)), ())
    app = _app(session)
    app._raw_mode = True

    async with app.run_test() as pilot:
        await pilot.pause()
        pane = app.query_one("#transcript-scroll", VerticalScroll)
        assert pane.size.height > 0
        assert pane.max_scroll_y > 0, "the transcript pane is not scrollable"
        assert pane.scroll_offset.y == pane.max_scroll_y, "did not follow the newest output"


@pytest.mark.anyio
async def test_diag_dumps_tokens_segments_and_steering_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A steering bug report needs the state the transcript cannot carry: token
    ids, per-segment provenance, and the exact recorded feature dicts."""

    session, _assistant_start_offset, _end = _pretty_fixture_session()
    path = tmp_path / "diagnostics.json"
    monkeypatch.setattr(app_module, "DIAGNOSTICS_PATH", str(path))
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "/diag")
        await pilot.pause()

    payload = json.loads(path.read_text())
    assert payload["checkpoint"] == "tiny.ckpt"
    assert payload["seed"] == 11
    assert payload["conversation_tokens"] == list(session.tokens)
    assert payload["displayed"] == "› hi\n\n• yo"
    assert payload["applied_steering"] == ["off", []]
    assert payload["raw_mode"] is False
    kinds = [segment["kind"] for segment in payload["segments"]]
    assert kinds == [SegmentKind.PREFIX_REAL.value, SegmentKind.SELFCOND.value]
    assert payload["segments"][1]["selfcond_ordinal"] == 0
    assert payload["segments"][1]["provenance"] == SegmentProvenance.PREDICTED.value
    assert payload["segments"][1]["steering"] is None
    assert any("diagnostics" in notification.message for notification in app._notifications)


@pytest.mark.anyio
async def test_help_modal_lists_commands_and_dismisses_on_escape() -> None:
    session = FakeSession()
    app = _app(session)

    async with app.run_test() as pilot:
        await _enter(pilot, "/help")
        modal = app.query_one(HelpModal)
        content = str(modal.content)
        for command in ("/convo", "/steer", "/clear", "/seed N", "/raw", "/help", "/quit"):
            assert command in content
        assert "esc to dismiss" in content

        await pilot.press("escape")
        await pilot.pause()
        assert not list(app.query(HelpModal))


@pytest.mark.anyio
async def test_unmount_shuts_down_description_service_once() -> None:
    class DescriptionSpy:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    service = DescriptionSpy()
    app = ChatApp(
        session=FakeSession(),  # type: ignore[arg-type]
        tokenizer=AsciiTokenizer(),
        checkpoint_label="tiny.ckpt",
        seed=11,
        description_service=service,  # type: ignore[arg-type]
    )

    async with app.run_test():
        pass

    assert service.shutdown_calls == 1
    app._shutdown_description_service()
    assert service.shutdown_calls == 1


@pytest.mark.anyio
async def test_regen_advances_pending_seed_on_model_thread() -> None:
    session = FakeSession()
    app = _app(session)
    async with app.run_test() as pilot:
        await _enter(pilot, "/seed 7")
        await _wait_for(pilot, lambda: not app.busy)
        await _enter(pilot, "/regen")
        await _wait_for(pilot, lambda: not app.busy)
        assert session.seed_calls == [(7, "chat-model"), (8, "chat-model")]
        assert "seed 8" in app.status_text
        assert "next reset" not in app.status_text
        await _enter(pilot, "/regen extra")
        assert session.seed_calls == [(7, "chat-model"), (8, "chat-model")]


@pytest.mark.anyio
async def test_failed_regen_keeps_displayed_seed() -> None:
    session = FakeSession()
    session.send_error = RuntimeError("regen failed")
    app = _app(session)
    async with app.run_test() as pilot:
        await _enter(pilot, "/regen")
        await _wait_for(pilot, lambda: not app.busy)
        assert app._seed == 11
        assert len(app.query(ErrorModal)) == 1
