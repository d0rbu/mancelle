"""Pilot tests for the lazy ``/convo`` picker."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.text import Text
from textual.widgets import Input, Static

from scripts.chat_tui.app import ChatApp
from scripts.chat_tui.convo import ConvoEntry, prefix_messages
from scripts.chat_tui.session import TranscriptSpan
from scripts.chat_tui.widgets.convo_picker import ConvoPicker, PickerMode


class MessageTokenizer:
    def render_conversation(
        self,
        conversation: dict[str, list[dict]],
        *,
        max_tokens: int,
    ) -> tuple[list[int], list[int]]:
        del max_tokens
        text = "|".join(
            f"{message['role']}:{message.get('content', '')}"
            for message in conversation["messages"]
        )
        return list(text.encode()), [1] * len(text.encode())

    def decode(self, tokens: tuple[int, ...] | list[int]) -> str:
        by_id = {token_id: name for name, token_id in self._SPECIALS.items()}
        if any(token in by_id for token in tokens):
            return "".join(
                by_id[token] if token in by_id else bytes([token]).decode() for token in tokens
            )
        return bytes(tokens).decode()

    _SPECIALS = {
        "<bos>": 0x110000,
        "<user_start>": 0x110001,
        "<user_end>": 0x110002,
        "<assistant_start>": 0x110003,
        "<assistant_end>": 0x110004,
    }

    def get_bos_token_id(self) -> int:
        return self._SPECIALS["<bos>"]

    def encode_special(self, name: str) -> int:
        return self._SPECIALS[name]


class StubCatalog:
    def __init__(self, entries: list[ConvoEntry]) -> None:
        self.entries = entries
        self.entry_threads: list[str] = []

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, index: int) -> ConvoEntry:
        self.entry_threads.append(threading.current_thread().name)
        return self.entries[index]


class PrefixSession:
    def __init__(self) -> None:
        self.tokens: tuple[int, ...] = ()
        self.loaded: list[list[dict]] = []
        self.load_threads: list[str] = []

    def load_prefix(self, messages: list) -> None:
        self.loaded.append(messages)
        self.load_threads.append(threading.current_thread().name)
        self.tokens = tuple(b"prefix loaded")

    def conversation_tokens(self) -> tuple[int, ...]:
        return self.tokens

    def ledger(self) -> tuple[()]:
        return ()

    def token_budget(self) -> tuple[int, int]:
        return len(self.tokens), 256

    def transcript_spans(self) -> tuple[str, tuple[TranscriptSpan, ...]]:
        return bytes(self.tokens).decode(), ()

    def open_user_text(self) -> str | None:
        return None


def _entry(
    index: int,
    *,
    turns: int = 1,
    snippet: str | None = None,
    skip_reason: str | None = None,
    long_first_answer: bool = False,
    system: bool = False,
) -> ConvoEntry:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": "system rule"})
    for turn in range(turns):
        messages.append({"role": "user", "content": f"question {index}.{turn}"})
        answer = "x" * 120 if long_first_answer and turn == 0 else f"answer {index}.{turn}"
        messages.append({"role": "assistant", "content": answer})
    return ConvoEntry(
        index=index,
        n_user_turns=0 if skip_reason is not None else turns,
        token_count=index + 9,
        first_user_snippet=snippet or f"question {index}.0",
        skip_reason=skip_reason,
        messages=() if skip_reason is not None else tuple(messages),
    )


def _entries(count: int = 45) -> list[ConvoEntry]:
    entries = [_entry(index) for index in range(count)]
    entries[0] = _entry(0, turns=2, snippet="first system conversation", system=True)
    entries[2] = _entry(2, skip_reason="no_assistant")
    entries[3] = _entry(3, turns=2, long_first_answer=True)
    entries[21] = _entry(21, turns=2)
    return entries


@pytest.fixture(autouse=True)
def _available_parquet_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    parquet_dir = tmp_path / "standard_sft_val"
    parquet_dir.mkdir()
    monkeypatch.setattr(
        "scripts.chat_tui.widgets.convo_picker.default_val_parquet_dir",
        lambda: parquet_dir,
    )
    return parquet_dir


def _app(
    catalog_loader: Callable[[], StubCatalog],
    *,
    sequence_len: int = 256,
    session: PrefixSession | None = None,
) -> ChatApp:
    return ChatApp(
        session=session or PrefixSession(),  # type: ignore[arg-type]
        tokenizer=MessageTokenizer(),
        checkpoint_label="tiny.ckpt",
        seed=11,
        catalog_loader=catalog_loader,  # type: ignore[arg-type]
        sequence_len=sequence_len,
        max_new_tokens_per_turn=32,
    )


async def _wait_for(pilot: Any, predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not reached")


async def _open(pilot: Any) -> None:
    await pilot.press(*"/convo", "enter")
    await pilot.pause()


def _picker(app: ChatApp) -> ConvoPicker:
    return app.query_one(ConvoPicker)


def _body(app: ChatApp) -> Text:
    rendered = app.query_one("#convo-body", Static).content
    assert isinstance(rendered, Text)
    return rendered


def _styles_at(text: Text, needle: str) -> str:
    position = text.plain.index(needle)
    return " ".join(str(span.style) for span in text._spans if span.start <= position < span.end)


@pytest.mark.anyio
async def test_lazy_catalog_and_page_load_render_fields_and_dim_skip_rows() -> None:
    release = threading.Event()
    loader_threads: list[str] = []
    catalog = StubCatalog(_entries())

    def load() -> StubCatalog:
        loader_threads.append(threading.current_thread().name)
        assert release.wait(timeout=2), "test did not release catalog load"
        return catalog

    app = _app(load)
    async with app.run_test() as pilot:
        await _open(pilot)
        assert "checking data…" in _body(app).plain

        release.set()
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        body = _body(app)
        assert "0 · 2 turns · 9 tokens · first system conversation" in body.plain
        assert "2 · 0 turns · 11 tokens · question 2.0 · no_assistant" in body.plain
        assert "dim" in _styles_at(body, "no_assistant")
        assert loader_threads == ["chat-model"]
        assert set(catalog.entry_threads) == {"chat-model"}

        await pilot.press("2", "enter", "enter")
        assert _picker(app).mode is PickerMode.LIST


@pytest.mark.anyio
async def test_digit_jump_paging_and_multi_turn_filter() -> None:
    catalog = StubCatalog(_entries())
    app = _app(lambda: catalog)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)

        await pilot.press("2", "1", "enter")
        await _wait_for(pilot, lambda: "question 21.0" in _body(app).plain)
        assert "20 · 1 turns" in _body(app).plain
        assert _picker(app).highlighted_index == 21

        await pilot.press("m")
        body = _body(app).plain
        assert "21 · 2 turns" in body
        assert "20 · 1 turns" not in body


@pytest.mark.anyio
async def test_turn_mode_includes_continue_row_and_dims_headroom_failures() -> None:
    catalog = StubCatalog(_entries())
    app = _app(lambda: catalog, sequence_len=100)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        await pilot.press("3", "enter", "enter")
        await _wait_for(pilot, lambda: _picker(app).mode is PickerMode.TURN)
        await _wait_for(pilot, lambda: "continue conversation" in _body(app).plain)

        body = _body(app)
        assert "0 · question 3.0" in body.plain
        assert "1 · question 3.1" in body.plain
        assert "2 · continue conversation (no replaced turn)" in body.plain
        assert "prefix uses" in body.plain
        assert "sequence_len is 100" in body.plain
        assert "dim" in _styles_at(body, "prefix uses")

        await pilot.press("down", "enter")
        assert _picker(app).mode is PickerMode.TURN


@pytest.mark.anyio
async def test_pick_verifies_then_loads_exact_role_position_prefix_and_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = StubCatalog(_entries())
    session = PrefixSession()
    verify_calls: list[tuple[StubCatalog, ConvoEntry, str]] = []

    def verify(
        seen_catalog: StubCatalog,
        entry: ConvoEntry,
        tokenizer: MessageTokenizer,
        parquet_dir: object,
    ) -> None:
        del tokenizer, parquet_dir
        verify_calls.append((seen_catalog, entry, threading.current_thread().name))

    monkeypatch.setattr("scripts.chat_tui.widgets.convo_picker.verify_against_parquet", verify)
    app = _app(lambda: catalog, session=session)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: "continue conversation" in _body(app).plain)
        await pilot.press("down", "enter")
        await _wait_for(pilot, lambda: session.loaded != [])
        await _wait_for(pilot, lambda: not app.busy)

        expected = prefix_messages(catalog.entries[0], 1)
        assert verify_calls == [(catalog, catalog.entries[0], "chat-model")]
        assert session.loaded == [expected]
        assert session.load_threads == ["chat-model"]
        assert len(app.query(ConvoPicker)) == 0
        assert app.transcript.plain == "prefix loaded"


@pytest.mark.anyio
async def test_probe_value_error_keeps_picker_open_and_notifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = StubCatalog(_entries())

    def fail(*args: object) -> None:
        del args
        raise ValueError("original_index=0: mismatch")

    monkeypatch.setattr("scripts.chat_tui.widgets.convo_picker.verify_against_parquet", fail)
    app = _app(lambda: catalog)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: "continue conversation" in _body(app).plain)
        await pilot.press("down", "enter")
        await _wait_for(
            pilot,
            lambda: (
                "original_index=0: mismatch"
                in [notification.message for notification in app._notifications]
            ),
        )
        assert _picker(app).mode is PickerMode.TURN
        assert session_loads(app) == []


def session_loads(app: ChatApp) -> list[list[dict]]:
    session = app._session
    assert isinstance(session, PrefixSession)
    return session.loaded


@pytest.mark.anyio
async def test_recall_places_original_user_turn_in_input() -> None:
    catalog = StubCatalog(_entries())
    app = _app(lambda: catalog)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: "continue conversation" in _body(app).plain)
        await pilot.press("down", "r")
        assert app.query_one("#chat-input", Input).value == "question 0.1"


@pytest.mark.anyio
async def test_escape_returns_to_list_then_closes_picker() -> None:
    catalog = StubCatalog(_entries())
    app = _app(lambda: catalog)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: "continue conversation" in _body(app).plain)

        await pilot.press("escape")
        assert _picker(app).mode is PickerMode.LIST
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(ConvoPicker)) == 0

        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        assert len(app.query(ConvoPicker)) == 1


@pytest.mark.anyio
async def test_missing_parquet_preflight_fails_loudly_and_only_escape_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing-standard-sft-val"
    loader_calls: list[None] = []
    monkeypatch.setattr(
        "scripts.chat_tui.widgets.convo_picker.default_val_parquet_dir",
        lambda: missing,
    )
    app = _app(lambda: loader_calls.append(None) or StubCatalog(_entries()))

    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: str(missing) in _body(app).plain)
        assert loader_calls == []
        assert str(app.query_one("#convo-help", Static).content) == "Esc close"
        await pilot.press("enter", "m", "down")
        assert str(missing) in _body(app).plain
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(ConvoPicker)) == 0


@pytest.mark.anyio
async def test_escape_abandons_in_flight_pick_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = StubCatalog(_entries())
    session = PrefixSession()
    probe_started = threading.Event()
    release_probe = threading.Event()
    probe_finished = threading.Event()

    def verify(*_args: object) -> None:
        probe_started.set()
        assert release_probe.wait(timeout=2), "test did not release parquet probe"
        probe_finished.set()

    monkeypatch.setattr("scripts.chat_tui.widgets.convo_picker.verify_against_parquet", verify)
    app = _app(lambda: catalog, session=session)
    async with app.run_test() as pilot:
        await _open(pilot)
        await _wait_for(pilot, lambda: "first system conversation" in _body(app).plain)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: "continue conversation" in _body(app).plain)
        await pilot.press("enter")
        await _wait_for(pilot, probe_started.is_set)
        await pilot.press("escape")
        assert _picker(app).mode is PickerMode.LIST
        release_probe.set()
        await _wait_for(pilot, probe_finished.is_set)
        await pilot.pause(0.05)

        assert session.loaded == []
        assert len(app.query(ConvoPicker)) == 1
