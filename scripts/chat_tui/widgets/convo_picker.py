"""Lazy, model-thread-backed conversation picker."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

from rich.text import Text
from textual.app import ComposeResult
from textual.events import Key
from textual.widget import Widget
from textual.widgets import Static

from scripts.chat_tui.convo import (
    ConvoCatalog,
    ConvoEntry,
    default_val_parquet_dir,
    prefix_headroom_error,
    prefix_messages,
    verify_against_parquet,
)

PAGE_SIZE = 20
SNIPPET_LEN = 80


class _ModelThread(Protocol):
    def submit(self, job: Callable[[], None]) -> None: ...


class PickerMode(Enum):
    """The two panes hosted by one modal widget."""

    LIST = auto()
    TURN = auto()


class ConvoPicker(Widget):
    """Pick a validation conversation and the user turn to replace."""

    can_focus = True

    DEFAULT_CSS = """
    ConvoPicker {
        width: 92%;
        height: 85%;
        margin: 2 4;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    ConvoPicker #convo-title {
        height: 1;
        text-style: bold;
    }

    ConvoPicker #convo-body {
        height: 1fr;
        overflow-y: auto;
    }

    ConvoPicker #convo-original {
        height: auto;
        max-height: 4;
        color: $text-muted;
    }

    ConvoPicker #convo-help {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        catalog_loader: Callable[[], ConvoCatalog],
        tokenizer: Any,
        sequence_len: int,
        max_new_tokens_per_turn: int,
        on_pick: Callable[[list], None],
        model_thread: _ModelThread,
        on_recall: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._catalog_loader = catalog_loader
        self._tokenizer = tokenizer
        self._sequence_len = sequence_len
        self._max_new_tokens_per_turn = max_new_tokens_per_turn
        self._on_pick = on_pick
        self._model_thread = model_thread
        self._on_recall = on_recall

        self.mode = PickerMode.LIST
        self.highlighted_index = 0
        self._catalog: ConvoCatalog | None = None
        self._catalog_size = 0
        self._entries: dict[int, ConvoEntry] = {}
        self._page = 0
        self._page_request = 0
        self._loading_page = False
        self._multi_turn_only = False
        self._digits = ""

        self._selected_entry: ConvoEntry | None = None
        self._highlighted_turn = 0
        self._turn_errors: tuple[str | None, ...] | None = None
        self._turn_request = 0
        self._picking = False
        self._pick_seq = 0
        self._parquet_dir: Path | None = None
        self._fatal_error: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Conversations", id="convo-title", markup=False)
        yield Static(Text("checking data…"), id="convo-body", markup=False)
        yield Static(Text(), id="convo-original", markup=False)
        yield Static("Esc close · digits+Enter jump · m multi-turn", id="convo-help", markup=False)

    def on_mount(self) -> None:
        self.focus()
        self.call_after_refresh(self._start_catalog_load)

    def _start_catalog_load(self) -> None:
        app = self.app

        def job() -> None:
            parquet_dir = default_val_parquet_dir()
            if not parquet_dir.is_dir():
                detail = f"validation parquet directory is unavailable: {parquet_dir}"
                app.call_from_thread(self._catalog_loaded, None, 0, detail, None)
                return
            try:
                catalog = self._catalog_loader()
                size = len(catalog)
            except Exception as exc:
                app.call_from_thread(
                    self._catalog_loaded,
                    None,
                    0,
                    str(exc),
                    parquet_dir,
                )
                return
            app.call_from_thread(self._catalog_loaded, catalog, size, None, parquet_dir)

        self._model_thread.submit(job)

    def on_key(self, event: Key) -> None:
        key = event.key
        handled = False
        if self._fatal_error is not None:
            if key != "escape":
                return
            self.remove()
            handled = True
        elif key == "escape":
            self._escape()
            handled = True
        elif self.mode is PickerMode.LIST:
            handled = self._handle_list_key(key)
        else:
            handled = self._handle_turn_key(key)
        if handled:
            event.prevent_default()
            event.stop()

    def _handle_list_key(self, key: str) -> bool:
        if len(key) == 1 and key.isdigit():
            self._digits += key
            self._render_list()
            return True
        if key == "enter":
            if self._digits:
                self._jump_to_digits()
            else:
                self._enter_highlighted_conversation()
            return True
        if key == "up":
            self._move_list(-1)
            return True
        if key == "down":
            self._move_list(1)
            return True
        if key == "pageup":
            self._change_page(-1)
            return True
        if key == "pagedown":
            self._change_page(1)
            return True
        if key == "m":
            self._multi_turn_only = not self._multi_turn_only
            visible = self._visible_indices()
            if visible and self.highlighted_index not in visible:
                self.highlighted_index = visible[0]
            self._render_list()
            return True
        return False

    def _handle_turn_key(self, key: str) -> bool:
        if key == "enter":
            self._pick_highlighted_turn()
            return True
        if key == "up":
            self._move_turn(-1)
            return True
        if key == "down":
            self._move_turn(1)
            return True
        if key == "r":
            self._recall_highlighted_turn()
            return True
        return False

    def _catalog_loaded(
        self,
        catalog: ConvoCatalog | None,
        size: int,
        error: str | None,
        parquet_dir: Path | None,
    ) -> None:
        if not self.is_mounted:
            return
        if error is not None or catalog is None:
            detail = error or "conversation catalog did not load"
            self._fatal_error = detail
            self.query_one("#convo-title", Static).update("Conversations · unavailable")
            self.query_one("#convo-body", Static).update(Text(detail, style="bold red"))
            self.query_one("#convo-help", Static).update("Esc close")
            self.notify(detail, severity="error", timeout=8, markup=False)
            return
        if parquet_dir is None:
            raise RuntimeError("conversation parquet preflight did not return a directory")
        self._parquet_dir = parquet_dir
        self._catalog = catalog
        self._catalog_size = size
        if size == 0:
            self.query_one("#convo-body", Static).update(Text("no conversations"))
            return
        self._load_page(0, highlighted_index=0)

    def _load_page(self, page: int, *, highlighted_index: int | None = None) -> None:
        catalog = self._catalog
        if catalog is None or self._catalog_size == 0:
            return
        loaded_catalog = catalog
        last_page = (self._catalog_size - 1) // PAGE_SIZE
        page = min(max(page, 0), last_page)
        self._page = page
        if highlighted_index is not None:
            self.highlighted_index = highlighted_index
        start = page * PAGE_SIZE
        stop = min(start + PAGE_SIZE, self._catalog_size)
        missing = [index for index in range(start, stop) if index not in self._entries]
        if not missing:
            self._loading_page = False
            self._render_list()
            return

        self._loading_page = True
        self._page_request += 1
        request = self._page_request
        self.query_one("#convo-body", Static).update(Text("loading conversations…"))
        app = self.app

        def job() -> None:
            try:
                entries = [loaded_catalog.entry(index) for index in missing]
            except Exception as exc:
                app.call_from_thread(self._page_loaded, request, [], str(exc))
                return
            app.call_from_thread(self._page_loaded, request, entries, None)

        self._model_thread.submit(job)

    def _page_loaded(
        self,
        request: int,
        entries: list[ConvoEntry],
        error: str | None,
    ) -> None:
        if not self.is_mounted:
            return
        for entry in entries:
            self._entries[entry.index] = entry
        if request != self._page_request:
            return
        self._loading_page = False
        if error is not None:
            self.query_one("#convo-body", Static).update(Text(error, style="bold red"))
            self.notify(error, severity="error", timeout=8, markup=False)
            return
        self._render_list()

    def _render_list(self) -> None:
        if not self.is_mounted or self._catalog is None or self._loading_page:
            return
        title = f"Conversations · page {self._page + 1}"
        if self._multi_turn_only:
            title += " · multi-turn only"
        self.query_one("#convo-title", Static).update(title)
        rendered = Text()
        visible = self._visible_indices()
        for row, index in enumerate(visible):
            entry = self._entries[index]
            style = "dim" if entry.skip_reason is not None else ""
            if index == self.highlighted_index:
                style = f"{style} reverse".strip()
            prefix = "> " if index == self.highlighted_index else "  "
            line = (
                f"{prefix}{entry.index} · {entry.n_user_turns} turns · "
                f"{entry.token_count} tokens · {entry.first_user_snippet}"
            )
            if entry.skip_reason is not None:
                line += f" · {entry.skip_reason}"
            rendered.append(line, style=style or None)
            if row != len(visible) - 1:
                rendered.append("\n")
        if not visible:
            rendered.append("no conversations on this page match the filter", style="dim")
        if self._digits:
            rendered.append(f"\n\njump to index: {self._digits}", style="bold")
        self.query_one("#convo-body", Static).update(rendered)
        self.query_one("#convo-original", Static).update(Text())

    def _visible_indices(self) -> list[int]:
        start = self._page * PAGE_SIZE
        stop = min(start + PAGE_SIZE, self._catalog_size)
        result = []
        for index in range(start, stop):
            entry = self._entries.get(index)
            if entry is None:
                continue
            if self._multi_turn_only and not (
                entry.n_user_turns >= 2 and entry.skip_reason is None
            ):
                continue
            result.append(index)
        return result

    def _jump_to_digits(self) -> None:
        target_text = self._digits
        self._digits = ""
        target = int(target_text)
        if not 0 <= target < self._catalog_size:
            self.notify(
                f"conversation index {target} is outside 0..{self._catalog_size - 1}",
                severity="error",
                markup=False,
            )
            self._render_list()
            return
        self._load_page(target // PAGE_SIZE, highlighted_index=target)

    def _move_list(self, delta: int) -> None:
        visible = self._visible_indices()
        if not visible:
            return
        try:
            position = visible.index(self.highlighted_index)
        except ValueError:
            position = 0 if delta > 0 else len(visible) - 1
        new_position = position + delta
        if 0 <= new_position < len(visible):
            self.highlighted_index = visible[new_position]
            self._render_list()
            return
        self._change_page(1 if delta > 0 else -1)

    def _change_page(self, delta: int) -> None:
        if self._catalog_size == 0:
            return
        last_page = (self._catalog_size - 1) // PAGE_SIZE
        target_page = min(max(self._page + delta, 0), last_page)
        if target_page == self._page:
            return
        if delta > 0:
            highlighted = target_page * PAGE_SIZE
        else:
            highlighted = min((target_page + 1) * PAGE_SIZE, self._catalog_size) - 1
        self._load_page(target_page, highlighted_index=highlighted)

    def _enter_highlighted_conversation(self) -> None:
        if self._loading_page:
            return
        entry = self._entries.get(self.highlighted_index)
        if entry is None or entry.skip_reason is not None:
            return
        selected_entry = entry
        self.mode = PickerMode.TURN
        self._selected_entry = entry
        self._highlighted_turn = 0
        self._turn_errors = None
        self._turn_request += 1
        request = self._turn_request
        self.query_one("#convo-title", Static).update(f"Conversation {entry.index} · choose turn")
        self.query_one("#convo-body", Static).update(Text("loading turns…"))
        self.query_one("#convo-help", Static).update("Esc back · Enter select · r recall")
        app = self.app

        def job() -> None:
            try:
                errors = tuple(
                    prefix_headroom_error(
                        selected_entry,
                        turn,
                        self._tokenizer,
                        sequence_len=self._sequence_len,
                        max_new_tokens_per_turn=self._max_new_tokens_per_turn,
                    )
                    for turn in range(selected_entry.n_user_turns + 1)
                )
            except Exception as exc:
                app.call_from_thread(self._turns_loaded, request, None, str(exc))
                return
            app.call_from_thread(self._turns_loaded, request, errors, None)

        self._model_thread.submit(job)

    def _turns_loaded(
        self,
        request: int,
        errors: tuple[str | None, ...] | None,
        error: str | None,
    ) -> None:
        if not self.is_mounted or request != self._turn_request or self.mode is not PickerMode.TURN:
            return
        if error is not None or errors is None:
            detail = error or "conversation turns did not load"
            self.query_one("#convo-body", Static).update(Text(detail, style="bold red"))
            self.notify(detail, severity="error", timeout=8, markup=False)
            return
        self._turn_errors = errors
        self._render_turns()

    def _render_turns(self) -> None:
        entry = self._selected_entry
        errors = self._turn_errors
        if not self.is_mounted or entry is None or errors is None:
            return
        rendered = Text()
        user_messages = [message for message in entry.messages if message.get("role") == "user"]
        for turn in range(entry.n_user_turns + 1):
            if turn == entry.n_user_turns:
                label = "continue conversation (no replaced turn)"
            else:
                content = str(user_messages[turn].get("content", ""))
                label = self._snippet(content)
            reason = errors[turn]
            style = "dim" if reason is not None else ""
            if turn == self._highlighted_turn:
                style = f"{style} reverse".strip()
            prefix = "> " if turn == self._highlighted_turn else "  "
            line = f"{prefix}{turn} · {label}"
            if reason is not None:
                line += f" · {reason}"
            rendered.append(line, style=style or None)
            if turn != entry.n_user_turns:
                rendered.append("\n")
        if self._picking:
            rendered.append("\n\nverifying conversation…", style="bold")
        self.query_one("#convo-body", Static).update(rendered)
        self._render_original()

    def _render_original(self) -> None:
        entry = self._selected_entry
        if entry is None or self._highlighted_turn >= entry.n_user_turns:
            self.query_one("#convo-original", Static).update(Text())
            return
        user_messages = [message for message in entry.messages if message.get("role") == "user"]
        original = str(user_messages[self._highlighted_turn].get("content", ""))
        self.query_one("#convo-original", Static).update(
            Text(f"original turn: {original}", style="dim")
        )

    def _move_turn(self, delta: int) -> None:
        entry = self._selected_entry
        if entry is None:
            return
        self._highlighted_turn = min(max(self._highlighted_turn + delta, 0), entry.n_user_turns)
        self._render_turns()

    def _pick_highlighted_turn(self) -> None:
        entry = self._selected_entry
        errors = self._turn_errors
        if entry is None or errors is None or self._picking:
            return
        if errors[self._highlighted_turn] is not None:
            return
        turn = self._highlighted_turn
        catalog = self._catalog
        if catalog is None:
            return
        parquet_dir = self._parquet_dir
        if parquet_dir is None:
            raise RuntimeError("conversation parquet preflight has not completed")
        self._picking = True
        self._pick_seq += 1
        request = self._pick_seq
        self._render_turns()
        app = self.app

        def job() -> None:
            try:
                prefix = prefix_messages(entry, turn)
                verify_against_parquet(
                    catalog,
                    entry,
                    self._tokenizer,
                    parquet_dir,
                )
            except Exception as exc:
                app.call_from_thread(
                    self._pick_finished,
                    request,
                    entry,
                    turn,
                    None,
                    str(exc),
                )
                return
            app.call_from_thread(
                self._pick_finished,
                request,
                entry,
                turn,
                prefix,
                None,
            )

        self._model_thread.submit(job)

    def _pick_finished(
        self,
        request: int,
        entry: ConvoEntry,
        turn: int,
        prefix: list | None,
        error: str | None,
    ) -> None:
        if (
            not self.is_mounted
            or request != self._pick_seq
            or self.mode is not PickerMode.TURN
            or (self._selected_entry, self._highlighted_turn) != (entry, turn)
        ):
            return
        self._picking = False
        if error is not None or prefix is None:
            detail = error or "conversation prefix did not load"
            self.notify(detail, severity="error", timeout=8, markup=False)
            self._render_turns()
            return
        try:
            self._on_pick(prefix)
        except Exception as exc:
            self.notify(str(exc), severity="error", timeout=8, markup=False)
            self._render_turns()
            return
        self.remove()

    def _recall_highlighted_turn(self) -> None:
        entry = self._selected_entry
        if entry is None or self._on_recall is None or self._highlighted_turn >= entry.n_user_turns:
            return
        user_messages = [message for message in entry.messages if message.get("role") == "user"]
        self._on_recall(str(user_messages[self._highlighted_turn].get("content", "")))

    def _escape(self) -> None:
        if self.mode is PickerMode.TURN:
            self._pick_seq += 1
            self.mode = PickerMode.LIST
            self._selected_entry = None
            self._turn_errors = None
            self._picking = False
            self.query_one("#convo-help", Static).update(
                "Esc close · digits+Enter jump · m multi-turn"
            )
            self._render_list()
            return
        self.remove()

    @staticmethod
    def _snippet(text: str) -> str:
        return " ".join(text.split())[:SNIPPET_LEN]
