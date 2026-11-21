"""Seed input widget for the completion TUI.

A small focusable widget that displays and edits the generation seed.
Entered via Opt+Right from SegmentSelector, exited via Opt+Left or Escape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widget import Widget

_MAX_DIGITS = 6


class SeedInput(Widget):
    """Focusable seed editor. Displays 'Seed: [N]' and accepts digit input."""

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class SeedInputExited(Message):
        """Posted when the user leaves the seed input (Opt+Left or Escape)."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    can_focus = True

    DEFAULT_CSS = """
    SeedInput {
        width: auto;
        height: 1;
        min-width: 14;
    }
    """

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._seed: int = seed
        self._first_digit: bool = True  # first digit overwrites

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value
        self._first_digit = True

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_digit(self, digit: int) -> None:
        """Process a digit keypress. First digit overwrites, subsequent append."""
        if self._first_digit:
            self._seed = digit
            self._first_digit = False
        elif len(str(self._seed)) < _MAX_DIGITS:
            self._seed = self._seed * 10 + digit

    def handle_backspace(self) -> None:
        """Remove the last digit. If only one digit remains, set seed to 0."""
        s = str(self._seed)
        if len(s) <= 1:
            self._seed = 0
        else:
            self._seed = int(s[:-1])
        self._first_digit = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Text:
        result = Text()
        label = f"Seed: [{self._seed}]"
        if self.has_focus:
            result.append(label, style="bold")
        else:
            result.append(label, style="dim")
        return result

    def reset_input(self) -> None:
        """Reset so the next digit overwrites (called on focus)."""
        self._first_digit = True

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        key = event.key

        # Digit input
        if len(key) == 1 and key.isdigit():
            self.handle_digit(int(key))
            self.refresh()
            event.prevent_default()
            event.stop()
            return

        handler = self._KEY_MAP.get(key)
        if handler is not None:
            handler(self)
            self.refresh()
            event.prevent_default()
            event.stop()

    def _key_escape(self) -> None:
        self.post_message(self.SeedInputExited())

    def _key_exit(self) -> None:
        self.post_message(self.SeedInputExited())

    def _key_backspace(self) -> None:
        self.handle_backspace()

    _KEY_MAP: dict[str, Callable[[SeedInput], None]] = {
        "escape": _key_escape,
        "enter": _key_exit,
        "ctrl+left": _key_exit,  # Opt+Left on macOS Ghostty
        "backspace": _key_backspace,
    }
