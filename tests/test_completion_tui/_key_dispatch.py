"""Test-owned adapter for exercising completion-TUI key event handlers."""

from __future__ import annotations

from typing import Any, Protocol, cast

from textual.events import Key
from textual.message import Message


class KeyHandler(Protocol):
    """Widget surface required to dispatch and observe a key event."""

    def on_key(self, event: Key) -> None: ...

    def post_message(self, message: Message) -> bool: ...


def dispatch_key(widget: KeyHandler, key: str) -> list[Message]:
    """Dispatch a real Textual key event and return messages posted by the widget."""
    messages: list[Message] = []
    original_post_message = widget.post_message

    def capture(message: Message) -> bool:
        messages.append(message)
        return True

    mutable_widget = cast(Any, widget)
    mutable_widget.post_message = capture
    try:
        character = key if len(key) == 1 else None
        widget.on_key(Key(key, character))
    finally:
        mutable_widget.post_message = original_post_message
    return messages
