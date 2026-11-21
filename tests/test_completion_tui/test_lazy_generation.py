"""Tests for LazyGenerationManager."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from scripts.completion_tui.generation import (
    GenerationHandle,
    GenerationManager,
    LazyGenerationManager,
)
from scripts.completion_tui.state import SessionState


def _make_state() -> SessionState:
    return SessionState(
        document_text="hello",
        segments=[],
        num_buckets=11,
        has_negatives=False,
    )


class TestLazyGenerationManager:
    def test_factory_not_called_on_init(self):
        factory = MagicMock(return_value=MagicMock(spec=GenerationManager))
        LazyGenerationManager(factory)
        factory.assert_not_called()

    def test_factory_called_on_first_start_generation(self):
        inner = MagicMock(spec=GenerationManager)
        handle = MagicMock(spec=GenerationHandle)
        inner.start_generation.return_value = handle
        factory = MagicMock(return_value=inner)

        mgr = LazyGenerationManager(factory)
        result = mgr.start_generation(_make_state())

        factory.assert_called_once()
        inner.start_generation.assert_called_once()
        assert result is handle

    def test_factory_called_only_once(self):
        inner = MagicMock(spec=GenerationManager)
        inner.start_generation.return_value = MagicMock(spec=GenerationHandle)
        factory = MagicMock(return_value=inner)

        mgr = LazyGenerationManager(factory)
        mgr.start_generation(_make_state())
        mgr.start_generation(_make_state())

        factory.assert_called_once()
        assert inner.start_generation.call_count == 2

    def test_cancel_delegates_after_load(self):
        inner = MagicMock(spec=GenerationManager)
        inner.start_generation.return_value = MagicMock(spec=GenerationHandle)
        factory = MagicMock(return_value=inner)

        mgr = LazyGenerationManager(factory)
        mgr.start_generation(_make_state())

        handle = MagicMock(spec=GenerationHandle)
        mgr.cancel(handle)
        inner.cancel.assert_called_once_with(handle)

    def test_cancel_sets_event_before_load(self):
        factory = MagicMock(return_value=MagicMock(spec=GenerationManager))
        mgr = LazyGenerationManager(factory)

        cancel_event = threading.Event()
        handle = MagicMock(spec=GenerationHandle)
        handle.cancel_event = cancel_event

        mgr.cancel(handle)
        assert cancel_event.is_set()
        factory.assert_not_called()
