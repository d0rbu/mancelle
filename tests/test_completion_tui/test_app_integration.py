"""Integration tests for CompletionApp.

Tests real widget composition and message flow with mocked model/data services.
Uses synchronous helpers to avoid needing pytest-asyncio.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.completion_tui.app import CompletionApp, DocumentLoaded, FocusPanel
from scripts.completion_tui.generation import GenerationHandle, GenerationSnapshot
from scripts.completion_tui.highlight_types import (
    FeatureHighlightRequest,
    FeatureHighlightResult,
    FeatureHighlightSpan,
)
from scripts.completion_tui.state import (
    DocumentData,
    FeatureCondition,
    Segment,
    SessionState,
)
from scripts.completion_tui.widgets.doc_selector import DocSelector
from scripts.completion_tui.widgets.feature_detail import FeatureDetail
from scripts.completion_tui.widgets.feature_editor import FeatureEditor
from scripts.completion_tui.widgets.feature_search import FeatureSearch
from scripts.completion_tui.widgets.seed_input import SeedInput
from scripts.completion_tui.widgets.segment_selector import SegmentSelector
from scripts.completion_tui.widgets.text_panel import TextPanel
from tests.test_completion_tui.conftest import WordMockTokenizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_segments(text: str = "hello world") -> list[Segment]:
    """Two segments splitting 'hello world' at the space."""
    return [
        Segment(
            segment_idx=0,
            char_start=0,
            char_end=5,
            color_idx=0,
            features=[FeatureCondition(feature_id=100, magnitude=3)],
            id="seg0aaaa",
            enabled=True,
        ),
        Segment(
            segment_idx=1,
            char_start=5,
            char_end=11,
            color_idx=1,
            features=[FeatureCondition(feature_id=200, magnitude=5)],
            id="seg1bbbb",
            enabled=True,
            is_generation_target=True,
        ),
    ]


def _make_state() -> SessionState:
    return SessionState(
        document_text="hello world",
        segments=_make_segments(),
        num_buckets=11,
        has_negatives=False,
    )


def _make_doc_loader() -> MagicMock:
    """Mock DocumentLoader that returns a predefined document."""
    loader = MagicMock()
    loader.load_document.return_value = DocumentData(
        doc_idx=42,
        text="quick brown fox",
        segments=[
            Segment(
                segment_idx=0,
                char_start=0,
                char_end=6,
                color_idx=0,
                features=[FeatureCondition(feature_id=10, magnitude=2)],
                id="docsel00",
            ),
            Segment(
                segment_idx=1,
                char_start=6,
                char_end=15,
                color_idx=1,
                features=[],
                id="docsel01",
                is_generation_target=True,
                target_token_count=64,
            ),
        ],
        total_segments=5,
    )
    return loader


def _make_multi_doc_loader() -> MagicMock:
    """Mock DocumentLoader returning different docs per doc_idx."""
    import copy as _copy

    docs: dict[int, DocumentData] = {
        10: DocumentData(
            doc_idx=10,
            text="doc ten text",
            segments=[
                Segment(
                    segment_idx=0, char_start=0, char_end=12, color_idx=0, features=[], id="d10s0"
                ),
            ],
            total_segments=3,
        ),
        20: DocumentData(
            doc_idx=20,
            text="doc twenty text here",
            segments=[
                Segment(
                    segment_idx=0, char_start=0, char_end=10, color_idx=0, features=[], id="d20s0"
                ),
                Segment(
                    segment_idx=1,
                    char_start=10,
                    char_end=20,
                    color_idx=1,
                    features=[],
                    id="d20s1",
                    is_generation_target=True,
                    target_token_count=64,
                ),
            ],
            total_segments=4,
        ),
        30: DocumentData(
            doc_idx=30,
            text="AAAABBBBCCCCDDDD",
            segments=[
                Segment(
                    segment_idx=0, char_start=0, char_end=4, color_idx=0, features=[], id="d30s0"
                ),
                Segment(
                    segment_idx=1, char_start=4, char_end=8, color_idx=1, features=[], id="d30s1"
                ),
                Segment(
                    segment_idx=2, char_start=8, char_end=12, color_idx=2, features=[], id="d30s2"
                ),
                Segment(
                    segment_idx=3,
                    char_start=12,
                    char_end=16,
                    color_idx=3,
                    features=[],
                    id="d30s3",
                    is_generation_target=True,
                    target_token_count=4,
                ),
            ],
            total_segments=4,
        ),
        42: DocumentData(
            doc_idx=42,
            text="quick brown fox",
            segments=[
                Segment(
                    segment_idx=0, char_start=0, char_end=6, color_idx=0, features=[], id="d42s0"
                ),
                Segment(
                    segment_idx=1,
                    char_start=6,
                    char_end=15,
                    color_idx=1,
                    features=[],
                    id="d42s1",
                    is_generation_target=True,
                    target_token_count=64,
                ),
            ],
            total_segments=5,
        ),
    }
    loader = MagicMock()

    def _load(doc_idx: int, num_segments: int | None = None) -> DocumentData | None:
        doc = docs.get(doc_idx)
        if doc is None:
            return None
        result = _copy.deepcopy(doc)
        if num_segments is not None and num_segments < len(result.segments):
            result.segments = result.segments[:num_segments]
        return result

    loader.load_document.side_effect = _load
    return loader


def _make_app(**overrides: Any) -> CompletionApp:
    """Build a CompletionApp with test defaults."""
    tok = WordMockTokenizer()
    state = _make_state()
    defaults: dict[str, Any] = {
        "state": state,
        "tokenizer": tok,
        "available_doc_ids": [10, 20, 30, 42],
        "total_segments_map": {10: 3, 20: 4, 30: 4, 42: 5},
    }
    defaults.update(overrides)
    return CompletionApp(**defaults)


class FakeLiveHighlightService:
    """Fast deterministic live-highlight service for app lifecycle tests."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls: list[FeatureHighlightRequest] = []
        self.started = threading.Event()

    def compute(self, request: FeatureHighlightRequest) -> FeatureHighlightResult:
        self.calls.append(request)
        self.started.set()
        if self.delay_s:
            time.sleep(self.delay_s)
        segment = request.segments[request.target_segment_index]
        return FeatureHighlightResult(
            request_key=request.request_key,
            segment_idx=segment.segment_idx,
            feature_id=request.feature_id,
            status="ready",
            spans=(
                FeatureHighlightSpan(
                    char_start=segment.char_start,
                    char_end=min(segment.char_end, segment.char_start + 1),
                    level=3,
                    percentile=99.5,
                    activation=10.0,
                ),
            ),
            max_percentile=99.5,
            max_activation=10.0,
            n_tokens=1,
        )


# ---------------------------------------------------------------------------
# 1. App mounts all widgets without error
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_app_mounts_all_widgets():
    app = _make_app()
    async with app.run_test():
        assert app.query_one("#doc-selector", DocSelector)
        assert app.query_one("#text-panel", TextPanel)
        assert app.query_one("#segment-selector", SegmentSelector)
        assert app.query_one("#seed-input", SeedInput)
        assert app.query_one("#segment-bar")
        # Initial focus should be TextPanel
        assert app._focus_panel == FocusPanel.TEXT_PANEL


# ---------------------------------------------------------------------------
# 2. Focus cycle: Ctrl+Down wraps correctly
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_focus_cycle_down():
    app = _make_app()
    async with app.run_test() as pilot:
        # Start at TextPanel
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # TextPanel -> SegmentSelector
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR

        # SegmentSelector -> DocSelector (wrap)
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR

        # DocSelector -> TextPanel
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.TEXT_PANEL


# ---------------------------------------------------------------------------
# 3. Focus cycle: Ctrl+Up reverse direction wraps correctly
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_focus_cycle_up():
    app = _make_app()
    async with app.run_test() as pilot:
        # Start at TextPanel
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # TextPanel -> DocSelector
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR

        # DocSelector -> SegmentSelector
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR

        # SegmentSelector -> TextPanel (wrap)
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.TEXT_PANEL


# ---------------------------------------------------------------------------
# 4. Escape from DocSelector and SegSelector → TextPanel
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_escape_returns_to_text_panel():
    app = _make_app()
    async with app.run_test() as pilot:
        # Navigate to DocSelector (TextPanel -> Seg -> Doc), then Escape
        await pilot.press("alt+down", "alt+down")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR
        await pilot.press("escape")
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # Navigate to SegmentSelector, then Escape
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR
        await pilot.press("escape")
        assert app._focus_panel == FocusPanel.TEXT_PANEL


# ---------------------------------------------------------------------------
# 5. Overlay replaces SegSelector in cycle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_overlay_replaces_seg_selector_in_cycle():
    app = _make_app()
    async with app.run_test() as pilot:
        # Go to SegmentSelector and open feature editor
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR
        # Trigger OpenFeatureEditor message
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()

        assert app._overlay_open
        assert app._focus_panel == FocusPanel.OVERLAY

        # Ctrl+Up from overlay -> TextPanel
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # Ctrl+Down from TextPanel -> back to overlay
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.OVERLAY


# ---------------------------------------------------------------------------
# 6. Deep overlay navigation: FeatureSearch open, cycle works
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deep_overlay_search_cycle():
    app = _make_app()
    async with app.run_test() as pilot:
        # Open feature editor
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()
        assert app._overlay_sub_panel == "editor"

        # Open search from editor
        editor = app.query_one("#feature-editor", FeatureEditor)
        editor.post_message(FeatureEditor.OpenFeatureSearch())
        await pilot.pause()
        assert app._overlay_sub_panel == "search"

        # Ctrl+Up from search -> TextPanel
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # Ctrl+Down from TextPanel -> back to overlay
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.OVERLAY


# ---------------------------------------------------------------------------
# 7. Escape chain: FeatureSearch → FeatureEditor → SegSelector → TextPanel
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_escape_chain():
    app = _make_app()
    async with app.run_test() as pilot:
        # Open feature editor
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()
        assert app._overlay_sub_panel == "editor"

        # Open search
        editor = app.query_one("#feature-editor", FeatureEditor)
        editor.post_message(FeatureEditor.OpenFeatureSearch())
        await pilot.pause()
        assert app._overlay_sub_panel == "search"

        # Escape from search -> editor
        search = app.query_one("#feature-search", FeatureSearch)
        search.post_message(FeatureSearch.FeatureSearchClosed())
        await pilot.pause()
        assert app._overlay_sub_panel == "editor"
        assert app._overlay_open

        # Escape from editor -> close overlay, focus SegSelector
        editor = app.query_one("#feature-editor", FeatureEditor)
        editor.post_message(FeatureEditor.FeatureEditorClosed())
        await pilot.pause()
        assert not app._overlay_open
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR

        # Escape from SegSelector -> TextPanel
        await pilot.press("escape")
        assert app._focus_panel == FocusPanel.TEXT_PANEL


# ---------------------------------------------------------------------------
# 8. Ctrl+I toggles conditioning: all segments disabled, then restored
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_toggle_conditioning():
    app = _make_app()
    async with app.run_test() as pilot:
        state = app._state
        assert state.global_conditioning is True
        assert all(s.enabled for s in state.segments)

        # Toggle off
        await pilot.press("ctrl+i")
        assert state.global_conditioning is False
        assert all(not s.enabled for s in state.segments)
        # Memory should store original states
        assert len(state.conditioning_memory) == 2

        # Toggle back on
        await pilot.press("ctrl+i")
        assert state.global_conditioning is True
        assert all(s.enabled for s in state.segments)
        assert len(state.conditioning_memory) == 0


# ---------------------------------------------------------------------------
# 9. DocumentSelected -> state populated, TextPanel shows text, overlay closes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_document_selected_populates_state():
    loader = _make_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # First open an overlay so we can verify it closes
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()
        assert app._overlay_open

        # Now trigger document selection
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(42, 2))
        await pilot.pause()

        # Overlay should be closed
        assert not app._overlay_open

        # State should be updated (text truncated — last segment is gen target)
        state = app._state
        assert state.document_text == "quick "
        assert len(state.segments) == 2
        assert state.segments[-1].is_generation_target
        assert state.segments[-1].char_start == state.segments[-1].char_end
        assert state.cursor_position == 0

        # Focus should be TextPanel
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        loader.load_document.assert_called_once_with(42, 2)


# ---------------------------------------------------------------------------
# 10. Full flow: load doc → select segment → edit feature → generate → undo
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_full_flow():
    """Load doc, select segment, edit feature magnitude, generate, undo."""
    tok = WordMockTokenizer()
    loader = _make_doc_loader()

    # Mock generation manager
    gen_mgr = MagicMock()
    gen_tokens = [tok.encode("x")[0]]  # single token "x"

    def fake_start_generation(state):
        snapshot = GenerationSnapshot(
            document_text=state.document_text,
            segments=state.segments,
            generated_text="",
            generated_token_ids=[],
            seed=state.seed,
        )
        return GenerationHandle(
            cancel_event=threading.Event(),
            token_iterator=iter(gen_tokens),
            snapshot=snapshot,
            target_segment_idx=1,
        )

    gen_mgr.start_generation.side_effect = fake_start_generation

    app = _make_app(
        doc_loader=loader,
        generation_manager=gen_mgr,
    )

    async with app.run_test() as pilot:
        state = app._state

        # 1. Load document (text truncated — last segment is gen target)
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(42, 2))
        await pilot.pause()
        assert state.document_text == "quick "
        pre_gen_text = state.document_text

        # 2. Select segment 1 (go to SegmentSelector, navigate right)
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR
        ss = app.query_one("#segment-selector", SegmentSelector)

        # 3. Open feature editor
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()
        assert app._overlay_open

        # 4. Edit magnitude (the editor has features from segment 0)
        editor = app.query_one("#feature-editor", FeatureEditor)
        assert len(editor.sorted_features) > 0
        old_mag = editor.sorted_features[0].magnitude
        editor.adjust_magnitude(1)
        assert editor.sorted_features[0].magnitude != old_mag

        # 5. Close editor (applies edits)
        editor.post_message(FeatureEditor.FeatureEditorClosed())
        await pilot.pause()
        assert not app._overlay_open
        # Edits should be applied to state
        assert state.segments[0].features[0].magnitude != old_mag

        # 6. Go back to TextPanel, trigger generation
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        tp = app.query_one("#text-panel", TextPanel)
        state.generation_redo_stack.append(
            GenerationSnapshot(
                document_text=state.document_text,
                segments=state.segments,
                generated_text="stale redo",
                generated_token_ids=[0],
                seed=state.seed,
            )
        )
        tp.post_message(TextPanel.GenerateRequested())
        await pilot.pause()
        # Give worker time to complete
        await pilot.pause()
        await pilot.pause()

        gen_mgr.start_generation.assert_called_once()

        # After generation completes, text should have the generated token
        # (the worker runs in a thread, may need additional pauses)
        for _ in range(10):
            await pilot.pause()
            if not tp.is_generating:
                break

        assert len(state.generation_history) >= 1
        assert state.generation_redo_stack == []

        # 7. Undo generation
        tp.post_message(TextPanel.UndoGeneration())
        await pilot.pause()

        # Should be back to pre-generation text
        assert state.document_text == pre_gen_text


# ---------------------------------------------------------------------------
# 11. FeatureDetail overlay: open, escape back to editor
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_feature_detail_overlay():
    app = _make_app()
    async with app.run_test() as pilot:
        # Open feature editor
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()
        assert app._overlay_sub_panel == "editor"

        # Open detail from editor
        editor = app.query_one("#feature-editor", FeatureEditor)
        fid = editor.sorted_features[0].feature_id
        editor.post_message(FeatureEditor.OpenFeatureDetail(fid))
        await pilot.pause()
        assert app._overlay_sub_panel == "detail"
        assert app._overlay_open

        # Ctrl+Up from detail -> TextPanel
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # Ctrl+Down -> back to overlay
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.OVERLAY

        # Escape from detail -> editor
        detail = app.query_one("#feature-detail", FeatureDetail)
        detail.post_message(FeatureDetail.FeatureDetailClosed())
        await pilot.pause()
        assert app._overlay_sub_panel == "editor"
        assert app._overlay_open


@pytest.mark.anyio
async def test_feature_detail_requests_live_highlight_for_open_segment():
    service = FakeLiveHighlightService()
    app = _make_app(live_highlight_service=service)
    async with app.run_test() as pilot:
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()

        editor = app.query_one("#feature-editor", FeatureEditor)
        fid = editor.sorted_features[0].feature_id
        editor.post_message(FeatureEditor.OpenFeatureDetail(fid))

        for _ in range(20):
            await pilot.pause()
            if (
                service.calls
                and app.query_one("#text-panel", TextPanel).feature_highlight_result is not None
            ):
                break

        assert service.calls
        request = service.calls[0]
        assert request.target_segment_index == 0
        assert request.feature_id == fid
        tp = app.query_one("#text-panel", TextPanel)
        assert tp.feature_highlight_result is not None
        assert tp.feature_highlight_result.segment_idx == 0
        assert "Live highlight" in app.query_one("#feature-detail", FeatureDetail).render()


@pytest.mark.anyio
async def test_feature_detail_close_clears_live_highlight():
    service = FakeLiveHighlightService()
    app = _make_app(live_highlight_service=service)
    async with app.run_test() as pilot:
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()

        editor = app.query_one("#feature-editor", FeatureEditor)
        editor.post_message(FeatureEditor.OpenFeatureDetail(editor.sorted_features[0].feature_id))
        for _ in range(20):
            await pilot.pause()
            if app.query_one("#text-panel", TextPanel).feature_highlight_result is not None:
                break

        detail = app.query_one("#feature-detail", FeatureDetail)
        detail.post_message(FeatureDetail.FeatureDetailClosed())
        await pilot.pause()

        assert app.query_one("#text-panel", TextPanel).feature_highlight_result is None


@pytest.mark.anyio
async def test_stale_live_highlight_result_is_ignored_after_text_change():
    service = FakeLiveHighlightService(delay_s=0.05)
    app = _make_app(live_highlight_service=service)
    async with app.run_test() as pilot:
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()

        editor = app.query_one("#feature-editor", FeatureEditor)
        editor.post_message(FeatureEditor.OpenFeatureDetail(editor.sorted_features[0].feature_id))
        await pilot.pause()
        assert service.started.wait(timeout=1.0)

        # Mutate state and notify the app while the worker is still computing.
        app._state.document_text = "X" + app._state.document_text
        app.on_text_panel_text_changed(TextPanel.TextChanged())

        for _ in range(20):
            await pilot.pause()

        assert app.query_one("#text-panel", TextPanel).feature_highlight_result is None


# ---------------------------------------------------------------------------
# 12. FeatureAdded adds feature to editor and selects it
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_feature_added_updates_editor():
    app = _make_app()
    async with app.run_test() as pilot:
        # Open feature editor for segment 0
        await pilot.press("alt+down")
        ss = app.query_one("#segment-selector", SegmentSelector)
        ss.post_message(SegmentSelector.OpenFeatureEditor(0))
        await pilot.pause()

        editor = app.query_one("#feature-editor", FeatureEditor)
        initial_count = len(editor.sorted_features)

        # Open search
        editor.post_message(FeatureEditor.OpenFeatureSearch())
        await pilot.pause()

        # Add a feature
        search = app.query_one("#feature-search", FeatureSearch)
        search.post_message(FeatureSearch.FeatureAdded(999))
        await pilot.pause()

        # Editor should have one more feature
        assert len(editor.sorted_features) == initial_count + 1
        # The new feature should be selected
        assert editor.sorted_features[editor.selected_idx].feature_id == 999
        # Magnitude should be 0 (default for added features)
        assert editor.sorted_features[editor.selected_idx].magnitude == 0


# ---------------------------------------------------------------------------
# 13. Doc bar preview: navigating updates text panel
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_on_navigation():
    """Navigating right in the doc bar should preview the hovered doc."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        original_text = app._state.document_text
        assert original_text == "hello world"

        # Go to doc bar (TextPanel -> DocSelector)
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR

        # Initial doc is 10 (first in sorted list), same as loaded_doc_idx.
        # Navigate right to doc 20 — should trigger a preview.
        await pilot.press("right")
        await pilot.pause()

        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.selected_doc_idx == 20
        # Text truncated to first segment; last segment is gen target
        assert app._state.document_text == "doc twenty"


# ---------------------------------------------------------------------------
# 14. Doc bar preview: escape restores committed state
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_restore_on_escape():
    """Escaping from doc bar should restore the committed document."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        original_text = app._state.document_text

        # Go to doc bar, navigate to preview a different doc
        await pilot.press("alt+up")
        await pilot.press("right")
        await pilot.pause()
        assert app._state.document_text != original_text

        # Escape — should restore committed state
        await pilot.press("escape")
        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert app._state.document_text == original_text


# ---------------------------------------------------------------------------
# 15. Doc bar preview: alt+down restores committed state
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_restore_on_cycle_down():
    """Cycling focus down from doc bar should restore committed state."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        original_text = app._state.document_text

        await pilot.press("alt+up")
        await pilot.press("right")
        await pilot.pause()
        assert app._state.document_text != original_text

        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert app._state.document_text == original_text


# ---------------------------------------------------------------------------
# 16. Doc bar preview: Enter commits new doc (no restore)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_confirm_keeps_new_doc():
    """Pressing Enter in doc bar should commit the previewed doc."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        await pilot.press("alt+up")
        await pilot.press("right")
        await pilot.pause()

        ds = app.query_one("#doc-selector", DocSelector)
        previewed_doc = ds.selected_doc_idx
        assert previewed_doc == 20

        # Enter confirms the doc
        await pilot.press("enter")
        await pilot.pause()

        # Should now be in text panel with the confirmed doc (truncated)
        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert app._state.document_text == "doc twenty"
        assert ds.loaded_doc_idx == 20


# ---------------------------------------------------------------------------
# 17. Doc bar preview: re-entering doc bar previews again
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_on_reentry():
    """Re-entering doc bar snaps to loaded doc and shows hybrid view."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        original_text = app._state.document_text

        # Enter doc bar, navigate right (to doc 20), then escape
        await pilot.press("alt+up")
        await pilot.press("right")
        await pilot.pause()
        assert app._state.document_text == "doc twenty"

        await pilot.press("escape")
        assert app._state.document_text == original_text

        # Re-enter doc bar — position preserved (stays on doc 20, not snapped back)
        await pilot.press("alt+up")
        await pilot.pause()
        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.selected_doc_idx == 20  # preserved, not snapped to loaded_doc_idx


# ---------------------------------------------------------------------------
# 18. Doc bar preview: navigating back to loaded doc shows fresh with bright
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_navigate_back_shows_bright():
    """Navigating back to the loaded doc shows fresh text with bright segments."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        await pilot.press("alt+up")
        # Navigate right to doc 20
        await pilot.press("right")
        await pilot.pause()
        assert app._state.document_text == "doc twenty"

        tp = app.query_one("#text-panel", TextPanel)
        # Non-loaded doc — no bright segments
        assert tp.bright_segment_indices is None

        # Navigate back left to doc 10 (loaded) — should preview fresh + bright
        await pilot.press("left")
        await pilot.pause()

        # Doc 10 has 1 segment; after truncation text is empty
        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.selected_doc_idx == 10
        assert ds.loaded_doc_idx == 10
        assert tp.bright_segment_indices is not None


# ---------------------------------------------------------------------------
# 19. Doc bar preview: no preview without doc_loader
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stale_document_result_cannot_replace_newer_selection():
    """A cancelled worker that finishes late must not mutate the preview."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test():
        app._document_load_seq = 2
        stale = loader.load_document(20, 2)
        assert stale is not None
        app.on_document_loaded(
            DocumentLoaded(
                1,
                doc_idx=20,
                num_segments=2,
                gen_prefilled=False,
                commit=False,
                document=stale,
            )
        )
        assert app._state.document_text == "hello world"


@pytest.mark.anyio
async def test_no_preview_without_loader():
    """Without a doc_loader, navigation in doc bar should not change text."""
    app = _make_app()  # no doc_loader
    async with app.run_test() as pilot:
        original_text = app._state.document_text

        await pilot.press("alt+up")
        await pilot.press("right")
        await pilot.pause()

        # Text should be unchanged (no loader to preview with)
        assert app._state.document_text == original_text


# ---------------------------------------------------------------------------
# 20. Doc bar preview: 1 segment shows only generation marker
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_preview_single_segment_is_generation_marker():
    """With segments reduced to 1, text should be empty and the sole segment
    should be a zero-width generation target (marker only)."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Go to doc bar
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR

        # Navigate to doc 20 (default segments = 2)
        await pilot.press("right")
        await pilot.pause()
        assert app._state.document_text == "doc twenty"

        # Decrease segments to 1 (up key decrements)
        await pilot.press("up")
        await pilot.pause()

        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.num_segments_for_selected == 1

        # With 1 segment the text should be empty — sole segment is the marker
        assert app._state.document_text == ""
        assert len(app._state.segments) == 1
        seg = app._state.segments[0]
        assert seg.is_generation_target
        assert seg.char_start == 0
        assert seg.char_end == 0
        assert seg.target_token_count is not None


# ---------------------------------------------------------------------------
# 21. Loaded doc in doc bar: matching segments bright after commit
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_loaded_doc_bright_segments_after_commit():
    """After committing a doc, entering doc bar auto-selects it with bright segments."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 20 (2 segments)
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(20, 2))
        await pilot.pause()
        assert app._state.document_text == "doc twenty"
        assert ds.loaded_doc_idx == 20

        # Enter doc bar — auto-selects loaded doc 20
        await pilot.press("alt+up")
        await pilot.pause()
        assert ds.selected_doc_idx == 20

        tp = app.query_one("#text-panel", TextPanel)
        assert tp.dimmed
        # All segments match fresh doc — bright set should be populated
        assert tp.bright_segment_indices is not None
        assert len(tp.bright_segment_indices) > 0


# ---------------------------------------------------------------------------
# 22. Non-loaded doc has no bright segments
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_non_loaded_doc_no_bright_segments():
    """Navigating to a non-loaded doc should not set bright segments."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        await pilot.press("alt+up")
        # Navigate to doc 20 (not loaded — loaded is 10)
        await pilot.press("right")
        await pilot.pause()

        tp = app.query_one("#text-panel", TextPanel)
        assert tp.dimmed
        assert tp.bright_segment_indices is None


# ---------------------------------------------------------------------------
# 23. Escape from doc bar clears bright segments
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_escape_clears_bright_segments():
    """Escaping doc bar restores committed state and clears bright segments."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        original_text = app._state.document_text

        await pilot.press("alt+up")
        await pilot.pause()

        tp = app.query_one("#text-panel", TextPanel)
        # Loaded doc 10 is selected — bright segments set
        assert tp.bright_segment_indices is not None

        await pilot.press("escape")
        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert not tp.dimmed
        assert tp.bright_segment_indices is None
        assert app._state.document_text == original_text


# ---------------------------------------------------------------------------
# 24. Enter on loaded doc commits fresh version
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_enter_on_loaded_doc_commits_fresh():
    """Pressing Enter on the loaded doc commits the fresh version."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 20
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(20, 2))
        await pilot.pause()
        assert app._state.document_text == "doc twenty"

        # Enter doc bar — auto-selects loaded doc 20
        await pilot.press("alt+up")
        await pilot.pause()
        assert ds.selected_doc_idx == 20

        tp = app.query_one("#text-panel", TextPanel)
        assert tp.dimmed
        assert tp.bright_segment_indices is not None

        # Press Enter to commit the loaded doc (fresh version)
        await pilot.press("enter")
        await pilot.pause()

        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert not tp.dimmed
        assert tp.bright_segment_indices is None
        assert app._state.document_text == "doc twenty"


# ---------------------------------------------------------------------------
# 25. Edit segment then doc bar: only segments before edit are bright
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edit_segment_then_doc_bar_partial_bright():
    """After editing seg 1 (with retokenization), seg 0 bright, rest dim."""
    from scripts.completion_tui.segment_ops import on_text_insert

    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Load doc 30 (4 segments: AAAA|BBBB|CCCC|DDDD, last is gen target)
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 4))
        await pilot.pause()

        # After truncation: text = "AAAABBBBCCCC", seg3 is zero-width gen target
        assert app._state.document_text == "AAAABBBBCCCC"
        assert len(app._state.segments) == 4

        # Edit segment 1 via on_text_insert (triggers retokenize_boundaries,
        # which shifts ALL segment boundaries — the real-world scenario).
        on_text_insert(app._state, 5, "X", app._require_tokenizer())
        assert app._state.document_text == "AAAABXBBBCCCC"

        # Enter doc bar — auto-selects loaded doc 30
        tp = app.query_one("#text-panel", TextPanel)
        await pilot.press("alt+up")
        await pilot.pause()
        assert ds.selected_doc_idx == 30
        assert ds.loaded_doc_idx == 30
        assert tp.dimmed

        # Seg 0 should be bright (text at fresh positions matches snapshot)
        # Seg 1+ should be dim (edit at position 5 invalidates from seg 1)
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        assert 1 not in tp.bright_segment_indices
        assert 2 not in tp.bright_segment_indices
        assert 3 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 26. 'g' toggles gen_prefilled on DocSelector
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_g_toggles_gen_prefilled():
    """Pressing 'g' in doc bar toggles gen_prefilled for the selected doc."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR

        ds = app.query_one("#doc-selector", DocSelector)
        assert not ds.gen_prefilled_for_selected

        # Toggle on
        await pilot.press("g")
        await pilot.pause()
        assert ds.gen_prefilled_for_selected

        # Toggle off
        await pilot.press("g")
        await pilot.pause()
        assert not ds.gen_prefilled_for_selected


# ---------------------------------------------------------------------------
# 27. 'g' toggle shows full text (no truncation) in preview
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gen_prefilled_shows_full_text():
    """With gen_prefilled on, preview shows full document text without truncation."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        await pilot.press("alt+up")

        # Navigate to doc 20 (text="doc twenty text here", 2 segments)
        await pilot.press("right")
        await pilot.pause()

        # Default: truncated (last segment is gen target)
        assert app._state.document_text == "doc twenty"
        assert app._state.segments[-1].is_generation_target

        # Toggle gen_prefilled on
        await pilot.press("g")
        await pilot.pause()

        # Full text up to last segment boundary, no generation marker
        assert app._state.document_text == "doc twenty text here"
        assert not app._state.segments[-1].is_generation_target

        # Toggle gen_prefilled off — back to truncated
        await pilot.press("g")
        await pilot.pause()

        assert app._state.document_text == "doc twenty"
        assert app._state.segments[-1].is_generation_target


# ---------------------------------------------------------------------------
# 28. gen_prefilled state clears on escape from doc bar
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gen_prefilled_clears_on_escape():
    """Escaping doc bar clears gen_prefilled and restores committed state."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        original_text = app._state.document_text

        await pilot.press("alt+up")
        await pilot.press("right")
        await pilot.pause()

        # Toggle gen_prefilled on
        await pilot.press("g")
        await pilot.pause()
        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.gen_prefilled_for_selected

        # Escape restores committed state and clears overrides
        await pilot.press("escape")
        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert app._state.document_text == original_text

        # Re-enter doc bar: gen_prefilled should be cleared
        await pilot.press("alt+up")
        ds = app.query_one("#doc-selector", DocSelector)
        assert not ds.gen_prefilled_for_selected


# ---------------------------------------------------------------------------
# 29. Enter with gen_prefilled commits prefilled text
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_enter_with_gen_prefilled_commits_full_text():
    """Pressing Enter with 'g' toggled on commits the prefilled document."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        await pilot.press("alt+up")

        # Navigate to doc 20, toggle gen_prefilled on
        await pilot.press("right")
        await pilot.pause()
        assert app._state.document_text == "doc twenty"

        await pilot.press("g")
        await pilot.pause()
        assert app._state.document_text == "doc twenty text here"

        # Enter commits the prefilled version
        await pilot.press("enter")
        await pilot.pause()

        assert app._focus_panel == FocusPanel.TEXT_PANEL
        assert app._state.document_text == "doc twenty text here"
        last = app._state.segments[-1]
        assert not last.is_generation_target
        assert last.char_end == 20
        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.loaded_doc_idx == 20


# ---------------------------------------------------------------------------
# 30. Appending to end of segment dims it in doc bar
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_append_to_segment_end_dims_it():
    """Appending text at a segment boundary makes that segment dim in doc bar."""
    from scripts.completion_tui.segment_ops import on_text_insert

    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Load doc 30 (AAAA|BBBB|CCCC|DDDD, last is gen target)
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 4))
        await pilot.pause()

        assert app._state.document_text == "AAAABBBBCCCC"

        # Append "X" at position 12 (end of seg2 / start of gen target)
        on_text_insert(app._state, 12, "X", app._require_tokenizer())
        assert "X" in app._state.document_text

        # Enter doc bar — auto-selects loaded doc 30
        tp = app.query_one("#text-panel", TextPanel)
        await pilot.press("alt+up")
        await pilot.pause()

        assert tp.bright_segment_indices is not None
        # Seg 0 and seg 1 should be bright (untouched)
        assert 0 in tp.bright_segment_indices
        assert 1 in tp.bright_segment_indices
        # Seg 2 dim — snap segment is wider (retokenization extended it)
        assert 2 not in tp.bright_segment_indices
        # Gen target should be dim
        assert 3 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 30b. Appending at mid-document boundary dims that segment
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_append_at_mid_boundary_dims_segment():
    """Inserting at a mid-document segment boundary dims from that point."""
    from scripts.completion_tui.segment_ops import on_text_insert

    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Load doc 30 with 4 segments (text = "AAAABBBBCCCC")
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 4))
        await pilot.pause()

        # Insert "X" at position 8 (boundary between seg1 and seg2)
        on_text_insert(app._state, 8, "X", app._require_tokenizer())

        # Enter doc bar
        tp = app.query_one("#text-panel", TextPanel)
        await pilot.press("alt+up")
        await pilot.pause()

        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        # Seg 1 should be dim — boundary check catches snap "X" vs fresh "C"
        assert 1 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 31. Changing segment count on loaded doc keeps matched segments bright
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_segment_count_change_keeps_matched_segments_bright():
    """Increasing segment count on the loaded doc should not dim matched segments."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 30 with 2 segments (text = "AAAA", seg0=[0,4), gen=[4,4))
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 2))
        await pilot.pause()
        assert app._state.document_text == "AAAA"

        # Enter doc bar — auto-selects loaded doc 30
        await pilot.press("alt+up")
        await pilot.pause()
        tp = app.query_one("#text-panel", TextPanel)
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices

        # Increase segment count to 3 (down key increments)
        await pilot.press("down")
        await pilot.pause()
        assert ds.num_segments_for_selected == 3

        # Seg 0 should still be bright — its text is unchanged
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        # Seg 1 is new text not in committed state — dim
        assert 1 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 32. Append then gen_prefill dims edited segment
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_append_then_gen_prefill_dims_edited_segment():
    """Appending at gen-target boundary then 'g': content segments bright,
    gen-target segment dim (snap has different text there)."""
    from scripts.completion_tui.segment_ops import on_text_insert

    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 30 with 4 segments (text = "AAAABBBBCCCC")
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 4))
        await pilot.pause()
        assert app._state.document_text == "AAAABBBBCCCC"

        # Append "X" at position 12 — retokenization extends seg2
        on_text_insert(app._state, 12, "X", app._require_tokenizer())

        # Enter doc bar
        await pilot.press("alt+up")
        await pilot.pause()
        tp = app.query_one("#text-panel", TextPanel)
        assert tp.bright_segment_indices is not None
        # Seg 0-1 bright, seg 2 dim (snap segment wider), gen dim
        assert 0 in tp.bright_segment_indices
        assert 1 in tp.bright_segment_indices
        assert 2 not in tp.bright_segment_indices
        assert 3 not in tp.bright_segment_indices

        # Press 'g' to prefill — seg 2 remains dim
        await pilot.press("g")
        await pilot.pause()
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        assert 1 in tp.bright_segment_indices
        assert 2 not in tp.bright_segment_indices
        assert 3 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 33. Unedited doc with gen_prefill ON: all segments bright
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gen_prefill_unedited_committed_bright_gen_dim():
    """On an unedited loaded doc, pressing 'g': committed content segments
    are bright, the ex-gen-target showing uncommitted text is dim."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 30 with 4 segments (3 content + gen target)
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 4))
        await pilot.pause()
        assert app._state.document_text == "AAAABBBBCCCC"

        # Enter doc bar — all bright (unedited, non-prefilled)
        await pilot.press("alt+up")
        await pilot.pause()
        tp = app.query_one("#text-panel", TextPanel)
        assert tp.bright_segment_indices is not None
        assert {0, 1, 2, 3} == tp.bright_segment_indices

        # Press 'g' to prefill — committed content bright, gen text dim
        await pilot.press("g")
        await pilot.pause()
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        assert 1 in tp.bright_segment_indices
        assert 2 in tp.bright_segment_indices
        # seg3 (ex-gen-target "DDDD") was not committed → dim
        assert 3 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 34. Append then fewer segments + gen_prefill dims last segment
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_append_fewer_segs_gen_prefill_dims_last():
    """Append to last content seg, decrement segments, press 'g':
    the last segment in the gen_prefilled view should be dim."""
    from scripts.completion_tui.segment_ops import on_text_insert

    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 30 with 3 segments (text = "AAAABBBB", gen at [8,8))
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 3))
        await pilot.pause()
        assert app._state.document_text == "AAAABBBB"

        # Append "X" at position 8 (end of seg1 / gen target boundary)
        on_text_insert(app._state, 8, "X", app._require_tokenizer())

        # Enter doc bar, decrement to 2 segments (up key decrements)
        await pilot.press("alt+up")
        await pilot.press("up")
        await pilot.pause()
        ds = app.query_one("#doc-selector", DocSelector)
        assert ds.num_segments_for_selected == 2

        tp = app.query_one("#text-panel", TextPanel)
        # Non-prefilled: gen markers dim (zero-width check)
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        assert 1 not in tp.bright_segment_indices

        # Press 'g' to prefill — last segment should still be dim
        await pilot.press("g")
        await pilot.pause()
        assert tp.bright_segment_indices is not None
        assert 0 in tp.bright_segment_indices
        assert 1 not in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 35. Committed content bright at all segment counts with gen_prefill
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_committed_content_bright_across_segment_counts():
    """With 4 committed segments (3 content + gen), gen_prefill ON:
    committed content segments are bright at any segment count,
    ex-gen-target text is dim."""
    loader = _make_multi_doc_loader()
    app = _make_app(doc_loader=loader)
    async with app.run_test() as pilot:
        # Commit doc 30 with 4 segments (text = "AAAABBBBCCCC")
        ds = app.query_one("#doc-selector", DocSelector)
        ds.post_message(DocSelector.DocumentSelected(30, 4))
        await pilot.pause()

        await pilot.press("alt+up")
        tp = app.query_one("#text-panel", TextPanel)

        # --- 4 segments, gen_prefill ON ---
        await pilot.press("g")
        await pilot.pause()
        assert tp.bright_segment_indices is not None
        assert {0, 1, 2} == tp.bright_segment_indices & {0, 1, 2}
        assert 3 not in tp.bright_segment_indices  # gen text not committed

        # --- 3 segments, gen_prefill ON ---
        await pilot.press("g")  # toggle off
        await pilot.press("up")  # decrement to 3 (up decrements)
        await pilot.press("g")  # toggle on
        await pilot.pause()
        assert ds.num_segments_for_selected == 3
        assert tp.bright_segment_indices is not None
        # All 3 shown segments are committed content → bright
        assert 0 in tp.bright_segment_indices
        assert 1 in tp.bright_segment_indices
        assert 2 in tp.bright_segment_indices

        # --- 2 segments, gen_prefill ON ---
        await pilot.press("g")  # toggle off
        await pilot.press("up")  # decrement to 2 (up decrements)
        await pilot.press("g")  # toggle on
        await pilot.pause()
        assert ds.num_segments_for_selected == 2
        assert tp.bright_segment_indices is not None
        # Both shown segments are committed content -> bright
        assert 0 in tp.bright_segment_indices
        assert 1 in tp.bright_segment_indices


# ---------------------------------------------------------------------------
# 36. Seed input: Tab from segment selector focuses seed widget
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tab_from_segment_selector_focuses_seed_input():
    app = _make_app()
    async with app.run_test() as pilot:
        # Navigate to segment selector
        await pilot.press("alt+down")  # TEXT -> SEGMENT_SELECTOR
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR
        # Tab should move to seed input
        await pilot.press("tab")
        await pilot.pause()
        assert app._focus_panel == FocusPanel.SEED_INPUT


# ---------------------------------------------------------------------------
# 37. Seed input: Escape from seed input returns to segment selector
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_escape_from_seed_input_returns_to_segment_selector():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")  # TEXT -> SEGMENT_SELECTOR
        await pilot.press("tab")  # -> SEED_INPUT
        assert app._focus_panel == FocusPanel.SEED_INPUT
        await pilot.press("escape")
        await pilot.pause()
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR


# ---------------------------------------------------------------------------
# 38. Seed input: digit input changes seed value
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_seed_input_digit_changes_seed():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")  # TEXT -> SEGMENT_SELECTOR
        await pilot.press("tab")  # -> SEED_INPUT
        si = app.query_one("#seed-input", SeedInput)
        # First digit overwrites
        await pilot.press("7")
        assert si.seed == 7
        # Second digit appends
        await pilot.press("3")
        assert si.seed == 73
        # Escape syncs back to state
        await pilot.press("escape")
        await pilot.pause()
        assert app._state.seed == 73


# ---------------------------------------------------------------------------
# 39. Segment navigation wraps around
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_segment_navigation_wraps():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")  # TEXT -> SEGMENT_SELECTOR
        ss = app.query_one("#segment-selector", SegmentSelector)
        assert ss.selected_idx == 0
        # Right to end then wrap
        await pilot.press("right")
        assert ss.selected_idx == 1
        await pilot.press("right")  # wraps to 0
        assert ss.selected_idx == 0
        # Left from 0 wraps to end
        await pilot.press("left")
        assert ss.selected_idx == 1


# ---------------------------------------------------------------------------
# 40. Seed input: Enter exits seed input back to segment selector
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_enter_from_seed_input_returns_to_segment_selector():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")  # TEXT -> SEGMENT_SELECTOR
        await pilot.press("tab")  # -> SEED_INPUT
        assert app._focus_panel == FocusPanel.SEED_INPUT
        await pilot.press("enter")
        await pilot.pause()
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR


# ---------------------------------------------------------------------------
# 41. Seed input: re-entering resets first-digit so new input overwrites
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_seed_input_reentry_overwrites():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")  # TEXT -> SEGMENT_SELECTOR
        await pilot.press("tab")  # -> SEED_INPUT
        si = app.query_one("#seed-input", SeedInput)
        await pilot.press("3")
        await pilot.press("4")
        await pilot.press("3")
        assert si.seed == 343
        # Go back to segment selector and re-enter seed
        await pilot.press("escape")
        await pilot.pause()
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR
        await pilot.press("tab")  # -> SEED_INPUT again
        await pilot.press("4")
        await pilot.press("4")
        assert si.seed == 44  # overwrites, not 34344


# ---------------------------------------------------------------------------
# Doc selector preserves browsing position on refocus
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_doc_selector_preserves_position_on_refocus():
    """Leaving and re-entering doc selector should NOT reset browsing position."""
    app = _make_app()
    async with app.run_test() as pilot:
        ds = app.query_one("#doc-selector", DocSelector)

        # Enter doc selector
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR
        initial_idx = ds._selected_idx

        # Navigate right twice
        await pilot.press("right")
        await pilot.press("right")
        browsed_idx = ds._selected_idx
        assert browsed_idx != initial_idx

        # Leave doc selector → text panel
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.TEXT_PANEL

        # Re-enter doc selector
        await pilot.press("alt+up")
        assert app._focus_panel == FocusPanel.DOC_SELECTOR

        # Position should be preserved, not reset to loaded doc
        assert ds._selected_idx == browsed_idx


# ---------------------------------------------------------------------------
# Alt+Left/Right switches segments in feature editor
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_feature_editor_alt_right_switches_to_next_segment():
    """Alt+Right in feature editor should close current and open next segment."""
    app = _make_app()
    async with app.run_test() as pilot:
        # Navigate to segment selector and open feature editor on seg 0
        await pilot.press("alt+down")
        assert app._focus_panel == FocusPanel.SEGMENT_SELECTOR
        await pilot.press("enter")
        await pilot.pause()
        assert app._overlay_open
        assert app._editing_segment_idx == 0

        # Alt+Right to switch to segment 1
        await pilot.press("alt+right")
        await pilot.pause()
        assert app._overlay_open
        assert app._editing_segment_idx == 1


@pytest.mark.anyio
async def test_feature_editor_alt_left_at_first_segment_wraps():
    """Alt+Left at first segment should wrap to last segment."""
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")
        await pilot.press("enter")
        await pilot.pause()
        assert app._editing_segment_idx == 0

        await pilot.press("alt+left")
        await pilot.pause()
        assert app._editing_segment_idx == 1  # wraps to last


@pytest.mark.anyio
async def test_feature_editor_alt_right_at_last_segment_wraps():
    """Alt+Right at last segment should wrap to first segment."""
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("alt+down")
        # Navigate to last segment (seg 1)
        await pilot.press("right")
        await pilot.press("enter")
        await pilot.pause()
        assert app._editing_segment_idx == 1

        await pilot.press("alt+right")
        await pilot.pause()
        assert app._editing_segment_idx == 0  # wraps to first
