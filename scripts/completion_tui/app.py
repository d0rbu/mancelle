"""Main Textual app integrating all completion TUI widgets.

Wires together DocSelector, TextPanel, SegmentSelector, SegmentBar,
and overlay widgets (FeatureEditor, FeatureSearch, FeatureDetail).
Manages focus cycling, global conditioning toggle, document loading,
generation streaming, and overlay lifecycle.
"""

from __future__ import annotations

import base64
import os
import random
import subprocess
import threading
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

import torch as t
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.events import Key
from textual.message import Message

from scripts.completion_tui.cluster_view import ClusterView
from scripts.completion_tui.data_loader import (
    TUIFeatureSamplingConfig,
    clustered_sample_segment_features,
    sample_segment_features,
)
from scripts.completion_tui.generation import (
    GenerationHandle,
    redo_generation,
    undo_generation,
)
from scripts.completion_tui.highlight_types import (
    FeatureHighlightRequest,
    FeatureHighlightResult,
    build_highlight_request,
)
from scripts.completion_tui.segment_ops import retokenize_boundaries
from scripts.completion_tui.state import (
    DocumentData,
    FeatureCondition,
    GenerationSnapshot,
    SegmentClusterEvidence,
    SessionState,
    deep_copy_segments,
)
from scripts.completion_tui.widgets.cluster_members import ClusterMembersPanel
from scripts.completion_tui.widgets.doc_selector import DocSelector
from scripts.completion_tui.widgets.feature_detail import FeatureDetail
from scripts.completion_tui.widgets.feature_editor import (
    DESCRIPTION_LOADING,
    FeatureEditor,
)
from scripts.completion_tui.widgets.feature_search import FeatureSearch
from scripts.completion_tui.widgets.seed_input import SeedInput
from scripts.completion_tui.widgets.segment_bar import SegmentBar
from scripts.completion_tui.widgets.segment_selector import SegmentSelector
from scripts.completion_tui.widgets.text_panel import TextPanel, TextPanelHelp
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.sampling import NegativeSamplingSelection, SamplingMode, SamplingSelection
from steering.tokenizer_protocols import TextTokenizer, require_text_tokenizer

if TYPE_CHECKING:
    from scripts.completion_tui.data_loader import DocumentLoader
    from scripts.completion_tui.generation import GenerationManager, LazyGenerationManager


class DescriptionServiceProtocol(Protocol):
    """Description-service capability consumed by ``CompletionApp``."""

    def get_description_sync(self, feature_id: int) -> str | None: ...

    def get_all_cached(self) -> dict[int, str]: ...

    def request_description(
        self,
        feature_id: int,
        callback: Callable[[int, str], None],
        /,
        *,
        on_start: Callable[[int], None] | None = None,
    ) -> None: ...

    def shutdown(self) -> None: ...


class LiveHighlightServiceProtocol(Protocol):
    """Live-highlight capability consumed by ``CompletionApp``."""

    def compute(self, request: FeatureHighlightRequest) -> FeatureHighlightResult: ...


class FocusPanel(Enum):
    """Tracks which major panel currently has focus."""

    DOC_SELECTOR = auto()
    TEXT_PANEL = auto()
    SEGMENT_SELECTOR = auto()
    SEED_INPUT = auto()
    OVERLAY = auto()


class LiveHighlightComputed(Message):
    """Thread-safe delivery of a completed live-highlight request."""

    def __init__(
        self,
        seq: int,
        request_key: str,
        result: FeatureHighlightResult,
    ) -> None:
        super().__init__()
        self.seq = seq
        self.request_key = request_key
        self.result = result


class LiveClusterEvidenceComputed(Message):
    """Thread-safe delivery of a composite member-evidence probe."""

    def __init__(
        self,
        seq: int,
        request_key: str,
        result: FeatureHighlightResult,
    ) -> None:
        super().__init__()
        self.seq = seq
        self.request_key = request_key
        self.result = result


class DocumentLoaded(Message):
    """Thread-safe delivery of a document preview or selection."""

    def __init__(
        self,
        seq: int,
        *,
        doc_idx: int,
        num_segments: int,
        gen_prefilled: bool,
        commit: bool,
        document: DocumentData | None,
        error: str | None = None,
    ) -> None:
        super().__init__()
        self.seq = seq
        self.doc_idx = doc_idx
        self.num_segments = num_segments
        self.gen_prefilled = gen_prefilled
        self.commit = commit
        self.document = document
        self.error = error


def _document_evidence_by_segment_id(
    document: DocumentData,
) -> dict[str, SegmentClusterEvidence]:
    """Copy position-keyed document evidence onto stable TUI segment IDs."""
    return {
        segment.id: SegmentClusterEvidence(
            fingerprint=entry.fingerprint,
            evidence=entry.evidence,
        )
        for index, segment in enumerate(document.segments)
        if (entry := document.evidence_by_segment_index.get(index)) is not None
    }


class CompletionApp(App):
    """Main TUI for interactive completion with per-segment SAE conditioning."""

    CSS = """
    #doc-selector {
        dock: top;
        height: auto;
        min-height: 1;
    }
    #content-area {
        height: 1fr;
        layers: default overlay;
    }
    #text-panel {
        height: 1fr;
    }
    #bottom-bar {
        dock: bottom;
        height: 2;
    }
    #segment-selector {
        width: 1fr;
    }
    #seed-input {
        width: auto;
        min-width: 14;
    }
    #segment-bar {
        dock: right;
        width: 3;
        display: none;
        layer: overlay;
    }
    .overlay-panel {
        dock: bottom;
        height: 60%;
    }
    """

    BINDINGS = [
        Binding("alt+down", "cycle_focus_down", "Next panel", priority=True),
        Binding("alt+up", "cycle_focus_up", "Previous panel", priority=True),
        Binding("ctrl+i", "toggle_conditioning", "Toggle conditioning", priority=True),
        Binding("ctrl+t", "toggle_conditioning", "Toggle conditioning", priority=True),
        Binding("ctrl+c", "copy_text", "Copy text", priority=True),
    ]

    def __init__(
        self,
        *,
        state: SessionState | None = None,
        tokenizer: object | None = None,
        doc_loader: DocumentLoader | None = None,
        description_service: DescriptionServiceProtocol | None = None,
        generation_manager: GenerationManager | LazyGenerationManager | None = None,
        available_doc_ids: list[int] | None = None,
        total_segments_map: dict[int, int] | None = None,
        sampling_config: TUIFeatureSamplingConfig | None = None,
        live_highlight_service: LiveHighlightServiceProtocol | None = None,
        cluster_view: ClusterView | None = None,
        initial_segment_evidence: dict[str, SegmentClusterEvidence] | None = None,
    ) -> None:
        super().__init__()
        self._state = state or SessionState(
            document_text="",
            segments=[],
            num_buckets=11,
            has_negatives=False,
        )
        self._tokenizer: TextTokenizer | None = (
            None
            if tokenizer is None
            else require_text_tokenizer(tokenizer, surface="completion TUI app")
        )
        self._doc_loader = doc_loader
        self._description_service = description_service
        self._generation_manager = generation_manager
        self._available_doc_ids = available_doc_ids or []
        self._total_segments_map = total_segments_map or {}
        self._sampling_config = sampling_config
        self._live_highlight_service = live_highlight_service
        self._cluster_view = cluster_view
        self._feature_mask: FeatureMask | None = (
            sampling_config.feature_mask if sampling_config is not None else None
        )
        self._segment_evidence: dict[str, SegmentClusterEvidence] = {
            segment_id: SegmentClusterEvidence(
                fingerprint=entry.fingerprint,
                evidence=entry.evidence,
            )
            for segment_id, entry in (initial_segment_evidence or {}).items()
        }
        self._prune_segment_evidence()
        self._last_seg_features: SegmentFeatures | None = None

        # Focus state
        self._focus_panel = FocusPanel.TEXT_PANEL

        # Overlay state
        self._overlay_open = False
        self._editing_segment_idx: int | None = None
        self._editor: FeatureEditor | None = None
        self._search: FeatureSearch | None = None
        self._detail: FeatureDetail | None = None
        self._cluster_panel: ClusterMembersPanel | None = None
        self._overlay_sub_panel: str = ""
        self._detail_return_panel: str = "editor"
        self._live_highlight_seq: int = 0
        self._live_highlight_request_key: str | None = None

        # Doc selector browsing state
        self._doc_selector_visited = False

        # Preview state (while browsing docs in doc selector)
        self._preview_snapshot: dict[str, Any] | None = None
        self._preview_doc_idx: int | None = None
        self._preview_num_segments: int | None = None
        self._preview_gen_prefilled: bool = False
        self._preview_canonical_text: str = ""
        self._document_load_seq: int = 0

        # Generation state
        self._active_generation: GenerationHandle | None = None
        self._generated_tokens: list[int] = []
        self._generated_text: str = ""
        self._pre_regen_snapshot: GenerationSnapshot | None = None

    def _require_tokenizer(self) -> TextTokenizer:
        """Return the validated tokenizer needed by token-aware actions."""
        if self._tokenizer is None:
            return require_text_tokenizer(
                self._tokenizer,
                surface="completion TUI app",
            )
        return self._tokenizer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_unmount(self) -> None:
        self._invalidate_document_loads()
        if self._description_service is not None:
            self._description_service.shutdown()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield DocSelector(
            self._available_doc_ids,
            self._total_segments_map,
            id="doc-selector",
        )
        with Container(id="content-area"):
            yield SegmentBar(id="segment-bar")
            yield TextPanelHelp(id="text-panel-help")
            yield TextPanel(
                self._state,
                self._tokenizer,
                id="text-panel",
            )
        with Horizontal(id="bottom-bar"):
            yield SegmentSelector(
                self._state.segments,
                seed=self._state.seed,
                id="segment-selector",
            )
            yield SeedInput(seed=self._state.seed, id="seed-input")

    def on_mount(self) -> None:
        self._set_focus(FocusPanel.TEXT_PANEL)

    # ------------------------------------------------------------------
    # Document preview (while browsing in DocSelector)
    # ------------------------------------------------------------------

    def _save_preview_snapshot(self) -> None:
        """Save current state before previewing documents."""
        state = self._state
        self._preview_snapshot = {
            "text": state.document_text,
            "segments": deep_copy_segments(state.segments),
            "cursor": state.cursor_position,
            "history": list(state.generation_history),
            "redo": list(state.generation_redo_stack),
            "conditioning": state.global_conditioning,
            "memory": dict(state.conditioning_memory),
        }
        self._preview_doc_idx = None
        self._preview_num_segments = None
        self._preview_gen_prefilled = False
        self._preview_canonical_text = ""

    def _restore_preview_snapshot(self, *, keep_snapshot: bool = False) -> None:
        """Restore committed state from snapshot.

        When *keep_snapshot* is True the snapshot is preserved so the
        committed state can be re-applied later (e.g. when navigating
        back to the loaded doc within the doc bar).
        """
        if self._preview_snapshot is None:
            return
        snap = self._preview_snapshot
        state = self._state
        state.document_text = snap["text"]
        state.segments = deep_copy_segments(snap["segments"])
        state.cursor_position = snap["cursor"]
        state.generation_history = list(snap["history"])
        state.generation_redo_stack = list(snap["redo"])
        state.global_conditioning = snap["conditioning"]
        state.conditioning_memory = dict(snap["memory"])
        self._prune_segment_evidence()
        if not keep_snapshot:
            self._preview_snapshot = None
        self._preview_doc_idx = None
        self._preview_num_segments = None
        self._preview_gen_prefilled = False
        self._preview_canonical_text = ""
        self._refresh_preview_panels()

    def _truncate_for_generation(self, segments: list, text: str) -> str:
        """Truncate *text* so the last segment is a zero-width generation target.

        Returns the truncated text.  Modifies the last segment in-place.
        With 1 segment the text becomes empty (the sole segment is the marker).
        """
        if not segments:
            return text
        last = segments[-1]
        cutoff = segments[-2].char_end if len(segments) >= 2 else 0
        if not last.is_generation_target:
            last.is_generation_target = True
        seg_text = text[last.char_start : last.char_end]
        last.target_token_count = max(
            1,
            len(self._require_tokenizer().encode(seg_text)) if seg_text else 1,
        )
        last.char_start = cutoff
        last.char_end = cutoff
        return text[:cutoff]

    def _prefill_generation(self, segments: list, text: str) -> str:
        """Keep the last segment's text intact for generation prefill.

        Returns text truncated to the last segment boundary.  Runs
        retokenize_boundaries so that segment borders align with the
        nanochat tokenizer, and computes target_token_count from the
        actual token count.  Clears ``is_generation_target`` since the
        segment is fully filled.
        """
        if not segments:
            return text
        last = segments[-1]
        truncated = text[: last.char_end]
        if last.is_generation_target:
            tokenizer = self._require_tokenizer()
            retokenize_boundaries(truncated, segments, tokenizer)
            seg_text = truncated[last.char_start : last.char_end]
            last.target_token_count = max(
                1,
                len(tokenizer.encode(seg_text)) if seg_text else 1,
            )
            last.is_generation_target = False
        return truncated

    def _preview_document(
        self, doc_idx: int, num_segments: int, gen_prefilled: bool = False
    ) -> None:
        """Request a document preview without blocking the app event loop."""
        if self._doc_loader is None:
            return
        if (
            doc_idx == self._preview_doc_idx
            and num_segments == self._preview_num_segments
            and gen_prefilled == self._preview_gen_prefilled
        ):
            return
        self._request_document_load(
            doc_idx,
            num_segments,
            gen_prefilled=gen_prefilled,
            commit=False,
        )

    def _request_document_load(
        self,
        doc_idx: int,
        num_segments: int,
        *,
        gen_prefilled: bool,
        commit: bool,
    ) -> None:
        """Load a document on a worker and apply only the latest request.

        Feature shard lookup and segment sampling can decompress a large
        resident shard.  Running that work synchronously from a selector
        event freezes Textual, especially when arrow keys generate several
        requests in quick succession.  Worker cancellation is best effort;
        the sequence guard is authoritative and prevents stale previews from
        ever mutating app state.
        """
        loader = self._doc_loader
        if loader is None:
            return
        self._document_load_seq += 1
        seq = self._document_load_seq

        def _worker() -> None:
            try:
                document = loader.load_document(doc_idx, num_segments)
                error = None
            except Exception as exc:  # pragma: no cover - exercised by runtime failures
                document = None
                error = str(exc)
            self.post_message(
                DocumentLoaded(
                    seq,
                    doc_idx=doc_idx,
                    num_segments=num_segments,
                    gen_prefilled=gen_prefilled,
                    commit=commit,
                    document=document,
                    error=error,
                )
            )

        self.run_worker(
            _worker,
            group="document-load",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _invalidate_document_loads(self) -> None:
        """Invalidate and best-effort cancel document workers."""
        self._document_load_seq += 1
        try:
            self.workers.cancel_group(self, "document-load")
        except (AttributeError, RuntimeError):
            # App teardown can race worker cleanup; sequence invalidation is
            # still sufficient to discard a result already in flight.
            pass

    def _apply_preview_document(
        self,
        doc: DocumentData,
        doc_idx: int,
        num_segments: int,
        gen_prefilled: bool,
    ) -> None:
        """Apply a loaded preview on the Textual app thread."""
        if doc is None:
            return
        self._cache_document_evidence(doc)
        state = self._state
        state.segments = doc.segments
        if gen_prefilled:
            state.document_text = self._prefill_generation(state.segments, doc.text)
        else:
            state.document_text = self._truncate_for_generation(state.segments, doc.text)
        state.cursor_position = 0
        state.global_conditioning = True
        state.conditioning_memory.clear()
        state.generation_history.clear()
        state.generation_redo_stack.clear()
        self._preview_doc_idx = doc_idx
        self._preview_num_segments = num_segments
        self._preview_gen_prefilled = gen_prefilled
        self._preview_canonical_text = doc.text
        self._refresh_preview_panels()

    def on_document_loaded(self, message: DocumentLoaded) -> None:
        """Apply the newest document worker result on the app thread."""
        if message.seq != self._document_load_seq:
            return
        if message.error is not None:
            self.notify(f"Document load failed: {message.error}", severity="error", timeout=8)
            return
        if message.document is None:
            return
        if message.commit:
            self._apply_committed_document(
                message.document,
                message.doc_idx,
                message.gen_prefilled,
            )
        else:
            self._apply_preview_document(
                message.document,
                message.doc_idx,
                message.num_segments,
                message.gen_prefilled,
            )
            ds = self.query_one("#doc-selector", DocSelector)
            if message.doc_idx == ds.loaded_doc_idx:
                self._compute_bright_segments()

    def _cache_document_evidence(self, document: DocumentData) -> None:
        """Copy a loaded document's evidence into the segment-ID cache."""
        self._segment_evidence.update(_document_evidence_by_segment_id(document))

    def _prune_segment_evidence(self) -> None:
        """Discard cached evidence for segments no longer present in state."""
        live_segment_ids = {segment.id for segment in self._state.segments}
        self._segment_evidence = {
            segment_id: entry
            for segment_id, entry in self._segment_evidence.items()
            if segment_id in live_segment_ids
        }

    def _refresh_preview_panels(self) -> None:
        """Refresh text panel and segment selector after state change."""
        tp = self.query_one("#text-panel", TextPanel)
        tp.selected_segment_idx = None
        tp.bright_segment_indices = None
        tp.refresh()
        ss = self.query_one("#segment-selector", SegmentSelector)
        ss._segments = self._state.segments
        ss.refresh()

    def _compute_bright_segments(self) -> None:
        """Compare snapshot text to fresh preview using fresh segment boundaries.

        Retokenization after edits shifts ALL segment boundaries, so we cannot
        compare snapshot vs fresh by their respective segment boundaries.
        Instead, slice both texts at the fresh doc's canonical boundaries.
        Text before the edit point occupies the same character positions in
        both texts, so the comparison is correct.

        Once a mismatch is found, all subsequent segments are dimmed — an edit
        invalidates everything from that point forward.

        Three checks per segment, in order:

        1. **Text comparison** — catches edits inside a segment and content
           shifted by inserts at the *previous* boundary.

        2. **Boundary check** — detects edits that insert at a segment
           boundary where the shifted content is not caught by the next
           segment's text comparison.

           *Non-prefilled*: only runs before the gen-target zone
           (``trunc_point``); a separate width check and zero-width check
           cover the last content segment and the gen target.

           *Gen-prefilled*: compares the snap character against the
           *canonical* document text (``_preview_canonical_text``).  This
           distinguishes user edits (snap ≠ canonical) from natural
           committed content that extends beyond the current view (snap ==
           canonical).

        3. **Width / zero-width checks** (non-prefilled only) — detect
           appends to the last content segment and snap content in a
           zero-width gen target.
        """
        snap = self._preview_snapshot
        if snap is None:
            return
        snap_text: str = snap["text"]
        snap_segs: list = snap["segments"]
        state = self._state
        canonical = self._preview_canonical_text
        bright: set[int] = set()

        # The generation-target zone starts at the last segment's char_start.
        trunc_point = state.segments[-1].char_start if state.segments else len(state.document_text)

        for idx, seg in enumerate(state.segments):
            fresh_slice = state.document_text[seg.char_start : seg.char_end]
            snap_slice = snap_text[seg.char_start : seg.char_end]
            if fresh_slice != snap_slice:
                break

            # --- Boundary check ---
            if self._preview_gen_prefilled:
                # Compare snap boundary against canonical doc text.
                # Matches → natural committed content (skip).
                # Mismatches → user edited at this boundary (dim).
                snap_boundary = snap_text[seg.char_end : seg.char_end + 1]
                if snap_boundary:
                    canonical_char = canonical[seg.char_end : seg.char_end + 1]
                    if snap_boundary != canonical_char:
                        break
            else:
                # Non-prefilled: only check before the gen-target zone.
                if seg.char_end < trunc_point:
                    snap_boundary = snap_text[seg.char_end : seg.char_end + 1]
                    if (
                        snap_boundary
                        and snap_boundary != state.document_text[seg.char_end : seg.char_end + 1]
                    ):
                        break

                # Width check at trunc_point (same segment count only).
                if (
                    seg.char_end >= trunc_point
                    and seg.char_start < seg.char_end
                    and len(state.segments) == len(snap_segs)
                    and idx < len(snap_segs)
                ):
                    snap_seg = snap_segs[idx]
                    if (snap_seg.char_end - snap_seg.char_start) > (seg.char_end - seg.char_start):
                        break

                # Zero-width segment at text end with snap content → dim.
                if (
                    seg.char_start == seg.char_end
                    and seg.char_start >= len(state.document_text)
                    and len(snap_text) > seg.char_start
                ):
                    break

            bright.add(seg.segment_idx)
        tp = self.query_one("#text-panel", TextPanel)
        tp.bright_segment_indices = bright
        tp.refresh()

    # ------------------------------------------------------------------
    # Focus management
    # ------------------------------------------------------------------

    def _set_focus(self, panel: FocusPanel) -> None:
        """Switch focus to the given panel, updating highlight state."""
        old = self._focus_panel
        self._focus_panel = panel

        # Preview management: save on enter, restore on leave
        if panel == FocusPanel.DOC_SELECTOR and old != FocusPanel.DOC_SELECTOR:
            self._save_preview_snapshot()
        elif old == FocusPanel.DOC_SELECTOR and panel != FocusPanel.DOC_SELECTOR:
            self._invalidate_document_loads()
            if self._preview_snapshot is not None:
                self._restore_preview_snapshot()

        # SegmentBar: visible only when DocSelector focused
        bar = self.query_one("#segment-bar", SegmentBar)
        if panel == FocusPanel.DOC_SELECTOR:
            bar.show_bar()
            bar.display = True
            # Seed loaded doc's segment count from current state, then sync bar
            ds = self.query_one("#doc-selector", DocSelector)
            if ds.loaded_doc_idx is not None and self._state.segments:
                ds.seed_segment_count(ds.loaded_doc_idx, len(self._state.segments))
            # Snap selection to loaded doc only on first visit so the user
            # starts at the current document, but preserve position on re-entry.
            if (
                not self._doc_selector_visited
                and ds.loaded_doc_idx is not None
                and ds.loaded_doc_idx in ds.filtered_docs
            ):
                ds._selected_idx = ds.filtered_docs.index(ds.loaded_doc_idx)
                self._doc_selector_visited = True
            bar.update_segments(ds.num_segments_for_selected, ds.total_segments_for_selected)
            # Preview highlighted doc (always — loaded doc gets bright segments)
            if old != FocusPanel.DOC_SELECTOR:
                doc_idx = ds.selected_doc_idx
                if doc_idx is not None:
                    self._preview_document(
                        doc_idx,
                        ds.num_segments_for_selected,
                        ds.gen_prefilled_for_selected,
                    )
                    if doc_idx == ds.loaded_doc_idx:
                        self._compute_bright_segments()
        else:
            bar.hide_bar()
            bar.display = False
            if old == FocusPanel.DOC_SELECTOR:
                self.query_one("#doc-selector", DocSelector).reset_segment_overrides()
        bar.refresh()

        # Dim text panel when browsing docs, restore otherwise
        tp = self.query_one("#text-panel", TextPanel)
        tp.dimmed = panel == FocusPanel.DOC_SELECTOR
        if not tp.dimmed:
            tp.bright_segment_indices = None
        tp.refresh()

        # Unhighlight text when leaving segment-related panels
        seg_panels = (FocusPanel.SEGMENT_SELECTOR, FocusPanel.SEED_INPUT, FocusPanel.OVERLAY)
        if old in seg_panels and panel not in seg_panels:
            tp.selected_segment_idx = None

        # Set Textual focus on target widget
        if panel == FocusPanel.DOC_SELECTOR:
            self.query_one("#doc-selector", DocSelector).focus()
        elif panel == FocusPanel.TEXT_PANEL:
            self.query_one("#text-panel", TextPanel).focus()
        elif panel == FocusPanel.SEGMENT_SELECTOR:
            ss = self.query_one("#segment-selector", SegmentSelector)
            ss.focus()
            if ss.segments:
                self.query_one("#text-panel", TextPanel).selected_segment_idx = ss.selected_idx
        elif panel == FocusPanel.SEED_INPUT:
            si = self.query_one("#seed-input", SeedInput)
            si.reset_input()
            si.focus()
        elif panel == FocusPanel.OVERLAY:
            if self._overlay_sub_panel == "editor" and self._editor:
                self._editor.focus()
            elif self._overlay_sub_panel == "search" and self._search:
                self._search.focus()
            elif self._overlay_sub_panel == "detail" and self._detail:
                self._detail.focus()
            elif self._overlay_sub_panel == "cluster" and self._cluster_panel:
                self._cluster_panel.focus()

    def action_cycle_focus_down(self) -> None:
        """Ctrl+Down: DocSelector -> TextPanel -> SegSelector/Overlay -> wrap."""
        third = FocusPanel.OVERLAY if self._overlay_open else FocusPanel.SEGMENT_SELECTOR
        cycle = {
            FocusPanel.DOC_SELECTOR: FocusPanel.TEXT_PANEL,
            FocusPanel.TEXT_PANEL: third,
            FocusPanel.SEGMENT_SELECTOR: FocusPanel.DOC_SELECTOR,
            FocusPanel.SEED_INPUT: FocusPanel.DOC_SELECTOR,
            FocusPanel.OVERLAY: FocusPanel.DOC_SELECTOR,
        }
        self._set_focus(cycle[self._focus_panel])

    def action_cycle_focus_up(self) -> None:
        """Ctrl+Up: reverse of cycle_down."""
        third = FocusPanel.OVERLAY if self._overlay_open else FocusPanel.SEGMENT_SELECTOR
        cycle = {
            FocusPanel.DOC_SELECTOR: third,
            FocusPanel.TEXT_PANEL: FocusPanel.DOC_SELECTOR,
            FocusPanel.SEGMENT_SELECTOR: FocusPanel.TEXT_PANEL,
            FocusPanel.SEED_INPUT: FocusPanel.TEXT_PANEL,
            FocusPanel.OVERLAY: FocusPanel.TEXT_PANEL,
        }
        self._set_focus(cycle[self._focus_panel])

    # ------------------------------------------------------------------
    # Escape handling (app-level, for keys not caught by widgets)
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and self._focus_panel == FocusPanel.DOC_SELECTOR:
            self._set_focus(FocusPanel.TEXT_PANEL)
            event.prevent_default()
            event.stop()

    # ------------------------------------------------------------------
    # Conditioning toggle (Ctrl+I / Cmd+I)
    # ------------------------------------------------------------------

    def action_toggle_conditioning(self) -> None:
        state = self._state
        if state.global_conditioning:
            # Save current enabled states then disable all
            state.conditioning_memory = {seg.id: seg.enabled for seg in state.segments}
            for seg in state.segments:
                seg.enabled = False
            state.global_conditioning = False
        else:
            # Restore from memory
            for seg in state.segments:
                seg.enabled = state.conditioning_memory.get(seg.id, True)
            state.conditioning_memory.clear()
            state.global_conditioning = True
        self.query_one("#segment-selector", SegmentSelector).refresh()
        self.query_one("#text-panel", TextPanel).refresh()

    # ------------------------------------------------------------------
    # Copy text (Ctrl+C)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_clipboard(text: str) -> bool:
        """Write *text* to the system clipboard. Returns True on success.

        Tries OSC 52 first (works over SSH + tmux with ``set-clipboard on``),
        then falls back to local clipboard tools.
        """
        # OSC 52: ask the terminal to set its clipboard directly.
        # Works over SSH through Ghostty/tmux without any X11/Wayland tooling.
        try:
            b64 = base64.b64encode(text.encode()).decode()
            in_tmux = bool(os.environ.get("TMUX"))
            if in_tmux:
                seq = f"\033Ptmux;\033\033]52;c;{b64}\a\033\\"
            else:
                seq = f"\033]52;c;{b64}\a"
            with open("/dev/tty", "wb") as tty:
                tty.write(seq.encode())
            return True
        except OSError:
            pass

        # Fallback: local clipboard tools (Wayland / X11).
        for cmd in (
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
                return True
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return False

    def action_copy_text(self) -> None:
        """Copy document text to clipboard, scoped to the highlighted segment."""
        tp = self.query_one("#text-panel", TextPanel)
        text = self._state.document_text
        selected_seg_idx = tp.selected_segment_idx
        if selected_seg_idx is not None:
            seg = next(
                (s for s in self._state.segments if s.segment_idx == selected_seg_idx),
                None,
            )
            if seg is not None:
                text = text[: seg.char_end]
        ok = self._write_clipboard(text)
        if ok:
            chars = len(text)
            self.notify(f"Copied {chars} chars", timeout=2)
        else:
            self.notify(
                "Clipboard unavailable (install xclip or xsel)", severity="warning", timeout=3
            )

    # ------------------------------------------------------------------
    # Document selection
    # ------------------------------------------------------------------

    def on_doc_selector_selection_changed(self, msg: DocSelector.SelectionChanged) -> None:
        bar = self.query_one("#segment-bar", SegmentBar)
        bar.update_segments(msg.num_segments, msg.total_segments)
        bar.refresh()

        # Update preview while browsing in the doc bar
        if (
            self._focus_panel == FocusPanel.DOC_SELECTOR
            and self._preview_snapshot is not None
            and msg.doc_idx is not None
        ):
            self._preview_document(msg.doc_idx, msg.num_segments, msg.gen_prefilled)

    def on_doc_selector_document_selected(self, msg: DocSelector.DocumentSelected) -> None:
        self._close_overlay()

        if self._doc_loader is None:
            return

        # Read gen_prefilled before _set_focus clears overrides
        ds = self.query_one("#doc-selector", DocSelector)
        gen_prefilled = ds.gen_prefilled_for_selected

        # Invalidate any in-flight preview. The committed document is applied
        # by ``on_document_loaded`` after the worker finishes.
        self._invalidate_document_loads()
        self._preview_snapshot = None
        self._preview_doc_idx = None
        self._preview_num_segments = None
        self._preview_gen_prefilled = False
        self._preview_canonical_text = ""

        self._request_document_load(
            msg.doc_idx,
            msg.num_segments,
            gen_prefilled=gen_prefilled,
            commit=True,
        )

    def _apply_committed_document(
        self,
        doc: DocumentData,
        doc_idx: int,
        gen_prefilled: bool,
    ) -> None:
        """Apply a confirmed document after background loading completes."""
        self._cache_document_evidence(doc)

        state = self._state
        state.segments = doc.segments
        if gen_prefilled:
            state.document_text = self._prefill_generation(state.segments, doc.text)
        else:
            state.document_text = self._truncate_for_generation(state.segments, doc.text)
        self._prune_segment_evidence()
        state.cursor_position = 0
        state.generation_history.clear()
        state.generation_redo_stack.clear()
        state.global_conditioning = True
        state.conditioning_memory.clear()
        ds = self.query_one("#doc-selector", DocSelector)
        ds.loaded_doc_idx = doc_idx
        ds.refresh()

        tp = self.query_one("#text-panel", TextPanel)
        tp.selected_segment_idx = None
        tp.refresh()

        ss = self.query_one("#segment-selector", SegmentSelector)
        ss._segments = state.segments
        ss._selected_idx = 0
        ss.refresh()

        bar = self.query_one("#segment-bar", SegmentBar)
        bar.update_segments(len(doc.segments), doc.total_segments)
        bar.refresh()

        self._set_focus(FocusPanel.TEXT_PANEL)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def on_text_panel_randomize_seed_requested(self, msg: TextPanel.RandomizeSeedRequested) -> None:
        self._state.seed = msg.seed
        si = self.query_one("#seed-input", SeedInput)
        si.seed = msg.seed
        si.refresh()

    def on_text_panel_generate_requested(self, msg: TextPanel.GenerateRequested) -> None:
        if self._generation_manager is None or self._active_generation is not None:
            return

        if self._detail is not None:
            self._clear_live_feature_highlight(detail_message="suspended during generation")

        tp = self.query_one("#text-panel", TextPanel)
        tp.is_generating = True

        # Sync seed from seed input widget
        si = self.query_one("#seed-input", SeedInput)
        self._state.seed = si.seed

        # Re-generate: if the target segment is full, snapshot + clear it first.
        # Also handles prefilled segments (is_generation_target=False but
        # target_token_count set) — promote the last such segment to target.
        self._pre_regen_snapshot = None
        target = next(
            (s for s in self._state.segments if s.is_generation_target),
            None,
        )
        if target is None:
            # Prefill mode: find last segment with target_token_count and promote it
            for seg in reversed(self._state.segments):
                if seg.target_token_count is not None:
                    seg.is_generation_target = True
                    target = seg
                    break
        if (
            target is not None
            and target.target_token_count is not None
            and target.char_end > target.char_start
        ):
            seg_text = self._state.document_text[target.char_start : target.char_end]
            tokens_in_target = len(self._require_tokenizer().encode(seg_text)) if seg_text else 0
            if tokens_in_target >= target.target_token_count:
                self._pre_regen_snapshot = GenerationSnapshot(
                    document_text=self._state.document_text,
                    segments=self._state.segments,
                    generated_text="",
                    generated_token_ids=[],
                    seed=self._state.seed,
                )
                removed = target.char_end - target.char_start
                self._state.document_text = (
                    self._state.document_text[: target.char_start]
                    + self._state.document_text[target.char_end :]
                )
                target.char_end = target.char_start
                for seg in self._state.segments:
                    if seg is not target and seg.char_start > target.char_start:
                        seg.char_start -= removed
                        seg.char_end -= removed
                self._state.cursor_position = min(
                    self._state.cursor_position,
                    len(self._state.document_text),
                )
                tp.refresh()

        self._generated_tokens = []
        self._generated_text = ""

        self.run_worker(
            lambda: self._start_and_stream_generation(),
            thread=True,
            exclusive=True,
        )

    def _start_and_stream_generation(self) -> None:
        """Worker thread: build inputs (may trigger lazy model load), then stream."""
        generation_manager = self._generation_manager
        if generation_manager is None:
            return
        try:
            handle = generation_manager.start_generation(self._state)
            self._active_generation = handle
            self._stream_generation(handle)
        except Exception as exc:
            self.call_from_thread(self._reset_generation_ui, exc)
            raise

    def _reset_generation_ui(self, exc: Exception | None = None) -> None:
        """Main-thread callback: reset generation state after an error."""
        tp = self.query_one("#text-panel", TextPanel)
        tp.is_generating = False
        self._active_generation = None
        self._pre_regen_snapshot = None
        tp.refresh()
        self._refresh_live_feature_highlight_if_needed()
        if exc is not None:
            self.notify(f"Generation failed: {exc}", severity="error", timeout=8)

    def _stream_generation(self, handle: GenerationHandle) -> None:
        """Worker thread: iterate tokens, update UI via call_from_thread."""
        try:
            tokenizer = self._require_tokenizer()
            for token_id in handle.token_iterator:
                if handle.cancel_event.is_set():
                    break
                self._generated_tokens.append(token_id)
                text = tokenizer.decode([token_id])
                self._generated_text += text
                self.call_from_thread(self._apply_streamed_token, text, handle.target_segment_idx)
        finally:
            self.call_from_thread(self._finalize_generation, handle)

    def _apply_streamed_token(self, text: str, target_idx: int) -> None:
        """Main-thread callback: insert decoded token text into state."""
        target = self._state.segments[target_idx]
        insert_pos = target.char_end
        self._state.document_text = (
            self._state.document_text[:insert_pos] + text + self._state.document_text[insert_pos:]
        )
        target.char_end += len(text)
        for seg in self._state.segments[target_idx + 1 :]:
            seg.char_start += len(text)
            seg.char_end += len(text)
        self._state.cursor_position = target.char_end
        self.query_one("#text-panel", TextPanel).refresh()

    def _finalize_generation(self, handle: GenerationHandle) -> None:
        """Main-thread callback: push snapshot, clean up generation state."""
        if self._generated_text:
            # Use pre-regen snapshot (if re-generating a full segment) so that
            # Ctrl+Z restores the state before the re-generation, not the
            # intermediate cleared state.
            base = self._pre_regen_snapshot or handle.snapshot
            self._pre_regen_snapshot = None
            snapshot = GenerationSnapshot(
                document_text=base.document_text,
                segments=base.segments,
                generated_text=self._generated_text,
                generated_token_ids=self._generated_tokens,
                seed=base.seed,
            )
            self._state.generation_history.append(snapshot)
            self._state.generation_redo_stack.clear()

        tp = self.query_one("#text-panel", TextPanel)
        tp.is_generating = False
        self._active_generation = None
        tp.refresh()
        self._refresh_live_feature_highlight_if_needed()

    def on_text_panel_text_input_during_generation(
        self, msg: TextPanel.TextInputDuringGeneration
    ) -> None:
        if self._active_generation is not None and self._generation_manager is not None:
            self._generation_manager.cancel(self._active_generation)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def on_text_panel_undo_generation(self, msg: TextPanel.UndoGeneration) -> None:
        if undo_generation(self._state):
            self._refresh_after_state_change()

    def on_text_panel_redo_generation(self, msg: TextPanel.RedoGeneration) -> None:
        if redo_generation(self._state):
            self._refresh_after_state_change()

    def _refresh_after_state_change(self) -> None:
        self.query_one("#text-panel", TextPanel).refresh()
        ss = self.query_one("#segment-selector", SegmentSelector)
        ss._segments = self._state.segments
        ss.refresh()
        self._refresh_live_feature_highlight_if_needed()

    # ------------------------------------------------------------------
    # Live feature-token highlight
    # ------------------------------------------------------------------

    def _current_live_highlight_request(
        self,
        feature_id: int,
    ) -> FeatureHighlightRequest | None:
        if self._editing_segment_idx is None:
            return None
        if not (0 <= self._editing_segment_idx < len(self._state.segments)):
            return None
        requested_ids: int | tuple[int, ...] = feature_id
        if self._cluster_view is not None and self._cluster_view.is_composite(feature_id):
            requested_ids = self._cluster_view.rep_to_members[feature_id]
        return build_highlight_request(
            self._state.document_text,
            self._state.segments,
            self._editing_segment_idx,
            requested_ids,
        )

    def _clear_live_feature_highlight(
        self,
        *,
        detail_message: str | None = None,
    ) -> None:
        self._live_highlight_seq += 1
        self._live_highlight_request_key = None
        try:
            self.workers.cancel_group(self, "live-feature-highlight")
        except (AttributeError, RuntimeError):
            pass
        try:
            self.query_one("#text-panel", TextPanel).feature_highlight_result = None
        except NoMatches:
            pass
        if self._detail is not None:
            if detail_message is None:
                self._detail.clear_highlight_status()
            else:
                self._detail.set_highlight_status("stale", detail_message)

    def _request_live_feature_highlight(self, feature_id: int) -> None:
        if self._detail is None:
            return
        if self._live_highlight_service is None:
            self._clear_live_feature_highlight(detail_message="unavailable for this run")
            return
        live_highlight_service = self._live_highlight_service
        request = self._current_live_highlight_request(feature_id)
        if request is None:
            self._clear_live_feature_highlight(detail_message="no active segment")
            return

        self._live_highlight_seq += 1
        seq = self._live_highlight_seq
        self._live_highlight_request_key = request.request_key
        self.query_one("#text-panel", TextPanel).feature_highlight_result = None
        self._detail.set_highlight_status("loading", "computing token activations...")

        def _worker() -> FeatureHighlightResult:
            try:
                return live_highlight_service.compute(request)
            except Exception as exc:
                return FeatureHighlightResult(
                    request_key=request.request_key,
                    segment_idx=request.segments[request.target_segment_index].segment_idx,
                    feature_id=request.feature_id,
                    status="error",
                    message=str(exc),
                )

        def _apply_from_thread() -> None:
            result = _worker()
            self.post_message(LiveHighlightComputed(seq, request.request_key, result))

        self.run_worker(
            _apply_from_thread,
            group="live-feature-highlight",
            thread=True,
            exclusive=True,
        )

    def _request_live_cluster_evidence(self, panel: ClusterMembersPanel) -> None:
        """Refresh member evidence from the current segment with one probe pass."""
        if self._live_highlight_service is None or self._editing_segment_idx is None:
            return
        request = self._current_live_highlight_request(panel.representative_id)
        if request is None:
            return
        service = self._live_highlight_service
        self._live_highlight_seq += 1
        seq = self._live_highlight_seq
        self._live_highlight_request_key = request.request_key
        try:
            self.query_one("#text-panel", TextPanel).feature_highlight_result = None
        except NoMatches:
            pass

        def _worker() -> FeatureHighlightResult:
            try:
                return service.compute(request)
            except Exception as exc:
                return FeatureHighlightResult(
                    request_key=request.request_key,
                    segment_idx=request.segments[request.target_segment_index].segment_idx,
                    feature_id=request.feature_id,
                    status="error",
                    message=str(exc),
                )

        def _apply_from_thread() -> None:
            self.post_message(
                LiveClusterEvidenceComputed(seq, request.request_key, _worker())
            )

        self.run_worker(
            _apply_from_thread,
            group="live-feature-highlight",
            thread=True,
            exclusive=True,
        )

    def on_live_cluster_evidence_computed(
        self, message: LiveClusterEvidenceComputed
    ) -> None:
        if message.seq != self._live_highlight_seq:
            return
        if message.request_key != self._live_highlight_request_key:
            return
        panel = self._cluster_panel
        if panel is None or self._overlay_sub_panel != "cluster":
            return
        if message.result.status != "ready":
            return
        panel.update_evidence(dict(message.result.per_feature_max_percentile))
        try:
            self.query_one("#text-panel", TextPanel).feature_highlight_result = message.result
        except NoMatches:
            pass
        panel.refresh()

    def on_live_highlight_computed(self, message: LiveHighlightComputed) -> None:
        """Apply a worker result after thread-safe delivery to the app loop."""
        self._apply_live_feature_highlight_result(
            message.seq,
            message.request_key,
            message.result,
        )

    def _apply_live_feature_highlight_result(
        self,
        seq: int,
        request_key: str,
        result: FeatureHighlightResult,
    ) -> None:
        if seq != self._live_highlight_seq:
            return
        if request_key != self._live_highlight_request_key:
            return
        if self._detail is None or self._detail.feature_id != result.feature_id:
            return
        current = self._current_live_highlight_request(result.feature_id)
        if current is None or current.request_key != result.request_key:
            return

        tp = self.query_one("#text-panel", TextPanel)
        if result.status == "ready":
            tp.feature_highlight_result = result
            if result.max_percentile is None:
                message = f"{result.n_tokens} tokens, no positive activations"
            else:
                message = f"{result.n_tokens} tokens, max p{result.max_percentile:.1f}"
            self._detail.set_highlight_status("ready", message)
            return

        tp.feature_highlight_result = None
        message = result.message or result.status
        self._detail.set_highlight_status(result.status, message)

    def _refresh_live_feature_highlight_if_needed(self) -> None:
        if self._detail is not None:
            self._request_live_feature_highlight(self._detail.feature_id)
        elif self._cluster_panel is not None and self._overlay_sub_panel == "cluster":
            self._request_live_cluster_evidence(self._cluster_panel)

    def on_text_panel_text_changed(self, msg: TextPanel.TextChanged) -> None:
        if self._detail is not None:
            self._clear_live_feature_highlight(detail_message="waiting for retokenize")

    def on_text_panel_retokenize_completed(self, msg: TextPanel.RetokenizeCompleted) -> None:
        self._refresh_live_feature_highlight_if_needed()

    # ------------------------------------------------------------------
    # Segment highlight
    # ------------------------------------------------------------------

    def on_segment_selector_segment_highlighted(
        self, msg: SegmentSelector.SegmentHighlighted
    ) -> None:
        self.query_one("#text-panel", TextPanel).selected_segment_idx = msg.segment_idx

    def on_segment_selector_segment_unhighlighted(
        self, msg: SegmentSelector.SegmentUnhighlighted
    ) -> None:
        self.query_one("#text-panel", TextPanel).selected_segment_idx = None
        # Escape from SegmentSelector moves focus to TextPanel
        if self._focus_panel == FocusPanel.SEGMENT_SELECTOR:
            self._set_focus(FocusPanel.TEXT_PANEL)

    def on_segment_selector_enter_seed_input(self, msg: SegmentSelector.EnterSeedInput) -> None:
        self._set_focus(FocusPanel.SEED_INPUT)

    def on_seed_input_seed_input_exited(self, msg: SeedInput.SeedInputExited) -> None:
        # Sync seed back to state and segment selector
        si = self.query_one("#seed-input", SeedInput)
        self._state.seed = si.seed
        ss = self.query_one("#segment-selector", SegmentSelector)
        ss.seed = si.seed
        self._set_focus(FocusPanel.SEGMENT_SELECTOR)

    # ------------------------------------------------------------------
    # Overlay lifecycle
    # ------------------------------------------------------------------

    def _close_overlay(self) -> None:
        """Close the entire overlay, apply editor edits, restore SegmentSelector."""
        self._clear_live_feature_highlight()
        if self._detail is not None:
            self._detail.remove()
            self._detail = None
        if self._search is not None:
            self._search.remove()
            self._search = None
        if self._cluster_panel is not None:
            self._cluster_panel.remove()
            self._cluster_panel = None
        if self._editor is not None:
            # Apply feature edits back to the segment
            if self._editing_segment_idx is not None and self._editing_segment_idx < len(
                self._state.segments
            ):
                seg = self._state.segments[self._editing_segment_idx]
                seg.features = list(self._editor.sorted_features)
            self._editor.remove()
            self._editor = None
        self._editing_segment_idx = None
        self._overlay_open = False
        self._overlay_sub_panel = ""
        self._detail_return_panel = "editor"
        # Restore SegmentSelector visibility
        try:
            self.query_one("#segment-selector", SegmentSelector).display = True
        except NoMatches:
            pass

    # --- Feature Editor ---

    def on_segment_selector_open_feature_editor(
        self, msg: SegmentSelector.OpenFeatureEditor
    ) -> None:
        if msg.segment_idx >= len(self._state.segments):
            return
        self._close_overlay()
        seg = self._state.segments[msg.segment_idx]
        self._editing_segment_idx = msg.segment_idx

        # Seed _last_seg_features so R can resample without ctrl+R first.
        if self._doc_loader is not None:
            try:
                ds = self.query_one("#doc-selector", DocSelector)
                doc_idx = (
                    self._preview_doc_idx
                    if self._preview_doc_idx is not None
                    else ds.loaded_doc_idx
                )
            except NoMatches:
                doc_idx = None
            if doc_idx is not None:
                raw = self._doc_loader.raw_lookup(doc_idx)
                if raw and msg.segment_idx < len(raw):
                    self._last_seg_features = raw[msg.segment_idx]

        editor = FeatureEditor(
            seg,
            num_buckets=self._state.num_buckets,
            has_negatives=self._state.has_negatives,
            descriptions_available=self._description_service is not None,
            sampling_label=self._sampling_label(),
            neg_sampling_label=self._neg_sampling_label(),
            cluster_view=self._cluster_view,
            id="feature-editor",
        )
        editor.add_class("overlay-panel")
        self.mount(editor)
        self._editor = editor
        self._overlay_open = True
        self._overlay_sub_panel = "editor"

        self.query_one("#segment-selector", SegmentSelector).display = False
        self._set_focus(FocusPanel.OVERLAY)
        self._fetch_editor_descriptions()

    def _fetch_editor_descriptions(self) -> None:
        """Populate descriptions for the current feature editor.

        First pass (sync): fills any descriptions already in cache.
        Second pass (async): queues network requests for the rest,
        with visible features submitted first for priority.
        """
        if self._editor is None:
            return
        self._fetch_descriptions_into(lambda: self._editor)

    def _fetch_descriptions_into(
        self,
        get_target: Callable[[], FeatureEditor | ClusterMembersPanel | None],
    ) -> None:
        """Populate cached and async descriptions into an editor-like target."""
        if self._description_service is None:
            return
        target = get_target()
        if target is None:
            return

        svc = self._description_service
        target_is_editor = isinstance(target, FeatureEditor)
        if target_is_editor:
            feature_ids = [feature.feature_id for feature in target.sorted_features]
        else:
            feature_ids = list(target.members)

        # Sync pass: populate from cache without network
        missing_ids: list[int] = []
        for feature_id in feature_ids:
            desc = svc.get_description_sync(feature_id)
            if desc is not None:
                target.update_description(feature_id, desc)
            else:
                missing_ids.append(feature_id)

        if not missing_ids:
            target.refresh()
            return

        target.refresh()  # show cached descriptions immediately

        # Async pass: request missing descriptions, visible first
        height = target.size.height if target.size.height > 0 else 20
        scroll_height = max(0, height - getattr(target, "_HEADER_LINES", 2))
        visible_ids = set(target.visible_feature_ids(scroll_height))

        visible_first = [fid for fid in missing_ids if fid in visible_ids]
        rest = [fid for fid in missing_ids if fid not in visible_ids]

        def apply_description(feature_id: int, description: str) -> None:
            current_target = get_target()
            if current_target is None:
                return
            if target_is_editor:
                if not isinstance(current_target, FeatureEditor):
                    return
            elif not isinstance(current_target, ClusterMembersPanel):
                return
            current_target.update_description(feature_id, description)
            current_target.refresh()

        def on_start(feature_id: int) -> None:
            """Set '...' when the worker thread starts fetching this feature."""
            if threading.current_thread() is threading.main_thread():
                apply_description(feature_id, DESCRIPTION_LOADING)
            else:
                try:
                    self.call_from_thread(
                        apply_description,
                        feature_id,
                        DESCRIPTION_LOADING,
                    )
                except RuntimeError:
                    pass  # App exited before callback fired; discard silently

        def on_done(feature_id: int, description: str) -> None:
            if threading.current_thread() is threading.main_thread():
                apply_description(feature_id, description)
            else:
                try:
                    self.call_from_thread(
                        apply_description,
                        feature_id,
                        description,
                    )
                except RuntimeError:
                    pass  # App exited before callback fired; discard silently

        for fid in visible_first + rest:
            svc.request_description(fid, on_done, on_start=on_start)

    def on_feature_editor_feature_editor_closed(
        self, msg: FeatureEditor.FeatureEditorClosed
    ) -> None:
        self._close_overlay()
        self._set_focus(FocusPanel.SEGMENT_SELECTOR)

    def on_feature_editor_switch_segment_requested(
        self, msg: FeatureEditor.SwitchSegmentRequested
    ) -> None:
        """Switch the feature editor to an adjacent segment."""
        if self._editing_segment_idx is None:
            return
        n = len(self._state.segments)
        new_idx = (self._editing_segment_idx + msg.delta) % n
        # Apply current edits, close overlay, re-open for the new segment
        self._close_overlay()
        ss = self.query_one("#segment-selector", SegmentSelector)
        ss._selected_idx = new_idx
        ss.post_message(SegmentSelector.OpenFeatureEditor(new_idx))

    def _sampling_label(self) -> str | None:
        """Short label describing the active positive feature selection mode."""
        if self._sampling_config is None or self._sampling_config.feature_selection is None:
            return None
        sel = self._sampling_config.feature_selection
        if isinstance(sel, SamplingSelection):
            if sel.mode == SamplingMode.top_k_percentile:
                return f"sel: top-k-pct (k~{sel.mean:.0f})"
            return f"sel: weighted (k~{sel.mean:.0f})"
        return "sel: threshold"

    def _neg_sampling_label(self) -> str | None:
        """Short label describing the active negative feature sampling distribution."""
        if self._sampling_config is None or self._sampling_config.negative_selection is None:
            return None
        neg: NegativeSamplingSelection = self._sampling_config.negative_selection
        return f"neg: {neg.distribution.value}(k~{neg.mean:.0f})"

    def _features_from_seg_features(self, sf: SegmentFeatures) -> list[FeatureCondition]:
        """Convert raw SegmentFeatures to FeatureConditions, applying sampling if configured."""
        if self._sampling_config is None:
            return []
        if self._sampling_config.consolidation is not None:
            clustered = clustered_sample_segment_features(
                sf,
                self._sampling_config,
                self._state.num_buckets,
                self._state.has_negatives,
            )
            if self._editing_segment_idx is not None and 0 <= self._editing_segment_idx < len(
                self._state.segments
            ):
                segment_id = self._state.segments[self._editing_segment_idx].id
                self._segment_evidence[segment_id] = clustered.evidence
            return clustered.features
        return sample_segment_features(
            sf,
            self._sampling_config,
            self._state.num_buckets,
            self._state.has_negatives,
            feature_mask=self._sampling_config.feature_mask,
        )

    def on_feature_editor_randomize_features_requested(
        self, msg: FeatureEditor.RandomizeFeaturesRequested
    ) -> None:
        """Replace current editor features with a random segment from any document."""
        if self._editor is None or self._doc_loader is None:
            return

        doc_ids = self._doc_loader.available_doc_ids()
        if not doc_ids:
            return

        seg_features = None
        for _ in range(10):  # retry until we find a non-empty segment
            doc_id = random.choice(doc_ids)
            raw = self._doc_loader.raw_lookup(doc_id)
            if raw:
                seg_features = random.choice(raw)
                break

        if seg_features is None:
            return

        self._last_seg_features = seg_features
        features = self._features_from_seg_features(seg_features)
        self._editor.replace_features(features)
        self._fetch_editor_descriptions()
        self._editor.refresh()

    def on_feature_editor_resample_requested(self, msg: FeatureEditor.ResampleRequested) -> None:
        """Resample from the last loaded segment (new negatives + new k for stochastic selection)."""
        if self._editor is None or self._last_seg_features is None:
            return

        features = self._features_from_seg_features(self._last_seg_features)
        self._editor.replace_features(features)
        self._fetch_editor_descriptions()
        self._editor.refresh()

    # --- Cluster Members ---

    def on_feature_editor_open_cluster_members(self, msg: FeatureEditor.OpenClusterMembers) -> None:
        if (
            self._editor is None
            or self._cluster_view is None
            or self._editing_segment_idx is None
            or not (0 <= self._editing_segment_idx < len(self._state.segments))
        ):
            return
        members = self._cluster_view.rep_to_members.get(msg.rep_id)
        if members is None:
            return

        self._editor.display = False
        segment = self._state.segments[self._editing_segment_idx]
        cached = self._segment_evidence.get(segment.id)
        current_fingerprint = frozenset(
            feature.feature_id
            for feature in self._editor.sorted_features
            if self._cluster_view.is_composite(feature.feature_id)
        )
        evidence = (
            cached.evidence.get(msg.rep_id)
            if cached is not None and cached.fingerprint == current_fingerprint
            else None
        )
        panel = ClusterMembersPanel(
            msg.rep_id,
            members,
            evidence,
            descriptions_available=self._description_service is not None,
            id="cluster-members",
        )
        panel.add_class("overlay-panel")
        self.mount(panel)
        self._cluster_panel = panel
        self._overlay_sub_panel = "cluster"
        panel.focus()
        self._fetch_descriptions_into(lambda: self._cluster_panel)
        self._request_live_cluster_evidence(panel)

    def on_cluster_members_panel_cluster_members_closed(
        self, msg: ClusterMembersPanel.ClusterMembersClosed
    ) -> None:
        self._clear_live_feature_highlight()
        if self._cluster_panel is not None:
            self._cluster_panel.remove()
            self._cluster_panel = None
        if self._editor is not None:
            self._editor.display = True
        self._overlay_sub_panel = "editor"
        if self._editor is not None:
            self._editor.focus()

    # --- Feature Search ---

    def on_feature_editor_open_feature_search(self, msg: FeatureEditor.OpenFeatureSearch) -> None:
        if self._editor is not None:
            self._editor.display = False

        available: dict[int, str] = {}
        if self._description_service is not None:
            available = dict(self._description_service.get_all_cached())
        if self._feature_mask is not None and available:
            available_ids = list(available)
            keep = self._feature_mask.keep_mask(t.tensor(available_ids, dtype=t.int64))
            available = {
                feature_id: available[feature_id]
                for feature_id, is_allowed in zip(available_ids, keep.tolist(), strict=True)
                if is_allowed
            }
        if self._cluster_view is not None:
            for representative, members in self._cluster_view.rep_to_members.items():
                if representative in available:
                    available[representative] = (
                        f"[{len(members)} composite features] {available[representative]}"
                    )
        excluded: set[int] = set()
        if self._editor is not None:
            if self._cluster_view is None:
                excluded = {fc.feature_id for fc in self._editor.sorted_features}
            else:
                present_representatives = {
                    self._cluster_view.member_to_rep.get(feature.feature_id, feature.feature_id)
                    for feature in self._editor.sorted_features
                }
                excluded = set(present_representatives)
                for representative in present_representatives:
                    excluded.update(self._cluster_view.rep_to_members.get(representative, ()))

        search = FeatureSearch(available, excluded, id="feature-search")
        search.add_class("overlay-panel")
        self.mount(search)
        self._search = search
        self._overlay_sub_panel = "search"
        search.focus()

    def on_feature_search_feature_search_closed(
        self, msg: FeatureSearch.FeatureSearchClosed
    ) -> None:
        if self._search is not None:
            self._search.remove()
            self._search = None
        if self._editor is not None:
            self._editor.display = True
        self._overlay_sub_panel = "editor"
        if self._editor is not None:
            self._editor.focus()

    def on_feature_search_feature_added(self, msg: FeatureSearch.FeatureAdded) -> None:
        if self._editor is None:
            return
        target = (
            self._cluster_view.member_to_rep.get(msg.feature_id, msg.feature_id)
            if self._cluster_view is not None
            else msg.feature_id
        )
        if self._cluster_view is not None:
            for index, feature in enumerate(self._editor._features):
                if feature.feature_id == target:
                    self._editor._selected_idx = index
                    self._editor.refresh()
                    self.notify(
                        f"feature {msg.feature_id} is a member of cluster "
                        f"@{target} (already present)"
                    )
                    return
        desc = None
        if self._description_service is not None:
            desc = self._description_service.get_description_sync(target)
        fc = FeatureCondition(feature_id=target, magnitude=0, description=desc)
        self._editor._features.append(fc)
        self._editor._sort()
        # Move selection to the newly added feature
        for i, f in enumerate(self._editor._features):
            if f is fc:
                self._editor._selected_idx = i
                break
        self._editor.refresh()
        if target != msg.feature_id:
            self.notify(f"added cluster @{target} (feature {msg.feature_id} is a member)")

    # --- Feature Detail ---

    def on_feature_editor_open_feature_detail(self, msg: FeatureEditor.OpenFeatureDetail) -> None:
        if self._editor is not None:
            self._editor.display = False
        self._detail_return_panel = "editor"

        desc = None
        if self._description_service is not None:
            desc = self._description_service.get_description_sync(msg.feature_id)

        detail = FeatureDetail(msg.feature_id, description=desc, id="feature-detail")
        # If we already have a description, show it immediately instead of
        # staying stuck on "Loading..." forever.
        if desc is not None:
            detail.set_data({"explanations": [{"description": desc}]})
        detail.add_class("overlay-panel")
        self.mount(detail)
        self._detail = detail
        self._overlay_sub_panel = "detail"
        detail.focus()
        self._request_live_feature_highlight(msg.feature_id)

    def on_cluster_members_panel_open_member_detail(
        self, msg: ClusterMembersPanel.OpenMemberDetail
    ) -> None:
        if self._cluster_panel is None:
            return
        self._cluster_panel.display = False
        self._detail_return_panel = "cluster"
        desc = None
        if self._description_service is not None:
            desc = self._description_service.get_description_sync(msg.member_id)

        detail = FeatureDetail(msg.member_id, description=desc, id="feature-detail")
        # If we already have a description, show it immediately instead of
        # staying stuck on "Loading..." forever.
        if desc is not None:
            detail.set_data({"explanations": [{"description": desc}]})
        detail.add_class("overlay-panel")
        self.mount(detail)
        self._detail = detail
        self._overlay_sub_panel = "detail"
        detail.focus()
        self._request_live_feature_highlight(msg.member_id)

    def on_feature_detail_feature_detail_closed(
        self, msg: FeatureDetail.FeatureDetailClosed
    ) -> None:
        self._clear_live_feature_highlight()
        if self._detail is not None:
            self._detail.remove()
            self._detail = None
        return_panel = self._detail_return_panel
        self._detail_return_panel = "editor"
        if return_panel == "cluster" and self._cluster_panel is not None:
            self._cluster_panel.display = True
            self._overlay_sub_panel = "cluster"
            self._cluster_panel.focus()
        else:
            if self._editor is not None:
                self._editor.display = True
            self._overlay_sub_panel = "editor"
            if self._editor is not None:
                self._editor.focus()

    # ------------------------------------------------------------------
    # Segment split
    # ------------------------------------------------------------------

    def on_text_panel_segment_split_completed(self, msg: TextPanel.SegmentSplitCompleted) -> None:
        ss = self.query_one("#segment-selector", SegmentSelector)
        ss._segments = self._state.segments
        ss.refresh()
        self._refresh_live_feature_highlight_if_needed()
