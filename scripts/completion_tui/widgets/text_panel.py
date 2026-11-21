"""Text panel widget for the completion TUI.

Scrollable, editable text area that renders document text with per-segment
coloring, a generation end marker, and debounced retokenization.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget

from scripts.completion_tui.color_manager import (
    DEFAULT_COLOR,
    color_for_idx,
    feature_highlight_color,
)
from scripts.completion_tui.highlight_types import FeatureHighlightResult
from scripts.completion_tui.segment_ops import (
    on_text_delete,
    on_text_insert,
    retokenize_boundaries,
    split_segment,
)
from scripts.completion_tui.state import Segment, SessionState
from steering.tokenizer_protocols import TextTokenizer, require_text_tokenizer

_RETOKENIZE_DEBOUNCE_S = 0.05  # 50 ms

# Word boundary pattern: split on space, underscore, or slash
_WORD_BOUNDARY = re.compile(r"[ _/]")


class TextPanel(Widget):
    """Editable text area with per-segment coloring."""

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class GenerateRequested(Message):
        """User pressed Enter — request autoregressive generation."""

    class SegmentSplitCompleted(Message):
        """A segment was successfully split at the cursor."""

    class UndoGeneration(Message):
        """User pressed Ctrl+Z — undo last generation."""

    class RedoGeneration(Message):
        """User pressed Ctrl+Shift+Z — redo generation."""

    class TextInputDuringGeneration(Message):
        """User tried to type while generation is active."""

    class TextChanged(Message):
        """User edited document text; highlight overlays are stale."""

    class RetokenizeCompleted(Message):
        """Debounced retokenization completed after a text edit."""

    class RandomizeSeedRequested(Message):
        """User pressed Ctrl+R — request a new random seed."""

        def __init__(self, seed: int) -> None:
            super().__init__()
            self.seed = seed

    # ------------------------------------------------------------------
    # Widget setup
    # ------------------------------------------------------------------

    can_focus = True

    DEFAULT_CSS = """
    TextPanel {
        overflow-y: auto;
        overflow-x: hidden;
        height: 1fr;
    }
    """

    def __init__(
        self,
        state: SessionState,
        tokenizer: object | None,
        *,
        selected_segment_idx: int | None = None,
        is_generating: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._tokenizer: TextTokenizer | None = (
            None
            if tokenizer is None
            else require_text_tokenizer(
                tokenizer,
                surface="completion TUI text panel",
            )
        )
        self._selected_segment_idx = selected_segment_idx
        self._is_generating = is_generating
        self._dimmed = False
        self._bright_segment_indices: set[int] | None = None
        self._feature_highlight_result: FeatureHighlightResult | None = None
        self._retokenize_timer: Timer | None = None
        self._desired_col: int | None = None  # sticky column for up/down

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        return self._state

    def _require_tokenizer(self) -> TextTokenizer:
        """Return the validated tokenizer needed by editing operations."""
        if self._tokenizer is None:
            return require_text_tokenizer(
                self._tokenizer,
                surface="completion TUI text panel",
            )
        return self._tokenizer

    @property
    def selected_segment_idx(self) -> int | None:
        return self._selected_segment_idx

    @selected_segment_idx.setter
    def selected_segment_idx(self, value: int | None) -> None:
        self._selected_segment_idx = value
        self.refresh()

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    @is_generating.setter
    def is_generating(self, value: bool) -> None:
        self._is_generating = value

    @property
    def feature_highlight_result(self) -> FeatureHighlightResult | None:
        return self._feature_highlight_result

    @feature_highlight_result.setter
    def feature_highlight_result(self, value: FeatureHighlightResult | None) -> None:
        self._feature_highlight_result = value
        self.refresh()

    # ------------------------------------------------------------------
    # Segment lookup
    # ------------------------------------------------------------------

    def _segment_at(self, pos: int) -> Segment | None:
        """Return the segment containing *pos*, or ``None``."""
        for seg in self._state.segments:
            if seg.char_start <= pos < seg.char_end:
                return seg
        return None

    def _segment_index_at(self, pos: int) -> int | None:
        """Return the list index of the segment containing *pos*."""
        for i, seg in enumerate(self._state.segments):
            if seg.char_start <= pos < seg.char_end:
                return i
        return None

    def _generation_target(self) -> Segment | None:
        for seg in self._state.segments:
            if seg.is_generation_target:
                return seg
        return None

    # ------------------------------------------------------------------
    # Color logic
    # ------------------------------------------------------------------

    @property
    def dimmed(self) -> bool:
        return self._dimmed

    @dimmed.setter
    def dimmed(self, value: bool) -> None:
        self._dimmed = value

    @property
    def bright_segment_indices(self) -> set[int] | None:
        return self._bright_segment_indices

    @bright_segment_indices.setter
    def bright_segment_indices(self, value: set[int] | None) -> None:
        self._bright_segment_indices = value

    def _color_at(self, pos: int) -> str:
        """Return the hex color (with optional dim) for the character at *pos*."""
        seg = self._segment_at(pos)
        highlight_level = self._highlight_level_at(pos, seg)
        if highlight_level is not None and seg is not None:
            color = feature_highlight_color(color_for_idx(seg.color_idx), highlight_level)
        else:
            if seg is None or (
                self._selected_segment_idx is not None
                and seg.segment_idx != self._selected_segment_idx
            ):
                color = DEFAULT_COLOR
            else:
                color = color_for_idx(seg.color_idx)
        if self._dimmed:
            if (
                self._bright_segment_indices is not None
                and seg is not None
                and seg.segment_idx in self._bright_segment_indices
            ):
                return color
            return f"{color} dim"
        return color

    def _highlight_level_at(self, pos: int, seg: Segment | None) -> int | None:
        """Return live feature highlight level for ``pos`` or ``None``."""
        result = self._feature_highlight_result
        if result is None or result.status != "ready" or seg is None:
            return None
        if seg.segment_idx != result.segment_idx:
            return None
        level = 0
        for span in result.spans:
            if span.char_start <= pos < span.char_end:
                level = max(level, span.level)
        return level

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def handle_char_input(self, char: str) -> None:
        """Insert *char* at the cursor, update segments, advance cursor."""
        if self._is_generating:
            return
        on_text_insert(
            self._state,
            self._state.cursor_position,
            char,
            self._require_tokenizer(),
        )
        self._state.cursor_position += len(char)
        self._desired_col = None

    def handle_backspace(self) -> None:
        """Delete the character before the cursor."""
        if self._state.cursor_position == 0:
            return
        on_text_delete(
            self._state,
            self._state.cursor_position - 1,
            1,
            self._require_tokenizer(),
        )
        self._state.cursor_position -= 1
        self._desired_col = None

    def handle_delete_key(self) -> None:
        """Delete the character at the cursor."""
        if self._state.cursor_position >= len(self._state.document_text):
            return
        on_text_delete(
            self._state,
            self._state.cursor_position,
            1,
            self._require_tokenizer(),
        )
        self._desired_col = None

    def move_cursor(self, delta: int) -> None:
        """Move the cursor by *delta*, clamping to ``[0, len(text)]``."""
        new_pos = self._state.cursor_position + delta
        self._state.cursor_position = max(0, min(len(self._state.document_text), new_pos))
        self._desired_col = None

    def move_cursor_to(self, pos: int) -> None:
        """Set the cursor to *pos*, clamping to ``[0, len(text)]``."""
        self._state.cursor_position = max(0, min(len(self._state.document_text), pos))
        self._desired_col = None

    # ------------------------------------------------------------------
    # Word-level movement
    # ------------------------------------------------------------------

    def _find_word_boundary_right(self) -> int:
        """Find the next word boundary to the right of the cursor."""
        text = self._state.document_text
        pos = self._state.cursor_position
        if pos >= len(text):
            return pos
        # Skip current non-boundary characters, then skip the boundary char
        m = _WORD_BOUNDARY.search(text, pos)
        if m is None:
            return len(text)
        return m.end()  # position after the boundary character

    def _find_word_boundary_left(self) -> int:
        """Find the previous word boundary to the left of the cursor."""
        text = self._state.document_text
        pos = self._state.cursor_position
        if pos <= 0:
            return 0
        # Search backwards: find the last boundary before pos
        i = pos - 1
        # Skip the boundary char we might be sitting on
        while i > 0 and _WORD_BOUNDARY.match(text[i - 1]):
            i -= 1
        # Now find the next boundary going left
        while i > 0 and not _WORD_BOUNDARY.match(text[i - 1]):
            i -= 1
        return i

    # ------------------------------------------------------------------
    # Line-level movement
    # ------------------------------------------------------------------

    def _find_line_start(self) -> int:
        """Find the start of the current line (after previous newline)."""
        text = self._state.document_text
        pos = self._state.cursor_position
        idx = text.rfind("\n", 0, pos)
        return idx + 1 if idx >= 0 else 0

    def _find_line_end(self) -> int:
        """Find the end of the current line (before next newline or end)."""
        text = self._state.document_text
        pos = self._state.cursor_position
        idx = text.find("\n", pos)
        return idx if idx >= 0 else len(text)

    # ------------------------------------------------------------------
    # Debounced retokenization
    # ------------------------------------------------------------------

    def _schedule_retokenize(self) -> None:
        """Schedule a debounced retokenize (~50 ms). Only works when mounted."""
        if self._retokenize_timer is not None:
            self._retokenize_timer.stop()
        self._retokenize_timer = self.set_timer(_RETOKENIZE_DEBOUNCE_S, self._fire_retokenize)

    def _fire_retokenize(self) -> None:
        retokenize_boundaries(
            self._state.document_text,
            self._state.segments,
            self._require_tokenizer(),
        )
        self._retokenize_timer = None
        self._post_retokenize_completed()
        self.refresh()

    def force_retokenize(self) -> None:
        """Cancel any pending debounce and retokenize immediately."""
        if self._retokenize_timer is not None:
            self._retokenize_timer.stop()
            self._retokenize_timer = None
        retokenize_boundaries(
            self._state.document_text,
            self._state.segments,
            self._require_tokenizer(),
        )
        self._post_retokenize_completed()

    def _post_text_changed(self) -> None:
        try:
            self.post_message(self.TextChanged())
        except RuntimeError:
            pass

    def _post_retokenize_completed(self) -> None:
        try:
            self.post_message(self.RetokenizeCompleted())
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _estimate_remaining_marker_chars(self, gen_target: Segment) -> int | None:
        """Estimated REMAINING character length for the generation target marker.

        Counts tokens already in the segment, subtracts from
        ``target_token_count``, and converts remaining tokens to an
        approximate character width (4 chars per token) for the marker.
        Returns 0 when all tokens have been generated.
        """
        tc = gen_target.target_token_count
        if tc is None or tc <= 0:
            return None
        seg_text = self._state.document_text[gen_target.char_start : gen_target.char_end]
        tokens_filled = len(self._require_tokenizer().encode(seg_text)) if seg_text else 0
        tokens_remaining = max(0, tc - tokens_filled)
        return tokens_remaining * 4

    def _visual_col(self, pos: int) -> int:
        """Column position within the current visual line (after last newline)."""
        text = self._state.document_text
        last_nl = text.rfind("\n", 0, pos)
        return pos - (last_nl + 1)

    def _line_up_pos(self, col: int) -> int:
        """Cursor position after moving up one line, targeting column *col*."""
        text = self._state.document_text
        pos = self._state.cursor_position
        line_start = text.rfind("\n", 0, pos)
        if line_start < 0:
            return pos  # already on first line
        prev_line_start_nl = text.rfind("\n", 0, line_start)
        prev_start = prev_line_start_nl + 1 if prev_line_start_nl >= 0 else 0
        prev_len = line_start - prev_start
        return prev_start + min(col, prev_len)

    def _line_down_pos(self, col: int) -> int:
        """Cursor position after moving down one line, targeting column *col*."""
        text = self._state.document_text
        pos = self._state.cursor_position
        next_nl = text.find("\n", pos)
        if next_nl < 0:
            return pos  # already on last line
        next_start = next_nl + 1
        next_nl2 = text.find("\n", next_start)
        next_len = (next_nl2 if next_nl2 >= 0 else len(text)) - next_start
        return next_start + min(col, next_len)

    def render(self) -> Text:
        result = Text()
        doc_text = self._state.document_text
        cursor_pos = self._state.cursor_position
        text_len = len(doc_text)

        gen_target = self._generation_target()
        seg_char_len: int | None = None
        if gen_target is not None:
            seg_char_len = self._estimate_remaining_marker_chars(gen_target)

        # Show marker when there is remaining estimated space to fill
        show_marker = gen_target is not None and seg_char_len is not None and seg_char_len > 0

        # Render document text with colors and cursor
        focused = self.has_focus
        for i, ch in enumerate(doc_text):
            color = self._color_at(i)
            style = color
            if focused and i == cursor_pos:
                if ch == "\n":
                    # Newline is invisible — show a cursor block before it
                    result.append(" ", style=f"{color} reverse")
                    result.append(ch, style=style)
                    continue
                style += " reverse"
            result.append(ch, style=style)

        # Generation end marker with filler dots.
        # Filler length = segment_char_len // 2 copies of " ·", with the
        # triangle replacing the last dot if the length is even.
        # Rich handles wrapping naturally via the spaces.
        if show_marker and gen_target is not None and seg_char_len is not None:
            target_color = color_for_idx(gen_target.color_idx)
            cursor_at_end = cursor_pos == text_len

            # Dim the marker when the panel is dimmed and the gen target
            # segment is not in the bright set.
            marker_dim = self._dimmed and (
                self._bright_segment_indices is None
                or gen_target.segment_idx not in self._bright_segment_indices
            )
            triangle_style = f"{target_color} dim" if marker_dim else target_color

            # Filler: seg_char_len // 2 copies of " ·", triangle at end.
            # If even length, triangle replaces the last dot.
            # When cursor is at end, it replaces the first filler space
            # so the total width stays the same.
            n_dots = seg_char_len // 2
            is_even = seg_char_len % 2 == 0

            if n_dots > 0:
                # First " ·" — cursor replaces the space if at end
                if cursor_at_end:
                    result.append(" ", style="reverse")
                else:
                    result.append(" ", style=f"{target_color} dim")
                result.append("\u00b7", style=f"{target_color} dim")

                end = n_dots - 1 if is_even else n_dots
                for _j in range(1, end):
                    result.append(" \u00b7", style=f"{target_color} dim")
                if is_even:
                    result.append(" ", style=f"{target_color} dim")
                    result.append("\u25bc", style=triangle_style)
                else:
                    result.append("\u25bc", style=triangle_style)
            else:
                if focused and cursor_at_end:
                    result.append("\u25bc", style=f"{target_color} reverse")
                    return result
                result.append("\u25bc", style=triangle_style)
        elif focused and cursor_pos >= text_len:
            result.append(" ", style="reverse")

        return result

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        key = event.key

        handler = self._KEY_MAP.get(key)
        if handler is not None:
            handler(self)
            self.refresh()
            event.prevent_default()
            event.stop()
            return

        # Printable character input
        if event.character and event.is_printable:
            if self._is_generating:
                self.post_message(self.TextInputDuringGeneration())
            else:
                self.handle_char_input(event.character)
                self._post_text_changed()
                self._schedule_retokenize()
            self.refresh()
            event.prevent_default()
            event.stop()

    # --- key handlers ---

    def _key_enter(self) -> None:
        """Insert a newline. Enter is the default editing key."""
        if self._is_generating:
            self.post_message(self.TextInputDuringGeneration())
            return
        self.handle_char_input("\n")
        self._post_text_changed()
        self._schedule_retokenize()

    def _key_generate(self) -> None:
        """Trigger generation. Bound to Ctrl+G."""
        self.force_retokenize()
        self.post_message(self.GenerateRequested())

    def _key_ctrl_k(self) -> None:
        seg_i = self._segment_index_at(self._state.cursor_position)
        if seg_i is not None:
            ok = split_segment(
                self._state,
                seg_i,
                self._state.cursor_position,
                self._require_tokenizer(),
            )
            if ok:
                self.post_message(self.SegmentSplitCompleted())

    def _key_ctrl_r(self) -> None:
        self.post_message(self.RandomizeSeedRequested(self._state.seed + 1))

    def _key_ctrl_z(self) -> None:
        self.post_message(self.UndoGeneration())

    def _key_ctrl_shift_z(self) -> None:
        self.post_message(self.RedoGeneration())

    def _key_left(self) -> None:
        self.move_cursor(-1)

    def _key_right(self) -> None:
        self.move_cursor(1)

    def _key_option_left(self) -> None:
        """Move cursor to previous word boundary (space/underscore/slash)."""
        self.move_cursor_to(self._find_word_boundary_left())

    def _key_option_right(self) -> None:
        """Move cursor to next word boundary (space/underscore/slash)."""
        self.move_cursor_to(self._find_word_boundary_right())

    def _key_cmd_left(self) -> None:
        """Move cursor to start of current line (Home)."""
        self.move_cursor_to(self._find_line_start())

    def _key_cmd_right(self) -> None:
        """Move cursor to end of current line (End)."""
        self.move_cursor_to(self._find_line_end())

    def _key_up(self) -> None:
        """Move cursor up one line, preserving sticky column."""
        if self._desired_col is None:
            self._desired_col = self._visual_col(self._state.cursor_position)
        new_pos = self._line_up_pos(self._desired_col)
        self._state.cursor_position = max(0, min(len(self._state.document_text), new_pos))
        # DON'T clear _desired_col — preserve across multiple up/down moves

    def _key_down(self) -> None:
        """Move cursor down one line, preserving sticky column."""
        if self._desired_col is None:
            self._desired_col = self._visual_col(self._state.cursor_position)
        new_pos = self._line_down_pos(self._desired_col)
        self._state.cursor_position = max(0, min(len(self._state.document_text), new_pos))

    def _key_home(self) -> None:
        self.move_cursor_to(self._find_line_start())

    def _key_end(self) -> None:
        self.move_cursor_to(self._find_line_end())

    def _key_backspace(self) -> None:
        if self._is_generating:
            self.post_message(self.TextInputDuringGeneration())
            return
        before = self._state.document_text
        self.handle_backspace()
        if self._state.document_text != before:
            self._post_text_changed()
            self._schedule_retokenize()

    def _key_option_backspace(self) -> None:
        """Delete the word before the cursor (back to previous word boundary)."""
        if self._is_generating:
            self.post_message(self.TextInputDuringGeneration())
            return
        pos = self._state.cursor_position
        target = self._find_word_boundary_left()
        if target < pos:
            on_text_delete(
                self._state,
                target,
                pos - target,
                self._require_tokenizer(),
            )
            self._state.cursor_position = target
            self._desired_col = None
            self._post_text_changed()
            self._schedule_retokenize()

    def _key_cmd_backspace(self) -> None:
        """Delete from cursor to start of line."""
        if self._is_generating:
            self.post_message(self.TextInputDuringGeneration())
            return
        pos = self._state.cursor_position
        line_start = self._find_line_start()
        if line_start < pos:
            on_text_delete(
                self._state,
                line_start,
                pos - line_start,
                self._require_tokenizer(),
            )
            self._state.cursor_position = line_start
            self._desired_col = None
            self._post_text_changed()
            self._schedule_retokenize()

    def _key_delete(self) -> None:
        if self._is_generating:
            self.post_message(self.TextInputDuringGeneration())
            return
        before = self._state.document_text
        self.handle_delete_key()
        if self._state.document_text != before:
            self._post_text_changed()
            self._schedule_retokenize()

    # Textual key names vary by terminal. Over SSH from macOS (Ghostty):
    #   Cmd+Left → ctrl+a, Cmd+Right → ctrl+e  (line start/end)
    #   Option+B/F → ctrl+left, ctrl+right      (word movement)
    #   Ctrl+Enter / Shift+Enter → no distinct event (use ctrl+o)
    _KEY_MAP: dict[str, Callable[[TextPanel], None]] = {
        "enter": _key_enter,  # newline (common editing action)
        "ctrl+g": _key_generate,  # trigger generation
        "ctrl+r": _key_ctrl_r,  # randomize seed
        "ctrl+k": _key_ctrl_k,
        "ctrl+z": _key_ctrl_z,
        "ctrl+shift+z": _key_ctrl_shift_z,
        "left": _key_left,
        "right": _key_right,
        "up": _key_up,
        "down": _key_down,
        "ctrl+left": _key_option_left,  # Option+B on Ghostty (word back)
        "ctrl+right": _key_option_right,  # Option+F on Ghostty (word forward)
        "alt+left": _key_option_left,  # option+left (terminals that support it)
        "alt+right": _key_option_right,
        "alt+b": _key_option_left,  # readline word-back
        "alt+f": _key_option_right,  # readline word-forward
        "ctrl+a": _key_cmd_left,  # Cmd+Left on Ghostty (line start)
        "ctrl+e": _key_cmd_right,  # Cmd+Right on Ghostty (line end)
        "home": _key_home,
        "end": _key_end,
        "backspace": _key_backspace,
        "ctrl+backspace": _key_option_backspace,  # Option+Backspace on Ghostty
        "ctrl+w": _key_option_backspace,  # readline word-delete
        "alt+backspace": _key_option_backspace,  # terminals that support it
        "ctrl+u": _key_cmd_backspace,  # Cmd+Backspace on Ghostty
        "delete": _key_delete,
    }


class TextPanelHelp(Widget):
    """Read-only sidebar showing TextPanel key bindings."""

    _INSTRUCTIONS: list[str] = [
        "^G  generate",
        "^Z  undo",
        "^⇧Z redo",
        "^K  split seg",
        "^R  seed+1",
        "^C  copy",
        "^T  cond on/off",
        "⌥↓↑ next panel",
    ]

    DEFAULT_CSS = """
    TextPanelHelp {
        dock: right;
        width: 16;
        height: 1fr;
    }
    """

    def render(self) -> Text:
        result = Text()
        for line in self._INSTRUCTIONS:
            result.append(line + "\n", style="dim")
        return result
