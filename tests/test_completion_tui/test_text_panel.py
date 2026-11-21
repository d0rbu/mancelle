"""Tests for the TextPanel widget.

Covers rendering (per-segment coloring, generation marker, selection highlight),
editing (char insert, backspace, cursor movement), key→message dispatch, and
edge cases.
"""

from __future__ import annotations

import pytest
from rich.text import Text

from scripts.completion_tui.color_manager import (
    DEFAULT_COLOR,
    color_for_idx,
    feature_highlight_color,
)
from scripts.completion_tui.highlight_types import (
    FeatureHighlightResult,
    FeatureHighlightSpan,
)
from scripts.completion_tui.state import Segment, SessionState
from scripts.completion_tui.widgets.text_panel import TextPanel
from tests.test_completion_tui._key_dispatch import dispatch_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    text: str,
    segments: list[Segment] | None = None,
    cursor_position: int = 0,
) -> SessionState:
    if segments is None:
        segments = [
            Segment(
                segment_idx=0,
                char_start=0,
                char_end=len(text),
                color_idx=0,
                features=[],
            )
        ]
    return SessionState(
        document_text=text,
        segments=segments,
        num_buckets=11,
        has_negatives=True,
        cursor_position=cursor_position,
    )


def _two_seg_state(
    text: str = "hello world",
    split_at: int = 5,
    cursor_position: int = 0,
) -> SessionState:
    return _make_state(
        text,
        segments=[
            Segment(
                segment_idx=0,
                char_start=0,
                char_end=split_at,
                color_idx=0,
                features=[],
            ),
            Segment(
                segment_idx=1,
                char_start=split_at,
                char_end=len(text),
                color_idx=1,
                features=[],
            ),
        ],
        cursor_position=cursor_position,
    )


def _panel(
    state: SessionState,
    tokenizer: object,
    selected_segment_idx: int | None = None,
    is_generating: bool = False,
) -> TextPanel:
    return TextPanel(
        state,
        tokenizer,
        selected_segment_idx=selected_segment_idx,
        is_generating=is_generating,
    )


def _style_at(rendered: Text, pos: int) -> str:
    """Return the style string applied at character *pos*."""
    for span in rendered._spans:
        if span.start <= pos < span.end:
            return str(span.style)
    return ""


def _has_color(style_str: str, color_hex: str) -> bool:
    return color_hex in style_str


def test_token_aware_action_fails_loudly_without_tokenizer() -> None:
    panel = _panel(_make_state("hello"), None)

    with pytest.raises(TypeError, match=r"text panel requires tokenizer\.encode"):
        panel.handle_char_input("!")


# ---------------------------------------------------------------------------
# 1. Text renders with correct colors per segment
# ---------------------------------------------------------------------------


class TestRenderColors:
    def test_correct_colors_per_segment(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world")
        p = _panel(state, word_tokenizer)
        rendered = p.render()

        c0 = color_for_idx(0)
        c1 = color_for_idx(1)

        for i in range(5):
            assert _has_color(_style_at(rendered, i), c0), f"char {i} should be {c0}"
        for i in range(5, 11):
            assert _has_color(_style_at(rendered, i), c1), f"char {i} should be {c1}"


# ---------------------------------------------------------------------------
# 2. Typing a character extends the enclosing segment's color
# ---------------------------------------------------------------------------


class TestTypingExtendsSegment:
    def test_typed_char_has_segment_color(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", split_at=5, cursor_position=3)
        p = _panel(state, word_tokenizer)

        p.handle_char_input("x")

        assert state.document_text == "helxlo world"
        rendered = p.render()
        # The inserted 'x' at position 3 should carry segment-0's color
        assert _has_color(_style_at(rendered, 3), color_for_idx(0))


# ---------------------------------------------------------------------------
# 3. Typing at end of text extends last segment
# ---------------------------------------------------------------------------


class TestTypingAtEnd:
    def test_extends_last_segment(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", cursor_position=11)
        p = _panel(state, word_tokenizer)

        p.handle_char_input("!")

        assert state.document_text == "hello world!"
        assert state.segments[-1].char_end >= 12


# ---------------------------------------------------------------------------
# 4. Cursor movement (left/right) within and across segments
# ---------------------------------------------------------------------------


class TestCursorMovement:
    def test_move_right(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p.move_cursor(1)
        assert state.cursor_position == 1

    def test_move_left(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", cursor_position=5)
        p = _panel(state, word_tokenizer)
        p.move_cursor(-1)
        assert state.cursor_position == 4

    def test_clamp_at_zero(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p.move_cursor(-1)
        assert state.cursor_position == 0

    def test_clamp_at_end(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", cursor_position=11)
        p = _panel(state, word_tokenizer)
        p.move_cursor(1)
        assert state.cursor_position == 11

    def test_across_segments(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world", split_at=5, cursor_position=4)
        p = _panel(state, word_tokenizer)
        p.move_cursor(1)
        assert state.cursor_position == 5  # now at start of segment 1
        p.move_cursor(1)
        assert state.cursor_position == 6


# ---------------------------------------------------------------------------
# 5. Shift+Enter inserts newline, Enter posts GenerateRequested
# ---------------------------------------------------------------------------


class TestEnterKeys:
    def test_enter_inserts_newline(self, word_tokenizer) -> None:
        state = _make_state("hello", cursor_position=3)
        p = _panel(state, word_tokenizer)

        # Enter inserts a newline (via handle_char_input)
        p.handle_char_input("\n")

        assert "\n" in state.document_text
        assert state.cursor_position == 4

    def test_ctrl_g_posts_generate_requested(self, word_tokenizer) -> None:
        state = _make_state("hello")
        p = _panel(state, word_tokenizer)

        msgs = dispatch_key(p, "ctrl+g")
        generate_requests = [
            message for message in msgs if isinstance(message, TextPanel.GenerateRequested)
        ]
        assert len(generate_requests) == 1


# ---------------------------------------------------------------------------
# 6. Cmd+K (Ctrl+K) posts SegmentSplitCompleted
# ---------------------------------------------------------------------------


class TestSplitSegment:
    def test_ctrl_k_posts_split_completed(self, word_tokenizer) -> None:
        # Single segment covering "hello world", cursor at 5 (token boundary)
        state = _make_state(
            "hello world",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=11,
                    color_idx=0,
                    features=[],
                ),
            ],
            cursor_position=5,
        )
        p = _panel(state, word_tokenizer)

        msgs = dispatch_key(p, "ctrl+k")
        assert any(isinstance(m, TextPanel.SegmentSplitCompleted) for m in msgs)
        # The segment should have been split
        assert len(state.segments) == 2


# ---------------------------------------------------------------------------
# 7. Generation end marker at correct position, colored correctly
# ---------------------------------------------------------------------------


class TestGenerationMarker:
    def test_marker_at_end_colored_correctly(self, word_tokenizer) -> None:
        state = _make_state(
            "hello",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=2,
                    features=[],
                    is_generation_target=True,
                    target_token_count=8,
                ),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()

        assert "\u25bc" in rendered.plain  # ▼
        marker_idx = rendered.plain.index("\u25bc")
        target_color = color_for_idx(2)
        assert _has_color(_style_at(rendered, marker_idx), target_color)

    # -------------------------------------------------------------------
    # 8. Marker hidden when text extends past target
    # -------------------------------------------------------------------

    def test_marker_hidden_when_text_extends_past(self, word_tokenizer) -> None:
        state = _make_state(
            "hello world",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                    is_generation_target=True,
                ),
                Segment(
                    segment_idx=1,
                    char_start=5,
                    char_end=11,
                    color_idx=1,
                    features=[],
                ),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()

        assert "\u25bc" not in rendered.plain

    # -------------------------------------------------------------------
    # 9. No text label next to the marker
    # -------------------------------------------------------------------

    def test_no_label_next_to_marker(self, word_tokenizer) -> None:
        state = _make_state(
            "hello",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                    is_generation_target=True,
                    target_token_count=8,
                ),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()

        plain = rendered.plain
        marker_idx = plain.index("\u25bc")
        # Before marker: text, cursor space, and filler dots — no text label
        before = plain[:marker_idx]
        text_part = before.replace("\u00b7", "").rstrip()
        assert text_part == "hello"
        # After marker: nothing meaningful
        after = plain[marker_idx + 1 :]
        assert after.strip() == ""


# ---------------------------------------------------------------------------
# 10. Selected segment highlighting (others go white/default)
# ---------------------------------------------------------------------------


class TestSelectedSegmentHighlight:
    def test_non_selected_segments_use_default_color(self, word_tokenizer) -> None:
        state = _two_seg_state("hello world")
        p = _panel(state, word_tokenizer, selected_segment_idx=1)
        rendered = p.render()

        # Segment 0 should be DEFAULT_COLOR (not selected)
        for i in range(5):
            assert _has_color(_style_at(rendered, i), DEFAULT_COLOR), (
                f"char {i} should be default color"
            )

        # Segment 1 should keep its real color
        c1 = color_for_idx(1)
        for i in range(5, 11):
            assert _has_color(_style_at(rendered, i), c1), f"char {i} should be {c1}"


# ---------------------------------------------------------------------------
# 10b. Live feature highlight overlay
# ---------------------------------------------------------------------------


class TestFeatureHighlightOverlay:
    def test_target_segment_uses_highlight_levels_and_grays_inactive_chars(
        self, word_tokenizer
    ) -> None:
        state = _two_seg_state("hello world")
        p = _panel(state, word_tokenizer)
        p.feature_highlight_result = FeatureHighlightResult(
            request_key="k",
            segment_idx=0,
            feature_id=100,
            status="ready",
            spans=(
                FeatureHighlightSpan(
                    char_start=1,
                    char_end=3,
                    level=1,
                    percentile=91.0,
                    activation=1.0,
                ),
                FeatureHighlightSpan(
                    char_start=4,
                    char_end=5,
                    level=3,
                    percentile=99.5,
                    activation=5.0,
                ),
            ),
            max_percentile=99.5,
            max_activation=5.0,
            n_tokens=5,
        )

        rendered = p.render()
        base = color_for_idx(0)

        assert _has_color(_style_at(rendered, 0), DEFAULT_COLOR)
        assert _has_color(_style_at(rendered, 1), feature_highlight_color(base, 1))
        assert _has_color(_style_at(rendered, 2), feature_highlight_color(base, 1))
        assert _has_color(_style_at(rendered, 3), DEFAULT_COLOR)
        assert _has_color(_style_at(rendered, 4), base)
        assert _has_color(_style_at(rendered, 5), color_for_idx(1))

    def test_highlight_overlay_overrides_non_selected_gray_for_target_segment(
        self, word_tokenizer
    ) -> None:
        state = _two_seg_state("hello world")
        p = _panel(state, word_tokenizer, selected_segment_idx=1)
        p.feature_highlight_result = FeatureHighlightResult(
            request_key="k",
            segment_idx=0,
            feature_id=100,
            status="ready",
            spans=(
                FeatureHighlightSpan(
                    char_start=0,
                    char_end=5,
                    level=3,
                    percentile=99.5,
                    activation=5.0,
                ),
            ),
            max_percentile=99.5,
            max_activation=5.0,
            n_tokens=1,
        )

        rendered = p.render()

        assert _has_color(_style_at(rendered, 0), color_for_idx(0))


# ---------------------------------------------------------------------------
# 11. Backspace at position 0 is a no-op
# ---------------------------------------------------------------------------


class TestBackspaceAtZero:
    def test_noop(self, word_tokenizer) -> None:
        state = _make_state("hello", cursor_position=0)
        p = _panel(state, word_tokenizer)

        p.handle_backspace()

        assert state.document_text == "hello"
        assert state.cursor_position == 0


# ---------------------------------------------------------------------------
# Additional: typing during generation blocks edits
# ---------------------------------------------------------------------------


class TestGenerationGuard:
    def test_char_input_blocked_during_generation(self, word_tokenizer) -> None:
        state = _make_state("hello", cursor_position=3)
        p = _panel(state, word_tokenizer, is_generating=True)

        p.handle_char_input("x")

        assert state.document_text == "hello"
        assert state.cursor_position == 3


# ---------------------------------------------------------------------------
# Additional: delete key
# ---------------------------------------------------------------------------


class TestDeleteKey:
    def test_delete_removes_char_at_cursor(self, word_tokenizer) -> None:
        state = _make_state("hello", cursor_position=2)
        p = _panel(state, word_tokenizer)

        p.handle_delete_key()

        assert state.document_text == "helo"
        assert state.cursor_position == 2

    def test_delete_at_end_is_noop(self, word_tokenizer) -> None:
        state = _make_state("hello", cursor_position=5)
        p = _panel(state, word_tokenizer)

        p.handle_delete_key()

        assert state.document_text == "hello"


# ---------------------------------------------------------------------------
# Additional: undo/redo message dispatch
# ---------------------------------------------------------------------------


class TestUndoRedoMessages:
    def test_ctrl_z_posts_undo(self, word_tokenizer) -> None:
        state = _make_state("hello")
        p = _panel(state, word_tokenizer)

        msgs = dispatch_key(p, "ctrl+z")
        assert len(msgs) == 1
        assert isinstance(msgs[0], TextPanel.UndoGeneration)

    def test_ctrl_shift_z_posts_redo(self, word_tokenizer) -> None:
        state = _make_state("hello")
        p = _panel(state, word_tokenizer)

        msgs = dispatch_key(p, "ctrl+shift+z")
        assert len(msgs) == 1
        assert isinstance(msgs[0], TextPanel.RedoGeneration)


# ---------------------------------------------------------------------------
# Word-level and line-level cursor movement
# ---------------------------------------------------------------------------


class TestWordMovement:
    def test_option_right_jumps_to_next_word(self, word_tokenizer) -> None:
        state = _make_state("foo bar baz", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p._key_option_right()
        assert state.cursor_position == 4  # after "foo "

    def test_option_right_at_end_stays(self, word_tokenizer) -> None:
        state = _make_state("foo bar", cursor_position=7)
        p = _panel(state, word_tokenizer)
        p._key_option_right()
        assert state.cursor_position == 7

    def test_option_left_jumps_to_prev_word(self, word_tokenizer) -> None:
        state = _make_state("foo bar baz", cursor_position=8)
        p = _panel(state, word_tokenizer)
        p._key_option_left()
        assert state.cursor_position == 4  # start of "bar"

    def test_option_left_at_start_stays(self, word_tokenizer) -> None:
        state = _make_state("foo bar", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p._key_option_left()
        assert state.cursor_position == 0

    def test_option_right_with_underscores(self, word_tokenizer) -> None:
        state = _make_state("foo_bar_baz", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p._key_option_right()
        assert state.cursor_position == 4  # after "foo_"

    def test_option_right_with_slashes(self, word_tokenizer) -> None:
        state = _make_state("path/to/file", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p._key_option_right()
        assert state.cursor_position == 5  # after "path/"


class TestLineMovement:
    def test_cmd_right_goes_to_line_end(self, word_tokenizer) -> None:
        state = _make_state("hello\nworld", cursor_position=2)
        p = _panel(state, word_tokenizer)
        p._key_cmd_right()
        assert state.cursor_position == 5  # before \n

    def test_cmd_left_goes_to_line_start(self, word_tokenizer) -> None:
        state = _make_state("hello\nworld", cursor_position=8)
        p = _panel(state, word_tokenizer)
        p._key_cmd_left()
        assert state.cursor_position == 6  # after \n

    def test_cmd_right_single_line(self, word_tokenizer) -> None:
        state = _make_state("hello world", cursor_position=3)
        p = _panel(state, word_tokenizer)
        p._key_cmd_right()
        assert state.cursor_position == 11  # end of text

    def test_cmd_left_at_start_stays(self, word_tokenizer) -> None:
        state = _make_state("hello world", cursor_position=0)
        p = _panel(state, word_tokenizer)
        p._key_cmd_left()
        assert state.cursor_position == 0


# ---------------------------------------------------------------------------
# Marker does not shift when cursor is at end
# ---------------------------------------------------------------------------


class TestMarkerCursorConsistency:
    def test_cursor_at_end_does_not_push_marker(self, word_tokenizer) -> None:
        """When cursor is at end of text, marker gap should be same width."""
        state_mid = _make_state(
            "hello",
            cursor_position=2,
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                    is_generation_target=True,
                    target_token_count=8,
                ),
            ],
        )
        state_end = _make_state(
            "hello",
            cursor_position=5,
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                    is_generation_target=True,
                    target_token_count=8,
                ),
            ],
        )
        p_mid = _panel(state_mid, word_tokenizer)
        p_end = _panel(state_end, word_tokenizer)
        r_mid = p_mid.render().plain
        r_end = p_end.render().plain

        # Cursor at end replaces the first filler space, so the marker
        # stays at the same position regardless of cursor location.
        mid_marker = r_mid.index("\u25bc")
        end_marker = r_end.index("\u25bc")
        assert end_marker == mid_marker


class TestMarkerShrinksDuringGeneration:
    """Marker dots+triangle should shrink as the generation target fills in."""

    def _gen_seg(self, char_start: int, char_end: int, token_count: int = 8) -> Segment:
        return Segment(
            segment_idx=1,
            char_start=char_start,
            char_end=char_end,
            color_idx=1,
            features=[],
            is_generation_target=True,
            target_token_count=token_count,
        )

    def test_zero_width_target_shows_full_marker(self, word_tokenizer) -> None:
        """Before generation starts, marker is at full estimated size."""
        state = _make_state(
            "hello",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                ),
                self._gen_seg(5, 5, token_count=8),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()
        # Full estimated = 8*4 = 32 chars → 16 dots
        assert "\u25bc" in rendered.plain
        n_dots = rendered.plain.count("\u00b7")
        assert n_dots > 0

    def test_partially_filled_target_has_fewer_dots(self, word_tokenizer) -> None:
        """As text fills into the generation target, marker shrinks."""
        # Zero-width: full marker (budget=4 tokens, 0 filled)
        state_empty = _make_state(
            "hello",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                ),
                self._gen_seg(5, 5, token_count=4),
            ],
        )
        # Partially filled: "world" = 1 token filled, 3 remaining
        state_partial = _make_state(
            "helloworld",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                ),
                self._gen_seg(5, 10, token_count=4),
            ],
        )

        p_empty = _panel(state_empty, word_tokenizer)
        p_partial = _panel(state_partial, word_tokenizer)
        dots_empty = p_empty.render().plain.count("\u00b7")
        dots_partial = p_partial.render().plain.count("\u00b7")

        assert dots_empty > dots_partial > 0

    def test_fully_filled_target_hides_marker(self, word_tokenizer) -> None:
        """When all budgeted tokens generated, marker disappears."""
        # "world" = 1 token, budget = 1 → 0 remaining → no marker
        state = _make_state(
            "helloworld",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                ),
                self._gen_seg(5, 10, token_count=1),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()
        assert "\u25bc" not in rendered.plain

    def test_overfilled_target_hides_marker(self, word_tokenizer) -> None:
        """When tokens exceed budget, marker stays hidden."""
        # "the quick" → ["the", " ", "quick"] = 3 tokens, budget = 2
        state = _make_state(
            "hellothe quick",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                ),
                self._gen_seg(5, 14, token_count=2),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()
        assert "\u25bc" not in rendered.plain

    def test_no_target_token_count_hides_marker(self, word_tokenizer) -> None:
        """No target_token_count → no marker regardless of segment width."""
        state = _make_state(
            "hello",
            segments=[
                Segment(
                    segment_idx=0,
                    char_start=0,
                    char_end=5,
                    color_idx=0,
                    features=[],
                    is_generation_target=True,
                    target_token_count=None,
                ),
            ],
        )
        p = _panel(state, word_tokenizer)
        rendered = p.render()
        assert "\u25bc" not in rendered.plain
