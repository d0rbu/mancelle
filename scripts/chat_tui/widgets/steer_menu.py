"""Two-pane segment and dense-signal steering modal."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from enum import Enum, auto
from typing import Literal, Protocol

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.events import Key
from textual.geometry import Region, Spacing
from textual.widget import Widget
from textual.widgets import Static

from scripts.chat_tui.session import (
    DiscardPreview,
    LedgerSegment,
    SegmentProvenance,
)
from scripts.chat_tui.steering import (
    STRENGTH_BUDGET,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)
from scripts.completion_tui.color_manager import color_for_idx

SignalState = Literal["unfitted", "fitting", "cached"]

STRENGTH_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)

# The chat contract pins the encoder to two buckets (0 = active, 1 = the
# reserved inactive bucket); the inactive one is DISPLAYED as -1 while the
# stored value and the 0/1 edit keys keep the real index.
_INACTIVE_BUCKET = 1


def _bucket_display(bucket: int) -> str:
    return "-1" if bucket == _INACTIVE_BUCKET else str(bucket)


class SignalsSource(Protocol):
    """Lazy fitted-signal capability consumed by ``SteerMenu``."""

    def list_concepts(self) -> tuple[tuple[int, str], ...]: ...

    def state_of(self, concept_id: int) -> SignalState: ...

    def ensure_fitted(
        self,
        concept_id: int,
        on_ready: Callable[[SteeringSignal], None],
        on_error: Callable[[str], None],
    ) -> None: ...

    def signal_of(self, concept_id: int) -> SteeringSignal: ...


class DescriptionSource(Protocol):
    def request_description(
        self,
        feature_id: int,
        callback: Callable[[int, str], None],
        /,
        *,
        on_start: Callable[[int], None] | None = None,
    ) -> None: ...

    def shutdown(self) -> None: ...


class _DiscardPreview(Protocol):
    def __call__(
        self,
        entry: LedgerSegment,
        *,
        is_modify: bool = False,
    ) -> DiscardPreview: ...


class SteerPane(Enum):
    SEGMENTS = auto()
    SIGNALS = auto()


class _Mode(Enum):
    BROWSE = auto()
    EDIT_FEATURES = auto()
    ADD_FEATURE = auto()
    FILTER_SIGNALS = auto()
    CONFIRM = auto()


class _ConfirmAction(Enum):
    REWIND = auto()
    MODIFY = auto()


class SteerMenu(Widget):
    """Inspect/rewrite ledger segments and configure dense steering signals."""

    can_focus = True

    DEFAULT_CSS = """
    SteerMenu {
        width: 92%;
        height: 85%;
        margin: 2 4;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    SteerMenu #steer-title {
        height: 1;
        text-style: bold;
    }

    SteerMenu #steer-scroll {
        height: 1fr;
        /* Keep a blank row between the last list row and the help line, so a
           highlight scrolled to the bottom is not flush against it. */
        margin-bottom: 1;
    }

    SteerMenu #steer-body {
        height: auto;
        /* The scroll maths treat a row index as a line index. A wrapped row
           would break that and drift the cursor further out of view the deeper
           into the list it got, so rows are clipped instead of wrapped. */
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    SteerMenu #steer-help {
        height: 1;
        color: $text-muted;
    }

    SteerMenu #steer-status {
        height: auto;
        color: $error;
    }
    """

    def __init__(
        self,
        session_view: Callable[[], tuple[LedgerSegment, ...]],
        discard_preview: _DiscardPreview,
        signals_source: SignalsSource,
        on_rewind: Callable[[LedgerSegment], None],
        on_modify: Callable[[LedgerSegment, dict[int, int]], None],
        on_set_steering: Callable[[SteeringSelection], None],
        on_close: Callable[[], None],
        description_service: DescriptionSource | None,
        budget: float = STRENGTH_BUDGET,
        initial_steering: tuple[SteeringMethod, tuple[tuple[int, float], ...]] | None = None,
        is_current: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("budget must be finite and positive")

        self._session_view = session_view
        self._is_current = is_current
        self._discard_preview = discard_preview
        self._signals_source = signals_source
        self._on_rewind = on_rewind
        self._on_modify = on_modify
        self._on_set_steering = on_set_steering
        self._on_close = on_close
        self._description_service = description_service
        self._budget = float(budget)

        self.pane = SteerPane.SEGMENTS
        self._mode = _Mode.BROWSE
        self._ledger: tuple[LedgerSegment, ...] = ()
        self._segment_highlight = 0
        self._edited_features: dict[int, int] | None = None
        self._feature_highlight = 0
        self._add_digits = ""
        self._confirm_action: _ConfirmAction | None = None
        self._confirm_return_mode = _Mode.BROWSE

        self._concepts = signals_source.list_concepts()
        self._signal_highlight = 0
        self._signal_filter = ""
        self._enabled: set[int] = set()
        self._ready_signals: dict[int, SteeringSignal] = {}
        self.strengths = {concept_id: 0.0 for concept_id, _name in self._concepts}
        self._method = SteeringMethod.OFF
        if initial_steering is not None:
            # Reflect the session's ACTIVE steering so reopening the menu never
            # misreports it as off (and an untouched re-apply never clears it).
            method, pairs = initial_steering
            self._method = method
            for concept_id, strength in pairs:
                if concept_id not in self.strengths:
                    raise ValueError(f"active steering references unknown concept {concept_id}")
                self._enabled.add(concept_id)
                self.strengths[concept_id] = strength
        self._pending: str | None = None
        self._status_message: str | None = None

        self._descriptions: dict[int, str] = {}
        self._description_pending: set[int] = set()

    def compose(self) -> ComposeResult:
        yield Static("Steer · segments", id="steer-title", markup=False)
        # A bare Static clips rather than scrolls (its virtual size is clamped
        # to its box), so long lists need a real scroll container around it.
        # It must not take focus, or it swallows the menu's own key handling.
        pane = VerticalScroll(id="steer-scroll")
        pane.can_focus = False
        with pane:
            yield Static(Text(), id="steer-body", markup=False)
        yield Static("", id="steer-help", markup=False)
        yield Static("", id="steer-status", markup=False)

    def on_mount(self) -> None:
        self.focus()
        self.call_after_refresh(self._load_session_view)

    def _load_session_view(self) -> None:
        self._ledger = self._session_view()
        self._segment_highlight = min(
            self._segment_highlight,
            max(0, len(self._ledger) - 1),
        )
        self._render_menu()

    def on_key(self, event: Key) -> None:
        handled = self._handle_key(event.key)
        if handled:
            event.prevent_default()
            event.stop()

    def _handle_key(self, key: str) -> bool:
        # Session mutations are short and serialized. Blocking every key,
        # including Escape, keeps the modal state stable until the app reports
        # the terminal outcome; the failure path below re-enables interaction.
        if self._pending is not None:
            return True
        if key == "escape":
            self._escape()
            return True
        if self._mode is _Mode.CONFIRM:
            return self._handle_confirm_key(key)
        if self._mode is _Mode.EDIT_FEATURES:
            return self._handle_editor_key(key)
        if self._mode is _Mode.ADD_FEATURE:
            return self._handle_add_key(key)
        if self._mode is _Mode.FILTER_SIGNALS:
            return self._handle_filter_key(key)
        if key == "tab":
            self.pane = SteerPane.SIGNALS if self.pane is SteerPane.SEGMENTS else SteerPane.SEGMENTS
            self._render_menu()
            return True
        if self.pane is SteerPane.SEGMENTS:
            return self._handle_segments_key(key)
        return self._handle_signals_key(key)

    def _handle_segments_key(self, key: str) -> bool:
        if key == "up":
            self._move_segment(-1)
        elif key == "down":
            self._move_segment(1)
        elif key == "enter":
            self._begin_confirm(_ConfirmAction.REWIND)
        elif key == "e":
            self._begin_editor()
        else:
            return False
        return True

    def _handle_editor_key(self, key: str) -> bool:
        features = self._edited_features
        if features is None:
            return False
        ids = sorted(features)
        if key == "up":
            self._feature_highlight = max(0, self._feature_highlight - 1)
        elif key == "down":
            self._feature_highlight = min(max(0, len(ids) - 1), self._feature_highlight + 1)
        elif key in {"0", "1"} and ids:
            features[ids[self._feature_highlight]] = int(key)
        elif key in {"backspace", "delete"} and ids:
            del features[ids[self._feature_highlight]]
            self._feature_highlight = min(
                self._feature_highlight,
                max(0, len(features) - 1),
            )
        elif key == "a":
            self._mode = _Mode.ADD_FEATURE
            self._add_digits = ""
        elif key == "enter":
            self._begin_confirm(_ConfirmAction.MODIFY)
        else:
            return False
        self._render_menu()
        return True

    def _handle_add_key(self, key: str) -> bool:
        if len(key) == 1 and key.isdigit():
            self._add_digits += key
        elif key == "backspace":
            self._add_digits = self._add_digits[:-1]
        elif key == "enter":
            if self._add_digits:
                feature_id = int(self._add_digits)
                assert self._edited_features is not None
                self._edited_features[feature_id] = 0
                ids = sorted(self._edited_features)
                self._feature_highlight = ids.index(feature_id)
            self._mode = _Mode.EDIT_FEATURES
            self._add_digits = ""
        else:
            return False
        self._render_menu()
        return True

    def _handle_confirm_key(self, key: str) -> bool:
        if key == "n":
            self._mode = self._confirm_return_mode
            self._confirm_action = None
            self._render_menu()
            return True
        if key != "y":
            return False
        entry = self._highlighted_entry()
        try:
            if self._confirm_action is _ConfirmAction.REWIND:
                self._pending = "rewind"
                self._status_message = None
                self._on_rewind(entry)
            elif self._confirm_action is _ConfirmAction.MODIFY:
                assert self._edited_features is not None
                self._pending = "modify"
                self._status_message = None
                self._on_modify(entry, dict(self._edited_features))
            else:
                raise RuntimeError("confirmation has no action")
        except Exception as exc:
            self.on_done(False, str(exc))
        else:
            self._render_menu()
        return True

    def _handle_signals_key(self, key: str) -> bool:
        if key == "m":
            methods = (
                SteeringMethod.OFF,
                SteeringMethod.INTERPOLATION,
                SteeringMethod.RESTRICTION,
            )
            self._method = methods[(methods.index(self._method) + 1) % len(methods)]
        elif key == "up":
            self._move_signal(-1)
        elif key == "down":
            self._move_signal(1)
        elif key == "space":
            self._toggle_signal()
        elif key == "left":
            self._step_strength(-1)
        elif key == "right":
            self._step_strength(1)
        elif key in {"/", "slash"}:
            self._mode = _Mode.FILTER_SIGNALS
            self._signal_filter = ""
            self._signal_highlight = 0
        elif len(key) == 1 and key.isdigit():
            self._signal_filter += key
            self._signal_highlight = 0
        elif key == "backspace" and self._signal_filter:
            self._signal_filter = self._signal_filter[:-1]
            self._signal_highlight = 0
        elif key == "enter":
            self._apply_steering()
        else:
            return False
        self._render_menu()
        return True

    def _handle_filter_key(self, key: str) -> bool:
        if key in {"/", "slash"}:
            self._signal_filter = ""
            self._signal_highlight = 0
        elif key == "enter":
            self._mode = _Mode.BROWSE
        elif key == "backspace":
            self._signal_filter = self._signal_filter[:-1]
            self._signal_highlight = 0
        elif len(key) == 1 and key.isprintable():
            self._signal_filter += key
            self._signal_highlight = 0
        else:
            return False
        self._render_menu()
        return True

    def _move_segment(self, delta: int) -> None:
        if self._ledger:
            self._segment_highlight = min(
                max(self._segment_highlight + delta, 0), len(self._ledger) - 1
            )
        self._render_menu()

    def _move_signal(self, delta: int) -> None:
        visible = self._visible_concepts()
        if visible:
            self._signal_highlight = min(max(self._signal_highlight + delta, 0), len(visible) - 1)
        self._render_menu()

    def _begin_editor(self) -> None:
        if not self._ledger:
            return
        entry = self._highlighted_entry()
        try:
            # The session decides editability (turn Chat real history is
            # read-only); ask before entering the editor, not at confirmation.
            self._discard_preview(entry, is_modify=True)
        except ValueError as exc:
            self._status_message = str(exc)
            self._render_menu()
            return
        self._status_message = None
        self._edited_features = dict(entry.features)
        self._feature_highlight = 0
        self._mode = _Mode.EDIT_FEATURES
        self._render_menu()

    def _begin_confirm(self, action: _ConfirmAction) -> None:
        if not self._ledger:
            return
        self._confirm_return_mode = (
            _Mode.EDIT_FEATURES if action is _ConfirmAction.MODIFY else _Mode.BROWSE
        )
        self._confirm_action = action
        self._mode = _Mode.CONFIRM
        self._render_menu()

    def _highlighted_entry(self) -> LedgerSegment:
        if not self._ledger:
            raise RuntimeError("the segment ledger is empty")
        return self._ledger[self._segment_highlight]

    def _toggle_signal(self) -> None:
        visible = self._visible_concepts()
        if not visible:
            return
        concept_id, _name = visible[self._signal_highlight]
        if concept_id in self._enabled:
            self._enabled.remove(concept_id)
            return
        self._enabled.add(concept_id)
        if self._method is SteeringMethod.OFF:
            # Enabling a concept is an unambiguous request to steer. Leaving the
            # method at its OFF default would build the zero policy, so the
            # apply would succeed while changing nothing.
            self._method = SteeringMethod.INTERPOLATION
        state = self._signals_source.state_of(concept_id)
        if state == "cached":
            self._ready_signals[concept_id] = self._signals_source.signal_of(concept_id)
            return

        def on_ready(signal: SteeringSignal) -> None:
            self._marshal(self._fit_ready, concept_id, signal)

        def on_error(detail: str) -> None:
            self._marshal(self._fit_error, concept_id, detail)

        self._signals_source.ensure_fitted(concept_id, on_ready, on_error)

    def _fit_ready(self, concept_id: int, signal: SteeringSignal) -> None:
        if self._is_current is not None and not self._is_current():
            return
        self._ready_signals[concept_id] = signal
        if self.is_mounted:
            self._render_menu()

    def _fit_error(self, concept_id: int, detail: str) -> None:
        if self._is_current is not None and not self._is_current():
            return
        self._enabled.discard(concept_id)
        self._status_message = detail
        if self.is_mounted:
            self.notify(detail, severity="error", timeout=8, markup=False)
            self._render_menu()

    def _step_strength(self, delta: int) -> None:
        visible = self._visible_concepts()
        if not visible:
            return
        concept_id, _name = visible[self._signal_highlight]
        current = self.strengths[concept_id]
        index = STRENGTH_GRID.index(current)
        index = min(max(index + delta, 0), len(STRENGTH_GRID) - 1)
        self.strengths[concept_id] = STRENGTH_GRID[index]

    def _apply_steering(self) -> None:
        if self._pending is not None:
            return
        enabled: list[tuple[SteeringSignal, float]] = []
        for concept_id, _name in self._concepts:
            if concept_id not in self._enabled:
                continue
            signal = self._ready_signals.get(concept_id)
            if signal is None and self._signals_source.state_of(concept_id) == "cached":
                signal = self._signals_source.signal_of(concept_id)
                self._ready_signals[concept_id] = signal
            if signal is None:
                self.notify(
                    "enabled steering signals are still fitting",
                    severity="warning",
                    markup=False,
                )
                return
            enabled.append((signal, self.strengths[concept_id]))
        selection = SteeringSelection(self._method, tuple(enabled))
        self._pending = "steering"
        self._status_message = None
        try:
            self._on_set_steering(selection)
        except Exception as exc:
            self.on_done(False, str(exc))
        else:
            self._render_menu()

    def enabled_state(self) -> tuple[SteeringMethod, tuple[tuple[int, float], ...]]:
        """Method plus (concept_id, strength) pairs for the enabled signals."""

        return (
            self._method,
            tuple((concept_id, self.strengths[concept_id]) for concept_id in sorted(self._enabled)),
        )

    def on_done(self, success: bool, message: str | None = None) -> None:
        """Complete the pending app-owned session mutation."""

        self._pending = None
        if success:
            self._status_message = None
            self._on_close()
        else:
            self._status_message = message or "operation failed"
        if not success and self.is_mounted:
            self._render_menu()

    def _visible_concepts(self) -> list[tuple[int, str]]:
        query = self._signal_filter.casefold()
        if not query:
            return list(self._concepts)
        if query.isdigit():
            return [item for item in self._concepts if str(item[0]).startswith(query)]
        return [item for item in self._concepts if query in item[1].casefold()]

    def _escape(self) -> None:
        if self._mode in {_Mode.EDIT_FEATURES, _Mode.ADD_FEATURE, _Mode.CONFIRM}:
            if (
                self._mode is _Mode.CONFIRM and self._confirm_return_mode is _Mode.EDIT_FEATURES
            ) or self._mode is _Mode.ADD_FEATURE:
                self._mode = _Mode.EDIT_FEATURES
            else:
                self._mode = _Mode.BROWSE
                self._edited_features = None
            self._confirm_action = None
            self._render_menu()
            return
        self._on_close()

    def _render_menu(self) -> None:
        if not self.is_mounted:
            return
        if self._mode is _Mode.CONFIRM:
            self._render_confirm()
        elif self._mode in {_Mode.EDIT_FEATURES, _Mode.ADD_FEATURE}:
            self._render_editor()
        elif self.pane is SteerPane.SEGMENTS:
            self._render_segments()
        else:
            self._render_signals()
        status = self._status_message
        if status is None and self._pending is not None:
            status = f"{self._pending}…"
        self.query_one("#steer-status", Static).update(status or "")

    def _render_segments(self) -> None:
        self.query_one("#steer-title", Static).update("Steer · segments")
        rendered = Text()
        glyphs = {
            SegmentProvenance.REAL: "R",
            SegmentProvenance.PREDICTED: "P",
            SegmentProvenance.STEERED: "S",
            SegmentProvenance.OVERRIDDEN: "O",
        }
        for index, entry in enumerate(self._ledger):
            style = "reverse" if index == self._segment_highlight else None
            prefix = "> " if index == self._segment_highlight else "  "
            rendered.append(prefix, style=style)
            rendered.append("█", Style(color=color_for_idx(entry.color_idx), reverse=bool(style)))
            line = (
                f" [{entry.token_start},{entry.token_end}) · {glyphs[entry.provenance]} · "
                f"{len(entry.features)} features"
            )
            if entry.steering is not None:
                line += f" · {entry.steering.method.value} Σf={entry.steering.total_strength:.1f}"
            rendered.append(line, style=style)
            if index != len(self._ledger) - 1:
                rendered.append("\n")
        if not self._ledger:
            rendered.append("no segments")
        self.query_one("#steer-body", Static).update(rendered)
        self._scroll_highlight_into_view(self._segment_highlight)
        self.query_one("#steer-help", Static).update(
            "Tab signals · ↑/↓ select · Enter rewind · e edit · Esc close"
        )

    def _render_editor(self) -> None:
        entry = self._highlighted_entry()
        self.query_one("#steer-title", Static).update(
            f"Steer · edit segment [{entry.token_start},{entry.token_end})"
        )
        rendered = Text()
        features = self._edited_features or {}
        ids = sorted(features)
        for feature_id in ids:
            self._request_description(feature_id)
        for index, feature_id in enumerate(ids):
            selected = index == self._feature_highlight
            style = "reverse" if selected else None
            prefix = "> " if selected else "  "
            line = f"{prefix}{feature_id}  {_bucket_display(features[feature_id])}"
            description = self._descriptions.get(feature_id)
            if description:
                line += f"  {description}"
            elif feature_id in self._description_pending:
                line += "  …"
            rendered.append(line, style=style)
            if index != len(ids) - 1:
                rendered.append("\n")
        if not ids:
            rendered.append("no features", style="dim")
        if self._mode is _Mode.ADD_FEATURE:
            rendered.append(f"\n\nfeature id: {self._add_digits}", style="bold")
        self.query_one("#steer-body", Static).update(rendered)
        self._scroll_highlight_into_view(self._feature_highlight)
        self.query_one("#steer-help", Static).update(
            "↑/↓ select · 0 active · 1 inactive (-1) · Del remove · a add · Enter apply · Esc back"
        )

    def _request_description(self, feature_id: int) -> None:
        service = self._description_service
        if (
            service is None
            or feature_id in self._descriptions
            or feature_id in self._description_pending
        ):
            return
        self._description_pending.add(feature_id)

        def on_done(returned_id: int, description: str) -> None:
            self._marshal(self._description_ready, returned_id, description)

        service.request_description(feature_id, on_done)

    def _description_ready(self, feature_id: int, description: str) -> None:
        self._description_pending.discard(feature_id)
        self._descriptions[feature_id] = description
        if self.is_mounted and self._mode in {_Mode.EDIT_FEATURES, _Mode.ADD_FEATURE}:
            self._render_editor()

    def _render_confirm(self) -> None:
        entry = self._highlighted_entry()
        preview = self._discard_preview(
            entry,
            is_modify=self._confirm_action is _ConfirmAction.MODIFY,
        )
        self.query_one("#steer-title", Static).update(f"Confirm {preview.operation}")
        self.query_one("#steer-body", Static).update(
            Text(
                f"Discard {preview.discarded_tokens} tokens and "
                f"{preview.discarded_segments} segments "
                "after target? y/n",
                style="bold yellow",
            )
        )
        self.query_one("#steer-help", Static).update("y confirm · n/Esc back")

    def _render_signals(self) -> None:
        self.query_one("#steer-title", Static).update(
            f"Steer · signals · method: {self._method.value}"
        )
        rendered = Text()
        visible = self._visible_concepts()
        self._signal_highlight = min(
            self._signal_highlight,
            max(0, len(visible) - 1),
        )
        for row, (concept_id, name) in enumerate(visible):
            selected = row == self._signal_highlight
            enabled = concept_id in self._enabled
            style = "reverse" if selected else None
            prefix = "> " if selected else "  "
            mark = "●" if enabled else "○"
            state = (
                "cached"
                if concept_id in self._ready_signals
                else self._signals_source.state_of(concept_id)
            )
            if state == "fitting":
                state_label = "fitting…"
            elif (
                enabled
                and self._method is SteeringMethod.RESTRICTION
                and self.strengths[concept_id] == 0.0
            ):
                state_label = "sign-only"
            else:
                state_label = state
            rendered.append(
                f"{prefix}{mark} {concept_id} · "
                f"{self.strengths[concept_id]:.1f} · {state_label} · {name}",
                style=style,
            )
            if row != len(visible) - 1:
                rendered.append("\n")
        if not visible:
            rendered.append("no concepts match the filter", style="dim")

        total = math.fsum(self.strengths[concept_id] for concept_id in self._enabled)
        rendered.append("\n\n")
        budget_style = "bold red" if total > self._budget else "bold"
        rendered.append(f"Σf {total:.1f}/{self._budget:.1f}", style=budget_style)
        if self._signal_filter or self._mode is _Mode.FILTER_SIGNALS:
            rendered.append(f" · filter: {self._signal_filter}", style="dim")
        if self._pending == "steering":
            rendered.append(" · applying…", style="bold")
        if self._method is SteeringMethod.OFF and self._enabled:
            rendered.append(
                f"\nmethod is off — the {len(self._enabled)} enabled signal(s) will be "
                "IGNORED; press m to pick a method",
                style="bold red",
            )
        self.query_one("#steer-body", Static).update(rendered)
        self._scroll_highlight_into_view(self._signal_highlight)
        self.query_one("#steer-help", Static).update(
            "Tab segments · m method · Space enable · ←/→ strength · / filter · Enter apply"
        )

    def _scroll_highlight_into_view(self, line: int) -> None:
        """Keep the highlighted row on screen once a list outgrows the pane.

        Row index doubles as line index: every list here renders exactly one
        line per row.
        """

        if not self.is_mounted:
            return
        pane = self.query_one("#steer-scroll", VerticalScroll)
        if pane.size.height <= 0:
            return
        # Delegate the offset maths to Textual rather than reimplementing it:
        # it accounts for scrollbars and clamping. The spacing keeps a row
        # visible on each side of the cursor, so a highlight at either end is
        # never flush against the pane edge.
        pane.scroll_to_region(
            Region(0, line, 1, 1),
            spacing=Spacing(1, 0, 1, 0),
            animate=False,
            force=True,
        )

    def _marshal(self, callback: Callable[..., None], *args: object) -> None:
        if threading.current_thread() is threading.main_thread():
            callback(*args)
            return
        try:
            self.app.call_from_thread(callback, *args)
        except RuntimeError:
            pass
