"""Textual application shell for continuous self-conditioned chat."""

from __future__ import annotations

import json
import math
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from loguru import logger
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widget import Widget
from textual.widgets import Input, Static

from scripts.chat_tui.convo import ConvoCatalog
from scripts.chat_tui.render import (
    build_special_display,
    pretty_view,
    stream_display_text,
)
from scripts.chat_tui.session import (
    ChatSession,
    GenerationEvent,
    GenerationEventKind,
    LedgerSegment,
    SegmentKind,
    TranscriptSpan,
)
from scripts.chat_tui.steering import SteeringMethod, SteeringSelection
from scripts.chat_tui.turn_session import TurnChatSession
from scripts.chat_tui.widgets.convo_picker import ConvoPicker
from scripts.chat_tui.widgets.steer_menu import DescriptionSource, SignalsSource, SteerMenu
from scripts.completion_tui.color_manager import color_for_idx

ModelJob = Callable[[], None]
ModalFactory = Callable[[], Widget]
_STOP = object()


class ModelThread:
    """One long-lived queue consumer that serializes all session/model work."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[ModelJob | object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="chat-model",
            daemon=True,
        )
        self._started = False
        self._shutdown = False

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        if self._shutdown:
            raise RuntimeError("cannot start a shut down model thread")
        if self._started:
            return
        self._started = True
        self._thread.start()

    def submit(self, job: ModelJob) -> None:
        if not self._started or self._shutdown:
            raise RuntimeError("model thread must be running before submit")
        self._jobs.put(job)

    def shutdown(self) -> None:
        if not self._started or self._shutdown:
            return
        self._shutdown = True
        self._jobs.put(_STOP)
        self._thread.join()

    def _run(self) -> None:
        while True:
            queued = self._jobs.get()
            try:
                if queued is _STOP:
                    return
                job = cast(ModelJob, queued)
                try:
                    job()
                except Exception:
                    # Session jobs surface their own errors to the UI. This is the
                    # final containment boundary that keeps the queue consumer alive.
                    logger.exception("chat model-thread job leaked an exception")
            finally:
                self._jobs.task_done()


class ErrorModal(Static):
    """Fail-loud model-operation detail that remains until Escape."""

    can_focus = True

    def __init__(self, detail: str) -> None:
        super().__init__(f"{detail}\n\nesc to dismiss", markup=False)
        self.detail = detail
        self.add_class("chat-modal", "chat-error")

    def on_mount(self) -> None:
        self.focus()

    def on_key(self, event: Key) -> None:
        if event.key != "escape":
            return
        self.remove()
        event.prevent_default()
        event.stop()


TRANSCRIPT_COPY_PATH = "out/chat_tui/transcript.txt"
DIAGNOSTICS_PATH = "out/chat_tui/diagnostics.json"

COMMAND_HELP: tuple[tuple[str, str], ...] = (
    ("/convo", "pick a validation conversation and starting turn"),
    ("/steer", "segments: view, rewind, edit conditioning, steer"),
    ("/clear", "reset the conversation"),
    ("/regen", "change the seed and regenerate the last assistant message"),
    ("/seed N", "set the RNG seed (applies at the next reset)"),
    ("/copy", "copy the transcript (clipboard + a file mirror)"),
    ("/diag", "dump full session state for bug reports"),
    ("/raw", "toggle raw token rendering"),
    ("/help", "this list"),
    ("/quit", "exit (alias: /exit)"),
)

KEY_HELP = (
    "enter send · enter (while generating) cancel · esc dismiss · "
    "ctrl+y copy · ctrl+t colors · ctrl+q quit"
)


class HelpModal(Static):
    """Slash-command and key reference that remains until Escape."""

    can_focus = True

    def __init__(self) -> None:
        commands = "\n".join(f"{name:<9} {summary}" for name, summary in COMMAND_HELP)
        super().__init__(f"{commands}\n\n{KEY_HELP}\n\nesc to dismiss", markup=False)
        self.add_class("chat-modal", "chat-help")

    def on_mount(self) -> None:
        self.focus()

    def on_key(self, event: Key) -> None:
        if event.key != "escape":
            return
        self.remove()
        event.prevent_default()
        event.stop()


class ChatApp(App):
    """Single-screen transcript, status, and input shell for ``ChatSession``."""

    DEFAULT_CSS = """
    Screen {
        layout: vertical;
    }

    #transcript-scroll {
        height: 1fr;
        padding: 1 2;
    }

    #transcript {
        height: auto;
    }

    #status {
        height: 1;
        padding: 0 1;
    }

    #chat-input {
        height: 3;
    }

    .chat-modal {
        layer: modal;
    }

    .chat-error {
        width: 80%;
        height: auto;
        max-height: 70%;
        margin: 4 8;
        padding: 1 2;
        border: round $error;
        background: $surface;
        overflow-y: auto;
    }

    .chat-help {
        width: 80%;
        height: auto;
        max-height: 70%;
        margin: 4 8;
        padding: 1 2;
        border: round $accent;
        background: $surface;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("ctrl+y", "copy_transcript", "Copy transcript"),
        Binding("ctrl+t", "toggle_colors", "Toggle colors"),
        Binding("ctrl+q", "request_quit", "Quit"),
        Binding("enter", "submit_or_cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        session: ChatSession | TurnChatSession,
        tokenizer: Any,
        checkpoint_label: str,
        seed: int,
        convo_modal_factory: ModalFactory | None = None,
        steer_modal_factory: ModalFactory | None = None,
        catalog_loader: Callable[[], ConvoCatalog] | None = None,
        sequence_len: int | None = None,
        max_new_tokens_per_turn: int | None = None,
        signals_source: SignalsSource | None = None,
        description_service: DescriptionSource | None = None,
        deterministic_rewind: bool = False,
        model_thread: ModelThread | None = None,
    ) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
            raise ValueError(f"seed must be a positive int, got {seed!r}")
        self._session = session
        self._tokenizer = tokenizer
        self._checkpoint_label = checkpoint_label
        self._seed = seed
        self._pending_seed: int | None = None
        self._steering_summary = "off"
        self._applied_steering: tuple[SteeringMethod, tuple[tuple[int, float], ...]] = (
            SteeringMethod.OFF,
            (),
        )
        self._convo_modal_factory = convo_modal_factory
        self._steer_modal_factory = steer_modal_factory
        self._signals_source = signals_source
        self._description_service = description_service
        self._description_service_shutdown = False
        self._deterministic_rewind = deterministic_rewind

        # An injected (possibly already-started) thread is adopted; start() is
        # idempotent, so on_mount's start call is a no-op for it.
        self._model_thread = model_thread if model_thread is not None else ModelThread()
        if (
            self._convo_modal_factory is None
            and catalog_loader is not None
            and sequence_len is not None
            and max_new_tokens_per_turn is not None
        ):
            self._convo_modal_factory = lambda: ConvoPicker(
                catalog_loader,
                self._tokenizer,
                sequence_len,
                max_new_tokens_per_turn,
                self._load_conversation_prefix,
                self._model_thread,
                on_recall=self._recall_conversation_turn,
            )
        if self._steer_modal_factory is None and self._signals_source is not None:
            self._steer_modal_factory = self._new_steer_menu
        self._busy = False
        self._generation_active = False
        self._cancel_requested = False
        self._active_cancel_token: dict[str, bool] | None = None
        self._quit_after_generation = False
        self._sequence = 0
        self._active_sequence: int | None = None
        self._colors_enabled = True
        self._raw_mode = False
        self._special_display = build_special_display(tokenizer)
        self._status_error: str | None = None
        self._rollback_detail: str | None = None

        self._tokens: tuple[int, ...] = ()
        self._ledger: tuple[LedgerSegment, ...] = ()
        self._used_tokens = 0
        self._total_tokens = 0
        self._transcript_text = ""
        self._transcript_spans: tuple[TranscriptSpan, ...] = ()
        self._rendered_transcript = Text()
        self._status_text = ""

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def model_thread(self) -> ModelThread:
        return self._model_thread

    @property
    def transcript(self) -> Text:
        return self._rendered_transcript

    @property
    def status_text(self) -> str:
        return self._status_text

    def compose(self) -> ComposeResult:
        # A bare Static clips rather than scrolls, so a conversation taller than
        # the pane would be unreachable; the container is what actually scrolls.
        # It must not take focus, or it swallows keys meant for the input.
        transcript_scroll = VerticalScroll(id="transcript-scroll")
        transcript_scroll.can_focus = False
        with transcript_scroll:
            yield Static(Text(), id="transcript")
        yield Static("", id="status", markup=False)
        yield Input(placeholder="Message or /command", id="chat-input")

    def on_mount(self) -> None:
        self._model_thread.start()
        self._refresh_from_session()
        self.query_one("#chat-input", Input).focus()

    def on_unmount(self) -> None:
        self._model_thread.shutdown()
        self._shutdown_description_service()

    def _shutdown_description_service(self) -> None:
        if self._description_service_shutdown:
            return
        self._description_service_shutdown = True
        service = self._description_service
        if service is not None:
            service.shutdown()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "submit_or_cancel":
            if self._busy:
                return self._generation_active
            return self.query_one("#chat-input", Input).has_focus and not bool(
                self.query(".chat-modal")
            )
        return super().check_action(action, parameters)

    def on_key(self, event: Key) -> None:
        if event.key != "escape" or not self._close_modal():
            return
        event.prevent_default()
        event.stop()

    def action_submit_or_cancel(self) -> None:
        if self._busy:
            if self._generation_active and not self._cancel_requested:
                self._cancel_requested = True
                if self._active_cancel_token is not None:
                    self._active_cancel_token["cancelled"] = True
            return

        input_widget = self.query_one("#chat-input", Input)
        text = input_widget.value
        input_widget.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._send(text)

    def action_toggle_colors(self) -> None:
        self._colors_enabled = not self._colors_enabled
        self._render_cached_transcript()

    def action_copy_transcript(self) -> None:
        """Copy the transcript exactly as displayed, and mirror it to a file.

        The clipboard write is OSC 52, which some terminals decline silently;
        the file mirror is the fallback that always works, so the notification
        names it rather than claiming a copy that may not have landed.
        """

        text = self._rendered_transcript.plain
        self.copy_to_clipboard(text)
        path = Path(TRANSCRIPT_COPY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.notify(
            f"copied {len(text)} characters to the clipboard (also written to {path})",
            markup=False,
        )

    def action_request_quit(self) -> None:
        if self._busy:
            self._quit_after_generation = True
            if self._generation_active and not self._cancel_requested:
                self._cancel_requested = True
                if self._active_cancel_token is not None:
                    self._active_cancel_token["cancelled"] = True
            return
        self.exit()

    def _next_sequence(self) -> int:
        self._sequence += 1
        self._active_sequence = self._sequence
        return self._sequence

    def _begin_job(
        self,
        *,
        generation: bool,
        cancel_token: dict[str, bool] | None = None,
    ) -> int:
        self._busy = True
        self._generation_active = generation
        self._cancel_requested = False
        self._active_cancel_token = cancel_token
        self._status_error = None
        self._rollback_detail = None
        sequence = self._next_sequence()
        self.query_one("#chat-input", Input).disabled = True
        self._refresh_status()
        return sequence

    def _send(self, text: str) -> None:
        open_text = self._session.open_user_text()
        if open_text is not None:
            # A rewind landed inside a user turn: the retained partial text is
            # committed (it lives in the KV at the rewind point), so the edited
            # input must extend it and only the suffix is forced.
            if not text.startswith(open_text):
                self._recall_conversation_turn(text)
                self._surface_error(
                    "the retained partial user text can only be extended; rewind further to edit it"
                )
                return
            text = text[len(open_text) :]
        cancel_token = {"cancelled": False}
        sequence = self._begin_job(generation=True, cancel_token=cancel_token)

        def on_event(event: GenerationEvent) -> None:
            self.call_from_thread(self._apply_event, sequence, event)

        def job() -> None:
            error: str | None = None
            try:
                self._session.send_user_turn(
                    text,
                    on_event=on_event,
                    cancel_check=lambda: cancel_token["cancelled"],
                )
            except Exception as exc:
                error = str(exc)
            # This is deliberately the job's final statement. The UI refresh may
            # use pure session getters while the queue is still structurally owned
            # by this job, and no model/session operation follows it.
            self.call_from_thread(
                self._generation_finished,
                sequence,
                error,
                None,
                False,
            )

        self._model_thread.submit(job)

    def _submit_session_operation(
        self,
        operation: Callable[[], None],
        *,
        reset_occurred: bool = False,
        new_seed: int | None = None,
    ) -> None:
        sequence = self._begin_job(generation=False)

        def job() -> None:
            error: str | None = None
            try:
                operation()
            except Exception as exc:
                error = str(exc)
            self.call_from_thread(
                self._operation_finished,
                sequence,
                error,
                reset_occurred,
                new_seed,
            )

        self._model_thread.submit(job)

    def _submit_event_operation(
        self,
        operation: Callable[[Callable[[GenerationEvent], None], Callable[[], bool]], None],
        *,
        on_done: Callable[[bool, str | None], None],
        consumes_pending_seed: bool,
        start_after_refresh: bool = False,
    ) -> None:
        cancel_token = {"cancelled": False}
        sequence = self._begin_job(generation=True, cancel_token=cancel_token)

        def on_event(event: GenerationEvent) -> None:
            self.call_from_thread(self._apply_event, sequence, event)

        def job() -> None:
            error: str | None = None
            try:
                operation(on_event, lambda: cancel_token["cancelled"])
            except Exception as exc:
                error = str(exc)
            self.call_from_thread(
                self._generation_finished,
                sequence,
                error,
                on_done,
                consumes_pending_seed,
            )

        if start_after_refresh:
            self.call_after_refresh(self._model_thread.submit, job)
        else:
            self._model_thread.submit(job)

    def _handle_command(self, text: str) -> None:
        parts = text.split()
        command = parts[0]
        if command in {"/quit", "/exit"} and len(parts) == 1:
            self.action_request_quit()
        elif command == "/clear" and len(parts) == 1:
            self._submit_session_operation(self._session.clear, reset_occurred=True)
        elif command == "/regen":
            if len(parts) != 1:
                self._surface_error("usage: /regen")
                return
            seed = self._seed + 1

            def on_done(success: bool, _error: str | None) -> None:
                if success:
                    self._seed = seed

            self._submit_event_operation(
                lambda on_event, cancel_check: self._session.regenerate(
                    seed, on_event, cancel_check=cancel_check
                ),
                on_done=on_done,
                consumes_pending_seed=True,
            )
        elif command == "/seed":
            if len(parts) != 2:
                self._surface_error("usage: /seed N")
                return
            try:
                seed = int(parts[1])
            except ValueError:
                self._surface_error(f"seed must be a positive int, got {parts[1]!r}")
                return
            if seed <= 0:
                self._surface_error(f"seed must be a positive int, got {seed!r}")
                return
            self._submit_session_operation(
                lambda: self._session.set_seed(seed),
                new_seed=seed,
            )
        elif command == "/convo" and len(parts) == 1:
            self._open_modal("convo", self._convo_modal_factory)
        elif command == "/steer" and len(parts) == 1:
            self._open_modal("steer", self._steer_modal_factory)
        elif command == "/copy" and len(parts) == 1:
            self.action_copy_transcript()
        elif command == "/diag" and len(parts) == 1:
            self._dump_diagnostics()
        elif command == "/raw" and len(parts) == 1:
            self._raw_mode = not self._raw_mode
            self._render_cached_transcript()
            self._refresh_status()
        elif command == "/help" and len(parts) == 1:
            self.mount(HelpModal())
        else:
            self._surface_error(f"unknown command: {command} (see /help)")

    def _open_modal(self, name: str, factory: ModalFactory | None) -> None:
        if self._busy:
            self.notify("generation in flight", severity="warning", markup=False)
            return
        if factory is None:
            self.notify(f"{name} unavailable")
            return
        widget = factory()
        widget.add_class("chat-modal")
        self.mount(widget)

    def _new_steer_menu(self) -> SteerMenu:
        signals_source = self._signals_source
        if signals_source is None:
            raise RuntimeError("steer unavailable")

        menu: SteerMenu
        turn_identity = (
            self._session.identity if isinstance(self._session, TurnChatSession) else None
        )

        def close() -> None:
            self._close_modal()

        def set_steering(selection: SteeringSelection) -> None:
            self._submit_steering(menu, selection, turn_identity=turn_identity)

        menu = SteerMenu(
            session_view=lambda: self._session.ledger(),
            discard_preview=self._session.discard_preview,
            signals_source=signals_source,
            on_rewind=lambda entry: self._rewind_segment(menu, entry),
            on_modify=lambda entry, features: self._modify_segment(menu, entry, features),
            on_set_steering=set_steering,
            on_close=close,
            description_service=self._description_service,
            initial_steering=self._applied_steering,
            is_current=(
                lambda: (
                    self._session.identity == turn_identity
                    if isinstance(self._session, TurnChatSession)
                    else True
                )
            ),
        )
        return menu

    def _ledger_index_by_identity(self, entry: LedgerSegment) -> int:
        for index, candidate in enumerate(self._session.ledger()):
            if candidate is entry:
                return index
        raise RuntimeError("selected ledger segment is no longer live")

    def _rewind_segment(self, menu: SteerMenu, entry: LedgerSegment) -> None:
        ledger_index = (
            self._ledger_index_by_identity(entry) if entry.kind is SegmentKind.PREFIX_REAL else None
        )

        def operation(
            on_event: Callable[[GenerationEvent], None],
            cancel_check: Callable[[], bool],
        ) -> None:
            if isinstance(self._session, TurnChatSession):
                self._session.assert_current_entry(entry)
            if entry.kind is SegmentKind.PREFIX_REAL:
                assert ledger_index is not None
                self._session.rewind_to_prefix(ledger_index, on_event, cancel_check=cancel_check)
                return
            if entry.selfcond_ordinal is None:
                raise RuntimeError("self-conditioned ledger segment has no ordinal")
            self._session.rewind(entry.selfcond_ordinal, on_event, cancel_check=cancel_check)

        # Prefix rewinds always rederive the RNG streams from the configured
        # seed; only ordinal rewinds restore snapshot RNG under deterministic
        # rewind and thereby leave a pending /seed unconsumed.
        reseeds_from_config = entry.kind is SegmentKind.PREFIX_REAL
        self._submit_event_operation(
            operation,
            on_done=menu.on_done,
            consumes_pending_seed=reseeds_from_config or not self._deterministic_rewind,
        )

    def _modify_segment(
        self,
        menu: SteerMenu,
        entry: LedgerSegment,
        features: dict[int, int],
    ) -> None:
        ledger_index = (
            self._ledger_index_by_identity(entry) if entry.kind is SegmentKind.PREFIX_REAL else None
        )

        def operation(
            on_event: Callable[[GenerationEvent], None],
            cancel_check: Callable[[], bool],
        ) -> None:
            if isinstance(self._session, TurnChatSession):
                self._session.assert_current_entry(entry)
            if entry.kind is SegmentKind.PREFIX_REAL:
                assert ledger_index is not None
                self._session.modify_prefix(
                    ledger_index, features, on_event, cancel_check=cancel_check
                )
                return
            if entry.selfcond_ordinal is None:
                raise RuntimeError("self-conditioned ledger segment has no ordinal")
            self._session.modify(
                entry.selfcond_ordinal, features, on_event, cancel_check=cancel_check
            )

        # Prefix edits and S0 edits re-prefill and rederive RNG from the
        # configured seed even under deterministic rewind; ordinal>=1 edits
        # restore snapshot RNG there and leave a pending /seed unconsumed.
        reseeds_from_config = entry.kind is SegmentKind.PREFIX_REAL or entry.selfcond_ordinal == 0
        if isinstance(self._session, TurnChatSession):
            reseeds_from_config = False  # deterministic turn edits restore epoch RNG snapshots
        self._submit_event_operation(
            operation,
            on_done=menu.on_done,
            consumes_pending_seed=reseeds_from_config or not self._deterministic_rewind,
        )

    def _submit_steering(
        self,
        menu: SteerMenu,
        selection: SteeringSelection,
        *,
        turn_identity: tuple[int, int] | None = None,
    ) -> None:
        session = self._session
        if isinstance(session, TurnChatSession):
            expected = session.identity if turn_identity is None else turn_identity

            def operation(on_event, cancel_check):
                if session.identity != expected:
                    raise RuntimeError("steering menu belongs to an earlier Chat epoch/revision")
                session.apply_steering(selection, on_event, cancel_check=cancel_check)

            def done(success, message):
                if success:
                    self._steering_summary = self._steering_selection_summary(selection)
                    self._applied_steering = menu.enabled_state()
                else:
                    menu.display = True
                menu.on_done(success, message)

            # Turn steering includes the whole regeneration transaction. Hide
            # the menu before starting it, retaining its edits for failures.
            menu.display = False
            self._submit_event_operation(
                operation,
                on_done=done,
                consumes_pending_seed=bool(session.ledger()) and not self._deterministic_rewind,
                start_after_refresh=True,
            )
            return
        sequence = self._begin_job(generation=False)

        def job() -> None:
            error: str | None = None
            try:
                self._session.set_steering(selection)
            except Exception as exc:
                error = str(exc)
            self.call_from_thread(
                self._steering_finished,
                sequence,
                menu,
                selection,
                error,
            )

        self._model_thread.submit(job)

    def _steering_finished(
        self,
        sequence: int,
        menu: SteerMenu,
        selection: SteeringSelection,
        error: str | None,
    ) -> None:
        if sequence != self._active_sequence:
            return
        self._active_sequence = None
        if error is None:
            self._steering_summary = self._steering_selection_summary(selection)
            self._applied_steering = menu.enabled_state()
        menu.on_done(error is None, error)
        self._finish_job(error)
        if error is None:
            self.call_after_refresh(self.query_one("#chat-input", Input).focus)
            # Let the menu removal reach the screen before model work starts.
            self.call_after_refresh(self._submit_resteer)

    def _submit_resteer(self) -> None:
        """Regenerate the live segment so a just-applied steering change is
        visible immediately instead of from the next boundary."""

        ledger = self._session.ledger()
        if not ledger:
            return
        if self._session.open_user_text() is not None:
            self.notify(
                "steering saved; it applies from the next segment (the partial user turn is kept)",
                markup=False,
            )
            return
        selfcond_count = sum(1 for entry in ledger if entry.kind is SegmentKind.SELFCOND)

        def operation(
            on_event: Callable[[GenerationEvent], None],
            cancel_check: Callable[[], bool],
        ) -> None:
            self._session.resteer(on_event, cancel_check=cancel_check)

        def on_done(success: bool, message: str | None) -> None:
            del success, message

        # The S0 path re-prefills (reseeding from the configured seed); the
        # replay path restores snapshot RNG under deterministic rewind.
        self._submit_event_operation(
            operation,
            on_done=on_done,
            consumes_pending_seed=selfcond_count < 2 or not self._deterministic_rewind,
        )

    @staticmethod
    def _steering_selection_summary(selection: SteeringSelection) -> str:
        if selection.method is SteeringMethod.OFF:
            return "off"
        method = "interp" if selection.method is SteeringMethod.INTERPOLATION else "restrict"
        total = math.fsum(strength for _signal, strength in selection.enabled)
        return f"{method} Σf={total:.1f} ({len(selection.enabled)})"

    def _load_conversation_prefix(self, messages: list) -> None:
        self._submit_session_operation(
            lambda: self._session.load_prefix(messages),
            reset_occurred=True,
        )

    def _recall_conversation_turn(self, text: str) -> None:
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = text
        input_widget.cursor_position = len(text)

    def _close_modal(self) -> bool:
        modals = list(self.query(".chat-modal"))
        if not modals:
            return False
        modals[-1].remove()
        return True

    def _apply_event(self, sequence: int, event: GenerationEvent) -> None:
        if sequence != self._active_sequence:
            return
        if isinstance(self._session, TurnChatSession) and not self._session.accepts_event(event):
            return
        if event.kind is GenerationEventKind.TRANSCRIPT:
            self._transcript_text = event.detail
            self._rendered_transcript = Text(self._stream_display_text())
            self.query_one("#transcript", Static).update(self._rendered_transcript)
        elif (delta := event.text_delta) is not None:
            if delta.rewritten_from is None:
                self._transcript_text += delta.appended
            else:
                self._transcript_text = (
                    self._transcript_text[: delta.rewritten_from] + delta.appended
                )
            self._rendered_transcript = Text(self._stream_display_text())
            self.query_one("#transcript", Static).update(self._rendered_transcript)
        if event.kind is GenerationEventKind.ERROR_ROLLBACK:
            # Remembered so the job's terminal error modal keeps the rollback
            # point instead of replacing it with the bare exception text.
            self._rollback_detail = event.detail
            self._surface_error(event.detail, modal=True)

    def _generation_finished(
        self,
        sequence: int,
        error: str | None,
        on_done: Callable[[bool, str | None], None] | None,
        consumes_pending_seed: bool,
    ) -> None:
        if sequence != self._active_sequence:
            return
        self._active_sequence = None
        if error is None and consumes_pending_seed:
            self._pending_seed = None
        self._refresh_from_session()
        if on_done is not None:
            on_done(error is None, error)
        self._finish_job(error)

    def _operation_finished(
        self,
        sequence: int,
        error: str | None,
        reset_occurred: bool,
        new_seed: int | None,
    ) -> None:
        if sequence != self._active_sequence:
            return
        self._active_sequence = None
        if error is None:
            if new_seed is not None:
                self._seed = new_seed
                self._pending_seed = new_seed
            if reset_occurred:
                self._pending_seed = None
        self._refresh_from_session()
        self._finish_job(error)

    def _finish_job(self, error: str | None) -> None:
        self._busy = False
        self._generation_active = False
        self._cancel_requested = False
        self._active_cancel_token = None
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = False
        input_widget.focus()
        if error is not None:
            self._surface_error(self._rollback_detail or error, modal=True)
        else:
            self._refresh_status()
        if self._quit_after_generation:
            self._quit_after_generation = False
            self.exit()

    def _refresh_from_session(self) -> None:
        self._used_tokens, self._total_tokens = self._session.token_budget()
        self._transcript_text, self._transcript_spans = self._session.transcript_spans()
        self._tokens = self._session.conversation_tokens()
        self._render_cached_transcript()
        self._refresh_status()
        open_text = self._session.open_user_text()
        if open_text and not self.query_one("#chat-input", Input).value:
            # Spec: after a rewind lands inside a user turn, the input line is
            # prefilled with the partial turn text for editing.
            self._recall_conversation_turn(open_text)

    def _render_cached_transcript(self) -> None:
        if self._raw_mode:
            text, spans = self._transcript_text, self._transcript_spans
        else:
            text, spans = pretty_view(
                self._tokenizer,
                self._tokens,
                self._transcript_spans,
                self._special_display,
            )
        rendered = Text(text)
        if self._colors_enabled:
            for span in spans:
                rendered.stylize(
                    Style(
                        color=color_for_idx(span.color_idx),
                        dim=span.kind is SegmentKind.PREFIX_REAL,
                        underline=span.is_live,
                    ),
                    span.char_start,
                    span.char_end,
                )
        self._rendered_transcript = rendered
        self.query_one("#transcript", Static).update(rendered)
        self.query_one("#transcript-scroll", VerticalScroll).scroll_end(animate=False)

    def _dump_diagnostics(self) -> None:
        """Write the full session state a steering bug report needs.

        The transcript alone cannot distinguish a model behaviour from a
        conditioning defect: the token ids, the per-segment provenance, and the
        exact feature dict the provider recorded are what separate them.
        """

        used, total = self._session.token_budget()
        payload = {
            "checkpoint": self._checkpoint_label,
            "seed": self._seed,
            "pending_seed": self._pending_seed,
            "steering_summary": self._steering_summary,
            "applied_steering": [
                self._applied_steering[0].value,
                [[concept_id, strength] for concept_id, strength in self._applied_steering[1]],
            ],
            "raw_mode": self._raw_mode,
            "tokens_used": used,
            "tokens_total": total,
            "conversation_tokens": list(self._session.conversation_tokens()),
            "transcript": self._transcript_text,
            "displayed": self._rendered_transcript.plain,
            "segments": [
                {
                    "kind": entry.kind.value,
                    "selfcond_ordinal": entry.selfcond_ordinal,
                    "encoder_slot_idx": entry.encoder_slot_idx,
                    "token_start": entry.token_start,
                    "token_end": entry.token_end,
                    "provenance": entry.provenance.value,
                    "steering": (
                        None
                        if entry.steering is None
                        else {
                            "method": entry.steering.method.value,
                            "signals": list(entry.steering.signal_names),
                            "total_strength": entry.steering.total_strength,
                        }
                    ),
                    "features": {str(key): value for key, value in sorted(entry.features.items())},
                }
                for entry in self._session.ledger()
            ],
        }
        path = Path(DIAGNOSTICS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        self.notify(f"wrote session diagnostics to {path}", markup=False)

    def _stream_display_text(self) -> str:
        """In-flight transcript text in the active display mode."""

        if self._raw_mode:
            return self._transcript_text
        return stream_display_text(self._transcript_text, self._special_display)

    def _refresh_status(self) -> None:
        pending = " (next reset)" if self._pending_seed is not None else ""
        status = (
            f"{self._checkpoint_label} · tokens {self._used_tokens}/{self._total_tokens} "
            f"· steer: {self._steering_summary} · seed {self._seed}{pending}"
        )
        if self._raw_mode:
            status += " · raw"
        if self._generation_active:
            status += " · generating…"
        if self._status_error is not None:
            status += f" · {self._status_error}"
        self._status_text = status
        self.query_one("#status", Static).update(status)

    def _surface_error(self, detail: str, *, modal: bool = False) -> None:
        self._status_error = detail
        self.notify(detail, severity="error", timeout=8, markup=False)
        if modal:
            existing = list(self.query(ErrorModal))
            if existing:
                existing[-1].detail = detail
                existing[-1].update(f"{detail}\n\nesc to dismiss")
                existing[-1].focus()
            else:
                self.mount(ErrorModal(detail))
        self._refresh_status()

    def set_steering_summary(self, summary: str) -> None:
        """Update display-only steering text after a serialized Task-12 operation."""

        self._steering_summary = summary
        self._refresh_status()
