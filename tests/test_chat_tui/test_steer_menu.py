"""Pilot tests for the two-pane ``/steer`` modal."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch as t
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from scripts.chat_tui.app import ChatApp
from scripts.chat_tui.descriptions import (
    OfflineDescriptionService,
    descriptions_from_autointerp_ranked,
    write_offline_descriptions,
)
from scripts.chat_tui.session import (
    DiscardPreview,
    GenerationEvent,
    LedgerSegment,
    SegmentKind,
    SegmentProvenance,
    SteeringStamp,
    TranscriptSpan,
)
from scripts.chat_tui.steering import (
    SignalSource,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)
from scripts.chat_tui.widgets.steer_menu import SteerMenu
from scripts.completion_tui.color_manager import color_for_idx


class AsciiTokenizer:
    _SPECIALS = {
        "<bos>": 0x110000,
        "<user_start>": 0x110001,
        "<user_end>": 0x110002,
        "<assistant_start>": 0x110003,
        "<assistant_end>": 0x110004,
    }

    def decode(self, tokens: tuple[int, ...] | list[int]) -> str:
        by_id = {token_id: name for name, token_id in self._SPECIALS.items()}
        return "".join(by_id.get(token) or chr(token) for token in tokens)

    def get_bos_token_id(self) -> int:
        return self._SPECIALS["<bos>"]

    def encode_special(self, name: str) -> int:
        return self._SPECIALS[name]


def _entry(
    kind: SegmentKind,
    provenance: SegmentProvenance,
    start: int,
    end: int,
    features: dict[int, int],
    color_idx: int,
    *,
    ordinal: int | None = None,
    steering: SteeringStamp | None = None,
) -> LedgerSegment:
    return LedgerSegment(
        kind=kind,
        encoder_slot_idx=color_idx,
        selfcond_ordinal=ordinal,
        token_start=start,
        token_end=end,
        features=features,
        provenance=provenance,
        steering=steering,
        color_idx=color_idx,
    )


def _ledger() -> tuple[LedgerSegment, ...]:
    return (
        _entry(
            SegmentKind.PREFIX_REAL,
            SegmentProvenance.REAL,
            0,
            4,
            {7: 0, 9: 1},
            1,
        ),
        _entry(
            SegmentKind.SELFCOND,
            SegmentProvenance.PREDICTED,
            4,
            8,
            {11: 0},
            2,
            ordinal=0,
        ),
        _entry(
            SegmentKind.SELFCOND,
            SegmentProvenance.STEERED,
            8,
            12,
            {12: 0, 13: 1},
            3,
            ordinal=1,
            steering=SteeringStamp(
                method=SteeringMethod.INTERPOLATION,
                signal_names=("Alpha", "Beta"),
                total_strength=2.0,
            ),
        ),
        _entry(
            SegmentKind.SELFCOND,
            SegmentProvenance.OVERRIDDEN,
            12,
            20,
            {14: 1},
            4,
            ordinal=2,
        ),
    )


class FakeSession:
    def __init__(self) -> None:
        self.segments = _ledger()
        self.tokens = tuple(ord("a") + index % 20 for index in range(20))
        self.rewinds: list[tuple[int, str]] = []
        self.prefix_rewinds: list[tuple[int, str]] = []
        self.modifies: list[tuple[int, dict[int, int], str]] = []
        self.prefix_modifies: list[tuple[int, dict[int, int], str]] = []
        self.steering: list[tuple[SteeringSelection, str]] = []
        self.steering_error: str | None = None
        self.rewind_error: str | None = None
        self.seed_calls: list[int] = []
        self.preview_calls: list[tuple[LedgerSegment, bool]] = []
        self.block_for_cancel = False
        self.observed_cancel: bool | None = None
        self.resteers: list[str] = []

    def conversation_tokens(self) -> tuple[int, ...]:
        return self.tokens

    def ledger(self) -> tuple[LedgerSegment, ...]:
        return self.segments

    def token_budget(self) -> tuple[int, int]:
        return len(self.tokens), 64

    def transcript_spans(self) -> tuple[str, tuple[TranscriptSpan, ...]]:
        text = "".join(chr(token) for token in self.tokens)
        return text, tuple(
            TranscriptSpan(
                entry.token_start,
                entry.token_end,
                entry.color_idx,
                entry.kind,
                index == len(self.segments) - 1 and entry.kind is SegmentKind.SELFCOND,
            )
            for index, entry in enumerate(self.segments)
        )

    def discard_preview(
        self,
        entry: LedgerSegment,
        *,
        is_modify: bool = False,
    ) -> DiscardPreview:
        self.preview_calls.append((entry, is_modify))
        if entry.kind is SegmentKind.PREFIX_REAL:
            retained, discarded_tokens, discarded_segments = (4, 16, 3)
        else:
            retained, discarded_tokens, discarded_segments = (8, 12, 2)
        return DiscardPreview(
            retained,
            discarded_tokens,
            discarded_segments,
            "modify" if is_modify else "rewind",
        )

    def set_seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def open_user_text(self) -> str | None:
        return None

    def rewind(
        self,
        ordinal: int,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        del on_event
        if self.rewind_error is not None:
            raise RuntimeError(self.rewind_error)
        if self.block_for_cancel:
            assert cancel_check is not None
            deadline = time.monotonic() + 2
            while not cancel_check() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.observed_cancel = bool(cancel_check())
        self.rewinds.append((ordinal, threading.current_thread().name))

    def rewind_to_prefix(
        self,
        ledger_index: int,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        del on_event, cancel_check
        self.prefix_rewinds.append((ledger_index, threading.current_thread().name))

    def modify(
        self,
        ordinal: int,
        features: dict[int, int],
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        del on_event, cancel_check
        self.modifies.append((ordinal, dict(features), threading.current_thread().name))

    def modify_prefix(
        self,
        ledger_index: int,
        features: dict[int, int],
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        del on_event, cancel_check
        self.prefix_modifies.append((ledger_index, dict(features), threading.current_thread().name))

    def set_steering(self, selection: SteeringSelection) -> None:
        if self.steering_error is not None:
            raise ValueError(self.steering_error)
        self.steering.append((selection, threading.current_thread().name))

    def resteer(
        self,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        del on_event, cancel_check
        self.resteers.append(threading.current_thread().name)


class StubSignalsSource:
    def __init__(self, *, cached: bool = False) -> None:
        self.concepts = ((10, "Alpha"), (20, "Beta concept"), (21, "Gamma"))
        initial = "cached" if cached else "unfitted"
        self.states = {concept_id: initial for concept_id, _name in self.concepts}
        self.signals = {
            concept_id: SteeringSignal(
                name=name,
                target_scores=t.tensor([0.5, -0.5], dtype=t.float32),
                source=SignalSource.AXBENCH_CONCEPT,
                identity=f"signal-{concept_id}",
            )
            for concept_id, name in self.concepts
        }
        self.pending: dict[
            int,
            tuple[Callable[[SteeringSignal], None], Callable[[str], None]],
        ] = {}

    def list_concepts(self) -> tuple[tuple[int, str], ...]:
        return self.concepts

    def state_of(self, concept_id: int) -> str:
        return self.states[concept_id]

    def ensure_fitted(
        self,
        concept_id: int,
        on_ready: Callable[[SteeringSignal], None],
        on_error: Callable[[str], None],
    ) -> None:
        if self.states[concept_id] == "cached":
            on_ready(self.signals[concept_id])
            return
        self.states[concept_id] = "fitting"
        self.pending[concept_id] = (on_ready, on_error)

    def signal_of(self, concept_id: int) -> SteeringSignal:
        if self.states[concept_id] != "cached":
            raise RuntimeError("signal is not cached")
        return self.signals[concept_id]

    def complete(self, concept_id: int) -> None:
        self.states[concept_id] = "cached"
        on_ready, _on_error = self.pending.pop(concept_id)
        on_ready(self.signals[concept_id])

    def fail(self, concept_id: int, detail: str) -> None:
        self.states[concept_id] = "unfitted"
        _on_ready, on_error = self.pending.pop(concept_id)
        on_error(detail)


class RecordingDescriptions:
    def __init__(self) -> None:
        self.requests: list[int] = []
        self.shutdown_calls = 0

    def request_description(
        self,
        feature_id: int,
        callback: Callable[[int, str], None],
        on_start: Callable[[int], None] | None = None,
    ) -> None:
        del on_start
        self.requests.append(feature_id)
        callback(feature_id, f"description {feature_id}")

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _app(
    session: FakeSession,
    signals: StubSignalsSource,
    *,
    descriptions: RecordingDescriptions | None = None,
    deterministic_rewind: bool = False,
) -> ChatApp:
    return ChatApp(
        session=session,  # type: ignore[arg-type]
        tokenizer=AsciiTokenizer(),
        checkpoint_label="tiny.ckpt",
        seed=11,
        signals_source=signals,
        description_service=descriptions,  # type: ignore[arg-type]
        deterministic_rewind=deterministic_rewind,
    )


async def _wait_for(pilot: Any, predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not reached")


async def _open(pilot: Any) -> None:
    await pilot.press(*"/steer", "enter")
    await pilot.pause()


def _menu(app: ChatApp) -> SteerMenu:
    return app.query_one(SteerMenu)


def _body(app: ChatApp) -> Text:
    body = app.query_one("#steer-body", Static).content
    assert isinstance(body, Text)
    return body


def _title(app: ChatApp) -> str:
    return str(app.query_one("#steer-title", Static).content)


def _styles_at(text: Text, needle: str) -> str:
    position = text.plain.index(needle)
    return " ".join(str(span.style) for span in text._spans if span.start <= position < span.end)


@pytest.mark.anyio
async def test_segment_rows_render_chip_span_glyph_count_and_stamp() -> None:
    session = FakeSession()
    app = _app(session, StubSignalsSource())

    async with app.run_test() as pilot:
        await _open(pilot)
        body = _body(app)
        for glyph, span, count in (
            ("R", "[0,4)", 2),
            ("P", "[4,8)", 1),
            ("S", "[8,12)", 2),
            ("O", "[12,20)", 1),
        ):
            assert glyph in body.plain
            assert span in body.plain
            assert f"{count} features" in body.plain
        assert "interpolation Σf=2.0" in body.plain
        assert color_for_idx(1) in _styles_at(body, "█")


@pytest.mark.anyio
async def test_rewind_confirm_counts_abort_selfcond_and_prefix_routing() -> None:
    session = FakeSession()
    app = _app(session, StubSignalsSource())

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("down", "enter")
        assert "Discard 12 tokens and 2 segments after target?" in _body(app).plain
        assert session.preview_calls[-1] == (session.segments[1], False)
        await pilot.press("n")
        assert session.rewinds == []

        await pilot.press("enter", "y")
        await _wait_for(pilot, lambda: session.rewinds != [])
        assert session.rewinds == [(0, "chat-model")]
        assert len(app.query(SteerMenu)) == 0

        await _open(pilot)
        await pilot.press("enter", "y")
        await _wait_for(pilot, lambda: session.prefix_rewinds != [])
        assert session.prefix_rewinds == [(0, "chat-model")]
        assert len(app.query(SteerMenu)) == 0


@pytest.mark.anyio
async def test_feature_editor_exact_copy_edit_add_remove_apply_and_escape() -> None:
    session = FakeSession()
    descriptions = RecordingDescriptions()
    app = _app(session, StubSignalsSource(), descriptions=descriptions)

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("e", "1", "down", "delete", "a", "1", "2", "enter")
        assert "12  0" in _body(app).plain
        assert descriptions.requests == [7, 9, 12]
        assert "description 12" in _body(app).plain
        await pilot.press("enter")
        assert "Discard 16 tokens and 3 segments after target?" in _body(app).plain
        assert session.preview_calls[-1] == (session.segments[0], True)
        await pilot.press("y")
        await _wait_for(pilot, lambda: session.prefix_modifies != [])
        assert session.prefix_modifies == [(0, {7: 1, 12: 0}, "chat-model")]

        await _wait_for(pilot, lambda: not app.busy)
        await _open(pilot)
        await pilot.press("e", "1", "escape", "e")
        assert "7  0" in _body(app).plain
        assert session.segments[0].features == {7: 0, 9: 1}


def _scroll_pane(app: ChatApp) -> VerticalScroll:
    return app.query_one(SteerMenu).query_one("#steer-scroll", VerticalScroll)


@pytest.mark.anyio
async def test_signal_list_scrolls_to_keep_the_highlight_visible() -> None:
    """A long concept list must follow the cursor; it used to run off the pane
    and leave the user steering blind."""

    signals = StubSignalsSource(cached=True)
    # Real AxBench concept names overflow the pane. If rows were allowed to
    # wrap, a row index would stop being a line index and the cursor would
    # drift further out of view the deeper into the list it went.
    signals.concepts = tuple(
        (index, f"references to rental services and associated equipment {index}")
        for index in range(60)
    )
    signals.states = {concept_id: "cached" for concept_id, _name in signals.concepts}
    signals.signals = {
        concept_id: SteeringSignal(
            name=name,
            target_scores=t.tensor([0.5, -0.5], dtype=t.float32),
            source=SignalSource.AXBENCH_CONCEPT,
            identity=f"signal-{concept_id}",
        )
        for concept_id, name in signals.concepts
    }
    app = _app(FakeSession(), signals)

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab")
        pane = _scroll_pane(app)
        height = pane.size.height
        assert height > 0
        assert pane.scroll_offset.y == 0

        # The scroll maths treat a row index as a line index, so no row may
        # wrap: 60 rows + a blank line + the budget line.
        body = app.query_one(SteerMenu).query_one("#steer-body", Static)
        assert body.virtual_size.height == len(signals.concepts) + 2, (
            f"rows are wrapping ({body.virtual_size.height} lines for "
            f"{len(signals.concepts)} rows), so the cursor will drift out of view"
        )

        menu = app.query_one(SteerMenu)
        for _ in range(len(signals.concepts) - 1):
            await pilot.press("down")
        await pilot.pause()

        assert menu._signal_highlight == len(signals.concepts) - 1
        top = _scroll_pane(app).scroll_offset.y
        assert top > 0, "the pane never scrolled"
        # Not merely visible: a row must remain below the cursor, or it sits on
        # the pane's bottom edge and reads as cut off.
        assert top <= menu._signal_highlight < top + height - 1, (
            f"cursor {menu._signal_highlight} is flush against the bottom "
            f"(top={top}, height={height})"
        )

        # A highlight scrolled to the bottom must not sit flush against the
        # help line, or the row it is on reads as covered.
        help_line = app.query_one(SteerMenu).query_one("#steer-help", Static)
        gap = help_line.region.y - (pane.region.y + pane.region.height)
        assert gap >= 1, f"no blank row between the list and the help line (gap={gap})"

        for _ in range(len(signals.concepts) - 1):
            await pilot.press("up")
        await pilot.pause()

        assert menu._signal_highlight == 0
        assert _scroll_pane(app).scroll_offset.y == 0


@pytest.mark.anyio
async def test_feature_editor_scrolls_to_keep_the_highlight_visible() -> None:
    session = FakeSession()
    crowded = replace(session.segments[0], features={index: index % 2 for index in range(60)})
    session.segments = (crowded, *session.segments[1:])
    app = _app(session, StubSignalsSource(cached=True))

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("e")
        pane = _scroll_pane(app)
        height = pane.size.height
        assert height > 0

        menu = app.query_one(SteerMenu)
        for _ in range(59):
            await pilot.press("down")
        await pilot.pause()

        assert menu._feature_highlight == 59
        top = _scroll_pane(app).scroll_offset.y
        assert top > 0, "the editor never scrolled"
        assert top <= menu._feature_highlight < top + height


@pytest.mark.anyio
async def test_enabling_a_signal_selects_interpolation_instead_of_silently_doing_nothing() -> None:
    """Regression: the method defaulted to OFF, so enabling a concept and
    dialling its strength built the ZERO policy — the apply succeeded, the
    status read 'off', and nothing was steered."""

    session = FakeSession()
    signals = StubSignalsSource(cached=True)
    app = _app(session, signals)

    async with app.run_test() as pilot:
        await _open(pilot)
        # Exactly the owner's key sequence: tab to signals, space, right x7, enter.
        await pilot.press("tab", "space", *("right" for _ in range(7)))
        assert "method: interpolation" in str(app.query_one("#steer-title", Static).content)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: session.steering != [])
        await _wait_for(pilot, lambda: not app.busy)

    selection, _thread = session.steering[0]
    assert selection.method is SteeringMethod.INTERPOLATION
    assert selection.enabled == ((signals.signals[10], 2.0),)
    assert "steer: interp" in app.status_text


@pytest.mark.anyio
async def test_cycling_back_to_off_warns_that_enabled_signals_are_ignored() -> None:
    session = FakeSession()
    app = _app(session, StubSignalsSource(cached=True))

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "space")
        # m cycles interpolation -> restriction -> off.
        await pilot.press("m", "m")
        body = _body(app).plain
        assert "method is off" in body
        assert "will be IGNORED" in body
        assert "press m to pick a method" in body


@pytest.mark.anyio
async def test_editor_displays_inactive_bucket_as_minus_one_but_stores_real_index() -> None:
    """The reserved inactive bucket renders as -1; edits and the applied dict
    keep the real index (the 0/1 keys are unchanged)."""

    session = FakeSession()
    app = _app(session, StubSignalsSource())

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("e")
        body = _body(app).plain
        assert "7  0" in body
        assert "9  -1" in body
        assert "9  1" not in body

        await pilot.press("1")
        assert "7  -1" in _body(app).plain
        await pilot.press("enter", "y")
        await _wait_for(pilot, lambda: session.prefix_modifies != [])
        assert session.prefix_modifies == [(0, {7: 1, 9: 1}, "chat-model")]


@pytest.mark.anyio
async def test_offline_descriptions_render_synchronously_after_the_bucket(tmp_path) -> None:
    path = tmp_path / "regenerated_descriptions.json"
    path.write_text(json.dumps({"7": "cluster seven", 9: "cluster nine"}))
    service = OfflineDescriptionService(path)
    session = FakeSession()
    app = _app(session, StubSignalsSource(), descriptions=service)  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("e")
        body = _body(app).plain
        assert "7  0  cluster seven" in body
        assert "9  -1  cluster nine" in body
        assert "…" not in body


@pytest.mark.anyio
async def test_generated_descriptions_json_renders_in_the_feature_editor(tmp_path) -> None:
    """The full local path: ranked artifact -> generated JSON -> rendered row."""

    ranked_path = tmp_path / "autointerp_ranked.yaml"
    ranked_path.write_text(
        "feature_dir: /features\n"
        "score_threshold: 0.6\n"
        "top_k: null\n"
        "targets:\n"
        "- description: such as\n"
        "  feature_ids:\n"
        "  - 7\n"
        "  score: 1.0\n"
        "- description: override\n"
        "  feature_ids:\n"
        "  - 9\n"
        "  score: 1.0\n"
        "  error_message: partial score from 16/40 requested token trials\n"
        "- description: null\n"
        "  feature_ids:\n"
        "  - 11\n"
        "  score: 0.4\n"
    )
    json_path = tmp_path / "descriptions.json"
    write_offline_descriptions(descriptions_from_autointerp_ranked(ranked_path), json_path)
    session = FakeSession()
    app = _app(
        session,
        StubSignalsSource(),
        descriptions=OfflineDescriptionService(json_path),  # type: ignore[arg-type]
    )

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("e")
        body = _body(app).plain
        assert "7  0  such as" in body
        assert "9  -1  override" in body

        # Segment 1 holds feature 11, whose target recorded no description.
        await pilot.press("escape", "down", "e")
        assert "11  0" in _body(app).plain
        assert "11  0  " not in _body(app).plain


@pytest.mark.anyio
async def test_signals_fit_method_grid_and_over_budget_footer() -> None:
    session = FakeSession()
    signals = StubSignalsSource()
    app = _app(session, signals)

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "m", "space")
        assert "method: interpolation" in _title(app)
        assert "fitting…" in _body(app).plain

        signals.complete(10)
        await pilot.pause()
        assert "cached" in _body(app).plain

        expected = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
        for strength in expected:
            await pilot.press("right")
            assert _menu(app).strengths[10] == strength

        # Build an exact 6.2 total: 3.0 + 3.0 + 0.2.
        _menu(app).strengths[10] = 3.0
        signals.states[20] = "cached"
        signals.states[21] = "cached"
        await pilot.press("down", *("right" for _ in range(8)), "space")
        await pilot.press("down", "right", "space")
        footer = _body(app)
        assert "Σf 6.2/6.0" in footer.plain
        assert "red" in _styles_at(footer, "Σf 6.2/6.0")


@pytest.mark.anyio
async def test_sign_only_appears_only_for_enabled_zero_restriction() -> None:
    app = _app(FakeSession(), StubSignalsSource(cached=True))

    async with app.run_test() as pilot:
        await _open(pilot)
        # Space auto-selects interpolation, so one m reaches restriction.
        await pilot.press("tab", "space", "m")
        assert "method: restriction" in _title(app)
        assert "sign-only" in _body(app).plain
        await pilot.press("m")
        assert "sign-only" not in _body(app).plain


@pytest.mark.anyio
async def test_signal_filters_names_and_id_prefixes() -> None:
    app = _app(FakeSession(), StubSignalsSource(cached=True))

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "/", "b", "e", "t", "a")
        assert "Beta concept" in _body(app).plain
        assert "Alpha" not in _body(app).plain
        assert "Gamma" not in _body(app).plain

        await pilot.press("/", "2", "1")
        assert "21 · 0.0 · cached · Gamma" in _body(app).plain
        assert "Beta concept" not in _body(app).plain


@pytest.mark.anyio
async def test_apply_builds_selection_success_updates_status_and_error_stays_open() -> None:
    session = FakeSession()
    signals = StubSignalsSource(cached=True)
    app = _app(session, signals)

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "m", "space", *("right" for _ in range(7)))
        await pilot.press("down", "space", *("right" for _ in range(7)), "enter")
        await _wait_for(pilot, lambda: session.steering != [])
        await _wait_for(pilot, lambda: session.resteers != [])
        await _wait_for(pilot, lambda: not app.busy)

        selection, thread_name = session.steering[0]
        assert thread_name == "chat-model"
        assert selection.method is SteeringMethod.INTERPOLATION
        assert selection.enabled == (
            (signals.signals[10], 2.0),
            (signals.signals[20], 2.0),
        )
        assert "steer: interp Σf=4.0 (2)" in app.status_text
        assert len(app.query(SteerMenu)) == 0
        assert session.resteers == ["chat-model"]

        # Reopening reflects the ACTIVE steering instead of a blank menu.
        session.steering_error = "over-ANDed pools: Alpha, Beta concept"
        await _open(pilot)
        await pilot.press("tab")
        reopened = _body(app).plain
        assert "● 10 · 2.0 · cached · Alpha" in reopened
        assert "● 20 · 2.0 · cached · Beta concept" in reopened

        await pilot.press("m", *("left" for _ in range(7)), "enter")
        await _wait_for(pilot, lambda: not app.busy)
        assert len(app.query(SteerMenu)) == 1
        assert "over-ANDed pools: Alpha, Beta concept" in [
            notification.message for notification in app._notifications
        ]
        assert "10 · 0.0 · sign-only · Alpha" in _body(app).plain
        # A failed apply neither updates the remembered state nor resteers.
        assert session.resteers == ["chat-model"]


@pytest.mark.anyio
async def test_steering_menu_is_removed_before_regeneration_starts() -> None:
    session = FakeSession()
    app = _app(session, StubSignalsSource(cached=True))
    menu_present_at_start: list[bool] = []
    original = app._submit_resteer

    def observe_start() -> None:
        menu_present_at_start.append(bool(list(app.query(SteerMenu))))
        original()

    app._submit_resteer = observe_start  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "m", "space", "right", "enter")
        await _wait_for(pilot, lambda: bool(session.resteers) and not app.busy)

    assert menu_present_at_start == [False]


@pytest.mark.anyio
async def test_rewind_failure_keeps_menu_open_with_error_and_success_then_closes() -> None:
    session = FakeSession()
    session.rewind_error = "rewind exploded"
    app = _app(session, StubSignalsSource())

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("down", "enter", "y")
        await _wait_for(pilot, lambda: not app.busy)
        assert len(app.query(SteerMenu)) == 1
        assert "rewind exploded" == str(app.query_one("#steer-status", Static).content)

        session.rewind_error = None
        await pilot.press("escape")
        await pilot.press("y")
        await _wait_for(pilot, lambda: not app.busy)
        assert len(app.query(SteerMenu)) == 0


@pytest.mark.anyio
async def test_enter_cancels_a_running_rewind_through_the_job_token() -> None:
    """Event operations get the same job-scoped cancel token as sends: pressing
    Enter during a rewind's auto-continue must reach the session's cancel_check
    (previously the token was never created and cancel was silently ignored)."""

    session = FakeSession()
    session.block_for_cancel = True
    app = _app(session, StubSignalsSource())

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("down", "enter", "y")
        await _wait_for(pilot, lambda: app.busy)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: not app.busy)

    assert session.observed_cancel is True
    assert session.rewinds == [(0, "chat-model")]


@pytest.mark.anyio
async def test_pending_seed_is_consumed_by_prefix_rewind_but_not_ordinal_under_deterministic() -> (
    None
):
    """Under deterministic rewind, ordinal rewinds restore snapshot RNG (pending
    /seed survives) while prefix rewinds rederive RNG from the configured seed
    (pending /seed is consumed); the status flag must match reality."""

    session = FakeSession()
    app = _app(session, StubSignalsSource(), deterministic_rewind=True)

    async with app.run_test() as pilot:
        await pilot.press(*"/seed 7", "enter")
        await _wait_for(pilot, lambda: session.seed_calls == [7])
        await _wait_for(pilot, lambda: not app.busy)
        assert app._pending_seed == 7

        await _open(pilot)
        await pilot.press("down", "enter", "y")
        await _wait_for(pilot, lambda: session.rewinds != [])
        await _wait_for(pilot, lambda: not app.busy)
        assert app._pending_seed == 7

        await _open(pilot)
        await pilot.press("enter", "y")
        await _wait_for(pilot, lambda: session.prefix_rewinds != [])
        await _wait_for(pilot, lambda: not app.busy)
        assert app._pending_seed is None


@pytest.mark.anyio
async def test_steering_pending_blocks_reentrant_apply_and_escape() -> None:
    session = FakeSession()
    signals = StubSignalsSource(cached=True)
    started = threading.Event()
    release = threading.Event()
    original = session.set_steering

    def slow(selection: SteeringSelection) -> None:
        started.set()
        assert release.wait(timeout=2), "test did not release steering apply"
        original(selection)

    session.set_steering = slow  # type: ignore[method-assign]
    app = _app(session, signals)

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "m", "space", "enter")
        await _wait_for(pilot, started.is_set)
        menu = _menu(app)
        assert menu._pending == "steering"
        await pilot.press("space", "enter", "escape")
        assert len(session.steering) == 0
        assert len(app.query(SteerMenu)) == 1
        release.set()
        await _wait_for(pilot, lambda: not app.busy)
        assert len(session.steering) == 1
        await _wait_for(pilot, lambda: len(app.query(SteerMenu)) == 0)


@pytest.mark.anyio
async def test_fit_error_returns_row_to_unfitted_and_shows_error_status() -> None:
    session = FakeSession()
    signals = StubSignalsSource()
    app = _app(session, signals)

    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "space")
        assert "fitting…" in _body(app).plain
        signals.fail(10, "fit exploded")
        await pilot.pause()

        assert "10 · 0.0 · unfitted · Alpha" in _body(app).plain
        assert "fit exploded" == str(app.query_one("#steer-status", Static).content)


@pytest.mark.anyio
async def test_fresh_take_rewind_consumes_pending_seed_indicator() -> None:
    session = FakeSession()
    app = _app(session, StubSignalsSource())

    async with app.run_test() as pilot:
        await pilot.press(*"/seed 9", "enter")
        await _wait_for(pilot, lambda: session.seed_calls == [9])
        await _wait_for(pilot, lambda: not app.busy)
        assert "seed 9 (next reset)" in app.status_text

        await _open(pilot)
        await pilot.press("down", "enter", "y")
        await _wait_for(pilot, lambda: not app.busy)
        assert "seed 9 (next reset)" not in app.status_text
        assert "seed 9" in app.status_text


@pytest.mark.anyio
async def test_steer_while_busy_notifies_and_does_not_mount() -> None:
    app = _app(FakeSession(), StubSignalsSource())

    async with app.run_test() as pilot:
        app._busy = True
        app._handle_command("/steer")
        await pilot.pause()
        assert len(app.query(SteerMenu)) == 0
        assert "generation in flight" in [
            notification.message for notification in app._notifications
        ]
