from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import torch as t

import scripts.completion_tui.app as app_module
from scripts.completion_tui.app import CompletionApp
from scripts.completion_tui.cluster_view import ClusterView
from scripts.completion_tui.data_loader import (
    ClusteredSegmentSample,
    TUIFeatureSamplingConfig,
)
from scripts.completion_tui.state import (
    DocumentData,
    FeatureCondition,
    Segment,
    SegmentClusterEvidence,
    SessionState,
)
from scripts.completion_tui.widgets.cluster_members import ClusterMembersPanel
from scripts.completion_tui.widgets.doc_selector import DocSelector
from scripts.completion_tui.widgets.feature_detail import FeatureDetail
from scripts.completion_tui.widgets.feature_editor import FeatureEditor
from scripts.completion_tui.widgets.feature_search import FeatureSearch
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from tests.test_completion_tui.conftest import WordMockTokenizer


def _cluster_view() -> ClusterView:
    members = (10, 11, 12)
    return ClusterView(
        rep_to_members={10: members},
        member_to_rep=dict.fromkeys(members, 10),
    )


def _segment(*, features: list[FeatureCondition] | None = None) -> Segment:
    return Segment(
        segment_idx=0,
        char_start=0,
        char_end=5,
        color_idx=0,
        features=features or [],
        id="segment-a",
    )


def _state(*, features: list[FeatureCondition] | None = None) -> SessionState:
    return SessionState(
        document_text="hello",
        segments=[_segment(features=features)],
        num_buckets=4,
        has_negatives=False,
    )


def _sampling_config(
    *,
    feature_mask: FeatureMask | None = None,
    clustered: bool = False,
) -> TUIFeatureSamplingConfig:
    return TUIFeatureSamplingConfig(
        boundaries=t.zeros((100, 3)),
        n_sae_features=100,
        feature_mask=feature_mask,
        consolidation=MagicMock() if clustered else None,
        effective_buckets=4 if clustered else None,
    )


def _sf() -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor([10, 11], dtype=t.int32),
        feature_mags=t.tensor([1.0, 2.0]),
        char_start=0,
        char_end=5,
        feature_percentiles=t.tensor([200, 220], dtype=t.uint8),
    )


class _DescriptionService:
    def __init__(self, cached: dict[int, str]) -> None:
        self.cached = cached

    def get_all_cached(self) -> dict[int, str]:
        return dict(self.cached)

    def get_description_sync(self, feature_id: int) -> str | None:
        return self.cached.get(feature_id)

    def request_description(
        self,
        feature_id: int,
        on_done: Callable[[int, str], None],
        *,
        on_start: Callable[[int], None] | None = None,
    ) -> None:
        if on_start is not None:
            on_start(feature_id)
        on_done(feature_id, self.cached[feature_id])

    def shutdown(self) -> None:
        pass


class _BlockingDescriptionService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()
        self._callbacks: list[Callable[[int, str], None]] = []
        self._lock = threading.Lock()

    def get_description_sync(self, feature_id: int) -> None:
        del feature_id
        return None

    def get_all_cached(self) -> dict[int, str]:
        return {}

    def request_description(
        self,
        feature_id: int,
        on_done: Callable[[int, str], None],
        *,
        on_start: Callable[[int], None] | None = None,
    ) -> None:
        with self._lock:
            self._callbacks.append(on_done)
            first_request = len(self._callbacks) == 1
        if not first_request:
            return

        def _complete() -> None:
            if on_start is not None:
                on_start(feature_id)
            self.started.set()
            assert self.release.wait(timeout=5)
            with self._lock:
                callbacks = list(self._callbacks)
            for callback in callbacks:
                callback(feature_id, "fresh description")
            self.done.set()

        threading.Thread(target=_complete, daemon=True).start()

    def shutdown(self) -> None:
        pass


def _patch_unmounted_overlay(monkeypatch) -> None:
    monkeypatch.setattr(CompletionApp, "mount", lambda self, *widgets: None)
    monkeypatch.setattr(FeatureEditor, "focus", lambda self: self)
    monkeypatch.setattr(FeatureEditor, "remove", lambda self: None)
    monkeypatch.setattr(FeatureSearch, "focus", lambda self: self)
    monkeypatch.setattr(FeatureSearch, "remove", lambda self: None)
    monkeypatch.setattr(ClusterMembersPanel, "focus", lambda self: self)
    monkeypatch.setattr(ClusterMembersPanel, "remove", lambda self: None)
    monkeypatch.setattr(FeatureDetail, "focus", lambda self: self)
    monkeypatch.setattr(FeatureDetail, "remove", lambda self: None)


def test_stale_description_callback_updates_current_editor_only(monkeypatch) -> None:
    service = _BlockingDescriptionService()
    app = CompletionApp(
        state=_state(features=[FeatureCondition(feature_id=10, magnitude=3)]),
        description_service=service,
    )
    monkeypatch.setattr(
        app,
        "call_from_thread",
        lambda callback, *args: callback(*args),
    )
    editor_a = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
    )
    app._editor = editor_a
    app._fetch_editor_descriptions()
    assert service.started.wait(timeout=5)

    app._editor = None
    editor_b = FeatureEditor(
        _segment(features=[FeatureCondition(feature_id=10, magnitude=3)]),
        num_buckets=4,
        has_negatives=False,
    )
    app._editor = editor_b
    app._fetch_editor_descriptions()
    service.release.set()
    assert service.done.wait(timeout=5)

    assert editor_a.sorted_features[0].description == app_module.DESCRIPTION_LOADING
    assert editor_b.sorted_features[0].description == "fresh description"


def test_initial_evidence_constructor_copies_mapping() -> None:
    initial = {
        "segment-a": SegmentClusterEvidence(
            fingerprint=frozenset({10}),
            evidence={10: {10: 0.8, 11: 0.6}},
        )
    }
    app = CompletionApp(
        state=_state(),
        cluster_view=_cluster_view(),
        initial_segment_evidence=initial,
    )
    initial["segment-a"].evidence[10][10] = 0.1
    assert app._segment_evidence == {
        "segment-a": SegmentClusterEvidence(
            fingerprint=frozenset({10}),
            evidence={10: {10: 0.8, 11: 0.6}},
        )
    }


def test_initial_evidence_pruned_to_live_segments() -> None:
    entry = SegmentClusterEvidence(
        fingerprint=frozenset({10}),
        evidence={10: {10: 0.8}},
    )
    app = CompletionApp(
        state=_state(),
        initial_segment_evidence={"segment-a": entry, "stale-segment": entry},
    )
    assert set(app._segment_evidence) == {"segment-a"}


def test_restore_preview_snapshot_prunes_preview_evidence(monkeypatch) -> None:
    committed = SegmentClusterEvidence(
        fingerprint=frozenset({10}),
        evidence={10: {10: 0.8}},
    )
    preview = SegmentClusterEvidence(
        fingerprint=frozenset({10}),
        evidence={10: {10: 0.2}},
    )
    app = CompletionApp(
        state=_state(),
        initial_segment_evidence={"segment-a": committed},
    )
    app._save_preview_snapshot()
    app._state.segments = [_segment()]
    app._state.segments[0].id = "preview-segment"
    app._segment_evidence["preview-segment"] = preview
    monkeypatch.setattr(app, "_refresh_preview_panels", lambda: None)

    app._restore_preview_snapshot()

    assert set(app._segment_evidence) == {"segment-a"}


def test_evidence_cache_pruned_after_preview_browse_and_confirm(
    monkeypatch,
) -> None:
    documents: dict[int, DocumentData] = {}
    for doc_idx in range(1, 5):
        segment = _segment()
        segment.id = f"preview-{doc_idx}"
        documents[doc_idx] = DocumentData(
            doc_idx=doc_idx,
            text="hello",
            segments=[segment],
            total_segments=1,
            evidence_by_segment_index={
                0: SegmentClusterEvidence(
                    fingerprint=frozenset({10}),
                    evidence={10: {10: doc_idx / 10}},
                )
            },
        )
    loader = MagicMock()
    loader.load_document.side_effect = lambda doc_idx, num_segments: documents[doc_idx]
    app = CompletionApp(
        state=_state(),
        tokenizer=WordMockTokenizer(),
        doc_loader=loader,
        initial_segment_evidence={
            "segment-a": SegmentClusterEvidence(
                fingerprint=frozenset({10}),
                evidence={10: {10: 0.9}},
            )
        },
    )
    monkeypatch.setattr(app, "_refresh_preview_panels", lambda: None)
    app._save_preview_snapshot()
    for doc_idx in documents:
        app._preview_document(doc_idx, 1)

    doc_selector = MagicMock()
    doc_selector.gen_prefilled_for_selected = False
    widgets = {
        "#doc-selector": doc_selector,
        "#text-panel": MagicMock(),
        "#segment-selector": MagicMock(),
        "#segment-bar": MagicMock(),
    }
    monkeypatch.setattr(app, "query_one", lambda selector, *args: widgets[selector])
    monkeypatch.setattr(app, "_close_overlay", lambda: None)
    monkeypatch.setattr(app, "_set_focus", lambda panel: None)

    app.on_doc_selector_document_selected(DocSelector.DocumentSelected(4, 1))

    live_ids = {segment.id for segment in app._state.segments}
    assert live_ids == {"preview-4"}
    assert set(app._segment_evidence) == live_ids


def test_preview_document_populates_evidence_cache(monkeypatch) -> None:
    doc = DocumentData(
        doc_idx=5,
        text="hello",
        segments=[_segment()],
        total_segments=1,
        evidence_by_segment_index={
            0: SegmentClusterEvidence(
                fingerprint=frozenset({10}),
                evidence={10: {10: 0.9, 11: 0.4}},
            )
        },
    )
    loader = MagicMock()
    loader.load_document.return_value = doc
    app = CompletionApp(
        state=_state(),
        tokenizer=WordMockTokenizer(),
        doc_loader=loader,
        cluster_view=_cluster_view(),
    )
    monkeypatch.setattr(app, "_refresh_preview_panels", lambda: None)
    app._preview_document(5, 1)
    assert app._segment_evidence["segment-a"] == SegmentClusterEvidence(
        fingerprint=frozenset({10}),
        evidence={10: {10: 0.9, 11: 0.4}},
    )


def test_selected_document_populates_evidence_cache(monkeypatch) -> None:
    doc = DocumentData(
        doc_idx=5,
        text="hello",
        segments=[_segment()],
        total_segments=1,
        evidence_by_segment_index={
            0: SegmentClusterEvidence(
                fingerprint=frozenset({10}),
                evidence={10: {10: 0.7}},
            )
        },
    )
    loader = MagicMock()
    loader.load_document.return_value = doc
    app = CompletionApp(
        state=_state(),
        tokenizer=WordMockTokenizer(),
        doc_loader=loader,
        cluster_view=_cluster_view(),
    )
    doc_selector = MagicMock()
    doc_selector.gen_prefilled_for_selected = False
    text_panel = MagicMock()
    segment_selector = MagicMock()
    segment_bar = MagicMock()
    widgets = {
        "#doc-selector": doc_selector,
        "#text-panel": text_panel,
        "#segment-selector": segment_selector,
        "#segment-bar": segment_bar,
    }
    monkeypatch.setattr(app, "query_one", lambda selector, *args: widgets[selector])
    monkeypatch.setattr(app, "_close_overlay", lambda: None)
    monkeypatch.setattr(app, "_set_focus", lambda panel: None)
    app.on_doc_selector_document_selected(DocSelector.DocumentSelected(5, 1))
    assert app._segment_evidence["segment-a"] == SegmentClusterEvidence(
        fingerprint=frozenset({10}),
        evidence={10: {10: 0.7}},
    )


def test_clustered_route_and_resample_store_current_segment_evidence(
    monkeypatch,
) -> None:
    cfg = _sampling_config(clustered=True)
    app = CompletionApp(
        state=_state(),
        sampling_config=cfg,
        cluster_view=_cluster_view(),
    )
    app._editing_segment_idx = 0
    expected = [FeatureCondition(feature_id=10, magnitude=3)]
    clustered = ClusteredSegmentSample(
        expected,
        SegmentClusterEvidence(
            fingerprint=frozenset({10}),
            evidence={10: {10: 0.9, 11: 0.5}},
        ),
    )
    cluster_calls: list[SegmentFeatures] = []

    def _clustered(sf, *args, **kwargs):
        cluster_calls.append(sf)
        return clustered

    monkeypatch.setattr(app_module, "clustered_sample_segment_features", _clustered)
    monkeypatch.setattr(
        app_module,
        "sample_segment_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy sampler reached")),
    )
    editor = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
        cluster_view=_cluster_view(),
    )
    app._editor = editor
    app._last_seg_features = _sf()
    monkeypatch.setattr(app, "_fetch_editor_descriptions", lambda: None)
    app.on_feature_editor_resample_requested(FeatureEditor.ResampleRequested())
    assert cluster_calls == [app._last_seg_features]
    assert editor.sorted_features == expected
    assert app._segment_evidence["segment-a"] == clustered.evidence


def test_open_submenu_builds_members_evidence_and_fetch_target(monkeypatch) -> None:
    _patch_unmounted_overlay(monkeypatch)
    app = CompletionApp(
        state=_state(features=[FeatureCondition(feature_id=10, magnitude=3)]),
        cluster_view=_cluster_view(),
        initial_segment_evidence={
            "segment-a": SegmentClusterEvidence(
                fingerprint=frozenset({10}),
                evidence={10: {10: 0.9, 11: 0.5}},
            )
        },
    )
    app._editing_segment_idx = 0
    app._editor = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
        cluster_view=_cluster_view(),
    )
    fetched: list[Callable[[], FeatureEditor | ClusterMembersPanel | None]] = []
    monkeypatch.setattr(app, "_fetch_descriptions_into", fetched.append)
    app.on_feature_editor_open_cluster_members(FeatureEditor.OpenClusterMembers(10))
    assert app._cluster_panel is not None
    assert app._cluster_panel.members == (10, 11, 12)
    assert app._cluster_panel._evidence == {10: 0.9, 11: 0.5}
    assert len(fetched) == 1
    assert fetched[0]() is app._cluster_panel
    assert app._overlay_sub_panel == "cluster"


def test_detail_return_stack_editor_cluster_detail(monkeypatch) -> None:
    _patch_unmounted_overlay(monkeypatch)
    app = CompletionApp(
        state=_state(features=[FeatureCondition(feature_id=10, magnitude=3)]),
        cluster_view=_cluster_view(),
    )
    app._editing_segment_idx = 0
    app._editor = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
        cluster_view=_cluster_view(),
    )
    app._cluster_panel = ClusterMembersPanel(10, (10, 11, 12), None)
    app._overlay_sub_panel = "cluster"
    monkeypatch.setattr(app, "_request_live_feature_highlight", lambda fid: None)
    monkeypatch.setattr(app, "_clear_live_feature_highlight", lambda **kwargs: None)

    app.on_cluster_members_panel_open_member_detail(ClusterMembersPanel.OpenMemberDetail(11))
    assert app._detail is not None
    assert app._detail.feature_id == 11
    assert app._detail_return_panel == "cluster"
    assert app._overlay_sub_panel == "detail"

    app.on_feature_detail_feature_detail_closed(FeatureDetail.FeatureDetailClosed())
    assert app._detail is None
    assert app._detail_return_panel == "editor"
    assert app._overlay_sub_panel == "cluster"
    assert app._cluster_panel.styles.display == "block"

    app.on_cluster_members_panel_cluster_members_closed(ClusterMembersPanel.ClusterMembersClosed())
    assert app._cluster_panel is None
    assert app._overlay_sub_panel == "editor"
    assert app._editor.styles.display == "block"


def test_close_overlay_cleans_cluster_panel_and_return_state(monkeypatch) -> None:
    _patch_unmounted_overlay(monkeypatch)
    app = CompletionApp(state=_state(), cluster_view=_cluster_view())
    removed: list[ClusterMembersPanel] = []
    panel = ClusterMembersPanel(10, (10, 11, 12), None)
    monkeypatch.setattr(panel, "remove", lambda: removed.append(panel))
    monkeypatch.setattr(app, "_clear_live_feature_highlight", lambda **kwargs: None)
    monkeypatch.setattr(app, "query_one", lambda *args: MagicMock())
    app._cluster_panel = panel
    app._detail_return_panel = "cluster"
    app._close_overlay()
    assert removed == [panel]
    assert app._cluster_panel is None
    assert app._detail_return_panel == "editor"


def test_search_available_excluded_mask_mapping_and_prefix(monkeypatch) -> None:
    _patch_unmounted_overlay(monkeypatch)
    mask = FeatureMask(FeatureMaskMode.blacklist, (77,))
    service = _DescriptionService(
        {
            10: "representative",
            11: "member eleven",
            12: "member twelve",
            20: "singleton",
            77: "masked",
        }
    )
    app = CompletionApp(
        state=_state(features=[FeatureCondition(feature_id=10, magnitude=3)]),
        description_service=service,
        sampling_config=_sampling_config(feature_mask=mask, clustered=True),
        cluster_view=_cluster_view(),
    )
    app._editor = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
        cluster_view=_cluster_view(),
    )
    app.on_feature_editor_open_feature_search(FeatureEditor.OpenFeatureSearch())
    assert app._search is not None
    assert 77 not in app._search._available
    assert app._search._available[10] == "[3 composite features] representative"
    assert app._search._available[11] == "member eleven"
    assert app._search._excluded == {10, 11, 12}


def test_feature_added_member_already_present_selects_and_notifies(monkeypatch) -> None:
    app = CompletionApp(
        state=_state(
            features=[
                FeatureCondition(feature_id=20, magnitude=2),
                FeatureCondition(feature_id=10, magnitude=1),
            ]
        ),
        cluster_view=_cluster_view(),
    )
    app._editor = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
        cluster_view=_cluster_view(),
    )
    notifications: list[str] = []
    monkeypatch.setattr(
        CompletionApp,
        "notify",
        lambda self, message, *args, **kwargs: notifications.append(message),
    )
    initial_count = len(app._editor.sorted_features)
    app.on_feature_search_feature_added(FeatureSearch.FeatureAdded(11))
    assert len(app._editor.sorted_features) == initial_count
    selected = app._editor.sorted_features[app._editor.selected_idx]
    assert selected.feature_id == 10
    assert notifications == ["feature 11 is a member of cluster @10 (already present)"]


def test_feature_added_member_adds_representative_and_notifies(monkeypatch) -> None:
    service = _DescriptionService({10: "representative", 11: "member"})
    app = CompletionApp(
        state=_state(features=[FeatureCondition(feature_id=20, magnitude=2)]),
        description_service=service,
        cluster_view=_cluster_view(),
    )
    app._editor = FeatureEditor(
        app._state.segments[0],
        num_buckets=4,
        has_negatives=False,
        cluster_view=_cluster_view(),
    )
    notifications: list[str] = []
    monkeypatch.setattr(
        CompletionApp,
        "notify",
        lambda self, message, *args, **kwargs: notifications.append(message),
    )
    app.on_feature_search_feature_added(FeatureSearch.FeatureAdded(11))
    added = app._editor.sorted_features[app._editor.selected_idx]
    assert added.feature_id == 10
    assert added.description == "representative"
    assert notifications == ["added cluster @10 (feature 11 is a member)"]


def test_unclustered_route_never_touches_cluster_sampler(monkeypatch) -> None:
    cfg = _sampling_config()
    app = CompletionApp(state=_state(), sampling_config=cfg)
    expected = [FeatureCondition(feature_id=11, magnitude=2)]
    legacy_calls: list[SegmentFeatures] = []

    def _legacy(sf, *args, **kwargs):
        legacy_calls.append(sf)
        return expected

    monkeypatch.setattr(app_module, "sample_segment_features", _legacy)
    monkeypatch.setattr(
        app_module,
        "clustered_sample_segment_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cluster sampler reached")),
        raising=False,
    )
    assert app._features_from_seg_features(_sf()) is expected
    assert len(legacy_calls) == 1
    assert app._cluster_view is None
    assert app._feature_mask is None


def test_launch_passes_cluster_view_and_initial_evidence() -> None:
    source = (Path(__file__).parents[2] / "scripts" / "completion_tui" / "__main__.py").read_text(
        encoding="utf-8"
    )
    assert "cluster_view=contract.cluster_view" in source
    assert "initial_segment_evidence=initial_segment_evidence" in source
