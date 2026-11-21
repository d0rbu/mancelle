from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import scripts.completion_tui.app as app_module
from scripts.completion_tui.app import (
    CompletionApp,
    _document_evidence_by_segment_id,
)
from scripts.completion_tui.cluster_view import ClusterView
from scripts.completion_tui.data_loader import (
    DocumentLoader,
    TUIFeatureSamplingConfig,
)
from scripts.completion_tui.generation import GenerationManager
from scripts.completion_tui.highlight_types import (
    FeatureHighlightRequest,
    FeatureHighlightResult,
)
from scripts.completion_tui.state import FeatureCondition, SessionState
from scripts.completion_tui.widgets.cluster_members import ClusterMembersPanel
from scripts.completion_tui.widgets.feature_detail import FeatureDetail
from scripts.completion_tui.widgets.feature_editor import FeatureEditor
from scripts.completion_tui.widgets.feature_search import FeatureSearch
from scripts.completion_tui.widgets.segment_selector import SegmentSelector
from scripts.completion_tui.widgets.text_panel import TextPanel
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.multiseg_generator import MultisegInputs
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from tests.test_completion_tui.conftest import WordMockTokenizer

N_SAE_FEATURES = 16
NUM_BUCKETS = 4
PERCENTILE_CHECKSUM = "a" * 64


@dataclass(frozen=True)
class _ClusteredEnvironment:
    feature_dir: Path
    parquet_dir: Path
    boundaries: t.Tensor
    mask: FeatureMask
    consolidation: FeatureClusterConsolidation
    cluster_view: ClusterView
    sampling_config: TUIFeatureSamplingConfig


class _DescriptionService:
    def __init__(self) -> None:
        self._cached = {
            feature_id: f"feature {feature_id} description" for feature_id in range(N_SAE_FEATURES)
        }

    def get_description_sync(self, feature_id: int) -> str | None:
        return self._cached.get(feature_id)

    def get_all_cached(self) -> dict[int, str]:
        return dict(self._cached)

    def request_description(
        self,
        feature_id: int,
        on_done: Callable[[int, str], None],
        *,
        on_start: Callable[[int], None] | None = None,
    ) -> None:
        if on_start is not None:
            on_start(feature_id)
        on_done(feature_id, self._cached[feature_id])

    def shutdown(self) -> None:
        pass


class _HighlightService:
    def __init__(self) -> None:
        self.calls: list[FeatureHighlightRequest] = []

    def compute(self, request: FeatureHighlightRequest) -> FeatureHighlightResult:
        self.calls.append(request)
        segment = request.segments[request.target_segment_index]
        return FeatureHighlightResult(
            request_key=request.request_key,
            segment_idx=segment.segment_idx,
            feature_id=request.feature_id,
            status="ready",
            n_tokens=1,
        )


def _build_consolidation(mask: FeatureMask) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((2, 3), (8, 9)))
    resolved = ResolvedFeatureClusters(
        source_path=Path("clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=N_SAE_FEATURES,
        ),
    )
    return FeatureClusterConsolidation.build(resolved, mask)


@pytest.fixture()
def clustered_environment(tmp_path: Path) -> _ClusteredEnvironment:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquets"
    feature_dir.mkdir()
    parquet_dir.mkdir()

    text = "hello world"
    pq.write_table(
        pa.table({"text": [text]}),
        parquet_dir / "shard_00000.parquet",
    )
    base_path = save_features(
        {
            "offsets": t.tensor([0, 4, 7], dtype=t.int64),
            "feature_ids": t.tensor([2, 3, 7, 15, 2, 8, 9], dtype=t.int32),
            "feature_mags": t.tensor([0.2, 0.9, 0.5, 9.0, 0.7, 0.3, 1.0], dtype=t.float32),
            "doc_ids": t.tensor([0, 0], dtype=t.int64),
            "segment_indices": t.tensor([0, 1], dtype=t.int32),
            "segment_positions": t.tensor([0, 1], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 6], dtype=t.int32),
            "segment_char_ends": t.tensor([6, 11], dtype=t.int32),
            "n_segments": 2,
            "n_docs": 1,
            "segment_size": 4,
            "layer": 8,
            "context_window": 16,
            "pooling": "max_pool",
            "segment_mode": "fixed",
        },
        feature_dir / "shard_00000_features",
    )
    pct_dir = overlay_dir(
        feature_dir,
        OverlayKind.PERCENTILE,
        PERCENTILE_CHECKSUM,
    )
    write_manifest(
        pct_dir,
        PERCENTILE_CHECKSUM,
        OverlayKind.PERCENTILE,
        base_shard_paths=[base_path],
    )
    save_overlay_shard(
        {
            TensorKey.FEATURE_PERCENTILES: t.tensor(
                [128, 230, 127, 250, 200, 100, 240], dtype=t.uint8
            )
        },
        overlay_shard_path(pct_dir, "shard_00000"),
    )

    mask = FeatureMask(FeatureMaskMode.blacklist, (15,))
    consolidation = _build_consolidation(mask)
    boundaries = t.tensor(
        [[0.5, 1.5]] * N_SAE_FEATURES,
        dtype=t.float64,
    )
    sampling_config = TUIFeatureSamplingConfig(
        boundaries=boundaries,
        n_sae_features=N_SAE_FEATURES,
        feature_selection=ThresholdSelection(
            ThresholdType.feature_percentile,
            0.0,
        ),
        negative_selection=NegativeSamplingSelection(
            mean=1.0,
            min_k=1,
            max_k=2,
        ),
        feature_mask=mask,
        consolidation=consolidation,
        effective_buckets=NUM_BUCKETS - 1,
    )
    return _ClusteredEnvironment(
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        boundaries=boundaries,
        mask=mask,
        consolidation=consolidation,
        cluster_view=ClusterView.build(consolidation),
        sampling_config=sampling_config,
    )


def _loader(
    environment: _ClusteredEnvironment,
    *,
    sampling_config: TUIFeatureSamplingConfig | None,
    clustered: bool,
) -> DocumentLoader:
    return DocumentLoader(
        str(environment.feature_dir),
        str(environment.parquet_dir),
        environment.boundaries,
        NUM_BUCKETS,
        True,
        sampling_config=sampling_config,
        percentile_checksum=PERCENTILE_CHECKSUM if clustered else None,
    )


def _app(
    environment: _ClusteredEnvironment,
    *,
    loader: DocumentLoader | None = None,
    sampling_config: TUIFeatureSamplingConfig | None = None,
    cluster_view: ClusterView | None = None,
    generation_manager: GenerationManager | None = None,
    live_highlight_service: _HighlightService | None = None,
) -> CompletionApp:
    if loader is None:
        loader = _loader(
            environment,
            sampling_config=environment.sampling_config,
            clustered=True,
        )
    document = loader.load_document(0, 2)
    assert document is not None
    initial_evidence = _document_evidence_by_segment_id(document)
    last = document.segments[-1]
    cutoff = document.segments[-2].char_end
    last.is_generation_target = True
    last.target_token_count = 4
    last.char_start = cutoff
    last.char_end = cutoff
    state = SessionState(
        document_text=document.text[:cutoff],
        segments=document.segments,
        num_buckets=NUM_BUCKETS,
        has_negatives=True,
    )
    return CompletionApp(
        state=state,
        tokenizer=WordMockTokenizer(),
        doc_loader=loader,
        description_service=_DescriptionService(),
        generation_manager=generation_manager,
        available_doc_ids=[0],
        total_segments_map={0: 2},
        sampling_config=(
            environment.sampling_config if sampling_config is None else sampling_config
        ),
        live_highlight_service=live_highlight_service,
        cluster_view=(environment.cluster_view if cluster_view is None else cluster_view),
        initial_segment_evidence=initial_evidence,
    )


async def _open_editor(app: CompletionApp, pilot) -> FeatureEditor:
    await pilot.press("alt+down")
    selector = app.query_one("#segment-selector", SegmentSelector)
    selector.post_message(SegmentSelector.OpenFeatureEditor(0))
    await pilot.pause()
    return app.query_one("#feature-editor", FeatureEditor)


def _install_inline_worker(app: CompletionApp, monkeypatch) -> None:
    def _call_from_thread(callback, *args):
        return callback(*args)

    def _run_worker(callback, **kwargs):
        try:
            return callback()
        except Exception:
            return None

    monkeypatch.setattr(app, "call_from_thread", _call_from_thread)
    monkeypatch.setattr(app, "run_worker", _run_worker)


@pytest.mark.anyio
async def test_clustered_editor_flow(clustered_environment, monkeypatch) -> None:
    highlight = _HighlightService()
    app = _app(clustered_environment, live_highlight_service=highlight)
    _install_inline_worker(app, monkeypatch)
    async with app.run_test() as pilot:
        editor = await _open_editor(app, pilot)
        rows = editor.sorted_features
        assert rows[0].feature_id == 2
        assert rows[0].magnitude == 2
        assert any(row.feature_id == 7 and row.magnitude == 1 for row in rows)
        assert any(row.magnitude == -1 for row in rows)
        assert 3 not in {row.feature_id for row in rows}
        assert 15 not in {row.feature_id for row in rows}
        assert "[2 composite features] feature 2 description" in (
            editor._build_feature_lines()[0].plain
        )

        await pilot.press("enter")
        await pilot.pause()
        panel = app.query_one("#cluster-members", ClusterMembersPanel)
        assert panel._build_member_lines()[0].plain == ("▶     2* [p0.50 ✓] feature 2 description")
        assert panel._build_member_lines()[1].plain == ("      3  [p0.90 ✓] feature 3 description")

        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#feature-detail", FeatureDetail).feature_id == 2
        assert highlight.calls

        await pilot.press("escape")
        await pilot.pause()
        assert app._overlay_sub_panel == "cluster"
        await pilot.press("escape")
        await pilot.pause()
        assert app._overlay_sub_panel == "editor"
        committed = [(row.feature_id, row.magnitude, row.enabled) for row in editor.sorted_features]
        await pilot.press("escape")
        await pilot.pause()
        assert not app._overlay_open
        assert [
            (row.feature_id, row.magnitude, row.enabled) for row in app._state.segments[0].features
        ] == committed


@pytest.mark.anyio
async def test_search_add_member_maps_to_rep(
    clustered_environment,
    monkeypatch,
) -> None:
    app = _app(clustered_environment)
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, *args, **kwargs: notifications.append(message),
    )
    async with app.run_test() as pilot:
        editor = await _open_editor(app, pilot)
        editor.post_message(FeatureEditor.OpenFeatureSearch())
        await pilot.pause()
        search = app.query_one("#feature-search", FeatureSearch)

        initial_count = len(editor.sorted_features)
        search.post_message(FeatureSearch.FeatureAdded(3))
        await pilot.pause()
        assert len(editor.sorted_features) == initial_count
        assert editor.sorted_features[editor.selected_idx].feature_id == 2

        editor._features = [row for row in editor.sorted_features if row.feature_id != 2]
        editor._selected_idx = 0
        search.post_message(FeatureSearch.FeatureAdded(3))
        await pilot.pause()
        assert editor.sorted_features[editor.selected_idx].feature_id == 2
        assert sum(row.feature_id == 2 for row in editor.sorted_features) == 1
        assert notifications == [
            "feature 3 is a member of cluster @2 (already present)",
            "added cluster @2 (feature 3 is a member)",
        ]


@pytest.mark.anyio
async def test_generation_receives_rep_only_ids(
    clustered_environment,
    monkeypatch,
) -> None:
    captured: list[MultisegInputs] = []

    def _stream(**kwargs):
        inputs = kwargs["encoder_inputs"]
        assert isinstance(inputs, MultisegInputs)
        captured.append(inputs)
        yield from ()

    monkeypatch.setattr(
        "scripts.completion_tui.generation.generate_completion_stream",
        _stream,
    )
    manager = GenerationManager(
        model=MagicMock(),
        tokenizer=WordMockTokenizer(),
        device=t.device("cpu"),
        boundaries=clustered_environment.boundaries,
        feature_mask=clustered_environment.mask,
        consolidation=clustered_environment.consolidation,
    )
    app = _app(clustered_environment, generation_manager=manager)
    _install_inline_worker(app, monkeypatch)
    async with app.run_test() as pilot:
        app.query_one("#text-panel", TextPanel).post_message(TextPanel.GenerateRequested())
        await pilot.pause()
        assert len(captured) == 1
        inputs = captured[0]
        feature_ids = inputs.enc_feature_ids[inputs.enc_mask]
        expected_ids = {
            feature.feature_id
            for segment in app._state.segments
            for feature in segment.features
            if feature.enabled
        }
        assert feature_ids.numel() > 0
        assert set(feature_ids.tolist()) == expected_ids
        clustered_environment.mask.assert_no_masked(feature_ids)
        clustered_environment.consolidation.assert_representatives_only(
            feature_ids,
            surface="clustered pilot capture",
        )


@pytest.mark.anyio
async def test_resample_keeps_clustered_semantics(clustered_environment) -> None:
    app = _app(clustered_environment)
    async with app.run_test() as pilot:
        editor = await _open_editor(app, pilot)
        assert app._last_seg_features is not None
        await pilot.press("r")
        await pilot.pause()
        ids = t.tensor(
            [row.feature_id for row in editor.sorted_features],
            dtype=t.int64,
        )
        clustered_environment.mask.assert_no_masked(ids)
        clustered_environment.consolidation.assert_representatives_only(
            ids,
            surface="clustered pilot resample",
        )
        assert any(row.feature_id == 2 and row.magnitude == 2 for row in editor.sorted_features)
        assert "[2 composite features]" in editor._build_feature_lines()[0].plain
        segment_id = app._state.segments[0].id
        cached = app._segment_evidence[segment_id]
        assert cached.fingerprint == frozenset(
            row.feature_id
            for row in editor.sorted_features
            if clustered_environment.cluster_view.is_composite(row.feature_id)
        )
        assert cached.evidence[2] == pytest.approx({2: 128 / 255, 3: 230 / 255})


@pytest.mark.anyio
async def test_randomize_then_generation_undo_hides_stale_cluster_evidence(
    clustered_environment,
    monkeypatch,
) -> None:
    tokenizer = WordMockTokenizer()

    def _stream(**kwargs):
        del kwargs
        yield tokenizer.encode("foo")[0]

    monkeypatch.setattr(
        "scripts.completion_tui.generation.generate_completion_stream",
        _stream,
    )
    manager = GenerationManager(
        model=MagicMock(),
        tokenizer=tokenizer,
        device=t.device("cpu"),
        boundaries=clustered_environment.boundaries,
        feature_mask=clustered_environment.mask,
        consolidation=clustered_environment.consolidation,
    )
    sampling_config = replace(
        clustered_environment.sampling_config,
        negative_selection=None,
    )
    loader = _loader(
        clustered_environment,
        sampling_config=sampling_config,
        clustered=True,
    )
    raw_segments = loader.raw_lookup(0)
    assert raw_segments is not None
    assert len(raw_segments) == 2
    foreign_segment = raw_segments[1]
    choices = iter((0, foreign_segment))
    monkeypatch.setattr(app_module.random, "choice", lambda values: next(choices))

    app = _app(
        clustered_environment,
        loader=loader,
        sampling_config=sampling_config,
        generation_manager=manager,
    )
    _install_inline_worker(app, monkeypatch)
    async with app.run_test() as pilot:
        app.query_one("#text-panel", TextPanel).post_message(TextPanel.GenerateRequested())
        await pilot.pause()
        assert len(app._state.generation_history) == 1

        editor = await _open_editor(app, pilot)
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert {2, 8}.issubset({row.feature_id for row in editor.sorted_features})
        await pilot.press("escape")
        await pilot.pause()

        text_panel = app.query_one("#text-panel", TextPanel)
        text_panel.focus()
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert app._state.generation_history == []
        assert 8 not in {row.feature_id for row in app._state.segments[0].features}

        editor = await _open_editor(app, pilot)
        editor._selected_idx = next(
            index for index, row in enumerate(editor.sorted_features) if row.feature_id == 2
        )
        await pilot.press("enter")
        await pilot.pause()
        panel = app.query_one("#cluster-members", ClusterMembersPanel)
        assert "(0 active in this segment)" in panel._build_header_line().plain
        assert all("[  —  · ]" in line.plain for line in panel._build_member_lines())


@pytest.mark.anyio
async def test_masked_unclustered_flow(
    clustered_environment,
    monkeypatch,
) -> None:
    mask = clustered_environment.mask
    sampling_config = TUIFeatureSamplingConfig(
        boundaries=clustered_environment.boundaries,
        n_sae_features=N_SAE_FEATURES,
        feature_mask=mask,
    )
    loader = _loader(
        clustered_environment,
        sampling_config=sampling_config,
        clustered=False,
    )
    manager = GenerationManager(
        model=MagicMock(),
        tokenizer=WordMockTokenizer(),
        device=t.device("cpu"),
        boundaries=clustered_environment.boundaries,
        feature_mask=mask,
    )
    app = _app(
        clustered_environment,
        loader=loader,
        sampling_config=sampling_config,
        cluster_view=None,
        generation_manager=manager,
    )
    app._cluster_view = None
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, *args, **kwargs: notifications.append(message),
    )
    _install_inline_worker(app, monkeypatch)
    async with app.run_test() as pilot:
        editor = await _open_editor(app, pilot)
        assert 15 not in {row.feature_id for row in editor.sorted_features}
        await pilot.press("escape")
        await pilot.pause()
        app._state.segments[0].features.append(FeatureCondition(feature_id=15, magnitude=1))
        text_panel = app.query_one("#text-panel", TextPanel)
        text_panel.post_message(TextPanel.GenerateRequested())
        await pilot.pause()
        assert any("Generation failed: feature_mask violated" in n for n in notifications)
        assert text_panel.is_generating is False
        assert app.query_one("#text-panel", TextPanel) is text_panel
        before_keypress = app._state.document_text
        text_panel.focus()
        await pilot.press("x")
        await pilot.pause()
        assert app._state.document_text != before_keypress


@pytest.mark.anyio
async def test_masked_unclustered_resample_and_randomize_exclude_masked_ids(
    clustered_environment,
    monkeypatch,
) -> None:
    mask = clustered_environment.mask
    sampling_config = TUIFeatureSamplingConfig(
        boundaries=clustered_environment.boundaries,
        n_sae_features=N_SAE_FEATURES,
        feature_mask=mask,
    )
    loader = _loader(
        clustered_environment,
        sampling_config=sampling_config,
        clustered=False,
    )
    app = _app(
        clustered_environment,
        loader=loader,
        sampling_config=sampling_config,
        cluster_view=None,
    )
    app._cluster_view = None
    monkeypatch.setattr(app_module.random, "choice", lambda values: values[0])

    async with app.run_test() as pilot:
        editor = await _open_editor(app, pilot)
        assert 15 not in {row.feature_id for row in editor.sorted_features}

        await pilot.press("r")
        await pilot.pause()
        assert 15 not in {row.feature_id for row in editor.sorted_features}

        await pilot.press("ctrl+r")
        await pilot.pause()
        assert 15 not in {row.feature_id for row in editor.sorted_features}


@pytest.mark.anyio
async def test_unmasked_app_unchanged(
    clustered_environment,
    monkeypatch,
) -> None:
    sampling_config = TUIFeatureSamplingConfig(
        boundaries=clustered_environment.boundaries,
        n_sae_features=N_SAE_FEATURES,
    )
    loader = _loader(
        clustered_environment,
        sampling_config=sampling_config,
        clustered=False,
    )
    monkeypatch.setattr(
        app_module,
        "clustered_sample_segment_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cluster path reached")),
    )
    app = _app(
        clustered_environment,
        loader=loader,
        sampling_config=sampling_config,
        cluster_view=None,
    )
    app._cluster_view = None
    async with app.run_test() as pilot:
        editor = await _open_editor(app, pilot)
        assert {row.feature_id for row in editor.sorted_features} == {2, 3, 7, 15}
        assert all("composite features" not in line.plain for line in editor._build_feature_lines())
        old_magnitude = editor.sorted_features[0].magnitude
        await pilot.press("right")
        assert editor.sorted_features[0].magnitude != old_magnitude
        await pilot.press("escape")
        await pilot.pause()
        assert not app._overlay_open
