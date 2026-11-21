from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from errors import FeatureMaskError
from scripts.completion_tui.__main__ import _load_and_launch
from scripts.completion_tui.app import CompletionApp
from scripts.completion_tui.checkpoint_contract import resolve_checkpoint_feature_contract
from scripts.completion_tui.data_loader import TUIFeatureSamplingConfig
from scripts.completion_tui.generation import (
    GenerationManager,
    LazyGenerationManager,
)
from scripts.completion_tui.state import (
    DocumentData,
    FeatureCondition,
    Segment,
    SegmentClusterEvidence,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt
from tests.test_completion_tui.test_main_launch_gates import (
    N_SAE,
    _cluster_metadata,
    _options,
    _patch_startup,
    _selected,
)


class _StubModel:
    def __init__(self, feature_vocab: FeatureVocab | None) -> None:
        self.feature_vocab = feature_vocab
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(n_sae_features=N_SAE),
        )
        self.dtype: object | None = None

    def to(self, *, dtype: object) -> _StubModel:
        self.dtype = dtype
        return self


class _RecordingDocumentLoader:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def available_doc_ids(self) -> list[int]:
        return [7]

    def load_document(self, doc_idx: int, num_segments: int | None = None) -> DocumentData:
        del num_segments
        sampling_config = self.kwargs["sampling_config"]
        clustered = sampling_config.consolidation is not None
        features = [FeatureCondition(feature_id=1, magnitude=1)] if clustered else []
        segment = Segment(
            segment_idx=0,
            char_start=0,
            char_end=5,
            color_idx=0,
            features=features,
            id="launch-segment",
        )
        evidence = (
            {
                0: SegmentClusterEvidence(
                    fingerprint=frozenset({1}),
                    evidence={1: {1: 0.5, 3: 0.75}},
                )
            }
            if clustered
            else {}
        )
        return DocumentData(
            doc_idx=doc_idx,
            text="hello",
            segments=[segment],
            total_segments=1,
            evidence_by_segment_index=evidence,
        )

    def total_segments(self, doc_idx: int) -> int:
        del doc_idx
        return 1


def _launch_meta(
    *,
    mask: FeatureMask | None,
    clustered: bool,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "user_config": {
            "feature_selection": {
                "threshold_type": "feature_percentile",
                "threshold": 0.9,
            },
            "num_buckets": 4,
            "segment_size": 8,
        }
    }
    if mask is not None:
        meta["feature_mask_canonical"] = mask.canonical_dict()
    if clustered:
        meta.update(_cluster_metadata())
    return meta


def _patch_positive_runtime(monkeypatch):
    loaders: list[_RecordingDocumentLoader] = []
    launched_apps: list[CompletionApp] = []
    launch_kwargs: list[dict[str, Any]] = []

    def _loader_factory(*args: Any, **kwargs: Any) -> _RecordingDocumentLoader:
        loader = _RecordingDocumentLoader(*args, **kwargs)
        loaders.append(loader)
        return loader

    original_init = CompletionApp.__init__

    def _recording_init(self, *args: Any, **kwargs: Any) -> None:
        launch_kwargs.append(dict(kwargs))
        original_init(self, *args, **kwargs)

    def _recording_run(self) -> None:
        launched_apps.append(self)

    monkeypatch.setattr(
        "scripts.completion_tui.__main__.DocumentLoader",
        _loader_factory,
    )
    monkeypatch.setattr(CompletionApp, "__init__", _recording_init)
    monkeypatch.setattr(CompletionApp, "run", _recording_run)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.enable_transparent_background",
        lambda app: None,
    )
    return loaders, launched_apps, launch_kwargs


@pytest.mark.parametrize(
    ("mode", "masked", "clustered"),
    [
        ("clustered-masked", True, True),
        ("clustered-unmasked", False, True),
        ("masked-unclustered", True, False),
        ("unmasked-unclustered", False, False),
    ],
)
def test_lazy_positive_launch_modes_build_matching_contracts(
    mode: str,
    masked: bool,
    clustered: bool,
    tmp_path: Path,
    monkeypatch,
) -> None:
    del mode
    mask = FeatureMask(FeatureMaskMode.blacklist, (2,)) if masked else None
    meta = _launch_meta(mask=mask, clustered=clustered)
    selected = _selected(tmp_path, meta)
    meta_path = Path(selected.checkpoint_dir) / "meta_000001.json"
    _patch_startup(monkeypatch)
    loaders, launched_apps, launch_kwargs = _patch_positive_runtime(monkeypatch)

    launch_contract = resolve_checkpoint_feature_contract(
        meta,
        n_sae_features=PositiveInt.parse(N_SAE),
        meta_path=str(meta_path),
    )
    expected_vocab = effective_feature_vocab(
        launch_contract.consolidation if clustered else mask, N_SAE
    )
    model = _StubModel(expected_vocab)
    build_calls: list[dict[str, Any]] = []

    def _build_model(*args: Any, **kwargs: Any):
        build_calls.append(dict(kwargs))
        current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return model, SimpleNamespace(), current_meta

    monkeypatch.setattr(
        "scripts.completion_tui.__main__.build_encdec_model",
        _build_model,
    )

    _load_and_launch(selected, lazy=True, options=_options())

    assert len(loaders) == 1
    assert len(launched_apps) == 1
    assert len(launch_kwargs) == 1
    app = launched_apps[0]
    kwargs = launch_kwargs[0]
    assert app._sampling_config is kwargs["sampling_config"]
    sampling_config = kwargs["sampling_config"]
    assert isinstance(sampling_config, TUIFeatureSamplingConfig)
    assert sampling_config.feature_mask == mask
    assert (sampling_config.consolidation is not None) is clustered
    assert sampling_config.effective_buckets == (4 if clustered else None)

    loader = loaders[0]
    assert ("percentile_checksum" in loader.kwargs) is clustered
    if clustered:
        assert loader.kwargs["percentile_checksum"]
        assert loader.kwargs["skip_feature_mags"] is True
        assert loader.kwargs["compact_feature_ids"] is True
        assert kwargs["cluster_view"] is not None
        initial_evidence = kwargs["initial_segment_evidence"]
        assert set(initial_evidence) == {"launch-segment"}
        assert initial_evidence["launch-segment"].fingerprint == frozenset({1})
    else:
        assert kwargs["cluster_view"] is None
        assert kwargs["initial_segment_evidence"] == {}
        assert "skip_feature_mags" not in loader.kwargs
        assert "compact_feature_ids" not in loader.kwargs

    lazy_manager = kwargs["generation_manager"]
    assert isinstance(lazy_manager, LazyGenerationManager)
    manager = lazy_manager._factory()
    assert isinstance(manager, GenerationManager)
    assert manager.feature_mask == mask
    assert manager.consolidation is sampling_config.consolidation
    assert len(build_calls) == 1


def test_lazy_model_build_rereads_meta_and_rejects_mask_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launch_mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    meta = _launch_meta(mask=launch_mask, clustered=False)
    selected = _selected(tmp_path, meta)
    meta_path = Path(selected.checkpoint_dir) / "meta_000001.json"
    _patch_startup(monkeypatch)
    _, _, launch_kwargs = _patch_positive_runtime(monkeypatch)

    model = _StubModel(FeatureVocab.from_mask(launch_mask, N_SAE))

    def _build_model(*args: Any, **kwargs: Any):
        del args, kwargs
        current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return model, SimpleNamespace(), current_meta

    monkeypatch.setattr(
        "scripts.completion_tui.__main__.build_encdec_model",
        _build_model,
    )
    _load_and_launch(selected, lazy=True, options=_options())
    lazy_manager = launch_kwargs[0]["generation_manager"]
    assert isinstance(lazy_manager, LazyGenerationManager)

    changed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    changed_meta["feature_mask_canonical"] = FeatureMask(
        FeatureMaskMode.blacklist,
        (3,),
    ).canonical_dict()
    meta_path.write_text(json.dumps(changed_meta), encoding="utf-8")

    with pytest.raises(
        FeatureMaskError,
        match="feature_mask_canonical changed between launch and lazy model loading",
    ):
        lazy_manager._factory()
