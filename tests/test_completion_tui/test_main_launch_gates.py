from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from errors import FeatureClustersError, GradientRankingUnsupportedError
from scripts.completion_tui.__main__ import LaunchOptions, _load_and_launch
from scripts.completion_tui.discovery import TUIConfig
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.overlay_io import OverlayIncompleteError
from steering.segmentation import SegmentMode
from steering.stats_types import PerFeatureStat
from tests.statistics_doubles import fixed_statistics_double

N_SAE = 16


class _StartupTokenizer:
    """Minimal reversible tokenizer contract used by launch-gate tests."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _stats(segment_mode: SegmentMode = SegmentMode.fixed_tokens):
    percentiles = {f"p{k}": k / 100 for k in range(1, 100)}
    per_feature_stats = {
        str(feature_id): PerFeatureStat(
            count=10,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=1.0,
            median_mag=0.5,
            percentiles=dict(percentiles),
        )
        for feature_id in range(N_SAE)
    }
    return fixed_statistics_double(
        segment_mode=segment_mode,
        per_feature_stats=per_feature_stats,
        segment_size=8,
        context_window=8,
        layer=0,
        n_sae_features=N_SAE,
    )


def _cluster_metadata() -> dict:
    definition = FeatureClusterDefinition(clusters=((1, 3, 5),))
    resolved = ResolvedFeatureClusters(
        source_path=Path("clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=N_SAE),
    )
    return feature_clusters_checkpoint_metadata(resolved)


def _selected(tmp_path: Path, meta: dict) -> TUIConfig:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "meta_000001.json").write_text(json.dumps(meta), encoding="utf-8")
    return TUIConfig(
        model_tag="test",
        checkpoint_dir=str(checkpoint_dir),
        nested_path="",
        step=1,
        feature_dir=str(tmp_path / "features"),
        parquet_dir=str(tmp_path / "parquet"),
        segment_size=8,
        num_buckets=4,
        has_negatives=False,
        label="test",
    )


def _options() -> LaunchOptions:
    return LaunchOptions(
        device_type="cpu",
        dtype="float32",
        temperature=0.7,
        top_k=50,
        max_new_tokens=16,
        neuronpedia_model_id="",
        neuronpedia_sae_source="",
        description_cache_path="out/neuronpedia_cache.yaml",
    )


def _patch_startup(monkeypatch):
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.compute_init",
        lambda _device_type: (0, 0, 0, 0, t.device("cpu")),
    )
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.get_tokenizer",
        _StartupTokenizer,
    )
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.find_statistics_path",
        lambda _feature_dir: Path("stats.safetensors"),
    )
    monkeypatch.setattr("scripts.completion_tui.__main__.load_statistics", lambda _path: _stats())


def _clustered_meta(feature_selection: dict | None) -> dict:
    meta = _cluster_metadata()
    meta["user_config"] = {
        "feature_selection": feature_selection,
        "num_buckets": 4,
        "segment_size": 8,
    }
    return meta


def test_lazy_launch_rejects_turn_checkpoint_before_feature_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = _clustered_meta({"threshold_type": "feature_percentile", "threshold": 0.9})
    meta["sft_segment_mode"] = "turn_delimited_fixed_tokens"
    _patch_startup(monkeypatch)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.load_statistics",
        lambda _path: pytest.fail("feature statistics reached"),
    )

    with pytest.raises(ValueError, match=r"completion TUI.*turn_delimited_fixed_tokens"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_lazy_partial_cluster_metadata_fails_before_document_loading(tmp_path: Path, monkeypatch):
    meta = _clustered_meta({"threshold_type": "feature_percentile", "threshold": 0.9})
    meta.pop("feature_clusters_canonical")
    _patch_startup(monkeypatch)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.DocumentLoader",
        lambda *args, **kwargs: pytest.fail("DocumentLoader reached"),
    )
    with pytest.raises(FeatureClustersError, match="incomplete"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_lazy_clustered_non_splice_delta_rejected_before_document_loading(
    tmp_path: Path, monkeypatch
):
    meta = _clustered_meta({"threshold_type": "feature_percentile", "threshold": 0.9})
    meta["user_config"]["feature_delta"] = {"mode": "prev_segment"}
    _patch_startup(monkeypatch)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.DocumentLoader",
        lambda *args, **kwargs: pytest.fail("DocumentLoader reached"),
    )
    with pytest.raises(ValueError, match="feature_delta"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_lazy_clustered_raw_threshold_rejected_before_document_loading(tmp_path: Path, monkeypatch):
    meta = _clustered_meta({"threshold_type": "segment_topk", "threshold": 2})
    _patch_startup(monkeypatch)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.DocumentLoader",
        lambda *args, **kwargs: pytest.fail("DocumentLoader reached"),
    )
    with pytest.raises(ValueError, match="segment_topk"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_lazy_clustered_gradient_uses_existing_live_selection_guard(tmp_path: Path, monkeypatch):
    meta = _clustered_meta(
        {
            "distribution": "geometric",
            "mean": 5.0,
            "mode": "top_k_percentile",
            "gradient": {"mode": "future_half_split"},
        }
    )
    _patch_startup(monkeypatch)
    with pytest.raises(GradientRankingUnsupportedError, match="offline-only"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_lazy_clustered_missing_percentile_overlay_fails_loud(tmp_path: Path, monkeypatch):
    meta = _clustered_meta({"threshold_type": "feature_percentile", "threshold": 0.9})
    _patch_startup(monkeypatch)

    def _missing_overlay(*args, **kwargs):
        assert kwargs.get("percentile_checksum")
        raise OverlayIncompleteError("percentile overlay missing")

    monkeypatch.setattr("scripts.completion_tui.__main__.DocumentLoader", _missing_overlay)
    with pytest.raises(OverlayIncompleteError, match="overlay missing"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_conditions_within_vocab_stance_loads_compacted_checkpoint():
    vocab = FeatureVocab.from_mask(FeatureMask(FeatureMaskMode.blacklist, (2,)), N_SAE)
    assert vocab is not None
    EvalConditioning.conditions_within_vocab("completion_tui").assert_loadable(
        vocab, checkpoint_dir="checkpoint"
    )


def test_main_source_declares_conditions_within_vocab():
    source = (Path(__file__).parents[2] / "scripts" / "completion_tui" / "__main__.py").read_text(
        encoding="utf-8"
    )
    assert 'EvalConditioning.conditions_within_vocab("completion_tui")' in source
    assert 'EvalConditioning.deferred("completion_tui")' not in source


def test_lazy_launch_rejects_turn_feature_statistics_before_contract_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: dropping the feature-artifact guard in _load_and_launch, which
    # would run a fixed checkpoint against turn-delimited feature statistics.
    meta = _clustered_meta({"threshold_type": "feature_percentile", "threshold": 0.9})
    _patch_startup(monkeypatch)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.load_statistics",
        lambda _path: _stats(SegmentMode.turn_delimited_fixed_tokens),
    )
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.resolve_checkpoint_feature_contract",
        lambda *_args, **_kwargs: pytest.fail("feature contract resolved for a turn artifact"),
    )

    with pytest.raises(ValueError, match=r"completion TUI.*turn_delimited_fixed_tokens"):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())


def test_lazy_model_build_rejects_turn_checkpoint_before_generation_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: dropping the guard inside the lazy _build_manager closure, which
    # re-reads checkpoint metadata that startup never authenticated.
    meta = _clustered_meta({"threshold_type": "feature_percentile", "threshold": 0.9})
    _patch_startup(monkeypatch)
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.DocumentLoader",
        lambda *_args, **_kwargs: SimpleNamespace(available_doc_ids=list),
    )
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.build_encdec_model",
        lambda *_args, **_kwargs: (
            object(),
            object(),
            {**meta, "sft_segment_mode": "turn_delimited_fixed_tokens"},
        ),
    )
    # _build_manager is only stored by the real LazyGenerationManager; call it
    # eagerly so the lazy branch's own guard is the one under test.
    monkeypatch.setattr(
        "scripts.completion_tui.__main__.LazyGenerationManager",
        lambda builder: builder(),
    )

    with pytest.raises(
        ValueError,
        match=r"completion TUI lazy model load.*turn_delimited_fixed_tokens",
    ):
        _load_and_launch(_selected(tmp_path, meta), lazy=True, options=_options())
