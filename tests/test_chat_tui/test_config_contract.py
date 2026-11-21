"""Hydra and startup-wiring contracts for the chat TUI entry point."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

from scripts.chat_tui import setup as chat_setup
from steering.segmentation import SegmentMode

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "serving")
_CONFIG_KEYS = sorted(
    {
        "description_cache_path",
        "descriptions_json_path",
        "deterministic_rewind",
        "device",
        "dtype",
        "eager_extraction",
        "feature_statistics_path",
        "max_new_tokens_per_turn",
        "neuronpedia_model_id",
        "neuronpedia_sae_source",
        "prepared_dir",
        "pretrain_meta_path",
        "seed",
        "self_cond_segment_size",
        "signal_cache_dir",
        "target_meta_path",
        "temperature",
        "top_k",
    }
)


def test_setup_file_sha256_hashes_raw_bytes_across_multiple_chunks(tmp_path: Path) -> None:
    payload = b"a" * (1024 * 1024) + b"rank-policy"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)

    assert chat_setup._file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_chat_tui_config_exact_key_set_and_defaults(hydra_compose) -> None:
    cfg = hydra_compose(
        _CONFIG_DIR,
        "chat_tui",
        ["target_meta_path=/tmp/target-meta.json"],
    )

    assert sorted(cfg.keys()) == _CONFIG_KEYS
    assert cfg.target_meta_path == "/tmp/target-meta.json"
    assert cfg.pretrain_meta_path is None
    assert cfg.feature_statistics_path == (
        "out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/"
        "base/context_independent/no_bos/feature_statistics.safetensors"
    )
    assert cfg.prepared_dir == "out/axbench/prepared"
    assert cfg.signal_cache_dir == "out/chat_tui/signal_cache"
    assert cfg.description_cache_path == "out/neuronpedia_cache.yaml"
    assert cfg.descriptions_json_path == ""
    assert cfg.neuronpedia_model_id == ""
    assert cfg.neuronpedia_sae_source == ""
    assert cfg.self_cond_segment_size == 256
    assert cfg.max_new_tokens_per_turn == 256
    assert cfg.temperature == 0.6
    assert cfg.top_k == 50
    assert cfg.seed == 42
    assert "masked_threshold_scale" not in cfg
    assert cfg.deterministic_rewind is False
    assert cfg.device == "cuda"
    assert cfg.dtype == "bfloat16"
    assert cfg.eager_extraction is False


def test_chat_tui_target_meta_path_is_mandatory(hydra_compose) -> None:
    cfg = hydra_compose(_CONFIG_DIR, "chat_tui")

    with pytest.raises(MissingMandatoryValue, match="target_meta_path"):
        _ = cfg.target_meta_path


def test_chat_tui_hydra_runtime_defaults_are_pinned() -> None:
    raw = OmegaConf.load(Path(_CONFIG_DIR) / "chat_tui.yaml")

    assert sorted(raw.keys()) == sorted([*_CONFIG_KEYS, "hydra"])
    assert raw.hydra.run.dir == "."
    assert raw.hydra.output_subdir is None
    assert raw.hydra.job.chdir is False


@pytest.mark.parametrize(
    "segment_mode", [SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens]
)
def test_build_app_smoke_preserves_startup_order_and_lazy_extraction(
    monkeypatch: pytest.MonkeyPatch,
    segment_mode: SegmentMode,
) -> None:
    from scripts.chat_tui import __main__ as entrypoint

    events: list[str] = []
    stack_calls: list[None] = []
    tokenizer = object()
    model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=1024))
    setup = SimpleNamespace(
        target_meta_path=Path("/checkpoints/meta_000001.json"),
        segment_mode=segment_mode,
        turn_self_cond_segment_size_rustbpe=37
        if segment_mode is SegmentMode.turn_delimited_fixed_tokens
        else None,
        feature_vocab=None,
        n_sae_features=8,
        output_negative_selection=object(),
        output_selection=SimpleNamespace(min_k=1),
        feature_mask=None,
        consolidation=None,
    )
    session = object()

    class ExtractionStub:
        def extraction_stack(self) -> tuple[object, object]:
            stack_calls.append(None)
            return object(), object()

    class AppStub:
        def __init__(self, **kwargs: Any) -> None:
            events.append("app")
            self.kwargs = kwargs
            # ChatApp adopts the injected already-started thread via its
            # constructor (owns-if-None); the stub mirrors that contract.
            self._model_thread = kwargs.get("model_thread")

        @property
        def model_thread(self):
            return self._model_thread

        def call_from_thread(self, callback) -> None:
            callback()

    signal_source = object()
    signal_kwargs: dict[str, Any] = {}

    pretrain_path = (
        "/pretrain.json" if segment_mode is SegmentMode.turn_delimited_fixed_tokens else None
    )

    def fake_setup(*_args: Any, pretrain_meta_path: Path | None) -> Any:
        assert pretrain_meta_path == (None if pretrain_path is None else Path(pretrain_path))
        events.append("setup")
        return setup

    def fake_model_load(
        actual_setup: Any,
        device: str,
        *,
        dtype: str,
    ) -> tuple[Any, object]:
        assert actual_setup is setup
        assert device == "cpu"
        assert dtype == "bfloat16"
        events.append("model")
        return model, tokenizer

    def fake_session(**kwargs: Any) -> object:
        assert kwargs["setup"] is setup
        assert kwargs["model"] is model
        assert kwargs["config"].self_cond_segment_size == (
            37 if segment_mode is SegmentMode.turn_delimited_fixed_tokens else 256
        )
        events.append("session")
        return session

    def fake_config_builder(*args: Any, **kwargs: Any) -> object:
        assert args[0] is setup
        assert kwargs["segment_mode"] is segment_mode
        events.append("builder")
        return object()

    def fake_provider(actual_model: Any, _config: Any, device: Any) -> object:
        assert actual_model is model
        assert str(device) == "cpu"
        events.append("probe")
        return object()

    def fake_signals(**kwargs: Any) -> object:
        events.append("signals")
        signal_kwargs.update(kwargs)
        return signal_source

    catalog = object()
    catalog_calls: list[object] = []

    def fake_catalog_load(actual_tokenizer: object) -> object:
        catalog_calls.append(actual_tokenizer)
        return catalog

    monkeypatch.setattr(entrypoint, "load_chat_setup", fake_setup)
    monkeypatch.setattr(entrypoint, "load_chat_target_model", fake_model_load)
    monkeypatch.setattr(entrypoint, "GemmaExtractionService", lambda *_args: ExtractionStub())
    monkeypatch.setattr(entrypoint, "TurnGemmaExtractionService", lambda *_args: ExtractionStub())
    monkeypatch.setattr(entrypoint, "ChatSession", fake_session)
    monkeypatch.setattr(entrypoint, "TurnChatSession", fake_session)
    monkeypatch.setattr(entrypoint, "build_score_policy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(entrypoint, "build_session_self_cond_config", fake_config_builder)
    monkeypatch.setattr(entrypoint, "SelfConditionedFeatureProvider", fake_provider)
    monkeypatch.setattr(entrypoint, "AxbenchSignalsSource", fake_signals)
    monkeypatch.setattr(entrypoint, "ChatApp", AppStub)
    monkeypatch.setattr(entrypoint.ConvoCatalog, "load", fake_catalog_load)

    cfg: DictConfig = OmegaConf.create(
        {
            "target_meta_path": "/target.json",
            "pretrain_meta_path": pretrain_path,
            "feature_statistics_path": "/stats.safetensors",
            "prepared_dir": "/prepared",
            "signal_cache_dir": "/signals",
            "description_cache_path": "/descriptions.yaml",
            "descriptions_json_path": "",
            "neuronpedia_model_id": "",
            "neuronpedia_sae_source": "",
            "self_cond_segment_size": 256,
            "max_new_tokens_per_turn": 256,
            "temperature": 0.6,
            "top_k": 50,
            "seed": 42,
            "deterministic_rewind": False,
            "device": "cpu",
            "dtype": "bfloat16",
            "eager_extraction": False,
        }
    )

    app, model_thread = entrypoint.build_app(cfg)
    try:
        assert events == ["setup", "model", "builder", "probe", "session", "signals", "app"]
        assert app.model_thread is model_thread
        assert model_thread.is_alive
        assert stack_calls == []
        assert signal_kwargs["model_thread"] is model_thread
        assert app.kwargs["session"] is session
        assert app.kwargs["signals_source"] is signal_source
        # The empty descriptions_json_path default leaves the steer menu without
        # descriptions; the operator opts in with a generated JSON.
        assert app.kwargs["description_service"] is None
        assert app.kwargs["deterministic_rewind"] is False
        assert app.kwargs["catalog_loader"]() is catalog
        assert catalog_calls == [tokenizer]
    finally:
        model_thread.shutdown()
