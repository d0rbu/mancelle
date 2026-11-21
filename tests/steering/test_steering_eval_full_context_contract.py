"""Steering eval public contract: encoder-decoder full-context only."""

from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch as t

import steering.eval.steering_eval as se
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.steering_eval import _evaluate_feature_at_magnitude, run_steering_eval
from steering.eval.target_conditioning import TargetConditioningMode


def test_run_steering_eval_rejects_decoder_only_baselines(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decoder-only unconditioned baselines"):
        run_steering_eval(
            config=object(),  # rejected before config is inspected  # ty: ignore[invalid-argument-type]
            judge=object(),  # ty: ignore[invalid-argument-type]
            output_dir=tmp_path,
            model=object(),  # ty: ignore[invalid-argument-type]
            tokenizer=object(),
            device=t.device("cpu"),
            context_provider=object(),  # ty: ignore[invalid-argument-type]
            target_mode=TargetConditioningMode.CURRENT,
        )


def test_run_steering_eval_has_no_skip_conditioning_escape_hatch() -> None:
    assert "skip_conditioning" not in inspect.signature(run_steering_eval).parameters


def test_run_steering_eval_requires_context_provider_argument() -> None:
    param = inspect.signature(run_steering_eval).parameters["context_provider"]
    assert param.default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Regression: next-segment target conditioning must be honored on the
# standalone steering-eval driver path (bug: target_mode was read from a model
# attribute that is never set outside training_progression, so every checkpoint
# silently ran with CURRENT routing — including next-segment-trained ones).
# ---------------------------------------------------------------------------


def test_run_steering_eval_requires_target_mode_argument() -> None:
    """target_mode must be a required parameter — no silent default.

    A default here is exactly what reintroduces the bug: callers that forget
    to resolve next-segment timing from checkpoint metadata would silently get
    CURRENT routing.
    """
    param = inspect.signature(run_steering_eval).parameters["target_mode"]
    assert param.default is inspect.Parameter.empty


def _isolated_config() -> se.SteeringEvalConfig:
    return se.SteeringEvalConfig(
        features=[se.FeatureSpec(id=0, description="trait", confidence=3)],
        magnitude_levels=[0],
    )


def test_run_steering_eval_threads_target_mode_into_isolated_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def _fake_isolated(*_args: object, target_mode: object, **_kwargs: object) -> None:
        seen["target_mode"] = target_mode

    monkeypatch.setattr(se, "_run_isolated_eval", _fake_isolated)
    # __new__ bypasses heavy __init__ while still satisfying isinstance().
    model = EncoderDecoderGPT.__new__(EncoderDecoderGPT)

    run_steering_eval(
        _isolated_config(),
        judge=object(),  # ty: ignore[invalid-argument-type]
        output_dir=tmp_path,
        model=model,
        tokenizer=object(),
        device=t.device("cpu"),
        context_provider=object(),  # ty: ignore[invalid-argument-type]
        target_mode=TargetConditioningMode.NEXT,
    )

    assert seen["target_mode"] is TargetConditioningMode.NEXT


def test_run_steering_eval_threads_target_mode_into_distractor_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def _fake_distractor(*_args: object, target_mode: object, **_kwargs: object) -> None:
        seen["target_mode"] = target_mode

    monkeypatch.setattr(se, "_run_distractor_eval", _fake_distractor)
    model = EncoderDecoderGPT.__new__(EncoderDecoderGPT)
    config = se.SteeringEvalConfig(
        features=[se.FeatureSpec(id=0, description="trait", confidence=3)],
        magnitude_levels=[0],
        distractors=se.DistractorConfig(stats_path="unused.safetensors"),
    )

    run_steering_eval(
        config,
        judge=object(),  # ty: ignore[invalid-argument-type]
        output_dir=tmp_path,
        model=model,
        tokenizer=object(),
        device=t.device("cpu"),
        context_provider=object(),  # ty: ignore[invalid-argument-type]
        target_mode=TargetConditioningMode.CURRENT,
    )

    assert seen["target_mode"] is TargetConditioningMode.CURRENT


def test_evaluate_feature_at_magnitude_uses_passed_target_mode_not_model_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leaf builds the engine with the threaded target_mode.

    The model object here intentionally has NO ``next_segment_conditioning_offset``
    attribute (a freshly loaded EncoderDecoderGPT does not). The old code read
    ``getattr(model, "next_segment_conditioning_offset", False)`` and so always
    produced CURRENT regardless of the checkpoint. The conditioned and baseline
    engines must both be built with the explicitly passed NEXT mode.
    """
    captured: list[object] = []

    class _FakeEngine:
        def generate(self, *_args: object, **_kwargs: object) -> str:
            return "generated text"

    def _fake_build(
        _model: object,
        _tokenizer: object,
        _device: object,
        *,
        features: object,
        context_provider: object,
        target_mode: object,
    ) -> _FakeEngine:
        captured.append(target_mode)
        return _FakeEngine()

    monkeypatch.setattr(se, "build_engine_for_features", _fake_build)
    monkeypatch.setattr(
        se, "judge_pair", lambda *_a, **_k: se.PairwiseResult(verdict=se.Verdict.TIE)
    )

    config = se.SteeringEvalConfig(
        features=[se.FeatureSpec(id=0, description="trait", confidence=3)],
        generation=se.GenerationConfig(samples_per_condition=1, max_new_tokens=2),
        prompts=["hello"],
    )

    _evaluate_feature_at_magnitude(
        feature=config.features[0],
        magnitude=3,
        model=object(),  # deliberately lacks next_segment_conditioning_offset  # ty: ignore[invalid-argument-type]
        tokenizer=object(),
        device=t.device("cpu"),
        judge=object(),  # ty: ignore[invalid-argument-type]
        config=config,
        existing_mag_data=None,
        context_provider=object(),  # ty: ignore[invalid-argument-type]
        target_mode=TargetConditioningMode.NEXT,
    )

    assert captured, "build_engine_for_features was never called"
    assert all(mode is TargetConditioningMode.NEXT for mode in captured)


def test_steering_eval_driver_resolves_target_mode_from_checkpoint_meta() -> None:
    """The standalone driver must resolve next-segment timing from ``meta``.

    Structural guard (file-based AST, robust to import order): ``main`` must
    call ``resolve_next_segment_conditioning_offset`` and forward a
    ``target_mode`` keyword into ``run_steering_eval``. Without this the driver
    cannot honor a next-segment-trained checkpoint.
    """
    import scripts.eval.steering_eval as driver

    tree = ast.parse(Path(driver.__file__).read_text(encoding="utf-8"))
    main_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called_names = {
        node.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "resolve_next_segment_conditioning_offset" in called_names

    run_calls = [
        node
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_steering_eval"
    ]
    assert run_calls, "driver never calls run_steering_eval"
    assert any(
        kw.arg == "target_mode" for call in run_calls for kw in call.keywords
    ), "run_steering_eval must be called with a target_mode keyword"


# ---------------------------------------------------------------------------
# Behavioral proof (stronger than the AST guard above): drive the real
# ``scripts.eval.steering_eval.main`` with mocked collaborators and assert the
# ``target_mode`` value actually PASSED into ``run_steering_eval`` is NEXT when
# the checkpoint metadata declares next-segment conditioning. This exercises
# the live ``resolve_next_segment_conditioning_offset`` +
# ``TargetConditioningMode.from_next_segment_offset`` wiring end-to-end (NOT
# mocked), mirroring the mocking style in
# tests/scripts/test_harmfulness_bfloat16_cast.py::TestMainBfloat16CastLogic.
# ---------------------------------------------------------------------------


def _encdec_stub_model() -> EncoderDecoderGPT:
    """A real ``EncoderDecoderGPT`` instance (so ``isinstance`` passes) with
    only the attributes ``main`` touches, bypassing heavy ``__init__``."""
    model = EncoderDecoderGPT.__new__(EncoderDecoderGPT)
    model.encoder = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        config=types.SimpleNamespace(n_sae_features=64, num_buckets=4)
    )
    model.dec_config = types.SimpleNamespace(sequence_len=128)  # ty: ignore[invalid-assignment]
    return model


class TestSteeringEvalDriverTargetModeBehavioral:
    """Behavioral driver test: target_mode resolved from checkpoint meta."""

    @patch("scripts.eval.steering_eval.run_steering_eval")
    @patch("scripts.eval.steering_eval.PairwiseJudge")
    @patch("scripts.eval.steering_eval.JudgeClient")
    @patch("scripts.eval.steering_eval.build_context_feature_provider")
    @patch("scripts.eval.steering_eval.validate_config_against_model")
    @patch("scripts.eval.steering_eval.load_config")
    @patch("scripts.eval.steering_eval.load_model")
    @patch("scripts.eval.steering_eval.autodetect_device_type")
    @patch("scripts.eval.steering_eval.OmegaConf")
    @patch("scripts.eval.steering_eval.compose")
    @patch("scripts.eval.steering_eval.initialize_config_dir")
    def test_driver_passes_next_target_mode_from_meta(
        self,
        mock_init_cfg,
        mock_compose,
        mock_omegaconf,
        mock_autodetect,
        mock_load_model,
        mock_load_config,
        mock_validate,
        mock_build_provider,
        mock_judge_client,
        mock_pairwise_judge,
        mock_run_steering_eval,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts.eval.steering_eval import main

        # Required API key (main exits early without it).
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # cfg with .get delegating to attrs (mirrors the harmfulness pattern).
        cfg = MagicMock()
        cfg.steering_config = str(tmp_path / "steering.yaml")
        (tmp_path / "steering.yaml").write_text("features: []\n")
        cfg.source = "encdec"
        cfg.model_tag = "test"
        cfg.step = None
        cfg.output_dir = str(tmp_path / "out")
        cfg.segment_size = None
        cfg.feature_selection = None
        cfg.requests_per_minute = 100
        cfg.judge_cache_path = ""
        # 'auto' makes resolve_next_segment_conditioning_offset read meta.
        cfg.next_segment_conditioning_offset = "auto"
        cfg.conditioning = {"context_cache_dir": str(tmp_path / "cache")}
        cfg.get = lambda key, default=None: getattr(cfg, key, default)
        mock_compose.return_value = cfg

        mock_init_cfg.return_value.__enter__ = MagicMock()
        mock_init_cfg.return_value.__exit__ = MagicMock()
        mock_omegaconf.to_yaml.return_value = "source: encdec"

        mock_autodetect.return_value = "cpu"

        model = _encdec_stub_model()
        # Checkpoint metadata declares next-segment conditioning truthy.
        meta = {"user_config": {"next_segment_conditioning_offset": True}}
        mock_load_model.return_value = (model, MagicMock(), meta)

        # load_config returns a config object with the attrs main logs.
        steering_cfg = MagicMock()
        steering_cfg.features = []
        steering_cfg.magnitude_levels = [0]
        steering_cfg.generation.samples_per_condition = 1
        steering_cfg.judge.model = "judge-model"
        mock_load_config.return_value = steering_cfg

        sentinel_provider = object()
        mock_build_provider.return_value = sentinel_provider

        main()

        # The driver must have called run_steering_eval exactly once.
        assert mock_run_steering_eval.call_count == 1
        _, kwargs = mock_run_steering_eval.call_args
        # The resolved target_mode must be NEXT (from meta via the REAL
        # resolve_next_segment_conditioning_offset, not mocked).
        assert kwargs["target_mode"] is TargetConditioningMode.NEXT
        # And the real model + provider are threaded through.
        assert kwargs["model"] is model
        assert kwargs["context_provider"] is sentinel_provider

    @patch("scripts.eval.steering_eval.run_steering_eval")
    @patch("scripts.eval.steering_eval.PairwiseJudge")
    @patch("scripts.eval.steering_eval.JudgeClient")
    @patch("scripts.eval.steering_eval.build_context_feature_provider")
    @patch("scripts.eval.steering_eval.validate_config_against_model")
    @patch("scripts.eval.steering_eval.load_config")
    @patch("scripts.eval.steering_eval.load_model")
    @patch("scripts.eval.steering_eval.autodetect_device_type")
    @patch("scripts.eval.steering_eval.OmegaConf")
    @patch("scripts.eval.steering_eval.compose")
    @patch("scripts.eval.steering_eval.initialize_config_dir")
    def test_driver_passes_current_target_mode_when_meta_disables_it(
        self,
        mock_init_cfg,
        mock_compose,
        mock_omegaconf,
        mock_autodetect,
        mock_load_model,
        mock_load_config,
        mock_validate,
        mock_build_provider,
        mock_judge_client,
        mock_pairwise_judge,
        mock_run_steering_eval,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Symmetric control: meta=false ⇒ CURRENT, so the NEXT assertion above
        is load-bearing rather than vacuously true."""
        from scripts.eval.steering_eval import main

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        cfg = MagicMock()
        cfg.steering_config = str(tmp_path / "steering.yaml")
        (tmp_path / "steering.yaml").write_text("features: []\n")
        cfg.source = "encdec"
        cfg.model_tag = "test"
        cfg.step = None
        cfg.output_dir = str(tmp_path / "out")
        cfg.segment_size = None
        cfg.feature_selection = None
        cfg.requests_per_minute = 100
        cfg.judge_cache_path = ""
        cfg.next_segment_conditioning_offset = "auto"
        cfg.conditioning = {"context_cache_dir": str(tmp_path / "cache")}
        cfg.get = lambda key, default=None: getattr(cfg, key, default)
        mock_compose.return_value = cfg

        mock_init_cfg.return_value.__enter__ = MagicMock()
        mock_init_cfg.return_value.__exit__ = MagicMock()
        mock_omegaconf.to_yaml.return_value = "source: encdec"
        mock_autodetect.return_value = "cpu"

        model = _encdec_stub_model()
        meta = {"user_config": {"next_segment_conditioning_offset": False}}
        mock_load_model.return_value = (model, MagicMock(), meta)

        steering_cfg = MagicMock()
        steering_cfg.features = []
        steering_cfg.magnitude_levels = [0]
        steering_cfg.generation.samples_per_condition = 1
        steering_cfg.judge.model = "judge-model"
        mock_load_config.return_value = steering_cfg
        mock_build_provider.return_value = object()

        main()

        assert mock_run_steering_eval.call_count == 1
        _, kwargs = mock_run_steering_eval.call_args
        assert kwargs["target_mode"] is TargetConditioningMode.CURRENT
