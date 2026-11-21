"""Unit and integration tests for DCLM CORE evaluation on encoder-decoder models."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import yaml
from omegaconf import OmegaConf

# Add project root to Python path so steering/ and nanochat/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import steering.eval.dclm_eval as dclm_eval_module
from nanochat.checkpoint_manager import (
    build_encdec_model,
    build_model,
    find_largest_model,
    find_last_step,
)
from nanochat.common import get_base_dir
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.eval_conditioning import EvalConditioning
from steering.eval.dclm_eval import (
    ConfigResult,
    EvalReport,
    FeatureConfig,
    evaluate_dclm_core,
    run_evaluation,
    write_results_yaml,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer, _make_encdec, _make_gpt

# ---------------------------------------------------------------------------
# Checkpoint detection for integration tests
# ---------------------------------------------------------------------------

_base_dir = get_base_dir()
has_decoder_checkpoint = os.path.exists(os.path.join(_base_dir, "base_checkpoints"))
_encdec_dir = os.path.join(_base_dir, "conditional_encdec_checkpoints")
has_encdec_checkpoint = (
    os.path.exists(_encdec_dir)
    and any(f.startswith("model_") and f.endswith(".pt") for f in os.listdir(_encdec_dir))
    if os.path.exists(_encdec_dir)
    else False
)
has_gpu = t.cuda.is_available()

# ---------------------------------------------------------------------------
# Mock evaluate_core return value
# ---------------------------------------------------------------------------

MOCK_CORE_RESULT = {
    "results": {"task1": 0.5},
    "centered_results": {"task1": 0.3},
    "core_metric": 0.3,
}


class _SyntheticContextProvider:
    """Return one full-prefix feature segment for any CORE prompt."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def get(self, token_ids):
        token_ids = [int(tok) for tok in token_ids]
        if not token_ids:
            return None
        char_end = token_to_char_offsets(self._tokenizer, token_ids)[-1]
        if char_end <= 0:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


def _context_provider(tokenizer):
    return _SyntheticContextProvider(tokenizer)


# ===========================================================================
# Unit tests
# ===========================================================================


class TestFeatureConfigValidation:
    """Test FeatureConfig dataclass construction and field defaults."""

    def test_empty_dict_features(self) -> None:
        cfg = FeatureConfig(name="empty", features={})
        assert cfg.features == {}

    def test_valid_features(self) -> None:
        cfg = FeatureConfig(name="topk", features={10: 2, 20: 3})
        assert cfg.features == {10: 2, 20: 3}


class TestEvaluateDclmCore:
    """Test evaluate_dclm_core dispatch logic (evaluate_core is mocked)."""

    def test_context_provider_argument_is_required(self) -> None:
        param = inspect.signature(evaluate_dclm_core).parameters["context_provider"]
        assert param.default is inspect.Parameter.empty

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_decoder_only_calls_evaluate_core_directly(self, mock_eval) -> None:
        model = _make_gpt()
        tokenizer = MockTokenizer()
        result = evaluate_dclm_core(
            model,
            tokenizer,
            "cpu",
            context_provider=None,
        )
        # evaluate_core should be called with the GPT model directly
        call_args = mock_eval.call_args
        assert call_args[0][0] is model
        assert "core_metric" in result

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_decoder_only_rejects_features(self, mock_eval) -> None:
        model = _make_gpt()
        tokenizer = MockTokenizer()
        with pytest.raises(ValueError, match=r"[Dd]ecoder.only"):
            evaluate_dclm_core(
                model,
                tokenizer,
                "cpu",
                feature_conditions={0: 1, 5: 2},
                context_provider=None,
            )

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_encdec_requires_context_provider(self, mock_eval) -> None:
        model = _make_encdec()
        tokenizer = MockTokenizer()
        with pytest.raises(ValueError, match="context_provider"):
            evaluate_dclm_core(
                model,
                tokenizer,
                "cpu",
                context_provider=None,
            )
        mock_eval.assert_not_called()

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_encdec_wraps_model_with_context_provider(self, mock_eval) -> None:
        model = _make_encdec()
        tokenizer = MockTokenizer()
        result = evaluate_dclm_core(
            model,
            tokenizer,
            "cpu",
            context_provider=_context_provider(tokenizer),
        )
        call_args = mock_eval.call_args
        wrapper = call_args[0][0]
        assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
        assert call_args.kwargs["core_example_batch_size"] == 1
        assert "core_metric" in result

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_encdec_forwards_core_example_batch_size(self, mock_eval) -> None:
        evaluate_dclm_core(
            _make_encdec(),
            MockTokenizer(),
            "cpu",
            core_example_batch_size=7,
            context_provider=_context_provider(MockTokenizer()),
        )

        assert mock_eval.call_args.kwargs["core_example_batch_size"] == 7

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_encdec_with_features_wraps_model(self, mock_eval) -> None:
        model = _make_encdec()
        tokenizer = MockTokenizer()
        features = {0: 1, 5: 2}
        result = evaluate_dclm_core(
            model,
            tokenizer,
            "cpu",
            feature_conditions=features,
            context_provider=_context_provider(tokenizer),
        )
        call_args = mock_eval.call_args
        wrapper = call_args[0][0]
        assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
        # The wrapper's target segments should contain the supplied features
        assert wrapper._target_segments == [{0: 1, 5: 2}]
        assert "core_metric" in result


class TestRunEvaluation:
    """Test run_evaluation multi-config orchestration (evaluate_core is mocked)."""

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_multi_config_returns_all_results(self, mock_eval) -> None:
        model = _make_encdec()
        tokenizer = MockTokenizer()
        configs = [
            FeatureConfig(name="topk", features={0: 1, 5: 2}),
            FeatureConfig(name="safety_delta", features={2: 1}),
        ]
        report = run_evaluation(
            model,
            tokenizer,
            "cpu",
            configs=configs,
            context_provider=_context_provider(tokenizer),
        )

        assert isinstance(report, EvalReport)
        assert len(report.config_results) == 2
        assert report.config_results[0].name == "topk"
        assert report.config_results[1].name == "safety_delta"
        assert "topk" in report.core_scores
        assert "safety_delta" in report.core_scores

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_encdec_rejects_empty_static_config(self, mock_eval) -> None:
        model = _make_encdec()
        tokenizer = MockTokenizer()
        with pytest.raises(ValueError, match="context_only"):
            run_evaluation(
                model,
                tokenizer,
                "cpu",
                configs=[FeatureConfig(name="context_only", features={})],
                context_provider=_context_provider(tokenizer),
            )
        mock_eval.assert_not_called()

    @patch("steering.eval.dclm_eval.evaluate_core", return_value=MOCK_CORE_RESULT)
    def test_default_no_static_configs(self, mock_eval) -> None:
        model = _make_gpt()
        tokenizer = MockTokenizer()
        report = run_evaluation(
            model,
            tokenizer,
            "cpu",
            configs=None,
            context_provider=None,
        )

        assert report.config_results == []
        mock_eval.assert_not_called()


class TestYamlOutput:
    """Test YAML serialization of results."""

    def test_write_results_yaml(self, tmp_path: Path) -> None:
        report = EvalReport(
            model="test-model",
            evaluation="dclm-core",
            config_results=[
                ConfigResult(
                    name="context_only",
                    results={"task1": 0.5},
                    centered_results={"task1": 0.3},
                    core_metric=0.3,
                ),
            ],
        )
        output_path = tmp_path / "results.yaml"
        write_results_yaml(report, str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["model"] == "test-model"
        assert loaded["summary"]["configs_evaluated"] == 1


class _InactiveBucketSentinel(Exception):
    """Raised by patched inactive-bucket validators in entrypoint wiring tests."""


def test_main_calls_inactive_bucket_guard_for_encdec() -> None:
    cfg = OmegaConf.create(
        {
            "device_type": "cpu",
            "source": "encdec",
            "model_tag": "d12",
            "step": None,
            "feature_configs": "ignored",
            "max_per_task": 1,
            "segment_size": None,
            "output_dir": None,
        }
    )
    model = _make_encdec()
    meta = {"step": 0, "checkpoint_dir": "/tmp/checkpoint", "user_config": {}}

    def _boom(configs, *, num_buckets, checkpoint_meta, checkpoint_label="checkpoint"):
        del configs, checkpoint_label
        assert num_buckets == model.encoder.config.num_buckets
        assert checkpoint_meta is meta
        raise _InactiveBucketSentinel("guard called")

    with (
        patch.object(dclm_eval_module, "compute_init", return_value=(False, 0, 0, 1, "cpu")),
        patch.object(
            dclm_eval_module,
            "load_model",
            return_value=(model, MockTokenizer(), meta),
        ),
        patch.object(
            dclm_eval_module,
            "parse_feature_configs",
            return_value=[
                FeatureConfig(
                    name="mixed",
                    features={0: model.encoder.config.num_buckets - 1},
                )
            ],
        ),
        patch.object(dclm_eval_module, "require_shared_context_cache_dir"),
        patch.object(
            dclm_eval_module,
            "validate_inactive_bucket_feature_configs",
            side_effect=_boom,
        ),
        pytest.raises(_InactiveBucketSentinel, match="guard called"),
    ):
        dclm_eval_module.main.__wrapped__(cfg)


# ===========================================================================
# Integration tests (skipped unless checkpoints and GPU are available)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not has_decoder_checkpoint or not has_gpu,
    reason="Requires decoder checkpoint and GPU",
)
class TestBaselineReproduction:
    """Reproduce baseline CORE metric with a real decoder-only checkpoint."""

    def test_decoder_only_core_score(self) -> None:
        """d26 decoder-only model should achieve CORE ~0.246 (±0.05 for subsampled)."""
        device = t.device("cuda")
        checkpoints_dir = os.path.join(_base_dir, "base_checkpoints")
        model_tag = find_largest_model(checkpoints_dir)
        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
        step = find_last_step(checkpoint_dir)
        model, tokenizer, _meta = build_model(checkpoint_dir, step, device, "eval")

        with t.amp.autocast(device_type="cuda", dtype=t.bfloat16):
            result = evaluate_dclm_core(
                model,
                tokenizer,
                device,
                feature_conditions=None,
                max_per_task=100,
                context_provider=None,
            )
        expected_core = 0.246
        assert abs(result["core_metric"] - expected_core) < 0.05, (
            f"CORE score {result['core_metric']:.4f} too far from expected {expected_core}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not has_decoder_checkpoint or not has_gpu,
    reason="Requires decoder checkpoint and GPU",
)
class TestDeterminism:
    """Verify that CORE evaluation is deterministic across runs."""

    def test_same_config_twice_identical(self) -> None:
        """Same eval config run twice should produce identical scores."""
        device = t.device("cuda")
        checkpoints_dir = os.path.join(_base_dir, "base_checkpoints")
        model_tag = find_largest_model(checkpoints_dir)
        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
        step = find_last_step(checkpoint_dir)
        model, tokenizer, _meta = build_model(checkpoint_dir, step, device, "eval")

        with t.amp.autocast(device_type="cuda", dtype=t.bfloat16):
            r1 = evaluate_dclm_core(
                model,
                tokenizer,
                device,
                max_per_task=50,
                context_provider=None,
            )
            r2 = evaluate_dclm_core(
                model,
                tokenizer,
                device,
                max_per_task=50,
                context_provider=None,
            )

        assert r1["core_metric"] == r2["core_metric"]
        for task in r1["results"]:
            assert r1["results"][task] == r2["results"][task], f"Mismatch on {task}"


@pytest.mark.slow
@pytest.mark.skipif(
    not has_encdec_checkpoint or not has_gpu,
    reason="Requires encoder-decoder checkpoint and GPU",
)
class TestEncoderDecoderEval:
    """Evaluate an encoder-decoder checkpoint with full-context features."""

    def test_encdec_context_conditioned_reasonable_score(self) -> None:
        """Enc-dec with full-context conditioning should produce a reasonable CORE score."""
        device = t.device("cuda")
        step = find_last_step(_encdec_dir)
        model, tokenizer, _meta = build_encdec_model(
            _encdec_dir,
            step,
            device,
            "eval",
            eval_conditioning=EvalConditioning.deferred("test_dclm"),
        )
        result = evaluate_dclm_core(
            model,
            tokenizer,
            device,
            max_per_task=100,
            context_provider=_context_provider(tokenizer),
        )
        assert result["core_metric"] > -0.1, (
            f"CORE score {result['core_metric']:.4f} is suspiciously low"
        )

    def test_target_delta_conditioning_runs(self) -> None:
        """Target-delta conditioning runs on top of full prompt/context features."""
        device = t.device("cuda")
        step = find_last_step(_encdec_dir)
        model, tokenizer, _meta = build_encdec_model(
            _encdec_dir,
            step,
            device,
            "eval",
            eval_conditioning=EvalConditioning.deferred("test_dclm"),
        )
        r_feat = evaluate_dclm_core(
            model,
            tokenizer,
            device,
            feature_conditions={0: 2, 1: 3},
            max_per_task=50,
            context_provider=_context_provider(tokenizer),
        )
        assert "core_metric" in r_feat
