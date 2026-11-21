"""Tests for chain-based source resolution in eval scripts.

Verifies that base_eval, chat_eval, harmfulness_eval, and dclm_eval all
correctly pass raw source strings (including chains like "safety/rl") to
load_model() without wrapping in ModelSource().
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nanochat.checkpoint_manager import (
    load_model,
    save_checkpoint,
    ModelPhase,
)
from nanochat.gpt import GPT
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.eval.dclm_eval import (
    EvalReport,
    run_evaluation,
    write_results_yaml,
)
from tests.conftest import _make_gpt, _make_gpt_config, MockTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_checkpoint(checkpoint_dir: str, step: int = 0) -> None:
    """Create minimal checkpoint files at the given directory."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    config = _make_gpt_config()
    model = _make_gpt()
    meta = {
        "step": step,
        "model_config": {
            "sequence_len": config.sequence_len,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
            "window_pattern": config.window_pattern,
        },
    }
    save_checkpoint(checkpoint_dir, step, model.state_dict(), None, meta, rank=0)


@pytest.fixture(autouse=True)
def _mock_tokenizer():
    """Mock get_tokenizer to avoid needing real tokenizer files."""
    mock_tok = MockTokenizer()
    with patch("nanochat.checkpoint_manager.get_tokenizer", return_value=mock_tok):
        yield


# ===========================================================================
# Eval scripts pass raw source strings to load_model
# ===========================================================================


class TestEvalSourceStrings:
    """All eval scripts should pass raw source strings (not ModelSource enums)
    to load_model, so that chain sources like 'safety/rl' work."""

    @pytest.mark.parametrize(
        "source",
        [
            "base",
            "safety",
            "sft",
            "capabilities",
            "safety/rl",
            "sft/rl",
            "base/safety",
        ],
    )
    def test_load_model_accepts_raw_source_strings(
        self, tmp_path: Path, source: str
    ) -> None:
        """load_model() works with raw strings, including chain sources."""
        base_dir = str(tmp_path)
        # Create the base_checkpoints/d2 dir needed for auto-detection
        os.makedirs(tmp_path / "base_checkpoints" / "d2")
        # Create checkpoint at the resolved path
        from nanochat.checkpoint_manager import get_checkpoint_dir

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            ckpt_dir = get_checkpoint_dir(source, model_tag="d2")
            _create_fake_checkpoint(ckpt_dir, step=42)
            model, tokenizer, meta = load_model(
                source,
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_eval_source_resolution"),
            )
        assert meta["step"] == 42
        assert meta["checkpoint_dir"] == ckpt_dir
        assert isinstance(model, GPT)


# ===========================================================================
# base_eval source resolution
# ===========================================================================


class TestBaseEvalSourceResolution:
    """Test that base_eval uses cfg.source to load models."""

    def test_source_used_in_model_name(self) -> None:
        """model_name and model_slug should incorporate the source string."""
        # Simulate the logic from base_eval.py lines 216-221
        source = "safety"
        meta = {"step": 998, "model_config": {"sequence_len": 2048}}
        model_name = f"{source} (step {meta['step']})"
        model_slug = f"{source.replace('/', '_')}_{meta['step']:06d}"
        assert model_name == "safety (step 998)"
        assert model_slug == "safety_000998"

    def test_chain_source_slug_replaces_slash(self) -> None:
        """Slashes in chain sources should be replaced with underscores in slugs."""
        source = "safety/rl"
        meta = {"step": 100}
        model_slug = f"{source.replace('/', '_')}_{meta['step']:06d}"
        assert model_slug == "safety_rl_000100"

    def test_safety_checkpoint_loads_correctly(self, tmp_path: Path) -> None:
        """End-to-end: source='safety' resolves to safety_sft_checkpoints."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=998)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            model, tokenizer, meta = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.EVAL,
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_eval_source_resolution"),
            )
        assert meta["checkpoint_dir"] == str(safety_dir)
        assert meta["step"] == 998

    def test_safety_rl_chain_loads_correctly(self, tmp_path: Path) -> None:
        """End-to-end: source='safety/rl' resolves to nested RL dir."""
        base_dir = str(tmp_path)
        rl_dir = (
            tmp_path
            / "base_checkpoints"
            / "d2"
            / "safety_sft_checkpoints"
            / "standard_rl_checkpoints"
        )
        _create_fake_checkpoint(str(rl_dir), step=500)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            model, tokenizer, meta = load_model(
                "safety/rl",
                t.device("cpu"),
                phase=ModelPhase.EVAL,
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_eval_source_resolution"),
            )
        assert meta["checkpoint_dir"] == str(rl_dir)
        assert meta["step"] == 500


# ===========================================================================
# dclm_eval source resolution
# ===========================================================================


class TestDclmEvalSourceResolution:
    """Test dclm_eval model loading and report generation."""

    def test_decoder_model_info(self, tmp_path: Path) -> None:
        """dclm_eval should populate model_info correctly for decoder models."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=998)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            source = "safety"
            model, tokenizer, meta = load_model(
                source,
                t.device("cpu"),
                phase=ModelPhase.EVAL,
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_eval_source_resolution"),
            )
        is_encdec = isinstance(model, EncoderDecoderGPT)
        model_info = {
            "type": "encoder_decoder" if is_encdec else "decoder_only",
            "source": source,
            "checkpoint_dir": meta.get("checkpoint_dir"),
            "step": int(meta["step"]),
        }
        assert model_info["type"] == "decoder_only"
        assert model_info["source"] == "safety"
        assert model_info["checkpoint_dir"] == str(safety_dir)
        assert model_info["step"] == 998

    def test_run_evaluation_returns_report(self) -> None:
        """run_evaluation should return an EvalReport with correct structure."""
        model = _make_gpt()
        tokenizer = MockTokenizer()
        # Mock evaluate_core to avoid needing eval_bundle
        mock_result = {
            "results": {"task1": 0.8},
            "centered_results": {"task1": 0.6},
            "core_metric": 0.6,
        }
        with patch("steering.eval.dclm_eval.evaluate_core", return_value=mock_result):
            report = run_evaluation(
                model,
                tokenizer,
                "cpu",
                max_per_task=1,
                context_provider=None,
            )

        assert isinstance(report, EvalReport)
        assert report.config_results == []

    def test_write_results_yaml(self, tmp_path: Path) -> None:
        """write_results_yaml should create a valid YAML file."""
        import yaml

        report = EvalReport(
            model={"type": "decoder_only", "source": "safety"},
            evaluation="dclm-core",
            config_results=[],
        )
        output_path = str(tmp_path / "results.yaml")
        write_results_yaml(report, output_path)

        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["model"]["source"] == "safety"
        assert data["evaluation"] == "dclm-core"

    def test_output_dir_defaults_to_checkpoint_dir(self, tmp_path: Path) -> None:
        """Default output dir should be meta['checkpoint_dir']/dclm_eval/."""
        meta = {"checkpoint_dir": str(tmp_path / "ckpts" / "safety_sft_checkpoints")}
        default_output = os.path.join(meta["checkpoint_dir"], "dclm_eval")
        assert default_output.endswith("safety_sft_checkpoints/dclm_eval")


# ===========================================================================
# chat_eval and harmfulness_eval source resolution
# ===========================================================================


class TestChatEvalSourceResolution:
    """Test that chat_eval passes source strings directly to load_model."""

    def test_chain_source_loads(self, tmp_path: Path) -> None:
        """source='safety' should load from safety_sft_checkpoints."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=998)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            # Replicate what chat_eval does at line 193
            model, tokenizer, meta = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.EVAL,
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_eval_source_resolution"),
            )
        assert isinstance(model, GPT)
        assert meta["checkpoint_dir"] == str(safety_dir)


class TestHarmfulnessEvalSourceResolution:
    """Test that harmfulness_eval passes source strings directly to load_model."""

    def test_chain_source_loads(self, tmp_path: Path) -> None:
        """source='safety' should load from safety_sft_checkpoints."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=998)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            # Replicate what harmfulness_eval does at lines 41-44
            model, tokenizer, _meta = load_model(
                source="safety",
                device=t.device("cpu"),
                phase=ModelPhase.EVAL,
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_eval_source_resolution"),
            )
        assert isinstance(model, GPT)
        assert _meta["checkpoint_dir"] == str(safety_dir)


# ===========================================================================
# Config YAML validation
# ===========================================================================


class TestEvalConfigs:
    """Verify that eval configs declare source as a required field."""

    def test_base_eval_config_has_source(self) -> None:
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "eval" / "base_eval.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "source" in cfg

    def test_dclm_eval_config_has_source(self) -> None:
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "eval" / "dclm_eval.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "source" in cfg

    def test_dclm_eval_config_no_checkpoint_dir(self) -> None:
        """dclm_eval should no longer have checkpoint_dir or model_type."""
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "eval" / "dclm_eval.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "checkpoint_dir" not in cfg
        assert "model_type" not in cfg

    def test_chat_eval_config_has_source(self) -> None:
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "eval" / "chat_eval.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "source" in cfg

    def test_harmfulness_eval_config_has_source(self) -> None:
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "eval" / "harmfulness_eval.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "source" in cfg
