"""Tests for configurable model source in chat_sft.

Verifies that chat_sft.py loads models via cfg.source (not hardcoded
ModelSource.BASE), that the training config declares the field, and that
downstream save paths nest correctly for chained fine-tuning pipelines
like base -> safety -> capabilities.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nanochat.checkpoint_manager import (
    get_checkpoint_dir,
    load_model,
    save_checkpoint,
    ModelPhase,
)
from nanochat.gpt import GPT
from steering.conditional.eval_conditioning import EvalConditioning
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
# Config YAML validation
# ===========================================================================


class TestSftConfigHasSource:
    """Verify that the SFT training config declares a source field."""

    _config_path = (
        Path(__file__).parent.parent.parent / "configs" / "training" / "chat_sft.yaml"
    )

    def test_config_has_source_field(self) -> None:
        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)
        assert "source" in cfg, "chat_sft.yaml must declare a 'source' field"

    def test_default_source_is_base(self) -> None:
        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["source"] == "base", (
            "Default source should be 'base' for backward compatibility"
        )

    def test_config_has_model_tag(self) -> None:
        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)
        assert "model_tag" in cfg

    def test_config_has_mixture(self) -> None:
        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)
        assert "mixture" in cfg


# ===========================================================================
# Source-driven model loading (replicates what chat_sft.py line 58 does)
# ===========================================================================


class TestSftSourceLoading:
    """Test that load_model works with various source strings in TRAIN phase,
    matching how chat_sft.py uses cfg.source."""

    def test_base_source_loads_from_base_checkpoints(self, tmp_path: Path) -> None:
        """Default source='base' loads from base_checkpoints/."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=100)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            model, tokenizer, meta = load_model(
                "base",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        assert isinstance(model, GPT)
        assert meta["checkpoint_dir"] == str(ckpt_dir)
        assert meta["step"] == 100

    def test_safety_source_loads_from_safety_sft(self, tmp_path: Path) -> None:
        """source='safety' loads from safety_sft_checkpoints/."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=200)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            model, tokenizer, meta = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        assert isinstance(model, GPT)
        assert meta["checkpoint_dir"] == str(safety_dir)
        assert meta["step"] == 200

    def test_sft_source_loads_from_standard_sft(self, tmp_path: Path) -> None:
        """source='sft' loads from standard_sft_checkpoints/."""
        base_dir = str(tmp_path)
        sft_dir = tmp_path / "base_checkpoints" / "d2" / "standard_sft_checkpoints"
        _create_fake_checkpoint(str(sft_dir), step=300)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            model, tokenizer, meta = load_model(
                "sft",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        assert isinstance(model, GPT)
        assert meta["checkpoint_dir"] == str(sft_dir)

    @pytest.mark.parametrize(
        "source",
        [
            "base",
            "safety",
            "sft",
            "capabilities",
        ],
    )
    def test_train_phase_returns_trainable_model(
        self, tmp_path: Path, source: str
    ) -> None:
        """Models loaded with TRAIN phase should be in training mode."""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            ckpt_dir = get_checkpoint_dir(source, model_tag="d2")
        _create_fake_checkpoint(ckpt_dir, step=0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            model, _, _ = load_model(
                source,
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        assert model.training, (
            f"Model from source={source!r} should be in training mode"
        )


# ===========================================================================
# SFT checkpoint save path nesting
# ===========================================================================


class TestSftSavePathNesting:
    """Test that SFT checkpoint save paths nest correctly under the loaded
    model's checkpoint_dir, matching chat_sft.py line 274:

        checkpoint_dir = os.path.join(meta["checkpoint_dir"],
                                      f"{cfg.mixture}_sft_checkpoints")
    """

    def test_standard_sft_on_base(self, tmp_path: Path) -> None:
        """base -> standard SFT saves under base_checkpoints/d2/standard_sft_checkpoints/."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "base",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        save_dir = os.path.join(meta["checkpoint_dir"], "standard_sft_checkpoints")
        expected = str(
            tmp_path / "base_checkpoints" / "d2" / "standard_sft_checkpoints"
        )
        assert save_dir == expected

    def test_safety_sft_on_base(self, tmp_path: Path) -> None:
        """base -> safety SFT saves under base_checkpoints/d2/safety_sft_checkpoints/."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "base",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        save_dir = os.path.join(meta["checkpoint_dir"], "safety_sft_checkpoints")
        expected = str(tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints")
        assert save_dir == expected

    def test_capabilities_sft_on_safety(self, tmp_path: Path) -> None:
        """safety -> capabilities SFT saves under
        base_checkpoints/d2/safety_sft_checkpoints/capabilities_sft_checkpoints/.

        This is the key use case that motivated making source configurable.
        """
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=50)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        save_dir = os.path.join(meta["checkpoint_dir"], "capabilities_sft_checkpoints")
        expected = str(
            tmp_path
            / "base_checkpoints"
            / "d2"
            / "safety_sft_checkpoints"
            / "capabilities_sft_checkpoints"
        )
        assert save_dir == expected

    def test_capabilities_sft_on_standard(self, tmp_path: Path) -> None:
        """sft -> capabilities SFT saves under
        base_checkpoints/d2/standard_sft_checkpoints/capabilities_sft_checkpoints/."""
        base_dir = str(tmp_path)
        sft_dir = tmp_path / "base_checkpoints" / "d2" / "standard_sft_checkpoints"
        _create_fake_checkpoint(str(sft_dir), step=75)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "sft",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        save_dir = os.path.join(meta["checkpoint_dir"], "capabilities_sft_checkpoints")
        expected = str(
            tmp_path
            / "base_checkpoints"
            / "d2"
            / "standard_sft_checkpoints"
            / "capabilities_sft_checkpoints"
        )
        assert save_dir == expected


# ===========================================================================
# chat_sft.py does not hardcode ModelSource.BASE
# ===========================================================================


class TestSftScriptUsesConfigSource:
    """Verify that scripts/chat_sft.py passes cfg.source to load_model,
    not a hardcoded ModelSource.BASE."""

    def test_load_model_call_uses_cfg_source(self) -> None:
        """Scan chat_sft.py for the load_model call and assert it uses cfg.source."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train" / "chat_sft.py"
        )
        source = script_path.read_text()

        # The old code had: load_model(ModelSource.BASE, ...)
        assert "load_model(ModelSource.BASE" not in source, (
            "chat_sft.py should NOT hardcode ModelSource.BASE in load_model call"
        )

        # The new code should have load_model called with cfg.source
        # (may be on the same line or the next line after the opening paren)
        assert "cfg.source" in source and "load_model" in source, (
            "chat_sft.py should use cfg.source in load_model call"
        )
        # Verify cfg.source appears near load_model (within a few lines)
        lines = source.splitlines()
        load_model_lines = [
            i for i, line in enumerate(lines) if "load_model" in line
        ]
        cfg_source_lines = [
            i for i, line in enumerate(lines) if "cfg.source" in line
        ]
        assert any(
            abs(lm - cs) <= 2
            for lm in load_model_lines
            for cs in cfg_source_lines
        ), (
            "cfg.source should appear within 2 lines of a load_model call"
        )

    def test_script_does_not_ignore_source_config(self) -> None:
        """Ensure 'source' appears in both the config yaml and the script."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train" / "chat_sft.py"
        )
        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "training" / "chat_sft.yaml"
        )

        script_text = script_path.read_text()
        config_text = config_path.read_text()

        assert "cfg.source" in script_text, "Script must reference cfg.source"
        assert "source:" in config_text, "Config must declare source field"


# ===========================================================================
# Model step discovery with non-base sources
# ===========================================================================


class TestSftStepDiscovery:
    """Test that load_model correctly discovers the latest step when loading
    from non-base sources (model_step=None)."""

    def test_finds_latest_step_from_safety(self, tmp_path: Path) -> None:
        """When model_step is None, load_model should find the latest step."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        # Create two checkpoints at different steps
        _create_fake_checkpoint(str(safety_dir), step=50)
        _create_fake_checkpoint(str(safety_dir), step=100)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                step=None,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        assert meta["step"] == 100, "Should load latest step when step=None"

    def test_explicit_step_from_safety(self, tmp_path: Path) -> None:
        """When model_step is specified, load that exact step."""
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=50)
        _create_fake_checkpoint(str(safety_dir), step=100)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                step=50,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
        assert meta["step"] == 50, "Should load the requested step"


# ===========================================================================
# End-to-end chained SFT simulation
# ===========================================================================


class TestChainedSftPipeline:
    """Simulate the full chained SFT pipeline: base -> safety -> capabilities.

    This replicates the exact logic from chat_sft.py:
    1. Load model from source (cfg.source)
    2. Train (not tested here)
    3. Save to os.path.join(meta["checkpoint_dir"], f"{mixture}_sft_checkpoints")
    """

    def test_full_chain_base_safety_capabilities(self, tmp_path: Path) -> None:
        """Simulate base -> safety -> capabilities checkpoint chain."""
        base_dir = str(tmp_path)
        base_ckpt = tmp_path / "base_checkpoints" / "d2"

        # Step 1: Create base checkpoint
        _create_fake_checkpoint(str(base_ckpt), step=1000)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            # Step 2: Load base, compute safety save path
            _, _, meta_base = load_model(
                "base",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
            safety_save_dir = os.path.join(
                meta_base["checkpoint_dir"],
                "safety_sft_checkpoints",
            )

            # Step 3: Create safety checkpoint (simulates training completion)
            _create_fake_checkpoint(safety_save_dir, step=500)

            # Step 4: Load safety, compute capabilities save path
            _, _, meta_safety = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag="d2",
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
            capabilities_save_dir = os.path.join(
                meta_safety["checkpoint_dir"],
                "capabilities_sft_checkpoints",
            )

        # Verify the full chain nests correctly
        assert meta_base["checkpoint_dir"] == str(base_ckpt)
        assert meta_safety["checkpoint_dir"] == safety_save_dir
        expected_capabilities = str(
            base_ckpt / "safety_sft_checkpoints" / "capabilities_sft_checkpoints"
        )
        assert capabilities_save_dir == expected_capabilities

    def test_full_chain_preserves_model_tag(self, tmp_path: Path) -> None:
        """Model tag should be consistent across the chain."""
        base_dir = str(tmp_path)
        tag = "d26"

        # Create base and safety checkpoints
        base_ckpt = tmp_path / "base_checkpoints" / tag
        safety_ckpt = base_ckpt / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(base_ckpt), step=13232)
        _create_fake_checkpoint(str(safety_ckpt), step=800)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta_base = load_model(
                "base",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag=tag,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
            _, _, meta_safety = load_model(
                "safety",
                t.device("cpu"),
                phase=ModelPhase.TRAIN,
                model_tag=tag,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )

        # Both should resolve under the same model tag directory
        assert tag in meta_base["checkpoint_dir"]
        assert tag in meta_safety["checkpoint_dir"]
        assert meta_safety["checkpoint_dir"].startswith(meta_base["checkpoint_dir"])
