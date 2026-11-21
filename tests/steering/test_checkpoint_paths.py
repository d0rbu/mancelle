"""Unit tests for nested checkpoint directory path construction."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from errors import DataLoadError
from nanochat.checkpoint_manager import (
    _find_checkpoint_leaf,
    get_checkpoint_dir,
    get_report_dir,
    load_model,
    load_model_from_dir,
    save_checkpoint,
)
from steering.conditional.eval_conditioning import EvalConditioning
from tests.conftest import MockTokenizer, _make_gpt, _make_gpt_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_checkpoint(checkpoint_dir: str, step: int = 0) -> None:
    """Create minimal checkpoint files (model, meta) at the given dir."""
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
# get_checkpoint_dir tests
# ===========================================================================


class TestGetCheckpointDir:
    """Test that get_checkpoint_dir resolves source strings to correct paths."""

    def test_base_with_tag(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("base", model_tag="d26")
        assert result == os.path.join(base_dir, "base_checkpoints", "d26")

    def test_sft_nested_under_base(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("sft", model_tag="d26")
        expected = os.path.join(
            base_dir, "base_checkpoints", "d26", "standard_sft_checkpoints"
        )
        assert result == expected

    def test_rl_nested_under_sft(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("rl", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "standard_sft_checkpoints",
            "standard_rl_checkpoints",
        )
        assert result == expected

    def test_encdec_with_tag(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "conditional_encdec_checkpoints" / "my_model")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("encdec", model_tag="my_model")
        expected = os.path.join(base_dir, "conditional_encdec_checkpoints", "my_model")
        assert result == expected

    def test_encdec_with_segment_and_threshold(self, tmp_path: Path) -> None:
        from steering.conditional.sampling import ThresholdSelection
        from steering.conditional.thresholding import ThresholdType

        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "conditional_encdec_checkpoints" / "d26")
        fs = ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2048.9)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir(
                "encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=fs,
            )
        expected = os.path.join(
            base_dir,
            "conditional_encdec_checkpoints",
            "d26",
            "s128",
            "t2048.9",
        )
        assert result == expected

    def test_standard_alias_for_sft(self, tmp_path: Path) -> None:
        """'standard' is an alias for 'sft'."""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result_sft = get_checkpoint_dir("sft", model_tag="d26")
            result_std = get_checkpoint_dir("standard", model_tag="d26")
        assert result_sft == result_std
        assert result_sft.endswith("standard_sft_checkpoints")

    def test_safety_sft(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("safety", model_tag="d26")
        expected = os.path.join(
            base_dir, "base_checkpoints", "d26", "safety_sft_checkpoints"
        )
        assert result == expected

    def test_capabilities_sft(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("capabilities", model_tag="d26")
        expected = os.path.join(
            base_dir, "base_checkpoints", "d26", "capabilities_sft_checkpoints"
        )
        assert result == expected

    def test_safety_rl_chain(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("safety/rl", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "safety_sft_checkpoints",
            "standard_rl_checkpoints",
        )
        assert result == expected

    def test_sft_rl_chain(self, tmp_path: Path) -> None:
        """'sft/rl' should resolve same as bare 'rl'."""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result_chain = get_checkpoint_dir("sft/rl", model_tag="d26")
            result_bare = get_checkpoint_dir("rl", model_tag="d26")
        assert result_chain == result_bare

    def test_base_safety_chain(self, tmp_path: Path) -> None:
        """'base/safety' should resolve same as bare 'safety'."""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result_chain = get_checkpoint_dir("base/safety", model_tag="d26")
            result_bare = get_checkpoint_dir("safety", model_tag="d26")
        assert result_chain == result_bare

    def test_encdec_safety_chain(self, tmp_path: Path) -> None:
        from steering.conditional.sampling import ThresholdSelection
        from steering.conditional.thresholding import ThresholdType

        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "conditional_encdec_checkpoints" / "d26")
        fs = ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2048.9)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir(
                "encdec/safety",
                model_tag="d26",
                segment_size=128,
                feature_selection=fs,
            )
        expected = os.path.join(
            base_dir,
            "conditional_encdec_checkpoints",
            "d26",
            "s128",
            "t2048.9",
            "safety_sft_checkpoints",
        )
        assert result == expected

    def test_unknown_source_raises(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            with pytest.raises(ValueError, match="Unknown source"):
                get_checkpoint_dir("invalid")

    def test_invalid_chain_segment_raises(self, tmp_path: Path) -> None:
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            with pytest.raises(ValueError, match="Invalid segment"):
                get_checkpoint_dir("base/invalid", model_tag="d26")

    def test_base_auto_tag(self, tmp_path: Path) -> None:
        """When model_tag is None, find_largest_model picks the biggest d* dir."""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d12")
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("base")
        assert result.endswith("d26")


# ===========================================================================
# load_model checkpoint_dir in meta tests
# ===========================================================================


class TestLoadModelMeta:
    """Test that load_model includes checkpoint_dir in the returned meta."""

    def test_base_meta_has_checkpoint_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "base",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        assert "checkpoint_dir" in meta
        assert meta["checkpoint_dir"] == str(ckpt_dir)

    def test_sft_meta_has_checkpoint_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        sft_dir = tmp_path / "base_checkpoints" / "d2" / "standard_sft_checkpoints"
        _create_fake_checkpoint(str(sft_dir), step=0)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "sft",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        assert "checkpoint_dir" in meta
        assert meta["checkpoint_dir"] == str(sft_dir)

    def test_safety_meta_has_checkpoint_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        safety_dir = tmp_path / "base_checkpoints" / "d2" / "safety_sft_checkpoints"
        _create_fake_checkpoint(str(safety_dir), step=0)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "safety",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        assert "checkpoint_dir" in meta
        assert meta["checkpoint_dir"] == str(safety_dir)

    def test_safety_rl_meta_has_checkpoint_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        rl_dir = (
            tmp_path
            / "base_checkpoints"
            / "d2"
            / "safety_sft_checkpoints"
            / "standard_rl_checkpoints"
        )
        _create_fake_checkpoint(str(rl_dir), step=0)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "safety/rl",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        assert "checkpoint_dir" in meta
        assert meta["checkpoint_dir"] == str(rl_dir)

    def test_load_model_from_dir_includes_checkpoint_dir(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "my_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        _, _, meta = load_model_from_dir(
            str(tmp_path / "my_checkpoints"),
            t.device("cpu"),
            "eval",
            model_tag="d2",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
        )
        assert "checkpoint_dir" in meta
        assert meta["checkpoint_dir"] == str(ckpt_dir)


# ===========================================================================
# Nested save path tests
# ===========================================================================


class TestNestedSavePaths:
    """Test that child checkpoints nest under their parent's checkpoint_dir."""

    def test_sft_saves_under_base(self, tmp_path: Path) -> None:
        """SFT checkpoint_dir should nest under base."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "base",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        # Simulate what chat_sft.py does with meta["checkpoint_dir"]
        sft_save_dir = os.path.join(meta["checkpoint_dir"], "standard_sft_checkpoints")
        expected = os.path.join(
            base_dir, "base_checkpoints", "d2", "standard_sft_checkpoints"
        )
        assert sft_save_dir == expected

    def test_safety_sft_saves_under_base(self, tmp_path: Path) -> None:
        """Safety SFT uses mixture name 'safety'."""
        base_dir = str(tmp_path)
        ckpt_dir = tmp_path / "base_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "base",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        sft_save_dir = os.path.join(meta["checkpoint_dir"], "safety_sft_checkpoints")
        expected = os.path.join(
            base_dir, "base_checkpoints", "d2", "safety_sft_checkpoints"
        )
        assert sft_save_dir == expected

    def test_rl_saves_under_sft(self, tmp_path: Path) -> None:
        """RL checkpoint_dir should nest under SFT."""
        base_dir = str(tmp_path)
        sft_dir = tmp_path / "base_checkpoints" / "d2" / "standard_sft_checkpoints"
        _create_fake_checkpoint(str(sft_dir), step=0)
        os.makedirs(tmp_path / "base_checkpoints" / "d2", exist_ok=True)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, _, meta = load_model(
                "sft",
                t.device("cpu"),
                "eval",
                model_tag="d2",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
            )
        rl_save_dir = os.path.join(meta["checkpoint_dir"], "standard_rl_checkpoints")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d2",
            "standard_sft_checkpoints",
            "standard_rl_checkpoints",
        )
        assert rl_save_dir == expected

    def test_checkpoints_dir_kwarg(self, tmp_path: Path) -> None:
        """load_model with explicit checkpoints_dir should use that path."""
        ckpt_dir = tmp_path / "custom_checkpoints" / "d2"
        _create_fake_checkpoint(str(ckpt_dir), step=0)
        _, _, meta = load_model(
            "base",
            t.device("cpu"),
            "eval",
            checkpoints_dir=str(tmp_path / "custom_checkpoints"),
            model_tag="d2",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_paths"),
        )
        assert meta["checkpoint_dir"] == str(ckpt_dir)


# ===========================================================================
# _find_checkpoint_leaf tests
# ===========================================================================


class TestFindCheckpointLeaf:
    """Test that _find_checkpoint_leaf traverses nested dirs to find model_*.pt."""

    def test_flat_dir(self, tmp_path: Path) -> None:
        """If model_*.pt is directly in root, return root."""
        _create_fake_checkpoint(str(tmp_path), step=0)
        assert _find_checkpoint_leaf(str(tmp_path)) == str(tmp_path)

    def test_nested_encdec_structure(self, tmp_path: Path) -> None:
        """Traverse s128/t2048.9/ to find checkpoint."""
        leaf = tmp_path / "s128" / "t2048.9"
        _create_fake_checkpoint(str(leaf), step=0)
        assert _find_checkpoint_leaf(str(tmp_path)) == str(leaf)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        """Empty directory raises DataLoadError."""
        with pytest.raises(DataLoadError):
            _find_checkpoint_leaf(str(tmp_path))

    def test_nested_with_sibling_dirs(self, tmp_path: Path) -> None:
        """Only returns dir with actual checkpoint files, not empty siblings."""
        (tmp_path / "s64").mkdir()
        leaf = tmp_path / "s128" / "t2048.9"
        _create_fake_checkpoint(str(leaf), step=0)
        assert _find_checkpoint_leaf(str(tmp_path)) == str(leaf)


# ===========================================================================
# get_report_dir tests
# ===========================================================================


class TestGetReportDir:
    """Test that get_report_dir returns {checkpoint_dir}/report/ for any source."""

    def test_base_report_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_report_dir("base", model_tag="d26")
        expected = os.path.join(base_dir, "base_checkpoints", "d26", "report")
        assert result == expected

    def test_safety_report_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_report_dir("safety", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "safety_sft_checkpoints",
            "report",
        )
        assert result == expected

    def test_safety_rl_report_dir(self, tmp_path: Path) -> None:
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_report_dir("safety/rl", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "safety_sft_checkpoints",
            "standard_rl_checkpoints",
            "report",
        )
        assert result == expected

    def test_encdec_report_dir(self, tmp_path: Path) -> None:
        from steering.conditional.sampling import ThresholdSelection
        from steering.conditional.thresholding import ThresholdType

        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "conditional_encdec_checkpoints" / "d26")
        fs = ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2048.9)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_report_dir(
                "encdec", model_tag="d26", segment_size=128, feature_selection=fs
            )
        expected = os.path.join(
            base_dir,
            "conditional_encdec_checkpoints",
            "d26",
            "s128",
            "t2048.9",
            "report",
        )
        assert result == expected

    def test_auto_tag_report_dir(self, tmp_path: Path) -> None:
        """When model_tag is None, auto-detects largest d* model."""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d12")
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_report_dir("base")
        assert result.endswith(os.path.join("d26", "report"))

    def test_passes_kwargs_through(self, tmp_path: Path) -> None:
        """Kwargs like segment_size and feature_selection are forwarded to get_checkpoint_dir."""
        from steering.conditional.sampling import ThresholdSelection
        from steering.conditional.thresholding import ThresholdType

        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "conditional_encdec_checkpoints" / "d26")
        fs = ThresholdSelection(ThresholdType.segment_topk, 100.0)
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_report_dir(
                "encdec", model_tag="d26", segment_size=64, feature_selection=fs
            )
        assert "s64" in result
        assert "t100.0" in result
        assert result.endswith("report")
