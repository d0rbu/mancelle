"""Tests for force_recompute pipeline option.

Verifies that _should_skip_stage correctly evaluates the three-way
skip condition (stage=ALL, output exists, not force_recompute), and
that the Hydra config plumbs the flag correctly.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

# Import will fail until we create the helper — that's the point of TDD.
from scripts.eval.completion_steering_eval import _should_skip_stage
from steering.eval.completion_steering_eval import EvalStage

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "completion_steering_eval"


class TestShouldSkipStage:
    """Exhaustive tests for _should_skip_stage helper."""

    def test_skip_when_all_and_exists_and_no_force(self, tmp_path: Path) -> None:
        """Only case that returns True: stage=ALL, file exists, force=False."""
        output = tmp_path / "validation_scan.safetensors"
        output.touch()
        assert _should_skip_stage(EvalStage.ALL, output, force_recompute=False) is True

    def test_no_skip_when_force_true(self, tmp_path: Path) -> None:
        """force_recompute=True overrides the skip even when output exists."""
        output = tmp_path / "validation_scan.safetensors"
        output.touch()
        assert _should_skip_stage(EvalStage.ALL, output, force_recompute=True) is False

    def test_no_skip_when_output_missing(self, tmp_path: Path) -> None:
        """No output file means stage must run regardless of force flag."""
        output = tmp_path / "validation_scan.safetensors"
        assert _should_skip_stage(EvalStage.ALL, output, force_recompute=False) is False

    def test_no_skip_when_output_missing_and_force(self, tmp_path: Path) -> None:
        """Missing output + force=True — still no skip."""
        output = tmp_path / "validation_scan.safetensors"
        assert _should_skip_stage(EvalStage.ALL, output, force_recompute=True) is False

    def test_no_skip_for_single_stage_scan(self, tmp_path: Path) -> None:
        """Single-stage runs never skip, even without force."""
        output = tmp_path / "validation_scan.safetensors"
        output.touch()
        assert (
            _should_skip_stage(EvalStage.SCAN, output, force_recompute=False) is False
        )

    def test_no_skip_for_single_stage_with_force(self, tmp_path: Path) -> None:
        """Single-stage + force=True — still no skip."""
        output = tmp_path / "validation_scan.safetensors"
        output.touch()
        assert _should_skip_stage(EvalStage.SCAN, output, force_recompute=True) is False

    def test_no_skip_for_all_single_stages(self, tmp_path: Path) -> None:
        """Every non-ALL stage returns False even with output present."""
        output = tmp_path / "output.safetensors"
        output.touch()
        for stage in (
            EvalStage.SCAN,
            EvalStage.THRESHOLD,
            EvalStage.SELECT,
            EvalStage.GENERATE,
            EvalStage.JUDGE,
        ):
            assert _should_skip_stage(stage, output, force_recompute=False) is False, (
                f"Expected no skip for single stage {stage.value}"
            )


class TestForceRecomputeConfig:
    """Verify Hydra config plumbing for force_recompute."""

    def test_default_is_false(self) -> None:
        """force_recompute defaults to false in the YAML config."""
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            cfg = compose(config_name=_CONFIG_NAME)
        assert cfg.force_recompute is False

    def test_override_to_true(self) -> None:
        """force_recompute can be overridden to true via Hydra."""
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name=_CONFIG_NAME,
                overrides=["force_recompute=true"],
            )
        assert cfg.force_recompute is True
