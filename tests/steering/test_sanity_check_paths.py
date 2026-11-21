"""Tests for sanity check path resolution."""

from pathlib import Path


class TestSanityCheckPaths:
    """Verify sanity check scripts use get_base_dir() not hardcoded paths."""

    def test_run_all_no_hardcoded_out_dir(self) -> None:
        run_all = Path("steering/eval/sanity_check/run_all.py").read_text()
        assert "out/decoder_training_artifacts" not in run_all

    def test_dclm_no_hardcoded_out_dir(self) -> None:
        content = Path("steering/eval/sanity_check/dclm.py").read_text()
        assert "out/decoder_training_artifacts" not in content

    def test_inference_no_hardcoded_out_dir(self) -> None:
        content = Path("steering/eval/sanity_check/inference.py").read_text()
        assert "out/decoder_training_artifacts" not in content

    def test_steering_sanity_is_decoder_only_control(self) -> None:
        content = Path("steering/eval/sanity_check/steering.py").read_text()
        assert "out/decoder_training_artifacts" not in content
        assert "run_steering_eval" not in content
        assert "ContextConditioningMode" not in content
        assert "skip_conditioning" not in content

    def test_harmfulness_no_hardcoded_out_dir(self) -> None:
        content = Path("steering/eval/sanity_check/harmfulness.py").read_text()
        assert "out/decoder_training_artifacts" not in content
        assert 'model="openai/gpt-5.6-luna"' in content
        assert "score_compliance_all" in content
        assert "compute_compliance_metrics" in content

    def test_dclm_test_no_hardcoded_out_dir(self) -> None:
        content = Path("tests/steering/test_dclm.py").read_text()
        assert "out/decoder_training_artifacts" not in content
        assert "out/encoder_decoder_full_attn_training_artifacts" not in content
