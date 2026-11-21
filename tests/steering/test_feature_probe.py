# tests/test_feature_probe.py
"""Tests for the feature probe eval stage."""

from __future__ import annotations

import math

import pytest
import torch as t
from transformers import AutoTokenizer

from steering.eval.completion_steering_eval import EvalStage
from steering.eval.config_validation import validate_evaluator_flags
from steering.eval.feature_probe import (
    build_completion_mask,
    compute_raw_probe,
    compute_sae_probe,
    group_probe_results,
    max_pool_masked,
    plot_probe_ablation,
    plot_probe_dose_response,
)
from steering.segmentation import token_to_char_offsets


class TestConfigValidation:
    """Verify that stage + run_judge + run_feature_analysis flags are consistent."""

    def test_scan_stage_rejects_run_judge(self):
        with pytest.raises(ValueError, match="run_judge"):
            validate_evaluator_flags(
                EvalStage.SCAN, run_judge=True, run_feature_analysis=False
            )

    def test_scan_stage_rejects_run_feature_analysis(self):
        with pytest.raises(ValueError, match="run_feature_analysis"):
            validate_evaluator_flags(
                EvalStage.SCAN, run_judge=False, run_feature_analysis=True
            )

    def test_threshold_stage_rejects_both_true(self):
        with pytest.raises(ValueError):
            validate_evaluator_flags(
                EvalStage.THRESHOLD, run_judge=True, run_feature_analysis=True
            )

    def test_generate_stage_rejects_run_judge(self):
        with pytest.raises(ValueError, match="run_judge"):
            validate_evaluator_flags(
                EvalStage.GENERATE, run_judge=True, run_feature_analysis=False
            )

    def test_select_stage_rejects_run_feature_analysis(self):
        with pytest.raises(ValueError):
            validate_evaluator_flags(
                EvalStage.SELECT, run_judge=False, run_feature_analysis=True
            )

    def test_judge_stage_requires_run_judge_true(self):
        with pytest.raises(ValueError, match="run_judge.*true"):
            validate_evaluator_flags(
                EvalStage.JUDGE, run_judge=False, run_feature_analysis=False
            )

    def test_judge_stage_rejects_run_feature_analysis(self):
        with pytest.raises(ValueError, match="run_feature_analysis.*false"):
            validate_evaluator_flags(
                EvalStage.JUDGE, run_judge=True, run_feature_analysis=True
            )

    def test_judge_stage_accepts_correct_flags(self):
        validate_evaluator_flags(
            EvalStage.JUDGE, run_judge=True, run_feature_analysis=False
        )

    def test_feature_analysis_stage_requires_run_feature_analysis_true(self):
        with pytest.raises(ValueError, match="run_feature_analysis.*true"):
            validate_evaluator_flags(
                EvalStage.FEATURE_ANALYSIS, run_judge=False, run_feature_analysis=False
            )

    def test_feature_analysis_stage_rejects_run_judge(self):
        with pytest.raises(ValueError, match="run_judge.*false"):
            validate_evaluator_flags(
                EvalStage.FEATURE_ANALYSIS, run_judge=True, run_feature_analysis=True
            )

    def test_feature_analysis_stage_accepts_correct_flags(self):
        validate_evaluator_flags(
            EvalStage.FEATURE_ANALYSIS, run_judge=False, run_feature_analysis=True
        )

    def test_all_stage_accepts_both_true(self):
        validate_evaluator_flags(
            EvalStage.ALL, run_judge=True, run_feature_analysis=True
        )

    def test_all_stage_accepts_both_false(self):
        validate_evaluator_flags(
            EvalStage.ALL, run_judge=False, run_feature_analysis=False
        )

    def test_all_stage_accepts_mixed_flags(self):
        validate_evaluator_flags(
            EvalStage.ALL, run_judge=True, run_feature_analysis=False
        )
        validate_evaluator_flags(
            EvalStage.ALL, run_judge=False, run_feature_analysis=True
        )


class _FakeSAEConfig:
    """Minimal stand-in for sae_lens SAE config."""

    def __init__(self, apply_b_dec_to_input: bool = True):
        self.apply_b_dec_to_input = apply_b_dec_to_input


class _FakeSAE:
    """Minimal stand-in for sae_lens SAE with JumpReLU behavior."""

    def __init__(
        self,
        W_enc: t.Tensor,
        b_enc: t.Tensor,
        b_dec: t.Tensor,
        threshold: t.Tensor,
        apply_b_dec_to_input: bool = True,
    ):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.threshold = threshold
        self.cfg = _FakeSAEConfig(apply_b_dec_to_input)

    def encode(self, x: t.Tensor) -> t.Tensor:
        sae_in = x - self.b_dec if self.cfg.apply_b_dec_to_input else x
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        return hidden_pre * (hidden_pre > self.threshold).float()


class TestRawProbeComputation:
    """Verify pre-activation linear probe math with known weights."""

    def test_basic_raw_probe(self):
        _d_model, _n_features = 4, 3
        feature_id = 1

        W_enc = t.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            dtype=t.float32,
        )
        b_enc = t.tensor([0.01, 0.02, 0.03], dtype=t.float32)
        b_dec = t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32)
        act = t.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=t.float32)

        result = compute_raw_probe(
            act, W_enc, b_enc, b_dec, feature_id, apply_b_dec=True
        )

        # sae_in = [[0.9,1.8,2.7,3.6],[0.4,1.3,2.2,3.1]]
        # W_enc[:, 1] = [0.2, 0.5, 0.8, 1.1]
        # token 0: 0.9*0.2+1.8*0.5+2.7*0.8+3.6*1.1+0.02 = 7.22
        # token 1: 0.4*0.2+1.3*0.5+2.2*0.8+3.1*1.1+0.02 = 5.92
        expected = t.tensor([7.22, 5.92], dtype=t.float32)
        assert t.allclose(result, expected, atol=1e-4)
        assert result.shape == (2,)

    def test_raw_probe_no_b_dec_subtraction(self):
        W_enc = t.tensor([[1.0], [0.0]], dtype=t.float32)
        b_enc = t.tensor([0.0], dtype=t.float32)
        b_dec = t.tensor([100.0, 100.0], dtype=t.float32)
        act = t.tensor([[1.0, 2.0]], dtype=t.float32)

        result = compute_raw_probe(
            act, W_enc, b_enc, b_dec, feature_id=0, apply_b_dec=False
        )
        assert t.allclose(result, t.tensor([1.0]))

        result_with = compute_raw_probe(
            act, W_enc, b_enc, b_dec, feature_id=0, apply_b_dec=True
        )
        assert t.allclose(result_with, t.tensor([-99.0]))

    def test_raw_probe_can_be_negative(self):
        W_enc = t.tensor([[1.0]], dtype=t.float32)
        b_enc = t.tensor([-10.0], dtype=t.float32)
        b_dec = t.tensor([0.0], dtype=t.float32)
        act = t.tensor([[1.0]], dtype=t.float32)

        result = compute_raw_probe(
            act, W_enc, b_enc, b_dec, feature_id=0, apply_b_dec=True
        )
        assert result.item() == pytest.approx(-9.0)
        assert result.item() < 0


class TestSAEProbeComputation:
    """Verify SAE encode path respects JumpReLU thresholding."""

    def test_sae_encode_zeros_below_threshold(self):
        W_enc = t.eye(2, dtype=t.float32)
        b_enc = t.zeros(2)
        b_dec = t.zeros(2)
        threshold = t.tensor([0.5, 0.5])
        sae = _FakeSAE(W_enc, b_enc, b_dec, threshold)

        act = t.tensor([[0.3, 0.8]], dtype=t.float32)
        result = compute_sae_probe(act, sae, feature_id=0)
        assert result.item() == 0.0, "0.3 < threshold 0.5, should be zero"

        result1 = compute_sae_probe(act, sae, feature_id=1)
        assert result1.item() == pytest.approx(0.8)

    def test_sae_encode_always_non_negative(self):
        W_enc = t.tensor([[1.0]], dtype=t.float32)
        b_enc = t.tensor([-10.0])
        b_dec = t.zeros(1)
        threshold = t.tensor([0.0])
        sae = _FakeSAE(W_enc, b_enc, b_dec, threshold)

        act = t.tensor([[1.0]], dtype=t.float32)
        result = compute_sae_probe(act, sae, feature_id=0)
        assert result.item() == 0.0
        assert result.item() >= 0

    def test_above_threshold_passes_through(self):
        W_enc = t.eye(1, dtype=t.float32)
        b_enc = t.zeros(1)
        b_dec = t.zeros(1)
        threshold = t.tensor([0.5])
        sae = _FakeSAE(W_enc, b_enc, b_dec, threshold)

        act = t.tensor([[3.0]], dtype=t.float32)
        result = compute_sae_probe(act, sae, feature_id=0)
        assert result.item() == pytest.approx(3.0)


class TestRawVsSAEConsistency:
    """Raw probe and SAE encode must agree when above threshold."""

    def test_both_paths_agree_above_threshold(self):
        d_model, n_features = 4, 2
        W_enc = t.randn(d_model, n_features)
        b_enc = t.zeros(n_features)
        b_dec = t.randn(d_model)
        threshold = t.tensor([-100.0, -100.0])  # very low → everything passes
        sae = _FakeSAE(W_enc, b_enc, b_dec, threshold, apply_b_dec_to_input=True)

        act = t.randn(3, d_model)
        for fid in range(n_features):
            raw = compute_raw_probe(act, W_enc, b_enc, b_dec, fid, apply_b_dec=True)
            sae_val = compute_sae_probe(act, sae, fid)
            assert t.allclose(raw, sae_val, atol=1e-5), (
                f"Feature {fid}: raw={raw} vs sae={sae_val}"
            )


class TestCompletionTokenIdentification:
    """Verify character-offset-based completion mask construction."""

    def test_basic_mask(self):
        # pre_segment = "Hello " (6 chars). Tokens starting at/after 6 are completion.
        char_offsets = [0, 3, 6, 9, 11]  # 4 tokens
        mask = build_completion_mask(char_offsets, completion_char_start=6)
        assert mask.tolist() == [False, False, True, True]

    def test_single_completion_token(self):
        char_offsets = [0, 5, 10, 15]
        mask = build_completion_mask(char_offsets, completion_char_start=10)
        assert mask.tolist() == [False, False, True]

    def test_all_completion(self):
        char_offsets = [0, 3, 6, 9]
        mask = build_completion_mask(char_offsets, completion_char_start=0)
        assert mask.tolist() == [True, True, True]

    def test_empty_mask_when_no_completion_tokens(self):
        char_offsets = [0, 3, 6, 9]
        mask = build_completion_mask(char_offsets, completion_char_start=100)
        assert mask.tolist() == [False, False, False]

    def test_unicode_boundary(self):
        """Unicode multi-byte chars: character offsets differ from byte offsets."""
        # Simulate: "Caf" (3 chars) | "é " (2 chars) | "ok" (2 chars)
        char_offsets = [0, 3, 5, 7]
        mask = build_completion_mask(char_offsets, completion_char_start=5)
        assert mask.tolist() == [False, False, True]  # only "ok" is completion

    def test_with_token_to_char_offsets(self):
        """Smoke test using the real token_to_char_offsets function."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pre = "Hello world "
        completion = "this is a test"
        full = pre + completion
        tokens = tokenizer.encode(full)  # ty: ignore[unresolved-attribute]
        offsets = token_to_char_offsets(tokenizer, tokens)

        mask = build_completion_mask(offsets, completion_char_start=len(pre))
        # At least some tokens should be True (the completion part)
        assert mask.any(), "Expected at least one completion token"
        # Tokens before the pre boundary should be False
        for i in range(mask.shape[0]):
            if offsets[i] < len(pre):
                assert not mask[i], (
                    f"Token {i} (offset {offsets[i]}) should not be in completion"
                )


class TestProbeResultsGrouping:
    """Verify grouping of probe values by (feature_id, condition)."""

    def test_groups_by_feature_and_condition(self):
        raw_scores: list[tuple[int, str, float]] = [
            (10, "real", 1.0),
            (10, "real", 1.5),
            (10, "ablated", 0.3),
            (20, "real", 2.0),
            (20, "shuffled", 0.8),
        ]
        results = group_probe_results(raw_scores)

        assert 10 in results
        assert 20 in results
        assert results[10].condition_scores["real"] == [1.0, 1.5]
        assert results[10].condition_scores["ablated"] == [0.3]
        assert results[20].condition_scores["real"] == [2.0]
        assert results[20].condition_scores["shuffled"] == [0.8]

    def test_no_cross_contamination(self):
        raw_scores = [(1, "real", 5.0), (2, "real", 9.0)]
        results = group_probe_results(raw_scores)
        assert results[1].condition_scores["real"] == [5.0]
        assert results[2].condition_scores["real"] == [9.0]

    def test_empty_input(self):
        results = group_probe_results([])
        assert results == {}


class TestMaxPooling:
    """Verify max-pool over masked completion tokens."""

    def test_basic_max_pool(self):
        probe = t.tensor([0.1, 0.5, -0.3, 0.8, 0.2])
        mask = t.tensor([False, False, True, True, True])
        result = max_pool_masked(probe, mask)
        assert result == pytest.approx(0.8)

    def test_single_completion_token(self):
        probe = t.tensor([1.0, 2.0, 3.0])
        mask = t.tensor([False, False, True])
        result = max_pool_masked(probe, mask)
        assert result == pytest.approx(3.0)

    def test_all_negative_completion(self):
        probe = t.tensor([-5.0, -1.0, -3.0, -2.0])
        mask = t.tensor([False, True, True, True])
        result = max_pool_masked(probe, mask)
        assert result == pytest.approx(-1.0)

    def test_empty_mask_returns_nan(self):
        probe = t.tensor([1.0, 2.0, 3.0])
        mask = t.tensor([False, False, False])
        result = max_pool_masked(probe, mask)
        assert math.isnan(result)


class TestPlotting:
    """Verify probe plots are created without errors."""

    def test_ablation_plot_creates_file(self, tmp_path):
        plot_probe_ablation(
            real_mean=2.5,
            ablated_mean=0.8,
            feature_id=42,
            feature_description="test feature",
            output_path=tmp_path / "probe_ablation_feature_42.png",
            real_stderr=0.3,
            ablated_stderr=0.2,
        )
        assert (tmp_path / "probe_ablation_feature_42.png").exists()

    def test_ablation_plot_handles_negative_values(self, tmp_path):
        plot_probe_ablation(
            real_mean=-1.0,
            ablated_mean=-3.0,
            feature_id=1,
            feature_description="negative probe",
            output_path=tmp_path / "probe_ablation_feature_1.png",
        )
        assert (tmp_path / "probe_ablation_feature_1.png").exists()

    def test_dose_response_plot_creates_file(self, tmp_path):
        plot_probe_dose_response(
            bucket_means=[0.5, 1.0, 1.5, 2.0],
            bucket_stderrs=[0.1, 0.15, 0.2, 0.1],
            feature_id=99,
            feature_description="dose response feature",
            output_path=tmp_path / "probe_dose_response_feature_99.png",
            real_baseline=1.8,
        )
        assert (tmp_path / "probe_dose_response_feature_99.png").exists()

    def test_dose_response_without_real_baseline(self, tmp_path):
        plot_probe_dose_response(
            bucket_means=[1.0, 2.0],
            bucket_stderrs=[0.1, 0.2],
            feature_id=7,
            feature_description="no real baseline",
            output_path=tmp_path / "probe_dose_response_feature_7.png",
        )
        assert (tmp_path / "probe_dose_response_feature_7.png").exists()

    def test_plot_filenames_have_probe_prefix(self, tmp_path):
        abl_path = tmp_path / "probe_ablation_feature_1.png"
        dose_path = tmp_path / "probe_dose_response_feature_1.png"

        plot_probe_ablation(
            real_mean=1.0,
            ablated_mean=0.5,
            feature_id=1,
            feature_description="x",
            output_path=abl_path,
        )
        plot_probe_dose_response(
            bucket_means=[1.0],
            bucket_stderrs=[0.1],
            feature_id=1,
            feature_description="x",
            output_path=dose_path,
        )

        assert "probe_ablation" in abl_path.name
        assert "probe_dose_response" in dose_path.name
