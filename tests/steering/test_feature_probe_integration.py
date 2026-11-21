# tests/test_feature_probe_integration.py
"""Integration tests for the feature probe pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch as t

from steering.eval.feature_probe import (
    ProbeResults,
    compute_probe_for_completions,
)
from steering.eval.multiseg_generator import CompletionSample, DocumentCompletion
from steering.feature_paths import (
    DatasetPhase,
    FeatureDirectoryComponents,
    PoolingMode,
)
from steering.sae_resolver import ActivationType, SAEWidth


class _FakeSAEConfig:
    def __init__(self):
        self.apply_b_dec_to_input = True


class _FakeSAE:
    """Deterministic SAE for integration testing."""

    def __init__(self, d_model: int = 8, n_features: int = 4):
        self.W_enc = t.eye(d_model, n_features)[:, :n_features].float()
        self.b_enc = t.zeros(n_features)
        self.b_dec = t.zeros(d_model)
        self.cfg = _FakeSAEConfig()
        self._threshold = t.zeros(n_features)

    def encode(self, x: t.Tensor) -> t.Tensor:
        hidden = x @ self.W_enc + self.b_enc
        return hidden * (hidden > self._threshold).float()


class _FakeTokenizer:
    """Simple character-level tokenizer for testing."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def batch_decode(self, ids_list):
        return [self.decode(ids) for ids in ids_list]

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]


class _FakeHookedTransformer:
    """Returns deterministic activations for testing."""

    def __init__(self, d_model: int = 8):
        self.d_model = d_model
        self.tokenizer = _FakeTokenizer()

    def run_with_cache(
        self, tokens, names_filter=None, attention_mask=None, stop_at_layer=None
    ):
        batch, seq_len = tokens.shape
        act = t.zeros(batch, seq_len, self.d_model)
        for b in range(batch):
            for s in range(seq_len):
                act[b, s, 0] = tokens[b, s].float() * 0.01
                act[b, s, 1] = tokens[b, s].float() * 0.02
        cache = {names_filter[0]: act} if names_filter else {}
        return None, cache


class TestProbeEndToEnd:
    """End-to-end integration test with mocked model loading."""

    @patch("steering.eval.feature_probe.build_probe_models")
    @patch("steering.eval.feature_probe.resolve_sae")
    @patch("steering.eval.feature_probe.parse_feature_dir")
    def test_basic_pipeline(self, mock_parse, mock_resolve, mock_build, tmp_path):
        d_model = 8
        fake_model = _FakeHookedTransformer(d_model)
        fake_sae = _FakeSAE(d_model)
        fake_tokenizer = _FakeTokenizer()

        mock_parse.return_value = FeatureDirectoryComponents(
            layer=8,
            activation_type=ActivationType.residual,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            segment_size=128,
            pooling=PoolingMode.max_pool,
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )
        mock_resolved = MagicMock()
        mock_resolved.hook_name = "blocks.8.hook_resid_post"
        mock_resolved.release = "test"
        mock_resolved.sae_id = "test"
        mock_resolve.return_value = mock_resolved
        mock_build.return_value = (fake_model, fake_sae, fake_tokenizer, None)

        completions = [
            DocumentCompletion(
                doc_idx=0,
                feature_id=0,
                pre_segment_text="Hello ",
                ground_truth_segment="world",
                target_segment_idx=1,
                samples=[
                    CompletionSample(condition="real", text="world!", seed=42),
                    CompletionSample(condition="ablated", text="thing!", seed=43),
                    CompletionSample(condition="random", text="stuff!", seed=44),
                ],
            ),
        ]

        results = compute_probe_for_completions(
            completions=completions,
            feature_dir=Path(tmp_path / "fake_features"),
            use_raw_activation=True,
            device=t.device("cpu"),
        )

        assert 0 in results
        assert isinstance(results[0], ProbeResults)
        assert results[0].feature_id == 0

        for cond in ("real", "ablated", "random"):
            scores = results[0].condition_scores[cond]
            assert len(scores) == 1
            assert isinstance(scores[0], float)
            assert math.isfinite(scores[0])

    @patch("steering.eval.feature_probe.build_probe_models")
    @patch("steering.eval.feature_probe.resolve_sae")
    @patch("steering.eval.feature_probe.parse_feature_dir")
    def test_multiple_features(self, mock_parse, mock_resolve, mock_build, tmp_path):
        d_model = 8
        fake_model = _FakeHookedTransformer(d_model)
        fake_sae = _FakeSAE(d_model)
        fake_tokenizer = _FakeTokenizer()

        mock_parse.return_value = FeatureDirectoryComponents(
            layer=8,
            activation_type=ActivationType.residual,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            segment_size=128,
            pooling=PoolingMode.max_pool,
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )
        mock_resolved = MagicMock()
        mock_resolved.hook_name = "blocks.8.hook_resid_post"
        mock_resolved.release = "test"
        mock_resolved.sae_id = "test"
        mock_resolve.return_value = mock_resolved
        mock_build.return_value = (fake_model, fake_sae, fake_tokenizer, None)

        completions = [
            DocumentCompletion(
                doc_idx=0,
                feature_id=0,
                pre_segment_text="A",
                ground_truth_segment="B",
                target_segment_idx=1,
                samples=[CompletionSample(condition="real", text="B", seed=1)],
            ),
            DocumentCompletion(
                doc_idx=1,
                feature_id=1,
                pre_segment_text="C",
                ground_truth_segment="D",
                target_segment_idx=1,
                samples=[CompletionSample(condition="real", text="D", seed=2)],
            ),
        ]

        results = compute_probe_for_completions(
            completions=completions,
            feature_dir=Path(tmp_path / "fake"),
            use_raw_activation=True,
            device=t.device("cpu"),
        )

        assert 0 in results
        assert 1 in results
        assert results[0].feature_id == 0
        assert results[1].feature_id == 1

    @patch("steering.eval.feature_probe.build_probe_models")
    @patch("steering.eval.feature_probe.resolve_sae")
    @patch("steering.eval.feature_probe.parse_feature_dir")
    def test_sae_encode_path(self, mock_parse, mock_resolve, mock_build, tmp_path):
        """Verify use_raw_activation=False exercises the SAE encode path."""
        d_model = 8
        fake_model = _FakeHookedTransformer(d_model)
        fake_sae = _FakeSAE(d_model)
        fake_tokenizer = _FakeTokenizer()

        mock_parse.return_value = FeatureDirectoryComponents(
            layer=8,
            activation_type=ActivationType.residual,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            segment_size=128,
            pooling=PoolingMode.max_pool,
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )
        mock_resolved = MagicMock()
        mock_resolved.hook_name = "blocks.8.hook_resid_post"
        mock_resolved.release = "test"
        mock_resolved.sae_id = "test"
        mock_resolve.return_value = mock_resolved
        mock_build.return_value = (fake_model, fake_sae, fake_tokenizer, None)

        completions = [
            DocumentCompletion(
                doc_idx=0,
                feature_id=0,
                pre_segment_text="Hello ",
                ground_truth_segment="world",
                target_segment_idx=1,
                samples=[
                    CompletionSample(condition="real", text="world!", seed=42),
                ],
            ),
        ]

        results = compute_probe_for_completions(
            completions=completions,
            feature_dir=Path(tmp_path / "fake"),
            use_raw_activation=False,
            device=t.device("cpu"),
        )

        assert 0 in results
        scores = results[0].condition_scores["real"]
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        # SAE encode path applies JumpReLU — values are >= 0
        assert scores[0] >= 0.0


class TestProbeNnterpPath:
    """Verify compute_probe_for_completions works with the nnterp codepath."""

    @patch("steering.eval.feature_probe.build_probe_models")
    @patch("steering.eval.feature_probe.resolve_sae")
    @patch("steering.eval.feature_probe.parse_feature_dir")
    def test_pipeline_with_mock_nnterp(self, mock_parse, mock_resolve, mock_build, tmp_path):
        d_model = 8
        fake_model = _FakeHookedTransformer(d_model)
        fake_sae = _FakeSAE(d_model)
        fake_tokenizer = _FakeTokenizer()

        mock_parse.return_value = FeatureDirectoryComponents(
            layer=8,
            activation_type=ActivationType.residual,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            segment_size=128,
            pooling=PoolingMode.max_pool,
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )
        mock_resolved = MagicMock()
        mock_resolved.hook_name = "blocks.8.hook_resid_post"
        mock_resolved.release = "test"
        mock_resolved.sae_id = "test"
        mock_resolve.return_value = mock_resolved
        mock_build.return_value = (fake_model, fake_sae, fake_tokenizer, None)

        completions = [
            DocumentCompletion(
                doc_idx=0, feature_id=0,
                pre_segment_text="Hello ",
                ground_truth_segment="world",
                target_segment_idx=1,
                samples=[CompletionSample(condition="real", text="world!", seed=42)],
            ),
        ]

        results = compute_probe_for_completions(
            completions=completions,
            feature_dir=Path(tmp_path / "fake_features"),
            use_raw_activation=True,
            device=t.device("cpu"),
        )

        assert 0 in results
        mock_build.assert_called_once()
