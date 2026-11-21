# tests/test_probe_activations.py
"""Unit tests for the probe activations dispatcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch as t

from errors import ExtractionError
from steering.sae_resolver import ResolvedSAE


class FakeHookedTransformer:
    def __init__(self, d_model: int = 8):
        self.d_model = d_model
        self._param = t.nn.Parameter(t.zeros(1))

    def parameters(self):
        return iter([self._param])

    def run_with_cache(self, tokens, names_filter=None, attention_mask=None, stop_at_layer=None):
        B, T = tokens.shape
        act = t.randn(B, T, self.d_model)
        cache = {names_filter[0]: act} if names_filter else {}
        return None, cache


class FakeTextTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode()


class FakeNnterpModel:
    """Minimal stand-in for nnterp StandardizedTransformer."""

    class model:
        """Empty HF model stub — no layers attribute, so trim_hf_model is a no-op."""


@pytest.fixture
def tl_model():
    return FakeHookedTransformer(d_model=8)


@pytest.fixture
def nn_model():
    return FakeNnterpModel()


class TestGetProbeActivations:
    @patch("steering.eval.probe_activations._get_activations_nnterp")
    @patch("steering.eval.probe_activations._get_activations_tl")
    @patch("steering.eval.probe_activations.maybe_gc_cuda")
    def test_nnterp_used_when_available(self, mock_gc, mock_tl, mock_nn, tl_model, nn_model):
        from steering.eval.probe_activations import get_probe_activations
        from steering.sae_resolver import ActivationType

        mock_nn.return_value = t.randn(1, 5, 8)
        tokens = t.zeros(1, 5, dtype=t.long)
        result = get_probe_activations(
            tl_model,
            tokens,
            layer=8,
            hook_name="blocks.8.hook_resid_post",
            activation_type=ActivationType.residual,
            nnterp_model=nn_model,
        )
        mock_nn.assert_called_once()
        mock_tl.assert_not_called()
        assert result.shape == (1, 5, 8)

    @patch("steering.eval.probe_activations._get_activations_nnterp")
    @patch("steering.eval.probe_activations._get_activations_tl")
    @patch("steering.eval.probe_activations.maybe_gc_cuda")
    def test_tl_fallback_on_nnterp_error(self, mock_gc, mock_tl, mock_nn, tl_model, nn_model):
        from steering.eval.probe_activations import get_probe_activations
        from steering.sae_resolver import ActivationType

        mock_nn.side_effect = ExtractionError("nnterp failed")
        mock_tl.return_value = t.randn(1, 5, 8)
        tokens = t.zeros(1, 5, dtype=t.long)
        result = get_probe_activations(
            tl_model,
            tokens,
            layer=8,
            hook_name="blocks.8.hook_resid_post",
            activation_type=ActivationType.residual,
            nnterp_model=nn_model,
        )
        mock_nn.assert_called_once()
        mock_tl.assert_called_once()
        assert result.shape == (1, 5, 8)

    @patch("steering.eval.probe_activations._get_activations_nnterp")
    @patch("steering.eval.probe_activations._get_activations_tl")
    def test_released_transformerlens_fallback_fails_loudly(
        self, mock_tl, mock_nn, nn_model,
    ):
        from steering.eval.probe_activations import get_probe_activations
        from steering.sae_resolver import ActivationType

        mock_nn.side_effect = ExtractionError("nnterp failed")
        tokens = t.zeros(1, 5, dtype=t.long)
        with pytest.raises(ExtractionError, match="fallback was released"):
            get_probe_activations(
                None,
                tokens,
                layer=8,
                hook_name="blocks.8.hook_resid_post",
                activation_type=ActivationType.residual,
                nnterp_model=nn_model,
            )
        mock_tl.assert_not_called()

    @patch("steering.eval.probe_activations._get_activations_tl")
    def test_tl_used_when_nnterp_none(self, mock_tl, tl_model):
        from steering.eval.probe_activations import get_probe_activations
        from steering.sae_resolver import ActivationType

        mock_tl.return_value = t.randn(1, 5, 8)
        tokens = t.zeros(1, 5, dtype=t.long)
        result = get_probe_activations(
            tl_model,
            tokens,
            layer=8,
            hook_name="blocks.8.hook_resid_post",
            activation_type=ActivationType.residual,
            nnterp_model=None,
        )
        mock_tl.assert_called_once()
        assert result.shape == (1, 5, 8)

    @patch("steering.eval.probe_activations._get_activations_nnterp")
    @patch("steering.eval.probe_activations._get_activations_tl")
    @patch("steering.eval.probe_activations.maybe_gc_cuda")
    def test_gc_called_after_nnterp_success(self, mock_gc, mock_tl, mock_nn, tl_model, nn_model):
        from steering.eval.probe_activations import get_probe_activations
        from steering.sae_resolver import ActivationType

        mock_nn.return_value = t.randn(1, 5, 8)
        tokens = t.zeros(1, 5, dtype=t.long)
        get_probe_activations(
            tl_model,
            tokens,
            layer=8,
            hook_name="blocks.8.hook_resid_post",
            activation_type=ActivationType.residual,
            nnterp_model=nn_model,
        )
        mock_gc.assert_called_once()


class TestTrimHfModel:
    """Verify HF model trimming deletes layers beyond target."""

    def test_trims_module_list(self):
        import torch.nn as nn

        from steering.eval.probe_activations import trim_hf_model

        class FakeHFModel:
            class model:
                layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

        fake = FakeHFModel()
        trim_hf_model(fake, layer=3)
        assert len(fake.model.layers) == 4  # layers 0, 1, 2, 3

    def test_trim_noop_if_already_short(self):
        import torch.nn as nn

        from steering.eval.probe_activations import trim_hf_model

        class FakeHFModel:
            class model:
                layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        fake = FakeHFModel()
        trim_hf_model(fake, layer=5)
        assert len(fake.model.layers) == 3  # unchanged

    def test_trim_gpt2_transformer_h(self):
        """GPT-2 uses transformer.h instead of model.layers."""
        import torch.nn as nn

        from steering.eval.probe_activations import trim_hf_model

        class FakeGPT2:
            class transformer:
                h = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

            model = None

        fake = FakeGPT2()
        trim_hf_model(fake, layer=3)
        assert len(fake.transformer.h) == 4


class TestBuildProbeModels:
    """Verify factory returns correct tuple with nnterp model."""

    @staticmethod
    def _resolved() -> ResolvedSAE:
        return ResolvedSAE(
            release="test-release",
            sae_id="test-sae-id",
            hook_name="blocks.8.hook_resid_post",
        )

    @patch("steering.eval.probe_activations.build_nnterp_model")
    @patch("steering.eval.probe_activations.load_sae_and_model")
    def test_returns_nnterp_on_cuda(self, mock_load, mock_build):
        from steering.eval.probe_activations import build_probe_models

        mock_load.return_value = (
            FakeHookedTransformer(),
            MagicMock(),
            FakeTextTokenizer(),
        )
        mock_build.return_value = FakeNnterpModel()
        resolved = self._resolved()
        tl, sae, tok, nn = build_probe_models(
            model_name="gpt2",
            layer=8,
            device="cuda",
            resolved=resolved,
        )
        assert nn is not None
        mock_load.assert_called_once_with(
            model_name="gpt2",
            layer=8,
            device="cuda",
            resolved=resolved,
        )
        mock_build.assert_called_once()

    @patch("steering.eval.probe_activations.build_nnterp_model")
    @patch("steering.eval.probe_activations.load_sae_and_model")
    def test_returns_none_on_cpu(self, mock_load, mock_build):
        from steering.eval.probe_activations import build_probe_models

        mock_load.return_value = (
            FakeHookedTransformer(),
            MagicMock(),
            FakeTextTokenizer(),
        )
        resolved = self._resolved()
        tl, sae, tok, nn = build_probe_models(
            model_name="gpt2",
            layer=8,
            device="cpu",
            resolved=resolved,
        )
        assert nn is None
        mock_build.assert_not_called()

    @patch("steering.eval.probe_activations.build_nnterp_model")
    @patch("steering.eval.probe_activations.load_sae_and_model")
    def test_returns_none_when_build_fails(self, mock_load, mock_build):
        from steering.eval.probe_activations import build_probe_models

        mock_load.return_value = (
            FakeHookedTransformer(),
            MagicMock(),
            FakeTextTokenizer(),
        )
        mock_build.return_value = None
        resolved = self._resolved()
        tl, sae, tok, nn = build_probe_models(
            model_name="gpt2",
            layer=8,
            device="cuda",
            resolved=resolved,
        )
        assert nn is None
