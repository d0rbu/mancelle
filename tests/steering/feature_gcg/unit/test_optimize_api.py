from __future__ import annotations

from types import SimpleNamespace

import pytest

import steering.conditional.feature_gcg.gradient as gradient
import steering.conditional.feature_gcg.optimize as optimize
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.optimize import _n_sae_features


def test_public_optimizer_api_shape() -> None:
    assert callable(optimize.optimize_feature_conditioning_universal)
    assert not hasattr(optimize, "optimize_feature_conditioning")
    assert optimize.__all__ == ["optimize_feature_conditioning_universal"]
    assert not hasattr(gradient, "assert_encoder_not_compacted")


def test_n_sae_features_reports_a_malformed_model() -> None:
    model = SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace()))
    with pytest.raises(FeatureGcgError) as exc_info:
        _n_sae_features(model)

    assert str(exc_info.value) == (
        "Cannot read n_sae_features from model.encoder.config.n_sae_features: "
        "'types.SimpleNamespace' object has no attribute 'n_sae_features'. "
        "Is this an encoder-decoder model with a loaded encoder?"
    )
