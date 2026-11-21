"""Error-path coverage: every currently-untested `raise` in the
gemma-scope-2 surface, asserting BOTH the exception type AND a message
substring. Default suite, no network.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch as t
from nnterp import StandardizedTransformer
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

import steering.sae_resolver_v2_registry as v2_registry
from errors import ExtractionError
from steering.extract_features import (
    Gemma3LoadError,
    _build_nnterp_gemma3,
    _get_activations_nnterp,
)
from steering.feature_io import save_features
from steering.feature_paths import (
    ActivationType as FPActivationType,
)
from steering.feature_paths import (
    DatasetPhase,
    FeatureDirectoryMismatchError,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
)
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    _resolve_gemmascope_2,
    _validate_gemmascope,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
    resolve_sae,
)

_TINY_VOCAB = 128
_TINY_HIDDEN = 32
_TINY_LAYERS = 4
_TINY_SEQ = 5


@pytest.fixture(scope="module")
def tiny_gemma3_nnterp() -> StandardizedTransformer:
    cfg = Gemma3TextConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
    )
    return StandardizedTransformer(Gemma3ForCausalLM(cfg).eval())


# ── _validate_gemmascope / _resolve_gemmascope (v1) ─────────────────────────


class TestV1ValidationErrors:
    def test_no_validation_table_for_unknown_v1_model(self):
        with pytest.raises(ValueError, match="No validation table for v1 model"):
            _validate_gemmascope(
                "gemma-2-unknown",
                ActivationType.residual,
                10,
                SAEWidth.W_16K,
            )

    def test_activation_not_published_27b_mlp(self):
        # gemma-2-27b publishes only residual; mlp is not published.
        with pytest.raises(ValueError, match="is not published"):
            _validate_gemmascope(
                "gemma-2-27b",
                ActivationType.mlp,
                22,
                SAEWidth.W_131K,
            )

    def test_width_not_available_for_9b_mlp(self):
        # 9b mlp publishes only {16k, 131k}; 65k is unavailable.
        with pytest.raises(ValueError, match=r"Width.*is not available"):
            _validate_gemmascope(
                "gemma-2-9b",
                ActivationType.mlp,
                10,
                SAEWidth.W_65K,
            )

    def test_layer_not_available_for_27b(self):
        # 27b residual 131k only ships layers {10, 22, 34}.
        with pytest.raises(ValueError, match=r"Layer 5 is not available"):
            _validate_gemmascope(
                "gemma-2-27b",
                ActivationType.residual,
                5,
                SAEWidth.W_131K,
            )


# ── resolve_sae final guard (defensive / mocked) ────────────────────────────


class TestResolveGemmascope2DefensiveGuard:
    def test_unknown_model_defensive_guard(self):
        # `_resolve_gemmascope_2`'s own `Unsupported model` guard is
        # unreachable via resolve_sae (the family crosscheck +
        # _validate_gemmascope_2 reject first), so cover the defensive
        # branch by calling the private resolver directly.
        with pytest.raises(ValueError, match="Unsupported model"):
            _resolve_gemmascope_2(
                "gemma-3-does-not-exist",
                ActivationType.residual,
                22,
                SAEWidth.W_16K,
                L0Tier.medium,
            )


class TestResolveSaeUnsupportedRelease:
    def test_unsupported_release_raises(self):
        # Both real SAERelease members are handled, so the final
        # `raise ValueError("Unsupported SAE release: ...")` is only
        # reachable with a release object equal to neither member. We
        # construct a stand-in that compares unequal to both yet carries a
        # `.value`, exercising the defensive branch.
        fake = mock.Mock()
        fake.value = "made_up_release"
        with pytest.raises(ValueError, match="Unsupported SAE release"):
            resolve_sae(
                model="some-non-gemma-model",
                release=fake,
                activation_type=ActivationType.residual,
                layer=10,
                width=SAEWidth.W_16K,
                l0=None,
            )


# ── resolve_neuronpedia_ids ─────────────────────────────────────────────────


class TestResolveNeuronpediaIdErrors:
    def test_v2_requires_l0tier(self):
        with pytest.raises(ValueError, match="requires an L0Tier"):
            resolve_neuronpedia_ids(
                "gemma-3-4b-pt",
                SAERelease.gemmascope_2,
                ActivationType.residual,
                22,
                SAEWidth.W_16K,
                l0=176,  # int, not L0Tier
            )

    def test_unsupported_release_for_neuronpedia(self):
        fake = mock.Mock()
        fake.value = "made_up_release"
        with pytest.raises(ValueError, match="Unsupported release for Neuronpedia"):
            resolve_neuronpedia_ids(
                "gemma-3-4b-pt",
                fake,
                ActivationType.residual,
                22,
                SAEWidth.W_16K,
                l0=L0Tier.medium,
            )


# ── parse_neuronpedia_source ────────────────────────────────────────────────


class TestParseNeuronpediaSourceErrors:
    def test_v2_layer_not_int(self):
        with pytest.raises(ValueError, match="must be int"):
            parse_neuronpedia_source("xx-gemmascope-2-res-16k-medium")

    def test_five_segment_v2_defaults_to_medium(self):
        assert parse_neuronpedia_source("22-gemmascope-2-res-16k") == (
            SAERelease.gemmascope_2,
            ActivationType.residual,
            22,
            SAEWidth.W_16K,
            L0Tier.medium,
        )

    def test_unknown_activation_suffix(self):
        with pytest.raises(ValueError, match="unknown activation suffix"):
            parse_neuronpedia_source("8-gemmascope-zzz-16k")

    def test_unknown_width(self):
        with pytest.raises(ValueError, match="unknown width"):
            parse_neuronpedia_source("8-gemmascope-res-999k")

    def test_unknown_v1_release(self):
        with pytest.raises(ValueError, match="unknown release"):
            parse_neuronpedia_source("8-notarelease-res-16k")

    def test_v2_unknown_l0_tier(self):
        with pytest.raises(ValueError, match="unknown l0 tier"):
            parse_neuronpedia_source("22-gemmascope-2-res-16k-ginormous")


# ── _build_nnterp_gemma3 ────────────────────────────────────────────────────


class TestBuildNnterpGemma3Errors:
    def test_unregistered_model_raises_gemma3_load_error(self):
        with pytest.raises(
            Gemma3LoadError, match="not registered in EXTRACTION_MODELS"
        ):
            _build_nnterp_gemma3(
                model_name="gemma-3-99b",
                tl_model=None,
                device=t.device("cpu"),
                model_dtype=None,
            )


# ── _get_activations_nnterp pre-projection hook_z ───────────────────────────


class TestGetActivationsNnterpPreProjection:
    def test_pre_projection_hook_z_raises_extraction_error(
        self, tiny_gemma3_nnterp: StandardizedTransformer
    ):
        ids = t.randint(0, _TINY_VOCAB, (1, _TINY_SEQ))
        # attention branch + hook_name without `hook_attn_out` (None) drives
        # the pre-projection path that nnterp cannot expose.
        with pytest.raises(ExtractionError, match="pre-projection hook_z"):
            _get_activations_nnterp(
                tiny_gemma3_nnterp,
                ids,
                layer=2,
                activation_type=ActivationType.attention,
                hook_name=None,
            )


# ── feature_paths.verify_feature_dir invalid-tier branch ────────────────────


def _write_shard_with_bogus_l0_tier(shard_path: Path) -> None:
    """Write a valid safetensors shard then overwrite its metadata so the
    `l0_tier` header value is a bogus string `read_l0_tier_from_shard`
    rejects (drives verify_feature_dir's `<invalid>` branch)."""
    shard = {
        "feature_ids": t.tensor([1, 2, 3], dtype=t.long),
        "feature_mags": t.tensor([0.5, 0.7, 0.3], dtype=t.float32),
        "offsets": t.tensor([0, 3], dtype=t.long),
        "layer": 8,
        "segment_size": 64,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed",
        "n_docs": 1,
        "n_segments": 1,
        "context_independent_features": False,
        "l0_tier": "totally_bogus_tier",
    }
    save_features(shard, shard_path)


class TestVerifyFeatureDirInvalidTier:
    def test_invalid_tier_and_missing_stats_surface_together(
        self, tmp_path: Path
    ):
        feature_dir = build_feature_dir(
            root=tmp_path,
            segment_size=64,
            layer=8,
            activation_type=FPActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        _write_shard_with_bogus_l0_tier(feature_dir / "shard_00001_features")
        # No statistics file written → accumulated "Statistics file not found"
        # plus the invalid-tier shard error must surface together.
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            verify_feature_dir(feature_dir)
        msg = str(ei.value)
        assert "Statistics file not found" in msg
        assert "totally_bogus_tier" in msg or "l0_tier" in msg


# ── sae_resolver_v2_registry schema-drift RuntimeError ──────────────────────


class TestRegistrySchemaDrift:
    def test_dropped_field_raises_schema_drifted(self, monkeypatch):
        real_fields = v2_registry.dataclasses.fields

        def fake_fields(cls):
            flds = list(real_fields(cls))
            # Drop one field so the introspected name set no longer equals
            # the production guard constant.
            return tuple(f for f in flds if f.name != "config_overrides")

        monkeypatch.setattr(
            v2_registry.dataclasses, "fields", fake_fields
        )
        with pytest.raises(RuntimeError, match="schema drifted"):
            v2_registry.inject_missing_v2_releases()
