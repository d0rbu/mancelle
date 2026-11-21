"""Unit tests for steering/sae_resolver.py — slice 01 Phase 1 audit.

Pins fixes for plan items A-5 and A-6: validation in resolve_neuronpedia_ids;
documented permissive behavior for unknown gemma sizes.
"""

import pytest

from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
    resolve_sae,
)

# ---------------------------------------------------------------------------
# A-6: resolve_neuronpedia_ids must validate
# ---------------------------------------------------------------------------


class TestResolveNeuronpediaIdsValidates:
    """Per plan A-6: resolve_neuronpedia_ids should call _validate_gemmascope."""

    def test_invalid_combo_raises_for_gemma_2b(self):
        """gemma-2-2b residual+32k+layer 8 is invalid (only layer 12 is allowed).

        Currently resolve_neuronpedia_ids silently produces a string for any
        combination; after fix it must raise the same way resolve_sae does.
        """
        with pytest.raises(ValueError):
            resolve_neuronpedia_ids(
                model="gemma-2-2b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=8,
                width=SAEWidth.W_32K,
            )

    def test_valid_combo_succeeds(self):
        """gemma-2-2b residual+16k+layer 8 is valid; should succeed."""
        model_id, sae_source = resolve_neuronpedia_ids(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=8,
            width=SAEWidth.W_16K,
        )
        assert model_id == "gemma-2-2b"
        # Format: {layer}-{release}-{suffix}-{width}
        assert sae_source == "8-gemmascope-res-16k"

    def test_valid_layer_12_residual_32k(self):
        """gemma-2-2b residual+32k+layer 12 is valid (special case)."""
        model_id, sae_source = resolve_neuronpedia_ids(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=12,
            width=SAEWidth.W_32K,
        )
        assert sae_source == "12-gemmascope-res-32k"


# ---------------------------------------------------------------------------
# A-5: gemma-2-9b/27b permissive validation is documented (no breaking change)
# ---------------------------------------------------------------------------


# NOTE: `TestGemma9bPermissiveValidation` was DELETED. It asserted the old
# contract "gemma-2-9b/27b have no validation table; resolve_sae allows all
# combos", which T12 removed (9b/27b are now strictly validated). The strict
# behavior is covered by `TestGemma29bValidation` / `TestGemma227bValidation`
# in this file and in test_sae_resolver_v2_unit.py.


# ---------------------------------------------------------------------------
# A-10: parse_neuronpedia_source / resolve_neuronpedia_ids round-trip
# ---------------------------------------------------------------------------


class TestNeuronpediaRoundTrip:
    """parse_neuronpedia_source should be the inverse of resolve_neuronpedia_ids."""

    def test_basic_roundtrip(self):
        original = (
            SAERelease.gemmascope,
            ActivationType.residual,
            8,
            SAEWidth.W_16K,
        )
        _, source = resolve_neuronpedia_ids(
            model="gemma-2-2b",
            release=original[0],
            activation_type=original[1],
            layer=original[2],
            width=original[3],
        )
        parsed = parse_neuronpedia_source(source)
        assert parsed == (*original, None)

    def test_parse_invalid_raises_clean_error(self):
        """Malformed source string with wrong segment count raises typed ValueError."""
        # 5 dash-separated segments -> wrong segment count
        with pytest.raises(ValueError, match="segments"):
            parse_neuronpedia_source("8-gemmascope-res-16k-extra")
        # 1 segment (no dashes) -> wrong segment count
        with pytest.raises(ValueError, match="segments"):
            parse_neuronpedia_source("nodashes")

    def test_parse_invalid_layer_raises_clean_error(self):
        with pytest.raises(ValueError, match="must be int"):
            parse_neuronpedia_source("not_int-gemmascope-res-16k")

    def test_parse_invalid_release_raises_clean_error(self):
        with pytest.raises(ValueError, match="release"):
            parse_neuronpedia_source("8-fake_release-res-16k")

    def test_parse_invalid_activation_raises_clean_error(self):
        with pytest.raises(ValueError, match="activation"):
            parse_neuronpedia_source("8-gemmascope-fake_act-16k")

    def test_parse_invalid_width_raises_clean_error(self):
        with pytest.raises(ValueError, match="width"):
            parse_neuronpedia_source("8-gemmascope-res-99k")


# ---------------------------------------------------------------------------
# T12: v1 gemma-2-9b / gemma-2-27b backfill validation
# ---------------------------------------------------------------------------


class TestGemma29bValidation:
    def test_9b_valid_combo_resolves(self):
        resolved = resolve_sae(
            model="gemma-2-9b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=20,
            width=SAEWidth.W_131K,
            l0=None,
        )
        assert "gemma-scope-9b-pt-res" in resolved.release

    def test_9b_layer_out_of_range_raises(self):
        # 9b has 42 layers (0-41). Layer 42 invalid.
        with pytest.raises(ValueError, match=r"Layer 42"):
            resolve_sae(
                model="gemma-2-9b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=42,
                width=SAEWidth.W_131K,
                l0=None,
            )

    def test_9b_canonical_rejects_262k_residual(self):
        # 9b-pt-res-canonical does NOT include 262k.
        with pytest.raises(ValueError, match=r"262k.*canonical"):
            resolve_sae(
                model="gemma-2-9b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=20,
                width=SAEWidth.W_262K,
                l0=None,  # canonical
            )

    def test_9b_non_canonical_accepts_262k_residual(self):
        # With an explicit average_l0, 262k IS published for 9b residual.
        resolved = resolve_sae(
            model="gemma-2-9b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=20,
            width=SAEWidth.W_262K,
            l0=176,
        )
        assert "average_l0_176" in resolved.sae_id


class TestGemma227bValidation:
    def test_27b_layer_10_residual_131k_resolves(self):
        resolved = resolve_sae(
            model="gemma-2-27b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=10,
            width=SAEWidth.W_131K,
            l0=None,
        )
        assert "gemma-scope-27b-pt-res" in resolved.release

    def test_27b_unsupported_layer_raises(self):
        with pytest.raises(ValueError, match=r"Layer 5"):
            resolve_sae(
                model="gemma-2-27b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=5,
                width=SAEWidth.W_131K,
                l0=None,
            )

    def test_27b_unsupported_activation_raises(self):
        with pytest.raises(ValueError, match=r"mlp.*not published"):
            resolve_sae(
                model="gemma-2-27b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.mlp,
                layer=10,
                width=SAEWidth.W_131K,
                l0=None,
            )


class TestV1NeuronpediaRegression:
    def test_9b_262k_residual_neuronpedia_source_not_rejected(self):
        # Regression: Neuronpedia sources carry no l0; gemma-2-9b residual
        # @262k is a valid Neuronpedia source. The T12 canonical-262k gate
        # must NOT fire on the URL-building path (it has no l0 concept).
        model_id, source = resolve_neuronpedia_ids(
            model="gemma-2-9b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=20,
            width=SAEWidth.W_262K,
        )
        assert (model_id, source) == ("gemma-2-9b", "20-gemmascope-res-262k")

    def test_9b_canonical_gate_still_active_on_resolve_sae(self):
        # The gate must still protect resolve_sae (l0=None canonical path).
        with pytest.raises(ValueError, match="canonical"):
            resolve_sae(
                model="gemma-2-9b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=20,
                width=SAEWidth.W_262K,
                l0=None,
            )
