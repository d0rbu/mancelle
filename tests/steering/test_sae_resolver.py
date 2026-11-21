"""Unit tests for steering.sae_resolver.

Run with:
    uv run pytest tests/test_sae_resolver.py -v
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
# Enum smoke tests
# ---------------------------------------------------------------------------


class TestActivationType:
    def test_enum_values(self):
        assert ActivationType.residual.value == "residual"
        assert ActivationType.mlp.value == "mlp"
        assert ActivationType.attention.value == "attention"

    def test_round_trip(self):
        for member in ActivationType:
            assert ActivationType(member.value) is member


class TestSAEWidth:
    def test_enum_values(self):
        assert SAEWidth.W_16K.value == "16k"
        assert SAEWidth.W_65K.value == "65k"
        assert SAEWidth.W_1M.value == "1m"

    def test_round_trip(self):
        for member in SAEWidth:
            assert SAEWidth(member.value) is member


class TestSAERelease:
    def test_enum_values(self):
        assert SAERelease.gemmascope.value == "gemmascope"


# ---------------------------------------------------------------------------
# resolve_sae — GemmaScope happy paths
# ---------------------------------------------------------------------------


class TestResolveSAE:
    def test_gemmascope_residual_16k_canonical(self):
        r = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_16K,
        )
        assert r.release == "gemma-scope-2b-pt-res-canonical"
        assert r.sae_id == "layer_12/width_16k/canonical"
        assert r.hook_name == "blocks.12.hook_resid_post"

    def test_gemmascope_mlp_16k_canonical(self):
        r = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.mlp,
            12,
            SAEWidth.W_16K,
        )
        assert r.release == "gemma-scope-2b-pt-mlp-canonical"
        assert r.sae_id == "layer_12/width_16k/canonical"
        assert r.hook_name == "blocks.12.hook_mlp_out"

    def test_gemmascope_attention_16k_canonical(self):
        r = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.attention,
            12,
            SAEWidth.W_16K,
        )
        assert r.release == "gemma-scope-2b-pt-att-canonical"
        assert r.sae_id == "layer_12/width_16k/canonical"
        assert r.hook_name == "blocks.12.attn.hook_z"

    def test_gemmascope_with_l0(self):
        r = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_65K,
            l0=62,
        )
        assert r.release == "gemma-scope-2b-pt-res"
        assert r.sae_id == "layer_12/width_65k/average_l0_62"
        assert r.hook_name == "blocks.12.hook_resid_post"

    def test_gemmascope_residual_1m_layer_12(self):
        r = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_1M,
        )
        assert r.release == "gemma-scope-2b-pt-res-canonical"
        assert r.sae_id == "layer_12/width_1m/canonical"

    def test_gemmascope_9b_model(self):
        r = resolve_sae(
            "gemma-2-9b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_16K,
        )
        assert r.release == "gemma-scope-9b-pt-res-canonical"
        assert "layer_12" in r.sae_id

    def test_gemmascope_27b_model(self):
        # v1 27b ships only residual SAEs at width 131k on layers 10/22/34
        # (T12 backfill). mlp / 16k variants are not published.
        r = resolve_sae(
            "gemma-2-27b",
            SAERelease.gemmascope,
            ActivationType.residual,
            10,
            SAEWidth.W_131K,
        )
        assert r.release == "gemma-scope-27b-pt-res-canonical"

    def test_resolved_sae_is_frozen(self):
        r = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_16K,
        )
        with pytest.raises(AttributeError):
            r.release = "something"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# resolve_sae — validation errors
# ---------------------------------------------------------------------------


class TestResolveSAEValidation:
    @pytest.mark.parametrize(
        ("activation_type", "width"),
        [
            (ActivationType.residual, SAEWidth.W_16K),
            (ActivationType.residual, SAEWidth.W_65K),
            (ActivationType.mlp, SAEWidth.W_16K),
            (ActivationType.mlp, SAEWidth.W_65K),
            (ActivationType.attention, SAEWidth.W_16K),
            (ActivationType.attention, SAEWidth.W_65K),
        ],
    )
    def test_gemma_2_2b_all_layer_combos_are_bounded(
        self, activation_type, width
    ):
        for layer in (0, 25):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                activation_type,
                layer,
                width,
            )

        for layer in (-1, 26):
            with pytest.raises(ValueError, match=rf"Layer {layer}"):
                resolve_sae(
                    "gemma-2-2b",
                    SAERelease.gemmascope,
                    activation_type,
                    layer,
                    width,
                )

    def test_invalid_width_for_mlp(self):
        with pytest.raises(ValueError, match="32k"):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                ActivationType.mlp,
                12,
                SAEWidth.W_32K,
            )

    def test_invalid_width_for_attention(self):
        with pytest.raises(ValueError, match="32k"):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                ActivationType.attention,
                12,
                SAEWidth.W_32K,
            )

    def test_invalid_layer_for_1m(self):
        with pytest.raises(ValueError, match="layer"):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                ActivationType.residual,
                3,
                SAEWidth.W_1M,
            )

    def test_invalid_layer_for_32k(self):
        with pytest.raises(ValueError, match="layer"):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                ActivationType.residual,
                0,
                SAEWidth.W_32K,
            )

    def test_unsupported_model(self):
        with pytest.raises(ValueError, match="model"):
            resolve_sae(
                "llama-7b",
                SAERelease.gemmascope,
                ActivationType.residual,
                12,
                SAEWidth.W_16K,
            )

    def test_valid_layers_for_1m_listed_in_error(self):
        with pytest.raises(ValueError, match=r"\[12\]"):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                ActivationType.residual,
                0,
                SAEWidth.W_1M,
            )

    def test_valid_widths_listed_in_error(self):
        with pytest.raises(ValueError, match="16k"):
            resolve_sae(
                "gemma-2-2b",
                SAERelease.gemmascope,
                ActivationType.mlp,
                12,
                SAEWidth.W_32K,
            )


# ---------------------------------------------------------------------------
# parse_neuronpedia_source (reverse of resolve_neuronpedia_ids)
# ---------------------------------------------------------------------------


class TestParseNeuronpediaSource:
    """The parser inverts :func:`resolve_neuronpedia_ids`'s ``sae_source`` format."""

    def test_parses_canonical_gemmascope_residual_16k(self):
        release, activation, layer, width, l0_tier = parse_neuronpedia_source(
            "8-gemmascope-res-16k",
        )
        assert release is SAERelease.gemmascope
        assert activation is ActivationType.residual
        assert layer == 8
        assert width is SAEWidth.W_16K
        assert l0_tier is None

    def test_parses_mlp_activation(self):
        _, activation, _, _, _ = parse_neuronpedia_source("12-gemmascope-mlp-65k")
        assert activation is ActivationType.mlp

    def test_parses_attention_activation(self):
        _, activation, _, _, _ = parse_neuronpedia_source("5-gemmascope-att-16k")
        assert activation is ActivationType.attention

    def test_parses_layer_zero(self):
        _, _, layer, _, _ = parse_neuronpedia_source("0-gemmascope-res-16k")
        assert layer == 0

    def test_parses_large_layer_index(self):
        _, _, layer, _, _ = parse_neuronpedia_source("25-gemmascope-res-16k")
        assert layer == 25

    @pytest.mark.parametrize(
        "width_str, expected",
        [
            ("16k", SAEWidth.W_16K),
            ("32k", SAEWidth.W_32K),
            ("65k", SAEWidth.W_65K),
            ("131k", SAEWidth.W_131K),
            ("262k", SAEWidth.W_262K),
            ("524k", SAEWidth.W_524K),
            ("1m", SAEWidth.W_1M),
        ],
    )
    def test_parses_every_width(self, width_str, expected):
        _, _, _, width, _ = parse_neuronpedia_source(
            f"12-gemmascope-res-{width_str}",
        )
        assert width is expected

    def test_round_trip_via_resolve_neuronpedia_ids(self):
        """parse ∘ resolve = identity on the source string."""
        for layer in (0, 8, 12, 25):
            for act in ActivationType:
                for width in SAEWidth:
                    try:
                        resolve_sae("gemma-2-2b", SAERelease.gemmascope,
                                    act, layer, width)
                    except ValueError:
                        # Skip invalid combos (the resolver rejects them).
                        continue
                    _, source = resolve_neuronpedia_ids(
                        "gemma-2-2b", SAERelease.gemmascope, act, layer, width,
                    )
                    release_back, act_back, layer_back, width_back, _ = (
                        parse_neuronpedia_source(source)
                    )
                    assert release_back is SAERelease.gemmascope
                    assert act_back is act
                    assert layer_back == layer
                    assert width_back is width

    def test_rejects_too_few_segments(self):
        with pytest.raises(ValueError, match="segments"):
            parse_neuronpedia_source("8-gemmascope-res")

    def test_rejects_too_many_segments(self):
        with pytest.raises(ValueError, match="segments"):
            parse_neuronpedia_source("8-gemmascope-res-16k-extra")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="segments"):
            parse_neuronpedia_source("")

    def test_rejects_non_integer_layer(self):
        with pytest.raises(ValueError, match="must be int"):
            parse_neuronpedia_source("abc-gemmascope-res-16k")

    def test_rejects_unknown_release(self):
        with pytest.raises(ValueError, match="unknown release"):
            parse_neuronpedia_source("8-bogus-res-16k")

    def test_rejects_unknown_activation_suffix(self):
        with pytest.raises(ValueError, match="unknown activation suffix"):
            parse_neuronpedia_source("8-gemmascope-bogus-16k")

    def test_rejects_unknown_width(self):
        with pytest.raises(ValueError, match="unknown width"):
            parse_neuronpedia_source("8-gemmascope-res-bogus")

    def test_preserves_chain_to_resolve_sae(self):
        """The parser's output feeds :func:`resolve_sae` cleanly.

        This is the usage contract: callers parse the Neuronpedia-format
        string, then pass the tuple to :func:`resolve_sae` to get the
        SAELens release + sae_id + hook_name. If either end drifts the
        round-trip stops working and this test catches it.
        """
        release, activation, layer, width, _ = parse_neuronpedia_source(
            "8-gemmascope-res-16k",
        )
        resolved = resolve_sae(
            "gemma-2-2b", release, activation, layer, width, l0=None,
        )
        assert resolved.release == "gemma-scope-2b-pt-res-canonical"
        assert resolved.sae_id == "layer_8/width_16k/canonical"
        assert resolved.hook_name == "blocks.8.hook_resid_post"
