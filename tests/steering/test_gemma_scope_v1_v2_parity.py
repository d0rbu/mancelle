"""Cross-release behavioural parity tests — v1 paths and v2 paths must
share no release strings and must not cross-contaminate."""

import pytest

from steering.feature_paths import sanitize_model_name, unsanitize_model_name
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
    resolve_sae,
)
from steering.sae_resolver_v2_data import (
    GEMMASCOPE_2_MODEL_SIZES,
    GEMMASCOPE_2_VALID_COMBOS,
)
from tests.steering.test_sae_resolver_v2_unit import _enumerate_all_v2_combos


def _enumerate_published_v2_neuronpedia_combos():
    for model, activation, layer, width, tier in _enumerate_all_v2_combos():
        try:
            resolve_neuronpedia_ids(
                model=model,
                release=SAERelease.gemmascope_2,
                activation_type=activation,
                layer=layer,
                width=width,
                l0=tier,
            )
        except ValueError:
            continue
        yield model, activation, layer, width, tier


class TestNoReleaseStringOverlap:
    def test_v1_and_v2_release_strings_disjoint(self):
        # Real invariant: every v2 release string starts with
        # "gemma-scope-2-" and NO v1 release does. Iterate ALL v2 models
        # (derived from the data module) at one representative layer each.
        v1_releases = set()
        for model in ("gemma-2-2b", "gemma-2-9b", "gemma-2-27b"):
            layer = {"gemma-2-2b": 12, "gemma-2-9b": 20, "gemma-2-27b": 22}[
                model
            ]
            width = (
                SAEWidth.W_131K if model == "gemma-2-27b" else SAEWidth.W_16K
            )
            r = resolve_sae(
                model=model,
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=layer,
                width=width,
                l0=None,
            )
            v1_releases.add(r.release)

        v2_releases = set()
        for model in GEMMASCOPE_2_MODEL_SIZES:
            # First valid (act, width, tier, layer) cell for this model.
            (act, width), tier_layers = next(
                iter(GEMMASCOPE_2_VALID_COMBOS[model].items())
            )
            tier, layers = next(iter(tier_layers.items()))
            r = resolve_sae(
                model=model,
                release=SAERelease.gemmascope_2,
                activation_type=act,
                layer=sorted(layers)[0],
                width=width,
                l0=tier,
            )
            v2_releases.add(r.release)

        for rel in v2_releases:
            assert rel.startswith("gemma-scope-2-"), (
                f"v2 release {rel!r} must start with 'gemma-scope-2-'"
            )
        for rel in v1_releases:
            assert not rel.startswith("gemma-scope-2-"), (
                f"v1 release {rel!r} must NOT start with 'gemma-scope-2-'"
            )
        assert v1_releases.isdisjoint(v2_releases), (
            f"unexpected overlap: {v1_releases & v2_releases}"
        )

    def test_v1_resolve_unchanged_after_v2_added(self):
        # Goldens for v1 must still match (regression catch).
        r = resolve_sae(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=8,
            width=SAEWidth.W_16K,
            l0=None,
        )
        assert r.release == "gemma-scope-2b-pt-res-canonical"
        assert r.sae_id == "layer_8/width_16k/canonical"
        assert r.hook_name == "blocks.8.hook_resid_post"

    def test_v2_resolve_golden(self):
        r = resolve_sae(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=SAEWidth.W_16K,
            l0=L0Tier.medium,
        )
        assert r.release == "gemma-scope-2-4b-it-res"
        assert r.sae_id == "layer_22_width_16k_l0_medium"
        assert r.hook_name == "blocks.22.hook_resid_post"


class TestNeuronpediaRoundTripBothReleases:
    def test_v1_round_trip(self):
        model_id, source = resolve_neuronpedia_ids(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=8,
            width=SAEWidth.W_16K,
        )
        release, act, layer, width, tier = parse_neuronpedia_source(source)
        assert (release, act, layer, width, tier) == (
            SAERelease.gemmascope,
            ActivationType.residual,
            8,
            SAEWidth.W_16K,
            None,
        )

    def test_v2_round_trip(self):
        model_id, source = resolve_neuronpedia_ids(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=SAEWidth.W_16K,
            l0=L0Tier.medium,
        )
        release, act, layer, width, tier = parse_neuronpedia_source(source)
        assert (release, act, layer, width, tier) == (
            SAERelease.gemmascope_2,
            ActivationType.residual,
            22,
            SAEWidth.W_16K,
            L0Tier.medium,
        )


@pytest.mark.parametrize(
    "model,activation,layer,width,tier",
    list(_enumerate_published_v2_neuronpedia_combos()),
)
def test_neuronpedia_roundtrip_published_v2_matrix(
    model, activation, layer, width, tier
):
    """Every registry-published v2 Neuronpedia cell must round-trip."""
    _model_id, source = resolve_neuronpedia_ids(
        model=model,
        release=SAERelease.gemmascope_2,
        activation_type=activation,
        layer=layer,
        width=width,
        l0=tier,
    )
    parsed = parse_neuronpedia_source(source)
    assert parsed == (
        SAERelease.gemmascope_2,
        activation,
        layer,
        width,
        tier,
    )


class TestFeaturePathRoundTripBothReleases:
    @pytest.mark.parametrize(
        "model",
        sorted(GEMMASCOPE_2_MODEL_SIZES.keys()),
    )
    def test_v2_model_name_round_trips(self, model):
        assert unsanitize_model_name(sanitize_model_name(model)) == model

    @pytest.mark.parametrize(
        "model",
        ["gemma-2-2b", "gemma-2-9b", "gemma-2-27b"],
    )
    def test_v1_model_name_round_trips(self, model):
        assert unsanitize_model_name(sanitize_model_name(model)) == model
