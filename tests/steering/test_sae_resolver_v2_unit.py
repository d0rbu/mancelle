"""Exhaustive tests for the gemma-scope-2 validation matrix and resolver."""

import itertools
import re

import pytest
from sae_lens.loading.pretrained_saes_directory import (
    get_pretrained_saes_directory,
)

from steering.models import EXTRACTION_MODELS
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    _resolve_gemmascope_2,
    _validate_gemmascope_2,
    resolve_sae,
)
from steering.sae_resolver_v2_data import (
    GEMMA2_9B_CANONICAL_WIDTHS,
    GEMMA2_9B_VALID_COMBOS,
    GEMMA2_27B_VALID_COMBOS,
    GEMMASCOPE_2_HF_MODEL_IDS,
    GEMMASCOPE_2_MODEL_SIZES,
    GEMMASCOPE_2_VALID_COMBOS,
)

# Independent oracle: HF Gemma-3 num_hidden_layers per size. Derived from the
# published Gemma-3 model configs (HF `transformers` Gemma3TextConfig for each
# checkpoint). Defined HERE in the test, NOT imported from the data module, so
# it cross-checks the production matrix rather than mirroring it.
_GEMMA3_NUM_HIDDEN_LAYERS: dict[str, int] = {
    "270m": 18,
    "1b": 26,
    "4b": 34,
    "12b": 48,
    "27b": 62,
}

# A gemma-3 model name is "gemma-3-<size>" optionally suffixed "-pt"/"-it".
_GEMMA3_NAME_RE = re.compile(r"^gemma-3-(270m|1b|4b|12b|27b)(-it|-pt)?$")


class TestModelSizesTable:
    def test_all_10_models_present(self):
        # Invariant assertions (NOT a byte-for-byte literal copy of the data):
        # exactly 10 keys, each name well-formed, each value a consistent
        # (size, variant) pair.
        assert len(GEMMASCOPE_2_MODEL_SIZES) == 10

        for name, value in GEMMASCOPE_2_MODEL_SIZES.items():
            m = _GEMMA3_NAME_RE.match(name)
            assert m is not None, f"malformed model name: {name!r}"

            size, variant = value
            assert variant in {"pt", "it"}, f"{name}: bad variant {variant!r}"

            # size must agree with an INDEPENDENT regex extraction from the key
            assert size == m.group(1), (
                f"{name}: size {size!r} disagrees with name-derived "
                f"{m.group(1)!r}"
            )

            # variant must agree with the name suffix, with the documented
            # 270m-PT exception (no suffix == pt).
            suffix = m.group(2)
            if suffix is None:
                assert variant == "pt", (
                    f"{name}: suffixless name must be the PT variant"
                )
            else:
                assert variant == suffix.lstrip("-"), (
                    f"{name}: variant {variant!r} disagrees with suffix "
                    f"{suffix!r}"
                )

        # All 5 sizes × both variants are represented.
        assert set(GEMMASCOPE_2_MODEL_SIZES.values()) == {
            (s, var) for s in _GEMMA3_NUM_HIDDEN_LAYERS for var in ("pt", "it")
        }

    def test_270m_pt_key_is_exactly_gemma_3_270m_no_pt_suffix(self):
        assert "gemma-3-270m" in GEMMASCOPE_2_MODEL_SIZES
        assert GEMMASCOPE_2_MODEL_SIZES["gemma-3-270m"] == ("270m", "pt")
        assert "gemma-3-270m-pt" not in GEMMASCOPE_2_MODEL_SIZES

    def test_270m_pt_has_no_pt_suffix_in_name(self):
        assert "gemma-3-270m" in GEMMASCOPE_2_MODEL_SIZES
        assert "gemma-3-270m-pt" not in GEMMASCOPE_2_MODEL_SIZES


class TestValidComboShape:
    @pytest.mark.parametrize("model", list({
        "gemma-3-270m", "gemma-3-270m-it",
        "gemma-3-1b-pt", "gemma-3-1b-it",
        "gemma-3-4b-pt", "gemma-3-4b-it",
        "gemma-3-12b-pt", "gemma-3-12b-it",
        "gemma-3-27b-pt", "gemma-3-27b-it",
    }))
    def test_every_model_has_a_combo_table(self, model):
        assert model in GEMMASCOPE_2_VALID_COMBOS
        assert len(GEMMASCOPE_2_VALID_COMBOS[model]) > 0

    def test_residual_has_medium_seed_1_only_at_262k(self):
        for model, combos in GEMMASCOPE_2_VALID_COMBOS.items():
            for (act, width), tier_layers in combos.items():
                if L0Tier.medium_seed_1 in tier_layers:
                    assert act == ActivationType.residual, (
                        f"medium_seed_1 on non-residual: {model}/{act}/{width}"
                    )
                    assert width == SAEWidth.W_262K, (
                        f"medium_seed_1 at non-262k: {model}/{act}/{width}"
                    )

    def test_selected_layers_per_size(self):
        # Structural / oracle assertions (NOT a mirror of _SELECTED_LAYERS):
        #   * each size selects exactly 4 layers,
        #   * strictly increasing,
        #   * every selected layer < that model's HF num_hidden_layers,
        # where num_hidden_layers comes from an INDEPENDENT in-test oracle.
        for model, combos in GEMMASCOPE_2_VALID_COMBOS.items():
            size = GEMMASCOPE_2_MODEL_SIZES[model][0]
            n_layers = _GEMMA3_NUM_HIDDEN_LAYERS[size]

            # Union of all layers used across every (act,width,tier) cell.
            all_layers: set[int] = set()
            for tier_layers in combos.values():
                for layers in tier_layers.values():
                    all_layers |= set(layers)

            ordered = sorted(all_layers)
            assert len(ordered) == 4, (
                f"{model}: expected exactly 4 selected layers, got {ordered}"
            )
            # strictly increasing
            assert all(a < b for a, b in itertools.pairwise(ordered)), ordered
            # every selected layer is a valid layer index for this model
            assert ordered[-1] < n_layers, (
                f"{model}: selected layer {ordered[-1]} >= "
                f"num_hidden_layers {n_layers}"
            )
            assert ordered[0] >= 0

            # Every individual cell uses the SAME 4 layers (base-release shape).
            for (act, width), tier_layers in combos.items():
                for tier, layers in tier_layers.items():
                    assert set(layers) == all_layers, (
                        f"{model}/{act}/{width}/{tier}: layer set "
                        f"{sorted(layers)} != size set {ordered}"
                    )


class TestGemma29bMatrix:
    def test_9b_res_16k_and_131k_cover_all_42_layers(self):
        # SAELens 6.42: only 16k & 131k ship all 42 residual layers.
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.residual, SAEWidth.W_16K)
        ] == frozenset(range(42))
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.residual, SAEWidth.W_131K)
        ] == frozenset(range(42))

    @pytest.mark.parametrize(
        "width", [SAEWidth.W_32K, SAEWidth.W_65K, SAEWidth.W_262K, SAEWidth.W_524K]
    )
    def test_9b_res_32k_65k_262k_524k_are_layer_20_only(self, width):
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.residual, width)
        ] == frozenset({20})

    def test_9b_res_1m_is_layers_9_20_31_only(self):
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.residual, SAEWidth.W_1M)
        ] == frozenset({9, 20, 31})

    def test_9b_mlp_16k_and_131k_cover_all_42_and_no_other_widths(self):
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.mlp, SAEWidth.W_16K)
        ] == frozenset(range(42))
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.mlp, SAEWidth.W_131K)
        ] == frozenset(range(42))
        mlp_widths = {w for (act, w) in GEMMA2_9B_VALID_COMBOS if act == ActivationType.mlp}
        assert mlp_widths == {SAEWidth.W_16K, SAEWidth.W_131K}

    def test_9b_att_16k_and_131k_cover_all_42_and_no_other_widths(self):
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.attention, SAEWidth.W_16K)
        ] == frozenset(range(42))
        assert GEMMA2_9B_VALID_COMBOS[
            (ActivationType.attention, SAEWidth.W_131K)
        ] == frozenset(range(42))
        att_widths = {
            w for (act, w) in GEMMA2_9B_VALID_COMBOS if act == ActivationType.attention
        }
        assert att_widths == {SAEWidth.W_16K, SAEWidth.W_131K}

    def test_9b_canonical_widths_exclude_262k_for_res(self):
        # The canonical v1 release for 9b residual omits 262k (verified
        # against the SAELens registry as of 6.42.0).
        assert SAEWidth.W_262K not in GEMMA2_9B_CANONICAL_WIDTHS[ActivationType.residual]
        assert SAEWidth.W_16K in GEMMA2_9B_CANONICAL_WIDTHS[ActivationType.residual]
        assert SAEWidth.W_1M in GEMMA2_9B_CANONICAL_WIDTHS[ActivationType.residual]


class TestGemma29bResolveValidation:
    """Behavioral tests through resolve_sae: the corrected matrix must reject
    the previously-wrongly-accepted v1 combos that 404 at SAELens fetch.
    """

    def test_res_262k_layer_5_now_rejected(self):
        with pytest.raises(ValueError):
            resolve_sae(
                "gemma-2-9b", SAERelease.gemmascope, ActivationType.residual,
                5, SAEWidth.W_262K, l0=176,
            )

    def test_res_262k_layer_20_ok(self):
        resolve_sae(
            "gemma-2-9b", SAERelease.gemmascope, ActivationType.residual,
            20, SAEWidth.W_262K, l0=176,
        )

    def test_res_1m_layer_9_ok(self):
        resolve_sae(
            "gemma-2-9b", SAERelease.gemmascope, ActivationType.residual,
            9, SAEWidth.W_1M, l0=176,
        )

    def test_res_1m_layer_10_rejected(self):
        with pytest.raises(ValueError):
            resolve_sae(
                "gemma-2-9b", SAERelease.gemmascope, ActivationType.residual,
                10, SAEWidth.W_1M, l0=176,
            )

    def test_res_16k_layer_0_canonical_ok(self):
        resolve_sae(
            "gemma-2-9b", SAERelease.gemmascope, ActivationType.residual,
            0, SAEWidth.W_16K, l0=None,
        )

    def test_mlp_32k_rejected(self):
        with pytest.raises(ValueError):
            resolve_sae(
                "gemma-2-9b", SAERelease.gemmascope, ActivationType.mlp,
                0, SAEWidth.W_32K, l0=176,
            )


class TestGemma227bMatrix:
    def test_27b_res_layers_are_10_22_34_only(self):
        assert GEMMA2_27B_VALID_COMBOS[(ActivationType.residual, SAEWidth.W_131K)] == frozenset({10, 22, 34})

    def test_27b_has_only_131k_residual_combo(self):
        assert set(GEMMA2_27B_VALID_COMBOS.keys()) == {
            (ActivationType.residual, SAEWidth.W_131K),
        }


def _registry_layer_sets_by_width(release: str) -> dict[str, set[int]]:
    """Parse `layer_<L>/width_<W>/...` keys from a SAELens release's saes_map.

    Returns {width_token: {layers...}}. `embedding/...` keys (no numeric
    layer) are intentionally skipped — our matrices are layer-indexed only.
    """
    directory = get_pretrained_saes_directory()
    rec = directory[release]
    pat = re.compile(r"^layer_(\d+)/width_(\w+?)/")
    out: dict[str, set[int]] = {}
    for key in rec.saes_map:
        m = pat.match(key)
        if m is None:
            continue
        out.setdefault(m.group(2), set()).add(int(m.group(1)))
    return out


@pytest.mark.very_slow
class TestV1MatrixRegistryConformance:
    """NETWORK-gated: probe the REAL SAELens registry and assert our
    hardcoded v1 9b/27b backfill matrices agree with it.

    This is the test that catches a *wrong* hardcoded backfill — the fast
    literal tests above are cheap golden pins but cannot see the registry.
    Skipped by default (`-m 'not very_slow'`).
    """

    def test_9b_residual_matrix_matches_registry(self):
        by_width = _registry_layer_sets_by_width("gemma-scope-9b-pt-res")
        for (act, width), layers in GEMMA2_9B_VALID_COMBOS.items():
            if act != ActivationType.residual:
                continue
            reg = by_width.get(width.value)
            assert reg is not None, (
                f"9b residual width {width.value} absent from registry"
            )
            assert set(layers) == reg, (
                f"9b residual/{width.value}: hardcoded {sorted(layers)} != "
                f"registry {sorted(reg)}"
            )

    def test_9b_mlp_matrix_matches_registry(self):
        by_width = _registry_layer_sets_by_width("gemma-scope-9b-pt-mlp")
        for (act, width), layers in GEMMA2_9B_VALID_COMBOS.items():
            if act != ActivationType.mlp:
                continue
            reg = by_width.get(width.value)
            assert reg is not None and set(layers) == reg, (
                f"9b mlp/{width.value}: hardcoded {sorted(layers)} != "
                f"registry {sorted(reg) if reg else None}"
            )

    def test_9b_attention_matrix_matches_registry(self):
        by_width = _registry_layer_sets_by_width("gemma-scope-9b-pt-att")
        for (act, width), layers in GEMMA2_9B_VALID_COMBOS.items():
            if act != ActivationType.attention:
                continue
            reg = by_width.get(width.value)
            assert reg is not None and set(layers) == reg, (
                f"9b att/{width.value}: hardcoded {sorted(layers)} != "
                f"registry {sorted(reg) if reg else None}"
            )

    def test_9b_residual_widths_match_registry(self):
        by_width = _registry_layer_sets_by_width("gemma-scope-9b-pt-res")
        hardcoded = {
            w.value
            for (a, w) in GEMMA2_9B_VALID_COMBOS
            if a == ActivationType.residual
        }
        assert hardcoded == set(by_width), (
            f"9b residual widths drift: hardcoded {sorted(hardcoded)} != "
            f"registry {sorted(by_width)}"
        )

    def test_27b_residual_matrix_matches_registry(self):
        by_width = _registry_layer_sets_by_width("gemma-scope-27b-pt-res")
        for (act, width), layers in GEMMA2_27B_VALID_COMBOS.items():
            assert act == ActivationType.residual
            reg = by_width.get(width.value)
            assert reg is not None and set(layers) == reg, (
                f"27b res/{width.value}: hardcoded {sorted(layers)} != "
                f"registry {sorted(reg) if reg else None}"
            )


class TestHfModelIdDriftGuard:
    """GEMMASCOPE_2_HF_MODEL_IDS and EXTRACTION_MODELS[*].hf_model_id
    independently map gemma-3 names → google/... ids. A silent drift is a
    SAE/activation repo mismatch with NO crash — worse than a crash.
    """

    def test_hf_ids_agree_for_every_shared_gemma3_key(self):
        shared = set(GEMMASCOPE_2_HF_MODEL_IDS) & set(EXTRACTION_MODELS)
        assert shared, "expected overlapping gemma-3 keys between the two maps"
        for key in sorted(shared):
            assert (
                GEMMASCOPE_2_HF_MODEL_IDS[key] == EXTRACTION_MODELS[key].hf_model_id
            ), (
                f"{key}: GEMMASCOPE_2_HF_MODEL_IDS={GEMMASCOPE_2_HF_MODEL_IDS[key]!r}"
                f" != EXTRACTION_MODELS[...].hf_model_id="
                f"{EXTRACTION_MODELS[key].hf_model_id!r}"
            )


class TestValidateGemmascope2:
    def test_valid_combo_returns_none(self):
        # gemma-3-4b-pt / residual / 16k / medium / layer 22 is valid.
        assert _validate_gemmascope_2(
            "gemma-3-4b-pt", ActivationType.residual, SAEWidth.W_16K, L0Tier.medium, 22,
        ) is None

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            _validate_gemmascope_2(
                "gemma-2-2b", ActivationType.residual, SAEWidth.W_16K, L0Tier.medium, 12,
            )

    def test_invalid_layer_lists_valid_alternatives(self):
        # gemma-3-4b-pt residual layers are {9, 17, 22, 29}. Layer 10 invalid.
        with pytest.raises(ValueError, match=r"Layer 10.*Valid layers"):
            _validate_gemmascope_2(
                "gemma-3-4b-pt", ActivationType.residual, SAEWidth.W_16K, L0Tier.medium, 10,
            )

    def test_invalid_width_lists_valid_widths(self):
        # Width 32k is not published for v2 residual.
        with pytest.raises(ValueError, match=r"Width.*32k.*Valid widths"):
            _validate_gemmascope_2(
                "gemma-3-4b-pt", ActivationType.residual, SAEWidth.W_32K, L0Tier.medium, 22,
            )

    def test_medium_seed_1_outside_residual_262k_raises(self):
        # medium_seed_1 is only published for residual at width 262k.
        with pytest.raises(ValueError, match="medium_seed_1"):
            _validate_gemmascope_2(
                "gemma-3-4b-pt", ActivationType.mlp, SAEWidth.W_16K, L0Tier.medium_seed_1, 22,
            )

    def test_27b_pt_has_no_mlp_in_registry_but_in_table(self):
        # 27b-pt-mlp is one of the injected releases; the table includes it.
        assert _validate_gemmascope_2(
            "gemma-3-27b-pt", ActivationType.mlp, SAEWidth.W_16K, L0Tier.medium, 16,
        ) is None


class TestResolveGemmascope2:
    def test_release_format_pt(self):
        resolved = _resolve_gemmascope_2(
            "gemma-3-4b-pt", ActivationType.residual, 22, SAEWidth.W_16K, L0Tier.medium,
        )
        assert resolved.release == "gemma-scope-2-4b-pt-res"
        assert resolved.sae_id == "layer_22_width_16k_l0_medium"
        assert resolved.hook_name == "blocks.22.hook_resid_post"

    def test_release_format_it(self):
        resolved = _resolve_gemmascope_2(
            "gemma-3-4b-it", ActivationType.mlp, 17, SAEWidth.W_262K, L0Tier.big,
        )
        assert resolved.release == "gemma-scope-2-4b-it-mlp"
        assert resolved.sae_id == "layer_17_width_262k_l0_big"
        assert resolved.hook_name == "blocks.17.hook_mlp_out"

    def test_270m_pt_release_uses_pt_suffix_despite_no_model_suffix(self):
        resolved = _resolve_gemmascope_2(
            "gemma-3-270m", ActivationType.residual, 12, SAEWidth.W_16K, L0Tier.small,
        )
        # Model name has no suffix; release name still uses 'pt'.
        assert resolved.release == "gemma-scope-2-270m-pt-res"

    def test_attention_uses_attn_hook_z(self):
        # gemma-scope-2 attention SAEs declare hook_name="blocks.{L}.attn.hook_z"
        # in their own cfg.metadata.hook_name (verified at gemma-3-1b-pt layers
        # 7 and 13, width 16k). d_in=1024 (=num_attention_heads*head_dim), the
        # pre-`o_proj` per-head output — identical to v1 gemmascope.
        # The previous assertion (hook_attn_out) encoded the bug.
        resolved = _resolve_gemmascope_2(
            "gemma-3-1b-pt", ActivationType.attention, 13, SAEWidth.W_16K, L0Tier.medium,
        )
        assert resolved.hook_name == "blocks.13.attn.hook_z"
        # v2 attention is pre-projection, same as v1 — NOT hook_attn_out.
        assert "hook_attn_out" not in resolved.hook_name

    def test_invalid_combo_propagates_validator_error(self):
        with pytest.raises(ValueError, match=r"Layer 10"):
            _resolve_gemmascope_2(
                "gemma-3-4b-pt", ActivationType.residual, 10, SAEWidth.W_16K, L0Tier.medium,
            )


class TestResolveSaeDispatchV2:
    def test_v2_release_routes_through_v2_resolver(self):
        resolved = resolve_sae(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=SAEWidth.W_16K,
            l0=L0Tier.medium,
        )
        assert resolved.release == "gemma-scope-2-4b-it-res"

    def test_v2_release_with_none_l0_raises(self):
        # v2 has no canonical release; l0 is required.
        with pytest.raises(ValueError, match=r"gemma-scope-2.*requires"):
            resolve_sae(
                model="gemma-3-4b-it",
                release=SAERelease.gemmascope_2,
                activation_type=ActivationType.residual,
                layer=22,
                width=SAEWidth.W_16K,
                l0=None,
            )

    def test_v2_release_with_int_l0_raises(self):
        with pytest.raises(ValueError, match=r"gemma-scope-2 requires an L0Tier"):
            resolve_sae(
                model="gemma-3-4b-it",
                release=SAERelease.gemmascope_2,
                activation_type=ActivationType.residual,
                layer=22,
                width=SAEWidth.W_16K,
                l0=176,
            )


def _enumerate_all_v2_combos():
    """Yield (model, activation, layer, width, tier) for every cell in the v2 matrix."""
    for model, combos in GEMMASCOPE_2_VALID_COMBOS.items():
        for (activation, width), tier_layers in combos.items():
            for tier, layers in tier_layers.items():
                for layer in sorted(layers):
                    yield model, activation, layer, width, tier


@pytest.mark.parametrize(
    "model,activation,layer,width,tier", list(_enumerate_all_v2_combos())
)
def test_every_cell_resolves(model, activation, layer, width, tier):
    resolved = resolve_sae(
        model=model,
        release=SAERelease.gemmascope_2,
        activation_type=activation,
        layer=layer,
        width=width,
        l0=tier,
    )
    # release must reference the right size and variant
    size, variant = GEMMASCOPE_2_MODEL_SIZES[model]
    assert f"gemma-scope-2-{size}-{variant}-" in resolved.release
    # sae_id is the v2 layer_*_width_*_l0_* format
    assert resolved.sae_id == f"layer_{layer}_width_{width.value}_l0_{tier.value}"
    # hook_name uses the activation-specific template
    assert f".{layer}." in resolved.hook_name


class TestModelFamilyGuards:
    def test_gemma3_model_with_v1_release_raises(self):
        with pytest.raises(ValueError, match=r"Gemma 3.*release=gemmascope_2"):
            resolve_sae(
                model="gemma-3-4b-pt",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=22,
                width=SAEWidth.W_16K,
                l0=None,
            )

    def test_gemma2_model_with_v2_release_raises(self):
        with pytest.raises(ValueError, match=r"Gemma 2.*release=gemmascope"):
            resolve_sae(
                model="gemma-2-2b",
                release=SAERelease.gemmascope_2,
                activation_type=ActivationType.residual,
                layer=12,
                width=SAEWidth.W_16K,
                l0=L0Tier.medium,
            )

    def test_270m_pt_typo_suggests_correct_name(self):
        with pytest.raises(ValueError, match=r"Did you mean 'gemma-3-270m'"):
            resolve_sae(
                model="gemma-3-270m-pt",  # typo
                release=SAERelease.gemmascope_2,
                activation_type=ActivationType.residual,
                layer=12,
                width=SAEWidth.W_16K,
                l0=L0Tier.medium,
            )
