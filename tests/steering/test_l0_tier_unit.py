"""Unit tests for the L0Tier enum, the SAERelease.gemmascope_2 member, and the int | L0Tier | None union."""

import pytest

from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
    l0_tier_for_resolved,
    resolve_sae,
)


class TestL0TierEnum:
    def test_members_exact(self):
        # Must match HF folder names verbatim, in canonical order.
        assert [t.value for t in L0Tier] == ["small", "medium", "medium_seed_1", "big"]

    def test_str_enum_values_serialise_to_value_string(self):
        # L0Tier is a StrEnum so str(L0Tier.medium) == "medium".
        assert str(L0Tier.medium) == "medium"
        assert str(L0Tier.medium_seed_1) == "medium_seed_1"
        assert str(L0Tier.big) == "big"
        assert str(L0Tier.small) == "small"

    def test_can_construct_from_string(self):
        assert L0Tier("medium") is L0Tier.medium
        assert L0Tier("medium_seed_1") is L0Tier.medium_seed_1

    def test_unknown_string_raises_value_error(self):
        with pytest.raises(ValueError):
            L0Tier("medium_seed_2")
        with pytest.raises(ValueError):
            L0Tier("MEDIUM")


class TestSAEReleaseEnum:
    def test_gemmascope_2_member_exists(self):
        assert SAERelease.gemmascope_2.value == "gemmascope_2"

    def test_v1_member_unchanged(self):
        assert SAERelease.gemmascope.value == "gemmascope"

    def test_no_overlap(self):
        values = [r.value for r in SAERelease]
        assert len(values) == len(set(values))


class TestL0Union:
    def test_v1_accepts_int_l0(self):
        resolved = resolve_sae(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=12,
            width=SAEWidth.W_16K,
            l0=105,
        )
        assert "average_l0_105" in resolved.sae_id

    def test_v1_accepts_none_l0(self):
        resolved = resolve_sae(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=12,
            width=SAEWidth.W_16K,
            l0=None,
        )
        assert "canonical" in resolved.sae_id

    def test_v1_rejects_l0_tier(self):
        with pytest.raises(ValueError, match="GemmaScope v1.*L0Tier"):
            resolve_sae(
                model="gemma-2-2b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=12,
                width=SAEWidth.W_16K,
                l0=L0Tier.medium,
            )


class TestL0TierForResolved:
    """``l0_tier_for_resolved`` is the single source of truth for the run's
    tier. It parses the tier back out of the *actually resolved* sae_id so
    it works identically for the structured-v1, structured-v2, and legacy
    (no ResolvedSAE) extraction paths.
    """

    @pytest.mark.parametrize(
        "tier",
        [L0Tier.small, L0Tier.medium, L0Tier.medium_seed_1, L0Tier.big],
    )
    def test_v2_every_tier_parsed_from_resolved_sae_id(self, tier):
        # medium_seed_1 is only published at residual/262k; the other tiers
        # are valid at residual/16k. Pick a width that publishes the tier so
        # resolve_sae succeeds and we exercise the real resolved sae_id.
        width = (
            SAEWidth.W_262K if tier is L0Tier.medium_seed_1 else SAEWidth.W_16K
        )
        resolved = resolve_sae(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=width,
            l0=tier,
        )
        assert l0_tier_for_resolved(resolved) is tier

    def test_v1_int_l0_returns_none(self):
        resolved = resolve_sae(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=12,
            width=SAEWidth.W_16K,
            l0=176,
        )
        assert l0_tier_for_resolved(resolved) is None

    def test_v1_canonical_none_l0_returns_none(self):
        resolved = resolve_sae(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=12,
            width=SAEWidth.W_16K,
            l0=None,
        )
        assert l0_tier_for_resolved(resolved) is None

    def test_non_v2_shaped_sae_id_returns_none(self):
        # Defensive: a hand-built ResolvedSAE whose sae_id is not the v2
        # "layer_{L}_width_{W}_l0_{tier}" grammar must not be misparsed.
        assert (
            l0_tier_for_resolved(
                ResolvedSAE(
                    release="x", sae_id="layer_8/width_16k/canonical",
                    hook_name="blocks.8.hook_resid_post",
                )
            )
            is None
        )

    def test_v2_shaped_sae_id_with_unknown_tier_returns_none(self):
        # A v2-grammar sae_id carrying a token that is not a known L0Tier
        # degrades to None rather than raising.
        assert (
            l0_tier_for_resolved(
                ResolvedSAE(
                    release="x", sae_id="layer_8_width_16k_l0_gigantic",
                    hook_name="blocks.8.hook_resid_post",
                )
            )
            is None
        )
