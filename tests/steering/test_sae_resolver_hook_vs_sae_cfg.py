"""Tests that resolver hook templates match actual SAE cfg.hook_name.

Fast structural unit test (no network) and a slow real-SAE cross-check
(@pytest.mark.slow) that loads real SAEs from HuggingFace to verify
resolve_sae(...).hook_name matches each SAE's own cfg.hook_name.

The fast unit tests in TestActivationMapStructure would have caught
the gemma-scope-2 attention bug (hook_attn_out vs attn.hook_z) immediately,
at zero network cost — they run in default CI.

The slow cross-check functions (test_v2_resolver_hook_matches_sae_cfg /
test_v1_resolver_hook_matches_sae_cfg) download real SAEs and assert
resolve_sae(...).hook_name == sae.cfg.hook_name, for both v1 (gemma-2-2b /
gemmascope / L6 / W16k) and v2 (gemma-3-1b-pt / gemma-scope-2 / L7 + L13 /
W16k / medium).
"""

import pytest
from sae_lens import SAE

from steering.sae_resolver import (
    _GEMMASCOPE_2_ACTIVATION_MAP,
    _GEMMASCOPE_ACTIVATION_MAP,
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)

# ---------------------------------------------------------------------------
# Fast structural guard — no SAE downloads, runs in default CI
# ---------------------------------------------------------------------------


class TestActivationMapStructure:
    """Structural invariants on _GEMMASCOPE_ACTIVATION_MAP and
    _GEMMASCOPE_2_ACTIVATION_MAP that would have caught the hook_attn_out bug
    immediately in CI.
    """

    def test_v2_attention_template_equals_v1_attention_template(self):
        """gemma-scope-2 attention uses the same pre-projection attn.hook_z
        template as v1 — NOT the post-projection hook_attn_out.

        This assertion FAILS on the buggy code (v2 has hook_attn_out),
        and PASSES after the fix (both maps have attn.hook_z).
        """
        v1_suffix, v1_hook = _GEMMASCOPE_ACTIVATION_MAP[ActivationType.attention]
        v2_suffix, v2_hook = _GEMMASCOPE_2_ACTIVATION_MAP[ActivationType.attention]
        assert v2_hook == v1_hook, (
            f"v2 attention hook template {v2_hook!r} != v1 {v1_hook!r}; "
            f"gemma-scope-2 attention SAEs declare hook_name=attn.hook_z "
            f"(pre-projection, d_in=n_heads*head_dim), same as v1"
        )

    def test_v2_attention_template_is_attn_hook_z(self):
        """The v2 attention hook template must be the pre-projection attn.hook_z
        form — not hook_attn_out (which is post-projection / wrong d_in).

        Verified against gemma-3-1b-pt layers 7 and 13 SAE cfg.metadata.hook_name.
        """
        _, hook_template = _GEMMASCOPE_2_ACTIVATION_MAP[ActivationType.attention]
        assert "attn.hook_z" in hook_template, (
            f"v2 attention hook template {hook_template!r} does not contain "
            f"'attn.hook_z'; gemma-scope-2 attention SAEs declare "
            f"hook_name='blocks.{{L}}.attn.hook_z', d_in=1024 (=n_heads*head_dim)"
        )
        assert "hook_attn_out" not in hook_template, (
            f"v2 attention hook template must NOT be hook_attn_out (post-projection); "
            f"got {hook_template!r}"
        )

    def test_v1_attention_template_is_attn_hook_z(self):
        """v1 attention must be attn.hook_z (guard against v1 regression)."""
        _, hook_template = _GEMMASCOPE_ACTIVATION_MAP[ActivationType.attention]
        assert "attn.hook_z" in hook_template, (
            f"v1 attention hook template {hook_template!r} unexpectedly changed"
        )

    def test_v2_residual_template_matches_v1(self):
        """residual templates are identical in v1 and v2."""
        _, v1_hook = _GEMMASCOPE_ACTIVATION_MAP[ActivationType.residual]
        _, v2_hook = _GEMMASCOPE_2_ACTIVATION_MAP[ActivationType.residual]
        assert v1_hook == v2_hook, (
            f"residual hook mismatch: v1={v1_hook!r}, v2={v2_hook!r}"
        )

    def test_v2_mlp_template_matches_v1(self):
        """mlp templates are identical in v1 and v2."""
        _, v1_hook = _GEMMASCOPE_ACTIVATION_MAP[ActivationType.mlp]
        _, v2_hook = _GEMMASCOPE_2_ACTIVATION_MAP[ActivationType.mlp]
        assert v1_hook == v2_hook, (
            f"mlp hook mismatch: v1={v1_hook!r}, v2={v2_hook!r}"
        )

    def test_v2_attention_release_suffix_unchanged(self):
        """The 'att' release suffix must be unchanged (only the hook template was
        wrong, not the suffix — surgical fix guard).
        """
        v2_suffix, _ = _GEMMASCOPE_2_ACTIVATION_MAP[ActivationType.attention]
        assert v2_suffix == "att", (
            f"v2 attention release suffix changed unexpectedly: {v2_suffix!r}"
        )

    def test_resolved_hook_name_for_v2_attention_contains_attn_hook_z(self):
        """End-to-end: resolve_sae for a valid v2 attention combo returns a
        hook_name with attn.hook_z, not hook_attn_out.
        """
        resolved = resolve_sae(
            model="gemma-3-1b-pt",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.attention,
            layer=7,
            width=SAEWidth.W_16K,
            l0=L0Tier.medium,
        )
        assert "attn.hook_z" in resolved.hook_name, (
            f"resolve_sae returned {resolved.hook_name!r}; expected attn.hook_z"
        )
        assert "hook_attn_out" not in resolved.hook_name, (
            f"resolve_sae returned post-projection hook_attn_out: {resolved.hook_name!r}"
        )
        assert resolved.hook_name == "blocks.7.attn.hook_z"


# ---------------------------------------------------------------------------
# Slow real-SAE cross-check — loads real SAEs from HuggingFace
# ---------------------------------------------------------------------------


def _get_sae_hook_name(sae: SAE) -> str | None:
    """Return the SAE's canonical hook_name, mirroring validate_features.py.

    Probes cfg.hook_name first; if None, falls back to cfg.metadata.hook_name.
    This matches the exact probe order in validate_features.py lines 809-812.
    Returns None only if neither location has a hook_name (should not happen
    for well-formed SAELens releases).
    """
    hook = getattr(sae.cfg, "hook_name", None)
    if hook is None:
        metadata = getattr(sae.cfg, "metadata", None)
        hook = getattr(metadata, "hook_name", None)
    return hook


_V2_CASES: list[tuple[str, ActivationType, int, SAEWidth, L0Tier]] = [
    # (model, activation_type, layer, width, l0_tier)
    # gemma-3-1b-pt, layer 7, W16k, medium — for all 3 activation types
    ("gemma-3-1b-pt", ActivationType.residual,  7,  SAEWidth.W_16K, L0Tier.medium),
    ("gemma-3-1b-pt", ActivationType.mlp,       7,  SAEWidth.W_16K, L0Tier.medium),
    ("gemma-3-1b-pt", ActivationType.attention,  7,  SAEWidth.W_16K, L0Tier.medium),
    # gemma-3-1b-pt, layer 13, W16k, medium — second layer for attention focus
    ("gemma-3-1b-pt", ActivationType.residual,  13, SAEWidth.W_16K, L0Tier.medium),
    ("gemma-3-1b-pt", ActivationType.mlp,       13, SAEWidth.W_16K, L0Tier.medium),
    ("gemma-3-1b-pt", ActivationType.attention,  13, SAEWidth.W_16K, L0Tier.medium),
]

_V1_CASES: list[tuple[str, ActivationType, int, SAEWidth, int | None]] = [
    # (model, activation_type, layer, width, l0)
    # gemma-2-2b, layer 6, W16k, canonical (l0=None) — all 3 activation types
    ("gemma-2-2b", ActivationType.residual,  6, SAEWidth.W_16K, None),
    ("gemma-2-2b", ActivationType.mlp,       6, SAEWidth.W_16K, None),
    ("gemma-2-2b", ActivationType.attention,  6, SAEWidth.W_16K, None),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "model,activation_type,layer,width,l0_tier",
    _V2_CASES,
    ids=[
        f"{m}-{a.value}-L{layer}-{w.value}-{t.value}"
        for m, a, layer, w, t in _V2_CASES
    ],
)
def test_v2_resolver_hook_matches_sae_cfg(
    model: str,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0_tier: L0Tier,
) -> None:
    """resolve_sae(...).hook_name must exactly match the loaded SAE's own
    cfg.metadata.hook_name (or cfg.hook_name as fallback).

    This test loads real SAEs from HuggingFace and would have immediately
    caught the gemma-scope-2 attention hook bug (1152 != 1024 on encode).
    """
    resolved = resolve_sae(
        model=model,
        release=SAERelease.gemmascope_2,
        activation_type=activation_type,
        layer=layer,
        width=width,
        l0=l0_tier,
    )
    sae = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cpu",
    )
    sae_cfg_hook = _get_sae_hook_name(sae)
    assert resolved.hook_name == sae_cfg_hook, (
        f"Resolver returned hook_name={resolved.hook_name!r} but the real SAE "
        f"cfg declares hook_name={sae_cfg_hook!r}. "
        f"Combo: {model}/{activation_type.value}/L{layer}/{width.value}/{l0_tier.value}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model,activation_type,layer,width,l0",
    _V1_CASES,
    ids=[
        f"{m}-{a.value}-L{layer}-{w.value}-l0_{l0}"
        for m, a, layer, w, l0 in _V1_CASES
    ],
)
def test_v1_resolver_hook_matches_sae_cfg(
    model: str,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0: int | None,
) -> None:
    """Same cross-check for v1 (gemmascope / Gemma 2) SAEs.

    Ensures v1 hook templates also match the real SAE cfg, guarding against
    any future v1 regression.
    """
    resolved = resolve_sae(
        model=model,
        release=SAERelease.gemmascope,
        activation_type=activation_type,
        layer=layer,
        width=width,
        l0=l0,
    )
    sae = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cpu",
    )
    sae_cfg_hook = _get_sae_hook_name(sae)
    assert resolved.hook_name == sae_cfg_hook, (
        f"Resolver returned hook_name={resolved.hook_name!r} but the real SAE "
        f"cfg declares hook_name={sae_cfg_hook!r}. "
        f"Combo: {model}/{activation_type.value}/L{layer}/{width.value}/l0={l0}"
    )
