"""Validation matrix data for gemma-scope-2 (v2) and v1 9b/27b backfill.

This module is pure data with no behavior. It exists separately from
:mod:`steering.sae_resolver` to keep the resolver readable and to let
:mod:`steering.sae_resolver_v2_registry` import the same source of truth
without a circular import via the resolver's runtime dispatch.

Source of truth: enumerated via the SAELens 6.42 registry + HuggingFace
file listings for the (unregistered-but-on-HF) PT releases. See the design
spec (docs/superpowers/specs/2026-05-15-gemma-scope-2-design.md, §2.4).
"""

from __future__ import annotations

from steering.sae_resolver import ActivationType, L0Tier, SAEWidth

# ── Model name → (size_str, release_variant_str) ────────────────────────────
# release_variant is the token that appears in the SAELens release name
# (e.g. "gemma-scope-2-{size}-{variant}-{activation}"). For Gemma-3 270m the
# *model* name has no suffix at all when pretrained, but the release name
# still uses "pt" — this table reconciles them.
GEMMASCOPE_2_MODEL_SIZES: dict[str, tuple[str, str]] = {
    "gemma-3-270m":      ("270m", "pt"),
    "gemma-3-270m-it":   ("270m", "it"),
    "gemma-3-1b-pt":     ("1b",   "pt"),
    "gemma-3-1b-it":     ("1b",   "it"),
    "gemma-3-4b-pt":     ("4b",   "pt"),
    "gemma-3-4b-it":     ("4b",   "it"),
    "gemma-3-12b-pt":    ("12b",  "pt"),
    "gemma-3-12b-it":    ("12b",  "it"),
    "gemma-3-27b-pt":    ("27b",  "pt"),
    "gemma-3-27b-it":    ("27b",  "it"),
}

# Model name → HF id used for tokenizer / model weight loading.
GEMMASCOPE_2_HF_MODEL_IDS: dict[str, str] = {
    "gemma-3-270m":      "google/gemma-3-270m",
    "gemma-3-270m-it":   "google/gemma-3-270m-it",
    "gemma-3-1b-pt":     "google/gemma-3-1b-pt",
    "gemma-3-1b-it":     "google/gemma-3-1b-it",
    "gemma-3-4b-pt":     "google/gemma-3-4b-pt",
    "gemma-3-4b-it":     "google/gemma-3-4b-it",
    "gemma-3-12b-pt":    "google/gemma-3-12b-pt",
    "gemma-3-12b-it":    "google/gemma-3-12b-it",
    "gemma-3-27b-pt":    "google/gemma-3-27b-pt",
    "gemma-3-27b-it":    "google/gemma-3-27b-it",
}

# Selected layers shipped in every base v2 release for a given size.
_SELECTED_LAYERS: dict[str, frozenset[int]] = {
    "270m": frozenset({5, 9, 12, 15}),
    "1b":   frozenset({7, 13, 17, 22}),
    "4b":   frozenset({9, 17, 22, 29}),
    "12b":  frozenset({12, 24, 31, 41}),
    "27b":  frozenset({16, 31, 40, 53}),
}


def _res_combos(size: str) -> dict[
    tuple[ActivationType, SAEWidth], dict[L0Tier, frozenset[int]]
]:
    """Residual SAEs: 4 widths × {small, medium, big}, + medium_seed_1 at 262k."""
    layers = _SELECTED_LAYERS[size]
    out: dict[tuple[ActivationType, SAEWidth], dict[L0Tier, frozenset[int]]] = {}
    for width in (SAEWidth.W_16K, SAEWidth.W_65K, SAEWidth.W_262K, SAEWidth.W_1M):
        tiers: dict[L0Tier, frozenset[int]] = {
            L0Tier.small: layers,
            L0Tier.medium: layers,
            L0Tier.big: layers,
        }
        if width == SAEWidth.W_262K:
            tiers[L0Tier.medium_seed_1] = layers
        out[(ActivationType.residual, width)] = tiers
    return out


def _mlp_combos(size: str) -> dict[
    tuple[ActivationType, SAEWidth], dict[L0Tier, frozenset[int]]
]:
    """MLP SAEs: widths {16k, 65k, 262k} × {small, medium, big}.

    1b is the one exception: 1b-{pt,it}-mlp also publishes width 1m.
    """
    layers = _SELECTED_LAYERS[size]
    widths = [SAEWidth.W_16K, SAEWidth.W_65K, SAEWidth.W_262K]
    if size == "1b":
        widths.append(SAEWidth.W_1M)
    return {
        (ActivationType.mlp, w): {
            L0Tier.small: layers,
            L0Tier.medium: layers,
            L0Tier.big: layers,
        }
        for w in widths
    }


def _att_combos(size: str) -> dict[
    tuple[ActivationType, SAEWidth], dict[L0Tier, frozenset[int]]
]:
    """Attention SAEs: widths {16k, 65k, 262k} × {small, medium, big}."""
    layers = _SELECTED_LAYERS[size]
    return {
        (ActivationType.attention, w): {
            L0Tier.small: layers,
            L0Tier.medium: layers,
            L0Tier.big: layers,
        }
        for w in (SAEWidth.W_16K, SAEWidth.W_65K, SAEWidth.W_262K)
    }


def _full_combos(size: str) -> dict[
    tuple[ActivationType, SAEWidth], dict[L0Tier, frozenset[int]]
]:
    return {**_res_combos(size), **_mlp_combos(size), **_att_combos(size)}


GEMMASCOPE_2_VALID_COMBOS: dict[
    str, dict[tuple[ActivationType, SAEWidth], dict[L0Tier, frozenset[int]]]
] = {
    model: _full_combos(size)
    for model, (size, _variant) in GEMMASCOPE_2_MODEL_SIZES.items()
}


# Folder name on HF for each activation type.
GEMMASCOPE_2_HF_FOLDER: dict[ActivationType, str] = {
    ActivationType.residual:  "resid_post",
    ActivationType.mlp:       "mlp_out",
    ActivationType.attention: "attn_out",
}


# Releases that are present on HF but missing from SAELens 6.42's registry.
# Each tuple is (release_name, model_name, hf_repo_id). The injection module
# (steering/sae_resolver_v2_registry.py) constructs PretrainedSAELookup
# records from the matching combo table at import time.
INJECTED_RELEASES: tuple[tuple[str, str, str], ...] = (
    ("gemma-scope-2-1b-pt-res",  "gemma-3-1b-pt",  "google/gemma-scope-2-1b-pt"),
    ("gemma-scope-2-1b-pt-mlp",  "gemma-3-1b-pt",  "google/gemma-scope-2-1b-pt"),
    ("gemma-scope-2-1b-pt-att",  "gemma-3-1b-pt",  "google/gemma-scope-2-1b-pt"),
    ("gemma-scope-2-12b-pt-res", "gemma-3-12b-pt", "google/gemma-scope-2-12b-pt"),
    ("gemma-scope-2-12b-pt-mlp", "gemma-3-12b-pt", "google/gemma-scope-2-12b-pt"),
    ("gemma-scope-2-12b-pt-att", "gemma-3-12b-pt", "google/gemma-scope-2-12b-pt"),
    ("gemma-scope-2-27b-pt-mlp", "gemma-3-27b-pt", "google/gemma-scope-2-27b-pt"),
    ("gemma-scope-2-27b-pt-att", "gemma-3-27b-pt", "google/gemma-scope-2-27b-pt"),
)


# ── Backfill: v1 gemma-2-9b validation matrix ───────────────────────────────
# Source: enumerated from the live SAELens 6.42 registry on 2026-05-15
# (gemma-scope-9b-pt-{res,mlp,att}). The T05 backfill originally claimed all
# 7 residual widths cover all 42 layers; that was WRONG and let
# _validate_gemmascope silently accept combos (e.g. 9b residual 262k layer 5)
# that 404 at SAELens fetch time. Per the registry: only 16k & 131k ship all
# 42 layers; 32k/65k/262k/524k are layer-20-ONLY; 1m is {9,20,31} only.
# mlp & att each ship only 16k & 131k, both at all 42 layers.
_GEMMA2_9B_ALL_LAYERS: frozenset[int] = frozenset(range(42))
_GEMMA2_9B_L20: frozenset[int] = frozenset({20})
_GEMMA2_9B_RES_1M: frozenset[int] = frozenset({9, 20, 31})

GEMMA2_9B_VALID_COMBOS: dict[
    tuple[ActivationType, SAEWidth], frozenset[int]
] = {
    # Residual: only 16k & 131k cover all 42 layers; the wide dictionaries
    # (32k/65k/262k/524k) ship layer 20 only; 1m ships {9,20,31}.
    (ActivationType.residual, SAEWidth.W_16K):   _GEMMA2_9B_ALL_LAYERS,
    (ActivationType.residual, SAEWidth.W_131K):  _GEMMA2_9B_ALL_LAYERS,
    (ActivationType.residual, SAEWidth.W_32K):   _GEMMA2_9B_L20,
    (ActivationType.residual, SAEWidth.W_65K):   _GEMMA2_9B_L20,
    (ActivationType.residual, SAEWidth.W_262K):  _GEMMA2_9B_L20,
    (ActivationType.residual, SAEWidth.W_524K):  _GEMMA2_9B_L20,
    (ActivationType.residual, SAEWidth.W_1M):    _GEMMA2_9B_RES_1M,
    # MLP: 2 widths, all 42 layers
    (ActivationType.mlp, SAEWidth.W_16K):        _GEMMA2_9B_ALL_LAYERS,
    (ActivationType.mlp, SAEWidth.W_131K):       _GEMMA2_9B_ALL_LAYERS,
    # Attention: 2 widths, all 42 layers
    (ActivationType.attention, SAEWidth.W_16K):  _GEMMA2_9B_ALL_LAYERS,
    (ActivationType.attention, SAEWidth.W_131K): _GEMMA2_9B_ALL_LAYERS,
}

# Widths available with l0=None (canonical release). The non-canonical
# release (explicit average_l0) supports the full set listed above.
# Note: 9b-pt-res-canonical does NOT include width 262k — verified against
# the registry. We model this as a separate canonical-only set so the
# resolver can reject l0=None at 262k with a precise message.
GEMMA2_9B_CANONICAL_WIDTHS: dict[ActivationType, frozenset[SAEWidth]] = {
    ActivationType.residual: frozenset({
        SAEWidth.W_16K, SAEWidth.W_32K, SAEWidth.W_65K,
        SAEWidth.W_131K, SAEWidth.W_524K, SAEWidth.W_1M,
    }),
    ActivationType.mlp:       frozenset({SAEWidth.W_16K, SAEWidth.W_131K}),
    ActivationType.attention: frozenset({SAEWidth.W_16K, SAEWidth.W_131K}),
}


# ── Backfill: v1 gemma-2-27b validation matrix ──────────────────────────────
# v1 27b ships exactly three residual SAEs at width 131k; mlp/att are not
# published. Source: SAELens 6.42 registry.
GEMMA2_27B_VALID_COMBOS: dict[
    tuple[ActivationType, SAEWidth], frozenset[int]
] = {
    (ActivationType.residual, SAEWidth.W_131K): frozenset({10, 22, 34}),
}
