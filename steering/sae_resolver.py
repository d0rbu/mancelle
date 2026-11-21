"""SAE ID resolution for GemmaScope and other SAE releases.

Resolves structured config (model, activation_type, layer, width) into
concrete SAELens release + sae_id strings and TransformerLens hook names.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory


class ActivationType(Enum):
    residual = "residual"
    mlp = "mlp"
    attention = "attention"


class SAEWidth(Enum):
    W_16K = "16k"
    W_32K = "32k"
    W_65K = "65k"
    W_131K = "131k"
    W_262K = "262k"
    W_524K = "524k"
    W_1M = "1m"


class L0Tier(StrEnum):
    """L0 tier used by GemmaScope-2 SAEs.

    Replaces the integer average-L0 value used by v1. v2 publishes a
    discrete set of tiers; ``medium_seed_1`` is an extra seed variant
    available only at width 262k for residual SAEs (see the v2 validation
    table in :mod:`steering.sae_resolver_v2_data`).

    Order of declaration matches the canonical small→big ordering used
    when emitting closest-match suggestions.
    """

    small = "small"
    medium = "medium"
    medium_seed_1 = "medium_seed_1"
    big = "big"


SAE_WIDTH_TO_NUM_FEATURES: dict[SAEWidth, int] = {
    SAEWidth.W_16K: 16384,
    SAEWidth.W_32K: 32768,
    SAEWidth.W_65K: 65536,
    SAEWidth.W_131K: 131072,
    SAEWidth.W_262K: 262144,
    SAEWidth.W_524K: 524288,
    SAEWidth.W_1M: 1048576,
}


class SAERelease(Enum):
    gemmascope = "gemmascope"
    gemmascope_2 = "gemmascope_2"


def coerce_config_l0(
    release: SAERelease, l0_raw: object
) -> int | L0Tier | None:
    """Coerce an untyped config ``sae.l0`` to ``resolve_sae``'s contract.

    OmegaConf/YAML has no enum type, so a gemma-scope-2 tier arrives as a
    plain string. V1 GemmaScope instead accepts an integer average-L0 or
    ``None``. Resolution still owns the release-specific validity checks.
    """
    if release == SAERelease.gemmascope_2 and isinstance(l0_raw, str):
        return L0Tier(l0_raw)
    if l0_raw is None or isinstance(l0_raw, (int, L0Tier)):
        return l0_raw
    raise ValueError(
        f"sae.l0={l0_raw!r} is not valid for release {release.value!r}: "
        f"expected an int (v1 average-L0), null (v1 canonical), or a "
        f"gemma-scope-2 tier name (small/medium/medium_seed_1/big)."
    )


@dataclass(frozen=True)
class ResolvedSAE:
    """Concrete SAELens identifiers resolved from structured config."""

    release: str  # SAELens release param
    sae_id: str  # SAELens sae_id param
    hook_name: str  # TransformerLens hook name


# ── GemmaScope mapping tables ──────────────────────────────────────────────

_GEMMASCOPE_MODEL_SIZES: dict[str, str] = {
    "gemma-2-2b": "2b",
    "gemma-2-9b": "9b",
    "gemma-2-27b": "27b",
}

_GEMMASCOPE_ACTIVATION_MAP: dict[ActivationType, tuple[str, str]] = {
    # activation_type → (release_suffix, hook_name_template)
    # {L} is replaced with the layer number
    ActivationType.residual: ("res", "blocks.{L}.hook_resid_post"),
    ActivationType.mlp: ("mlp", "blocks.{L}.hook_mlp_out"),
    ActivationType.attention: ("att", "blocks.{L}.attn.hook_z"),
}

_GEMMASCOPE_2_ACTIVATION_MAP: dict[ActivationType, tuple[str, str]] = {
    # activation_type → (release_suffix, hook_name_template)
    # {L} is replaced with the layer number.
    # gemma-scope-2 attention SAEs declare hook_name="blocks.{L}.attn.hook_z"
    # in their own cfg.metadata.hook_name (verified at gemma-3-1b-pt layers 7
    # and 13, width 16k). d_in=1024 (= num_attention_heads * head_dim), the
    # pre-`o_proj` per-head attention output — identical to v1 gemmascope.
    ActivationType.residual:  ("res", "blocks.{L}.hook_resid_post"),
    ActivationType.mlp:       ("mlp", "blocks.{L}.hook_mlp_out"),
    ActivationType.attention: ("att", "blocks.{L}.attn.hook_z"),
}

# Valid (activation_type, width) → set of allowed layers for gemma-2-2b.
_ALL_GEMMA2_2B_LAYERS = set(range(26))

_GEMMA2_2B_VALID_COMBOS: dict[
    tuple[ActivationType, SAEWidth], set[int] | frozenset[int] | None
] = {
    # Attention: 16k and 65k, all layers
    (ActivationType.attention, SAEWidth.W_16K): _ALL_GEMMA2_2B_LAYERS,
    (ActivationType.attention, SAEWidth.W_65K): _ALL_GEMMA2_2B_LAYERS,
    # MLP: 16k and 65k, all layers
    (ActivationType.mlp, SAEWidth.W_16K): _ALL_GEMMA2_2B_LAYERS,
    (ActivationType.mlp, SAEWidth.W_65K): _ALL_GEMMA2_2B_LAYERS,
    # Residual 16k/65k: all layers
    (ActivationType.residual, SAEWidth.W_16K): _ALL_GEMMA2_2B_LAYERS,
    (ActivationType.residual, SAEWidth.W_65K): _ALL_GEMMA2_2B_LAYERS,
    # Residual 32k/131k/262k/524k: layer 12 only
    (ActivationType.residual, SAEWidth.W_32K): {12},
    (ActivationType.residual, SAEWidth.W_131K): {12},
    (ActivationType.residual, SAEWidth.W_262K): {12},
    (ActivationType.residual, SAEWidth.W_524K): {12},
    # Residual 1m: layer 12 only
    (ActivationType.residual, SAEWidth.W_1M): {12},
}

# ── GemmaScope v2 + v1 backfill validation data ────────────────────────────
# Imported here (after the enum definitions above) because
# :mod:`steering.sae_resolver_v2_data` consumes ``ActivationType``,
# ``SAEWidth`` and ``L0Tier`` from this module to populate its tables, so
# the import has to be placed below those definitions to break the cycle.
# This is still a module-level import — no function-local imports.
# Keep the v2 size/combination tables and the v1 9b/27b backfill tables
# together here for ``_validate_gemmascope``.
from steering.sae_resolver_v2_data import (  # noqa: E402  (placed after enums by design)
    GEMMA2_9B_CANONICAL_WIDTHS,
    GEMMA2_9B_VALID_COMBOS,
    GEMMA2_27B_VALID_COMBOS,
    GEMMASCOPE_2_MODEL_SIZES,
    GEMMASCOPE_2_VALID_COMBOS,
)

# Model → validation table. ``Mapping`` (covariant in its value type) keeps
# this assignable from the narrower ``dict[..., frozenset[int]]`` v1-backfill
# tables in :mod:`steering.sae_resolver_v2_data` without forcing those
# upstream literals to use a wider value type.
_VALID_COMBOS: Mapping[
    str, Mapping[tuple[ActivationType, SAEWidth], set[int] | frozenset[int] | None]
] = {
    "gemma-2-2b": _GEMMA2_2B_VALID_COMBOS,
    "gemma-2-9b": GEMMA2_9B_VALID_COMBOS,
    "gemma-2-27b": GEMMA2_27B_VALID_COMBOS,
}


def _validate_gemmascope(
    model: str,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0: int | None = None,
    *,
    skip_canonical_gate: bool = False,
) -> None:
    """Validate a v1 GemmaScope selection against the published matrix.

    Args:
        skip_canonical_gate: When True, the 9b canonical-only width gate
            (which fires when ``l0 is None``) is skipped. Used by the
            Neuronpedia URL-building path, whose source strings carry no
            l0 concept so the canonical-vs-noncanonical distinction is
            meaningless there — layer/width/activation validation still runs.
    """
    if model not in _VALID_COMBOS:
        raise ValueError(
            f"No validation table for v1 model {model!r}. "
            f"Supported: {sorted(_VALID_COMBOS)}."
        )

    combos = _VALID_COMBOS[model]
    key = (activation_type, width)
    if key not in combos:
        valid_widths = sorted(
            {w.value for (a, w) in combos if a == activation_type},
            key=lambda v: list(SAEWidth).index(SAEWidth(v)),
        )
        if not valid_widths:
            raise ValueError(
                f"Activation {activation_type.value} is not published for "
                f"{model} (gemmascope v1)."
            )
        raise ValueError(
            f"Width {width.value} is not available for {activation_type.value} "
            f"activations on {model}. Valid widths: {valid_widths}."
        )

    allowed_layers = combos[key]
    if allowed_layers is not None and layer not in allowed_layers:
        raise ValueError(
            f"Layer {layer} is not available for {activation_type.value}/"
            f"{width.value} on {model}. Valid layers: {sorted(allowed_layers)}."
        )

    # Canonical-only width check for 9b residual.
    if (
        not skip_canonical_gate
        and l0 is None
        and model == "gemma-2-9b"
        and width not in GEMMA2_9B_CANONICAL_WIDTHS.get(activation_type, frozenset())
    ):
        canonical_widths = sorted(
            w.value for w in GEMMA2_9B_CANONICAL_WIDTHS.get(activation_type, frozenset())
        )
        raise ValueError(
            f"Width {width.value} is not available in the canonical "
            f"gemma-scope-9b-pt-{activation_type.value} release. Pass an explicit "
            f"average L0 (e.g. l0=176) or pick a canonical width: {canonical_widths}."
        )


def _validate_gemmascope_2(
    model: str,
    activation_type: ActivationType,
    width: SAEWidth,
    l0_tier: L0Tier,
    layer: int,
) -> None:
    """Hard-validate a v2 SAE selection against the published matrix.

    Raises:
        ValueError: with a precise message listing the closest valid
            alternatives whenever the (model, activation, width, tier, layer)
            tuple is not in :data:`GEMMASCOPE_2_VALID_COMBOS`.
    """
    if model not in GEMMASCOPE_2_VALID_COMBOS:
        raise ValueError(
            f"Unsupported model {model!r} for gemma-scope-2. "
            f"Supported: {sorted(GEMMASCOPE_2_VALID_COMBOS)}"
        )

    combos = GEMMASCOPE_2_VALID_COMBOS[model]
    key = (activation_type, width)
    if key not in combos:
        valid_widths = sorted(
            {w.value for (a, w) in combos if a == activation_type},
            key=lambda v: list(SAEWidth).index(SAEWidth(v)),
        )
        raise ValueError(
            f"Width {width.value} is not published for {activation_type.value} "
            f"activations on {model} (gemma-scope-2). Valid widths: {valid_widths}."
        )

    tier_layers = combos[key]
    if l0_tier not in tier_layers:
        valid_tiers = [t.value for t in L0Tier if t in tier_layers]
        raise ValueError(
            f"L0 tier {l0_tier.value} is not published for "
            f"{activation_type.value}/{width.value} on {model} (gemma-scope-2). "
            f"Valid tiers: {valid_tiers}."
            + (
                " (medium_seed_1 is only published for residual at width 262k.)"
                if l0_tier == L0Tier.medium_seed_1
                else ""
            )
        )

    allowed_layers = tier_layers[l0_tier]
    if layer not in allowed_layers:
        sorted_layers = sorted(allowed_layers)
        nearest = min(sorted_layers, key=lambda x: abs(x - layer))
        raise ValueError(
            f"Layer {layer} is not published for {activation_type.value}/"
            f"{width.value}/{l0_tier.value} on {model} (gemma-scope-2). "
            f"Valid layers: {sorted_layers}. Closest valid: {nearest}."
        )


def _resolve_gemmascope_2(
    model: str,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0_tier: L0Tier,
) -> ResolvedSAE:
    """Resolve a gemma-scope-2 (Gemma 3) SAE configuration."""
    if model not in GEMMASCOPE_2_MODEL_SIZES:
        raise ValueError(
            f"Unsupported model {model!r} for gemma-scope-2. "
            f"Supported: {sorted(GEMMASCOPE_2_MODEL_SIZES)}"
        )

    _validate_gemmascope_2(model, activation_type, width, l0_tier, layer)

    size, release_variant = GEMMASCOPE_2_MODEL_SIZES[model]
    suffix, hook_template = _GEMMASCOPE_2_ACTIVATION_MAP[activation_type]
    hook_name = hook_template.replace("{L}", str(layer))

    release = f"gemma-scope-2-{size}-{release_variant}-{suffix}"
    sae_id = f"layer_{layer}_width_{width.value}_l0_{l0_tier.value}"
    return ResolvedSAE(release=release, sae_id=sae_id, hook_name=hook_name)


def l0_tier_for_resolved(resolved: "ResolvedSAE") -> L0Tier | None:
    """Derive the L0 tier of a run from its resolved SAE config.

    This is the *single source of truth* for "what l0_tier should this
    extraction run stamp into its shards". It parses the tier back out of
    :attr:`ResolvedSAE.sae_id`, i.e. the exact identifier that
    :func:`resolve_sae` actually produced — not a re-derivation from raw
    config — so it cannot drift from what was loaded.

    Only gemma-scope-2 sae_ids carry a tier; v2's sae_id is exactly
    ``layer_{L}_width_{W}_l0_{tier}`` (see :func:`_resolve_gemmascope_2`).
    v1 sae_ids are slash-delimited (``layer_{L}/width_{W}/canonical`` or
    ``.../average_l0_{int}``) and have no tier, so they parse to ``None``.
    Args:
        resolved: The :class:`ResolvedSAE` from :func:`resolve_sae`.

    Returns:
        The run's :class:`L0Tier` for gemma-scope-2, else ``None``.
    """
    # v2 sae_id grammar: "layer_{L}_width_{W}_l0_{tier}". The tier token is
    # everything after the literal "_l0_"; v1 ids contain no "_l0_" run
    # (they use "/canonical" or "/average_l0_{int}" with slashes), so the
    # marker check below rejects them before any L0Tier coercion.
    marker = "_l0_"
    idx = resolved.sae_id.rfind(marker)
    if idx == -1 or "/" in resolved.sae_id:
        return None
    tier_token = resolved.sae_id[idx + len(marker):]
    try:
        return L0Tier(tier_token)
    except ValueError:
        return None


def resolve_sae(
    model: str,
    release: SAERelease,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0: int | L0Tier | None = None,
) -> ResolvedSAE:
    """Resolve structured SAE config into concrete SAELens identifiers.

    Args:
        model: TransformerLens model name (e.g. "gemma-2-2b").
        release: SAE release family.
        activation_type: Which activation site to capture.
        layer: Transformer layer index.
        width: SAE dictionary width.
        l0: Specific average L0 value, or None for canonical.

    Returns:
        ResolvedSAE with release, sae_id, and hook_name strings.

    Raises:
        ValueError: If the model is unsupported or the combo is invalid.
    """
    # Family/release crosscheck — fail fast on Gemma 2 ↔ v2 mismatches.
    if model.startswith("gemma-3"):
        if release == SAERelease.gemmascope:
            raise ValueError(
                f"Model {model!r} is Gemma 3; use release=gemmascope_2 (not "
                f"gemmascope)."
            )
        if model == "gemma-3-270m-pt":
            raise ValueError(
                "Unknown model 'gemma-3-270m-pt'. Did you mean 'gemma-3-270m'? "
                "(The 270m PT model has no -pt suffix on HuggingFace.)"
            )
    elif model.startswith("gemma-2") and release == SAERelease.gemmascope_2:
        raise ValueError(
            f"Model {model!r} is Gemma 2; use release=gemmascope (not "
            f"gemmascope_2)."
        )

    if release == SAERelease.gemmascope:
        if isinstance(l0, L0Tier):
            raise ValueError(
                f"GemmaScope v1 uses an int average L0 (or None for canonical), "
                f"not an L0Tier. Got: {l0!r}. Use release=SAERelease.gemmascope_2 "
                f"for tier-based SAEs."
            )
        return _resolve_gemmascope(model, activation_type, layer, width, l0)

    if release == SAERelease.gemmascope_2:
        if l0 is None:
            raise ValueError(
                "gemma-scope-2 requires an L0Tier (small/medium/medium_seed_1/big); "
                "this release family has no canonical variant."
            )
        if isinstance(l0, int):
            raise ValueError(
                f"gemma-scope-2 requires an L0Tier, not an int average L0. "
                f"Got: {l0!r}. Valid: small / medium / medium_seed_1 / big."
            )
        return _resolve_gemmascope_2(model, activation_type, layer, width, l0)

    raise ValueError(f"Unsupported SAE release: {release.value}")


def _resolve_gemmascope(
    model: str,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0: int | None,
) -> ResolvedSAE:
    """Resolve a GemmaScope SAE configuration."""
    if model not in _GEMMASCOPE_MODEL_SIZES:
        raise ValueError(
            f"Unsupported model '{model}' for GemmaScope. "
            f"Supported: {sorted(_GEMMASCOPE_MODEL_SIZES.keys())}"
        )

    _validate_gemmascope(model, activation_type, layer, width, l0)

    size = _GEMMASCOPE_MODEL_SIZES[model]
    suffix, hook_template = _GEMMASCOPE_ACTIVATION_MAP[activation_type]
    hook_name = hook_template.replace("{L}", str(layer))

    if l0 is None:
        # Canonical release
        release_str = f"gemma-scope-{size}-pt-{suffix}-canonical"
        sae_id = f"layer_{layer}/width_{width.value}/canonical"
    else:
        # Specific L0
        release_str = f"gemma-scope-{size}-pt-{suffix}"
        sae_id = f"layer_{layer}/width_{width.value}/average_l0_{l0}"

    return ResolvedSAE(release=release_str, sae_id=sae_id, hook_name=hook_name)


def resolve_neuronpedia_ids(
    model: str,
    release: SAERelease,
    activation_type: ActivationType,
    layer: int,
    width: SAEWidth,
    l0: int | L0Tier | None = None,
) -> tuple[str, str]:
    """Derive Neuronpedia ``(model_id, sae_source)`` from structured SAE config.

    For GemmaScope v1, the source string follows the Neuronpedia naming
    convention ``{layer}-{release}-{activation_abbrev}-{width}``. For
    GemmaScope-2, use the SAELens registry's published ``neuronpedia_id``;
    public medium-tier URLs omit the L0 tier (for example
    ``22-gemmascope-2-res-16k``), and unpublished tiers fail loudly.

    The same combo validation as :func:`resolve_sae` is applied so that
    callers building Neuronpedia URLs from typed config receive an early,
    actionable error rather than a 404 when the URL is fetched. See
    consolidation manifest §1 (slice 1) for context.

    Raises:
        ValueError: If the combo is invalid for the release, if the v2 combo
            has no published Neuronpedia id, or if the release is unsupported
            for Neuronpedia.
    """
    if release == SAERelease.gemmascope:
        # Validate the combo using the same gate as resolve_sae so callers
        # building Neuronpedia URLs from typed config also get early errors.
        if model in _GEMMASCOPE_MODEL_SIZES:
            _validate_gemmascope(
                model, activation_type, layer, width, skip_canonical_gate=True
            )
        suffix, _ = _GEMMASCOPE_ACTIVATION_MAP[activation_type]
        sae_source = f"{layer}-{release.value}-{suffix}-{width.value}"
        return model, sae_source

    if release == SAERelease.gemmascope_2:
        if not isinstance(l0, L0Tier):
            raise ValueError(
                "gemma-scope-2 Neuronpedia source requires an L0Tier. Got: "
                f"{l0!r}."
            )
        resolved = _resolve_gemmascope_2(model, activation_type, layer, width, l0)
        directory = get_pretrained_saes_directory()
        lookup = directory.get(resolved.release)
        neuronpedia_id = None
        if lookup is not None:
            neuronpedia_id = (lookup.neuronpedia_id or {}).get(resolved.sae_id)
        if not neuronpedia_id:
            raise ValueError(
                "Neuronpedia does not publish a source for "
                f"{resolved.release}/{resolved.sae_id} according to the "
                "SAELens registry."
            )
        if "/" not in neuronpedia_id:
            raise ValueError(
                f"SAELens registry has malformed Neuronpedia id "
                f"{neuronpedia_id!r} for {resolved.release}/{resolved.sae_id}"
            )
        model_id, sae_source = neuronpedia_id.split("/", 1)
        return model_id, sae_source

    raise ValueError(f"Unsupported release for Neuronpedia: {release.value}")


_SUFFIX_TO_ACTIVATION: dict[str, ActivationType] = {
    suffix: act for act, (suffix, _hook) in _GEMMASCOPE_ACTIVATION_MAP.items()
}


def coerce_activation_type(raw: ActivationType | str) -> ActivationType:
    """Coerce a recorded activation token into an :class:`ActivationType`.

    Accepts an ``ActivationType`` (returned unchanged), its enum value
    (``"residual"`` / ``"mlp"`` / ``"attention"``), or the Neuronpedia release
    suffix (``"res"`` / ``"mlp"`` / ``"att"``). The suffix spelling is looked up
    in :data:`_SUFFIX_TO_ACTIVATION`, which is derived from the canonical
    ``_GEMMASCOPE_ACTIVATION_MAP`` templates rather than hardcoded, so it cannot
    drift from the resolver's own release suffixes.

    Raises:
        ValueError: if ``raw`` is none of the accepted spellings.
    """
    if isinstance(raw, ActivationType):
        return raw
    token = str(raw)
    try:
        return ActivationType(token)
    except ValueError:
        pass
    activation = _SUFFIX_TO_ACTIVATION.get(token)
    if activation is not None:
        return activation
    raise ValueError(
        f"unrecognized activation {raw!r}; expected an ActivationType, one of "
        f"{[a.value for a in ActivationType]}, or a release suffix "
        f"{sorted(_SUFFIX_TO_ACTIVATION)}"
    )


def coerce_sae_width(raw: SAEWidth | str | int) -> SAEWidth:
    """Coerce a recorded width into an :class:`SAEWidth`.

    Accepts a ``SAEWidth`` (returned unchanged), its slug (``"16k"``), or an
    integer feature count (mapped back via :data:`SAE_WIDTH_TO_NUM_FEATURES`, the
    single width↔count authority).

    Raises:
        ValueError: if ``raw`` is neither a known slug nor a known feature count.
    """
    if isinstance(raw, SAEWidth):
        return raw
    try:
        return SAEWidth(str(raw))
    except ValueError:
        pass
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unrecognized SAE width {raw!r}; expected an SAEWidth, a width slug "
            f"{[w.value for w in SAEWidth]}, or an integer feature count"
        ) from exc
    for width, num_features in SAE_WIDTH_TO_NUM_FEATURES.items():
        if num_features == count:
            return width
    raise ValueError(
        f"SAE width feature count {count} matches no known SAEWidth "
        f"(known counts: {sorted(SAE_WIDTH_TO_NUM_FEATURES.values())})"
    )


def parse_neuronpedia_source(
    source: str,
) -> tuple[SAERelease, ActivationType, int, SAEWidth, "L0Tier | None"]:
    """Reverse of :func:`resolve_neuronpedia_ids`.

    Accepts three grammars:
      - v1 (4 segments): ``{layer}-{release}-{activation}-{width}``
      - v2 public medium (5 segments):
        ``{layer}-gemmascope-2-{activation}-{width}``
      - v2 explicit tier (6 segments):
        ``{layer}-gemmascope-2-{activation}-{width}-{tier}``

    Note that v2's release token is the literal "gemmascope-2" (with the
    hyphen). Neuronpedia's published GemmaScope-2 medium URLs omit the tier;
    that 5-segment form parses as :class:`L0Tier.medium`. For the v1 form the
    release is parsed via :class:`SAERelease` and the L0 tier is returned as
    ``None``.

    Args:
        source: Neuronpedia source string. v1 example:
            ``"8-gemmascope-res-16k"``. v2 public-medium example:
            ``"22-gemmascope-2-res-16k"``. v2 explicit-tier example:
            ``"22-gemmascope-2-res-16k-medium"``.

    Returns:
        Tuple ``(release, activation_type, layer, width, l0_tier)`` that can
        be passed (with ``l0=l0_tier``) to :func:`resolve_sae` to obtain the
        SAELens ``release`` + ``sae_id`` + TransformerLens ``hook_name``.
        ``l0_tier`` is ``None`` for v1 sources.

    Raises:
        ValueError: If the string has the wrong number of segments, an
            unknown release / activation / width / l0 tier, or a non-integer
            layer.
    """
    parts = source.split("-")
    # v2 has 5 or 6 segments because "gemmascope-2" itself contains a hyphen:
    # "22-gemmascope-2-res-16k" -> ["22","gemmascope","2","res","16k"]
    # "22-gemmascope-2-res-16k-medium" -> ["22","gemmascope","2","res","16k","medium"]
    # v1 has 4: ["8","gemmascope","res","16k"]
    # A source whose 2nd/3rd tokens are "gemmascope"/"2" is *intended* as v2;
    # require exactly 5 or 6 segments so a truncated v2 string (e.g.
    # "22-gemmascope-2-res") raises a clear segment-count error instead of
    # being misparsed by the v1 branch.
    looks_like_v2 = (
        len(parts) >= 3 and parts[1] == "gemmascope" and parts[2] == "2"
    )
    if looks_like_v2 and len(parts) == 5:
        layer_str, _, _, activation_str, width_str = parts
        release = SAERelease.gemmascope_2
        l0_tier = L0Tier.medium
    elif looks_like_v2 and len(parts) == 6:
        layer_str, _, _, activation_str, width_str, tier_str = parts
        release = SAERelease.gemmascope_2
        try:
            l0_tier: L0Tier | None = L0Tier(tier_str)
        except ValueError as exc:
            raise ValueError(
                f"Neuronpedia source {source!r}: unknown l0 tier {tier_str!r}; "
                f"valid: {[t.value for t in L0Tier]}"
            ) from exc
    elif not looks_like_v2 and len(parts) == 4:
        layer_str, release_str, activation_str, width_str = parts
        try:
            release = SAERelease(release_str)
        except ValueError as exc:
            raise ValueError(
                f"Neuronpedia source {source!r}: unknown release "
                f"{release_str!r}; valid: {sorted(r.value for r in SAERelease)}"
            ) from exc
        l0_tier = None
    else:
        raise ValueError(
            f"Neuronpedia source must have 4 (v1), 5 (v2 public medium), or "
            f"6 (v2 explicit tier) dash-separated segments; got {source!r} "
            f"with {len(parts)} segments."
        )

    try:
        layer = int(layer_str)
    except ValueError as exc:
        raise ValueError(
            f"Neuronpedia source {source!r}: layer {layer_str!r} must be int."
        ) from exc

    if activation_str not in _SUFFIX_TO_ACTIVATION:
        raise ValueError(
            f"Neuronpedia source {source!r}: unknown activation suffix "
            f"{activation_str!r}; valid: {sorted(_SUFFIX_TO_ACTIVATION.keys())}"
        )
    activation_type = _SUFFIX_TO_ACTIVATION[activation_str]
    try:
        width = SAEWidth(width_str)
    except ValueError as exc:
        raise ValueError(
            f"Neuronpedia source {source!r}: unknown width {width_str!r}; "
            f"valid: {sorted(w.value for w in SAEWidth)}"
        ) from exc
    return release, activation_type, layer, width, l0_tier
