"""Canonical filesystem paths for eval target caches.

Mirrors the steering.feature_paths convention so operators recognize the
component nesting. Caches live under ``nanochat_cache/eval_caches/`` by
default.

Layout::

    {base_dir}/eval_caches/{suite}/m_{model}/l{layer}/a_{act}/p_{pool}/f{width}/c{model_max_seq_len}/{feature_selection_hash[:16]}/

The ``feature_selection_hash`` lives at the leaf so different
SAE feature-selection configs (different hash) get separate caches
automatically. We use the first 16 hex chars of the hash to balance
uniqueness (~6e19 distinct values) with path readability. A pinned or
canonically resolved 64-character hash is required.

The path component order mirrors :func:`steering.feature_paths.build_feature_dir`
with one re-ordering: ``m_{model}`` is first under the suite (operators
typically slice across all sites for a single model). Other than that
re-ordering, the activation / pooling / width / context-length / hash
nesting matches.
"""

from __future__ import annotations

import re
from pathlib import Path

from beartype import beartype

from nanochat.common import get_base_dir
from steering.eval.context_cache_io import (
    CacheIdentity,
    feature_selection_hash_leaf,
)
from steering.feature_paths import sanitize_model_name
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAEWidth,
)

# Allowed pooling labels — keep aligned with feature_paths.PoolingMode values
# and CacheIdentity.pooling values.
_VALID_POOLING_LABELS = frozenset({"max_pool", "mean_pool", "last_token"})

# GemmaScope full activation name → short directory label, matching
# steering.feature_paths.ACTIVATION_LABELS.
_FULL_TO_LABEL = {
    "residual": "res",
    "mlp": "mlp",
    "attention": "att",
    # Short aliases pass through unchanged.
    "res": "res",
    "att": "att",
}

# Regex for GemmaScope-style sae_id strings:
# ``gemmascope-{activation}-l{layer}-{width}``
# where activation in {residual, mlp, attention} (or short aliases res/att/mlp)
# and width in {16k, 32k, 65k, 131k, 262k, 524k, 1m}.
_SAE_ID_RE = re.compile(
    r"^gemmascope-([a-z]+)-l(\d+)-([0-9]+[km]|1m)$",
    re.IGNORECASE,
)


def _parse_sae_id(sae_id: str) -> tuple[str, int]:
    """Best-effort parse of ``(activation_label, width)`` from a
    GemmaScope-style ``sae_id``.

    Recognized forms:
      * ``"gemmascope-residual-l8-16k"`` -> ``("res", 16384)``
      * ``"gemmascope-mlp-l8-65k"``      -> ``("mlp", 65536)``
      * ``"gemmascope-attention-l8-16k"`` -> ``("att", 16384)``
      * ``"gemmascope-res-l8-16k"``       -> ``("res", 16384)``  (short alias)
      * ``"gemmascope-att-l8-16k"``       -> ``("att", 16384)``  (short alias)

    Raises ``ValueError`` if it can't parse — operators can override via
    the optional ``activation_label`` / ``sae_width`` args of
    :func:`default_target_cache_dir`.
    """
    m = _SAE_ID_RE.match(sae_id)
    if not m:
        raise ValueError(
            f"cannot parse sae_id {sae_id!r}: expected GemmaScope-style "
            f"'gemmascope-{{activation}}-l{{layer}}-{{width}}' "
            f"(e.g. 'gemmascope-residual-l8-16k'). Pass activation_label "
            f"and sae_width to default_target_cache_dir to override."
        )
    activation_str, _layer_str, width_str = m.group(1), m.group(2), m.group(3)
    activation_str_lower = activation_str.lower()
    if activation_str_lower not in _FULL_TO_LABEL:
        # Try via ActivationType values (normalize "residual" / "mlp" /
        # "attention").
        try:
            ActivationType(activation_str_lower)
        except ValueError as exc:
            raise ValueError(
                f"cannot parse sae_id {sae_id!r}: unknown activation "
                f"{activation_str!r}; expected one of: "
                f"{sorted(_FULL_TO_LABEL.keys())}"
            ) from exc
    activation_label = _FULL_TO_LABEL[activation_str_lower]
    width_str_lower = width_str.lower()
    try:
        width_enum = SAEWidth(width_str_lower)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse sae_id {sae_id!r}: unknown width "
            f"{width_str!r}; expected one of: {sorted(w.value for w in SAEWidth)}"
        ) from exc
    width = SAE_WIDTH_TO_NUM_FEATURES[width_enum]
    return activation_label, width


@beartype
def default_target_cache_dir(
    *,
    suite: str,
    sae_identity: CacheIdentity,
    model_max_seq_len: PositiveInt,
    base_dir: str | Path | None = None,
    activation_label: str | None = None,
    sae_width: int | None = None,
) -> Path:
    """Return the canonical filesystem path for an eval target cache.

    Layout::

        {base_dir}/eval_caches/{suite}/m_{model}/l{layer}/a_{act}/p_{pool}/f{width}/c{model_max_seq_len}/{feature_selection_hash[:16]}/

    Args:
        suite: One of ``"chat_target" | "core_target" | "samples" |
            "harmfulness"`` (or another suite name registered by an operator).
            Empty string is rejected.
        sae_identity: The :class:`CacheIdentity` instance carrying
            ``model_id``, ``sae_id``, ``layer``, ``pooling``,
            ``feature_selection_hash``.
        model_max_seq_len: Context length the cache was built for.
            Must be a positive integer.
        base_dir: Filesystem root for caches. When ``None``, falls back to
            :func:`nanochat.common.get_base_dir`. Pass an absolute path
            to control the location explicitly (e.g. for tests).
        activation_label: Override the parsed activation label
            (``"res" | "mlp" | "att"``). If passed, ``sae_width`` must
            also be passed; ``_parse_sae_id`` is bypassed entirely.
        sae_width: Override the parsed SAE width as a raw integer
            (e.g. ``16384``). Must be passed together with
            ``activation_label``.

    Returns:
        The canonical :class:`Path` for the cache. The directory itself
        is NOT created — the caller is responsible for mkdir'ing.

    Raises:
        ValueError: When ``suite`` is empty, ``model_max_seq_len`` is
            non-positive, ``sae_identity.pooling`` is not one of the
            allowed labels, or only one of ``activation_label``/``sae_width``
            is passed (both must be passed together), or when
            ``_parse_sae_id`` fails and no overrides are provided.
    """
    if not suite or not isinstance(suite, str):
        raise ValueError(f"suite must be a non-empty string; got {suite!r}")
    if sae_identity.pooling not in _VALID_POOLING_LABELS:
        raise ValueError(
            f"sae_identity.pooling must be one of "
            f"{sorted(_VALID_POOLING_LABELS)}; got {sae_identity.pooling!r}"
        )
    # Both override args must be passed together (or neither).
    if (activation_label is None) != (sae_width is None):
        raise ValueError(
            "both activation_label and sae_width must be provided together "
            "when overriding sae_id parsing; got "
            f"activation_label={activation_label!r}, sae_width={sae_width!r}"
        )

    # Resolve activation/width — overrides win, otherwise parse the sae_id.
    if activation_label is not None and sae_width is not None:
        act_label = activation_label
        width = sae_width
    else:
        act_label, width = _parse_sae_id(sae_identity.sae_id)

    hash_leaf = feature_selection_hash_leaf(sae_identity.feature_selection_hash)

    # Resolve base directory.
    if base_dir is None:
        base = Path(get_base_dir())
    else:
        base = Path(base_dir)

    encoded_model = sanitize_model_name(sae_identity.model_id)
    return (
        base
        / "eval_caches"
        / suite
        / f"m_{encoded_model}"
        / f"l{int(sae_identity.layer)}"
        / f"a_{act_label}"
        / f"p_{sae_identity.pooling}"
        / f"f{int(width)}"
        / f"c{model_max_seq_len}"
        / hash_leaf
    )
