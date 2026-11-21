"""Register gemma-scope-2 SAELens releases that exist on HuggingFace but
are missing from SAELens 6.42's `pretrained_saes_directory`.

The 8 releases handled here (1b-pt-{res,mlp,att}, 12b-pt-{res,mlp,att},
27b-pt-{mlp,att}) point at `google/gemma-scope-2-{size}-pt` HF repos and
use the `gemma_3` conversion function SAELens already ships. We construct
`PretrainedSAELookup` records from the same validation matrix the resolver
consults, so the validation table is the single source of truth for what
SAEs are loadable.

The module performs its side effect at import time and is idempotent.
:func:`inject_missing_v2_releases` is exposed for test fixtures and
re-import scenarios.
"""

from __future__ import annotations

import dataclasses

from loguru import logger
from sae_lens.loading.pretrained_saes_directory import (
    PretrainedSAELookup,
    get_pretrained_saes_directory,
)

from steering.sae_resolver import ActivationType
from steering.sae_resolver_v2_data import (
    GEMMASCOPE_2_HF_FOLDER,
    GEMMASCOPE_2_HF_MODEL_IDS,
    GEMMASCOPE_2_VALID_COMBOS,
    INJECTED_RELEASES,
)

_EXPECTED_PRETRAINED_SAE_LOOKUP_FIELDS: frozenset[str] = frozenset({
    "release", "repo_id", "model", "conversion_func", "saes_map",
    "neuronpedia_id", "expected_var_explained", "expected_l0",
    "norm_scaling_factor", "config_overrides",
})


def _assert_schema_unchanged() -> None:
    names = {f.name for f in dataclasses.fields(PretrainedSAELookup)}
    if names != _EXPECTED_PRETRAINED_SAE_LOOKUP_FIELDS:
        raise RuntimeError(
            "PretrainedSAELookup schema drifted from what the registry "
            f"injection expects.\n  expected: {sorted(_EXPECTED_PRETRAINED_SAE_LOOKUP_FIELDS)}\n"
            f"  found:    {sorted(names)}\n"
            "Update steering/sae_resolver_v2_registry.py to match."
        )


def _activation_for_release_suffix(suffix: str) -> ActivationType:
    return {
        "res": ActivationType.residual,
        "mlp": ActivationType.mlp,
        "att": ActivationType.attention,
    }[suffix]


def _build_saes_map(model_name: str, activation: ActivationType) -> dict[str, str]:
    """Build a saes_map dict from the validation table for one (model, activation)."""
    folder = GEMMASCOPE_2_HF_FOLDER[activation]
    out: dict[str, str] = {}
    for (act, width), tier_layers in GEMMASCOPE_2_VALID_COMBOS[model_name].items():
        if act != activation:
            continue
        for tier, layers in tier_layers.items():
            for L in layers:
                key = f"layer_{L}_width_{width.value}_l0_{tier.value}"
                out[key] = f"{folder}/{key}"
    return out


def inject_missing_v2_releases() -> None:
    """Insert the 8 unregistered v2 PT releases into SAELens's cached directory.

    Idempotent: a release already present in the directory is skipped, so
    subsequent calls are no-ops without any module-global guard. Never
    mutates an existing registry entry; if SAELens ever adds one of these
    releases upstream, we log a debug message and leave the upstream record
    alone.
    """
    _assert_schema_unchanged()

    directory = get_pretrained_saes_directory()
    for release_name, model_name, repo_id in INJECTED_RELEASES:
        if release_name in directory:
            logger.debug(
                f"Skipping registry injection for {release_name!r}: "
                f"already registered upstream."
            )
            continue
        suffix = release_name.rsplit("-", 1)[1]  # "res" / "mlp" / "att"
        activation = _activation_for_release_suffix(suffix)
        saes_map = _build_saes_map(model_name, activation)
        directory[release_name] = PretrainedSAELookup(
            release=release_name,
            repo_id=repo_id,
            model=GEMMASCOPE_2_HF_MODEL_IDS[model_name],
            conversion_func="gemma_3",
            saes_map=saes_map,
            neuronpedia_id={},
            expected_var_explained={},
            expected_l0={},
            norm_scaling_factor={},
            config_overrides=None,
        )


inject_missing_v2_releases()
