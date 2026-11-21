"""Offline registry proofs for AxBench-Base runtime SAE identity checks.

The GemmaScope-2 half is a registry-level proof, not a loader-execution proof.
It deliberately does not call ``get_gemma_3_config_from_hf`` because published
repositories with ``config.json`` make that helper access Hugging Face.
"""

import sae_lens
from sae_lens.loading import pretrained_sae_loaders
from sae_lens.loading.pretrained_saes_directory import (
    get_config_overrides,
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)

from steering.eval.axbench_base.config import bare_model_id
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
    resolve_sae,
)
from steering.sae_resolver_v2_data import (
    GEMMASCOPE_2_HF_MODEL_IDS,
    GEMMASCOPE_2_VALID_COMBOS,
)

_V1_BARE_MODEL_IDS = (
    "gemma-2-2b",
    "gemma-2-9b",
    "gemma-2-27b",
)


def _assert_sae_lens_version() -> None:
    assert sae_lens.__version__ == "6.42.0", (
        "runtime SAE identity registry proofs are pinned to sae_lens 6.42.0; "
        "a version bump requires re-verifying the loader's model-name "
        "derivation before these offline proofs can certify it"
    )


def test_all_registry_resolvable_v1_identities_match_runtime_expectations() -> None:
    """Every v1 resolver/registry overlap agrees on reported identities."""
    _assert_sae_lens_version()
    directory = get_pretrained_saes_directory()
    accepted_count = 0
    checked_count = 0

    for model in _V1_BARE_MODEL_IDS:
        for activation_type in ActivationType:
            for width in SAEWidth:
                # Gemma-2 models have fewer than 64 transformer layers. The
                # bound intentionally exceeds every sae_lens 6.42.0 v1 model
                # depth while keeping this registry-only proof explicit.
                for layer in range(64):
                    try:
                        resolved = resolve_sae(
                            model=model,
                            release=SAERelease.gemmascope,
                            activation_type=activation_type,
                            layer=layer,
                            width=width,
                            l0=None,
                        )
                    except ValueError:
                        continue
                    accepted_count += 1

                    lookup = directory.get(resolved.release)
                    if lookup is None or resolved.sae_id not in lookup.saes_map:
                        continue

                    registry_neuronpedia_id = lookup.neuronpedia_id.get(
                        resolved.sae_id,
                    )
                    assert registry_neuronpedia_id is not None
                    neuronpedia_model_id, neuronpedia_source = (
                        resolve_neuronpedia_ids(
                            model=model,
                            release=SAERelease.gemmascope,
                            activation_type=activation_type,
                            layer=layer,
                            width=width,
                            l0=None,
                        )
                    )
                    assert (
                        f"{neuronpedia_model_id}/{neuronpedia_source}"
                        == registry_neuronpedia_id
                    )
                    repo_id, folder_name = get_repo_id_and_folder_name(
                        resolved.release,
                        resolved.sae_id,
                    )
                    loader_config = (
                        pretrained_sae_loaders.get_gemma_2_config_from_hf(
                            repo_id,
                            folder_name,
                            device="cpu",
                        )
                    )
                    assert loader_config["model_name"] == bare_model_id(model)
                    checked_count += 1

    assert checked_count >= 400, (
        "v1 registry overlap unexpectedly became vacuous: "
        f"checked={checked_count}, resolver_accepted={accepted_count}"
    )


def test_all_registry_resolvable_v2_identities_match_runtime_expectations() -> None:
    """The 116 published v2 Neuronpedia identities agree with runtime expectations.

    The other registry overlaps have null Neuronpedia ids because those
    combinations are unpublished. They are not checkable by this proof:
    ``resolve_neuronpedia_ids`` rejects them, and production deliberately
    tolerates that rejection via ``except ValueError``.
    """
    _assert_sae_lens_version()
    directory = get_pretrained_saes_directory()
    accepted_count = 0
    registry_overlap_count = 0
    published_neuronpedia_count = 0

    for model, combinations in GEMMASCOPE_2_VALID_COMBOS.items():
        # SAELens 6.42.0 pretrained_sae_loaders.py:638-641 force-prefixes
        # Gemma-3 ids with ``google/``. Derive that loader expectation from the
        # repository's registry source of truth without calling the networked
        # get_gemma_3_config_from_hf helper.
        loader_derived_model_name = GEMMASCOPE_2_HF_MODEL_IDS[model]
        assert bare_model_id(loader_derived_model_name) == bare_model_id(model)

        for (activation_type, width), tiers in combinations.items():
            for l0, layers in tiers.items():
                for layer in layers:
                    try:
                        resolved = resolve_sae(
                            model=model,
                            release=SAERelease.gemmascope_2,
                            activation_type=activation_type,
                            layer=layer,
                            width=width,
                            l0=l0,
                        )
                    except ValueError:
                        continue
                    accepted_count += 1

                    lookup = directory.get(resolved.release)
                    if lookup is None or resolved.sae_id not in lookup.saes_map:
                        continue
                    registry_overlap_count += 1

                    registry_neuronpedia_id = lookup.neuronpedia_id.get(
                        resolved.sae_id,
                    )
                    # The registry deliberately publishes Neuronpedia ids only
                    # for tiers that Neuronpedia itself exposes. The runtime
                    # likewise validates this field only when the loaded SAE
                    # reports it (D7 tolerate-absent semantics).
                    if registry_neuronpedia_id is None:
                        continue
                    neuronpedia_model_id, neuronpedia_source = (
                        resolve_neuronpedia_ids(
                            model=model,
                            release=SAERelease.gemmascope_2,
                            activation_type=activation_type,
                            layer=layer,
                            width=width,
                            l0=l0,
                        )
                    )
                    assert (
                        f"{neuronpedia_model_id}/{neuronpedia_source}"
                        == registry_neuronpedia_id
                    )
                    published_neuronpedia_count += 1

    assert registry_overlap_count >= 1_200, (
        "v2 registry overlap unexpectedly became vacuous: "
        f"overlap={registry_overlap_count}, resolver_accepted={accepted_count}"
    )
    assert published_neuronpedia_count >= 100, (
        "v2 published Neuronpedia overlap unexpectedly became vacuous: "
        f"published={published_neuronpedia_count}, "
        f"registry_overlap={registry_overlap_count}"
    )


# ---------------------------------------------------------------------------
# Shipped-config tier provability.
#
# The v2 fail-closed rule raises when a loaded SAE exposes neither a
# tier-discriminating ``sae_id`` nor a ``neuronpedia_id``. Against real
# SAELens 6.42.0 loads ``sae_id`` is never populated (``LLM_METADATA_KEYS``
# has no such field), so for GemmaScope-2 the tier is provable ONLY via the
# registry-injected ``neuronpedia_id`` -- which is null for 1,148 of the
# 1,264 registry-overlapping combinations.
#
# A shipped config whose combination fell in that null majority would fail
# closed at Stage 1. Nothing else in this suite pins that, and its absence
# is exactly what let two independent reviews reach opposite conclusions
# about whether this repository's own Gemma-3 config still loads. This test
# settles it, offline, and will fail loudly if a future SAELens registry
# drops the entry the shipped config depends on.
# ---------------------------------------------------------------------------

_SHIPPED_NEURONPEDIA_SOURCES = (
    # (config file, extraction model id, neuronpedia_source)
    (
        "configs/eval/axbench_base/axbench_base.yaml",
        "gemma-2-2b",
        "8-gemmascope-res-16k",
    ),
    (
        "configs/eval/axbench_base/axbench_base_gemma3_4bpt_l22_f65k.yaml",
        "gemma-3-4b-pt",
        "22-gemmascope-2-res-65k",
    ),
)


def test_shipped_configs_can_prove_their_l0_tier() -> None:
    """Every shipped AxBench-Base SAE source survives the v2 fail-closed rule.

    v1 sources are exempt by construction -- the rule only fires for
    ``SAERelease.gemmascope_2`` -- so this asserts exemption for those and
    real registry provability for the v2 ones.
    """
    checked = 0
    for config_path, model, source in _SHIPPED_NEURONPEDIA_SOURCES:
        release, activation_type, layer, width, l0 = parse_neuronpedia_source(source)
        resolved = resolve_sae(
            model=model,
            release=release,
            activation_type=activation_type,
            layer=layer,
            width=width,
            l0=l0,
        )
        if release is not SAERelease.gemmascope_2:
            # Fail-closed does not apply; nothing to prove.
            checked += 1
            continue
        registry_neuronpedia_id = get_config_overrides(
            resolved.release, resolved.sae_id
        ).get("neuronpedia_id")
        assert registry_neuronpedia_id, (
            f"{config_path} resolves to {resolved.release}/{resolved.sae_id}, "
            "which the SAELens registry does not publish a neuronpedia_id for. "
            "A real load therefore cannot prove its L0 tier and the v2 "
            "fail-closed rule will reject this shipped config at Stage 1."
        )
        expected_model_id, expected_source = resolve_neuronpedia_ids(
            model=model,
            release=release,
            activation_type=activation_type,
            layer=layer,
            width=width,
            l0=l0,
        )
        assert registry_neuronpedia_id == f"{expected_model_id}/{expected_source}"
        checked += 1

    assert checked == len(_SHIPPED_NEURONPEDIA_SOURCES)
