"""Offline registry proof for unclustered runtime SAE identity checks."""

import sae_lens
from sae_lens.loading import pretrained_sae_loaders
from sae_lens.loading.pretrained_saes_directory import (
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)

from steering.eval.axbench_base.config import bare_model_id
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
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


def test_all_registry_resolvable_v1_identities_match_runtime_expectations() -> None:
    """U6: every resolver/registry overlap agrees on model and Neuronpedia ids."""
    assert sae_lens.__version__ == "6.42.0", (
        "test_all_registry_resolvable_v1_identities_match_runtime_expectations "
        "is pinned to sae_lens 6.42.0; a version bump requires re-verifying the "
        "loader's model-name derivation before this offline proof can certify it"
    )
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
        "registry overlap unexpectedly became vacuous: "
        f"checked={checked_count}, resolver_accepted={accepted_count}"
    )


def test_all_registry_resolvable_v2_identities_match_runtime_expectations() -> None:
    """U7: registry-level v2 proof, not a loader-execution proof.

    The test deliberately does not call ``get_gemma_3_config_from_hf`` because
    it downloads ``config.json`` when present. Instead, the loader-derived
    model name is taken from ``GEMMASCOPE_2_HF_MODEL_IDS``. SAELens 6.42.0's
    ``pretrained_sae_loaders.py:638-641`` force-prefixes bare Gemma-3 model
    names with ``google/`` and normalizes the 270m PT spelling, exactly matching
    that table.
    """
    assert sae_lens.__version__ == "6.42.0", (
        "test_all_registry_resolvable_v2_identities_match_runtime_expectations "
        "is pinned to sae_lens 6.42.0; a version bump requires re-verifying the "
        "Gemma-3 loader's model-name derivation before this offline proof can "
        "certify it"
    )
    directory = get_pretrained_saes_directory()
    accepted_count = 0
    registry_count = 0
    neuronpedia_count = 0

    for model, combinations in GEMMASCOPE_2_VALID_COMBOS.items():
        expected_loader_model_name = GEMMASCOPE_2_HF_MODEL_IDS[model]
        assert bare_model_id(expected_loader_model_name) == bare_model_id(model)

        for (activation_type, width), tier_layers in combinations.items():
            for l0, layers in tier_layers.items():
                for layer in layers:
                    resolved = resolve_sae(
                        model=model,
                        release=SAERelease.gemmascope_2,
                        activation_type=activation_type,
                        layer=layer,
                        width=width,
                        l0=l0,
                    )
                    accepted_count += 1

                    lookup = directory.get(resolved.release)
                    if lookup is None or resolved.sae_id not in lookup.saes_map:
                        continue
                    registry_count += 1
                    assert lookup.model == expected_loader_model_name

                    registry_neuronpedia_id = (lookup.neuronpedia_id or {}).get(
                        resolved.sae_id,
                    )
                    if registry_neuronpedia_id is None:
                        continue
                    neuronpedia_count += 1
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

    assert accepted_count >= 1200
    assert registry_count >= 1200, (
        "v2 registry overlap unexpectedly became vacuous: "
        f"registry={registry_count}, resolver_accepted={accepted_count}"
    )
    assert neuronpedia_count >= 100, (
        "v2 Neuronpedia registry overlap unexpectedly became vacuous: "
        f"neuronpedia={neuronpedia_count}, registry={registry_count}"
    )
