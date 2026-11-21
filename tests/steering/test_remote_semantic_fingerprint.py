from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import steering.remote_training as remote_module
from steering.remote_training import (
    RemoteTrainingError,
    build_remote_run_fingerprint,
    build_remote_semantic_fingerprint_v2,
)


def _contracts() -> dict[str, dict[str, str]]:
    return {
        label: {"selection_checksum": char * 64}
        for label, char in zip(("train", "derived_val", "base_val"), "abc", strict=True)
    }


def _minimal_config() -> dict[str, Any]:
    return {
        "max_seq_len": 1024,
        "depth": 12,
        "device_batch_size": 16,
        "core_metric_every": 250,
        "sample_every": 250,
        "decoder_segment_size": None,
    }


def test_v1_and_v2_golden_serializations_are_pinned():
    config = _minimal_config()
    contracts = _contracts()

    assert (
        build_remote_run_fingerprint(
            resolved_training_config=config,
            world_size=2,
            selection_contracts=contracts,
        )
        == "930abec12673cad416dfb27b9f9c6c028974c819d45ce416b2233369635432d9"
    )
    assert (
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=config,
            world_size=2,
            selection_contracts=contracts,
        )
        == "d06033d6eee10e8d344d2e48f6accd6ac74c1722844f4f5cadc495b2100fb84a"
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("device_batch_size", 8),
        ("core_metric_every", 0),
        ("sample_every", 0),
        ("save_every", 17),
        ("feature_dir", "/different/local/features"),
        ("feature_mask_freq_min", 0.25),
        ("selection_num_workers", 99),
        (
            "remote_preselection",
            {"staging_dir": "/tmp/producer", "max_output_shard_bytes": 1_000_000},
        ),
    ],
)
def test_v2_excludes_observation_execution_and_producer_inputs(key, value):
    config = _minimal_config()
    baseline = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=2,
        selection_contracts=_contracts(),
    )

    assert (
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={**config, key: value},
            world_size=2,
            selection_contracts=_contracts(),
        )
        == baseline
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("depth", 13),
        ("total_batch_size", 2**21),
        ("fp8", True),
        ("fp8_recipe", "rowwise"),
        ("seed", 43),
        ("label_dropout", 0.25),
    ],
)
def test_v2_binds_semantic_inputs(key, value):
    config = _minimal_config()
    baseline = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=2,
        selection_contracts=_contracts(),
    )

    assert (
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={**config, key: value},
            world_size=2,
            selection_contracts=_contracts(),
        )
        != baseline
    )


def test_v2_binds_world_size_and_complete_selection_contracts():
    config = _minimal_config()
    contracts = _contracts()
    baseline = build_remote_semantic_fingerprint_v2(
        resolved_training_config=config,
        world_size=2,
        selection_contracts=contracts,
    )

    changed = copy.deepcopy(contracts)
    changed["train"]["selection_checksum"] = "f" * 64
    assert (
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=config,
            world_size=2,
            selection_contracts=changed,
        )
        != baseline
    )
    assert (
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=config,
            world_size=3,
            selection_contracts=contracts,
        )
        != baseline
    )


@pytest.mark.parametrize("world_size", [True, 0, -1, 2.0])
def test_v2_rejects_invalid_world_sizes(world_size):
    with pytest.raises(BeartypeCallHintParamViolation, match="world_size"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=_minimal_config(),
            world_size=cast(int, world_size),
            selection_contracts=_contracts(),
        )


def test_v2_classifier_fails_loudly_for_unknown_or_unsupported_values():
    config = _minimal_config()
    contracts = _contracts()

    with pytest.raises(RemoteTrainingError, match="not classified"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={**config, "future_training_knob": 1},
            world_size=2,
            selection_contracts=contracts,
        )
    with pytest.raises(RemoteTrainingError, match="unclassified nested"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={**config, "encoder": {"future_geometry": 1}},
            world_size=2,
            selection_contracts=contracts,
        )
    with pytest.raises(RemoteTrainingError, match="unclassified nested mapping"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={
                **config,
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.1,
                    "layer": -1,
                    "loss_mode": "max_pool",
                    "loss_fn": {"future": "bce"},
                },
            },
            world_size=2,
            selection_contracts=contracts,
        )
    with pytest.raises(RemoteTrainingError, match="non-string root key"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=cast(
                Mapping[str, Any],
                {**config, 7: "invalid"},
            ),
            world_size=2,
            selection_contracts=contracts,
        )
    with pytest.raises(RemoteTrainingError, match="must remain null"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={**config, "decoder_segment_size": 128},
            world_size=2,
            selection_contracts=contracts,
        )
    with pytest.raises(RemoteTrainingError, match="unsupported value"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=config,
            world_size=2,
            selection_contracts={
                **contracts,
                "train": {"selection_checksum": object()},
            },
        )
    with pytest.raises(RemoteTrainingError, match="non-finite"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={**config, "weight_decay": float("nan")},
            world_size=2,
            selection_contracts=contracts,
        )
    with pytest.raises(RemoteTrainingError, match="train, derived_val, and base_val"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=config,
            world_size=2,
            selection_contracts={"train": contracts["train"]},
        )


def test_v2_classifier_rejects_multiply_classified_keys(monkeypatch):
    categories = dict(remote_module._CONFIG_CATEGORIES_V2)
    categories["observation"] = categories["observation"] | {"depth"}
    monkeypatch.setattr(remote_module, "_CONFIG_CATEGORIES_V2", categories)

    with pytest.raises(RemoteTrainingError, match="classifier is ambiguous"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=_minimal_config(),
            world_size=2,
            selection_contracts=_contracts(),
        )


@pytest.mark.parametrize("field,value", [("block", "transformer"), ("head_layers", 4)])
def test_feature_head_architecture_is_semantic(field, value):
    config = {**_minimal_config(), "feature_prediction": {"block": "mlp", "head_layers": 0}}
    changed = {**config, "feature_prediction": {**config["feature_prediction"], field: value}}
    fingerprints = [
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=candidate, world_size=2, selection_contracts=_contracts()
        )
        for candidate in (config, changed)
    ]
    assert fingerprints[0] != fingerprints[1]


def test_all_shipped_remote_presets_have_complete_disjoint_v2_classification():
    config_dir = Path("configs/training").resolve()
    config_names = tuple(
        path.relative_to(config_dir).with_suffix("").as_posix()
        for path in sorted(
            (config_dir / "cluster_fp").glob(
                "conditional_encdec_splice_cf_gemma3pt_c1024_remote*.yaml"
            )
        )
    )
    assert config_names == (
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote",
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_100m_2node_2xh100",
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed",
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_2xh200",
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100",
    )

    categories = remote_module._CONFIG_CATEGORIES_V2
    classified = [key for keys in categories.values() for key in keys]
    assert len(classified) == len(set(classified))
    fingerprints: set[str] = set()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for config_name in config_names:
            resolved = OmegaConf.to_container(
                compose(config_name=config_name),
                resolve=True,
            )
            assert isinstance(resolved, dict)
            assert set(resolved) <= set(classified)
            fingerprints.add(
                build_remote_semantic_fingerprint_v2(
                    resolved_training_config=cast(Mapping[str, Any], resolved),
                    world_size=2,
                    selection_contracts=_contracts(),
                )
            )
    assert fingerprints == {
        "ab9083882654f6a002353df6e7edb45c10d9949532789ea862f07a5063cb39e7",
        "60bd794b21ba7f8cb47f5f57df38522bdd324d791a5e044d5c753fd0d5f64243",
    }


def test_every_shipped_root_key_has_declared_v2_fingerprint_behavior():
    config_dir = Path("configs/training").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        resolved = OmegaConf.to_container(
            compose(
                config_name=(
                    "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100"
                )
            ),
            resolve=True,
        )
    assert isinstance(resolved, dict)
    resolved = cast(dict[str, Any], resolved)
    categories = remote_module._CONFIG_CATEGORIES_V2
    baseline = build_remote_semantic_fingerprint_v2(
        resolved_training_config=resolved,
        world_size=8,
        selection_contracts=_contracts(),
    )

    for key in sorted(resolved):
        without_key = dict(resolved)
        del without_key[key]
        changed = build_remote_semantic_fingerprint_v2(
            resolved_training_config=without_key,
            world_size=8,
            selection_contracts=_contracts(),
        )
        if key == "conditioning_cooldown":
            assert resolved[key]["enabled"] is False
            assert changed == baseline, "disabled cooldown must preserve historical identity"
        elif key in categories["semantic"]:
            assert changed != baseline, f"semantic root key escaped identity: {key}"
        else:
            assert changed == baseline, f"non-semantic root key entered identity: {key}"
