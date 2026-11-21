"""Hydra and source/destination contracts for the topology migration tool."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf

from scripts.train.conditional_encdec_train import _checkpoint_user_config
from steering.tools.migrate_encdec_train_state import (
    _require_bool,
    _validate_config_contract,
)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")
TRAINING_CONFIG_DIR = str(Path(CONFIG_DIR) / "training")
SOURCE_CONFIG = "full3b_clustered_c2048_long_w31"
DESTINATION_CONFIG = "full3b_clustered_c2048_long_w32_bf16_b14"


def test_tool_config_composes_the_canonical_w32_destination(hydra_compose) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "tools/migrate_encdec_train_state",
        [
            "source_checkpoint_dir=/checkpoint",
            "source_step=12345",
            "packed_row_dir=/packed",
            "selection_dir=/selection",
            "destination_dir=/destination",
        ],
    )

    assert cfg.source_step == 12345
    assert cfg.destination_world_size == 32
    assert cfg.dry_run is True
    assert cfg.max_replay_rows is None
    assert cfg.destination_training.run == DESTINATION_CONFIG
    assert cfg.destination_training.fp8 is False
    assert cfg.destination_training.device_batch_size == 14
    assert cfg.destination_training.total_batch_size == 32 * 14 * 2048 * 2


def test_canonical_w31_to_w32_contract_preserves_raw_step_and_accumulation(
    hydra_compose,
) -> None:
    source = hydra_compose(TRAINING_CONFIG_DIR, SOURCE_CONFIG)
    destination = hydra_compose(TRAINING_CONFIG_DIR, DESTINATION_CONFIG)
    source_user = _checkpoint_user_config(source)
    source_meta = {"user_config": source_user}

    target_user, source_total, destination_total, device_batch, accumulation = (
        _validate_config_contract(
            source_meta,
            destination,
            source_world_size=31,
            destination_world_size=32,
            step=98_765,
        )
    )

    assert source_total == 31 * 14 * 2048 * 2
    assert destination_total == 32 * 14 * 2048 * 2
    assert device_batch == 14
    assert accumulation == 2
    assert target_user["resume_from_step"] == 98_765
    assert target_user["num_iterations"] == source_user["num_iterations"] == 250_000
    assert target_user["target_param_data_ratio"] == source_user["target_param_data_ratio"]
    assert math.sqrt(destination_total / source_total) == pytest.approx(math.sqrt(32 / 31))


def test_packaged_tool_destination_matches_native_checkpoint_config(
    hydra_compose,
) -> None:
    source = hydra_compose(TRAINING_CONFIG_DIR, SOURCE_CONFIG)
    tool = hydra_compose(
        CONFIG_DIR,
        "tools/migrate_encdec_train_state",
        [
            "source_checkpoint_dir=/checkpoint",
            "source_step=5000",
            "packed_row_dir=/packed",
            "selection_dir=/packed/sel/hash",
            "destination_dir=/destination",
        ],
    )

    target_user, *_ = _validate_config_contract(
        {"user_config": _checkpoint_user_config(source)},
        tool.destination_training,
        source_world_size=31,
        destination_world_size=32,
        step=5000,
    )

    assert "hydra" not in target_user
    assert target_user["resume_from_step"] == 5000


@pytest.mark.parametrize("schema", ["canonical", "legacy", "partial", "conflict", "null_mirror"])
def test_topology_migration_admits_only_explicit_consistent_head_architecture(hydra_compose, schema):
    source = hydra_compose(TRAINING_CONFIG_DIR, SOURCE_CONFIG)
    destination = hydra_compose(TRAINING_CONFIG_DIR, DESTINATION_CONFIG)
    raw_head = {
        "weight": 0.1, "layer": -1, "loss_mode": "max_pool", "loss_fn": "bce",
        "block": "transformer", "head_layers": 2,
    }
    source.feature_prediction = raw_head
    destination.feature_prediction = raw_head
    source_user = _checkpoint_user_config(source)
    meta = {"user_config": source_user, "feature_prediction_config": dict(raw_head)}
    if schema == "legacy":
        source_user["feature_prediction"].pop("block")
        source_user["feature_prediction"]["mlp_layers"] = source_user["feature_prediction"].pop("head_layers")
    elif schema == "partial":
        source_user["feature_prediction"].pop("head_layers")
    elif schema == "conflict":
        meta["feature_prediction_config"]["head_layers"] = 4
    elif schema == "null_mirror":
        meta["feature_prediction_config"] = None
    kwargs = {"source_world_size": 31, "destination_world_size": 32, "step": 5000}
    if schema == "canonical":
        target_user, *_ = _validate_config_contract(meta, destination, **kwargs)
        assert target_user["feature_prediction"] == raw_head
    else:
        with pytest.raises(ValueError, match="removed|explicit|conflicting"):
            _validate_config_contract(meta, destination, **kwargs)


def test_config_contract_rejects_non_topology_schedule_changes(hydra_compose) -> None:
    source = hydra_compose(TRAINING_CONFIG_DIR, SOURCE_CONFIG)
    destination = hydra_compose(TRAINING_CONFIG_DIR, DESTINATION_CONFIG)
    destination = cast(
        DictConfig,
        OmegaConf.create(OmegaConf.to_container(destination, resolve=True)),
    )
    destination.warmup_ratio += 0.001

    with pytest.raises(ValueError, match="differs from the source beyond topology identity"):
        _validate_config_contract(
            {"user_config": _checkpoint_user_config(source)},
            destination,
            source_world_size=31,
            destination_world_size=32,
            step=1,
        )


@pytest.mark.parametrize("mutation", ["missing_null", "int_for_float"])
def test_config_contract_uses_strict_key_and_scalar_identity(
    hydra_compose,
    mutation: str,
) -> None:
    source = hydra_compose(TRAINING_CONFIG_DIR, SOURCE_CONFIG)
    destination = hydra_compose(TRAINING_CONFIG_DIR, DESTINATION_CONFIG)
    source_user = _checkpoint_user_config(source)
    if mutation == "missing_null":
        assert source_user.pop("profile_trace") is None
    else:
        assert source_user["warmup_ratio"] == 0.0
        source_user["warmup_ratio"] = 0

    with pytest.raises(ValueError, match="differs from the source beyond topology identity"):
        _validate_config_contract(
            {"user_config": source_user},
            destination,
            source_world_size=31,
            destination_world_size=32,
            step=1,
        )


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_dry_run_requires_a_literal_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="dry_run must be a bool"):
        _require_bool(value, name="dry_run")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("fp8", True, "requires BF16"),
        ("label_dropout", 0.1, "requires label_dropout=0"),
        ("use_precomputed_batches", False, "requires precomputed batches"),
    ],
)
def test_config_contract_rejects_unsupported_source_semantics(
    hydra_compose,
    field: str,
    value: object,
    match: str,
) -> None:
    source = hydra_compose(TRAINING_CONFIG_DIR, SOURCE_CONFIG)
    destination = hydra_compose(TRAINING_CONFIG_DIR, DESTINATION_CONFIG)
    destination = cast(
        DictConfig,
        OmegaConf.create(OmegaConf.to_container(destination, resolve=True)),
    )
    source_user = _checkpoint_user_config(source)
    source_user[field] = value
    destination[field] = value

    with pytest.raises(ValueError, match=match):
        _validate_config_contract(
            {"user_config": source_user},
            destination,
            source_world_size=31,
            destination_world_size=32,
            step=1,
        )
