"""Checkpoint lineage, remote identity, and post-training cooldown contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from omegaconf import open_dict

from errors import SelfCondTrainingError
from scripts.train.conditional_encdec_selfcond import validate_selfcond_startup
from scripts.train.conditional_encdec_sft import _reject_unsupported_sft_pretrain_keys
from scripts.train.conditional_encdec_train import _checkpoint_user_config, _resolved_user_config
from steering.conditional.checkpoint_conditioning import (
    PRETRAIN_CONDITIONING_META_KEY,
    assert_conditioning_supported,
    read_checkpoint_conditioning,
)
from steering.conditional.encoder_decoder.conditioning_cooldown import ConditioningCooldown
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondDataSource,
    validate_selfcond_prerequisites,
)
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_checkpoint import (
    validate_grpo_checkpoint_eligibility,
)
from steering.remote_training import (
    RemoteTrainingError,
    build_remote_run_fingerprint,
    build_remote_semantic_fingerprint_v2,
)
from steering.tools.migrate_encdec_train_state import _validate_config_contract
from tests.scripts.test_conditional_encdec_selfcond_sft_sources import (
    _build_checkpoint_meta,
    _sft_shaped_meta,
)
from tests.steering.selfcond.helpers import tiny_head_model
from tests.steering.selfcond.test_failloud_guards import _cfg, _valid_meta
from tests.steering.selfcond.test_sft_grpo_checkpoint import _metadata, _model

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")
DEFAULT_COOLDOWN = {
    "enabled": False,
    "start_fraction": 1.0,
    "end_fraction": 0.5,
    "start_step": None,
    "end_step": None,
    "cooldown_tokens": 20_000_000_000,
    "granularity": "segment",
}


def _record(*, end_fraction: float = 0.5) -> dict:
    return ConditioningCooldown(
        start_fraction=1.0,
        end_fraction=end_fraction,
        start_step=10,
        end_step=29,
        granularity="segment",
        seed=42,
        world_size=2,
        device_batch_size=2,
        max_seq_len=4,
        total_batch_size=16,
        num_iterations=30,
        device_type="cpu",
    ).metadata_dict()


def _contracts() -> dict:
    return {label: {"selection_checksum": label} for label in ("train", "derived_val", "base_val")}


def test_cooldown_lineage_survives_two_sft_checkpoints_without_running_schedule() -> None:
    source = {
        "user_config": {
            "label_dropout": 0.0,
            "conditioning_cooldown": {
                **DEFAULT_COOLDOWN,
                "enabled": True,
                "start_step": 10,
                "end_step": 29,
            },
        },
        "conditioning_cooldown": _record(),
    }
    conditioning = read_checkpoint_conditioning(source)
    assert conditioning.conditioning_cooldown == _record()
    assert conditioning.has_scheduled_dropout()
    assert_conditioning_supported(conditioning, checkpoint_label="cooldown SFT input")
    for _ in range(2):
        source = {
            "user_config": {"dataset": "standard", "label_dropout": 0.0},
            "encoder_config": {},
            **conditioning.metadata_dict(),
        }
        conditioning = read_checkpoint_conditioning(json.loads(json.dumps(source)))
        assert conditioning.conditioning_cooldown == _record()
        assert conditioning.has_scheduled_dropout()
        assert "conditioning_cooldown" not in source["user_config"]


@pytest.mark.parametrize("raw", [None, DEFAULT_COOLDOWN])
def test_disabled_lineage_is_exactly_legacy(raw) -> None:
    source = {"user_config": {"label_dropout": 0.0}}
    expected = read_checkpoint_conditioning(source).metadata_dict()
    source["user_config"]["conditioning_cooldown"] = raw
    conditioning = read_checkpoint_conditioning(source)
    assert conditioning.metadata_dict() == expected
    assert conditioning.conditioning_cooldown is None
    assert not conditioning.has_scheduled_dropout()


def test_active_checkpoint_requires_resolved_record() -> None:
    source = {"user_config": {"conditioning_cooldown": {"enabled": True}}}
    with pytest.raises(ValueError, match="no resolved"):
        read_checkpoint_conditioning(source)


def test_raw_dropout_cannot_be_hidden_by_all_retained_resolved_record() -> None:
    source = {
        "user_config": {
            "conditioning_cooldown": {
                "enabled": True,
                "start_step": 10,
                "end_step": 29,
                "end_fraction": 0.5,
            },
        },
        "conditioning_cooldown": _record(end_fraction=1.0),
    }
    with pytest.raises(ValueError, match="config and resolved metadata disagree"):
        read_checkpoint_conditioning(source)


def test_inherited_dropout_cannot_be_hidden_by_all_retained_direct_record() -> None:
    source = {
        "conditioning_cooldown": _record(end_fraction=1.0),
        PRETRAIN_CONDITIONING_META_KEY: {"conditioning_cooldown": _record()},
    }
    with pytest.raises(
        ValueError, match="direct and inherited conditioning_cooldown metadata disagree"
    ):
        read_checkpoint_conditioning(source)


def test_direct_record_does_not_hide_malformed_inherited_provenance() -> None:
    source = {
        "conditioning_cooldown": _record(end_fraction=1.0),
        PRETRAIN_CONDITIONING_META_KEY: {"conditioning_cooldown": {"schema_version": 999}},
    }
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        read_checkpoint_conditioning(source)


def test_matching_direct_and_inherited_provenance_preserves_completed_source() -> None:
    source = {
        "step": 30,
        "user_config": {
            "conditioning_cooldown": {
                "enabled": True,
                "start_step": 10,
                "end_step": 29,
            },
        },
        "conditioning_cooldown": _record(),
        PRETRAIN_CONDITIONING_META_KEY: {"conditioning_cooldown": _record()},
    }
    conditioning = read_checkpoint_conditioning(source)
    assert conditioning.conditioning_cooldown == _record()
    assert conditioning.has_scheduled_dropout()


def test_all_retained_selfcond_checkpoint_keeps_resolved_cooldown_record() -> None:
    source = _sft_shaped_meta()
    source["conditioning_cooldown"] = _record(end_fraction=1.0)
    result = _build_checkpoint_meta(base_meta=source)
    assert result["conditioning_cooldown"] == source["conditioning_cooldown"]
    assert result["conditioning_cooldown"] is not source["conditioning_cooldown"]
    conditioning = read_checkpoint_conditioning(result)
    assert conditioning.conditioning_cooldown == _record(end_fraction=1.0)
    assert not conditioning.has_scheduled_dropout()


@pytest.mark.parametrize("location", ["top", "inherited"])
@pytest.mark.parametrize("mutation", ["unknown", "fraction", "endpoint", "version"])
def test_malformed_resolved_lineage_fails_before_sft(location, mutation) -> None:
    record = _record()
    if mutation == "unknown":
        record["unknown"] = 1
    elif mutation == "fraction":
        record["end_fraction"] = float("nan")
    elif mutation == "endpoint":
        record["start_step"] = record["end_step"]
    else:
        record["schema_version"] = 99
    source = {"conditioning_cooldown": record}
    if location == "inherited":
        source = {PRETRAIN_CONDITIONING_META_KEY: source}
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        read_checkpoint_conditioning(source)


@pytest.mark.parametrize("data_source", list(SelfCondDataSource))
def test_selfcond_rejects_direct_and_inherited_scheduled_dropout(data_source) -> None:
    source = _valid_meta()
    if data_source is SelfCondDataSource.PRETRAINING:
        source["conditioning_cooldown"] = _record()
    else:
        source[PRETRAIN_CONDITIONING_META_KEY] = {"conditioning_cooldown": _record()}
    with pytest.raises(SelfCondTrainingError, match="C5: conditioning_cooldown"):
        validate_selfcond_prerequisites(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32),
            source,
            _cfg(),
            32,
            data_source=data_source,
        )


def test_selfcond_rejects_unwired_runtime_schedule_even_if_all_retained() -> None:
    with pytest.raises(SelfCondTrainingError, match="conditioning_cooldown is not wired"):
        validate_selfcond_startup(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32),
            _valid_meta(),
            _cfg(),
            32,
            {"conditioning_cooldown": {"enabled": True, "end_fraction": 1.0}},
        )


@pytest.mark.parametrize("end_fraction", [0.0, 0.5, 1.0])
def test_grpo_checks_inherited_scheduled_dropout_with_zero_scalar(end_fraction) -> None:
    source = _metadata()
    source[PRETRAIN_CONDITIONING_META_KEY]["conditioning_cooldown"] = _record(
        end_fraction=end_fraction
    )
    if end_fraction < 1.0:
        with pytest.raises(SFTGRPOError, match="conditioning_cooldown"):
            validate_grpo_checkpoint_eligibility(
                _model(),
                source,
                checkpoint_dir="/checkpoint",
                action_mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
                seed=4,
            )
    else:
        assert (
            validate_grpo_checkpoint_eligibility(
                _model(),
                source,
                checkpoint_dir="/checkpoint",
                action_mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
                seed=4,
            ).feature_count
            == 8
        )


def test_sft_rejects_schedule_knobs_instead_of_ignoring_them() -> None:
    with pytest.raises(ValueError, match="pretrain-only key 'conditioning_cooldown'"):
        _reject_unsupported_sft_pretrain_keys({"conditioning_cooldown": DEFAULT_COOLDOWN})


@pytest.mark.parametrize(
    "builder", [build_remote_run_fingerprint, build_remote_semantic_fingerprint_v2]
)
@pytest.mark.parametrize("raw", [None, DEFAULT_COOLDOWN])
def test_disabled_raw_config_preserves_remote_fingerprint(builder, raw) -> None:
    cfg = {"max_seq_len": 1024, "label_dropout": 0.0}
    expected = builder(resolved_training_config=cfg, world_size=2, selection_contracts=_contracts())
    actual = builder(
        resolved_training_config={**cfg, "conditioning_cooldown": raw},
        world_size=2,
        selection_contracts=_contracts(),
    )
    assert actual == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_fraction", 0.9),
        ("end_fraction", 0.3),
        ("start_step", 10),
        ("end_step", 100),
        ("cooldown_tokens", 40_000_000_000),
        ("granularity", "document"),
    ],
)
def test_remote_identity_binds_every_active_schedule_field(field, value) -> None:
    block = {**DEFAULT_COOLDOWN, "enabled": True}
    cfg = {"max_seq_len": 1024, "conditioning_cooldown": block}
    original = build_remote_semantic_fingerprint_v2(
        resolved_training_config=cfg, world_size=2, selection_contracts=_contracts()
    )
    block[field] = value
    assert (
        build_remote_semantic_fingerprint_v2(
            resolved_training_config=cfg, world_size=2, selection_contracts=_contracts()
        )
        != original
    )


def test_remote_rejects_unknown_cooldown_knob_even_when_disabled() -> None:
    with pytest.raises(RemoteTrainingError, match="unclassified nested"):
        build_remote_semantic_fingerprint_v2(
            resolved_training_config={"conditioning_cooldown": {"enabled": False, "typo": 1}},
            world_size=2,
            selection_contracts=_contracts(),
        )


def test_disabled_runtime_and_checkpoint_configs_are_exact_legacy(hydra_compose) -> None:
    cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_train")
    legacy = copy.deepcopy(cfg)
    with open_dict(legacy):
        del legacy.conditioning_cooldown
    assert _resolved_user_config(cfg) == _resolved_user_config(legacy)
    assert _checkpoint_user_config(cfg) == _checkpoint_user_config(legacy)


def test_topology_migration_rejects_resolved_cooldown(hydra_compose) -> None:
    source = hydra_compose(CONFIG_DIR, "full3b_clustered_c2048_long_w31")
    destination = hydra_compose(CONFIG_DIR, "full3b_clustered_c2048_long_w32_bf16_b14")
    source_meta = {
        "user_config": _checkpoint_user_config(source),
        "conditioning_cooldown": _record(),
    }
    with pytest.raises(ValueError, match="requires conditioning_cooldown disabled"):
        _validate_config_contract(
            source_meta,
            destination,
            source_world_size=31,
            destination_world_size=32,
            step=20,
        )
