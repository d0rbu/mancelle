"""Exact logical optimizer-state resharding tests for topology migration."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch as t

from scripts.train.conditional_encdec_train import build_encdec_meta_model_from_training_cfg
from steering.tools.migrate_encdec_train_state import (
    GroupLayout,
    OptimizerLayout,
    ParameterLayout,
    _adam_is_replicated,
    _assert_optimizer_shard_exact,
    _build_destination_optimizer_shard,
    _build_optimizer_layout,
    _canonical_groups_at_checkpoint,
    _setup_optimizer,
    _validate_optimizer_shards,
)

TRAINING_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")


def _layout() -> OptimizerLayout:
    specs = {
        # replicated -> replicated (small)
        0: ("small", (4, 4)),
        # replicated at W2 -> sharded at W3
        1: ("rep_to_shard", (9, 128)),
        # sharded at W2 -> replicated at W3
        2: ("shard_to_rep", (8, 128)),
        # sharded at both worlds
        3: ("shard_to_shard", (12, 128)),
    }
    parameters = {
        param_id: ParameterLayout(
            param_id=param_id,
            name=name,
            shape=shape,
            dtype=t.float32,
            numel=math.prod(shape),
            kind="adamw",
            group_index=0,
        )
        for param_id, (name, shape) in specs.items()
    }
    for param_id in range(4, 9):
        parameters[param_id] = ParameterLayout(
            param_id=param_id,
            name=f"muon_{param_id}",
            shape=(2, 3),
            dtype=t.float32,
            numel=6,
            kind="muon",
            group_index=1,
        )
    return OptimizerLayout(
        parameters=parameters,
        groups=(
            GroupLayout(kind="adamw", param_ids=(0, 1, 2, 3)),
            GroupLayout(kind="muon", param_ids=(4, 5, 6, 7, 8)),
        ),
    )


def _groups() -> list[dict]:
    return [
        {
            "kind": "adamw",
            "params": [0, 1, 2, 3],
            "lr": 0.01,
            "initial_lr": 0.02,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
        {
            "kind": "muon",
            "params": [4, 5, 6, 7, 8],
            "lr": 0.03,
            "initial_lr": 0.04,
            "momentum": 0.95,
            "ns_steps": 5,
            "beta2": 0.95,
            "weight_decay": 0.1,
        },
    ]


def _logical_tensor(shape: tuple[int, ...], offset: int) -> t.Tensor:
    return t.arange(math.prod(shape), dtype=t.float32).reshape(shape) + offset


def _source_shards(world_size: int) -> tuple[dict, ...]:
    layout = _layout()
    shards = [
        {"state": {}, "param_groups": copy.deepcopy(_groups())} for _rank in range(world_size)
    ]
    for param_id in layout.groups[0].param_ids:
        parameter = layout.parameters[param_id]
        full_avg = _logical_tensor(parameter.shape, 100 * param_id)
        full_sq = _logical_tensor(parameter.shape, 100 * param_id + 50)
        replicated = _adam_is_replicated(parameter, world_size)
        for rank, shard in enumerate(shards):
            if replicated:
                avg = full_avg.clone()
                sq = full_sq.clone()
            else:
                rows = parameter.shape[0] // world_size
                avg = full_avg[rank * rows : (rank + 1) * rows].clone()
                sq = full_sq[rank * rows : (rank + 1) * rows].clone()
            shard["state"][param_id] = {
                "step": 7,
                "exp_avg": avg,
                "exp_avg_sq": sq,
            }

    muon_group = layout.groups[1]
    first_id = muon_group.param_ids[0]
    chunk = math.ceil(len(muon_group.param_ids) / world_size)
    logical_momentum = _logical_tensor((len(muon_group.param_ids), 2, 3), 1_000)
    logical_second = _logical_tensor((len(muon_group.param_ids), 1, 3), 2_000)
    for rank, shard in enumerate(shards):
        start = rank * chunk
        owned = min(chunk, max(0, len(muon_group.param_ids) - start))
        momentum = t.zeros(chunk, 2, 3)
        second = t.zeros(chunk, 1, 3)
        if owned:
            momentum[:owned].copy_(logical_momentum[start : start + owned])
            second[:owned].copy_(logical_second[start : start + owned])
        shard["state"][first_id] = {
            "momentum_buffer": momentum,
            "second_momentum_buffer": second,
        }
    return tuple(shards)


def _reshard(shards: tuple[dict, ...], destination_world: int) -> tuple[dict, ...]:
    return tuple(
        _build_destination_optimizer_shard(
            shards,
            layout=_layout(),
            target_param_groups=_groups(),
            destination_world_size=destination_world,
            destination_rank=rank,
        )
        for rank in range(destination_world)
    )


def test_all_adam_ownership_transitions_and_muon_padding_validate() -> None:
    source = _source_shards(2)
    _validate_optimizer_shards(
        source,
        layout=_layout(),
        canonical_groups=_groups(),
        step=7,
    )
    target = _reshard(source, 3)
    _validate_optimizer_shards(
        target,
        layout=_layout(),
        canonical_groups=_groups(),
        step=7,
    )

    assert target[0]["state"][0]["exp_avg"].shape == (4, 4)
    assert target[0]["state"][1]["exp_avg"].shape == (3, 128)
    assert target[0]["state"][2]["exp_avg"].shape == (8, 128)
    assert target[0]["state"][3]["exp_avg"].shape == (4, 128)
    assert target[2]["state"][4]["momentum_buffer"].shape == (2, 2, 3)
    assert not bool(target[2]["state"][4]["momentum_buffer"][1:].count_nonzero())


def test_two_to_three_to_two_roundtrip_is_tensor_exact() -> None:
    source = _source_shards(2)
    roundtrip = _reshard(_reshard(source, 3), 2)
    _validate_optimizer_shards(
        roundtrip,
        layout=_layout(),
        canonical_groups=_groups(),
        step=7,
    )
    for original, restored in zip(source, roundtrip, strict=True):
        assert original["param_groups"] == restored["param_groups"]
        assert set(original["state"]) == set(restored["state"])
        for param_id, original_state in original["state"].items():
            restored_state = restored["state"][param_id]
            assert set(original_state) == set(restored_state)
            for field, expected in original_state.items():
                actual = restored_state[field]
                if isinstance(expected, t.Tensor):
                    assert t.equal(actual, expected), (param_id, field)
                else:
                    assert actual == expected


def test_empty_step_zero_state_migrates_without_manufacturing_buffers() -> None:
    source = tuple({"state": {}, "param_groups": copy.deepcopy(_groups())} for _rank in range(2))
    _validate_optimizer_shards(
        source,
        layout=_layout(),
        canonical_groups=_groups(),
        step=0,
    )
    target = _reshard(source, 3)
    assert all(shard["state"] == {} for shard in target)


def test_replicated_adam_mismatch_fails_loud() -> None:
    source = list(_source_shards(2))
    source[1]["state"][1]["exp_avg"][0, 0] += 1
    with pytest.raises(ValueError, match="replicated Adam state.*differs"):
        _validate_optimizer_shards(
            tuple(source),
            layout=_layout(),
            canonical_groups=_groups(),
            step=7,
        )


def test_nonfinite_state_and_dirty_muon_padding_fail_loud() -> None:
    nonfinite = list(_source_shards(2))
    nonfinite[0]["state"][2]["exp_avg"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _validate_optimizer_shards(
            tuple(nonfinite),
            layout=_layout(),
            canonical_groups=_groups(),
            step=7,
        )

    dirty_padding = list(_source_shards(2))
    dirty_padding[1]["state"][4]["momentum_buffer"][2, 0, 0] = 1
    with pytest.raises(ValueError, match="nonzero source padding"):
        _validate_optimizer_shards(
            tuple(dirty_padding),
            layout=_layout(),
            canonical_groups=_groups(),
            step=7,
        )


def test_param_group_drift_fails_before_resharding() -> None:
    source = list(_source_shards(2))
    source[0]["param_groups"][0]["initial_lr"] *= 2
    with pytest.raises(ValueError, match="param_groups are not canonical"):
        _validate_optimizer_shards(
            tuple(source),
            layout=_layout(),
            canonical_groups=_groups(),
            step=7,
        )


def test_adam_counter_must_equal_raw_checkpoint_step() -> None:
    source = list(_source_shards(2))
    for shard in source:
        shard["state"][0]["step"] = 6
    with pytest.raises(ValueError, match="does not match checkpoint step 7"):
        _validate_optimizer_shards(
            tuple(source),
            layout=_layout(),
            canonical_groups=_groups(),
            step=7,
        )


def test_serialized_optimizer_exactness_detects_value_change() -> None:
    expected = _source_shards(2)[0]
    actual = copy.deepcopy(expected)
    actual["state"][2]["exp_avg"][0, 0] += 1

    with pytest.raises(ValueError, match="changed during serialization"):
        _assert_optimizer_shard_exact(
            actual,
            expected,
            context="optimizer rank 0",
        )


def test_canonical_w31_w32_production_optimizer_layout_and_scaling(
    hydra_compose,
) -> None:
    """Pin the real 3B architecture/group layout used by the migration."""

    source_cfg = hydra_compose(
        TRAINING_CONFIG_DIR,
        "full3b_clustered_c2048_long_w31",
    )
    destination_cfg = hydra_compose(
        TRAINING_CONFIG_DIR,
        "full3b_clustered_c2048_long_w32_bf16_b14",
    )
    source_model = build_encdec_meta_model_from_training_cfg(
        source_cfg,
        base_vocab=32_768,
        n_sae_features=65_536,
    )
    destination_model = build_encdec_meta_model_from_training_cfg(
        destination_cfg,
        base_vocab=32_768,
        n_sae_features=65_536,
    )
    model_state = source_model.state_dict()
    source_optimizer = _setup_optimizer(
        source_model,
        source_cfg,
        total_batch_size=int(source_cfg.total_batch_size),
        base_vocab=32_768,
        n_sae_features=65_536,
        feature_vocab=None,
    )
    destination_optimizer = _setup_optimizer(
        destination_model,
        destination_cfg,
        total_batch_size=int(destination_cfg.total_batch_size),
        base_vocab=32_768,
        n_sae_features=65_536,
        feature_vocab=None,
    )
    source_layout = _build_optimizer_layout(source_model, model_state, source_optimizer)
    destination_layout = _build_optimizer_layout(
        destination_model,
        model_state,
        destination_optimizer,
    )

    assert source_layout == destination_layout
    assert len(source_layout.parameters) == 359
    assert len(source_layout.groups) == 12
    muon_counts = [
        len(group.param_ids) for group in source_layout.groups if group.kind == "muon"
    ]
    assert muon_counts == [28, 108, 54, 54, 54, 27]
    assert all(math.ceil(count / 31) == math.ceil(count / 32) for count in muon_counts)

    source_groups = _canonical_groups_at_checkpoint(
        source_optimizer,
        source_cfg,
        step=125_000,
    )
    destination_groups = _canonical_groups_at_checkpoint(
        destination_optimizer,
        destination_cfg,
        step=125_000,
    )
    expected_ratio = math.sqrt(32 / 31)
    assert len(source_groups) == len(destination_groups) == 12
    for source_group, destination_group in zip(
        source_groups,
        destination_groups,
        strict=True,
    ):
        assert destination_group["initial_lr"] / source_group["initial_lr"] == pytest.approx(
            expected_ratio,
        )
