"""Replay-only dataloader topology-migration contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.topology_resume import (
    PackedRowInventory,
    PackedShardInfo,
    RowReference,
    build_replay_boundary_envelope,
)
from steering.packed_batch_io import save_packed_row_shard
from tests.encdec_test_helpers import N_SAE_FEATURES, _make_test_shard, build_threshold_pipeline


def _inventory(rows: list[int], *, context_len: int = 8) -> PackedRowInventory:
    return PackedRowInventory(
        [
            PackedShardInfo(
                name=f"shard_{index:05d}.safetensors.zst",
                rows=count,
                context_len=context_len,
                compressed_bytes=100 + index,
            )
            for index, count in enumerate(rows)
        ]
    )


def _empty_rng_state() -> list[dict]:
    return []


def _rng_state_hex(seed: int) -> str:
    state = t.Generator(device="cpu").manual_seed(seed).get_state()
    return bytes(state.tolist()).hex()


def _source_envelope(
    inventory: PackedRowInventory,
    *,
    world_size: int,
    batch_size: int,
    consumed_batches: int,
) -> tuple[dict, list[set[RowReference]]]:
    boundary = RowReference(1, 0, 0)
    rank_states: list[dict] = []
    consumed: list[set[RowReference]] = []
    for rank in range(world_size):
        first = inventory.first_owned_at_or_after(
            boundary,
            rank=rank,
            world_size=world_size,
        )
        stream = inventory.consecutive_owned(
            first,
            rank=rank,
            world_size=world_size,
            count=consumed_batches * batch_size + batch_size,
        )
        rank_consumed = set(stream[: consumed_batches * batch_size])
        pending = stream[consumed_batches * batch_size :]
        last = pending[-1]
        rank_states.append(
            {
                "batch_idx": consumed_batches,
                "shard": inventory.shards[last.shard_index].name,
                "shard_batch_offset": consumed_batches,
                "epoch": last.epoch,
                "max_seg_size": 128,
                "pad_F": 16,
                "ddp_world_size": world_size,
                "batch_rows": [list(inventory.to_tuple(row)) for row in pending],
                "feature_pipeline_rng_state": _empty_rng_state(),
            }
        )
        consumed.append(rank_consumed)
    return {"ddp_world_size": world_size, "rank_states": rank_states}, consumed


def _all_references_before(
    inventory: PackedRowInventory,
    boundary: RowReference,
) -> set[RowReference]:
    rows: set[RowReference] = set()
    for epoch in range(1, boundary.epoch + 1):
        for shard_index, shard in enumerate(inventory.shards):
            for row_index in range(shard.rows):
                reference = RowReference(epoch, shard_index, row_index)
                if inventory.ordinal(reference) >= inventory.ordinal(boundary):
                    return rows
                rows.add(reference)
    return rows


def test_counterexample_replays_consumed_hole_without_skipping() -> None:
    inventory = _inventory([5, 5])
    source = {
        "ddp_world_size": 3,
        "rank_states": [
            {
                "batch_idx": 2,
                "shard": "shard_00001.safetensors.zst",
                "shard_batch_offset": 0,
                "epoch": 1,
                "max_seg_size": 128,
                "pad_F": 16,
                "ddp_world_size": 3,
                "batch_rows": [[1, "shard_00001.safetensors.zst", 0]],
                "feature_pipeline_rng_state": [],
            },
            {
                "batch_idx": 2,
                "shard": "shard_00001.safetensors.zst",
                "shard_batch_offset": 0,
                "epoch": 1,
                "max_seg_size": 128,
                "pad_F": 16,
                "ddp_world_size": 3,
                "batch_rows": [[1, "shard_00001.safetensors.zst", 1]],
                "feature_pipeline_rng_state": [],
            },
            {
                "batch_idx": 2,
                "shard": "shard_00000.safetensors.zst",
                "shard_batch_offset": 0,
                "epoch": 2,
                "max_seg_size": 128,
                "pad_F": 16,
                "ddp_world_size": 3,
                "batch_rows": [[2, "shard_00000.safetensors.zst", 2]],
                "feature_pipeline_rng_state": [],
            },
        ],
    }

    target, plan = build_replay_boundary_envelope(
        source,
        inventory=inventory,
        source_world_size=3,
        destination_world_size=4,
        device_batch_size=1,
        seed=42,
    )

    assert plan.boundary == (1, "shard_00001.safetensors.zst", 0)
    assert plan.replay_rows == 1
    assert plan.replay_rows_by_source_rank == (0, 0, 1)
    assert plan.replay_tokens == 8
    assert [state["batch_rows"] for state in target["rank_states"]] == [
        [[1, "shard_00001.safetensors.zst", 0]],
        [[1, "shard_00001.safetensors.zst", 1]],
        [[1, "shard_00001.safetensors.zst", 2]],
        [[1, "shard_00001.safetensors.zst", 3]],
    ]


@given(
    shard_rows=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=5),
    source_world=st.integers(min_value=1, max_value=5),
    destination_world=st.integers(min_value=1, max_value=5),
    batch_size=st.integers(min_value=1, max_value=4),
    consumed_batches=st.integers(min_value=0, max_value=12),
)
@settings(max_examples=120, deadline=None)
def test_replay_boundary_property_has_no_skips_and_exact_replay_count(
    shard_rows,
    source_world,
    destination_world,
    batch_size,
    consumed_batches,
) -> None:
    # Every rank must own at least one row in the cyclic inventory.
    if max(shard_rows) < max(source_world, destination_world):
        return
    inventory = _inventory(shard_rows)
    source, consumed_by_rank = _source_envelope(
        inventory,
        world_size=source_world,
        batch_size=batch_size,
        consumed_batches=consumed_batches,
    )
    target, plan = build_replay_boundary_envelope(
        source,
        inventory=inventory,
        source_world_size=source_world,
        destination_world_size=destination_world,
        device_batch_size=batch_size,
        seed=17,
    )

    boundary = inventory.parse(plan.boundary, context="plan boundary")
    all_consumed = set().union(*consumed_by_rank)
    assert _all_references_before(inventory, boundary) <= all_consumed
    expected_replay = {
        row for row in all_consumed if inventory.ordinal(row) >= inventory.ordinal(boundary)
    }
    assert plan.replay_rows == len(expected_replay)
    assert plan.replay_tokens == len(expected_replay) * inventory.context_len
    assert target["ddp_world_size"] == destination_world
    for rank, state in enumerate(target["rank_states"]):
        parsed = [
            inventory.parse(raw, context=f"target rank {rank}") for raw in state["batch_rows"]
        ]
        assert parsed == inventory.consecutive_owned(
            parsed[0],
            rank=rank,
            world_size=destination_world,
            count=batch_size,
        )


def test_synthesized_target_state_is_accepted_by_real_loader(tmp_path: Path) -> None:
    packed = tmp_path / "packed"
    packed.mkdir()
    save_packed_row_shard(_make_test_shard(n_rows=5, seed=1), packed / "shard_00000")
    save_packed_row_shard(_make_test_shard(n_rows=5, seed=2), packed / "shard_00001")
    (packed / "manifest.json").write_text(json.dumps({"metadata": {}}))
    inventory = PackedRowInventory.from_directory(packed)
    source, _consumed = _source_envelope(
        inventory,
        world_size=2,
        batch_size=2,
        consumed_batches=2,
    )
    target, _plan = build_replay_boundary_envelope(
        source,
        inventory=inventory,
        source_world_size=2,
        destination_world_size=3,
        device_batch_size=2,
        seed=42,
    )

    boundaries = t.linspace(0.25, 0.75, 3, dtype=t.float64).repeat(N_SAE_FEATURES, 1)
    pipeline = build_threshold_pipeline(boundaries)
    for rank, state in enumerate(target["rank_states"]):
        loader = precomputed_data_loader(
            packed_row_dir=packed,
            B=2,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
            ddp_rank=rank,
            ddp_world_size=3,
            resume_state_dict=json.loads(json.dumps(state)),
        )
        replay = next(loader)
        assert replay[9]["batch_rows"] == state["batch_rows"]


def test_destination_feature_rng_is_fresh_rank_seeded_and_role_exact() -> None:
    inventory = _inventory([8])
    source, _consumed = _source_envelope(
        inventory,
        world_size=2,
        batch_size=1,
        consumed_batches=1,
    )
    for rank, state in enumerate(source["rank_states"]):
        state["feature_pipeline_rng_state"] = [
            {
                "roles": ["selection", "negative"],
                "device_type": "cpu",
                "device_index": None,
                "state_hex": _rng_state_hex(100 + rank),
            }
        ]

    target, _plan = build_replay_boundary_envelope(
        source,
        inventory=inventory,
        source_world_size=2,
        destination_world_size=3,
        device_batch_size=1,
        seed=17,
    )

    for rank, state in enumerate(target["rank_states"]):
        assert state["feature_pipeline_rng_state"] == [
            {
                "roles": ["selection", "negative"],
                "device_type": "cpu",
                "device_index": None,
                "state_hex": _rng_state_hex((17 * 1_000_003 + rank) % (2**63)),
            }
        ]


def test_duplicate_or_noncanonical_feature_rng_roles_fail_loud() -> None:
    inventory = _inventory([8])
    source, _consumed = _source_envelope(
        inventory,
        world_size=1,
        batch_size=1,
        consumed_batches=1,
    )
    source["rank_states"][0]["feature_pipeline_rng_state"] = [
        {
            "roles": ["negative", "selection"],
            "device_type": "cpu",
            "device_index": None,
            "state_hex": _rng_state_hex(1),
        }
    ]

    with pytest.raises(ValueError, match="roles must be unique and canonical"):
        build_replay_boundary_envelope(
            source,
            inventory=inventory,
            source_world_size=1,
            destination_world_size=2,
            device_batch_size=1,
            seed=17,
        )
