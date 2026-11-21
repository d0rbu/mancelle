"""Per-rank encoder-decoder dataloader checkpoint-state envelope tests."""

from __future__ import annotations

import json
import socket

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    build_dataloader_state_envelope,
    select_dataloader_rank_state,
)


def test_single_process_envelope_never_calls_collective(monkeypatch) -> None:
    monkeypatch.setattr(t.distributed, "is_initialized", lambda: False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("single-process save called all_gather_object")

    monkeypatch.setattr(t.distributed, "all_gather_object", forbidden)
    local = {
        "ddp_world_size": 1,
        "batch_idx": 7,
        "batch_rows": [[1, "shard.safetensors.zst", 3]],
    }

    envelope = build_dataloader_state_envelope(
        local,
        ddp_rank=0,
        ddp_world_size=1,
    )

    assert envelope == {"ddp_world_size": 1, "rank_states": [local]}
    assert select_dataloader_rank_state(
        json.loads(json.dumps(envelope)),
        ddp_rank=0,
        ddp_world_size=1,
    ) == local


def test_legacy_flat_state_defaults_missing_topology_to_current() -> None:
    legacy = {"batch_idx": 2, "shard": "shard.safetensors.zst", "epoch": 1}
    assert select_dataloader_rank_state(
        legacy,
        ddp_rank=3,
        ddp_world_size=8,
    ) == legacy


def test_envelope_rejects_topology_change_before_rank_selection() -> None:
    envelope = {
        "ddp_world_size": 8,
        "rank_states": [{"ddp_world_size": 8} for _ in range(8)],
    }
    with pytest.raises(ValueError, match=r"world.size.*8.*4|8.*4"):
        select_dataloader_rank_state(
            envelope,
            ddp_rank=0,
            ddp_world_size=4,
        )


def test_multi_rank_save_requires_initialized_process_group(monkeypatch) -> None:
    monkeypatch.setattr(t.distributed, "is_initialized", lambda: False)
    with pytest.raises(ValueError, match="initialized"):
        build_dataloader_state_envelope(
            {"ddp_world_size": 2},
            ddp_rank=0,
            ddp_world_size=2,
        )


def _gloo_envelope_target(rank: int, world_size: int) -> dict:
    local = {
        "ddp_world_size": world_size,
        "pq_idx": 10 + rank,
        "rg_idx": 20 + rank,
        "feature_pipeline_rng_state": [
            {
                "roles": ["selection"],
                "device_type": "cpu",
                "device_index": None,
                "state_hex": f"0{rank}",
            }
        ],
    }
    envelope = build_dataloader_state_envelope(
        local,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    envelope = json.loads(json.dumps(envelope))
    selected = select_dataloader_rank_state(
        envelope,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    rejected = False
    try:
        select_dataloader_rank_state(
            envelope,
            ddp_rank=rank,
            ddp_world_size=world_size + 1,
        )
    except ValueError:
        rejected = True
    return {
        "envelope": envelope,
        "selected": selected,
        "rejected": rejected,
    }


def test_two_rank_gloo_gathers_and_restores_rank_local_state(
    gloo_multiprocess,
) -> None:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pytest.skip("sandbox forbids localhost sockets required by CPU/Gloo")
    else:
        probe.close()

    results = gloo_multiprocess(2, _gloo_envelope_target)

    assert results[0]["envelope"] == results[1]["envelope"]
    assert [result["selected"]["pq_idx"] for result in results] == [10, 11]
    assert [result["selected"]["rg_idx"] for result in results] == [20, 21]
    assert all(result["rejected"] for result in results)
