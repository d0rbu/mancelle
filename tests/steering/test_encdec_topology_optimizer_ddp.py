"""Gloo integration: migrated DistMuonAdamW state is update-equivalent."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import torch.distributed as dist

from nanochat.optim import DistMuonAdamW
from steering.tools.migrate_encdec_train_state import (
    GroupLayout,
    OptimizerLayout,
    ParameterLayout,
    _build_destination_optimizer_shard,
    _validate_optimizer_shards,
)

SOURCE_WORLD = 2
DESTINATION_WORLD = 3
TRAIN_STEPS = 2


class _CompletedWork:
    """Gloo-compatible stand-in for collectives lacking Work.get_future()."""

    def __init__(self, value: t.Tensor) -> None:
        self._future: t.futures.Future[t.Tensor] = t.futures.Future()
        self._future.set_result(value)

    def get_future(self) -> t.futures.Future[t.Tensor]:
        return self._future


def _parameters() -> list[t.nn.Parameter]:
    shapes = [
        (4, 4),
        (9, 128),
        (8, 128),
        (12, 128),
        *((2, 3),) * 4,
    ]
    return [
        t.nn.Parameter(
            (t.arange(shape[0] * shape[1], dtype=t.float32).reshape(shape) + index) / 256
        )
        for index, shape in enumerate(shapes)
    ]


def _param_groups(parameters: list[t.nn.Parameter]) -> list[dict]:
    return [
        {
            "kind": "adamw",
            "params": parameters[:4],
            "lr": 1 / 1024,
            "initial_lr": 1 / 1024,
            "betas": (0.5, 0.75),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
        {
            "kind": "muon",
            "params": parameters[4:],
            "lr": 1 / 1024,
            "initial_lr": 1 / 1024,
            "momentum": 0.5,
            "ns_steps": 2,
            "beta2": 0.75,
            "weight_decay": 0.0,
        },
    ]


def _layout() -> OptimizerLayout:
    shapes = [(4, 4), (9, 128), (8, 128), (12, 128), *((2, 3),) * 4]
    parameters = {
        param_id: ParameterLayout(
            param_id=param_id,
            name=f"parameter_{param_id}",
            shape=shape,
            dtype=t.float32,
            numel=shape[0] * shape[1],
            kind="adamw" if param_id < 4 else "muon",
            group_index=0 if param_id < 4 else 1,
        )
        for param_id, shape in enumerate(shapes)
    }
    return OptimizerLayout(
        parameters=parameters,
        groups=(
            GroupLayout(kind="adamw", param_ids=(0, 1, 2, 3)),
            GroupLayout(kind="muon", param_ids=(4, 5, 6, 7)),
        ),
    )


def _set_gradients(parameters: list[t.nn.Parameter], *, step: int) -> None:
    for index, parameter in enumerate(parameters):
        values = t.arange(parameter.numel(), dtype=t.float32).reshape(parameter.shape)
        parameter.grad = values.remainder(17).add_(1 + index + step).div_(256)


def _collective_patches():
    real_reduce_scatter = dist.reduce_scatter_tensor
    real_all_gather = dist.all_gather_into_tensor

    def reduce_scatter_with_future(output, input, *, op, async_op):
        assert async_op
        real_reduce_scatter(output, input, op=op, async_op=False)
        return _CompletedWork(output)

    def all_gather_with_future(output, input, *, async_op):
        assert async_op
        real_all_gather(output, input, async_op=False)
        return _CompletedWork(output)

    return (
        patch.object(dist, "reduce_scatter_tensor", reduce_scatter_with_future),
        patch.object(dist, "all_gather_into_tensor", all_gather_with_future),
    )


def _source_worker(rank: int, init_file: str, root: str, queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=SOURCE_WORLD,
        )
        parameters = _parameters()
        optimizer = DistMuonAdamW(_param_groups(parameters))
        reduce_patch, gather_patch = _collective_patches()
        with reduce_patch, gather_patch:
            for step in range(TRAIN_STEPS):
                _set_gradients(parameters, step=step)
                optimizer.step()
            t.save(optimizer.state_dict(), Path(root) / f"source_rank{rank}.pt")
            if rank == 0:
                t.save(
                    [parameter.detach().clone() for parameter in parameters],
                    Path(root) / "parameters_before_migration.pt",
                )
            dist.barrier()
            _set_gradients(parameters, step=TRAIN_STEPS)
            optimizer.step()
            if rank == 0:
                t.save(
                    [parameter.detach().clone() for parameter in parameters],
                    Path(root) / "reference_after_step.pt",
                )
        queue.put(("ok", rank))
    except BaseException as exc:  # noqa: BLE001 - report child failures
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _destination_worker(rank: int, init_file: str, root: str, queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=DESTINATION_WORLD,
        )
        parameters = _parameters()
        source_parameters = t.load(
            Path(root) / "parameters_before_migration.pt",
            map_location="cpu",
            weights_only=True,
        )
        for parameter, source in zip(parameters, source_parameters, strict=True):
            parameter.data.copy_(source)
        optimizer = DistMuonAdamW(_param_groups(parameters))
        optimizer.load_state_dict(
            t.load(
                Path(root) / f"destination_rank{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        reduce_patch, gather_patch = _collective_patches()
        with reduce_patch, gather_patch:
            _set_gradients(parameters, step=TRAIN_STEPS)
            optimizer.step()
        if rank == 0:
            t.save(
                [parameter.detach().clone() for parameter in parameters],
                Path(root) / "migrated_after_step.pt",
            )
        queue.put(("ok", rank))
    except BaseException as exc:  # noqa: BLE001 - report child failures
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_world(worker, *, world_size: int, init_file: Path, root: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(rank, str(init_file), str(root), queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=240) for _process in processes]
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    assert all(result[0] == "ok" for result in results), results
    assert [process.exitcode for process in processes] == [0] * world_size


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_migrated_state_matches_existing_optimizer_next_step(tmp_path: Path) -> None:
    _run_world(
        _source_worker,
        world_size=SOURCE_WORLD,
        init_file=tmp_path / "source-init",
        root=tmp_path,
    )
    source_shards = tuple(
        t.load(
            tmp_path / f"source_rank{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for rank in range(SOURCE_WORLD)
    )
    source_groups = source_shards[0]["param_groups"]
    _validate_optimizer_shards(
        source_shards,
        layout=_layout(),
        canonical_groups=source_groups,
        step=TRAIN_STEPS,
    )
    destination_shards = tuple(
        _build_destination_optimizer_shard(
            source_shards,
            layout=_layout(),
            target_param_groups=source_groups,
            destination_world_size=DESTINATION_WORLD,
            destination_rank=rank,
        )
        for rank in range(DESTINATION_WORLD)
    )
    _validate_optimizer_shards(
        destination_shards,
        layout=_layout(),
        canonical_groups=source_groups,
        step=TRAIN_STEPS,
    )
    for rank, shard in enumerate(destination_shards):
        t.save(shard, tmp_path / f"destination_rank{rank}.pt")

    _run_world(
        _destination_worker,
        world_size=DESTINATION_WORLD,
        init_file=tmp_path / "destination-init",
        root=tmp_path,
    )
    reference = t.load(
        tmp_path / "reference_after_step.pt",
        map_location="cpu",
        weights_only=True,
    )
    migrated = t.load(
        tmp_path / "migrated_after_step.pt",
        map_location="cpu",
        weights_only=True,
    )
    for index, (expected, actual) in enumerate(zip(reference, migrated, strict=True)):
        assert t.equal(actual, expected), index
