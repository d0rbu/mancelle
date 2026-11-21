"""Two-rank AdamW-only distributed optimizer coverage for the HF backend."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import torch.distributed as dist

from nanochat.optim import DistMuonAdamW, MuonAdamW
from steering.sft.clipping import apply_gradient_clipping


class _CompletedWork:
    """Gloo-compatible stand-in for collectives lacking Work.get_future()."""

    def __init__(self, value: t.Tensor) -> None:
        self._future: t.futures.Future[t.Tensor] = t.futures.Future()
        self._future.set_result(value)

    def get_future(self) -> t.futures.Future[t.Tensor]:
        return self._future


def _worker(rank: int, world_size: int, init_file: str, queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        # Both exceed DistMuonAdamW's small-parameter cutoff. The first takes
        # the reduce-scatter path; the odd first dimension of the second pins
        # the documented all-reduce fallback.
        divisible = t.nn.Parameter(t.arange(2048, dtype=t.float32).reshape(1024, 2))
        indivisible = t.nn.Parameter(t.arange(2050, dtype=t.float32).reshape(1025, 2))
        parameters = [divisible, indivisible]
        for parameter in parameters:
            parameter.grad = t.full_like(parameter, 10.0 if rank == 0 else 1.0)
        pre_clip = apply_gradient_clipping(parameters, 1.0)
        assert pre_clip is not None
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

        # Gloo implements both tensor collectives but its Work object does not
        # expose get_future(). Preserve the real collective and adapt only that
        # completion interface so the production optimizer's async path runs.
        with (
            patch.object(dist, "reduce_scatter_tensor", reduce_scatter_with_future),
            patch.object(dist, "all_gather_into_tensor", all_gather_with_future),
        ):
            optimizer = DistMuonAdamW(
                [
                    {
                        "kind": "adamw",
                        "params": parameters,
                        "lr": 2e-5,
                        "betas": (0.8, 0.95),
                        "eps": 1e-10,
                        "weight_decay": 0.0,
                    }
                ]
            )
            optimizer.step()
        queue.put(
            (
                "ok",
                rank,
                float(pre_clip.item()),
                divisible.detach().tolist(),
                indivisible.detach().tolist(),
            )
        )
    except BaseException as exc:  # noqa: BLE001 - report child failures
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_two_rank_adamw_group_syncs_sharded_and_indivisible_parameters(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(target=_worker, args=(rank, 2, init_file, queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=120) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    assert all(result[0] == "ok" for result in results), results
    by_rank = {result[1]: result for result in results}
    assert by_rank[0][3:] == by_rank[1][3:]
    # Independently reconstruct the documented clip-then-average contract.
    # This is stronger than observing rank equality: averaging raw gradients
    # before clipping would also leave the ranks equal, but would produce a
    # different update whenever local norms differ.
    clipped_by_rank: list[list[t.Tensor]] = []
    reference_norms: list[float] = []
    for raw_gradient in (10.0, 1.0):
        local_parameters = [
            t.nn.Parameter(t.arange(2048, dtype=t.float32).reshape(1024, 2)),
            t.nn.Parameter(t.arange(2050, dtype=t.float32).reshape(1025, 2)),
        ]
        for parameter in local_parameters:
            parameter.grad = t.full_like(parameter, raw_gradient)
        local_norm = apply_gradient_clipping(local_parameters, 1.0)
        assert local_norm is not None
        reference_norms.append(float(local_norm.item()))
        clipped_gradients: list[t.Tensor] = []
        for parameter in local_parameters:
            assert parameter.grad is not None
            clipped_gradients.append(parameter.grad.detach().clone())
        clipped_by_rank.append(clipped_gradients)

    serial_parameters = [
        t.nn.Parameter(t.arange(2048, dtype=t.float32).reshape(1024, 2)),
        t.nn.Parameter(t.arange(2050, dtype=t.float32).reshape(1025, 2)),
    ]
    for parameter_index, parameter in enumerate(serial_parameters):
        parameter.grad = (
            clipped_by_rank[0][parameter_index] + clipped_by_rank[1][parameter_index]
        ) / 2
    serial_optimizer = MuonAdamW(
        [
            {
                "kind": "adamw",
                "params": serial_parameters,
                "lr": 2e-5,
                "betas": (0.8, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            }
        ]
    )
    serial_optimizer.step()

    assert by_rank[0][2] > by_rank[1][2]
    assert [by_rank[rank][2] for rank in (0, 1)] == reference_norms
    assert by_rank[0][3] == serial_parameters[0].detach().tolist()
    assert by_rank[0][4] == serial_parameters[1].detach().tolist()
    assert [process.exitcode for process in processes] == [0, 0]
