"""Independent-oracle tests for exact global supervised-token SFT loss."""

from __future__ import annotations

import multiprocessing
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import torch.distributed as dist
import torch.nn.functional as F

from steering.sft.hf_contract import HFDistributedImplementation
from steering.sft.hf_decoder import (
    build_hf_optimizer,
    distributed_loss_backward_scale,
    reduce_global_loss_mean,
    reduce_global_target_counts,
)


class _TinyTokenModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = t.nn.Embedding(17, 6)
        self.projection = t.nn.Linear(6, 7, bias=True)

    def forward(self, inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
        logits = self.projection(self.embedding(inputs))
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )


_RANK_BATCHES = {
    0: [
        (t.tensor([[1, 2, 3, 4]]), t.tensor([[1, -1, 2, 3]])),
        (t.tensor([[5, 6, 7]]), t.tensor([[-1, 4, 5]])),
    ],
    1: [
        (t.tensor([[8, 9]]), t.tensor([[2, -1]])),
        (t.tensor([[10, 11, 12, 13, 14]]), t.tensor([[6, 1, -1, -1, 0]])),
    ],
}


class _CompletedWork:
    def __init__(self, value: t.Tensor) -> None:
        self._future: t.futures.Future[t.Tensor] = t.futures.Future()
        self._future.set_result(value)

    def get_future(self) -> t.futures.Future[t.Tensor]:
        return self._future


def _flatten(model: t.nn.Module) -> list[float]:
    return t.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).tolist()


def _worker(
    rank: int,
    init_file: str,
    implementation_value: str,
    queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        t.manual_seed(12345)
        raw_model = _TinyTokenModel()
        implementation = HFDistributedImplementation(implementation_value)
        if implementation is HFDistributedImplementation.DDP_FUSED_ADAMW:
            training_model: t.nn.Module = t.nn.parallel.DistributedDataParallel(raw_model)
        else:
            training_model = raw_model
        optimizer = build_hf_optimizer(
            raw_model,
            ddp=True,
            learning_rate=3e-4,
            betas=(0.8, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            init_lr_frac=1.0,
            distributed_implementation=implementation,
        )
        local_supervised = sum(
            int((targets != -1).sum().item()) for _inputs, targets in _RANK_BATCHES[rank]
        )
        counts = reduce_global_target_counts(
            local_nonpadding=sum(targets.numel() for _, targets in _RANK_BATCHES[rank]),
            local_supervised=local_supervised,
            device="cpu",
        )
        scale = distributed_loss_backward_scale(counts.supervised, world_size=2)
        detached_sum = t.zeros(())
        for micro_step, (inputs, targets) in enumerate(_RANK_BATCHES[rank]):
            is_last = micro_step == len(_RANK_BATCHES[rank]) - 1
            no_sync = getattr(training_model, "no_sync", None)
            sync_context = no_sync() if not is_last and callable(no_sync) else nullcontext()
            with sync_context:
                loss_sum = training_model(inputs, targets)
                detached_sum.add_(loss_sum.detach())
                (loss_sum * scale).backward()
        global_mean = reduce_global_loss_mean(
            detached_sum, global_supervised=counts.supervised
        )

        if implementation is HFDistributedImplementation.DIST_MUON_ADAMW:
            real_all_reduce = dist.all_reduce

            def all_reduce_with_future(tensor, op=dist.ReduceOp.SUM, async_op=False):
                if not async_op:
                    return real_all_reduce(tensor, op=op, async_op=False)
                real_all_reduce(tensor, op=op, async_op=False)
                return _CompletedWork(tensor)

            with patch.object(dist, "all_reduce", all_reduce_with_future):
                optimizer.step()
        else:
            optimizer.step()
        queue.put(("ok", rank, counts.nonpadding, counts.supervised, global_mean, _flatten(raw_model)))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _oracle() -> tuple[int, int, float, list[float]]:
    t.manual_seed(12345)
    model = _TinyTokenModel()
    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    losses: list[t.Tensor] = []
    supervised = 0
    nonpadding = 0
    for rank in range(2):
        for inputs, targets in _RANK_BATCHES[rank]:
            losses.append(model(inputs, targets))
            supervised += int((targets != -1).sum().item())
            nonpadding += targets.numel()
    total = t.stack(losses).sum()
    mean = total / supervised
    mean.backward()
    optimizer.step()
    return nonpadding, supervised, float(mean.detach()), _flatten(model)


@pytest.mark.parametrize("implementation", list(HFDistributedImplementation))
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_two_rank_unequal_masks_match_one_concatenated_oracle(
    tmp_path: Path,
    implementation: HFDistributedImplementation,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / f"gloo-{implementation.value}")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, init_file, implementation.value, queue),
        )
        for rank in range(2)
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
    assert [process.exitcode for process in processes] == [0, 0]
    results.sort(key=lambda result: result[1])
    oracle_nonpadding, oracle_supervised, oracle_mean, oracle_parameters = _oracle()
    for result in results:
        assert result[2] == oracle_nonpadding
        assert result[3] == oracle_supervised
        assert result[4] == pytest.approx(oracle_mean, abs=1e-7, rel=1e-7)
        assert result[5] == pytest.approx(oracle_parameters, abs=2e-7, rel=2e-6)
    assert results[0][5] == pytest.approx(results[1][5], abs=1e-7, rel=1e-7)


@pytest.mark.parametrize(
    ("local_nonpadding", "local_supervised", "match"),
    [(-1, 0, "non-negative"), (1, -1, "non-negative"), (1, 2, "exceed")],
)
def test_target_count_domain_fails_before_collectives(
    local_nonpadding: int,
    local_supervised: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        reduce_global_target_counts(
            local_nonpadding=local_nonpadding,
            local_supervised=local_supervised,
            device="cpu",
        )


def test_zero_global_supervision_and_invalid_scales_fail_loud() -> None:
    with pytest.raises(RuntimeError, match="zero global supervised"):
        reduce_global_target_counts(
            local_nonpadding=10,
            local_supervised=0,
            device="cpu",
        )
    for supervised, world_size in [(0, 1), (1, 0), (-1, 1)]:
        with pytest.raises(ValueError):
            distributed_loss_backward_scale(supervised, world_size=world_size)


def test_single_rank_global_mean_preserves_sum_denominator() -> None:
    counts = reduce_global_target_counts(
        local_nonpadding=11,
        local_supervised=4,
        device="cpu",
    )
    assert counts.nonpadding == 11
    assert counts.supervised == 4
    assert distributed_loss_backward_scale(4, world_size=1) == 0.25
    assert reduce_global_loss_mean(t.tensor(10.0), global_supervised=4) == 2.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_global_loss_fails_before_optimizer_step(value: float) -> None:
    with pytest.raises(FloatingPointError, match="global supervised-token loss"):
        reduce_global_loss_mean(t.tensor(value), global_supervised=4)
