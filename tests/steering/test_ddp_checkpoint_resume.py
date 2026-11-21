"""Checkpoint-exec DDP initialization unit and two-CUDA regressions."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
import torch.distributed as dist

import nanochat.common as common
import steering._common.ddp as resume_ddp
from nanochat.common import compute_cleanup
from scripts.train.conditional_encdec_sft_grpo import _reexec_from_checkpoint
from tests.torchrun_helpers import run_torchrun


@pytest.mark.parametrize(
    ("device_type", "resume_step", "ddp_requested"),
    (("cuda", -1, True), ("cuda", 1, False), ("cpu", 1, True)),
)
def test_checkpoint_resume_init_delegates_other_cases(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    resume_step: int,
    ddp_requested: bool,
) -> None:
    expected = (ddp_requested, 3, 4, 5, t.device(device_type))
    calls = []
    monkeypatch.setattr(resume_ddp, "is_ddp_requested", lambda: ddp_requested)
    monkeypatch.setattr(
        resume_ddp,
        "compute_init",
        lambda value: calls.append(value) or expected,
    )
    assert (
        resume_ddp.compute_init_for_checkpoint_resume(
            device_type,
            resume_step=resume_step,
        )
        == expected
    )
    assert calls == [device_type]


@pytest.fixture
def resume_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    events: list[object] = []
    prefixes: list[str] = []
    init_calls: list[dict[str, object]] = []

    class Store:
        def set_timeout(self, value: timedelta) -> None:
            events.append(("set_timeout", value))

    raw_store = Store()
    monkeypatch.setattr(resume_ddp, "is_ddp_requested", lambda: True)
    monkeypatch.setattr(
        resume_ddp.t.cuda,
        "is_available",
        lambda: events.append("cuda_available") or True,
    )
    monkeypatch.setattr(resume_ddp.t, "manual_seed", lambda value: events.append(("seed", value)))
    monkeypatch.setattr(
        resume_ddp.t.cuda,
        "manual_seed",
        lambda value: events.append(("cuda_seed", value)),
    )
    monkeypatch.setattr(
        resume_ddp.t,
        "set_float32_matmul_precision",
        lambda value: events.append(("precision", value)),
    )
    def get_dist_info() -> tuple[bool, int, int, int]:
        events.append("dist_info")
        return True, 0, 1, 2

    monkeypatch.setattr(resume_ddp, "get_dist_info", get_dist_info)
    monkeypatch.setattr(common, "get_dist_info", get_dist_info)
    monkeypatch.setattr(
        resume_ddp.t.cuda,
        "set_device",
        lambda value: events.append(("set_device", value)),
    )

    def rendezvous(url: str, *, timeout: timedelta):
        events.append(("rendezvous", url, timeout))
        return iter(((raw_store, 0, 2),))

    def prefix_store(prefix: str, store: object) -> tuple[str, str, object]:
        events.append(("prefix", prefix))
        prefixes.append(prefix)
        return ("prefix", prefix, store)

    def init_process_group(**kwargs: object) -> None:
        events.append("init")
        init_calls.append(kwargs)

    monkeypatch.setattr(resume_ddp.dist, "rendezvous", rendezvous)
    monkeypatch.setattr(resume_ddp.dist, "PrefixStore", prefix_store)
    monkeypatch.setattr(resume_ddp.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(resume_ddp.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(resume_ddp.logger, "info", lambda value: events.append(("log", value)))
    return SimpleNamespace(events=events, prefixes=prefixes, init_calls=init_calls, raw_store=raw_store)


@pytest.mark.parametrize("timeout_seconds", (None, 19))
def test_checkpoint_resume_init_preserves_cuda_setup_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    resume_runtime: SimpleNamespace,
    timeout_seconds: int | None,
) -> None:
    if timeout_seconds is None:
        monkeypatch.delenv("NCCL_TIMEOUT_SEC", raising=False)
        expected_timeout = resume_ddp.default_pg_nccl_timeout
    else:
        monkeypatch.setenv("NCCL_TIMEOUT_SEC", str(timeout_seconds))
        expected_timeout = timedelta(seconds=timeout_seconds)
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "4")

    vendor_result = common.compute_init("cuda")
    vendor_events = list(resume_runtime.events)
    vendor_init = resume_runtime.init_calls.pop()
    resume_runtime.events.clear()
    result = resume_ddp.compute_init_for_checkpoint_resume("cuda", resume_step=7)

    assert result == vendor_result == (True, 0, 1, 2, t.device("cuda", 1))
    init_index = vendor_events.index("init")
    assert resume_runtime.events == [
        *vendor_events[:init_index],
        ("rendezvous", "env://", expected_timeout),
        ("set_timeout", expected_timeout),
        ("prefix", "checkpoint_resume_step_7_restart_4"),
        *vendor_events[init_index:],
    ]
    assert resume_runtime.prefixes == ["checkpoint_resume_step_7_restart_4"]
    adapter_init = resume_runtime.init_calls.pop()
    assert vendor_init == {
        "backend": "nccl",
        "device_id": t.device("cuda", 1),
        **({} if timeout_seconds is None else {"timeout": expected_timeout}),
    }
    assert {key: adapter_init.pop(key) for key in vendor_init} == vendor_init
    assert adapter_init == {
        "store": ("prefix", "checkpoint_resume_step_7_restart_4", resume_runtime.raw_store),
        "rank": 0,
        "world_size": 2,
    }
    assert vendor_events == [
        "cuda_available",
        ("seed", 42),
        ("cuda_seed", 42),
        ("precision", "high"),
        "dist_info",
        ("set_device", t.device("cuda", 1)),
        "init",
        "barrier",
        ("log", "Distributed world size: 2"),
    ]


def test_checkpoint_resume_generations_use_distinct_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    resume_runtime: SimpleNamespace,
) -> None:
    monkeypatch.delenv("NCCL_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    for step in (1, 2):
        resume_ddp.compute_init_for_checkpoint_resume("cuda", resume_step=step)
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    resume_ddp.compute_init_for_checkpoint_resume("cuda", resume_step=2)
    assert resume_runtime.prefixes == [
        "checkpoint_resume_step_1_restart_0",
        "checkpoint_resume_step_2_restart_0",
        "checkpoint_resume_step_2_restart_1",
    ]


def _worker(output: Path) -> None:
    resume_step = next(
        (int(value.removeprefix("resume_from_step=")) for value in sys.argv if value.startswith("resume_from_step=")),
        -1,
    )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    generation = max(0, resume_step)
    if resume_step > 0 and rank == 0:
        time.sleep(3)
    _, actual_rank, _, actual_world_size, device = resume_ddp.compute_init_for_checkpoint_resume(
        "cuda",
        resume_step=resume_step,
    )
    assert (actual_rank, actual_world_size) == (rank, world_size)
    value = t.tensor(float((generation + 1) * (rank + 1)), device=device)
    dist.all_reduce(value)
    expected = float((generation + 1) * sum(range(1, world_size + 1)))
    assert value.item() == expected
    receipt = {
        "generation": generation,
        "rank": rank,
        "pid": os.getpid(),
        "all_reduce": value.item(),
        "expected": expected,
    }
    (output / f"rank{rank}-generation{generation}.json").write_text(
        json.dumps(receipt, sort_keys=True)
    )
    if generation < 2:
        _reexec_from_checkpoint(generation + 1)
    compute_cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("world_size", (1, 2))
def test_checkpoint_resume_real_cuda_generations(tmp_path: Path, world_size: int) -> None:
    if t.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA device(s)")
    repo = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["NCCL_TIMEOUT_SEC"] = "30"
    environment["PYTHONPATH"] = f"{repo}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    # Three cold Python/CUDA starts on a shared filesystem can exceed two
    # minutes. Each NCCL process-group timeout remains 30 seconds.
    result = run_torchrun(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world_size}",
            str(Path(__file__).resolve()),
            "--worker",
            str(tmp_path),
        ],
        cwd=repo,
        env=environment,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    receipts = [
        json.loads((tmp_path / f"rank{rank}-generation{generation}.json").read_text())
        for rank in range(world_size)
        for generation in range(3)
    ]
    assert len(receipts) == world_size * 3
    for rank in range(world_size):
        per_rank = [receipt for receipt in receipts if receipt["rank"] == rank]
        assert len({receipt["pid"] for receipt in per_rank}) == 1
        assert [receipt["generation"] for receipt in per_rank] == [0, 1, 2]
        assert all(receipt["all_reduce"] == receipt["expected"] for receipt in per_rank)


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    _worker(Path(sys.argv[2]))
