"""Trainer ownership of issue #1569 storage-observability gates."""

import ast
import inspect
import multiprocessing
import textwrap
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import torch.distributed as dist
from loguru import logger

import scripts.train.conditional_encdec_train as train_module
from scripts.train.conditional_encdec_train import (
    ValidationEvalSpec,
    _log_loader_health_gate,
    _remote_checkpoint_save_rank_skew,
    _validation_loader_health,
    train_one_config,
)
from steering.conditional.encoder_decoder.dataloader import (
    _loader_health_record,
    _loader_health_reset,
    loader_health_snapshot,
)


@pytest.fixture(autouse=True)
def _isolated_loader_health():
    _loader_health_reset()
    yield
    _loader_health_reset()


def _capture() -> tuple[StringIO, int]:
    sink = StringIO()
    handler_id = logger.add(sink, format="{level} | {message}", level="INFO")
    return sink, handler_id


def _remote_save_skew_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    failing_rank: int | None,
    result_queue: Any,
) -> None:
    sink = StringIO()
    handler_id = logger.add(sink, format="{level} | {message}", level="INFO")
    status: tuple[str, int, str]
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        def save_action() -> None:
            if rank == failing_rank:
                raise OSError(f"rank-{rank}-save-failed")
            if rank == 1:
                time.sleep(0.25)
            (Path(output_dir) / f"rank-{rank}.saved").write_bytes(b"checkpoint")

        try:
            with _remote_checkpoint_save_rank_skew(
                step=17,
                world_size=world_size,
                rank=rank,
            ):
                save_action()
        except BaseException as exc:  # noqa: BLE001 - report child failures
            status = ("error", rank, f"{type(exc).__name__}: {exc}")
        else:
            status = ("ok", rank, "")
    except BaseException as exc:  # noqa: BLE001 - report child setup failures
        status = ("setup_error", rank, f"{type(exc).__name__}: {exc}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.remove(handler_id)
        result_queue.put((*status, sink.getvalue()))


def _spec(tmp_path: Path, *, split: str = "val", packed: bool = True) -> ValidationEvalSpec:
    return ValidationEvalSpec(
        metric_prefix=split,
        log_label=f"{split} validation",
        packed_row_dir=tmp_path / split if packed else None,
        selection_dir=None,
    )


@pytest.mark.parametrize("split", ["val", "val_spliced"])
def test_packed_validation_reports_exact_split_rank_and_outcome(
    tmp_path: Path,
    split: str,
) -> None:
    sink, handler_id = _capture()
    try:
        with _validation_loader_health(
            step=5000,
            spec=_spec(tmp_path, split=split),
            ddp_rank=3,
        ):
            _loader_health_record("owner_decode", 1.0)
            _loader_health_record("owner_decode", 3.0)
            _loader_health_record("direct_load", 4.0, fallback=True)
    finally:
        logger.remove(handler_id)

    lines = [line for line in sink.getvalue().splitlines() if "loader health eval:" in line]
    assert lines == [
        "INFO | loader health eval: "
        f"step=05000 split={split} rank=3 outcome=complete | "
        "loads=3 owner=2 attach=0 direct=1 fallbacks=1 | "
        "mean/max s: owner_decode=2.0/3.0s attach_wait=- direct=4.0/4.0s"
    ]


def test_exact_validation_excludes_concurrent_training_prefetch(tmp_path: Path) -> None:
    background_done = threading.Event()

    def training_prefetch_record() -> None:
        _loader_health_record("direct_load", 9.0, fallback=True)
        background_done.set()

    sink, handler_id = _capture()
    try:
        with _validation_loader_health(
            step=250,
            spec=_spec(tmp_path),
            ddp_rank=0,
        ):
            thread = threading.Thread(
                target=training_prefetch_record,
                name="simulated-training-prefetch",
                daemon=True,
            )
            thread.start()
            assert background_done.wait(timeout=5.0)
            thread.join(timeout=5.0)
            assert not thread.is_alive()
            _loader_health_record("shared_attach", 2.0)
    finally:
        logger.remove(handler_id)

    eval_line = next(line for line in sink.getvalue().splitlines() if "loader health eval:" in line)
    assert "loads=1 owner=0 attach=1 direct=0 fallbacks=0" in eval_line
    assert "attach_wait=2.0/2.0s" in eval_line
    assert "9.0" not in eval_line

    global_health = loader_health_snapshot()
    assert global_health["base_loads"] == 2.0
    assert global_health["shared_attaches"] == 1.0
    assert global_health["direct_loads"] == 1.0
    assert global_health["fallbacks"] == 1.0


def test_validation_error_is_reported_without_replacing_the_exception(tmp_path: Path) -> None:
    original = OSError("simulated validation shard read failure")
    sink, handler_id = _capture()
    try:
        with pytest.raises(OSError) as raised:
            with _validation_loader_health(
                step=42,
                spec=_spec(tmp_path),
                ddp_rank=1,
            ):
                _loader_health_record("direct_load", 6.0)
                raise original
    finally:
        logger.remove(handler_id)

    assert raised.value is original
    lines = [line for line in sink.getvalue().splitlines() if "loader health eval:" in line]
    assert len(lines) == 1
    assert "step=00042 split=val rank=1 outcome=error" in lines[0]
    assert "loads=1 owner=0 attach=0 direct=1 fallbacks=0" in lines[0]
    assert "direct=6.0/6.0s" in lines[0]


def test_live_validation_does_not_claim_zero_packed_io(tmp_path: Path) -> None:
    sink, handler_id = _capture()
    try:
        with _validation_loader_health(
            step=100,
            spec=_spec(tmp_path, packed=False),
            ddp_rank=0,
        ):
            # Artificially prove that the live-policy branch did not install a
            # packed eval scope: the record remains global-only.
            _loader_health_record("direct_load", 5.0)
    finally:
        logger.remove(handler_id)

    assert "loader health eval:" not in sink.getvalue()
    assert loader_health_snapshot()["direct_loads"] == 1.0


def test_gate_cursor_advances_and_never_reuses_a_historical_maximum() -> None:
    baseline = loader_health_snapshot()
    _loader_health_record("direct_load", 10.0)

    sink, handler_id = _capture()
    try:
        after_eval = _log_loader_health_gate(
            step=5000,
            gate="eval",
            ddp_rank=0,
            previous=baseline,
        )
        _loader_health_record("direct_load", 2.0, fallback=True)
        after_save = _log_loader_health_gate(
            step=5000,
            gate="save",
            ddp_rank=0,
            previous=after_eval,
        )
    finally:
        logger.remove(handler_id)

    assert after_eval["direct_load_s_max"] == 10.0
    assert after_save["direct_load_s_max"] == 10.0
    lines = [line for line in sink.getvalue().splitlines() if "loader health gate:" in line]
    assert lines == [
        "INFO | loader health gate: step=05000 gate=eval rank=0 | "
        "loads=1 owner=0 attach=0 direct=1 fallbacks=0 | "
        "mean s: owner_decode=- attach_wait=- direct=10.0s",
        "INFO | loader health gate: step=05000 gate=save rank=0 | "
        "loads=1 owner=0 attach=0 direct=1 fallbacks=1 | "
        "mean s: owner_decode=- attach_wait=- direct=2.0s",
    ]
    assert all("mean/max" not in line for line in lines)


def test_zero_activity_gate_is_still_greppable() -> None:
    baseline = loader_health_snapshot()
    sink, handler_id = _capture()
    try:
        current = _log_loader_health_gate(
            step=0,
            gate="eval",
            ddp_rank=7,
            previous=baseline,
        )
    finally:
        logger.remove(handler_id)

    assert current == baseline
    assert sink.getvalue().strip() == (
        "INFO | loader health gate: step=00000 gate=eval rank=7 | "
        "loads=0 owner=0 attach=0 direct=0 fallbacks=0 | "
        "mean s: owner_decode=- attach_wait=- direct=-"
    )


def test_gate_snapshot_includes_activity_from_other_threads() -> None:
    baseline = loader_health_snapshot()

    def record() -> None:
        _loader_health_record("owner_decode", 1.5)

    thread = threading.Thread(target=record, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    sink, handler_id = _capture()
    try:
        current = _log_loader_health_gate(
            step=1,
            gate="save",
            ddp_rank=0,
            previous=baseline,
        )
    finally:
        logger.remove(handler_id)

    assert current["owner_decodes"] == 1.0
    assert "loads=1 owner=1 attach=0 direct=0 fallbacks=0" in sink.getvalue()
    assert "owner_decode=1.5s" in sink.getvalue()


def test_remote_save_skew_gather_runs_after_success_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    timestamps = iter([10.0, 13.25])
    monkeypatch.setattr(train_module.time, "monotonic", lambda: next(timestamps))

    def save_action() -> None:
        events.append("save")

    def collective_result(*, local_error: BaseException | None, world_size: int, rank: int) -> None:
        events.append(("consensus", local_error, world_size, rank))

    def rank_skew(
        label: str,
        seconds: float,
        *,
        rank: int,
        world_size: int,
    ) -> dict[str, float]:
        events.append(("skew", label, seconds, world_size, rank))
        return {}

    monkeypatch.setattr(train_module, "ddp_collective_result", collective_result)
    monkeypatch.setattr(train_module, "log_rank_duration_skew", rank_skew)

    with _remote_checkpoint_save_rank_skew(
        step=42,
        world_size=8,
        rank=3,
    ):
        save_action()

    assert events == [
        "save",
        ("consensus", None, 8, 3),
        ("skew", "checkpoint save (step 42)", 3.25, 8, 3),
    ]


def test_remote_save_failure_reaches_consensus_and_never_enters_skew_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OSError("local checkpoint write failed")
    observed: list[BaseException | None] = []
    timestamps = iter([20.0, 23.0])
    monkeypatch.setattr(train_module.time, "monotonic", lambda: next(timestamps))

    def fail_save() -> None:
        raise original

    def fail_consensus(
        *,
        local_error: BaseException | None,
        world_size: int,
        rank: int,
    ) -> None:
        assert (world_size, rank) == (8, 5)
        observed.append(local_error)
        raise RuntimeError("aligned remote-save failure") from local_error

    def forbidden_skew(*args: object, **kwargs: object) -> None:
        raise AssertionError("failed rank must not enter the skew collective")

    monkeypatch.setattr(train_module, "ddp_collective_result", fail_consensus)
    monkeypatch.setattr(train_module, "log_rank_duration_skew", forbidden_skew)

    with pytest.raises(RuntimeError, match="aligned remote-save failure") as exc_info:
        with _remote_checkpoint_save_rank_skew(
            step=42,
            world_size=8,
            rank=5,
        ):
            fail_save()

    assert observed == [original]
    assert exc_info.value.__cause__ is original


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
@pytest.mark.parametrize("failing_rank", [None, 1], ids=["success", "rank_failure"])
def test_remote_save_skew_two_rank_collective_choreography(
    tmp_path: Path,
    failing_rank: int | None,
) -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_remote_save_skew_worker,
            args=(rank, 2, init_file, str(tmp_path), failing_rank, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [result_queue.get(timeout=60) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    by_rank = {result[1]: result for result in results}
    assert set(by_rank) == {0, 1}
    if failing_rank is None:
        assert [by_rank[rank][0] for rank in range(2)] == ["ok", "ok"]
        assert (tmp_path / "rank-0.saved").read_bytes() == b"checkpoint"
        assert (tmp_path / "rank-1.saved").read_bytes() == b"checkpoint"
        assert "rank duration min/median/max" in by_rank[0][3]
        assert "slowest rank 1" in by_rank[0][3]
        assert "rank duration min/median/max" not in by_rank[1][3]
    else:
        assert [by_rank[rank][0] for rank in range(2)] == ["error", "error"]
        assert all("checkpoint local-save failure" in by_rank[rank][2] for rank in range(2))
        assert all("rank-1-save-failed" in by_rank[rank][2] for rank in range(2))
        assert all("rank duration min/median/max" not in by_rank[rank][3] for rank in range(2))


def _named_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_train_loop_wires_one_validation_scope_and_both_completed_gates() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(train_one_config)))

    snapshot_calls = _named_calls(tree, "loader_health_snapshot")
    assert len(snapshot_calls) == 1

    scope_calls = _named_calls(tree, "_validation_loader_health")
    assert len(scope_calls) == 1
    scope_with = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(item.context_expr is scope_calls[0] for item in node.items)
    )
    assert len(_named_calls(scope_with, "_build_validation_loader")) == 1
    assert len(_named_calls(scope_with, "evaluate_encdec_bpb_conditioned")) == 1

    gate_calls = _named_calls(tree, "_log_loader_health_gate")
    assert len(gate_calls) == 2
    gate_labels = {
        keyword.value.value
        for call in gate_calls
        for keyword in call.keywords
        if keyword.arg == "gate" and isinstance(keyword.value, ast.Constant)
    }
    assert gate_labels == {"eval", "save"}

    remote_save_calls = _named_calls(tree, "_remote_checkpoint_save_rank_skew")
    assert len(remote_save_calls) == 1
    remote_save_with = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(item.context_expr is remote_save_calls[0] for item in node.items)
    )
    assert len(_named_calls(remote_save_with, "save_checkpoint")) == 1
    publish_calls = _named_calls(tree, "ddp_publish_saved_checkpoint")
    assert len(publish_calls) == 1
    assert remote_save_calls[0].lineno < publish_calls[0].lineno
