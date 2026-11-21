"""Real-process coverage for rank-0 checkpoint model broadcasts."""

from __future__ import annotations

import ast
import collections
import pickle
import time
from pathlib import Path
from typing import Any

import pytest
import torch as t
import torch.distributed as dist

import nanochat.flash_attention as _flash_attention
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.model import ForwardResult
from tests.encdec_test_helpers import (
    build_tiny_model,
    make_precomputed_dataloader,
    setup_optimizer_simple,
    train_steps,
)
from tests.padded_encoder_reference import use_test_owned_cpu_encoder_reference

_STEP = 0
_PROMPT_BOUND_S = 10.0


def _fixture_model_data() -> collections.OrderedDict[str, object]:
    model_data = collections.OrderedDict(
        [
            (
                "bf16_weight",
                t.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=t.bfloat16),
            ),
            ("training_epoch", 17),
            ("fp32_weight", t.tensor([0.25, -7.0, 11.5], dtype=t.float32)),
            ("transposed_weight", t.arange(6, dtype=t.float32).reshape(2, 3).t()),
            ("grad_weight", t.tensor([2.0, -3.0], requires_grad=True)),
        ]
    )
    model_data._metadata = collections.OrderedDict(
        [
            ("", {"version": 3}),
            ("encoder", {"version": 1}),
        ]
    )
    return model_data


def _assert_model_data_exact(
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert type(actual) is type(expected)
    assert actual.keys() == expected.keys()
    assert getattr(actual, "_metadata", None) == getattr(expected, "_metadata", None)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, t.Tensor):
            assert isinstance(actual_value, t.Tensor)
            assert actual_value.dtype == expected_value.dtype
            assert actual_value.shape == expected_value.shape
            assert actual_value.requires_grad == expected_value.requires_grad
            assert t.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _model_summary(model_data: dict[str, object]) -> dict[str, object]:
    return {
        "container": type(model_data).__name__,
        "keys": list(model_data),
        "bf16_dtype": str(model_data["bf16_weight"].dtype),
        "fp32_dtype": str(model_data["fp32_weight"].dtype),
        "scalar": model_data["training_epoch"],
        "transposed_values": model_data["transposed_weight"].tolist(),
        "grad_requires_grad": model_data["grad_weight"].requires_grad,
        "metadata": getattr(model_data, "_metadata", None),
    }


def _assert_nested_exact(actual: Any, expected: Any) -> None:
    if isinstance(expected, t.Tensor):
        assert isinstance(actual, t.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert t.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_exact(actual_value, expected_value)
    else:
        assert actual == expected


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    optimizer_by_rank: list[dict[str, object]] | None = None,
) -> None:
    model_data = _fixture_model_data()
    meta_data = {"step": _STEP, "fixture": "checkpoint broadcast"}
    if optimizer_by_rank is None:
        save_checkpoint(
            str(checkpoint_dir),
            _STEP,
            model_data,
            None,
            meta_data,
            rank=0,
        )
        return
    for rank, optimizer_data in enumerate(optimizer_by_rank):
        save_checkpoint(
            str(checkpoint_dir),
            _STEP,
            model_data,
            optimizer_data,
            meta_data,
            rank=rank,
        )


def _model_read_count_target(
    rank: int,
    _world_size: int,
    checkpoint_dir: str,
    load_rank: int | None,
) -> dict[str, object]:
    model_path = Path(checkpoint_dir) / f"model_{_STEP:06d}.pt"
    original_load = t.load
    model_load_count = 0

    def counting_load(path: object, *args: object, **kwargs: object) -> object:
        nonlocal model_load_count
        if isinstance(path, (str, Path)) and Path(path) == model_path:
            model_load_count += 1
        return original_load(path, *args, **kwargs)

    t.load = counting_load
    try:
        checkpoint_rank = rank if load_rank is None else load_rank
        direct, direct_optimizer, direct_meta = load_checkpoint(
            checkpoint_dir,
            _STEP,
            t.device("cpu"),
            rank=checkpoint_rank,
            broadcast_model=False,
        )
        direct_reads = model_load_count
        assert direct_reads == 1
        assert direct_optimizer is None
        assert direct_meta == {"step": _STEP, "fixture": "checkpoint broadcast"}

        model_load_count = 0
        broadcast, broadcast_optimizer, broadcast_meta = load_checkpoint(
            checkpoint_dir,
            _STEP,
            t.device("cpu"),
            rank=checkpoint_rank,
            broadcast_model=True,
        )
        broadcast_reads = model_load_count
        assert broadcast_reads == (1 if rank == 0 else 0)
        assert broadcast_optimizer is None
        assert broadcast_meta == direct_meta
        return {
            "direct_reads": direct_reads,
            "broadcast_reads": broadcast_reads,
            "direct": _model_summary(direct),
            "broadcast": _model_summary(broadcast),
        }
    finally:
        t.load = original_load


def _flag_off_rank_conditional_target(
    rank: int,
    _world_size: int,
    checkpoint_dir: str,
) -> int:
    if rank == 0:
        model_data, optimizer_data, _meta_data = load_checkpoint(
            checkpoint_dir,
            _STEP,
            t.device("cpu"),
            rank=rank,
            broadcast_model=False,
        )
        _assert_model_data_exact(model_data, _fixture_model_data())
        assert optimizer_data is None
    return rank


def _model_failure_target(
    rank: int,
    _world_size: int,
    checkpoint_dir: str,
    failure_case: str,
) -> dict[str, object]:
    root_expectations = {
        "missing": (FileNotFoundError, r"model_000000\.pt"),
        "corrupt": (pickle.UnpicklingError, r"Weights only load failed"),
        "malformed": (AttributeError, r"'list' object has no attribute 'items'"),
    }
    receiver_matches = {
        "missing": (
            r"rank 0 failed to load model checkpoint: "
            r"FileNotFoundError:.*model_000000\.pt"
        ),
        "corrupt": (
            r"rank 0 failed to load model checkpoint: "
            r"UnpicklingError: Weights only load failed"
        ),
        "malformed": (
            r"rank 0 failed to load model checkpoint: "
            r"AttributeError: 'list' object has no attribute 'items'"
        ),
    }
    expected_type, root_match = root_expectations[failure_case]
    match = root_match if rank == 0 else receiver_matches[failure_case]
    started = time.monotonic()
    with pytest.raises(expected_type if rank == 0 else RuntimeError, match=match) as exc_info:
        load_checkpoint(
            checkpoint_dir,
            _STEP,
            t.device("cpu"),
            rank=rank,
            broadcast_model=True,
        )
    return {"elapsed": time.monotonic() - started, "message": str(exc_info.value)}


def _allocation_failure_target(
    rank: int,
    _world_size: int,
    checkpoint_dir: str,
) -> dict[str, object]:
    original_empty = t.empty

    def failing_empty(*args: object, **kwargs: object) -> t.Tensor:
        if kwargs.get("dtype") == t.bfloat16:
            raise RuntimeError("injected receiver-only allocation failure")
        return original_empty(*args, **kwargs)

    if rank == 1:
        t.empty = failing_empty
    try:
        started = time.monotonic()
        with pytest.raises(
            RuntimeError,
            match=(
                r"model checkpoint tensor allocation failed: "
                r"rank 1: RuntimeError: injected receiver-only allocation failure"
            ),
        ) as exc_info:
            load_checkpoint(
                checkpoint_dir,
                _STEP,
                t.device("cpu"),
                rank=rank,
                broadcast_model=True,
            )
        return {"elapsed": time.monotonic() - started, "message": str(exc_info.value)}
    finally:
        t.empty = original_empty


def _world_one_target(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> bool:
    assert rank == 0
    assert world_size == 1
    assert dist.is_initialized()
    direct, _, direct_meta = load_checkpoint(
        checkpoint_dir,
        _STEP,
        t.device("cpu"),
    )
    broadcast, _, broadcast_meta = load_checkpoint(
        checkpoint_dir,
        _STEP,
        t.device("cpu"),
        broadcast_model=True,
    )
    _assert_model_data_exact(broadcast, direct)
    assert broadcast_meta == direct_meta
    return True


def _plain_dict_target(
    rank: int,
    _world_size: int,
    checkpoint_dir: str,
) -> bool:
    model_data, _, _ = load_checkpoint(
        checkpoint_dir,
        _STEP,
        t.device("cpu"),
        rank=rank,
        broadcast_model=True,
    )
    assert type(model_data) is dict
    assert t.equal(model_data["weight"], t.tensor([1.0, 2.0]))
    return True


def _optimizer_target(
    rank: int,
    _world_size: int,
    checkpoint_dir: str,
) -> int:
    model_data, optimizer_data, _meta_data = load_checkpoint(
        checkpoint_dir,
        _STEP,
        t.device("cpu"),
        load_optimizer=True,
        rank=rank,
        broadcast_model=True,
    )
    _assert_model_data_exact(model_data, _fixture_model_data())
    optimizer_path = Path(checkpoint_dir) / f"optim_{_STEP:06d}_rank{rank}.pt"
    direct_optimizer = t.load(optimizer_path, map_location="cpu", weights_only=True)
    _assert_nested_exact(optimizer_data, direct_optimizer)
    assert optimizer_data["rank_marker"] == rank
    return optimizer_data["rank_marker"]


def _train_one_batch(
    model: t.nn.Module,
    optimizer: t.optim.Optimizer,
    batch: tuple[Any, ...],
) -> t.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    (
        x,
        y,
        enc_fids,
        enc_mags,
        enc_mask,
        _enc_pcts,
        seg_map,
        seg_doc_id,
        _fp_targets,
        _state,
    ) = batch
    result = model(
        x,
        y,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id=seg_doc_id,
    )
    loss = result.loss if isinstance(result, ForwardResult) else result
    loss.backward()
    optimizer.step()
    return loss.detach().clone()


@use_test_owned_cpu_encoder_reference()
def _resume_equivalence_target(
    rank: int,
    _world_size: int,
    root: str,
) -> dict[str, float]:
    old_attention_impl = _flash_attention._override_impl
    _flash_attention._override_impl = FlashAttnImpl.SDPA
    try:
        rank_root = Path(root) / f"rank{rank}"
        checkpoint_dir = Path(root) / "checkpoint"

        t.manual_seed(1234)
        source_model = build_tiny_model(n_layer=1, xattn_frequency=1)
        source_optimizer = setup_optimizer_simple(source_model)
        source_loader = make_precomputed_dataloader(
            rank_root / "source",
            B=2,
            n_rows=8,
            seed=77,
        )
        train_steps(source_model, source_optimizer, source_loader, n_steps=1)
        save_checkpoint(
            str(checkpoint_dir),
            1,
            source_model.state_dict(),
            source_optimizer.state_dict(),
            {"step": 1},
            rank=rank,
        )
        dist.barrier()

        broadcast_state, broadcast_optimizer_state, _ = load_checkpoint(
            str(checkpoint_dir),
            1,
            t.device("cpu"),
            load_optimizer=True,
            rank=rank,
            broadcast_model=True,
        )
        direct_state, direct_optimizer_state, _ = load_checkpoint(
            str(checkpoint_dir),
            1,
            t.device("cpu"),
            load_optimizer=True,
            rank=rank,
            broadcast_model=False,
        )
        _assert_model_data_exact(broadcast_state, direct_state)
        _assert_nested_exact(broadcast_optimizer_state, direct_optimizer_state)

        t.manual_seed(9000)
        broadcast_model = build_tiny_model(n_layer=1, xattn_frequency=1)
        direct_model = build_tiny_model(n_layer=1, xattn_frequency=1)
        broadcast_model.load_state_dict(broadcast_state, strict=True, assign=True)
        direct_model.load_state_dict(direct_state, strict=True, assign=True)
        broadcast_optimizer = setup_optimizer_simple(broadcast_model)
        direct_optimizer = setup_optimizer_simple(direct_model)
        broadcast_optimizer.load_state_dict(broadcast_optimizer_state)
        direct_optimizer.load_state_dict(direct_optimizer_state)

        broadcast_loader = make_precomputed_dataloader(
            rank_root / "broadcast",
            B=2,
            n_rows=8,
            seed=77,
        )
        direct_loader = make_precomputed_dataloader(
            rank_root / "direct",
            B=2,
            n_rows=8,
            seed=77,
        )
        next(broadcast_loader)
        next(direct_loader)
        broadcast_batch = next(broadcast_loader)
        direct_batch = next(direct_loader)
        _assert_nested_exact(broadcast_batch, direct_batch)

        broadcast_loss = _train_one_batch(
            broadcast_model,
            broadcast_optimizer,
            broadcast_batch,
        )
        direct_loss = _train_one_batch(
            direct_model,
            direct_optimizer,
            direct_batch,
        )
        assert t.equal(broadcast_loss, direct_loss)
        _assert_nested_exact(broadcast_model.state_dict(), direct_model.state_dict())
        _assert_nested_exact(
            broadcast_optimizer.state_dict(),
            direct_optimizer.state_dict(),
        )
        return {
            "broadcast_loss": broadcast_loss.item(),
            "direct_loss": direct_loss.item(),
        }
    finally:
        _flash_attention._override_impl = old_attention_impl


def _assert_parent_expected_results(
    results: list[dict[str, object]],
    expected: dict[str, object],
) -> None:
    assert [result["direct_reads"] for result in results] == [1, 1]
    assert [result["broadcast_reads"] for result in results] == [1, 0]
    for result in results:
        assert result["direct"] == expected
        assert result["broadcast"] == expected


def _assert_prompt_results(results: list[dict[str, object]]) -> None:
    assert all(result["elapsed"] < _PROMPT_BOUND_S for result in results)


@pytest.mark.parametrize(
    "load_rank",
    [
        pytest.param(None, id="receiver-does-not-read-model"),
        pytest.param(0, id="distributed-rank-selects-role"),
    ],
)
def test_two_rank_broadcast_model_read_and_role_contracts(
    tmp_path: Path,
    gloo_multiprocess,
    load_rank: int | None,
) -> None:
    _write_checkpoint(tmp_path)
    model_path = tmp_path / f"model_{_STEP:06d}.pt"
    expected = _model_summary(t.load(model_path, map_location="cpu", weights_only=True))
    results = gloo_multiprocess(
        2,
        _model_read_count_target,
        str(tmp_path),
        load_rank,
    )
    _assert_parent_expected_results(results, expected)


def test_flag_off_rank_conditional_caller_does_not_deadlock(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    _write_checkpoint(tmp_path)
    assert gloo_multiprocess(
        2,
        _flag_off_rank_conditional_target,
        str(tmp_path),
    ) == [0, 1]


@pytest.mark.parametrize(
    "failure_case",
    [
        pytest.param("missing", id="missing-model"),
        pytest.param("corrupt", id="corrupt-pickle"),
        pytest.param("malformed", id="malformed-container"),
    ],
)
def test_rank_zero_model_failure_reaches_both_ranks_promptly(
    tmp_path: Path,
    gloo_multiprocess,
    failure_case: str,
) -> None:
    checkpoint_dir = tmp_path
    if failure_case == "missing":
        checkpoint_dir /= "missing"
    elif failure_case == "corrupt":
        _write_checkpoint(checkpoint_dir)
        model_path = checkpoint_dir / f"model_{_STEP:06d}.pt"
        model_path.write_bytes(b"not a torch checkpoint")
    else:
        save_checkpoint(
            str(checkpoint_dir),
            _STEP,
            [t.tensor([1.0])],
            None,
            {"step": _STEP},
            rank=0,
        )
    results = gloo_multiprocess(
        2,
        _model_failure_target,
        str(checkpoint_dir),
        failure_case,
    )
    _assert_prompt_results(results)


def test_receiver_allocation_failure_reaches_both_ranks_promptly(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    _write_checkpoint(tmp_path)
    results = gloo_multiprocess(
        2,
        _allocation_failure_target,
        str(tmp_path),
    )
    _assert_prompt_results(results)


def test_world_one_broadcast_flag_uses_plain_load(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    _write_checkpoint(tmp_path)
    assert gloo_multiprocess(1, _world_one_target, str(tmp_path)) == [True]


def test_uninitialized_process_group_broadcast_flag_uses_plain_load(
    tmp_path: Path,
) -> None:
    assert not dist.is_initialized()
    _write_checkpoint(tmp_path)
    direct, _, direct_meta = load_checkpoint(
        str(tmp_path),
        _STEP,
        t.device("cpu"),
    )
    broadcast, _, broadcast_meta = load_checkpoint(
        str(tmp_path),
        _STEP,
        t.device("cpu"),
        broadcast_model=True,
    )
    _assert_model_data_exact(broadcast, direct)
    assert broadcast_meta == direct_meta


def test_two_rank_broadcast_preserves_plain_dict_container(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    save_checkpoint(
        str(tmp_path),
        _STEP,
        {"weight": t.tensor([1.0, 2.0])},
        None,
        {"step": _STEP},
        rank=0,
    )
    assert gloo_multiprocess(2, _plain_dict_target, str(tmp_path)) == [True, True]


def test_broadcast_keeps_optimizer_reads_rank_local(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    optimizers = [
        {"rank_marker": rank, "momentum": t.tensor([rank + 0.5])}
        for rank in range(2)
    ]
    _write_checkpoint(tmp_path, optimizer_by_rank=optimizers)
    assert gloo_multiprocess(2, _optimizer_target, str(tmp_path)) == [0, 1]


def test_encdec_resume_call_enables_model_broadcast() -> None:
    # This is a narrow wiring pin; distributed behavior is covered above.
    source_path = (
        Path(__file__).parents[2] / "scripts" / "train" / "conditional_encdec_train.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    train_one_config = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_config"
    )
    resume_calls = [
        node
        for node in ast.walk(train_one_config)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_checkpoint"
    ]
    assert len(resume_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in resume_calls[0].keywords}
    assert isinstance(keywords.get("broadcast_model"), ast.Constant)
    assert keywords["broadcast_model"].value is True


def test_two_rank_encdec_resume_broadcast_matches_direct_continuation(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(2, _resume_equivalence_target, str(tmp_path))
    assert results[0]["broadcast_loss"] == results[0]["direct_loss"]
    assert results[1]["broadcast_loss"] == results[1]["direct_loss"]
    assert results[0] == results[1]
