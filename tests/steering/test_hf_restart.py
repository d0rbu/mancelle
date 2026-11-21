"""Semantic, fault-injection, and multiprocess tests for HF SFT restart."""

from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch as t
import torch.distributed as dist
import torch.nn.functional as F

from steering.sft.hf_contract import HFDistributedImplementation
from steering.sft.hf_restart import (
    HF_RESTART_FORMAT,
    HF_RESTART_FORMAT_VERSION,
    HFRestartDescriptor,
    HFRestartIdentity,
    assert_distributed_hf_restart_identity,
    atomic_json_dump,
    build_hf_restart_identity,
    capture_hf_restart_runtime,
    load_hf_restart_model_in_local_rank_slots,
    load_hf_restart_rank_payload_in_local_rank_slots,
    restart_paths,
    restore_hf_restart_rng,
    save_hf_restart_distributed,
    validate_hf_restart,
    validate_hf_restart_runtime,
)
from steering.sft.iteration import sft_data_generator_bos_bestfit_core
from steering.sft.loss_mask import LossMaskMode


def _config(*, restart_from_step: int = -1) -> dict[str, Any]:
    return {
        "model_backend": "huggingface",
        "num_iterations": 12,
        "max_seq_len": 16,
        "profile_timing": False,
        "profile_trace_steps": [],
        "profile_trace_ranks": [],
        "profile_output_dir": "out/profile",
        "huggingface": {
            "model_id": "example/model",
            "revision": "a" * 40,
            "restart_from_step": restart_from_step,
            "restart_save_every": 3,
            "distributed_implementation": "ddp_fused_adamw",
        },
    }


def _identity(
    *,
    config: dict[str, Any] | None = None,
    world_size: int = 1,
    local_world_size: int = 1,
) -> HFRestartIdentity:
    return build_hf_restart_identity(
        resolved_config=_config() if config is None else config,
        repository="0123456789abcdef",
        source={
            "model_id": "example/model",
            "revision": "a" * 40,
            "architecture": "ExampleForCausalLM",
            "model_type": "example",
        },
        tokenizer_contract={
            "chat_template_sha256": "b" * 64,
            "model_vocab_size": 17,
            "tokenizer_vocab_size": 17,
            "structural_token_ids": {"bos": 1, "eos": 2, "pad": 0},
            "tied_embeddings": True,
        },
        train_mixture_fingerprint={"split": "train", "sha256": "c" * 64},
        val_mixture_fingerprint={"split": "val", "sha256": "d" * 64},
        world_size=world_size,
        local_world_size=local_world_size,
    )


def _dataloader_state(*, rank: int = 0, world_size: int = 1) -> dict[str, Any]:
    return {
        "next_cursor": rank + world_size * 7,
        "buffered_cursor_positions": [rank + world_size * 5, rank + world_size * 6],
        "consumed": 5,
        "micro_batch_count": 2,
    }


def _runtime(*, rank: int = 0, world_size: int = 1) -> dict[str, Any]:
    return capture_hf_restart_runtime(
        dataloader_state=_dataloader_state(rank=rank, world_size=world_size),
        completed_steps=3,
        smooth_train_loss=1.75,
        latest_val_bpb=2.5,
        min_val_bpb=2.25,
        total_training_time=17.0,
        device="cpu",
    )


def _model_and_optimizer() -> tuple[t.nn.Linear, t.optim.AdamW]:
    t.manual_seed(7)
    model = t.nn.Linear(4, 3)
    optimizer = t.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(t.arange(8, dtype=t.float32).reshape(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _save_world_one(
    root: Path,
    *,
    identity: HFRestartIdentity | None = None,
    implementation: HFDistributedImplementation = (HFDistributedImplementation.DDP_FUSED_ADAMW),
) -> tuple[t.nn.Linear, t.optim.AdamW, HFRestartIdentity]:
    model, optimizer = _model_and_optimizer()
    expected_identity = _identity() if identity is None else identity
    save_hf_restart_distributed(
        output_root=root,
        completed_step=3,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        runtime_state=_runtime(),
        identity=expected_identity,
        implementation=implementation,
        world_size=1,
        local_world_size=1,
        rank=0,
        master_process=True,
        device="cpu",
    )
    return model, optimizer, expected_identity


def test_identity_is_canonical_and_excludes_only_restore_and_diagnostics() -> None:
    first = _identity(config=_config(restart_from_step=-1))
    reordered = {
        "huggingface": {
            "restart_save_every": 3,
            "revision": "a" * 40,
            "model_id": "example/model",
            "distributed_implementation": "ddp_fused_adamw",
            "restart_from_step": 999,
        },
        "profile_timing": False,
        "max_seq_len": 16,
        "num_iterations": 12,
        "model_backend": "huggingface",
    }
    second = _identity(config=reordered)
    assert first == second
    assert first.payload["resolved_config"]["huggingface"]["restart_from_step"] == -1
    assert (
        not {
            "profile_timing",
            "profile_trace_steps",
            "profile_trace_ranks",
            "profile_output_dir",
        }
        & first.payload["resolved_config"].keys()
    )
    changed = _config()
    changed["max_seq_len"] = 32
    assert _identity(config=changed).sha256 != first.sha256
    assert json.loads(json.dumps(first.payload, allow_nan=False)) == first.payload


@pytest.mark.parametrize(
    ("world_size", "local_world_size"),
    [(0, 1), (1, 0), (2, 3), (3, 2)],
)
def test_invalid_identity_topology_fails(
    world_size: int,
    local_world_size: int,
) -> None:
    with pytest.raises(ValueError, match="topology"):
        _identity(world_size=world_size, local_world_size=local_world_size)


def _identity_consensus_worker(
    rank: int,
    init_file: str,
    mismatch: bool,
    queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        config = _config()
        if mismatch and rank == 1:
            config["num_iterations"] = 13
        identity = _identity(
            config=config,
            world_size=2,
            local_world_size=2,
        )
        assert_distributed_hf_restart_identity(identity, device="cpu")
        queue.put(("ok", rank))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_two_rank_semantic_identity_consensus_is_fail_closed(
    tmp_path: Path,
    mismatch: bool,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / f"identity-consensus-{mismatch}")
    processes = [
        context.Process(
            target=_identity_consensus_worker,
            args=(rank, init_file, mismatch, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=60) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    assert [process.exitcode for process in processes] == [0, 0]
    if mismatch:
        assert all(result[0] == "error" for result in results), results
        assert all("differs across distributed ranks" in result[2] for result in results)
    else:
        assert sorted(results) == [("ok", 0), ("ok", 1)]


def test_world_one_roundtrip_restores_model_optimizer_runtime_and_rng(tmp_path: Path) -> None:
    root = tmp_path / "output"
    model, optimizer, identity = _save_world_one(root)
    expected_parameters = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    descriptor = validate_hf_restart(
        output_root=root,
        step=3,
        expected_identity=identity,
        implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
        world_size=1,
        local_world_size=1,
    )

    with t.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    load_hf_restart_model_in_local_rank_slots(
        descriptor,
        model=model,  # type: ignore[arg-type] - load_state_dict is the runtime protocol
        local_rank=0,
        local_world_size=1,
        device="cpu",
    )
    for name, tensor in model.state_dict().items():
        assert t.equal(tensor, expected_parameters[name])

    rank_payload = load_hf_restart_rank_payload_in_local_rank_slots(
        descriptor,
        rank=0,
        local_rank=0,
        local_world_size=1,
        device="cpu",
    )
    loaded_optimizer = rank_payload["optimizer"]
    assert loaded_optimizer.keys() == expected_optimizer.keys()
    optimizer.load_state_dict(dict(loaded_optimizer))
    assert optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    runtime = rank_payload["runtime"]
    assert runtime["completed_steps"] == 3
    assert runtime["dataloader_state"] == _dataloader_state()
    assert runtime["smooth_train_loss"] == 1.75
    assert runtime["latest_val_bpb"] == 2.5
    assert runtime["min_val_bpb"] == 2.25
    assert runtime["total_training_time"] == 17.0

    expected_next = t.rand(5)
    t.manual_seed(999)
    restore_hf_restart_rng(runtime, device="cpu")
    actual_next = t.rand(5)
    assert t.equal(actual_next, expected_next)


def test_complete_marker_is_last_and_binds_exact_nonempty_sizes(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _model, _optimizer, identity = _save_world_one(root)
    paths = restart_paths(root, 3, world_size=1)
    marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    assert marker["format"] == HF_RESTART_FORMAT
    assert marker["format_version"] == HF_RESTART_FORMAT_VERSION
    assert marker["completed_optimizer_step"] == 3
    assert marker["semantic_identity_sha256"] == identity.sha256
    assert marker["files"] == {
        paths["model"].name: paths["model"].stat().st_size,
        paths["meta"].name: paths["meta"].stat().st_size,
        paths["rank_0"].name: paths["rank_0"].stat().st_size,
    }
    assert all(size > 0 for size in marker["files"].values())
    assert not paths["lock"].exists()


def test_complete_marker_is_published_only_after_lock_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "output"
    paths = restart_paths(root, 3, world_size=1)
    observed_lock_states: list[bool] = []
    real_marker_write = atomic_json_dump

    def record_marker_write(payload, path) -> None:
        observed_lock_states.append(paths["lock"].exists())
        real_marker_write(payload, path)

    with patch(
        "steering.sft.hf_restart.atomic_json_dump",
        side_effect=record_marker_write,
    ):
        _save_world_one(root)

    assert observed_lock_states == [False]
    assert paths["complete"].is_file()


def test_data_write_failure_leaves_no_loadable_marker_and_retains_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    model, optimizer = _model_and_optimizer()
    with patch(
        "steering.sft.hf_restart.save_checkpoint",
        side_effect=OSError("injected data write failure"),
    ):
        with pytest.raises(OSError, match="injected"):
            save_hf_restart_distributed(
                output_root=root,
                completed_step=3,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                runtime_state=_runtime(),
                identity=_identity(),
                implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
                world_size=1,
                local_world_size=1,
                rank=0,
                master_process=True,
                device="cpu",
            )
    paths = restart_paths(root, 3, world_size=1)
    assert paths["lock"].is_file()
    assert not paths["complete"].exists()
    with pytest.raises(FileNotFoundError, match="no complete marker"):
        validate_hf_restart(
            output_root=root,
            step=3,
            expected_identity=_identity(),
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
        )


def test_marker_write_failure_leaves_complete_data_but_no_loadable_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    model, optimizer = _model_and_optimizer()
    with patch(
        "steering.sft.hf_restart.atomic_json_dump",
        side_effect=OSError("injected marker failure"),
    ):
        with pytest.raises(OSError, match="marker"):
            save_hf_restart_distributed(
                output_root=root,
                completed_step=3,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                runtime_state=_runtime(),
                identity=_identity(),
                implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
                world_size=1,
                local_world_size=1,
                rank=0,
                master_process=True,
                device="cpu",
            )
    paths = restart_paths(root, 3, world_size=1)
    assert paths["model"].is_file()
    assert paths["meta"].is_file()
    assert paths["rank_0"].is_file()
    assert not paths["lock"].exists()
    assert not paths["complete"].exists()


def test_lock_cleanup_failure_prevents_marker_publication(tmp_path: Path) -> None:
    root = tmp_path / "output"
    paths = restart_paths(root, 3, world_size=1)
    with patch.object(Path, "unlink", side_effect=OSError("injected lock failure")):
        with pytest.raises(OSError, match="lock failure"):
            _save_world_one(root)

    assert paths["model"].is_file()
    assert paths["meta"].is_file()
    assert paths["rank_0"].is_file()
    assert paths["lock"].is_file()
    assert not paths["complete"].exists()


def test_existing_partial_or_complete_step_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "output"
    paths = restart_paths(root, 3, world_size=1)
    paths["model"].parent.mkdir(parents=True)
    paths["model"].write_bytes(b"partial")
    model, optimizer = _model_and_optimizer()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_hf_restart_distributed(
            output_root=root,
            completed_step=3,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            runtime_state=_runtime(),
            identity=_identity(),
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
            rank=0,
            master_process=True,
            device="cpu",
        )
    assert paths["model"].read_bytes() == b"partial"


def test_complete_marker_with_an_in_progress_lock_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _save_world_one(root)
    paths = restart_paths(root, 3, world_size=1)
    paths["lock"].write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="complete marker.*in-progress lock"):
        validate_hf_restart(
            output_root=root,
            step=3,
            expected_identity=_identity(),
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
        )


@pytest.mark.parametrize("target", ["model", "meta", "rank_0"])
def test_post_marker_truncation_is_detected_by_inventory(
    tmp_path: Path,
    target: str,
) -> None:
    root = tmp_path / target
    _model, _optimizer, identity = _save_world_one(root)
    paths = restart_paths(root, 3, world_size=1)
    paths[target].write_bytes(b"x")
    with pytest.raises(RuntimeError, match="inventory"):
        validate_hf_restart(
            output_root=root,
            step=3,
            expected_identity=identity,
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("completed_optimizer_step", 4, "completed_optimizer_step"),
        ("world_size", 2, "world_size"),
        ("local_world_size", 2, "local_world_size"),
        ("distributed_implementation", "dist_muon_adamw", "distributed_implementation"),
        ("format", "other", "format"),
        ("format_version", 99, "format_version"),
    ],
)
def test_marker_contract_tampering_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    root = tmp_path / field
    _model, _optimizer, identity = _save_world_one(root)
    marker_path = restart_paths(root, 3, world_size=1)["complete"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        validate_hf_restart(
            output_root=root,
            step=3,
            expected_identity=identity,
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
        )


def test_marker_and_metadata_reject_unknown_schema_keys(tmp_path: Path) -> None:
    for target in ("complete", "meta"):
        root = tmp_path / target
        _model, _optimizer, identity = _save_world_one(root)
        paths = restart_paths(root, 3, world_size=1)
        path = paths[target]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = "drift"
        path.write_text(json.dumps(payload), encoding="utf-8")
        if target == "meta":
            marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
            marker["files"][paths["meta"].name] = paths["meta"].stat().st_size
            paths["complete"].write_text(json.dumps(marker), encoding="utf-8")
        with pytest.raises(RuntimeError, match="keys differ.*unexpected"):
            validate_hf_restart(
                output_root=root,
                step=3,
                expected_identity=identity,
                implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
                world_size=1,
                local_world_size=1,
            )


def test_semantic_identity_change_fails_before_any_tensor_load(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _model, _optimizer, _identity_saved = _save_world_one(root)
    changed_config = _config()
    changed_config["num_iterations"] = 13
    with patch("torch.load", side_effect=AssertionError("must not load tensors")):
        with pytest.raises(RuntimeError, match="semantic_identity_sha256"):
            validate_hf_restart(
                output_root=root,
                step=3,
                expected_identity=_identity(config=changed_config),
                implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
                world_size=1,
                local_world_size=1,
            )


def test_rng_runtime_rejects_wrong_device_and_malformed_states() -> None:
    runtime = _runtime()
    bad = dict(runtime)
    bad["torch_cpu_rng_state"] = [1, 2, 3]
    with pytest.raises(RuntimeError, match="CPU RNG"):
        restore_hf_restart_rng(bad, device="cpu")
    bad = dict(runtime)
    bad["torch_cuda_rng_state"] = t.zeros(4, dtype=t.uint8)
    with pytest.raises(RuntimeError, match="unexpectedly contains"):
        restore_hf_restart_rng(bad, device="cpu")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("completed_steps", True, "completed_steps"),
        ("completed_steps", "3", "completed_steps"),
        ("completed_steps", -1, "completed_steps"),
        ("smooth_train_loss", 1, "smooth_train_loss"),
        ("smooth_train_loss", float("nan"), "smooth_train_loss"),
        ("smooth_train_loss", -0.1, "smooth_train_loss"),
        ("latest_val_bpb", float("inf"), "latest_val_bpb"),
        ("latest_val_bpb", -0.1, "latest_val_bpb"),
        ("min_val_bpb", "2.0", "min_val_bpb"),
        ("total_training_time", -0.1, "total_training_time"),
        ("torch_cpu_rng_state", t.zeros((2, 2), dtype=t.uint8), "CPU RNG"),
        ("torch_cuda_rng_state", t.tensor([], dtype=t.uint8), "CUDA RNG"),
    ],
)
def test_runtime_schema_rejects_tampered_scalar_and_rng_domains(
    field: str,
    value: object,
    match: str,
) -> None:
    runtime = _runtime()
    runtime[field] = value
    with pytest.raises(RuntimeError, match=match):
        validate_hf_restart_runtime(runtime, expected_step=3)


def test_runtime_schema_rejects_key_and_checkpoint_step_drift() -> None:
    runtime = _runtime()
    runtime["unexpected"] = 1
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_hf_restart_runtime(runtime, expected_step=3)
    runtime = _runtime()
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        validate_hf_restart_runtime(runtime, expected_step=4)


@pytest.mark.parametrize(
    ("implementation", "rank", "master_process", "optimizer_state", "match"),
    [
        ("ddp_fused_adamw", 0, False, {}, "master_process"),
        ("ddp_fused_adamw", 0, True, None, "rank zero"),
        ("ddp_fused_adamw", 1, False, {}, "nonzero rank"),
        ("dist_muon_adamw", 1, False, None, "every rank"),
    ],
)
def test_restart_save_rejects_strategy_ownership_before_io(
    tmp_path: Path,
    implementation: str,
    rank: int,
    master_process: bool,
    optimizer_state: dict[str, Any] | None,
    match: str,
) -> None:
    model, _optimizer = _model_and_optimizer()
    with pytest.raises(ValueError, match=match):
        save_hf_restart_distributed(
            output_root=tmp_path,
            completed_step=3,
            model_state=model.state_dict(),
            optimizer_state=optimizer_state,
            runtime_state=_runtime(rank=rank, world_size=2),
            identity=_identity(world_size=2, local_world_size=2),
            implementation=implementation,
            world_size=2,
            local_world_size=2,
            rank=rank,
            master_process=master_process,
            device="cpu",
        )
    assert not (tmp_path / "restart").exists()


def test_restart_save_rejects_forged_identity_digest_before_io(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    expected = _identity()
    forged = HFRestartIdentity(payload=expected.payload, sha256="0" * 64)
    with pytest.raises(ValueError, match="identity digest"):
        save_hf_restart_distributed(
            output_root=tmp_path,
            completed_step=3,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            runtime_state=_runtime(),
            identity=forged,
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
            rank=0,
            master_process=True,
            device="cpu",
        )
    assert not (tmp_path / "restart").exists()


def test_rank_payload_load_rejects_missing_strategy_owned_optimizer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    _model, _optimizer, identity = _save_world_one(root)
    paths = restart_paths(root, 3, world_size=1)
    t.save({"optimizer": None, "runtime": _runtime()}, paths["rank_0"])
    marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    marker["files"][paths["rank_0"].name] = paths["rank_0"].stat().st_size
    paths["complete"].write_text(json.dumps(marker), encoding="utf-8")
    descriptor = validate_hf_restart(
        output_root=root,
        step=3,
        expected_identity=identity,
        implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
        world_size=1,
        local_world_size=1,
    )
    with pytest.raises(RuntimeError, match="rank zero is missing"):
        load_hf_restart_rank_payload_in_local_rank_slots(
            descriptor,
            rank=0,
            local_rank=0,
            local_world_size=1,
            device="cpu",
        )


def _distributed_worker(
    rank: int,
    init_file: str,
    output_root: str,
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
        implementation = HFDistributedImplementation(implementation_value)
        t.manual_seed(5)
        model = t.nn.Linear(2, 2)
        optimizer_payload = {"state_rank": rank, "tensor": t.tensor([rank])}
        optimizer_state = (
            optimizer_payload
            if implementation is HFDistributedImplementation.DIST_MUON_ADAMW or rank == 0
            else None
        )
        identity = _identity(world_size=2, local_world_size=2)
        save_hf_restart_distributed(
            output_root=output_root,
            completed_step=3,
            model_state=model.state_dict(),
            optimizer_state=optimizer_state,
            runtime_state=_runtime(rank=rank, world_size=2),
            identity=identity,
            implementation=implementation,
            world_size=2,
            local_world_size=2,
            rank=rank,
            master_process=rank == 0,
            device="cpu",
        )
        descriptor = validate_hf_restart(
            output_root=output_root,
            step=3,
            expected_identity=identity,
            implementation=implementation,
            world_size=2,
            local_world_size=2,
        )
        own = load_hf_restart_rank_payload_in_local_rank_slots(
            descriptor,
            rank=rank,
            local_rank=rank,
            local_world_size=2,
            device="cpu",
        )
        loaded_optimizer = own["optimizer"]
        queue.put(
            (
                "ok",
                rank,
                own["runtime"]["dataloader_state"]["next_cursor"],
                loaded_optimizer["state_rank"],
                own["optimizer"] is None,
            )
        )
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("implementation", list(HFDistributedImplementation))
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_two_rank_marker_runtime_and_optimizer_ownership(
    tmp_path: Path,
    implementation: HFDistributedImplementation,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / f"gloo-{implementation.value}")
    output_root = str(tmp_path / f"output-{implementation.value}")
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, init_file, output_root, implementation.value, queue),
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
    by_rank = {result[1]: result for result in results}
    assert by_rank[0][2] == 14
    assert by_rank[1][2] == 15
    if implementation is HFDistributedImplementation.DIST_MUON_ADAMW:
        assert by_rank[0][3] == 0
        assert by_rank[1][3] == 1
        assert by_rank[0][4] is False
        assert by_rank[1][4] is False
    else:
        assert by_rank[0][3] == by_rank[1][3] == 0
        assert by_rank[0][4] is False
        assert by_rank[1][4] is False


def _distributed_invalid_save_worker(
    rank: int,
    init_file: str,
    output_root: str,
    queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        runtime = _runtime(rank=rank, world_size=2)
        if rank == 1:
            runtime["completed_steps"] = "3"
        save_hf_restart_distributed(
            output_root=output_root,
            completed_step=3,
            model_state=t.nn.Linear(2, 2).state_dict(),
            optimizer_state={"rank": rank},
            runtime_state=runtime,
            identity=_identity(world_size=2, local_world_size=2),
            implementation=HFDistributedImplementation.DIST_MUON_ADAMW,
            world_size=2,
            local_world_size=2,
            rank=rank,
            master_process=rank == 0,
            device="cpu",
        )
        queue.put(("unexpected-success", rank))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put((type(exc).__name__, rank, str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_rank_local_save_validation_aborts_every_rank_before_io(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "invalid-save-gloo")
    output_root = tmp_path / "invalid-save-output"
    processes = [
        context.Process(
            target=_distributed_invalid_save_worker,
            args=(rank, init_file, str(output_root), queue),
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
    assert [process.exitcode for process in processes] == [0, 0]
    assert {result[0] for result in results} == {"AlignedRankFailure"}
    assert all("completed_steps" in result[2] for result in results)
    assert not output_root.exists()


def _distributed_invalid_load_worker(
    rank: int,
    init_file: str,
    queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        runtime = _runtime(rank=rank, world_size=2)
        if rank == 1:
            runtime["completed_steps"] = 4
        loaded = {"optimizer": {"rank": rank}, "runtime": runtime}
        descriptor = HFRestartDescriptor(
            root=Path("unused"),
            step=3,
            metadata={
                "distributed_implementation": HFDistributedImplementation.DIST_MUON_ADAMW.value
            },
        )
        with patch("steering.sft.hf_restart.t.load", return_value=loaded):
            load_hf_restart_rank_payload_in_local_rank_slots(
                descriptor,
                rank=rank,
                local_rank=0,
                local_world_size=1,
                device="cpu",
            )
        queue.put(("unexpected-success", rank))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put((type(exc).__name__, rank, str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_rank_local_loaded_payload_validation_aborts_every_rank(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "invalid-load-gloo")
    processes = [
        context.Process(
            target=_distributed_invalid_load_worker,
            args=(rank, init_file, queue),
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
    assert [process.exitcode for process in processes] == [0, 0]
    assert {result[0] for result in results} == {"AlignedRankFailure"}
    assert all("differs from checkpoint step" in result[2] for result in results)


def test_rank_and_local_slot_domains_fail_before_io(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    with pytest.raises(ValueError, match="outside world_size"):
        save_hf_restart_distributed(
            output_root=tmp_path,
            completed_step=3,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            runtime_state=_runtime(),
            identity=_identity(),
            implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            world_size=1,
            local_world_size=1,
            rank=1,
            master_process=False,
            device="cpu",
        )


class _ResumeDataset:
    def __len__(self) -> int:
        return 9

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


class _ResumeTokenizer:
    def get_bos_token_id(self) -> int:
        return 1

    def render_conversation(self, conversation: dict[str, int]) -> tuple[list[int], list[int]]:
        index = conversation["index"]
        length = 3 + (index % 5)
        return [1] + [2 + index * 7 + offset for offset in range(length - 1)], [0] + [1] * (
            length - 1
        )


class _ResumeModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = t.nn.Embedding(128, 8)
        self.dropout = t.nn.Dropout(0.35)
        self.projection = t.nn.Linear(8, 128)

    def forward(self, inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
        logits = self.projection(self.dropout(self.embedding(inputs)))
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="mean",
        )


def _resume_generator(state: dict[str, Any] | None = None):
    return sft_data_generator_bos_bestfit_core(
        dataset=_ResumeDataset(),
        tokenizer=_ResumeTokenizer(),
        max_seq_len=12,
        device_batch_size=2,
        ddp_rank=0,
        ddp_world_size=1,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=5,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        resume_state=state,
    )


def _training_step(model, optimizer, generator):
    inputs, targets, progress = next(generator)
    loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return inputs.clone(), targets.clone(), progress, float(loss.detach())


def _assert_nested_tensor_state_equal(left: Any, right: Any) -> None:
    if isinstance(left, t.Tensor):
        assert isinstance(right, t.Tensor)
        assert t.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_tensor_state_equal(left[key], right[key])
        return
    if isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_tensor_state_equal(left_item, right_item)
        return
    assert left == right


def test_uninterrupted_and_restart_training_are_bit_exact_with_dropout(
    tmp_path: Path,
) -> None:
    t.manual_seed(20260826)
    uninterrupted_model = _ResumeModel()
    uninterrupted_optimizer = t.optim.AdamW(
        uninterrupted_model.parameters(), lr=2e-4, betas=(0.8, 0.95)
    )
    uninterrupted_generator = _resume_generator()
    first = _training_step(uninterrupted_model, uninterrupted_optimizer, uninterrupted_generator)
    second = _training_step(uninterrupted_model, uninterrupted_optimizer, uninterrupted_generator)
    assert second[2].micro_batch_count == 2

    output_root = tmp_path / "bit-exact"
    identity = _identity()
    runtime = capture_hf_restart_runtime(
        dataloader_state=second[2].dataloader_state,
        completed_steps=2,
        smooth_train_loss=second[3],
        latest_val_bpb=None,
        min_val_bpb=None,
        total_training_time=0.0,
        device="cpu",
    )
    save_hf_restart_distributed(
        output_root=output_root,
        completed_step=2,
        model_state=uninterrupted_model.state_dict(),
        optimizer_state=uninterrupted_optimizer.state_dict(),
        runtime_state=runtime,
        identity=identity,
        implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
        world_size=1,
        local_world_size=1,
        rank=0,
        master_process=True,
        device="cpu",
    )
    expected_third = _training_step(
        uninterrupted_model, uninterrupted_optimizer, uninterrupted_generator
    )
    expected_fourth = _training_step(
        uninterrupted_model, uninterrupted_optimizer, uninterrupted_generator
    )
    expected_model = copy.deepcopy(uninterrupted_model.state_dict())
    expected_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())

    # Construction deliberately consumes random numbers. Restoring the saved
    # state after construction must still reproduce both subsequent dropout masks.
    t.manual_seed(999)
    resumed_model = _ResumeModel()
    resumed_optimizer = t.optim.AdamW(resumed_model.parameters(), lr=2e-4, betas=(0.8, 0.95))
    descriptor = validate_hf_restart(
        output_root=output_root,
        step=2,
        expected_identity=identity,
        implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
        world_size=1,
        local_world_size=1,
    )
    load_hf_restart_model_in_local_rank_slots(
        descriptor,
        model=resumed_model,  # type: ignore[arg-type]
        local_rank=0,
        local_world_size=1,
        device="cpu",
    )
    payload = load_hf_restart_rank_payload_in_local_rank_slots(
        descriptor,
        rank=0,
        local_rank=0,
        local_world_size=1,
        device="cpu",
    )
    resumed_optimizer.load_state_dict(payload["optimizer"])
    restore_hf_restart_rng(payload["runtime"], device="cpu")
    resumed_generator = _resume_generator(payload["runtime"]["dataloader_state"])
    actual_third = _training_step(resumed_model, resumed_optimizer, resumed_generator)
    actual_fourth = _training_step(resumed_model, resumed_optimizer, resumed_generator)

    assert first[2].micro_batch_count == 1
    for expected, actual in (
        (expected_third, actual_third),
        (expected_fourth, actual_fourth),
    ):
        assert t.equal(expected[0], actual[0])
        assert t.equal(expected[1], actual[1])
        assert expected[2].dataloader_state == actual[2].dataloader_state
        assert expected[3] == actual[3]
    for name, tensor in resumed_model.state_dict().items():
        assert t.equal(tensor, expected_model[name])
    _assert_nested_tensor_state_equal(resumed_optimizer.state_dict(), expected_optimizer)
