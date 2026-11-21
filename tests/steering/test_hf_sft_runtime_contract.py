"""Strict runtime/config/profiling contracts for distributed HF decoder SFT."""

from __future__ import annotations

import copy
import inspect
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train.chat_sft import (
    _distributed_directory_exists,
    _hf_persistence_due,
    _maybe_profile_hf_step,
    _record_hf_paid_step,
    _resolve_hf_profile_settings,
    main,
)
from steering.profiling import TimingRecorder
from steering.profiling.startup import StartupTrace
from steering.sft.hf_contract import (
    HFAttentionImplementation,
    HFDistributedImplementation,
)
from steering.sft.hf_decoder import (
    build_hf_optimizer,
    prepare_hf_training_model,
    validate_decoder_sft_config,
    validated_local_world_size,
)

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")


def _composed(name: str = "chat_sft_pythia_2_8b") -> dict:
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=name)
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("attention_implementation", "fa4", "attention_implementation"),
        ("distributed_implementation", "fsdp", "distributed_implementation"),
        ("torch_compile", 1, "torch_compile"),
        ("restart_from_step", -2, "restart_from_step"),
        ("restart_from_step", True, "restart_from_step"),
        ("restart_save_every", -1, "restart_save_every"),
        ("restart_save_every", 1.5, "restart_save_every"),
        ("data_snapshot_sha256", "A" * 64, "data_snapshot_sha256"),
        ("data_snapshot_sha256", "a" * 63, "data_snapshot_sha256"),
    ],
)
def test_every_new_hf_config_domain_rejects_near_misses(
    field: str,
    value: object,
    match: str,
) -> None:
    cfg = _composed()
    cfg["huggingface"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_decoder_sft_config(cfg)


@pytest.mark.parametrize("attention", list(HFAttentionImplementation))
@pytest.mark.parametrize("implementation", list(HFDistributedImplementation))
def test_all_supported_attention_optimizer_pairs_validate(
    attention: HFAttentionImplementation,
    implementation: HFDistributedImplementation,
) -> None:
    cfg = _composed()
    cfg["num_iterations"] = 10
    cfg["huggingface"]["attention_implementation"] = attention.value
    cfg["huggingface"]["distributed_implementation"] = implementation.value
    cfg["huggingface"]["data_snapshot_sha256"] = "0" * 64
    validate_decoder_sft_config(cfg, world_size=32)


def test_distributed_dataset_horizon_is_rejected_but_single_rank_remains_compatible() -> None:
    cfg = _composed()
    cfg["num_iterations"] = -1
    validate_decoder_sft_config(cfg, world_size=1)
    with pytest.raises(ValueError, match="positive num_iterations"):
        validate_decoder_sft_config(cfg, world_size=2)
    cfg["num_iterations"] = 1
    validate_decoder_sft_config(cfg, world_size=32)


@pytest.mark.parametrize(
    (
        "completed_steps",
        "last_step",
        "dry_run",
        "portable_every",
        "restart_every",
        "loaded_step",
        "expected",
    ),
    [
        (0, False, False, 1, 1, None, (False, False)),
        (1, False, True, 1, 1, None, (False, False)),
        (30, False, False, 0, 30, None, (False, True)),
        (30, False, False, 30, 0, None, (True, False)),
        (30, False, False, 30, 30, None, (True, True)),
        (31, False, False, 30, 30, None, (False, False)),
        (31, True, False, 0, 0, None, (True, False)),
        (31, True, False, 0, 30, None, (True, True)),
        # The first boundary after loading is observational only. Rewriting the
        # source checkpoint would collide with fail-closed no-overwrite state.
        (30, False, False, 30, 30, 30, (False, False)),
        (30, True, False, 30, 30, 30, (False, False)),
        (31, True, False, 30, 30, 30, (True, True)),
    ],
)
def test_hf_persistence_schedule_never_rewrites_the_loaded_boundary(
    completed_steps: int,
    last_step: bool,
    dry_run: bool,
    portable_every: int,
    restart_every: int,
    loaded_step: int | None,
    expected: tuple[bool, bool],
) -> None:
    assert (
        _hf_persistence_due(
            completed_steps=completed_steps,
            last_step=last_step,
            dry_run=dry_run,
            portable_save_every=portable_every,
            restart_save_every=restart_every,
            loaded_restart_step=loaded_step,
        )
        == expected
    )


def test_restart_publication_precedes_portable_publication_in_the_loop() -> None:
    source = inspect.getsource(main)
    assert source.index("save_hf_restart_distributed(") < source.index(
        "publish_hf_checkpoint_distributed("
    )


def test_restart_rng_is_restored_after_external_logging_setup() -> None:
    source = inspect.getsource(main)
    logging_setup = source.index("configure_wandb_step_metric(wandb_run)")
    rng_restore = source.index("restore_hf_restart_rng(")
    train_loader = source.index("train_loader = sft_data_generator_bos_bestfit(")
    assert logging_setup < rng_restore < train_loader


def test_restart_rank_local_checks_are_aligned_before_later_collectives() -> None:
    source = inspect.getsource(main)
    for stage, operation in (
        ("HF restart checkpoint validation", "hf_restart_descriptor = validate_hf_restart("),
        ("HF restart optimizer restoration", "optimizer.load_state_dict("),
        ("HF restart runtime validation", "runtime = validate_hf_restart_runtime("),
        ("HF restart RNG restoration", "restore_hf_restart_rng("),
        ("HF restart state capture", "runtime_state = capture_hf_restart_runtime("),
    ):
        stage_position = source.index(f'rank_synchronized_stage("{stage}"')
        operation_position = source.index(operation, stage_position)
        assert stage_position < operation_position


def test_profiler_artifact_failures_are_aligned_before_training_continues() -> None:
    profile_source = inspect.getsource(_maybe_profile_hf_step)
    start_stage = profile_source.index('rank_synchronized_stage(f"HF profiler start step')
    assert start_stage < profile_source.index("profiler_context.__enter__()", start_stage)
    trace_publication = profile_source.index('f"HF profiler publication step')
    assert trace_publication < profile_source.index("profiler_context.__exit__", trace_publication)
    assert trace_publication < profile_source.index(
        "profiler.export_chrome_trace", trace_publication
    )

    main_source = inspect.getsource(main)
    publication_stage = main_source.index('"HF profiling evidence publication"')
    assert publication_stage < main_source.index("timing_recorder.to_csv(", publication_stage)
    assert publication_stage < main_source.index("startup_trace.finish(", publication_stage)


def test_startup_clock_begins_before_distributed_compute_initialization() -> None:
    source = inspect.getsource(main)
    assert source.index('StartupTrace(event_prefix="chat_sft")') < source.index(
        "compute_init(device_type)"
    )


def test_distributed_directory_probe_is_strict_in_world_one(tmp_path: Path) -> None:
    candidate = tmp_path / "step_000002"
    assert _distributed_directory_exists(candidate, device="cpu") is False
    candidate.mkdir()
    assert _distributed_directory_exists(candidate, device="cpu") is True
    candidate.rmdir()
    candidate.write_text("not a checkpoint directory", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected a directory"):
        _distributed_directory_exists(candidate, device="cpu")


@pytest.mark.parametrize(
    ("config_name", "model_id", "revision"),
    [
        (
            "chat_sft_pythia_1_4b_h200_4x8_benchmark",
            "EleutherAI/pythia-1.4b",
            "fedc38a16eea3bd36a96b906d78d11d2ce18ed79",
        ),
        (
            "chat_sft_pythia_2_8b_h200_4x8_benchmark",
            "EleutherAI/pythia-2.8b",
            "2a259cdd96a4beb1cdf467512e3904197345f6a9",
        ),
        (
            "chat_sft_smollm3_3b_h200_4x8_benchmark",
            "HuggingFaceTB/SmolLM3-3B-checkpoints",
            "524d818535ee98e56a4f0d621c84dd5c8fb8d3d4",
        ),
    ],
)
def test_h200_canaries_pin_the_same_w32_geometry_and_safe_baseline(
    config_name: str,
    model_id: str,
    revision: str,
) -> None:
    cfg = _composed(config_name)

    assert cfg["num_iterations"] == 60
    assert cfg["device_batch_size"] == 1
    assert cfg["max_seq_len"] == 1024
    assert cfg["total_batch_size"] == 524288
    assert cfg["total_batch_size"] // (32 * 1 * 1024) == 16
    assert cfg["eval_every"] == 0
    assert cfg["save_every"] == 0
    assert cfg["dry_run"] is True
    assert cfg["huggingface"] == {
        "model_id": model_id,
        "revision": revision,
        "learning_rate": 2e-5,
        "adam_betas": [0.8, 0.95],
        "adam_eps": 1e-10,
        "gradient_checkpointing": True,
        "attention_implementation": "sdpa",
        "distributed_implementation": "dist_muon_adamw",
        "torch_compile": False,
        "restart_from_step": -1,
        "restart_save_every": 0,
        "data_snapshot_sha256": None,
        "output_dir": None,
        "validate_export": False,
    }
    validate_decoder_sft_config(cfg, world_size=32)


def _profile_cfg(**overrides):
    payload = {
        "num_iterations": 60,
        "profile_timing": False,
        "profile_trace_steps": [],
        "profile_trace_ranks": [],
        "profile_output_dir": "out/profile/chat_sft",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_profile_defaults_are_disabled_and_side_effect_free() -> None:
    settings = _resolve_hf_profile_settings(_profile_cfg(), world_size=32)
    assert settings.timing is False
    assert settings.trace_steps == frozenset()
    assert settings.trace_ranks == frozenset()
    with patch("torch.profiler.profile", side_effect=AssertionError("must remain off")):
        with _maybe_profile_hf_step(settings, completed_step=12, rank=0, device="cpu"):
            pass


def test_profile_settings_accept_real_hydra_list_configs() -> None:
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="chat_sft_pythia_2_8b")
    settings = _resolve_hf_profile_settings(cfg, world_size=32)
    assert settings.trace_steps == settings.trace_ranks == frozenset()


def test_paid_step_recorder_keeps_the_exact_outer_measurement() -> None:
    recorder = TimingRecorder()

    _record_hf_paid_step(
        recorder,
        rank=7,
        completed_step=19,
        step_s=0.125,
        scheduled_tokens=1024,
        nonpadding_tokens=900,
        supervised_tokens=731,
        loss=2.5,
        algorithmic_mfu_percent=42.0,
        peak_allocated_bytes=123456,
    )

    assert recorder.rows() == [
        {
            "name": "chat_sft/paid_step",
            "elapsed_ns": 125_000_000,
            "rank": 7,
            "step": 19,
            "scheduled_tokens": 1024,
            "nonpadding_tokens": 900,
            "supervised_tokens": 731,
            "loss": 2.5,
            "scheduled_tok_per_sec": 8192.0,
            "nonpadding_tok_per_sec": 7200.0,
            "supervised_tok_per_sec": 5848.0,
            "algorithmic_mfu_percent": 42.0,
            "peak_allocated_bytes": 123456,
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"profile_timing": 1}, "profile_timing"),
        ({"profile_trace_steps": "12"}, "profile_trace_steps"),
        ({"profile_trace_steps": [0]}, "entries"),
        ({"profile_trace_steps": [12, 12]}, "duplicates"),
        ({"profile_trace_ranks": [-1]}, "entries"),
        ({"profile_trace_ranks": [32]}, "world_size"),
        (
            {
                "profile_trace_steps": [61],
                "profile_trace_ranks": [0],
            },
            "optimizer-step horizon",
        ),
        ({"profile_trace_steps": [12]}, "both be"),
        ({"profile_trace_ranks": [0]}, "both be"),
        ({"profile_output_dir": ""}, "profile_output_dir"),
    ],
)
def test_profile_schema_and_inert_combinations_fail_closed(
    overrides: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_hf_profile_settings(_profile_cfg(**overrides), world_size=32)


def test_profiler_enters_only_for_exact_selected_rank_and_step(tmp_path: Path) -> None:
    settings = _resolve_hf_profile_settings(
        _profile_cfg(
            profile_trace_steps=[12, 14],
            profile_trace_ranks=[0, 8],
            profile_output_dir=str(tmp_path),
        ),
        world_size=32,
    )
    entered: list[tuple[int, int]] = []

    class _FakeProfiler:
        def __enter__(self):
            entered.append((0, 0))
            return self

        def __exit__(self, *_args):
            return False

        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text("trace", encoding="utf-8")

    with patch("torch.profiler.profile", return_value=_FakeProfiler()):
        with _maybe_profile_hf_step(settings, completed_step=11, rank=0, device="cpu"):
            pass
        with _maybe_profile_hf_step(settings, completed_step=12, rank=1, device="cpu"):
            pass
        with _maybe_profile_hf_step(settings, completed_step=12, rank=0, device="cpu"):
            pass
    assert entered == [(0, 0)]
    assert (tmp_path / "chat_sft_trace_rank0_step12.json").read_text() == "trace"


def _distributed_profiler_fault_worker(
    rank: int,
    init_file: str,
    output_dir: str,
    fault: str,
    queue,
) -> None:
    body_entered = False

    class _FaultProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def export_chrome_trace(self, _path: str) -> None:
            if fault == "publication":
                raise RuntimeError("profiler publication fault")

    def profile_factory(**_kwargs):
        if fault == "start":
            raise RuntimeError("profiler start fault")
        return _FaultProfiler()

    try:
        t.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        settings = _resolve_hf_profile_settings(
            _profile_cfg(
                profile_trace_steps=[1],
                profile_trace_ranks=[1],
                profile_output_dir=output_dir,
            ),
            world_size=2,
        )
        with patch("torch.profiler.profile", side_effect=profile_factory):
            with _maybe_profile_hf_step(
                settings,
                completed_step=1,
                rank=rank,
                device="cpu",
            ):
                body_entered = True
        queue.put(("unexpected-success", rank, body_entered))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put((type(exc).__name__, rank, str(exc), body_entered))
    finally:
        if t.distributed.is_initialized():
            t.distributed.destroy_process_group()


@pytest.mark.skipif(
    not t.distributed.is_available() or not t.distributed.is_gloo_available(),
    reason="Gloo is unavailable",
)
@pytest.mark.parametrize(
    ("fault", "body_entered"),
    [("start", False), ("publication", True)],
)
def test_rank_local_profiler_faults_abort_every_rank_at_the_same_boundary(
    tmp_path: Path,
    fault: str,
    body_entered: bool,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / f"profiler-{fault}-gloo")
    processes = [
        context.Process(
            target=_distributed_profiler_fault_worker,
            args=(rank, init_file, str(tmp_path / "traces"), fault, queue),
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
    assert all(f"profiler {fault} fault" in result[2] for result in results)
    assert {result[3] for result in results} == {body_entered}


def test_startup_prefix_is_customizable_without_changing_default_schema() -> None:
    default_lines: list[str] = []
    default = StartupTrace(clock=Mock(side_effect=[0.0, 1.0, 2.0, 3.0]), emit=default_lines.append)
    with default.phase("load"):
        pass
    default.finish(status="no_training_step")
    assert default_lines[0].startswith("encdec_startup_phase")
    assert default_lines[-1].startswith("encdec_startup_summary")

    custom_lines: list[str] = []
    custom = StartupTrace(
        clock=Mock(side_effect=[0.0, 1.0, 2.0, 3.0]),
        emit=custom_lines.append,
        event_prefix="chat_sft",
    )
    with custom.phase("load"):
        pass
    custom.finish(status="no_training_step")
    assert custom_lines[0].startswith("chat_sft_startup_phase")
    assert custom_lines[-1].startswith("chat_sft_startup_summary")
    for bad in ("", "has space", "bad-hyphen"):
        with pytest.raises(ValueError, match="event_prefix"):
            StartupTrace(event_prefix=bad)


class _TinyModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.arange(12, dtype=t.float32).reshape(4, 3))
        self.bias = t.nn.Parameter(t.zeros(4))


@pytest.mark.parametrize("implementation", list(HFDistributedImplementation))
def test_optimizer_strategy_has_one_exact_group_and_plain_adamw_parity(
    implementation: HFDistributedImplementation,
) -> None:
    # The vendored fused AdamW step is shape-specialized. Each parameterized
    # strategy must start independently of TorchDynamo's process-global cache.
    t.compiler.reset()
    model = _TinyModel()
    reference = copy.deepcopy(model)
    optimizer = build_hf_optimizer(
        model,
        ddp=False,
        learning_rate=2e-4,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        init_lr_frac=0.5,
        distributed_implementation=implementation,
    )
    reference_optimizer = t.optim.AdamW(
        reference.parameters(),
        lr=1e-4,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["kind"] == "adamw"
    assert group["initial_lr"] == group["lr"] == 1e-4
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        gradient = t.linspace(-1, 1, parameter.numel()).reshape_as(parameter)
        parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
    optimizer.step()
    reference_optimizer.step()
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert t.allclose(parameter, reference_parameter, atol=1e-7, rtol=1e-6)


def test_prepare_training_model_applies_compile_before_ddp() -> None:
    model = _TinyModel()
    compiled = Mock(spec=t.nn.Module)
    ddp_wrapped = Mock(spec=t.nn.Module)
    events: list[str] = []

    def fake_compile(module, *, dynamic):
        assert module is model
        assert dynamic is None
        events.append("compile")
        return compiled

    def fake_ddp(module, **kwargs):
        assert module is compiled
        events.append("ddp")
        assert kwargs == {
            "device_ids": [3],
            "output_device": 3,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
            "find_unused_parameters": False,
        }
        return ddp_wrapped

    with (
        patch("torch.compile", side_effect=fake_compile),
        patch("torch.nn.parallel.DistributedDataParallel", side_effect=fake_ddp),
    ):
        actual = prepare_hf_training_model(
            model,
            ddp=True,
            local_rank=3,
            distributed_implementation=HFDistributedImplementation.DDP_FUSED_ADAMW,
            torch_compile=True,
        )
    assert actual is ddp_wrapped
    assert events == ["compile", "ddp"]


def test_dist_strategy_never_wraps_ddp_and_eager_path_is_identity() -> None:
    model = _TinyModel()
    with patch(
        "torch.nn.parallel.DistributedDataParallel",
        side_effect=AssertionError("must not wrap"),
    ):
        actual = prepare_hf_training_model(
            model,
            ddp=True,
            local_rank=0,
            distributed_implementation=HFDistributedImplementation.DIST_MUON_ADAMW,
            torch_compile=False,
        )
    assert actual is model


def test_local_world_size_requires_explicit_distributed_geometry(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert validated_local_world_size(local_rank=0, device="cpu") == 1
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert validated_local_world_size(local_rank=3, device="cpu") == 8
    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(ValueError, match="disagrees"):
        validated_local_world_size(local_rank=3, device="cpu")


def _local_world_geometry_worker(
    rank: int,
    init_file: str,
    rank_one_value: str,
    queue,
) -> None:
    try:
        t.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        os.environ["LOCAL_WORLD_SIZE"] = "2" if rank == 0 else rank_one_value
        os.environ["LOCAL_RANK"] = str(rank)
        result = validated_local_world_size(local_rank=rank, device="cpu")
        queue.put(("ok", rank, result))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        queue.put((type(exc).__name__, rank, str(exc)))
    finally:
        if t.distributed.is_initialized():
            t.distributed.destroy_process_group()


@pytest.mark.skipif(
    not t.distributed.is_available() or not t.distributed.is_gloo_available(),
    reason="Gloo is unavailable",
)
@pytest.mark.parametrize(
    ("rank_one_value", "exception_type", "message"),
    [
        ("invalid", "AlignedRankFailure", "LOCAL_WORLD_SIZE"),
        ("3", "ValueError", "differs across distributed ranks"),
    ],
)
def test_rank_asymmetric_local_world_geometry_fails_every_rank_without_hanging(
    tmp_path: Path,
    rank_one_value: str,
    exception_type: str,
    message: str,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "local-world-gloo")
    processes = [
        context.Process(
            target=_local_world_geometry_worker,
            args=(rank, init_file, rank_one_value, queue),
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
    assert {result[0] for result in results} == {exception_type}
    assert all(message in result[2] for result in results)
