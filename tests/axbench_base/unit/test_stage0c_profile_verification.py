"""Tests for the Stage 0c verification profiler guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from scripts.benchmark import profile_stage0c_verification as profile_mod
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cluster_config import axbench_base_config_to_dict
from tests.axbench_base.unit.test_config import make_valid_config


def test_profile_stage0c_verification_requires_serial_prefix_mode() -> None:
    cfg = make_valid_config(verification_cross_prefix_batch_size=4)

    with pytest.raises(
        AxBenchBaseError,
        match="verification_cross_prefix_batch_size=1",
    ):
        profile_mod._require_serial_prefix_profile_mode(cfg)


def test_profile_stage0c_verification_accepts_serial_prefix_mode() -> None:
    cfg = make_valid_config(verification_cross_prefix_batch_size=1)

    profile_mod._require_serial_prefix_profile_mode(cfg)


def test_profile_stage0c_rejects_legacy_shadow_before_environment_or_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_tag = "stage0c-profile-shadow"
    cfg = make_valid_config(
        output_root=str(tmp_path / "custom"),
        run_tag=run_tag,
        stage="0c",
        verification_cross_prefix_batch_size=1,
    )
    (tmp_path / "eval/axbench_base" / run_tag).mkdir(parents=True)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(profile_mod, "cfg_to_dataclass", lambda _cfg: cfg)
    monkeypatch.setattr(
        profile_mod,
        "validate_env_for_stages",
        lambda *_args, **_kwargs: pytest.fail(
            "environment validation ran before legacy-shadow preflight"
        ),
    )
    monkeypatch.setattr(
        profile_mod,
        "compute_init",
        lambda **_kwargs: pytest.fail(
            "compute initialization ran before legacy-shadow preflight"
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Refusing to start a second AxBench-Base artifact tree",
    ):
        profile_mod._run(OmegaConf.create({}))

    assert not (tmp_path / "custom" / run_tag).exists()


def test_profile_stage0c_resolves_custom_profile_dir_and_cleans_up_init_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "custom"),
        run_tag="stage0c-profile-cleanup",
        stage="0c",
        verification_cross_prefix_batch_size=1,
    )
    cleanup_calls: list[str] = []
    seen_runtime_state: dict[str, Any] = {}

    def fail_runtime_state(**kwargs: Any) -> None:
        seen_runtime_state.update(kwargs)
        raise RuntimeError("injected runtime-state construction failure")

    monkeypatch.setattr(profile_mod, "cfg_to_dataclass", lambda _cfg: cfg)
    monkeypatch.setattr(
        profile_mod,
        "validate_env_for_stages",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(profile_mod, "compute_init", lambda **_kwargs: None)
    monkeypatch.setattr(
        profile_mod.axbench_cli,
        "_resolve_ddp_coordinates",
        lambda: (0, 1),
    )
    monkeypatch.setattr(profile_mod, "RuntimeState", fail_runtime_state)
    monkeypatch.setattr(
        profile_mod,
        "compute_cleanup",
        lambda: cleanup_calls.append("cleanup"),
    )
    real_profile_state = profile_mod._ProfileState
    output_dirs: list[Path] = []

    def capture_profile_state(**kwargs: Any) -> Any:
        output_dirs.append(kwargs["output_dir"])
        return real_profile_state(**kwargs)

    monkeypatch.setattr(profile_mod, "_ProfileState", capture_profile_state)

    with pytest.raises(
        RuntimeError,
        match="injected runtime-state construction failure",
    ):
        profile_mod._run(OmegaConf.create({}))

    assert output_dirs == [
        tmp_path
        / "custom"
        / "stage0c-profile-cleanup"
        / "profile"
        / "stage0c_verify_candidate"
    ]
    assert seen_runtime_state["cfg"] == cfg
    assert seen_runtime_state["ddp_rank"] == 0
    assert seen_runtime_state["ddp_world_size"] == 1
    assert cleanup_calls == ["cleanup"]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_profile_stage0c_success_path_restores_every_resource_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cuda_available: bool,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "custom"),
        run_tag="stage0c-profile-success",
        stage="0c",
        verification_cross_prefix_batch_size=1,
    )
    events: list[str] = []
    profiler_kwargs: dict[str, Any] = {}
    report_kwargs: dict[str, Any] = {}

    class FakeRuntimeState:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"cfg": cfg, "ddp_rank": 0, "ddp_world_size": 1}
            events.append("runtime.init")

        def close(self) -> None:
            events.append("runtime.close")

    class FakeWrappers:
        def __init__(self, state: Any) -> None:
            assert state.output_dir == (
                tmp_path
                / "custom"
                / "stage0c-profile-success"
                / "profile"
                / "stage0c_verify_candidate"
            )

        def install(self) -> None:
            events.append("wrappers.install")

        def restore(self) -> None:
            events.append("wrappers.restore")

    class FakeCProfiler:
        def enable(self) -> None:
            events.append("cprofile.enable")

        def disable(self) -> None:
            events.append("cprofile.disable")

    class FakeTorchProfiler:
        def __enter__(self) -> object:
            events.append("torch_profile.enter")
            return self

        def __exit__(self, *args: Any) -> None:
            del args
            events.append("torch_profile.exit")

    def fake_torch_profile(**kwargs: Any) -> FakeTorchProfiler:
        profiler_kwargs.update(kwargs)
        return FakeTorchProfiler()

    def stop_after_limit(_cfg: Any, _state: Any) -> None:
        events.append("dispatch")
        raise profile_mod._Stage0cProfileComplete

    def capture_report(**kwargs: Any) -> None:
        report_kwargs.update(kwargs)
        events.append("report")

    monkeypatch.setattr(profile_mod, "cfg_to_dataclass", lambda _cfg: cfg)
    monkeypatch.setattr(
        profile_mod,
        "validate_env_for_stages",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        profile_mod,
        "compute_init",
        lambda **_kwargs: events.append("compute.init"),
    )
    monkeypatch.setattr(
        profile_mod,
        "compute_cleanup",
        lambda: events.append("compute.cleanup"),
    )
    monkeypatch.setattr(
        profile_mod.axbench_cli,
        "_resolve_ddp_coordinates",
        lambda: (0, 1),
    )
    monkeypatch.setattr(
        profile_mod.axbench_cli,
        "_dispatch_stage",
        stop_after_limit,
    )
    monkeypatch.setattr(profile_mod, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(profile_mod, "_InstalledWrappers", FakeWrappers)
    monkeypatch.setattr(profile_mod.cProfile, "Profile", FakeCProfiler)
    monkeypatch.setattr(profile_mod, "torch_profile", fake_torch_profile)
    monkeypatch.setattr(
        profile_mod.t.cuda,
        "is_available",
        lambda: cuda_available,
    )
    monkeypatch.setattr(profile_mod, "_write_reports", capture_report)

    profile_mod._run(OmegaConf.create({}))

    expected_activities = [profile_mod.ProfilerActivity.CPU]
    if cuda_available:
        expected_activities.append(profile_mod.ProfilerActivity.CUDA)
    assert profiler_kwargs == {
        "activities": expected_activities,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
    }
    assert report_kwargs["cfg"] == cfg
    assert report_kwargs["status"] == "stopped_after_limit"
    assert events == [
        "compute.init",
        "runtime.init",
        "wrappers.install",
        "torch_profile.enter",
        "cprofile.enable",
        "dispatch",
        "cprofile.disable",
        "torch_profile.exit",
        "wrappers.restore",
        "runtime.close",
        "report",
        "compute.cleanup",
    ]


@pytest.mark.parametrize(
    "conditioning",
    [
        {"feature_mask": None},
        {"feature_mask": None, "feature_clusters": None},
        {
            "feature_mask": None,
            "feature_clusters": {"path": "clusters.yaml"},
        },
    ],
)
def test_summary_metadata_uses_cluster_aware_config_serialization(
    tmp_path: Path,
    conditioning: dict[str, Any],
) -> None:
    cfg = make_valid_config(conditioning=conditioning)
    assert isinstance(cfg, AxBenchBaseConfig)
    state = profile_mod._ProfileState(verify_limit=1, output_dir=tmp_path)

    payload = profile_mod._build_summary_payload(
        cfg=cfg,
        state=state,
        status="ok",
        torch_trace_path=tmp_path / "missing-trace.json",
        cprofile_path=tmp_path / "cprofile.txt",
    )

    config_payload = payload["config"]
    payload_conditioning = config_payload["conditioning"]
    if "feature_clusters" not in conditioning:
        assert config_payload == axbench_base_config_to_dict(cfg)
    elif conditioning["feature_clusters"] is None:
        assert "feature_clusters" not in payload_conditioning
    else:
        assert payload_conditioning["feature_clusters"] == {
            "path": "clusters.yaml"
        }
