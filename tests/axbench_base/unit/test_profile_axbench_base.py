"""Unit tests for the profiled AxBench-Base CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

import scripts.benchmark.profile_axbench_base as profile_cli
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.profiler import get_current_profiler
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.types import AxBenchBaseError, Stage
from steering.eval.cluster_config import axbench_base_config_to_dict
from tests.axbench_base import paths as test_paths


def _minimal_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "feature_dir": (
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/"
            "m_gemma-2-2b/base"
        ),
        "feature_statistics_path": "out/data/features/test/stats.safetensors",
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cpu",
        "output_root": None,
        "run_tag": "profile-test",
        "stage": "0a",
        "run_negative_mode": True,
        "concept_examples_qc_target_surviving": 5,
        "concept_description_fetch_limit": 50,
        "verification_candidate_scan_limit": 1000,
        "verification_candidate_attempt_limit": 300,
    }
    base.update(overrides)
    return base


def _dict_cfg(**overrides: Any) -> DictConfig:
    return OmegaConf.create(_minimal_dict(**overrides))


@pytest.fixture(autouse=True)
def _pin_nanochat_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))


def test_profile_run_generates_tag_validates_dispatches_and_writes_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    captured_states: list[Any] = []

    class FakeRuntimeState:
        def __init__(self, cfg: AxBenchBaseConfig) -> None:
            self.cfg = cfg
            self.closed = False
            captured_states.append(self)
            events.append(("state_init", cfg.run_tag or ""))

        def close(self) -> None:
            self.closed = True
            events.append(("state_close", self.cfg.run_tag or ""))

    def fake_generate(cfg: AxBenchBaseConfig) -> str:
        events.append(("generate", cfg.run_tag or ""))
        return "profile-generated"

    def fake_validate_env(
        stages: list[Stage],
        *,
        cfg: AxBenchBaseConfig | None = None,
    ) -> None:
        assert get_current_profiler() is None
        assert stages == [Stage.PREPARE_CONCEPTS]
        assert cfg is not None
        events.append(("validate_env", cfg.run_tag or ""))

    def fake_dispatch(cfg: AxBenchBaseConfig, state: RuntimeState) -> None:
        assert get_current_profiler() is not None
        events.append(("dispatch", cfg.run_tag or ""))
        assert state is captured_states[-1]

    monkeypatch.setattr(profile_cli.axbench_cli, "_generate_run_tag", fake_generate)
    monkeypatch.setattr(profile_cli, "validate_env_for_stages", fake_validate_env)
    monkeypatch.setattr(profile_cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(profile_cli.axbench_cli, "_dispatch_stage", fake_dispatch)

    custom_root = tmp_path / "custom-profile-root"
    profile_cli._run(_dict_cfg(run_tag=None, output_root=str(custom_root)))

    assert events == [
        ("generate", ""),
        ("validate_env", "profile-generated"),
        ("state_init", "profile-generated"),
        ("dispatch", "profile-generated"),
        ("state_close", "profile-generated"),
    ]
    assert captured_states[0].closed is True
    assert get_current_profiler() is None

    output_paths = paths.AxBenchBasePaths(custom_root, "profile-generated")
    profile_dir = output_paths.profile_dir()
    assert (profile_dir / "events.jsonl").is_file()
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    metadata = summary["metadata"]
    assert summary["status"] == "ok"
    assert metadata["run_tag"] == "profile-generated"
    assert metadata["stages"] == ["0a"]
    assert metadata["configured_output_root"] == str(custom_root)
    assert metadata["effective_output_root"] == str(custom_root)
    assert metadata["run_output_root"] == str(output_paths.run_output_root())
    assert metadata["concept_examples_qc_target_surviving"] == 5
    assert metadata["concept_description_fetch_limit"] == 50
    assert metadata["verification_candidate_scan_limit"] == 1000
    assert metadata["verification_candidate_attempt_limit"] == 300


def test_profile_run_rejects_legacy_shadow_before_environment_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_tag = "profile-shadow"
    (tmp_path / "eval/axbench_base" / run_tag).mkdir(parents=True)
    monkeypatch.setattr(
        profile_cli,
        "validate_env_for_stages",
        lambda *_args, **_kwargs: pytest.fail(
            "environment validation ran before legacy-shadow preflight"
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Refusing to start a second AxBench-Base artifact tree",
    ):
        profile_cli._run(
            _dict_cfg(
                run_tag=run_tag,
                output_root=str(tmp_path / "custom"),
            )
        )

    assert not (tmp_path / "custom" / run_tag).exists()


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
def test_profile_metadata_uses_cluster_aware_config_serialization(
    monkeypatch: pytest.MonkeyPatch,
    conditioning: dict[str, Any],
) -> None:
    captured_metadata: list[dict[str, Any]] = []
    captured_configs: list[AxBenchBaseConfig] = []

    class CapturingProfiler:
        def __init__(
            self,
            _profile_dir: Path,
            *,
            metadata: dict[str, Any],
        ) -> None:
            captured_metadata.append(metadata)

        def __enter__(self) -> CapturingProfiler:
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    class FakeRuntimeState:
        def __init__(self, cfg: AxBenchBaseConfig) -> None:
            captured_configs.append(cfg)

        def close(self) -> None:
            pass

    monkeypatch.setattr(profile_cli, "AxBenchProfiler", CapturingProfiler)
    monkeypatch.setattr(profile_cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(profile_cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        profile_cli,
        "validate_env_for_stages",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        profile_cli.axbench_cli,
        "_dispatch_stage",
        lambda _cfg, _state: None,
    )

    profile_cli._run(
        _dict_cfg(
            conditioning=conditioning,
            run_tag="cluster-aware-profile-metadata",
        )
    )

    assert len(captured_metadata) == len(captured_configs) == 1
    cfg = captured_configs[0]
    assert isinstance(cfg, AxBenchBaseConfig)
    config_payload = captured_metadata[0]["config"]
    payload_conditioning = config_payload["conditioning"]

    if "feature_clusters" not in conditioning:
        assert config_payload == axbench_base_config_to_dict(cfg)
    elif conditioning["feature_clusters"] is None:
        assert "feature_clusters" not in payload_conditioning
    else:
        assert payload_conditioning["feature_clusters"] == {
            "path": "clusters.yaml"
        }


def test_profile_run_closes_state_and_writes_error_summary_on_dispatch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_states: list[Any] = []

    class FakeRuntimeState:
        def __init__(self, cfg: AxBenchBaseConfig) -> None:
            self.cfg = cfg
            self.closed = False
            captured_states.append(self)

        def close(self) -> None:
            self.closed = True

    def fake_dispatch(cfg: AxBenchBaseConfig, state: RuntimeState) -> None:
        assert get_current_profiler() is not None
        raise RuntimeError("dispatch exploded")

    monkeypatch.setattr(profile_cli, "validate_env_for_stages", lambda *a, **k: None)
    monkeypatch.setattr(profile_cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(profile_cli.axbench_cli, "_dispatch_stage", fake_dispatch)

    with pytest.raises(RuntimeError, match="dispatch exploded"):
        profile_cli._run(_dict_cfg(run_tag="profile-failure"))

    assert captured_states and captured_states[0].closed is True
    assert get_current_profiler() is None
    profile_dir = test_paths.default_paths("profile-failure").run_output_root() / "profile"
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    assert summary["status"] == "error"
