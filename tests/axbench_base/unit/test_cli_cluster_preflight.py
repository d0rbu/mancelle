"""Focused tests for clustered AxBench-Base CLI preflight orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import scripts.eval.axbench_base_eval as cli
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContractRequest,
)
from tests.axbench_base.unit.test_config import make_valid_config

_MASK = {"mode": "whitelist", "path": "/mask-sentinel.txt"}
_CLUSTERS = {"path": "/clusters-sentinel.yaml"}


def _clustered_config(
    *,
    step: object = 17,
    conditioning: dict[str, Any] | None = None,
    encdec: dict[str, Any] | None = None,
    output_root: str | None = None,
    run_tag: str = "test",
) -> AxBenchBaseConfig:
    return make_valid_config(
        feature_dir="feature-dir-sentinel",
        feature_statistics_path="statistics-sentinel",
        concept_feature_whitelist_path="concept-whitelist-sentinel",
        conditioning=(
            {
                "feature_mask": dict(_MASK),
                "feature_clusters": dict(_CLUSTERS),
            }
            if conditioning is None
            else conditioning
        ),
        encdec=(
            {
                "source": "checkpoint-source-sentinel",
                "model_tag": "checkpoint-model-tag-sentinel",
                "step": step,
            }
            if encdec is None
            else encdec
        ),
        output_root=output_root,
        run_tag=run_tag,
    )


def test_cluster_contract_request_no_conditioning_is_passthrough() -> None:
    cfg = make_valid_config(conditioning=None, encdec={})

    assert cli._cluster_contract_request(cfg) is None


def test_cluster_contract_request_null_feature_clusters_is_passthrough() -> None:
    cfg = make_valid_config(
        conditioning={
            "feature_mask": dict(_MASK),
            "feature_clusters": None,
        },
        encdec={},
    )

    assert cli._cluster_contract_request(cfg) is None


def test_cluster_contract_request_requires_feature_mask() -> None:
    cfg = _clustered_config(
        conditioning={"feature_clusters": dict(_CLUSTERS)},
    )

    with pytest.raises(
        AxBenchBaseError,
        match="requires feature_mask and feature_clusters mappings",
    ):
        cli._cluster_contract_request(cfg)


def test_cluster_contract_request_maps_every_field_exactly() -> None:
    cfg = _clustered_config()

    assert cli._cluster_contract_request(cfg) == ClusterFeatureContractRequest(
        feature_dir="feature-dir-sentinel",
        feature_statistics_path="statistics-sentinel",
        feature_mask=_MASK,
        feature_clusters=_CLUSTERS,
        checkpoint_source="checkpoint-source-sentinel",
        checkpoint_model_tag="checkpoint-model-tag-sentinel",
        checkpoint_step=17,
        checkpoint_tag="checkpoint-model-tag-sentinel",
        concept_feature_whitelist_path="concept-whitelist-sentinel",
    )


def test_cluster_contract_request_maps_last_step_to_none() -> None:
    request = cli._cluster_contract_request(_clustered_config(step="last"))

    assert request is not None
    assert request.checkpoint_step is None


def test_cluster_contract_request_propagates_forced_shard_revalidation() -> None:
    request = cli._cluster_contract_request(
        replace(
            _clustered_config(),
            force_cluster_shard_revalidation=True,
        )
    )

    assert request is not None
    assert request.force_cluster_shard_revalidation is True


@pytest.mark.parametrize("raw_step", [3.5, "17"])
def test_cluster_contract_request_rejects_bad_step_type(
    raw_step: object,
) -> None:
    with pytest.raises(AxBenchBaseError, match="step must be"):
        cli._cluster_contract_request(_clustered_config(step=raw_step))


def test_cluster_contract_request_rejects_bool_step() -> None:
    with pytest.raises(AxBenchBaseError, match="step must be"):
        cli._cluster_contract_request(_clustered_config(step=True))


def test_cluster_contract_request_rejects_negative_step() -> None:
    with pytest.raises(AxBenchBaseError, match="step must be non-negative"):
        cli._cluster_contract_request(_clustered_config(step=-1))


@pytest.mark.parametrize("source", [None, ""])
def test_cluster_contract_request_requires_encdec_source(
    source: object,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"encdec\.source must be a non-empty string",
    ):
        cli._cluster_contract_request(
            _clustered_config(
                encdec={
                    "source": source,
                    "model_tag": "checkpoint-model-tag-sentinel",
                    "step": 17,
                }
            )
        )


def test_clustered_run_installs_marker_before_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live CLI path resolves the contract and installs its root marker."""

    cfg_obj = _clustered_config()
    cfg = OmegaConf.create(asdict(cfg_obj))
    contract = SimpleNamespace(
        checkpoint=SimpleNamespace(step=23),
        feature_contract_fingerprint="a" * 64,
    )
    resolved_requests: list[ClusterFeatureContractRequest] = []

    def _resolve(request: ClusterFeatureContractRequest) -> Any:
        resolved_requests.append(request)
        return contract

    marker = MagicMock()
    runtime_state = MagicMock()
    runtime_factory = MagicMock(return_value=runtime_state)
    events: list[str] = []

    def _install(*args: object, **kwargs: object) -> None:
        events.append("marker")
        marker(*args, **kwargs)

    def _runtime(*args: object, **kwargs: object) -> Any:
        events.append("runtime")
        return runtime_factory(*args, **kwargs)

    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        cli, "validate_env_for_stages", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli, "resolve_cluster_feature_contract_ddp", _resolve,
    )
    monkeypatch.setattr(
        cli, "install_or_validate_cluster_root_marker_ddp", _install,
    )
    monkeypatch.setattr(cli, "RuntimeState", _runtime)
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda **_kwargs: (True, 2, 1, 4, "cpu-device"),
    )
    monkeypatch.setattr(cli, "_resolve_ddp_coordinates", lambda: (2, 4))
    monkeypatch.setattr(
        cli, "resolve_profile_gpu_device_index", lambda *_args, **_kwargs: None,
    )
    cleanup = MagicMock()
    monkeypatch.setattr(cli, "compute_cleanup", cleanup)
    dispatch = MagicMock()
    monkeypatch.setattr(cli, "_dispatch_stage", dispatch)

    cli._run(cfg)

    assert len(resolved_requests) == 1
    assert resolved_requests[0].checkpoint_step == 17
    assert events == ["marker", "runtime"]
    marker.assert_called_once()
    marker_cfg, marker_contract = marker.call_args.args
    assert marker_cfg.encdec["step"] == 23
    assert marker_contract is contract
    assert marker.call_args.kwargs == {"rank": 2, "world_size": 4}
    runtime_factory.assert_called_once()
    assert runtime_factory.call_args.kwargs["cfg"] is marker_cfg
    assert runtime_factory.call_args.kwargs["cluster_contract"] is contract
    assert runtime_factory.call_args.kwargs["ddp_rank"] == 2
    assert runtime_factory.call_args.kwargs["ddp_world_size"] == 4
    dispatch.assert_called_once_with(marker_cfg, runtime_state)
    runtime_state.close.assert_called_once_with()
    cleanup.assert_called_once_with()


def test_explicit_clustered_legacy_shadow_fails_before_compute_or_contract_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_tag = "historical-clustered-run"
    custom_root = tmp_path / "custom"
    legacy_run = tmp_path / "eval/axbench_base" / run_tag
    legacy_run.mkdir(parents=True)
    cfg = OmegaConf.create(
        asdict(
            _clustered_config(
                output_root=str(custom_root),
                run_tag=run_tag,
            )
        )
    )
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    compute = MagicMock(
        side_effect=AssertionError(
            "compute_init ran before the legacy-shadow preflight"
        )
    )
    resolver = MagicMock(
        side_effect=AssertionError(
            "cluster contract resolution ran before the legacy-shadow preflight"
        )
    )
    monkeypatch.setattr(cli, "compute_init", compute)
    monkeypatch.setattr(cli, "resolve_cluster_feature_contract_ddp", resolver)

    with pytest.raises(
        AxBenchBaseError,
        match="Refusing to start a second AxBench-Base artifact tree",
    ):
        cli._run(cfg)

    compute.assert_not_called()
    resolver.assert_not_called()
    assert not (custom_root / run_tag).exists()


def test_unclustered_marker_check_is_local_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """U checks only the shared marker path and never enters the DDP wrapper."""

    cfg = make_valid_config(
        run_tag="unclustered-local-marker",
        conditioning=None,
        encdec={},
    )
    marker = tmp_path / "feature_contract.json"
    monkeypatch.setattr(
        cli.paths.AxBenchBasePaths,
        "feature_contract_json",
        lambda _self: marker,
    )
    ddp_wrapper = MagicMock(
        side_effect=AssertionError("unclustered marker check used a collective"),
    )
    monkeypatch.setattr(
        cli,
        "install_or_validate_cluster_root_marker_ddp",
        ddp_wrapper,
    )

    cli._install_cluster_marker_ddp(
        cfg,
        None,
        rank=3,
        world_size=8,
    )
    ddp_wrapper.assert_not_called()

    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(
        AxBenchBaseError,
        match="Unclustered run rejects cluster-stamped output",
    ):
        cli._install_cluster_marker_ddp(
            cfg,
            None,
            rank=3,
            world_size=8,
        )
    ddp_wrapper.assert_not_called()


def test_unclustered_run_checks_marker_before_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_obj = make_valid_config(
        run_tag="unclustered-stale-root",
        conditioning=None,
        encdec={},
    )
    cfg = OmegaConf.create(asdict(cfg_obj))
    marker = MagicMock(
        side_effect=AxBenchBaseError("stale unclustered root marker"),
    )
    runtime_factory = MagicMock()
    cleanup = MagicMock()
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        cli,
        "validate_env_for_stages",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(cli, "_install_cluster_marker_ddp", marker)
    monkeypatch.setattr(cli, "RuntimeState", runtime_factory)
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda **_kwargs: (True, 2, 1, 4, "cpu-device"),
    )
    monkeypatch.setattr(cli, "_resolve_ddp_coordinates", lambda: (2, 4))
    monkeypatch.setattr(cli, "compute_cleanup", cleanup)

    with pytest.raises(AxBenchBaseError, match="stale unclustered root marker"):
        cli._run(cfg)

    marker.assert_called_once()
    marker_cfg, marker_contract = marker.call_args.args
    assert marker_cfg.run_tag == "unclustered-stale-root"
    assert marker_contract is None
    assert marker.call_args.kwargs == {"rank": 2, "world_size": 4}
    runtime_factory.assert_not_called()
    cleanup.assert_called_once_with()
