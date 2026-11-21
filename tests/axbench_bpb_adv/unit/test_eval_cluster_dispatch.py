from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import scripts.eval.axbench_bpb_adv_eval as cli
import steering.eval.axbench_bpb_adv.source_validation as source_validation
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb_adv import config_loader
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    OursGcgConfig,
    ProfileConfig,
    Stage,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureContractRequest,
)
from tests.axbench_bpb_adv.source_run_builders import (
    SourceSeedTokenizer,
    both_class_rows,
    write_scalar_direction_baselines,
    write_stage_0_3_source_run,
)

_FINGERPRINT = "a" * 64


@contextmanager
def _capture_loguru_exception_records() -> Iterator[list[Mapping[str, Any]]]:
    records: list[Mapping[str, Any]] = []
    handler_id = logger.add(
        lambda message: records.append(message.record),
        level="ERROR",
    )
    try:
        yield records
    finally:
        logger.remove(handler_id)


def _assert_rank_zero_traceback(
    records: list[Mapping[str, Any]],
    *,
    message: str,
    exception_type: type[BaseException],
    exception_message: str,
    frame_name: str,
    cause_message: str | None = None,
) -> None:
    matches = [record for record in records if record["message"] == message]
    assert len(matches) == 1, f"expected one traceback record for {message!r}"
    exception = matches[0]["exception"]
    assert exception is not None
    assert exception.type is exception_type
    assert str(exception.value) == exception_message
    traceback = exception.traceback
    frame_names: list[str] = []
    while traceback is not None:
        frame_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert frame_name in frame_names
    if cause_message is not None:
        assert exception.value.__cause__ is not None
        assert str(exception.value.__cause__) == cause_message


class _SourceContract:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        fingerprint = payload["feature_contract_fingerprint"]
        self.feature_contract_fingerprint = None if fingerprint is None else str(fingerprint)

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def _validated_source(contract: _SourceContract, **updates: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "contract": contract,
        "concept_ids": frozenset({1}),
        "n_concepts": 1,
        "n_positive_synth_rows": 2,
        "n_negative_synth_rows": 2,
        "n_dropped": 0,
        "require_directions": lambda *args, **kwargs: None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _cluster_contract(*, step: int = 17) -> ClusterFeatureContract:
    return cast(
        ClusterFeatureContract,
        SimpleNamespace(
            feature_contract_fingerprint=_FINGERPRINT,
            checkpoint=SimpleNamespace(step=step),
        ),
    )


def _run_cfg(tmp_path: Path) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        stage=Stage.VALIDATE_SOURCE.value,
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        feature_mask=None,
        feature_clusters=None,
        # Read by build_cluster_contract_request; omitting them makes every
        # clustered dispatch test die on AttributeError before its assertions.
        selection_splice_batch_size=None,
        selection_manifest_path=None,
        base_methods=None,
        filter_unaccepted_samples=True,
        target_models=["encdec"],
        steering_metric="roundtrip",
        scores_path=None,
        encdec_target={"step": "latest"},
        ours_gcg=OursGcgConfig(top_k_pos=None, top_k_neg=None),
        profile=ProfileConfig(
            enabled=False,
            output_dir=None,
            synchronize_cuda=False,
            gpu_utilization=False,
            gpu_poll_interval_ms=50,
            gpu_idle_threshold=10.0,
        ),
    )


def _raw_cfg() -> DictConfig:
    raw = OmegaConf.create({})
    assert isinstance(raw, DictConfig)
    return raw


def _patch_run_shell(
    monkeypatch: pytest.MonkeyPatch,
    cfg: AxBenchBpbAdvConfig,
    *,
    contract: object | None,
) -> tuple[list[object], list[tuple[object | None, int, int]], list[dict[str, Any]]]:
    resolved_requests: list[object] = []
    marker_calls: list[tuple[object | None, int, int]] = []
    run_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(config_loader, "cfg_to_dataclass", lambda _raw: cfg)
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        cli,
        "resolve_cluster_feature_contract_ddp",
        lambda actual_request: (
            resolved_requests.append(actual_request)
            or (
                contract if contract is not None else pytest.fail("unexpected clustered resolution")
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda **_kwargs: (None, None, 0, None, "cpu"),
    )
    monkeypatch.setattr(cli.dist, "is_available", lambda: False)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        cli,
        "_preflight_and_install_feature_contract_marker_ddp",
        lambda _cfg, actual_contract, *, rank, world_size: marker_calls.append(
            (actual_contract, rank, world_size)
        ),
    )
    monkeypatch.setattr(
        cli,
        "resolve_profile_gpu_device_index",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli,
        "axbench_bpb_adv_config_to_dict",
        lambda _cfg: {"stable": True},
    )

    def _capture_run(actual_cfg: object, **kwargs: Any) -> object:
        assert actual_cfg is cfg
        run_calls.append(kwargs)
        return actual_cfg

    monkeypatch.setattr(cli, "_run_resolved", _capture_run)
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    return resolved_requests, marker_calls, run_calls


def test_clustered_run_builds_resolves_and_threads_one_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.feature_mask = {
        "mode": "whitelist",
        "path": str(tmp_path / "feature_mask.txt"),
    }
    cfg.feature_clusters = {"path": str(tmp_path / "feature_clusters.yaml")}
    cfg.encdec_target = {
        "source": "local",
        "model_tag": "clustered-model",
        "step": 11,
    }
    cfg.ours_gcg = OursGcgConfig(top_k_pos=13, top_k_neg=7)
    request = ClusterFeatureContractRequest(
        feature_dir=cfg.feature_dir,
        feature_statistics_path=cfg.feature_statistics_path,
        feature_mask=cfg.feature_mask,
        feature_clusters=cfg.feature_clusters,
        checkpoint_source="local",
        checkpoint_model_tag="clustered-model",
        checkpoint_step=11,
        checkpoint_tag="clustered-model",
        gcg_top_k_pos=13,
        gcg_top_k_neg=7,
        expected_feature_contract_path=Path(cfg.source_run_dir) / "feature_contract.json",
    )
    contract = _cluster_contract(step=23)
    resolved_requests, marker_calls, run_calls = _patch_run_shell(
        monkeypatch,
        cfg,
        contract=contract,
    )

    assert cli._run(_raw_cfg()) is cfg

    actual_request = resolved_requests[0]
    assert isinstance(actual_request, ClusterFeatureContractRequest)
    assert resolved_requests == [request]
    assert actual_request.feature_dir == cfg.feature_dir
    assert actual_request.feature_statistics_path == cfg.feature_statistics_path
    assert actual_request.feature_mask == cfg.feature_mask
    assert actual_request.feature_clusters == cfg.feature_clusters
    assert actual_request.checkpoint_source == "local"
    assert actual_request.checkpoint_model_tag == "clustered-model"
    assert actual_request.checkpoint_step == 11
    assert actual_request.checkpoint_tag == "clustered-model"
    assert actual_request.gcg_top_k_pos == 13
    assert actual_request.gcg_top_k_neg == 7
    assert actual_request.expected_feature_contract_path == (
        Path(cfg.source_run_dir) / "feature_contract.json"
    )
    assert cfg.encdec_target["step"] == 23
    assert marker_calls == [(contract, 0, 1)]
    assert run_calls == [
        {
            "ddp_rank": 0,
            "ddp_world_size": 1,
            "cluster_contract": contract,
        }
    ]


def test_unclustered_run_never_resolves_or_threads_contract_kwarg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    resolved_requests, marker_calls, run_calls = _patch_run_shell(
        monkeypatch,
        cfg,
        contract=None,
    )

    assert cli._run(_raw_cfg()) is cfg

    assert resolved_requests == []
    assert cfg.encdec_target == {"step": "latest"}
    assert marker_calls == [(None, 0, 1)]
    assert run_calls == [{"ddp_rank": 0, "ddp_world_size": 1}]


@pytest.mark.parametrize(
    "stage",
    [stage for stage in Stage if stage is not Stage.ALL],
)
def test_every_standalone_stage_rejects_direction_mismatch_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: Stage,
) -> None:
    source_root = tmp_path / "source"
    write_stage_0_3_source_run(
        source_root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    write_scalar_direction_baselines(
        source_root / "baselines",
        layer=10,
        families=("diffmean",),
        targets=("decoder_only",),
        modes=("positive",),
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_root),
        output_root=str(tmp_path / "out"),
        run_tag="mismatch",
        device="cpu",
        stage=stage.value,
        base_methods=["diffmean_decoder_only"],
        target_models=["decoder_only"],
        adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0,), max_steps=0)],
        feature_dir=str(tmp_path / "unused-features"),
        feature_statistics_path=str(tmp_path / "unused-stats.safetensors"),
        diffmean_and_probe_layer=6,
    )
    monkeypatch.setattr(config_loader, "cfg_to_dataclass", lambda _raw: cfg)
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda **_kwargs: (None, None, 0, None, "cpu"),
    )
    monkeypatch.setattr(cli.dist, "is_available", lambda: False)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(
        source_validation,
        "get_tokenizer",
        lambda: SourceSeedTokenizer(),
    )
    monkeypatch.setattr(
        cli,
        "_run_resolved",
        lambda *_args, **_kwargs: pytest.fail(
            f"{stage.value} dispatch must not start after direction mismatch"
        ),
    )

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        cli._run(_raw_cfg())

    message = str(exc_info.value)
    assert "requested/applied layer=6" in message
    assert "artifact training layer=10" in message
    assert not (Path(cfg.output_root) / "mismatch").exists()


def test_clustered_dispatch_threads_contract_to_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank-zero handler receives the resolved clustered contract."""

    contract = _cluster_contract()
    cfg = AxBenchBpbAdvConfig()
    calls: list[tuple[Stage, object, object]] = []
    handlers = {
        stage: (
            lambda actual_cfg, actual_contract, stage=stage: calls.append(
                (stage, actual_cfg, actual_contract)
            )
        )
        for stage in Stage
        if stage not in {Stage.ALL, Stage.PREPARE_CONTEXT_CACHE}
    }
    monkeypatch.setattr(cli, "_STAGE_HANDLERS", handlers)
    context_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "_stage_prepare_context_cache",
        lambda actual_cfg, **kwargs: context_calls.append({"cfg": actual_cfg, **kwargs}),
    )

    for stage in Stage:
        if stage is Stage.ALL:
            continue
        cli._run_stage(
            stage,
            cfg,
            ddp_rank=0,
            ddp_world_size=2,
            cluster_contract=contract,
        )

    assert [stage for stage, _, _ in calls] == list(handlers)
    assert all(actual_cfg is cfg for _, actual_cfg, _ in calls)
    assert all(actual_contract is contract for _, _, actual_contract in calls)
    assert context_calls == [
        {
            "cfg": cfg,
            "ddp_rank": 0,
            "ddp_world_size": 2,
            "cluster_contract": contract,
        }
    ]


def test_unclustered_dispatch_passes_none_contract_and_preserves_context_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AxBenchBpbAdvConfig()
    handler_calls: list[tuple[Stage, object, object]] = []
    handlers = {
        stage: (
            lambda actual_cfg, actual_contract, stage=stage: handler_calls.append(
                (stage, actual_cfg, actual_contract)
            )
        )
        for stage in Stage
        if stage not in {Stage.ALL, Stage.PREPARE_CONTEXT_CACHE}
    }
    monkeypatch.setattr(cli, "_STAGE_HANDLERS", handlers)
    context_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "_stage_prepare_context_cache",
        lambda actual_cfg, **kwargs: context_calls.append({"cfg": actual_cfg, **kwargs}),
    )

    for stage in Stage:
        if stage is Stage.ALL:
            continue
        cli._run_stage(stage, cfg, ddp_rank=0, ddp_world_size=2)

    assert [stage for stage, _, _ in handler_calls] == list(handlers)
    assert all(actual_cfg is cfg for _, actual_cfg, _ in handler_calls)
    assert all(actual_contract is None for _, _, actual_contract in handler_calls)
    assert context_calls == [{"cfg": cfg, "ddp_rank": 0, "ddp_world_size": 2}]


def test_marker_install_is_rank_zero_only_and_broadcasts_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    ensured: list[tuple[Path, object]] = []
    broadcasts: list[tuple[list[str | None], int]] = []
    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda path, actual_contract: ensured.append((path, actual_contract)),
    )
    monkeypatch.setattr(
        cli.dist,
        "broadcast_object_list",
        lambda payload, *, src: broadcasts.append((list(payload), src)),
    )

    with _capture_loguru_exception_records() as records:
        cli._install_feature_contract_marker_ddp(
            cfg,
            contract,
            rank=0,
            world_size=2,
        )

    assert ensured == [(Path(cfg.output_root) / "run", contract)]
    assert broadcasts == [([None], 0)]
    assert records == []


def test_nonzero_rank_marker_collective_propagates_rank_zero_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    broadcasts: list[tuple[list[str | None], int]] = []
    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not install the marker"),
    )

    def _broadcast(payload: list[str | None], *, src: int) -> None:
        broadcasts.append((list(payload), src))
        payload[0] = "RuntimeError: rank-zero marker sentinel"

    monkeypatch.setattr(cli.dist, "broadcast_object_list", _broadcast)

    with _capture_loguru_exception_records() as records:
        with pytest.raises(AxBenchBpbAdvError, match="rank-zero marker sentinel"):
            cli._install_feature_contract_marker_ddp(
                cfg,
                contract,
                rank=1,
                world_size=2,
            )

    assert broadcasts == [([None], 0)]
    assert records == []


def test_nonzero_rank_preflight_uses_rank_zero_broadcast_without_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not validate the source"),
    )

    def _broadcast(payload: list[str | None], *, src: int) -> None:
        assert src == 0
        payload[0] = "AxBenchBpbAdvError: rank-zero sentinel"

    monkeypatch.setattr(cli.dist, "broadcast_object_list", _broadcast)

    with pytest.raises(AxBenchBpbAdvError, match="rank-zero sentinel"):
        cli._preflight_source_ddp(
            AxBenchBpbAdvConfig(),
            _cluster_contract(),
            rank=1,
            world_size=2,
        )


def test_nonzero_rank_full_preflight_joins_all_collectives_without_committing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    broadcasts: list[list[str | None]] = []
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not validate the source"),
    )
    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not install the marker"),
    )
    monkeypatch.setattr(
        cli,
        "_commit_exact_clustered_source_contract",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not commit the source"),
    )
    monkeypatch.setattr(
        cli.dist,
        "broadcast_object_list",
        lambda payload, *, src: (
            pytest.fail("broadcast source must be rank zero")
            if src != 0
            else broadcasts.append(list(payload))
        ),
    )

    cli._preflight_and_install_feature_contract_marker_ddp(
        cfg,
        _cluster_contract(),
        rank=1,
        world_size=2,
    )

    assert broadcasts == [[None], [None], [None]]


def test_clustered_stage_rejects_foreign_source_contract_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    foreign = _SourceContract(
        {
            "feature_contract_fingerprint": "b" * 64,
            "source": "foreign",
        }
    )
    validation = _validated_source(foreign)
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(
        cli,
        "write_source_contract",
        lambda *_args, **_kwargs: pytest.fail("foreign contract must not be written"),
    )

    with pytest.raises(AxBenchBpbAdvError, match="source=.*target="):
        cli._stage_validate_source(cfg, _cluster_contract())


def test_preflight_fingerprint_mismatch_never_leaves_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    foreign = _SourceContract(
        {
            "feature_contract_fingerprint": "b" * 64,
            "source": "foreign",
        }
    )
    validation = _validated_source(foreign)
    marker = Path(cfg.output_root) / "run" / "feature_contract.json"
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: validation,
    )

    def _install_marker(path: Path, _contract: object) -> None:
        path.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"poisoned": true}', encoding="utf-8")

    monkeypatch.setattr(cli, "ensure_feature_contract_run_marker", _install_marker)
    monkeypatch.setattr(
        cli,
        "write_source_contract",
        lambda *_args, **_kwargs: pytest.fail("foreign source contract must not be written"),
    )

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        cli._preflight_and_install_feature_contract_marker_ddp(
            cfg,
            _cluster_contract(),
            rank=0,
            world_size=1,
        )
        # If the preflight fingerprint raise is stripped, the later stage
        # check still raises. The postcondition below distinguishes that
        # poisoned-marker path from a correct preflight rejection.
        cli._stage_validate_source(cfg, _cluster_contract())

    assert not marker.exists()


def test_failing_preflight_runs_before_marker_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    marker = Path(cfg.output_root) / "run" / "feature_contract.json"

    def _raise_chained_preflight(*_args: Any, **_kwargs: Any) -> None:
        try:
            raise ValueError("preflight root cause")
        except ValueError as cause:
            raise RuntimeError("preflight ordering sentinel") from cause

    monkeypatch.setattr(cli, "validate_stage_0_3_source_run", _raise_chained_preflight)

    def _install_marker(path: Path, _contract: object) -> None:
        path.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"installed": true}', encoding="utf-8")

    monkeypatch.setattr(cli, "ensure_feature_contract_run_marker", _install_marker)

    with _capture_loguru_exception_records() as records:
        with pytest.raises(AxBenchBpbAdvError, match="preflight ordering sentinel"):
            cli._preflight_and_install_feature_contract_marker_ddp(
                cfg,
                _cluster_contract(),
                rank=0,
                world_size=1,
            )

    _assert_rank_zero_traceback(
        records,
        message="AxBench-BPB-Adv rank-0 source preflight failed",
        exception_type=RuntimeError,
        exception_message="preflight ordering sentinel",
        frame_name="_preflight_source_ddp",
        cause_message="preflight root cause",
    )

    assert not marker.exists()


def test_successful_preflight_commits_its_snapshot_after_feature_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    source_contract = _SourceContract(
        {
            "feature_contract_fingerprint": _FINGERPRINT,
            "source": "preflight-snapshot",
        }
    )
    validation = _validated_source(source_contract)
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: events.append(("validate", validation)) or validation,
    )
    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda _path, actual_contract: events.append(("feature_marker", actual_contract)),
    )
    monkeypatch.setattr(
        cli,
        "write_source_contract",
        lambda _path, actual_contract: events.append(("source_contract", actual_contract)),
    )

    cli._preflight_and_install_feature_contract_marker_ddp(
        cfg,
        contract,
        rank=0,
        world_size=1,
    )

    assert events == [
        ("validate", validation),
        ("source_contract", source_contract),
        ("feature_marker", contract),
    ]


def test_preflight_source_commit_failure_is_branded_after_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    validation = _validated_source(
        _SourceContract(
            {
                "feature_contract_fingerprint": _FINGERPRINT,
                "source": "preflight-snapshot",
            }
        ),
    )
    marker_installed: list[object] = []
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda _path, actual_contract: marker_installed.append(actual_contract),
    )
    monkeypatch.setattr(
        cli,
        "write_source_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("source-commit sentinel")),
    )

    with _capture_loguru_exception_records() as records:
        with pytest.raises(
            AxBenchBpbAdvError,
            match="preflight commit failed before run-marker creation.*source-commit sentinel",
        ):
            cli._preflight_and_install_feature_contract_marker_ddp(
                cfg,
                contract,
                rank=0,
                world_size=1,
            )

    _assert_rank_zero_traceback(
        records,
        message="AxBench-BPB-Adv rank-0 source contract commit failed",
        exception_type=OSError,
        exception_message="source-commit sentinel",
        frame_name="_preflight_and_install_feature_contract_marker_ddp",
    )

    assert marker_installed == []


def test_unclustered_preflight_commits_v6_source_before_any_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    validation = _validated_source(
        _SourceContract(
            {
                "feature_contract_fingerprint": None,
                "source": "unclustered",
            }
        )
    )
    writes: list[object] = []
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(
        cli,
        "write_source_contract",
        lambda _path, contract: writes.append(contract),
    )

    cli._preflight_and_install_feature_contract_marker_ddp(
        cfg,
        None,
        rank=0,
        world_size=1,
    )
    assert writes == [validation.contract]


def test_repeated_clustered_stage_validation_commits_source_contract_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    source_contract = _SourceContract(
        {
            "feature_contract_fingerprint": _FINGERPRINT,
            "source": "stable",
        }
    )
    validation = _validated_source(source_contract)
    monkeypatch.setattr(
        cli,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: validation,
    )
    writes: list[Path] = []

    def _write(path: Path, contract: _SourceContract) -> None:
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == (contract.to_json_dict())
            return
        writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract.to_json_dict()), encoding="utf-8")

    monkeypatch.setattr(cli, "write_source_contract", _write)
    resolved: list[object] = []
    sentinel = object()
    monkeypatch.setattr(
        cli,
        "resolve_clustered_stage_source",
        lambda **kwargs: resolved.append(kwargs["validation"]) or sentinel,
    )

    assert cli._validate_clustered_source_for_stage(cfg, _cluster_contract()) is sentinel
    assert cli._validate_clustered_source_for_stage(cfg, _cluster_contract()) is sentinel

    assert writes == [Path(cfg.output_root) / "run" / "source_contract.json"]
    assert resolved == [validation, validation]


def test_preflight_source_snapshot_is_committed_before_stage_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source change after preflight is rejected, never silently committed."""

    cfg = _run_cfg(tmp_path)
    cfg.feature_mask = {
        "mode": "whitelist",
        "path": str(tmp_path / "feature_mask.txt"),
    }
    cfg.feature_clusters = {"path": str(tmp_path / "feature_clusters.yaml")}
    cfg.encdec_target = {
        "source": "local",
        "model_tag": "clustered-model",
        "step": 17,
    }
    source_identity = tmp_path / "source_identity.json"

    def _replace_source(version: str) -> None:
        source_identity.write_text(
            json.dumps(
                {
                    "feature_contract_fingerprint": _FINGERPRINT,
                    "source_version": version,
                }
            ),
            encoding="utf-8",
        )

    reads: list[str] = []

    def _validate(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        payload = json.loads(source_identity.read_text(encoding="utf-8"))
        reads.append(str(payload["source_version"]))
        if len(reads) == 1:
            _replace_source("commit")
        return _validated_source(_SourceContract(payload))

    marker_contracts: list[object] = []
    contract = _cluster_contract()
    monkeypatch.setattr(config_loader, "cfg_to_dataclass", lambda _raw: cfg)
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        cli,
        "resolve_cluster_feature_contract_ddp",
        lambda _request: contract,
    )
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda **_kwargs: (None, None, 0, None, "cpu"),
    )
    monkeypatch.setattr(cli.dist, "is_available", lambda: False)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        cli,
        "resolve_profile_gpu_device_index",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli,
        "axbench_bpb_adv_config_to_dict",
        lambda _cfg: {"stable": True},
    )
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(cli, "validate_stage_0_3_source_run", _validate)
    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda _path, contract: marker_contracts.append(contract),
    )
    monkeypatch.setattr(
        cli,
        "write_source_contract",
        lambda path, source_contract: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_text(
                json.dumps(source_contract.to_json_dict()),
                encoding="utf-8",
            ),
        ),
    )

    _replace_source("preflight")
    with pytest.raises(
        AxBenchBpbAdvError,
        match="source contract changed within an existing run",
    ):
        cli._run(_raw_cfg())

    committed = json.loads(
        (Path(cfg.output_root) / "run" / "source_contract.json").read_text(encoding="utf-8")
    )
    assert reads == ["preflight", "commit"]
    assert marker_contracts == [contract]
    assert committed["source_version"] == "preflight"
    assert json.loads(source_identity.read_text())["source_version"] == "commit"


@pytest.mark.parametrize(
    ("metric", "score_schema", "runner_name", "artifact_family"),
    [
        (
            "roundtrip",
            ROUNDTRIP_SCORE_SCHEMA,
            "run_model_backed_roundtrip_score_stage",
            "roundtrip_scores",
        ),
        (
            "bpb_class_margin",
            SCORE_SCHEMA,
            "run_model_backed_bpb_adv_score_stage",
            "class_margin_scores",
        ),
    ],
)
def test_clustered_score_revalidates_and_threads_paired_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    score_schema: pa.Schema,
    runner_name: str,
    artifact_family: str,
) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.steering_metric = cast(Any, metric)
    contract = _cluster_contract()
    candidates = SimpleNamespace(num_rows=3)
    scores = pa.Table.from_pylist([], schema=score_schema)
    events: list[str] = []
    load_calls: list[tuple[Path, dict[str, Any]]] = []
    score_calls: list[dict[str, Any]] = []
    bind_calls: list[dict[str, Any]] = []
    write_calls: list[tuple[pa.Table, Path, dict[str, Any]]] = []
    source_identity = object()
    stage_source = SimpleNamespace(source_identity=source_identity)
    candidate_snapshot = SimpleNamespace(table=candidates)

    def _revalidate(actual_cfg: object, actual_contract: object) -> object:
        assert actual_cfg is cfg
        assert actual_contract is contract
        events.append("revalidate")
        return stage_source

    def _load(path: Path, **kwargs: Any) -> object:
        events.append("load_candidates")
        load_calls.append((path, kwargs))
        return candidate_snapshot

    def _score(**kwargs: Any) -> pa.Table:
        events.append("score")
        score_calls.append(kwargs)
        return scores

    def _bind(table: pa.Table, **kwargs: Any) -> pa.Table:
        events.append("bind")
        bind_calls.append(kwargs)
        return table

    def _write(table: pa.Table, path: Path, **kwargs: Any) -> None:
        events.append("write")
        write_calls.append((table, path, kwargs))

    monkeypatch.setattr(cli, "_validate_clustered_source_for_stage", _revalidate)
    monkeypatch.setattr(cli, "load_clustered_candidate_snapshot", _load)
    monkeypatch.setattr(cli, runner_name, _score)
    monkeypatch.setattr(cli, "bind_clustered_table", _bind)
    monkeypatch.setattr(cli, "write_clustered_table_atomic", _write)

    cli._stage_score(cfg, contract)

    paths = cli._paths(cfg)
    assert events == ["revalidate", "load_candidates", "score", "bind", "write"]
    assert len(load_calls) == 1
    candidate_path, candidate_kwargs = load_calls[0]
    stage_paths = candidate_kwargs["paths"]
    assert candidate_path == paths.candidates_path()
    assert stage_paths.root == paths.root
    assert candidate_kwargs == {
        "paths": stage_paths,
        "contract": contract,
        "source_identity": source_identity,
    }

    assert len(score_calls) == 1
    assert score_calls[0] == {
        "cfg": cfg,
        "paths": stage_paths,
        "candidates": candidates,
        "stats": {},
        "cluster_contract": contract,
        "stage_source": stage_source,
        "candidate_snapshot": candidate_snapshot,
    }
    assert bind_calls == [
        {
            "paths": stage_paths,
            "contract": contract,
            "artifact_family": artifact_family,
            "candidate_snapshot": candidate_snapshot,
            "source_identity": source_identity,
        }
    ]
    assert len(write_calls) == 1
    written_table, written_path, written_kwargs = write_calls[0]
    assert written_table is scores
    expected_path = paths.roundtrip_scores_path() if metric == "roundtrip" else paths.scores_path()
    assert written_path == expected_path
    assert written_kwargs == {
        "paths": stage_paths,
        "contract": contract,
        "artifact_family": artifact_family,
        "candidate_snapshot": candidate_snapshot,
        "source_identity": source_identity,
    }


def test_clustered_general_val_dispatch_threads_one_paired_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    assert cfg.general_val_bpb.enabled
    contract = _cluster_contract()
    stage_source = object()
    events: list[tuple[str, object]] = []
    run_calls: list[dict[str, Any]] = []

    def _validate(actual_cfg: object, actual_contract: object) -> object:
        assert actual_cfg is cfg
        assert actual_contract is contract
        events.append(("validate", actual_contract))
        return stage_source

    def _run(**kwargs: Any) -> pa.Table:
        events.append(("run", kwargs))
        run_calls.append(kwargs)
        return pa.table({"general_val_bpb": [1.0]})

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        _validate,
    )
    monkeypatch.setattr(cli, "run_general_val_bpb_stage", _run)

    cli._stage_score_general_val(cfg, contract)

    assert [name for name, _value in events] == ["validate", "run"]
    assert events[0] == ("validate", contract)
    assert len(run_calls) == 1
    run_kwargs = run_calls[0]
    assert run_kwargs["cfg"] is cfg
    assert run_kwargs["cluster_contract"] is contract
    assert run_kwargs["stage_source"] is stage_source
    assert run_kwargs["paths"].root == cli._paths(cfg).root


def test_clustered_prepare_attack_rows_consumes_committed_validation_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-path replacement after contract commit cannot change samples."""

    cfg = _run_cfg(tmp_path)
    cfg.base_methods = []
    contract = _cluster_contract()
    events: list[str] = []
    sampled_markers: list[str] = []
    loader_markers: list[str] = []
    positive_a = [{"concept_id": 1, "marker": "positive-A"}]
    negative_a = [{"concept_id": 1, "marker": "negative-A"}]
    validation = SimpleNamespace(
        positive_rows=lambda: events.append("positive-A") or list(positive_a),
        negative_rows=lambda: events.append("negative-A") or list(negative_a),
    )

    def _validate(*_args: Any, **_kwargs: Any) -> object:
        events.append("validate-A")
        return validation

    def _commit(*_args: Any, **_kwargs: Any) -> None:
        events.append("commit-A-and-replace-paths-with-B")

    def _load_replacement(path: Path, **_kwargs: Any) -> list[dict[str, Any]]:
        marker = f"{Path(path).stem}-B"
        loader_markers.append(marker)
        return [{"concept_id": 1, "marker": marker}]

    def _sample(row: dict[str, Any], *, mode: object, tokenizer: object) -> object:
        del tokenizer
        sampled_markers.append(str(row["marker"]))
        return SimpleNamespace(
            concept_id=1,
            mode=mode,
            prefix_text=f"prefix-{row['marker']}",
        )

    work_specs = [
        SimpleNamespace(mode="positive"),
        SimpleNamespace(mode="negative"),
    ]
    attack_rows = pa.Table.from_pylist([], schema=ATTACK_ROWS_SCHEMA)
    writes: list[pa.Table] = []

    monkeypatch.setattr(cli, "validate_stage_0_3_source_run", _validate)
    monkeypatch.setattr(cli, "_commit_exact_clustered_source_contract", _commit)
    monkeypatch.setattr(
        cli,
        "_target_runtime_info_from_runtime",
        lambda actual_cfg, actual_contract: (
            pytest.fail("wrong config") if actual_cfg is not cfg else {},
            pytest.fail("wrong contract") if actual_contract is not contract else object(),
        ),
    )
    monkeypatch.setattr(cli, "work_specs_from_method_registry", lambda **_kwargs: work_specs)
    monkeypatch.setattr(cli, "load_raw_synth_rows", _load_replacement)
    monkeypatch.setattr(cli, "_read_source_concept_ids", lambda _path: {1})
    monkeypatch.setattr(
        cli,
        "_filter_synth_rows_to_source_concepts",
        lambda rows, **_kwargs: list(rows),
    )
    monkeypatch.setattr(cli, "bpb_attack_sample_from_synth_row", _sample)
    monkeypatch.setattr(cli, "pair_samples_into_prefixes", lambda samples: list(samples))
    monkeypatch.setattr(
        cli,
        "evaluation_prefixes_for_scoring",
        lambda prefixes, **_kwargs: list(prefixes),
    )
    monkeypatch.setattr(
        cli,
        "prepare_bpb_attack_rows",
        lambda **_kwargs: attack_rows,
    )
    monkeypatch.setattr(cli, "write_evaluation_population", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "bind_clustered_table", lambda table, **_kwargs: table)
    monkeypatch.setattr(
        cli,
        "write_clustered_table_atomic",
        lambda table, *_args, **_kwargs: writes.append(table),
    )

    cli._stage_prepare_attack_rows(cfg, contract)

    assert events == [
        "validate-A",
        "commit-A-and-replace-paths-with-B",
        "positive-A",
        "negative-A",
    ]
    assert loader_markers == []
    assert sampled_markers == ["positive-A", "negative-A"]
    assert writes == [attack_rows]


@pytest.mark.parametrize(
    ("metric", "score_family", "score_path_name", "aggregate_name"),
    [
        (
            "bpb_class_margin",
            "class_margin_scores",
            "scores.parquet",
            "aggregate_bpb_adv_scores",
        ),
        (
            "roundtrip",
            "roundtrip_scores",
            "roundtrip_scores.parquet",
            "aggregate_roundtrip_adv_scores",
        ),
    ],
)
def test_clustered_aggregate_consumes_exact_tables_validated_before_artifact_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    score_family: str,
    score_path_name: str,
    aggregate_name: str,
) -> None:
    """A post-validation path swap cannot change either aggregation input."""

    cfg = _run_cfg(tmp_path)
    cfg.steering_metric = cast(Any, metric)
    cfg.general_val_bpb = GeneralValBpbConfig(enabled=True)
    contract = _cluster_contract()
    paths = cli._paths(cfg)
    score_path = paths.root / score_path_name
    general_val_path = paths.general_val_bpb_path()

    validated_score = object()
    replacement_score = object()
    validated_general_val = object()
    replacement_general_val = object()
    current_artifacts = {
        score_path: validated_score,
        general_val_path: validated_general_val,
    }
    reads: list[Path] = []
    identity_checks: list[tuple[object, str, dict[str, Any]]] = []
    consumed: dict[str, object | None] = {}

    def _read_table(path: Path) -> object:
        normalized = Path(path)
        reads.append(normalized)
        return current_artifacts[normalized]

    def _assert_identity(table: object, **kwargs: Any) -> None:
        artifact_family = str(kwargs["artifact_family"])
        identity_checks.append((table, artifact_family, kwargs))
        if artifact_family == score_family:
            assert table is validated_score
            current_artifacts[score_path] = replacement_score
            return
        assert artifact_family == "general_val_bpb"
        assert table is validated_general_val
        current_artifacts[general_val_path] = replacement_general_val

    def _aggregate(
        actual_score_path: Path,
        _output_dir: Path,
        *,
        general_val_path: Path | None,
        scores_table: object | None = None,
        general_val_table: object | None = None,
    ) -> None:
        consumed["score"] = _read_table(actual_score_path) if scores_table is None else scores_table
        consumed["general_val"] = (
            _read_table(general_val_path)
            if general_val_table is None and general_val_path is not None
            else general_val_table
        )

    source_identity = object()
    stage_source = SimpleNamespace(source_identity=source_identity)
    candidate_snapshot = object()
    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: stage_source,
    )
    monkeypatch.setattr(
        cli,
        "load_clustered_candidate_snapshot",
        lambda *_args, **kwargs: (
            candidate_snapshot
            if kwargs["source_identity"] is source_identity
            else pytest.fail("aggregate snapshot used a different source identity")
        ),
    )
    monkeypatch.setattr(cli.pq, "read_table", _read_table)
    monkeypatch.setattr(cli, "assert_table_cluster_identity", _assert_identity)
    monkeypatch.setattr(cli, aggregate_name, _aggregate)

    cli._stage_aggregate(cfg, contract)

    assert [(table, family) for table, family, _kwargs in identity_checks] == [
        (validated_score, score_family),
        (validated_general_val, "general_val_bpb"),
    ]
    assert current_artifacts == {
        score_path: replacement_score,
        general_val_path: replacement_general_val,
    }
    assert consumed == {
        "score": validated_score,
        "general_val": validated_general_val,
    }
    assert reads == [score_path, general_val_path]
    assert all(
        kwargs["source_identity"] is source_identity
        and kwargs["candidate_snapshot"] is candidate_snapshot
        for _table, _family, kwargs in identity_checks
    )


@pytest.mark.parametrize(
    ("metric", "aggregate_name"),
    [
        ("bpb_class_margin", "aggregate_bpb_adv_scores"),
        ("roundtrip", "aggregate_roundtrip_adv_scores"),
    ],
)
def test_clustered_aggregate_without_general_val_passes_explicit_none_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    aggregate_name: str,
) -> None:
    """Disabled general-val never leaves an undefined or path-backed input."""

    cfg = _run_cfg(tmp_path)
    cfg.steering_metric = cast(Any, metric)
    cfg.general_val_bpb = GeneralValBpbConfig(enabled=False)
    contract = _cluster_contract()
    score_table = object()
    calls: list[dict[str, Any]] = []

    source_identity = object()
    stage_source = SimpleNamespace(source_identity=source_identity)
    candidate_snapshot = object()
    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: stage_source,
    )
    monkeypatch.setattr(
        cli,
        "load_clustered_candidate_snapshot",
        lambda *_args, **_kwargs: candidate_snapshot,
    )
    monkeypatch.setattr(cli.pq, "read_table", lambda _path: score_table)
    monkeypatch.setattr(cli, "assert_table_cluster_identity", lambda *_args, **_kwargs: None)

    def _aggregate(
        _score_path: Path,
        _output_dir: Path,
        **kwargs: Any,
    ) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(cli, aggregate_name, _aggregate)

    cli._stage_aggregate(cfg, contract)

    assert calls == [
        {
            "general_val_path": None,
            "scores_table": score_table,
            "general_val_table": None,
        }
    ]


@pytest.mark.parametrize(
    ("metric", "aggregate_name"),
    [
        ("bpb_class_margin", "aggregate_bpb_adv_scores"),
        ("roundtrip", "aggregate_roundtrip_adv_scores"),
    ],
)
@pytest.mark.parametrize("general_val_enabled", [False, True])
def test_unclustered_aggregate_keeps_legacy_path_only_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    aggregate_name: str,
    general_val_enabled: bool,
) -> None:
    """The clustered fix must not change the schema-v6 aggregation call."""

    cfg = _run_cfg(tmp_path)
    cfg.steering_metric = cast(Any, metric)
    cfg.general_val_bpb = GeneralValBpbConfig(enabled=general_val_enabled)
    calls: list[tuple[Path, Path, dict[str, Any]]] = []

    monkeypatch.setattr(
        cli.pq,
        "read_table",
        lambda _path: pytest.fail("unclustered aggregation must remain path-only"),
    )

    def _aggregate(
        score_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> None:
        calls.append((score_path, output_dir, kwargs))

    monkeypatch.setattr(cli, aggregate_name, _aggregate)

    cli._stage_aggregate(cfg)

    paths = cli._paths(cfg)
    expected_score_path = (
        paths.roundtrip_scores_path() if metric == "roundtrip" else paths.scores_path()
    )
    expected_output_dir = (
        paths.roundtrip_aggregates_dir() if metric == "roundtrip" else paths.aggregates_dir()
    )
    assert calls == [
        (
            expected_score_path,
            expected_output_dir,
            {"general_val_path": (paths.general_val_bpb_path() if general_val_enabled else None)},
        )
    ]


def test_clustered_build_candidates_threads_one_paired_stage_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    source_identity = object()
    stage_source = SimpleNamespace(source_identity=source_identity)
    candidate_snapshot = SimpleNamespace(
        table=pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA),
    )
    candidate_calls: list[dict[str, Any]] = []
    snapshot_calls: list[tuple[Path, dict[str, Any]]] = []
    identity_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: stage_source,
    )
    monkeypatch.setattr(
        cli,
        "run_model_backed_bpb_candidate_stage",
        lambda **kwargs: candidate_calls.append(kwargs) or 0,
    )

    def _read_table(path: Path) -> pa.Table:
        if path.name == "universal_triggers.parquet":
            return pa.Table.from_pylist([], schema=UNIVERSAL_TRIGGERS_SCHEMA)
        pytest.fail(f"unexpected table read: {path}")

    monkeypatch.setattr(cli.pq, "read_table", _read_table)
    monkeypatch.setattr(
        cli,
        "load_clustered_candidate_snapshot",
        lambda path, **kwargs: snapshot_calls.append((path, kwargs)) or candidate_snapshot,
    )
    monkeypatch.setattr(
        cli,
        "assert_table_cluster_identity",
        lambda _table, **kwargs: identity_calls.append(kwargs),
    )

    cli._stage_build_candidates(cfg, contract)

    assert len(candidate_calls) == 1
    assert candidate_calls[0]["cluster_contract"] is contract
    assert candidate_calls[0]["stage_source"] is stage_source
    assert len(snapshot_calls) == 1
    snapshot_path, snapshot_kwargs = snapshot_calls[0]
    assert snapshot_path == cli._paths(cfg).candidates_path()
    assert snapshot_kwargs["paths"].candidates_path() == snapshot_path
    assert snapshot_kwargs["contract"] is contract
    assert snapshot_kwargs["source_identity"] is source_identity
    assert [call["artifact_family"] for call in identity_calls] == [
        "universal_triggers",
    ]
    assert identity_calls[0]["source_identity"] is source_identity
    assert identity_calls[0]["candidate_snapshot"] is candidate_snapshot


def test_clustered_context_cache_builds_clustered_cache_and_runs_both_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.filter_unaccepted_samples = True
    contract = _cluster_contract()
    extraction_tokenizer = object()
    prebuild_tokenizer = object()
    decoder_tokenizer = object()
    events: list[str] = []
    runtime_kwargs: list[dict[str, Any]] = []
    build_kwargs: list[dict[str, Any]] = []
    checksum_tokenizers: list[object] = []
    bos_pairs: list[tuple[object, object]] = []

    class _RuntimeState:
        def __init__(self, **kwargs: Any) -> None:
            runtime_kwargs.append(kwargs)
            self.released = False

        def base_gemma_and_sae(self) -> tuple[object, object]:
            events.append("load_extractor")
            return "base_model", "base_sae"

        def extraction_tokenizer(self) -> object:
            return extraction_tokenizer

        def release_base_gemma_and_sae(self) -> None:
            events.append("release_extractor")
            self.released = True

        def decoder_only(self) -> SimpleNamespace:
            assert self.released
            events.append("load_target")
            return SimpleNamespace(tokenizer=decoder_tokenizer)

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: (
            events.append("validate_source")
            or SimpleNamespace(
                validation=SimpleNamespace(
                    context_prefixes=lambda: {
                        "validated-prefix": make_prefix_hash("validated-prefix")
                    }
                )
            )
        ),
    )
    bpb_cfg = SimpleNamespace(feature_dir=cfg.feature_dir)
    monkeypatch.setattr(cli, "_bpb_config_from_adv_config", lambda _cfg: bpb_cfg)
    monkeypatch.setattr(cli, "RuntimeState", _RuntimeState)
    monkeypatch.setattr(cli, "get_tokenizer", lambda: prebuild_tokenizer)
    monkeypatch.setattr(
        cli,
        "assert_tokenizer_checksum_matches",
        lambda *, tokenizer, feature_dir: checksum_tokenizers.append(tokenizer),
    )
    monkeypatch.setattr(
        cli,
        "assert_bos_lengths_match",
        lambda *, tokenizer_extraction, tokenizer_nanochat: bos_pairs.append(
            (tokenizer_extraction, tokenizer_nanochat)
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **kwargs: build_kwargs.append(kwargs),
    )
    monkeypatch.setattr(
        cli,
        "_run_bpb_context_cache_stage",
        lambda *_args: pytest.fail("clustered path must not use legacy builder"),
    )
    monkeypatch.setattr(
        cli,
        "SourcePaths",
        lambda *_args, **_kwargs: pytest.fail(
            "clustered context cache must not reopen source paths"
        ),
    )

    cli._stage_prepare_context_cache(
        cfg,
        ddp_rank=0,
        ddp_world_size=1,
        cluster_contract=contract,
    )

    assert runtime_kwargs == [
        {
            "cfg": bpb_cfg,
            "ddp_rank": 0,
            "ddp_world_size": 1,
            "cluster_contract": contract,
        }
    ]
    assert len(build_kwargs) == 1
    assert build_kwargs[0]["cluster_contract"] is contract
    assert build_kwargs[0]["base_model"] == "base_model"
    assert build_kwargs[0]["base_sae"] == "base_sae"
    assert build_kwargs[0]["prefix_population"].as_dict() == {
        "validated-prefix": make_prefix_hash("validated-prefix")
    }
    assert "source_positive" not in build_kwargs[0]
    assert "source_negative" not in build_kwargs[0]
    assert checksum_tokenizers == [prebuild_tokenizer, decoder_tokenizer]
    assert bos_pairs == [
        (extraction_tokenizer, prebuild_tokenizer),
        (extraction_tokenizer, decoder_tokenizer),
    ]
    assert events == [
        "validate_source",
        "load_extractor",
        "release_extractor",
        "load_target",
    ]


def test_nonzero_rank_context_cache_consumes_rank_zero_prefix_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    extraction_tokenizer = object()
    target_tokenizer = object()
    build_kwargs: list[dict[str, Any]] = []
    broadcasts: list[list[object]] = []

    class _RuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def base_gemma_and_sae(self) -> tuple[object, object]:
            return object(), object()

        def extraction_tokenizer(self) -> object:
            return extraction_tokenizer

        def release_base_gemma_and_sae(self) -> None:
            pass

        def decoder_only(self) -> SimpleNamespace:
            return SimpleNamespace(tokenizer=target_tokenizer)

    def _broadcast(payload: list[object], *, src: int) -> None:
        assert src == 0
        broadcasts.append(list(payload))
        if len(broadcasts) == 1:
            assert payload == [None]
            payload[0] = {
                "error": None,
                "prefixes": {"rank-zero-prefix": make_prefix_hash("rank-zero-prefix")},
            }

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: pytest.fail("nonzero rank must not read the source"),
    )
    monkeypatch.setattr(
        cli,
        "SourcePaths",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not reopen source paths"),
    )
    monkeypatch.setattr(
        cli,
        "_bpb_config_from_adv_config",
        lambda _cfg: SimpleNamespace(feature_dir=cfg.feature_dir),
    )
    monkeypatch.setattr(cli, "RuntimeState", _RuntimeState)
    monkeypatch.setattr(cli, "get_tokenizer", lambda: target_tokenizer)
    monkeypatch.setattr(
        cli,
        "assert_tokenizer_checksum_matches",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        cli,
        "assert_bos_lengths_match",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(cli.dist, "broadcast_object_list", _broadcast)
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **kwargs: build_kwargs.append(kwargs),
    )

    cli._stage_prepare_context_cache(
        cfg,
        ddp_rank=1,
        ddp_world_size=2,
        cluster_contract=contract,
    )

    assert broadcasts == [[None], [None], [None]]
    assert len(build_kwargs) == 1
    assert build_kwargs[0]["ddp_rank"] == 1
    assert build_kwargs[0]["ddp_world_size"] == 2
    assert build_kwargs[0]["prefix_population"].as_dict() == {
        "rank-zero-prefix": make_prefix_hash("rank-zero-prefix")
    }


def test_context_cache_source_failure_broadcasts_before_runtime_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    broadcasts: list[list[object]] = []
    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("source sentinel")),
    )
    monkeypatch.setattr(
        cli.dist,
        "broadcast_object_list",
        lambda payload, *, src: broadcasts.append(list(payload)),
    )
    monkeypatch.setattr(
        cli,
        "RuntimeState",
        lambda **_kwargs: pytest.fail("source failure must precede model runtime"),
    )
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **_kwargs: pytest.fail("source failure must precede cache writes"),
    )

    with _capture_loguru_exception_records() as records:
        with pytest.raises(AxBenchBpbAdvError, match="source sentinel"):
            cli._stage_prepare_context_cache(
                cfg,
                ddp_rank=0,
                ddp_world_size=2,
                cluster_contract=_cluster_contract(),
            )

    _assert_rank_zero_traceback(
        records,
        message="AxBench-BPB-Adv rank-0 context-cache source preflight failed",
        exception_type=RuntimeError,
        exception_message="source sentinel",
        frame_name="_stage_prepare_context_cache",
    )

    assert broadcasts == [
        [
            {
                "error": "RuntimeError: source sentinel",
                "prefixes": None,
            }
        ]
    ]


@pytest.mark.parametrize(
    ("rank_zero_payload", "match"),
    [
        (None, "malformed payload"),
        (
            {"error": None, "prefixes": {}, "extra": True},
            "malformed fields",
        ),
        ({"error": 1, "prefixes": None}, "malformed error"),
        (
            {"error": None, "prefixes": None},
            "without a prefix population",
        ),
    ],
)
def test_nonzero_rank_context_cache_rejects_malformed_source_broadcast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank_zero_payload: object,
    match: str,
) -> None:
    cfg = _run_cfg(tmp_path)

    def _broadcast(payload: list[object], *, src: int) -> None:
        assert src == 0
        payload[0] = rank_zero_payload

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: pytest.fail("nonzero rank must not read the source"),
    )
    monkeypatch.setattr(cli.dist, "broadcast_object_list", _broadcast)
    monkeypatch.setattr(
        cli,
        "RuntimeState",
        lambda **_kwargs: pytest.fail("malformed source payload must fail before model runtime"),
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cli._stage_prepare_context_cache(
            cfg,
            ddp_rank=1,
            ddp_world_size=2,
            cluster_contract=_cluster_contract(),
        )


def test_clustered_context_guard_failure_is_broadcast_before_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.filter_unaccepted_samples = True
    broadcasts: list[list[str | None]] = []

    class _RuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def base_gemma_and_sae(self) -> tuple[object, object]:
            return object(), object()

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_a: SimpleNamespace(
            validation=SimpleNamespace(
                context_prefixes=lambda: {"validated-prefix": make_prefix_hash("validated-prefix")}
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_bpb_config_from_adv_config",
        lambda _cfg: SimpleNamespace(feature_dir=cfg.feature_dir),
    )
    monkeypatch.setattr(cli, "RuntimeState", _RuntimeState)
    monkeypatch.setattr(
        cli,
        "get_tokenizer",
        lambda: (_ for _ in ()).throw(RuntimeError("tokenizer sentinel")),
    )
    monkeypatch.setattr(
        cli.dist,
        "broadcast_object_list",
        lambda payload, *, src: broadcasts.append(list(payload)),
    )
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **_kwargs: pytest.fail("guard failure must precede builder"),
    )

    with _capture_loguru_exception_records() as records:
        with pytest.raises(AxBenchBpbAdvError, match="before extraction or cache writes"):
            cli._stage_prepare_context_cache(
                cfg,
                ddp_rank=0,
                ddp_world_size=2,
                cluster_contract=_cluster_contract(),
            )

    _assert_rank_zero_traceback(
        records,
        message="AxBench-BPB-Adv rank-0 pre-build tokenizer guard failed",
        exception_type=RuntimeError,
        exception_message="tokenizer sentinel",
        frame_name="_stage_prepare_context_cache",
    )

    assert len(broadcasts) == 2
    assert broadcasts[0] == [
        {
            "error": None,
            "prefixes": {"validated-prefix": make_prefix_hash("validated-prefix")},
        }
    ]
    assert "tokenizer sentinel" in str(broadcasts[1][0])


def test_clustered_context_postbuild_guard_logs_and_broadcasts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _run_cfg(tmp_path)
    contract = _cluster_contract()
    extraction_tokenizer = object()
    prebuild_tokenizer = object()
    decoder_tokenizer = object()
    broadcasts: list[list[object]] = []
    events: list[str] = []

    class _RuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            self.released = False

        def base_gemma_and_sae(self) -> tuple[object, object]:
            events.append("load_extractor")
            return object(), object()

        def extraction_tokenizer(self) -> object:
            return extraction_tokenizer

        def release_base_gemma_and_sae(self) -> None:
            events.append("release_extractor")
            self.released = True

        def decoder_only(self) -> SimpleNamespace:
            assert self.released
            events.append("load_target")
            return SimpleNamespace(tokenizer=decoder_tokenizer)

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_args: SimpleNamespace(
            validation=SimpleNamespace(
                context_prefixes=lambda: {"validated-prefix": make_prefix_hash("validated-prefix")}
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_bpb_config_from_adv_config",
        lambda _cfg: SimpleNamespace(feature_dir=cfg.feature_dir),
    )
    monkeypatch.setattr(cli, "RuntimeState", _RuntimeState)
    monkeypatch.setattr(cli, "get_tokenizer", lambda: prebuild_tokenizer)

    def _checksum_guard(*, tokenizer: object, feature_dir: str) -> None:
        del feature_dir
        if tokenizer is decoder_tokenizer:
            raise RuntimeError("postbuild tokenizer sentinel")

    monkeypatch.setattr(cli, "assert_tokenizer_checksum_matches", _checksum_guard)
    monkeypatch.setattr(cli, "assert_bos_lengths_match", lambda **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **_kwargs: events.append("build_cache"),
    )
    monkeypatch.setattr(
        cli.dist,
        "broadcast_object_list",
        lambda payload, *, src: broadcasts.append(list(payload)),
    )

    with _capture_loguru_exception_records() as records:
        with pytest.raises(
            AxBenchBpbAdvError,
            match="tokenizer compatibility guard failed: RuntimeError: postbuild tokenizer sentinel",
        ):
            cli._stage_prepare_context_cache(
                cfg,
                ddp_rank=0,
                ddp_world_size=2,
                cluster_contract=contract,
            )

    _assert_rank_zero_traceback(
        records,
        message="AxBench-BPB-Adv rank-0 post-build tokenizer guard failed",
        exception_type=RuntimeError,
        exception_message="postbuild tokenizer sentinel",
        frame_name="_stage_prepare_context_cache",
    )
    assert broadcasts == [
        [
            {
                "error": None,
                "prefixes": {"validated-prefix": make_prefix_hash("validated-prefix")},
            }
        ],
        [None],
        ["RuntimeError: postbuild tokenizer sentinel"],
    ]
    assert events == [
        "load_extractor",
        "build_cache",
        "release_extractor",
        "load_target",
    ]


def test_unclustered_run_rejects_a_cluster_stamped_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unclustered run refuses a root that already carries a contract marker.

    Run tags separate the two regimes for freshly generated tags (the clustered
    config keeps ``feature_clusters`` in the run-tag hash while an unclustered one
    drops it), so this guard is not reachable by accident. It IS reachable whenever
    an operator pins ``run_tag`` explicitly, or re-runs a clustered root with
    clustering switched off — and without it the run would quietly write unclustered
    artifacts into a clustered root, mixing two regimes' evidence under one tag with
    nothing downstream to notice.

    Deleting the guard previously left this suite entirely green, which is why this
    test exists.
    """
    cfg = _run_cfg(tmp_path)
    root = Path(cfg.output_root) / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "feature_contract.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda *_args, **_kwargs: pytest.fail(
            "unclustered run must not install or refresh a contract marker"
        ),
    )

    with _capture_loguru_exception_records() as records:
        with pytest.raises(
            AxBenchBpbAdvError,
            match="unclustered BPB run rejects cluster-stamped output",
        ):
            cli._install_feature_contract_marker_ddp(
                cfg,
                None,
                rank=0,
                world_size=1,
            )

    _assert_rank_zero_traceback(
        records,
        message="AxBench-BPB-Adv rank-0 feature-contract marker installation failed",
        exception_type=AxBenchBpbAdvError,
        exception_message="unclustered BPB run rejects cluster-stamped output; choose a fresh run_tag",
        frame_name="_install_feature_contract_marker_ddp",
    )


def test_unclustered_run_accepts_a_clean_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard above must not fire on an ordinary unclustered run.

    Pins the negative half so the rejection cannot be satisfied by a blanket raise —
    an unclustered run over a root with no marker installs nothing and returns.
    """
    cfg = _run_cfg(tmp_path)
    (Path(cfg.output_root) / "run").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        cli,
        "ensure_feature_contract_run_marker",
        lambda *_args, **_kwargs: pytest.fail("unclustered run must not install a contract marker"),
    )

    cli._install_feature_contract_marker_ddp(cfg, None, rank=0, world_size=1)
