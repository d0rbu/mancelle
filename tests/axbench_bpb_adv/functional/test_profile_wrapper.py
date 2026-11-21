"""Functional tests for the ``scripts.benchmark.profile_axbench_bpb_adv`` wrapper.

The wrapper's job is small and well-defined:
  1. Synchronize the run tag (matching the production CLI's protocol).
  2. Resolve the profile output dir under that tag.
  3. Activate :class:`AxBenchProfiler` around the production dispatcher
     (``_run_resolved``) on rank 0; non-zero ranks run the dispatcher
     without a profiler.
  4. Write the profile artifacts (events.jsonl / summary.yaml / summary.csv)
     on rank 0.

These tests mock ``_run_resolved`` and ``validate_config`` so we exercise the
wrapper in isolation — the production dispatcher's behavior is covered by
existing tests in :mod:`tests.axbench_bpb_adv.unit` and the upstream
:mod:`scripts.eval.axbench_bpb_adv_eval` test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

import scripts.benchmark.profile_axbench_bpb_adv as profile_cli
import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
from steering.eval.axbench_base.profiler import get_current_profiler
from steering.eval.axbench_bpb_adv.config import (
    validate_all_stage_representative_target_contract,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


@pytest.fixture(autouse=True)
def _mock_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass `validate_config` (which reaches into the source run dir)."""
    monkeypatch.setattr(
        bpb_adv_cli,
        "validate_config",
        lambda cfg_obj: None,  # noqa: ARG005
    )
    monkeypatch.setattr(
        bpb_adv_cli,
        "validate_all_stage_representative_target_contract",
        lambda cfg_obj: None,  # noqa: ARG005
    )


def _minimal_cfg(tmp_path: Path) -> DictConfig:
    """Build the smallest config the wrapper can resolve."""
    cfg_dict = {
        "source_run_dir": str(tmp_path / "source_run"),
        "output_root": str(tmp_path / "out"),
        "run_tag": "profile-wrapper-test",
        "device": "cpu",
        "stage": "validate_source",
        "target_models": ["decoder_only"],
        "base_methods": None,
        "adversarial_methods": [
            {
                "method": "additive_direction",
                "families": ["diffmean"],
                "magnitudes": [0.0, 1.0],
            }
        ],
        "feature_dir": str(tmp_path / "features"),
        "feature_statistics_path": str(
            tmp_path / "features" / "feature_statistics.safetensors"
        ),
        "extraction_tokenizer_id": "google/gemma-2-2b",
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_source": "8-gemmascope-res-16k",
        "extraction_max_tokens": 2048,
        "extraction_threshold": 0.0,
        "decoder_only_target": {},
        "encdec_target": {},
        "lora_adapter": {},
        "filter_unaccepted_samples": True,
        "next_segment_conditioning_offset": False,
        "diffmean_and_probe_layer": 6,
        "context_cache": {},
        "offload_between_stages": False,
        "scores_path": "",
    }
    return OmegaConf.create(cfg_dict)


def test_wrapper_activates_profiler_around_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_current_profiler()`` must be non-None during the dispatcher call,
    and back to None after the wrapper returns."""
    seen: dict[str, Any] = {"profiler_active_during": None}

    def fake_run_resolved(cfg_obj, *, ddp_rank, ddp_world_size):  # noqa: ANN001, ARG001
        seen["profiler_active_during"] = get_current_profiler() is not None
        return cfg_obj

    monkeypatch.setattr(bpb_adv_cli, "_run_resolved", fake_run_resolved)
    cfg = _minimal_cfg(tmp_path)
    profile_cli._run(cfg)

    assert seen["profiler_active_during"] is True
    assert get_current_profiler() is None


def test_wrapper_writes_profile_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """events.jsonl + summary.yaml + summary.csv land under
    ``<output_root>/<run_tag>/profile/``."""
    monkeypatch.setattr(
        bpb_adv_cli,
        "_run_resolved",
        lambda cfg_obj, **_: cfg_obj,
    )
    cfg = _minimal_cfg(tmp_path)
    profile_cli._run(cfg)

    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    assert (profile_dir / "events.jsonl").is_file()
    assert (profile_dir / "summary.yaml").is_file()
    assert (profile_dir / "summary.csv").is_file()


def test_wrapper_run_tag_auto_generated_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When run_tag is None, the wrapper synchronizes one and uses it for
    the profile output dir."""
    monkeypatch.setattr(
        bpb_adv_cli,
        "_run_resolved",
        lambda cfg_obj, **_: cfg_obj,
    )
    cfg = _minimal_cfg(tmp_path)
    cfg.run_tag = None
    profile_cli._run(cfg)

    out_root = Path(cfg.output_root)
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "profile" / "events.jsonl").is_file()


def test_wrapper_propagates_stage_error_with_status_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dispatcher exception must propagate; the profiler still writes its
    summary with `status="error"`."""

    class _Boom(RuntimeError):
        pass

    def failing_dispatcher(cfg_obj, **_):  # noqa: ANN001, ARG001
        raise _Boom("dispatch fail")

    monkeypatch.setattr(bpb_adv_cli, "_run_resolved", failing_dispatcher)
    cfg = _minimal_cfg(tmp_path)
    with pytest.raises(_Boom):
        profile_cli._run(cfg)

    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "error"


def test_wrapper_metadata_includes_run_tag_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profiler's `profiler_start` event records run_tag, stage, and the
    full resolved config."""
    monkeypatch.setattr(
        bpb_adv_cli,
        "_run_resolved",
        lambda cfg_obj, **_: cfg_obj,
    )
    cfg = _minimal_cfg(tmp_path)
    profile_cli._run(cfg)

    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    events = [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start_events = [e for e in events if e.get("type") == "profiler_start"]
    assert len(start_events) == 1
    metadata = start_events[0]["attrs"]
    assert metadata["run_tag"] == "profile-wrapper-test"
    assert metadata["stage"] == "validate_source"
    assert "config" in metadata


def test_production_eval_profile_enabled_binds_profiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real BPB-Adv eval CLI, not only the benchmark wrapper, should bind
    an AxBenchProfiler when ``profile.enabled=true`` on rank 0."""
    seen: dict[str, Any] = {}

    def fake_run_resolved(cfg_obj, *, ddp_rank, ddp_world_size):  # noqa: ANN001
        seen["profiler_active"] = get_current_profiler() is not None
        seen["ddp_rank"] = ddp_rank
        seen["ddp_world_size"] = ddp_world_size
        return cfg_obj

    monkeypatch.setattr(
        bpb_adv_cli,
        "compute_init",
        lambda *, device_type: (False, 0, 0, 1, "cpu"),  # noqa: ARG005
    )
    monkeypatch.setattr(bpb_adv_cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(bpb_adv_cli, "_run_resolved", fake_run_resolved)
    monkeypatch.setattr(
        bpb_adv_cli,
        "_preflight_and_install_feature_contract_marker_ddp",
        lambda *_args, **_kwargs: None,
    )

    cfg = _minimal_cfg(tmp_path)
    cfg.device = "cuda:1"
    cfg.profile = {
        "enabled": True,
        "gpu_utilization": False,
        "synchronize_cuda": False,
    }
    bpb_adv_cli._run(cfg)

    assert seen == {"profiler_active": True, "ddp_rank": 0, "ddp_world_size": 1}
    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    assert (profile_dir / "events.jsonl").is_file()
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["metadata"]["pipeline"] == "axbench_bpb_adv"
    assert summary["metadata"]["profile_gpu_device_index"] == 1


def test_wrapper_passes_ddp_args_to_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper must forward ddp_rank/ddp_world_size to `_run_resolved`.

    Also asserts the profiler IS bound during the dispatcher call when rank
    is 0 (kills Mutation #10: invert-rank-guard) — the original test was
    permissive enough that flipping the rank guard slipped through it."""
    seen: dict[str, Any] = {}

    def fake(cfg_obj, *, ddp_rank, ddp_world_size):  # noqa: ANN001, ARG001
        seen["ddp_rank"] = ddp_rank
        seen["ddp_world_size"] = ddp_world_size
        seen["profiler_bound"] = get_current_profiler() is not None

    monkeypatch.setattr(bpb_adv_cli, "_run_resolved", fake)
    profile_cli._run(_minimal_cfg(tmp_path))

    # Without an initialized torch.distributed, the wrapper sets (0, 1).
    assert seen["ddp_rank"] == 0
    assert seen["ddp_world_size"] == 1
    # Rank 0 ⇒ profiler IS bound during the dispatcher call.
    assert seen["profiler_bound"] is True


def test_wrapper_does_not_bind_profiler_on_non_zero_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAP-5: when torch.distributed is initialized and the local rank is
    > 0, the wrapper must NOT bind an :class:`AxBenchProfiler` (concurrent
    file writes to events.jsonl would race). The production dispatcher
    still runs, just without a profiler — so `profile_region` / `record_counter`
    inside it become no-ops on those ranks.

    This test mocks ``dist.is_available()`` + ``dist.is_initialized()`` +
    ``dist.get_rank()`` + ``dist.get_world_size()`` to simulate a rank-1
    torchrun process and asserts (1) no profiler is bound during dispatch,
    (2) no profile_dir artifacts are written.
    """
    seen: dict[str, Any] = {}

    def fake_run_resolved(cfg_obj, *, ddp_rank, ddp_world_size):  # noqa: ANN001, ARG001
        seen["profiler_bound"] = get_current_profiler() is not None
        seen["ddp_rank"] = ddp_rank
        seen["ddp_world_size"] = ddp_world_size

    monkeypatch.setattr(bpb_adv_cli, "_run_resolved", fake_run_resolved)
    monkeypatch.setattr(profile_cli.dist, "is_available", lambda: True)
    monkeypatch.setattr(profile_cli.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(profile_cli.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(profile_cli.dist, "get_world_size", lambda: 2)
    # `_synchronize_run_tag` issues a real `dist.broadcast_object_list` that
    # requires an initialized process group. We bypass it for this unit test
    # by patching the helper to simply preserve the existing run_tag.
    monkeypatch.setattr(
        bpb_adv_cli,
        "_synchronize_run_tag",
        lambda cfg_obj, *, ddp_rank, ddp_world_size: None,  # noqa: ARG005
    )

    cfg = _minimal_cfg(tmp_path)
    profile_cli._run(cfg)

    # Dispatcher saw rank=1, world_size=2, and NO profiler bound.
    assert seen["ddp_rank"] == 1
    assert seen["ddp_world_size"] == 2
    assert seen["profiler_bound"] is False
    # Non-zero ranks write no profile artifacts: AxBenchProfiler is never
    # constructed, and profile_dir() is a pure path constructor (no mkdir), so
    # the directory must not exist at all on rank > 0.
    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    assert not profile_dir.exists()


def test_wrapper_accepts_stage_all_and_records_it_in_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAP-10: when ``stage=all`` is configured, the wrapper must dispatch
    cleanly and record ``stage="all"`` in the profiler_start metadata. This
    pins the multi-stage entry-point so a future change that special-cases
    individual stages can't accidentally regress the omnibus path."""
    seen: dict[str, Any] = {}

    def fake(cfg_obj, *, ddp_rank, ddp_world_size):  # noqa: ANN001, ARG001
        seen["stage"] = cfg_obj.stage

    monkeypatch.setattr(bpb_adv_cli, "_run_resolved", fake)
    cfg = _minimal_cfg(tmp_path)
    cfg.stage = "all"
    profile_cli._run(cfg)

    # Dispatcher saw stage="all".
    assert seen["stage"] == "all"
    # Metadata records stage="all".
    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    events = [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start_events = [e for e in events if e.get("type") == "profiler_start"]
    assert len(start_events) == 1
    assert start_events[0]["attrs"]["stage"] == "all"


def test_profile_wrapper_rejects_impossible_all_stage_before_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bpb_adv_cli,
        "validate_all_stage_representative_target_contract",
        validate_all_stage_representative_target_contract,
    )
    monkeypatch.setattr(
        bpb_adv_cli,
        "_synchronize_run_tag",
        lambda **_: pytest.fail("run-tag setup must not start after config rejection"),
    )
    cfg = _minimal_cfg(tmp_path)
    cfg.stage = "all"

    with pytest.raises(AxBenchBpbAdvError, match="stage='all'"):
        profile_cli._run(cfg)

    assert not (Path(cfg.output_root) / cfg.run_tag).exists()


def test_production_eval_rejects_impossible_all_stage_before_compute_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bpb_adv_cli,
        "validate_all_stage_representative_target_contract",
        validate_all_stage_representative_target_contract,
    )
    monkeypatch.setattr(
        bpb_adv_cli,
        "compute_init",
        lambda **_: pytest.fail("compute_init must not run after config rejection"),
    )
    cfg = _minimal_cfg(tmp_path)
    cfg.stage = "all"

    with pytest.raises(AxBenchBpbAdvError, match="stage='all'"):
        bpb_adv_cli._run(cfg)


def test_wrapper_metadata_includes_config_field_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAP-6: tightened metadata assertion — the previous test only checked
    `"config" in metadata`. Now we verify selected content fields land
    correctly so a mutation that records an empty dict would fail."""
    monkeypatch.setattr(
        bpb_adv_cli, "_run_resolved", lambda cfg_obj, **_: cfg_obj
    )
    cfg = _minimal_cfg(tmp_path)
    profile_cli._run(cfg)

    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    events = [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start_events = [e for e in events if e.get("type") == "profiler_start"]
    assert len(start_events) == 1
    metadata = start_events[0]["attrs"]
    cfg_dict = metadata["config"]
    assert isinstance(cfg_dict, dict)
    # Non-empty + carries the resolved scalar fields.
    assert cfg_dict["output_root"] == str(tmp_path / "out")
    assert cfg_dict["run_tag"] == "profile-wrapper-test"
    assert cfg_dict["device"] == "cpu"
    assert cfg_dict["stage"] == "validate_source"
    assert cfg_dict["target_models"] == ["decoder_only"]
    # And the adversarial_methods list is preserved (this is the most-
    # likely-to-drift nested field).
    assert (
        metadata["adversarial_methods"]
        and metadata["adversarial_methods"][0]["method"] == "additive_direction"
    )
