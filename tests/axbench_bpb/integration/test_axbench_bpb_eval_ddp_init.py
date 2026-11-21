"""Regression tests for axbench_bpb_eval.main()'s DDP init + cleanup
wiring.

Commit 03b25fe9 added ``compute_init(device_type=cfg.device)`` at the
start of ``main`` and wrapped the stage loop in
``try/finally: compute_cleanup()``. Without these calls, every rank
under torchrun silently saw ``world_size=1`` and re-ran the full
pipeline against shared on-disk paths.

These tests mirror the axbench-base equivalents at
``tests/axbench_base/integration/test_axbench_base_eval_ddp_init.py``.

Test plan:
1. compute_init is called BEFORE the script reads dist coordinates
2. compute_cleanup runs on the success path
3. compute_cleanup runs on the exception path
4. Under a real 2-rank gloo spawn, ranks see distinct rank/ws (skipped)
5. Single-process (no torchrun env) does NOT call init_process_group
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch.distributed as dist
from omegaconf import OmegaConf

from scripts.eval import axbench_bpb_eval as cli
from steering.eval.axbench_bpb.types import Stage


def _build_minimal_cfg(tmp_path: Path) -> Any:
    """Build a DictConfig minimally sufficient for main() before it
    delegates to _run_stage. The script reads source_run_dir,
    output_root, feature_dir, feature_statistics_path, decoder_only,
    encdec, device, run_tag, stage. Everything else is defaulted by
    the YAML."""
    return OmegaConf.create({
        "source_run_dir": str(tmp_path / "source"),
        "output_root": str(tmp_path / "out"),
        "feature_dir": str(tmp_path / "feat"),
        "feature_statistics_path": str(tmp_path / "stats.safetensors"),
        "extraction_tokenizer_id": "google/gemma-2-2b",
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_source": "8-gemmascope-res-16k",
        "extraction_max_tokens": 2048,
        "extraction_threshold": 0.0,
        "decoder_only": {"source": "x", "model_tag": "y", "step": 1},
        "encdec": {"source": "x", "model_tag": "y", "step": 1},
        "device": "cpu",  # cpu keeps compute_init a near-no-op
        "diffmean_and_probe_layer": 8,
        "lora": {"rank": 16, "alpha": 32.0},
        "stage": "prepare_context_cache",
        "enabled_methods": None,
        "disabled_methods": [],
        "filter_unaccepted_samples": True,
        "device_batch_size": 16,
        "context_cache": {
            "extraction_batch_size": 32, "force_rebuild": False,
            "segment_size": 256, "num_buckets": 2,
        },
        "aggregation": {
            "min_rows_per_side": 10, "invalid_warning_threshold": 0.01,
            "invalid_drop_threshold": 0.05,
            "primary_metric": "macro_mean_best_primary_margin",
        },
        "log_progress_every": 1000,
        "offload_between_stages": True,
        "run_tag": "test-ddp-init",
    })


def _make_source_artifacts(tmp_path: Path) -> None:
    """Create the minimum-viable source_run_dir artifacts that
    validate_bpb_config / SourcePaths.validate_stage_0_3_artifacts
    check at startup. We don't need realistic content because the
    stage handler is stubbed."""
    src = tmp_path / "source"
    src.mkdir(parents=True, exist_ok=True)
    (src / "concepts_stage1c.json").write_text("[]")
    (src / "synth_training_positive.parquet").write_text("")
    (src / "synth_training_negative.parquet").write_text("")
    (src / "concept_features.safetensors").write_text("")
    (src / "concept_features_neg.safetensors").write_text("")
    (src / "baselines").mkdir(exist_ok=True)


# ---------- Unit: compute_init then dist coordinate read ----------


def test_main_calls_compute_init_first_and_cleanup_last(monkeypatch, tmp_path):
    """The script must call compute_init at the top of main() and
    compute_cleanup at the end. We spy on compute_init, compute_cleanup,
    and _run_stage; the call log must show compute_init first, the
    stage handler in the middle, and compute_cleanup last.

    This pins commit 03b25fe9's contract -- without these calls, every
    rank under torchrun silently saw world_size=1.
    """
    call_log: list[str] = []

    def _spy_init(device_type: str = "cuda"):
        call_log.append(f"compute_init(device_type={device_type})")
        return False, 0, 0, 1, None  # mimics single-process return

    def _spy_cleanup():
        call_log.append("compute_cleanup")

    def _spy_run_stage(stage: Any, rs: Any):
        call_log.append(f"run_stage({Stage.coerce(stage).value})")

    monkeypatch.setattr(cli, "compute_init", _spy_init)
    monkeypatch.setattr(cli, "compute_cleanup", _spy_cleanup)
    monkeypatch.setattr(cli, "_run_stage", _spy_run_stage)
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    cli.main(cfg)

    # compute_init must come first; compute_cleanup must come last.
    assert call_log[0] == "compute_init(device_type=cpu)", (
        f"main() must call compute_init first; log was {call_log!r}"
    )
    assert call_log[-1] == "compute_cleanup", (
        f"main() must call compute_cleanup last; log was {call_log!r}"
    )
    # Stage was invoked between init and cleanup.
    assert "run_stage(prepare_context_cache)" in call_log


def test_main_calls_compute_cleanup_on_exception(monkeypatch, tmp_path):
    """If _run_stage raises, compute_cleanup must STILL run -- without
    this the process group leaks into subsequent runs."""
    cleanup_calls: list[int] = []

    monkeypatch.setattr(cli, "compute_init", lambda **_: None)
    monkeypatch.setattr(
        cli, "compute_cleanup", lambda: cleanup_calls.append(1),
    )
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)

    def _raising_stage(stage: Any, rs: Any):
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(cli, "_run_stage", _raising_stage)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="simulated stage failure"):
        cli.main(cfg)

    assert cleanup_calls == [1], (
        f"compute_cleanup must run exactly once on the exception path; "
        f"got {cleanup_calls!r}"
    )


# ---------- Functional: single-process behavior with no torchrun env ----------


def test_main_single_process_does_not_call_init_process_group(
    monkeypatch, tmp_path,
):
    """Without torchrun env vars set, compute_init falls through
    without calling dist.init_process_group. We spy on the real init
    helper to assert it was never invoked."""
    for var in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    ):
        monkeypatch.delenv(var, raising=False)

    init_calls: list[Any] = []

    def _spy_init_pg(*args: Any, **kwargs: Any):
        init_calls.append((args, kwargs))

    monkeypatch.setattr(dist, "init_process_group", _spy_init_pg)
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)
    monkeypatch.setattr(cli, "_run_stage", lambda stage, rs: None)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    cli.main(cfg)

    assert init_calls == [], (
        f"Single-process main() must not invoke init_process_group; "
        f"got {init_calls!r}"
    )


# ---------- Integration: real gloo spawn assigns distinct ranks ----------


def _ddp_init_target(
    rank: int, world_size: int, tmp_path_str: str,
) -> dict[str, Any]:
    """Worker target: invoke main() with stubbed stage handler. Returns
    the (ddp_rank, ddp_world_size) the RuntimeState saw."""
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)

    seen: dict[str, Any] = {
        "ddp_rank": None, "ddp_world_size": None, "stages_invoked": [],
    }

    def _spy_run_stage(stage: Any, rs: Any):
        seen["ddp_rank"] = int(rs.ddp_rank)
        seen["ddp_world_size"] = int(rs.ddp_world_size)
        seen["stages_invoked"].append(Stage.coerce(stage).value)

    # This test intentionally replaces strongly typed module functions with
    # lightweight test doubles, so make that boundary explicit.
    cli_module: Any = cli
    cli_module._run_stage = _spy_run_stage
    cli_module.validate_bpb_config = lambda _: None

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    # Override device to cpu so compute_init is the no-op CPU path
    # (gloo backend handles cross-process barriers without nccl).
    cfg.device = "cpu"

    cli.main(cfg)
    return seen


# ---------- Post-review hardening: C1 (validate-failure leak), C2 (device
# normalization), C4 (deterministic run_tag), plus Must-add coverage for
# Stage.ALL iteration and auto-run-tag generation. ----------


def test_main_does_not_init_pg_when_validate_config_raises(monkeypatch, tmp_path):
    """C1: validation runs BEFORE compute_init, so a validation failure
    must not leak the process group. Spy on compute_init / compute_cleanup;
    when validate_bpb_config raises, neither should have been called.

    Pre-fix, validate ran AFTER compute_init — a validation failure would
    leak the PG into subsequent runs.
    """
    init_calls: list[Any] = []
    cleanup_calls: list[Any] = []

    monkeypatch.setattr(
        cli, "compute_init", lambda **kw: init_calls.append(kw),
    )
    monkeypatch.setattr(
        cli, "compute_cleanup", lambda: cleanup_calls.append(1),
    )

    def _raising_validate(_cfg: Any) -> None:
        from steering.eval.axbench_bpb.types import AxBenchBpbError
        raise AxBenchBpbError("simulated validation failure")

    monkeypatch.setattr(cli, "validate_bpb_config", _raising_validate)
    monkeypatch.setattr(cli, "_run_stage", lambda *_a, **_kw: None)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    with pytest.raises(Exception, match="simulated validation failure"):
        cli.main(cfg)

    assert init_calls == [], (
        f"compute_init must NOT be called before validation — got {init_calls!r}"
    )
    assert cleanup_calls == [], (
        f"compute_cleanup must NOT be called when init never ran — got {cleanup_calls!r}"
    )


def test_main_calls_compute_cleanup_when_compute_init_raises(monkeypatch, tmp_path):
    """C1 (defense-in-depth): compute_init runs INSIDE the try/finally
    so a partial-init failure (e.g., init_process_group succeeds but
    the post-init barrier raises) does not leak the PG. compute_cleanup
    is called even when compute_init raises; it short-circuits safely
    when is_ddp_initialized() is False."""
    cleanup_calls: list[int] = []

    def _raising_init(**_kw: Any) -> None:
        raise RuntimeError("simulated compute_init failure")

    monkeypatch.setattr(cli, "compute_init", _raising_init)
    monkeypatch.setattr(
        cli, "compute_cleanup", lambda: cleanup_calls.append(1),
    )
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)
    monkeypatch.setattr(cli, "_run_stage", lambda *_a, **_kw: None)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="simulated compute_init failure"):
        cli.main(cfg)

    assert cleanup_calls == [1], (
        f"compute_cleanup must run even when compute_init raises (so a "
        f"partially-initialized PG can be torn down); got {cleanup_calls!r}"
    )


def test_main_does_not_init_pg_when_cfg_to_dataclass_raises(monkeypatch, tmp_path):
    """C1 (companion): _cfg_to_dataclass also runs BEFORE compute_init.
    If it raises (e.g. unknown config key), the PG must not be touched."""
    init_calls: list[Any] = []
    cleanup_calls: list[Any] = []

    monkeypatch.setattr(
        cli, "compute_init", lambda **kw: init_calls.append(kw),
    )
    monkeypatch.setattr(
        cli, "compute_cleanup", lambda: cleanup_calls.append(1),
    )

    def _raising_cfg(_cfg: Any) -> Any:
        from steering.eval.axbench_bpb.types import AxBenchBpbError
        raise AxBenchBpbError("Unknown axbench-bpb config keys: ['typo']")

    monkeypatch.setattr(cli, "_cfg_to_dataclass", _raising_cfg)
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)
    monkeypatch.setattr(cli, "_run_stage", lambda *_a, **_kw: None)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    with pytest.raises(Exception, match="Unknown.*config keys"):
        cli.main(cfg)

    assert init_calls == []
    assert cleanup_calls == []


def test_main_normalizes_device_string_for_compute_init(monkeypatch, tmp_path):
    """C2: cfg.device='cuda:0' (or 'cuda:7' etc.) must be normalized to
    'cuda' before being passed to compute_init, which only accepts the
    bare DeviceType StrEnum value. Pre-fix, compute_init's
    ``assert device_type in DeviceType`` crashed on any pinned form."""
    seen_device: list[str] = []

    def _spy_init(*, device_type: str) -> None:
        seen_device.append(device_type)

    monkeypatch.setattr(cli, "compute_init", _spy_init)
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)
    monkeypatch.setattr(cli, "_run_stage", lambda *_a, **_kw: None)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    cfg.device = "cuda:0"  # the operator's typical pinned-form
    cli.main(cfg)

    assert seen_device == ["cuda"], (
        f"compute_init must receive 'cuda' (not 'cuda:0'); got {seen_device!r}"
    )


def test_device_type_from_cfg_strips_pin_and_lowercases():
    """C2 (unit): the helper strips the `:N` suffix and lowercases."""
    assert cli._device_type_from_cfg("cuda") == "cuda"
    assert cli._device_type_from_cfg("cuda:0") == "cuda"
    assert cli._device_type_from_cfg("cuda:7") == "cuda"
    assert cli._device_type_from_cfg("CUDA:0") == "cuda"
    assert cli._device_type_from_cfg("cpu") == "cpu"
    assert cli._device_type_from_cfg("mps") == "mps"
    assert cli._device_type_from_cfg(" cuda ") == "cuda"


def test_device_type_from_cfg_rejects_unsupported_devices():
    """C2 (unit): explicitly unsupported devices raise a clear domain
    error rather than passing through to compute_init's opaque assert."""
    from steering.eval.axbench_bpb.types import AxBenchBpbError
    with pytest.raises(AxBenchBpbError, match="Unsupported"):
        cli._device_type_from_cfg("xpu")
    with pytest.raises(AxBenchBpbError, match="Unsupported"):
        cli._device_type_from_cfg("tpu:0")


def test_generate_run_tag_is_deterministic_for_same_cfg():
    """C4: two calls with the same cfg in the same UTC second produce
    the same tag. Under torchrun this guarantees every rank resolves
    to the same output tree."""
    from steering.eval.axbench_bpb.config import AxBenchBpbConfig

    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/src",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    tag1 = cli._generate_run_tag(cfg)
    tag2 = cli._generate_run_tag(cfg)
    # Same cfg → same suffix. Timestamp may differ if we crossed a UTC
    # second boundary; assert at least the suffix matches.
    assert tag1.split("-")[-1] == tag2.split("-")[-1], (
        f"_generate_run_tag must be deterministic over cfg; got {tag1!r} vs {tag2!r}"
    )


def test_generate_run_tag_changes_when_cfg_changes():
    """C4: cfg sensitivity. Different cfgs must produce different
    tag suffixes so re-runs against different inputs land in different
    output trees."""
    from steering.eval.axbench_bpb.config import AxBenchBpbConfig

    cfg_a = AxBenchBpbConfig(
        source_run_dir="/tmp/src_a",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    cfg_b = AxBenchBpbConfig(
        source_run_dir="/tmp/src_b",  # only difference
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    tag_a = cli._generate_run_tag(cfg_a)
    tag_b = cli._generate_run_tag(cfg_b)
    assert tag_a.split("-")[-1] != tag_b.split("-")[-1], (
        f"Different cfgs must produce different tag suffixes; "
        f"got {tag_a!r} == {tag_b!r}"
    )


def test_generate_run_tag_format_is_yyyymmdd_hhmmss_8hex():
    """C4 (format pin): the tag matches YYYYMMDD-HHMMSS-<8 hex>."""
    import re

    from steering.eval.axbench_bpb.config import AxBenchBpbConfig

    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/src",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    tag = cli._generate_run_tag(cfg)
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", tag), (
        f"run_tag format violation: {tag!r}"
    )


def test_main_auto_generates_run_tag_when_none(monkeypatch, tmp_path):
    """Must-add: cfg.run_tag = null path is exercised. Auto-generated
    tag is non-empty and matches the format contract."""
    captured: dict = {}

    def _spy_run_stage(stage: Any, rs: Any) -> None:
        captured["run_tag"] = rs.cfg.run_tag

    monkeypatch.setattr(cli, "compute_init", lambda **_: None)
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)
    monkeypatch.setattr(cli, "_run_stage", _spy_run_stage)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    cfg.run_tag = None  # trigger the auto-gen branch

    cli.main(cfg)
    assert captured.get("run_tag") is not None
    assert len(captured["run_tag"]) > 0


def test_main_handles_stage_all_iteration(monkeypatch, tmp_path):
    """Stage.ALL expansion runs all stages in canonical
    order. The ALL path was previously uncovered (tests pinned a
    single stage)."""
    invocations: list[str] = []

    def _spy_run_stage(stage: Any, rs: Any) -> None:
        invocations.append(Stage.coerce(stage).value)

    monkeypatch.setattr(cli, "compute_init", lambda **_: None)
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(cli, "validate_bpb_config", lambda _: None)
    monkeypatch.setattr(cli, "_run_stage", _spy_run_stage)

    _make_source_artifacts(tmp_path)
    cfg = _build_minimal_cfg(tmp_path)
    cfg.stage = "all"

    cli.main(cfg)
    assert invocations == [
        "prepare_context_cache",
        "bpb_score",
        "bpb_aggregate",
    ], f"Stage.ALL must dispatch in canonical order; got {invocations!r}"


def test_resolve_stages_to_run_for_all_returns_canonical_order():
    """Unit: the helper produces the canonical stage list for
    Stage.ALL."""
    stages = cli._resolve_stages_to_run(Stage.ALL)
    assert stages == [
        Stage.PREPARE_CONTEXT_CACHE, Stage.BPB_SCORE, Stage.BPB_AGGREGATE,
    ]


def test_resolve_stages_to_run_for_single_stage_returns_singleton():
    """Unit: any non-ALL Stage returns a one-element list."""
    for stage in (
        Stage.PREPARE_CONTEXT_CACHE, Stage.BPB_SCORE, Stage.BPB_AGGREGATE,
    ):
        assert cli._resolve_stages_to_run(stage) == [stage]


@pytest.mark.skip(
    reason=(
        "compute_init only initializes the PG when device_type==cuda. "
        "Multi-process gloo PG is set up by the gloo_multiprocess "
        "fixture, but axbench_bpb_eval.main() runs on cpu so it does "
        "not init the PG itself. The fixture-provided PG IS visible to "
        "main() via dist.is_initialized() True. Enable this test once "
        "compute_init grows a gloo-on-cpu fast path or restructure the "
        "test to call into the dist coordinate read directly."
    )
)
def test_main_under_torchrun_assigns_distinct_ranks(
    tmp_path, gloo_multiprocess,
):
    """When 2 worker processes are spawned with gloo (which emulates
    torchrun's RANK/WORLD_SIZE/LOCAL_RANK env vars), each rank's
    RuntimeState must see its own ddp_rank and a shared
    ddp_world_size == 2."""
    results = gloo_multiprocess(2, _ddp_init_target, str(tmp_path))
    rank0 = results[0]
    rank1 = results[1]

    assert rank0["ddp_rank"] == 0
    assert rank1["ddp_rank"] == 1
    assert rank0["ddp_world_size"] == 2
    assert rank1["ddp_world_size"] == 2
