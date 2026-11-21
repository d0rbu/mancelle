"""Profiling contracts for self-conditioning post-training."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch as t
from torch.profiler import ProfilerActivity, profile

import nanochat.flash_attention as fa_module
import scripts.train.conditional_encdec_selfcond as selfcond_script
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import OptimizerKind
from scripts.train.conditional_encdec_selfcond import (
    _resolve_selfcond_profile_settings,
    _SelfCondProfileSession,
    _validate_selfcond_profile_window,
    run_selfcond_train_step,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondTrainConfig,
    SelfCondTrainScope,
    freeze_for_head_only,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.profiling.markers import (
    TimingRecorder,
    recorder_active,
    timed_region,
    timing_tags,
    use_profiler_markers,
    use_recorder,
)
from steering.refined_types import PositiveFloat, PositiveInt
from tests.steering.selfcond.helpers import make_batch, tiny_head_model


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU backward is unsupported by FlexAttention; use the production SDPA path."""

    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _settings(**overrides: object):
    values: dict[str, object] = {
        "profile_timing": False,
        "profile_trace": None,
        "profile_trace_steps": 1,
        "profile_output_dir": "out/profiles/encdec_selfcond",
    }
    values.update(overrides)
    return _resolve_selfcond_profile_settings(values)


def _train_config(
    *,
    scope: SelfCondTrainScope = SelfCondTrainScope.HEAD_ONLY,
) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=scope,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=0,
            max_k=1,
        ),
        seed=719,
    )


def _optimizer(model: t.nn.Module) -> t.optim.AdamW:
    freeze_for_head_only(model)
    optimizer = t.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.002,
        weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["kind"] = OptimizerKind.ADAMW
    return optimizer


def _assert_nested_exact(left: Any, right: Any) -> None:
    if isinstance(left, t.Tensor):
        assert isinstance(right, t.Tensor)
        assert t.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_exact(left_item, right_item)
    else:
        assert left == right


def _run_step(model: Any, optimizer: t.optim.Optimizer, batches: list[Any]):
    return run_selfcond_train_step(
        model,
        optimizer,
        batches,
        _train_config(),
        global_step=1,
        ddp_rank=0,
        num_iterations=PositiveInt.parse(4),
        grad_clip_max_norm=PositiveFloat.parse(1.0),
        static_max_seg_size=None,
        grad_accum_steps=2,
    )


def test_shipped_profile_defaults_are_disabled_and_side_effect_free(
    hydra_compose,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "training")
    cfg = hydra_compose(
        config_dir,
        "conditional_encdec_selfcond",
        ["model_tag=test"],
    )
    assert cfg.profile_timing is False
    assert cfg.profile_trace is None
    assert cfg.profile_trace_steps == 1
    assert cfg.profile_output_dir == "out/profiles/encdec_selfcond"

    output_dir = tmp_path / "must-not-exist"
    settings = _settings(profile_output_dir=str(output_dir))

    class UnexpectedRecorder:
        def __init__(self) -> None:
            raise AssertionError("disabled profiling constructed a timing recorder")

    monkeypatch.setattr(selfcond_script, "TimingRecorder", UnexpectedRecorder)
    session = _SelfCondProfileSession(settings, rank=0, device_type="cpu")
    session.prepare_output_dir()
    assert session.enabled is False
    assert not output_dir.exists()
    assert session.finalize() is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"profile_timing": 1}, "profile_timing must be a boolean"),
        ({"profile_trace": True}, "profile_trace must be a non-negative integer"),
        ({"profile_trace": -1}, "profile_trace must be a non-negative integer"),
        ({"profile_trace": "1"}, "profile_trace must be a non-negative integer"),
        ({"profile_trace": 1.0}, "profile_trace must be a non-negative integer"),
        ({"profile_trace_steps": False}, "profile_trace_steps must be a positive integer"),
        ({"profile_trace_steps": 0}, "profile_trace_steps must be a positive integer"),
        ({"profile_trace_steps": "2"}, "profile_trace_steps must be a positive integer"),
        ({"profile_output_dir": "  "}, "profile_output_dir must be a non-empty string"),
        ({"profile_output_dir": Path("out")}, "profile_output_dir must be a non-empty string"),
        (
            {"profile_timing": True, "profile_trace": 0},
            "profile_timing and profile_trace cannot be enabled together",
        ),
    ],
)
def test_profile_settings_reject_ambiguous_values(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings(**override)


def test_profile_settings_accept_each_diagnostic_arm_separately(tmp_path: Path) -> None:
    timing = _settings(
        profile_timing=True,
        profile_output_dir=str(tmp_path / "timing"),
    )
    assert timing.enabled
    assert timing.timing_enabled
    assert timing.trace_step is None

    trace = _settings(
        profile_trace=8,
        profile_trace_steps=3,
        profile_output_dir=str(tmp_path / "trace"),
    )
    assert trace.enabled
    assert trace.trace_step == 8
    assert trace.trace_steps == 3
    assert not trace.timing_enabled


def test_trace_window_is_absolute_resume_aware_and_closed() -> None:
    settings = _settings(profile_trace=7, profile_trace_steps=3)
    _validate_selfcond_profile_window(settings, start_step=7, num_iterations=10)

    with pytest.raises(ValueError, match="precedes resume start_step=8"):
        _validate_selfcond_profile_window(settings, start_step=8, num_iterations=12)
    with pytest.raises(ValueError, match=r"\[7, 10\).*num_iterations=9"):
        _validate_selfcond_profile_window(settings, start_step=0, num_iterations=9)

    _validate_selfcond_profile_window(
        _settings(profile_timing=True),
        start_step=99,
        num_iterations=100,
    )


class _FakeProfiler:
    def __init__(self) -> None:
        self.entered = 0
        self.exit_args: tuple[object, object, object] | None = None
        self.exported: list[Path] = []

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *args: object) -> None:
        assert len(args) == 3
        self.exit_args = (args[0], args[1], args[2])

    def export_chrome_trace(self, path: str) -> None:
        output = Path(path)
        output.write_text("{}")
        self.exported.append(output)


def test_trace_session_uses_pinned_profiler_options_and_rank_local_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = _FakeProfiler()
    captured: dict[str, object] = {}

    def fake_profile(**kwargs: object) -> _FakeProfiler:
        captured.update(kwargs)
        return profiler

    monkeypatch.setattr(selfcond_script.t.profiler, "profile", fake_profile)
    monkeypatch.setattr(
        selfcond_script.t.profiler,
        "_ExperimentalConfig",
        lambda **kwargs: dict(kwargs),
    )
    settings = _settings(
        profile_trace=4,
        profile_trace_steps=2,
        profile_output_dir=str(tmp_path),
    )
    session = _SelfCondProfileSession(settings, rank=3, device_type="cpu")
    session.prepare_output_dir()

    assert not session.should_start_trace(3)
    assert session.should_start_trace(4)
    assert not session.should_stop_trace(4)
    assert session.should_stop_trace(5)
    session.start_trace(4)
    with pytest.raises(RuntimeError, match="cannot stop at unconfigured step"):
        session.stop_and_export_trace(4)
    path = session.stop_and_export_trace(5)

    assert profiler.entered == 1
    assert profiler.exit_args == (None, None, None)
    assert captured["activities"] == [ProfilerActivity.CPU]
    assert captured["record_shapes"] is True
    assert captured["with_stack"] is True
    assert captured["experimental_config"] == {"profile_all_threads": True}
    assert path == tmp_path / "selfcond_trace_step4_rank3.json"
    assert path.read_text() == "{}"
    assert session.finalize() is None


def test_failure_cleanup_exports_partial_trace_without_masking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = _FakeProfiler()
    monkeypatch.setattr(selfcond_script.t.profiler, "profile", lambda **_kwargs: profiler)
    settings = _settings(
        profile_trace=2,
        profile_output_dir=str(tmp_path),
    )
    session = _SelfCondProfileSession(settings, rank=1, device_type="cpu")
    session.prepare_output_dir()
    session.start_trace(2)
    original = RuntimeError("training failed")

    session.close_on_error(original)

    assert profiler.exit_args is not None
    assert profiler.exit_args[0] is RuntimeError
    assert profiler.exit_args[1] is original
    assert profiler.exported == [
        tmp_path / "selfcond_trace_step2_rank1_partial.json"
    ]


def test_timing_session_exports_rank_csv_and_tags_every_row(tmp_path: Path) -> None:
    settings = _settings(
        profile_timing=True,
        profile_output_dir=str(tmp_path),
    )
    session = _SelfCondProfileSession(settings, rank=6, device_type="cpu")
    session.prepare_output_dir()
    with session.recording(11):
        with timed_region("selfcond/test_region", sync_cuda=False):
            pass
    path = session.finalize()

    assert path == tmp_path / "selfcond_timing_rank6.csv"
    assert path is not None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["name"] == "selfcond/test_region"
    assert int(rows[0]["elapsed_ns"]) >= 0
    assert rows[0]["rank"] == "6"
    assert rows[0]["step"] == "11"


def test_timing_markers_cover_every_train_phase_with_static_names() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=720)
    optimizer = _optimizer(model)
    batches = [
        make_batch([[3]], T=9, F=3, n_sae=8, seed=721),
        make_batch([[3]], T=9, F=3, n_sae=8, seed=722),
    ]
    recorder = TimingRecorder()
    with use_recorder(recorder):
        with timing_tags(rank=2, step=7):
            with timed_region("selfcond/train_step", sync_cuda=False):
                _run_step(model, optimizer, batches)

    rows = recorder.rows()
    names = [row["name"] for row in rows]
    required = {
        "selfcond/train_step",
        "selfcond/zero_grad_start",
        "selfcond/data_loading",
        "selfcond/rollout",
        "selfcond/rollout/slot",
        "selfcond/rollout/slot_gate",
        "selfcond/rollout/model_forward",
        "selfcond/rollout/pool",
        "selfcond/rollout/sample_select",
        "selfcond/rollout/write_slot",
        "selfcond/loss_forward",
        "selfcond/pass1/model_forward",
        "selfcond/pass1/pool_and_ste",
        "selfcond/pass2/encoder",
        "selfcond/pass2/lm_forward",
        "selfcond/metrics",
        "selfcond/backward",
        "selfcond/materialize_head_zero_grads",
        "selfcond/loss_scalar",
        "selfcond/grad_clip",
        "selfcond/optimizer_step",
        "selfcond/zero_grad_end",
    }
    assert set(names) == required
    assert names.count("selfcond/data_loading") == 2
    assert names.count("selfcond/rollout") == 2
    assert names.count("selfcond/backward") == 2
    assert names.count("selfcond/rollout/slot") == 4
    assert all(row["rank"] == 2 and row["step"] == 7 for row in rows)

    micro_rows = [
        row
        for row in rows
        if row["name"]
        in {
            "selfcond/data_loading",
            "selfcond/rollout",
            "selfcond/loss_forward",
            "selfcond/backward",
        }
    ]
    assert [row["micro_step"] for row in micro_rows if row["name"] == "selfcond/data_loading"] == [
        0,
        1,
    ]
    slot_rows = [row for row in rows if row["name"] == "selfcond/rollout/slot"]
    assert [(row["micro_step"], row["slot"]) for row in slot_rows] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]


def test_cpu_profiler_sees_production_nested_markers_without_timing_recorder() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=723)
    optimizer = _optimizer(model)
    batches = [
        make_batch([[2]], T=8, F=3, n_sae=8, seed=724),
        make_batch([[2]], T=8, F=3, n_sae=8, seed=725),
    ]

    with profile(activities=[ProfilerActivity.CPU], with_stack=False) as profiler:
        with timed_region("selfcond/train_step", sync_cuda=False):
            _run_step(model, optimizer, batches)

    event_names = {event.name for event in profiler.events()}
    assert {
        "selfcond/train_step",
        "selfcond/data_loading",
        "selfcond/rollout/model_forward",
        "selfcond/pass1/model_forward",
        "selfcond/pass2/encoder",
        "selfcond/backward",
        "selfcond/optimizer_step",
    } <= event_names


def test_explicit_listener_restores_markers_for_all_thread_profiler_without_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selfcond_script.t.cuda,
        "synchronize",
        lambda: pytest.fail("trace-only marker scope synchronized CUDA"),
    )
    assert not recorder_active()
    with profile(
        activities=[ProfilerActivity.CPU],
        with_stack=False,
        experimental_config=t.profiler._ExperimentalConfig(
            profile_all_threads=True
        ),
    ) as profiler:
        with use_profiler_markers():
            assert recorder_active()
            with timed_region("selfcond/all_thread_listener"):
                _ = t.ones(2) + 1
    assert not recorder_active()
    assert "selfcond/all_thread_listener" in {
        event.name for event in profiler.events()
    }


def test_timing_instrumentation_is_bit_exact_for_outputs_model_and_optimizer() -> None:
    plain_model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=726)
    timed_model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=726)
    plain_optimizer = _optimizer(plain_model)
    timed_optimizer = _optimizer(timed_model)
    plain_batches = [
        make_batch([[3]], T=9, F=3, n_sae=8, seed=727),
        make_batch([[2, 1]], T=9, F=3, n_sae=8, seed=728),
    ]
    timed_batches = [
        make_batch([[3]], T=9, F=3, n_sae=8, seed=727),
        make_batch([[2, 1]], T=9, F=3, n_sae=8, seed=728),
    ]

    plain_result = _run_step(plain_model, plain_optimizer, plain_batches)
    recorder = TimingRecorder()
    with use_recorder(recorder):
        with timing_tags(rank=0, step=1):
            timed_result = _run_step(timed_model, timed_optimizer, timed_batches)

    assert plain_result == timed_result
    _assert_nested_exact(plain_model.state_dict(), timed_model.state_dict())
    _assert_nested_exact(plain_optimizer.state_dict(), timed_optimizer.state_dict())
    assert len(recorder) > 0
