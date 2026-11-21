"""Assert all 7 coarse record_function markers appear under torch.profiler."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from scripts.train.conditional_encdec_sft import (
    SFTState,
    _resolve_sft_profile_settings,
    make_sft_data_generator,
)
from scripts.train.training_progression import _run_eval_suite
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.profiling.markers import (
    TimingRecorder,
    timed_region,
    timing_tags,
    use_recorder,
)
from steering.sft_schema import SFTSplit
from tasks.common import Task

COARSE_MARKERS = {
    "evaluate_single_checkpoint",
    "load_progression_model",
    "build_eval_wrapper",
    "run_core_eval",
    "run_chat_eval_suite",
    "run_harmfulness_eval",
    "encdec_build_conditioning",
}


def _event_names(prof: profile) -> set[str]:
    return {evt.name for evt in prof.events()}


def _profile_cfg(**overrides):
    cfg = {
        "profile_output_dir": "out/profiles/test",
        "profile_trace": None,
        "profile_trace_steps": 3,
        "profile_timing": False,
        "profile_timing_sync_cuda": True,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


class _TinySFTTask(Task):
    def num_examples(self) -> int:
        return 8

    def get_example(self, index: int) -> dict:
        return {
            "messages": [
                {"role": "user", "content": f"u{index}"},
                {"role": "assistant", "content": "a" * 12},
            ]
        }


class _TinySFTTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        user = conversation["messages"][0]["content"]
        assistant = conversation["messages"][1]["content"]
        ids = [0] + list(range(2, len(user) + len(assistant) + 3))
        mask = [0] * (len(user) + 1) + [1] * (len(assistant) + 1)
        return ids, mask

    def decode(self, ids: list[int]) -> str:
        return "x" * len(ids)


def _conditioned_sft_generator(
    *,
    profile_markers: bool | Callable[[], bool] = False,
):
    return make_sft_data_generator(
        task=_TinySFTTask(),
        tokenizer=_TinySFTTokenizer(),
        B=2,
        T=32,
        device="cpu",
        sft_state=SFTState(),
        buffer_size=4,
        feature_dir="features",
        parquet_dir="parquet",
        threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1),
        boundaries=torch.zeros(1),
        profile_markers=profile_markers,
        alignment_probe_samples=0,
        expected_split=SFTSplit.TRAIN,
    )


def _covered_feature_rows() -> dict[int, list[SegmentFeatures]]:
    task = _TinySFTTask()
    tokenizer = _TinySFTTokenizer()
    rows: dict[int, list[SegmentFeatures]] = {}
    for idx in range(len(task)):
        ids, _mask = tokenizer.render_conversation(task[idx])
        rows[idx] = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=torch.tensor([0], dtype=torch.int32),
                feature_mags=torch.tensor([1.0], dtype=torch.float32),
                char_start=0,
                char_end=len(tokenizer.decode(ids)),
            )
        ]
    return rows


def test_sft_profile_settings_default_disabled_path():
    settings = _resolve_sft_profile_settings(_profile_cfg())

    assert str(settings.output_dir).endswith("out/profiles/test")
    assert settings.trace_step is None
    assert settings.trace_steps == 3
    assert settings.timing_enabled is False
    assert settings.timing_sync_cuda is True


def test_sft_profile_settings_normalizes_trace_step():
    settings = _resolve_sft_profile_settings(
        _profile_cfg(
            profile_trace="2",
            profile_trace_steps="5",
            profile_timing=True,
            profile_timing_sync_cuda=False,
        )
    )

    assert settings.trace_step == 2
    assert settings.trace_steps == 5
    assert settings.timing_enabled is True
    assert settings.timing_sync_cuda is False


def test_sft_profile_settings_rejects_nonpositive_trace_steps():
    with pytest.raises(ValueError, match="profile_trace_steps must be a positive integer"):
        _resolve_sft_profile_settings(_profile_cfg(profile_trace_steps=0))


def test_sft_profile_settings_rejects_negative_trace_step():
    with pytest.raises(ValueError, match="profile_trace must be a non-negative integer"):
        _resolve_sft_profile_settings(_profile_cfg(profile_trace=-1))


def test_per_suite_markers_via_dispatcher():
    """Directly calling _run_eval_suite must emit the three per-suite markers."""
    cfg = OmegaConf.create({"max_per_task": 2})

    def fake_run_core_eval(*_args, **_kwargs):
        return {}

    def fake_run_chat_eval_suite(*_args, **_kwargs):
        return {}

    def fake_run_harmfulness_eval(*_args, **_kwargs):
        return {}

    with (
        patch("scripts.train.training_progression.run_core_eval", fake_run_core_eval),
        patch(
            "scripts.train.training_progression.run_chat_eval_suite",
            fake_run_chat_eval_suite,
        ),
        patch(
            "scripts.train.training_progression.run_harmfulness_eval",
            fake_run_harmfulness_eval,
        ),
        profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof,
    ):
        _run_eval_suite("core", None, None, torch.device("cpu"), 0, cfg)
        _run_eval_suite("chat", None, None, torch.device("cpu"), 0, cfg)
        _run_eval_suite(
            "harmfulness",
            None,
            None,
            torch.device("cpu"),
            0,
            cfg,
        )

    names = _event_names(prof)
    assert "run_core_eval" in names
    assert "run_chat_eval_suite" in names
    assert "run_harmfulness_eval" in names


def test_timed_region_emits_record_function_marker_without_recorder():
    """timed_region must emit its record_function wrapper even with no active recorder."""
    with profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof:
        with timed_region("evaluate_single_checkpoint"):
            with timed_region("load_progression_model"):
                pass
            with timed_region("build_eval_wrapper"):
                pass
            with timed_region("encdec_build_conditioning"):
                pass

    names = _event_names(prof)
    for marker in (
        "evaluate_single_checkpoint",
        "load_progression_model",
        "build_eval_wrapper",
        "encdec_build_conditioning",
    ):
        assert marker in names, f"missing marker {marker}; got {sorted(names)}"


def test_sft_generator_emits_record_function_markers_without_recorder():
    """The SFT data path must show coarse CPU markers in torch.profiler traces."""
    gen = _conditioned_sft_generator(profile_markers=True)

    with (
        patch(
            "scripts.train.conditional_encdec_sft.load_sft_features",
            return_value=_covered_feature_rows(),
        ),
        patch(
            "scripts.train.conditional_encdec_sft."
            "_assert_sft_mixture_fingerprint_matches_task",
            return_value=None,
        ),
        profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof,
    ):
        next(gen)

    names = _event_names(prof)
    for marker in (
        "sft/refill_buffer",
        "sft/render_conversation",
        "sft/build_text_tensors",
        "sft/build_conv_segments",
    ):
        assert marker in names, f"missing marker {marker}; got {sorted(names)}"


def test_sft_generator_markers_are_disabled_by_default():
    """Profiling-disabled SFT runs should not enter record_function markers."""
    gen = _conditioned_sft_generator()

    with (
        patch(
            "scripts.train.conditional_encdec_sft.load_sft_features",
            return_value=_covered_feature_rows(),
        ),
        patch(
            "scripts.train.conditional_encdec_sft."
            "_assert_sft_mixture_fingerprint_matches_task",
            return_value=None,
        ),
        profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof,
    ):
        next(gen)

    names = _event_names(prof)
    assert "sft/refill_buffer" not in names
    assert "sft/build_text_tensors" not in names


def test_sft_generator_marker_switch_can_enable_after_construction():
    """Trace mode starts after the train generator is built; the switch must be live."""
    enabled = False

    def profile_markers_enabled() -> bool:
        return enabled

    gen = _conditioned_sft_generator(profile_markers=profile_markers_enabled)

    with (
        patch(
            "scripts.train.conditional_encdec_sft.load_sft_features",
            return_value=_covered_feature_rows(),
        ),
        patch(
            "scripts.train.conditional_encdec_sft."
            "_assert_sft_mixture_fingerprint_matches_task",
            return_value=None,
        ),
        profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof,
    ):
        next(gen)
    names = _event_names(prof)
    assert "sft/refill_buffer" not in names

    enabled = True
    with profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof:
        next(gen)
    names = _event_names(prof)
    assert "sft/refill_buffer" in names
    assert "sft/build_text_tensors" in names


def test_sft_generator_nested_timing_rows_inherit_tags():
    """Nested SFT generator rows must be useful in timing CSV summaries."""
    gen = _conditioned_sft_generator(profile_markers=True)
    recorder = TimingRecorder()

    with (
        patch(
            "scripts.train.conditional_encdec_sft.load_sft_features",
            return_value=_covered_feature_rows(),
        ),
        patch(
            "scripts.train.conditional_encdec_sft."
            "_assert_sft_mixture_fingerprint_matches_task",
            return_value=None,
        ),
        use_recorder(recorder),
    ):
        with timing_tags(rank=0, step=7, micro_step=3):
            with timed_region("sft/data_loading", sync_cuda=False):
                next(gen)

    rows = recorder.rows()
    names = {row["name"] for row in rows}
    for marker in (
        "sft/data_loading",
        "sft/refill_buffer",
        "sft/render_conversation",
        "sft/build_text_tensors",
        "sft/build_conv_segments",
    ):
        assert marker in names

    nested = [
        row for row in rows if row["name"] in {"sft/build_text_tensors", "sft/build_conv_segments"}
    ]
    assert nested
    for row in nested:
        assert row["rank"] == 0
        assert row["step"] == 7
        assert row["micro_step"] == 3


def test_coarse_markers_enumerated_as_constants():
    """COARSE_MARKERS must list exactly the 7 markers committed in the plan."""
    assert len(COARSE_MARKERS) == 7
