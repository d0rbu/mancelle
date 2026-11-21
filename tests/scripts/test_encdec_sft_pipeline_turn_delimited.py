import sys
from unittest.mock import patch

import pytest

from scripts.train.run_encdec_sft_pipeline import (
    build_feature_dir_for_phase,
    main,
    run_pipeline,
)
from steering.segmentation import SegmentMode


def test_build_feature_dir_for_phase_appends_turn_leaf():
    path = build_feature_dir_for_phase(
        feature_root="/features",
        segment_size=128,
        layer=8,
        activation_type="residual",
        pooling="max_pool",
        sae_width="16k",
        model="gemma-2-2b",
        dataset_phase="standard_sft",
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    assert path.endswith("/standard_sft/no_bos/turn_delimited")


def test_cli_extracts_feature_segment_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_encdec_sft_pipeline",
            "model_tag=d12",
            "feature_segment_mode=turn_delimited_fixed_tokens",
        ],
    )
    with patch("scripts.train.run_encdec_sft_pipeline.run_pipeline") as run:
        main()
    assert run.call_args.kwargs["feature_segment_mode"] == "turn_delimited_fixed_tokens"


def test_invalid_feature_segment_mode_raises():
    with pytest.raises(ValueError, match="not a valid SegmentMode"):
        run_pipeline(model_tag="d12", feature_segment_mode="sentences")


@pytest.mark.parametrize(
    "feature_segment_mode,expected_mode,expected_suffix",
    [
        (None, SegmentMode.fixed_tokens, "/standard_sft/no_bos"),
        (
            SegmentMode.turn_delimited_fixed_tokens.value,
            SegmentMode.turn_delimited_fixed_tokens,
            "/standard_sft/no_bos/turn_delimited",
        ),
    ],
)
def test_run_pipeline_propagates_segment_mode_to_every_phase_and_consumer(
    feature_segment_mode,
    expected_mode,
    expected_suffix,
):
    kwargs = {
        "model_tag": "d12",
        "phase_order": "safety,standard",
        "feature_root": "/features",
    }
    if feature_segment_mode is not None:
        kwargs["feature_segment_mode"] = feature_segment_mode

    with (
        patch("scripts.train.run_encdec_sft_pipeline.get_base_dir", return_value="/base"),
        patch(
            "scripts.train.run_encdec_sft_pipeline.build_feature_dir_for_phase",
            wraps=build_feature_dir_for_phase,
        ) as build,
        patch("scripts.train.run_encdec_sft_pipeline.ensure_feature_statistics") as stats,
        patch("scripts.train.run_encdec_sft_pipeline.verify_feature_dir") as verify,
        patch("scripts.train.run_encdec_sft_pipeline.run_sft") as sft,
    ):
        run_pipeline(**kwargs)

    assert build.call_count == 2
    assert all(
        item.kwargs["segment_mode"] is expected_mode for item in build.call_args_list
    )
    feature_dirs = [item.args[0] for item in stats.call_args_list]
    assert [item.args[0] for item in verify.call_args_list] == feature_dirs
    assert [item.kwargs["feature_dir"] for item in sft.call_args_list] == feature_dirs
    expected_phases = ("safety_sft", "standard_sft")
    for feature_dir, phase in zip(feature_dirs, expected_phases, strict=True):
        assert feature_dir.endswith(expected_suffix.replace("standard_sft", phase))
        if expected_mode is SegmentMode.fixed_tokens:
            assert not feature_dir.endswith("/turn_delimited")


def test_run_pipeline_propagates_feature_directory_verification_error():
    with (
        patch("scripts.train.run_encdec_sft_pipeline.get_base_dir", return_value="/base"),
        patch("scripts.train.run_encdec_sft_pipeline.ensure_feature_statistics"),
        patch(
            "scripts.train.run_encdec_sft_pipeline.verify_feature_dir",
            side_effect=ValueError("wrong segment mode"),
        ) as verify,
        patch("scripts.train.run_encdec_sft_pipeline.run_sft") as sft,
        pytest.raises(ValueError, match="wrong segment mode"),
    ):
        run_pipeline(model_tag="d12", phase_order="safety")
    verify.assert_called_once()
    sft.assert_not_called()
