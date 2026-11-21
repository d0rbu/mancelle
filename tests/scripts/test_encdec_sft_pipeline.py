"""Conditioned-only tests for the encoder-decoder SFT pipeline."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from errors import FeatureClustersError
from scripts.train.run_encdec_sft_pipeline import (
    CONDITIONED_FEATURE_DATASETS,
    build_checkpoint_dir,
    build_feature_dir_for_phase,
    build_sft_checkpoint_dir,
    parse_phase_order,
    validate_phase_order,
)
from scripts.train.run_encdec_sft_pipeline import (
    run_pipeline as _run_pipeline,
)
from scripts.train.run_encdec_sft_pipeline import (
    run_sft as _run_sft,
)
from steering.feature_paths import DatasetPhase


def run_sft(*args: Any, **kwargs: Any) -> Any:
    """Pass raw command values through the production beartype boundary."""
    return _run_sft(*args, **kwargs)


def run_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Pass raw pipeline values through the production beartype boundary."""
    return _run_pipeline(*args, **kwargs)


class TestPhaseOrder:
    def test_parse_default(self) -> None:
        assert parse_phase_order(None) == ["safety", "capabilities"]

    def test_parse_comma_form(self) -> None:
        assert parse_phase_order(" capabilities , safety ") == ["capabilities", "safety"]

    def test_rejects_hydra_brackets(self) -> None:
        with pytest.raises(ValueError, match="comma-separated"):
            parse_phase_order("[safety,capabilities]")

    def test_validate_requires_nonempty_unique_supported_phases(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            validate_phase_order([])
        with pytest.raises(ValueError, match="duplicate"):
            validate_phase_order(["safety", "safety"])
        with pytest.raises(
            ValueError,
            match=r"pre-extracted SFT feature phases.*capabilities.*safety.*standard",
        ):
            validate_phase_order(["math"])
        assert validate_phase_order([" capabilities ", "safety"]) == [
            "capabilities",
            "safety",
        ]
        assert validate_phase_order(parse_phase_order("standard")) == ["standard"]
        assert validate_phase_order(parse_phase_order("safety,capabilities,standard")) == [
            "safety",
            "capabilities",
            "standard",
        ]

    def test_conditioned_dataset_set_matches_feature_phases(self) -> None:
        expected = {
            phase.value.removesuffix("_sft")
            for phase in DatasetPhase
            if phase.value.endswith("_sft")
        }
        assert CONDITIONED_FEATURE_DATASETS == frozenset(expected)


def test_build_checkpoint_dir() -> None:
    assert build_checkpoint_dir("/base", "d12", 128, "dist/x") == (
        "/base/conditional_encdec_checkpoints/d12/s128/dist/x"
    )


def test_build_sft_checkpoint_dir_is_conditioned_only() -> None:
    assert build_sft_checkpoint_dir("/base/d12", "safety") == (
        "/base/d12/safety_sft_conditioned_checkpoints"
    )


@pytest.mark.parametrize("dataset", ["standard", "capabilities", "safety"])
def test_build_feature_dir_for_phase_targets_no_bos_leaf(dataset: str) -> None:
    assert DatasetPhase(f"{dataset}_sft").value == f"{dataset}_sft"
    path = build_feature_dir_for_phase(
        feature_root="/features",
        segment_size=128,
        layer=8,
        activation_type="residual",
        pooling="max_pool",
        sae_width="16k",
        model="gemma-2-2b",
        dataset_phase=f"{dataset}_sft",
    )
    assert path.endswith(f"/{dataset}_sft/no_bos")


class TestRunSft:
    def test_single_gpu_command_requires_feature_dir(self) -> None:
        result = MagicMock(returncode=0)
        with patch("scripts.train.run_encdec_sft_pipeline.subprocess") as subproc:
            subproc.run.return_value = result
            run_sft(
                dataset="safety",
                model_tag="d12",
                sft_overrides=["max_seq_len=1024"],
                feature_dir="/features/safety_sft",
            )

        cmd = subproc.run.call_args.args[0]
        assert cmd[:3] == [sys.executable, "-m", "scripts.train.conditional_encdec_sft"]
        assert "dataset=safety" in cmd
        assert "model_tag=d12" in cmd
        assert "feature_dir=/features/safety_sft" in cmd
        assert "max_seq_len=1024" in cmd
        assert not any("label_features" in part for part in cmd)

    def test_multi_gpu_uses_torchrun(self) -> None:
        result = MagicMock(returncode=0)
        with patch("scripts.train.run_encdec_sft_pipeline.subprocess") as subproc:
            subproc.run.return_value = result
            run_sft(
                dataset="capabilities",
                model_tag="d12",
                sft_overrides=[],
                base_checkpoint_dir="/parent",
                feature_dir="/features/capabilities_sft",
                num_gpus=4,
            )

        cmd = subproc.run.call_args.args[0]
        assert cmd[:4] == ["torchrun", "--standalone", "--nproc_per_node=4", "-m"]
        assert "base_checkpoint_dir=/parent" in cmd
        assert "feature_dir=/features/capabilities_sft" in cmd

    def test_failure_raises(self) -> None:
        result = MagicMock(returncode=3)
        with patch("scripts.train.run_encdec_sft_pipeline.subprocess") as subproc:
            subproc.run.return_value = result
            with pytest.raises(RuntimeError, match="safety SFT failed"):
                run_sft(
                    dataset="safety",
                    model_tag="d12",
                    sft_overrides=[],
                    feature_dir="/features/safety_sft",
                )


def test_run_pipeline_chains_conditioned_phases() -> None:
    with (
        patch("scripts.train.run_encdec_sft_pipeline.get_base_dir", return_value="/base"),
        patch(
            "scripts.train.run_encdec_sft_pipeline.build_feature_dir_for_phase",
            side_effect=lambda *args, **kwargs: f"/features/{args[-1]}",
        ) as build_feature_dir,
        patch("scripts.train.run_encdec_sft_pipeline.ensure_feature_statistics") as stats,
        patch("scripts.train.run_encdec_sft_pipeline.verify_feature_dir"),
        patch("scripts.train.run_encdec_sft_pipeline.run_sft") as sft,
    ):
        result = run_pipeline(
            model_tag="d12",
            phase_order="safety,capabilities",
            sft_overrides=["device_batch_size=4"],
        )

    assert result == {
        "safety_sft": (
            "/base/conditional_encdec_checkpoints/d12/s128/"
            "dist_geometric/mean_10.0/std_10.0/"
            "safety_sft_conditioned_checkpoints"
        ),
        "capabilities_sft": (
            "/base/conditional_encdec_checkpoints/d12/s128/"
            "dist_geometric/mean_10.0/std_10.0/"
            "safety_sft_conditioned_checkpoints/"
            "capabilities_sft_conditioned_checkpoints"
        ),
    }
    assert [call.args[1] for call in stats.call_args_list] == [
        "safety_sft",
        "capabilities_sft",
    ]
    assert build_feature_dir.call_count == 2
    first_call, second_call = sft.call_args_list
    assert first_call.kwargs["base_checkpoint_dir"] is None
    assert first_call.kwargs["feature_dir"] == "/features/safety_sft"
    assert second_call.kwargs["base_checkpoint_dir"] == result["safety_sft"]
    assert second_call.kwargs["feature_dir"] == "/features/capabilities_sft"


def test_run_pipeline_rejects_clustered_chain_at_startup() -> None:
    with (
        patch("scripts.train.run_encdec_sft_pipeline.get_base_dir", return_value="/base"),
        pytest.raises(
            FeatureClustersError,
            match=(
                "clustered chained SFT is not supported by the pipeline driver; "
                "run phases individually with explicit base_checkpoint_dir"
            ),
        ),
    ):
        run_pipeline(
            model_tag="d12",
            sft_overrides=["feature_clusters={path:/tmp/clusters.yaml}"],
        )


@pytest.mark.parametrize("num_gpus", [0, -1])
def test_run_pipeline_rejects_non_positive_num_gpus(num_gpus: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="num_gpus"):
        run_pipeline(model_tag="d12", num_gpus=num_gpus)
