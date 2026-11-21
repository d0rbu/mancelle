"""T-S1: Hydra and shipped-config-family contracts for self-conditioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from errors import SelfCondTrainingError
from scripts.train.conditional_encdec_selfcond import resolve_base_checkpoint_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")


def test_required_model_tag_and_pinned_defaults(hydra_compose) -> None:
    unresolved = hydra_compose(CONFIG_DIR, "conditional_encdec_selfcond")
    assert unresolved.model_tag is None
    with pytest.raises(SelfCondTrainingError, match="model_tag is required.*base_checkpoint_dir"):
        resolve_base_checkpoint_dir(unresolved, "/tmp/base")

    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_selfcond",
        ["model_tag=tiny-base"],
    )
    assert cfg.model_tag == "tiny-base"
    assert cfg.model_step is None
    assert cfg.selfcond.train_scope == "head_only"
    assert cfg.selfcond.ste_form == "dense"
    assert cfg.selfcond.pool_mode is None
    assert cfg.selfcond.positive_counts is None
    assert cfg.selfcond.negative_counts is None
    assert cfg.compile is False
    assert cfg.num_iterations == 500
    # Regression pin: the in-loop evals v1 does not wire MUST resolve to 0 on the
    # COMPOSED config. base_train defaults them to 2000; if the shipped config
    # omitted these, validate_selfcond_startup would raise on the documented
    # quickstart invocation. Assert on cfg (not a hand-built runtime dict) so the
    # crash-on-default cannot regress silently.
    assert cfg.core_metric_every == 0
    assert cfg.sample_every == 0
    assert cfg.steering_eval_every == 0
    assert cfg.train_prefetch_batches is None  # null = grad_accum_steps parity
    assert cfg.profile_timing is False
    assert cfg.profile_trace is None
    assert cfg.profile_trace_steps == 1
    assert cfg.profile_output_dir == "out/profiles/encdec_selfcond"
    assert set(cfg.selfcond) == {
        "train_scope",
        "ste_form",
        "pool_mode",
        "positive_counts",
        "negative_counts",
        "data_source",
        "sft_dataset",
        "sft_feature_dir",
        "sft_parquet_dir",
        "val_sft_feature_dir",
        "val_sft_parquet_dir",
    }
    assert cfg.selfcond.data_source == "pretraining"
    assert cfg.selfcond.sft_dataset is None
    # kills: restoring the removed cadence knob to the config -- the
    # generated-assistant cadence is measured from the TRAIN artifacts and
    # a declared value is refused by the trainer.
    assert "turn_self_cond_segment_size_rustbpe" not in cfg.selfcond

    explicit = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_selfcond",
        ["base_checkpoint_dir=/tmp/checkpoint"],
    )
    assert resolve_base_checkpoint_dir(explicit, "/ignored") == "/tmp/checkpoint"


def test_selfcond_override_roundtrip(hydra_compose) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_selfcond",
        [
            "model_tag=tiny-base",
            "model_step=17",
            "selfcond.train_scope=full",
            "selfcond.ste_form=selected",
            "selfcond.pool_mode=last_token",
            "selfcond.positive_counts={distribution:geometric,mean:3.0,std:3.0,min_k:1,max_k:7}",
            "selfcond.negative_counts={distribution:normal,mean:2.0,std:1.0,min_k:0,max_k:5}",
        ],
    )
    assert cfg.model_step == 17
    assert cfg.selfcond.train_scope == "full"
    assert cfg.selfcond.ste_form == "selected"
    assert cfg.selfcond.pool_mode == "last_token"
    assert dict(cfg.selfcond.positive_counts) == {
        "distribution": "geometric",
        "mean": 3.0,
        "std": 3.0,
        "min_k": 1,
        "max_k": 7,
    }
    assert dict(cfg.selfcond.negative_counts) == {
        "distribution": "normal",
        "mean": 2.0,
        "std": 1.0,
        "min_k": 0,
        "max_k": 5,
    }


def test_config_is_checkpoint_shaped_not_pretraining_shaped(hydra_compose) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_selfcond",
        ["model_tag=tiny-base"],
    )
    assert "encoder" not in cfg
    assert "cross_attention" not in cfg
    assert "feature_delta" not in cfg
    assert "sae" not in cfg
    assert "negative_selection" not in cfg
