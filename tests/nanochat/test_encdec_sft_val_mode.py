"""Conditioned-only SFT validation surface tests."""

from __future__ import annotations

from pathlib import Path

import torch as t
from hydra import compose, initialize_config_dir

from scripts.train.conditional_encdec_sft import (
    SFTState,
    _build_sft_val_loader,
    _SFTValidationLoader,
)
from steering.sft.loss_mask import LossMaskMode
from steering.sft_schema import SFTSplit

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "training"


def test_sft_config_has_no_validation_mode_knob() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="conditional_encdec_sft")

    assert "label_features" not in cfg
    assert "val_label_features" not in cfg
    assert cfg.val_feature_dir is None
    assert cfg.val_sft_parquet_dir is None


def test_val_loader_always_requests_conditioned_generator(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_make_sft_data_generator(**kwargs):
        captured.update(kwargs)
        yield ("batch",)

    monkeypatch.setattr(
        "scripts.train.conditional_encdec_sft.make_sft_data_generator",
        fake_make_sft_data_generator,
    )

    loader = _build_sft_val_loader(
        val_task=object(),
        tokenizer=object(),
        B=2,
        T=16,
        device="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        feature_dir="/val/features",
        expected_feature_dir="/val/features",
        parquet_dir="/val/parquets",
        threshold_cfg=object(),
        boundaries=object(),
        precomputed_thresholds=None,
        sampling_bundle=None,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=8,
        n_sae_features=8,
        fp_percentile_table=t.empty(0),
        alignment_probe_samples=0,
    )

    # The factory's return type is part of its contract: ``evaluate_encdec_bpb``
    # reads the covered-document bound off it, and a refactor that handed back
    # a bare generator again would silently restore wrap-around validation
    # (review finding 10, PR #1311).
    assert isinstance(loader, _SFTValidationLoader)
    assert isinstance(loader.state, SFTState)

    assert next(loader) == ("batch",)
    assert captured["single_pass"] is True
    assert "label_features" not in captured
    assert captured["feature_dir"] == "/val/features"
    assert captured["parquet_dir"] == "/val/parquets"
    assert captured["expected_split"] is SFTSplit.VAL
    assert captured["loss_mask_mode"] is LossMaskMode.FULL_SEQUENCE
    assert captured["n_sae_features"] == 8
    assert captured["fp_percentile_table"] is not None
