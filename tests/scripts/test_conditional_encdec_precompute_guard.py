"""Entrypoint regression for producer-only precomputed-batch settings."""

from __future__ import annotations

from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from scripts.train.conditional_encdec_train import train_one_config


def test_training_entrypoint_rejects_non_null_producer_only_packed_row_dir() -> None:
    with pytest.raises(
        ValueError,
        match="only consumed by scripts.preprocess.precompute_selected_features",
    ):
        train_one_config(
            OmegaConf.create({"packed_row_dir": "/tmp/packed"}),
            cast(Any, None),
            None,
        )
