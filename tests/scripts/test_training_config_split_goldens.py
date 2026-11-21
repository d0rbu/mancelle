"""Pre-split composed-config goldens for issue #1281."""

from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "training"
GOLDEN_DIR = Path(__file__).parent / "golden" / "training"
ORACLE_COMMIT = "5562455fbfe7fbbdaef0d44083b85062039010bf"


def _load_golden(config_name: str) -> dict:
    path = GOLDEN_DIR / f"{config_name}_composed.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _compose(config_name: str) -> dict:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    return data


def test_base_train_composition_matches_pre_split_golden() -> None:
    golden = _load_golden("base_train")
    assert golden["oracle_commit"] == ORACLE_COMMIT
    assert golden["config_name"] == "base_train"
    assert _compose("base_train") == golden["config"]


def test_conditional_encdec_train_composition_matches_pre_split_golden() -> None:
    golden = _load_golden("conditional_encdec_train")
    assert golden["oracle_commit"] == ORACLE_COMMIT
    assert golden["config_name"] == "conditional_encdec_train"
    composed = _compose("conditional_encdec_train")
    assert composed.pop("conditioning_cooldown") == {
        "enabled": False,
        "start_fraction": 1.0,
        "end_fraction": 0.5,
        "start_step": None,
        "end_step": None,
        "cooldown_tokens": 20_000_000_000,
        "granularity": "segment",
    }
    assert composed == golden["config"]
