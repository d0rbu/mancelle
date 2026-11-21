"""The conditioned-SFT trainer must take the SAE block from the base checkpoint's authoritative record."""

from __future__ import annotations

import pytest

from scripts.train.conditional_encdec_sft import base_sae_config
from steering.conditional.encoder_decoder.sft_grpo_online_sae import ONLINE_SAE_LINEAGE_KEY

_PRODUCTION_PROBE = {
    "release": "gemmascope_2",
    "model": "gemma-3-4b-pt",
    "activation_type": "residual",
    "layer": 22,
    "width": "65k",
    "l0": "medium",
}
_STALE_DEFAULT = {
    "release": "gemmascope",
    "model": "gemma-2-2b",
    "activation_type": "residual",
    "layer": 8,
    "width": "16k",
}


def _chained_base_meta() -> dict[str, object]:
    # A conditioned-SFT checkpoint used as the base of a second SFT stage: its
    # user_config.sae is the first stage's Hydra default, its lineage is the truth.
    return {
        "user_config": {"sae": dict(_STALE_DEFAULT)},
        ONLINE_SAE_LINEAGE_KEY: {**_PRODUCTION_PROBE, "schema_version": 1, "feature_dir": "/x"},
    }


def test_chained_base_uses_lineage_over_stale_user_config() -> None:
    meta = _chained_base_meta()
    resolved = base_sae_config(meta, meta["user_config"], sft_user_config={"sae": _STALE_DEFAULT})
    assert resolved == _PRODUCTION_PROBE


def test_pretrain_base_uses_its_user_config_sae() -> None:
    meta = {"user_config": {"sae": dict(_PRODUCTION_PROBE)}}
    resolved = base_sae_config(meta, meta["user_config"], sft_user_config={"sae": _STALE_DEFAULT})
    assert resolved == _PRODUCTION_PROBE


def test_base_without_sae_falls_through_to_run_config() -> None:
    meta = {"user_config": {}}
    resolved = base_sae_config(meta, meta["user_config"], sft_user_config={"sae": _PRODUCTION_PROBE})
    assert resolved == _PRODUCTION_PROBE


def test_lineage_missing_probe_field_fails_loud() -> None:
    meta = _chained_base_meta()
    del meta[ONLINE_SAE_LINEAGE_KEY]["width"]
    with pytest.raises(ValueError, match=r"lacks probe fields \['width'\]"):
        base_sae_config(meta, meta["user_config"], sft_user_config={})
