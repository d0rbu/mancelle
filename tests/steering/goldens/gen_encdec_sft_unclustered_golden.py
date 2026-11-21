"""Capture the pre-#1279 unclustered SFT identity surface.

Run this module only from the pinned base commit recorded below::

    uv run python -m tests.steering.goldens.gen_encdec_sft_unclustered_golden

The production change must consume the committed JSON; it must not regenerate
the oracle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train.conditional_encdec_sft import (
    _build_sft_checkpoint_dir,
    _checkpoint_user_config_with_fp,
)
from steering._common.strict_json import string_keyed_mapping
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _make_gen,
    build_synthetic_features,
    patch_load_sft_features,
)

ORACLE_COMMIT = "66cdaecfa0c37dc3690e7e5e8b21ddd179439de2"
GOLDEN_PATH = Path(__file__).with_name("encdec_sft_unclustered_golden.json")
_TENSOR_FIELDS = {
    "targets": 1,
    "enc_fids": 2,
    "enc_mags": 3,
    "enc_mask": 4,
    "seg_map": 5,
}


def _tensor_record(value: t.Tensor) -> dict[str, Any]:
    cpu = value.detach().cpu().contiguous()
    raw = cpu.view(t.uint8).numpy().tobytes()
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "values": cpu.tolist(),
    }


def build_capture() -> dict[str, Any]:
    """Return deterministic tensors plus path/config identity anchors."""
    t.manual_seed(1279)
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=6, user_chars=3, asst_chars=7, prefix="g")
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        features_per_segment=3,
    )
    with patch_load_sft_features(features):
        batch = next(
            _make_gen(
                task,
                tokenizer,
                B=2,
                T=48,
                alignment_probe_samples=0,
            )
        )

    config_dir = str(Path("configs/training").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="conditional_encdec_sft",
            overrides=["model_tag=golden", "feature_dir=/golden/features"],
        )
    raw = string_keyed_mapping(OmegaConf.to_container(cfg, resolve=True))
    assert raw is not None
    checkpoint_user_config = _checkpoint_user_config_with_fp(raw, None)

    return {
        "schema_version": 1,
        "oracle_commit": ORACLE_COMMIT,
        "torch_seed": 1279,
        "tensors": {name: _tensor_record(batch[index]) for name, index in _TENSOR_FIELDS.items()},
        "checkpoint_path": _build_sft_checkpoint_dir("/checkpoints/d26/s128/t10", "safety"),
        "checkpoint_user_config_keys": sorted(checkpoint_user_config),
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(build_capture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(GOLDEN_PATH)


if __name__ == "__main__":
    main()
