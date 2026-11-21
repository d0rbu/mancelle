"""Capture the pre-#1276 full-width SFT feature-prediction target.

Run this module only from the pinned #1279 tip recorded below::

    uv run python -m tests.steering.goldens.gen_encdec_sft_compact_fp_targets_legacy_golden

The compact-target implementation must consume the committed JSON; it must
not regenerate the oracle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch as t

from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.refined_types import PositiveInt
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    build_synthetic_features,
    patch_load_sft_features,
)
from tests.steering.test_encdec_sft_feature_prediction import _make_gen_with_fp

ORACLE_COMMIT = "94c19d72b305342e3e7fd3b3175dde0500933d4e"
GOLDEN_PATH = Path(__file__).with_name("encdec_sft_compact_fp_targets_legacy_golden.json")


def capture_legacy_fp_targets() -> t.Tensor:
    """Return the deterministic full-width target built by the #1279 tip."""
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=4, user_chars=3, asst_chars=8, prefix="g")
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        features_per_segment=3,
    )
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(200),
        feature_mask=None,
        feature_clusters=None,
        feature_vocab=None,
    )
    with patch_load_sft_features(features):
        batch = next(
            _make_gen_with_fp(
                task,
                tokenizer,
                B=1,
                T=16,
                return_seg_doc_id=True,
                fp_target_layout=layout,
            )
        )
    fp_targets = batch[-2]
    assert fp_targets is not None
    return fp_targets.detach().cpu().contiguous()


def _sparse_tensor_record(value: t.Tensor) -> dict[str, Any]:
    nonzero = value != 0
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.view(t.uint8).numpy().tobytes()).hexdigest(),
        "nonzero_indices": nonzero.nonzero(as_tuple=False).tolist(),
        "nonzero_values": value[nonzero].tolist(),
    }


def build_capture() -> dict[str, Any]:
    """Return the pinned tensor identity without expanding 594 zero cells."""
    return {
        "schema_version": 1,
        "oracle_commit": ORACLE_COMMIT,
        "fp_targets": _sparse_tensor_record(capture_legacy_fp_targets()),
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(build_capture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(GOLDEN_PATH)


if __name__ == "__main__":
    main()
