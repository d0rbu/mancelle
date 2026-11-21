"""Capture the pre-#1285 aligned conditioned-SFT batch.

Oracle provenance: ``f2cb2f78e7f24d34ae541a517902fc9a65f9431e`` is the
tip of #1286 and the base of #1285.  Run this module only when deliberately
re-capturing the pre-alignment-hardening behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch as t

from scripts.preprocess.convert_sft_to_parquet import (
    render_conversation_to_full_nanochat_text,
)
from steering.sft.loss_mask import LossMaskMode
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft_schema import SFTSplit
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _make_gen,
    build_synthetic_features,
    patch_load_sft_features,
)

ORACLE_COMMIT = "f2cb2f78e7f24d34ae541a517902fc9a65f9431e"
GOLDEN_PATH = Path(__file__).with_name("sft_conditioning_alignment_golden.json")
TORCH_SEED = 1285


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
    """Return one deterministic, correctly aligned conditioned-SFT batch."""
    t.manual_seed(TORCH_SEED)
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=6, user_chars=4, asst_chars=8, prefix="aligned")
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        features_per_segment=3,
    )
    fingerprint = compute_mixture_fingerprint(
        task,
        mixture_name="capabilities",
        split=SFTSplit.TRAIN,
    )
    texts: dict[int, str] = {}
    for index in range(len(task)):
        text, skip_reason = render_conversation_to_full_nanochat_text(
            tokenizer,
            task[index],
        )
        assert skip_reason is None
        assert text is not None
        texts[index] = text
    with patch_load_sft_features(
        features,
        enforce_fingerprints=True,
        split=SFTSplit.TRAIN,
        mixture="capabilities",
        parquet_fingerprint=fingerprint,
        feature_fingerprint=fingerprint,
        texts=texts,
    ):
        batch = next(
            _make_gen(
                task,
                tokenizer,
                B=2,
                T=64,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
                alignment_probe_samples=8,
            )
        )

    return {
        "schema_version": 1,
        "oracle_commit": ORACLE_COMMIT,
        "torch_seed": TORCH_SEED,
        "batch": [None if value is None else _tensor_record(value) for value in batch],
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(build_capture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(GOLDEN_PATH)


if __name__ == "__main__":
    main()
