"""Capture the pre-#1286 full-sequence targets for both SFT trainers.

Oracle provenance: the committed JSON first appears at capture commit
``ccbf20845``, which recorded the relocation-only behavior before masking.
This generator has since evolved and is not re-runnable at that commit.
Regeneration requires re-capturing the intended pre-feature behavior and
updating the fixed SHA-256 in ``test_sft_loss_mask_goldens.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch as t

from steering.sft.iteration import sft_data_generator_bos_bestfit_core
from steering.sft.loss_mask import LossMaskMode
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _make_gen,
    build_synthetic_features,
    patch_load_sft_features,
)

ORACLE_COMMIT = "d14ba7aa443f4b26d127ff6505180bf20a31971b"
GOLDEN_PATH = Path(__file__).with_name("sft_loss_mask_off_golden.json")
TORCH_SEED = 1286


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
    """Return deterministic decoder-only and encoder-decoder text batches."""
    t.manual_seed(TORCH_SEED)
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=6, user_chars=3, asst_chars=7, prefix="m")

    decoder_batch = next(
        sft_data_generator_bos_bestfit_core(
            dataset=task,
            tokenizer=tokenizer,
            max_seq_len=48,
            device_batch_size=2,
            ddp_rank=0,
            ddp_world_size=1,
            device=t.device("cpu"),
            pin_memory=False,
            loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        )
    )

    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        features_per_segment=3,
    )
    with patch_load_sft_features(features):
        encdec_batch = next(
            _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=2,
                T=48,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            )
        )

    return {
        "schema_version": 1,
        "oracle_commit": ORACLE_COMMIT,
        "torch_seed": TORCH_SEED,
        "decoder_only": {
            "inputs": _tensor_record(decoder_batch[0]),
            "targets": _tensor_record(decoder_batch[1]),
        },
        "encoder_decoder": {
            "inputs": _tensor_record(encdec_batch[0]),
            "targets": _tensor_record(encdec_batch[1]),
        },
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(build_capture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(GOLDEN_PATH)


if __name__ == "__main__":
    main()
