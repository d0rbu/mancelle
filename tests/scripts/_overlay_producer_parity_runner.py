"""Isolated-process runner for overlay producer byte-identity tests."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf

from scripts.preprocess.precompute_percentiles import run as run_percentiles
from scripts.preprocess.precompute_token_boundaries import run as run_token_boundaries

_FIXED_GMTIME = time.struct_time((2026, 7, 20, 12, 34, 56, 0, 202, -1))


class FrozenCharTokenizer:
    """Deterministic scalar/batch character tokenizer with a decoded BOS."""

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 0x110001

    def _encode_one(self, text: str, prepend: int | None) -> list[int]:
        result = [ord(char) + 1 for char in text]
        if prepend is not None:
            result.insert(0, int(prepend))
        return result

    def encode(
        self,
        text: str | list[str],
        prepend: int | None = None,
        num_threads: int | None = None,
    ) -> list[int] | list[list[int]]:
        del num_threads
        if isinstance(text, list):
            return [self._encode_one(item, prepend) for item in text]
        return self._encode_one(text, prepend)

    def decode(self, token_ids: list[int]) -> str:
        return "".join("<bos>" if token_id == 0 else chr(token_id - 1) for token_id in token_ids)


def _percentile_cfg(feature_dir: Path, *, resume: bool) -> DictConfig:
    return OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "segment_size": 8,
            "device": "cpu",
            "resume": resume,
            "force": False,
            "dry_run": False,
            "zstd_level": 12,
        }
    )


def _token_cfg(feature_dir: Path, data_dir: Path, *, resume: bool) -> DictConfig:
    return OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "segment_size": 8,
            "data_dir": str(data_dir),
            "workers": 1,
            "resume": resume,
            "force": False,
            "dry_run": False,
            "profile": False,
            "shard_start": 0,
            "shard_end": -1,
            "zstd_level": 12,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    tokenizer = FrozenCharTokenizer()
    with (
        patch("steering.overlay_io.time.gmtime", return_value=_FIXED_GMTIME),
        patch(
            "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
            return_value=tokenizer,
        ),
    ):
        percentile_status = run_percentiles(
            _percentile_cfg(args.feature_dir, resume=args.resume)
        )
        if percentile_status != 0:
            raise SystemExit(f"percentile producer returned {percentile_status}")
        token_status = run_token_boundaries(
            _token_cfg(args.feature_dir, args.data_dir, resume=args.resume)
        )
        if token_status != 0:
            raise SystemExit(f"token producer returned {token_status}")


if __name__ == "__main__":
    main()
