"""Test-only builder for fully committed AxBench schema-v2 seed pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanochat.atomic_io import atomic_json_dump
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
    compute_extraction_fingerprint,
    compute_seed_fingerprint,
    save_seed_pool,
    seed_config_fingerprint,
    seed_config_payload,
    sha256_file,
)
from steering.eval.axbench_base.types import SeedRow
from steering.segmentation import compute_tokenizer_checksum


def write_committed_seed_pair(
    cfg: AxBenchBaseConfig,
    *,
    tokenizer: Any,
    train_rows: list[SeedRow],
    eval_rows: list[SeedRow],
) -> tuple[Path, Path]:
    """Write a coherent train/eval/manifest triple for runtime tests."""

    assert cfg.run_tag is not None
    if not train_rows or not eval_rows:
        raise ValueError("test seed pair requires non-empty train and eval rows")
    train_path = paths.AxBenchBasePaths.from_config(cfg).seeds_train()
    eval_path = paths.AxBenchBasePaths.from_config(cfg).seeds_eval()
    config_json = canonical_json(seed_config_payload(cfg))
    config_fingerprint = seed_config_fingerprint(cfg)
    nano_checksum = compute_tokenizer_checksum(tokenizer)
    source_hashes = {"/test/source.parquet": "3" * 64}
    parameter_hashes = {
        "forward_model_sha256": "4" * 64,
        "sae_sha256": "5" * 64,
    }
    runtime_identity = {"backend": "test", "device_type": "cpu"}
    extraction_settings = {"pooling": "max_pool", "segment_size": 128}
    candidate_digest = "6" * 64
    extraction_fingerprint = compute_extraction_fingerprint(
        extraction_settings=extraction_settings,
        runtime_identity=runtime_identity,
        parameter_hashes=parameter_hashes,
    )
    seed_fingerprint = compute_seed_fingerprint(
        candidate_digest=candidate_digest,
        config_fingerprint=config_fingerprint,
        extraction_fingerprint=extraction_fingerprint,
        nanochat_tokenizer_checksum=nano_checksum,
        source_parquet_hashes=source_hashes,
    )
    common = {
        "schema_version": str(SEED_SCHEMA_VERSION),
        "seed_fingerprint": seed_fingerprint,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "nanochat_tokenizer_checksum": nano_checksum,
        "extraction_tokenizer_checksum": "2" * 64,
        "config_fingerprint": config_fingerprint,
        "seed_config_json": config_json,
        "source_parquet_hashes_json": canonical_json(source_hashes),
        "parameter_hashes_json": canonical_json(parameter_hashes),
        "runtime_identity_json": canonical_json(runtime_identity),
        "extraction_settings_json": canonical_json(extraction_settings),
    }
    save_seed_pool(
        train_path,
        train_rows,
        metadata={**common, "split": "train"},
    )
    save_seed_pool(
        eval_path,
        eval_rows,
        metadata={**common, "split": "eval"},
    )
    atomic_json_dump(
        {
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_fingerprint": seed_fingerprint,
            "token_semantics": SEED_TOKEN_SEMANTICS,
            "candidate_digest": candidate_digest,
            "candidate_count": len(train_rows) + len(eval_rows),
            "skip_counts": {},
            "source_parquet_hashes": source_hashes,
            "config_fingerprint": config_fingerprint,
            "extraction_fingerprint": extraction_fingerprint,
            "nanochat_tokenizer_checksum": nano_checksum,
            "extraction_tokenizer_checksum": "2" * 64,
            "parameter_hashes": parameter_hashes,
            "runtime_identity": runtime_identity,
            "extraction_settings": extraction_settings,
            "files": {
                train_path.name: {
                    "sha256": sha256_file(train_path),
                    "row_count": len(train_rows),
                    "split": "train",
                },
                eval_path.name: {
                    "sha256": sha256_file(eval_path),
                    "row_count": len(eval_rows),
                    "split": "eval",
                },
            },
        },
        paths.AxBenchBasePaths.from_config(cfg).seed_pool_manifest(),
    )
    return train_path, eval_path
