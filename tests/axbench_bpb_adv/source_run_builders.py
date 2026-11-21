"""Shared Stage 0-3 source-run builders for axbench-bpb-adv tests.

The source-validation test modules (``test_config_source``,
``test_source_contract_artifact_metrics``, ``test_source_validation_fail_loud``)
each need a valid Stage 0-3 source run on disk. The common scaffolding —
``concepts.json`` + dual ``synth_training_{positive,negative}.parquet`` + a committed
seed-v2 pair + the two concept-feature stubs + a ``baselines/`` tree — lives here so
it stays in one place
as the upstream contract (``source_paths.validate_stage_0_3_artifacts`` /
``source_validation._required_artifacts``) evolves. Callers supply their own synth
rows and an optional ``baselines_builder``. The default is the dedicated
directionless fixture: a prompting artifact and no residual directions. Tests
that consume directions must supply realistic stamped safetensors.

This is a plain helper module, not ``conftest.py`` (which by convention holds only
fixtures), imported directly the way the functional fixtures already are.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from safetensors.torch import save_file

from nanochat.atomic_io import atomic_json_dump
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base.direction_artifacts import expected_artifact_kind
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
    compute_extraction_fingerprint,
    compute_seed_fingerprint,
    save_seed_pool,
    sha256_bytes,
    sha256_file,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import SeedRow
from steering.segmentation import compute_tokenizer_checksum

SYNTH_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("content_class", pa.string()),
        ("prefix_text", pa.string()),
        ("output_text", pa.string()),
        ("qc_accepted", pa.bool_()),
        ("prefix_source_shard", pa.int32()),
        ("prefix_source_doc", pa.int32()),
    ]
)

# The two stub Stage 0-3 artifacts the validator requires beyond concepts.json,
# synth data, and the real seed-v2 fixture
# and the two synth parquets; each is written as a 1-byte ``b"x"`` stub in tests.
STUB_FILE_ARTIFACTS = (
    "concept_features.safetensors",
    "concept_features_neg.safetensors",
)


class SourceSeedTokenizer:
    """Reversible tokenizer shared by source-contract builders and fixtures."""

    def get_bos_token_id(self) -> int:
        return 99

    def encode(self, text: str) -> list[int]:
        return [1_000 + ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token - 1_000) for token in tokens)


def write_seed_contract(root: Path) -> None:
    """Write a complete schema-v2 seed pair and commit manifest under ``root``."""

    tokenizer = SourceSeedTokenizer()
    row = SeedRow(
        shard_idx=0,
        doc_idx=0,
        source_nanochat_token_count=3,
        prefix_source_token_start=0,
        prefix_source_token_end=2,
        prefix_text="ab",
        prefix_content_nanochat_tokens=tokenizer.encode("ab"),
        sae_feature_ids=[1],
        sae_feature_mags=[1.0],
        continuation_text="c",
        continuation_content_nanochat_tokens=tokenizer.encode("c"),
    )
    config_json = canonical_json(
        {
            "feature_directory": {"width": "16k"},
            "seed_continuation_nanochat_tokens": 1,
            "seed_prefix_max_nanochat_tokens": 2,
        }
    )
    config_fingerprint = sha256_bytes(config_json.encode("utf-8"))
    source_hashes = {"/test/source.parquet": "3" * 64}
    parameter_hashes = {"model": "4" * 64, "sae": "5" * 64}
    runtime_identity = {"backend": "test"}
    extraction_settings = {"pooling": "max_pool"}
    nanochat_tokenizer_checksum = compute_tokenizer_checksum(tokenizer)
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
        nanochat_tokenizer_checksum=nanochat_tokenizer_checksum,
        source_parquet_hashes=source_hashes,
    )
    common = {
        "schema_version": str(SEED_SCHEMA_VERSION),
        "seed_fingerprint": seed_fingerprint,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "nanochat_tokenizer_checksum": nanochat_tokenizer_checksum,
        "extraction_tokenizer_checksum": "2" * 64,
        "config_fingerprint": config_fingerprint,
        "seed_config_json": config_json,
        "source_parquet_hashes_json": canonical_json(source_hashes),
        "parameter_hashes_json": canonical_json(parameter_hashes),
        "runtime_identity_json": canonical_json(runtime_identity),
        "extraction_settings_json": canonical_json(extraction_settings),
    }
    train_path = root / "seeds_train.safetensors"
    eval_path = root / "seeds_eval.safetensors"
    save_seed_pool(train_path, [row], metadata={**common, "split": "train"})
    save_seed_pool(eval_path, [row], metadata={**common, "split": "eval"})
    atomic_json_dump(
        {
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_fingerprint": seed_fingerprint,
            "token_semantics": SEED_TOKEN_SEMANTICS,
            "candidate_digest": candidate_digest,
            "candidate_count": 2,
            "skip_counts": {},
            "source_parquet_hashes": source_hashes,
            "config_fingerprint": config_fingerprint,
            "extraction_fingerprint": extraction_fingerprint,
            "nanochat_tokenizer_checksum": nanochat_tokenizer_checksum,
            "extraction_tokenizer_checksum": "2" * 64,
            "parameter_hashes": parameter_hashes,
            "runtime_identity": runtime_identity,
            "extraction_settings": extraction_settings,
            "files": {
                train_path.name: {
                    "sha256": sha256_file(train_path),
                    "row_count": 1,
                    "split": "train",
                },
                eval_path.name: {
                    "sha256": sha256_file(eval_path),
                    "row_count": 1,
                    "split": "eval",
                },
            },
        },
        root / "seed_pool_manifest.json",
    )


def synth_row(
    prefix: str,
    *,
    content_class: str,
    concept_id: int = 1,
    qc_accepted: bool = True,
    output_text: str = "de",
) -> dict:
    """One synth-training row matching ``SYNTH_SCHEMA``."""
    return {
        "concept_id": concept_id,
        "content_class": content_class,
        "prefix_text": prefix,
        "output_text": output_text,
        "qc_accepted": qc_accepted,
        "prefix_source_shard": 0,
        "prefix_source_doc": 0,
    }


def both_class_rows(prefix: str, *, concept_id: int = 1) -> list[dict]:
    """A ``concept_rich`` + ``concept_avoiding`` pair for ``prefix``."""
    return [
        synth_row(prefix, content_class=cls, concept_id=concept_id)
        for cls in ("concept_rich", "concept_avoiding")
    ]


def one_sided_rows(
    prefix: str, *, content_class: str, concept_id: int = 1
) -> list[dict]:
    """A single-class (incomplete) prefix."""
    return [synth_row(prefix, content_class=content_class, concept_id=concept_id)]


def write_directionless_baselines(baselines: Path) -> None:
    """Dedicated no-direction source fixture using a real prompting coordinate."""

    prompting = baselines / "prompting"
    prompting.mkdir(parents=True)
    (prompting / "positive_descriptions.json").write_text("{}", encoding="utf-8")


def write_scalar_direction_baselines(
    baselines: Path,
    *,
    concept_ids: tuple[int, ...] = (1,),
    layer: int = 6,
    width: int = 8,
    families: tuple[str, ...] = ("diffmean", "linear_probe"),
    targets: tuple[str, ...] = ("decoder_only", "encdec"),
    modes: tuple[str, ...] = ("positive", "negative"),
) -> None:
    """Write canonical unclustered directions for consumer-facing fixtures."""

    for family in families:
        for target in targets:
            for mode in modes:
                for concept_id in concept_ids:
                    path = (
                        baselines
                        / family
                        / target
                        / mode
                        / f"concept_{concept_id:04d}.safetensors"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    direction = t.arange(width, dtype=t.float32) + float(concept_id + 1)
                    save_file(
                        {"direction": direction},
                        str(path),
                        metadata={
                            "artifact_kind": expected_artifact_kind(family),
                            "concept_id": str(concept_id),
                            "layer": str(layer),
                            "mode": mode,
                            "target_model": target,
                        },
                    )


def write_stage_0_3_source_run(
    root: Path,
    *,
    positive_rows: list[dict],
    negative_rows: list[dict],
    baselines_builder: Callable[[Path], None] = write_directionless_baselines,
    concepts: list[dict] | None = None,
) -> None:
    """Write a valid Stage 0-3 source run at ``root``.

    Creates immutable Stage 0c ``concepts.json`` and the final
    ``concepts_stage1c.json`` authority (a single concept_id=1 by default), the
    two synth parquets from the supplied rows, a committed seed-v2 pair, the
    concept feature stubs, and a ``baselines/`` directory via
    ``baselines_builder``.
    """
    root.mkdir(parents=True)
    concepts_payload = (
        concepts
        if concepts is not None
        else [{"concept_id": 1, "description": "concept"}]
    )
    concepts_json = json.dumps(concepts_payload)
    (root / "concepts.json").write_text(concepts_json, encoding="utf-8")
    (root / "concepts_stage1c.json").write_text(
        concepts_json,
        encoding="utf-8",
    )
    positive_schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            SYNTH_SCHEMA,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(
        pa.Table.from_pylist(positive_rows, schema=positive_schema),
        root / "synth_training_positive.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            negative_rows,
            schema=schema_with_stage1_continuation_contract(
                SYNTH_SCHEMA,
                role="stage1c_negative",
            ),
        ),
        root / "synth_training_negative.parquet",
    )
    for name in STUB_FILE_ARTIFACTS:
        (root / name).write_bytes(b"x")
    write_seed_contract(root)
    baselines_builder(root / "baselines")
