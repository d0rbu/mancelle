"""Stage 3e: enumerate every exact encoder-decoder prompt used by Stage 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from loguru import logger

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet, atomic_write_text
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.cluster_artifacts import (
    _load_stable_json,
    install_clustered_bytes_once,
    load_cluster_candidate_identity,
    write_clustered_parquet_once,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    BaselineKind,
    Method,
    get_method_by_name,
)
from steering.eval.axbench_base.seed_contract import (
    load_seed_pool_fingerprint,
    sha256_bytes,
    sha256_file,
)
from steering.eval.axbench_base.stage4_prompts import (
    build_stage4_prompt_tokens,
    load_prompting_descriptions,
    prompt_token_key,
    prompting_description_hashes,
    prompting_prefix_text,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    SteeringMode,
    TargetModelMetadata,
)
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.eval.context_enumeration import context_text_for_extraction
from steering.immutable_io import is_sha256_hex
from steering.segmentation import compute_tokenizer_checksum

STAGE4_CONTEXTS_SCHEMA_VERSION = 1
CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION = 2
_FRESH_RUN_TAG = (
    "Stage 3e context artifacts do not match the current seed/config contract; "
    "use a fresh run tag."
)
_CLUSTER_MANIFEST_FIELDS = (
    "feature_contract_fingerprint",
    "candidate_concepts_sha256",
    "candidate_cluster_provenance_sha256",
    "cluster_concept_projection_version",
    "concepts_sha256",
)

def _validate_cluster_manifest_stamp(manifest: dict[str, Any]) -> None:
    """Enforce bidirectional ownership between schema and cluster fields."""

    schema_version = manifest.get("schema_version")
    if schema_version == CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION:
        for field in (
            "feature_contract_fingerprint",
            "candidate_concepts_sha256",
            "candidate_cluster_provenance_sha256",
            "concepts_sha256",
        ):
            if not is_sha256_hex(manifest.get(field)):
                raise AxBenchBaseError(_FRESH_RUN_TAG)
        projection_version = manifest.get("cluster_concept_projection_version")
        if not isinstance(projection_version, str) or not projection_version:
            raise AxBenchBaseError(_FRESH_RUN_TAG)
        return
    if any(field in manifest for field in _CLUSTER_MANIFEST_FIELDS):
        raise AxBenchBaseError(_FRESH_RUN_TAG)


def _enabled_encdec_methods(
    cfg: AxBenchBaseConfig,
    encdec_meta: TargetModelMetadata,
) -> list[Method]:
    if cfg.enabled_methods is not None:
        for name in cfg.enabled_methods:
            get_method_by_name(name)
    for name in cfg.disabled_methods:
        get_method_by_name(name)
    methods: list[Method] = []
    for method in ALL_METHODS:
        if method.target_model != "encdec":
            continue
        if not cfg.run_negative_mode and method.mode is SteeringMode.NEGATIVE:
            continue
        if cfg.enabled_methods is not None and method.name not in cfg.enabled_methods:
            continue
        if method.name in cfg.disabled_methods:
            continue
        if method.requires_negative_sampling and encdec_meta.negative_selection is None:
            continue
        methods.append(method)
    return methods


def _seeds_for_mode(
    cfg: AxBenchBaseConfig,
    mode: SteeringMode,
    *,
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
) -> list[SeedRow]:
    if mode is SteeringMode.POSITIVE:
        seeds = eval_seeds_positive
        cap = cfg.generation.eval_seeds_per_concept_positive
    else:
        seeds = eval_seeds_negative
        cap = cfg.generation.eval_seeds_per_concept_negative
    if cap < 0:
        raise AxBenchBaseError(f"Stage 3e seed cap must be non-negative, got {cap}")
    return seeds[:cap]


def _method_fingerprint_payload(
    cfg: AxBenchBaseConfig,
    methods: list[Method],
) -> dict[str, Any]:
    return {
        "methods": [
            {
                "name": method.name,
                "mode": method.mode.value,
                "baseline": method.baseline.value,
                "target_model": method.target_model,
                "requires_negative_sampling": method.requires_negative_sampling,
            }
            for method in methods
        ],
        "enabled_methods": cfg.enabled_methods,
        "disabled_methods": cfg.disabled_methods,
        "run_negative_mode": cfg.run_negative_mode,
        "eval_seeds_per_concept_positive": (
            cfg.generation.eval_seeds_per_concept_positive
        ),
        "eval_seeds_per_concept_negative": (
            cfg.generation.eval_seeds_per_concept_negative
        ),
    }


def _enumerate_records(
    cfg: AxBenchBaseConfig,
    *,
    methods: list[Method],
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    prompting_modes = {
        method.mode
        for method in methods
        if method.baseline is BaselineKind.PROMPTING
    }
    descriptions = {
        mode: load_prompting_descriptions(cfg, mode)
        for mode in sorted(prompting_modes, key=lambda value: value.value)
    }
    records_by_key: dict[str, dict[str, Any]] = {}
    for method in methods:
        seeds = _seeds_for_mode(
            cfg,
            method.mode,
            eval_seeds_positive=eval_seeds_positive,
            eval_seeds_negative=eval_seeds_negative,
        )
        if not concepts or not seeds:
            continue
        if method.baseline is BaselineKind.PROMPTING:
            contexts = [
                (concept, seed_index, seed)
                for concept in concepts
                for seed_index, seed in enumerate(seeds)
            ]
        else:
            contexts = [
                (concepts[0], seed_index, seed)
                for seed_index, seed in enumerate(seeds)
            ]
        for concept, seed_index, seed in contexts:
            prompt_prefix: str | None = None
            if method.baseline is BaselineKind.PROMPTING:
                description = descriptions[method.mode].get(concept.concept_id)
                if description is None:
                    raise AxBenchBaseError(
                        "Stage 3e prompting description is missing for "
                        f"mode={method.mode.value}, concept={concept.concept_id}"
                    )
                prompt_prefix = prompting_prefix_text(method.mode, description)
            token_ids = build_stage4_prompt_tokens(
                tokenizer=tokenizer,
                seed_content_tokens=seed.prefix_content_nanochat_tokens,
                prompting_prefix=prompt_prefix,
            )
            key = prompt_token_key(token_ids)
            existing = records_by_key.get(key)
            use = {
                "method_name": method.name,
                "mode": method.mode.value,
                "concept_id": concept.concept_id,
                "eval_seed": seed_index,
            }
            if existing is not None:
                if existing["token_ids"] != token_ids:
                    raise AxBenchBaseError(
                        f"Stage 3e token key collision for {key}"
                    )
                existing["uses"].append(use)
                continue
            records_by_key[key] = {
                "context_key": key,
                "token_ids": token_ids,
                "context_text": context_text_for_extraction(tokenizer, token_ids),
                "uses": [use],
            }
    return list(records_by_key.values())


def _jsonl_payload(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, record in enumerate(records):
        lines.append(
            json.dumps(
                {
                    "suite": "axbench_base_stage4",
                    "task": None,
                    "seed": None,
                    "example_index": index,
                    "row_index": None,
                    "token_ids": record["token_ids"],
                    "context_text": record["context_text"],
                    "metadata": {
                        "context_key": record["context_key"],
                        "uses": record["uses"],
                    },
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def load_stage4_contexts_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and fully validate one committed Stage 3e manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(f"Unreadable Stage 3e manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in {
        STAGE4_CONTEXTS_SCHEMA_VERSION,
        CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION,
    }:
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    _validate_cluster_manifest_stamp(manifest)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"jsonl", "parquet"}:
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    resolved_files: dict[str, Path] = {}
    for label, artifact in files.items():
        if not isinstance(artifact, dict):
            raise AxBenchBaseError(_FRESH_RUN_TAG)
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise AxBenchBaseError(_FRESH_RUN_TAG)
        path = (manifest_path.parent / relative_path).resolve()
        try:
            path.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise AxBenchBaseError(_FRESH_RUN_TAG) from exc
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise AxBenchBaseError(_FRESH_RUN_TAG)
        resolved_files[label] = path

    semantic_payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"contexts_fingerprint", "files"}
    }
    expected_fingerprint = sha256_bytes(
        canonical_json(semantic_payload).encode("utf-8")
    )
    if manifest.get("contexts_fingerprint") != expected_fingerprint:
        raise AxBenchBaseError(_FRESH_RUN_TAG)

    json_records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            resolved_files["jsonl"].read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                raise AxBenchBaseError(
                    f"Blank Stage 3e JSONL row at line {line_number}"
                )
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                raise AxBenchBaseError(_FRESH_RUN_TAG)
            token_ids = [int(token) for token in row.get("token_ids", [])]
            key = row["metadata"].get("context_key")
            if not token_ids or key != prompt_token_key(token_ids):
                raise AxBenchBaseError(_FRESH_RUN_TAG)
            json_records.append(
                {
                    "key": key,
                    "tokens": token_ids,
                    "text": row.get("context_text"),
                }
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AxBenchBaseError(_FRESH_RUN_TAG) from exc
    keys = [record["key"] for record in json_records]
    counts = [len(record["tokens"]) for record in json_records]
    if (
        keys != manifest.get("context_keys")
        or counts != manifest.get("context_token_counts")
        or len(json_records) != manifest.get("context_count")
        or len(keys) != len(set(keys))
    ):
        raise AxBenchBaseError(_FRESH_RUN_TAG)

    try:
        table = pq.read_table(resolved_files["parquet"])
    except (OSError, pa.ArrowException) as exc:
        raise AxBenchBaseError(_FRESH_RUN_TAG) from exc
    required_columns = {"candidate_index", "text", "context_key", "token_count"}
    if set(table.column_names) != required_columns:
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    if table.column("candidate_index").to_pylist() != list(range(len(keys))):
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    if (
        table.column("context_key").to_pylist() != keys
        or table.column("token_count").to_pylist() != counts
        or table.column("text").to_pylist()
        != [record["text"] for record in json_records]
    ):
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    return manifest


def _validate_existing_manifest(
    manifest_path: Path,
    expected_semantics: dict[str, Any],
) -> bool:
    if not manifest_path.exists():
        return False
    manifest = load_stage4_contexts_manifest(manifest_path)
    semantic_keys = tuple(expected_semantics)
    if any(manifest.get(key) != expected_semantics[key] for key in semantic_keys):
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    return True


def _build_expected_contract(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    tokenizer: Any,
    encdec_meta: TargetModelMetadata,
    encdec_sequence_len: int,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the pure semantic contract shared by enumeration and validation."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before Stage 3e")
    if (
        not isinstance(encdec_sequence_len, int)
        or isinstance(encdec_sequence_len, bool)
        or encdec_sequence_len <= 0
    ):
        raise AxBenchBaseError(
            f"Stage 3e encdec_sequence_len must be positive, got {encdec_sequence_len!r}"
        )
    cluster_semantics: dict[str, Any] = {}
    schema_version = STAGE4_CONTEXTS_SCHEMA_VERSION
    if cluster_contract is not None:
        schema_version = CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION
        # Named bug-fix deviation from the oracle: bind the exact hashed
        # concepts.json snapshot to the ordered concept_id/feature_index pairs
        # that Stage 3e consumes for records and representative validation.
        concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
        if not concepts_path.is_file():
            raise AxBenchBaseError(
                f"Clustered Stage 3e requires committed final concepts at {concepts_path}"
            )
        committed_concepts, concepts_sha256 = _load_stable_json(
            concepts_path,
            surface="final concepts",
        )
        if not isinstance(committed_concepts, list):
            raise AxBenchBaseError(
                f"Clustered final concepts at {concepts_path} must be a JSON array"
            )
        committed_identity: list[tuple[int, int]] = []
        for index, row in enumerate(committed_concepts):
            if not isinstance(row, dict):
                raise AxBenchBaseError(
                    f"Clustered final concepts[{index}] must be a JSON object"
                )
            concept_id = row.get("concept_id")
            feature_index = row.get("feature_index")
            if type(concept_id) is not int or type(feature_index) is not int:
                raise AxBenchBaseError(
                    "Clustered final concept identity fields must be strict ints: "
                    f"index={index}, concept_id={concept_id!r}, "
                    f"feature_index={feature_index!r}"
                )
            committed_identity.append((concept_id, feature_index))
        supplied_identity: list[tuple[int, int]] = []
        for index, concept in enumerate(concepts):
            if (
                type(concept.concept_id) is not int
                or type(concept.feature_index) is not int
            ):
                raise AxBenchBaseError(
                    "Clustered Stage 3e caller concept identity fields must be "
                    f"strict ints at index {index}"
                )
            supplied_identity.append((concept.concept_id, concept.feature_index))
        if supplied_identity != committed_identity:
            raise AxBenchBaseError(
                "Clustered Stage 3e caller concepts do not match the committed "
                f"final concepts at {concepts_path}: "
                f"committed={committed_identity}, supplied={supplied_identity}"
            )
        feature_ids = t.tensor(
            [concept.feature_index for concept in concepts],
            dtype=t.int64,
        )
        cluster_contract.consolidation.assert_representatives_only(
            feature_ids,
            surface="AxBench Stage 3e concept list",
        )
        candidate_identity = load_cluster_candidate_identity(
            cfg,
            expected_fingerprint=cluster_contract.feature_contract_fingerprint,
        )
        if candidate_identity is None:
            raise AxBenchBaseError(
                "Clustered Stage 3e requires the committed Stage 0a candidate provenance"
            )
        cluster_semantics = {
            "feature_contract_fingerprint": (
                cluster_contract.feature_contract_fingerprint
            ),
            "candidate_concepts_sha256": (
                candidate_identity.candidate_concepts_sha256
            ),
            "candidate_cluster_provenance_sha256": (
                candidate_identity.candidate_provenance_sha256
            ),
            "cluster_concept_projection_version": (
                candidate_identity.concept_projection_version
            ),
            # Stage 0c may reject candidates or change their public concept
            # IDs while leaving Stage 0a bytes untouched. Bind Stage 3e to the
            # exact final concept artifact so a same-prompt coincidence cannot
            # legitimize stale context provenance.
            "concepts_sha256": concepts_sha256,
        }

    methods = _enabled_encdec_methods(cfg, encdec_meta)
    records = _enumerate_records(
        cfg,
        methods=methods,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
        tokenizer=tokenizer,
    )
    for record in records:
        prompt_count = len(record["token_ids"])
        if prompt_count + cfg.generation.max_new_tokens > encdec_sequence_len:
            raise AxBenchBaseError(
                "Stage 3e prompt exceeds the encoder-decoder checkpoint length: "
                f"context_key={record['context_key']}, prompt_token_count={prompt_count}, "
                f"max_new_tokens={cfg.generation.max_new_tokens}, "
                f"sequence_len={encdec_sequence_len}"
            )
    seed_fingerprint = load_seed_pool_fingerprint(paths.AxBenchBasePaths.from_config(cfg))
    tokenizer_checksum = compute_tokenizer_checksum(tokenizer)
    prompting_modes = sorted(
        {method.mode for method in methods if method.baseline is BaselineKind.PROMPTING},
        key=lambda value: value.value,
    )
    prompt_hashes = prompting_description_hashes(cfg, prompting_modes)
    method_payload = _method_fingerprint_payload(cfg, methods)
    method_fingerprint = sha256_bytes(canonical_json(method_payload).encode("utf-8"))
    context_keys = [record["context_key"] for record in records]
    semantics = {
        "schema_version": schema_version,
        "seed_pool_fingerprint": seed_fingerprint,
        "prompting_description_hashes": prompt_hashes,
        "method_config_fingerprint": method_fingerprint,
        "method_config": method_payload,
        "tokenizer_checksum": tokenizer_checksum,
        "encdec_sequence_len": encdec_sequence_len,
        "context_count": len(records),
        "context_keys": context_keys,
        "context_token_counts": [len(record["token_ids"]) for record in records],
        **cluster_semantics,
    }
    return records, {
        **semantics,
        "contexts_fingerprint": sha256_bytes(
            canonical_json(semantics).encode("utf-8")
        ),
    }


def enumerate_stage4_contexts(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    tokenizer: Any,
    encdec_meta: TargetModelMetadata,
    encdec_sequence_len: int,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Path:
    """Commit Stage 3e JSONL, extraction parquet, and manifest atomically."""

    records, expected_semantics = _build_expected_contract(
        cfg,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
        tokenizer=tokenizer,
        encdec_meta=encdec_meta,
        encdec_sequence_len=encdec_sequence_len,
        cluster_contract=cluster_contract,
    )
    context_keys = [record["context_key"] for record in records]
    assert cfg.run_tag is not None
    manifest_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_manifest()
    if _validate_existing_manifest(manifest_path, expected_semantics):
        logger.info("Stage 3e context artifacts already validate; reusing them")
        return manifest_path

    jsonl_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_jsonl()
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_parquet()
    if cluster_contract is not None:
        orphaned = [path for path in (jsonl_path, parquet_path) if path.exists()]
        if orphaned:
            raise AxBenchBaseError(
                "Clustered Stage 3e found context artifacts without their exact "
                f"commit manifest: {orphaned}. Use a fresh run tag."
            )
    if cluster_contract is None:
        atomic_write_text(_jsonl_payload(records), jsonl_path)
    else:
        install_clustered_bytes_once(
            _jsonl_payload(records).encode("utf-8"),
            jsonl_path,
            surface="Stage 3e context JSONL",
        )
    table = pa.table(
        {
            "candidate_index": pa.array(range(len(records)), type=pa.int64()),
            "text": pa.array(
                [record["context_text"] for record in records], type=pa.string()
            ),
            "context_key": pa.array(context_keys, type=pa.string()),
            "token_count": pa.array(
                [len(record["token_ids"]) for record in records], type=pa.int32()
            ),
        }
    )
    if cluster_contract is None:
        atomic_write_parquet(table, parquet_path)
    else:
        write_clustered_parquet_once(
            table,
            parquet_path,
            surface="Stage 3e extraction contexts",
        )
    if pq.read_table(parquet_path).column("context_key").to_pylist() != context_keys:
        raise AxBenchBaseError("Stage 3e parquet ordering changed during write")
    run_root = paths.AxBenchBasePaths.from_config(cfg).run_output_root()
    manifest = {
        **expected_semantics,
        "files": {
            "jsonl": {
                "relative_path": str(jsonl_path.relative_to(run_root)),
                "sha256": sha256_file(jsonl_path),
            },
            "parquet": {
                "relative_path": str(parquet_path.relative_to(run_root)),
                "sha256": sha256_file(parquet_path),
            },
        },
    }
    if cluster_contract is None:
        atomic_json_dump(manifest, manifest_path)
    else:
        install_clustered_bytes_once(
            json.dumps(manifest, indent=2).encode("utf-8"),
            manifest_path,
            surface="Stage 3e context commit manifest",
        )
    logger.info(
        "Stage 3e committed {} unique encoder-decoder prompt contexts to {}",
        len(records),
        manifest_path,
    )
    return manifest_path


def validate_stage4_contexts_for_generation(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    tokenizer: Any,
    encdec_meta: TargetModelMetadata,
    encdec_sequence_len: int,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require Stage 3e to match every current Stage 4 determinant."""

    records, expected_semantics = _build_expected_contract(
        cfg,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
        tokenizer=tokenizer,
        encdec_meta=encdec_meta,
        encdec_sequence_len=encdec_sequence_len,
        cluster_contract=cluster_contract,
    )
    assert cfg.run_tag is not None
    manifest_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_manifest()
    if not manifest_path.is_file():
        raise AxBenchBaseError(
            f"Stage 3e manifest is missing at {manifest_path}; run Stage 3e "
            "and rebuild the exact context cache before Stage 4"
        )
    manifest = load_stage4_contexts_manifest(manifest_path)
    if any(
        manifest.get(key) != value
        for key, value in expected_semantics.items()
    ):
        raise AxBenchBaseError(_FRESH_RUN_TAG)
    return manifest, records


__all__ = [
    "CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION",
    "STAGE4_CONTEXTS_SCHEMA_VERSION",
    "enumerate_stage4_contexts",
    "load_stage4_contexts_manifest",
    "validate_stage4_contexts_for_generation",
]
