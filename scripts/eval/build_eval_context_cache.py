"""Build safetensors context-feature caches for conditioned progression eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import scripts.eval.validate_eval_context_cache as validate_cache_mod
from errors import FeatureClustersError
from nanochat.tokenizer import get_tokenizer
from scripts.eval.context_cache_common import (
    identity_from_config as _identity_from_config,
)
from scripts.eval.context_cache_common import (
    load_context_records,
    load_turn_context_inputs,
    turn_contract_from_config,
    validate_turn_cache_from_config,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import resolve_feature_mask
from steering.conditional.feature_pipeline import (
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import SamplingBundle, assert_no_gradient_ranking
from steering.eval.axbench_base.seed_contract import sha256_file
from steering.eval.axbench_base.stage4_contexts import (
    load_stage4_contexts_manifest,
)
from steering.eval.cache_builder_stage import CacheBuilderStage
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.clustered_cache_seed_contract import (
    assert_clustered_seed_matches_checkpoint,
)
from steering.eval.context_cache_build import (
    build_cache_entries_from_extracted_feature_shards,
    build_turn_cache_entries,
)
from steering.eval.context_cache_io import (
    SafetensorsContextFeatureProvider,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    feature_selection_hash_leaf,
    save_context_cache,
)
from steering.eval.context_cache_validation import validate_turn_source_geometry
from steering.eval.context_enumeration import (
    EvalContextRecord,
    context_text_for_extraction,
    dedupe_context_records,
    validate_turn_context_records,
)
from steering.eval.segment_mode_guard import require_fixed_feature_dir_segment_mode
from steering.feature_io import glob_features
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.removed_features import reject_feature_clustering_config
from steering.segmentation import (
    SegmentMode,
    compute_tokenizer_checksum,
    resolve_bos_token_id,
    segment_mode_from_metadata,
)
from steering.stats_io import load_statistics, verify_statistics_sha256

if TYPE_CHECKING:
    from steering.eval.clustered_cache_seed_contract import CheckpointSeedProvenance


def _resolve_clustered_cache_contract(
    cfg: DictConfig,
    *,
    feature_mask,
    checkpoint_seed_provenance: CheckpointSeedProvenance | None = None,
) -> tuple[
    FeatureClusterConsolidation | None,
    FeatureVocab | None,
    dict,
]:
    """Resolve the optional clustered prompt-cache identity and metadata.

    ``checkpoint_seed_provenance`` is additive (PR #739 Track D): the caller's
    own early ``assert_clustered_seed_matches_checkpoint`` result (``None`` for
    every checkpoint-ref-free clustered build, and for both existing callers
    of this helper that never pass it -- ``validate_cache_from_config`` here
    and this module's own pre-Track-D call shape). Passing ``None`` (the
    default) reproduces this helper's exact pre-Track-D metadata, byte for
    byte (U-invariant).
    """
    n_sae_features = int(cfg.n_sae_features)
    resolved_clusters = resolve_feature_clusters(cfg, n_sae_features)
    if resolved_clusters is None:
        return None, None, {}
    clusters = FeatureClusterConsolidation.build(
        resolved_clusters,
        feature_mask,
    )
    feature_vocab = effective_feature_vocab(clusters, n_sae_features)
    configured_stats_sha = cfg.get("stats_source_sha256", None)
    if configured_stats_sha is None:
        raise ValueError(
            "clustered prompt-context cache builds require the exact "
            "stats_source_sha256"
        )
    verified_stats_sha = verify_statistics_sha256(
        Path(str(cfg.stats_path)),
        str(configured_stats_sha),
        surface="clustered prompt-context cache",
    )
    metadata = clustered_context_cache_metadata(
        feature_mask=feature_mask,
        feature_clusters=clusters,
        feature_vocab=feature_vocab,
        stats_source_sha256=verified_stats_sha,
        checkpoint_seed_provenance=checkpoint_seed_provenance,
    )
    return clusters, feature_vocab, metadata


def _enumerate_expected_eval_contexts(cfg: DictConfig, tokenizer) -> list[EvalContextRecord]:
    """Delegate to the validator's context enumeration implementation."""
    return validate_cache_mod.enumerate_expected_eval_contexts(cfg, tokenizer)


def write_context_records(path: str | Path, records: list[EvalContextRecord]) -> Path:
    """Write eval context records as JSONL in feature-extraction order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "suite": record.suite,
                        "task": record.task,
                        "seed": record.seed,
                        "example_index": record.example_index,
                        "row_index": record.row_index,
                        "token_ids": list(record.token_ids),
                        "context_text": record.context_text,
                        "metadata": record.metadata,
                        **({"turn_char_starts": list(record.turn_char_starts)} if record.turn_char_starts is not None else {}),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return output_path


def write_context_parquet(
    parquet_dir: str | Path,
    records: list[EvalContextRecord],
) -> Path:
    """Write the exact eval contexts as one parquet shard for SAE extraction."""

    turn_rows = any(record.turn_char_starts is not None for record in records)
    if turn_rows and any(record.turn_char_starts is None for record in records):
        raise ValueError("turn parquet requires non-null turn_char_starts on every row")
    output_dir = Path(parquet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shard_00000.parquet"
    table = pa.table(
        {
            "text": [record.context_text for record in records],
            "context_key": [record.key for record in records],
            "suite": [record.suite for record in records],
            "task": [record.task or "" for record in records],
            "token_count": [len(record.token_ids) for record in records],
        }
    )
    if turn_rows:
        table = table.append_column(
            pa.field("turn_char_starts", pa.list_(pa.int64()), nullable=False),
            pa.array([record.turn_char_starts for record in records], type=pa.list_(pa.int64())),
        )
    pq.write_table(table, output_path)
    return output_path


def _required_contexts_contract(
    cfg: DictConfig,
    records: list[EvalContextRecord],
) -> tuple[dict | None, NonNegativeInt]:
    """Validate an optional Stage 3e manifest and resolve BOS char length."""

    manifest_value = cfg.get("required_contexts_manifest")
    if manifest_value is None or str(manifest_value) == "":
        return None, NonNegativeInt.parse(int(cfg.source_bos_char_len))
    manifest_path = Path(str(manifest_value))
    manifest = load_stage4_contexts_manifest(manifest_path)
    nanochat_tokenizer = get_tokenizer()
    actual_tokenizer_checksum = compute_tokenizer_checksum(nanochat_tokenizer)
    if manifest.get("tokenizer_checksum") != actual_tokenizer_checksum:
        raise ValueError(
            "required_contexts_manifest was enumerated with a different "
            "nanochat tokenizer"
        )
    manifest_jsonl = manifest_path.parent / manifest["files"]["jsonl"][
        "relative_path"
    ]
    manifest_parquet = manifest_path.parent / manifest["files"]["parquet"][
        "relative_path"
    ]
    contexts_path = Path(str(cfg.contexts_file))
    if sha256_file(contexts_path) != sha256_file(manifest_jsonl):
        raise ValueError(
            "contexts_file does not match required_contexts_manifest JSONL"
        )
    record_keys = [record.key for record in records]
    if record_keys != manifest["context_keys"]:
        raise ValueError(
            "contexts_file ordering/keys do not match required_contexts_manifest"
        )
    if [len(record.token_ids) for record in records] != manifest[
        "context_token_counts"
    ]:
        raise ValueError(
            "contexts_file token counts do not match required_contexts_manifest"
        )
    configured_parquet = Path(str(cfg.context_parquet_dir)) / manifest_parquet.name
    if not configured_parquet.is_file() or sha256_file(configured_parquet) != (
        sha256_file(manifest_parquet)
    ):
        raise ValueError(
            "context_parquet_dir does not contain the exact parquet committed "
            "by required_contexts_manifest"
        )

    tokenizer_id = cfg.get("extraction_tokenizer_id")
    if tokenizer_id is None or str(tokenizer_id) == "":
        raise ValueError(
            "extraction_tokenizer_id is required with required_contexts_manifest"
        )
    extraction_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_id))
    if extraction_tokenizer is None:
        raise ValueError(
            f"Could not load configured extraction tokenizer {tokenizer_id!r}"
        )
    extraction_tokenizer_checksum = compute_tokenizer_checksum(extraction_tokenizer)
    bos_id = resolve_bos_token_id(extraction_tokenizer)
    if bos_id is None:
        raise ValueError(
            f"Configured extraction tokenizer {tokenizer_id!r} has no BOS token"
        )
    decoded_bos_char_len = len(extraction_tokenizer.decode([bos_id]))
    if decoded_bos_char_len <= 0:
        raise ValueError(
            f"Extraction tokenizer {tokenizer_id!r} decodes BOS to empty text"
        )
    supplied_bos_char_len = cfg.get("source_bos_char_len")
    if supplied_bos_char_len is not None and int(supplied_bos_char_len) != (
        decoded_bos_char_len
    ):
        raise ValueError(
            "source_bos_char_len disagrees with the configured extraction "
            f"tokenizer: supplied={supplied_bos_char_len}, "
            f"derived={decoded_bos_char_len}"
        )
    manifest = {
        **manifest,
        "extraction_tokenizer_checksum": extraction_tokenizer_checksum,
    }
    return manifest, NonNegativeInt.parse(decoded_bos_char_len)


def enumerate_contexts_from_config(cfg: DictConfig) -> list[EvalContextRecord]:
    """Enumerate eval contexts and write JSONL plus extraction parquet."""

    reject_feature_clustering_config(cfg, "eval context cache enumeration")
    # Validate configured plural static artifacts before writing enumeration
    # outputs, even though their contents do not alter the token contexts.
    if cfg.get("feature_clusters", None) is not None:
        n_sae_features = int(cfg.n_sae_features)
        resolve_feature_mask(cfg, n_sae_features)
        resolve_feature_clusters(cfg, n_sae_features)

    tokenizer = get_tokenizer()
    records = dedupe_context_records(_enumerate_expected_eval_contexts(cfg, tokenizer))
    if not records:
        raise ValueError("Eval context enumeration produced no records")

    mode = segment_mode_from_metadata(cfg, source="eval context enumeration")
    normalized_records = records if mode is SegmentMode.turn_delimited_fixed_tokens else [
        EvalContextRecord(
            suite=record.suite,
            task=record.task,
            seed=record.seed,
            example_index=record.example_index,
            row_index=record.row_index,
            token_ids=record.token_ids,
            context_text=context_text_for_extraction(tokenizer, record.token_ids),
            metadata=record.metadata,
        )
        for record in records
    ]
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        validate_turn_context_records(normalized_records, tokenizer)
    contexts_path = write_context_records(cfg.contexts_file, normalized_records)
    parquet_path = write_context_parquet(cfg.context_parquet_dir, normalized_records)
    logger.info(
        "Wrote {} unique eval contexts to {} and extraction parquet {}",
        len(normalized_records),
        contexts_path,
        parquet_path,
    )
    return normalized_records


def build_cache_from_config(cfg: DictConfig) -> Path:
    """Convert extracted feature shards into a safetensors eval context cache."""

    reject_feature_clustering_config(cfg, "eval context cache conversion")
    if segment_mode_from_metadata(cfg, source="eval context conversion") is SegmentMode.turn_delimited_fixed_tokens:
        return _build_turn_cache_from_config(cfg)
    n_sae_features = PositiveInt.parse(int(cfg.n_sae_features))
    num_buckets = PositiveInt.parse(int(cfg.num_buckets))
    shard_size = PositiveInt.parse(int(cfg.shard_size))

    # D-contract: fail loud, before ANY shard/context I/O -- not merely
    # before _resolve_clustered_cache_contract below, which already runs
    # after both load_context_records (next line) and glob_features (a few
    # lines further) -- when a clustered build's configured seed disagrees
    # with the target checkpoint's real training-time preselection seed.
    # Opt-in: returns None immediately unless a checkpoint reference (any
    # checkpoint_* cfg field) is configured, so unclustered and
    # checkpoint-ref-free clustered builds are unaffected (U-invariant). The
    # check reads only the checkpoint's real seed + file provenance (a narrow
    # resolve_checkpoint_pin), never the feature_mask -- so no mask config is
    # threaded here and it imposes no mask-shape constraint on the build.
    checkpoint_seed_provenance = None
    if cfg.get("feature_clusters") is not None:
        checkpoint_seed_provenance = assert_clustered_seed_matches_checkpoint(cfg)

    records = load_context_records(cfg.contexts_file)
    required_manifest, source_bos_char_len = _required_contexts_contract(
        cfg, records
    )
    shard_paths = glob_features(cfg.feature_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No feature shards found in {cfg.feature_dir}")

    stats = load_statistics(cfg.stats_path)
    require_fixed_feature_dir_segment_mode(
        cfg.feature_dir,
        surface="eval context cache conversion",
        statistics_segment_mode=stats.metadata.segment_mode,
    )
    # D-seed: a clustered cache's sampled CONTENT must use the same real
    # seed its IDENTITY hash is pinned to (steering/eval/cache_identity.py's
    # site6-eval-context-cache-clustered branch), or the path-level hash
    # change is cosmetic while every actual selection still silently
    # reproduces build_feature_pipeline's seed=42 default. Strict --
    # int(cfg.seed) raises when unset rather than soft-defaulting, so a
    # clustered build can never silently drift from the identity hash
    # regardless of call ordering. Unclustered builds omit the kwarg
    # entirely and stay byte-identical to today (U-invariant).
    clusters_configured = cfg.get("feature_clusters") is not None
    if clusters_configured:
        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=stats.per_feature_stats,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            seed=int(cfg.seed),
        )
    else:
        # Byte-identical to pre-D-seed: no seed= kwarg at all, so
        # build_feature_pipeline's own seed=42 default applies exactly as
        # it always has (U-invariant).
        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=stats.per_feature_stats,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
        )
    # Early guard: context-cache building live-selects per segment.
    assert_no_gradient_ranking(
        getattr(getattr(pipeline, "selection_config", None), "selection", None),
        "eval context cache build (live selection)",
    )
    feature_clusters, _feature_vocab, clustered_metadata = (
        _resolve_clustered_cache_contract(
            cfg,
            feature_mask=pipeline.feature_mask,
            checkpoint_seed_provenance=checkpoint_seed_provenance,
        )
    )
    if feature_clusters is not None and not isinstance(
        pipeline.selection_config,
        SamplingBundle,
    ):
        raise FeatureClustersError(
            "clustered threshold context selection is unsupported; configure "
            "the audited sampling policy"
        )
    # Resolve and compare any explicit identity before the potentially costly
    # context conversion. The clustered canonical resolver also rejects delta
    # and retained delta_weights prompt configs.
    clustered_identity = (
        _identity_from_config(cfg) if feature_clusters is not None else None
    )

    tokenizer = get_tokenizer()
    entries = build_cache_entries_from_extracted_feature_shards(
        context_records=records,
        feature_shard_paths=shard_paths,
        pipeline=pipeline,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        tokenizer=tokenizer,
        source_bos_char_len=source_bos_char_len,
        feature_clusters=feature_clusters,
    )
    model_max_seq_len = int(cfg.model_max_seq_len)
    metadata = OmegaConf.to_container(cfg.metadata, resolve=True) or {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must resolve to a mapping or null")
    existing_context_len = metadata.get("model_max_seq_len")
    if existing_context_len is not None and int(existing_context_len) != model_max_seq_len:
        raise ValueError(
            "metadata.model_max_seq_len must match model_max_seq_len; "
            f"metadata={existing_context_len}, model_max_seq_len={model_max_seq_len}"
        )
    metadata["model_max_seq_len"] = model_max_seq_len
    metadata.update(clustered_metadata)
    if required_manifest is not None:
        metadata["contexts_fingerprint"] = required_manifest[
            "contexts_fingerprint"
        ]
        metadata["context_extraction_tokenizer_checksum"] = required_manifest[
            "extraction_tokenizer_checksum"
        ]
    identity = clustered_identity or _identity_from_config(cfg)
    cache_dir = context_length_cache_dir(
        cfg.output_dir,
        model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
    )
    manifest = save_context_cache(
        cache_dir,
        entries,
        identity=identity,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        metadata=metadata,
        shard_size=shard_size,
        overwrite=bool(cfg.overwrite),
        zstd_level=cfg.get("zstd_level"),
    )
    logger.info(f"Wrote eval context cache manifest: {manifest}")
    return manifest


def _build_turn_cache_from_config(cfg: DictConfig) -> Path:
    tokenizer = get_tokenizer()
    contract, extraction_tokenizer = turn_contract_from_config(cfg, tokenizer)
    identity = cache_identity_from_config(cfg.get("identity"), turn_contract=contract)
    records, shard_paths, provenance = load_turn_context_inputs(cfg, tokenizer)
    validate_turn_source_geometry(records=records, feature_shard_paths=shard_paths,
                                  contract=contract, extraction_tokenizer=extraction_tokenizer, tokenizer=tokenizer)
    entries = build_turn_cache_entries(
        context_records=records, feature_shard_paths=shard_paths,
        turn_contract=contract, tokenizer=tokenizer,
    )
    metadata = OmegaConf.to_container(cfg.get("metadata", {}), resolve=True) or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    required_metadata = {**contract.provenance, **provenance}
    for key, value in required_metadata.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"turn cache provenance override for {key}")
    metadata.update(required_metadata)
    cache_dir = context_length_cache_dir(cfg.output_dir, contract.model_max_seq_len,
                                         hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash))
    return save_context_cache(
        cache_dir, entries, identity=identity,
        n_sae_features=PositiveInt.parse(contract.lineage.n_sae_features),
        num_buckets=PositiveInt.parse(contract.lineage.num_buckets), metadata=metadata,
        shard_size=PositiveInt.parse(int(cfg.shard_size)), overwrite=bool(cfg.overwrite),
        zstd_level=cfg.get("zstd_level"), source_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        turn_contract=contract,
    )


def validate_cache_from_config(cfg: DictConfig) -> None:
    reject_feature_clustering_config(cfg, "eval context cache validation")
    if segment_mode_from_metadata(cfg, source="eval context validation") is SegmentMode.turn_delimited_fixed_tokens:
        summary = validate_turn_cache_from_config(cfg, get_tokenizer(), cache_root=cfg.output_dir)
        logger.info("Validated {} turn cache rows", summary["cached_contexts"])
        return
    model_max_seq_len = int(cfg.model_max_seq_len)
    output_dir = cfg.output_dir
    n_sae_features = int(cfg.n_sae_features)
    feature_mask = resolve_feature_mask(cfg, n_sae_features)
    feature_clusters, feature_vocab, clustered_metadata = (
        _resolve_clustered_cache_contract(
            cfg,
            feature_mask=feature_mask,
        )
    )
    identity = _identity_from_config(cfg)
    cache_dir = context_length_cache_dir(
        output_dir,
        model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
    )
    expected_metadata = {"model_max_seq_len": model_max_seq_len}
    expected_metadata.update(clustered_metadata)
    manifest_value = cfg.get("required_contexts_manifest")
    if manifest_value is not None and str(manifest_value) != "":
        records = load_context_records(cfg.contexts_file)
        required_manifest, _source_bos_char_len = _required_contexts_contract(
            cfg, records
        )
        assert required_manifest is not None
        expected_metadata["contexts_fingerprint"] = required_manifest[
            "contexts_fingerprint"
        ]
        expected_metadata["context_extraction_tokenizer_checksum"] = (
            required_manifest["extraction_tokenizer_checksum"]
        )
    provider = SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_n_sae_features=int(cfg.n_sae_features),
        expected_num_buckets=int(cfg.num_buckets),
        expected_metadata=expected_metadata,
        expected_feature_vocab=(
            feature_vocab if feature_clusters is not None else None
        ),
    )
    if len(provider) == 0:
        raise ValueError(f"Context cache is empty: {provider.path}")
    logger.info(f"Validated {len(provider)} cached context rows from {provider.path}")


@hydra.main(
    config_path="../../configs/eval",
    config_name="build_eval_context_cache",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    match CacheBuilderStage(cfg.stage):
        case CacheBuilderStage.ENUMERATE:
            enumerate_contexts_from_config(cfg)
        case CacheBuilderStage.CONVERT:
            build_cache_from_config(cfg)
        case CacheBuilderStage.VALIDATE:
            validate_cache_from_config(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
