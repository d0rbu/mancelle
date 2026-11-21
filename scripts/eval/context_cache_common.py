"""Shared helpers for eval context-cache build and validation scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_cache_io import (
    CacheIdentity,
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
    feature_selection_hash_leaf,
)
from steering.eval.context_cache_validation import validate_turn_context_cache_records
from steering.eval.context_enumeration import EvalContextRecord, validate_turn_context_records
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.turn_checkpoint_provenance import (
    authenticate_turn_checkpoint_provenance,
    turn_checkpoint_provenance,
)
from steering.eval.turn_context_contract import (
    TURN_ENUMERATION_SEMANTICS,
    TURN_SELECTION_SEMANTICS,
    TurnContextContract,
    canonical_turn_payload,
    resolve_turn_context_contract,
    turn_extraction_tokenizer_id,
)
from steering.feature_io import glob_features
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY


def load_context_records(path: str | Path) -> list[EvalContextRecord]:
    """Load enumerated eval contexts from JSONL."""

    records: list[EvalContextRecord] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            if "token_ids" not in row:
                raise ValueError(f"{path}:{line_no}: missing token_ids")
            metadata = row.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_no}: metadata must be a JSON object")
            records.append(
                EvalContextRecord(
                    suite=str(row.get("suite", "")),
                    task=row.get("task"),
                    seed=row.get("seed"),
                    example_index=row.get("example_index"),
                    row_index=row.get("row_index"),
                    token_ids=tuple(int(tok) for tok in row["token_ids"]),
                    context_text=str(row.get("context_text", "")),
                    metadata=dict(metadata),
                    turn_char_starts=(tuple(row["turn_char_starts"]) if "turn_char_starts" in row else None),
                )
            )
    if not records:
        raise ValueError(f"No context records found in {path}")
    return records


def identity_from_config(cfg: DictConfig) -> CacheIdentity:
    """Extract the mandatory SAE-identity bundle from a Hydra cfg."""

    try:
        identity_cfg = cfg.identity
    except Exception as exc:  # pragma: no cover - omegaconf key missing
        raise ValueError(
            "cfg.identity is required (model_id/sae_id/layer/pooling/"
            "feature_selection_hash); context caches must be keyed by SAE "
            "identity."
        ) from exc
    raw = OmegaConf.to_container(identity_cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.identity must resolve to a mapping")
    return cache_identity_from_config(raw, cfg, prefix="cfg.identity")


def turn_contract_from_config(cfg: DictConfig, tokenizer) -> tuple[TurnContextContract, object]:
    """Read checkpoint metadata once and treat optional CLI settings as assertions."""
    checkpoint_path = cfg.get("checkpoint_meta_path")
    if not checkpoint_path:
        raise ValueError("turn cache requires checkpoint_meta_path")
    payload = Path(str(checkpoint_path)).read_bytes()
    meta = json.loads(payload)
    if not isinstance(meta, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    lineage = meta.get(ONLINE_SAE_LINEAGE_META_KEY)
    if not isinstance(lineage, dict) or not isinstance(lineage.get("model"), str):
        raise ValueError("turn cache requires complete online-SAE lineage")
    model_id = lineage["model"]
    tokenizer_id = turn_extraction_tokenizer_id(model_id, cfg.get("extraction_tokenizer_id"))
    extraction_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    contract = resolve_turn_context_contract(
        meta, tokenizer=tokenizer, extraction_tokenizer=extraction_tokenizer,
        statistics_path=cfg.get("stats_path"),
    )
    contract.provenance.update(turn_checkpoint_provenance(Path(str(checkpoint_path)), payload, meta))
    assertions = {
        "n_sae_features": contract.lineage.n_sae_features,
        "num_buckets": contract.lineage.num_buckets,
        "source_bos_char_len": contract.semantic["source_bos_char_len"],
        "source_segment_size": contract.semantic["source_segment_size"],
        "source_context_window": contract.semantic["source_context_window"],
        "model_max_seq_len": contract.model_max_seq_len,
        "seed": contract.lineage.selection_seed,
        "stats_source_sha256": contract.lineage.statistics_sha256,
        "feature_selection": dict(contract.lineage.feature_selection),
        "negative_selection": contract.lineage.negative_selection,
        "feature_mask": contract.semantic["feature_mask"],
    }
    if cfg.get("feature_clusters") is not None or cfg.get("feature_delta") is not None:
        raise ValueError("turn ChatCORE cache does not support clustered/delta selection overrides")
    for field, expected in assertions.items():
        value = cfg.get(field)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if value is not None and canonical_turn_payload({field: value}) != canonical_turn_payload({field: expected}):
            raise ValueError(f"configured {field} conflicts with checkpoint turn contract; use the turn cache preset")
    return contract, extraction_tokenizer


def load_turn_context_inputs(cfg: DictConfig, tokenizer) -> tuple[list[EvalContextRecord], list[Path], dict]:
    """Bind ordered JSONL/parquet rows and record ordered extraction input/output provenance."""
    records = load_context_records(cfg.contexts_file)
    validate_turn_context_records(records, tokenizer)
    parquet_paths = sorted(Path(str(cfg.context_parquet_dir)).glob("*.parquet"))
    if not parquet_paths:
        raise ValueError("turn context_parquet_dir contains no parquet files")
    parquet_rows = []
    for path in parquet_paths:
        table = pq.read_table(path)
        if "turn_char_starts" not in table.schema.names or table.schema.field("turn_char_starts") != pa.field(
            "turn_char_starts", pa.list_(pa.int64()), nullable=False,
        ):
            raise ValueError("turn parquet requires non-null list<int64> turn_char_starts")
        parquet_rows.extend(table.to_pylist())
    expected_rows = [{"text": record.context_text, "context_key": record.key,
                      "suite": record.suite, "task": record.task or "",
                      "token_count": len(record.token_ids), "turn_char_starts": list(record.turn_char_starts or ())}
                     for record in records]
    if parquet_rows != expected_rows:
        raise ValueError("turn JSONL and parquet row order/keys/text/boundaries disagree")
    shard_paths = sorted(glob_features(cfg.feature_dir))
    if not shard_paths:
        raise ValueError("turn feature_dir contains no extracted shards")
    fingerprint = hashlib.sha256(canonical_turn_payload({
        "enumeration_semantics": TURN_ENUMERATION_SEMANTICS,
        "selection_semantics": TURN_SELECTION_SEMANTICS,
        "rows": [{**row, "token_ids": list(record.token_ids), "metadata": record.metadata}
                 for row, record in zip(expected_rows, records, strict=True)],
    })).hexdigest()
    provenance = {
        "contexts_fingerprint": fingerprint,
        "input_provenance": {
            "contexts_file": str(cfg.contexts_file), "evaluation_feature_dir": str(cfg.feature_dir),
            "jsonl_sha256": hashlib.sha256(Path(str(cfg.contexts_file)).read_bytes()).hexdigest(),
            "parquet_files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in parquet_paths],
            "feature_files": [{"name": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                              for path in shard_paths],
        },
    }
    return records, [Path(path) for path in shard_paths], provenance


def validate_turn_cache_from_config(cfg: DictConfig, tokenizer, *, cache_root: str | Path) -> dict:
    """One deep validation implementation for builder and standalone entry points."""
    if cfg.get("target_feature_configs") is not None:
        raise ValueError("turn validation supports target-real configs only")
    contract, extraction_tokenizer = turn_contract_from_config(cfg, tokenizer)
    identity = cache_identity_from_config(cfg.get("identity"), turn_contract=contract)
    records, paths, provenance = load_turn_context_inputs(cfg, tokenizer)
    cache_dir = context_length_cache_dir(cache_root, contract.model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash))
    provider = SafetensorsContextFeatureProvider(cache_dir, identity=identity,
        expected_source_segment_mode=SegmentMode.turn_delimited_fixed_tokens, expected_turn_contract=contract,
        expected_metadata={"contexts_fingerprint": provenance["contexts_fingerprint"]})
    authenticate_turn_checkpoint_provenance(provider.manifest["metadata"], contract.provenance)
    stored = provider.manifest["metadata"].get("input_provenance")
    if not isinstance(stored, dict):
        raise ValueError("turn cache lacks ordered input provenance")
    # Relocated identical files remain valid: compare ordered names/digests, not directories.
    for field in ("jsonl_sha256", "parquet_files", "feature_files"):
        if stored.get(field) != provenance["input_provenance"][field]:
            raise ValueError(f"turn cache ordered {field} provenance disagrees")
    summary = validate_turn_context_cache_records(
        records=records, provider=provider, tokenizer=tokenizer,
        turn_contract=contract, extraction_tokenizer=extraction_tokenizer, feature_shard_paths=paths,
        require_nonempty_context_features=bool(cfg.get("require_nonempty_context_features", True)),
        require_nonempty_context_segments=bool(cfg.get("require_nonempty_context_segments", False)),
        target_conditioning_mode=cfg.get("target_conditioning_mode", TargetConditioningMode.CURRENT.value),
    ).to_dict()
    summary_file = cfg.get("summary_file")
    if summary_file:
        path = Path(str(summary_file))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
