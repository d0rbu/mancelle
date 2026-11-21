"""CPU plan/publish orchestration for a fresh, authenticated fixed ChatCORE cache.

The extraction command is a separate GPU stage. Publication uses the existing
converter and schema-4 writer, never an in-place legacy schema upgrade.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path

import hydra
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoTokenizer

from nanochat.tokenizer import get_tokenizer
from scripts.eval.build_eval_context_cache import write_context_parquet, write_context_records
from scripts.eval.chat_eval import ALL_CHAT_TASKS, _chat_task
from scripts.eval.context_cache_common import load_context_records
from steering._common.file_inventory import file_inventory
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.eval.chat_eval_result_identity import ChatCacheSnapshot, semantic_digest
from steering.eval.chat_self_conditioned_cache import resolve_fixed_chat_cache_contract
from steering.eval.chat_self_conditioning import ChatTaskEvalType
from steering.eval.context_cache_build import build_cache_entries_from_extracted_feature_shards
from steering.eval.context_cache_io import (
    LEGACY_SCHEMA_VERSION,
    CacheIdentity,
    SafetensorsContextFeatureProvider,
    _tensors_to_entries,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.context_cache_validation import validate_context_cache_records
from steering.eval.context_enumeration import (
    dedupe_context_records,
    enumerate_chat_categorical_contexts,
    enumerate_chat_generative_contexts,
)
from steering.eval.turn_context_contract import turn_extraction_tokenizer_id
from steering.extraction_identity import EXTRACTION_DOCUMENTS_VERSION, extraction_documents_sha256
from steering.feature_io import MetadataKey, glob_features, load_features
from steering.overlay_io import load_overlay_shard_bytes
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth
from steering.segmentation import SegmentMode, compute_tokenizer_checksum, resolve_bos_token_id
from steering.sft.segment_mode_identity import (
    FIXED_CADENCE_SOURCE_WIDTH,
)


class PreparationStage(StrEnum):
    PLAN = "plan"
    PUBLISH = "publish"


class LegacyReplay(StrEnum):
    REPLAY = "replay_original_enumeration"
    REBUILD = "rebuild_requested_prefixes"


def authenticate_legacy_cache(root: Path, *, contract, sequence_len: int):
    manifests = list(root.glob("c*/*/manifest.json")) if not (root / "manifest.json").is_file() else [root / "manifest.json"]
    if len(manifests) != 1:
        raise ValueError("legacy cache must identify exactly one schema-3 manifest")
    snapshot = ChatCacheSnapshot.capture(manifests[0].parent)
    manifest = json.loads(snapshot.manifest_bytes)
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != LEGACY_SCHEMA_VERSION:
        raise ValueError("legacy cache input requires schema 3")
    identity = CacheIdentity.from_dict(manifest["identity"])
    # An old selection hash/algorithm may require rebuilding; SAE identity may not change.
    for key in ("model_id", "sae_id", "layer", "pooling"):
        if getattr(identity, key) != getattr(contract.identity, key):
            raise ValueError(f"legacy SAE identity mismatch: {key}")
    if (manifest["n_sae_features"], manifest["num_buckets"], manifest["metadata"].get("model_max_seq_len")) != (
            contract.config["n_sae_features"], contract.config["num_buckets"], sequence_len):
        raise ValueError("legacy cache geometry mismatch")
    entries = []
    for shard in manifest["shards"]:
        decoded = _tensors_to_entries(load_overlay_shard_bytes((snapshot.root / shard["file"]).read_bytes()),
            identity=identity, n_sae_features=PositiveInt.parse(manifest["n_sae_features"]),
            num_buckets=PositiveInt.parse(manifest["num_buckets"]), source=shard["file"])
        entries.extend(decoded)
    rows = {entry.token_ids: entry for entry in entries}
    if len(rows) != len(entries):
        raise ValueError("legacy cache duplicate exact keys")
    snapshot.assert_stable()
    return rows, snapshot


def _assert_inventory(inventory):
    if any(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"] for row in inventory):
        raise ValueError("cache preparation input changed during plan/conversion/publication")


def _parquet_matches(directory, records):
    paths = sorted(Path(directory).glob("*.parquet"))
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    if [(row["text"], row["context_key"]) for row in rows] != [(row.context_text, row.key) for row in records]:
        raise ValueError("parquet text/key order disagrees with exact JSONL enumeration")
    return paths


def _convert(records, paths, *, contract, tokenizer, source_tokenizer, pipeline):
    offset = 0
    for path in paths:
        data = load_features(path)
        expected = {"segment_mode": SegmentMode.fixed_tokens, "segment_size": FIXED_CADENCE_SOURCE_WIDTH,
            "pooling": contract.source["pooling"], "layer": contract.source["sae"]["layer"],
            "l0_tier": contract.source["sae"]["l0"], "n_sae_features": contract.config["n_sae_features"],
            **contract.source["extraction_identity"],
            "context_independent_features": True,
            "ignore_bos_features": True}
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"extraction source geometry mismatch: {key}")
        n_docs = data.get(MetadataKey.N_DOCS)
        if type(n_docs) is not int or n_docs <= 0 or offset + n_docs > len(records):
            raise ValueError("extraction docs disagree with planned enumeration")
        expected_documents = extraction_documents_sha256(
            row.context_text for row in records[offset:offset + n_docs]
        )
        if data.get(MetadataKey.EXTRACTION_DOCUMENTS_SHA256) != expected_documents:
            raise ValueError("extraction documents do not authenticate the planned enumeration")
        offset += n_docs
    if offset != len(records):
        raise ValueError("extraction docs disagree with planned enumeration")
    bos = resolve_bos_token_id(source_tokenizer)
    if bos is None:
        raise ValueError("fixed extraction tokenizer requires BOS")
    return build_cache_entries_from_extracted_feature_shards(context_records=records, feature_shard_paths=paths,
        pipeline=pipeline, n_sae_features=PositiveInt.parse(contract.config["n_sae_features"]),
        num_buckets=PositiveInt.parse(contract.config["num_buckets"]), tokenizer=tokenizer,
        source_bos_char_len=NonNegativeInt.parse(len(source_tokenizer.decode([bos]))),
        feature_clusters=contract.feature_contract.consolidation)


def require_exact_prefixes(records, entries):
    keys = {entry.token_ids for entry in entries}
    if len(keys) != len(entries) or keys != {row.token_ids for row in records}:
        raise ValueError("refusing publication: missing/duplicate/unrequested exact completion prefixes")


def run(cfg):
    stage = PreparationStage(cfg.stage)
    meta_path = Path(cfg.checkpoint_meta_path).resolve(strict=True)
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes)
    tokenizer = get_tokenizer()
    width = (meta["n_sae_features"] if "n_sae_features" in meta else
             SAE_WIDTH_TO_NUM_FEATURES[SAEWidth(meta["user_config"]["sae"]["width"])])
    feature_contract = resolve_checkpoint_feature_contract(meta,
        n_sae_features=PositiveInt.parse(width), surface="fixed ChatCORE preparation", source_path=str(meta_path))
    contract = resolve_fixed_chat_cache_contract(meta, tokenizer, feature_contract)
    tasks = list(cfg.chat_tasks or ALL_CHAT_TASKS)
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(ALL_CHAT_TASKS):
        raise ValueError("chat_tasks requires unique canonical ChatCORE task names")
    if type(cfg.chat_max_problems) is not int or cfg.chat_max_problems <= 0:
        raise ValueError("chat_max_problems must be a positive integer")
    records, population = [], {}
    for name in tasks:
        task = _chat_task(name)
        population[name] = {"task_size": len(task), "requested": min(len(task), cfg.chat_max_problems)}
        categorical = ChatTaskEvalType(task.eval_type) is ChatTaskEvalType.CATEGORICAL
        enumeration = enumerate_chat_categorical_contexts if categorical else enumerate_chat_generative_contexts
        records.extend(enumeration(tokenizer=tokenizer, task_object=task, task_name=name,
            max_problems=cfg.chat_max_problems, **({"batch_size": 1} if categorical else {})))
    records = dedupe_context_records(records)
    legacy, legacy_snapshot, legacy_records, legacy_inputs = {}, None, [], []
    if cfg.legacy_cache_dir:
        legacy, legacy_snapshot = authenticate_legacy_cache(Path(cfg.legacy_cache_dir), contract=contract,
            sequence_len=meta["model_config"]["sequence_len"])
    if cfg.legacy_contexts_file:
        legacy_records = load_context_records(cfg.legacy_contexts_file)
        if len(legacy_records) != len(legacy) or {row.token_ids for row in legacy_records} != set(legacy):
            raise ValueError("legacy JSONL keys do not authenticate the complete legacy cache")
        legacy_inputs.append(Path(cfg.legacy_contexts_file))
        if cfg.legacy_parquet_dir:
            legacy_inputs.extend(_parquet_matches(cfg.legacy_parquet_dir, legacy_records))
    replay = LegacyReplay.REPLAY if legacy_records and cfg.legacy_feature_dir and cfg.legacy_parquet_dir else LegacyReplay.REBUILD
    if replay is LegacyReplay.REPLAY:
        legacy_inputs.extend(Path(path) for path in glob_features(cfg.legacy_feature_dir))
        if not glob_features(cfg.legacy_feature_dir):
            raise ValueError("legacy replay requested but extracted features are absent")
    work = Path(str(cfg.output_dir) + ".prepare").resolve()
    output = Path(cfg.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing cache output root: {output}")
    source_tokenizer = AutoTokenizer.from_pretrained(turn_extraction_tokenizer_id(contract.source["sae"]["model"]))
    assert source_tokenizer is not None
    legacy_inventory = file_inventory(legacy_inputs)
    authority_inventory = [
        {"path": str(meta_path), "sha256": hashlib.sha256(meta_bytes).hexdigest()},
        {"path": str(contract.statistics_path.resolve()), "sha256": contract.statistics_sha256},
    ]
    pipeline, replayed, replay_failure = contract.pipeline(), [], None
    if replay is LegacyReplay.REPLAY:
        try:
            replayed = _convert(legacy_records, [Path(path) for path in glob_features(cfg.legacy_feature_dir)],
                contract=contract, tokenizer=tokenizer, source_tokenizer=source_tokenizer, pipeline=pipeline)
        except ValueError as exc:
            # Authenticate old selected bytes above; incompatible optional raw
            # artifacts require fresh prefix extraction, with a recorded reason.
            replay_failure = {"type": type(exc).__name__, "message": str(exc)}
            replay, pipeline = LegacyReplay.REBUILD, contract.pipeline()
    _assert_inventory(authority_inventory + legacy_inventory)
    if legacy_snapshot is not None:
        legacy_snapshot.assert_stable()
    missing = [row for row in records if row.token_ids not in legacy]
    extraction = missing if replay is LegacyReplay.REPLAY else records
    source_ids = [source_tokenizer.encode(row.context_text) for row in extraction]
    extraction_window = max((len(ids) for ids in source_ids), default=FIXED_CADENCE_SOURCE_WIDTH)
    feature_dir = work / "extracted"
    sae = contract.source["sae"]
    command = ["uv", "run", "--no-sync", "python", "-m", "steering.extract_features",
        f"model={sae['model']}", f"activation_type={sae['activation_type']}", f"layer={sae['layer']}",
        f"sae.release={sae['release']}", f"sae.num_features={sae['width']}", f"sae.l0={sae.get('l0') or 'null'}",
        "segment_mode=fixed_tokens", f"segment_size={FIXED_CADENCE_SOURCE_WIDTH}", "pooling=max_pool", "context_independent_features=true",
        f"context_window={extraction_window}", f"data_dir={work / 'parquet'}", f"output_dir={feature_dir}",
        "num_shards=1", "device=cuda"]
    plan = {"version": 1, "checkpoint_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "extraction_tokenizer_checksum": compute_tokenizer_checksum(source_tokenizer),
        "contract": contract.metadata, "identity": contract.identity.as_dict(), "population": population,
        "requested_keys": [row.key for row in records], "missing_keys": [row.key for row in missing],
        "extraction_keys": [row.key for row in extraction], "replay": replay.value,
        "legacy_replay_failure": replay_failure,
        "legacy_manifest_sha256": None if legacy_snapshot is None else hashlib.sha256(legacy_snapshot.manifest_bytes).hexdigest(),
        "legacy_inventory": legacy_inventory, "extraction_command": command}
    plan_path = work / "plan.json"
    if stage is PreparationStage.PLAN:
        if plan_path.exists():
            stored = json.loads(plan_path.read_text())
            if {key: stored[key] for key in plan} != plan:
                raise ValueError("existing preparation plan disagrees; use a fresh output_dir")
            _assert_inventory(stored["enumeration_inventory"])
        else:
            work.mkdir(parents=True, exist_ok=True)
            files = [write_context_records(work / "requested.jsonl", records),
                     write_context_records(work / "missing.jsonl", missing),
                     write_context_records(work / "extraction.jsonl", extraction)]
            if extraction:
                files.append(write_context_parquet(work / "parquet", extraction))
            script = work / "extract.sh"
            script.write_text("#!/bin/sh\nset -eu\nexec " + shlex.join(command) + "\n")
            files.append(script)
            plan["enumeration_inventory"] = file_inventory(files)
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        logger.info("Requested {} prefixes, {} missing; {} require extraction. Plan: {}\n{}", len(records), len(missing), len(extraction), plan_path, shlex.join(command))
        return plan_path
    stored = json.loads(plan_path.read_text())
    if {key: stored[key] for key in plan} != plan:
        raise ValueError("preparation plan changed; refusing publication")
    _assert_inventory(stored["enumeration_inventory"])
    if extraction:
        _parquet_matches(work / "parquet", extraction)
    new_paths = [Path(path) for path in glob_features(feature_dir)]
    inputs = authority_inventory + legacy_inventory + file_inventory(new_paths)
    # The pipeline retains the CPU replay's RNG state. Missing prefixes follow
    # ALL old rows, including CORE/full rows not requested by this consumer.
    fresh = _convert(extraction, new_paths, contract=contract, tokenizer=tokenizer,
        source_tokenizer=source_tokenizer, pipeline=pipeline) if extraction else []
    rebuilt = {entry.token_ids: entry for entry in [*replayed, *fresh]}
    entries, changes, preserved = [], [], []
    for row in records:
        current = rebuilt.get(row.token_ids)
        if current is None:
            raise ValueError(f"missing converted completion prefix: {row.key}")
        old = legacy.get(row.token_ids)
        if old is not None and current == old:
            current = old
            preserved.append(row.key)
        elif old is not None:
            changes.append({"key": row.key, "old_payload_sha256": semantic_digest(asdict(old)),
                "new_payload_sha256": semantic_digest(asdict(current)),
                "reason": replay.value})
        entries.append(current)
    require_exact_prefixes(records, entries)
    metadata = {**contract.metadata, "contexts_fingerprint": semantic_digest([row.key for row in records]),
        "population": population, "preparation_provenance": {"plan": stored,
            "documents_version": EXTRACTION_DOCUMENTS_VERSION,
            "requested_documents_sha256": extraction_documents_sha256(row.context_text for row in records),
            "inputs": inputs, "preserved_keys": preserved, "content_changes": changes}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged_root = Path(temporary) / "cache"
        leaf = context_length_cache_dir(staged_root, meta["model_config"]["sequence_len"], hash_leaf=contract.identity.feature_selection_hash[:16])
        manifest = save_context_cache(leaf, entries, identity=contract.identity,
            n_sae_features=PositiveInt.parse(contract.config["n_sae_features"]), num_buckets=PositiveInt.parse(contract.config["num_buckets"]),
            metadata=metadata, shard_size=PositiveInt.parse(cfg.shard_size))
        provider = SafetensorsContextFeatureProvider(leaf, identity=contract.identity, expected_metadata=contract.metadata,
            expected_feature_vocab=effective_feature_vocab(feature_contract.universe, contract.config["n_sae_features"]) if feature_contract.clustered else None)
        validate_context_cache_records(records=records, provider=provider, tokenizer=tokenizer,
            feature_vocab=effective_feature_vocab(feature_contract.universe, contract.config["n_sae_features"]),
            require_nonempty_context_features=True, fail_on_extra_cached_contexts=True)
        _assert_inventory(inputs)
        _assert_inventory(stored["enumeration_inventory"])
        if legacy_snapshot is not None:
            legacy_snapshot.assert_stable()
        relative_manifest = manifest.relative_to(staged_root)
        os.rename(staged_root, output)
    return output / relative_manifest


@hydra.main(config_path="../../configs/eval", config_name="prepare_chat_self_conditioned_cache", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Prepared {}", run(cfg))


if __name__ == "__main__":
    main()
