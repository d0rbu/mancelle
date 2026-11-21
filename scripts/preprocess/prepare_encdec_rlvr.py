"""Prepare pinned, question-only ChatCORE data for encoder-decoder RLVR."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.tokenizer import get_tokenizer
from scripts.eval.build_eval_context_cache import write_context_parquet, write_context_records
from scripts.eval.context_cache_common import turn_contract_from_config
from steering.conditional.encoder_decoder.rlvr_data import (
    RLVR_BUNDLE_FORMAT_VERSION,
    RLVR_CACHE_ROLE,
    RLVR_DATASET_REVISIONS,
    RLVR_GRADER_CONTRACT,
    RLVR_MAX_NEW_TOKENS,
    RLVR_RENDER_CONTRACT,
    RLVR_TASKS,
    RLVRBundle,
    RLVRTaskSource,
    admit_rlvr_prompt_cache,
    bundle_manifest_sha256,
    jsonl_bytes,
    load_rlvr_jsonl,
    ordered_row_ids_sha256,
    prepare_rlvr_rows,
)
from steering.eval.context_cache_io import (
    CacheIdentity,
    SafetensorsContextFeatureProvider,
    feature_selection_hash_leaf,
)
from steering.eval.context_enumeration import (
    dedupe_context_records,
    enumerate_chat_rlvr_contexts,
)
from steering.eval.dataset_revision import validate_dataset_revision
from steering.eval.turn_context_contract import (
    turn_context_contract_hash,
    turn_context_identity_fields,
)
from steering.immutable_io import install_bytes_once, stream_hash_file_descriptor_bound
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.mixtures import exclude_heldout_questions, question_key
from tasks.mmlu import MMLU


def _sha256_file(path: Path) -> str:
    return stream_hash_file_descriptor_bound(path.resolve(strict=True)).sha256


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    if len(sha) != 40:
        raise ValueError(f"git rev-parse HEAD returned malformed SHA {sha!r}")
    return sha


def _load_task(cls, *args: str, revision: str):
    task = cls(*args, revision=revision)
    validate_dataset_revision(task.ds, revision)
    return task


def load_pinned_sources() -> tuple[list[RLVRTaskSource], list[RLVRTaskSource], dict[str, int]]:
    """Load the exact ChatCORE sources and exclude every held-out train question."""

    source_specs = (
        (ARC, "ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "ARC-Easy", "arc"),
        (ARC, "ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "ARC-Challenge", "arc"),
        (MMLU, "MMLU", "cais/mmlu", "auxiliary_train", "all", "mmlu"),
        (GSM8K, "GSM8K", "openai/gsm8k", "main", "main", "gsm8k"),
    )
    train_objects = [
        _load_task(cls, train_subset, "train", revision=RLVR_DATASET_REVISIONS[revision_key])
        for cls, _task, _source, train_subset, _validation_subset, revision_key in source_specs
    ]
    validation_objects = [
        _load_task(cls, validation_subset, "test", revision=RLVR_DATASET_REVISIONS[revision_key])
        for cls, _task, _source, _train_subset, validation_subset, revision_key in source_specs
    ]
    heldout = {
        question_key(question) for obj in validation_objects for question in obj.ds["question"]
    }
    # Every shared Task shuffles the raw pinned split with NumPy seed 42.
    # Capture raw-HF row identity before held-out filtering. MMLU's subsequent
    # map flattens its indices table, so derive the documented permutation
    # instead of depending on a private datasets implementation detail.
    train_source_indices = [
        np.random.default_rng(42).permutation(len(task)).tolist() for task in train_objects
    ]
    validation_source_indices = [
        tuple(np.random.default_rng(42).permutation(len(task)).tolist())
        for task in validation_objects
    ]
    retained_source_indices = [
        tuple(
            source_index
            for source_index, question in zip(
                source_indices,
                obj.ds["question"],
                strict=True,
            )
            if question_key(question) not in heldout
        )
        for obj, source_indices in zip(
            train_objects,
            train_source_indices,
            strict=True,
        )
    ]
    before = [len(task) for task in train_objects]
    # This is deliberately the shared ChatCORE leakage implementation. It
    # applies NFKC + casefold + whitespace collapse to all held-out sources,
    # preserves row order, and leaves duplicate multiplicity intact.
    exclude_heldout_questions(train_objects, validation_objects)
    removed = {
        task: old - len(obj)
        for task, old, obj in zip(RLVR_TASKS, before, train_objects, strict=True)
    }

    overlap = [
        (task, index)
        for task, obj in zip(RLVR_TASKS, train_objects, strict=True)
        for index, question in enumerate(obj.ds["question"])
        if question_key(question) in heldout
    ]
    if overlap:
        raise ValueError(f"held-out question exclusion left overlap at {overlap[0]}")

    train: list[RLVRTaskSource] = []
    validation: list[RLVRTaskSource] = []
    for spec, train_obj, train_indices, validation_obj, validation_indices in zip(
        source_specs,
        train_objects,
        retained_source_indices,
        validation_objects,
        validation_source_indices,
        strict=True,
    ):
        _cls, task, source, train_subset, validation_subset, revision_key = spec
        revision = RLVR_DATASET_REVISIONS[revision_key]
        train.append(
            RLVRTaskSource(task, source, train_subset, "train", revision, train_obj, train_indices)
        )
        validation.append(
            RLVRTaskSource(
                task,
                source,
                validation_subset,
                "test",
                revision,
                validation_obj,
                validation_indices,
            )
        )
    return train, validation, removed


def _mapping(cfg: DictConfig, field: str) -> dict[str, int] | None:
    value = cfg.get(field)
    if value is None:
        return None
    raw = OmegaConf.to_container(value, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a mapping or null")
    minimum = 1 if field.endswith("_caps") else 0
    if set(raw) != set(RLVR_TASKS) or any(
        type(item) is not int or item < minimum for item in raw.values()
    ):
        qualifier = "positive" if minimum else "nonnegative"
        raise ValueError(
            f"{field} must contain exactly {qualifier} integer counts for {RLVR_TASKS}"
        )
    return {str(key): int(item) for key, item in raw.items()}


def _counts(sources: list[RLVRTaskSource]) -> dict[str, int]:
    return {source.task: len(source.task_object) for source in sources}


def _require_counts(
    actual: Mapping[str, int], expected: Mapping[str, int] | None, label: str
) -> None:
    if expected is not None and dict(actual) != dict(expected):
        raise ValueError(
            f"{label} counts disagree: expected={dict(expected)}, actual={dict(actual)}"
        )


def _file_info(path: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "rows": rows,
    }


def prepare_data_from_config(cfg: DictConfig) -> Path:
    """Build immutable JSONL and exact extraction inputs before cache production."""

    revisions = OmegaConf.to_container(cfg.revisions, resolve=True)
    if revisions != RLVR_DATASET_REVISIONS:
        raise ValueError("configured dataset revisions disagree with the pinned RLVR contract")
    output = Path(str(cfg.output_dir))
    preparation_path = output / "preparation.json"
    if preparation_path.exists() or (output / "manifest.json").exists():
        raise FileExistsError(f"RLVR preparation or bundle already exists under {output}")

    tokenizer = get_tokenizer()
    contract, extraction_tokenizer = turn_contract_from_config(cfg, tokenizer)
    if contract.model_max_seq_len != int(cfg.model_max_seq_len):
        raise ValueError(
            f"configured model_max_seq_len={cfg.model_max_seq_len} disagrees with "
            f"base checkpoint={contract.model_max_seq_len}"
        )
    model_path = Path(str(cfg.base_checkpoint_path))
    model_sha256 = _sha256_file(model_path)
    if model_sha256 != str(cfg.base_checkpoint_sha256):
        raise ValueError(
            f"base checkpoint SHA-256 mismatch: expected={cfg.base_checkpoint_sha256}, "
            f"actual={model_sha256}"
        )
    meta_path = Path(str(cfg.checkpoint_meta_path))
    stats_path = Path(str(cfg.stats_path))

    train_sources, validation_sources, removed = load_pinned_sources()
    full_train_counts = _counts(train_sources)
    full_validation_counts = _counts(validation_sources)
    _require_counts(full_train_counts, _mapping(cfg, "expected_train_counts"), "retained train")
    _require_counts(
        full_validation_counts,
        _mapping(cfg, "expected_validation_counts"),
        "validation",
    )
    _require_counts(removed, _mapping(cfg, "expected_removed_counts"), "held-out removal")

    train_caps = _mapping(cfg, "train_caps")
    validation_caps = _mapping(cfg, "validation_caps")
    train_rows = prepare_rlvr_rows(
        train_sources,
        split="train",
        tokenizer=tokenizer,
        model_max_seq_len=contract.model_max_seq_len,
        caps=train_caps,
    )
    validation_rows = prepare_rlvr_rows(
        validation_sources,
        split="validation",
        tokenizer=tokenizer,
        model_max_seq_len=contract.model_max_seq_len,
        caps=validation_caps,
    )

    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    install_bytes_once(jsonl_bytes(train_rows), train_path, surface="RLVR train JSONL")
    install_bytes_once(
        jsonl_bytes(validation_rows),
        validation_path,
        surface="RLVR validation JSONL",
    )

    rows = [*train_rows, *validation_rows]
    contexts = dedupe_context_records(
        enumerate_chat_rlvr_contexts(
            tokenizer=tokenizer,
            rows=rows,
        )
    )
    contexts_path = write_context_records(output / "contexts.jsonl", contexts)
    parquet_path = write_context_parquet(output / "context_parquet", contexts)

    contract_sha256 = turn_context_contract_hash(contract)
    identity = {
        **turn_context_identity_fields(contract),
        "feature_selection_hash": contract_sha256,
    }
    cache_relative = (
        Path("context_cache")
        / f"c{contract.model_max_seq_len}"
        / feature_selection_hash_leaf(contract_sha256)
    )
    ordered_ids_sha256 = ordered_row_ids_sha256([row.row_id for row in rows])
    manifest: dict[str, Any] = {
        "format_version": RLVR_BUNDLE_FORMAT_VERSION,
        "bundle_sha256": "",
        "files": {
            "train": _file_info(train_path, rows=len(train_rows)),
            "validation": _file_info(validation_path, rows=len(validation_rows)),
            "contexts": _file_info(contexts_path, rows=len(contexts)),
            "parquet": {
                **_file_info(parquet_path, rows=len(contexts)),
                "path": str(parquet_path.relative_to(output)),
            },
        },
        "dataset_revisions": dict(RLVR_DATASET_REVISIONS),
        "tokenizers": {
            "decoder_sha256": compute_tokenizer_checksum(tokenizer),
            "extraction_sha256": compute_tokenizer_checksum(extraction_tokenizer),
        },
        "render_contract": RLVR_RENDER_CONTRACT,
        "grader_contract": RLVR_GRADER_CONTRACT,
        "context_cache": {
            "role": RLVR_CACHE_ROLE,
            "schema_version": 5,
            "segment_mode": SegmentMode.turn_delimited_fixed_tokens.value,
            "turn_context_contract_sha256": contract_sha256,
            "identity": identity,
            "path": str(cache_relative),
            "ordered_row_ids_sha256": ordered_ids_sha256,
        },
        "base_checkpoint": {
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "metadata_path": str(meta_path),
            "metadata_sha256": _sha256_file(meta_path),
        },
        "statistics": {
            "path": str(stats_path),
            "sha256": _sha256_file(stats_path),
        },
        "preparation": {
            "producer_git_sha": _git_sha(),
            "full_train_counts": full_train_counts,
            "full_validation_counts": full_validation_counts,
            "heldout_removed_counts": removed,
            "train_caps": train_caps,
            "validation_caps": validation_caps,
            "prepared_train_counts": {
                task: sum(row.task == task for row in train_rows) for task in RLVR_TASKS
            },
            "prepared_validation_counts": {
                task: sum(row.task == task for row in validation_rows) for task in RLVR_TASKS
            },
            "model_max_seq_len": contract.model_max_seq_len,
            "max_new_tokens": dict(RLVR_MAX_NEW_TOKENS),
            "heldout_normalization": "NFKC+casefold+whitespace_collapse",
            "qualification_caps_apply_after_heldout_filter": True,
        },
    }
    manifest["bundle_sha256"] = bundle_manifest_sha256(manifest)
    install_bytes_once(
        json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        preparation_path,
        surface="RLVR data preparation manifest",
    )
    logger.info(
        "Prepared RLVR data {}: {} train, {} validation, {} distinct prompt contexts",
        preparation_path,
        len(train_rows),
        len(validation_rows),
        len(contexts),
    )
    return preparation_path


def finalize_from_config(cfg: DictConfig) -> Path:
    """Authenticate the produced schema-5 cache and commit manifest.json last."""

    output = Path(str(cfg.output_dir))
    preparation_path = output / "preparation.json"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"RLVR bundle already committed at {manifest_path}")
    try:
        manifest = json.loads(preparation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read RLVR preparation {preparation_path}: {exc}") from exc
    if manifest.get("bundle_sha256") != bundle_manifest_sha256(manifest):
        raise ValueError("RLVR preparation manifest checksum disagrees")
    data_sha256 = manifest["bundle_sha256"]
    authenticated_jsonl: dict[str, bytes] = {}
    for name, info in manifest["files"].items():
        artifact = output / info["path"]
        if name in {"train", "validation"}:
            payload = artifact.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            authenticated_jsonl[name] = payload
        else:
            digest = _sha256_file(artifact)
        if digest != info["sha256"]:
            raise ValueError(f"RLVR prepared {name} artifact changed before finalization")

    tokenizer = get_tokenizer()
    contract, _extraction_tokenizer = turn_contract_from_config(cfg, tokenizer)
    cache_info = manifest["context_cache"]
    if cache_info["turn_context_contract_sha256"] != turn_context_contract_hash(contract):
        raise ValueError("RLVR preparation turn contract disagrees during finalization")
    cache_dir = output / cache_info["path"]
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest_sha256 = _sha256_file(cache_manifest_path)
    identity = CacheIdentity.from_dict(cache_info["identity"])
    expected_cache_metadata = {
        "rlvr_data_sha256": data_sha256,
        "rlvr_base_model_sha256": manifest["base_checkpoint"]["model_sha256"],
        "rlvr_base_metadata_sha256": manifest["base_checkpoint"]["metadata_sha256"],
    }
    provider = SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_metadata=expected_cache_metadata,
        expected_source_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        expected_turn_contract=contract,
    )
    train = load_rlvr_jsonl(
        output / "train.jsonl",
        expected_split="train",
        payload=authenticated_jsonl["train"],
    )
    validation = load_rlvr_jsonl(
        output / "validation.jsonl",
        expected_split="validation",
        payload=authenticated_jsonl["validation"],
    )
    ordered_ids_sha256 = ordered_row_ids_sha256([row.row_id for row in (*train, *validation)])
    if cache_info["ordered_row_ids_sha256"] != ordered_ids_sha256:
        raise ValueError("RLVR ordered row population changed before finalization")

    manifest["context_cache"] = {
        **cache_info,
        "data_sha256": data_sha256,
        "manifest_sha256": cache_manifest_sha256,
        "shards": [
            {"file": shard["file"], "sha256": shard["sha256"]}
            for shard in provider.manifest["shards"]
        ],
    }
    admit_rlvr_prompt_cache(RLVRBundle(output, train, validation, manifest), provider)
    manifest["bundle_sha256"] = bundle_manifest_sha256(manifest)
    install_bytes_once(
        json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        manifest_path,
        surface="RLVR bundle manifest",
    )
    logger.info("Finalized authenticated RLVR bundle {}", manifest_path)
    return manifest_path


def prepare_from_config(cfg: DictConfig) -> Path:
    """Dispatch the two explicit preparation stages."""

    stage = str(cfg.stage)
    if stage == "prepare":
        return prepare_data_from_config(cfg)
    if stage == "finalize":
        return finalize_from_config(cfg)
    raise ValueError(f"stage must be prepare or finalize, got {stage!r}")


@hydra.main(
    config_path="../../configs/data",
    config_name="prepare_encdec_rlvr",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    prepare_from_config(cfg)


if __name__ == "__main__":
    main()
