"""Validate produced eval context caches against exact eval token contexts."""

from __future__ import annotations

import csv
import json
from functools import partial
from pathlib import Path

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from nanochat.common import get_base_dir
from nanochat.core_eval import TaskMeta, TaskType
from nanochat.tokenizer import get_tokenizer
from scripts.eval.chat_eval import ALL_CHAT_TASKS
from scripts.eval.context_cache_common import (
    identity_from_config as _identity_from_config,
)
from scripts.eval.context_cache_common import load_context_records, validate_turn_cache_from_config
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.rlvr_data import load_rlvr_jsonl
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import resolve_feature_mask
from steering.eval.context_cache_io import (
    SafetensorsContextFeatureProvider,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    feature_selection_hash_leaf,
)
from steering.eval.context_cache_validation import (
    compare_context_record_sets,
    validate_context_cache_records,
)
from steering.eval.context_enumeration import (
    EvalContextRecord,
    dedupe_context_records,
    enumerate_chat_categorical_contexts,
    enumerate_chat_generative_contexts,
    enumerate_chat_rlvr_contexts,
    enumerate_chat_target_contexts,
    enumerate_harmfulness_pool_contexts,
    enumerate_sample_prompt_contexts,
    enumerate_selected_core_target_contexts,
    enumerate_selected_core_task_contexts,
)
from steering.eval.core_dataset import load_selected_core_task_data
from steering.eval.feature_config import parse_strict_feature_configs
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.removed_features import reject_feature_clustering_config
from steering.segmentation import SegmentMode, segment_mode_from_metadata
from steering.stats_io import verify_statistics_sha256
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

CHAT_TASK_FACTORIES = {
    "HumanEval": HumanEval,
    "MMLU": partial(MMLU, subset="all", split="test"),
    "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "GSM8K": partial(GSM8K, subset="main", split="test"),
    "SpellingBee": partial(SpellingBee, size=256, split="test"),
}


def eval_seeds(cfg: DictConfig) -> list[int]:
    return [int(cfg.base_seed) + i * 1000 for i in range(int(cfg.num_seeds))]


def enumerate_expected_eval_contexts(cfg: DictConfig, tokenizer) -> list[EvalContextRecord]:
    """Enumerate exact context token sequences for the selected eval suites."""

    suites = [suite.strip() for suite in str(cfg.evals).split(",") if suite.strip()]
    mode = segment_mode_from_metadata(cfg, source="eval context enumeration")
    if mode is SegmentMode.turn_delimited_fixed_tokens and (
        not suites or set(suites) - {"chat_target", "harmfulness", "chat_rlvr"}
    ):
        raise ValueError(
            "turn enumeration supports chat_target, harmfulness, and chat_rlvr only; "
            "core_target stays fixed-only"
        )
    records: list[EvalContextRecord] = []
    seeds = eval_seeds(cfg)
    for suite in suites:
        if suite == "core":
            for seed in seeds:
                records.extend(_enumerate_core_contexts(cfg, tokenizer, seed=seed))
        elif suite == "chat":
            records.extend(_enumerate_chat_contexts(cfg, tokenizer))
        elif suite == "harmfulness":
            for seed in seeds:
                records.extend(_enumerate_harmfulness_contexts(cfg, tokenizer, seed=seed))
        elif suite == "samples":
            if "sample_prompts" not in cfg or not cfg.sample_prompts:
                raise ValueError(
                    "eval suite 'samples' requires sample_prompts to be set in the config "
                    "(e.g. sample_prompts: ['The capital of France is', ...]). "
                    "This list must match cfg.sample_prompts in conditional_encdec_train.yaml."
                )
            records.extend(
                enumerate_sample_prompt_contexts(
                    prompts=list(cfg.sample_prompts),
                    tokenizer=tokenizer,
                    max_seq_len=int(cfg.model_max_seq_len),
                )
            )
        elif suite == "chat_target":
            records.extend(_enumerate_chat_target_contexts(cfg, tokenizer))
        elif suite == "core_target":
            for seed in seeds:
                records.extend(_enumerate_core_target_contexts(cfg, tokenizer, seed=seed))
        elif suite == "chat_rlvr":
            paths = cfg.get("rlvr_jsonl_files")
            if not paths or isinstance(paths, str):
                raise ValueError("chat_rlvr enumeration requires rlvr_jsonl_files=[train,validation]")
            rows = []
            for path in paths:
                rows.extend(load_rlvr_jsonl(str(path)))
            records.extend(enumerate_chat_rlvr_contexts(tokenizer=tokenizer, rows=rows))
        else:
            raise ValueError(f"Unsupported eval suite {suite!r}")
    return records


def _enumerate_core_contexts(
    cfg: DictConfig,
    tokenizer,
    *,
    seed: int,
) -> list[EvalContextRecord]:
    max_seq_len = _required_core_model_max_seq_len(cfg)
    eval_bundle_dir = Path(get_base_dir()) / "eval_bundle"
    if not eval_bundle_dir.is_dir():
        raise FileNotFoundError(
            f"CORE eval bundle not found: {eval_bundle_dir}. Run a CORE eval once "
            "or place eval_bundle under the nanochat base directory before "
            "validating the context cache."
        )

    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    metadata_path = eval_bundle_dir / "eval_meta_data.csv"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"CORE eval bundle is incomplete under {eval_bundle_dir}")

    with open(config_path, encoding="utf-8") as f:
        core_config = yaml.safe_load(f)
    _load_core_metadata(metadata_path)

    records: list[EvalContextRecord] = []
    for task in core_config["icl_tasks"]:
        label = task["label"]
        task_meta = TaskMeta(
            task_type=TaskType(task["icl_task_type"]),
            num_fewshot=task["num_fewshot"][0],
            continuation_delimiter=task.get("continuation_delimiter", " "),
        )
        data_path = data_base_path / task["dataset_uri"]
        data = load_selected_core_task_data(
            data_path,
            max_per_task=int(cfg.max_per_task),
        )
        records.extend(
            enumerate_selected_core_task_contexts(
                tokenizer=tokenizer,
                data=data,
                task_meta=task_meta,
                seed=seed,
                task_label=label,
                max_seq_len=max_seq_len,
            )
        )
    return records


def _required_core_model_max_seq_len(cfg: DictConfig) -> int:
    try:
        raw = cfg.model_max_seq_len
    except MissingMandatoryValue as exc:
        raise ValueError(
            "model_max_seq_len is required when enumerating CORE contexts. "
            "Set it to the checkpoint metadata max_seq_len/model.max_seq_len "
            "so cache enumeration skips the same oversized rows as "
            "nanochat.core_eval."
        ) from exc
    if raw is None:
        raise ValueError(
            "model_max_seq_len is required when enumerating CORE contexts. "
            "Set it to the checkpoint metadata max_seq_len/model.max_seq_len "
            "so cache enumeration skips the same oversized rows as "
            "nanochat.core_eval."
        )
    max_seq_len = int(raw)
    if max_seq_len <= 0:
        raise ValueError(f"model_max_seq_len must be positive, got {max_seq_len}")
    return max_seq_len


def _load_core_metadata(path: Path) -> None:
    # Fail early on missing/malformed metadata; progression CORE eval expects it.
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "Eval Task" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing 'Eval Task' column")
        for _row in reader:
            return
    raise ValueError(f"{path}: metadata file has no rows")


def _enumerate_chat_contexts(cfg: DictConfig, tokenizer) -> list[EvalContextRecord]:
    task_names = (
        ALL_CHAT_TASKS
        if cfg.chat_tasks is None
        else [name.strip() for name in str(cfg.chat_tasks).split(",") if name.strip()]
    )
    records: list[EvalContextRecord] = []
    for task_name in task_names:
        task_object = CHAT_TASK_FACTORIES[task_name]()
        if task_object.eval_type == "categorical":
            records.extend(
                enumerate_chat_categorical_contexts(
                    tokenizer=tokenizer,
                    task_object=task_object,
                    batch_size=int(cfg.chat_batch_size),
                    max_problems=cfg.chat_max_problems,
                    task_name=task_name,
                )
            )
        elif task_object.eval_type == "generative":
            records.extend(
                enumerate_chat_generative_contexts(
                    tokenizer=tokenizer,
                    task_object=task_object,
                    max_problems=cfg.chat_max_problems,
                    task_name=task_name,
                )
            )
        else:
            raise ValueError(
                f"Unsupported chat task eval_type={task_object.eval_type!r} for {task_name}"
            )
    return records


def _enumerate_chat_target_contexts(cfg: DictConfig, tokenizer) -> list[EvalContextRecord]:
    """Iterate cfg.chat_tasks and emit chat_target records for each."""
    task_names = (
        ALL_CHAT_TASKS
        if cfg.chat_tasks is None
        else [name.strip() for name in str(cfg.chat_tasks).split(",") if name.strip()]
    )
    records: list[EvalContextRecord] = []
    for task_name in task_names:
        task_object = CHAT_TASK_FACTORIES[task_name]()
        records.extend(
            enumerate_chat_target_contexts(
                tokenizer=tokenizer,
                task_object=task_object,
                max_problems=cfg.chat_max_problems,
                task_name=task_name,
                segment_mode=segment_mode_from_metadata(cfg, source="chat_target enumeration"),
            )
        )
    return records


def _enumerate_core_target_contexts(
    cfg: DictConfig,
    tokenizer,
    *,
    seed: int,
) -> list[EvalContextRecord]:
    """Iterate the eval_bundle core.yaml tasks and emit core_target records per seed."""
    max_seq_len = _required_core_model_max_seq_len(cfg)
    eval_bundle_dir = Path(get_base_dir()) / "eval_bundle"
    if not eval_bundle_dir.is_dir():
        raise FileNotFoundError(
            f"CORE eval bundle not found: {eval_bundle_dir}. Run a CORE eval once "
            "or place eval_bundle under the nanochat base directory before "
            "validating the context cache."
        )
    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    metadata_path = eval_bundle_dir / "eval_meta_data.csv"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"CORE eval bundle is incomplete under {eval_bundle_dir}")

    with open(config_path, encoding="utf-8") as f:
        core_config = yaml.safe_load(f)
    _load_core_metadata(metadata_path)

    records: list[EvalContextRecord] = []
    for task in core_config["icl_tasks"]:
        label = task["label"]
        task_meta = TaskMeta(
            task_type=TaskType(task["icl_task_type"]),
            num_fewshot=task["num_fewshot"][0],
            continuation_delimiter=task.get("continuation_delimiter", " "),
        )
        data_path = data_base_path / task["dataset_uri"]
        data = load_selected_core_task_data(
            data_path,
            max_per_task=int(cfg.max_per_task),
        )
        records.extend(
            enumerate_selected_core_target_contexts(
                tokenizer=tokenizer,
                data=data,
                task_meta=task_meta,
                seed=seed,
                task_label=label,
                max_seq_len=max_seq_len,
            )
        )
    return records


def _enumerate_harmfulness_contexts(
    cfg: DictConfig,
    tokenizer,
    *,
    seed: int,
) -> list[EvalContextRecord]:
    pool = resolve_harmfulness_prompt_pool(
        seed=seed,
        max_harmful=cfg.get("harm_num_harmful"),
        num_benign=cfg.harm_num_benign,
    )
    return enumerate_harmfulness_pool_contexts(
        tokenizer=tokenizer,
        pool=pool,
        segment_mode=segment_mode_from_metadata(cfg, source="harmfulness context enumeration"),
    )


def validate_cache_from_config(cfg: DictConfig) -> dict:
    reject_feature_clustering_config(
        cfg,
        "standalone eval context cache validation",
    )
    if segment_mode_from_metadata(cfg, source="standalone context validation") is SegmentMode.turn_delimited_fixed_tokens:
        tokenizer = get_tokenizer()
        if cfg.get("enumerate_evals", True):
            expected = dedupe_context_records(enumerate_expected_eval_contexts(cfg, tokenizer))
            observed = load_context_records(cfg.contexts_file)
            if expected != observed:
                raise ValueError("turn selected eval enumeration differs from ordered contexts_file")
        return validate_turn_cache_from_config(cfg, tokenizer, cache_root=cfg.cache_dir)
    model_max_seq_len = int(cfg.model_max_seq_len)
    cache_root = cfg.cache_dir
    # Resolve the authoritative cluster contract before tokenizer/model/cache
    # work so a stale or missing static artifact fails without side effects.
    expected_metadata = {"model_max_seq_len": model_max_seq_len}
    expected_feature_vocab = None
    if cfg.get("feature_clusters", None) is not None:
        if cfg.n_sae_features is None:
            raise ValueError(
                "clustered context-cache validation requires n_sae_features"
            )
        n_sae_features = int(cfg.n_sae_features)
        feature_mask = resolve_feature_mask(cfg, n_sae_features)
        resolved_clusters = resolve_feature_clusters(cfg, n_sae_features)
        assert resolved_clusters is not None
        feature_clusters = FeatureClusterConsolidation.build(
            resolved_clusters,
            feature_mask,
        )
        expected_feature_vocab = effective_feature_vocab(
            feature_clusters,
            n_sae_features,
        )
        stats_source_sha256 = cfg.get("stats_source_sha256", None)
        if stats_source_sha256 is None:
            raise ValueError(
                "clustered context-cache validation requires stats_source_sha256"
            )
        stats_path = cfg.get("stats_path", None)
        if stats_path is not None:
            verify_statistics_sha256(
                Path(str(stats_path)),
                str(stats_source_sha256),
                surface="clustered context-cache validation",
            )
        expected_metadata.update(
            clustered_context_cache_metadata(
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                feature_vocab=expected_feature_vocab,
                stats_source_sha256=str(stats_source_sha256),
            )
        )

    identity = _identity_from_config(cfg)
    cache_dir = context_length_cache_dir(
        cache_root,
        model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
    )
    tokenizer = get_tokenizer()
    enumerated_records: list[EvalContextRecord] = []
    file_records: list[EvalContextRecord] = []

    if bool(cfg.enumerate_evals):
        enumerated_records = enumerate_expected_eval_contexts(cfg, tokenizer)
    contexts_file = cfg.get("contexts_file")
    if contexts_file is not None:
        file_records = load_context_records(contexts_file)

    if enumerated_records and file_records:
        compare_context_record_sets(
            enumerated_records,
            file_records,
            expected_label="selected evals",
            observed_label=str(contexts_file),
        )

    records = enumerated_records or file_records
    if not records:
        raise ValueError("No contexts to validate; set enumerate_evals=true or contexts_file")
    records = dedupe_context_records(records)

    provider = SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_n_sae_features=(
            int(cfg.n_sae_features) if cfg.n_sae_features is not None else None
        ),
        expected_num_buckets=(int(cfg.num_buckets) if cfg.num_buckets is not None else None),
        expected_metadata=expected_metadata,
        expected_feature_vocab=expected_feature_vocab,
    )
    raw_target_feature_configs = (
        None
        if cfg.target_feature_configs is None
        else OmegaConf.to_container(cfg.target_feature_configs, resolve=True)
    )
    target_feature_configs = parse_strict_feature_configs(raw_target_feature_configs)
    summary = validate_context_cache_records(
        records=records,
        provider=provider,
        tokenizer=tokenizer,
        target_feature_configs=target_feature_configs,
        require_nonempty_context_features=bool(cfg.require_nonempty_context_features),
        require_nonempty_context_segments=bool(cfg.require_nonempty_context_segments),
        fail_on_extra_cached_contexts=bool(cfg.fail_on_extra_cached_contexts),
        target_conditioning_mode=str(cfg.get("target_conditioning_mode", "current")),
    )

    summary_dict = summary.to_dict()
    summary_file = cfg.get("summary_file")
    if summary_file:
        path = Path(summary_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary_dict, indent=2, sort_keys=True) + "\n")
        logger.info(f"Wrote context cache validation summary: {path}")

    logger.info(
        "Validated eval context cache: "
        f"{summary.unique_contexts} unique eval contexts, "
        f"{summary.total_context_segments} context segments, "
        f"{summary.total_context_features} context features"
    )
    return summary_dict


@hydra.main(
    config_path="../../configs/eval",
    config_name="validate_eval_context_cache",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    validate_cache_from_config(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
