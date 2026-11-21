"""Prepare isolated ChatCORE training corpora and fit six shared-background targets."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import torch as t
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.execution import execute_code
from nanochat.tokenizer import get_tokenizer
from scripts.eval.chat_eval import ALL_CHAT_TASKS
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.missing_percentile import (
    MissingPercentileResolver,
    check_percentile_coverage,
    enforce_percentile_coverage,
)
from steering.conditional.percentile_table import build_percentile_table
from steering.eval.axbench_base.concept_feature_mapping import _batched_dense_sae_max_pool
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.chat_benchmark_targets import (
    TARGET_FORMAT_VERSION,
    BackgroundOverlapPolicy,
    benchmark_fit_contract,
    normalize_input,
    write_benchmark_targets,
)
from steering.eval.chat_eval_result_identity import self_conditioned_code_identity
from steering.eval.chat_generation import render_chat_completion
from steering.eval.chat_self_conditioned_cache import resolve_fixed_chat_cache_contract
from steering.eval.context_enumeration import context_text_for_extraction
from steering.eval.dataset_revision import validate_dataset_revision
from steering.eval.dense_feature_targets import (
    dense_diffmean_from_means,
    dense_mean_percentiles,
    load_dense_extraction_stack,
)
from steering.eval.target_conditioning import token_context_key
from steering.eval.turn_context_contract import (
    canonical_turn_payload,
    resolve_turn_context_contract,
    turn_extraction_tokenizer_id,
)
from steering.immutable_io import install_bytes_once
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from steering.sft.segment_mode_identity import resolve_checkpoint_sft_segment_mode
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SpellingBee, resolve_word_list_path


@dataclass(frozen=True)
class TrainingExample:
    source_id: str
    input_key: str
    answer_key: str
    conversation: dict


def calibration_group(task: str, key: str, seed: int) -> bool:
    digest = hashlib.sha256(canonical_turn_payload({"version": 1, "task": task, "key": key, "seed": seed})).digest()
    return int.from_bytes(digest, "big") % 5 == 0


def task_input_keys(task: Any, index: int) -> tuple[str, ...]:
    """Read only raw input fields; never reference answers, solutions or tests."""
    if isinstance(task, SpellingBee):
        return ("word:" + normalize_input(task._example_rng_and_word(index)[1]),)
    row = task.ds[index]
    if isinstance(task, HumanEval):
        tree = ast.parse(row["prompt"])
        descriptions = [ast.get_docstring(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return tuple("code:" + normalize_input(text) for text in [row["prompt"], *descriptions] if text)
    return ("question:" + normalize_input(row["question"]),)


def training_examples(task: Any, source: str) -> list[TrainingExample]:
    examples = []
    for index in range(len(task)):
        conversation = task[index]
        answer = conversation["messages"][-1]["content"]
        if isinstance(task, MMLU):
            row = task.ds[index]
            answer = normalize_input(row["choices"][row["answer"]])
        elif isinstance(task, ARC):
            row = task.ds[index]
            answer = normalize_input(row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])])
        else:
            answer = canonical_turn_payload({"answer": answer}).decode()
        examples.append(TrainingExample(f"{source}/{index}", task_input_keys(task, index)[0], answer, conversation))
    return examples


def mbpp_examples(rows: Any, split: str, exclusions: dict[str, str]) -> tuple[list[TrainingExample], list[dict]]:
    examples, excluded = [], []
    for row in rows:
        source_id = f"mbpp/sanitized/{split}/{row['task_id']}"
        if str(row["task_id"]) in exclusions:
            excluded.append({"source_id": source_id, "input_key": "code:" + normalize_input(row["prompt"]),
                             "reason": exclusions[str(row["task_id"])]})
            continue
        # Test imports and assertions qualify the reference but never enter its text.
        program = "\n".join([*row["test_imports"], row["code"], *row["test_list"]])
        outcome = execute_code(program)
        if not outcome.success:
            raise ValueError(f"MBPP reference failed qualification: {source_id}: {outcome}")
        conversation = {"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["code"]}]}
        examples.append(TrainingExample(source_id, "code:" + normalize_input(row["prompt"]), row["code"], conversation))
    return examples, excluded


def load_sources(revisions: dict[str, str], coding_exclusions: dict[str, str]) -> tuple[dict, dict, dict, Any, list[dict]]:
    """Six concrete adapters reuse native rendering and leave runtime dispatch alone."""
    def task(cls, *args, revision):
        loaded = cls(*args, revision=revision)
        validate_dataset_revision(loaded.ds, revision)
        return loaded

    evaluation = {
        "ARC-Easy": task(ARC, "ARC-Easy", "test", revision=revisions["arc"]),
        "ARC-Challenge": task(ARC, "ARC-Challenge", "test", revision=revisions["arc"]),
        "MMLU": task(MMLU, "all", "test", revision=revisions["mmlu"]),
        "GSM8K": task(GSM8K, "main", "test", revision=revisions["gsm8k"]),
        "HumanEval": task(HumanEval, revision=revisions["humaneval"]),
        "SpellingBee": SpellingBee(size=256, split="test"),
    }
    fit, calibration = {}, {}
    for name in ("ARC-Easy", "ARC-Challenge"):
        fit[name] = training_examples(task(ARC, name, "train", revision=revisions["arc"]), f"{name}/train")
        calibration[name] = training_examples(task(ARC, name, "validation", revision=revisions["arc"]), f"{name}/validation")
    fit["MMLU"] = training_examples(task(MMLU, "all", "dev", revision=revisions["mmlu"]), "MMLU/dev")
    calibration["MMLU"] = training_examples(task(MMLU, "all", "validation", revision=revisions["mmlu"]), "MMLU/validation")
    fit["GSM8K"] = training_examples(task(GSM8K, "main", "train", revision=revisions["gsm8k"]), "GSM8K/train")
    fit["SpellingBee"] = training_examples(SpellingBee(size=10_000, split="train"), "SpellingBee/train")
    calibration["GSM8K"], calibration["SpellingBee"] = [], []
    excluded = []
    for split, destination in (("train", fit), ("validation", calibration)):
        rows = load_dataset("google-research-datasets/mbpp", "sanitized", split=split, revision=revisions["mbpp"])
        validate_dataset_revision(rows, revisions["mbpp"])
        destination["HumanEval"], removed = mbpp_examples(rows, split, coding_exclusions)
        excluded.extend(removed)
    return evaluation, fit, calibration, task(SmolTalk, "train", revision=revisions["background"]), excluded


def fitting_text(conversation: dict, tokenizer: Any, source_tokenizer: Any, window: int) -> str | None:
    messages = conversation["messages"]
    if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["content"]:
        raise ValueError("fitting requires a complete nonempty reference response")
    ids, _ = tokenizer.render_conversation(conversation, max_tokens=2**31)
    if not ids or ids[0] != tokenizer.get_bos_token_id() or ids[-1] != tokenizer.encode_special("<assistant_end>"):
        raise ValueError("fitting render must preserve decoder BOS and complete assistant terminator")
    text = context_text_for_extraction(tokenizer, ids)
    content = source_tokenizer.encode(text, add_special_tokens=False)
    if not content:
        raise ValueError("fitting text has no source content tokens")
    return text if len(content) + 1 <= window else None


def prepare_populations(
    evaluation: dict, fit: dict, calibration: dict, background: Any, *, tokenizer: Any,
    source_tokenizer: Any, window: int, positive_cap: int, background_count: int, seed: int,
    excluded_coding_inputs: tuple[str, ...] = (),
) -> tuple[dict[str, list[str]], list[str], dict, dict]:
    """Freeze full test exclusion before grouping, length checks and deterministic caps."""
    fit, calibration = {k: list(v) for k, v in fit.items()}, {k: list(v) for k, v in calibration.items()}
    forbidden, population = set(), {}
    for name in ALL_CHAT_TASKS:
        task = evaluation[name]
        population[name] = []
        for index in range(len(task)):
            forbidden.update(task_input_keys(task, index))
            population[name].append(token_context_key(render_chat_completion(tokenizer, task[index]).token_ids))
    # Exclude background duplicates of ALL candidate inputs, independent of caps.
    background_forbidden = forbidden | set(excluded_coding_inputs) | {row.input_key for groups in (fit, calibration) for rows in groups.values() for row in rows}
    for name in ("GSM8K", "SpellingBee"):
        calibration[name] = [row for row in fit[name] if calibration_group(name, row.input_key, seed)]
        fit[name] = [row for row in fit[name] if not calibration_group(name, row.input_key, seed)]
    reserved = {row.input_key for rows in calibration.values() for row in rows}
    report: dict[str, Any] = {"accepted": {}, "excluded": [], "positive_cap": positive_cap,
                              "background_count": background_count, "partition_seed": seed, "partition_version": 1}
    texts, calibration_texts = {}, {}
    for split, groups, destination in (("calibration", calibration, calibration_texts), ("fit", fit, texts)):
        for name in ALL_CHAT_TASKS:
            rows, kept, seen = groups[name], [], set()
            answers: dict[str, set[str]] = {}
            for row in rows:
                answers.setdefault(canonical_turn_payload({"messages": row.conversation["messages"][:-1]}).decode(), set()).add(row.answer_key)
            for row in rows:
                reason = ("evaluation_overlap" if row.input_key in forbidden else
                          "calibration_overlap" if split == "fit" and row.input_key in reserved else
                          "conflicting_references" if len(answers[canonical_turn_payload({"messages": row.conversation["messages"][:-1]}).decode()]) > 1 else
                          "duplicate_input" if row.input_key in seen else None)
                seen.add(row.input_key)
                text = None if reason else fitting_text(row.conversation, tokenizer, source_tokenizer, window)
                if reason is None and text is None:
                    reason = "source_overlength"
                if reason is None and split == "fit" and len(kept) >= positive_cap:
                    reason = "positive_cap"
                if reason:
                    report["excluded"].append({"source_id": row.source_id, "reason": reason})
                    continue
                kept.append((row, text))
            if not kept:
                raise ValueError(f"no eligible {split} documents for {name}")
            destination[name] = [text for _, text in kept]
            report["accepted"][f"{name}/{split}"] = [{"source_id": row.source_id, "input_key": row.input_key,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest()} for row, text in kept]
    negatives, negative_ids, removed = prepare_background(background,
        forbidden=tuple(key.split(":", 1)[1] for key in sorted(background_forbidden) if not key.startswith("word:")),
        tokenizer=tokenizer, source_tokenizer=source_tokenizer, window=window, count=background_count)
    report["excluded"].extend(removed)
    report["accepted"]["background"] = negative_ids
    return texts, negatives, {"report": report, "calibration_texts": calibration_texts}, population


def prepare_background(background: Any, *, forbidden: tuple[str, ...], tokenizer: Any,
                       source_tokenizer: Any, window: int, count: int,
                       overlap_policy: BackgroundOverlapPolicy = BackgroundOverlapPolicy.EMBEDDED,
                       ) -> tuple[list[str], list[dict], list[dict]]:
    """Freeze the same SmolTalk first-exchange background for dense target builders."""
    embedded, background_count, excluded = forbidden, count, []
    forbidden_set = set(forbidden)
    negatives, negative_ids, seen_background = [], [], set()
    for index in range(len(background)):
        conversation = background[index]
        messages = conversation["messages"]
        offset = int(messages[0]["role"] == "system")
        first = {"messages": messages[:offset + 2]}
        body = normalize_input("\n".join(message["content"] for message in first["messages"]))
        prompt = normalize_input(messages[offset]["content"])
        match overlap_policy:
            case BackgroundOverlapPolicy.EMBEDDED:
                overlaps = any(key in body for key in embedded)
            case BackgroundOverlapPolicy.EXACT_MESSAGE:
                overlaps = any(normalize_input(message["content"]) in forbidden_set for message in first["messages"])
            case _:
                raise ValueError(f"unknown background overlap policy: {overlap_policy!r}")
        reason = ("duplicate_input" if prompt in seen_background else
                  "source_overlap" if overlaps else None)
        seen_background.add(prompt)
        text = None if reason else fitting_text(first, tokenizer, source_tokenizer, window)
        if reason is None and text is None:
            reason = "source_overlength"
        if reason:
            excluded.append({"source_id": f"background/train/{index}", "reason": reason})
            continue
        assert text is not None
        negatives.append(text)
        negative_ids.append({"source_id": f"background/train/{index}", "input_key": prompt,
                             "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(negatives) == background_count:
            break
    if len(negatives) != background_count:
        raise ValueError(f"insufficient eligible background: {len(negatives)} < {background_count}")
    return negatives, negative_ids, excluded


@t.inference_mode()
def fit_targets(positive_texts: dict, background: list[str], *, model: Any, sae: Any,
                sae_model: str, sae_layer: int, percentile_table: t.Tensor, allowed_ids: t.Tensor,
                consolidation: Any, missing: MissingPercentileResolver, feature_vocab: Any,
                batch_size: int) -> dict[str, t.Tensor]:
    def mean(texts):
        activations = _batched_dense_sae_max_pool(list(texts), model, sae, sae_layer=sae_layer,
            tokenizer_id=turn_extraction_tokenizer_id(sae_model), batch_size=batch_size)
        return dense_mean_percentiles(activations, percentile_table=percentile_table,
            allowed_ids=allowed_ids, consolidation=consolidation, missing=missing)
    negative_mean = mean(background)
    targets = {}
    for name, texts in positive_texts.items():
        logger.info("Fitting {} from {} positives and {} shared background documents", name, len(texts), len(background))
        targets[name] = dense_diffmean_from_means(mean(texts), negative_mean,
            feature_vocab=feature_vocab, label=name)
    return targets


@dataclass(frozen=True)
class DenseTargetEnvironment:
    meta_bytes: bytes
    tokenizer: Any
    source_tokenizer: Any
    features: Any
    width: int
    sae_config: dict
    table: t.Tensor
    missing: MissingPercentileResolver
    fit_contract: dict
    feature_ids: t.Tensor
    window: int
    allowed_ids: t.Tensor


def resolve_fitting_environment(cfg: DictConfig) -> DenseTargetEnvironment:
    """Resolve authenticated checkpoint/source geometry once for dense builders."""
    meta_path = Path(cfg.checkpoint_meta_path)
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes)
    tokenizer = get_tokenizer()
    mode = resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=str(meta_path.parent))
    lineage = OnlineSAELineage.from_checkpoint_meta(meta) if mode is SegmentMode.turn_delimited_fixed_tokens else None
    width = PositiveInt.parse(lineage.n_sae_features if lineage is not None else
        SAE_WIDTH_TO_NUM_FEATURES[SAEWidth(meta["user_config"]["sae"]["width"])])
    features = resolve_checkpoint_feature_contract(meta, n_sae_features=width,
        surface="ChatCORE benchmark preparation", source_path=str(meta_path))
    turn, fixed = None, None
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        assert lineage is not None
        sae_config = {name: getattr(lineage, name) for name in ("model", "release", "activation_type", "layer", "width", "l0")}
        source_tokenizer = get_extraction_tokenizer(turn_extraction_tokenizer_id(lineage.model))
        turn = resolve_turn_context_contract(meta, tokenizer=tokenizer, extraction_tokenizer=source_tokenizer,
            statistics_path=cfg.stats_path, checkpoint_contract=features)
        statistics = turn.statistics
        context = turn.build_selection_context()
        table, missing = context.fp_percentile_table, context.missing_percentile
        assert table is not None  # The turn contract builds this from its authenticated statistics.
    else:
        fixed = resolve_fixed_chat_cache_contract(meta, tokenizer, features, statistics_path=cfg.stats_path)
        sae_config = fixed.source["sae"]
        source_tokenizer = get_extraction_tokenizer(turn_extraction_tokenizer_id(sae_config["model"]))
        statistics, missing = fixed.statistics, MissingPercentileResolver.error()
        table = build_percentile_table(statistics.per_feature_stats, width)
    fit_contract, feature_ids = benchmark_fit_contract(feature_contract=features, n_sae_features=width,
        tokenizer=tokenizer, turn_contract=turn, fixed_contract=fixed)
    window = statistics.metadata.context_window
    if type(window) is not int or window <= 1:
        raise ValueError("fitting requires a positive BOS-inclusive source context window")
    consolidation = features.consolidation
    allowed = (consolidation.allowed_universe if consolidation is not None else
               t.ones(width, dtype=t.bool) if features.feature_mask is None else features.feature_mask.allowed_universe_mask(width))
    allowed_ids = t.nonzero(allowed, as_tuple=False).squeeze(1)
    if turn is not None:
        coverage = check_percentile_coverage(percentile_table=table, required_ids=allowed_ids,
            table_name="checkpoint statistics", required_set_name="benchmark mask-allowed feature IDs")
        enforce_percentile_coverage([coverage], missing.policy)
    return DenseTargetEnvironment(meta_bytes, tokenizer, source_tokenizer, features, width,
        sae_config, table, missing, fit_contract, feature_ids, window, allowed_ids)


def run(cfg: DictConfig) -> Path:
    for key in ("positive_cap", "background_count", "extraction_batch_size"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.partition_seed) is not int or type(cfg.dry_run) is not bool:
        raise ValueError("partition_seed must be an integer and dry_run a boolean")
    revisions = OmegaConf.to_container(cfg.revisions, resolve=True)
    if not isinstance(revisions, dict) or set(revisions) != {"mmlu", "arc", "gsm8k", "humaneval", "mbpp", "background"}:
        raise ValueError("builder requires all six dataset revision pins")
    if any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{40}", value) is None for value in revisions.values()):
        raise ValueError("dataset revisions must be immutable commit SHAs")
    revisions = cast(dict[str, str], revisions)
    exclusions = OmegaConf.to_container(cfg.coding_exclusions, resolve=True)
    if not isinstance(exclusions, dict) or any(not isinstance(key, str) or not key.isdecimal() or not isinstance(reason, str) or not reason.strip()
           for key, reason in exclusions.items()):
        raise ValueError("coding_exclusions must map task ID strings to nonblank review reasons")
    exclusions = cast(dict[str, str], exclusions)
    code_identity = self_conditioned_code_identity()
    output = Path(cfg.output_dir)
    environment = resolve_fitting_environment(cfg)
    meta_bytes, tokenizer, source_tokenizer = environment.meta_bytes, environment.tokenizer, environment.source_tokenizer
    features, width, sae_config = environment.features, environment.width, environment.sae_config
    table, missing = environment.table, environment.missing
    fit_contract, feature_ids = environment.fit_contract, environment.feature_ids
    window, allowed_ids, consolidation = environment.window, environment.allowed_ids, features.consolidation
    word_list_sha256 = hashlib.sha256(resolve_word_list_path().read_bytes()).hexdigest()
    evaluation, fit, calibration, background, excluded = load_sources(revisions, exclusions)
    positives, negatives, prepared, population = prepare_populations(evaluation, fit, calibration, background,
        tokenizer=tokenizer, source_tokenizer=source_tokenizer, window=window,
        positive_cap=cfg.positive_cap, background_count=cfg.background_count, seed=cfg.partition_seed,
        excluded_coding_inputs=tuple(row["input_key"] for row in excluded))
    if hashlib.sha256(resolve_word_list_path().read_bytes()).hexdigest() != word_list_sha256:
        raise ValueError("spelling word list changed during benchmark preparation")
    report = prepared["report"]
    report.update(dataset_revisions=revisions, coding_exclusions=exclusions, coding_removed=excluded,
        word_list_sha256=word_list_sha256)
    manifest = {"format_version": TARGET_FORMAT_VERSION, "fit_contract": fit_contract,
        "preparation": report, "evaluation_prompt_keys": population,
        "fit_provenance": {"checkpoint_metadata_sha256": hashlib.sha256(meta_bytes).hexdigest(),
            "extraction_tokenizer_checksum": compute_tokenizer_checksum(source_tokenizer),
            "sae": sae_config, "source_weight_revision": "unrecorded_by_existing_loader",
            "extraction_batch_size": cfg.extraction_batch_size, "source_window": window,
            "render_reduction": "native_complete_chat_bos_excluded_document_percentile_diffmean_v1",
            "code": code_identity}}
    if self_conditioned_code_identity() != code_identity:
        raise ValueError("source code changed during benchmark preparation")
    install_bytes_once(canonical_turn_payload(manifest), output / "preparation.json", surface="benchmark preparation")
    corpus = {"fit": positives, "background": negatives, "calibration": prepared["calibration_texts"]}
    install_bytes_once(canonical_turn_payload(corpus), output / "corpus.json", surface="benchmark source corpus")
    logger.info("Prepared target corpora: {}; shared background {}", {k: len(v) for k, v in positives.items()}, len(negatives))
    if cfg.dry_run:
        return output / "preparation.json"
    release = SAERelease(sae_config["release"])
    resolved = resolve_sae(model=sae_config["model"], release=release,
        activation_type=ActivationType(sae_config["activation_type"]), layer=sae_config["layer"],
        width=SAEWidth(sae_config["width"]), l0=coerce_config_l0(release, sae_config["l0"]))
    model, sae = load_dense_extraction_stack(sae_model=sae_config["model"], sae_layer=sae_config["layer"],
        resolved_sae=resolved, device=cfg.device)
    if window > model.cfg.n_ctx:
        raise ValueError("training source window exceeds extraction model context")
    model.eval()
    sae.eval()
    if getattr(model, "nnterp_model", None) is not None:
        model.nnterp_model.model.eval()
    targets = fit_targets(positives, negatives, model=model, sae=sae, sae_model=sae_config["model"],
        sae_layer=sae_config["layer"], percentile_table=table, allowed_ids=allowed_ids,
        consolidation=consolidation, missing=missing, feature_vocab=effective_feature_vocab(features.universe, width),
        batch_size=cfg.extraction_batch_size)
    manifest["fit_provenance"].update(backend="nnterp" if getattr(model, "nnterp_model", None) is not None else "transformerlens",
        sae_dtype=str(next(sae.parameters()).dtype), device=str(cfg.device), resolved_sae={"release": resolved.release, "sae_id": resolved.sae_id, "hook_name": resolved.hook_name})
    if self_conditioned_code_identity() != code_identity:
        raise ValueError("source code changed during benchmark fitting")
    path = output / "targets.safetensors"
    write_benchmark_targets(path, targets=targets, feature_ids=feature_ids, manifest=manifest, task_names=ALL_CHAT_TASKS)
    return path


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="build_chat_benchmark_targets")
def main(cfg: DictConfig) -> None:
    logger.info("Wrote {}", run(cfg))


if __name__ == "__main__":
    main()
