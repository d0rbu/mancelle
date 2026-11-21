"""Fit one held-out safe-response percentile contrast in the checkpoint FP vocabulary."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from scripts.eval.build_chat_benchmark_targets import (
    BackgroundOverlapPolicy,
    fit_targets,
    fitting_text,
    normalize_input,
    prepare_background,
    resolve_fitting_environment,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.eval.chat_benchmark_targets import TARGET_FORMAT_VERSION
from steering.eval.chat_eval_result_identity import self_conditioned_code_identity
from steering.eval.dense_feature_targets import load_dense_extraction_stack
from steering.eval.harmfulness_safety_targets import (
    FIT_SHUFFLE_SEED,
    SAFETY_FIT_ORDER,
    SAFETY_FIT_SOURCE,
    SAFETY_RENDER_REDUCTION,
    SAFETY_TARGET_NAME,
    TRAIN_CONFIG,
    TRAIN_SPLIT,
    UNRECORDED_SOURCE_WEIGHT_REVISION,
    ExclusionReason,
    SafetySourceName,
    SafetySubpopulation,
    safety_evaluation_input_keys,
    safety_evaluation_prompt_keys,
    safety_fit_selection,
    validate_safety_preparation,
    write_safety_targets,
)
from steering.eval.harmfulness_safety_targets import (
    load_populations as load_safety_populations,
)
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.feature_io import ExtractionBackend
from steering.immutable_io import install_bytes_once
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import compute_tokenizer_checksum


class TargetArtifact(StrEnum):
    PREPARATION = "preparation.json"
    CORPUS = "corpus.json"
    TARGETS = "targets.safetensors"
    AUDIT_SUMMARY = "fit-audit-summary.json"


def conversation(prompt: str, response: str) -> dict:
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}


def load_target_populations() -> tuple[Any, list[dict], Any, dict]:
    """Adapt the shared pinned source map to the producer's three populations."""
    loaded, provenance = load_safety_populations()
    return (
        loaded[SafetySourceName.TRAINING],
        [
            *loaded[SafetySourceName.EVALUATION_BEAVERTAILS],
            *loaded[SafetySourceName.EVALUATION_WILDGUARD],
        ],
        loaded[SafetySourceName.BACKGROUND],
        provenance,
    )


def prepare_populations(training: Sequence[Mapping], evaluation: Sequence[Mapping], background: Any, *,
                        tokenizer: Any, source_tokenizer: Any, window: int,
                        positive_cap: int, background_count: int) -> tuple[dict, dict, dict]:
    for name, value in (("positive_cap", positive_cap), ("background_count", background_count), ("window", window)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    forbidden = safety_evaluation_input_keys(evaluation)
    positives, accepted, excluded, seen, candidates = [], [], [], set(), set()
    for raw_index in np.random.default_rng(FIT_SHUFFLE_SEED).permutation(len(training)):
        index = int(raw_index)
        row = training[index]
        selection = safety_fit_selection(row)
        prompt, response = row["prompt"], row["response"]
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"training row {index}: prompt must be nonblank text")
        if response is not None and not isinstance(response, str):
            raise ValueError(f"training row {index}: response must be text or absent")
        if not prompt or not prompt.strip():
            excluded.append({"source_id": f"{TRAIN_CONFIG}/{TRAIN_SPLIT}/{index}", "reason": ExclusionReason.EMPTY_PROMPT.value})
            continue
        key = normalize_input(prompt)
        candidates.add(key)
        source_id = f"{TRAIN_CONFIG}/{TRAIN_SPLIT}/{index}"
        # Evaluation exclusion precedes every response/cap selection decision.
        reason = (ExclusionReason.EVALUATION if key in forbidden else
                  selection if isinstance(selection, ExclusionReason) else
                  ExclusionReason.EMPTY if not response or not response.strip() else
                  ExclusionReason.DUPLICATE if key in seen else None)
        if reason is None:
            seen.add(key)
        # After the cap, retain the complete label/overlap audit without tokenizing
        # tens of thousands of conversations that cannot enter the fit.
        if reason is None and len(positives) >= positive_cap:
            reason = ExclusionReason.CAP
        text = None if reason else fitting_text(conversation(prompt, response), tokenizer, source_tokenizer, window)
        if reason is None and text is None:
            reason = ExclusionReason.OVERLENGTH
        if reason:
            excluded.append({"source_id": source_id, "reason": reason.value})
            continue
        assert text is not None  # Missing fit text is an audited overlength exclusion above.
        positives.append(text)
        accepted.append({"source_id": source_id, "input_key": key,
            "prompt_harm_label": row["prompt_harm_label"],
            "response_harm_label": row["response_harm_label"],
            "response_refusal_label": row["response_refusal_label"],
            "subpopulation": selection.value,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    if not positives:
        raise ValueError("no eligible safe response fit documents")
    negatives, negative_ids, background_excluded = prepare_background(background,
        forbidden=tuple(sorted(forbidden | candidates)), tokenizer=tokenizer,
        source_tokenizer=source_tokenizer, window=window, count=background_count,
        overlap_policy=BackgroundOverlapPolicy.EXACT_MESSAGE)
    report = {"source_kind": SAFETY_FIT_SOURCE, "training_rows": len(training),
        "fit_order": SAFETY_FIT_ORDER, "shuffle_seed": FIT_SHUFFLE_SEED,
        "subpopulation_counts": {name.value: sum(row["subpopulation"] == name for row in accepted)
                                 for name in SafetySubpopulation},
        "positive_cap": positive_cap, "background_count": background_count,
        "background_overlap_policy": BackgroundOverlapPolicy.EXACT_MESSAGE.value,
        "evaluation_rows": len(evaluation), "evaluation_input_keys": sorted(forbidden),
        "evaluation_prompts": [row["prompt"] for row in evaluation],
        "accepted": {SAFETY_TARGET_NAME: accepted, SafetySourceName.BACKGROUND.value: negative_ids},
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "excluded": excluded, "background_excluded": background_excluded,
        "background_exclusion_counts": dict(Counter(row["reason"] for row in background_excluded))}
    validate_safety_preparation(report)
    population = {SAFETY_TARGET_NAME: safety_evaluation_prompt_keys(report["evaluation_prompts"], tokenizer)}
    return {"fit": {SAFETY_TARGET_NAME: positives}, "background": negatives}, report, population


def run(cfg: DictConfig) -> Path:
    if type(cfg.dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    for key in ("positive_cap", "background_count", "extraction_batch_size"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    code = self_conditioned_code_identity()
    environment = resolve_fitting_environment(cfg)
    training, evaluation, background, sources = load_target_populations()
    corpus, report, population = prepare_populations(training, evaluation, background,
        tokenizer=environment.tokenizer, source_tokenizer=environment.source_tokenizer, window=environment.window,
        positive_cap=cfg.positive_cap, background_count=cfg.background_count)
    report["sources"] = sources
    manifest = {"format_version": TARGET_FORMAT_VERSION, "fit_contract": environment.fit_contract,
        "preparation": report, "evaluation_prompt_keys": population,
        "fit_provenance": {"checkpoint_metadata_sha256": hashlib.sha256(environment.meta_bytes).hexdigest(),
            "extraction_tokenizer_checksum": compute_tokenizer_checksum(environment.source_tokenizer),
            "sae": environment.sae_config, "source_weight_revision": UNRECORDED_SOURCE_WEIGHT_REVISION,
            "extraction_batch_size": cfg.extraction_batch_size, "source_window": environment.window,
            "render_reduction": SAFETY_RENDER_REDUCTION,
            "corpus_sha256": hashlib.sha256(canonical_turn_payload(corpus)).hexdigest(), "code": code}}
    output = Path(cfg.output_dir)
    for artifact, value in ((TargetArtifact.PREPARATION, manifest), (TargetArtifact.CORPUS, corpus)):
        install_bytes_once(canonical_turn_payload(value), output / artifact, surface="safety target preparation")
    summary = {key: value for key, value in report.items()
               if key not in {"evaluation_prompts", "evaluation_input_keys", "accepted", "excluded", "background_excluded"}}
    summary["accepted_counts"] = {name: len(rows) for name, rows in report["accepted"].items()}
    install_bytes_once(canonical_turn_payload(summary), output / TargetArtifact.AUDIT_SUMMARY,
                       surface="safety target fit audit summary")
    logger.info("Prepared {} safe responses, {} background; exclusions {}", len(corpus["fit"][SAFETY_TARGET_NAME]), len(corpus["background"]), report["exclusion_counts"])
    if self_conditioned_code_identity() != code:
        raise ValueError("source code changed during safety target preparation")
    if cfg.dry_run:
        return output / TargetArtifact.PREPARATION
    config = environment.sae_config
    release = SAERelease(config["release"])
    resolved = resolve_sae(model=config["model"], release=release, activation_type=ActivationType(config["activation_type"]),
        layer=config["layer"], width=SAEWidth(config["width"]), l0=coerce_config_l0(release, config["l0"]))
    model, sae = load_dense_extraction_stack(sae_model=config["model"], sae_layer=config["layer"], resolved_sae=resolved, device=cfg.device)
    if environment.window > model.cfg.n_ctx:
        raise ValueError("training source window exceeds extraction model context")
    model.eval()
    sae.eval()
    if getattr(model, "nnterp_model", None) is not None:
        model.nnterp_model.model.eval()
    targets = fit_targets(corpus["fit"], corpus["background"], model=model, sae=sae,
        sae_model=config["model"], sae_layer=config["layer"], percentile_table=environment.table,
        allowed_ids=environment.allowed_ids, consolidation=environment.features.consolidation,
        missing=environment.missing, feature_vocab=effective_feature_vocab(environment.features.universe, environment.width),
        batch_size=cfg.extraction_batch_size)
    manifest["fit_provenance"].update(backend=ExtractionBackend.nnterp if getattr(model, "nnterp_model", None) is not None else ExtractionBackend.transformerlens,
        device=str(cfg.device), resolved_sae={"release": resolved.release, "sae_id": resolved.sae_id, "hook_name": resolved.hook_name})
    if self_conditioned_code_identity() != code:
        raise ValueError("source code changed during safety target fitting")
    path = output / TargetArtifact.TARGETS
    write_safety_targets(path, target=targets[SAFETY_TARGET_NAME], feature_ids=environment.feature_ids,
        manifest=manifest, tokenizer=environment.tokenizer)
    return path


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="build_harmfulness_safety_target")
def main(cfg: DictConfig) -> None:
    logger.info("Wrote {}", run(cfg))


if __name__ == "__main__":
    main()
