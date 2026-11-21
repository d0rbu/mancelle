"""Hydra integration and common target admission for natural-chat sources."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from nanochat.tokenizer import RustBPETokenizer
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_negative.admission import load_conditioning_admission
from steering.eval.axbench_negative.api import (
    ChatApiClient,
    ConversationQualityJudge,
    SamplingConfig,
    strict_object,
)
from steering.eval.axbench_negative.artifacts import NegativeSourceBank
from steering.eval.axbench_negative.producer import ProducerConfig, build_negative_source
from steering.eval.axbench_negative.qc import ChatQcError, ChatScorer, build_chat_scorer
from steering.eval.axbench_negative.source import (
    SOURCE_FORMAT,
    FeatureTarget,
    SourceCandidate,
    load_source_candidates,
)
from steering.eval.axbench_upstream.data import load_metadata
from steering.eval.axbench_upstream.native_baselines import resolve_native_model_contract
from steering.eval.axbench_upstream.native_decoder import validate_native_checkpoint_metadata
from steering.eval.axbench_upstream.roundtrip import parse_sae_link
from steering.eval.chat_generation import render_messages_for_chat_completion
from steering.immutable_io import stream_hash_file_descriptor_bound
from steering.sae_resolver import (
    SAERelease,
    SAEWidth,
    coerce_activation_type,
    coerce_config_l0,
    resolve_neuronpedia_ids,
)


def _bound_hash(path: Path, expected: str | None = None) -> str:
    observed = stream_hash_file_descriptor_bound(path.resolve(strict=True)).sha256
    if expected is not None and observed != expected:
        raise ValueError(f"content hash differs for {path}")
    return observed


def load_render_tokenizer(path: Path, expected_sha256: str | None = None) -> Any:
    """Load the exact native tokenizer file after checking its content identity."""
    path = path.resolve(strict=True)
    if path.name != "tokenizer.pkl":
        raise ValueError("render tokenizer must be a native tokenizer.pkl")
    before = _bound_hash(path, expected_sha256)
    tokenizer = RustBPETokenizer.from_directory(str(path.parent))
    _bound_hash(path, before)
    return tokenizer


def create_chat_api(config: Mapping[str, Any], cache: SynthResponseCache) -> ChatApiClient:
    """Construct the shared transport with explicit, bounded request settings."""
    required = {"model", "temperature", "top_p", "max_tokens", "seed",
                "requests_per_minute", "max_retries", "transport_retry_budget_seconds"}
    if set(config) not in (required, required | {"provider"}):
        raise ValueError(f"API config requires {sorted(required)} and optional provider")
    provider = config.get("provider", "openrouter")
    if provider not in ("openrouter", "openai"):
        raise ValueError("API provider must be openrouter or openai")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError("API model must be explicit")
    for key in ("requests_per_minute", "max_retries"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    budget = config["transport_retry_budget_seconds"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("transport_retry_budget_seconds must be finite and nonnegative")
    sampling = SamplingConfig(**{key: config[key] for key in ("temperature", "top_p", "max_tokens", "seed")})
    client = OpenRouterLLMClient(
        LLMClientConfig(provider=provider, model=config["model"], requests_per_minute=config["requests_per_minute"],
                        transport_retry_budget_seconds=float(budget)),
        max_retries=config["max_retries"], response_cache=cache,
    )
    return ChatApiClient(client, sampling)


def load_feature_targets(
    metadata_path: Path, expected_sha256: str, concept_ids: Sequence[int],
    extractor: Mapping[str, Any],
) -> tuple[list[FeatureTarget], str]:
    """Bind canonical AxBench concept descriptions to their actual SAE namespace."""
    _bound_hash(metadata_path, expected_sha256)
    if not concept_ids or any(type(value) is not int or value < 0 for value in concept_ids):
        raise ValueError("concept_ids must contain nonnegative integers")
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("concept_ids must be unique")
    release = SAERelease(extractor["sae_release"])
    l0 = coerce_config_l0(release, extractor["sae_l0"])
    if release is SAERelease.gemmascope and l0 is not None:
        raise ValueError("canonical AxBench metadata requires the canonical GemmaScope SAE")
    model_id, source_id = resolve_neuronpedia_ids(
        model=extractor["model"], release=release,
        activation_type=coerce_activation_type(extractor["activation_type"]),
        layer=extractor["layer"], width=SAEWidth(extractor["sae_width"]), l0=l0,
    )
    prefix = f"https://www.neuronpedia.org/{model_id}/{source_id}"
    metadata = load_metadata(metadata_path)
    targets = []
    for concept_id in concept_ids:
        if concept_id >= len(metadata):
            raise ValueError(f"concept_id outside metadata: {concept_id}")
        row = metadata[concept_id]
        ref = parse_sae_link(row["ref"])
        if (ref.model_id, ref.source) != (model_id, source_id):
            raise ValueError("concept metadata and screening SAE namespace differ")
        targets.append(FeatureTarget(ref.feature_index, (ref.feature_index,), row["concept"]))
    if len({target.feature_id for target in targets}) != len(targets):
        raise ValueError("selected metadata repeats an SAE feature")
    _bound_hash(metadata_path, expected_sha256)
    return targets, prefix


def admit_target_contexts(
    candidates: Sequence[SourceCandidate], configurations: Sequence[Mapping[str, Any]],
) -> tuple[list[SourceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze the intersection of complete prompt budgets before any provider call."""
    if not configurations:
        raise ValueError("at least one target context contract is required")
    required = {"family", "meta_path", "tokenizer_path", "max_new_tokens",
                "expected_meta_sha256", "expected_model_sha256", "expected_tokenizer_sha256",
                "feature_statistics_path", "pretrain_meta_path"}
    contracts: list[dict[str, Any]] = []
    tokenizers = []
    conditioners = []
    for config in configurations:
        if set(config) != required or config["family"] not in {"native", "encdec"}:
            raise ValueError("invalid target admission configuration")
        if config["family"] in {contract["family"] for contract in contracts}:
            raise ValueError("target families must be unique")
        budget = config["max_new_tokens"]
        if type(budget) is not int or budget <= 0:
            raise ValueError("target max_new_tokens must be a positive integer")
        meta_path = Path(config["meta_path"]).resolve(strict=True)
        meta_hash = _bound_hash(meta_path, config["expected_meta_sha256"])
        meta = strict_object(meta_path.read_text(encoding="utf-8"))
        sequence_len = meta.get("model_config", {}).get("sequence_len")
        if type(sequence_len) is not int or sequence_len <= 0 or budget >= sequence_len:
            raise ValueError("target metadata has an invalid context/generation budget")
        if (config["family"] == "encdec") != ("encoder_config" in meta):
            raise ValueError("target family differs from checkpoint metadata")
        if not meta_path.name.startswith("meta_") or meta_path.suffix != ".json":
            raise ValueError("target metadata must name an explicit meta_<step>.json")
        if config["family"] == "native":
            validate_native_checkpoint_metadata(meta, int(meta_path.stem[5:]))
            resolve_native_model_contract(meta["model_config"])
        model_path = meta_path.with_name("model_" + meta_path.stem[5:] + ".pt")
        model_hash = _bound_hash(model_path, config["expected_model_sha256"])
        tokenizer_path = Path(config["tokenizer_path"])
        tokenizer_hash = _bound_hash(tokenizer_path, config["expected_tokenizer_sha256"])
        tokenizers.append(load_render_tokenizer(tokenizer_path, tokenizer_hash))
        conditioning = None
        if config["family"] == "encdec":
            if not isinstance(config["feature_statistics_path"], str) or not config["feature_statistics_path"]:
                raise ValueError("encoder-decoder admission requires feature_statistics_path")
            conditioning = load_conditioning_admission(
                meta_path, Path(config["feature_statistics_path"]),
                None if config["pretrain_meta_path"] is None else Path(config["pretrain_meta_path"]),
            )
        elif config["feature_statistics_path"] is not None or config["pretrain_meta_path"] is not None:
            raise ValueError("native admission requires null feature statistics and pretrain metadata")
        conditioners.append(conditioning)
        _bound_hash(meta_path, meta_hash)
        contracts.append({"family": config["family"], "meta_sha256": meta_hash,
                          "model_sha256": model_hash, "tokenizer_sha256": tokenizer_hash,
                          "max_new_tokens": budget, "sequence_len": sequence_len,
                          "conditioning": None if conditioning is None else conditioning.identity})
    accepted, rejected = [], []
    for candidate in candidates:
        reasons = []
        for contract, tokenizer, conditioning in zip(contracts, tokenizers, conditioners, strict=True):
            rendered = render_messages_for_chat_completion(tokenizer, candidate.messages)
            if len(rendered.token_ids) + contract["max_new_tokens"] > contract["sequence_len"]:
                reasons.append({"family": contract["family"], "prompt_tokens": len(rendered.token_ids),
                                "reason": "target_context_overflow"})
            if conditioning is not None:
                feature_tokens = conditioning.prompt_token_count(rendered.token_ids, tokenizer)
                if feature_tokens > conditioning.identity["context_window"]:
                    reasons.append({"family": contract["family"], "prompt_tokens": feature_tokens,
                                    "reason": "target_conditioning_context_overflow"})
        if reasons:
            rejected.append({"candidate_id": candidate.candidate_id, "reasons": reasons})
        else:
            accepted.append(candidate)
    return accepted, contracts, rejected


def admit_extraction_contexts(
    candidates: Sequence[SourceCandidate], scorer: ChatScorer, reserved_tokens: int,
) -> tuple[list[SourceCandidate], list[dict[str, Any]]]:
    """Reserve extractor-token headroom; returned completions are still checked in full."""
    if type(reserved_tokens) is not int or not 0 <= reserved_tokens < scorer.context_limit:
        raise ValueError("reserved_extraction_tokens must be an integer within the extraction context")
    accepted, rejected = [], []
    for candidate in candidates:
        try:
            token_count = len(scorer.prepare(candidate).token_ids)
        except ChatQcError as exc:
            rejected.append({"candidate_id": candidate.candidate_id,
                             "reason": "extraction_preparation_error", "detail": str(exc)})
            continue
        if token_count + reserved_tokens > scorer.context_limit:
            rejected.append({"candidate_id": candidate.candidate_id,
                             "reason": "extraction_reserved_context_overflow", "prompt_tokens": token_count})
        else:
            accepted.append(candidate)
    return accepted, rejected


def run_negative_build(cfg: DictConfig) -> NegativeSourceBank:
    """Resolve source/target/SAE identities, then run the real negative producer."""
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise ValueError("negative build configuration must be a mapping")
    candidates = load_source_candidates(Path(raw["source_path"]), raw["source_sha256"])
    preparation = None
    if raw["source_manifest_path"] is not None:
        receipt = stream_hash_file_descriptor_bound(Path(raw["source_manifest_path"]).resolve(strict=True), retain_payload=True)
        if receipt.payload is None:
            raise ValueError("missing source preparation manifest bytes")
        document = strict_object(receipt.payload.decode("utf-8"))
        if (document.get("format") != SOURCE_FORMAT or document.get("complete") is not True
                or document.get("source", {}).get("sha256") != raw["source_sha256"]):
            raise ValueError("source preparation manifest does not bind the original-message source")
        preparation = {"sha256": receipt.sha256, "manifest": document}
    targets, ref_prefix = load_feature_targets(
        Path(raw["metadata_path"]), raw["metadata_sha256"], raw["concept_ids"], raw["extractor"],
    )
    admitted, target_contracts, rejected = admit_target_contexts(candidates, raw["target_admission"])
    renderer = load_render_tokenizer(Path(raw["render_tokenizer_path"]), raw["render_tokenizer_sha256"])
    scorer = build_chat_scorer(raw["extractor"], renderer)
    admitted, extraction_rejections = admit_extraction_contexts(admitted, scorer, raw["reserved_extraction_tokens"])
    semantic_identity = {
        "source_sha256": raw["source_sha256"], "metadata_sha256": raw["metadata_sha256"],
        "source_preparation": preparation,
        "concept_ids": raw["concept_ids"], "feature_ref_prefix": ref_prefix,
        "scorer": scorer.identity, "targets": target_contracts,
        "context_admission": {"candidate_count": len(candidates), "admitted_count": len(admitted),
                              "rejections": rejected, "extraction_rejections": extraction_rejections,
                              "reserved_extraction_tokens": raw["reserved_extraction_tokens"]},
    }
    with SynthResponseCache(Path(raw["response_cache"])) as cache:
        return build_negative_source(
            candidates=admitted, targets=targets, scorer=scorer,
            neutral_api=create_chat_api(raw["neutral"], cache),
            avoiding_api=create_chat_api(raw["avoiding"], cache),
            quality_judge=ConversationQualityJudge(create_chat_api(raw["quality"], cache)),
            config=ProducerConfig(**raw["producer"]), output_root=Path(raw["output_root"]),
            semantic_identity=semantic_identity,
        )
