"""Typed scalar ChatCORE dispatch; only independently rendered prefixes enter inference."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch as t
import yaml
from transformers import AutoTokenizer

from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.feature_vocab import assert_resume_feature_vocab_matches
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.chat_benchmark_targets import (
    BenchmarkSteeringSettings,
    benchmark_fit_contract,
    load_benchmark_targets,
)
from steering.eval.chat_eval_result_identity import (
    ChatCacheSnapshot,
    build_self_conditioned_run_identity,
)
from steering.eval.chat_generation import render_chat_completion
from steering.eval.chat_self_conditioned_cache import resolve_fixed_chat_cache_contract
from steering.eval.chat_self_conditioned_generation import generate_self_conditioned_chat
from steering.eval.compacted_eval_loading import conditioned_eval_cache_contract_metadata
from steering.eval.context_cache_io import (
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
)
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.progression_results import (
    SelfConditionedSuiteResult,
    dump_suite_to_dict,
)
from steering.eval.self_conditioning_policy import (
    build_pure_model_self_conditioning_config,
    self_conditioning_example_seeds,
)
from steering.eval.target_conditioning import ContextRoutingMode
from steering.eval.turn_context_contract import (
    resolve_turn_context_contract,
    turn_extraction_tokenizer_id,
)
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    resolve_turn_self_cond_segment_size_rustbpe,
)


class ChatEvalMode(StrEnum):
    UNCONDITIONED = "unconditioned"
    TARGET_REAL = "target_real"
    SELF_CONDITIONED = "self_conditioned"


class ChatTaskEvalType(StrEnum):
    CATEGORICAL = "categorical"
    GENERATIVE = "generative"


def parse_chat_eval_mode(cfg) -> ChatEvalMode:
    explicit = cfg.get("evaluation_mode")
    mode = ChatEvalMode(explicit) if explicit is not None else (
        ChatEvalMode.TARGET_REAL if cfg.get("target_real_configs") else ChatEvalMode.UNCONDITIONED)
    block = cfg.get("self_conditioned")
    if (isinstance(block, Mapping) and block.get("benchmark_steering") is not None
            and mode is not ChatEvalMode.SELF_CONDITIONED):
        raise ValueError("benchmark_steering requires evaluation_mode=self_conditioned")
    if explicit is not None:
        if mode is ChatEvalMode.UNCONDITIONED and str(cfg.get("source", "")).startswith("encdec"):
            raise ValueError("explicit unconditioned mode contradicts encoder-decoder source")
        if mode is ChatEvalMode.TARGET_REAL:
            if cfg.get("hf_path") or not str(cfg.get("source", "")).startswith("encdec"):
                raise ValueError("explicit target_real requires a native encoder-decoder source")
            if not cfg.get("target_real_configs") or cfg.get("feature_configs"):
                raise ValueError("explicit target_real requires target_real_configs and empty feature_configs")
        elif cfg.get("target_real_configs") or cfg.get("feature_configs"):
            raise ValueError(f"explicit {mode.value} contradicts static/target-real configs")
        if mode is not ChatEvalMode.SELF_CONDITIONED and (
            cfg.get("self_conditioned", {}).get("cache_dir") or cfg.get("self_conditioned_output_dir")):
            raise ValueError("self_conditioned settings contradict evaluation_mode")
    return mode


@dataclass(frozen=True)
class ChatSelfConditionedSettings:
    cache_dir: Path
    seed: int
    fixed_segment_size_assertion: int | None
    benchmark_steering: BenchmarkSteeringSettings | None = None

    @classmethod
    def parse(cls, cfg) -> ChatSelfConditionedSettings:
        block = cfg.get("self_conditioned")
        allowed = {"cache_dir", "seed", "fixed_segment_size_assertion", "benchmark_steering"}
        if not isinstance(block, Mapping) or set(block) - allowed:
            raise ValueError("self_conditioned accepts cache_dir, seed, fixed assertion and benchmark_steering; no CLI cadence")
        for key in cfg:
            if key == "segment_size" or "cadence" in key or ("self_cond" in key and "segment_size" in key):
                raise ValueError("self_conditioned rejects CLI cadence; use recorded checkpoint metadata")
        if not isinstance(block.get("cache_dir"), str) or not block["cache_dir"].strip():
            raise ValueError("self_conditioned.cache_dir is required")
        if type(block.get("seed")) is not int:
            raise ValueError("self_conditioned.seed must be an integer")
        output_dir = cfg.get("self_conditioned_output_dir")
        if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
            raise ValueError("self_conditioned_output_dir must be null or a nonblank path")
        for key in ("batch_size", "num_samples"):
            if type(cfg.get(key)) is not int or cfg[key] != 1:
                raise ValueError(f"self_conditioned requires {key}=1")
        if type(cfg.get("temperature")) not in (int, float) or cfg.temperature != 0:
            raise ValueError("self_conditioned requires temperature=0")
        if cfg.get("greedy_microbatch_size", 1) != 1 or cfg.get("greedy_python_tools", False):
            raise ValueError("self_conditioned requires scalar chat driver")
        if cfg.get("hf_path") or not str(cfg.get("source", "")).startswith("encdec"):
            raise ValueError("self_conditioned requires a native encoder-decoder checkpoint")
        if cfg.get("next_segment_conditioning_offset") not in (False, "auto"):
            raise ValueError("self_conditioned does not accept static next-segment routing")
        for key in ("max_new_tokens", "model_max_seq_len"):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise ValueError(f"self_conditioned requires positive integer {key}")
        if cfg.get("top_k") is not None and (type(cfg.top_k) is not int or cfg.top_k <= 0):
            raise ValueError("top_k must be null or a positive integer")
        return cls(Path(block["cache_dir"]), block["seed"],
                   block.get("fixed_segment_size_assertion"),
                   None if block.get("benchmark_steering") is None else
                   BenchmarkSteeringSettings.parse(block["benchmark_steering"]))


@t.inference_mode()
def run_self_conditioned_chat_eval(
    cfg, *, settings, model, tokenizer, meta, mode, feature_contract,
    checkpoint_snapshot, checkpoint_identity, device_type, task_factory, task_names, all_task_names, problem_range, metric,
) -> Path:
    if feature_contract is None:
        raise ValueError("self_conditioned requires the complete encoder-decoder feature contract")
    if cfg.get("segment_mode") is not None and SegmentMode(cfg.segment_mode) is not mode:
        raise ValueError("chat_eval segment_mode disagrees with checkpoint metadata")
    if model.dec_config.sequence_len != cfg.model_max_seq_len:
        raise ValueError("model sequence_len differs from model_max_seq_len")
    checkpoint_timing = resolve_next_segment_conditioning_offset(
        {"next_segment_conditioning_offset": "auto"}, meta)
    if resolve_next_segment_conditioning_offset(cfg, meta) != checkpoint_timing:
        raise ValueError("next_segment_conditioning_offset disagrees with checkpoint metadata")
    if checkpoint_timing:
        raise ValueError("self_conditioned requires checkpoint current-segment prompt routing")
    directory = meta["checkpoint_dir"]
    # Bind the stored compact vocabulary digest as well as the resolved mask/clusters.
    assert_resume_feature_vocab_matches(model.feature_vocab, meta, checkpoint_dir=directory)
    turn_contract, source_tokenizer, fixed_contract = None, None, None
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        if settings.fixed_segment_size_assertion is not None:
            raise ValueError("fixed cadence assertion is invalid on turn checkpoints")
        cadence = resolve_turn_self_cond_segment_size_rustbpe(meta, checkpoint_dir=directory)
        lineage = OnlineSAELineage.from_checkpoint_meta(meta)
        source_tokenizer = AutoTokenizer.from_pretrained(turn_extraction_tokenizer_id(lineage.model, cfg.get("extraction_tokenizer_id")))
        turn_contract = resolve_turn_context_contract(meta, tokenizer=tokenizer, extraction_tokenizer=source_tokenizer,
            statistics_path=cfg.get("stats_path"), checkpoint_contract=feature_contract)
        identity = cache_identity_from_config({}, turn_contract=turn_contract)
        # Schema 5 binds mask/clusters/vocabulary through expected_turn_contract.
        # Its writer does not emit the fixed cache's feature_mask_canonical key.
        expected_metadata = {}
    else:
        fixed_contract = resolve_fixed_chat_cache_contract(meta, tokenizer, feature_contract, statistics_path=cfg.get("stats_path"),
            segment_size_assertion=settings.fixed_segment_size_assertion, loaded_model_sha256=checkpoint_identity["model_sha256"])
        cadence = fixed_contract.cadence
        identity = fixed_contract.identity
        expected_metadata = {**fixed_contract.metadata, **conditioned_eval_cache_contract_metadata(feature_contract)}
    cache_dir = context_length_cache_dir(settings.cache_dir, cfg.model_max_seq_len, hash_leaf=identity.feature_selection_hash[:16])
    cache_snapshot = ChatCacheSnapshot.capture(cache_dir)
    provider = SafetensorsContextFeatureProvider(cache_dir, identity=identity,
        expected_n_sae_features=model.encoder.config.n_sae_features, expected_num_buckets=model.encoder.config.num_buckets,
        expected_metadata=expected_metadata,
        expected_feature_vocab=model.feature_vocab if feature_contract.clustered or turn_contract is not None else None,
        expected_source_segment_mode=mode, expected_turn_contract=turn_contract)
    cache_snapshot.bind_loaded(provider)
    # Mask-only fixed caches also prove every original id against the checkpoint.
    if model.feature_vocab is not None:
        allowed = set(model.feature_vocab.original_ids)
        for _, segments in provider._rows.values():
            if any(set(segment.feature_ids) - allowed for segment in segments):
                raise ValueError("cache feature IDs outside checkpoint vocabulary")

    assert cadence is not None  # The family-specific reader requires the recorded K.
    policy = build_pure_model_self_conditioning_config(model=model, checkpoint_meta=meta,
        n_sae_features=model.encoder.config.n_sae_features, feature_mask=feature_contract.feature_mask,
        consolidation=feature_contract.consolidation, segment_mode=mode, segment_size=cadence,
        selection_seed=settings.seed, percentile_table=t.empty((0, 0), dtype=t.float64))
    if not task_names or len(task_names) != len(set(task_names)):
        raise ValueError("self_conditioned requires unique known task names")
    targets = None
    if settings.benchmark_steering is not None:
        fit_contract, feature_ids = benchmark_fit_contract(feature_contract=feature_contract,
            n_sae_features=model.encoder.config.n_sae_features, tokenizer=tokenizer,
            turn_contract=turn_contract, fixed_contract=fixed_contract)
        targets = load_benchmark_targets(settings.benchmark_steering.targets,
            expected_contract=fit_contract, expected_feature_ids=feature_ids, task_names=all_task_names)
    prepared: dict[str, tuple[Any, range, list[tuple[Any, list[int]]], Any]] = {}
    for name in task_names:
        task_policy = policy if targets is None else targets.policy(policy, name, settings.benchmark_steering.strength)
        task = task_factory(name)
        indices = problem_range(name, len(task), cfg.get("problem_start"), cfg.max_problems)
        rows = []
        for index in indices:
            conversation = task[index]
            ids = render_chat_completion(tokenizer, conversation).token_ids
            if len(ids) < 2 or len(ids) > model.dec_config.sequence_len:
                raise ValueError(f"{name} problem {index}: overlength or empty prompt before evaluation")
            if targets is not None:
                targets.validate_example(name, len(task), index, ids)
            segments = provider.get(ids)
            if not segments:
                raise ValueError(f"{name} problem {index}: missing prefix cache row before evaluation")
            ChatTaskEvalType(task.eval_type)
            rows.append((conversation, ids))
        prepared[name] = (task, indices, rows, task_policy)
    run_identity = build_self_conditioned_run_identity(cfg, settings=settings, model=model, meta=meta, mode=mode, cadence=cadence,
        tokenizer=tokenizer, policy=prepared[task_names[0]][3], feature_contract=feature_contract, checkpoint_identity=checkpoint_identity,
        provider=provider, cache_snapshot=cache_snapshot, turn_contract=turn_contract, fixed_contract=fixed_contract,
        device_type=device_type, benchmark_targets_sha256=None if targets is None else targets.sha256)
    results, details = {}, {}
    for name, (task, indices, rows, policy) in prepared.items():
        passed = 0
        for index, (conversation, ids) in zip(indices, rows, strict=True):
            token_seed, selection_seed = self_conditioning_example_seeds(base_seed=settings.seed, task_name=name,
                original_problem_index=index, prompt_token_key=tuple(ids), sample_index=0)
            engine = build_engine_self_conditioned(model, tokenizer, model.get_device(), replace(policy, selection=replace(policy.selection,
                    rng=t.Generator(device="cpu").manual_seed(selection_seed))),
                context_provider=provider, turn_contract=turn_contract,
                context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE if turn_contract else ContextRoutingMode.DECODER_ALIGNED)
            if ChatTaskEvalType(task.eval_type) is ChatTaskEvalType.CATEGORICAL:
                letters = conversation["letters"]
                encoded = [tokenizer.encode(letter) for letter in letters]
                if not encoded or any(len(value) != 1 for value in encoded):
                    raise ValueError("categorical letters must each encode as one token")
                logits = engine.prepare_self_conditioned_prefill(ids, max_new_tokens=0).logits
                if not bool(t.isfinite(logits).all()):
                    raise ValueError("nonfinite categorical logits")
                prediction = letters[int(logits[0, [value[0] for value in encoded]].argmax())]
            else:
                output = generate_self_conditioned_chat(engine, ids, max_tokens=cfg.max_new_tokens,
                    temperature=0.0, top_k=cfg.top_k, seed=token_seed, extraction_tokenizer=source_tokenizer)
                prediction = tokenizer.decode(list(output.token_ids[len(ids):]))
            passed += int(bool(task.evaluate(conversation, prediction)))
        results[name] = passed / len(indices)
        details[name] = {"problem_start": indices.start, "max_problems": len(indices), "task_size": len(task),
                         "num_passed": passed, "total": len(indices)}
    body = {"results": results, "task_details": details, "run_identity": run_identity}
    value = metric(results)
    if value is not None:
        body["chatcore_metric"] = value
    payload = dump_suite_to_dict(SelfConditionedSuiteResult(
        "self_conditioned", str(settings.cache_dir), {"adhoc": body}))
    checkpoint_snapshot.assert_stable()
    cache_snapshot.assert_stable()
    if fixed_contract is not None and hashlib.sha256(fixed_contract.statistics_path.read_bytes()).hexdigest() != fixed_contract.statistics_sha256:
        raise ValueError("fixed training statistics changed during evaluation")
    # Refuse changing an existing result; publication is the final operation.
    output_path = Path(cfg.get("self_conditioned_output_dir") or Path(directory) / "chat_eval") / "self_conditioned/result.yaml"
    serialized = yaml.safe_dump(json.loads(json.dumps(payload)), sort_keys=False).encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(serialized)
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
