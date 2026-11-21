"""Small harmfulness adapter for the shared self-conditioning engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import torch as t
from transformers import AutoTokenizer

from steering.conditional.checkpoint_conditioning import read_checkpoint_conditioning
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.delta import DeltaMode
from steering.conditional.encoder_decoder.config import FeaturePredLossFn
from steering.conditional.encoder_decoder.eval_adapter import ContextProvider
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.feature_vocab import assert_resume_feature_vocab_matches
from steering.conditional.sampling import SELFCOND_ACTION_RANKING_POLICY
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.chat_benchmark_targets import benchmark_fit_contract
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.chat_self_conditioned_cache import (
    FixedChatCacheContract,
    resolve_fixed_chat_cache_contract,
)
from steering.eval.chat_self_conditioned_generation import generate_self_conditioned_chat
from steering.eval.compacted_eval_loading import conditioned_eval_cache_contract_metadata
from steering.eval.context_cache_io import (
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
    feature_selection_hash_leaf,
)
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.harmfulness_generation import SELF_CONDITIONED_SEED_SCHEDULE
from steering.eval.harmfulness_safety_targets import (
    SafetySteeringSettings,
    SafetyTargets,
    load_safety_targets,
)
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.self_conditioning_policy import build_pure_model_self_conditioning_config
from steering.eval.target_conditioning import ContextRoutingMode
from steering.eval.turn_context_contract import (
    TurnContextContract,
    resolve_turn_context_contract,
    turn_extraction_tokenizer_id,
)
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
)

PROMPT_RNG_VERSION: Final = SELF_CONDITIONED_SEED_SCHEDULE
TOKEN_RNG_DOMAIN: Final = "tokens"
SELECTION_RNG_DOMAIN: Final = "feature_selection"


def harmfulness_prompt_seeds(
    *,
    base_seed: int,
    stable_pool_index: int,
    prompt: str,
    pool: str,
    category: str,
) -> tuple[int, int]:
    """Derive independent prompt-local token and feature-selection seeds."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if (
        isinstance(stable_pool_index, bool)
        or not isinstance(stable_pool_index, int)
        or stable_pool_index < 0
    ):
        raise ValueError("stable_pool_index must be a non-negative integer")
    identity = {
        "stable_pool_index": stable_pool_index,
        "prompt": prompt,
        "pool": pool,
        "category": category,
        "sample_index": 0,
    }

    def derive(domain: str) -> int:
        payload = {
            "version": PROMPT_RNG_VERSION,
            "domain": domain,
            "base_seed": base_seed,
            "prompt": identity,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    return derive(TOKEN_RNG_DOMAIN), derive(SELECTION_RNG_DOMAIN)


@dataclass(frozen=True)
class HarmfulnessContextRuntime:
    """The same authenticated prompt geometry and cadence used by ChatCORE."""

    provider: SafetensorsContextFeatureProvider
    cadence: int
    mode: SegmentMode
    turn_contract: TurnContextContract | None
    fixed_contract: FixedChatCacheContract | None
    source_tokenizer: Any


def parse_harmfulness_safety_settings(cfg: Mapping[str, Any]) -> SafetySteeringSettings | None:
    block = cfg.get("self_conditioned")
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) - {"safety_steering"}:
        raise ValueError("harmfulness self_conditioned accepts only safety_steering")
    raw = block.get("safety_steering")
    return None if raw is None else SafetySteeringSettings.parse(raw)


def build_harmfulness_context_runtime(
    cfg: Mapping[str, Any], *, model: EncoderDecoderGPT, tokenizer: Any,
    meta: Mapping[str, Any], feature_contract: CheckpointFeatureContract,
) -> HarmfulnessContextRuntime:
    """Resolve checkpoint-authoritative fixed/turn caches before any forward."""
    directory = str(meta["checkpoint_dir"])
    assert_resume_feature_vocab_matches(model.feature_vocab, meta, checkpoint_dir=directory)
    mode = resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=directory)
    if resolve_next_segment_conditioning_offset({"next_segment_conditioning_offset": "auto"}, meta):
        raise ValueError("harmfulness self-conditioning requires checkpoint current-segment prompt routing")
    cond_cfg = cfg.get("conditioning")
    if not isinstance(cond_cfg, Mapping) or not cond_cfg.get("context_cache_dir"):
        raise ValueError("Encoder-decoder harmfulness requires conditioning.context_cache_dir")
    turn_contract, fixed_contract, source_tokenizer = None, None, None
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        if cfg.get("self_cond_segment_size") is not None:
            raise ValueError("turn harmfulness rejects self_cond_segment_size; cadence comes from checkpoint metadata")
        cadence = resolve_turn_self_cond_segment_size_rustbpe(meta, checkpoint_dir=directory)
        assert cadence is not None  # The checkpoint mode was authenticated above.
        lineage = OnlineSAELineage.from_checkpoint_meta(meta)
        source_tokenizer = AutoTokenizer.from_pretrained(
            turn_extraction_tokenizer_id(lineage.model, cfg.get("extraction_tokenizer_id")))
        turn_contract = resolve_turn_context_contract(meta, tokenizer=tokenizer,
            extraction_tokenizer=source_tokenizer, statistics_path=cfg.get("stats_path"),
            checkpoint_contract=feature_contract)
        identity = cache_identity_from_config({}, turn_contract=turn_contract)
        expected_metadata = {}
    else:
        fixed_contract = resolve_fixed_chat_cache_contract(meta, tokenizer, feature_contract,
            statistics_path=cfg.get("stats_path"), segment_size_assertion=cfg.get("self_cond_segment_size"),
            loaded_model_sha256=meta.get("loaded_model_sha256"))
        cadence = fixed_contract.cadence
        identity = fixed_contract.identity
        expected_metadata = {**fixed_contract.metadata,
            **conditioned_eval_cache_contract_metadata(feature_contract)}
    identity_assertions = cond_cfg.get("identity")
    if identity_assertions is not None:
        expected_identity = identity.as_dict()
        if not isinstance(identity_assertions, Mapping) or set(identity_assertions) - expected_identity.keys():
            raise ValueError("harmfulness conditioning.identity must contain only cache identity fields")
        for field, supplied in identity_assertions.items():
            if supplied is not None and (
                type(supplied) is not type(expected_identity[field]) or supplied != expected_identity[field]
            ):
                raise ValueError(f"harmfulness conditioning.identity.{field} conflicts with checkpoint cache identity")
    provider = SafetensorsContextFeatureProvider(
        context_length_cache_dir(Path(cond_cfg["context_cache_dir"]), model.dec_config.sequence_len,
            hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash)),
        identity=identity, expected_n_sae_features=model.encoder.config.n_sae_features,
        expected_num_buckets=model.encoder.config.num_buckets, expected_metadata=expected_metadata,
        expected_feature_vocab=model.feature_vocab if feature_contract.clustered or turn_contract is not None else None,
        expected_source_segment_mode=mode, expected_turn_contract=turn_contract)
    if model.feature_vocab is not None:
        allowed = set(model.feature_vocab.original_ids)
        for _, segments in provider._rows.values():
            if any(set(segment.feature_ids) - allowed for segment in segments):
                raise ValueError("harmfulness cache feature IDs outside checkpoint vocabulary")
    return HarmfulnessContextRuntime(provider, int(cadence), mode, turn_contract, fixed_contract, source_tokenizer)


class HarmfulnessConditionName(StrEnum):
    UNCONDITIONAL = "unconditional"
    SELF_CONDITIONED = "self_conditioned"
    SAFETY_STEERED = "safety_steered"


@dataclass(frozen=True)
class HarmfulnessSafetySteering:
    settings: SafetySteeringSettings
    targets: SafetyTargets

    def assert_unchanged(self) -> None:
        self.targets.assert_matches_file(self.settings.targets)


@dataclass(frozen=True)
class HarmfulnessSelfCondSetup:
    """One validated dense policy, with independent per-prompt random streams."""

    model: EncoderDecoderGPT
    tokenizer: Any
    device: t.device
    context_provider: ContextProvider
    segment_size: PositiveInt
    policy: SelfCondConfig
    turn_contract: TurnContextContract | None = None
    source_tokenizer: Any = None
    safety_steering: HarmfulnessSafetySteering | None = None

    @property
    def safety_strength(self) -> float:
        return 0.0 if self.safety_steering is None else self.safety_steering.settings.strength

    def assert_target_unchanged(self) -> None:
        if self.safety_steering is not None:
            self.safety_steering.assert_unchanged()

    def make_config(self, *, selection_seed: int) -> SelfCondConfig:
        return replace(self.policy, selection=replace(self.policy.selection,
            rng=t.Generator(device="cpu").manual_seed(selection_seed)))


def harmfulness_completion_provenance(setup: HarmfulnessSelfCondSetup) -> dict[str, Any]:
    return {
        "self_cond_segment_size": int(setup.segment_size),
        "strength": setup.safety_strength,
        "action_ranking_policy": SELFCOND_ACTION_RANKING_POLICY,
        "safety_steering": None if setup.safety_steering is None else {
            "targets_sha256": setup.safety_steering.targets.sha256,
            "strength": setup.safety_strength,
        },
    }


def resolve_harmfulness_self_cond_setup(
    *, model: EncoderDecoderGPT, tokenizer: Any, device: t.device,
    context_provider: ContextProvider, meta: Mapping[str, Any],
    feature_contract: CheckpointFeatureContract, segment_size: int,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    turn_contract: TurnContextContract | None = None, source_tokenizer: Any = None,
    safety_steering: HarmfulnessSafetySteering | None = None,
) -> HarmfulnessSelfCondSetup:
    """Reuse the AxBench/ChatCORE policy, including its exact weight-zero arm."""
    if model.feature_pred_config is None or FeaturePredLossFn(model.feature_pred_config.loss_fn) is not FeaturePredLossFn.BCE:
        raise ValueError("harmfulness self-conditioning requires feature_prediction.loss_fn=bce")
    delta_mode = read_checkpoint_conditioning(meta).delta_mode()
    if delta_mode is not None and delta_mode is not DeltaMode.splice_counterfactual:
        raise ValueError("harmfulness self-conditioning cannot remove cross-segment "
            f"feature_delta.mode={delta_mode.value!r}")
    policy = build_pure_model_self_conditioning_config(model=model, checkpoint_meta=meta,
        n_sae_features=model.encoder.config.n_sae_features, feature_mask=feature_contract.feature_mask,
        consolidation=feature_contract.consolidation, segment_mode=segment_mode, segment_size=segment_size,
        selection_seed=0, percentile_table=t.empty((0, 0), dtype=t.float64))
    if safety_steering is not None:
        policy = safety_steering.targets.policy(policy, safety_steering.settings.strength)
    SelfConditionedFeatureProvider(model, policy, device)
    return HarmfulnessSelfCondSetup(model, tokenizer, device, context_provider,
        PositiveInt.parse(segment_size), policy, turn_contract, source_tokenizer,
        safety_steering)


def prepare_harmfulness_self_conditioning(
    cfg: Mapping[str, Any], *, model: EncoderDecoderGPT, tokenizer: Any, device: t.device,
    meta: Mapping[str, Any], feature_contract: CheckpointFeatureContract,
    prompt_token_ids: Sequence[Sequence[int]],
) -> HarmfulnessSelfCondSetup:
    runtime = build_harmfulness_context_runtime(cfg, model=model, tokenizer=tokenizer,
        meta=meta, feature_contract=feature_contract)
    settings = parse_harmfulness_safety_settings(cfg)
    steering = None
    if settings is not None:
        fit_contract, feature_ids = benchmark_fit_contract(feature_contract=feature_contract,
            n_sae_features=model.encoder.config.n_sae_features, tokenizer=tokenizer,
            turn_contract=runtime.turn_contract, fixed_contract=runtime.fixed_contract)
        targets = load_safety_targets(settings.targets, tokenizer=tokenizer, expected_contract=fit_contract,
            expected_feature_ids=feature_ids)
        targets.validate_prompts(prompt_token_ids)
        steering = HarmfulnessSafetySteering(settings, targets)
    for ids in prompt_token_ids:
        if len(ids) < 2 or len(ids) > model.dec_config.sequence_len:
            raise ValueError("harmfulness prompt is empty or overlength before evaluation")
        if len(ids) == model.dec_config.sequence_len:
            raise ValueError("harmfulness prompt leaves no generation budget before evaluation")
        if not runtime.provider.get(list(ids)):
            raise ValueError("harmfulness prompt has no real context cache row before evaluation")
    return resolve_harmfulness_self_cond_setup(model=model, tokenizer=tokenizer, device=device,
        context_provider=runtime.provider, meta=meta, feature_contract=feature_contract,
        segment_size=runtime.cadence, segment_mode=runtime.mode, turn_contract=runtime.turn_contract,
        source_tokenizer=runtime.source_tokenizer,
        safety_steering=steering)


def generate_harmfulness_response_self_conditioned(
    *, setup: HarmfulnessSelfCondSetup, stable_pool_index: int, prompt: str,
    pool: str, category: str, base_seed: int, max_new_tokens: int,
    temperature: float, top_k: int | None,
) -> str:
    """Generate with the shared turn-aware scalar ChatCORE driver."""
    token_seed, selection_seed = harmfulness_prompt_seeds(base_seed=base_seed,
        stable_pool_index=stable_pool_index, prompt=prompt, pool=pool, category=category)
    ids = render_prompt_for_chat_completion(setup.tokenizer, prompt).token_ids
    if setup.safety_steering is not None:
        setup.assert_target_unchanged()
        setup.safety_steering.targets.validate_prompts([ids])
    engine = build_engine_self_conditioned(setup.model, setup.tokenizer, setup.device,
        setup.make_config(selection_seed=selection_seed), context_provider=setup.context_provider,
        turn_contract=setup.turn_contract,
        context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE if setup.turn_contract else ContextRoutingMode.DECODER_ALIGNED)
    result = generate_self_conditioned_chat(engine, ids, max_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, seed=token_seed,
        extraction_tokenizer=setup.source_tokenizer)
    return setup.tokenizer.decode(list(result.token_ids[len(ids):]))
