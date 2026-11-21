"""CPU admission and immutable inputs for Campaign 2's shared safety runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import torch as t
from omegaconf import DictConfig, OmegaConf

from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.common import COMPUTE_DTYPE, get_base_dir
from steering._common.file_inventory import file_signature
from steering.conditional.checkpoint_feature_contract import (
    resolve_and_assert_model_feature_contract,
)
from steering.conditional.eval_conditioning import EvalConditioning
from steering.eval.axbench_upstream.methods import MethodSetup
from steering.eval.chat_eval_result_identity import (
    ChatCacheSnapshot,
    SelfConditionedChatEvalCheckpointSnapshot,
    effective_architecture_sha256,
    self_conditioned_code_identity,
    semantic_digest,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.eval.harmfulness_safety_targets import (
    SAFETY_EVALUATION_SOURCE_NAMES,
    SAFETY_TARGET_NAME,
    SafetySteeringSettings,
    load_populations,
)
from steering.eval.harmfulness_self_conditioning import (
    PROMPT_RNG_VERSION,
    HarmfulnessConditionName,
    HarmfulnessSelfCondSetup,
    harmfulness_completion_provenance,
    harmfulness_prompt_seeds,
    prepare_harmfulness_self_conditioning,
)
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.segmentation import SegmentMode

SAFETY_POSITIVE_COUNT = 128


def cadence_evidence(setup: MethodSetup) -> dict[str, Any]:
    """Admit both campaign views and existing measured ordinary-SFT exports."""
    path = setup.target_meta_path.parent / "campaign2_cadence.json"
    embedded = setup.meta.get("turn_self_cond_cadence_provenance")
    witnesses = [json.loads(path.read_bytes())] if path.exists() else []
    if embedded is not None:
        witnesses.append(embedded)
    if not witnesses:
        raise ValueError("safety cell requires measured RustBPE cadence evidence")
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise ValueError("measured cadence evidence must be an object")
        for key in ("documents", "windows", "minimum", "maximum"):
            if type(witness.get(key)) is not int or witness[key] <= 0:
                raise ValueError(f"measured cadence requires positive {key}")
        try:
            mean = Fraction(witness["mean"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("measured cadence requires a finite exact mean") from exc
        if (not witness["minimum"] <= mean <= witness["maximum"]
                or round(mean) != setup.turn_self_cond_segment_size_rustbpe
                or witness.get("cadence", round(mean)) != round(mean)):
            raise ValueError("measured cadence evidence disagrees with checkpoint")
    return {"cadence": setup.turn_self_cond_segment_size_rustbpe, "witnesses": witnesses}


@dataclass(frozen=True)
class SafetyCell:
    runtime: HarmfulnessSelfCondSetup
    rows: list[dict[str, Any]]
    identity: dict[str, Any]
    checkpoint_snapshot: SelfConditionedChatEvalCheckpointSnapshot
    cache_snapshot: ChatCacheSnapshot
    input_signatures: tuple[tuple[Path, tuple[int, ...]], ...] = ()
    request_sha256: str = ""

    def assert_stable(self) -> None:
        self.checkpoint_snapshot.assert_stable()
        self.cache_snapshot.assert_stable()
        self.runtime.assert_target_unchanged()
        if any(file_signature(path) != signature for path, signature in self.input_signatures):
            raise ValueError("safety statistics/tokenizer/cadence/code changed during evaluation")


def prepare_safety_cell(cfg: DictConfig, setup: MethodSetup) -> SafetyCell:
    """Load weights, target and every requested real cache row on CPU only."""
    settings = SafetySteeringSettings.parse(
        {"targets": cfg.safety.targets, "strength": cfg.safety.strength}
    )
    repo = Path(__file__).resolve().parents[2]
    inputs = [Path(cfg.statistics), *(Path(get_base_dir()) / "tokenizer").iterdir()]
    inputs += [setup.target_meta_path.parent / "campaign2_cadence.json"]
    inputs += [path for directory in ("steering", "nanochat", "tasks", "scripts")
               for path in (repo / directory).rglob("*.py")]
    signatures = tuple((path, file_signature(path)) for path in inputs if path.is_file())
    if setup.segment_mode is not SegmentMode.turn_delimited_fixed_tokens or setup.prompt_segment_size != 32:
        raise ValueError("Campaign 2 safety requires turn-delimited source s32")
    cadence = cadence_evidence(setup)
    metadata_path = setup.target_meta_path.resolve(strict=True)
    payload = metadata_path.read_bytes()
    if json.loads(payload) != setup.meta:
        raise ValueError("checkpoint metadata changed after campaign setup")
    snapshot = cast(SelfConditionedChatEvalCheckpointSnapshot,
        SelfConditionedChatEvalCheckpointSnapshot.capture(metadata_path, payload, cfg.step))
    model, tokenizer, meta = build_encdec_model(str(metadata_path.parent), cfg.step, t.device("cpu"),
        ModelPhase.EVAL, eval_conditioning=EvalConditioning.conditions_within_vocab(HarmfulnessConditionName.SAFETY_STEERED))
    meta = {**meta, "checkpoint_dir": str(metadata_path.parent)}
    checkpoint_identity = snapshot.bind_loaded(meta)
    features = resolve_and_assert_model_feature_contract(model, meta, checkpoint_dir=str(metadata_path.parent),
        surface="Campaign 2 safety", missing_metadata_label="canonical feature contract",
        missing_metadata_guidance="Stage the checkpoint's original metadata and vocabulary.")
    pool = resolve_harmfulness_prompt_pool(seed=cfg.seed, max_harmful=cfg.num_harmful, num_benign=cfg.num_benign)
    rows = [dict(row) for row in pool.rows]
    prompt_ids = [render_prompt_for_chat_completion(tokenizer, row["prompt"]).token_ids for row in rows]
    for index, ids in enumerate(prompt_ids):
        if len(ids) + cfg.max_new_tokens > model.dec_config.sequence_len:
            raise ValueError(f"prompt {index} plus generation budget exceeds the checkpoint context")
    runtime_cfg = {"stats_path": cfg.statistics,
        "conditioning": OmegaConf.to_container(cfg.conditioning, resolve=True),
        "self_conditioned": {"safety_steering": {
            "targets": str(settings.targets), "strength": settings.strength}}}
    runtime = prepare_harmfulness_self_conditioning(runtime_cfg, model=model,
        tokenizer=tokenizer, device=t.device("cpu"), meta=meta, feature_contract=features, prompt_token_ids=prompt_ids)
    provider = runtime.context_provider
    if not isinstance(provider, SafetensorsContextFeatureProvider):
        raise ValueError("Campaign 2 safety requires a persisted real prompt cache")
    cache_snapshot = ChatCacheSnapshot.capture(provider.cache_dir)
    cache_snapshot.bind_loaded(provider)
    if runtime.safety_steering is None:
        raise ValueError("safety target disappeared during admission")
    targets = runtime.safety_steering.targets
    manifest = targets.manifest
    preparation = manifest["preparation"]
    populations, sources = load_populations(SAFETY_EVALUATION_SOURCE_NAMES)
    evaluation = [
        row
        for name in SAFETY_EVALUATION_SOURCE_NAMES
        for row in populations[name]
    ]
    if (preparation["evaluation_prompts"] != [row["prompt"] for row in evaluation]
            or any(preparation["sources"][name] != spec for name, spec in sources.items())):
        raise ValueError("safety target must exclude the complete pinned evaluation sources")
    accepted = preparation["accepted"][SAFETY_TARGET_NAME]
    if preparation["positive_cap"] != SAFETY_POSITIVE_COUNT or len(accepted) != SAFETY_POSITIVE_COUNT:
        raise ValueError(f"Campaign 2 requires exactly {SAFETY_POSITIVE_COUNT} safe training positives")
    if model.feature_pred_config is None or runtime.policy.score_policy is None:
        raise ValueError("safety runtime requires resolved feature prediction and score policies")
    identity = {
        "checkpoint": {**checkpoint_identity, "architecture_sha256": effective_architecture_sha256(model),
            "feature_prediction": asdict(model.feature_pred_config), "cadence": cadence},
        "target": {"path": str(settings.targets.resolve(strict=True)), "sha256": targets.sha256,
            "tensor": f"target.{SAFETY_TARGET_NAME}", "fit_contract": manifest["fit_contract"],
            "feature_ids": (list(model.feature_vocab.original_ids) if model.feature_vocab is not None
                            else list(range(model.encoder.config.n_sae_features))),
            "sources": preparation["sources"], "response_provenance": accepted,
            "preparation_sha256": semantic_digest(preparation), "fit_provenance": manifest["fit_provenance"]},
        "completion": harmfulness_completion_provenance(runtime),
        "selection": {"positive": asdict(runtime.policy.selection.selection),
            "negative": asdict(runtime.policy.score_policy.negative_selection),
            "pool_mode": runtime.policy.pool_mode.value,
            "selection_producer": runtime.policy.selection_producer.value,
            "bucket_producer": runtime.policy.bucket_producer.value},
        "cache": {"path": str(cache_snapshot.root),
            "manifest_sha256": hashlib.sha256(cache_snapshot.manifest_bytes).hexdigest(),
            "schema_version": provider.manifest["schema_version"],
            "shards": provider.manifest["shards"]},
        "prompts_sha256": semantic_digest(rows), "prompt_rng_version": PROMPT_RNG_VERSION,
        "prompt_seeds": [list(harmfulness_prompt_seeds(base_seed=cfg.seed, stable_pool_index=index,
            prompt=row["prompt"], pool=row["pool"], category=row["category"])) for index, row in enumerate(rows)],
        "compute_dtype": str(COMPUTE_DTYPE), "cuda_parameter_dtype": str(t.bfloat16),
        "execution_device_type": t.device(cfg.device).type,
        "code": self_conditioned_code_identity(),
    }
    # Sampling dataclasses contain Enum members; disk/resume identity uses JSON primitives.
    identity = json.loads(canonical_turn_payload(identity))
    cell = SafetyCell(runtime, rows, identity, snapshot, cache_snapshot, signatures,
        semantic_digest(OmegaConf.to_container(cfg, resolve=True)))
    cell.assert_stable()
    return cell
