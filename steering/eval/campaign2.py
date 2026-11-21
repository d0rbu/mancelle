"""Campaign cell identity and CPU-only preflight shared by local/Slurm runners."""

from __future__ import annotations

import math
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from nanochat.common import get_base_dir
from scripts.utils.campaign2 import Stage, load_strict_json_object, required_file_digest
from steering.eval.axbench_upstream.methods import load_method_setup
from steering.eval.campaign2_safety import SafetyCell, prepare_safety_cell
from steering.eval.campaign2_training_pairs import validate_training_pairs
from steering.eval.chat_eval_result_identity import semantic_digest
from steering.eval.harmfulness_safety_targets import SafetySteeringSettings
from steering.eval.hf_chat_generation import resolve_repository_hf_checkpoint_dir
from steering.sft.hf_contract import validate_repository_hf_sft_metadata
from steering.sft.loss_mask import LossMaskMode
from tasks.mixture_names import SFTMixture


class EvalMode(StrEnum):
    PREFLIGHT = "preflight"
    GENERATE = "generate"
    JUDGE = "judge"


class Variant(StrEnum):
    ENCDEC = "encdec"
    ENCDEC_SAFETY = "encdec_safety"
    PYTHIA = "pythia"
    PYTHIA_DIFFMEAN = "pythia_diffmean"
    PYTHIA_PROMPT = "pythia_prompt"


STAGE_MIXTURE = {Stage.INSTRUCT: SFTMixture.STANDARD_STEERED,
                 Stage.SAFETY: SFTMixture.SAFETY, Stage.CAPABILITIES: SFTMixture.CAPABILITIES}

def strength_leaf(value: Any) -> str:
    settings = SafetySteeringSettings.parse({"targets": ".", "strength": value})
    return f"strength-{settings.strength:.17g}"

def cell_output_dir(cfg: DictConfig) -> Path:
    root = Path(cfg.output_dir).resolve()
    return root / strength_leaf(cfg.safety.strength) if Variant(cfg.variant) is Variant.ENCDEC_SAFETY else root

def safety_record_receipt(identity: dict, index: int, record: dict) -> dict:
    return {"cell_sha256": semantic_digest(identity), "stable_pool_index": index,
        "prompt_seeds": identity["safety_binding"]["prompt_seeds"][index],
        "response_sha256": semantic_digest(record)}

def validate_safety_record(record: dict, row: dict, index: int, identity: dict) -> dict:
    plain = {key: value for key, value in record.items() if key != "generation"}
    if ({key: value for key, value in plain.items() if key != "response"} != row
            or not isinstance(plain.get("response"), str)
            or record.get("generation") != safety_record_receipt(identity, index, plain)):
        raise ValueError(f"safety response prompt/receipt identity mismatch at {index}")
    return plain

def preflight(cfg: DictConfig, *, safety_cell: SafetyCell | None = None) -> dict[str, Any]:
    mode, variant, stage = EvalMode(cfg.mode), Variant(cfg.variant), Stage(cfg.stage)
    if variant is Variant.ENCDEC_SAFETY:
        SafetySteeringSettings.parse({"targets": cfg.safety.targets, "strength": cfg.safety.strength})
        if not isinstance(cfg.conditioning.context_cache_dir, str) or not cfg.conditioning.context_cache_dir.strip():
            raise ValueError("encoder-decoder safety requires conditioning.context_cache_dir")
    if mode is EvalMode.JUDGE and not cfg.allow_paid_judge:
        raise ValueError("judge mode requires allow_paid_judge=true after owner authorizes provider calls")
    if type(cfg.max_new_tokens) is not int or cfg.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(cfg.seed) is not int or cfg.seed < 0 or type(cfg.num_benign) is not int or cfg.num_benign < 0:
        raise ValueError("seed and num_benign must be nonnegative integers")
    if cfg.num_harmful is not None and (type(cfg.num_harmful) is not int or cfg.num_harmful <= 0):
        raise ValueError("num_harmful must be null or a positive integer")
    if not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
        raise ValueError("Campaign 2 requires a finite positive temperature")
    if cfg.top_k != 50:
        raise ValueError("Campaign 2 pins top_k=50 across generation backends")
    checkpoint = Path(cfg.checkpoint_dir).resolve(strict=True)
    identities: dict[str, Any] = {}
    if variant in (Variant.ENCDEC, Variant.ENCDEC_SAFETY):
        if type(cfg.step) is not int or cfg.step < 0 or cfg.statistics is None:
            raise ValueError("encoder-decoder requires explicit step and matching statistics")
        metadata = checkpoint / f"meta_{cfg.step:06d}.json"
        meta = load_strict_json_object(metadata)
        if meta.get("step") != cfg.step:
            raise ValueError("encoder-decoder metadata step disagrees with the selected step")
        setup = load_method_setup(metadata, Path(cfg.statistics))
        if setup.prompt_segment_size != 32:
            raise ValueError("Campaign 2 requires source segment_size=32")
        names = [metadata.name, f"model_{cfg.step:06d}.pt"]
        if variant is Variant.ENCDEC or (checkpoint / "campaign2_cadence.json").exists():
            names.append("campaign2_cadence.json")
        identities = {
            name: required_file_digest(checkpoint / name).sha256 for name in names
        }
        identities["statistics"] = required_file_digest(Path(cfg.statistics)).sha256
        tokenizer_root = Path(get_base_dir()) / "tokenizer"
        if not tokenizer_root.is_dir():
            raise ValueError("stage the nanochat RustBPE tokenizer before evaluation")
        identities["tokenizer"] = {
            path.name: required_file_digest(path).sha256
            for path in sorted(tokenizer_root.iterdir())
            if path.is_file()
        }
        if not identities["tokenizer"]:
            raise ValueError("nanochat tokenizer directory is empty")
        mixture = meta.get("dataset")
    else:
        if cfg.step is not None or cfg.statistics is not None:
            raise ValueError("Pythia evaluation requires step=null and statistics=null")
        checkpoint = resolve_repository_hf_checkpoint_dir(cfg.checkpoint_dir)
        meta = validate_repository_hf_sft_metadata(
            load_strict_json_object(checkpoint / "sft_metadata.json")
        )
        if meta["output"]["model_type"] != "gpt_neox":
            raise ValueError("Campaign 2 HF evaluation requires Pythia GPTNeoX")
        if meta["completed_optimizer_step"] != int(checkpoint.name.removeprefix("step_")):
            raise ValueError("HF export step mismatch")
        if not list(checkpoint.glob("*.safetensors")):
            raise ValueError("HF export has no model safetensors")
        identities = {
            path.name: required_file_digest(path).sha256
            for path in sorted(checkpoint.iterdir())
            if path.is_file()
        }
        mixture = meta.get("mixture")
    if mixture != STAGE_MIXTURE[stage].value or meta.get("sft_loss_mask") != LossMaskMode.COMPLETION_ONLY.value:
        raise ValueError("checkpoint must match the requested stage and completion_only mask")
    if any(key in meta for key in ("selfcond", "selfcond_data", "grpo")):
        raise ValueError("Campaign 2 evaluates ordinary SFT checkpoints, not RL")
    safety_identity = None
    if variant is Variant.ENCDEC_SAFETY:
        admitted = prepare_safety_cell(cfg, setup) if safety_cell is None else safety_cell
        admitted.assert_stable()
        if admitted.request_sha256 != semantic_digest(OmegaConf.to_container(cfg, resolve=True)):
            raise ValueError("prepared safety runtime belongs to a different campaign configuration")
        safety_identity = admitted.identity
        if (safety_identity["checkpoint"]["metadata_sha256"] != identities[metadata.name]
                or safety_identity["checkpoint"]["model_sha256"] != identities[f"model_{cfg.step:06d}.pt"]
                or safety_identity["target"]["fit_contract"]["statistics_sha256"] != identities["statistics"]):
            raise ValueError("safety checkpoint/statistics bytes changed during preflight")
    if variant is Variant.PYTHIA_DIFFMEAN:
        if not isinstance(cfg.safety.training_pairs, str) or not cfg.safety.training_pairs:
            raise ValueError("DiffMean requires safety.training_pairs")
        pairs = Path(cfg.safety.training_pairs)
        identities.update(validate_training_pairs(pairs))
        if not math.isfinite(cfg.safety.factor) or cfg.safety.factor < 0 or cfg.safety.layer != 25:
            raise ValueError("DiffMean requires nonnegative finite factor and layer=25")
        if type(cfg.safety.fit_batch_size) is not int or cfg.safety.fit_batch_size <= 0:
            raise ValueError("fit_batch_size must be a positive integer")
    if variant is Variant.PYTHIA_PROMPT and not str(cfg.safety.instruction).strip():
        raise ValueError("prompt steering requires a nonempty safety instruction")
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    for key in (
        "mode",
        "allow_paid_judge",
        "judge_refusal_fallback_model",
        "requests_per_minute",
        "device",
    ):
        config.pop(key)
    repo = Path(__file__).resolve().parents[2]
    identity = {"format": "campaign2_cell_v1", "config": config, "checkpoint_files": identities,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "uv_lock_sha256": required_file_digest(repo / "uv.lock").sha256,
            "axbench_lock_sha256": required_file_digest(
                repo / "vendor/axbench/uv.lock"
            ).sha256}
    if safety_identity is not None:
        config["safety"]["strength"] = SafetySteeringSettings.parse(
            {"targets": cfg.safety.targets, "strength": cfg.safety.strength}
        ).strength
        identity["safety_binding"] = safety_identity
    else:
        # New safety inputs must not invalidate already generated baseline cells.
        config.pop("conditioning", None)
        config["safety"].pop("targets", None)
        config["safety"].pop("strength", None)
    return identity
