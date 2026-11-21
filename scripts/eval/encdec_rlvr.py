"""Evaluate the exact canonical feature policy used by encoder-decoder RLVR."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_json_dump
from nanochat.checkpoint_manager import find_last_step, load_checkpoint
from nanochat.common import compute_cleanup, compute_init
from nanochat.hydra_cli import parse_config_name_and_overrides
from scripts.train.conditional_encdec_rlvr import (
    evaluate_rlvr,
    feature_action_identity,
    load_rlvr_inputs,
)
from scripts.train.conditional_encdec_sft_grpo import (
    _assert_attention_impl,
    _configure_deterministic_algorithms,
    _reexec_for_deterministic_kernels,
    _sha256_file,
)
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder.config import validate_feature_prediction_resume
from steering.conditional.encoder_decoder.rlvr_runtime import (
    RLVR_GENERATION_VERSION,
    RLVR_META_KEY,
    RLVR_OBJECTIVE_VERSION,
    assert_complete_checkpoint,
)
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.eval.chat_eval_result_identity import TurnChatEvalCheckpointSnapshot
from steering.eval.turn_context_contract import canonical_turn_payload


def evaluate(cfg) -> None:
    _, rank, _, world_size, device = compute_init(str(cfg.device_type))
    device = t.device(device)
    try:
        with rank_synchronized_stage("RLVR evaluation admission", device=device):
            attention = _assert_attention_impl(device_type=device.type)
            _configure_deterministic_algorithms(device_type=device.type, attention_impl=attention)
            loaded = load_rlvr_inputs(cfg, device=device, rank=rank)
            checkpoint_identity = loaded.base_identity
            snapshot = None
            if cfg.rlvr_checkpoint_dir is not None:
                directory = Path(cfg.rlvr_checkpoint_dir)
                step = find_last_step(str(directory)) if cfg.rlvr_step is None else int(cfg.rlvr_step)
                meta_path = directory / f"meta_{step:06d}.json"
                meta_payload = meta_path.read_bytes()
                meta = json.loads(meta_payload)
                snapshot = TurnChatEvalCheckpointSnapshot.capture(meta_path, meta_payload, step)
                loop = meta.get(RLVR_META_KEY)
                if not isinstance(loop, dict) or loop.get("completed_optimizer_step") != step:
                    raise SFTGRPOError("evaluation checkpoint is not a completed RLVR step")
                identity = loop.get("resume_identity")
                if not isinstance(identity, dict) or identity.get("sha256") != hashlib.sha256(
                    canonical_turn_payload({key: value for key, value in identity.items() if key != "sha256"})
                ).hexdigest():
                    raise SFTGRPOError("RLVR evaluation checkpoint resume identity digest disagrees")
                assert_complete_checkpoint(directory, step, identity["world_size"])
                if identity["base_checkpoint"] != loaded.base_identity or identity["bundle"] != loaded.bundle.identity_sha256:
                    raise SFTGRPOError("RLVR evaluation base checkpoint or data bundle disagrees with training")
                for key, expected in {
                    "turn_contract": loaded.turn_contract.semantic,
                    "cadence": loaded.cadence,
                    "feature_action": feature_action_identity(loaded.policy),
                    "objective": RLVR_OBJECTIVE_VERSION,
                    "generation": RLVR_GENERATION_VERSION,
                }.items():
                    if identity.get(key) != expected:
                        raise SFTGRPOError(f"RLVR evaluation {key} disagrees with trained feature policy")
                for key in ("seed", "validation_seed", "temperature", "top_k", "mc_max_tokens", "gsm8k_max_tokens"):
                    if cfg[key] != identity["config"][key]:
                        raise SFTGRPOError(f"RLVR evaluation {key} disagrees with trained generation policy")
                state, _, loaded_meta = load_checkpoint(str(directory), step, device, load_optimizer=False, rank=rank)
                snapshot.assert_stable()
                if loaded_meta != meta:
                    raise SFTGRPOError("RLVR evaluation checkpoint metadata changed during admission")
                validate_feature_prediction_resume(meta, loaded.model.feature_pred_config)
                loaded.model.load_state_dict(state, strict=True, assign=False)
                loaded.model.assert_vocab_consistent()
                del state
                checkpoint_identity = {"step": step, "model_sha256": _sha256_file(directory / f"model_{step:06d}.pt"),
                                       "resume_identity_sha256": identity["sha256"]}
                snapshot.assert_stable()
            elif cfg.rlvr_step is not None:
                raise SFTGRPOError("rlvr_step requires rlvr_checkpoint_dir")
            output = Path(cfg.output_path)
            if output.exists():
                raise SFTGRPOError("RLVR evaluation requires a fresh output_path")
        metrics, samples = evaluate_rlvr(loaded, cfg=cfg, rank=rank, world_size=world_size)
        with rank_synchronized_stage("RLVR evaluation publication", device=device):
            loaded.cache_snapshot.assert_stable()
            if snapshot is not None:
                snapshot.assert_stable()
            if rank == 0:
                atomic_json_dump({"config": OmegaConf.to_container(cfg, resolve=True),
                                  "checkpoint": checkpoint_identity, "bundle_sha256": loaded.bundle.identity_sha256,
                                  "metrics": metrics, "samples": samples}, output)
    finally:
        compute_cleanup()


def main() -> None:
    _reexec_for_deterministic_kernels()
    name, overrides = parse_config_name_and_overrides(sys.argv[1:], default_config_name="encdec_rlvr")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs" / "eval"), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    evaluate(cfg)


if __name__ == "__main__":
    main()
