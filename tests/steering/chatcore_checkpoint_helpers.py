"""Production checkpoint layout with real resolution and tiny weight loading."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch as t

from nanochat import checkpoint_manager
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY

PRODUCTION_TAG = "full3b_clustered_c2048_long_w32_bf16_b14_fp"
SFT_LEAF = "standard_sft_conditioned_checkpoints"
PRODUCTION_MODEL_TAG = f"{PRODUCTION_TAG}/{SFT_LEAF}"


def stage_checkpoint(tmp_path, monkeypatch, meta, model, tokenizer, *, real_loader=False):
    base = tmp_path / "nanochat"
    leaf = base / "conditional_encdec_checkpoints" / PRODUCTION_MODEL_TAG
    leaf.mkdir(parents=True)
    tokenizer_dir = tmp_path / "staged_tokenizer"
    tokenizer_dir.mkdir()
    (base / "tokenizer").symlink_to(tokenizer_dir, target_is_directory=True)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base))
    if real_loader:
        meta["model_config"] = asdict(model.dec_config)
        del meta["model_config"]["window_pattern"]
        meta["user_config"].update(negative_selection=None, feature_prediction=None)
        ancestor = {"user_config": {
            "encoder": asdict(model.enc_config), "cross_attention": asdict(model.xattn_config),
            "num_buckets": meta["num_buckets"],
            "negative_selection": meta[ONLINE_SAE_LINEAGE_META_KEY]["negative_selection"],
            "feature_prediction": asdict(FeaturePredictionConfig()),
        }}
        (leaf.parent / "meta_000100.json").write_text(json.dumps(ancestor))
        t.save(model.state_dict(), leaf.parent / "model_000100.pt")
        monkeypatch.setattr(tokenizer, "get_vocab_size", lambda: model.dec_config.vocab_size, raising=False)
        monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: tokenizer)
    meta_path = leaf / "meta_001116.json"
    meta_path.write_text(json.dumps(meta))
    weights = model.state_dict() if real_loader else {"step": 1116, "state_dict": model.state_dict()}
    t.save(weights, leaf / "model_001116.pt")
    if "fixed_self_cond_cadence_provenance" in meta:
        model_path = leaf / "model_001116.pt"
        meta["fixed_self_cond_cadence_provenance"]["original_checkpoint"] = [
            {"path": str(model_path), "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
        ]
        meta_path.write_text(json.dumps(meta))
    loaded = []
    real_build = checkpoint_manager.build_encdec_model

    def build_tiny(checkpoint_dir, step, device, phase, *, eval_conditioning):
        directory = Path(checkpoint_dir)
        if real_loader:
            result = real_build(checkpoint_dir, step, device, phase, eval_conditioning=eval_conditioning)
            loaded.append((directory, step))
            return result
        # These opens intentionally fail for the doubled-leaf source chain.
        snapshot = json.loads((directory / f"meta_{step:06d}.json").read_bytes())
        weights = t.load(directory / f"model_{step:06d}.pt", map_location=device, weights_only=True)
        assert weights["step"] == step
        model.load_state_dict(weights["state_dict"])
        eval_conditioning.assert_loadable(model.feature_vocab, checkpoint_dir=str(directory))
        loaded.append((directory, step))
        return model, tokenizer, snapshot

    monkeypatch.setattr(checkpoint_manager, "build_encdec_model", build_tiny)
    return meta_path, loaded
