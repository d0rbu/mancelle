"""Shared Campaign 2 config and repository-HF fixtures."""

import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from steering.sft.hf_contract import HF_SFT_FORMAT_VERSION

ROOT = Path(__file__).resolve().parents[2]


def campaign_config(tmp_path: Path, **values: Any) -> Any:
    with initialize_config_dir(config_dir=str(ROOT / "configs/eval"), version_base=None):
        cfg = compose(config_name="campaign2", overrides=[
            f"checkpoint_dir={tmp_path}/checkpoint", f"output_dir={tmp_path}/output",
        ])
    return OmegaConf.merge(cfg, values)


def export_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "step_000007"
    root.mkdir()
    meta = {
        "format": "huggingface_standard_sft", "format_version": HF_SFT_FORMAT_VERSION,
        "backend": "huggingface", "family": "pythia", "resumable": False,
        "completed_optimizer_step": 7, "mixture": "standard_steered", "sft_loss_mask": "completion_only",
        "source": {"model_id": "EleutherAI/pythia-2.8b", "revision": "a" * 40,
                   "architecture": "GPTNeoXForCausalLM", "model_type": "gpt_neox"},
        "output": {"architecture": "GPTNeoXForCausalLM", "model_type": "gpt_neox"},
        "precision": {"export": "bfloat16"}, "batch": {"max_seq_len": 2048},
        "vocabulary": {"model": 50304, "tokenizer": 50277},
        "attention_implementation": "sdpa", "chat_template_sha256": "b" * 64,
    }
    (root / "sft_metadata.json").write_text(json.dumps(meta))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"):
        (root / name).write_text("fixture")
    return root
