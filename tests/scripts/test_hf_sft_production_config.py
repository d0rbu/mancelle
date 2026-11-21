"""Effective production defaults and the single-launcher loss-mask override."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from steering.sft.hf_contract import HFAttentionImplementation, HFDistributedImplementation
from steering.sft.hf_decoder import validate_decoder_sft_config
from steering.sft.loss_mask import LossMaskMode
from tasks.mixture_names import SFTMixture

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "training"


def test_production_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEERED_JSONL", "/snapshot/data/messages_steered.jsonl")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="chat_sft_pythia_2_8b_standard_steered")
    validate_decoder_sft_config(cfg, world_size=4)
    assert cfg.mixture == SFTMixture.STANDARD_STEERED.value
    assert cfg.steered_jsonl == "/snapshot/data/messages_steered.jsonl"
    assert cfg.sft_loss_mask == LossMaskMode.FULL_SEQUENCE.value
    assert (cfg.max_seq_len, cfg.total_batch_size, cfg.device_batch_size) == (2048, 524288, 1)
    assert cfg.num_iterations == 1102
    assert (cfg.eval_every, cfg.save_every, cfg.huggingface.restart_save_every) == (150, 100, 100)
    assert cfg.dry_run is False
    assert cfg.huggingface.validate_export is True
    assert cfg.huggingface.gradient_checkpointing is True
    assert cfg.huggingface.torch_compile is False
    assert cfg.huggingface.attention_implementation == HFAttentionImplementation.SDPA.value
    assert (
        cfg.huggingface.distributed_implementation
        == HFDistributedImplementation.DIST_MUON_ADAMW.value
    )
    assert cfg.huggingface.learning_rate == 2e-5
    assert list(cfg.huggingface.adam_betas) == [0.8, 0.95]
    assert cfg.huggingface.adam_eps == 1e-10
    assert cfg.weight_decay == 0
    assert cfg.init_lr_frac == 1
    assert cfg.grad_clip_max_norm is None
    assert cfg.huggingface.model_id == "EleutherAI/pythia-2.8b"
    assert cfg.huggingface.revision == "2a259cdd96a4beb1cdf467512e3904197345f6a9"
    assert cfg.total_batch_size // (4 * cfg.device_batch_size * cfg.max_seq_len) == 64


def test_loss_mask_and_save_cadence_are_config_overrides() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="chat_sft_pythia_2_8b_standard_steered",
            overrides=["sft_loss_mask=completion_only", "save_every=200"],
        )
    assert cfg.sft_loss_mask == LossMaskMode.COMPLETION_ONLY.value
    assert cfg.huggingface.restart_save_every == cfg.save_every == 200
