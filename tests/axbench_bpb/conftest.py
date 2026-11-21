# tests/axbench_bpb/conftest.py
"""Shared fixtures for axbench-bpb tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT


@pytest.fixture
def axbench_bpb_test_root() -> Path:
    """Root of axbench-bpb test directory (for locating fixtures)."""
    return Path(__file__).parent


@pytest.fixture
def tiny_decoder_only(axbench_bpb_test_root):
    """Load the tiny decoder_only checkpoint; return (model, config)."""
    ckpt_path = axbench_bpb_test_root / "fixtures" / "tiny_decoder_only" / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**state["config"])
    with torch.device("meta"):
        model = GPT(cfg, pad_vocab_size_to=state["pad_vocab_size_to"])
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.eval()
    return model, cfg


@pytest.fixture
def tiny_encdec(axbench_bpb_test_root):
    """Load the tiny encdec checkpoint; return (model, dec_cfg, enc_cfg, xattn_cfg)."""
    ckpt_path = axbench_bpb_test_root / "fixtures" / "tiny_encdec" / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec_cfg = GPTConfig(**state["dec_config"])
    enc_cfg = EncoderConfig(**state["enc_config"])
    xattn_cfg = CrossAttentionConfig(**state["xattn_config"])
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config=dec_cfg, enc_config=enc_cfg, xattn_config=xattn_cfg,
            n_sae_features=state["n_sae_features"],
            num_buckets=state["num_buckets"],
            pad_vocab_size_to=state["pad_vocab_size_to"],
        )
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.eval()
    return model, dec_cfg, enc_cfg, xattn_cfg
