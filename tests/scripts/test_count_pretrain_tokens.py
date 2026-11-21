"""Unit tests for ``scripts.preprocess.count_tokens.count_pretrain_tokens``.

The pretrain-token estimate is a pure arithmetic function of the
config; we exercise it with a tiny ``depth`` so the meta-device GPT
build is fast.
"""

from __future__ import annotations

import math

from omegaconf import DictConfig

from scripts.preprocess.count_tokens import (
    build_model_meta,
    count_pretrain_tokens,
)


def _cfg(
    depth: int = 12,
    target_ratio: float = 8.25,
    aspect_ratio: int | float = 64,
) -> DictConfig:
    return DictConfig(
        {
            "depth": depth,
            "aspect_ratio": aspect_ratio,
            "head_dim": 128,
            "max_seq_len": 2048,
            "window_pattern": "SSSL",
            "target_param_data_ratio": target_ratio,
        }
    )


def test_build_model_meta_returns_meta_device_model():
    cfg = _cfg(depth=12)
    model = build_model_meta(cfg, vocab_size=32_768)
    # Meta-device model has no allocated tensors; .device for a parameter
    # should be 'meta'.
    p = next(model.parameters())
    assert p.device.type == "meta"


def test_build_model_meta_dimensions_match_formula():
    """`base_dim = depth * aspect_ratio`; `model_dim = round_up(base_dim, head_dim)`."""
    cfg = _cfg(depth=12)
    model = build_model_meta(cfg, vocab_size=32_768)
    base_dim = cfg.depth * cfg.aspect_ratio
    expected_model_dim = (
        (base_dim + cfg.head_dim - 1) // cfg.head_dim
    ) * cfg.head_dim
    expected_n_head = expected_model_dim // cfg.head_dim
    assert model.config.n_embd == expected_model_dim
    assert model.config.n_head == expected_n_head
    assert model.config.n_kv_head == expected_n_head
    assert model.config.n_layer == cfg.depth


def test_build_model_meta_accepts_fractional_aspect_ratio():
    cfg = _cfg(depth=27, aspect_ratio=1664 / 27)
    model = build_model_meta(cfg, vocab_size=32_768)
    assert model.config.n_embd == 1664
    assert model.config.n_head == 13
    assert model.config.n_kv_head == 13
    assert type(model.config.n_embd) is int
    assert type(model.config.n_head) is int


def test_count_pretrain_tokens_is_consistent_with_arithmetic():
    cfg = _cfg(depth=12)
    res = count_pretrain_tokens(cfg, vocab_size=32_768)

    # Derived value sanity checks (consistency rather than absolute pin):
    assert res["target_tokens"] == int(
        cfg.target_param_data_ratio * res["num_scaling_params"]
    )
    # total_tokens should be batch_size * num_iterations and at most target_tokens.
    assert res["total_tokens"] == res["total_batch_size"] * res["num_iterations"]
    assert res["total_tokens"] <= res["target_tokens"]
    # Batch size is power-of-two by construction.
    bs = res["total_batch_size"]
    assert bs > 0 and (bs & (bs - 1)) == 0


def test_count_pretrain_tokens_param_counts_keys():
    cfg = _cfg(depth=12)
    res = count_pretrain_tokens(cfg, vocab_size=32_768)
    assert "transformer_matrices" in res["param_counts"]
    assert "lm_head" in res["param_counts"]


def test_count_pretrain_tokens_scales_with_depth():
    """A deeper model should have more scaling params and more target tokens."""
    res_d12 = count_pretrain_tokens(_cfg(depth=12), vocab_size=32_768)
    res_d24 = count_pretrain_tokens(_cfg(depth=24), vocab_size=32_768)
    assert res_d24["num_scaling_params"] > res_d12["num_scaling_params"]
    assert res_d24["target_tokens"] > res_d12["target_tokens"]


def test_count_pretrain_tokens_scales_with_target_ratio():
    """`target_tokens` is linear in `target_param_data_ratio`."""
    res_low = count_pretrain_tokens(_cfg(depth=12, target_ratio=4.0), vocab_size=32_768)
    res_hi = count_pretrain_tokens(_cfg(depth=12, target_ratio=8.0), vocab_size=32_768)
    # Expect approximately 2x target_tokens (subject to int truncation).
    ratio = res_hi["target_tokens"] / res_low["target_tokens"]
    assert math.isclose(ratio, 2.0, rel_tol=0.01)
