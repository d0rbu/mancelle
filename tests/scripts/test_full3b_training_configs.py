"""Exact contracts for the local full3B c2048 production presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from scripts.train.conditional_encdec_train import _checkpoint_user_config
from steering.conditional.encoder_decoder.dataloader import (
    resolve_train_prefetch_batches,
    resolve_train_shard_lookahead,
)
from steering.node_shard_cache import resolve_train_node_cache
from steering.refined_types import PositiveInt
from steering.remote_training import _semantic_config_v2

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")
BASE_CONFIG = "full3b_clustered_c2048"
LONG_CONFIG = "full3b_clustered_c2048_long"
LONG_W31_CONFIG = "full3b_clustered_c2048_long_w31"
LONG_W32_CONFIG = "full3b_clustered_c2048_long_w32_bf16_b14"

GLOBAL_BATCH_TOKENS = 1_048_576
SCALING_PARAMETERS = 1_769_543_414
BASE_UPDATES = 168_756
BASE_TOKENS = 176_953_491_456
LONG_UPDATES = 500_000
LONG_TOKENS = 524_288_000_000
W31_BATCH_TOKENS = 1_777_664
W32_BATCH_TOKENS = 1_835_008
W31_W32_UPDATES = 250_000
W31_TOKENS = 444_416_000_000
W32_TOKENS = 458_752_000_000
PACKED_ROWS = 26_278_580
TOKENS_PER_EPOCH = 53_818_531_840


def test_full3b_base_pins_exact_production_geometry(hydra_compose) -> None:
    cfg = hydra_compose(CONFIG_DIR, BASE_CONFIG)

    assert cfg.run == BASE_CONFIG
    assert cfg.model_tag == BASE_CONFIG
    assert cfg.depth == 27
    assert cfg.aspect_ratio == pytest.approx(61.62962962962963)
    assert cfg.head_dim == 128
    assert cfg.max_seq_len == 2048
    assert cfg.device_batch_size == 16
    assert cfg.total_batch_size == GLOBAL_BATCH_TOKENS
    assert cfg.total_batch_size == cfg.device_batch_size * cfg.max_seq_len * 32
    assert cfg.num_iterations == -1
    assert cfg.target_param_data_ratio == 100.0
    assert cfg.fp8 is True
    assert cfg.attn_impl == "fa4"
    assert cfg.train_prefetch_batches == 64
    assert cfg.train_shard_lookahead == 4
    assert cfg.train_node_cache_dir is None
    assert cfg.train_node_cache_max_bytes is None
    assert cfg.train_node_cache_reserve_bytes == 3_000_000_000_000
    assert cfg.eval_every == 5000
    assert cfg.save_every == 5000
    assert cfg.core_metric_every == 0
    assert cfg.sample_every == 0
    assert cfg.steering_eval_every == 0


def test_full3b_long_changes_only_identity_and_horizon(hydra_compose) -> None:
    base = hydra_compose(CONFIG_DIR, BASE_CONFIG)
    long = hydra_compose(CONFIG_DIR, LONG_CONFIG)

    assert long.run == LONG_CONFIG
    assert long.model_tag == LONG_CONFIG
    assert long.num_iterations == LONG_UPDATES
    assert long.num_iterations * long.total_batch_size == LONG_TOKENS
    assert long.target_param_data_ratio == pytest.approx(LONG_TOKENS / SCALING_PARAMETERS)

    allowed_differences = {
        "run",
        "model_tag",
        "num_iterations",
        "target_param_data_ratio",
    }
    base_mapping = cast(dict[str, Any], OmegaConf.to_container(base, resolve=True))
    long_mapping = cast(dict[str, Any], OmegaConf.to_container(long, resolve=True))
    assert {
        key
        for key in set(base_mapping) | set(long_mapping)
        if base_mapping.get(key) != long_mapping.get(key)
    } == allowed_differences


def test_full3b_long_w32_changes_only_identity_and_world_scaled_batch(
    hydra_compose,
) -> None:
    w31 = hydra_compose(CONFIG_DIR, LONG_W31_CONFIG)
    w32 = hydra_compose(CONFIG_DIR, LONG_W32_CONFIG)

    assert w31.fp8 is False
    assert w32.fp8 is False
    assert w31.device_batch_size == w32.device_batch_size == 14
    assert w31.max_seq_len == w32.max_seq_len == 2048
    assert w31.num_iterations == w32.num_iterations == W31_W32_UPDATES
    assert w31.total_batch_size == W31_BATCH_TOKENS
    assert w32.total_batch_size == W32_BATCH_TOKENS
    assert w31.total_batch_size == 31 * 14 * 2048 * 2
    assert w32.total_batch_size == 32 * 14 * 2048 * 2
    assert w31.num_iterations * w31.total_batch_size == W31_TOKENS
    assert w32.num_iterations * w32.total_batch_size == W32_TOKENS
    assert w31.train_node_cache_dir is None
    assert w31.train_node_cache_max_bytes is None
    assert w32.train_node_cache_dir == "/scratch/henryac/ngpt-shard-cache"
    assert w32.train_node_cache_max_bytes == 3_000_000_000_000

    w31_mapping = cast(dict[str, Any], OmegaConf.to_container(w31, resolve=True))
    w32_mapping = cast(dict[str, Any], OmegaConf.to_container(w32, resolve=True))
    assert {
        key
        for key in set(w31_mapping) | set(w32_mapping)
        if w31_mapping.get(key) != w32_mapping.get(key)
    } == {
        "run",
        "model_tag",
        "total_batch_size",
        "train_node_cache_dir",
        "train_node_cache_max_bytes",
    }


def test_full3b_long_w32_uses_canonical_batch_lr_scaling(hydra_compose) -> None:
    w31 = hydra_compose(CONFIG_DIR, LONG_W31_CONFIG)
    w32 = hydra_compose(CONFIG_DIR, LONG_W32_CONFIG)

    expected_ratio = (32 / 31) ** 0.5
    assert (w32.total_batch_size / w31.total_batch_size) ** 0.5 == pytest.approx(expected_ratio)
    for field in (
        "embedding_lr",
        "unembedding_lr",
        "matrix_lr",
        "scalar_lr",
        "encoder_embedding_lr",
        "encoder_matrix_lr",
        "xattn_matrix_lr",
        "weight_decay",
    ):
        assert getattr(w32, field) == getattr(w31, field)


def test_documented_token_and_epoch_arithmetic_is_exact() -> None:
    assert 100 * SCALING_PARAMETERS // GLOBAL_BATCH_TOKENS == BASE_UPDATES
    assert BASE_UPDATES * GLOBAL_BATCH_TOKENS == BASE_TOKENS
    assert PACKED_ROWS * 2048 == TOKENS_PER_EPOCH
    assert BASE_TOKENS / TOKENS_PER_EPOCH == pytest.approx(3.2879657881)
    assert LONG_TOKENS / TOKENS_PER_EPOCH == pytest.approx(9.7417744794)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 7), (0, 0), (1, 1), (32, 32), (64, 64)],
)
def test_train_prefetch_depth_resolution(raw: object, expected: int) -> None:
    assert resolve_train_prefetch_batches(raw, PositiveInt.parse(7)) == expected


@pytest.mark.parametrize("raw", [False, True, -1, 1.5, "32"])
def test_train_prefetch_depth_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(TypeError):
        resolve_train_prefetch_batches(raw, PositiveInt.parse(7))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 0), (0, 0), (1, 1), (4, 4), (64, 64)],
)
def test_train_shard_lookahead_resolution(raw: object, expected: int) -> None:
    assert resolve_train_shard_lookahead(raw) == expected


@pytest.mark.parametrize("raw", [False, True, -1, 1.5, "4"])
def test_train_shard_lookahead_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(TypeError, match="Could not parse.*NonNegativeInt"):
        resolve_train_shard_lookahead(raw)


def test_train_node_cache_config_resolves(hydra_compose) -> None:
    cfg = hydra_compose(CONFIG_DIR, LONG_W32_CONFIG)
    resolved_cache = resolve_train_node_cache(
        cfg.train_node_cache_dir,
        cfg.train_node_cache_max_bytes,
        cfg.train_node_cache_reserve_bytes,
    )
    assert resolved_cache is not None
    assert resolved_cache.dir == Path("/scratch/henryac/ngpt-shard-cache")
    assert resolved_cache.max_bytes == 3_000_000_000_000
    assert resolved_cache.reserve_bytes == 3_000_000_000_000


@pytest.mark.parametrize(
    ("config_name", "key"),
    [
        (BASE_CONFIG, "train_prefetch_batches"),
        (BASE_CONFIG, "train_shard_lookahead"),
        (LONG_W32_CONFIG, "train_node_cache_dir"),
        (LONG_W32_CONFIG, "train_node_cache_max_bytes"),
        (LONG_W32_CONFIG, "train_node_cache_reserve_bytes"),
    ],
)
def test_loader_execution_config_is_omitted_from_identity(
    hydra_compose,
    config_name: str,
    key: str,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    assert key not in _checkpoint_user_config(cfg)
    resolved = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    semantic = _semantic_config_v2(resolved, surface="training_config")
    assert key not in semantic
