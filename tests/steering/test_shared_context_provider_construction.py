"""Shared full-context provider validation for normal encoder-decoder evals."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from steering.eval.target_real_runner import build_shared_context_feature_provider


@pytest.mark.parametrize("eval_name", ["eval", "dclm_eval"])
def test_raises_when_cache_dir_unset(eval_name: str) -> None:
    cfg = OmegaConf.create({"conditioning": {"context_cache_dir": None}})
    with pytest.raises(ValueError, match="conditioning.context_cache_dir"):
        build_shared_context_feature_provider(cfg, eval_name=eval_name)


@pytest.mark.parametrize("eval_name", ["eval", "dclm_eval"])
def test_raises_when_conditioning_block_missing(eval_name: str) -> None:
    cfg = OmegaConf.create({})
    with pytest.raises(ValueError, match="conditioning block"):
        build_shared_context_feature_provider(cfg, eval_name=eval_name)


@pytest.mark.parametrize("eval_name", ["eval", "dclm_eval"])
def test_raises_when_cache_dir_set_but_identity_missing(eval_name: str) -> None:
    cfg = OmegaConf.create(
        {
            "conditioning": {"context_cache_dir": "/tmp/nonexistent"},
            "model_max_seq_len": 512,
        }
    )
    with pytest.raises(ValueError, match="conditioning.identity"):
        build_shared_context_feature_provider(cfg, eval_name=eval_name)


@pytest.mark.parametrize("eval_name", ["eval", "dclm_eval"])
def test_raises_when_identity_layer_is_null(eval_name: str) -> None:
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_cache_dir": "/tmp/x",
                "identity": {
                    "model_id": "gemma-2-2b",
                    "sae_id": "test",
                    "layer": None,
                    "pooling": "max_pool",
                },
            },
            "model_max_seq_len": 512,
        }
    )
    with pytest.raises(ValueError, match="layer"):
        build_shared_context_feature_provider(cfg, eval_name=eval_name)


@pytest.mark.parametrize("eval_name", ["eval", "dclm_eval"])
def test_raises_when_identity_layer_is_string(eval_name: str) -> None:
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_cache_dir": "/tmp/x",
                "identity": {
                    "model_id": "gemma-2-2b",
                    "sae_id": "test",
                    "layer": "not-an-int",
                    "pooling": "max_pool",
                },
            },
            "model_max_seq_len": 512,
        }
    )
    with pytest.raises(ValueError, match="layer"):
        build_shared_context_feature_provider(cfg, eval_name=eval_name)


@pytest.mark.parametrize("eval_name", ["eval", "dclm_eval"])
def test_raises_when_model_id_missing(eval_name: str) -> None:
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_cache_dir": "/tmp/x",
                "identity": {
                    "sae_id": "test",
                    "layer": 8,
                    "pooling": "max_pool",
                },
            },
            "model_max_seq_len": 512,
        }
    )
    with pytest.raises(ValueError, match="model_id"):
        build_shared_context_feature_provider(cfg, eval_name=eval_name)
