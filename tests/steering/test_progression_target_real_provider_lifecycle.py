"""PR-D: target-real provider lifecycle.

Pinned invariants:
  - One provider constructed per (cache_dir, model_max_seq_len).
  - Two configs sharing a cache_dir share the provider instance.
  - Provider construction happens at sweep startup, not per checkpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from scripts.train.training_progression import (
    _context_provider_for_cfg,
    _target_real_providers_for_cfg,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.target_real_runner import parse_target_real_configs_from_cfg


def test_returns_empty_dict_when_no_target_real_configs():
    cfg = OmegaConf.create({"conditioning": {"target_real_configs": []}})
    assert (
        _target_real_providers_for_cfg(
            cfg,
            model_max_seq_len=512,
            target_real_configs=[],
        )
        == {}
    )


@patch("scripts.train.training_progression.SafetensorsContextFeatureProvider")
def test_one_provider_per_cache_dir(mock_provider):
    mock_provider.side_effect = lambda *args, **kwargs: MagicMock()
    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": [
                {"name": "lockstep_a", "mode": "lockstep", "cache_dir": "/tmp/a"},
                {"name": "pooled_a", "mode": "pooled", "cache_dir": "/tmp/a"},  # SAME cache_dir
                {"name": "lockstep_b", "mode": "lockstep", "cache_dir": "/tmp/b"},
            ],
            "identity": {
                "model_id": "x", "sae_id": "y", "layer": 8,
                "pooling": "max_pool", "feature_selection_hash": "a" * 64,
            },
            "n_sae_features": 16, "num_buckets": 4,
        }
    })
    out = _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=512,
        target_real_configs=parse_target_real_configs_from_cfg(
            cfg.conditioning,
            model_max_seq_len=512,
        ),
    )
    assert set(out.keys()) == {"lockstep_a", "pooled_a", "lockstep_b"}
    # Configs sharing /tmp/a share the same provider instance.
    assert out["lockstep_a"] is out["pooled_a"]
    # Different cache_dir -> different instance.
    assert out["lockstep_a"] is not out["lockstep_b"]


@patch("scripts.train.training_progression.SafetensorsContextFeatureProvider")
def test_static_provider_receives_checkpoint_vocab_and_metadata(mock_provider):
    expected_vocab = FeatureVocab(16, (1, 2, 4))
    expected_contract = {
        "feature_mask_canonical": {"mode": "whitelist", "ids": [1, 2, 4]},
        "feature_clusters": None,
    }
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_cache_dir": "/tmp/cache",
                "identity": {
                    "model_id": "x",
                    "sae_id": "y",
                    "layer": 8,
                    "pooling": "max_pool",
                    "feature_selection_hash": "a" * 64,
                },
            }
        }
    )

    _context_provider_for_cfg(
        cfg,
        model_max_seq_len=512,
        expected_feature_vocab=expected_vocab,
        expected_contract_metadata=expected_contract,
    )

    assert mock_provider.call_args.kwargs["expected_feature_vocab"] == expected_vocab
    assert mock_provider.call_args.kwargs["expected_metadata"] == {
        "model_max_seq_len": 512,
        **expected_contract,
    }


@patch("scripts.train.training_progression.SafetensorsContextFeatureProvider")
def test_target_real_provider_receives_checkpoint_vocab_and_metadata(mock_provider):
    mock_provider.return_value = MagicMock()
    expected_vocab = FeatureVocab(16, (1, 2, 4))
    expected_contract = {"feature_mask_canonical": {"mode": "whitelist", "ids": [1, 2, 4]}}
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "target_real_configs": [
                    {"name": "pooled", "mode": "pooled", "cache_dir": "/tmp/a"}
                ],
                "identity": {
                    "model_id": "x",
                    "sae_id": "y",
                    "layer": 8,
                    "pooling": "max_pool",
                    "feature_selection_hash": "a" * 64,
                },
                "n_sae_features": 16,
                "num_buckets": 4,
            }
        }
    )

    _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=512,
        target_real_configs=parse_target_real_configs_from_cfg(
            cfg.conditioning,
            model_max_seq_len=512,
        ),
        expected_feature_vocab=expected_vocab,
        expected_contract_metadata=expected_contract,
    )

    assert mock_provider.call_args.kwargs["expected_feature_vocab"] == expected_vocab
    assert mock_provider.call_args.kwargs["expected_metadata"] == {
        "model_max_seq_len": 512,
        **expected_contract,
    }
