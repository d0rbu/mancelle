"""Full-context startup validation for base_eval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scripts.eval.base_eval as base_eval_module
from steering.eval.feature_config import FeatureConfig
from steering.eval.target_conditioning import LockstepTargetRealConfig
from steering.eval.target_real_runner import require_shared_context_cache_dir
from tests.conftest import MockTokenizer, _make_encdec, _make_gpt

_BASE_EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval"


def test_base_eval_default_feature_configs_are_empty() -> None:
    with initialize_config_dir(config_dir=str(_BASE_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base_eval", overrides=["source=base"])

    fcs = OmegaConf.to_container(cfg.feature_configs, resolve=True)
    assert fcs == []


def test_base_eval_sampling_block_has_no_unconditioned_settings() -> None:
    with initialize_config_dir(config_dir=str(_BASE_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base_eval", overrides=["source=base"])

    assert cfg.sampling.per_config_max_new_tokens == 16
    assert cfg.sampling.per_config_temperature == 0.0
    assert not any(str(key).startswith("unconditioned") for key in cfg.sampling)


def test_full_context_validation_requires_cache_dir() -> None:
    cfg = OmegaConf.create({"conditioning": {"context_cache_dir": None}})
    with pytest.raises(ValueError, match="context_cache_dir"):
        require_shared_context_cache_dir(cfg, eval_name="base_eval")


def test_full_context_validation_passes_with_cache_dir() -> None:
    cfg = OmegaConf.create({"conditioning": {"context_cache_dir": "/cache"}})
    require_shared_context_cache_dir(cfg, eval_name="base_eval")


def test_base_eval_rejects_empty_static_feature_config_for_encdec() -> None:
    cfg = OmegaConf.create(
        {
            "eval": "core",
            "device_type": "cpu",
            "hf_path": None,
            "source": "encdec",
            "model_tag": "d12",
            "step": None,
            "feature_configs": "ignored",
        }
    )
    meta = {
        "model_config": {"sequence_len": 16},
        "step": 0,
        "checkpoint_dir": "/tmp/checkpoint",
        "user_config": {},
    }

    with (
        patch.object(
            base_eval_module,
            "compute_init",
            return_value=(False, 0, 0, 1, "cpu"),
        ),
        patch.object(
            base_eval_module,
            "load_model",
            return_value=(_make_encdec(), MockTokenizer(), meta),
        ),
        patch.object(
            base_eval_module,
            "parse_feature_configs",
            return_value=[FeatureConfig(name="context_only", features={})],
        ),
        pytest.raises(ValueError, match="context_only"),
    ):
        base_eval_module.main.__wrapped__(cfg)


@pytest.mark.parametrize("eval_modes", ["bpb", "core,bpb"])
def test_base_eval_encdec_bpb_allows_empty_feature_configs(
    tmp_path,
    eval_modes: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "eval": eval_modes,
            "device_type": "cpu",
            "hf_path": None,
            "source": "encdec",
            "model_tag": "d12",
            "step": None,
            "feature_configs": [],
            "target_real_configs": [],
            "device_batch_size": 1,
            "split_tokens": 16,
            "model_max_seq_len": 16,
            "next_segment_conditioning_offset": False,
            "conditioning": {
                "n_sae_features": 64,
                "num_buckets": 2,
                "feature_selection": {
                    "threshold_type": "feature_percentile",
                    "threshold": 0.95,
                },
                "negative_selection": None,
            },
            "bpb": {
                "val_packed_row_dir": str(tmp_path),
                "selection_hash": None,
            },
        }
    )
    meta = {
        "model_config": {"sequence_len": 16},
        "step": 0,
        "checkpoint_dir": str(tmp_path / "checkpoint"),
        "user_config": {},
    }
    pipeline = SimpleNamespace(
        feature_mask=None,
        feature_clustering=None,
        effective_buckets=2,
    )
    metrics = SimpleNamespace(bpb_conditioned=1.25, avg_features_per_segment=3.5)
    report = SimpleNamespace(log=lambda *args, **kwargs: None)

    with (
        patch.object(
            base_eval_module,
            "compute_init",
            return_value=(False, 0, 0, 1, "cpu"),
        ),
        patch.object(
            base_eval_module,
            "load_model",
            return_value=(_make_encdec(), MockTokenizer(), meta),
        ),
        patch.object(base_eval_module, "parse_feature_configs", return_value=[]),
        patch.object(
            base_eval_module,
            "ensure_core_eval_has_work",
            side_effect=AssertionError("BPB-only eval must not need CORE configs"),
        ),
        patch.object(base_eval_module, "get_token_bytes", return_value=[]),
        patch.object(base_eval_module, "build_feature_pipeline_from_cfg", return_value=pipeline),
        patch.object(
            base_eval_module,
            "_resolve_bpb_selection_dir",
            return_value=None,
        ) as resolve_selection_dir,
        patch.object(
            base_eval_module,
            "precomputed_data_loader",
            return_value=iter(()),
        ) as precomputed_loader,
        patch.object(
            base_eval_module,
            "evaluate_encdec_bpb_conditioned",
            return_value=metrics,
        ),
        patch.object(base_eval_module, "get_report", return_value=report),
        patch.object(base_eval_module, "compute_cleanup"),
    ):
        base_eval_module.main.__wrapped__(cfg)

    resolve_selection_dir.assert_called_once()
    assert resolve_selection_dir.call_args.kwargs["feature_selection_raw"] == {
        "threshold_type": "feature_percentile",
        "threshold": 0.95,
    }
    loader_kwargs = precomputed_loader.call_args.kwargs
    assert "n_sae_features" not in loader_kwargs
    assert "fp_target_layout" not in loader_kwargs


def test_base_eval_encdec_bpb_rejects_missing_selection_before_dir_resolution(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "eval": "bpb",
            "device_type": "cpu",
            "hf_path": None,
            "source": "encdec",
            "model_tag": "d12",
            "step": None,
            "feature_configs": [],
            "target_real_configs": [],
            "device_batch_size": 1,
            "split_tokens": 16,
            "model_max_seq_len": 16,
            "next_segment_conditioning_offset": False,
            "conditioning": {
                "n_sae_features": 64,
                "num_buckets": 2,
                "feature_selection": None,
                "negative_selection": None,
            },
            "bpb": {
                "val_packed_row_dir": str(tmp_path),
                "selection_hash": None,
            },
        }
    )
    meta = {
        "model_config": {"sequence_len": 16},
        "step": 0,
        "checkpoint_dir": str(tmp_path / "checkpoint"),
        "user_config": {},
    }

    with (
        patch.object(
            base_eval_module,
            "compute_init",
            return_value=(False, 0, 0, 1, "cpu"),
        ),
        patch.object(
            base_eval_module,
            "load_model",
            return_value=(_make_encdec(), MockTokenizer(), meta),
        ),
        patch.object(base_eval_module, "parse_feature_configs", return_value=[]),
        patch.object(base_eval_module, "get_token_bytes", return_value=[]),
        patch.object(base_eval_module, "_resolve_bpb_selection_dir") as resolver,
        pytest.raises(ValueError, match="conditioning.feature_selection"),
    ):
        base_eval_module.main.__wrapped__(cfg)

    resolver.assert_not_called()


def test_base_eval_rejects_decoder_only_feature_config_before_token_bytes() -> None:
    cfg = OmegaConf.create(
        {
            "eval": "core",
            "device_type": "cpu",
            "hf_path": None,
            "source": "base",
            "model_tag": "d12",
            "step": None,
            "feature_configs": "ignored",
            "conditioning": {},
        }
    )
    meta = {
        "model_config": {"sequence_len": 16},
        "step": 0,
        "checkpoint_dir": "/tmp/checkpoint",
        "user_config": {},
    }

    with (
        patch.object(
            base_eval_module,
            "compute_init",
            return_value=(False, 0, 0, 1, "cpu"),
        ),
        patch.object(
            base_eval_module,
            "load_model",
            return_value=(_make_gpt(), MockTokenizer(), meta),
        ),
        patch.object(
            base_eval_module,
            "parse_feature_configs",
            return_value=[FeatureConfig(name="real_features", features={1: 2})],
        ),
        patch.object(
            base_eval_module,
            "get_token_bytes",
            side_effect=AssertionError("token bytes must load after startup validation"),
        ),
        pytest.raises(ValueError, match="EncoderDecoderGPT"),
    ):
        base_eval_module.main.__wrapped__(cfg)


def test_base_eval_rejects_hf_feature_config_before_token_bytes() -> None:
    cfg = OmegaConf.create(
        {
            "eval": "core",
            "device_type": "cpu",
            "hf_path": "local/test-model",
            "source": "base",
            "model_tag": "d12",
            "step": None,
            "feature_configs": "ignored",
            "conditioning": {},
        }
    )

    class FakeHFModel:
        max_seq_len = 16

    with (
        patch.object(
            base_eval_module,
            "compute_init",
            return_value=(False, 0, 0, 1, "cpu"),
        ),
        patch.object(
            base_eval_module,
            "load_hf_model",
            return_value=(FakeHFModel(), MockTokenizer()),
        ),
        patch.object(
            base_eval_module,
            "parse_feature_configs",
            return_value=[FeatureConfig(name="real_features", features={1: 2})],
        ),
        patch.object(
            base_eval_module,
            "get_hf_token_bytes",
            side_effect=AssertionError("HF token bytes must load after startup validation"),
        ),
        pytest.raises(ValueError, match="EncoderDecoderGPT"),
    ):
        base_eval_module.main.__wrapped__(cfg)


def test_base_eval_rejects_target_real_config_before_token_bytes() -> None:
    cfg = OmegaConf.create(
        {
            "eval": "core",
            "device_type": "cpu",
            "hf_path": None,
            "source": "base",
            "model_tag": "d12",
            "step": None,
            "feature_configs": "ignored",
            "conditioning": {},
        }
    )
    meta = {
        "model_config": {"sequence_len": 16},
        "step": 0,
        "checkpoint_dir": "/tmp/checkpoint",
        "user_config": {},
    }
    trc = LockstepTargetRealConfig(
        name="real_target",
        cache_dir=Path("/tmp/cache"),
    )

    with (
        patch.object(
            base_eval_module,
            "compute_init",
            return_value=(False, 0, 0, 1, "cpu"),
        ),
        patch.object(
            base_eval_module,
            "load_model",
            return_value=(_make_gpt(), MockTokenizer(), meta),
        ),
        patch.object(
            base_eval_module,
            "parse_feature_configs",
            return_value=[],
        ),
        patch.object(
            base_eval_module,
            "parse_target_real_configs_from_cfg",
            return_value=[trc],
        ),
        patch.object(
            base_eval_module,
            "get_token_bytes",
            side_effect=AssertionError("token bytes must load after target-real validation"),
        ),
        pytest.raises(ValueError, match="target_real"),
    ):
        base_eval_module.main.__wrapped__(cfg)
