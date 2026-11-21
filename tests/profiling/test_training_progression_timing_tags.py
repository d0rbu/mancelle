from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from scripts.train.training_progression import CheckpointSpec, evaluate_single_checkpoint
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.profiling.markers import current_timing_tags


def test_evaluate_single_checkpoint_tags_conditioned_eval(tmp_path):
    cfg = OmegaConf.create(
        {
            "checkpoint_source": "local",
            "cleanup_after_eval": False,
            "model_family": "encoder_decoder",
            "conditioning": {
                "context_cache_dir": str(tmp_path / "context_cache"),
                "conditioned_configs": [
                    {"name": "safe", "features": {"1": 2}},
                ],
            },
        }
    )
    spec = CheckpointSpec(step=1, phase="pretrain", subdir="")

    class _EncCfg:
        n_sae_features = 10
        num_buckets = 4

    class _Encoder:
        config = _EncCfg()

    class _DecCfg:
        sequence_len = 16

    class _Model:
        encoder = _Encoder()
        dec_config = _DecCfg()

    captured: list[dict] = []

    def fake_run(*_args, **_kwargs):
        captured.append(current_timing_tags())
        return {"metric": len(captured)}

    with (
        patch("scripts.train.training_progression.resolve_local_checkpoint_dir", return_value=str(tmp_path)),
        patch(
            "scripts.train.training_progression.load_progression_model",
            return_value=(
                _Model(),
                object(),
                {},
                CheckpointFeatureContract(None, None),
            ),
        ),
        patch("scripts.train.training_progression._is_encoder_decoder_model", return_value=True),
        patch("scripts.train.training_progression._validate_conditioned_configs"),
        patch("scripts.train.training_progression._assert_static_conditioning_context_coverage"),
        patch(
            "scripts.train.training_progression._context_provider_for_cfg",
            return_value=SimpleNamespace(cache_dir=tmp_path / "context_cache"),
        ),
        patch("scripts.train.training_progression._wrap_conditioned_model", return_value="cond"),
        patch("scripts.train.training_progression._run_eval_suite", side_effect=fake_run),
    ):
        evaluate_single_checkpoint(
            spec,
            torch.device("cpu"),
            torch.float32,
            ["core"],
            [1234],
            cfg,
        )

    assert captured == [
        {
            "suite": "core",
            "eval_variant": "conditioned",
            "conditioned_config": "safe",
            "seed": 1234,
        },
    ]
