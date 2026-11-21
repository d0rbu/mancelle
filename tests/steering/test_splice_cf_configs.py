"""Example splice_counterfactual configs must compose + parse (issue #596)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from steering.conditional.delta import DeltaMode, parse_feature_delta

_CFG_DIR = str(Path("configs/training").resolve())
_NAMES = [
    "conditional_encdec_splice_cf_gemma3pt_4090",
    "conditional_encdec_splice_cf_gemma3pt_8xpro6000",
]


@pytest.mark.parametrize(
    ("name", "expected_precomputed"),
    [
        ("conditional_encdec_splice_cf_gemma3pt_4090", False),
        ("conditional_encdec_splice_cf_gemma3pt_8xpro6000", False),
    ],
)
def test_config_composes_and_parses(name, expected_precomputed):
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        cfg = compose(config_name=name)
    fd = parse_feature_delta(dict(cfg.feature_delta))
    assert fd is not None and fd.mode is DeltaMode.splice_counterfactual
    assert cfg.feature_selection.delta_weights == "percentile_delta"
    assert cfg.use_precomputed_batches is expected_precomputed
    assert cfg.sae.model == "gemma-3-4b-pt"
    assert cfg.sae.layer == 22
    assert str(cfg.sae.width) == "65k"
    assert cfg.sae.release == "gemmascope_2"
    assert cfg.sae.l0 == "medium"
    assert cfg.segment_size == 128


def test_production_config_pins_issue_768_artifacts_policy_and_cadences():
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        cfg = compose(
            config_name=(
                "cluster_fp/"
                "conditional_encdec_splice_cf_gemma3pt_8xpro6000"
            )
        )

    assert dict(cfg.feature_selection) == {
        "distribution": "geometric",
        "mean": 12.0,
        "std": 12.0,
        "min_k": 1,
        "max_k": 49,
        "mode": "top_k_percentile",
        "delta_weights": "percentile_delta",
    }
    assert dict(cfg.negative_selection) == {
        "distribution": "geometric",
        "mean": 12.0,
        "min_k": 0,
        "max_k": 49,
    }
    assert cfg.feature_mask.mode == "whitelist"
    assert str(cfg.feature_mask.path).endswith("manual_feature_filter_with_descriptions.yaml")
    assert str(cfg.feature_clusters.path).endswith("feature_clusters_two_stage_schema_v1.yaml")
    assert dict(cfg.feature_prediction) == {
        "block": "mlp",
        "head_layers": 0,
        "weight": 0.1,
        "layer": -1,
        "loss_mode": "max_pool",
        "loss_fn": "bce",
    }
    assert cfg.initialize_encoder_embeddings_from_sae == "svd"
    assert cfg.initialize_encoder_embeddings_cluster_vectors == "mean"
    # D-seed (PR #739 Track D, 2026-07-24) migration: clustered
    # eval-context-cache identity is now pinned to the real
    # conditioning.seed (42) with delta_weights retained as null (Codex
    # cross-review Finding 1, HIGH -- matching the packed-row producer's
    # payload convention) rather than a legacy hardcoded seed=0 -- see the
    # config's own migration comment and
    # tests/steering/test_issue768_production_artifacts.py's
    # _PROMPT_SELECTION_HASH_LEGACY/_PROMPT_SELECTION_HASH split (the
    # latter now identical to _BASE_SELECTION_HASH) for the full
    # old-> new derivation.
    assert cfg.conditioning.identity.feature_selection_hash == (
        "6be4095ad0116d3e1b7ca93a2fe0a47439eaba770c6d0ef08ab164a0652a19bc"
    )
    assert cfg.core_metric_every == 2000
    assert cfg.sample_every == 2000
    assert cfg.steering_eval_every == 250
