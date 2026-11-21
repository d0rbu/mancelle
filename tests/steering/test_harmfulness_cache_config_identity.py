"""Cross-config contract for the phase-A L22 harmfulness context cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.eval.harmfulness_eval as harmfulness_eval
import scripts.eval.validate_eval_context_cache as cache_validator
import steering.eval.harmfulness_prompt_pool as prompt_pool_module
from steering.conditional.checkpoint_conditioning import (
    checkpoint_negative_selection,
)
from steering.conditional.checkpoint_feature_contract import (
    resolve_checkpoint_feature_contract,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_enumeration import enumerate_harmfulness_pool_contexts
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.refined_types import PositiveInt
from tests.conftest import build_tiny_tokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs" / "eval")
_BUILDER_CONFIG = "build_harmfulness_context_cache_l22"
_EVALUATOR_CONFIG = "harmfulness_eval_l22"
_MASK_PATH = (
    "out/autointerp_filter/"
    "gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_20260710_145543/"
    "manual_feature_filter_with_descriptions.yaml"
)
_CLUSTERS_PATH = "out/artifacts/feature_clusters_two_stage_schema_v1.yaml"
_PRETRAIN_STATS_PATH = (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos/feature_statistics.safetensors"
)
_PRETRAIN_STATS_SHA256 = "dadf6d4f8d3b56ba1b40e0f49f4bb92bfb635c64f78a530f04f624b1d5773b5d"
_CHECKPOINT_DIR = (
    Path.home() / ".cache/nanochat/conditional_encdec_checkpoints/"
    "l22_main_h100_8rank_db16_v1/s128/dist_geometric/mean_12.0/std_12.0/"
    "wl_9172c540310ba842/"
    "fc_5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039/"
    "capabilities_sft_conditioned_checkpoints"
)


def _compose_configs(hydra_compose):
    return (
        hydra_compose(_CONFIG_DIR, _BUILDER_CONFIG),
        hydra_compose(_CONFIG_DIR, _EVALUATOR_CONFIG),
    )


def _install_tiny_identity_artifacts(builder, evaluator, tmp_path: Path) -> None:
    """Keep the identity test hermetic while exercising production parsers."""
    clusters_path = tmp_path / "clusters.yaml"
    clusters_path.write_text("schema_version: 1\nclusters:\n  - [1, 2]\n")
    mask = {"mode": "whitelist", "ids": [1, 2, 3]}
    builder.feature_mask = mask
    evaluator.conditioning.feature_mask = mask
    builder.feature_clusters.path = str(clusters_path)
    evaluator.conditioning.feature_clusters.path = str(clusters_path)


def test_l22_harmfulness_configs_compose(hydra_compose) -> None:
    builder, evaluator = _compose_configs(hydra_compose)
    assert builder.evals == "harmfulness"
    assert evaluator.source == "encdec/capabilities_cond"


@pytest.mark.parametrize(
    "argv",
    [
        ["--config-name=harmfulness_eval_l22"],
        ["--config-name", "harmfulness_eval_l22"],
    ],
)
def test_l22_evaluator_preset_is_routed_from_the_production_cli(argv) -> None:
    evaluator = harmfulness_eval._compose_cli_config(argv)

    assert evaluator.source == "encdec/capabilities_cond"
    assert evaluator.model_tag == "l22_main_h100_8rank_db16_v1"


def test_l22_evaluator_rejects_compose_api_config_selector() -> None:
    with pytest.raises(ValueError, match="--config-name=harmfulness_eval_l22"):
        harmfulness_eval._compose_cli_config(["config_name=harmfulness_eval_l22"])


def test_l22_evaluator_pins_pure_self_conditioning(hydra_compose) -> None:
    _, evaluator = _compose_configs(hydra_compose)
    assert evaluator.feature_configs is None
    assert evaluator.self_conditioned.safety_steering is None
    assert evaluator.self_cond_segment_size is None
    assert evaluator.batch_size == 1


def test_builder_and_evaluator_cache_identities_are_equal(
    hydra_compose,
    tmp_path: Path,
) -> None:
    builder, evaluator = _compose_configs(hydra_compose)
    _install_tiny_identity_artifacts(builder, evaluator, tmp_path)

    builder_identity = cache_identity_from_config(builder.identity, builder)
    evaluator_identity = cache_identity_from_config(
        evaluator.conditioning.identity,
        evaluator.conditioning,
    )

    assert builder_identity == evaluator_identity


def test_pinned_identity_coordinates_and_artifact_paths(hydra_compose) -> None:
    builder, evaluator = _compose_configs(hydra_compose)
    expected_identity = {
        "model_id": "gemma-3-4b-pt",
        "sae_id": "layer_22_width_65k_l0_medium",
        "layer": 22,
        "pooling": "max_pool",
        "feature_selection_hash": None,
    }
    assert OmegaConf.to_container(builder.identity) == expected_identity
    assert OmegaConf.to_container(evaluator.conditioning.identity) == expected_identity
    assert builder.feature_mask.path == _MASK_PATH
    assert evaluator.conditioning.feature_mask.path == _MASK_PATH
    assert builder.feature_clusters.path == _CLUSTERS_PATH
    assert evaluator.conditioning.feature_clusters.path == _CLUSTERS_PATH


def test_l22_configs_pin_checkpoint_pretraining_statistics(hydra_compose) -> None:
    builder, evaluator = _compose_configs(hydra_compose)

    assert builder.stats_path == _PRETRAIN_STATS_PATH
    assert builder.stats_source_sha256 == _PRETRAIN_STATS_SHA256
    assert evaluator.conditioning.stats_source_sha256 == _PRETRAIN_STATS_SHA256


def test_builder_seed_set_is_exactly_the_evaluator_prompt_seed(hydra_compose) -> None:
    builder, evaluator = _compose_configs(hydra_compose)
    assert cache_validator.eval_seeds(builder) == [int(evaluator.seed)] == [42]


def test_builder_and_evaluator_resolve_the_same_ordered_prompt_pool(
    hydra_compose,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, evaluator = _compose_configs(hydra_compose)
    harmful = [{"prompt": f"harmful-{i}", "category": "harmful"} for i in range(700)]
    benign = [{"prompt": f"benign-{i}", "category": "benign"} for i in range(500)]
    monkeypatch.setattr(
        prompt_pool_module.harmfulness_judge,
        "load_harmful_prompts",
        lambda: harmful,
    )
    monkeypatch.setattr(
        prompt_pool_module.harmfulness_judge,
        "load_benign_prompts",
        lambda *, n, seed: benign[:n],
    )

    evaluator_pool = resolve_harmfulness_prompt_pool(
        seed=int(evaluator.seed),
        max_harmful=evaluator.num_harmful,
        num_benign=int(evaluator.num_benign),
    )
    captured_identities: list[tuple[str, str]] = []

    def capture_pool(*, tokenizer, pool, **kwargs):
        captured_identities.extend((row["prompt"], row["pool"]) for row in pool.rows)
        return enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool, **kwargs)

    monkeypatch.setattr(cache_validator, "enumerate_harmfulness_pool_contexts", capture_pool)
    records = cache_validator.enumerate_expected_eval_contexts(
        builder,
        build_tiny_tokenizer(),
    )
    evaluator_identities = [(row["prompt"], row["pool"]) for row in evaluator_pool.rows]

    assert len(records) == len(evaluator_pool.rows) == 1200
    assert captured_identities == evaluator_identities


def test_builder_matches_real_checkpoint_contract_when_present(hydra_compose) -> None:
    checkpoint_meta_path = _CHECKPOINT_DIR / "meta_000016.json"
    parent_meta_path = _CHECKPOINT_DIR.parent / "meta_054600.json"
    if not checkpoint_meta_path.is_file() or not parent_meta_path.is_file():
        pytest.skip("phase-A L22 SFT checkpoint or its parent pretraining metadata is absent")

    builder, evaluator = _compose_configs(hydra_compose)
    checkpoint_meta = json.loads(checkpoint_meta_path.read_text())
    parent_meta = json.loads(parent_meta_path.read_text())

    assert "seed" not in checkpoint_meta
    assert "seed" not in checkpoint_meta["user_config"]
    assert int(builder.seed) == parent_meta["user_config"]["seed"] == 42
    assert int(builder.model_max_seq_len) == checkpoint_meta["model_config"]["sequence_len"]
    assert int(builder.n_sae_features) == checkpoint_meta["feature_vocab"]["n_sae_features"]
    assert int(builder.num_buckets) == checkpoint_meta["num_buckets"]
    assert checkpoint_meta.get("negative_selection") is None
    assert checkpoint_meta["user_config"].get("negative_selection") is None
    assert OmegaConf.to_container(builder.negative_selection) == (
        checkpoint_negative_selection(checkpoint_meta)
    )

    n_sae_features = PositiveInt.parse(checkpoint_meta["feature_vocab"]["n_sae_features"])
    contract = resolve_checkpoint_feature_contract(
        checkpoint_meta,
        n_sae_features=n_sae_features,
        surface="harmfulness_eval_l22 preset test",
        source_path=str(checkpoint_meta_path),
    )
    feature_vocab = effective_feature_vocab(contract.universe, n_sae_features)
    assert feature_vocab is not None
    assert feature_vocab.metadata_dict() == checkpoint_meta["feature_vocab"]
    assert evaluator.feature_configs is None
    assert evaluator.self_conditioned.safety_steering is None
