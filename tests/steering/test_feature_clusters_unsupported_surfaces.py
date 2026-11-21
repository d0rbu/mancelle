"""Fail-loud support boundary for the preselection-only cluster V1."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.conditional.feature_gcg.optimize as optimize_module
import steering.eval.cluster_feature_contract as contract_module
from errors import (
    FeatureClustersError,
    FeatureVocabError,
    RemovedFeatureError,
    SelfCondTrainingError,
)
from scripts.eval import base_eval, completion_steering_eval
from scripts.train import conditional_encdec_sft
from scripts.train import run_encdec_sweep_fast as fast_sweep
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.selfcond_training import (
    validate_selfcond_prerequisites,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.optimize import optimize_feature_conditioning_universal
from steering.conditional.feature_gcg.partition import partition_bounds_from_model
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode, canonical_dict
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    feature_vocab_from_state_dict,
)
from steering.eval import cache_identity
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)
from steering.eval.shuffled_conditioning_eval import assert_no_feature_clustering
from tests.steering.feature_gcg.unit.test_clustered_optimizer_contract import (
    _contract_model_pair,
)

REPO_ROOT = Path(__file__).parents[2]

# Every entry must retain a literal rejection guard. Most are unsupported;
# feature-GCG's three entries are conditionally supported only when a validated
# cluster contract is supplied, and still reject on the no-contract path.
GUARDED_METADATA_CONSUMERS = {
    "scripts/chat/chat_cli.py",
    "scripts/chat/chat_web.py",
    "scripts/chat/steer_chat_cli.py",
    "scripts/eval/base_eval.py",
    "scripts/eval/chat_eval.py",
    "scripts/eval/completion_steering_eval.py",
    "scripts/eval/dump_chat_responses.py",
    "scripts/eval/steering_eval.py",
    "scripts/train/chat_rl.py",
    "scripts/train/chat_sft.py",
    "scripts/train/conditional_encdec_sft.py",
    "steering/eval/axbench_base/model_detection.py",
    "steering/eval/axbench_base/runtime_state.py",
    "steering/eval/axbench_bpb/runtime_state.py",
    "steering/eval/dclm_eval.py",
    "steering/eval/inline_steering_eval.py",
    "steering/conditional/feature_gcg/partition.py",
    "steering/conditional/feature_gcg/optimize.py",
    "steering/eval/axbench_bpb_adv/gcg_bags.py",
    "steering/eval/axbench_bpb_adv/gcg_config.py",
    "steering/eval/shuffled_conditioning_eval.py",
}

# Each supported consumer maps to its RESOLVER CHAIN: the files that, between
# them, must resolve static clusters from checkpoint metadata. A chain rather
# than a single file because the resolution may be delegated -- the completion
# TUI's resolver now calls the shared
# steering/conditional/checkpoint_feature_contract, so asserting the literal in
# the TUI file alone would pin the CALL SITE (a proxy) instead of the property
# (that the chain resolves clusters from checkpoint metadata at all).
SUPPORTED_METADATA_CONSUMERS = {
    "scripts/completion_tui/__main__.py": (
        "scripts/completion_tui/checkpoint_contract.py",
        "steering/conditional/checkpoint_feature_contract.py",
    ),
}

SUPPORTED_CONDITIONED_EVAL_CONSUMERS = {
    "scripts/eval/harmfulness_eval.py": "harmfulness_eval",
    "scripts/train/training_progression.py": "training_progression",
}

UNSUPPORTED_CONFIG_CONSUMERS = {
    "scripts/eval/base_eval.py",
    "scripts/eval/completion_steering_eval.py",
    "scripts/preprocess/convert_decoder_to_encdec.py",
    "steering/eval/autointerp_filter/cli.py",
}


@pytest.mark.parametrize("relative_path", sorted(GUARDED_METADATA_CONSUMERS))
def test_every_guarded_checkpoint_consumer_has_new_schema_guard(
    relative_path: str,
):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "reject_feature_clusters_metadata" in source, relative_path
    assert "reject_feature_clustering_metadata" not in source, relative_path


@pytest.mark.parametrize(
    ("relative_path", "resolver_chain"),
    sorted(SUPPORTED_METADATA_CONSUMERS.items()),
)
def test_every_supported_checkpoint_consumer_resolves_new_schema(
    relative_path: str,
    resolver_chain: tuple[str, ...],
):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    resolver_sources = {
        path: (REPO_ROOT / path).read_text(encoding="utf-8") for path in resolver_chain
    }
    assert "resolve_checkpoint_feature_contract" in source, relative_path
    assert any(
        "resolve_feature_clusters_from_checkpoint_meta" in resolver_source
        for resolver_source in resolver_sources.values()
    ), sorted(resolver_sources)
    assert 'conditions_within_vocab("completion_tui")' in source, relative_path
    assert "reject_feature_clusters_metadata" not in source, relative_path
    assert "EvalConditioning.deferred" not in source, relative_path


@pytest.mark.parametrize(
    ("relative_path", "eval_name"),
    sorted(SUPPORTED_CONDITIONED_EVAL_CONSUMERS.items()),
)
def test_conditioned_eval_consumers_use_validated_cluster_aware_loader(
    relative_path: str,
    eval_name: str,
):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    resolver_source = (
        REPO_ROOT / "steering/eval/compacted_eval_loading.py"
    ).read_text(encoding="utf-8")
    assert "assert_conditioned_eval_model" in source, relative_path
    assert f'conditioned_eval_conditioning("{eval_name}")' in source, relative_path
    assert "resolve_and_assert_model_feature_contract" in resolver_source
    assert "EvalConditioning.conditions_within_vocab" in resolver_source
    assert "reject_feature_clusters_metadata" not in source, relative_path
    assert "EvalConditioning.deferred" not in source, relative_path


@pytest.mark.parametrize("relative_path", sorted(UNSUPPORTED_CONFIG_CONSUMERS))
def test_every_unsupported_config_consumer_has_new_schema_guard(relative_path: str):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "reject_feature_clusters_config" in source, relative_path
    assert "reject_feature_clustering_config" not in source, relative_path


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/preprocess/precompute_selected_features.py",
        "scripts/train/conditional_encdec_train.py",
        "steering/conditional/feature_pipeline.py",
    ],
)
def test_supported_core_paths_do_not_install_a_global_new_config_rejection(
    relative_path: str,
):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "reject_feature_clusters_config" not in source, relative_path


def _clustered_meta() -> dict:
    return {
        "feature_clusters_descriptor": {
            "schema_version": 1,
            "n_sae_features": 65_536,
            "content_sha256": "1" * 64,
            "semantics": "post_mask_post_delta_max_percentile_v1",
        },
        "feature_clusters_canonical": {
            "schema_version": 1,
            "clusters": [[1, 2]],
        },
        "user_config": {
            "feature_clusters": {"path": "/producer/clusters.yaml"},
        },
    }


@pytest.fixture
def effective_clustered_checkpoint(tmp_path: Path) -> SimpleNamespace:
    """A strict clustered checkpoint whose physical rows are the effective vocab.

    The state dict contains the same self-describing buffers as a real compacted
    encoder.  Its mask allows four ids and its active cluster consolidates id 2 into
    representative 1, so the three-row table could not be mistaken for either the full
    eight-row dictionary or the four-row mask-only pre-migration shape.
    """
    n_sae_features = 8
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4))
    definition = FeatureClusterDefinition(clusters=((1, 2),))
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=n_sae_features,
    )
    resolved = ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.json",
        definition=definition,
        descriptor=descriptor,
    )
    consolidation = FeatureClusterConsolidation.build(resolved, feature_mask)
    vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert vocab is not None
    assert vocab.original_ids == (1, 3, 4)

    model_data = {
        FEATURE_EMBED_KEY: t.zeros((vocab.size, 4)),
        VOCAB_IDS_KEY: t.tensor(vocab.original_ids, dtype=t.int64),
        VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
    }
    recovered_n_sae, recovered_vocab = feature_vocab_from_state_dict(model_data)
    assert recovered_n_sae == n_sae_features
    assert recovered_vocab == vocab

    feature_clusters = {"path": str(resolved.source_path)}
    meta = {
        "feature_mask_canonical": canonical_dict(feature_mask),
        **feature_clusters_checkpoint_metadata(resolved),
        "user_config": {
            "feature_mask": canonical_dict(feature_mask),
            "feature_clusters": feature_clusters,
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "max_pool",
            },
            "num_buckets": 2,
            "negative_selection": {"distribution": "normal"},
        },
    }
    return SimpleNamespace(
        consolidation=consolidation,
        feature_mask=feature_mask,
        meta=meta,
        model_data=model_data,
        n_sae_features=n_sae_features,
        vocab=vocab,
    )


def test_public_unsupported_guards_also_reject_removed_legacy_schema():
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        cache_identity.resolve_feature_selection_hash(
            OmegaConf.create({"feature_selection_hash": None}),
            OmegaConf.create({"feature_clustering": None}),
        )
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        assert_no_feature_clustering({"feature_clustering": {"enabled": True}})


def test_shuffled_eval_rejects_new_clustered_checkpoint():
    with pytest.raises(FeatureClustersError, match="shuffled_conditioning_eval"):
        assert_no_feature_clustering(_clustered_meta())


def test_effective_compacted_checkpoint_reaches_cluster_guard_in_shuffled_eval(
    effective_clustered_checkpoint: SimpleNamespace,
):
    with pytest.raises(FeatureClustersError, match="shuffled_conditioning_eval"):
        assert_no_feature_clustering(effective_clustered_checkpoint.meta)


def test_sft_entrypoint_accepts_new_cluster_config_surface(hydra_compose):
    config_dir = str(REPO_ROOT / "configs" / "training")
    cfg = hydra_compose(
        config_dir,
        "conditional_encdec_sft",
        overrides=["feature_clusters={path:/tmp/clusters.yaml}"],
    )

    assert cfg.feature_clusters.path == "/tmp/clusters.yaml"
    source = (REPO_ROOT / "scripts/train/conditional_encdec_sft.py").read_text(encoding="utf-8")
    assert "resolve_feature_clusters(cfg, n_sae_features)" in source
    assert "reject_feature_clusters_config" not in source


def _sft_base_meta(cluster_meta: dict) -> dict:
    return {
        **cluster_meta,
        "model_config": {
            "sequence_len": 16,
            "vocab_size": 128,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 64,
            "window_pattern": "L",
        },
        "user_config": {
            **cluster_meta.get("user_config", {}),
            "encoder": {
                "n_embd": 32,
                "depth": 1,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
            "cross_attention": {
                "frequency": 1,
                "n_head": 2,
                "n_kv_head": 2,
            },
            "num_buckets": 2,
        },
    }


def test_sft_checkpoint_rejects_before_wandb_profile_or_dataset_work(
    effective_clustered_checkpoint: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = OmegaConf.create(
        {
            "dataset": "safety",
            "feature_clusters": None,
            "feature_selection": {
                "threshold_type": "segment_topk",
                "threshold": 1,
            },
            "device_type": "cpu",
            "base_checkpoint_dir": str(tmp_path / "base_checkpoint"),
            "model_tag": "unit",
            "resume_from_step": -1,
            "model_step": 1,
            "run": "unit",
            "max_seq_len": None,
            "num_buckets": None,
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
        }
    )
    events: list[str] = []

    monkeypatch.setattr(conditional_encdec_sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(
        conditional_encdec_sft,
        "initialize_config_dir",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(conditional_encdec_sft, "compose", lambda **_: cfg)
    monkeypatch.setattr(
        conditional_encdec_sft,
        "compute_init",
        lambda _device_type: (False, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(conditional_encdec_sft, "get_base_dir", lambda: str(tmp_path))

    def clustered_checkpoint(*_args, **_kwargs):
        events.append("checkpoint")
        return (
            effective_clustered_checkpoint.model_data,
            None,
            _sft_base_meta(effective_clustered_checkpoint.meta),
        )

    def unexpected_wandb(*_args, **_kwargs):
        events.append("wandb")
        raise AssertionError("clustered SFT checkpoint reached W&B initialization")

    def unexpected_profile(*_args, **_kwargs):
        events.append("profile")
        raise AssertionError("clustered SFT checkpoint reached profile resolution")

    def unexpected_tokenizer(*_args, **_kwargs):
        events.append("tokenizer")
        raise AssertionError("clustered SFT checkpoint reached tokenizer loading")

    monkeypatch.setattr(conditional_encdec_sft, "load_checkpoint", clustered_checkpoint)
    monkeypatch.setattr(conditional_encdec_sft, "create_wandb_run", unexpected_wandb)
    monkeypatch.setattr(
        conditional_encdec_sft,
        "_resolve_sft_profile_settings",
        unexpected_profile,
    )
    monkeypatch.setattr(conditional_encdec_sft, "get_tokenizer", unexpected_tokenizer)

    with pytest.raises(FeatureClustersError, match="base checkpoint"):
        conditional_encdec_sft.main()

    assert events == ["checkpoint"]


def test_sft_agreeing_cluster_contract_reaches_vocab_guard_before_external_work(
    effective_clustered_checkpoint: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Configured clusters are accepted, then vocabulary equality remains load-bearing."""
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 2]]}),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "dataset": "safety",
            "feature_clusters": {"path": str(cluster_path)},
            "feature_mask": {
                "mode": "whitelist",
                "ids": [1, 2, 3, 4],
            },
            "feature_selection": {
                "threshold_type": "feature_percentile",
                "threshold": 0.5,
            },
            "device_type": "cpu",
            "base_checkpoint_dir": str(tmp_path / "base_checkpoint"),
            "model_tag": "unit",
            "resume_from_step": -1,
            "model_step": 1,
            "run": "unit",
            "max_seq_len": None,
            "num_buckets": None,
            "lr_warmup_steps": 0,
            "num_iterations": -1,
            "feature_prediction_block": "mlp",
            "feature_prediction_head_layers": 0,
            "sft_loss_mask": "full_sequence",
            "missing_percentile_policy": "error",
        }
    )
    events: list[str] = []

    monkeypatch.setattr(conditional_encdec_sft.sys, "argv", ["conditional_encdec_sft.py"])
    monkeypatch.setattr(
        conditional_encdec_sft,
        "initialize_config_dir",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(conditional_encdec_sft, "compose", lambda **_: cfg)
    monkeypatch.setattr(
        conditional_encdec_sft,
        "compute_init",
        lambda _device_type: (False, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(conditional_encdec_sft, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        conditional_encdec_sft,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            effective_clustered_checkpoint.model_data,
            None,
            _sft_base_meta(effective_clustered_checkpoint.meta),
        ),
    )

    def reached_vocab_guard(*_args, **_kwargs):
        events.append("vocab")
        raise FeatureVocabError("vocabulary sentinel after agreeing cluster contract")

    def unexpected_wandb(*_args, **_kwargs):
        events.append("wandb")
        raise AssertionError("clustered SFT checkpoint reached W&B initialization")

    monkeypatch.setattr(
        conditional_encdec_sft,
        "_assert_sft_universe_matches_checkpoint_vocab",
        reached_vocab_guard,
    )
    monkeypatch.setattr(conditional_encdec_sft, "create_wandb_run", unexpected_wandb)

    with pytest.raises(FeatureVocabError, match="vocabulary sentinel"):
        conditional_encdec_sft.main()

    assert events == ["vocab"]


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/train/chat_sft.py",
        "scripts/train/chat_rl.py",
    ],
)
def test_module_level_sft_consumers_guard_loaded_metadata_before_wandb(
    relative_path: str,
):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    load_index = source.index("load_model(")
    guard_index = source.index("reject_feature_clusters_metadata(")
    wandb_index = source.index("wandb.init(")
    assert load_index < guard_index < wandb_index


def test_fast_sweep_shared_setup_rejects_clusters_before_build_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sweep_config = fast_sweep.build_sweep_configs()[0]
    entry_cfg = OmegaConf.create(
        {
            "mode": 0,
            "min_idx": None,
            "max_idx": None,
            "resume": False,
            "shuffle": False,
            "shuffle_seed": 42,
            "multigpu_config": None,
            "output_dir": str(tmp_path / "output"),
            "config_name": "conditional_encdec_sweep",
            "training_overrides": [],
        }
    )
    base_cfg = OmegaConf.create({"feature_clusters": {"path": str(tmp_path / "clusters.yaml")}})
    build_called = False

    def unexpected_build(*_args, **_kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("clustered fast sweep reached shared-state construction")

    monkeypatch.setattr(fast_sweep, "build_sweep_configs", lambda: [sweep_config])
    monkeypatch.setattr(fast_sweep, "print_banner", lambda: None)
    monkeypatch.setattr(fast_sweep.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(fast_sweep, "_compose_config", lambda **_kwargs: base_cfg)
    monkeypatch.setattr(fast_sweep, "build_shared_state", unexpected_build)

    with pytest.raises(FeatureClustersError, match="run_encdec_sweep_fast"):
        fast_sweep.main.__wrapped__(entry_cfg)

    assert build_called is False
    assert not (tmp_path / "output").exists()


def test_fast_sweep_per_config_guard_precedes_wandb_and_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sweep_config = fast_sweep.build_sweep_configs()[0]
    per_config = OmegaConf.create({"feature_clusters": {"path": str(tmp_path / "clusters.yaml")}})
    events: list[str] = []

    monkeypatch.setattr(fast_sweep, "_compose_config", lambda **_kwargs: per_config)
    monkeypatch.setattr(
        fast_sweep,
        "_resolve_splice_counterfactual_data_dir",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("clustered fast sweep reached splice resolution")
        ),
    )

    def unexpected_wandb(**_kwargs):
        events.append("wandb")
        raise AssertionError("clustered fast sweep reached W&B")

    def unexpected_train(*_args):
        events.append("train")
        raise AssertionError("clustered fast sweep reached training")

    monkeypatch.setattr(
        fast_sweep.wandb,
        "init",
        unexpected_wandb,
    )
    monkeypatch.setattr(
        fast_sweep,
        "train_one_config",
        unexpected_train,
    )

    result = fast_sweep._run_one(
        sweep_config,
        index=0,
        total=1,
        shared=SimpleNamespace(ddp_rank=0, feature_dir=tmp_path),  # ty: ignore[invalid-argument-type]
    )

    assert result.error is not None
    assert "run_encdec_sweep_fast per-config" in result.error
    assert events == []


def test_completion_eval_rejects_new_cluster_config_before_feature_resolution(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = OmegaConf.create({"feature_clusters": {"path": "/tmp/clusters.yaml"}})
    monkeypatch.setattr(
        completion_steering_eval.sys,
        "argv",
        ["completion_steering_eval.py"],
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "initialize_config_dir",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(completion_steering_eval, "compose", lambda **_: cfg)
    with pytest.raises(FeatureClustersError, match="completion_steering_eval"):
        completion_steering_eval.main()


def test_base_eval_rejects_top_level_new_cluster_config_before_ddp_setup():
    cfg = OmegaConf.create({"feature_clusters": {"path": "/tmp/clusters.yaml"}})
    with pytest.raises(FeatureClustersError, match="base_eval config"):
        base_eval.main.__wrapped__(cfg)


def test_base_eval_rejects_conditioning_new_cluster_config_before_ddp_setup():
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "feature_clusters": {"path": "/tmp/clusters.yaml"},
            }
        }
    )
    with pytest.raises(FeatureClustersError, match="base_eval conditioning"):
        base_eval.main.__wrapped__(cfg)


def test_selfcond_effective_checkpoint_loads_then_rejects_compact_fp(
    effective_clustered_checkpoint: SimpleNamespace,
):
    """Self-describing V_eff buffers must load before the intended late C2 guard."""
    n_sae_features, vocab = feature_vocab_from_state_dict(effective_clustered_checkpoint.model_data)
    assert n_sae_features == effective_clustered_checkpoint.n_sae_features
    assert vocab == effective_clustered_checkpoint.vocab
    model = SimpleNamespace(
        feature_pred_head=SimpleNamespace(out_features=vocab.size),  # ty: ignore[unresolved-attribute]
        feature_vocab=vocab,
        feature_mask=effective_clustered_checkpoint.feature_mask,
        encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2)),
    )

    with pytest.raises(NotImplementedError, match="compact feature-prediction"):
        validate_selfcond_prerequisites(
            model,
            effective_clustered_checkpoint.meta,
            SimpleNamespace(),  # ty: ignore[invalid-argument-type]
            n_sae_features,
        )


def test_selfcond_cluster_metadata_reaches_c5_when_c2_is_not_applicable(
    effective_clustered_checkpoint: SimpleNamespace,
):
    """The independent C5 cluster guard remains live behind the model-load boundary."""
    n_sae_features, vocab = feature_vocab_from_state_dict(effective_clustered_checkpoint.model_data)
    assert vocab == effective_clustered_checkpoint.vocab
    model = SimpleNamespace(
        feature_pred_head=SimpleNamespace(out_features=n_sae_features),
        feature_vocab=None,
        feature_mask=None,
        encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2)),
    )
    cluster_only_meta = {
        **effective_clustered_checkpoint.meta,
        "user_config": {
            **effective_clustered_checkpoint.meta["user_config"],
            "feature_mask": None,
        },
    }

    with pytest.raises(
        SelfCondTrainingError,
        match="C5: feature_clusters is unsupported",
    ):
        validate_selfcond_prerequisites(
            model,
            cluster_only_meta,
            SimpleNamespace(),  # ty: ignore[invalid-argument-type]
            n_sae_features,
        )


def _clustered_feature_gcg_model(
    meta: dict | None = None,
    *,
    feature_vocab: object | None = None,
):
    return SimpleNamespace(
        _loaded_meta_data=_clustered_meta() if meta is None else meta,
        feature_vocab=feature_vocab,
        encoder=SimpleNamespace(
            config=SimpleNamespace(num_buckets=2),
        ),
    )


def test_feature_gcg_partition_rejects_clustered_checkpoint_before_bounds():
    with pytest.raises(FeatureClustersError, match="feature_gcg partition"):
        partition_bounds_from_model(
            _clustered_feature_gcg_model(),
            FeatureGcgConfig(),
        )


def test_feature_gcg_partition_rejects_effective_compacted_checkpoint_as_clusters(
    effective_clustered_checkpoint: SimpleNamespace,
):
    model = _clustered_feature_gcg_model(
        effective_clustered_checkpoint.meta,
        feature_vocab=effective_clustered_checkpoint.vocab,
    )
    with pytest.raises(FeatureClustersError, match="feature_gcg partition"):
        partition_bounds_from_model(model, FeatureGcgConfig())


def test_feature_gcg_universal_optimizer_rejects_effective_compacted_checkpoint(
    effective_clustered_checkpoint: SimpleNamespace,
):
    def objective_factory(_model):
        raise AssertionError("effective-compacted clustered model reached objective construction")

    with pytest.raises(FeatureClustersError, match="optimize checkpoint"):
        optimize_feature_conditioning_universal(
            _clustered_feature_gcg_model(
                effective_clustered_checkpoint.meta,
                feature_vocab=effective_clustered_checkpoint.vocab,
            ),
            [],
            FeatureConditioning(entries=()),
            objective_factory=objective_factory,
            config=FeatureGcgConfig(),
        )


def test_feature_gcg_bucket_layout_rejects_effective_compacted_checkpoint(
    effective_clustered_checkpoint: SimpleNamespace,
):
    with pytest.raises(FeatureClustersError, match="bucket-layout"):
        assert_single_positive_bucket(
            _clustered_feature_gcg_model(
                effective_clustered_checkpoint.meta,
                feature_vocab=effective_clustered_checkpoint.vocab,
            )
        )


def test_feature_gcg_top_k_rejects_effective_compacted_checkpoint(
    effective_clustered_checkpoint: SimpleNamespace,
):
    with pytest.raises(FeatureClustersError, match="top-k"):
        resolve_k(
            _clustered_feature_gcg_model(
                effective_clustered_checkpoint.meta,
                feature_vocab=effective_clustered_checkpoint.vocab,
            ),
            OursGcgConfig(top_k_pos=2, top_k_neg=2),
        )


def test_feature_gcg_universal_optimizer_rejects_before_objective_factory():
    objective_factory_called = False

    def objective_factory(_model):
        nonlocal objective_factory_called
        objective_factory_called = True
        raise AssertionError("clustered model reached objective construction")

    with pytest.raises(FeatureClustersError, match="optimize checkpoint"):
        optimize_feature_conditioning_universal(
            _clustered_feature_gcg_model(),
            [],
            FeatureConditioning(entries=()),
            objective_factory=objective_factory,
            config=FeatureGcgConfig(),
        )
    assert objective_factory_called is False


def test_feature_gcg_bucket_layout_rejects_clustered_checkpoint():
    with pytest.raises(FeatureClustersError, match="bucket-layout"):
        assert_single_positive_bucket(_clustered_feature_gcg_model())


def test_feature_gcg_top_k_rejects_clustered_checkpoint_even_with_overrides():
    with pytest.raises(FeatureClustersError, match="top-k"):
        resolve_k(
            _clustered_feature_gcg_model(),
            OursGcgConfig(top_k_pos=2, top_k_neg=2),
        )


def test_feature_gcg_partition_accepts_valid_contract_before_bounds(
    tmp_path: Path,
):
    contract, model, _ = _contract_model_pair(tmp_path)

    bounds = partition_bounds_from_model(
        model,
        FeatureGcgConfig(),
        cluster_contract=contract,
    )

    assert bounds.num_buckets == 2


def test_feature_gcg_partition_accepts_effective_compacted_checkpoint_with_contract(
    tmp_path: Path,
):
    contract, model, _ = _contract_model_pair(tmp_path)

    bounds = partition_bounds_from_model(
        model,
        FeatureGcgConfig(min_k_pos=1, max_k_pos=3),
        cluster_contract=contract,
    )

    assert (bounds.min_k_pos, bounds.max_k_pos) == (1, 3)


def test_feature_gcg_universal_optimizer_accepts_effective_checkpoint_with_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Only the disk rehash is patched; loaded-state validation remains real."""

    contract, model, _ = _contract_model_pair(tmp_path)
    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        lambda _contract: None,
    )
    expected = FeatureConditioning(entries=((0, 0),))
    monkeypatch.setattr(
        optimize_module,
        "run_gcg",
        lambda **_kwargs: [SimpleNamespace(payload={"bag": expected})],
    )

    actual = optimize_feature_conditioning_universal(
        model,
        [object()],
        expected,
        objective_factory=lambda _model: cast(Any, object()),
        config=FeatureGcgConfig(),
        cluster_contract=contract,
    )

    assert actual is expected


def test_feature_gcg_bucket_layout_accepts_effective_checkpoint_with_contract(
    tmp_path: Path,
):
    contract, model, _ = _contract_model_pair(tmp_path)

    assert assert_single_positive_bucket(
        model,
        cluster_contract=contract,
    ) == (0, 1)


def test_feature_gcg_top_k_accepts_effective_checkpoint_with_contract(
    tmp_path: Path,
):
    contract, model, _ = _contract_model_pair(tmp_path)

    assert resolve_k(
        model,
        OursGcgConfig(),
        cluster_contract=contract,
    ) == (2, 1)


def test_feature_gcg_universal_optimizer_validates_then_builds_objective_with_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Only the disk rehash is patched; loaded-state validation remains real."""

    contract, model, _ = _contract_model_pair(tmp_path)
    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        lambda _contract: None,
    )
    expected = FeatureConditioning(entries=((0, 0),))
    monkeypatch.setattr(
        optimize_module,
        "run_gcg",
        lambda **_kwargs: [SimpleNamespace(payload={"bag": expected})],
    )
    objective_factory_calls = 0

    def objective_factory(_model):
        nonlocal objective_factory_calls
        objective_factory_calls += 1
        return cast(Any, object())

    optimize_feature_conditioning_universal(
        model,
        [object()],
        expected,
        objective_factory=objective_factory,
        config=FeatureGcgConfig(),
        cluster_contract=contract,
    )

    assert objective_factory_calls == 1


def test_feature_gcg_top_k_accepts_clustered_checkpoint_with_overrides_and_contract(
    tmp_path: Path,
):
    contract, model, _ = _contract_model_pair(tmp_path)

    assert resolve_k(
        model,
        OursGcgConfig(top_k_pos=7, top_k_neg=5),
        cluster_contract=contract,
    ) == (7, 5)
