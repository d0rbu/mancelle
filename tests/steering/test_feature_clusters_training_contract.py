"""Trainer startup/config/checkpoint contracts for static feature clusters."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf

from errors import FeatureClustersError
from scripts.train.conditional_encdec_train import (
    _checkpoint_user_config,
    _validate_feature_clusters_training_config,
)
from steering.conditional.compaction_summary import (
    format_compaction_summary,
    no_compaction_reason,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab

REPO_ROOT = Path(__file__).parents[2]
TRAINING_CONFIG_DIR = str(REPO_ROOT / "configs" / "training")


def _cfg(
    *,
    feature_clusters=True,
    feature_selection: dict | None = None,
    feature_delta: dict | None = None,
    **overrides,
):
    raw = {
        "feature_clusters": (
            {"path": "/tmp/static-clusters.yaml"} if feature_clusters is True else feature_clusters
        ),
        "use_precomputed_batches": True,
        "feature_prediction": None,
        "core_metric_every": 0,
        "sample_every": 0,
        "steering_eval_every": 0,
        "eval_every": 100,
        "next_segment_conditioning_offset": False,
        "feature_selection": feature_selection
        or {
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,
        },
        "feature_delta": feature_delta,
    }
    raw.update(overrides)
    return OmegaConf.create(raw)


def test_disabled_config_is_exact_guard_noop():
    _validate_feature_clusters_training_config(_cfg(feature_clusters=None))


def test_absent_config_is_guard_noop():
    cfg = _cfg(feature_clusters=None)
    del cfg["feature_clusters"]
    _validate_feature_clusters_training_config(cfg)


def test_reviewed_sampling_precomputed_training_is_accepted():
    _validate_feature_clusters_training_config(_cfg())


@pytest.mark.parametrize(
    "feature_selection",
    [
        {"threshold_type": "feature_percentile", "threshold": 0.8},
        {
            "threshold_type": "feature_percentile_segment_topk",
            "threshold": 10.8,
        },
    ],
)
def test_reviewed_percentile_threshold_training_is_accepted(feature_selection):
    _validate_feature_clusters_training_config(_cfg(feature_selection=feature_selection))


def test_conditioned_bpb_validation_remains_supported():
    _validate_feature_clusters_training_config(_cfg(eval_every=1))


def test_splice_counterfactual_sampling_training_is_accepted():
    _validate_feature_clusters_training_config(
        _cfg(
            feature_delta={
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 0.5,
            }
        )
    )


def test_live_or_fused_training_is_rejected():
    with pytest.raises(FeatureClustersError, match="use_precomputed_batches=true"):
        _validate_feature_clusters_training_config(_cfg(use_precomputed_batches=False))


def test_compact_feature_prediction_is_accepted_with_precomputed_rows():
    _validate_feature_clusters_training_config(_cfg(feature_prediction={"weight": 0.1}))


@pytest.mark.parametrize("key", ["core_metric_every", "sample_every"])
def test_clustered_cache_backed_core_and_samples_are_accepted(key: str):
    cfg = _cfg()
    cfg[key] = 1
    _validate_feature_clusters_training_config(cfg)


def test_clustered_splice_in_training_steering_is_accepted():
    cfg = _cfg(
        feature_delta={
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        }
    )
    cfg.steering_eval_every = 1
    _validate_feature_clusters_training_config(cfg)


def test_clustered_nonsplice_in_training_steering_is_rejected():
    cfg = _cfg()
    cfg.steering_eval_every = 1
    with pytest.raises(FeatureClustersError, match="requires.*splice_counterfactual"):
        _validate_feature_clusters_training_config(cfg)


def test_next_segment_offset_is_rejected():
    with pytest.raises(FeatureClustersError, match="next_segment_conditioning_offset"):
        _validate_feature_clusters_training_config(_cfg(next_segment_conditioning_offset=True))


@pytest.mark.parametrize("mode", ["prev_segment", "cum_max"])
def test_nonsplice_delta_modes_are_rejected(mode: str):
    with pytest.raises(FeatureClustersError, match="splice_counterfactual only"):
        _validate_feature_clusters_training_config(
            _cfg(feature_delta={"mode": mode, "neg_delta_ratio": 1.0})
        )


@pytest.mark.parametrize(
    "feature_selection",
    [
        {"threshold_type": "segment_topk", "threshold": 10},
        {"threshold_type": "segment_percentile", "threshold": 0.9},
    ],
)
def test_ordinary_raw_score_thresholds_are_rejected(feature_selection):
    with pytest.raises(FeatureClustersError, match="percentile-based thresholds"):
        _validate_feature_clusters_training_config(_cfg(feature_selection=feature_selection))


@pytest.mark.parametrize(
    "feature_selection",
    [
        {"threshold_type": "feature_percentile", "threshold": 0.8},
        {
            "threshold_type": "feature_percentile_segment_topk",
            "threshold": 10.8,
        },
    ],
)
def test_splice_threshold_selection_is_rejected_even_when_percentile_based(
    feature_selection,
):
    with pytest.raises(FeatureClustersError, match="sampling selection only"):
        _validate_feature_clusters_training_config(
            _cfg(
                feature_selection=feature_selection,
                feature_delta={
                    "mode": "splice_counterfactual",
                    "neg_delta_ratio": 1.0,
                },
            )
        )


def test_checkpoint_user_config_omits_null_cluster_key_byte_compatibly():
    cfg = OmegaConf.create(
        {
            "depth": 12,
            "feature_clusters": None,
            "feature_mask": None,
        }
    )
    actual = _checkpoint_user_config(cfg)
    assert actual == {"depth": 12, "feature_mask": None}
    assert "feature_clusters" not in actual


def test_checkpoint_user_config_absent_cluster_key_stays_absent():
    cfg = OmegaConf.create({"depth": 12, "feature_mask": None})
    assert _checkpoint_user_config(cfg) == {
        "depth": 12,
        "feature_mask": None,
    }


def test_checkpoint_user_config_retains_active_source_path_as_provenance(
    tmp_path: Path,
):
    path = tmp_path / "clusters.yaml"
    cfg = OmegaConf.create(
        {
            "depth": 12,
            "feature_clusters": {"path": str(path)},
        }
    )
    actual = _checkpoint_user_config(cfg)
    assert actual["feature_clusters"] == {"path": str(path)}


def test_documented_cluster_override_composes_from_null_default(
    hydra_compose,
    tmp_path: Path,
):
    cluster_path = tmp_path / "clusters.yaml"
    cfg = hydra_compose(
        TRAINING_CONFIG_DIR,
        "conditional_encdec_train",
        [
            f"+feature_clusters.path={cluster_path}",
            "feature_delta=null",
        ],
    )
    assert cfg.feature_clusters.path == str(cluster_path)
    assert cfg.feature_delta is None


@pytest.mark.parametrize(
    ("relative_path", "override"),
    [
        ("docs/cli-reference.md", "feature_clusters.path=<clusters.yaml>"),
        (
            "docs/pipelines/encoder-decoder-training.md",
            "feature_clusters.path=<clusters.yaml>",
        ),
        ("docs/reference/file-reference.md", "feature_clusters.path=clusters.yaml"),
    ],
)
def test_documented_cluster_cli_uses_hydra_add_override(
    relative_path: str,
    override: str,
):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert override not in source.replace(f"+{override}", "")
    assert f"+{override}" in source


def _cluster_state(
    tmp_path: Path,
    *,
    mask: FeatureMask | None,
) -> FeatureClusterConsolidation:
    path = tmp_path / "summary-clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 2, 3]]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        8,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _synthetic_identity_cluster_state(
    tmp_path: Path,
    *,
    mask: FeatureMask | None,
) -> FeatureClusterConsolidation:
    ordinary = _cluster_state(tmp_path, mask=mask)
    allowed = t.ones(8, dtype=t.bool) if mask is None else mask.allowed_universe_mask(8)
    return replace(
        ordinary,
        rep_of=t.arange(8, dtype=t.int64),
        consolidated_away=t.zeros(8, dtype=t.bool),
        allowed_universe=allowed,
        cluster_allowed_universe=allowed,
        is_identity_after_mask=True,
    )


def test_cluster_aware_no_compaction_reasons_are_truthful(tmp_path: Path):
    assert no_compaction_reason(None, 8) == "no feature_mask is configured"

    all_allowing = FeatureMask(FeatureMaskMode.whitelist, tuple(range(8)))
    legacy = no_compaction_reason(all_allowing, 8)
    assert legacy == (
        "the configured feature_mask (mode=whitelist) allows ALL 8 features, so it "
        "restricts nothing to compact"
    )

    inert_unmasked = _synthetic_identity_cluster_state(tmp_path, mask=None)
    assert no_compaction_reason(
        None,
        8,
        feature_clusters=inert_unmasked,
    ) == ("no feature_mask is configured and the configured feature_clusters consolidate nothing")

    inert_all_allowing = _synthetic_identity_cluster_state(
        tmp_path,
        mask=all_allowing,
    )
    assert "feature_clusters consolidate nothing after the mask" in no_compaction_reason(
        all_allowing,
        8,
        feature_clusters=inert_all_allowing,
    )


def test_compaction_summary_reports_cluster_provenance_count():
    vocab = FeatureVocab(8, (0, 1, 4, 5, 6))
    model = SimpleNamespace(
        feature_vocab=vocab,
        encoder=SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=8,
                n_embd=4,
                n_kv_head=1,
                head_dim=4,
            ),
            value_embeds={"0": object()},
        ),
    )
    summary = format_compaction_summary(
        model,  # ty: ignore[invalid-argument-type]
        no_vocab_reason="unused",
        consolidated_away=2,
    )
    assert "5/8 rows kept" in summary
    assert "Of the dropped rows, 2 were consolidated away by feature_clusters." in summary


def test_effective_vocab_uses_the_consolidations_embedded_mask(tmp_path: Path):
    built = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4))
    state = _cluster_state(tmp_path, mask=built)
    vocab = effective_feature_vocab(state, 8)
    assert vocab is not None
    assert set(vocab.original_ids) <= set(built.ids)


def test_effective_vocab_width_guard_names_both_sides(tmp_path: Path):
    state = _cluster_state(tmp_path, mask=None)
    with pytest.raises(
        FeatureClustersError,
        match=r"feature_clusters\.n_sae_features=8, n_sae_features=9",
    ):
        effective_feature_vocab(state, 9)


def test_trainer_wires_effective_vocab_summary_and_resume_order():
    source = (REPO_ROOT / "scripts/train/conditional_encdec_train.py").read_text(encoding="utf-8")
    assert "shared_feature_vocab = effective_feature_vocab(" in source
    assert "per_config_vocab = effective_feature_vocab(" in source
    assert "consolidated_away=(" in source
    cluster_guard = source.index("assert_resume_feature_clusters_match(")
    migration_guard = source.index(
        "_assert_not_pre_effective_compaction_checkpoint(",
        cluster_guard,
    )
    generic_guard = source.index("assert_checkpoint_vocab_compatible(", migration_guard)
    assert cluster_guard < migration_guard < generic_guard
