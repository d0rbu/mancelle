"""Contracts for self-conditioning SFT data-source resolution."""

from __future__ import annotations

import inspect
import json
import re
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch as t

import scripts.train.conditional_encdec_selfcond as selfcond_script
import scripts.train.conditional_encdec_sft as sft_script
from errors import DataLoadError, SelfCondTrainingError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from scripts.train.conditional_encdec_selfcond import (
    SelfCondSftData,
    SelfCondSftEvalConfig,
    SelfCondSftSources,
    assert_resume_selfcond_data_matches,
    build_selfcond_sft_data,
    clone_val_bundles,
    resolve_selfcond_data_source,
    run_sft_validation_cycle,
)
from scripts.train.conditional_encdec_sft import _build_sft_val_loader
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondDataSource,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
    resolve_feature_clusters_from_checkpoint_meta,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    NegativeBundle,
    NegativeSamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.axbench_upstream import run_methods
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    sft_phase_for,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft.segment_mode_identity import (
    ONLINE_SAE_LINEAGE_META_KEY,
    ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY,
    SFT_SEGMENT_MODE_META_KEY,
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    resolve_turn_self_cond_segment_size_rustbpe,
)
from steering.sft_schema import SFTSplit
from steering.stats_io import save_tokenizer_checksum
from tasks.mixtures import SFTMixture
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.scripts.test_conditional_encdec_selfcond_startup import (
    _config,
    _recorded_selfcond_block,
    _sft_shaped_meta,
)
from tests.steering.selfcond.helpers import tiny_head_model
from tests.steering.test_sft_features import _write_parquet, _write_shard


def _canonical_feature_dir(
    tmp_path: Path,
    phase: DatasetPhase,
    *,
    segment_size: int = 128,
    root_name: str = "features",
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> Path:
    return build_feature_dir(
        root=tmp_path / root_name,
        segment_size=segment_size,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=phase,
        segment_mode=segment_mode,
    )


def _populate_val_artifacts(feature_dir: Path, parquet_dir: Path) -> None:
    feature_dir.mkdir(parents=True)
    parquet_dir.mkdir(parents=True)
    _write_shard(feature_dir, "shard_00000", doc_ids=[0])
    _write_parquet(parquet_dir, "shard_0000", [0])


def _meta_with_feature_dir(feature_dir: str | Path) -> dict:
    meta = deepcopy(_sft_shaped_meta())
    meta["user_config"]["feature_dir"] = str(feature_dir)
    meta[SFT_SEGMENT_MODE_META_KEY] = SegmentMode.fixed_tokens.value
    return meta


def _write_steered_jsonl(tmp_path: Path, rows: int = 4) -> str:
    path = tmp_path / "steered.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt {index}"},
                    {"role": "assistant", "content": f"steered answer {index}"},
                ]
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return str(path)


def _steered_base_meta(
    tmp_path: Path,
    dataset: str,
    phase: DatasetPhase,
    *,
    steered_jsonl: str | None,
) -> dict:
    """An SFT leaf as recorded by a steered ``conditional_encdec_sft`` run."""
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, phase))
    meta["dataset"] = dataset
    meta["user_config"]["steered_jsonl"] = steered_jsonl
    return meta


def _compact_checkpoint_meta(
    *,
    n_sae_features: int = 64,
    mask_ids: tuple[int, ...] = tuple(range(16)),
    clusters: tuple[tuple[int, ...], ...] = (
        (0, 1, 2),
        (4, 5, 6),
        (8, 9),
    ),
) -> tuple[
    dict[str, object],
    FeatureMask,
    FeatureClusterConsolidation,
    FeatureVocab,
]:
    """Build the small real-shaped compact identity shared with the E2E test."""
    mask = FeatureMask(FeatureMaskMode.whitelist, mask_ids)
    definition = FeatureClusterDefinition(clusters=clusters)
    resolved = ResolvedFeatureClusters(
        source_path=Path("/recorded/compact-clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved, mask)
    feature_vocab = effective_feature_vocab(consolidation, n_sae_features)
    if feature_vocab is None:
        raise AssertionError("compact checkpoint fixture unexpectedly resolved to identity")

    meta = deepcopy(_sft_shaped_meta())
    meta["feature_mask_canonical"] = mask.canonical_dict()
    meta.update(feature_clusters_checkpoint_metadata(resolved))
    meta["feature_vocab"] = feature_vocab.metadata_dict()
    # The SFT trainer records the compact FP head width beside the vocab.
    meta["feature_prediction_width"] = int(feature_vocab.size)
    user_config = meta["user_config"]
    assert isinstance(user_config, dict)
    user_config["feature_mask"] = mask.canonical_dict()
    user_config["feature_clusters"] = {"path": str(resolved.source_path)}
    return meta, mask, consolidation, feature_vocab


def _compact_sources(tmp_path: Path, *, n_sae_features: int = 64) -> SelfCondSftSources:
    feature_dir = tmp_path / "compact_features"
    stats_path = write_steering_stats(
        feature_dir / "feature_statistics.safetensors",
        n_sae_features=n_sae_features,
    )
    save_tokenizer_checksum(stats_path, "compact-test-tokenizer")
    return SelfCondSftSources(
        dataset=SFTMixture.SAFETY,
        train_parquet_dir="/synthetic/safety",
        train_feature_dir=str(feature_dir),
        val_parquet_dir="/synthetic/safety_val",
        val_feature_dir="/synthetic/features/safety_val",
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        feature_source_bos_char_len=0,
        alignment_probe_samples=0,
        segment_mode=SegmentMode.fixed_tokens,
    )


def _compact_sampling_identity() -> dict[str, object]:
    return {
        "feature_selection": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 1,
            "max_k": 4,
            "mode": "top_k_percentile",
            "delta_weights": None,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 0,
            "max_k": 3,
        },
    }


def test_compact_reconstruction_is_bound_to_the_state_dict_vocab(
    tmp_path: Path,
) -> None:
    """The state dict is the authority: internally consistent metadata mirrors
    with the SAME width but DIFFERENT ids must be rejected (review finding)."""
    meta, _mask, _consolidation, feature_vocab = _compact_checkpoint_meta()
    forged_ids = tuple(i + 32 for i in range(feature_vocab.size))
    forged_state_vocab = FeatureVocab(
        n_sae_features=64,
        original_ids=forged_ids,
    )
    assert forged_state_vocab.size == feature_vocab.size

    with pytest.raises(SelfCondTrainingError, match="state dict's FeatureVocab"):
        build_selfcond_sft_data(
            sources=_compact_sources(tmp_path),
            base_meta=meta,
            identity=_compact_sampling_identity(),
            n_sae_features=64,
            num_buckets=2,
            ddp_rank=0,
            tokenizer=object(),
            state_feature_vocab=forged_state_vocab,
        )


def test_compact_state_dict_without_canonical_metadata_is_corrupt(
    tmp_path: Path,
) -> None:
    """A physically compact checkpoint whose canonical mirrors were stripped
    must fail at reconstruction, not later at the encoder OOV assert."""
    meta, _mask, _consolidation, feature_vocab = _compact_checkpoint_meta()
    for key in (
        "feature_vocab",
        "feature_mask_canonical",
        "feature_clusters_descriptor",
        "feature_clusters_canonical",
    ):
        meta.pop(key, None)
    user_config = meta["user_config"]
    assert isinstance(user_config, dict)
    user_config.pop("feature_mask", None)
    user_config.pop("feature_clusters", None)

    with pytest.raises(SelfCondTrainingError, match="none of the canonical keys"):
        build_selfcond_sft_data(
            sources=_compact_sources(tmp_path),
            base_meta=meta,
            identity=_compact_sampling_identity(),
            n_sae_features=64,
            num_buckets=2,
            ddp_rank=0,
            tokenizer=object(),
            state_feature_vocab=feature_vocab,
        )


def test_canonical_metadata_without_state_vocab_is_corrupt(tmp_path: Path) -> None:
    meta, _mask, _consolidation, _feature_vocab = _compact_checkpoint_meta()
    with pytest.raises(SelfCondTrainingError, match="no.*FeatureVocab buffers"):
        build_selfcond_sft_data(
            sources=_compact_sources(tmp_path),
            base_meta=meta,
            identity=_compact_sampling_identity(),
            n_sae_features=64,
            num_buckets=2,
            ddp_rank=0,
            tokenizer=object(),
            state_feature_vocab=None,
        )


def _state_vocab_for(meta: object) -> FeatureVocab:
    """The fixture's AUTHORITATIVE state-dict vocab.

    Deliberately independent of ``meta``: corruption tests mutate the
    metadata mirrors while the (simulated) state dict stays intact — exactly
    the divergence the state-binding guard exists to catch.
    """
    del meta
    _, _, _, feature_vocab = _compact_checkpoint_meta()
    return feature_vocab


def _resolve_sources(
    *,
    selfcond_block: dict,
    base_meta: dict,
    base_dir: Path,
    eval_every: int,
    segment_size: int = 128,
    base_checkpoint_dir: str = "/checkpoints/sft-leaf-under-test",
):
    return selfcond_script.resolve_selfcond_sft_sources(
        selfcond_block,
        base_meta,
        str(base_dir),
        base_checkpoint_dir=base_checkpoint_dir,
        eval_every=eval_every,
        segment_size=segment_size,
    )


@pytest.mark.parametrize("mixture", sorted(selfcond_script.STEERED_MIXTURES))
@pytest.mark.parametrize("records_jsonl", [True, False])
def test_every_steered_checkpoint_is_rejected_before_any_artifact_preflight(
    tmp_path: Path,
    mixture: SFTMixture,
    records_jsonl: bool,
) -> None:
    """Steered mixtures are unsupported in SFT mode, whatever they recorded.

    The SFT generator opens a new feature-prediction group at each steered
    conversation's assistant turn; the rollout would have to reproduce that
    split slot-for-slot. The rejection is CATEGORICAL — it fires for a leaf
    that recorded its ``steered_jsonl`` and for one that did not — and it
    fires before any artifact preflight: no val parquet or feature dir exists
    under ``tmp_path`` here, so a later rejection would surface the val
    artifact error instead of this message.
    """
    phase = sft_phase_for(mixture.value, SFTSplit.TRAIN)
    meta = _steered_base_meta(
        tmp_path,
        mixture.value,
        phase,
        steered_jsonl=_write_steered_jsonl(tmp_path) if records_jsonl else None,
    )

    with pytest.raises(
        SelfCondTrainingError,
        match=(
            "self-conditioning SFT mode does not support steered mixtures "
            + re.escape(f"(dataset={mixture.value!r})")
        ),
    ):
        _resolve_sources(
            selfcond_block={},
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=1,
            base_checkpoint_dir="/checkpoints/steered-leaf",
        )


def test_a_non_steered_checkpoint_is_unaffected_by_the_steered_rejection(
    tmp_path: Path,
) -> None:
    """Non-vacuity: the same fixture shape resolves for a plain mixture.

    Without this, the rejection test above would pass even if resolution
    raised for every dataset.
    """
    _populate_val_artifacts(
        _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT_VAL),
        tmp_path / "sft_data" / "safety_val",
    )
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=1,
        base_checkpoint_dir="/checkpoints/plain-leaf",
    )

    assert sources.dataset is SFTMixture.SAFETY
    assert sources.val_parquet_dir == str(tmp_path / "sft_data" / "safety_val")


def test_selfcond_word_list_preflight_covers_only_the_spelling_mixtures() -> None:
    """The steered members are rejected upstream, so the gate is the plain pair.

    Pinned at source level because the branch is inside ``main()``'s SFT arm,
    which the resolver-level tests never enter.
    """
    source = Path(selfcond_script.__file__).read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", "", source)
    assert "ifsources.datasetin{SFTMixture.STANDARD,SFTMixture.CAPABILITIES}:" in normalized
    assert "STEERED_MIXTURES" not in normalized.split("defmain(")[-1]


def test_selfcond_carries_no_steered_plumbing_after_the_rejection() -> None:
    """The rejection is the SINGLE mechanism: nothing downstream is steered-aware.

    ``SelfCondSftSources`` has no ``steered_jsonl`` field, the mixture is built
    with no overrides, and no identity key is recorded — otherwise the dead
    branches would rot untested behind an unreachable dataset.
    """
    assert "steered_jsonl" not in {field.name for field in fields(SelfCondSftSources)}
    source = Path(selfcond_script.__file__).read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", "", source)
    assert "steered_mixture_overrides" not in normalized
    assert "build_task_mixtures(sources.dataset)" in normalized
    assert 'selfcond_data_dict["steered_jsonl"]' not in normalized


def test_resolver_shape_error_names_the_checkpoint_leaf_not_base_dir(
    tmp_path: Path,
) -> None:
    """The D1 error must point at the actual checkpoint dir, not base_dir."""
    with pytest.raises(SelfCondTrainingError, match="/checkpoints/pretrain-leaf"):
        _resolve_sources(
            selfcond_block={},
            base_meta={"user_config": {"encoder": {}}},
            base_dir=tmp_path,
            eval_every=0,
            base_checkpoint_dir="/checkpoints/pretrain-leaf",
        )


def test_data_source_parses_and_rejects_unknown() -> None:
    assert (
        resolve_selfcond_data_source({"data_source": "pretraining"})
        is SelfCondDataSource.PRETRAINING
    )
    assert (
        resolve_selfcond_data_source({"data_source": "sft", "sft_dataset": "safety"})
        is SelfCondDataSource.SFT
    )

    with pytest.raises(
        SelfCondTrainingError,
        match=r"data_source.*packed.*pretraining.*sft",
    ):
        resolve_selfcond_data_source({"data_source": "packed"})


@pytest.mark.parametrize(
    "key",
    [
        "sft_dataset",
        "sft_feature_dir",
        "sft_parquet_dir",
        "val_sft_feature_dir",
        "val_sft_parquet_dir",
    ],
)
def test_pretraining_mode_rejects_each_stray_sft_key(key: str) -> None:
    with pytest.raises(SelfCondTrainingError, match=key):
        resolve_selfcond_data_source({"data_source": "pretraining", key: "/explicit/value"})


def test_sft_mode_accepts_sft_specific_keys() -> None:
    block = {
        "data_source": "sft",
        "sft_dataset": "capabilities",
        "sft_feature_dir": "/features/train",
        "sft_parquet_dir": "/parquet/train",
        "val_sft_feature_dir": "/features/val",
        "val_sft_parquet_dir": "/parquet/val",
    }
    assert resolve_selfcond_data_source(block) is SelfCondDataSource.SFT


# kills: accepting a configured cadence again -- an operator-declared value
# would silently override the measurement for EVERY source/mode combination,
# including an explicit null, which is what the removed knob's yaml default was.
@pytest.mark.parametrize(
    ("data_source", "mode"),
    [
        (SelfCondDataSource.PRETRAINING, SegmentMode.fixed_tokens),
        (SelfCondDataSource.SFT, SegmentMode.fixed_tokens),
        (SelfCondDataSource.SFT, SegmentMode.turn_delimited_fixed_tokens),
    ],
)
@pytest.mark.parametrize("raw", [None, 32, "32"])
def test_configured_turn_cadence_is_rejected_everywhere(
    data_source: SelfCondDataSource,
    mode: SegmentMode,
    raw: object,
) -> None:
    with pytest.raises(
        SelfCondTrainingError,
        match=r"is not a config knob.*measured from the training artifacts",
    ):
        selfcond_script.assert_turn_cadence_is_not_configured(
            {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: raw},
            data_source=data_source,
            segment_mode=mode,
        )


# kills: dropping the presence test's key entirely, which would let every
# supported (source, mode) pair pass the guard vacuously.
@pytest.mark.parametrize(
    ("data_source", "mode"),
    [
        (SelfCondDataSource.PRETRAINING, SegmentMode.fixed_tokens),
        (SelfCondDataSource.SFT, SegmentMode.fixed_tokens),
        (SelfCondDataSource.SFT, SegmentMode.turn_delimited_fixed_tokens),
    ],
)
def test_unconfigured_turn_cadence_is_accepted(
    data_source: SelfCondDataSource,
    mode: SegmentMode,
) -> None:
    assert (
        selfcond_script.assert_turn_cadence_is_not_configured(
            {"data_source": data_source.value},
            data_source=data_source,
            segment_mode=mode,
        )
        is None
    )


def test_turn_checkpoint_cannot_use_pretraining_selfcond_source() -> None:
    with pytest.raises(SelfCondTrainingError, match=r"turn-delimited.*pretraining"):
        selfcond_script.assert_turn_cadence_is_not_configured(
            {},
            data_source=SelfCondDataSource.PRETRAINING,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )


def test_sources_recover_every_field_from_sft_checkpoint(
    tmp_path: Path,
) -> None:
    train_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    val_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT_VAL)
    val_parquet_dir = tmp_path / "sft_data" / "safety_val"
    _populate_val_artifacts(val_feature_dir, val_parquet_dir)

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=_meta_with_feature_dir(train_feature_dir),
        base_dir=tmp_path,
        eval_every=1,
    )

    assert sources.dataset is SFTMixture.SAFETY
    assert sources.train_feature_dir == str(train_feature_dir)
    assert sources.train_parquet_dir == str(tmp_path / "sft_data" / "safety")
    assert sources.val_feature_dir == str(val_feature_dir)
    assert sources.val_parquet_dir == str(val_parquet_dir)
    assert sources.loss_mask_mode is LossMaskMode.FULL_SEQUENCE
    assert sources.min_loss_tokens == 8
    assert sources.feature_source_bos_char_len == 5
    assert sources.alignment_probe_samples == 8
    assert sources.segment_mode is SegmentMode.fixed_tokens


def test_structure_guard_accepts_matching_turn_mode(tmp_path: Path) -> None:
    train_feature_dir = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.SAFETY_SFT,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    meta = _meta_with_feature_dir(train_feature_dir)
    meta[SFT_SEGMENT_MODE_META_KEY] = SegmentMode.turn_delimited_fixed_tokens.value

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.segment_mode is SegmentMode.turn_delimited_fixed_tokens


@pytest.mark.parametrize(
    ("directory_mode", "checkpoint_mode"),
    [
        (SegmentMode.turn_delimited_fixed_tokens, SegmentMode.fixed_tokens),
        (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
    ],
)
def test_structure_guard_rejects_segment_mode_mismatch(
    tmp_path: Path,
    directory_mode: SegmentMode,
    checkpoint_mode: SegmentMode,
) -> None:
    feature_dir = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.SAFETY_SFT,
        segment_mode=directory_mode,
    )
    meta = _meta_with_feature_dir(feature_dir)
    meta[SFT_SEGMENT_MODE_META_KEY] = checkpoint_mode.value

    with pytest.raises(
        SelfCondTrainingError,
        match=(
            rf"segment_mode='{directory_mode.value}'.*"
            rf"records segment_mode='{checkpoint_mode.value}'"
        ),
    ):
        _resolve_sources(
            selfcond_block={},
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=0,
        )


def test_legacy_segment_mode_defaults_to_fixed_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        selfcond_script.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    del meta[SFT_SEGMENT_MODE_META_KEY]
    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.segment_mode is SegmentMode.fixed_tokens
    assert any(
        "no sft_segment_mode" in message and "fixed_tokens" in message for message in warnings
    )


def test_unknown_base_segment_mode_is_rejected(tmp_path: Path) -> None:
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    meta[SFT_SEGMENT_MODE_META_KEY] = "bogus"

    with pytest.raises(
        SelfCondTrainingError,
        match=(
            r"checkpoint .*invalid SFT segmentation metadata.*"
            r"unknown sft_segment_mode='bogus'.*"
            r"fixed_tokens.*turn_delimited_fixed_tokens"
        ),
    ):
        _resolve_sources(
            selfcond_block={},
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=0,
        )


def test_selfcond_data_identity_rejects_segment_mode_change() -> None:
    recorded = {
        "selfcond_data": {
            SFT_SEGMENT_MODE_META_KEY: SegmentMode.fixed_tokens.value,
        }
    }
    current = {
        SFT_SEGMENT_MODE_META_KEY: SegmentMode.turn_delimited_fixed_tokens.value,
    }

    with pytest.raises(
        SelfCondTrainingError,
        match=r"sft_segment_mode.*fixed_tokens.*turn_delimited_fixed_tokens",
    ):
        assert_resume_selfcond_data_matches(recorded, current)


def test_legacy_selfcond_data_identity_requires_fresh_sft_base() -> None:
    with pytest.raises(
        SelfCondTrainingError,
        match=(
            r"(?s)sft_segment_mode: checkpoint='<absent>' current='fixed_tokens'.*"
            r"start a fresh run from the SFT base"
        ),
    ):
        assert_resume_selfcond_data_matches(
            {"selfcond_data": {"dataset": "safety"}},
            {
                "dataset": "safety",
                SFT_SEGMENT_MODE_META_KEY: SegmentMode.fixed_tokens.value,
            },
        )


def test_explicit_overrides_win_and_every_resolved_value_is_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_train = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT, root_name="recorded")
    explicit_train = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT, root_name="explicit")
    explicit_val = _canonical_feature_dir(
        tmp_path, DatasetPhase.SAFETY_SFT_VAL, root_name="explicit"
    )
    train_parquet = tmp_path / "explicit_train_parquet"
    val_parquet = tmp_path / "explicit_val_parquet"
    _populate_val_artifacts(explicit_val, val_parquet)
    info_messages: list[str] = []
    monkeypatch.setattr(
        selfcond_script.logger,
        "info",
        lambda message: info_messages.append(str(message)),
    )

    sources = _resolve_sources(
        selfcond_block={
            "sft_dataset": "safety",
            "sft_feature_dir": str(explicit_train),
            "sft_parquet_dir": str(train_parquet),
            "val_sft_feature_dir": str(explicit_val),
            "val_sft_parquet_dir": str(val_parquet),
        },
        base_meta=_meta_with_feature_dir(recorded_train),
        base_dir=tmp_path,
        eval_every=1,
    )

    assert sources.train_feature_dir == str(explicit_train)
    assert sources.train_parquet_dir == str(train_parquet)
    assert sources.val_feature_dir == str(explicit_val)
    assert sources.val_parquet_dir == str(val_parquet)
    joined = "\n".join(info_messages)
    for label, value in (
        ("dataset", "safety"),
        ("train_feature_dir", explicit_train),
        ("train_parquet_dir", train_parquet),
        ("val_feature_dir", explicit_val),
        ("val_parquet_dir", val_parquet),
        ("loss_mask_mode", "full_sequence"),
        ("segment_mode", SegmentMode.fixed_tokens.value),
        ("min_loss_tokens", "8"),
        ("feature_source_bos_char_len", "5"),
        ("alignment_probe_samples", "8"),
    ):
        assert label in joined
        assert str(value) in joined


@pytest.mark.parametrize("explicit", [False, True])
def test_pretraining_phase_feature_dir_is_rejected_for_recovered_and_explicit_paths(
    tmp_path: Path,
    explicit: bool,
) -> None:
    pretraining_dir = _canonical_feature_dir(tmp_path, DatasetPhase.BASE)
    meta = _meta_with_feature_dir(
        pretraining_dir
        if not explicit
        else _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT, root_name="recorded")
    )
    block = {"sft_feature_dir": str(pretraining_dir)} if explicit else {}

    with pytest.raises(SelfCondTrainingError) as exc_info:
        _resolve_sources(
            selfcond_block=block,
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=0,
        )

    message = str(exc_info.value)
    assert DatasetPhase.BASE.value in message
    assert DatasetPhase.SAFETY_SFT.value in message


def test_symlinked_canonical_sft_name_is_guarded_by_resolved_phase(
    tmp_path: Path,
) -> None:
    real_base_dir = _canonical_feature_dir(tmp_path, DatasetPhase.BASE)
    _write_shard(real_base_dir, "shard_00000", doc_ids=[0])
    safety_symlink = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    safety_symlink.parent.mkdir(parents=True, exist_ok=True)
    safety_symlink.symlink_to(real_base_dir, target_is_directory=True)
    correctly_recorded = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.SAFETY_SFT,
        root_name="recorded",
    )

    cases = (
        ({}, _meta_with_feature_dir(safety_symlink)),
        (
            {"sft_feature_dir": str(safety_symlink)},
            _meta_with_feature_dir(correctly_recorded),
        ),
    )
    for selfcond_block, base_meta in cases:
        with pytest.raises(SelfCondTrainingError, match="phase"):
            _resolve_sources(
                selfcond_block=selfcond_block,
                base_meta=base_meta,
                base_dir=tmp_path,
                eval_every=0,
            )


def test_explicit_val_feature_dir_rejects_base_phase_after_preflight(
    tmp_path: Path,
) -> None:
    train_feature_dir = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.SAFETY_SFT,
        root_name="train_features",
    )
    val_feature_dir = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.BASE,
        root_name="val_features",
    )
    val_parquet_dir = tmp_path / "val_parquet"
    _populate_val_artifacts(val_feature_dir, val_parquet_dir)

    with pytest.raises(SelfCondTrainingError, match=r"validation.*phase"):
        _resolve_sources(
            selfcond_block={
                "val_sft_feature_dir": str(val_feature_dir),
                "val_sft_parquet_dir": str(val_parquet_dir),
            },
            base_meta=_meta_with_feature_dir(train_feature_dir),
            base_dir=tmp_path,
            eval_every=1,
        )


def test_explicit_val_feature_dir_rejects_wrong_sft_phase(
    tmp_path: Path,
) -> None:
    train_feature_dir = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.SAFETY_SFT,
        root_name="train_features",
    )
    wrong_val_feature_dir = _canonical_feature_dir(
        tmp_path,
        DatasetPhase.CAPABILITIES_SFT_VAL,
        root_name="val_features",
    )
    val_parquet_dir = tmp_path / "val_parquet"
    _populate_val_artifacts(wrong_val_feature_dir, val_parquet_dir)

    with pytest.raises(SelfCondTrainingError, match=r"validation.*phase"):
        _resolve_sources(
            selfcond_block={
                "val_sft_feature_dir": str(wrong_val_feature_dir),
                "val_sft_parquet_dir": str(val_parquet_dir),
            },
            base_meta=_meta_with_feature_dir(train_feature_dir),
            base_dir=tmp_path,
            eval_every=1,
        )


def test_noncanonical_recovered_feature_dir_is_corrupt_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SelfCondTrainingError,
        match=r"recovered.*non-canonical|non-canonical.*checkpoint",
    ):
        _resolve_sources(
            selfcond_block={},
            base_meta=_meta_with_feature_dir(tmp_path / "legacy_features"),
            base_dir=tmp_path,
            eval_every=0,
        )


def test_noncanonical_explicit_feature_dir_is_escape_hatch_with_explicit_val_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_train = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    explicit_train = tmp_path / "legacy" / "train_features"
    explicit_val = tmp_path / "legacy" / "val_features"
    val_parquet = tmp_path / "legacy" / "val_parquet"
    _populate_val_artifacts(explicit_val, val_parquet)
    warnings: list[str] = []
    monkeypatch.setattr(
        selfcond_script.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )

    sources = _resolve_sources(
        selfcond_block={
            "sft_feature_dir": str(explicit_train),
            "val_sft_feature_dir": str(explicit_val),
            "val_sft_parquet_dir": str(val_parquet),
        },
        base_meta=_meta_with_feature_dir(recorded_train),
        base_dir=tmp_path,
        eval_every=1,
    )

    assert sources.train_feature_dir == str(explicit_train)
    assert sources.val_feature_dir == str(explicit_val)
    assert sources.val_parquet_dir == str(val_parquet)
    # Both explicit non-canonical dirs warn independently: the escape hatch
    # covers the train dir AND the validation dir (review finding: explicit
    # val overrides previously received no structural guard at all).
    assert len(warnings) == 2
    assert "non-canonical SFT train feature dir accepted" in warnings[0]
    assert "non-canonical SFT validation feature dir accepted" in warnings[1]
    for message in warnings:
        assert "phase isolation cannot be verified" in message


@pytest.mark.parametrize(
    "val_overrides",
    [
        {},
        {"val_sft_feature_dir": "/explicit/val/features"},
        {"val_sft_parquet_dir": "/explicit/val/parquet"},
    ],
)
def test_noncanonical_explicit_feature_dir_requires_both_val_overrides_when_eval_on(
    tmp_path: Path,
    val_overrides: dict[str, str],
) -> None:
    block = {"sft_feature_dir": str(tmp_path / "noncanonical"), **val_overrides}
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))

    with pytest.raises(SelfCondTrainingError) as exc_info:
        _resolve_sources(
            selfcond_block=block,
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=1,
        )

    message = str(exc_info.value)
    assert "val_sft_feature_dir" in message
    assert "val_sft_parquet_dir" in message


def test_noncanonical_explicit_feature_dir_needs_no_val_overrides_when_eval_off(
    tmp_path: Path,
) -> None:
    explicit_train = tmp_path / "noncanonical"
    sources = _resolve_sources(
        selfcond_block={"sft_feature_dir": str(explicit_train)},
        base_meta=_meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)),
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.train_feature_dir == str(explicit_train)
    assert sources.val_feature_dir is None
    assert sources.val_parquet_dir is None


def test_canonical_feature_dir_segment_size_must_match_runtime(
    tmp_path: Path,
) -> None:
    train_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT, segment_size=256)
    with pytest.raises(SelfCondTrainingError, match=r"segment_size.*256.*128"):
        _resolve_sources(
            selfcond_block={},
            base_meta=_meta_with_feature_dir(train_feature_dir),
            base_dir=tmp_path,
            eval_every=0,
            segment_size=128,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda meta: meta["user_config"].update(encoder={"depth": 1}),
        lambda meta: meta.pop("encoder_config"),
        lambda meta: meta.pop("dataset"),
    ],
)
def test_sft_mode_rejects_each_non_sft_base_shape(
    tmp_path: Path,
    mutation,
) -> None:
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    mutation(meta)
    with pytest.raises(SelfCondTrainingError) as exc_info:
        selfcond_script.assert_sft_shaped_base(meta, "/checkpoint/leaf")

    message = str(exc_info.value)
    assert "base_checkpoint_dir=" in message
    assert "/checkpoint/leaf" in message
    assert "SFT checkpoint leaf" in message
    assert "tag" in message


def test_explicit_dataset_must_match_recorded_dataset(tmp_path: Path) -> None:
    with pytest.raises(
        SelfCondTrainingError,
        match=r"capabilities.*safety|safety.*capabilities",
    ):
        _resolve_sources(
            selfcond_block={"sft_dataset": "capabilities"},
            base_meta=_meta_with_feature_dir(
                _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
            ),
            base_dir=tmp_path,
            eval_every=0,
        )


@pytest.mark.parametrize("explicit", [False, True])
def test_unknown_dataset_is_typed_and_lists_valid_mixtures(
    tmp_path: Path,
    explicit: bool,
) -> None:
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    block: dict[str, str] = {}
    if explicit:
        block["sft_dataset"] = "other"
    else:
        meta["dataset"] = "other"

    with pytest.raises(SelfCondTrainingError) as exc_info:
        _resolve_sources(
            selfcond_block=block,
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=0,
        )

    message = str(exc_info.value)
    assert "other" in message
    for mixture in SFTMixture:
        assert mixture.value in message


def test_lowest_missing_percentile_pilot_base_is_honored_with_pilot_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy continuity, not rejection: the recorded missing-percentile
    policy is conditioning identity (the owner's l22 production family is a
    `lowest` pilot), mirroring SFT's own resume contract. The NON-COMPARABLE
    pilot warning must fire and the policy must be carried on the sources."""
    from steering.conditional.missing_percentile import MissingPercentilePolicy

    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    meta["missing_percentile_policy"] = "lowest"
    warnings: list[str] = []
    monkeypatch.setattr(
        selfcond_script.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.missing_percentile_policy is MissingPercentilePolicy.LOWEST
    assert any("NOT comparable" in message for message in warnings)


def test_error_policy_default_when_base_records_none(tmp_path: Path) -> None:
    from steering.conditional.missing_percentile import MissingPercentilePolicy

    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    meta.pop("missing_percentile_policy", None)
    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=0,
    )
    assert sources.missing_percentile_policy is MissingPercentilePolicy.ERROR


def test_eval_on_requires_real_val_artifacts_and_names_both_dirs(
    tmp_path: Path,
) -> None:
    train_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    expected_val_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT_VAL)
    expected_val_parquet_dir = tmp_path / "sft_data" / "safety_val"

    with pytest.raises(DataLoadError) as exc_info:
        _resolve_sources(
            selfcond_block={},
            base_meta=_meta_with_feature_dir(train_feature_dir),
            base_dir=tmp_path,
            eval_every=1,
        )

    message = str(exc_info.value)
    assert str(expected_val_feature_dir) in message
    assert str(expected_val_parquet_dir) in message
    assert "parquet_ready=False" in message
    assert "feature_ready=False" in message


def test_eval_off_resolves_no_val_dirs_and_touches_no_artifacts(
    tmp_path: Path,
) -> None:
    sources = _resolve_sources(
        selfcond_block={},
        base_meta=_meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)),
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.val_parquet_dir is None
    assert sources.val_feature_dir is None
    assert not (tmp_path / "sft_data").exists()


def test_train_parquet_dir_recovers_checkpoint_override_before_default(
    tmp_path: Path,
) -> None:
    checkpoint_parquet = tmp_path / "checkpoint_train_parquet"
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    meta["user_config"]["sft_parquet_dir"] = str(checkpoint_parquet)

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.train_parquet_dir == str(checkpoint_parquet)


def test_missing_loss_mask_uses_legacy_default_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    del meta["sft_loss_mask"]
    warnings: list[str] = []
    monkeypatch.setattr(
        selfcond_script.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=0,
    )

    assert sources.loss_mask_mode is LossMaskMode.FULL_SEQUENCE
    assert len(warnings) == 1
    assert "sft_loss_mask" in warnings[0]
    assert "full_sequence" in warnings[0]


def test_legacy_thin_sft_base_names_pretrain_conditioning_remedy(
    tmp_path: Path,
) -> None:
    meta = _meta_with_feature_dir(_canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT))
    meta["pretrain_conditioning"]["negative_selection"] = None
    meta["user_config"].pop("negative_selection", None)

    with pytest.raises(SelfCondTrainingError, match="pretrain_conditioning"):
        _resolve_sources(
            selfcond_block={},
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=0,
        )


def test_threshold_selection_builds_precomputed_pipeline_and_fresh_val_rngs(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "threshold_features"
    stats_path = write_steering_stats(
        feature_dir / "feature_statistics.safetensors",
        n_sae_features=8,
    )
    save_tokenizer_checksum(stats_path, "threshold-test-tokenizer")
    sources = SelfCondSftSources(
        dataset=SFTMixture.SAFETY,
        train_parquet_dir="/synthetic/safety",
        train_feature_dir=str(feature_dir),
        val_parquet_dir=None,
        val_feature_dir=None,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        feature_source_bos_char_len=0,
        alignment_probe_samples=0,
        segment_mode=SegmentMode.fixed_tokens,
    )
    data = build_selfcond_sft_data(
        sources=sources,
        identity={
            "feature_selection": {
                "threshold": 0.9,
                "threshold_type": "feature_percentile",
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "min_k": 0,
                "max_k": 3,
            },
        },
        n_sae_features=8,
        num_buckets=2,
        ddp_rank=0,
        tokenizer=object(),
    )

    assert data.threshold_cfg is not None
    assert data.threshold_cfg.threshold_type is ThresholdType.feature_percentile
    assert float(data.threshold_cfg.threshold) == 0.9
    assert data.sampling_bundle is None
    assert data.precomputed_thresholds is not None
    assert data.precomputed_thresholds.shape == (8,)
    assert bool(t.isfinite(data.precomputed_thresholds).all())
    negative_bundle = data.pipeline.negative_bundle
    assert negative_bundle is not None
    original_rng_state = negative_bundle.rng.get_state().clone()

    sampling1, negative1 = clone_val_bundles(None, negative_bundle, seed=7)
    sampling2, negative2 = clone_val_bundles(None, negative_bundle, seed=7)

    assert sampling1 is sampling2 is None
    assert negative1 is not negative_bundle
    assert negative1.rng is not negative_bundle.rng
    assert negative1.rng is not negative2.rng
    first_draws = t.randint(0, 1000, (16,), generator=negative1.rng)
    second_draws = t.randint(0, 1000, (16,), generator=negative2.rng)
    assert t.equal(first_draws, second_draws)
    assert t.equal(negative_bundle.rng.get_state(), original_rng_state)


def test_compact_canonical_metadata_reconstructs_mask_clusters_and_negative_pool(
    tmp_path: Path,
) -> None:
    meta, expected_mask, expected_consolidation, feature_vocab = _compact_checkpoint_meta()
    sources = _compact_sources(tmp_path)

    data = build_selfcond_sft_data(
        sources=sources,
        base_meta=meta,
        identity=_compact_sampling_identity(),
        n_sae_features=64,
        num_buckets=2,
        ddp_rank=0,
        tokenizer=object(),
        state_feature_vocab=feature_vocab,
    )

    assert data.feature_mask == expected_mask
    assert data.pipeline.feature_mask == expected_mask
    assert data.consolidation is not None
    assert data.consolidation.resolved.definition == (expected_consolidation.resolved.definition)
    assert data.consolidation.resolved.descriptor == (expected_consolidation.resolved.descriptor)
    assert data.consolidation.resolved.source_path == Path("/recorded/compact-clusters.json")
    assert t.equal(
        data.consolidation.cluster_allowed_universe,
        expected_consolidation.cluster_allowed_universe,
    )
    representatives = tuple(
        int(feature_id)
        for feature_id in data.consolidation.cluster_allowed_universe.nonzero().flatten().tolist()
    )
    assert representatives == feature_vocab.original_ids
    assert data.cluster_threshold_percentile_table is None

    negative_bundle = data.pipeline.negative_bundle
    assert negative_bundle is not None
    assert negative_bundle.feature_mask == (expected_consolidation.effective_negative_mask())
    assert negative_bundle.feature_mask != expected_mask
    assert negative_bundle.feature_mask is not None
    allowed_negative_ids = negative_bundle.feature_mask.allowed_universe_mask(64)
    assert t.equal(
        allowed_negative_ids,
        expected_consolidation.cluster_allowed_universe,
    )


def test_compact_clustered_threshold_selection_builds_full_percentile_table(
    tmp_path: Path,
) -> None:
    meta, expected_mask, _expected_consolidation, _feature_vocab = _compact_checkpoint_meta()
    data = build_selfcond_sft_data(
        sources=_compact_sources(tmp_path),
        base_meta=meta,
        identity={
            "feature_selection": {
                "threshold": 0.9,
                "threshold_type": "feature_percentile",
            },
            "negative_selection": _compact_sampling_identity()["negative_selection"],
        },
        n_sae_features=64,
        num_buckets=2,
        ddp_rank=0,
        tokenizer=object(),
        state_feature_vocab=_state_vocab_for(meta),
    )

    assert data.feature_mask == expected_mask
    assert data.consolidation is not None
    assert data.threshold_cfg is not None
    assert data.sampling_bundle is None
    table = data.cluster_threshold_percentile_table
    assert table is not None
    assert table.shape[0] == 64
    assert table.ndim == 2
    # Match build_full_percentile_table exactly; the shared SFT path retains
    # the statistics table's float64 rank currency rather than casting it.
    assert table.dtype == t.float64


@pytest.mark.parametrize(
    ("removed_keys", "expected_missing"),
    [
        (("feature_mask_canonical",), ("feature_mask_canonical",)),
        (("feature_vocab",), ("feature_vocab",)),
        (("feature_clusters_descriptor",), ("feature_clusters_descriptor",)),
        (("feature_clusters_canonical",), ("feature_clusters_canonical",)),
        (
            ("feature_clusters_descriptor", "feature_clusters_canonical"),
            ("feature_clusters_descriptor", "feature_clusters_canonical"),
        ),
    ],
)
def test_compact_partial_canonical_metadata_fails_naming_every_missing_key(
    tmp_path: Path,
    removed_keys: tuple[str, ...],
    expected_missing: tuple[str, ...],
) -> None:
    meta, _mask, _consolidation, _feature_vocab = _compact_checkpoint_meta()
    for key in removed_keys:
        del meta[key]

    with pytest.raises(SelfCondTrainingError) as exc_info:
        build_selfcond_sft_data(
            sources=_compact_sources(tmp_path),
            base_meta=meta,
            identity=_compact_sampling_identity(),
            n_sae_features=64,
            num_buckets=2,
            ddp_rank=0,
            tokenizer=object(),
            state_feature_vocab=_state_vocab_for(meta),
        )

    message = str(exc_info.value)
    assert "corrupt" in message.lower()
    for key in expected_missing:
        assert key in message


@pytest.mark.parametrize(
    "null_key",
    [
        "feature_vocab",
        "feature_mask_canonical",
        "feature_clusters_descriptor",
        "feature_clusters_canonical",
    ],
)
def test_compact_null_canonical_metadata_is_corrupt_and_names_the_key(
    tmp_path: Path,
    null_key: str,
) -> None:
    meta, _mask, _consolidation, _feature_vocab = _compact_checkpoint_meta()
    meta[null_key] = None

    with pytest.raises(SelfCondTrainingError) as exc_info:
        build_selfcond_sft_data(
            sources=_compact_sources(tmp_path),
            base_meta=meta,
            identity=_compact_sampling_identity(),
            n_sae_features=64,
            num_buckets=2,
            ddp_rank=0,
            tokenizer=object(),
            state_feature_vocab=_state_vocab_for(meta),
        )

    assert null_key in str(exc_info.value)


def test_compact_val_generator_threading_and_clone_preserve_representative_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, expected_mask, expected_consolidation, _feature_vocab = _compact_checkpoint_meta()
    data = build_selfcond_sft_data(
        sources=_compact_sources(tmp_path),
        base_meta=meta,
        identity=_compact_sampling_identity(),
        n_sae_features=64,
        num_buckets=2,
        ddp_rank=0,
        tokenizer=object(),
        state_feature_vocab=_state_vocab_for(meta),
    )
    negative_bundle = data.pipeline.negative_bundle
    assert negative_bundle is not None
    expected_negative_mask = expected_consolidation.effective_negative_mask()
    assert negative_bundle.feature_mask == expected_negative_mask

    calls: list[dict[str, object]] = []

    def recording_generator(**kwargs):
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(
        sft_script,
        "make_sft_data_generator",
        recording_generator,
    )
    monkeypatch.setattr(
        selfcond_script,
        "evaluate_encdec_bpb",
        lambda *_args, **_kwargs: 1.25,
    )

    def fake_selfcond_eval(*_args, **kwargs):
        assert kwargs["consolidation"] is data.consolidation
        return 1.5

    monkeypatch.setattr(
        selfcond_script,
        "evaluate_selfcond_bpb_sft",
        fake_selfcond_eval,
    )

    result = run_sft_validation_cycle(
        model=tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=64),
        sources=_compact_sources(tmp_path),
        sft_data=data,
        val_task=object(),
        tokenizer=object(),
        cfg_eval=SelfCondSftEvalConfig(B=1, T=4, eval_steps=1, seed=1),
        selfcond_cfg=_config(),
        ddp_rank=0,
        ddp_world_size=1,
        device="cpu",
        token_bytes=t.ones(100, dtype=t.int64),
    )

    assert result == (1.25, 1.5)
    assert len(calls) == 2
    assert calls[0]["expected_split"] is SFTSplit.VAL
    for call in calls:
        assert call["feature_mask"] == expected_mask
        assert call["consolidation"] is data.consolidation
        assert call["cluster_threshold_percentile_table"] is None
        cloned_negative = call["negative_bundle"]
        assert isinstance(cloned_negative, NegativeBundle)
        assert cloned_negative is not negative_bundle
        assert cloned_negative.feature_mask is negative_bundle.feature_mask
        assert cloned_negative.feature_mask == expected_negative_mask
    assert calls[0]["negative_bundle"] is not calls[1]["negative_bundle"]


def test_validation_cycle_requires_resolved_val_dirs() -> None:
    negative_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(min_k=0, max_k=3),
        n_sae_features=8,
        rng=t.Generator().manual_seed(1),
    )
    selection = ThresholdSelection(
        threshold_type=ThresholdType.feature_percentile,
        threshold=0.9,
    )
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=selection,
            precomputed_thresholds=t.ones(8),
        ),
        boundaries=t.empty(8, 0, dtype=t.float64),
        negative_bundle=negative_bundle,
        effective_buckets=1,
    )
    sources = SelfCondSftSources(
        dataset=SFTMixture.SAFETY,
        train_parquet_dir="/train/parquet",
        train_feature_dir="/train/features",
        val_parquet_dir=None,
        val_feature_dir=None,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        feature_source_bos_char_len=0,
        alignment_probe_samples=0,
        segment_mode=SegmentMode.fixed_tokens,
    )
    sft_data = SelfCondSftData(
        pipeline=pipeline,
        threshold_cfg=None,
        precomputed_thresholds=None,
        sampling_bundle=None,
        boundaries=pipeline.boundaries,
        tokenizer_checksum="test",
        percentile_checksum=None,
    )

    with pytest.raises(SelfCondTrainingError, match="val_feature_dir"):
        run_sft_validation_cycle(
            model=tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8),
            sources=sources,
            sft_data=sft_data,
            val_task=object(),
            tokenizer=object(),
            cfg_eval=SelfCondSftEvalConfig(B=1, T=4, eval_steps=1, seed=1),
            selfcond_cfg=_config(),
            ddp_rank=0,
            ddp_world_size=1,
            device="cpu",
            token_bytes=t.ones(8, dtype=t.int64),
        )


def test_validation_cycle_requires_negative_bundle() -> None:
    selection = ThresholdSelection(
        threshold_type=ThresholdType.feature_percentile,
        threshold=0.9,
    )
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=selection,
            precomputed_thresholds=t.ones(8),
        ),
        boundaries=t.empty(8, 0, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=1,
    )
    sources = SelfCondSftSources(
        dataset=SFTMixture.SAFETY,
        train_parquet_dir="/train/parquet",
        train_feature_dir="/train/features",
        val_parquet_dir="/val/parquet",
        val_feature_dir="/val/features",
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        feature_source_bos_char_len=0,
        alignment_probe_samples=0,
        segment_mode=SegmentMode.fixed_tokens,
    )
    sft_data = SelfCondSftData(
        pipeline=pipeline,
        threshold_cfg=None,
        precomputed_thresholds=None,
        sampling_bundle=None,
        boundaries=pipeline.boundaries,
        tokenizer_checksum="test",
        percentile_checksum=None,
    )

    with pytest.raises(SelfCondTrainingError, match="C4"):
        run_sft_validation_cycle(
            model=tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8),
            sources=sources,
            sft_data=sft_data,
            val_task=object(),
            tokenizer=object(),
            cfg_eval=SelfCondSftEvalConfig(B=1, T=4, eval_steps=1, seed=1),
            selfcond_cfg=_config(),
            ddp_rank=0,
            ddp_world_size=1,
            device="cpu",
            token_bytes=t.ones(8, dtype=t.int64),
        )


def test_sft_feature_pipeline_seed_is_named_and_used_at_its_call_site() -> None:
    assert sft_script.SFT_FEATURE_PIPELINE_SEED == 42
    source = inspect.getsource(sft_script.main)
    assert "seed=SFT_FEATURE_PIPELINE_SEED" in source


def test_val_loader_threads_return_seg_doc_id_to_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_make_sft_data_generator(**kwargs):
        captured.update(kwargs)
        yield ("batch",)

    monkeypatch.setattr(
        sft_script,
        "make_sft_data_generator",
        fake_make_sft_data_generator,
    )

    loader = _build_sft_val_loader(
        val_task=object(),
        tokenizer=object(),
        B=2,
        T=16,
        device="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        feature_dir="/val/features",
        expected_feature_dir="/val/features",
        parquet_dir="/val/parquets",
        threshold_cfg=object(),
        boundaries=object(),
        precomputed_thresholds=None,
        sampling_bundle=None,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=8,
        n_sae_features=8,
        fp_percentile_table=t.empty(0),
        alignment_probe_samples=0,
        return_seg_doc_id=True,
    )

    assert next(loader) == ("batch",)
    assert captured["return_seg_doc_id"] is True
    assert captured["expected_split"] is SFTSplit.VAL


def _selfcond_data_identity() -> dict[str, object]:
    return {
        "dataset": "safety",
        "train_parquet_dir": "/data/sft_data/safety",
        "train_feature_dir": "/data/features/safety_sft",
        "val_parquet_dir": "/data/sft_data/safety_val",
        "val_feature_dir": "/data/features/safety_sft_val",
        "sft_loss_mask": "full_sequence",
        "min_loss_tokens": 8,
        "feature_source_bos_char_len": 5,
        "tokenizer_checksum": "tokenizer-sha256",
        SFT_SEGMENT_MODE_META_KEY: SegmentMode.fixed_tokens.value,
    }


def _online_lineage_identity(mode: SegmentMode) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": "gemma-2-2b",
        "release": "gemmascope",
        "activation_type": "residual",
        "layer": 8,
        "width": "16k",
        "l0": None,
        "source_segment_size": 128,
        ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY: mode.value,
        "pooling": "max_pool",
        "context_independent_features": True,
        "ignore_bos_features": True,
        "feature_dir": "/training-host/features",
        "statistics_path": "/training-host/features/feature_statistics.safetensors",
        "statistics_sha256": "a" * 64,
        "n_sae_features": 16_384,
        "num_buckets": 2,
        "feature_selection": {"mode": "top_k_percentile"},
        "negative_selection": None,
        "missing_percentile_policy": "error",
        "selection_seed": 42,
    }


def _sft_artifact_coverage() -> dict[str, object]:
    return {
        "dataset_size": 20,
        "parquet_mixture_fingerprint": {
            "dataset": "safety",
            "ordered_shard_sha256": "a" * 64,
        },
        "feature_mixture_fingerprint": {
            "dataset_phase": "safety_sft",
            "index_map_sha256": "b" * 64,
        },
    }


def _resume_meta_for_source(data_source: str | None) -> dict[str, object]:
    selfcond = _recorded_selfcond_block()
    if data_source is not None:
        selfcond["data_source"] = data_source
    return {"selfcond": selfcond}


@pytest.mark.parametrize(
    ("recorded_source", "current_source"),
    [
        (SelfCondDataSource.PRETRAINING.value, SelfCondDataSource.SFT),
        (SelfCondDataSource.SFT.value, SelfCondDataSource.PRETRAINING),
    ],
)
def test_resume_rejects_data_source_mismatch(
    recorded_source: str,
    current_source: SelfCondDataSource,
) -> None:
    with pytest.raises(SelfCondTrainingError, match="data_source"):
        selfcond_script.assert_resume_selfcond_matches(
            _resume_meta_for_source(recorded_source),
            _config(),
            data_source=current_source,
        )


def test_legacy_resume_defaults_absent_data_source_to_pretraining_with_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        selfcond_script.logger,
        "info",
        lambda message: messages.append(str(message)),
    )

    selfcond_script.assert_resume_selfcond_matches(
        _resume_meta_for_source(None),
        _config(),
        data_source=SelfCondDataSource.PRETRAINING,
    )

    assert len(messages) == 1
    assert "data_source" in messages[0]
    assert SelfCondDataSource.PRETRAINING.value in messages[0]
    assert "legacy" in messages[0].lower()


def test_legacy_resume_cannot_default_absent_data_source_to_sft() -> None:
    with pytest.raises(SelfCondTrainingError, match="data_source"):
        selfcond_script.assert_resume_selfcond_matches(
            _resume_meta_for_source(None),
            _config(),
            data_source=SelfCondDataSource.SFT,
        )


# kills: leaving the cadence in the rollout-policy comparison after it moved
# to the metadata top level -- every turn resume would then compare a recorded
# `selfcond` entry that build_selfcond_checkpoint_meta no longer writes and
# fail against the freshly measured value.
def test_resume_policy_identity_no_longer_carries_the_turn_cadence() -> None:
    resume_meta = _resume_meta_for_source(SelfCondDataSource.SFT.value)
    recorded = resume_meta["selfcond"]
    assert isinstance(recorded, dict)
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in recorded

    selfcond_script.assert_resume_selfcond_matches(
        resume_meta,
        _config(),
        data_source=SelfCondDataSource.SFT,
    )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("dataset", "capabilities"),
        ("train_feature_dir", "/data/features/other_safety_sft"),
        ("tokenizer_checksum", "different-tokenizer-sha256"),
    ],
)
def test_resume_rejects_each_selfcond_data_identity_mismatch(
    field: str,
    changed_value: str,
) -> None:
    recorded = _selfcond_data_identity()
    current = _selfcond_data_identity()
    current[field] = changed_value

    with pytest.raises(SelfCondTrainingError) as exc_info:
        selfcond_script.assert_resume_selfcond_data_matches(
            {"selfcond_data": recorded},
            current,
        )

    message = str(exc_info.value)
    assert "selfcond_data" in message
    assert field in message
    assert repr(recorded[field]) in message
    assert repr(changed_value) in message


@pytest.mark.parametrize(
    "recorded",
    [
        None,
        "not-a-mapping",
        {"dataset": "safety"},
    ],
)
def test_resume_requires_complete_selfcond_data_identity(recorded: object) -> None:
    resume_meta = {} if recorded is None else {"selfcond_data": recorded}
    with pytest.raises(SelfCondTrainingError, match="selfcond_data"):
        selfcond_script.assert_resume_selfcond_data_matches(
            resume_meta,
            _selfcond_data_identity(),
        )


def test_resume_identity_distinguishes_missing_key_from_recorded_none() -> None:
    current = _selfcond_data_identity()
    current["val_feature_dir"] = None
    recorded = deepcopy(current)
    del recorded["val_feature_dir"]

    with pytest.raises(SelfCondTrainingError, match="val_feature_dir"):
        selfcond_script.assert_resume_selfcond_data_matches(
            {"selfcond_data": recorded},
            current,
        )


def test_resume_accepts_exact_selfcond_data_identity() -> None:
    identity = _selfcond_data_identity()
    selfcond_script.assert_resume_selfcond_data_matches(
        {"selfcond_data": deepcopy(identity)},
        identity,
    )


def test_resume_rejects_nested_sft_artifact_coverage_drift() -> None:
    recorded = _sft_artifact_coverage()
    current = deepcopy(recorded)
    feature_fingerprint: Any = current["feature_mixture_fingerprint"]
    assert isinstance(feature_fingerprint, dict)
    feature_fingerprint["index_map_sha256"] = "c" * 64

    with pytest.raises(SelfCondTrainingError) as exc_info:
        selfcond_script.assert_resume_sft_artifact_coverage(
            {"loop_state": {"sft_artifact_coverage": recorded}},
            current,
        )

    message = str(exc_info.value)
    assert "sft_artifact_coverage" in message
    assert "index_map_sha256" in message
    assert "b" * 64 in message
    assert "c" * 64 in message


@pytest.mark.parametrize(
    ("resume_meta", "current_coverage"),
    [
        ({}, _sft_artifact_coverage()),
        ({"loop_state": {}}, _sft_artifact_coverage()),
        (
            {"loop_state": {"sft_artifact_coverage": "not-a-mapping"}},
            _sft_artifact_coverage(),
        ),
        (
            {"loop_state": {"sft_artifact_coverage": _sft_artifact_coverage()}},
            None,
        ),
    ],
)
def test_sft_artifact_coverage_has_no_legacy_or_none_tolerance(
    resume_meta: dict[str, object],
    current_coverage: dict[str, object] | None,
) -> None:
    with pytest.raises(SelfCondTrainingError, match="sft_artifact_coverage"):
        selfcond_script.assert_resume_sft_artifact_coverage(
            resume_meta,
            current_coverage,
        )


def test_resume_accepts_exact_sft_artifact_coverage() -> None:
    coverage = _sft_artifact_coverage()
    selfcond_script.assert_resume_sft_artifact_coverage(
        {"loop_state": {"sft_artifact_coverage": deepcopy(coverage)}},
        coverage,
    )


@pytest.mark.parametrize("semantics", [None, "prefetched_v0", "committed_v2"])
def test_sft_mode_resume_rejects_missing_or_wrong_cursor_semantics(
    semantics: str | None,
) -> None:
    loop_state: dict[str, object] = {"sft_consumed": 40}
    if semantics is not None:
        loop_state["sft_cursor_semantics"] = semantics

    with pytest.raises(SelfCondTrainingError) as exc_info:
        selfcond_script.resolve_sft_mode_resume_state({"loop_state": loop_state})

    message = str(exc_info.value)
    assert "sft_cursor_semantics" in message
    assert selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS in message
    assert repr(semantics) in message


@pytest.mark.parametrize("consumed", [None, -1, 2.5, True, "40"])
def test_sft_mode_resume_requires_nonnegative_integer_committed_cursor(
    consumed: object,
) -> None:
    loop_state = {
        "sft_cursor_semantics": selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS,
        "sft_consumed": consumed,
    }
    with pytest.raises(SelfCondTrainingError, match="sft_consumed"):
        selfcond_script.resolve_sft_mode_resume_state({"loop_state": loop_state})


def test_sft_mode_resume_returns_committed_cursor_and_buffer_envelope() -> None:
    envelope = {"consumed": 40, "next_cursor": 41, "buffered_cursor_positions": []}
    assert selfcond_script.resolve_sft_mode_resume_state(
        {
            "loop_state": {
                "sft_cursor_semantics": (selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS),
                "sft_consumed": 40,
                "sft_dataloader_state": envelope,
            }
        }
    ) == (40, envelope)


def test_sft_mode_resume_rejects_nonzero_cursor_without_buffer_envelope() -> None:
    """buffered_v2 is cursor + buffer; the cursor alone cannot resume exactly."""
    with pytest.raises(SelfCondTrainingError, match="sft_dataloader_state"):
        selfcond_script.resolve_sft_mode_resume_state(
            {
                "loop_state": {
                    "sft_cursor_semantics": (selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS),
                    "sft_consumed": 40,
                }
            }
        )


def test_sft_mode_resume_allows_zero_cursor_without_buffer_envelope() -> None:
    """A fresh cursor has no buffered rows, so the envelope is legitimately absent."""
    assert selfcond_script.resolve_sft_mode_resume_state(
        {
            "loop_state": {
                "sft_cursor_semantics": (selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS),
                "sft_consumed": 0,
            }
        }
    ) == (0, None)


def _build_checkpoint_meta(**overrides: Any) -> dict[str, Any]:
    """Mint a self-conditioning checkpoint meta; overrides name what varies."""
    kwargs: dict[str, Any] = {
        "model": tiny_head_model(_config().pool_mode, n_sae=8, num_buckets=2, seed=801),
        "base_meta": _sft_shaped_meta(),
        "identity": {},
        "cfg": _config(),
        "data_source": SelfCondDataSource.SFT,
        "selfcond_data": _selfcond_data_identity(),
        "dataloader_state_dict": None,
        "loop_state": {"step": 1},
        "base_checkpoint_dir": "/checkpoints/sft-leaf",
    }
    kwargs.update(overrides)
    return selfcond_script.build_selfcond_checkpoint_meta(**kwargs)


def test_sft_checkpoint_meta_round_trips_all_identity_and_cursor_blocks(
    tmp_path: Path,
) -> None:
    model = tiny_head_model(
        _config().pool_mode,
        n_sae=8,
        num_buckets=2,
        seed=801,
    )
    selfcond_data = _selfcond_data_identity()
    coverage = _sft_artifact_coverage()
    loop_state = {
        "step": 7,
        "smooth_train_loss": 1.25,
        "min_val_bpb": 2.5,
        "min_val_bpb_selfcond": 2.25,
        "total_training_time": 10.0,
        "sft_cursor_semantics": selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS,
        "sft_consumed": 40,
        "sft_artifact_coverage": coverage,
    }
    meta = _build_checkpoint_meta(
        model=model,
        selfcond_data=selfcond_data,
        loop_state=loop_state,
        base_checkpoint_dir="/checkpoints/safety-sft-leaf",
    )
    checkpoint_dir = tmp_path / "selfcond_head_only_dense_checkpoints"

    save_checkpoint(
        str(checkpoint_dir),
        7,
        model.state_dict(),
        None,
        meta,
        rank=0,
    )
    loaded_model, loaded_optimizer, loaded_meta = load_checkpoint(
        str(checkpoint_dir),
        7,
        "cpu",
        load_optimizer=False,
        rank=0,
    )

    assert loaded_model.keys() == model.state_dict().keys()
    assert loaded_optimizer is None
    assert loaded_meta["selfcond"]["data_source"] == "sft"
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in loaded_meta["selfcond"]
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in loaded_meta
    assert loaded_meta["selfcond_data"] == selfcond_data
    assert loaded_meta["loop_state"] == loop_state
    assert loaded_meta["base_checkpoint_dir"] == "/checkpoints/safety-sft-leaf"
    assert "dataloader_state_dict" not in loaded_meta
    assert SFT_SEGMENT_MODE_META_KEY not in loaded_meta
    assert ONLINE_SAE_LINEAGE_META_KEY not in loaded_meta


# kills: dropping any of the base identity blocks post-training must
# re-publish. `resolve_feature_clusters_from_checkpoint_meta` rejects a
# checkpoint carrying only one half of the cluster pair, and
# `run_methods._target_checkpoint_identity` needs the top-level `step` to
# locate model_{step:06d}.pt -- the two failures GPU jobs 22039772 and 22039415
# hit. Also kills copying the base's step (500) instead of loop_state's (22),
# and re-publishing the blocks by reference rather than as copies.
def test_compacted_clustered_selfcond_meta_stays_readable_downstream(
    tmp_path: Path,
) -> None:
    base_meta, mask, _consolidation, feature_vocab = _compact_checkpoint_meta()
    model = tiny_head_model(_config().pool_mode, n_sae=8, num_buckets=2, seed=801)

    meta = _build_checkpoint_meta(model=model, base_meta=base_meta, loop_state={"step": 22})

    assert meta["step"] == 22
    assert base_meta["step"] == 500
    inherited = [key for key in selfcond_script.INHERITED_BASE_META_KEYS if key in base_meta]
    assert inherited == [
        "dataset",
        "sft_loss_mask",
        "missing_percentile_policy",
        "pretrain_conditioning",
        "feature_mask_canonical",
        "feature_clusters_descriptor",
        "feature_clusters_canonical",
        "feature_vocab",
        "feature_prediction_width",
    ]
    # This legacy fixture predates cooldown. The new optional record has its
    # own round-trip test; every other inherited key is exercised here.
    assert [
        key for key in selfcond_script.INHERITED_BASE_META_KEYS
        if key != "conditioning_cooldown"
    ] == inherited
    for key in inherited:
        assert meta[key] == base_meta[key], key
        # Every nested block is a copy, so a later in-place edit of either
        # checkpoint's metadata cannot silently rewrite the other's.
        if isinstance(base_meta[key], dict):
            assert meta[key] is not base_meta[key], key
    assert meta["feature_mask_canonical"] == mask.canonical_dict()
    assert meta["feature_vocab"] == feature_vocab.metadata_dict()

    resolved = resolve_feature_clusters_from_checkpoint_meta(
        meta,
        surface="self-conditioning checkpoint",
        source_path=str(tmp_path),
    )
    assert resolved is not None
    assert resolved.definition.clusters == ((0, 1, 2), (4, 5, 6), (8, 9))

    checkpoint_dir = tmp_path / "selfcond_head_only_dense_checkpoints"
    save_checkpoint(str(checkpoint_dir), 22, model.state_dict(), None, meta, rank=0)

    identity = run_methods._target_checkpoint_identity(checkpoint_dir / "meta_000022.json")

    weights = checkpoint_dir / "model_000022.pt"
    assert identity["name"] == "model_000022.pt"
    assert identity["byte_length"] == weights.stat().st_size


def test_turn_checkpoint_meta_republishes_mode_lineage_and_rustbpe_cadence() -> None:
    mode = SegmentMode.turn_delimited_fixed_tokens
    base_meta = _sft_shaped_meta()
    base_meta[SFT_SEGMENT_MODE_META_KEY] = mode.value
    lineage = _online_lineage_identity(mode)
    base_meta[ONLINE_SAE_LINEAGE_META_KEY] = lineage
    selfcond_data = _selfcond_data_identity()
    selfcond_data[SFT_SEGMENT_MODE_META_KEY] = mode.value

    meta = _build_checkpoint_meta(
        base_meta=base_meta,
        selfcond_data=selfcond_data,
        turn_self_cond_segment_size_rustbpe=37,
    )

    assert meta[SFT_SEGMENT_MODE_META_KEY] == mode.value
    assert meta[ONLINE_SAE_LINEAGE_META_KEY] == lineage
    assert meta[ONLINE_SAE_LINEAGE_META_KEY] is not lineage
    # Nested blocks are copies too, like every inherited identity block.
    assert meta[ONLINE_SAE_LINEAGE_META_KEY]["feature_selection"] is not lineage["feature_selection"]
    # kills: writing the measured cadence back into the rollout-policy
    # `selfcond` block, which the reader no longer consults.
    assert meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] == 37
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in meta["selfcond"]
    assert (
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/turn-selfcond-leaf",
        )
        == 37
    )


# kills: narrowing build_selfcond_checkpoint_meta's lineage republish from
# `if ONLINE_SAE_LINEAGE_META_KEY in base_meta` to a turn-only condition, which
# silently drops online extraction provenance from every fixed-base checkpoint;
# and republishing the block by reference rather than as a copy.
def test_fixed_base_online_sae_lineage_is_republished_as_a_copy() -> None:
    base_meta = _sft_shaped_meta()
    base_meta[SFT_SEGMENT_MODE_META_KEY] = SegmentMode.fixed_tokens.value
    lineage = _online_lineage_identity(SegmentMode.fixed_tokens)
    base_meta[ONLINE_SAE_LINEAGE_META_KEY] = lineage

    meta = _build_checkpoint_meta(base_meta=base_meta)

    assert meta[ONLINE_SAE_LINEAGE_META_KEY] == lineage
    assert meta[ONLINE_SAE_LINEAGE_META_KEY] is not lineage
    assert meta[SFT_SEGMENT_MODE_META_KEY] == SegmentMode.fixed_tokens.value
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in meta["selfcond"]


# kills: dropping build_selfcond_checkpoint_meta's trailing re-validation of
# the CONSTRUCTED metadata. The base resolves fixed, so no cadence is written,
# but the caller-supplied selfcond_data claims turn segmentation -- an identity
# the checkpoint must never be written with.
def test_checkpoint_meta_revalidates_its_own_constructed_identity() -> None:
    base_meta = _sft_shaped_meta()
    assert SFT_SEGMENT_MODE_META_KEY not in base_meta
    selfcond_data = _selfcond_data_identity()
    selfcond_data[SFT_SEGMENT_MODE_META_KEY] = SegmentMode.turn_delimited_fixed_tokens.value

    with pytest.raises(
        SelfCondTrainingError,
        match=r"turn-delimited checkpoints require top-level post-training",
    ):
        _build_checkpoint_meta(base_meta=base_meta, selfcond_data=selfcond_data)


def test_turn_checkpoint_meta_refuses_missing_rustbpe_cadence() -> None:
    mode = SegmentMode.turn_delimited_fixed_tokens
    base_meta = _sft_shaped_meta()
    base_meta[SFT_SEGMENT_MODE_META_KEY] = mode.value
    selfcond_data = _selfcond_data_identity()
    selfcond_data[SFT_SEGMENT_MODE_META_KEY] = mode.value

    with pytest.raises(
        SelfCondTrainingError,
        match=r"require top-level post-training metadata",
    ):
        _build_checkpoint_meta(base_meta=base_meta, selfcond_data=selfcond_data)


def test_checkpoint_meta_wraps_nonmapping_online_lineage_as_training_error() -> None:
    base_meta = _sft_shaped_meta()
    base_meta[SFT_SEGMENT_MODE_META_KEY] = SegmentMode.fixed_tokens.value
    base_meta[ONLINE_SAE_LINEAGE_META_KEY] = None

    with pytest.raises(
        SelfCondTrainingError,
        match=r"invalid SFT segmentation metadata.*must be a mapping",
    ):
        _build_checkpoint_meta(base_meta=base_meta)


def test_turn_resume_requires_exact_online_sae_lineage() -> None:
    mode = SegmentMode.turn_delimited_fixed_tokens
    base_meta: dict[str, Any] = {
        SFT_SEGMENT_MODE_META_KEY: mode.value,
        ONLINE_SAE_LINEAGE_META_KEY: _online_lineage_identity(mode),
    }
    resume_meta = deepcopy(base_meta)
    base_lineage = selfcond_script._resolve_selfcond_online_sae_lineage(
        base_meta,
        segment_mode=mode,
        checkpoint_dir="/checkpoints/base",
        purpose="base",
    )
    resume_lineage = selfcond_script._resolve_selfcond_online_sae_lineage(
        resume_meta,
        segment_mode=mode,
        checkpoint_dir="/checkpoints/resume",
        purpose="resume",
    )

    selfcond_script.assert_resume_online_sae_lineage_matches(
        base_lineage,
        resume_lineage,
        segment_mode=mode,
    )

    resume_meta[ONLINE_SAE_LINEAGE_META_KEY]["statistics_sha256"] = "b" * 64
    changed_resume_lineage = selfcond_script._resolve_selfcond_online_sae_lineage(
        resume_meta,
        segment_mode=mode,
        checkpoint_dir="/checkpoints/resume",
        purpose="resume",
    )
    with pytest.raises(SelfCondTrainingError, match=r"lineage disagrees"):
        selfcond_script.assert_resume_online_sae_lineage_matches(
            base_lineage,
            changed_resume_lineage,
            segment_mode=mode,
        )


# kills: dropping the segment-mode conjunct from the resume lineage guard -- a
# fixed-mode resume checkpoint minted before lineage existed carries none, and
# only turn mode may bind the base's lineage to it.
def test_fixed_resume_does_not_add_a_new_lineage_identity_requirement() -> None:
    mode = SegmentMode.fixed_tokens
    base_lineage = selfcond_script._resolve_selfcond_online_sae_lineage(
        {
            SFT_SEGMENT_MODE_META_KEY: mode.value,
            ONLINE_SAE_LINEAGE_META_KEY: _online_lineage_identity(mode),
        },
        segment_mode=mode,
        checkpoint_dir="/checkpoints/base",
        purpose="base",
    )
    assert base_lineage is not None

    selfcond_script.assert_resume_online_sae_lineage_matches(
        base_lineage,
        None,
        segment_mode=mode,
    )


def test_recorded_val_locations_are_recovered_when_config_is_null(
    tmp_path: Path,
) -> None:
    """The l22 family records NON-default val locations in user_config; the
    recovery contract must honor them (deriving by convention would resolve a
    different parquet family and fail the mixture-fingerprint gate)."""
    train_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    recorded_val_features = _canonical_feature_dir(
        tmp_path, DatasetPhase.SAFETY_SFT_VAL, root_name="recorded_val"
    )
    recorded_val_parquet = tmp_path / "sft_data" / "safety_l99_val"
    _populate_val_artifacts(recorded_val_features, recorded_val_parquet)

    meta = _meta_with_feature_dir(train_feature_dir)
    meta["user_config"]["val_sft_parquet_dir"] = str(recorded_val_parquet)
    meta["user_config"]["val_feature_dir"] = str(recorded_val_features)

    sources = _resolve_sources(
        selfcond_block={},
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=1,
    )

    assert sources.val_parquet_dir == str(recorded_val_parquet)
    assert sources.val_feature_dir == str(recorded_val_features)


def test_explicit_val_overrides_beat_recorded_val_locations(
    tmp_path: Path,
) -> None:
    train_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    recorded_val_features = _canonical_feature_dir(
        tmp_path, DatasetPhase.SAFETY_SFT_VAL, root_name="recorded_val"
    )
    explicit_val_features = _canonical_feature_dir(
        tmp_path, DatasetPhase.SAFETY_SFT_VAL, root_name="explicit_val"
    )
    explicit_val_parquet = tmp_path / "explicit_val_parquet"
    _populate_val_artifacts(explicit_val_features, explicit_val_parquet)

    meta = _meta_with_feature_dir(train_feature_dir)
    meta["user_config"]["val_sft_parquet_dir"] = str(tmp_path / "recorded_parquet")
    meta["user_config"]["val_feature_dir"] = str(recorded_val_features)

    sources = _resolve_sources(
        selfcond_block={
            "val_sft_parquet_dir": str(explicit_val_parquet),
            "val_sft_feature_dir": str(explicit_val_features),
        },
        base_meta=meta,
        base_dir=tmp_path,
        eval_every=1,
    )

    assert sources.val_parquet_dir == str(explicit_val_parquet)
    assert sources.val_feature_dir == str(explicit_val_features)


def test_recovered_noncanonical_val_feature_dir_stays_strict(
    tmp_path: Path,
) -> None:
    """A RECOVERED val dir never gets the operator-explicit escape hatch."""
    train_feature_dir = _canonical_feature_dir(tmp_path, DatasetPhase.SAFETY_SFT)
    recorded_val_parquet = tmp_path / "sft_data" / "safety_l99_val"
    noncanonical_val = tmp_path / "legacy" / "val_features"
    _populate_val_artifacts(noncanonical_val, recorded_val_parquet)

    meta = _meta_with_feature_dir(train_feature_dir)
    meta["user_config"]["val_sft_parquet_dir"] = str(recorded_val_parquet)
    meta["user_config"]["val_feature_dir"] = str(noncanonical_val)

    with pytest.raises(SelfCondTrainingError, match="non-canonical"):
        _resolve_sources(
            selfcond_block={},
            base_meta=meta,
            base_dir=tmp_path,
            eval_every=1,
        )
