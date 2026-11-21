"""Functional and integration tests for clustered selection-shard consumption.

These tests serialize real PackedRowShard/SelectionShard artifacts and drive the
production precomputed loader. They deliberately validate every identity and ID
boundary before a batch can be yielded; mocks would miss the safetensors dtype,
CSR, manifest, reconstruction, padding, and resume interactions covered here.
"""

from __future__ import annotations

import copy
import json
import shutil
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest
import torch as t
import zstandard as zstd

import nanochat.flash_attention as _fa_mod
from errors import FeatureClustersError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.train.conditional_encdec_train import (
    _assert_compact_fp_checkpoint_compatible,
)
from steering.conditional.delta import (
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    DeltaMode,
    FeatureDeltaConfig,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder import dataloader as dataloader_module
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.dataloader import (
    ClosableBatchPrefetcher,
    precomputed_data_loader,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.fp_target_source import certify_fp_target_source
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    assert_resume_feature_clusters_match,
    build_clustered_selection_shard_integrity,
    feature_clusters_checkpoint_metadata,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    assert_resume_feature_mask_matches,
    canonical_dict,
)
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    assert_resume_feature_vocab_matches,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.splice_counterfactual import (
    canonical_feature_delta_dict,
    derived_feature_fingerprint,
    packed_row_fingerprint,
    splice_sidecar_fingerprint,
    tensor_content_fingerprint,
    validate_packed_provenance,
)
from steering.conditional.thresholding import ThresholdType
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    load_selection_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.refined_types import PositiveInt
from steering.remote_body_prefetch import RemoteBodyBinding
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    REMOTE_PACKED_INVENTORY_FILENAME,
    RemoteMaterializationCancelled,
    RemoteObjectInfo,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
    _read_cache_body_bytes,
    build_remote_loader_certificate,
    publish_remote_packed_inventory,
)

N_SAE: PositiveInt = PositiveInt.parse(32)
CONTEXT_LEN = 8


@pytest.fixture
def _force_sdpa_for_identity_parity():
    """Mirror the production attention backend/cache prerequisites on CPU."""
    old_attention = _fa_mod._override_impl
    old_cache_limit = t._dynamo.config.cache_size_limit
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    # build_shared_state raises this to 64 because the compiled Muon step has
    # one graph per parameter shape. The default limit of 8 is insufficient
    # even for this small encoder-decoder's complete optimizer grouping.
    t._dynamo.config.cache_size_limit = 64
    yield
    t._dynamo.config.cache_size_limit = old_cache_limit
    _fa_mod._override_impl = old_attention


def _pipeline(mask: FeatureMask | None) -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.0),
            precomputed_thresholds=t.zeros(N_SAE, dtype=t.float32),
        ),
        boundaries=t.linspace(0.25, 0.75, 3, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE, -1)
        .contiguous(),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )


def _cluster_state(
    tmp_path: Path,
    *,
    mask: FeatureMask | None = None,
    clusters: list[list[int]] | None = None,
    name: str = "clusters.json",
) -> FeatureClusterConsolidation:
    source = tmp_path / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clusters": clusters or [[3, 5], [7, 8]],
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(source)}},
        N_SAE,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _base_shard(*, token_offset: int = 0) -> PackedRowShard:
    return PackedRowShard(
        tokens=(
            t.arange(2 * (CONTEXT_LEN + 1), dtype=t.int64)
            .reshape(2, CONTEXT_LEN + 1)
            .add(token_offset)
            .remainder(31)
            .add(1)
            .to(t.uint16)
        ),
        row_seg_offsets=t.tensor([0, 1, 2], dtype=t.int32),
        seg_token_start=t.tensor([0, 0], dtype=t.uint16),
        seg_token_end=t.tensor([CONTEXT_LEN, CONTEXT_LEN], dtype=t.uint16),
        seg_doc_id=t.tensor([0, 0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, 3, 5], dtype=t.int32),
        feature_ids=t.tensor([1, 3, 5, 7, 8], dtype=t.uint16),
        feature_pcts=t.tensor([20, 100, 240, 180, 220], dtype=t.uint8),
    )


def _selection_shard(
    *,
    positive_ids: list[int] | None = None,
    negative_ids: list[int] | None = None,
) -> SelectionShard:
    positives = positive_ids if positive_ids is not None else [3, 10, 7]
    negatives = negative_ids if negative_ids is not None else [12, 13]
    assert len(positives) == 3
    assert len(negatives) == 2
    return SelectionShard(
        sel_offsets=t.tensor([0, 2, 3], dtype=t.int32),
        sel_fids=t.tensor(positives, dtype=t.uint16),
        sel_buckets=t.tensor([3, 2, 3], dtype=t.uint8),
        sel_pcts=t.tensor([240, 150, 220], dtype=t.uint8),
        neg_offsets=t.tensor([0, 1, 2], dtype=t.int32),
        neg_fids=t.tensor(negatives, dtype=t.uint16),
    )


def _compact_fp_training_shard() -> PackedRowShard:
    """Two documents with two segments each and non-trivial next-segment FP targets."""
    return PackedRowShard(
        tokens=(
            t.arange(2 * (CONTEXT_LEN + 1), dtype=t.int64)
            .reshape(2, CONTEXT_LEN + 1)
            .remainder(31)
            .add(1)
            .to(t.uint16)
        ),
        row_seg_offsets=t.tensor([0, 2, 4], dtype=t.int32),
        seg_token_start=t.tensor([0, 4, 0, 4], dtype=t.uint16),
        seg_token_end=t.tensor([4, 8, 4, 8], dtype=t.uint16),
        seg_doc_id=t.tensor([0, 0, 0, 0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, 1, 3, 4, 6], dtype=t.int32),
        feature_ids=t.tensor([10, 3, 5, 14, 7, 8], dtype=t.uint16),
        feature_pcts=t.tensor([120, 100, 240, 110, 180, 220], dtype=t.uint8),
    )


def _compact_fp_training_selection() -> SelectionShard:
    """Representative-only conditioning that omits both target siblings."""
    return SelectionShard(
        sel_offsets=t.tensor([0, 1, 2, 3, 4], dtype=t.int32),
        sel_fids=t.tensor([10, 3, 14, 7], dtype=t.uint16),
        sel_buckets=t.tensor([2, 3, 2, 3], dtype=t.uint8),
        # Deliberately differ from the raw 240/220 sibling maxima. Targets must
        # come from the packed base CSR, not this conditioning overlay.
        sel_pcts=t.tensor([120, 10, 110, 20], dtype=t.uint8),
        neg_offsets=t.tensor([0, 1, 2, 3, 4], dtype=t.int32),
        neg_fids=t.tensor([12, 13, 12, 13], dtype=t.uint16),
    )


def _write_artifacts(
    tmp_path: Path,
    *,
    clusters: FeatureClusterConsolidation | None,
    mask: FeatureMask | None = None,
    base: PackedRowShard | None = None,
    selection: SelectionShard | None = None,
    status: str = "complete",
    stem: str = "shard_00000",
    max_n_seg: int = 1,
    max_seg_size: int = CONTEXT_LEN,
) -> tuple[Path, Path]:
    packed_dir = tmp_path / "packed"
    selection_dir = tmp_path / "selection"
    packed_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)
    base_path = save_packed_row_shard(
        base if base is not None else _base_shard(),
        packed_dir / stem,
    )
    selection_path = save_selection_shard(
        selection or _selection_shard(),
        selection_dir / stem,
    )
    packed_metadata = {
        "max_n_seg": max_n_seg,
        "max_seg_size": max_seg_size,
    }
    write_manifest(
        packed_dir,
        "packed-test",
        OverlayKind.PACKED_ROW,
        metadata=packed_metadata,
        base_shard_paths=[base_path],
    )
    selection_metadata = {
        "packed_rows": {
            "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
        },
    }
    if clusters is not None:
        if mask is not None and mask != clusters.feature_mask:
            raise AssertionError("explicit fixture mask disagrees with cluster state")
        mask = clusters.feature_mask
        cluster_metadata = clusters.metadata_dict()
        cluster_metadata["status"] = status
        selection_metadata["feature_clusters"] = cluster_metadata
        if status == "complete":
            selection_metadata[SELECTION_SHARD_INTEGRITY_KEY] = (
                build_clustered_selection_shard_integrity([selection_path])
            )
    mask_metadata = canonical_dict(mask)
    if mask_metadata is not None:
        selection_metadata["feature_mask"] = mask_metadata
    write_manifest(
        selection_dir,
        "selection-test",
        OverlayKind.PACKED_ROW,
        metadata=selection_metadata,
        base_shard_paths=[base_path],
        cache_dir=packed_dir,
    )
    return packed_dir, selection_dir


def _identity_parity_model(mask: FeatureMask, *, seed: int) -> EncoderDecoderGPT:
    """Build through the production meta -> to_empty -> init path."""
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    assert vocab is not None
    with t.device("meta"):
        model = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=CONTEXT_LEN,
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="SL",
            ),
            EncoderConfig(
                n_embd=32,
                depth=2,
                n_head=None,
                n_kv_head=None,
                head_dim=16,
            ),
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            # The fixture selection uses four positive magnitude buckets and
            # bucket 4 as the negative sentinel, matching training's
            # effective_buckets = cfg.num_buckets - 1 contract.
            num_buckets=5,
            feature_vocab=vocab,
        )
    model.to_empty(device="cpu")
    t.manual_seed(seed)
    model.init_weights()
    model.feature_mask = mask
    return model


def _compact_fp_model(
    vocab: FeatureVocab,
    mask: FeatureMask,
    *,
    loss_fn: FeaturePredLossFn,
    seed: int,
) -> EncoderDecoderGPT:
    """Build the compact-FP fixture through production's meta/init sequence."""
    with t.device("meta"):
        model = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=CONTEXT_LEN,
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="SL",
            ),
            EncoderConfig(
                n_embd=32,
                depth=2,
                n_head=None,
                n_kv_head=None,
                head_dim=16,
            ),
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=5,
            feature_pred_config=FeaturePredictionConfig(
                weight=0.1,
                layer=-1,
                loss_mode=FeaturePredLossMode.MAX_POOL,
                loss_fn=loss_fn,
            ),
            feature_vocab=vocab,
        )
    model.to_empty(device="cpu")
    t.manual_seed(seed)
    model.init_weights()
    model.feature_mask = mask
    return model


def _compact_fp_train_step(
    model: EncoderDecoderGPT,
    optimizer: t.optim.Optimizer,
    batch: tuple[Any, ...],
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    result = model(
        batch[0],
        batch[1],
        batch[2],
        batch[3],
        batch[4],
        batch[6],
        seg_doc_id=batch[7],
        fp_targets=batch[8],
    )
    assert isinstance(result, ForwardResult)
    assert result.loss.item() > 0.0
    assert result.lm_loss.item() > 0.0
    assert result.feature_pred_loss.item() > 0.0
    assert t.isfinite(result.loss)
    assert t.isfinite(result.lm_loss)
    assert t.isfinite(result.feature_pred_loss)
    result.loss.backward()
    assert model.feature_pred_head is not None
    assert model.feature_pred_head.weight.grad is not None
    assert model.feature_pred_head.weight.grad.abs().sum().item() > 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert t.isfinite(parameter.grad).all(), name
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return (
        result.loss.detach().clone(),
        result.lm_loss.detach().clone(),
        result.feature_pred_loss.detach().clone(),
    )


def _assert_nested_exact(actual: Any, expected: Any, *, path: str = "root") -> None:
    """Exact recursive comparison for production optimizer state dictionaries."""
    assert type(actual) is type(expected), (
        f"{path}: {type(actual).__name__} != {type(expected).__name__}"
    )
    if isinstance(actual, t.Tensor):
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert actual.stride() == expected.stride(), path
        assert actual.storage_offset() == expected.storage_offset(), path
        actual_bits = actual.detach().cpu().contiguous().reshape(-1).view(t.uint8)
        expected_bits = expected.detach().cpu().contiguous().reshape(-1).view(t.uint8)
        assert t.equal(actual_bits, expected_bits), path
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_nested_exact(
                actual[key],
                expected[key],
                path=f"{path}[{key!r}]",
            )
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_nested_exact(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
    else:
        assert actual == expected, path


def _loader(
    packed_dir: Path,
    selection_dir: Path | None,
    *,
    clusters: FeatureClusterConsolidation | None,
    pipeline: FeaturePipelineConfig,
    resume_state_dict: dict | None = None,
    n_sae_features: int | None = N_SAE,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
):
    return precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=2,
        feature_pipeline=pipeline,
        device="cpu",
        selection_dir=selection_dir,
        effective_buckets=4,
        n_sae_features=(n_sae_features if fp_target_layout is None else None),
        fp_target_layout=fp_target_layout,
        resume_state_dict=resume_state_dict,
        feature_clusters=clusters,
    )


def test_clustered_compact_fp_targets_use_raw_base_features_and_member_max(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _cluster_state(tmp_path, mask=mask)
    selection = SelectionShard(
        sel_offsets=t.tensor([0, 2, 3], dtype=t.int32),
        sel_fids=t.tensor([3, 10, 7], dtype=t.uint16),
        sel_buckets=t.tensor([3, 2, 3], dtype=t.uint8),
        # Deliberately disagree with raw base percentiles. FP must ignore this
        # overlay, while conditioning still consumes it.
        sel_pcts=t.tensor([10, 150, 20], dtype=t.uint8),
        neg_offsets=t.tensor([0, 1, 2], dtype=t.int32),
        neg_fids=t.tensor([12, 13], dtype=t.uint16),
    )
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        selection=selection,
    )
    certify_fp_target_source(packed_dir, n_sae_features=N_SAE)
    vocab = effective_feature_vocab(clusters, N_SAE)
    assert vocab is not None
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=N_SAE,
        feature_mask=mask,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )

    batch = next(
        _loader(
            packed_dir,
            selection_dir,
            clusters=clusters,
            pipeline=_pipeline(mask),
            fp_target_layout=layout,
        )
    )
    targets = batch[-2]
    assert targets is not None
    assert targets.shape == (2, 1, vocab.size)
    row_3 = vocab.original_ids.index(3)
    row_7 = vocab.original_ids.index(7)
    row_10 = vocab.original_ids.index(10)
    assert targets[0, 0, row_3].item() == pytest.approx(240 / 255)
    assert targets[1, 0, row_7].item() == pytest.approx(220 / 255)
    assert targets[:, :, row_10].count_nonzero().item() == 0


@pytest.mark.parametrize(
    "loss_fn",
    [
        pytest.param(FeaturePredLossFn.BCE, id="max-pool-bce"),
        pytest.param(FeaturePredLossFn.MSE, id="max-pool-mse"),
    ],
)
def test_compact_fp_real_artifacts_train_save_and_resume_next_step_exactly(
    tmp_path: Path,
    _force_sdpa_for_identity_parity,
    loss_fn: FeaturePredLossFn,
) -> None:
    """Exercise the complete serialized-artifact-to-resumed-step boundary.

    This deliberately composes production components that lower-level tests
    cover separately: packed and selection safetensors, manifest binding, the
    source-integrity certificate, member-max compact targets, original-ID
    conditioning, the compact FP head, the production optimizer, checkpoint
    IO, resume guards, and the dataloader cursor.
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _cluster_state(tmp_path, mask=mask)
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        base=_compact_fp_training_shard(),
        selection=_compact_fp_training_selection(),
        max_n_seg=2,
        max_seg_size=4,
    )
    certify_fp_target_source(packed_dir, n_sae_features=N_SAE)
    vocab = effective_feature_vocab(clusters, N_SAE)
    assert vocab is not None
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=N_SAE,
        feature_mask=mask,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )

    loader = _loader(
        packed_dir,
        selection_dir,
        clusters=clusters,
        pipeline=_pipeline(mask),
        fp_target_layout=layout,
    )
    first_batch = next(loader)
    fp_targets = first_batch[8]
    assert fp_targets is not None
    assert fp_targets.shape == (2, 2, vocab.size)
    assert fp_targets.dtype is t.float32
    representative_3 = vocab.original_ids.index(3)
    representative_7 = vocab.original_ids.index(7)
    assert fp_targets[0, 1, representative_3].item() == pytest.approx(240 / 255)
    assert fp_targets[1, 1, representative_7].item() == pytest.approx(220 / 255)
    assert 5 not in vocab.original_ids
    assert 8 not in vocab.original_ids
    active_conditioning_ids = first_batch[2][first_batch[4]].tolist()
    assert 5 not in active_conditioning_ids
    assert 8 not in active_conditioning_ids

    model = _compact_fp_model(vocab, mask, loss_fn=loss_fn, seed=8128)
    assert model.feature_pred_head is not None
    assert model.feature_pred_head.out_features == vocab.size
    assert model.feature_pred_head.weight.shape == (vocab.size, 64)
    optimizer = model.setup_optimizer()
    _compact_fp_train_step(model, optimizer, first_batch)

    head_weight_state = optimizer.state[model.feature_pred_head.weight]
    head_bias_state = optimizer.state[model.feature_pred_head.bias]
    assert head_weight_state["exp_avg"].shape == (vocab.size, 64)
    assert head_weight_state["exp_avg_sq"].shape == (vocab.size, 64)
    assert head_bias_state["exp_avg"].shape == (vocab.size,)
    assert head_bias_state["exp_avg_sq"].shape == (vocab.size,)

    # The trainer checkpoints the already-prefetched next batch's cursor. A
    # resumed loader must replay this exact batch against the saved model and
    # optimizer state, rather than skipping or duplicating a trained batch.
    next_batch = next(loader)
    dataloader_state = next_batch[-1]
    source_digest = dataloader_state["fp_target_source_certificate_sha256"]
    assert dataloader_state[FEATURE_VOCAB_META_KEY] == vocab.metadata_dict()
    assert dataloader_state["feature_prediction_width"] == vocab.size

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_meta = {
        "step": 1,
        "model_config": {
            "sequence_len": CONTEXT_LEN,
            "vocab_size": 32,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 64,
            "window_pattern": "SL",
        },
        "user_config": {
            "encoder": {
                "n_embd": 32,
                "depth": 2,
                "n_head": None,
                "n_kv_head": None,
                "head_dim": 16,
            },
            "cross_attention": {
                "frequency": 1,
                "n_head": None,
                "n_kv_head": None,
            },
            "num_buckets": 5,
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": FeaturePredLossMode.MAX_POOL.value,
                "loss_fn": loss_fn.value,
            },
        },
        "dataloader_state_dict": dataloader_state,
        "feature_mask_canonical": canonical_dict(mask),
        FEATURE_VOCAB_META_KEY: vocab.metadata_dict(),
        **feature_clusters_checkpoint_metadata(clusters.resolved),
    }
    save_checkpoint(
        str(checkpoint_dir),
        1,
        model.state_dict(),
        optimizer.state_dict(),
        checkpoint_meta,
        rank=0,
    )

    reference_losses = _compact_fp_train_step(model, optimizer, next_batch)
    current_model_state = model.state_dict()
    reference_model_state = type(current_model_state)(
        (key, value.detach().clone()) for key, value in current_model_state.items()
    )
    reference_optimizer_state = optimizer.state_dict()

    model_data, optimizer_data, loaded_meta = load_checkpoint(
        str(checkpoint_dir),
        1,
        "cpu",
        load_optimizer=True,
        rank=0,
    )
    assert optimizer_data is not None
    resumed_model = _compact_fp_model(vocab, mask, loss_fn=loss_fn, seed=9999)
    assert_resume_feature_mask_matches(
        mask,
        loaded_meta,
        checkpoint_dir=str(checkpoint_dir),
    )
    assert_resume_feature_clusters_match(
        clusters.resolved,
        loaded_meta,
        checkpoint_dir=str(checkpoint_dir),
    )
    assert_checkpoint_vocab_compatible(
        vocab,
        model_data,
        checkpoint_dir=str(checkpoint_dir),
    )
    _assert_compact_fp_checkpoint_compatible(
        layout,
        source_digest,
        model_data,
        loaded_meta,
        checkpoint_dir=str(checkpoint_dir),
    )
    resumed_model.load_state_dict(model_data, strict=True, assign=True)
    resumed_model.assert_vocab_consistent()
    assert_resume_feature_vocab_matches(
        vocab,
        loaded_meta,
        checkpoint_dir=str(checkpoint_dir),
    )
    resumed_optimizer = resumed_model.setup_optimizer()
    resumed_optimizer.load_state_dict(optimizer_data)

    resumed_loader = _loader(
        packed_dir,
        selection_dir,
        clusters=clusters,
        pipeline=_pipeline(mask),
        fp_target_layout=layout,
        resume_state_dict=loaded_meta["dataloader_state_dict"],
    )
    resumed_batch = next(resumed_loader)
    _assert_nested_exact(resumed_batch, next_batch, path="resumed_batch")
    resumed_losses = _compact_fp_train_step(
        resumed_model,
        resumed_optimizer,
        resumed_batch,
    )

    _assert_nested_exact(resumed_losses, reference_losses, path="resumed_losses")
    _assert_nested_exact(
        resumed_model.state_dict(),
        reference_model_state,
        path="resumed_model",
    )
    _assert_nested_exact(
        resumed_optimizer.state_dict(),
        reference_optimizer_state,
        path="resumed_optimizer",
    )


def test_clustered_loader_reconstructs_only_original_representative_ids(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _cluster_state(tmp_path, mask=mask)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)

    batch = next(
        _loader(
            packed_dir,
            selection_dir,
            clusters=clusters,
            pipeline=_pipeline(mask),
        )
    )
    enc_fids, enc_mask, state = batch[2], batch[4], batch[-1]
    active_ids = enc_fids[enc_mask]

    assert active_ids.tolist() == [3, 10, 12, 7, 13]
    assert 5 not in active_ids.tolist()
    assert 8 not in active_ids.tolist()
    assert state["feature_clusters_descriptor"] == clusters.resolved.descriptor_dict()
    assert "cluster_delta_semantics" not in state


def test_active_inert_clusters_are_bit_exact_through_loader_training_and_eval(
    tmp_path: Path,
    _force_sdpa_for_identity_parity,
) -> None:
    """Differential supported-surface proof, including mask-derived compaction.

    The two runs have intentionally different artifact/checkpoint identities.
    Numerically, however, filtering leaves one member in each active cluster,
    so every serialized tensor, loader tensor, validation output, training
    gradient, production-optimizer state, and model parameter must be exact.
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    clusters = _cluster_state(
        tmp_path,
        mask=mask,
        clusters=[[1, 3], [7, 8]],
        name="identity_after_mask.json",
    )
    assert not bool(clusters.consolidated_away.any())
    assert t.equal(clusters.rep_of, t.arange(N_SAE, dtype=t.int64))
    assert clusters.effective_negative_mask() == mask

    selection = _selection_shard(
        positive_ids=[3, 10, 8],
        negative_ids=[12, 13],
    )
    unclustered_packed, unclustered_selection = _write_artifacts(
        tmp_path / "unclustered",
        clusters=None,
        mask=mask,
        selection=selection,
    )
    clustered_packed, clustered_selection = _write_artifacts(
        tmp_path / "clustered",
        clusters=clusters,
        selection=selection,
    )
    unclustered_shard = next(unclustered_selection.glob("*.safetensors.zst"))
    clustered_shard = next(clustered_selection.glob("*.safetensors.zst"))
    assert clustered_shard.read_bytes() == unclustered_shard.read_bytes()

    unclustered_loader = _loader(
        unclustered_packed,
        unclustered_selection,
        clusters=None,
        pipeline=_pipeline(mask),
    )
    clustered_loader = _loader(
        clustered_packed,
        clustered_selection,
        clusters=clusters,
        pipeline=_pipeline(mask),
    )

    def next_exact_batches():
        unclustered_batch = next(unclustered_loader)
        clustered_batch = next(clustered_loader)
        for index, (clustered_value, unclustered_value) in enumerate(
            zip(clustered_batch[:-1], unclustered_batch[:-1], strict=True)
        ):
            if clustered_value is None or unclustered_value is None:
                assert clustered_value is unclustered_value, f"batch[{index}]"
            else:
                assert clustered_value.dtype == unclustered_value.dtype, f"batch[{index}]"
                assert t.equal(clustered_value, unclustered_value), f"batch[{index}]"

        clustered_state = dict(clustered_batch[-1])
        assert clustered_state.pop("feature_clusters_descriptor") == (
            clusters.resolved.descriptor_dict()
        )
        assert "feature_clusters_descriptor" not in unclustered_batch[-1]
        _assert_nested_exact(
            clustered_state,
            unclustered_batch[-1],
            path="loader_state_without_intentional_cluster_identity",
        )
        return unclustered_batch, clustered_batch

    first_unclustered, first_clustered = next_exact_batches()
    active_original_ids = first_clustered[2][first_clustered[4]].to(t.int64)
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    assert vocab is not None and vocab.size == N_SAE - len(mask.ids)
    assert bool((vocab.to_compact(active_original_ids) >= 0).all())
    # The loader must preserve original SAE IDs; compaction happens exactly
    # once, at the encoder boundary.
    assert t.equal(first_clustered[2], first_unclustered[2])
    assert not t.equal(
        active_original_ids,
        vocab.to_compact(active_original_ids),
    )

    unclustered_model = _identity_parity_model(mask, seed=714)
    clustered_model = _identity_parity_model(mask, seed=714)
    _assert_nested_exact(
        clustered_model.state_dict(),
        unclustered_model.state_dict(),
        path="initialized_model",
    )
    unclustered_model.train()
    clustered_model.train()
    unclustered_optimizer = unclustered_model.setup_optimizer()
    clustered_optimizer = clustered_model.setup_optimizer()

    # Reuse the already-compared first batch, then cross two epoch boundaries.
    paired_batches = [(first_unclustered, first_clustered)]
    paired_batches.extend(next_exact_batches() for _ in range(2))
    for step, (unclustered_batch, clustered_batch) in enumerate(paired_batches):
        unclustered_optimizer.zero_grad(set_to_none=True)
        clustered_optimizer.zero_grad(set_to_none=True)
        unclustered_loss = unclustered_model(
            unclustered_batch[0],
            unclustered_batch[1],
            unclustered_batch[2],
            unclustered_batch[3],
            unclustered_batch[4],
            unclustered_batch[6],
        )
        clustered_loss = clustered_model(
            clustered_batch[0],
            clustered_batch[1],
            clustered_batch[2],
            clustered_batch[3],
            clustered_batch[4],
            clustered_batch[6],
        )
        assert t.equal(clustered_loss.detach(), unclustered_loss.detach()), (
            f"step {step}: training loss"
        )
        unclustered_loss.backward()
        clustered_loss.backward()
        for (clustered_name, clustered_param), (
            unclustered_name,
            unclustered_param,
        ) in zip(
            clustered_model.named_parameters(),
            unclustered_model.named_parameters(),
            strict=True,
        ):
            assert clustered_name == unclustered_name
            if clustered_param.grad is None or unclustered_param.grad is None:
                assert clustered_param.grad is unclustered_param.grad, (
                    f"step {step}: {clustered_name} gradient presence"
                )
            else:
                assert t.equal(clustered_param.grad, unclustered_param.grad), (
                    f"step {step}: {clustered_name} gradient"
                )
        unclustered_optimizer.step()
        clustered_optimizer.step()
        _assert_nested_exact(
            clustered_model.state_dict(),
            unclustered_model.state_dict(),
            path=f"step_{step}.model",
        )
        _assert_nested_exact(
            clustered_optimizer.state_dict(),
            unclustered_optimizer.state_dict(),
            path=f"step_{step}.optimizer",
        )

    eval_unclustered, eval_clustered = next_exact_batches()
    unclustered_model.eval()
    clustered_model.eval()
    with t.no_grad():
        unclustered_eval_loss = unclustered_model(
            eval_unclustered[0],
            eval_unclustered[1],
            eval_unclustered[2],
            eval_unclustered[3],
            eval_unclustered[4],
            eval_unclustered[6],
        )
        clustered_eval_loss = clustered_model(
            eval_clustered[0],
            eval_clustered[1],
            eval_clustered[2],
            eval_clustered[3],
            eval_clustered[4],
            eval_clustered[6],
        )
        assert t.equal(clustered_eval_loss, unclustered_eval_loss)
        unclustered_logits = unclustered_model(
            eval_unclustered[0],
            None,
            eval_unclustered[2],
            eval_unclustered[3],
            eval_unclustered[4],
            eval_unclustered[6],
        )
        clustered_logits = clustered_model(
            eval_clustered[0],
            None,
            eval_clustered[2],
            eval_clustered[3],
            eval_clustered[4],
            eval_clustered[6],
        )
        assert t.equal(clustered_logits, unclustered_logits)


def test_clustered_loader_does_not_remap_original_ids_to_dense_cluster_indices(
    tmp_path: Path,
):
    clusters = _cluster_state(tmp_path, clusters=[[20, 21], [28, 29]])
    selection = _selection_shard(
        positive_ids=[20, 10, 28],
        negative_ids=[12, 13],
    )
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        selection=selection,
    )
    batch = next(
        _loader(
            packed_dir,
            selection_dir,
            clusters=clusters,
            pipeline=_pipeline(None),
        )
    )
    active_ids = batch[2][batch[4]].tolist()
    assert 20 in active_ids
    assert 28 in active_ids
    assert 0 not in active_ids
    assert 1 not in active_ids


def test_clustered_loader_requires_selection_dir_before_fused_fallback(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, _selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    with pytest.raises(ValueError, match="requires selection_dir"):
        next(
            _loader(
                packed_dir,
                None,
                clusters=clusters,
                pipeline=_pipeline(None),
            )
        )


def test_clustered_loader_rejects_state_built_from_different_mask(tmp_path: Path):
    source_mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _cluster_state(tmp_path, mask=source_mask)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    different_mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
    with pytest.raises(ValueError, match="different FeatureMask"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(different_mask),
            )
        )


def test_clustered_loader_rejects_original_width_mismatch(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    with pytest.raises(ValueError, match="n_sae_features"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
                n_sae_features=N_SAE + 1,
            )
        )


def test_clustered_loader_rejects_missing_manifest_cluster_block(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    manifest_path = selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"].pop("feature_clusters")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FeatureClustersError, match="missing feature_clusters"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
            )
        )


def test_clustered_loader_rejects_in_progress_manifest(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        status="in_progress",
    )
    with pytest.raises(FeatureClustersError, match="status"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
            )
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda shard: shard.sel_buckets.__setitem__(0, 0),
            id="positive-bucket",
        ),
        pytest.param(
            lambda shard: shard.sel_pcts.__setitem__(0, 1),
            id="positive-percentile",
        ),
        pytest.param(
            lambda shard: shard.sel_fids.__setitem__(1, 11),
            id="positive-id",
        ),
        pytest.param(
            lambda shard: shard.neg_fids.__setitem__(0, 14),
            id="negative-id",
        ),
    ],
)
def test_clustered_loader_rejects_structurally_valid_shard_content_mutation(
    tmp_path: Path,
    mutation,
):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    selection_path = next(selection_dir.glob("*.safetensors.zst"))
    selection = load_selection_shard(selection_path)
    mutation(selection)
    save_selection_shard(selection, selection_path)

    with pytest.raises(FeatureClustersError, match="content fingerprint"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
            )
        )


def test_clustered_loader_rejects_different_canonical_cluster_content(tmp_path: Path):
    configured = _cluster_state(tmp_path, name="configured.json")
    artifact = _cluster_state(
        tmp_path,
        clusters=[[3, 6], [7, 8]],
        name="artifact.json",
    )
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=artifact)
    with pytest.raises(FeatureClustersError, match="identity mismatch"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=configured,
                pipeline=_pipeline(None),
            )
        )


@pytest.mark.parametrize(
    ("positive_ids", "negative_ids", "match"),
    [
        ([5, 10, 7], [12, 13], "consolidated-away feature ID 5"),
        ([3, 10, 7], [8, 13], "consolidated-away feature ID 8"),
        ([3, 10, 7], [12, N_SAE], "out of range"),
    ],
)
def test_clustered_loader_full_scans_every_positive_and_negative_id(
    tmp_path: Path,
    positive_ids: list[int],
    negative_ids: list[int],
    match: str,
):
    clusters = _cluster_state(tmp_path)
    corrupt = _selection_shard(
        positive_ids=positive_ids,
        negative_ids=negative_ids,
    )
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        selection=corrupt,
    )
    with pytest.raises(FeatureClustersError, match=match):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
            )
        )


def test_clustered_loader_rejects_mask_disallowed_representative(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 3))
    # Cluster [3,5] now has representative 5. Artifact ID 3 is masked, not
    # merely consolidated away, so the error must preserve that distinction.
    clusters = _cluster_state(tmp_path, mask=mask)
    corrupt = _selection_shard(positive_ids=[3, 10, 7])
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        selection=corrupt,
    )
    with pytest.raises(FeatureClustersError, match="mask-disallowed feature ID 3"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(mask),
            )
        )


def test_clustered_loader_validates_all_shards_before_first_yield(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    second_base_path = save_packed_row_shard(
        _base_shard(token_offset=7),
        packed_dir / "shard_00001",
    )
    save_selection_shard(
        _selection_shard(positive_ids=[5, 10, 7]),
        selection_dir / "shard_00001",
    )

    packed_manifest_path = packed_dir / "manifest.json"
    packed_manifest = json.loads(packed_manifest_path.read_text(encoding="utf-8"))
    # The loader only consumes metadata here; update the selection binding below
    # from the actual serialized shard set.
    packed_manifest_path.write_text(json.dumps(packed_manifest), encoding="utf-8")
    selection_manifest_path = selection_dir / "manifest.json"
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selection_manifest["metadata"]["packed_rows"]["packed_row_fingerprint"] = (
        packed_row_fingerprint(packed_dir)
    )
    selection_manifest["metadata"][SELECTION_SHARD_INTEGRITY_KEY] = (
        build_clustered_selection_shard_integrity(sorted(selection_dir.glob("*.safetensors.zst")))
    )
    selection_manifest_path.write_text(json.dumps(selection_manifest), encoding="utf-8")
    assert second_base_path.exists()

    with pytest.raises(FeatureClustersError, match="shard_00001.*feature ID 5"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
            )
        )


def test_clustered_loader_resume_accepts_exact_descriptor(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    first = next(
        _loader(
            packed_dir,
            selection_dir,
            clusters=clusters,
            pipeline=_pipeline(None),
        )
    )
    state = first[-1]
    resumed = _loader(
        packed_dir,
        selection_dir,
        clusters=clusters,
        pipeline=_pipeline(None),
        resume_state_dict=state,
    )
    assert next(resumed)[-1]["feature_clusters_descriptor"] == (clusters.resolved.descriptor_dict())


@pytest.mark.parametrize(
    ("saved_descriptor", "error_type", "match"),
    [
        (None, ValueError, "feature cluster identity mismatch"),
        (
            {"schema_version": 1, "n_sae_features": N_SAE},
            FeatureClustersError,
            "invalid feature cluster descriptor",
        ),
    ],
)
def test_clustered_loader_resume_rejects_missing_or_different_descriptor(
    tmp_path: Path,
    saved_descriptor,
    error_type,
    match: str,
):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    state = {
        "batch_idx": 0,
        "epoch": 1,
        "shard": "shard_00000.safetensors",
        "shard_batch_offset": 0,
    }
    if saved_descriptor is not None:
        state["feature_clusters_descriptor"] = saved_descriptor
    with pytest.raises(error_type, match=match):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
                resume_state_dict=state,
            )
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda descriptor: descriptor.__setitem__("schema_version", True),
        lambda descriptor: descriptor.__setitem__("unexpected", 1),
    ],
)
def test_clustered_loader_resume_strictly_parses_descriptor(
    tmp_path: Path,
    mutate,
):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    descriptor = clusters.resolved.descriptor_dict()
    mutate(descriptor)
    state = {
        "batch_idx": 0,
        "epoch": 1,
        "shard": "shard_00000.safetensors",
        "shard_batch_offset": 0,
        "feature_clusters_descriptor": descriptor,
    }
    with pytest.raises(FeatureClustersError, match="invalid feature cluster descriptor"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=clusters,
                pipeline=_pipeline(None),
                resume_state_dict=state,
            )
        )


def test_unclustered_loader_rejects_clustered_manifest_bidirectionally(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    with pytest.raises(FeatureClustersError, match="cannot consume"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                clusters=None,
                pipeline=_pipeline(None),
            )
        )


def test_unclustered_fused_loader_state_has_no_cluster_key(tmp_path: Path):
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    save_packed_row_shard(_base_shard(), packed_dir / "shard_00000")
    batch = next(
        _loader(
            packed_dir,
            None,
            clusters=None,
            pipeline=_pipeline(None),
            n_sae_features=None,
        )
    )
    assert "feature_clusters_descriptor" not in batch[-1]


def test_unclustered_loader_rejects_clustered_resume_state(tmp_path: Path):
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    save_packed_row_shard(_base_shard(), packed_dir / "shard_00000")
    state = {
        "batch_idx": 0,
        "epoch": 1,
        "feature_clusters_descriptor": {"unexpected": True},
    }
    with pytest.raises(FeatureClustersError, match="invalid feature cluster descriptor"):
        next(
            _loader(
                packed_dir,
                None,
                clusters=None,
                pipeline=_pipeline(None),
                resume_state_dict=state,
                n_sae_features=None,
            )
        )


def test_manifest_source_path_may_differ_on_consumer_host(tmp_path: Path):
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    manifest_path = selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["feature_clusters"]["source_path"] = "/mnt/producer/static-clusters.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    next(
        _loader(
            packed_dir,
            selection_dir,
            clusters=clusters,
            pipeline=_pipeline(None),
        )
    )


class _LoaderFakeBucket:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, remote_path: str) -> Path:
        return self.root / remote_path

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        if not path.is_file():
            return None
        result = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(remote_path, path.stat().st_size, str(result.hash))

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        return []

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        raise AssertionError("loader test never uploads")


class _PrepublishedBodyPrefetcher:
    """Cache-only stand-in for parity tests whose transport is tested elsewhere."""

    def publish_cursor(self, **_kwargs: Any) -> None:
        pass

    def mark_loaded(self, **_kwargs: Any) -> None:
        pass

    def check_worker_errors(self) -> None:
        pass

    def read_body(
        self,
        *,
        path: Path,
        entry: dict[str, Any],
        trust_existing_cache: bool,
        cancel_event: threading.Event | None,
        **_kwargs: Any,
    ) -> bytes:
        if cancel_event is not None and cancel_event.is_set():
            raise RemoteMaterializationCancelled("test prepublished cache read was cancelled")
        return _read_cache_body_bytes(
            path,
            entry=entry,
            verify_content=not trust_existing_cache,
        )


def test_closable_batch_prefetcher_stops_blocked_nondaemon_producer() -> None:
    started = threading.Event()
    cancelled = threading.Event()
    finalized = threading.Event()

    def source():
        value = 0
        try:
            while True:
                started.set()
                yield value
                value += 1
        finally:
            finalized.set()

    prefetched = ClosableBatchPrefetcher(
        source(),
        prefetch_batches=PositiveInt.parse(1),
        stop_callback=cancelled.set,
        failure_check=lambda: None,
    )
    assert started.wait(timeout=1)
    assert not prefetched._thread.daemon

    prefetched.close()
    prefetched.close()

    assert cancelled.is_set()
    assert finalized.is_set()
    assert not prefetched._thread.is_alive()


def _attach_prepublished_bodies(
    materializer: RemotePackedRowMaterializer | RemoteSelectionMaterializer,
    *,
    bucket: _LoaderFakeBucket,
    controller: _PrepublishedBodyPrefetcher,
    kind: str,
    schedule_id: str,
    paired: bool,
    train_schedule: bool,
) -> None:
    if isinstance(materializer, RemotePackedRowMaterializer):
        names = [entry["name"] for entry in materializer.inventory["shards"]]
    else:
        names = list(materializer.inventory)
    for name in names:
        bucket.download(
            f"{materializer.remote_dir}/{name}",
            materializer.local_dir / name,
        )
    materializer.validate_cache_namespace()
    binding = RemoteBodyBinding(
        family_id=f"prepublished-{kind}-{id(materializer)}",
        schedule_id=schedule_id,
        kind=cast(Any, kind),
        index_by_name=tuple((name, index) for index, name in enumerate(names)),
        paired=paired,
        train_schedule=train_schedule,
    )
    materializer.attach_body_prefetch(controller, binding)


def test_remote_clustered_loader_is_tensor_and_state_identical_to_all_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusters = _cluster_state(tmp_path)
    packed_dir, selection_dir = _write_artifacts(tmp_path, clusters=clusters)
    local_batch = next(
        _loader(
            packed_dir,
            selection_dir,
            clusters=clusters,
            pipeline=_pipeline(None),
        )
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_load_owner_shard",
        lambda _path: pytest.fail("remote materialization invoked the owner helper"),
    )

    manifest_path = selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_path = next(selection_dir.glob("*.safetensors.zst"))
    inventory = manifest["metadata"][SELECTION_SHARD_INTEGRITY_KEY]
    manifest["checksum"] = "a" * 64
    manifest["metadata"][REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum="a" * 64,
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=packed_row_fingerprint(packed_dir, cache_dir=packed_dir),
        max_features_per_segment=3,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bucket = _LoaderFakeBucket(tmp_path / "bucket")
    remote_dir = "features/selection"
    remote_manifest = bucket._path(remote_dir + "/manifest.json")
    remote_manifest.parent.mkdir(parents=True)
    shutil.copyfile(manifest_path, remote_manifest)
    shutil.copyfile(selection_path, bucket._path(remote_dir + "/" + selection_path.name))
    selection_path.unlink()

    remote_cfg = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2, "cache_shards": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints/conditional_encdec",
            "selection_wait_timeout_seconds": 1,
            "checkpoint_retry_timeout_seconds": 1,
            "retry_interval_seconds": 0.001,
            "resume_latest": True,
        }
    )
    assert remote_cfg is not None
    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=remote_cfg,
        local_selection_dir=selection_dir,
        feature_root=tmp_path,
        expected_selection_checksum="a" * 64,
        packed_row_fingerprint=packed_row_fingerprint(packed_dir, cache_dir=packed_dir),
        packed_stems={"shard_00000"},
        label="loader differential",
    )
    _attach_prepublished_bodies(
        materializer,
        bucket=bucket,
        controller=_PrepublishedBodyPrefetcher(),
        kind="selection",
        schedule_id="loader-differential",
        paired=False,
        train_schedule=True,
    )
    remote_batch = next(
        precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=2,
            feature_pipeline=_pipeline(None),
            device="cpu",
            selection_dir=selection_dir,
            effective_buckets=4,
            n_sae_features=N_SAE,
            feature_clusters=clusters,
            remote_materializer=materializer,
        )
    )
    _assert_nested_exact(remote_batch, local_batch)


@dataclass(frozen=True)
class _StreamingSpliceFixture:
    producer_root: Path
    packed_dir: Path
    selection_dir: Path
    splice_feature_dir: Path
    selection_checksum: str
    derived_feature_fingerprint: str
    clusters: FeatureClusterConsolidation


def _streaming_splice_pipeline(
    clusters: FeatureClusterConsolidation,
) -> FeaturePipelineConfig:
    base = _pipeline(clusters.feature_mask)
    return FeaturePipelineConfig(
        selection_config=base.selection_config,
        boundaries=base.boundaries,
        negative_bundle=base.negative_bundle,
        effective_buckets=base.effective_buckets,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            ),
            percentile_table=t.zeros(N_SAE, 99, dtype=t.float64),
        ),
        feature_mask=clusters.feature_mask,
    )


def _write_streaming_splice_fixture(
    tmp_path: Path,
) -> _StreamingSpliceFixture:
    producer_root = tmp_path / "producer"
    packed_dir = producer_root / "family" / "c1024" / "sall"
    selection_checksum = "c" * 64
    selection_dir = packed_dir / "sel" / selection_checksum
    splice_feature_dir = producer_root / "derived"
    provenance_dir(packed_dir).mkdir(parents=True)
    selection_dir.mkdir(parents=True)
    sidecar_dir = splice_feature_dir / "_splice_provenance"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "source_00000.json").write_text(
        json.dumps({"schema_version": 2, "test": True}),
        encoding="utf-8",
    )
    (splice_feature_dir / "source_00000_features.safetensors.zst").write_bytes(
        b"derived-feature-lineage-body"
    )
    clusters = _cluster_state(tmp_path / "cluster-input")
    pipeline = _streaming_splice_pipeline(clusters)

    packed_paths: list[Path] = []
    selection_paths: list[Path] = []
    for index in range(3):
        stem = f"shard_{index:05d}"
        base = _base_shard(token_offset=index * 3)
        packed_paths.append(save_packed_row_shard(base, packed_dir / stem))
        save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.tensor([0, 0], dtype=t.int32),
                PROVENANCE_DOC_IDX_KEY: t.tensor([2 * index, 2 * index + 1], dtype=t.int32),
                PROVENANCE_SEG_ORDINAL_KEY: t.tensor([0, 0], dtype=t.int32),
            },
            provenance_dir(packed_dir) / stem,
        )
        selection_paths.append(
            save_selection_shard(
                _selection_shard(),
                selection_dir / stem,
            )
        )
    write_manifest(
        packed_dir,
        "packed-streaming-test",
        OverlayKind.PACKED_ROW,
        metadata={
            "context_len": 1024,
            "base_shard_stems": ["source_00000"],
            "provenance_shards": len(packed_paths),
            "max_n_seg": 1,
            "max_seg_size": CONTEXT_LEN,
        },
        base_shard_paths=packed_paths,
    )
    certify_fp_target_source(packed_dir, n_sae_features=N_SAE)
    packed_info = validate_packed_provenance(
        packed_dir,
        cache_dir=packed_dir,
    )
    selection_integrity = build_clustered_selection_shard_integrity(selection_paths)
    cluster_metadata = clusters.metadata_dict()
    cluster_metadata["status"] = "complete"
    derived_fingerprint = derived_feature_fingerprint(
        splice_feature_dir,
        expected_stems=packed_info.base_shard_stems,
    )
    splice_metadata = {
        **splice_counterfactual_manifest_identity(clustered=True),
        "status": "complete",
        "derived_shard_stems": packed_info.base_shard_stems,
        "sidecar_fingerprint": splice_sidecar_fingerprint(splice_feature_dir),
        "derived_feature_fingerprint": derived_fingerprint,
        "boundary_fingerprint": tensor_content_fingerprint(pipeline.boundaries),
    }
    loader_certificate = build_remote_loader_certificate(
        selection_checksum=selection_checksum,
        selection_inventory_aggregate=selection_integrity["aggregate_sha256"],
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        max_features_per_segment=3,
        cluster_delta_semantics=(CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS),
    )
    (selection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": selection_checksum,
                "metadata": {
                    "feature_clusters": cluster_metadata,
                    "feature_delta": canonical_feature_delta_dict(
                        FeatureDeltaConfig(
                            mode=DeltaMode.splice_counterfactual,
                            neg_delta_ratio=1.0,
                        )
                    ),
                    "packed_rows": {"packed_row_fingerprint": (packed_info.packed_row_fingerprint)},
                    "packed_provenance": {
                        "packed_stem_fingerprint": (packed_info.packed_stem_fingerprint),
                        "packed_row_fingerprint": (packed_info.packed_row_fingerprint),
                        "provenance_shard_count": len(packed_info.provenance_shard_paths),
                        "provenance_fingerprint": (packed_info.provenance_fingerprint),
                    },
                    "splice_counterfactual": splice_metadata,
                    SELECTION_SHARD_INTEGRITY_KEY: selection_integrity,
                    REMOTE_LOADER_KEY: loader_certificate,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    publish_remote_packed_inventory(packed_dir)
    return _StreamingSpliceFixture(
        producer_root=producer_root,
        packed_dir=packed_dir,
        selection_dir=selection_dir,
        splice_feature_dir=splice_feature_dir,
        selection_checksum=selection_checksum,
        derived_feature_fingerprint=derived_fingerprint,
        clusters=clusters,
    )


def _upload_streaming_fixture(
    fixture: _StreamingSpliceFixture,
    bucket: _LoaderFakeBucket,
) -> None:
    packed_remote_dir = (
        "features/" + fixture.packed_dir.relative_to(fixture.producer_root).as_posix()
    )
    for name in (
        "manifest.json",
        REMOTE_PACKED_INVENTORY_FILENAME,
        *[path.name for path in sorted(fixture.packed_dir.glob("*.safetensors.zst"))],
    ):
        target = bucket._path(f"{packed_remote_dir}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture.packed_dir / name, target)
    selection_remote_dir = f"{packed_remote_dir}/sel/{fixture.selection_checksum}"
    for path in (
        fixture.selection_dir / "manifest.json",
        *sorted(fixture.selection_dir.glob("*.safetensors.zst")),
    ):
        target = bucket._path(f"{selection_remote_dir}/{path.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _streaming_materializers(
    fixture: _StreamingSpliceFixture,
    tmp_path: Path,
    *,
    trust_existing_cache: bool = False,
) -> tuple[
    Path,
    Path,
    Path,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
]:
    bucket = _LoaderFakeBucket(tmp_path / "bucket")
    _upload_streaming_fixture(fixture, bucket)
    fresh_root = tmp_path / "fresh"
    fresh_packed = fresh_root / fixture.packed_dir.relative_to(fixture.producer_root)
    fresh_provenance = provenance_dir(fresh_packed)
    fresh_provenance.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        provenance_dir(fixture.packed_dir),
        fresh_provenance,
    )
    fresh_splice = fresh_root / fixture.splice_feature_dir.relative_to(fixture.producer_root)
    shutil.copytree(fixture.splice_feature_dir, fresh_splice)
    remote_cfg = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints/conditional_encdec",
            "selection_wait_timeout_seconds": 1,
            "checkpoint_retry_timeout_seconds": 1,
            "retry_interval_seconds": 0.001,
            "resume_latest": True,
            "train_packed": {
                "trust_existing_cache": trust_existing_cache,
            },
        }
    )
    assert remote_cfg is not None
    packed_materializer = RemotePackedRowMaterializer.discover(
        backend=bucket,
        remote=remote_cfg,
        local_packed_dir=fresh_packed,
        feature_root=fresh_root,
        cache_shards=2,
        label="loader packed differential",
    )
    fresh_selection = fresh_packed / "sel" / fixture.selection_checksum
    selection_materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=remote_cfg,
        local_selection_dir=fresh_selection,
        feature_root=fresh_root,
        expected_selection_checksum=fixture.selection_checksum,
        packed_row_fingerprint=(packed_materializer.packed_row_fingerprint),
        packed_stems=packed_materializer.stems(),
        label="loader selection differential",
    )
    controller = _PrepublishedBodyPrefetcher()
    schedule_id = f"streaming-{id(controller)}"
    _attach_prepublished_bodies(
        selection_materializer,
        bucket=bucket,
        controller=controller,
        kind="selection",
        schedule_id=schedule_id,
        paired=True,
        train_schedule=True,
    )
    _attach_prepublished_bodies(
        packed_materializer,
        bucket=bucket,
        controller=controller,
        kind="packed",
        schedule_id=schedule_id,
        paired=True,
        train_schedule=True,
    )
    return (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    )


def _streaming_loader(
    fixture: _StreamingSpliceFixture,
    *,
    packed_dir: Path,
    selection_dir: Path,
    splice_feature_dir: Path,
    B: int,
    label_dropout: float = 0.0,
    resume_state_dict: dict[str, Any] | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    packed_materializer: RemotePackedRowMaterializer | None = None,
    selection_materializer: RemoteSelectionMaterializer | None = None,
    remote_derived_feature_fingerprint: str | None = None,
    remote_read_cancel_event: threading.Event | None = None,
):
    return precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=B,
        feature_pipeline=_streaming_splice_pipeline(fixture.clusters),
        device="cpu",
        label_dropout=label_dropout,
        selection_dir=selection_dir,
        effective_buckets=4,
        n_sae_features=(N_SAE if fp_target_layout is None else None),
        fp_target_layout=fp_target_layout,
        resume_state_dict=resume_state_dict,
        splice_feature_dir=splice_feature_dir,
        feature_clusters=fixture.clusters,
        remote_materializer=selection_materializer,
        packed_materializer=packed_materializer,
        remote_read_cancel_event=remote_read_cancel_event,
        remote_derived_feature_fingerprint=(
            remote_derived_feature_fingerprint
            if remote_derived_feature_fingerprint is not None
            else fixture.derived_feature_fingerprint
            if packed_materializer is not None
            else None
        ),
    )


def test_streaming_loader_propagates_one_cancel_event_to_both_remote_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    cancel_event = threading.Event()
    packed_events: list[threading.Event | None] = []
    selection_events: list[threading.Event | None] = []
    real_packed_read = packed_materializer.read
    real_selection_read = selection_materializer.read
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_load_owner_shard",
        lambda _path: pytest.fail("remote packed materializer invoked the owner helper"),
    )

    def packed_read(
        path: Path,
        *,
        cancel_event: threading.Event | None = None,
        epoch: int = 1,
    ) -> bytes:
        packed_events.append(cancel_event)
        return real_packed_read(path, cancel_event=cancel_event, epoch=epoch)

    def selection_read(
        path: Path,
        *,
        cancel_event: threading.Event | None = None,
        epoch: int = 1,
    ) -> bytes:
        selection_events.append(cancel_event)
        return real_selection_read(path, cancel_event=cancel_event, epoch=epoch)

    monkeypatch.setattr(packed_materializer, "read", packed_read)
    monkeypatch.setattr(selection_materializer, "read", selection_read)
    loader = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
        remote_read_cancel_event=cancel_event,
    )

    next(loader)
    loader.close()

    assert packed_events and all(event is cancel_event for event in packed_events)
    assert selection_events and all(event is cancel_event for event in selection_events)


@pytest.mark.parametrize("body_kind", ["packed", "selection"])
def test_corrupt_remote_decode_preserves_type_and_adds_exact_body_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_kind: str,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    materializer = packed_materializer if body_kind == "packed" else selection_materializer
    expected_dir = fresh_packed if body_kind == "packed" else fresh_selection

    def corrupt_read(
        _path: Path,
        *,
        cancel_event: threading.Event | None = None,
        epoch: int = 1,
    ) -> bytes:
        assert cancel_event is None
        assert epoch == 1
        return b"corrupt-zstd-payload"

    monkeypatch.setattr(materializer, "read", corrupt_read)
    loader = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )

    with pytest.raises(zstd.ZstdError) as error:
        next(loader)

    expected_path = expected_dir / "shard_00000.safetensors.zst"
    notes = getattr(error.value, "__notes__", [])
    assert any(f"path={expected_path}" in note for note in notes)
    assert any("logical_epoch=1" in note for note in notes)
    assert any(f"remote {body_kind}-body decode/validation context" in note for note in notes)
    loader.close()


def test_remote_selection_semantic_failure_adds_exact_body_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    failure = ValueError("synthetic remote selection semantic failure")

    def fail_validation(self: object, **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "assert_selection_shard",
        fail_validation,
    )
    loader = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )

    with pytest.raises(ValueError, match="synthetic remote selection semantic failure") as error:
        next(loader)

    expected_path = fresh_selection / "shard_00000.safetensors.zst"
    assert error.value is failure
    notes = getattr(error.value, "__notes__", [])
    assert any(f"path={expected_path}" in note for note in notes)
    assert any("logical_epoch=1" in note for note in notes)
    loader.close()


def test_remote_packed_semantic_failure_adds_exact_body_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    failure = RemoteTrainingError("synthetic remote packed semantic failure")

    def fail_validation(
        _path: Path,
        *,
        total_segments: int,
    ) -> None:
        assert total_segments > 0
        raise failure

    monkeypatch.setattr(
        packed_materializer,
        "validate_loaded_shard",
        fail_validation,
    )
    loader = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )

    with pytest.raises(
        RemoteTrainingError,
        match="synthetic remote packed semantic failure",
    ) as error:
        next(loader)

    expected_path = fresh_packed / "shard_00000.safetensors.zst"
    assert error.value is failure
    notes = getattr(error.value, "__notes__", [])
    assert any(f"path={expected_path}" in note for note in notes)
    assert any("logical_epoch=1" in note for note in notes)
    loader.close()


def test_streaming_packed_loader_is_byte_identical_across_two_epochs(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    t.manual_seed(991)
    local_loader = _streaming_loader(
        fixture,
        packed_dir=fixture.packed_dir,
        selection_dir=fixture.selection_dir,
        splice_feature_dir=fixture.splice_feature_dir,
        B=3,
        label_dropout=0.25,
    )
    local_batches = [next(local_loader) for _ in range(5)]
    t.manual_seed(991)
    remote_loader = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=3,
        label_dropout=0.25,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )
    remote_batches = [next(remote_loader) for _ in range(5)]
    _assert_nested_exact(
        remote_batches,
        local_batches,
        path="two_epoch_streaming_parity",
    )


def test_remote_cache_integrity_failure_propagates_through_closable_prefetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    local = _streaming_loader(
        fixture,
        packed_dir=fixture.packed_dir,
        selection_dir=fixture.selection_dir,
        splice_feature_dir=fixture.splice_feature_dir,
        B=2,
    )
    expected_first = next(local)
    remote = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )
    second_read_started = threading.Event()
    release_second_read = threading.Event()
    real_read = packed_materializer.read

    def blocking_read(
        path: Path,
        *,
        cancel_event: threading.Event | None = None,
        epoch: int = 1,
    ) -> bytes:
        if path.name == "shard_00001.safetensors.zst":
            second_read_started.set()
            assert release_second_read.wait(timeout=5)
        return real_read(path, cancel_event=cancel_event, epoch=epoch)

    monkeypatch.setattr(packed_materializer, "read", blocking_read)
    prefetched = ClosableBatchPrefetcher(
        remote,
        prefetch_batches=PositiveInt.parse(1),
        stop_callback=lambda: None,
        failure_check=lambda: None,
    )
    first = next(prefetched)
    _assert_nested_exact(first, expected_first, path="prefetch_first")
    assert second_read_started.wait(timeout=5)

    cached_second = fresh_packed / "shard_00001.safetensors.zst"
    body = cached_second.read_bytes()
    cached_second.write_bytes(bytes([body[0] ^ 1]) + body[1:])
    release_second_read.set()
    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        next(prefetched)
    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        prefetched.close()


def test_trusted_and_verified_streaming_paths_are_bit_exact(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    safe = _streaming_materializers(fixture, tmp_path / "safe")
    trusted = _streaming_materializers(
        fixture,
        tmp_path / "trusted",
        trust_existing_cache=True,
    )

    def batches(materializers):
        packed, selection, splice, packed_mat, selection_mat = materializers
        loader = _streaming_loader(
            fixture,
            packed_dir=packed,
            selection_dir=selection,
            splice_feature_dir=splice,
            B=3,
            label_dropout=0.25,
            packed_materializer=packed_mat,
            selection_materializer=selection_mat,
        )
        return [next(loader) for _ in range(5)]

    t.manual_seed(826)
    expected = batches(safe)
    t.manual_seed(826)
    actual = batches(trusted)
    _assert_nested_exact(actual, expected, path="trusted_vs_verified")


@pytest.mark.parametrize(
    ("source_trusted", "resumed_trusted"),
    [(False, True), (True, False)],
)
def test_trusted_and_verified_resume_are_bit_exact_in_both_directions(
    tmp_path: Path,
    source_trusted: bool,
    resumed_trusted: bool,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    source_materializers = _streaming_materializers(
        fixture,
        tmp_path / "source",
        trust_existing_cache=source_trusted,
    )
    resumed_materializers = _streaming_materializers(
        fixture,
        tmp_path / "resumed",
        trust_existing_cache=resumed_trusted,
    )

    def loader(materializers, *, resume_state_dict=None):
        packed, selection, splice, packed_mat, selection_mat = materializers
        return _streaming_loader(
            fixture,
            packed_dir=packed,
            selection_dir=selection,
            splice_feature_dir=splice,
            B=3,
            resume_state_dict=resume_state_dict,
            packed_materializer=packed_mat,
            selection_materializer=selection_mat,
        )

    source = loader(source_materializers)
    prior = [next(source) for _ in range(2)]
    resumed = loader(
        resumed_materializers,
        resume_state_dict=copy.deepcopy(prior[-1][-1]),
    )
    _assert_nested_exact(next(resumed), prior[-1], path="resumed_batch")
    _assert_nested_exact(next(resumed), next(source), path="resumed_continuation")


@pytest.mark.parametrize(
    ("B", "batches_before_checkpoint"),
    [(2, 2), (3, 2)],
)
def test_streaming_and_local_resume_are_bidirectionally_byte_identical(
    tmp_path: Path,
    B: int,
    batches_before_checkpoint: int,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")

    local = _streaming_loader(
        fixture,
        packed_dir=fixture.packed_dir,
        selection_dir=fixture.selection_dir,
        splice_feature_dir=fixture.splice_feature_dir,
        B=B,
    )
    local_batches = [next(local) for _ in range(batches_before_checkpoint)]
    local_state = copy.deepcopy(local_batches[-1][-1])
    remote_resumed = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=B,
        resume_state_dict=local_state,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )
    _assert_nested_exact(
        next(remote_resumed),
        local_batches[-1],
        path="local_to_streaming_resume",
    )
    local_next = next(local)
    _assert_nested_exact(
        next(remote_resumed),
        local_next,
        path="local_to_streaming_continuation",
    )

    remote = _streaming_loader(
        fixture,
        packed_dir=fresh_packed,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=B,
        packed_materializer=packed_materializer,
        selection_materializer=selection_materializer,
    )
    remote_batches = [next(remote) for _ in range(batches_before_checkpoint)]
    remote_state = copy.deepcopy(remote_batches[-1][-1])
    local_resumed = _streaming_loader(
        fixture,
        packed_dir=fixture.packed_dir,
        selection_dir=fixture.selection_dir,
        splice_feature_dir=fixture.splice_feature_dir,
        B=B,
        resume_state_dict=remote_state,
    )
    _assert_nested_exact(
        next(local_resumed),
        remote_batches[-1],
        path="streaming_to_local_resume",
    )
    _assert_nested_exact(
        next(local_resumed),
        next(remote),
        path="streaming_to_local_continuation",
    )


def test_streaming_compact_fp_certificate_digest_matches_local(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    vocab = effective_feature_vocab(fixture.clusters, N_SAE)
    assert vocab is not None
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=N_SAE,
        feature_mask=fixture.clusters.feature_mask,
        feature_clusters=fixture.clusters,
        feature_vocab=vocab,
    )
    local_batch = next(
        _streaming_loader(
            fixture,
            packed_dir=fixture.packed_dir,
            selection_dir=fixture.selection_dir,
            splice_feature_dir=fixture.splice_feature_dir,
            B=2,
            fp_target_layout=layout,
        )
    )
    remote_batch = next(
        _streaming_loader(
            fixture,
            packed_dir=fresh_packed,
            selection_dir=fresh_selection,
            splice_feature_dir=fresh_splice,
            B=2,
            fp_target_layout=layout,
            packed_materializer=packed_materializer,
            selection_materializer=selection_materializer,
        )
    )
    _assert_nested_exact(remote_batch, local_batch, path="compact_fp")
    assert (
        remote_batch[-1]["fp_target_source_certificate_sha256"]
        == local_batch[-1]["fp_target_source_certificate_sha256"]
    )


def test_mode_off_explicit_none_is_byte_identical_to_omitted(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    omitted = _streaming_loader(
        fixture,
        packed_dir=fixture.packed_dir,
        selection_dir=fixture.selection_dir,
        splice_feature_dir=fixture.splice_feature_dir,
        B=3,
    )
    explicit_none = precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=3,
        feature_pipeline=_streaming_splice_pipeline(fixture.clusters),
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=4,
        n_sae_features=N_SAE,
        splice_feature_dir=fixture.splice_feature_dir,
        feature_clusters=fixture.clusters,
        remote_materializer=None,
        packed_materializer=None,
        remote_derived_feature_fingerprint=None,
    )
    _assert_nested_exact(
        next(explicit_none),
        next(omitted),
        path="mode_off_explicit_none",
    )


@pytest.mark.parametrize("stream_train_packed", [False, True])
def test_streaming_loader_rejects_selection_manifest_fingerprint_mutation(
    tmp_path: Path,
    stream_train_packed: bool,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    manifest_path = fresh_selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["splice_counterfactual"]["derived_feature_fingerprint"] = "d" * 64
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    loader = _streaming_loader(
        fixture,
        packed_dir=(fresh_packed if stream_train_packed else fixture.packed_dir),
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        packed_materializer=(packed_materializer if stream_train_packed else None),
        selection_materializer=selection_materializer,
    )

    with pytest.raises(
        ValueError,
        match="derived_feature_fingerprint mismatch",
    ):
        next(loader)


def test_streaming_derived_val_loader_rejects_resolved_fingerprint_mutation(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        _fresh_packed,
        fresh_selection,
        fresh_splice,
        _packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    manifest_path = fresh_selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["splice_counterfactual"]["derived_feature_fingerprint"] = "d" * 64
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    loader = _streaming_loader(
        fixture,
        packed_dir=fixture.packed_dir,
        selection_dir=fresh_selection,
        splice_feature_dir=fresh_splice,
        B=2,
        selection_materializer=selection_materializer,
        remote_derived_feature_fingerprint=(fixture.derived_feature_fingerprint),
    )

    with pytest.raises(
        ValueError,
        match="derived_feature_fingerprint mismatch",
    ):
        next(loader)


def test_streaming_parameter_guards_fail_before_body_use(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    feature_pipeline = _streaming_splice_pipeline(fixture.clusters)
    with pytest.raises(ValueError, match="requires remote selection"):
        next(
            precomputed_data_loader(
                packed_row_dir=fresh_packed,
                B=2,
                feature_pipeline=feature_pipeline,
                device="cpu",
                selection_dir=fresh_selection,
                effective_buckets=4,
                n_sae_features=N_SAE,
                splice_feature_dir=fresh_splice,
                feature_clusters=fixture.clusters,
                packed_materializer=packed_materializer,
                remote_derived_feature_fingerprint=(fixture.derived_feature_fingerprint),
            )
        )
    with pytest.raises(ValueError, match="requires the verified remote"):
        next(
            precomputed_data_loader(
                packed_row_dir=fresh_packed,
                B=2,
                feature_pipeline=feature_pipeline,
                device="cpu",
                selection_dir=fresh_selection,
                effective_buckets=4,
                n_sae_features=N_SAE,
                splice_feature_dir=fresh_splice,
                feature_clusters=fixture.clusters,
                remote_materializer=selection_materializer,
                packed_materializer=packed_materializer,
            )
        )
    with pytest.raises(ValueError, match="does not match the remote packed"):
        next(
            precomputed_data_loader(
                packed_row_dir=fixture.packed_dir,
                B=2,
                feature_pipeline=feature_pipeline,
                device="cpu",
                selection_dir=fresh_selection,
                effective_buckets=4,
                n_sae_features=N_SAE,
                splice_feature_dir=fresh_splice,
                feature_clusters=fixture.clusters,
                remote_materializer=selection_materializer,
                packed_materializer=packed_materializer,
                remote_derived_feature_fingerprint=(fixture.derived_feature_fingerprint),
            )
        )
