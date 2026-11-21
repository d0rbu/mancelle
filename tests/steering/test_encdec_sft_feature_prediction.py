"""Tests for next-segment feature-prediction auxiliary loss in encoder-decoder SFT.

The pretraining script (scripts/conditional_encdec_train.py) builds the model
with ``feature_pred_config`` and threads ``fp_targets`` through the forward
pass. The SFT script previously did not, so checkpoints from runs with
``feature_prediction`` (e.g. ``conditional_encdec_d12_fp_s32``) crashed at
``state_dict`` load with "Unexpected key(s) in state_dict:
feature_pred_head.weight, feature_pred_head.bias".

This file pins:

* **Unit:** ``_resolve_fp_config_from_meta`` extracts the config from
  pretrain or chained-SFT metadata and returns ``None`` for baseline runs.
* **Functional:** ``make_sft_data_generator`` yields 7-tuples when FP
  is inactive, and model-safe 8-tuples with ``seg_doc_id`` whenever dense
  ``fp_targets`` are present; the dense target tensor scatters all-feature
  percentiles into the right ``(B, n_seg, n_sae)`` cells; conditioned-without-fp
  paths emit ``None`` in that slot.
* **Integration (tiny CPU):** model with fp_head returns ``ForwardResult``
  on a SFT batch; gradient flows through ``feature_pred_head``; combined
  loss decreases over a short training run.
* **Hard-fail invariants:** missing percentile table when fp_config is
  active raises with a clear message; n_sae_features mismatch between
  config-derived value and checkpoint-derived value is caught.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_sft import (
    SFTState,
    _checkpoint_user_config_with_fp,
    _resolve_fp_config_from_meta,
    _serialize_fp_config,
    make_sft_data_generator,
)
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.model import ForwardResult
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.percentile_table import (
    build_percentile_table,
    magnitude_to_percentile,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingSelection,
)
from steering.conditional.sampling import (
    sample_features as real_sample_features,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.sft.loss_mask import LossMaskMode
from steering.sft_schema import SFTSplit
from steering.stats_types import PerFeatureStat
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_tiny_model,
    setup_optimizer_simple,
)
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    build_synthetic_features,
    patch_load_sft_features,
)


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    """Force SDPA attention so backward works on CPU.

    flex_attention has no CPU backward — same fixture as the integration
    suite. Restoring on teardown keeps the rest of the suite's behavior
    unchanged.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Synthetic percentile table + fp config helpers
# ---------------------------------------------------------------------------


def _make_per_feature_stats(
    n_features: int = N_SAE_FEATURES,
) -> dict[str, PerFeatureStat]:
    """Build a deterministic ``PerFeatureStat`` dict for ``build_percentile_table``.

    Every feature gets the same monotonically increasing p1..p99 grid so
    ``magnitude_to_percentile`` always returns interpretable mid-range
    values for the synthetic feature magnitudes used in tests.
    """
    pcts = {f"p{k}": float(k) for k in range(1, 100)}
    return {
        str(fid): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=50.0,
            mean_mag_incl_zeros=25.0,
            std_mag=20.0,
            min_mag=1.0,
            max_mag=99.0,
            median_mag=50.0,
            percentiles=pcts,
        )
        for fid in range(n_features)
    }


def _make_synthetic_percentile_table(
    n_features: int = N_SAE_FEATURES,
) -> t.Tensor:
    """Convenience: stats → table in one call."""
    return build_percentile_table(_make_per_feature_stats(n_features), n_features)


def _make_fp_config(
    weight: float = 0.1,
    layer: int = -1,
    loss_mode: FeaturePredLossMode = FeaturePredLossMode.FULL_SEGMENT,
    loss_fn: FeaturePredLossFn = FeaturePredLossFn.BCE,
) -> FeaturePredictionConfig:
    return FeaturePredictionConfig(weight=weight, layer=layer, loss_mode=loss_mode, loss_fn=loss_fn)


FP_FORWARD_SEGMENTS_PER_CONV = 2


def _build_forward_fp_features(task, tokenizer):
    return build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=FP_FORWARD_SEGMENTS_PER_CONV,
    )


def _build_tiny_encdec_with_fp(
    fp_config: FeaturePredictionConfig | None = None,
):
    """Tiny EncoderDecoderGPT with fp_head wired through cross-attention every layer."""
    return build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=fp_config or _make_fp_config(),
        n_layer=2,
        n_embd=64,
        seq_len=64,
        xattn_frequency=1,
    )


def _make_gen_with_fp(
    task,
    tokenizer,
    *,
    fp_active: bool = True,
    B: int = 2,
    T: int = 64,
    threshold: int = 1000,
    n_sae_features: int = N_SAE_FEATURES,
    sft_state: SFTState | None = None,
    device: str = "cpu",
    return_seg_doc_id: bool = False,
    feature_mask: FeatureMask | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    percentile_table: t.Tensor | None = None,
    sampling_bundle: SamplingBundle | None = None,
):
    """Build a conditioned ``make_sft_data_generator`` with the new fp kwargs.

    Mirrors ``_make_gen`` from the integration suite but adds the
    ``n_sae_features`` and ``fp_percentile_table`` plumbing introduced in
    this PR. ``fp_active=False`` deactivates the fp target build (passes
    ``None`` for both kwargs) so the same helper covers the
    "no-fp-conditioning" smoke tests.
    """
    threshold_cfg = ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=threshold,
    )
    boundaries = (
        t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )
    table = (
        percentile_table
        if percentile_table is not None
        else _make_synthetic_percentile_table(n_sae_features)
        if fp_active
        else None
    )
    return make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device=device,
        sft_state=sft_state or SFTState(),
        feature_dir="/dummy/features",
        parquet_dir="/dummy/parquet",
        threshold_cfg=threshold_cfg,
        boundaries=boundaries,
        sampling_bundle=sampling_bundle,
        n_sae_features=(n_sae_features if fp_active and fp_target_layout is None else None),
        fp_percentile_table=table,
        fp_target_layout=fp_target_layout,
        return_seg_doc_id=return_seg_doc_id,
        feature_mask=feature_mask,
        min_loss_tokens=1,
        alignment_probe_samples=0,
        expected_split=SFTSplit.TRAIN,
    )


def _fp_targets_from_batch(batch):
    return batch[-2] if len(batch) == 8 else batch[-1]


# ---------------------------------------------------------------------------
# Unit: _resolve_fp_config_from_meta
# ---------------------------------------------------------------------------


class TestResolveFPConfigFromMeta:
    """``_resolve_fp_config_from_meta`` extracts the FP config from
    checkpoint metadata. Returns ``None`` for baseline (no-FP) runs.

    This is architecture-only — the helper answers *does this
    checkpoint carry an fp_head?*. Whether the SFT run actually
    computes the auxiliary loss depends on the fp kwargs and availability of
    ``fp_percentile_table``.
    """

    def test_pretrain_meta_returns_config(self) -> None:
        meta = {
            "user_config": {
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.5,
                    "layer": -1,
                    "loss_mode": "best_token",
                    "loss_fn": "bce",
                }
            }
        }
        cfg = _resolve_fp_config_from_meta(meta)
        assert isinstance(cfg, FeaturePredictionConfig)
        assert cfg.weight == 0.5
        assert cfg.layer == -1
        assert cfg.loss_mode is FeaturePredLossMode.BEST_TOKEN
        assert cfg.loss_fn is FeaturePredLossFn.BCE

    def test_chained_sft_meta_returns_config(self) -> None:
        """SFT checkpoints store fp config under top-level
        ``feature_prediction_config`` (not ``user_config.feature_prediction``)
        because the SFT user_config schema doesn't include pretrain fields."""
        meta = {
            "user_config": {},
            "feature_prediction_config": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.25,
                "layer": -2,
                "loss_mode": "full_segment",
                "loss_fn": "mse",
            },
        }
        cfg = _resolve_fp_config_from_meta(meta)
        assert isinstance(cfg, FeaturePredictionConfig)
        assert cfg.weight == 0.25
        assert cfg.layer == -2
        assert cfg.loss_mode is FeaturePredLossMode.FULL_SEGMENT
        assert cfg.loss_fn is FeaturePredLossFn.MSE

    def test_conflicting_checkpoint_mirrors_are_rejected(self) -> None:
        """A stale mirror must not silently override the effective saved config."""
        meta = {
            "user_config": {
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.1,
                    "layer": -1,
                    "loss_mode": "best_token",
                    "loss_fn": "bce",
                }
            },
            "feature_prediction_config": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 99.0,
                "layer": 0,
                "loss_mode": "full_segment",
                "loss_fn": "mse",
            },
        }
        with pytest.raises(ValueError, match="conflicting.*mirrors"):
            _resolve_fp_config_from_meta(meta)

    def test_missing_returns_none(self) -> None:
        cfg = _resolve_fp_config_from_meta({"user_config": {}})
        assert cfg is None

    def test_null_user_config_value_returns_none(self) -> None:
        """Pretrain checkpoints with ``feature_prediction: null`` (the
        baseline default) explicitly serialise the key as ``None`` —
        we must not try to construct a config from ``None``."""
        meta = {"user_config": {"feature_prediction": None}}
        cfg = _resolve_fp_config_from_meta(meta)
        assert cfg is None

    def test_returns_config_independent_of_training_step(self) -> None:
        """The helper is architecture-only and always returns the
        checkpoint's fp_config when present. Whether the dataloader emits
        fp targets is controlled only by the fp kwargs passed to the SFT
        generator."""
        meta = {
            "user_config": {
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.5,
                    "layer": -1,
                    "loss_mode": "best_token",
                    "loss_fn": "bce",
                }
            }
        }
        cfg = _resolve_fp_config_from_meta(meta)
        assert isinstance(cfg, FeaturePredictionConfig)
        assert cfg.weight == 0.5

    def test_default_layer_when_pretrain_omits_field(self) -> None:
        """Missing ``layer`` field should default to -1 (last decoder block)
        — matches the pretraining default in ``conditional_encdec_train.yaml``."""
        meta = {
            "user_config": {
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.1,
                    "loss_mode": "best_token",
                    "loss_fn": "bce",
                }
            }
        }
        cfg = _resolve_fp_config_from_meta(meta)
        assert cfg is not None
        assert cfg.layer == -1


# ---------------------------------------------------------------------------
# Functional: make_sft_data_generator yields model-safe tuples
# ---------------------------------------------------------------------------


class TestSFTGeneratorYieldsModelSafeTuple:
    """FP-inactive batches use the legacy 7-tuple. FP-active batches include
    ``seg_doc_id`` as an 8th element because the model requires it for
    cross-document feature-prediction masking."""

    def test_conditioned_fp_inactive_yields_7tuple_with_none_fp_target(self) -> None:
        """Full conditioning with fp inactive yields a 7-tuple ending in None."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=False,
                B=2,
                T=64,
            )
            batch = next(gen)
        assert len(batch) == 7, f"Expected 7-tuple, got {len(batch)}"
        *_, fp_targets = batch
        assert fp_targets is None

    def test_conditioned_no_fp_yields_7tuple_with_none_fp_target(self) -> None:
        """No fp percentile table means no fp targets are built."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=False,
                B=2,
                T=64,
            )
            batch = next(gen)
        assert len(batch) == 7
        *_, fp_targets = batch
        assert fp_targets is None

    def test_conditioned_with_fp_yields_dense_targets(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(
            task,
            tokenizer,
            features_per_segment=3,
        )
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=True,
                B=2,
                T=64,
            )
            batch = next(gen)
        assert len(batch) == 8
        _inp, _tgt, _fids, _mags, _mask, seg_map, fp_targets, seg_doc_id = batch
        assert fp_targets is not None
        assert seg_doc_id.shape == fp_targets.shape[:2]
        B, T = seg_map.shape
        assert fp_targets.dim() == 3
        assert fp_targets.shape[0] == B
        assert fp_targets.shape[2] == N_SAE_FEATURES
        assert fp_targets.dtype == t.float32

    def test_fp_targets_normalized_to_0_1(self) -> None:
        """Percentiles from ``magnitude_to_percentile`` live in [0.5, 99.5];
        normalised to [0, 1] via /100. The dataloader must emit values in
        that range and never exceed 1.0 (post-clamp)."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=True,
                B=2,
                T=64,
            )
            batch = next(gen)
        fp_targets = _fp_targets_from_batch(batch)
        assert fp_targets is not None
        assert (fp_targets >= 0.0).all()
        assert (fp_targets <= 1.0).all()

    def test_missing_sft_percentile_rows_raise_and_never_fabricate_targets(
        self,
    ) -> None:
        """An unseen feature must abort SFT, not get a fabricated FP target.

        The feature-prediction path supervises the model TOWARD the percentile
        it looks up. Substituting any default for a missing statistics row
        would teach a specific activation level for a feature whose activation
        is simply unknown, so conditioned SFT keeps the primitive's raise.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        table = _make_synthetic_percentile_table()
        table[0].zero_()

        with (
            patch_load_sft_features(features),
            pytest.raises(
                ValueError,
                match=r"missing for 1 distinct feature ID\(s\) \(first 1: \[0\]\)",
            ),
        ):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=True,
                B=2,
                T=64,
                percentile_table=table,
            )
            next(gen)

    def test_missing_percentile_error_names_the_actionable_fix(self) -> None:
        """The raise must tell the operator WHICH statistics set to rebuild."""
        table = _make_synthetic_percentile_table(n_features=4)
        table[0].zero_()
        table[2].zero_()

        with pytest.raises(ValueError) as excinfo:
            magnitude_to_percentile(
                t.tensor([0, 1, 2], dtype=t.int64),
                t.tensor([5.0, 50.0, 5.0]),
                table,
            )

        message = str(excinfo.value)
        assert "2 distinct feature ID(s)" in message
        assert "(first 2: [0, 2])" in message
        assert "RAW pre-selection feature ids" in message
        assert "not only the allowed representatives" in message

    def test_sampling_without_fp_also_raises_on_missing_rows(self) -> None:
        """Sampling-only SFT inherits the same fail-loud percentile contract."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        table = _make_synthetic_percentile_table()
        table[0].zero_()
        sampling_bundle = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=100.0,
                std=1e-12,
            ),
            percentile_table=table,
            rng=t.Generator().manual_seed(123),
        )

        with (
            patch_load_sft_features(features),
            pytest.raises(
                ValueError,
                match=r"missing for 1 distinct feature ID\(s\)",
            ),
        ):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=False,
                B=2,
                T=64,
                sampling_bundle=sampling_bundle,
            )
            next(gen)

    def test_fp_targets_sparse_uses_only_features_present(self) -> None:
        """Most cells must be 0 (the synthetic build only uses ~3 fids per
        segment, but fp_targets has ``n_sae_features`` columns)."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(
            task,
            tokenizer,
            features_per_segment=3,
        )
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=True,
                B=2,
                T=64,
            )
            batch = next(gen)
        fp_targets = _fp_targets_from_batch(batch)
        assert fp_targets is not None
        nonzero = (fp_targets > 0).sum().item()
        total = fp_targets.numel()
        assert nonzero / total < 0.1, (
            f"fp_targets density {nonzero / total:.3%} is too high — sparse "
            f"scatter must zero-fill non-active features"
        )
        assert nonzero > 0, "Some features must be present in fp_targets"

    def test_fp_targets_match_segment_count_in_batch(self) -> None:
        """``batch_n_seg`` is computed by the generator. ``fp_targets`` must
        have exactly that many segments along dim 1."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=True,
                B=2,
                T=64,
            )
            batch = next(gen)
        assert len(batch) == 8
        _inp, _tgt, enc_fids, _mags, _mask, _seg, fp_targets, seg_doc_id = batch
        assert fp_targets is not None
        assert seg_doc_id.shape == fp_targets.shape[:2]
        assert fp_targets.shape[1] == enc_fids.shape[1], (
            f"fp_targets n_seg ({fp_targets.shape[1]}) != enc_fids n_seg ({enc_fids.shape[1]})"
        )

    def test_feature_mask_filters_conditioning_and_fp_targets(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer, features_per_segment=3)
        mask = FeatureMask("blacklist", (0,))
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                fp_active=True,
                B=2,
                T=64,
                feature_mask=mask,
            )
            batch = next(gen)

        _inp, _tgt, enc_fids, _mags, enc_mask, _seg, fp_targets, _seg_doc_id = batch
        assert fp_targets is not None
        valid_fids = enc_fids[enc_mask]
        assert valid_fids.numel() > 0
        assert not bool((valid_fids == 0).any())
        assert bool((fp_targets > 0).any())
        assert bool((fp_targets[..., 0] == 0).all())

    def test_sampling_reuses_fp_percentile_scores_when_table_identity_matches(
        self,
    ) -> None:
        """The generator should pass all-feature percentile scores to sampling."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        table = _make_synthetic_percentile_table()
        boundaries = (
            t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
            .unsqueeze(0)
            .expand(N_SAE_FEATURES, -1)
            .contiguous()
        )
        sampling_bundle = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=100.0,
                std=1e-12,
            ),
            percentile_table=table,
            rng=t.Generator().manual_seed(123),
        )
        captured: list[tuple[t.Tensor, t.Tensor, t.Tensor | None]] = []

        def _capture_sample_features(
            feature_ids,
            feature_mags,
            percentile_table,
            config,
            rng,
            override_weights=None,
        ):
            captured.append(
                (
                    feature_ids.detach().clone(),
                    feature_mags.detach().clone(),
                    None if override_weights is None else override_weights.detach().clone(),
                )
            )
            return real_sample_features(
                feature_ids,
                feature_mags,
                percentile_table,
                config,
                rng,
                override_weights=override_weights,
            )

        with (
            patch_load_sft_features(features),
            patch(
                "steering.conditional.encoder_decoder.sft_selection.sample_features",
                side_effect=_capture_sample_features,
            ),
        ):
            gen = make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                boundaries=boundaries,
                sampling_bundle=sampling_bundle,
                n_sae_features=N_SAE_FEATURES,
                fp_percentile_table=table,
                min_loss_tokens=1,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )
            next(gen)

        assert captured
        for feature_ids, feature_mags, override_weights in captured:
            assert override_weights is not None
            expected = magnitude_to_percentile(feature_ids.to(t.int64), feature_mags, table)
            t.testing.assert_close(override_weights, expected)

    def test_feature_mask_filters_sampling_candidates_and_weights(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        table = _make_synthetic_percentile_table()
        boundaries = (
            t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
            .unsqueeze(0)
            .expand(N_SAE_FEATURES, -1)
            .contiguous()
        )
        sampling_bundle = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=100.0,
                std=1e-12,
            ),
            percentile_table=table,
            rng=t.Generator().manual_seed(123),
        )
        mask = FeatureMask("blacklist", (0,))
        captured: list[tuple[t.Tensor, t.Tensor, t.Tensor | None]] = []

        def _capture_sample_features(
            feature_ids,
            feature_mags,
            percentile_table,
            config,
            rng,
            override_weights=None,
        ):
            captured.append(
                (
                    feature_ids.detach().clone(),
                    feature_mags.detach().clone(),
                    None if override_weights is None else override_weights.detach().clone(),
                )
            )
            return real_sample_features(
                feature_ids,
                feature_mags,
                percentile_table,
                config,
                rng,
                override_weights=override_weights,
            )

        with (
            patch_load_sft_features(features),
            patch(
                "steering.conditional.encoder_decoder.sft_selection.sample_features",
                side_effect=_capture_sample_features,
            ),
        ):
            gen = make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                boundaries=boundaries,
                sampling_bundle=sampling_bundle,
                n_sae_features=N_SAE_FEATURES,
                fp_percentile_table=table,
                feature_mask=mask,
                min_loss_tokens=1,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )
            next(gen)

        assert captured
        for feature_ids, feature_mags, override_weights in captured:
            assert not bool((feature_ids == 0).any())
            assert override_weights is not None
            expected = magnitude_to_percentile(feature_ids.to(t.int64), feature_mags, table)
            t.testing.assert_close(override_weights, expected)

    def test_sampling_does_not_reuse_fp_scores_when_tables_differ(self) -> None:
        """Different percentile-table objects must keep sampling independent."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        fp_table = _make_synthetic_percentile_table()
        sampling_table = fp_table.clone()
        boundaries = (
            t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
            .unsqueeze(0)
            .expand(N_SAE_FEATURES, -1)
            .contiguous()
        )
        sampling_bundle = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=100.0,
                std=1e-12,
            ),
            percentile_table=sampling_table,
            rng=t.Generator().manual_seed(123),
        )
        captured_override_weights: list[t.Tensor | None] = []

        def _capture_sample_features(
            feature_ids,
            feature_mags,
            percentile_table,
            config,
            rng,
            override_weights=None,
        ):
            captured_override_weights.append(override_weights)
            return real_sample_features(
                feature_ids,
                feature_mags,
                percentile_table,
                config,
                rng,
                override_weights=override_weights,
            )

        with (
            patch_load_sft_features(features),
            patch(
                "steering.conditional.encoder_decoder.sft_selection.sample_features",
                side_effect=_capture_sample_features,
            ),
        ):
            gen = make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                boundaries=boundaries,
                sampling_bundle=sampling_bundle,
                n_sae_features=N_SAE_FEATURES,
                fp_percentile_table=fp_table,
                min_loss_tokens=1,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )
            next(gen)

        assert captured_override_weights
        assert all(weights is None for weights in captured_override_weights)


# ---------------------------------------------------------------------------
# Functional: _build_sft_val_loader passes fp args through
# ---------------------------------------------------------------------------


class TestBuildSFTValLoaderPassthrough:
    """``_build_sft_val_loader`` must accept and forward
    ``n_sae_features`` and ``fp_percentile_table`` to
    ``make_sft_data_generator``."""

    def test_val_loader_accepts_fp_args_and_yields_model_safe_tuple(self) -> None:
        from scripts.train.conditional_encdec_sft import _build_sft_val_loader

        tokenizer = SFTMockTokenizer()
        train_task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        val_task = SingleTurnTask(n=4, user_chars=7, asst_chars=13)
        assert train_task[0] != val_task[0]
        features = build_synthetic_features(val_task, tokenizer)
        boundaries = (
            t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
            .unsqueeze(0)
            .expand(N_SAE_FEATURES, -1)
            .contiguous()
        )
        threshold_cfg = ThresholdConfig(
            threshold_type=ThresholdType.segment_topk,
            threshold=1000,
        )
        table = _make_synthetic_percentile_table()

        with patch_load_sft_features(
            features,
            split=SFTSplit.VAL,
            mixture="capabilities",
        ):
            loader = _build_sft_val_loader(
                val_task=val_task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                ddp_rank=0,
                ddp_world_size=1,
                feature_dir="/dummy",
                expected_feature_dir="/dummy",
                parquet_dir="/dummy",
                threshold_cfg=threshold_cfg,
                boundaries=boundaries,
                precomputed_thresholds=None,
                sampling_bundle=None,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
                min_loss_tokens=8,
                n_sae_features=N_SAE_FEATURES,
                fp_percentile_table=table,
                alignment_probe_samples=0,
            )
            batch = next(loader)
        assert len(batch) == 8
        fp_targets = batch[-2]
        seg_doc_id = batch[-1]
        assert fp_targets is not None, (
            "Conditioned val with fp_percentile_table must build fp_targets"
        )
        assert seg_doc_id.shape == fp_targets.shape[:2]
        assert fp_targets.shape[0] == 2
        assert fp_targets.shape[2] == N_SAE_FEATURES


# ---------------------------------------------------------------------------
# Integration: tiny CPU forward + ForwardResult unpacking
# ---------------------------------------------------------------------------


class TestSFTForwardWithFP:
    """Tiny EncoderDecoderGPT with fp_head + an SFT batch carrying
    fp_targets must produce ``ForwardResult`` (not a plain loss tensor)."""

    def test_forward_returns_forward_result_when_fp_active(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = _build_forward_fp_features(task, tokenizer)
        model = _build_tiny_encdec_with_fp()
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, return_seg_doc_id=True)
            batch = next(gen)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
        model.eval()
        with t.no_grad():
            result = model(
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                fp_targets=fp_targets,
                seg_doc_id=seg_doc_id,
            )
        assert isinstance(result, ForwardResult)
        assert math.isfinite(result.loss.item())
        assert math.isfinite(result.lm_loss.item())
        assert math.isfinite(result.feature_pred_loss.item())
        assert result.loss.item() >= result.lm_loss.item() - 1e-6, (
            f"combined ({result.loss.item()}) should be ≥ lm "
            f"({result.lm_loss.item()}) within numerical tolerance"
        )

    def test_forward_returns_plain_loss_when_fp_inactive(self) -> None:
        """Same model, but with fp_targets=None ⇒ model returns a plain
        scalar loss (the fp_head is silently skipped)."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        model = _build_tiny_encdec_with_fp()
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                B=2,
                T=64,
                fp_active=False,
            )
            batch = next(gen)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
        model.eval()
        with t.no_grad():
            result = model(
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                fp_targets=None,
            )
        assert not isinstance(result, ForwardResult)
        assert math.isfinite(result.item())

    def test_baseline_model_without_fp_head_returns_plain_loss(self) -> None:
        """Sanity: a baseline model (no fp_head) ignores fp_targets entirely
        and returns a plain scalar loss. Pins backward compatibility."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        model = build_tiny_model(
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            feature_pred_config=None,
            n_layer=2,
            n_embd=64,
            seq_len=64,
            xattn_frequency=1,
        )
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, fp_active=True)
            batch = next(gen)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
        model.eval()
        with t.no_grad():
            result = model(
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                fp_targets=fp_targets,
                seg_doc_id=seg_doc_id,
            )
        assert not isinstance(result, ForwardResult)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# Integration: backward + gradient flow through fp_head
# ---------------------------------------------------------------------------


class TestSFTBackwardThroughFP:
    """A backward pass on ``ForwardResult.loss`` must populate gradients
    in ``feature_pred_head.weight`` and ``feature_pred_head.bias``."""

    def test_fp_head_receives_gradient(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = _build_forward_fp_features(task, tokenizer)
        model = _build_tiny_encdec_with_fp()
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, return_seg_doc_id=True)
            batch = next(gen)

        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
        result.loss.backward()

        assert model.feature_pred_head is not None
        w_grad = model.feature_pred_head.weight.grad
        b_grad = model.feature_pred_head.bias.grad
        assert w_grad is not None
        assert b_grad is not None
        assert float(w_grad.abs().sum().item()) > 0.0, (
            "fp_head.weight must receive non-zero gradient"
        )
        assert float(b_grad.abs().sum().item()) > 0.0, "fp_head.bias must receive non-zero gradient"

    def test_decoder_still_receives_gradient_when_fp_active(self) -> None:
        """Sanity: adding fp loss must not zero out the LM gradient path."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = _build_forward_fp_features(task, tokenizer)
        model = _build_tiny_encdec_with_fp()
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, return_seg_doc_id=True)
            batch = next(gen)

        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
        result.loss.backward()

        any_decoder_grad = False
        for name, param in model.named_parameters():
            if name.startswith("transformer.") and param.grad is not None:
                if float(param.grad.abs().sum().item()) > 0.0:
                    any_decoder_grad = True
                    break
        assert any_decoder_grad

    def test_fp_loss_decreases_over_short_run(self) -> None:
        """Multi-step training must decrease combined loss on a stable
        conditioned dataset. This is a learnability smoke test — if the
        backward pass is broken (fp_head detached, wrong target tensor,
        etc.) the loss will oscillate around the random-init value
        instead of trending down."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = _build_forward_fp_features(task, tokenizer)
        model = _build_tiny_encdec_with_fp()
        optimizer = setup_optimizer_simple(model, lr=0.01)

        losses: list[float] = []
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, return_seg_doc_id=True)
            for _ in range(16):
                batch = next(gen)
                (
                    inputs,
                    targets,
                    enc_fids,
                    enc_mags,
                    enc_mask,
                    seg_map,
                    fp_targets,
                    seg_doc_id,
                ) = batch
                model.train()
                optimizer.zero_grad(set_to_none=True)
                result = model(
                    inputs,
                    targets,
                    enc_fids,
                    enc_mags,
                    enc_mask,
                    seg_map,
                    fp_targets=fp_targets,
                    seg_doc_id=seg_doc_id,
                )
                result.loss.backward()
                losses.append(float(result.loss.item()))
                optimizer.step()

        first_half = sum(losses[: len(losses) // 2]) / max(1, len(losses) // 2)
        second_half = sum(losses[len(losses) // 2 :]) / max(1, len(losses) - len(losses) // 2)
        assert math.isfinite(first_half) and math.isfinite(second_half)
        assert second_half < first_half, (
            f"Combined loss must trend down; first half {first_half:.4f}, "
            f"second half {second_half:.4f}, losses {losses}"
        )

    def test_fp_loss_components_are_separable(self) -> None:
        """``ForwardResult.loss == lm_loss + weight * feature_pred_loss``
        within numerical tolerance. Pins the combination formula."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = _build_forward_fp_features(task, tokenizer)
        weight = 0.25
        model = _build_tiny_encdec_with_fp(
            fp_config=_make_fp_config(weight=weight),
        )
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, return_seg_doc_id=True)
            batch = next(gen)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
        model.eval()
        with t.no_grad():
            result = model(
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                fp_targets=fp_targets,
                seg_doc_id=seg_doc_id,
            )
        expected = result.lm_loss.item() + weight * result.feature_pred_loss.item()
        assert math.isclose(
            result.loss.item(),
            expected,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ), (
            f"loss ({result.loss.item()}) != lm_loss ({result.lm_loss.item()}) "
            f"+ {weight} * fp_loss ({result.feature_pred_loss.item()})"
        )


# ---------------------------------------------------------------------------
# Functional: SFT seg_doc_id metadata for feature-prediction loss
# ---------------------------------------------------------------------------


class TestSFTSegDocIdForFeaturePrediction:
    """SFT packs multiple conversations into one row, just like the
    pretraining dataloader packs multiple documents. ``seg_doc_id`` must mark
    those conversation boundaries so the next-segment FP loss never trains
    across them.
    """

    def test_seg_doc_id_marks_packed_conversation_boundaries(self) -> None:
        """A row containing many short conversations should have repeated
        local ids per conversation, `-1` for segment 0/padding, and at least
        one non-negative doc-id transition at a conversation boundary."""

        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=1, asst_chars=6)
        features = _build_forward_fp_features(task, tokenizer)

        with patch_load_sft_features(features):
            batch = next(
                _make_gen_with_fp(
                    task,
                    tokenizer,
                    B=1,
                    T=128,
                    return_seg_doc_id=True,
                )
            )

        inputs, _targets, enc_fids, _mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
        assert fp_targets is not None
        assert seg_doc_id.shape == enc_fids.shape[:2]
        assert seg_doc_id.dtype == t.int64
        assert seg_doc_id.device == inputs.device

        # Segment 0 is the shared empty encoder segment. Padding slots also
        # use -1, while real selected feature segments carry a local
        # conversation id.
        assert int(seg_doc_id[0, 0].item()) == -1
        assert t.equal(seg_doc_id >= 0, enc_mask.any(dim=-1))

        real_doc_ids = [int(x) for x in seg_doc_id[0].tolist() if x >= 0]
        assert real_doc_ids[:4] == [0, 0, 1, 1]

        nonzero_seg_indices = seg_map[0][seg_map[0] > 0].unique()
        assert nonzero_seg_indices.numel() >= 4
        assert (seg_doc_id[0, nonzero_seg_indices] >= 0).all()

        same_doc = (seg_doc_id[:, :-1] == seg_doc_id[:, 1:]) & (seg_doc_id[:, :-1] >= 0)
        cross_doc = (seg_doc_id[:, :-1] >= 0) & (seg_doc_id[:, 1:] >= 0) & ~same_doc
        assert same_doc.any(), "At least one within-conversation segment pair is needed"
        assert cross_doc.any(), "At least one cross-conversation boundary is needed"
        assert not same_doc[cross_doc].any()


# ---------------------------------------------------------------------------
# Hardening: edge cases that surfaced during code review
# ---------------------------------------------------------------------------


class TestResolveFPConfigEdgeCases:
    """Beyond the happy-path cases covered in ``TestResolveFPConfigFromMeta``:
    the helper must tolerate missing/empty meta dicts without crashing so a
    broken checkpoint produces a clean ``None`` rather than a TypeError
    traceback deep inside ``main()``.
    """

    def test_empty_meta_returns_none(self) -> None:
        """A totally empty ``meta`` dict (no ``user_config`` key at all)
        must return ``None`` — this is the "baseline pretrain checkpoint
        from before ``user_config`` was standardised" scenario."""
        assert _resolve_fp_config_from_meta({}) is None

    def test_user_config_missing_but_sft_key_present(self) -> None:
        """Chained-SFT meta doesn't need a ``user_config`` key at all —
        the fallback path hits ``feature_prediction_config`` directly."""
        meta = {
            "feature_prediction_config": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "best_token",
                "loss_fn": "bce",
            }
        }
        cfg = _resolve_fp_config_from_meta(meta)
        assert cfg is not None
        assert cfg.weight == 0.1

    def test_invalid_loss_mode_raises_value_error(self) -> None:
        """An unknown ``loss_mode`` string must surface as ``ValueError``
        at checkpoint load rather than silently constructing a config
        with a stringly-typed value. The enum constructor already
        enforces this — this test pins that the helper doesn't swallow
        the exception."""
        meta = {
            "user_config": {
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.1,
                    "layer": -1,
                    "loss_mode": "this_mode_does_not_exist",
                    "loss_fn": "bce",
                }
            }
        }
        with pytest.raises(ValueError, match="this_mode_does_not_exist"):
            _resolve_fp_config_from_meta(meta)

    def test_invalid_loss_fn_raises_value_error(self) -> None:
        meta = {
            "user_config": {
                "feature_prediction": {
                    "block": "mlp",
                    "head_layers": 0,
                    "weight": 0.1,
                    "layer": -1,
                    "loss_mode": "best_token",
                    "loss_fn": "made_up_loss",
                }
            }
        }
        with pytest.raises(ValueError, match="made_up_loss"):
            _resolve_fp_config_from_meta(meta)


class TestFPTargetsGateDependsOnBothKwargs:
    """The ``fp_targets`` build inside ``make_sft_data_generator`` is gated
    on BOTH ``n_sae_features is not None`` AND ``fp_percentile_table is not
    None``. Either one alone must NOT cause the scatter to fire (it would
    either crash with a shape error or produce silently-wrong targets).
    """

    def _make_gen_with_kwargs(
        self,
        *,
        n_sae_features: int | None,
        fp_percentile_table: t.Tensor | None,
    ):
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        boundaries = (
            t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
            .unsqueeze(0)
            .expand(N_SAE_FEATURES, -1)
            .contiguous()
        )
        threshold_cfg = ThresholdConfig(
            threshold_type=ThresholdType.segment_topk,
            threshold=1000,
        )
        with patch_load_sft_features(features):
            gen = make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=SFTState(),
                feature_dir="/dummy",
                parquet_dir="/dummy",
                threshold_cfg=threshold_cfg,
                boundaries=boundaries,
                n_sae_features=n_sae_features,
                fp_percentile_table=fp_percentile_table,
                min_loss_tokens=1,
                alignment_probe_samples=0,
                expected_split=SFTSplit.TRAIN,
            )
            return next(gen)

    def test_n_sae_features_alone_does_not_produce_fp_targets(self) -> None:
        """``n_sae_features`` without a table has no table to read
        percentiles from — the dataloader must NOT allocate fp_targets."""
        batch = self._make_gen_with_kwargs(
            n_sae_features=N_SAE_FEATURES,
            fp_percentile_table=None,
        )
        assert batch[-1] is None

    def test_fp_percentile_table_alone_does_not_produce_fp_targets(self) -> None:
        """A table without ``n_sae_features`` has no dense-target width
        to scatter into — the dataloader must NOT allocate fp_targets."""
        table = _make_synthetic_percentile_table()
        batch = self._make_gen_with_kwargs(
            n_sae_features=None,
            fp_percentile_table=table,
        )
        assert batch[-1] is None

    def test_both_together_produce_fp_targets(self) -> None:
        """Control: both present ⇒ dense fp_targets plus seg_doc_id."""
        table = _make_synthetic_percentile_table()
        batch = self._make_gen_with_kwargs(
            n_sae_features=N_SAE_FEATURES,
            fp_percentile_table=table,
        )
        assert len(batch) == 8
        fp_targets = batch[-2]
        seg_doc_id = batch[-1]
        assert fp_targets is not None
        assert seg_doc_id.shape == fp_targets.shape[:2]


class TestFPTargetsForMultiTurnConversations:
    """Full-sequence SFT supervises and conditions every non-padding turn."""

    def test_multi_turn_fp_targets_align_with_full_sequence_segments(self) -> None:
        """Every supervised full-sequence segment should carry FP targets."""
        from tests.steering.test_encdec_sft_conditioned_integration import MultiTurnTask

        tokenizer = SFTMockTokenizer()
        task = MultiTurnTask(n=4, user_chars=5, asst_chars=16)
        features = build_synthetic_features(task, tokenizer, features_per_segment=3)

        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(
                task,
                tokenizer,
                B=1,
                T=48,
            )
            batch = next(gen)

        assert len(batch) == 8
        _inp, targets, _fids, _mags, _mask, seg_map, fp_targets, seg_doc_id = batch
        assert fp_targets is not None
        assert seg_doc_id.shape == fp_targets.shape[:2]
        # Segment 0 is reserved empty — fp_targets[:, 0, :] must stay zero
        assert fp_targets[0, 0, :].sum().item() == 0.0, (
            "Segment 0 is reserved empty; fp_targets at seg 0 must remain zero"
        )
        supervised = targets[0] != -1
        assert supervised.any()
        assert (seg_map[0][supervised] > 0).all()
        for seg_i in seg_map[0][supervised].unique().tolist():
            assert fp_targets[0, seg_i, :].sum().item() > 0.0, (
                f"Full-sequence segment {seg_i} should carry fp targets"
            )


class TestFPTargetsForPredictionTargetOnlySegments:
    """When a conversation is truncated and has a trailing
    prediction-target-only segment, that segment must reach fp_targets
    even though it has no decoder positions in seg_map."""

    def test_prediction_target_extra_segment_reaches_fp_targets(self) -> None:
        """Uses a conversation long enough to trigger truncation,
        asserts that the extra prediction-target segment's features
        reach fp_targets (at a segment index NOT pointed to by
        seg_map)."""
        tokenizer = SFTMockTokenizer()
        # Make assistant text long enough to force truncation at T=32.
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=200)
        # Build features covering the FULL assistant text in 2 segments
        # so one gets clipped to the extra prediction-target slot.
        features = build_synthetic_features(
            task,
            tokenizer,
            n_segments_per_conv=2,
            features_per_segment=3,
        )

        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=1, T=32, return_seg_doc_id=True)
            batch = next(gen)

        _inp, _tgt, _fids, _mags, _mask, seg_map, fp_targets, seg_doc_id = batch
        assert fp_targets is not None
        # There must be at least one fp_targets row (segment index) that
        # is non-zero but doesn't correspond to any seg_map position —
        # that's the prediction-target-only extra segment.
        seg_map_values = set(seg_map[0].unique().tolist())
        nonzero_fp_segs = set()
        for s in range(fp_targets.shape[1]):
            if fp_targets[0, s, :].sum().item() > 0.0:
                nonzero_fp_segs.add(s)
        extra_only = nonzero_fp_segs - seg_map_values
        assert len(extra_only) == 1, (
            "The forced-truncation fixture should produce exactly one "
            f"prediction-target-only segment; got {extra_only}"
        )
        extra_seg_i = extra_only.pop()
        assert extra_seg_i > 0
        assert fp_targets[0, extra_seg_i, :].sum().item() > 0.0
        assert int(seg_doc_id[0, extra_seg_i].item()) >= 0
        assert seg_doc_id[0, extra_seg_i].item() == seg_doc_id[0, extra_seg_i - 1].item(), (
            "The prediction-target-only extra segment must keep the same local "
            "conversation id as the in-range segment that predicts it"
        )


class TestTruncateSegFeaturesPreservesAllFeatureFields:
    """``_truncate_seg_features`` uses ``dataclasses.replace`` to build the
    prediction-target-only segment and to clip the last in-range segment.
    Both must preserve the new ``all_feature_ids`` / ``all_feature_pcts``
    fields introduced in this PR — otherwise fp_targets silently loses
    data for truncated conversations."""

    def test_all_feature_fields_preserved_across_truncation(self) -> None:
        from scripts.train.conditional_encdec_sft import _truncate_seg_features
        from steering.conditional.encoder_decoder.dataloader import SegFeatures

        sf1 = SegFeatures(
            feature_ids=t.tensor([1, 2, 3], dtype=t.int64),
            buckets=t.tensor([1, 2, 3], dtype=t.int64),
            percentiles=t.zeros(3, dtype=t.float32),
            char_start=0,
            char_end=100,
            all_feature_ids=t.tensor([10, 11, 12], dtype=t.int64),
            all_feature_pcts=t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
        )
        sf2 = SegFeatures(
            feature_ids=t.tensor([4, 5, 6], dtype=t.int64),
            buckets=t.tensor([4, 5, 6], dtype=t.int64),
            percentiles=t.zeros(3, dtype=t.float32),
            char_start=100,
            char_end=200,
            all_feature_ids=t.tensor([20, 21, 22], dtype=t.int64),
            all_feature_pcts=t.tensor([0.4, 0.5, 0.6], dtype=t.float32),
        )
        sf3 = SegFeatures(
            feature_ids=t.tensor([7, 8, 9], dtype=t.int64),
            buckets=t.tensor([7, 8, 9], dtype=t.int64),
            percentiles=t.zeros(3, dtype=t.float32),
            char_start=200,
            char_end=300,
            all_feature_ids=t.tensor([30, 31, 32], dtype=t.int64),
            all_feature_pcts=t.tensor([0.7, 0.8, 0.9], dtype=t.float32),
        )

        # Truncate to 150 chars: sf1 in-range (unchanged), sf2 in-range
        # and LAST in-range (so char_end clipped to 150), sf3 extra
        # prediction-target.
        result = _truncate_seg_features([sf1, sf2, sf3], total_chars=150)
        assert result is not None
        assert len(result) == 3  # sf1 + sf2_clipped + sf3_extra

        def _assert_all_feature_fields(
            seg,
            expected_fids: list[int],
            expected_pcts: list[float],
        ) -> None:
            """Type-narrowing helper: asserts both fields are non-None
            (so ``t.equal`` gets ``Tensor`` not ``Tensor | None``) and
            compares element-wise."""
            fids = seg.all_feature_ids
            pcts = seg.all_feature_pcts
            assert fids is not None
            assert pcts is not None
            assert t.equal(fids, t.tensor(expected_fids, dtype=t.int64))
            assert t.equal(pcts, t.tensor(expected_pcts, dtype=t.float32))

        # sf1 unclipped: all-feature fields must survive
        unclipped = result[0]
        assert unclipped.char_end == 100
        _assert_all_feature_fields(unclipped, [10, 11, 12], [0.1, 0.2, 0.3])

        # sf2 clipped: ``replace(last, char_end=total_chars)`` must
        # preserve the all_feature fields.
        clipped = result[1]
        assert clipped.char_end == 150
        _assert_all_feature_fields(clipped, [20, 21, 22], [0.4, 0.5, 0.6])

        # sf3 extra: ``replace(seg, prediction_target_only=True)`` must
        # preserve the all_feature fields.
        extra = result[2]
        assert extra.prediction_target_only is True
        _assert_all_feature_fields(extra, [30, 31, 32], [0.7, 0.8, 0.9])


class TestPaddedSegmentsStayZeroInFPTargets:
    """When a batch contains rows with different numbers of segments, the
    dense fp_targets tensor is padded to the batch max along the n_seg
    axis. Those padding slots must stay zero so they don't contribute
    spurious signal to the fp loss (which masks via seg_doc_id anyway,
    but defense-in-depth)."""

    def test_padding_segments_have_zero_fp_targets(self) -> None:
        """Build a 2-row batch where one row has more segments than the
        other. The shorter row's padded slots must be all-zero."""
        tokenizer = SFTMockTokenizer()
        task_short = SingleTurnTask(n=2, user_chars=3, asst_chars=6)
        # Same task but different lengths would trip the packing —
        # just use one task type. Build features for every example.
        features = build_synthetic_features(
            task_short,
            tokenizer,
            features_per_segment=3,
        )

        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task_short, tokenizer, B=2, T=64)
            batch = next(gen)

        assert len(batch) == 8
        _inp, _tgt, enc_fids, _mags, enc_mask, _seg, fp_targets, seg_doc_id = batch
        assert fp_targets is not None
        assert seg_doc_id.shape == fp_targets.shape[:2]
        B, batch_n_seg, _ = fp_targets.shape
        # For each row, segments beyond the number of *real* encoder
        # segments (where enc_mask is all-False) must have zero
        # fp_targets.
        for row_idx in range(B):
            for seg_i in range(batch_n_seg):
                if not enc_mask[row_idx, seg_i, :].any():
                    # This is a padded slot — fp_targets must be zero.
                    assert fp_targets[row_idx, seg_i, :].sum().item() == 0.0, (
                        f"Padded segment (row {row_idx}, seg {seg_i}) must have zero fp_targets"
                    )


@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
class TestFPTargetsDevicePlacement:
    """When the data generator is given ``device='cuda'``, the ``fp_targets``
    tensor must land on the CUDA device — not CPU. This was the cause of
    a device-mismatch crash during the real-model 4090 smoke."""

    def test_fp_targets_on_cuda_when_device_is_cuda(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=8)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            gen = _make_gen_with_fp(task, tokenizer, B=2, T=64, device="cuda")
            batch = next(gen)
        fp_targets = _fp_targets_from_batch(batch)
        assert fp_targets is not None
        assert fp_targets.device.type == "cuda", (
            f"fp_targets must live on CUDA; got device={fp_targets.device}"
        )
        # Other tensors must also be on CUDA.
        for i, tsr in enumerate(batch[:6]):
            if isinstance(tsr, t.Tensor):
                assert tsr.device.type == "cuda", f"batch[{i}] must be on CUDA; got {tsr.device}"


class TestSaveCheckpointIncludesFPConfig:
    """``main()`` saves ``feature_prediction_config`` in the checkpoint meta
    when fp is active, so a subsequent SFT (``resume_from_step`` or a
    chained SFT) can reconstruct the fp_head without needing the original
    pretrain checkpoint.  We pin the dict schema here by constructing a
    fp_config, serialising the way ``save_checkpoint`` does, and
    verifying ``_resolve_fp_config_from_meta`` round-trips it.
    """

    def test_save_then_resolve_roundtrip(self) -> None:
        """Schema round-trip via ``_resolve_fp_config_from_meta``:
        encode a config the way main() does, feed the resulting meta
        back through the resolver, and confirm every field matches."""
        fp_config = FeaturePredictionConfig(
            weight=0.42,
            layer=-2,
            loss_mode=FeaturePredLossMode.FULL_SEGMENT,
            loss_fn=FeaturePredLossFn.MSE,
        )
        saved_dict = {
            "block": "mlp",
            "head_layers": 0,
            "weight": fp_config.weight,
            "layer": fp_config.layer,
            "loss_mode": fp_config.loss_mode.value,
            "loss_fn": fp_config.loss_fn.value,
        }
        # Simulate what main() writes to ``meta_XXXXXX.json``.
        meta = {"feature_prediction_config": saved_dict, "user_config": {}}
        recovered = _resolve_fp_config_from_meta(meta)
        assert recovered is not None
        assert recovered.weight == fp_config.weight
        assert recovered.layer == fp_config.layer
        assert recovered.loss_mode is fp_config.loss_mode
        assert recovered.loss_fn is fp_config.loss_fn

    def test_json_roundtrip_is_lossless(self) -> None:
        """The saved dict must survive a JSON round-trip (which is
        what ``save_checkpoint``'s ``json.dump`` does) without loss
        of precision or type."""
        fp_config = FeaturePredictionConfig(
            weight=0.1337,
            layer=5,
            loss_mode=FeaturePredLossMode.BEST_TOKEN,
            loss_fn=FeaturePredLossFn.BCE,
        )
        saved_dict = {
            "block": "mlp",
            "head_layers": 0,
            "weight": fp_config.weight,
            "layer": fp_config.layer,
            "loss_mode": fp_config.loss_mode.value,
            "loss_fn": fp_config.loss_fn.value,
        }
        meta = {"feature_prediction_config": saved_dict, "user_config": {}}
        # Explicit JSON round-trip
        roundtripped_meta = json.loads(json.dumps(meta))
        recovered = _resolve_fp_config_from_meta(roundtripped_meta)
        assert recovered is not None
        assert recovered.weight == fp_config.weight
        assert recovered.layer == fp_config.layer
        assert recovered.loss_mode is fp_config.loss_mode
        assert recovered.loss_fn is fp_config.loss_fn

    def test_sft_save_meta_carries_user_config_and_top_level_fp_config(self) -> None:
        """SFT checkpoints must satisfy both SFT chaining and checkpoint_manager."""
        fp_config = FeaturePredictionConfig(
            weight=0.2,
            layer=-1,
            loss_mode=FeaturePredLossMode.BEST_TOKEN,
            loss_fn=FeaturePredLossFn.BCE,
        )
        saved_dict = _serialize_fp_config(fp_config)
        user_config = {"dataset": "safety"}
        checkpoint_user_config = _checkpoint_user_config_with_fp(user_config, fp_config)
        meta = {
            "user_config": checkpoint_user_config,
            "feature_prediction_config": saved_dict,
        }

        assert meta["user_config"]["feature_prediction"] == saved_dict
        assert meta["feature_prediction_config"] == saved_dict
        assert _resolve_fp_config_from_meta(meta) is not None

    def test_sft_checkpoint_with_user_config_fp_loads_via_checkpoint_manager(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        fp_config = _make_fp_config()
        model = build_tiny_model(
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            feature_pred_config=fp_config,
            n_layer=2,
            n_embd=64,
            seq_len=64,
            xattn_frequency=1,
        )
        fp_dict = _serialize_fp_config(fp_config)
        assert fp_dict is not None
        save_checkpoint(
            str(tmp_path),
            1,
            model.state_dict(),
            None,
            {
                "step": 1,
                "model_config": asdict(model.dec_config),
                "user_config": {
                    "num_buckets": NUM_BUCKETS,
                    "feature_prediction": fp_dict,
                },
                "encoder_config": asdict(model.enc_config),
                "xattn_config": asdict(model.xattn_config),
                "num_buckets": NUM_BUCKETS,
                "feature_prediction_config": fp_dict,
            },
            rank=0,
        )

        class _Tokenizer:
            def get_vocab_size(self) -> int:
                return model.dec_config.vocab_size

        import nanochat.checkpoint_manager as checkpoint_manager

        monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _Tokenizer())
        loaded, _tokenizer, _meta = build_encdec_model(
            str(tmp_path),
            1,
            t.device("cpu"),
            ModelPhase.TRAIN,
            eval_conditioning=EvalConditioning.not_evaluating(),
        )
        assert loaded.feature_pred_head is not None


# ---------------------------------------------------------------------------
# E2E: main() subprocess smoke test
# ---------------------------------------------------------------------------


def _build_pretrain_checkpoint(
    tmp_path: Path,
    *,
    with_fp: bool,
    n_layer: int = 2,
    n_embd: int = 64,
    seq_len: int = 64,
    n_sae_features: int = 64,
    num_buckets: int = 4,
) -> tuple[Path, str]:
    """Write a tiny pretrain-style checkpoint to disk.

    Returns ``(checkpoint_dir, model_tag)``. The directory layout matches
    what ``main()`` expects: ``conditional_encdec_checkpoints/{tag}/s{seg}/{fs}/``.
    """
    tag = "test_fp" if with_fp else "test_baseline"
    seg_size = 32
    fs_dirname = "dist_geometric/mean_5.0/std_5.0"
    checkpoint_dir = tmp_path / "conditional_encdec_checkpoints" / tag / f"s{seg_size}" / fs_dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fp_config = (
        _make_fp_config(weight=0.1, layer=-1, loss_mode=FeaturePredLossMode.FULL_SEGMENT)
        if with_fp
        else None
    )
    model = build_tiny_model(
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        feature_pred_config=fp_config,
        n_layer=n_layer,
        n_embd=n_embd,
        seq_len=seq_len,
        xattn_frequency=1,
    )

    user_config = {
        "encoder": {
            "n_embd": n_embd // 2,
            "depth": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "head_dim": 16,
            "ve_gate_channels": 32,
        },
        "cross_attention": {"frequency": 1, "n_head": 2, "n_kv_head": 2},
        "num_buckets": num_buckets,
        "feature_prediction": (
            {
                "block": "mlp",
                "head_layers": 0,
                "weight": fp_config.weight,
                "layer": fp_config.layer,
                "loss_mode": fp_config.loss_mode.value,
                "loss_fn": fp_config.loss_fn.value,
            }
            if fp_config
            else None
        ),
    }
    meta = {
        "step": 100,
        "model_config": asdict(model.dec_config),
        "user_config": user_config,
    }
    save_checkpoint(
        str(checkpoint_dir),
        100,
        model.state_dict(),
        None,
        meta,
        rank=0,
    )
    return checkpoint_dir, tag


class TestMainEndToEnd:
    """Subprocess smoke tests for the full ``main()`` entry point.

    These tests do NOT call ``main()`` directly because Hydra's
    ``initialize_config_dir`` is process-global and conflicts with the
    pytest test runner's own Hydra usage in other test files. Subprocess
    isolation also catches import-order bugs that wouldn't surface in
    a same-process test.
    """

    def _run_sft(
        self,
        tmp_path: Path,
        tag: str,
        extra_args: list[str],
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["NANOCHAT_BASE_DIR"] = str(tmp_path)
        env["WANDB_MODE"] = "disabled"
        if env_overrides:
            env.update(env_overrides)
        repo_root = Path(__file__).resolve().parent.parent.parent
        cmd = [
            sys.executable,
            "-m",
            "scripts.train.conditional_encdec_sft",
            f"model_tag={tag}",
            "model_step=100",
            "device_type=cpu",
            "device_batch_size=2",
            "max_seq_len=64",
            "total_batch_size=128",
            "num_iterations=2",
            "eval_every=0",
            "save_every=2",
            "run=dummy",
            "grad_clip_max_norm=1.0",
            *extra_args,
        ]
        return subprocess.run(
            cmd,
            env=env,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

    @pytest.mark.slow
    def test_loads_fp_pretrain_checkpoint_and_runs_one_step(self, tmp_path: Path) -> None:
        """Loading a pretrain checkpoint with fp head must succeed
        (strict load), and ``main()`` must complete at least one step
        without crashing.
        """
        checkpoint_dir, tag = _build_pretrain_checkpoint(tmp_path, with_fp=True)
        from nanochat.common import get_base_dir

        if not (Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl").exists():
            pytest.skip("nanochat tokenizer pickle missing")
        result = self._run_sft(tmp_path, tag, [])
        assert result.returncode == 0, (
            f"main() crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    @pytest.mark.slow
    def test_baseline_pretrain_checkpoint_unaffected(self, tmp_path: Path) -> None:
        """Backward-compat: a checkpoint without fp head must still run
        unchanged. This pins that the new code path never accidentally
        builds fp_head when meta lacks the config."""
        checkpoint_dir, tag = _build_pretrain_checkpoint(tmp_path, with_fp=False)
        from nanochat.common import get_base_dir

        if not (Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl").exists():
            pytest.skip("nanochat tokenizer pickle missing")
        result = self._run_sft(tmp_path, tag, [])
        assert result.returncode == 0, (
            f"baseline main() crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
