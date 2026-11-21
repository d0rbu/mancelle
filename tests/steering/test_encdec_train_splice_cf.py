"""D8 startup behavior + D12 live dual validation for splice_counterfactual (issue #596)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest
import torch as t
from loguru import logger
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from nanochat.dataset import DatasetSplit
from scripts.train.conditional_encdec_train import (
    DualValidationMode,
    _assert_live_val_pair_aligned,
    _build_delta_free_val_pipeline,
    _build_validation_eval_specs,
    _dual_validation_mode,
    _require_splice_cf_dual_val,
    _resolve_splice_counterfactual_data_dir,
    _splice_counterfactual_enabled,
    train_one_config,
)
from steering.conditional.encoder_decoder.dataloader import (
    encoder_decoder_data_loader_with_state,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_pipeline import (
    SamplingBundle,
    build_feature_pipeline_from_cfg_block,
)
from steering.stats_types import PerFeatureStat
from tests.steering._splice_cf_fixtures import build_derived_env


def _cfg(feature_delta=None, data_dir=None, **extra):
    base = {"feature_delta": feature_delta, "data_dir": data_dir}
    base.update(extra)
    return OmegaConf.create(base)


def _derived(tmp_path, *, with_prov=True, with_parquets=True):
    d = tmp_path / "derived"
    d.mkdir()
    if with_prov:
        (d / "_splice_provenance").mkdir()
        (d / "_splice_provenance" / "shard_00000.json").write_text("{}")
    if with_parquets:
        (d / "_parquets").mkdir()
    return d


class TestEnabledPredicate:
    def test_none_disabled(self) -> None:
        assert not _splice_counterfactual_enabled(_cfg(None))

    def test_other_mode_disabled(self) -> None:
        assert not _splice_counterfactual_enabled(_cfg({"mode": "prev_segment"}))

    def test_enabled(self) -> None:
        assert _splice_counterfactual_enabled(_cfg({"mode": "splice_counterfactual"}))


class TestResolveDataDir:
    def test_noop_when_mode_off(self, tmp_path) -> None:
        cfg = _cfg({"mode": "prev_segment"})
        _resolve_splice_counterfactual_data_dir(cfg, str(tmp_path))
        assert cfg.data_dir is None

    def test_auto_derives_canonical(self, tmp_path) -> None:
        d = _derived(tmp_path)
        cfg = _cfg({"mode": "splice_counterfactual"})
        _resolve_splice_counterfactual_data_dir(cfg, str(d))
        assert cfg.data_dir == str(d / "_parquets")

    def test_missing_provenance_dir_raises(self, tmp_path) -> None:
        d = _derived(tmp_path, with_prov=False)
        cfg = _cfg({"mode": "splice_counterfactual"})
        with pytest.raises(ValueError, match="_splice_provenance"):
            _resolve_splice_counterfactual_data_dir(cfg, str(d))

    def test_missing_parquets_dir_raises_on_autoderive(self, tmp_path) -> None:
        d = _derived(tmp_path, with_parquets=False)
        cfg = _cfg({"mode": "splice_counterfactual"})
        with pytest.raises(ValueError, match="_parquets"):
            _resolve_splice_counterfactual_data_dir(cfg, str(d))

    def test_explicit_canonical_respected(self, tmp_path) -> None:
        d = _derived(tmp_path)
        cfg = _cfg({"mode": "splice_counterfactual"}, data_dir=str(d / "_parquets"))
        _resolve_splice_counterfactual_data_dir(cfg, str(d))
        assert cfg.data_dir == str(d / "_parquets")

    def test_explicit_noncanonical_warns_and_respected(self, tmp_path) -> None:
        d = _derived(tmp_path)
        other = tmp_path / "elsewhere"
        other.mkdir()
        cfg = _cfg({"mode": "splice_counterfactual"}, data_dir=str(other))
        records = []
        hid = logger.add(lambda m: records.append(m), level="WARNING")
        try:
            _resolve_splice_counterfactual_data_dir(cfg, str(d))
        finally:
            logger.remove(hid)
        assert cfg.data_dir == str(other)
        assert any("canonical" in str(r) for r in records)


class TestLiveLoaderDataDirThreading:
    def test_train_live_loader_call_passes_data_dir(self) -> None:
        src = Path("scripts/train/conditional_encdec_train.py").read_text()
        assert 'data_dir=cfg.get("data_dir")' in src

    def test_train_one_config_reapplies_splice_data_dir_derivation(self) -> None:
        src = inspect.getsource(train_one_config)
        derive_idx = src.index("_resolve_splice_counterfactual_data_dir(cfg, feature_dir)")
        snapshot_idx = src.index("user_config =")
        pipeline_idx = src.index("build_feature_pipeline_from_cfg_block")
        assert derive_idx < snapshot_idx
        assert derive_idx < pipeline_idx


class TestDualValidationMode:
    def test_none_when_no_original_dir(self) -> None:
        assert _dual_validation_mode(
            original_val_feature_dir=None,
            splice_cf_enabled=False,
        ) is DualValidationMode.none

    def test_precomputed_for_splice_cf_defaults_to_packed(self, tmp_path) -> None:
        assert _dual_validation_mode(
            original_val_feature_dir=tmp_path,
            splice_cf_enabled=True,
        ) is DualValidationMode.packed

    def test_live_for_splice_cf_when_precomputed_disabled(self, tmp_path) -> None:
        assert _dual_validation_mode(
            original_val_feature_dir=tmp_path,
            splice_cf_enabled=True,
            use_precomputed=False,
        ) is DualValidationMode.live

    def test_packed_for_no_delta(self, tmp_path) -> None:
        assert _dual_validation_mode(
            original_val_feature_dir=tmp_path,
            splice_cf_enabled=False,
        ) is DualValidationMode.packed

    def test_packed_path_for_other_delta_modes(self, tmp_path) -> None:
        """prev/cum + dual val stays on the packed path, where #606's existing
        feature_delta raise fires unchanged (asserted at the build level, not
        here — this gate just must NOT divert them to live)."""
        assert _dual_validation_mode(
            original_val_feature_dir=tmp_path,
            splice_cf_enabled=False,
        ) is DualValidationMode.packed


def _stats(n=8):
    percentiles = {f"p{k}": 0.01 * k for k in range(1, 100)}
    return {
        str(i): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(percentiles),
        )
        for i in range(n)
    }


_SAMPLING_CF_CFG = OmegaConf.create({
    "feature_selection": {
        "distribution": "geometric", "mean": 3.0, "std": 3.0,
        "min_k": 1, "max_k": 10, "delta_weights": "percentile_delta",
        "mode": "weighted",
    },
    "negative_selection": {
        "distribution": "geometric", "mean": 2.0, "std": 2.0,
        "min_k": 0, "max_k": 5,
    },
    "feature_delta": {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
    "num_buckets": 4,
})


class TestDeltaFreeValPipeline:
    def test_strips_delta_and_delta_weights(self) -> None:
        pipe = _build_delta_free_val_pipeline(
            _SAMPLING_CF_CFG, per_feature_stats=_stats(), n_sae_features=8,
            num_buckets=4, seed=42, feature_mask=None,
        )
        assert pipe.delta_bundle is None
        assert isinstance(pipe.selection_config, SamplingBundle)
        assert pipe.selection_config.selection.delta_weights is None
        # Other selection knobs preserved
        assert pipe.selection_config.selection.mean == 3.0
        assert pipe.negative_bundle is not None

    def test_threshold_selection_passthrough(self) -> None:
        cfg = OmegaConf.create({
            "feature_selection": {"threshold_type": "segment_topk", "threshold": 2},
            "feature_delta": {"mode": "splice_counterfactual"},
            "num_buckets": 4,
        })
        pipe = _build_delta_free_val_pipeline(
            cfg, per_feature_stats=_stats(), n_sae_features=8,
            num_buckets=4, seed=42, feature_mask=None,
        )
        assert pipe.delta_bundle is None
        assert pipe.negative_bundle is None


class TestBuildValidationEvalSpecsLiveDual:
    def test_live_dual_returns_base_then_spliced(self, tmp_path) -> None:
        fake_index = object()
        fake_spliced_index = object()
        fake_pipeline = object()
        specs = _build_validation_eval_specs(
            packed_row_dir=None,
            active_val_packed_row_dir=None,
            active_val_selection_dir=None,
            original_val_packed_row_dir=None,
            original_val_selection_dir=None,
            live_dual=True,
            original_val_feature_index=fake_index,
            spliced_val_feature_index=fake_spliced_index,
            original_val_data_dir=str(tmp_path / "base_data"),
            spliced_data_dir=str(tmp_path / "derived" / "_parquets"),
            delta_free_pipeline=fake_pipeline,
        )
        assert [s.metric_prefix for s in specs] == ["val", "val_spliced"]
        base, spliced = specs
        assert base.packed_row_dir is None and spliced.packed_row_dir is None
        assert base.feature_index is fake_index
        assert base.pipeline is fake_pipeline
        assert base.data_dir == str(tmp_path / "base_data")
        # val_spliced uses a DEDICATED derived-dir index (never shared.feature_index,
        # which the train prefetch thread mutates concurrently — thread-safety).
        assert spliced.feature_index is fake_spliced_index
        assert spliced.pipeline is None            # ⇒ main feature_pipeline
        assert spliced.data_dir == str(tmp_path / "derived" / "_parquets")

    def test_non_dual_shapes_unchanged(self) -> None:
        specs = _build_validation_eval_specs(
            packed_row_dir=None,
            active_val_packed_row_dir=None,
            active_val_selection_dir=None,
            original_val_packed_row_dir=None,
            original_val_selection_dir=None,
        )
        assert [s.metric_prefix for s in specs] == ["val"]
        assert specs[0].data_dir is None and specs[0].feature_index is None

    def test_non_dual_live_val_uses_configured_data_dir(self, tmp_path) -> None:
        data_dir = str(tmp_path / "custom_parquets")
        specs = _build_validation_eval_specs(
            packed_row_dir=None,
            active_val_packed_row_dir=None,
            active_val_selection_dir=None,
            original_val_packed_row_dir=None,
            original_val_selection_dir=None,
            live_data_dir=data_dir,
        )
        assert len(specs) == 1
        assert specs[0].metric_prefix == "val"
        assert specs[0].packed_row_dir is None
        assert specs[0].data_dir == data_dir

    def test_non_dual_packed_val_ignores_live_data_dir(self, tmp_path) -> None:
        packed = tmp_path / "packed"
        specs = _build_validation_eval_specs(
            packed_row_dir=tmp_path / "train_packed",
            active_val_packed_row_dir=packed,
            active_val_selection_dir=None,
            original_val_packed_row_dir=None,
            original_val_selection_dir=None,
            live_data_dir=str(tmp_path / "custom_parquets"),
        )
        assert len(specs) == 1
        assert specs[0].packed_row_dir == packed
        assert specs[0].data_dir is None

    def test_packed_dual_base_val_uses_delta_free_pipeline(self, tmp_path) -> None:
        fake_pipeline = object()
        specs = _build_validation_eval_specs(
            packed_row_dir=tmp_path / "train_packed",
            active_val_packed_row_dir=tmp_path / "derived_val_packed",
            active_val_selection_dir=tmp_path / "derived_sel",
            original_val_packed_row_dir=tmp_path / "base_val_packed",
            original_val_selection_dir=tmp_path / "base_sel",
            delta_free_pipeline=fake_pipeline,
        )
        assert [s.metric_prefix for s in specs] == ["val", "val_spliced"]
        base, spliced = specs
        assert base.packed_row_dir == tmp_path / "base_val_packed"
        assert base.selection_dir == tmp_path / "base_sel"
        assert base.pipeline is fake_pipeline
        assert spliced.packed_row_dir == tmp_path / "derived_val_packed"
        assert spliced.selection_dir == tmp_path / "derived_sel"
        assert spliced.pipeline is None


class TestPackedRaiseUnchanged:
    def test_606_delta_raise_text_still_present(self) -> None:
        """#606's packed-path raise for other delta modes must survive verbatim."""
        src = Path("scripts/train/conditional_encdec_train.py").read_text()
        assert "feature_delta is live-dataloader-only in conditional_encdec_train" in src


class TestBaseValPassIsDeltaFree:
    def test_delta_free_twin_equals_manual_no_delta_pipeline(self, tmp_path) -> None:
        """Stream-level: a live VAL loader over the BASE dir driven by the
        delta-free twin yields t.equal batches to one driven by a manually
        built no-delta pipeline (same seeds) — proving the twin builder
        introduces no delta machinery."""
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base_dir = str(Path(env["feature_dir"]).parents[2])  # .../s2/base

        def _run(pipe):
            fi = FeatureIndex(base_dir)
            gen = encoder_decoder_data_loader_with_state(
                tokenizer=env["tokenizer"], B=2, T=32, split=DatasetSplit.VAL,
                feature_index=fi, pipeline=pipe, segment_size=env["seg_size"],
                decoder_segment_size=None, device="cpu", resume_state_dict=None,
                tokenizer_batch_size=16, buffer_size=16, data_dir=base_dir,
                prefetch_batches=0, label_dropout=0.0,
            )
            return [next(gen) for _ in range(2)]

        twin = _build_delta_free_val_pipeline(
            _SAMPLING_CF_CFG, per_feature_stats=_stats(4096), n_sae_features=4096,
            num_buckets=4, seed=42, feature_mask=None,
        )
        sampling_raw = cast(
            dict[str, Any],
            OmegaConf.to_container(_SAMPLING_CF_CFG, resolve=True),
        )
        cfg_no_delta = OmegaConf.create({**sampling_raw, "feature_delta": None})
        cfg_no_delta.feature_selection.delta_weights = None
        manual = build_feature_pipeline_from_cfg_block(
            cfg_no_delta, per_feature_stats=_stats(4096), n_sae_features=4096,
            num_buckets=4, seed=42, ddp_rank=0, feature_delta=None,
            feature_mask=None,
        )
        for a, b in zip(_run(twin), _run(manual), strict=True):
            for x, y in zip(a, b, strict=True):
                if isinstance(x, t.Tensor):
                    assert t.equal(x, y)
                else:
                    assert x == y


class TestRequireDualVal:
    """[2] splice_counterfactual must ALWAYS run dual validation — raise when the
    base/original feature dir cannot be resolved (review finding [2])."""

    def test_splice_cf_without_base_raises(self) -> None:
        cfg = _cfg({"mode": "splice_counterfactual"})
        with pytest.raises(ValueError, match="requires dual validation"):
            _require_splice_cf_dual_val(cfg, "/some/derived", None)

    def test_splice_cf_with_base_ok(self) -> None:
        cfg = _cfg({"mode": "splice_counterfactual"})
        _require_splice_cf_dual_val(cfg, "/derived", "/base")  # no raise

    def test_other_delta_mode_without_base_ok(self) -> None:
        cfg = _cfg({"mode": "prev_segment"})
        _require_splice_cf_dual_val(cfg, "/derived", None)  # no raise

    def test_no_delta_without_base_ok(self) -> None:
        cfg = _cfg(None)
        _require_splice_cf_dual_val(cfg, "/derived", None)  # no raise


class TestBaseValPairGuard:
    """[1] the base/original live-val pass fails fast at startup when its corpus
    does not pair with its feature dir (review finding [1])."""

    def test_aligned_pair_passes(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base_dir = str(Path(env["feature_dir"]).parents[2])  # .../s2/base
        fi = FeatureIndex(base_dir)
        _assert_live_val_pair_aligned(fi, base_dir, None, "base val")

    def test_stem_mismatch_raises(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base_dir = str(Path(env["feature_dir"]).parents[2])
        fi = FeatureIndex(base_dir)
        wrong = tmp_path / "wrongstem"
        wrong.mkdir()
        atomic_write_parquet(pa.table({"text": ["x", "y"]}), wrong / "shard_99999.parquet")
        with pytest.raises(ValueError, match="no matching feature shard"):
            _assert_live_val_pair_aligned(fi, str(wrong), None, "base val")

    def test_row_count_mismatch_raises(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base_dir = str(Path(env["feature_dir"]).parents[2])
        fi = FeatureIndex(base_dir)
        val_stem = fi.shard_stems[-1]
        n_docs = len(fi.doc_ids_for_shard(val_stem))
        wrong = tmp_path / "wrongrows"
        wrong.mkdir()
        atomic_write_parquet(
            pa.table({"text": ["x"] * (n_docs + 3)}), wrong / f"{val_stem}.parquet"
        )
        with pytest.raises(ValueError, match="does not pair"):
            _assert_live_val_pair_aligned(fi, str(wrong), None, "base val")
