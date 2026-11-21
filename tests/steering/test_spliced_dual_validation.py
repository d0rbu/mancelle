"""Issue 605: original + spliced validation routing for encdec training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_train as train_mod
from steering.conditional.encoder_decoder.loss_eval import ConditionedBpbMetrics
from steering.overlay_io import CHECKSUM_PREFIX_LEN


class RecordingRun:
    def __init__(self) -> None:
        self.logs: list[dict] = []
        self.summary: dict[str, object] = {}

    def log(self, payload, *args, **kwargs) -> None:
        self.logs.append(dict(payload))


def _touch_overlay_shard(directory: Path, stem: str = "shard_00000") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.safetensors.zst").write_bytes(b"x")


def test_original_val_feature_dir_auto_detects_canonical_splice_layout(tmp_path: Path) -> None:
    base = tmp_path / "s128" / "l8" / "a_res" / "p_max_pool" / "f16k" / "m_gemma-2-2b" / "base"
    derived = base / "splice" / "cut_frac0.15" / "seed42"
    derived_scientific = base / "splice" / "cut_frac1e-05" / "seed42"

    assert train_mod._auto_original_val_feature_dir(derived) == base
    assert train_mod._auto_original_val_feature_dir(derived_scientific) == base
    assert train_mod._auto_original_val_feature_dir(base) is None


@pytest.mark.parametrize("seed,pool", [(42, 64), (-7, 2), (0, 128)])
def test_original_val_feature_dir_auto_detects_opposite_pairing(
    tmp_path: Path, seed: int, pool: int
) -> None:
    base = tmp_path / "base" / "context_independent" / "no_bos"
    derived = base / "splice" / "cut_frac0.15" / f"seed{seed}_opposite_percentile_pool{pool}"
    assert train_mod._auto_original_val_feature_dir(derived) == base


@pytest.mark.parametrize(
    "params_leaf,seed_leaf",
    [
        ("rate0.3", "seed42"),
        ("p0.3", "seed42"),
        ("cut_frac0.15", "run42"),
        ("cut_frac0.15", "seed42_opposite_percentile"),
        ("cut_frac0.15", "seed42_opposite_percentile_pool0"),
        ("cut_frac0.15", "seed42_opposite_percentile_pool1"),
        ("cut_frac0.15", "seed42_opposite_percentile_pool64_extra"),
    ],
)
def test_original_val_feature_dir_rejects_malformed_splice_layout(
    tmp_path: Path,
    params_leaf: str,
    seed_leaf: str,
) -> None:
    base = tmp_path / "base"
    derived = base / "splice" / params_leaf / seed_leaf

    with pytest.raises(
        ValueError,
        match=r"canonical <base>/splice/cut_frac\{cut_fraction\}/seed\{seed\}",
    ):
        train_mod._auto_original_val_feature_dir(derived)


def test_original_val_feature_dir_override_enables_custom_output_dir(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    derived = tmp_path / "custom_spliced"
    cfg = OmegaConf.create({"original_val_feature_dir": str(base)})

    assert train_mod._resolve_original_val_feature_dir(cfg, derived) == base


def test_original_val_feature_dir_rejects_active_dir(tmp_path: Path) -> None:
    active = tmp_path / "base"
    active.mkdir()
    cfg = OmegaConf.create({"original_val_feature_dir": str(active)})

    with pytest.raises(ValueError, match="active feature_dir"):
        train_mod._resolve_original_val_feature_dir(cfg, active)


def test_packed_row_train_and_val_dir_resolution(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    pct = "a" * 64
    tok = "b" * 64
    train_dir = train_mod._packed_row_train_dir(
        feature_dir,
        percentile_checksum=pct,
        tokenizer_checksum=tok,
        context_len=2048,
        max_train_shards=None,
    )

    assert train_dir == (
        feature_dir
        / "pct"
        / ("a" * CHECKSUM_PREFIX_LEN)
        / "tok"
        / ("b" * CHECKSUM_PREFIX_LEN)
        / "c2048"
        / "sall"
    )
    val_dir = train_dir.parent / "val_sall"
    _touch_overlay_shard(val_dir)
    assert train_mod.discover_val_packed_row_dir(train_dir) == val_dir


def test_validation_specs_normal_packed_logs_only_val(tmp_path: Path) -> None:
    active_val = tmp_path / "derived" / "val_sall"
    specs = train_mod._build_validation_eval_specs(
        packed_row_dir=tmp_path / "derived" / "sall",
        active_val_packed_row_dir=active_val,
        active_val_selection_dir=None,
        original_val_packed_row_dir=None,
        original_val_selection_dir=None,
    )

    assert [(s.metric_prefix, s.log_label, s.packed_row_dir) for s in specs] == [
        ("val", "validation", active_val)
    ]


def test_validation_specs_spliced_routes_original_then_spliced(tmp_path: Path) -> None:
    original_val = tmp_path / "base" / "val_sall"
    spliced_val = tmp_path / "derived" / "val_sall"
    specs = train_mod._build_validation_eval_specs(
        packed_row_dir=tmp_path / "derived" / "sall",
        active_val_packed_row_dir=spliced_val,
        active_val_selection_dir=None,
        original_val_packed_row_dir=original_val,
        original_val_selection_dir=None,
    )

    assert [(s.metric_prefix, s.log_label, s.packed_row_dir) for s in specs] == [
        ("val", "original validation", original_val),
        ("val_spliced", "spliced validation", spliced_val),
    ]


def test_validation_specs_spliced_requires_active_val(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both original and spliced"):
        train_mod._build_validation_eval_specs(
            packed_row_dir=tmp_path / "derived" / "sall",
            active_val_packed_row_dir=None,
            active_val_selection_dir=None,
            original_val_packed_row_dir=tmp_path / "base" / "val_sall",
            original_val_selection_dir=None,
        )


def test_validation_specs_live_legacy_path_stays_val_only() -> None:
    specs = train_mod._build_validation_eval_specs(
        packed_row_dir=None,
        active_val_packed_row_dir=None,
        active_val_selection_dir=None,
        original_val_packed_row_dir=None,
        original_val_selection_dir=None,
    )

    assert len(specs) == 1
    assert specs[0].metric_prefix == "val"
    assert specs[0].packed_row_dir is None


def test_val_spliced_metric_key_set_mirrors_val(monkeypatch) -> None:
    monkeypatch.setattr(train_mod.wandb, "Histogram", lambda values: ("hist", tuple(values)))
    cfg = OmegaConf.create({"num_buckets": 3})
    metrics = ConditionedBpbMetrics(
        bpb_conditioned=1.25,
        features_per_segment=[1, 2, 3, 4],
        avg_features_per_segment=2.5,
        avg_features_per_batch=10.0,
        magnitude_buckets=[0, 1, 1, 2],
    )

    def collect(prefix: str) -> set[str]:
        run = RecordingRun()
        keys = set(train_mod._validation_bpb_log(prefix, metrics))
        train_mod._log_validation_feature_stats(
            wandb_run=run,
            prefix=prefix,
            metrics=metrics,
            cfg=cfg,
            master_process=True,
        )
        for payload in run.logs:
            keys.update(payload)
        keys.update(run.summary)
        return {key.removeprefix(f"{prefix}/") for key in keys}

    assert collect("val_spliced") == collect("val")


def test_validation_loader_routes_original_and_spliced_packed_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    original_val = tmp_path / "base" / "val_sall"
    spliced_val = tmp_path / "derived" / "val_sall"
    original_sel = original_val / "sel" / "abc"
    spliced_sel = spliced_val / "sel" / "abc"
    calls = []

    def fake_precomputed_data_loader(**kwargs):
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(train_mod, "precomputed_data_loader", fake_precomputed_data_loader)
    cfg = OmegaConf.create({"device_batch_size": 4})
    shared = SimpleNamespace(
        pad_F=32,
        ddp_rank=1,
        ddp_world_size=2,
        feature_clusters=None,
        remote_materializers={},
    )
    feature_pipeline = SimpleNamespace(effective_buckets=3)
    specs = train_mod._build_validation_eval_specs(
        packed_row_dir=tmp_path / "derived" / "sall",
        active_val_packed_row_dir=spliced_val,
        active_val_selection_dir=spliced_sel,
        original_val_packed_row_dir=original_val,
        original_val_selection_dir=original_sel,
    )

    for spec in specs:
        train_mod._build_validation_loader(
            spec=spec,
            cfg=cfg,
            shared=shared,  # ty: ignore[invalid-argument-type]
            feature_pipeline=feature_pipeline,  # ty: ignore[invalid-argument-type]
            tokenizer=object(),
            device=t.device("cpu"),
        )

    assert [call["packed_row_dir"] for call in calls] == [original_val, spliced_val]
    assert [call["selection_dir"] for call in calls] == [original_sel, spliced_sel]
    assert all(call["B"] == 4 for call in calls)
    assert all(call["pad_F"] == 32 for call in calls)
    assert all(call["ddp_rank"] == 1 and call["ddp_world_size"] == 2 for call in calls)
    assert all(call["fp_target_layout"] is None for call in calls)


def test_validation_loader_live_path_uses_val_split_and_no_label_dropout(
    monkeypatch,
) -> None:
    calls = []

    def fake_live_loader(**kwargs):
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(
        train_mod,
        "encoder_decoder_data_loader_with_state",
        fake_live_loader,
    )
    cfg = OmegaConf.create(
        {
            "device_batch_size": 2,
            "max_seq_len": 8,
            "max_train_shards": 7,
            "next_segment_conditioning_offset": True,
        }
    )
    shared = SimpleNamespace(
        feature_index=object(),
        extraction_segment_size=128,
        pad_F=16,
        live_pad_n_seg=24,
    )
    spec = train_mod.ValidationEvalSpec(
        metric_prefix="val",
        log_label="validation",
        packed_row_dir=None,
        selection_dir=None,
    )
    train_mod._build_validation_loader(
        spec=spec,
        cfg=cfg,
        shared=shared,  # ty: ignore[invalid-argument-type]
        feature_pipeline=object(),  # ty: ignore[invalid-argument-type]
        tokenizer=object(),
        device=t.device("cpu"),
    )

    assert calls[0]["split"] == train_mod.DatasetSplit.VAL
    assert calls[0]["label_dropout"] == 0.0
    assert calls[0]["max_shards"] == 7
    assert calls[0]["next_segment_conditioning_offset"] is True
    assert calls[0]["pad_F"] == 16
    assert calls[0]["pad_n_seg"] == 24
    assert calls[0]["fp_target_layout"] is None


def test_conditional_encdec_train_config_exposes_original_val_feature_dir() -> None:
    config_dir = str(Path(__file__).parents[2] / "configs" / "training")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="conditional_encdec_train")
        overridden = compose(
            config_name="conditional_encdec_train",
            overrides=["original_val_feature_dir=/tmp/base_features"],
        )

    assert cfg.original_val_feature_dir is None
    assert overridden.original_val_feature_dir == "/tmp/base_features"
