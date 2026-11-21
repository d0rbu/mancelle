"""Unit tests for AxBenchBpbConfig and validate_bpb_config."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from safetensors.torch import save_file

from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.config import (
    AggregationConfig,
    AxBenchBpbConfig,
    validate_bpb_config,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError


def test_config_construction_with_required_fields():
    cfg = AxBenchBpbConfig(
        source_run_dir="/src/run",
        output_root="/out",
        feature_dir="/feat",
        feature_statistics_path="/feat/stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
    )
    assert cfg.source_run_dir == "/src/run"
    assert cfg.device == "cuda"
    assert cfg.stage == "all"
    assert cfg.device_batch_size == 16
    assert cfg.filter_unaccepted_samples is True
    assert cfg.aggregation.min_rows_per_side == 10
    assert cfg.aggregation.invalid_warning_threshold == 0.01
    assert cfg.aggregation.invalid_drop_threshold == 0.05
    assert cfg.aggregation.primary_metric == "macro_mean_best_primary_margin"
    assert cfg.context_cache.extraction_batch_size == 32
    assert cfg.context_cache.force_rebuild is False


def test_config_nested_dataclasses_independent():
    cfg_a = AxBenchBpbConfig(
        source_run_dir="/a", output_root="/a_out", feature_dir="/f",
        feature_statistics_path="/s",
        decoder_only={}, encdec={},
    )
    cfg_b = AxBenchBpbConfig(
        source_run_dir="/b", output_root="/b_out", feature_dir="/f",
        feature_statistics_path="/s",
        decoder_only={}, encdec={},
    )
    cfg_a.aggregation.min_rows_per_side = 999
    assert cfg_b.aggregation.min_rows_per_side == 10  # not poisoned


def _base_cfg(tmp_path, **overrides):
    # Create a minimal source run dir with all Stage 0-3 artifact placeholders
    src = tmp_path / "src"
    src.mkdir()
    (src / "concepts_stage1c.json").write_text(
        json.dumps([{"concept_id": 0}]),
        encoding="utf-8",
    )
    positive_schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            TRAINING_SCHEMA,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    positive_rows = [
        {
            "concept_id": 0,
            "category": "positive",
            "content_class": "concept_rich",
            "prefix_text": "prefix",
            "output_text": "rich",
            "output_concept": "concept",
            "n_prefix_nanochat_tokens": 2,
            "gen_source": "test",
            "gen_seed": 1,
            "qc_feature_id": 1,
            "qc_seed_percentile": 10.0,
            "qc_output_percentile": 50.0,
            "qc_delta": 40.0,
            "qc_attempts": 1,
            "qc_accepted": True,
            "qc_zero_activation_verified": False,
        },
        {
            "concept_id": 0,
            "category": "negative",
            "content_class": "concept_avoiding",
            "prefix_text": "prefix",
            "output_text": "avoid",
            "output_concept": "concept",
            "n_prefix_nanochat_tokens": 2,
            "gen_source": "test",
            "gen_seed": 1,
            "qc_feature_id": 1,
            "qc_seed_percentile": float("nan"),
            "qc_output_percentile": float("nan"),
            "qc_delta": float("nan"),
            "qc_attempts": 1,
            "qc_accepted": True,
            "qc_zero_activation_verified": True,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(positive_rows, schema=positive_schema),
        src / "synth_training_positive.parquet",
    )
    negative_table = pa.Table.from_pylist(
        [
            {
                "concept_id": 0,
                "content_class": "concept_rich",
                "qc_accepted": True,
            },
            {
                "concept_id": 0,
                "content_class": "concept_avoiding",
                "qc_accepted": True,
            },
        ]
    )
    pq.write_table(
        negative_table.cast(
            schema_with_stage1_continuation_contract(
                negative_table.schema,
                role="stage1c_negative",
            )
        ),
        src / "synth_training_negative.parquet",
    )
    (src / "seeds_train.safetensors").touch()
    (src / "seeds_eval.safetensors").touch()
    (src / "seed_pool_manifest.json").touch()
    (src / "concept_features.safetensors").touch()
    (src / "concept_features_neg.safetensors").touch()
    baselines = src / "baselines"
    for family in ("diffmean", "linear_probe"):
        for target in ("decoder_only", "encdec"):
            for mode in ("positive", "negative"):
                path = baselines / family / target / mode / "concept_0000.safetensors"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {"direction": t.tensor([1.0, -1.0], dtype=t.float32)},
                    str(path),
                    metadata={
                        "concept_id": "0",
                        "mode": mode,
                        "target_model": target,
                        "layer": "8",
                    },
                )
    cfg = AxBenchBpbConfig(
        source_run_dir=str(src),
        output_root=str(tmp_path / "out"),
        feature_dir=str(tmp_path / "feat"),
        feature_statistics_path=str(tmp_path / "feat/stats.safetensors"),
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
    )
    for key, value in overrides.items():
        # Several tests deliberately install invalid raw values so validation,
        # rather than the typed constructor, owns the failure boundary.
        setattr(cfg, key, value)
    return cfg


def test_validate_accepts_ok_config(tmp_path):
    validate_bpb_config(_base_cfg(tmp_path))  # must not raise


def test_validate_rejects_direction_layer_mismatch_before_runtime_init(tmp_path):
    cfg = _base_cfg(tmp_path)
    path = (
        Path(cfg.source_run_dir)
        / "baselines/diffmean/decoder_only/positive/concept_0000.safetensors"
    )
    save_file(
        {"direction": t.tensor([1.0, -1.0], dtype=t.float32)},
        str(path),
        metadata={
            "concept_id": "0",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": "10",
        },
    )

    with pytest.raises(AxBenchBpbError, match="layer mismatch") as exc_info:
        validate_bpb_config(cfg)

    message = str(exc_info.value)
    assert "requested/applied layer=8" in message
    assert "artifact training layer=10" in message
    assert str(path) in message


def test_validate_rejects_missing_source_run_dir(tmp_path):
    cfg = _base_cfg(tmp_path, source_run_dir="")
    with pytest.raises(AxBenchBpbError, match="source_run_dir"):
        validate_bpb_config(cfg)


def test_validate_rejects_missing_output_root(tmp_path):
    cfg = _base_cfg(tmp_path, output_root="")
    with pytest.raises(AxBenchBpbError, match="output_root"):
        validate_bpb_config(cfg)


def test_validate_rejects_missing_feature_dir(tmp_path):
    cfg = _base_cfg(tmp_path, feature_dir="")
    with pytest.raises(AxBenchBpbError, match="feature_dir"):
        validate_bpb_config(cfg)


def test_validate_rejects_missing_feature_statistics_path(tmp_path):
    cfg = _base_cfg(tmp_path, feature_statistics_path="")
    with pytest.raises(AxBenchBpbError, match="feature_statistics_path"):
        validate_bpb_config(cfg)


def test_validate_rejects_bad_stage(tmp_path):
    cfg = _base_cfg(tmp_path, stage="nonsense")
    with pytest.raises(AxBenchBpbError, match="Unknown stage"):
        validate_bpb_config(cfg)


def test_config_rejects_bad_batch_size():
    with pytest.raises(BeartypeCallHintParamViolation, match="device_batch_size"):
        AxBenchBpbConfig(device_batch_size=0)


@pytest.mark.parametrize("value", ["", "   ", 17])
def test_config_rejects_invalid_feature_extraction_model_id(
    value: object,
) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="feature_extraction_model_id",
    ):
        AxBenchBpbConfig(feature_extraction_model_id=value)  # ty: ignore[invalid-argument-type]


def test_validate_accepts_null_feature_extraction_model_id(tmp_path):
    """None is the legacy default and must not be rejected."""
    cfg = _base_cfg(tmp_path, feature_extraction_model_id=None)
    validate_bpb_config(cfg)  # must not raise


def test_validate_accepts_nonempty_feature_extraction_model_id(tmp_path):
    cfg = _base_cfg(tmp_path, feature_extraction_model_id="gemma-3-4b-pt")
    validate_bpb_config(cfg)  # must not raise


def test_config_rejects_bad_min_rows():
    with pytest.raises(BeartypeCallHintParamViolation, match="min_rows_per_side"):
        AggregationConfig(min_rows_per_side=0)


def test_validate_rejects_threshold_order(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.aggregation = AggregationConfig(
        invalid_warning_threshold=0.5, invalid_drop_threshold=0.1
    )
    with pytest.raises(AxBenchBpbError, match="threshold"):
        validate_bpb_config(cfg)


def test_validate_rejects_unknown_primary_metric(tmp_path):
    """An unknown primary_metric (e.g. ``"macro_blah"``) must surface as an
    ``AxBenchBpbError`` listing the 4 valid ``LeaderboardMetric`` values
    rather than failing later inside ``write_leaderboard_csv``."""
    cfg = _base_cfg(tmp_path)
    cfg.aggregation.primary_metric = "macro_blah"
    with pytest.raises(AxBenchBpbError, match="primary_metric"):
        validate_bpb_config(cfg)


def test_validate_rejects_missing_stage_0_3(tmp_path):
    cfg = _base_cfg(tmp_path)
    shutil.rmtree(cfg.source_run_dir)
    (tmp_path / "src").mkdir()
    with pytest.raises(AxBenchBpbError, match="missing"):
        validate_bpb_config(cfg)


def test_config_default_diffmean_and_probe_layer():
    """diffmean_and_probe_layer defaults to 8 (matches axbench-base)."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/s",
        output_root="/tmp/o",
        feature_dir="/tmp/f",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    assert cfg.diffmean_and_probe_layer == 8


def test_config_default_lora_dict():
    """lora field defaults to empty dict (caller fills with rank/alpha/etc)."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/s",
        output_root="/tmp/o",
        feature_dir="/tmp/f",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    assert cfg.lora == {}


def test_config_lora_field_overrideable():
    """lora can be set with custom values."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/s",
        output_root="/tmp/o",
        feature_dir="/tmp/f",
        feature_statistics_path="/tmp/stats.safetensors",
        lora={"rank": 8, "alpha": 16.0},
    )
    assert cfg.lora == {"rank": 8, "alpha": 16.0}
