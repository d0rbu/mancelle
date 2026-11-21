"""Slice 17 audit (2026-05-05) P0.G regression: validate_bpb_config mutation.

``validate_bpb_config`` mutates ``cfg.aggregation.primary_metric`` in place
to coerce the Hydra-supplied string into the ``LeaderboardMetric`` enum.
The fix documents this side effect explicitly. This test pins the contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
    ContextCacheConfig,
    validate_bpb_config,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, LeaderboardMetric
from tests.runtime_attribute_helpers import set_runtime_attribute


def _populate_required_artifacts(source_run_dir: Path) -> None:
    """Create every required file plus ``baselines/`` under source_run_dir."""
    source_run_dir.mkdir(parents=True, exist_ok=True)
    sp = SourcePaths(source_run_dir=source_run_dir)
    sp.concepts_json().write_text(
        json.dumps([{"concept_id": 0}]),
        encoding="utf-8",
    )
    schema = schema_with_stage1_continuation_contract(
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
        pa.Table.from_pylist(positive_rows, schema=schema),
        sp.synth_training_positive(),
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
        sp.synth_training_negative(),
    )
    sp.seeds_train().write_bytes(b"")
    sp.seeds_eval().write_bytes(b"")
    sp.seed_pool_manifest().write_text("{}")
    sp.concept_features().write_bytes(b"")
    sp.concept_features_neg().write_bytes(b"")
    sp.baselines_root().mkdir(parents=True, exist_ok=True)


def _build_minimal_cfg(tmp_path: Path, **overrides) -> AxBenchBpbConfig:
    src = tmp_path / "source"
    _populate_required_artifacts(src)
    return AxBenchBpbConfig(
        source_run_dir=str(src),
        output_root=str(tmp_path / "output"),
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        device="cpu",
        context_cache=ContextCacheConfig(),
        aggregation=AggregationConfig(),
        decoder_only={"source": "x", "model_tag": "y", "step": 1},
        encdec={"source": "x", "model_tag": "y", "step": 1},
        enabled_methods=["control_decoder_only"],
        **overrides,
    )


def test_string_primary_metric_coerced_to_enum(tmp_path):
    """Hydra string → LeaderboardMetric enum after validate_bpb_config runs."""
    cfg = _build_minimal_cfg(tmp_path)
    set_runtime_attribute(
        cfg.aggregation,
        "primary_metric",
        "macro_mean_best_primary_margin",
    )
    validate_bpb_config(cfg)
    assert isinstance(cfg.aggregation.primary_metric, LeaderboardMetric)
    assert cfg.aggregation.primary_metric == LeaderboardMetric.MACRO_MEAN_BEST_PRIMARY_MARGIN


def test_enum_passes_through_unchanged(tmp_path):
    """Already-coerced enum stays as-is."""
    cfg = _build_minimal_cfg(tmp_path)
    cfg.aggregation.primary_metric = LeaderboardMetric.MACRO_MEAN_BEST_MARGIN_INTERVENED
    validate_bpb_config(cfg)
    assert cfg.aggregation.primary_metric == LeaderboardMetric.MACRO_MEAN_BEST_MARGIN_INTERVENED


def test_unknown_metric_raises(tmp_path):
    """Unknown primary_metric → AxBenchBpbError."""
    cfg = _build_minimal_cfg(tmp_path)
    set_runtime_attribute(
        cfg.aggregation,
        "primary_metric",
        "not_a_real_metric",
    )
    with pytest.raises(AxBenchBpbError):
        validate_bpb_config(cfg)


def test_validate_bpb_config_docstring_documents_mutation():
    """The docstring must document the in-place mutation (P0.G)."""
    doc = validate_bpb_config.__doc__ or ""
    assert "MUTATES" in doc.upper() or "in place" in doc.lower(), (
        "validate_bpb_config docstring must document the side effect on "
        "cfg.aggregation.primary_metric (P0.G)."
    )
