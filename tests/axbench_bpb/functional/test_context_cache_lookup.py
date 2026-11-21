# tests/axbench_bpb/functional/test_context_cache_lookup.py
"""Functional tests: manifest load + build_context_cache idempotency."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.negative_mode import NEG_TRAINING_SCHEMA
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb import context_cache
from steering.eval.axbench_bpb.context_cache import (
    build_context_cache,
    load_manifest,
)
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.target_conditioning import ContextFeatureSegment


def test_load_manifest_ok(tmp_path):
    op = OutputPaths(output_root=tmp_path, run_tag="r1")
    op.context_cache_dir().mkdir(parents=True, exist_ok=True)
    op.context_cache_manifest().write_text(
        json.dumps({"abc123": {"n_segments": 2, "n_features_total": 10}})
    )
    m = load_manifest(op)
    assert m["abc123"]["n_segments"] == 2
    assert m["abc123"]["n_features_total"] == 10


def test_load_manifest_missing_raises(tmp_path):
    op = OutputPaths(output_root=tmp_path, run_tag="r1")
    with pytest.raises(AxBenchBpbError, match="manifest"):
        load_manifest(op)


# --- Idempotency test ---

@dataclass
class _CacheCfg:
    force_rebuild: bool = False


@dataclass
class _Cfg:
    """Minimal stand-in for AxBenchBpbConfig (only fields build_context_cache reads)."""

    feature_dir: str = "legacy-explicit-fixed-features"
    filter_unaccepted_samples: bool = True
    extraction_max_tokens: int = 2048
    extraction_threshold: float = 0.0
    context_cache: _CacheCfg = field(default_factory=_CacheCfg)


def _pos_row(prefix: str, output: str, cid: int = 0) -> dict:
    return {
        "concept_id": cid, "category": "positive",
        "content_class": "concept_rich", "prefix_text": prefix,
        "output_text": output, "output_concept": "",
        "n_prefix_nanochat_tokens": 1, "gen_source": "",
        "gen_seed": 0, "qc_feature_id": 0,
        "qc_seed_percentile": 10.0, "qc_output_percentile": 40.0,
        "qc_delta": 30.0, "qc_attempts": 1, "qc_accepted": True,
        "qc_zero_activation_verified": False,
    }


def _write_pos(tmp_path: Path, rows: list) -> Path:
    path = tmp_path / "pos.parquet"
    schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            TRAINING_SCHEMA,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _write_neg_empty(tmp_path: Path) -> Path:
    path = tmp_path / "neg.parquet"
    schema = schema_with_stage1_continuation_contract(
        NEG_TRAINING_SCHEMA,
        role="stage1c_negative",
    )
    pq.write_table(pa.Table.from_pylist([], schema=schema), path)
    return path


def test_build_context_cache_idempotent(tmp_path, monkeypatch):
    """Second call with same inputs must not re-invoke extract_prefix_features;
    force_rebuild=True must re-invoke."""
    pos_path = _write_pos(tmp_path, [_pos_row("hello", "A"), _pos_row("world", "B", cid=1)])
    neg_path = _write_neg_empty(tmp_path)

    extract_count = {"n": 0}

    def _fake_extract(**kwargs):
        extract_count["n"] += 1
        return (
            ContextFeatureSegment(
                feature_ids=(1,), buckets=(2,),
                char_start=0, char_end=len(kwargs["prefix_text"]),
            ),
        )

    monkeypatch.setattr(context_cache, "extract_prefix_features", _fake_extract)

    cfg = _Cfg()
    op = OutputPaths(output_root=tmp_path, run_tag="rtest")

    # First call: extract both prefixes
    manifest_path1 = build_context_cache(
        cfg=cfg, output_paths=op,
        source_positive=pos_path, source_negative=neg_path,
        base_model=None, base_sae=None,
        boundaries=torch.zeros(64, 9),
        segment_size=128, sae_layer=8,
    )
    assert manifest_path1 is not None
    assert manifest_path1.exists()
    assert extract_count["n"] == 2  # one per unique prefix

    # Second call: no re-extraction
    build_context_cache(
        cfg=cfg, output_paths=op,
        source_positive=pos_path, source_negative=neg_path,
        base_model=None, base_sae=None,
        boundaries=torch.zeros(64, 9),
        segment_size=128, sae_layer=8,
    )
    assert extract_count["n"] == 2  # unchanged

    # Third call with force_rebuild: re-extracts both
    cfg.context_cache.force_rebuild = True
    build_context_cache(
        cfg=cfg, output_paths=op,
        source_positive=pos_path, source_negative=neg_path,
        base_model=None, base_sae=None,
        boundaries=torch.zeros(64, 9),
        segment_size=128, sae_layer=8,
    )
    assert extract_count["n"] == 4  # 2 more
