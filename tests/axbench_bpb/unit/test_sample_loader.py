# tests/axbench_bpb/unit/test_sample_loader.py
"""Unit tests for sample loading and sample_id generation."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.negative_mode import NEG_TRAINING_SCHEMA
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.sample_loader import (
    collect_unique_prefixes,
    load_raw_synth_rows,
    make_prefix_hash,
    make_sample_id,
)
from steering.eval.axbench_bpb.types import SteeringMode


def _write_positive_parquet(tmp_path, rows):
    path = tmp_path / "synth_training_positive.parquet"
    schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            TRAINING_SCHEMA,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def test_make_sample_id_deterministic():
    sid1 = make_sample_id(
        concept_id=42, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="the ", output_text="quick brown fox",
    )
    sid2 = make_sample_id(
        concept_id=42, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="the ", output_text="quick brown fox",
    )
    assert sid1 == sid2
    assert len(sid1) == 16
    assert all(c in "0123456789abcdef" for c in sid1)


def _sample_id(
    *,
    concept_id: int = 42,
    mode: SteeringMode = SteeringMode.POSITIVE,
    content_class: str = "concept_rich",
    prefix_text: str = "the ",
    output_text: str = "quick brown fox",
) -> str:
    return make_sample_id(
        concept_id=concept_id,
        mode=mode,
        content_class=content_class,
        prefix_text=prefix_text,
        output_text=output_text,
    )


def test_make_sample_id_distinguishes_inputs():
    sid_base = _sample_id()
    assert _sample_id(concept_id=43) != sid_base
    assert _sample_id(mode=SteeringMode.NEGATIVE) != sid_base
    assert _sample_id(content_class="concept_avoiding") != sid_base
    assert _sample_id(prefix_text="The ") != sid_base
    assert _sample_id(output_text="slow brown fox") != sid_base


def test_make_prefix_hash_deterministic():
    assert make_prefix_hash("hello") == make_prefix_hash("hello")
    assert make_prefix_hash("hello") != make_prefix_hash("world")
    assert len(make_prefix_hash("hello")) == 16


def test_load_positive_rows_filters_unaccepted(tmp_path):
    rows = [
        {
            "concept_id": 1, "category": "positive", "content_class": "concept_rich",
            "prefix_text": "A", "output_text": "B", "output_concept": "",
            "n_prefix_nanochat_tokens": 1, "gen_source": "openrouter:x", "gen_seed": 0,
            "qc_feature_id": 99, "qc_seed_percentile": 0.1, "qc_output_percentile": 0.8,
            "qc_delta": 0.7, "qc_attempts": 1, "qc_accepted": True,
        },
        {
            "concept_id": 1, "category": "positive", "content_class": "concept_rich",
            "prefix_text": "A2", "output_text": "B2", "output_concept": "",
            "n_prefix_nanochat_tokens": 1, "gen_source": "openrouter:x", "gen_seed": 0,
            "qc_feature_id": 99, "qc_seed_percentile": 0.1, "qc_output_percentile": 0.2,
            "qc_delta": 0.1, "qc_attempts": 4, "qc_accepted": False,
        },
    ]
    path = _write_positive_parquet(tmp_path, rows)
    kept = load_raw_synth_rows(path, filter_unaccepted=True)
    assert len(kept) == 1
    assert kept[0]["prefix_text"] == "A"


def test_load_positive_rows_no_filter(tmp_path):
    rows = [
        {
            "concept_id": 1, "category": "positive", "content_class": "concept_rich",
            "prefix_text": "A", "output_text": "B", "output_concept": "",
            "n_prefix_nanochat_tokens": 1, "gen_source": "", "gen_seed": 0,
            "qc_feature_id": 99, "qc_seed_percentile": 0.1, "qc_output_percentile": 0.8,
            "qc_delta": 0.7, "qc_attempts": 1, "qc_accepted": False,
        },
    ]
    path = _write_positive_parquet(tmp_path, rows)
    kept = load_raw_synth_rows(path, filter_unaccepted=False)
    assert len(kept) == 1


def _write_negative_parquet(tmp_path, rows):
    path = tmp_path / "synth_training_negative.parquet"
    schema = schema_with_stage1_continuation_contract(
        NEG_TRAINING_SCHEMA,
        role="stage1c_negative",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def test_load_negative_rows_preserves_extra_columns_and_filters(tmp_path):
    """Negative-mode schema has 4 extra columns not present on positive;
    verify they round-trip through the loader, AND qc_accepted filter works."""
    rows = [
        {
            # An accepted concept_rich verification continuation
            "concept_id": 5, "category": "positive",  # per spec: concept_rich side of neg mode
            "content_class": "concept_rich",
            "prefix_text": "PREF_A", "output_text": "OUT_A", "output_concept": "desc",
            "n_prefix_nanochat_tokens": 3, "gen_source": "local:base/d13", "gen_seed": -1,
            "qc_feature_id": 99, "qc_seed_percentile": float("nan"),
            "qc_output_percentile": 75.0, "qc_delta": float("nan"),
            "qc_attempts": 0, "qc_accepted": True,
            "prefix_source_shard": 12, "prefix_source_doc": 345,
            "prefix_source_seg_idx": 2, "qc_zero_activation_verified": False,
        },
        {
            # An accepted concept_avoiding LLM continuation
            "concept_id": 5, "category": "negative",
            "content_class": "concept_avoiding",
            "prefix_text": "PREF_A", "output_text": "OUT_B", "output_concept": "desc",
            "n_prefix_nanochat_tokens": 3, "gen_source": "openrouter:x", "gen_seed": 42,
            "qc_feature_id": 99, "qc_seed_percentile": float("nan"),
            "qc_output_percentile": float("nan"), "qc_delta": float("nan"),
            "qc_attempts": 2, "qc_accepted": True,
            "prefix_source_shard": 12, "prefix_source_doc": 345,
            "prefix_source_seg_idx": 2, "qc_zero_activation_verified": True,
        },
        {
            # An UNACCEPTED row — must be filtered out when filter_unaccepted=True
            "concept_id": 5, "category": "negative",
            "content_class": "concept_avoiding",
            "prefix_text": "PREF_B", "output_text": "BAD", "output_concept": "desc",
            "n_prefix_nanochat_tokens": 3, "gen_source": "openrouter:x", "gen_seed": 43,
            "qc_feature_id": 99, "qc_seed_percentile": float("nan"),
            "qc_output_percentile": float("nan"), "qc_delta": float("nan"),
            "qc_attempts": 4, "qc_accepted": False,
            "prefix_source_shard": 12, "prefix_source_doc": 345,
            "prefix_source_seg_idx": 3, "qc_zero_activation_verified": False,
        },
    ]
    path = _write_negative_parquet(tmp_path, rows)

    # With filter: drop the 1 unaccepted row
    kept = load_raw_synth_rows(path, filter_unaccepted=True)
    assert len(kept) == 2

    # Extra columns round-trip correctly
    first = kept[0]
    assert first["prefix_source_shard"] == 12
    assert first["prefix_source_doc"] == 345
    assert first["prefix_source_seg_idx"] == 2
    assert first["qc_zero_activation_verified"] is False

    second = kept[1]
    assert second["qc_zero_activation_verified"] is True
    assert second["content_class"] == "concept_avoiding"

    # Without filter: keep all 3
    unfiltered = load_raw_synth_rows(path, filter_unaccepted=False)
    assert len(unfiltered) == 3
    assert unfiltered[2]["qc_accepted"] is False


def test_collect_unique_prefixes_dedupes_across_parquets(tmp_path):
    """Unique prefixes across both positive + negative parquets, with hash."""
    # Two positive rows share prefix "same"; negative has "other"
    pos_rows = [
        {
            "concept_id": 1, "category": "positive",
            "content_class": "concept_rich",
            "prefix_text": "same", "output_text": "A",
            "output_concept": "", "n_prefix_nanochat_tokens": 1,
            "gen_source": "openrouter:x", "gen_seed": 0,
            "qc_feature_id": 1, "qc_seed_percentile": 0.0,
            "qc_output_percentile": 0.9, "qc_delta": 0.9,
            "qc_attempts": 1, "qc_accepted": True,
        },
        {
            "concept_id": 2, "category": "positive",
            "content_class": "concept_rich",
            "prefix_text": "same", "output_text": "B",
            "output_concept": "", "n_prefix_nanochat_tokens": 1,
            "gen_source": "openrouter:x", "gen_seed": 0,
            "qc_feature_id": 2, "qc_seed_percentile": 0.0,
            "qc_output_percentile": 0.9, "qc_delta": 0.9,
            "qc_attempts": 1, "qc_accepted": True,
        },
    ]
    pos_path = _write_positive_parquet(tmp_path, pos_rows)

    neg_rows = [
        {
            "concept_id": 1, "category": "positive",
            "content_class": "concept_rich",
            "prefix_text": "other",
            "output_text": "X", "output_concept": "",
            "n_prefix_nanochat_tokens": 1, "gen_source": "",
            "gen_seed": 0, "qc_feature_id": 1,
            "qc_seed_percentile": float("nan"), "qc_output_percentile": 0.0,
            "qc_delta": float("nan"), "qc_attempts": 0, "qc_accepted": True,
            "prefix_source_shard": 0, "prefix_source_doc": 0,
            "prefix_source_seg_idx": 0, "qc_zero_activation_verified": False,
        },
    ]
    neg_path = _write_negative_parquet(tmp_path, neg_rows)

    unique = collect_unique_prefixes(pos_path, neg_path, filter_unaccepted=True)
    assert set(unique.keys()) == {"same", "other"}
    assert len(unique) == 2
    # Values are prefix hashes (make_prefix_hash imported at module top)
    assert unique["same"] == make_prefix_hash("same")
    assert unique["other"] == make_prefix_hash("other")

    final_only = collect_unique_prefixes(
        pos_path,
        neg_path,
        filter_unaccepted=True,
        concept_ids={2},
    )
    assert final_only == {"same": make_prefix_hash("same")}


def test_load_rows_filters_to_explicit_final_concept_population(tmp_path):
    rows = [
        {
            "concept_id": concept_id,
            "category": "positive",
            "content_class": "concept_rich",
            "prefix_text": f"prefix-{concept_id}",
            "output_text": f"output-{concept_id}",
            "output_concept": "",
            "n_prefix_nanochat_tokens": 1,
            "gen_source": "openrouter:x",
            "gen_seed": 0,
            "qc_feature_id": concept_id,
            "qc_seed_percentile": 0.0,
            "qc_output_percentile": 0.9,
            "qc_delta": 0.9,
            "qc_attempts": 1,
            "qc_accepted": True,
        }
        for concept_id in (1, 4, 9)
    ]
    path = _write_positive_parquet(tmp_path, rows)

    kept = load_raw_synth_rows(
        path,
        filter_unaccepted=True,
        concept_ids={1, 9},
    )

    assert [row["concept_id"] for row in kept] == [1, 9]
    assert {row["prefix_text"] for row in kept} == {"prefix-1", "prefix-9"}
