"""Unit tests for Stage 2 concept_feature_mapping (Task 19)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors import safe_open

from steering.eval.axbench_base import concept_feature_mapping, paths
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.concept_feature_mapping import (
    _batched_dense_sae_max_pool,
    _compute_ranked_file,
    _concept_mode_filter,
    _load_synth_rows,
    _raise_on_empty_synth_rows,
    concept_feature_mapping_canonical,
    concept_feature_mapping_delta_ranked,
)
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    TargetModelMetadata,
)
from steering.immutable_io import ImmutableArtifactError
from tests.axbench_base.unit.test_config import make_valid_config


def _stub_meta(sae_layer: int = 8) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=sae_layer,
        segment_size=128,
    )


def concept_feature_mapping_fn(
    cfg: Any,
    concepts: list[Concept],
    synth_training_positive_path: Path,
    synth_training_negative_path: Path | None = None,
    base_model: Any = None,
    base_sae: Any = None,
    target_model_meta: TargetModelMetadata | None = None,
) -> tuple[Path, Path | None]:
    """Test adapter over the two Stage 2 producers used by the live CLI."""
    if cfg.concept_feature_mode == "canonical":
        return concept_feature_mapping_canonical(cfg, concepts)
    assert cfg.concept_feature_mode == "delta_ranked"
    assert base_model is not None
    assert base_sae is not None
    assert target_model_meta is not None
    return concept_feature_mapping_delta_ranked(
        cfg,
        concepts,
        synth_training_positive_path,
        synth_training_negative_path,
        base_model,
        base_sae,
        target_model_meta,
    )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _concept(
    concept_id: int = 0, feature_index: int = 42,
) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {concept_id}",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _fake_sae(n_features: int = 16) -> SimpleNamespace:
    """Return a fake SAE with the minimum config surface."""
    return SimpleNamespace(
        cfg=SimpleNamespace(
            hook_name="blocks.8.hook_resid_post",
            release="gemmascope-res-16k",
            d_sae=n_features,
        ),
    )


def _fake_gemma() -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(n_ctx=2048))


def _cfg(
    tmp_path, **overrides,
):
    base = make_valid_config(
        run_tag="test_stage2",
        output_root=str(tmp_path / "eval_out"),
        n_rank_store=8,
        concept_feature_mode="canonical",
        run_negative_mode=True,
    )
    if overrides:
        return base.replace(**overrides)
    return base


def _synth_row(
    concept_id: int,
    content_class: str,
    output_text: str,
    category: str = "positive",
) -> dict[str, Any]:
    """Build a row matching the Stage 1b/1c parquet schema."""
    return {
        "concept_id": concept_id,
        "category": category,
        "content_class": content_class,
        "prefix_text": "prefix",
        "output_text": output_text,
        "output_concept": "some concept",
        "n_prefix_nanochat_tokens": 4,
        "gen_source": "local:base/d13/10000",
        "gen_seed": 123,
        "qc_feature_id": -1,
        "qc_seed_percentile": float("nan"),
        "qc_output_percentile": float("nan"),
        "qc_delta": float("nan"),
        "qc_attempts": 1,
        "qc_accepted": True,
    }


def _write_synth_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    tokenizer_id: str = "google/gemma-2-2b",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        rows,
        schema=schema_with_stage1_continuation_contract(
            schema_with_positive_qc_contract(
                schema_with_extraction_tokenizer_metadata(
                    TRAINING_SCHEMA,
                    tokenizer_id,
                ),
                accept_threshold=0.3,
            ),
            role=(
                "stage1c_negative"
                if "neg" in path.stem
                else "stage1b_positive"
            ),
        ),
    )
    pq.write_table(table, path)


def _read_safetensors(
    path: Path,
) -> tuple[dict[str, t.Tensor], dict[str, str]]:
    """Return (tensors, metadata) from a safetensors file."""
    tensors: dict[str, t.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        metadata = f.metadata() or {}
    return tensors, metadata


# ---------------------------------------------------------------------------
# Canonical mode
# ---------------------------------------------------------------------------


def test_canonical_mode_writes_positive_file(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(0, 42), _concept(1, 99)]
    pos_path, neg_path = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    assert pos_path.exists()
    assert neg_path is None
    tensors, _ = _read_safetensors(pos_path)
    assert tensors["feature_ids"].shape == (2, cfg.n_rank_store)
    assert tensors["percentile_deltas"].shape == (2, cfg.n_rank_store)
    assert tensors["canonical_feature_id"].shape == (2,)
    assert tensors["canonical_rank"].shape == (2,)


def test_canonical_mode_writes_neg_file_when_run_negative_mode(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=True)
    concepts = [_concept(0, 42)]
    pos_path, neg_path = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    assert pos_path.exists()
    assert neg_path is not None
    assert neg_path.exists()
    # Both files should have identical content in canonical mode.
    pos_tensors, _ = _read_safetensors(pos_path)
    neg_tensors, _ = _read_safetensors(neg_path)
    assert t.equal(pos_tensors["feature_ids"], neg_tensors["feature_ids"])
    assert t.equal(
        pos_tensors["canonical_feature_id"], neg_tensors["canonical_feature_id"],
    )


def test_canonical_mode_skips_neg_file_when_run_negative_mode_false(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(0, 42)]
    pos_path, neg_path = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    assert pos_path.exists()
    assert neg_path is None
    # Verify the negative path was NOT written.
    expected_neg = paths.AxBenchBasePaths.from_config(cfg).concept_features_neg()
    assert not expected_neg.exists()


def test_canonical_mode_sets_canonical_feature_id(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(0, 42), _concept(1, 100), _concept(2, 1024)]
    pos_path, _ = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    tensors, _ = _read_safetensors(pos_path)
    assert tensors["canonical_feature_id"].tolist() == [42, 100, 1024]


def test_canonical_mode_sets_canonical_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(i, i + 10) for i in range(3)]
    pos_path, _ = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    tensors, _ = _read_safetensors(pos_path)
    assert tensors["canonical_rank"].tolist() == [0, 0, 0]


def test_canonical_mode_slot_zero_holds_feature(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(0, 42), _concept(1, 99)]
    pos_path, _ = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    tensors, _ = _read_safetensors(pos_path)
    feature_ids = tensors["feature_ids"]
    assert int(feature_ids[0, 0]) == 42
    assert int(feature_ids[1, 0]) == 99
    # Other slots are -1.
    assert (feature_ids[0, 1:] == -1).all()
    assert (feature_ids[1, 1:] == -1).all()


def test_canonical_mode_preserves_sparse_original_concept_ids(
    tmp_path,
    monkeypatch,
):
    """Dropping concept 1 must not compact original concept 2 into row 1."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False, n_rank_store=4)
    concepts = [_concept(0, 42), _concept(2, 99)]

    pos_path, _ = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    tensors, _ = _read_safetensors(pos_path)

    assert tensors["feature_ids"].shape == (3, 4)
    assert tensors["feature_ids"][:, 0].tolist() == [42, -1, 99]
    assert bool((tensors["feature_ids"][1] == -1).all())
    assert bool(t.isnan(tensors["percentile_deltas"][1]).all())
    assert tensors["canonical_feature_id"].tolist() == [42, -1, 99]
    assert tensors["canonical_rank"].tolist() == [0, 4, 0]


def test_canonical_mode_does_not_call_gemma(tmp_path, monkeypatch):
    """Canonical mode should never invoke the base Gemma forward pass."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    def _boom(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("canonical mode should not call the extractor")

    monkeypatch.setattr(
        concept_feature_mapping, "_batched_dense_sae_max_pool", _boom,
    )
    cfg = _cfg(tmp_path, run_negative_mode=True)
    concepts = [_concept(0, 42)]
    pos_path, neg_path = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    assert pos_path.exists()
    assert neg_path is not None


def test_canonical_mode_metadata_is_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(0, 42)]
    pos_path, _ = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    _, metadata = _read_safetensors(pos_path)
    assert metadata.get("mode") == "canonical"


def test_canonical_mode_percentile_deltas_are_nan(tmp_path, monkeypatch):
    """Canonical mode fills percentile_deltas with NaN throughout."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    concepts = [_concept(0, 42), _concept(1, 99)]
    pos_path, _ = concept_feature_mapping_fn(cfg, concepts, tmp_path)
    tensors, _ = _read_safetensors(pos_path)
    pds = tensors["percentile_deltas"]
    # All entries NaN.
    assert bool(t.isnan(pds).all())


# ---------------------------------------------------------------------------
# Delta-ranked mode
# ---------------------------------------------------------------------------


def test_concept_mode_filter_picks_per_concept_rich_and_avoiding():
    """The collapsed filter is per-concept in BOTH modes: positives and
    negatives are both gated on ``concept_id == cid``. Positive mode now
    emits a paired per-concept ``concept_avoiding`` row per rich row, so
    there is no longer any concept-agnostic global pool to fold in.
    """
    rows = [
        _synth_row(0, "concept_rich", "pos_0_a"),
        _synth_row(0, "concept_rich", "pos_0_b"),
        _synth_row(0, "concept_avoiding", "neg_0_a"),
        _synth_row(0, "concept_avoiding", "neg_0_b"),
        _synth_row(1, "concept_rich", "pos_1_a"),
        _synth_row(1, "concept_avoiding", "neg_1_a"),
    ]
    pos, neg = _concept_mode_filter(rows, 0)
    assert pos == ["pos_0_a", "pos_0_b"]
    assert neg == ["neg_0_a", "neg_0_b"]
    pos1, neg1 = _concept_mode_filter(rows, 1)
    assert pos1 == ["pos_1_a"]
    assert neg1 == ["neg_1_a"]


def test_concept_mode_filter_ignores_other_concepts_avoiding():
    """A ``concept_avoiding`` row keyed to another concept must NOT leak
    into this concept's negatives (no global-pool fall-through)."""
    rows = [
        _synth_row(0, "concept_rich", "pos_0"),
        _synth_row(0, "concept_avoiding", "neg_0"),
        _synth_row(1, "concept_avoiding", "neg_1"),
    ]
    pos, neg = _concept_mode_filter(rows, 0)
    assert pos == ["pos_0"]
    assert neg == ["neg_0"]  # neg_1 belongs to concept 1, excluded


def test_concept_mode_filter_identical_in_both_modes():
    """The same filter behaves identically on a positive-mode corpus and
    a negative-mode corpus: both are now per-concept rich/avoiding pairs.
    """
    pos_mode_rows = [
        _synth_row(7, "concept_rich", "rich_7"),
        _synth_row(7, "concept_avoiding", "avoid_7"),
    ]
    neg_mode_rows = [
        _synth_row(7, "concept_rich", "rich_7"),
        _synth_row(7, "concept_avoiding", "avoid_7"),
    ]
    assert _concept_mode_filter(pos_mode_rows, 7) == (["rich_7"], ["avoid_7"])
    assert _concept_mode_filter(neg_mode_rows, 7) == (["rich_7"], ["avoid_7"])


def test_raise_on_empty_synth_rows_raises_on_empty_rich():
    """An empty/whitespace concept_rich text raises loudly (invariant violation:
    empty rows must be rejected at synth QC time, never reach Stage 2)."""
    with pytest.raises(AxBenchBaseError, match="1 empty/whitespace concept_rich"):
        _raise_on_empty_synth_rows(["a", "", "b"], ["n"], concept_id=7)


def test_raise_on_empty_synth_rows_raises_on_empty_avoiding():
    with pytest.raises(
        AxBenchBaseError, match="empty/whitespace concept_avoiding",
    ):
        _raise_on_empty_synth_rows(["a"], ["n", "   "], concept_id=3)


def test_raise_on_empty_synth_rows_ok_when_all_nonempty():
    # No raise when every text has content (internal whitespace is fine).
    _raise_on_empty_synth_rows(["a", "b c"], ["n"], concept_id=0)
    _raise_on_empty_synth_rows([], [], concept_id=0)


@pytest.fixture
def ranked_env(monkeypatch, tmp_path):
    """Stub out the SAE forward + percentile table for delta-ranked tests."""
    n_features = 16
    state: dict[str, Any] = {
        "n_features": n_features,
        # Map text -> dense feature vector (max-pooled). Defaults to zeros.
        "dense_vectors": {},
        "percentile_table": t.zeros(n_features, 99, dtype=t.float64),
        "dense_calls": [],
        "tokenizer_ids": [],
    }

    def _fake_batched(
        texts, base_model, base_sae, *, sae_layer, tokenizer_id, batch_size=8,
    ):
        del base_model, base_sae, batch_size, sae_layer
        state["dense_calls"].append(list(texts))
        state["tokenizer_ids"].append(tokenizer_id)
        if not texts:
            return t.zeros(0, state["n_features"])
        rows = []
        for text in texts:
            vec = state["dense_vectors"].get(
                text, t.zeros(state["n_features"]),
            )
            rows.append(vec)
        return t.stack(rows, dim=0)

    def _fake_table(stats_path, n_sae_features):
        del stats_path, n_sae_features
        return state["percentile_table"]

    monkeypatch.setattr(
        concept_feature_mapping, "_batched_dense_sae_max_pool", _fake_batched,
    )
    monkeypatch.setattr(
        concept_feature_mapping, "build_percentile_table_from_stats", _fake_table,
    )
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    return state


def _monotonic_table(n_features: int, scale: float = 1.0) -> t.Tensor:
    """Build a percentile table where each row is [1..99] scaled by a constant.

    This lets tests use activation magnitudes as "easy" mappings to
    percentile values.
    """
    row = t.arange(1, 100, dtype=t.float64) * scale
    return row.unsqueeze(0).repeat(n_features, 1)


def test_delta_ranked_computes_ranked_file(
    ranked_env, tmp_path, monkeypatch,
):
    """End-to-end: stub the SAE pass, verify the ranked output schema."""
    n_features = 16
    # Use a monotonic table so magnitudes trivially become percentiles.
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    # Concept 0's canonical feature is feature 3.
    # Arrange pos/neg so delta favors feature 3 strongly.
    pos_vec = t.zeros(n_features)
    pos_vec[3] = 50.0  # mid range
    neg_vec = t.zeros(n_features)
    ranked_env["dense_vectors"]["pos_text"] = pos_vec
    ranked_env["dense_vectors"]["neg_text"] = neg_vec

    # Write synth parquet.
    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    out_pos, out_neg = concept_feature_mapping_fn(
        cfg, concepts, pos_path,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    assert out_pos.exists()
    assert out_neg is None
    _tensors, metadata = _read_safetensors(out_pos)
    assert metadata["mode"] == "delta_ranked"
    assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY] == cfg.extraction_tokenizer_id
    tensors, metadata = _read_safetensors(out_pos)
    assert metadata["mode"] == "delta_ranked"
    # Feature 3 should be ranked first because it's the only one with positive delta.
    assert int(tensors["feature_ids"][0, 0]) == 3
    assert int(tensors["canonical_feature_id"][0]) == 3
    assert int(tensors["canonical_rank"][0]) == 0
    # Shape checks
    assert tensors["feature_ids"].shape == (1, cfg.n_rank_store)
    assert tensors["percentile_deltas"].shape == (1, cfg.n_rank_store)


def test_delta_ranked_raises_on_empty_synth_row(ranked_env, tmp_path):
    """#602 finding 3 (LOUD): empty output_text is rejected at synth QC time, so
    an empty concept_rich row reaching Stage 2 is an invariant violation — the
    delta driver RAISES loudly rather than silently dropping it (which would hide
    the upstream bug that let it through).
    """
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", ""),           # invariant violation
        _synth_row(0, "concept_rich", "real rich"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    with pytest.raises(
        AxBenchBaseError, match="empty/whitespace concept_rich synth row",
    ):
        concept_feature_mapping_fn(
            cfg, concepts, pos_path,
            base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
            target_model_meta=_stub_meta(),
        )


def test_delta_ranked_canonical_rank_when_in_top_n(
    ranked_env, tmp_path,
):
    """Concept canonical feature lands at rank 2 in the delta-sorted list."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    # Arrange vectors so feature 5 has the largest delta, 7 second, concept's
    # canonical feature 3 third.
    pos_vec = t.zeros(n_features)
    pos_vec[5] = 80.0
    pos_vec[7] = 60.0
    pos_vec[3] = 40.0
    neg_vec = t.zeros(n_features)
    ranked_env["dense_vectors"]["pos_text"] = pos_vec
    ranked_env["dense_vectors"]["neg_text"] = neg_vec

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=8,
    )
    concepts = [_concept(0, 3)]

    out_pos, _ = concept_feature_mapping_fn(
        cfg, concepts, pos_path,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    tensors, _ = _read_safetensors(out_pos)
    # Canonical feature 3 should land at rank 2 (0:5, 1:7, 2:3).
    assert int(tensors["canonical_rank"][0]) == 2
    assert int(tensors["feature_ids"][0, 0]) == 5
    assert int(tensors["feature_ids"][0, 1]) == 7
    assert int(tensors["feature_ids"][0, 2]) == 3


def test_delta_ranked_canonical_rank_when_not_in_top_n(
    ranked_env, tmp_path,
):
    """Canonical feature not in top-N -> rank is set to sentinel n_rank_store."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    # Fill many features with high deltas; canonical feature 3 is never activated.
    pos_vec = t.zeros(n_features)
    for i in range(n_features):
        if i != 3:
            pos_vec[i] = float(90 - i)  # Descending magnitudes, but all > 0.
    neg_vec = t.zeros(n_features)
    ranked_env["dense_vectors"]["pos_text"] = pos_vec
    ranked_env["dense_vectors"]["neg_text"] = neg_vec

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    # Tiny n_rank_store so feature 3 can't fit.
    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=2,
    )
    concepts = [_concept(0, 3)]

    out_pos, _ = concept_feature_mapping_fn(
        cfg, concepts, pos_path,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    tensors, _ = _read_safetensors(out_pos)
    # Canonical feature 3 was NOT in the top-2; rank should be sentinel.
    assert int(tensors["canonical_rank"][0]) == cfg.n_rank_store


def test_delta_ranked_missing_rich_for_authoritative_concept_raises(
    ranked_env, tmp_path,
):
    """Corpus presence cannot silently remove a concept from the final list."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    pos_path = tmp_path / "pos.parquet"
    # No rows for concept 0.
    rows = [
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    with pytest.raises(
        AxBenchBaseError,
        match="authoritative filtered concept_id=0 is missing concept_rich",
    ):
        concept_feature_mapping_fn(
            cfg,
            concepts,
            pos_path,
            base_model=_fake_gemma(),
            base_sae=_fake_sae(n_features),
            target_model_meta=_stub_meta(),
        )


def test_delta_ranked_missing_avoiding_for_authoritative_concept_raises(
    ranked_env, tmp_path,
):
    """A final concept cannot receive a fabricated zero negative baseline."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    pos_vec = t.zeros(n_features)
    pos_vec[3] = 80.0
    ranked_env["dense_vectors"]["pos_text"] = pos_vec

    pos_path = tmp_path / "pos.parquet"
    rows = [_synth_row(0, "concept_rich", "pos_text")]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    with pytest.raises(
        AxBenchBaseError,
        match="authoritative filtered concept_id=0 is missing concept_avoiding",
    ):
        concept_feature_mapping_fn(
            cfg,
            concepts,
            pos_path,
            base_model=_fake_gemma(),
            base_sae=_fake_sae(n_features),
            target_model_meta=_stub_meta(),
        )


def test_delta_ranked_preserves_sparse_original_concept_ids(
    ranked_env,
    tmp_path,
):
    """Delta-ranked tensors retain sentinel rows for filtered-out IDs."""

    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    for concept_id, feature_id in ((0, 3), (2, 5)):
        rich_text = f"rich-{concept_id}"
        avoid_text = f"avoid-{concept_id}"
        rich = t.zeros(n_features)
        rich[feature_id] = 80.0
        ranked_env["dense_vectors"][rich_text] = rich
        ranked_env["dense_vectors"][avoid_text] = t.zeros(n_features)

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "rich-0"),
        _synth_row(0, "concept_avoiding", "avoid-0"),
        _synth_row(2, "concept_rich", "rich-2"),
        _synth_row(2, "concept_avoiding", "avoid-2"),
    ]
    _write_synth_parquet(pos_path, rows)
    cfg = _cfg(
        tmp_path,
        concept_feature_mode="delta_ranked",
        run_negative_mode=False,
        n_rank_store=4,
    )

    out_pos, _ = concept_feature_mapping_fn(
        cfg,
        [_concept(0, 3), _concept(2, 5)],
        pos_path,
        base_model=_fake_gemma(),
        base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    tensors, _ = _read_safetensors(out_pos)

    assert tensors["feature_ids"].shape == (3, 4)
    assert int(tensors["feature_ids"][0, 0]) == 3
    assert bool((tensors["feature_ids"][1] == -1).all())
    assert int(tensors["feature_ids"][2, 0]) == 5
    assert bool(t.isnan(tensors["percentile_deltas"][1]).all())
    assert tensors["canonical_feature_id"].tolist() == [3, -1, 5]
    assert tensors["canonical_rank"].tolist() == [0, 4, 0]


def test_n_rank_store_respected(ranked_env, tmp_path):
    """Stored top-N count matches cfg.n_rank_store."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    pos_vec = t.arange(n_features, dtype=t.float32).float() + 1.0
    neg_vec = t.zeros(n_features)
    ranked_env["dense_vectors"]["pos_text"] = pos_vec
    ranked_env["dense_vectors"]["neg_text"] = neg_vec

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=5,
    )
    concepts = [_concept(0, 3)]

    out_pos, _ = concept_feature_mapping_fn(
        cfg, concepts, pos_path,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    tensors, _ = _read_safetensors(out_pos)
    assert tensors["feature_ids"].shape == (1, 5)
    assert tensors["percentile_deltas"].shape == (1, 5)


def test_metadata_mode_delta_ranked(ranked_env, tmp_path):
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    ranked_env["dense_vectors"]["pos"] = t.zeros(n_features)
    ranked_env["dense_vectors"]["neg"] = t.zeros(n_features)

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos"),
        _synth_row(0, "concept_avoiding", "neg", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    out_pos, _ = concept_feature_mapping_fn(
        cfg, concepts, pos_path,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    _, metadata = _read_safetensors(out_pos)
    assert metadata["mode"] == "delta_ranked"


def test_delta_ranked_writes_neg_file(ranked_env, tmp_path):
    """Delta-ranked + run_negative_mode writes the neg file using the
    collapsed per-concept filter for both modes."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    pos_vec = t.zeros(n_features)
    pos_vec[3] = 80.0
    neg_vec = t.zeros(n_features)
    for text in ["pos_a", "pos_b", "neg_pos_a", "neg_pos_b", "neg_neg_a"]:
        ranked_env["dense_vectors"][text] = t.zeros(n_features)
    ranked_env["dense_vectors"]["pos_a"] = pos_vec
    ranked_env["dense_vectors"]["neg_pos_a"] = pos_vec

    pos_path = tmp_path / "pos.parquet"
    neg_path = tmp_path / "neg.parquet"
    pos_rows = [
        _synth_row(0, "concept_rich", "pos_a"),
        _synth_row(0, "concept_rich", "pos_b"),
        _synth_row(0, "concept_avoiding", "pos_avoid", category="negative"),
    ]
    _write_synth_parquet(pos_path, pos_rows)
    ranked_env["dense_vectors"]["pos_avoid"] = neg_vec

    neg_rows = [
        _synth_row(0, "concept_rich", "neg_pos_a"),
        _synth_row(0, "concept_avoiding", "neg_neg_a"),
    ]
    _write_synth_parquet(neg_path, neg_rows)
    ranked_env["dense_vectors"]["neg_neg_a"] = neg_vec

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=True, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    out_pos, out_neg = concept_feature_mapping_fn(
        cfg, concepts, pos_path,
        synth_training_negative_path=neg_path,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    assert out_pos.exists()
    assert out_neg is not None
    assert out_neg.exists()
    _, meta_neg = _read_safetensors(out_neg)
    assert meta_neg["mode"] == "delta_ranked"


def test_missing_synth_parquet_raises(tmp_path):
    """_load_synth_rows raises when the parquet doesn't exist."""
    with pytest.raises(AxBenchBaseError, match="not found"):
        _load_synth_rows(
            tmp_path / "nonexistent.parquet",
            cfg=_cfg(tmp_path),
            continuation_role="stage1b_positive",
        )


def test_load_synth_rows_rejects_missing_tokenizer_metadata(tmp_path):
    path = tmp_path / "synth.parquet"
    rows = [_synth_row(0, "concept_rich", "pos")]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TRAINING_SCHEMA), path)

    with pytest.raises(
        AxBenchBaseError,
        match="Stage 2: .*missing 'extraction_tokenizer_id' metadata",
    ):
        _load_synth_rows(
            path,
            cfg=_cfg(
                tmp_path,
                extraction_tokenizer_id="custom/current-tokenizer",
            ),
            continuation_role="stage1b_positive",
        )


def test_load_synth_rows_rejects_mismatched_tokenizer_metadata(tmp_path):
    path = tmp_path / "synth.parquet"
    _write_synth_parquet(
        path,
        [_synth_row(0, "concept_rich", "pos")],
        tokenizer_id="custom/stale-tokenizer",
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Stage 2: .*custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        _load_synth_rows(
            path,
            cfg=_cfg(
                tmp_path,
                extraction_tokenizer_id="custom/current-tokenizer",
            ),
            continuation_role="stage1b_positive",
        )


def test_load_positive_synth_rows_rejects_legacy_qc_contract(tmp_path):
    path = tmp_path / "positive.parquet"
    schema = schema_with_extraction_tokenizer_metadata(
        TRAINING_SCHEMA,
        "custom/current-tokenizer",
    )
    schema = schema_with_stage1_continuation_contract(
        schema,
        role="stage1b_positive",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_synth_row(0, "concept_rich", "pos")],
            schema=schema,
        ),
        path,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"Stage 2.*invalid or legacy.*fresh run_tag",
    ):
        _load_synth_rows(
            path,
            cfg=_cfg(
                tmp_path,
                extraction_tokenizer_id="custom/current-tokenizer",
                accept_threshold=0.3,
            ),
            continuation_role="stage1b_positive",
        )


def test_load_positive_synth_rows_rejects_threshold_mismatch(tmp_path):
    path = tmp_path / "positive.parquet"
    _write_synth_parquet(path, [_synth_row(0, "concept_rich", "pos")])

    with pytest.raises(
        AxBenchBaseError,
        match=r"Stage 2.*artifact threshold does not match.*current fraction=0.5",
    ):
        _load_synth_rows(
            path,
            cfg=_cfg(tmp_path, accept_threshold=0.5),
            continuation_role="stage1b_positive",
        )


def test_run_negative_mode_without_neg_path_raises(
    ranked_env, tmp_path,
):
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    ranked_env["dense_vectors"]["pos"] = t.zeros(n_features)
    ranked_env["dense_vectors"]["neg"] = t.zeros(n_features)

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos"),
        _synth_row(0, "concept_avoiding", "neg", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=True, n_rank_store=4,
    )
    concepts = [_concept(0, 3)]

    with pytest.raises(AxBenchBaseError, match="synth_training_negative_path is None"):
        concept_feature_mapping_fn(
            cfg, concepts, pos_path,
            synth_training_negative_path=None,
            base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
            target_model_meta=_stub_meta(),
        )


# ---------------------------------------------------------------------------
# _batched_dense_sae_max_pool
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Whitespace tokenizer stub matching the real extraction tokenizer
    interface. Exposes ``bos_token_id = 0``, disjoint from every positional
    content id (which start at 1), so
    :func:`~steering.segmentation.resolve_bos_token_id` resolves a real BOS
    and callers that inject-and-exclude a BOS (e.g.
    ``_batched_dense_sae_max_pool``) are exercised the same way a real
    Gemma tokenizer would drive them.
    """

    bos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        return [i + 1 for i, _ in enumerate(text.split())]


def test_batched_dense_sae_max_pool_empty_texts_returns_empty(monkeypatch):
    result = _batched_dense_sae_max_pool(
        [], base_model=object(), base_sae=_fake_sae(8), sae_layer=8,
    )
    assert result.shape == (0, 8)


def test_batched_dense_sae_max_pool_loads_requested_tokenizer_id(monkeypatch):
    seen: list[str] = []

    def _fake_tokenizer(tokenizer_id: str) -> _StubTokenizer:
        seen.append(tokenizer_id)
        return _StubTokenizer()

    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        _fake_tokenizer,
    )
    monkeypatch.setattr(
        concept_feature_mapping,
        "_run_batched_extractor",
        lambda **kwargs: [
            t.ones((len(tokens), 8), dtype=t.float32)
            for tokens in kwargs["token_id_lists"]
        ],
    )

    result = _batched_dense_sae_max_pool(
        ["alpha beta"],
        base_model=_fake_gemma(),
        base_sae=_fake_sae(8),
        sae_layer=8,
        tokenizer_id="custom/stage2-tokenizer",
    )

    assert result.shape == (1, 8)
    assert seen == ["custom/stage2-tokenizer"]


def test_batched_dense_sae_max_pool_batches(monkeypatch):
    """Passing texts beyond batch_size calls the extractor multiple times."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    call_records: list[dict[str, Any]] = []

    def _fake_extractor(**kwargs):
        call_records.append(kwargs)
        n_lists = len(kwargs["token_id_lists"])
        return [t.ones(len(toks), 8) for toks in kwargs["token_id_lists"][:n_lists]]

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _fake_extractor,
    )
    # 5 texts, batch_size=2 -> 3 calls.
    texts = ["a", "b b", "c c c", "d d d d", "e e e e e"]
    result = _batched_dense_sae_max_pool(
        texts, base_model=_fake_gemma(), base_sae=_fake_sae(8),
        sae_layer=8, batch_size=2,
    )
    assert len(call_records) == 3
    assert result.shape == (5, 8)
    assert (result == 1.0).all()
    # sae_layer is forwarded to the extractor.
    for rec in call_records:
        assert rec["layer"] == 8


def test_batched_dense_sae_max_pool_clamps_to_base_model_context(monkeypatch):
    """Delta-ranked extraction must not forward over-context token lists.

    The injected BOS occupies one of the ``n_ctx`` slots: raw tokens are
    ``[1, 2, 3, 4]``, BOS-prepended is ``[0, 1, 2, 3, 4]``, and clamping to
    ``n_ctx=2`` keeps ``[0, 1]`` (BOS + first content token) -- NOT the
    pre-fix ``[1, 2]`` (first two content tokens, no BOS).
    """
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    call_records: list[dict[str, Any]] = []

    def _fake_extractor(**kwargs):
        call_records.append(kwargs)
        return [t.ones(len(toks), 8) for toks in kwargs["token_id_lists"]]

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _fake_extractor,
    )

    result = _batched_dense_sae_max_pool(
        ["alpha beta gamma delta"],
        base_model=SimpleNamespace(cfg=SimpleNamespace(n_ctx=2)),
        base_sae=_fake_sae(8),
        sae_layer=8,
    )

    assert result.shape == (1, 8)
    assert call_records[0]["token_id_lists"] == [[0, 1]]


def test_batched_dense_sae_max_pool_rejects_invalid_context_window(monkeypatch):
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )

    with pytest.raises(AxBenchBaseError, match="n_ctx must be positive"):
        _batched_dense_sae_max_pool(
            ["alpha"],
            base_model=SimpleNamespace(cfg=SimpleNamespace(n_ctx=0)),
            base_sae=_fake_sae(8),
            sae_layer=8,
        )


def test_batched_dense_sae_max_pool_rejects_extractor_cardinality_mismatch(
    monkeypatch,
):
    """A short/long extractor result would corrupt row ordering; fail loudly."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    monkeypatch.setattr(
        concept_feature_mapping,
        "_run_batched_extractor",
        lambda **kwargs: [],
    )

    with pytest.raises(
        AxBenchBaseError,
        match="batched extractor returned 0 activation tensors for 1 token lists",
    ):
        _batched_dense_sae_max_pool(
            ["alpha beta"],
            base_model=_fake_gemma(),
            base_sae=_fake_sae(8),
            sae_layer=8,
        )


def test_batched_dense_sae_max_pool_forwards_custom_sae_layer(monkeypatch):
    """Custom sae_layer is honored, not the previously-hardcoded 8."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    call_records: list[dict[str, Any]] = []

    def _fake_extractor(**kwargs):
        call_records.append(kwargs)
        return [t.ones(len(toks), 8) for toks in kwargs["token_id_lists"]]

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _fake_extractor,
    )
    _batched_dense_sae_max_pool(
        ["hello world"], base_model=_fake_gemma(), base_sae=_fake_sae(8),
        sae_layer=12,
    )
    assert call_records[0]["layer"] == 12


def test_batched_dense_sae_max_pool_raises_on_empty_concept_text(monkeypatch):
    """A concept text that tokenizes to [] RAISES (concept texts must be
    non-empty) rather than emitting a silent zero feature-row that would
    understate the concept."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )

    def _fake_extractor(**kwargs):
        return [
            t.full((len(toks), 8), 2.0, dtype=t.float32)
            for toks in kwargs["token_id_lists"]
        ]

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _fake_extractor,
    )
    # A batch containing an empty concept text raises (no silent zero-row).
    texts = ["a b", "", "c d e"]
    with pytest.raises(
        AxBenchBaseError, match="concept text tokenized to zero tokens"
    ):
        _batched_dense_sae_max_pool(
            texts, base_model=_fake_gemma(), base_sae=_fake_sae(8),
            sae_layer=8, batch_size=4,
        )


def test_batched_dense_sae_max_pool_all_empty_batch_raises(monkeypatch):
    """An all-empty batch raises on the first empty concept text, BEFORE the
    extractor is ever called."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    call_count = {"n": 0}

    def _fake_extractor(**kwargs):
        call_count["n"] += 1
        return [t.ones(len(toks), 8) for toks in kwargs["token_id_lists"]]

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _fake_extractor,
    )
    texts = ["", "", ""]
    with pytest.raises(
        AxBenchBaseError, match="concept text tokenized to zero tokens"
    ):
        _batched_dense_sae_max_pool(
            texts, base_model=object(), base_sae=_fake_sae(8),
            sae_layer=8, batch_size=4,
        )
    # Raised before the extractor was ever called.
    assert call_count["n"] == 0


def test_batched_dense_sae_max_pool_returns_fp32_when_sae_emits_bf16(
    monkeypatch,
) -> None:
    """After ``_load_feature_extraction_pair`` casts the SAE to bf16
    (runtime_state.py:389), ``_run_batched_extractor`` returns bf16
    activations. Downstream consumers (delta_ranked mean, percentile
    lookup, Stage 3 DiffMean) need fp32 to avoid silently degrading
    baselines — bf16 has ~3 decimal digits of precision, not enough
    for stable mean-over-many-texts accumulation.

    Pin the fp32-cast boundary at the SAE/downstream edge so a future
    change removing the cast fails loudly here.
    """
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )

    def _fake_extractor_bf16(**kwargs):
        # Emulate the production path: bf16 SAE → bf16 activations.
        return [
            t.ones(len(toks), 8, dtype=t.bfloat16)
            for toks in kwargs["token_id_lists"]
        ]

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _fake_extractor_bf16,
    )
    texts = ["hello", "world"]
    result = _batched_dense_sae_max_pool(
        texts, base_model=_fake_gemma(), base_sae=_fake_sae(8),
        sae_layer=8, batch_size=4,
    )
    assert result.dtype == t.float32, (
        f"expected fp32 at the SAE/downstream boundary, got {result.dtype}; "
        f"downstream mean + percentile lookup needs the extra precision"
    )
    # Shape sanity: same rows, same feature count.
    assert result.shape == (2, 8)


def test_batched_dense_sae_max_pool_zero_rows_are_fp32() -> None:
    """The empty-texts zero path (line 135) also emits fp32 so callers
    can rely on a single dtype regardless of the path taken. Without
    this the zero row would default to fp32 anyway (``t.zeros``
    default), but pinning the dtype keeps the contract explicit."""
    result = _batched_dense_sae_max_pool(
        [], base_model=object(), base_sae=_fake_sae(8),
        sae_layer=8, batch_size=4,
    )
    assert result.dtype == t.float32
    assert result.shape == (0, 8)


def test_batched_dense_injects_bos_per_text(monkeypatch):
    """_batched_dense_sae_max_pool injects one BOS per text and excludes it.

    Asserts on the literal ``token_id_lists`` the extractor receives (not
    just the pooled output) -- otherwise this test would pass vacuously
    against the pre-fix code too, since a feature that's only ever hot on
    an un-injected BOS is trivially "excluded". Mirrors
    ``test_dense_injects_bos_and_excludes_it`` in test_extraction.py.
    """
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    n_features = 8
    calls: list[list[list[int]]] = []

    def _extractor(**kwargs):
        calls.append(kwargs["token_id_lists"])
        out = []
        for ids in kwargs["token_id_lists"]:
            rows = t.zeros(len(ids), n_features)
            for pos, tid in enumerate(ids):
                rows[pos, tid % n_features] = 100.0
            out.append(rows)
        return out

    monkeypatch.setattr(
        concept_feature_mapping, "_run_batched_extractor", _extractor,
    )
    # _StubTokenizer.bos_token_id == 0 -> feature 0 % 8 == 0. Content
    # "alpha beta" -> tokens [1, 2] -> features 1, 2.
    result = _batched_dense_sae_max_pool(
        ["alpha beta"],
        base_model=_fake_gemma(),
        base_sae=_fake_sae(n_features),
        sae_layer=8,
    )
    # The extractor must see exactly one BOS (id 0) prepended per text.
    assert calls == [[[0, 1, 2]]]
    assert result.shape == (1, n_features)
    assert result[0, 1] == 100.0
    assert result[0, 2] == 100.0
    assert result[0, 0] == 0.0  # BOS-only feature must be excluded from the pool


def test_batched_dense_sae_max_pool_rejects_per_row_shape_mismatch(monkeypatch):
    """A row's acts not matching its BOS-injected token count would make the
    acts[1:] BOS-exclusion slice silently misaligned; fail loudly instead."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )
    monkeypatch.setattr(
        concept_feature_mapping,
        "_run_batched_extractor",
        # "alpha beta" -> BOS-injected model_input is [0, 1, 2] (3 tokens);
        # return only 2 rows to simulate a desynced extractor.
        lambda **kwargs: [t.ones(2, 8)],
    )
    with pytest.raises(
        AxBenchBaseError,
        match="extractor returned 2 rows for a 3-token",
    ):
        _batched_dense_sae_max_pool(
            ["alpha beta"], base_model=_fake_gemma(), base_sae=_fake_sae(8),
            sae_layer=8,
        )


def test_batched_dense_sae_max_pool_raises_when_clamp_leaves_only_bos(
    monkeypatch,
):
    """n_ctx=1 clamps away every content token, leaving only the injected
    BOS. Pooling ``acts[1:]`` on a 1-row tensor would be an empty-tensor
    ``max()`` crash, so guard loudly (and BEFORE calling the extractor)
    instead."""
    monkeypatch.setattr(
        concept_feature_mapping,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _StubTokenizer(),
    )

    def _boom(**kwargs):
        raise AssertionError(
            "extractor must not be called once the BOS-only clamp guard "
            "should have raised first"
        )

    monkeypatch.setattr(concept_feature_mapping, "_run_batched_extractor", _boom)
    with pytest.raises(AxBenchBaseError, match="no content"):
        _batched_dense_sae_max_pool(
            ["alpha beta"],
            base_model=SimpleNamespace(cfg=SimpleNamespace(n_ctx=1)),
            base_sae=_fake_sae(8),
            sae_layer=8,
        )


# ---------------------------------------------------------------------------
# Direct delta_ranked helper (bypassing the entry-point for error paths)
# ---------------------------------------------------------------------------


def test_concept_feature_mapping_delta_ranked_direct(
    ranked_env, tmp_path,
):
    """Call the delta_ranked helper directly without going through the dispatcher."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    pos_vec = t.zeros(n_features)
    pos_vec[3] = 80.0
    ranked_env["dense_vectors"]["pos_text"] = pos_vec
    ranked_env["dense_vectors"]["neg_text"] = t.zeros(n_features)

    pos_path = tmp_path / "pos.parquet"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    _write_synth_parquet(pos_path, rows)

    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    out_pos, out_neg = concept_feature_mapping_delta_ranked(
        cfg, [_concept(0, 3)], pos_path, None,
        base_model=_fake_gemma(), base_sae=_fake_sae(n_features),
        target_model_meta=_stub_meta(),
    )
    assert out_pos.exists()
    assert out_neg is None


def test_concept_feature_mapping_canonical_direct(
    tmp_path, monkeypatch,
):
    """Call the canonical helper directly."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    out_pos, out_neg = concept_feature_mapping_canonical(
        cfg, [_concept(0, 42)],
    )
    assert out_pos.exists()
    assert out_neg is None


def test_clustered_canonical_mapping_is_immutable_and_exact_replay_only(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=True)
    consolidation = SimpleNamespace(
        assert_representatives_only=lambda _ids, *, surface: None,
        feature_mask=None,
    )
    contract = cast(
        Any,
        SimpleNamespace(
            consolidation=consolidation,
            feature_contract_fingerprint="a" * 64,
        ),
    )

    pos, neg = concept_feature_mapping_canonical(
        cfg,
        [_concept(0, 42)],
        cluster_contract=contract,
    )
    assert neg is not None
    _, pos_metadata = _read_safetensors(pos)
    _, neg_metadata = _read_safetensors(neg)
    assert (
        pos_metadata["feature_contract_fingerprint"]
        == contract.feature_contract_fingerprint
    )
    assert (
        neg_metadata["feature_contract_fingerprint"]
        == contract.feature_contract_fingerprint
    )
    original = {pos: pos.read_bytes(), neg: neg.read_bytes()}

    replay_pos, replay_neg = concept_feature_mapping_canonical(
        cfg,
        [_concept(0, 42)],
        cluster_contract=contract,
    )
    assert (replay_pos, replay_neg) == (pos, neg)
    assert {path: path.read_bytes() for path in original} == original

    pos.write_bytes(b"different existing bytes")
    with pytest.raises(AxBenchBaseError, match="refusing to overwrite") as exc_info:
        concept_feature_mapping_canonical(
            cfg,
            [_concept(0, 42)],
            cluster_contract=contract,
        )
    assert isinstance(exc_info.value.__cause__, ImmutableArtifactError)
    assert pos.read_bytes() == b"different existing bytes"
    assert neg.read_bytes() == original[neg]


def test_clustered_canonical_mapping_rejects_noncanonical_mode(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, concept_feature_mode="delta_ranked")
    contract = cast(
        Any,
        SimpleNamespace(
            consolidation=MagicMock(feature_mask=None),
            feature_contract_fingerprint="a" * 64,
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="requires concept_feature_mode=canonical",
    ):
        concept_feature_mapping_canonical(
            cfg,
            [_concept(0, 42)],
            cluster_contract=contract,
        )


def test_clustered_canonical_mapping_checks_feature_mask(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_negative_mode=False)
    feature_mask = SimpleNamespace(assert_no_masked=MagicMock())
    consolidation = SimpleNamespace(
        feature_mask=feature_mask,
        assert_representatives_only=MagicMock(),
    )
    contract = cast(
        Any,
        SimpleNamespace(
            consolidation=consolidation,
            feature_contract_fingerprint="a" * 64,
        ),
    )

    concept_feature_mapping_canonical(
        cfg,
        [_concept(0, 42)],
        cluster_contract=contract,
    )

    representative_ids = consolidation.assert_representatives_only.call_args.args[0]
    masked_ids = feature_mask.assert_no_masked.call_args.args[0]
    assert t.equal(representative_ids, t.tensor([42], dtype=t.int64))
    assert t.equal(masked_ids, representative_ids)


def test_compute_ranked_file_direct(ranked_env, tmp_path):
    """Call _compute_ranked_file directly to exercise per-concept internals."""
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    pos_vec = t.zeros(n_features)
    pos_vec[3] = 80.0
    ranked_env["dense_vectors"]["pos_text"] = pos_vec
    ranked_env["dense_vectors"]["neg_text"] = t.zeros(n_features)

    out_path = tmp_path / "direct.safetensors"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    cfg = _cfg(
        tmp_path, concept_feature_mode="delta_ranked",
        run_negative_mode=False, n_rank_store=4,
    )
    _compute_ranked_file(
        cfg, [_concept(0, 3)], rows, _concept_mode_filter,
        _fake_gemma(), _fake_sae(n_features), _monotonic_table(n_features),
        out_path,
        sae_layer=8,
    )
    assert out_path.exists()
    tensors, metadata = _read_safetensors(out_path)
    assert metadata["mode"] == "delta_ranked"
    assert int(tensors["feature_ids"][0, 0]) == 3


def test_compute_ranked_file_passes_cfg_tokenizer_id(ranked_env, tmp_path):
    n_features = 16
    ranked_env["percentile_table"] = _monotonic_table(n_features)
    ranked_env["dense_vectors"]["pos_text"] = t.zeros(n_features)
    ranked_env["dense_vectors"]["neg_text"] = t.zeros(n_features)

    out_path = tmp_path / "tokenizer_scoped.safetensors"
    rows = [
        _synth_row(0, "concept_rich", "pos_text"),
        _synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    cfg = _cfg(
        tmp_path,
        concept_feature_mode="delta_ranked",
        run_negative_mode=False,
        extraction_tokenizer_id="custom/stage2-tokenizer",
    )
    _compute_ranked_file(
        cfg, [_concept(0, 3)], rows, _concept_mode_filter,
        _fake_gemma(), _fake_sae(n_features), _monotonic_table(n_features),
        out_path,
        sae_layer=8,
    )

    assert ranked_env["tokenizer_ids"] == [
        "custom/stage2-tokenizer",
        "custom/stage2-tokenizer",
    ]
