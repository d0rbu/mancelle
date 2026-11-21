"""Stage 2 (concept -> feature mapping) end-to-end integration tests.

Exercises the two Stage 2 concept-feature producers:

* ``canonical`` — trivial slot-0 copy of each concept's canonical
  feature. No GPU work. Verifies both positive and negative output
  files + metadata tag.
* ``delta_ranked`` — runs base Gemma + SAE over concept_rich vs
  concept_avoiding synth rows and stores the top-N feature ids
  ranked by pos_pct - neg_pct. GPU plumbing is stubbed via
  monkey-patching ``_batched_dense_sae_max_pool`` and
  ``build_percentile_table_from_stats`` so the test runs on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors import safe_open

from steering.eval.axbench_base import (
    concept_feature_mapping as concept_feature_mapping_mod,
)
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.concept_feature_mapping import (
    concept_feature_mapping_canonical,
    concept_feature_mapping_delta_ranked,
)
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.negative_mode import NEG_TRAINING_SCHEMA
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    TargetModelMetadata,
)
from tests.axbench_base import paths as test_paths

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_target_meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _concept(concept_id: int, feature_index: int) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {concept_id}",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _pos_synth_row(
    concept_id: int,
    content_class: str,
    output_text: str,
    category: str = "positive",
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "category": category,
        "content_class": content_class,
        "prefix_text": "prefix",
        "output_text": output_text,
        "output_concept": "desc",
        "n_prefix_nanochat_tokens": 4,
        "gen_source": "test:stub",
        "gen_seed": 123,
        "qc_feature_id": -1,
        "qc_seed_percentile": float("nan"),
        "qc_output_percentile": float("nan"),
        "qc_delta": float("nan"),
        "qc_attempts": 1,
        "qc_accepted": True,
    }


def _neg_synth_row(
    concept_id: int,
    content_class: str,
    output_text: str,
) -> dict[str, Any]:
    base = _pos_synth_row(concept_id, content_class, output_text)
    base.update({
        "prefix_source_shard": 0,
        "prefix_source_doc": 0,
        "prefix_source_seg_idx": 0,
        "qc_zero_activation_verified": content_class == "concept_avoiding",
    })
    return base


def _write_positive_synth_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    tokenizer_id: str = "google/gemma-2-2b",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=schema_with_positive_qc_contract(
                schema_with_extraction_tokenizer_metadata(
                    TRAINING_SCHEMA,
                    tokenizer_id,
                ),
                accept_threshold=0.3,
            ),
        ),
        path,
    )


def _write_negative_synth_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    tokenizer_id: str = "google/gemma-2-2b",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=schema_with_extraction_tokenizer_metadata(
                NEG_TRAINING_SCHEMA,
                tokenizer_id,
            ),
        ),
        path,
    )


def _read_safetensors(
    path: Path,
) -> tuple[dict[str, t.Tensor], dict[str, str]]:
    tensors: dict[str, t.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        metadata = f.metadata() or {}
    return tensors, metadata


def _monotonic_percentile_table(n_features: int) -> t.Tensor:
    """Row j is [1..99] so magnitude == percentile trivially."""
    row = t.arange(1, 100, dtype=t.float64)
    return row.unsqueeze(0).repeat(n_features, 1)


# ---------------------------------------------------------------------------
# Canonical mode tests
# ---------------------------------------------------------------------------


def test_stage_2_canonical_mode_writes_concept_features(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    """Stage 2 canonical mode writes both pos + neg files with metadata."""
    cfg = make_valid_axbench_cfg(
        run_tag="stage2_canonical",
        n_rank_store=8,
        concept_feature_mode="canonical",
        run_negative_mode=True,
    )
    concepts = [_concept(0, 42), _concept(1, 99)]

    pos_path, neg_path = concept_feature_mapping_canonical(cfg, concepts)

    assert pos_path == test_paths.default_paths("stage2_canonical").concept_features()
    assert pos_path.exists()
    assert neg_path is not None
    assert neg_path == test_paths.default_paths("stage2_canonical").concept_features_neg()
    assert neg_path.exists()

    pos_tensors, pos_meta = _read_safetensors(pos_path)
    assert pos_meta["mode"] == "canonical"
    assert pos_tensors["feature_ids"].shape == (2, cfg.n_rank_store)
    assert int(pos_tensors["feature_ids"][0, 0]) == 42
    assert int(pos_tensors["feature_ids"][1, 0]) == 99
    assert pos_tensors["canonical_rank"].tolist() == [0, 0]
    # All other slots are -1.
    assert (pos_tensors["feature_ids"][:, 1:] == -1).all()

    neg_tensors, neg_meta = _read_safetensors(neg_path)
    assert neg_meta["mode"] == "canonical"
    # Canonical mode: positive and negative outputs identical.
    assert t.equal(pos_tensors["feature_ids"], neg_tensors["feature_ids"])


def test_stage_2_delta_ranked_mode_writes_concept_features(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 2 delta-ranked mode writes a ranked safetensors with metadata."""
    n_features = 16
    cfg = make_valid_axbench_cfg(
        run_tag="stage2_delta",
        n_rank_store=4,
        concept_feature_mode="delta_ranked",
        run_negative_mode=False,
        extraction_tokenizer_id="custom/stage2-tokenizer",
    )

    # Build a tiny positive-mode synth parquet with one complete owned pair.
    pos_rows = [
        _pos_synth_row(0, "concept_rich", "pos_text"),
        _pos_synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    pos_corpus = tmp_run_root / "pos.parquet"
    _write_positive_synth_parquet(
        pos_corpus,
        pos_rows,
        tokenizer_id=cfg.extraction_tokenizer_id,
    )

    # Stub the GPU pipeline: return a dense vector keyed by text.
    pos_vec = t.zeros(n_features)
    pos_vec[3] = 50.0
    neg_vec = t.zeros(n_features)
    dense_lookup: dict[str, t.Tensor] = {
        "pos_text": pos_vec,
        "neg_text": neg_vec,
    }
    seen_tokenizer_ids: list[str] = []

    def _fake_batched(
        texts: list[str], base_model: Any, base_sae: Any, *,
        sae_layer: int, tokenizer_id: str, batch_size: int = 8,
    ) -> t.Tensor:
        del base_model, base_sae, batch_size, sae_layer
        seen_tokenizer_ids.append(tokenizer_id)
        if not texts:
            return t.zeros(0, n_features)
        return t.stack([dense_lookup[text] for text in texts], dim=0)

    monkeypatch.setattr(
        concept_feature_mapping_mod, "_batched_dense_sae_max_pool", _fake_batched,
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod, "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: _monotonic_percentile_table(n_features),
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod, "_infer_n_sae_features",
        lambda base_sae: n_features,
    )

    concepts = [_concept(0, 3)]
    pos_path, neg_path = concept_feature_mapping_delta_ranked(
        cfg,
        concepts=concepts,
        synth_training_positive_path=pos_corpus,
        synth_training_negative_path=None,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert pos_path.exists()
    assert neg_path is None
    tensors, metadata = _read_safetensors(pos_path)
    assert metadata["mode"] == "delta_ranked"
    assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY] == cfg.extraction_tokenizer_id
    assert seen_tokenizer_ids
    assert set(seen_tokenizer_ids) == {cfg.extraction_tokenizer_id}
    # Feature 3 should win (only feature with positive pos_pct - neg_pct).
    assert int(tensors["feature_ids"][0, 0]) == 3
    assert int(tensors["canonical_feature_id"][0]) == 3
    assert int(tensors["canonical_rank"][0]) == 0


def test_stage_2_negative_mode_writes_concept_features_neg(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``run_negative_mode=True``, delta-ranked writes a _neg file."""
    n_features = 16
    cfg = make_valid_axbench_cfg(
        run_tag="stage2_delta_neg",
        n_rank_store=4,
        concept_feature_mode="delta_ranked",
        run_negative_mode=True,
        extraction_tokenizer_id="custom/stage2-tokenizer",
    )

    pos_rows = [
        _pos_synth_row(0, "concept_rich", "pos_text"),
        _pos_synth_row(0, "concept_avoiding", "neg_text", category="negative"),
    ]
    pos_corpus = tmp_run_root / "pos.parquet"
    _write_positive_synth_parquet(
        pos_corpus,
        pos_rows,
        tokenizer_id=cfg.extraction_tokenizer_id,
    )

    neg_rows = [
        _neg_synth_row(0, "concept_rich", "pos_text"),
        _neg_synth_row(0, "concept_avoiding", "avoiding_text"),
    ]
    neg_corpus = tmp_run_root / "neg.parquet"
    _write_negative_synth_parquet(
        neg_corpus,
        neg_rows,
        tokenizer_id=cfg.extraction_tokenizer_id,
    )

    pos_vec = t.zeros(n_features)
    pos_vec[3] = 50.0
    neg_vec = t.zeros(n_features)
    avoid_vec = t.zeros(n_features)
    avoid_vec[5] = 30.0
    dense_lookup: dict[str, t.Tensor] = {
        "pos_text": pos_vec,
        "neg_text": neg_vec,
        "avoiding_text": avoid_vec,
    }
    seen_tokenizer_ids: list[str] = []

    def _fake_batched(
        texts: list[str], base_model: Any, base_sae: Any, *,
        sae_layer: int, tokenizer_id: str, batch_size: int = 8,
    ) -> t.Tensor:
        del base_model, base_sae, batch_size, sae_layer
        seen_tokenizer_ids.append(tokenizer_id)
        if not texts:
            return t.zeros(0, n_features)
        return t.stack([dense_lookup[text] for text in texts], dim=0)

    monkeypatch.setattr(
        concept_feature_mapping_mod, "_batched_dense_sae_max_pool", _fake_batched,
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod, "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: _monotonic_percentile_table(n_features),
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod, "_infer_n_sae_features",
        lambda base_sae: n_features,
    )

    concepts = [_concept(0, 3)]
    pos_path, neg_path = concept_feature_mapping_delta_ranked(
        cfg,
        concepts=concepts,
        synth_training_positive_path=pos_corpus,
        synth_training_negative_path=neg_corpus,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert pos_path == test_paths.default_paths("stage2_delta_neg").concept_features()
    assert pos_path.exists()
    assert neg_path is not None
    assert neg_path == test_paths.default_paths("stage2_delta_neg").concept_features_neg()
    assert neg_path.exists()
    _, pos_meta = _read_safetensors(pos_path)
    _, neg_meta = _read_safetensors(neg_path)
    assert pos_meta["mode"] == "delta_ranked"
    assert neg_meta["mode"] == "delta_ranked"
    assert pos_meta[EXTRACTION_TOKENIZER_METADATA_KEY] == cfg.extraction_tokenizer_id
    assert neg_meta[EXTRACTION_TOKENIZER_METADATA_KEY] == cfg.extraction_tokenizer_id
    assert seen_tokenizer_ids
    assert set(seen_tokenizer_ids) == {cfg.extraction_tokenizer_id}


def test_stage_2_delta_ranked_rejects_missing_positive_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    cfg = make_valid_axbench_cfg(
        run_tag="stage2_missing_pos_tokenizer_metadata",
        concept_feature_mode="delta_ranked",
        run_negative_mode=False,
        extraction_tokenizer_id="custom/stage2-tokenizer",
    )
    pos_corpus = tmp_run_root / "pos_missing_metadata.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [_pos_synth_row(0, "concept_rich", "pos_text")],
            schema=TRAINING_SCHEMA,
        ),
        pos_corpus,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Stage 2: .*missing 'extraction_tokenizer_id' metadata",
    ):
        concept_feature_mapping_delta_ranked(
            cfg,
            concepts=[_concept(0, 3)],
            synth_training_positive_path=pos_corpus,
            synth_training_negative_path=None,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )


def test_stage_2_delta_ranked_rejects_mismatched_positive_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    cfg = make_valid_axbench_cfg(
        run_tag="stage2_mismatch_pos_tokenizer_metadata",
        concept_feature_mode="delta_ranked",
        run_negative_mode=False,
        extraction_tokenizer_id="custom/stage2-current-tokenizer",
    )
    pos_corpus = tmp_run_root / "pos_mismatch_metadata.parquet"
    _write_positive_synth_parquet(
        pos_corpus,
        [_pos_synth_row(0, "concept_rich", "pos_text")],
        tokenizer_id="custom/stage2-stale-tokenizer",
    )

    with pytest.raises(
        AxBenchBaseError,
        match="custom/stage2-stale-tokenizer.*custom/stage2-current-tokenizer",
    ):
        concept_feature_mapping_delta_ranked(
            cfg,
            concepts=[_concept(0, 3)],
            synth_training_positive_path=pos_corpus,
            synth_training_negative_path=None,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )


def test_stage_2_delta_ranked_rejects_mismatched_negative_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_valid_axbench_cfg(
        run_tag="stage2_mismatch_neg_tokenizer_metadata",
        concept_feature_mode="delta_ranked",
        run_negative_mode=True,
        extraction_tokenizer_id="custom/stage2-current-tokenizer",
    )
    pos_corpus = tmp_run_root / "pos.parquet"
    _write_positive_synth_parquet(
        pos_corpus,
        [
            _pos_synth_row(0, "concept_rich", "pos_text"),
            _pos_synth_row(0, "concept_avoiding", "neg_text", category="negative"),
        ],
        tokenizer_id=cfg.extraction_tokenizer_id,
    )
    neg_corpus = tmp_run_root / "neg_mismatch_metadata.parquet"
    _write_negative_synth_parquet(
        neg_corpus,
        [_neg_synth_row(0, "concept_rich", "pos_text")],
        tokenizer_id="custom/stage2-stale-tokenizer",
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: _monotonic_percentile_table(16),
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod,
        "_infer_n_sae_features",
        lambda base_sae: 16,
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod,
        "_compute_ranked_file",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="custom/stage2-stale-tokenizer.*custom/stage2-current-tokenizer",
    ):
        concept_feature_mapping_delta_ranked(
            cfg,
            concepts=[_concept(0, 3)],
            synth_training_positive_path=pos_corpus,
            synth_training_negative_path=neg_corpus,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )


def test_stage_2_canonical_and_delta_ranked_have_same_output_schema(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both modes must produce the same four tensor keys with matching shapes."""
    n_features = 16
    cfg_canonical = make_valid_axbench_cfg(
        run_tag="stage2_schema_canonical",
        n_rank_store=4,
        concept_feature_mode="canonical",
        run_negative_mode=False,
    )
    cfg_delta = make_valid_axbench_cfg(
        run_tag="stage2_schema_delta",
        n_rank_store=4,
        concept_feature_mode="delta_ranked",
        run_negative_mode=False,
    )

    concepts = [_concept(0, 3), _concept(1, 7)]

    canonical_pos, _ = concept_feature_mapping_canonical(
        cfg_canonical,
        concepts,
    )
    canonical_tensors, canonical_meta = _read_safetensors(canonical_pos)
    assert canonical_meta["mode"] == "canonical"
    expected_keys = {
        "feature_ids",
        "percentile_deltas",
        "canonical_feature_id",
        "canonical_rank",
    }
    assert set(canonical_tensors.keys()) == expected_keys

    # Run delta_ranked with one complete owned pair per concept.
    pos_rows = [
        _pos_synth_row(0, "concept_rich", "pos_text_0"),
        _pos_synth_row(0, "concept_avoiding", "neg_text_0", category="negative"),
        _pos_synth_row(1, "concept_rich", "pos_text_1"),
        _pos_synth_row(1, "concept_avoiding", "neg_text_1", category="negative"),
    ]
    pos_corpus = tmp_run_root / "schema_pos.parquet"
    _write_positive_synth_parquet(pos_corpus, pos_rows)

    dense: dict[str, t.Tensor] = {}
    for key in ["pos_text_0", "pos_text_1", "neg_text_0", "neg_text_1"]:
        dense[key] = t.zeros(n_features)
    dense["pos_text_0"][3] = 50.0
    dense["pos_text_1"][7] = 50.0

    monkeypatch.setattr(
        concept_feature_mapping_mod,
        "_batched_dense_sae_max_pool",
        lambda texts, base_model, base_sae, *, sae_layer, tokenizer_id, batch_size=8: (
            t.stack([dense[t_] for t_ in texts], dim=0)
            if texts
            else t.zeros(0, n_features)
        ),
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod, "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: _monotonic_percentile_table(n_features),
    )
    monkeypatch.setattr(
        concept_feature_mapping_mod, "_infer_n_sae_features",
        lambda base_sae: n_features,
    )

    delta_pos, _ = concept_feature_mapping_delta_ranked(
        cfg_delta,
        concepts,
        pos_corpus,
        None,
        object(),
        object(),
        _stub_target_meta(),
    )
    delta_tensors, delta_meta = _read_safetensors(delta_pos)
    assert delta_meta["mode"] == "delta_ranked"
    assert delta_meta[EXTRACTION_TOKENIZER_METADATA_KEY] == (
        cfg_delta.extraction_tokenizer_id
    )
    assert set(delta_tensors.keys()) == expected_keys
    for key in expected_keys:
        assert canonical_tensors[key].shape == delta_tensors[key].shape, (
            f"Shape mismatch for {key}: canonical "
            f"{canonical_tensors[key].shape} vs delta {delta_tensors[key].shape}"
        )
