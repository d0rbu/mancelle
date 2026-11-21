import json
from pathlib import Path

import pytest

from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
)
from steering.eval.axbench_base.validate import (
    load_and_validate_candidate_concepts,
    load_and_validate_concepts,
    validate_candidate_concepts,
    validate_concepts,
)


def _make_concept(concept_id: int, feature_index: int, description: str = "desc") -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=description,
        corpus_count=100,
        description_source="neuronpedia",
    )


def _make_candidate(feature_index: int, description: str = "desc") -> CandidateConcept:
    return CandidateConcept(
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=description,
        corpus_count=100,
        description_source="neuronpedia",
    )


# ---------------------------------------------------------------------------
# validate_concepts: in-memory invariant checks
# ---------------------------------------------------------------------------


def test_valid_concepts_pass():
    concepts = [
        _make_concept(0, 100, "first"),
        _make_concept(1, 200, "second"),
        _make_concept(2, 300, "third"),
    ]
    validate_concepts(concepts, n_sae_features=16384)


def test_empty_list_rejected():
    with pytest.raises(AxBenchBaseError, match="empty"):
        validate_concepts([], n_sae_features=16384)


def test_non_contiguous_concept_ids_rejected():
    concepts = [
        _make_concept(0, 100),
        _make_concept(5, 200),
        _make_concept(2, 300),
    ]
    with pytest.raises(AxBenchBaseError, match="contiguous"):
        validate_concepts(concepts, n_sae_features=16384)


def test_duplicate_feature_index_rejected():
    concepts = [
        _make_concept(0, 100),
        _make_concept(1, 100),
    ]
    with pytest.raises(AxBenchBaseError, match="more than once"):
        validate_concepts(concepts, n_sae_features=16384)


def test_out_of_bounds_feature_index_rejected():
    concepts = [_make_concept(0, 99999)]
    with pytest.raises(AxBenchBaseError, match="out of"):
        validate_concepts(concepts, n_sae_features=16384)


def test_unicode_description_passes():
    # Kharoshthi script sample (U+10A00-U+10A33)
    kharoshthi = "𐨀𐨤𐨬𐨟 𐨨𐨱𐨠𐨪𐨞"
    concepts = [_make_concept(0, 100, description=kharoshthi)]
    validate_concepts(concepts, n_sae_features=16384)


# ---------------------------------------------------------------------------
# load_and_validate_concepts: JSON-level stage-boundary reader
# ---------------------------------------------------------------------------


def _write_concepts_json(tmp_path, payload) -> Path:
    p = tmp_path / "concepts.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_and_validate_concepts_happy_path(tmp_path):
    payload = [
        {
            "concept_id": 0,
            "feature_index": 100,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "first",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        },
        {
            "concept_id": 1,
            "feature_index": 200,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "second",
            "corpus_count": 200,
            "description_source": "neuronpedia",
        },
    ]
    p = _write_concepts_json(tmp_path, payload)
    result = load_and_validate_concepts(p, n_sae_features=16384)
    assert len(result) == 2
    assert all(isinstance(c, Concept) for c in result)
    assert result[0].concept_id == 0
    assert result[0].feature_index == 100
    assert result[0].description == "first"
    assert result[0].layer == 8
    assert result[0].sae_source == "8-gemmascope-res-16k"
    assert result[0].corpus_count == 100
    assert result[0].description_source == "neuronpedia"
    assert result[1].feature_index == 200
    assert result[1].description == "second"


def test_load_and_validate_concepts_rejects_non_array(tmp_path):
    # Top-level JSON must be an array, not an object.
    p = _write_concepts_json(tmp_path, {"not": "a list"})
    with pytest.raises(AxBenchBaseError, match="must be a JSON array"):
        load_and_validate_concepts(p, n_sae_features=16384)


def test_load_and_validate_concepts_rejects_non_dict_item(tmp_path):
    # Each item must be an object.
    p = _write_concepts_json(tmp_path, [1, 2, 3])
    with pytest.raises(AxBenchBaseError, match="must be a JSON object"):
        load_and_validate_concepts(p, n_sae_features=16384)


def test_missing_required_key_rejected(tmp_path):
    # A Concept dataclass with a missing field cannot be constructed,
    # so the missing-key check now lives in the JSON loader.
    payload = [{"concept_id": 0, "feature_index": 100, "layer": 8}]  # no description etc.
    p = _write_concepts_json(tmp_path, payload)
    with pytest.raises(
        AxBenchBaseError,
        match="missing a required field|description",
    ):
        load_and_validate_concepts(p, n_sae_features=16384)


def test_load_and_validate_concepts_extra_field_rejected(tmp_path):
    # An unexpected key must surface via the TypeError from Concept.__init__.
    payload = [
        {
            "concept_id": 0,
            "feature_index": 100,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "ok",
            "corpus_count": 100,
            "description_source": "neuronpedia",
            "foo": "bar",
        }
    ]
    p = _write_concepts_json(tmp_path, payload)
    with pytest.raises(
        AxBenchBaseError,
        match="unexpected field|unexpected keyword argument",
    ):
        load_and_validate_concepts(p, n_sae_features=16384)


def test_load_and_validate_concepts_wraps_invalid_field_domain(tmp_path):
    payload = [
        {
            "concept_id": 0,
            "feature_index": True,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "ok",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        }
    ]
    p = _write_concepts_json(tmp_path, payload)
    with pytest.raises(AxBenchBaseError, match="invalid fields.*feature_index"):
        load_and_validate_concepts(p, n_sae_features=16384)


def test_load_and_validate_concepts_runs_validate_concepts(tmp_path):
    # Valid schema but non-contiguous concept_ids: validate_concepts
    # should fire after the Concept construction step.
    payload = [
        {
            "concept_id": 0,
            "feature_index": 100,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "first",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        },
        {
            "concept_id": 5,  # should be 1
            "feature_index": 200,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "second",
            "corpus_count": 200,
            "description_source": "neuronpedia",
        },
    ]
    p = _write_concepts_json(tmp_path, payload)
    with pytest.raises(AxBenchBaseError, match="contiguous"):
        load_and_validate_concepts(p, n_sae_features=16384)


# ---------------------------------------------------------------------------
# candidate_concepts.json pre-QC schema
# ---------------------------------------------------------------------------


def _write_candidate_concepts_json(tmp_path, payload) -> Path:
    p = tmp_path / "candidate_concepts.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_validate_candidate_concepts_happy_path():
    validate_candidate_concepts(
        [_make_candidate(100), _make_candidate(200)],
        n_sae_features=16384,
    )


def test_validate_candidate_concepts_rejects_duplicate_feature():
    with pytest.raises(AxBenchBaseError, match="more than once"):
        validate_candidate_concepts(
            [_make_candidate(100), _make_candidate(100)],
            n_sae_features=16384,
        )


def test_validate_candidate_concepts_rejects_out_of_bounds_feature():
    with pytest.raises(AxBenchBaseError, match="feature_index.*out of"):
        validate_candidate_concepts(
            [_make_candidate(16384)],
            n_sae_features=16384,
        )


def test_load_and_validate_candidate_concepts_happy_path(tmp_path):
    payload = [
        {
            "feature_index": 100,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "first",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        },
        {
            "feature_index": 200,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "second",
            "corpus_count": 200,
            "description_source": "neuronpedia",
        },
    ]
    p = _write_candidate_concepts_json(tmp_path, payload)
    result = load_and_validate_candidate_concepts(p, n_sae_features=16384)
    assert [c.feature_index for c in result] == [100, 200]
    assert all(isinstance(c, CandidateConcept) for c in result)


def test_load_and_validate_candidate_concepts_rejects_old_concept_schema(
    tmp_path,
):
    payload = [
        {
            "concept_id": 0,
            "feature_index": 100,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": "old schema",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        }
    ]
    p = _write_candidate_concepts_json(tmp_path, payload)
    with pytest.raises(AxBenchBaseError, match="re-run Stage 0a"):
        load_and_validate_candidate_concepts(p, n_sae_features=16384)


def test_load_and_validate_candidate_concepts_wraps_invalid_field_domain(
    tmp_path,
):
    payload = [
        {
            "feature_index": 100,
            "layer": False,
            "sae_source": "8-gemmascope-res-16k",
            "description": "ok",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        }
    ]
    p = _write_candidate_concepts_json(tmp_path, payload)
    with pytest.raises(AxBenchBaseError, match="invalid fields.*layer"):
        load_and_validate_candidate_concepts(p, n_sae_features=16384)
