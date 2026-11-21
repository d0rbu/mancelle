"""Unit tests for prompting descriptions writer (Task 26).

Covers ``steering.eval.axbench_base.baselines.prompting.write_prompting_descriptions``.

TDD design choices:
  * Keys in the written JSON are ``str(concept_id)`` (string form of the int).
  * Values are the ``description`` field from each ``Concept``.
  * Both modes written when ``cfg.run_negative_mode=True``; only positive when False.
  * ``cfg.run_tag is None`` → ``AxBenchBaseError``.
  * Parent directories are created with ``parents=True, exist_ok=True``.
  * Writing twice replaces the file (not appends).
  * Empty concept list produces an empty JSON object ``{}``.
  * Unicode is preserved (``ensure_ascii=False``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.eval.axbench_base.baselines.prompting import write_prompting_descriptions
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(concept_id: int, description: str) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id,
        layer=8,
        sae_source="test-sae",
        description=description,
        corpus_count=100,
        description_source="test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_writes_both_files_when_both_modes_enabled(tmp_path: Path, monkeypatch) -> None:
    """run_negative_mode=True → positive and negative files both created."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run1", run_negative_mode=True)
    concepts = [_make_concept(0, "the first concept"), _make_concept(1, "the second concept")]

    result = write_prompting_descriptions(cfg, concepts)

    pos_path = test_paths.default_paths("run1").prompting_descriptions(SteeringMode.POSITIVE)
    neg_path = test_paths.default_paths("run1").prompting_descriptions(SteeringMode.NEGATIVE)
    assert len(result) == 2
    assert pos_path in result
    assert neg_path in result
    assert pos_path.exists()
    assert neg_path.exists()


def test_writes_only_positive_when_neg_mode_off(tmp_path: Path, monkeypatch) -> None:
    """run_negative_mode=False → only positive file written."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run2", run_negative_mode=False)
    concepts = [_make_concept(5, "some concept")]

    result = write_prompting_descriptions(cfg, concepts)

    pos_path = test_paths.default_paths("run2").prompting_descriptions(SteeringMode.POSITIVE)
    neg_path = test_paths.default_paths("run2").prompting_descriptions(SteeringMode.NEGATIVE)
    assert len(result) == 1
    assert pos_path in result
    assert neg_path not in result
    assert pos_path.exists()
    assert not neg_path.exists()


def test_json_structure_is_concept_id_to_description(tmp_path: Path, monkeypatch) -> None:
    """Loaded JSON is {str(concept_id): description} for each concept."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run3", run_negative_mode=False)
    concepts = [
        _make_concept(0, "zero desc"),
        _make_concept(42, "forty-two desc"),
        _make_concept(99, "ninety-nine desc"),
    ]

    write_prompting_descriptions(cfg, concepts)

    pos_path = test_paths.default_paths("run3").prompting_descriptions(SteeringMode.POSITIVE)
    data = json.loads(pos_path.read_text(encoding="utf-8"))

    assert data == {
        "0": "zero desc",
        "42": "forty-two desc",
        "99": "ninety-nine desc",
    }


def test_unicode_descriptions_preserved(tmp_path: Path, monkeypatch) -> None:
    """ensure_ascii=False: café, résumé, naïve, and emoji survive the round-trip."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run4", run_negative_mode=False)
    descriptions = ["café latte", "résumé writing", "naïve approach", "rocket \U0001f680 fuel"]
    concepts = [_make_concept(i, d) for i, d in enumerate(descriptions)]

    write_prompting_descriptions(cfg, concepts)

    pos_path = test_paths.default_paths("run4").prompting_descriptions(SteeringMode.POSITIVE)
    raw_text = pos_path.read_text(encoding="utf-8")
    # Verify no escaping happened (genuine unicode in file)
    assert "café" in raw_text
    assert "résumé" in raw_text
    assert "naïve" in raw_text
    assert "\U0001f680" in raw_text

    data = json.loads(raw_text)
    for i, d in enumerate(descriptions):
        assert data[str(i)] == d


def test_run_tag_none_raises() -> None:
    """cfg.run_tag is None → AxBenchBaseError before any file I/O."""
    cfg = make_valid_config(run_tag=None)
    concepts = [_make_concept(0, "desc")]

    with pytest.raises(AxBenchBaseError, match="run_tag"):
        write_prompting_descriptions(cfg, concepts)


def test_empty_concepts_produces_empty_json(tmp_path: Path, monkeypatch) -> None:
    """Empty concept list → empty JSON object {} in the written file."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run5", run_negative_mode=False)

    result = write_prompting_descriptions(cfg, [])

    assert len(result) == 1
    pos_path = test_paths.default_paths("run5").prompting_descriptions(SteeringMode.POSITIVE)
    data = json.loads(pos_path.read_text(encoding="utf-8"))
    assert data == {}


def test_parent_dir_created_if_missing(tmp_path: Path, monkeypatch) -> None:
    """Parent directory is created automatically even when absent."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="brand_new_run", run_negative_mode=False)
    concepts = [_make_concept(1, "test")]

    pos_path = test_paths.default_paths("brand_new_run").prompting_descriptions(SteeringMode.POSITIVE)
    # Confirm parent does not exist yet
    assert not pos_path.parent.exists()

    write_prompting_descriptions(cfg, concepts)

    assert pos_path.parent.exists()
    assert pos_path.exists()


def test_overwrites_existing_file(tmp_path: Path, monkeypatch) -> None:
    """Writing twice replaces the file content, not appends."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run6", run_negative_mode=False)

    concepts_v1 = [_make_concept(0, "original description")]
    write_prompting_descriptions(cfg, concepts_v1)

    concepts_v2 = [_make_concept(0, "updated description"), _make_concept(1, "brand new")]
    write_prompting_descriptions(cfg, concepts_v2)

    pos_path = test_paths.default_paths("run6").prompting_descriptions(SteeringMode.POSITIVE)
    data = json.loads(pos_path.read_text(encoding="utf-8"))

    assert data == {"0": "updated description", "1": "brand new"}
    assert "original description" not in pos_path.read_text(encoding="utf-8")


def test_negative_file_has_same_structure(tmp_path: Path, monkeypatch) -> None:
    """Negative file has identical concept_id→description structure as positive."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run7", run_negative_mode=True)
    concepts = [_make_concept(10, "ten"), _make_concept(20, "twenty")]

    write_prompting_descriptions(cfg, concepts)

    neg_path = test_paths.default_paths("run7").prompting_descriptions(SteeringMode.NEGATIVE)
    data = json.loads(neg_path.read_text(encoding="utf-8"))
    assert data == {"10": "ten", "20": "twenty"}


def test_returns_paths_exist_on_disk(tmp_path: Path, monkeypatch) -> None:
    """All returned Path objects exist and point to actual files."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="run8", run_negative_mode=True)
    concepts = [_make_concept(3, "three")]

    result = write_prompting_descriptions(cfg, concepts)

    for p in result:
        assert p.exists(), f"Returned path does not exist: {p}"
        assert p.is_file(), f"Returned path is not a file: {p}"
