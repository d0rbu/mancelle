"""Stage 0a ``concept_feature_whitelist_path`` integration tests.

Exercises :func:`prepare_concept_list` against real
``manual_feature_filter.yaml`` / ``accepted_feature_ids.txt`` whitelist
files on disk (the exact artifact shapes produced by
``scripts/preprocess/precompute_autointerp_filter.py``), with feature
statistics and the Neuronpedia client stubbed the same way
``tests/axbench_base/integration/test_stage_0_prepare.py`` stubs Stage 0a.

Unlike the unit-level whitelist tests in
``tests/axbench_base/unit/test_concept_list.py`` (which drive
``prepare_concept_list`` against in-memory config permutations), these
tests assert the full on-disk contract: a whitelist file written to
``tmp_path``, parsed by the real parser, resolved before the Neuronpedia
fetch, and reflected byte-for-byte in the written
``candidate_concepts.json``.

Both tests assert two invariants required by issue #683:

* Neuronpedia's ``get_descriptions`` is called with ONLY the
  whitelist-surviving candidate IDs (never the full feature-statistics
  universe) — the whitelist bounds the fetch, not just the final pool.
* Every ``feature_index`` written to ``candidate_concepts.json`` is a
  member of the configured whitelist (pool subset-of-whitelist).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_base import concept_list as concept_list_mod
from steering.eval.axbench_base.concept_list import prepare_concept_list
from steering.eval.axbench_base.types import AxBenchBaseError, CandidateConcept
from steering.eval.axbench_base.validate import load_and_validate_candidate_concepts
from steering.segmentation import SegmentMode
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)
from tests.axbench_base import paths as test_paths


def _make_per_feature_stat(count: int) -> PerFeatureStat:
    return PerFeatureStat(
        count=count,
        pct_segments_active=0.1,
        mean_mag=1.0,
        mean_mag_incl_zeros=0.1,
        std_mag=0.5,
        min_mag=0.01,
        max_mag=5.0,
        median_mag=0.8,
        percentiles={"p50": 0.8, "p90": 3.0, "p99": 4.5},
        histogram=None,
    )


def _make_stats(
    per_feature: dict[str, PerFeatureStat],
    n_sae_features: int = 16384,
) -> ExtractionStatistics:
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=1024,
            layer=8,
            n_sae_features=n_sae_features,
            generated_at="2026-07-10",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=10_000,
            mean_tokens_per_doc=100.0,
            median_tokens_per_doc=100,
        ),
        features=SegmentFeatureStats(
            n_segments=1000,
            n_features_seen=len(per_feature),
            mean_active_per_segment=10.0,
            median_active_per_segment=10,
            p90_active_per_segment=20,
            max_active_per_segment=50,
            min_active_per_segment=0,
            n_total_sae_features=n_sae_features,
            pct_features_active=0.01,
            count_percentiles={"p50": 50},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={"p50": 10.0},
        ),
        storage=StorageStats(
            total_bytes=1_000_000,
            mean_bytes_per_doc=10_000.0,
            mean_bytes_per_token=1.0,
        ),
        per_feature_stats=per_feature,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=100_000_000,
            estimated_storage_gb=100.0,
            estimated_wall_time_hours=10.0,
        ),
    )


class _RecordingNeuronpediaClient:
    """Fake Neuronpedia client that records every ``get_descriptions`` call.

    Mirrors the real client's "drop on error" contract for missing
    features (silently absent from the returned dict) so the fake stays
    faithful to production semantics.
    """

    def __init__(self, descriptions: dict[int, str]) -> None:
        self._descriptions = descriptions
        self.get_descriptions_calls: list[list[int]] = []

    def get_descriptions(
        self,
        feature_ids: list[int],
        *,
        cache_only: bool = False,
        max_not_published_rate: float = 0.0,
        max_missing_rate: float = 0.0,
        max_failed_rate: float = 0.0,
        min_mismatch_features: int = 0,
    ) -> dict[int, str]:
        del cache_only, max_not_published_rate
        del max_missing_rate, max_failed_rate, min_mismatch_features
        self.get_descriptions_calls.append(list(feature_ids))
        return {
            fid: self._descriptions[fid]
            for fid in feature_ids
            if fid in self._descriptions
        }


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    stats: ExtractionStatistics,
    client: _RecordingNeuronpediaClient,
) -> None:
    monkeypatch.setattr(concept_list_mod, "_load_feature_statistics", lambda path: stats)
    monkeypatch.setattr(concept_list_mod, "_get_neuronpedia_client", lambda cfg: client)
    monkeypatch.setattr(
        concept_list_mod,
        "_build_concept_examples_qc_client",
        lambda cfg, response_cache=None: (_ for _ in ()).throw(
            AssertionError("Stage 0a should not call the LLM examples-QC judge")
        ),
    )


def test_stage0a_with_manual_filter_yaml(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real ``manual_feature_filter.yaml`` (subset uncommented) bounds
    both the Neuronpedia fetch and the written candidate pool.

    Fids 0..9 all pass ``min_corpus_count`` (counts 5, 15, ..., 95 with
    default ``min_corpus_count=10`` keeping fids 1..9), but the whitelist
    restricts candidacy to {2, 4, 6} BEFORE any Neuronpedia call. Fid 4's
    description is deliberately too short, exercising the
    "curated features below thresholds are dropped, logged loudly" path
    while the surviving pool stays a strict whitelist subset.
    """
    manual = tmp_run_root / "manual_feature_filter.yaml"
    manual.write_text(
        '# - "00009"  # score=0.91 (commented out -- not selected)\n'
        '- "00002"  # score=0.83\n'
        '- "00004"  # score=0.75\n'
        '- "00006"  # score=0.60\n'
    )

    stats_path = tmp_run_root / "stats.safetensors"
    stats_path.write_bytes(b"stub")
    per_feature = {str(i): _make_per_feature_stat(count=5 + i * 10) for i in range(10)}
    stats = _make_stats(per_feature, n_sae_features=16384)
    descriptions = {
        i: f"Concept description number {i} is long enough to pass the filter."
        for i in range(10)
    }
    descriptions[4] = "short"  # below min_description_length -> dropped post-fetch

    client = _RecordingNeuronpediaClient(descriptions)
    _install_stubs(monkeypatch, stats, client)

    cfg = make_valid_axbench_cfg(
        feature_statistics_path=str(stats_path),
        run_tag="stage0a_whitelist_yaml",
        concept_feature_whitelist_path=str(manual),
        max_concepts=2,
        min_description_length=10,
        min_corpus_count=10,
    )

    out = prepare_concept_list(cfg)

    assert out == test_paths.default_paths("stage0a_whitelist_yaml").candidate_concepts_json()
    concepts = json.loads(out.read_text())
    fids = {c["feature_index"] for c in concepts}

    # Neuronpedia was queried with ONLY the whitelist-surviving candidates
    # (whitelist ∩ stats ∩ min_corpus_count), never the full 0..9 universe.
    assert client.get_descriptions_calls == [[2, 4, 6]]

    # Fid 4 was dropped by the description-length filter, so the final
    # pool is a strict subset of the whitelist, not the whole whitelist.
    assert fids == {2, 6}
    assert fids <= {2, 4, 6}
    for c in concepts:
        assert isinstance(c["feature_index"], int)
        assert c["feature_index"] in {2, 4, 6}


def test_stage0a_with_accepted_feature_ids_txt(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real ``accepted_feature_ids.txt`` (the other supported whitelist
    format) bounds the fetch and the pool identically to the YAML path.
    """
    txt = tmp_run_root / "accepted_feature_ids.txt"
    txt.write_text("# accepted feature ids\n1\n3\n5\n")

    stats_path = tmp_run_root / "stats.safetensors"
    stats_path.write_bytes(b"stub")
    per_feature = {str(i): _make_per_feature_stat(count=5 + i * 10) for i in range(10)}
    stats = _make_stats(per_feature, n_sae_features=16384)
    descriptions = {
        i: f"Concept description number {i} is long enough to pass the filter."
        for i in range(10)
    }

    client = _RecordingNeuronpediaClient(descriptions)
    _install_stubs(monkeypatch, stats, client)

    cfg = make_valid_axbench_cfg(
        feature_statistics_path=str(stats_path),
        run_tag="stage0a_whitelist_txt",
        concept_feature_whitelist_path=str(txt),
        max_concepts=3,
        min_description_length=10,
        min_corpus_count=10,
    )

    out = prepare_concept_list(cfg)

    concepts = json.loads(out.read_text())
    fids = {c["feature_index"] for c in concepts}

    assert client.get_descriptions_calls == [[1, 3, 5]]
    assert fids == {1, 3, 5}

    loaded = load_and_validate_candidate_concepts(out, n_sae_features=16384)
    assert {c.feature_index for c in loaded} == {1, 3, 5}
    for c in loaded:
        assert isinstance(c, CandidateConcept)
        assert c.feature_index in {1, 3, 5}


def test_stage0a_absent_in_range_id_fails_before_client_or_artifact(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whitelist_path = tmp_run_root / "accepted_feature_ids.txt"
    whitelist_path.write_text("2\n20\n")
    stats_path = tmp_run_root / "stats.safetensors"
    stats_path.write_bytes(b"stub")
    stats = _make_stats(
        {str(i): _make_per_feature_stat(count=20) for i in range(10)},
        n_sae_features=16384,
    )
    client = _RecordingNeuronpediaClient({})
    _install_stubs(monkeypatch, stats, client)

    def _unexpected_client_factory(cfg):
        del cfg
        raise AssertionError("Neuronpedia client must not be created")

    monkeypatch.setattr(
        concept_list_mod,
        "_get_neuronpedia_client",
        _unexpected_client_factory,
    )
    cfg = make_valid_axbench_cfg(
        feature_statistics_path=str(stats_path),
        run_tag="stage0a_absent_whitelist_id",
        concept_feature_whitelist_path=str(whitelist_path),
        max_concepts=1,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        prepare_concept_list(cfg)

    message = str(exc_info.value)
    assert "absent_ids=[20]" in message
    assert str(stats_path) in message
    assert "dead feature" in message
    assert "wrong SAE or layer" in message
    assert not test_paths.default_paths("stage0a_absent_whitelist_id").candidate_concepts_json().exists()
