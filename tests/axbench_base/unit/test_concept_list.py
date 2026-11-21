import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from loguru import logger as loguru_logger

from errors import DataLoadError
from steering.eval.axbench_base import concept_list, paths
from steering.eval.axbench_base.concept_examples_qc import (
    ConceptExamplesQC,
    QCFeatureScore,
    QCReport,
)
from steering.eval.axbench_base.concept_list import prepare_concept_list
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    QcOutcome,
)
from steering.eval.neuronpedia_client import NeuronpediaSourceMismatchError
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
from tests.axbench_base.unit.test_config import make_valid_config


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
            generated_at="2026-04-14",
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


class _AcceptAllQCClient:
    """Default test double: accepts every candidate and returns a
    matching ``QCReport`` where every concept has ``outcome="accepted"``.

    Accepts the ``target_surviving`` kwarg supported by production QC so test flows that go
    through ``prepare_concept_list`` exercise the production call
    shape. The kwargs are recorded on the instance so individual tests
    can assert they were forwarded correctly.
    """

    def __init__(self):
        self.filter_calls: list[list[tuple[int, str]]] = []
        self.last_target_surviving: int | None = None

    def filter(
        self,
        candidates,
        target_surviving: int | None = None,
    ):
        self.filter_calls.append(list(candidates))
        self.last_target_surviving = target_surviving
        ordered = list(candidates)
        if target_surviving is not None:
            ordered = ordered[:target_surviving]
        per_feature = [
            QCFeatureScore(
                feature_index=fid,
                description=desc,
                outcome=QcOutcome.ACCEPTED,
                score=2,
                n_examples=5,
                error_message=None,
            )
            for fid, desc in ordered
        ]
        report = QCReport(
            n_total=len(candidates),
            n_accepted=len(ordered),
            n_rejected=0,
            n_fetch_errors=0,
            n_judge_errors=0,
            n_empty_examples=0,
            score_histogram={2: len(ordered)} if ordered else {},
            per_feature=per_feature,
            judge_min_score=1,
            n_processed=len(ordered),
        )
        return ordered, report


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """Replace stats and Neuronpedia with test doubles.

    Stage 0a must not build the examples-QC judge after the lazy-QC
    refactor. The QC factory is patched to raise so any accidental
    eager LLM work fails loudly.
    """
    # Default stats: fids 0..9, counts 5/15/25/35/... (so count >= 10 picks fids 1..9 = 9 fids)
    default_per_feature = {
        str(i): _make_per_feature_stat(count=5 + i * 10) for i in range(10)
    }
    stats_holder = {"stats": _make_stats(default_per_feature)}
    statistics_hash_digests = ["b" * 64]
    statistics_hash_calls: list[Path] = []
    # Simulates the fused read's own content check inside the REAL
    # _load_feature_statistics (see its docstring's named BUG-FIX
    # deviation): defaults to matching statistics_hash_digests[0] so every
    # existing test that never touches this holder is unaffected. A test
    # that wants to exercise the "changed while loading" raise through
    # this mock now sets this holder instead of appending a 2nd
    # statistics_hash_digests value (that 2nd, separate post-load
    # stream_hash_file_stable call no longer exists in production).
    statistics_content_sha256_holder = {"sha256": "b" * 64}

    def _fake_load(path: str | Path, expected_sha256: str | None = None):
        if (
            expected_sha256 is not None
            and statistics_content_sha256_holder["sha256"] != expected_sha256
        ):
            raise AxBenchBaseError(
                "Clustered Stage 0a feature statistics bytes changed while loading: "
                f"expected_sha256={expected_sha256}, "
                f"actual_sha256={statistics_content_sha256_holder['sha256']}"
            )
        return stats_holder["stats"]

    def _fake_hash(path: str | Path):
        statistics_hash_calls.append(Path(path))
        digest = (
            statistics_hash_digests.pop(0)
            if len(statistics_hash_digests) > 1
            else statistics_hash_digests[0]
        )
        return SimpleNamespace(sha256=digest)

    descriptions_holder = {
        "descriptions": {i: f"Description of feature {i} (long enough)" for i in range(10)},
    }
    # ``cached_fids`` simulates the on-disk Neuronpedia cache. None means "every
    # describable feature is cached" (so cache_only == full fetch of the cache);
    # a set restricts what cache_only mode can return.
    cache_holder: dict[str, set[int] | None] = {"cached_fids": None}

    class FakeClient:
        def __init__(self, model_id, sae_source, cache_path):
            self.model_id = model_id
            self.sae_source = sae_source
            self.cache_path = cache_path
            self.get_descriptions_calls: list[list[int]] = []
            self.cache_only_calls: list[bool] = []
            self.max_not_published_rate_calls: list[float] = []
            self.max_missing_rate_calls: list[float] = []
            self.max_failed_rate_calls: list[float] = []
            self.min_mismatch_features_calls: list[int] = []

        def get_description(self, feature_id: int) -> str:
            if feature_id not in descriptions_holder["descriptions"]:
                # Raise the SAME exception class that the real client
                # uses, so the fake's drop contract matches production
                # exactly (the real `get_descriptions` only catches
                # `NeuronpediaError`).
                from steering.eval.neuronpedia_client import NeuronpediaFetchError
                raise NeuronpediaFetchError(f"feature {feature_id} missing")
            return descriptions_holder["descriptions"][feature_id]

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
            """Mirror the real client's "drop on NeuronpediaError" contract.

            Catching the broader ``Exception`` would let the fake hide
            divergence (e.g., a real client bug that raised RuntimeError
            would propagate in production but not in tests). Narrow
            catch keeps the fake faithful.

            ``cache_only`` mirrors the real client: it returns only entries
            in the (simulated) cache and makes no "network" calls.
            ``max_not_published_rate`` is recorded so tests can assert Stage
            0a opts into the real client's source-mismatch guard; the fake
            never raises on it (the guard itself is covered by the real
            client's unit tests)."""
            from steering.eval.neuronpedia_client import NeuronpediaError
            self.get_descriptions_calls.append(list(feature_ids))
            self.cache_only_calls.append(cache_only)
            self.max_not_published_rate_calls.append(max_not_published_rate)
            self.max_missing_rate_calls.append(max_missing_rate)
            self.max_failed_rate_calls.append(max_failed_rate)
            self.min_mismatch_features_calls.append(min_mismatch_features)
            cached = cache_holder["cached_fids"]
            out: dict[int, str] = {}
            for fid in feature_ids:
                if cache_only and cached is not None and fid not in cached:
                    continue
                try:
                    out[fid] = self.get_description(fid)
                except NeuronpediaError:
                    continue
            return out

    created_clients = []

    def _fake_get_client(cfg):
        c = FakeClient(
            model_id=cfg.neuronpedia_model_id,
            sae_source=cfg.neuronpedia_source,
            cache_path=str(tmp_path / "neuronpedia_cache.yaml"),
        )
        created_clients.append(c)
        return c

    qc_factory_calls: list[dict] = []

    def _unexpected_qc_factory(cfg, response_cache=None):
        qc_factory_calls.append({"cfg": cfg, "response_cache": response_cache})
        raise AssertionError("Stage 0a should not call LLM")

    monkeypatch.setattr(concept_list, "_load_feature_statistics", _fake_load)
    monkeypatch.setattr(concept_list, "stream_hash_file_stable", _fake_hash)
    monkeypatch.setattr(concept_list, "_get_neuronpedia_client", _fake_get_client)
    monkeypatch.setattr(
        concept_list, "_build_concept_examples_qc_client", _unexpected_qc_factory,
    )
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    return {
        "tmp_path": tmp_path,
        "stats_holder": stats_holder,
        "statistics_hash_digests": statistics_hash_digests,
        "statistics_hash_calls": statistics_hash_calls,
        "statistics_content_sha256_holder": statistics_content_sha256_holder,
        "descriptions_holder": descriptions_holder,
        "cache_holder": cache_holder,
        "created_clients": created_clients,
        "qc_factory_calls": qc_factory_calls,
    }


def _cfg(tmp_path, **overrides):
    base = make_valid_config(
        run_tag="test_concept_list_run",
        output_root=str(tmp_path / "eval_out"),
        min_corpus_count=10,
        min_description_length=10,
        max_concepts=500,
        concept_list_seed=42,
        neuronpedia_model_id="gemma-2-2b",
        neuronpedia_source="8-gemmascope-res-16k",
    )
    if overrides:
        return base.replace(**overrides)
    return base


def _cluster_contract_for_concept_list(cfg):
    return SimpleNamespace(
        n_sae_features=10,
        source_to_representative=((1, 1), (2, 1), (3, 3)),
        concept_source_domain=(1, 2, 3, 4),
        representative_to_allowed_members=((1, (1, 2)), (3, (3,))),
        representative_to_projecting_sources=((1, (1, 2)), (3, (3,))),
        feature_directory=SimpleNamespace(layer=8),
        statistics=SimpleNamespace(file_sha256="b" * 64),
        feature_contract_fingerprint="a" * 64,
        provenance_dict=lambda: {
            "feature_statistics_path": str(
                Path(cfg.feature_statistics_path).expanduser().resolve()
            )
        },
        semantic_dict=lambda: {
            "algorithms": {"concept_projection": "cluster_concept_projection_v1"}
        },
    )


def test_filters_by_count(stub_env):
    cfg = _cfg(stub_env["tmp_path"], min_corpus_count=25)
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    # With min=25: counts 5..95, keep counts >= 25 → fids 2..9 (8 features)
    assert len(concepts) == 8
    for c in concepts:
        assert c["corpus_count"] >= 25


def test_filters_by_description_length(stub_env):
    # Set description for fid 3 to be too short
    stub_env["descriptions_holder"]["descriptions"][3] = "short"
    cfg = _cfg(stub_env["tmp_path"], min_corpus_count=10, min_description_length=10)
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    # fid 0 has count=5 (filtered), fid 3 has short description (filtered)
    # Remaining: 1, 2, 4, 5, 6, 7, 8, 9 = 8 features
    assert len(concepts) == 8
    assert all(c["feature_index"] != 3 for c in concepts)


def test_deterministic_shuffle(stub_env):
    cfg = _cfg(stub_env["tmp_path"])
    out1 = prepare_concept_list(cfg)
    concepts1 = json.loads(out1.read_text())
    out1.unlink()
    out2 = prepare_concept_list(cfg)
    concepts2 = json.loads(out2.read_text())
    assert [c["feature_index"] for c in concepts1] == [c["feature_index"] for c in concepts2]


def test_different_seed_different_order(stub_env):
    cfg_a = _cfg(stub_env["tmp_path"], concept_list_seed=1, run_tag="seed_a")
    cfg_b = _cfg(stub_env["tmp_path"], concept_list_seed=2, run_tag="seed_b")
    ca = json.loads(prepare_concept_list(cfg_a).read_text())
    cb = json.loads(prepare_concept_list(cfg_b).read_text())
    # Very unlikely to be identical across different seeds on 9 shuffled items
    assert [c["feature_index"] for c in ca] != [c["feature_index"] for c in cb]


def test_raises_when_no_features_survive_filter(stub_env):
    cfg = _cfg(stub_env["tmp_path"], min_corpus_count=10**9)
    with pytest.raises(AxBenchBaseError, match="count"):
        prepare_concept_list(cfg)


def test_raises_when_no_descriptions_pass_length_filter(stub_env):
    # All descriptions set to very short
    stub_env["descriptions_holder"]["descriptions"] = dict.fromkeys(range(10), "a")
    cfg = _cfg(stub_env["tmp_path"], min_description_length=10)
    with pytest.raises(AxBenchBaseError, match="description"):
        prepare_concept_list(cfg)


def test_neuronpedia_client_called_with_correct_args(stub_env):
    cfg = _cfg(stub_env["tmp_path"])
    prepare_concept_list(cfg)
    assert len(stub_env["created_clients"]) == 1
    client = stub_env["created_clients"][0]
    assert client.model_id == "gemma-2-2b"
    assert client.sae_source == "8-gemmascope-res-16k"


def test_candidate_schema_has_no_concept_id_or_judge_score(stub_env):
    cfg = _cfg(stub_env["tmp_path"])
    out = prepare_concept_list(cfg)
    rows = json.loads(out.read_text())
    assert rows
    for row in rows:
        assert set(row) == {
            "feature_index",
            "layer",
            "sae_source",
            "description",
            "corpus_count",
            "description_source",
        }
        assert "concept_id" not in row
        assert "judge_score" not in row
        assert "llm_judge_score" not in row


def test_feature_index_uniqueness(stub_env):
    cfg = _cfg(stub_env["tmp_path"])
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    fids = [c["feature_index"] for c in concepts]
    assert len(fids) == len(set(fids))


# ============================================================================
# Clustered Stage 0a (PR #739 Track C2). The 6 tests below are ported
# verbatim from the oracle (@ f357458b). See _cluster_contract_for_concept_
# list above for the hand-typed SimpleNamespace fixture they share.
# ============================================================================


def test_clustered_concept_list_projects_deduplicates_and_uses_rep_own_evidence(
    stub_env,
):
    stub_env["stats_holder"]["stats"] = _make_stats(
        {
            "1": _make_per_feature_stat(count=31),
            # A sibling has deliberately different evidence. It may project to
            # rep 1 but must never donate its count or description.
            "2": _make_per_feature_stat(count=999),
            "3": _make_per_feature_stat(count=43),
        },
        n_sae_features=10,
    )
    stub_env["descriptions_holder"]["descriptions"] = {
        1: "Representative one own description",
        2: "Sibling description must never be requested",
        3: "Representative three own description",
    }
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=2,
        min_corpus_count=1,
        min_description_length=1,
        run_tag="clustered_concepts",
    )
    contract = _cluster_contract_for_concept_list(cfg)

    out = prepare_concept_list(cfg, cluster_contract=contract)
    rows = json.loads(out.read_text())

    assert {row["feature_index"] for row in rows} == {1, 3}
    assert {row["feature_index"]: row["corpus_count"] for row in rows} == {
        1: 31,
        3: 43,
    }
    assert stub_env["created_clients"][0].get_descriptions_calls == [[1, 3]]
    provenance = json.loads(
        paths.AxBenchBasePaths.from_config(cfg)
        .candidate_concepts_cluster_provenance_json()
        .read_text()
    )
    by_rep = {row["representative_id"]: row for row in provenance["representatives"]}
    assert by_rep[1]["allowed_member_ids"] == [1, 2]
    assert by_rep[1]["projecting_source_ids"] == [1, 2]
    assert by_rep[1]["statistics_source_id"] == 1
    assert provenance["counts"] == {
        "raw_source_ids": 4,
        "mask_allowed_source_ids": 3,
        "unique_representatives": 2,
        "representatives_with_statistics": 2,
        "count_survivors": 2,
        "description_fetch_survivors": 2,
        "described_survivors": 2,
    }


def test_clustered_concept_list_exact_resume_precedes_statistics_and_api(
    stub_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_env["stats_holder"]["stats"] = _make_stats(
        {
            "1": _make_per_feature_stat(count=31),
            "2": _make_per_feature_stat(count=999),
            "3": _make_per_feature_stat(count=43),
        },
        n_sae_features=10,
    )
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=2,
        min_corpus_count=1,
        min_description_length=1,
        run_tag="clustered_concepts_resume",
    )
    contract = _cluster_contract_for_concept_list(cfg)
    output = prepare_concept_list(cfg, cluster_contract=contract)
    provenance_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).candidate_concepts_cluster_provenance_json()
    candidate_before = output.read_bytes()
    provenance_before = provenance_path.read_bytes()

    monkeypatch.setattr(
        concept_list,
        "load_cluster_candidate_identity",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        concept_list,
        "_load_feature_statistics",
        lambda *_args, **_kwargs: pytest.fail(
            "exact clustered Stage 0a resume must precede statistics loading"
        ),
    )
    monkeypatch.setattr(
        concept_list,
        "_get_neuronpedia_client",
        lambda *_args, **_kwargs: pytest.fail(
            "exact clustered Stage 0a resume must precede API client creation"
        ),
    )

    assert prepare_concept_list(cfg, cluster_contract=contract) == output
    assert output.read_bytes() == candidate_before
    assert provenance_path.read_bytes() == provenance_before


def test_clustered_concept_list_resume_reapplies_current_max_concepts_underfill(
    stub_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for review finding 3 (HIGH): the resume path
    validates the committed artifact's INTERNAL consistency (fingerprint +
    sha256 agree) but must ALSO re-apply the CURRENT cfg.max_concepts,
    exactly like a fresh run would -- an artifact committed under
    max_concepts=2 must not be silently reused once a LATER run under the
    SAME run_tag asks for more concepts than the committed pool actually
    contains. Named BUG-FIX deviation from the oracle: the oracle's own
    resume path has this same gap (pre-fix, this scenario silently
    returned the under-filled 2-row artifact instead of raising).

    Mocks ``load_cluster_candidate_identity`` for the resume calls, same
    technique as test_clustered_concept_list_exact_resume_precedes_
    statistics_and_api above: the hand-typed SimpleNamespace contract
    fixture has no ``.document()`` method the real C1 marker/sidecar
    machinery (``install_or_validate_cluster_root_marker`` ->
    ``write_feature_contract_sidecar``) would need, and this test's own
    focus -- the NEW ``cfg.max_concepts`` re-check this file's code adds
    after ``identity`` resolves -- is independent of how ``identity``
    itself gets validated. The real, non-mocked resume path (identity
    resolution included) is exercised by the functional tier's real-
    contract tests elsewhere."""
    stub_env["stats_holder"]["stats"] = _make_stats(
        {
            "1": _make_per_feature_stat(count=31),
            "2": _make_per_feature_stat(count=999),
            "3": _make_per_feature_stat(count=43),
        },
        n_sae_features=10,
    )
    cfg_create = _cfg(
        stub_env["tmp_path"],
        max_concepts=2,
        min_corpus_count=1,
        min_description_length=1,
        run_tag="clustered_resume_underfill",
    )
    contract = _cluster_contract_for_concept_list(cfg_create)
    created = prepare_concept_list(cfg_create, cluster_contract=contract)
    created_bytes = created.read_bytes()
    assert len(json.loads(created.read_text())) == 2

    monkeypatch.setattr(
        concept_list,
        "load_cluster_candidate_identity",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    # RED (pre-fix): resuming under the same run_tag with a HIGHER
    # max_concepts than the committed pool actually contains must now
    # raise the same underfill error a fresh run would -- not silently
    # return the under-filled artifact.
    cfg_resume_more = cfg_create.replace(max_concepts=3)
    with pytest.raises(
        AxBenchBaseError,
        match=r"underfilled_at=described_survivors, max_concepts=3.*described_survivors=2",
    ):
        prepare_concept_list(cfg_resume_more, cluster_contract=contract)
    # Refusing to reuse must not mutate the already-committed artifact.
    assert created.read_bytes() == created_bytes

    # GREEN: resume at the SAME max_concepts the artifact was created
    # under still succeeds, byte-identical.
    cfg_resume_same = cfg_create.replace(max_concepts=2)
    resumed_same = prepare_concept_list(cfg_resume_same, cluster_contract=contract)
    assert resumed_same == created
    assert resumed_same.read_bytes() == created_bytes

    # GREEN: resume at a LOWER max_concepts than the artifact was created
    # under also still succeeds.
    cfg_resume_less = cfg_create.replace(max_concepts=1)
    resumed_less = prepare_concept_list(cfg_resume_less, cluster_contract=contract)
    assert resumed_less == created
    assert resumed_less.read_bytes() == created_bytes


def test_clustered_concept_list_partial_pair_fails_before_statistics_or_api(
    stub_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=2,
        min_corpus_count=1,
        min_description_length=1,
        run_tag="clustered_concepts_partial",
    )
    contract = _cluster_contract_for_concept_list(cfg)
    output = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"partial-candidate-evidence")
    before = output.read_bytes()
    monkeypatch.setattr(
        concept_list,
        "load_cluster_candidate_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AxBenchBaseError("candidate provenance commit marker is missing")
        ),
    )
    monkeypatch.setattr(
        concept_list,
        "_load_feature_statistics",
        lambda *_args, **_kwargs: pytest.fail(
            "partial clustered Stage 0a must fail before statistics loading"
        ),
    )
    monkeypatch.setattr(
        concept_list,
        "_get_neuronpedia_client",
        lambda *_args, **_kwargs: pytest.fail(
            "partial clustered Stage 0a must fail before API client creation"
        ),
    )

    with pytest.raises(AxBenchBaseError, match="commit marker is missing"):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert output.read_bytes() == before
    assert (
        not paths.AxBenchBasePaths.from_config(cfg)
        .candidate_concepts_cluster_provenance_json()
        .exists()
    )


def test_clustered_concept_underfill_fails_before_neuronpedia_client(stub_env):
    stub_env["stats_holder"]["stats"] = _make_stats(
        {
            str(feature_id): _make_per_feature_stat(count=100)
            for feature_id in range(10)
        },
        n_sae_features=10,
    )
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=3,
        min_corpus_count=1,
        run_tag="clustered_underfill",
    )
    contract = _cluster_contract_for_concept_list(cfg)

    with pytest.raises(
        AxBenchBaseError,
        match=(
            r"underfilled_at=unique_representatives, max_concepts=3.*"
            r"unique_representatives=2"
        ),
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []
    assert not paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json().exists()


def test_clustered_concept_list_rejects_statistics_changed_after_preflight(
    stub_env,
):
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=2,
        min_corpus_count=1,
        run_tag="clustered_statistics_changed_before_load",
    )
    contract = _cluster_contract_for_concept_list(cfg)
    stub_env["statistics_hash_digests"][:] = ["c" * 64]

    with pytest.raises(
        AxBenchBaseError,
        match="feature statistics bytes changed after preflight",
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []
    assert not paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json().exists()


def test_clustered_concept_list_rejects_statistics_changed_while_loading(
    stub_env,
):
    """Named mechanism deviation from the oracle's own version of this
    test (review-fix, mirrors C1's Finding-1(b)): the oracle simulated
    this via a SECOND ``stream_hash_file_stable`` call disagreeing with
    the first (``statistics_hash_digests = ["b"*64, "c"*64]``) -- that
    second, separate post-load re-hash-of-the-path no longer exists in
    production (see ``_load_feature_statistics``'s docstring): the
    "while loading" check is now fused onto the exact bytes the load
    itself reads, verified via ``expected_sha256`` rather than a second
    independent path digest. ``stub_env``'s ``statistics_content_sha256_
    holder`` simulates that fused check's own mismatch at the mock
    boundary; the REAL fused implementation is proven directly (not just
    at this mock boundary) by
    test_load_feature_statistics_fused_read_rejects_mismatched_bytes and
    test_load_feature_statistics_fused_read_parses_verified_bytes below.
    """
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=2,
        min_corpus_count=1,
        run_tag="clustered_statistics_changed_while_loading",
    )
    contract = _cluster_contract_for_concept_list(cfg)
    stub_env["statistics_content_sha256_holder"]["sha256"] = "c" * 64

    with pytest.raises(
        AxBenchBaseError,
        match="feature statistics bytes changed while loading",
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []
    assert not paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json().exists()


# ============================================================================
# C2 authorship: coverage gaps the oracle's own 6 tests miss, plus the
# raise-site 1:1 map's remaining sites (A, C, F, I -- G/D/E are exercised
# above; B/H are the 2 named coverage gaps) and the backward-compat pin.
# ============================================================================


def test_clustered_concept_list_raises_without_run_tag(stub_env):
    """Raise-site A (oracle:76-79): the clustered branch has its own
    run_tag guard, checked before ``output``/``provenance_path`` (both
    derived from ``cfg.run_tag``) are ever computed -- distinct from the
    legacy branch's identically-worded guard a few lines later, since the
    clustered dispatch happens first and never falls through to it."""
    cfg = _cfg(stub_env["tmp_path"], run_tag=None)
    contract = _cluster_contract_for_concept_list(cfg)

    with pytest.raises(
        AxBenchBaseError,
        match="run_tag must be set before calling prepare_concept_list",
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []


def test_clustered_concept_list_rejects_statistics_path_mismatch(stub_env):
    """Raise-site C (oracle:101-106): the contract's own preflight-recorded
    ``feature_statistics_path`` must match ``cfg.feature_statistics_path``
    exactly -- a config pointed at a different statistics file than the one
    the contract was resolved against is refused before any file is read."""
    cfg_for_contract = _cfg(
        stub_env["tmp_path"],
        run_tag="clustered_stats_path_mismatch",
        feature_statistics_path="stats_a.safetensors",
    )
    contract = _cluster_contract_for_concept_list(cfg_for_contract)
    cfg = cfg_for_contract.replace(feature_statistics_path="stats_b.safetensors")

    with pytest.raises(
        AxBenchBaseError,
        match="feature statistics path differs from preflight",
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []
    assert not paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json().exists()


def test_clustered_concept_list_rejects_statistics_width_changed_after_preflight(
    stub_env,
):
    """Raise-site F (oracle:128-132): even when the statistics file's bytes
    pass both hash checks (no TOCTOU drift -- see
    test_clustered_concept_list_rejects_statistics_changed_{after_preflight,
    while_loading} above), a stale/foreign statistics file whose declared
    ``n_sae_features`` no longer matches what the contract's preflight
    recorded must still be refused: byte stability alone does not imply the
    file's own metadata still agrees with the contract width."""
    stub_env["stats_holder"]["stats"] = _make_stats(
        {"1": _make_per_feature_stat(count=31), "3": _make_per_feature_stat(count=43)},
        n_sae_features=99,
    )
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=1,
        min_corpus_count=1,
        run_tag="clustered_width_changed",
    )
    contract = _cluster_contract_for_concept_list(cfg)  # n_sae_features=10

    with pytest.raises(
        AxBenchBaseError,
        match=r"statistics width changed after preflight: 99 != 10",
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []
    assert not paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json().exists()


def test_clustered_concept_list_pre_fetch_boundary_raises_when_config_is_unclustered(
    stub_env,
) -> None:
    """Coverage gap 1/2 -- raise-site B, pre-fetch (oracle:91-95): a caller
    passes an explicit ``cluster_contract`` (forcing the clustered branch)
    while ``cfg.conditioning`` does not mark clustering enabled. A stray
    ``candidate_concepts.json`` already sits at the run_tag path (e.g. left
    over from an unrelated unclustered run reusing the same run_tag) with no
    clustered marker/provenance sidecar alongside it --
    ``load_cluster_candidate_identity`` legitimately returns ``None`` for an
    unclustered config with no cluster-stamped artifacts present, and
    ``_prepare_clustered_concept_list`` must refuse to treat that as a
    reusable clustered candidate pool rather than silently adopting it. NOT
    covered by the oracle's own 6 tests."""
    cfg = _cfg(
        stub_env["tmp_path"],
        run_tag="clustered_boundary_pre_fetch",
        conditioning=None,
    )
    contract = _cluster_contract_for_concept_list(cfg)
    output = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"stray-unclustered-candidate-evidence")

    with pytest.raises(AxBenchBaseError, match="unclustered configuration boundary"):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert stub_env["created_clients"] == []
    assert output.read_bytes() == b"stray-unclustered-candidate-evidence"


def test_clustered_concept_list_post_fetch_boundary_raises_when_config_is_unclustered(
    stub_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage gap 2/2 -- raise-site H, post-fetch (oracle:277-280): the
    pre-fetch guard is skipped because nothing exists at entry, but a
    concurrent writer creates ``candidate_concepts.json`` (without the
    clustered marker/provenance sidecar) while this run's own description
    fetch is in flight -- the SAME unclustered-config boundary check,
    re-run after the fetch, must catch this too, not just the pre-fetch
    site. No production seam exists to trigger this from the outside
    (concept_list.py never calls ``clustered_source_enabled`` itself, and
    the whole point is that the race is invisible until the post-fetch
    re-check runs), so this simulates the race by making candidate
    creation a side effect of the Neuronpedia client factory -- the one
    real seam between the pre- and post-fetch checks a test can reach
    without editing concept_list.py's own code. NOT covered by the
    oracle's own 6 tests."""
    stub_env["stats_holder"]["stats"] = _make_stats(
        {
            "1": _make_per_feature_stat(count=31),
            "3": _make_per_feature_stat(count=43),
        },
        n_sae_features=10,
    )
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=1,
        min_corpus_count=1,
        min_description_length=1,
        run_tag="clustered_boundary_post_fetch",
        conditioning=None,
    )
    contract = _cluster_contract_for_concept_list(cfg)
    output = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()

    real_get_client = concept_list._get_neuronpedia_client

    def _get_client_with_race(cfg_inner):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"raced-in-unclustered-candidate-evidence")
        return real_get_client(cfg_inner)

    monkeypatch.setattr(concept_list, "_get_neuronpedia_client", _get_client_with_race)

    with pytest.raises(AxBenchBaseError, match="unclustered configuration boundary"):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert output.read_bytes() == b"raced-in-unclustered-candidate-evidence"
    assert (
        not paths.AxBenchBasePaths.from_config(cfg)
        .candidate_concepts_cluster_provenance_json()
        .exists()
    )


def test_clustered_concept_list_rejects_content_changed_under_existing_run_tag(
    stub_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise-site I (oracle:281-289): a concurrent writer commits a
    DIFFERENT, otherwise-valid clustered candidate pair under this exact
    run_tag while this run's own (much slower) description fetch is in
    flight. The post-fetch re-check must detect the sha256 mismatch and
    refuse to silently overwrite already-committed evidence -- unlike
    raise-site H above, ``load_cluster_candidate_identity`` resolves a
    perfectly valid identity here; it just disagrees with what this run
    independently computed."""
    stub_env["stats_holder"]["stats"] = _make_stats(
        {
            "1": _make_per_feature_stat(count=31),
            "3": _make_per_feature_stat(count=43),
        },
        n_sae_features=10,
    )
    cfg = _cfg(
        stub_env["tmp_path"],
        max_concepts=1,
        min_corpus_count=1,
        min_description_length=1,
        run_tag="clustered_content_changed",
    )
    contract = _cluster_contract_for_concept_list(cfg)
    output = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()

    real_get_client = concept_list._get_neuronpedia_client

    def _get_client_with_race(cfg_inner):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"concurrently-committed-different-candidate-bytes")
        return real_get_client(cfg_inner)

    monkeypatch.setattr(concept_list, "_get_neuronpedia_client", _get_client_with_race)
    monkeypatch.setattr(
        concept_list,
        "load_cluster_candidate_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            candidate_concepts_sha256="0" * 64,
            candidate_provenance_sha256="0" * 64,
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="candidate content changed under an existing run_tag",
    ):
        prepare_concept_list(cfg, cluster_contract=contract)

    assert output.read_bytes() == b"concurrently-committed-different-candidate-bytes"
    assert (
        not paths.AxBenchBasePaths.from_config(cfg)
        .candidate_concepts_cluster_provenance_json()
        .exists()
    )


def test_prepare_concept_list_backward_compat_no_kwarg_call_matches_explicit_none(
    stub_env,
) -> None:
    """Backward-compat pin (binding constraint #4, Import-path
    preservation): the existing call site at
    ``scripts/eval/axbench_base_eval.py`` (``prepare_concept_list(cfg)``, no
    keyword argument at all) must keep working unchanged now that
    ``cluster_contract`` is a new keyword-only parameter defaulting to
    ``None``. Also doubles as this file's byte-identity regression anchor
    for the untouched legacy body: the no-kwarg call must be byte-identical
    to an explicit ``cluster_contract=None`` call, and reproduce the exact
    pre-C2 output shape/content (same filter/schema pin as
    test_filters_by_count / test_candidate_schema_has_no_concept_id_or_
    judge_score above)."""
    cfg_implicit = _cfg(stub_env["tmp_path"], run_tag="backward_compat_implicit")
    cfg_explicit = _cfg(stub_env["tmp_path"], run_tag="backward_compat_explicit")

    out_implicit = prepare_concept_list(cfg_implicit)
    out_explicit = prepare_concept_list(cfg_explicit, cluster_contract=None)

    implicit_rows = json.loads(out_implicit.read_text())
    explicit_rows = json.loads(out_explicit.read_text())
    assert implicit_rows == explicit_rows
    assert implicit_rows  # non-empty: the stub universe survives default filters
    for row in implicit_rows:
        assert set(row) == {
            "feature_index",
            "layer",
            "sae_source",
            "description",
            "corpus_count",
            "description_source",
        }
    # min_corpus_count=10 (the _cfg default) keeps fids 1..9 out of the
    # stub universe's 0..9 -- the untouched _cfg default itself is the pin
    # (test_filters_by_count exercises the same filter with an explicit
    # min_corpus_count=25 override instead).
    assert {row["feature_index"] for row in implicit_rows} == set(range(1, 10))


# ============================================================================
# Review-fix regression tests: the statistics-hash TOCTOU fix (mirrors C1's
# Finding-1(b)). These call _load_feature_statistics's REAL fused
# implementation directly (never mocked here), proving the fix's own core
# invariant -- "whatever gets parsed is provably the bytes that were
# verified" -- independent of the mock-boundary simulation the
# _prepare_clustered_concept_list-level tests above use. Filesystem-
# independent: monkeypatches the read/hash seam (``Path.read_bytes`` and
# ``load_statistics_and_checksum_from_bytes``) rather than relying on real
# FS race timing.
# ============================================================================


def test_load_feature_statistics_fused_read_rejects_mismatched_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates a same-hash A->B->A round-trip staged entirely inside the
    load window: ``expected_sha256`` is content A's digest (as if an
    earlier ``stream_hash_file_stable`` preflight check had already
    passed against A), but the actual read returns content B (the
    transient swap). The oracle's before/after PATH-hash bracket could
    never catch this -- before==expected and after==expected implies
    before==after by transitivity, and neither check ever looked at what
    the load itself actually read. The fused implementation must catch
    it here, at the one point that matters: the bytes about to be parsed.
    """
    content_a = b"content-A-the-digest-was-computed-against"
    content_b = b"content-B-a-transient-swap-mid-load-window"
    assert content_a != content_b
    expected_sha256 = hashlib.sha256(content_a).hexdigest()

    monkeypatch.setattr(Path, "read_bytes", lambda self: content_b)
    parse_calls: list[bytes] = []
    monkeypatch.setattr(
        concept_list,
        "load_statistics_and_checksum_from_bytes",
        lambda payload: (parse_calls.append(payload), (None, None, None))[1],
    )

    with pytest.raises(
        AxBenchBaseError,
        match="feature statistics bytes changed while loading",
    ):
        concept_list._load_feature_statistics(
            "irrelevant/path/for/this/mocked/read.safetensors",
            expected_sha256=expected_sha256,
        )

    # Never reached parsing with the unverified (wrong) content -- fail
    # loud BEFORE any attempt to build stats out of un-verified bytes.
    assert parse_calls == []


def test_load_feature_statistics_fused_read_parses_verified_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive counterpart: when the read bytes DO match expected_sha256,
    the fused path parses THOSE EXACT bytes (not a bypass, not a
    different buffer) -- pins the fix's core invariant directly rather
    than only proving the failure side."""
    content = b"content-matching-its-own-digest"
    expected_sha256 = hashlib.sha256(content).hexdigest()
    sentinel_stats = object()

    monkeypatch.setattr(Path, "read_bytes", lambda self: content)
    received_payloads: list[bytes] = []

    def _fake_parse(payload: bytes):
        received_payloads.append(payload)
        return sentinel_stats, "some-percentile-checksum", None

    monkeypatch.setattr(
        concept_list,
        "load_statistics_and_checksum_from_bytes",
        _fake_parse,
    )

    result = concept_list._load_feature_statistics(
        "irrelevant/path/for/this/mocked/read.safetensors",
        expected_sha256=expected_sha256,
    )

    assert result is sentinel_stats
    assert received_payloads == [content]


def test_unicode_descriptions_preserved(stub_env):
    stub_env["descriptions_holder"]["descriptions"][5] = "Descripción con árboles 🌳 y más cosas"
    cfg = _cfg(stub_env["tmp_path"])
    out = prepare_concept_list(cfg)
    # Use read_bytes + decode to confirm the raw file is utf-8 and not \uXXXX-escaped
    raw = out.read_bytes().decode("utf-8")
    assert "árboles" in raw
    assert "🌳" in raw
    # Also verify it parses back as valid JSON
    concepts = json.loads(raw)
    assert any("árboles" in c["description"] for c in concepts)


def test_neuronpedia_failure_skips_feature(stub_env):
    # Make fid 5 raise when described
    desc_map = stub_env["descriptions_holder"]["descriptions"]
    del desc_map[5]
    cfg = _cfg(stub_env["tmp_path"])
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    assert all(c["feature_index"] != 5 for c in concepts)


def test_stage_0a_writes_full_candidate_pool_without_max_concepts_truncation(
    stub_env,
):
    cfg = _cfg(stub_env["tmp_path"], max_concepts=3)
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    assert out == paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    assert len(concepts) == 9
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()


# ---------------------------------------------------------------------------
# Coverage tests for thin wrappers (monkey-patch targets) and defensive guards
# ---------------------------------------------------------------------------


def test_load_feature_statistics_delegates_to_stats_io(monkeypatch):
    """``_load_feature_statistics`` should forward its path argument to
    ``steering.stats_io.load_statistics`` wrapped in ``Path(...)``.

    We patch the imported ``load_statistics`` name in the ``concept_list``
    module so the wrapper hits our stub without touching the real stats loader.
    """
    sentinel = object()
    recorded: list[Path] = []

    def _fake_load_statistics(p):
        recorded.append(p)
        return sentinel

    monkeypatch.setattr(concept_list, "load_statistics", _fake_load_statistics)
    result = concept_list._load_feature_statistics("some/path.safetensors")
    assert result is sentinel
    assert recorded == [Path("some/path.safetensors")]


def test_get_neuronpedia_client_wires_cfg_fields(tmp_path, monkeypatch):
    """``_get_neuronpedia_client`` must pass ``cfg``-derived values to the
    NeuronpediaClient constructor, specifically:

      * ``model_id`` from ``cfg.neuronpedia_model_id``
      * ``sae_source`` from ``cfg.neuronpedia_source``
      * ``cache_path`` is identity-scoped beneath the effective output root

    We patch the imported ``NeuronpediaClient`` symbol in the ``concept_list``
    module with a recording stub that captures the kwargs it was called with.
    """
    captured: dict = {}
    sentinel_client = object()

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return sentinel_client

    monkeypatch.setattr(concept_list, "NeuronpediaClient", _fake_client)
    cfg = make_valid_config(
        run_tag="client_wire_test",
        output_root=str(tmp_path / "some" / "nested" / "out"),
        neuronpedia_model_id="gemma-2-2b",
        neuronpedia_source="8-gemmascope-res-16k",
    )
    result = concept_list._get_neuronpedia_client(cfg)
    assert result is sentinel_client
    assert captured == {
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "cache_path": paths.AxBenchBasePaths.from_config(cfg).neuronpedia_cache(
            model_id=cfg.neuronpedia_model_id,
            sae_source=cfg.neuronpedia_source,
        ),
        "config": cfg.neuronpedia,
    }


def test_get_neuronpedia_client_forwards_neuronpedia_config(
    tmp_path, monkeypatch,
):
    """``_get_neuronpedia_client`` must pass ``cfg.neuronpedia`` so per-call
    retries / concurrency are tunable from the top-level config without poking
    the constructor."""
    from steering.eval.neuronpedia_config import NeuronpediaConfig

    captured: dict = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(concept_list, "NeuronpediaClient", _fake_client)
    cfg = make_valid_config(
        run_tag="np_cfg_forward",
        output_root=str(tmp_path),
    ).replace(neuronpedia=NeuronpediaConfig(max_concurrent=2, max_retries=7))
    concept_list._get_neuronpedia_client(cfg)
    assert captured["config"] is cfg.neuronpedia
    assert captured["config"].max_concurrent == 2
    assert captured["config"].max_retries == 7


def test_real_client_and_identity_resolver_isolate_two_sources_under_custom_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.neuronpedia_client as client_mod

    custom_root = tmp_path / "custom-root"
    cfg_a = make_valid_config(
        output_root=str(custom_root),
        run_tag="run-a",
        neuronpedia_model_id="model-a",
        neuronpedia_source="source-a",
    )
    cfg_b = cfg_a.replace(
        run_tag="run-b",
        neuronpedia_model_id="model-b",
        neuronpedia_source="source-b",
    )
    monkeypatch.setenv(client_mod.NEURONPEDIA_API_KEY_ENV, "np-test")

    def fake_http_get(url, *, headers, timeout):
        del headers, timeout
        identity = "/".join(url.split("/")[-3:-1])
        return SimpleNamespace(
            status_code=200,
            headers={},
            text="",
            json=lambda: {
                "explanations": [{"description": f"description-from-{identity}"}]
            },
        )

    monkeypatch.setattr(client_mod.requests, "get", fake_http_get)
    client_a = concept_list._get_neuronpedia_client(cfg_a)
    client_b = concept_list._get_neuronpedia_client(cfg_b)

    assert client_a.get_description(7) == "description-from-model-a/source-a"
    assert client_b.get_description(7) == "description-from-model-b/source-b"
    cache_a = _canonical_neuronpedia_cache(cfg_a)
    cache_b = _canonical_neuronpedia_cache(cfg_b)
    assert cache_a != cache_b
    assert yaml.safe_load(cache_a.read_text(encoding="utf-8")) == {
        "model_id": "model-a",
        "sae_source": "source-a",
        "descriptions": {7: "description-from-model-a/source-a"},
    }
    assert yaml.safe_load(cache_b.read_text(encoding="utf-8")) == {
        "model_id": "model-b",
        "sae_source": "source-b",
        "descriptions": {7: "description-from-model-b/source-b"},
    }
    assert not (custom_root / "neuronpedia_cache.yaml").exists()
    assert not (tmp_path / "eval/axbench_base/run-a").exists()
    assert not (tmp_path / "eval/axbench_base/run-b").exists()


def _canonical_neuronpedia_cache(cfg) -> Path:
    return paths.AxBenchBasePaths.from_config(cfg).neuronpedia_cache(
        model_id=cfg.neuronpedia_model_id,
        sae_source=cfg.neuronpedia_source,
    )


def test_cache_only_requires_canonical_identity_addressed_file(
    tmp_path: Path,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "root"),
        run_tag="cache-only-missing",
        concept_descriptions_cache_only=True,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="requires an existing identity-addressed Neuronpedia cache",
    ):
        concept_list._get_neuronpedia_client(cfg)

    assert not _canonical_neuronpedia_cache(cfg).exists()


def test_cache_only_does_not_accept_legacy_unkeyed_file(
    tmp_path: Path,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "root"),
        run_tag="cache-only-legacy-path",
        concept_descriptions_cache_only=True,
    )
    legacy = Path(cfg.output_root) / "neuronpedia_cache.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("7: legacy-description\n", encoding="utf-8")
    bytes_before = legacy.read_bytes()

    with pytest.raises(
        AxBenchBaseError,
        match="requires an existing identity-addressed Neuronpedia cache",
    ):
        concept_list._get_neuronpedia_client(cfg)

    assert legacy.read_bytes() == bytes_before
    assert not _canonical_neuronpedia_cache(cfg).exists()


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "7: legacy-description\n",
        "model_id: gemma-2-2b\n"
        "sae_source: 8-gemmascope-res-16k\n"
        "descriptions: {}\n",
    ],
)
def test_cache_only_rejects_blank_legacy_or_empty_canonical_cache(
    tmp_path: Path,
    payload: str,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "root"),
        run_tag="cache-only-empty",
        concept_descriptions_cache_only=True,
    )
    cache = _canonical_neuronpedia_cache(cfg)
    cache.parent.mkdir(parents=True)
    cache.write_text(payload, encoding="utf-8")
    bytes_before = cache.read_bytes()

    with pytest.raises(
        AxBenchBaseError,
        match="requires at least one identity-validated description",
    ):
        concept_list._get_neuronpedia_client(cfg)

    assert cache.read_bytes() == bytes_before


def test_cache_only_loads_matching_nonempty_canonical_cache(
    tmp_path: Path,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "root"),
        run_tag="cache-only-valid",
        concept_descriptions_cache_only=True,
    )
    cache = _canonical_neuronpedia_cache(cfg)
    cache.parent.mkdir(parents=True)
    cache.write_text(
        "model_id: gemma-2-2b\n"
        "sae_source: 8-gemmascope-res-16k\n"
        "descriptions:\n"
        "  7: validated-description\n",
        encoding="utf-8",
    )

    client = concept_list._get_neuronpedia_client(cfg)

    assert client.cache_size() == 1
    assert client.get_cached(7) == "validated-description"


def test_cache_only_mismatched_embedded_identity_still_fails_loud(
    tmp_path: Path,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "root"),
        run_tag="cache-only-mismatch",
        concept_descriptions_cache_only=True,
    )
    cache = _canonical_neuronpedia_cache(cfg)
    cache.parent.mkdir(parents=True)
    cache.write_text(
        "model_id: foreign-model\n"
        "sae_source: foreign-source\n"
        "descriptions:\n"
        "  7: foreign-description\n",
        encoding="utf-8",
    )
    bytes_before = cache.read_bytes()

    with pytest.raises(DataLoadError, match="foreign-model.*gemma-2-2b"):
        concept_list._get_neuronpedia_client(cfg)

    assert cache.read_bytes() == bytes_before


def test_run_tag_none_raises(stub_env):
    """``prepare_concept_list`` must guard against a missing ``run_tag``."""
    cfg = _cfg(stub_env["tmp_path"]).replace(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        prepare_concept_list(cfg)


def test_prepare_concept_list_returns_candidate_dataclass_contents_in_json(
    stub_env,
):
    """The fields written to candidate_concepts.json map 1:1 onto
    ``CandidateConcept``. Old fully-judged ``Concept`` fields are not
    accepted in the Stage 0a artifact anymore.
    """
    cfg = _cfg(stub_env["tmp_path"])
    out = prepare_concept_list(cfg)
    raw = json.loads(out.read_text())
    assert len(raw) > 0
    for item in raw:
        # This will raise TypeError if the JSON has extra or missing keys.
        concept = CandidateConcept(**item)
        assert isinstance(concept, CandidateConcept)
        # Spot-check that the expected fields are present and the right type.
        assert isinstance(concept.feature_index, int)
        assert isinstance(concept.layer, int)
        assert isinstance(concept.sae_source, str)
        assert isinstance(concept.description, str)
        assert isinstance(concept.corpus_count, int)
        assert concept.description_source == "neuronpedia"


def test_prepare_concept_list_uses_extraction_layer_from_feature_dir(stub_env):
    """The ``Concept.layer`` field must be derived from
    ``parse_feature_dir(cfg.feature_dir)``, not hardcoded to 8."""
    cfg = _cfg(
        stub_env["tmp_path"],
        feature_dir="out/data/features/s128/l12/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
    )
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    assert len(concepts) > 0
    for c in concepts:
        assert c["layer"] == 12


# ---------------------------------------------------------------------------
# Lazy concept examples QC boundary
# ---------------------------------------------------------------------------


class _DropFidsQCClient:
    """Test double that simulates ``ConceptExamplesQC.filter``.

    Drops concepts whose feature_id is in ``drop_fids`` and builds
    a matching QCReport. Accepts ``target_surviving`` for call-shape
    compatibility with production QC.
    """

    def __init__(self, drop_fids):
        self.drop_fids = set(drop_fids)
        self.filter_calls: list[list[tuple[int, str]]] = []
        self.last_target_surviving: int | None = None

    def filter(
        self,
        candidates,
        target_surviving: int | None = None,
    ):
        self.filter_calls.append(list(candidates))
        self.last_target_surviving = target_surviving
        ordered = list(candidates)
        per_feature: list[QCFeatureScore] = []
        survivors: list[tuple[int, str]] = []
        for fid, desc in ordered:
            if fid in self.drop_fids:
                per_feature.append(
                    QCFeatureScore(
                        feature_index=fid,
                        description=desc,
                        outcome=QcOutcome.REJECTED,
                        score=0,
                        n_examples=5,
                        error_message=None,
                    )
                )
            else:
                per_feature.append(
                    QCFeatureScore(
                        feature_index=fid,
                        description=desc,
                        outcome=QcOutcome.ACCEPTED,
                        score=2,
                        n_examples=5,
                        error_message=None,
                    )
                )
                survivors.append((fid, desc))
        report = QCReport(
            n_total=len(candidates),
            n_accepted=len(survivors),
            n_rejected=len(candidates) - len(survivors),
            n_fetch_errors=0,
            n_judge_errors=0,
            n_empty_examples=0,
            score_histogram={
                2: len(survivors),
                0: len(candidates) - len(survivors),
            },
            per_feature=per_feature,
            judge_min_score=1,
            n_processed=len(candidates),
        )
        return survivors, report


def test_prepare_concept_list_qc_enabled_by_default_but_stage_0a_does_zero_llm(
    stub_env,
):
    """Regression guard: Stage 0a never builds or calls the LLM judge.

    The config flag still defaults to True, but the work moved to Stage
    0c. The stub_env QC factory raises AssertionError if called.
    """
    cfg = _cfg(stub_env["tmp_path"])
    assert cfg.enable_concept_examples_qc is True
    out = prepare_concept_list(cfg)
    rows = json.loads(out.read_text())
    assert len(rows) > 0
    assert stub_env["qc_factory_calls"] == []


def test_prepare_concept_list_shuffles_prefilter_pool_with_concept_list_seed(
    stub_env,
):
    """Stage 0a applies the deterministic shuffle itself after filtering."""
    cfg = _cfg(
        stub_env["tmp_path"], concept_list_seed=1234, max_concepts=7,
    )
    out = prepare_concept_list(cfg)
    rows = json.loads(out.read_text())
    expected = list(range(1, 10))
    random.Random(1234).shuffle(expected)
    assert [row["feature_index"] for row in rows] == expected


def test_prepare_concept_list_caps_neuronpedia_description_fetch(
    stub_env,
):
    """Profiling/dev runs can cap the cold Neuronpedia description fanout."""
    cfg = _cfg(
        stub_env["tmp_path"],
        concept_list_seed=1234,
        concept_description_fetch_limit=4,
    )
    out = prepare_concept_list(cfg)
    client = stub_env["created_clients"][0]
    assert len(client.get_descriptions_calls) == 1
    fetched = client.get_descriptions_calls[0]
    expected_fetched = list(range(1, 10))
    random.Random(1234).shuffle(expected_fetched)
    expected_fetched = expected_fetched[:4]
    assert fetched == expected_fetched
    rows = json.loads(out.read_text())
    expected_rows = list(expected_fetched)
    random.Random(1234).shuffle(expected_rows)
    assert [row["feature_index"] for row in rows] == expected_rows


def test_prepare_concept_list_opts_into_source_mismatch_guard(stub_env):
    """Stage 0a forwards its configured guard thresholds (404 rate, missing
    rate, total-failed rate, absolute floor) to the Neuronpedia client so a
    source/SAE mismatch fails loud (issue #657)."""
    cfg = _cfg(
        stub_env["tmp_path"],
        max_not_published_description_rate=0.25,
        max_missing_description_rate=0.6,
        max_failed_description_rate=0.7,
        min_source_mismatch_features=7,
    )
    prepare_concept_list(cfg)
    client = stub_env["created_clients"][0]
    assert client.max_not_published_rate_calls == [0.25]
    assert client.max_missing_rate_calls == [0.6]
    assert client.max_failed_rate_calls == [0.7]
    assert client.min_mismatch_features_calls == [7]


def test_prepare_concept_list_propagates_source_mismatch(stub_env, monkeypatch):
    """When the client's mismatch guard fires, Stage 0a fails loud rather
    than swallowing it into a truncated concept pool (issue #657)."""

    class _MismatchClient:
        def __init__(self) -> None:
            self.model_id = "gemma-2-2b"

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
            raise NeuronpediaSourceMismatchError(
                "49018/65355 (75.0%) returned 404 / not-published"
            )

    monkeypatch.setattr(
        concept_list, "_get_neuronpedia_client", lambda cfg: _MismatchClient(),
    )
    cfg = _cfg(stub_env["tmp_path"])
    with pytest.raises(NeuronpediaSourceMismatchError, match="not-published"):
        prepare_concept_list(cfg)


def test_prepare_concept_list_qc_disabled_same_stage_0a_behavior(
    stub_env, monkeypatch,
):
    """Disabling QC no longer changes Stage 0a; it still emits the same
    pre-QC pool and leaves all LLM work to Stage 0c.
    """
    called = {"n": 0}

    def _unexpected_factory(cfg, response_cache=None):
        called["n"] += 1
        return _DropFidsQCClient(drop_fids=[])

    monkeypatch.setattr(
        concept_list, "_build_concept_examples_qc_client", _unexpected_factory,
    )
    cfg_enabled = _cfg(
        stub_env["tmp_path"], run_tag="qc_enabled_boundary",
    )
    cfg_disabled = _cfg(
        stub_env["tmp_path"], run_tag="qc_disabled_boundary",
    ).replace(enable_concept_examples_qc=False)
    out_enabled = prepare_concept_list(cfg_enabled)
    out = prepare_concept_list(cfg_disabled)
    assert called["n"] == 0
    assert json.loads(out.read_text()) == json.loads(out_enabled.read_text())


def test_prepare_concept_list_does_not_write_qc_report(stub_env, monkeypatch):
    """The QC report moved to the Stage 0c epilogue."""
    cfg = _cfg(stub_env["tmp_path"])
    prepare_concept_list(cfg)
    report_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    assert not report_path.exists()


def test_prepare_concept_list_does_not_open_synth_response_cache(stub_env):
    """The SynthResponseCache is opened by Stage 0c, not Stage 0a."""
    cfg = _cfg(stub_env["tmp_path"])
    prepare_concept_list(cfg)
    assert not paths.AxBenchBasePaths.from_config(cfg).synth_response_cache().exists()


def test_build_concept_examples_qc_client_wires_config(tmp_path, monkeypatch):
    """``_build_concept_examples_qc_client`` must construct the OpenRouter client
    and wrap it via ``make_llm_concept_examples_judge``, forwarding all QC fields
    from the config.
    """
    captured = {}

    class _SentinelClient:
        def __init__(self, cfg, response_cache=None):
            captured["llm_client_cfg"] = cfg
            captured["response_cache"] = response_cache

    def _fake_make_judge(client, max_examples_in_prompt, *, model):
        captured["judge_built_with"] = {
            "client": client,
            "max_examples_in_prompt": max_examples_in_prompt,
            "model": model,
        }

        def _stub_judge(desc, examples):
            return 2

        return _stub_judge

    class _SentinelTokenizer:
        name = "sentinel"

    def _fake_get_tokenizer(tokenizer_id):
        captured["tokenizer_id"] = tokenizer_id
        return _SentinelTokenizer()

    monkeypatch.setattr(concept_list, "OpenRouterLLMClient", _SentinelClient)
    monkeypatch.setattr(concept_list, "make_llm_concept_examples_judge", _fake_make_judge)
    monkeypatch.setattr(concept_list, "get_extraction_tokenizer", _fake_get_tokenizer)
    cfg = make_valid_config(
        run_tag="qc_wire_test",
        output_root=str(tmp_path / "out"),
        enable_concept_examples_qc=True,
        n_activation_examples=7,
        activation_example_judge_min_score=2,
    )
    qc = concept_list._build_concept_examples_qc_client(cfg)
    assert isinstance(qc, ConceptExamplesQC)
    assert qc._n_examples == 7
    assert qc._min_score == 2
    assert qc._model_id == cfg.neuronpedia_model_id
    assert qc._sae_source == cfg.neuronpedia_source
    assert isinstance(qc._tokenizer, _SentinelTokenizer)
    assert captured["tokenizer_id"] == cfg.extraction_tokenizer_id
    assert isinstance(captured["llm_client_cfg"], type(cfg.concept_examples_qc_judge))
    assert captured["judge_built_with"]["max_examples_in_prompt"] == 7
    assert captured["judge_built_with"]["model"] == cfg.concept_examples_qc_judge.model
    # Default call with no explicit cache -> response_cache is None.
    assert captured["response_cache"] is None

    # And when a cache is passed, it's threaded through to the LLM client.
    sentinel_cache = object()
    concept_list._build_concept_examples_qc_client(
        cfg, response_cache=sentinel_cache,
    )
    assert captured["response_cache"] is sentinel_cache


def test_build_concept_examples_qc_client_wraps_api_key_error(tmp_path, monkeypatch):
    """When OpenRouterLLMClient raises AxBenchBaseError (missing API key),
    the factory must re-raise with a message pointing at the config toggle.
    """

    class _SentinelTokenizer:
        pass

    def _raising_client(cfg, response_cache=None):
        raise AxBenchBaseError("OPENROUTER_API_KEY not set in environment")

    monkeypatch.setattr(concept_list, "OpenRouterLLMClient", _raising_client)
    monkeypatch.setattr(
        concept_list,
        "get_extraction_tokenizer",
        lambda tid: _SentinelTokenizer(),
    )
    cfg = make_valid_config(
        run_tag="qc_api_key_err",
        output_root=str(tmp_path / "out"),
        enable_concept_examples_qc=True,
    )
    with pytest.raises(AxBenchBaseError) as exc_info:
        concept_list._build_concept_examples_qc_client(cfg)
    msg = str(exc_info.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "enable_concept_examples_qc" in msg


def test_cache_only_defaults_false_and_passes_through(stub_env):
    """Default config fetches normally (cache_only=False reaches the client)."""
    cfg = _cfg(stub_env["tmp_path"])
    assert cfg.concept_descriptions_cache_only is False
    prepare_concept_list(cfg)
    client = stub_env["created_clients"][0]
    assert client.cache_only_calls == [False]


def test_cache_only_true_restricts_selection_to_cached_features(stub_env):
    """cache_only=True passes through and selects ONLY from cached features."""
    # Simulate a partial on-disk cache: only fids 1, 2, 3 are cached.
    stub_env["cache_holder"]["cached_fids"] = {1, 2, 3}
    cfg = _cfg(stub_env["tmp_path"], concept_descriptions_cache_only=True)
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    client = stub_env["created_clients"][0]
    assert client.cache_only_calls == [True]
    selected = {c["feature_index"] for c in concepts}
    assert selected
    assert selected.issubset({1, 2, 3})


def test_cache_only_reproduces_full_fetch_of_same_cached_set(stub_env):
    """cache_only over a cache K == a full fetch that returns exactly K.

    With every describable feature cached, cache_only selection must be
    byte-identical to the normal (network) selection -- the reproduction
    guarantee that lets us regenerate prior concepts without re-querying.
    """
    # cached_fids=None means "all describable are cached" in the fake.
    stub_env["cache_holder"]["cached_fids"] = None
    cfg_full = _cfg(stub_env["tmp_path"], concept_descriptions_cache_only=False)
    full = json.loads(prepare_concept_list(cfg_full).read_text())
    cfg_cache = _cfg(stub_env["tmp_path"], concept_descriptions_cache_only=True)
    cached = json.loads(prepare_concept_list(cfg_cache).read_text())
    assert [c["feature_index"] for c in cached] == [c["feature_index"] for c in full]


# ---------------------------------------------------------------------------
# concept_feature_whitelist_path (Stage 0a file application)
# ---------------------------------------------------------------------------


def _write_txt_whitelist(stub_env, ids: list[int], *, name: str) -> Path:
    path = stub_env["tmp_path"] / name
    path.write_text("".join(f"{feature_id}\n" for feature_id in ids))
    return path


@contextmanager
def _capture_loguru_records():
    """Capture (level_name, message) tuples for every loguru record at
    INFO+ emitted during the block.

    Loguru does not propagate to pytest's ``caplog``, so tests must add
    a temporary sink (mirrors ``_capture_loguru_warnings`` in
    ``test_runtime_state.py``). Unlike the shared ``capture_loguru_info``
    helper (which discards level metadata), this keeps the level name so
    tests can select the ``info`` drop-count logs.
    """
    records: list[tuple[str, str]] = []
    handler_id = loguru_logger.add(
        lambda msg: records.append(
            (msg.record["level"].name, msg.record["message"])
        ),
        level="INFO",
    )
    try:
        yield records
    finally:
        loguru_logger.remove(handler_id)


def test_whitelist_path_restricts_and_fetches_only_whitelisted(stub_env):
    path = _write_txt_whitelist(stub_env, [2, 4, 6], name="restrict.txt")
    cfg = _cfg(
        stub_env["tmp_path"],
        concept_feature_whitelist_path=str(path),
        max_concepts=1,
    )
    out = prepare_concept_list(cfg)
    concepts = json.loads(out.read_text())
    fids = {c["feature_index"] for c in concepts}
    assert fids == {2, 4, 6}
    client = stub_env["created_clients"][0]
    assert sorted(client.get_descriptions_calls[0]) == [2, 4, 6]  # no non-whitelisted fetch


def test_whitelist_txt_path(stub_env):
    p = stub_env["tmp_path"] / "accepted_feature_ids.txt"
    p.write_text("2\n4\n")
    cfg = _cfg(stub_env["tmp_path"], concept_feature_whitelist_path=str(p), max_concepts=1)
    concepts = json.loads(prepare_concept_list(cfg).read_text())
    assert {c["feature_index"] for c in concepts} == {2, 4}


def test_whitelist_yaml_path(stub_env):
    p = stub_env["tmp_path"] / "manual_feature_filter.yaml"
    p.write_text('# - "00009"  # score=0.9\n- "00002"\n- 4\n')
    cfg = _cfg(stub_env["tmp_path"], concept_feature_whitelist_path=str(p), max_concepts=1)
    concepts = json.loads(prepare_concept_list(cfg).read_text())
    assert {c["feature_index"] for c in concepts} == {2, 4}


def test_whitelist_drops_below_count_and_logs(stub_env):
    # fid 0 has count 5 (< 10); it is whitelisted but must be dropped by count.
    path = _write_txt_whitelist(stub_env, [0, 2, 4, 6], name="below_count.txt")
    cfg = _cfg(
        stub_env["tmp_path"],
        concept_feature_whitelist_path=str(path),
        max_concepts=3,
    )
    with _capture_loguru_records() as records:
        concepts = json.loads(prepare_concept_list(cfg).read_text())
    fids = {c["feature_index"] for c in concepts}
    assert 0 not in fids
    assert fids == {2, 4, 6}

    infos = [msg for level, msg in records if level == "INFO"]
    assert any(
        "1 whitelisted IDs dropped by min_corpus_count=10" in msg
        for msg in infos
    ), f"expected min_corpus_count drop info log, got {infos!r}"


def test_whitelist_drops_short_description_and_logs(stub_env):
    stub_env["descriptions_holder"]["descriptions"][8] = "short"
    path = _write_txt_whitelist(stub_env, [2, 4, 6, 8], name="short_desc.txt")
    cfg = _cfg(
        stub_env["tmp_path"],
        concept_feature_whitelist_path=str(path),
        max_concepts=3,
    )
    with _capture_loguru_records() as records:
        concepts = json.loads(prepare_concept_list(cfg).read_text())
    assert {c["feature_index"] for c in concepts} == {2, 4, 6}

    infos = [msg for level, msg in records if level == "INFO"]
    assert any(
        "1 whitelisted IDs dropped during Neuronpedia description "
        "fetch/filter" in msg
        for msg in infos
    ), f"expected description-fetch drop info log, got {infos!r}"


def test_whitelist_id_absent_from_stats_fails_before_client_or_write(stub_env):
    path = _write_txt_whitelist(stub_env, [2, 20], name="absent_from_stats.txt")
    cfg = _cfg(
        stub_env["tmp_path"],
        run_tag="absent_from_stats",
        concept_feature_whitelist_path=str(path),
        max_concepts=1,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        prepare_concept_list(cfg)

    message = str(exc_info.value)
    assert "absent_ids=[20]" in message
    assert "feature_statistics_path='stats.safetensors'" in message
    assert "dead feature" in message
    assert "wrong SAE or layer" in message
    assert stub_env["created_clients"] == []
    assert not test_paths.default_paths("absent_from_stats").candidate_concepts_json().exists()


def test_whitelist_multiple_absent_ids_reported_without_mutating_existing_output(
    stub_env,
):
    path = _write_txt_whitelist(
        stub_env,
        [30, 2, 20],
        name="multiple_absent_from_stats.txt",
    )
    run_tag = "multiple_absent_from_stats"
    existing = test_paths.default_paths(run_tag).candidate_concepts_json()
    existing.parent.mkdir(parents=True, exist_ok=True)
    sentinel = b'[{"preexisting": true}]\n'
    existing.write_bytes(sentinel)
    cfg = _cfg(
        stub_env["tmp_path"],
        run_tag=run_tag,
        concept_feature_whitelist_path=str(path),
        max_concepts=1,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        prepare_concept_list(cfg)

    assert "absent_ids=[20, 30]" in str(exc_info.value)
    assert stub_env["created_clients"] == []
    assert existing.read_bytes() == sentinel


def test_whitelist_all_below_count_fails_before_client_or_write(stub_env):
    path = _write_txt_whitelist(stub_env, [0], name="all_below_count.txt")
    run_tag = "all_below_count"
    cfg = _cfg(
        stub_env["tmp_path"],
        run_tag=run_tag,
        concept_feature_whitelist_path=str(path),
        max_concepts=1,
    )

    with pytest.raises(AxBenchBaseError, match="none pass min_corpus_count"):
        prepare_concept_list(cfg)

    assert stub_env["created_clients"] == []
    assert not test_paths.default_paths(run_tag).candidate_concepts_json().exists()


def test_whitelist_fast_fail_when_pool_below_max_concepts(stub_env):
    path = _write_txt_whitelist(stub_env, [2, 4], name="too_small.txt")
    cfg = _cfg(
        stub_env["tmp_path"],
        concept_feature_whitelist_path=str(path),
        max_concepts=5,
    )
    with pytest.raises(AxBenchBaseError, match="max_concepts=5"):
        prepare_concept_list(cfg)


def test_whitelist_fetch_limit_below_max_fails_before_client_or_write(stub_env):
    path = _write_txt_whitelist(
        stub_env,
        [2, 4, 6, 8],
        name="fetch_limit_too_small.txt",
    )
    run_tag = "fetch_limit_too_small"
    cfg = _cfg(
        stub_env["tmp_path"],
        run_tag=run_tag,
        concept_feature_whitelist_path=str(path),
        concept_description_fetch_limit=2,
        max_concepts=3,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=(
            r"concept_description_fetch_limit=2 leaves only 2 candidate\(s\)"
            r".*max_concepts=3"
        ),
    ):
        prepare_concept_list(cfg)

    assert stub_env["created_clients"] == []
    assert not test_paths.default_paths(run_tag).candidate_concepts_json().exists()


def test_whitelist_description_filter_below_max_fails_without_output(stub_env):
    stub_env["descriptions_holder"]["descriptions"][4] = "short"
    path = _write_txt_whitelist(stub_env, [2, 4], name="description_too_small.txt")
    run_tag = "description_too_small"
    cfg = _cfg(
        stub_env["tmp_path"],
        run_tag=run_tag,
        concept_feature_whitelist_path=str(path),
        max_concepts=2,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"only 1 candidate\(s\) with quality descriptions.*max_concepts=2",
    ):
        prepare_concept_list(cfg)

    assert len(stub_env["created_clients"]) == 1
    assert stub_env["created_clients"][0].get_descriptions_calls == [[2, 4]]
    assert not test_paths.default_paths(run_tag).candidate_concepts_json().exists()


def test_whitelist_empty_selection_raises(stub_env):
    p = stub_env["tmp_path"] / "manual_feature_filter.yaml"
    p.write_text("# - 2  # all commented\n")
    cfg = _cfg(stub_env["tmp_path"], concept_feature_whitelist_path=str(p))
    with pytest.raises(AxBenchBaseError, match="empty selection"):
        prepare_concept_list(cfg)


def test_whitelist_out_of_range_raises(stub_env):
    path = _write_txt_whitelist(stub_env, [99999], name="out_of_range.txt")
    cfg = _cfg(stub_env["tmp_path"], concept_feature_whitelist_path=str(path))
    with pytest.raises(AxBenchBaseError, match="out of range"):
        prepare_concept_list(cfg)


def test_permissive_whitelist_byte_identical_to_no_whitelist(stub_env):
    base = _cfg(stub_env["tmp_path"], run_tag="perm_none")
    none_bytes = prepare_concept_list(base).read_bytes()
    # max_concepts=9 caps at the natural full pool size (fids 1..9 survive
    # min_corpus_count=10 in stub_env) so the whitelist-active
    # pool-size fast-fail (a NEW guard, orthogonal to this test's AND-clause
    # byte-identity assertion) doesn't fire merely because the default
    # _cfg max_concepts=500 exceeds the stub environment's 10-fid universe.
    path = _write_txt_whitelist(
        stub_env, list(range(10)), name="permissive.txt"
    )
    wl = _cfg(
        stub_env["tmp_path"],
        run_tag="perm_wl",
        concept_feature_whitelist_path=str(path),
        max_concepts=9,
    )
    wl_bytes = prepare_concept_list(wl).read_bytes()
    assert wl_bytes == none_bytes
