"""Stage 0 (prepare) end-to-end integration tests.

Exercises the three Stage 0 entry points using tiny fixtures and
monkey-patched external dependencies:

* Stage 0a ``prepare_concept_list``  — Neuronpedia mocked.
* Stage 0b ``prepare_seed_pools``    — real tokenizer on tiny shard.
* Stage 0c ``prepare_negative_prefixes`` — Gemma generation + LLM QC stubbed.

Each test runs in isolation against ``tmp_path`` so outputs land under
``tmp_run_root`` and do not collide across runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch as t

from nanochat.common import get_base_dir
from steering.eval.axbench_base import (
    _stage0c_internals as stage0c_internals_mod,
)
from steering.eval.axbench_base import (
    concept_list as concept_list_mod,
)
from steering.eval.axbench_base import (
    extraction as extraction_mod,
)
from steering.eval.axbench_base import (
    seed_pools as seed_pools_mod,
)
from steering.eval.axbench_base import (
    stage0c_runner as stage0c_runner_mod,
)
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
)
from steering.eval.axbench_base.concept_examples_qc import (
    QCFeatureScore,
    QcOutcome,
    QCReport,
)
from steering.eval.axbench_base.concept_list import prepare_concept_list
from steering.eval.axbench_base.negative_prefixes import prepare_negative_prefixes
from steering.eval.axbench_base.seed_pools import prepare_seed_pools
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    RejectedConceptReason,
    TargetModelMetadata,
)
from steering.eval.axbench_base.validate import (
    load_and_validate_candidate_concepts,
    load_and_validate_concepts,
)
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
from tests.axbench_base.fixtures.build_tiny_shard import TINY_TEXTS, build_tiny_shard
from tests.runtime_attribute_helpers import set_runtime_attribute
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    """Auto-accepting QC client matching ``ConceptExamplesQC.filter``."""

    def filter(
        self,
        candidates: list[tuple[int, str]],
        target_surviving: int | None = None,
    ) -> tuple[list[tuple[int, str]], QCReport]:
        del target_surviving
        ordered = list(candidates)
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


def _candidate_from_concept(concept: Concept) -> CandidateConcept:
    return CandidateConcept(
        feature_index=concept.feature_index,
        layer=concept.layer,
        sae_source=concept.sae_source,
        description=concept.description,
        corpus_count=concept.corpus_count,
        description_source=concept.description_source,
    )


class _FakeNeuronpediaClient:
    """Neuronpedia stand-in: returns one canned description per feature id.

    Mirrors the real client's "drop on NeuronpediaError" contract for
    ``get_descriptions`` so a missing fid is silently absent from the
    returned dict (rather than raising and aborting the batch). Uses
    the same exception class as the real client so the fake is faithful
    to production semantics — broader catches would hide programming
    bugs in callers."""

    def __init__(self, descriptions: dict[int, str]) -> None:
        self._descriptions = descriptions

    def get_description(self, feature_id: int) -> str:
        if feature_id not in self._descriptions:
            from steering.eval.neuronpedia_client import NeuronpediaFetchError
            raise NeuronpediaFetchError(f"feature {feature_id} missing")
        return self._descriptions[feature_id]

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
        # These mirror the real client's signature (offline/repro path + the
        # issue-#657 source-mismatch guards). The fake serves every canned
        # description from its in-memory map, so there is no network tier to
        # skip and no 404/empty/failed rate to guard. Accepted for call-site
        # fidelity.
        del cache_only, max_not_published_rate
        del max_missing_rate, max_failed_rate, min_mismatch_features
        from steering.eval.neuronpedia_client import NeuronpediaError
        out: dict[int, str] = {}
        for fid in feature_ids:
            try:
                out[fid] = self.get_description(fid)
            except NeuronpediaError:
                continue
        return out


def _find_real_tokenizer_dir() -> Path | None:
    """Locate a cached nanochat tokenizer directory on disk.

    Mirrors the search strategy used by the Stage 0b unit test: the
    real nanochat tokenizer is required because ``prepare_seed_pools``
    calls ``get_tokenizer()`` which reads from
    ``$NANOCHAT_BASE_DIR/tokenizer``.
    """
    real_base_env = os.environ.get("NANOCHAT_BASE_DIR", "")
    candidates: list[Path] = []
    if real_base_env:
        candidates.append(Path(real_base_env) / "tokenizer")
    candidates.append(Path(get_base_dir()) / "tokenizer")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "nanochat_cache" / "tokenizer"
        if candidate.exists():
            candidates.append(candidate)
    for c in candidates:
        if c.exists() and (c / "tokenizer.pkl").exists():
            return c
    return None


def _link_real_tokenizer_into(tmp_base: Path) -> None:
    """Symlink a real nanochat tokenizer into ``tmp_base/tokenizer``."""
    real_tok_dir = _find_real_tokenizer_dir()
    if real_tok_dir is None:
        pytest.skip(
            "Real nanochat tokenizer not found in $NANOCHAT_BASE_DIR, "
            "get_base_dir(), or any ancestor nanochat_cache/; Stage 0b "
            "integration tests require a cached tokenizer.",
        )
    link_target = tmp_base / "tokenizer"
    if not link_target.exists():
        link_target.symlink_to(real_tok_dir)


def _stub_target_meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _base_model(n_ctx: int = 8192) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(n_ctx=n_ctx))


def _stage0b_extraction_pair() -> tuple[TinyHookedTransformer, TinySAE]:
    """Return a deterministic pair that exercises production ``process_shard``."""

    t.manual_seed(0)
    model = TinyHookedTransformer(d_model=16, vocab_size=1024, device="cpu")
    sae = TinySAE(d_model=16, n_features=64, device="cpu")
    set_runtime_attribute(sae.cfg, "hook_name", "blocks.8.hook_resid_post")
    set_runtime_attribute(
        model,
        "state_dict",
        lambda: {"dummy": model._dummy_param, "embedding": model._embedding},
    )
    set_runtime_attribute(
        sae,
        "state_dict",
        lambda: {
            "dummy": sae._dummy_param,
            "W_enc": sae.W_enc,
            "b_enc": sae.b_enc,
        },
    )
    return model, sae


class _FakeExtractionTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        body = [i + 10 for i, _ in enumerate(text)]
        if add_special_tokens:
            return [self.bos_token_id, *body]
        return body

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{tok}" for tok in token_ids)


class _FakeGemmaGenerator:
    """Deterministic stream: yields a small fixed token list per call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, base_model: Any, token_ids: list[int], **kwargs: Any) -> Any:
        self.calls.append({
            "base_model": base_model,
            "token_ids": list(token_ids),
            **kwargs,
        })
        yield from [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Stage 0a tests
# ---------------------------------------------------------------------------


def test_stage_0a_produces_candidate_concepts_json(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 0a writes the full candidate pool using mocked Neuronpedia."""
    stats_path = tmp_run_root / "stats.safetensors"
    # Dummy file so the AxBenchBaseConfig path validation (if any) passes;
    # the real reader is monkey-patched below.
    stats_path.write_bytes(b"stub")

    per_feature = {
        str(i): _make_per_feature_stat(count=50) for i in range(10)
    }
    stats = _make_stats(per_feature, n_sae_features=16384)
    descriptions = {
        i: f"Concept description number {i} (long enough)" for i in range(10)
    }

    monkeypatch.setattr(
        concept_list_mod,
        "_load_feature_statistics",
        lambda path: stats,
    )
    monkeypatch.setattr(
        concept_list_mod,
        "_get_neuronpedia_client",
        lambda cfg: _FakeNeuronpediaClient(descriptions),
    )
    monkeypatch.setattr(
        concept_list_mod,
        "_build_concept_examples_qc_client",
        lambda cfg, response_cache=None: (_ for _ in ()).throw(
            AssertionError("Stage 0a should not call LLM")
        ),
    )

    cfg = make_valid_axbench_cfg(
        feature_statistics_path=str(stats_path),
        run_tag="stage0a_e2e",
        max_concepts=5,
        min_description_length=5,
        min_corpus_count=10,
    )

    out_path = prepare_concept_list(cfg)

    assert out_path == test_paths.default_paths("stage0a_e2e").candidate_concepts_json()
    assert out_path.exists()
    assert not test_paths.default_paths("stage0a_e2e").concepts_json().exists()
    loaded = load_and_validate_candidate_concepts(out_path, n_sae_features=16384)
    assert len(loaded) == 10
    for c in loaded:
        assert isinstance(c, CandidateConcept)
        assert c.description_source == "neuronpedia"
        assert c.description.strip()
    assert all(not hasattr(c, "concept_id") for c in loaded)


def test_stage_0b_produces_both_train_and_eval_pools(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 0b writes both ``seeds_train`` and ``seeds_eval`` pools."""
    _link_real_tokenizer_into(tmp_run_root)
    data_dir = tmp_run_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, _feature_shard = build_tiny_shard(data_dir, shard_idx=0)

    monkeypatch.setattr(
        seed_pools_mod, "_list_parquets_for_feature_dir",
        lambda cfg: [parquet_path],
    )
    cfg = make_valid_axbench_cfg(
        run_tag="stage0b_e2e",
        validation_shards=[0],
        seed_prefix_max_nanochat_tokens=32,
        seed_continuation_nanochat_tokens=4,
        seed_pool_train_ratio=0.7,
        seed_pool_seed=42,
        seed_extraction_gpu_batch_size=1,
    )

    base_model, base_sae = _stage0b_extraction_pair()
    train_path, eval_path = prepare_seed_pools(
        cfg,
        base_model=base_model,
        base_sae=base_sae,
    )

    assert train_path == test_paths.default_paths("stage0b_e2e").seeds_train()
    assert eval_path == test_paths.default_paths("stage0b_e2e").seeds_eval()
    assert train_path.exists() and train_path.stat().st_size > 0
    assert eval_path.exists() and eval_path.stat().st_size > 0

    from steering.eval.axbench_base.seed_pools import load_seed_pool
    tokenizer = seed_pools_mod.get_tokenizer()
    train_rows = load_seed_pool(train_path, tokenizer=tokenizer)
    eval_rows = load_seed_pool(eval_path, tokenizer=tokenizer)
    assert len(train_rows) + len(eval_rows) == len(TINY_TEXTS)
    assert len(train_rows) >= 1
    assert len(eval_rows) >= 1


def test_stage_0c_rejects_when_no_prefixes_verified(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 0c records rejection and raises when no candidates verify."""
    _link_real_tokenizer_into(tmp_run_root)
    data_dir = tmp_run_root / "data"
    feature_dir = tmp_run_root / "features"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, feature_shard_source = build_tiny_shard(data_dir, shard_idx=0)
    feature_shard = feature_dir / feature_shard_source.name
    feature_shard_source.rename(feature_shard)

    candidate = CandidateConcept(
        feature_index=0,  # feature 0 is present in docs 0 and 5
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="test concept",
        corpus_count=100,
        description_source="neuronpedia",
    )
    concepts_path = tmp_run_root / "candidate_concepts.json"
    concepts_path.write_text(json.dumps([{
        "feature_index": 0,
        "layer": 8,
        "sae_source": "8-gemmascope-res-16k",
        "description": "test concept",
        "corpus_count": 100,
        "description_source": "neuronpedia",
    }]))

    monkeypatch.setattr(
        stage0c_internals_mod, "_list_parquets_for_feature_dir",
        lambda cfg: [parquet_path],
    )
    monkeypatch.setattr(
        stage0c_runner_mod, "glob_features",
        lambda cfg: [feature_shard],
    )
    fake_generator = _FakeGemmaGenerator()

    def _fake_batched_generate(
        base_model: Any,
        token_ids: list[int],
        *,
        seeds: list[int],
        **kwargs: Any,
    ) -> list[list[int]]:
        return [
            list(fake_generator(base_model, token_ids, seed=seed, **kwargs))
            for seed in seeds
        ]

    monkeypatch.setattr(
        stage0c_internals_mod,
        "batched_generate_seeds",
        _fake_batched_generate,
    )

    def _fake_cross_prefix_batched_generate(
        base_model: Any,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **kwargs: Any,
    ) -> list[list[list[int]]]:
        # Stage 0c verification routes through the cross-prefix kernel at
        # the default verification_cross_prefix_batch_size (> 1), so it
        # must be stubbed too; mirror the per-seed fake per prefix to match
        # production's nested ``[prefix][seed][token]`` return shape.
        return [
            [
                list(fake_generator(base_model, token_ids, seed=seed, **kwargs))
                for seed in seeds
            ]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(
        stage0c_internals_mod,
        "cross_prefix_batched_generate_seeds",
        _fake_cross_prefix_batched_generate,
    )
    expected_tokenizer_id = "custom/stage0c-reject-tokenizer"
    tokenizer_ids: list[str] = []

    def _fake_extraction_tokenizer(tokenizer_id):
        tokenizer_ids.append(tokenizer_id)
        assert tokenizer_id == expected_tokenizer_id
        return _FakeExtractionTokenizer()

    monkeypatch.setattr(
        stage0c_internals_mod,
        "get_extraction_tokenizer",
        _fake_extraction_tokenizer,
        raising=False,
    )
    # Force verifications to always fail by returning below-threshold percentile.
    qc_tokenizer_ids: list[str] = []

    def _fake_compute_percentiles(
        prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id,
    ):
        del prefix, cont, f_c, gemma, sae, table, sae_layer
        qc_tokenizer_ids.append(tokenizer_id)
        assert tokenizer_id == expected_tokenizer_id
        return (10.0, 10.0)

    monkeypatch.setattr(
        stage0c_internals_mod,
        "compute_fc_percentiles",
        _fake_compute_percentiles,
    )
    monkeypatch.setattr(
        extraction_mod, "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: t.zeros(16384, 99),
    )
    monkeypatch.setattr(
        stage0c_runner_mod, "load_and_validate_candidate_concepts",
        lambda path, n_sae_features: [candidate],
        raising=False,
    )
    monkeypatch.setattr(
        concept_list_mod,
        "_build_concept_examples_qc_client",
        lambda cfg, response_cache=None: _AcceptAllQCClient(),
        raising=False,
    )
    monkeypatch.setattr(
        extraction_mod, "_infer_n_sae_features",
        lambda base_sae: 16384,
    )

    cfg = make_valid_axbench_cfg(
        run_tag="stage0c_reject",
        validation_shards=[0],
        n_verification_seeds=3,
        min_consistency_passes=2,
        verification_percentile_threshold=0.7,
        verification_length=3,
        min_prefix_chars=1,
        max_prefix_chars=2048,
        n_verified_prefixes_per_concept_target=2,
        extraction_tokenizer_id=expected_tokenizer_id,
    )

    with pytest.raises(AxBenchBaseError, match="accepted 0 from 1 candidates"):
        prepare_negative_prefixes(
            cfg=cfg,
            concepts_path=concepts_path,
            base_model=_base_model(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )

    assert not test_paths.default_paths("stage0c_reject").negative_mode_prefixes().exists()
    assert tokenizer_ids
    assert qc_tokenizer_ids
    rejected = json.loads(
        test_paths.default_paths("stage0c_reject").rejected_concepts_json().read_text()
    )
    assert rejected == [
        {
            "candidate_index": 0,
            "feature_index": 0,
            "description": "test concept",
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": 2,
        }
    ]


# ---------------------------------------------------------------------------
# Stage 0 full end-to-end
# ---------------------------------------------------------------------------


def test_stage_0_prepare_end_to_end(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run Stages 0a -> 0b -> 0c back-to-back against a tiny fixture.

    Asserts that Stage 0a's candidate pool and Stage 0c's final concept
    list both round-trip through validation, Stage 0b's pools have >= 1
    row each, and Stage 0c's parquet contains at least one row per final concept.
    """
    _link_real_tokenizer_into(tmp_run_root)

    # --- Stage 0a setup (Neuronpedia + stats) ---
    stats_path = tmp_run_root / "stats.safetensors"
    stats_path.write_bytes(b"stub")
    # Use features 0 and 10 because the tiny shard pre-extracts those ids
    # in docs 0 and 5 respectively. Counts >= 10 so both survive the count
    # filter.
    per_feature = {
        str(fid): _make_per_feature_stat(count=42) for fid in [0, 10]
    }
    stats = _make_stats(per_feature, n_sae_features=16384)
    descriptions = {
        0: "the canonical concept zero description (long enough)",
        10: "the canonical concept ten description (long enough)",
    }

    monkeypatch.setattr(
        concept_list_mod,
        "_load_feature_statistics",
        lambda path: stats,
    )
    monkeypatch.setattr(
        concept_list_mod,
        "_get_neuronpedia_client",
        lambda cfg: _FakeNeuronpediaClient(descriptions),
    )
    monkeypatch.setattr(
        concept_list_mod,
        "_build_concept_examples_qc_client",
        lambda cfg, response_cache=None: (_ for _ in ()).throw(
            AssertionError("Stage 0a should not call LLM")
        ),
    )

    # --- Stage 0b setup (tiny shard) ---
    data_dir = tmp_run_root / "data"
    feature_dir = tmp_run_root / "features"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, feature_shard_source = build_tiny_shard(data_dir, shard_idx=0)
    feature_shard = feature_dir / feature_shard_source.name
    feature_shard_source.rename(feature_shard)

    monkeypatch.setattr(
        seed_pools_mod, "_list_parquets_for_feature_dir",
        lambda cfg: [parquet_path],
    )
    # --- Stage 0c setup (fake Gemma generator + always-passing QC) ---
    monkeypatch.setattr(
        stage0c_internals_mod, "_list_parquets_for_feature_dir",
        lambda cfg: [parquet_path],
    )
    monkeypatch.setattr(
        stage0c_runner_mod, "glob_features",
        lambda cfg: [feature_shard],
    )
    fake_generator = _FakeGemmaGenerator()

    def _fake_batched_generate(
        base_model: Any,
        token_ids: list[int],
        *,
        seeds: list[int],
        **kwargs: Any,
    ) -> list[list[int]]:
        return [
            list(fake_generator(base_model, token_ids, seed=seed, **kwargs))
            for seed in seeds
        ]

    monkeypatch.setattr(
        stage0c_internals_mod,
        "batched_generate_seeds",
        _fake_batched_generate,
    )

    def _fake_cross_prefix_batched_generate(
        base_model: Any,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **kwargs: Any,
    ) -> list[list[list[int]]]:
        # Stage 0c verification routes through the cross-prefix kernel at
        # the default verification_cross_prefix_batch_size (> 1), so it
        # must be stubbed too; mirror the per-seed fake per prefix to match
        # production's nested ``[prefix][seed][token]`` return shape.
        return [
            [
                list(fake_generator(base_model, token_ids, seed=seed, **kwargs))
                for seed in seeds
            ]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(
        stage0c_internals_mod,
        "cross_prefix_batched_generate_seeds",
        _fake_cross_prefix_batched_generate,
    )
    expected_tokenizer_id = "custom/stage0c-e2e-tokenizer"
    tokenizer_ids: list[str] = []

    def _fake_extraction_tokenizer(tokenizer_id):
        tokenizer_ids.append(tokenizer_id)
        assert tokenizer_id == expected_tokenizer_id
        return _FakeExtractionTokenizer()

    monkeypatch.setattr(
        stage0c_internals_mod,
        "get_extraction_tokenizer",
        _fake_extraction_tokenizer,
        raising=False,
    )
    # Always-passing: raw percentile 85 > 0.7 * 100.
    qc_tokenizer_ids: list[str] = []

    def _fake_compute_percentiles(
        prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id,
    ):
        del prefix, cont, f_c, gemma, sae, table, sae_layer
        qc_tokenizer_ids.append(tokenizer_id)
        assert tokenizer_id == expected_tokenizer_id
        return (10.0, 85.0)

    monkeypatch.setattr(
        stage0c_internals_mod,
        "compute_fc_percentiles",
        _fake_compute_percentiles,
    )
    monkeypatch.setattr(
        extraction_mod, "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: t.zeros(16384, 99),
    )
    monkeypatch.setattr(
        extraction_mod, "_infer_n_sae_features",
        lambda base_sae: 16384,
    )
    monkeypatch.setattr(
        concept_list_mod,
        "_build_concept_examples_qc_client",
        lambda cfg, response_cache=None: _AcceptAllQCClient(),
        raising=False,
    )
    # Use the real candidate loader path (reads candidate_concepts.json).

    cfg = make_valid_axbench_cfg(
        feature_statistics_path=str(stats_path),
        run_tag="stage0_e2e",
        max_concepts=2,
        min_description_length=5,
        min_corpus_count=10,
        validation_shards=[0],
        seed_prefix_max_nanochat_tokens=32,
        seed_continuation_nanochat_tokens=4,
        seed_pool_train_ratio=0.7,
        seed_pool_seed=42,
        seed_extraction_gpu_batch_size=1,
        n_verification_seeds=3,
        min_consistency_passes=2,
        verification_percentile_threshold=0.7,
        verification_length=3,
        min_prefix_chars=1,
        max_prefix_chars=2048,
        n_verified_prefixes_per_concept_target=1,
        extraction_tokenizer_id=expected_tokenizer_id,
    )

    # --- Run all three stages ---
    candidate_concepts_path = prepare_concept_list(cfg)
    assert candidate_concepts_path.exists()
    loaded_candidates = load_and_validate_candidate_concepts(
        candidate_concepts_path, n_sae_features=16384,
    )
    # The tiny shard only has fids {0, 10} + 1..4 + 11..14, so we should
    # always get rows for the two features we fed in.
    fids = {c.feature_index for c in loaded_candidates}
    assert fids == {0, 10}
    assert len(loaded_candidates) == 2
    assert not test_paths.default_paths("stage0_e2e").concepts_json().exists()

    stage0b_model, stage0b_sae = _stage0b_extraction_pair()
    train_path, eval_path = prepare_seed_pools(
        cfg,
        base_model=stage0b_model,
        base_sae=stage0b_sae,
    )
    assert train_path.exists()
    assert eval_path.exists()
    from steering.eval.axbench_base.seed_pools import load_seed_pool
    tokenizer = seed_pools_mod.get_tokenizer()
    train_rows = load_seed_pool(train_path, tokenizer=tokenizer)
    eval_rows = load_seed_pool(eval_path, tokenizer=tokenizer)
    assert len(train_rows) >= 1
    assert len(eval_rows) >= 1

    neg_prefixes_path = prepare_negative_prefixes(
        cfg=cfg,
        concepts_path=candidate_concepts_path,
        base_model=_base_model(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )
    assert neg_prefixes_path.exists()
    final_concepts = load_and_validate_concepts(
        test_paths.default_paths("stage0_e2e").concepts_json(),
        n_sae_features=16384,
    )
    assert {c.feature_index for c in final_concepts} == {0, 10}
    neg_table = pq.read_table(neg_prefixes_path)
    metadata = neg_table.schema.metadata or {}
    assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] == (
        expected_tokenizer_id.encode("utf-8")
    )
    assert tokenizer_ids
    assert qc_tokenizer_ids
    # At least one verified prefix per concept (target is 1 in cfg).
    assert neg_table.num_rows >= len(final_concepts)
    seen_cids = set(neg_table.column("concept_id").to_pylist())
    for c in final_concepts:
        assert c.concept_id in seen_cids, (
            f"Stage 0c missed concept {c.concept_id} "
            f"(feature {c.feature_index}) in verified prefixes"
        )
