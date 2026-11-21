"""Unit tests for prepare_negative_prefixes (Task 17)."""

from __future__ import annotations

import inspect
import json
import random
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import zstandard
from loguru import logger as loguru_logger

from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.eval.axbench_base import (
    _stage0c_internals as _si,
)
from steering.eval.axbench_base import (
    concept_list,
    extraction,
    negative_prefixes,
    paths,
    stage0c_runner,
)
from steering.eval.axbench_base import (
    validate as validate_mod,
)
from steering.eval.axbench_base._stage0c_internals import (
    _STAGE0C_CHECKPOINT_SCHEMA_VERSION,
    _CandidatePrefixIndex,
    _find_candidate_prefixes,
    _PrefixCandidate,
    _VerifiedNegativePrefix,
    _verify_candidate,
)
from steering.eval.axbench_base.concept_examples_qc import (
    ActivatingExample,
    ConceptExamplesQC,
    QCFeatureScore,
    QCReport,
    make_llm_concept_examples_judge,
)
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import LLMJudgeClient
from steering.eval.axbench_base.negative_prefixes import (
    prepare_negative_prefixes,
)
from steering.eval.axbench_base.profiler import ProcessMemorySnapshot
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    QcOutcome,
    RejectedConceptReason,
    TargetModelMetadata,
)
from steering.eval.cluster_checkpoint_pin import _statistics_identity
from steering.eval.cluster_contract_identity import _sha256_json
from steering.eval.cluster_feature_contract import ClusterFeatureTarget
from steering.feature_io import save_features
from steering.feature_paths import (
    DatasetPhase,
    FeatureDirectoryComponents,
    PoolingMode,
    build_feature_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.stats_io import load_statistics, save_percentile_checksum
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.build_tiny_shard import build_tiny_shard
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
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


def _base_model(n_ctx: int = 8192) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(n_ctx=n_ctx))


def _concept(
    concept_id: int = 0,
    feature_index: int = 0,
    sae_source: str = "8-gemmascope-res-16k",
) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source=sae_source,
        description=f"concept {concept_id}",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _candidate(feature_index: int = 0, description: str | None = None) -> CandidateConcept:
    return CandidateConcept(
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=description or f"concept {feature_index}",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _candidate_from_concept(concept: Concept) -> CandidateConcept:
    return CandidateConcept(
        feature_index=concept.feature_index,
        layer=concept.layer,
        sae_source=concept.sae_source,
        description=concept.description,
        corpus_count=concept.corpus_count,
        description_source=concept.description_source,
    )


class _FakeExtractionTokenizer:
    """Deterministic HuggingFace-like tokenizer for Stage 0c tests."""

    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __init__(self) -> None:
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append({
            "text": text,
            "add_special_tokens": add_special_tokens,
        })
        body = [((ord(ch) % 89) + 11) for ch in text]
        if add_special_tokens:
            return [self.bos_token_id, *body]
        return body

    def decode(self, token_ids: list[int]) -> str:
        self.decode_calls.append(list(token_ids))
        return " ".join(f"gemma{tok}" for tok in token_ids)


class _FakeGemmaGenerator:
    """Deterministic spy for the Gemma generation helper."""

    def __init__(self, token_list: list[int] | None = None) -> None:
        self.token_list: list[int] = (
            token_list if token_list is not None else [10, 20, 30]
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, base_model: Any, token_ids: list[int], **kwargs: Any) -> Any:
        self.calls.append({
            "base_model": base_model,
            "token_ids": list(token_ids),
            **kwargs,
        })
        yield from self.token_list


class _FakeBatchedGenerator:
    """Deterministic spy for the batched_generate_seeds kernel.

    Records each batched call (one call per ``_verify_candidate`` invocation)
    and returns ``token_list`` repeated once per seed so the decoding loop
    gets a valid continuation for every seed.
    """

    def __init__(self, token_list: list[int] | None = None) -> None:
        self.token_list: list[int] = (
            token_list if token_list is not None else [10, 20, 30]
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        base_model: Any,
        token_ids: list[int],
        *,
        seeds: list[int],
        **kwargs: Any,
    ) -> list[list[int]]:
        assert isinstance(seeds, list), (
            f"seeds must be list, got {type(seeds).__name__}"
        )
        self.calls.append({
            "base_model": base_model,
            "token_ids": list(token_ids),
            "seeds": list(seeds),
            **kwargs,
        })
        return [list(self.token_list) for _ in seeds]


class _FakeCrossPrefixBatchedGenerator:
    """Deterministic spy for the cross-prefix batched generation kernel."""

    def __init__(self, token_list: list[int] | None = None) -> None:
        self.token_list: list[int] = (
            token_list if token_list is not None else [10, 20, 30]
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        base_model: Any,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **kwargs: Any,
    ) -> list[list[list[int]]]:
        assert len(token_id_rows) == len(seed_rows)
        self.calls.append({
            "base_model": base_model,
            "token_id_rows": [list(row) for row in token_id_rows],
            "seed_rows": [list(row) for row in seed_rows],
            **kwargs,
        })
        return [
            [list(self.token_list) for _ in seeds]
            for seeds in seed_rows
        ]




def _verified_rows(n: int, concept_id: int = 0) -> list[_VerifiedNegativePrefix]:
    return [
        _VerifiedNegativePrefix(
            concept_id=concept_id,
            prefix_text=f"prefix {concept_id}-{i}",
            prefix_source_shard=0,
            prefix_source_doc=i,
            prefix_activating_seg_idx=i,
            verification_continuations=[
                f"continuation {i}-{seed_idx}" for seed_idx in range(3)
            ],
            verification_seeds=[1000 + i * 10 + seed_idx for seed_idx in range(3)],
            verification_percentiles=[80.0, 81.0, 82.0],
        )
        for i in range(n)
    ]


class _PatternQCClient:
    """ConceptExamplesQC-compatible fake with per-feature outcomes."""

    def __init__(self, outcomes: dict[int, str | int] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[tuple[int, str]] = []

    def filter(
        self,
        candidates: list[tuple[int, str]],
        target_surviving: int | None = None,
    ) -> tuple[list[tuple[int, str]], QCReport]:
        survivors: list[tuple[int, str]] = []
        per_feature: list[QCFeatureScore] = []
        n_fetch_errors = 0
        n_judge_errors = 0
        n_empty_examples = 0
        score_histogram: dict[int, int] = {}
        for fid, desc in candidates:
            # Honour ``target_surviving`` early-termination, matching the
            # real ``ConceptExamplesQC.filter`` contract. This is the
            # mechanism that lets ``QcBuffer`` avoid wasted live-API
            # candidate evaluations when batching.
            if (
                target_surviving is not None
                and len(survivors) >= target_surviving
            ):
                break
            self.calls.append((fid, desc))
            outcome = self.outcomes.get(fid, "accepted")
            if outcome == "fetch_error":
                n_fetch_errors += 1
                per_feature.append(
                    QCFeatureScore(
                        fid, desc, QcOutcome.FETCH_ERROR, None, 0, "fetch"
                    )
                )
                continue
            if outcome == "judge_error":
                n_judge_errors += 1
                per_feature.append(
                    QCFeatureScore(
                        fid, desc, QcOutcome.JUDGE_ERROR, None, 1, "judge"
                    )
                )
                continue
            if outcome == "empty_examples":
                n_empty_examples += 1
                per_feature.append(
                    QCFeatureScore(
                        fid, desc, QcOutcome.EMPTY_EXAMPLES, None, 0, None
                    )
                )
                continue
            score = 2 if outcome == "accepted" else int(outcome)
            score_histogram[score] = score_histogram.get(score, 0) + 1
            if score >= 1:
                survivors.append((fid, desc))
                per_feature.append(
                    QCFeatureScore(fid, desc, QcOutcome.ACCEPTED, score, 1, None)
                )
            else:
                per_feature.append(
                    QCFeatureScore(fid, desc, QcOutcome.REJECTED, score, 1, None)
                )
        n_accepted = len(survivors)
        n_processed = len(per_feature)
        n_rejected = (
            n_processed
            - n_accepted
            - n_fetch_errors
            - n_judge_errors
            - n_empty_examples
        )
        return survivors, QCReport(
            n_total=len(candidates),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_fetch_errors=n_fetch_errors,
            n_judge_errors=n_judge_errors,
            n_empty_examples=n_empty_examples,
            score_histogram=score_histogram,
            per_feature=per_feature,
            judge_min_score=1,
            n_processed=n_processed,
        )


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """Set up a minimal environment: tiny shard on disk, fake LLM/QC."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "nanochat_cache"))
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, feature_shard = build_tiny_shard(data_dir, shard_idx=0)
    target = feature_dir / feature_shard.name
    feature_shard.rename(target)

    state: dict[str, Any] = {
        "parquets": [parquet_path],
        "feature_shards": [target],
        "base_model": _base_model(),
        "generator": _FakeGemmaGenerator(),
        "batched_generator": _FakeBatchedGenerator(),
        "cross_prefix_generator": _FakeCrossPrefixBatchedGenerator(),
        "tokenizer": _FakeExtractionTokenizer(),
        "percentile_pair": (10.0, 80.0),  # default: all verifications pass
        "percentile_table": t.zeros(16384, 99),
        "concepts": [_concept(concept_id=0, feature_index=0)],  # feature 0 is in docs 0 and 5
        "qc_client": _PatternQCClient(),
    }

    def _fake_list_parquets(cfg):
        del cfg
        return state["parquets"]

    def _fake_glob_features(cfg):
        del cfg
        return state["feature_shards"]

    def _fake_compute_fc_percentiles(
        prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id,
    ):
        del prefix, cont, f_c, gemma, sae, table, sae_layer, tokenizer_id
        return state["percentile_pair"]

    def _fake_build_table(stats_path, n_sae_features):
        del stats_path, n_sae_features
        return state["percentile_table"]

    real_load_candidates = negative_prefixes.load_and_validate_candidate_concepts

    def _fake_load_candidates(path, n_sae_features):
        candidate_path = Path(path)
        if candidate_path.is_file():
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            if payload:
                return real_load_candidates(
                    candidate_path,
                    n_sae_features=n_sae_features,
                )
        return [_candidate_from_concept(c) for c in state["concepts"]]

    def _fake_infer_n_sae_features(base_sae):
        del base_sae
        return 16384  # stub value; production helper raises on broken SAEs

    monkeypatch.setattr(
        extraction, "_infer_n_sae_features", _fake_infer_n_sae_features,
    )
    monkeypatch.setattr(
        _si, "_list_parquets_for_feature_dir", _fake_list_parquets,
    )
    monkeypatch.setattr(
        stage0c_runner, "glob_features", _fake_glob_features,
    )
    monkeypatch.setattr(
        _si,
        "batched_generate_seeds",
        state["batched_generator"],
        raising=False,
    )
    monkeypatch.setattr(
        _si,
        "cross_prefix_batched_generate_seeds",
        state["cross_prefix_generator"],
        raising=False,
    )
    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: state["tokenizer"],
        raising=False,
    )
    monkeypatch.setattr(
        _si, "compute_fc_percentiles", _fake_compute_fc_percentiles,
    )
    monkeypatch.setattr(
        extraction, "build_percentile_table_from_stats", _fake_build_table,
    )
    monkeypatch.setattr(
        negative_prefixes,
        "load_and_validate_candidate_concepts",
        _fake_load_candidates,
        raising=False,
    )
    monkeypatch.setattr(
        stage0c_runner,
        "load_and_validate_candidate_concepts",
        _fake_load_candidates,
    )
    monkeypatch.setattr(
        concept_list,
        "_build_concept_examples_qc_client",
        lambda cfg, response_cache=None: state["qc_client"],
        raising=False,
    )

    return state


def _cfg(tmp_path, **overrides):
    base = make_valid_config(
        run_tag="test_neg_prefixes",
        output_root=None,
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        validation_shards=[0],
        n_verification_seeds=5,
        min_consistency_passes=3,
        verification_percentile_threshold=0.7,
        verification_length=5,
        min_prefix_chars=1,
        max_prefix_chars=2048,
        n_verified_prefixes_per_concept_target=2,
        max_concepts=1,
        verification_cross_prefix_batch_size=1,
        verification_api_model=None,
    )
    if overrides:
        return base.replace(**overrides)
    return base


# ---------------------------------------------------------------------------
# Gemma generation helper
# ---------------------------------------------------------------------------




















































# ---------------------------------------------------------------------------
# _find_candidate_prefixes
# ---------------------------------------------------------------------------


def test_find_candidates_matches_feature_id(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    # fixture fids are (i%5, i%5+10) so feature 0 is present in docs 0 and 5.
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    assert all(isinstance(c, _PrefixCandidate) for c in candidates)
    doc_indices = [c.doc_idx for c in candidates]
    assert 0 in doc_indices
    assert 5 in doc_indices


def test_find_candidates_respects_scan_limit(stub_env, tmp_path):
    uncapped_cfg = _cfg(tmp_path)
    uncapped = _find_candidate_prefixes(
        uncapped_cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    assert len(uncapped) >= 2

    capped_cfg = _cfg(tmp_path, verification_candidate_scan_limit=1)
    capped = _find_candidate_prefixes(
        capped_cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )

    assert capped == uncapped[:1]


def test_find_candidates_respects_min_prefix_chars(stub_env, tmp_path):
    cfg = _cfg(tmp_path, min_prefix_chars=10_000)  # too long for any doc
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    assert candidates == []


def test_find_candidates_clips_to_max_prefix_chars(stub_env, tmp_path):
    cfg = _cfg(tmp_path, max_prefix_chars=20)
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    for c in candidates:
        assert len(c.prefix_text) <= 20


_DEEP_SEGMENT_MARKER = "TAXADVICEMARKER"
_DEEP_SEGMENT_FEATURE = 4782


def _write_deep_segment_shard(out_dir: Path) -> tuple[Path, Path]:
    """Write a (parquet, feature shard) pair with ONE doc whose activating
    segment ends deeper than the 2048-char prefix budget, with the concept
    marker living inside that deep segment near the document end.

    Returns ``(parquet_path, feature_shard_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    f_c = _DEEP_SEGMENT_FEATURE
    head = "x " * 1300  # 2600 chars of filler -> pushes the segment past 2048
    concept = f"... {_DEEP_SEGMENT_MARKER} on filing deductions and tax credits ..."
    text = head + concept
    seg_end = len(text)        # > 2048
    seg_start = len(head)      # segment covers the concept marker region
    assert seg_end > 2048

    parquet_path = out_dir / "shard_00000.parquet"
    pq.write_table(pa.Table.from_pydict({"text": [text]}), parquet_path)
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([f_c], dtype=t.int32),
        "feature_mags": t.tensor([2.0], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([len(text) // 2], dtype=t.int32),
        "segment_char_starts": t.tensor([seg_start], dtype=t.int32),
        "segment_char_ends": t.tensor([seg_end], dtype=t.int32),
    }
    feature_shard_path = save_features(data, out_dir / "shard_00000_features")
    return parquet_path, feature_shard_path


def test_find_candidates_window_ends_at_activating_segment(tmp_path, monkeypatch):
    """The prefix must CONTAIN the activating segment's text, even when the
    segment ends deeper than ``max_prefix_chars``.

    Regression: the prefix was cut as ``full_text[:max_prefix_chars]`` — the
    document HEAD. When the activating segment ended past ``max_prefix_chars``
    (the majority case for high-frequency features in long docs), the concept
    content was amputated entirely, so the prefix neither contained the
    concept nor led the base model to re-express it during verification. The
    window must instead END at the activating segment, bounded to
    ``max_prefix_chars`` chars.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "nanochat_cache"))
    parquet_path, feature_shard_path = _write_deep_segment_shard(tmp_path / "data")

    cfg = _cfg(tmp_path, max_prefix_chars=2048)
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=feature_shard_path,
        parquet_path=parquet_path,
        f_c=_DEEP_SEGMENT_FEATURE,
        base_context_window=4096,  # large enough that the prefix is not overflow-dropped
    )
    assert len(candidates) == 1
    # The concept content must survive into the prefix...
    assert _DEEP_SEGMENT_MARKER in candidates[0].prefix_text
    # ...and the prefix must still respect the char budget.
    assert len(candidates[0].prefix_text) <= cfg.max_prefix_chars


def test_candidate_index_window_ends_at_activating_segment(tmp_path, monkeypatch):
    """The PRODUCTION candidate path also windows the prefix to end at the
    activating segment.

    Stage 0c verification runs through ``_CandidatePrefixIndex``, not the
    standalone ``_find_candidate_prefixes``. The two share the same windowing
    helper, so this pins the path the eval actually executes and guards
    against the twin implementations drifting apart again.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "nanochat_cache"))
    parquet_path, feature_shard_path = _write_deep_segment_shard(tmp_path / "data")

    cfg = _cfg(tmp_path, max_prefix_chars=2048)  # validation_shards=[0]
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[parquet_path],
        feature_shards=[feature_shard_path],
        candidate_feature_ids={_DEEP_SEGMENT_FEATURE},
        base_context_window=4096,
    )
    candidates = index.candidates_for_feature(_DEEP_SEGMENT_FEATURE)
    assert len(candidates) == 1
    assert _DEEP_SEGMENT_MARKER in candidates[0].prefix_text
    assert len(candidates[0].prefix_text) <= cfg.max_prefix_chars


def test_find_candidates_skips_when_feature_not_in_segment(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    # Feature 999 is not in any segment
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=999,
        base_context_window=1024,
    )
    assert candidates == []


def test_find_candidates_skips_out_of_range_doc(stub_env, tmp_path, monkeypatch):
    """Defensive: if feature shard references doc_id >= len(parquet_texts),
    the candidate is skipped.
    """
    cfg = _cfg(tmp_path)

    # Monkey-patch load_features to return a synthetic featureset whose
    # doc_ids contain an index past the parquet length (10 docs).
    def _fake_load_features(path):
        del path
        return {
            "doc_ids": t.tensor([0, 999], dtype=t.int32),
            "offsets": t.tensor([0, 1, 2], dtype=t.int64),
            "feature_ids": t.tensor([0, 0], dtype=t.int32),
            "feature_mags": t.tensor([1.0, 1.0], dtype=t.float32),
            "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([30, 30], dtype=t.int32),
        }

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    # Only the valid doc_id=0 candidate survives (999 is dropped).
    assert len(candidates) == 1
    assert candidates[0].doc_idx == 0


def test_find_candidates_skips_empty_tokenization(stub_env, tmp_path, monkeypatch):
    """A tokenizer producing only BOS tokens still yields a candidate.

    Stage 0c now uses the extraction tokenizer, not nanochat. A
    degenerate extraction tokenizer may still return only the BOS
    special token when ``add_special_tokens=True``; that should remain a
    valid prompt rather than being silently dropped.
    """
    cfg = _cfg(tmp_path)

    class _OnlyBosTokenizer:
        bos_token_id = 0

        def encode(self, text, add_special_tokens=False, *args, **kwargs):
            del text, args, kwargs
            if add_special_tokens:
                return [self.bos_token_id]
            return []

    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _OnlyBosTokenizer(),
    )
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    # At least one candidate survives the scan; its tokens are
    # exactly the BOS list.
    assert len(candidates) >= 1
    for c in candidates:
        assert c.prefix_gemma_tokens == [0]


def test_find_candidates_drops_truly_empty_extraction_tokenization(
    stub_env, tmp_path, monkeypatch,
):
    class _EmptyTokenizer:
        def encode(self, text, add_special_tokens=False, *args, **kwargs):
            del text, add_special_tokens, args, kwargs
            return []

    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _EmptyTokenizer(),
    )

    candidates = _find_candidate_prefixes(
        _cfg(tmp_path), shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )

    assert candidates == []


def test_find_candidates_filters_token_overflow_boundary_and_logs(
    stub_env, tmp_path, monkeypatch,
):
    """A prefix at the token budget passes; budget+1 is dropped."""
    cfg = _cfg(tmp_path, verification_length=5)
    max_prompt_tokens = 10
    base_context_window = max_prompt_tokens + cfg.verification_length

    class _LengthByPrefixTokenizer:
        bos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            total_len = 10 if text.startswith("The cat") else 11
            body = list(range(1, total_len))
            if add_special_tokens:
                return [self.bos_token_id] + body
            return body

    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _LengthByPrefixTokenizer(),
    )
    collected, handler_id = _capture_info_logs()
    try:
        candidates = _find_candidate_prefixes(
            cfg, shard_idx=0,
            feature_shard_path=stub_env["feature_shards"][0],
            parquet_path=stub_env["parquets"][0],
            f_c=0,
            base_context_window=base_context_window,
        )
    finally:
        loguru_logger.remove(handler_id)

    assert [len(c.prefix_gemma_tokens) for c in candidates] == [10]
    assert candidates[0].doc_idx == 0
    assert any(
        "dropped 1 prefixes as overflow (> 10 tokens)" in msg
        for msg in collected
    )


def test_find_candidates_drops_char_fit_token_expanding_prefix(
    stub_env, tmp_path, monkeypatch,
):
    """Regression: char-bounded prefixes that tokenize past the prompt budget
    are dropped during scan, before verification can hit the Gemma generator.
    """
    cfg = _cfg(tmp_path, verification_length=64, max_prefix_chars=2048)

    class _ExpandingTokenizer:
        bos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del text
            body = list(range(1, 1327))
            if add_special_tokens:
                return [self.bos_token_id] + body
            return body

    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _ExpandingTokenizer(),
    )
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    assert candidates == []


def test_find_candidates_uses_extraction_tokenizer_special_tokens(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, extraction_tokenizer_id="custom/extraction-tokenizer")
    tokenizer = stub_env["tokenizer"]
    seen_tokenizer_ids: list[str] = []
    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id: seen_tokenizer_ids.append(tokenizer_id) or tokenizer,
    )

    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=8192,
    )

    assert candidates
    assert seen_tokenizer_ids == [cfg.extraction_tokenizer_id]
    assert tokenizer.encode_calls
    assert all(call["add_special_tokens"] is True for call in tokenizer.encode_calls)
    assert all(c.prefix_gemma_tokens[0] == tokenizer.bos_token_id for c in candidates)


def test_find_candidates_uses_gemma_n_ctx_for_overflow_filter(
    stub_env, tmp_path, monkeypatch,
):
    """A 1500-Gemma-token prefix passes under Gemma n_ctx=8192.

    The same prefix would have been dropped under a d13-style
    1024-token target context with ``verification_length=64``.
    """
    cfg = _cfg(tmp_path, verification_length=64)

    class _LongGemmaTokenizer:
        bos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del text
            body = list(range(1, 1500))
            if add_special_tokens:
                return [self.bos_token_id] + body
            return body

    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _LongGemmaTokenizer(),
    )
    candidates = _find_candidate_prefixes(
        cfg, shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=8192,
    )

    assert candidates
    assert all(len(c.prefix_gemma_tokens) == 1500 for c in candidates)
    assert all(
        len(c.prefix_gemma_tokens) > 1024 - cfg.verification_length
        for c in candidates
    )


# ---------------------------------------------------------------------------
# _CandidatePrefixIndex
# ---------------------------------------------------------------------------


def test_candidate_prefix_index_canonicalizes_validation_shard_order(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path, validation_shards=[2, 0, 1])
    visited: list[int] = []
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[],
        feature_shards=[],
        candidate_feature_ids={7},
        base_context_window=1024,
    )

    monkeypatch.setattr(
        index,
        "_load_shard",
        lambda shard_idx: visited.append(shard_idx) or object(),
    )
    monkeypatch.setattr(
        index,
        "_materialize_shard_candidates",
        lambda shard, feature_id: [],
    )

    assert index.candidates_for_feature(7) == []
    assert visited == [0, 1, 2]


def test_legacy_candidate_scan_canonicalizes_validation_shard_order(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path, validation_shards=[2, 0, 1])
    visited: list[int] = []
    monkeypatch.setattr(
        _si,
        "_resolve_parquet",
        lambda paths, idx: Path(f"p{idx}"),
    )
    monkeypatch.setattr(
        _si,
        "_resolve_feature_shard",
        lambda paths, idx, feature_dir: Path(f"f{idx}"),
    )

    def _record_scan(
        cfg_arg,
        shard_idx,
        feature_shard_path,
        parquet_path,
        feature_id,
        base_context_window,
    ):
        del (
            cfg_arg,
            feature_shard_path,
            parquet_path,
            feature_id,
            base_context_window,
        )
        visited.append(shard_idx)
        return []

    monkeypatch.setattr(_si, "_find_candidate_prefixes", _record_scan)

    assert _si._verify_concept(
        cfg,
        _concept(concept_id=0, feature_index=7),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.zeros(1),
        sae_layer=8,
        base_context_window=1024,
        candidate_index=None,
    ) == []
    assert visited == [0, 1, 2]


def test_clustered_candidate_scan_visits_every_member_uncapped(
    tmp_path,
    monkeypatch,
):
    """_verify_concept's clustered discovery branch (``cluster_target`` set,
    ``candidate_index`` unset -- PR #739 C3) scans EVERY allowed member ID
    per shard, not just the representative, and does so with the per-shard
    scan limit DISABLED: :func:`_canonical_cluster_candidate_union` applies
    that budget once, centrally, after the union -- applying it per member
    independently would give larger clusters more verification chances.
    """
    cfg = _cfg(
        tmp_path,
        validation_shards=[0, 1],
        verification_candidate_scan_limit=5,
    )
    monkeypatch.setattr(
        _si,
        "_resolve_parquet",
        lambda paths, idx: Path(f"p{idx}"),
    )
    monkeypatch.setattr(
        _si,
        "_resolve_feature_shard",
        lambda paths, idx, feature_dir: Path(f"f{idx}"),
    )

    scanned: list[tuple[int, int, int | None]] = []

    def _record_scan(
        cfg_arg,
        shard_idx,
        feature_shard_path,
        parquet_path,
        feature_id,
        base_context_window,
    ):
        del feature_shard_path, parquet_path, base_context_window
        scanned.append(
            (shard_idx, feature_id, cfg_arg.verification_candidate_scan_limit)
        )
        return []

    monkeypatch.setattr(_si, "_find_candidate_prefixes", _record_scan)

    result = _si._verify_concept(
        cfg,
        _concept(concept_id=0, feature_index=3),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.zeros(1),
        sae_layer=8,
        base_context_window=1024,
        candidate_index=None,
        cluster_target=ClusterFeatureTarget(
            representative_id=3,
            allowed_member_ids=(3, 9),
            projecting_source_ids=(3, 9),
            feature_contract_fingerprint="a" * 64,
        ),
    )

    # Both members (3 and 9), in both shards (0 and 1), each scanned with the
    # per-shard limit DISABLED (None) even though cfg configures 5.
    assert sorted(scanned) == [
        (0, 3, None), (0, 9, None), (1, 3, None), (1, 9, None),
    ]
    assert result == []


def test_candidate_prefix_index_matches_legacy_order_and_scan_limit(
    stub_env, tmp_path,
):
    cfg = _cfg(tmp_path, verification_candidate_scan_limit=1)
    legacy = _find_candidate_prefixes(
        cfg,
        shard_idx=0,
        feature_shard_path=stub_env["feature_shards"][0],
        parquet_path=stub_env["parquets"][0],
        f_c=0,
        base_context_window=1024,
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=stub_env["parquets"],
        feature_shards=stub_env["feature_shards"],
        candidate_feature_ids={0},
        base_context_window=1024,
    )

    indexed = index.candidates_for_feature(0)

    assert indexed == legacy
    assert indexed is not index.candidates_for_feature(0)
    assert index.candidates_for_feature(0) == legacy


class _ReferenceStageLifetimeCandidateIndex:
    """Test-only model of the pre-fix retained candidate-list behavior."""

    def __init__(self, inner: _CandidatePrefixIndex) -> None:
        self._inner = inner
        self._candidates_by_feature: dict[int, list[_PrefixCandidate]] = {}

    def candidates_for_feature(
        self,
        feature_id: int,
    ) -> list[_PrefixCandidate]:
        cached = self._candidates_by_feature.get(feature_id)
        if cached is None:
            cached = self._inner.candidates_for_feature(feature_id)
            self._candidates_by_feature[feature_id] = cached
        return list(cached)


def test_candidate_prefix_index_exactly_matches_retaining_reference(
    stub_env,
    tmp_path,
):
    cfg = _cfg(tmp_path, verification_candidate_scan_limit=2)
    kwargs = {
        "cfg": cfg,
        "parquets": stub_env["parquets"],
        "feature_shards": stub_env["feature_shards"],
        "candidate_feature_ids": {0, 1},
        "base_context_window": 1024,
    }
    production = _CandidatePrefixIndex(**kwargs)
    reference = _ReferenceStageLifetimeCandidateIndex(
        _CandidatePrefixIndex(**kwargs)
    )

    for feature_id in (0, 1, 0):
        actual = production.candidates_for_feature(feature_id)
        expected = reference.candidates_for_feature(feature_id)
        assert actual == expected
        assert [
            (
                candidate.shard_idx,
                candidate.doc_idx,
                candidate.seg_idx,
                candidate.prefix_text,
                candidate.prefix_gemma_tokens,
                candidate.source_bridge_text,
            )
            for candidate in actual
        ] == [
            (
                candidate.shard_idx,
                candidate.doc_idx,
                candidate.seg_idx,
                candidate.prefix_text,
                candidate.prefix_gemma_tokens,
                candidate.source_bridge_text,
            )
            for candidate in expected
        ]


def test_candidate_prefix_index_lazy_loads_and_reuses_shard(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)
    feature_loads: list[Path] = []
    text_loads: list[Path] = []
    orig_load_features = _si.load_features
    orig_load_text = _si.read_parquet_texts

    def _spy_load_features(path: Path) -> dict[str, Any]:
        feature_loads.append(path)
        return orig_load_features(path)

    def _spy_load_text(path: Path) -> list[str]:
        text_loads.append(path)
        return orig_load_text(path)

    monkeypatch.setattr(_si, "load_features", _spy_load_features)
    monkeypatch.setattr(_si, "read_parquet_texts", _spy_load_text)
    index = _CandidatePrefixIndex(
        cfg,
        parquets=stub_env["parquets"],
        feature_shards=stub_env["feature_shards"],
        candidate_feature_ids={0, 1},
        base_context_window=8192,
    )

    assert len(index._loaded_shards) == 0
    first = index.candidates_for_feature(0)
    second = index.candidates_for_feature(1)
    first_again = index.candidates_for_feature(0)

    assert first
    assert second
    assert first_again == first
    assert len(index._loaded_shards) == 1
    assert feature_loads == [stub_env["feature_shards"][0]]
    assert text_loads == [stub_env["parquets"][0]]


def test_candidate_prefix_index_empty_candidates(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    index = _CandidatePrefixIndex(
        cfg,
        parquets=stub_env["parquets"],
        feature_shards=stub_env["feature_shards"],
        candidate_feature_ids={999},
        base_context_window=8192,
    )

    assert index.candidates_for_feature(999) == []


def test_candidate_prefix_index_missing_feature_shard_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    index = _CandidatePrefixIndex(
        cfg,
        parquets=stub_env["parquets"],
        feature_shards=[],
        candidate_feature_ids={0},
        base_context_window=8192,
    )

    with pytest.raises(AxBenchBaseError, match="No feature shard"):
        index.candidates_for_feature(0)


def test_candidate_prefix_index_corrupt_feature_shard_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    bad_shard = tmp_path / "shard_00000_features.safetensors.zst"
    bad_shard.write_bytes(b"not a zstd-compressed safetensors shard")
    index = _CandidatePrefixIndex(
        cfg,
        parquets=stub_env["parquets"],
        feature_shards=[bad_shard],
        candidate_feature_ids={0},
        base_context_window=8192,
    )

    # Slice 16 audit (X2 §4.3): narrow from broad-Exception. The
    # zstd decoder raises ``zstandard.ZstdError`` on malformed input;
    # the contract here is that bad shards surface a clear loader
    # error rather than silently returning empty results.
    with pytest.raises(zstandard.ZstdError):
        index.candidates_for_feature(0)


def test_candidate_prefix_index_deduplicates_feature_within_segment(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)

    def _fake_load_features(path: Path) -> dict[str, t.Tensor]:
        del path
        return {
            "doc_ids": t.tensor([0, 1], dtype=t.int32),
            "offsets": t.tensor([0, 3, 6], dtype=t.int64),
            "feature_ids": t.tensor([7, 7, 8, 7, 9, 9], dtype=t.int32),
            "feature_mags": t.ones(6, dtype=t.float32),
            "segment_char_ends": t.tensor([30, 30], dtype=t.int32),
        }

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    monkeypatch.setattr(
        _si,
        "read_parquet_texts",
        lambda path: ["a" * 40, "b" * 40],
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[Path("/tmp/shard_00000.parquet")],
        feature_shards=[Path("/tmp/shard_00000_features.safetensors.zst")],
        candidate_feature_ids={7, 9},
        base_context_window=8192,
    )

    feature_7 = index.candidates_for_feature(7)
    feature_9 = index.candidates_for_feature(9)

    assert [(c.doc_idx, c.seg_idx) for c in feature_7] == [(0, 0), (1, 1)]
    assert [(c.doc_idx, c.seg_idx) for c in feature_9] == [(1, 1)]


def test_candidate_prefix_index_maps_offset_boundary_after_empty_segment(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)

    def _fake_load_features(path: Path) -> dict[str, t.Tensor]:
        del path
        return {
            "doc_ids": t.tensor([0, 1, 2], dtype=t.int32),
            "offsets": t.tensor([0, 2, 2, 4], dtype=t.int64),
            "feature_ids": t.tensor([3, 4, 7, 8], dtype=t.int32),
            "feature_mags": t.ones(4, dtype=t.float32),
            "segment_char_ends": t.tensor([30, 30, 30], dtype=t.int32),
        }

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    monkeypatch.setattr(
        _si,
        "read_parquet_texts",
        lambda path: ["doc0" * 10, "doc1" * 10, "doc2" * 10],
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[Path("/tmp/shard_00000.parquet")],
        feature_shards=[Path("/tmp/shard_00000_features.safetensors.zst")],
        candidate_feature_ids={7},
        base_context_window=8192,
    )

    candidates = index.candidates_for_feature(7)

    assert [(c.doc_idx, c.seg_idx) for c in candidates] == [(2, 2)]


def test_candidate_prefix_index_ignores_activation_ties_for_order(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)

    def _fake_load_features(path: Path) -> dict[str, t.Tensor]:
        del path
        return {
            "doc_ids": t.tensor([2, 0, 1], dtype=t.int32),
            "offsets": t.tensor([0, 1, 2, 3], dtype=t.int64),
            "feature_ids": t.tensor([5, 5, 5], dtype=t.int32),
            "feature_mags": t.tensor([1.0, 1.0, 1.0], dtype=t.float32),
            "segment_char_ends": t.tensor([30, 30, 30], dtype=t.int32),
        }

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    monkeypatch.setattr(
        _si,
        "read_parquet_texts",
        lambda path: ["doc0" * 10, "doc1" * 10, "doc2" * 10],
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[Path("/tmp/shard_00000.parquet")],
        feature_shards=[Path("/tmp/shard_00000_features.safetensors.zst")],
        candidate_feature_ids={5},
        base_context_window=8192,
    )

    candidates = index.candidates_for_feature(5)

    assert [(c.seg_idx, c.doc_idx) for c in candidates] == [(0, 2), (1, 0), (2, 1)]


def test_candidate_prefix_index_does_not_cache_feature_segments(
    stub_env, tmp_path,
):
    cfg = _cfg(tmp_path)
    index = _CandidatePrefixIndex(
        cfg,
        parquets=stub_env["parquets"],
        feature_shards=stub_env["feature_shards"],
        candidate_feature_ids={0},
        base_context_window=8192,
    )

    shard = index._load_shard(0)
    first = index._lookup_feature_segments(shard, 0)
    second = index._lookup_feature_segments(shard, 0)

    assert list(first) == list(second)
    assert second is not first
    assert not hasattr(index, "_candidates_by_feature")
    assert not hasattr(index, "_segments_for_feature")
    assert not hasattr(shard, "segments_by_feature")


def test_candidate_prefix_lookup_protocol_stub_is_callable():
    class _DummyLookup(_si._CandidatePrefixLookup):
        pass

    assert (
        _DummyLookup.candidates_for_feature(
            _DummyLookup(), 0,
        ) is None
    )


def test_candidate_prefix_index_drops_missing_doc_short_and_empty_tokenization(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, min_prefix_chars=5)

    class _BranchyTokenizer(_FakeExtractionTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text.startswith("empty"):
                return []
            return super().encode(text, add_special_tokens=add_special_tokens)

    def _fake_load_features(path: Path) -> dict[str, t.Tensor]:
        del path
        return {
            "doc_ids": t.tensor([3, 0, 1, 2], dtype=t.int32),
            "offsets": t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
            "feature_ids": t.tensor([7, 7, 7, 7], dtype=t.int32),
            "feature_mags": t.ones(4, dtype=t.float32),
            "segment_char_ends": t.tensor([20, 2, 20, 20], dtype=t.int32),
        }

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    monkeypatch.setattr(
        _si,
        "read_parquet_texts",
        lambda path: [
            "abcd",
            "empty tokenization example",
            "valid prefix candidate text",
        ],
    )
    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _BranchyTokenizer(),
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[Path("/tmp/shard_00000.parquet")],
        feature_shards=[Path("/tmp/shard_00000_features.safetensors.zst")],
        candidate_feature_ids={7},
        base_context_window=8192,
    )

    candidates = index.candidates_for_feature(7)

    assert [(c.doc_idx, c.seg_idx) for c in candidates] == [(2, 3)]
    assert candidates[0].prefix_text == "valid prefix candida"


@pytest.mark.parametrize(
    ("offsets", "feature_ids", "match"),
    [
        (
            t.tensor([0, 1, 2, 3], dtype=t.int64),
            t.tensor([7, 8, 9], dtype=t.int32),
            "offsets length",
        ),
        (
            t.tensor([1, 2, 3], dtype=t.int64),
            t.tensor([7, 8, 9], dtype=t.int32),
            r"offsets\[0\] == 0",
        ),
        (
            t.tensor([0, 3, 2], dtype=t.int64),
            t.tensor([7, 8, 9], dtype=t.int32),
            "monotonic CSR offsets",
        ),
        (
            t.tensor([0, 1, 2], dtype=t.int64),
            t.tensor([7, 8, 9, 10], dtype=t.int32),
            r"offsets\[-1\] == len\(feature_ids\)",
        ),
    ],
)
def test_candidate_prefix_index_rejects_invalid_csr_offsets(
    stub_env, tmp_path, monkeypatch, offsets, feature_ids, match,
):
    cfg = _cfg(tmp_path)

    def _fake_load_features(path: Path) -> dict[str, t.Tensor]:
        del path
        return {
            "doc_ids": t.tensor([0, 1], dtype=t.int32),
            "offsets": offsets,
            "feature_ids": feature_ids,
            "feature_mags": t.ones(int(feature_ids.numel()), dtype=t.float32),
            "segment_char_ends": t.tensor([30, 30], dtype=t.int32),
        }

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    monkeypatch.setattr(
        _si,
        "read_parquet_texts",
        lambda path: ["a" * 40, "b" * 40],
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[Path("/tmp/shard_00000.parquet")],
        feature_shards=[Path("/tmp/shard_00000_features.safetensors.zst")],
        candidate_feature_ids={7},
        base_context_window=8192,
    )

    with pytest.raises(AxBenchBaseError, match=match):
        index.candidates_for_feature(7)


def test_candidate_prefix_index_matches_legacy_across_multiple_shards(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "nanochat_cache"))
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquets: list[Path] = []
    feature_shards: list[Path] = []
    for shard_idx in (0, 1):
        parquet_path, feature_shard = build_tiny_shard(data_dir, shard_idx=shard_idx)
        target = feature_dir / feature_shard.name
        feature_shard.rename(target)
        parquets.append(parquet_path)
        feature_shards.append(target)

    cfg = _cfg(
        tmp_path,
        validation_shards=[0, 1],
        verification_candidate_scan_limit=1,
        verification_candidate_attempt_limit=2,
    )
    legacy: list[_PrefixCandidate] = []
    for shard_idx, parquet_path, feature_shard_path in zip(
        cfg.validation_shards,
        parquets,
        feature_shards,
        strict=True,
    ):
        legacy.extend(
            _find_candidate_prefixes(
                cfg,
                shard_idx=shard_idx,
                feature_shard_path=feature_shard_path,
                parquet_path=parquet_path,
                f_c=0,
                base_context_window=1024,
            )
        )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=parquets,
        feature_shards=feature_shards,
        candidate_feature_ids={0},
        base_context_window=1024,
    )
    indexed = index.candidates_for_feature(0)

    assert indexed == legacy

    attempted: list[tuple[int, int]] = []

    def _reject_candidate(
        cfg_arg,
        candidate,
        f_c,
        concept_id,
        base_model,
        base_sae,
        percentile_table,
        *,
        sae_layer,
    ):
        del cfg_arg, f_c, concept_id, base_model, base_sae, percentile_table, sae_layer
        attempted.append((candidate.shard_idx, candidate.doc_idx))
        return None

    monkeypatch.setattr(_si, "_verify_candidate", _reject_candidate)
    legacy_rows = _si._verify_concept(
        cfg,
        _concept(concept_id=0, feature_index=0),
        parquets=parquets,
        feature_shards=feature_shards,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        sae_layer=8,
        base_context_window=1024,
        candidate_index=None,
    )
    legacy_attempted = list(attempted)
    attempted.clear()
    indexed_rows = _si._verify_concept(
        cfg,
        _concept(concept_id=0, feature_index=0),
        parquets=parquets,
        feature_shards=feature_shards,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        sae_layer=8,
        base_context_window=1024,
        candidate_index=index,
    )

    assert legacy_rows == indexed_rows == []
    assert attempted == legacy_attempted


def test_verify_concept_with_index_preserves_shuffle_and_attempt_limit(
    tmp_path, monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=1,
        verification_candidate_attempt_limit=3,
    )
    candidates = [
        _PrefixCandidate(
            shard_idx=0,
            doc_idx=idx,
            seg_idx=idx,
            prefix_text=f"prefix {idx}",
            prefix_gemma_tokens=[101, idx + 1],
        )
        for idx in range(6)
    ]

    class _FakeIndex:
        def candidates_for_feature(self, feature_id: int) -> list[_PrefixCandidate]:
            assert feature_id == 0
            return list(candidates)

    attempted: list[int] = []

    def _reject_candidate(
        cfg_arg,
        candidate,
        f_c,
        concept_id,
        base_model,
        base_sae,
        percentile_table,
        *,
        sae_layer,
    ):
        del cfg_arg, f_c, concept_id, base_model, base_sae, percentile_table, sae_layer
        attempted.append(candidate.doc_idx)
        return None

    monkeypatch.setattr(_si, "_verify_candidate", _reject_candidate)

    rows = _si._verify_concept(
        cfg,
        _concept(concept_id=2, feature_index=0),
        parquets=[],
        feature_shards=[],
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        sae_layer=8,
        base_context_window=8192,
        candidate_index=_FakeIndex(),
    )
    expected = list(candidates)
    rng = random.Random(
        cfg.generation.master_seed + _si._VERIFY_SEED_OFFSET + 2,
    )
    rng.shuffle(expected)

    assert rows == []
    assert attempted == [candidate.doc_idx for candidate in expected[:3]]


# ---------------------------------------------------------------------------
# Stage 0c partial-progress checkpointing
# ---------------------------------------------------------------------------


def test_atomic_write_parquet_writes_table_at_target(tmp_path):
    """``atomic_write_parquet`` writes a pyarrow Table to ``final_path``.

    The helper now lives in ``nanochat.atomic_io`` (consolidation of
    Stage 0c-local duplicates with the other atomic-write helpers); the
    canonical signature is ``atomic_write_parquet(table, final_path)``.
    """
    from nanochat.atomic_io import atomic_write_parquet

    target = tmp_path / "target.parquet"
    table = pa.table({"a": [1, 2, 3]})
    atomic_write_parquet(table, target)
    read_back = pq.read_table(target)
    assert read_back.to_pylist() == [{"a": 1}, {"a": 2}, {"a": 3}]
    # No leftover tmp files (tmp pattern is `<name>.tmp.<pid>.<uuid8>`)
    leftover = list(target.parent.glob(f"{target.name}.tmp.*"))
    assert leftover == [], f"Unexpected leftover tmp files: {leftover}"


def test_compute_checkpoint_fingerprint_includes_complete_field_set(tmp_path):
    """The fingerprint dict has every field listed in spec §5.4."""
    cfg = _cfg(tmp_path)
    fp = _si._compute_checkpoint_fingerprint(cfg)
    expected_keys = {
        "stage0c_algorithm_version",
        "feature_dir",
        "feature_statistics_path",
        "validation_shards",
        # Extractor identity additions (schema v5)
        "neuronpedia_model_id",
        "feature_extraction_model_id",
        "neuronpedia_source",
        "extraction_tokenizer_id",
        "n_verified_prefixes_per_concept_target",
        "n_verification_seeds",
        "min_consistency_passes",
        "verification_percentile_threshold",
        "verification_length",
        "min_prefix_chars",
        "max_prefix_chars",
        "verification_zero_hit_bail_after",
        "verification_candidate_attempt_limit",
        "verification_candidate_scan_limit",
        "enable_concept_examples_qc",
        # Complete conditional QC identity (schema v6)
        "concept_examples_qc_config",
        "generation_master_seed",
        "generation_temperature",
        "generation_top_k_sampling",
        # PR-2 additions
        "qc_batch_size",
        "qc_pipeline",
        "verification_prefill_mode",
        # PR-4 addition
        "stage0c_multi_gpu_block_size",
        # Cross-prefix batching addition
        "verification_cross_prefix_batch_size",
    }
    assert set(fp.keys()) == expected_keys
    assert len(expected_keys) == 28  # safety: count check (the field list above)
    assert fp["stage0c_algorithm_version"] == _STAGE0C_CHECKPOINT_SCHEMA_VERSION
    assert fp["concept_examples_qc_config"] == {
        "n_activation_examples": cfg.n_activation_examples,
        "activation_example_judge_min_score": (
            cfg.activation_example_judge_min_score
        ),
        "concept_examples_qc_target_surviving": (
            cfg.concept_examples_qc_target_surviving
        ),
        "judge": asdict(cfg.concept_examples_qc_judge),
        "neuronpedia_max_concurrent": cfg.neuronpedia.max_concurrent,
    }


def test_compute_checkpoint_fingerprint_qc_disabled_normalizes_qc_config(tmp_path):
    """Dormant examples-QC settings do not invalidate disabled-QC state."""
    cfg = _cfg(tmp_path, enable_concept_examples_qc=False)
    fp = _si._compute_checkpoint_fingerprint(cfg)
    assert fp["concept_examples_qc_config"] is None
    assert fp["enable_concept_examples_qc"] is False
    changed_cfg = cfg.replace(
        n_activation_examples=99,
        activation_example_judge_min_score=2,
        concept_examples_qc_target_surviving=7,
        concept_examples_qc_judge=LLMClientConfig(
            provider="other-provider",
            model="other-model",
            requests_per_minute=3,
            max_concurrent=2,
        ),
        neuronpedia=replace(
            cfg.neuronpedia,
            max_concurrent=cfg.neuronpedia.max_concurrent + 1,
        ),
    )
    assert _si._compute_checkpoint_fingerprint(changed_cfg) == fp


def test_concept_round_trips_through_json_dict():
    """A Concept survives asdict + json.dumps + json.loads + reconstruction."""
    original = Concept(
        concept_id=7,
        feature_index=12345,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="round-trip test",
        corpus_count=100,
        description_source="neuronpedia",
    )
    js = json.dumps(asdict(original))
    parsed = json.loads(js)
    rebuilt = Concept(**parsed)
    assert rebuilt == original


def test_rejected_concept_round_trips_with_reason_enum():
    """``_RejectedConcept`` survives the existing _rejected_concept_to_json
    serializer plus a rebuild that re-coerces ``reason`` back to its enum."""
    original = _si._RejectedConcept(
        candidate_index=3,
        feature_index=42,
        description="rejected test",
        reason=RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES,
        n_verified_found=2,
        llm_judge_score=1,
    )
    js = json.dumps(_si._rejected_concept_to_json(original))
    parsed = json.loads(js)
    rebuilt = _si._rejected_concept_from_json(parsed)
    assert rebuilt == original


def test_qc_feature_score_round_trips_through_json_dict():
    """A QCFeatureScore survives asdict + json.dumps + json.loads."""
    original = QCFeatureScore(
        feature_index=42,
        description="qc round-trip",
        outcome=QcOutcome.ACCEPTED,
        score=2,
        n_examples=5,
        error_message=None,
    )
    js = json.dumps(asdict(original))
    parsed = json.loads(js)
    rebuilt = QCFeatureScore(**parsed)
    assert rebuilt == original


































# ---------------------------------------------------------------------------
# Zero-hit bailout
# ---------------------------------------------------------------------------


def _scripted_verify_candidate(outcomes: list[bool]) -> Any:
    """Stub ``_verify_candidate`` that succeeds/fails per ``outcomes``.

    Each entry in ``outcomes`` corresponds to one call: ``True`` means the
    stub returns a fully populated ``_VerifiedNegativePrefix``; ``False``
    means it returns ``None`` (a verification miss). Exhausting the script
    raises ``AssertionError`` so a test that miscounts iterations fails
    loudly rather than silently succeeding via implicit ``None``.

    Returns the stub callable; callers can also count calls externally
    by wrapping the stub with the ``_call_counter_wrapping`` helper
    defined later in this section, which returns ``(wrapper, calls)``
    where ``calls`` is a list appended to on each invocation.
    """
    state = {"i": 0}

    def _stub(
        cfg_arg, candidate, f_c, concept_id,
        base_model, base_sae, percentile_table,
        *, sae_layer,
    ):
        del cfg_arg, f_c, base_model, base_sae, percentile_table, sae_layer
        i = state["i"]
        if i >= len(outcomes):
            raise AssertionError(
                f"_scripted_verify_candidate exhausted: tried to call "
                f"#{i + 1} but only {len(outcomes)} outcomes were scripted"
            )
        state["i"] += 1
        if outcomes[i]:
            return _VerifiedNegativePrefix(
                concept_id=concept_id,
                prefix_text=candidate.prefix_text,
                prefix_source_shard=candidate.shard_idx,
                prefix_source_doc=candidate.doc_idx,
                prefix_activating_seg_idx=candidate.seg_idx,
                verification_continuations=["c"],
                verification_seeds=[1],
                verification_percentiles=[99.0],
            )
        return None

    return _stub


def _make_candidate(idx: int) -> _PrefixCandidate:
    """Build a minimal _PrefixCandidate for bailout tests."""
    return _PrefixCandidate(
        shard_idx=0,
        doc_idx=idx,
        seg_idx=idx,
        prefix_text=f"prefix {idx}",
        prefix_gemma_tokens=[101, idx + 1],
    )


class _FixedIndex:
    """Returns a fixed candidate list regardless of feature_id."""

    def __init__(self, candidates: list[_PrefixCandidate]) -> None:
        self._candidates = list(candidates)

    def candidates_for_feature(
        self, feature_id: int,
    ) -> list[_PrefixCandidate]:
        del feature_id
        return list(self._candidates)


def _call_counter_wrapping(stub: Any) -> tuple[Any, list[int]]:
    """Wrap a verify-candidate stub to count calls. Returns (wrapper, calls)."""
    calls: list[int] = []

    def _wrapper(*args, **kwargs):
        calls.append(1)
        return stub(*args, **kwargs)

    return _wrapper, calls


def _capture_record_counter(monkeypatch) -> list[tuple[str, int, dict]]:
    """Replace ``_si.record_counter`` with a list-spy.

    Returns the list, which receives ``(name, value, attrs)`` tuples for
    every counter recorded during the test.
    """
    captured: list[tuple[str, int, dict]] = []

    def _spy(name: str, value: int, **attrs: Any) -> None:
        captured.append((name, value, dict(attrs)))

    monkeypatch.setattr(_si, "record_counter", _spy)
    return captured


def test_zero_hit_bailout_triggers_at_threshold(tmp_path, monkeypatch):
    """Bail at exactly n_tried == bail_after when verified is empty."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=5,
    )
    candidates = [_make_candidate(i) for i in range(10)]
    stub = _scripted_verify_candidate([False] * 10)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=3, feature_index=42),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    assert len(calls) == 5, (
        f"Expected exactly 5 calls (bail_after=5); got {len(calls)}"
    )
    bailout_lines = [
        m for m in collected
        if "zero-hit bailout triggered" in m
    ]
    assert len(bailout_lines) == 1, (
        f"Expected exactly one bailout log line, got {bailout_lines}"
    )
    line = bailout_lines[0]
    assert "Concept 3" in line
    assert "feature 42" in line
    assert "after 5 verification attempts" in line
    assert "threshold=5" in line
    assert "not_enough_verified_prefixes" in line
    bailout_counters = [
        c for c in captured if c[0] == "stage0c.verification_zero_hit_bailouts"
    ]
    assert bailout_counters == [
        (
            "stage0c.verification_zero_hit_bailouts",
            1,
            {"concept_id": 3, "feature_id": 42},
        )
    ]


def test_zero_hit_bailout_one_below_threshold_does_not_trigger(
    tmp_path, monkeypatch,
):
    """n_tried = bail_after - 1: loop runs to natural for-end, no bailout."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=5,
    )
    candidates = [_make_candidate(i) for i in range(4)]  # < 5
    stub = _scripted_verify_candidate([False] * 4)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    assert len(calls) == 4
    assert not any("zero-hit bailout triggered" in m for m in collected)
    assert not any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    )


def test_zero_hit_bailout_disabled_when_first_hit_at_threshold(
    tmp_path, monkeypatch,
):
    """If the first hit lands at iteration ``bail_after``, the bailout check
    sees ``not verified == False`` and the loop continues. With target=2
    unmet, the loop exhausts the pool naturally."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=5,
    )
    candidates = [_make_candidate(i) for i in range(10)]
    # Outcomes: 4 misses, hit, 5 misses.
    stub = _scripted_verify_candidate(
        [False, False, False, False, True, False, False, False, False, False]
    )
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert len(rows) == 1
    assert len(calls) == 10  # full pool consumed; target=2 not met
    assert not any("zero-hit bailout triggered" in m for m in collected)
    assert not any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    )


def test_zero_hit_bailout_disabled_after_early_hit_then_drought(
    tmp_path, monkeypatch,
):
    """Once any prefix is verified, ``not verified`` is False permanently
    for the rest of that concept's loop, regardless of subsequent failures."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=3,
    )
    candidates = [_make_candidate(i) for i in range(8)]
    stub = _scripted_verify_candidate([True] + [False] * 7)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert len(rows) == 1
    assert len(calls) == 8  # entire pool consumed
    assert not any("zero-hit bailout triggered" in m for m in collected)
    assert not any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    )


def test_zero_hit_bailout_none_disables_legacy_behavior(
    tmp_path, monkeypatch,
):
    """``bail_after = None`` reproduces pre-change behavior bit-for-bit."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=None,
    )
    candidates = [_make_candidate(i) for i in range(50)]
    stub = _scripted_verify_candidate([False] * 50)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    assert len(calls) == 50  # legacy "scan everything" behavior
    assert not any("zero-hit bailout triggered" in m for m in collected)
    assert not any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    )


def test_zero_hit_bailout_at_value_one_triggers_immediately(
    tmp_path, monkeypatch,
):
    """Smallest valid setting: bail after the first zero-hit attempt."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=1,
    )
    candidates = [_make_candidate(i) for i in range(5)]
    stub = _scripted_verify_candidate([False] * 5)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    assert len(calls) == 1
    assert any("zero-hit bailout triggered" in m for m in collected)
    assert any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    )


def test_zero_hit_warning_uses_n_tried_not_pool_size(
    tmp_path, monkeypatch,
):
    """The pre-existing zero-verified warning reports n_tried (the actual
    number of candidates attempted), not len(all_candidates) (which was
    misleading whenever verification_candidate_attempt_limit truncated
    the pool, and would be even more misleading after the bailout patch).
    """
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=None,  # disable bailout
        verification_candidate_attempt_limit=3,
    )
    candidates = [_make_candidate(i) for i in range(10)]
    stub = _scripted_verify_candidate([False] * 10)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)

    warning_messages: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: warning_messages.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    assert len(calls) == 3  # attempt_limit truncated to 3
    zero_warnings = [
        m for m in warning_messages if "ZERO verified prefixes" in m
    ]
    assert len(zero_warnings) == 1, (
        f"Expected exactly one zero-verified warning, got {zero_warnings}"
    )
    line = zero_warnings[0]
    assert "after 3 verification attempts" in line, (
        f"Warning should report n_tried (3), got: {line!r}"
    )
    # The attempt count should appear, but the pool size (10) must not be
    # mistaken for the count of attempts. (We cannot blanket-forbid "10"
    # because the description / concept_id might happen to contain it; the
    # explicit check above is the contract.)


def test_zero_hit_bailout_smaller_than_attempt_limit_fires_first(
    tmp_path, monkeypatch,
):
    """When bail_after < attempt_limit, bailout terminates the loop first."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=10,
        verification_candidate_attempt_limit=50,
    )
    candidates = [_make_candidate(i) for i in range(100)]
    stub = _scripted_verify_candidate([False] * 100)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)
    collected, handler_id = _capture_info_logs()

    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    assert len(calls) == 10  # bail_after wins over attempt_limit
    bailout_lines = [m for m in collected if "zero-hit bailout triggered" in m]
    assert len(bailout_lines) == 1
    bailout_counters = [
        c for c in captured if c[0] == "stage0c.verification_zero_hit_bailouts"
    ]
    assert len(bailout_counters) == 1


def test_zero_hit_bailout_larger_than_attempt_limit_never_fires(
    tmp_path, monkeypatch,
):
    """When attempt_limit < bail_after, the truncated pool exhausts before
    n_tried can reach bail_after. Bailout never fires; the existing
    zero-verified warning fires once and reports n_tried (= attempt_limit)."""
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=100,
        verification_candidate_attempt_limit=20,
    )
    candidates = [_make_candidate(i) for i in range(100)]
    stub = _scripted_verify_candidate([False] * 100)
    wrapped, calls = _call_counter_wrapping(stub)
    monkeypatch.setattr(_si, "_verify_candidate", wrapped)
    captured = _capture_record_counter(monkeypatch)

    info_collected, info_handler = _capture_info_logs()
    warning_collected: list[str] = []
    warning_handler = loguru_logger.add(
        lambda msg: warning_collected.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex(candidates),
        )
    finally:
        loguru_logger.remove(info_handler)
        loguru_logger.remove(warning_handler)

    assert rows == []
    assert len(calls) == 20  # attempt_limit wins; bailout never fires
    assert not any(
        "zero-hit bailout triggered" in m for m in info_collected
    )
    assert not any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    )
    zero_warnings = [
        m for m in warning_collected if "ZERO verified prefixes" in m
    ]
    assert len(zero_warnings) == 1
    assert "after 20 verification attempts" in zero_warnings[0]


def test_zero_hit_bailout_routes_through_not_enough_verified_prefixes_path(
    stub_env, tmp_path, monkeypatch,
):
    """End-to-end: a candidate that bails out is recorded in
    rejected_concepts.json with reason=not_enough_verified_prefixes,
    n_verified_found=0, and llm_judge_score preserved through the bail
    path. The bailout output is indistinguishable from a natural
    zero-hit rejection (I4: downstream parity).

    Pool sizing: the default stub_env shard yields only 2 candidates
    for feature 0 (docs 0 and 5). To prove the bailout is the active
    path (not just pool exhaustion), expand the pool with one extra
    shard so 4 candidates are available, and set bail_after=3 so the
    bailout fires at iteration 3 with one candidate remaining in the
    pool. The counter assertion below is the discriminating contract:
    if the bailout were disabled, the counter would not fire.
    """
    data_dir = tmp_path / "data"
    feature_dir = stub_env["feature_shards"][0].parent
    pq_path, feat_path = build_tiny_shard(data_dir, shard_idx=1)
    target = feature_dir / feat_path.name
    feat_path.rename(target)
    stub_env["parquets"].append(pq_path)
    stub_env["feature_shards"].append(target)

    cfg = _cfg(
        tmp_path,
        verification_zero_hit_bail_after=3,
        validation_shards=[0, 1],
    )
    # All percentile checks fail -> all _verify_candidate calls return None
    # -> bailout fires at n_tried=3, before pool of 4 is exhausted.
    stub_env["percentile_pair"] = (10.0, 30.0)
    captured = _capture_record_counter(monkeypatch)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    with pytest.raises(
        AxBenchBaseError,
        match="lowering verification_percentile_threshold",
    ):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert rejected == [
        {
            "candidate_index": 0,
            "feature_index": 0,
            "description": "concept 0",
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": 2,
        }
    ]
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    # Discriminating assertion: prove the bailout was the active path
    # (without this, the test would pass equally with bail_after=None
    # because the rejection schema is identical between bailout and
    # pool-exhaustion paths).
    bailout_counters = [
        c for c in captured if c[0] == "stage0c.verification_zero_hit_bailouts"
    ]
    assert len(bailout_counters) == 1, (
        f"Expected exactly one bailout counter, got {bailout_counters}"
    )


def test_zero_hit_bailout_does_not_fire_when_concept_has_any_verified_hit(
    stub_env, tmp_path, monkeypatch,
):
    """End-to-end: a candidate that gets even one verified prefix never
    bails; subsequent failures route through the partial-verified path
    (n_verified_found = 1) instead of the zero-hit bailout path
    (n_verified_found = 0). Confirms the "any positive signal preserves
    the candidate" semantics survive end-to-end."""
    # Need a target above what we can deliver but enough candidates so
    # the loop does many attempts after the first hit.
    # Extend the shard pool with two extra shards so the loop has > 2
    # candidates for feature 0 (each shard contributes 2 for docs 0 and 5).
    # Also configure validation_shards=[0, 1, 2] so all three are scanned.
    data_dir = tmp_path / "data"
    feature_dir = stub_env["feature_shards"][0].parent
    for extra_idx in (1, 2):
        pq_path, feat_path = build_tiny_shard(data_dir, shard_idx=extra_idx)
        target = feature_dir / feat_path.name
        feat_path.rename(target)
        stub_env["parquets"].append(pq_path)
        stub_env["feature_shards"].append(target)
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=100,
        verification_zero_hit_bail_after=2,
        validation_shards=[0, 1, 2],
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    call_count = {"n": 0}

    def _first_hit_then_fail(
        cfg_arg, candidate, f_c, concept_id,
        base_model, base_sae, percentile_table,
        *, sae_layer,
    ):
        del cfg_arg, f_c, base_model, base_sae, percentile_table, sae_layer
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _VerifiedNegativePrefix(
                concept_id=concept_id,
                prefix_text=candidate.prefix_text,
                prefix_source_shard=candidate.shard_idx,
                prefix_source_doc=candidate.doc_idx,
                prefix_activating_seg_idx=candidate.seg_idx,
                verification_continuations=["c"],
                verification_seeds=[1],
                verification_percentiles=[99.0],
            )
        return None

    monkeypatch.setattr(
        _si,
        "_verify_candidate",
        _first_hit_then_fail,
    )

    collected, handler_id = _capture_info_logs()
    try:
        with pytest.raises(AxBenchBaseError):
            prepare_negative_prefixes(
                cfg,
                concepts_path,
                base_model=_base_model(),
                base_sae=SimpleNamespace(),
                target_model_meta=_stub_meta(),
            )
    finally:
        loguru_logger.remove(handler_id)

    assert call_count["n"] > 2, (
        f"Bailout fired too early; only {call_count['n']} calls before the loop terminated"
    )
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert rejected[0]["reason"] == (RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value)
    assert rejected[0]["n_verified_found"] == 1
    assert rejected[0]["llm_judge_score"] == 2
    assert not any("zero-hit bailout triggered" in m for m in collected)


def test_zero_hit_bailout_empty_pool_no_bailout(tmp_path, monkeypatch):
    """Empty candidate pool is a degenerate case: the for-loop body never
    executes, n_tried stays at 0, and the bailout guard never fires
    regardless of bail_after (since any valid bail_after >= 1 makes the
    `n_tried >= bail_after` predicate vacuously False). The existing
    zero-verified warning fires once with n_tried=0, and the function
    returns []. This documents the invariant; not a regression risk under
    the new bailout logic, but worth pinning so a future refactor that
    moves the bailout's `n_tried >= bail_after` check above the loop
    (or rephrases it as `n_tried > 0`) would surface here.
    """
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        verification_zero_hit_bail_after=1,  # smallest valid; would fire immediately if the guard were placed wrong
    )
    captured = _capture_record_counter(monkeypatch)
    warning_messages: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: warning_messages.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[],
            feature_shards=[],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
            candidate_index=_FixedIndex([]),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    # The bailout MUST NOT fire on an empty pool: n_tried stays 0, which is
    # below any valid bail_after threshold (>= 1).
    assert not any(
        c[0] == "stage0c.verification_zero_hit_bailouts" for c in captured
    ), f"Bailout should not fire on empty pool; got {captured}"
    # The existing zero-verified warning fires exactly once and reports
    # n_tried=0 (the side fix applies uniformly).
    zero_warnings = [
        m for m in warning_messages if "ZERO verified prefixes" in m
    ]
    assert len(zero_warnings) == 1, (
        f"Expected exactly one zero-verified warning, got {zero_warnings}"
    )
    assert "after 0 verification attempts" in zero_warnings[0], (
        f"Warning should report n_tried=0, got: {zero_warnings[0]!r}"
    )


# ---------------------------------------------------------------------------
# _verify_candidate
# ---------------------------------------------------------------------------


def _build_percentile_counter(pass_first_n: int) -> Any:
    """Return a stub compute_fc_percentiles that passes the first N calls,
    fails the rest.
    """
    state = {"n": 0}

    def _stubbed(prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id):
        del prefix, cont, f_c, gemma, sae, table, sae_layer, tokenizer_id
        state["n"] += 1
        if state["n"] <= pass_first_n:
            return (10.0, 80.0)  # out_pct=80 > 70 threshold -> pass
        return (10.0, 50.0)  # out_pct=50 < 70 threshold -> fail

    return _stubbed


def test_verify_candidate_accepts_when_3_of_5_pass(stub_env, tmp_path, monkeypatch):
    cfg = _cfg(
        tmp_path, verification_percentile_threshold=0.7, min_consistency_passes=3,
    )
    # Make the FIRST 3 percentile calls pass, the LAST 2 fail.
    monkeypatch.setattr(
        _si, "compute_fc_percentiles", _build_percentile_counter(3),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="short prefix",
        prefix_gemma_tokens=[1, 2, 3, 4],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is not None
    assert isinstance(row, _VerifiedNegativePrefix)
    assert len(row.verification_continuations) == 3  # only passing ones kept
    assert len(row.verification_seeds) == 3
    assert len(row.verification_percentiles) == 3
    for pct in row.verification_percentiles:
        assert pct >= 70.0
    # Row carries the right provenance.
    assert row.concept_id == 0
    assert row.prefix_source_shard == 0
    assert row.prefix_source_doc == 0
    assert row.prefix_activating_seg_idx == 0


def test_verify_candidate_rejects_when_only_2_of_5_pass(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, min_consistency_passes=3)
    monkeypatch.setattr(
        _si, "compute_fc_percentiles", _build_percentile_counter(2),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="prefix",
        prefix_gemma_tokens=[1, 2, 3],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is None


def test_verify_candidate_uses_seed_variance(stub_env, tmp_path, monkeypatch):
    """Each of the ``n_verification_seeds`` calls receives a distinct seed."""
    cfg = _cfg(
        tmp_path,
        n_verification_seeds=5,
        min_consistency_passes=3,
        verification_prefill_mode="repeated",
    )
    monkeypatch.setattr(
        _si, "compute_fc_percentiles", _build_percentile_counter(5),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=3, seg_idx=0,
        prefix_text="prefix",
        prefix_gemma_tokens=[1, 2, 3],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=7,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is not None
    seeds = row.verification_seeds
    assert len(set(seeds)) == 5  # all distinct
    # The batched kernel receives all 5 seeds in a single call.
    assert len(stub_env["batched_generator"].calls) >= 1
    call_seeds = stub_env["batched_generator"].calls[-1]["seeds"]
    assert len(set(call_seeds)) == 5


def test_verify_candidate_uses_exact_seed_formula(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        n_verification_seeds=3,
        min_consistency_passes=1,
        verification_prefill_mode="repeated",
    )
    monkeypatch.setattr(
        _si, "compute_fc_percentiles", _build_percentile_counter(3),
    )
    stub_env["batched_generator"].calls.clear()
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=4, seg_idx=0,
        prefix_text="prefix",
        prefix_gemma_tokens=[1, 2, 3],
    )

    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=7,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )

    assert row is not None
    expected = [
        cfg.generation.master_seed
        + _si._VERIFY_SEED_OFFSET
        + 7 * _si._CONCEPT_STRIDE
        + vseed_idx * _si._VSEED_INCREMENT
        + candidate.doc_idx
        for vseed_idx in range(cfg.n_verification_seeds)
    ]
    assert row.verification_seeds == expected
    # The batched kernel receives all seeds in one call.
    assert len(stub_env["batched_generator"].calls) == 1
    assert stub_env["batched_generator"].calls[0]["seeds"] == expected


def test_verify_candidate_calls_gemma_generator_with_base_tokens(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        n_verification_seeds=1,
        min_consistency_passes=1,
        verification_prefill_mode="repeated",
    )
    monkeypatch.setattr(
        _si, "compute_fc_percentiles", _build_percentile_counter(1),
    )
    base_model = _base_model()
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=2, seg_idx=0,
        prefix_text="prefix",
        prefix_gemma_tokens=[101, 222, 333],
    )

    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=4,
        base_model=base_model, base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )

    assert row is not None
    # The batched kernel is called once with a single-element seeds list.
    assert len(stub_env["batched_generator"].calls) == 1
    call = stub_env["batched_generator"].calls[0]
    assert call["base_model"] is base_model
    assert call["token_ids"] == [101, 222, 333]
    assert "feature_conditions" not in call


def test_verify_candidate_decodes_with_extraction_tokenizer(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        n_verification_seeds=1,
        min_consistency_passes=1,
        extraction_tokenizer_id="custom/stage0c-tokenizer",
    )
    seen_continuations: list[str] = []
    seen_tokenizer_ids: list[str] = []

    def _spy_percentile(prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id):
        del prefix, f_c, gemma, sae, table, sae_layer
        seen_continuations.append(cont)
        seen_tokenizer_ids.append(tokenizer_id)
        return (10.0, 80.0)

    monkeypatch.setattr(_si, "compute_fc_percentiles", _spy_percentile)
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="prefix",
        prefix_gemma_tokens=[101, 5, 6],
    )

    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )

    assert row is not None
    assert stub_env["tokenizer"].decode_calls == [[10, 20, 30]]
    assert seen_continuations == ["gemma10 gemma20 gemma30"]
    assert seen_tokenizer_ids == [cfg.extraction_tokenizer_id]


def test_verify_candidate_accepts_gemma_rich_continuation_fixture(
    stub_env, tmp_path, monkeypatch,
):
    """Controlled regression fixture for the d13 -> Gemma switch.

    The generator emits a token sequence that the extraction tokenizer
    decodes as concept-rich. Verification accepts because the percentile
    spy keys off that Gemma-decoded continuation; a target-decoder stub
    would not be involved in this path at all.
    """
    cfg = _cfg(
        tmp_path,
        n_verification_seeds=3,
        min_consistency_passes=2,
        verification_percentile_threshold=0.7,
        verification_prefill_mode="repeated",
    )
    stub_env["batched_generator"].token_list = [900]

    class _ConceptRichTokenizer(_FakeExtractionTokenizer):
        def decode(self, token_ids: list[int]) -> str:
            self.decode_calls.append(list(token_ids))
            if token_ids == [900]:
                return "gemma concept rich"
            return super().decode(token_ids)

    tokenizer = _ConceptRichTokenizer()
    stub_env["tokenizer"] = tokenizer

    def _activation_spy(
        prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id,
    ):
        del prefix, f_c, gemma, sae, table, sae_layer, tokenizer_id
        if cont == "gemma concept rich":
            return (10.0, 90.0)
        return (10.0, 0.0)

    monkeypatch.setattr(_si, "compute_fc_percentiles", _activation_spy)
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="prefix",
        prefix_gemma_tokens=[101, 7],
    )

    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )

    assert row is not None
    assert row.verification_continuations == [
        "gemma concept rich",
        "gemma concept rich",
        "gemma concept rich",
    ]
    # One batched call with 3 seeds (one per verification seed).
    assert len(stub_env["batched_generator"].calls) == 1
    assert len(stub_env["batched_generator"].calls[0]["seeds"]) == 3


# ---------------------------------------------------------------------------
# Strict-threshold semantics (p > threshold, not p >= threshold)
#
# Surfaced by a 175k-candidate pathological case on feature 6985 where
# every continuation had an inactive feature (percentile == 0.0), and an
# inclusive threshold of 0.0 would have admitted all of them as
# "verified" — producing meaningless negative-mode prefixes. Strict
# comparison excludes the truly-inactive case while still admitting the
# lowest positive percentiles.
# ---------------------------------------------------------------------------


def _build_percentile_returner(out_pcts: list[float]) -> Any:
    """Return a stub ``compute_fc_percentiles`` that hands back each
    value in ``out_pcts`` in order, one per call. Overshooting the list
    is an error (the test is mis-sized)."""
    idx = {"i": 0}

    def _stubbed(prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id):
        del prefix, cont, f_c, gemma, sae, table, sae_layer, tokenizer_id
        i = idx["i"]
        idx["i"] += 1
        if i >= len(out_pcts):
            raise AssertionError(
                f"stub received {i + 1} calls but only {len(out_pcts)} "
                f"values pre-arranged",
            )
        return (10.0, out_pcts[i])

    return _stubbed


def test_verify_candidate_strict_threshold_rejects_exact_equal(
    stub_env, tmp_path, monkeypatch,
):
    """A continuation whose percentile is exactly equal to the threshold
    must be treated as a miss (``p > threshold``, not ``>=``).

    If a future refactor flips this back to ``>=``, the threshold=0.0
    escape hatch loses its teeth — every truly-inactive continuation
    would be admitted as "verified".
    """
    cfg = _cfg(
        tmp_path,
        verification_percentile_threshold=0.7,  # raw_threshold = 70.0
        min_consistency_passes=3,
        n_verification_seeds=5,
    )
    # All 5 seeds return exactly the threshold. Strict-gt => 0 passes.
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        _build_percentile_returner([70.0, 70.0, 70.0, 70.0, 70.0]),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="p", prefix_gemma_tokens=[1, 2],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is None, "equal-to-threshold must be rejected"


def test_verify_candidate_strict_threshold_accepts_just_above(
    stub_env, tmp_path, monkeypatch,
):
    """A hair above the threshold passes; a hair below fails."""
    cfg = _cfg(
        tmp_path,
        verification_percentile_threshold=0.7,
        min_consistency_passes=3,
        n_verification_seeds=5,
    )
    # 3 just above, 2 below -> 3/5 passes, candidate accepted.
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        _build_percentile_returner([70.01, 70.01, 70.01, 69.99, 69.99]),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="p", prefix_gemma_tokens=[1, 2],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is not None
    assert len(row.verification_percentiles) == 3
    for pct in row.verification_percentiles:
        assert pct > 70.0


def test_verify_candidate_threshold_zero_rejects_truly_inactive(
    stub_env, tmp_path, monkeypatch,
):
    """At ``threshold=0.0``, a continuation where the feature never fires
    (``compute_fc_percentiles`` returns exactly 0.0) must be rejected.

    This is the headline use case: an operator who doesn't want Stage 0c
    to grind forever on a pathological feature can set the threshold to
    0.0, which admits every non-empty continuation (any positive
    percentile) while still filtering out truly-inactive ones. With the
    old ``>=`` semantics, 0.0 admitted everything including inactivity,
    which produced meaningless verified rows.
    """
    cfg = _cfg(
        tmp_path,
        verification_percentile_threshold=0.0,  # raw_threshold = 0.0
        min_consistency_passes=3,
        n_verification_seeds=5,
    )
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        _build_percentile_returner([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="p", prefix_gemma_tokens=[1, 2],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is None, "truly-inactive must be rejected even at threshold=0"


def test_verify_candidate_threshold_zero_accepts_any_positive_percentile(
    stub_env, tmp_path, monkeypatch,
):
    """At ``threshold=0.0``, any strictly-positive percentile admits the
    continuation. This is the essential property that makes
    ``threshold=0.0`` a useful escape hatch for pathological features."""
    cfg = _cfg(
        tmp_path,
        verification_percentile_threshold=0.0,
        min_consistency_passes=3,
        n_verification_seeds=5,
    )
    # 5/5 tiny positive percentiles. Strict-gt -> all pass.
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        _build_percentile_returner([0.5, 0.5, 0.5, 0.5, 0.5]),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="p", prefix_gemma_tokens=[1, 2],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is not None
    assert len(row.verification_percentiles) == 5
    for pct in row.verification_percentiles:
        assert pct > 0.0


def test_verify_candidate_threshold_zero_mixed_truly_inactive_and_active(
    stub_env, tmp_path, monkeypatch,
):
    """Mix of truly-inactive (0.0) and barely-active (0.5) at
    threshold=0.0: only the barely-active seeds count toward passes."""
    cfg = _cfg(
        tmp_path,
        verification_percentile_threshold=0.0,
        min_consistency_passes=3,
        n_verification_seeds=5,
    )
    # 2 inactive + 3 active -> 3 passes, candidate accepted, kept list
    # has exactly the 3 active percentiles (inactive ones excluded).
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        _build_percentile_returner([0.0, 0.5, 0.0, 0.6, 0.7]),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="p", prefix_gemma_tokens=[1, 2],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is not None
    assert sorted(row.verification_percentiles) == [0.5, 0.6, 0.7]
    assert 0.0 not in row.verification_percentiles


def test_verify_candidate_threshold_zero_insufficient_active_rejects(
    stub_env, tmp_path, monkeypatch,
):
    """Even at ``threshold=0.0``, the ``min_consistency_passes`` contract
    still holds: if fewer than ``min_consistency_passes`` continuations
    have any activation at all, the candidate is rejected."""
    cfg = _cfg(
        tmp_path,
        verification_percentile_threshold=0.0,
        min_consistency_passes=3,
        n_verification_seeds=5,
    )
    # Only 2 active; 3 inactive. Need 3 passes, got 2 -> reject.
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        _build_percentile_returner([0.0, 0.5, 0.0, 0.0, 0.5]),
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="p", prefix_gemma_tokens=[1, 2],
    )
    row = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        percentile_table=stub_env["percentile_table"],
        sae_layer=8,
    )
    assert row is None


# ---------------------------------------------------------------------------
# prepare_negative_prefixes (end-to-end with stubs)
# ---------------------------------------------------------------------------


def test_writes_parquet(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")  # content ignored; stub in place
    out = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == cfg.n_verified_prefixes_per_concept_target
    final_concepts = json.loads(paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text())
    assert [c["concept_id"] for c in final_concepts] == [0]
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert rejected == []
    # Schema check.
    col_names = set(table.schema.names)
    assert col_names == {
        "concept_id",
        "prefix_text",
        "prefix_source_shard",
        "prefix_source_doc",
        "prefix_activating_seg_idx",
        "verification_continuations",
        "verification_seeds",
        "verification_percentiles",
    }


def test_preflight_rejects_no_room_for_verification(stub_env, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, verification_length=64)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    candidate_work_started = {"value": False}

    def _unexpected_list_parquets(cfg_arg):
        del cfg_arg
        candidate_work_started["value"] = True
        return []

    monkeypatch.setattr(
        _si, "_list_parquets_for_feature_dir",
        _unexpected_list_parquets,
    )
    with pytest.raises(AxBenchBaseError, match="n_ctx .* greater"):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(64), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    assert candidate_work_started["value"] is False


def test_prepare_negative_prefixes_drops_overflow_before_inference(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        verification_length=64,
        n_verified_prefixes_per_concept_target=1,
    )

    class _ExpandingTokenizer:
        bos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del text
            body = list(range(1, 1327))
            if add_special_tokens:
                return [self.bos_token_id] + body
            return body

    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _ExpandingTokenizer(),
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="accepted 0 from 1 candidates"):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(1024), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    assert stub_env["generator"].calls == []


def test_zero_verified_rejects_candidate_and_exhaustion_raises(
    stub_env, tmp_path,
):
    """If all percentiles miss, the candidate is recorded as rejected."""
    cfg = _cfg(tmp_path)
    stub_env["percentile_pair"] = (10.0, 30.0)  # out_pct=30 < 70 threshold
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(
        AxBenchBaseError,
        match="lowering verification_percentile_threshold",
    ):
        prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert rejected == [
        {
            "candidate_index": 0,
            "feature_index": 0,
            "description": "concept 0",
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": 2,
        }
    ]
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()


def test_partial_verified_rejects_candidate_and_exhaustion_raises(
    stub_env, tmp_path,
):
    """If fewer than target verify, the candidate is rejected, not accepted."""
    cfg = _cfg(tmp_path, n_verified_prefixes_per_concept_target=100)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="accepted 0 from 1 candidates"):
        prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert rejected[0]["reason"] == (RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value)
    assert rejected[0]["n_verified_found"] == 2
    assert rejected[0]["llm_judge_score"] == 2


def test_shuffle_deterministic(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    out1 = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    rows1 = pq.read_table(out1).to_pylist()
    out1.unlink()
    # Reset generator call log before second run.
    stub_env["generator"].calls.clear()
    out2 = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    rows2 = pq.read_table(out2).to_pylist()
    assert [r["prefix_source_doc"] for r in rows1] == [
        r["prefix_source_doc"] for r in rows2
    ]


def test_stops_after_target_reached(stub_env, tmp_path):
    """Once the per-concept target is met, later candidates are skipped."""
    cfg = _cfg(tmp_path, n_verified_prefixes_per_concept_target=1)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    out = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    table = pq.read_table(out)
    assert table.num_rows == 1


def test_run_tag_none_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path).replace(run_tag=None)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )


def test_empty_concepts_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    stub_env["concepts"] = []
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="empty"):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )


def test_multiple_concepts_merged(stub_env, tmp_path):
    """Multi-concept end-to-end: each concept's verified rows end up in output."""
    stub_env["concepts"] = [
        _concept(concept_id=0, feature_index=0),
        _concept(concept_id=1, feature_index=1),  # fid 1 is in doc 1, 6
    ]
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        max_concepts=2,
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    out = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    concept_ids = {r["concept_id"] for r in rows}
    assert 0 in concept_ids
    assert 1 in concept_ids


def test_duplicate_candidate_features_preserve_validation_failure(
    stub_env, tmp_path,
):
    """Duplicate accepted features fail before canonical sidecar writes."""
    stub_env["concepts"] = [
        _concept(concept_id=0, feature_index=0),
        _concept(concept_id=1, feature_index=0),
    ]
    cfg = _cfg(
        tmp_path,
        run_tag="duplicate_candidate_features",
        n_verified_prefixes_per_concept_target=1,
        max_concepts=2,
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    with pytest.raises(AxBenchBaseError, match="appears more than once"):
        prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()


def test_indexed_stage0c_matches_legacy_stage0c_outputs(
    stub_env, tmp_path, monkeypatch,
):
    """Functional comparison: old scan path and indexed path write identical rows."""
    stub_env["concepts"] = [
        _concept(concept_id=0, feature_index=0),
        _concept(concept_id=1, feature_index=1),
    ]
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    original_verify_concept = _si._verify_concept

    def _legacy_verify_concept(*args, **kwargs):
        kwargs["candidate_index"] = None
        return original_verify_concept(*args, **kwargs)

    legacy_cfg = _cfg(
        tmp_path,
        run_tag="stage0c_legacy_compare",
        n_verified_prefixes_per_concept_target=1,
        max_concepts=2,
    )
    monkeypatch.setattr(
        _si,
        "_verify_concept",
        _legacy_verify_concept,
    )
    legacy_out = prepare_negative_prefixes(
        legacy_cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    legacy_concepts = json.loads(
        test_paths.default_paths(legacy_cfg.run_tag).concepts_json().read_text()
    )
    legacy_rejected = json.loads(
        test_paths.default_paths(legacy_cfg.run_tag).rejected_concepts_json().read_text()
    )
    legacy_rows = pq.read_table(legacy_out).to_pylist()

    indexed_cfg = _cfg(
        tmp_path,
        run_tag="stage0c_indexed_compare",
        n_verified_prefixes_per_concept_target=1,
        max_concepts=2,
    )
    monkeypatch.setattr(
        _si,
        "_verify_concept",
        original_verify_concept,
    )
    indexed_out = prepare_negative_prefixes(
        indexed_cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    indexed_concepts = json.loads(
        test_paths.default_paths(indexed_cfg.run_tag).concepts_json().read_text()
    )
    indexed_rejected = json.loads(
        test_paths.default_paths(indexed_cfg.run_tag).rejected_concepts_json().read_text()
    )
    indexed_rows = pq.read_table(indexed_out).to_pylist()

    assert indexed_concepts == legacy_concepts
    assert indexed_rejected == legacy_rejected
    assert indexed_rows == legacy_rows


def test_cross_prefix_stage0c_matches_serial_stage0c_artifacts(
    stub_env, tmp_path,
):
    """THE GATE: cross-prefix verification preserves Stage 0c semantics.

    PyArrow is free to vary physical parquet encoding while preserving the
    exact schema and logical rows, so raw file bytes are not the parity unit.
    """
    stub_env["concepts"] = [
        _concept(concept_id=0, feature_index=0),
        _concept(concept_id=1, feature_index=1),
        _concept(concept_id=2, feature_index=2),
    ]
    stub_env["percentile_pair"] = (10.0, 80.0)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    common_overrides = {
        "max_concepts": 3,
        "n_verified_prefixes_per_concept_target": 2,
        "verification_candidate_scan_limit": 8,
    }

    serial_cfg = _cfg(
        tmp_path,
        run_tag="stage0c_serial_prefix_gate",
        verification_cross_prefix_batch_size=1,
        **common_overrides,
    )
    serial_out = prepare_negative_prefixes(
        serial_cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    serial_concepts = test_paths.default_paths(serial_cfg.run_tag).concepts_json()
    serial_rejected = test_paths.default_paths(serial_cfg.run_tag).rejected_concepts_json()

    cross_cfg = _cfg(
        tmp_path,
        run_tag="stage0c_cross_prefix_gate",
        verification_cross_prefix_batch_size=4,
        **common_overrides,
    )
    cross_out = prepare_negative_prefixes(
        cross_cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    cross_concepts = test_paths.default_paths(cross_cfg.run_tag).concepts_json()
    cross_rejected = test_paths.default_paths(cross_cfg.run_tag).rejected_concepts_json()

    assert stub_env["cross_prefix_generator"].calls
    assert cross_concepts.read_bytes() == serial_concepts.read_bytes()
    assert cross_rejected.read_bytes() == serial_rejected.read_bytes()
    serial_table = pq.read_table(serial_out)
    cross_table = pq.read_table(cross_out)
    assert cross_table.schema == serial_table.schema
    assert cross_table.to_pylist() == serial_table.to_pylist()


def test_stage_0c_pool_first_five_accepts(stub_env, tmp_path, monkeypatch):
    """All candidates pass LLM + verification: only max_concepts are judged.

    Uses ``qc_batch_size=1`` and ``qc_pipeline=False`` so the per-candidate
    ``qc_client.calls`` assertion below maps 1:1 onto QC fetches (matching
    legacy ``filter([candidate])`` synchronous semantics).
    """
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
        qc_batch_size=1,
        qc_pipeline=False,
    )
    verify_calls: list[tuple[int, int]] = []

    def _accept_all(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        verify_calls.append(concept.feature_index)
        return _verified_rows(1)

    monkeypatch.setattr(_si, "_verify_concept", _accept_all)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    final_concepts = json.loads(paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text())
    assert [c["feature_index"] for c in final_concepts] == [0, 1, 2, 3, 4]
    assert [c["concept_id"] for c in final_concepts] == [0, 1, 2, 3, 4]
    assert len(stub_env["qc_client"].calls) == 5
    assert verify_calls == [0, 1, 2, 3, 4]
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert rejected == []


def test_stage_0c_qc_survivor_cap_limits_live_qc_and_verification(
    stub_env, tmp_path, monkeypatch,
):
    """Dev profiling cap stops after the requested number of QC survivors.

    This bounds live examples-QC/API fanout without moving QC back into
    Stage 0a. If the cap is lower than ``max_concepts``, Stage 0c should
    fail loudly after writing the partial QC/rejection diagnostics.
    """
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
        concept_examples_qc_target_surviving=3,
    )
    verify_calls: list[tuple[int, int]] = []

    def _accept_verified(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        verify_calls.append((concept.feature_index, concept.concept_id))
        return _verified_rows(1, concept_id=concept.concept_id)

    monkeypatch.setattr(_si, "_verify_concept", _accept_verified)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(
        AxBenchBaseError,
        match="concept_examples_qc_target_surviving=3",
    ):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert len(stub_env["qc_client"].calls) == 3
    assert verify_calls == [(0, 0), (1, 1), (2, 2)]
    report = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report().read_text()
    )
    assert report["n_total"] == 10
    assert report["n_processed"] == 3
    assert report["n_accepted"] == 3
    assert (
        json.loads(paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text())
        == []
    )
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()


def test_stage_0c_llm_rejects_first_three_then_accepts_next_five(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    stub_env["qc_client"] = _PatternQCClient({0: 0, 1: 0, 2: 0})
    # ``qc_batch_size=1`` and ``qc_pipeline=False`` preserve the legacy
    # "QC called once per candidate" contract that the
    # ``len(qc_client.calls) == 8`` assertion below depends on.
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
        qc_batch_size=1,
        qc_pipeline=False,
    )
    verify_calls: list[tuple[int, int]] = []

    def _accept_verified(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        verify_calls.append((concept.feature_index, concept.concept_id))
        return _verified_rows(1, concept_id=concept.concept_id)

    monkeypatch.setattr(_si, "_verify_concept", _accept_verified)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    final_concepts = json.loads(paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text())
    assert [c["feature_index"] for c in final_concepts] == [3, 4, 5, 6, 7]
    assert [c["concept_id"] for c in final_concepts] == [0, 1, 2, 3, 4]
    rows = pq.read_table(
        paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    ).to_pylist()
    assert [row["concept_id"] for row in rows] == [0, 1, 2, 3, 4]
    assert len(stub_env["qc_client"].calls) == 8
    # Verification uses stable candidate indices as provisional IDs; final
    # dense IDs are assigned only by merge.
    assert verify_calls == [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert [r["feature_index"] for r in rejected] == [0, 1, 2]
    assert all(
        r["reason"] == RejectedConceptReason.LLM_JUDGE_BELOW_MIN_SCORE.value for r in rejected
    )
    assert all(r["n_verified_found"] is None for r in rejected)
    assert all(r["llm_judge_score"] == 0 for r in rejected)


def test_stage_0c_verification_rejects_first_four_then_accepts_next_five(
    stub_env,
    tmp_path,
    monkeypatch,
):
    stub_env["concepts"] = [_concept(concept_id=i, feature_index=i) for i in range(10)]
    # ``qc_batch_size=1`` and ``qc_pipeline=False`` preserve the legacy
    # "QC called once per candidate" contract that the
    # ``len(qc_client.calls) == 9`` assertion below depends on.
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
        qc_batch_size=1,
        qc_pipeline=False,
    )
    verify_calls: list[int] = []

    def _pattern(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        verify_calls.append(concept.feature_index)
        if concept.feature_index < 4:
            return []
        return _verified_rows(1, concept_id=concept.concept_id)

    monkeypatch.setattr(_si, "_verify_concept", _pattern)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    final_concepts = json.loads(paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text())
    assert [c["feature_index"] for c in final_concepts] == [4, 5, 6, 7, 8]
    assert [c["concept_id"] for c in final_concepts] == [0, 1, 2, 3, 4]
    assert len(stub_env["qc_client"].calls) == 9
    assert verify_calls == list(range(9))
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert [r["feature_index"] for r in rejected] == [0, 1, 2, 3]
    assert all(
        r["reason"] == RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value for r in rejected
    )
    assert all(r["n_verified_found"] == 0 for r in rejected)
    assert all(r["llm_judge_score"] == 2 for r in rejected)


def test_stage_0c_mixed_rejections_have_reason_enum_values(
    stub_env,
    tmp_path,
    monkeypatch,
):
    stub_env["concepts"] = [_concept(concept_id=i, feature_index=i) for i in range(10)]
    stub_env["qc_client"] = _PatternQCClient(
        {
            0: 0,
            1: "fetch_error",
            2: "judge_error",
            3: "empty_examples",
        }
    )
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
    )

    def _pattern(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        if concept.feature_index == 4:
            return []
        return _verified_rows(1, concept_id=concept.concept_id)

    monkeypatch.setattr(_si, "_verify_concept", _pattern)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert [(r["feature_index"], r["reason"]) for r in rejected] == [
        (0, RejectedConceptReason.LLM_JUDGE_BELOW_MIN_SCORE.value),
        (1, RejectedConceptReason.LLM_JUDGE_FETCH_ERROR.value),
        (2, RejectedConceptReason.LLM_JUDGE_ERROR.value),
        (3, RejectedConceptReason.LLM_JUDGE_EMPTY_EXAMPLES.value),
        (4, RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value),
    ]


def test_rejected_concept_reason_from_qc_outcome_returns_enum() -> None:
    assert RejectedConceptReason.from_qc_outcome(QcOutcome.REJECTED) is (
        RejectedConceptReason.LLM_JUDGE_BELOW_MIN_SCORE
    )
    assert RejectedConceptReason.from_qc_outcome(QcOutcome.FETCH_ERROR) is (
        RejectedConceptReason.LLM_JUDGE_FETCH_ERROR
    )
    assert RejectedConceptReason.from_qc_outcome(QcOutcome.JUDGE_ERROR) is (
        RejectedConceptReason.LLM_JUDGE_ERROR
    )
    assert RejectedConceptReason.from_qc_outcome(QcOutcome.EMPTY_EXAMPLES) is (
        RejectedConceptReason.LLM_JUDGE_EMPTY_EXAMPLES
    )
    with pytest.raises(AxBenchBaseError, match="Unexpected"):
        RejectedConceptReason.from_qc_outcome(QcOutcome.ACCEPTED)


def test_build_qc_report_rejects_unknown_outcome() -> None:
    # Intentionally fabricates an out-of-enum outcome value to verify
    # the runtime guard. ``QcOutcome`` is a :class:`StrEnum` so we cast
    # through ``Any`` to silence the type-checker — the test itself is
    # asserting what happens when somebody writes a NEW outcome string
    # and forgets to teach the dispatch table about it. Production
    # callers rely on the StrEnum at type-check time; this test is for
    # the runtime belt-and-braces.
    with pytest.raises(AxBenchBaseError, match="Unexpected concept examples QC"):
        _si._build_qc_report(
            n_total=1,
            per_feature=[
                QCFeatureScore(
                    feature_index=0,
                    description="concept",
                    outcome=cast(Any, "surprising"),
                    score=None,
                    n_examples=0,
                    error_message=None,
                )
            ],
            judge_min_score=1,
        )


def test_stage_0c_writes_qc_report_for_judged_candidates(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    stub_env["qc_client"] = _PatternQCClient({0: 0, 1: "fetch_error"})
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
    )
    monkeypatch.setattr(
        _si,
        "_verify_concept",
        lambda cfg_arg, concept, *args, **kwargs: _verified_rows(1),
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    report = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report().read_text()
    )
    assert set(report) == {
        "n_total",
        "n_accepted",
        "n_rejected",
        "n_fetch_errors",
        "n_judge_errors",
        "n_empty_examples",
        "score_histogram",
        "per_feature",
        "judge_min_score",
        "n_processed",
    }
    assert report["n_total"] == 10
    assert report["n_processed"] == 7
    assert report["n_accepted"] == 5
    assert report["n_rejected"] == 1
    assert report["n_fetch_errors"] == 1
    assert [row["feature_index"] for row in report["per_feature"]] == list(range(7))


# Legacy ``test_stage_0c_rejects_malformed_qc_survivor_contract`` and
# ``test_stage_0c_requires_one_qc_report_row_per_candidate`` were
# removed when Stage 0c migrated from per-candidate
# ``qc_client.filter([candidate])`` to batched ``QcBuffer.next(idx)``.
# Those tests asserted invariants between ``survivors`` and
# ``per_feature`` and the "exactly one per-feature row per candidate"
# contract — both internal details of the legacy single-candidate
# call path. ``QcBuffer`` keys results solely on ``per_feature`` and
# tolerates missing rows by returning ``None`` (see
# ``test_qc_buffer.py::test_qc_buffer_returns_none_when_per_feature_missing``),
# so the legacy invariants no longer apply.


def test_prepare_negative_prefixes_requires_cluster_contract_together_with_clustered_config(
    tmp_path,
):
    """Fail-loud pairing (PR #739 C4): a clustered config demands a
    validated ``ClusterFeatureContract`` be threaded through explicitly --
    there is no silent "run unclustered anyway" fallback. Fires before any
    model/candidate-pool work, so no fixtures beyond ``cfg`` are needed."""
    cfg = _cfg(tmp_path).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    with pytest.raises(
        AxBenchBaseError,
        match="requires config and validated feature-contract state",
    ):
        prepare_negative_prefixes(
            cfg, Path("/unused/concepts.json"),
            base_model=None, base_sae=None, target_model_meta=_stub_meta(),
        )


def test_prepare_negative_prefixes_rejects_cluster_contract_for_unclustered_config(
    tmp_path,
):
    """The converse pairing failure: an unclustered config must not silently
    accept a caller-supplied contract (e.g. a stale one from a previous
    clustered run_tag) -- same fail-loud check, opposite direction."""
    cfg = _cfg(tmp_path)
    assert cfg.conditioning is None
    with pytest.raises(
        AxBenchBaseError,
        match="requires config and validated feature-contract state",
    ):
        prepare_negative_prefixes(
            cfg, Path("/unused/concepts.json"),
            base_model=None, base_sae=None, target_model_meta=_stub_meta(),
            cluster_contract=cast(Any, object()),
        )


def test_prepare_negative_prefixes_clustered_routes_through_collective_agreement_wiring(
    tmp_path, monkeypatch,
):
    """Wiring proof (PR #739 FU-D-stage0c fix batch): ``prepare_negative_
    prefixes``'s clustered branch must call ``cluster_artifacts.validate_
    cluster_candidate_identity_with_collective_agreement`` -- not the bare,
    pre-fix ``load_cluster_candidate_identity`` -- with exactly the
    contract/candidate_path/n_sae_features/world_size this call site
    has in scope, computed BEFORE the check (not after, as before this fix
    batch moved the ``n_sae_features`` computation earlier), and must
    propagate whatever it raises without swallowing it.
    """
    cfg = _cfg(tmp_path).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    contract = cast(Any, SimpleNamespace(feature_contract_fingerprint="a" * 64))
    concepts_path = Path("/unused/concepts.json")
    base_sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=16384))
    seen: dict[str, Any] = {}

    def _fake_validate(cfg_arg, contract_arg, *, candidate_path, n_sae_features, world_size):
        seen["cfg"] = cfg_arg
        seen["contract"] = contract_arg
        seen["candidate_path"] = candidate_path
        seen["n_sae_features"] = n_sae_features
        seen["world_size"] = world_size
        raise AxBenchBaseError("sentinel from the collective-agreement wrapper")

    monkeypatch.setattr(
        negative_prefixes,
        "validate_cluster_candidate_identity_with_collective_agreement",
        _fake_validate,
    )
    # negative_prefixes.py no longer imports the bare `load_cluster_
    # candidate_identity` name at all (only the collective-agreement
    # wrapper above) -- confirmed by grep; there is no leftover call site
    # left to patch away.
    assert not hasattr(negative_prefixes, "load_cluster_candidate_identity")

    with pytest.raises(AxBenchBaseError, match="sentinel from the collective-agreement wrapper"):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=None, base_sae=base_sae, target_model_meta=_stub_meta(),
            rank=2, world_size=5,
            cluster_contract=contract,
        )

    assert seen == {
        "cfg": cfg,
        "contract": contract,
        "candidate_path": concepts_path,
        "n_sae_features": 16384,
        "world_size": 5,
    }


def test_prepare_negative_prefixes_qc_disabled_calls_dispatcher_directly(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, enable_concept_examples_qc=False)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    expected = tmp_path / "negative_mode_prefixes.parquet"
    calls: list[dict[str, Any]] = []

    def _fake_dispatch(
        cfg_arg,
        concepts_path_arg,
        base_model,
        base_sae,
        target_model_meta,
        *,
        n_sae_features,
        response_cache,
        rank=0,
        world_size=1,
    ):
        calls.append({
            "cfg": cfg_arg,
            "concepts_path": concepts_path_arg,
            "base_model": base_model,
            "base_sae": base_sae,
            "target_model_meta": target_model_meta,
            "n_sae_features": n_sae_features,
            "response_cache": response_cache,
            "rank": rank,
            "world_size": world_size,
        })
        return expected

    monkeypatch.setattr(
        stage0c_runner, "dispatch_prepare_negative_prefixes", _fake_dispatch,
    )
    base_model = _base_model()
    base_sae = SimpleNamespace()
    target_model_meta = _stub_meta()

    got = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=base_model,
        base_sae=base_sae,
        target_model_meta=target_model_meta,
    )

    assert got == expected
    assert calls == [{
        "cfg": cfg,
        "concepts_path": concepts_path,
        "base_model": base_model,
        "base_sae": base_sae,
        "target_model_meta": target_model_meta,
        "n_sae_features": 16384,
        "response_cache": None,
        "rank": 0,
        "world_size": 1,
    }]


def test_prepare_negative_prefixes_api_mode_opens_shared_response_cache(
    stub_env, tmp_path, monkeypatch,
):
    """API generation needs the synth cache even when examples QC is off."""
    cfg = _cfg(
        tmp_path,
        enable_concept_examples_qc=False,
        verification_api_model="openai/gpt-5-nano",
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    expected = tmp_path / "negative_mode_prefixes.parquet"
    opened_paths: list[Path] = []
    seen_caches: list[Any] = []

    class _FakeCache:
        def __init__(self, path: Path) -> None:
            opened_paths.append(path)

        def __enter__(self) -> _FakeCache:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    def _fake_dispatch(*args: Any, response_cache: Any, **kwargs: Any) -> Path:
        del args, kwargs
        seen_caches.append(response_cache)
        return expected

    monkeypatch.setattr(negative_prefixes, "SynthResponseCache", _FakeCache)
    monkeypatch.setattr(
        stage0c_runner,
        "dispatch_prepare_negative_prefixes",
        _fake_dispatch,
    )

    got = prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )

    assert got == expected
    assert opened_paths == [paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()]
    assert len(seen_caches) == 1
    assert isinstance(seen_caches[0], _FakeCache)


def test_stage_0c_pool_exhaustion_after_llm_rejections_names_remediation_levers(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(4)
    ]
    stub_env["qc_client"] = _PatternQCClient({0: 0, 1: 0, 2: 0, 3: 0})
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
    )
    verify_calls: list[int] = []

    def _unexpected_verify(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        verify_calls.append(concept.feature_index)
        return _verified_rows(1)

    monkeypatch.setattr(_si, "_verify_concept", _unexpected_verify)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError) as excinfo:
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    msg = str(excinfo.value)
    assert "lowering verification_percentile_threshold" in msg
    assert "raising min_corpus_count" in msg
    assert "base SAE LM is genuinely too weak" in msg
    assert verify_calls == []
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert len(rejected) == 4


def _one_activation_example() -> ActivatingExample:
    return ActivatingExample(
        tokens=["token"],
        text="token",
        max_value=1.0,
        max_value_token_index=0,
        peak_char_offset=0,
    )


class _NoopDetokenizer:
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


class _CacheAwareLLM(LLMJudgeClient):
    complete_calls = 0
    api_calls = 0

    def __init__(self, cfg, response_cache=None):
        self._cfg = cfg
        self._response_cache = response_cache

    @classmethod
    def reset_counts(cls) -> None:
        cls.complete_calls = 0
        cls.api_calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str:
        del messages
        _CacheAwareLLM.complete_calls += 1
        if cache_key is not None and self._response_cache is not None:
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                return cached
        _CacheAwareLLM.api_calls += 1
        content = "Rating: 2\nExplanation: examples match the concept."
        if cache_key is not None and self._response_cache is not None:
            self._response_cache.set(cache_key, content)
        return content


def _cache_qc_factory(cfg, response_cache=None) -> ConceptExamplesQC:
    llm = _CacheAwareLLM(
        cfg.concept_examples_qc_judge,
        response_cache=response_cache,
    )
    judge = make_llm_concept_examples_judge(
        llm,
        max_examples_in_prompt=cfg.n_activation_examples,
        model=cfg.concept_examples_qc_judge.model,
    )
    return ConceptExamplesQC(
        model_id=cfg.neuronpedia_model_id,
        sae_source=cfg.neuronpedia_source,
        n_examples=cfg.n_activation_examples,
        judge=judge,
        tokenizer=_NoopDetokenizer(),
        min_score=cfg.activation_example_judge_min_score,
        fetch_fn=lambda *args: [_one_activation_example()],
        max_workers=1,
    )


def test_stage_0c_uses_prepopulated_synth_response_cache(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    # ``qc_batch_size=1`` and ``qc_pipeline=False`` preserve the legacy
    # contract that _CacheAwareLLM.complete_calls counts QC fetches per
    # *visited* candidate (one per outer-loop iteration).
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
        qc_batch_size=1,
        qc_pipeline=False,
    )
    monkeypatch.setattr(concept_list, "_build_concept_examples_qc_client", _cache_qc_factory)
    monkeypatch.setattr(
        _si,
        "_verify_concept",
        lambda cfg_arg, concept, *args, **kwargs: _verified_rows(1),
    )
    with SynthResponseCache(
        paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()
    ) as cache:
        qc = _cache_qc_factory(cfg, response_cache=cache)
        for fid in range(5):
            survivors, report = qc.filter([(fid, f"concept {fid}")])
            assert survivors == [(fid, f"concept {fid}")]
            assert report.per_feature[0].score == 2

    _CacheAwareLLM.reset_counts()
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    assert _CacheAwareLLM.complete_calls == 5
    assert _CacheAwareLLM.api_calls == 0


def test_stage_0c_rerun_with_same_run_tag_makes_zero_new_llm_api_calls(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    # ``qc_batch_size=1`` and ``qc_pipeline=False`` preserve the legacy
    # contract that ``_CacheAwareLLM.api_calls`` counts QC API calls per
    # *visited* candidate (one per outer-loop iteration).
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
        qc_batch_size=1,
        qc_pipeline=False,
    )
    monkeypatch.setattr(concept_list, "_build_concept_examples_qc_client", _cache_qc_factory)
    monkeypatch.setattr(
        _si,
        "_verify_concept",
        lambda cfg_arg, concept, *args, **kwargs: _verified_rows(1),
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    _CacheAwareLLM.reset_counts()
    prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    assert _CacheAwareLLM.complete_calls == 5
    assert _CacheAwareLLM.api_calls == 5

    _CacheAwareLLM.reset_counts()
    prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    # Idempotent resume: with complete artifacts from the first run
    # (concepts.json + rejected_concepts.json + negative_mode_prefixes.parquet)
    # present under the same run_tag, Stage 0c short-circuits BEFORE
    # any LLM work — stricter than the old cache-hit behavior, which
    # would have called ``complete`` 5 times and hit the cache for
    # zero API calls. Zero LLM work end-to-end is the goal for a
    # resume after a crash in a later stage.
    assert _CacheAwareLLM.complete_calls == 0
    assert _CacheAwareLLM.api_calls == 0


def test_stage_0c_all_rejects_names_remediation_levers(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
    )
    monkeypatch.setattr(
        _si, "_verify_concept", lambda *args, **kwargs: [],
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError) as excinfo:
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    msg = str(excinfo.value)
    assert "lowering verification_percentile_threshold" in msg
    assert "raising min_corpus_count" in msg
    assert "base SAE LM is genuinely too weak" in msg
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert len(rejected) == 10
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()


def test_stage_0c_four_accepts_six_rejects_raises(
    stub_env, tmp_path, monkeypatch,
):
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(10)
    ]
    cfg = _cfg(
        tmp_path,
        max_concepts=5,
        n_verified_prefixes_per_concept_target=1,
    )

    def _first_four_accept(cfg_arg, concept, *args, **kwargs):
        del cfg_arg, args, kwargs
        if concept.concept_id < 4:
            return _verified_rows(1)
        return []

    monkeypatch.setattr(
        _si,
        "_verify_concept",
        _first_four_accept,
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="accepted 4 from 10 candidates"):
        prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text()
    )
    assert len(rejected) == 6
    assert [r["feature_index"] for r in rejected] == [4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Resolver / glob / parquet wrappers (monkeypatch targets)
# ---------------------------------------------------------------------------


def test_resolve_parquet_out_of_range_raises():
    with pytest.raises(AxBenchBaseError, match="5"):
        _si._resolve_parquet([Path("/a")], 5)


def test_resolve_parquet_negative_raises():
    with pytest.raises(AxBenchBaseError, match="validation_shards"):
        _si._resolve_parquet([Path("/a"), Path("/b")], -1)


def test_resolve_feature_shard_ok():
    shards = [
        Path("/d/shard_00005_features.safetensors.zst"),
        Path("/d/shard_00006_features.safetensors.zst"),
    ]
    assert _si._resolve_feature_shard(shards, 5, "/d") == shards[0]


def test_resolve_feature_shard_missing_raises():
    shards = [Path("/d/shard_00005_features.safetensors.zst")]
    with pytest.raises(AxBenchBaseError, match="No feature shard"):
        _si._resolve_feature_shard(shards, 999, "/d")


def test_list_parquets_wrapper_phase_none(monkeypatch):
    """``_list_parquets_for_feature_dir`` with parsed dataset_phase=DatasetPhase.BASE."""
    sentinel_data_dir = Path("/stub/data/dir")
    stub_parquet_strings = ["/stub/data/dir/shard_00000.parquet"]

    def _fake_parse_feature_dir(p):
        del p
        return FeatureDirectoryComponents(
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )

    def _fake_resolve_data_dir(root, phase_value):
        del root, phase_value
        return sentinel_data_dir

    def _fake_list_parquet_files(data_dir):
        del data_dir
        return stub_parquet_strings

    monkeypatch.setattr(
        _si, "parse_feature_dir", _fake_parse_feature_dir,
    )
    monkeypatch.setattr(
        _si, "resolve_data_dir", _fake_resolve_data_dir,
    )
    monkeypatch.setattr(
        _si, "list_parquet_files", _fake_list_parquet_files,
    )
    cfg = make_valid_config(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
    )
    result = _si._list_parquets_for_feature_dir(cfg)
    assert result == [Path("/stub/data/dir/shard_00000.parquet")]


def test_list_parquets_wrapper_phase_enum(monkeypatch):
    """``_list_parquets_for_feature_dir`` with a non-base DatasetPhase member."""
    sentinel_data_dir = Path("/stub/data/dir/standard_sft")
    captured: dict[str, Any] = {}

    def _fake_parse_feature_dir(p):
        del p
        return FeatureDirectoryComponents(
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.STANDARD_SFT,
            context_independent_features=False,
        )

    def _fake_resolve_data_dir(root, phase_value):
        captured["phase_value"] = phase_value
        return sentinel_data_dir

    def _fake_list_parquet_files(data_dir):
        del data_dir
        return ["/stub/data/dir/standard_sft/shard_00000.parquet"]

    monkeypatch.setattr(
        _si, "parse_feature_dir", _fake_parse_feature_dir,
    )
    monkeypatch.setattr(
        _si, "resolve_data_dir", _fake_resolve_data_dir,
    )
    monkeypatch.setattr(
        _si, "list_parquet_files", _fake_list_parquet_files,
    )
    cfg = make_valid_config(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/standard_sft"
        ),
    )
    result = _si._list_parquets_for_feature_dir(cfg)
    assert result == [Path("/stub/data/dir/standard_sft/shard_00000.parquet")]
    assert captured["phase_value"] == DatasetPhase.STANDARD_SFT.value


def test_prepare_negative_prefixes_signature_excludes_target_decoder_only():
    params = inspect.signature(prepare_negative_prefixes).parameters
    assert "target_decoder_only" not in params


def test_base_context_window_reads_gemma_n_ctx():
    assert _si._base_context_window(_base_model(8192)) == 8192


def test_base_context_window_missing_n_ctx_raises():
    with pytest.raises(AxBenchBaseError, match="n_ctx"):
        _si._base_context_window(SimpleNamespace(cfg=SimpleNamespace()))


# ---------------------------------------------------------------------------
# Observability: Stage 0c progress logging
#
# Stage 0c can burn 10-20 minutes on a single concept silently — just
# model forward passes, no flushed log lines. That makes a healthy
# pipeline look identical to a deadlock. These tests pin the logs that
# make progress visible: scan summary, within-concept heartbeat, end-
# of-concept summary, and per-concept stage progress.
# ---------------------------------------------------------------------------


def _capture_info_logs() -> tuple[list[str], int]:
    """Attach a loguru sink that records every INFO+ message to a list.

    Returns ``(messages, handler_id)``. The caller is responsible for
    removing the handler via ``logger.remove(handler_id)`` in a
    ``finally:`` so a failing assertion doesn't leak the sink into the
    rest of the suite.
    """
    collected: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: collected.append(msg.record["message"]),
        level="INFO",
    )
    return collected, handler_id


def test_verify_concept_logs_scan_summary(stub_env, tmp_path):
    """``_verify_concept`` must emit a "candidate scan yielded N prefixes"
    log so operators can tell whether the bottleneck is shard scanning
    (no candidates returned) vs. verification (all candidates failing).
    """
    cfg = _cfg(tmp_path)
    concepts_path = tmp_path / "concepts.json"
    concepts_path.write_text("[]")
    collected, handler_id = _capture_info_logs()
    try:
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert any(
        "candidate scan yielded" in m and "prefixes" in m for m in collected
    ), f"no scan-summary log found; got: {collected}"


def test_verify_concept_logs_end_of_concept_summary(stub_env, tmp_path):
    """After finishing a concept, emit "Concept X done: verified Y/Z" so
    operators can track progress per-concept and spot low pass rates
    early."""
    cfg = _cfg(tmp_path)
    concepts_path = tmp_path / "concepts.json"
    concepts_path.write_text("[]")
    collected, handler_id = _capture_info_logs()
    try:
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    finally:
        loguru_logger.remove(handler_id)

    # Per-concept "done" line carries pass_rate and elapsed.
    done_lines = [m for m in collected if "done: verified" in m]
    assert done_lines, f"no end-of-concept log found; got: {collected}"
    for line in done_lines:
        assert "pass_rate=" in line
        assert "elapsed=" in line


def test_prepare_negative_prefixes_logs_per_concept_completion(
    stub_env, tmp_path,
):
    """Stage-level log lines mark each candidate's completion with ETA for
    the remainder of the stage. Without this, a chain of 5+ candidates
    looks opaque: no intermediate progress between 0c start and end."""
    cfg = _cfg(tmp_path)
    concepts_path = tmp_path / "concepts.json"
    concepts_path.write_text("[]")
    collected, handler_id = _capture_info_logs()
    try:
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    finally:
        loguru_logger.remove(handler_id)

    stage_lines = [m for m in collected if m.startswith("[Stage 0c]")]
    assert stage_lines, f"no [Stage 0c] stage-level logs; got: {collected}"
    # Should include both "Verifying" and "Completed".
    assert any("Verifying negative prefixes" in m for m in stage_lines)
    assert any("Completed candidate" in m for m in stage_lines)
    # Completion lines carry eta remaining.
    completion_lines = [m for m in stage_lines if "Completed candidate" in m]
    for line in completion_lines:
        assert "eta remaining" in line


def test_verify_concept_logs_within_concept_progress_heartbeat(
    tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, n_verified_prefixes_per_concept_target=2)
    candidates = [
        _PrefixCandidate(
            shard_idx=0,
            doc_idx=idx,
            seg_idx=idx,
            prefix_text=f"prefix {idx}",
            prefix_gemma_tokens=[101, idx + 1],
        )
        for idx in range(2)
    ]

    monkeypatch.setattr(
        _si,
        "_PROGRESS_EVERY_N_CANDIDATES",
        1,
    )
    monkeypatch.setattr(
        _si,
        "_find_candidate_prefixes",
        lambda *args, **kwargs: list(candidates),
    )
    monkeypatch.setattr(
        _si,
        "_verify_candidate",
        lambda *args, **kwargs: None,
    )
    collected, handler_id = _capture_info_logs()
    try:
        rows = _si._verify_concept(
            cfg,
            _concept(concept_id=0, feature_index=0),
            parquets=[Path("/tmp/shard.parquet")],
            feature_shards=[Path("/tmp/shard_00000_features.safetensors")],
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            sae_layer=8,
            base_context_window=8192,
        )
    finally:
        loguru_logger.remove(handler_id)

    assert rows == []
    heartbeat_lines = [
        m for m in collected
        if "verified 0/2 after" in m and "pass_rate=" in m and "eta" in m
    ]
    assert heartbeat_lines, f"no heartbeat log found; got: {collected}"


def test_verify_concept_respects_attempt_limit(tmp_path, monkeypatch):
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=1,
        verification_candidate_attempt_limit=2,
    )
    candidates = [
        _PrefixCandidate(
            shard_idx=0,
            doc_idx=idx,
            seg_idx=idx,
            prefix_text=f"prefix {idx}",
            prefix_gemma_tokens=[101, idx + 1],
        )
        for idx in range(5)
    ]
    attempted: list[int] = []

    monkeypatch.setattr(
        _si,
        "_find_candidate_prefixes",
        lambda *args, **kwargs: list(candidates),
    )

    def _reject_candidate(
        cfg_arg,
        candidate,
        f_c,
        concept_id,
        base_model,
        base_sae,
        percentile_table,
        *,
        sae_layer,
    ):
        del cfg_arg, f_c, concept_id, base_model, base_sae, percentile_table, sae_layer
        attempted.append(candidate.doc_idx)
        return None

    monkeypatch.setattr(_si, "_verify_candidate", _reject_candidate)

    rows = _si._verify_concept(
        cfg,
        _concept(concept_id=0, feature_index=0),
        parquets=[Path("/tmp/shard.parquet")],
        feature_shards=[Path("/tmp/shard_00000_features.safetensors")],
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        sae_layer=8,
        base_context_window=8192,
    )

    assert rows == []
    assert len(attempted) == 2
    assert set(attempted).issubset({0, 1, 2, 3, 4})


def test_verify_concept_progress_heartbeat_threshold_constant_is_small():
    """Contract: heartbeat every N candidates must be small enough that a
    long verification doesn't go silent. If someone bumps this to 10k
    we lose the observability win entirely.
    """
    assert _si._PROGRESS_EVERY_N_CANDIDATES <= 50
    assert _si._PROGRESS_EVERY_SECONDS <= 120.0
    # Sanity: must be positive or we'd get division by zero / infinite
    # loops downstream.
    assert _si._PROGRESS_EVERY_N_CANDIDATES >= 1
    assert _si._PROGRESS_EVERY_SECONDS > 0


# ---------------------------------------------------------------------------
# Stage 0c idempotent-resume: skip if outputs are already complete
#
# Stage 0c can spend hours on verification (observed: 5h on a single
# target for the dev-tier smoke run). If a subsequent pipeline stage
# crashes (e.g. Stage 1b errors out on a QC threshold) and the
# operator re-runs, the default behavior re-does all of Stage 0c's
# work. Idempotent resume checks for a complete artifact set and
# short-circuits.
# ---------------------------------------------------------------------------


def _write_complete_stage_0c_artifacts(
    cfg,
    *,
    n_concepts: int = 1,
    rows_per_concept: int | None = None,
    write_qc_report: bool | None = None,
    rejected_records: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    """Populate minimal files that match a completed Stage 0c run."""
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    candidate_concepts_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()

    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts = [
        Concept(
            concept_id=i,
            feature_index=100 + i,
            layer=8,
            sae_source=cfg.neuronpedia_source,
            description=f"fixture concept {i}",
            corpus_count=50,
            description_source="neuronpedia",
        )
        for i in range(n_concepts)
    ]
    concepts_data = [asdict(concept) for concept in concepts]
    concepts_path.write_text(json.dumps(concepts_data))
    rejected_payload = [] if rejected_records is None else rejected_records
    rejected_by_index = {
        record["candidate_index"]: record for record in rejected_payload
    }
    assert len(rejected_by_index) == len(rejected_payload)
    concept_iter = iter(concepts_data)
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_idx in range(n_concepts + len(rejected_payload)):
        rejected = rejected_by_index.get(candidate_idx)
        if rejected is not None:
            candidate_payloads.append(
                {
                    "feature_index": rejected["feature_index"],
                    "layer": 8,
                    "sae_source": cfg.neuronpedia_source,
                    "description": rejected["description"],
                    "corpus_count": 50,
                    "description_source": "neuronpedia",
                }
            )
        else:
            concept = next(concept_iter)
            candidate_payloads.append(
                {key: value for key, value in concept.items() if key != "concept_id"}
            )
    assert next(concept_iter, None) is None
    candidate_concepts_path.write_text(
        json.dumps(candidate_payloads)
    )
    candidate_concepts = validate_mod.load_and_validate_candidate_concepts(
        candidate_concepts_path,
        n_sae_features=16384,
    )
    pool_identity = _si.candidate_pool_identity(candidate_concepts)
    rejected_path.write_text(json.dumps(rejected_payload))
    should_write_qc_report = (
        cfg.enable_concept_examples_qc if write_qc_report is None
        else write_qc_report
    )
    qc_report_payload: dict[str, Any] | None = None
    if cfg.enable_concept_examples_qc:
        per_feature: list[dict[str, Any]] = []
        for candidate_idx, candidate in enumerate(candidate_concepts):
            rejected = rejected_by_index.get(candidate_idx)
            if rejected is None or (
                rejected["reason"]
                == RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value
            ):
                outcome = QcOutcome.ACCEPTED.value
                score = 2 if rejected is None else rejected["llm_judge_score"]
                n_examples = 1
                error_message = None
            else:
                outcome = {
                    RejectedConceptReason.LLM_JUDGE_BELOW_MIN_SCORE.value: (
                        QcOutcome.REJECTED.value
                    ),
                    RejectedConceptReason.LLM_JUDGE_FETCH_ERROR.value: (
                        QcOutcome.FETCH_ERROR.value
                    ),
                    RejectedConceptReason.LLM_JUDGE_ERROR.value: (
                        QcOutcome.JUDGE_ERROR.value
                    ),
                    RejectedConceptReason.LLM_JUDGE_EMPTY_EXAMPLES.value: (
                        QcOutcome.EMPTY_EXAMPLES.value
                    ),
                }[rejected["reason"]]
                score = rejected["llm_judge_score"]
                n_examples = 1 if outcome in {
                    QcOutcome.REJECTED.value,
                    QcOutcome.JUDGE_ERROR.value,
                } else 0
                error_message = (
                    "fixture error"
                    if outcome in {
                        QcOutcome.FETCH_ERROR.value,
                        QcOutcome.JUDGE_ERROR.value,
                    }
                    else None
                )
            per_feature.append(
                {
                    "feature_index": candidate.feature_index,
                    "description": candidate.description,
                    "outcome": outcome,
                    "score": score,
                    "n_examples": n_examples,
                    "error_message": error_message,
                }
            )
        outcome_counts = {
            outcome.value: sum(
                row["outcome"] == outcome.value for row in per_feature
            )
            for outcome in QcOutcome
        }
        score_histogram = {
            str(score): sum(row["score"] == score for row in per_feature)
            for score in range(3)
            if any(row["score"] == score for row in per_feature)
        }
        qc_report_payload = {
            "n_total": len(candidate_concepts),
            "n_accepted": outcome_counts[QcOutcome.ACCEPTED.value],
            "n_rejected": outcome_counts[QcOutcome.REJECTED.value],
            "n_fetch_errors": outcome_counts[QcOutcome.FETCH_ERROR.value],
            "n_judge_errors": outcome_counts[QcOutcome.JUDGE_ERROR.value],
            "n_empty_examples": outcome_counts[QcOutcome.EMPTY_EXAMPLES.value],
            "score_histogram": score_histogram,
            "per_feature": per_feature,
            "judge_min_score": cfg.activation_example_judge_min_score,
            "n_processed": len(per_feature),
        }
    if should_write_qc_report:
        paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report().write_text(
            json.dumps(qc_report_payload)
        )

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = (
        cfg.n_verified_prefixes_per_concept_target if rows_per_concept is None else rows_per_concept
    )
    n_passes = max(cfg.min_consistency_passes, 1)
    pct = min(99.5, cfg.verification_percentile_threshold * 100.0 + 1.0)
    rows = [
        {
            "concept_id": concept_id,
            "prefix_text": f"complete prefix {concept_id}-{row_idx}",
            "prefix_source_shard": 0,
            "prefix_source_doc": concept_id * 100 + row_idx,
            "prefix_activating_seg_idx": row_idx,
            "verification_continuations": [
                f"continuation {concept_id}-{row_idx}-{pass_idx}"
                for pass_idx in range(n_passes)
            ],
            "verification_seeds": [
                10_000 + concept_id * 100 + row_idx * 10 + pass_idx
                for pass_idx in range(n_passes)
            ],
            "verification_percentiles": [pct for _ in range(n_passes)],
        }
        for concept_id in range(n_concepts)
        for row_idx in range(n_rows)
    ]
    schema = _si._schema_with_completed_negative_prefixes_metadata(
        cfg,
        candidate_pool_identity=pool_identity,
    )
    schema = _si._schema_with_completed_artifact_set_identities(
        schema,
        concepts=concepts,
        rejected_concepts=rejected_payload,
        concept_examples_qc_report=qc_report_payload,
        stage0c_fingerprint=_si._fingerprint_with_pool(
            cfg,
            candidate_pool_identity=pool_identity,
        ),
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, parquet_path)
    return concepts_path, rejected_path, parquet_path


def _valid_rejected_concept_record() -> dict[str, Any]:
    return {
        "candidate_index": 0,
        "feature_index": 99,
        "description": "fixture rejection",
        "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
        "n_verified_found": 0,
        "llm_judge_score": 1,
    }


def _valid_resume_negative_prefix_row() -> dict[str, Any]:
    return {
        "concept_id": 0,
        "prefix_text": "valid prefix",
        "prefix_source_shard": 0,
        "prefix_source_doc": 0,
        "prefix_activating_seg_idx": 0,
        "verification_continuations": ["continuation"],
        "verification_seeds": [123],
        "verification_percentiles": [80.0],
    }


def test_semantic_json_identity_is_key_order_and_format_independent():
    left = json.loads(
        '''
        {
          "z": [{"second": 2, "first": "café"}],
          "a": null
        }
        '''
    )
    right = {
        "a": None,
        "z": [{"first": "café", "second": 2}],
    }

    assert _si._semantic_json_identity(left) == (
        _si._semantic_json_identity(right)
    )
    candidate = _candidate(feature_index=7, description="café")
    assert _si.candidate_pool_identity([candidate]) == (
        _si._semantic_json_identity([asdict(candidate)])
    )


def test_resume_negative_prefixes_if_complete_requires_run_tag(
    stub_env, tmp_path,
):
    cfg = _cfg(tmp_path).replace(run_tag=None)

    with pytest.raises(AxBenchBaseError, match="cfg.run_tag must be set"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_resume_negative_prefixes_if_complete_returns_none_when_artifacts_missing(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)
    stats_reads = {"count": 0}

    def _unexpected_stats_read(path: Path) -> int:
        del path
        stats_reads["count"] += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    assert negative_prefixes.resume_negative_prefixes_if_complete(cfg) is None
    assert stats_reads["count"] == 0


# ---------------------------------------------------------------------------
# Clustered Stage 0c resume guard (PR #739 C4)
#
# ``_assert_no_partial_clustered_final_artifacts`` / clustered
# ``resume_negative_prefixes_if_complete`` -- rejects a torn clustered
# canonical artifact set before any stats-derived work (feature-statistics
# read, candidate-pool load). Ported from the oracle alongside
# ``negative_prefixes.py``'s own clustered port; C3 deliberately deferred
# this test because its production code (``_assert_no_partial_clustered_
# final_artifacts``) lives in ``negative_prefixes.py``, outside C3's
# ``_stage0c_internals.py`` + ``synth/qc.py`` scope.
# ---------------------------------------------------------------------------


def test_clustered_public_resume_rejects_partial_canonical_set_before_stats(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path, enable_concept_examples_qc=False).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    partial = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"diagnostic evidence")

    def _unexpected_stats_read(_path: Path) -> int:
        raise AssertionError("clustered partial state must fail before stats")

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match="partial canonical artifact set"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert partial.read_bytes() == b"diagnostic evidence"


# ---------------------------------------------------------------------------
# _canonical_cluster_candidate_union -- occurrence union/order (PR #739 C3)
# ---------------------------------------------------------------------------


def test_canonical_cluster_candidate_union_dedupes_by_location_and_orders_canonically():
    """Two cluster members can both rediscover the same source location.

    The union keeps exactly one row per ``(shard_idx, doc_idx, seg_idx)``
    (first-seen wins) and returns rows in canonical location order,
    independent of the member scan order that produced them.
    """
    dup_first_seen = _PrefixCandidate(
        shard_idx=0, doc_idx=5, seg_idx=2,
        prefix_text="first-seen", prefix_gemma_tokens=[1],
    )
    dup_second_seen = _PrefixCandidate(
        shard_idx=0, doc_idx=5, seg_idx=2,
        prefix_text="second-seen", prefix_gemma_tokens=[2],
    )
    later_shard = _PrefixCandidate(
        shard_idx=1, doc_idx=0, seg_idx=0,
        prefix_text="shard-1", prefix_gemma_tokens=[3],
    )
    earlier_doc_same_shard = _PrefixCandidate(
        shard_idx=0, doc_idx=1, seg_idx=9,
        prefix_text="shard-0-doc-1", prefix_gemma_tokens=[4],
    )

    result = _si._canonical_cluster_candidate_union(
        [dup_first_seen, later_shard, dup_second_seen, earlier_doc_same_shard],
        per_shard_limit=None,
    )

    # Canonical (shard_idx, doc_idx, seg_idx) order: (0,1,9) < (0,5,2) < (1,0,0).
    assert result == [earlier_doc_same_shard, dup_first_seen, later_shard]
    # The duplicate location keeps the FIRST-SEEN candidate, not the second.
    assert result[1].prefix_text == "first-seen"


def test_canonical_cluster_candidate_union_caps_only_within_shard():
    """The configured per-shard limit applies post-dedup, per shard independently."""
    shard0 = [
        _PrefixCandidate(
            shard_idx=0, doc_idx=i, seg_idx=0,
            prefix_text=f"s0-{i}", prefix_gemma_tokens=[i],
        )
        for i in range(3)
    ]
    shard1 = [
        _PrefixCandidate(
            shard_idx=1, doc_idx=i, seg_idx=0,
            prefix_text=f"s1-{i}", prefix_gemma_tokens=[i],
        )
        for i in range(3)
    ]

    result = _si._canonical_cluster_candidate_union(
        shard0 + shard1, per_shard_limit=2,
    )

    assert [c.shard_idx for c in result] == [0, 0, 1, 1]
    assert [c.doc_idx for c in result] == [0, 1, 0, 1]


def test_canonical_cluster_candidate_union_uncapped_when_limit_is_none():
    candidates = [
        _PrefixCandidate(
            shard_idx=0, doc_idx=i, seg_idx=0,
            prefix_text=f"c{i}", prefix_gemma_tokens=[i],
        )
        for i in range(5)
    ]

    result = _si._canonical_cluster_candidate_union(candidates, per_shard_limit=None)

    assert len(result) == 5


def test_canonical_cluster_candidate_union_rejects_negative_limit():
    with pytest.raises(AxBenchBaseError, match="per-shard limit"):
        _si._canonical_cluster_candidate_union([], per_shard_limit=-1)


@pytest.mark.parametrize(
    "missing_artifact",
    [
        "concepts_json",
        "rejected_concepts_json",
        "negative_mode_prefixes",
    ],
)
def test_public_resume_does_not_read_stats_when_any_final_artifact_is_missing(
    stub_env,
    tmp_path,
    monkeypatch,
    missing_artifact,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    getattr(output_paths, missing_artifact)().unlink()
    stats_reads = 0

    def _count_stats_reads(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _count_stats_reads,
    )

    assert negative_prefixes.resume_negative_prefixes_if_complete(cfg) is None
    assert stats_reads == 0


def test_resume_negative_prefixes_if_complete_reads_stats_and_delegates(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, max_concepts=1, n_verified_prefixes_per_concept_target=1)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    seen: dict[str, Any] = {}
    expected = Path("/tmp/already_done.parquet")

    def _fake_read_n_sae_features(path: Path) -> int:
        seen["stats_path"] = path
        return 321

    def _fake_resume(
        cfg_arg,
        *,
        n_sae_features: int,
        candidate_concepts_path: Path,
    ) -> Path:
        seen["cfg"] = cfg_arg
        seen["n_sae_features"] = n_sae_features
        seen["candidate_concepts_path"] = candidate_concepts_path
        return expected

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _fake_read_n_sae_features,
    )
    monkeypatch.setattr(
        negative_prefixes,
        "_resume_if_artifacts_complete",
        _fake_resume,
    )

    out = negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert out == expected
    assert seen["cfg"] == cfg
    assert seen["n_sae_features"] == 321
    assert seen["stats_path"] == Path(cfg.feature_statistics_path)
    assert (
        seen["candidate_concepts_path"]
        == paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    )


def test_resume_short_circuits_when_all_artifacts_present(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Complete artifacts skip expensive Stage 0c work."""
    cfg = _cfg(tmp_path, max_concepts=1, n_verified_prefixes_per_concept_target=3)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)

    work_done = {"verify_called": 0}

    def _unexpected_list_parquets(cfg_arg):
        del cfg_arg
        work_done["verify_called"] += 1
        return []

    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _unexpected_list_parquets,
    )

    candidates_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    out = prepare_negative_prefixes(
        cfg,
        candidates_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    assert out == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    assert work_done["verify_called"] == 0, "resume path must not enter the full Stage 0c work"


def test_stage_0c_writes_completed_artifact_identity_metadata(
    stub_env,
    tmp_path,
):
    cfg = _cfg(
        tmp_path,
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-16k",
        extraction_tokenizer_id="custom/stage0c-tokenizer",
    )
    stub_env["concepts"] = [
        _concept(
            concept_id=0,
            feature_index=0,
            sae_source=cfg.neuronpedia_source,
        )
    ]
    candidates_path = tmp_path / "candidate_concepts.json"
    candidates_path.write_text("[]")

    out = prepare_negative_prefixes(
        cfg, candidates_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )

    metadata = pq.read_schema(out).metadata or {}
    assert metadata[b"extraction_tokenizer_id"] == (
        cfg.extraction_tokenizer_id.encode("utf-8")
    )
    assert metadata[b"neuronpedia_model_id"] == b"gemma-3-4b"
    assert metadata[b"feature_extraction_model_id"] == b"gemma-3-4b-pt"
    assert metadata[b"neuronpedia_source"] == b"22-gemmascope-2-res-16k"
    expected_pool = [_candidate_from_concept(c) for c in stub_env["concepts"]]
    assert metadata[b"candidate_pool_identity"] == (
        _si.candidate_pool_identity(expected_pool).encode("utf-8")
    )
    expected_concepts = cast(list[Concept], stub_env["concepts"])
    assert metadata[b"concepts_identity"] == (
        _si.concepts_identity(expected_concepts).encode("utf-8")
    )
    rejected_payload = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_text(encoding="utf-8")
    )
    assert metadata[b"rejected_concepts_identity"] == (
        _si.rejected_concepts_identity(rejected_payload).encode("utf-8")
    )
    qc_report_payload = json.loads(
        paths.AxBenchBasePaths.from_config(cfg)
        .concept_examples_qc_report()
        .read_text(encoding="utf-8")
    )
    assert metadata[b"concept_examples_qc_report_identity"] == (
        _si.concept_examples_qc_report_identity(qc_report_payload).encode("utf-8")
    )
    expected_fingerprint = _si._fingerprint_with_pool(
        cfg,
        candidate_pool_identity=_si.candidate_pool_identity(expected_pool),
    )
    assert metadata[b"stage0c_fingerprint_identity"] == (
        _si.stage0c_fingerprint_identity(expected_fingerprint).encode("utf-8")
    )


def test_public_resume_accepts_semantically_reformatted_json_sidecars(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    rejected = _valid_rejected_concept_record()
    concepts_path, rejected_path, parquet_path = _write_complete_stage_0c_artifacts(
        cfg,
        rejected_records=[rejected],
    )
    concepts_payload = json.loads(concepts_path.read_text(encoding="utf-8"))
    rejected_payload = json.loads(rejected_path.read_text(encoding="utf-8"))
    qc_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    qc_payload = json.loads(qc_path.read_text(encoding="utf-8"))
    concepts_path.write_text(
        json.dumps(
            [dict(reversed(list(record.items()))) for record in concepts_payload],
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )
    rejected_path.write_text(
        json.dumps(
            [dict(reversed(list(record.items()))) for record in rejected_payload],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    qc_payload["per_feature"] = [
        dict(reversed(list(record.items())))
        for record in qc_payload["per_feature"]
    ]
    qc_path.write_text(
        json.dumps(
            dict(reversed(list(qc_payload.items()))),
            ensure_ascii=False,
            indent=3,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    assert negative_prefixes.resume_negative_prefixes_if_complete(cfg) == (
        parquet_path
    )


@pytest.mark.parametrize(
    ("sidecar", "metadata_key", "mutate"),
    [
        (
            "concepts",
            "concepts_identity",
            lambda records: [{**records[0], "description": "changed final concept"}],
        ),
        (
            "rejected",
            "rejected_concepts_identity",
            lambda records: [{**records[0], "llm_judge_score": 2}],
        ),
    ],
)
def test_public_resume_rejects_semantic_sidecar_mismatch_and_preserves_artifacts(
    stub_env,
    tmp_path,
    monkeypatch,
    sidecar,
    metadata_key,
    mutate,
):
    cfg = _cfg(tmp_path)
    concepts_path, rejected_path, parquet_path = _write_complete_stage_0c_artifacts(
        cfg,
        rejected_records=[_valid_rejected_concept_record()],
    )
    changed_path = concepts_path if sidecar == "concepts" else rejected_path
    changed_payload = mutate(
        json.loads(changed_path.read_text(encoding="utf-8"))
    )
    changed_path.write_text(json.dumps(changed_payload), encoding="utf-8")
    artifact_paths = (concepts_path, rejected_path, parquet_path)
    preserved = {path: path.read_bytes() for path in artifact_paths}
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match=metadata_key):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert {path: path.read_bytes() for path in artifact_paths} == preserved


def test_public_resume_rejects_semantic_qc_report_mismatch(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    concepts_path, rejected_path, parquet_path = _write_complete_stage_0c_artifacts(cfg)
    qc_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    qc_payload = json.loads(qc_path.read_text(encoding="utf-8"))
    qc_payload["per_feature"][0]["n_examples"] = 2
    qc_path.write_text(json.dumps(qc_payload), encoding="utf-8")
    artifact_paths = (concepts_path, rejected_path, qc_path, parquet_path)
    preserved = {path: path.read_bytes() for path in artifact_paths}
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="concept_examples_qc_report_identity",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert {path: path.read_bytes() for path in artifact_paths} == preserved


def test_public_resume_rejects_internally_inconsistent_qc_report(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    qc_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    qc_payload = json.loads(qc_path.read_text(encoding="utf-8"))
    qc_payload["n_accepted"] = 0
    qc_path.write_text(json.dumps(qc_payload), encoding="utf-8")
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="summary counts disagree"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


@pytest.mark.parametrize(
    "qc_feature_indices",
    [
        [99, 100, 101],
        [100, 99],
    ],
    ids=["future-row", "noncanonical-order"],
)
def test_completed_sidecars_require_exact_ordered_terminal_qc_rows(
    tmp_path,
    qc_feature_indices,
):
    cfg = _cfg(tmp_path, enable_concept_examples_qc=True, max_concepts=1)
    accepted = Concept(
        concept_id=0,
        feature_index=100,
        layer=8,
        sae_source=cfg.neuronpedia_source,
        description="fixture concept 0",
        corpus_count=50,
        description_source="neuronpedia",
    )
    candidates = [
        CandidateConcept(
            feature_index=99,
            layer=8,
            sae_source=cfg.neuronpedia_source,
            description="fixture rejection",
            corpus_count=50,
            description_source="neuronpedia",
        ),
        _candidate_from_concept(accepted),
        CandidateConcept(
            feature_index=101,
            layer=8,
            sae_source=cfg.neuronpedia_source,
            description="future candidate",
            corpus_count=50,
            description_source="neuronpedia",
        ),
    ]
    rows_by_feature = {
        99: {
            "feature_index": 99,
            "outcome": QcOutcome.ACCEPTED.value,
            "score": 1,
        },
        100: {
            "feature_index": 100,
            "outcome": QcOutcome.ACCEPTED.value,
            "score": 2,
        },
        101: {
            "feature_index": 101,
            "outcome": QcOutcome.ACCEPTED.value,
            "score": 2,
        },
    }
    report = {
        "per_feature": [rows_by_feature[fid] for fid in qc_feature_indices]
    }

    with pytest.raises(
        AxBenchBaseError,
        match="exactly one row.*canonical candidate order",
    ):
        negative_prefixes._validate_completed_sidecars_against_candidate_pool(
            candidate_concepts=candidates,
            concepts=[accepted],
            rejected_concepts=[_valid_rejected_concept_record()],
            concept_examples_qc_report=report,
            cfg=cfg,
        )


def test_public_resume_rejects_full_stage0c_fingerprint_mismatch(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    concepts_path, rejected_path, parquet_path = _write_complete_stage_0c_artifacts(
        cfg
    )
    changed_cfg = cfg.replace(
        generation=replace(
            cfg.generation,
            temperature=cfg.generation.temperature + 0.1,
        )
    )
    artifact_paths = (concepts_path, rejected_path, parquet_path)
    preserved = {path: path.read_bytes() for path in artifact_paths}
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="stage0c_fingerprint_identity"):
        negative_prefixes.resume_negative_prefixes_if_complete(changed_cfg)

    assert {path: path.read_bytes() for path in artifact_paths} == preserved


@pytest.mark.parametrize(
    ("stored_mode", "current_mode"),
    [("shared", "repeated"), ("repeated", "shared")],
)
def test_public_resume_rejects_cross_prefill_mode_completed_artifact(
    stub_env,
    tmp_path,
    monkeypatch,
    stored_mode: str,
    current_mode: str,
) -> None:
    stored_cfg = _cfg(tmp_path).replace(
        verification_prefill_mode=stored_mode,
    )
    concepts_path, rejected_path, parquet_path = _write_complete_stage_0c_artifacts(stored_cfg)
    current_cfg = stored_cfg.replace(
        verification_prefill_mode=current_mode,
    )
    artifact_paths = (concepts_path, rejected_path, parquet_path)
    preserved = {path: path.read_bytes() for path in artifact_paths}
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="stage0c_fingerprint_identity",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(current_cfg)

    assert {path: path.read_bytes() for path in artifact_paths} == preserved


@pytest.mark.parametrize(
    "metadata_key",
    _si._NEG_PREFIXES_COMPLETED_ARTIFACT_IDENTITY_KEYS,
)
def test_public_resume_rejects_missing_completed_artifact_identity_metadata(
    stub_env,
    tmp_path,
    monkeypatch,
    metadata_key,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(metadata_key.encode("utf-8"))
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    preserved = parquet_path.read_bytes()
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match=rf"missing '{metadata_key}'"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert parquet_path.read_bytes() == preserved


@pytest.mark.parametrize(
    "metadata_key",
    _si._NEG_PREFIXES_COMPLETED_ARTIFACT_IDENTITY_KEYS,
)
@pytest.mark.parametrize(
    ("metadata_value", "match"),
    [
        (b"not-a-sha256", "64-character lowercase SHA-256"),
        (b"F" * 64, "64-character lowercase SHA-256"),
        (b"\xff", "non-UTF-8"),
    ],
)
def test_public_resume_rejects_malformed_completed_artifact_identity_metadata(
    stub_env,
    tmp_path,
    monkeypatch,
    metadata_key,
    metadata_value,
    match,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[metadata_key.encode("utf-8")] = metadata_value
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    preserved = parquet_path.read_bytes()
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match=match):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert parquet_path.read_bytes() == preserved


@pytest.mark.parametrize("pool_change", ["membership", "order", "description"])
def test_public_resume_rejects_completed_candidate_pool_mismatch(
    stub_env,
    tmp_path,
    monkeypatch,
    pool_change,
):
    """A rerun of Stage 0a under a pinned run_tag invalidates final Stage 0c."""
    cfg = _cfg(tmp_path)
    rejected_records = (
        [_valid_rejected_concept_record()] if pool_change == "order" else None
    )
    _write_complete_stage_0c_artifacts(
        cfg,
        rejected_records=rejected_records,
    )
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    preserved = {
        path: path.read_bytes()
        for path in (
            paths.AxBenchBasePaths.from_config(cfg).concepts_json(),
            paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json(),
            parquet_path,
        )
    }

    candidate_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    if pool_change == "membership":
        candidate_payload.append(
            {
                "feature_index": 101,
                "layer": 8,
                "sae_source": cfg.neuronpedia_source,
                "description": "new Stage 0a candidate",
                "corpus_count": 40,
                "description_source": "neuronpedia",
            }
        )
    elif pool_change == "order":
        candidate_payload.reverse()
    else:
        candidate_payload[0]["description"] = "changed Stage 0a description"
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"candidate_pool_identity=.*current ordered candidate_concepts",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    for path, expected_bytes in preserved.items():
        assert path.read_bytes() == expected_bytes


def test_public_resume_rejects_missing_completed_candidate_pool_metadata(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"candidate_pool_identity")
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match="missing 'candidate_pool_identity'"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert b"candidate_pool_identity" not in (
        pq.read_schema(parquet_path).metadata or {}
    )


@pytest.mark.parametrize(
    ("metadata_value", "match"),
    [
        (b"not-a-sha256", "64-character lowercase SHA-256"),
        (b"F" * 64, "64-character lowercase SHA-256"),
        (b"\xff", "non-UTF-8"),
    ],
)
def test_public_resume_rejects_malformed_candidate_pool_metadata(
    stub_env,
    tmp_path,
    monkeypatch,
    metadata_value,
    match,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"candidate_pool_identity"] = metadata_value
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match=match):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert (pq.read_schema(parquet_path).metadata or {})[
        b"candidate_pool_identity"
    ] == metadata_value


def test_prepare_resume_validates_the_explicit_candidate_pool_path(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Direct callers cannot bypass pool identity via a non-canonical path."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    explicit_path = tmp_path / "explicit_candidate_concepts.json"
    explicit_path.write_text("[]", encoding="utf-8")
    stale_pool = [_candidate_from_concept(c) for c in stub_env["concepts"]]
    changed_pool = [*stale_pool, _candidate(feature_index=1)]
    loaded_paths: list[Path] = []
    work_calls = 0

    def _load_explicit(path, *, n_sae_features):
        del n_sae_features
        loaded_paths.append(Path(path))
        return changed_pool

    def _unexpected_stage0c_work(cfg_arg):
        nonlocal work_calls
        del cfg_arg
        work_calls += 1
        return []

    monkeypatch.setattr(
        negative_prefixes,
        "load_and_validate_candidate_concepts",
        _load_explicit,
    )
    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _unexpected_stage0c_work,
    )

    with pytest.raises(AxBenchBaseError, match="candidate_pool_identity"):
        prepare_negative_prefixes(
            cfg,
            explicit_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert loaded_paths == [explicit_path]
    assert work_calls == 0


@pytest.mark.parametrize("candidate_state", ["missing", "malformed"])
def test_public_resume_rejects_unverifiable_live_candidate_pool(
    stub_env,
    tmp_path,
    monkeypatch,
    candidate_state,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    candidate_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    if candidate_state == "missing":
        candidate_path.unlink()
    else:
        candidate_path.write_text("{malformed", encoding="utf-8")
    monkeypatch.setattr(
        negative_prefixes,
        "load_and_validate_candidate_concepts",
        validate_mod.load_and_validate_candidate_concepts,
    )
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="completed artifact set is invalid.*candidate",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


@pytest.mark.parametrize(
    ("field", "artifact_value", "current_value"),
    [
        ("neuronpedia_model_id", "gemma-2-2b", "gemma-2-9b"),
        (
            "feature_extraction_model_id",
            "gemma-3-4b-pt",
            "gemma-3-12b-pt",
        ),
        (
            "neuronpedia_source",
            "22-gemmascope-2-res-16k",
            "34-gemmascope-2-res-16k",
        ),
    ],
)
def test_public_resume_rejects_completed_extractor_identity_mismatch(
    stub_env,
    tmp_path,
    monkeypatch,
    field,
    artifact_value,
    current_value,
):
    """Every extractor-identity component invalidates a completed resume."""
    current_cfg = _cfg(tmp_path).replace(**{field: current_value})
    _write_complete_stage_0c_artifacts(current_cfg)
    parquet_path = test_paths.default_paths(current_cfg.run_tag).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[field.encode("utf-8")] = artifact_value.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=rf"{field}.*{re.escape(artifact_value)}.*{re.escape(current_value)}",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(current_cfg)


@pytest.mark.parametrize(
    "missing_key",
    [
        "neuronpedia_model_id",
        "feature_extraction_model_id",
        "neuronpedia_source",
    ],
)
def test_public_resume_rejects_missing_completed_extractor_metadata_key(
    stub_env,
    tmp_path,
    monkeypatch,
    missing_key,
):
    """Legacy or partially-tagged final parquets are ambiguous and stale."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(missing_key.encode("utf-8"))
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match=rf"missing '{missing_key}'"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_public_resume_rejects_non_utf8_extractor_metadata(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"feature_extraction_model_id"] = b"\xff"
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="non-UTF-8"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_public_resume_rejects_legacy_tokenizer_only_metadata(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """A pre-identity artifact must not fall into the tokenizer rerun path."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    pq.write_table(
        table.replace_schema_metadata(
            {
                b"extraction_tokenizer_id": (
                    cfg.extraction_tokenizer_id.encode("utf-8")
                )
            }
        ),
        parquet_path,
    )
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="missing 'neuronpedia_model_id'"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_public_resume_rejects_mixed_identity_concepts_json(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    concepts_path, _, _ = _write_complete_stage_0c_artifacts(cfg)
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    concepts[0]["sae_source"] = "22-gemmascope-2-res-16k"
    concepts_path.write_text(json.dumps(concepts), encoding="utf-8")
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="concepts_identity"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


@pytest.mark.parametrize(
    "sidecar_problem",
    ["missing_qc_report", "malformed_concepts", "malformed_rejected"],
)
def test_extractor_identity_validation_precedes_completed_sidecar_checks(
    stub_env,
    tmp_path,
    monkeypatch,
    sidecar_problem,
):
    """No missing/corrupt sidecar may bypass the fail-loud identity gate."""
    cfg = _cfg(
        tmp_path,
        enable_concept_examples_qc=(sidecar_problem == "missing_qc_report"),
        feature_extraction_model_id="current/model",
    )
    concepts_path, rejected_path, parquet_path = (
        _write_complete_stage_0c_artifacts(
            cfg,
            write_qc_report=(sidecar_problem != "missing_qc_report"),
        )
    )
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"feature_extraction_model_id"] = b"stale/model"
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    if sidecar_problem == "malformed_concepts":
        concepts_path.write_text("{malformed", encoding="utf-8")
    elif sidecar_problem == "malformed_rejected":
        rejected_path.write_text("{malformed", encoding="utf-8")
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"feature_extraction_model_id='stale/model'.*'current/model'",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)
    assert stats_reads == 0


@pytest.mark.parametrize("explicit_on_artifact", [False, True])
def test_completed_resume_normalizes_effective_feature_extraction_model_id(
    stub_env,
    tmp_path,
    monkeypatch,
    explicit_on_artifact,
):
    """``None`` and an explicit fallback-equivalent model id are identical."""
    model_id = "gemma-3-4b-pt"
    artifact_cfg = _cfg(
        tmp_path,
        neuronpedia_model_id=model_id,
        feature_extraction_model_id=(model_id if explicit_on_artifact else None),
    )
    _write_complete_stage_0c_artifacts(artifact_cfg)
    current_cfg = artifact_cfg.replace(
        feature_extraction_model_id=(None if explicit_on_artifact else model_id),
    )
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    assert (
        negative_prefixes.resume_negative_prefixes_if_complete(
            current_cfg,
        )
        == test_paths.default_paths(current_cfg.run_tag).negative_mode_prefixes()
    )


def test_completed_resume_accepts_distinct_public_and_local_model_ids(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Gemma-3's Neuronpedia namespace and local PT model may differ."""
    cfg = _cfg(
        tmp_path,
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-16k",
    )
    _write_complete_stage_0c_artifacts(cfg)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    assert (
        negative_prefixes.resume_negative_prefixes_if_complete(
            cfg,
        )
        == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    )


@pytest.mark.parametrize(
    ("field", "stale_value", "current_value"),
    [
        ("neuronpedia_model_id", "gemma-2-9b", "gemma-2-2b"),
        (
            "feature_extraction_model_id",
            "gemma-3-12b-pt",
            "gemma-3-4b-pt",
        ),
        (
            "neuronpedia_source",
            "34-gemmascope-2-res-16k",
            "22-gemmascope-2-res-16k",
        ),
    ],
)
def test_prepare_fails_before_work_on_completed_extractor_identity_mismatch(
    stub_env,
    tmp_path,
    monkeypatch,
    field,
    stale_value,
    current_value,
):
    """A Stage 0c-only rerun cannot safely consume old Stage 0a candidates."""
    stale_cfg = _cfg(tmp_path).replace(**{field: stale_value})
    _write_complete_stage_0c_artifacts(stale_cfg)
    cfg = stale_cfg.replace(**{field: current_value})
    list_calls = 0
    original = _si._list_parquets_for_feature_dir

    def _counting_list_parquets(cfg_arg):
        nonlocal list_calls
        list_calls += 1
        return original(cfg_arg)

    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _counting_list_parquets,
    )
    candidates_path = tmp_path / "candidate_concepts.json"
    candidates_path.write_text("[]")

    with pytest.raises(AxBenchBaseError, match=field):
        prepare_negative_prefixes(
            cfg,
            candidates_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert list_calls == 0


def test_prepare_rejects_candidate_pool_from_different_sae_source(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Deleting final outputs must not let stale Stage 0a rows slip through."""
    cfg = _cfg(tmp_path, neuronpedia_source="22-gemmascope-2-res-16k")
    # ``stub_env`` intentionally retains its default GemmaScope-v1 candidate.
    list_calls = 0

    def _unexpected_list_parquets(cfg_arg):
        nonlocal list_calls
        list_calls += 1
        return []

    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _unexpected_list_parquets,
    )
    candidates_path = tmp_path / "candidate_concepts.json"
    candidates_path.write_text("[]")

    with pytest.raises(AxBenchBaseError, match="different extractor identity"):
        prepare_negative_prefixes(
            cfg,
            candidates_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert list_calls == 0


def test_prepare_rejects_runtime_sae_layer_mismatch_before_shard_work(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    list_calls = 0

    def _unexpected_list_parquets(cfg_arg):
        nonlocal list_calls
        del cfg_arg
        list_calls += 1
        return []

    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _unexpected_list_parquets,
    )
    candidates_path = tmp_path / "candidate_concepts.json"
    candidates_path.write_text("[]")

    with pytest.raises(AxBenchBaseError, match=r"SAE layer\(s\).*sae_layer=9"):
        prepare_negative_prefixes(
            cfg,
            candidates_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(sae_layer=9),
        )

    assert list_calls == 0


def test_prepare_rejects_runtime_sae_layer_mismatch_on_completed_resume(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        max_concepts=1,
        n_verified_prefixes_per_concept_target=1,
    )
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    list_calls = 0

    def _unexpected_list_parquets(cfg_arg):
        nonlocal list_calls
        del cfg_arg
        list_calls += 1
        return []

    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _unexpected_list_parquets,
    )

    with pytest.raises(AxBenchBaseError, match=r"SAE layer\(s\).*sae_layer=9"):
        prepare_negative_prefixes(
            cfg,
            paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json(),
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(sae_layer=9),
        )

    assert list_calls == 0


def test_public_resume_rejects_unreadable_completed_parquet(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    corrupt_bytes = b"not parquet"
    parquet_path.write_bytes(corrupt_bytes)
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match="unreadable"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert parquet_path.read_bytes() == corrupt_bytes


@pytest.mark.parametrize("identity_state", ["missing", "mismatched"])
def test_public_resume_rejects_wrong_schema_before_any_rerun(
    stub_env,
    tmp_path,
    monkeypatch,
    identity_state,
):
    """A wrong column schema cannot bypass completed provenance checks."""
    cfg = _cfg(tmp_path, feature_extraction_model_id="current/model")
    _write_complete_stage_0c_artifacts(cfg)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    metadata = dict(pq.read_schema(parquet_path).metadata or {})
    if identity_state == "missing":
        metadata.pop(b"feature_extraction_model_id")
    else:
        metadata[b"feature_extraction_model_id"] = b"stale/model"
    wrong_schema_table = pa.Table.from_pylist(
        [{"concept_id": 0}],
    ).replace_schema_metadata(metadata)
    pq.write_table(wrong_schema_table, parquet_path)
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match="has schema"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert pq.read_schema(parquet_path).metadata == metadata


def test_completed_resume_ignores_stale_partial_checkpoint_leftovers(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Canonical completion wins if cleanup crashed after final writes."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg)
    assert cfg.run_tag is not None
    partial_json = test_paths.default_paths(cfg.run_tag).stage0c_partial_rank_json(0)
    partial_parquet = test_paths.default_paths(cfg.run_tag).stage0c_partial_rank_parquet(0)
    partial_json.write_text("{malformed")
    partial_parquet.write_bytes(b"stale partial")
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    assert (
        negative_prefixes.resume_negative_prefixes_if_complete(
            cfg,
        )
        == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    )


def test_resume_fails_loudly_when_tokenizer_metadata_missing(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"extraction_tokenizer_id")
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)

    work_done = {"list_parquets": 0}
    orig = _si._list_parquets_for_feature_dir

    def _counting_list_parquets(cfg_arg):
        work_done["list_parquets"] += 1
        return orig(cfg_arg)

    monkeypatch.setattr(
        _si, "_list_parquets_for_feature_dir",
        _counting_list_parquets,
    )

    candidates_path = tmp_path / "candidate_concepts.json"
    candidates_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="extraction_tokenizer_id"):
        prepare_negative_prefixes(
            cfg, candidates_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert work_done["list_parquets"] == 0
    assert b"extraction_tokenizer_id" not in (
        pq.read_schema(parquet_path).metadata or {}
    )


def test_resume_fails_loudly_when_tokenizer_metadata_mismatches(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, extraction_tokenizer_id="custom/current-tokenizer")
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"extraction_tokenizer_id"] = b"custom/stale-tokenizer"
    pq.write_table(
        table.replace_schema_metadata(metadata),
        parquet_path,
    )

    work_done = {"list_parquets": 0}
    orig = _si._list_parquets_for_feature_dir

    def _counting_list_parquets(cfg_arg):
        work_done["list_parquets"] += 1
        return orig(cfg_arg)

    monkeypatch.setattr(
        _si, "_list_parquets_for_feature_dir",
        _counting_list_parquets,
    )

    candidates_path = tmp_path / "candidate_concepts.json"
    candidates_path.write_text("[]")
    with pytest.raises(AxBenchBaseError, match="custom/stale-tokenizer"):
        prepare_negative_prefixes(
            cfg, candidates_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert work_done["list_parquets"] == 0
    assert (pq.read_schema(parquet_path).metadata or {})[
        b"extraction_tokenizer_id"
    ] == b"custom/stale-tokenizer"


def test_public_resume_helper_rejects_missing_tokenizer_metadata(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"extraction_tokenizer_id")
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="extraction_tokenizer_id"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_public_resume_helper_rejects_mismatched_tokenizer_metadata(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, extraction_tokenizer_id="custom/current-tokenizer")
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"extraction_tokenizer_id"] = b"custom/stale-tokenizer"
    pq.write_table(
        table.replace_schema_metadata(metadata),
        parquet_path,
    )
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="custom/stale-tokenizer"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_public_resume_helper_wraps_non_utf8_tokenizer_metadata(
    stub_env,
    tmp_path,
):
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"extraction_tokenizer_id"] = b"\xff"
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)

    with pytest.raises(
        AxBenchBaseError,
        match="non-UTF-8.*extraction_tokenizer_id",
    ):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_resume_fails_loudly_when_qc_enabled_and_qc_report_missing(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, enable_concept_examples_qc=True)
    _write_complete_stage_0c_artifacts(
        cfg,
        n_concepts=1,
        write_qc_report=False,
    )
    assert not paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report().exists()

    work_done = {"list_parquets": 0}
    orig = _si._list_parquets_for_feature_dir

    def _counting_list_parquets(cfg_arg):
        work_done["list_parquets"] += 1
        return orig(cfg_arg)

    monkeypatch.setattr(
        _si,
        "_list_parquets_for_feature_dir",
        _counting_list_parquets,
    )

    candidates_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    with pytest.raises(AxBenchBaseError, match="examples QC.*missing"):
        prepare_negative_prefixes(
            cfg,
            candidates_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert work_done["list_parquets"] == 0
    assert not paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report().exists()


def test_resume_fails_loudly_on_stale_qc_report_when_qc_disabled(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        enable_concept_examples_qc=False,
        max_concepts=1,
    )
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    report_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    report_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="QC is disabled.*stale"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert report_path.read_text(encoding="utf-8") == "{}"


def test_stage0c_all_qc_rejects_do_not_load_candidate_shards(
    stub_env, tmp_path, monkeypatch,
):
    cfg = _cfg(tmp_path, max_concepts=1)
    stub_env["concepts"] = [
        _concept(concept_id=0, feature_index=0),
        _concept(concept_id=1, feature_index=1),
    ]
    stub_env["qc_client"] = _PatternQCClient({0: 0, 1: 0})
    feature_loads: list[Path] = []
    text_loads: list[Path] = []
    orig_load_features = _si.load_features
    orig_load_text = _si.read_parquet_texts

    def _spy_load_features(path: Path) -> dict[str, Any]:
        feature_loads.append(path)
        return orig_load_features(path)

    def _spy_load_text(path: Path) -> list[str]:
        text_loads.append(path)
        return orig_load_text(path)

    monkeypatch.setattr(_si, "load_features", _spy_load_features)
    monkeypatch.setattr(_si, "read_parquet_texts", _spy_load_text)

    concepts_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    with pytest.raises(
        AxBenchBaseError,
        match="Stage 0c exhausted candidate_concepts.json",
    ):
        prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )

    assert feature_loads == []
    assert text_loads == []


@pytest.mark.parametrize(
    "remove_artifact",
    [
        "concepts_json",
        "rejected_concepts_json",
        "negative_mode_prefixes",
    ],
)
def test_resume_runs_when_any_required_artifact_is_missing(
    stub_env, tmp_path, remove_artifact,
):
    """Any partial artifact set triggers a full re-run."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    getattr(output_paths, remove_artifact)().unlink()

    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    out = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    table = pq.read_table(out)
    assert table.num_rows == cfg.n_verified_prefixes_per_concept_target


def test_resume_fails_loudly_when_concepts_count_mismatches_max_concepts(
    stub_env, tmp_path, monkeypatch,
):
    """A stale concepts count must not silently short-circuit."""
    cfg = _cfg(tmp_path, max_concepts=3)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)

    concepts_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()

    work_done = {"verify_called": 0}
    orig = _si._list_parquets_for_feature_dir

    def _counting_list_parquets(cfg_arg):
        work_done["verify_called"] += 1
        return orig(cfg_arg)

    monkeypatch.setattr(
        _si, "_list_parquets_for_feature_dir",
        _counting_list_parquets,
    )
    with pytest.raises(AxBenchBaseError, match="cfg.max_concepts=3"):
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    assert work_done["verify_called"] == 0, (
        "mismatched completed artifacts must fail before Stage 0c work"
    )


def test_resume_fails_loudly_when_concepts_json_malformed(
    stub_env, tmp_path, monkeypatch,
):
    """Corrupt completed concepts are preserved for diagnosis."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    malformed = "{ not json"
    concepts_path.write_text(malformed)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="completed artifact set is invalid"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert concepts_path.read_text() == malformed


def test_resume_fails_loudly_when_concepts_json_not_list(
    stub_env, tmp_path, monkeypatch,
):
    """A wrong completed JSON shape is an error, not an auto-repair signal."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    invalid = '{"not": "a list"}'
    concepts_path.write_text(invalid)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="must be a JSON array"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert concepts_path.read_text() == invalid


def test_resume_fails_loudly_when_concepts_json_schema_invalid(
    stub_env, tmp_path, monkeypatch,
):
    """A list with the right length must still satisfy the row contract."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    invalid = json.dumps([{"concept_id": 0}])
    concepts_path.write_text(invalid)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="completed artifact set is invalid"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert concepts_path.read_text() == invalid


def test_resume_fails_loudly_when_rejected_json_invalid(
    stub_env, tmp_path, monkeypatch,
):
    """Corrupt completed audit artifacts are preserved for diagnosis."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()
    invalid = '{"not": "a list"}'
    rejected_path.write_text(invalid)
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="must be a JSON array"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert rejected_path.read_text() == invalid


def test_resume_fails_loudly_when_parquet_schema_invalid(
    stub_env, tmp_path, monkeypatch,
):
    """A wrong completed parquet schema must not trigger an overwrite."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    bad_table = pa.Table.from_pylist([{"concept_id": 0}])
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    pq.write_table(bad_table, parquet_path)
    stats_reads = 0

    def _unexpected_stats_read(path):
        nonlocal stats_reads
        stats_reads += 1
        return 16384

    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        _unexpected_stats_read,
    )

    with pytest.raises(AxBenchBaseError, match="has schema"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert stats_reads == 0
    assert pq.read_table(parquet_path).schema == bad_table.schema


def test_resume_fails_loudly_when_parquet_row_count_mismatches_target(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Changing or truncating completed prefix rows fails loudly."""
    cfg = _cfg(tmp_path, n_verified_prefixes_per_concept_target=2)
    _write_complete_stage_0c_artifacts(
        cfg,
        n_concepts=1,
        rows_per_concept=1,
    )
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="has 1 rows.*expected 2"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert pq.read_table(parquet_path).num_rows == 1


def test_resume_fails_loudly_when_parquet_percentile_below_current_threshold(
    stub_env, tmp_path, monkeypatch,
):
    """A completed artifact incompatible with a new threshold is preserved."""
    cfg = _cfg(tmp_path)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    original_table = pq.read_table(parquet_path)
    rows = original_table.to_pylist()
    for row in rows:
        n_passes = len(row["verification_percentiles"])
        row["verification_percentiles"] = [0.0 for _ in range(n_passes)]
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=original_table.schema,
        ),
        parquet_path,
    )
    monkeypatch.setattr(
        negative_prefixes,
        "read_n_sae_features_from_stats",
        lambda path: 16384,
    )

    with pytest.raises(AxBenchBaseError, match="percentile 0.0"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    preserved_rows = pq.read_table(parquet_path).to_pylist()
    assert all(
        set(row["verification_percentiles"]) == {0.0}
        for row in preserved_rows
    )


def test_resume_short_circuit_emits_info_log(stub_env, tmp_path):
    """The resume path explains why Stage 0c skipped expensive work."""
    cfg = _cfg(tmp_path, max_concepts=1, n_verified_prefixes_per_concept_target=3)
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)

    collected, handler_id = _capture_info_logs()
    candidates_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    try:
        prepare_negative_prefixes(
            cfg,
            candidates_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    finally:
        loguru_logger.remove(handler_id)

    resume_lines = [m for m in collected if "Stage 0c resume" in m]
    assert resume_lines, f"no Stage 0c resume log emitted; got: {collected}"
    assert any("skipping verification" in m for m in resume_lines), (
        "resume log should clearly state we're skipping Stage 0c work"
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda record: [0],
            "must be a JSON object",
        ),
        (
            lambda record: [{k: v for k, v in record.items() if k != "reason"}],
            "expected",
        ),
        (
            lambda record: [{**record, "candidate_index": True}],
            "candidate_index must be a non-negative int",
        ),
        (
            lambda record: [{**record, "feature_index": True}],
            "feature_index must be a non-negative int",
        ),
        (
            lambda record: [{**record, "description": "   "}],
            "description is empty or not a string",
        ),
        (
            lambda record: [{**record, "reason": "bogus"}],
            "reason has unknown value",
        ),
        (
            lambda record: [{**record, "n_verified_found": -1}],
            "n_verified_found must be null or a non-negative int",
        ),
        (
            lambda record: [{**record, "llm_judge_score": "bad"}],
            "llm_judge_score must be null or an int",
        ),
    ],
)
def test_validate_rejected_concepts_json_rejects_invalid_records(
    tmp_path, mutate, match,
):
    path = tmp_path / "rejected_concepts.json"
    path.write_text(json.dumps(mutate(_valid_rejected_concept_record())))

    with pytest.raises(AxBenchBaseError, match=match):
        negative_prefixes._validate_rejected_concepts_json(path)


@pytest.mark.parametrize(
    ("mutate", "cfg_overrides", "match"),
    [
        (
            lambda row: {**row, "concept_id": True},
            {},
            "concept_id must be an int",
        ),
        (
            lambda row: {**row, "concept_id": 9},
            {},
            "does not correspond to a final concept",
        ),
        (
            lambda row: {**row, "prefix_text": ""},
            {},
            "prefix_text is empty or not a string",
        ),
        (
            lambda row: {**row, "prefix_text": "x" * 2049},
            {},
            "outside current bounds",
        ),
        (
            lambda row: {**row, "prefix_source_doc": -1},
            {},
            "must be non-negative",
        ),
        (
            lambda row: {**row, "verification_seeds": 123},
            {},
            "verification fields must all be lists",
        ),
        (
            lambda row: {**row, "verification_seeds": [123, 124]},
            {},
            "verification list lengths differ",
        ),
        (
            lambda row: {
                **row,
                "verification_continuations": [],
                "verification_seeds": [],
                "verification_percentiles": [],
            },
            {},
            "below cfg.min_consistency_passes",
        ),
        (
            lambda row: {
                **row,
                "verification_continuations": ["a", "b"],
                "verification_seeds": [123, 124],
                "verification_percentiles": [80.0, 81.0],
            },
            {},
            "above cfg.n_verification_seeds",
        ),
        (
            lambda row: {**row, "verification_continuations": [123]},
            {},
            "must contain only strings",
        ),
        (
            lambda row: {**row, "verification_seeds": [True]},
            {},
            "verification_seeds must contain only ints",
        ),
        (
            lambda row: {
                **row,
                "verification_continuations": ["a", "b"],
                "verification_seeds": [123, 123],
                "verification_percentiles": [80.0, 81.0],
            },
            {"n_verification_seeds": 2},
            "contains duplicates",
        ),
        (
            lambda row: {**row, "verification_percentiles": [float("nan")]},
            {},
            "does not exceed current threshold",
        ),
    ],
)
def test_validate_negative_prefixes_parquet_rejects_invalid_rows(
    tmp_path, monkeypatch, mutate, cfg_overrides, match,
):
    cfg_kwargs = {
        "n_verified_prefixes_per_concept_target": 1,
        "n_verification_seeds": 1,
        "min_consistency_passes": 1,
    }
    cfg_kwargs.update(cfg_overrides)
    cfg = _cfg(tmp_path, **cfg_kwargs)
    path = tmp_path / "negative_mode_prefixes.parquet"
    row = mutate(_valid_resume_negative_prefix_row())

    class _FakeParquetTable:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.schema = (
                _si._schema_with_completed_negative_prefixes_metadata(
                    cfg,
                    candidate_pool_identity="0" * 64,
                )
            )
            self._rows = rows

        def to_pylist(self) -> list[dict[str, Any]]:
            return list(self._rows)

    monkeypatch.setattr(
        negative_prefixes.pq,
        "read_table",
        lambda table_path: _FakeParquetTable([row]),
    )

    with pytest.raises(AxBenchBaseError, match=match):
        negative_prefixes._validate_negative_prefixes_parquet(
            path,
            concepts=[_concept(concept_id=0, feature_index=100)],
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Stage 0c completion-heartbeat ETA: project to N accepts, not pool walk
#
# Earlier, the "Completed candidate" log line computed ETA as
# ``avg_per_candidate * (n_candidates - visited)`` — i.e. "time to walk
# the entire pool". For the dev-tier smoke run with ~16k candidates and
# a ~500s/candidate verify rate, the ETA printed as ~94 days after the
# first candidate. The correct projection is "time to hit max_concepts
# accepts at the observed accept rate".
# ---------------------------------------------------------------------------


def test_eta_zero_when_target_already_reached():
    """``accepts_still_needed == 0`` → no more work → eta = 0.0."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=5, n_accepted=5,
        max_concepts=5, n_candidates=16_000,
    )
    assert got == 0.0


def test_eta_zero_when_target_exceeded():
    """Defensive: over-accept (shouldn't happen in practice) returns 0.0."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=10, n_accepted=7,
        max_concepts=5, n_candidates=16_000,
    )
    assert got == 0.0


def test_eta_zero_when_pool_exhausted():
    """Even if we still need accepts, an empty pool means no more work."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=16_000, n_accepted=3,
        max_concepts=5, n_candidates=16_000,
    )
    assert got == 0.0


def test_eta_all_accept_extrapolates_to_single_candidate_per_accept():
    """Every candidate accepted maps remaining accepts 1:1 to candidates."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=5, n_accepted=5,
        max_concepts=10, n_candidates=1000,
    )
    assert got == pytest.approx(100.0)


def test_eta_fractional_accept_rate_projects_correctly():
    """10 visited, 2 accepted, 3 accepts needed → 15 more candidates."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=10, n_accepted=2,
        max_concepts=5, n_candidates=1000,
    )
    assert got == pytest.approx(150.0)


def test_eta_caps_at_remaining_pool_size():
    """Accept-rate extrapolation cannot exceed the remaining pool."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=100, n_accepted=1,
        max_concepts=5, n_candidates=150,
    )
    assert got == pytest.approx(50.0)


def test_eta_zero_accept_rate_bounds_by_pool():
    """No accepts yet falls back to a finite pool-walk upper bound."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=100.0, n_visited=100, n_accepted=0,
        max_concepts=5, n_candidates=1000,
    )
    assert got == pytest.approx(900.0)


def test_eta_realistic_dev_tier_scenario_is_reasonable():
    """Dev-tier shape: 16k pool, 500s/candidate, accept rate around 20%."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=2500.0, n_visited=5, n_accepted=1,
        max_concepts=5, n_candidates=16_000,
    )
    assert got == pytest.approx(10_000.0)
    assert got < 100_000


def test_eta_regression_no_94_day_nonsense():
    """Regression guard for the specific dev-tier 94-day ETA."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=497.0,
        n_visited=1, n_accepted=1,
        max_concepts=5, n_candidates=16_373,
    )
    old_formula = 497.0 * (16_373 - 1)
    assert got == pytest.approx(1988.0)
    assert got < 86_400
    assert old_formula > 8_000_000
    assert got < old_formula / 1000


def test_eta_n_visited_zero_is_zero_defensively():
    """Guard against division-by-zero."""
    got = _si._project_completion_eta_seconds(
        stage_elapsed=0.0, n_visited=0, n_accepted=0,
        max_concepts=5, n_candidates=100,
    )
    assert got == 0.0


def test_log_candidate_completion_emits_bounded_eta(stub_env, tmp_path):
    """Integration: candidate heartbeat log must print a finite sub-day ETA."""
    cfg = _cfg(tmp_path)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")
    collected, handler_id = _capture_info_logs()
    try:
        prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    finally:
        loguru_logger.remove(handler_id)

    completion_lines = [
        m for m in collected
        if m.startswith("[Stage 0c] Completed candidate")
    ]
    assert completion_lines, f"no completion log; got: {collected}"
    for line in completion_lines:
        match = re.search(r"eta remaining ≈ (\d+)s(?:,|\))", line)
        assert match is not None
        eta_seconds = int(match.group(1))
        assert eta_seconds < 86_400, (
            f"eta in {line!r} should be < 1 day; got {eta_seconds}s"
        )


def test_log_candidate_completion_reports_after_cleanup_process_memory(
    monkeypatch,
):
    snapshot = ProcessMemorySnapshot(
        rss_kb=100,
        rss_anon_kb=70,
        swap_kb=5,
        anonymous_total_kb=75,
        peak_rss_kb=120,
    )
    counters: list[tuple[str, int, dict[str, int]]] = []
    monkeypatch.setattr(_si, "process_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        _si,
        "record_counter",
        lambda name, value, **attrs: counters.append((name, value, attrs)),
    )
    monkeypatch.setattr(_si.time, "monotonic", lambda: 10.0)

    collected, handler_id = _capture_info_logs()
    try:
        _si._log_candidate_completion(
            stage_started_at=5.0,
            concept_idx=2,
            n_candidates=10,
            accepted_concepts=[_concept(concept_id=0)],
            rejected_concepts=[],
            all_rows=[],
            max_concepts=2,
        )
    finally:
        loguru_logger.remove(handler_id)

    completion = next(
        message
        for message in collected
        if message.startswith("[Stage 0c] Completed candidate")
    )
    assert (
        "memory after cleanup: current_rss_kb=100, "
        "anonymous_rss_kb=70, swap_kb=5, anonymous_total_kb=75, "
        "peak_rss_kb=120"
    ) in completion
    assert counters == [
        (
            "stage0c.memory.current_rss_kb",
            100,
            {"candidate_index": 2, "candidates_visited": 3},
        ),
        (
            "stage0c.memory.anonymous_rss_kb",
            70,
            {"candidate_index": 2, "candidates_visited": 3},
        ),
        (
            "stage0c.memory.swap_kb",
            5,
            {"candidate_index": 2, "candidates_visited": 3},
        ),
        (
            "stage0c.memory.anonymous_total_kb",
            75,
            {"candidate_index": 2, "candidates_visited": 3},
        ),
        (
            "stage0c.memory.peak_rss_kb",
            120,
            {"candidate_index": 2, "candidates_visited": 3},
        ),
    ]


def test_log_candidate_completion_ignores_unavailable_process_memory(
    monkeypatch,
):
    counters: list[tuple[str, int]] = []
    monkeypatch.setattr(_si, "process_memory_snapshot", lambda: None)
    monkeypatch.setattr(
        _si,
        "record_counter",
        lambda name, value, **attrs: counters.append((name, value)),
    )
    monkeypatch.setattr(_si.time, "monotonic", lambda: 10.0)

    collected, handler_id = _capture_info_logs()
    try:
        _si._log_candidate_completion(
            stage_started_at=5.0,
            concept_idx=0,
            n_candidates=1,
            accepted_concepts=[],
            rejected_concepts=[],
            all_rows=[],
            max_concepts=1,
        )
    finally:
        loguru_logger.remove(handler_id)

    completion = next(
        message
        for message in collected
        if message.startswith("[Stage 0c] Completed candidate")
    )
    assert "memory after cleanup" not in completion
    assert counters == []


@pytest.mark.parametrize(
    ("snapshot", "expected_fragment", "expected_counters"),
    [
        (
            ProcessMemorySnapshot(
                rss_kb=100,
                rss_anon_kb=None,
                swap_kb=5,
                anonymous_total_kb=None,
                peak_rss_kb=120,
            ),
            (
                "memory after cleanup: current_rss_kb=100, "
                "swap_kb=5, peak_rss_kb=120"
            ),
            [
                "stage0c.memory.current_rss_kb",
                "stage0c.memory.swap_kb",
                "stage0c.memory.peak_rss_kb",
            ],
        ),
        (
            ProcessMemorySnapshot(
                rss_kb=None,
                rss_anon_kb=None,
                swap_kb=None,
                anonymous_total_kb=None,
                peak_rss_kb=None,
            ),
            None,
            [],
        ),
    ],
)
def test_log_candidate_completion_reports_only_available_memory_fields(
    monkeypatch,
    snapshot,
    expected_fragment,
    expected_counters,
):
    counters: list[str] = []
    monkeypatch.setattr(_si, "process_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        _si,
        "record_counter",
        lambda name, value, **attrs: counters.append(name),
    )
    monkeypatch.setattr(_si.time, "monotonic", lambda: 10.0)

    collected, handler_id = _capture_info_logs()
    try:
        _si._log_candidate_completion(
            stage_started_at=5.0,
            concept_idx=0,
            n_candidates=1,
            accepted_concepts=[],
            rejected_concepts=[],
            all_rows=[],
            max_concepts=1,
        )
    finally:
        loguru_logger.remove(handler_id)

    completion = next(
        message
        for message in collected
        if message.startswith("[Stage 0c] Completed candidate")
    )
    if expected_fragment is None:
        assert "memory after cleanup" not in completion
    else:
        assert expected_fragment in completion
    assert counters == expected_counters


# ---------------------------------------------------------------------------
# Stage 0c user-requested scenario tests (Task 9)
# ---------------------------------------------------------------------------


class _KillSentinel(Exception):
    """Used by tests to interrupt Stage 0c verification mid-loop."""


def _wrap_verify_candidate_to_kill_after(
    monkeypatch, n_completed: int,
) -> dict[str, int]:
    """Wrap _verify_candidate so it raises _KillSentinel on call N+1.

    Returns a dict {"calls": k} the caller can inspect after the test.
    The wrapper delegates to the original _verify_candidate for the
    first n_completed calls (so those candidates complete normally and
    get committed), then raises _KillSentinel on call n_completed + 1
    to simulate a crash *during* candidate verification (i.e., before
    that candidate's commit).

    Callers MUST restore _verify_candidate manually after the kill by
    re-patching it with the real function (do NOT call monkeypatch.undo(),
    which also tears down the stub_env fixture patches).
    """
    state = {"calls": 0}
    original = _si._verify_candidate

    def _wrapper(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] > n_completed:
            raise _KillSentinel(f"kill after {n_completed} verify_candidate calls")
        return original(*args, **kwargs)

    monkeypatch.setattr(_si, "_verify_candidate", _wrapper)
    return state


def _restore_verify_candidate(monkeypatch) -> None:
    """Re-patch _verify_candidate back to the real function.

    Called after a kill-and-resume cycle to undo only the killing wrapper
    without tearing down the broader stub_env fixture patches (which
    monkeypatch.undo() would incorrectly tear down too).
    """
    monkeypatch.setattr(_si, "_verify_candidate", _REAL_VERIFY_CANDIDATE)


# Store the unpatched _verify_candidate at import time so _restore_verify_candidate
# can always recover it regardless of how many wrapper layers have been applied.
_REAL_VERIFY_CANDIDATE = _si._verify_candidate


def test_resume_produces_identical_final_artifacts(
    stub_env, tmp_path, monkeypatch,
):
    """User scenario 2: uninterrupted vs. resumed run produces byte-identical
    concepts.json + rejected_concepts.json + negative_mode_prefixes.parquet."""
    base_concepts = [
        _concept(concept_id=0, feature_index=0),
        _concept(concept_id=1, feature_index=1),
        _concept(concept_id=2, feature_index=2),
    ]
    stub_env["percentile_pair"] = (10.0, 80.0)
    stub_env["concepts"] = base_concepts

    # Uninterrupted run in a separate tmp dir
    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted_root.mkdir()
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(uninterrupted_root))
    cfg_u = _cfg(
        tmp_path / "uninterrupted",
        run_tag="run_uninterrupted",
        n_verified_prefixes_per_concept_target=2,
        max_concepts=3,
    )
    concepts_path = tmp_path / "uninterrupted" / "candidate_concepts.json"
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_path.write_text("[]")
    out_u = prepare_negative_prefixes(
        cfg_u, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    rows_u = pq.read_table(out_u).to_pylist()
    concepts_u = json.loads(test_paths.default_paths(cfg_u.run_tag).concepts_json().read_text())
    rejected_u = json.loads(test_paths.default_paths(cfg_u.run_tag).rejected_concepts_json().read_text())

    # Resumed run in another tmp dir
    resumed_root = tmp_path / "resumed"
    resumed_root.mkdir()
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(resumed_root))
    stub_env["generator"].calls.clear()
    cfg_r = _cfg(
        tmp_path / "resumed",
        run_tag="run_resumed",
        n_verified_prefixes_per_concept_target=2,
        max_concepts=3,
    )
    concepts_path_r = tmp_path / "resumed" / "candidate_concepts.json"
    concepts_path_r.parent.mkdir(parents=True, exist_ok=True)
    concepts_path_r.write_text("[]")

    # First attempt: kill after 1 candidate
    _wrap_verify_candidate_to_kill_after(monkeypatch, n_completed=1)
    with pytest.raises(_KillSentinel):
        prepare_negative_prefixes(
            cfg_r, concepts_path_r,
            base_model=_base_model(), base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    # Restore only the kill wrapper; leave stub_env fixture patches intact.
    _restore_verify_candidate(monkeypatch)

    # Second attempt: complete normally
    stub_env["generator"].calls.clear()
    out_r = prepare_negative_prefixes(
        cfg_r, concepts_path_r,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    rows_r = pq.read_table(out_r).to_pylist()
    concepts_r = json.loads(test_paths.default_paths(cfg_r.run_tag).concepts_json().read_text())
    rejected_r = json.loads(test_paths.default_paths(cfg_r.run_tag).rejected_concepts_json().read_text())

    # Final artifacts byte-identical (modulo run_tag in each, which differs
    # by design across the two runs). Compare the concept-level data.
    assert [c["concept_id"] for c in concepts_u] == [c["concept_id"] for c in concepts_r]
    assert [c["feature_index"] for c in concepts_u] == [c["feature_index"] for c in concepts_r]
    assert rejected_u == rejected_r
    assert len(rows_u) == len(rows_r)
    for u_row, r_row in zip(rows_u, rows_r, strict=True):
        for col in (
            "concept_id", "prefix_text", "prefix_source_shard",
            "prefix_source_doc", "prefix_activating_seg_idx",
            "verification_continuations", "verification_seeds",
            "verification_percentiles",
        ):
            assert u_row[col] == r_row[col], (
                f"Row mismatch on column {col}: {u_row[col]} != {r_row[col]}"
            )

    # Schema metadata equality: catches tokenizer_id or other key-value
    # metadata divergence between the partial-write path and the fresh-
    # write path.
    table_u = pq.read_table(out_u)
    table_r = pq.read_table(out_r)
    assert table_u.schema.metadata == table_r.schema.metadata, (
        f"Parquet schema metadata differs:\n"
        f"  uninterrupted: {table_u.schema.metadata}\n"
        f"  resumed:       {table_r.schema.metadata}"
    )


def test_complete_artifact_resume_still_works(
    stub_env, tmp_path,
):
    """Existing complete Stage 0c artifacts still skip regeneration."""
    stub_env["percentile_pair"] = (10.0, 80.0)
    stub_env["concepts"] = [_concept(concept_id=0, feature_index=0)]
    cfg = _cfg(tmp_path, n_verified_prefixes_per_concept_target=2)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    # Run once to write all final artifacts
    out_first = prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=_base_model(), base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    # All three final artifacts are present.
    assert out_first == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    assert paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    assert paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()

    # Re-invoke: complete-artifact resume should fire (stage skipped entirely)
    initial_gen_calls = len(stub_env["generator"].calls)
    out_second = prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    assert out_second == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    # The generator must NOT have been called again
    assert len(stub_env["generator"].calls) == initial_gen_calls
















































# ---------------------------------------------------------------------------
# Task 10: Boundary / correctness tests (spec §7.3, tests 6-13)
# ---------------------------------------------------------------------------


def test_resume_preserves_concept_id_density(
    stub_env, tmp_path, monkeypatch,
):
    """Spec test 11: final concept_ids stay 0..N-1 with no gaps after resume,
    even when the per-resume-run mix includes verification rejections."""
    stub_env["concepts"] = [
        _concept(concept_id=i, feature_index=i) for i in range(6)
    ]
    cfg = _cfg(
        tmp_path,
        n_verified_prefixes_per_concept_target=2,
        max_concepts=3,
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    _wrap_verify_candidate_to_kill_after(monkeypatch, n_completed=4)
    stub_env["percentile_pair"] = (10.0, 80.0)  # all pass
    with pytest.raises(_KillSentinel):
        prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=_base_model(),
            base_sae=SimpleNamespace(),
            target_model_meta=_stub_meta(),
        )
    _restore_verify_candidate(monkeypatch)

    stub_env["generator"].calls.clear()
    prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    final_concepts = json.loads(paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text())
    concept_ids = [c["concept_id"] for c in final_concepts]
    assert concept_ids == list(range(len(concept_ids)))


def test_prepare_negative_prefixes_starts_fresh(
    stub_env, tmp_path,
):
    """An empty run directory starts a fresh Stage 0c run."""
    stub_env["percentile_pair"] = (10.0, 80.0)
    stub_env["concepts"] = [_concept(concept_id=0, feature_index=0)]
    cfg = _cfg(tmp_path, n_verified_prefixes_per_concept_target=2)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]")

    out_path = prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_stub_meta(),
    )
    assert out_path == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()










# ---------------------------------------------------------------------------
# Fingerprint parametric tests (Task 11)
# ---------------------------------------------------------------------------

# Each tuple: (field_name, cfg_replace_kwargs_for_fresh, cfg_replace_kwargs_for_changed)
















# ---------------------------------------------------------------------------
# Clustered completed-resume live-statistics re-validation (PR #739
# FU-D-stage0c fix batch, Codex cross-review of C4, Bug 1 HIGH -- third
# named path: negative_prefixes.py:493/505/638,
# ``resume_negative_prefixes_if_complete`` / ``_resume_if_artifacts_
# complete``). This entrypoint deliberately never resolves a full
# ``ClusterFeatureContract`` (it must decide whether to reuse a completed
# run without loading a checkpoint or SAE), so it re-validates the live
# statistics against the already-committed ``feature_contract.json``
# marker instead of an in-memory contract.
# ---------------------------------------------------------------------------


def _stub_cluster_fingerprint_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the C1-owned Stage 0a identity chain with a canned, internally
    consistent fingerprint -- mirrors stage0c_runner test module's own
    helper of the same name -- so these tests can exercise the clustered
    branch of ``_compute_checkpoint_fingerprint`` (folded into the
    completed parquet's ``stage0c_fingerprint`` metadata by ``_write_
    complete_stage_0c_artifacts``) without installing a full Stage 0a
    candidate-provenance sidecar. Unrelated to (and does not interfere
    with) the live-statistics re-check under test here.
    """
    monkeypatch.setattr(
        _si,
        "clustered_stage0c_fingerprint_fields",
        lambda cfg: {
            "feature_contract_fingerprint": "a" * 64,
            "candidate_cluster_provenance_sha256": "b" * 64,
            "cluster_concept_projection_version": "v1",
        },
    )


def _stats_feature_dir(tmp_path: Path, *, name: str = "clustered-features") -> Path:
    feature_dir = build_feature_dir(
        tmp_path / name,
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=False,
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir


def _write_versioned_stats(
    feature_dir: Path,
    *,
    n_sae_features: int = 16_384,
    feature_ids: tuple[int, ...] = (0, 1, 2),
    max_mags: dict[int, float],
) -> Path:
    """Write a real, self-consistent statistics file under ``feature_dir``,
    overwriting the same final path when called twice (``atomic_
    safetensors_save`` is an unconditional ``os.replace``) -- how these
    tests swap the on-disk bytes at a fixed path.
    """
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags=dict(max_mags),
        absent=[i for i in range(n_sae_features) if i not in feature_ids],
    )
    stats = load_statistics(statistics_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, checksum)
    return statistics_path


def _write_marker_with_statistics(run_tag: str, *, pinned_statistics: dict[str, Any]) -> None:
    """Hand-write a self-consistent ``feature_contract.json`` root marker
    whose ``semantic.statistics`` block is ``pinned_statistics``, mirroring
    ``test_cluster_artifacts.py``'s helper of the same name.
    """
    semantic = {
        "statistics": pinned_statistics,
        "concept_source": {
            "mode": "statistics_domain",
            "domain": [0],
            "source_to_representative": [[0, 0]],
        },
    }
    document = {
        "schema_version": 1,
        "feature_contract_fingerprint": _sha256_json(semantic),
        "semantic": semantic,
        "provenance": {},
    }
    marker = test_paths.default_paths(run_tag).feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(document), encoding="utf-8")


def _clustered_cfg_for_completed_resume(
    tmp_path: Path, *, run_tag: str, feature_dir: Path, statistics_path: Path,
) -> Any:
    return _cfg(
        tmp_path,
        run_tag=run_tag,
        max_concepts=1,
        n_verified_prefixes_per_concept_target=1,
        feature_dir=str(feature_dir),
        feature_statistics_path=str(statistics_path),
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )


def test_resume_negative_prefixes_if_complete_clustered_noop_when_statistics_unchanged(
    stub_env, tmp_path, monkeypatch,
):
    """Normal-path proof: a completed clustered run whose statistics file
    is untouched must still short-circuit resume exactly as before -- no
    false raise."""
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    run_tag = "clustered-completed-unchanged"
    cfg = _clustered_cfg_for_completed_resume(
        tmp_path, run_tag=run_tag, feature_dir=feature_dir, statistics_path=statistics_path,
    )
    _write_marker_with_statistics(run_tag, pinned_statistics=pinned.canonical_dict())
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)

    result = negative_prefixes.resume_negative_prefixes_if_complete(cfg)

    assert result == test_paths.default_paths(run_tag).negative_mode_prefixes()


def test_resume_negative_prefixes_if_complete_clustered_raises_on_swapped_statistics(
    stub_env, tmp_path, monkeypatch,
):
    """RED/GREEN core proof for Bug 1's third named path (completed
    resume): the committed feature_contract.json marker pins statistics A;
    a completed Stage 0c artifact set exists; the statistics file is then
    swapped in place to B (same width/domain, different bytes). The
    lightweight, checkpoint/SAE-free public resume check must still catch
    it and refuse to silently reuse the completed set.
    """
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    run_tag = "clustered-completed-swapped"
    cfg = _clustered_cfg_for_completed_resume(
        tmp_path, run_tag=run_tag, feature_dir=feature_dir, statistics_path=statistics_path,
    )
    _write_marker_with_statistics(run_tag, pinned_statistics=pinned_a.canonical_dict())
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)

    # Swap in place: same path, same n_sae_features/feature_ids domain,
    # different max_mag bytes.
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        negative_prefixes.resume_negative_prefixes_if_complete(cfg)


def test_resume_if_artifacts_complete_clustered_raises_on_swapped_statistics(
    stub_env, tmp_path, monkeypatch,
):
    """Same RED/GREEN proof, but through ``prepare_negative_prefixes``'s
    own internal resume path (``_resume_if_artifacts_complete`` called
    directly, not via the standalone CLI helper) -- proves the fix covers
    BOTH callers of the shared choke point, not just the public one.
    """
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    run_tag = "clustered-internal-resume-swapped"
    cfg = _clustered_cfg_for_completed_resume(
        tmp_path, run_tag=run_tag, feature_dir=feature_dir, statistics_path=statistics_path,
    )
    _write_marker_with_statistics(run_tag, pinned_statistics=pinned_a.canonical_dict())
    _write_complete_stage_0c_artifacts(cfg, n_concepts=1)
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        negative_prefixes._resume_if_artifacts_complete(
            cfg,
            n_sae_features=16_384,
            candidate_concepts_path=test_paths.default_paths(run_tag).candidate_concepts_json(),
        )
