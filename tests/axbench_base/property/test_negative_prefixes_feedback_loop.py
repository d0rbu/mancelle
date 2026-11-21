"""Property tests for Stage 0c candidate-pool verification semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch as t
from hypothesis import given
from hypothesis import strategies as st

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
from steering.eval.axbench_base._stage0c_internals import (
    _find_candidate_prefixes,
    _VerifiedNegativePrefix,
)
from steering.eval.axbench_base.concept_examples_qc import QCFeatureScore, QCReport
from steering.eval.axbench_base.negative_prefixes import (
    prepare_negative_prefixes,
)
from steering.eval.axbench_base.types import (
    CandidateConcept,
    Concept,
    QcOutcome,
    TargetModelMetadata,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config


def _concept(i: int) -> Concept:
    return Concept(
        concept_id=i,
        feature_index=i,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {i}",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _candidate(i: int) -> CandidateConcept:
    return CandidateConcept(
        feature_index=i,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {i}",
        corpus_count=100,
        description_source="neuronpedia",
    )


class _MaskQCClient:
    def __init__(self, accept_bits: list[bool]) -> None:
        self.accept_bits = accept_bits
        self.calls: list[tuple[int, str]] = []

    def filter(
        self,
        candidates: list[tuple[int, str]],
        target_surviving: int | None = None,
    ) -> tuple[list[tuple[int, str]], QCReport]:
        # Honor target_surviving: the real ConceptExamplesQC.filter
        # breaks the inner loop once `len(survivors) >= target_surviving`.
        # The QcBuffer (PR-2 Task 10/11) relies on this to keep the
        # "lazy LLM" property — without it, batched fetches would
        # process every candidate in a batch even after the cap is met.
        survivors: list[tuple[int, str]] = []
        per_feature: list[QCFeatureScore] = []
        score_histogram: dict[int, int] = {}
        for fid, desc in candidates:
            self.calls.append((fid, desc))
            score = 2 if self.accept_bits[fid] else 0
            score_histogram[score] = score_histogram.get(score, 0) + 1
            if score >= 1:
                survivors.append((fid, desc))
                outcome = QcOutcome.ACCEPTED
            else:
                outcome = QcOutcome.REJECTED
            per_feature.append(
                QCFeatureScore(
                    feature_index=fid,
                    description=desc,
                    outcome=outcome,
                    score=score,
                    n_examples=1,
                    error_message=None,
                )
            )
            if (
                target_surviving is not None
                and len(survivors) >= target_surviving
            ):
                break
        n_accepted = len(survivors)
        return survivors, QCReport(
            n_total=len(candidates),
            n_accepted=n_accepted,
            n_rejected=len(candidates) - n_accepted,
            n_fetch_errors=0,
            n_judge_errors=0,
            n_empty_examples=0,
            score_histogram=score_histogram,
            per_feature=per_feature,
            judge_min_score=1,
            n_processed=len(candidates),
        )


def _meta() -> TargetModelMetadata:
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


def _row(concept_id: int) -> _VerifiedNegativePrefix:
    return _VerifiedNegativePrefix(
        concept_id=concept_id,
        prefix_text=f"prefix {concept_id}",
        prefix_source_shard=0,
        prefix_source_doc=concept_id,
        prefix_activating_seg_idx=concept_id,
        verification_continuations=[f"continuation {concept_id}"],
        verification_seeds=[10_000 + concept_id],
        verification_percentiles=[90.0],
    )


@st.composite
def _viable_stage0c_outcomes(draw) -> list[tuple[bool, bool]]:
    """Generate LLM/verification decisions with at least 5 final accepts."""
    pool_size = draw(st.integers(min_value=5, max_value=16))
    mandatory_accepts = set(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=pool_size - 1),
                min_size=5,
                max_size=5,
                unique=True,
            )
        )
    )
    outcomes: list[tuple[bool, bool]] = []
    for i in range(pool_size):
        if i in mandatory_accepts:
            outcomes.append((True, True))
        else:
            outcomes.append(
                draw(
                    st.sampled_from([
                        (False, False),
                        (False, True),
                        (True, False),
                        (True, True),
                    ])
                )
            )
    return outcomes


@given(lengths=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=25))
def test_candidate_scan_never_returns_prompt_over_budget(
    tmp_path, monkeypatch, lengths: list[int],
) -> None:
    max_prompt_tokens = 50
    cfg = make_valid_config(
        run_tag="scan_property",
        output_root=None,
        verification_length=14,
        min_prefix_chars=1,
        max_prefix_chars=2048,
    )

    def _fake_load_features(path: Path) -> dict[str, t.Tensor]:
        del path
        n = len(lengths)
        return {
            "doc_ids": t.arange(n, dtype=t.int32),
            "offsets": t.arange(n + 1, dtype=t.int64),
            "feature_ids": t.zeros(n, dtype=t.int32),
            "segment_char_ends": t.full((n,), 32, dtype=t.int32),
        }

    texts = [f"doc_{i}" for i in range(len(lengths))]

    class _LengthTokenizer:
        bos_token_id = 0

        def encode(
            self,
            text: str,
            add_special_tokens: bool = False,
        ) -> list[int]:
            idx = int(text.split("_")[1])
            total_len = lengths[idx]
            body = list(range(1, total_len))
            if add_special_tokens:
                return [self.bos_token_id] + body
            return body

    monkeypatch.setattr(_si, "load_features", _fake_load_features)
    monkeypatch.setattr(_si, "read_parquet_texts", lambda path: texts)
    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda tokenizer_id=None: _LengthTokenizer(),
    )

    candidates = _find_candidate_prefixes(
        cfg,
        shard_idx=0,
        feature_shard_path=Path("unused_features"),
        parquet_path=Path("unused_parquet"),
        f_c=0,
        base_context_window=max_prompt_tokens + cfg.verification_length,
    )
    assert candidates
    assert all(
        len(candidate.prefix_gemma_tokens) <= max_prompt_tokens
        for candidate in candidates
    )


@given(outcomes=_viable_stage0c_outcomes())
def test_stage_0c_lazy_llm_and_verification_accounting_property(
    tmp_path,
    monkeypatch,
    outcomes: list[tuple[bool, bool]],
) -> None:
    max_concepts = 5
    pool_size = len(outcomes)
    llm_accept = [outcome[0] for outcome in outcomes]
    verification_pass = [outcome[1] for outcome in outcomes]
    combined_accept = [
        llm_accept[i] and verification_pass[i] for i in range(pool_size)
    ]

    candidates = [_candidate(i) for i in range(pool_size)]
    qc_client = _MaskQCClient(llm_accept)
    cfg = make_valid_config(
        run_tag="stage0c_property",
        output_root=None,
        device="cpu",
        max_concepts=max_concepts,
        min_consistency_passes=1,
        n_verified_prefixes_per_concept_target=1,
        verification_length=8,
        # The synthetic ``_row()`` fixtures below carry short (<100 char)
        # ``prefix_text``. Stage 0c's merge-table validation enforces
        # ``[cfg.min_prefix_chars, cfg.max_prefix_chars]``, so relax the
        # floor here (mirrors ``test_candidate_scan_never_returns_prompt_
        # over_budget``) — this test exercises LLM/verification accounting,
        # not prefix-length selection.
        min_prefix_chars=1,
        # PR-2: this property test asserts the LEGACY lazy-LLM contract
        # (LLM-QC fires once per candidate as the verification loop
        # reaches it). The QcBuffer's batched-fetch mode would
        # legitimately pre-fetch entire batches upfront, which is a
        # different (still bounded) contract. Pin qc_batch_size=1 +
        # qc_pipeline=False so this test verifies the per-candidate
        # legacy semantics; QcBuffer's batched semantics are covered
        # separately by tests/axbench_base/unit/test_qc_buffer*.py.
        qc_batch_size=1,
        qc_pipeline=False,
    )
    # Hypothesis reuses ``tmp_path`` across examples (the fixture is
    # function-scoped and ``HealthCheck.function_scoped_fixture`` is
    # suppressed for this module). Stage 0c's idempotent-resume check
    # would otherwise short-circuit on stale artifacts from a prior
    # example. Scrub before each invocation so every example runs fresh.
    for stale in (
        paths.AxBenchBasePaths.from_config(cfg).concepts_json(),
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json(),
        paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes(),
        # Per-rank Stage 0c partial checkpoints live at a fixed
        # ``run_tag``-keyed path (not under ``tmp_path``). A partial left by
        # a prior example carries that example's ``candidate_pool_identity``
        # and would trip the current example's resume fingerprint check, so
        # scrub it alongside the JSON artifacts above.
        test_paths.default_paths(cfg.run_tag).stage0c_partial_rank_json(0),
        test_paths.default_paths(cfg.run_tag).stage0c_partial_rank_parquet(0),
    ):
        stale.unlink(missing_ok=True)

    def _fake_verify(
        cfg_arg: Any,
        concept: Concept,
        *args: Any,
        **kwargs: Any,
    ) -> list[_VerifiedNegativePrefix]:
        del cfg_arg, args, kwargs
        if verification_pass[concept.feature_index]:
            return [_row(concept.concept_id)]
        return []

    monkeypatch.setattr(extraction, "_infer_n_sae_features", lambda sae: 16384)
    monkeypatch.setattr(
        negative_prefixes,
        "load_and_validate_candidate_concepts",
        lambda path, n_sae_features: candidates,
        raising=False,
    )
    monkeypatch.setattr(
        stage0c_runner,
        "load_and_validate_candidate_concepts",
        lambda path, n_sae_features: candidates,
    )
    monkeypatch.setattr(
        concept_list,
        "_build_concept_examples_qc_client",
        lambda cfg_arg, response_cache=None: qc_client,
    )
    monkeypatch.setattr(_si, "_list_parquets_for_feature_dir", lambda cfg_arg: [])
    monkeypatch.setattr(stage0c_runner, "glob_features", lambda feature_dir: [])
    monkeypatch.setattr(
        extraction,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: t.zeros(16384, 99),
    )
    monkeypatch.setattr(_si, "_verify_concept", _fake_verify)

    candidate_path = tmp_path / "candidate_concepts.json"
    candidate_path.write_text("[]", encoding="utf-8")

    prepare_negative_prefixes(
        cfg,
        candidate_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_meta(),
    )
    first_concepts_bytes = paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_bytes()
    first_rejected_bytes = (
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_bytes()
    )
    first_llm_call_count = len(qc_client.calls)

    prepare_negative_prefixes(
        cfg,
        candidate_path,
        base_model=_base_model(),
        base_sae=SimpleNamespace(),
        target_model_meta=_meta(),
    )
    second_concepts_bytes = paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_bytes()
    second_rejected_bytes = (
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().read_bytes()
    )

    assert first_concepts_bytes == second_concepts_bytes
    assert first_rejected_bytes == second_rejected_bytes

    accepted_indices: list[int] = []
    for i, accepted in enumerate(combined_accept):
        if accepted:
            accepted_indices.append(i)
            if len(accepted_indices) == max_concepts:
                break
    fifth_accept_idx = accepted_indices[-1]

    final_concepts = json.loads(first_concepts_bytes)
    rejected = json.loads(first_rejected_bytes)
    assert [c["feature_index"] for c in final_concepts] == accepted_indices
    assert [c["concept_id"] for c in final_concepts] == list(range(max_concepts))
    assert len(final_concepts) == max_concepts
    assert first_llm_call_count <= fifth_accept_idx + 1

    accounted = [c["feature_index"] for c in final_concepts] + [
        r["feature_index"] for r in rejected
    ]
    assert sorted(accounted) == list(range(fifth_accept_idx + 1))
    assert len(accounted) == len(set(accounted))
