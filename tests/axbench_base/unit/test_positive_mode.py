"""Unit tests for synthesize_positive_mode (Task 16)."""

from __future__ import annotations

import json
import math
import random
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import yaml

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import SynthConfig
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.seed_contract import (
    _stage1b_seed_rows_identity,
    seed_rows_to_tensors,
)
from steering.eval.axbench_base.synth import (
    continuation_protocol,
    positive_checkpoint,
    positive_mode,
    response_parser,
)
from steering.eval.axbench_base.synth.concept_yield import (
    concept_yield_report_is_committed,
)
from steering.eval.axbench_base.synth.positive_mode import (
    _infer_n_sae_features,
    _validate_seed_feature_ids_sorted,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    TargetModelMetadata,
)
from steering.eval.cluster_feature_contract import ClusterFeatureTarget
from tests.axbench_base._loguru_capture import capture_loguru_info
from tests.axbench_base.positive_mode_reference import (
    handle_positive_attempt_result,
)
from tests.axbench_base.stage1_protocol_helpers import (
    TEST_STAGE1_TEMPLATE,
    frame_stage1_response,
)
from tests.axbench_base.unit.test_config import make_valid_config


def _test_seed_rows_sha256(seeds_train: list[SeedRow]) -> str:
    tensors = seed_rows_to_tensors(seeds_train)
    return _stage1b_seed_rows_identity(tensors, row_count=len(seeds_train))


def _draw_next_candidate(
    source: positive_mode._PositiveCandidateSource,
) -> positive_mode._SelectedPositiveSeed | None:
    selected = source.prepare_next()
    if selected is not None:
        source.commit_prepared(selected)
    return selected


def synthesize_positive_mode(
    cfg: Any,
    concepts: list[Concept],
    seeds_train: list[SeedRow],
    *args: Any,
    **kwargs: Any,
) -> Path:
    """Test adapter for the production-required precomputed seed identity."""

    kwargs.setdefault("seed_rows_sha256", _test_seed_rows_sha256(seeds_train))
    return positive_mode.synthesize_positive_mode(
        cfg,
        concepts,
        seeds_train,
        *args,
        **kwargs,
    )


def _rows_byte_equal(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
) -> bool:
    """NaN-aware equality for emitted training rows.

    Plain ``==`` fails on the avoid rows because they carry ``float('nan')``
    in the QC percentile fields (``nan != nan``). Treat two NaNs as equal.
    """
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b, strict=True):
        if row_a.keys() != row_b.keys():
            return False
        for key in row_a:
            va, vb = row_a[key], row_b[key]
            if (
                isinstance(va, float)
                and isinstance(vb, float)
                and math.isnan(va)
                and math.isnan(vb)
            ):
                continue
            if va != vb:
                return False
    return True


def _stub_meta(sae_layer: int = 8) -> TargetModelMetadata:
    """Stub TargetModelMetadata for tests that don't care about encoder fields."""
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


def _per_concept_positives(
    cfg: Any,
    concept: Concept,
    seeds_train: list[SeedRow],
    base_model: Any,
    base_sae: Any,
    llm_client: Any,
    percentile_table: Any,
    target_model_meta: TargetModelMetadata,
    prefix_cache: Any | None = None,
    cluster_target: ClusterFeatureTarget | None = None,
    effective_prefixes: Any | None = None,
) -> list[dict[str, Any]]:
    """Exercise one concept through the real production tracker and engine."""
    if effective_prefixes is None:
        effective_prefixes = positive_mode._resolve_stage1b_prefixes(cfg, seeds_train)
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds_train,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=target_model_meta.sae_layer,
        prefix_cache=prefix_cache,
        cluster_target=cluster_target,
        effective_prefixes=effective_prefixes,
    )
    return positive_mode._run_positive_target_refill_engine(
        cfg,
        [tracker],
        base_model,
        base_sae,
        llm_client,
        percentile_table,
        target_model_meta,
    ).get(concept.concept_id, [])


def _seed_row(
    idx: int,
    text: str = "prefix",
    sae_ids: list[int] | None = None,
) -> SeedRow:
    sae_ids = sorted(sae_ids or [])
    return SeedRow(
        shard_idx=0,
        doc_idx=idx,
        source_nanochat_token_count=6,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text=text,
        prefix_content_nanochat_tokens=[1, 2, 3],
        sae_feature_ids=sae_ids,
        sae_feature_mags=[1.0] * len(sae_ids),
        continuation_text="cont",
        continuation_content_nanochat_tokens=[10, 20, 30],
    )


def _concept(concept_id: int = 0, feature_index: int = 42) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {concept_id}",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _cluster_target(
    fingerprint: str = "c" * 64,
) -> ClusterFeatureTarget:
    return ClusterFeatureTarget(
        representative_id=2,
        allowed_member_ids=(2, 5),
        projecting_source_ids=(2, 5),
        feature_contract_fingerprint=fingerprint,
    )


class _StubSynthClient:
    """Deterministic stub for OpenRouterSynthClient.

    Returns canned responses from a queue, or raises a pre-set exception.
    ``cache_key`` is accepted and recorded for the cache-integration tests.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        raise_for: Exception | None = None,
    ) -> None:
        self.responses: list[str] = list(responses or [])
        self.raise_for = raise_for
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del system_prompt
        self.calls.append({"prompt": prompt, "cache_key": cache_key})
        if self.raise_for is not None:
            raise self.raise_for
        if not self.responses:
            return frame_stage1_response(prompt, "default continuation")
        return frame_stage1_response(prompt, self.responses.pop(0))


class _StubNanochatTokenizer:
    def get_bos_token_id(self) -> int:
        return 1

    def encode(
        self,
        text: str,
        prepend: int | None = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        content = [ord(char) for char in text]
        return ([prepend] if prepend is not None else []) + content

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _StubExtractionTokenizer:
    def __init__(self) -> None:
        self._next_id = 100
        self._texts: dict[int, str] = {}

    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_id = self._next_id
        self._next_id += 1
        self._texts[token_id] = text
        return [token_id]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._texts[token] for token in tokens)


class _ConcurrentSynthClient:
    def __init__(self, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del system_prompt
        with self.lock:
            self.calls.append({"prompt": prompt, "cache_key": cache_key})
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            threading.Event().wait(self.delay_s)
            return frame_stage1_response(prompt, "concurrent continuation")
        finally:
            with self.lock:
                self.in_flight -= 1


class _ShuffledConceptSynthClient:
    """Complete later concepts first while recording cross-concept overlap."""

    def __init__(
        self,
        *,
        initial_parties: int | None = None,
        preferred_first: str | None = None,
        preferred_completions: int = 1,
    ) -> None:
        self.initial_barrier = (
            None
            if initial_parties is None
            else threading.Barrier(initial_parties)
        )
        self.preferred_first = preferred_first
        self.preferred_completions = preferred_completions
        self.preferred_done = threading.Event()
        self.call_count = 0
        self.completion_order: list[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del cache_key, system_prompt
        description = next(
            line.removeprefix("DESCRIPTION=")
            for line in prompt.splitlines()
            if line.startswith("DESCRIPTION=")
        )
        with self.lock:
            call_index = self.call_count
            self.call_count += 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if (
                self.initial_barrier is not None
                and call_index < self.initial_barrier.parties
            ):
                self.initial_barrier.wait(timeout=5)
            if (
                self.preferred_first is not None
                and description != self.preferred_first
                and not self.preferred_done.wait(timeout=5)
            ):
                raise AssertionError("preferred concept did not complete")
            with self.lock:
                self.completion_order.append(description)
                if self.preferred_first is not None:
                    preferred_count = self.completion_order.count(
                        self.preferred_first
                    )
                    if preferred_count >= self.preferred_completions:
                        self.preferred_done.set()
            return frame_stage1_response(prompt, "deterministic continuation")
        finally:
            with self.lock:
                self.in_flight -= 1


class _ScriptedConcurrentSynthClient:
    def __init__(
        self,
        responses_by_prompt: dict[str, list[Any]],
        delay_s: float = 0.02,
    ) -> None:
        self.responses_by_prompt = {
            prompt: list(actions) for prompt, actions in responses_by_prompt.items()
        }
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del system_prompt
        with self.lock:
            self.calls.append({"prompt": prompt, "cache_key": cache_key})
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            threading.Event().wait(self.delay_s)
            actions = self.responses_by_prompt.get(prompt)
            if not actions and prompt.startswith("PREFIX="):
                lines = prompt.splitlines()
                prefix = lines[0].removeprefix("PREFIX=")
                description = lines[1].removeprefix("DESCRIPTION=")
                actions = self.responses_by_prompt.get(f"{prefix}::{description}")
            if not actions:
                raise AssertionError(f"unexpected synth call for prompt={prompt!r}")
            action = actions.pop(0)
            if isinstance(action, Exception):
                raise action
            return frame_stage1_response(prompt, action)
        finally:
            with self.lock:
                self.in_flight -= 1


class _FakePrefixCache:
    """Minimal PrefixPercentileCache test double."""

    def __init__(self, pct_by_text: dict[str, float]) -> None:
        self.pct_by_text = pct_by_text
        self.calls: list[dict[str, Any]] = []
        self.batch_size = 8

    def get_many(
        self,
        prefix_texts: list[str],
        f_c: int,
        *,
        cluster_target: ClusterFeatureTarget | None = None,
    ) -> list[float]:
        del cluster_target
        self.calls.append({"prefix_texts": list(prefix_texts), "f_c": f_c})
        return [self.pct_by_text.get(text, 0.0) for text in prefix_texts]


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """Replace all the production LLM + Gemma + percentile-table plumbing."""
    state: dict[str, Any] = {
        "synth_client": _StubSynthClient(responses=["canned continuation"] * 100),
        "percentile_pairs": [(10.0, 80.0)],  # (seed_pct, out_pct) per call
        "percentile_pair_calls": [],
        "prefix_percentiles": {},  # fid -> pct for fc_percentile_on_prefix
        # Strict-zero QC for the paired concept_avoiding row. Default: the
        # feature is INACTIVE on every continuation (natural tier accepts).
        # Tests override either the default or per-continuation behavior.
        "fc_active_default": False,
        "fc_active_by_continuation": {},  # continuation_text -> bool
        # Token-contribution control for the avoid classifier's NO_TOKENS trap.
        # Default (absent): any non-empty continuation adds tokens. Tests that
        # exercise the trap set continuation_text -> False here.
        "adds_tokens_by_continuation": {},  # continuation_text -> bool
        "avoid_qc_calls": [],
    }

    class _StubPrefixCache:
        batch_size = 8

        def __init__(self, **kwargs) -> None:
            state["prefix_cache_init"] = dict(kwargs)

        def get_many(
            self,
            prefix_texts: list[str],
            f_c: int,
            *,
            cluster_target: ClusterFeatureTarget | None = None,
        ) -> list[float]:
            del cluster_target
            state.setdefault("prefix_cache_calls", []).append(
                {"prefix_texts": list(prefix_texts), "f_c": f_c}
            )
            by_text = state.setdefault("prefix_percentiles_by_text", {})
            return [
                float(by_text.get(text, state["prefix_percentiles"].get(f_c, 0.0)))
                for text in prefix_texts
            ]

    def _fake_build_client(cfg, response_cache=None):
        del cfg
        state["captured_response_cache"] = response_cache
        return state["synth_client"]

    def _fake_build_table(stats_path, n_sae_features=None):
        del stats_path
        return t.zeros(max(n_sae_features or 64, 64), 99)

    def _fake_output_percentile(
        prefix,
        continuation,
        f_c,
        gemma,
        sae,
        table,
        *,
        sae_layer,
        tokenizer_id,
        cluster_target=None,
    ):
        del gemma, sae, table, sae_layer, cluster_target
        state["percentile_pair_calls"].append(
            {
                "prefix": prefix,
                "continuation": continuation,
                "f_c": f_c,
                "tokenizer_id": tokenizer_id,
            }
        )
        if state["percentile_pairs"]:
            return state["percentile_pairs"].pop(0)[1]
        return 80.0

    def _fake_output_percentiles(
        inputs,
        gemma,
        sae,
        table,
        *,
        sae_layer,
        tokenizer_id,
        release_memory=True,
    ):
        del release_memory
        return [
            _fake_output_percentile(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                gemma,
                sae,
                table,
                sae_layer=sae_layer,
                tokenizer_id=tokenizer_id,
                cluster_target=item.cluster_target,
            )
            for item in inputs
        ]

    def _fake_prefix_percentile(
        prefix,
        f_c,
        gemma,
        sae,
        table,
        *,
        sae_layer,
        tokenizer_id,
    ):
        del prefix, gemma, sae, table, sae_layer, tokenizer_id
        return state["prefix_percentiles"].get(f_c, 0.0)

    def _fake_classify_avoid(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold,
        tokenizer_id,
        cluster_target=None,
    ):
        del f_c, base_model, base_sae, sae_layer, cluster_target
        state["avoid_qc_calls"].append(
            {
                "prefix": prefix_text,
                "continuation": continuation_text,
                "tokenizer_id": tokenizer_id,
                "extraction_threshold": extraction_threshold,
            },
        )
        adds = state["adds_tokens_by_continuation"].get(
            continuation_text,
            bool(str(continuation_text).strip()),
        )
        if not adds:
            return positive_mode.AvoidContinuationOutcome.NO_TOKENS
        active = state["fc_active_by_continuation"].get(
            continuation_text,
            state["fc_active_default"],
        )
        return (
            positive_mode.AvoidContinuationOutcome.ACTIVE
            if active
            else positive_mode.AvoidContinuationOutcome.INACTIVE
        )

    def _fake_classify_avoids(
        inputs,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold,
        tokenizer_id,
        profile_region_name="synth_qc.classify_avoid_continuations",
    ):
        del profile_region_name
        return [
            _fake_classify_avoid(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                base_model,
                base_sae,
                sae_layer=sae_layer,
                extraction_threshold=extraction_threshold,
                tokenizer_id=tokenizer_id,
                cluster_target=item.cluster_target,
            )
            for item in inputs
        ]

    def _fake_load_template(name):
        del name
        return TEST_STAGE1_TEMPLATE

    def _fake_get_tokenizer(tokenizer_id):
        del tokenizer_id
        return _StubExtractionTokenizer()

    def _fake_infer_n_sae_features(base_sae):
        del base_sae
        return 16384  # stub value; production helper raises on broken SAEs

    monkeypatch.setattr(positive_mode, "_build_synth_client", _fake_build_client)
    monkeypatch.setattr(
        positive_mode,
        "_infer_n_sae_features",
        _fake_infer_n_sae_features,
    )
    monkeypatch.setattr(
        positive_mode,
        "build_percentile_table_from_stats",
        _fake_build_table,
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        _fake_output_percentile,
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        _fake_output_percentiles,
    )
    monkeypatch.setattr(
        positive_mode,
        "fc_percentile_on_prefix",
        _fake_prefix_percentile,
    )
    # Single combined avoid-QC seam (issue #513): the classifier returns the
    # tri-state from adds_tokens_by_continuation (NO_TOKENS trap) +
    # fc_active_by_continuation / fc_active_default (ACTIVE/INACTIVE). Avoids a
    # real extraction-tokenizer load in unit tests (the real tokenize-once logic
    # is covered in test_synth_qc.py).
    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuation",
        _fake_classify_avoid,
    )
    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuations",
        _fake_classify_avoids,
    )
    monkeypatch.setattr(positive_mode, "load_template", _fake_load_template)
    monkeypatch.setattr(positive_mode, "get_extraction_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(positive_mode, "get_tokenizer", lambda: _StubNanochatTokenizer())
    monkeypatch.setattr(positive_mode, "PrefixPercentileCache", _StubPrefixCache)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    return state


def _cfg(tmp_path, **overrides):
    # Historical tests expressed the Stage 1b semantic-attempt budget as
    # ``max_retries``. Preserve their requested total while the production
    # contract now reserves ``max_retries`` for Stage 0c/1c.
    if (
        "max_retries" in overrides
        and "positive_max_attempts_per_seed" not in overrides
    ):
        overrides["positive_max_attempts_per_seed"] = (
            int(overrides["max_retries"]) + 1
        )
    statistics_path = tmp_path / "stats.safetensors"
    statistics_path.write_bytes(b"stage1b-positive-mode-test-statistics\n")
    base = make_valid_config(
        run_tag="test_pos_mode",
        output_root=None,
        feature_statistics_path=str(statistics_path),
        n_positives_per_concept=2,
        min_prefix_chars=1,
        accept_threshold=0.3,
        max_retries=1,
        positive_max_attempts_per_seed=2,
        qc_fallback_on_exhaust="error",
        output_length=16,
        # Synthetic unit prefixes are intentionally tiny. Keep them eligible
        # unless a test is explicitly exercising the production character
        # bounds.
    )
    if overrides:
        return base.replace(**overrides)
    return base


def _select_seeds_for_concept(
    concept,
    seeds_train,
    cfg,
    base_model,
    base_sae,
    percentile_table,
    *,
    sae_layer,
    prefix_cache=None,
    cluster_target=None,
    effective_prefixes=None,
):
    """Expose production tracker admission for selection assertions."""
    if effective_prefixes is None:
        effective_prefixes = positive_mode._resolve_stage1b_prefixes(
            cfg,
            seeds_train,
        )
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds_train,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=sae_layer,
        prefix_cache=prefix_cache,
        cluster_target=cluster_target,
        effective_prefixes=effective_prefixes,
    )
    works, _next_order = positive_mode._admit_positive_work_items(
        tracker,
        start_order=0,
    )
    if len(works) != cfg.n_positives_per_concept:
        raise AxBenchBaseError(
            "Stage 1b candidate source exhausted after successful "
            "initial-target preflight"
        )
    return [work.pending.selected for work in works]


def _pick_seed_indices_for_concept(*args, **kwargs) -> list[int]:
    """Production-admission view for index-order assertions in this file."""

    return [selected.seed_idx for selected in _select_seeds_for_concept(*args, **kwargs)]


def _selected_records(
    seeds: list[SeedRow],
    indices: list[int],
    *,
    seed_percentile: float = 0.0,
) -> list[Any]:
    return [
        positive_mode._SelectedPositiveSeed(
            seed_idx=idx,
            seed=seeds[idx],
            prefix_text=seeds[idx].prefix_text,
            seed_percentile=seed_percentile,
            stored_clean=True,
        )
        for idx in indices
    ]


def test_cluster_cache_suffix_is_conditional_and_fingerprint_specific() -> None:
    target_a = _cluster_target("a" * 64)
    target_b = _cluster_target("b" * 64)
    tag_prefix = "pos_c3_s0_a0"

    unclustered_suffix = positive_mode._cluster_cache_suffix(None)
    suffix_a = positive_mode._cluster_cache_suffix(target_a)
    suffix_b = positive_mode._cluster_cache_suffix(target_b)
    key_a = positive_mode.make_cache_key(
        model="test-model",
        prompt="same prompt",
        tag=f"{tag_prefix}{suffix_a}",
    )
    key_b = positive_mode.make_cache_key(
        model="test-model",
        prompt="same prompt",
        tag=f"{tag_prefix}{suffix_b}",
    )

    assert unclustered_suffix == ""
    assert suffix_a == f"_fc{target_a.feature_contract_fingerprint}"
    assert suffix_b == f"_fc{target_b.feature_contract_fingerprint}"
    assert key_a != key_b


def test_stage1b_fingerprint_binds_semantics_and_excludes_operations(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    concept = _concept()
    seed = _seed_row(0)
    meta = _stub_meta()
    templates = {
        "synth_continue_with_concept.txt": "rich-template-v1",
        "synth_continue_avoiding_concept.txt": "avoid-template-v1",
    }
    tokenizer_identity = ["nanochat-tokenizer-v1"]
    monkeypatch.setattr(
        positive_mode,
        "load_template",
        lambda name: templates[name],
    )
    monkeypatch.setattr(
        positive_mode,
        "compute_tokenizer_checksum",
        lambda _tokenizer: tokenizer_identity[0],
    )

    def fingerprint(
        *,
        cfg_arg=cfg,
        concepts_arg=None,
        seeds_arg=None,
        meta_arg=meta,
    ):
        return positive_mode._stage1b_checkpoint_fingerprint(
            cfg_arg,
            [concept] if concepts_arg is None else concepts_arg,
            _test_seed_rows_sha256(
                [seed] if seeds_arg is None else seeds_arg
            ),
            meta_arg,
            cluster_contract=None,
        )

    baseline = fingerprint()
    assert set(baseline) == {
        "checkpoint_algorithm_version",
        "positive_selection_contract",
        "ordered_concepts_sha256",
        "seed_rows_sha256",
        "feature_statistics_sha256",
        "extraction_model_id",
        "sae_identity",
        "extraction_tokenizer_id",
        "nanochat_tokenizer_checksum",
        "master_seed",
        "output_length",
        "synth_provider",
        "synth_model",
        "rich_prompt_template_sha256",
        "avoiding_prompt_template_sha256",
        "continuation_contract",
        "cluster_feature_contract_fingerprint",
        "extraction_threshold",
        "qc_fallback_on_exhaust",
        "max_concept_drop_fraction",
        "max_dropped_concept_fraction",
    }
    phase1_golden_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "stage1b_phase1_target72_fingerprint.json"
    )
    phase1_golden = json.loads(
        phase1_golden_path.read_text(encoding="utf-8")
    )
    legacy_v3 = dict(baseline)
    legacy_v3["checkpoint_algorithm_version"] = (
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
    )
    legacy_v3["seed_rows_sha256"] = positive_mode._seed_rows_identity([seed])
    assert positive_mode._stage1b_checkpoint_fingerprint_v1(
        cfg,
        legacy_v3,
    ) == phase1_golden

    semantic_cfgs = [
        cfg.replace(qc_batch_size=cfg.qc_batch_size + 1),
        cfg.replace(
            positive_max_attempts_per_seed=(
                cfg.positive_max_attempts_per_seed + 1
            )
        ),
        cfg.replace(positive_zero_hit_bail_after=None),
        cfg.replace(accept_threshold=0.2),
        cfg.replace(positive_prefix_headroom_ceiling=0.8),
        cfg.replace(min_prefix_chars=0),
        cfg.replace(max_prefix_chars=cfg.max_prefix_chars + 1),
        cfg.replace(output_length=cfg.output_length + 1),
        cfg.replace(extraction_threshold=cfg.extraction_threshold + 0.1),
        cfg.replace(extraction_tokenizer_id="different/tokenizer"),
        cfg.replace(feature_extraction_model_id="different-model"),
        cfg.replace(neuronpedia_model_id="different-neuronpedia-model"),
        cfg.replace(neuronpedia_source="different-source"),
        cfg.replace(qc_fallback_on_exhaust="warn"),
        cfg.replace(max_concept_drop_fraction=0.1),
        cfg.replace(max_dropped_concept_fraction=0.1),
        cfg.replace(
            generation=replace(
                cfg.generation,
                master_seed=cfg.generation.master_seed + 1,
            )
        ),
        cfg.replace(synth=replace(cfg.synth, provider="different-provider")),
        cfg.replace(synth=replace(cfg.synth, model="different/model")),
        cfg.replace(synth=replace(cfg.synth, max_concurrent=1)),
    ]
    assert all(
        fingerprint(cfg_arg=semantic_cfg) != baseline
        for semantic_cfg in semantic_cfgs
    )
    assert fingerprint(
        concepts_arg=[replace(concept, description="changed concept")]
    ) != baseline
    assert fingerprint(
        seeds_arg=[replace(seed, continuation_text="changed continuation")]
    ) != baseline
    assert fingerprint(meta_arg=replace(meta, sae_layer=9)) != baseline

    different_stats = tmp_path / "different-stats.safetensors"
    different_stats.write_bytes(b"different-stage1b-statistics-content")
    assert fingerprint(
        cfg_arg=cfg.replace(feature_statistics_path=str(different_stats))
    ) != baseline

    templates["synth_continue_with_concept.txt"] = "rich-template-v2"
    assert fingerprint() != baseline
    templates["synth_continue_with_concept.txt"] = "rich-template-v1"
    templates["synth_continue_avoiding_concept.txt"] = "avoid-template-v2"
    assert fingerprint() != baseline
    templates["synth_continue_avoiding_concept.txt"] = "avoid-template-v1"
    tokenizer_identity[0] = "nanochat-tokenizer-v2"
    assert fingerprint() != baseline
    tokenizer_identity[0] = "nanochat-tokenizer-v1"

    operational = cfg.replace(
        output_root=str(tmp_path / "elsewhere"),
        run_tag="different-tag",
        enable_prefix_percentile_store=(
            not cfg.enable_prefix_percentile_store
        ),
        synth=replace(
            cfg.synth,
            requests_per_minute=1,
            transport_max_retries=99,
            transport_retry_budget_seconds=999.0,
        ),
    )
    assert fingerprint(cfg_arg=operational) == baseline
    assert fingerprint(
        cfg_arg=cfg.replace(n_positives_per_concept=100)
    ) == baseline


def test_legacy_seed_identity_preserves_numeric_tags_and_rejects_non_numeric() -> None:
    integer = _seed_row(0, sae_ids=[2])
    integer.sae_feature_mags[0] = 1
    floating = _seed_row(0, sae_ids=[2])
    assert positive_mode._seed_rows_identity([integer]) != (
        positive_mode._seed_rows_identity([floating])
    )
    invalid = _seed_row(0, sae_ids=[2])
    invalid_magnitudes: Any = invalid.sae_feature_mags
    invalid_magnitudes[0] = "1"
    with pytest.raises(AxBenchBaseError, match="non-numeric"):
        positive_mode._seed_rows_identity([invalid])


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64])
def test_public_stage1b_rejects_noncanonical_seed_digest_before_checkpoint_work(
    stub_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
) -> None:
    del stub_env
    monkeypatch.setattr(
        positive_mode,
        "build_percentile_table_from_stats",
        lambda *_args, **_kwargs: pytest.fail("checkpoint setup was reached"),
    )
    with pytest.raises(AxBenchBaseError, match="lowercase 64-character"):
        positive_mode.synthesize_positive_mode(
            _cfg(tmp_path),
            [_concept()],
            [_seed_row(0)],
            object(),
            object(),
            _stub_meta(),
            seed_rows_sha256=digest,
        )


def test_fresh_and_v4_resume_never_call_legacy_seed_identity(
    stub_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    seeds = [_seed_row(0)]
    original_writer = positive_mode._write_training_parquet

    def fail_legacy_identity(_rows: list[SeedRow]) -> str:
        raise AssertionError("legacy identity was computed")

    def leave_v4_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("leave v4 checkpoint")

    monkeypatch.setattr(
        positive_mode,
        "_seed_rows_identity",
        fail_legacy_identity,
    )
    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        leave_v4_checkpoint,
    )
    with pytest.raises(RuntimeError, match="leave v4 checkpoint"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            seeds,
            object(),
            object(),
            _stub_meta(),
        )
    checkpoint_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).stage1b_partial_json()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["fingerprint"]["checkpoint_algorithm_version"] == (
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
    )

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        original_writer,
    )
    provider_calls = len(stub_env["synth_client"].calls)
    out = synthesize_positive_mode(
        cfg,
        [_concept()],
        seeds,
        object(),
        object(),
        _stub_meta(),
    )
    assert out.is_file()
    assert len(stub_env["synth_client"].calls) == provider_calls


def test_parallel_chain_threads_cluster_target_and_cache_suffix(
    tmp_path,
    monkeypatch,
) -> None:
    target = _cluster_target()
    captured_targets: list[ClusterFeatureTarget | None] = []
    captured_tags: list[str] = []
    monkeypatch.setattr(
        positive_mode,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: _StubNanochatTokenizer(),
    )
    monkeypatch.setattr(
        positive_mode,
        "get_tokenizer",
        lambda: _StubNanochatTokenizer(),
    )
    monkeypatch.setattr(
        positive_mode,
        "load_template",
        lambda _name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode,
        "fc_percentile_on_prefix",
        lambda *_args, **_kwargs: 0.0,
    )

    def _cache_key(*, model: str, prompt: str, tag: str) -> str:
        del model, prompt
        captured_tags.append(tag)
        return tag

    def _complete(_client, requests, *, handle_results, **_kwargs):
        pending = list(requests)
        while pending:
            pairs = [
                (
                    request,
                    positive_mode.SynthResult(
                        index=request.index,
                        response_text=frame_stage1_response(
                            request.prompt,
                            "cluster continuation",
                        ),
                    ),
                )
                for request in pending
            ]
            pending = [
                request
                for request in handle_results(pairs)
                if request is not None
            ]

    def _rich_scores(inputs, *_args, **_kwargs):
        captured_targets.extend(item.cluster_target for item in inputs)
        return [80.0] * len(inputs)

    strict_calls = 0

    def _strict_outcomes(inputs, *_args, **_kwargs):
        nonlocal strict_calls
        captured_targets.extend(item.cluster_target for item in inputs)
        strict_calls += 1
        outcome = (
            positive_mode.AvoidContinuationOutcome.ACTIVE
            if strict_calls == 1
            else positive_mode.AvoidContinuationOutcome.INACTIVE
        )
        return [outcome] * len(inputs)

    monkeypatch.setattr(positive_mode, "make_cache_key", _cache_key)
    monkeypatch.setattr(positive_mode, "complete_synth_request_chains", _complete)
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        _rich_scores,
    )
    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuations",
        _strict_outcomes,
    )
    monkeypatch.setattr(
        positive_mode,
        "_validate_checkpoint_ordinary_rows",
        lambda *_args, **_kwargs: None,
    )
    rows = _per_concept_positives(
        _cfg(
            tmp_path,
            n_positives_per_concept=1,
            qc_batch_size=16,
            synth=SynthConfig(
                requests_per_minute=0,
                max_concurrent=2,
            ),
        ),
        _concept(concept_id=3, feature_index=2),
        [_seed_row(7)],
        base_model=object(),
        base_sae=object(),
        llm_client=_StubSynthClient(),
        percentile_table=object(),
        target_model_meta=_stub_meta(),
        cluster_target=target,
    )

    assert captured_targets == [target, target, target]
    assert captured_tags == [
        f"pos_c3_s0_a0_fc{target.feature_contract_fingerprint}",
        f"pos_avoid_c3_s0_a0_fc{target.feature_contract_fingerprint}",
    ]
    assert [row["content_class"] for row in rows] == [
        "concept_rich",
        "concept_avoiding",
    ]


def test_terminal_chain_state_is_pruned_and_duplicate_callback_rejected(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    duplicate_errors: list[str] = []

    def _complete(
        llm_client,
        requests,
        *,
        handle_result,
        terminal_chain_refill,
        **_kwargs,
    ) -> None:
        assert len(requests) == 1
        request = requests[0]
        while request is not None:
            response = llm_client.complete(
                request.prompt,
                cache_key=request.cache_key,
            )
            request = handle_result(
                request,
                positive_mode.SynthResult(
                    index=request.index,
                    response_text=response,
                ),
            )
        terminal_request = requests[0]
        assert terminal_chain_refill(terminal_request) == []
        with pytest.raises(
            AxBenchBaseError,
            match="duplicate terminal chain",
        ) as exc_info:
            terminal_chain_refill(terminal_request)
        duplicate_errors.append(str(exc_info.value))

    monkeypatch.setattr(
        positive_mode,
        "complete_synth_request_chains",
        _complete,
    )
    rows = _per_concept_positives(
        _cfg(
            tmp_path,
            n_positives_per_concept=1,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
        ),
        _concept(),
        [_seed_row(0)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert len(rows) == 2
    assert len(duplicate_errors) == 1


def _reference_pick_seed_indices_for_concept(
    concept: Concept,
    seeds_train: list[SeedRow],
    cfg,
    prefix_percentiles: dict[str, float],
) -> list[int]:
    """Legacy reference for Stage 1b seed picking.

    This encodes the pre-cache semantics directly in the test:
    shuffle deterministically, partition into clean/dirty via plain
    membership, then score dirty prefixes individually and sort by
    prefix percentile ascending.
    """
    rng = random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    )
    indices = list(range(len(seeds_train)))
    rng.shuffle(indices)
    n = min(cfg.n_positives_per_concept, len(seeds_train))
    clean: list[int] = []
    dirty: list[tuple[int, float]] = []
    for idx in indices:
        seed = seeds_train[idx]
        if concept.feature_index in seed.sae_feature_ids:
            dirty.append((idx, prefix_percentiles[seed.prefix_text]))
        else:
            clean.append(idx)
    if len(clean) >= n:
        return clean[:n]
    dirty.sort(key=lambda pair: pair[1])
    return clean + [idx for idx, _pct in dirty[: n - len(clean)]]


def _make_random_seed_case(
    rng: random.Random,
    *,
    case_id: int,
    feature_id: int,
) -> tuple[list[SeedRow], dict[str, float]]:
    """Build a deterministic randomized seed pool plus dirty prefix percentiles."""
    seeds: list[SeedRow] = []
    prefix_percentiles: dict[str, float] = {}
    n_seeds = rng.randint(1, 18)
    feature_universe = [fid for fid in range(256) if fid != feature_id]
    percentile_choices = [0.0, 0.5, 5.0, 10.0, 25.0, 50.0, 75.0, 99.0]
    for idx in range(n_seeds):
        text = f"case-{case_id}-seed-{idx}"
        width = rng.randint(0, 96)
        sae_ids = sorted(rng.sample(feature_universe, k=width))
        if rng.random() < 0.6 and feature_id not in sae_ids:
            sae_ids.append(feature_id)
            sae_ids.sort()
        seeds.append(_seed_row(idx, text=text, sae_ids=sae_ids))
        prefix_percentiles[text] = rng.choice(percentile_choices)
    return seeds, prefix_percentiles


# ---------------------------------------------------------------------------
# _pick_seed_indices_for_concept
# ---------------------------------------------------------------------------


def test_seed_filtering_prefers_clean_seeds(stub_env, tmp_path):
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    seeds = [
        _seed_row(0, sae_ids=[42]),  # dirty
        _seed_row(1, sae_ids=[]),  # clean
        _seed_row(2, sae_ids=[]),  # clean
        _seed_row(3, sae_ids=[42]),  # dirty
    ]
    concept = _concept(feature_index=42)
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
    )
    # Both clean seeds should be selected.
    assert sorted(picked) == [1, 2]


def test_seed_filtering_scores_compatibility_stream_even_when_quota_filled(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Clean classification is only a priority; exact headroom is still scored."""
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    seeds = [
        _seed_row(0, text="dirty", sae_ids=[42]),
        _seed_row(1, text="clean-1", sae_ids=[]),
        _seed_row(2, text="clean-2", sae_ids=[]),
        _seed_row(3, text="clean-3", sae_ids=[]),
    ]
    scored: list[str] = []

    def _score(prefix, *args, **kwargs):
        del args, kwargs
        scored.append(prefix)
        return 0.0

    monkeypatch.setattr(positive_mode, "fc_percentile_on_prefix", _score)
    concept = _concept(feature_index=42)
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
    )
    assert len(picked) == 2
    assert all(42 not in seeds[idx].sae_feature_ids for idx in picked)
    assert scored == ["clean-2", "clean-1", "clean-3", "dirty"]


def test_seed_filtering_cache_scores_compatibility_prefix(stub_env, tmp_path):
    """A small target still scores the target-independent physical prefix."""
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    seeds = [
        _seed_row(0, sae_ids=[42]),
        _seed_row(1, sae_ids=[]),
        _seed_row(2, sae_ids=[]),
        _seed_row(3, sae_ids=[]),
    ]

    class _RecordingCache:
        batch_size = 8

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
            del f_c
            assert all(text != "prefix" for text in prefix_texts) is False
            self.calls.append(list(prefix_texts))
            return [0.0] * len(prefix_texts)

    concept = _concept(feature_index=42)
    cache = _RecordingCache()
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        prefix_cache=cache,
    )
    assert len(picked) == 2
    assert all(42 not in seeds[idx].sae_feature_ids for idx in picked)
    assert sum(map(len, cache.calls)) == 4


def test_fallback_to_least_activating_when_insufficient_clean(
    stub_env,
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path, n_positives_per_concept=3)
    seeds = [
        _seed_row(0, text="prefix 0", sae_ids=[42]),
        _seed_row(1, text="prefix 1", sae_ids=[42]),
        _seed_row(2, text="prefix 2", sae_ids=[42]),
        _seed_row(3, text="prefix 3", sae_ids=[]),  # only clean
    ]
    percentile_by_text = {"prefix 0": 50.0, "prefix 1": 10.0, "prefix 2": 90.0}

    def _by_text(prefix_text, f_c, gemma, sae, table, *, sae_layer, tokenizer_id):
        del f_c, gemma, sae, table, sae_layer, tokenizer_id
        return percentile_by_text.get(prefix_text, 0.0)

    monkeypatch.setattr(positive_mode, "fc_percentile_on_prefix", _by_text)
    concept = _concept(feature_index=42)
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
    )
    # Clean idx 3 first, then dirty ordered by ascending percentile: 1 (10), 0 (50), 2 (90).
    # n_positives_per_concept=3 -> clean=[3], dirty_needed=2 -> [1, 0]
    assert picked[0] == 3
    assert picked[1] == 1
    assert picked[2] == 0


def test_fallback_uses_prefix_cache_and_preserves_dirty_order(stub_env, tmp_path):
    cfg = _cfg(tmp_path, n_positives_per_concept=3)
    seeds = [
        _seed_row(0, text="prefix 0", sae_ids=[42]),
        _seed_row(1, text="prefix 1", sae_ids=[42]),
        _seed_row(2, text="prefix 2", sae_ids=[42]),
        _seed_row(3, text="prefix 3", sae_ids=[]),
    ]
    cache = _FakePrefixCache(
        {"prefix 0": 50.0, "prefix 1": 10.0, "prefix 2": 90.0},
    )
    concept = _concept(feature_index=42)
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        prefix_cache=cache,
    )
    assert picked == [3, 1, 0]
    rng = random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    )
    shuffled = list(range(len(seeds)))
    rng.shuffle(shuffled)
    expected_dirty_texts = [
        seeds[idx].prefix_text for idx in shuffled if 42 in seeds[idx].sae_feature_ids
    ]
    assert cache.calls == [
        {
            "prefix_texts": ["prefix 3"],
            "f_c": 42,
        },
        {
            "prefix_texts": expected_dirty_texts,
            "f_c": 42,
        },
    ]


def test_all_clean_seeds_cap_at_n_positives(stub_env, tmp_path):
    """If more clean seeds than n_positives_per_concept, cap at the limit."""
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    seeds = [_seed_row(i, sae_ids=[]) for i in range(5)]  # 5 clean seeds
    concept = _concept(feature_index=42)
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
    )
    assert len(picked) == 2


def test_all_dirty_seeds_no_clean(stub_env, tmp_path, monkeypatch):
    """All seeds dirty -> all selections come from dirty sorted by pct."""
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    seeds = [
        _seed_row(0, text="a", sae_ids=[42]),
        _seed_row(1, text="b", sae_ids=[42]),
        _seed_row(2, text="c", sae_ids=[42]),
    ]
    pct_by_text = {"a": 80.0, "b": 20.0, "c": 40.0}

    def _by_text(prefix_text, f_c, gemma, sae, table, *, sae_layer, tokenizer_id):
        del f_c, gemma, sae, table, sae_layer, tokenizer_id
        return pct_by_text[prefix_text]

    monkeypatch.setattr(positive_mode, "fc_percentile_on_prefix", _by_text)
    concept = _concept(feature_index=42)
    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
    )
    # Sorted ascending: b (20) -> c (40) -> a (80). Take 2 -> [1, 2].
    assert picked == [1, 2]


def test_pick_fewer_seeds_than_requested_fails_loud(stub_env, tmp_path):
    """A thin positive corpus must never be emitted silently."""
    cfg = _cfg(tmp_path, n_positives_per_concept=10)
    seeds = [_seed_row(i, sae_ids=[]) for i in range(3)]
    concept = _concept()
    with pytest.raises(
        AxBenchBaseError,
        match=r"too few admissible prefixes.*required=10, selected=3",
    ):
        _pick_seed_indices_for_concept(
            concept,
            seeds,
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
        )


def test_selection_applies_headroom_to_clean_and_dirty_candidates(
    stub_env,
    tmp_path,
):
    """Stored cleanliness is a priority, never a substitute for exact scoring."""

    del stub_env
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        accept_threshold=0.3,
        positive_prefix_headroom_ceiling=0.9,
    )
    seeds = [
        _seed_row(0, text="clean-too-hot", sae_ids=[]),
        _seed_row(1, text="clean-at-cap", sae_ids=[]),
        _seed_row(2, text="clean-low", sae_ids=[]),
        _seed_row(3, text="dirty-too-hot", sae_ids=[42]),
        _seed_row(4, text="dirty-at-cap", sae_ids=[42]),
        _seed_row(5, text="dirty-low", sae_ids=[42]),
    ]
    percentiles = {
        "clean-too-hot": 60.000001,
        "clean-at-cap": 60.0,
        "clean-low": 1.0,
        "dirty-too-hot": 99.5,
        "dirty-at-cap": 60.0,
        "dirty-low": 2.0,
    }

    selected = _select_seeds_for_concept(
        _concept(feature_index=42),
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache(percentiles),
    )

    assert len(selected) == 3
    assert all(candidate.seed_percentile <= 60.0 for candidate in selected)
    assert "clean-too-hot" not in {candidate.prefix_text for candidate in selected}
    assert [candidate.stored_clean for candidate in selected[:2]] == [
        True,
        True,
    ]
    assert selected[2].stored_clean is False
    assert selected[2].prefix_text == "dirty-low"
    assert selected[2].seed_percentile == 2.0


def test_selection_dirty_equal_score_ties_preserve_seed_shuffle_order(
    stub_env,
    tmp_path,
):
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=3)
    concept = _concept(concept_id=19, feature_index=42)
    seeds = [_seed_row(idx, text=f"dirty-{idx}", sae_ids=[42]) for idx in range(8)]
    rng = random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    )
    expected = list(range(len(seeds)))
    rng.shuffle(expected)

    selected = _select_seeds_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 12.0 for seed in seeds}),
    )

    assert [candidate.seed_idx for candidate in selected] == expected[:3]


@pytest.mark.parametrize(
    "invalid_score",
    [
        None,
        True,
        "12.0",
        float("-inf"),
        -0.000001,
        0.499999,
        99.500001,
        float("inf"),
        float("nan"),
    ],
)
def test_selection_rejects_invalid_prefix_scorer_currency(
    stub_env,
    tmp_path,
    invalid_score,
):
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    with pytest.raises(
        AxBenchBaseError,
        match="prefix scorer returned an invalid raw percentile",
    ):
        _select_seeds_for_concept(
            _concept(),
            [_seed_row(0, text="candidate")],
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            prefix_cache=_FakePrefixCache({"candidate": invalid_score}),
        )


@pytest.mark.parametrize("max_concurrent", [1, 2])
@pytest.mark.parametrize(
    "invalid_score",
    [
        None,
        True,
        "12.0",
        float("-inf"),
        -0.000001,
        0.499999,
        99.500001,
        float("inf"),
        float("nan"),
    ],
)
def test_rich_qc_rejects_invalid_output_scorer_currency(
    stub_env,
    tmp_path,
    max_concurrent,
    invalid_score,
):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_retries=0,
        synth=SynthConfig(
            requests_per_minute=0,
            max_concurrent=max_concurrent,
        ),
    )
    stub_env["percentile_pairs"] = [(0.0, invalid_score)]

    with pytest.raises(
        AxBenchBaseError,
        match="output scorer returned an invalid raw percentile",
    ):
        _per_concept_positives(
            cfg,
            _concept(),
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )


@pytest.mark.parametrize("returned_scores", [[], [0.0, 1.0]])
def test_selection_rejects_prefix_scorer_cardinality_mismatch(
    stub_env,
    tmp_path,
    returned_scores,
):
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=1)

    class _BrokenCache:
        batch_size = 8

        def get_many(self, prefix_texts, f_c):
            del prefix_texts, f_c
            return list(returned_scores)

    with pytest.raises(
        AxBenchBaseError,
        match=r"returned .* scores for 1 candidates",
    ):
        _select_seeds_for_concept(
            _concept(),
            [_seed_row(0, text="candidate")],
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            prefix_cache=_BrokenCache(),
        )


def test_selection_underfill_error_reports_every_relevant_count(
    stub_env,
    tmp_path,
):
    del stub_env
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        min_prefix_chars=5,
    )
    seeds = [
        _seed_row(0, text="x", sae_ids=[]),
        _seed_row(1, text="clean-hot", sae_ids=[]),
        _seed_row(2, text="dirty-low", sae_ids=[42]),
        _seed_row(3, text="dirty-hot", sae_ids=[42]),
    ]
    with pytest.raises(AxBenchBaseError) as exc_info:
        _select_seeds_for_concept(
            _concept(),
            seeds,
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            prefix_cache=_FakePrefixCache(
                {
                    "clean-hot": 90.0,
                    "dirty-low": 10.0,
                    "dirty-hot": 90.0,
                }
            ),
        )

    message = str(exc_info.value)
    for expected_fragment in (
        "required=3",
        "selected=1",
        "pool=4",
        "clean=1",
        "dirty=2",
        "too_short=1",
        "clean_scored=1",
        "dirty_scored=2",
        "headroom_rejected=2",
        "required_delta=30.0",
        "maximum_seed_percentile=60.0",
    ):
        assert expected_fragment in message


def test_candidate_source_continues_once_clean_then_dirty_without_rescoring(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    concept = _concept(concept_id=7, feature_index=42)
    seeds = [
        _seed_row(0, text="clean-0"),
        _seed_row(1, text="dirty-1", sae_ids=[42]),
        _seed_row(2, text="clean-2"),
        _seed_row(3, text="dirty-3", sae_ids=[42]),
        _seed_row(4, text="clean-4"),
    ]
    cache = _FakePrefixCache(
        {seed.prefix_text: 10.0 for seed in seeds}
    )
    source = positive_mode._PositiveCandidateSource(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        prefix_cache=cache,
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            cfg,
            seeds,
        ),
    )

    source.preflight_initial_target(1)
    selected = []
    while (candidate := _draw_next_candidate(source)) is not None:
        selected.append(candidate)

    shuffled = list(range(len(seeds)))
    random.Random(
        cfg.generation.master_seed
        + positive_mode._CONCEPT_SEED_OFFSET
        + concept.concept_id
    ).shuffle(shuffled)
    expected_clean = [
        idx for idx in shuffled if 42 not in seeds[idx].sae_feature_ids
    ]
    expected_dirty = [
        idx for idx in shuffled if 42 in seeds[idx].sae_feature_ids
    ]
    assert [candidate.seed_idx for candidate in selected] == [
        *expected_clean,
        *expected_dirty,
    ]
    assert [candidate.candidate_rank for candidate in selected] == list(
        range(len(seeds))
    )
    scored_texts = [
        text
        for call in cache.calls
        for text in call["prefix_texts"]
    ]
    assert sorted(scored_texts) == sorted(
        seed.prefix_text for seed in seeds
    )
    assert len(scored_texts) == len(set(scored_texts))
    # The physical pool is too small to reach the 72-row compatibility goal,
    # so all clean candidates and then the dirty pool are scored up front.
    assert [len(call["prefix_texts"]) for call in cache.calls] == [3, 2]


@pytest.mark.parametrize(
    ("target", "expected_widths", "expected_buffered"),
    [
        (1, [16, 16, 16, 16, 8], 72),
        (15, [16, 16, 16, 16, 8], 72),
        (16, [16, 16, 16, 16, 8], 72),
        (17, [16, 16, 16, 16, 8], 72),
        (71, [16, 16, 16, 16, 8], 72),
        (72, [16, 16, 16, 16, 8], 72),
        (73, [16, 16, 16, 16, 8, 16], 88),
        (100, [16, 16, 16, 16, 8, 16, 16], 104),
    ],
)
def test_candidate_stream_target_matrix_has_compatibility_batching(
    tmp_path,
    target: int,
    expected_widths: list[int],
    expected_buffered: int,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=target,
        qc_batch_size=16,
    )
    concept = _concept(concept_id=7)
    seeds = [
        _seed_row(index, text=f"matrix-{index}") for index in range(120)
    ]
    cache = _FakePrefixCache(
        {seed.prefix_text: 0.0 for seed in seeds}
    )
    cache.batch_size = 16
    source = positive_mode._PositiveCandidateSource(
        concept,
        seeds,
        cfg,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=cache,
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            cfg,
            seeds,
        ),
    )

    source.preflight_initial_target(target)

    assert [len(call["prefix_texts"]) for call in cache.calls] == expected_widths
    assert len(source._buffer) == expected_buffered
    shuffled = list(range(len(seeds)))
    random.Random(
        cfg.generation.master_seed
        + positive_mode._CONCEPT_SEED_OFFSET
        + concept.concept_id
    ).shuffle(shuffled)
    assert [
        text for call in cache.calls for text in call["prefix_texts"]
    ] == [seeds[index].prefix_text for index in shuffled[:expected_buffered]]


def test_candidate_stream_headroom_refill_is_target_independent_below_72(
    tmp_path,
) -> None:
    calls_by_target: dict[int, list[list[str]]] = {}
    for target in (1, 72):
        cfg = _cfg(
            tmp_path,
            n_positives_per_concept=target,
            qc_batch_size=16,
        )
        concept = _concept(concept_id=9)
        seeds = [
            _seed_row(index, text=f"headroom-{index}")
            for index in range(100)
        ]
        shuffled = list(range(len(seeds)))
        random.Random(
            cfg.generation.master_seed
            + positive_mode._CONCEPT_SEED_OFFSET
            + concept.concept_id
        ).shuffle(shuffled)
        rejected = set(shuffled[:4])
        cache = _FakePrefixCache(
            {
                seed.prefix_text: (90.0 if index in rejected else 0.0)
                for index, seed in enumerate(seeds)
            }
        )
        cache.batch_size = 16
        source = positive_mode._PositiveCandidateSource(
            concept,
            seeds,
            cfg,
            object(),
            object(),
            None,
            sae_layer=8,
            prefix_cache=cache,
            cluster_target=None,
            effective_prefixes=positive_mode._resolve_stage1b_prefixes(
                cfg,
                seeds,
            ),
        )

        source.preflight_initial_target(target)
        calls_by_target[target] = [
            list(call["prefix_texts"]) for call in cache.calls
        ]
        assert source.headroom_rejected == 4
        assert len(source._buffer) == 72

    assert calls_by_target[1] == calls_by_target[72]
    assert [len(call) for call in calls_by_target[72]] == [16, 16, 16, 16, 12]


def test_candidate_stream_physical_exhaustion_allows_only_achievable_target(
    tmp_path,
) -> None:
    seeds = [_seed_row(index, text=f"short-{index}") for index in range(10)]
    concept = _concept()

    for target, succeeds in ((1, True), (15, False)):
        cfg = _cfg(tmp_path, n_positives_per_concept=target, qc_batch_size=16)
        cache = _FakePrefixCache(
            {seed.prefix_text: 0.0 for seed in seeds}
        )
        cache.batch_size = 16
        source = positive_mode._PositiveCandidateSource(
            concept,
            seeds,
            cfg,
            object(),
            object(),
            None,
            sae_layer=8,
            prefix_cache=cache,
            cluster_target=None,
            effective_prefixes=positive_mode._resolve_stage1b_prefixes(
                cfg,
                seeds,
            ),
        )
        if succeeds:
            source.preflight_initial_target(target)
            assert source.physically_exhausted
            assert len(source._buffer) == 10
        else:
            with pytest.raises(AxBenchBaseError, match="required=15, selected=10"):
                source.preflight_initial_target(target)
        assert [len(call["prefix_texts"]) for call in cache.calls] == [10]


def test_post_compatibility_partial_batch_occurs_only_at_clean_exhaustion(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=73, qc_batch_size=16)
    seeds = [_seed_row(index, text=f"partial-{index}") for index in range(75)]
    cache = _FakePrefixCache({seed.prefix_text: 0.0 for seed in seeds})
    cache.batch_size = 16
    source = positive_mode._PositiveCandidateSource(
        _concept(),
        seeds,
        cfg,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=cache,
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )

    source.preflight_initial_target(73)

    assert [len(call["prefix_texts"]) for call in cache.calls] == [
        16,
        16,
        16,
        16,
        8,
        3,
    ]
    assert source.physically_exhausted
    assert len(source._buffer) == 75


@pytest.mark.parametrize(
    ("stored_target", "current_target"),
    [
        (1, 15),
        (15, 16),
        (16, 17),
        (17, 71),
        (71, 72),
        (72, 73),
        (73, 100),
        (1, 100),
    ],
)
def test_candidate_stream_restore_growth_matrix_matches_fresh_stream(
    tmp_path,
    stored_target: int,
    current_target: int,
) -> None:
    concept = _concept(concept_id=13)
    seeds = [
        _seed_row(index, text=f"restore-growth-{index}")
        for index in range(120)
    ]
    percentiles = {seed.prefix_text: 0.0 for seed in seeds}

    def prepare(target: int, cache: _FakePrefixCache):
        cfg = _cfg(
            tmp_path,
            n_positives_per_concept=target,
            qc_batch_size=16,
        )
        cache.batch_size = 16
        tracker = positive_mode._prepare_positive_target_tracker(
            cfg,
            concept,
            seeds,
            object(),
            object(),
            None,
            sae_layer=8,
            prefix_cache=cache,
            cluster_target=None,
            effective_prefixes=positive_mode._resolve_stage1b_prefixes(
                cfg,
                seeds,
            ),
        )
        return cfg, tracker

    stored_cache = _FakePrefixCache(percentiles)
    stored_cfg, stored_tracker = prepare(stored_target, stored_cache)
    document, rows = positive_mode._stage1b_checkpoint_snapshot(
        stored_cfg,
        [stored_tracker],
        fingerprint={"test": "source-growth-matrix"},
        highest_requested_target=stored_target,
        next_chain_index=0,
        next_acceptance_sequence=0,
    )

    current_cfg = stored_cfg.replace(
        n_positives_per_concept=current_target
    )
    restored_cache = _FakePrefixCache(percentiles)
    restored_cache.batch_size = 16
    restored_tracker = positive_mode._restore_positive_target_trackers(
        current_cfg,
        [concept],
        seeds,
        object(),
        object(),
        None,
        _stub_meta(),
        prefix_cache=restored_cache,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            current_cfg,
            seeds,
        ),
        cluster_contract=None,
        checkpoint_document=document,
        checkpoint_rows=rows,
    )[0]

    fresh_cache = _FakePrefixCache(percentiles)
    fresh_cfg, fresh_tracker = prepare(current_target, fresh_cache)
    restored_document, _ = positive_mode._stage1b_checkpoint_snapshot(
        current_cfg,
        [restored_tracker],
        fingerprint={"test": "source-growth-matrix"},
        highest_requested_target=current_target,
        next_chain_index=0,
        next_acceptance_sequence=0,
    )
    fresh_document, _ = positive_mode._stage1b_checkpoint_snapshot(
        fresh_cfg,
        [fresh_tracker],
        fingerprint={"test": "source-growth-matrix"},
        highest_requested_target=current_target,
        next_chain_index=0,
        next_acceptance_sequence=0,
    )

    restored_calls = [*stored_cache.calls, *restored_cache.calls]
    assert restored_calls == fresh_cache.calls
    assert restored_document == fresh_document


@pytest.mark.parametrize("already_owned", [True, -1, 2])
def test_restored_candidate_preflight_rejects_invalid_owned_count(
    tmp_path,
    already_owned: int,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    seed = _seed_row(0)
    source = positive_mode._PositiveCandidateSource(
        _concept(),
        [seed],
        cfg,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 0.0}),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            cfg,
            [seed],
        ),
    )

    with pytest.raises(AxBenchBaseError, match="invalid owned count"):
        source.preflight_restored_target(
            target=1,
            already_owned=already_owned,
        )


def test_replacement_scoring_updates_legacy_and_summary_counters(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    seeds = [
        _seed_row(idx, text=f"candidate-{idx}")
        for idx in range(5)
    ]
    order = _all_clean_candidate_order(cfg, count=len(seeds))
    percentiles = {
        f"candidate-{idx}": (90.0 if idx in order[2:4] else 0.0)
        for idx in order
    }
    cache = _FakePrefixCache(percentiles)
    rejected_prefix = f"candidate-{order[0]}"
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda prefix, *_args, **_kwargs: (
            0.0 if prefix == rejected_prefix else 80.0
        ),
    )
    profile_dir = tmp_path / "replacement-counter-profile"

    with AxBenchProfiler(profile_dir):
        rows = _per_concept_positives(
            cfg,
            _concept(),
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
            prefix_cache=cache,
        )

    assert [row["prefix_text"] for row in rows] == [
        f"candidate-{order[1]}",
        f"candidate-{order[1]}",
    ]
    assert [len(call["prefix_texts"]) for call in cache.calls] == [5]
    totals = _read_counter_totals(profile_dir)
    assert totals["stage1b.prefix_candidates_scored"] == 5.0
    assert totals["stage1b.summary.scored"] == 5.0
    assert totals["stage1b.headroom_rejected_seeds"] == 2.0
    assert totals["stage1b.summary.headroom_rejected"] == 2.0


class _TrackerCandidateSource(positive_mode._PositiveCandidateSource):
    def __init__(
        self,
        candidates: list[positive_mode._SelectedPositiveSeed],
    ) -> None:
        self.candidates = list(candidates)

    def draw_next(self) -> positive_mode._SelectedPositiveSeed | None:
        selected = self.prepare_next()
        if selected is not None:
            self.commit_prepared(selected)
        return selected

    def prepare_next(self) -> positive_mode._SelectedPositiveSeed | None:
        if not self.candidates:
            return None
        return self.candidates[0]

    def commit_prepared(
        self,
        selected: positive_mode._SelectedPositiveSeed,
    ) -> None:
        assert self.candidates[0] is selected
        self.candidates.pop(0)


def _tracker_for_invariant_tests(
    *,
    target: int = 1,
    candidate_count: int = 1,
) -> positive_mode._PositiveTargetTracker:
    candidates = [
        positive_mode._SelectedPositiveSeed(
            seed_idx=idx,
            seed=_seed_row(idx),
            prefix_text="prefix",
            seed_percentile=0.0,
            stored_clean=True,
            candidate_rank=idx,
        )
        for idx in range(candidate_count)
    ]
    return positive_mode._PositiveTargetTracker(
        concept=_concept(),
        source=_TrackerCandidateSource(candidates),
        target=target,
        zero_hit_bail_after=None,
        cluster_target=None,
        started_at=0.0,
    )


def test_target_tracker_rejects_duplicate_or_stale_completion() -> None:
    selected = positive_mode._SelectedPositiveSeed(
        seed_idx=3,
        seed=_seed_row(3),
        prefix_text="prefix",
        seed_percentile=0.0,
        stored_clean=True,
        candidate_rank=0,
    )
    tracker = positive_mode._PositiveTargetTracker(
        concept=_concept(),
        source=_TrackerCandidateSource([selected]),
        target=1,
        zero_hit_bail_after=None,
        cluster_target=None,
        started_at=0.0,
    )
    work = tracker.admit_next(order=0)
    assert work is not None
    work.rich_row = {"content_class": "concept_rich"}
    work.avoiding_row = {"content_class": "concept_avoiding"}
    tracker.complete_success(
        work, natural_avoid=True, acceptance_sequence=0
    )

    with pytest.raises(AxBenchBaseError, match="duplicate or stale"):
        tracker.complete_success(
            work, natural_avoid=True, acceptance_sequence=0
        )


def test_target_tracker_rejects_duplicate_seed_admission() -> None:
    selected = positive_mode._SelectedPositiveSeed(
        seed_idx=3,
        seed=_seed_row(3),
        prefix_text="prefix",
        seed_percentile=0.0,
        stored_clean=True,
        candidate_rank=0,
    )
    tracker = positive_mode._PositiveTargetTracker(
        concept=_concept(),
        source=_TrackerCandidateSource([selected, selected]),
        target=2,
        zero_hit_bail_after=None,
        cluster_target=None,
        started_at=0.0,
    )
    assert tracker.admit_next(order=0) is not None
    with pytest.raises(AxBenchBaseError, match="duplicate seed admission"):
        tracker.admit_next(order=1)


def test_target_tracker_rejects_accepted_unadmitted_seed() -> None:
    selected = positive_mode._SelectedPositiveSeed(
        seed_idx=3,
        seed=_seed_row(3),
        prefix_text="prefix",
        seed_percentile=0.0,
        stored_clean=True,
        candidate_rank=0,
    )
    tracker = positive_mode._PositiveTargetTracker(
        concept=_concept(),
        source=_TrackerCandidateSource([selected]),
        target=1,
        zero_hit_bail_after=None,
        cluster_target=None,
        started_at=0.0,
    )
    work = tracker.admit_next(order=0)
    assert work is not None
    work.rich_row = {"content_class": "concept_rich"}
    work.avoiding_row = {"content_class": "concept_avoiding"}
    tracker.complete_success(
        work, natural_avoid=True, acceptance_sequence=0
    )
    tracker.admitted_seed_indices.clear()
    tracker.seeds_drawn = 0

    with pytest.raises(AxBenchBaseError, match="accepted unadmitted seed"):
        tracker._assert_invariants()


def test_target_tracker_rejects_invalid_target() -> None:
    with pytest.raises(AxBenchBaseError, match="target >= 1"):
        _tracker_for_invariant_tests(target=0, candidate_count=0)


def test_target_tracker_rejects_target_overflow() -> None:
    tracker = _tracker_for_invariant_tests()
    work = tracker.admit_next(order=0)
    assert work is not None
    tracker.accepted_items.append(work)

    with pytest.raises(AxBenchBaseError, match="exceeded.*target"):
        tracker._assert_invariants()


def test_target_tracker_rejects_outstanding_unadmitted_seed() -> None:
    tracker = _tracker_for_invariant_tests()
    assert tracker.admit_next(order=0) is not None
    tracker.admitted_seed_indices.clear()
    tracker.seeds_drawn = 0

    with pytest.raises(AxBenchBaseError, match="outstanding unadmitted seed"):
        tracker._assert_invariants()


def test_target_tracker_rejects_duplicate_accepted_seed() -> None:
    tracker = _tracker_for_invariant_tests(target=2)
    work = tracker.admit_next(order=0)
    assert work is not None
    work.rich_row = {"content_class": "concept_rich"}
    work.avoiding_row = {"content_class": "concept_avoiding"}
    tracker.complete_success(
        work, natural_avoid=True, acceptance_sequence=0
    )
    tracker.accepted_items.append(work)

    with pytest.raises(AxBenchBaseError, match="duplicate accepted seeds"):
        tracker._assert_invariants()


def test_target_tracker_rejects_accepted_outstanding_overlap() -> None:
    tracker = _tracker_for_invariant_tests(target=2)
    work = tracker.admit_next(order=0)
    assert work is not None
    tracker.accepted_items.append(work)

    with pytest.raises(AxBenchBaseError, match="both accepted and outstanding"):
        tracker._assert_invariants()


def test_target_tracker_rejects_seed_draw_count_mismatch() -> None:
    tracker = _tracker_for_invariant_tests(candidate_count=0)
    tracker.seeds_drawn = 1

    with pytest.raises(AxBenchBaseError, match="seed-draw count is inconsistent"):
        tracker._assert_invariants(allow_refill_deficit=True)


def test_target_tracker_rejects_unexplained_deficit() -> None:
    tracker = _tracker_for_invariant_tests(candidate_count=0)

    with pytest.raises(AxBenchBaseError, match="unexplained target deficit"):
        tracker._assert_invariants()


def test_target_tracker_validation_failures_do_not_mutate_live_chain() -> None:
    tracker = _tracker_for_invariant_tests()
    work = tracker.admit_next(order=0)
    assert work is not None
    work.rich_row = {"content_class": "concept_rich"}

    with pytest.raises(AxBenchBaseError, match="incomplete pair"):
        tracker.complete_success(
            work, natural_avoid=True, acceptance_sequence=0
        )
    assert tracker.outstanding == {work.pending.seed_idx: work}
    assert not work.terminal

    with pytest.raises(AxBenchBaseError, match="unknown failure side"):
        tracker.complete_failure(work, side="typo")
    assert tracker.outstanding == {work.pending.seed_idx: work}
    assert not work.terminal


def test_target_refill_engine_rejects_duplicate_concept_ids_before_requests(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    trackers = [
        positive_mode._PositiveTargetTracker(
            concept=_concept(),
            source=_TrackerCandidateSource([]),
            target=1,
            zero_hit_bail_after=None,
            cluster_target=None,
            started_at=0.0,
        )
        for _ in range(2)
    ]
    client = _StubSynthClient()

    with pytest.raises(AxBenchBaseError, match="duplicate concept IDs.*\\[0\\]"):
        positive_mode._run_positive_target_refill_engine(
            cfg,
            trackers,
            base_model=object(),
            base_sae=object(),
            llm_client=client,
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )

    assert client.calls == []


def test_seed_has_feature_matches_plain_membership_reference():
    """The bisect-based fast path must match plain Python membership exactly."""
    rng = random.Random(0)
    lengths = [0, 1, 7, 32, 63, 64, 65, 96]
    for case_id, length in enumerate(lengths):
        sae_ids = sorted(rng.sample(range(2048), k=length))
        seed = _seed_row(case_id, sae_ids=sae_ids)
        present = list(sae_ids[: min(8, len(sae_ids))])
        absent = list(range(2056, 2064))
        queries = present + absent + rng.sample(range(2048), k=16)
        for feature_id in queries:
            assert positive_mode._seed_has_feature(
                seed,
                feature_id,
            ) == (feature_id in sae_ids)


def test_validate_seed_feature_ids_sorted_rejects_unsorted_long_rows():
    ordered = list(range(80))
    unsorted_ids = list(ordered)
    unsorted_ids[20], unsorted_ids[21] = unsorted_ids[21], unsorted_ids[20]
    malformed = _seed_row(1, sae_ids=ordered)
    object.__setattr__(malformed, "sae_feature_ids", unsorted_ids)
    with pytest.raises(AxBenchBaseError, match="requires sorted SeedRow"):
        _validate_seed_feature_ids_sorted([_seed_row(0, sae_ids=ordered), malformed])


def test_pick_seed_indices_matches_reference_without_cache_across_randomized_cases(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Current picker matches the legacy no-cache logic across mixed pools."""
    del stub_env
    rng = random.Random(0)
    concept = _concept(concept_id=7, feature_index=42)

    def _by_text(prefix_text, f_c, gemma, sae, table, *, sae_layer, tokenizer_id):
        del f_c, gemma, sae, table, sae_layer, tokenizer_id
        return prefix_percentiles[prefix_text]

    for case_id in range(32):
        seeds, prefix_percentiles = _make_random_seed_case(
            rng,
            case_id=case_id,
            feature_id=concept.feature_index,
        )
        cfg = _cfg(
            tmp_path,
            n_positives_per_concept=rng.randint(1, min(8, len(seeds))),
            accept_threshold=0.001,
            positive_prefix_headroom_ceiling=1.0,
        )
        monkeypatch.setattr(positive_mode, "fc_percentile_on_prefix", _by_text)
        expected = _reference_pick_seed_indices_for_concept(
            concept,
            seeds,
            cfg,
            prefix_percentiles,
        )
        actual = _pick_seed_indices_for_concept(
            concept,
            seeds,
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
        )
        assert actual == expected


def test_pick_seed_indices_matches_reference_with_cache_across_randomized_cases(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """The cached picker preserves exact legacy row selection semantics."""
    del stub_env, monkeypatch
    rng = random.Random(1)
    concept = _concept(concept_id=11, feature_index=42)
    for case_id in range(32):
        seeds, prefix_percentiles = _make_random_seed_case(
            rng,
            case_id=case_id,
            feature_id=concept.feature_index,
        )
        cfg = _cfg(
            tmp_path,
            n_positives_per_concept=rng.randint(1, min(8, len(seeds))),
            accept_threshold=0.001,
            positive_prefix_headroom_ceiling=1.0,
        )
        prefix_cache = _FakePrefixCache(prefix_percentiles)
        expected = _reference_pick_seed_indices_for_concept(
            concept,
            seeds,
            cfg,
            prefix_percentiles,
        )
        actual = _pick_seed_indices_for_concept(
            concept,
            seeds,
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            prefix_cache=prefix_cache,
        )
        assert actual == expected


def test_per_concept_positives_match_reference_no_cache_rows(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Per-concept output rows are identical to the no-cache reference path."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        accept_threshold=0.5,
        max_retries=2,
        qc_fallback_on_exhaust="best",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    concept = _concept(concept_id=3, feature_index=42)
    seeds = [
        _seed_row(0, text="dirty-a", sae_ids=[42]),
        _seed_row(1, text="clean-a", sae_ids=[]),
        _seed_row(2, text="dirty-b", sae_ids=[1, 42, 99]),
        _seed_row(3, text="clean-b", sae_ids=[]),
        _seed_row(4, text="dirty-c", sae_ids=[42]),
    ]
    prefix_percentiles = {
        "dirty-a": 60.0,
        "clean-a": 0.0,
        "dirty-b": 10.0,
        "clean-b": 0.0,
        "dirty-c": 40.0,
    }
    responses = [
        "ref cont 0a",
        "ref cont 0b",
        "ref cont 1a",
        "ref cont 2a",
        "ref cont 2b",
        "ref cont 2c",
    ]
    percentile_pairs = [
        (20.0, 35.0),  # fail
        (20.0, 80.0),  # accept
        (10.0, 90.0),  # accept
        (30.0, 45.0),  # fail
        (30.0, 60.0),  # fail, best
        (30.0, 55.0),  # fail
    ]

    def _run(*, prefix_cache):
        stub_env["synth_client"] = _StubSynthClient(responses=list(responses))
        stub_env["percentile_pairs"] = list(percentile_pairs)
        stub_env["percentile_pair_calls"].clear()
        return _per_concept_positives(
            cfg,
            concept,
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
            prefix_cache=prefix_cache,
        )

    stub_env["prefix_percentiles"] = dict(prefix_percentiles)
    monkeypatch.setattr(
        positive_mode,
        "fc_percentile_on_prefix",
        lambda prefix_text, f_c, gemma, sae, table, *, sae_layer, tokenizer_id: prefix_percentiles[
            prefix_text
        ],
    )
    expected = _run(prefix_cache=None)
    actual = _run(prefix_cache=_FakePrefixCache(prefix_percentiles))
    assert _rows_byte_equal(actual, expected)


# ---------------------------------------------------------------------------
# _per_concept_positives -- QC retry loop
# ---------------------------------------------------------------------------


def test_accept_on_first_attempt(stub_env, tmp_path):
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    stub_env["percentile_pairs"] = [(10.0, 80.0)]  # delta=70
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    # Each accepted concept_rich row is now paired with a concept_avoiding row.
    assert len(rows) == 2
    assert rows[0]["content_class"] == "concept_rich"
    assert rows[0]["qc_accepted"] is True
    assert rows[0]["qc_attempts"] == 1
    assert rows[1]["content_class"] == "concept_avoiding"
    # Rich LLM called once; the avoid side accepts via the natural tier (no LLM).
    assert len(stub_env["synth_client"].calls) == 1


def test_exact_prepared_prefix_and_cached_score_flow_through_entire_row(
    stub_env,
    tmp_path,
):
    """Selection, prompt, retry QC, rich row, and avoid row share one string."""

    source_prefix = "old material that must be removed|CANONICAL12"
    expected_prefix = source_prefix[-11:]
    assert expected_prefix == "CANONICAL12"
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        min_prefix_chars=11,
        max_prefix_chars=11,
        accept_threshold=0.3,
    )
    seed = _seed_row(0, text=source_prefix, sae_ids=[])
    stub_env["percentile_pairs"] = [(999.0, 47.0)]
    prefix_cache = _FakePrefixCache({expected_prefix: 17.0})

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [seed],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
        prefix_cache=prefix_cache,
    )

    rich, avoiding = rows
    assert prefix_cache.calls == [{"prefix_texts": [expected_prefix], "f_c": 42}]
    assert expected_prefix in stub_env["synth_client"].calls[0]["prompt"]
    assert source_prefix not in stub_env["synth_client"].calls[0]["prompt"]
    assert stub_env["percentile_pair_calls"] == [
        {
            "prefix": expected_prefix,
            "continuation": "canned continuation",
            "f_c": 42,
            "tokenizer_id": cfg.extraction_tokenizer_id,
        }
    ]
    assert rich["prefix_text"] == expected_prefix
    assert rich["qc_seed_percentile"] == 17.0
    assert rich["qc_output_percentile"] == 47.0
    assert rich["qc_delta"] == 30.0
    assert rich["qc_accepted"] is True
    assert avoiding["prefix_text"] == expected_prefix


def test_raw_percentile_delta_boundary_retries_below_then_accepts_exactly(
    stub_env,
    tmp_path,
):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.3,
        max_retries=1,
    )
    stub_env["percentile_pairs"] = [
        (999.0, 41.999999),
        (999.0, 42.0),
    ]
    stub_env["synth_client"].responses = ["first", "second"]
    prefix_cache = _FakePrefixCache({"prefix": 12.0})

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
        prefix_cache=prefix_cache,
    )

    rich = rows[0]
    assert rich["output_text"] == "second"
    assert rich["qc_seed_percentile"] == 12.0
    assert rich["qc_output_percentile"] == 42.0
    assert rich["qc_delta"] == 30.0
    assert rich["qc_attempts"] == 2
    assert len(stub_env["percentile_pair_calls"]) == 2


def test_rich_source_copy_is_retryable_and_never_reaches_sae_qc(
    stub_env,
    tmp_path,
):
    copied = "seven substantial source words copied exactly into this output"
    assert len(copied.split()) >= 7
    assert len(copied) >= 32
    prefix = f"prefix opening {copied} prefix ending"
    stub_env["synth_client"].responses = [
        f"new opening {copied} new ending",
        "independent concept-bearing continuation",
    ]
    stub_env["percentile_pairs"] = [(0.0, 80.0)]

    rows = _per_concept_positives(
        _cfg(
            tmp_path,
            n_positives_per_concept=1,
            max_retries=1,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
        ),
        _concept(),
        [_seed_row(0, text=prefix)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert rows[0]["output_text"] == "independent concept-bearing continuation"
    assert rows[0]["qc_attempts"] == 2
    assert len(stub_env["percentile_pair_calls"]) == 1


def test_rich_repeated_phrase_loop_is_retryable_before_sae_qc(
    stub_env,
    tmp_path,
):
    loop = "these deliberately substantial words form a repeated provider loop"
    assert len(loop.split()) >= 8
    assert len(loop) >= 48
    stub_env["synth_client"].responses = [
        f"{loop}. interlude. {loop}.",
        "independent concept-bearing continuation",
    ]
    stub_env["percentile_pairs"] = [(0.0, 80.0)]

    rows = _per_concept_positives(
        _cfg(
            tmp_path,
            n_positives_per_concept=1,
            max_retries=1,
        ),
        _concept(),
        [_seed_row(0)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert rows[0]["output_text"] == "independent concept-bearing continuation"
    assert rows[0]["qc_attempts"] == 2
    assert len(stub_env["percentile_pair_calls"]) == 1


def test_parallel_rich_source_copy_retries_with_identical_guard_semantics(
    stub_env,
    tmp_path,
):
    copied = "seven substantial source words copied exactly into this output"
    prefix = f"prefix opening {copied} prefix ending"
    stub_env["synth_client"] = _StubSynthClient(
        responses=[
            f"new opening {copied} new ending",
            "independent concept-bearing continuation",
        ],
    )
    stub_env["percentile_pairs"] = [(0.0, 80.0)]
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_retries=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0, text=prefix)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert rows[0]["output_text"] == "independent concept-bearing continuation"
    assert rows[0]["qc_attempts"] == 2
    assert len(stub_env["percentile_pair_calls"]) == 1


def test_retry_on_qc_fail_then_accept(stub_env, tmp_path, monkeypatch):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    counters: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        positive_mode,
        "record_counter",
        lambda name, value=1, **attrs: counters.append((name, value, attrs)),
    )
    # First attempt: delta=10 (fails), second: delta=60 (passes)
    stub_env["percentile_pairs"] = [(10.0, 20.0), (10.0, 70.0)]
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    rich = [r for r in rows if r["content_class"] == "concept_rich"]
    assert len(rich) == 1
    assert rich[0]["qc_attempts"] == 2
    assert rich[0]["qc_accepted"] is True
    assert (
        "stage1b.rich_qc_attempts_to_accept",
        2,
        {
            "concept_id": concept.concept_id,
            "seed_idx": 0,
            "outcome": "accepted",
        },
    ) in counters


def test_positive_retry_past_special_token_refusal_then_accept(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Regression (#602 finding 1): a special-token-polluted LLM generation is
    REFUSED (LlmSpecialTokenRefusal) at the parse boundary, but the synth stage
    must NOT crash — it skips that attempt and retries with a fresh cache key,
    accepting the clean generation. Pre-fix the raise escaped the retry loop and
    aborted (and cache-poisoned) the whole stage.
    """
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    # Attempt 0 emits a literal special-token string; attempt 1 is clean.
    stub_env["synth_client"].responses = ["polluted <eos>", "clean continuation"]
    # QC only runs for the clean attempt (the refusal skips before QC): delta=70
    # passes the 50.0 threshold.
    stub_env["percentile_pairs"] = [(10.0, 80.0)]

    original_parse = continuation_protocol.parse_stage1_continuation_response

    def _refuse_special(**kwargs):
        if "<eos>" in kwargs["response"]:
            raise response_parser.LlmSpecialTokenRefusal("simulated LLM special")
        return original_parse(**kwargs)

    monkeypatch.setattr(
        continuation_protocol,
        "parse_stage1_continuation_response",
        _refuse_special,
    )

    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    rich = [r for r in rows if r["content_class"] == "concept_rich"]
    # The stage completed (no crash) and accepted the clean attempt on retry.
    assert len(rich) == 1
    assert rich[0]["qc_accepted"] is True
    assert rich[0]["output_text"] == "clean continuation"
    # attempt 0 refused+skipped, attempt 1 accepted.
    assert rich[0]["qc_attempts"] == 2
    # Both attempts hit the LLM — the refusal did not short-circuit the retry.
    assert len(stub_env["synth_client"].calls) == 2


def test_handle_positive_attempt_result_skips_on_special_refusal(tmp_path, monkeypatch):
    """Regression (#602 finding 1), BATCHED (parallel-wave) callback path: a
    refused (special-token-polluted) generation must NOT raise out of the
    per-attempt callback — it returns to skip the attempt (item stays not-done),
    exactly as a failed LLM call (result.error) does,
    so the parallel driver retries. Pre-fix the raise escaped the callback and
    crashed the whole request chain. (The inline-loop sibling is covered by
    test_positive_retry_past_special_token_refusal_then_accept above.)
    """
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    concept = _concept()
    seed = _seed_row(0, sae_ids=[])
    item = positive_mode._PositivePending(
        order=0,
        selected=_selected_records([seed], [0])[0],
    )
    request = positive_mode._stage1b_continuation_request(
        TEST_STAGE1_TEMPLATE,
        concept=concept,
        prefix_text=item.selected.prefix_text,
        seed_idx=item.seed_idx,
        attempt=0,
        content_class="concept_rich",
        cluster_target=None,
    )
    result = positive_mode.SynthResult(
        index=0,
        response_text=frame_stage1_response(
            request.prompt,
            "polluted <eos>",
        ),
    )
    original_parse = continuation_protocol.parse_stage1_continuation_response

    def _refuse_special(**kwargs):
        if "<eos>" in kwargs["response"]:
            raise response_parser.LlmSpecialTokenRefusal("simulated LLM special")
        return original_parse(**kwargs)

    monkeypatch.setattr(
        continuation_protocol,
        "parse_stage1_continuation_response",
        _refuse_special,
    )

    # Must NOT raise; the refusal skips the attempt without reaching QC / the
    # row build below.
    handle_positive_attempt_result(
        result,
        item,
        request,
        cfg,
        concept,
        object(),
        object(),
        None,
        object(),
        "src",
        sae_layer=8,
        attempt=0,
    )
    assert item.done is False
    assert item.accepted_row is None


def test_retry_qc_still_evaluates_each_attempt_with_prefix_cache(stub_env, tmp_path):
    """The prefix cache is seed-selection-only; retry QC still runs per attempt."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
    )
    seeds = [_seed_row(0, text="dirty prefix", sae_ids=[42])]
    concept = _concept(feature_index=42)
    prefix_cache = _FakePrefixCache({"dirty prefix": 0.0})
    stub_env["percentile_pairs"] = [(10.0, 20.0), (10.0, 70.0)]
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
        prefix_cache=prefix_cache,
    )
    rich = [r for r in rows if r["content_class"] == "concept_rich"]
    assert len(rich) == 1
    assert rich[0]["qc_attempts"] == 2
    # rich QC ran twice; avoid side accepts via natural tier (no compute_fc).
    assert len(stub_env["percentile_pair_calls"]) == 2
    # rich LLM called twice; avoid natural tier needs no LLM call.
    assert len(stub_env["synth_client"].calls) == 2
    assert len(prefix_cache.calls) == 1


def test_retry_qc_receives_cfg_tokenizer_id(stub_env, tmp_path):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        extraction_tokenizer_id="custom/stage1b-tokenizer",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert stub_env["percentile_pair_calls"]
    assert {call["tokenizer_id"] for call in stub_env["percentile_pair_calls"]} == {
        cfg.extraction_tokenizer_id
    }


def test_positive_rows_match_legacy_dirty_seed_selection_with_cache(
    stub_env,
    tmp_path,
):
    """Tiny functional guard: new cached picker selects the legacy dirty rows."""
    cfg = _cfg(tmp_path, n_positives_per_concept=2, accept_threshold=0.5)
    seeds = [_seed_row(i, text=f"prefix {i}", sae_ids=[42]) for i in range(5)]
    pct_by_text = {
        "prefix 0": 50.0,
        "prefix 1": 10.0,
        "prefix 2": 90.0,
        "prefix 3": 5.0,
        "prefix 4": 10.0,
    }
    concept = _concept(feature_index=42)

    rng = random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    )
    legacy_indices = list(range(len(seeds)))
    rng.shuffle(legacy_indices)
    legacy_dirty = [(idx, pct_by_text[seeds[idx].prefix_text]) for idx in legacy_indices]
    legacy_dirty.sort(key=lambda pair: pair[1])
    expected_prefixes = [seeds[idx].prefix_text for idx, _pct in legacy_dirty[:2]]

    prefix_cache = _FakePrefixCache(pct_by_text)
    stub_env["synth_client"] = _StubSynthClient(
        responses=[f"continuation {i}" for i in range(10)],
    )
    stub_env["percentile_pairs"] = [(0.0, 80.0)] * 10
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
        prefix_cache=prefix_cache,
    )
    rich_prefixes = [row["prefix_text"] for row in rows if row["content_class"] == "concept_rich"]
    assert rich_prefixes == expected_prefixes


def test_hard_error_on_retries_exhausted(stub_env, tmp_path):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="error",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    stub_env["percentile_pairs"] = [(10.0, 20.0), (10.0, 25.0)]
    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*rich_discards=1",
    ):
        _per_concept_positives(
            cfg,
            concept,
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )


def test_qc_fallback_warn_skips_row(stub_env, tmp_path):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    stub_env["percentile_pairs"] = [(10.0, 20.0), (10.0, 25.0)]
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    assert rows == []


def test_qc_fallback_best_never_accepts_failed_attempt(stub_env, tmp_path):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
        qc_fallback_on_exhaust="best",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    # The selected prefix-alone canonical seed score is 0. Three output
    # attempts follow; the best canonical delta is therefore 40.
    stub_env["percentile_pairs"] = [
        (10.0, 15.0),  # delta=5
        (10.0, 40.0),  # canonical delta=40  <- best
        (10.0, 20.0),  # delta=10
    ]
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    assert rows == []
    assert len(stub_env["synth_client"].calls) == 3


def test_qc_fallback_best_does_not_swallow_transport_exhaustion(
    stub_env,
    tmp_path,
):
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
        qc_fallback_on_exhaust="best",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    stub_env["synth_client"].raise_for = AxBenchBaseError("openrouter down")
    with pytest.raises(
        AxBenchBaseError,
        match=r"provider transport exhausted.*side=rich.*semantic_attempt=1",
    ):
        _per_concept_positives(
            cfg,
            concept,
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )
    assert len(stub_env["synth_client"].calls) == 1


def test_qc_fallback_best_rejects_empty_continuations_no_row(stub_env, tmp_path):
    """#602 finding 3 (early rejection): a blank continuation is rejected at synth
    time BEFORE it can become a best-fallback candidate, so even under the 'best'
    fallback an all-empty seed yields NO concept_rich row. Empty output_text can
    thus never reach the corpus (Stage 2 treats any empty row as a hard invariant
    violation). Pre-fix, the blank continuation flowed into the row + best
    fallback and was written as an empty concept_rich row.
    """
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
        qc_fallback_on_exhaust="best",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    # Every attempt's response parses (pass-through) to a blank continuation.
    stub_env["synth_client"].responses = ["   ", "", "\t\n"]
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    # No empty concept_rich row written, even under the lenient 'best' fallback.
    assert [r for r in rows if r["content_class"] == "concept_rich"] == []


def test_qc_fallback_warn_does_not_swallow_transport_exhaustion(
    stub_env,
    tmp_path,
):
    """Transport exhaustion aborts before fallback dispatch."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concept = _concept()
    stub_env["synth_client"].raise_for = AxBenchBaseError("openrouter down")
    with pytest.raises(
        AxBenchBaseError,
        match=r"provider transport exhausted.*side=rich.*semantic_attempt=1",
    ):
        _per_concept_positives(
            cfg,
            concept,
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )
    assert len(stub_env["synth_client"].calls) == 1


# ---------------------------------------------------------------------------
# Paired concept_avoiding row (two-tier: natural -> LLM fallback)
# ---------------------------------------------------------------------------


def _split_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Group emitted rows into (rich, avoiding) pairs by emission order.

    Asserts the emission order is strictly [rich, avoiding, rich, ...].
    """
    pairs: list[tuple[dict, dict]] = []
    it = iter(rows)
    for rich in it:
        assert rich["content_class"] == "concept_rich"
        avoid = next(it)
        assert avoid["content_class"] == "concept_avoiding"
        pairs.append((rich, avoid))
    return pairs


def test_every_accepted_rich_row_has_paired_avoid_row(stub_env, tmp_path):
    """(a) Each accepted rich row is paired with an avoid row, same id/prefix."""
    cfg = _cfg(tmp_path, n_positives_per_concept=2, accept_threshold=0.5)
    seeds = [
        _seed_row(0, text="prefix A", sae_ids=[]),
        _seed_row(1, text="prefix B", sae_ids=[]),
    ]
    concept = _concept(concept_id=7, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    pairs = _split_pairs(rows)
    assert len(pairs) == 2
    for rich, avoid in pairs:
        assert rich["concept_id"] == avoid["concept_id"] == 7
        assert rich["prefix_text"] == avoid["prefix_text"]
        assert avoid["category"] == "negative"
        assert avoid["content_class"] == "concept_avoiding"
        assert avoid["qc_feature_id"] == 42
        # avoid output_text must never be empty.
        assert avoid["output_text"].strip() != ""


def test_natural_tier_accept_when_feature_inactive(stub_env, tmp_path):
    """(b) Natural tier: feature inactive -> corpus gen_source, QC verified."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    # The seed's natural continuation does NOT fire the feature.
    stub_env["fc_active_by_continuation"] = {"cont": False}
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    ((_, avoid),) = _split_pairs(rows)
    # Output is the seed's natural continuation.
    assert avoid["output_text"] == "cont"
    assert avoid["gen_source"] == "corpus:doc_continuation@0/0"
    assert avoid["qc_zero_activation_verified"] is True
    # No LLM fallback call was needed for the avoid side: only the rich call.
    assert len(stub_env["synth_client"].calls) == 1
    # Strict-zero QC was evaluated on the natural continuation.
    assert any(c["continuation"] == "cont" for c in stub_env["avoid_qc_calls"])


def test_llm_avoid_source_copy_retries_before_strict_zero_qc(
    stub_env,
    tmp_path,
):
    copied = "seven substantial source words copied exactly into this output"
    prefix = f"prefix opening {copied} prefix ending"
    copied_avoid = f"new opening {copied} new ending"
    stub_env["synth_client"] = _StubSynthClient(
        responses=["RICH", copied_avoid, "INDEPENDENT AVOID"],
    )
    stub_env["fc_active_by_continuation"] = {
        "cont": True,
        "INDEPENDENT AVOID": False,
    }

    rows = _per_concept_positives(
        _cfg(
            tmp_path,
            n_positives_per_concept=1,
            max_retries=1,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
        ),
        _concept(),
        [_seed_row(0, text=prefix, sae_ids=[])],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    ((_, avoid),) = _split_pairs(rows)
    assert avoid["output_text"] == "INDEPENDENT AVOID"
    assert avoid["qc_attempts"] == 2
    assert [call["continuation"] for call in stub_env["avoid_qc_calls"]] == [
        "cont",
        "INDEPENDENT AVOID",
    ]
    assert copied_avoid not in {call["continuation"] for call in stub_env["avoid_qc_calls"]}
    assert len(stub_env["synth_client"].calls) == 3


def test_llm_fallback_when_natural_fires(stub_env, tmp_path):
    """(c) Natural continuation fires -> fall back to LLM avoiding synth."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    # rich LLM returns "RICH"; avoiding LLM returns "AVOID-OK".
    stub_env["synth_client"] = _StubSynthClient(responses=["RICH", "AVOID-OK"])
    # natural "cont" fires (fail), "AVOID-OK" does not (LLM accepts).
    stub_env["fc_active_by_continuation"] = {"cont": True, "AVOID-OK": False}
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    ((rich, avoid),) = _split_pairs(rows)
    assert rich["output_text"] == "RICH"
    assert avoid["output_text"] == "AVOID-OK"
    # LLM-tier avoid row carries the synth provenance, not the corpus tag.
    assert avoid["gen_source"].startswith("openrouter:")
    assert avoid["qc_zero_activation_verified"] is True
    # Two LLM calls: one rich, one avoiding fallback.
    assert len(stub_env["synth_client"].calls) == 2


def test_natural_tier_routes_to_llm_when_continuation_adds_no_tokens(
    stub_env,
    tmp_path,
):
    """QC-trap guard: a non-empty natural continuation that adds NO tokens
    beyond the prefix (n_prefix >= len(full); a false strict-zero PASS) is
    skipped, and synthesis falls back to the LLM tier rather than emitting a
    degenerate avoiding row."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["synth_client"] = _StubSynthClient(responses=["RICH", "AVOID-OK"])
    # Natural "cont" would PASS strict-zero (inactive), but it adds no tokens
    # (NO_TOKENS); the LLM avoid "AVOID-OK" adds tokens and is inactive (accepted).
    stub_env["fc_active_by_continuation"] = {"cont": False, "AVOID-OK": False}
    stub_env["adds_tokens_by_continuation"] = {"cont": False}
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    ((_, avoid),) = _split_pairs(rows)
    # Routed to the LLM tier even though the natural continuation was inactive.
    assert avoid["output_text"] == "AVOID-OK"
    assert avoid["gen_source"].startswith("openrouter:")
    assert len(stub_env["synth_client"].calls) == 2


def test_empty_natural_continuation_routes_to_llm(stub_env, tmp_path):
    """(d) Empty/whitespace natural continuation -> skip natural -> LLM tier.

    Guards the QC trap: an empty continuation never reaches the strict-zero
    classifier (the natural tier short-circuits on `natural.strip()`), so it can
    never read as a false 'pass'. We must NOT emit an empty output_text; route
    to the LLM tier instead.
    """
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    # Natural continuation is whitespace-only.
    seed = _seed_row(0, text="prefix", sae_ids=[])
    seed = SeedRow(
        shard_idx=seed.shard_idx,
        doc_idx=seed.doc_idx,
        source_nanochat_token_count=seed.source_nanochat_token_count,
        prefix_source_token_start=seed.prefix_source_token_start,
        prefix_source_token_end=seed.prefix_source_token_end,
        prefix_text=seed.prefix_text,
        prefix_content_nanochat_tokens=seed.prefix_content_nanochat_tokens,
        sae_feature_ids=seed.sae_feature_ids,
        sae_feature_mags=seed.sae_feature_mags,
        continuation_text="   \n\t  ",
        continuation_content_nanochat_tokens=(seed.continuation_content_nanochat_tokens),
    )
    stub_env["synth_client"] = _StubSynthClient(responses=["RICH", "AVOID-OK"])
    # The whitespace continuation would "pass" strict-zero QC (False), but we
    # must reject it BEFORE QC. The LLM continuation "AVOID-OK" passes.
    stub_env["fc_active_by_continuation"] = {"AVOID-OK": False}
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        [seed],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    ((_, avoid),) = _split_pairs(rows)
    assert avoid["output_text"] == "AVOID-OK"
    assert avoid["output_text"].strip() != ""
    assert avoid["gen_source"].startswith("openrouter:")
    # The whitespace natural continuation must never reach strict-zero QC.
    assert all(c["continuation"].strip() != "" for c in stub_env["avoid_qc_calls"])
    # rich + avoiding LLM fallback = 2 calls.
    assert len(stub_env["synth_client"].calls) == 2


def test_avoid_qc_exhaustion_error(stub_env, tmp_path):
    """(e) error mode: avoid LLM exhausts strict-zero QC -> raise."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="error",
    )
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["RICH", "AVOID-1", "AVOID-2"],
    )
    # natural "cont" fires; every avoid attempt also fires -> exhausted.
    stub_env["fc_active_default"] = True
    concept = _concept(concept_id=3, feature_index=42)
    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*avoid_discards=1",
    ):
        _per_concept_positives(
            cfg,
            concept,
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )


def test_llm_tier_rejects_nonempty_no_tokens_response(stub_env, tmp_path):
    """LLM tier: a NON-empty avoid response that classifies NO_TOKENS must be
    rejected (``continue``), never emitted as a degenerate row.

    Drives the natural tier to fail (the seed continuation fires the feature),
    then the LLM tier sees two responses: the first ("AVOID-NOTOK") is non-empty
    but adds no tokens beyond the prefix (NO_TOKENS -> a false strict-zero PASS),
    so it must be skipped; the second ("AVOID-OK") is inactive and is emitted.
    This pins the non-empty NO_TOKENS branch (the key output-coverage gap): it
    ``continue``s rather than emitting "AVOID-NOTOK".
    """
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
    )
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["RICH", "AVOID-NOTOK", "AVOID-OK"],
    )
    # Natural "cont" fires (ACTIVE -> rejected -> LLM tier). The LLM "AVOID-NOTOK"
    # adds no tokens (NO_TOKENS -> rejected); "AVOID-OK" is inactive (accepted).
    stub_env["fc_active_by_continuation"] = {"cont": True, "AVOID-OK": False}
    stub_env["adds_tokens_by_continuation"] = {"AVOID-NOTOK": False}
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    ((_, avoid),) = _split_pairs(rows)
    # The NO_TOKENS response was rejected: the emitted row is the later INACTIVE
    # response, NOT the degenerate "AVOID-NOTOK".
    assert avoid["output_text"] == "AVOID-OK"
    assert avoid["gen_source"].startswith("openrouter:")
    # 3 synth calls: RICH + 2 avoid attempts (NO_TOKENS reject, then INACTIVE).
    assert len(stub_env["synth_client"].calls) == 3
    # Sanity: the NO_TOKENS response must never be emitted as an output_text.
    assert all(r["output_text"] != "AVOID-NOTOK" for r in rows)


def _read_counter_totals(out_dir) -> dict[str, float]:
    """Read aggregate counter totals from a finished profiler's summary.yaml.

    Uses the same public surface the profiler tests exercise (the dumped
    ``summary.yaml`` ``counters`` list of ``{name, count, total}``), rather
    than reaching into the profiler's private ``_counter_aggregates``.
    """
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    return {c["name"]: c["total"] for c in summary["counters"]}


def test_avoid_qc_counter_mapping_per_outcome(stub_env, tmp_path):
    """Each avoid-QC outcome increments exactly its mapped counter.

    Runs three single-seed ``_per_concept_positives`` calls inside one shared
    ``AxBenchProfiler`` (one call per scenario, so the per-call synth FIFO and
    seed-selection order are unambiguous), then reads the aggregate counter
    totals back from ``summary.yaml`` and asserts the exact mapping:

    * natural-INACTIVE accept -> ``stage1b.avoid_natural_accepts`` (+1), and
      NOT ``stage1b.avoid_natural_rejects``;
    * natural-ACTIVE then LLM-INACTIVE accept -> ``stage1b.avoid_natural_rejects``
      (+1) and ``stage1b.avoid_llm_accepts`` (+1);
    * natural-ACTIVE then LLM-NO_TOKENS then LLM-INACTIVE accept ->
      another ``stage1b.avoid_natural_rejects`` (+1), one
      ``stage1b.avoid_llm_empty_rejects`` (+1, the NO_TOKENS attempt), and
      another ``stage1b.avoid_llm_accepts`` (+1);
    * natural-NO_TOKENS then LLM-INACTIVE accept -> the natural NO_TOKENS
      fall-through records NEITHER ``stage1b.avoid_natural_accepts`` NOR
      ``stage1b.avoid_natural_rejects`` (the exact control-flow branch changed
      by #513), and the avoid row is the LLM one (``stage1b.avoid_llm_accepts``
      +1).
    """
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
    )
    concept = _concept(concept_id=5, feature_index=42)

    def _run_one_seed() -> list[dict[str, Any]]:
        return _per_concept_positives(
            cfg,
            concept,
            [_seed_row(0, text="prefix", sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )

    out_dir = tmp_path / "profile"
    with AxBenchProfiler(out_dir):
        # Scenario 1: natural continuation inactive -> natural-tier accept.
        stub_env["synth_client"] = _StubSynthClient(responses=["RICH"])
        stub_env["fc_active_by_continuation"] = {"cont": False}
        stub_env["adds_tokens_by_continuation"] = {}
        ((_, avoid1),) = _split_pairs(_run_one_seed())
        assert avoid1["output_text"] == "cont"
        assert avoid1["gen_source"].startswith("corpus:")

        # Scenario 2: natural fires -> reject; LLM continuation inactive -> accept.
        stub_env["synth_client"] = _StubSynthClient(responses=["RICH", "AVOID-OK"])
        stub_env["fc_active_by_continuation"] = {"cont": True, "AVOID-OK": False}
        stub_env["adds_tokens_by_continuation"] = {}
        ((_, avoid2),) = _split_pairs(_run_one_seed())
        assert avoid2["output_text"] == "AVOID-OK"
        assert avoid2["gen_source"].startswith("openrouter:")

        # Scenario 3: natural fires -> reject; LLM NO_TOKENS attempt rejected;
        # next LLM continuation inactive -> accept.
        stub_env["synth_client"] = _StubSynthClient(
            responses=["RICH", "AVOID-NOTOK", "AVOID-OK"],
        )
        stub_env["fc_active_by_continuation"] = {"cont": True, "AVOID-OK": False}
        stub_env["adds_tokens_by_continuation"] = {"AVOID-NOTOK": False}
        ((_, avoid3),) = _split_pairs(_run_one_seed())
        assert avoid3["output_text"] == "AVOID-OK"

        # Scenario 4: natural continuation itself classifies NO_TOKENS (the
        # natural NO_TOKENS fall-through, the exact branch #513 changed). It
        # must record NEITHER natural counter. The natural "cont" is INACTIVE
        # (would otherwise be accepted), but adds no tokens -> NO_TOKENS ->
        # fall through to the LLM tier, where "AVOID-OK" is inactive -> accept.
        stub_env["synth_client"] = _StubSynthClient(responses=["RICH", "AVOID-OK"])
        stub_env["fc_active_by_continuation"] = {"cont": False, "AVOID-OK": False}
        stub_env["adds_tokens_by_continuation"] = {"cont": False}
        ((_, avoid4),) = _split_pairs(_run_one_seed())
        # The avoid row is the LLM one, NOT the degenerate NO_TOKENS natural.
        assert avoid4["output_text"] == "AVOID-OK"
        assert avoid4["gen_source"].startswith("openrouter:")

    totals = _read_counter_totals(out_dir)
    # One natural-INACTIVE accept (scenario 1). The scenario-4 natural
    # NO_TOKENS fall-through contributes +0 here (it is neither accept nor
    # reject).
    assert totals.get("stage1b.avoid_natural_accepts", 0.0) == 1.0
    # Two natural-ACTIVE rejects (scenarios 2 and 3). Scenario 4's natural
    # NO_TOKENS contributes +0 here too -- the natural NO_TOKENS branch records
    # NEITHER natural counter.
    assert totals.get("stage1b.avoid_natural_rejects", 0.0) == 2.0
    # One NO_TOKENS LLM attempt rejected (scenario 3's first avoid attempt).
    # Scenario 4's NO_TOKENS is on the NATURAL tier, so it does NOT bump the
    # LLM-empty counter.
    assert totals.get("stage1b.avoid_llm_empty_rejects", 0.0) == 1.0
    # Three LLM-INACTIVE accepts (scenarios 2, 3, and 4).
    assert totals.get("stage1b.avoid_llm_accepts", 0.0) == 3.0
    # No LLM-ACTIVE rejects in any scenario here.
    assert totals.get("stage1b.avoid_llm_rejects", 0.0) == 0.0


def test_avoid_qc_receives_cfg_extraction_threshold(stub_env, tmp_path):
    """``_per_concept_positives`` threads ``cfg.extraction_threshold`` into the
    avoid classifier.

    Uses a non-default threshold so the assertion is non-vacuous: the recorded
    ``extraction_threshold`` on the natural-tier avoid-QC call must equal the
    cfg value, not the ``0.0`` default.
    """
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        extraction_threshold=0.375,
    )
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    # Natural continuation is inactive -> natural-tier accept (one avoid-QC call).
    stub_env["fc_active_by_continuation"] = {"cont": False}
    concept = _concept(concept_id=3, feature_index=42)
    _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    assert stub_env["avoid_qc_calls"]
    assert {call["extraction_threshold"] for call in stub_env["avoid_qc_calls"]} == {
        cfg.extraction_threshold
    }


def test_avoid_qc_exhaustion_warn_discards_atomic_pair(stub_env, tmp_path):
    """Warn mode never exposes the accepted rich half of a failed pair."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
    )
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["RICH", "AVOID-1", "AVOID-2"],
    )
    stub_env["fc_active_default"] = True
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    assert rows == []


def test_avoid_qc_exhaustion_best_treated_as_skip(stub_env, tmp_path):
    """(e) best mode: strict-zero QC has no 'best'; treated as skip."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="best",
    )
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["RICH", "AVOID-1", "AVOID-2"],
    )
    stub_env["fc_active_default"] = True
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    assert rows == []


def test_avoid_serial_parallel_byte_equivalence(stub_env, tmp_path):
    """(f) serial-vs-parallel byte-equivalence including avoid rows."""
    seeds = [_seed_row(i, text=f"seed-{i}", sae_ids=[]) for i in range(3)]
    concept = _concept(concept_id=5, feature_index=42)
    # "cont" is the shared natural continuation; make it fire so every avoid
    # side falls back to the LLM, then accept the LLM continuation. The two
    # paths consume LLM responses in DIFFERENT orders (serial interleaves
    # rich/avoid; parallel batches rich then avoid), so the stub must key on
    # cache_key (which encodes the rich vs avoid tag + seed) rather than a
    # shared FIFO queue -- exactly like the production cache.
    cfg_model = make_valid_config().synth.model
    response_by_key: dict[str, str] = {}
    for s in range(3):
        rich_prompt = positive_mode._stage1b_continuation_request(
            TEST_STAGE1_TEMPLATE,
            concept=concept,
            prefix_text=f"seed-{s}",
            seed_idx=s,
            attempt=0,
            content_class="concept_rich",
            cluster_target=None,
        ).prompt
        avoid_prompt = positive_mode._stage1b_continuation_request(
            TEST_STAGE1_TEMPLATE,
            concept=concept,
            prefix_text=f"seed-{s}",
            seed_idx=s,
            attempt=0,
            content_class="concept_avoiding",
            cluster_target=None,
        ).prompt
        response_by_key[
            positive_mode.make_cache_key(
                model=cfg_model,
                prompt=rich_prompt,
                tag=f"pos_c5_s{s}_a0",
            )
        ] = f"RICH-{s}"
        response_by_key[
            positive_mode.make_cache_key(
                model=cfg_model,
                prompt=avoid_prompt,
                tag=f"pos_avoid_c5_s{s}_a0",
            )
        ] = f"AVOID-{s}"

    class _KeyedClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            self.calls.append({"prompt": prompt, "cache_key": cache_key})
            assert cache_key in response_by_key, f"unexpected cache_key {cache_key}"
            return frame_stage1_response(prompt, response_by_key[cache_key])

    def _make_client():
        return _KeyedClient()

    fc_by_cont = {
        "cont": True,
        "AVOID-0": False,
        "AVOID-1": False,
        "AVOID-2": False,
    }

    serial_cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        accept_threshold=0.5,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    parallel_cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        accept_threshold=0.5,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=3),
    )

    stub_env["fc_active_by_continuation"] = dict(fc_by_cont)
    stub_env["synth_client"] = _make_client()
    serial_rows = _per_concept_positives(
        serial_cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    stub_env["fc_active_by_continuation"] = dict(fc_by_cont)
    stub_env["synth_client"] = _make_client()
    parallel_rows = _per_concept_positives(
        parallel_cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert _rows_byte_equal(serial_rows, parallel_rows)
    # Sanity: pairs emitted, avoid rows came from the LLM tier.
    pairs = _split_pairs(serial_rows)
    assert len(pairs) == 3
    for _, avoid in pairs:
        assert avoid["gen_source"].startswith("openrouter:")


def _all_clean_candidate_order(
    cfg,
    *,
    count: int,
    concept_id: int = 0,
) -> list[int]:
    indices = list(range(count))
    random.Random(
        cfg.generation.master_seed
        + positive_mode._CONCEPT_SEED_OFFSET
        + concept_id
    ).shuffle(indices)
    return indices


def test_rich_exhaustion_refills_until_complete_pair_target(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=2,
        positive_max_attempts_per_seed=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    seeds = [
        _seed_row(idx, text=f"candidate-{idx}", sae_ids=[])
        for idx in range(4)
    ]
    order = _all_clean_candidate_order(cfg, count=len(seeds))
    rejected = {f"candidate-{idx}" for idx in order[:2]}

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda prefix, *_args, **_kwargs: (
            0.0 if prefix in rejected else 80.0
        ),
    )

    rows = _per_concept_positives(
        cfg,
        _concept(),
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    rich = [
        row for row in rows if row["content_class"] == "concept_rich"
    ]
    assert [row["prefix_text"] for row in rich] == [
        f"candidate-{idx}" for idx in order[2:]
    ]
    assert len(rows) == 4
    assert len(stub_env["synth_client"].calls) == 4
    assert all(row["qc_accepted"] for row in rich)


def test_clean_exhaustion_refills_from_headroom_safe_dirty_stream(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    seeds = [
        _seed_row(0, text="clean", sae_ids=[]),
        _seed_row(1, text="dirty", sae_ids=[42]),
    ]
    stub_env["prefix_percentiles_by_text"] = {"dirty": 20.0}
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda prefix, *_args, **_kwargs: (
            0.0 if prefix == "clean" else 80.0
        ),
    )

    rows = _per_concept_positives(
        cfg,
        _concept(),
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
        prefix_cache=positive_mode.PrefixPercentileCache(
            base_model=object(),
            base_sae=object(),
            percentile_table=t.zeros(64, 99),
            sae_layer=8,
            tokenizer_id=cfg.extraction_tokenizer_id,
            percentile_namespace="test",
            batch_size=cfg.qc_batch_size,
        ),
    )

    assert [row["prefix_text"] for row in rows] == ["dirty", "dirty"]
    assert len(stub_env["synth_client"].calls) == 2


def test_avoid_exhaustion_discards_rich_and_refills_atomic_pair(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=2,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    seeds = [
        _seed_row(idx, text=f"candidate-{idx}", sae_ids=[])
        for idx in range(2)
    ]
    order = _all_clean_candidate_order(cfg, count=len(seeds))
    rejected_prefix = f"candidate-{order[0]}"

    def _classify(
        prefix_text,
        continuation_text,
        *_args,
        **_kwargs,
    ):
        del continuation_text
        if prefix_text == rejected_prefix:
            return positive_mode.AvoidContinuationOutcome.ACTIVE
        return positive_mode.AvoidContinuationOutcome.INACTIVE

    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuation",
        _classify,
    )

    rows = _per_concept_positives(
        cfg,
        _concept(),
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert [row["prefix_text"] for row in rows] == [
        f"candidate-{order[1]}",
        f"candidate-{order[1]}",
    ]
    # First candidate: rich + two failed LLM avoids. Replacement: rich only,
    # because its natural avoiding continuation succeeds without an API call.
    assert len(stub_env["synth_client"].calls) == 4


def test_zero_hit_guard_success_at_boundary_lifts_and_refills_deficits(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        positive_max_attempts_per_seed=1,
        positive_zero_hit_bail_after=2,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    seeds = [
        _seed_row(idx, text=f"candidate-{idx}", sae_ids=[])
        for idx in range(5)
    ]
    order = _all_clean_candidate_order(cfg, count=len(seeds))
    first_prefix = f"candidate-{order[0]}"

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda prefix, *_args, **_kwargs: (
            0.0 if prefix == first_prefix else 80.0
        ),
    )

    rows = _per_concept_positives(
        cfg,
        _concept(),
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    rich_prefixes = [
        row["prefix_text"]
        for row in rows
        if row["content_class"] == "concept_rich"
    ]
    assert rich_prefixes == [
        f"candidate-{idx}" for idx in order[1:4]
    ]
    assert len(stub_env["synth_client"].calls) == 4


def test_zero_hit_guard_never_admits_candidate_beyond_boundary(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=2,
        positive_max_attempts_per_seed=1,
        positive_zero_hit_bail_after=1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    stub_env["percentile_pairs"] = [(0.0, 0.0)] * 3

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(idx, text=f"candidate-{idx}") for idx in range(3)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert rows == []
    assert len(stub_env["synth_client"].calls) == 1


def test_error_shortfall_cancels_blocked_sibling_immediately(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        qc_batch_size=1,
        qc_fallback_on_exhaust="error",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    concepts = [_concept(0, 42), _concept(1, 43)]
    seeds = [_seed_row(0, text="only-candidate")]
    effective_prefixes = positive_mode._resolve_stage1b_prefixes(
        cfg,
        seeds,
    )
    trackers = []
    for concept in concepts:
        source = positive_mode._PositiveCandidateSource(
            concept,
            seeds,
            cfg,
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            prefix_cache=_FakePrefixCache({"only-candidate": 0.0}),
            cluster_target=None,
            effective_prefixes=effective_prefixes,
        )
        source.preflight_initial_target(1)
        trackers.append(
            positive_mode._PositiveTargetTracker(
                concept=concept,
                source=source,
                target=1,
                zero_hit_bail_after=None,
                cluster_target=None,
                started_at=0.0,
            )
        )

    blocked_started = threading.Event()
    release_blocked = threading.Event()

    class _BlockedSecondConceptClient:
        def __init__(self) -> None:
            self.descriptions: list[str] = []
            self.lock = threading.Lock()

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del cache_key, system_prompt
            description = next(
                line.removeprefix("DESCRIPTION=")
                for line in prompt.splitlines()
                if line.startswith("DESCRIPTION=")
            )
            with self.lock:
                self.descriptions.append(description)
            if description == "concept 1":
                blocked_started.set()
                assert release_blocked.wait(timeout=10)
            else:
                assert blocked_started.wait(timeout=5)
            return frame_stage1_response(
                prompt,
                "deterministic continuation",
            )

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda _prefix, _continuation, feature_id, *_args, **_kwargs: (
            0.0 if feature_id == 42 else 80.0
        ),
    )
    client = _BlockedSecondConceptClient()
    failures: list[BaseException] = []
    finished = threading.Event()

    def _invoke() -> None:
        try:
            positive_mode._run_positive_target_refill_engine(
                cfg,
                trackers,
                base_model=object(),
                base_sae=object(),
                llm_client=client,
                percentile_table=None,
                target_model_meta=_stub_meta(),
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    runner = threading.Thread(target=_invoke)
    runner.start()
    try:
        assert finished.wait(timeout=5), "error shortfall drained sibling work"
        assert not release_blocked.is_set()
        assert len(failures) == 1
        assert isinstance(failures[0], AxBenchBaseError)
        assert "complete-pair target shortfall" in str(failures[0])
        assert "concept_id=0" in str(failures[0])
        assert trackers[0].discarded_rich == 1
        assert trackers[0].termination_reason == "pool_exhausted"
        assert trackers[1].semantic_rich_attempts == 0
        assert sorted(client.descriptions) == ["concept 0", "concept 1"]
    finally:
        release_blocked.set()
        runner.join(timeout=5)
        assert not runner.is_alive()


def test_transport_exhaustion_with_replacement_pool_aborts_without_refill(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["synth_client"].raise_for = AxBenchBaseError(
        "terminal transport failure"
    )
    counters: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        positive_mode,
        "record_counter",
        lambda name, value=1, **attrs: counters.append(
            (name, value, attrs)
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"provider transport exhausted.*semantic_attempt=1",
    ):
        _per_concept_positives(
            cfg,
            _concept(),
            [_seed_row(0), _seed_row(1)],
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )

    assert len(stub_env["synth_client"].calls) == 1
    assert (
        "stage1b.summary.terminal",
        1,
        {"concept_id": 0, "termination_reason": "stage_abort"},
    ) in counters
    assert (
        "stage1b.summary.rich_semantic_attempts",
        0,
        {"concept_id": 0, "termination_reason": "stage_abort"},
    ) in counters


def test_transport_abort_rerun_reuses_durable_cache_and_matches_uninterrupted(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """A warm-cache rerun is byte-identical to uninterrupted synthesis."""

    class _DurableCachingClient:
        def __init__(
            self,
            response_cache,
            *,
            fail_prompt_fragment: str | None,
        ) -> None:
            self.response_cache = response_cache
            self.fail_prompt_fragment = fail_prompt_fragment
            self.successful_misses = 0
            self.cache_hits = 0

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            assert cache_key is not None
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return cached
            if (
                self.fail_prompt_fragment is not None
                and self.fail_prompt_fragment in prompt
            ):
                raise AxBenchBaseError(
                    "synthetic terminal transport outage"
                )
            response = frame_stage1_response(
                prompt,
                "durably cached continuation",
            )
            self.response_cache.set(cache_key, response)
            self.successful_misses += 1
            return response

    clients: list[_DurableCachingClient] = []

    def build_client(_cfg, response_cache=None):
        assert response_cache is not None
        client = _DurableCachingClient(
            response_cache,
            fail_prompt_fragment="seed-1" if not clients else None,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(positive_mode, "_build_synth_client", build_client)
    seeds = [_seed_row(i, text=f"seed-{i}", sae_ids=[]) for i in range(2)]
    concept = _concept()
    interrupted_cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "interrupted"),
        n_positives_per_concept=2,
        qc_batch_size=16,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="provider transport exhausted",
    ):
        synthesize_positive_mode(
            interrupted_cfg,
            [concept],
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    interrupted_paths = paths.AxBenchBasePaths.from_config(interrupted_cfg)
    assert not interrupted_paths.synth_training_positive().exists()
    assert clients[0].successful_misses == 1
    checkpoint = json.loads(
        interrupted_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    tracker_state = checkpoint["concept_states"][0]["tracker"]
    assert checkpoint["accepted_pair_count"] == 0
    assert tracker_state["semantic_rich_attempts"] == 0
    assert len(tracker_state["outstanding"]) == 2
    prefix_score_calls_before_resume = len(stub_env["prefix_cache_calls"])

    resumed_out = synthesize_positive_mode(
        interrupted_cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    # The batch transaction applied neither sibling before the fatal result.
    # Resume replays both outstanding chains: the authenticated successful
    # sibling is recovered from its durable cache, while only the failed
    # request reaches the provider.
    assert clients[1].cache_hits == 1
    assert clients[1].successful_misses == 1
    assert len(stub_env["prefix_cache_calls"]) == prefix_score_calls_before_resume

    baseline_cfg = interrupted_cfg.replace(
        output_root=str(tmp_path / "baseline"),
    )
    baseline_out = synthesize_positive_mode(
        baseline_cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    baseline_paths = paths.AxBenchBasePaths.from_config(baseline_cfg)

    assert resumed_out.read_bytes() == baseline_out.read_bytes()
    assert (
        interrupted_paths.stage1b_concepts().read_bytes()
        == baseline_paths.stage1b_concepts().read_bytes()
    )
    assert (
        interrupted_paths.stage1b_concept_yield_report().read_bytes()
        == baseline_paths.stage1b_concept_yield_report().read_bytes()
    )


def test_outstanding_resume_replays_cache_committed_response_without_provider_call(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """A response cached before handler interruption is a resume cache hit."""

    class _CachingClient:
        def __init__(self, response_cache) -> None:
            self.response_cache = response_cache
            self.cache_hits = 0
            self.provider_calls = 0

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            assert cache_key is not None
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.provider_calls += 1
            response = frame_stage1_response(
                prompt,
                "cache committed before handler interruption",
            )
            self.response_cache.set(cache_key, response)
            return response

    clients: list[_CachingClient] = []

    def build_client(_cfg, response_cache=None):
        assert response_cache is not None
        client = _CachingClient(response_cache)
        clients.append(client)
        return client

    original_scorer = positive_mode.output_fc_percentile_for_continuation
    scorer_calls = 0

    def interrupt_first_scorer(*args, **kwargs):
        nonlocal scorer_calls
        scorer_calls += 1
        if scorer_calls == 1:
            raise KeyboardInterrupt("interrupt after durable response cache")
        return original_scorer(*args, **kwargs)

    monkeypatch.setattr(positive_mode, "_build_synth_client", build_client)
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        interrupt_first_scorer,
    )
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "cache-replay"),
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )

    with pytest.raises(
        KeyboardInterrupt,
        match="interrupt after durable response cache",
    ):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0, sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    assert clients[0].provider_calls == 1
    assert clients[0].cache_hits == 0

    out = synthesize_positive_mode(
        cfg,
        [_concept()],
        [_seed_row(0, sae_ids=[])],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert out.exists()
    assert clients[1].provider_calls == 0
    assert clients[1].cache_hits == 1


# ---------------------------------------------------------------------------
# synthesize_positive_mode outer entrypoint
# ---------------------------------------------------------------------------


def test_row_has_all_schema_fields(stub_env, tmp_path):
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, text="hello", sae_ids=[])]
    concepts = [_concept(concept_id=0, feature_index=42)]
    out = synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    # rich + paired avoiding row.
    assert len(rows) == 2
    row = rows[0]
    expected_fields = {
        "concept_id",
        "category",
        "content_class",
        "prefix_text",
        "output_text",
        "output_concept",
        "n_prefix_nanochat_tokens",
        "gen_source",
        "gen_seed",
        "qc_feature_id",
        "qc_seed_percentile",
        "qc_output_percentile",
        "qc_delta",
        "qc_attempts",
        "qc_accepted",
        "qc_zero_activation_verified",
    }
    assert set(row.keys()) == expected_fields
    assert row["concept_id"] == 0
    assert row["qc_feature_id"] == 42
    assert row["gen_source"].startswith("openrouter:")
    assert row["category"] == "positive"
    assert row["content_class"] == "concept_rich"
    assert row["qc_zero_activation_verified"] is False
    avoid = rows[1]
    assert avoid["content_class"] == "concept_avoiding"
    assert avoid["category"] == "negative"
    assert avoid["concept_id"] == 0
    assert avoid["prefix_text"] == row["prefix_text"]
    assert avoid["qc_zero_activation_verified"] is True
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    filtered = json.loads(output_paths.stage1b_concepts().read_text(encoding="utf-8"))
    report = json.loads(output_paths.stage1b_concept_yield_report().read_text(encoding="utf-8"))
    assert filtered == [concepts[0].__dict__]
    assert report["status"] == "ok"
    assert report["surviving_concept_ids"] == [0]
    assert report["concepts"][0]["attempted"] == 1
    assert report["concepts"][0]["accepted"] == 1
    assert report["concepts"][0]["dropped"] == 0


def test_replaced_seed_is_not_counted_as_yield_drop(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    seeds = [
        _seed_row(idx, text=f"candidate-{idx}", sae_ids=[])
        for idx in range(2)
    ]
    order = _all_clean_candidate_order(cfg, count=len(seeds))
    first_prefix = f"candidate-{order[0]}"
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda prefix, *_args, **_kwargs: (
            0.0 if prefix == first_prefix else 80.0
        ),
    )

    out = synthesize_positive_mode(
        cfg,
        [_concept()],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    rich = [
        row
        for row in pq.read_table(out).to_pylist()
        if row["content_class"] == "concept_rich"
    ]
    assert [row["prefix_text"] for row in rich] == [
        f"candidate-{order[1]}"
    ]
    report = json.loads(
        paths.AxBenchBasePaths.from_config(
            cfg
        ).stage1b_concept_yield_report().read_text(encoding="utf-8")
    )
    assert report["concepts"][0]["attempted"] == 1
    assert report["concepts"][0]["accepted"] == 1
    assert report["concepts"][0]["dropped"] == 0


def test_no_global_negatives_prepended(stub_env, tmp_path):
    """Stage 1b no longer loads or prepends the Stage 1a global pool.

    Every output row is a per-concept rich/avoid row; nothing has the
    legacy ``concept_id == -1`` sentinel.
    """
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, sae_ids=[])]
    concepts = [_concept()]
    out = synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    # rich + paired avoiding, both per-concept; no global-pool sentinel rows.
    assert len(rows) == 2
    assert all(row["concept_id"] == 0 for row in rows)
    assert [row["content_class"] for row in rows] == [
        "concept_rich",
        "concept_avoiding",
    ]


def test_raises_on_empty_concepts(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    seeds = [_seed_row(0)]
    with pytest.raises(AxBenchBaseError, match="empty concepts"):
        synthesize_positive_mode(
            cfg,
            [],
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_raises_on_empty_seeds(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    concepts = [_concept()]
    with pytest.raises(AxBenchBaseError, match="empty seeds"):
        synthesize_positive_mode(
            cfg,
            concepts,
            [],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_run_tag_none_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path).replace(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_unicode_seed_roundtrips(stub_env, tmp_path):
    """A seed with non-ASCII text should round-trip through parquet."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    seeds = [_seed_row(0, text="café résumé naïve 日本語", sae_ids=[])]
    concepts = [_concept()]
    out = synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    pos_rows = [r for r in rows if r["category"] == "positive"]
    assert "café" in pos_rows[0]["prefix_text"]
    assert "日本語" in pos_rows[0]["prefix_text"]


def test_multiple_concepts_in_order(stub_env, tmp_path):
    """Positive rows should be in concept_id order after the negatives."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, sae_ids=[])]
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]
    out = synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    pos_rows = [r for r in rows if r["category"] == "positive"]
    assert [r["concept_id"] for r in pos_rows] == [0, 1, 2]
    assert [r["qc_feature_id"] for r in pos_rows] == [10, 20, 30]


def test_stage1b_global_pool_is_byte_exact_under_reverse_concept_completion(
    stub_env,
    tmp_path,
):
    """Cross-concept overlap and completion order cannot affect publication."""
    seeds = [_seed_row(0, sae_ids=[])]
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]
    baseline_cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "forward"),
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=3),
    )
    reverse_cfg = baseline_cfg.replace(
        output_root=str(tmp_path / "reverse"),
    )

    baseline_client = _ShuffledConceptSynthClient(initial_parties=3)
    stub_env["synth_client"] = baseline_client
    stub_env["percentile_pairs"] = [(0.0, 80.0)] * len(concepts)
    stub_env["fc_active_by_continuation"] = {
        "cont": True,
        "deterministic continuation": False,
    }
    baseline_out = synthesize_positive_mode(
        baseline_cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    baseline_bytes = baseline_out.read_bytes()
    baseline_yield_bytes = (
        paths.AxBenchBasePaths.from_config(baseline_cfg).stage1b_concept_yield_report().read_bytes()
    )
    baseline_concepts_bytes = (
        paths.AxBenchBasePaths.from_config(baseline_cfg)
        .stage1b_concepts()
        .read_bytes()
    )

    parallel_client = _ShuffledConceptSynthClient(
        initial_parties=3,
        preferred_first="concept 2",
        preferred_completions=1,
    )
    stub_env["synth_client"] = parallel_client
    stub_env["percentile_pairs"] = [(0.0, 80.0)] * len(concepts)
    stub_env["fc_active_by_continuation"] = {
        "cont": True,
        "deterministic continuation": False,
    }
    parallel_out = synthesize_positive_mode(
        reverse_cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    parallel_bytes = parallel_out.read_bytes()
    parallel_yield_bytes = (
        paths.AxBenchBasePaths.from_config(reverse_cfg).stage1b_concept_yield_report().read_bytes()
    )
    parallel_concepts_bytes = (
        paths.AxBenchBasePaths.from_config(reverse_cfg)
        .stage1b_concepts()
        .read_bytes()
    )

    assert parallel_client.peak_in_flight == 3
    # Chain-local rich -> avoid transitions may interleave differently across
    # worker implementations; the artifact contract is independent of that
    # completion schedule.
    assert parallel_client.completion_order[0] == "concept 2"
    assert sorted(parallel_client.completion_order) == [
        "concept 0",
        "concept 0",
        "concept 1",
        "concept 1",
        "concept 2",
        "concept 2",
    ]
    assert parallel_bytes == baseline_bytes
    assert parallel_yield_bytes == baseline_yield_bytes
    assert parallel_concepts_bytes == baseline_concepts_bytes


def test_stage1b_replacement_artifacts_are_exact_across_failure_schedules(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """Rich failures/refills cannot make publication schedule-dependent."""

    class _ScheduledClient:
        def __init__(self, schedule: str) -> None:
            self.schedule = schedule
            self.initial_barrier = (
                None
                if schedule == "serial"
                else threading.Barrier(
                    4,
                    action=(
                        self._release_first_randomized
                        if schedule == "randomized"
                        else None
                    ),
                )
            )
            self.concept_one_completed = threading.Event()
            self.randomized_events: dict[str, threading.Event] = {}
            self.randomized_order: list[str] = []
            self.calls: list[str] = []
            self.completions: list[str] = []
            self.in_flight = 0
            self.peak_in_flight = 0
            self.lock = threading.Lock()

        def _release_first_randomized(self) -> None:
            with self.lock:
                self.randomized_order = sorted(
                    self.calls[:4],
                    key=lambda prompt: random.Random(prompt).random(),
                )
                for prompt in self.randomized_order:
                    self.randomized_events[prompt] = threading.Event()
                self.randomized_events[
                    self.randomized_order[0]
                ].set()

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del cache_key, system_prompt
            description = next(
                line.removeprefix("DESCRIPTION=")
                for line in prompt.splitlines()
                if line.startswith("DESCRIPTION=")
            )
            with self.lock:
                call_index = len(self.calls)
                self.calls.append(prompt)
                self.in_flight += 1
                self.peak_in_flight = max(
                    self.peak_in_flight,
                    self.in_flight,
                )
            try:
                if (
                    self.initial_barrier is not None
                    and call_index < self.initial_barrier.parties
                ):
                    self.initial_barrier.wait(timeout=5)
                if (
                    self.schedule == "reverse"
                    and description == "concept 0"
                    and not self.concept_one_completed.wait(timeout=5)
                ):
                    raise AssertionError("concept 1 did not complete first")
                if self.schedule == "randomized" and call_index < 4:
                    if not self.randomized_events[prompt].wait(timeout=5):
                        raise AssertionError("randomized call was not released")
                response = frame_stage1_response(
                    prompt,
                    "deterministic continuation",
                )
                with self.lock:
                    self.completions.append(prompt)
                    if description == "concept 1":
                        self.concept_one_completed.set()
                    if self.schedule == "randomized" and call_index < 4:
                        position = self.randomized_order.index(prompt)
                        if position + 1 < len(self.randomized_order):
                            self.randomized_events[
                                self.randomized_order[position + 1]
                            ].set()
                return response
            finally:
                with self.lock:
                    self.in_flight -= 1

    seeds = [
        _seed_row(idx, text=f"candidate-{idx}", sae_ids=[])
        for idx in range(6)
    ]
    concepts = [_concept(0, 10), _concept(1, 20)]
    base_cfg = _cfg(
        tmp_path,
        n_positives_per_concept=2,
        positive_max_attempts_per_seed=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )
    rejected = {
        (
            concept.feature_index,
            f"candidate-{_all_clean_candidate_order(
                base_cfg,
                count=len(seeds),
                concept_id=concept.concept_id,
            )[0]}",
        )
        for concept in concepts
    }
    def _scheduled_output_percentile(
        prefix,
        _continuation,
        feature_id,
        *_args,
        **_kwargs,
    ):
        return (
            0.0 if (feature_id, prefix) in rejected else 80.0
        )

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        _scheduled_output_percentile,
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        lambda inputs, *_args, **_kwargs: [
            _scheduled_output_percentile(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
            )
            for item in inputs
        ],
    )

    artifacts: dict[str, tuple[bytes, bytes, bytes]] = {}
    clients: dict[str, _ScheduledClient] = {}
    for schedule in ("serial", "reverse", "randomized", "simultaneous"):
        client = _ScheduledClient(schedule)
        clients[schedule] = client
        stub_env["synth_client"] = client
        cfg = base_cfg.replace(
            output_root=str(tmp_path / schedule),
            synth=SynthConfig(
                requests_per_minute=0,
                max_concurrent=4,
            ),
        )
        out = synthesize_positive_mode(
            cfg,
            concepts,
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
        output_paths = paths.AxBenchBasePaths.from_config(cfg)
        artifacts[schedule] = (
            out.read_bytes(),
            output_paths.stage1b_concepts().read_bytes(),
            output_paths.stage1b_concept_yield_report().read_bytes(),
        )

    assert clients["serial"].peak_in_flight >= 1
    assert clients["simultaneous"].peak_in_flight == 4
    assert clients["reverse"].peak_in_flight > 1
    assert clients["randomized"].peak_in_flight > 1
    assert "DESCRIPTION=concept 1" in clients["reverse"].completions[0]
    assert all(len(client.calls) == 6 for client in clients.values())
    assert all(
        artifact == artifacts["serial"]
        for artifact in artifacts.values()
    )


@pytest.mark.parametrize(
    ("max_concurrent", "expected_release_count"),
    [(1, 6), (3, 4)],
)
def test_synthesize_positive_mode_releases_cuda_at_serial_concept_boundaries(
    stub_env,
    tmp_path,
    monkeypatch,
    max_concurrent,
    expected_release_count,
):
    """Serial Stage 1b restores per-concept cleanup; parallel cleans per pool."""
    calls: list[str] = []
    monkeypatch.setattr(
        positive_mode,
        "release_cuda_iteration_memory",
        lambda: calls.append("release"),
    )
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        synth=SynthConfig(
            requests_per_minute=0,
            max_concurrent=max_concurrent,
        ),
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]

    out = synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    rows = pq.read_table(out).to_pylist()
    pos_rows = [r for r in rows if r["content_class"] == "concept_rich"]
    assert [r["concept_id"] for r in pos_rows] == [0, 1, 2]
    assert calls == ["release"] * expected_release_count


def test_synthesize_positive_mode_releases_cuda_when_preparation_raises(
    stub_env,
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    def _raise_during_preparation(*args, **kwargs):
        del args, kwargs
        raise AxBenchBaseError("synthetic concept failure")

    monkeypatch.setattr(
        positive_mode,
        "release_cuda_iteration_memory",
        lambda: calls.append("release"),
    )
    monkeypatch.setattr(
        positive_mode,
        "_PositiveCandidateSource",
        _raise_during_preparation,
    )
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)

    with pytest.raises(AxBenchBaseError, match="synthetic concept failure"):
        synthesize_positive_mode(
            cfg,
            [_concept(0, 10)],
            [_seed_row(0, sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    assert calls == ["release"]


def test_parallel_positive_chain_pool_honors_cap_and_restores_seed_order(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Out-of-order synth completions still emit rows in selected-seed order."""
    synth_client = _ConcurrentSynthClient(delay_s=0.03)
    stub_env["synth_client"] = synth_client
    stub_env["percentile_pairs"] = [(0.0, 10.0)] * 5
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=5,
        accept_threshold=0.05,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    seeds = [_seed_row(i, text=f"seed-{i}", sae_ids=[]) for i in range(5)]

    rows = _per_concept_positives(
        cfg,
        _concept(),
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=synth_client,
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    rich_prefixes = [row["prefix_text"] for row in rows if row["content_class"] == "concept_rich"]
    expected_indices = list(range(5))
    random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET
    ).shuffle(expected_indices)
    assert rich_prefixes == [f"seed-{i}" for i in expected_indices]
    # Emission order interleaves [rich, avoiding] per seed.
    assert [row["content_class"] for row in rows] == [
        cc for _ in range(5) for cc in ("concept_rich", "concept_avoiding")
    ]
    assert synth_client.peak_in_flight <= 2
    assert synth_client.peak_in_flight > 1


def test_parallel_positive_outputs_match_serial_for_deterministic_fixture(
    stub_env,
    tmp_path,
):
    """max_concurrent=1 and >1 produce identical rows for deterministic fakes."""
    seeds = [_seed_row(i, text=f"seed-{i}", sae_ids=[]) for i in range(4)]
    concept = _concept()
    serial_cfg = _cfg(
        tmp_path,
        n_positives_per_concept=4,
        accept_threshold=0.05,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    parallel_cfg = _cfg(
        tmp_path,
        n_positives_per_concept=4,
        accept_threshold=0.05,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )

    stub_env["synth_client"] = _StubSynthClient(responses=["same"] * 10)
    stub_env["percentile_pairs"] = [(0.0, 10.0)] * 10
    serial_rows = _per_concept_positives(
        serial_cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    stub_env["synth_client"] = _StubSynthClient(responses=["same"] * 10)
    stub_env["percentile_pairs"] = [(0.0, 10.0)] * 10
    parallel_rows = _per_concept_positives(
        parallel_cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert _rows_byte_equal(parallel_rows, serial_rows)


def test_parallel_positive_retries_only_pending_seeds_with_prefix_cache(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Accepted seeds terminate their chain without speculative retries."""
    seeds = [
        _seed_row(0, text="clean-0", sae_ids=[]),
        _seed_row(1, text="clean-1", sae_ids=[]),
        _seed_row(2, text="dirty-2", sae_ids=[42]),
    ]
    concept = _concept()
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        accept_threshold=0.5,
        max_retries=1,
        extraction_tokenizer_id="custom/stage1b-parallel-tokenizer",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=3),
    )
    prefix_cache = _FakePrefixCache({"dirty-2": 0.0})

    prompt_for_seed = {
        seed.prefix_text: f"{seed.prefix_text}::{concept.description}" for seed in seeds
    }
    synth_client = _ScriptedConcurrentSynthClient(
        {
            prompt_for_seed["clean-0"]: ["clean-0 attempt-0"],
            prompt_for_seed["clean-1"]: ["clean-1 attempt-0", "clean-1 attempt-1"],
            prompt_for_seed["dirty-2"]: [
                "dirty-2 attempt-0",
                "dirty-2 attempt-1",
            ],
        },
        delay_s=0.03,
    )

    qc_calls: dict[str, int] = {}
    qc_tokenizer_ids: list[str] = []

    def _compute_output_percentile(
        prefix,
        continuation,
        f_c,
        gemma,
        sae,
        table,
        *,
        sae_layer,
        tokenizer_id,
    ):
        del continuation, f_c, gemma, sae, table, sae_layer
        qc_tokenizer_ids.append(tokenizer_id)
        attempt = qc_calls.get(prefix, 0)
        qc_calls[prefix] = attempt + 1
        if prefix == "clean-0":
            return 80.0
        if prefix == "clean-1":
            return 10.0 if attempt == 0 else 80.0
        if prefix == "dirty-2":
            return 10.0 if attempt == 0 else 80.0
        raise AssertionError(f"unexpected prefix={prefix!r}")

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        _compute_output_percentile,
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        lambda inputs, gemma, sae, table, *, sae_layer, tokenizer_id: [
            _compute_output_percentile(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                gemma,
                sae,
                table,
                sae_layer=sae_layer,
                tokenizer_id=tokenizer_id,
            )
            for item in inputs
        ],
    )

    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=synth_client,
        percentile_table=None,
        target_model_meta=_stub_meta(),
        prefix_cache=prefix_cache,
    )

    rich_prefixes = [row["prefix_text"] for row in rows if row["content_class"] == "concept_rich"]
    assert rich_prefixes == ["clean-0", "clean-1", "dirty-2"]
    assert qc_tokenizer_ids
    assert set(qc_tokenizer_ids) == {cfg.extraction_tokenizer_id}
    assert synth_client.peak_in_flight <= 3
    assert synth_client.peak_in_flight > 1
    assert len(synth_client.calls) == 5
    calls_by_seed = {
        seed: [
            call
            for call in synth_client.calls
            if call["prompt"].startswith(f"PREFIX={seed}\n")
        ]
        for seed in prompt_for_seed
    }
    assert len(calls_by_seed["clean-0"]) == 1
    assert len(calls_by_seed["clean-1"]) == 2
    assert len(calls_by_seed["dirty-2"]) == 2
    assert prefix_cache.calls == [
        {
            "prefix_texts": ["clean-0", "clean-1"],
            "f_c": concept.feature_index,
        },
        {"prefix_texts": ["dirty-2"], "f_c": concept.feature_index},
    ]


def test_post_provider_batch_combines_natural_and_llm_avoid_qc(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """One callback coalesces natural and already-LLM avoiding continuations."""
    rich_batches: list[list[tuple[str, str]]] = []
    avoid_batches: list[list[tuple[str, str]]] = []

    def _rich_batch(inputs, *_args, **_kwargs):
        rich_batches.append(
            [
                (item.prefix_text, item.continuation_text)
                for item in inputs
            ]
        )
        return [80.0] * len(inputs)

    def _avoid_batch(inputs, *_args, **_kwargs):
        batch = [
            (item.prefix_text, item.continuation_text)
            for item in inputs
        ]
        avoid_batches.append(batch)
        return [
            (
                positive_mode.AvoidContinuationOutcome.ACTIVE
                if prefix == "seed-0" and continuation == "cont"
                else positive_mode.AvoidContinuationOutcome.INACTIVE
            )
            for prefix, continuation in batch
        ]

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        _rich_batch,
    )
    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuations",
        _avoid_batch,
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scalar rich QC must not run")
        ),
    )
    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scalar avoid QC must not run")
        ),
    )

    def _complete(
        llm_client,
        requests,
        *,
        handle_results,
        result_batch_size,
        terminal_chain_refill,
        **_kwargs,
    ) -> None:
        assert result_batch_size == 16
        assert len(requests) == 2

        def _result(request):
            response = llm_client.complete(
                request.prompt,
                cache_key=request.cache_key,
            )
            return positive_mode.SynthResult(
                index=request.index,
                response_text=response,
            )

        first_rich, second_rich = requests
        first_avoid = handle_results(
            [(first_rich, _result(first_rich))]
        )[0]
        assert first_avoid is not None
        next_requests = handle_results(
            [
                (first_avoid, _result(first_avoid)),
                (second_rich, _result(second_rich)),
            ]
        )
        assert next_requests == [None, None]
        assert terminal_chain_refill(first_avoid) == []
        assert terminal_chain_refill(second_rich) == []

    monkeypatch.setattr(
        positive_mode,
        "complete_synth_request_chains",
        _complete,
    )
    rows = _per_concept_positives(
        _cfg(
            tmp_path,
            n_positives_per_concept=2,
            positive_max_attempts_per_seed=1,
            qc_batch_size=16,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
        ),
        _concept(),
        [
            _seed_row(0, text="seed-0", sae_ids=[]),
            _seed_row(1, text="seed-1", sae_ids=[]),
        ],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert [len(batch) for batch in rich_batches] == [1, 1]
    assert avoid_batches == [
        [("seed-0", "cont")],
        [
            ("seed-0", "canned continuation"),
            ("seed-1", "cont"),
        ],
    ]
    assert [row["content_class"] for row in rows] == [
        "concept_rich",
        "concept_avoiding",
        "concept_rich",
        "concept_avoiding",
    ]


@pytest.mark.parametrize("failure_phase", ["rich", "strict_zero"])
def test_batch_scoring_failure_checkpoints_whole_ready_batch_as_outstanding(
    stub_env,
    tmp_path,
    monkeypatch,
    failure_phase: str,
) -> None:
    """Neither GPU phase mutates tracker state before both phases succeed."""
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / failure_phase),
        n_positives_per_concept=2,
        positive_max_attempts_per_seed=1,
        qc_batch_size=16,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    concept = _concept()
    seeds = [
        _seed_row(0, text="seed-0", sae_ids=[]),
        _seed_row(1, text="seed-1", sae_ids=[]),
    ]
    rich_scorer = positive_mode.output_fc_percentiles_for_continuations
    strict_scorer = positive_mode.classify_avoid_continuations

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt(f"interrupt in {failure_phase} batch")

    if failure_phase == "rich":
        monkeypatch.setattr(
            positive_mode,
            "output_fc_percentiles_for_continuations",
            _interrupt,
        )
    else:
        monkeypatch.setattr(
            positive_mode,
            "classify_avoid_continuations",
            _interrupt,
        )

    with pytest.raises(KeyboardInterrupt, match="interrupt in"):
        synthesize_positive_mode(
            cfg,
            [concept],
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    checkpoint = json.loads(
        output_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    tracker_state = checkpoint["concept_states"][0]["tracker"]
    assert checkpoint["accepted_pair_count"] == 0
    assert len(tracker_state["outstanding"]) == 2
    assert tracker_state["semantic_rich_attempts"] == 0
    assert tracker_state["semantic_avoid_attempts"] == 0

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        rich_scorer,
    )
    monkeypatch.setattr(
        positive_mode,
        "classify_avoid_continuations",
        strict_scorer,
    )
    resumed = synthesize_positive_mode(
        cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    baseline_cfg = cfg.replace(output_root=str(tmp_path / f"{failure_phase}-baseline"))
    baseline = synthesize_positive_mode(
        baseline_cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert resumed.read_bytes() == baseline.read_bytes()


def test_batch_ordered_application_checkpoint_preserves_applied_prefix(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """An application error commits earlier pairs and replays the suffix."""
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "ordered-application"),
        n_positives_per_concept=2,
        positive_max_attempts_per_seed=1,
        qc_batch_size=16,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    concept = _concept()
    seeds = [
        _seed_row(0, text="seed-0", sae_ids=[]),
        _seed_row(1, text="seed-1", sae_ids=[]),
    ]
    original_apply = positive_mode._apply_positive_rich_result
    apply_calls = 0

    def _fail_before_second_apply(*args, **kwargs):
        nonlocal apply_calls
        apply_calls += 1
        if apply_calls == 2:
            raise RuntimeError("ordered application failure")
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        positive_mode,
        "_apply_positive_rich_result",
        _fail_before_second_apply,
    )
    with pytest.raises(RuntimeError, match="ordered application failure"):
        synthesize_positive_mode(
            cfg,
            [concept],
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    checkpoint = json.loads(
        output_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    tracker_state = checkpoint["concept_states"][0]["tracker"]
    assert checkpoint["accepted_pair_count"] == 1
    assert len(tracker_state["outstanding"]) == 1
    assert pq.read_table(output_paths.stage1b_partial_parquet()).num_rows == 2

    monkeypatch.setattr(
        positive_mode,
        "_apply_positive_rich_result",
        original_apply,
    )
    resumed = synthesize_positive_mode(
        cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert len(pq.read_table(resumed)) == 4
    baseline_cfg = cfg.replace(output_root=str(tmp_path / "ordered-baseline"))
    stub_env["synth_client"] = _StubSynthClient(
        responses=["canned continuation"] * 10,
    )
    baseline = synthesize_positive_mode(
        baseline_cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    resumed_paths = paths.AxBenchBasePaths.from_config(cfg)
    baseline_paths = paths.AxBenchBasePaths.from_config(baseline_cfg)
    assert resumed.read_bytes() == baseline.read_bytes()
    assert (
        resumed_paths.stage1b_concept_yield_report().read_bytes()
        == baseline_paths.stage1b_concept_yield_report().read_bytes()
    )
    assert (
        resumed_paths.stage1b_concepts().read_bytes()
        == baseline_paths.stage1b_concepts().read_bytes()
    )


def test_batch_terminal_mutations_checkpoint_before_scheduler_refills(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """A crash after batch application can resume without terminal refills."""
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "before-refills"),
        n_positives_per_concept=2,
        positive_max_attempts_per_seed=1,
        qc_batch_size=16,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    concept = _concept()
    seeds = [
        _seed_row(0, text="seed-0", sae_ids=[]),
        _seed_row(1, text="seed-1", sae_ids=[]),
    ]
    real_scheduler = positive_mode.complete_synth_request_chains

    def _interrupt_before_refills(
        llm_client,
        initial_requests,
        *,
        handle_results,
        terminal_chain_refill,
        **_kwargs,
    ) -> None:
        pairs = []
        for request in initial_requests:
            response = llm_client.complete(
                request.prompt,
                cache_key=request.cache_key,
            )
            pairs.append(
                (
                    request,
                    positive_mode.SynthResult(
                        index=request.index,
                        response_text=response,
                    ),
                )
            )
        assert handle_results(pairs) == [None, None]
        del terminal_chain_refill
        raise KeyboardInterrupt("after terminal mutations")

    monkeypatch.setattr(
        positive_mode,
        "complete_synth_request_chains",
        _interrupt_before_refills,
    )
    with pytest.raises(KeyboardInterrupt, match="after terminal mutations"):
        synthesize_positive_mode(
            cfg,
            [concept],
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    checkpoint = json.loads(
        output_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    assert checkpoint["accepted_pair_count"] == 2
    assert checkpoint["concept_states"][0]["tracker"]["outstanding"] == []

    monkeypatch.setattr(
        positive_mode,
        "complete_synth_request_chains",
        real_scheduler,
    )
    resumed_client = _StubSynthClient(responses=[])
    stub_env["synth_client"] = resumed_client
    resumed = synthesize_positive_mode(
        cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert len(pq.read_table(resumed)) == 4
    assert resumed_client.calls == []
    baseline_cfg = cfg.replace(output_root=str(tmp_path / "before-refills-baseline"))
    stub_env["synth_client"] = _StubSynthClient(
        responses=["canned continuation"] * 10,
    )
    baseline = synthesize_positive_mode(
        baseline_cfg,
        [concept],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    resumed_paths = paths.AxBenchBasePaths.from_config(cfg)
    baseline_paths = paths.AxBenchBasePaths.from_config(baseline_cfg)
    assert resumed.read_bytes() == baseline.read_bytes()
    assert (
        resumed_paths.stage1b_concept_yield_report().read_bytes()
        == baseline_paths.stage1b_concept_yield_report().read_bytes()
    )
    assert (
        resumed_paths.stage1b_concepts().read_bytes()
        == baseline_paths.stage1b_concepts().read_bytes()
    )


# ---------------------------------------------------------------------------
# _infer_n_sae_features (re-exported from extraction; fix 5: raises on broken sae)
# ---------------------------------------------------------------------------


class _FakeSAEConfigDSae:
    d_sae = 32768


class _FakeSAEConfigNFeatures:
    n_features = 65536


class _FakeSAENoCfg:
    pass


def test_infer_n_sae_features_from_d_sae():
    sae = type("_SAE", (), {"cfg": _FakeSAEConfigDSae()})()
    assert _infer_n_sae_features(sae) == 32768


def test_infer_n_sae_features_from_n_features():
    sae = type("_SAE", (), {"cfg": _FakeSAEConfigNFeatures()})()
    assert _infer_n_sae_features(sae) == 65536


def test_infer_n_sae_features_raises_when_no_cfg():
    sae = _FakeSAENoCfg()
    with pytest.raises(AxBenchBaseError, match="no .cfg attribute"):
        _infer_n_sae_features(sae)


def test_infer_n_sae_features_raises_when_cfg_has_neither():
    class _Cfg:
        pass

    sae = type("_SAE", (), {"cfg": _Cfg()})()
    with pytest.raises(AxBenchBaseError, match="neither d_sae nor n_features"):
        _infer_n_sae_features(sae)


# ---------------------------------------------------------------------------
# _build_synth_client direct test
# ---------------------------------------------------------------------------


def test_synthesize_positive_mode_opens_cache_and_passes_cache_key(
    stub_env,
    tmp_path,
):
    """Stage 1b must open the SynthResponseCache and pass cache_keys on each call."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, sae_ids=[])]
    concepts = [_concept(concept_id=0, feature_index=42)]
    synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    # The stub synth client recorded at least one call.
    calls = stub_env["synth_client"].calls
    assert len(calls) >= 1
    assert stub_env["captured_response_cache"] is not None
    # Every call received a non-None cache_key.
    for call in calls:
        assert call["cache_key"] is not None
        assert len(call["cache_key"]) == 64  # sha256 hex
    # The production cache is opened on disk and threaded to the factory.
    cache_file = paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()
    assert cache_file.exists()


def test_synthesize_positive_mode_prefix_cache_uses_cfg_tokenizer(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Stage 1b prefix cache is scoped to the configured extraction tokenizer."""
    del stub_env
    cfg = _cfg(tmp_path, extraction_tokenizer_id="nondefault/test-tokenizer")
    captured: dict[str, Any] = {}

    class _StubPrefixCache:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.batch_size = kwargs["batch_size"]

        def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
            del prefix_texts, f_c
            return []

    monkeypatch.setattr(positive_mode, "PrefixPercentileCache", _StubPrefixCache)
    with pytest.raises(AxBenchBaseError, match="prefix scorer returned 0 scores"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0, sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    assert captured["tokenizer_id"] == cfg.extraction_tokenizer_id


def test_synthesize_positive_mode_cache_key_includes_attempt(
    stub_env,
    tmp_path,
):
    """Retry attempts must get distinct cache_keys so QC retries fetch fresh."""
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=2,
    )
    seeds = [_seed_row(0, sae_ids=[])]
    concepts = [_concept()]
    # First attempt fails QC, second passes -> 2 LLM calls
    stub_env["percentile_pairs"] = [(10.0, 20.0), (10.0, 80.0)]
    synthesize_positive_mode(
        cfg,
        concepts,
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    calls = stub_env["synth_client"].calls
    assert len(calls) == 2
    # The two cache keys must differ (they encode a different attempt).
    assert calls[0]["cache_key"] != calls[1]["cache_key"]


def test_build_synth_client_constructs_real_client(monkeypatch):
    """The factory forwards only the independent HTTP retry budget."""
    captured: dict[str, Any] = {}

    class _StubClient:
        def __init__(self, synth_cfg, max_retries, response_cache=None):
            captured["synth_cfg"] = synth_cfg
            captured["max_retries"] = max_retries
            captured["response_cache"] = response_cache

    monkeypatch.setattr(positive_mode, "OpenRouterSynthClient", _StubClient)
    base = make_valid_config(max_retries=100)
    cfg = base.replace(
        synth=replace(base.synth, transport_max_retries=2),
    )
    # Default call: no cache forwarded.
    client = positive_mode._build_synth_client(cfg)
    assert isinstance(client, _StubClient)
    assert captured["max_retries"] == 2
    assert captured["synth_cfg"] is cfg.synth
    assert captured["response_cache"] is None

    # With a cache forwarded, the factory threads it through verbatim.
    sentinel_cache = object()
    positive_mode._build_synth_client(cfg, response_cache=sentinel_cache)
    assert captured["response_cache"] is sentinel_cache


def test_positive_attempt_cap_is_immediate_independent_and_cache_bounded(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """The Stage 1b cap is independent of legacy and transport retry budgets."""
    base = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=100,
        positive_max_attempts_per_seed=8,
        qc_fallback_on_exhaust="error",
    )
    cfg = base.replace(
        synth=replace(base.synth, transport_max_retries=1),
    )
    stub_env["percentile_pairs"] = [(0.0, 0.0)] * 8
    counters: list[tuple[str, int | float, dict[str, Any]]] = []

    def _forbid_sleep(_seconds: float) -> None:
        raise AssertionError("content/QC rejection sampling must not sleep")

    def _record(name: str, value: int | float = 1, **attrs: Any) -> None:
        counters.append((name, value, attrs))

    monkeypatch.setattr("time.sleep", _forbid_sleep)
    monkeypatch.setattr(positive_mode, "record_counter", _record)

    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*rich_discards=1",
    ):
        _per_concept_positives(
            cfg,
            _concept(),
            [_seed_row(0, sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )

    calls = stub_env["synth_client"].calls
    assert len(calls) == 8
    cache_keys = [call["cache_key"] for call in calls]
    assert None not in cache_keys
    assert len(set(cache_keys)) == 8
    assert (
        "stage1b.rich_qc_attempts_on_exhaustion",
        8,
        {
            "concept_id": 0,
            "seed_idx": 0,
            "outcome": "discarded",
        },
    ) in counters


def test_positive_avoid_attempt_cap_is_independently_eight(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=8,
        max_retries=0,
        qc_fallback_on_exhaust="error",
    )
    stub_env["fc_active_default"] = True
    counters: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        positive_mode,
        "record_counter",
        lambda name, value=1, **attrs: counters.append(
            (name, value, attrs)
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*avoid_discards=1",
    ):
        _per_concept_positives(
            cfg,
            _concept(),
            [_seed_row(0, sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )

    calls = stub_env["synth_client"].calls
    assert len(calls) == 9  # one accepted rich call plus eight avoid attempts
    avoid_cache_keys = [call["cache_key"] for call in calls[1:]]
    assert None not in avoid_cache_keys
    assert len(set(avoid_cache_keys)) == 8
    assert (
        "stage1b.avoid_qc_attempts_on_exhaustion",
        8,
        {
            "concept_id": 0,
            "seed_idx": 0,
            "outcome": "discarded",
        },
    ) in counters


class _RaiseOnAvoidAttemptClient:
    """Returns RICH first, raises on the first avoid attempt, then succeeds.

    Exercises the avoid-tier LLM-call exception handler (retry on
    AxBenchBaseError) without the unconditional raise of _StubSynthClient.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del system_prompt
        self.calls.append({"prompt": prompt, "cache_key": cache_key})
        n = len(self.calls)
        if n == 1:
            return frame_stage1_response(prompt, "RICH")
        if n == 2:
            raise AxBenchBaseError("avoid synth LLM transiently unavailable")
        return frame_stage1_response(prompt, "AVOID-OK")


def test_avoid_transport_exhaustion_aborts_without_semantic_retry(
    stub_env,
    tmp_path,
):
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5, max_retries=3)
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["synth_client"] = _RaiseOnAvoidAttemptClient()
    # Natural "cont" fires -> route to LLM; the LLM avoid "AVOID-OK" is inactive.
    stub_env["fc_active_by_continuation"] = {"cont": True, "AVOID-OK": False}
    concept = _concept(concept_id=3, feature_index=42)
    with pytest.raises(
        AxBenchBaseError,
        match=r"provider transport exhausted.*side=avoid.*semantic_attempt=1",
    ):
        _per_concept_positives(
            cfg,
            concept,
            seeds,
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )
    assert len(stub_env["synth_client"].calls) == 2


def test_avoid_llm_empty_continuation_rejected_then_retry(stub_env, tmp_path):
    """Avoid-tier LLM empty output is rejected (no degenerate row), then retried."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5, max_retries=3)
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    # rich -> "RICH"; avoid attempt 1 -> "" (rejected); avoid attempt 2 -> "AVOID-OK".
    stub_env["synth_client"] = _StubSynthClient(responses=["RICH", "", "AVOID-OK"])
    stub_env["fc_active_by_continuation"] = {"cont": True, "AVOID-OK": False}
    concept = _concept(concept_id=3, feature_index=42)
    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    ((_, avoid),) = _split_pairs(rows)
    assert avoid["output_text"] == "AVOID-OK"
    assert avoid["output_text"].strip() != ""


def test_avoid_path_calls_classify_once_per_continuation(stub_env, tmp_path):
    """Issue #513: each checked avoid continuation is classified exactly once
    (no separate adds-tokens + fc-active double call)."""
    cfg = _cfg(tmp_path, n_positives_per_concept=1, accept_threshold=0.5)
    seeds = [_seed_row(0, text="prefix", sae_ids=[])]
    stub_env["fc_active_by_continuation"] = {"cont": False}  # natural accepts
    concept = _concept(concept_id=3, feature_index=42)
    _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    natural_calls = [c for c in stub_env["avoid_qc_calls"] if c["continuation"] == "cont"]
    assert len(natural_calls) == 1


# ---------------------------------------------------------------------------
# Issue #1051: bounded Stage 1b prefix views and shared continuation guards
# ---------------------------------------------------------------------------


def test_resolve_stage1b_prefixes_applies_closed_bounds_counts_and_no_mutation(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    seeds = [
        _seed_row(0, text="ab"),
        _seed_row(1, text="abc"),
        _seed_row(2, text="abcd"),
        _seed_row(3, text="abcde"),
        _seed_row(4, text="HEAD😀e\u0301漢Z"),
    ]
    original_prefixes = [seed.prefix_text for seed in seeds]
    original_prefix_tokens = [list(seed.prefix_content_nanochat_tokens) for seed in seeds]
    profile_dir = tmp_path / "prefix-resolution-profile"

    with AxBenchProfiler(profile_dir):
        resolved = positive_mode._resolve_stage1b_prefixes(cfg, seeds)

    assert resolved == (None, "abc", "abcd", "bcde", "e\u0301漢Z")
    assert [seed.prefix_text for seed in seeds] == original_prefixes
    assert [seed.prefix_content_nanochat_tokens for seed in seeds] == original_prefix_tokens
    totals = _read_counter_totals(profile_dir)
    assert totals["stage1b.seed_prefixes_eligible"] == 4.0
    assert totals["stage1b.seed_prefixes_truncated"] == 2.0
    assert totals["stage1b.seed_prefixes_below_min"] == 1.0


def test_resolve_stage1b_prefixes_allows_empty_only_when_minimum_is_zero(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path, min_prefix_chars=0, max_prefix_chars=1)
    seeds = [_seed_row(0, text=""), _seed_row(1, text="xy")]

    resolved = positive_mode._resolve_stage1b_prefixes(cfg, seeds)

    assert resolved == ("", "y")


def test_resolve_stage1b_prefixes_fails_if_shared_window_breaks_postcondition(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path, min_prefix_chars=1, max_prefix_chars=4)
    seeds = [_seed_row(0, text="abcdef")]
    monkeypatch.setattr(
        positive_mode,
        "window_prefix_text",
        lambda _text, _end, _maximum: "five!",
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"seed_idx=0, length=5, expected \[1, 4\]",
    ):
        positive_mode._resolve_stage1b_prefixes(cfg, seeds)


def test_pick_seed_indices_rejects_misaligned_effective_prefix_table(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    seeds = [_seed_row(0), _seed_row(1)]

    with pytest.raises(
        AxBenchBaseError,
        match=r"prefix table length does not match.*1 != 2",
    ):
        _pick_seed_indices_for_concept(
            _concept(),
            seeds,
            _cfg(tmp_path),
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            effective_prefixes=("only-one",),
        )


def test_pick_seed_indices_defensively_rejects_all_ineligible_explicit_table(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    seeds = [_seed_row(0, text="short")]

    with pytest.raises(
        AxBenchBaseError,
        match=r"Concept 0: Stage 1b has no eligible positive seed prefixes",
    ):
        _pick_seed_indices_for_concept(
            _concept(),
            seeds,
            _cfg(tmp_path),
            base_model=object(),
            base_sae=object(),
            percentile_table=None,
            sae_layer=8,
            effective_prefixes=(None,),
        )


def test_pick_seed_indices_shuffles_original_indices_then_filters_ineligible(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=4)
    concept = _concept(concept_id=17)
    seeds = [_seed_row(idx, text=f"seed-{idx}") for idx in range(7)]
    effective = (
        "seed-0",
        None,
        "seed-2",
        None,
        "seed-4",
        "seed-5",
        "seed-6",
    )
    shuffled = list(range(len(seeds)))
    random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    ).shuffle(shuffled)
    expected = [idx for idx in shuffled if effective[idx] is not None][:4]

    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        effective_prefixes=effective,
    )

    assert picked == expected
    assert all(idx in {0, 2, 4, 5, 6} for idx in picked)


def test_dirty_prefix_cache_scores_only_bounded_tails_in_shuffled_order(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    concept = _concept(concept_id=9, feature_index=42)
    seeds = [
        _seed_row(0, text="HEAD-aaa-tail-0", sae_ids=[42]),
        _seed_row(1, text="HEAD-bbb-tail-1", sae_ids=[42]),
        _seed_row(2, text="HEAD-ccc-tail-2", sae_ids=[42]),
    ]
    effective = ("tail-0", "tail-1", "tail-2")
    cache = _FakePrefixCache(
        {"tail-0": 50.0, "tail-1": 10.0, "tail-2": 90.0},
    )
    shuffled = list(range(len(seeds)))
    random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    ).shuffle(shuffled)

    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=object(),
        base_sae=object(),
        percentile_table=None,
        sae_layer=8,
        prefix_cache=cache,
        effective_prefixes=effective,
    )

    assert cache.calls == [
        {
            "prefix_texts": [effective[idx] for idx in shuffled],
            "f_c": 42,
        }
    ]
    assert picked == [1, 0]
    assert all("HEAD" not in text for text in cache.calls[0]["prefix_texts"])


def test_uncached_dirty_scoring_receives_only_bounded_prefixes(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    concept = _concept(concept_id=11, feature_index=42)
    seeds = [
        _seed_row(0, text="unbounded-prefix-zero", sae_ids=[42]),
        _seed_row(1, text="unbounded-prefix-one", sae_ids=[42]),
        _seed_row(2, text="unbounded-prefix-two", sae_ids=[42]),
    ]
    effective = ("tail-zero", "tail-one", "tail-two")
    scores = {"tail-zero": 80.0, "tail-one": 5.0, "tail-two": 20.0}
    calls: list[dict[str, Any]] = []

    def _score(
        prefix_text,
        f_c,
        base_model,
        base_sae,
        percentile_table,
        *,
        sae_layer,
        tokenizer_id,
    ):
        calls.append(
            {
                "prefix_text": prefix_text,
                "f_c": f_c,
                "base_model": base_model,
                "base_sae": base_sae,
                "percentile_table": percentile_table,
                "sae_layer": sae_layer,
                "tokenizer_id": tokenizer_id,
            }
        )
        return scores[prefix_text]

    monkeypatch.setattr(positive_mode, "fc_percentile_on_prefix", _score)
    model = object()
    sae = object()
    table = object()

    picked = _pick_seed_indices_for_concept(
        concept,
        seeds,
        cfg,
        base_model=model,
        base_sae=sae,
        percentile_table=table,
        sae_layer=13,
        effective_prefixes=effective,
    )

    assert {call["prefix_text"] for call in calls} == set(effective)
    assert all(call["base_model"] is model for call in calls)
    assert all(call["base_sae"] is sae for call in calls)
    assert all(call["percentile_table"] is table for call in calls)
    assert all(call["sae_layer"] == 13 for call in calls)
    assert all(call["tokenizer_id"] == cfg.extraction_tokenizer_id for call in calls)
    assert picked == [1, 2]


def test_serial_stage1b_threads_one_bounded_prefix_through_every_consumer(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        min_prefix_chars=8,
        max_prefix_chars=8,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    short_seed = _seed_row(0, text="short")
    full_prefix = "HEAD-01234567"
    bounded_prefix = "01234567"
    eligible_seed = _seed_row(1, text=full_prefix)
    seeds = [short_seed, eligible_seed]
    concept = _concept(concept_id=6, feature_index=42)
    stub_env["synth_client"] = _StubSynthClient(responses=["novel continuation"])
    stub_env["fc_active_by_continuation"] = {"cont": False}

    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    ((rich, avoid),) = _split_pairs(rows)
    expected_prompt = positive_mode._stage1b_continuation_request(
        TEST_STAGE1_TEMPLATE,
        concept=concept,
        prefix_text=bounded_prefix,
        seed_idx=1,
        attempt=0,
        content_class="concept_rich",
        cluster_target=None,
    ).prompt
    expected_cache_key = positive_mode.make_cache_key(
        model=cfg.synth.model,
        prompt=expected_prompt,
        tag=f"pos_c{concept.concept_id}_s1_a0",
    )
    assert stub_env["synth_client"].calls == [
        {"prompt": expected_prompt, "cache_key": expected_cache_key}
    ]
    assert stub_env["percentile_pair_calls"] == [
        {
            "prefix": bounded_prefix,
            "continuation": "novel continuation",
            "f_c": 42,
            "tokenizer_id": cfg.extraction_tokenizer_id,
        }
    ]
    assert stub_env["avoid_qc_calls"] == [
        {
            "prefix": bounded_prefix,
            "continuation": "cont",
            "tokenizer_id": cfg.extraction_tokenizer_id,
            "extraction_threshold": cfg.extraction_threshold,
        }
    ]
    assert rich["prefix_text"] == avoid["prefix_text"] == bounded_prefix
    assert rich["gen_seed"] == (
        cfg.generation.master_seed
        + positive_mode._CONCEPT_SEED_OFFSET
        + concept.concept_id * 1000
        + 1
    )
    assert avoid["gen_seed"] == positive_mode._avoiding_gen_seed(cfg, concept.concept_id, 1)
    assert rich["n_prefix_nanochat_tokens"] == positive_mode._exact_output_start(
        bounded_prefix, rich["output_text"]
    )
    assert avoid["n_prefix_nanochat_tokens"] == positive_mode._exact_output_start(
        bounded_prefix, avoid["output_text"]
    )
    assert seeds[1].prefix_text == full_prefix
    assert seeds[1].prefix_content_nanochat_tokens == [1, 2, 3]

    legacy_prompt = f"{full_prefix}::{concept.description}"
    legacy_key = positive_mode.make_cache_key(
        model=cfg.synth.model,
        prompt=legacy_prompt,
        tag=f"pos_c{concept.concept_id}_s1_a0",
    )
    assert expected_cache_key != legacy_key


def test_serial_rich_source_echo_is_retried_before_sae_qc(
    stub_env,
    tmp_path,
) -> None:
    prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
    clean = "A genuinely new continuation about a quiet winter garden."
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_prefix_chars=len(prefix),
        max_retries=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["synth_client"] = _StubSynthClient(responses=[prefix, clean])

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0, text=prefix)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    ((rich, _avoid),) = _split_pairs(rows)
    assert rich["output_text"] == clean
    assert rich["qc_attempts"] == 2
    assert [call["continuation"] for call in stub_env["percentile_pair_calls"]] == [clean]
    assert len(stub_env["synth_client"].calls) == 2
    assert (
        stub_env["synth_client"].calls[0]["cache_key"]
        != stub_env["synth_client"].calls[1]["cache_key"]
    )


def test_serial_rich_repeated_phrase_is_retried_before_sae_qc(
    stub_env,
    tmp_path,
) -> None:
    phrase = "astronomy geology mathematics philosophy architecture biology chemistry physics"
    repeated = f"{phrase}. {phrase}."
    clean = "Fresh material with no repeated sequence."
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_retries=1,
        max_prefix_chars=256,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["synth_client"] = _StubSynthClient(responses=[repeated, clean])

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0, text="an unrelated and sufficiently ordinary prefix")],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    ((rich, _avoid),) = _split_pairs(rows)
    assert rich["output_text"] == clean
    assert rich["qc_attempts"] == 2
    assert [call["continuation"] for call in stub_env["percentile_pair_calls"]] == [clean]


def test_parallel_rich_guard_retries_only_the_rejected_seed_and_restores_order(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=2,
        max_retries=1,
        max_prefix_chars=256,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    concept = _concept(concept_id=12)
    prefix_a = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    prefix_b = "kilo lima mike november oscar papa quebec romeo sierra tango"
    seeds = [
        _seed_row(0, text=prefix_a),
        _seed_row(1, text=prefix_b),
    ]
    prompt_a = f"{prefix_a}::{concept.description}"
    prompt_b = f"{prefix_b}::{concept.description}"
    client = _ScriptedConcurrentSynthClient(
        {
            prompt_a: [prefix_a, "clean continuation A"],
            prompt_b: ["clean continuation B"],
        },
        delay_s=0.001,
    )

    rows = _per_concept_positives(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=client,
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    pairs = _split_pairs(rows)
    shuffled = [0, 1]
    random.Random(
        cfg.generation.master_seed + positive_mode._CONCEPT_SEED_OFFSET + concept.concept_id
    ).shuffle(shuffled)
    assert [rich["prefix_text"] for rich, _avoid in pairs] == [
        seeds[idx].prefix_text for idx in shuffled
    ]
    rich_by_prefix = {rich["prefix_text"]: rich for rich, _avoid in pairs}
    assert rich_by_prefix[prefix_a]["output_text"] == "clean continuation A"
    assert rich_by_prefix[prefix_a]["qc_attempts"] == 2
    assert rich_by_prefix[prefix_b]["output_text"] == "clean continuation B"
    assert rich_by_prefix[prefix_b]["qc_attempts"] == 1
    assert sum(call["prompt"].startswith(f"PREFIX={prefix_a}\n") for call in client.calls) == 2
    assert sum(call["prompt"].startswith(f"PREFIX={prefix_b}\n") for call in client.calls) == 1
    assert len(client.calls) == 3
    assert {call["continuation"] for call in stub_env["percentile_pair_calls"]} == {
        "clean continuation A",
        "clean continuation B",
    }


def test_serial_parallel_rows_are_exact_with_filtering_and_tail_windowing(
    stub_env,
    tmp_path,
) -> None:
    concept = _concept(concept_id=19)
    seeds = [
        _seed_row(0, text="tiny"),
        _seed_row(1, text="discarded-head-AAAA1111"),
        _seed_row(2, text="discarded-head-BBBB2222"),
        _seed_row(3, text="CCCC3333"),
    ]
    bounded_prefixes = ("AAAA1111", "BBBB2222", "CCCC3333")
    responses = {
        f"{prefix}::{concept.description}": f"novel response {index}"
        for index, prefix in enumerate(bounded_prefixes)
    }

    class _PromptClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, str | None]] = []

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            self.calls.append({"prompt": prompt, "cache_key": cache_key})
            lines = prompt.splitlines()
            semantic_prompt = (
                f"{lines[0].removeprefix('PREFIX=')}::"
                f"{lines[1].removeprefix('DESCRIPTION=')}"
            )
            return frame_stage1_response(prompt, responses[semantic_prompt])

    common = {
        "n_positives_per_concept": 3,
        "min_prefix_chars": 8,
        "max_prefix_chars": 8,
        "max_retries": 0,
    }
    serial_cfg = _cfg(
        tmp_path,
        **common,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    parallel_cfg = _cfg(
        tmp_path,
        **common,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )

    serial_client = _PromptClient()
    serial_rows = _per_concept_positives(
        serial_cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=serial_client,
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )
    parallel_client = _PromptClient()
    parallel_rows = _per_concept_positives(
        parallel_cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        llm_client=parallel_client,
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert _rows_byte_equal(serial_rows, parallel_rows)
    assert {row["prefix_text"] for row in serial_rows} == set(bounded_prefixes)
    assert all(row["prefix_text"] != "tiny" for row in serial_rows)
    assert seeds[1].prefix_text == "discarded-head-AAAA1111"
    assert seeds[2].prefix_text == "discarded-head-BBBB2222"

    def _semantic_prompts(client: _PromptClient) -> set[str]:
        prompts: set[str] = set()
        for call in client.calls:
            prompt = call["prompt"]
            assert prompt is not None
            lines = prompt.splitlines()
            prompts.add(
                f"{lines[0].removeprefix('PREFIX=')}::"
                f"{lines[1].removeprefix('DESCRIPTION=')}"
            )
        return prompts

    assert _semantic_prompts(serial_client) == set(responses)
    assert _semantic_prompts(parallel_client) == set(responses)


@pytest.mark.parametrize(
    "rejected_avoid",
    [
        ("alpha bravo charlie delta echo foxtrot golf hotel india juliet"),
        (
            "astronomy geology mathematics philosophy architecture biology "
            "chemistry physics astronomy geology mathematics philosophy "
            "architecture biology chemistry physics"
        ),
    ],
    ids=["source-echo", "repeated-phrase"],
)
def test_llm_avoiding_guard_retries_before_strict_zero_qc(
    rejected_avoid: str,
    stub_env,
    tmp_path,
) -> None:
    source_prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_retries=1,
        max_prefix_chars=256,
    )
    stub_env["synth_client"] = _StubSynthClient(
        responses=["rich novel text", rejected_avoid, "clean avoiding text"],
    )
    stub_env["fc_active_by_continuation"] = {
        "cont": True,
        "clean avoiding text": False,
    }

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0, text=source_prefix)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    ((_rich, avoid),) = _split_pairs(rows)
    assert avoid["output_text"] == "clean avoiding text"
    assert avoid["qc_attempts"] == 2
    assert [call["continuation"] for call in stub_env["avoid_qc_calls"]] == [
        "cont",
        "clean avoiding text",
    ]
    assert all(row["output_text"] != rejected_avoid for row in rows)


def test_trusted_natural_avoiding_continuation_remains_outside_llm_guards(
    stub_env,
    tmp_path,
) -> None:
    phrase = "astronomy geology mathematics philosophy architecture biology chemistry physics"
    natural = f"{phrase} {phrase}"
    seed = replace(
        _seed_row(0, text=natural),
        continuation_text=natural,
    )
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_prefix_chars=len(natural),
    )
    stub_env["synth_client"] = _StubSynthClient(responses=["novel rich text"])
    stub_env["fc_active_by_continuation"] = {natural: False}

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [seed],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    ((_rich, avoid),) = _split_pairs(rows)
    assert avoid["output_text"] == natural
    assert avoid["gen_source"] == "corpus:doc_continuation@0/0"
    assert avoid["qc_attempts"] == 0
    assert len(stub_env["synth_client"].calls) == 1


@pytest.mark.parametrize("fallback_mode", ["warn", "best"])
def test_all_guard_rejected_rich_attempts_never_become_fallback_rows(
    fallback_mode: str,
    stub_env,
    tmp_path,
) -> None:
    prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_retries=1,
        max_prefix_chars=256,
        qc_fallback_on_exhaust=fallback_mode,
    )
    stub_env["synth_client"] = _StubSynthClient(responses=[prefix, prefix])

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0, text=prefix)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert rows == []
    assert stub_env["percentile_pair_calls"] == []


def test_all_guard_rejected_rich_attempts_raise_under_error_policy(
    stub_env,
    tmp_path,
) -> None:
    prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        max_retries=1,
        max_prefix_chars=256,
        qc_fallback_on_exhaust="error",
    )
    stub_env["synth_client"] = _StubSynthClient(responses=[prefix, prefix])

    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*rich_discards=1",
    ):
        _per_concept_positives(
            cfg,
            _concept(),
            [_seed_row(0, text=prefix)],
            base_model=object(),
            base_sae=object(),
            llm_client=stub_env["synth_client"],
            percentile_table=None,
            target_model_meta=_stub_meta(),
        )

    assert stub_env["percentile_pair_calls"] == []


def test_best_fallback_never_keeps_a_failed_rich_attempt(
    stub_env,
    tmp_path,
) -> None:
    prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    legitimate = "A legitimate but low-scoring novel continuation."
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
        max_retries=1,
        max_prefix_chars=256,
        qc_fallback_on_exhaust="best",
    )
    stub_env["synth_client"] = _StubSynthClient(
        responses=[legitimate, prefix],
    )
    stub_env["percentile_pairs"] = [(10.0, 20.0)]

    rows = _per_concept_positives(
        cfg,
        _concept(),
        [_seed_row(0, text=prefix)],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        percentile_table=None,
        target_model_meta=_stub_meta(),
    )

    assert rows == []
    assert [call["continuation"] for call in stub_env["percentile_pair_calls"]] == [legitimate]


def test_protocol_failure_counter_records_exact_attempt_and_content_class(
    tmp_path,
) -> None:
    source = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    repeated_phrase = (
        "astronomy geology mathematics philosophy architecture biology chemistry physics"
    )
    profile_dir = tmp_path / "protocol-profile"
    concept = _concept(concept_id=31)
    tokenizer = _StubExtractionTokenizer()

    with AxBenchProfiler(profile_dir):
        rich_request = positive_mode._stage1b_continuation_request(
            TEST_STAGE1_TEMPLATE,
            concept=concept,
            prefix_text=source,
            seed_idx=7,
            attempt=0,
            content_class="concept_rich",
            cluster_target=None,
        )
        assert positive_mode.parse_stage1_continuation_or_none(
            response=frame_stage1_response(rich_request.prompt, source),
            request=rich_request,
            prefix_text=source,
            output_length=64,
            tokenizer=tokenizer,
            subject="Stage 1b concept_rich continuation",
            log_label="rich",
            protocol_counter="stage1b.continuation_protocol_failures",
            special_token_counter="stage1b.special_token_refusals",
            counter_attrs={
                "concept_id": 31,
                "attempt": 0,
                "content_class": "concept_rich",
            },
        ) is None
        avoid_request = positive_mode._stage1b_continuation_request(
            TEST_STAGE1_TEMPLATE,
            concept=concept,
            prefix_text="unrelated source context",
            seed_idx=7,
            attempt=2,
            content_class="concept_avoiding",
            cluster_target=None,
        )
        assert positive_mode.parse_stage1_continuation_or_none(
            response=frame_stage1_response(
                avoid_request.prompt,
                f"{repeated_phrase} {repeated_phrase}",
            ),
            request=avoid_request,
            prefix_text="unrelated source context",
            output_length=64,
            tokenizer=tokenizer,
            subject="Stage 1b concept_avoiding continuation",
            log_label="avoid",
            protocol_counter="stage1b.continuation_protocol_failures",
            special_token_counter="stage1b.special_token_refusals",
            counter_attrs={
                "concept_id": 31,
                "attempt": 2,
                "content_class": "concept_avoiding",
            },
        ) is None
        clean_request = positive_mode._stage1b_continuation_request(
            TEST_STAGE1_TEMPLATE,
            concept=concept,
            prefix_text=source,
            seed_idx=7,
            attempt=3,
            content_class="concept_rich",
            cluster_target=None,
        )
        assert positive_mode.parse_stage1_continuation_or_none(
            response=frame_stage1_response(
                clean_request.prompt,
                "clean and non-repeating continuation",
            ),
            request=clean_request,
            prefix_text=source,
            output_length=64,
            tokenizer=tokenizer,
            subject="Stage 1b concept_rich continuation",
            log_label="clean",
            protocol_counter="stage1b.continuation_protocol_failures",
            special_token_counter="stage1b.special_token_refusals",
            counter_attrs={
                "concept_id": 31,
                "attempt": 3,
                "content_class": "concept_rich",
            },
        ) == "clean and non-repeating continuation"

    events = [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    protocol_events = [
        event
        for event in events
        if event["type"] == "counter" and event["name"] == "stage1b.continuation_protocol_failures"
    ]
    assert [(event["value"], event["attrs"]) for event in protocol_events] == [
        (
            1.0,
            {
                "concept_id": 31,
                "attempt": 0,
                "content_class": "concept_rich",
            },
        ),
        (
            1.0,
            {
                "concept_id": 31,
                "attempt": 2,
                "content_class": "concept_avoiding",
            },
        ),
    ]
    assert _read_counter_totals(profile_dir)["stage1b.continuation_protocol_failures"] == 2.0


def _issue_1051_valid_row(
    prefix_text: str,
    *,
    content_class: str,
    concept_id: int = 4,
    output_text: str = "novel continuation",
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "content_class": content_class,
        "prefix_text": prefix_text,
        "output_text": output_text,
        "n_prefix_nanochat_tokens": positive_mode._exact_output_start(
            prefix_text,
            output_text,
        ),
    }


def test_validate_emitted_positive_rows_accepts_closed_prefix_bounds_and_pair(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    rich = _issue_1051_valid_row("abc", content_class="concept_rich")
    avoid = _issue_1051_valid_row("abc", content_class="concept_avoiding")
    upper_bound_rich = _issue_1051_valid_row(
        "wxyz",
        content_class="concept_rich",
        concept_id=5,
    )

    positive_mode._validate_emitted_positive_rows(
        cfg,
        [rich, avoid, upper_bound_rich],
    )


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        ("ab", r"prefix length 2, outside \[3, 4\]"),
        ("abcde", r"prefix length 5, outside \[3, 4\]"),
        (None, r"non-string prefix_text"),
        (123, r"non-string prefix_text"),
    ],
)
def test_validate_emitted_positive_rows_rejects_bad_prefix(
    bad_value: object,
    message: str,
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    row = _issue_1051_valid_row("abc", content_class="concept_rich")
    row["prefix_text"] = bad_value

    with pytest.raises(AxBenchBaseError, match=message):
        positive_mode._validate_emitted_positive_rows(cfg, [row])


@pytest.mark.parametrize("bad_output", ["", " ", "\n\t", None, 123])
def test_validate_emitted_positive_rows_rejects_empty_or_nonstring_output(
    bad_output: object,
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    row = _issue_1051_valid_row("abc", content_class="concept_rich")
    row["output_text"] = bad_output

    with pytest.raises(
        AxBenchBaseError,
        match=r"row 0 has empty or non-string output_text",
    ):
        positive_mode._validate_emitted_positive_rows(cfg, [row])


def test_validate_emitted_positive_rows_rejects_stale_output_boundary(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    row = _issue_1051_valid_row("abc", content_class="concept_rich")
    expected = row["n_prefix_nanochat_tokens"]
    row["n_prefix_nanochat_tokens"] = expected + 1

    with pytest.raises(
        AxBenchBaseError,
        match=rf"stale nanochat output boundary {expected + 1}; recomputed {expected}",
    ):
        positive_mode._validate_emitted_positive_rows(cfg, [row])


def test_validate_emitted_positive_rows_rejects_orphan_avoiding_row(
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    avoid = _issue_1051_valid_row("abc", content_class="concept_avoiding")

    with pytest.raises(AxBenchBaseError, match=r"without a preceding"):
        positive_mode._validate_emitted_positive_rows(cfg, [avoid])


@pytest.mark.parametrize(
    ("rich_overrides", "message"),
    [
        (
            {"content_class": "concept_avoiding"},
            r"without a preceding",
        ),
        (
            {"concept_id": 99},
            r"prefix is not paired",
        ),
        (
            {"prefix_text": "wxyz"},
            r"prefix is not paired",
        ),
    ],
)
def test_validate_emitted_positive_rows_rejects_mismatched_pair(
    rich_overrides: dict[str, object],
    message: str,
    stub_env,
    tmp_path,
) -> None:
    del stub_env
    cfg = _cfg(tmp_path, min_prefix_chars=3, max_prefix_chars=4)
    rich = _issue_1051_valid_row("abc", content_class="concept_rich")
    avoid = _issue_1051_valid_row("abc", content_class="concept_avoiding")
    rich.update(rich_overrides)
    if "prefix_text" in rich_overrides:
        rich["n_prefix_nanochat_tokens"] = positive_mode._exact_output_start(
            str(rich["prefix_text"]),
            str(rich["output_text"]),
        )

    with pytest.raises(AxBenchBaseError, match=message):
        positive_mode._validate_emitted_positive_rows(cfg, [rich, avoid])


def test_public_stage1b_fails_before_percentile_qc_or_api_work_when_no_prefix_is_eligible(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    del stub_env
    cfg = _cfg(
        tmp_path,
        min_prefix_chars=5,
        max_prefix_chars=10,
        n_positives_per_concept=1,
    )
    paths_for_cfg = paths.AxBenchBasePaths.from_config(cfg)

    def _must_not_build(*_args, **_kwargs):
        raise AssertionError("percentile/QC work must not start")

    monkeypatch.setattr(
        positive_mode,
        "build_percentile_table_from_stats",
        _must_not_build,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"no eligible positive seed prefixes.*\[5, 10\].*1 Stage 0b seeds",
    ):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0, text="tiny")],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    assert not paths_for_cfg.synth_response_cache().exists()
    assert not paths_for_cfg.synth_training_positive().exists()


def test_public_stage1b_preflights_every_concept_before_provider_access(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        qc_fallback_on_exhaust="warn",
    )
    stub_env["prefix_percentiles"] = {
        10: 0.0,
        20: 90.0,
    }

    with pytest.raises(
        AxBenchBaseError,
        match=r"concept_id=1.*selected=0",
    ):
        synthesize_positive_mode(
            cfg,
            [_concept(0, 10), _concept(1, 20)],
            [_seed_row(0), _seed_row(1)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    assert stub_env["synth_client"].calls == []
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    assert not output_paths.synth_response_cache().exists()
    assert not output_paths.synth_training_positive().exists()


def test_public_stage1b_runtime_shortfall_error_publishes_no_artifacts(
    stub_env,
    tmp_path,
) -> None:
    prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        max_prefix_chars=256,
        qc_fallback_on_exhaust="error",
    )
    stub_env["synth_client"] = _StubSynthClient(responses=[prefix])

    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*rich_discards=1",
    ):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0, text=prefix)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    assert output_paths.synth_response_cache().exists()
    assert not output_paths.synth_training_positive().exists()
    assert not output_paths.stage1b_concepts().exists()
    assert not output_paths.stage1b_concept_yield_report().exists()


@pytest.mark.parametrize("fallback_mode", ["warn", "best"])
def test_public_stage1b_tolerated_shortfall_publishes_only_yield_drop(
    fallback_mode: str,
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        qc_fallback_on_exhaust=fallback_mode,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["synth_client"] = _StubSynthClient(
        responses=["failed rich continuation", "accepted rich continuation"],
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda _prefix, _continuation, feature_id, *_args, **_kwargs: (
            0.0 if feature_id == 42 else 80.0
        ),
    )
    concepts = [
        _concept(concept_id=0, feature_index=42),
        _concept(concept_id=1, feature_index=43),
    ]

    out = synthesize_positive_mode(
        cfg,
        concepts,
        [_seed_row(0)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    assert out == output_paths.synth_training_positive()
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 2
    assert {row["concept_id"] for row in rows} == {1}
    assert [row["content_class"] for row in rows] == [
        "concept_rich",
        "concept_avoiding",
    ]
    assert json.loads(
        output_paths.stage1b_concepts().read_text(encoding="utf-8")
    ) == [concepts[1].__dict__]
    report = json.loads(
        output_paths.stage1b_concept_yield_report().read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "ok"
    assert report["surviving_concept_ids"] == [1]
    assert report["n_concepts_dropped"] == 1
    assert report["dropped_concept_fraction"] == 0.5
    failed, accepted = report["concepts"]
    assert failed == {
        "accepted": 0,
        "attempted": 1,
        "concept_id": 0,
        "drop_fraction": 1.0,
        "dropped": 1,
        "emitted_rows": 0,
        "excluded": True,
        "exclusion_reason": "drop_fraction_exceeded",
        "feature_index": 42,
        "reason_counts": {"missing_rich": 1},
    }
    assert accepted["concept_id"] == 1
    assert accepted["accepted"] == accepted["attempted"] == 1
    assert accepted["dropped"] == 0
    assert accepted["excluded"] is False


def test_public_stage1b_validates_rows_before_parquet_publication(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    del stub_env
    cfg = _cfg(
        tmp_path,
        min_prefix_chars=1,
        max_prefix_chars=4,
        n_positives_per_concept=1,
    )
    invalid_rich = _issue_1051_valid_row(
        "too-long",
        content_class="concept_rich",
    )
    invalid_avoid = _issue_1051_valid_row(
        "too-long",
        content_class="concept_avoiding",
    )
    write_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        positive_mode,
        "_run_positive_target_refill_engine",
        lambda *_args, **_kwargs: {0: [invalid_rich, invalid_avoid]},
    )
    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        lambda *args, **kwargs: write_calls.append((args, kwargs)),
    )

    with pytest.raises(AxBenchBaseError, match=r"prefix length 8.*outside \[1, 4\]"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0, text="abcd")],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    assert write_calls == []
    assert not paths.AxBenchBasePaths.from_config(cfg).synth_training_positive().exists()


def test_stage1b_parquet_failure_leaves_report_uncommitted(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    del stub_env
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        accept_threshold=0.5,
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    corpus_path = output_paths.synth_training_positive()
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    positive_mode._write_training_parquet(
        [],
        corpus_path,
        tokenizer_id=cfg.extraction_tokenizer_id,
        cfg=cfg,
    )
    previous_corpus = corpus_path.read_bytes()

    def _fail_write(*_args, **_kwargs) -> None:
        raise RuntimeError("injected Stage 1b parquet failure")

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        _fail_write,
    )
    with pytest.raises(RuntimeError, match="injected Stage 1b parquet failure"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0, sae_ids=[])],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    report_path = output_paths.stage1b_concept_yield_report()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "preparing"
    assert not concept_yield_report_is_committed(report_path)
    assert corpus_path.read_bytes() == previous_corpus


def test_stage1b_target_mismatch_fails_before_scorer_or_provider(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "target-mismatch"),
        n_positives_per_concept=2,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    failing_client = _StubSynthClient(
        raise_for=AxBenchBaseError("synthetic provider outage")
    )
    stub_env["synth_client"] = failing_client
    with pytest.raises(AxBenchBaseError, match="provider transport exhausted"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0), _seed_row(1)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    score_calls = list(stub_env["prefix_cache_calls"])
    provider_calls = len(failing_client.calls)
    with pytest.raises(
        AxBenchBaseError,
        match=r"target decrease is unsupported",
    ):
        synthesize_positive_mode(
            cfg.replace(n_positives_per_concept=1),
            [_concept()],
            [_seed_row(0), _seed_row(1)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    assert stub_env["prefix_cache_calls"] == score_calls
    assert len(failing_client.calls) == provider_calls


@pytest.mark.parametrize(
    ("initial_target", "final_target", "seed_count", "max_concurrent"),
    [
        (1, 15, 120, 1),
        (15, 16, 120, 4),
        (16, 17, 120, 1),
        (17, 71, 120, 4),
        (71, 72, 120, 1),
        (72, 73, 120, 4),
        (73, 100, 120, 1),
        (1, 100, 120, 4),
    ],
)
def test_stage1b_partial_target_growth_matches_fresh_larger_target_bytes(
    stub_env,
    tmp_path,
    monkeypatch,
    initial_target: int,
    final_target: int,
    seed_count: int,
    max_concurrent: int,
) -> None:
    seeds = [
        _seed_row(index, text=f"growth-{index}")
        for index in range(seed_count)
    ]
    growth_cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "growth-output"),
        run_tag="growth-run",
        n_positives_per_concept=initial_target,
        synth=SynthConfig(
            requests_per_minute=0,
            max_concurrent=max_concurrent,
        ),
    )
    if max_concurrent > 1:
        stub_env["synth_client"] = _ConcurrentSynthClient(delay_s=0.001)
    original_writer = positive_mode._write_training_parquet

    def interrupt_publication(*_args, **_kwargs) -> None:
        raise RuntimeError("interrupt after target-one checkpoint")

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        interrupt_publication,
    )
    with pytest.raises(RuntimeError, match="target-one checkpoint"):
        synthesize_positive_mode(
            growth_cfg,
            [_concept()],
            seeds,
            object(),
            object(),
            _stub_meta(),
        )
    growth_paths = paths.AxBenchBasePaths.from_config(growth_cfg)
    partial = json.loads(
        growth_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    assert partial["highest_requested_target"] == initial_target
    assert partial["accepted_pair_count"] == initial_target

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        original_writer,
    )
    growth_client = stub_env["synth_client"]
    grown_cfg = growth_cfg.replace(
        n_positives_per_concept=final_target
    )
    grown_path = synthesize_positive_mode(
        grown_cfg,
        [_concept()],
        seeds,
        object(),
        object(),
        _stub_meta(),
    )

    fresh_client = (
        _StubSynthClient(responses=["canned continuation"] * final_target)
        if max_concurrent == 1
        else _ConcurrentSynthClient(delay_s=0.001)
    )
    stub_env["synth_client"] = fresh_client
    fresh_cfg = grown_cfg.replace(
        output_root=str(tmp_path / "fresh-output"),
        run_tag="fresh-run",
    )
    fresh_path = synthesize_positive_mode(
        fresh_cfg,
        [_concept()],
        seeds,
        object(),
        object(),
        _stub_meta(),
    )

    grown_paths = paths.AxBenchBasePaths.from_config(grown_cfg)
    fresh_paths = paths.AxBenchBasePaths.from_config(fresh_cfg)
    assert grown_path.read_bytes() == fresh_path.read_bytes()
    assert (
        grown_paths.stage1b_concepts().read_bytes()
        == fresh_paths.stage1b_concepts().read_bytes()
    )
    assert (
        grown_paths.stage1b_concept_yield_report().read_bytes()
        == fresh_paths.stage1b_concept_yield_report().read_bytes()
    )
    assert sorted(
        (call["prompt"], call["cache_key"])
        for call in growth_client.calls
    ) == sorted(
        (call["prompt"], call["cache_key"])
        for call in fresh_client.calls
    )
    assert len(growth_client.calls) == final_target
    assert not grown_paths.stage1b_partial_json().exists()
    assert not grown_paths.stage1b_partial_parquet().exists()


@pytest.mark.parametrize("current_target", [72, 73])
def test_stage1b_schema_v1_target_72_migration_matches_fresh_bytes(
    stub_env,
    tmp_path,
    monkeypatch,
    current_target: int,
) -> None:
    seeds = [
        _seed_row(index, text=f"v1-migration-{index}")
        for index in range(80)
    ]
    phase1_cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "v1-migration-output"),
        run_tag="v1-migration-run",
        n_positives_per_concept=72,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    original_writer = positive_mode._write_training_parquet

    def interrupt_publication(*_args, **_kwargs) -> None:
        raise RuntimeError("leave target-72 checkpoint")

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        interrupt_publication,
    )
    with pytest.raises(RuntimeError, match="leave target-72 checkpoint"):
        synthesize_positive_mode(
            phase1_cfg,
            [_concept()],
            seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    migrated_paths = paths.AxBenchBasePaths.from_config(phase1_cfg)
    partial_json_path = migrated_paths.stage1b_partial_json()
    phase1_document = json.loads(
        partial_json_path.read_text(encoding="utf-8")
    )
    phase1_document["schema_version"] = (
        positive_checkpoint.CHECKPOINT_SCHEMA_VERSION_V1
    )
    phase1_document.pop("highest_requested_target")
    legacy_v3_fingerprint = dict(phase1_document["fingerprint"])
    legacy_v3_fingerprint["checkpoint_algorithm_version"] = (
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
    )
    legacy_v3_fingerprint["seed_rows_sha256"] = (
        positive_mode._seed_rows_identity(seeds)
    )
    phase1_document["fingerprint"] = (
        positive_mode._stage1b_checkpoint_fingerprint_v1(
            phase1_cfg,
            legacy_v3_fingerprint,
        )
    )
    partial_json_path.write_text(
        json.dumps(phase1_document, sort_keys=True),
        encoding="utf-8",
    )

    current_cfg = phase1_cfg.replace(
        n_positives_per_concept=current_target
    )

    def assert_v2_migrated_then_write(*args, **kwargs) -> None:
        committed = json.loads(
            partial_json_path.read_text(encoding="utf-8")
        )
        assert committed["schema_version"] == (
            positive_checkpoint.CHECKPOINT_SCHEMA_VERSION
        )
        assert committed["highest_requested_target"] == current_target
        original_writer(*args, **kwargs)

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        assert_v2_migrated_then_write,
    )
    migrated_client = stub_env["synth_client"]
    migrated_path = synthesize_positive_mode(
        current_cfg,
        [_concept()],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    fresh_client = _StubSynthClient(
        responses=["canned continuation"] * current_target
    )
    stub_env["synth_client"] = fresh_client
    fresh_cfg = current_cfg.replace(
        output_root=str(tmp_path / "v1-migration-fresh"),
        run_tag="v1-migration-fresh",
    )
    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        original_writer,
    )
    fresh_path = synthesize_positive_mode(
        fresh_cfg,
        [_concept()],
        seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    fresh_paths = paths.AxBenchBasePaths.from_config(fresh_cfg)

    assert migrated_path.read_bytes() == fresh_path.read_bytes()
    assert (
        migrated_paths.stage1b_concepts().read_bytes()
        == fresh_paths.stage1b_concepts().read_bytes()
    )
    assert (
        migrated_paths.stage1b_concept_yield_report().read_bytes()
        == fresh_paths.stage1b_concept_yield_report().read_bytes()
    )
    assert sorted(
        (call["prompt"], call["cache_key"])
        for call in migrated_client.calls
    ) == sorted(
        (call["prompt"], call["cache_key"])
        for call in fresh_client.calls
    )
    assert len(migrated_client.calls) == current_target


def test_completed_target_72_corpus_refuses_target_73_extension_before_scoring(
    stub_env,
    tmp_path,
) -> None:
    cfg_72 = _cfg(
        tmp_path,
        output_root=str(tmp_path / "completed-output"),
        run_tag="completed-run",
        n_positives_per_concept=72,
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg_72)
    completed_path = output_paths.synth_training_positive()
    completed_path.parent.mkdir(parents=True, exist_ok=True)
    positive_mode._write_training_parquet(
        [],
        completed_path,
        tokenizer_id=cfg_72.extraction_tokenizer_id,
        cfg=cfg_72,
    )
    before = completed_path.read_bytes()
    score_calls = list(stub_env.get("prefix_cache_calls", []))
    provider_calls = list(stub_env["synth_client"].calls)

    with pytest.raises(
        AxBenchBaseError,
        match=(
            "completed positive corpus target mismatch.*stored target=72.*"
            "current target=73.*fresh explicit run_tag"
        ),
    ):
        synthesize_positive_mode(
            cfg_72.replace(n_positives_per_concept=73),
            [_concept()],
            [_seed_row(0)],
            object(),
            object(),
            _stub_meta(),
        )

    assert completed_path.read_bytes() == before
    assert stub_env.get("prefix_cache_calls", []) == score_calls
    assert stub_env["synth_client"].calls == provider_calls
    assert not output_paths.stage1b_partial_json().exists()
    assert not output_paths.stage1b_partial_parquet().exists()


def test_completed_growth_refusal_applies_during_partial_cleanup_crash_window(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "completed-crash-window"),
        run_tag="completed-crash-window",
        n_positives_per_concept=1,
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)

    def fail_cleanup(**_kwargs) -> None:
        raise OSError("synthetic checkpoint cleanup failure")

    monkeypatch.setattr(
        positive_checkpoint,
        "delete_checkpoint",
        fail_cleanup,
    )
    completed_path = synthesize_positive_mode(
        cfg,
        [_concept()],
        [_seed_row(0, sae_ids=[])],
        object(),
        object(),
        _stub_meta(),
    )
    assert completed_path.is_file()
    assert output_paths.stage1b_partial_json().is_file()
    previous_json = output_paths.stage1b_partial_json().read_bytes()
    score_calls = list(stub_env.get("prefix_cache_calls", []))
    provider_calls = list(stub_env["synth_client"].calls)

    with pytest.raises(
        AxBenchBaseError,
        match=(
            "completed positive corpus target mismatch.*stored target=1.*"
            "current target=2.*final-publication crash window.*"
            "fresh explicit run_tag"
        ),
    ):
        synthesize_positive_mode(
            cfg.replace(n_positives_per_concept=2),
            [_concept()],
            [_seed_row(0, sae_ids=[])],
            object(),
            object(),
            _stub_meta(),
        )

    assert output_paths.stage1b_partial_json().read_bytes() == previous_json
    assert stub_env.get("prefix_cache_calls", []) == score_calls
    assert stub_env["synth_client"].calls == provider_calls


def test_completed_same_target_corruption_keeps_native_parquet_error(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "completed-corrupt"),
        run_tag="completed-corrupt",
        n_positives_per_concept=1,
    )
    completed_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).synth_training_positive()
    completed_path.parent.mkdir(parents=True, exist_ok=True)
    completed_path.write_bytes(b"not a parquet corpus")

    with pytest.raises(pa.ArrowInvalid, match="Parquet magic bytes"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            object(),
            object(),
            _stub_meta(),
        )
    assert stub_env["synth_client"].calls == []


def test_completed_same_target_semantic_drift_keeps_contract_error(
    stub_env,
    tmp_path,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "completed-semantic-drift"),
        run_tag="completed-semantic-drift",
        n_positives_per_concept=1,
        accept_threshold=0.3,
    )
    completed_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).synth_training_positive()
    completed_path.parent.mkdir(parents=True, exist_ok=True)
    positive_mode._write_training_parquet(
        [],
        completed_path,
        tokenizer_id=cfg.extraction_tokenizer_id,
        cfg=cfg,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="selection contract mismatch.*accept_fraction",
    ):
        synthesize_positive_mode(
            cfg.replace(accept_threshold=0.5),
            [_concept()],
            [_seed_row(0)],
            object(),
            object(),
            _stub_meta(),
        )
    assert stub_env["synth_client"].calls == []


def test_infeasible_growth_preflight_preserves_previous_target_authority(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "infeasible-growth"),
        run_tag="infeasible-growth",
        n_positives_per_concept=1,
        qc_fallback_on_exhaust="warn",
    )
    seeds = [
        _seed_row(index, text=f"infeasible-{index}", sae_ids=[])
        for index in range(3)
    ]
    original_training_writer = positive_mode._write_training_parquet

    def stop_after_target_one(*_args, **_kwargs) -> None:
        raise RuntimeError("stop after target one")

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        stop_after_target_one,
    )
    with pytest.raises(RuntimeError, match="stop after target one"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            seeds,
            object(),
            object(),
            _stub_meta(),
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    previous_json = output_paths.stage1b_partial_json().read_bytes()
    previous_provider_calls = list(stub_env["synth_client"].calls)

    with pytest.raises(
        AxBenchBaseError,
        match="too few admissible prefixes.*required=4.*selected=3",
    ):
        synthesize_positive_mode(
            cfg.replace(n_positives_per_concept=4),
            [_concept()],
            seeds,
            object(),
            object(),
            _stub_meta(),
        )

    assert output_paths.stage1b_partial_json().read_bytes() == previous_json
    assert json.loads(previous_json)["highest_requested_target"] == 1
    assert stub_env["synth_client"].calls == previous_provider_calls

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        original_training_writer,
    )
    synthesize_positive_mode(
        cfg,
        [_concept()],
        seeds,
        object(),
        object(),
        _stub_meta(),
    )
    assert stub_env["synth_client"].calls == previous_provider_calls
    assert not output_paths.stage1b_partial_json().exists()


def test_growth_checkpoint_write_failure_preserves_previous_target_authority(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "growth-write-failure"),
        run_tag="growth-write-failure",
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    original_training_writer = positive_mode._write_training_parquet

    def stop_after_target_one(*_args, **_kwargs) -> None:
        raise RuntimeError("stop after target one")

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        stop_after_target_one,
    )
    with pytest.raises(RuntimeError, match="stop after target one"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(index, text=f"authority-{index}") for index in range(3)],
            object(),
            object(),
            _stub_meta(),
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    previous_json = output_paths.stage1b_partial_json().read_bytes()
    provider_calls = list(stub_env["synth_client"].calls)
    original_checkpoint_writer = positive_checkpoint.write_checkpoint

    def fail_growth_write(**kwargs) -> None:
        if kwargs["document"]["highest_requested_target"] == 3:
            raise OSError("synthetic growth checkpoint failure")
        original_checkpoint_writer(**kwargs)

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        original_training_writer,
    )
    monkeypatch.setattr(
        positive_checkpoint,
        "write_checkpoint",
        fail_growth_write,
    )
    with pytest.raises(OSError, match="growth checkpoint failure"):
        synthesize_positive_mode(
            cfg.replace(n_positives_per_concept=3),
            [_concept()],
            [_seed_row(index, text=f"authority-{index}") for index in range(3)],
            object(),
            object(),
            _stub_meta(),
        )

    assert output_paths.stage1b_partial_json().read_bytes() == previous_json
    assert json.loads(previous_json)["highest_requested_target"] == 1
    assert stub_env["synth_client"].calls == provider_calls


def test_stage1b_terminal_shortfall_rejects_changed_publication_policy(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "relaxed-salvage"),
        n_positives_per_concept=1,
        positive_max_attempts_per_seed=1,
        qc_fallback_on_exhaust="error",
        max_concept_drop_fraction=0.1,
        max_dropped_concept_fraction=0.1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda _prefix, _continuation, feature_id, *_args, **_kwargs: (
            0.0 if feature_id == 42 else 80.0
        ),
    )
    concepts = [
        _concept(concept_id=0, feature_index=42),
        _concept(concept_id=1, feature_index=43),
    ]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["discarded first concept"]
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"complete-pair target shortfall.*concept_id=0",
    ):
        synthesize_positive_mode(
            cfg,
            concepts,
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    resumed_client = _StubSynthClient(
        responses=["accepted surviving concept"]
    )
    stub_env["synth_client"] = resumed_client
    resumed_cfg = cfg.replace(
        qc_fallback_on_exhaust="warn",
        max_concept_drop_fraction=0.9,
        max_dropped_concept_fraction=0.9,
    )
    score_calls = list(stub_env["prefix_cache_calls"])
    with pytest.raises(
        AxBenchBaseError,
        match=r"fingerprint mismatch.*max_concept_drop_fraction",
    ):
        synthesize_positive_mode(
            resumed_cfg,
            concepts,
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    assert resumed_client.calls == []
    assert stub_env["prefix_cache_calls"] == score_calls


def test_stage1b_keyboard_interrupt_emergency_checkpoint_resumes(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    class _InterruptingClient:
        calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise KeyboardInterrupt("synthetic Ctrl-C")

    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "keyboard-interrupt"),
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    interrupting = _InterruptingClient()
    stub_env["synth_client"] = interrupting
    with pytest.raises(KeyboardInterrupt, match="synthetic Ctrl-C"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    checkpoint = json.loads(
        output_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    assert checkpoint["accepted_pair_count"] == 0
    assert len(checkpoint["concept_states"][0]["tracker"]["outstanding"]) == 1
    assert pq.read_table(output_paths.stage1b_partial_parquet()).num_rows == 0

    resumed_cfg = cfg.replace(
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    resumed_fingerprint = positive_mode._stage1b_checkpoint_fingerprint(
        resumed_cfg,
        [_concept()],
        _test_seed_rows_sha256([_seed_row(0)]),
        _stub_meta(),
        cluster_contract=None,
    )
    with pytest.raises(
        AxBenchBaseError,
        match=r"fingerprint mismatch.*post_provider_qc_batch_size",
    ):
        positive_checkpoint.validate_fingerprint(
            checkpoint["fingerprint"],
            resumed_fingerprint,
            json_path=output_paths.stage1b_partial_json(),
        )

    # Model an exact checkpoint written by the pre-batching v2 algorithm. Its
    # outstanding work adopts the configured post-provider shape once, then
    # the next commit is protected by the current v4 fingerprint.
    legacy_v3_fingerprint = dict(checkpoint["fingerprint"])
    legacy_v3_fingerprint["checkpoint_algorithm_version"] = (
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
    )
    legacy_v3_fingerprint["seed_rows_sha256"] = (
        positive_mode._seed_rows_identity([_seed_row(0)])
    )
    checkpoint["fingerprint"] = (
        positive_mode._stage1b_checkpoint_fingerprint_v2(
            legacy_v3_fingerprint
        )
    )
    output_paths.stage1b_partial_json().write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    resumed_client = _StubSynthClient(responses=["resumed continuation"])
    stub_env["synth_client"] = resumed_client
    original_batch_scorer = positive_mode.output_fc_percentiles_for_continuations
    rich_batch_sizes: list[int] = []

    def _record_batch(inputs, *args, **kwargs):
        rich_batch_sizes.append(len(inputs))
        return original_batch_scorer(inputs, *args, **kwargs)

    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        _record_batch,
    )
    # The explicit v2 migration allows the pre-batching scalar checkpoint to
    # resume outstanding work through the current batched callback.
    out = synthesize_positive_mode(
        resumed_cfg,
        [_concept()],
        [_seed_row(0)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert out.exists()
    assert rich_batch_sizes == [1]
    assert len(resumed_client.calls) == 1
    assert not output_paths.stage1b_partial_json().exists()
    assert not output_paths.stage1b_partial_parquet().exists()


def test_checkpoint_state_commit_defers_sigint_until_state_is_coherent(
    monkeypatch,
) -> None:
    events: list[int] = []
    state: list[str] = []

    def fake_pthread_sigmask(how, _mask):
        events.append(how)
        if how == positive_mode.signal.SIG_SETMASK:
            raise KeyboardInterrupt("deferred synthetic Ctrl-C")
        return set()

    monkeypatch.setattr(
        positive_mode.signal,
        "pthread_sigmask",
        fake_pthread_sigmask,
    )

    with pytest.raises(KeyboardInterrupt, match="deferred synthetic Ctrl-C"):
        with positive_mode._defer_sigint_for_checkpoint_state():
            state.extend(["tracker", "work", "sequence"])

    assert state == ["tracker", "work", "sequence"]
    assert events == [
        positive_mode.signal.SIG_BLOCK,
        positive_mode.signal.SIG_SETMASK,
    ]


def test_candidate_admission_defers_sigint_until_ownership_is_coherent(
    monkeypatch,
) -> None:
    tracker = _tracker_for_invariant_tests()

    def fake_pthread_sigmask(how, _mask):
        if how == positive_mode.signal.SIG_SETMASK:
            raise KeyboardInterrupt("deferred admission Ctrl-C")
        return set()

    monkeypatch.setattr(
        positive_mode.signal,
        "pthread_sigmask",
        fake_pthread_sigmask,
    )
    with pytest.raises(KeyboardInterrupt, match="deferred admission Ctrl-C"):
        tracker.admit_next(order=0)

    assert tracker.seeds_drawn == 1
    assert tracker.admitted_seed_indices == {0}
    assert set(tracker.outstanding) == {0}
    source = tracker.source
    assert isinstance(source, _TrackerCandidateSource)
    assert source.candidates == []
    tracker._assert_invariants(allow_refill_deficit=True)


def test_scheduler_admission_defers_sigint_until_chain_index_is_coherent(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """A Ctrl-C after ownership transfer must still admit an emergency save."""

    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    concept = _concept()
    seed = _seed_row(0)
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        [seed],
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 0.0}),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, [seed]),
    )

    def fake_pthread_sigmask(how, _mask):
        if how == positive_mode.signal.SIG_SETMASK:
            raise KeyboardInterrupt("deferred scheduler-admission Ctrl-C")
        return set()

    monkeypatch.setattr(
        positive_mode.signal,
        "pthread_sigmask",
        fake_pthread_sigmask,
    )
    json_path = tmp_path / "partial.json"
    parquet_path = tmp_path / "partial.parquet"
    with pytest.raises(
        KeyboardInterrupt,
        match="deferred scheduler-admission Ctrl-C",
    ):
        positive_mode._run_positive_target_refill_engine(
            cfg,
            [tracker],
            object(),
            object(),
            stub_env["synth_client"],
            None,
            _stub_meta(),
            checkpoint_fingerprint={"test": "scheduler-admission-sigint"},
            checkpoint_json_path=json_path,
            checkpoint_parquet_path=parquet_path,
        )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["next_chain_index"] == 1
    outstanding = document["concept_states"][0]["tracker"]["outstanding"]
    assert len(outstanding) == 1
    assert outstanding[0]["chain_order"] == 0
    assert pq.read_table(parquet_path).num_rows == 0


def test_abort_summary_does_not_swallow_a_second_keyboard_interrupt(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["synth_client"] = _StubSynthClient(
        raise_for=AxBenchBaseError("initial provider failure")
    )
    original_record_counter = positive_mode.record_counter

    def interrupt_summary(name, *args, **kwargs):
        if name == "stage1b.summary.eligible":
            raise KeyboardInterrupt("second Ctrl-C during abort summary")
        return original_record_counter(name, *args, **kwargs)

    monkeypatch.setattr(
        positive_mode,
        "record_counter",
        interrupt_summary,
    )

    with pytest.raises(
        KeyboardInterrupt,
        match="second Ctrl-C during abort summary",
    ):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_stage1b_emergency_checkpoint_failure_preserves_original_error(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    class _InterruptingClient:
        def complete(self, *_args, **_kwargs):
            raise KeyboardInterrupt("primary stage failure")

    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "emergency-failure"),
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["synth_client"] = _InterruptingClient()
    original_write = positive_checkpoint.write_checkpoint
    write_calls = 0

    def fail_second_write(**kwargs):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("synthetic emergency checkpoint failure")
        return original_write(**kwargs)

    monkeypatch.setattr(
        positive_checkpoint,
        "write_checkpoint",
        fail_second_write,
    )
    with pytest.raises(KeyboardInterrupt, match="primary stage failure") as exc_info:
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    assert write_calls == 2
    assert any(
        "emergency checkpoint failed" in note
        for note in (exc_info.value.__notes__ or [])
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    assert output_paths.stage1b_partial_json().exists()
    assert output_paths.stage1b_partial_parquet().exists()


def test_stage1b_complete_checkpoint_republishes_without_provider_work(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "final-replay"),
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    original_writer = positive_mode._write_training_parquet

    def fail_publication(*_args, **_kwargs):
        raise RuntimeError("synthetic final publication interruption")

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        fail_publication,
    )
    with pytest.raises(RuntimeError, match="publication interruption"):
        synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    checkpoint = json.loads(
        output_paths.stage1b_partial_json().read_text(encoding="utf-8")
    )
    assert checkpoint["accepted_pair_count"] == 1
    assert checkpoint["concept_states"][0]["tracker"]["outstanding"] == []

    monkeypatch.setattr(
        positive_mode,
        "_write_training_parquet",
        original_writer,
    )
    no_call_client = _StubSynthClient(
        raise_for=AssertionError("provider must not run")
    )
    stub_env["synth_client"] = no_call_client
    out = synthesize_positive_mode(
        cfg,
        [_concept()],
        [_seed_row(0)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert out.exists()
    assert no_call_client.calls == []
    assert not output_paths.stage1b_partial_json().exists()
    assert not output_paths.stage1b_partial_parquet().exists()


def test_stage1b_cleanup_warning_reports_authoritative_json_marker(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        output_root=str(tmp_path / "cleanup-warning"),
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )

    def fail_cleanup(**_kwargs):
        raise OSError("synthetic JSON unlink failure")

    monkeypatch.setattr(
        positive_checkpoint,
        "delete_checkpoint",
        fail_cleanup,
    )
    with capture_loguru_info() as captured:
        out = synthesize_positive_mode(
            cfg,
            [_concept()],
            [_seed_row(0)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    assert out.exists()
    assert output_paths.stage1b_partial_json().exists()
    assert output_paths.stage1b_partial_parquet().exists()
    warning = next(
        line for line in captured if "partial-checkpoint cleanup failed" in line
    )
    assert "JSON marker exists=True" in warning
    assert "parquet exists=True" in warning
    assert "authoritative committed checkpoint" in warning


def test_stage1b_periodic_checkpoint_follows_terminal_refill(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=3,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    concept = _concept()
    seeds = [_seed_row(index, text=f"seed-{index}") for index in range(3)]
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache(
            {seed.prefix_text: 0.0 for seed in seeds}
        ),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    writes: list[dict[str, Any]] = []
    validation_batch_sizes: list[int] = []
    original_write = positive_checkpoint.write_checkpoint
    original_validate = positive_mode._validate_checkpoint_ordinary_rows

    def capture_write(**kwargs):
        writes.append(json.loads(json.dumps(kwargs["document"])))
        return original_write(**kwargs)

    def capture_validation(cfg_arg, rows):
        validation_batch_sizes.append(len(rows))
        return original_validate(cfg_arg, rows)

    monkeypatch.setattr(
        positive_mode,
        "_STAGE1B_CHECKPOINT_TERMINAL_INTERVAL",
        2,
    )
    monkeypatch.setattr(
        positive_checkpoint,
        "write_checkpoint",
        capture_write,
    )
    monkeypatch.setattr(
        positive_mode,
        "_validate_checkpoint_ordinary_rows",
        capture_validation,
    )
    rows = positive_mode._run_positive_target_refill_engine(
        cfg,
        [tracker],
        object(),
        object(),
        stub_env["synth_client"],
        None,
        _stub_meta(),
        checkpoint_fingerprint={"test": "periodic"},
        checkpoint_json_path=tmp_path / "partial.json",
        checkpoint_parquet_path=tmp_path / "partial.parquet",
    )

    assert len(rows[concept.concept_id]) == 6
    assert [write["accepted_pair_count"] for write in writes] == [0, 2, 3]
    assert validation_batch_sizes == [2, 2, 2]
    periodic = writes[1]
    assert periodic["next_chain_index"] == 3
    periodic_outstanding = periodic["concept_states"][0]["tracker"][
        "outstanding"
    ]
    assert len(periodic_outstanding) == 1
    assert periodic_outstanding[0]["chain_order"] == 2


def test_stage1b_periodic_checkpoint_failure_aborts_with_coherent_emergency_save(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=2,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    concept = _concept()
    seeds = [_seed_row(index, text=f"periodic-failure-{index}") for index in range(2)]
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 0.0 for seed in seeds}),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    json_path = tmp_path / "periodic-failure.json"
    parquet_path = tmp_path / "periodic-failure.parquet"
    original_write = positive_checkpoint.write_checkpoint
    write_calls = 0

    def fail_periodic_write(**kwargs):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("synthetic periodic checkpoint failure")
        return original_write(**kwargs)

    monkeypatch.setattr(
        positive_mode,
        "_STAGE1B_CHECKPOINT_TERMINAL_INTERVAL",
        1,
    )
    monkeypatch.setattr(
        positive_checkpoint,
        "write_checkpoint",
        fail_periodic_write,
    )
    with pytest.raises(OSError, match="periodic checkpoint failure"):
        positive_mode._run_positive_target_refill_engine(
            cfg,
            [tracker],
            object(),
            object(),
            stub_env["synth_client"],
            None,
            _stub_meta(),
            checkpoint_fingerprint={"test": "periodic-failure"},
            checkpoint_json_path=json_path,
            checkpoint_parquet_path=parquet_path,
        )

    assert write_calls == 3
    document = json.loads(json_path.read_text(encoding="utf-8"))
    tracker_state = document["concept_states"][0]["tracker"]
    assert document["accepted_pair_count"] == 1
    assert len(tracker_state["outstanding"]) == 1
    assert pq.read_table(parquet_path).num_rows == 2


def test_stage1b_final_forced_checkpoint_failure_preserves_complete_state(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    concept = _concept()
    seed = _seed_row(0)
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        [seed],
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 0.0}),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, [seed]),
    )
    json_path = tmp_path / "final-forced-failure.json"
    parquet_path = tmp_path / "final-forced-failure.parquet"
    original_write = positive_checkpoint.write_checkpoint
    write_calls = 0

    def fail_final_forced_write(**kwargs):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("synthetic final forced checkpoint failure")
        return original_write(**kwargs)

    monkeypatch.setattr(
        positive_checkpoint,
        "write_checkpoint",
        fail_final_forced_write,
    )
    with pytest.raises(OSError, match="final forced checkpoint failure"):
        positive_mode._run_positive_target_refill_engine(
            cfg,
            [tracker],
            object(),
            object(),
            stub_env["synth_client"],
            None,
            _stub_meta(),
            checkpoint_fingerprint={"test": "final-forced-failure"},
            checkpoint_json_path=json_path,
            checkpoint_parquet_path=parquet_path,
        )

    assert write_calls == 3
    document = json.loads(json_path.read_text(encoding="utf-8"))
    tracker_state = document["concept_states"][0]["tracker"]
    assert document["accepted_pair_count"] == 1
    assert tracker_state["outstanding"] == []
    assert pq.read_table(parquet_path).num_rows == 2


def test_stage1b_source_restore_skips_scoring_and_preserves_next_batch(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    concept = _concept()
    seeds = [
        _seed_row(index, text=f"candidate-{index}", sae_ids=[])
        for index in range(80)
    ]
    percentiles = {seed.prefix_text: 0.0 for seed in seeds}
    original_cache = _FakePrefixCache(percentiles)
    original_cache.batch_size = 2
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=original_cache,
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    works, next_order = positive_mode._admit_positive_work_items(
        tracker,
        start_order=0,
    )
    # Consume the compatibility buffer through replacements, leaving its last
    # candidate outstanding and the next physical scorer batch untouched.
    for _ in range(71):
        work = works[0]
        work.rich_semantic_attempts = 1
        tracker.semantic_rich_attempts += 1
        tracker.complete_failure(work, side="rich")
        works, next_order = positive_mode._admit_positive_work_items(
            tracker,
            start_order=next_order,
        )
        assert len(works) == 1
    document, rows = positive_mode._stage1b_checkpoint_snapshot(
        cfg,
        [tracker],
        fingerprint={"test": "source-restore"},
        highest_requested_target=cfg.n_positives_per_concept,
        next_chain_index=next_order,
        next_acceptance_sequence=0,
    )

    def advance_to_next_scoring_batch(
        live_tracker,
        *,
        start_order: int,
    ) -> int:
        order = start_order
        work = next(iter(live_tracker.outstanding.values()))
        work.rich_semantic_attempts = 1
        live_tracker.semantic_rich_attempts += 1
        live_tracker.complete_failure(work, side="rich")
        works, order = positive_mode._admit_positive_work_items(
            live_tracker,
            start_order=order,
        )
        assert len(works) == 1
        return order

    advance_to_next_scoring_batch(tracker, start_order=next_order)
    expected_next_batch = original_cache.calls[-1]

    restored_cache = _FakePrefixCache(percentiles)
    restored_cache.batch_size = 2
    restored = positive_mode._restore_positive_target_trackers(
        cfg,
        [concept],
        seeds,
        object(),
        object(),
        None,
        _stub_meta(),
        prefix_cache=restored_cache,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
        cluster_contract=None,
        checkpoint_document=document,
        checkpoint_rows=rows,
    )[0]
    assert restored_cache.calls == []
    assert restored.discarded_rich == 71
    assert restored.seeds_drawn == 72
    advance_to_next_scoring_batch(restored, start_order=next_order)
    assert restored_cache.calls == [expected_next_batch]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("concept_population", "concept population length mismatch"),
        ("concept_identity", "does not match current canonical concept"),
        ("clean_cursor", "clean_cursor=.*outside"),
        ("clean_scored", "clean_scored=.*does not equal"),
        ("dirty_complete_count", "dirty scoring is complete"),
        ("dirty_before_scoring", "dirty_scored=1 before dirty scoring"),
        ("exhausted", "persisted exhausted=.*does not match"),
        ("seed_idx", "is not an eligible current seed"),
        ("prefix", "persisted prefix differs"),
        ("partition", "persisted clean/dirty partition differs"),
        ("percentile", "invalid raw percentile"),
        ("candidate_rank", "candidate ranks do not cover"),
        ("headroom", "headroom_rejected"),
        ("next_candidate_rank", "headroom_rejected"),
    ],
)
def test_stage1b_source_restore_corruption_fails_before_scoring(
    tmp_path,
    mutation: str,
    match: str,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=2)
    concept = _concept()
    seeds = [
        _seed_row(index, text=f"restore-corruption-{index}")
        for index in range(80)
    ]
    source_cache = _FakePrefixCache(
        {seed.prefix_text: 0.0 for seed in seeds}
    )
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=source_cache,
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    document, rows = positive_mode._stage1b_checkpoint_snapshot(
        cfg,
        [tracker],
        fingerprint={"test": "source-restore-corruption"},
        highest_requested_target=cfg.n_positives_per_concept,
        next_chain_index=0,
        next_acceptance_sequence=0,
    )
    state = document["concept_states"][0]
    source = state["source"]
    buffered = source["buffered_candidates"]

    if mutation == "concept_population":
        document["concept_states"] = []
    elif mutation == "concept_identity":
        state["feature_id"] += 1
    elif mutation == "clean_cursor":
        source["clean_cursor"] = len(seeds) + 1
        source["clean_scored"] = len(seeds) + 1
    elif mutation == "clean_scored":
        source["clean_scored"] += 1
    elif mutation == "dirty_complete_count":
        source["dirty_scored_once"] = True
        source["dirty_scored"] = 1
    elif mutation == "dirty_before_scoring":
        source["dirty_scored"] = 1
    elif mutation == "exhausted":
        source["exhausted"] = True
    elif mutation == "seed_idx":
        buffered[0]["seed_idx"] = len(seeds) + 10
    elif mutation == "prefix":
        buffered[0]["prefix_text"] = "different bounded prefix"
    elif mutation == "partition":
        buffered[0]["stored_clean"] = False
    elif mutation == "percentile":
        buffered[0]["seed_percentile"] = math.inf
    elif mutation == "candidate_rank":
        buffered[0]["candidate_rank"] = 999
    elif mutation == "headroom":
        source["headroom_rejected"] = 1
    elif mutation == "next_candidate_rank":
        source["next_candidate_rank"] += 1
    else:  # pragma: no cover - parameter list is the closed mutation set.
        raise AssertionError(mutation)

    restored_cache = _FakePrefixCache(
        {seed.prefix_text: 0.0 for seed in seeds}
    )
    with pytest.raises(AxBenchBaseError, match=match):
        positive_mode._restore_positive_target_trackers(
            cfg,
            [concept],
            seeds,
            object(),
            object(),
            None,
            _stub_meta(),
            prefix_cache=restored_cache,
            effective_prefixes=positive_mode._resolve_stage1b_prefixes(
                cfg,
                seeds,
            ),
            cluster_contract=None,
            checkpoint_document=document,
            checkpoint_rows=rows,
        )
    assert restored_cache.calls == []


def test_stage1b_clustered_outstanding_resume_rebinds_target_without_scoring(
    tmp_path,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    concept = _concept(feature_index=2)
    target = _cluster_target()
    seeds = [
        _seed_row(0, text="cluster-dirty", sae_ids=[5]),
        _seed_row(1, text="cluster-clean", sae_ids=[]),
    ]
    cache = _FakePrefixCache({seed.prefix_text: 0.0 for seed in seeds})
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=cache,
        cluster_target=target,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    works, next_order = positive_mode._admit_positive_work_items(
        tracker,
        start_order=0,
    )
    assert works[0].pending.seed_idx == 1
    document, rows = positive_mode._stage1b_checkpoint_snapshot(
        cfg,
        [tracker],
        fingerprint={"test": "clustered-resume"},
        highest_requested_target=cfg.n_positives_per_concept,
        next_chain_index=next_order,
        next_acceptance_sequence=0,
    )

    class _Contract:
        feature_contract_fingerprint = target.feature_contract_fingerprint

        @staticmethod
        def target_for_representative(feature_id: int) -> ClusterFeatureTarget:
            assert feature_id == target.representative_id
            return target

    contract: Any = _Contract()
    restored_cache = _FakePrefixCache(
        {seed.prefix_text: 0.0 for seed in seeds}
    )
    grown_cfg = cfg.replace(n_positives_per_concept=2)
    restored = positive_mode._restore_positive_target_trackers(
        grown_cfg,
        [concept],
        seeds,
        object(),
        object(),
        None,
        _stub_meta(),
        prefix_cache=restored_cache,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            grown_cfg,
            seeds,
        ),
        cluster_contract=contract,
        checkpoint_document=document,
        checkpoint_rows=rows,
    )[0]

    assert restored_cache.calls == []
    restored_work = next(iter(restored.outstanding.values()))
    assert restored.target == 2
    assert restored.cluster_target is target
    assert restored_work.cluster_target is target
    assert restored_work.pending.seed_idx == 1


def _mixed_checkpoint_tracker(tmp_path):
    """Build one valid accepted/discarded/outstanding checkpoint state."""

    cfg = _cfg(tmp_path, n_positives_per_concept=3)
    concept = _concept()
    seeds = [_seed_row(index, text=f"mixed-{index}") for index in range(80)]
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 0.0 for seed in seeds}),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    works, next_order = positive_mode._admit_positive_work_items(
        tracker,
        start_order=0,
    )
    accepted, discarded, outstanding = works

    accepted.rich_row = {"qc_attempts": 1}
    accepted.avoiding_row = {
        "qc_attempts": 0,
        "gen_source": "corpus:doc_continuation@0/0",
    }
    accepted.rich_semantic_attempts = 1
    tracker.semantic_rich_attempts += 1
    tracker.complete_success(
        accepted,
        natural_avoid=True,
        acceptance_sequence=0,
    )

    discarded.rich_semantic_attempts = 2
    tracker.semantic_rich_attempts += 2
    tracker.complete_failure(discarded, side="rich")

    outstanding.rich_semantic_attempts = 1
    tracker.semantic_rich_attempts += 1
    return cfg, tracker, accepted, discarded, outstanding, next_order


@pytest.mark.parametrize("highest_requested_target", [True, 0, -1])
def test_stage1b_snapshot_rejects_invalid_highest_requested_target(
    tmp_path,
    highest_requested_target: int,
) -> None:
    cfg, tracker, _accepted, _discarded, _outstanding, next_order = (
        _mixed_checkpoint_tracker(tmp_path)
    )

    with pytest.raises(
        AxBenchBaseError,
        match="invalid highest_requested_target",
    ):
        positive_mode._stage1b_checkpoint_snapshot(
            cfg,
            [tracker],
            fingerprint={"test": "invalid-highest-target"},
            highest_requested_target=highest_requested_target,
            next_chain_index=next_order,
            next_acceptance_sequence=1,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate_concept", "duplicate concept IDs"),
        ("clean_count", "clean_scored"),
        ("dirty_count", "dirty_scored=1 before dirty scoring"),
        ("headroom_count", "headroom_rejected"),
        ("rank_omission", "candidate ranks do not cover"),
        ("rank_overlap", "ownership sets overlap"),
        ("discard_counter", "discard counters mismatch"),
        ("semantic_counter", "semantic counters mismatch"),
        ("natural_counter", "natural-avoid counter mismatch"),
        ("missing_sequence", "missing/duplicate acceptance sequence"),
        ("half_pair", "cannot persist a half-pair"),
        ("rich_attempts", "rich-attempt mismatch"),
        ("avoid_attempts", "avoid-attempt mismatch"),
        ("discard_side", "invalid discard side"),
        ("outstanding_terminal", "marks an outstanding chain terminal"),
        ("outstanding_sequence", "outstanding chain parquet state"),
        ("chain_bound", "next_chain_index does not exceed"),
        ("bad_zero_hit", "inconsistent zero-hit termination"),
        ("premature_pool_exhaustion", "pool exhaustion prematurely"),
        ("bad_termination", "invalid termination reason"),
        ("noncontiguous_sequence", "acceptance sequences are not contiguous"),
        ("next_sequence", "next_acceptance_sequence mismatch"),
        ("wrong_tracker", "wrong tracker"),
    ],
)
def test_stage1b_snapshot_invariants_fail_closed(
    tmp_path,
    mutation: str,
    match: str,
) -> None:
    cfg, tracker, accepted, _discarded, outstanding, next_order = (
        _mixed_checkpoint_tracker(tmp_path)
    )
    trackers = [tracker]
    next_sequence = 1

    if mutation == "duplicate_concept":
        trackers.append(tracker)
    elif mutation == "clean_count":
        tracker.source.clean_scored += 1
    elif mutation == "dirty_count":
        tracker.source.dirty_scored = 1
    elif mutation == "headroom_count":
        tracker.source.headroom_rejected = 1
    elif mutation == "rank_omission":
        accepted.pending.selected = replace(
            accepted.pending.selected,
            candidate_rank=999,
        )
    elif mutation == "rank_overlap":
        outstanding.pending.selected = replace(
            outstanding.pending.selected,
            candidate_rank=accepted.pending.selected.candidate_rank,
        )
    elif mutation == "discard_counter":
        tracker.discarded_rich += 1
    elif mutation == "semantic_counter":
        tracker.semantic_rich_attempts += 1
    elif mutation == "natural_counter":
        tracker.natural_avoid_acceptances = 0
    elif mutation == "missing_sequence":
        accepted.acceptance_sequence = None
    elif mutation == "half_pair":
        accepted.rich_row = None
    elif mutation == "rich_attempts":
        accepted.rich_semantic_attempts += 1
        tracker.semantic_rich_attempts += 1
    elif mutation == "avoid_attempts":
        accepted.avoid_semantic_attempts = 1
        tracker.semantic_avoid_attempts = 1
    elif mutation == "discard_side":
        tracker.discard_records[0] = replace(
            tracker.discard_records[0],
            side="unknown",
        )
        tracker.discarded_rich = 0
    elif mutation == "outstanding_terminal":
        outstanding.terminal = True
    elif mutation == "outstanding_sequence":
        outstanding.acceptance_sequence = 1
    elif mutation == "chain_bound":
        next_order = outstanding.order
    elif mutation == "bad_zero_hit":
        tracker.termination_reason = "zero_hit_guard"
    elif mutation == "premature_pool_exhaustion":
        tracker.termination_reason = "pool_exhausted"
    elif mutation == "bad_termination":
        tracker.termination_reason = "unknown"
    elif mutation == "noncontiguous_sequence":
        accepted.acceptance_sequence = 1
    elif mutation == "next_sequence":
        next_sequence = 2
    elif mutation == "wrong_tracker":
        accepted.concept = _concept(concept_id=99)
    else:  # pragma: no cover - the parameter list is the closed state set.
        raise AssertionError(mutation)

    with pytest.raises(AxBenchBaseError, match=match):
        positive_mode._stage1b_checkpoint_snapshot(
            cfg,
            trackers,
            fingerprint={"test": "snapshot-invariants"},
            highest_requested_target=cfg.n_positives_per_concept,
            next_chain_index=next_order,
            next_acceptance_sequence=next_sequence,
        )


def test_stage1b_checkpoints_before_initial_request_construction(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path, n_positives_per_concept=1)
    concept = _concept()
    seed = _seed_row(0)
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        [seed],
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=_FakePrefixCache({seed.prefix_text: 0.0}),
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            cfg, [seed]
        ),
    )

    def fail_request_construction(*_args, **_kwargs):
        raise RuntimeError("synthetic request-construction failure")

    monkeypatch.setattr(
        positive_mode,
        "_positive_rich_chain_request",
        fail_request_construction,
    )
    json_path = tmp_path / "partial.json"
    parquet_path = tmp_path / "partial.parquet"
    with pytest.raises(RuntimeError, match="request-construction failure"):
        positive_mode._run_positive_target_refill_engine(
            cfg,
            [tracker],
            object(),
            object(),
            stub_env["synth_client"],
            None,
            _stub_meta(),
            checkpoint_fingerprint={"test": "request-construction"},
            checkpoint_json_path=json_path,
            checkpoint_parquet_path=parquet_path,
        )

    checkpoint = json.loads(json_path.read_text(encoding="utf-8"))
    outstanding = checkpoint["concept_states"][0]["tracker"]["outstanding"]
    assert checkpoint["next_chain_index"] == 1
    assert len(outstanding) == 1
    assert outstanding[0]["chain_order"] == 0
    assert pq.read_table(parquet_path).num_rows == 0


def test_stage1b_refill_scorer_failure_leaves_coherent_checkpoint(
    stub_env,
    tmp_path,
) -> None:
    class _FailingRefillPrefixCache(_FakePrefixCache):
        def get_many(
            self,
            prefix_texts: list[str],
            f_c: int,
            *,
            cluster_target: ClusterFeatureTarget | None = None,
        ) -> list[float]:
            if self.calls:
                raise RuntimeError("synthetic refill scorer failure")
            return super().get_many(
                prefix_texts,
                f_c,
                cluster_target=cluster_target,
            )

    cfg = _cfg(
        tmp_path,
        n_positives_per_concept=72,
        positive_max_attempts_per_seed=1,
        accept_threshold=0.5,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    concept = _concept()
    seeds = [_seed_row(index, text=f"seed-{index}") for index in range(73)]
    prefix_cache = _FailingRefillPrefixCache(
        {seed.prefix_text: 0.0 for seed in seeds}
    )
    prefix_cache.batch_size = 72
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        object(),
        object(),
        None,
        sae_layer=8,
        prefix_cache=prefix_cache,
        cluster_target=None,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(cfg, seeds),
    )
    stub_env["percentile_pairs"] = [(0.0, 0.0)]
    json_path = tmp_path / "partial.json"
    parquet_path = tmp_path / "partial.parquet"

    with pytest.raises(RuntimeError, match="refill scorer failure"):
        positive_mode._run_positive_target_refill_engine(
            cfg,
            [tracker],
            object(),
            object(),
            stub_env["synth_client"],
            None,
            _stub_meta(),
            checkpoint_fingerprint={
                "checkpoint_algorithm_version": (
                    positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
                ),
                "test": "refill-scoring",
            },
            checkpoint_json_path=json_path,
            checkpoint_parquet_path=parquet_path,
        )

    checkpoint = json.loads(json_path.read_text(encoding="utf-8"))
    state = checkpoint["concept_states"][0]
    assert state["source"]["clean_cursor"] == 72
    assert state["source"]["clean_scored"] == 72
    assert state["tracker"]["discarded_rich"] == 1
    assert len(state["tracker"]["discarded"]) == 1
    assert len(state["tracker"]["outstanding"]) == 71
    loaded = positive_checkpoint.load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint={
            "checkpoint_algorithm_version": (
                positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
            ),
            "test": "refill-scoring",
        },
        training_schema=positive_mode.TRAINING_SCHEMA,
        current_target=72,
        legacy_fingerprint_factory=lambda _version: {
            "checkpoint_algorithm_version": (
                positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
            ),
            "test": "refill-scoring",
        },
    )
    assert loaded is not None
