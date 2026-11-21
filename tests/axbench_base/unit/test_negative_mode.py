"""Unit tests for synthesize_negative_mode (Task 18)."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base import paths
from steering.eval.axbench_base._stage0c_internals import (
    _schema_with_completed_negative_prefixes_metadata,
)
from steering.eval.axbench_base.config import SynthConfig
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.synth import negative_mode
from steering.eval.axbench_base.synth.concept_yield import (
    concept_yield_report_is_committed,
)
from steering.eval.axbench_base.synth.negative_mode import (
    NEG_TRAINING_SCHEMA,
    _copy_concept_rich_rows,
    _exact_output_start,
    _load_negative_prefixes,
    _stable_hash_int,
    _synthesize_concept_negatives,
    synthesize_negative_mode,
)
from steering.eval.axbench_base.synth.qc import AvoidContinuationOutcome
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    TargetModelMetadata,
)
from tests.axbench_base.stage1_protocol_helpers import (
    TEST_STAGE1_TEMPLATE,
    frame_stage1_response,
)
from tests.axbench_base.unit.test_config import make_valid_config

_CANDIDATE_POOL_IDENTITY = "0" * 64


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


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


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


def _prefix_row(
    concept_id: int = 0,
    prefix_text: str = "hello world prefix",
    shard: int = 0,
    doc: int = 5,
    seg: int = 2,
    continuations: list[str] | None = None,
    percentiles: list[float] | None = None,
) -> dict[str, Any]:
    conts = (
        continuations
        if continuations is not None
        else ["cont A concept-rich", "cont B concept-rich", "cont C concept-rich"]
    )
    pcts = percentiles if percentiles is not None else [80.0, 85.0, 90.0]
    return {
        "concept_id": concept_id,
        "prefix_text": prefix_text,
        "prefix_source_shard": shard,
        "prefix_source_doc": doc,
        "prefix_activating_seg_idx": seg,
        "verification_continuations": conts,
        "verification_seeds": list(range(100, 100 + len(conts))),
        "verification_percentiles": pcts,
    }


class _StubSynthClient:
    """Deterministic stub for OpenRouterSynthClient.

    ``responses`` is a queue of canned responses returned in order.
    ``raise_for`` may be a single exception (raised on every call) or a
    list of per-call exceptions (each entry is either an exception
    instance or ``None``; the list is popped from the front).

    ``cache_key`` is accepted and recorded for the cache-integration tests.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        raise_for: Any = None,
    ) -> None:
        self.responses: list[str] = list(responses or [])
        self.raise_for = raise_for
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "cache_key": cache_key})
        if isinstance(self.raise_for, list) and self.raise_for:
            exc = self.raise_for.pop(0)
            if exc is not None:
                raise exc
        elif self.raise_for is not None and not isinstance(self.raise_for, list):
            raise self.raise_for
        if not self.responses:
            return "default negative continuation"
        return frame_stage1_response(prompt, self.responses.pop(0))


class _ConcurrentSynthClient:
    def __init__(self, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = threading.Lock()

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        with self.lock:
            self.calls.append({"prompt": prompt, "cache_key": cache_key})
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            threading.Event().wait(self.delay_s)
            return frame_stage1_response(prompt, "negative continuation")
        finally:
            with self.lock:
                self.in_flight -= 1


class _ShuffledPrefixSynthClient:
    """Complete later canonical prefixes first and record actual overlap."""

    def __init__(self, delays_by_prefix: dict[str, float]) -> None:
        self.delays_by_prefix = delays_by_prefix
        self.completion_order: list[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = threading.Lock()

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        del cache_key
        prefix_text = next(
            line.removeprefix("PREFIX=")
            for line in prompt.splitlines()
            if line.startswith("PREFIX=")
        )
        with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            threading.Event().wait(self.delays_by_prefix[prefix_text])
            with self.lock:
                self.completion_order.append(prefix_text)
            return frame_stage1_response(prompt, "deterministic avoiding continuation")
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

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
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
                actions = self.responses_by_prompt.get(
                    f"neg_template {prefix} :: {description}"
                )
            if not actions:
                raise AssertionError(f"unexpected synth call for prompt={prompt!r}")
            action = actions.pop(0)
            if isinstance(action, Exception):
                raise action
            return frame_stage1_response(prompt, action)
        finally:
            with self.lock:
                self.in_flight -= 1


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
        base = [ord(c) for c in text]
        return ([prepend] if prepend is not None else []) + base

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


# ---------------------------------------------------------------------------
# Shared stub environment
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """Replace the LLM + Gemma + parquet-loading plumbing with in-memory stubs."""
    state: dict[str, Any] = {
        "synth_client": _StubSynthClient(
            responses=["stub neg continuation"] * 100,
        ),
        # A sequence consumed by the stubbed classify_avoid_continuation.
        # Entries may be AvoidContinuationOutcome members or legacy bools
        # (True -> ACTIVE/reject, False -> INACTIVE/accept). If the test
        # runs more checks than the sequence has entries, the LAST value
        # is repeated.
        "fc_active_sequence": [False],
        "fc_active_idx": 0,
        "fc_active_calls": [],
        "negative_prefixes": [],  # populated per test
        "template": TEST_STAGE1_TEMPLATE,
        "parse_calls": [],
    }

    def _fake_build_client(synth_cfg, max_retries, response_cache=None):
        del synth_cfg, max_retries
        state["captured_response_cache"] = response_cache
        return state["synth_client"]

    def _fake_load_prefixes(run_tag):
        del run_tag
        return list(state["negative_prefixes"])

    def _to_outcome(entry):
        if isinstance(entry, negative_mode.AvoidContinuationOutcome):
            return entry
        # legacy bool: True -> ACTIVE (fired -> reject), False -> INACTIVE (accept)
        return (
            negative_mode.AvoidContinuationOutcome.ACTIVE
            if entry
            else negative_mode.AvoidContinuationOutcome.INACTIVE
        )

    def _fake_classify_avoid(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold=0.0,
        tokenizer_id,
        cluster_target=None,
    ):
        del base_model, base_sae, cluster_target
        state["fc_active_calls"].append(
            {
                "prefix_text": prefix_text,
                "continuation_text": continuation_text,
                "f_c": f_c,
                "sae_layer": sae_layer,
                "extraction_threshold": extraction_threshold,
                "tokenizer_id": tokenizer_id,
            }
        )
        idx = state["fc_active_idx"]
        state["fc_active_idx"] = idx + 1
        seq = state["fc_active_sequence"]
        entry = seq[idx] if idx < len(seq) else seq[-1]
        return _to_outcome(entry)

    def _fake_classify_avoids(
        inputs,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold=0.0,
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

    def _fake_template(name):
        del name
        return state["template"]

    def _fake_extraction_tokenizer(tokenizer_id):
        del tokenizer_id
        return _StubExtractionTokenizer()

    def _fake_get_tokenizer():
        return _StubNanochatTokenizer()

    monkeypatch.setattr(negative_mode, "OpenRouterSynthClient", _fake_build_client)
    monkeypatch.setattr(negative_mode, "_load_negative_prefixes", _fake_load_prefixes)
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuation",
        _fake_classify_avoid,
    )
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuations",
        _fake_classify_avoids,
    )
    monkeypatch.setattr(negative_mode, "load_template", _fake_template)
    monkeypatch.setattr(
        negative_mode,
        "get_extraction_tokenizer",
        _fake_extraction_tokenizer,
    )
    monkeypatch.setattr(negative_mode, "get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    return state


def _cfg(tmp_path, **overrides):
    base = make_valid_config(
        run_tag="test_neg_synth",
        output_root=None,
        max_retries=2,
        qc_fallback_on_exhaust="error",
        output_length=16,
    )
    if overrides:
        return base.replace(**overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: concept_rich row construction
# ---------------------------------------------------------------------------


def test_emits_concept_rich_rows_from_verification(stub_env, tmp_path):
    """One prefix with 3 verification continuations -> 3 concept_rich rows
    plus 1 concept_avoiding row (strict-zero passes on first attempt).
    """
    cfg = _cfg(tmp_path)
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["fc_active_sequence"] = [False]
    concepts = [_concept(0, 42)]

    out = synthesize_negative_mode(
        cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    content_classes = [r["content_class"] for r in rows]
    assert content_classes.count("concept_rich") == 3
    assert content_classes.count("concept_avoiding") == 1
    assert len(rows) == 4
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    filtered = json.loads(output_paths.stage1c_concepts().read_text(encoding="utf-8"))
    report = json.loads(output_paths.stage1c_concept_yield_report().read_text(encoding="utf-8"))
    assert filtered == [concepts[0].__dict__]
    assert report["status"] == "ok"
    assert report["surviving_concept_ids"] == [0]
    assert report["concepts"][0]["attempted"] == 1
    assert report["concepts"][0]["accepted"] == 1


def test_stage1c_parquet_failure_leaves_report_uncommitted(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["fc_active_sequence"] = [False]
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    corpus_path = output_paths.synth_training_negative()
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_bytes(b"previous committed corpus")

    def _fail_write(*_args, **_kwargs) -> None:
        raise RuntimeError("injected Stage 1c parquet failure")

    monkeypatch.setattr(negative_mode.pq, "write_table", _fail_write)

    with pytest.raises(RuntimeError, match="injected Stage 1c parquet failure"):
        synthesize_negative_mode(
            cfg,
            [_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )

    report_path = output_paths.stage1c_concept_yield_report()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "preparing"
    assert not concept_yield_report_is_committed(report_path)
    assert corpus_path.read_bytes() == b"previous committed corpus"


def test_concept_rich_rows_copy_percentiles(stub_env, tmp_path):
    """qc_output_percentile should match the input verification_percentiles."""
    cfg = _cfg(tmp_path)
    pcts = [72.5, 88.25, 99.0]
    stub_env["negative_prefixes"] = [_prefix_row(percentiles=pcts)]
    stub_env["fc_active_sequence"] = [False]

    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    rich_rows = [r for r in rows if r["content_class"] == "concept_rich"]
    assert sorted(r["qc_output_percentile"] for r in rich_rows) == pytest.approx(
        sorted(pcts),
        abs=1e-4,
    )
    # All concept_rich rows have 0 attempts + qc_accepted=True + gen_seed=-1.
    for r in rich_rows:
        assert r["qc_attempts"] == 0
        assert r["qc_accepted"] is True
        assert r["gen_seed"] == -1
        assert r["category"] == "positive"  # spec: concept_rich side
        assert r["qc_zero_activation_verified"] is False


def test_concept_rich_row_gen_source_from_decoder_only(stub_env, tmp_path):
    """gen_source for concept_rich rows should encode decoder_only dict."""
    cfg = _cfg(tmp_path).replace(
        decoder_only={"source": "mybase", "model_tag": "d20", "step": 999},
    )
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [False]
    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    rich = [r for r in rows if r["content_class"] == "concept_rich"]
    assert rich
    for r in rich:
        assert r["gen_source"] == "local:mybase/d20"


def test_concept_rich_row_gen_source_handles_missing_keys(stub_env, tmp_path):
    """Missing source/model_tag -> falls back to 'unknown'."""
    cfg = _cfg(tmp_path).replace(decoder_only={})
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [False]
    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    rich = [r for r in rows if r["content_class"] == "concept_rich"]
    assert rich[0]["gen_source"] == "local:unknown/unknown"


def test_copy_concept_rich_rows_helper_direct(stub_env, tmp_path):
    """_copy_concept_rich_rows helper tested directly for determinism."""
    cfg = _cfg(tmp_path)
    concept = _concept(concept_id=3, feature_index=77)
    prow = _prefix_row(
        concept_id=3,
        prefix_text="pref",
        shard=11,
        doc=123,
        seg=4,
        continuations=["c1", "c2"],
        percentiles=[10.0, 20.0],
    )
    rows = _copy_concept_rich_rows(concept, prow, cfg)
    assert len(rows) == 2
    for i, r in enumerate(rows):
        assert r["concept_id"] == 3
        assert r["qc_feature_id"] == 77
        assert r["prefix_text"] == "pref"
        assert r["output_text"] == f"c{i + 1}"
        assert r["prefix_source_shard"] == 11
        assert r["prefix_source_doc"] == 123
        assert r["prefix_source_seg_idx"] == 4


# ---------------------------------------------------------------------------
# Tests: concept_avoiding QC retry loop
# ---------------------------------------------------------------------------


def test_accept_on_first_attempt_concept_avoiding(stub_env, tmp_path):
    """LLM returns once, classify_avoid_continuation -> INACTIVE -> accept @ attempt 1."""
    cfg = _cfg(tmp_path)
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [False]
    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(avoiding) == 1
    assert avoiding[0]["qc_attempts"] == 1
    assert avoiding[0]["qc_zero_activation_verified"] is True
    # LLM called exactly once.
    assert len(stub_env["synth_client"].calls) == 1


def test_strict_zero_qc_receives_cfg_tokenizer_id(stub_env, tmp_path):
    cfg = _cfg(
        tmp_path,
        extraction_tokenizer_id="custom/stage1c-tokenizer",
    )
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [False]

    synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    assert stub_env["fc_active_calls"]
    assert {call["tokenizer_id"] for call in stub_env["fc_active_calls"]} == {
        cfg.extraction_tokenizer_id
    }


def test_synthesize_negative_mode_loads_prefixes_with_full_cfg(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Entrypoint must pass cfg so Stage 0c tokenizer metadata is validated."""
    cfg = _cfg(tmp_path, extraction_tokenizer_id="custom/stage1c-tokenizer")
    seen: dict[str, Any] = {}

    def _load_with_cfg(arg):
        seen["arg"] = arg
        assert arg is cfg
        return [_prefix_row()]

    monkeypatch.setattr(negative_mode, "_load_negative_prefixes", _load_with_cfg)
    synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    assert seen["arg"] is cfg


def test_retry_on_strict_zero_fail_then_accept(stub_env, tmp_path):
    """First attempt fc_active=True, second fc_active=False -> attempts=2 accepted."""
    cfg = _cfg(tmp_path, max_retries=2)
    stub_env["negative_prefixes"] = [_prefix_row()]
    # First call: fires (fail). Second: silent (accept).
    stub_env["fc_active_sequence"] = [True, False]

    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(avoiding) == 1
    assert avoiding[0]["qc_attempts"] == 2
    assert avoiding[0]["qc_zero_activation_verified"] is True
    assert len(stub_env["synth_client"].calls) == 2


def test_hard_error_on_retries_exhausted(stub_env, tmp_path):
    """All attempts fire -> error mode -> AxBenchBaseError."""
    cfg = _cfg(tmp_path, max_retries=1, qc_fallback_on_exhaust="error")
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [True]  # always active
    with pytest.raises(AxBenchBaseError, match="strict-zero QC"):
        synthesize_negative_mode(
            cfg,
            [_concept()],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_qc_fallback_warn_skips_row(stub_env, tmp_path):
    """Warn mode drops the concept, then the all-dropped run fails loudly."""
    cfg = _cfg(tmp_path, max_retries=1, qc_fallback_on_exhaust="warn")
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [True]
    with pytest.raises(AxBenchBaseError, match="population guard failed"):
        synthesize_negative_mode(
            cfg,
            [_concept()],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_qc_fallback_best_falls_through_to_warn(stub_env, tmp_path):
    """Best acts like warn for strict-zero QC; all-dropped runs fail loudly."""
    cfg = _cfg(tmp_path, max_retries=1, qc_fallback_on_exhaust="best")
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [True]
    with pytest.raises(AxBenchBaseError, match="population guard failed"):
        synthesize_negative_mode(
            cfg,
            [_concept()],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_llm_error_logged_and_continues(stub_env, tmp_path):
    """If the LLM raises on attempt 1, attempt 2 should still proceed."""
    cfg = _cfg(tmp_path, max_retries=1)
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["second try good"],
        raise_for=[AxBenchBaseError("openrouter timeout"), None],
    )
    stub_env["fc_active_sequence"] = [False]  # second attempt strict-zero passes
    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(avoiding) == 1
    assert avoiding[0]["output_text"] == "second try good"
    # attempts is the 1-indexed successful attempt (index 1 succeeded).
    assert avoiding[0]["qc_attempts"] == 2


def test_all_llm_calls_raise_exhausts(stub_env, tmp_path):
    """All transport failures drop the concept and trip the population guard."""
    cfg = _cfg(tmp_path, max_retries=1, qc_fallback_on_exhaust="warn")
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["synth_client"] = _StubSynthClient(
        raise_for=AxBenchBaseError("always down"),
    )
    # fc_active should never be called because no continuation is ever parsed.
    stub_env["fc_active_sequence"] = [False]
    with pytest.raises(AxBenchBaseError, match="population guard failed"):
        synthesize_negative_mode(
            cfg,
            [_concept()],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    # fc_active never called (no continuation was ever parsed).
    assert stub_env["fc_active_calls"] == []


# ---------------------------------------------------------------------------
# Tests: NO_TOKENS / empty-continuation rejection (issue #518)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_no_tokens_rejected_then_accept(stub_env, tmp_path, mc):
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["synth_client"] = _StubSynthClient(
        responses=["seam-merge text", "clean avoiding text"],
    )
    # attempt 0 -> NO_TOKENS (reject), attempt 1 -> INACTIVE (accept)
    stub_env["fc_active_sequence"] = [
        negative_mode.AvoidContinuationOutcome.NO_TOKENS,
        negative_mode.AvoidContinuationOutcome.INACTIVE,
    ]
    out = synthesize_negative_mode(
        cfg,
        [_concept(0, 42)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(avoiding) == 1
    assert avoiding[0]["output_text"] == "clean avoiding text"
    assert avoiding[0]["qc_attempts"] == 2
    assert avoiding[0]["qc_zero_activation_verified"] is True


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_empty_continuation_rejected_then_accept(stub_env, tmp_path, mc):
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["synth_client"] = _StubSynthClient(responses=["   ", "good avoiding"])
    stub_env["fc_active_sequence"] = [negative_mode.AvoidContinuationOutcome.INACTIVE]
    out = synthesize_negative_mode(
        cfg,
        [_concept(0, 42)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(avoiding) == 1
    assert avoiding[0]["output_text"] == "good avoiding"
    # The empty attempt never reached the classifier (only the accepted attempt did).
    assert len(stub_env["fc_active_calls"]) == 1
    assert stub_env["fc_active_calls"][0]["continuation_text"] == "good avoiding"


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_all_no_tokens_exhausts_error(stub_env, tmp_path, mc):
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        qc_fallback_on_exhaust="error",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["fc_active_sequence"] = [negative_mode.AvoidContinuationOutcome.NO_TOKENS]
    with pytest.raises(AxBenchBaseError, match="strict-zero QC"):
        synthesize_negative_mode(
            cfg,
            [_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_all_no_tokens_exhausts_warn_skips(stub_env, tmp_path, mc):
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["fc_active_sequence"] = [negative_mode.AvoidContinuationOutcome.NO_TOKENS]
    with pytest.raises(AxBenchBaseError, match="population guard failed"):
        synthesize_negative_mode(
            cfg,
            [_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_drops_exhausted_keeps_paired_in_order(stub_env, tmp_path, monkeypatch, mc):
    """#508: a fully-avoid-exhausted concept is dropped while a paired concept
    is kept, in BOTH serial (mc=1) and parallel (mc=4) paths, with the kept
    concept's row order preserved (rich*, then avoid)."""
    Outcome = negative_mode.AvoidContinuationOutcome
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=0, prefix_text="PREFIX-ZERO"),
        _prefix_row(concept_id=1, prefix_text="PREFIX-ONE"),
    ]
    stub_env["synth_client"] = _StubSynthClient(responses=["avoid text"] * 20)

    def _classify_by_prefix(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold=0.0,
        tokenizer_id,
    ):
        del continuation_text, f_c, base_model, base_sae
        del sae_layer, extraction_threshold, tokenizer_id
        return Outcome.INACTIVE if "ZERO" in prefix_text else Outcome.ACTIVE

    monkeypatch.setattr(negative_mode, "classify_avoid_continuation", _classify_by_prefix)
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuations",
        lambda inputs, base_model, base_sae, *, sae_layer,
        extraction_threshold=0.0, tokenizer_id,
        profile_region_name="synth_qc.classify_avoid_continuations": [
            _classify_by_prefix(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                base_model,
                base_sae,
                sae_layer=sae_layer,
                extraction_threshold=extraction_threshold,
                tokenizer_id=tokenizer_id,
            )
            for item in inputs
        ],
    )

    out_dir = tmp_path / "prof_drop"
    with AxBenchProfiler(out_dir):
        out = synthesize_negative_mode(
            cfg,
            [_concept(0, 42), _concept(1, 43)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    rows = pq.read_table(out).to_pylist()
    assert {r["concept_id"] for r in rows} == {0}, "concept 1 must be dropped"
    c0 = [r["content_class"] for r in rows if r["concept_id"] == 0]
    assert c0 == ["concept_rich", "concept_rich", "concept_rich", "concept_avoiding"]
    # Teeth for record_counter('stage1c.rows_for_concept', ..., concept_id=...):
    # events.jsonl gives per-concept granularity (summary.yaml aggregates all).
    events = [
        json.loads(line) for line in (out_dir / "events.jsonl").read_text().splitlines() if line
    ]
    rows_by_concept = {
        e["attrs"]["concept_id"]: int(e["value"])
        for e in events
        if e.get("type") == "counter" and e.get("name") == "stage1c.rows_for_concept"
    }
    assert rows_by_concept.get(0) == 4, (
        f"concept 0 should have 4 rows (3 rich + 1 avoiding); got {rows_by_concept}"
    )
    assert rows_by_concept.get(1) == 0, (
        f"concept 1 should have 0 rows (all ACTIVE → dropped); got {rows_by_concept}"
    )


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_parallel_multiconcept_order_no_drop(stub_env, tmp_path, monkeypatch, mc):
    """Both concepts present — no drop — row order is canonical (rich* then avoid) for each.

    Tests that the parallel (mc=4) and serial (mc=1) paths both produce the
    same canonical per-concept ordering when neither concept is exhausted.
    """
    Outcome = negative_mode.AvoidContinuationOutcome
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=0, prefix_text="PREFIX-ZERO"),
        _prefix_row(concept_id=1, prefix_text="PREFIX-ONE"),
    ]
    stub_env["synth_client"] = _StubSynthClient(responses=["avoid text"] * 20)

    # BOTH concepts get INACTIVE -> both produce avoiding rows -> neither is dropped.
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuation",
        lambda prefix_text, continuation_text, f_c, base_model, base_sae, *, sae_layer, extraction_threshold=0.0, tokenizer_id: (
            Outcome.INACTIVE
        ),
    )

    out = synthesize_negative_mode(
        cfg,
        [_concept(0, 42), _concept(1, 43)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    # Both concepts must be present.
    assert {r["concept_id"] for r in rows} == {0, 1}, (
        "Both concepts should be present (neither exhausted)"
    )
    # Each concept must have canonical order: rich rows first, then avoiding.
    for cid in (0, 1):
        classes = [r["content_class"] for r in rows if r["concept_id"] == cid]
        assert classes, f"concept {cid} has no rows"
        # Rich rows (from _copy_concept_rich_rows) come before avoiding rows.
        avoiding_indices = [i for i, c in enumerate(classes) if c == "concept_avoiding"]
        rich_indices = [i for i, c in enumerate(classes) if c == "concept_rich"]
        if avoiding_indices and rich_indices:
            assert max(rich_indices) < min(avoiding_indices), (
                f"concept {cid}: rich rows must precede avoiding rows; got order: {classes}"
            )


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_qc_counter_mapping_per_outcome(stub_env, tmp_path, mc):
    Outcome = negative_mode.AvoidContinuationOutcome
    cfg = _cfg(
        tmp_path,
        max_retries=3,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["synth_client"] = _StubSynthClient(responses=["c0", "c1", "c2"] * 10)
    # attempt 0 ACTIVE (qc_rejects), 1 NO_TOKENS (qc_empty_rejects), 2 INACTIVE (qc_accepts)
    stub_env["fc_active_sequence"] = [Outcome.ACTIVE, Outcome.NO_TOKENS, Outcome.INACTIVE]
    out_dir = tmp_path / "prof"
    with AxBenchProfiler(out_dir):
        synthesize_negative_mode(
            cfg,
            [_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text())
    totals = {c["name"]: c["total"] for c in summary["counters"]}
    assert totals.get("stage1c.qc_rejects", 0) == 1
    assert totals.get("stage1c.qc_empty_rejects", 0) == 1
    assert totals.get("stage1c.qc_accepts", 0) == 1


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_strip_guard_increments_protocol_failures(stub_env, tmp_path, mc):
    """Pure whitespace is a protocol failure before strict-zero classification.
    """
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    # Whitespace response -> the .strip() empty guard rejects BEFORE the classifier
    # (no fc_active_sequence entry consumed); then a clean continuation is accepted.
    stub_env["synth_client"] = _StubSynthClient(responses=["   ", "good avoiding"])
    stub_env["fc_active_sequence"] = [negative_mode.AvoidContinuationOutcome.INACTIVE]
    out_dir = tmp_path / "prof_strip"
    with AxBenchProfiler(out_dir):
        out = synthesize_negative_mode(
            cfg,
            [_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text())
    totals = {c["name"]: c["total"] for c in summary["counters"]}
    # The authenticated parser owns empty-output rejection; QC never sees it.
    assert totals.get("stage1c.continuation_protocol_failures", 0) == 1
    assert totals.get("stage1c.qc_empty_rejects", 0) == 0
    assert totals.get("stage1c.qc_rejects", 0) == 0
    assert totals.get("stage1c.qc_accepts", 0) == 1
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(avoiding) == 1 and avoiding[0]["output_text"] == "good avoiding"
    # The classifier saw only the accepted (non-empty) continuation.
    assert [c["continuation_text"] for c in stub_env["fc_active_calls"]] == ["good avoiding"]


@pytest.mark.parametrize("mc", [1, 4])
def test_negative_qc_receives_cfg_extraction_threshold(stub_env, tmp_path, mc):
    cfg = _cfg(
        tmp_path,
        extraction_threshold=0.375,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["fc_active_sequence"] = [negative_mode.AvoidContinuationOutcome.INACTIVE]
    synthesize_negative_mode(
        cfg,
        [_concept(0, 42)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    assert stub_env["fc_active_calls"]
    assert {c["extraction_threshold"] for c in stub_env["fc_active_calls"]} == {0.375}


@pytest.mark.parametrize("mc", [1, 4])
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
@given(
    outcomes=st.lists(
        st.sampled_from(
            [
                negative_mode.AvoidContinuationOutcome.NO_TOKENS,
                negative_mode.AvoidContinuationOutcome.INACTIVE,
                negative_mode.AvoidContinuationOutcome.ACTIVE,
            ]
        ),
        min_size=1,
        max_size=4,
    ),
)
def test_negative_emits_row_iff_first_inactive(mc, outcomes, stub_env, tmp_path):
    cfg = _cfg(
        tmp_path,
        max_retries=len(outcomes) - 1,
        qc_fallback_on_exhaust="warn",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=mc),
    )
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=0)]
    stub_env["synth_client"] = _StubSynthClient(responses=["resp"] * (len(outcomes) + 2))
    stub_env["fc_active_sequence"] = list(outcomes)
    stub_env["fc_active_idx"] = 0
    expect_emit = negative_mode.AvoidContinuationOutcome.INACTIVE in outcomes
    if not expect_emit:
        with pytest.raises(AxBenchBaseError, match="population guard failed"):
            synthesize_negative_mode(
                cfg,
                [_concept(0, 42)],
                base_model=object(),
                base_sae=object(),
                target_model_meta=_stub_meta(),
            )
        return
    out = synthesize_negative_mode(
        cfg,
        [_concept(0, 42)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    # The stage accepts on the FIRST INACTIVE; NO_TOKENS/ACTIVE advance to the next attempt.
    assert (len(avoiding) == 1) == expect_emit
    if expect_emit:
        assert avoiding[0]["qc_zero_activation_verified"] is True


# ---------------------------------------------------------------------------
# Tests: schema + multi-concept + edge cases
# ---------------------------------------------------------------------------


def test_schema_includes_negative_mode_extras(stub_env, tmp_path):
    """Output table has the 4 negative-mode extra columns."""
    cfg = _cfg(tmp_path)
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [False]
    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    tbl = pq.read_table(out)
    assert "prefix_source_shard" in tbl.schema.names
    assert "prefix_source_doc" in tbl.schema.names
    assert "prefix_source_seg_idx" in tbl.schema.names
    assert "qc_zero_activation_verified" in tbl.schema.names
    # And all 15 Stage-1b columns are preserved.
    expected_stage1b_cols = {
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
    }
    assert expected_stage1b_cols.issubset(set(tbl.schema.names))
    # Full schema column set matches NEG_TRAINING_SCHEMA.
    assert set(tbl.schema.names) == set(NEG_TRAINING_SCHEMA.names)


def test_empty_prefixes_for_concept_logs_and_skips(stub_env, tmp_path):
    """A survivor missing its Stage 0c prefix is a corrupt handoff."""
    cfg = _cfg(tmp_path)
    # Only concept 1 has prefixes; concept 0 has none.
    stub_env["negative_prefixes"] = [_prefix_row(concept_id=1)]
    stub_env["fc_active_sequence"] = [False]
    with pytest.raises(AxBenchBaseError, match="has no verified prefixes"):
        synthesize_negative_mode(
            cfg,
            [_concept(0, 10), _concept(1, 20)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_missing_negative_prefixes_parquet_raises(tmp_path, monkeypatch):
    """_load_negative_prefixes raises a clear error if the parquet is missing."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(AxBenchBaseError, match="not found"):
        _load_negative_prefixes(_cfg(tmp_path, run_tag="nonexistent_run"))


def test_load_negative_prefixes_rejects_unversioned_parquet(tmp_path, monkeypatch):
    """Every Stage 1c load enforces the production provenance contract."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "load_test"
    cfg = _cfg(tmp_path, run_tag=run_tag)
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a tiny table using the same schema Stage 0c uses.
    schema = pa.schema(
        [
            ("concept_id", pa.int32()),
            ("prefix_text", pa.string()),
            ("prefix_source_shard", pa.int32()),
            ("prefix_source_doc", pa.int32()),
            ("prefix_activating_seg_idx", pa.int32()),
            ("verification_continuations", pa.list_(pa.string())),
            ("verification_seeds", pa.list_(pa.int64())),
            ("verification_percentiles", pa.list_(pa.float32())),
        ]
    )
    rows = [
        {
            "concept_id": 0,
            "prefix_text": "p0",
            "prefix_source_shard": 1,
            "prefix_source_doc": 2,
            "prefix_activating_seg_idx": 3,
            "verification_continuations": ["c1", "c2"],
            "verification_seeds": [100, 200],
            "verification_percentiles": [80.0, 90.0],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path)
    with pytest.raises(AxBenchBaseError, match="missing 'neuronpedia_model_id'"):
        _load_negative_prefixes(cfg)


def test_load_negative_prefixes_with_cfg_rejects_missing_tokenizer_metadata(
    tmp_path,
    monkeypatch,
):
    """Stage 1c refuses old Stage 0c artifacts without tokenizer metadata."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_tag="missing_tokenizer_meta")
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_prefix_row()]
    completed_schema = _schema_with_completed_negative_prefixes_metadata(
        cfg,
        candidate_pool_identity=_CANDIDATE_POOL_IDENTITY,
    )
    metadata = dict(completed_schema.metadata or {})
    metadata.pop(b"extraction_tokenizer_id")
    schema = completed_schema.with_metadata(metadata)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path)

    with pytest.raises(AxBenchBaseError, match="rerunning Stage 0c|Rerun Stage 0c"):
        _load_negative_prefixes(cfg)


def test_load_negative_prefixes_with_cfg_rejects_mismatched_tokenizer_metadata(
    tmp_path,
    monkeypatch,
):
    """Stage 1c refuses Stage 0c artifacts scoped to another tokenizer."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(
        tmp_path,
        run_tag="mismatched_tokenizer_meta",
        extraction_tokenizer_id="custom/current-tokenizer",
    )
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_prefix_row()]
    completed_schema = _schema_with_completed_negative_prefixes_metadata(
        cfg,
        candidate_pool_identity=_CANDIDATE_POOL_IDENTITY,
    )
    metadata = dict(completed_schema.metadata or {})
    metadata[b"extraction_tokenizer_id"] = b"custom/stale-tokenizer"
    schema = completed_schema.with_metadata(metadata)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path)

    with pytest.raises(AxBenchBaseError, match="Rerun Stage 0c"):
        _load_negative_prefixes(cfg)


@pytest.mark.parametrize(
    "missing_key",
    [
        "neuronpedia_model_id",
        "feature_extraction_model_id",
        "neuronpedia_source",
    ],
)
def test_load_negative_prefixes_with_cfg_rejects_missing_extractor_identity(
    tmp_path,
    monkeypatch,
    missing_key,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(tmp_path, run_tag=f"missing_{missing_key}")
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_schema = _schema_with_completed_negative_prefixes_metadata(
        cfg,
        candidate_pool_identity=_CANDIDATE_POOL_IDENTITY,
    )
    metadata = dict(completed_schema.metadata or {})
    metadata.pop(missing_key.encode("utf-8"))
    pq.write_table(
        pa.Table.from_pylist(
            [_prefix_row()],
            schema=completed_schema.with_metadata(metadata),
        ),
        out_path,
    )

    with pytest.raises(AxBenchBaseError, match=rf"missing '{missing_key}'"):
        _load_negative_prefixes(cfg)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("neuronpedia_model_id", "gemma-2-9b"),
        ("feature_extraction_model_id", "gemma-3-12b-pt"),
        ("neuronpedia_source", "34-gemmascope-2-res-16k"),
    ],
)
def test_load_negative_prefixes_with_cfg_rejects_extractor_identity_mismatch(
    tmp_path,
    monkeypatch,
    field,
    stale_value,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(
        tmp_path,
        run_tag=f"mismatch_{field}",
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-16k",
    )
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_schema = _schema_with_completed_negative_prefixes_metadata(
        cfg,
        candidate_pool_identity=_CANDIDATE_POOL_IDENTITY,
    )
    metadata = dict(completed_schema.metadata or {})
    metadata[field.encode("utf-8")] = stale_value.encode("utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [_prefix_row()],
            schema=completed_schema.with_metadata(metadata),
        ),
        out_path,
    )

    with pytest.raises(AxBenchBaseError, match=field):
        _load_negative_prefixes(cfg)


def test_load_negative_prefixes_with_cfg_accepts_split_extractor_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg(
        tmp_path,
        run_tag="split_identity",
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-16k",
    )
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_prefix_row()]
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=_schema_with_completed_negative_prefixes_metadata(
                cfg,
                candidate_pool_identity=_CANDIDATE_POOL_IDENTITY,
            ),
        ),
        out_path,
    )

    assert _load_negative_prefixes(cfg) == rows


def test_empty_concepts_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(AxBenchBaseError, match="empty concepts"):
        synthesize_negative_mode(
            cfg,
            [],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


def test_run_tag_none_raises(stub_env, tmp_path):
    cfg = _cfg(tmp_path).replace(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        synthesize_negative_mode(
            cfg,
            [_concept()],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_meta(),
        )


# ---------------------------------------------------------------------------
# Tests: helpers (stable hash + token count)
# ---------------------------------------------------------------------------


def test_stable_hash_int_is_deterministic():
    """Same input -> same output across invocations (and across sessions)."""
    assert _stable_hash_int("hello") == _stable_hash_int("hello")
    assert _stable_hash_int("hello") != _stable_hash_int("world")
    assert 0 <= _stable_hash_int("some text") < 1000
    # Custom mod range.
    assert 0 <= _stable_hash_int("x", mod=7) < 7


def test_stable_hash_int_matches_known_digest():
    """Sanity check the deterministic hash against a hand-computed value."""
    text = "deterministic check"
    expected = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000
    assert _stable_hash_int(text) == expected


def test_exact_output_start_includes_owned_bos(monkeypatch):
    """The synth boundary is derived from one combined encoding plus BOS."""

    class _TokStub:
        def get_bos_token_id(self) -> int:
            return 99

        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(token) for token in tokens)

    def _fake_get_tok():
        return _TokStub()

    monkeypatch.setattr(negative_mode, "get_tokenizer", _fake_get_tok)
    assert _exact_output_start("abc", "z") == 4  # BOS plus 3 prefix tokens


# ---------------------------------------------------------------------------
# Tests: unicode + multi-prefix behaviors
# ---------------------------------------------------------------------------


def test_unicode_prefix_roundtrips(stub_env, tmp_path):
    """A prefix with non-ASCII text should survive the full pipeline via parquet."""
    cfg = _cfg(tmp_path)
    stub_env["negative_prefixes"] = [
        _prefix_row(prefix_text="café résumé 日本語"),
    ]
    stub_env["fc_active_sequence"] = [False]
    out = synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    assert any("café" in r["prefix_text"] for r in rows)
    assert any("日本語" in r["prefix_text"] for r in rows)


def test_concept_avoiding_gen_seed_is_stable(stub_env, tmp_path):
    """concept_avoiding rows share a stable gen_seed across runs for the same prefix."""
    cfg = _cfg(tmp_path, max_retries=0)
    stub_env["negative_prefixes"] = [_prefix_row(prefix_text="same")]
    stub_env["fc_active_sequence"] = [False]

    out1 = synthesize_negative_mode(
        cfg,
        [_concept(0, 42)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    # Reset fc_active_idx so the second invocation starts clean.
    stub_env["fc_active_idx"] = 0
    stub_env["synth_client"] = _StubSynthClient(responses=["same"] * 10)
    out2 = synthesize_negative_mode(
        cfg,
        [_concept(0, 42)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows1 = pq.read_table(out1).to_pylist()
    rows2 = pq.read_table(out2).to_pylist()
    avoiding1 = [r for r in rows1 if r["content_class"] == "concept_avoiding"]
    avoiding2 = [r for r in rows2 if r["content_class"] == "concept_avoiding"]
    assert avoiding1[0]["gen_seed"] == avoiding2[0]["gen_seed"]


def test_multi_concept_rows_are_grouped(stub_env, tmp_path):
    """Interleaved concept rows in the input should be grouped per-concept."""
    cfg = _cfg(tmp_path)
    # Interleave prefixes across two concepts in the input parquet.
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=1, prefix_text="p1"),
        _prefix_row(concept_id=0, prefix_text="p0a"),
        _prefix_row(concept_id=1, prefix_text="p1b"),
        _prefix_row(concept_id=0, prefix_text="p0b"),
    ]
    stub_env["fc_active_sequence"] = [False]  # always pass strict-zero
    concepts = [_concept(0, 10), _concept(1, 20)]
    out = synthesize_negative_mode(
        cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    rows = pq.read_table(out).to_pylist()
    concept_0_rows = [r for r in rows if r["concept_id"] == 0]
    concept_1_rows = [r for r in rows if r["concept_id"] == 1]
    # Each prefix yields 3 concept_rich + 1 concept_avoiding = 4 rows.
    assert len(concept_0_rows) == 8
    assert len(concept_1_rows) == 8


def test_parallel_negative_chain_pool_honors_cap_and_restores_order(
    stub_env,
    tmp_path,
):
    """Out-of-order avoiding completions still write concept/prefix order."""
    synth_client = _ConcurrentSynthClient(delay_s=0.03)
    stub_env["synth_client"] = synth_client
    cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=2),
    )
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=0, prefix_text="p0"),
        _prefix_row(concept_id=1, prefix_text="p1"),
        _prefix_row(concept_id=2, prefix_text="p2"),
    ]
    stub_env["fc_active_sequence"] = [False]
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]

    out = synthesize_negative_mode(
        cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    rows = pq.read_table(out).to_pylist()
    avoiding = [row for row in rows if row["content_class"] == "concept_avoiding"]
    assert [row["concept_id"] for row in avoiding] == [0, 1, 2]
    assert synth_client.peak_in_flight <= 2
    assert synth_client.peak_in_flight > 1


def test_parallel_negative_outputs_match_serial_for_deterministic_fixture(
    stub_env,
    tmp_path,
):
    """max_concurrent=1 and >1 produce identical Stage 1c rows."""
    prefixes = [
        _prefix_row(concept_id=0, prefix_text="p0"),
        _prefix_row(concept_id=1, prefix_text="p1"),
    ]
    concepts = [_concept(0, 10), _concept(1, 20)]

    serial_cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    parallel_cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )

    stub_env["negative_prefixes"] = list(prefixes)
    stub_env["synth_client"] = _StubSynthClient(responses=["same"] * 10)
    stub_env["fc_active_sequence"] = [False]
    stub_env["fc_active_idx"] = 0
    serial_out = synthesize_negative_mode(
        serial_cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    serial_rows = pq.read_table(serial_out).to_pylist()

    stub_env["negative_prefixes"] = list(prefixes)
    stub_env["synth_client"] = _StubSynthClient(responses=["same"] * 10)
    stub_env["fc_active_sequence"] = [False]
    stub_env["fc_active_idx"] = 0
    parallel_out = synthesize_negative_mode(
        parallel_cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    parallel_rows = pq.read_table(parallel_out).to_pylist()

    def _normalize_nan(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: ("NaN" if isinstance(value, float) and value != value else value)
            for key, value in row.items()
        }

    assert [_normalize_nan(row) for row in parallel_rows] == [
        _normalize_nan(row) for row in serial_rows
    ]


def test_stage1c_global_pool_is_byte_exact_under_reverse_completion(
    stub_env,
    tmp_path,
):
    """Completion order cannot perturb the Stage 1c corpus or yield report."""
    prefixes = [
        _prefix_row(concept_id=0, prefix_text="prefix-0"),
        _prefix_row(concept_id=1, prefix_text="prefix-1"),
        _prefix_row(concept_id=2, prefix_text="prefix-2"),
    ]
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]
    serial_cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    parallel_cfg = serial_cfg.replace(
        synth=SynthConfig(requests_per_minute=0, max_concurrent=3),
    )

    stub_env["negative_prefixes"] = list(prefixes)
    stub_env["fc_active_sequence"] = [AvoidContinuationOutcome.INACTIVE]
    stub_env["fc_active_idx"] = 0
    stub_env["synth_client"] = _ShuffledPrefixSynthClient(
        {prefix["prefix_text"]: 0.0 for prefix in prefixes}
    )
    serial_out = synthesize_negative_mode(
        serial_cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    serial_bytes = serial_out.read_bytes()
    serial_yield_bytes = (
        paths.AxBenchBasePaths.from_config(serial_cfg).stage1c_concept_yield_report().read_bytes()
    )

    parallel_client = _ShuffledPrefixSynthClient(
        {
            "prefix-0": 0.06,
            "prefix-1": 0.03,
            "prefix-2": 0.0,
        }
    )
    stub_env["negative_prefixes"] = list(prefixes)
    stub_env["fc_active_sequence"] = [AvoidContinuationOutcome.INACTIVE]
    stub_env["fc_active_idx"] = 0
    stub_env["synth_client"] = parallel_client
    parallel_out = synthesize_negative_mode(
        parallel_cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    assert parallel_client.peak_in_flight == 3
    assert parallel_client.completion_order == [
        "prefix-2",
        "prefix-1",
        "prefix-0",
    ]
    assert parallel_out.read_bytes() == serial_bytes
    assert (
        paths.AxBenchBasePaths.from_config(parallel_cfg)
        .stage1c_concept_yield_report()
        .read_bytes()
        == serial_yield_bytes
    )


def test_synthesize_negative_mode_releases_cuda_after_each_serial_concept(
    stub_env,
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    monkeypatch.setattr(
        negative_mode,
        "release_cuda_iteration_memory",
        lambda: calls.append("release"),
    )
    cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=0, prefix_text="p0"),
        _prefix_row(concept_id=1, prefix_text="p1"),
    ]
    stub_env["fc_active_sequence"] = [False]

    out = synthesize_negative_mode(
        cfg,
        [_concept(0, 10), _concept(1, 20)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    rows = pq.read_table(out).to_pylist()
    avoiding = [row for row in rows if row["content_class"] == "concept_avoiding"]
    assert [row["concept_id"] for row in avoiding] == [0, 1]
    assert calls == ["release", "release"]


def test_synthesize_negative_mode_releases_cuda_after_parallel_chain_pool(
    stub_env,
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    pool_calls = 0
    monkeypatch.setattr(
        negative_mode,
        "release_cuda_iteration_memory",
        lambda: calls.append("release"),
    )

    def _complete_chains(
        llm_client,
        initial_requests,
        *,
        handle_result=None,
        handle_results=None,
        **kwargs,
    ):
        del llm_client, kwargs
        nonlocal pool_calls
        pool_calls += 1
        for initial_request in initial_requests:
            request = initial_request
            while request is not None:
                result = negative_mode.SynthResult(
                    index=request.index,
                    response_text=frame_stage1_response(
                        request.prompt,
                        f"chain-{request.index}-attempt-{request.attrs['attempt']}",
                    ),
                )
                if handle_result is not None:
                    request = handle_result(request, result)
                else:
                    assert handle_results is not None
                    request = handle_results([(request, result)])[0]

    monkeypatch.setattr(
        negative_mode,
        "complete_synth_request_chains",
        _complete_chains,
    )
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=0, prefix_text="p0"),
        _prefix_row(concept_id=1, prefix_text="p1"),
    ]
    # Chain p0 retries once; p1 succeeds on its first request.
    stub_env["fc_active_sequence"] = [
        negative_mode.AvoidContinuationOutcome.ACTIVE,
        negative_mode.AvoidContinuationOutcome.INACTIVE,
        negative_mode.AvoidContinuationOutcome.INACTIVE,
    ]

    out = synthesize_negative_mode(
        cfg,
        [_concept(0, 10), _concept(1, 20)],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    rows = pq.read_table(out).to_pylist()
    avoiding = [row for row in rows if row["content_class"] == "concept_avoiding"]
    assert [row["concept_id"] for row in avoiding] == [0, 1]
    assert pool_calls == 1
    assert calls == ["release"]


def test_negative_serial_parallel_byte_identical_multiprefix(
    stub_env,
    tmp_path,
):
    """serial (max_concurrent=1) and parallel (max_concurrent=4) produce byte-identical
    Stage 1c rows when a concept has >=2 prefixes and all pass QC (no drop).

    Teeth: if parallel changes row ordering or content, the tuple comparison will fail.
    """
    # Each concept has 2 prefixes and all continuations are accepted (fc=INACTIVE/False).
    prefixes = [
        _prefix_row(concept_id=0, prefix_text="c0p0"),
        _prefix_row(concept_id=0, prefix_text="c0p1"),
        _prefix_row(concept_id=1, prefix_text="c1p0"),
        _prefix_row(concept_id=1, prefix_text="c1p1"),
    ]
    concepts = [_concept(0, 10), _concept(1, 20)]

    serial_cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    parallel_cfg = _cfg(
        tmp_path,
        max_retries=0,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )

    stub_env["negative_prefixes"] = list(prefixes)
    stub_env["synth_client"] = _StubSynthClient(responses=["same_output"] * 20)
    stub_env["fc_active_sequence"] = [False]  # all accepted — no drops
    stub_env["fc_active_idx"] = 0
    serial_out = synthesize_negative_mode(
        serial_cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    serial_rows = pq.read_table(serial_out).to_pylist()

    stub_env["negative_prefixes"] = list(prefixes)
    stub_env["synth_client"] = _StubSynthClient(responses=["same_output"] * 20)
    stub_env["fc_active_sequence"] = [False]
    stub_env["fc_active_idx"] = 0
    parallel_out = synthesize_negative_mode(
        parallel_cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    parallel_rows = pq.read_table(parallel_out).to_pylist()

    def _key(row: dict) -> tuple:
        return (row["concept_id"], row["content_class"], row["output_text"])

    serial_keys = [_key(r) for r in serial_rows]
    parallel_keys = [_key(r) for r in parallel_rows]
    assert parallel_keys == serial_keys, (
        f"serial vs parallel mismatch: serial={serial_keys!r} parallel={parallel_keys!r}"
    )


def test_parallel_negative_retries_only_pending_prefixes(
    stub_env,
    tmp_path,
    monkeypatch,
):
    """Accepted prefixes terminate their chain without speculative retries."""
    cfg = _cfg(
        tmp_path,
        max_retries=1,
        extraction_tokenizer_id="custom/stage1c-parallel-tokenizer",
        synth=SynthConfig(requests_per_minute=0, max_concurrent=3),
    )
    prefixes = [
        _prefix_row(concept_id=0, prefix_text="p0"),
        _prefix_row(concept_id=1, prefix_text="p1"),
        _prefix_row(concept_id=2, prefix_text="p2"),
    ]
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]
    stub_env["negative_prefixes"] = list(prefixes)

    prompt_for_prefix = {
        row["prefix_text"]: (f"neg_template {row['prefix_text']} :: concept {row['concept_id']}")
        for row in prefixes
    }
    stub_env["synth_client"] = _ScriptedConcurrentSynthClient(
        {
            prompt_for_prefix["p0"]: ["p0 attempt-0"],
            prompt_for_prefix["p1"]: ["p1 attempt-0", "p1 attempt-1"],
            prompt_for_prefix["p2"]: [
                AxBenchBaseError("temporary outage"),
                "p2 attempt-1",
            ],
        },
        delay_s=0.03,
    )

    qc_calls: dict[str, int] = {}
    qc_tokenizer_ids: list[str] = []

    def _fc_active(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold=0.0,
        tokenizer_id,
    ):
        del continuation_text, f_c, base_model, base_sae, sae_layer, extraction_threshold
        qc_tokenizer_ids.append(tokenizer_id)
        attempt = qc_calls.get(prefix_text, 0)
        qc_calls[prefix_text] = attempt + 1
        active_map = AvoidContinuationOutcome.ACTIVE
        inactive_map = AvoidContinuationOutcome.INACTIVE
        if prefix_text == "p0":
            return inactive_map
        if prefix_text == "p1":
            return active_map if attempt == 0 else inactive_map
        if prefix_text == "p2":
            return inactive_map
        raise AssertionError(f"unexpected prefix_text={prefix_text!r}")

    monkeypatch.setattr(negative_mode, "classify_avoid_continuation", _fc_active)
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuations",
        lambda inputs, base_model, base_sae, *, sae_layer,
        extraction_threshold=0.0, tokenizer_id,
        profile_region_name="synth_qc.classify_avoid_continuations": [
            _fc_active(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                base_model,
                base_sae,
                sae_layer=sae_layer,
                extraction_threshold=extraction_threshold,
                tokenizer_id=tokenizer_id,
            )
            for item in inputs
        ],
    )

    out = synthesize_negative_mode(
        cfg,
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    rows = pq.read_table(out).to_pylist()
    avoiding = [row for row in rows if row["content_class"] == "concept_avoiding"]
    assert [row["concept_id"] for row in avoiding] == [0, 1, 2]
    assert qc_tokenizer_ids
    assert set(qc_tokenizer_ids) == {cfg.extraction_tokenizer_id}
    assert stub_env["synth_client"].peak_in_flight <= 3
    assert stub_env["synth_client"].peak_in_flight > 1
    assert len(stub_env["synth_client"].calls) == 5
    calls_by_prefix = {
        prefix_text: [
            call
            for call in stub_env["synth_client"].calls
            if call["prompt"].startswith(f"PREFIX={prefix_text}\n")
        ]
        for prefix_text in prompt_for_prefix
    }
    assert len(calls_by_prefix["p0"]) == 1
    assert len(calls_by_prefix["p1"]) == 2
    assert len(calls_by_prefix["p2"]) == 2


def test_parallel_negative_scores_one_ready_batch(
    stub_env,
    tmp_path,
    monkeypatch,
) -> None:
    """Stage 1c scatters mixed strict-zero outcomes from one GPU batch."""
    qc_batches: list[list[tuple[str, int]]] = []

    def _classify_batch(inputs, *_args, **_kwargs):
        qc_batches.append(
            [(item.prefix_text, item.feature_id) for item in inputs]
        )
        return [
            AvoidContinuationOutcome.INACTIVE,
            AvoidContinuationOutcome.ACTIVE,
            AvoidContinuationOutcome.NO_TOKENS,
        ]

    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuations",
        _classify_batch,
    )
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scalar strict-zero QC must not run")
        ),
    )

    def _complete(
        llm_client,
        requests,
        *,
        handle_results,
        result_batch_size,
        **_kwargs,
    ) -> None:
        assert result_batch_size == 16
        pairs = []
        for request in requests:
            response = llm_client.complete(
                request.prompt,
                cache_key=request.cache_key,
            )
            pairs.append(
                (
                    request,
                    negative_mode.SynthResult(
                        index=request.index,
                        response_text=response,
                    ),
                )
            )
        assert handle_results(pairs) == [None, None, None]

    monkeypatch.setattr(
        negative_mode,
        "complete_synth_request_chains",
        _complete,
    )
    concepts = [_concept(0, 10), _concept(1, 20), _concept(2, 30)]
    stub_env["negative_prefixes"] = [
        _prefix_row(concept_id=0, prefix_text="p0"),
        _prefix_row(concept_id=1, prefix_text="p1"),
        _prefix_row(concept_id=2, prefix_text="p2"),
    ]
    out = synthesize_negative_mode(
        _cfg(
            tmp_path,
            max_retries=0,
            qc_batch_size=16,
            qc_fallback_on_exhaust="warn",
            max_dropped_concept_fraction=0.75,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=3),
        ),
        concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )

    assert qc_batches == [[("p0", 10), ("p1", 20), ("p2", 30)]]
    rows = pq.read_table(out).to_pylist()
    assert {row["concept_id"] for row in rows} == {0}
    assert [row["content_class"] for row in rows][-1] == "concept_avoiding"


def test_synthesize_concept_negatives_empty_list_returns_empty(stub_env, tmp_path):
    """The per-concept helper short-circuits on empty prefixes."""
    cfg = _cfg(tmp_path)
    result = _synthesize_concept_negatives(
        cfg,
        _concept(),
        concept_prefixes=[],
        base_model=object(),
        base_sae=object(),
        llm_client=stub_env["synth_client"],
        target_model_meta=_stub_meta(),
    )
    assert result == []


def test_synthesize_negative_mode_opens_cache_and_passes_cache_key(
    stub_env,
    tmp_path,
):
    """Stage 1c must open the SynthResponseCache and forward cache_keys."""
    cfg = _cfg(tmp_path)
    stub_env["negative_prefixes"] = [_prefix_row()]
    stub_env["fc_active_sequence"] = [False]
    synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    calls = stub_env["synth_client"].calls
    assert len(calls) >= 1
    for call in calls:
        assert call["cache_key"] is not None
        assert len(call["cache_key"]) == 64  # sha256 hex
    # The production cache file exists after the stage.
    cache_file = paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()
    assert cache_file.exists()


def test_synthesize_negative_mode_cache_key_includes_attempt(stub_env, tmp_path):
    """concept_avoiding retries must get distinct cache_keys."""
    cfg = _cfg(tmp_path, max_retries=2)
    stub_env["negative_prefixes"] = [_prefix_row()]
    # First attempt: fc_active=True (fail), second: fc_active=False (accept)
    stub_env["fc_active_sequence"] = [True, False]
    synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_meta(),
    )
    calls = stub_env["synth_client"].calls
    assert len(calls) == 2
    assert calls[0]["cache_key"] != calls[1]["cache_key"]
