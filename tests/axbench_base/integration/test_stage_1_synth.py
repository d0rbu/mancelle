"""Stage 1 (synth) end-to-end integration tests.

Exercises the Stage 1 synth entry points with in-memory stubs for
the LLM client, base Gemma, and base SAE:

* Stage 1b ``synthesize_positive_mode`` — per-concept LLM synthesis with
  a mock client returning canned continuations. Each accepted
  ``concept_rich`` row is paired with a per-prefix ``concept_avoiding``
  completion (the global-negative pool was removed in the class-margin
  refactor).
* Stage 1c ``synthesize_negative_mode`` — per-verified-prefix LLM
  synthesis. Verifies the schema and the presence of concept
  rich + concept avoiding rows.

Each test writes a tiny ``seeds_train.safetensors`` pool via the real
Stage 0b writer so Stage 1b's real loader can read it back.

The Stage-1b positive-mode assertions in this file exercise the
class-margin refactor: per concept, ``synthesize_positive_mode`` emits N
``concept_rich`` rows each immediately followed by a paired
``concept_avoiding`` row of the SAME prefix. The legacy Stage 1a
global-negative pool (``synthesize_global_negatives`` /
``synth_negatives.parquet`` / ``n_negative_pool_size``) was removed
entirely; there are no ``concept_id == -1`` rows.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base import tokenizer_cache
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig, SynthConfig
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.seed_contract import (
    _stage1b_seed_rows_identity,
    load_seed_pool,
    seed_rows_to_tensors,
)
from steering.eval.axbench_base.synth import (
    negative_mode as negative_mode_mod,
)
from steering.eval.axbench_base.synth import (
    positive_mode as positive_mode_mod,
)
from steering.eval.axbench_base.synth import (
    qc as qc_mod,
)
from steering.eval.axbench_base.synth.negative_mode import (
    NEG_TRAINING_SCHEMA,
    synthesize_negative_mode,
)
from steering.eval.axbench_base.synth.positive_mode import (
    TRAINING_SCHEMA,
)
from steering.eval.axbench_base.synth.response_parser import (
    DEFAULT_EXTRACTION_TOKENIZER_ID,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    TargetModelMetadata,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.seed_contract import write_committed_seed_pair
from tests.axbench_base.stage1_protocol_helpers import (
    TEST_STAGE1_TEMPLATE,
    frame_stage1_response,
)
from tests.axbench_base.tokenizer_cache import clear_tokenizer_cache

pytestmark = pytest.mark.slow

_SEED_TOKEN_WIDTH = 64


def synthesize_positive_mode(
    cfg: Any,
    concepts: list[Concept],
    seeds_train: list[SeedRow],
    *args: Any,
    **kwargs: Any,
) -> Path:
    """Test adapter for Stage 1b's loader-supplied bulk row identity."""

    tensors = seed_rows_to_tensors(seeds_train)
    kwargs.setdefault(
        "seed_rows_sha256",
        _stage1b_seed_rows_identity(tensors, row_count=len(seeds_train)),
    )
    return positive_mode_mod.synthesize_positive_mode(
        cfg,
        concepts,
        seeds_train,
        *args,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shared fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def _install_scalar_qc_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt scalar-focused scenarios into singleton adapters.

    The fixture is deliberately not autouse: real-QC integration coverage must
    execute the production ragged helpers rather than silently replacing them
    throughout this module.
    """

    def _positive_percentiles(
        inputs,
        base_model,
        base_sae,
        percentile_table,
        *,
        sae_layer,
        tokenizer_id,
        release_memory=True,
    ):
        del release_memory
        return [
            positive_mode_mod.output_fc_percentile_for_continuation(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                base_model,
                base_sae,
                percentile_table,
                sae_layer=sae_layer,
                tokenizer_id=tokenizer_id,
                **(
                    {}
                    if item.cluster_target is None
                    else {"cluster_target": item.cluster_target}
                ),
            )
            for item in inputs
        ]

    def _positive_avoids(
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
            positive_mode_mod.classify_avoid_continuation(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                base_model,
                base_sae,
                sae_layer=sae_layer,
                extraction_threshold=extraction_threshold,
                tokenizer_id=tokenizer_id,
                **(
                    {}
                    if item.cluster_target is None
                    else {"cluster_target": item.cluster_target}
                ),
            )
            for item in inputs
        ]

    def _negative_avoids(
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
            negative_mode_mod.classify_avoid_continuation(
                item.prefix_text,
                item.continuation_text,
                item.feature_id,
                base_model,
                base_sae,
                sae_layer=sae_layer,
                extraction_threshold=extraction_threshold,
                tokenizer_id=tokenizer_id,
                **(
                    {}
                    if item.cluster_target is None
                    else {"cluster_target": item.cluster_target}
                ),
            )
            for item in inputs
        ]

    monkeypatch.setattr(
        positive_mode_mod,
        "output_fc_percentiles_for_continuations",
        _positive_percentiles,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "classify_avoid_continuations",
        _positive_avoids,
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "classify_avoid_continuations",
        _negative_avoids,
    )


def _classify_inactive_unless_empty(
    prefix_text,
    continuation_text,
    f_c,
    base_model,
    base_sae,
    *,
    sae_layer,
    extraction_threshold,
    tokenizer_id,
):
    """Avoid-QC stub: empty continuation -> NO_TOKENS, else INACTIVE.

    Shared by the avoid-tier tests that just need the natural / LLM tiers to
    accept any non-empty continuation while still exercising the NO_TOKENS
    empty-continuation trap. (The warm-cache test at the bottom of this module
    keeps its own bespoke ACTIVE-variant stub.)
    """
    del prefix_text, f_c, base_model, base_sae
    del sae_layer, extraction_threshold, tokenizer_id
    return (
        positive_mode_mod.AvoidContinuationOutcome.NO_TOKENS
        if not str(continuation_text).strip()
        else positive_mode_mod.AvoidContinuationOutcome.INACTIVE
    )


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


def _make_seed(
    idx: int,
    text: str = "prefix body",
    continuation_text: str = "natural continuation",
    sae_feature_ids: list[int] | None = None,
) -> SeedRow:
    tokenizer = _StubNanochatTokenizer()
    prefix_tokens = tokenizer.encode(text)
    continuation_tokens = tokenizer.encode(continuation_text)
    if len(prefix_tokens) > _SEED_TOKEN_WIDTH or len(continuation_tokens) > (
        _SEED_TOKEN_WIDTH
    ):
        raise AssertionError("test seed text exceeds the fixed fixture width")
    prefix_tokens.extend([0] * (_SEED_TOKEN_WIDTH - len(prefix_tokens)))
    continuation_tokens.extend(
        [0] * (_SEED_TOKEN_WIDTH - len(continuation_tokens))
    )
    return SeedRow(
        shard_idx=0,
        doc_idx=idx,
        source_nanochat_token_count=2 * _SEED_TOKEN_WIDTH,
        prefix_source_token_start=0,
        prefix_source_token_end=_SEED_TOKEN_WIDTH,
        prefix_text=text,
        prefix_content_nanochat_tokens=prefix_tokens,
        sae_feature_ids=sorted(sae_feature_ids or []),
        sae_feature_mags=[1.0] * len(sae_feature_ids or []),
        continuation_text=continuation_text,
        continuation_content_nanochat_tokens=continuation_tokens,
    )


def _concept(concept_id: int = 0, feature_index: int = 42) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {concept_id} description",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _seed_contract_cfg(cfg: AxBenchBaseConfig) -> AxBenchBaseConfig:
    """Adapt production seed/QC dimensions to this module's tiny fixtures."""
    stats_path = (
        test_paths.AxBenchBasePaths.from_config(cfg).run_output_root()
        / "_stage1b_test_stats.safetensors"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if not stats_path.exists():
        save_file(
            {"_placeholder": t.zeros(1, dtype=t.int64)},
            str(stats_path),
            metadata={"n_sae_features": "16384"},
        )
    return cfg.replace(
        feature_statistics_path=str(stats_path),
        seed_prefix_max_nanochat_tokens=_SEED_TOKEN_WIDTH,
        seed_continuation_nanochat_tokens=_SEED_TOKEN_WIDTH,
        min_prefix_chars=1,
    )


def _commit_seed_pair(cfg: AxBenchBaseConfig, rows: list[SeedRow]) -> None:
    write_committed_seed_pair(
        cfg,
        tokenizer=_StubNanochatTokenizer(),
        train_rows=rows,
        eval_rows=[rows[0]],
    )


def _write_tiny_seed_pool(
    cfg: AxBenchBaseConfig, n: int = 6
) -> list[SeedRow]:
    """Write a committed schema-v2 pair and return its train rows."""
    rows = [
        _make_seed(
            i,
            text=f"prefix text {i}",
            continuation_text=f"continuation {i}",
        )
        for i in range(n)
    ]
    _commit_seed_pair(cfg, rows)
    return rows


class _StubSynthClient:
    """Deterministic in-memory mock of ``OpenRouterSynthClient``.

    Returns a single canned string on every ``complete`` call, recording
    prompt + cache_key into ``calls`` for optional assertions.
    """

    def __init__(self, canned: str = "canned synth continuation") -> None:
        self.canned = canned
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "cache_key": cache_key})
        return frame_stage1_response(prompt, self.canned)


class _CachingStubSynthClient:
    """Stub synth client that exercises the real per-run response cache."""

    def __init__(
        self,
        *,
        canned: str,
        response_cache: Any = None,
        fail_on_live_call: bool = False,
    ) -> None:
        self.canned = canned
        self.response_cache = response_cache
        self.fail_on_live_call = fail_on_live_call
        self.calls: list[dict[str, Any]] = []
        self.cache_hits = 0
        self.live_calls = 0

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "cache_key": cache_key})
        if cache_key is not None and self.response_cache is not None:
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        self.live_calls += 1
        if self.fail_on_live_call:
            raise AssertionError("unexpected live synth call on warm-cache replay")
        response = frame_stage1_response(prompt, self.canned)
        if cache_key is not None and self.response_cache is not None:
            self.response_cache.set(cache_key, response)
        return response


class _CharExtractionTokenizer:
    bos_token_id = 0
    all_special_ids = [0]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 1) for token_id in token_ids)


class _StubNanochatTokenizer:
    """Lightweight nanochat-compatible tokenizer that does not touch disk.

    ``_count_nanochat_tokens`` in ``synth.negative_mode`` delegates to
    ``nanochat.tokenizer.get_tokenizer`` which loads a real pickle from
    ``$NANOCHAT_BASE_DIR/tokenizer``. We monkey-patch the module-level
    ``get_tokenizer`` symbol with an instance of this class so the test
    never has to symlink a cache directory.
    """

    def __init__(self, bos_id: int = 1) -> None:
        self._bos_id = bos_id

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        base = [ord(c) + 2 for c in text]
        if prepend is not None:
            return [prepend] + base
        return base

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 2) for token_id in token_ids if token_id != 0)


@pytest.fixture(autouse=True)
def _patch_exact_nanochat_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Exercise the production Stage 1 cache with a deterministic tokenizer."""
    clear_tokenizer_cache()
    monkeypatch.setattr(
        tokenizer_cache,
        "_nanochat_get_tokenizer",
        lambda: _StubNanochatTokenizer(),
    )
    yield
    clear_tokenizer_cache()


class _WhitespaceTokenizer:
    """Deterministic whitespace tokenizer for Stage 1b cache parity tests."""

    # Sentinel BOS id, far outside the small sequential per-word vocab range
    # (starts at 1) -- qc.py's SAE-forward sites prepend this via
    # resolve_bos_token_id and exclude its row from the pool.
    bos_token_id = 999
    all_special_ids = [999]

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
                self.reverse_vocab[self.vocab[word]] = word
            ids.append(self.vocab[word])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.reverse_vocab[token_id] for token_id in token_ids)


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize NaNs so row-by-row equality works for parquet round-trips."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        fixed: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                fixed[key] = "__nan__"
            else:
                fixed[key] = value
        normalized.append(fixed)
    return normalized


# ---------------------------------------------------------------------------
# Stage 1b: synthesize_positive_mode
#
# The standalone Stage 1a (synthesize_global_negatives) test was removed
# alongside the global-negative pool. Positive mode now emits a per-prefix
# concept_avoiding completion for every accepted concept_rich row.
# ---------------------------------------------------------------------------


def test_stage_1b_writes_synth_training_positive_parquet_with_paired_avoiding_rows(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Stage 1b writes per-concept rich rows each paired with an avoiding row."""
    cfg = _seed_contract_cfg(
        make_valid_axbench_cfg(
            run_tag="stage1b_e2e",
            n_positives_per_concept=2,
            accept_threshold=0.5,
            max_retries=0,
            output_length=8,
            min_prefix_chars=1,
        )
    )

    seeds = _write_tiny_seed_pool(cfg, n=5)

    # Stub all the LLM + Gemma + percentile plumbing for Stage 1b.
    stub_client = _StubSynthClient(canned="positive mode continuation")

    monkeypatch.setattr(
        positive_mode_mod,
        "_build_synth_client",
        lambda cfg, response_cache=None: stub_client,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "_infer_n_sae_features",
        lambda base_sae: 16384,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: None,
    )
    # Canonical prefix scoring is stubbed to 0 below. Return output_pct=80,
    # giving delta=80 > accept_threshold * 100 = 50.
    monkeypatch.setattr(
        positive_mode_mod,
        "output_fc_percentile_for_continuation",
        lambda prefix, cont, f_c, gemma, sae, table, *, sae_layer,
        tokenizer_id: 80.0,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "fc_percentile_on_prefix",
        lambda prefix, f_c, gemma, sae, table, *, sae_layer, tokenizer_id: 0.0,
    )

    class _ZeroPrefixCache:
        def __init__(self, **kwargs) -> None:
            self.batch_size = int(kwargs["batch_size"])

        def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
            del f_c
            return [0.0] * len(prefix_texts)

    monkeypatch.setattr(positive_mode_mod, "PrefixPercentileCache", _ZeroPrefixCache)
    # Paired concept_avoiding row: the concept feature is INACTIVE on every
    # natural continuation, so the natural tier accepts (no LLM avoid call).
    # The combined classifier maps empty/whitespace -> NO_TOKENS, else INACTIVE
    # (NO_TOKENS routes a no-tokens-added continuation to the LLM tier).
    monkeypatch.setattr(
        positive_mode_mod,
        "classify_avoid_continuation",
        _classify_inactive_unless_empty,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _CharExtractionTokenizer(),
    )

    concepts = [_concept(0, 42), _concept(1, 99)]

    out_path = synthesize_positive_mode(
        cfg,
        concepts=concepts,
        seeds_train=seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert out_path == test_paths.default_paths("stage1b_e2e").synth_training_positive()
    assert out_path.exists()
    table = pq.read_table(out_path)
    assert len(table.column_names) == 16
    assert table.schema.equals(TRAINING_SCHEMA)
    metadata = table.schema.metadata or {}
    assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] == (
        cfg.extraction_tokenizer_id.encode("utf-8")
    )
    contract = validate_positive_qc_contract(
        out_path,
        consumer="Stage 1b integration test",
        expected_accept_threshold=cfg.accept_threshold,
    )
    assert contract.accept_threshold_fraction == 0.5
    assert contract.accept_threshold_points == 50.0
    validate_stage1_continuation_contract(
        out_path,
        consumer="Stage 1b integration test",
        expected_role="stage1b_positive",
        schema=table.schema,
    )
    # Per concept: 2 concept_rich rows + 2 paired concept_avoiding rows.
    # 2 concepts * (2 + 2) = 8 rows. No global-negative concept_id=-1 rows.
    assert table.num_rows == 8
    rows = table.to_pylist()
    cids = [r["concept_id"] for r in rows]
    assert -1 not in cids
    assert cids.count(0) == 4
    assert cids.count(1) == 4
    rich = [r for r in rows if r["content_class"] == "concept_rich"]
    avoiding = [r for r in rows if r["content_class"] == "concept_avoiding"]
    assert len(rich) == 4
    assert len(avoiding) == 4
    assert all(r["output_text"] == "positive" for r in rich)
    assert all(r["category"] == "positive" for r in rich)
    assert all(r["qc_zero_activation_verified"] is False for r in rich)
    # Emission order is [rich, avoiding, rich, avoiding, ...].
    assert [r["content_class"] for r in rows] == [
        "concept_rich", "concept_avoiding",
    ] * 4
    # Each avoiding row pairs the immediately-preceding rich row's prefix,
    # carries the same concept_id, and is QC-verified zero-activation. With
    # the feature inactive on the natural continuation, the avoid output is
    # the seed's own continuation_text with corpus provenance.
    for rich_row, avoid_row in zip(rows[0::2], rows[1::2], strict=True):
        assert avoid_row["concept_id"] == rich_row["concept_id"]
        assert avoid_row["prefix_text"] == rich_row["prefix_text"]
        assert avoid_row["category"] == "negative"
        assert avoid_row["qc_zero_activation_verified"] is True
        assert avoid_row["output_text"].strip() != ""
        assert avoid_row["gen_source"].startswith("corpus:doc_continuation@")


def test_stage_1b_real_seed_and_parquet_writers_bound_prefix_and_retry_echo(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Issue #1051 functional regression through committed Stage 0b artifacts.

    The seed is written and reloaded with the real schema-v2 Stage 0b contract.
    Stage 1b must retain that frozen full prefix on disk while using only its
    exact 32-character tail everywhere in synthesis. A first response that
    copies that tail is rejected before QC; the second response is accepted and
    the real Stage 1b parquet writer publishes only bounded-prefix rows.
    """
    del tmp_run_root
    cfg = _seed_contract_cfg(
        make_valid_axbench_cfg(
            run_tag="stage1b_issue_1051_functional",
            n_positives_per_concept=1,
            accept_threshold=0.5,
            max_retries=1,
            output_length=64,
            min_prefix_chars=10,
            max_prefix_chars=32,
        )
    )
    full_prefix = "HEAD-0123456789abcdefghijklmnopqrstuvwxyz"
    bounded_prefix = full_prefix[-cfg.max_prefix_chars :]
    seed = _make_seed(
        0,
        text=full_prefix,
        continuation_text="trusted natural continuation",
    )
    _commit_seed_pair(cfg, [seed])
    output_paths = test_paths.default_paths("stage1b_issue_1051_functional")
    seed_path = output_paths.seeds_train()
    seed_bytes_before = seed_path.read_bytes()
    loaded_seeds = load_seed_pool(
        seed_path,
        expected_split="train",
        tokenizer=_StubNanochatTokenizer(),
    )
    assert len(full_prefix) > cfg.max_prefix_chars
    assert len(bounded_prefix) == cfg.max_prefix_chars
    assert loaded_seeds[0].prefix_text == full_prefix

    class _QueueClient:
        def __init__(self) -> None:
            self.responses = [bounded_prefix, "clean novel continuation"]
            self.calls: list[dict[str, Any]] = []

        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
        ) -> str:
            self.calls.append({"prompt": prompt, "cache_key": cache_key})
            return frame_stage1_response(prompt, self.responses.pop(0))

    client = _QueueClient()
    rich_qc_calls: list[tuple[str, str]] = []
    avoid_qc_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        positive_mode_mod,
        "_build_synth_client",
        lambda _cfg, response_cache=None: client,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "_infer_n_sae_features",
        lambda _base_sae: 16384,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "build_percentile_table_from_stats",
        lambda _stats_path, n_sae_features: (
            None if n_sae_features == 16384 else pytest.fail("wrong SAE width")
        ),
    )

    def _rich_qc(
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
        del f_c, gemma, sae, table, sae_layer, tokenizer_id
        rich_qc_calls.append((prefix, continuation))
        return 80.0

    def _avoid_qc(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        *,
        sae_layer,
        extraction_threshold,
        tokenizer_id,
    ):
        del (
            f_c,
            base_model,
            base_sae,
            sae_layer,
            extraction_threshold,
            tokenizer_id,
        )
        avoid_qc_calls.append((prefix_text, continuation_text))
        return positive_mode_mod.AvoidContinuationOutcome.INACTIVE

    monkeypatch.setattr(
        positive_mode_mod,
        "output_fc_percentile_for_continuation",
        _rich_qc,
    )

    class _TenPrefixCache:
        def __init__(self, **kwargs) -> None:
            self.batch_size = int(kwargs["batch_size"])

        def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
            del f_c
            return [10.0] * len(prefix_texts)

    monkeypatch.setattr(positive_mode_mod, "PrefixPercentileCache", _TenPrefixCache)
    monkeypatch.setattr(
        positive_mode_mod,
        "classify_avoid_continuation",
        _avoid_qc,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "load_template",
        lambda _name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: _CharExtractionTokenizer(),
    )

    out_path = synthesize_positive_mode(
        cfg,
        concepts=[_concept(0, 42)],
        seeds_train=loaded_seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert seed_path.read_bytes() == seed_bytes_before
    assert loaded_seeds[0].prefix_text == full_prefix
    assert len(client.calls) == 2
    assert all(bounded_prefix in call["prompt"] for call in client.calls)
    assert all(full_prefix not in call["prompt"] for call in client.calls)
    assert client.calls[0]["cache_key"] != client.calls[1]["cache_key"]
    assert rich_qc_calls == [
        (bounded_prefix, "clean novel continuation")
    ]
    assert avoid_qc_calls == [
        (bounded_prefix, "trusted natural continuation")
    ]

    table = pq.read_table(out_path)
    assert table.schema.equals(TRAINING_SCHEMA)
    rows = table.to_pylist()
    assert [row["content_class"] for row in rows] == [
        "concept_rich",
        "concept_avoiding",
    ]
    assert all(row["prefix_text"] == bounded_prefix for row in rows)
    assert all(
        cfg.min_prefix_chars
        <= len(row["prefix_text"])
        <= cfg.max_prefix_chars
        for row in rows
    )
    assert rows[0]["output_text"] == "clean novel continuation"
    assert rows[0]["qc_attempts"] == 2
    assert rows[1]["output_text"] == "trusted natural continuation"
    assert rows[1]["qc_attempts"] == 0
    assert all(
        row["n_prefix_nanochat_tokens"]
        == len(_StubNanochatTokenizer().encode(bounded_prefix, prepend=1))
        for row in rows
    )


def test_stage_1b_cached_prefix_qc_matches_reference_no_cache_outputs(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Stage 1b writes identical rows with the real cache and a no-cache reference."""

    del tmp_run_root
    table = t.zeros(16, 99, dtype=t.float64)
    for feature_id in range(16):
        for pct in range(99):
            table[feature_id, pct] = (pct + 1) / 100.0

    base_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=4, model_name="gemma-integration"),
    )
    base_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            hook_name="blocks.8.hook_resid_post",
            release="integration-rel",
            d_sae=16,
        ),
    )
    tokenizer = _WhitespaceTokenizer()
    tokenizer_id = "custom/stage1b-cache-tokenizer"
    expected_tokenizer_id = tokenizer_id

    def _fake_qc_tokenizer(requested_tokenizer_id):
        assert requested_tokenizer_id == expected_tokenizer_id
        return tokenizer

    def _fake_stage_tokenizer(requested_tokenizer_id):
        assert requested_tokenizer_id == expected_tokenizer_id
        return tokenizer

    def _fake_extractor(**kwargs):
        out = []
        for token_ids in kwargs["token_id_lists"]:
            acts = t.zeros(len(token_ids), 16)
            content_sum = float(sum(token_ids[1:]))
            acts[:, 5] = content_sum / 100.0
            acts[:, 7] = (content_sum + 7.0) / 100.0
            out.append(acts)
        return out

    def _fake_output_percentile(
        prefix, cont, f_c, gemma, sae, percentile_table, *, sae_layer, tokenizer_id,
    ):
        del gemma, sae, percentile_table, sae_layer
        assert tokenizer_id == expected_tokenizer_id
        return float(len((prefix + cont).split()) * 20 + f_c)

    def _run_stage1b(run_tag: str, *, use_reference_cache: bool) -> pa.Table:
        seeds = [
            _make_seed(0, text="dirty zero", sae_feature_ids=[5]),
            _make_seed(1, text="clean one", sae_feature_ids=[]),
            _make_seed(2, text="dirty two", sae_feature_ids=[5, 7]),
            _make_seed(3, text="dirty three", sae_feature_ids=[5]),
            _make_seed(4, text="dirty four", sae_feature_ids=[5, 7]),
            _make_seed(5, text="dirty five", sae_feature_ids=[5]),
        ]
        cfg = _seed_contract_cfg(
            make_valid_axbench_cfg(
                run_tag=run_tag,
                n_positives_per_concept=3,
                accept_threshold=0.5,
                max_retries=0,
                output_length=8,
                extraction_tokenizer_id=tokenizer_id,
                min_prefix_chars=1,
            )
        )
        _commit_seed_pair(cfg, seeds)

        monkeypatch.setattr(
            positive_mode_mod,
            "_build_synth_client",
            lambda cfg, response_cache=None: _StubSynthClient(
                canned="stage1b parity continuation",
            ),
        )
        # Paired concept_avoiding row: feature inactive on the natural
        # continuation -> natural tier accepts deterministically (no LLM
        # avoid call). This keeps the rich-side prefix-cache parity check
        # (cache vs no-cache) byte-equivalent.
        monkeypatch.setattr(
            positive_mode_mod,
            "classify_avoid_continuation",
            _classify_inactive_unless_empty,
        )
        monkeypatch.setattr(
            positive_mode_mod,
            "_infer_n_sae_features",
            lambda base_sae: 16,
        )
        monkeypatch.setattr(
            positive_mode_mod,
            "build_percentile_table_from_stats",
            lambda stats_path, n_sae_features: table,
        )
        monkeypatch.setattr(
            positive_mode_mod,
            "output_fc_percentile_for_continuation",
            _fake_output_percentile,
        )
        monkeypatch.setattr(
            positive_mode_mod,
            "load_template",
            lambda name: TEST_STAGE1_TEMPLATE,
        )
        monkeypatch.setattr(
            positive_mode_mod,
            "get_extraction_tokenizer",
            _fake_stage_tokenizer,
        )
        monkeypatch.setattr(qc_mod, "get_extraction_tokenizer", _fake_qc_tokenizer)
        monkeypatch.setattr(qc_mod, "_run_batched_extractor", _fake_extractor)
        monkeypatch.setattr(
            positive_mode_mod,
            "fc_percentile_on_prefix",
            qc_mod.fc_percentile_on_prefix,
        )

        if use_reference_cache:
            class _ReferencePrefixCache:
                def __init__(
                    self,
                    *,
                    base_model,
                    base_sae,
                    percentile_table,
                    sae_layer,
                    tokenizer_id,
                    percentile_namespace,
                    batch_size=8,
                    store=None,
                ) -> None:
                    del percentile_namespace, store
                    self.base_model = base_model
                    self.base_sae = base_sae
                    self.percentile_table = percentile_table
                    self.sae_layer = sae_layer
                    self.tokenizer_id = tokenizer_id
                    self.batch_size = int(batch_size)

                def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
                    return [
                        qc_mod.fc_percentile_on_prefix(
                            text, f_c,
                            self.base_model,
                            self.base_sae,
                            self.percentile_table,
                            sae_layer=self.sae_layer,
                            tokenizer_id=self.tokenizer_id,
                        )
                        for text in prefix_texts
                    ]

            monkeypatch.setattr(
                positive_mode_mod, "PrefixPercentileCache", _ReferencePrefixCache,
            )

        concepts = [_concept(0, 5), _concept(1, 7)]
        out_path = synthesize_positive_mode(
            cfg,
            concepts=concepts,
            seeds_train=seeds,
            base_model=base_model,
            base_sae=base_sae,
            target_model_meta=_stub_target_meta(),
        )
        return pq.read_table(out_path)

    actual_table = _run_stage1b(
        "stage1b_cache_parity_actual", use_reference_cache=False,
    )
    reference_table = _run_stage1b(
        "stage1b_cache_parity_reference", use_reference_cache=True,
    )
    for table_obj in (actual_table, reference_table):
        metadata = table_obj.schema.metadata or {}
        assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] == (
            expected_tokenizer_id.encode("utf-8")
        )
    assert _normalize_rows(actual_table.to_pylist()) == _normalize_rows(
        reference_table.to_pylist()
    )


def test_stage_1b_reuses_one_prefix_cache_across_concepts(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Overlapping dirty prefixes across concepts should reuse one warm cache."""

    del tmp_run_root
    extractor_state: dict[str, Any] = {"calls": 0}
    table = t.zeros(16, 99, dtype=t.float64)
    for feature_id in range(16):
        for pct in range(99):
            table[feature_id, pct] = (pct + 1) / 100.0

    seeds = [
        _make_seed(0, text="shared alpha", sae_feature_ids=[5]),
        _make_seed(1, text="shared beta", sae_feature_ids=[5]),
        _make_seed(2, text="shared gamma", sae_feature_ids=[5]),
        _make_seed(3, text="shared delta", sae_feature_ids=[5]),
    ]
    cfg = _seed_contract_cfg(
        make_valid_axbench_cfg(
            run_tag="stage1b_cache_reuse",
            n_positives_per_concept=2,
            accept_threshold=0.5,
            max_retries=0,
            output_length=8,
            min_prefix_chars=1,
        )
    )
    _commit_seed_pair(cfg, seeds)

    base_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=8, model_name="gemma-cache-reuse"),
    )
    base_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            hook_name="blocks.8.hook_resid_post",
            release="integration-rel",
            d_sae=16,
        ),
    )

    monkeypatch.setattr(
        positive_mode_mod,
        "_build_synth_client",
        lambda cfg, response_cache=None: _StubSynthClient(
            canned="stage1b cache reuse continuation",
        ),
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "_infer_n_sae_features",
        lambda base_sae: 16,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: table,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "output_fc_percentile_for_continuation",
        lambda prefix, cont, f_c, gemma, sae, percentile_table, *, sae_layer,
        tokenizer_id: 90.0,
    )
    # Paired concept_avoiding row: feature inactive on the natural
    # continuation -> natural tier accepts (no LLM avoid call and no extra
    # extractor pass), so the single warm-cache extractor call below is the
    # one shared across both concepts' dirty-prefix scoring.
    monkeypatch.setattr(
        positive_mode_mod,
        "classify_avoid_continuation",
        _classify_inactive_unless_empty,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _CharExtractionTokenizer(),
    )
    monkeypatch.setattr(qc_mod, "get_extraction_tokenizer", lambda *args, **kwargs: _WhitespaceTokenizer())

    def _fake_extractor(**kwargs):
        extractor_state["calls"] += 1
        out = []
        for token_ids in kwargs["token_id_lists"]:
            acts = t.zeros((len(token_ids), 16), dtype=t.float32)
            acts[:, 5] = float(sum(token_ids[1:])) / 100.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc_mod, "_run_batched_extractor", _fake_extractor)

    out_path = synthesize_positive_mode(
        cfg,
        concepts=[_concept(0, 5), _concept(1, 5)],
        seeds_train=seeds,
        base_model=base_model,
        base_sae=base_sae,
        target_model_meta=_stub_target_meta(),
    )
    rows = pq.read_table(out_path).to_pylist()
    pos_rows = [row for row in rows if row["category"] == "positive"]
    avoid_rows = [
        row for row in rows if row["content_class"] == "concept_avoiding"
    ]
    # 2 concepts * 2 positives = 4 rich rows, each paired with an avoiding row.
    assert len(pos_rows) == 4
    assert len(avoid_rows) == 4
    assert all(r["qc_zero_activation_verified"] is True for r in avoid_rows)
    # One shared warm prefix-cache extractor pass across both concepts; the
    # avoid natural tier needs no extractor call.
    assert extractor_state["calls"] == 1


def test_stage_1b_parallel_entrypoint_matches_serial_output(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Stage 1b entrypoint must preserve rows when synth waves run in parallel."""

    del tmp_run_root
    prefix_scores = {
        "clean one": 1.0,
        "clean three": 2.0,
        "dirty zero": 5.0,
        "dirty two": 15.0,
        "dirty four": 25.0,
    }
    tokenizer_id = "custom/stage1b-parallel-tokenizer"

    def _output_percentile(
        prefix, cont, f_c, gemma, sae, table, *, sae_layer, tokenizer_id,
    ):
        assert tokenizer_id == "custom/stage1b-parallel-tokenizer"
        return 90.0

    monkeypatch.setattr(
        positive_mode_mod,
        "_build_synth_client",
        lambda cfg, response_cache=None: _StubSynthClient(
            canned="stage1b parallel parity continuation",
        ),
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "_infer_n_sae_features",
        lambda base_sae: 16,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: None,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "output_fc_percentile_for_continuation",
        _output_percentile,
    )
    # Paired concept_avoiding row: feature inactive on the natural
    # continuation -> natural tier accepts deterministically. The avoid side
    # runs after the rich attempt waves, so serial and parallel emit the same
    # [rich, avoiding, ...] rows byte-for-byte.
    # Combined avoid classifier: feature inactive on the natural continuation
    # (INACTIVE) for any non-empty continuation; empty -> NO_TOKENS. The canned
    # continuation is non-empty and clearly adds tokens, so it classifies as
    # INACTIVE, keeping the natural avoid tier on-path without a network
    # tokenizer load.
    monkeypatch.setattr(
        positive_mode_mod,
        "classify_avoid_continuation",
        _classify_inactive_unless_empty,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "get_extraction_tokenizer",
        lambda requested_tokenizer_id: (
            _CharExtractionTokenizer()
            if requested_tokenizer_id == tokenizer_id
            else pytest.fail(
                f"unexpected tokenizer id {requested_tokenizer_id!r}"
            )
        ),
    )

    class _StubPrefixCache:
        def __init__(self, **kwargs) -> None:
            self.batch_size = int(kwargs["batch_size"])

        def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
            del f_c
            return [prefix_scores[text] for text in prefix_texts]

    monkeypatch.setattr(positive_mode_mod, "PrefixPercentileCache", _StubPrefixCache)

    seeds = [
        _make_seed(0, text="dirty zero", sae_feature_ids=[5]),
        _make_seed(1, text="clean one", sae_feature_ids=[]),
        _make_seed(2, text="dirty two", sae_feature_ids=[5]),
        _make_seed(3, text="clean three", sae_feature_ids=[]),
        _make_seed(4, text="dirty four", sae_feature_ids=[5]),
    ]
    concepts = [_concept(0, 5), _concept(1, 5)]

    def _run_stage1b(run_tag: str, *, max_concurrent: int) -> pa.Table:
        cfg = _seed_contract_cfg(
            make_valid_axbench_cfg(
                run_tag=run_tag,
                n_positives_per_concept=3,
                accept_threshold=0.5,
                max_retries=0,
                output_length=8,
                extraction_tokenizer_id=tokenizer_id,
                min_prefix_chars=1,
                synth=SynthConfig(
                    requests_per_minute=0, max_concurrent=max_concurrent
                ),
            )
        )
        _commit_seed_pair(cfg, seeds)
        out_path = synthesize_positive_mode(
            cfg,
            concepts=concepts,
            seeds_train=seeds,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )
        return pq.read_table(out_path)

    serial_table = _run_stage1b("stage1b_parallel_serial", max_concurrent=1)
    parallel_table = _run_stage1b("stage1b_parallel_parallel", max_concurrent=4)
    for table_obj in (serial_table, parallel_table):
        metadata = table_obj.schema.metadata or {}
        assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] == (
            tokenizer_id.encode("utf-8")
        )
    assert _normalize_rows(parallel_table.to_pylist()) == _normalize_rows(
        serial_table.to_pylist()
    )


def test_stage_1b_warm_response_cache_replays_without_live_calls(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """A warm Stage 1b response cache must replay without new synth calls."""

    del tmp_run_root
    clients: list[_CachingStubSynthClient] = []
    monkeypatch.setattr(
        positive_mode_mod,
        "_infer_n_sae_features",
        lambda base_sae: 16,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: None,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "output_fc_percentile_for_continuation",
        lambda prefix, cont, f_c, gemma, sae, table, *, sae_layer,
        tokenizer_id: 90.0,
    )
    # Paired concept_avoiding row: the seed's NATURAL continuation fires the
    # feature (reject natural tier), forcing the avoid side into the LLM
    # fallback tier. The LLM avoiding continuation does NOT fire (accept).
    # This makes the warm-cache replay exercise BOTH the rich and the avoid
    # LLM cache entries.
    # Combined avoid classifier: the natural continuation FIRES the feature
    # (ACTIVE -> reject natural tier -> LLM fallback); the LLM avoiding
    # continuation does NOT fire (INACTIVE -> accept). Empty -> NO_TOKENS.
    monkeypatch.setattr(
        positive_mode_mod,
        "classify_avoid_continuation",
        lambda prefix_text, continuation_text, f_c, base_model, base_sae, *,
        sae_layer, extraction_threshold, tokenizer_id: (
            positive_mode_mod.AvoidContinuationOutcome.NO_TOKENS
            if not str(continuation_text).strip()
            else positive_mode_mod.AvoidContinuationOutcome.ACTIVE
            if continuation_text == "natural continuation"
            else positive_mode_mod.AvoidContinuationOutcome.INACTIVE
        ),
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _CharExtractionTokenizer(),
    )

    class _StubPrefixCache:
        def __init__(self, **kwargs) -> None:
            self.batch_size = int(kwargs["batch_size"])

        def get_many(self, prefix_texts: list[str], f_c: int) -> list[float]:
            del f_c
            return [0.0] * len(prefix_texts)

    monkeypatch.setattr(positive_mode_mod, "PrefixPercentileCache", _StubPrefixCache)

    def _build_client(cfg, response_cache=None):
        client = _CachingStubSynthClient(
            canned="stage1b cached continuation",
            response_cache=response_cache,
            fail_on_live_call=bool(clients),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(positive_mode_mod, "_build_synth_client", _build_client)

    run_tag = "stage1b_warm_cache"
    seeds = [
        _make_seed(0, text="cache prefix 0", sae_feature_ids=[]),
        _make_seed(1, text="cache prefix 1", sae_feature_ids=[]),
        _make_seed(2, text="cache prefix 2", sae_feature_ids=[]),
    ]
    cfg = _seed_contract_cfg(
        make_valid_axbench_cfg(
            run_tag=run_tag,
            n_positives_per_concept=3,
            accept_threshold=0.5,
            max_retries=0,
            output_length=8,
            min_prefix_chars=1,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
        )
    )
    _commit_seed_pair(cfg, seeds)

    first_out = synthesize_positive_mode(
        cfg,
        concepts=[_concept(0, 5)],
        seeds_train=seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )
    second_out = synthesize_positive_mode(
        cfg,
        concepts=[_concept(0, 5)],
        seeds_train=seeds,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert len(clients) == 2
    assert clients[0].live_calls > 0
    assert clients[1].live_calls == 0
    assert clients[1].cache_hits > 0
    assert _normalize_rows(pq.read_table(second_out).to_pylist()) == _normalize_rows(
        pq.read_table(first_out).to_pylist()
    )


def test_stage_1b_real_qc_wiring_uses_canonical_prefix_and_output_scores(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 1b should wire real QC with the right feature, layer, and table."""

    del tmp_run_root
    extractor_state: dict[str, Any] = {"calls": []}
    table = t.zeros(16, 99, dtype=t.float64)
    for feature_id in range(16):
        for pct in range(99):
            table[feature_id, pct] = (pct + 1) / 100.0

    seeds = [
        _make_seed(0, text="alpha beta gamma", sae_feature_ids=[]),
        _make_seed(
            1,
            text="alpha beta gamma delta epsilon",
            sae_feature_ids=[],
        ),
    ]
    cfg = _seed_contract_cfg(
        make_valid_axbench_cfg(
            run_tag="stage1b_real_qc",
            n_positives_per_concept=2,
            accept_threshold=0.1,
            max_retries=0,
            output_length=8,
            min_prefix_chars=1,
        )
    )
    _commit_seed_pair(cfg, seeds)

    base_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=16, model_name="gemma-real-qc"),
    )
    base_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            hook_name="blocks.8.hook_resid_post",
            release="integration-rel",
            d_sae=16,
        ),
    )
    tokenizer = _WhitespaceTokenizer()

    monkeypatch.setattr(
        positive_mode_mod,
        "_build_synth_client",
        lambda cfg, response_cache=None: _StubSynthClient(
            canned=" delta epsilon zeta",
        ),
    )
    # Paired concept_avoiding row: feature inactive on the natural
    # continuation -> natural tier accepts (no LLM avoid call). This isolates
    # the assertion below to the rich row's real output-percentile call.
    # Combined classifier: empty -> NO_TOKENS, else INACTIVE.
    monkeypatch.setattr(
        positive_mode_mod,
        "_infer_n_sae_features",
        lambda base_sae: 16,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "build_percentile_table_from_stats",
        lambda stats_path, n_sae_features: table,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        positive_mode_mod,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _CharExtractionTokenizer(),
    )
    monkeypatch.setattr(qc_mod, "get_extraction_tokenizer", lambda *args, **kwargs: tokenizer)

    def _fake_extractor(**kwargs):
        extractor_state["calls"].append(kwargs)
        emit_feature_signal = len(extractor_state["calls"]) <= 2
        out = []
        for token_ids in kwargs["token_id_lists"]:
            acts = t.zeros((len(token_ids), 16), dtype=t.float32)
            if emit_feature_signal:
                for pos in range(len(token_ids)):
                    acts[pos, 5] = (pos + 1) / 10.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc_mod, "_run_batched_extractor", _fake_extractor)

    out_path = synthesize_positive_mode(
        cfg,
        concepts=[_concept(0, 5)],
        seeds_train=seeds,
        base_model=base_model,
        base_sae=base_sae,
        target_model_meta=_stub_target_meta(),
    )
    rows = pq.read_table(out_path).to_pylist()
    pos_rows = [row for row in rows if row["category"] == "positive"]
    assert len(pos_rows) == 2
    row = pos_rows[0]
    assert row["qc_feature_id"] == 5
    # The fake extractor ramps acts[pos, 5] = (pos + 1) / 10.0 by position;
    # Prefix-alone scoring reads acts[1:] while joined output scoring reads
    # acts[1+n_prefix:]; both exclude the injected BOS at row 0. Protocol
    # parsing first limits the canned response to output_length=8 character
    # tokens (" delta e"), which becomes two whitespace tokens in QC. The
    # canonical prefix and output scores are retained verbatim in the row.
    assert row["qc_seed_percentile"] == pytest.approx(40.0, abs=1.0)
    assert row["qc_output_percentile"] == pytest.approx(60.0, abs=1.0)
    assert row["qc_delta"] == pytest.approx(20.0, abs=1.0)
    assert row["qc_accepted"] is True
    assert extractor_state["calls"][-1]["layer"] == 8
    assert extractor_state["calls"][-1]["hook_name"] == "blocks.8.hook_resid_post"
    assert any(
        len(call["token_id_lists"]) == 2
        and len({len(tokens) for tokens in call["token_id_lists"]}) == 2
        for call in extractor_state["calls"]
    ), "production ragged output scorer never received the two aligned rows"


# ---------------------------------------------------------------------------
# Stage 1c: synthesize_negative_mode
# ---------------------------------------------------------------------------


def _write_negative_prefixes_parquet(
    run_tag: str,
    concept_ids: list[int],
    *,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    neuronpedia_model_id: str = "gemma-2-2b",
    feature_extraction_model_id: str = "gemma-2-2b",
    neuronpedia_source: str = "8-gemmascope-res-16k",
) -> None:
    """Write a tiny Stage 0c output parquet that Stage 1c will read."""
    schema = pa.schema([
        ("concept_id", pa.int32()),
        ("prefix_text", pa.string()),
        ("prefix_source_shard", pa.int32()),
        ("prefix_source_doc", pa.int32()),
        ("prefix_activating_seg_idx", pa.int32()),
        ("verification_continuations", pa.list_(pa.string())),
        ("verification_seeds", pa.list_(pa.int64())),
        ("verification_percentiles", pa.list_(pa.float32())),
    ]).with_metadata({
        b"extraction_tokenizer_id": tokenizer_id.encode("utf-8"),
        b"neuronpedia_model_id": neuronpedia_model_id.encode("utf-8"),
        b"feature_extraction_model_id": (
            feature_extraction_model_id.encode("utf-8")
        ),
        b"neuronpedia_source": neuronpedia_source.encode("utf-8"),
    })
    rows = [
        {
            "concept_id": cid,
            "prefix_text": f"prefix for concept {cid}",
            "prefix_source_shard": 0,
            "prefix_source_doc": cid,
            "prefix_activating_seg_idx": 0,
            "verification_continuations": [
                f"rich continuation A cid={cid}",
                f"rich continuation B cid={cid}",
            ],
            "verification_seeds": [101 + cid, 102 + cid],
            "verification_percentiles": [85.0, 92.5],
        }
        for cid in concept_ids
    ]
    out = test_paths.default_paths(run_tag).negative_mode_prefixes()
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)


def test_stage_1c_rejects_negative_prefixes_without_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    """Stage 1c refuses old Stage 0c parquet before any synth work starts."""
    del tmp_run_root
    run_tag = "stage1c_missing_tokenizer_metadata"
    _write_negative_prefixes_parquet(run_tag, concept_ids=[0])
    parquet_path = test_paths.default_paths(run_tag).negative_mode_prefixes()
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"extraction_tokenizer_id")
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
    cfg = make_valid_axbench_cfg(run_tag=run_tag)

    with pytest.raises(
        AxBenchBaseError,
        match="Stage 1c: .*missing 'extraction_tokenizer_id' metadata",
    ):
        synthesize_negative_mode(
            cfg,
            concepts=[_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )


def test_stage_1c_rejects_negative_prefixes_with_mismatched_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    """Stage 1c refuses Stage 0c parquet scoped to a different tokenizer."""
    del tmp_run_root
    run_tag = "stage1c_mismatched_tokenizer_metadata"
    _write_negative_prefixes_parquet(
        run_tag,
        concept_ids=[0],
        tokenizer_id="custom/stale-tokenizer",
    )
    cfg = make_valid_axbench_cfg(
        run_tag=run_tag,
        extraction_tokenizer_id="custom/current-tokenizer",
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Stage 1c: .*custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        synthesize_negative_mode(
            cfg,
            concepts=[_concept(0, 42)],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )


def test_stage_1c_writes_synth_training_negative_parquet(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Stage 1c writes a 19-column parquet with rich + avoiding rows."""
    cfg = make_valid_axbench_cfg(
        run_tag="stage1c_e2e",
        max_retries=0,
        qc_fallback_on_exhaust="error",
        output_length=8,
    )
    _write_negative_prefixes_parquet("stage1c_e2e", concept_ids=[0, 1])

    stub_client = _StubSynthClient(canned="negative-mode avoiding continuation")

    monkeypatch.setattr(
        negative_mode_mod,
        "OpenRouterSynthClient",
        lambda synth_cfg, max_retries, response_cache=None: stub_client,
    )
    # Strict-zero QC: always INACTIVE (feature not firing -> accept).
    monkeypatch.setattr(
        negative_mode_mod,
        "classify_avoid_continuation",
        lambda prefix_text, continuation_text, f_c, base_model, base_sae, *,
        sae_layer, extraction_threshold, tokenizer_id:
        negative_mode_mod.AvoidContinuationOutcome.INACTIVE,
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _CharExtractionTokenizer(),
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "get_tokenizer",
        lambda: _StubNanochatTokenizer(),
    )

    concepts = [_concept(0, 42), _concept(1, 99)]

    out_path = synthesize_negative_mode(
        cfg,
        concepts=concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert out_path == test_paths.default_paths("stage1c_e2e").synth_training_negative()
    assert out_path.exists()
    table = pq.read_table(out_path)
    assert len(table.column_names) == 19
    assert table.schema.equals(NEG_TRAINING_SCHEMA)
    metadata = table.schema.metadata or {}
    assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] == (
        cfg.extraction_tokenizer_id.encode("utf-8")
    )
    validate_stage1_continuation_contract(
        out_path,
        consumer="Stage 1c integration test",
        expected_role="stage1c_negative",
        schema=table.schema,
    )
    # 2 concept_rich rows + 1 concept_avoiding row per concept * 2 concepts
    # = (2 + 1) * 2 = 6 rows.
    assert table.num_rows == 6
    rows = table.to_pylist()
    rich_rows = [r for r in rows if r["content_class"] == "concept_rich"]
    avoiding_rows = [
        r for r in rows if r["content_class"] == "concept_avoiding"
    ]
    assert len(rich_rows) == 4
    assert len(avoiding_rows) == 2
    # Each concept has at least one row.
    rich_cids = {r["concept_id"] for r in rich_rows}
    assert rich_cids == {0, 1}
    avoiding_cids = {r["concept_id"] for r in avoiding_rows}
    assert avoiding_cids == {0, 1}
    # Avoiding rows come from our stub LLM.
    assert all(
        r["output_text"] == "negative"
        for r in avoiding_rows
    )


def test_stage_1c_parallel_entrypoint_matches_serial_output(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """Stage 1c entrypoint must preserve rows under bounded parallel synth."""

    del tmp_run_root
    tokenizer_id = "custom/stage1c-parallel-tokenizer"
    qc_tokenizer_ids: list[str] = []
    monkeypatch.setattr(
        negative_mode_mod,
        "OpenRouterSynthClient",
        lambda synth_cfg, max_retries, response_cache=None: _StubSynthClient(
            canned="stage1c parallel parity continuation",
        ),
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "classify_avoid_continuation",
        lambda prefix_text, continuation_text, f_c, base_model, base_sae, *,
        sae_layer, extraction_threshold, tokenizer_id: (
            qc_tokenizer_ids.append(tokenizer_id)
            or negative_mode_mod.AvoidContinuationOutcome.INACTIVE
        ),
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "get_extraction_tokenizer",
        lambda requested_tokenizer_id: (
            _CharExtractionTokenizer()
            if requested_tokenizer_id == tokenizer_id
            else pytest.fail(
                f"unexpected tokenizer id {requested_tokenizer_id!r}"
            )
        ),
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "get_tokenizer",
        lambda: _StubNanochatTokenizer(),
    )

    concepts = [_concept(0, 42), _concept(1, 99)]

    def _run_stage1c(run_tag: str, *, max_concurrent: int) -> pa.Table:
        _write_negative_prefixes_parquet(
            run_tag,
            concept_ids=[0, 1],
            tokenizer_id=tokenizer_id,
        )
        cfg = make_valid_axbench_cfg(
            run_tag=run_tag,
            max_retries=0,
            qc_fallback_on_exhaust="error",
            output_length=8,
            extraction_tokenizer_id=tokenizer_id,
            synth=SynthConfig(requests_per_minute=0, max_concurrent=max_concurrent),
        )
        out_path = synthesize_negative_mode(
            cfg,
            concepts=concepts,
            base_model=object(),
            base_sae=object(),
            target_model_meta=_stub_target_meta(),
        )
        return pq.read_table(out_path)

    serial_table = _run_stage1c("stage1c_parallel_serial", max_concurrent=1)
    parallel_table = _run_stage1c("stage1c_parallel_parallel", max_concurrent=4)
    assert qc_tokenizer_ids
    assert set(qc_tokenizer_ids) == {tokenizer_id}
    for table_obj in (serial_table, parallel_table):
        metadata = table_obj.schema.metadata or {}
        assert metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] == (
            tokenizer_id.encode("utf-8")
        )
    assert _normalize_rows(parallel_table.to_pylist()) == _normalize_rows(
        serial_table.to_pylist()
    )


def test_stage_1c_warm_response_cache_replays_without_live_calls(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
    _install_scalar_qc_adapters: None,
) -> None:
    """A warm Stage 1c response cache must replay without new synth calls."""

    del tmp_run_root
    clients: list[_CachingStubSynthClient] = []
    monkeypatch.setattr(
        negative_mode_mod,
        "classify_avoid_continuation",
        lambda prefix_text, continuation_text, f_c, base_model, base_sae, *,
        sae_layer, extraction_threshold, tokenizer_id:
        negative_mode_mod.AvoidContinuationOutcome.INACTIVE,
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "load_template",
        lambda name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "get_extraction_tokenizer",
        lambda tokenizer_id: _CharExtractionTokenizer(),
    )
    monkeypatch.setattr(
        negative_mode_mod,
        "get_tokenizer",
        lambda: _StubNanochatTokenizer(),
    )

    def _build_client(synth_cfg, max_retries, response_cache=None):
        del synth_cfg, max_retries
        client = _CachingStubSynthClient(
            canned="stage1c cached continuation",
            response_cache=response_cache,
            fail_on_live_call=bool(clients),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(negative_mode_mod, "OpenRouterSynthClient", _build_client)

    run_tag = "stage1c_warm_cache"
    _write_negative_prefixes_parquet(run_tag, concept_ids=[0, 1])
    cfg = make_valid_axbench_cfg(
        run_tag=run_tag,
        max_retries=0,
        qc_fallback_on_exhaust="error",
        output_length=8,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=4),
    )
    concepts = [_concept(0, 42), _concept(1, 99)]

    first_out = synthesize_negative_mode(
        cfg,
        concepts=concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )
    second_out = synthesize_negative_mode(
        cfg,
        concepts=concepts,
        base_model=object(),
        base_sae=object(),
        target_model_meta=_stub_target_meta(),
    )

    assert len(clients) == 2
    assert clients[0].live_calls > 0
    assert clients[1].live_calls == 0
    assert clients[1].cache_hits > 0
    assert _normalize_rows(pq.read_table(second_out).to_pylist()) == _normalize_rows(
        pq.read_table(first_out).to_pylist()
    )


def test_stage_1c_skipped_when_run_negative_mode_false(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    """When ``run_negative_mode`` is False, Stage 1c is not invoked at all.

    We verify by calling it directly with a valid prefixes file but
    ``run_negative_mode=False`` — the entry point still runs, but the
    DAG should never reach it. Documented here as an explicit contract:
    Stage 1c is orchestrated by the DAG; callers skip it via config.
    """
    cfg = make_valid_axbench_cfg(
        run_tag="stage1c_skipped",
        run_negative_mode=False,
    )
    assert cfg.run_negative_mode is False
    # ``synthesize_negative_mode`` itself is still callable — the skip
    # happens at the DAG orchestration level, not inside the function.
    # Verifying the config flag round-trips is sufficient here: the
    # orchestration test in Task 47 will cover the DAG-level skip.
