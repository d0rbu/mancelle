"""Stage-counter contract tests for all 7 BPB-Adv stage handlers.

Drives each ``_stage_*`` function with mocked dependencies under an active
:class:`AxBenchProfiler` and asserts that the exact set of ``record_counter``
names declared by the production code is emitted at runtime.

Closes the **GAP-1** mutation-coverage hole identified by adversarial review:
the existing functional tests mock ``_run_resolved`` and therefore never
exercise stage bodies — so Mutation #12 (delete every ``record_counter`` from
a stage) and Mutation #13 (typo a counter name) both passed silently. These
tests drive the real stage functions, so any drift between the declared
counter set and the runtime emission set is caught by name.

Each test follows the same recipe:

  1. Build the minimal :class:`AxBenchBpbAdvConfig` the stage needs.
  2. Monkeypatch the stage's outward calls (loaders, validators, scoring
     primitives, schema-checked tables) to return tiny in-memory fakes.
  3. Run the stage under an :class:`AxBenchProfiler` bound to ``tmp_path``.
  4. Assert that the names + values of the counter events match the
     stage's documented contract.

Stages with no body-level ``record_counter`` calls
(``prepare_context_cache`` and ``aggregate``) are still covered: the
``@_profiled`` decorator's ``_record_memory_counters`` post-hook always
emits the per-stage ``cpu_rss_kb`` counter (on POSIX), and that emission is
asserted instead so deleting the decorator gets caught too.
"""

from __future__ import annotations

import ast
import json
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

import scripts.eval.axbench_bpb_adv_eval as cli
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
)
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    MACRO_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)


def _candidates_read_table(
    candidate_table: pa.Table,
    *,
    universal_triggers: pa.Table | None = None,
    universal_triggers_rows: int = 0,
) -> Any:
    """A path-aware ``pq.read_table`` mock for ``_stage_build_candidates``.

    The stage reads BOTH the candidates artifact and the canonical
    universal_triggers artifact, so a single-return mock would feed the
    candidate table into the universal-triggers schema guard. This dispatches by
    filename: ``universal_triggers.parquet`` returns the (empty by default,
    schema-valid) trigger table; everything else returns ``candidate_table``.
    """

    triggers = (
        universal_triggers
        if universal_triggers is not None
        else _empty_table(UNIVERSAL_TRIGGERS_SCHEMA, n_rows=universal_triggers_rows)
    )

    def _read(path: Any) -> pa.Table:
        if str(path).endswith("universal_triggers.parquet"):
            return triggers
        return candidate_table

    return _read


def _read_counters(profile_dir: Path) -> dict[str, list[float]]:
    """Group emitted counter events by name; values preserve emission order."""
    grouped: dict[str, list[float]] = {}
    for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["type"] != "counter":
            continue
        grouped.setdefault(event["name"], []).append(event["value"])
    return grouped


def _empty_table(schema: pa.Schema, n_rows: int = 0) -> pa.Table:
    """Build a zero/n-row pyarrow Table conforming to ``schema``.

    List-typed columns get rows of empty lists rather than NULL so that
    sum/len operations on the column don't blow up. (The token-counter
    aggregator iterates ``len(...)`` per row and would raise on None.)
    """
    arrays = []
    for field in schema:
        if pa.types.is_list(field.type):
            inner_type = field.type.value_type
            arrays.append(pa.array([[] for _ in range(n_rows)], type=pa.list_(inner_type)))
        else:
            arrays.append(pa.nulls(n_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _minimal_cfg(tmp_path: Path) -> AxBenchBpbAdvConfig:
    """Smallest dataclass cfg the stage handlers need."""
    cfg_dict = {
        "source_run_dir": str(tmp_path / "source_run"),
        "output_root": str(tmp_path / "out"),
        "run_tag": "stage-contract",
        "device": "cpu",
        "stage": "validate_source",
        "target_models": ["decoder_only"],
        "base_methods": None,
        "adversarial_methods": [
            {
                "method": "additive_direction",
                "families": ["diffmean"],
                "magnitudes": [0.0, 1.0],
            }
        ],
        "feature_dir": str(tmp_path / "features"),
        "feature_statistics_path": str(tmp_path / "features" / "feature_statistics.safetensors"),
        "extraction_tokenizer_id": "google/gemma-2-2b",
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_source": "8-gemmascope-res-16k",
        "extraction_max_tokens": 2048,
        "extraction_threshold": 0.0,
        "decoder_only_target": {},
        "encdec_target": {},
        "lora_adapter": {},
        "filter_unaccepted_samples": True,
        "next_segment_conditioning_offset": False,
        "diffmean_and_probe_layer": 6,
        "steering_metric": "bpb_class_margin",
        "context_cache": {},
        "offload_between_stages": False,
        "scores_path": "",
    }
    return cfg_to_dataclass(OmegaConf.create(cfg_dict))


@pytest.fixture
def cfg(tmp_path: Path) -> AxBenchBpbAdvConfig:
    return _minimal_cfg(tmp_path)


# ---------------------------------------------------------------------------
# Stage 1: validate_source — emits {concepts, positive_rows, negative_rows,
#                                   synth_rows_total} + cpu_rss_kb via decorator
# ---------------------------------------------------------------------------
def test_validate_source_emits_documented_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_validation = MagicMock(
        n_concepts=7,
        n_positive_synth_rows=23,
        n_negative_synth_rows=11,
        n_dropped=0,
        contract=MagicMock(),
    )
    monkeypatch.setattr(cli, "validate_stage_0_3_source_run", lambda _src, **_kw: fake_validation)
    monkeypatch.setattr(cli, "write_source_contract", lambda _p, _c: None)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_validate_source(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.validate_source.concepts"] == [7]
    assert counters["axbench_bpb_adv.validate_source.positive_rows"] == [23]
    assert counters["axbench_bpb_adv.validate_source.negative_rows"] == [11]
    assert counters["axbench_bpb_adv.validate_source.synth_rows_total"] == [34]
    assert counters["axbench_bpb_adv.validate_source.dropped"] == [0]
    # The @_profiled decorator's post-hook fires `_record_memory_counters`,
    # which on POSIX always emits cpu_rss_kb.
    if cli._cpu_rss_kb() is not None:
        assert "axbench_bpb_adv.validate_source.cpu_rss_kb" in counters


# ---------------------------------------------------------------------------
# Stage 2: prepare_context_cache — NO body-level counters; cpu_rss_kb only
# ---------------------------------------------------------------------------
def test_prepare_context_cache_decorator_records_memory_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = MagicMock(name="RuntimeState")
    monkeypatch.setattr(cli, "_bpb_config_from_adv_config", lambda _cfg: object())
    monkeypatch.setattr(cli, "RuntimeState", lambda **_kw: sentinel)
    runs: list[Any] = []
    monkeypatch.setattr(cli, "_run_bpb_context_cache_stage", lambda state: runs.append(state))

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_prepare_context_cache(cfg, ddp_rank=0, ddp_world_size=1)

    assert runs == [sentinel]
    if cli._cpu_rss_kb() is not None:
        counters = _read_counters(profile_dir)
        assert "axbench_bpb_adv.prepare_context_cache.cpu_rss_kb" in counters


def _clustered_stage_source(events: list[str]) -> Any:
    """A stage source faithful to what ``_stage_prepare_context_cache`` consumes.

    Production reads ``.validation.context_prefixes()`` and feeds the result to
    ``canonicalize_prefix_population``, which requires a ``Mapping[str, str]`` and
    raises on ``None``. A bare ``None`` stub therefore dies inside the preflight's
    blanket ``except Exception``, which reports a programming error as a source
    validation failure — the shape that hid this from an earlier revision.

    The population is also hash-verified, so the value is computed with the real
    ``make_prefix_hash`` rather than a hardcoded digest; a literal would rot the
    moment that function changed.
    """

    prefix_text = "a prefix"
    return types.SimpleNamespace(
        validation=types.SimpleNamespace(
            context_prefixes=lambda: {prefix_text: make_prefix_hash(prefix_text)},
        )
    )


def test_clustered_context_cache_releases_extractor_before_target_guard(
    cfg: AxBenchBpbAdvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real 4B extractor must be gone before the nanochat target loads."""

    events: list[str] = []
    extraction_tokenizer = object()
    decoder_tokenizer = object()

    class FakeRuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            self.released = False

        def base_gemma_and_sae(self) -> tuple[object, object]:
            events.append("load_extractor")
            return object(), object()

        def release_base_gemma_and_sae(self) -> None:
            events.append("release_extractor")
            self.released = True

        def decoder_only(self) -> Any:
            assert self.released, "target loaded while extractor was still resident"
            events.append("load_target")
            return types.SimpleNamespace(tokenizer=decoder_tokenizer)

        def extraction_tokenizer(self) -> object:
            events.append(
                "reuse_extraction_tokenizer_after_release"
                if self.released
                else "load_extraction_tokenizer_preflight"
            )
            return extraction_tokenizer

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_a: events.append("validate_clustered_source")
        or _clustered_stage_source(events),
    )
    monkeypatch.setattr(
        cli,
        "_bpb_config_from_adv_config",
        lambda _cfg: types.SimpleNamespace(feature_dir=cfg.feature_dir),
    )
    monkeypatch.setattr(cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(
        cli,
        "get_tokenizer",
        lambda: events.append("load_target_tokenizer_preflight") or decoder_tokenizer,
    )
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **_kwargs: events.append("build_cache"),
    )
    monkeypatch.setattr(cli, "assert_tokenizer_checksum_matches", lambda **_kw: None)
    monkeypatch.setattr(cli, "assert_bos_lengths_match", lambda **_kw: None)

    cli._stage_prepare_context_cache(
        cfg,
        ddp_rank=0,
        ddp_world_size=1,
        cluster_contract=object(),
    )

    assert events == [
        # The clustered source preflight runs FIRST: its whole purpose is to fail
        # before any model load or cache write.
        "validate_clustered_source",
        "load_extractor",
        "load_target_tokenizer_preflight",
        "load_extraction_tokenizer_preflight",
        "build_cache",
        "release_extractor",
        "load_target",
        "reuse_extraction_tokenizer_after_release",
    ]


def test_clustered_context_cache_tokenizer_preflight_fails_before_builder(
    cfg: AxBenchBpbAdvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeRuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def base_gemma_and_sae(self) -> tuple[object, object]:
            events.append("load_extractor")
            return object(), object()

        def extraction_tokenizer(self) -> object:
            events.append("load_extraction_tokenizer")
            return object()

    monkeypatch.setattr(
        cli,
        "_validate_clustered_source_for_stage",
        lambda *_a: events.append("validate_clustered_source")
        or _clustered_stage_source(events),
    )
    monkeypatch.setattr(
        cli,
        "_bpb_config_from_adv_config",
        lambda _cfg: types.SimpleNamespace(feature_dir=cfg.feature_dir),
    )
    monkeypatch.setattr(cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(
        cli,
        "get_tokenizer",
        lambda: events.append("load_target_tokenizer") or object(),
    )
    monkeypatch.setattr(
        cli,
        "assert_tokenizer_checksum_matches",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("checksum mismatch")),
    )
    monkeypatch.setattr(
        cli,
        "build_clustered_context_cache",
        lambda **_kwargs: pytest.fail("preflight failure must not build the cache"),
    )

    with pytest.raises(AxBenchBpbAdvError, match="before extraction or cache writes"):
        cli._stage_prepare_context_cache(
            cfg,
            ddp_rank=0,
            ddp_world_size=1,
            cluster_contract=object(),
        )

    assert events == [
        "validate_clustered_source",
        "load_extractor",
        "load_target_tokenizer",
    ]


# ---------------------------------------------------------------------------
# Stage 3: prepare_attack_rows — work_specs, samples, positive_rows_kept,
#          negative_rows_kept, prefix_groups, core_prefixes, dropped,
#          attack_rows
# ---------------------------------------------------------------------------
def test_prepare_attack_rows_emits_documented_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #505 class-margin: samples carry real concept_id/mode/prefix_text so the
    # stage can compute n_groups = len(unique (concept_id, mode, prefix)) and
    # then pair them into core prefixes. We pick 2 positive + 2 negative rows
    # that form exactly 2 prefix groups (one positive prefix, one negative
    # prefix, each with 2 rows sharing the prefix_text), and mock the pairing
    # to return a 2-element list → n_dropped = n_groups - len(prefixes) = 0.
    fake_validation = MagicMock(contract=MagicMock(), n_dropped=0)
    monkeypatch.setattr(cli, "validate_stage_0_3_source_run", lambda _src, **_kw: fake_validation)
    monkeypatch.setattr(cli, "write_source_contract", lambda _p, _c: None)
    tokenizer = MagicMock()
    monkeypatch.setattr(
        cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg: ({"decoder_only": MagicMock()}, tokenizer),
    )
    work_specs = [
        types.SimpleNamespace(mode=SteeringMode.POSITIVE),
        types.SimpleNamespace(mode=SteeringMode.NEGATIVE),
        types.SimpleNamespace(mode=SteeringMode.POSITIVE),
    ]
    monkeypatch.setattr(cli, "work_specs_from_method_registry", lambda **_kw: work_specs)
    monkeypatch.setattr(
        cli,
        "evaluation_prefixes_for_scoring",
        lambda prefixes, modes, ours: list(prefixes),
    )
    raw_rows = [{"concept_id": 1}, {"concept_id": 2}]
    monkeypatch.setattr(
        cli,
        "load_raw_synth_rows",
        lambda _path, filter_unaccepted, **_kwargs: list(raw_rows),
    )
    monkeypatch.setattr(cli, "_read_source_concept_ids", lambda _path: {1, 2})
    monkeypatch.setattr(
        cli,
        "_filter_synth_rows_to_source_concepts",
        lambda rows, concept_ids, mode: list(rows),
    )

    # Each call returns a sample with a real concept_id / mode / prefix_text.
    # POSITIVE rows share prefix "pos-prefix"; NEGATIVE rows share "neg-prefix".
    # → exactly 2 unique (concept_id?, mode, prefix) groups... but concept_id
    # differs per row, so to land exactly 2 groups we key all POSITIVE samples
    # to concept_id=1 and all NEGATIVE to concept_id=2 with a shared prefix.
    def fake_sample_from_row(row, mode, tokenizer):  # noqa: ANN001, ARG001
        if mode == SteeringMode.POSITIVE:
            return types.SimpleNamespace(
                concept_id=1,
                mode=SteeringMode.POSITIVE.value,
                prefix_text="pos-prefix",
            )
        return types.SimpleNamespace(
            concept_id=2,
            mode=SteeringMode.NEGATIVE.value,
            prefix_text="neg-prefix",
        )

    monkeypatch.setattr(cli, "bpb_attack_sample_from_synth_row", fake_sample_from_row)
    # 2 prefix groups → pair into 2 core prefixes → n_dropped = 0.
    prefixes = [MagicMock(name="prefix0"), MagicMock(name="prefix1")]
    monkeypatch.setattr(cli, "pair_samples_into_prefixes", lambda samples, **_kw: list(prefixes))
    fake_attack_table = _empty_table(ATTACK_ROWS_SCHEMA, n_rows=42)
    monkeypatch.setattr(
        cli,
        "prepare_bpb_attack_rows",
        lambda prefixes, work_specs, cfg: fake_attack_table,
    )
    monkeypatch.setattr(cli, "write_evaluation_population", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli.pq, "write_table", lambda _t, _p: None)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_prepare_attack_rows(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.prepare_attack_rows.work_specs"] == [3]
    # 2 positive + 2 negative = 4 samples
    assert counters["axbench_bpb_adv.prepare_attack_rows.samples"] == [4]
    assert counters["axbench_bpb_adv.prepare_attack_rows.positive_rows_kept"] == [2]
    assert counters["axbench_bpb_adv.prepare_attack_rows.negative_rows_kept"] == [2]
    # 2 unique (concept_id, mode, prefix) groups.
    assert counters["axbench_bpb_adv.prepare_attack_rows.prefix_groups"] == [2]
    # paired into 2 complete core prefixes → 0 dropped.
    assert counters["axbench_bpb_adv.prepare_attack_rows.core_prefixes"] == [2]
    assert counters["axbench_bpb_adv.prepare_attack_rows.evaluation_core_prefixes"] == [2]
    assert counters["axbench_bpb_adv.prepare_attack_rows.dropped"] == [0]
    assert counters["axbench_bpb_adv.prepare_attack_rows.attack_rows"] == [42]


# ---------------------------------------------------------------------------
# Stage 4: build_candidates — candidates_written, candidate_rows
# ---------------------------------------------------------------------------
def test_build_candidates_emits_documented_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run_model_backed_bpb_candidate_stage", lambda **_kw: 99)
    fake_candidates = _empty_table(CANDIDATES_SCHEMA, n_rows=99)
    # Path-aware: the candidates read returns the candidate table; the
    # universal_triggers read returns a schema-valid 7-row trigger table.
    monkeypatch.setattr(
        cli.pq,
        "read_table",
        _candidates_read_table(fake_candidates, universal_triggers_rows=7),
    )

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_build_candidates(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.build_candidates.candidates_written"] == [99]
    assert counters["axbench_bpb_adv.build_candidates.candidate_rows"] == [99]
    assert counters["axbench_bpb_adv.build_candidates.universal_triggers_rows"] == [7]


# ---------------------------------------------------------------------------
# Stage 5: score — candidates_loaded, score_rows
# ---------------------------------------------------------------------------
def test_score_emits_documented_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #505: the score stage no longer loads neutral references at all. It
    # loads candidates, runs the model-backed scorer, checks SCORE_SCHEMA, and
    # records candidates_loaded + score_rows.
    fake_candidates = _empty_table(CANDIDATES_SCHEMA, n_rows=17)
    monkeypatch.setattr(cli, "load_candidate_table", lambda _p: fake_candidates)
    fake_scores = _empty_table(SCORE_SCHEMA, n_rows=29)
    monkeypatch.setattr(cli, "run_model_backed_bpb_adv_score_stage", lambda **_kw: fake_scores)
    monkeypatch.setattr(cli.pq, "write_table", lambda _t, _p: None)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_score(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.score.candidates_loaded"] == [17]
    assert counters["axbench_bpb_adv.score.score_rows"] == [29]


# ---------------------------------------------------------------------------
# Stage 6: aggregate — NO body-level counters; cpu_rss_kb only
# ---------------------------------------------------------------------------
def test_aggregate_decorator_records_memory_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        cli,
        "aggregate_bpb_adv_scores",
        lambda score_path, aggregates_dir, general_val_path=None: calls.append(
            (score_path, aggregates_dir)
        ),
    )

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_aggregate(cfg)

    assert len(calls) == 1
    if cli._cpu_rss_kb() is not None:
        counters = _read_counters(profile_dir)
        assert "axbench_bpb_adv.aggregate.cpu_rss_kb" in counters


# ---------------------------------------------------------------------------
# Stage 7: plot — macro_rows, files_written
# ---------------------------------------------------------------------------
def test_plot_emits_documented_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_stage_plot` records macro_rows + (gated) files_written.

    The fake mirrors the REAL layout of ``plot_bpb_adv_summaries``
    (`aggregation.py:649-745`): two top-level CSVs (``plot_series_legend.csv``
    + ``plot_index.csv``) plus PNGs in nested ``<mode>/`` subdirs.
    So the counter must include nested files via ``rglob`` — a non-recursive
    glob would silently miss every PNG and report 2 instead of N.
    """
    # Make macro_path exist so the stage skips re-aggregation.
    out_paths = cli._paths(cfg)
    out_paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    (out_paths.aggregates_dir() / "macro.parquet").touch()
    # #505 added a macro-schema freshness check (pq.read_schema(...).equals(
    # MACRO_AGG_SCHEMA)) before skipping re-aggregation; make the touched stub
    # read as fresh so the stage skips re-aggregation (these tests' intent).
    monkeypatch.setattr(cli.pq, "read_schema", lambda _p: MACRO_AGG_SCHEMA)
    fake_macro = pa.Table.from_pydict({"x": list(range(53))})
    monkeypatch.setattr(cli.pq, "read_table", lambda _path: fake_macro)

    def fake_plot(_macro_table: pa.Table, plots_dir: Path, **_kwargs: Any) -> list[Path]:
        # 2 top-level CSVs + 4 PNGs in 2 subdirs (matches the real fanout
        # of ``plot_bpb_adv_summaries`` which writes to mode/*.png).
        plots_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for fname in ("plot_series_legend.csv", "plot_index.csv"):
            p = plots_dir / fname
            p.write_text("")
            written.append(p)
        for mode in ("positive", "negative"):
            subdir = plots_dir / mode
            subdir.mkdir(parents=True, exist_ok=True)
            for plot_name in ("series_grid.png", "metric_vs_strength.png"):
                p = subdir / plot_name
                p.write_bytes(b"")
                written.append(p)
        return written

    monkeypatch.setattr(cli, "plot_bpb_adv_summaries", fake_plot)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_plot(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.plot.macro_rows"] == [53]
    # 2 CSVs (top-level) + 2 subdirs × 2 PNGs = 6 files written total
    assert counters["axbench_bpb_adv.plot.files_written"] == [6]


# ---------------------------------------------------------------------------
# Cross-stage contract: every @_profiled stage emits at least one
# memory-counter or body-level counter (kills the "delete @_profiled
# decorator" mutation).
# ---------------------------------------------------------------------------
def _candidates_with_tokens(n_rows: int, prompt_len: int, output_len: int) -> pa.Table:
    """A small CANDIDATES_SCHEMA-conforming table with real token lists.

    Used by token-counter tests so the per-row ``len(...)`` sum has a known
    ground truth. Every other column is filled with valid-typed placeholders.

    #505 class-margin: the old single ``output_tokens`` column is replaced by
    two per-completion columns — ``concept_rich_output_tokens`` (length
    ``output_len``) and ``concept_avoiding_output_tokens`` (length
    ``output_len + 2`` so the two are distinguishable).
    """
    src_tokens = [list(range(prompt_len)) for _ in range(n_rows)]
    atk_tokens = [
        list(range(prompt_len + 1)) for _ in range(n_rows)
    ]  # +1 so the two are distinguishable
    rich_tokens = [list(range(output_len)) for _ in range(n_rows)]
    avoid_tokens = [
        list(range(output_len + 2)) for _ in range(n_rows)
    ]  # +2 so the two are distinguishable
    pydict: dict[str, list[Any]] = {
        "candidate_id": [f"c{i}" for i in range(n_rows)],
        "attack_row_id": [f"a{i}" for i in range(n_rows)],
        "core_prefix_id": [f"p{i}" for i in range(n_rows)],
        "concept_rich_sample_id": [f"rich{i}" for i in range(n_rows)],
        "concept_avoiding_sample_id": [f"avoid{i}" for i in range(n_rows)],
        "concept_id": list(range(n_rows)),
        "mode": ["positive"] * n_rows,
        "method_name": ["m"] * n_rows,
        "target_model": ["decoder_only"] * n_rows,
        "coefficient_str": ["1.0"] * n_rows,
        "adversarial_method": ["additive_direction"] * n_rows,
        "adversarial_strength": [0.0] * n_rows,
        "strength_unit": ["raw"] * n_rows,
        "strength_index": [0] * n_rows,
        "context_signal_mode": ["fixed_context"] * n_rows,
        "prefix_text": [""] * n_rows,
        "prefix_hash": [""] * n_rows,
        "source_prompt_tokens": src_tokens,
        "source_prompt_token_digest": [""] * n_rows,
        "attacked_prompt_text": [""] * n_rows,
        "attacked_prompt_tokens": atk_tokens,
        "attacked_prompt_token_digest": [""] * n_rows,
        "concept_rich_output_tokens": rich_tokens,
        "concept_avoiding_output_tokens": avoid_tokens,
        "concept_rich_output_digest": [""] * n_rows,
        "concept_avoiding_output_digest": [""] * n_rows,
        "editable_token_start": [0] * n_rows,
        "editable_token_count": [0] * n_rows,
        "replacement_tokens": [[] for _ in range(n_rows)],
        "replacement_text": [""] * n_rows,
        "objective_value": [0.0] * n_rows,
        "candidate_metadata_json": ['{"candidate":"no_adversary_control"}'] * n_rows,
    }
    return pa.Table.from_pydict(pydict, schema=CANDIDATES_SCHEMA)


def test_build_candidates_emits_token_counters_when_profiler_bound(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#501/#505: when a profiler is bound, ``_stage_build_candidates`` must
    emit aggregate token counters (sum of per-row token-list lengths) for the
    four CANDIDATES_SCHEMA token columns: source_prompt_tokens,
    attacked_prompt_tokens, concept_rich_output_tokens, and
    concept_avoiding_output_tokens."""
    monkeypatch.setattr(cli, "run_model_backed_bpb_candidate_stage", lambda **_kw: 5)
    fake = _candidates_with_tokens(n_rows=5, prompt_len=7, output_len=11)
    monkeypatch.setattr(cli.pq, "read_table", _candidates_read_table(fake))

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_build_candidates(cfg)

    counters = _read_counters(profile_dir)
    # 5 rows × 7 tokens each
    assert counters["axbench_bpb_adv.build_candidates.source_prompt_tokens_total"] == [35]
    # 5 rows × 8 tokens each (prompt_len + 1)
    assert counters["axbench_bpb_adv.build_candidates.attacked_prompt_tokens_total"] == [40]
    # 5 rows × 11 tokens each (output_len)
    assert counters["axbench_bpb_adv.build_candidates.concept_rich_output_tokens_total"] == [55]
    # 5 rows × 13 tokens each (output_len + 2)
    assert counters["axbench_bpb_adv.build_candidates.concept_avoiding_output_tokens_total"] == [65]


def test_build_candidates_skips_token_aggregation_when_no_profiler(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict byte-identity: aggregating per-row token lengths is O(rows)
    work whose only consumer is the profiler. Without a bound profiler the
    aggregation helper must NOT be invoked.

    We spy on the module-level ``_sum_list_lengths`` helper (mutable, unlike
    pyarrow.Table) and assert zero calls. The realistic candidate fake is
    reused from :func:`_candidates_with_tokens` so the test would otherwise
    succeed by accident if the helper *were* called.
    """
    monkeypatch.setattr(cli, "run_model_backed_bpb_candidate_stage", lambda **_kw: 5)
    fake = _candidates_with_tokens(n_rows=3, prompt_len=4, output_len=5)
    monkeypatch.setattr(cli.pq, "read_table", _candidates_read_table(fake))

    helper_calls: list[str] = []
    real_helper = cli._sum_list_lengths

    def spy(table: pa.Table, column_name: str) -> int:
        helper_calls.append(column_name)
        return real_helper(table, column_name)

    monkeypatch.setattr(cli, "_sum_list_lengths", spy)

    # No profiler bound.
    cli._stage_build_candidates(cfg)
    assert helper_calls == [], (
        f"_sum_list_lengths called {helper_calls!r} with no profiler bound — "
        "aggregation must be gated on get_current_profiler()"
    )


def _score_table(
    n_rows: int,
    *,
    n_rich_output_tokens: list[int] | None = None,
    n_avoid_output_tokens: list[int] | None = None,
) -> pa.Table:
    """Build a SCORE_SCHEMA-conforming table with valid-typed placeholders.

    Every column is populated by type from the live SCORE_SCHEMA so the table
    is schema-equal. The two per-completion output-token-count columns are
    overridable so token-counter tests have a known ground truth.
    """
    rich = n_rich_output_tokens if n_rich_output_tokens is not None else [0] * n_rows
    avoid = n_avoid_output_tokens if n_avoid_output_tokens is not None else [0] * n_rows
    overrides: dict[str, list[Any]] = {
        "n_rich_output_tokens": rich,
        "n_avoid_output_tokens": avoid,
        "source_prompt_tokens": [[1, 2]] * n_rows,
        "attacked_prompt_tokens": [[1, 2, 3]] * n_rows,
    }
    arrays = []
    for field in SCORE_SCHEMA:
        if field.name in overrides:
            arrays.append(pa.array(overrides[field.name], type=field.type))
        elif pa.types.is_list(field.type):
            inner = field.type.value_type
            arrays.append(pa.array([[] for _ in range(n_rows)], type=pa.list_(inner)))
        elif pa.types.is_boolean(field.type):
            arrays.append(pa.array([False] * n_rows, type=field.type))
        elif pa.types.is_floating(field.type):
            arrays.append(pa.array([0.0] * n_rows, type=field.type))
        elif pa.types.is_integer(field.type):
            arrays.append(pa.array([0] * n_rows, type=field.type))
        else:
            arrays.append(pa.array([""] * n_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=SCORE_SCHEMA)


def _roundtrip_score_table(
    n_rows: int,
    *,
    n_generated_tokens: list[int] | None = None,
    control_n_generated_tokens: list[int] | None = None,
) -> pa.Table:
    generated = n_generated_tokens if n_generated_tokens is not None else [0] * n_rows
    control_generated = (
        control_n_generated_tokens if control_n_generated_tokens is not None else [0] * n_rows
    )
    overrides: dict[str, list[Any]] = {
        "n_generated_tokens": generated,
        "control_n_generated_tokens": control_generated,
        "source_prompt_tokens": [[1, 2]] * n_rows,
        "attacked_prompt_tokens": [[1, 2, 3]] * n_rows,
        "roundtrip_prompt_tokens": [[1, 2, 3]] * n_rows,
        "control_roundtrip_prompt_tokens": [[1, 2, 4]] * n_rows,
        "replacement_tokens": [[] for _ in range(n_rows)],
    }
    arrays = []
    for field in ROUNDTRIP_SCORE_SCHEMA:
        if field.name in overrides:
            arrays.append(pa.array(overrides[field.name], type=field.type))
        elif pa.types.is_list(field.type):
            inner = field.type.value_type
            arrays.append(pa.array([[] for _ in range(n_rows)], type=pa.list_(inner)))
        elif pa.types.is_boolean(field.type):
            arrays.append(pa.array([False] * n_rows, type=field.type))
        elif pa.types.is_floating(field.type):
            arrays.append(pa.array([0.0] * n_rows, type=field.type))
        elif pa.types.is_integer(field.type):
            arrays.append(pa.array([0] * n_rows, type=field.type))
        else:
            arrays.append(pa.array([""] * n_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=ROUNDTRIP_SCORE_SCHEMA)


def test_score_emits_token_counters_when_profiler_bound(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#501/#505: when a profiler is bound, ``_stage_score`` must emit aggregate
    counters for prompt tokens (from the candidates table) and per-completion
    output tokens (from the SCORE_SCHEMA ``n_rich_output_tokens`` /
    ``n_avoid_output_tokens`` int32 columns, already pre-aggregated per row by
    the dual-completion scorer)."""
    # Real candidates with token lists so the prompt-token sum is meaningful.
    candidates_with_tokens = _candidates_with_tokens(n_rows=4, prompt_len=10, output_len=20)
    monkeypatch.setattr(cli, "load_candidate_table", lambda _p: candidates_with_tokens)

    # Build a SCORE_SCHEMA table with the per-completion token-count columns
    # populated (rich sum=34, avoid sum=18).
    scores = _score_table(
        4,
        n_rich_output_tokens=[7, 8, 9, 10],
        n_avoid_output_tokens=[3, 4, 5, 6],
    )
    monkeypatch.setattr(cli, "run_model_backed_bpb_adv_score_stage", lambda **_kw: scores)
    monkeypatch.setattr(cli.pq, "write_table", lambda _t, _p: None)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_score(cfg)

    counters = _read_counters(profile_dir)
    # source_prompt_tokens: 4 rows × 10 tokens (candidates) = 40
    assert counters["axbench_bpb_adv.score.source_prompt_tokens_total"] == [40]
    # attacked_prompt_tokens: 4 rows × 11 tokens (candidates, prompt_len+1) = 44
    assert counters["axbench_bpb_adv.score.attacked_prompt_tokens_total"] == [44]
    # rich_output_tokens: sum of n_rich_output_tokens = 7+8+9+10 = 34
    assert counters["axbench_bpb_adv.score.rich_output_tokens_total"] == [34]
    # avoid_output_tokens: sum of n_avoid_output_tokens = 3+4+5+6 = 18
    assert counters["axbench_bpb_adv.score.avoid_output_tokens_total"] == [18]


def test_roundtrip_score_emits_roundtrip_profiler_counters(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg.steering_metric = "roundtrip"
    candidates_with_tokens = _candidates_with_tokens(n_rows=2, prompt_len=4, output_len=5)
    monkeypatch.setattr(cli, "load_candidate_table", lambda _p: candidates_with_tokens)
    scores = _roundtrip_score_table(
        2,
        n_generated_tokens=[7, 11],
        control_n_generated_tokens=[3, 5],
    )

    def fake_roundtrip_score_stage(**kwargs: Any) -> pa.Table:
        stats = kwargs["stats"]
        stats.update(
            {
                "cache_hits": 13,
                "scored_unique": 17,
                "generation_cache_hits": 19,
                "generated_unique": 23,
                "generated_tokens": 41,
                "control_generation_cache_hits": 29,
                "control_generated_unique": 31,
                "control_generated_tokens": 43,
                "probe_rows": 37,
            }
        )
        return scores

    monkeypatch.setattr(cli, "run_model_backed_roundtrip_score_stage", fake_roundtrip_score_stage)
    monkeypatch.setattr(cli.pq, "write_table", lambda _t, _p: None)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_score(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.score.scoring_cache_hits"] == [13]
    assert counters["axbench_bpb_adv.score.scored_unique"] == [17]
    assert counters["axbench_bpb_adv.score.bpb_scoring_cache_hits"] == [13]
    assert counters["axbench_bpb_adv.score.bpb_scored_unique"] == [17]
    assert counters["axbench_bpb_adv.score.roundtrip_generation_cache_hits"] == [19]
    assert counters["axbench_bpb_adv.score.roundtrip_generated_unique"] == [23]
    assert counters["axbench_bpb_adv.score.roundtrip_control_generation_cache_hits"] == [29]
    assert counters["axbench_bpb_adv.score.roundtrip_control_generated_unique"] == [31]
    assert counters["axbench_bpb_adv.score.roundtrip_generated_tokens_total"] == [41]
    assert counters["axbench_bpb_adv.score.roundtrip_control_generated_tokens_total"] == [43]
    assert counters["axbench_bpb_adv.score.roundtrip_probe_rows"] == [37]
    assert all("empty_response" not in key for key in counters)


def test_plot_falls_back_to_aggregate_when_macro_missing(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_stage_plot` must call `aggregate_bpb_adv_scores` when `macro.parquet`
    doesn't already exist (the documented fallback so plot is runnable
    standalone without an earlier aggregate stage).

    A regression that dropped the fallback would crash on the subsequent
    `pq.read_table(macro_path)` with a less actionable error.
    """
    out_paths = cli._paths(cfg)
    aggregate_calls: list[Any] = []

    def fake_aggregate(
        score_path: Path, aggregates_dir: Path, *, general_val_path: Path | None = None
    ) -> None:
        aggregate_calls.append((score_path, aggregates_dir))
        aggregates_dir.mkdir(parents=True, exist_ok=True)
        # Write a minimal macro.parquet so the next read_table succeeds.
        pq.write_table(
            pa.Table.from_pydict({"x": [1, 2]}),
            aggregates_dir / "macro.parquet",
        )

    monkeypatch.setattr(cli, "aggregate_bpb_adv_scores", fake_aggregate)
    monkeypatch.setattr(cli, "plot_bpb_adv_summaries", lambda *_a, **_kw: None)

    # NOTE: Do NOT touch macro.parquet here — the fallback must create it.
    assert not (out_paths.aggregates_dir() / "macro.parquet").exists()

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_plot(cfg)

    assert len(aggregate_calls) == 1, (
        "macro.parquet absent → _stage_plot must call aggregate_bpb_adv_scores"
    )
    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.plot.macro_rows"] == [2]


def test_score_skips_token_aggregation_when_no_profiler(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric to `test_build_candidates_skips_token_aggregation_when_no_profiler`.

    Strict byte-identity: aggregating per-row token lengths is O(rows) work
    whose only consumer is the profiler. Without a bound profiler, `_stage_score`
    must NOT invoke `_sum_list_lengths` on the candidates table. Mirrors the
    build_candidates gate test so a mutation that drops the score-stage gate is
    caught immediately.

    Note: the score stage's four token counters (the two `_sum_list_lengths`
    candidate scans AND the two `scores.column(...).to_pylist()` int-column
    scans) all live inside ONE shared ``if get_current_profiler() is not None:``
    block. So the `_sum_list_lengths`-not-called assertion below transitively
    proves the int-column scans were skipped too — the whole block did not run.
    """
    candidates_with_tokens = _candidates_with_tokens(n_rows=3, prompt_len=4, output_len=5)
    monkeypatch.setattr(cli, "load_candidate_table", lambda _p: candidates_with_tokens)
    scores = _empty_table(SCORE_SCHEMA, n_rows=0)
    monkeypatch.setattr(cli, "run_model_backed_bpb_adv_score_stage", lambda **_kw: scores)
    monkeypatch.setattr(cli.pq, "write_table", lambda _t, _p: None)

    helper_calls: list[str] = []
    real_helper = cli._sum_list_lengths

    def spy(table: pa.Table, column_name: str) -> int:
        helper_calls.append(column_name)
        return real_helper(table, column_name)

    monkeypatch.setattr(cli, "_sum_list_lengths", spy)

    # No profiler bound.
    cli._stage_score(cfg)
    assert helper_calls == [], (
        f"_sum_list_lengths called {helper_calls!r} in _stage_score "
        "with no profiler bound — aggregation must be gated"
    )


def test_build_candidates_raises_on_schema_drift(
    cfg: AxBenchBpbAdvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One representative schema-drift guard test. The `if not ...schema.equals(SCHEMA): raise`
    pattern repeats in 3 stages (prepare_attack_rows, build_candidates, score);
    a mutation that removes this defensive guard from `_stage_build_candidates`
    is now caught.
    """
    monkeypatch.setattr(cli, "run_model_backed_bpb_candidate_stage", lambda **_kw: 1)
    # The candidates read returns a WRONG-schema table (single 'foo' column); the
    # universal_triggers read is valid, so this pins the CANDIDATE guard (which
    # runs first), not the universal-triggers guard.
    wrong = pa.Table.from_pydict({"foo": [1]})
    monkeypatch.setattr(cli.pq, "read_table", _candidates_read_table(wrong))

    with pytest.raises(AxBenchBpbAdvError, match="candidate schema drift"):
        cli._stage_build_candidates(cfg)


def test_build_candidates_raises_on_universal_triggers_schema_drift(
    cfg: AxBenchBpbAdvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The universal-triggers read-back guard catches a schema drift in the
    canonical universal_triggers artifact, independently of the candidate guard.

    The candidates read returns a VALID candidate table (so the candidate guard
    passes), but the universal_triggers read returns a WRONG-schema table — only
    the universal-triggers guard can raise here, so this pins it (and would die
    if the guard were removed).
    """

    monkeypatch.setattr(cli, "run_model_backed_bpb_candidate_stage", lambda **_kw: 1)
    valid_candidates = _empty_table(CANDIDATES_SCHEMA, n_rows=1)
    wrong_triggers = pa.Table.from_pydict({"oops": [1]})
    monkeypatch.setattr(
        cli.pq,
        "read_table",
        _candidates_read_table(valid_candidates, universal_triggers=wrong_triggers),
    )

    with pytest.raises(AxBenchBpbAdvError, match="universal_triggers schema drift"):
        cli._stage_build_candidates(cfg)


def test_score_raises_on_schema_drift(
    cfg: AxBenchBpbAdvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second representative schema-drift guard test (score stage's drift
    catch comes AFTER `run_model_backed_bpb_adv_score_stage` returns; this
    pins the post-scorer schema check)."""
    candidates_with_tokens = _candidates_with_tokens(n_rows=1, prompt_len=2, output_len=3)
    monkeypatch.setattr(cli, "load_candidate_table", lambda _p: candidates_with_tokens)

    # Scorer returns a table with the WRONG schema.
    wrong_scores = pa.Table.from_pydict({"bar": [1.0]})
    monkeypatch.setattr(cli, "run_model_backed_bpb_adv_score_stage", lambda **_kw: wrong_scores)

    with pytest.raises(AxBenchBpbAdvError, match="score schema drift"):
        cli._stage_score(cfg)


def test_plot_skips_files_written_glob_when_no_profiler(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict byte-identity: ``_stage_plot`` calls ``plots_dir().glob('*')``
    only to feed the ``files_written`` counter. When no profiler is bound the
    counter no-ops, so the glob is wasted I/O — production should NOT touch
    the filesystem on its behalf.

    Identified by Agent A's adversarial review of PR #503.
    """
    out_paths = cli._paths(cfg)
    out_paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    (out_paths.aggregates_dir() / "macro.parquet").touch()
    # #505 added a macro-schema freshness check (pq.read_schema(...).equals(
    # MACRO_AGG_SCHEMA)) before skipping re-aggregation; make the touched stub
    # read as fresh so the stage skips re-aggregation (these tests' intent).
    monkeypatch.setattr(cli.pq, "read_schema", lambda _p: MACRO_AGG_SCHEMA)
    fake_macro = pa.Table.from_pydict({"x": [1, 2, 3]})
    monkeypatch.setattr(cli.pq, "read_table", lambda _p: fake_macro)
    monkeypatch.setattr(cli, "plot_bpb_adv_summaries", lambda *_a, **_kw: None)

    # Spy: instrument Path.glob to record any call against plots_dir().
    plots_dir = out_paths.plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)
    real_glob = type(plots_dir).glob
    glob_calls: list[object] = []

    def tracking_glob(
        self: Path,
        pattern: str,
        *,
        case_sensitive: bool | None = None,
        recurse_symlinks: bool = False,
    ) -> Iterator[Path]:
        if self == plots_dir:
            glob_calls.append(
                (
                    pattern,
                    case_sensitive,
                    recurse_symlinks,
                )
            )
        return real_glob(
            self,
            pattern,
            case_sensitive=case_sensitive,
            recurse_symlinks=recurse_symlinks,
        )

    monkeypatch.setattr(type(plots_dir), "glob", tracking_glob)

    # No profiler bound → glob MUST NOT be called on plots_dir.
    cli._stage_plot(cfg)
    assert glob_calls == [], (
        "_stage_plot called plots_dir.glob() with no profiler bound — "
        "this is wasted I/O on the production path"
    )


def test_score_emits_scoring_cache_hits_counter(
    cfg: AxBenchBpbAdvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#511: when a profiler is bound, `_stage_score` emits the dedup counters
    `scoring_cache_hits` + `scored_unique`, read from the stats dict the score
    stage threads through `run_model_backed_bpb_adv_score_stage`."""
    fake_candidates = _empty_table(CANDIDATES_SCHEMA, n_rows=4)
    monkeypatch.setattr(cli, "load_candidate_table", lambda _p: fake_candidates)
    fake_scores = _empty_table(SCORE_SCHEMA, n_rows=4)

    def _fake_score_stage(*, cfg, paths, candidates, stats=None):  # noqa: ANN001, ARG001
        if stats is not None:
            stats["candidates_total"] = 4
            stats["scored_unique"] = 1
            stats["cache_hits"] = 3
        return fake_scores

    monkeypatch.setattr(cli, "run_model_backed_bpb_adv_score_stage", _fake_score_stage)
    monkeypatch.setattr(cli.pq, "write_table", lambda _t, _p: None)

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={}):
        cli._stage_score(cfg)

    counters = _read_counters(profile_dir)
    assert counters["axbench_bpb_adv.score.scoring_cache_hits"] == [3]
    assert counters["axbench_bpb_adv.score.scored_unique"] == [1]


def test_every_stage_handler_is_decorated_with_profiled() -> None:
    """Static contract: every ``_stage_*`` function defined in the CLI module
    must carry an ``@_profiled(...)`` decorator.

    This is structural enforcement — a developer who adds a 9th stage and
    forgets the decorator gets caught here. Uses ``ast`` (not regex) so the
    check is quote-agnostic + immune to formatting/decorator-stacking changes.
    """
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    stage_defs: set[str] = set()
    decorated_defs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("_stage_"):
            continue
        stage_defs.add(node.name)
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call):
                func = deco.func
                if isinstance(func, ast.Name) and func.id == "_profiled":
                    decorated_defs.add(node.name)
                    break
    missing = stage_defs - decorated_defs
    assert not missing, f"stage handlers missing @_profiled decorator: {sorted(missing)}"
