from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import scripts.benchmark.benchmark_autointerp_token_examples as benchmark_module
import steering.eval.autointerp_filter.cache as cache_module
import steering.eval.autointerp_filter.source_index as source_index_module
import steering.eval.autointerp_filter.token_examples as token_examples_module
import steering.eval.autointerp_filter.types as types_module
import tests.steering.autointerp_filter._batch_acceptance as batch_acceptance_module
from scripts.benchmark.benchmark_autointerp_token_examples import (
    REQUIRED_REPORT_FIELDS,
    PhaseRecorder,
    TokenizerTimingProxy,
    build_anchor_profile_observations,
    build_cli_overrides,
    build_report,
    build_steady_state_measurement,
    compose_benchmark_config,
    install_instrumentation,
    install_model_instrumentation,
    parse_args,
    positive_ref_counts,
    run_benchmark,
    select_benchmark_targets,
)
from steering.eval.autointerp_filter.batch_validation import (
    PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
    BatchToleranceProfile,
    NumericTolerance,
)
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    SourceIndexConfig,
    TokenExamplesConfig,
)
from steering.eval.autointerp_filter.parity import (
    build_trial_surfaces,
    compare_rendered_trial_surfaces,
    compare_token_example_surface,
    compare_trial_surfaces,
    load_token_example_set_by_target,
)
from steering.eval.autointerp_filter.token_examples import CachedTokenExampleProvider
from steering.eval.autointerp_filter.types import (
    AutoInterpBatchValidationError,
    AutoInterpCacheError,
    AutoInterpDataError,
    ScoringTarget,
    TargetKind,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.feature_io import TensorKey, save_features
from steering.refined_types import PositiveInt
from tests.steering.autointerp_filter._batch_acceptance import (
    ValidationTargetPlan,
    evaluate_batch_performance_replicates,
    feature_activation_counts,
    select_validation_target_plan,
)

_PHASE_NAMES: frozenset[str] = frozenset(
    {
        "prepare",
        "model_load",
        "source_index_load",
        "sqlite_row_load",
        "sqlite_apply",
        "parquet_text_load",
        "tokenization",
        "token_decode",
        "model_forward",
        "sae_projection",
        "window_selection",
        "doc_activations",
        "stable_key_hash_window",
        "stable_key_hash_docref",
        "token_cache_open",
        "token_cache_read",
        "token_cache_write",
        "examples",
        "trial_build",
    }
)

MODULE_SEAMS = [
    (token_examples_module, "load_or_build_source_index"),
    (source_index_module, "_load_index"),
    (token_examples_module, "_shard_row_group_offsets"),
    (token_examples_module, "_load_shard_row_group"),
    (token_examples_module, "_truncate_to_training_window"),
    (token_examples_module, "_truncate_to_focused_window"),
    (token_examples_module, "_get_activations_native"),
    (token_examples_module, "_project_selected_sae_activations"),
]

_ISSUE_REQUIRED = {
    "source_index_load_elapsed_s",
    "sqlite_apply_elapsed_s",
    "parquet_text_load_elapsed_s",
    "tokenization_elapsed_s",
    "model_forward_elapsed_s",
    "sae_projection_elapsed_s",
    "window_selection_elapsed_s",
    "stable_key_hash_elapsed_s",
    "token_cache_write_elapsed_s",
    "trial_build_elapsed_s",
    "ru_maxrss_kib",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
}


def _scoring_target(feature_id: int) -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        target_key=f"feature:{feature_id}",
        display_name=f"feature_{feature_id}",
    )


def test_steady_state_measurement_excludes_same_first_two_blocks_for_serial_and_batch() -> None:
    targets = [_scoring_target(0), _scoring_target(128), _scoring_target(256)]
    per_target = [
        {
            "target_key": target.target_key,
            "phase_deltas": {"examples": {"inclusive_s": elapsed}},
        }
        for target, elapsed in zip(targets, (10.0, 20.0, 0.3), strict=True)
    ]
    serial = build_steady_state_measurement(
        targets=targets,
        materialization_batch_size=1,
        per_target=per_target,
        batch_pair_timings=[],
    )
    batched = build_steady_state_measurement(
        targets=targets,
        materialization_batch_size=128,
        per_target=[],
        batch_pair_timings=[
            {
                "block_ids": (0, 1),
                "target_keys": ("feature:0", "feature:128"),
                "elapsed_ns": 30_000_000_000,
                "scope": "complete_bulk_partition_v1",
            },
            {
                "block_ids": (2,),
                "target_keys": ("feature:256",),
                "elapsed_ns": 400_000_000,
                "scope": "complete_bulk_partition_v1",
            },
        ],
    )

    assert (
        serial["warmup_target_keys"]
        == batched["warmup_target_keys"]
        == [
            "feature:0",
            "feature:128",
        ]
    )
    assert serial["target_count"] == batched["target_count"] == 1
    assert serial["elapsed_s"] == pytest.approx(0.3)
    assert batched["elapsed_s"] == pytest.approx(0.4)
    assert serial["available"] is batched["available"] is True


def test_steady_state_measurement_rejects_missing_rows_and_noncanonical_pair_timings() -> None:
    targets = [_scoring_target(0), _scoring_target(128), _scoring_target(256)]
    with pytest.raises(AutoInterpDataError, match="one timing row per target"):
        build_steady_state_measurement(
            targets=targets,
            materialization_batch_size=1,
            per_target=[],
            batch_pair_timings=[],
        )
    with pytest.raises(AutoInterpDataError, match="timing count"):
        build_steady_state_measurement(
            targets=targets,
            materialization_batch_size=128,
            per_target=[],
            batch_pair_timings=[],
        )
    with pytest.raises(AutoInterpDataError, match="canonical target plan"):
        build_steady_state_measurement(
            targets=targets,
            materialization_batch_size=128,
            per_target=[],
            batch_pair_timings=[
                {
                    "block_ids": (0, 1),
                    "target_keys": ("feature:128", "feature:0"),
                    "elapsed_ns": 1,
                    "scope": "complete_bulk_partition_v1",
                },
                {
                    "block_ids": (2,),
                    "target_keys": ("feature:256",),
                    "elapsed_ns": 1,
                    "scope": "complete_bulk_partition_v1",
                },
            ],
        )
    with pytest.raises(AutoInterpDataError, match="timing 0 is malformed"):
        build_steady_state_measurement(
            targets=targets,
            materialization_batch_size=128,
            per_target=[],
            batch_pair_timings=[
                {
                    "block_ids": (0, 1),
                    "target_keys": ["feature:0", "feature:128"],
                    "elapsed_ns": 0,
                    "scope": "complete_bulk_partition_v1",
                },
                {
                    "block_ids": (2,),
                    "target_keys": ("feature:256",),
                    "elapsed_ns": 1,
                    "scope": "complete_bulk_partition_v1",
                },
            ],
        )


def _mk_ref(doc_id: int = 1, activations=(0.0, 5.0)) -> TokenWindowRef:
    token_ids = tuple(100 + i for i in range(len(activations)))
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=doc_id,
        source_text_hash=f"h{doc_id}",
        token_start=0,
        token_end=len(token_ids),
        token_ids=token_ids,
        token_strings=tuple(f"t{i}" for i in token_ids),
        target_feature_ids=(0,),
        activations=tuple(float(a) for a in activations),
        max_activation=float(max(activations)),
        quantile=None,
        active=max(activations) > 0,
    )


def _write_cache(path: Path, rows: list[tuple[str, str, TokenExampleSet]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE token_example_sets (namespace TEXT, target_key TEXT, payload TEXT)")
    for namespace, target_key, examples in rows:
        conn.execute(
            "INSERT INTO token_example_sets VALUES (?, ?, ?)",
            (namespace, target_key, json.dumps(examples.to_json())),
        )
    conn.commit()
    conn.close()


def test_surface_report_to_json_roundtrips_plain_dicts() -> None:
    examples = TokenExampleSet(positives=(_mk_ref(),), non_activating=(_mk_ref(2, (0.0, 0.0)),))
    report = compare_token_example_surface(examples, examples)
    payload = report.to_json()
    assert payload["positives"]["identity_set_equal"] is True
    assert payload["positives"]["same_identity_order_fraction"] == 1.0
    json.dumps(payload)  # must be JSON-serializable as-is


def _anchor_profile_report() -> tuple[dict, BatchToleranceProfile]:
    target = ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(0,),
        target_key="feature:0",
        display_name="feature_0",
    )
    examples = _many_windows()
    profile = BatchToleranceProfile(
        name="anchor-observation-test",
        batch_size=128,
        selected_sae=NumericTolerance(max_abs=0.1, p99_abs=0.1, mean_abs=0.1),
        raw_residual=NumericTolerance(max_abs=0.1, p99_abs=0.1, mean_abs=0.1),
        calibration_contract=PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
        max_support_mismatches=0,
    )
    target_report = {
        "examples": compare_token_example_surface(examples, examples).to_json(),
        "trials": compare_trial_surfaces(
            examples,
            examples,
            target=target,
        ).to_json(),
        "rendered_prompts": compare_rendered_trial_surfaces(
            examples,
            examples,
            target=target,
        ).to_json(),
    }
    observations = [
        {
            "target_key": "feature:filler",
            "surface": "selected_sae_startup",
            "surface_kind": "selected_sae",
            "context": "excluded filler startup",
            "require_support_bound": True,
            "count": 2,
            "support_mismatches": 0,
            "max_abs": 0.09,
            "p99_abs": 0.09,
            "mean_abs": 0.09,
        },
        {
            "target_key": target.target_key,
            "surface": "selected_sae_periodic_shadow",
            "surface_kind": "selected_sae",
            "context": "anchor selected",
            "require_support_bound": True,
            "count": 2,
            "support_mismatches": 0,
            "max_abs": 0.0,
            "p99_abs": 0.0,
            "mean_abs": 0.0,
        },
        {
            "target_key": target.target_key,
            "surface": "raw_residual_calibration_shadow",
            "surface_kind": "raw_residual",
            "context": "anchor residual",
            "require_support_bound": False,
            "count": 4,
            "support_mismatches": 0,
            "max_abs": 0.0,
            "p99_abs": 0.0,
            "mean_abs": 0.0,
        },
    ]
    return (
        {
            "batch_validation": {"observations": observations},
            "parity": {"per_target": {target.target_key: target_report}},
        },
        profile,
    )


def test_anchor_profile_observations_exclude_fillers_and_include_output_metrics() -> None:
    report, profile = _anchor_profile_report()

    observations = build_anchor_profile_observations(report, profile=profile)

    surfaces = [observation["surface"] for observation in observations]
    assert "selected_sae_startup" not in surfaces
    assert surfaces[:2] == [
        "selected_sae_periodic_shadow",
        "raw_residual_calibration_shadow",
    ]
    assert set(surfaces[2:]) == {
        "selected_sae_output_positives_activation",
        "selected_sae_output_positives_max_activation",
        "selected_sae_output_non_activating_activation",
        "selected_sae_output_non_activating_max_activation",
    }
    assert {observation["target_key"] for observation in observations} == {"feature:0"}
    assert all(observation["support_mismatches"] == 0 for observation in observations)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (
            lambda report: report["parity"]["per_target"]["feature:0"][
                "rendered_prompts"
            ].__setitem__("all_message_bytes_exact", False),
            "changed trial or rendered prompt bytes",
        ),
        (
            lambda report: report["parity"]["per_target"]["feature:0"]["examples"][
                "positives"
            ].__setitem__("identity_set_equal", False),
            "changed the exact positives window surface",
        ),
        (
            lambda report: report["parity"]["per_target"]["feature:0"]["examples"]["positives"][
                "activation"
            ].update({"max_abs": 0.2, "p99_abs": 0.2, "mean_abs": 0.2}),
            "numeric validation failed",
        ),
        (
            lambda report: report["batch_validation"]["observations"].pop(),
            "both selected-SAE and raw-residual",
        ),
        (
            lambda report: report["batch_validation"]["observations"][2].__setitem__(
                "target_key", "feature:other"
            ),
            "raw_residual target coverage mismatch",
        ),
    ],
)
def test_anchor_profile_observations_fail_loud_on_every_contract_surface(
    mutate,
    fragment: str,
) -> None:
    report, profile = _anchor_profile_report()
    mutate(report)
    with pytest.raises(
        (AutoInterpDataError, AutoInterpBatchValidationError),
        match=fragment,
    ):
        build_anchor_profile_observations(report, profile=profile)


def test_anchor_profile_observation_schema_rejects_missing_and_malformed_nodes() -> None:
    report, profile = _anchor_profile_report()
    malformed = json.loads(json.dumps(report))
    malformed.pop("parity")
    with pytest.raises(AutoInterpDataError, match="require batch-validation and parity"):
        build_anchor_profile_observations(malformed, profile=profile)

    malformed = json.loads(json.dumps(report))
    malformed["batch_validation"]["observations"] = {}
    with pytest.raises(AutoInterpDataError, match="surfaces are malformed"):
        build_anchor_profile_observations(malformed, profile=profile)

    malformed = json.loads(json.dumps(report))
    malformed["parity"]["per_target"]["feature:0"] = []
    with pytest.raises(AutoInterpDataError, match="parity report.*malformed"):
        build_anchor_profile_observations(malformed, profile=profile)

    malformed = json.loads(json.dumps(report))
    malformed["parity"]["per_target"]["feature:0"].pop("examples")
    with pytest.raises(AutoInterpDataError, match="parity surfaces.*incomplete"):
        build_anchor_profile_observations(malformed, profile=profile)

    malformed = json.loads(json.dumps(report))
    malformed["parity"]["per_target"]["feature:0"]["examples"]["positives"] = []
    with pytest.raises(AutoInterpDataError, match="missing positives parity"):
        build_anchor_profile_observations(malformed, profile=profile)

    malformed = json.loads(json.dumps(report))
    malformed["parity"]["per_target"]["feature:0"]["examples"]["positives"]["activation"] = []
    with pytest.raises(AutoInterpDataError, match="missing positives/activation metrics"):
        build_anchor_profile_observations(malformed, profile=profile)


def test_anchor_profile_observations_handle_empty_metrics_and_reject_corrupt_empty_or_count() -> (
    None
):
    report, profile = _anchor_profile_report()
    group = report["parity"]["per_target"]["feature:0"]["examples"]["non_activating"]
    group.update(
        {
            "reference_count": 0,
            "candidate_count": 0,
            "matched_window_count": 0,
            "same_identity_order_count": 0,
            "quantile_exact_count": 0,
            "highlight_exact_count": 0,
        }
    )
    empty_metric = {
        "reference_values": 0,
        "candidate_values": 0,
        "support_mismatches": 0,
        "max_abs": 0.0,
        "p99_abs": 0.0,
        "mean_abs": 0.0,
        "within_abs_tolerance": True,
    }
    group["activation"] = dict(empty_metric)
    group["max_activation"] = dict(empty_metric)

    observations = build_anchor_profile_observations(report, profile=profile)
    assert not any("non_activating" in row["surface"] for row in observations)

    corrupt = json.loads(json.dumps(report))
    corrupt["parity"]["per_target"]["feature:0"]["examples"]["non_activating"]["activation"][
        "max_abs"
    ] = 0.1
    with pytest.raises(AutoInterpDataError, match="empty anchor metric.*nonzero"):
        build_anchor_profile_observations(corrupt, profile=profile)

    report, profile = _anchor_profile_report()
    report["parity"]["per_target"]["feature:0"]["examples"]["positives"]["activation"][
        "candidate_values"
    ] -= 1
    with pytest.raises(AutoInterpDataError, match="changed value count"):
        build_anchor_profile_observations(report, profile=profile)


def test_surface_comparison_rejects_duplicate_source_window_identity() -> None:
    first = _mk_ref()
    duplicate = replace(
        first,
        activations=(0.0, 6.0),
        max_activation=6.0,
    )
    examples = TokenExampleSet(
        positives=(first, duplicate),
        non_activating=(),
    )
    with pytest.raises(AutoInterpCacheError, match="duplicate token-window identity"):
        compare_token_example_surface(examples, examples)


def test_load_by_target_rejects_missing_and_ambiguous(tmp_path: Path) -> None:
    examples = TokenExampleSet(positives=(_mk_ref(),), non_activating=())
    path = tmp_path / "cache.sqlite"
    _write_cache(path, [("ns1", "feature:0", examples), ("ns2", "feature:0", examples)])
    with pytest.raises(AutoInterpCacheError, match="2 namespaces"):
        load_token_example_set_by_target(path, "feature:0")
    with pytest.raises(AutoInterpCacheError, match="no token_example_sets row"):
        load_token_example_set_by_target(path, "feature:1")


def test_load_by_target_returns_set(tmp_path: Path) -> None:
    examples = TokenExampleSet(positives=(_mk_ref(),), non_activating=())
    path = tmp_path / "cache.sqlite"
    _write_cache(path, [("ns1", "feature:0", examples)])
    assert load_token_example_set_by_target(path, "feature:0") == examples


def _many_windows() -> TokenExampleSet:
    positives = tuple(_mk_ref(doc_id=i, activations=(0.0, float(20 - i))) for i in range(1, 7))
    non_activating = tuple(_mk_ref(doc_id=100 + i, activations=(0.0, 0.0)) for i in range(6))
    return TokenExampleSet(positives=positives, non_activating=non_activating)


def test_trial_surfaces_depend_on_target_and_default_to_legacy_probe() -> None:
    """Pins the D1 fix: trial surfaces are target-dependent, and the default
    arguments reproduce the legacy feature:0 probe exactly."""
    examples = _many_windows()
    legacy = build_trial_surfaces(examples, n_positive_examples=2, n_trials=8)
    explicit_f0 = build_trial_surfaces(
        examples,
        n_positive_examples=2,
        n_trials=8,
        target=ScoringTarget(
            kind=TargetKind.feature,
            feature_ids=(0,),
            target_key="feature:0",
            display_name="feature:0",
        ),
    )
    other_target = build_trial_surfaces(
        examples,
        n_positive_examples=2,
        n_trials=8,
        target=ScoringTarget(
            kind=TargetKind.feature,
            feature_ids=(17,),
            target_key="feature:17",
            display_name="feature:17",
        ),
    )
    assert legacy == explicit_f0
    legacy_keys = [row[0] for row in legacy]
    other_keys = [row[0] for row in other_target]
    assert legacy_keys and other_keys
    assert set(legacy_keys) != set(other_keys)


def test_phase_recorder_nested_exclusive_times() -> None:
    rec = PhaseRecorder()
    with rec.phase("outer"):
        with rec.phase("inner"):
            pass
    snap = rec.snapshot()
    outer_inc, outer_exc, outer_n = snap["outer"]
    inner_inc, inner_exc, inner_n = snap["inner"]
    assert outer_n == inner_n == 1
    assert inner_inc == inner_exc > 0
    assert outer_inc >= inner_inc
    assert outer_exc == outer_inc - inner_inc


def test_phase_recorder_exception_safe_and_reentrant() -> None:
    rec = PhaseRecorder()
    with pytest.raises(ValueError):
        with rec.phase("a"):
            with rec.phase("a"):
                raise ValueError("boom")
    snap = rec.snapshot()
    assert snap["a"][2] == 2  # both entries recorded despite the raise
    with rec.phase("a"):
        pass
    assert rec.snapshot()["a"][2] == 3


def test_phase_recorder_wrap_preserves_result_and_counts() -> None:
    rec = PhaseRecorder()
    original = lambda x, y=1: x + y  # noqa: E731 - identity-compared below
    wrapped = rec.wrap(original, "add")
    assert wrapped(2, y=3) == 5
    assert inspect.unwrap(wrapped) is original  # functools.wraps sets __wrapped__
    inclusive_ns, exclusive_ns, count = rec.snapshot()["add"]
    assert count == 1
    assert inclusive_ns == exclusive_ns > 0


def test_phase_recorder_sibling_children_exclusive_accounting() -> None:
    rec = PhaseRecorder()
    with rec.phase("outer"):
        with rec.phase("a"):
            pass
        with rec.phase("b"):
            with rec.phase("c"):
                pass
    snap = rec.snapshot()
    assert snap["outer"][1] == snap["outer"][0] - snap["a"][0] - snap["b"][0]
    assert snap["b"][1] == snap["b"][0] - snap["c"][0]


def test_phase_recorder_keeps_overlapping_thread_stacks_independent() -> None:
    rec = PhaseRecorder()
    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()
    errors: list[BaseException] = []

    def worker_a() -> None:
        try:
            with rec.phase("a_outer"):
                with rec.phase("a_child"):
                    a_entered.set()
                    assert b_entered.wait(timeout=5)
                a_exited.set()
        except BaseException as exc:
            errors.append(exc)

    def worker_b() -> None:
        try:
            assert a_entered.wait(timeout=5)
            with rec.phase("b_outer"):
                with rec.phase("b_child"):
                    b_entered.set()
                    assert a_exited.wait(timeout=5)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker_a), threading.Thread(target=worker_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    snap = rec.snapshot()
    assert snap["a_outer"][1] == snap["a_outer"][0] - snap["a_child"][0]
    assert snap["b_outer"][1] == snap["b_outer"][0] - snap["b_child"][0]


class _StubTokenizer:
    bos_token_id = 7

    def __call__(self, text: str, padding: bool = False):
        return {"input_ids": [len(text)]}

    def decode(self, ids):
        return f"d{ids[0]}"


def test_tokenizer_proxy_times_decode_only_and_passes_through() -> None:
    rec = PhaseRecorder()
    proxy = TokenizerTimingProxy(_StubTokenizer(), rec)
    assert proxy("abc") == {"input_ids": [3]}
    assert proxy.decode([5]) == "d5"
    assert proxy.bos_token_id == 7
    snap = rec.snapshot()
    assert snap["token_decode"][2] == 1
    assert set(snap) == {"token_decode"}


def test_select_benchmark_targets_is_deterministic_and_stratified() -> None:
    counts = [(fid, (fid % 5) + 1) for fid in range(30)]
    first = select_benchmark_targets(counts, 9)
    assert first == select_benchmark_targets(counts, 9)
    assert 0 in first and len(first) == 9 and list(first) == sorted(first)
    low, mid, high = (
        [f for f in first if f < 10],
        [f for f in first if 10 <= f < 20],
        [f for f in first if f >= 20],
    )
    assert low and mid and high  # every id-tercile represented


def test_select_benchmark_targets_validates_inputs() -> None:
    with pytest.raises(AutoInterpDataError, match="include"):
        select_benchmark_targets([(1, 2)], 1, include=(0,))
    with pytest.raises(AutoInterpDataError, match="n="):
        select_benchmark_targets([(0, 1), (1, 1)], 3)
    with pytest.raises(AutoInterpDataError, match="exhausted cells"):
        select_benchmark_targets([(0, 1), (0, 2)], 2)
    assert select_benchmark_targets([(0, 1), (1, 0)], 1) == (0,)


def test_positive_ref_counts_reads_index(tmp_path: Path) -> None:
    path = tmp_path / "idx.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE positive_refs (feature_id INTEGER, rank INTEGER, magnitude REAL,"
        " stable_key TEXT, shard_stem TEXT, shard_index INTEGER, doc_id INTEGER,"
        " segment_token_start INTEGER, segment_token_end INTEGER)"
    )
    conn.executemany(
        "INSERT INTO positive_refs VALUES (?, ?, 1.0, 'k', 's', 0, 0, 0, 1)",
        [(0, 0), (0, 1), (2, 0)],
    )
    conn.commit()
    conn.close()
    assert positive_ref_counts(path) == [(0, 2), (2, 1)]


def test_feature_activation_counts_loads_canonical_stats_and_sorts_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "feature_statistics.safetensors"
    monkeypatch.setattr(batch_acceptance_module, "find_statistics_path", lambda _path: stats_path)
    monkeypatch.setattr(
        batch_acceptance_module,
        "load_statistics",
        lambda path: SimpleNamespace(
            per_feature_stats={
                "9": SimpleNamespace(count=81),
                "2": SimpleNamespace(count=7),
            }
        ),
    )

    assert feature_activation_counts(tmp_path) == [(2, 7), (9, 81)]


def test_feature_activation_counts_fails_loud_without_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(batch_acceptance_module, "find_statistics_path", lambda _path: None)
    with pytest.raises(AutoInterpDataError, match="feature_statistics.safetensors"):
        feature_activation_counts(tmp_path)


def test_feature_activation_counts_rejects_non_integer_stats_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        batch_acceptance_module,
        "find_statistics_path",
        lambda _path: tmp_path / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        batch_acceptance_module,
        "load_statistics",
        lambda _path: SimpleNamespace(per_feature_stats={"not-an-id": SimpleNamespace(count=7)}),
    )
    with pytest.raises(AutoInterpDataError, match="non-integer feature ID"):
        feature_activation_counts(tmp_path)


def test_all_patch_seams_exist() -> None:
    for module, attr in MODULE_SEAMS:
        assert callable(getattr(module, attr)), (module.__name__, attr)
    assert callable(cache_module.TokenExampleCache.get)
    assert callable(cache_module.TokenExampleCache.set)
    assert callable(types_module.TokenWindowRef.stable_key)
    assert callable(types_module.SourceDocRef.stable_key)
    for attr in (
        "_apply_source_index",
        "_positive_windows_for_doc",
        "_non_activating_windows_for_doc",
        "_doc_activations",
    ):
        assert callable(getattr(CachedTokenExampleProvider, attr)), attr


def _bare_provider(tmp_path: Path) -> CachedTokenExampleProvider:
    cfg = AutoInterpFilterConfig(
        feature_dir=str(tmp_path / "features"),
        output_dir=str(tmp_path / "out"),
        token_example_cache_path=str(tmp_path / "out" / "token_example_cache.sqlite"),
        token_examples=TokenExamplesConfig(),
        source_index=SourceIndexConfig(),
    )
    return CachedTokenExampleProvider(cfg, feature_dir=tmp_path / "features", n_sae_features=3)


def test_install_and_restore_round_trips_all_seams(tmp_path: Path) -> None:
    provider = _bare_provider(tmp_path)
    originals = {(m.__name__, a): getattr(m, a) for m, a in MODULE_SEAMS}
    orig_cache = (cache_module.TokenExampleCache.get, cache_module.TokenExampleCache.set)
    orig_keys = (types_module.TokenWindowRef.stable_key, types_module.SourceDocRef.stable_key)
    handle = install_instrumentation(
        provider,
        recorder=PhaseRecorder(),
        tier=2,
        cuda_sync=False,
    )
    for module, attr in MODULE_SEAMS:
        assert getattr(module, attr) is not originals[(module.__name__, attr)]
    assert "_doc_activations" in provider.__dict__
    handle.restore()
    for module, attr in MODULE_SEAMS:
        assert getattr(module, attr) is originals[(module.__name__, attr)]
    assert (cache_module.TokenExampleCache.get, cache_module.TokenExampleCache.set) == orig_cache
    assert (
        types_module.TokenWindowRef.stable_key,
        types_module.SourceDocRef.stable_key,
    ) == orig_keys
    assert "_doc_activations" not in provider.__dict__


def test_install_rolls_back_on_mid_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _bare_provider(tmp_path)
    originals = {(m.__name__, a): getattr(m, a) for m, a in MODULE_SEAMS}
    orig_cache_init = cache_module.TokenExampleCache.__init__
    monkeypatch.delattr(types_module.TokenWindowRef, "stable_key")
    with pytest.raises(AttributeError):
        install_instrumentation(
            provider,
            recorder=PhaseRecorder(),
            tier=2,
            cuda_sync=False,
        )
    for module, attr in MODULE_SEAMS:
        assert getattr(module, attr) is originals[(module.__name__, attr)], attr
    assert cache_module.TokenExampleCache.__init__ is orig_cache_init
    assert "_doc_activations" not in provider.__dict__


def test_report_contains_every_issue_required_field() -> None:
    assert _ISSUE_REQUIRED <= REQUIRED_REPORT_FIELDS
    rec = PhaseRecorder()
    with rec.phase("examples"):
        pass
    report = build_report(
        recorder=rec,
        per_target=[],
        meta={
            "benchmark": {"name": "x"},
            "git": {},
            "environment": {},
            "config": {},
            "targets": [],
        },
        preconditions={},
        parity=None,
        instrumentation={"tier": 1},
        cuda_device=None,
    )
    assert REQUIRED_REPORT_FIELDS <= set(report)
    assert report["examples_elapsed_s"] >= 0.0
    assert report["phases"]["examples"]["count"] == 1
    assert report["cuda_max_memory_allocated_bytes"] is None
    assert report["cuda_max_memory_reserved_bytes"] is None
    json.dumps(report, sort_keys=True)


def test_build_report_rejects_missing_meta() -> None:
    with pytest.raises(AutoInterpDataError, match="missing fields"):
        build_report(
            recorder=PhaseRecorder(),
            per_target=[],
            meta={"benchmark": {"name": "x"}},
            preconditions={},
            parity=None,
            instrumentation={"tier": 1},
            cuda_device=None,
        )


def test_build_report_reads_peaks_from_exact_configured_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = t.device("cuda:7")
    seen: list[tuple[str, t.device]] = []

    def _allocated(actual: t.device) -> int:
        seen.append(("allocated", actual))
        return 123

    def _reserved(actual: t.device) -> int:
        seen.append(("reserved", actual))
        return 456

    monkeypatch.setattr(t.cuda, "max_memory_allocated", _allocated)
    monkeypatch.setattr(t.cuda, "max_memory_reserved", _reserved)
    report = build_report(
        recorder=PhaseRecorder(),
        per_target=[],
        meta={
            "benchmark": {},
            "git": {},
            "environment": {},
            "config": {},
            "targets": [],
        },
        preconditions={},
        parity=None,
        instrumentation={},
        cuda_device=device,
    )

    assert seen == [("allocated", device), ("reserved", device)]
    assert report["cuda_max_memory_allocated_bytes"] == 123
    assert report["cuda_max_memory_reserved_bytes"] == 456


def test_compute_process_count_is_conservative_when_nvidia_smi_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1, "stdout": "9\n"})(),
    )
    assert benchmark_module._nvidia_compute_process_count() == 0

    def _missing(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr(benchmark_module.subprocess, "run", _missing)
    assert benchmark_module._nvidia_compute_process_count() == 0


class _FakeSae(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(
            t.tensor([[0.6, -0.2, 0.1], [0.1, 0.5, -0.3], [-0.4, 0.2, 0.7]]),
            requires_grad=False,
        )
        self.b_enc = t.nn.Parameter(
            t.tensor([0.05, -0.1, 0.2]),
            requires_grad=False,
        )
        self.threshold = None

    def process_sae_in(self, acts: t.Tensor) -> t.Tensor:
        return acts

    def activation_fn(self, hidden_pre: t.Tensor) -> t.Tensor:
        return t.relu(hidden_pre)


class _FakeTLModel:
    def parameters(self):
        yield t.zeros(1)

    def run_with_cache(self, batch, *, names_filter, stop_at_layer, return_type):
        tokens = batch.squeeze(0)
        pos = t.arange(tokens.shape[0], dtype=t.float32).unsqueeze(1)
        acts = (
            (tokens.float().unsqueeze(1) * t.tensor([[1.0, 2.0, 3.0]])) % 7.0
        ) * 0.1 + pos * 0.01
        return None, {names_filter[0]: acts.unsqueeze(0)}


class _FakeTokenizer:
    def __call__(self, text: str, padding: bool = False):
        return {"input_ids": [(3 + (b % 11)) for b in text.encode("utf-8")][:16]}

    def decode(self, ids):
        return f"t{ids[0]}»"


def _fake_model_hook(provider) -> None:
    provider._sae = _FakeSae()
    provider._model = _FakeTLModel()
    provider._native_model = None
    provider._tokenizer = _FakeTokenizer()
    provider._layer = 0
    provider._activation_type = None
    provider._hook_name = "blocks.0.hook_resid_pre"
    provider._context_window = 64
    provider._sae_max_context = 64
    provider._activation_backend = "transformerlens"
    provider._ignore_bos_token_id = 999


def _write_synthetic_env(tmp_path: Path) -> tuple[Path, Path]:
    """Write the shared shard/parquet fixture exactly once per tmp_path."""
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    if feature_dir.exists():
        return feature_dir, data_dir
    feature_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    save_features(
        {
            TensorKey.OFFSETS: t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
            TensorKey.FEATURE_IDS: t.tensor([0, 0, 1, 2], dtype=t.int64),
            TensorKey.FEATURE_MAGS: t.tensor([10.0, 8.0, 3.0, 2.0]),
            TensorKey.DOC_IDS: t.tensor([0, 1, 2, 3], dtype=t.int64),
            TensorKey.SEGMENT_INDICES: t.tensor([0, 0, 0, 0], dtype=t.int64),
            TensorKey.SEGMENT_POSITIONS: t.tensor([0, 0, 0, 0], dtype=t.int64),
            TensorKey.SEGMENT_CHAR_STARTS: t.tensor([0, 0, 0, 0], dtype=t.int64),
            TensorKey.SEGMENT_CHAR_ENDS: t.tensor([10, 10, 10, 10], dtype=t.int64),
            "n_docs": 4,
            "n_segments": 4,
            "layer": 0,
            "context_window": 64,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 16,
            "context_independent_features": "false",
            "ignore_bos_features": "true",
        },
        feature_dir / "shard_00000_features.safetensors.zst",
    )
    pq.write_table(
        pa.table({"text": ["alpha doc zero", "bravo doc one", "charlie two", "delta three"]}),
        data_dir / "shard_00000.parquet",
        row_group_size=1,
    )
    return feature_dir, data_dir


def _synthetic_cfg(tmp_path: Path, tag: str) -> AutoInterpFilterConfig:
    feature_dir, data_dir = _write_synthetic_env(tmp_path)
    out_dir = tmp_path / f"out_{tag}"
    return AutoInterpFilterConfig(
        feature_dir=str(feature_dir),
        output_dir=str(out_dir),
        data_dir=str(data_dir),
        n_sae_features=PositiveInt.parse(3),
        feature_ids=[0, 1],
        token_example_cache_path=str(out_dir / "token_example_cache.sqlite"),
        token_examples=TokenExamplesConfig(
            max_windows_per_source_doc=PositiveInt.parse(2),
            materialization_batch_size=PositiveInt.parse(1),
        ),
        source_index=SourceIndexConfig(enabled=False),
    )


def test_run_benchmark_synthetic_phases_and_schema(tmp_path: Path) -> None:
    report = run_benchmark(
        _synthetic_cfg(tmp_path, "a"),
        [0],
        tier=2,
        post_prepare_hook=_fake_model_hook,
    )
    assert REQUIRED_REPORT_FIELDS <= set(report)
    phases = report["phases"]
    for name in (
        "prepare",
        "examples",
        "trial_build",
        "tokenization",
        "doc_activations",
        "window_selection",
        "sae_projection",
        "model_forward",
        "token_decode",
        "token_cache_write",
        "stable_key_hash_window",
        "parquet_text_load",
    ):
        assert phases.get(name, {}).get("count", 0) > 0, name
    row = report["per_target"][0]
    assert row["target_key"] == "feature:0"
    assert row["positive_windows"] > 0
    assert len(row["payload_sha256"]) == 64
    assert set(report["phases"]) <= _PHASE_NAMES


def test_run_benchmark_batch_branch_uses_one_bulk_materialization_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_cfg = _synthetic_cfg(tmp_path, "batch-branch")
    cfg = replace(
        base_cfg,
        token_examples=replace(
            base_cfg.token_examples,
            materialization_batch_size=128,
        ),
    )
    base_examples = _many_windows()

    def _retarget(target: ScoringTarget) -> TokenExampleSet:
        return TokenExampleSet(
            positives=tuple(
                replace(ref, target_feature_ids=target.feature_ids)
                for ref in base_examples.positives
            ),
            non_activating=tuple(
                replace(ref, target_feature_ids=target.feature_ids)
                for ref in base_examples.non_activating
            ),
        )

    def _bulk(
        self: CachedTokenExampleProvider,
        targets: list[ScoringTarget],
    ) -> dict[str, TokenExampleSet]:
        assert self._batch_pair_timing_sink is not None
        self._batch_pair_timing_sink.append(
            {
                "block_ids": (0,),
                "target_keys": tuple(target.target_key for target in targets),
                "elapsed_ns": 123,
                "scope": "complete_bulk_partition_v1",
            }
        )
        return {target.target_key: _retarget(target) for target in targets}

    monkeypatch.setattr(CachedTokenExampleProvider, "examples_for_targets", _bulk)
    monkeypatch.setattr(
        token_examples_module,
        "runtime_numeric_fingerprint",
        lambda _device=None: {"synthetic": True},
    )

    report = run_benchmark(
        cfg,
        [0, 1],
        tier=1,
        post_prepare_hook=_fake_model_hook,
        instrument=False,
    )

    assert [row["target_key"] for row in report["per_target"]] == [
        "feature:0",
        "feature:1",
    ]
    bulk_deltas = [row["bulk_materialization_phase_deltas"] for row in report["per_target"]]
    assert bulk_deltas[0] == bulk_deltas[1]
    assert bulk_deltas[0]["examples"]["count"] == 1
    assert report["benchmark"]["batch_pair_timings"] == [
        {
            "block_ids": (0,),
            "target_keys": ("feature:0", "feature:1"),
            "elapsed_ns": 123,
            "scope": "complete_bulk_partition_v1",
        }
    ]
    assert report["benchmark"]["steady_state"]["available"] is False


def test_run_benchmark_is_transparent_and_deterministic(tmp_path: Path) -> None:
    instrumented = run_benchmark(
        _synthetic_cfg(tmp_path, "i"),
        [0],
        tier=2,
        post_prepare_hook=_fake_model_hook,
    )
    raw = run_benchmark(
        _synthetic_cfg(tmp_path, "r"),
        [0],
        tier=2,
        post_prepare_hook=_fake_model_hook,
        instrument=False,
    )
    again = run_benchmark(
        _synthetic_cfg(tmp_path, "d"),
        [0],
        tier=2,
        post_prepare_hook=_fake_model_hook,
    )
    keys = (
        "payload_sha256",
        "namespace",
        "trial_keys_sha256",
        "positive_windows",
        "non_activating_windows",
        "trials_built",
    )
    for key in keys:
        assert instrumented["per_target"][0][key] == raw["per_target"][0][key], key
        assert instrumented["per_target"][0][key] == again["per_target"][0][key], key


def test_run_benchmark_parity_against_reference(tmp_path: Path) -> None:
    ref_cfg = _synthetic_cfg(tmp_path, "ref")
    first = run_benchmark(ref_cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)
    report = run_benchmark(
        _synthetic_cfg(tmp_path, "cand"),
        [0],
        tier=1,
        post_prepare_hook=_fake_model_hook,
        reference_cache_path=Path(ref_cfg.token_example_cache_path),
    )
    parity = report["parity"]["per_target"]["feature:0"]
    assert parity["examples"]["positives"]["identity_set_equal"] is True
    assert parity["trials"]["same_trial_key_count"] == parity["trials"]["reference_count"]
    assert first["per_target"][0]["payload_sha256"] == report["per_target"][0]["payload_sha256"]


def test_payload_hash_matches_stored_cache_text(tmp_path: Path) -> None:
    cfg = _synthetic_cfg(tmp_path, "hash")
    report = run_benchmark(cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)
    with closing(sqlite3.connect(f"file:{cfg.token_example_cache_path}?mode=ro", uri=True)) as conn:
        stored = conn.execute(
            "SELECT payload FROM token_example_sets WHERE target_key = 'feature:0'"
        ).fetchone()[0]
    assert any(ord(ch) > 127 for ch in stored)
    stored_sha = hashlib.sha256(stored.encode("utf-8")).hexdigest()
    assert report["per_target"][0]["payload_sha256"] == stored_sha


def test_cpu_variant_never_touches_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(t.cuda, "synchronize", lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(t.cuda, "reset_peak_memory_stats", lambda *a, **k: calls.append("reset"))
    monkeypatch.setattr(t.cuda, "max_memory_allocated", lambda *a, **k: calls.append("alloc") or 0)
    monkeypatch.setattr(
        t.cuda, "max_memory_reserved", lambda *a, **k: calls.append("reserved") or 0
    )
    report = run_benchmark(
        _synthetic_cfg(tmp_path, "cpu"),
        [0],
        tier=2,
        post_prepare_hook=_fake_model_hook,
    )
    assert report["cuda_max_memory_allocated_bytes"] is None
    assert report["cuda_max_memory_reserved_bytes"] is None
    assert calls == []


def test_run_benchmark_rejects_preexisting_cache_rows(tmp_path: Path) -> None:
    cfg = _synthetic_cfg(tmp_path, "dup")
    run_benchmark(cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)
    with pytest.raises(AutoInterpDataError, match="already exists"):
        run_benchmark(cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)


def test_run_benchmark_rejects_rebuild_and_shared_cache_configs(tmp_path: Path) -> None:
    base = _synthetic_cfg(tmp_path, "guard")
    rebuild_cfg = replace(
        base,
        source_index=SourceIndexConfig(
            enabled=True,
            force_rebuild=True,
            path=str(tmp_path / "idx.sqlite"),
            manifest_path=str(tmp_path / "idx.manifest.json"),
        ),
    )
    with pytest.raises(AutoInterpDataError, match="force_rebuild"):
        run_benchmark(rebuild_cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)
    shared_cfg = replace(
        base,
        source_index=SourceIndexConfig(
            enabled=True,
            shared_cache_dir=str(tmp_path / "shared"),
            path=str(tmp_path / "idx.sqlite"),
            manifest_path=str(tmp_path / "idx.manifest.json"),
        ),
    )
    with pytest.raises(AutoInterpDataError, match="shared_cache_dir"):
        run_benchmark(shared_cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)
    missing_cfg = replace(
        base,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "missing-index.sqlite"),
            manifest_path=str(tmp_path / "missing-index.manifest.json"),
        ),
    )
    with pytest.raises(AutoInterpDataError, match="ALREADY-BUILT source index"):
        run_benchmark(missing_cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)


def test_parse_args_requires_exactly_one_target_mode(tmp_path: Path) -> None:
    base = ["--output-dir", str(tmp_path), "--report", str(tmp_path / "r.json")]
    with pytest.raises(SystemExit):
        parse_args(base)
    with pytest.raises(SystemExit):
        parse_args([*base, "--targets", "0", "--select-targets", "5"])
    args = parse_args([*base, "--targets", "0,17"])
    assert args.targets == [0, 17]


@pytest.mark.parametrize(
    ("mode", "tier", "instrument"),
    [
        ("off", 2, False),
        ("tier1", 1, True),
        ("tier2", 2, True),
    ],
)
def test_parse_args_exposes_headline_instrumentation_off_mode(
    tmp_path: Path,
    mode: str,
    tier: int,
    instrument: bool,
) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
            "--targets",
            "0",
            "--instrumentation",
            mode,
        ]
    )
    assert args.instrumentation == mode
    assert args.tier == tier
    assert args.instrument is instrument


@pytest.mark.parametrize(("value", "enabled"), [("on", True), ("off", False)])
def test_parse_args_exposes_benchmark_only_prefetch_toggle(
    tmp_path: Path,
    value: str,
    enabled: bool,
) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
            "--targets",
            "0",
            "--materialization-prefetch",
            value,
        ]
    )
    assert (args.materialization_prefetch == "on") is enabled


def test_parse_args_preserves_hidden_legacy_tier_alias(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
            "--targets",
            "0",
            "--tier",
            "1",
        ]
    )
    assert args.instrumentation == "tier1"
    assert args.tier == 1
    assert args.instrument is True


def test_parse_args_accepts_distinct_anchor_and_materialization_target_lists(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
            "--targets",
            "0,129",
            "--materialization-targets",
            "0,1,127,128,129,255",
        ]
    )
    assert args.targets == [0, 129]
    assert args.materialization_targets == [0, 1, 127, 128, 129, 255]


@pytest.mark.parametrize("value", ["", "0,", "0,nope", ",0"])
def test_parse_args_rejects_malformed_materialization_target_list(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--report",
                str(tmp_path / "report.json"),
                "--targets",
                "0",
                "--materialization-targets",
                value,
            ]
        )


def test_parse_args_rejects_legacy_tier_with_explicit_nondefault_instrumentation(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--report",
                str(tmp_path / "report.json"),
                "--targets",
                "0",
                "--instrumentation",
                "off",
                "--tier",
                "1",
            ]
        )


def test_build_cli_overrides_assembles_expected_strings(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            "/abs/out",
            "--report",
            "/abs/r.json",
            "--targets",
            "0",
            "--feature-dir",
            "/abs/features",
            "--data-dir",
            "/abs/data",
            "--source-index-path",
            "/abs/idx.sqlite",
            "--source-index-manifest-path",
            "/abs/idx.manifest.json",
            "--device",
            "cpu",
            "--activation-backend",
            "transformerlens",
            "--model-dtype",
            "float32",
            "--extra-override",
            "seed=43",
        ]
    )
    overrides = build_cli_overrides(args)
    assert overrides == [
        "output_dir=/abs/out",
        "neuronpedia.enabled=false",
        "feature_dir=/abs/features",
        "data_dir=/abs/data",
        "source_index.enabled=true",
        "source_index.path=/abs/idx.sqlite",
        "source_index.manifest_path=/abs/idx.manifest.json",
        "token_examples.device=cpu",
        "token_examples.activation_backend=transformerlens",
        "token_examples.model_dtype=float32",
        "seed=43",
    ]


def test_parse_args_rejects_partial_source_index_pair(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--report",
                str(tmp_path / "r.json"),
                "--targets",
                "0",
                "--source-index-path",
                "/abs/idx.sqlite",
            ]
        )


def test_parse_args_rejects_source_index_extra_overrides(tmp_path: Path) -> None:
    for override in (
        "source_index.force_rebuild=true",
        "source_index.shared_cache_dir=/x",
        "+source_index.force_rebuild=true",
        "++source_index.force_rebuild=true",
        "source_index={force_rebuild: true}",
        "~source_index",  # hydra deletion would flip the run onto the full-scan path
        "~source_index.force_rebuild",
        "output_dir=/elsewhere",  # last-wins duplicate would re-point the cache
        "token_example_cache_path=/elsewhere/cache.sqlite",
    ):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--output-dir",
                    str(tmp_path),
                    "--report",
                    str(tmp_path / "r.json"),
                    "--targets",
                    "0",
                    "--extra-override",
                    override,
                ]
            )


def test_run_benchmark_rejects_duplicate_target_ids(tmp_path: Path) -> None:
    with pytest.raises(AutoInterpDataError, match="duplicate benchmark target ids"):
        run_benchmark(
            _synthetic_cfg(tmp_path, "dupids"),
            [0, 0],
            tier=1,
            post_prepare_hook=_fake_model_hook,
        )


@pytest.mark.parametrize(
    ("materialization_ids", "fragment"),
    [
        ([0, 0], "duplicate benchmark materialization"),
        ([0, 999], "unknown benchmark materialization"),
        ([1], "anchor targets must be included"),
        ([], "at least one materialization target"),
    ],
)
def test_run_benchmark_validates_materialization_target_superset_before_work(
    tmp_path: Path,
    materialization_ids: list[int],
    fragment: str,
) -> None:
    cfg = _synthetic_cfg(tmp_path, f"mat-{len(materialization_ids)}-{sum(materialization_ids)}")
    with pytest.raises(AutoInterpDataError, match=fragment):
        run_benchmark(
            cfg,
            [0],
            materialization_target_ids=materialization_ids,
            tier=1,
            post_prepare_hook=_fake_model_hook,
        )
    assert not Path(cfg.token_example_cache_path).exists()


def test_run_benchmark_records_anchor_and_materialization_target_metadata_separately(
    tmp_path: Path,
) -> None:
    cfg = _synthetic_cfg(tmp_path, "mat-metadata")
    report = run_benchmark(
        cfg,
        [0],
        materialization_target_ids=[0, 1],
        tier=1,
        post_prepare_hook=_fake_model_hook,
    )
    assert [row["feature_ids"] for row in report["targets"]] == [[0]]
    assert [row["feature_ids"] for row in report["materialization_targets"]] == [[0], [1]]
    assert report["benchmark"]["target_count"] == 1
    assert report["benchmark"]["materialization_target_count"] == 2
    assert report["benchmark"]["materialization_prefetch"] is True


def test_run_benchmark_records_prefetch_off_without_changing_config_identity(
    tmp_path: Path,
) -> None:
    cfg = _synthetic_cfg(tmp_path, "prefetch-off")
    report = run_benchmark(
        cfg,
        [0],
        tier=1,
        post_prepare_hook=_fake_model_hook,
        materialization_prefetch=False,
    )
    assert report["benchmark"]["materialization_prefetch"] is False
    assert "materialization_prefetch" not in report["config"]["token_examples"]


@pytest.mark.parametrize("bad", [0, 1, "off", None])
def test_run_benchmark_prefetch_override_is_strict_bool(tmp_path: Path, bad) -> None:
    with pytest.raises(AutoInterpDataError, match="materialization_prefetch.*strict bool"):
        run_benchmark(
            _synthetic_cfg(tmp_path, f"bad-prefetch-{bad}"),
            [0],
            materialization_prefetch=bad,
        )


def _calibration_profile(batch_size: int = 128) -> BatchToleranceProfile:
    return BatchToleranceProfile(
        name="test-unvalidated-calibration",
        batch_size=batch_size,
        selected_sae=NumericTolerance(
            max_abs=1_000_000.0,
            p99_abs=1_000_000.0,
            mean_abs=1_000_000.0,
        ),
        raw_residual=NumericTolerance(
            max_abs=1_000_000.0,
            p99_abs=1_000_000.0,
            mean_abs=1_000_000.0,
        ),
        calibration_contract=PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
        max_support_mismatches=0,
    )


def test_calibration_profile_cannot_leak_into_normal_benchmark_mode(tmp_path: Path) -> None:
    cfg = _synthetic_cfg(tmp_path, "calibration-leak")
    with pytest.raises(AutoInterpDataError, match="only with calibration_mode=true"):
        run_benchmark(cfg, [0], calibration_profile=_calibration_profile())
    assert not Path(cfg.token_example_cache_path).exists()


def test_calibration_mode_requires_profile_batch_and_reference_cache(tmp_path: Path) -> None:
    serial_cfg = _synthetic_cfg(tmp_path, "calibration-serial")
    with pytest.raises(AutoInterpDataError, match="requires an explicit profile"):
        run_benchmark(serial_cfg, [0], calibration_mode=True)
    with pytest.raises(AutoInterpDataError, match="batch size > 1"):
        run_benchmark(
            serial_cfg,
            [0],
            calibration_mode=True,
            calibration_profile=_calibration_profile(),
        )

    batch_base = _synthetic_cfg(tmp_path, "calibration-batch")
    batch_cfg = replace(
        batch_base,
        token_examples=replace(
            batch_base.token_examples,
            materialization_batch_size=128,
        ),
    )
    with pytest.raises(AutoInterpDataError, match="requires a serial reference cache"):
        run_benchmark(
            batch_cfg,
            [0],
            calibration_mode=True,
            calibration_profile=_calibration_profile(),
        )


def test_full_anchor_validation_mode_is_strict_and_cannot_alias_calibration(
    tmp_path: Path,
) -> None:
    serial_cfg = _synthetic_cfg(tmp_path, "strict-anchor-mode")
    with pytest.raises(AutoInterpDataError, match="calibration_mode must be a strict bool"):
        run_benchmark(serial_cfg, [0], calibration_mode=cast(Any, 1))
    with pytest.raises(
        AutoInterpDataError,
        match="validate_all_anchor_surfaces must be a strict bool",
    ):
        run_benchmark(
            serial_cfg,
            [0],
            validate_all_anchor_surfaces=cast(Any, 1),
        )
    with pytest.raises(AutoInterpDataError, match="do not set both modes"):
        run_benchmark(
            serial_cfg,
            [0],
            calibration_mode=True,
            calibration_profile=_calibration_profile(),
            validate_all_anchor_surfaces=True,
        )
    with pytest.raises(AutoInterpDataError, match="materialization batch size > 1"):
        run_benchmark(serial_cfg, [0], validate_all_anchor_surfaces=True)

    batch_cfg = replace(
        serial_cfg,
        token_examples=replace(
            serial_cfg.token_examples,
            materialization_batch_size=128,
        ),
    )
    with pytest.raises(AutoInterpDataError, match="requires a serial reference cache"):
        run_benchmark(batch_cfg, [0], validate_all_anchor_surfaces=True)


def test_serial_benchmark_reports_no_batch_profile_or_unvalidated_candidate(
    tmp_path: Path,
) -> None:
    report = run_benchmark(
        _synthetic_cfg(tmp_path, "serial-validation-meta"),
        [0],
        tier=1,
        post_prepare_hook=_fake_model_hook,
    )
    assert report["benchmark"]["unvalidated_candidate"] is False
    assert report["benchmark"]["full_anchor_validation"] is False
    assert report["batch_validation"] == {
        "profile": None,
        "unvalidated_candidate": False,
        "observations": [],
        "profile_observations": None,
    }


def test_run_benchmark_validates_reference_cache_upfront(tmp_path: Path) -> None:
    """A bad --reference-cache must fail BEFORE materialization, not after."""
    cfg = _synthetic_cfg(tmp_path, "refpre")
    with pytest.raises(AutoInterpDataError, match="does not exist"):
        run_benchmark(
            cfg,
            [0],
            tier=1,
            post_prepare_hook=_fake_model_hook,
            reference_cache_path=tmp_path / "missing.sqlite",
        )
    assert not Path(cfg.token_example_cache_path).exists()  # nothing ran


def test_reference_cache_must_contain_selected_targets(tmp_path: Path) -> None:
    ref_cfg = _synthetic_cfg(tmp_path, "refonly1")
    run_benchmark(ref_cfg, [1], tier=1, post_prepare_hook=_fake_model_hook)
    cfg = _synthetic_cfg(tmp_path, "refcand")
    with pytest.raises(AutoInterpDataError, match="missing target keys"):
        run_benchmark(
            cfg,
            [0],
            tier=1,
            post_prepare_hook=_fake_model_hook,
            reference_cache_path=Path(ref_cfg.token_example_cache_path),
        )
    assert not Path(cfg.token_example_cache_path).exists()


def test_reference_cache_rejects_ambiguous_target_namespaces(tmp_path: Path) -> None:
    ref_cfg = _synthetic_cfg(tmp_path, "ref-ambiguous")
    run_benchmark(ref_cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)
    reference_path = Path(ref_cfg.token_example_cache_path)
    with sqlite3.connect(reference_path) as conn:
        conn.execute(
            "INSERT INTO token_example_sets (namespace, target_key, payload) "
            "SELECT namespace || '-duplicate', target_key, payload "
            "FROM token_example_sets WHERE target_key = 'feature:0'"
        )
        conn.commit()

    cfg = _synthetic_cfg(tmp_path, "candidate-ambiguous")
    with pytest.raises(AutoInterpDataError, match="multiple namespaces"):
        run_benchmark(
            cfg,
            [0],
            tier=1,
            post_prepare_hook=_fake_model_hook,
            reference_cache_path=reference_path,
        )
    assert not Path(cfg.token_example_cache_path).exists()


def test_model_instrumentation_restores_tokenizer_and_instance_methods(
    tmp_path: Path,
) -> None:
    """A dropped tokenizer-restore closure must not survive this test."""
    provider = _bare_provider(tmp_path)
    _fake_model_hook(provider)
    original_tokenizer = provider._tokenizer
    recorder = PhaseRecorder()
    handle = install_instrumentation(provider, recorder=recorder, tier=2, cuda_sync=False)
    install_model_instrumentation(provider, handle=handle, tier=2, cuda_sync=False)
    assert isinstance(provider._tokenizer, TokenizerTimingProxy)
    assert "run_with_cache" in provider._model.__dict__
    handle.restore()
    assert provider._tokenizer is original_tokenizer
    assert "run_with_cache" not in provider._model.__dict__
    assert "_doc_activations" not in provider.__dict__


def test_run_benchmark_refuses_hot_sqlite_sidecars(tmp_path: Path) -> None:
    """Production loaders open the index WRITABLE; a stale rollback journal
    would be rolled back into the read-only input mid-benchmark."""
    base = _synthetic_cfg(tmp_path, "journal")
    idx = tmp_path / "idx.sqlite"
    idx.write_bytes(b"")
    (tmp_path / "idx.manifest.json").write_text("{}")
    (tmp_path / "idx.sqlite-journal").write_bytes(b"")
    cfg = replace(
        base,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(idx),
            manifest_path=str(tmp_path / "idx.manifest.json"),
        ),
    )
    with pytest.raises(AutoInterpDataError, match="hot sqlite sidecar"):
        run_benchmark(cfg, [0], tier=1, post_prepare_hook=_fake_model_hook)


def test_parse_args_rejects_extra_overrides_colliding_with_dedicated_flags(
    tmp_path: Path,
) -> None:
    base = ["--output-dir", str(tmp_path), "--report", str(tmp_path / "r.json"), "--targets", "0"]
    with pytest.raises(SystemExit):  # neuronpedia.enabled is always pinned
        parse_args([*base, "--extra-override", "neuronpedia.enabled=true"])
    with pytest.raises(SystemExit):  # hydra last-wins would beat --device
        parse_args([*base, "--device", "cpu", "--extra-override", "token_examples.device=cuda"])
    # unpinned keys still pass through
    args = parse_args([*base, "--extra-override", "seed=43"])
    assert args.extra_override == ["seed=43"]


def test_compare_surface_threads_highlight_threshold_ratio() -> None:
    reference = TokenExampleSet(
        positives=(_mk_ref(doc_id=1, activations=(0.0, 10.0, 3.5, 0.0)),),
        non_activating=(),
    )
    candidate = TokenExampleSet(
        positives=(_mk_ref(doc_id=1, activations=(0.0, 10.0, 5.1, 0.0)),),
        non_activating=(),
    )
    at_default = compare_token_example_surface(reference, candidate)
    at_half = compare_token_example_surface(reference, candidate, highlight_threshold_ratio=0.5)
    assert at_default.positives.highlight_exact_count == 1  # both highlight {1, 2}
    assert at_half.positives.highlight_exact_count == 0  # ref {1} vs cand {1, 2}


def test_select_benchmark_targets_respects_allowed_set() -> None:
    counts = [(0, 3), (1, 2), (2, 1)]
    assert select_benchmark_targets(counts, 2, allowed={0, 2}) == (0, 2)
    with pytest.raises(AutoInterpDataError, match="benchmarkable candidate set"):
        select_benchmark_targets(counts, 1, allowed={1, 2})  # pinned 0 excluded


def test_validation_target_plan_is_deterministic_preselected_60_40_and_block_complete() -> None:
    counts = [(feature_id, (feature_id * 37) % 101 + 1) for feature_id in range(2048)]
    allowed = set(range(2048))

    first = select_validation_target_plan(counts, allowed=allowed)
    second = select_validation_target_plan(list(reversed(counts)), allowed=set(allowed))

    assert first == second
    assert len(first.calibration_target_ids) == 60
    assert len(first.heldout_target_ids) == 40
    assert len(first.anchor_target_ids) == 100
    assert not set(first.calibration_target_ids) & set(first.heldout_target_ids)
    assert set(first.anchor_target_ids) <= set(first.materialization_target_ids)
    assert first.block_ids == tuple(sorted({fid // 128 for fid in first.anchor_target_ids}))
    assert first.materialization_target_ids == tuple(
        fid for fid in sorted(allowed) if fid // 128 in set(first.block_ids)
    )
    calibration_materialization = set(first.calibration_materialization_target_ids)
    heldout_materialization = set(first.heldout_materialization_target_ids)
    assert set(first.calibration_target_ids) <= calibration_materialization
    assert set(first.heldout_target_ids) <= heldout_materialization
    assert not calibration_materialization & set(first.heldout_target_ids)
    assert not heldout_materialization & set(first.calibration_target_ids)
    assert calibration_materialization | heldout_materialization == set(
        first.materialization_target_ids
    )
    assert first.calibration_block_ids == tuple(
        sorted({feature_id // 128 for feature_id in first.calibration_target_ids})
    )
    assert first.heldout_block_ids == tuple(
        sorted({feature_id // 128 for feature_id in first.heldout_target_ids})
    )
    assert 0 in first.anchor_target_ids
    assert len(first.selection_inputs_sha256) == len(first.selection_sha256) == 64
    payload = first.to_json()
    assert payload["schema"] == "autointerp_batch_validation_plan_v2"
    assert payload["selection_method"].endswith("v5_stats_grid10_midpoint_split_isolated")
    assert payload["split_materialization_policy"] == "opposite_anchor_dummy_v1"
    assert first.resolved_target_count == 2048
    assert first.completed_reference_target_count == 2048
    assert first.selection_candidate_count == 2048
    assert payload["selection_sha256"] == first.selection_sha256


def test_validation_target_plan_represents_id_density_and_block_slot_boundaries() -> None:
    counts = [(feature_id, feature_id + 1) for feature_id in range(1024)]
    plan = select_validation_target_plan(counts, allowed=set(range(1024)))
    anchors = set(plan.anchor_target_ids)
    assert 0 in anchors
    assert 1023 in anchors
    assert min(anchors) == 0
    assert max(anchors) == 1023
    slots = {feature_id % 128 for feature_id in anchors}
    assert min(slots) == 0
    assert max(slots) == 127
    anchor_counts = [dict(counts)[feature_id] for feature_id in anchors]
    assert min(anchor_counts) == 1
    assert max(anchor_counts) == 1024


def test_validation_target_plan_spreads_anchors_across_fine_id_density_grid() -> None:
    counts = [(feature_id, (feature_id * 997) % 4093 + 1) for feature_id in range(4096)]
    plan = select_validation_target_plan(counts, allowed=set(range(4096)))
    by_id = [feature_id for feature_id, _count in counts]
    by_density = [
        feature_id for feature_id, _count in sorted(counts, key=lambda item: (item[1], item[0]))
    ]

    def deciles(ordered: list[int]) -> dict[int, int]:
        return {
            feature_id: min(9, 10 * index // len(ordered))
            for index, feature_id in enumerate(ordered)
        }

    id_decile = deciles(by_id)
    density_decile = deciles(by_density)
    anchor_cells = {
        (id_decile[feature_id], density_decile[feature_id]) for feature_id in plan.anchor_target_ids
    }

    assert {cell[0] for cell in anchor_cells} == set(range(10))
    assert {cell[1] for cell in anchor_cells} == set(range(10))
    assert len(anchor_cells) >= 85


def test_validation_target_plan_separates_completed_reference_anchors_from_block_fillers() -> None:
    counts = [(feature_id, feature_id + 1) for feature_id in range(1024)]
    allowed = set(range(1024))
    completed_reference = set(range(256, 512))

    plan = select_validation_target_plan(
        counts,
        allowed=allowed,
        anchor_allowed=completed_reference,
    )

    assert set(plan.anchor_target_ids) <= completed_reference
    assert min(plan.anchor_target_ids) == 256
    assert max(plan.anchor_target_ids) == 511
    assert plan.materialization_target_ids == tuple(range(256, 512))


@pytest.mark.parametrize(
    ("anchor_allowed", "fragment"),
    [
        (set(), "non-empty anchor_allowed"),
        ({-1, 0}, "non-negative strict ints"),
        ({0, 1024}, "subset of allowed"),
    ],
)
def test_validation_target_plan_rejects_invalid_reference_anchor_universe(
    anchor_allowed: set[int],
    fragment: str,
) -> None:
    with pytest.raises(AutoInterpDataError, match=fragment):
        select_validation_target_plan(
            [(feature_id, 1) for feature_id in range(1024)],
            allowed=set(range(1024)),
            anchor_allowed=anchor_allowed,
        )


def test_validation_target_plan_digest_pins_activation_counts_allowed_set_and_batch_size() -> None:
    counts = [(feature_id, feature_id + 1) for feature_id in range(300)]
    allowed = set(range(300))
    baseline = select_validation_target_plan(counts, allowed=allowed, batch_size=128)
    changed_count = select_validation_target_plan(
        [*counts[:-1], (299, 9999)],
        allowed=allowed,
        batch_size=128,
    )
    changed_allowed = select_validation_target_plan(
        counts,
        allowed=allowed - {298},
        batch_size=128,
    )
    changed_batch = select_validation_target_plan(
        counts,
        allowed=allowed,
        batch_size=64,
    )
    assert changed_count.selection_inputs_sha256 != baseline.selection_inputs_sha256
    assert changed_allowed.selection_inputs_sha256 != baseline.selection_inputs_sha256
    assert changed_batch.selection_sha256 != baseline.selection_sha256


def test_validation_target_plan_json_roundtrip_and_digest_reject_tampering() -> None:
    plan = select_validation_target_plan(
        [(feature_id, feature_id + 1) for feature_id in range(512)],
        allowed=set(range(512)),
    )
    payload = plan.to_json()

    assert ValidationTargetPlan.from_json(json.loads(json.dumps(payload))) == plan

    tampered = json.loads(json.dumps(payload))
    tampered["materialization_target_ids"].pop()
    with pytest.raises(AutoInterpDataError):
        ValidationTargetPlan.from_json(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["anchor_target_ids"][0] += 1
    with pytest.raises(AutoInterpDataError, match="anchor_target_ids"):
        ValidationTargetPlan.from_json(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["unexpected"] = True
    with pytest.raises(AutoInterpDataError, match="exactly the versioned plan fields"):
        ValidationTargetPlan.from_json(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["schema"] = "future"
    with pytest.raises(AutoInterpDataError, match="unsupported schema"):
        ValidationTargetPlan.from_json(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["selection_method"] = "future"
    with pytest.raises(AutoInterpDataError, match="unsupported selection_method"):
        ValidationTargetPlan.from_json(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["split_materialization_policy"] = "future"
    with pytest.raises(AutoInterpDataError, match="unsupported split_materialization_policy"):
        ValidationTargetPlan.from_json(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["block_ids"] = "not-a-list"
    with pytest.raises(AutoInterpDataError, match="block_ids must be a list"):
        ValidationTargetPlan.from_json(tampered)


def test_validation_target_plan_object_rejects_every_structural_corruption() -> None:
    plan = select_validation_target_plan(
        [(feature_id, feature_id + 1) for feature_id in range(512)],
        allowed=set(range(512)),
    )
    cases = [
        ({"batch_size": True}, "batch_size must be a strict int"),
        ({"batch_size": 1}, "batch_size must be >= 2"),
        ({"resolved_target_count": True}, "strict positive int"),
        ({"resolved_target_count": 1}, "smaller than its materialization"),
        ({"completed_reference_target_count": 1}, "smaller than its anchors"),
        ({"selection_candidate_count": 99}, "population counts violate"),
        (
            {"calibration_target_ids": tuple(reversed(plan.calibration_target_ids))},
            "sorted unique tuple",
        ),
        ({"calibration_target_ids": ()}, "sets must be non-empty"),
        (
            {"heldout_target_ids": (plan.calibration_target_ids[0],)},
            "sets overlap",
        ),
        (
            {
                "materialization_target_ids": tuple(
                    feature_id
                    for feature_id in plan.materialization_target_ids
                    if feature_id != plan.calibration_target_ids[0]
                )
            },
            "lost an anchor",
        ),
        ({"block_ids": plan.block_ids[:-1]}, "block IDs"),
        (
            {
                "materialization_target_ids": (
                    *plan.materialization_target_ids,
                    (max(plan.block_ids) + 2) * plan.batch_size,
                ),
                "resolved_target_count": plan.resolved_target_count + 1,
            },
            "outside its anchor blocks",
        ),
        ({"selection_inputs_sha256": "not-a-sha"}, "64 lowercase hex"),
        ({"selection_sha256": "0" * 64}, "does not match its exact contents"),
    ]
    for updates, fragment in cases:
        with pytest.raises(AutoInterpDataError, match=fragment):
            replace(plan, **updates)


def test_validation_target_plan_split_materialization_defenses_are_fail_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise defenses that are redundant with earlier field invariants.

    The public properties derive these split surfaces, so ordinary constructor
    inputs cannot violate the checks.  Replacing each derived property is the
    narrowest way to prove the defensive checks themselves remain live.
    """
    plan = select_validation_target_plan(
        [(feature_id, feature_id + 1) for feature_id in range(512)],
        allowed=set(range(512)),
    )
    calibration = plan.calibration_materialization_target_ids
    heldout = plan.heldout_materialization_target_ids

    with monkeypatch.context() as patch:
        patch.setattr(
            ValidationTargetPlan,
            "calibration_materialization_target_ids",
            property(lambda _plan: ()),
        )
        with pytest.raises(AutoInterpDataError, match="calibration materialization lost"):
            replace(plan)

    with monkeypatch.context() as patch:
        patch.setattr(
            ValidationTargetPlan,
            "heldout_materialization_target_ids",
            property(lambda _plan: ()),
        )
        with pytest.raises(AutoInterpDataError, match="heldout materialization lost"):
            replace(plan)

    leaked_heldout = plan.heldout_target_ids[0]
    with monkeypatch.context() as patch:
        patch.setattr(
            ValidationTargetPlan,
            "calibration_materialization_target_ids",
            property(lambda _plan: tuple(sorted((*calibration, leaked_heldout)))),
        )
        with pytest.raises(AutoInterpDataError, match="leaked heldout anchors"):
            replace(plan)

    leaked_calibration = plan.calibration_target_ids[0]
    with monkeypatch.context() as patch:
        patch.setattr(
            ValidationTargetPlan,
            "heldout_materialization_target_ids",
            property(lambda _plan: tuple(sorted((*heldout, leaked_calibration)))),
        )
        with pytest.raises(AutoInterpDataError, match="leaked calibration anchors"):
            replace(plan)

    filler = next(
        feature_id
        for feature_id in plan.materialization_target_ids
        if feature_id not in set(plan.anchor_target_ids)
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            ValidationTargetPlan,
            "calibration_materialization_target_ids",
            property(lambda _plan: tuple(fid for fid in calibration if fid != filler)),
        )
        patch.setattr(
            ValidationTargetPlan,
            "heldout_materialization_target_ids",
            property(lambda _plan: tuple(fid for fid in heldout if fid != filler)),
        )
        with pytest.raises(AutoInterpDataError, match="do not cover the combined blocks"):
            replace(plan)


def test_validation_target_plan_population_coverage_handles_small_splits_and_rejects_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = [(feature_id, feature_id + 1) for feature_id in range(200)]
    allowed = set(range(200))

    small = select_validation_target_plan(
        counts,
        allowed=allowed,
        n_calibration=4,
        n_heldout=4,
    )
    assert len(small.calibration_target_ids) == 4
    assert len(small.heldout_target_ids) == 4

    monkeypatch.setattr(
        batch_acceptance_module,
        "_select_validation_anchors",
        lambda _counts, *, total, mandatory: tuple(range(total)),
    )
    with pytest.raises(AutoInterpDataError, match="hash split lost population coverage"):
        select_validation_target_plan(counts, allowed=allowed)


def test_validation_target_plan_rejects_zero_positive_reference_candidates() -> None:
    with pytest.raises(AutoInterpDataError, match="no positive allowed candidates"):
        select_validation_target_plan(
            [(feature_id, 0) for feature_id in range(200)],
            allowed=set(range(200)),
        )


def test_validation_target_plan_rejects_topk_capped_counts_as_fake_density() -> None:
    with pytest.raises(AutoInterpDataError, match="top-k-capped source-index"):
        select_validation_target_plan(
            [(feature_id, 128) for feature_id in range(1024)],
            allowed=set(range(1024)),
        )


def test_validation_anchor_kernel_fails_loud_on_broken_internal_preconditions() -> None:
    with pytest.raises(AutoInterpDataError, match="anchor count"):
        batch_acceptance_module._select_validation_anchors(
            [(0, 1)],
            total=2,
            mandatory=(0,),
        )
    with pytest.raises(AutoInterpDataError, match="mandatory validation anchors"):
        batch_acceptance_module._select_validation_anchors(
            [(0, 1), (1, 2)],
            total=1,
            mandatory=(2,),
        )
    with pytest.raises(AutoInterpDataError, match="grid exhausted"):
        batch_acceptance_module._select_validation_anchors(
            [(0, 1), (0, 2)],
            total=2,
            mandatory=(0,),
        )


def test_benchmark_stat_helpers_cover_absent_stable_and_changed_inputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact"
    assert benchmark_module._stat_tuple(path) is None
    path.write_bytes(b"abc")
    observed = benchmark_module._stat_tuple(path)
    assert observed == (3, path.stat().st_mtime_ns)
    benchmark_module._require_unchanged_stat(
        label="source index",
        before=observed,
        after=observed,
    )
    with pytest.raises(AutoInterpDataError, match="source index stat changed"):
        benchmark_module._require_unchanged_stat(
            label="source index",
            before=observed,
            after=(4, observed[1]),
        )
    with pytest.raises(AutoInterpDataError, match="source index manifest stat changed"):
        benchmark_module._require_unchanged_stat(
            label="source index manifest",
            before=None,
            after=observed,
        )


def test_environment_metadata_records_unreadable_uptime_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_text = Path.read_text

    def _read_text(path: Path, *args, **kwargs) -> str:
        if path == Path("/proc/uptime"):
            raise OSError("injected unreadable uptime")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_text)
    monkeypatch.setattr(benchmark_module, "_nvidia_compute_process_count", lambda: 0)
    metadata = benchmark_module._environment_metadata(cuda_device=None)
    assert metadata["uptime_seconds"] is None
    assert metadata["gpu_name"] is None


@pytest.mark.parametrize(
    ("counts", "allowed", "fragment"),
    [
        ([[0, 1]], {0}, "must be .* tuples"),
        ([(True, 1)], {0}, "strict non-negative integer IDs"),
        ([(0, True)], {0}, "strict non-negative integer counts"),
        ([(0, -1)], {0}, "strict non-negative integer counts"),
        ([(0, 1), (0, 2)], {0}, "duplicate feature ID"),
        ([(feature_id, feature_id + 1) for feature_id in range(200)], {0, 200}, "missing"),
    ],
)
def test_validation_target_plan_rejects_malformed_activation_counts(
    counts: list,
    allowed: set[int],
    fragment: str,
) -> None:
    with pytest.raises(AutoInterpDataError, match=fragment):
        select_validation_target_plan(counts, allowed=allowed)


@pytest.mark.parametrize("include", [(True,), (-1,), (1, 1), "1"])
def test_validation_target_plan_rejects_invalid_included_ids(include: object) -> None:
    with pytest.raises(AutoInterpDataError, match="included target IDs"):
        select_validation_target_plan(
            [(feature_id, feature_id + 1) for feature_id in range(200)],
            allowed=set(range(200)),
            include=cast(Any, include),
        )


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"allowed": set()}, "non-empty allowed"),
        ({"allowed": {0, -1}}, "non-negative strict"),
        ({"allowed": {0, 1}, "batch_size": 1}, "strict int >= 2"),
        ({"allowed": {0, 1}, "batch_size": True}, "strict int >= 2"),
        ({"allowed": {0, 1}, "n_calibration": 0}, "n_calibration"),
        ({"allowed": {0, 1}, "n_heldout": 0}, "n_heldout"),
    ],
)
def test_validation_target_plan_rejects_invalid_contract(
    kwargs: dict,
    fragment: str,
) -> None:
    with pytest.raises(AutoInterpDataError, match=fragment):
        select_validation_target_plan([(0, 1), (1, 1)], **kwargs)


def test_performance_acceptance_applies_1_2x_and_3_percent_plus_ci_thresholds() -> None:
    report = evaluate_batch_performance_replicates(
        serial_seconds=[12.2, 12.0, 11.9, 12.1, 12.0],
        batched_seconds=[10.0, 10.0, 9.9, 10.1, 10.0],
        prefetched_seconds=[9.6, 9.6, 9.5, 9.7, 9.6],
        bootstrap_samples=1000,
    )
    assert report["batch_speedup"] >= 1.2
    assert report["prefetch_speedup"] >= 1.03
    assert report["prefetch_paired_speedup_95ci"][0] > 1.0
    assert report["batch_pass"] is True
    assert report["prefetch_pass"] is True
    assert report["retain_prefetch"] is True
    assert report["batch_threshold"] == 1.2
    assert report["prefetch_threshold"] == 1.03


def test_performance_acceptance_rejects_batch_below_1_2x() -> None:
    report = evaluate_batch_performance_replicates(
        serial_seconds=[11.9, 12.0, 12.1],
        batched_seconds=[10.1, 10.1, 10.1],
        prefetched_seconds=[9.7, 9.7, 9.7],
        bootstrap_samples=500,
    )
    assert report["batch_speedup"] < 1.2
    assert report["batch_pass"] is False


def test_performance_acceptance_rejects_prefetch_without_three_percent_median() -> None:
    report = evaluate_batch_performance_replicates(
        serial_seconds=[13.0, 13.0, 13.0],
        batched_seconds=[10.0, 10.0, 10.0],
        prefetched_seconds=[9.8, 9.8, 9.8],
        bootstrap_samples=500,
    )
    assert report["batch_pass"] is True
    assert report["prefetch_speedup"] < 1.03
    assert report["prefetch_pass"] is False
    assert report["retain_prefetch"] is False


def test_performance_acceptance_rejects_prefetch_when_ci_includes_no_improvement() -> None:
    report = evaluate_batch_performance_replicates(
        serial_seconds=[13.0] * 7,
        batched_seconds=[10.0] * 7,
        prefetched_seconds=[8.0, 8.0, 8.0, 9.0, 11.0, 11.0, 11.0],
        bootstrap_samples=2000,
    )
    assert report["prefetch_speedup"] >= 1.03
    assert report["prefetch_paired_speedup_95ci"][0] <= 1.0
    assert report["prefetch_pass"] is False


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        (
            {
                "serial_seconds": [1.0, 1.0, 1.0],
                "batched_seconds": [1.0, 1.0],
                "prefetched_seconds": [1.0, 1.0, 1.0],
            },
            "equally-sized",
        ),
        (
            {
                "serial_seconds": [1.0, 1.0],
                "batched_seconds": [1.0, 1.0],
                "prefetched_seconds": [1.0, 1.0],
            },
            "at least 3",
        ),
        (
            {
                "serial_seconds": [1.0, 0.0, 1.0],
                "batched_seconds": [1.0, 1.0, 1.0],
                "prefetched_seconds": [1.0, 1.0, 1.0],
            },
            "finite and > 0",
        ),
    ],
)
def test_performance_acceptance_rejects_invalid_repeat_series(
    kwargs: dict,
    fragment: str,
) -> None:
    with pytest.raises(AutoInterpDataError, match=fragment):
        evaluate_batch_performance_replicates(**kwargs, bootstrap_samples=100)


@pytest.mark.parametrize("bad", [True, 99, 100.0])
def test_performance_acceptance_requires_strict_bootstrap_sample_count(bad) -> None:
    with pytest.raises(AutoInterpDataError, match="bootstrap_samples"):
        evaluate_batch_performance_replicates(
            serial_seconds=[2.0, 2.0, 2.0],
            batched_seconds=[1.0, 1.0, 1.0],
            prefetched_seconds=[0.9, 0.9, 0.9],
            bootstrap_samples=bad,
        )


def test_load_by_target_rejects_duplicate_rows_in_one_namespace(tmp_path: Path) -> None:
    examples = TokenExampleSet(positives=(_mk_ref(),), non_activating=())
    path = tmp_path / "cache.sqlite"
    _write_cache(path, [("ns1", "feature:0", examples), ("ns1", "feature:0", examples)])
    with pytest.raises(AutoInterpCacheError, match="2 rows"):
        load_token_example_set_by_target(path, "feature:0")


_CANONICAL_ROOT = Path("/home/d0rb/Documents/Github/nano-gpt-pretrain-steer")
_ASSET_ROOT = Path(os.environ.get("ISSUE678_ASSET_ROOT", str(_CANONICAL_ROOT)))
_PROD_FEATURES = _ASSET_ROOT / (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos"
)
_PROD_DATA = _ASSET_ROOT / "nanochat_cache/base_data"
# The recorded run dir is a dated artifact; glob so a regenerated index dir
# does not silently turn these gates into permanent skips.
_RECORDED_INDEX_DIRNAME = "gemma3_4bpt_l22_f65k_ci_nobos_whitelist_full_root_20260709_132415"


def _prod_index_dir() -> Path | None:
    recorded = _ASSET_ROOT / "out/autointerp_filter" / _RECORDED_INDEX_DIRNAME
    if (recorded / "source_doc_index.sqlite").exists():
        return recorded
    candidates = sorted(
        (_ASSET_ROOT / "out/autointerp_filter").glob(
            "gemma3_4bpt_l22_f65k_ci_nobos_whitelist_full_root_*"
        )
    )
    for candidate in reversed(candidates):
        if (candidate / "source_doc_index.sqlite").exists():
            return candidate
    return None


_TL_CPU_EXPECTED_NAMESPACE = "561912c19eb50a8013b604de4857f210dbf848a8690d9b41255b423e3a1f7a88"
_TL_CPU_EXPECTED_PAYLOAD = "dc874de18c1b00472737eee1bbbfdae707539538de5a93823813f47c7df2fed3"


def _namespace_is_reproducible(index_dir: Path) -> bool:
    """cache_context hashes the ABSOLUTE feature_dir/data_dir/source-index
    path strings, so the recorded namespace only reproduces from the canonical
    root and the recorded index dir; the payload gate is path-independent and
    always applies."""
    return _ASSET_ROOT == _CANONICAL_ROOT and index_dir.name == _RECORDED_INDEX_DIRNAME


def _skip_unless_prod_assets() -> Path:
    index_dir = _prod_index_dir()
    if index_dir is None:
        pytest.skip(f"no completed source index under {_ASSET_ROOT / 'out/autointerp_filter'}")
    for path in (
        _PROD_FEATURES,
        _PROD_DATA,
        index_dir / "source_doc_index.sqlite",
        index_dir / "source_doc_index.manifest.json",
    ):
        if not path.exists():
            pytest.skip(f"production asset missing: {path}")
    return index_dir


@pytest.mark.benchmark
@pytest.mark.requires_feature_data
def test_real_tl_cpu_feature0_reproduces_recorded_namespace_and_payload(
    tmp_path: Path,
) -> None:
    index_dir = _skip_unless_prod_assets()
    cfg = compose_benchmark_config(
        "precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos",
        [
            f"output_dir={tmp_path / 'out'}",
            "neuronpedia.enabled=false",
            f"feature_dir={_PROD_FEATURES}",
            f"data_dir={_PROD_DATA}",
            "source_index.enabled=true",
            f"source_index.path={index_dir / 'source_doc_index.sqlite'}",
            f"source_index.manifest_path={index_dir / 'source_doc_index.manifest.json'}",
            "token_examples.device=cpu",
            "token_examples.activation_backend=transformerlens",
            "token_examples.model_dtype=float32",
            "token_examples.max_positive_windows_per_target=16",
            "token_examples.max_non_activating_windows_per_target=16",
            "token_examples.materialization_batch_size=1",
        ],
    )
    report = run_benchmark(cfg, [0], tier=1)
    row = report["per_target"][0]
    # Payload is content-only and must ALWAYS reproduce; the namespace embeds
    # absolute asset paths and only reproduces from the canonical layout.
    assert row["payload_sha256"] == _TL_CPU_EXPECTED_PAYLOAD
    if _namespace_is_reproducible(index_dir):
        assert row["namespace"] == _TL_CPU_EXPECTED_NAMESPACE
    assert report["preconditions"]["index_stat_unchanged"] is True


@pytest.mark.benchmark
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="needs CUDA")
def test_real_native_f32_feature0_reproduces_recorded_namespace_and_payload(
    tmp_path: Path,
) -> None:
    index_dir = _skip_unless_prod_assets()
    cfg = compose_benchmark_config(
        "precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos",
        [
            f"output_dir={tmp_path / 'out'}",
            "neuronpedia.enabled=false",
            f"feature_dir={_PROD_FEATURES}",
            f"data_dir={_PROD_DATA}",
            "source_index.enabled=true",
            f"source_index.path={index_dir / 'source_doc_index.sqlite'}",
            f"source_index.manifest_path={index_dir / 'source_doc_index.manifest.json'}",
            "token_examples.model_dtype=float32",
            "token_examples.max_positive_windows_per_target=16",
            "token_examples.max_non_activating_windows_per_target=16",
            "token_examples.materialization_batch_size=1",
        ],
    )
    report = run_benchmark(cfg, [0], tier=1)
    row = report["per_target"][0]
    # Payload gate always applies; namespace gate only from the canonical layout.
    assert row["payload_sha256"] == (
        "b8caef8907a1ef578cb663370b3af0c27a885d142aa2e28b5f75ee3034e44788"
    )
    if _namespace_is_reproducible(index_dir):
        assert row["namespace"] == (
            "85cc0bb83676a972ff4542259f83ee42ff896ea0d68f1009676310129fa04ccc"
        )
