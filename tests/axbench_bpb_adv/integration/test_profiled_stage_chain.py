"""End-to-end PROFILED stage-chain integration test for the BPB-Adv eval (#501).

This is the highest-value guard for the profiling instrumentation: it drives the
REAL stage dispatch (``validate_source -> ... -> plot`` via ``_run_stage``) on a
tiny both-classes-per-prefix source fixture **with an ``AxBenchProfiler``
bound**, then asserts the per-stage counters land in ``events.jsonl`` with
correct, finite values.

Why it matters: every per-stage ``record_counter`` is gated behind
``get_current_profiler() is not None`` (zero-overhead in production). The
unit/functional contract tests bind a profiler but feed *synthesized* tables;
this test feeds tables produced by the REAL ``prepare_attack_rows`` /
``build_candidates`` (model-backed candidate builder) / ``score`` artifact
writers, so the token counters read the REAL ``CANDIDATES_SCHEMA`` /
``SCORE_SCHEMA`` columns. A renamed/removed column (the #505 class-margin
refactor renamed ``output_tokens`` -> ``concept_rich/avoiding_output_tokens``
and ``n_output_tokens`` -> ``n_rich/avoid_output_tokens``) would raise a
``KeyError`` *only* on the profiled path — invisible to a green production run.
This test is the only thing in the suite that exercises that path on real data.

The model-heavy runtime pieces (target-model load, context-cache extraction, the
GPU scorer forward) are stubbed exactly as in
``tests.axbench_bpb_adv.functional.test_cli_stage_chain`` — everything else
(source validation, attack-row pairing + schema, candidate building,
aggregation, plotting, dispatch, AND the profiler instrumentation) runs for
real. The fixture builders are imported from that module so this test stays in
lockstep with the canonical end-to-end harness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import torch as t

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackWorkSpec
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    Stage,
)
from steering.eval.axbench_bpb_adv.model_optimize import (
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import SCORE_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table

# Reuse the canonical end-to-end fixture builders so this profiled variant
# never drifts from #505's stage-chain harness.
from tests.axbench_bpb_adv.functional.test_cli_stage_chain import (
    _VOCAB,
    _dual_score_pair,
    _meta,
    _TinyTokenizer,
    _write_source_run,
)
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)


def _read_events(profile_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _counters(events: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for event in events:
        if event["type"] == "counter":
            grouped.setdefault(event["name"], []).append(event["value"])
    return grouped


def _regions(events: list[dict[str, Any]]) -> set[str]:
    return {e["name"] for e in events if e["type"] == "region"}


def _with_fair_feature_diffmean_score_targets(scores: pa.Table) -> pa.Table:
    rows = scores.to_pylist()
    existing = {
        (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        for row in rows
        if str(row["target_model"]) == "encdec"
        and str(row["method_name"])
        in {"feature_diffmean_encdec", "neg_feature_diffmean_encdec"}
        and bool(row["is_control"])
    }
    targets: list[dict[str, Any]] = []
    seen = set(existing)
    for row in rows:
        if not bool(row["is_control"]):
            continue
        key = (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        if key in seen:
            continue
        seen.add(key)
        mode = str(row["mode"])
        method_name = (
            "neg_feature_diffmean_encdec"
            if mode == "negative"
            else "feature_diffmean_encdec"
        )
        targets.append(
            {
                **row,
                "candidate_id": f"fair-feature-diffmean-target-{mode}",
                "method_name": method_name,
                "target_model": "encdec",
                "coefficient_str": "None",
                "n_rich_output_tokens": 0,
                "n_avoid_output_tokens": 0,
                "n_rich_output_bytes": 0,
                "n_avoid_output_bytes": 0,
            }
        )
    return pa.Table.from_pylist([*rows, *targets], schema=SCORE_SCHEMA)


def test_profiled_stage_chain_emits_all_counters_on_real_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    _write_source_run(source_root)
    output_root = tmp_path / "out"

    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_root),
        output_root=str(output_root),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0, 1),
                max_steps=0,
            )
        ],
        # score_general_val isn't mocked in this chain; general_val_bpb now
        # defaults to enabled=True (needs a real model), so keep it off.
        general_val_bpb=GeneralValBpbConfig(enabled=False),
        scatter_baseline_x="baseline_loss",
    )
    tokenizer = _TinyTokenizer()

    # --- Stub ONLY the model-heavy runtime pieces (same as the stage-chain
    # harness). prepare_attack_rows / build_candidates / score artifact writers
    # and the profiler instrumentation run for real. ---
    monkeypatch.setattr(
        bpb_adv_cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg: ({"decoder_only": _meta()}, tokenizer),
    )

    def _fake_work_specs(**_kwargs: Any) -> list[BpbAttackWorkSpec]:
        return [
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
            BpbAttackWorkSpec(
                method_name="neg_diffmean_decoder_only",
                mode=SteeringMode.NEGATIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
        ]

    monkeypatch.setattr(bpb_adv_cli, "work_specs_from_method_registry", _fake_work_specs)
    # prepare_context_cache delegates to the BPB context-cache extractor (model
    # heavy) — stub it; the @_profiled decorator + its cpu_rss counter still run.
    monkeypatch.setattr(
        bpb_adv_cli, "_run_bpb_context_cache_stage", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        bpb_adv_cli, "RuntimeState", lambda **_k: object()
    )
    monkeypatch.setattr(
        bpb_adv_cli, "_bpb_config_from_adv_config", lambda _cfg: object()
    )

    def _fake_candidate_stage(
        *, cfg: AxBenchBpbAdvConfig, paths: AxBenchBpbAdvPaths
    ) -> int:
        return run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=tokenizer,
            embedding_weight=t.zeros(_VOCAB, 1),
            test_objective=zero_objective,
            test_gradient_provider_batch=zero_gradient_batch,
        )

    monkeypatch.setattr(
        bpb_adv_cli, "run_model_backed_bpb_candidate_stage", _fake_candidate_stage
    )

    def _fake_score_stage(
        *,
        cfg: AxBenchBpbAdvConfig,
        paths: AxBenchBpbAdvPaths,
        candidates: pa.Table,
        stats: dict[str, int] | None = None,
    ) -> pa.Table:
        scores = build_bpb_adv_score_table(
            candidates=candidates,
            context_segments_by_prefix_hash={},
            tokenizer=tokenizer,
            token_bytes=t.ones(_VOCAB, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 32},
            score_pair=_dual_score_pair,
        )
        return _with_fair_feature_diffmean_score_targets(scores)

    monkeypatch.setattr(
        bpb_adv_cli, "run_model_backed_bpb_adv_score_stage", _fake_score_stage
    )

    # --- Drive the REAL dispatch under a bound profiler. ---
    profile_dir = output_root / "run" / "profile"
    with AxBenchProfiler(profile_dir, metadata={"test": "profiled-stage-chain"}):
        for stage in bpb_adv_cli._resolve_stages(Stage.ALL):
            bpb_adv_cli._run_stage(stage, cfg, ddp_rank=0, ddp_world_size=1)

    events = _read_events(profile_dir)
    counters = _counters(events)
    regions = _regions(events)

    # (1) Every stage ran as a profiled region.
    for stage_name in (
        "validate_source",
        "prepare_context_cache",
        "prepare_attack_rows",
        "build_candidates",
        "score",
        "aggregate",
        "plot",
    ):
        assert f"axbench_bpb_adv.{stage_name}" in regions, (
            f"stage region {stage_name} missing from profile"
        )
    # The nested sub-region inside prepare_attack_rows (the model-load cost,
    # the single most expensive sub-stage) must also be emitted — the
    # decorator-namespacing contract test only parses @_profiled args, so this
    # inline `with profile_region(...)` is otherwise untested for deletion.
    assert "axbench_bpb_adv.prepare_attack_rows.target_runtime" in regions, (
        "nested target_runtime profile_region missing from profile"
    )

    # (2) The body-level counters that read REAL schema fields are present.
    # If any column was renamed/removed by #505 these would have raised
    # KeyError inside the profiled stage — the headline silent-breakage risk.
    required = [
        "axbench_bpb_adv.validate_source.concepts",
        "axbench_bpb_adv.validate_source.positive_rows",
        "axbench_bpb_adv.validate_source.negative_rows",
        "axbench_bpb_adv.validate_source.synth_rows_total",
        "axbench_bpb_adv.validate_source.dropped",
        "axbench_bpb_adv.prepare_attack_rows.work_specs",
        "axbench_bpb_adv.prepare_attack_rows.samples",
        "axbench_bpb_adv.prepare_attack_rows.positive_rows_kept",
        "axbench_bpb_adv.prepare_attack_rows.negative_rows_kept",
        "axbench_bpb_adv.prepare_attack_rows.prefix_groups",
        "axbench_bpb_adv.prepare_attack_rows.core_prefixes",
        "axbench_bpb_adv.prepare_attack_rows.evaluation_core_prefixes",
        "axbench_bpb_adv.prepare_attack_rows.dropped",
        "axbench_bpb_adv.prepare_attack_rows.attack_rows",
        "axbench_bpb_adv.build_candidates.candidates_written",
        "axbench_bpb_adv.build_candidates.candidate_rows",
        "axbench_bpb_adv.build_candidates.source_prompt_tokens_total",
        "axbench_bpb_adv.build_candidates.attacked_prompt_tokens_total",
        "axbench_bpb_adv.build_candidates.concept_rich_output_tokens_total",
        "axbench_bpb_adv.build_candidates.concept_avoiding_output_tokens_total",
        "axbench_bpb_adv.score.candidates_loaded",
        "axbench_bpb_adv.score.score_rows",
        "axbench_bpb_adv.score.source_prompt_tokens_total",
        "axbench_bpb_adv.score.attacked_prompt_tokens_total",
        "axbench_bpb_adv.score.rich_output_tokens_total",
        "axbench_bpb_adv.score.avoid_output_tokens_total",
        "axbench_bpb_adv.plot.macro_rows",
        "axbench_bpb_adv.plot.files_written",
    ]
    missing = [name for name in required if name not in counters]
    assert not missing, f"counters missing from real-driver profile: {missing}"

    # (3) Every counter value is a single, finite, non-negative number.
    for name in required:
        values = counters[name]
        assert len(values) == 1, f"{name} emitted {len(values)} times, expected 1"
        value = values[0]
        assert math.isfinite(value) and value >= 0, f"{name} = {value!r}"

    # (4) Deterministic invariants from the complete (no-drop) fixture.
    assert counters["axbench_bpb_adv.validate_source.dropped"] == [0]
    assert counters["axbench_bpb_adv.prepare_attack_rows.dropped"] == [0]
    # No groups dropped -> every source group survived as a core prefix.
    assert (
        counters["axbench_bpb_adv.prepare_attack_rows.prefix_groups"]
        == counters["axbench_bpb_adv.prepare_attack_rows.core_prefixes"]
    )
    # The fixture has two positive + two negative accepted source prefixes; the
    # universal held-out population then selects one prefix per mode.
    assert counters["axbench_bpb_adv.prepare_attack_rows.prefix_groups"] == [4]
    assert counters["axbench_bpb_adv.prepare_attack_rows.evaluation_core_prefixes"] == [2]
    # Candidate count is internally consistent.
    assert (
        counters["axbench_bpb_adv.build_candidates.candidates_written"]
        == counters["axbench_bpb_adv.build_candidates.candidate_rows"]
    )
    # score sees the same candidates the build stage wrote.
    assert (
        counters["axbench_bpb_adv.score.candidates_loaded"]
        == counters["axbench_bpb_adv.build_candidates.candidate_rows"]
    )
    # The real plot writer rendered at least the nested PNGs the rglob counts.
    assert counters["axbench_bpb_adv.plot.files_written"][0] >= 1

    # Pin the per-completion token totals to their exact deterministic values on
    # the real driver — this catches a rich<->avoid column SWAP on the real-data
    # path (not just the contract test). The rich/avoid output-token totals
    # differ (the fixture's concept_rich and concept_avoiding completions have
    # distinct lengths), so a swap would flip these two numbers.
    rich_cand = counters["axbench_bpb_adv.build_candidates.concept_rich_output_tokens_total"][0]
    avoid_cand = counters["axbench_bpb_adv.build_candidates.concept_avoiding_output_tokens_total"][0]
    rich_score = counters["axbench_bpb_adv.score.rich_output_tokens_total"][0]
    avoid_score = counters["axbench_bpb_adv.score.avoid_output_tokens_total"][0]
    # The candidate-stage rich/avoid token lists and the score-stage rich/avoid
    # token counts are derived from the SAME completions, so they must agree.
    assert rich_cand == rich_score, (
        f"rich token total disagrees build={rich_cand} score={rich_score}"
    )
    assert avoid_cand == avoid_score, (
        f"avoid token total disagrees build={avoid_cand} score={avoid_score}"
    )
    # rich and avoid completions are distinct -> their totals must differ, so a
    # column swap is observable (not masked by equal values).
    assert rich_cand != avoid_cand, (
        "rich and avoid token totals are equal -> a column swap would be invisible; "
        "the fixture must use distinct-length completions"
    )

    # (5) The per-stage memory counters fire (CPU RSS is always available on the
    # POSIX CI host); proves the @_profiled exit-hook ran for every stage.
    if bpb_adv_cli._cpu_rss_kb() is not None:
        for stage_name in (
            "validate_source",
            "prepare_attack_rows",
            "build_candidates",
            "score",
            "aggregate",
            "plot",
        ):
            assert f"axbench_bpb_adv.{stage_name}.cpu_rss_kb" in counters
