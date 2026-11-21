"""Real Gemma-3-4B/L22/f65k acceptance audit for Issue #1049.

This is the model-and-data gate for positive-prefix headroom. It uses the
production 50-concept clustered run and its full 23,953-row Stage 0b train
pool, loads the real Gemma-3-4B-PT/GemmaScope-2 L22 65k pair on CUDA, and
scores the exact 3,600 prefixes that the pre-#1049 selector would have used.

Run manually when the GPU is free::

    AXBENCH_1049_REAL_ROOT=/path/to/nano-gpt-pretrain-steer \
      uv run pytest \
        tests/axbench_base/integration/test_issue_1049_real_gpu.py \
        -v -m 'very_slow and requires_feature_data'

The test is read-only with respect to the configured real root. Its JSON audit
is written only to pytest's temporary directory.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    SynthConfig,
    derive_positive_qc_thresholds,
    validate_config,
)
from steering.eval.axbench_base.extraction import (
    build_percentile_table_from_stats,
)
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.seed_contract import load_seed_pool
from steering.eval.axbench_base.synth import positive_mode
from steering.eval.axbench_base.synth.positive_mode import (
    _CONCEPT_SEED_OFFSET,
    _admit_positive_work_items,
    _PositiveCandidateSource,
    _PositiveTargetTracker,
    _prepare_positive_target_tracker,
    _resolve_stage1b_prefixes,
    _run_positive_target_refill_engine,
    _seed_has_feature,
    _write_training_parquet,
)
from steering.eval.axbench_base.synth.qc import (
    PrefixPercentileCache,
    compute_fc_percentiles,
    fc_percentile_on_prefix,
    output_fc_percentile_for_continuation,
)
from steering.eval.axbench_base.types import (
    Concept,
    SeedRow,
    TargetModelMetadata,
)
from steering.eval.axbench_base.validate import load_and_validate_concepts
from steering.eval.cluster_feature_contract import (
    ClusterFeatureTarget,
    load_feature_contract_sidecar,
)
from tests.axbench_base.stage1_protocol_helpers import frame_stage1_response

_RUN_TAG = "l22_f65k_clustered_std50"
_FEATURE_REL = Path(
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos"
)
_RUN_REL = Path("nanochat_cache/eval/axbench_base") / _RUN_TAG
_EXPECTED_CONCEPTS = 50
_EXPECTED_TRAIN_ROWS = 23_953
_EXPECTED_LEGACY_SELECTIONS = 3_600
_DIAGNOSTIC_CONCEPTS = 10
_DIAGNOSTIC_REPLACEMENTS = 8


def _draw_next_candidate(
    source: _PositiveCandidateSource,
) -> positive_mode._SelectedPositiveSeed | None:
    selected = source.prepare_next()
    if selected is not None:
        source.commit_prepared(selected)
    return selected


def _select_initial_seeds_for_concept(
    concept: Concept,
    seeds: list[SeedRow],
    cfg: AxBenchBaseConfig,
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    *,
    sae_layer: int,
    prefix_cache: PrefixPercentileCache,
    cluster_target: ClusterFeatureTarget,
    effective_prefixes: tuple[str | None, ...],
) -> list[Any]:
    """Expose the exact production tracker admission for the real audit."""
    tracker = _prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=sae_layer,
        prefix_cache=prefix_cache,
        cluster_target=cluster_target,
        effective_prefixes=effective_prefixes,
    )
    works, _next_order = _admit_positive_work_items(
        tracker,
        start_order=0,
    )
    assert len(works) == cfg.n_positives_per_concept
    return [work.pending.selected for work in works]


def _candidate_real_roots() -> list[Path]:
    configured = os.environ.get("AXBENCH_1049_REAL_ROOT")
    if configured:
        return [Path(configured).expanduser().resolve()]

    checkout = Path(__file__).resolve().parents[3]
    candidates = [checkout]
    # A Codex worktree lives at <shared-root>/.codex/worktrees/<name>.
    if checkout.parent.name == "worktrees" and checkout.parent.parent.name == ".codex":
        candidates.append(checkout.parents[2])
    return candidates


def _resolve_real_artifacts() -> tuple[Path, Path, Path]:
    attempted: list[Path] = []
    for root in _candidate_real_roots():
        feature_dir = root / _FEATURE_REL
        run_dir = root / _RUN_REL
        required = (
            feature_dir / "feature_statistics.safetensors",
            run_dir / "seeds_train.safetensors",
            run_dir / "seed_pool_manifest.json",
            run_dir / "concepts.json",
            run_dir / "feature_contract.json",
        )
        attempted.extend(required)
        if all(path.is_file() for path in required):
            return root, feature_dir, run_dir

    if os.environ.get("AXBENCH_1049_REAL_ROOT"):
        pytest.fail(f"Issue #1049 real artifacts are incomplete: {attempted}")
    pytest.skip(
        "set AXBENCH_1049_REAL_ROOT to a checkout containing the audited "
        "L22/f65k feature and Stage 0b artifacts"
    )


def _cluster_targets(
    concepts: list[Concept],
    sidecar_path: Path,
) -> dict[int, ClusterFeatureTarget]:
    document = load_feature_contract_sidecar(sidecar_path)
    semantic = document["semantic"]
    feature_clusters = semantic["feature_clusters"]
    raw_mapping = feature_clusters["representative_to_allowed_members"]
    members_by_representative = {
        int(representative): tuple(int(value) for value in members)
        for representative, members in raw_mapping
    }
    fingerprint = str(document["feature_contract_fingerprint"])

    assert len(members_by_representative) == len(feature_clusters["representative_universe"])
    targets: dict[int, ClusterFeatureTarget] = {}
    for concept in concepts:
        members = members_by_representative.get(concept.feature_index)
        assert members is not None, (
            f"concept feature {concept.feature_index} is not a cluster representative"
        )
        targets[concept.concept_id] = ClusterFeatureTarget(
            representative_id=concept.feature_index,
            allowed_member_ids=members,
            projecting_source_ids=(),
            feature_contract_fingerprint=fingerprint,
        )
    return targets


def _legacy_clean_selection(
    concept: Concept,
    seeds: list[SeedRow],
    target: ClusterFeatureTarget,
    cfg: AxBenchBaseConfig,
) -> list[int]:
    """Reproduce the exact pre-#1049 clean-first selection order.

    The audited production pool must have a full clean quota for every
    concept. Failing here is intentional: a dirty backfill would require the
    old selector's model-scored ordering and would invalidate the simple,
    independently implemented legacy oracle below.
    """

    indices = list(range(len(seeds)))
    rng = random.Random(cfg.generation.master_seed + _CONCEPT_SEED_OFFSET + concept.concept_id)
    rng.shuffle(indices)
    clean = [
        index
        for index in indices
        if not any(
            _seed_has_feature(seeds[index], member_id) for member_id in target.allowed_member_ids
        )
    ]
    assert len(clean) >= cfg.n_positives_per_concept, (
        f"legacy audit needs dirty backfill for concept={concept.concept_id}: "
        f"clean={len(clean)}, required={cfg.n_positives_per_concept}"
    )
    return clean[: cfg.n_positives_per_concept]


def _assert_raw_percentile(value: float) -> None:
    assert math.isfinite(value)
    assert value == 0.0 or 0.5 <= value <= 99.5


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
def test_issue_1049_full_real_clustered_prefix_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Audit legacy and corrected selection on the full real production pool."""

    if not t.cuda.is_available():
        pytest.fail("Issue #1049 real acceptance requires a CUDA GPU")

    root, feature_dir, run_dir = _resolve_real_artifacts()
    statistics_path = feature_dir / "feature_statistics.safetensors"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(root / "nanochat_cache"))

    cfg = AxBenchBaseConfig(
        feature_dir=str(feature_dir),
        feature_statistics_path=str(statistics_path),
        decoder_only={"source": "base", "model_tag": "d12_base", "step": 54_600},
        encdec={
            "source": "encdec",
            "model_tag": "l22_main_h100_8rank_db16_v1",
            "step": 54_600,
        },
        device="cuda",
        output_root=str(tmp_path / "unused-output"),
        run_tag=_RUN_TAG,
        neuronpedia_model_id="gemma-3-4b",
        neuronpedia_source="22-gemmascope-2-res-65k",
        feature_extraction_model_id="gemma-3-4b-pt",
        extraction_tokenizer_id="google/gemma-3-4b-pt",
        n_positives_per_concept=72,
        accept_threshold=0.3,
        positive_prefix_headroom_ceiling=0.9,
        min_prefix_chars=100,
        max_prefix_chars=2_048,
    )
    validate_config(cfg)
    thresholds = derive_positive_qc_thresholds(
        cfg.accept_threshold,
        cfg.positive_prefix_headroom_ceiling,
    )
    assert thresholds.required_delta == 30.0
    assert thresholds.completion_ceiling == 90.0
    assert thresholds.maximum_seed_percentile == 60.0

    concepts = load_and_validate_concepts(
        run_dir / "concepts.json",
        n_sae_features=65_536,
    )
    seeds = load_seed_pool(
        run_dir / "seeds_train.safetensors",
        expected_split="train",
    )
    assert len(concepts) == _EXPECTED_CONCEPTS
    assert len(seeds) == _EXPECTED_TRAIN_ROWS
    targets = _cluster_targets(concepts, run_dir / "feature_contract.json")

    legacy_by_concept = {
        concept.concept_id: _legacy_clean_selection(
            concept,
            seeds,
            targets[concept.concept_id],
            cfg,
        )
        for concept in concepts
    }
    assert sum(map(len, legacy_by_concept.values())) == (_EXPECTED_LEGACY_SELECTIONS)

    state = RuntimeState(cfg=cfg)
    audit_rows: list[dict[str, Any]] = []
    scalar_batch_drift: list[float] = []
    scalar_cache_drift: list[float] = []
    cluster_representative_drift: list[float] = []
    output_pair_drift: list[float] = []
    batch_shape_threshold_flips = 0
    try:
        with state.base_model() as base_model, state.base_sae() as base_sae:
            assert next(base_sae.parameters()).device.type == "cuda"
            assert int(base_sae.cfg.d_sae) == 65_536
            percentile_table = build_percentile_table_from_stats(
                str(statistics_path),
                n_sae_features=65_536,
            )
            assert tuple(percentile_table.shape) == (65_536, 99)
            legacy_cache = PrefixPercentileCache(
                base_model=base_model,
                base_sae=base_sae,
                percentile_table=percentile_table,
                sae_layer=22,
                tokenizer_id=cfg.extraction_tokenizer_id,
                percentile_namespace=str(statistics_path),
                batch_size=cfg.qc_batch_size,
            )
            production_cache = PrefixPercentileCache(
                base_model=base_model,
                base_sae=base_sae,
                percentile_table=percentile_table,
                sae_layer=22,
                tokenizer_id=cfg.extraction_tokenizer_id,
                percentile_namespace=str(statistics_path),
                batch_size=cfg.qc_batch_size,
            )
            repeat_cache = PrefixPercentileCache(
                base_model=base_model,
                base_sae=base_sae,
                percentile_table=percentile_table,
                sae_layer=22,
                tokenizer_id=cfg.extraction_tokenizer_id,
                percentile_namespace=str(statistics_path),
                batch_size=cfg.qc_batch_size,
            )
            scalar_cache = PrefixPercentileCache(
                base_model=base_model,
                base_sae=base_sae,
                percentile_table=percentile_table,
                sae_layer=22,
                tokenizer_id=cfg.extraction_tokenizer_id,
                percentile_namespace=str(statistics_path),
                batch_size=1,
            )
            effective_prefixes = _resolve_stage1b_prefixes(cfg, seeds)

            for concept_position, concept in enumerate(concepts):
                target = targets[concept.concept_id]
                legacy_indices = legacy_by_concept[concept.concept_id]
                prepared_by_index = {
                    index: effective_prefixes[index] for index in legacy_indices
                }
                too_short = [
                    index for index, prepared in prepared_by_index.items() if prepared is None
                ]
                scoreable_indices = [
                    index for index in legacy_indices if prepared_by_index[index] is not None
                ]
                prepared_texts = [prepared_by_index[index] for index in scoreable_indices]
                assert all(isinstance(text, str) for text in prepared_texts)
                scores = legacy_cache.get_many(
                    [str(text) for text in prepared_texts],
                    concept.feature_index,
                    cluster_target=target,
                )
                assert len(scores) == len(scoreable_indices)
                score_by_index = dict(zip(scoreable_indices, scores, strict=True))
                for score in scores:
                    _assert_raw_percentile(score)

                selected = _select_initial_seeds_for_concept(
                    concept,
                    seeds,
                    cfg,
                    base_model,
                    base_sae,
                    percentile_table,
                    sae_layer=22,
                    prefix_cache=production_cache,
                    cluster_target=target,
                    effective_prefixes=effective_prefixes,
                )
                assert len(selected) == cfg.n_positives_per_concept
                assert all(
                    item.seed_percentile <= thresholds.maximum_seed_percentile for item in selected
                )
                assert all(
                    item.prefix_text == effective_prefixes[item.seed_idx]
                    for item in selected
                )

                legacy_set = set(legacy_indices)
                selected_set = {item.seed_idx for item in selected}
                assert len(selected_set) == len(selected)
                rejected_legacy = {
                    index
                    for index, score in score_by_index.items()
                    if score > thresholds.maximum_seed_percentile
                }
                assert len(selected_set - legacy_set) == len(legacy_set - selected_set)

                if concept_position < _DIAGNOSTIC_CONCEPTS:
                    source = _PositiveCandidateSource(
                        concept,
                        seeds,
                        cfg,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        prefix_cache=production_cache,
                        cluster_target=target,
                        effective_prefixes=effective_prefixes,
                    )
                    source.preflight_initial_target(
                        cfg.n_positives_per_concept
                    )
                    source_initial = [
                        _draw_next_candidate(source)
                        for _ in range(cfg.n_positives_per_concept)
                    ]
                    assert all(item is not None for item in source_initial)
                    assert [
                        (
                            item.seed_idx,
                            item.prefix_text,
                            item.seed_percentile,
                            item.stored_clean,
                        )
                        for item in source_initial
                        if item is not None
                    ] == [
                        (
                            item.seed_idx,
                            item.prefix_text,
                            item.seed_percentile,
                            item.stored_clean,
                        )
                        for item in selected
                    ]
                    replacements = [
                        _draw_next_candidate(source)
                        for _ in range(_DIAGNOSTIC_REPLACEMENTS)
                    ]
                    assert all(item is not None for item in replacements)
                    assert all(
                        item.seed_percentile
                        <= thresholds.maximum_seed_percentile
                        for item in replacements
                        if item is not None
                    )
                    source_seed_indices = [
                        item.seed_idx
                        for item in [*source_initial, *replacements]
                        if item is not None
                    ]
                    assert len(source_seed_indices) == len(
                        set(source_seed_indices)
                    )

                    repeated = _select_initial_seeds_for_concept(
                        concept,
                        seeds,
                        cfg,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        prefix_cache=repeat_cache,
                        cluster_target=target,
                        effective_prefixes=effective_prefixes,
                    )
                    assert [
                        (
                            item.seed_idx,
                            item.prefix_text,
                            item.seed_percentile,
                            item.stored_clean,
                        )
                        for item in repeated
                    ] == [
                        (
                            item.seed_idx,
                            item.prefix_text,
                            item.seed_percentile,
                            item.stored_clean,
                        )
                        for item in selected
                    ]
                    diagnostic = max(
                        selected,
                        key=lambda item: item.seed_percentile,
                    )
                    scalar_cached = scalar_cache.get_many(
                        [diagnostic.prefix_text],
                        concept.feature_index,
                        cluster_target=target,
                    )[0]
                    scalar_cluster = fc_percentile_on_prefix(
                        diagnostic.prefix_text,
                        concept.feature_index,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                        cluster_target=target,
                    )
                    scalar_representative = fc_percentile_on_prefix(
                        diagnostic.prefix_text,
                        concept.feature_index,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                    )
                    scalar_cache_drift.append(abs(scalar_cluster - scalar_cached))
                    scalar_batch_drift.append(abs(scalar_cluster - diagnostic.seed_percentile))
                    batch_shape_threshold_flips += int(
                        (
                            scalar_cluster
                            <= thresholds.maximum_seed_percentile
                        )
                        != (
                            diagnostic.seed_percentile
                            <= thresholds.maximum_seed_percentile
                        )
                    )
                    cluster_representative_drift.append(scalar_cluster - scalar_representative)

                    continuation = diagnostic.seed.continuation_text
                    output_only = output_fc_percentile_for_continuation(
                        diagnostic.prefix_text,
                        continuation,
                        concept.feature_index,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                        cluster_target=target,
                    )
                    _joined_seed, pair_output = compute_fc_percentiles(
                        diagnostic.prefix_text,
                        continuation,
                        concept.feature_index,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                        cluster_target=target,
                    )
                    _assert_raw_percentile(output_only)
                    _assert_raw_percentile(pair_output)
                    output_pair_drift.append(abs(output_only - pair_output))

                prepared_lengths = [
                    len(str(prepared))
                    for prepared in prepared_by_index.values()
                    if prepared is not None
                ]
                audit_rows.append(
                    {
                        "concept_id": concept.concept_id,
                        "feature_index": concept.feature_index,
                        "legacy_selected": len(legacy_indices),
                        "legacy_too_short": len(too_short),
                        "legacy_headroom_rejected": len(rejected_legacy),
                        "selection_replacements": len(selected_set - legacy_set),
                        "selected_maximum_score": max(
                            item.seed_percentile for item in selected
                        ),
                        "minimum_legacy_score": (min(scores) if scores else None),
                        "maximum_legacy_score": (max(scores) if scores else None),
                        "minimum_prepared_chars": (
                            min(prepared_lengths) if prepared_lengths else None
                        ),
                        "maximum_prepared_chars": (
                            max(prepared_lengths) if prepared_lengths else None
                        ),
                    }
                )

            # Couple the real clustered candidate source to the production
            # complete-pair scheduler. The provider output and rich decision
            # are fixed test inputs: candidate rank 0 fails QC, while ranks 1
            # and 2 pass. Prefix scoring, headroom admission, cleanliness,
            # cluster targets, parsing, replacement, pair assembly, ordering,
            # and parquet serialization remain production paths.
            scheduler_concepts = concepts[:2]
            scheduler_cfg = cfg.replace(
                n_positives_per_concept=2,
                positive_max_attempts_per_seed=1,
                positive_zero_hit_bail_after=None,
                synth=SynthConfig(
                    requests_per_minute=0,
                    max_concurrent=1,
                ),
            )
            expected_rich_prefixes: dict[int, list[str]] = {}
            rejected_prefix_by_feature: dict[int, str] = {}
            for concept in scheduler_concepts:
                target = targets[concept.concept_id]
                oracle_source = _PositiveCandidateSource(
                    concept,
                    seeds,
                    scheduler_cfg,
                    base_model,
                    base_sae,
                    percentile_table,
                    sae_layer=22,
                    prefix_cache=production_cache,
                    cluster_target=target,
                    effective_prefixes=effective_prefixes,
                )
                oracle_source.preflight_initial_target(2)
                oracle_stream = [
                    _draw_next_candidate(oracle_source)
                    for _ in range(3)
                ]
                assert all(item is not None for item in oracle_stream)
                candidates = [
                    item for item in oracle_stream if item is not None
                ]
                assert len(candidates) == 3
                assert len(
                    {candidate.prefix_text for candidate in candidates}
                ) == 3
                assert all(
                    candidate.seed_percentile
                    <= thresholds.maximum_seed_percentile
                    for candidate in candidates
                )
                assert all(
                    candidate.stored_clean
                    == (
                        not any(
                            _seed_has_feature(
                                candidate.seed,
                                member_id,
                            )
                            for member_id in target.allowed_member_ids
                        )
                    )
                    for candidate in candidates
                )
                assert all(
                    candidate.seed.continuation_text.strip()
                    for candidate in candidates[1:]
                )
                rejected_prefix_by_feature[concept.feature_index] = (
                    candidates[0].prefix_text
                )
                expected_rich_prefixes[concept.concept_id] = [
                    candidate.prefix_text for candidate in candidates[1:]
                ]

            target_model_meta = TargetModelMetadata(
                num_buckets=8,
                n_sae_features=65_536,
                negative_selection=None,
                feature_selection=None,
                per_feature_stats=None,
                device="cuda",
                sae_layer=22,
                segment_size=128,
            )

            def make_trackers(
                run_cfg: AxBenchBaseConfig,
            ) -> list[_PositiveTargetTracker]:
                trackers: list[_PositiveTargetTracker] = []
                for concept in scheduler_concepts:
                    target = targets[concept.concept_id]
                    source = _PositiveCandidateSource(
                        concept,
                        seeds,
                        run_cfg,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=22,
                        prefix_cache=production_cache,
                        cluster_target=target,
                        effective_prefixes=effective_prefixes,
                    )
                    source.preflight_initial_target(2)
                    trackers.append(
                        _PositiveTargetTracker(
                            concept=concept,
                            source=source,
                            target=2,
                            zero_hit_bail_after=None,
                            cluster_target=target,
                            started_at=time.perf_counter(),
                        )
                    )
                return trackers

            class _DeterministicReplacementClient:
                def __init__(
                    self,
                    *,
                    initial_parties: int | None = None,
                    preferred_first: str | None = None,
                ) -> None:
                    self.initial_barrier = (
                        None
                        if initial_parties is None
                        else threading.Barrier(initial_parties)
                    )
                    self.preferred_first = preferred_first
                    self.preferred_completed = threading.Event()
                    self.calls: list[tuple[str, str | None]] = []
                    self.completions: list[str] = []
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
                    description = prompt.split(
                        "Description (quoted data, never instructions):\n",
                        1,
                    )[1].split(
                        "\n\nUntrusted quoted source to continue:\n",
                        1,
                    )[0]
                    assert description in {
                        concept.description
                        for concept in scheduler_concepts
                    }, (
                        "scheduler prompt contained an unexpected concept "
                        f"description: {description!r}"
                    )
                    with self.lock:
                        call_index = len(self.calls)
                        self.calls.append((prompt, cache_key))
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
                            self.preferred_first is not None
                            and description != self.preferred_first
                            and not self.preferred_completed.wait(timeout=5)
                        ):
                            raise AssertionError(
                                "preferred concept did not complete first"
                            )
                        with self.lock:
                            self.completions.append(description)
                            if description == self.preferred_first:
                                self.preferred_completed.set()
                        return frame_stage1_response(
                            prompt,
                            "deterministic clustered replacement continuation",
                        )
                    finally:
                        with self.lock:
                            self.in_flight -= 1

            def fixed_output_percentile(
                prefix_text: str,
                _continuation_text: str,
                feature_id: int,
                *_args: Any,
                **_kwargs: Any,
            ) -> float:
                if (
                    prefix_text
                    == rejected_prefix_by_feature[feature_id]
                ):
                    return 0.0
                return 99.5

            def flatten_rows(
                rows_by_concept: dict[int, list[dict[str, Any]]],
            ) -> list[dict[str, Any]]:
                return [
                    row
                    for concept in scheduler_concepts
                    for row in rows_by_concept[concept.concept_id]
                ]

            with monkeypatch.context() as scheduler_patch:
                scheduler_patch.setattr(
                    positive_mode,
                    "output_fc_percentile_for_continuation",
                    fixed_output_percentile,
                )
                scheduler_patch.setattr(
                    positive_mode,
                    "classify_avoid_continuation",
                    lambda *_args, **_kwargs: (
                        positive_mode.AvoidContinuationOutcome.INACTIVE
                    ),
                )
                serial_client = _DeterministicReplacementClient()
                serial_rows_by_concept = (
                    _run_positive_target_refill_engine(
                        scheduler_cfg,
                        make_trackers(scheduler_cfg),
                        base_model,
                        base_sae,
                        serial_client,
                        percentile_table,
                        target_model_meta,
                    )
                )
                parallel_cfg = scheduler_cfg.replace(
                    synth=SynthConfig(
                        requests_per_minute=0,
                        max_concurrent=4,
                    ),
                )
                parallel_client = _DeterministicReplacementClient(
                    initial_parties=4,
                    preferred_first=scheduler_concepts[1].description,
                )
                parallel_rows_by_concept = (
                    _run_positive_target_refill_engine(
                        parallel_cfg,
                        make_trackers(parallel_cfg),
                        base_model,
                        base_sae,
                        parallel_client,
                        percentile_table,
                        target_model_meta,
                    )
                )

            serial_rows = flatten_rows(serial_rows_by_concept)
            parallel_rows = flatten_rows(parallel_rows_by_concept)
            for concept in scheduler_concepts:
                rich_prefixes = [
                    row["prefix_text"]
                    for row in serial_rows_by_concept[concept.concept_id]
                    if row["content_class"] == "concept_rich"
                ]
                assert rich_prefixes == expected_rich_prefixes[
                    concept.concept_id
                ]
            assert len(serial_client.calls) == 6
            assert len(parallel_client.calls) == 6
            assert {cache_key for _prompt, cache_key in serial_client.calls} == {
                cache_key for _prompt, cache_key in parallel_client.calls
            }
            assert None not in {
                cache_key for _prompt, cache_key in serial_client.calls
            }
            assert serial_client.peak_in_flight == 1
            assert parallel_client.peak_in_flight > 1
            assert parallel_client.completions[0] == (
                scheduler_concepts[1].description
            )
            serial_parquet = tmp_path / "scheduler_serial.parquet"
            parallel_parquet = tmp_path / "scheduler_parallel.parquet"
            _write_training_parquet(
                serial_rows,
                serial_parquet,
                tokenizer_id=cfg.extraction_tokenizer_id,
                cfg=scheduler_cfg,
            )
            _write_training_parquet(
                parallel_rows,
                parallel_parquet,
                tokenizer_id=cfg.extraction_tokenizer_id,
                cfg=parallel_cfg,
            )
            assert serial_parquet.read_bytes() == parallel_parquet.read_bytes()
    finally:
        state.close()

    total_rejected = sum(row["legacy_headroom_rejected"] for row in audit_rows)
    total_too_short = sum(row["legacy_too_short"] for row in audit_rows)
    assert total_rejected > 0, (
        "real audit found no infeasible legacy prefix; the Issue #1049 "
        "regression fixture no longer reproduces the reported problem"
    )
    assert max(scalar_cache_drift, default=0.0) <= 1e-6
    assert max(output_pair_drift, default=0.0) <= 1e-6

    summary = {
        "real_root": str(root),
        "concepts": len(concepts),
        "train_rows": len(seeds),
        "legacy_selections": _EXPECTED_LEGACY_SELECTIONS,
        "legacy_headroom_rejected": total_rejected,
        "legacy_too_short": total_too_short,
        "rejection_rate": total_rejected / _EXPECTED_LEGACY_SELECTIONS,
        "maximum_scalar_batch_drift": max(
            scalar_batch_drift,
            default=0.0,
        ),
        "maximum_scalar_cache_drift": max(
            scalar_cache_drift,
            default=0.0,
        ),
        "batch_shape_threshold_flips": batch_shape_threshold_flips,
        "maximum_output_pair_drift": max(output_pair_drift, default=0.0),
        "maximum_cluster_representative_delta": max(
            cluster_representative_drift,
            key=abs,
            default=0.0,
        ),
        "concepts_detail": audit_rows,
    }
    audit_path = tmp_path / "issue_1049_real_gpu_audit.json"
    audit_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "Issue #1049 real GPU audit passed: concepts={}, train_rows={}, "
        "legacy_selections={}, rejected={} ({:.3%}), too_short={}, "
        "max_scalar_batch_drift={}, max_scalar_cache_drift={}, "
        "batch_shape_threshold_flips={}, max_output_pair_drift={}, audit={}",
        len(concepts),
        len(seeds),
        _EXPECTED_LEGACY_SELECTIONS,
        total_rejected,
        summary["rejection_rate"],
        total_too_short,
        summary["maximum_scalar_batch_drift"],
        summary["maximum_scalar_cache_drift"],
        summary["batch_shape_threshold_flips"],
        summary["maximum_output_pair_drift"],
        audit_path,
    )
