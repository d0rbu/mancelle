"""Real-GPU differential gate for Issue #1191 post-provider QC batching.

Run with the production Gemma-2/GemmaScope feature directory visible::

    AXBENCH_ISSUE_1191_FEATURE_DIR=/abs/path/to/\
      s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    uv run pytest \
      tests/axbench_base/integration/\
test_issue_1191_post_provider_qc_real_gpu.py \
      -v -m 'slow and requires_feature_data'

No provider is contacted. The test loads the production extraction pair and
compares only the representative/member SAE columns needed by each mixed
continuation target.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

from steering.eval.axbench_base.config import AxBenchBaseConfig, validate_config
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.extraction import (
    _infer_n_sae_features,
    build_percentile_table_from_stats,
)
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.synth import qc as synth_qc
from steering.eval.axbench_base.synth.qc import (
    BosPrependedActivations,
    ContinuationQcInput,
)
from steering.eval.cluster_feature_contract import ClusterFeatureTarget

_ATOL = 1e-3
_SINGLETON_ATOL = 1e-6
_RICH_DELTA_THRESHOLD = 30.0
_MEMORY_SLOP_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _Observation:
    selected_activations: t.Tensor
    output_percentile: float
    strict_zero_active: bool
    max_completion_activation: float
    completion_start: int
    completion_length: int


def _default_feature_dir() -> Path:
    relative = Path(
        "out/data/features/s128/l8/a_res/p_max_pool/f16k/"
        "m_gemma-2-2b/base"
    )
    test_root = Path(__file__).resolve().parents[3]
    candidates = [test_root / relative]
    # A linked Codex worktree stores the shared heavy artifacts in the main
    # repository, six parents above this file rather than inside the worktree.
    parents = Path(__file__).resolve().parents
    if len(parents) > 6:
        candidates.append(parents[6] / relative)
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _real_feature_dir() -> tuple[Path, bool]:
    configured = os.environ.get("AXBENCH_ISSUE_1191_FEATURE_DIR")
    if configured:
        return Path(configured).expanduser().resolve(), True
    return _default_feature_dir().resolve(), False


def _cfg(feature_dir: Path, tmp_path: Path) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir=str(feature_dir),
        feature_statistics_path=str(
            feature_dir / "feature_statistics.safetensors"
        ),
        decoder_only={
            "source": "base",
            "model_tag": "unused-issue-1191-gate",
            "step": 1,
        },
        encdec={
            "source": "encdec",
            "model_tag": "unused-issue-1191-gate",
            "step": 1,
        },
        device="cuda",
        output_root=str(tmp_path / "unused-output"),
        run_tag="issue-1191-real-gpu",
    )


def _cluster_target(feature_id: int, member_count: int) -> ClusterFeatureTarget:
    members = tuple(range(feature_id, feature_id + member_count))
    fingerprint = hashlib.sha256(
        ",".join(str(member) for member in members).encode("ascii")
    ).hexdigest()
    return ClusterFeatureTarget(
        representative_id=feature_id,
        allowed_member_ids=members,
        projecting_source_ids=members,
        feature_contract_fingerprint=fingerprint,
    )


def _qc_inputs(n_features: int) -> list[ContinuationQcInput]:
    text_pairs = [
        ("The engineer was run", "ning a calibration before sunrise."),
        ("A compact token", "ization boundary can alter the final subword."),
        ("The team studied international", "ization across several corpora."),
        ("This micro", "architecture uses a deliberately short seam."),
        ("A red kite crossed the field.", " It climbed above the old oak tree."),
        ("The experiment had one premise.", " Every result was checked twice."),
        ("At noon the harbor was quiet.", " By dusk, strong winds had arrived."),
        ("Researchers saved the first trace.", " The second trace was much longer and contained several additional observations."),
        ("A mathematician opened the notebook.", " She wrote a lemma, a proof sketch, and three counterexamples before closing it."),
        ("The small robot entered the room.", " Its camera mapped each wall while the control loop logged every correction."),
        ("One sentence is intentionally brief.", " Another continuation adds enough words to create a visibly ragged batch without approaching the context limit."),
        ("The archive contained weather reports.", " Some described snow, while others recorded dry heat and unusually persistent coastal fog."),
        ("A violinist tuned the lowest string.", " The orchestra waited until the pitch stabilized and then began the rehearsal."),
        ("The compiler accepted the source file.", " A later optimization pass removed redundant branches but retained observable behavior."),
        ("Several samples remained unlabeled.", " Reviewers independently assigned categories and reconciled the few disagreements."),
        ("The final prefix is the longest of the set because it exercises padding over a broader range of sequence lengths while remaining ordinary natural language.", " Its continuation confirms that completion slicing remains aligned after the ragged production forward."),
    ]
    requested_ids = [
        0,
        17,
        42,
        101,
        257,
        488,
        777,
        1024,
        2048,
        4096,
        6000,
        8192,
        10_000,
        12_000,
        14_000,
        15_000,
    ]
    member_counts = [1, 2, 3, 4] * 4
    inputs: list[ContinuationQcInput] = []
    for position, ((prefix, continuation), requested_id, member_count) in enumerate(
        zip(text_pairs, requested_ids, member_counts, strict=True)
    ):
        max_representative = n_features - member_count
        feature_id = min(requested_id, max_representative)
        target = (
            None
            if position % 4 == 0
            else _cluster_target(feature_id, member_count)
        )
        inputs.append(
            ContinuationQcInput(
                prefix_text=prefix,
                continuation_text=continuation,
                feature_id=feature_id,
                cluster_target=target,
            )
        )
    return inputs


def _members(item: ContinuationQcInput) -> tuple[int, ...]:
    if item.cluster_target is None:
        return (item.feature_id,)
    return item.cluster_target.allowed_member_ids


def _observe_batches(
    inputs: list[ContinuationQcInput],
    prepared: list[tuple[int, list[int]]],
    *,
    batch_size: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    sae_layer: int,
) -> tuple[list[_Observation], int, int]:
    observations: list[_Observation] = []
    t.cuda.reset_peak_memory_stats()
    for start in range(0, len(inputs), batch_size):
        item_chunk = inputs[start : start + batch_size]
        prepared_chunk = prepared[start : start + batch_size]
        acts_list = synth_qc._run_sae_on_token_lists(
            [tokens for _n_prefix, tokens in prepared_chunk],
            base_model,
            base_sae,
            sae_layer=sae_layer,
        )
        for item, (n_prefix, tokens), acts in zip(
            item_chunk,
            prepared_chunk,
            acts_list,
            strict=True,
        ):
            assert int(acts.shape[0]) == len(tokens)
            members = _members(item)
            selected = acts[:, list(members)]
            assert bool(t.isfinite(selected).all())
            spans = BosPrependedActivations.from_forward(
                acts,
                tokens,
                n_prefix=n_prefix,
            )
            completion = spans.completion()
            if item.cluster_target is None:
                output_percentile = synth_qc._max_percentile(
                    completion[:, item.feature_id],
                    item.feature_id,
                    percentile_table,
                )
            else:
                output_percentile = synth_qc._max_percentile_for_target(
                    completion,
                    item.cluster_target,
                    percentile_table,
                )
            completion_members = completion[:, list(members)]
            max_activation = (
                float(completion_members.max().item())
                if completion_members.numel()
                else 0.0
            )
            observations.append(
                _Observation(
                    selected_activations=selected.detach().float().cpu(),
                    output_percentile=float(output_percentile),
                    strict_zero_active=bool((completion_members > 0.0).any().item()),
                    max_completion_activation=max_activation,
                    completion_start=1 + n_prefix,
                    completion_length=int(completion.shape[0]),
                )
            )
        # Drop every loop-local CUDA view before sampling post-iteration
        # memory.  Deleting only the owning list would leave the final
        # iteration's ``acts``/completion views live in Python locals and make
        # the memory-stability assertion measure the test harness instead of
        # the production cleanup boundary.
        del completion_members, completion, spans, selected, acts
        del acts_list
        release_cuda_iteration_memory()
    peak_allocated = int(t.cuda.max_memory_allocated())
    peak_reserved = int(t.cuda.max_memory_reserved())
    return observations, peak_allocated, peak_reserved


@pytest.mark.slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_post_provider_qc_batch_matches_singleton_real_gpu(
    tmp_path: Path,
) -> None:
    feature_dir, explicitly_configured = _real_feature_dir()
    statistics_path = feature_dir / "feature_statistics.safetensors"
    if not feature_dir.is_dir() or not statistics_path.is_file():
        if explicitly_configured:
            pytest.fail(
                "Issue #1191 real-GPU gate requires a feature directory and "
                f"statistics file, got {feature_dir}"
            )
        pytest.skip(
            "set AXBENCH_ISSUE_1191_FEATURE_DIR to run the real-GPU gate"
        )

    cfg = _cfg(feature_dir, tmp_path)
    validate_config(cfg)
    state = RuntimeState(cfg=cfg)
    try:
        with state.base_model() as base_model:
            with state.base_sae() as base_sae:
                n_features = _infer_n_sae_features(base_sae)
                percentile_table = build_percentile_table_from_stats(
                    str(statistics_path),
                    n_sae_features=n_features,
                )
                inputs = _qc_inputs(n_features)
                prepared = []
                seam_merges = 0
                tokenizer = synth_qc.get_extraction_tokenizer(
                    cfg.extraction_tokenizer_id
                )
                for item in inputs:
                    prefix_tokens, full_tokens = synth_qc._tokenize_prefix_and_full(
                        item.prefix_text,
                        item.continuation_text,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                    )
                    continuation_tokens = tokenizer.encode(
                        item.continuation_text,
                        add_special_tokens=False,
                    )
                    seam_merges += int(
                        len(full_tokens) - 1
                        < len(prefix_tokens) + len(continuation_tokens)
                    )
                    prepared.append((len(prefix_tokens), full_tokens))
                assert seam_merges > 0
                assert len({len(tokens) for _n_prefix, tokens in prepared}) > 4

                singleton_started_at = time.perf_counter()
                singleton, singleton_peak_allocated, singleton_peak_reserved = (
                    _observe_batches(
                        inputs,
                        prepared,
                        batch_size=1,
                        base_model=base_model,
                        base_sae=base_sae,
                        percentile_table=percentile_table,
                        sae_layer=8,
                    )
                )
                seconds_by_batch: dict[int, float] = {
                    1: time.perf_counter() - singleton_started_at
                }
                memory_by_batch: dict[int, tuple[int, int]] = {
                    1: (singleton_peak_allocated, singleton_peak_reserved)
                }

                # Establish the singleton baseline against the production
                # scalar seams for every ragged item, not against a second run
                # of the same batch-size-one helper.
                scalar_percentiles: list[float] = []
                scalar_outcomes: list[synth_qc.AvoidContinuationOutcome] = []
                for position, (item, (_n_prefix, full_tokens)) in enumerate(
                    zip(inputs, prepared, strict=True)
                ):
                    scalar_acts = synth_qc._run_sae_on_full(
                        full_tokens,
                        base_model,
                        base_sae,
                        sae_layer=8,
                    )[:, list(_members(item))]
                    t.testing.assert_close(
                        scalar_acts.detach().float().cpu(),
                        singleton[position].selected_activations,
                        atol=_SINGLETON_ATOL,
                        rtol=0.0,
                    )
                    del scalar_acts
                    release_cuda_iteration_memory()
                    scalar_percentiles.append(
                        synth_qc.output_fc_percentile_for_continuation(
                            item.prefix_text,
                            item.continuation_text,
                            item.feature_id,
                            base_model,
                            base_sae,
                            percentile_table,
                            sae_layer=8,
                            tokenizer_id=cfg.extraction_tokenizer_id,
                            cluster_target=item.cluster_target,
                        )
                    )
                    scalar_outcomes.append(
                        synth_qc.classify_avoid_continuation(
                            item.prefix_text,
                            item.continuation_text,
                            item.feature_id,
                            base_model,
                            base_sae,
                            sae_layer=8,
                            tokenizer_id=cfg.extraction_tokenizer_id,
                            cluster_target=item.cluster_target,
                        )
                    )
                assert scalar_percentiles == [
                    observation.output_percentile for observation in singleton
                ]
                assert scalar_outcomes == [
                    (
                        synth_qc.AvoidContinuationOutcome.ACTIVE
                        if observation.strict_zero_active
                        else synth_qc.AvoidContinuationOutcome.INACTIVE
                    )
                    for observation in singleton
                ]

                seed_percentiles = [
                    synth_qc.fc_percentile_on_prefix(
                        item.prefix_text,
                        item.feature_id,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=8,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                        cluster_target=item.cluster_target,
                    )
                    for item in inputs
                ]

                all_drifts: list[float] = []
                rich_flip_distances: list[float] = []
                strict_flip_distances: list[float] = []
                activation_violations: list[str] = []
                activation_drift_by_batch: dict[int, dict[str, float]] = {}
                activation_drift_by_batch[1] = {
                    "full": 0.0,
                    "bos": 0.0,
                    "prefix": 0.0,
                    "completion": 0.0,
                }
                for batch_size in (2, 4, 8, 16):
                    started_at = time.perf_counter()
                    observed, peak_allocated, peak_reserved = _observe_batches(
                        inputs,
                        prepared,
                        batch_size=batch_size,
                        base_model=base_model,
                        base_sae=base_sae,
                        percentile_table=percentile_table,
                        sae_layer=8,
                    )
                    seconds_by_batch[batch_size] = time.perf_counter() - started_at
                    memory_by_batch[batch_size] = (
                        peak_allocated,
                        peak_reserved,
                    )
                    region_max = {
                        "full": 0.0,
                        "bos": 0.0,
                        "prefix": 0.0,
                        "completion": 0.0,
                    }
                    assert len(observed) == len(singleton)
                    for position, (reference, candidate) in enumerate(
                        zip(singleton, observed, strict=True)
                    ):
                        assert candidate.completion_start == reference.completion_start
                        assert candidate.completion_length == reference.completion_length
                        activation_diff = (
                            candidate.selected_activations
                            - reference.selected_activations
                        ).abs()
                        completion_start = reference.completion_start
                        per_region = {
                            "full": activation_diff,
                            "bos": activation_diff[:1],
                            "prefix": activation_diff[1:completion_start],
                            "completion": activation_diff[completion_start:],
                        }
                        for region, values in per_region.items():
                            maximum = (
                                float(values.max().item())
                                if values.numel()
                                else 0.0
                            )
                            region_max[region] = max(region_max[region], maximum)
                        # The production scalar checks above establish the
                        # full-tensor singleton reference. Stage 1
                        # post-provider QC owns only completion() rows: BOS is
                        # always excluded and the
                        # rich baseline comes from the separately scored prefix
                        # cache. Keep excluded-span drift in diagnostics while
                        # enforcing the repository tolerance on every value
                        # that can affect a continuation percentile or
                        # strict-zero decision.
                        gated_diff = per_region["completion"]
                        tolerance = _ATOL
                        gated_max = (
                            float(gated_diff.max().item())
                            if gated_diff.numel()
                            else 0.0
                        )
                        if gated_max > tolerance:
                            flat_index = int(gated_diff.argmax().item())
                            row = (
                                flat_index // int(gated_diff.shape[1])
                                + completion_start
                            )
                            column = flat_index % int(gated_diff.shape[1])
                            activation_violations.append(
                                "batch_size={} item={} row={} member_column={} "
                                "region={} max_abs_diff={:.6g}".format(
                                    batch_size,
                                    position,
                                    row,
                                    column,
                                    (
                                        "bos"
                                        if row == 0
                                        else (
                                            "prefix"
                                            if row < completion_start
                                            else "completion"
                                        )
                                    ),
                                    gated_max,
                                )
                            )
                        drift = abs(
                            candidate.output_percentile
                            - reference.output_percentile
                        )
                        all_drifts.append(drift)
                        reference_rich = (
                            reference.output_percentile
                            - seed_percentiles[position]
                            >= _RICH_DELTA_THRESHOLD
                        )
                        candidate_rich = (
                            candidate.output_percentile
                            - seed_percentiles[position]
                            >= _RICH_DELTA_THRESHOLD
                        )
                        if reference_rich != candidate_rich:
                            rich_flip_distances.append(
                                abs(
                                    reference.output_percentile
                                    - seed_percentiles[position]
                                    - _RICH_DELTA_THRESHOLD
                                )
                            )
                        if (
                            reference.strict_zero_active
                            != candidate.strict_zero_active
                        ):
                            strict_flip_distances.append(
                                abs(reference.max_completion_activation)
                            )
                        assert t.isfinite(
                            t.tensor(candidate.output_percentile)
                        )
                        assert 0.0 <= candidate.output_percentile <= 99.5
                    activation_drift_by_batch[batch_size] = region_max

                post_allocated: list[int] = []
                post_reserved: list[int] = []
                for _ in range(3):
                    _observe_batches(
                        inputs,
                        prepared,
                        batch_size=16,
                        base_model=base_model,
                        base_sae=base_sae,
                        percentile_table=percentile_table,
                        sae_layer=8,
                    )
                    post_allocated.append(int(t.cuda.memory_allocated()))
                    post_reserved.append(int(t.cuda.memory_reserved()))
                assert post_allocated[-1] <= (
                    post_allocated[0] + _MEMORY_SLOP_BYTES
                )
                assert post_reserved[-1] <= (
                    post_reserved[0] + _MEMORY_SLOP_BYTES
                )
    finally:
        state.close()

    drift_tensor = t.tensor(all_drifts, dtype=t.float64)
    logger.info(
        "Issue #1191 real-GPU diagnostics: max_percentile_drift={:.6g}, "
        "median_drift={:.6g}, p99_drift={:.6g}, rich_flips={}, "
        "rich_singleton_distances={}, strict_zero_flips={}, "
        "strict_singleton_distances={}, peak_cuda_bytes_by_batch={}, "
        "activation_drift_by_batch={}, seconds_by_batch={}, "
        "batch16_local_speedup={:.3f}x, post_allocated={}, post_reserved={}",
        float(drift_tensor.max().item()),
        float(drift_tensor.median().item()),
        float(t.quantile(drift_tensor, 0.99).item()),
        len(rich_flip_distances),
        rich_flip_distances,
        len(strict_flip_distances),
        strict_flip_distances,
        memory_by_batch,
        activation_drift_by_batch,
        seconds_by_batch,
        seconds_by_batch[1] / seconds_by_batch[16],
        post_allocated,
        post_reserved,
    )
    assert not activation_violations, (
        "Issue #1191 QC-owned activation tolerance violations (first 20): "
        + "; ".join(activation_violations[:20])
    )
