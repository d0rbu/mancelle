"""Model-backed candidate building entry point.

The pure Faster-GCG and GEPA candidate builders live in
``steering.eval.axbench_bpb_adv.optimize`` so their invariants are unit-testable
without loading target models. This module owns the durable stage wrapper.

Sign convention (pinned). There is NO neutral reference. Each candidate is
scored by teacher-forcing BOTH class-margin completions (``concept_rich`` and
``concept_avoiding``) under ONE steered intervention. The adversary degrades
steering by MINIMIZING the signed class margin ``primary_margin_y``. The search
objective minimized by Faster-GCG/GEPA is ``g = loss_contrast - loss_desired``
with gradient ``grad_contrast - grad_desired`` (see ``optimize`` module
docstring). Mode mapping: positive -> desired=rich, contrast=avoid; negative ->
desired=avoid, contrast=rich. ``g`` is exactly ``primary_margin_y``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import torch._dynamo
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from loguru import logger

from nanochat.atomic_io import atomic_write_parquet
from steering.conditional.feature_gcg.edit_primitives import EditableSpan
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.direction_scale import DirectionScalePreflight
from steering.eval.axbench_base.extraction import build_percentile_table_from_stats
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.sample_loader import load_raw_synth_rows
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import LoraParams, SteeringMode, TargetModel
from steering.eval.axbench_bpb_adv.artifact_identity import (
    assert_table_cluster_identity,
    bind_clustered_table,
    write_clustered_table_atomic,
)
from steering.eval.axbench_bpb_adv.attack_rows import (
    bpb_attack_sample_from_synth_row,
    pair_samples_into_prefixes,
)
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    require_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    AxBenchBpbAdvConfig,
    ContextSignalMode,
    GepaConfig,
    get_additive_direction_config,
    get_faster_gcg_config,
    get_gepa_config,
)
from steering.eval.axbench_bpb_adv.context import (
    project_fixed_context_segments,
)
from steering.eval.axbench_bpb_adv.editable_tokens import replace_editable_tokens
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    read_evaluation_population,
    require_rows_in_evaluation_population,
)
from steering.eval.axbench_bpb_adv.feature_diffmean_bags import (
    active_fair_feature_diffmean_modes,
    active_roofline_feature_diffmean_modes,
    build_feature_diffmean_bags_fair,
    build_feature_diffmean_bags_roofline,
    fair_feature_diffmean_methods_active,
    roofline_feature_diffmean_methods_active,
)
from steering.eval.axbench_bpb_adv.gcg_bags import (
    ClusteredWarmStartCache,
    active_fair_gcg_modes,
    active_roofline_gcg_modes,
    assert_sae_matches_encdec,
    build_clustered_warm_start_cache,
    build_ours_gcg_bags_fair,
    build_ours_gcg_bags_roofline,
    fair_gcg_methods_active,
    roofline_gcg_methods_active,
)
from steering.eval.axbench_bpb_adv.gcg_family import GcgFamily
from steering.eval.axbench_bpb_adv.optimize import (
    GradientBatchProvider,
    ObjectiveBatchFn,
    build_additive_direction_candidates,
    build_control_candidate,
    build_gepa_candidates,
    build_universal_gcg_candidates,
    concat_candidate_tables,
    objective_gradient,
    objective_value,
    parse_gepa_token_candidate,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import (
    ModelBackedDualCompletionScorer,
    _bpb_config_from_adv_config,
    _build_token_bytes,
    _concepts_from_entries,
    _preflight_direction_scales_for_rows,
    _read_concepts,
    _token_list,
    load_context_segments_by_prefix_hash,
    require_precomputed_bag_prompt_rows_cover_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.source_validation import (
    _filter_to_source_concepts,
    _read_source_concept_ids,
)
from steering.eval.axbench_bpb_adv.surrogate import (
    bpb_surrogate_embedding_weight,
    teacher_forced_bpb_gradient_batch,
    teacher_forced_bpb_loss_shared_prefix_kv_batch,
)
from steering.eval.axbench_bpb_adv.token_filter import build_allowed_token_ids
from steering.eval.axbench_bpb_adv.universal_gcg import concept_groups
from steering.eval.axbench_bpb_adv.universal_gcg_checkpoint import (
    UniversalGcgCheckpointStore,
    assert_target_checkpoints_unchanged,
    build_checkpoint_fingerprint,
    current_runtime_fingerprint,
    resolve_and_pin_target_checkpoints,
    warn_on_runtime_mismatch,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.source_validation import (
        BpbAdvSourceValidation,
    )
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

_MIN_DYNAMO_RECOMPILE_LIMIT = 256
_MIN_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT = 2_048


def concat_universal_trigger_tables(tables: Sequence[pa.Table]) -> pa.Table:
    """Concat per-target universal-trigger tables into one (schema-stable).

    Each target's ``build_universal_gcg_candidates`` call yields one
    UNIVERSAL_TRIGGERS_SCHEMA-shaped table (one row per non-control concept
    group). The candidate stage concatenates them across targets before the
    single ``paths.universal_triggers_path()`` write. An all-empty input (no
    non-control faster_gcg groups on any target) returns the canonical empty
    table so the artifact always serializes with a stable schema. Mirrors
    ``optimize.concat_candidate_tables``.
    """

    non_empty = [table for table in tables if table.num_rows > 0]
    if not non_empty:
        return pa.Table.from_pylist([], schema=UNIVERSAL_TRIGGERS_SCHEMA)
    return pa.concat_tables(non_empty, promote_options="default")


@dataclass(frozen=True)
class _EvalCacheEntry:
    """A per-(attack-row, token-digest) evaluation cache entry.

    ``loss_desired``/``loss_contrast`` are ALWAYS present (the loss-only path
    populates them; see OPT #2). The gradients are present only for a FULL entry
    (the fwd+bwd path); a loss-only entry leaves them ``None``. ``objective()``
    serves losses from either entry kind; ``gradient_provider_batch()`` only
    serves FULL entries and otherwise recomputes -- so a gradient is NEVER
    served stale or from a loss-only entry, while the byte-identical losses are
    shared.
    """

    loss_desired: float
    loss_contrast: float
    grad_desired: t.Tensor | None = None
    grad_contrast: t.Tensor | None = None


@dataclass
class BpbObjectiveCallbacks:
    """Class-margin BPB objective callbacks for Faster-GCG / GEPA.

    Each candidate is scored by teacher-forcing BOTH class-margin completions
    under ONE steered intervention (no neutral reference). ``_evaluate`` returns
    ``(loss_desired, loss_contrast, grad_desired, grad_contrast)`` mapped by the
    candidate's steering mode (positive -> desired=rich, contrast=avoid;
    negative -> desired=avoid, contrast=rich); the search objective ``g`` is
    minimized (see module docstring).
    """

    target_bundle: TargetModelBundle
    target_model_meta: Any
    tokenizer: Any
    token_bytes: t.Tensor
    source_paths: SourcePaths
    pair_scorer: ModelBackedDualCompletionScorer
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]]
    device: t.device
    lora: LoraParams | None = None
    # Microbatch size for the batched candidate-eval forward (bounds live
    # activations; no_grad keeps it cheap). Faster-GCG may override this without
    # changing gradient/GEPA/scoring batch composition.
    device_batch_size: int = 32
    candidate_eval_device_batch_size: int | None = None

    def __post_init__(self) -> None:
        self._embedding_weight = bpb_surrogate_embedding_weight(self.target_bundle.model)
        self._cache: dict[tuple[str, str], _EvalCacheEntry] = {}

    def embedding_weight(self) -> t.Tensor:
        return self._embedding_weight

    def clear_eval_cache(self) -> None:
        """Evict every cached evaluation (the per-GROUP eviction bound, v2 C3).

        The universal optimizer interleaves a concept group's pairs within a
        single GCG step, so the legacy per-``attack_row_id`` clear (which assumed
        one row was optimised at a time) would thrash the cache to uselessness.
        The cache is keyed by ``(attack_row_id, token_digest)`` and is NEVER
        cleared per row; instead the builder calls this at each GROUP boundary,
        so the working set is bounded by one concept group's distinct
        (pair, trigger-substituted-prompt) evaluations.
        """

        self._cache.clear()

    def objective(self, row: dict[str, Any], tokens: Sequence[int]) -> float:
        # A batch of one: the candidate ranking + accept gate go through
        # objective_batch, so a lone call shares that exact (batched) path.
        return self.objective_batch([(row, tokens)])[0]

    def objective_batch(self, items: Sequence[tuple[dict[str, Any], Sequence[int]]]) -> list[float]:
        return self._objective_batch(
            items,
            device_batch_size=(
                self.device_batch_size
                if self.candidate_eval_device_batch_size is None
                else self.candidate_eval_device_batch_size
            ),
        )

    def gepa_objective(
        self,
        row: dict[str, Any],
        tokens: Sequence[int],
    ) -> float:
        """Shared-prefix GEPA objective using the top-level device batch size."""

        return self._objective_batch(
            [(row, tokens)],
            device_batch_size=self.device_batch_size,
        )[0]

    def _objective_batch(
        self,
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        *,
        device_batch_size: int,
    ) -> list[float]:
        """Batched class-margin objective for many (row, tokens) at once.

        Loss-only (candidate ranking needs no gradient). Misses are grouped by
        their pair's intervention identity and scored by the single production
        shared-prefix KV path. Faster-GCG candidate/current-target comparisons and
        GEPA use this same scorer.
        """

        results: list[float | None] = [None] * len(items)
        misses: list[int] = []
        for index, (row, tokens) in enumerate(items):
            cached = self._cache.get(self._cache_key(row, tokens))
            if cached is not None:
                results[index] = objective_value(
                    loss_desired=cached.loss_desired,
                    loss_contrast=cached.loss_contrast,
                )
            else:
                misses.append(index)

        # Group misses by intervention identity (one padded forward per concept-group
        # PAIR, holding that pair's candidate triggers' rich + avoid completions).
        pairs: dict[tuple[Any, ...], list[int]] = {}
        for index in misses:
            pairs.setdefault(self._pair_intervention_key(items[index][0]), []).append(index)

        for indices in pairs.values():
            self._score_pair_batch(
                items,
                indices,
                results,
                device_batch_size=device_batch_size,
            )

        scored: list[float] = []
        for value in results:
            if value is None:
                raise AxBenchBpbAdvError("internal error: objective_batch left an item unscored")
            scored.append(float(value))
        return scored

    def _score_pair_batch(
        self,
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        indices: list[int],
        results: list[float | None],
        *,
        device_batch_size: int,
    ) -> None:
        """Score one pair's candidate microbatch (rich+avoid per item) in a single
        padded forward under the pair's shared intervention; fill + cache
        ``results[i]`` for every ``i`` in ``indices``."""

        intervention = self._steered_intervention_for(items[indices[0]][0])
        prompts: list[list[int]] = []
        outputs: list[list[int]] = []
        segments: list[tuple[ContextFeatureSegment, ...]] = []
        source_prompts: list[list[int]] = []
        editable_spans: list[EditableSpan] = []
        per_item: list[tuple[int, dict[str, Any], Sequence[int], SteeringMode]] = []
        for index in indices:
            row, tokens = items[index]
            (
                prompt_tokens,
                rich_output,
                avoid_output,
                prefix_segments,
                steering_mode,
            ) = self._item_io(row, tokens)
            # rich then avoid, contiguous, so the result maps back as 2k / 2k+1.
            prompts.append(prompt_tokens)
            outputs.append(rich_output)
            segments.append(prefix_segments)
            prompts.append(prompt_tokens)
            outputs.append(avoid_output)
            segments.append(prefix_segments)
            source_prompt = _token_list(
                row["source_prompt_tokens"], field_name="source_prompt_tokens"
            )
            editable_span = EditableSpan(
                start=int(row["editable_token_start"]),
                count=int(row["editable_token_count"]),
            )
            source_prompts.extend((source_prompt, source_prompt))
            editable_spans.extend((editable_span, editable_span))
            per_item.append((index, row, tokens, steering_mode))

        losses = teacher_forced_bpb_loss_shared_prefix_kv_batch(
            target_bundle=self.target_bundle,
            prompts=prompts,
            source_prompts=source_prompts,
            outputs=outputs,
            prefix_context_segments_list=segments,
            editable_spans=editable_spans,
            output_token_bytes=self.token_bytes,
            intervention=intervention,
            device=self.device,
            lora=self.lora,
            device_batch_size=PositiveInt.parse(device_batch_size),
        )
        for offset, (index, row, tokens, steering_mode) in enumerate(per_item):
            loss_desired, loss_contrast = self._desired_contrast(
                steering_mode, losses[2 * offset], losses[2 * offset + 1]
            )
            self._cache[self._cache_key(row, tokens)] = _EvalCacheEntry(
                loss_desired=float(loss_desired),
                loss_contrast=float(loss_contrast),
            )
            results[index] = objective_value(
                loss_desired=float(loss_desired),
                loss_contrast=float(loss_contrast),
            )

    def gradient_provider_batch(
        self,
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        _step: int,
    ) -> list[t.Tensor]:
        """Batched per-pair objective gradient for the universal step's minibatch.

        Mirrors :meth:`objective_batch` for the gradient: serves cached FULL
        (with-gradient) entries, groups the misses by pair intervention identity,
        and computes each group's rich+avoid gradients in ONE fwd+bwd via
        ``teacher_forced_bpb_gradient_batch`` -- replacing ``2 * n_pairs``
        serialized single-prompt fwd+bwd, the dominant per-step cost. Returns one
        ``objective_gradient`` tensor per item (``grad_contrast - grad_desired``);
        ``optimize.aggregate_trigger_gradient`` then slices each to the trigger span
        and means. The aggregation is UNCHANGED, so the batched per-pair gradient is
        the only new surface (pinned byte-identical to the single path on an exact
        model by test_surrogate_gradient_batch; within bf16 tolerance on the real
        model by test_gcg_exact_speedup_integration).
        """

        results: list[t.Tensor | None] = [None] * len(items)
        misses: list[int] = []
        for index, (row, tokens) in enumerate(items):
            cached = self._cache.get(self._cache_key(row, tokens))
            if (
                cached is not None
                and cached.grad_desired is not None
                and cached.grad_contrast is not None
            ):
                results[index] = objective_gradient(
                    grad_loss_desired=cached.grad_desired,
                    grad_loss_contrast=cached.grad_contrast,
                )
            else:
                misses.append(index)

        loss_misses = [
            items[index]
            for index in misses
            if self._cache.get(
                self._cache_key(items[index][0], items[index][1])
            )
            is None
        ]
        if loss_misses:
            # A direct gradient-first call has no objective-facing cache entry.
            # Establish it with the production loss regime before attaching the
            # deliberately uncached gradients; never seed an objective entry with
            # the grad-enabled forward's loss.
            self._objective_batch(
                loss_misses,
                device_batch_size=(
                    self.device_batch_size
                    if self.candidate_eval_device_batch_size is None
                    else self.candidate_eval_device_batch_size
                ),
            )

        # Same grouping as objective_batch: one batched fwd+bwd per concept-group
        # pair (its intervention is shared across the pair's rich+avoid).
        pairs: dict[tuple[Any, ...], list[int]] = {}
        for index in misses:
            pairs.setdefault(self._pair_intervention_key(items[index][0]), []).append(index)

        for indices in pairs.values():
            self._grad_pair_batch(items, indices, results)

        grads: list[t.Tensor] = []
        for value in results:
            if value is None:
                raise AxBenchBpbAdvError(
                    "internal error: gradient_provider_batch left an item ungraded"
                )
            grads.append(value)
        return grads

    def _grad_pair_batch(
        self,
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        indices: list[int],
        results: list[t.Tensor | None],
    ) -> None:
        """One pair's rich+avoid gradients in a single batched fwd+bwd under the
        pair's shared intervention; cache the FULL entry and fill ``results[i]``
        with the per-pair objective gradient for every ``i`` in ``indices``."""

        intervention = self._steered_intervention_for(items[indices[0]][0])
        prompts: list[list[int]] = []
        outputs: list[list[int]] = []
        segments: list[tuple[ContextFeatureSegment, ...]] = []
        per_item: list[tuple[int, dict[str, Any], Sequence[int], SteeringMode]] = []
        for index in indices:
            row, tokens = items[index]
            (
                prompt_tokens,
                rich_output,
                avoid_output,
                prefix_segments,
                mode,
            ) = self._item_io(row, tokens)
            # rich then avoid, contiguous, so the result maps back as 2k / 2k+1.
            prompts.append(prompt_tokens)
            outputs.append(rich_output)
            segments.append(prefix_segments)
            prompts.append(prompt_tokens)
            outputs.append(avoid_output)
            segments.append(prefix_segments)
            per_item.append((index, row, tokens, mode))

        grad_results = teacher_forced_bpb_gradient_batch(
            target_bundle=self.target_bundle,
            prompts=prompts,
            outputs=outputs,
            prefix_context_segments_list=segments,
            output_token_bytes=self.token_bytes,
            intervention=intervention,
            device=self.device,
            lora=self.lora,
            embedding_weight=self._embedding_weight,
            device_batch_size=PositiveInt.parse(self.device_batch_size),
        )
        for offset, (index, row, tokens, mode) in enumerate(per_item):
            desired, contrast = self._desired_contrast(
                mode, grad_results[2 * offset], grad_results[2 * offset + 1]
            )
            key = self._cache_key(row, tokens)
            # Attach the gradients (which MUST come from the grad-enabled forward),
            # but PRESERVE the objective-facing losses in the no_grad regime the
            # ranking path established. The candidate ranking + the accept gate
            # compare via the no_grad shared-prefix scorer; serving this
            # gradient forward's grad-enabled losses for the SAME comparison would
            # straddle two reduction regimes that diverge ~1e-1 on the compiled
            # flex-attention model (no_grad vs grad-enabled; see surrogate #510),
            # biasing accept/reject. The current trigger's no_grad losses are always
            # cached first (init / on-accept ``_group_objective`` over the full
            # eval_batch); a missing entry here is an internal contract failure.
            existing = self._cache.get(key)
            if existing is None:
                raise AxBenchBpbAdvError(
                    "gradient path is missing its shared-prefix objective cache entry"
                )
            self._cache[key] = _EvalCacheEntry(
                loss_desired=existing.loss_desired,
                loss_contrast=existing.loss_contrast,
                grad_desired=desired.token_gradient,
                grad_contrast=contrast.token_gradient,
            )
            results[index] = objective_gradient(
                grad_loss_desired=desired.token_gradient,
                grad_loss_contrast=contrast.token_gradient,
            )

    def _cache_key(
        self,
        row: dict[str, Any],
        tokens: Sequence[int],
    ) -> tuple[str, str]:
        # Distinct pairs never collide even on identical prompt tokens. Entries
        # persist across interleaved minibatch evals (the per-row clear is gone;
        # the builder evicts per group via clear_eval_cache). See v2 C3.
        digest = "|".join(str(int(token_id)) for token_id in tokens)
        return (str(row["attack_row_id"]), digest)

    @staticmethod
    def _pair_intervention_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, str]:
        """Identity of the ONE shared steered intervention a batched forward spans.

        Items sharing this key are scored (``_score_pair_batch``) or
        gradient-evaluated (``_grad_pair_batch``) together in a single padded forward
        under ONE residual hook, so the key MUST be a superset of the fields
        ``ModelBackedDualCompletionScorer._base_steered_intervention`` seeds its RNG
        on (``method_name``, ``concept_id``, ``core_prefix_id``, ``coefficient_str``)
        -- plus ``target_model`` (which selects the target bundle + metadata). Mixing
        two distinct interventions into one forward would silently corrupt every
        sample's loss/gradient. This is the SINGLE source for that key (both batch
        methods call it), mirroring the canonical ``universal_gcg.group_key``
        single-source discipline for the coarser concept-group key.
        """

        return (
            str(row["method_name"]),
            int(row["concept_id"]),
            str(row["core_prefix_id"]),
            str(row["coefficient_str"]),
            str(row["target_model"]),
        )

    def _steered_intervention_for(self, row: dict[str, Any]) -> Any:
        """The base steered intervention for ``row`` (method/concept/coefficient +
        the per-prefix RNG seed). Shared by every candidate trigger of the SAME
        concept-group pair, so the batched objective builds it once per pair and
        reuses it across that pair's candidate microbatch."""

        method = get_method_by_name(str(row["method_name"]))
        concept = self.pair_scorer.concepts_by_id.get(int(row["concept_id"]))
        if concept is None:
            raise AxBenchBpbAdvError(
                f"source concepts do not contain concept_id={row['concept_id']}"
            )
        return self.pair_scorer._base_steered_intervention(
            method=method,
            concept=concept,
            coefficient=self._coefficient(row["coefficient_str"]),
            candidate_row=row,
        )

    def _item_io(
        self,
        row: dict[str, Any],
        tokens: Sequence[int],
    ) -> tuple[list[int], list[int], list[int], tuple[ContextFeatureSegment, ...], SteeringMode]:
        """The per-(row, tokens) IO the loss eval needs, WITHOUT the intervention
        (which is pair-shared and built once by the batched path)."""

        prompt_tokens = [int(token_id) for token_id in tokens]
        rich_output = _token_list(
            row["concept_rich_output_tokens"],
            field_name="concept_rich_output_tokens",
        )
        avoid_output = _token_list(
            row["concept_avoiding_output_tokens"],
            field_name="concept_avoiding_output_tokens",
        )
        # Context conditioning depends only on the attacked PROMPT (encdec
        # conditions at <assistant_start>, never on the completion), so compute
        # once and reuse for both completions.
        prefix_segments = self._context_segments(row, prompt_tokens)
        mode = SteeringMode(str(row["mode"]))
        return prompt_tokens, rich_output, avoid_output, prefix_segments, mode

    @staticmethod
    def _desired_contrast(
        mode: SteeringMode,
        rich: Any,
        avoid: Any,
    ) -> tuple[Any, Any]:
        # positive -> desired=rich, contrast=avoid; negative flips the pair.
        if mode == SteeringMode.POSITIVE:
            return rich, avoid
        return avoid, rich

    def _context_segments(
        self,
        row: Mapping[str, Any],
        attacked_tokens: list[int],
    ) -> tuple[ContextFeatureSegment, ...]:
        target_model = str(row["target_model"])
        if target_model == TargetModel.DECODER_ONLY.value:
            return ()
        source_segments = self.context_segments_by_prefix_hash.get(str(row["prefix_hash"]))
        if source_segments is None:
            raise AxBenchBpbAdvError(f"context cache miss for prefix_hash={row['prefix_hash']!r}")
        source_tokens = _token_list(
            row["source_prompt_tokens"],
            field_name="source_prompt_tokens",
        )
        mode = str(row["context_signal_mode"])
        if mode == ContextSignalMode.FIXED_CONTEXT.value:
            return project_fixed_context_segments(
                tokenizer=self.tokenizer,
                source_tokens=source_tokens,
                attacked_tokens=attacked_tokens,
                source_segments=source_segments,
            )
        if mode == ContextSignalMode.NOT_APPLICABLE.value:
            return source_segments
        raise AxBenchBpbAdvError(f"unsupported context_signal_mode={mode!r}")

    @staticmethod
    def _coefficient(value: Any) -> int | float | None:
        text = str(value)
        if text == "None":
            return None
        try:
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        except ValueError as exc:
            raise AxBenchBpbAdvError(f"unrecognized coefficient string: {text!r}") from exc


def _gepa_optimizer_for_objective(
    *,
    cfg: GepaConfig,
    objective: BpbObjectiveCallbacks,
) -> Any:
    def _optimizer(
        row: dict[str, Any],
        seed_tokens: Sequence[int],
        budget: int,
    ) -> Sequence[int]:
        source_tokens = _token_list(
            row["source_prompt_tokens"],
            field_name="source_prompt_tokens",
        )
        span = EditableSpan(
            start=int(row["editable_token_start"]),
            count=int(row["editable_token_count"]),
        )
        vocab_size = int(objective.embedding_weight().shape[0])
        seed_json = "[" + ",".join(str(int(token_id)) for token_id in seed_tokens) + "]"

        def _evaluator(candidate: str) -> tuple[float, dict[str, Any]]:
            try:
                replacement = parse_gepa_token_candidate(
                    candidate,
                    expected_len=span.count,
                    vocab_size=vocab_size,
                )
                attacked = replace_editable_tokens(source_tokens, span, replacement)
                # GEPA uses the same shared-prefix scorer as Faster-GCG, with the
                # top-level batch size rather than Faster-GCG's scoped override.
                gepa_objective = getattr(
                    objective,
                    "gepa_objective",
                    objective.objective,
                )
                value = gepa_objective(row, attacked)
            except Exception as exc:
                return (
                    -1.0e12,
                    {
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "expected_json_length": span.count,
                    },
                )
            return (
                -float(value),
                {
                    "accepted": True,
                    "search_objective_g": float(value),
                },
            )

        gepa_cfg = GEPAConfig(
            engine=EngineConfig(
                seed=cfg.master_seed,
                max_metric_calls=int(budget),
                display_progress_bar=False,
                parallel=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=cfg.reflection_model,
            ),
        )
        result = optimize_anything(
            seed_candidate=seed_json,
            evaluator=_evaluator,
            objective=(
                "Return only a JSON list of token ids. The list length must "
                f"be exactly {span.count}; each id must be in [0, {vocab_size})."
            ),
            config=gepa_cfg,
        )
        best = result.best_candidate
        if not isinstance(best, str):
            raise AxBenchBpbAdvError("GEPA returned a non-string best candidate")
        return parse_gepa_token_candidate(
            best,
            expected_len=span.count,
            vocab_size=vocab_size,
        )

    return _optimizer


def build_noop_candidate_table(
    *,
    attack_rows: pa.Table,
    tokenizer: Any,
) -> pa.Table:
    """Build exact k=0 prompt-token controls."""

    if not attack_rows.schema.equals(ATTACK_ROWS_SCHEMA):
        raise AxBenchBpbAdvError("expected attack rows with ATTACK_ROWS_SCHEMA")
    rows = [
        build_control_candidate(row=row, tokenizer=tokenizer)
        for row in attack_rows.to_pylist()
        if int(row["editable_token_count"]) == 0
        and str(row["adversarial_method"]) != AdversarialMethod.ADDITIVE_DIRECTION.value
    ]
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def _counter_str(counter: Counter[Any]) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    )


def _should_log_progress(row_index: int, total_rows: int, interval: int) -> bool:
    return row_index in (1, total_rows) or row_index % interval == 0


def _format_optional_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _log_group_event(
    event: Mapping[str, Any],
    *,
    family: str,
    target_name: str,
    interval: int,
) -> None:
    """Log a universal-GCG ``group_start`` / ``group_done`` event (v2 M6).

    Groups have no ``attack_row_id``; the vocabulary is ``group_key_str`` /
    ``n_pairs`` / ``editable_token_count``. Throttled on ``group_index`` /
    ``total_groups`` the same way rows are.
    """

    group_index = int(event.get("group_index", 0))
    total_groups = int(event.get("total_groups", 0))
    if not _should_log_progress(group_index, total_groups, interval):
        return
    if str(event.get("event")) == "group_start":
        logger.info(
            "{} target={} group {}/{} start: key={} n_pairs={} method={} coeff={} "
            "strength={} context={} editable={} objective={}",
            family,
            target_name,
            group_index,
            total_groups,
            event.get("group_key_str"),
            event.get("n_pairs", "n/a"),
            event.get("method_name"),
            event.get("coefficient_str", "n/a"),
            event.get("adversarial_strength", "n/a"),
            event.get("context_signal_mode", "n/a"),
            event.get("editable_token_count", "n/a"),
            _format_optional_float(event.get("objective_value")),
        )
        return
    logger.info(
        "{} target={} group {}/{} done: key={} n_pairs={} control={} "
        "accepted_steps={} objective={} visited={}",
        family,
        target_name,
        group_index,
        total_groups,
        event.get("group_key_str"),
        event.get("n_pairs", "n/a"),
        bool(event.get("control", False)),
        event.get("accepted_steps", "n/a"),
        _format_optional_float(event.get("objective_value")),
        event.get("visited_sequences", "n/a"),
    )


def _log_group_step(
    event: Mapping[str, Any],
    *,
    family: str,
    target_name: str,
    interval: int,
) -> None:
    """Log a universal-GCG ``step_done`` event on the GROUP axis (v2 M6)."""

    group_index = int(event.get("group_index", 0))
    total_groups = int(event.get("total_groups", 0))
    if not _should_log_progress(group_index, total_groups, interval):
        return
    step_index = int(event.get("step_index", 0))
    max_steps = int(event.get("max_steps", 0))
    if step_index not in {0, max_steps - 1} and not bool(event.get("accepted")):
        return
    logger.info(
        "{} target={} group {}/{} step {}/{}: key={} n_pairs={} candidates={} "
        "accepted={} objective={} best={} visited={}",
        family,
        target_name,
        group_index,
        total_groups,
        step_index + 1,
        max_steps,
        event.get("group_key_str"),
        event.get("n_pairs", "n/a"),
        event.get("candidate_count", "n/a"),
        bool(event.get("accepted")),
        _format_optional_float(event.get("objective_value")),
        _format_optional_float(event.get("best_value")),
        event.get("visited_sequences", "n/a"),
    )


def _candidate_progress_logger(
    *,
    family: str,
    target_name: str,
    log_every: int,
):
    interval = max(1, int(log_every))

    def _callback(event: Mapping[str, Any]) -> None:
        event_name = str(event.get("event"))
        # The universal optimizer speaks a GROUP vocabulary (v2 M6): a concept
        # group spans many prefixes, so events carry group_index/total_groups +
        # group_key_str/n_pairs instead of row_index/attack_row_id. Handle those
        # event names first; the per-prefix vocabulary (row_*) falls through below.
        if event_name in {"group_start", "group_done"}:
            _log_group_event(
                event,
                family=family,
                target_name=target_name,
                interval=interval,
            )
            return
        # The universal step_done also lacks attack_row_id; log it on the group
        # axis when group_index is present, else fall back to the per-row axis.
        if event_name == "step_done" and "group_index" in event:
            _log_group_step(
                event,
                family=family,
                target_name=target_name,
                interval=interval,
            )
            return
        row_index = int(event.get("row_index", 0))
        total_rows = int(event.get("total_rows", 0))
        if not _should_log_progress(row_index, total_rows, interval):
            return
        if event_name == "row_start":
            logger.info(
                "{} target={} row {}/{} start: attack_row_id={} method={} "
                "coeff={} strength={} context={} editable={} objective={}",
                family,
                target_name,
                row_index,
                total_rows,
                event.get("attack_row_id"),
                event.get("method_name"),
                event.get("coefficient_str", "n/a"),
                event.get("adversarial_strength", "n/a"),
                event.get("context_signal_mode", "n/a"),
                event.get("editable_token_count", "n/a"),
                _format_optional_float(event.get("objective_value")),
            )
        elif event_name == "step_done":
            step_index = int(event.get("step_index", 0))
            max_steps = int(event.get("max_steps", 0))
            if step_index not in {0, max_steps - 1} and not bool(event.get("accepted")):
                return
            logger.info(
                "{} target={} row {}/{} step {}/{}: attack_row_id={} "
                "candidates={} accepted={} objective={} best={} visited={}",
                family,
                target_name,
                row_index,
                total_rows,
                step_index + 1,
                max_steps,
                event.get("attack_row_id"),
                event.get("candidate_count", "n/a"),
                bool(event.get("accepted")),
                _format_optional_float(event.get("objective_value")),
                _format_optional_float(event.get("best_value")),
                event.get("visited_sequences", "n/a"),
            )
        elif event_name == "row_done":
            logger.info(
                "{} target={} row {}/{} done: attack_row_id={} control={} "
                "accepted_steps={} objective={} visited={}",
                family,
                target_name,
                row_index,
                total_rows,
                event.get("attack_row_id"),
                bool(event.get("control", False)),
                event.get("accepted_steps", "n/a"),
                _format_optional_float(event.get("objective_value")),
                event.get("visited_sequences", "n/a"),
            )

    return _callback


def _raise_dynamo_recompile_limit() -> None:
    """Let the compiled flex-attention kernel stay fused across the GCG shapes.

    nanochat compiles flex-attention with ``dynamic=False``, so every distinct
    ``(B, T)`` forward shape recompiles. Batched GCG ranking pads each group-call
    to its own ``T`` (one shape per prefix) and the per-pair gradient adds B=1
    shapes, so a search over a handful of prefixes already crosses the default
    ``recompile_limit`` (8). Past that, dynamo falls back to EAGER flex-attention
    -- the unfused path that materializes the full scores matrix (~1.7x slower
    here, and worse at larger batches). Raising the limit keeps both the batched
    ranking forward and the gradient forward compiled. Idempotent; set once per
    process at stage entry rather than as an import side effect."""

    cfg = torch._dynamo.config
    if cfg.recompile_limit < _MIN_DYNAMO_RECOMPILE_LIMIT:
        # PyTorch's stub types this mutable setting as its startup Literal[8].
        cfg.recompile_limit = _MIN_DYNAMO_RECOMPILE_LIMIT  # ty: ignore[invalid-assignment]
    # The accumulated guard trips the GLOBAL eager fallback once the total
    # distinct-shape count crosses it, regardless of the per-function limit.
    if cfg.accumulated_recompile_limit < _MIN_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT:
        # PyTorch's stub types this mutable setting as its startup Literal[256].
        cfg.accumulated_recompile_limit = (  # ty: ignore[invalid-assignment]
            _MIN_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT
        )


# GcgFamily -> its derive. Feature-DiffMean is deliberately absent: it shares the
# model/SAE/percentile setup below, but it is not a GCG method family.
_GCG_DERIVE_BY_FAMILY: dict[GcgFamily, Callable[..., None]] = {
    GcgFamily.FAIR: build_ours_gcg_bags_fair,
    GcgFamily.ROOFLINE: build_ours_gcg_bags_roofline,
}


def _filter_precomputed_bag_rows_to_source_concepts(
    rows: list[dict[str, Any]],
    *,
    concept_ids: set[int],
    mode: SteeringMode,
) -> list[dict[str, Any]]:
    """Reuse the validator's source-concept predicate before pairing rows."""
    filtered = _filter_to_source_concepts(rows, concept_ids=concept_ids)
    skipped = len(rows) - len(filtered)
    if skipped:
        logger.info(
            f"precomputed-bag derive filtered {skipped} {mode.value} synth row(s) "
            "whose concept_id is not in concepts.json"
        )
    return filtered


@dataclass(frozen=True)
class _PrecomputedBagSetup:
    """Source/model/SAE objects shared by every precomputed-bag derive family.

    Fair GCG, roofline GCG, and fair/roofline feature-DiffMean all consume the
    same source prefix pairing, encdec checkpoint, base SAE, percentile table, and
    score-time context cache. Built once (over the union of active modes) and
    reused across families so the synth parquets, percentile table, and whole
    context-segment cache are not re-read per active family. ``prefixes`` spans
    the union; each family filters it down to its own modes.
    """

    prefixes: list[Any]
    model: Any
    base_model: Any
    sae: Any
    percentile_table: Any
    context_segments_by_prefix_hash: dict[Any, Any]
    device: Any


def _prepare_precomputed_bag_setup(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    modes: set[SteeringMode],
    source_validation: BpbAdvSourceValidation | None = None,
) -> _PrecomputedBagSetup:
    """Load the shared derive setup for *modes* once (see :class:`_PrecomputedBagSetup`)."""
    cluster_contract = runtime_state.cluster_contract
    source_paths = SourcePaths(Path(cfg.source_run_dir))
    concept_ids = (
        _read_source_concept_ids(source_paths.concepts_json())
        if source_validation is None
        else set(source_validation.concept_ids)
    )
    samples: list[Any] = []
    if SteeringMode.POSITIVE in modes:
        positive_rows = (
            _filter_precomputed_bag_rows_to_source_concepts(
                load_raw_synth_rows(
                    source_paths.synth_training_positive(),
                    filter_unaccepted=cfg.filter_unaccepted_samples,
                    require_positive_qc_contract=True,
                    expected_continuation_role="stage1b_positive",
                ),
                concept_ids=concept_ids,
                mode=SteeringMode.POSITIVE,
            )
            if source_validation is None
            else source_validation.positive_rows()
        )
        samples.extend(
            bpb_attack_sample_from_synth_row(row, mode=SteeringMode.POSITIVE, tokenizer=tokenizer)
            for row in positive_rows
        )
    if SteeringMode.NEGATIVE in modes:
        negative_rows = (
            _filter_precomputed_bag_rows_to_source_concepts(
                load_raw_synth_rows(
                    source_paths.synth_training_negative(),
                    filter_unaccepted=cfg.filter_unaccepted_samples,
                    expected_continuation_role="stage1c_negative",
                ),
                concept_ids=concept_ids,
                mode=SteeringMode.NEGATIVE,
            )
            if source_validation is None
            else source_validation.negative_rows()
        )
        samples.extend(
            bpb_attack_sample_from_synth_row(row, mode=SteeringMode.NEGATIVE, tokenizer=tokenizer)
            for row in negative_rows
        )
    prefixes = [
        prefix
        for prefix in pair_samples_into_prefixes(samples)
        if SteeringMode(prefix.mode) in modes
    ]
    if not prefixes:
        # Reached only when a precomputed-bag method is active for >= 1 mode, so
        # zero class-margin prefixes for the active modes is a real data/config
        # error (e.g. all source samples filtered), NOT a routine skip. Raise
        # BEFORE loading the encdec/SAE (nothing to derive bags from anyway).
        raise AxBenchBpbAdvError(
            "precomputed-bag derive: no class-margin prefixes for active modes "
            f"{sorted(mode.value for mode in modes)} (source_run_dir="
            f"{cfg.source_run_dir!r}); an active precomputed-bag method has nothing to derive "
            "bags from. Check the source samples and mode filter."
        )

    if cfg.offload_between_stages:
        # Earlier candidate-stage work may have loaded BOTH target families.
        # The precomputed-bag phase alternates only the encoder-decoder target
        # with the much larger Gemma+SAE extractor. Offload every cached target
        # before restoring the encoder-decoder below so a decoder-only bundle
        # cannot remain invisibly resident through every DiffMean warm start.
        # This ownership transition is output-inert but essential on 24 GiB
        # devices (Gemma-3-4B + the 65k SAE otherwise has <1 GiB headroom).
        runtime_state.offload_models()
        release_cuda_iteration_memory()

    encdec_bundle = runtime_state.encdec()
    model = encdec_bundle.model
    # Fail fast on a same-width-DIFFERENT-SAE misconfig BEFORE loading the SAE /
    # building the percentile table: the warm-start fids would silently corrupt.
    assert_sae_matches_encdec(
        cfg,
        model,
        **(
            {}
            if cluster_contract is None
            else {"cluster_contract": cluster_contract}
        ),
    )
    base_model, sae = runtime_state.base_gemma_and_sae()
    n_sae = int(model.encoder.config.n_sae_features)
    percentile_table = build_percentile_table_from_stats(cfg.feature_statistics_path, n_sae)
    # Precomputed feature-conditioning bags are ALWAYS encdec, so the score-time
    # context cache MUST exist. Load the SAME prefix-hash -> segments map the
    # score stage uses (this raises loudly if the cache is absent) so each derive
    # builds members under the identical multi-segment layout scoring applies.
    context_segments_by_prefix_hash = load_context_segments_by_prefix_hash(
        paths,
        **(
            {}
            if cluster_contract is None
            else {"cluster_contract": cluster_contract}
        ),
    )
    return _PrecomputedBagSetup(
        prefixes=prefixes,
        model=model,
        base_model=base_model,
        sae=sae,
        percentile_table=percentile_table,
        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
        device=t.device(runtime_state.target_device()),
    )


def _build_model_backed_precomputed_bags(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    modes: set[SteeringMode],
    derive: Callable[..., None],
    label: str,
    shared: _PrecomputedBagSetup | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Run one derive family over the shared precomputed-bag setup.

    ``shared`` is the setup built once by :func:`_prepare_precomputed_bag_setup`
    over the union of active modes; when ``None`` (the single-family path) it is
    built here for exactly *modes*. Either way the derive sees only this family's
    modes -- keeping feature-DiffMean neutral (not a GCG family) while avoiding
    duplicated model/source loading.
    """
    if not modes:
        return
    cluster_contract = runtime_state.cluster_contract
    if shared is None:
        shared = _prepare_precomputed_bag_setup(
            cfg=cfg,
            paths=paths,
            runtime_state=runtime_state,
            tokenizer=tokenizer,
            modes=modes,
        )
    prefixes = [
        prefix for prefix in shared.prefixes if SteeringMode(prefix.mode) in modes
    ]
    if not prefixes:
        raise AxBenchBpbAdvError(
            f"{label} derive: no class-margin prefixes for active modes "
            f"{sorted(mode.value for mode in modes)} (source_run_dir="
            f"{cfg.source_run_dir!r}); an active precomputed-bag method has nothing to derive "
            "bags from. Check the source samples and mode filter."
        )
    derive_kwargs: dict[str, Any] = {}
    if clustered_warm_starts is not None:
        derive_kwargs["clustered_warm_starts"] = clustered_warm_starts
    elif shared.base_model is None or shared.sae is None:
        # #1245: the extraction pair is released once a clustered warm-start
        # cache exists, because every derive family then takes the
        # `clustered_warm_starts is not None` branch and never reads it. Reaching
        # here means a caller released the pair AND is asking for the derive path
        # that genuinely needs it. Raise with both facts rather than letting the
        # derive fail deep inside on a None model, which would surface as an
        # opaque AttributeError far from the cause.
        raise AxBenchBpbAdvError(
            f"{label} derive requires the base extraction pair, but it has been "
            "released (base_model="
            f"{'None' if shared.base_model is None else 'set'}, sae="
            f"{'None' if shared.sae is None else 'set'}) and no "
            "clustered_warm_starts were supplied. The pair is only safe to "
            "release when warm starts cover every active derive family."
        )
    derive(
        prefixes=prefixes,
        model=shared.model,
        base_model=shared.base_model,
        sae=shared.sae,
        percentile_table=shared.percentile_table,
        cfg=cfg,
        paths=paths,
        device=shared.device,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=shared.context_segments_by_prefix_hash,
        **(
            {}
            if cluster_contract is None
            else {"cluster_contract": cluster_contract}
        ),
        **derive_kwargs,
    )


def _build_model_backed_gcg_bags(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    modes: set[SteeringMode],
    family: GcgFamily,
    shared: _PrecomputedBagSetup | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache a feature-GCG bag family after shared model setup."""
    _build_model_backed_precomputed_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=runtime_state,
        tokenizer=tokenizer,
        modes=modes,
        derive=_GCG_DERIVE_BY_FAMILY[family],
        label=f"{family.value} feature-GCG",
        shared=shared,
        clustered_warm_starts=clustered_warm_starts,
    )


def _build_fair_gcg_bags(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    shared: _PrecomputedBagSetup | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache the fair feature-GCG conditioning bags (Task 5).

    Gated by the caller on a fair gcg method being active (``ours_gcg_encdec`` /
    ``neg_ours_gcg_encdec``): the universal-over-the-optimize-subset derive.
    """
    _build_model_backed_gcg_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=runtime_state,
        tokenizer=tokenizer,
        modes=active_fair_gcg_modes(cfg),
        family=GcgFamily.FAIR,
        shared=shared,
        clustered_warm_starts=clustered_warm_starts,
    )


def _build_roofline_gcg_bags(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    shared: _PrecomputedBagSetup | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache the roofline feature-GCG conditioning bags (Task 6).

    Gated by the caller on a roofline gcg method being active
    (``ours_gcg_roofline_encdec`` / ``neg_ours_gcg_roofline_encdec``): the
    per-prefix in-sample upper-bound derive.
    """
    _build_model_backed_gcg_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=runtime_state,
        tokenizer=tokenizer,
        modes=active_roofline_gcg_modes(cfg),
        family=GcgFamily.ROOFLINE,
        shared=shared,
        clustered_warm_starts=clustered_warm_starts,
    )


def _build_fair_feature_diffmean_bags(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    shared: _PrecomputedBagSetup | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache the fair scored feature-DiffMean conditioning bags."""
    _build_model_backed_precomputed_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=runtime_state,
        tokenizer=tokenizer,
        modes=active_fair_feature_diffmean_modes(cfg),
        derive=build_feature_diffmean_bags_fair,
        label="feature-DiffMean",
        shared=shared,
        clustered_warm_starts=clustered_warm_starts,
    )


def _build_roofline_feature_diffmean_bags(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    runtime_state: RuntimeState,
    tokenizer: Any,
    shared: _PrecomputedBagSetup | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache roofline feature-DiffMean conditioning bags."""
    _build_model_backed_precomputed_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=runtime_state,
        tokenizer=tokenizer,
        modes=active_roofline_feature_diffmean_modes(cfg),
        derive=build_feature_diffmean_bags_roofline,
        label="roofline feature-DiffMean",
        shared=shared,
        clustered_warm_starts=clustered_warm_starts,
    )


def run_model_backed_bpb_candidate_stage(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    tokenizer: Any | None = None,
    embedding_weight: t.Tensor | None = None,
    test_objective: ObjectiveBatchFn | None = None,
    test_gradient_provider_batch: GradientBatchProvider | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    stage_source: ClusteredStageSource | None = None,
) -> int:
    """Build final candidate rows for every configured adversarial setting."""

    resolved_stage_source = require_clustered_stage_source(
        stage_source=stage_source,
        contract=cluster_contract,
        surface="candidate stage",
    )
    source_validation = (
        None
        if resolved_stage_source is None
        else resolved_stage_source.validation
    )
    source_identity = (
        None
        if resolved_stage_source is None
        else resolved_stage_source.source_identity
    )
    _raise_dynamo_recompile_limit()
    if not paths.attack_rows_path().exists():
        raise AxBenchBpbAdvError(f"missing axbench-bpb-adv attack rows: {paths.attack_rows_path()}")
    with profile_region("axbench_bpb_adv.build_candidates.read_attack_rows"):
        attack_rows = pq.read_table(paths.attack_rows_path())
    if not attack_rows.schema.equals(ATTACK_ROWS_SCHEMA):
        raise AxBenchBpbAdvError("attack rows have unexpected schema")
    # Resolve the clustered source identity ONCE for the whole stage (#936). Every
    # stamp below is bound to this single read, so the attack-rows validation and the
    # candidate/universal-trigger stamps cannot record source hashes from different
    # reads of `source_contract.json` -- a disagreement that is silent downstream,
    # because the candidate identity check and `evaluation_population.json` compare
    # only the feature-contract fingerprint.
    assert_table_cluster_identity(
        attack_rows,
        paths=paths,
        contract=cluster_contract,
        artifact_family="attack_rows",
        source_identity=source_identity,
    )
    attack_row_list = attack_rows.to_pylist()
    record_counter("axbench_bpb_adv.build_candidates.attack_rows", len(attack_row_list))
    if tokenizer is None and embedding_weight is not None:
        raise AxBenchBpbAdvError(
            "tokenizer must be provided when embedding_weight is provided; "
            "the test-only candidate path must not use production tokenizer fallback"
        )

    with profile_region("axbench_bpb_adv.build_candidates.population_guard"):
        evaluation_population = read_evaluation_population(
            paths.evaluation_population(),
            **(
                {}
                if cluster_contract is None
                else {
                    "expected_feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
        attack_row_list = require_rows_in_evaluation_population(
            attack_row_list,
            population=evaluation_population,
            label="candidate-stage attack rows",
        )
    if not attack_row_list:
        raise AxBenchBpbAdvError(
            "candidate-stage universal evaluation-population check found no "
            "attack row"
        )
    if len(attack_row_list) != attack_rows.num_rows:
        attack_rows = pa.Table.from_pylist(attack_row_list, schema=ATTACK_ROWS_SCHEMA)
    prompt_rows = [
        row
        for row in attack_row_list
        if str(row["adversarial_method"])
        in {
            AdversarialMethod.FASTER_GCG.value,
            AdversarialMethod.GEPA.value,
        }
        and int(row["editable_token_count"]) > 0
    ]
    method_counts = Counter(str(row["adversarial_method"]) for row in attack_row_list)
    target_counts = Counter(str(row["target_model"]) for row in attack_row_list)
    prompt_counts = Counter(
        (str(row["target_model"]), str(row["adversarial_method"])) for row in prompt_rows
    )
    logger.info(
        "candidate stage input: total_rows={} methods=[{}] targets=[{}] "
        "optimizable_prompt_rows=[{}]",
        len(attack_row_list),
        _counter_str(method_counts),
        _counter_str(target_counts),
        _counter_str(prompt_counts),
    )
    for method, count in method_counts.items():
        record_counter(
            "axbench_bpb_adv.build_candidates.attack_rows_by_method",
            count,
            method=method,
        )
    for target, count in target_counts.items():
        record_counter(
            "axbench_bpb_adv.build_candidates.attack_rows_by_target",
            count,
            target_model=target,
        )
    record_counter(
        "axbench_bpb_adv.build_candidates.optimizable_prompt_rows",
        len(prompt_rows),
    )

    bpb_cfg = _bpb_config_from_adv_config(cfg)
    faster_cfg = get_faster_gcg_config(cfg)
    faster_group_rows = [
        row
        for row in prompt_rows
        if str(row["adversarial_method"]) == AdversarialMethod.FASTER_GCG.value
    ]
    faster_groups = concept_groups(faster_group_rows)
    non_control_faster_groups = [group for group in faster_groups if not group.is_control]
    active_faster_targets = {
        str(group.rows[0]["target_model"])
        for group in non_control_faster_groups
    }
    if faster_groups and faster_cfg is None:
        raise AxBenchBpbAdvError(
            "Faster-GCG rows require a faster_gcg adversarial method config"
        )
    resolved_faster_checkpoints: dict[str, Any] = {}
    state: RuntimeState | None = None
    target_bundles: dict[str, TargetModelBundle] = {}

    def _ensure_state() -> RuntimeState:
        nonlocal state, resolved_faster_checkpoints
        if state is None:
            if embedding_weight is None and active_faster_targets:
                resolved_faster_checkpoints = resolve_and_pin_target_checkpoints(
                    bpb_cfg,
                    active_faster_targets,
                )
            runtime_kwargs: dict[str, Any] = {
                "cfg": bpb_cfg,
                "ddp_rank": 0,
                "ddp_world_size": 1,
            }
            if cluster_contract is not None:
                runtime_kwargs["cluster_contract"] = cluster_contract
            state = RuntimeState(**runtime_kwargs)
        return state

    def _ensure_target_bundles(targets: set[str]) -> dict[str, TargetModelBundle]:
        runtime_state = _ensure_state()
        if "decoder_only" in targets and "decoder_only" not in target_bundles:
            target_bundles["decoder_only"] = runtime_state.decoder_only()
        if "encdec" in targets and "encdec" not in target_bundles:
            target_bundles["encdec"] = runtime_state.encdec()
        unknown_targets = targets - set(target_bundles)
        if unknown_targets:
            raise AxBenchBpbAdvError(f"unsupported target_model values: {sorted(unknown_targets)}")
        return {
            target: target_bundles[target] for target in sorted(targets) if target in target_bundles
        }

    try:
        if tokenizer is None:
            all_targets = {str(row["target_model"]) for row in attack_row_list}
            bundles = _ensure_target_bundles(all_targets)
            if not bundles:
                raise AxBenchBpbAdvError("cannot resolve tokenizer from empty attack rows")
            resolved_tokenizer = next(iter(bundles.values())).tokenizer
        else:
            resolved_tokenizer = tokenizer

        # Direction-owned scale (#1228) fail-fast preflight, hoisted to run BEFORE
        # the feature-GCG derive (and prompt optimization) so invalid direction
        # metadata raises up front rather than after expensive GCG compute.
        # No-op when no scaled (diffmean/linear_probe/additive-counter) candidate is
        # present, so it is free for gcg-only runs. Gated on BOTH the real
        # model-backed path AND non-empty prompt_rows: with nothing to optimize
        # there is nothing to preflight, and the concepts.json read must stay lazy
        # (a zero-optimizable-rows run on a source without concepts.json built its
        # noop candidates fine before the hoist and must keep doing so). The pair
        # built here is REUSED by the model-backed candidate build below (same
        # gate: embedding_weight is None and prompt_rows non-empty). The
        # precomputed-bag coverage guard below never adds rows, and the preflight
        # ignores bag-backed rows without scaled directions.
        shared_concepts_by_id: dict[int, Any] | None = None
        shared_direction_preflight: DirectionScalePreflight | None = None
        if embedding_weight is None and prompt_rows:
            with profile_region("axbench_bpb_adv.build_candidates.max_act_preflight"):
                shared_concepts_by_id = (
                    _read_concepts(SourcePaths(Path(cfg.source_run_dir)))
                    if source_validation is None
                    else _concepts_from_entries(
                        source_validation.concept_entries()
                    )
                )
                additive_cfg = get_additive_direction_config(cfg)
                shared_direction_preflight = _preflight_direction_scales_for_rows(
                    candidate_rows=prompt_rows,
                    concepts_by_id=shared_concepts_by_id,
                    source_paths=SourcePaths(Path(cfg.source_run_dir)),
                    expected_layer=cfg.diffmean_and_probe_layer,
                    counter_direction_family=(
                        additive_cfg.counter_direction_family
                        if additive_cfg is not None
                        else "diffmean"
                    ),
                    source_validation=source_validation,
                    stage="candidate",
                )

        # Precomputed conditioning-bag derives (Tasks 5/6 + feature-DiffMean): fair
        # /roofline feature-GCG and fair/roofline feature-DiffMean. Only on the real
        # model-backed path (the embedding_weight test path injects synthetic
        # surrogates and never loads the encdec/SAE needed here). All four share the
        # same source pairing, encdec/SAE, percentile table, and context cache, so
        # build that setup ONCE over the union of active modes and reuse it for every
        # active family instead of re-loading it per family.
        shared_bag_setup: _PrecomputedBagSetup | None = None
        clustered_warm_starts: ClusteredWarmStartCache | None = None
        if embedding_weight is None:
            active_bag_modes = (
                active_fair_gcg_modes(cfg)
                | active_roofline_gcg_modes(cfg)
                | active_fair_feature_diffmean_modes(cfg)
                | active_roofline_feature_diffmean_modes(cfg)
            )
            if active_bag_modes:
                with profile_region(
                    "axbench_bpb_adv.build_candidates.precomputed_bag_setup",
                    n_modes=len(active_bag_modes),
                ):
                    shared_bag_setup = _prepare_precomputed_bag_setup(
                        cfg=cfg,
                        paths=paths,
                        runtime_state=_ensure_state(),
                        tokenizer=resolved_tokenizer,
                        modes=active_bag_modes,
                        source_validation=source_validation,
                    )
                    if cluster_contract is not None:
                        # Compute each unique clustered DiffMean signal once while
                        # Gemma+SAE are the sole large CUDA owner.  Every family
                        # below then reuses these exact immutable bags with only
                        # the encoder-decoder resident.  The unclustered path does
                        # not enter this branch or receive a new derive kwarg.
                        clustered_warm_starts = build_clustered_warm_start_cache(
                            prefixes=shared_bag_setup.prefixes,
                            model=shared_bag_setup.model,
                            base_model=shared_bag_setup.base_model,
                            sae=shared_bag_setup.sae,
                            percentile_table=shared_bag_setup.percentile_table,
                            cfg=cfg,
                            device=shared_bag_setup.device,
                            tokenizer=resolved_tokenizer,
                            context_segments_by_prefix_hash=(
                                shared_bag_setup.context_segments_by_prefix_hash
                            ),
                            fair_modes=(
                                active_fair_gcg_modes(cfg)
                                | active_fair_feature_diffmean_modes(cfg)
                            ),
                            roofline_modes=(
                                active_roofline_gcg_modes(cfg)
                                | active_roofline_feature_diffmean_modes(cfg)
                            ),
                            cluster_contract=cluster_contract,
                        )
                    # #1245: make the ownership transition the comment above
                    # already describes ACTUAL. Every derive family branches on
                    # `if clustered_warm_starts is None` and touches
                    # base_model/sae only in the None arm (gcg_bags.py fair +
                    # roofline, feature_diffmean_bags.py fair + roofline), so
                    # once the warm-start cache exists the extraction pair is
                    # dead weight -- ~10 GiB of Gemma-3-4B + the 65k SAE held
                    # alongside the EncoderDecoderGPT and the decoder.
                    #
                    # Without this, `ours_gcg_encdec` OOMs ~9s into the fair
                    # feature-GCG derive on a 23.51 GiB card, in
                    # cross_attention.forward, with four models resident. It is
                    # NOT batch-bound: cutting mean_g_steer_batch 16->4,
                    # max_forward_batch 8->4 and device_batch_size 16->8 moved
                    # peak 23.19 -> 23.42 GiB, i.e. not at all.
                    #
                    # BOTH references must go. Nulling the setup fields alone
                    # leaves RuntimeState's cache holding them; calling
                    # release_base_gemma_and_sae() alone leaves this frozen
                    # dataclass holding them. Either half on its own frees
                    # nothing.
                    #
                    # Gated on the warm-start cache EXISTING, not on
                    # cluster_contract: the cache is the thing every derive
                    # falls back on, and it is the exact precondition the guard
                    # in _build_model_backed_precomputed_bags states. On the
                    # unclustered path nothing is released and the derives read
                    # the pair as they always did.
                    if clustered_warm_starts is not None:
                        shared_bag_setup = replace(
                            shared_bag_setup, base_model=None, sae=None
                        )
                        _ensure_state().release_base_gemma_and_sae()
                        release_cuda_iteration_memory()
        clustered_warm_kwargs = (
            {}
            if clustered_warm_starts is None
            else {"clustered_warm_starts": clustered_warm_starts}
        )
        if embedding_weight is None and fair_gcg_methods_active(cfg):
            with profile_region("axbench_bpb_adv.build_candidates.fair_feature_gcg"):
                _build_fair_gcg_bags(
                    cfg=cfg,
                    paths=paths,
                    runtime_state=_ensure_state(),
                    tokenizer=resolved_tokenizer,
                    shared=shared_bag_setup,
                    **clustered_warm_kwargs,
                )
        # The four derives are independent so every active family runs; each filters
        # the shared prefixes down to its own modes.
        if embedding_weight is None and roofline_gcg_methods_active(cfg):
            with profile_region("axbench_bpb_adv.build_candidates.roofline_feature_gcg"):
                _build_roofline_gcg_bags(
                    cfg=cfg,
                    paths=paths,
                    runtime_state=_ensure_state(),
                    tokenizer=resolved_tokenizer,
                    shared=shared_bag_setup,
                    **clustered_warm_kwargs,
                )
        if embedding_weight is None and fair_feature_diffmean_methods_active(cfg):
            with profile_region("axbench_bpb_adv.build_candidates.fair_feature_diffmean"):
                _build_fair_feature_diffmean_bags(
                    cfg=cfg,
                    paths=paths,
                    runtime_state=_ensure_state(),
                    tokenizer=resolved_tokenizer,
                    shared=shared_bag_setup,
                    **clustered_warm_kwargs,
                )
        if embedding_weight is None and roofline_feature_diffmean_methods_active(cfg):
            with profile_region("axbench_bpb_adv.build_candidates.roofline_feature_diffmean"):
                _build_roofline_feature_diffmean_bags(
                    cfg=cfg,
                    paths=paths,
                    runtime_state=_ensure_state(),
                    tokenizer=resolved_tokenizer,
                    shared=shared_bag_setup,
                    **clustered_warm_kwargs,
                )

        # Precomputed-bag coverage guard: the universal evaluation-population
        # check has already required held-out prefixes. A
        # bag-backed held-out prompt row missing from the family scored-prefix
        # artifact is stale derive state, so raise instead of silently optimizing
        # fewer prompts for that method family.
        with profile_region(
            "axbench_bpb_adv.build_candidates.precomputed_bag_coverage_guard",
            rows=len(prompt_rows),
        ):
            if cluster_contract is None:
                prompt_rows = require_precomputed_bag_prompt_rows_cover_scored_prefixes(
                    prompt_rows,
                    paths=paths,
                )
            else:
                prompt_rows = require_precomputed_bag_prompt_rows_cover_scored_prefixes(
                    prompt_rows,
                    paths=paths,
                    feature_contract_fingerprint=(
                        cluster_contract.feature_contract_fingerprint
                    ),
                )

        noop_table = build_noop_candidate_table(
            attack_rows=attack_rows,
            tokenizer=resolved_tokenizer,
        )
        logger.info(f"built {noop_table.num_rows} no-op prompt-token candidate rows")
        tables: list[pa.Table] = [noop_table]
        # One UNIVERSAL_TRIGGERS_SCHEMA table per target (one row per non-control
        # faster_gcg concept group), concatenated across targets and written to
        # paths.universal_triggers_path() (H1/H2). general_val reads this artifact
        # to substitute the shared trigger into the validation docs.
        trigger_tables: list[pa.Table] = []

        additive_rows = pa.Table.from_pylist(
            [
                row
                for row in attack_row_list
                if str(row["adversarial_method"]) == AdversarialMethod.ADDITIVE_DIRECTION.value
            ],
            schema=ATTACK_ROWS_SCHEMA,
        )
        if additive_rows.num_rows:
            logger.info(
                f"building {additive_rows.num_rows} additive_direction candidate rows"
            )
            with profile_region(
                "axbench_bpb_adv.build_candidates.additive_direction",
                rows=additive_rows.num_rows,
            ):
                tables.append(
                    build_additive_direction_candidates(
                        attack_rows=additive_rows,
                        tokenizer=resolved_tokenizer,
                    )
                )
            logger.info(
                f"built {additive_rows.num_rows} additive_direction candidate rows"
            )

        gepa_cfg = get_gepa_config(cfg)
        if prompt_rows and embedding_weight is not None:
            prompt_table = pa.Table.from_pylist(prompt_rows, schema=ATTACK_ROWS_SCHEMA)
            faster_rows = pa.Table.from_pylist(
                [
                    row
                    for row in prompt_rows
                    if str(row["adversarial_method"]) == AdversarialMethod.FASTER_GCG.value
                ],
                schema=ATTACK_ROWS_SCHEMA,
            )
            if faster_rows.num_rows:
                if faster_cfg is None:
                    raise AxBenchBpbAdvError(
                        "Faster-GCG rows require a faster_gcg adversarial method config"
                    )
                if test_objective is None or test_gradient_provider_batch is None:
                    raise AxBenchBpbAdvError(
                        "embedding_weight test path requires explicit "
                        "test_objective and test_gradient_provider_batch callbacks"
                    )
                logger.info(
                    "building universal Faster-GCG candidates for test embedding "
                    "path: rows={} max_steps={} candidate_batch_size={} "
                    "minibatch={}",
                    faster_rows.num_rows,
                    faster_cfg.max_steps,
                    faster_cfg.candidate_batch_size,
                    faster_cfg.universal_minibatch_size,
                )
                with profile_region(
                    "axbench_bpb_adv.build_candidates.universal_faster_gcg",
                    target_model="embedding_weight",
                    rows=faster_rows.num_rows,
                ):
                    faster_table, faster_triggers = build_universal_gcg_candidates(
                        attack_rows=faster_rows,
                        tokenizer=resolved_tokenizer,
                        cfg=faster_cfg,
                        embedding_weight=embedding_weight,
                        allowed_token_ids=[
                            int(token_id)
                            for token_id in build_allowed_token_ids(resolved_tokenizer)
                            if 0 <= int(token_id) < int(embedding_weight.shape[0])
                        ],
                        objective=test_objective,
                        gradient_provider_batch=test_gradient_provider_batch,
                        progress_callback=_candidate_progress_logger(
                            family=AdversarialMethod.FASTER_GCG.value,
                            target_name="embedding_weight",
                            log_every=cfg.log_progress_every,
                        ),
                    )
                tables.append(faster_table)
                trigger_tables.append(
                    pa.Table.from_pylist(faster_triggers, schema=UNIVERSAL_TRIGGERS_SCHEMA)
                )
            gepa_rows = pa.Table.from_pylist(
                [
                    row
                    for row in prompt_table.to_pylist()
                    if str(row["adversarial_method"]) == AdversarialMethod.GEPA.value
                ],
                schema=ATTACK_ROWS_SCHEMA,
            )
            if gepa_rows.num_rows:
                if gepa_cfg is None:
                    raise AxBenchBpbAdvError("GEPA rows require a gepa adversarial method config")
                logger.info(
                    "building GEPA candidates for test embedding path: rows={}",
                    gepa_rows.num_rows,
                )
                with profile_region(
                    "axbench_bpb_adv.build_candidates.gepa",
                    target_model="embedding_weight",
                    rows=gepa_rows.num_rows,
                ):
                    tables.append(
                        build_gepa_candidates(
                            attack_rows=gepa_rows,
                            tokenizer=resolved_tokenizer,
                            optimizer=lambda _row, seed, _budget: list(seed),
                            progress_callback=_candidate_progress_logger(
                                family=AdversarialMethod.GEPA.value,
                                target_name="embedding_weight",
                                log_every=cfg.log_progress_every,
                            ),
                        )
                    )
        elif prompt_rows:
            targets = {str(row["target_model"]) for row in prompt_rows}
            with profile_region(
                "axbench_bpb_adv.build_candidates.target_runtime_setup",
                n_targets=len(targets),
            ):
                target_bundles_for_prompt = _ensure_target_bundles(targets)
                runtime_state = _ensure_state()
                token_bytes = _build_token_bytes(resolved_tokenizer)
            source_paths = SourcePaths(Path(cfg.source_run_dir))
            detect_kwargs = (
                {}
                if cluster_contract is None
                else {"cluster_contract": cluster_contract}
            )
            target_metas = {
                target: detect_target_model_metadata(
                    bundle.model,
                    bpb_cfg,
                    **detect_kwargs,
                )
                for target, bundle in target_bundles_for_prompt.items()
            }
            with profile_region("axbench_bpb_adv.build_candidates.context_segments_load"):
                context_segments_by_hash = (
                    load_context_segments_by_prefix_hash(
                        paths,
                        **(
                            {}
                            if cluster_contract is None
                            else {"cluster_contract": cluster_contract}
                        ),
                    )
                    if "encdec" in targets
                    else {}
                )
            lora = (
                LoraParams(
                    cfg=LoraConfig(**cfg.lora_adapter),
                    layer=cfg.diffmean_and_probe_layer,
                )
                if cfg.lora_adapter
                else None
            )
            # The GCG objective scores the direction-artifact-scaled baseline via
            # the scorer's _base_steered_intervention, so it reuses the concepts
            # and direction preflight established once above. Re-reading
            # concepts.json or revalidating a second artifact set here would let
            # the objective drift from the score-stage contract.
            if shared_concepts_by_id is None or shared_direction_preflight is None:
                raise AxBenchBpbAdvError(
                    "internal invariant broken: the hoisted direction preflight must "
                    "have built concepts_by_id/preflight for the model-backed "
                    "candidate build (embedding_weight is None and prompt_rows "
                    "was non-empty)."
                )
            concepts_by_id = shared_concepts_by_id
            additive_cfg = get_additive_direction_config(cfg)
            scorer_cluster_kwargs: dict[str, Any] = (
                {}
                if cluster_contract is None
                else {
                    "cluster_contract": cluster_contract,
                    "source_validation": source_validation,
                }
            )
            scorer = ModelBackedDualCompletionScorer(
                target_bundles=target_bundles_for_prompt,
                target_model_metas=target_metas,
                concepts_by_id=concepts_by_id,
                source_paths=source_paths,
                tokenizer=resolved_tokenizer,
                device=t.device(runtime_state.target_device()),
                diffmean_layer=cfg.diffmean_and_probe_layer,
                inactive_sampler=runtime_state.inactive_sampler() if "encdec" in targets else None,
                apply_max_act_scale=True,
                lora=lora,
                counter_direction_family=(
                    additive_cfg.counter_direction_family
                    if additive_cfg is not None
                    else "diffmean"
                ),
                # Run artifact paths let _build_ours_intervention locate each
                # bag-backed method's precomputed conditioning bag. Without this,
                # those methods crash the OPTIMIZE stage's per-pair steered
                # intervention (build_intervention raises on adv_paths=None).
                adv_paths=paths,
                **scorer_cluster_kwargs,
            )
            checkpoint_store: UniversalGcgCheckpointStore | None = None
            restored_triggers: dict[str, dict[str, Any]] = {}
            runtime_fingerprint_sha256: str | None = None
            checkpoint_commit_callback: Callable[[str, Mapping[str, Any]], None] | None = None
            if non_control_faster_groups:
                if faster_cfg is None:
                    raise AxBenchBpbAdvError(
                        "Faster-GCG rows require a faster_gcg adversarial method config"
                    )
                assert_target_checkpoints_unchanged(resolved_faster_checkpoints)
                loaded_metadata = {
                    target_name: runtime_state.loaded_target_metadata(target_name)
                    for target_name in sorted(active_faster_targets)
                }
                fingerprint = build_checkpoint_fingerprint(
                    cfg=cfg,
                    paths=paths,
                    faster_cfg=faster_cfg,
                    groups=non_control_faster_groups,
                    resolved_targets=resolved_faster_checkpoints,
                    loaded_metadata=loaded_metadata,
                    cluster_feature_contract_fingerprint=(
                        None
                        if cluster_contract is None
                        else cluster_contract.feature_contract_fingerprint
                    ),
                )
                _runtime_payload, runtime_fingerprint_sha256 = (
                    current_runtime_fingerprint(runtime_state.target_device())
                )
                checkpoint_store = UniversalGcgCheckpointStore.prepare(
                    checkpoint_dir=paths.universal_gcg_checkpoint_dir(),
                    fingerprint=fingerprint,
                    expected_groups=non_control_faster_groups,
                    tokenizer=resolved_tokenizer,
                    max_steps=int(faster_cfg.max_steps),
                )
                restored_triggers = checkpoint_store.load_triggers()
                warn_on_runtime_mismatch(
                    checkpoint_store,
                    runtime_fingerprint_sha256,
                )
                pending_count = len(non_control_faster_groups) - len(restored_triggers)
                logger.info(
                    "Faster-GCG checkpoint summary: expected_groups={} restored_groups={} "
                    "pending_groups={} runtime_fingerprint_sha256={}",
                    len(non_control_faster_groups),
                    len(restored_triggers),
                    pending_count,
                    runtime_fingerprint_sha256,
                )
                record_counter(
                    "axbench_bpb_adv.build_candidates.faster_gcg_groups_restored",
                    len(restored_triggers),
                )
                record_counter(
                    "axbench_bpb_adv.build_candidates.faster_gcg_groups_pending",
                    pending_count,
                )

                def _commit_checkpoint_trigger(
                    key: str,
                    row: Mapping[str, Any],
                ) -> None:
                    if checkpoint_store is None or runtime_fingerprint_sha256 is None:
                        raise AxBenchBpbAdvError(
                            "internal invariant broken: Faster-GCG checkpoint callback "
                            "was invoked without a prepared store/runtime fingerprint"
                        )
                    checkpoint_store.commit_trigger(
                        key,
                        row,
                        runtime_fingerprint_sha256,
                    )

                checkpoint_commit_callback = _commit_checkpoint_trigger
            for target_name, bundle in sorted(target_bundles_for_prompt.items()):
                target_rows = [
                    row for row in prompt_rows if str(row["target_model"]) == target_name
                ]
                if not target_rows:
                    continue
                candidate_eval_device_batch_size = (
                    int(cfg.device_batch_size)
                    if faster_cfg is None
                    or faster_cfg.candidate_eval_device_batch_size is None
                    else int(faster_cfg.candidate_eval_device_batch_size)
                )
                logger.info(
                    "candidate build target={} prompt_rows={} "
                    "candidate_eval_algorithm=shared_prefix_kv "
                    "faster_gcg_candidate_eval_device_batch_size={} "
                    "gepa_candidate_eval_device_batch_size={}",
                    target_name,
                    len(target_rows),
                    candidate_eval_device_batch_size,
                    cfg.device_batch_size,
                )
                objective = BpbObjectiveCallbacks(
                    target_bundle=bundle,
                    target_model_meta=target_metas[target_name],
                    tokenizer=resolved_tokenizer,
                    token_bytes=token_bytes,
                    source_paths=source_paths,
                    pair_scorer=scorer,
                    context_segments_by_prefix_hash=context_segments_by_hash,
                    device=t.device(runtime_state.target_device()),
                    lora=lora,
                    device_batch_size=cfg.device_batch_size,
                    candidate_eval_device_batch_size=(
                        candidate_eval_device_batch_size
                    ),
                )
                allowed = [
                    int(token_id)
                    for token_id in build_allowed_token_ids(resolved_tokenizer)
                    if 0 <= int(token_id) < int(objective.embedding_weight().shape[0])
                ]
                if not allowed:
                    raise AxBenchBpbAdvError(
                        f"allowed token filter produced no ids for {target_name!r}"
                    )
                faster_rows = pa.Table.from_pylist(
                    [
                        row
                        for row in target_rows
                        if str(row["adversarial_method"]) == AdversarialMethod.FASTER_GCG.value
                    ],
                    schema=ATTACK_ROWS_SCHEMA,
                )
                if faster_rows.num_rows:
                    if faster_cfg is None:
                        raise AxBenchBpbAdvError(
                            "Faster-GCG rows require a faster_gcg adversarial method config"
                        )
                    logger.info(
                        "building universal Faster-GCG candidates for target={}: "
                        "rows={} max_steps={} candidate_batch_size={} top_k={} "
                        "minibatch={}",
                        target_name,
                        faster_rows.num_rows,
                        faster_cfg.max_steps,
                        faster_cfg.candidate_batch_size,
                        faster_cfg.top_k,
                        faster_cfg.universal_minibatch_size,
                    )
                    record_counter(
                        "axbench_bpb_adv.build_candidates.faster_gcg_candidate_eval",
                        1,
                        target_model=target_name,
                        algorithm="shared_prefix_kv",
                        device_batch_size=candidate_eval_device_batch_size,
                    )
                    with profile_region(
                        "axbench_bpb_adv.build_candidates.universal_faster_gcg",
                        target_model=target_name,
                        rows=faster_rows.num_rows,
                        candidate_eval_algorithm="shared_prefix_kv",
                        candidate_eval_device_batch_size=(
                            candidate_eval_device_batch_size
                        ),
                    ):
                        faster_table, faster_triggers = build_universal_gcg_candidates(
                            attack_rows=faster_rows,
                            tokenizer=resolved_tokenizer,
                            cfg=faster_cfg,
                            embedding_weight=objective.embedding_weight(),
                            allowed_token_ids=allowed,
                            objective=objective.objective_batch,
                            gradient_provider_batch=objective.gradient_provider_batch,
                            progress_callback=_candidate_progress_logger(
                                family=AdversarialMethod.FASTER_GCG.value,
                                target_name=target_name,
                                log_every=cfg.log_progress_every,
                            ),
                            # v2 C3: evict the per-(attack_row_id,digest) eval cache at
                            # each concept-group boundary (its bound under a minibatch).
                            group_boundary_callback=objective.clear_eval_cache,
                            # Rank on the model's device. The surrogate embedding
                            # snapshot is CPU (the gradient capture forces it there),
                            # and the ranking inherited that placement, so the
                            # (P, D) @ (D, V) scoring matmul over a 0.66 GiB pool ran
                            # on the host: profiled at 14.9% of build_candidates,
                            # 55.2 ms/call. The pool is materialized ONCE for the whole
                            # search (one edit space outside the group loop), so this
                            # is a single transfer, not a per-group cost.
                            pool_device=cfg.device,
                            restored_triggers={
                                key: row
                                for key, row in restored_triggers.items()
                                if str(row["target_model"]) == target_name
                            },
                            on_trigger_complete=checkpoint_commit_callback,
                        )
                    tables.append(faster_table)
                    trigger_tables.append(
                        pa.Table.from_pylist(faster_triggers, schema=UNIVERSAL_TRIGGERS_SCHEMA)
                    )
                # The faster_gcg builder evicts at each GROUP boundary (fired at a
                # group's START), so the LAST group's entries -- which can hold CPU
                # gradient tensors -- still linger here. Clear before GEPA (which
                # reuses the SAME objective/cache and would otherwise pile its own
                # evals on top) so each stage's cache is bounded to its own working
                # set (#546 review #6).
                objective.clear_eval_cache()
                gepa_rows = pa.Table.from_pylist(
                    [
                        row
                        for row in target_rows
                        if str(row["adversarial_method"]) == AdversarialMethod.GEPA.value
                    ],
                    schema=ATTACK_ROWS_SCHEMA,
                )
                if gepa_rows.num_rows:
                    if gepa_cfg is None:
                        raise AxBenchBpbAdvError(
                            "GEPA rows require a gepa adversarial method config"
                        )
                    logger.info(
                        "building GEPA candidates for target={}: rows={}",
                        target_name,
                        gepa_rows.num_rows,
                    )
                    with profile_region(
                        "axbench_bpb_adv.build_candidates.gepa",
                        target_model=target_name,
                        rows=gepa_rows.num_rows,
                    ):
                        tables.append(
                            build_gepa_candidates(
                                attack_rows=gepa_rows,
                                tokenizer=resolved_tokenizer,
                                optimizer=_gepa_optimizer_for_objective(
                                    cfg=gepa_cfg,
                                    objective=objective,
                                ),
                                progress_callback=_candidate_progress_logger(
                                    family=AdversarialMethod.GEPA.value,
                                    target_name=target_name,
                                    log_every=cfg.log_progress_every,
                                ),
                            )
                        )
                # Free this target's GEPA evals before the next target (or before
                # finishing): the per-group eviction never fires for GEPA, so its
                # cache would otherwise persist until the objective is replaced on
                # the next loop iteration (#546 review #6).
                objective.clear_eval_cache()
        candidates = concat_candidate_tables(tables)
        candidates = bind_clustered_table(
            candidates,
            paths=paths,
            contract=cluster_contract,
            artifact_family="candidates",
            source_identity=source_identity,
        )
        logger.info(f"candidate stage built {candidates.num_rows} total candidate rows")
    finally:
        if state is not None and cfg.offload_between_stages:
            state.offload_models()
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    # Atomic write (tmp + fsync + replace), matching the universal_triggers write
    # below: a crash mid-write must not leave a torn candidates.parquet for the
    # score / general_val stages that resume from it (#546 review #13).
    with profile_region(
        "axbench_bpb_adv.build_candidates.write_candidates",
        rows=candidates.num_rows,
    ):
        candidate_snapshot = None
        if cluster_contract is None:
            atomic_write_parquet(candidates, paths.candidates_path())
        else:
            candidate_snapshot = write_clustered_table_atomic(
                candidates,
                paths.candidates_path(),
                paths=paths,
                contract=cluster_contract,
                artifact_family="candidates",
                source_identity=source_identity,
            )
    # The canonical universal-triggers artifact (one row per non-control concept
    # group, concatenated across targets). Always written -- an empty set still
    # serializes a schema-stable table -- so general_val can unconditionally read
    # paths.universal_triggers_path(). Atomic write (tmp + fsync + replace) so a
    # crash never leaves a torn parquet for the downstream general_val read.
    universal_triggers = concat_universal_trigger_tables(trigger_tables)
    universal_triggers = bind_clustered_table(
        universal_triggers,
        paths=paths,
        contract=cluster_contract,
        artifact_family="universal_triggers",
        source_identity=source_identity,
        candidate_snapshot=candidate_snapshot,
    )
    with profile_region(
        "axbench_bpb_adv.build_candidates.write_universal_triggers",
        rows=universal_triggers.num_rows,
    ):
        if cluster_contract is None:
            atomic_write_parquet(
                universal_triggers,
                paths.universal_triggers_path(),
            )
        else:
            write_clustered_table_atomic(
                universal_triggers,
                paths.universal_triggers_path(),
                paths=paths,
                contract=cluster_contract,
                artifact_family="universal_triggers",
                source_identity=source_identity,
                candidate_snapshot=candidate_snapshot,
            )
    record_counter(
        "axbench_bpb_adv.build_candidates.candidate_rows_built",
        candidates.num_rows,
    )
    record_counter(
        "axbench_bpb_adv.build_candidates.universal_trigger_rows",
        universal_triggers.num_rows,
    )
    logger.info(
        "candidate stage wrote {} universal-trigger rows to {}",
        universal_triggers.num_rows,
        paths.universal_triggers_path(),
    )
    return int(candidates.num_rows)
