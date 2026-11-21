"""Completion sampling for the AxBench-BPB-Adv eval (Feature A).

A finished BPB-Adv run is scored TEACHER-FORCED -- there is no generated text in
the score table. This module produces the missing qualitative artifact: for each
steering FAMILY (at its representative coefficient -- the SAME one the plot
selects) it samples ``n`` decoder-only completions at every counter-steering
strength, from a handful of concept core-prefixes, under the EXACT steered
intervention the eval scores, and assembles one nested YAML doc per family.

Three concerns, all importing only pure dependencies (the model-backed path is
the lone exception -- it threads ``apply_intervention`` + ``InferenceEngine``):

* **selection (2.2, model-free):** pick the families + their representative
  coefficient (reusing ``aggregation._select_representative_coefficient`` so the
  YAML and the plot agree), the first-N concepts and their first-N prefixes, and
  the attacked prompt tokens for one datapoint.
* **generation (2.3):** ``generate_completions_for_datapoint`` runs decoder-only
  sampling under one ``Intervention`` (built by the shared
  ``intervention_build.build_steered_intervention``, proven byte-identical to the
  scorer's). Seeds are ``base_seed + i`` reused ACROSS strengths so SAMPLING noise
  is held fixed. What actually varies a datapoint depends on the adversarial
  method: for ``additive_direction`` the prompt is fixed and only the (additive
  counter) intervention varies; for prompt-token attacks (``faster_gcg`` /
  ``gepa``) the BASE intervention is constant and the ATTACKED PROMPT varies per
  strength. Under the UNIVERSAL ``faster_gcg`` optimizer a concept's prefixes
  share ONE trigger per strength, so a datapoint's candidates (one per rich x
  avoid pair) carry an IDENTICAL attacked prompt + shared ``objective_value`` --
  the displayed prompt is the representative (min ``objective_value``,
  ``candidate_id`` tie-break; the tie-break is what actually decides among the
  now-equal candidates), still distinct across strengths, while the scored metric
  averages over all candidates.
* **assembly (2.4, pure):** ``build_family_yaml_doc`` joins per-datapoint metrics
  (``primary_margin_y`` / ``general_val_bpb`` / ``adversarial_strength`` /
  ``is_control``) onto the generated completions in a nested dict.

Encoder-decoder sampling reuses the exact scored context cache and persisted
fair/roofline conditioning bags. Clustered runs validate representative-only
context and target IDs immediately before encoder packing.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_text
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb_adv.aggregation import (
    _join_general_val_bpb,
    _join_roundtrip_general_val_bpb,
    _representative_family_key,
    _select_representative_coefficient,
    aggregate_bpb_adv_scores,
    aggregate_roundtrip_adv_scores,
    compute_macro_aggregation,
    compute_per_candidate_aggregation,
    compute_roundtrip_macro_aggregation,
    compute_roundtrip_per_candidate_aggregation,
)
from steering.eval.axbench_bpb_adv.artifact_identity import (
    ClusteredCandidateSnapshot,
    assert_table_cluster_identity,
    clustered_artifact_identity,
    load_clustered_candidate_snapshot,
    require_clustered_candidate_snapshot,
)
from steering.eval.axbench_bpb_adv.candidate_metadata import (
    CandidateMetadata,
    candidate_metadata_to_dict,
    parse_candidate_metadata_json,
)
from steering.eval.axbench_bpb_adv.cluster_preflight import (
    build_cluster_contract_request,
)
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    require_clustered_stage_source,
    resolve_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    AxBenchBpbAdvConfig,
    SteeringMetric,
    get_additive_direction_config,
    validate_config,
)
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.axbench_bpb_adv.direction_requirements import (
    requirements_from_candidate_rows,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.intervention_build import build_steered_intervention
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.roundtrip_scoring import _roundtrip_generation_inputs
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import (
    _bpb_config_from_adv_config,
    _concepts_from_entries,
    _preflight_direction_scales_for_rows,
    _read_concepts,
    load_context_segments_by_prefix_hash,
)
from steering.eval.axbench_bpb_adv.source_validation import (
    BpbAdvSourceContract,
    validate_stage_0_3_source_run,
)
from steering.eval.cluster_feature_contract import (
    ensure_feature_contract_run_marker,
    resolve_cluster_feature_contract,
)
from steering.eval.inference_engine import InferenceEngine, build_engine_for_features
from steering.immutable_io import ImmutableArtifactError, install_bytes_once

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.source_validation import (
        BpbAdvSourceValidation,
    )
    from steering.eval.cluster_feature_contract import ClusterFeatureContract
    from steering.eval.target_conditioning import (
        ContextFeatureSegment,
        SingleEntryContextProvider,
    )

# ----------------------------------------------------------------------------
# 2.2 -- pure selection + prompt retrieval (model-free)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenFamily:
    """One steering family at its representative coefficient.

    The identity fields mirror ``aggregation._representative_family_key`` (mode,
    method_name, target_model, adversarial_method, context_signal_mode);
    ``representative_coefficient`` is the coefficient the plot selects for this
    family (``None`` when no usable control margin exists, in which case the
    family is kept but has no single coefficient).
    """

    mode: str
    method_name: str
    target_model: str
    adversarial_method: str
    context_signal_mode: str
    representative_coefficient: str | None

    def family_id(self) -> str:
        """Stable human-readable identity string for the YAML's ``family`` key."""

        return (
            f"{self.mode}|{self.method_name}|{self.target_model}|"
            f"{self.adversarial_method}|{self.context_signal_mode}"
        )


@dataclass(frozen=True)
class GenPrefix:
    """One concept core-prefix (the seed completion text + its id)."""

    core_prefix_id: str
    prefix_text: str


@dataclass(frozen=True)
class GenConcept:
    """One selected concept with its chosen core-prefixes."""

    concept_id: int
    concept_label: str
    prefixes: list[GenPrefix]


@dataclass(frozen=True)
class FamilyStrengthAxis:
    """The counter-steering strength axis of ONE family, sourced from candidates.

    ``by_prefix`` maps ``(concept_id, core_prefix_id)`` to ``{strength_index ->
    adversarial_strength}`` -- the distinct strengths the SAME family swept for
    that mode-correct prefix. Reading the axis from the candidate rows (rather than
    the per_candidate aggregates) guarantees the prompt the completions are sampled
    from and the reported ``adversarial_strength`` come from the SAME per-prefix
    source, so they can never desync. ``strength_unit`` is the family-wide sweep
    unit (e.g. ``counter_direction_magnitude``); ``None`` only when the family has
    no candidate rows at all.
    """

    by_prefix: dict[tuple[int, str], dict[int, float]]
    strength_unit: str | None

    def points_for(self, concept_id: int, core_prefix_id: str) -> dict[int, float]:
        """``{strength_index -> adversarial_strength}`` for one (concept, prefix)."""

        return self.by_prefix.get((concept_id, core_prefix_id), {})


def _gen_family_key(family: GenFamily) -> tuple[Any, ...]:
    return (
        family.mode,
        family.method_name,
        family.target_model,
        family.adversarial_method,
        family.context_signal_mode,
    )


def families_with_representative_coefficient(
    macro_rows: list[dict[str, Any]],
    *,
    target_model: str,
    baseline_metric: str = "baseline_primary_margin",
    allow_encdec: bool = False,
) -> list[GenFamily]:
    """Build one ``GenFamily`` per steering family of ``target_model``.

    Filters the macro rows to ``target_model``, runs the SAME representative-
    coefficient selector the plot uses
    (``aggregation._select_representative_coefficient``), and emits one
    ``GenFamily`` per ``_representative_family_key`` carrying that family's
    selected coefficient. The result is sorted deterministically by the family's
    identity fields. A family that selects ``None`` (no usable control margin) is
    KEPT with ``representative_coefficient=None``.
    """

    if target_model == "encdec" and not allow_encdec:
        raise AxBenchBpbAdvError(
            "families_with_representative_coefficient: encoder-decoder sampling is "
            "a deliberate follow-up; only target_model='decoder_only' is supported"
        )
    if target_model not in {"decoder_only", "encdec"}:
        raise AxBenchBpbAdvError(
            "families_with_representative_coefficient: target_model must be "
            f"'decoder_only' or 'encdec', got {target_model!r}"
        )
    filtered = [row for row in macro_rows if str(row["target_model"]) == target_model]
    representative = _select_representative_coefficient(
        filtered,
        baseline_metric=baseline_metric,
        baseline_rows=macro_rows,
    )
    families: dict[tuple[Any, ...], GenFamily] = {}
    for row in filtered:
        family_key = _representative_family_key(row)
        if family_key in families:
            continue
        families[family_key] = GenFamily(
            mode=str(row["mode"]),
            method_name=str(row["method_name"]),
            target_model=str(row["target_model"]),
            adversarial_method=str(row["adversarial_method"]),
            context_signal_mode=str(row["context_signal_mode"]),
            representative_coefficient=representative.get(family_key),
        )
    return sorted(families.values(), key=_gen_family_key)


def select_concepts_and_prefixes(
    candidate_rows: list[dict[str, Any]],
    *,
    n_concepts: int,
    n_prefixes_per_concept: int,
    concept_labels: dict[int, str],
) -> list[GenConcept]:
    """Select the first ``n_concepts`` concepts and their first prefixes.

    Groups candidate rows by ``concept_id``, takes the first ``n_concepts``
    (sorted ascending by ``concept_id``), and for each takes the first
    ``n_prefixes_per_concept`` distinct ``(core_prefix_id, prefix_text)`` pairs
    (sorted ascending by ``core_prefix_id``). Logs how many concepts/prefixes
    were available vs chosen so no truncation is silent. A concept_id absent from
    ``concept_labels`` raises ``AxBenchBpbAdvError`` rather than producing a bogus
    label.
    """

    by_concept: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_concept.setdefault(int(row["concept_id"]), []).append(row)

    all_concept_ids = sorted(by_concept)
    chosen_concept_ids = all_concept_ids[:n_concepts]
    logger.info(
        "select_concepts_and_prefixes: {} concepts available, {} chosen (n_concepts={})",
        len(all_concept_ids),
        len(chosen_concept_ids),
        n_concepts,
    )

    concepts: list[GenConcept] = []
    for concept_id in chosen_concept_ids:
        if concept_id not in concept_labels:
            raise AxBenchBpbAdvError(
                f"no concept label for concept_id={concept_id}; concept_labels must "
                "cover every selected concept"
            )
        # Distinct (core_prefix_id, prefix_text) sorted by core_prefix_id.
        distinct: dict[str, str] = {}
        for row in by_concept[concept_id]:
            core_prefix_id = str(row["core_prefix_id"])
            if core_prefix_id not in distinct:
                distinct[core_prefix_id] = str(row["prefix_text"])
        all_prefix_ids = sorted(distinct)
        chosen_prefix_ids = all_prefix_ids[:n_prefixes_per_concept]
        logger.info(
            "select_concepts_and_prefixes: concept_id={} has {} prefixes "
            "available, {} chosen (n_prefixes_per_concept={})",
            concept_id,
            len(all_prefix_ids),
            len(chosen_prefix_ids),
            n_prefixes_per_concept,
        )
        concepts.append(
            GenConcept(
                concept_id=concept_id,
                concept_label=str(concept_labels[concept_id]),
                prefixes=[
                    GenPrefix(
                        core_prefix_id=core_prefix_id,
                        prefix_text=distinct[core_prefix_id],
                    )
                    for core_prefix_id in chosen_prefix_ids
                ],
            )
        )
    return concepts


def _datapoint_matches(
    row: dict[str, Any],
    *,
    concept_id: int,
    core_prefix_id: str,
    method_name: str,
    adversarial_method: str,
    coefficient_str: str,
    strength_index: int,
    mode: str,
) -> bool:
    return (
        int(row["concept_id"]) == concept_id
        and str(row["core_prefix_id"]) == core_prefix_id
        and str(row["method_name"]) == method_name
        and str(row["adversarial_method"]) == adversarial_method
        and str(row["coefficient_str"]) == coefficient_str
        and int(row["strength_index"]) == strength_index
        and str(row["mode"]) == mode
    )


@dataclass(frozen=True)
class RepresentativeAttack:
    """The single attacked prompt chosen to represent one datapoint.

    Selected once per (concept, prefix, strength) and used for BOTH generation
    (``prompt_tokens``) and the YAML view (``attacked_prompt_text`` /
    ``replacement_text`` / ``objective_value``), so the displayed prompt can never
    desync from the prompt the completions were sampled from. ``attacked_prompt_text``
    and ``replacement_text`` are the chosen candidate's STORED fields (decoded at
    candidate-build time by the same model tokenizer, ``optimize._candidate_row``),
    i.e. exactly what the eval scored -- not a re-decode. ``objective_value`` is the
    selected candidate's search objective ``g`` (``None`` for control / additive /
    gepa).

    Under the UNIVERSAL ``faster_gcg`` optimizer this ``g`` is the CONCEPT-LEVEL
    shared-trigger objective -- the group-aggregated (mean-over-pairs) value the one
    trigger achieved for the whole concept group -- NOT a per-prefix strongest single
    attack. Because that one trigger is substituted into every prefix-pair, a
    datapoint's candidates carry the SAME ``attacked_prompt_text`` /
    ``replacement_text`` / ``objective_value`` and differ only in ``candidate_id``, so
    the representative is well-defined (the min-``objective_value`` selection reduces
    to the ``candidate_id`` tie-break, and any candidate yields the same shared
    fields). It remains distinct from the datapoint-averaged ``primary_margin_y``
    (a different statistic over the same candidates).
    """

    prompt_tokens: list[int]
    attacked_prompt_text: str
    replacement_text: str
    objective_value: float | None
    candidate_id: str
    # The chosen candidate's typed ``candidate_metadata_json`` payload. Faster-GCG
    # carries optimization progress; k=0 and GEPA carry their candidate type.
    optimization: CandidateMetadata


def _representative_sort_key(row: dict[str, Any]) -> tuple[bool, float, str]:
    """Sort key selecting a datapoint's representative attack, None-safe.

    The worst-case adversary minimizes ``objective_value`` (``g``; lower =
    stronger -- see ``optimize.py``), so the representative is the candidate with
    the lowest ``objective_value``. The leading ``objective is None`` term is
    REQUIRED, not cosmetic: ``objective_value`` is ``None`` for additive / control /
    gepa candidates and Python raises ``TypeError`` on ``None < float``. Bucketing
    ``None`` into a bool slot (and substituting ``0.0`` -- never compared against a
    real ``g`` because the bool slot already separates the groups) keeps ``min``
    total and crash-free. A row WITH an objective sorts before one without
    (``False < True``); among real attacks the lowest ``g`` wins; ``candidate_id``
    breaks exact ties deterministically.

    Under the UNIVERSAL ``faster_gcg`` optimizer the real-attack case ALSO ties:
    a datapoint's candidates share the one trigger's concept-level ``g``, so the
    ``candidate_id`` tie-break is what decides among them (and every candidate
    yields the same surfaced prompt anyway, so the choice is well-defined). When
    every candidate's objective is ``None`` (a control / additive / gepa datapoint
    -- uniform because ``k`` is fixed within a datapoint) this reduces EXACTLY to
    ``min(candidate_id)``: the pre-#543 behavior that keeps additive output
    byte-identical.
    """

    objective = row["objective_value"]
    return (
        objective is None,
        float(objective) if objective is not None else 0.0,
        str(row["candidate_id"]),
    )


def representative_attack_for_datapoint(
    candidate_rows: list[dict[str, Any]],
    *,
    concept_id: int,
    core_prefix_id: str,
    method_name: str,
    adversarial_method: str,
    coefficient_str: str,
    strength_index: int,
    mode: str,
) -> RepresentativeAttack:
    """Return the representative attacked prompt for exactly one datapoint.

    Filters ``candidate_rows`` to the (concept, core_prefix, method, adversarial,
    coefficient, strength_index, mode) datapoint -- which may hold
    MULTIPLE candidates (one per rich x avoid pair) -- and selects the representative
    (lowest ``objective_value``, ``candidate_id`` tie-break) via
    ``_representative_sort_key``. Under the UNIVERSAL ``faster_gcg`` optimizer those
    candidates share ONE concept-level trigger, so they carry an IDENTICAL attacked
    prompt + ``objective_value`` and the selection is decided by the ``candidate_id``
    tie-break (well-defined: every candidate yields the same surfaced fields). The
    scorer scores EVERY candidate of the datapoint and averages; this picks a single
    representative to seed the qualitative sample AND to display. Returns a
    ``RepresentativeAttack`` carrying the chosen candidate's
    ``attacked_prompt_tokens`` (for generation), stored ``attacked_prompt_text`` /
    ``replacement_text`` (for the YAML), ``objective_value``, and ``candidate_id``.
    Raises ``AxBenchBpbAdvError`` if no candidate matches.
    """

    matches = [
        row
        for row in candidate_rows
        if _datapoint_matches(
            row,
            concept_id=concept_id,
            core_prefix_id=core_prefix_id,
            method_name=method_name,
            adversarial_method=adversarial_method,
            coefficient_str=coefficient_str,
            strength_index=strength_index,
            mode=mode,
        )
    ]
    if not matches:
        raise AxBenchBpbAdvError(
            "representative_attack_for_datapoint: no candidate row for "
            f"concept_id={concept_id} core_prefix_id={core_prefix_id!r} "
            f"method_name={method_name!r} adversarial_method={adversarial_method!r} "
            f"coefficient_str={coefficient_str!r} strength_index={strength_index} "
            f"mode={mode!r}"
        )
    chosen = min(matches, key=_representative_sort_key)
    return RepresentativeAttack(
        prompt_tokens=[int(token_id) for token_id in chosen["attacked_prompt_tokens"]],
        attacked_prompt_text=str(chosen["attacked_prompt_text"]),
        replacement_text=str(chosen["replacement_text"]),
        objective_value=(
            None if chosen["objective_value"] is None else float(chosen["objective_value"])
        ),
        candidate_id=str(chosen["candidate_id"]),
        optimization=parse_candidate_metadata_json(chosen.get("candidate_metadata_json")),
    )


# ----------------------------------------------------------------------------
# 2.3 -- decoder-only generation under intervention
# ----------------------------------------------------------------------------


def generate_completions_for_datapoint(
    *,
    model: Any,
    tokenizer: Any,
    device: t.device,
    intervention: Intervention,
    prompt_tokens: list[int],
    n_completions: int,
    temperature: float,
    top_k: int | None,
    max_new_tokens: int,
    base_seed: int,
    lora: LoraParams | None = None,
    context_provider: SingleEntryContextProvider | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    pre_forward_guard: Callable[[], None] | None = None,
) -> list[str]:
    """Sample ``n_completions`` completions under ``intervention``.

    Installs the steered (+ optional additive counter) intervention via the SAME
    ``apply_intervention`` context manager the scorer/surrogate use (``lora`` is
    keyword-only and threaded through identically), then streams ``n_completions``
    completions from ``prompt_tokens`` with seeds ``base_seed + i``. Reusing the
    seed offsets across strengths holds SAMPLING noise fixed, so a completion's
    drift across strengths is attributable to the swept axis -- which differs by
    adversarial method: for ``additive_direction`` the prompt is fixed and the
    intervention varies, whereas for prompt-token attacks (``faster_gcg`` /
    ``gepa``) the base intervention is constant and the ATTACKED ``prompt_tokens``
    vary per strength (the caller passes the datapoint's representative -- lowest
    ``objective_value``, ``candidate_id`` tie-break; under the universal trigger a
    datapoint's candidates share that prompt so the tie-break decides -- while the
    scored metric averages over all candidates).
    Returns the decoded completion strings.

    Encoder-decoder targets use the exact scored prefix context through
    ``build_engine_for_features``. A clustered target validates the fixed target
    bag immediately before the wrapper packs it.
    """
    if intervention.target_model == "encdec" and cluster_contract is None:
        raise AxBenchBpbAdvError(
            "generate_completions_for_datapoint: encoder-decoder generation is a "
            "deliberate follow-up; only decoder-only interventions are supported"
        )
    completions: list[str] = []
    with apply_intervention(model, intervention, lora=lora):
        if intervention.target_model == "decoder_only":
            if context_provider is not None:
                raise AxBenchBpbAdvError(
                    "decoder-only sample generation must not receive a context provider"
                )
            engine = InferenceEngine(model, tokenizer, device)
        elif intervention.target_model == "encdec":
            if context_provider is None:
                raise AxBenchBpbAdvError(
                    "encoder-decoder sample generation requires the scored context cache"
                )
            feature_conditions = dict(intervention.feature_conditions or {})
            # A real runtime check, deliberately not a typing-only assertion.
            # The entry guard above already rejects encdec-without-contract, so
            # this is unreachable today -- but a type-checker promise is not an
            # assertion, and the previous form's safety rested on the broad
            # `except Exception` below converting the resulting AttributeError
            # into a clean error. A future caller reaching this branch deserves
            # a named error, not an incidental one.
            if cluster_contract is None:
                raise AxBenchBpbAdvError(
                    "generate_completions_for_datapoint: reached encoder packing "
                    "for an encoder-decoder target without a validated clustered "
                    "feature contract"
                )
            validated_contract = cluster_contract
            ids = t.tensor(sorted(feature_conditions), dtype=t.int64)
            try:
                validated_contract.consolidation.assert_representatives_only(
                    ids,
                    surface="sample generation encoder input",
                )
                validated_contract.feature_mask.assert_no_masked(ids)
            except Exception as exc:
                raise AxBenchBpbAdvError(
                    "sample generation reached encoder packing with an invalid "
                    f"clustered bag: {exc}"
                ) from exc
            engine = build_engine_for_features(
                model,
                tokenizer,
                device,
                feature_conditions,
                context_provider=context_provider,
            )
        else:
            raise AxBenchBpbAdvError(
                f"unsupported sample-generation target {intervention.target_model!r}"
            )
        for i in range(n_completions):
            if cluster_contract is not None and pre_forward_guard is None:
                raise AxBenchBpbAdvError(
                    "clustered sample generation is missing its validated "
                    "pre-forward checkpoint guard"
                )
            if pre_forward_guard is not None:
                pre_forward_guard()
            ids = list(
                engine.generate_stream(
                    token_ids=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=base_seed + i,
                )
            )
            completions.append(tokenizer.decode(ids))
    return completions


# ----------------------------------------------------------------------------
# 2.4 -- YAML doc assembly (pure)
# ----------------------------------------------------------------------------


def _datapoint_metrics_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Omits ``target_model`` and ``context_signal_mode`` safely: ``method_name``
    # -> ``target_model`` and ``adversarial_method`` -> ``context_signal_mode`` are
    # each 1:1 within a family, so neither adds discriminating power here. (A
    # collision under this key would mean one of those invariants broke, which the
    # per-row metrics filter in ``build_family_yaml_doc`` -- which fixes
    # method_name/adversarial_method/mode -- would surface, not hide.)
    return (
        int(row["concept_id"]),
        str(row["coefficient_str"]),
        str(row["adversarial_method"]),
        int(row["strength_index"]),
        str(row["mode"]),
    )


# Adversarial methods whose ATTACKED PROMPT varies per datapoint. Only these
# families surface attacked_prompt_text / replacement_text / objective_value;
# additive_direction leaves the prompt unchanged so its datapoints stay
# byte-identical to the pre-#543 output.
PROMPT_TOKEN_ADVERSARIAL_METHODS: frozenset[str] = frozenset(
    {AdversarialMethod.FASTER_GCG.value, AdversarialMethod.GEPA.value}
)


def build_family_yaml_doc(
    *,
    family: GenFamily,
    concepts: list[GenConcept],
    strength_axis: FamilyStrengthAxis,
    per_candidate_rows: list[dict[str, Any]],
    completions_by_key: dict[tuple[int, str, int], list[str]],
    attack_by_key: dict[tuple[int, str, int], RepresentativeAttack],
    sampling: dict[str, Any],
    scatter_baseline_x: str,
    primary_metric_field: str = "primary_margin_y",
    primary_metric_output_key: str = "primary_margin_y",
) -> dict[str, Any]:
    """Assemble the nested per-family YAML doc (pure -- no model, no IO).

    The datapoints of each (concept, prefix) are the sorted distinct
    ``strength_index`` of that (concept, prefix) in ``strength_axis`` -- which is
    derived from the family's MODE-CORRECT candidate rows (so the prompt the
    completions were sampled from and the ``adversarial_strength`` reported here
    share the SAME per-prefix source). Per-datapoint metrics (``primary_margin_y``
    / ``general_val_bpb`` / ``is_control``) are joined from ``per_candidate_rows``
    filtered to ``(concept_id, representative_coefficient, adversarial_method,
    strength_index, mode)``. ``completions_by_key`` is keyed
    ``(concept_id, core_prefix_id, strength_index)``; a missing key raises (the
    orchestrator must have generated the completions). ``general_val_bpb`` may be
    null.

    ``attack_by_key`` (same keying) carries the per-datapoint
    ``RepresentativeAttack``. It is consulted ONLY for families whose
    ``adversarial_method`` is in ``PROMPT_TOKEN_ADVERSARIAL_METHODS`` (faster_gcg /
    gepa): those datapoints additionally emit ``attacked_prompt_text`` /
    ``replacement_text`` / ``objective_value`` / ``optimization`` (after
    ``adversarial_strength``); a missing entry raises, symmetric with the
    missing-completion guard. For ``additive_direction`` the attack map is ignored
    and the six pre-#543 keys are emitted unchanged (byte-identical).
    """

    representative_coefficient = family.representative_coefficient
    # Index the per-candidate metrics by their full datapoint identity, keeping
    # only this family's representative coefficient + adversarial_method + mode
    # (so a different coefficient's rows can never leak into the doc).
    metrics_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in per_candidate_rows:
        if str(row["coefficient_str"]) != str(representative_coefficient):
            continue
        if str(row["adversarial_method"]) != family.adversarial_method:
            continue
        if str(row["mode"]) != family.mode:
            continue
        if str(row["method_name"]) != family.method_name:
            continue
        metrics_by_key[_datapoint_metrics_key(row)] = row

    concept_docs: list[dict[str, Any]] = []
    for concept in concepts:
        prefix_docs: list[dict[str, Any]] = []
        for prefix in concept.prefixes:
            datapoint_docs: list[dict[str, Any]] = []
            # The strength set comes from the CANDIDATES (this concept+prefix),
            # never from per_candidate -- so prompt and strength axis agree.
            strength_points = strength_axis.points_for(concept.concept_id, prefix.core_prefix_id)
            for strength_index in sorted(strength_points):
                adversarial_strength = strength_points[strength_index]
                completion_key = (
                    concept.concept_id,
                    prefix.core_prefix_id,
                    strength_index,
                )
                if completion_key not in completions_by_key:
                    raise AxBenchBpbAdvError(
                        "build_family_yaml_doc: no completions for "
                        f"(concept_id={concept.concept_id}, "
                        f"core_prefix_id={prefix.core_prefix_id!r}, "
                        f"strength_index={strength_index}); the orchestrator must "
                        "generate a completion list for every datapoint"
                    )
                metrics_key = (
                    concept.concept_id,
                    str(representative_coefficient),
                    family.adversarial_method,
                    strength_index,
                    family.mode,
                )
                metrics = metrics_by_key.get(metrics_key)
                # FAIL LOUD, symmetric with the missing-completion raise above: an
                # emitted datapoint with NO core metrics row would otherwise carry a
                # non-null ``adversarial_strength`` (from the candidate strength
                # axis) beside null ``is_control`` / ``primary_margin_y`` -- a silent
                # inconsistency. ``general_val_bpb`` legitimately stays null (its
                # absence is allowed); only the core metrics row absence raises.
                if metrics is None:
                    raise AxBenchBpbAdvError(
                        "build_family_yaml_doc: no per_candidate metrics row for "
                        f"(concept_id={concept.concept_id}, "
                        f"coefficient_str={representative_coefficient!r}, "
                        f"adversarial_method={family.adversarial_method!r}, "
                        f"strength_index={strength_index}, "
                        f"mode={family.mode!r}); every emitted datapoint must have "
                        "matching core metrics"
                    )
                # Insertion order is the YAML field order (sort_keys=False). For a
                # prompt-token family the attacked prompt + edit + objective slot in
                # right after adversarial_strength; additive skips the block, so its
                # six-key datapoint stays byte-identical to the pre-#543 output.
                datapoint: dict[str, Any] = {
                    "strength_index": strength_index,
                    "adversarial_strength": adversarial_strength,
                }
                if family.adversarial_method in PROMPT_TOKEN_ADVERSARIAL_METHODS:
                    attack = attack_by_key.get(completion_key)
                    if attack is None:
                        # Symmetric with the missing-completion raise above: a
                        # prompt-token datapoint with no recorded RepresentativeAttack
                        # would emit a GCG sample with no visible attacked prompt.
                        raise AxBenchBpbAdvError(
                            "build_family_yaml_doc: no representative attack for "
                            f"(concept_id={concept.concept_id}, "
                            f"core_prefix_id={prefix.core_prefix_id!r}, "
                            f"strength_index={strength_index}); the orchestrator must "
                            "record a RepresentativeAttack for every prompt-token "
                            "datapoint"
                        )
                    datapoint["attacked_prompt_text"] = attack.attacked_prompt_text
                    datapoint["replacement_text"] = attack.replacement_text
                    # The concept-level shared-trigger g (null at k=0 control / gepa):
                    # under the universal optimizer one trigger serves all of the
                    # datapoint's pair-candidates, so this is the group-aggregated
                    # objective -- distinct from the datapoint-averaged
                    # primary_margin_y below.
                    datapoint["objective_value"] = attack.objective_value
                    # The representative candidate's typed optimization metadata:
                    # faster_gcg surfaces max_steps / accepted_steps /
                    # visited_sequences (how hard the GCG search worked for this
                    # datapoint); the k=0 control + gepa carry only their candidate tag.
                    datapoint["optimization"] = candidate_metadata_to_dict(attack.optimization)
                # Null-tolerant, symmetric with general_val_bpb below: the core
                # metrics columns are nullable (pyarrow default), so a null cell emits
                # None rather than crashing on bool(None)/float(None). A null
                # is_control is ANOMALOUS (a real control flag is always populated) ->
                # surface it as None instead of silently coercing to False.
                datapoint["is_control"] = (
                    None if metrics["is_control"] is None else bool(metrics["is_control"])
                )
                primary_metric = metrics.get(primary_metric_field)
                datapoint[primary_metric_output_key] = (
                    None if primary_metric is None else float(primary_metric)
                )
                datapoint["general_val_bpb"] = (
                    None
                    if metrics["general_val_bpb"] is None
                    else float(metrics["general_val_bpb"])
                )
                datapoint["completions"] = list(completions_by_key[completion_key])
                datapoint_docs.append(datapoint)
            prefix_docs.append(
                {
                    "core_prefix_id": prefix.core_prefix_id,
                    "prefix_text": prefix.prefix_text,
                    "datapoints": datapoint_docs,
                }
            )
        concept_docs.append(
            {
                "concept_id": concept.concept_id,
                "concept_label": concept.concept_label,
                "prefixes": prefix_docs,
            }
        )

    return {
        "family": family.family_id(),
        "mode": family.mode,
        "method_name": family.method_name,
        "target_model": family.target_model,
        "adversarial_method": family.adversarial_method,
        "representative_coefficient": representative_coefficient,
        "scatter_baseline_x": scatter_baseline_x,
        "strength_unit": strength_axis.strength_unit,
        "sampling": sampling,
        "concepts": concept_docs,
    }


# ----------------------------------------------------------------------------
# 2.5 -- orchestration (model loading split from the testable core)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters + concept/prefix fan-out for the generator.

    Mirrors the ``generation:`` config block. ``methods`` is an optional whitelist
    of ``method_name`` values; ``None`` generates every decoder-only family present
    in the macro table.
    """

    n_concepts: int = 5
    n_prefixes_per_concept: int = 1
    n_completions: int = 5
    temperature: float = 0.7
    top_k: int | None = 50
    max_new_tokens: int = 128
    base_seed: int = 0
    methods: list[str] | None = None

    def sampling_dict(self) -> dict[str, Any]:
        """The ``sampling`` sub-doc echoed into every per-family YAML."""

        return {
            "n_completions": int(self.n_completions),
            "temperature": float(self.temperature),
            "top_k": None if self.top_k is None else int(self.top_k),
            "max_new_tokens": int(self.max_new_tokens),
            "base_seed": int(self.base_seed),
        }


def _family_rng(family: GenFamily) -> random.Random:
    """Deterministic, prefix-INDEPENDENT RNG seed for a family's interventions.

    The intervention build is seeded once per family (NOT per prefix/strength) so
    the RNG itself never varies a datapoint's completions -- the swept axis does
    (the additive counter intervention for ``additive_direction``, the attacked
    prompt for ``faster_gcg`` / ``gepa``) -- mirroring the scorer's per-config seed
    derivation in spirit (a stable hash of the family identity).

    This DELIBERATELY diverges from the scorer's per-core-prefix seed: the RNG is
    consumed only by ``ours``-kind negative sampling in ``build_intervention``, and
    the decoder-only families generated here (diffmean / linear_probe / lora /
    control) never touch it, so a per-family seed is byte-equivalent for them and
    simply keeps a family's completions reproducible.
    """

    return random.Random(family.family_id())


def _family_candidates(
    candidate_rows: list[dict[str, Any]], *, family: GenFamily
) -> list[dict[str, Any]]:
    """The candidate rows belonging to ONE family at its representative coefficient.

    Filters ``candidate_rows`` to the family's full identity -- mode, method_name,
    target_model, adversarial_method, representative coefficient.
    Because ``core_prefix_id`` is MODE-SPECIFIC, restricting to ``family.mode``
    here is what makes the downstream concept/prefix selection mode-correct: a
    negative family never sees a positive family's prefixes (the F1 bug).
    """

    representative = str(family.representative_coefficient)
    return [
        row
        for row in candidate_rows
        if str(row["mode"]) == family.mode
        and str(row["method_name"]) == family.method_name
        and str(row["target_model"]) == family.target_model
        and str(row["adversarial_method"]) == family.adversarial_method
        and str(row["coefficient_str"]) == representative
    ]


def _family_strength_axis(
    family_candidates: list[dict[str, Any]], *, family: GenFamily
) -> FamilyStrengthAxis:
    """The counter-steering strength axis for one family, read from its candidates.

    Builds ``(concept_id, core_prefix_id) -> {strength_index ->
    adversarial_strength}`` directly from the family's MODE-CORRECT candidate rows,
    so the prompt the completions are sampled from and the reported
    ``adversarial_strength`` come from the SAME per-prefix source (fixes the
    strength/prompt desync). ``strength_unit`` is the single sweep unit carried on
    those rows; more than one trips a loud raise (a family must sweep one axis).
    """

    by_prefix: dict[tuple[int, str], dict[int, float]] = {}
    strength_units: set[str] = set()
    for row in family_candidates:
        key = (int(row["concept_id"]), str(row["core_prefix_id"]))
        by_prefix.setdefault(key, {})[int(row["strength_index"])] = float(
            row["adversarial_strength"]
        )
        strength_units.add(str(row["strength_unit"]))
    if len(strength_units) > 1:
        raise AxBenchBpbAdvError(
            f"_family_strength_axis: family {family.family_id()!r} has multiple "
            f"strength_units {sorted(strength_units)}; a family must sweep a single "
            "strength axis"
        )
    strength_unit = next(iter(strength_units)) if strength_units else None
    return FamilyStrengthAxis(by_prefix=by_prefix, strength_unit=strength_unit)


def _sample_path(samples_dir: Path, *, family: GenFamily) -> Path:
    """``samples/<mode>/<method>__<adv>.yaml`` for one family."""

    return samples_dir / family.mode / f"{family.method_name}__{family.adversarial_method}.yaml"


def _validate_table(table: pa.Table, schema: pa.Schema, *, name: str) -> None:
    if not table.schema.equals(schema):
        raise AxBenchBpbAdvError(
            f"{name} has an unexpected schema; expected the canonical {schema!r}"
        )


@dataclass(frozen=True)
class _SamplingMetrics:
    """Aggregate tables and metric naming for qualitative sample YAML."""

    macro: pa.Table
    per_candidate: pa.Table
    representative_baseline_metric: str
    primary_metric_field: str
    primary_metric_output_key: str


_CLUSTERED_SAMPLING_SNAPSHOT_TOKEN = object()


@dataclass(frozen=True)
class _ClusteredSamplingSnapshot:
    """Exact in-memory aggregate inputs derived from validated score tables."""

    metrics: _SamplingMetrics = field(repr=False, compare=False)
    steering_metric: str
    candidate_sha256: str
    source_contract_sha256: str
    _token: object | None = field(repr=False, compare=False, default=None)
    _seal: tuple[int, int, str, str, str] | None = field(
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self._token is not _CLUSTERED_SAMPLING_SNAPSHOT_TOKEN:
            raise TypeError(
                "_ClusteredSamplingSnapshot must be built by the clustered "
                "sample aggregate refresh"
            )
        expected_seal = (
            id(self.metrics.macro),
            id(self.metrics.per_candidate),
            self.steering_metric,
            self.candidate_sha256,
            self.source_contract_sha256,
        )
        if self._seal != expected_seal:
            raise AxBenchBpbAdvError(
                "clustered sampling snapshot fields changed after validation"
            )
        metric = SteeringMetric.coerce(self.steering_metric)
        if metric is SteeringMetric.ROUNDTRIP:
            macro_schema = ROUNDTRIP_MACRO_AGG_SCHEMA
            per_candidate_schema = ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA
            expected_names = (
                "primary_roundtrip_y",
                "primary_roundtrip_y",
                "primary_roundtrip_y",
            )
        else:
            macro_schema = MACRO_AGG_SCHEMA
            per_candidate_schema = PER_CANDIDATE_AGG_SCHEMA
            expected_names = (
                "baseline_primary_margin",
                "primary_margin_y",
                "primary_margin_y",
            )
        if (
            not self.metrics.macro.schema.equals(macro_schema)
            or not self.metrics.per_candidate.schema.equals(per_candidate_schema)
        ):
            raise AxBenchBpbAdvError(
                "clustered sampling snapshot has the wrong aggregate schema"
            )
        actual_names = (
            self.metrics.representative_baseline_metric,
            self.metrics.primary_metric_field,
            self.metrics.primary_metric_output_key,
        )
        if actual_names != expected_names:
            raise AxBenchBpbAdvError(
                "clustered sampling snapshot has inconsistent metric names"
            )


def _build_clustered_sampling_snapshot(
    *,
    metrics: _SamplingMetrics,
    steering_metric: str | SteeringMetric,
    candidate_snapshot: ClusteredCandidateSnapshot,
    stage_source: ClusteredStageSource,
) -> _ClusteredSamplingSnapshot:
    metric = SteeringMetric.coerce(steering_metric).value
    seal = (
        id(metrics.macro),
        id(metrics.per_candidate),
        metric,
        candidate_snapshot.sha256,
        stage_source.source_identity.source_contract_sha256,
    )
    return _ClusteredSamplingSnapshot(
        metrics=metrics,
        steering_metric=metric,
        candidate_sha256=candidate_snapshot.sha256,
        source_contract_sha256=(
            stage_source.source_identity.source_contract_sha256
        ),
        _token=_CLUSTERED_SAMPLING_SNAPSHOT_TOKEN,
        _seal=seal,
    )


def _require_clustered_sampling_snapshot(
    snapshot: _ClusteredSamplingSnapshot | None,
    *,
    steering_metric: str | SteeringMetric,
    candidate_snapshot: ClusteredCandidateSnapshot | None,
    stage_source: ClusteredStageSource | None,
    clustered: bool,
) -> _ClusteredSamplingSnapshot | None:
    if not clustered:
        if snapshot is not None:
            raise AxBenchBpbAdvError(
                "unclustered sample generation rejects clustered aggregate inputs"
            )
        return None
    if type(snapshot) is not _ClusteredSamplingSnapshot:
        raise AxBenchBpbAdvError(
            "clustered sample generation requires exact refreshed aggregate inputs"
        )
    if (
        type(candidate_snapshot) is not ClusteredCandidateSnapshot
        or type(stage_source) is not ClusteredStageSource
    ):
        raise AxBenchBpbAdvError(
            "clustered aggregate inputs require paired candidate and source owners"
        )
    snapshot.__post_init__()
    expected_metric = SteeringMetric.coerce(steering_metric).value
    if snapshot.steering_metric != expected_metric:
        raise AxBenchBpbAdvError(
            "clustered sampling snapshot was built for a different steering metric"
        )
    if snapshot.candidate_sha256 != candidate_snapshot.sha256:
        raise AxBenchBpbAdvError(
            "clustered sampling snapshot belongs to different candidate bytes"
        )
    if (
        snapshot.source_contract_sha256
        != stage_source.source_identity.source_contract_sha256
    ):
        raise AxBenchBpbAdvError(
            "clustered sampling snapshot belongs to a different source contract"
        )
    return snapshot


def _load_sampling_metrics(
    paths: AxBenchBpbAdvPaths,
    *,
    steering_metric: str | SteeringMetric,
) -> _SamplingMetrics:
    """Load the aggregate tables that drive sample-generation metadata.

    Select the aggregate tree from the requested ``steering_metric``. Finished
    run dirs may contain both legacy class-margin aggregates and roundtrip
    aggregates after an in-place metric switch; the configured metric is the
    source of truth, never directory-existence order.
    """

    metric = SteeringMetric.coerce(steering_metric)
    legacy_macro_path = paths.aggregates_dir() / "macro.parquet"
    legacy_per_candidate_path = paths.aggregates_dir() / "per_candidate.parquet"
    roundtrip_macro_path = paths.roundtrip_aggregates_dir() / "macro.parquet"
    roundtrip_per_candidate_path = paths.roundtrip_aggregates_dir() / "per_candidate.parquet"

    legacy_exists = (legacy_macro_path.exists(), legacy_per_candidate_path.exists())
    roundtrip_exists = (
        roundtrip_macro_path.exists(),
        roundtrip_per_candidate_path.exists(),
    )

    if metric is SteeringMetric.ROUNDTRIP:
        selected_label = "roundtrip"
        selected_metric = SteeringMetric.ROUNDTRIP.value
        selected_macro_path = roundtrip_macro_path
        selected_per_candidate_path = roundtrip_per_candidate_path
        selected_exists = roundtrip_exists
        selected_macro_schema = ROUNDTRIP_MACRO_AGG_SCHEMA
        selected_per_candidate_schema = ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA
        selected_macro_name = "roundtrip_aggregates/macro.parquet"
        selected_per_candidate_name = "roundtrip_aggregates/per_candidate.parquet"
        other_label = "class-margin"
        other_metric = SteeringMetric.BPB_CLASS_MARGIN.value
        other_macro_path = legacy_macro_path
        other_per_candidate_path = legacy_per_candidate_path
        other_exists = legacy_exists
        representative_baseline_metric = "primary_roundtrip_y"
        primary_metric_field = "primary_roundtrip_y"
        primary_metric_output_key = "primary_roundtrip_y"
    else:
        selected_label = "class-margin"
        selected_metric = SteeringMetric.BPB_CLASS_MARGIN.value
        selected_macro_path = legacy_macro_path
        selected_per_candidate_path = legacy_per_candidate_path
        selected_exists = legacy_exists
        selected_macro_schema = MACRO_AGG_SCHEMA
        selected_per_candidate_schema = PER_CANDIDATE_AGG_SCHEMA
        selected_macro_name = "aggregates/macro.parquet"
        selected_per_candidate_name = "aggregates/per_candidate.parquet"
        other_label = "roundtrip"
        other_metric = SteeringMetric.ROUNDTRIP.value
        other_macro_path = roundtrip_macro_path
        other_per_candidate_path = roundtrip_per_candidate_path
        other_exists = roundtrip_exists
        representative_baseline_metric = "baseline_primary_margin"
        primary_metric_field = "primary_margin_y"
        primary_metric_output_key = "primary_margin_y"

    selected_any = any(selected_exists)
    selected_complete = all(selected_exists)
    other_complete = all(other_exists)
    if selected_any and not selected_complete:
        raise AxBenchBpbAdvError(
            f"sample generation found incomplete {selected_label} aggregates for "
            f"steering_metric={selected_metric!r}; expected both "
            f"{selected_macro_path} and {selected_per_candidate_path}"
        )

    if not selected_complete:
        if other_complete:
            raise AxBenchBpbAdvError(
                "sample generation metric mismatch: requested "
                f"steering_metric={selected_metric!r}, but only {other_label} "
                f"aggregates for steering_metric={other_metric!r} are available "
                f"({other_macro_path} and {other_per_candidate_path}); expected "
                f"{selected_macro_path} and {selected_per_candidate_path}"
            )
        raise AxBenchBpbAdvError(
            "sample generation requires finished aggregate artifacts for "
            f"steering_metric={selected_metric!r}: expected both "
            f"{selected_macro_path} and {selected_per_candidate_path}"
        )

    if other_complete:
        logger.warning(
            "sample generation requested steering_metric={!r} and found complete "
            "non-selected {} aggregates at {} and {}; ignoring them",
            selected_metric,
            other_label,
            other_macro_path,
            other_per_candidate_path,
        )

    macro = pq.read_table(selected_macro_path)
    _validate_table(macro, selected_macro_schema, name=selected_macro_name)
    per_candidate = pq.read_table(selected_per_candidate_path)
    _validate_table(
        per_candidate,
        selected_per_candidate_schema,
        name=selected_per_candidate_name,
    )
    return _SamplingMetrics(
        macro=macro,
        per_candidate=per_candidate,
        representative_baseline_metric=representative_baseline_metric,
        primary_metric_field=primary_metric_field,
        primary_metric_output_key=primary_metric_output_key,
    )


def _generate_family_completions(
    family: GenFamily,
    *,
    concepts: list[GenConcept],
    concepts_by_id: dict[int, Concept],
    family_candidates: list[dict[str, Any]],
    strength_axis: FamilyStrengthAxis,
    model: Any,
    tokenizer: Any,
    device: t.device,
    paths: AxBenchBpbAdvPaths,
    source_paths: SourcePaths,
    target_model_meta: Any,
    inactive_sampler: Any | None,
    diffmean_layer: int,
    counter_direction_family: str,
    generation: GenerationConfig,
    lora: LoraParams | None,
    apply_max_act_scale: bool,
    context_segments_by_prefix_hash: Mapping[
        str, tuple[ContextFeatureSegment, ...]
    ]
    | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    source_validation: BpbAdvSourceValidation | None = None,
    pre_forward_guard: Callable[[], None] | None = None,
) -> tuple[
    dict[tuple[int, str, int], list[str]],
    dict[tuple[int, str, int], RepresentativeAttack],
]:
    """Sample completions for every (concept, prefix, strength) of one family.

    The strength set per (concept, prefix) is read from ``strength_axis`` (built
    from the family's MODE-CORRECT candidates) and the prompt for each datapoint is
    pulled from those same ``family_candidates`` -- so prompt + strength share one
    per-prefix source.

    ``lora`` is the LoRA adapter bundle (``LoraParams``: ``cfg`` + ``layer``)
    threaded through to ``apply_intervention`` for the ``lora`` baseline, exactly
    as the scorer threads it -- so a ``lora_decoder_only`` family generates WITH
    its adapter attached. ``None`` for the (default) non-LoRA families.

    Returns ``(completions_by_key, attack_by_key)``, both keyed
    ``(concept_id, core_prefix_id, strength_index)``. ``attack_by_key`` carries the
    single ``RepresentativeAttack`` that seeded each datapoint's generation, so the
    YAML can surface EXACTLY the prompt the completions were sampled from (one
    selection per datapoint -- never re-selected downstream).
    """

    method = get_method_by_name(family.method_name)
    coefficient = _coefficient_from_str(str(family.representative_coefficient))
    rng = _family_rng(family)
    completions_by_key: dict[tuple[int, str, int], list[str]] = {}
    attack_by_key: dict[tuple[int, str, int], RepresentativeAttack] = {}
    for concept in concepts:
        # The REAL Concept loaded from the source run (fail loud on a missing
        # concept_id -- the selection already validated labels, but this binds the
        # typed Concept the intervention builders are declared to take).
        real_concept = concepts_by_id.get(concept.concept_id)
        if real_concept is None:
            raise AxBenchBpbAdvError(
                f"no Concept for concept_id={concept.concept_id} in the source "
                "run's concepts.json; cannot build its steered intervention"
            )
        for prefix in concept.prefixes:
            strength_points = strength_axis.points_for(concept.concept_id, prefix.core_prefix_id)
            sorted_strength_indices = sorted(strength_points)
            logger.info(
                "generate_samples: family={} concept_id={} core_prefix_id={} ({} strengths)",
                family.family_id(),
                concept.concept_id,
                prefix.core_prefix_id,
                len(sorted_strength_indices),
            )
            for strength_index in sorted_strength_indices:
                adversarial_strength = strength_points[strength_index]
                representative = representative_attack_for_datapoint(
                    family_candidates,
                    concept_id=concept.concept_id,
                    core_prefix_id=prefix.core_prefix_id,
                    method_name=family.method_name,
                    adversarial_method=family.adversarial_method,
                    coefficient_str=str(family.representative_coefficient),
                    strength_index=strength_index,
                    mode=family.mode,
                )
                intervention = build_steered_intervention(
                    method=method,
                    concept=real_concept,
                    coefficient=coefficient,
                    source_paths=source_paths,
                    diffmean_layer=diffmean_layer,
                    target_model_meta=target_model_meta,
                    inactive_sampler=inactive_sampler,
                    tokenizer=tokenizer,
                    rng=rng,
                    adversarial_method=family.adversarial_method,
                    adversarial_strength=adversarial_strength,
                    counter_direction_family=counter_direction_family,
                    target=family.target_model,
                    apply_max_act_scale=apply_max_act_scale,
                    model=model,
                    adv_paths=paths,
                    core_prefix_id=prefix.core_prefix_id,
                    cluster_contract=cluster_contract,
                    source_validation=source_validation,
                )
                generation_prompt_tokens = representative.prompt_tokens
                context_provider = None
                if family.target_model == "encdec":
                    if context_segments_by_prefix_hash is None:
                        raise AxBenchBpbAdvError(
                            "encoder-decoder sample generation requires loaded "
                            "context segments"
                        )
                    representative_rows = [
                        row
                        for row in family_candidates
                        if str(row["candidate_id"]) == representative.candidate_id
                    ]
                    if len(representative_rows) != 1:
                        raise AxBenchBpbAdvError(
                            "sample generation could not resolve the unique candidate "
                            f"row for candidate_id={representative.candidate_id!r}"
                        )
                    inputs = _roundtrip_generation_inputs(
                        candidate_row=representative_rows[0],
                        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
                        tokenizer=tokenizer,
                        intervention=intervention,
                        cluster_contract=cluster_contract,
                    )
                    generation_prompt_tokens = inputs.prompt_tokens
                    context_provider = inputs.context_provider
                completions = generate_completions_for_datapoint(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    intervention=intervention,
                    prompt_tokens=generation_prompt_tokens,
                    n_completions=generation.n_completions,
                    temperature=generation.temperature,
                    top_k=generation.top_k,
                    max_new_tokens=generation.max_new_tokens,
                    base_seed=generation.base_seed,
                    lora=lora,
                    context_provider=context_provider,
                    cluster_contract=cluster_contract,
                    pre_forward_guard=pre_forward_guard,
                )
                key = (concept.concept_id, prefix.core_prefix_id, strength_index)
                completions_by_key[key] = completions
                attack_by_key[key] = representative
    return completions_by_key, attack_by_key


def run_generate_samples_core(
    *,
    paths: AxBenchBpbAdvPaths,
    source_run_dir: str | Path,
    model: Any,
    tokenizer: Any,
    device: t.device,
    target_model: str,
    target_model_meta: Any,
    inactive_sampler: Any | None,
    diffmean_layer: int,
    counter_direction_family: str,
    scatter_baseline_x: str,
    steering_metric: str | SteeringMetric,
    generation: GenerationConfig,
    apply_max_act_scale: bool,
    lora: LoraParams | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    stage_source: ClusteredStageSource | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
    sampling_snapshot: _ClusteredSamplingSnapshot | None = None,
    pre_forward_guard: Callable[[], None] | None = None,
) -> list[Path]:
    """Generate per-datapoint sampled completions for one ALREADY-LOADED bundle.

    The testable core: it takes the loaded decoder-only ``(model, tokenizer,
    device)`` bundle + its metadata (no model loading here) and the finished run
    dir's tables, and for every decoder-only family (at its representative
    coefficient) samples ``generation.n_completions`` completions at every
    counter-steering strength, from the first-N concepts and their first prefixes,
    assembling one nested YAML per family written under ``paths.samples_dir()``.
    Returns the written YAML paths.

    ``lora`` is the LoRA adapter bundle threaded through to ``apply_intervention``
    for the ``lora`` baseline (the SAME bundle the scorer builds from
    ``cfg.lora_adapter``), so a ``lora_decoder_only`` family generates WITH its
    adapter; ``None`` (default) for runs with no LoRA configured.

    Encoder-decoder families reuse the scored context-cache segments and
    precomputed fair/roofline bags. Clustered tables, context, bags, and final
    encoder inputs must all match ``cluster_contract``.
    """
    if target_model == "encdec" and cluster_contract is None:
        raise AxBenchBpbAdvError(
            "run_generate_samples_core: encoder-decoder sampling is a deliberate "
            "follow-up; only target_model='decoder_only' is supported"
        )
    if target_model not in {"decoder_only", "encdec"}:
        raise AxBenchBpbAdvError(
            "run_generate_samples_core: target_model must be 'decoder_only' or "
            f"'encdec', got {target_model!r}"
        )

    resolved_stage_source = require_clustered_stage_source(
        stage_source=stage_source,
        contract=cluster_contract,
        surface="sample generation",
    )
    resolved_candidate_snapshot = require_clustered_candidate_snapshot(
        candidate_snapshot,
        paths=paths,
        contract=cluster_contract,
        source_identity=(
            None
            if resolved_stage_source is None
            else resolved_stage_source.source_identity
        ),
        surface="sample generation",
    )
    resolved_sampling_snapshot = _require_clustered_sampling_snapshot(
        sampling_snapshot,
        steering_metric=steering_metric,
        candidate_snapshot=resolved_candidate_snapshot,
        stage_source=resolved_stage_source,
        clustered=cluster_contract is not None,
    )
    if resolved_candidate_snapshot is None:
        candidates = pq.read_table(paths.candidates_path())
        _validate_table(candidates, CANDIDATES_SCHEMA, name="candidates.parquet")
        assert_table_cluster_identity(
            candidates,
            paths=paths,
            contract=None,
            artifact_family="candidates",
        )
    else:
        candidates = resolved_candidate_snapshot.table
    metrics = (
        _load_sampling_metrics(paths, steering_metric=steering_metric)
        if resolved_sampling_snapshot is None
        else resolved_sampling_snapshot.metrics
    )
    context_segments_by_prefix_hash = (
        load_context_segments_by_prefix_hash(
            paths,
            cluster_contract=cluster_contract,
        )
        if target_model == "encdec"
        else None
    )

    candidate_rows = candidates.to_pylist()
    per_candidate_rows = metrics.per_candidate.to_pylist()
    source_paths = SourcePaths(source_run_dir=Path(source_run_dir))
    # Load the REAL Concept objects ONCE (reusing the scorer's reader), then
    # derive the YAML labels from each concept's description -- so the labels and
    # the typed Concept threaded into build_steered_intervention share one source.
    source_validation = (
        None
        if resolved_stage_source is None
        else resolved_stage_source.validation
    )
    concepts_by_id = (
        _read_concepts(source_paths)
        if source_validation is None
        else _concepts_from_entries(source_validation.concept_entries())
    )
    concept_labels = {
        concept_id: concept.description for concept_id, concept in concepts_by_id.items()
    }
    families = families_with_representative_coefficient(
        metrics.macro.to_pylist(),
        target_model=target_model,
        baseline_metric=metrics.representative_baseline_metric,
        allow_encdec=cluster_contract is not None,
    )
    if generation.methods is not None:
        whitelist = set(generation.methods)
        kept = [family for family in families if family.method_name in whitelist]
        logger.info(
            "generate_samples: methods whitelist keeps {}/{} families ({})",
            len(kept),
            len(families),
            sorted(whitelist),
        )
        families = kept

    selected_families: list[
        tuple[
            GenFamily,
            list[dict[str, Any]],
            FamilyStrengthAxis,
            list[GenConcept],
        ]
    ] = []
    consumed_candidate_rows: list[dict[str, Any]] = []
    for family in families:
        if family.representative_coefficient is None:
            logger.warning(
                "generate_samples: family={} has no representative coefficient "
                "(no usable control margin); skipping (no single coefficient to "
                "sample at)",
                family.family_id(),
            )
            continue
        family_candidates = _family_candidates(candidate_rows, family=family)
        if not family_candidates:
            logger.warning(
                "generate_samples: family={} has a representative coefficient "
                "{!r} but an empty candidate strength slice (no matching candidate "
                "rows to sample at); skipping",
                family.family_id(),
                family.representative_coefficient,
            )
            continue
        strength_axis = _family_strength_axis(family_candidates, family=family)
        concepts = select_concepts_and_prefixes(
            family_candidates,
            n_concepts=generation.n_concepts,
            n_prefixes_per_concept=generation.n_prefixes_per_concept,
            concept_labels=concept_labels,
        )
        if apply_max_act_scale:
            selected_keys = {
                (concept.concept_id, prefix.core_prefix_id)
                for concept in concepts
                for prefix in concept.prefixes
            }
            consumed_candidate_rows.extend(
                row
                for row in family_candidates
                if (int(row["concept_id"]), str(row["core_prefix_id"]))
                in selected_keys
            )
        selected_families.append(
            (family, family_candidates, strength_axis, concepts)
        )

    # Preflight only artifacts reachable after target, method, representative,
    # family, concept, and prefix selection. This remains before the first model
    # generation while avoiding failures from rows this invocation cannot consume.
    if apply_max_act_scale:
        _preflight_direction_scales_for_rows(
            candidate_rows=consumed_candidate_rows,
            concepts_by_id=concepts_by_id,
            source_paths=source_paths,
            expected_layer=diffmean_layer,
            counter_direction_family=counter_direction_family,
            source_validation=source_validation,
            stage="generate_samples",
        )

    samples_dir = paths.samples_dir()
    sampling = generation.sampling_dict()
    written: list[Path] = []
    logger.info(
        "generate_samples: {} {} family/families to sample",
        len(families),
        target_model,
    )
    for family, family_candidates, strength_axis, concepts in selected_families:
        # Selection + the strength axis are PER FAMILY, sourced from the family's
        # MODE-CORRECT candidate rows: core_prefix_id is mode-specific, so a global
        # (mode-agnostic) selection would hand a negative-mode family a positive
        # family's prefix and the prompt lookup would crash the whole run (F1).
        # When the macro lists a representative coefficient but the candidates carry
        # no matching rows (an empty candidate strength slice -- e.g. a coefficient
        # present only in the aggregates), skip LOUDLY -- symmetric with the
        # no-representative-coefficient skip above -- rather than writing a
        # degenerate empty-datapoint YAML. (Every candidate row contributes a
        # (concept, prefix, strength) point, so a non-empty slice always yields a
        # non-empty strength axis.)
        # Mode-correct concept/prefix selection for THIS family (reverts the earlier
        # family-independent hoist, which was the F1 bug).
        completions_by_key, attack_by_key = _generate_family_completions(
            family,
            concepts=concepts,
            concepts_by_id=concepts_by_id,
            family_candidates=family_candidates,
            strength_axis=strength_axis,
            model=model,
            tokenizer=tokenizer,
            device=device,
            paths=paths,
            source_paths=source_paths,
            target_model_meta=target_model_meta,
            inactive_sampler=inactive_sampler,
            diffmean_layer=diffmean_layer,
            counter_direction_family=counter_direction_family,
            generation=generation,
            lora=lora,
            apply_max_act_scale=apply_max_act_scale,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            cluster_contract=cluster_contract,
            source_validation=source_validation,
            pre_forward_guard=pre_forward_guard,
        )
        doc = build_family_yaml_doc(
            family=family,
            concepts=concepts,
            strength_axis=strength_axis,
            per_candidate_rows=per_candidate_rows,
            completions_by_key=completions_by_key,
            attack_by_key=attack_by_key,
            sampling=sampling,
            scatter_baseline_x=scatter_baseline_x,
            primary_metric_field=metrics.primary_metric_field,
            primary_metric_output_key=metrics.primary_metric_output_key,
        )
        if cluster_contract is not None:
            assert resolved_stage_source is not None
            assert resolved_candidate_snapshot is not None
            identity = clustered_artifact_identity(
                paths=paths,
                contract=cluster_contract,
                artifact_family="samples",
                source_identity=resolved_stage_source.source_identity,
                candidate_snapshot=resolved_candidate_snapshot,
            )
            doc = {**identity, **doc}
        out_path = _sample_path(samples_dir, family=family)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(doc, sort_keys=False)
        if cluster_contract is None:
            atomic_write_text(rendered, out_path)
        else:
            try:
                install_bytes_once(
                    rendered.encode("utf-8"),
                    out_path,
                    surface="clustered sample artifact",
                )
            except ImmutableArtifactError as exc:
                raise AxBenchBpbAdvError(
                    "clustered sample artifact content changed on resume; use a "
                    f"fresh run_tag instead of overwriting {out_path}: {exc}"
                ) from exc
        written.append(out_path)
        logger.info("generate_samples: wrote {}", out_path)
    logger.info("generate_samples: wrote {} per-family YAML doc(s)", len(written))
    return written


# ----------------------------------------------------------------------------
# 2.6 -- model-loading entry (thin) + Hydra DictConfig adapter
# ----------------------------------------------------------------------------


def _resolve_counter_direction_family(cfg: AxBenchBpbAdvConfig) -> str:
    """The additive counter-direction family the scorer would use, with fallback.

    Mirrors the scorer: read the additive method's ``counter_direction_family``
    when an additive method is configured, else default to ``"diffmean"`` (the
    same fallback ``run_model_backed_bpb_adv_score_stage`` applies).
    """

    additive_cfg = get_additive_direction_config(cfg)
    if additive_cfg is not None:
        return additive_cfg.counter_direction_family
    return "diffmean"


def _resolve_sample_cluster_stage(
    cfg: AxBenchBpbAdvConfig,
    *,
    paths: AxBenchBpbAdvPaths,
) -> tuple[
    ClusterFeatureContract | None,
    ClusteredStageSource | None,
]:
    """Resolve target, exact source snapshot, and committed source identity."""

    request = build_cluster_contract_request(cfg)
    marker = paths.root / "feature_contract.json"
    if request is None:
        if marker.exists():
            raise AxBenchBpbAdvError(
                "unclustered sample generation rejects cluster-stamped run output"
            )
        return None, None
    required_finished_artifacts = (
        marker,
        paths.source_contract(),
        paths.candidates_path(),
    )
    missing = [path for path in required_finished_artifacts if not path.is_file()]
    if missing:
        raise AxBenchBpbAdvError(
            "clustered sample generation requires an already-finished scored run; "
            "missing completion artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    contract = resolve_cluster_feature_contract(request)
    cfg.encdec_target = {
        **cfg.encdec_target,
        "step": contract.checkpoint.step,
    }
    ensure_feature_contract_run_marker(paths.root, contract)
    validation = validate_stage_0_3_source_run(
        cfg.source_run_dir,
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    if (
        validation.contract.feature_contract_fingerprint
        != contract.feature_contract_fingerprint
    ):
        raise AxBenchBpbAdvError(
            "sample generation source and target feature contracts disagree"
        )
    source_contract_path = paths.source_contract()
    try:
        existing_source_contract = json.loads(
            source_contract_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            "clustered sample generation requires the finished run's exact "
            f"source contract at {source_contract_path}"
        ) from exc
    if existing_source_contract != validation.contract.to_json_dict():
        raise AxBenchBpbAdvError(
            "clustered sample generation source contract is stale; use a fresh run_tag"
        )
    stage_source = resolve_clustered_stage_source(
        validation=validation,
        paths=paths,
        contract=contract,
        expected_source_run_dir=cfg.source_run_dir,
        expected_filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    return contract, stage_source


def _refresh_clustered_sample_aggregates(
    cfg: AxBenchBpbAdvConfig,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    stage_source: ClusteredStageSource,
    candidate_snapshot: ClusteredCandidateSnapshot,
) -> _ClusteredSamplingSnapshot:
    """Rebuild unstamped aggregate tables from exact clustered score inputs."""

    resolved_stage_source = require_clustered_stage_source(
        stage_source=stage_source,
        contract=contract,
        surface="sample aggregate refresh",
    )
    assert resolved_stage_source is not None
    resolved_candidate_snapshot = require_clustered_candidate_snapshot(
        candidate_snapshot,
        paths=paths,
        contract=contract,
        source_identity=resolved_stage_source.source_identity,
        surface="sample aggregate refresh",
    )
    assert resolved_candidate_snapshot is not None
    metric = SteeringMetric.coerce(cfg.steering_metric)
    score_path = (
        Path(cfg.scores_path)
        if cfg.scores_path
        else (
            paths.roundtrip_scores_path()
            if metric is SteeringMetric.ROUNDTRIP
            else paths.scores_path()
        )
    )
    score_family = (
        "roundtrip_scores"
        if metric is SteeringMetric.ROUNDTRIP
        else "class_margin_scores"
    )
    try:
        score_table = pq.read_table(score_path)
    except (OSError, pa.ArrowException) as exc:
        raise AxBenchBpbAdvError(
            f"clustered sample generation cannot read scored artifact {score_path}"
        ) from exc
    assert_table_cluster_identity(
        score_table,
        paths=paths,
        contract=contract,
        artifact_family=score_family,
        source_identity=resolved_stage_source.source_identity,
        candidate_snapshot=resolved_candidate_snapshot,
    )

    general_val_path = (
        paths.general_val_bpb_path() if cfg.general_val_bpb.enabled else None
    )
    general_val_table: pa.Table | None = None
    if general_val_path is not None:
        try:
            general_val_table = pq.read_table(general_val_path)
        except (OSError, pa.ArrowException) as exc:
            raise AxBenchBpbAdvError(
                "clustered sample generation cannot read general-validation "
                f"artifact {general_val_path}"
            ) from exc
        assert_table_cluster_identity(
            general_val_table,
            paths=paths,
            contract=contract,
            artifact_family="general_val_bpb",
            source_identity=resolved_stage_source.source_identity,
            candidate_snapshot=resolved_candidate_snapshot,
        )

    if metric is SteeringMetric.ROUNDTRIP:
        per_candidate = compute_roundtrip_per_candidate_aggregation(score_table)
        if general_val_table is not None:
            per_candidate = _join_roundtrip_general_val_bpb(
                per_candidate,
                general_val_table,
            )
        expected_macro = compute_roundtrip_macro_aggregation(per_candidate)
        macro = aggregate_roundtrip_adv_scores(
            score_path,
            paths.roundtrip_aggregates_dir(),
            general_val_path=general_val_path,
            scores_table=score_table,
            general_val_table=general_val_table,
        )
        metrics = _SamplingMetrics(
            macro=macro,
            per_candidate=per_candidate,
            representative_baseline_metric="primary_roundtrip_y",
            primary_metric_field="primary_roundtrip_y",
            primary_metric_output_key="primary_roundtrip_y",
        )
    else:
        per_candidate = compute_per_candidate_aggregation(score_table)
        if general_val_table is not None:
            per_candidate = _join_general_val_bpb(
                per_candidate,
                general_val_table,
            )
        expected_macro = compute_macro_aggregation(per_candidate)
        macro = aggregate_bpb_adv_scores(
            score_path,
            paths.aggregates_dir(),
            general_val_path=general_val_path,
            scores_table=score_table,
            general_val_table=general_val_table,
        )
        metrics = _SamplingMetrics(
            macro=macro,
            per_candidate=per_candidate,
            representative_baseline_metric="baseline_primary_margin",
            primary_metric_field="primary_margin_y",
            primary_metric_output_key="primary_margin_y",
        )
    if not macro.equals(expected_macro, check_metadata=True):
        raise AxBenchBpbAdvError(
            "clustered sample aggregate output differs from exact in-memory "
            "aggregation; refusing path-backed sample selection"
        )
    return _build_clustered_sampling_snapshot(
        metrics=metrics,
        steering_metric=metric,
        candidate_snapshot=resolved_candidate_snapshot,
        stage_source=resolved_stage_source,
    )


def run_generate_samples(
    cfg: AxBenchBpbAdvConfig,
    *,
    generation: GenerationConfig,
    counter_direction_family: str | None = None,
) -> list[Path]:
    """Load the target model(s) the eval scored, then run the core.

    The thin model-loading boundary: it builds the SAME decoder-only
    ``(model, tokenizer, device)`` bundle the scorer obtains (via
    ``RuntimeState(_bpb_config_from_adv_config(cfg)).decoder_only()``), detects its
    metadata, resolves the additive counter-direction family + inactive sampler
    exactly as ``run_model_backed_bpb_adv_score_stage`` does, then delegates to the
    pure-ish ``run_generate_samples_core``. Legacy runs remain decoder-only;
    clustered runs may additionally generate through the contract-validated
    encoder-decoder target.

    ``counter_direction_family`` overrides the scorer-resolved family when the
    generate-samples config carries it as a top-level field; ``None`` falls back to
    the additive method's value (``get_additive_direction_config(cfg)``), matching
    the scorer.

    A ``lora_decoder_only`` family is generated WITH its adapter: the LoRA bundle
    is built from ``cfg.lora_adapter`` exactly as
    ``run_model_backed_bpb_adv_score_stage`` builds it
    (``LoraParams(cfg=LoraConfig(**cfg.lora_adapter), layer=...)``), so generation
    reuses the scored intervention; an empty ``cfg.lora_adapter`` yields ``None``.

    For unclustered runs, the legacy decoder-only behavior and warning/skip policy
    for other targets is unchanged. A clustered run honors ``decoder_only`` and
    ``encdec`` entries and rejects any other target.
    """

    if cfg.run_tag is None:
        raise AxBenchBpbAdvError("cfg.run_tag must be set before generating samples")
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag=cfg.run_tag)
    resolved_counter_family = (
        counter_direction_family
        if counter_direction_family is not None
        else _resolve_counter_direction_family(cfg)
    )
    # Standalone sample generation validates the current source contract and
    # exact candidate-selected direction coordinates before loading a target
    # model. This applies to unclustered and clustered finished runs alike.
    source_contract_path = paths.source_contract()
    try:
        existing_source_contract = json.loads(
            source_contract_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            "sample generation requires the finished run's current v6/v7 "
            f"source contract at {source_contract_path}"
        ) from exc
    schema = (
        existing_source_contract.get("schema_version")
        if isinstance(existing_source_contract, dict)
        else None
    )
    if schema in {4, 5}:
        raise AxBenchBpbAdvError(
            "sample generation rejects legacy v4/v5 source contracts; use a fresh run tag"
        )
    if not isinstance(existing_source_contract, dict):
        raise AxBenchBpbAdvError(
            f"sample generation source contract must be an object: {source_contract_path}"
        )
    persisted_contract = BpbAdvSourceContract.from_json_dict(
        existing_source_contract,
        path=source_contract_path,
    )
    source_preflight = validate_stage_0_3_source_run(
        cfg.source_run_dir,
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    if persisted_contract.to_json_dict() != source_preflight.contract.to_json_dict():
        raise AxBenchBpbAdvError(
            "sample generation source contract is stale; use a fresh run_tag"
        )
    try:
        preflight_candidates = pq.read_table(paths.candidates_path())
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"sample generation cannot read candidates at {paths.candidates_path()}"
        ) from exc
    _validate_table(preflight_candidates, CANDIDATES_SCHEMA, name="candidates.parquet")
    source_preflight.require_directions(
        requirements_from_candidate_rows(
            preflight_candidates.to_pylist(),
            layer=cfg.diffmean_and_probe_layer,
            counter_direction_family=resolved_counter_family,
            method_whitelist=(
                None if generation.methods is None else set(generation.methods)
            ),
        ),
        surface="BPB-Adv sample generation",
    )
    cluster_contract, stage_source = _resolve_sample_cluster_stage(
        cfg,
        paths=paths,
    )
    candidate_snapshot = None
    sampling_snapshot = None
    if cluster_contract is not None:
        assert stage_source is not None
        candidate_snapshot = load_clustered_candidate_snapshot(
            paths.candidates_path(),
            paths=paths,
            contract=cluster_contract,
            source_identity=stage_source.source_identity,
        )
        # Aggregates deliberately retain their legacy schemas and therefore
        # have no artifact-local feature stamp. Derive their exact in-memory
        # equivalents from the stamped score inputs before they choose sample
        # coefficients; the path writes remain compatibility outputs only.
        sampling_snapshot = _refresh_clustered_sample_aggregates(
            cfg,
            paths=paths,
            contract=cluster_contract,
            stage_source=stage_source,
            candidate_snapshot=candidate_snapshot,
        )
    unsupported_targets = [
        target
        for target in cfg.target_models
        if str(target) != "decoder_only"
        and not (str(target) == "encdec" and cluster_contract is not None)
    ]
    if unsupported_targets:
        logger.warning(
            "generate_samples: target_models {} include non-decoder-only target(s) "
            "{}; encoder-decoder (and any non-decoder-only) sampling is a deliberate "
            "follow-up and will be SKIPPED -- generating decoder-only families only",
            list(cfg.target_models),
            unsupported_targets,
        )
    # Mirror the scorer's LoRA bundle so a lora_decoder_only family generates with
    # the SAME adapter the eval scored (empty cfg.lora_adapter -> no LoRA).
    lora = (
        LoraParams(
            cfg=LoraConfig(**cfg.lora_adapter),
            layer=cfg.diffmean_and_probe_layer,
        )
        if cfg.lora_adapter
        else None
    )
    bpb_cfg = _bpb_config_from_adv_config(cfg)
    runtime_state = RuntimeState(
        cfg=bpb_cfg,
        ddp_rank=0,
        ddp_world_size=1,
        cluster_contract=cluster_contract,
    )
    try:
        targets = (
            ["decoder_only"]
            if cluster_contract is None
            else [str(target) for target in cfg.target_models]
        )
        written: list[Path] = []
        for target_model in targets:
            if target_model == "decoder_only":
                bundle = runtime_state.decoder_only()
                inactive_sampler = None
                detect_kwargs: dict[str, Any] = {}
            elif target_model == "encdec":
                bundle = runtime_state.encdec()
                inactive_sampler = runtime_state.inactive_sampler()
                detect_kwargs = {"cluster_contract": cluster_contract}
            else:
                raise AxBenchBpbAdvError(
                    f"unsupported clustered sample target {target_model!r}"
                )
            target_model_meta = detect_target_model_metadata(
                bundle.model,
                bpb_cfg,
                **detect_kwargs,
            )
            written.extend(
                run_generate_samples_core(
                    paths=paths,
                    source_run_dir=cfg.source_run_dir,
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    device=t.device(runtime_state.target_device()),
                    target_model=target_model,
                    target_model_meta=target_model_meta,
                    inactive_sampler=inactive_sampler,
                    diffmean_layer=cfg.diffmean_and_probe_layer,
                    counter_direction_family=resolved_counter_family,
                    scatter_baseline_x=cfg.scatter_baseline_x,
                    steering_metric=cfg.steering_metric,
                    generation=generation,
                    apply_max_act_scale=True,
                    lora=lora if target_model == "decoder_only" else None,
                    cluster_contract=cluster_contract,
                    stage_source=stage_source,
                    candidate_snapshot=candidate_snapshot,
                    sampling_snapshot=sampling_snapshot,
                    pre_forward_guard=(
                        bundle.ensure_ready_for_forward
                        if cluster_contract is not None
                        else None
                    ),
                )
            )
        return written
    finally:
        if cfg.offload_between_stages:
            runtime_state.offload_models()


def _parse_top_k(payload: Mapping[str, Any]) -> int | None:
    """Resolve ``top_k`` from the generation block (absent ⟹ the field default).

    An ABSENT ``top_k`` falls back to the ``GenerationConfig`` field default
    (currently 50), NOT ``None`` -- an absent key must not silently switch
    sampling to unbounded. A present ``top_k`` that is ``None`` (explicit
    ``top_k: null``) OR ``<= 0`` is treated as DISABLED (``None``), so a stray
    ``top_k: 0`` never reaches ``torch.topk(0)``.
    """

    if "top_k" not in payload:
        # The field default (read off a default-constructed instance so the
        # value's static type stays ``int | None``), NOT None.
        return GenerationConfig().top_k
    raw = payload["top_k"]
    if raw is None:
        return None
    parsed = int(raw)
    return parsed if parsed > 0 else None


def _generation_config_from_mapping(value: Any) -> GenerationConfig:
    """Parse the ``generation:`` config block into a ``GenerationConfig``."""

    if value is None:
        return GenerationConfig()
    if not isinstance(value, Mapping):
        raise AxBenchBpbAdvError("generation must be a config object")
    payload = {str(k): v for k, v in value.items()}
    methods_raw = payload.get("methods")
    methods: list[str] | None
    if methods_raw is None:
        methods = None
    elif isinstance(methods_raw, str | bytes) or not isinstance(methods_raw, Sequence):
        # A scalar `methods: diffmean` (a bare string) would otherwise be iterated
        # CHARACTER by character into a bogus per-letter whitelist that matches no
        # family -> a silent zero-output run. Require an explicit list/tuple.
        raise AxBenchBpbAdvError(
            "generation.methods must be null or a list of method names, got "
            f"{type(methods_raw).__name__} {methods_raw!r}"
        )
    else:
        methods = [str(method) for method in methods_raw]
    try:
        return GenerationConfig(
            n_concepts=int(payload.get("n_concepts", 5)),
            n_prefixes_per_concept=int(payload.get("n_prefixes_per_concept", 1)),
            n_completions=int(payload.get("n_completions", 5)),
            temperature=float(payload.get("temperature", 0.7)),
            top_k=_parse_top_k(payload),
            max_new_tokens=int(payload.get("max_new_tokens", 128)),
            base_seed=int(payload.get("base_seed", 0)),
            methods=methods,
        )
    except (TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(f"invalid generation config: {exc}") from exc


def run_generate_samples_from_dictconfig(cfg: DictConfig) -> list[Path]:
    """Hydra entry: parse the DictConfig, load the model, run the core.

    Reuses the shared ``config_loader.cfg_to_dataclass`` (the single source of
    truth for base-config parsing) but additionally peels off the ``generation:``
    block into
    a ``GenerationConfig`` before constructing the base ``AxBenchBpbAdvConfig``.
    ``run_tag`` defaults to ``"samples"`` when unset so the generator writes under a
    stable, resumable run directory.
    """

    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    resolved = {str(k): v for k, v in container.items()}
    resolved.pop("hydra", None)
    generation = _generation_config_from_mapping(resolved.pop("generation", None))
    # ``counter_direction_family`` is a top-level generate-samples convenience knob
    # (the eval keeps it nested under the additive method). Peel it off before
    # cfg_to_dataclass (which would reject the unknown key) and thread it through.
    counter_direction_family_raw = resolved.pop("counter_direction_family", None)
    counter_direction_family = (
        None if counter_direction_family_raw is None else str(counter_direction_family_raw)
    )
    # The explicit top-level override is peeled off BEFORE cfg_to_dataclass, so it
    # bypasses validate_config's residual-family restriction. Re-apply that same
    # {"diffmean", "linear_probe"} guard here (the counter direction must be a
    # residual-direction family) rather than let a bogus value reach the loader.
    if counter_direction_family is not None and counter_direction_family not in {
        "diffmean",
        "linear_probe",
    }:
        raise AxBenchBpbAdvError(
            "counter_direction_family must be 'diffmean' or 'linear_probe', got "
            f"{counter_direction_family!r}"
        )
    base_cfg = cfg_to_dataclass(OmegaConf.create(resolved))
    if base_cfg.run_tag is None:
        base_cfg.run_tag = "samples"
    validate_config(base_cfg)
    return run_generate_samples(
        base_cfg,
        generation=generation,
        counter_direction_family=counter_direction_family,
    )


__all__ = [
    "FamilyStrengthAxis",
    "GenConcept",
    "GenFamily",
    "GenPrefix",
    "GenerationConfig",
    "RepresentativeAttack",
    "build_family_yaml_doc",
    "families_with_representative_coefficient",
    "generate_completions_for_datapoint",
    "representative_attack_for_datapoint",
    "run_generate_samples",
    "run_generate_samples_core",
    "run_generate_samples_from_dictconfig",
    "select_concepts_and_prefixes",
]
