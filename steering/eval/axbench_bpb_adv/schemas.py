"""PyArrow schemas for AxBench-BPB-Adv artifacts.

Each attack/candidate row describes one *core prefix* (a concept/mode/
prefix_text triple, keyed by ``core_prefix_id``) under one steering+adversarial
configuration. A core prefix owns BOTH AxBench completions: ``concept_rich``
(the concept-favoring continuation) and ``concept_avoiding`` (the contrast
continuation). The legacy BPB score schema teacher-forces both completions under
the same steered intervention and reports signed class margin downstream. The
roundtrip score schema instead stores generated primary/control completions plus
SAE reactivation probe outputs, while retaining BPB baseline losses for scatter
plots.
"""

from __future__ import annotations

import pyarrow as pa

ATTACK_ROWS_SCHEMA: pa.Schema = pa.schema(
    [
        ("attack_row_id", pa.string()),
        ("core_prefix_id", pa.string()),
        ("concept_rich_sample_id", pa.string()),
        ("concept_avoiding_sample_id", pa.string()),
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("prefix_text", pa.string()),
        ("prefix_hash", pa.string()),
        ("source_prompt_tokens", pa.list_(pa.int64())),
        ("source_prompt_token_digest", pa.string()),
        ("concept_rich_output_tokens", pa.list_(pa.int64())),
        ("concept_avoiding_output_tokens", pa.list_(pa.int64())),
        ("concept_rich_output_digest", pa.string()),
        ("concept_avoiding_output_digest", pa.string()),
        ("editable_token_start", pa.int32()),
        ("editable_token_count", pa.int32()),
        ("original_editable_tokens", pa.list_(pa.int64())),
    ]
)


CANDIDATES_SCHEMA: pa.Schema = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("attack_row_id", pa.string()),
        ("core_prefix_id", pa.string()),
        ("concept_rich_sample_id", pa.string()),
        ("concept_avoiding_sample_id", pa.string()),
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("prefix_text", pa.string()),
        ("prefix_hash", pa.string()),
        ("source_prompt_tokens", pa.list_(pa.int64())),
        ("source_prompt_token_digest", pa.string()),
        ("attacked_prompt_text", pa.string()),
        ("attacked_prompt_tokens", pa.list_(pa.int64())),
        ("attacked_prompt_token_digest", pa.string()),
        ("concept_rich_output_tokens", pa.list_(pa.int64())),
        ("concept_avoiding_output_tokens", pa.list_(pa.int64())),
        ("concept_rich_output_digest", pa.string()),
        ("concept_avoiding_output_digest", pa.string()),
        ("editable_token_start", pa.int32()),
        ("editable_token_count", pa.int32()),
        ("replacement_tokens", pa.list_(pa.int64())),
        ("replacement_text", pa.string()),
        ("objective_value", pa.float64()),
        ("candidate_metadata_json", pa.string()),
    ]
)


SCORE_SCHEMA: pa.Schema = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("core_prefix_id", pa.string()),
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("source_prompt_tokens", pa.list_(pa.int64())),
        ("source_prompt_token_digest", pa.string()),
        ("attacked_prompt_tokens", pa.list_(pa.int64())),
        ("attacked_prompt_token_digest", pa.string()),
        ("editable_token_start", pa.int32()),
        ("editable_token_count", pa.int32()),
        ("replacement_tokens", pa.list_(pa.int64())),
        ("loss_concept_rich", pa.float64()),
        ("loss_concept_avoiding", pa.float64()),
        ("is_control", pa.bool_()),
        ("n_rich_output_tokens", pa.int32()),
        ("n_avoid_output_tokens", pa.int32()),
        ("n_rich_output_bytes", pa.int32()),
        ("n_avoid_output_bytes", pa.int32()),
        ("rich_truncated", pa.bool_()),
        ("avoid_truncated", pa.bool_()),
    ]
)


PER_CANDIDATE_AGG_SCHEMA: pa.Schema = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("mean_loss_concept_rich", pa.float64()),
        ("mean_loss_concept_avoiding", pa.float64()),
        ("primary_margin_y", pa.float64()),
        ("baseline_primary_margin", pa.float64()),
        # General-validation BPB under the SAME steering+adversarial intervention
        # (see steering/eval/axbench_bpb_adv/general_val.py). Nullable: stays null
        # when general_val_bpb is disabled or no general-val row matches this
        # (target, method, concept, coefficient, adversarial) intervention.
        ("general_val_bpb", pa.float64()),
        # Corpus identity of the general-val docs this value was scored on
        # ("nd=<n>;nb=<bytes>;h=<32hex>"). Null iff general_val_bpb is null.
        # Issue #635: figures raise on mixed signatures.
        ("general_val_corpus", pa.string()),
        ("n_samples", pa.int32()),
        ("truncation_rate", pa.float64()),
        ("n_rich_output_tokens", pa.int64()),
        ("n_avoid_output_tokens", pa.int64()),
        ("n_rich_output_bytes", pa.int64()),
        ("n_avoid_output_bytes", pa.int64()),
        ("is_control", pa.bool_()),
    ]
)


MACRO_AGG_SCHEMA: pa.Schema = pa.schema(
    [
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("mean_loss_concept_rich", pa.float64()),
        ("mean_loss_concept_avoiding", pa.float64()),
        ("primary_margin_y", pa.float64()),
        ("baseline_primary_margin", pa.float64()),
        # Concept-averaged general-validation BPB under this intervention (mean
        # over the concepts contributing to the macro group). Nullable; null when
        # general_val_bpb is disabled or unmatched. See general_val.py.
        ("general_val_bpb", pa.float64()),
        # Corpus identity of the general-val docs this value was scored on
        # ("nd=<n>;nb=<bytes>;h=<32hex>"). Null iff general_val_bpb is null.
        # Issue #635: figures raise on mixed signatures.
        ("general_val_corpus", pa.string()),
        ("n_concepts", pa.int32()),
        ("n_samples", pa.int32()),
        ("truncation_rate", pa.float64()),
        ("is_control", pa.bool_()),
    ]
)


ROUNDTRIP_SCORE_SCHEMA: pa.Schema = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("core_prefix_id", pa.string()),
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("source_prompt_tokens", pa.list_(pa.int64())),
        ("source_prompt_token_digest", pa.string()),
        ("attacked_prompt_tokens", pa.list_(pa.int64())),
        ("attacked_prompt_token_digest", pa.string()),
        ("roundtrip_prompt_tokens", pa.list_(pa.int64())),
        ("roundtrip_prompt_token_digest", pa.string()),
        ("roundtrip_prompt_text", pa.string()),
        ("control_roundtrip_prompt_tokens", pa.list_(pa.int64())),
        ("control_roundtrip_prompt_token_digest", pa.string()),
        ("control_roundtrip_prompt_text", pa.string()),
        ("editable_token_start", pa.int32()),
        ("editable_token_count", pa.int32()),
        ("replacement_tokens", pa.list_(pa.int64())),
        ("loss_concept_rich", pa.float64()),
        ("loss_concept_avoiding", pa.float64()),
        ("sample_index", pa.int32()),
        ("reference_segment_token_count", pa.int32()),
        ("generation_seed", pa.int64()),
        ("generated_text", pa.string()),
        ("n_generated_tokens", pa.int32()),
        ("stop_reason", pa.string()),
        ("control_generation_seed", pa.int64()),
        ("control_generated_text", pa.string()),
        ("control_n_generated_tokens", pa.int32()),
        ("control_stop_reason", pa.string()),
        ("is_control", pa.bool_()),
        ("target_feature_id", pa.int32()),
        ("n_roundtrip_features", pa.int32()),
        ("roundtrip_pre_act", pa.float64()),
        ("roundtrip_post_act", pa.float64()),
        ("roundtrip_percentile", pa.float64()),
        ("control_n_roundtrip_features", pa.int32()),
        ("control_roundtrip_pre_act", pa.float64()),
        ("control_roundtrip_post_act", pa.float64()),
        ("control_roundtrip_percentile", pa.float64()),
    ]
)


# Per-intervention general-validation BPB, written by the ``score_general_val``
# stage (steering/eval/axbench_bpb_adv/general_val.py). One row per distinct
# ``(target_model, method_name, concept_id, coefficient_str, adversarial_method,
# adversarial_strength)`` combination present in the candidates.
# The expensive BPB scoring is deduplicated per intervention SIGNATURE; under the
# UNIVERSAL Faster-GCG trigger attack the docs are substituted with the
# per-concept trigger, which differs per ``strength_index/k``, so
# ``general_val_bpb`` is now NON-FLAT across the GCG axis (only the
# additive-magnitude-0 control and gepa stay on the shared ``base`` intervention).
# The scored value is broadcast to every matching combination so aggregation can
# left-join on the SIX identity key columns. ``mode``
# is redundant with ``method_name`` but kept for filtering.
# ``general_val_corpus`` (D2) is the per-target scored-doc-set identity computed
# by ``general_val._corpus_signature``: ``nd=<n>;nb=<bytes>;h=<32hex>``. It
# describes WHICH docs were scored (never the intervention), so downstream
# strength-frontier comparisons can assert two runs actually scored the same
# corpus before comparing their ``general_val_bpb`` values.
GENERAL_VAL_BPB_SCHEMA: pa.Schema = pa.schema(
    [
        ("target_model", pa.string()),
        ("method_name", pa.string()),
        ("mode", pa.string()),
        ("concept_id", pa.int32()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("general_val_bpb", pa.float64()),
        ("n_documents", pa.int32()),
        ("n_val_bytes", pa.int64()),
        ("intervention_signature", pa.string()),
        ("general_val_corpus", pa.string()),
    ]
)


# One row per non-control UNIVERSAL Faster-GCG concept group, written by the
# candidate stage (steering/eval/axbench_bpb_adv/model_optimize.py) to
# ``paths.universal_triggers_path()``. A concept group is the set of attack rows
# sharing the 7-tuple ``(concept_id, mode, method_name, target_model,
# coefficient_str, adversarial_method, strength_index)``; every
# member prefix is optimised against ONE shared k-token trigger
# (``adversarial_strength == k``), substituted into each prefix's own last-k span.
# ``general_val.py`` reads this table directly (keyed by the combo) to substitute
# the SAME trigger into the validation docs, so ``general_val_bpb`` measures
# general LM ability under the attack. ``objective_value`` is the final
# minibatch-aggregated search objective ``g``; ``n_pairs`` is the group's
# minibatch population. The column set is identical to the per-group dict emitted
# by ``optimize._trigger_row`` (one row per group).
UNIVERSAL_TRIGGERS_SCHEMA: pa.Schema = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("strength_index", pa.int32()),
        ("adversarial_strength", pa.float64()),
        ("trigger_tokens", pa.list_(pa.int64())),
        ("trigger_text", pa.string()),
        ("objective_value", pa.float64()),
        ("max_steps", pa.int32()),
        ("accepted_steps", pa.int32()),
        ("visited_sequences", pa.int32()),
        ("n_pairs", pa.int32()),
    ]
)


ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA: pa.Schema = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("mean_loss_concept_rich", pa.float64()),
        ("mean_loss_concept_avoiding", pa.float64()),
        ("mean_roundtrip_pre_act", pa.float64()),
        ("mean_roundtrip_post_act", pa.float64()),
        ("mean_roundtrip_percentile", pa.float64()),
        ("primary_roundtrip_y", pa.float64()),
        ("mean_control_roundtrip_pre_act", pa.float64()),
        ("mean_control_roundtrip_post_act", pa.float64()),
        ("mean_control_roundtrip_percentile", pa.float64()),
        ("control_primary_roundtrip_y", pa.float64()),
        # General-validation BPB under the SAME steering+adversarial intervention.
        # Nullable: stays null when general_val_bpb is disabled or no row matches.
        ("general_val_bpb", pa.float64()),
        # Corpus identity of the general-val docs this value was scored on
        # ("nd=<n>;nb=<bytes>;h=<32hex>"). Null iff general_val_bpb is null.
        # Issue #635: figures raise on mixed signatures.
        ("general_val_corpus", pa.string()),
        ("n_samples", pa.int32()),
        ("n_generated_tokens", pa.int64()),
        ("n_control_generated_tokens", pa.int64()),
        ("max_tokens_rate", pa.float64()),
        ("control_max_tokens_rate", pa.float64()),
        ("is_control", pa.bool_()),
    ]
)


ROUNDTRIP_MACRO_AGG_SCHEMA: pa.Schema = pa.schema(
    [
        ("mode", pa.string()),
        ("method_name", pa.string()),
        ("target_model", pa.string()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_unit", pa.string()),
        ("strength_index", pa.int32()),
        ("context_signal_mode", pa.string()),
        ("mean_loss_concept_rich", pa.float64()),
        ("mean_loss_concept_avoiding", pa.float64()),
        ("mean_roundtrip_pre_act", pa.float64()),
        ("mean_roundtrip_post_act", pa.float64()),
        ("mean_roundtrip_percentile", pa.float64()),
        ("primary_roundtrip_y", pa.float64()),
        ("mean_control_roundtrip_pre_act", pa.float64()),
        ("mean_control_roundtrip_post_act", pa.float64()),
        ("mean_control_roundtrip_percentile", pa.float64()),
        ("control_primary_roundtrip_y", pa.float64()),
        # Concept-averaged general-validation BPB under this intervention.
        # Nullable: stays null when general_val_bpb is disabled or unmatched.
        ("general_val_bpb", pa.float64()),
        # Corpus identity of the general-val docs this value was scored on
        # ("nd=<n>;nb=<bytes>;h=<32hex>"). Null iff general_val_bpb is null.
        # Issue #635: figures raise on mixed signatures.
        ("general_val_corpus", pa.string()),
        ("n_concepts", pa.int32()),
        ("n_samples", pa.int32()),
        ("n_generated_tokens", pa.int64()),
        ("n_control_generated_tokens", pa.int64()),
        ("max_tokens_rate", pa.float64()),
        ("control_max_tokens_rate", pa.float64()),
        ("is_control", pa.bool_()),
    ]
)
