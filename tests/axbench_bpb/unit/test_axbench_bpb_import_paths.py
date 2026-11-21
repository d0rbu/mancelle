"""Import-path preservation for steering.eval.axbench_bpb.* consumer-facing
symbols (PR #739 Track F, Binding Constraint 4).

Downstream Track G imports `steering.eval.axbench_bpb.*` directly; every
new symbol F1 adds must resolve at its expected path, and the pre-existing
`__all__` of every file F1 touches must stay pinned wherever the oracle (see
docs/plans/2026-07-24-pr739-track-f-plan-part1.md, `.claude/worktrees/
pr739-oracle` @ f357458b) adds nothing new to it.

Scope note: this file covers BOTH F1a (`config.py`, `source_paths.py`,
`intervention.py` -- the "leaves" PR) and F1b (`teacher_force.py`,
`runtime_state.py` -- the "forward-path" PR, merges after F1a). F1b EXTENDS
this same file with its own frozen-name-list sections rather than creating a
second import-path test file, per the plan's Test design table (this
filename appears in both F1a's and F1b's gate lists).

Mirrors tests/steering/test_cluster_feature_contract.py's frozen-name-list +
hasattr/getattr pattern.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from steering.eval.axbench_bpb import config as config_module
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb import runtime_state as runtime_state_module
from steering.eval.axbench_bpb import source_paths as source_paths_module
from steering.eval.axbench_bpb import teacher_force as teacher_force_module
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import TargetModel

# ============================================================================
# intervention.py.__all__ is pinned UNCHANGED. The oracle @ f357458b adds
# nothing new to it -- the MAIN-DRIFT CHECK confirmed main's __all__
# already matches the oracle's byte-for-byte (both:
# ["build_intervention", "apply_intervention", "normalize_direction"]).
# Frozen, hardcoded -- do not resync from a live import (the whole point is
# an independent expectation a silent drift can't quietly satisfy).
# ============================================================================

_ORACLE_INTERVENTION_ALL = (
    "apply_intervention",
    "build_intervention",
    "normalize_direction",
)


def test_intervention_all_matches_the_oracle_all() -> None:
    assert tuple(sorted(intervention_module.__all__)) == _ORACLE_INTERVENTION_ALL


@pytest.mark.parametrize("name", _ORACLE_INTERVENTION_ALL)
def test_every_intervention_all_name_imports_from_the_module(name: str) -> None:
    assert hasattr(intervention_module, name), (
        f"{name!r} is in intervention.py's __all__ but does not import from "
        "steering.eval.axbench_bpb.intervention"
    )


def test_all_intervention_all_names_are_actually_importable() -> None:
    """__all__ entries must be real attributes, not stale strings."""
    missing = [n for n in intervention_module.__all__ if not hasattr(intervention_module, n)]
    assert missing == []


# ============================================================================
# New consumer-facing symbols F1a adds.
# ============================================================================


def test_config_gains_feature_extraction_model_id_field() -> None:
    field_names = {f.name for f in fields(AxBenchBpbConfig)}
    assert "feature_extraction_model_id" in field_names


def test_config_does_not_duplicate_effective_extraction_model_id() -> None:
    """Binding Constraint 3's dedup decision: F1a reuses
    `steering.eval.axbench_base.config.effective_extraction_model_id`
    (predates this campaign, already defined there) rather than defining a
    second copy -- or a second `ExtractionModelConfig` Protocol -- in
    `axbench_bpb.config`."""
    assert not hasattr(config_module, "effective_extraction_model_id")
    assert not hasattr(config_module, "ExtractionModelConfig")


def test_axbench_base_config_still_owns_effective_extraction_model_id() -> None:
    """The reused helper must still resolve at its one true home."""
    from steering.eval.axbench_base.config import effective_extraction_model_id

    assert callable(effective_extraction_model_id)


_NEW_SOURCE_PATH_METHODS = (
    "feature_contract",
    "candidate_concepts",
    "candidate_cluster_provenance",
    "negative_mode_prefixes",
)


@pytest.mark.parametrize("name", _NEW_SOURCE_PATH_METHODS)
def test_source_paths_new_cluster_path_methods_resolve(name: str) -> None:
    assert hasattr(SourcePaths, name), (
        f"{name!r} does not resolve on steering.eval.axbench_bpb.source_paths.SourcePaths"
    )
    assert callable(getattr(SourcePaths, name))


def test_source_paths_module_still_offers_no_write_methods() -> None:
    """The four new methods must not break source_paths.py's read-only
    contract (mirrors test_source_paths.py's existing guard)."""
    assert not any("write" in n.lower() for n in dir(source_paths_module))


@pytest.mark.parametrize(
    "func_name",
    ["build_intervention", "_build_ours_intervention", "_build_direction_intervention"],
)
def test_intervention_functions_gain_cluster_contract_kwarg(func_name: str) -> None:
    func = getattr(intervention_module, func_name)
    sig = inspect.signature(func)
    assert "cluster_contract" in sig.parameters, f"{func_name} does not accept cluster_contract"
    param = sig.parameters["cluster_contract"]
    assert param.default is None, (
        f"{func_name}'s cluster_contract kwarg must default to None (additive-only)"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{func_name}'s cluster_contract kwarg must be keyword-only"
    )


# ============================================================================
# teacher_force.py.__all__ and runtime_state.py.__all__ (F1b portion).
# runtime_state.py.__all__ is pinned UNCHANGED (the oracle adds nothing new
# to it: both main and oracle are ``["RuntimeState"]``).
# teacher_force.py.__all__ gains exactly "assert_enc_ids_match_cluster_
# contract" (oracle-confirmed). Frozen, hardcoded -- do not resync from a
# live import.
# ============================================================================

_ORACLE_RUNTIME_STATE_ALL = ("RuntimeState",)

_ORACLE_TEACHER_FORCE_ALL = (
    "TargetModelBundle",
    "accumulate_bpb",
    "assert_enc_ids_in_vocab",
    "assert_enc_ids_match_cluster_contract",
    "build_output_only_mask",
    "score_sample",
    "score_triple",
)


def test_runtime_state_all_matches_the_oracle_all() -> None:
    assert tuple(sorted(runtime_state_module.__all__)) == _ORACLE_RUNTIME_STATE_ALL


def test_teacher_force_all_matches_the_oracle_all() -> None:
    assert tuple(sorted(teacher_force_module.__all__)) == _ORACLE_TEACHER_FORCE_ALL


@pytest.mark.parametrize("name", _ORACLE_RUNTIME_STATE_ALL)
def test_every_runtime_state_all_name_imports_from_the_module(name: str) -> None:
    assert hasattr(runtime_state_module, name)


@pytest.mark.parametrize("name", _ORACLE_TEACHER_FORCE_ALL)
def test_every_teacher_force_all_name_imports_from_the_module(name: str) -> None:
    assert hasattr(teacher_force_module, name)


def test_all_runtime_state_all_names_are_actually_importable() -> None:
    missing = [n for n in runtime_state_module.__all__ if not hasattr(runtime_state_module, n)]
    assert missing == []


def test_all_teacher_force_all_names_are_actually_importable() -> None:
    missing = [n for n in teacher_force_module.__all__ if not hasattr(teacher_force_module, n)]
    assert missing == []


# ============================================================================
# New consumer-facing symbols F1b adds.
# ============================================================================


def test_target_model_bundle_gains_cluster_fields() -> None:
    field_names = {f.name for f in fields(TargetModelBundle)}
    assert "cluster_contract" in field_names
    assert "cluster_pre_forward_guard" in field_names


def test_target_model_bundle_cluster_fields_default_to_none() -> None:
    """Additive-only: an unclustered construction (every existing call
    site) must not need to pass either new field."""
    bundle_fields = {f.name: f for f in fields(TargetModelBundle)}
    assert bundle_fields["cluster_contract"].default is None
    assert bundle_fields["cluster_pre_forward_guard"].default is None


def test_target_model_bundle_coerces_valid_string_target() -> None:
    bundle = TargetModelBundle(
        model=object(),
        tokenizer=object(),
        max_seq_len=1,
        target_name="encdec",
    )
    assert bundle.target_name is TargetModel.ENCDEC


def test_target_model_bundle_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown target model"):
        TargetModelBundle(
            model=object(),
            tokenizer=object(),
            max_seq_len=1,
            target_name="bogus",
        )


def test_target_model_bundle_gains_ensure_ready_for_forward() -> None:
    assert hasattr(TargetModelBundle, "ensure_ready_for_forward")
    assert callable(TargetModelBundle.ensure_ready_for_forward)


def test_assert_enc_ids_match_cluster_contract_resolves_at_teacher_force() -> None:
    assert hasattr(teacher_force_module, "assert_enc_ids_match_cluster_contract")
    assert callable(teacher_force_module.assert_enc_ids_match_cluster_contract)


def test_runtime_state_gains_cluster_contract_field_appended_last() -> None:
    """Binding constraint: cluster_contract is appended LAST so every
    existing RuntimeState(...) call site (which all use keyword args) stays
    additive-only regardless of field order."""
    field_names = [f.name for f in fields(RuntimeState) if not f.name.startswith("_")]
    assert field_names[-1] == "cluster_contract"
    assert field_names[:-1] == ["cfg", "ddp_rank", "ddp_world_size"]


def test_runtime_state_cluster_contract_field_defaults_to_none() -> None:
    rs_fields = {f.name: f for f in fields(RuntimeState)}
    assert rs_fields["cluster_contract"].default is None


@pytest.mark.parametrize(
    "name",
    ["release_base_gemma_and_sae", "release_target_models"],
)
def test_runtime_state_gains_release_methods(name: str) -> None:
    assert hasattr(RuntimeState, name), (
        f"{name!r} does not resolve on steering.eval.axbench_bpb.runtime_state.RuntimeState"
    )
    assert callable(getattr(RuntimeState, name))


def test_runtime_state_gains_build_cluster_pre_forward_guard_helper() -> None:
    """Module-private (not in __all__) but a real, directly-imported symbol
    -- tests/axbench_bpb/unit/test_runtime_state_target_device.py imports it
    directly, exactly like tests/steering/test_cluster_feature_contract.py's
    pattern for other module-private helpers."""
    assert hasattr(runtime_state_module, "_build_cluster_pre_forward_guard")
    assert callable(runtime_state_module._build_cluster_pre_forward_guard)


def test_base_gemma_and_sae_gains_offload_transformer_tail_after_kwarg() -> None:
    sig = inspect.signature(RuntimeState.base_gemma_and_sae)
    assert "offload_transformer_tail_after" in sig.parameters
    param = sig.parameters["offload_transformer_tail_after"]
    assert param.default is None, (
        "offload_transformer_tail_after must default to None (additive-only)"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
