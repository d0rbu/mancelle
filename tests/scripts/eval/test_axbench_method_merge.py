"""Focused semantic tests for the AxBench ten-method merge."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from scripts.eval.axbench_upstream import merge_methods as merge_module
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    KEY_COLUMNS,
    MERGED_INPUT_FILENAME,
    METHOD_FACTORS,
    FrameSource,
    MergedFrameSurface,
    _require_generation_health,
    merge_steering_frames,
    merge_steering_parquets,
    read_canonical_merged_frame,
    validate_steering_frame,
)

_METHODS = {
    "main": ("DiffMean", "LinearProbe"),
    "aux": ("PromptSteering", "LoRA"),
    "encdec_main": ("EncDecDiffMean", "EncDecLinearProbe"),
    "encdec_aux": ("EncDecPromptSteering", "EncDecLoRA"),
    "project": ("SelfConditioning", "SelfConditioningMasked"),
    "project_legacy": ("SelfConditioning",),
}
_PERPLEXITY_METHODS = frozenset({"PromptSteering", "LoRA"})


def test_declared_method_factor_contract_is_explicit() -> None:
    assert FrameSource.ENCDEC_MAIN.value == "encdec_main"
    assert FrameSource.ENCDEC_AUX.value == "encdec_aux"
    assert METHOD_FACTORS == {
        "DiffMean": FULL_FACTORS,
        "LinearProbe": FULL_FACTORS,
        "PromptSteering": (1.0,),
        "LoRA": (1.0,),
        "EncDecDiffMean": FULL_FACTORS,
        "EncDecLinearProbe": FULL_FACTORS,
        "EncDecPromptSteering": (1.0,),
        "EncDecLoRA": (1.0,),
        "SelfConditioning": FULL_FACTORS,
        "SelfConditioningMasked": FULL_FACTORS,
    }
    assert merge_module.SUPPORTED_METHOD_FACTORS == {
        **METHOD_FACTORS,
        "NLSteering": FULL_FACTORS,
    }
    assert merge_module.NL_PROJECT_METHODS == (
        "SelfConditioning",
        "SelfConditioningMasked",
        "NLSteering",
    )


def test_method_registry_derivations_are_consistent() -> None:
    """Pin every derived collection against LITERAL values, not the registry.

    Deriving the expectations from ``_METHOD_SPECS`` itself would make this
    test tautological: a method registered with the wrong source or metric
    would shift the derived tables and the derived expectation in lockstep.
    The literals below are the ratified contract.
    """

    assert merge_module._SOURCE_METHODS == {
        FrameSource.MAIN: ("DiffMean", "LinearProbe"),
        FrameSource.AUX: ("PromptSteering", "LoRA"),
        FrameSource.ENCDEC_MAIN: ("EncDecDiffMean", "EncDecLinearProbe"),
        FrameSource.ENCDEC_AUX: ("EncDecPromptSteering", "EncDecLoRA"),
        FrameSource.PROJECT_INPUT: (),
        FrameSource.PROJECT: ("SelfConditioning", "SelfConditioningMasked"),
        FrameSource.MERGED: (
            "DiffMean",
            "LinearProbe",
            "PromptSteering",
            "LoRA",
            "EncDecDiffMean",
            "EncDecLinearProbe",
            "EncDecPromptSteering",
            "EncDecLoRA",
            "SelfConditioning",
            "SelfConditioningMasked",
        ),
    }
    assert merge_module._STRENGTH_METHODS == frozenset(
        {
            "DiffMean",
            "LinearProbe",
            "EncDecDiffMean",
            "EncDecLinearProbe",
            "EncDecPromptSteering",
            "EncDecLoRA",
            "SelfConditioning",
            "SelfConditioningMasked",
        }
    )
    assert merge_module._PERPLEXITY_METHODS == frozenset({"PromptSteering", "LoRA"})
    assert tuple(METHOD_FACTORS) == merge_module._SOURCE_METHODS[FrameSource.MERGED]
    assert merge_module.methods_for_source(FrameSource.MERGED) == tuple(METHOD_FACTORS)
    # Structural invariants that must hold for ANY future registry content.
    specs = merge_module._METHOD_SPECS
    assert len({spec.name for spec in specs}) == len(specs)
    per_source = [
        merge_module.methods_for_source(source)
        for source in FrameSource
        if source is not FrameSource.MERGED
    ]
    assert sum(map(len, per_source)) == len(set().union(*map(set, per_source)))


def test_metric_column_rejects_an_unknown_method_loudly() -> None:
    with pytest.raises(ValueError, match="unknown steering method: 'Unknown'"):
        merge_module._metric_column("Unknown")


def test_explicit_nl_project_selection_uses_strength_without_expanding_defaults() -> None:
    project = _frame("project")
    project["NLSteering_steered_generation"] = project["factor"].map(
        lambda factor: f"NL retrieval generated a healthy answer at {factor}"
    )
    project["NLSteering_strength"] = project["factor"].astype("float64")
    project = project.loc[
        :, [*INPUT_COLUMNS, *merge_module.method_columns(merge_module.NL_PROJECT_METHODS)]
    ]

    validate_steering_frame(
        project,
        source="project",
        max_concepts=2,
        methods=merge_module.NL_PROJECT_METHODS,
    )
    assert merge_module.methods_for_source(FrameSource.PROJECT) == _METHODS["project"]
    assert "NLSteering" not in METHOD_FACTORS


@pytest.mark.parametrize(
    "methods",
    [
        ("SelfConditioning", "NLSteering", "NLSteering"),
        ("SelfConditioning", "Unknown"),
        ("DiffMean", "NLSteering"),
    ],
)
def test_explicit_project_method_selection_rejects_malformed_sets(
    methods: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="duplicates|unknown|not 'project'"):
        validate_steering_frame(
            _frame("project"),
            source="project",
            max_concepts=2,
            methods=methods,
        )


def test_methods_override_validates_the_pinned_single_method_project_frame() -> None:
    legacy = _frame("project_legacy")
    with pytest.raises(ValueError, match="columns do not match"):
        validate_steering_frame(legacy, source="project", max_concepts=2)

    validate_steering_frame(
        legacy,
        source="project",
        max_concepts=2,
        methods=("SelfConditioning",),
    )


def test_methods_override_rejects_a_frame_with_extra_method_columns() -> None:
    project = _frame("project")
    with pytest.raises(ValueError, match="unknown method result columns"):
        validate_steering_frame(
            project,
            source="project",
            max_concepts=2,
            methods=("SelfConditioning",),
        )

    validate_steering_frame(project, source="project", max_concepts=2)


def _metric_column(method: str) -> str:
    suffix = "perplexity" if method in _PERPLEXITY_METHODS else "strength"
    return f"{method}_{suffix}"


def _frame(source: str, *, max_concepts: int = 2) -> pd.DataFrame:
    factors = (1.0,) if source in {"aux", "encdec_aux"} else FULL_FACTORS
    rows = []
    for concept_id in range(max_concepts):
        for input_id in range(10):
            for factor in factors:
                row: dict[str, object] = {
                    "dataset_name": "AlpacaEval",
                    "concept_id": concept_id,
                    "input_concept": f"concept-{concept_id}",
                    "input_id": input_id,
                    "factor": factor,
                    "original_prompt": f"original prompt {concept_id}/{input_id}",
                    "steered_input": f"steered input {concept_id}/{input_id}",
                    "input": f"plain input {concept_id}/{input_id}",
                    "suppress_original": f"original suppression {concept_id}/{input_id}",
                    "suppress_rewrite": f"rewritten suppression {concept_id}/{input_id}",
                    "steered_prompt": f"steered prompt {concept_id}/{input_id}",
                    "defense": "none",
                    "sae_link": f"https://example.test/sae/{concept_id}",
                    "sae_id": 100 + concept_id,
                }
                for method in _METHODS[source]:
                    row[f"{method}_steered_generation"] = (
                        f"{method} generated a healthy response for "
                        f"{concept_id}/{input_id}/{factor}"
                    )
                    row[_metric_column(method)] = (
                        2.0 + concept_id + input_id / 100.0
                        if method in _PERPLEXITY_METHODS
                        else factor * (concept_id + 1.0)
                    )
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["concept_id"] = frame["concept_id"].astype("int64")
    frame["input_id"] = frame["input_id"].astype("int64")
    frame["sae_id"] = frame["sae_id"].astype("int64")
    frame["factor"] = frame["factor"].astype("float64")
    for method in _METHODS[source]:
        frame[_metric_column(method)] = frame[_metric_column(method)].astype("float64")
    return frame.loc[:, [*INPUT_COLUMNS, *frame.columns.difference(INPUT_COLUMNS, sort=False)]]


def _valid_frames() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    main = _frame("main")
    aux = _frame("aux")
    aux["steered_input"] = aux["steered_input"].map(lambda value: f"prompt-selected: {value}")
    encdec_aux = _frame("encdec_aux")
    encdec_aux["steered_input"] = encdec_aux["steered_input"].map(
        lambda value: f"encdec-prompt-selected: {value}"
    )
    return main, aux, _frame("encdec_main"), encdec_aux, _frame("project")


def test_validates_and_merges_two_concepts_in_canonical_order() -> None:
    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    main = main.sample(frac=1.0, random_state=1).reset_index(drop=True)
    aux = aux.sample(frac=1.0, random_state=2).reset_index(drop=True)
    encdec_main = encdec_main.sample(frac=1.0, random_state=3).reset_index(drop=True)
    encdec_aux = encdec_aux.sample(frac=1.0, random_state=4).reset_index(drop=True)
    project = project.sample(frac=1.0, random_state=5).reset_index(drop=True)

    merged = merge_steering_frames(
        main,
        aux,
        encdec_main,
        encdec_aux,
        project,
        max_concepts=2,
    )

    validate_steering_frame(merged, source="merged", max_concepts=2)
    assert len(merged) == 2 * 10 * len(FULL_FACTORS)
    assert list(merged.columns) == [
        *INPUT_COLUMNS,
        "DiffMean_steered_generation",
        "DiffMean_strength",
        "LinearProbe_steered_generation",
        "LinearProbe_strength",
        "PromptSteering_steered_generation",
        "PromptSteering_perplexity",
        "LoRA_steered_generation",
        "LoRA_perplexity",
        "EncDecDiffMean_steered_generation",
        "EncDecDiffMean_strength",
        "EncDecLinearProbe_steered_generation",
        "EncDecLinearProbe_strength",
        "EncDecPromptSteering_steered_generation",
        "EncDecPromptSteering_strength",
        "EncDecLoRA_steered_generation",
        "EncDecLoRA_strength",
        "SelfConditioning_steered_generation",
        "SelfConditioning_strength",
        "SelfConditioningMasked_steered_generation",
        "SelfConditioningMasked_strength",
    ]
    expected_keys = pd.MultiIndex.from_product(
        (range(2), range(10), FULL_FACTORS), names=KEY_COLUMNS
    )
    assert pd.MultiIndex.from_frame(merged.loc[:, list(KEY_COLUMNS)]).equals(expected_keys)
    aux_like_columns = [
        "PromptSteering_steered_generation",
        "PromptSteering_perplexity",
        "LoRA_steered_generation",
        "LoRA_perplexity",
        "EncDecPromptSteering_steered_generation",
        "EncDecPromptSteering_strength",
        "EncDecLoRA_steered_generation",
        "EncDecLoRA_strength",
    ]
    assert merged.loc[merged["factor"].eq(1.0), aux_like_columns].notna().all().all()
    assert merged.loc[merged["factor"].ne(1.0), aux_like_columns].isna().all().all()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda frame: frame.assign(Unknown_strength=1.0),
            "unknown method result columns",
        ),
        (
            lambda frame: frame.drop(index=0),
            "exact key product",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate.*keys",
        ),
        (
            lambda frame: frame.assign(input_id=frame["input_id"].astype("int32")),
            "dtype int64",
        ),
        (
            lambda frame: frame.assign(
                DiffMean_strength=frame["DiffMean_strength"].astype("float32")
            ),
            "dtype float64",
        ),
        (
            lambda frame: frame.assign(original_prompt=None),
            "required input column.*nulls",
        ),
    ],
)
def test_rejects_schema_geometry_and_dtype_defects(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    error: str,
) -> None:
    frame = mutate(_frame("main"))
    with pytest.raises(ValueError, match=error):
        validate_steering_frame(frame, source="main", max_concepts=2)


@pytest.mark.parametrize("source", [FrameSource.ENCDEC_MAIN, FrameSource.ENCDEC_AUX])
@pytest.mark.parametrize(
    ("defect", "error"),
    [
        ("columns", "columns do not match"),
        ("dtype", "dtype float64"),
        ("key_product", "exact key product"),
        ("input_invariant", "required input column.*nulls"),
        ("method_value", "must be finite"),
        ("generation_health", "blank generations"),
    ],
)
def test_encdec_sources_reject_every_source_frame_contract_defect(
    source: FrameSource,
    defect: str,
    error: str,
) -> None:
    frame = _frame(source)
    method = _METHODS[source][0]
    metric_column = _metric_column(method)

    if defect == "columns":
        frame = frame.drop(columns=f"{method}_steered_generation")
    elif defect == "dtype":
        frame[metric_column] = frame[metric_column].astype("float32")
    elif defect == "key_product":
        frame = frame.drop(index=0)
    elif defect == "input_invariant":
        frame.loc[0, "original_prompt"] = None
    elif defect == "method_value":
        frame.loc[0, metric_column] = float("inf")
    elif defect == "generation_health":
        frame.loc[frame.index[:3], f"{method}_steered_generation"] = " "
    else:
        raise AssertionError(f"unknown test defect: {defect}")

    with pytest.raises(ValueError, match=error):
        validate_steering_frame(frame, source=source, max_concepts=2)


@pytest.mark.parametrize("source", [FrameSource.ENCDEC_MAIN, FrameSource.ENCDEC_AUX])
def test_encdec_sources_reject_input_and_concept_metadata_drift(source: FrameSource) -> None:
    frame = _frame(source)
    frame.loc[
        (frame["concept_id"].eq(0)) & (frame["input_id"].eq(1)),
        "sae_link",
    ] = "https://example.test/different"
    with pytest.raises(ValueError, match="sae_link.*varies across inputs"):
        validate_steering_frame(frame, source=source, max_concepts=2)

    if source is FrameSource.ENCDEC_MAIN:
        frame = _frame(source)
        frame.loc[
            (frame["concept_id"].eq(0)) & (frame["input_id"].eq(0)) & (frame["factor"].eq(0.2)),
            "original_prompt",
        ] = "different prompt"
        with pytest.raises(ValueError, match="original_prompt.*varies across factors"):
            validate_steering_frame(frame, source=source, max_concepts=2)


def test_rejects_factor_and_concept_metadata_drift() -> None:
    factor_drift = _frame("main")
    factor_drift.loc[
        (factor_drift["concept_id"].eq(0))
        & (factor_drift["input_id"].eq(0))
        & (factor_drift["factor"].eq(0.2)),
        "original_prompt",
    ] = "different prompt"
    with pytest.raises(ValueError, match="original_prompt.*varies across factors"):
        validate_steering_frame(factor_drift, source="main", max_concepts=2)

    metadata_drift = _frame("project")
    metadata_drift.loc[
        (metadata_drift["concept_id"].eq(0)) & (metadata_drift["input_id"].eq(1)),
        "sae_link",
    ] = "https://example.test/different"
    with pytest.raises(ValueError, match="sae_link.*varies across inputs"):
        validate_steering_frame(metadata_drift, source="project", max_concepts=2)

    wrong_dataset = _frame("main")
    wrong_dataset.loc[0, "dataset_name"] = "not-alpaca"
    with pytest.raises(ValueError, match="dataset_name must be exactly 'AlpacaEval'"):
        validate_steering_frame(wrong_dataset, source="main", max_concepts=2)


def test_rejects_cross_frame_input_mismatches_but_allows_aux_like_steered_input() -> None:
    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    merged = merge_steering_frames(
        main,
        aux,
        encdec_main,
        encdec_aux,
        project,
        max_concepts=2,
    )
    assert len(merged) == 240

    project.loc[
        project["concept_id"].eq(0) & project["input_id"].eq(0),
        "input",
    ] = "mismatch"
    with pytest.raises(ValueError, match="project inputs do not match"):
        merge_steering_frames(
            main,
            aux,
            encdec_main,
            encdec_aux,
            project,
            max_concepts=2,
        )

    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    encdec_main.loc[
        encdec_main["concept_id"].eq(0) & encdec_main["input_id"].eq(0),
        "steered_input",
    ] = "mismatch"
    with pytest.raises(ValueError, match="encdec_main inputs do not match"):
        merge_steering_frames(
            main,
            aux,
            encdec_main,
            encdec_aux,
            project,
            max_concepts=2,
        )

    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    aux.loc[0, "original_prompt"] = "mismatch"
    with pytest.raises(ValueError, match="aux inputs do not match"):
        merge_steering_frames(
            main,
            aux,
            encdec_main,
            encdec_aux,
            project,
            max_concepts=2,
        )

    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    aux.loc[0, "steered_input"] = "   "
    with pytest.raises(ValueError, match="aux steered_input values must be nonblank"):
        merge_steering_frames(
            main,
            aux,
            encdec_main,
            encdec_aux,
            project,
            max_concepts=2,
        )

    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    encdec_aux.loc[0, "original_prompt"] = "mismatch"
    with pytest.raises(ValueError, match="encdec_aux inputs do not match"):
        merge_steering_frames(
            main,
            aux,
            encdec_main,
            encdec_aux,
            project,
            max_concepts=2,
        )

    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    encdec_aux.loc[0, "steered_input"] = "   "
    with pytest.raises(ValueError, match="encdec_aux steered_input values must be nonblank"):
        merge_steering_frames(
            main,
            aux,
            encdec_main,
            encdec_aux,
            project,
            max_concepts=2,
        )


@pytest.mark.parametrize(
    ("source", "column", "value", "error"),
    [
        ("main", "DiffMean_strength", float("inf"), "must be finite"),
        ("aux", "PromptSteering_perplexity", 0.0, "must be positive"),
        ("encdec_main", "EncDecDiffMean_strength", float("inf"), "must be finite"),
        (
            "encdec_aux",
            "EncDecPromptSteering_steered_generation",
            None,
            "null at a declared factor",
        ),
        ("project", "SelfConditioning_steered_generation", None, "null at a declared factor"),
    ],
)
def test_rejects_invalid_declared_method_values(
    source: FrameSource,
    column: str,
    value: object,
    error: str,
) -> None:
    frame = _frame(source)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        validate_steering_frame(frame, source=source, max_concepts=2)


def test_structural_only_validation_skips_only_method_values() -> None:
    frame = _frame("main")
    frame["DiffMean_steered_generation"] = None
    frame["DiffMean_strength"] = float("inf")

    validate_steering_frame(
        frame,
        source="main",
        max_concepts=2,
        check_method_values=False,
    )

    frame.loc[0, "original_prompt"] = None
    with pytest.raises(ValueError, match="required input column.*nulls"):
        validate_steering_frame(
            frame,
            source="main",
            max_concepts=2,
            check_method_values=False,
        )


def test_structural_validation_flag_must_be_a_real_boolean() -> None:
    with pytest.raises(TypeError, match="check_method_values must be a boolean"):
        validate_steering_frame(
            _frame("main"),
            source="main",
            max_concepts=2,
            check_method_values=cast(Any, 1),
        )


def test_enforces_blank_and_collapse_thresholds() -> None:
    allowed = _frame("main")
    allowed.loc[:1, "DiffMean_steered_generation"] = " "
    validate_steering_frame(allowed, source="main", max_concepts=2)

    too_many_blanks = _frame("main")
    too_many_blanks.loc[:2, "DiffMean_steered_generation"] = " "
    with pytest.raises(ValueError, match="blank generations"):
        validate_steering_frame(too_many_blanks, source="main", max_concepts=2)

    collapsed = _frame("project")
    collapsed["SelfConditioning_steered_generation"] = "short"
    with pytest.raises(ValueError, match="collapsed output"):
        validate_steering_frame(collapsed, source="project", max_concepts=2)


# ---------------------------------------------------------------------------
# Generation-health collapse tripwires (#1313 transplant): blank-fraction
# bound and median-length floor, each scoped per method over DECLARED cells.
# ---------------------------------------------------------------------------


def test_blank_fraction_exactly_at_the_bound_passes() -> None:
    """5 concepts x 10 inputs x 12 factors = 600 declared cells/method, so
    1% is an exact integer count (6) -- the boundary itself, not an
    approximation of it."""
    frame = _frame("main", max_concepts=5)
    frame.loc[frame.index[:6], "DiffMean_steered_generation"] = " "
    validate_steering_frame(frame, source="main", max_concepts=5)


def test_one_blank_over_the_bound_raises_naming_source_method_count_fraction_bound() -> None:
    frame = _frame("main", max_concepts=5)
    frame.loc[frame.index[:7], "DiffMean_steered_generation"] = " "
    with pytest.raises(
        ValueError,
        match=(
            r"main column 'DiffMean_steered_generation' for method 'DiffMean' has "
            r"7 of 600 blank generations after stripping \(fraction 0\.0117\) "
            r"exceeding the 0\.01 bound"
        ),
    ):
        validate_steering_frame(frame, source="main", max_concepts=5)


def test_max_blank_fraction_override_relaxes_the_bound_for_a_different_population() -> None:
    """A caller pinned to a population this default was never calibrated
    against (see ``merge_baseline_methods.BASELINE_MAX_BLANK_GENERATION_FRACTION``)
    may pass a wider bound; every other caller keeps the strict default."""
    frame = _frame("main", max_concepts=5)
    frame.loc[frame.index[:7], "DiffMean_steered_generation"] = " "

    with pytest.raises(ValueError, match="exceeding the 0.01 bound"):
        validate_steering_frame(frame, source="main", max_concepts=5)

    validate_steering_frame(frame, source="main", max_concepts=5, max_blank_fraction=0.02)


@pytest.mark.parametrize("blank_value", [" ", "\n", "\t", "  \n\t "])
def test_every_whitespace_spelling_counts_as_blank(blank_value: str) -> None:
    frame = _frame("main", max_concepts=5)
    frame.loc[frame.index[:7], "DiffMean_steered_generation"] = blank_value
    with pytest.raises(ValueError, match=r"7 of 600 blank generations"):
        validate_steering_frame(frame, source="main", max_concepts=5)


def test_median_seven_raises_median_eight_passes() -> None:
    below_floor = _frame("main", max_concepts=1)
    below_floor["DiffMean_steered_generation"] = "x" * 7
    with pytest.raises(
        ValueError,
        match=r"collapsed output; median stripped length is 7, minimum is 8",
    ):
        validate_steering_frame(below_floor, source="main", max_concepts=1)

    at_floor = _frame("main", max_concepts=1)
    at_floor["DiffMean_steered_generation"] = "x" * 8
    validate_steering_frame(at_floor, source="main", max_concepts=1)


def test_generation_health_check_runs_per_method_and_names_the_failing_one() -> None:
    frame = _frame("main", max_concepts=5)
    frame.loc[frame.index[:7], "LinearProbe_steered_generation"] = " "
    with pytest.raises(ValueError, match="method 'LinearProbe'") as exc:
        validate_steering_frame(frame, source="main", max_concepts=5)
    message = str(exc.value)
    assert "method 'LinearProbe'" in message
    assert "DiffMean" not in message


# ---------------------------------------------------------------------------
# Small-n delta (controller ruling): the bound is a PURE fraction, with no
# absolute-count floor, applied identically to every declared method --
# including the two aux methods, which declare only factor == 1.0 and so
# have far smaller populations than the FULL_FACTORS methods. Below 100
# declared cells a single blank already exceeds 1%; this is the most
# consequential behavioral difference from the ad hoc count-floor this
# module used to carry, so it is pinned directly rather than left implicit.
# ---------------------------------------------------------------------------


def test_one_blank_in_twenty_declared_aux_cells_raises_naming_the_method() -> None:
    """PromptSteering/LoRA declare only factor == 1.0 (see METHOD_FACTORS),
    so at max_concepts=2 -- the size most fixtures in this file use -- their
    declared population is just 10 * 2 * 1 = 20 cells. A single blank there
    is 5%, nowhere close to the 1% bound's tolerance at a large sweep."""
    frame = _frame("aux", max_concepts=2)
    assert len(frame) == 20
    frame.loc[frame.index[:1], "PromptSteering_steered_generation"] = " "
    with pytest.raises(
        ValueError,
        match=r"method 'PromptSteering'.*1 of 20 blank generations",
    ):
        validate_steering_frame(frame, source="aux", max_concepts=2)


def test_one_blank_in_exactly_one_hundred_declared_cells_passes() -> None:
    """The crossover itself: 1/100 = 1.0%, and the bound is `<=`, not `<` --
    a population landing exactly on the boundary must still pass, matching
    `test_blank_fraction_exactly_at_the_bound_passes` above but pinned at
    the specific population size where a single blank sits exactly on it."""
    frame = _frame("aux", max_concepts=10)
    assert len(frame) == 100
    frame.loc[frame.index[:1], "PromptSteering_steered_generation"] = " "
    validate_steering_frame(frame, source="aux", max_concepts=10)


def test_one_blank_below_the_hundred_cell_crossover_raises_for_a_project_method() -> None:
    """The same strictness is not an aux-only quirk: it applies to the
    project-source SelfConditioning method too. Real
    project-source frames can never carry fewer than 120 declared cells per
    method (`INPUTS_PER_CONCEPT * len(FULL_FACTORS)`, at max_concepts=1), so
    this exercises `_require_generation_health` directly against a
    hand-built 80-cell population -- below the n=100 crossover -- bypassing
    the exact-key-product gate that only `validate_steering_frame` (not this
    function) enforces."""
    frame = pd.DataFrame(
        {
            "SelfConditioning_steered_generation": ["a healthy generation"] * 79 + [" "],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            r"project column 'SelfConditioning_steered_generation' for method "
            r"'SelfConditioning' has 1 of 80 blank generations after stripping "
            r"\(fraction 0\.0125\) exceeding the 0\.01 bound"
        ),
    ):
        _require_generation_health(
            frame,
            FrameSource.PROJECT,
            ("SelfConditioning",),
        )


def test_old_feature_diffmean_project_columns_fail_exact_schema() -> None:
    project = _frame("project")
    project["FeatureDiffMean_steered_generation"] = "legacy generation"
    project["FeatureDiffMean_strength"] = project["factor"]
    with pytest.raises(ValueError, match="unknown method result columns.*FeatureDiffMean"):
        validate_steering_frame(project, source="project", max_concepts=2)


def test_previously_valid_six_method_merged_frame_fails_missing_encdec_columns() -> None:
    ten_method_merged = merge_steering_frames(*_valid_frames(), max_concepts=2)
    encdec_columns = [
        column
        for method in (
            "EncDecDiffMean",
            "EncDecLinearProbe",
            "EncDecPromptSteering",
            "EncDecLoRA",
        )
        for column in (f"{method}_steered_generation", _metric_column(method))
    ]
    six_method_merged = ten_method_merged.drop(columns=encdec_columns)

    with pytest.raises(ValueError, match="missing=.*EncDecDiffMean"):
        validate_steering_frame(six_method_merged, source="merged", max_concepts=2)


@pytest.mark.parametrize(
    "method",
    ["PromptSteering", "LoRA", "EncDecPromptSteering", "EncDecLoRA"],
)
def test_merged_contract_rejects_aux_like_values_off_grid(method: str) -> None:
    merged = merge_steering_frames(*_valid_frames(), max_concepts=2)
    off_grid = merged["factor"].ne(1.0)
    merged.loc[off_grid.idxmax(), _metric_column(method)] = 2.0
    with pytest.raises(ValueError, match="must be null outside factor 1.0"):
        validate_steering_frame(merged, source="merged", max_concepts=2)


@pytest.mark.parametrize(
    "method",
    list(METHOD_FACTORS),
)
def test_merged_contract_rejects_nulls_on_declared_grid(method: str) -> None:
    merged = merge_steering_frames(*_valid_frames(), max_concepts=2)
    declared = merged["factor"].isin(METHOD_FACTORS[method])
    merged.loc[declared.idxmax(), f"{method}_steered_generation"] = None
    with pytest.raises(ValueError, match="null at a declared factor"):
        validate_steering_frame(merged, source="merged", max_concepts=2)


def test_invalid_input_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    project.loc[
        project["concept_id"].eq(0) & project["input_id"].eq(0),
        "input",
    ] = "mismatch"
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    encdec_main_path = tmp_path / "encdec_main.parquet"
    encdec_aux_path = tmp_path / "encdec_aux.parquet"
    project_path = tmp_path / "project.parquet"
    output_path = tmp_path / "merged.parquet"
    main.to_parquet(main_path, index=False)
    aux.to_parquet(aux_path, index=False)
    encdec_main.to_parquet(encdec_main_path, index=False)
    encdec_aux.to_parquet(encdec_aux_path, index=False)
    project.to_parquet(project_path, index=False)
    output_path.write_bytes(b"keep existing output")

    with pytest.raises(ValueError, match="project inputs do not match"):
        merge_steering_parquets(
            main_path,
            aux_path,
            encdec_main_path,
            encdec_aux_path,
            project_path,
            output_path,
            max_concepts=2,
            force=True,
        )

    assert output_path.read_bytes() == b"keep existing output"


def test_merge_refuses_to_overwrite_an_input_parquet(tmp_path: Path) -> None:
    main, aux, encdec_main, encdec_aux, project = _valid_frames()
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    encdec_main_path = tmp_path / "encdec_main.parquet"
    encdec_aux_path = tmp_path / "encdec_aux.parquet"
    project_path = tmp_path / "project.parquet"
    main.to_parquet(main_path, index=False)
    aux.to_parquet(aux_path, index=False)
    encdec_main.to_parquet(encdec_main_path, index=False)
    encdec_aux.to_parquet(encdec_aux_path, index=False)
    project.to_parquet(project_path, index=False)
    before = main_path.read_bytes()

    with pytest.raises(ValueError, match="output path must differ"):
        merge_steering_parquets(
            main_path,
            aux_path,
            encdec_main_path,
            encdec_aux_path,
            project_path,
            main_path,
            max_concepts=2,
        )

    assert main_path.read_bytes() == before


def test_parquet_wrapper_writes_only_the_validated_merged_frame(tmp_path: Path) -> None:
    paths = [
        tmp_path / name
        for name in (
            "main.parquet",
            "aux.parquet",
            "encdec_main.parquet",
            "encdec_aux.parquet",
            "project.parquet",
        )
    ]
    for frame, path in zip(_valid_frames(), paths, strict=True):
        frame.to_parquet(path, index=False)
    output_path = tmp_path / "nested" / "merged.parquet"

    returned = merge_steering_parquets(
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        paths[4],
        output_path,
        max_concepts=2,
    )
    written = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(written, returned)
    validate_steering_frame(written, source="merged", max_concepts=2)


def test_existing_merge_output_requires_force_before_any_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "merged.parquet"
    output_path.write_bytes(b"preserve-me")
    reads = 0

    def forbidden_read(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal reads
        reads += 1
        raise AssertionError("existing-output refusal must precede input reads")

    monkeypatch.setattr(merge_module.pd, "read_parquet", forbidden_read)
    with pytest.raises(FileExistsError, match="pass force=True"):
        merge_steering_parquets(
            tmp_path / "main.parquet",
            tmp_path / "aux.parquet",
            tmp_path / "encdec_main.parquet",
            tmp_path / "encdec_aux.parquet",
            tmp_path / "project.parquet",
            output_path,
            max_concepts=2,
        )
    assert reads == 0
    assert output_path.read_bytes() == b"preserve-me"


def test_merge_force_must_be_a_real_boolean_before_any_input_read(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="force must be a boolean"):
        merge_steering_parquets(
            tmp_path / "main.parquet",
            tmp_path / "aux.parquet",
            tmp_path / "encdec_main.parquet",
            tmp_path / "encdec_aux.parquet",
            tmp_path / "project.parquet",
            tmp_path / "merged.parquet",
            max_concepts=2,
            force=cast(Any, 1),
        )


def test_force_atomically_replaces_existing_merge_with_expected_byte_parity(
    tmp_path: Path,
) -> None:
    paths = [
        tmp_path / name
        for name in (
            "main.parquet",
            "aux.parquet",
            "encdec_main.parquet",
            "encdec_aux.parquet",
            "project.parquet",
        )
    ]
    for frame, path in zip(_valid_frames(), paths, strict=True):
        frame.to_parquet(path, index=False)
    expected = merge_steering_frames(*_valid_frames(), max_concepts=2)
    expected_path = tmp_path / "expected.parquet"
    expected.to_parquet(expected_path, index=False)
    output_path = tmp_path / "merged.parquet"
    output_path.write_bytes(b"old-complete-output")

    returned = merge_steering_parquets(
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        paths[4],
        output_path,
        max_concepts=2,
        force=True,
    )

    pd.testing.assert_frame_equal(returned, expected, check_exact=True)
    assert output_path.read_bytes() == expected_path.read_bytes()


def test_merge_cli_forwards_force_exactly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_merge(
        main_path: Path,
        aux_path: Path,
        encdec_main_path: Path,
        encdec_aux_path: Path,
        project_path: Path,
        output_path: Path,
        *,
        max_concepts: int,
        force: bool,
    ) -> pd.DataFrame:
        captured.update(
            main=main_path,
            aux=aux_path,
            encdec_main=encdec_main_path,
            encdec_aux=encdec_aux_path,
            project=project_path,
            output=output_path,
            max_concepts=max_concepts,
            force=force,
        )
        return pd.DataFrame(index=range(7))

    monkeypatch.setattr(merge_module, "merge_steering_parquets", fake_merge)
    merge_module.main(
        [
            "--main",
            str(tmp_path / "main.parquet"),
            "--aux",
            str(tmp_path / "aux.parquet"),
            "--encdec-main",
            str(tmp_path / "encdec_main.parquet"),
            "--encdec-aux",
            str(tmp_path / "encdec_aux.parquet"),
            "--project",
            str(tmp_path / "project.parquet"),
            "--output",
            str(tmp_path / "merged.parquet"),
            "--max-concepts",
            "2",
            "--force",
        ]
    )
    assert captured == {
        "main": tmp_path / "main.parquet",
        "aux": tmp_path / "aux.parquet",
        "encdec_main": tmp_path / "encdec_main.parquet",
        "encdec_aux": tmp_path / "encdec_aux.parquet",
        "project": tmp_path / "project.parquet",
        "output": tmp_path / "merged.parquet",
        "max_concepts": 2,
        "force": True,
    }


@pytest.mark.parametrize(
    "bad_override",
    [
        (),
        "SelfConditioning",
        ("SelfConditioning", ""),
        ("SelfConditioning", 3),
    ],
)
def test_methods_override_domain_is_validated(bad_override: Any) -> None:
    legacy = _frame("project_legacy")
    with pytest.raises(ValueError, match="methods override must be a nonempty tuple"):
        validate_steering_frame(
            legacy,
            source="project",
            max_concepts=2,
            methods=bad_override,
        )


@pytest.mark.parametrize("source", ["main", "aux"])
def test_methods_override_is_scoped_to_project_and_merged_sources(source: str) -> None:
    frame = _frame(source)
    with pytest.raises(ValueError, match="only for consumers pinned to a"):
        validate_steering_frame(
            frame,
            source=cast(FrameSource, source),
            max_concepts=2,
            methods=("SelfConditioning",),
        )


def test_methods_override_validates_a_baseline_only_merged_frame() -> None:
    """A merged frame with only the four vendored baseline methods (e.g.

    Pythia, which has no project/encoder-decoder method comparison) is
    accepted with an explicit override and rejected without one.
    """

    baseline_methods = ("DiffMean", "LinearProbe", "PromptSteering", "LoRA")
    main = _frame("main")
    aux = _frame("aux")
    merged = main.merge(
        aux.loc[:, [*KEY_COLUMNS, *merge_module.method_columns(("PromptSteering", "LoRA"))]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )

    with pytest.raises(ValueError, match="missing="):
        validate_steering_frame(merged, source="merged", max_concepts=2)

    validate_steering_frame(merged, source="merged", max_concepts=2, methods=baseline_methods)


def test_read_canonical_merged_frame_rejects_an_unknown_surface(tmp_path: Path) -> None:
    """The ``surface`` guard is a runtime check, not just a static ``Literal``.

    ``surface`` is typed ``MergedFrameSurface`` so legitimate callers only
    ever pass ``"evaluation"`` or ``"roundtrip"``, but the dict lookup that
    backs the filename-error wording still has to fail loudly for anything
    else -- e.g. a value that reached this function past the type checker
    (a ``cast``, an untyped caller). The bogus value is asserted to reach the
    unknown-surface branch before any file-existence check runs, so a
    nonexistent path is enough here.
    """
    with pytest.raises(ValueError, match="unknown merged-frame reader surface: 'bogus'"):
        read_canonical_merged_frame(
            tmp_path / MERGED_INPUT_FILENAME,
            surface=cast(MergedFrameSurface, "bogus"),
        )
