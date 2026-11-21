"""Canonical AxBench-Upstream work-grid contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS
from steering.eval.axbench_upstream import work_grid
from steering.eval.axbench_upstream.seeds import derive_generation_seed, derive_selection_seed
from steering.eval.axbench_upstream.work_grid import (
    ENCDEC_AUX_METHODS,
    ENCDEC_MAIN_METHODS,
    NL_PROJECT_METHODS,
    PROJECT_METHODS,
    CellKey,
    build_encdec_cell_grid,
    build_project_cell_grid,
    owned_cells,
    owned_encdec_cells,
    owned_project_cells,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _frame(
    *,
    concepts: int = 1,
    inputs: int = 1,
    factors: tuple[float, ...] = FULL_FACTORS,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"concept_id": concept_id, "input_id": input_id, "factor": factor}
            for concept_id in range(concepts)
            for input_id in range(inputs)
            for factor in factors
        ]
    )


def test_method_orders_are_the_production_orders() -> None:
    assert PROJECT_METHODS == ("SelfConditioning", "SelfConditioningMasked")
    assert NL_PROJECT_METHODS == (*PROJECT_METHODS, "NLSteering")
    assert ENCDEC_MAIN_METHODS == ("EncDecDiffMean", "EncDecLinearProbe")
    assert ENCDEC_AUX_METHODS == ("EncDecPromptSteering", "EncDecLoRA")


def test_project_grid_rejects_sibling_factor_registry_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = dict(work_grid.METHOD_FACTORS)
    factors["SelfConditioningMasked"] = (0.0, 1.0)
    monkeypatch.setattr(work_grid, "METHOD_FACTORS", factors)

    with pytest.raises(ValueError, match="project methods must declare identical factor grids"):
        build_project_cell_grid(_frame(), seed=42)


def test_grid_rejects_missing_factor_registry_method(monkeypatch: pytest.MonkeyPatch) -> None:
    factors = dict(work_grid.METHOD_FACTORS)
    del factors["SelfConditioningMasked"]
    monkeypatch.setattr(work_grid, "METHOD_FACTORS", factors)

    with pytest.raises(ValueError, match="factor registry is missing method"):
        build_project_cell_grid(_frame(), seed=42)


def test_encdec_grid_rejects_main_and_aux_factor_registry_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = dict(work_grid.METHOD_FACTORS)
    factors["EncDecLinearProbe"] = (0.0, 1.0)
    monkeypatch.setattr(work_grid, "METHOD_FACTORS", factors)
    with pytest.raises(
        ValueError,
        match="encoder-decoder main methods must declare identical factor grids",
    ):
        build_encdec_cell_grid(_frame(), seed=42)

    factors = dict(work_grid.METHOD_FACTORS)
    factors["EncDecLinearProbe"] = factors["EncDecDiffMean"]
    factors["EncDecPromptSteering"] = (1.0, 2.0)
    factors["EncDecLoRA"] = (1.0, 2.0)
    monkeypatch.setattr(work_grid, "METHOD_FACTORS", factors)
    with pytest.raises(ValueError, match="auxiliary methods must declare one factor"):
        build_encdec_cell_grid(_frame(), seed=42)


def test_project_c2_has_exactly_480_cells() -> None:
    cells = build_project_cell_grid(_frame(concepts=2, inputs=10), seed=42)

    assert len(cells) == 2 * 10 * 12 * 2 == 480
    assert {cell.driver for cell in cells} == {"project"}
    assert tuple(cell.method for cell in cells[:4]) == (
        "SelfConditioning",
        "SelfConditioningMasked",
        "SelfConditioning",
        "SelfConditioningMasked",
    )
    assert [cell.source_row_position for cell in cells[:4]] == [0, 0, 1, 1]


def test_nl_project_grid_is_explicit_and_preserves_old_cell_identity() -> None:
    frame = _frame(concepts=2, inputs=2)
    ordinary = build_project_cell_grid(frame, seed=42)
    enabled = build_project_cell_grid(frame, seed=42, methods=NL_PROJECT_METHODS)

    assert len(enabled) == len(ordinary) * 3 // 2
    assert tuple(cell for cell in enabled if cell.method in PROJECT_METHODS) == ordinary
    assert tuple(cell.method for cell in enabled[:3]) == NL_PROJECT_METHODS
    nl = tuple(cell for cell in enabled if cell.method == "NLSteering")
    assert len(nl) == len(frame)
    for cell in nl:
        assert cell.selection_seed == derive_selection_seed(
            42,
            cell.concept_id,
            cell.input_id,
        )
        assert cell.token_seed == derive_generation_seed(
            42,
            "NLSteering",
            cell.concept_id,
            cell.input_id,
            float.fromhex(cell.factor_hex),
        )
    factor_one = next(
        cell
        for cell in nl
        if (cell.concept_id, cell.input_id, cell.factor_hex) == (0, 0, (1.0).hex())
    )
    assert factor_one.token_seed == 1_228_402_498


@pytest.mark.parametrize(
    "methods",
    [(), ("NLSteering",), (*PROJECT_METHODS, "NLSteering", "NLSteering")],
)
def test_project_grid_rejects_noncanonical_method_selections(methods: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="default or NL-enabled tuple"):
        build_project_cell_grid(_frame(), seed=42, methods=methods)


def test_nl_project_ownership_keeps_complete_prompt_groups() -> None:
    cells = build_project_cell_grid(
        _frame(concepts=2, inputs=10),
        seed=42,
        methods=NL_PROJECT_METHODS,
    )
    stripes = [
        owned_project_cells(
            cells,
            rank=rank,
            world_size=4,
            methods=NL_PROJECT_METHODS,
        )
        for rank in range(4)
    ]

    assert tuple(cell for stripe in stripes for cell in stripe) != cells
    assert {cell for stripe in stripes for cell in stripe} == set(cells)
    for concept_id in range(2):
        for input_id in range(10):
            owners = [
                rank
                for rank, stripe in enumerate(stripes)
                if any(
                    (cell.concept_id, cell.input_id) == (concept_id, input_id)
                    for cell in stripe
                )
            ]
            assert len(owners) == 1


def test_encdec_c2_has_exactly_520_cells() -> None:
    cells = build_encdec_cell_grid(_frame(concepts=2, inputs=10), seed=42)

    assert len(cells) == (2 * 10 * 12 * 2) + (2 * 10 * 2) == 520
    assert {cell.driver for cell in cells} == {"encdec"}
    factor_one = [cell for cell in cells if cell.factor_hex == (1.0).hex()]
    assert len(factor_one) == 2 * 10 * 4
    assert {cell.method for cell in factor_one} == {
        *ENCDEC_MAIN_METHODS,
        *ENCDEC_AUX_METHODS,
    }


def test_c500_arithmetic_matches_the_declared_campaign_cost_model() -> None:
    frame = _frame(concepts=500, inputs=10)

    assert len(build_project_cell_grid(frame, seed=42)) == 120_000
    assert len(build_encdec_cell_grid(frame, seed=42)) == 130_000


def test_project_cell_identity_has_literal_seed_and_float_goldens() -> None:
    cell = build_project_cell_grid(
        pd.DataFrame([{"concept_id": 0, "input_id": 0, "factor": 1.0}]),
        seed=42,
    )[0]

    assert cell == CellKey(
        driver="project",
        method="SelfConditioning",
        source_row_position=0,
        concept_id=0,
        input_id=0,
        factor_hex="0x1.0000000000000p+0",
        selection_seed=448_567_669,
        token_seed=320_911_683,
    )
    assert cell.to_dict() == {
        "driver": "project",
        "method": "SelfConditioning",
        "source_row_position": 0,
        "concept_id": 0,
        "input_id": 0,
        "factor_hex": "0x1.0000000000000p+0",
        "selection_seed": 448_567_669,
        "token_seed": 320_911_683,
    }


@pytest.mark.parametrize(
    ("builder", "owner"),
    [
        (build_project_cell_grid, "project"),
        (build_encdec_cell_grid, "encoder-decoder"),
    ],
)
def test_cell_source_validation_preserves_signed_zero(builder: Any, owner: str) -> None:
    frame = _frame(factors=(-0.0,))
    cell = builder(frame, seed=42)[0]
    rows = tuple(frame.itertuples(index=False))

    concept_id, input_id, factor = cell.validated_source_values(rows)
    assert (concept_id, input_id) == (0, 0)
    assert factor.hex() == "-0x0.0p+0"
    with pytest.raises(RuntimeError, match=rf"^{owner} CellKey no longer matches"):
        replace(cell, factor_hex=(0.0).hex()).validated_source_values(rows)


@pytest.mark.parametrize(
    ("field_name", "mismatched_value"),
    [("concept_id", 1), ("input_id", 1), ("factor_hex", (2.0).hex())],
)
@pytest.mark.parametrize(
    ("builder", "owner"),
    [
        (build_project_cell_grid, "project"),
        (build_encdec_cell_grid, "encoder-decoder"),
    ],
)
def test_cell_source_validation_rejects_each_identity_mismatch(
    builder: Any,
    owner: str,
    field_name: str,
    mismatched_value: Any,
) -> None:
    frame = _frame(factors=(1.0,))
    cell = builder(frame, seed=42)[0]

    with pytest.raises(RuntimeError, match=rf"^{owner} CellKey no longer matches"):
        replace(cell, **{field_name: mismatched_value}).validated_source_values(
            tuple(frame.itertuples(index=False))
        )


def test_selection_seed_is_shared_by_methods_and_factors_but_token_seed_is_not() -> None:
    cells = build_project_cell_grid(_frame(factors=(0.0, 1.0)), seed=42)

    assert len({cell.selection_seed for cell in cells}) == 1
    assert len({cell.token_seed for cell in cells}) == 4
    assert len({cell.factor_hex for cell in cells}) == 2


def test_auxiliary_methods_exist_only_at_exact_factor_one() -> None:
    cells = build_encdec_cell_grid(_frame(factors=(0.0, 1.0, 2.0)), seed=91)
    by_position: dict[int, list[str]] = {}
    for cell in cells:
        by_position.setdefault(cell.source_row_position, []).append(cell.method)

    assert by_position == {
        0: list(ENCDEC_MAIN_METHODS),
        1: [*ENCDEC_MAIN_METHODS, *ENCDEC_AUX_METHODS],
        2: list(ENCDEC_MAIN_METHODS),
    }


def test_source_row_position_tracks_supplied_order_without_reordering() -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 8, "input_id": 2, "factor": 2.0},
            {"concept_id": 3, "input_id": 9, "factor": 0.0},
            {"concept_id": 8, "input_id": 1, "factor": 1.0},
        ],
        index=[91, 17, 400],
    )

    cells = build_project_cell_grid(frame, seed=3)

    assert [(cell.source_row_position, cell.concept_id, cell.input_id) for cell in cells[::2]] == [
        (0, 8, 2),
        (1, 3, 9),
        (2, 8, 1),
    ]


def test_grid_builders_do_not_mutate_the_input_frame() -> None:
    frame = _frame(concepts=2, inputs=2)
    before = frame.copy(deep=True)

    build_project_cell_grid(frame, seed=42)
    build_encdec_cell_grid(frame, seed=42)

    pd.testing.assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("world_size", [1, 2, 3, 7, 100])
def test_rank_stripes_form_an_ordered_disjoint_exact_partition(world_size: int) -> None:
    cells = build_encdec_cell_grid(_frame(concepts=2, inputs=2), seed=42)
    stripes = [owned_cells(cells, rank=rank, world_size=world_size) for rank in range(world_size)]

    assert sum(len(stripe) for stripe in stripes) == len(cells)
    assert set().union(*(set(stripe) for stripe in stripes)) == set(cells)
    for left_rank, left in enumerate(stripes):
        expected = tuple(
            cell for index, cell in enumerate(cells) if index % world_size == left_rank
        )
        assert left == expected
        for right in stripes[left_rank + 1 :]:
            assert set(left).isdisjoint(right)


def test_more_ranks_than_cells_produces_explicit_empty_stripes() -> None:
    cells = build_project_cell_grid(_frame(factors=(1.0,)), seed=42)

    assert len(cells) == 2
    assert owned_cells(cells, rank=0, world_size=4) == (cells[0],)
    assert owned_cells(cells, rank=1, world_size=4) == (cells[1],)
    assert owned_cells(cells, rank=2, world_size=4) == ()
    assert owned_cells(cells, rank=3, world_size=4) == ()


@pytest.mark.parametrize("owner", [owned_cells, owned_encdec_cells, owned_project_cells])
@pytest.mark.parametrize("world_size", [0, -1, True, False, 1.5, "2", None])
def test_ownership_rejects_invalid_world_size(owner: Any, world_size: object) -> None:
    with pytest.raises(ValueError, match="world_size must be a positive int excluding bool"):
        owner((), rank=0, world_size=cast(Any, world_size))


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 2), (2, 2), (3, 2), (True, 2), (False, 2), (0.0, 2), ("0", 2), (None, 2)],
)
@pytest.mark.parametrize("owner", [owned_cells, owned_encdec_cells, owned_project_cells])
def test_ownership_rejects_invalid_rank(owner: Any, rank: object, world_size: int) -> None:
    with pytest.raises(ValueError, match=r"rank must be an int excluding bool in \[0, 2\)"):
        owner((), rank=cast(Any, rank), world_size=world_size)


@pytest.mark.parametrize(
    ("builder", "owner"),
    [
        (build_project_cell_grid, owned_project_cells),
        (build_encdec_cell_grid, owned_encdec_cells),
    ],
)
@pytest.mark.parametrize("world_size", [1, 2, 3, 7, 100])
def test_prompt_group_stripes_are_stable_ordered_disjoint_and_exhaustive(
    builder: Any,
    owner: Any,
    world_size: int,
) -> None:
    cells = builder(_frame(concepts=2, inputs=2), seed=42)
    stripes = [owner(cells, rank=rank, world_size=world_size) for rank in range(world_size)]

    assert stripes == [
        owner(cells, rank=rank, world_size=world_size) for rank in range(world_size)
    ]
    assert sum(len(stripe) for stripe in stripes) == len(cells)
    assert set().union(*(set(stripe) for stripe in stripes)) == set(cells)
    positions = {cell: index for index, cell in enumerate(cells)}
    for left_rank, left in enumerate(stripes):
        assert [positions[cell] for cell in left] == sorted(positions[cell] for cell in left)
        for right in stripes[left_rank + 1 :]:
            assert set(left).isdisjoint(right)

    for pair in {(cell.concept_id, cell.input_id) for cell in cells}:
        pair_cells = tuple(cell for cell in cells if (cell.concept_id, cell.input_id) == pair)
        owners = [
            rank for rank, stripe in enumerate(stripes) if set(pair_cells).intersection(stripe)
        ]
        assert len(owners) == 1
        assert tuple(cell for cell in stripes[owners[0]] if cell in pair_cells) == pair_cells


@pytest.mark.parametrize(
    ("world_size", "expected_pairs_per_rank"),
    [(1, [20]), (2, [10, 10]), (4, [5, 5, 5, 5])],
)
def test_canonical_project_prompt_groups_keep_exact_width_24(
    world_size: int,
    expected_pairs_per_rank: list[int],
) -> None:
    cells = build_project_cell_grid(_frame(concepts=2, inputs=10), seed=42)
    stripes = [
        owned_project_cells(cells, rank=rank, world_size=world_size) for rank in range(world_size)
    ]

    pairs_per_rank: list[int] = []
    for stripe in stripes:
        pairs = {(cell.concept_id, cell.input_id) for cell in stripe}
        pairs_per_rank.append(len(pairs))
        for pair in pairs:
            pair_cells = tuple(cell for cell in stripe if (cell.concept_id, cell.input_id) == pair)
            assert len(pair_cells) == len(FULL_FACTORS) * len(PROJECT_METHODS) == 24
            assert tuple(cell.method for cell in pair_cells) == PROJECT_METHODS * len(FULL_FACTORS)
            assert tuple(float.fromhex(cell.factor_hex) for cell in pair_cells[::2]) == FULL_FACTORS
    assert pairs_per_rank == expected_pairs_per_rank


def test_canonical_project_w4_first_and_bulk_stages_have_exact_group_loads() -> None:
    cells = build_project_cell_grid(_frame(concepts=2, inputs=10), seed=42)
    stripes = [owned_project_cells(cells, rank=rank, world_size=4) for rank in range(4)]

    first_pairs = [
        {(cell.concept_id, cell.input_id) for cell in stripe if cell.concept_id == 0}
        for stripe in stripes
    ]
    bulk_pairs = [
        {(cell.concept_id, cell.input_id) for cell in stripe if cell.concept_id != 0}
        for stripe in stripes
    ]

    assert [len(pairs) for pairs in first_pairs] == [3, 3, 2, 2]
    assert [len(pairs) for pairs in bulk_pairs] == [2, 2, 3, 3]
    assert [sum(cell.concept_id == 0 for cell in stripe) for stripe in stripes] == [72, 72, 48, 48]
    assert [sum(cell.concept_id != 0 for cell in stripe) for stripe in stripes] == [48, 48, 72, 72]


def test_project_prompt_group_ownership_handles_interleaved_uneven_groups_and_empty_ranks() -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 9, "input_id": 4, "factor": 0.0},
            {"concept_id": 2, "input_id": 8, "factor": 0.0},
            {"concept_id": 9, "input_id": 4, "factor": 1.0},
            {"concept_id": 7, "input_id": 1, "factor": 2.0},
            {"concept_id": 2, "input_id": 8, "factor": 1.0},
        ]
    )
    cells = build_project_cell_grid(frame, seed=19)
    expected_pairs = ((9, 4), (2, 8), (7, 1))

    stripes = [owned_project_cells(cells, rank=rank, world_size=5) for rank in range(5)]

    for rank, pair in enumerate(expected_pairs):
        assert stripes[rank] == tuple(
            cell for cell in cells if (cell.concept_id, cell.input_id) == pair
        )
    assert [len(stripe) for stripe in stripes] == [4, 4, 2, 0, 0]


@pytest.mark.parametrize(
    "invalid",
    [
        build_encdec_cell_grid(_frame(factors=(1.0,)), seed=42)[0],
        replace(
            build_project_cell_grid(_frame(factors=(1.0,)), seed=42)[0],
            method="UnknownProjectMethod",
        ),
    ],
)
def test_project_group_ownership_rejects_noncanonical_cells(invalid: CellKey) -> None:
    with pytest.raises(ValueError, match="ownership requires canonical project cells"):
        owned_project_cells((invalid,), rank=0, world_size=1)


def test_project_group_ownership_rejects_unsupported_selected_methods() -> None:
    with pytest.raises(ValueError, match="methods must be the default or NL-enabled tuple"):
        owned_project_cells(
            (),
            rank=0,
            world_size=1,
            methods=("SelfConditioning",),
        )


@pytest.mark.parametrize(
    ("world_size", "expected_pairs_per_rank"),
    [(2, [10, 10]), (4, [5, 5, 5, 5])],
)
def test_canonical_encdec_direction_groups_keep_width_24_with_auxiliary_cells(
    world_size: int,
    expected_pairs_per_rank: list[int],
) -> None:
    cells = build_encdec_cell_grid(_frame(concepts=2, inputs=10), seed=42)
    stripes = [
        owned_encdec_cells(cells, rank=rank, world_size=world_size) for rank in range(world_size)
    ]

    pairs_per_rank: list[int] = []
    for stripe in stripes:
        pairs = {(cell.concept_id, cell.input_id) for cell in stripe}
        pairs_per_rank.append(len(pairs))
        for pair in pairs:
            direction_cells = [
                cell
                for cell in stripe
                if (cell.concept_id, cell.input_id) == pair and cell.method in ENCDEC_MAIN_METHODS
            ]
            auxiliary_cells = [
                cell
                for cell in stripe
                if (cell.concept_id, cell.input_id) == pair and cell.method in ENCDEC_AUX_METHODS
            ]
            assert len(direction_cells) == len(FULL_FACTORS) * len(ENCDEC_MAIN_METHODS) == 24
            assert [cell.method for cell in auxiliary_cells] == list(ENCDEC_AUX_METHODS)
    assert pairs_per_rank == expected_pairs_per_rank


def test_encdec_prompt_group_ownership_handles_interleaved_uneven_groups_and_empty_ranks() -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 9, "input_id": 4, "factor": 0.0},
            {"concept_id": 2, "input_id": 8, "factor": 0.0},
            {"concept_id": 9, "input_id": 4, "factor": 1.0},
            {"concept_id": 7, "input_id": 1, "factor": 2.0},
            {"concept_id": 2, "input_id": 8, "factor": 1.0},
        ]
    )
    cells = build_encdec_cell_grid(frame, seed=19)
    expected_pairs = ((9, 4), (2, 8), (7, 1))

    stripes = [owned_encdec_cells(cells, rank=rank, world_size=5) for rank in range(5)]

    for rank, pair in enumerate(expected_pairs):
        assert stripes[rank] == tuple(
            cell for cell in cells if (cell.concept_id, cell.input_id) == pair
        )
    assert [len(stripe) for stripe in stripes] == [6, 6, 2, 0, 0]


@pytest.mark.parametrize(
    "invalid",
    [
        build_project_cell_grid(_frame(factors=(1.0,)), seed=42)[0],
        replace(
            build_encdec_cell_grid(_frame(factors=(1.0,)), seed=42)[0],
            method="UnknownEncDecMethod",
        ),
    ],
)
def test_encdec_group_ownership_rejects_noncanonical_cells(invalid: CellKey) -> None:
    with pytest.raises(
        ValueError,
        match="ownership requires canonical encoder-decoder cells",
    ):
        owned_encdec_cells((invalid,), rank=0, world_size=1)


@pytest.mark.parametrize("builder", [build_project_cell_grid, build_encdec_cell_grid])
@pytest.mark.parametrize("missing", ["concept_id", "input_id", "factor"])
def test_grid_rejects_each_missing_identity_column(builder: Any, missing: str) -> None:
    frame = _frame().drop(columns=missing)

    with pytest.raises(ValueError, match=rf"missing required columns: \['{missing}'\]"):
        builder(frame, seed=42)


@pytest.mark.parametrize("builder", [build_project_cell_grid, build_encdec_cell_grid])
@pytest.mark.parametrize("repeated", ["concept_id", "input_id", "factor"])
def test_grid_rejects_each_repeated_identity_column(builder: Any, repeated: str) -> None:
    frame = _frame()
    frame.insert(len(frame.columns), repeated, frame[repeated], allow_duplicates=True)

    with pytest.raises(ValueError, match=rf"repeats required columns: \['{repeated}'\]"):
        builder(frame, seed=42)


@pytest.mark.parametrize("builder", [build_project_cell_grid, build_encdec_cell_grid])
def test_grid_rejects_empty_frame(builder: Any) -> None:
    with pytest.raises(ValueError, match="must contain at least one row"):
        builder(_frame().iloc[:0], seed=42)


@pytest.mark.parametrize("column", ["concept_id", "input_id"])
@pytest.mark.parametrize("bad", [-1, True, False, 1.5, "1", None])
def test_grid_rejects_malformed_integer_identity(column: str, bad: object) -> None:
    frame = _frame(factors=(1.0,)).astype({"concept_id": "object", "input_id": "object"})
    frame.loc[0, column] = bad

    with pytest.raises(ValueError, match=f"{column} must be a nonnegative int excluding bool"):
        build_project_cell_grid(frame, seed=42)


@pytest.mark.parametrize("bad", [True, False, "1", None, object()])
def test_grid_rejects_nonnumeric_factor(bad: object) -> None:
    frame = _frame(factors=(1.0,)).astype({"factor": "object"})
    frame.loc[0, "factor"] = bad

    with pytest.raises(ValueError, match="factor must be a real number excluding bool"):
        build_encdec_cell_grid(frame, seed=42)


@pytest.mark.parametrize("bad", [-1.0, 0.1, 7.0, float("inf"), float("nan")])
@pytest.mark.parametrize("builder", [build_project_cell_grid, build_encdec_cell_grid])
def test_grid_rejects_factor_outside_declared_grid(builder: Any, bad: float) -> None:
    frame = pd.DataFrame([{"concept_id": 0, "input_id": 0, "factor": bad}])

    with pytest.raises(ValueError, match="outside the declared method grid"):
        builder(frame, seed=42)


@pytest.mark.parametrize("bad_seed", [-1, True, 1.5, 1 << 63])
def test_grid_reuses_authoritative_seed_domain_validation(bad_seed: object) -> None:
    with pytest.raises(ValueError, match="base_seed must be an integer"):
        build_project_cell_grid(_frame(factors=(1.0,)), seed=cast(Any, bad_seed))


def test_grid_json_is_stable_across_hash_seeds(tmp_path: Path) -> None:
    script = tmp_path / "print_grid.py"
    script.write_text(
        """import json
import pandas as pd
from steering.eval.axbench_upstream.work_grid import build_encdec_cell_grid
frame = pd.DataFrame([{"concept_id": 2, "input_id": 4, "factor": 1.0}])
print(json.dumps([cell.to_dict() for cell in build_encdec_cell_grid(frame, seed=42)], sort_keys=True))
""",
        encoding="utf-8",
    )
    outputs = []
    for hash_seed in ("0", "1", "123456"):
        completed = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            env={
                **os.environ,
                "PYTHONHASHSEED": hash_seed,
                "PYTHONPATH": str(_REPO_ROOT),
            },
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs[0] == outputs[1] == outputs[2]
