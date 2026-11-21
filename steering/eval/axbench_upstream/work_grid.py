"""Canonical logical generation-cell grids for AxBench-Upstream producers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

from scripts.eval.axbench_upstream.merge_methods import (
    METHOD_FACTORS,
    NL_PROJECT_METHODS,
    SUPPORTED_METHOD_FACTORS,
)
from steering.eval.axbench_upstream.seeds import (
    derive_generation_seed,
    derive_selection_seed,
)

DriverName = Literal["project", "encdec"]

PROJECT_METHODS = ("SelfConditioning", "SelfConditioningMasked")
ENCDEC_MAIN_METHODS = ("EncDecDiffMean", "EncDecLinearProbe")
ENCDEC_AUX_METHODS = ("EncDecPromptSteering", "EncDecLoRA")
_REQUIRED_COLUMNS = ("concept_id", "input_id", "factor")


@dataclass(frozen=True, order=True)
class CellKey:
    """Complete placement-free identity of one generation call."""

    driver: DriverName
    method: str
    source_row_position: int
    concept_id: int
    input_id: int
    factor_hex: str
    selection_seed: int
    token_seed: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-ready representation with stable field order."""

        return asdict(self)

    def validated_source_values(
        self,
        rows: Sequence[Any],
    ) -> tuple[int, int, float]:
        """Return source values after proving this cell still identifies its row."""

        row = rows[self.source_row_position]
        concept_id = int(row.concept_id)
        input_id = int(row.input_id)
        factor = float(row.factor)
        if (
            concept_id != self.concept_id
            or input_id != self.input_id
            or factor.hex() != self.factor_hex
        ):
            owner = "encoder-decoder" if self.driver == "encdec" else self.driver
            raise RuntimeError(f"{owner} CellKey no longer matches its source row: {self}")
        return concept_id, input_id, factor


def _validated_rows(frame: pd.DataFrame) -> tuple[tuple[int, int, float], ...]:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"AxBench cell grid frame is missing required columns: {missing}")
    repeated = [column for column in _REQUIRED_COLUMNS if list(frame.columns).count(column) != 1]
    if repeated:
        raise ValueError(f"AxBench cell grid frame repeats required columns: {repeated}")
    rows: list[tuple[int, int, float]] = []
    identity_rows = frame.loc[:, list(_REQUIRED_COLUMNS)].itertuples(index=False, name=None)
    for position, (concept_id, input_id, factor) in enumerate(identity_rows):
        if type(concept_id) is not int or concept_id < 0:
            raise ValueError(
                "AxBench cell grid concept_id must be a nonnegative int excluding bool "
                f"at source row {position}; got {concept_id!r}"
            )
        if type(input_id) is not int or input_id < 0:
            raise ValueError(
                "AxBench cell grid input_id must be a nonnegative int excluding bool "
                f"at source row {position}; got {input_id!r}"
            )
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ValueError(
                "AxBench cell grid factor must be a real number excluding bool "
                f"at source row {position}; got {factor!r}"
            )
        rows.append((concept_id, input_id, float(factor)))
    if not rows:
        raise ValueError("AxBench cell grid frame must contain at least one row")
    return tuple(rows)


def _cell(
    *,
    driver: DriverName,
    method: str,
    source_row_position: int,
    concept_id: int,
    input_id: int,
    factor: float,
    seed: int,
) -> CellKey:
    return CellKey(
        driver=driver,
        method=method,
        source_row_position=source_row_position,
        concept_id=concept_id,
        input_id=input_id,
        factor_hex=float(factor).hex(),
        selection_seed=derive_selection_seed(seed, concept_id, input_id),
        token_seed=derive_generation_seed(seed, method, concept_id, input_id, factor),
    )


def _require_matching_factor_grids(methods: tuple[str, ...], *, label: str) -> tuple[float, ...]:
    """Return one method grid after proving every sibling declares it exactly."""

    try:
        registry = SUPPORTED_METHOD_FACTORS if methods == NL_PROJECT_METHODS else METHOD_FACTORS
        first = tuple(registry[methods[0]])
        mismatches = {
            method: tuple(registry[method])
            for method in methods[1:]
            if tuple(registry[method]) != first
        }
    except KeyError as exc:
        raise ValueError(f"{label} factor registry is missing method {exc.args[0]!r}") from exc
    if mismatches:
        raise ValueError(
            f"{label} methods must declare identical factor grids: "
            f"{methods[0]}={first!r}, mismatches={mismatches!r}"
        )
    return first


def build_project_cell_grid(
    frame: pd.DataFrame,
    *,
    seed: int,
    methods: tuple[str, ...] = PROJECT_METHODS,
) -> tuple[CellKey, ...]:
    """Return project-method cells in source-row then method order."""

    if methods not in {PROJECT_METHODS, NL_PROJECT_METHODS}:
        raise ValueError(
            "project cell-grid methods must be the default or NL-enabled tuple; "
            f"got {methods!r}"
        )
    project_factors = _require_matching_factor_grids(methods, label="project")
    cells: list[CellKey] = []
    for position, (concept_id, input_id, factor) in enumerate(_validated_rows(frame)):
        if factor not in project_factors:
            raise ValueError(
                f"project cell-grid factor {factor!r} at source row {position} "
                "is outside the declared method grid"
            )
        for method in methods:
            cells.append(
                _cell(
                    driver="project",
                    method=method,
                    source_row_position=position,
                    concept_id=concept_id,
                    input_id=input_id,
                    factor=factor,
                    seed=seed,
                )
            )
    return tuple(cells)


def build_encdec_cell_grid(frame: pd.DataFrame, *, seed: int) -> tuple[CellKey, ...]:
    """Return encoder-decoder cells in source-row and production method order."""

    main_factors = _require_matching_factor_grids(
        ENCDEC_MAIN_METHODS,
        label="encoder-decoder main",
    )
    aux_factors = _require_matching_factor_grids(
        ENCDEC_AUX_METHODS,
        label="encoder-decoder auxiliary",
    )
    if len(aux_factors) != 1 or aux_factors[0] not in main_factors:
        raise ValueError(
            "encoder-decoder auxiliary methods must declare one factor from the main grid: "
            f"main={main_factors!r}, auxiliary={aux_factors!r}"
        )
    cells: list[CellKey] = []
    for position, (concept_id, input_id, factor) in enumerate(_validated_rows(frame)):
        if factor not in main_factors:
            raise ValueError(
                f"encoder-decoder cell-grid factor {factor!r} at source row {position} "
                "is outside the declared method grid"
            )
        methods = (
            (*ENCDEC_MAIN_METHODS, *ENCDEC_AUX_METHODS)
            if factor == aux_factors[0]
            else ENCDEC_MAIN_METHODS
        )
        for method in methods:
            cells.append(
                _cell(
                    driver="encdec",
                    method=method,
                    source_row_position=position,
                    concept_id=concept_id,
                    input_id=input_id,
                    factor=factor,
                    seed=seed,
                )
            )
    return tuple(cells)


def _validate_ownership_coordinates(*, rank: int, world_size: int) -> None:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError(f"world_size must be a positive int excluding bool, got {world_size!r}")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"rank must be an int excluding bool in [0, {world_size}), got {rank!r}")


def owned_cells(
    cells: tuple[CellKey, ...],
    *,
    rank: int,
    world_size: int,
) -> tuple[CellKey, ...]:
    """Return the stable rank-striped subset of ``cells``."""

    _validate_ownership_coordinates(rank=rank, world_size=world_size)
    return tuple(cell for index, cell in enumerate(cells) if index % world_size == rank)


def owned_project_cells(
    cells: tuple[CellKey, ...],
    *,
    rank: int,
    world_size: int,
    methods: tuple[str, ...] = PROJECT_METHODS,
) -> tuple[CellKey, ...]:
    """Return stable rank-striped project prompt groups.

    Every cell for one ``(concept_id, input_id)`` pair has the same owner, so
    both project methods retain their complete factor grid on one rank.
    Filtering the original sequence preserves caller order even when pair rows
    are interleaved.
    """

    _validate_ownership_coordinates(rank=rank, world_size=world_size)
    if methods not in {PROJECT_METHODS, NL_PROJECT_METHODS}:
        raise ValueError(
            "project ownership methods must be the default or NL-enabled tuple; "
            f"got {methods!r}"
        )
    allowed_methods = frozenset(methods)
    unexpected = [
        cell for cell in cells if cell.driver != "project" or cell.method not in allowed_methods
    ]
    if unexpected:
        raise ValueError(
            f"project ownership requires canonical project cells; first invalid: {unexpected[0]}"
        )

    group_indices: dict[tuple[int, int], int] = {}
    owners: list[int] = []
    for cell in cells:
        pair = (cell.concept_id, cell.input_id)
        group_index = group_indices.setdefault(pair, len(group_indices))
        owners.append(group_index % world_size)
    return tuple(cell for cell, owner in zip(cells, owners, strict=True) if owner == rank)


def owned_encdec_cells(
    cells: tuple[CellKey, ...],
    *,
    rank: int,
    world_size: int,
) -> tuple[CellKey, ...]:
    """Return stable rank-striped encoder-decoder prompt groups.

    Every cell for one ``(concept_id, input_id)`` pair has the same owner.  In
    particular, this keeps the two direction methods batch-compatible across
    their complete factor grid.  Filtering the original sequence preserves
    the caller's cell order even when pair rows are interleaved.
    """

    _validate_ownership_coordinates(rank=rank, world_size=world_size)
    allowed_methods = frozenset((*ENCDEC_MAIN_METHODS, *ENCDEC_AUX_METHODS))
    unexpected = [
        cell for cell in cells if cell.driver != "encdec" or cell.method not in allowed_methods
    ]
    if unexpected:
        raise ValueError(
            "encoder-decoder ownership requires canonical encoder-decoder cells; first invalid: "
            f"{unexpected[0]}"
        )

    group_indices: dict[tuple[int, int], int] = {}
    owners: list[int] = []
    for cell in cells:
        pair = (cell.concept_id, cell.input_id)
        group_index = group_indices.setdefault(pair, len(group_indices))
        owners.append(group_index % world_size)
    return tuple(cell for cell, owner in zip(cells, owners, strict=True) if owner == rank)


__all__ = (
    "ENCDEC_AUX_METHODS",
    "ENCDEC_MAIN_METHODS",
    "NL_PROJECT_METHODS",
    "PROJECT_METHODS",
    "CellKey",
    "build_encdec_cell_grid",
    "build_project_cell_grid",
    "owned_cells",
    "owned_encdec_cells",
    "owned_project_cells",
)
