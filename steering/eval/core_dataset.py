"""Efficient, reproducible loading of capped CORE task datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

CORE_SUBSAMPLE_SEED = 1337


def load_selected_core_task_data(
    path: str | Path,
    *,
    max_per_task: int,
) -> list[dict[str, Any]]:
    """Load the same shuffled CORE rows while sparsely parsing capped tasks.

    CORE historically parses every JSONL row, shuffles the resulting objects
    with seed 1337, and then applies ``max_per_task``.  For a positive cap
    smaller than the dataset, the shuffle depends only on row positions.  Scan
    byte offsets, shuffle those positions identically, and JSON-decode only the
    selected rows.  Uncapped tasks retain the sequential full-load path.
    """
    data_path = Path(path)
    if max_per_task <= 0:
        return _load_all_and_shuffle(data_path)

    offsets: list[int] = []
    with data_path.open("rb") as handle:
        while line := handle.readline():
            offsets.append(handle.tell() - len(line))

    if max_per_task >= len(offsets):
        return _load_all_and_shuffle(data_path)

    indices = list(range(len(offsets)))
    random.Random(CORE_SUBSAMPLE_SEED).shuffle(indices)
    selected_indices = indices[:max_per_task]
    with data_path.open("rb") as handle:
        selected: list[dict[str, Any]] = []
        for index in selected_indices:
            handle.seek(offsets[index])
            selected.append(json.loads(handle.readline().strip()))
    return selected


def _load_all_and_shuffle(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = [json.loads(line.strip()) for line in handle]
    random.Random(CORE_SUBSAMPLE_SEED).shuffle(data)
    return data
