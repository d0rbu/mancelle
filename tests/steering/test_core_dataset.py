from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from steering.eval import core_dataset
from steering.eval.core_dataset import (
    CORE_SUBSAMPLE_SEED,
    load_selected_core_task_data,
)


def _write_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [
        {
            "index": index,
            "text": f"row {index} — exact UTF-8 byte offsets",
            "choices": [index, index + 1],
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _legacy_selection(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    selected = list(rows)
    random.Random(CORE_SUBSAMPLE_SEED).shuffle(selected)
    return selected[:cap] if cap > 0 else selected


@pytest.mark.parametrize("cap", [-1, 1, 7, 40])
def test_load_selected_core_task_data_matches_legacy_shuffle_exactly(
    tmp_path: Path,
    cap: int,
) -> None:
    data_path = tmp_path / "task.jsonl"
    rows = _write_rows(data_path, 31)

    assert load_selected_core_task_data(data_path, max_per_task=cap) == (
        _legacy_selection(rows, cap)
    )


def test_capped_core_loader_decodes_only_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "task.jsonl"
    _write_rows(data_path, 100)
    real_loads = json.loads
    decoded = 0

    def counting_loads(value):
        nonlocal decoded
        decoded += 1
        return real_loads(value)

    monkeypatch.setattr(core_dataset.json, "loads", counting_loads)

    selected = load_selected_core_task_data(data_path, max_per_task=5)

    assert len(selected) == 5
    assert decoded == 5
