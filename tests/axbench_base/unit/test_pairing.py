"""Unit tests for the shared synth pairing helper (issue #508)."""
from __future__ import annotations

from steering.eval.axbench_base.synth.pairing import drop_unpaired_rich_rows


def _rich(cid: int = 5) -> dict:
    return {"concept_id": cid, "content_class": "concept_rich"}


def _avoid(cid: int = 5) -> dict:
    return {"concept_id": cid, "content_class": "concept_avoiding"}


def test_drops_rich_only_concept():
    assert drop_unpaired_rich_rows([_rich(), _rich()], 5) == []


def test_keeps_paired_concept():
    rows = [_rich(), _avoid()]
    assert drop_unpaired_rich_rows(rows, 5) == rows


def test_keeps_when_some_avoid_present():
    rows = [_rich(), _rich(), _avoid()]
    assert drop_unpaired_rich_rows(rows, 5) == rows


def test_keeps_empty_input():
    assert drop_unpaired_rich_rows([], 5) == []


def test_keeps_avoid_only():
    rows = [_avoid()]
    assert drop_unpaired_rich_rows(rows, 5) == rows
