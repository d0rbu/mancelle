"""BPB-Adv test fixtures for strict AxBench source seed validation."""

from __future__ import annotations

import pytest

from steering.eval.axbench_bpb_adv import source_validation
from tests.axbench_bpb_adv.source_run_builders import SourceSeedTokenizer


@pytest.fixture(autouse=True)
def _source_seed_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_validation, "get_tokenizer", lambda: SourceSeedTokenizer()
    )
