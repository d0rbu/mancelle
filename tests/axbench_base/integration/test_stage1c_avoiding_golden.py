"""Committed-golden exact-equality harness for Stage 1c negative concept_avoiding output.

The committed golden parquet anchors the **pre-#518** Stage 1c concept_avoiding output. It
is captured by driving ``_synthesize_concept_negatives`` (the serial Stage 1c path) through
the then-current QC path, mocking ONLY at the extraction-tokenizer + batched-extractor
boundary (``qc.get_extraction_tokenizer`` / ``qc._run_batched_extractor``). That boundary is
stable across the #518 swap from ``fc_active_on_continuation`` to ``classify_avoid_continuation``,
so the post-#518 code reproduces byte-identical rows from the SAME scenario.

The scenario is deliberately NO_TOKENS-free (only INACTIVE-accept and ACTIVE->retry->INACTIVE
rows): #518 INTENTIONALLY changes behavior on the NO_TOKENS / empty region (accept -> reject),
so byte-identity here proves only that the non-degenerate region is preserved exactly. The
degenerate region is covered by ``test_stage1c_seam_merge_no_tokens_rejected`` and
``test_stage1c_empty_continuation_rejected`` in this same file.

WARNING: re-running with ``AXBENCH_CAPTURE_GOLDEN=1`` on POST-#518 code RE-BASELINES the golden
to current behavior and DESTROYS the pre-change equivalence anchor. Re-capture ONLY for an
intentional new baseline, and NEVER as a "fix" for a failing diff (a failing diff means behavior
drifted — investigate it, do not paper over it).

Capture (run ONCE on the pre-#518 code, then commit the parquet; see WARNING):

    AXBENCH_CAPTURE_GOLDEN=1 uv run pytest \
      tests/axbench_base/integration/test_stage1c_avoiding_golden.py::test_stage1c_avoiding_golden -q

Verify (every normal run):

    uv run pytest tests/axbench_base/integration/test_stage1c_avoiding_golden.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.synth import negative_mode, qc
from steering.eval.axbench_base.synth.negative_mode import (
    NEG_TRAINING_SCHEMA,
    _synthesize_concept_negatives,
)
from steering.eval.axbench_base.types import Concept
from tests.axbench_base.integration._golden_harness import (
    FC,
    GoldenExtractionTokenizer,
    PromptKeyedSynthClient,
    StubNanochatTokenizer,
    assert_rows_equal,
    fake_base_model,
    fake_base_sae,
    golden_extractor,
    make_meta,
)
from tests.axbench_base.stage1_protocol_helpers import TEST_STAGE1_TEMPLATE
from tests.axbench_base.unit.test_config import make_valid_config

_GOLDEN_PATH = (
    Path(__file__).parent.parent / "golden" / "stage1c_avoiding_golden.parquet"
)
_AVOID_TEMPLATE = "MODE=AVOID\n" + TEST_STAGE1_TEMPLATE


def _prefix_row(
    prefix_text: str,
    doc: int,
    continuations: list[str],
    percentiles: list[float],
) -> dict[str, Any]:
    return {
        "concept_id": 0,
        "prefix_text": prefix_text,
        "prefix_source_shard": 0,
        "prefix_source_doc": doc,
        "prefix_activating_seg_idx": 2,
        "verification_continuations": list(continuations),
        "verification_seeds": list(range(100, 100 + len(continuations))),
        "verification_percentiles": list(percentiles),
    }


def _concept() -> Concept:
    return Concept(
        concept_id=0, feature_index=FC, layer=8,
        sae_source="8-gemmascope-res-16k", description="alpha concept",
        corpus_count=100, description_source="neuronpedia",
    )


def _common_stubs(monkeypatch) -> None:
    tokenizer = GoldenExtractionTokenizer()
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", golden_extractor)
    monkeypatch.setattr(
        negative_mode, "load_template",
        lambda name: _AVOID_TEMPLATE,
    )
    monkeypatch.setattr(
        negative_mode,
        "get_extraction_tokenizer",
        lambda tokenizer_id: tokenizer,
    )
    monkeypatch.setattr(negative_mode, "get_tokenizer", lambda: StubNanochatTokenizer())


def _cfg(tmp_path: Path, **overrides: Any):
    params: dict[str, Any] = {
        "run_tag": "golden_518",
        "output_root": str(tmp_path / "eval_out"),
        "max_retries": 1,
        "qc_fallback_on_exhaust": "error",
        "output_length": 16,
        "extraction_threshold": 0.0,
    }
    params.update(overrides)
    return make_valid_config(**params)


def _run_golden_scenario(tmp_path: Path, monkeypatch) -> list[dict[str, Any]]:
    """Drive _synthesize_concept_negatives over a NO_TOKENS-free scenario; return rows."""
    cfg = _cfg(tmp_path)
    concept = _concept()
    prefixes = [
        # avoid attempt-1 INACTIVE -> accept (qc_attempts == 1)
        _prefix_row("prefix one", doc=0, continuations=["rich a", "rich b"],
                    percentiles=[80.0, 85.0]),
        # avoid attempt-1 ACTIVE (FIRE) -> reject -> attempt-2 INACTIVE -> accept (qc_attempts == 2)
        _prefix_row("prefix two", doc=1, continuations=["rich c"], percentiles=[90.0]),
    ]
    client = PromptKeyedSynthClient({
        "MODE=AVOID\nPREFIX=prefix one": ["quiet field"],
        "MODE=AVOID\nPREFIX=prefix two": ["big FIRE", "quiet pond"],
    })
    _common_stubs(monkeypatch)
    return _synthesize_concept_negatives(
        cfg, concept, prefixes, base_model=fake_base_model(),
        base_sae=fake_base_sae(), llm_client=client, target_model_meta=make_meta(),
    )


def _golden_schema():
    return schema_with_extraction_tokenizer_metadata(
        NEG_TRAINING_SCHEMA, "google/gemma-2-2b",
    )


def test_stage1c_avoiding_golden(tmp_path, monkeypatch):
    rows = _run_golden_scenario(tmp_path, monkeypatch)
    if os.environ.get("AXBENCH_CAPTURE_GOLDEN") == "1":
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=_golden_schema())
        pq.write_table(table, _GOLDEN_PATH)
        pytest.skip(f"captured golden -> {_GOLDEN_PATH}")
    assert _GOLDEN_PATH.exists(), "golden missing; run with AXBENCH_CAPTURE_GOLDEN=1 first"
    assert_rows_equal(rows, _GOLDEN_PATH, _golden_schema())


def _run_rejection_scenario(
    tmp_path: Path, monkeypatch, avoid_response: str,
) -> list[dict[str, Any]]:
    """Single prefix whose avoid continuation is degenerate; warn fallback -> skip."""
    cfg = _cfg(tmp_path, max_retries=0, qc_fallback_on_exhaust="warn")
    concept = _concept()
    prefixes = [
        _prefix_row("prefix three", doc=2, continuations=["rich x"], percentiles=[88.0]),
    ]
    client = PromptKeyedSynthClient({
        "MODE=AVOID\nPREFIX=prefix three": [avoid_response],
    })
    _common_stubs(monkeypatch)
    return _synthesize_concept_negatives(
        cfg, concept, prefixes, base_model=fake_base_model(),
        base_sae=fake_base_sae(), llm_client=client, target_model_meta=make_meta(),
    )


def test_stage1c_seam_merge_no_tokens_rejected(tmp_path, monkeypatch):
    # The protocol parser accepts "MERGE" as one standalone continuation token.
    # QC then joins "prefix three" + "MERGE" without a space, yielding
    # ["prefix", "threeMERGE"]: the full-text token count equals the prefix
    # count, so classify_avoid_continuation returns NO_TOKENS. The row is
    # rejected gracefully and the unpaired rich-only concept is dropped.
    rows = _run_rejection_scenario(tmp_path, monkeypatch, avoid_response="MERGE")
    assert all(r["content_class"] != "concept_avoiding" for r in rows), (
        "NO_TOKENS continuation must NOT be emitted as a concept_avoiding row"
    )
    assert rows == [], "rich-only concept must be dropped to absent (#508)"


def test_stage1c_empty_continuation_rejected(tmp_path, monkeypatch):
    # Whitespace-only continuation -> .strip() empty guard rejects (post-fix).
    # Pre-fix: fc_active("prefix three", "   ") -> NO_TOKENS region -> False -> accept (BUG).
    # Task 3 (#508): the unpaired rich-only concept is then dropped to absent.
    rows = _run_rejection_scenario(tmp_path, monkeypatch, avoid_response="   ")
    assert all(r["content_class"] != "concept_avoiding" for r in rows), (
        "empty/whitespace continuation must NOT be emitted as a concept_avoiding row"
    )
    assert rows == [], "rich-only concept must be dropped to absent (#508)"
