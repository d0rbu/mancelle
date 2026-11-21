"""Committed-golden exact-equality harness for Stage 1b positive-avoid output.

The committed golden parquet anchors the **pre-#513** Stage 1b avoid output.
It was captured at commit ``3c789b0e`` (BEFORE ``classify_avoid_continuation``
existed) by driving ``_per_concept_positives`` through the then-current avoid
path, mocking ONLY at the extraction-tokenizer + batched-extractor boundary
(``qc.get_extraction_tokenizer`` / ``qc._run_batched_extractor``). That boundary
is stable across the #513 tokenize-once refactor, so post-#513
``classify_avoid_continuation`` reproduces byte-identical rows from the same
scenario. That pre-vs-post equivalence — current output matching a baseline
captured before the refactor — is precisely this test's value.

WARNING: re-running with ``AXBENCH_CAPTURE_GOLDEN=1`` on the CURRENT (post-#513)
code RE-BASELINES the golden to current behavior and DESTROYS the pre-change
equivalence anchor — the verify run would then trivially compare current output
to itself, proving nothing about the refactor. Re-capture ONLY for an
intentional new baseline (e.g. a deliberate schema change), and NEVER as a
"fix" for a failing diff (a failing diff means behavior drifted — investigate
the drift, do not paper over it by re-baselining).

Capture (run ONCE on the pre-change code, then commit the parquet; see WARNING):

    AXBENCH_CAPTURE_GOLDEN=1 uv run pytest \
      tests/axbench_base/integration/test_stage1b_avoid_golden.py -q

Verify (every normal run):

    uv run pytest tests/axbench_base/integration/test_stage1b_avoid_golden.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.synth import positive_mode, qc
from steering.eval.axbench_base.tokenizer_utils import tokenize_labeled_pair
from steering.eval.axbench_base.types import Concept, SeedRow
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

_GOLDEN_PATH = Path(__file__).parent.parent / "golden" / "stage1b_avoid_golden.parquet"
_RICH_TEMPLATE = "MODE=RICH\n" + TEST_STAGE1_TEMPLATE
_AVOID_TEMPLATE = "MODE=AVOID\n" + TEST_STAGE1_TEMPLATE


def _seed(idx: int, text: str, continuation_text: str) -> SeedRow:
    return SeedRow(
        shard_idx=0,
        doc_idx=idx,
        source_nanochat_token_count=6,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text=text,
        prefix_content_nanochat_tokens=[1, 2, 3],
        sae_feature_ids=[],  # clean -> no dirty-prefix scoring
        sae_feature_mags=[],
        continuation_text=continuation_text,
        continuation_content_nanochat_tokens=[10, 20, 30],
    )


def _run_scenario(tmp_path: Path, monkeypatch) -> list[dict[str, Any]]:
    """Run the unified production engine through the real avoid path."""
    cfg = make_valid_config(
        run_tag="golden_513",
        output_root=str(tmp_path / "eval_out"),
        n_positives_per_concept=3,
        accept_threshold=0.5,
        max_retries=1,
        qc_fallback_on_exhaust="error",
        output_length=16,
        extraction_threshold=0.0,
        min_prefix_chars=1,
    )
    concept = Concept(
        concept_id=0, feature_index=FC, layer=8,
        sae_source="8-gemmascope-res-16k", description="alpha concept",
        corpus_count=100, description_source="neuronpedia",
    )
    seeds = [
        # natural INACTIVE -> accept natural (corpus gen_source)
        _seed(0, text="prefix one", continuation_text="calm river"),
        # natural ACTIVE (FIRE) -> reject -> LLM avoid INACTIVE -> accept
        _seed(1, text="prefix two", continuation_text="big FIRE"),
        # natural NO_TOKENS (MERGE) -> LLM avoid INACTIVE -> accept
        _seed(2, text="prefix three", continuation_text="MERGE"),
    ]

    tokenizer = GoldenExtractionTokenizer()
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", golden_extractor)

    # Rich-row QC is unrelated to issue 513: preserve the golden's historical
    # seed/output pair (10, 80) across the now-separate canonical scorers.
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentile_for_continuation",
        lambda prefix, cont, f_c, gemma, sae, table, *, sae_layer,
        tokenizer_id: 80.0,
    )
    monkeypatch.setattr(
        positive_mode,
        "output_fc_percentiles_for_continuations",
        lambda inputs, gemma, sae, table, *, sae_layer, tokenizer_id,
        release_memory=True: [80.0] * len(inputs),
    )
    monkeypatch.setattr(
        positive_mode, "fc_percentile_on_prefix",
        lambda prefix, f_c, gemma, sae, table, *, sae_layer, tokenizer_id: 10.0,
    )
    monkeypatch.setattr(
        positive_mode, "load_template",
        lambda name: (_RICH_TEMPLATE
                      if "with_concept" in name
                      else _AVOID_TEMPLATE),
    )
    monkeypatch.setattr(
        positive_mode,
        "get_extraction_tokenizer",
        lambda tokenizer_id: tokenizer,
    )
    monkeypatch.setattr(positive_mode, "get_tokenizer", lambda: StubNanochatTokenizer())

    client = PromptKeyedSynthClient({
        "MODE=RICH\nPREFIX=prefix one": "rich one",
        "MODE=RICH\nPREFIX=prefix two": "rich two",
        "MODE=RICH\nPREFIX=prefix three": "rich three",
        "MODE=AVOID\nPREFIX=prefix two": "quiet field",  # no FIRE -> accept
        "MODE=AVOID\nPREFIX=prefix three": "still pond",  # no FIRE -> accept
    })

    base_model = fake_base_model()
    base_sae = fake_base_sae()
    target_model_meta = make_meta()
    effective_prefixes = positive_mode._resolve_stage1b_prefixes(cfg, seeds)
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        base_model,
        base_sae,
        None,
        sae_layer=target_model_meta.sae_layer,
        prefix_cache=None,
        cluster_target=None,
        effective_prefixes=effective_prefixes,
    )
    return positive_mode._run_positive_target_refill_engine(
        cfg,
        [tracker],
        base_model,
        base_sae,
        client,
        None,
        target_model_meta,
    )[concept.concept_id]


def _golden_schema():
    return positive_mode.schema_with_extraction_tokenizer_metadata(
        positive_mode.TRAINING_SCHEMA, "google/gemma-2-2b",
    )


def test_stage1b_avoid_golden(tmp_path, monkeypatch):
    rows = _run_scenario(tmp_path, monkeypatch)
    if os.environ.get("AXBENCH_CAPTURE_GOLDEN") == "1":
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        table = positive_mode.pa.Table.from_pylist(rows, schema=_golden_schema())
        pq.write_table(table, _GOLDEN_PATH)
        pytest.skip(f"captured golden -> {_GOLDEN_PATH}")
    assert _GOLDEN_PATH.exists(), "golden missing; run with AXBENCH_CAPTURE_GOLDEN=1 first"
    tokenizer = StubNanochatTokenizer()

    def _issue_728_boundary(row: dict[str, Any]) -> dict[str, Any]:
        # The committed golden predates the exact combined-token boundary.
        # Preserve it as the oracle for every other field while replacing the
        # one intentionally versioned field with the Issue #728 contract.
        _tokens, output_start = tokenize_labeled_pair(
            tokenizer, row["prefix_text"], row["output_text"]
        )
        row["n_prefix_nanochat_tokens"] = output_start
        return row

    assert_rows_equal(
        rows,
        _GOLDEN_PATH,
        _golden_schema(),
        golden_row_transform=_issue_728_boundary,
    )
