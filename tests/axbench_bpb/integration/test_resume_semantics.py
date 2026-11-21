"""Crash-at-flush resume with a real compacted live-``ours`` intervention."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch as t

from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb import bpb_scoring as bpb_scoring_module
from steering.eval.axbench_bpb.bpb_scoring import run_bpb_score_stage
from steering.eval.axbench_bpb.live_ours_vocab_contract import (
    LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME,
)
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.tokenization import encode_output
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.axbench_bpb.functional.test_live_ours_compacted_model import (
    N_SAE,
    _concept,
    _inputs,
    _meta,
    _model,
    _source_paths,
)
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _rows(scores_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(scores_dir.rglob("*.parquet")):
        rows.extend(pq.read_table(shard).to_pylist())
    return sorted(
        rows,
        key=lambda row: (
            row["method_name"],
            row["concept_id"],
            row["sample_id"],
            row["coefficient_str"],
            row["target"],
        ),
    )


def test_resume_after_crash_produces_identical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash after a durable batch, resume, and equal a clean one-shot run."""
    model = _model()
    tokenizer = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=16,
        target_name="encdec",
    )
    method = get_method_by_name("ours_extract_and_add_encdec")
    concept = _concept()
    base_input = _inputs(tokenizer, mode=method.mode)
    samples = []
    for index, output_text in enumerate(("de", "fg", "hi", "jk")):
        output_tokens = encode_output(output_text, tokenizer)
        samples.append(
            replace(
                base_input,
                sample_id=f"resume-sample-{index}",
                output_tokens=output_tokens,
                n_output=len(output_tokens),
                output_byte_weights=t.ones(
                    len(output_tokens),
                    dtype=t.int64,
                ),
            )
        )

    source_paths = _source_paths(tmp_path / "source")
    stats_path = write_steering_stats(
        tmp_path / "stats.safetensors",
        n_sae_features=N_SAE,
        max_mags={concept.feature_index: 1.0},
    )
    cfg = SimpleNamespace(
        feature_statistics_path=str(stats_path),
        log_progress_every=1000,
    )

    def kwargs(run_tag: str) -> dict[str, Any]:
        return {
            "cfg": cfg,
            "output_paths": OutputPaths(
                output_root=tmp_path / "outputs",
                run_tag=run_tag,
            ),
            "source_paths": source_paths,
            "methods": [method],
            "concepts": [concept],
            "samples_by_concept_mode": {(concept.concept_id, method.mode): samples},
            "coefficient_grids": {method.name: [2]},
            "decoder_only_bundle": bundle,
            "encdec_bundle": bundle,
            "target_model_metas": {
                "decoder_only": _meta(),
                "encdec": _meta(),
            },
            "tokenizer_nanochat": tokenizer,
            "device": t.device("cpu"),
            "ddp_rank": 0,
            "ddp_world_size": 1,
            "diffmean_layer": 8,
            "batch_flush_size": 2,
        }

    real_writer = bpb_scoring_module.write_bpb_score_rows
    flushes = 0

    def crash_after_first_flush(rows, path):
        nonlocal flushes
        real_writer(rows, path)
        flushes += 1
        if flushes == 1:
            raise RuntimeError("deterministic crash after durable batch")

    monkeypatch.setattr(
        bpb_scoring_module,
        "write_bpb_score_rows",
        crash_after_first_flush,
    )
    with pytest.raises(RuntimeError, match="after durable batch"):
        run_bpb_score_stage(**kwargs("resume"))

    resume_paths = kwargs("resume")["output_paths"]
    assert isinstance(resume_paths, OutputPaths)
    marker = resume_paths.bpb_scores_dir() / LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME
    assert marker.is_file()
    assert len(list(resume_paths.bpb_scores_dir().rglob("*.parquet"))) == 1

    monkeypatch.setattr(
        bpb_scoring_module,
        "write_bpb_score_rows",
        real_writer,
    )
    run_bpb_score_stage(**kwargs("resume"))
    run_bpb_score_stage(**kwargs("clean"))

    assert _rows(resume_paths.bpb_scores_dir()) == _rows(
        kwargs("clean")["output_paths"].bpb_scores_dir()
    )
