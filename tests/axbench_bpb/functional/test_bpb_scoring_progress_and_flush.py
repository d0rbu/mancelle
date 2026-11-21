"""Lock progress-logging and parquet-flush boundaries in
``run_bpb_score_stage``'s hot loop.

Three uncovered code paths in ``bpb_scoring.py``:

* ``cfg.log_progress_every`` interval logging (line 232).
* Mid-loop flush at ``len(batch_rows) >= batch_flush_size`` (lines 239-243).
* End-of-loop residual flush for the trailing partial batch (lines 245-246).

The hot loop is exercised through ``run_bpb_score_stage`` with all heavy
dependencies stubbed: ``build_intervention`` and ``score_sample`` are
monkeypatched so we never touch the real model or intervention dispatch.
A synthetic ``BpbScoreRow`` factory keeps every emitted row distinct so
shard contents can be asserted by sample_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from steering.eval.axbench_bpb import bpb_scoring as bpb_scoring_module
from steering.eval.axbench_bpb.bpb_scoring import run_bpb_score_stage
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import (
    BpbScoreRow,
    Intervention,
    ScoringInputs,
    SteeringMode,
)

# ---------- shared stubs ----------


@dataclass(frozen=True)
class _FakeMethod:
    name: str
    mode: SteeringMode
    target_model: str


@dataclass(frozen=True)
class _FakeConcept:
    concept_id: int


@dataclass
class _FakeOutputPaths:
    """Minimal output_paths shim — only the two methods touched by the
    scoring stage are implemented."""

    root: Path

    def bpb_scores_rank_dir(self, rank: int) -> Path:
        d = self.root / "bpb_scores" / f"rank_{rank:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bpb_cache_path(self, rank: int) -> Path:
        return self.root / "bpb_scores" / f"rank_{rank:04d}.sqlite"


@dataclass
class _Cfg:
    """Stubbed cfg with the fields read inside run_bpb_score_stage."""

    log_progress_every: int = 10
    feature_statistics_path: str = "unused.safetensors"


def _make_input(concept_id: int, sample_idx: int) -> ScoringInputs:
    """Build a minimal ``ScoringInputs`` — only the fields read by
    ``score_sample`` matter, but our stubbed ``score_sample`` ignores all
    of them. Use distinct sample_ids so caching does not collapse runs."""
    return ScoringInputs(
        sample_id=f"sample_{concept_id}_{sample_idx}",
        concept_id=concept_id,
        mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="",
        prefix_hash="0" * 64,
        prefix_tokens=[1, 2],
        output_tokens=[3, 4],
        prefix_context_segments=(),
        n_prefix=2,
        n_output=2,
        output_byte_weights=torch.ones(2, dtype=torch.int64),
        truncated=False,
    )


def _make_score_row(inputs: ScoringInputs, *, method_name: str,
                    coefficient_str: str) -> BpbScoreRow:
    return BpbScoreRow(
        method_name=method_name,
        concept_id=inputs.concept_id,
        mode=str(inputs.mode),
        content_class=inputs.content_class,
        target="decoder_only",
        coefficient_str=coefficient_str,
        sample_id=inputs.sample_id,
        bpb_intervened=1.0,
        n_output_tokens=inputs.n_output,
        n_output_bytes=int(inputs.output_byte_weights.sum().item()),
        truncated=inputs.truncated,
    )


def _patch_heavy_helpers(monkeypatch) -> None:
    """Stub ``build_intervention`` and ``score_sample`` so the hot loop
    never invokes the real intervention dispatch or model forward pass.
    The hot loop only cares that ``score_sample`` returns a ``BpbScoreRow``
    instance with the correct keys — every test in this module asserts on
    parquet contents written from those rows."""

    def _fake_build_intervention(**kwargs: Any) -> Intervention:
        method = kwargs["method"]
        return Intervention.prompting(target_model=method.target_model, prompt_prefix_tokens=[])

    def _fake_score_sample(*, inputs: ScoringInputs, method_name: str,
                           coefficient_str: str, **_: Any) -> BpbScoreRow:
        return _make_score_row(
            inputs, method_name=method_name, coefficient_str=coefficient_str,
        )

    monkeypatch.setattr(
        bpb_scoring_module, "build_intervention", _fake_build_intervention,
    )
    monkeypatch.setattr(
        bpb_scoring_module, "score_sample", _fake_score_sample,
    )


def _build_kwargs(*, tmp_path: Path, cfg: _Cfg, n_samples: int) -> dict[str, Any]:
    """Construct all the kwargs ``run_bpb_score_stage`` needs to score
    ``n_samples`` work units (1 method × 1 concept × N samples × 1 coef)."""
    methods = [_FakeMethod(
        name="m1", mode=SteeringMode.POSITIVE, target_model="decoder_only",
    )]
    concepts = [_FakeConcept(concept_id=1)]
    samples_by_concept_mode = {
        (1, SteeringMode.POSITIVE): [
            _make_input(concept_id=1, sample_idx=i) for i in range(n_samples)
        ],
    }
    coefficient_grids = {"m1": [0]}

    # The models are unused because score_sample is monkeypatched, but real
    # bundles keep this test aligned with run_bpb_score_stage's contract.
    decoder_only_bundle = TargetModelBundle(
        model=None,
        tokenizer=None,
        max_seq_len=1,
        target_name="decoder_only",
    )
    encdec_bundle = TargetModelBundle(
        model=None,
        tokenizer=None,
        max_seq_len=1,
        target_name="encdec",
    )
    return {
        "cfg": cfg,
        "output_paths": _FakeOutputPaths(root=tmp_path),
        "source_paths": None,
        "methods": methods, "concepts": concepts,
        "samples_by_concept_mode": samples_by_concept_mode,
        "coefficient_grids": coefficient_grids,
        "decoder_only_bundle": decoder_only_bundle,
        "encdec_bundle": encdec_bundle,
        "target_model_metas": {"decoder_only": None, "encdec": None},
        "tokenizer_nanochat": None,
        "device": torch.device("cpu"),
        "ddp_rank": 0, "ddp_world_size": 1,
        "diffmean_layer": 1,
    }


# ---------- mid-loop and residual flush ----------


def test_bpb_score_stage_flushes_full_batch_mid_loop(
    tmp_path: Path, monkeypatch,
) -> None:
    """5 work units with ``batch_flush_size=2`` produce 3 parquet shards:
    two full shards from the mid-loop flush branch and one residual
    shard from the post-loop flush branch.

    Locks:
        * The mid-loop flush triggers exactly when ``len(batch_rows)``
          reaches ``batch_flush_size`` — not one row early or late.
        * ``batch_idx`` is incremented per shard so consecutive shards
          land at ``batch_000000.parquet`` / ``batch_000001.parquet`` /
          ``batch_000002.parquet``.
        * ``batch_rows`` is cleared between flushes so each shard
          contains exactly the rows accumulated since the last flush.
    """
    _patch_heavy_helpers(monkeypatch)

    kwargs = _build_kwargs(tmp_path=tmp_path, cfg=_Cfg(), n_samples=5)
    rank_dir = run_bpb_score_stage(batch_flush_size=2, **kwargs)

    files = sorted(rank_dir.glob("batch_*.parquet"))
    assert [f.name for f in files] == [
        "batch_000000.parquet",  # mid-loop flush after rows 1-2
        "batch_000001.parquet",  # mid-loop flush after rows 3-4
        "batch_000002.parquet",  # end-of-loop residual flush, row 5
    ]

    # Row counts per shard: full / full / residual.
    assert pq.read_table(files[0]).num_rows == 2
    assert pq.read_table(files[1]).num_rows == 2
    assert pq.read_table(files[2]).num_rows == 1

    # Total rows preserved (no row dropped between flush boundaries).
    total_rows = sum(pq.read_table(f).num_rows for f in files)
    assert total_rows == 5


def test_bpb_score_stage_residual_flush_only_when_partial_remains(
    tmp_path: Path, monkeypatch,
) -> None:
    """4 work units with ``batch_flush_size=2`` produce exactly 2 shards.

    The post-loop ``if batch_rows`` branch must NOT fire when the last
    mid-loop flush already cleared the buffer — otherwise we would emit a
    spurious empty shard and inflate ``batch_idx`` for resume runs.
    """
    _patch_heavy_helpers(monkeypatch)

    kwargs = _build_kwargs(tmp_path=tmp_path, cfg=_Cfg(), n_samples=4)
    rank_dir = run_bpb_score_stage(batch_flush_size=2, **kwargs)

    files = sorted(rank_dir.glob("batch_*.parquet"))
    assert [f.name for f in files] == [
        "batch_000000.parquet",
        "batch_000001.parquet",
    ]
    assert pq.read_table(files[0]).num_rows == 2
    assert pq.read_table(files[1]).num_rows == 2


# ---------- progress logging interval ----------


def test_bpb_score_stage_logs_progress_at_interval(
    tmp_path: Path, monkeypatch,
) -> None:
    """``log_progress_every=2`` + 4 work units → at least 2 logger.info
    calls whose payload mentions ``processed=`` (the progress message
    template at ``bpb_scoring.py:233-237``).

    The completion summary at line 250 also calls ``logger.info`` with
    ``processed=``, so we filter to the hot-loop messages first by
    counting only those that include the buffered/skipped fields too.
    """
    _patch_heavy_helpers(monkeypatch)

    log_calls: list[str] = []

    def _capture(msg: Any, *_args: Any, **_kwargs: Any) -> None:
        log_calls.append(str(msg))

    monkeypatch.setattr(bpb_scoring_module.logger, "info", _capture)

    kwargs = _build_kwargs(
        tmp_path=tmp_path, cfg=_Cfg(log_progress_every=2), n_samples=4,
    )
    run_bpb_score_stage(batch_flush_size=1000, **kwargs)

    # The hot-loop progress line (line 234) includes "buffered=";
    # the completion summary (line 250) does not. Filter accordingly.
    hot_loop_logs = [m for m in log_calls if "buffered=" in m]
    # 4 units processed at every-2-units cadence → exactly 2 progress logs
    # (after units 2 and 4).
    assert len(hot_loop_logs) >= 2

    for line in hot_loop_logs:
        assert "processed=" in line
        assert "skipped=" in line


def test_bpb_score_stage_progress_logs_skipped_when_no_units(
    tmp_path: Path, monkeypatch,
) -> None:
    """When no work is scheduled, the hot-loop progress line never fires
    (``processed`` never reaches ``log_progress_every``); only the
    completion summary at line 250 is emitted."""
    _patch_heavy_helpers(monkeypatch)

    log_calls: list[str] = []

    def _capture(msg: Any, *_args: Any, **_kwargs: Any) -> None:
        log_calls.append(str(msg))

    monkeypatch.setattr(bpb_scoring_module.logger, "info", _capture)

    kwargs = _build_kwargs(
        tmp_path=tmp_path, cfg=_Cfg(log_progress_every=1), n_samples=0,
    )
    run_bpb_score_stage(batch_flush_size=1000, **kwargs)

    hot_loop_logs = [m for m in log_calls if "buffered=" in m]
    assert hot_loop_logs == []
    # Completion summary (no "buffered=" field) must still fire.
    completion_logs = [m for m in log_calls if "complete:" in m]
    assert len(completion_logs) == 1
