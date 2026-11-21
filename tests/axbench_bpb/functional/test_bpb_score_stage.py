"""Functional tests for run_bpb_score_stage orchestration.

Uses tiny_decoder_only + tiny_encdec fixtures + monkeypatched
build_intervention to exercise the scoring loop without requiring real
concept_features / direction safetensors on disk.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.eval import axbench_bpb_eval as axbench_bpb_eval_module
from scripts.eval.axbench_bpb_eval import (
    _require_complete_scoring_population,
    _stage_bpb_aggregate,
)
from steering.eval.axbench_base.direction_scale import (
    STEERING_SCALE_MARKER_NAME,
    direction_scale_identity,
    encode_max_activation,
)
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb import bpb_scoring as bpb_scoring_module
from steering.eval.axbench_bpb.bpb_scoring import (
    _coefficient_from_str,
    _next_batch_index,
    run_bpb_score_stage,
)
from steering.eval.axbench_bpb.live_ours_vocab_contract import (
    LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME,
)
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    BpbScoreRow,
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from tests.cross_tokenizer_helpers import SimpleTokenizer


@dataclass(frozen=True)
class _FakeMethod:
    name: str
    baseline: str  # "ours" | "diffmean" | ... (not dispatched to in these tests — we monkeypatch build_intervention)
    mode: SteeringMode
    target_model: str


@dataclass(frozen=True)
class _FakeConcept:
    concept_id: int


@dataclass
class _FakeOutputPaths:
    root: Path

    def bpb_scores_dir(self) -> Path:
        return self.root / "bpb_scores"

    def bpb_scores_rank_dir(self, rank: int) -> Path:
        d = self.root / "bpb_scores" / f"rank_{rank:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bpb_cache_path(self, rank: int) -> Path:
        return self.root / "bpb_scores" / f"rank_{rank:04d}.sqlite"


@dataclass
class _Cfg:
    log_progress_every: int = 10
    feature_statistics_path: str = "unused.safetensors"


def _make_scoring_input(tok, prefix, output, concept_id, mode):
    prefix_tokens = encode_prefix(prefix, tok)
    output_tokens = encode_output(output, tok)
    byte_weights = torch.ones(len(output_tokens), dtype=torch.int64)
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=concept_id,
            mode=mode,
            content_class="concept_rich",
            prefix_text=prefix,
            output_text=output,
        ),
        concept_id=concept_id,
        mode=mode,
        content_class="concept_rich",
        prefix_text=prefix,
        prefix_hash=make_prefix_hash(prefix),
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=(),
        n_prefix=len(prefix_tokens),
        n_output=len(output_tokens),
        output_byte_weights=byte_weights,
        truncated=False,
    )


def test_run_bpb_score_stage_end_to_end(
    tmp_path,
    tiny_decoder_only,
    monkeypatch,
):
    """Single rank, 1 method × 2 concepts × 2 samples × 2 coefficients = 8 units."""

    # Monkeypatch build_intervention to return Intervention.prompting for all kinds
    # (we're exercising the orchestration, not the intervention logic).
    def _fake_build_intervention(**kwargs):
        method = kwargs["method"]
        return Intervention.prompting(target_model=method.target_model, prompt_prefix_tokens=[])

    monkeypatch.setattr(
        bpb_scoring_module,
        "build_intervention",
        _fake_build_intervention,
    )

    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )

    methods = [
        _FakeMethod(
            name="m1",
            baseline="ours",
            mode=SteeringMode.POSITIVE,
            target_model="decoder_only",
        )
    ]
    concepts = [_FakeConcept(concept_id=1), _FakeConcept(concept_id=2)]
    samples_by_concept_mode = {
        (1, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "abc", "x", 1, SteeringMode.POSITIVE),
            _make_scoring_input(tok, "def", "y", 1, SteeringMode.POSITIVE),
        ],
        (2, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "ghi", "z", 2, SteeringMode.POSITIVE),
            _make_scoring_input(tok, "jkl", "w", 2, SteeringMode.POSITIVE),
        ],
    }
    coefficient_grids = {"m1": [0, 1]}

    rank_dir = run_bpb_score_stage(
        cfg=_Cfg(),
        output_paths=_FakeOutputPaths(root=tmp_path),
        source_paths=None,  # not used because build_intervention is stubbed
        methods=methods,
        concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
        decoder_only_bundle=bundle,
        encdec_bundle=bundle,  # unused for this decoder_only method
        target_model_metas={"decoder_only": None, "encdec": None},
        tokenizer_nanochat=tok,
        device=torch.device("cpu"),
        ddp_rank=0,
        ddp_world_size=1,
        diffmean_layer=1,
    )

    # 1 method × 2 concepts × 2 samples × 2 coefficients = 8 rows
    parquets = list(rank_dir.glob("batch_*.parquet"))
    assert len(parquets) >= 1
    all_rows = []
    for p in sorted(parquets):
        all_rows.extend(pq.read_table(p).to_pylist())
    assert len(all_rows) == 8
    # All rows should be decoder_only target, method m1
    assert all(r["target"] == "decoder_only" for r in all_rows)
    assert all(r["method_name"] == "m1" for r in all_rows)


def test_run_bpb_score_stage_does_not_skip_feature_diffmean_work_units(
    tmp_path,
    tiny_decoder_only,
    monkeypatch,
):
    """Feature-DiffMean work units are not silently availability-filtered."""

    build_calls: list[tuple[str, int]] = []

    def _fake_build_intervention(**kwargs):
        method = kwargs["method"]
        concept = kwargs["concept"]
        build_calls.append((method.name, concept.concept_id))
        return Intervention.prompting(
            target_model=method.target_model,
            prompt_prefix_tokens=[],
        )

    def _fake_score_sample(
        *,
        inputs,
        target_bundle,
        intervention,
        device,
        coefficient_str,
        method_name,
        lora,
    ) -> BpbScoreRow:
        return BpbScoreRow(
            method_name=method_name,
            concept_id=inputs.concept_id,
            mode=inputs.mode.value,
            content_class=inputs.content_class,
            target=target_bundle.target_name,
            coefficient_str=coefficient_str,
            sample_id=inputs.sample_id,
            bpb_intervened=1.0,
            n_output_tokens=inputs.n_output,
            n_output_bytes=int(inputs.output_byte_weights.sum().item()),
            truncated=False,
        )

    monkeypatch.setattr(
        bpb_scoring_module,
        "build_intervention",
        _fake_build_intervention,
    )
    monkeypatch.setattr(
        bpb_scoring_module,
        "score_sample",
        _fake_score_sample,
    )

    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="encdec",
    )

    methods = [
        _FakeMethod(
            name="feature_diffmean_encdec",
            baseline="ours",
            mode=SteeringMode.POSITIVE,
            target_model="encdec",
        ),
        _FakeMethod(
            name="regular_ours_encdec",
            baseline="ours",
            mode=SteeringMode.POSITIVE,
            target_model="encdec",
        ),
    ]
    concepts = [_FakeConcept(concept_id=1), _FakeConcept(concept_id=2)]
    samples_by_concept_mode = {
        (1, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "abc", "x", 1, SteeringMode.POSITIVE),
        ],
        (2, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "def", "y", 2, SteeringMode.POSITIVE),
        ],
    }

    rank_dir = run_bpb_score_stage(
        cfg=_Cfg(),
        output_paths=_FakeOutputPaths(root=tmp_path),
        source_paths=None,
        methods=methods,
        concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids={
            "feature_diffmean_encdec": [None],
            "regular_ours_encdec": [None],
        },
        decoder_only_bundle=bundle,
        encdec_bundle=bundle,
        target_model_metas={"decoder_only": None, "encdec": None},
        tokenizer_nanochat=tok,
        device=torch.device("cpu"),
        ddp_rank=0,
        ddp_world_size=1,
        diffmean_layer=1,
    )

    all_rows = []
    for p in sorted(rank_dir.glob("batch_*.parquet")):
        all_rows.extend(pq.read_table(p).to_pylist())

    scored = {(row["method_name"], row["concept_id"], row["coefficient_str"]) for row in all_rows}
    assert scored == {
        ("feature_diffmean_encdec", 1, "None"),
        ("feature_diffmean_encdec", 2, "None"),
        ("regular_ours_encdec", 1, "None"),
        ("regular_ours_encdec", 2, "None"),
    }
    assert ("feature_diffmean_encdec", 2) in build_calls
    assert ("regular_ours_encdec", 2) in build_calls


def test_run_bpb_score_stage_skips_cached_work(
    tmp_path,
    tiny_decoder_only,
    monkeypatch,
):
    """A second invocation with the same inputs should do no work."""

    def _fake_build_intervention(**kwargs):
        method = kwargs["method"]
        return Intervention.prompting(target_model=method.target_model, prompt_prefix_tokens=[])

    monkeypatch.setattr(
        bpb_scoring_module,
        "build_intervention",
        _fake_build_intervention,
    )

    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )

    methods = [
        _FakeMethod(
            name="m1",
            baseline="ours",
            mode=SteeringMode.POSITIVE,
            target_model="decoder_only",
        )
    ]
    concepts = [_FakeConcept(concept_id=1)]
    samples_by_concept_mode = {
        (1, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "abc", "x", 1, SteeringMode.POSITIVE),
        ],
    }
    coefficient_grids = {"m1": [0]}

    cfg = _Cfg()
    output_paths = _FakeOutputPaths(root=tmp_path)
    rank_dir = run_bpb_score_stage(
        cfg=cfg,
        output_paths=output_paths,
        source_paths=None,
        methods=methods,
        concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
        decoder_only_bundle=bundle,
        encdec_bundle=bundle,
        target_model_metas={"decoder_only": None, "encdec": None},
        tokenizer_nanochat=tok,
        device=torch.device("cpu"),
        ddp_rank=0,
        ddp_world_size=1,
        diffmean_layer=1,
    )
    parquets_first = sorted(rank_dir.glob("batch_*.parquet"))
    assert len(parquets_first) == 1
    mtime_first = parquets_first[0].stat().st_mtime_ns

    # Second run: cache.has() returns True, everything is skipped,
    # no new parquet shards are written (empty batch_rows at end).
    rank_dir_2 = run_bpb_score_stage(
        cfg=cfg,
        output_paths=output_paths,
        source_paths=None,
        methods=methods,
        concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
        decoder_only_bundle=bundle,
        encdec_bundle=bundle,
        target_model_metas={"decoder_only": None, "encdec": None},
        tokenizer_nanochat=tok,
        device=torch.device("cpu"),
        ddp_rank=0,
        ddp_world_size=1,
        diffmean_layer=1,
    )
    assert rank_dir_2 == rank_dir
    parquets_second = sorted(rank_dir_2.glob("batch_*.parquet"))
    assert len(parquets_second) == len(parquets_first)
    # Original shard unchanged
    assert parquets_second[0].stat().st_mtime_ns == mtime_first


def test_run_bpb_score_stage_ddp_partitioning(
    tmp_path,
    tiny_decoder_only,
    monkeypatch,
):
    """4 work units split across world_size=2; each rank scores ~2 of them."""

    def _fake_build_intervention(**kwargs):
        method = kwargs["method"]
        return Intervention.prompting(target_model=method.target_model, prompt_prefix_tokens=[])

    monkeypatch.setattr(
        bpb_scoring_module,
        "build_intervention",
        _fake_build_intervention,
    )

    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )

    methods = [
        _FakeMethod(
            name="m",
            baseline="ours",
            mode=SteeringMode.POSITIVE,
            target_model="decoder_only",
        )
    ]
    concepts = [_FakeConcept(concept_id=i) for i in range(4)]
    samples_by_concept_mode = {
        (i, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, f"p{i}", "x", i, SteeringMode.POSITIVE),
        ]
        for i in range(4)
    }
    coefficient_grids = {"m": [0]}

    cfg = _Cfg()
    output_paths = _FakeOutputPaths(root=tmp_path)

    def _run_rank(rank: int) -> Path:
        return run_bpb_score_stage(
            cfg=cfg,
            output_paths=output_paths,
            source_paths=None,
            methods=methods,
            concepts=concepts,
            samples_by_concept_mode=samples_by_concept_mode,
            coefficient_grids=coefficient_grids,
            decoder_only_bundle=bundle,
            encdec_bundle=bundle,
            target_model_metas={"decoder_only": None, "encdec": None},
            tokenizer_nanochat=tok,
            device=torch.device("cpu"),
            ddp_rank=rank,
            ddp_world_size=2,
            diffmean_layer=1,
        )

    r0 = _run_rank(0)
    r1 = _run_rank(1)

    rows_0 = []
    for p in sorted(r0.glob("batch_*.parquet")):
        rows_0.extend(pq.read_table(p).to_pylist())
    rows_1 = []
    for p in sorted(r1.glob("batch_*.parquet")):
        rows_1.extend(pq.read_table(p).to_pylist())

    # Combined: all 4 concepts covered, no duplicates
    all_concept_ids = {r["concept_id"] for r in rows_0} | {r["concept_id"] for r in rows_1}
    assert all_concept_ids == {0, 1, 2, 3}
    # Disjoint partition
    ids_0 = {(r["concept_id"], r["sample_id"]) for r in rows_0}
    ids_1 = {(r["concept_id"], r["sample_id"]) for r in rows_1}
    assert ids_0.isdisjoint(ids_1)
    assert len(rows_0) + len(rows_1) == 4


def test_coefficient_from_str_handles_all_grids():
    """Verifies round-trip for every coefficient-string shape used by plans."""
    assert _coefficient_from_str("None") is None
    assert _coefficient_from_str("0") == 0
    assert _coefficient_from_str("3") == 3
    assert _coefficient_from_str("-2") == -2
    assert _coefficient_from_str("1.5") == 1.5
    assert _coefficient_from_str("-0.25") == -0.25
    assert _coefficient_from_str("1e-3") == 1e-3


def test_coefficient_from_str_invalid_raises():
    """Garbage input → AxBenchBpbError (not silent string pass-through)."""
    from steering.eval.axbench_bpb.types import AxBenchBpbError

    with pytest.raises(AxBenchBpbError, match="unrecognized coefficient"):
        _coefficient_from_str("not_a_number")


def test_next_batch_index_picks_up_after_existing():
    """_next_batch_index returns len(existing) for consecutive batches."""
    with tempfile.TemporaryDirectory() as tmpd:
        tmpd_path = Path(tmpd)
        # No existing shards → 0
        assert _next_batch_index(tmpd_path) == 0
        # One existing shard → 1
        (tmpd_path / "batch_000000.parquet").write_bytes(b"")
        assert _next_batch_index(tmpd_path) == 1
        # Two existing shards (000003 + 000005) → next is 6
        (tmpd_path / "batch_000003.parquet").write_bytes(b"")
        (tmpd_path / "batch_000005.parquet").write_bytes(b"")
        assert _next_batch_index(tmpd_path) == 6
    # Nonexistent dir → 0
    assert _next_batch_index(Path("/nonexistent/path/does/not/exist")) == 0


def _complete_scoring_samples(
    concept_ids: tuple[int, ...],
) -> dict[tuple[int, SteeringMode], list[SimpleNamespace]]:
    return {
        (concept_id, mode): [
            SimpleNamespace(content_class="concept_rich"),
            SimpleNamespace(content_class="concept_avoiding"),
        ]
        for concept_id in concept_ids
        for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE)
    }


def test_require_complete_scoring_population_accepts_full_cartesian_contract():
    concepts = [SimpleNamespace(concept_id=0), SimpleNamespace(concept_id=7)]
    samples = _complete_scoring_samples((0, 7))

    _require_complete_scoring_population(cast(Any, concepts), samples)


def test_require_complete_scoring_population_rejects_wholly_absent_mode():
    concepts = [SimpleNamespace(concept_id=0)]
    samples = _complete_scoring_samples((0,))
    samples.pop((0, SteeringMode.NEGATIVE))

    with pytest.raises(
        AxBenchBpbError,
        match="negative.*concept_rich.*concept_avoiding",
    ):
        _require_complete_scoring_population(cast(Any, concepts), samples)


# ---------------------------------------------------------------------------
# #588 steering-scale resume guard on the BPB leaderboard (review finding [1])
# ---------------------------------------------------------------------------


def _diffmean_guard_kwargs(tmp_path, bundle, tok):
    # diffmean baseline -> the guard fires; build_intervention is stubbed so the
    # diffmean dispatch itself is not exercised, only validate_all + the guard.
    concepts = [
        SimpleNamespace(concept_id=0, feature_index=0),
        SimpleNamespace(concept_id=1, feature_index=1),
    ]
    methods = [
        _FakeMethod(
            name="dm",
            baseline="diffmean",
            mode=SteeringMode.POSITIVE,
            target_model="decoder_only",
        )
    ]
    samples_by_concept_mode = {
        (0, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "abc", "x", 0, SteeringMode.POSITIVE),
        ],
        (1, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "def", "y", 1, SteeringMode.POSITIVE),
        ],
    }
    baselines_root = tmp_path / "baselines"

    def diffmean_path(target: str, mode: str, concept_id: int) -> Path:
        return (
            baselines_root
            / "diffmean"
            / target
            / mode
            / f"concept_{concept_id:04d}.safetensors"
        )

    def linear_probe_path(target: str, mode: str, concept_id: int) -> Path:
        return (
            baselines_root
            / "linear_probe"
            / target
            / mode
            / f"concept_{concept_id:04d}.safetensors"
        )

    for concept_id, value in ((0, 44.0), (1, 9.0)):
        path = diffmean_path("decoder_only", "positive", concept_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"direction": torch.tensor([1.0, 0.0])},
            str(path),
            metadata={
                "concept_id": str(concept_id),
                "mode": "positive",
                "target_model": "decoder_only",
                "layer": "1",
                "n_pos": "2",
                "n_neg": "2",
                "max_activation": encode_max_activation(value),
            },
        )
    source_paths = SimpleNamespace(
        diffmean_direction=diffmean_path,
        linear_probe_direction=linear_probe_path,
    )
    return {
        "cfg": _Cfg(feature_statistics_path=str(tmp_path / "unused.safetensors")),
        "output_paths": _FakeOutputPaths(root=tmp_path),
        "source_paths": source_paths,
        "methods": methods,
        "concepts": concepts,
        "samples_by_concept_mode": samples_by_concept_mode,
        "coefficient_grids": {"dm": [0, 1]},
        "decoder_only_bundle": bundle,
        "encdec_bundle": bundle,
        "target_model_metas": {"decoder_only": None, "encdec": None},
        "tokenizer_nanochat": tok,
        "device": torch.device("cpu"),
        "ddp_rank": 0,
        "ddp_world_size": 1,
        "diffmean_layer": 1,
    }


def _stub_build_intervention(monkeypatch):
    def _fake(**kwargs):
        method = kwargs["method"]
        concept = kwargs["concept"]
        target = method.target_model
        path_fn = (
            kwargs["source_paths"].diffmean_direction
            if method.baseline == BaselineKind.DIFFMEAN
            else kwargs["source_paths"].linear_probe_direction
        )
        with safe_open(
            str(path_fn(target, method.mode.value, concept.concept_id)),
            framework="pt",
        ) as handle:
            max_activation = float(handle.metadata()["max_activation"])
        coefficient = float(kwargs["coefficient"])
        if method.mode is SteeringMode.NEGATIVE:
            coefficient = -abs(coefficient)
        model_width = kwargs["model"].config.n_embd
        return Intervention.from_direction(
            kind=method.baseline,
            target_model=target,
            direction=torch.zeros(model_width),
            direction_layer=kwargs["diffmean_layer"],
            coefficient=coefficient * max_activation,
        )

    monkeypatch.setattr(bpb_scoring_module, "build_intervention", _fake)


def test_bpb_rejects_direction_regenerated_after_preflight(
    tmp_path, tiny_decoder_only, monkeypatch
):
    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )
    kwargs = _diffmean_guard_kwargs(tmp_path, bundle, tok)
    kwargs["coefficient_grids"] = {"dm": [1]}
    original_build = bpb_scoring_module.build_intervention

    def _regenerate_then_build(**build_kwargs):
        method = build_kwargs["method"]
        concept = build_kwargs["concept"]
        path = build_kwargs["source_paths"].diffmean_direction(
            method.target_model,
            method.mode.value,
            concept.concept_id,
        )
        save_file(
            {"direction": torch.tensor([1.0, 0.0])},
            str(path),
            metadata={
                "concept_id": str(concept.concept_id),
                "mode": method.mode.value,
                "target_model": method.target_model,
                "layer": "1",
                "max_activation": encode_max_activation(123.0),
            },
        )
        return original_build(**build_kwargs)

    monkeypatch.setattr(
        bpb_scoring_module,
        "build_intervention",
        _regenerate_then_build,
    )
    with pytest.raises(
        AxBenchBpbError,
        match="bpb-score direction-scale mid-run regeneration hazard.*artifact key",
    ):
        run_bpb_score_stage(**kwargs)


def test_bpb_resume_guard_fresh_stamps_marker(tmp_path, tiny_decoder_only, monkeypatch):
    _stub_build_intervention(monkeypatch)
    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )
    rank_dir = run_bpb_score_stage(**_diffmean_guard_kwargs(tmp_path, bundle, tok))
    expected = direction_scale_identity(
        {
            ("diffmean", "decoder_only", "positive", 0): 44.0,
            ("diffmean", "decoder_only", "positive", 1): 9.0,
        }
    )
    assert (rank_dir / STEERING_SCALE_MARKER_NAME).read_text() == expected


def test_bpb_resume_guard_mismatch_raises(tmp_path, tiny_decoder_only, monkeypatch):
    _stub_build_intervention(monkeypatch)
    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )
    kwargs = _diffmean_guard_kwargs(tmp_path, bundle, tok)
    rank_dir = run_bpb_score_stage(**kwargs)  # fresh run: stamps marker + shards
    # Simulate a different scale regime: overwrite the marker.
    (rank_dir / STEERING_SCALE_MARKER_NAME).write_text("dirmaxact:staleidentity0")
    with pytest.raises(AxBenchBaseError, match="DIFFERENT steering-scale regime"):
        run_bpb_score_stage(**kwargs)


def test_bpb_resume_guard_old_maxact_marker_raises(
    tmp_path, tiny_decoder_only, monkeypatch
):
    _stub_build_intervention(monkeypatch)
    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )
    kwargs = _diffmean_guard_kwargs(tmp_path, bundle, tok)
    rank_dir = run_bpb_score_stage(**kwargs)
    (rank_dir / STEERING_SCALE_MARKER_NAME).write_text(
        "maxact:0123456789abcdef"
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"pre-#1228 SAE max_act regime.*fresh run_tag",
    ):
        run_bpb_score_stage(**kwargs)


def test_bpb_resume_guard_preguard_shards_without_marker_raise(
    tmp_path, tiny_decoder_only, monkeypatch
):
    _stub_build_intervention(monkeypatch)
    model, cfg_gpt = tiny_decoder_only
    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tok,
        max_seq_len=cfg_gpt.sequence_len,
        target_name="decoder_only",
    )
    kwargs = _diffmean_guard_kwargs(tmp_path, bundle, tok)
    rank_dir = run_bpb_score_stage(**kwargs)  # fresh: stamps marker + shards
    # Simulate pre-#588 shards: remove the marker, keep the durable shards.
    (rank_dir / STEERING_SCALE_MARKER_NAME).unlink()
    assert list(rank_dir.glob("batch_*.parquet"))
    with pytest.raises(AxBenchBaseError, match="predate the issue #1228"):
        run_bpb_score_stage(**kwargs)


def test_require_complete_scoring_population_rejects_one_missing_class():
    concepts = [SimpleNamespace(concept_id=0), SimpleNamespace(concept_id=7)]
    samples = _complete_scoring_samples((0, 7))
    samples[(7, SteeringMode.POSITIVE)] = [SimpleNamespace(content_class="concept_rich")]

    with pytest.raises(
        AxBenchBpbError,
        match=r"positive.*7.*concept_avoiding",
    ):
        _require_complete_scoring_population(cast(Any, concepts), samples)


def _live_ours_contract_kwargs(tmp_path: Path) -> dict[str, Any]:
    model = SimpleNamespace(feature_vocab=None)
    bundle = TargetModelBundle(
        model=model,
        tokenizer=object(),
        max_seq_len=16,
        target_name="encdec",
    )
    method = _FakeMethod(
        name="ours_extract_and_add_encdec",
        baseline="ours",
        mode=SteeringMode.POSITIVE,
        target_model="encdec",
    )
    return {
        "cfg": _Cfg(),
        "output_paths": _FakeOutputPaths(root=tmp_path),
        "source_paths": None,
        "methods": [method],
        "concepts": [],
        "samples_by_concept_mode": {},
        "coefficient_grids": {method.name: [0]},
        "decoder_only_bundle": bundle,
        "encdec_bundle": bundle,
        "target_model_metas": {"decoder_only": None, "encdec": None},
        "tokenizer_nanochat": object(),
        "device": torch.device("cpu"),
        "ddp_rank": 0,
        "ddp_world_size": 1,
        "diffmean_layer": 1,
    }


def test_live_ours_contract_marker_is_written_before_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StopAfterOrderingProof(Exception):
        pass

    def reconcile_spy(*, rank_dir: Path, cache_path: Path) -> None:
        del cache_path
        marker = rank_dir.parent / LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME
        assert marker.read_text().strip() == ("axbench-bpb-live-ours-v1:uncompacted")
        raise _StopAfterOrderingProof

    monkeypatch.setattr(
        bpb_scoring_module,
        "reconcile_cache",
        reconcile_spy,
    )
    with pytest.raises(_StopAfterOrderingProof):
        run_bpb_score_stage(**_live_ours_contract_kwargs(tmp_path))


def test_premarker_affected_rows_reject_before_reconcile_or_cache_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "bpb_scores" / "rank_0001" / "batch_000000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"method_name": ["ours_extract_and_add_encdec"]}),
        shard,
    )
    monkeypatch.setattr(
        bpb_scoring_module,
        "reconcile_cache",
        lambda **_kwargs: pytest.fail("contract rejection must precede cache reconciliation"),
    )
    with pytest.raises(AxBenchBpbError, match="fresh run_tag|clear the old"):
        run_bpb_score_stage(**_live_ours_contract_kwargs(tmp_path))


def test_aggregate_stage_rejects_premarker_affected_rows_before_rollup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_paths = _FakeOutputPaths(root=tmp_path)
    shard = output_paths.bpb_scores_rank_dir(0) / "batch_000000.parquet"
    pq.write_table(
        pa.table({"method_name": ["ours_extract_and_add_encdec"]}),
        shard,
    )
    monkeypatch.setattr(
        axbench_bpb_eval_module,
        "compute_per_tuple_rollup",
        lambda _table: pytest.fail("aggregation contract rejection must precede rollup"),
    )
    runtime_state = cast(
        RuntimeState,
        SimpleNamespace(
            ddp_rank=0,
            output_paths=lambda: output_paths,
        ),
    )
    with pytest.raises(AxBenchBpbError, match="fresh run_tag|score it again"):
        _stage_bpb_aggregate(runtime_state)
