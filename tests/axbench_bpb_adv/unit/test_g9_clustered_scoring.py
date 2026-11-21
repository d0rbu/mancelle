from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.eval.axbench_bpb_adv.scoring as scoring
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    CLUSTERED_CONTEXT_CACHE_KIND,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA
from steering.eval.target_conditioning import ContextFeatureSegment


def _candidate(*, target_model: str = "encdec") -> dict[str, Any]:
    source = [97, 98]
    rich = [99]
    avoid = [100]
    return {
        "candidate_id": "candidate",
        "attack_row_id": "attack",
        "core_prefix_id": "prefix",
        "concept_rich_sample_id": "rich",
        "concept_avoiding_sample_id": "avoid",
        "concept_id": 1,
        "mode": "positive",
        "method_name": (
            "control_encdec"
            if target_model == "encdec"
            else "control_decoder_only"
        ),
        "target_model": target_model,
        "coefficient_str": "0.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 0.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 0,
        "context_signal_mode": "fixed_context",
        "prefix_text": "ab",
        "prefix_hash": "prefix-hash",
        "source_prompt_tokens": source,
        "source_prompt_token_digest": token_sequence_digest(source),
        "attacked_prompt_text": "ab",
        "attacked_prompt_tokens": source,
        "attacked_prompt_token_digest": token_sequence_digest(source),
        "concept_rich_output_tokens": rich,
        "concept_avoiding_output_tokens": avoid,
        "concept_rich_output_digest": token_sequence_digest(rich),
        "concept_avoiding_output_digest": token_sequence_digest(avoid),
        "editable_token_start": len(source),
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": None,
        "candidate_metadata_json": "{}",
    }


def _candidate_table(*, target_model: str = "encdec") -> pa.Table:
    return pa.Table.from_pylist(
        [_candidate(target_model=target_model)],
        schema=CANDIDATES_SCHEMA,
    )


def _paths(tmp_path: Path) -> AxBenchBpbAdvPaths:
    return AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")


def test_load_candidate_table_legacy_mode_does_not_read_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidates.parquet"
    expected = _candidate_table()
    pq.write_table(expected, path)
    monkeypatch.setattr(
        scoring,
        "assert_table_cluster_identity",
        lambda *_args, **_kwargs: pytest.fail("legacy load read identity"),
    )

    actual = scoring.load_candidate_table(path)

    assert actual.equals(expected)


def test_load_candidate_table_with_paths_validates_unclustered_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidates.parquet"
    expected = _candidate_table()
    pq.write_table(expected, path)
    paths = _paths(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scoring,
        "assert_table_cluster_identity",
        lambda table, **kwargs: calls.append({"table": table, **kwargs}),
    )

    actual = scoring.load_candidate_table(path, paths=paths)

    assert actual.equals(expected)
    assert calls == [
        {
            "table": actual,
            "paths": paths,
            "contract": None,
            "artifact_family": "candidates",
            "source_identity": None,
        }
    ]


def test_load_candidate_table_threads_supplied_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidates.parquet"
    expected = _candidate_table()
    pq.write_table(expected, path)
    paths = _paths(tmp_path)
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)
    source_identity = object()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scoring,
        "assert_table_cluster_identity",
        lambda table, **kwargs: calls.append({"table": table, **kwargs}),
    )

    actual = scoring.load_candidate_table(
        path,
        paths=paths,
        cluster_contract=contract,
        source_identity=source_identity,  # ty: ignore[invalid-argument-type]
    )

    assert actual.equals(expected)
    assert calls == [
        {
            "table": actual,
            "paths": paths,
            "contract": contract,
            "artifact_family": "candidates",
            "source_identity": source_identity,
        }
    ]


def test_load_candidate_table_clustered_requires_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidates.parquet"
    pq.write_table(_candidate_table(), path)
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="clustered candidate loading requires AxBenchBpbAdvPaths",
    ):
        scoring.load_candidate_table(path, cluster_contract=contract)

    observed: list[Any] = []
    monkeypatch.setattr(
        scoring,
        "assert_table_cluster_identity",
        lambda _table, **kwargs: observed.append(kwargs["contract"]),
    )
    scoring.load_candidate_table(
        path,
        paths=_paths(tmp_path),
        cluster_contract=contract,
    )
    assert observed == [contract]


def test_context_loader_clustered_dispatches_without_reading_legacy_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)
    expected = {"prefix": ()}
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scoring,
        "load_clustered_context_segments_by_prefix_hash",
        lambda **kwargs: calls.append(kwargs) or expected,
    )

    assert (
        scoring.load_context_segments_by_prefix_hash(
            paths,
            cluster_contract=contract,
        )
        is expected
    )
    assert calls == [{"paths": paths, "contract": contract}]
    assert not paths.context_cache_manifest().exists()


def test_context_loader_unclustered_never_dispatches_clustered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.context_cache_manifest().parent.mkdir(parents=True)
    paths.context_cache_manifest().write_text(
        json.dumps({"prefix": {"ignored": True}}),
        encoding="utf-8",
    )
    segment = ContextFeatureSegment(
        feature_ids=(3,),
        buckets=(4,),
        char_start=0,
        char_end=2,
    )
    monkeypatch.setattr(
        scoring,
        "load_clustered_context_segments_by_prefix_hash",
        lambda **_kwargs: pytest.fail("unclustered load dispatched clustered"),
    )
    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        scoring,
        "load_context_segments",
        lambda path: loaded_paths.append(path) or (segment,),
    )

    assert scoring.load_context_segments_by_prefix_hash(paths) == {
        "prefix": (segment,)
    }
    assert loaded_paths == [paths.context_cache_file("prefix")]


def test_context_loader_rejects_non_object_manifest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.context_cache_manifest().parent.mkdir(parents=True)
    paths.context_cache_manifest().write_text("[]", encoding="utf-8")

    with pytest.raises(
        AxBenchBpbAdvError,
        match="context-cache manifest must be a JSON object",
    ):
        scoring.load_context_segments_by_prefix_hash(paths)

    paths.context_cache_manifest().write_text("{}", encoding="utf-8")
    assert scoring.load_context_segments_by_prefix_hash(paths) == {}


def test_unclustered_context_reader_rejects_clustered_manifest_before_segment_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.context_cache_manifest().parent.mkdir(parents=True)
    paths.context_cache_manifest().write_text(
        json.dumps(
            {
                "kind": CLUSTERED_CONTEXT_CACHE_KIND,
                "prefixes": {"prefix": {}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scoring,
        "load_context_segments",
        lambda _path: pytest.fail("segment loaded before manifest rejection"),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="unclustered BPB-Adv reader rejects a clustered",
    ):
        scoring.load_context_segments_by_prefix_hash(paths)

    paths.context_cache_manifest().write_text("{}", encoding="utf-8")
    assert scoring.load_context_segments_by_prefix_hash(paths) == {}


def test_bpb_config_threads_optional_extraction_model_without_legacy_drift() -> None:
    legacy = scoring._bpb_config_from_adv_config(AxBenchBpbAdvConfig())
    explicit = scoring._bpb_config_from_adv_config(
        AxBenchBpbAdvConfig(feature_extraction_model_id="google/gemma-3-4b-pt")
    )

    expected_legacy = asdict(legacy)
    assert expected_legacy["feature_extraction_model_id"] is None
    assert explicit.feature_extraction_model_id == "google/gemma-3-4b-pt"
    expected_legacy["feature_extraction_model_id"] = explicit.feature_extraction_model_id
    assert asdict(explicit) == expected_legacy


@pytest.mark.parametrize("clustered", [False, True])
def test_scorer_threads_cluster_contract_only_when_set(
    clustered: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract: Any = (
        SimpleNamespace(feature_contract_fingerprint="f" * 64)
        if clustered
        else None
    )
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scoring,
        "build_intervention",
        lambda **kwargs: observed.append(kwargs) or Intervention.none(),
    )
    scorer = scoring.ModelBackedDualCompletionScorer(
        target_bundles=cast(Any, {"encdec": SimpleNamespace(model=object())}),
        target_model_metas={"encdec": object()},
        concepts_by_id={},
        source_paths=cast(Any, SimpleNamespace(root=tmp_path)),
        tokenizer=object(),
        device=t.device("cpu"),
        diffmean_layer=8,
        inactive_sampler=None,
        apply_max_act_scale=False,
        cluster_contract=contract,
    )
    scorer._base_steered_intervention(
        method=object(),
        concept=cast(Any, SimpleNamespace()),
        coefficient=1.0,
        candidate_row={
            "method_name": "method",
            "concept_id": 1,
            "core_prefix_id": "prefix",
            "coefficient_str": "1.0",
            "target_model": "encdec",
        },
    )

    assert len(observed) == 1
    if clustered:
        assert observed[0]["cluster_contract"] is contract
    else:
        assert "cluster_contract" not in observed[0]


@pytest.mark.parametrize(
    "wrapper_name,rows_kind",
    [
        ("require_precomputed_bag_candidates_cover_scored_prefixes", "table"),
        ("require_precomputed_bag_prompt_rows_cover_scored_prefixes", "rows"),
    ],
)
@pytest.mark.parametrize("fingerprint", [None, "f" * 64])
def test_precomputed_bag_coverage_threads_fingerprint_exactly(
    wrapper_name: str,
    rows_kind: str,
    fingerprint: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _candidate()
    row["method_name"] = "ours_gcg_encdec"
    rows = [row]
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scoring,
        "read_ours_gcg_scored_prefixes",
        lambda _path, **kwargs: observed.append(kwargs) or ["prefix"],
    )
    wrapper = getattr(scoring, wrapper_name)
    value: Any = (
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)
        if rows_kind == "table"
        else rows
    )

    returned = wrapper(
        value,
        paths=_paths(tmp_path),
        feature_contract_fingerprint=fingerprint,
    )

    assert returned is value
    assert observed == (
        [{}]
        if fingerprint is None
        else [{"expected_feature_contract_fingerprint": fingerprint}]
    )


def test_precomputed_bag_coverage_early_return_never_reads_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scoring,
        "read_ours_gcg_scored_prefixes",
        lambda *_args, **_kwargs: pytest.fail("non-bag row read artifact"),
    )
    rows = [_candidate(target_model="decoder_only")]

    assert (
        scoring.require_precomputed_bag_prompt_rows_cover_scored_prefixes(
            rows,
            paths=_paths(tmp_path),
            feature_contract_fingerprint="f" * 64,
        )
        is rows
    )


@pytest.mark.parametrize("target_model", ["encdec", "decoder_only"])
def test_clustered_score_stage_threads_contract_on_every_target_branch(
    target_model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)
    candidates = _candidate_table(target_model=target_model)
    paths = _paths(tmp_path)
    observed: dict[str, Any] = {}
    validated_concept = Concept(
        concept_id=1,
        feature_index=11,
        layer=12,
        sae_source="validated-a",
        description="validated concept",
        corpus_count=13,
        description_source="fixture",
    )
    source_identity = object()
    stage_source = SimpleNamespace(
        validation=SimpleNamespace(
            concept_entries=lambda: [asdict(validated_concept)]
        ),
        source_identity=source_identity,
    )
    candidate_snapshot = object()

    def _require_stage_source(**kwargs: Any) -> object:
        observed["stage_source_guard"] = kwargs
        return stage_source

    monkeypatch.setattr(
        scoring,
        "require_clustered_stage_source",
        _require_stage_source,
    )
    monkeypatch.setattr(
        scoring,
        "require_clustered_candidate_snapshot",
        lambda snapshot, **kwargs: observed.setdefault(
            "candidate_snapshot_guard",
            {"snapshot": snapshot, **kwargs},
        )
        or snapshot,
    )

    monkeypatch.setattr(
        scoring,
        "read_evaluation_population",
        lambda _path, **kwargs: observed.setdefault("population", kwargs) or object(),
    )
    monkeypatch.setattr(
        scoring,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: table,
    )

    def _coverage(table: pa.Table, **kwargs: Any) -> pa.Table:
        observed["coverage"] = kwargs
        return table

    monkeypatch.setattr(
        scoring,
        "require_precomputed_bag_candidates_cover_scored_prefixes",
        _coverage,
    )

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            observed["runtime"] = kwargs

        def decoder_only(self) -> Any:
            return SimpleNamespace(model=object(), tokenizer=_Tokenizer(), max_seq_len=32)

        def encdec(self) -> Any:
            return SimpleNamespace(model=object(), tokenizer=_Tokenizer(), max_seq_len=32)

        def target_device(self) -> str:
            return "cpu"

        def inactive_sampler(self) -> Any:
            return None

        def offload_models(self) -> None:
            observed["offloaded"] = True

    class _Tokenizer:
        def get_vocab_size(self) -> int:
            return 1

        def decode(self, _tokens: list[int]) -> str:
            return ""

    monkeypatch.setattr(scoring, "RuntimeState", _Runtime)
    monkeypatch.setattr(
        scoring,
        "load_context_segments_by_prefix_hash",
        lambda _paths, **kwargs: observed.setdefault("context", kwargs) or {},
    )
    monkeypatch.setattr(
        scoring,
        "detect_target_model_metadata",
        lambda _model, _cfg, **kwargs: observed.setdefault("detect", kwargs)
        or object(),
    )
    monkeypatch.setattr(
        scoring,
        "_read_concepts",
        lambda _paths: pytest.fail(
            "clustered score stage must not reopen concepts.json"
        ),
    )
    monkeypatch.setattr(
        scoring,
        "_preflight_direction_scales_for_rows",
        lambda **_kwargs: SimpleNamespace(identity="dirmaxact:test"),
    )
    monkeypatch.setattr(
        scoring,
        "ModelBackedDualCompletionScorer",
        lambda **kwargs: observed.setdefault("scorer", kwargs) or object(),
    )
    expected = pa.table({"ok": [True]})
    monkeypatch.setattr(
        scoring,
        "build_bpb_adv_score_table",
        lambda **_kwargs: expected,
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path),
        output_root=str(tmp_path),
        run_tag="run",
        device="cpu",
        target_models=cast(Any, [target_model]),
        feature_statistics_path="stats",
    )

    assert (
        scoring.run_model_backed_bpb_adv_score_stage(
            cfg=cfg,
            paths=paths,
            candidates=candidates,
            cluster_contract=contract,
            stage_source=stage_source,  # ty: ignore[invalid-argument-type]
            candidate_snapshot=cast(Any, candidate_snapshot),
        )
        is expected
    )
    assert observed["population"] == {
        "expected_feature_contract_fingerprint": contract.feature_contract_fingerprint
    }
    assert observed["coverage"]["feature_contract_fingerprint"] == (
        contract.feature_contract_fingerprint
    )
    assert observed["runtime"]["cluster_contract"] is contract
    assert observed["stage_source_guard"] == {
        "stage_source": stage_source,
        "contract": contract,
        "surface": "class-margin score stage",
    }
    assert observed["candidate_snapshot_guard"] == {
        "snapshot": candidate_snapshot,
        "paths": paths,
        "contract": contract,
        "source_identity": source_identity,
        "surface": "class-margin score stage",
        "table": candidates,
    }
    assert observed["detect"]["cluster_contract"] is contract
    assert observed["scorer"]["cluster_contract"] is contract
    assert observed["scorer"]["source_validation"] is stage_source.validation
    assert observed["scorer"]["concepts_by_id"] == {1: validated_concept}
    if target_model == "encdec":
        assert observed["context"]["cluster_contract"] is contract
    else:
        assert "context" not in observed


def test_clustered_score_stage_requires_source_before_input_checks(
    tmp_path: Path,
) -> None:
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="class-margin score stage requires one resolved stage source",
    ):
        scoring.run_model_backed_bpb_adv_score_stage(
            cfg=AxBenchBpbAdvConfig(run_tag=None),
            paths=_paths(tmp_path),
            candidates=pa.table({}),
            cluster_contract=contract,
        )


def test_unclustered_score_stage_omits_runtime_cluster_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidate_table(target_model="decoder_only")
    monkeypatch.setattr(
        scoring,
        "read_evaluation_population",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        scoring,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: table,
    )
    monkeypatch.setattr(
        scoring,
        "require_precomputed_bag_candidates_cover_scored_prefixes",
        lambda table, **_kwargs: table,
    )

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            assert set(kwargs) == {"cfg", "ddp_rank", "ddp_world_size"}
            raise RuntimeError("unclustered runtime sentinel")

    monkeypatch.setattr(scoring, "RuntimeState", _Runtime)
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path),
        output_root=str(tmp_path),
        run_tag="run",
        target_models=["decoder_only"],
    )

    with pytest.raises(RuntimeError, match="unclustered runtime sentinel"):
        scoring.run_model_backed_bpb_adv_score_stage(
            cfg=cfg,
            paths=_paths(tmp_path),
            candidates=candidates,
        )
