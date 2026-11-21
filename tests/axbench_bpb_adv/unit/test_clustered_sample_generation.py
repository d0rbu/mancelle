from __future__ import annotations

import inspect
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.eval.axbench_bpb_adv.intervention_build as ib
import steering.eval.axbench_bpb_adv.sample_generation as sg
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.candidate_metadata import (
    NoAdversaryControlMetadata,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    GeneralValBpbConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
)
from steering.eval.target_conditioning import SingleEntryContextProvider


class _Tokenizer:
    def decode(self, ids: list[int]) -> str:
        return ",".join(str(value) for value in ids)


class _Engine:
    def generate_stream(self, **kwargs: Any) -> list[int]:
        return [int(kwargs["seed"]), *kwargs["token_ids"]]


class _Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def to_pylist(self) -> list[dict[str, Any]]:
        return self.rows


class _Consolidation:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[int] = []

    def assert_representatives_only(self, ids: t.Tensor, *, surface: str) -> None:
        assert surface == "sample generation encoder input"
        self.seen = [int(value) for value in ids.tolist()]
        if self.fail:
            raise ValueError("non-representative")


class _Mask:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[int] = []

    def assert_no_masked(self, ids: t.Tensor) -> None:
        assert ids.dtype == t.int64
        self.seen = [int(value) for value in ids.tolist()]
        if self.fail:
            raise ValueError("masked")


def _contract(*, fail: bool = False, mask_fail: bool = False) -> Any:
    return SimpleNamespace(
        consolidation=_Consolidation(fail=fail),
        feature_mask=_Mask(fail=mask_fail),
    )


def _config(
    *,
    source_run_dir: str = "source",
    output_root: str = "out",
    run_tag: str | None = None,
    target_models: list[str] | None = None,
    encdec_target: dict[str, Any] | None = None,
    filter_unaccepted_samples: bool = True,
    steering_metric: str = "bpb_class_margin",
    scores_path: str | None = None,
    general_val_enabled: bool = False,
) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir=source_run_dir,
        output_root=output_root,
        run_tag=run_tag,
        target_models=cast(Any, target_models or ["decoder_only", "encdec"]),
        encdec_target=encdec_target or {},
        filter_unaccepted_samples=filter_unaccepted_samples,
        steering_metric=cast(Any, steering_metric),
        scores_path=scores_path,
        feature_statistics_path="stats.safetensors",
        general_val_bpb=GeneralValBpbConfig(enabled=general_val_enabled),
        scatter_baseline_x="baseline_loss",
    )


def _concept() -> Concept:
    return Concept(
        concept_id=1,
        feature_index=3,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="concept",
        corpus_count=10,
        description_source="test",
    )


_LEGACY_FAMILIES_ENCDEC_ERROR = (
    "families_with_representative_coefficient: encoder-decoder sampling is a "
    "deliberate follow-up; only target_model='decoder_only' is supported"
)
_LEGACY_GENERATION_ENCDEC_ERROR = (
    "generate_completions_for_datapoint: encoder-decoder generation is a "
    "deliberate follow-up; only decoder-only interventions are supported"
)
_LEGACY_CORE_ENCDEC_ERROR = (
    "run_generate_samples_core: encoder-decoder sampling is a deliberate "
    "follow-up; only target_model='decoder_only' is supported"
)


def _capture_exception(call: Any) -> Exception:
    try:
        call()
    except Exception as exc:
        return exc
    raise AssertionError("expected the named legacy rejection, but no exception was raised")


def test_build_steered_intervention_threads_clustered_bag_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    base = Intervention.none()
    monkeypatch.setattr(
        ib,
        "build_intervention",
        lambda **kwargs: captured.update(kwargs) or base,
    )
    monkeypatch.setattr(
        ib,
        "apply_additive_counter",
        lambda intervention, **kwargs: intervention,
    )
    contract = _contract()
    paths = object()

    result = ib.build_steered_intervention(
        method=SimpleNamespace(),
        concept=_concept(),
        coefficient=1.0,
        source_paths=SourcePaths(Path("source")),
        diffmean_layer=3,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        rng=random.Random(0),
        adversarial_method="additive_direction",
        adversarial_strength=0.0,
        counter_direction_family="diffmean",
        target="decoder_only",
        apply_max_act_scale=False,
        adv_paths=paths,
        core_prefix_id="prefix",
        cluster_contract=contract,
    )

    assert result is base
    assert captured.get("adv_paths") is paths
    assert captured.get("core_prefix_id") == "prefix"
    assert captured.get("cluster_contract") is contract


@pytest.mark.parametrize(
    ("function", "parameter_names"),
    [
        (
            sg.generate_completions_for_datapoint,
            ("context_provider", "cluster_contract", "pre_forward_guard"),
        ),
        (
            sg.run_generate_samples_core,
            ("cluster_contract", "pre_forward_guard"),
        ),
        (
            ib.build_steered_intervention,
            ("adv_paths", "core_prefix_id", "cluster_contract"),
        ),
    ],
)
def test_g10_optional_parameters_are_keyword_only_and_none_defaulted(
    function: Any,
    parameter_names: tuple[str, ...],
) -> None:
    parameters = inspect.signature(function).parameters
    for name in parameter_names:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None


def test_unclustered_families_encdec_rejection_is_legacy_exact() -> None:
    caught = _capture_exception(
        lambda: sg.families_with_representative_coefficient(
            [],
            target_model="encdec",
        )
    )
    assert type(caught) is AxBenchBpbAdvError
    assert str(caught) == _LEGACY_FAMILIES_ENCDEC_ERROR


def test_families_reject_unknown_target_directly() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="target_model must be.*got 'unknown'",
    ):
        sg.families_with_representative_coefficient([], target_model="unknown")


def test_unclustered_generation_encdec_rejection_is_legacy_exact() -> None:
    caught = _capture_exception(
        lambda: sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=Intervention.ours(feature_conditions={3: 0}),
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
        )
    )
    assert type(caught) is AxBenchBpbAdvError
    assert str(caught) == _LEGACY_GENERATION_ENCDEC_ERROR


def test_unclustered_core_encdec_rejection_is_legacy_exact(tmp_path: Path) -> None:
    caught = _capture_exception(
        lambda: sg.run_generate_samples_core(
            paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run"),
            source_run_dir=tmp_path,
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            target_model="encdec",
            target_model_meta=None,
            inactive_sampler=None,
            diffmean_layer=0,
            counter_direction_family="diffmean",
            scatter_baseline_x="baseline_loss",
            steering_metric="bpb_class_margin",
            generation=sg.GenerationConfig(),
            apply_max_act_scale=False,
        )
    )
    assert type(caught) is AxBenchBpbAdvError
    assert str(caught) == _LEGACY_CORE_ENCDEC_ERROR


def test_core_rejects_unknown_target_directly(tmp_path: Path) -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="target_model must be.*got 'unknown'",
    ):
        sg.run_generate_samples_core(
            paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run"),
            source_run_dir=tmp_path,
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            target_model="unknown",
            target_model_meta=None,
            inactive_sampler=None,
            diffmean_layer=0,
            counter_direction_family="diffmean",
            scatter_baseline_x="baseline_loss",
            steering_metric="bpb_class_margin",
            generation=sg.GenerationConfig(),
            apply_max_act_scale=False,
            cluster_contract=_contract(),
        )


def test_clustered_encdec_generation_uses_context_and_validates_target_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _build_engine(
        model: Any,
        tokenizer: Any,
        device: t.device,
        features: dict[int, int],
        *,
        context_provider: Any,
    ) -> _Engine:
        captured.update(
            model=model,
            tokenizer=tokenizer,
            device=device,
            features=features,
            context_provider=context_provider,
        )
        return _Engine()

    monkeypatch.setattr(sg, "build_engine_for_features", _build_engine)
    provider = SingleEntryContextProvider([4, 5], ())
    contract = _contract()
    model = object()
    tokenizer = _Tokenizer()
    guard_calls: list[str] = []
    completions = sg.generate_completions_for_datapoint(
        model=model,
        tokenizer=tokenizer,
        device=t.device("cpu"),
        intervention=Intervention.ours(feature_conditions={7: 1, 3: 0}),
        prompt_tokens=[4, 5],
        n_completions=2,
        temperature=0.0,
        top_k=None,
        max_new_tokens=3,
        base_seed=11,
        context_provider=provider,
        cluster_contract=contract,
        pre_forward_guard=lambda: guard_calls.append("checked"),
    )
    assert captured["features"] == {7: 1, 3: 0}
    assert captured["context_provider"] is provider
    assert contract.consolidation.seen == [3, 7]
    assert guard_calls == ["checked", "checked"]
    assert completions == ["11,4,5", "12,4,5"]


def test_clustered_encdec_generation_requires_guard_before_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NeverForwardEngine:
        def generate_stream(self, **_kwargs: Any) -> list[int]:
            pytest.fail("model forward must not start without the checkpoint guard")

    monkeypatch.setattr(
        sg,
        "build_engine_for_features",
        lambda *args, **kwargs: _NeverForwardEngine(),
    )
    with pytest.raises(AxBenchBpbAdvError, match="missing.*pre-forward"):
        sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=Intervention.ours(feature_conditions={3: 0}),
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
            context_provider=SingleEntryContextProvider([4], ()),
            cluster_contract=_contract(),
        )


def test_clustered_encdec_generation_rejects_nonrepresentative_before_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sg,
        "build_engine_for_features",
        lambda *args, **kwargs: pytest.fail("encoder packing must not be reached"),
    )
    with pytest.raises(AxBenchBpbAdvError, match="invalid clustered bag"):
        sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=Intervention.ours(feature_conditions={9: 1}),
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
            context_provider=SingleEntryContextProvider([4], ()),
            cluster_contract=_contract(fail=True),
        )


def test_clustered_encdec_generation_rejects_masked_before_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sg,
        "build_engine_for_features",
        lambda *args, **kwargs: pytest.fail("encoder packing must not be reached"),
    )
    contract = _contract(mask_fail=True)
    with pytest.raises(
        AxBenchBpbAdvError,
        match="invalid clustered bag.*masked",
    ):
        sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=Intervention.ours(feature_conditions={9: 1}),
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
            context_provider=SingleEntryContextProvider([4], ()),
            cluster_contract=contract,
        )
    assert contract.consolidation.seen == [9]
    assert contract.feature_mask.seen == [9]


def test_decoder_only_generation_rejects_context_provider_directly() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="decoder-only.*must not receive a context provider",
    ):
        sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=Intervention.none(),
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
            context_provider=SingleEntryContextProvider([4], ()),
        )


def test_unclustered_decoder_generation_runs_without_cluster_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sg,
        "InferenceEngine",
        lambda model, tokenizer, device: _Engine(),
    )

    completions = sg.generate_completions_for_datapoint(
        model=object(),
        tokenizer=_Tokenizer(),
        device=t.device("cpu"),
        intervention=Intervention.none(),
        prompt_tokens=[4, 5],
        n_completions=2,
        temperature=0.0,
        top_k=None,
        max_new_tokens=3,
        base_seed=11,
    )

    assert completions == ["11,4,5", "12,4,5"]


def test_clustered_encdec_generation_requires_scored_context_cache() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="requires the scored context cache",
    ):
        sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=Intervention.ours(feature_conditions={3: 0}),
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
            cluster_contract=_contract(),
        )


def test_generation_rejects_corrupted_unknown_target_directly() -> None:
    intervention = Intervention.none()
    object.__setattr__(intervention, "target_model", "unknown")
    with pytest.raises(
        AxBenchBpbAdvError,
        match="unsupported sample-generation target 'unknown'",
    ):
        sg.generate_completions_for_datapoint(
            model=object(),
            tokenizer=_Tokenizer(),
            device=t.device("cpu"),
            intervention=intervention,
            prompt_tokens=[4],
            n_completions=1,
            temperature=0.0,
            top_k=None,
            max_new_tokens=1,
            base_seed=0,
        )


def test_clustered_sample_preflight_requires_finished_run_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="incomplete")
    monkeypatch.setattr(sg, "build_cluster_contract_request", lambda cfg: object())
    monkeypatch.setattr(
        sg,
        "resolve_cluster_feature_contract",
        lambda request: pytest.fail("target resolution must not start"),
    )

    cfg = _config()
    with pytest.raises(AxBenchBpbAdvError, match="already-finished scored run"):
        sg._resolve_sample_cluster_stage(cfg, paths=paths)

    assert not paths.root.exists()


def test_unclustered_sample_preflight_is_noop_without_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="legacy")
    monkeypatch.setattr(sg, "build_cluster_contract_request", lambda cfg: None)
    assert sg._resolve_sample_cluster_stage(
        _config(),
        paths=paths,
    ) == (None, None)
    assert not paths.root.exists()


def test_unclustered_sample_preflight_rejects_cluster_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="mixed")
    paths.root.mkdir(parents=True)
    (paths.root / "feature_contract.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sg, "build_cluster_contract_request", lambda cfg: None)
    with pytest.raises(
        AxBenchBpbAdvError,
        match="unclustered sample generation rejects cluster-stamped run output",
    ):
        sg._resolve_sample_cluster_stage(
            _config(),
            paths=paths,
        )


def _prepare_clustered_preflight_paths(
    tmp_path: Path,
    *,
    source_contract: object,
) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="clustered")
    paths.candidates_dir().mkdir(parents=True)
    (paths.root / "feature_contract.json").write_text("{}", encoding="utf-8")
    paths.candidates_path().touch()
    paths.source_contract().write_text(
        json.dumps(source_contract),
        encoding="utf-8",
    )
    return paths


def _install_clustered_preflight_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: Any,
    source_contract: Any,
) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    monkeypatch.setattr(sg, "build_cluster_contract_request", lambda cfg: "request")
    monkeypatch.setattr(
        sg,
        "resolve_cluster_feature_contract",
        lambda request: events.append(("resolve", request)) or contract,
    )
    monkeypatch.setattr(
        sg,
        "ensure_feature_contract_run_marker",
        lambda root, value: events.append(("marker", (root, value))),
    )
    monkeypatch.setattr(
        sg,
        "validate_stage_0_3_source_run",
        lambda source_run_dir, *, filter_unaccepted_samples: (
            events.append(
                (
                    "source",
                    (source_run_dir, filter_unaccepted_samples),
                )
            )
            or SimpleNamespace(contract=source_contract)
        ),
    )
    monkeypatch.setattr(
        sg,
        "resolve_clustered_stage_source",
        lambda **kwargs: events.append(("stage_source", kwargs))
        or SimpleNamespace(validation=kwargs["validation"]),
    )
    return events


def test_clustered_sample_preflight_accepts_identical_finished_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = {"contract_hash": "same"}
    source_contract = SimpleNamespace(
        feature_contract_fingerprint="f" * 64,
        to_json_dict=lambda: source_payload,
    )
    contract = SimpleNamespace(
        feature_contract_fingerprint="f" * 64,
        checkpoint=SimpleNamespace(step=37),
    )
    paths = _prepare_clustered_preflight_paths(
        tmp_path,
        source_contract=source_payload,
    )
    events = _install_clustered_preflight_stubs(
        monkeypatch,
        contract=contract,
        source_contract=source_contract,
    )
    cfg = _config(
        encdec_target={"source": "artifact"},
        source_run_dir="source-run",
        filter_unaccepted_samples=True,
    )

    resolved_contract, resolved_source = sg._resolve_sample_cluster_stage(
        cfg,
        paths=paths,
    )
    assert resolved_contract is contract
    assert resolved_source is not None
    assert resolved_source.validation.contract is source_contract
    assert cfg.encdec_target == {"source": "artifact", "step": 37}
    assert events[:3] == [
        ("resolve", "request"),
        ("marker", (paths.root, contract)),
        ("source", ("source-run", True)),
    ]
    assert events[3][0] == "stage_source"
    stage_kwargs = events[3][1]
    assert stage_kwargs["validation"].contract is source_contract
    assert stage_kwargs["paths"] is paths
    assert stage_kwargs["contract"] is contract
    assert stage_kwargs["expected_source_run_dir"] == "source-run"
    assert stage_kwargs["expected_filter_unaccepted_samples"] is True


def test_clustered_sample_preflight_rejects_feature_contract_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_contract = SimpleNamespace(
        feature_contract_fingerprint="source",
        to_json_dict=dict,
    )
    contract = SimpleNamespace(
        feature_contract_fingerprint="target",
        checkpoint=SimpleNamespace(step=1),
    )
    paths = _prepare_clustered_preflight_paths(tmp_path, source_contract={})
    _install_clustered_preflight_stubs(
        monkeypatch,
        contract=contract,
        source_contract=source_contract,
    )
    cfg = _config(
        source_run_dir="source",
        filter_unaccepted_samples=False,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="source and target feature contracts disagree",
    ):
        sg._resolve_sample_cluster_stage(
            cfg,
            paths=paths,
        )

    source_contract.feature_contract_fingerprint = "target"
    resolved_contract, resolved_source = sg._resolve_sample_cluster_stage(
        cfg,
        paths=paths,
    )
    assert resolved_contract is contract
    assert resolved_source is not None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "finished run's exact source contract"),
        (json.dumps({"stale": True}), "source contract is stale"),
    ],
)
def test_clustered_sample_preflight_rejects_unreadable_or_stale_source_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    expected = {"current": True}
    source_contract = SimpleNamespace(
        feature_contract_fingerprint="same",
        to_json_dict=lambda: expected,
    )
    contract = SimpleNamespace(
        feature_contract_fingerprint="same",
        checkpoint=SimpleNamespace(step=1),
    )
    paths = _prepare_clustered_preflight_paths(tmp_path, source_contract={})
    paths.source_contract().write_text(payload, encoding="utf-8")
    _install_clustered_preflight_stubs(
        monkeypatch,
        contract=contract,
        source_contract=source_contract,
    )
    cfg = _config(
        source_run_dir="source",
        filter_unaccepted_samples=False,
    )
    with pytest.raises(AxBenchBpbAdvError, match=message):
        sg._resolve_sample_cluster_stage(
            cfg,
            paths=paths,
        )

    paths.source_contract().write_text(json.dumps(expected), encoding="utf-8")
    resolved_contract, resolved_source = sg._resolve_sample_cluster_stage(
        cfg,
        paths=paths,
    )
    assert resolved_contract is contract
    assert resolved_source is not None


def _install_refresh_owner_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric: str,
) -> tuple[Any, Any, pa.Table, object]:
    stage_source = SimpleNamespace(source_identity=object())
    candidate_snapshot = object()
    expected_macro = pa.table({"exact": [True]})
    per_candidate = object()
    sampling_snapshot = object()
    monkeypatch.setattr(
        sg,
        "require_clustered_stage_source",
        lambda **_kwargs: stage_source,
    )
    monkeypatch.setattr(
        sg,
        "require_clustered_candidate_snapshot",
        lambda *_args, **_kwargs: candidate_snapshot,
    )
    monkeypatch.setattr(
        sg,
        "_build_clustered_sampling_snapshot",
        lambda **_kwargs: sampling_snapshot,
    )
    if metric == "roundtrip":
        monkeypatch.setattr(
            sg,
            "compute_roundtrip_per_candidate_aggregation",
            lambda _scores: per_candidate,
        )
        monkeypatch.setattr(
            sg,
            "compute_roundtrip_macro_aggregation",
            lambda actual: (
                expected_macro
                if actual is per_candidate
                else pytest.fail("wrong roundtrip per-candidate table")
            ),
        )
        monkeypatch.setattr(
            sg,
            "_join_roundtrip_general_val_bpb",
            lambda actual, _general: actual,
        )
    else:
        monkeypatch.setattr(
            sg,
            "compute_per_candidate_aggregation",
            lambda _scores: per_candidate,
        )
        monkeypatch.setattr(
            sg,
            "compute_macro_aggregation",
            lambda actual: (
                expected_macro
                if actual is per_candidate
                else pytest.fail("wrong class-margin per-candidate table")
            ),
        )
        monkeypatch.setattr(
            sg,
            "_join_general_val_bpb",
            lambda actual, _general: actual,
        )
    return stage_source, candidate_snapshot, expected_macro, sampling_snapshot


def _sampling_metrics(metric: str) -> sg._SamplingMetrics:
    if metric == "roundtrip":
        return sg._SamplingMetrics(
            macro=pa.Table.from_pylist([], schema=ROUNDTRIP_MACRO_AGG_SCHEMA),
            per_candidate=pa.Table.from_pylist(
                [],
                schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
            ),
            representative_baseline_metric="primary_roundtrip_y",
            primary_metric_field="primary_roundtrip_y",
            primary_metric_output_key="primary_roundtrip_y",
        )
    return sg._SamplingMetrics(
        macro=pa.Table.from_pylist([], schema=MACRO_AGG_SCHEMA),
        per_candidate=pa.Table.from_pylist(
            [],
            schema=PER_CANDIDATE_AGG_SCHEMA,
        ),
        representative_baseline_metric="baseline_primary_margin",
        primary_metric_field="primary_margin_y",
        primary_metric_output_key="primary_margin_y",
    )


def _class_margin_score_table() -> pa.Table:
    rows = []
    for strength_index, (strength, is_control, rich_loss) in enumerate(
        [(0.0, True, 1.0), (1.0, False, 2.0)]
    ):
        rows.append(
            {
                "candidate_id": f"candidate-{strength_index}",
                "core_prefix_id": "prefix-1",
                "concept_id": 1,
                "mode": "positive",
                "method_name": "diffmean_decoder_only",
                "target_model": "decoder_only",
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": strength,
                "strength_unit": "optimized_tokens",
                "strength_index": strength_index,
                "context_signal_mode": "fixed_context",
                "source_prompt_tokens": [97, 98],
                "source_prompt_token_digest": "source-digest",
                "attacked_prompt_tokens": (
                    [97, 98] if is_control else [97, 120]
                ),
                "attacked_prompt_token_digest": (
                    "source-digest" if is_control else "attacked-digest"
                ),
                "editable_token_start": 2 if is_control else 1,
                "editable_token_count": 0 if is_control else 1,
                "replacement_tokens": [] if is_control else [120],
                "loss_concept_rich": rich_loss,
                "loss_concept_avoiding": 3.0,
                "is_control": is_control,
                "n_rich_output_tokens": 2,
                "n_avoid_output_tokens": 3,
                "n_rich_output_bytes": 2,
                "n_avoid_output_bytes": 5,
                "rich_truncated": False,
                "avoid_truncated": False,
            }
        )
    return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)


def _roundtrip_score_table() -> pa.Table:
    rows = []
    for strength_index, (strength, is_control, percentile) in enumerate(
        [(0.0, True, 80.0), (1.0, False, 40.0)]
    ):
        attacked = [97, 98] if is_control else [97, 120]
        rows.append(
            {
                "candidate_id": f"candidate-{strength_index}",
                "core_prefix_id": "prefix-1",
                "concept_id": 1,
                "mode": "positive",
                "method_name": "diffmean_decoder_only",
                "target_model": "decoder_only",
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": strength,
                "strength_unit": "optimized_tokens",
                "strength_index": strength_index,
                "context_signal_mode": "fixed_context",
                "source_prompt_tokens": [97, 98],
                "source_prompt_token_digest": "source-digest",
                "attacked_prompt_tokens": attacked,
                "attacked_prompt_token_digest": "attacked-digest",
                "roundtrip_prompt_tokens": attacked,
                "roundtrip_prompt_token_digest": "roundtrip-digest",
                "roundtrip_prompt_text": "ab" if is_control else "ax",
                "control_roundtrip_prompt_tokens": attacked,
                "control_roundtrip_prompt_token_digest": "control-digest",
                "control_roundtrip_prompt_text": (
                    "ab" if is_control else "ax"
                ),
                "editable_token_start": 2 if is_control else 1,
                "editable_token_count": 0 if is_control else 1,
                "replacement_tokens": [] if is_control else [120],
                "loss_concept_rich": 1.0,
                "loss_concept_avoiding": 3.0,
                "sample_index": 0,
                "reference_segment_token_count": 3,
                "generation_seed": 100,
                "generated_text": "generated",
                "n_generated_tokens": 3,
                "stop_reason": "max_tokens",
                "control_generation_seed": 200,
                "control_generated_text": "control",
                "control_n_generated_tokens": 2,
                "control_stop_reason": "eos",
                "is_control": is_control,
                "target_feature_id": 42,
                "n_roundtrip_features": 1,
                "roundtrip_pre_act": percentile / 10.0,
                "roundtrip_post_act": percentile / 20.0,
                "roundtrip_percentile": percentile,
                "control_n_roundtrip_features": 1,
                "control_roundtrip_pre_act": percentile / 10.0,
                "control_roundtrip_post_act": percentile / 20.0,
                "control_roundtrip_percentile": percentile,
            }
        )
    return pa.Table.from_pylist(rows, schema=ROUNDTRIP_SCORE_SCHEMA)


@pytest.mark.parametrize("metric", ["bpb_class_margin", "roundtrip"])
def test_clustered_sampling_snapshot_seals_exact_tables_and_coordinates(
    metric: str,
) -> None:
    metrics = _sampling_metrics(metric)
    candidate_snapshot: Any = SimpleNamespace(sha256="a" * 64)
    stage_source: Any = SimpleNamespace(
        source_identity=SimpleNamespace(source_contract_sha256="b" * 64),
    )

    snapshot = sg._build_clustered_sampling_snapshot(
        metrics=metrics,
        steering_metric=metric,
        candidate_snapshot=candidate_snapshot,
        stage_source=stage_source,
    )

    snapshot.__post_init__()
    assert snapshot.metrics is metrics
    assert snapshot.candidate_sha256 == "a" * 64
    assert snapshot.source_contract_sha256 == "b" * 64
    assert snapshot.steering_metric == metric


@pytest.mark.parametrize(
    ("field_name", "replacement_value"),
    [
        ("steering_metric", "roundtrip"),
        ("candidate_sha256", "c" * 64),
        ("source_contract_sha256", "d" * 64),
    ],
)
def test_clustered_sampling_snapshot_rejects_field_replacement(
    field_name: str,
    replacement_value: object,
) -> None:
    candidate_snapshot: Any = SimpleNamespace(sha256="a" * 64)
    stage_source: Any = SimpleNamespace(
        source_identity=SimpleNamespace(source_contract_sha256="b" * 64),
    )
    snapshot = sg._build_clustered_sampling_snapshot(
        metrics=_sampling_metrics("bpb_class_margin"),
        steering_metric="bpb_class_margin",
        candidate_snapshot=candidate_snapshot,
        stage_source=stage_source,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="fields changed after validation",
    ):
        replace(snapshot, **{field_name: replacement_value})


def test_clustered_sampling_snapshot_rejects_equal_table_substitution() -> None:
    candidate_snapshot: Any = SimpleNamespace(sha256="a" * 64)
    stage_source: Any = SimpleNamespace(
        source_identity=SimpleNamespace(source_contract_sha256="b" * 64),
    )
    snapshot = sg._build_clustered_sampling_snapshot(
        metrics=_sampling_metrics("bpb_class_margin"),
        steering_metric="bpb_class_margin",
        candidate_snapshot=candidate_snapshot,
        stage_source=stage_source,
    )
    replacement_metrics = replace(
        snapshot.metrics,
        macro=pa.concat_tables([snapshot.metrics.macro]),
    )
    assert replacement_metrics.macro.equals(snapshot.metrics.macro)
    assert replacement_metrics.macro is not snapshot.metrics.macro

    with pytest.raises(
        AxBenchBpbAdvError,
        match="fields changed after validation",
    ):
        replace(snapshot, metrics=replacement_metrics)


def test_clustered_sampling_snapshot_rejects_wrong_schema_and_names() -> None:
    candidate_snapshot: Any = SimpleNamespace(sha256="a" * 64)
    stage_source: Any = SimpleNamespace(
        source_identity=SimpleNamespace(source_contract_sha256="b" * 64),
    )
    wrong_schema = replace(
        _sampling_metrics("bpb_class_margin"),
        macro=pa.table({"wrong": [1]}),
    )
    with pytest.raises(AxBenchBpbAdvError, match="wrong aggregate schema"):
        sg._build_clustered_sampling_snapshot(
            metrics=wrong_schema,
            steering_metric="bpb_class_margin",
            candidate_snapshot=candidate_snapshot,
            stage_source=stage_source,
        )

    wrong_names = replace(
        _sampling_metrics("bpb_class_margin"),
        primary_metric_output_key="wrong",
    )
    with pytest.raises(AxBenchBpbAdvError, match="inconsistent metric names"):
        sg._build_clustered_sampling_snapshot(
            metrics=wrong_names,
            steering_metric="bpb_class_margin",
            candidate_snapshot=candidate_snapshot,
            stage_source=stage_source,
        )


def test_clustered_sampling_snapshot_rejects_direct_construction() -> None:
    with pytest.raises(TypeError, match="must be built"):
        sg._ClusteredSamplingSnapshot(
            metrics=_sampling_metrics("bpb_class_margin"),
            steering_metric="bpb_class_margin",
            candidate_sha256="a" * 64,
            source_contract_sha256="b" * 64,
        )


def test_clustered_sampling_snapshot_consumer_rejects_every_mispairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Candidate:
        def __init__(self, sha256: str) -> None:
            self.sha256 = sha256

    class Stage:
        def __init__(self, source_sha256: str) -> None:
            self.source_identity = SimpleNamespace(
                source_contract_sha256=source_sha256,
            )

    monkeypatch.setattr(sg, "ClusteredCandidateSnapshot", Candidate)
    monkeypatch.setattr(sg, "ClusteredStageSource", Stage)
    candidate = Candidate("a" * 64)
    stage = Stage("b" * 64)
    snapshot = sg._build_clustered_sampling_snapshot(
        metrics=_sampling_metrics("bpb_class_margin"),
        steering_metric="bpb_class_margin",
        candidate_snapshot=candidate,  # ty: ignore[invalid-argument-type]
        stage_source=stage,  # ty: ignore[invalid-argument-type]
    )

    with pytest.raises(AxBenchBpbAdvError, match="unclustered"):
        sg._require_clustered_sampling_snapshot(
            snapshot,
            steering_metric="bpb_class_margin",
            candidate_snapshot=candidate,  # ty: ignore[invalid-argument-type]
            stage_source=stage,  # ty: ignore[invalid-argument-type]
            clustered=False,
        )
    with pytest.raises(AxBenchBpbAdvError, match="exact refreshed"):
        sg._require_clustered_sampling_snapshot(
            None,
            steering_metric="bpb_class_margin",
            candidate_snapshot=candidate,  # ty: ignore[invalid-argument-type]
            stage_source=stage,  # ty: ignore[invalid-argument-type]
            clustered=True,
        )
    with pytest.raises(AxBenchBpbAdvError, match="paired candidate and source"):
        sg._require_clustered_sampling_snapshot(
            snapshot,
            steering_metric="bpb_class_margin",
            candidate_snapshot=None,
            stage_source=stage,  # ty: ignore[invalid-argument-type]
            clustered=True,
        )
    with pytest.raises(AxBenchBpbAdvError, match="different steering metric"):
        sg._require_clustered_sampling_snapshot(
            snapshot,
            steering_metric="roundtrip",
            candidate_snapshot=candidate,  # ty: ignore[invalid-argument-type]
            stage_source=stage,  # ty: ignore[invalid-argument-type]
            clustered=True,
        )
    with pytest.raises(AxBenchBpbAdvError, match="different candidate bytes"):
        sg._require_clustered_sampling_snapshot(
            snapshot,
            steering_metric="bpb_class_margin",
            candidate_snapshot=Candidate("c" * 64),  # ty: ignore[invalid-argument-type]
            stage_source=stage,  # ty: ignore[invalid-argument-type]
            clustered=True,
        )
    with pytest.raises(AxBenchBpbAdvError, match="different source contract"):
        sg._require_clustered_sampling_snapshot(
            snapshot,
            steering_metric="bpb_class_margin",
            candidate_snapshot=candidate,  # ty: ignore[invalid-argument-type]
            stage_source=Stage("d" * 64),  # ty: ignore[invalid-argument-type]
            clustered=True,
        )
    assert (
        sg._require_clustered_sampling_snapshot(
            snapshot,
            steering_metric="bpb_class_margin",
            candidate_snapshot=candidate,  # ty: ignore[invalid-argument-type]
            stage_source=stage,  # ty: ignore[invalid-argument-type]
            clustered=True,
        )
        is snapshot
    )


@pytest.mark.parametrize(
    ("metric", "score_table_factory", "aggregate_dir_name"),
    [
        (
            "bpb_class_margin",
            _class_margin_score_table,
            "aggregates",
        ),
        (
            "roundtrip",
            _roundtrip_score_table,
            "roundtrip_aggregates",
        ),
    ],
)
def test_clustered_refresh_tables_exactly_match_existing_aggregator_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metric: str,
    score_table_factory: Any,
    aggregate_dir_name: str,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    score_table = score_table_factory()
    score_path = (
        paths.roundtrip_scores_path()
        if metric == "roundtrip"
        else paths.scores_path()
    )
    real_read_table = sg.pq.read_table
    identity_checks: list[tuple[pa.Table, dict[str, Any]]] = []
    stage_source: Any = SimpleNamespace(
        source_identity=SimpleNamespace(source_contract_sha256="b" * 64),
    )
    candidate_snapshot: Any = SimpleNamespace(sha256="a" * 64)
    monkeypatch.setattr(
        sg,
        "require_clustered_stage_source",
        lambda **_kwargs: stage_source,
    )
    monkeypatch.setattr(
        sg,
        "require_clustered_candidate_snapshot",
        lambda *_args, **_kwargs: candidate_snapshot,
    )
    monkeypatch.setattr(
        sg.pq,
        "read_table",
        lambda path: (
            score_table
            if Path(path) == score_path
            else real_read_table(path)
        ),
    )
    monkeypatch.setattr(
        sg,
        "assert_table_cluster_identity",
        lambda table, **kwargs: identity_checks.append((table, kwargs)),
    )

    snapshot = sg._refresh_clustered_sample_aggregates(
        _config(
            steering_metric=metric,
            general_val_enabled=False,
        ),
        paths=paths,
        contract=_contract(),
        stage_source=stage_source,
        candidate_snapshot=candidate_snapshot,
    )

    aggregate_dir = paths.root / aggregate_dir_name
    persisted_macro = real_read_table(aggregate_dir / "macro.parquet")
    persisted_per_candidate = real_read_table(
        aggregate_dir / "per_candidate.parquet"
    )
    assert snapshot.metrics.macro.equals(
        persisted_macro,
        check_metadata=True,
    )
    assert snapshot.metrics.per_candidate.equals(
        persisted_per_candidate,
        check_metadata=True,
    )
    assert identity_checks[0][0] is score_table
    assert identity_checks[0][1]["candidate_snapshot"] is candidate_snapshot
    assert identity_checks[0][1]["source_identity"] is (
        stage_source.source_identity
    )


@pytest.mark.parametrize(
    ("metric", "score_name", "family", "aggregate_name"),
    [
        (
            "bpb_class_margin",
            "scores.parquet",
            "class_margin_scores",
            "aggregate_bpb_adv_scores",
        ),
        (
            "roundtrip",
            "roundtrip_scores.parquet",
            "roundtrip_scores",
            "aggregate_roundtrip_adv_scores",
        ),
    ],
)
def test_clustered_sample_aggregates_are_rebuilt_from_stamped_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metric: str,
    score_name: str,
    family: str,
    aggregate_name: str,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    score_table = object()
    events: list[tuple[Any, ...]] = []
    stage_source, candidate_snapshot, expected_macro, sampling_snapshot = (
        _install_refresh_owner_stubs(monkeypatch, metric=metric)
    )
    monkeypatch.setattr(
        sg.pq,
        "read_table",
        lambda path: events.append(("read", Path(path).name)) or score_table,
    )
    monkeypatch.setattr(
        sg,
        "assert_table_cluster_identity",
        lambda table, **kwargs: events.append(
            (
                "identity",
                table,
                kwargs["artifact_family"],
                kwargs,
            )
        ),
    )
    monkeypatch.setattr(
        sg,
        aggregate_name,
        lambda score_path, output_dir, *, general_val_path, scores_table=None, general_val_table=None: (
            events.append(
                (
                    "aggregate",
                    Path(score_path).name,
                    Path(output_dir).name,
                    general_val_path,
                )
            )
            or expected_macro
        ),
    )
    cfg = _config(
        steering_metric=metric,
        scores_path=None,
        general_val_enabled=False,
    )

    actual_snapshot = sg._refresh_clustered_sample_aggregates(
        cfg,
        paths=paths,
        contract=_contract(),
        stage_source=stage_source,
        candidate_snapshot=candidate_snapshot,
    )

    assert actual_snapshot is sampling_snapshot
    assert events[0] == ("read", score_name)
    assert events[1][:3] == ("identity", score_table, family)
    assert events[1][3]["source_identity"] is stage_source.source_identity
    assert events[1][3]["candidate_snapshot"] is candidate_snapshot
    assert events[2][0] == "aggregate"
    assert events[2][1] == score_name
    assert events[2][3] is None


@pytest.mark.parametrize(
    ("metric", "aggregate_name", "score_family"),
    [
        ("bpb_class_margin", "aggregate_bpb_adv_scores", "class_margin_scores"),
        ("roundtrip", "aggregate_roundtrip_adv_scores", "roundtrip_scores"),
    ],
)
def test_clustered_sample_aggregate_refresh_validates_general_val_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metric: str,
    aggregate_name: str,
    score_family: str,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    score_table = object()
    general_val_table = object()
    events: list[tuple[Any, ...]] = []
    stage_source, candidate_snapshot, expected_macro, sampling_snapshot = (
        _install_refresh_owner_stubs(
            monkeypatch,
            metric=metric,
        )
    )

    def _read_table(path: Path) -> object:
        return (
            general_val_table
            if Path(path) == paths.general_val_bpb_path()
            else score_table
        )

    monkeypatch.setattr(sg.pq, "read_table", _read_table)
    monkeypatch.setattr(
        sg,
        "assert_table_cluster_identity",
        lambda table, **kwargs: events.append(
            (
                "identity",
                table,
                kwargs["artifact_family"],
                kwargs,
            )
        ),
    )
    monkeypatch.setattr(
        sg,
        aggregate_name,
        lambda score_path, output_dir, *, general_val_path, scores_table=None, general_val_table=None: (
            events.append(
                (
                    "aggregate",
                    Path(score_path),
                    Path(output_dir),
                    Path(general_val_path),
                )
            )
            or expected_macro
        ),
    )
    cfg = _config(
        steering_metric=metric,
        scores_path=None,
        general_val_enabled=True,
    )

    actual_snapshot = sg._refresh_clustered_sample_aggregates(
        cfg,
        paths=paths,
        contract=_contract(),
        stage_source=stage_source,
        candidate_snapshot=candidate_snapshot,
    )

    assert actual_snapshot is sampling_snapshot
    assert [event[:3] for event in events[:2]] == [
        ("identity", score_table, score_family),
        ("identity", general_val_table, "general_val_bpb"),
    ]
    assert all(
        event[3]["source_identity"] is stage_source.source_identity
        and event[3]["candidate_snapshot"] is candidate_snapshot
        for event in events[:2]
    )
    assert events[2][0] == "aggregate"
    assert events[2][3] == paths.general_val_bpb_path()


def test_clustered_sample_refresh_rejects_aggregate_implementation_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    score_table = object()
    stage_source, candidate_snapshot, _expected_macro, _snapshot = (
        _install_refresh_owner_stubs(
            monkeypatch,
            metric="bpb_class_margin",
        )
    )
    monkeypatch.setattr(sg.pq, "read_table", lambda _path: score_table)
    monkeypatch.setattr(
        sg,
        "assert_table_cluster_identity",
        lambda table, **_kwargs: (
            None
            if table is score_table
            else pytest.fail("identity checked the wrong score table")
        ),
    )
    monkeypatch.setattr(
        sg,
        "aggregate_bpb_adv_scores",
        lambda *_args, **_kwargs: pa.table({"exact": [False]}),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="differs from exact in-memory aggregation",
    ):
        sg._refresh_clustered_sample_aggregates(
            _config(
                steering_metric="bpb_class_margin",
                general_val_enabled=False,
            ),
            paths=paths,
            contract=_contract(),
            stage_source=stage_source,
            candidate_snapshot=candidate_snapshot,
        )


def test_clustered_sample_aggregate_refresh_rejects_unreadable_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    stage_source, candidate_snapshot, _macro, _snapshot = (
        _install_refresh_owner_stubs(
            monkeypatch,
            metric="bpb_class_margin",
        )
    )
    monkeypatch.setattr(
        sg.pq,
        "read_table",
        lambda path: (_ for _ in ()).throw(OSError("score sentinel")),
    )
    cfg = _config(
        steering_metric="bpb_class_margin",
        scores_path=str(tmp_path / "custom-scores.parquet"),
        general_val_enabled=False,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="cannot read scored artifact.*custom-scores.parquet",
    ):
        sg._refresh_clustered_sample_aggregates(
            cfg,
            paths=paths,
            contract=_contract(),
            stage_source=stage_source,
            candidate_snapshot=candidate_snapshot,
        )


def test_clustered_sample_aggregate_refresh_rejects_unreadable_general_val(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    score_table = object()
    stage_source, candidate_snapshot, expected_macro, _snapshot = (
        _install_refresh_owner_stubs(
            monkeypatch,
            metric="bpb_class_margin",
        )
    )

    def _read(path: Path) -> object:
        if Path(path) == paths.general_val_bpb_path():
            raise OSError("general-val sentinel")
        return score_table

    monkeypatch.setattr(sg.pq, "read_table", _read)
    monkeypatch.setattr(sg, "assert_table_cluster_identity", lambda *args, **kwargs: None)
    cfg = _config(
        steering_metric="bpb_class_margin",
        scores_path=None,
        general_val_enabled=True,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="cannot read general-validation artifact",
    ):
        sg._refresh_clustered_sample_aggregates(
            cfg,
            paths=paths,
            contract=_contract(),
            stage_source=stage_source,
            candidate_snapshot=candidate_snapshot,
        )

    monkeypatch.setattr(sg.pq, "read_table", lambda path: score_table)
    monkeypatch.setattr(
        sg,
        "aggregate_bpb_adv_scores",
        lambda *args, **kwargs: expected_macro,
    )
    sg._refresh_clustered_sample_aggregates(
        cfg,
        paths=paths,
        contract=_contract(),
        stage_source=stage_source,
        candidate_snapshot=candidate_snapshot,
    )


def _install_clustered_core_write_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    doc: dict[str, Any],
    family: sg.GenFamily,
) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    candidates = _Rows([{"candidate": "row"}])
    metrics = SimpleNamespace(
        macro=_Rows([{"macro": "row"}]),
        per_candidate=_Rows([{"metric": "row"}]),
        representative_baseline_metric="baseline_primary_margin",
        primary_metric_field="primary_margin_y",
        primary_metric_output_key="primary_margin_y",
    )
    stage_source = SimpleNamespace(
        validation=SimpleNamespace(
            concept_entries=lambda: [asdict(_concept())],
        ),
        source_identity=object(),
    )
    candidate_snapshot = SimpleNamespace(table=candidates)
    sampling_snapshot = SimpleNamespace(metrics=metrics)
    monkeypatch.setattr(
        sg,
        "require_clustered_stage_source",
        lambda **kwargs: (
            None if kwargs["contract"] is None else stage_source
        ),
    )
    monkeypatch.setattr(
        sg,
        "require_clustered_candidate_snapshot",
        lambda *_args, **kwargs: (
            None
            if kwargs["contract"] is None
            else (
                events.append(
                    (
                        "candidate_identity",
                        (candidate_snapshot, kwargs["contract"]),
                    )
                )
                or candidate_snapshot
            )
        ),
    )
    monkeypatch.setattr(
        sg,
        "_require_clustered_sampling_snapshot",
        lambda *_args, **kwargs: (
            sampling_snapshot if kwargs["clustered"] else None
        ),
    )
    monkeypatch.setattr(sg.pq, "read_table", lambda path: candidates)
    monkeypatch.setattr(sg, "_validate_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sg,
        "assert_table_cluster_identity",
        lambda table, **kwargs: events.append(
            ("candidate_identity", (table, kwargs["contract"]))
        ),
    )
    monkeypatch.setattr(
        sg,
        "_load_sampling_metrics",
        lambda *args, **kwargs: metrics,
    )
    monkeypatch.setattr(
        sg,
        "_read_concepts",
        lambda source_paths: {1: SimpleNamespace(description="concept")},
    )
    monkeypatch.setattr(
        sg,
        "families_with_representative_coefficient",
        lambda *args, **kwargs: [family],
    )
    monkeypatch.setattr(
        sg,
        "_family_candidates",
        lambda *args, **kwargs: [{"candidate": "row"}],
    )
    monkeypatch.setattr(
        sg,
        "_family_strength_axis",
        lambda *args, **kwargs: sg.FamilyStrengthAxis(
            by_prefix={(1, "prefix"): {0: 0.0}},
            strength_unit="counter_direction_magnitude",
        ),
    )
    monkeypatch.setattr(
        sg,
        "select_concepts_and_prefixes",
        lambda *args, **kwargs: [
            sg.GenConcept(
                concept_id=1,
                concept_label="concept",
                prefixes=[sg.GenPrefix(core_prefix_id="prefix", prefix_text="prompt")],
            )
        ],
    )
    monkeypatch.setattr(
        sg,
        "_generate_family_completions",
        lambda *args, **kwargs: (
            {(1, "prefix", 0): ["completion"]},
            {},
        ),
    )
    monkeypatch.setattr(
        sg,
        "build_family_yaml_doc",
        lambda **kwargs: dict(doc),
    )
    monkeypatch.setattr(
        sg,
        "clustered_artifact_identity",
        lambda **kwargs: {
            "feature_contract_fingerprint": "f" * 64,
            "clustered_artifact_family": kwargs["artifact_family"],
        },
    )
    return events


def _run_stubbed_core(
    tmp_path: Path,
    *,
    contract: Any,
) -> tuple[AxBenchBpbAdvPaths, list[Path]]:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    written = sg.run_generate_samples_core(
        paths=paths,
        source_run_dir=tmp_path,
        model=object(),
        tokenizer=_Tokenizer(),
        device=t.device("cpu"),
        target_model="decoder_only",
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=0,
        counter_direction_family="diffmean",
        scatter_baseline_x="baseline_loss",
        steering_metric="bpb_class_margin",
        generation=sg.GenerationConfig(),
        apply_max_act_scale=False,
        cluster_contract=contract,
        stage_source=object(),  # ty: ignore[invalid-argument-type]
        candidate_snapshot=object(),  # ty: ignore[invalid-argument-type]
        sampling_snapshot=object(),  # ty: ignore[invalid-argument-type]
        pre_forward_guard=lambda: None,
    )
    return paths, written


def test_clustered_sample_yaml_is_identity_stamped_and_replay_immutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = sg.GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="additive_direction",
        context_signal_mode="not_applicable",
        representative_coefficient="4.0",
    )
    mutable_doc = {"method_name": family.method_name, "payload": "stable"}
    events = _install_clustered_core_write_stubs(
        monkeypatch,
        doc=mutable_doc,
        family=family,
    )
    contract = _contract()

    paths, first = _run_stubbed_core(tmp_path, contract=contract)
    expected_path = (
        paths.samples_dir()
        / "positive"
        / "diffmean_decoder_only__additive_direction.yaml"
    )
    assert first == [expected_path]
    first_bytes = expected_path.read_bytes()
    assert first_bytes.startswith(
        (
            "feature_contract_fingerprint: "
            + "f" * 64
            + "\nclustered_artifact_family: samples\n"
        ).encode()
    )
    assert events == [("candidate_identity", (events[0][1][0], contract))]

    _paths, second = _run_stubbed_core(tmp_path, contract=contract)
    assert second == [expected_path]
    assert expected_path.read_bytes() == first_bytes

    mutable_doc["payload"] = "mutated"
    with pytest.raises(
        AxBenchBpbAdvError,
        match="content changed on resume.*fresh run_tag",
    ):
        _run_stubbed_core(tmp_path, contract=contract)
    mutable_doc["payload"] = "stable"
    _run_stubbed_core(tmp_path, contract=contract)


def test_clustered_core_never_reopens_candidates_aggregates_or_concepts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = sg.GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="additive_direction",
        context_signal_mode="not_applicable",
        representative_coefficient="4.0",
    )
    _install_clustered_core_write_stubs(
        monkeypatch,
        doc={"payload": "owned"},
        family=family,
    )
    candidates = _Rows([{"candidate": "owned"}])
    metrics = SimpleNamespace(
        macro=_Rows([{"macro": "owned"}]),
        per_candidate=_Rows([{"metric": "owned"}]),
        representative_baseline_metric="baseline_primary_margin",
        primary_metric_field="primary_margin_y",
        primary_metric_output_key="primary_margin_y",
    )
    validation = SimpleNamespace(
        concept_entries=lambda: [asdict(_concept())],
    )
    source_identity = object()
    stage_source = SimpleNamespace(
        validation=validation,
        source_identity=source_identity,
    )
    candidate_snapshot = SimpleNamespace(table=candidates)
    sampling_snapshot = SimpleNamespace(metrics=metrics)
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        sg,
        "require_clustered_stage_source",
        lambda **_kwargs: stage_source,
    )
    monkeypatch.setattr(
        sg,
        "require_clustered_candidate_snapshot",
        lambda *_args, **_kwargs: candidate_snapshot,
    )
    monkeypatch.setattr(
        sg,
        "_require_clustered_sampling_snapshot",
        lambda *_args, **_kwargs: sampling_snapshot,
    )
    monkeypatch.setattr(
        sg.pq,
        "read_table",
        lambda _path: pytest.fail(
            "clustered sample core must not reopen a parquet input"
        ),
    )
    monkeypatch.setattr(
        sg,
        "_load_sampling_metrics",
        lambda *_args, **_kwargs: pytest.fail(
            "clustered sample core must not reopen aggregate paths"
        ),
    )
    monkeypatch.setattr(
        sg,
        "_read_concepts",
        lambda _paths: pytest.fail(
            "clustered sample core must not reopen concepts.json"
        ),
    )
    def _generate(*_args: Any, **kwargs: Any) -> tuple[Any, Any]:
        observed["generation"] = kwargs
        return {(1, "prefix", 0): ["completion"]}, {}

    def _identity(**kwargs: Any) -> dict[str, str]:
        observed["identity"] = kwargs
        return {
            "feature_contract_fingerprint": "f" * 64,
            "clustered_artifact_family": "samples",
        }

    monkeypatch.setattr(sg, "_generate_family_completions", _generate)
    monkeypatch.setattr(sg, "clustered_artifact_identity", _identity)
    contract = _contract()

    written = sg.run_generate_samples_core(
        paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run"),
        source_run_dir=tmp_path,
        model=object(),
        tokenizer=_Tokenizer(),
        device=t.device("cpu"),
        target_model="decoder_only",
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=0,
        counter_direction_family="diffmean",
        scatter_baseline_x="baseline_loss",
        steering_metric="bpb_class_margin",
        generation=sg.GenerationConfig(),
        apply_max_act_scale=False,
        cluster_contract=contract,
        stage_source=object(),  # ty: ignore[invalid-argument-type]
        candidate_snapshot=object(),  # ty: ignore[invalid-argument-type]
        sampling_snapshot=object(),  # ty: ignore[invalid-argument-type]
        pre_forward_guard=lambda: None,
    )

    assert len(written) == 1
    assert observed["generation"]["source_validation"] is validation
    assert observed["identity"]["source_identity"] is source_identity
    assert observed["identity"]["candidate_snapshot"] is candidate_snapshot


def test_unclustered_sample_yaml_bytes_and_path_remain_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = sg.GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="additive_direction",
        context_signal_mode="not_applicable",
        representative_coefficient="4.0",
    )
    doc = {"method_name": family.method_name, "payload": "legacy"}
    events = _install_clustered_core_write_stubs(
        monkeypatch,
        doc=doc,
        family=family,
    )

    paths, written = _run_stubbed_core(tmp_path, contract=None)
    expected_path = (
        paths.samples_dir()
        / "positive"
        / "diffmean_decoder_only__additive_direction.yaml"
    )
    assert written == [expected_path]
    assert expected_path.read_bytes() == (
        b"method_name: diffmean_decoder_only\npayload: legacy\n"
    )
    assert events == [("candidate_identity", (events[0][1][0], None))]


@pytest.mark.parametrize("skip_branch", ["no_coefficient", "empty_candidates"])
def test_clustered_core_skip_branches_still_validate_candidates_before_no_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    skip_branch: str,
) -> None:
    family = sg.GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="additive_direction",
        context_signal_mode="not_applicable",
        representative_coefficient=(
            None if skip_branch == "no_coefficient" else "4.0"
        ),
    )
    events = _install_clustered_core_write_stubs(
        monkeypatch,
        doc={"unreachable": True},
        family=family,
    )
    if skip_branch == "empty_candidates":
        monkeypatch.setattr(sg, "_family_candidates", lambda *args, **kwargs: [])

    paths, written = _run_stubbed_core(tmp_path, contract=_contract())
    assert written == []
    assert not paths.samples_dir().exists()
    assert [event[0] for event in events] == ["candidate_identity"]


def test_clustered_encdec_empty_family_return_still_loads_validated_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = _encdec_family()
    events = _install_clustered_core_write_stubs(
        monkeypatch,
        doc={"unreachable": True},
        family=family,
    )
    monkeypatch.setattr(
        sg,
        "families_with_representative_coefficient",
        lambda *args, **kwargs: (
            events.append(("family_gate", kwargs["allow_encdec"])) or []
        ),
    )
    contract = _contract()
    loaded: list[tuple[Any, Any]] = []
    monkeypatch.setattr(
        sg,
        "load_context_segments_by_prefix_hash",
        lambda paths, *, cluster_contract: (
            loaded.append((paths, cluster_contract)) or {}
        ),
    )
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")

    written = sg.run_generate_samples_core(
        paths=paths,
        source_run_dir=tmp_path,
        model=object(),
        tokenizer=_Tokenizer(),
        device=t.device("cpu"),
        target_model="encdec",
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=0,
        counter_direction_family="diffmean",
        scatter_baseline_x="baseline_loss",
        steering_metric="bpb_class_margin",
            generation=sg.GenerationConfig(),
            apply_max_act_scale=False,
            cluster_contract=contract,
        stage_source=object(),  # ty: ignore[invalid-argument-type]
        candidate_snapshot=object(),  # ty: ignore[invalid-argument-type]
        sampling_snapshot=object(),  # ty: ignore[invalid-argument-type]
        pre_forward_guard=lambda: None,
    )

    assert written == []
    assert loaded == [(paths, contract)]
    assert [event[0] for event in events] == [
        "candidate_identity",
        "family_gate",
    ]
    assert events[1] == ("family_gate", True)
    assert not paths.samples_dir().exists()


def _encdec_family() -> sg.GenFamily:
    return sg.GenFamily(
        mode="positive",
        method_name="feature_diffmean_encdec",
        target_model="encdec",
        adversarial_method="additive_direction",
        context_signal_mode="fixed_context",
        representative_coefficient="1.0",
    )


def _run_one_family(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_rows: list[dict[str, Any]],
    context_segments: Any,
) -> tuple[dict[str, Any], Any]:
    family = _encdec_family()
    representative = sg.RepresentativeAttack(
        prompt_tokens=[9, 9],
        attacked_prompt_text="attacked",
        replacement_text="",
        objective_value=0.0,
        candidate_id="chosen",
        optimization=NoAdversaryControlMetadata(),
    )
    captured: dict[str, Any] = {}
    intervention = Intervention.ours(feature_conditions={3: 1})
    provider = SingleEntryContextProvider([101, 102], ())

    def _build(**kwargs: Any) -> Intervention:
        captured["build"] = kwargs
        return intervention

    def _roundtrip(**kwargs: Any) -> Any:
        captured["roundtrip"] = kwargs
        return SimpleNamespace(
            prompt_tokens=[101, 102],
            context_provider=provider,
        )

    def _generate(**kwargs: Any) -> list[str]:
        captured["generation"] = kwargs
        return ["done"]

    monkeypatch.setattr(sg, "get_method_by_name", lambda name: object())
    monkeypatch.setattr(
        sg,
        "representative_attack_for_datapoint",
        lambda *args, **kwargs: representative,
    )
    monkeypatch.setattr(
        sg,
        "build_steered_intervention",
        _build,
    )
    monkeypatch.setattr(
        sg,
        "_roundtrip_generation_inputs",
        _roundtrip,
    )
    monkeypatch.setattr(
        sg,
        "generate_completions_for_datapoint",
        _generate,
    )
    contract = _contract()
    source_validation: Any = object()
    paths = AxBenchBpbAdvPaths(output_root=Path("/tmp"), run_tag="run")
    outputs = sg._generate_family_completions(
        family,
        concepts=[
            sg.GenConcept(
                concept_id=1,
                concept_label="concept",
                prefixes=[sg.GenPrefix(core_prefix_id="prefix", prefix_text="prompt")],
            )
        ],
        concepts_by_id={1: _concept()},
        family_candidates=candidate_rows,
        strength_axis=sg.FamilyStrengthAxis(
            by_prefix={(1, "prefix"): {0: 0.0}},
            strength_unit="counter_direction_magnitude",
        ),
        model=object(),
        tokenizer=_Tokenizer(),
        device=t.device("cpu"),
        paths=paths,
        source_paths=SourcePaths(Path("source")),
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=0,
        counter_direction_family="diffmean",
        generation=sg.GenerationConfig(n_completions=1),
        apply_max_act_scale=False,
        lora=None,
        context_segments_by_prefix_hash=context_segments,
        cluster_contract=contract,
        source_validation=source_validation,
        pre_forward_guard=lambda: None,
    )
    captured["paths"] = paths
    captured["contract"] = contract
    captured["source_validation"] = source_validation
    return captured, outputs


def test_clustered_encdec_family_uses_assistant_start_roundtrip_inputs_never_attack_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {"candidate_id": "chosen", "sentinel": "scored-row"}
    captured, outputs = _run_one_family(
        monkeypatch,
        candidate_rows=[candidate],
        context_segments={"prefix": ()},
    )

    assert captured["roundtrip"]["candidate_row"] is candidate
    assert captured["roundtrip"].get("cluster_contract") is captured["contract"]
    assert captured["generation"]["prompt_tokens"] == [101, 102]
    assert captured["generation"]["prompt_tokens"] != [9, 9]
    assert captured["generation"]["context_provider"] is not None
    assert captured["build"]["adv_paths"] is captured["paths"]
    assert captured["build"]["core_prefix_id"] == "prefix"
    assert captured["build"]["cluster_contract"] is captured["contract"]
    assert captured["build"]["source_validation"] is captured["source_validation"]
    assert outputs[0][(1, "prefix", 0)] == ["done"]


def test_unclustered_decoder_family_uses_attack_prompt_without_roundtrip_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = sg.GenFamily(
        mode="positive",
        method_name="diffmean",
        target_model="decoder_only",
        adversarial_method="additive_direction",
        context_signal_mode="not_applicable",
        representative_coefficient="1.0",
    )
    representative = sg.RepresentativeAttack(
        prompt_tokens=[9, 9],
        attacked_prompt_text="attacked",
        replacement_text="",
        objective_value=0.0,
        candidate_id="chosen",
        optimization=NoAdversaryControlMetadata(),
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(sg, "get_method_by_name", lambda name: object())
    monkeypatch.setattr(
        sg,
        "representative_attack_for_datapoint",
        lambda *args, **kwargs: representative,
    )

    def _build(**kwargs: Any) -> Intervention:
        captured["build"] = kwargs
        return Intervention.none()

    monkeypatch.setattr(
        sg,
        "build_steered_intervention",
        _build,
    )
    monkeypatch.setattr(
        sg,
        "_roundtrip_generation_inputs",
        lambda **kwargs: pytest.fail("decoder-only generation must not use roundtrip context"),
    )

    def _generate(**kwargs: Any) -> list[str]:
        captured["generation"] = kwargs
        return ["done"]

    monkeypatch.setattr(sg, "generate_completions_for_datapoint", _generate)
    paths = AxBenchBpbAdvPaths(output_root=Path("/tmp"), run_tag="run")
    outputs = sg._generate_family_completions(
        family,
        concepts=[
            sg.GenConcept(
                concept_id=1,
                concept_label="concept",
                prefixes=[sg.GenPrefix(core_prefix_id="prefix", prefix_text="prompt")],
            )
        ],
        concepts_by_id={1: _concept()},
        family_candidates=[{"candidate_id": "chosen"}],
        strength_axis=sg.FamilyStrengthAxis(
            by_prefix={(1, "prefix"): {0: 0.0}},
            strength_unit="counter_direction_magnitude",
        ),
        model=object(),
        tokenizer=_Tokenizer(),
        device=t.device("cpu"),
        paths=paths,
        source_paths=SourcePaths(Path("source")),
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=0,
        counter_direction_family="diffmean",
            generation=sg.GenerationConfig(n_completions=1),
            apply_max_act_scale=False,
            lora=None,
    )

    assert captured["generation"]["prompt_tokens"] == [9, 9]
    assert captured["generation"]["context_provider"] is None
    assert captured["generation"]["cluster_contract"] is None
    assert captured["build"]["adv_paths"] is paths
    assert captured["build"]["core_prefix_id"] == "prefix"
    assert captured["build"]["cluster_contract"] is None
    assert outputs[0][(1, "prefix", 0)] == ["done"]


def test_clustered_encdec_family_requires_loaded_context_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="requires loaded context segments"):
        _run_one_family(
            monkeypatch,
            candidate_rows=[{"candidate_id": "chosen"}],
            context_segments=None,
        )

    captured, outputs = _run_one_family(
        monkeypatch,
        candidate_rows=[{"candidate_id": "chosen"}],
        context_segments={"prefix": ()},
    )
    assert captured["generation"]["context_provider"] is not None
    assert outputs[0]


@pytest.mark.parametrize("matching_rows", [0, 2])
def test_clustered_encdec_family_requires_unique_scored_candidate_row(
    monkeypatch: pytest.MonkeyPatch,
    matching_rows: int,
) -> None:
    rows = [{"candidate_id": "chosen"} for _ in range(matching_rows)]
    if matching_rows == 0:
        rows = [{"candidate_id": "other"}]
    with pytest.raises(
        AxBenchBpbAdvError,
        match="could not resolve the unique candidate row",
    ):
        _run_one_family(
            monkeypatch,
            candidate_rows=rows,
            context_segments={"prefix": ()},
        )

    captured, outputs = _run_one_family(
        monkeypatch,
        candidate_rows=[{"candidate_id": "chosen"}],
        context_segments={"prefix": ()},
    )
    assert captured["roundtrip"]["candidate_row"]["candidate_id"] == "chosen"
    assert outputs[0]


def _install_run_source_preflight(
    monkeypatch: pytest.MonkeyPatch,
    cfg: AxBenchBpbAdvConfig,
) -> None:
    assert cfg.run_tag is not None
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag=cfg.run_tag)
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    payload = sg.BpbAdvSourceContract(
        seed_fingerprint="f" * 64,
        source_run_dir=cfg.source_run_dir,
    ).to_json_dict()
    paths.source_contract().write_text(json.dumps(payload), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist([], schema=sg.CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    validation = SimpleNamespace(
        contract=SimpleNamespace(to_json_dict=lambda: payload),
        require_directions=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sg,
        "validate_stage_0_3_source_run",
        lambda *args, **kwargs: validation,
    )


def test_clustered_sample_stale_aggregate_guard_runs_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _config(
        run_tag="run",
        output_root=str(tmp_path),
    )
    _install_run_source_preflight(monkeypatch, cfg)
    stage_source = SimpleNamespace(source_identity=object())
    candidate_snapshot = object()
    monkeypatch.setattr(
        sg,
        "_resolve_sample_cluster_stage",
        lambda *args, **kwargs: (_contract(), stage_source),
    )
    monkeypatch.setattr(
        sg,
        "load_clustered_candidate_snapshot",
        lambda *args, **kwargs: candidate_snapshot,
    )

    def _reject_stale(*_args: Any, **_kwargs: Any) -> None:
        raise AxBenchBpbAdvError("sentinel stale aggregate input")

    monkeypatch.setattr(sg, "_refresh_clustered_sample_aggregates", _reject_stale)
    monkeypatch.setattr(
        sg,
        "RuntimeState",
        lambda *args, **kwargs: pytest.fail("model load must not start"),
    )

    with pytest.raises(AxBenchBpbAdvError, match="sentinel stale aggregate input"):
        sg.run_generate_samples(
            cfg,
            generation=sg.GenerationConfig(),
        )


def _clustered_run_cfg(
    tmp_path: Path,
    *,
    targets: list[str],
) -> AxBenchBpbAdvConfig:
    return _config(
        run_tag="run",
        output_root=str(tmp_path),
        target_models=targets,
        source_run_dir="source",
    )


def test_clustered_run_generates_decoder_and_encdec_with_exact_contract_plumbing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    stage_source = SimpleNamespace(source_identity=object())
    candidate_snapshot = object()
    sampling_snapshot = object()
    candidate_loads: list[dict[str, Any]] = []
    refresh_calls: list[dict[str, Any]] = []
    calls: list[tuple[str, Any]] = []

    class _Bundle:
        def __init__(self, target: str) -> None:
            self.target = target
            self.model = f"{target}-model"
            self.tokenizer = f"{target}-tokenizer"

        def ensure_ready_for_forward(self) -> None:
            calls.append(("guard", self.target))

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("runtime", kwargs))

        def decoder_only(self) -> _Bundle:
            calls.append(("bundle", "decoder_only"))
            return _Bundle("decoder_only")

        def encdec(self) -> _Bundle:
            calls.append(("bundle", "encdec"))
            return _Bundle("encdec")

        def inactive_sampler(self) -> str:
            calls.append(("sampler", "encdec"))
            return "inactive"

        def target_device(self) -> str:
            return "cpu"

        def offload_models(self) -> None:
            calls.append(("offload", None))

    monkeypatch.setattr(
        sg,
        "_resolve_sample_cluster_stage",
        lambda *args, **kwargs: (contract, stage_source),
    )
    monkeypatch.setattr(
        sg,
        "load_clustered_candidate_snapshot",
        lambda *args, **kwargs: (
            candidate_loads.append(kwargs) or candidate_snapshot
        ),
    )
    monkeypatch.setattr(
        sg,
        "_refresh_clustered_sample_aggregates",
        lambda *args, **kwargs: (
            refresh_calls.append(kwargs)
            or calls.append(("refresh", kwargs["contract"]))
            or sampling_snapshot
        ),
    )
    monkeypatch.setattr(sg, "_bpb_config_from_adv_config", lambda cfg: "bpb-cfg")
    monkeypatch.setattr(sg, "RuntimeState", _Runtime)
    monkeypatch.setattr(
        sg,
        "detect_target_model_metadata",
        lambda model, cfg, **kwargs: calls.append(
            ("detect", (model, cfg, kwargs))
        )
        or f"{model}-meta",
    )
    monkeypatch.setattr(
        sg.logger,
        "warning",
        lambda *args, **kwargs: calls.append(("unexpected_warning", args)),
    )

    def _core(**kwargs: Any) -> list[Path]:
        calls.append(("core", kwargs))
        guard = kwargs["pre_forward_guard"]
        assert callable(guard), "clustered target must receive a pre-forward guard"
        guard()
        return [tmp_path / f"{kwargs['target_model']}.yaml"]

    monkeypatch.setattr(sg, "run_generate_samples_core", _core)
    cfg = _clustered_run_cfg(
        tmp_path,
        targets=["decoder_only", "encdec"],
    )
    _install_run_source_preflight(monkeypatch, cfg)

    assert sg.run_generate_samples(
        cfg,
        generation=sg.GenerationConfig(),
        counter_direction_family="diffmean",
    ) == [
        tmp_path / "decoder_only.yaml",
        tmp_path / "encdec.yaml",
    ]
    runtime_kwargs = next(value for name, value in calls if name == "runtime")
    assert runtime_kwargs["cluster_contract"] is contract
    assert ("refresh", contract) in calls
    assert len(candidate_loads) == 1
    assert candidate_loads[0]["source_identity"] is stage_source.source_identity
    assert len(refresh_calls) == 1
    assert refresh_calls[0]["stage_source"] is stage_source
    assert refresh_calls[0]["candidate_snapshot"] is candidate_snapshot
    assert not [value for name, value in calls if name == "unexpected_warning"]
    core_calls = [value for name, value in calls if name == "core"]
    assert [call["target_model"] for call in core_calls] == [
        "decoder_only",
        "encdec",
    ]
    assert core_calls[0]["inactive_sampler"] is None
    assert core_calls[1]["inactive_sampler"] == "inactive"
    assert core_calls[0]["lora"] is None
    assert core_calls[1]["lora"] is None
    assert all(call["cluster_contract"] is contract for call in core_calls)
    assert all(call["apply_max_act_scale"] is True for call in core_calls)
    assert all(call["stage_source"] is stage_source for call in core_calls)
    assert all(
        call["candidate_snapshot"] is candidate_snapshot
        for call in core_calls
    )
    assert all(
        call["sampling_snapshot"] is sampling_snapshot
        for call in core_calls
    )
    detects = [value for name, value in calls if name == "detect"]
    assert detects[0][2] == {}
    assert detects[1][2] == {"cluster_contract": contract}
    assert ("guard", "decoder_only") in calls
    assert ("guard", "encdec") in calls
    assert calls[-1] == ("offload", None)


def test_clustered_run_rejects_unknown_target_and_still_offloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    stage_source = SimpleNamespace(source_identity=object())
    candidate_snapshot = object()
    events: list[str] = []

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def offload_models(self) -> None:
            events.append("offload")

    monkeypatch.setattr(
        sg,
        "_resolve_sample_cluster_stage",
        lambda *args, **kwargs: (contract, stage_source),
    )
    monkeypatch.setattr(
        sg,
        "load_clustered_candidate_snapshot",
        lambda *args, **kwargs: candidate_snapshot,
    )
    monkeypatch.setattr(
        sg,
        "_refresh_clustered_sample_aggregates",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(sg, "_bpb_config_from_adv_config", lambda cfg: "bpb-cfg")
    monkeypatch.setattr(sg, "RuntimeState", _Runtime)
    cfg = _clustered_run_cfg(tmp_path, targets=["decoder_only"])
    _install_run_source_preflight(monkeypatch, cfg)
    # Bypass the typed construction boundary deliberately to retain the
    # downstream defensive/offload regression.
    cfg.target_models = cast(Any, ["unknown"])

    with pytest.raises(
        AxBenchBpbAdvError,
        match="unsupported clustered sample target 'unknown'",
    ):
        sg.run_generate_samples(
            cfg,
            generation=sg.GenerationConfig(),
            counter_direction_family="diffmean",
        )
    assert events == ["offload"]


def test_encoder_packing_uses_runtime_check_not_cast() -> None:
    """The encoder-packing branch must name its own failure.

    Review lens B reached this branch by deleting the entry guard and supplying a
    real ``context_provider``: the previous ``cast(...)`` produced an
    ``AttributeError`` on ``None.consolidation`` that only became a clean error
    because a broad ``except Exception`` happened to wrap it. That is safety by
    accident, and a cast is a promise to the type checker rather than a check.

    The branch is unreachable through the public function today (the entry guard
    dominates it), so this is a STRUCTURAL pin rather than a behavioural one --
    stated plainly rather than dressed up as an execution test.
    """

    import steering.eval.axbench_bpb_adv.sample_generation as sg

    source = inspect.getsource(sg.generate_completions_for_datapoint)
    assert "cast(" not in source, (
        "encoder packing must use a runtime check, not cast(): the cast's safety "
        "depended on a broad except Exception converting an AttributeError"
    )
    assert "without a validated clustered" in source, (
        "the packing branch must raise its own named AxBenchBpbAdvError"
    )


def test_unclustered_sample_write_uses_atomic_overwrite_not_install_once() -> None:
    """Pin the UNCLUSTERED write branch independently (lens B mutation 6b).

    Swapping ONLY this branch to ``install_bytes_once`` survived the entire unit
    AND functional golden-byte suites, because the two branches were only ever
    pinned together as one if/else. They differ semantically:
    ``install_bytes_once`` RAISES on content drift while ``atomic_write_text``
    silently overwrites, so an unclustered rerun under the same ``run_tag``
    would begin failing.
    """

    import steering.eval.axbench_bpb_adv.sample_generation as sg

    source = inspect.getsource(sg.run_generate_samples_core)
    marker = "if cluster_contract is None:\n            atomic_write_text("
    assert marker in source, (
        "the unclustered write branch must remain atomic_write_text (silent "
        "overwrite); install_bytes_once there would make legacy reruns raise"
    )
    assert "install_bytes_once(" in source, "the clustered branch must stay immutable-once"


@pytest.mark.parametrize(
    ("metric", "aggregate_name"),
    [
        ("bpb_class_margin", "aggregate_bpb_adv_scores"),
        ("roundtrip", "aggregate_roundtrip_adv_scores"),
    ],
)
def test_refresh_hands_the_validated_table_to_aggregation_not_the_path(
    metric: str,
    aggregate_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The aggregate step must consume the table that was validated.

    This is the flow-level TOCTOU bind. Previously the refresh validated ONE read
    of the score artifact and then handed aggregation only the PATH, which
    re-read it -- so a table swapped between the two reads was aggregated even
    though validation had passed, and the aggregate output carries no score
    digest to catch it downstream.

    Non-vacuous by construction: the spy asserts the object identity of the
    supplied table, so deleting the ``scores_table=`` pass-through in
    ``_refresh_clustered_sample_aggregates`` fails here rather than silently
    reverting to the two-read behaviour.
    """

    import steering.eval.axbench_bpb_adv.sample_generation as sg

    validated = object()
    supplied: dict[str, object] = {}
    stage_source, candidate_snapshot, expected_macro, _snapshot = (
        _install_refresh_owner_stubs(monkeypatch, metric=metric)
    )

    monkeypatch.setattr(sg.pq, "read_table", lambda path: validated)
    monkeypatch.setattr(sg, "assert_table_cluster_identity", lambda table, **kw: None)

    def _spy(score_path, output_dir, *, general_val_path, scores_table=None, general_val_table=None):
        supplied["scores_table"] = scores_table
        return expected_macro

    monkeypatch.setattr(sg, aggregate_name, _spy)

    cfg = _config(
        steering_metric=metric,
        scores_path=None,
        general_val_enabled=False,
    )
    sg._refresh_clustered_sample_aggregates(
        cfg,
        paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run"),
        contract=_contract(),
        stage_source=stage_source,
        candidate_snapshot=candidate_snapshot,
    )

    assert supplied["scores_table"] is validated, (
        "aggregation must receive the already-validated table object; passing "
        "only the path reopens the validate-then-reread TOCTOU window"
    )
